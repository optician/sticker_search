//! Composition root: parse inputs, build adapters, run the `ScrapePacks` use-case.

use clap::Parser;
use std::collections::HashSet;
use std::path::PathBuf;
use sticker_core::error::RepoError;
use sticker_core::ports::PackRequests;
use sticker_core::{ScrapePacks, normalize_pack_name};
use sticker_infra::{BotApiGateway, FsImageStore, SqliteRepository};
use thiserror::Error;

/// Download Telegram sticker packs (thumbnail image + metadata) into a local
/// directory and SQLite database.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Pack names (the `<name>` in t.me/addstickers/<name>). Merged with --packs-file.
    pack_names: Vec<String>,

    /// File with one pack name per line (`#` comments and blanks ignored).
    #[arg(long, default_value = "packs.txt")]
    packs_file: PathBuf,

    /// Also scrape every pack the bot has queued (the `pack_requests` table).
    #[arg(long)]
    from_queue: bool,

    /// Output directory for images and meta.sqlite.
    #[arg(long, default_value = "stickers")]
    out: PathBuf,
}

#[derive(Debug, Error)]
enum AppError {
    #[error("TELEGRAM_BOT_TOKEN must be set (get one from @BotFather)")]
    MissingToken,
    #[error("no pack names given; pass them as arguments or list them in {0}")]
    NoPacks(String),
    #[error("reading {path}: {source}")]
    ReadPacks {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("creating {path}: {source}")]
    CreateOut {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("opening meta.sqlite: {0}")]
    OpenDb(#[from] RepoError),
    #[error("initializing tracing: {0}")]
    Tracing(String),
}

/// Pack names from the file (if it exists), CLI args, and `queue` (the bot's
/// requested packs when `--from-queue` is set). Each entry may be a bare id or a
/// share link. Normalized, blanks dropped, de-duplicated with first-seen order
/// preserved.
fn collect_pack_names(cli: &Cli, queue: &[String]) -> Result<Vec<String>, AppError> {
    let mut names = Vec::new();
    if cli.packs_file.exists() {
        let text =
            std::fs::read_to_string(&cli.packs_file).map_err(|source| AppError::ReadPacks {
                path: cli.packs_file.display().to_string(),
                source,
            })?;
        for line in text.lines() {
            let line = line.trim();
            if !line.is_empty() && !line.starts_with('#') {
                names.push(normalize_pack_name(line));
            }
        }
    }
    names.extend(cli.pack_names.iter().map(|n| normalize_pack_name(n)));
    names.extend(queue.iter().map(|n| normalize_pack_name(n)));

    names.retain(|n| !n.is_empty());
    let mut seen = HashSet::new();
    names.retain(|n| seen.insert(n.clone()));
    Ok(names)
}

async fn run() -> Result<(), AppError> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .try_init()
        .map_err(|e| AppError::Tracing(e.to_string()))?;

    let token = std::env::var("TELEGRAM_BOT_TOKEN").map_err(|_| AppError::MissingToken)?;

    std::fs::create_dir_all(&cli.out).map_err(|source| AppError::CreateOut {
        path: cli.out.display().to_string(),
        source,
    })?;

    // Open the store up front so we can drain the bot's queue before scraping;
    // the same connection is then moved into the use-case.
    let repo = SqliteRepository::open(cli.out.join("meta.sqlite"))?;
    let queue: Vec<String> = if cli.from_queue {
        repo.list_requests()?.into_iter().map(|r| r.name).collect()
    } else {
        Vec::new()
    };

    let names = collect_pack_names(&cli, &queue)?;
    if names.is_empty() {
        return Err(AppError::NoPacks(cli.packs_file.display().to_string()));
    }

    let gateway = BotApiGateway::new(token);
    let images = FsImageStore::new(&cli.out);

    let app = ScrapePacks::new(gateway, repo, images);

    tracing::info!(packs = names.len(), out = %cli.out.display(), "starting scrape");
    let summary = app.run(&names).await;

    println!(
        "done: packs {} ok / {} failed | stickers {} downloaded, {} skipped, {} failed",
        summary.packs_ok,
        summary.packs_failed,
        summary.downloaded,
        summary.skipped_existing,
        summary.failed,
    );
    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{Cli, collect_pack_names};
    use std::path::PathBuf;

    /// A `Cli` with no packs-file on disk (the file branch is skipped) and the
    /// given CLI args.
    fn cli(pack_names: &[&str]) -> Cli {
        Cli {
            pack_names: pack_names.iter().map(|s| s.to_string()).collect(),
            packs_file: PathBuf::from("/nonexistent/packs.txt"),
            from_queue: false,
            out: PathBuf::from("stickers"),
        }
    }

    #[test]
    fn merges_cli_and_queue_normalized_and_deduped() {
        let queue = [
            "https://t.me/addstickers/from_queue".to_string(),
            "dup".to_string(),
        ];
        let names = collect_pack_names(&cli(&["dup", "  cli_pack  "]), &queue).unwrap();

        // CLI first (first-seen order), then queue; links normalized; "dup" once.
        assert_eq!(names, vec!["dup", "cli_pack", "from_queue"]);
    }
}
