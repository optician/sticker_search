//! Composition root: parse inputs, build adapters, run the `ScrapePacks` use-case.

use clap::Parser;
use sticker_core::ScrapePacks;
use sticker_core::error::RepoError;
use sticker_infra::{BotApiGateway, FsImageStore, SqliteRepository};
use std::collections::HashSet;
use std::path::PathBuf;
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

/// Extract the bare pack name from a share link or `tg://` URL, accepting:
/// `crazy_klutzy`, `https://t.me/addstickers/crazy_klutzy`,
/// `t.me/addstickers/crazy_klutzy`, `tg://addstickers?set=crazy_klutzy`.
/// Anything that isn't a recognized link is returned trimmed as-is.
fn normalize_pack_name(raw: &str) -> String {
    let s = raw.trim();
    if s.contains("addstickers") {
        if let Some(rest) = s.rsplit_once("addstickers/").map(|(_, r)| r) {
            // https://t.me/addstickers/<name>[/?#...]
            return rest.split(['/', '?', '#']).next().unwrap_or(rest).to_string();
        }
        if let Some(rest) = s.split_once("set=").map(|(_, r)| r) {
            // tg://addstickers?set=<name>[&#...]
            return rest.split(['&', '#']).next().unwrap_or(rest).to_string();
        }
    }
    s.to_string()
}

/// Pack names from the file (if it exists) followed by CLI args. Each entry may
/// be a bare id or a share link. Normalized, blanks dropped, de-duplicated with
/// first-seen order preserved.
fn collect_pack_names(cli: &Cli) -> Result<Vec<String>, AppError> {
    let mut names = Vec::new();
    if cli.packs_file.exists() {
        let text = std::fs::read_to_string(&cli.packs_file).map_err(|source| {
            AppError::ReadPacks { path: cli.packs_file.display().to_string(), source }
        })?;
        for line in text.lines() {
            let line = line.trim();
            if !line.is_empty() && !line.starts_with('#') {
                names.push(normalize_pack_name(line));
            }
        }
    }
    names.extend(cli.pack_names.iter().map(|n| normalize_pack_name(n)));

    names.retain(|n| !n.is_empty());
    let mut seen = HashSet::new();
    names.retain(|n| seen.insert(n.clone()));
    Ok(names)
}

async fn run() -> Result<(), AppError> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .try_init()
        .map_err(|e| AppError::Tracing(e.to_string()))?;

    let token = std::env::var("TELEGRAM_BOT_TOKEN").map_err(|_| AppError::MissingToken)?;

    let names = collect_pack_names(&cli)?;
    if names.is_empty() {
        return Err(AppError::NoPacks(cli.packs_file.display().to_string()));
    }

    std::fs::create_dir_all(&cli.out)
        .map_err(|source| AppError::CreateOut { path: cli.out.display().to_string(), source })?;

    let gateway = BotApiGateway::new(token);
    let repo = SqliteRepository::open(cli.out.join("meta.sqlite"))?;
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
    use super::normalize_pack_name;
    use rstest::rstest;

    #[rstest]
    #[case("crazy_klutzy", "crazy_klutzy")]
    #[case("  crazy_klutzy  ", "crazy_klutzy")]
    #[case("https://t.me/addstickers/crazy_klutzy", "crazy_klutzy")]
    #[case("http://t.me/addstickers/crazy_klutzy", "crazy_klutzy")]
    #[case("t.me/addstickers/crazy_klutzy", "crazy_klutzy")]
    #[case("https://t.me/addstickers/crazy_klutzy/", "crazy_klutzy")]
    #[case("https://t.me/addstickers/crazy_klutzy?foo=bar", "crazy_klutzy")]
    #[case("tg://addstickers?set=crazy_klutzy", "crazy_klutzy")]
    #[case("tg://addstickers?set=crazy_klutzy&mode=x", "crazy_klutzy")]
    fn normalizes_links_and_ids(#[case] input: &str, #[case] expected: &str) {
        assert_eq!(normalize_pack_name(input), expected);
    }
}
