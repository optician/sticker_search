//! Composition root for the embedder. Reads captions from meta.sqlite, embeds
//! each with a local model (Ollama) and stores the vectors in Qdrant, one
//! collection per `(caption_model, prompt_version, embed_model)` set.

use clap::{Args, Parser};
use std::path::PathBuf;
use std::time::Duration;
use sticker_core::error::{EmbedError, EmbeddingGatewayError, RepoError, VectorStoreError};
use sticker_core::{
    EmbedCaptions, EmbedEvent, EmbedProgress, EmbedRun, EmbedSummary, collection_name,
};
use sticker_infra::{OllamaEmbeddingGateway, QdrantVectorStore, SqliteRepository};
use thiserror::Error;

/// The caption set to embed by default — matches the captioner's defaults.
const DEFAULT_CAPTION_MODEL: &str = "qwen3-vl:8b";
const DEFAULT_PROMPT_VERSION: &str = "v1";
/// The embedding model. `bge-m3` is multilingual (Russian incl.) and emits 1024
/// dims. Bump `--embed-model`/`--dim` together when swapping models.
const DEFAULT_EMBED_MODEL: &str = "bge-m3";
const DEFAULT_DIM: usize = 1024;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Embed stored captions into Qdrant for vector search.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Metadata SQLite database (written by the scraper / captioner).
    #[arg(long, default_value = "stickers/meta.sqlite")]
    db: PathBuf,

    #[command(flatten)]
    run: RunArgs,
}

#[derive(Args, Debug)]
struct RunArgs {
    /// Caption model whose captions to embed (selects the source set).
    #[arg(long, default_value = DEFAULT_CAPTION_MODEL)]
    caption_model: String,
    /// Prompt version of the captions to embed.
    #[arg(long, default_value = DEFAULT_PROMPT_VERSION)]
    prompt_version: String,
    /// Embedding model (Ollama tag).
    #[arg(long, default_value = DEFAULT_EMBED_MODEL)]
    embed_model: String,
    /// Vector dimensionality the model emits (must match `--embed-model`).
    #[arg(long, default_value_t = DEFAULT_DIM)]
    dim: usize,
    /// Re-embed stickers whose vector already exists in the collection.
    #[arg(long)]
    force: bool,
    /// Stop after N captions — handy for cheap test runs.
    #[arg(long)]
    limit: Option<usize>,
    /// Ollama base URL. Falls back to $OLLAMA_HOST, then localhost.
    #[arg(long)]
    ollama_url: Option<String>,
    /// Qdrant base URL.
    #[arg(long, default_value = "http://localhost:6333")]
    qdrant_url: String,
}

#[derive(Debug, Error)]
enum AppError {
    #[error("database error: {0}")]
    Repo(#[from] RepoError),
    #[error("building the embedding gateway: {0}")]
    Gateway(#[from] EmbeddingGatewayError),
    #[error("connecting to the vector store: {0}")]
    Store(#[from] VectorStoreError),
    #[error(transparent)]
    Run(#[from] EmbedError),
    #[error("initializing tracing: {0}")]
    Tracing(String),
}

async fn run() -> Result<(), AppError> {
    let cli = Cli::parse();
    cmd_run(&cli.db, cli.run).await
}

async fn cmd_run(db: &std::path::Path, args: RunArgs) -> Result<(), AppError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .try_init()
        .map_err(|e| AppError::Tracing(e.to_string()))?;

    let url = args
        .ollama_url
        .clone()
        .or_else(|| std::env::var("OLLAMA_HOST").ok())
        .unwrap_or_else(|| "http://localhost:11434".to_string());

    let gateway = OllamaEmbeddingGateway::new(&url, &args.embed_model, args.dim, REQUEST_TIMEOUT)?;
    let repo = SqliteRepository::open(db)?;
    let store = QdrantVectorStore::new(&args.qdrant_url, REQUEST_TIMEOUT)?;

    let app = EmbedCaptions::new(gateway, repo, store).on_progress(print_progress);
    let collection = collection_name(&args.caption_model, &args.prompt_version, &args.embed_model);

    tracing::info!(
        embed_model = %args.embed_model,
        caption_model = %args.caption_model,
        prompt = %args.prompt_version,
        collection = %collection,
        ollama = %url,
        qdrant = %args.qdrant_url,
        "starting embedder",
    );

    let cfg = EmbedRun {
        caption_model: &args.caption_model,
        prompt_version: &args.prompt_version,
        force: args.force,
        limit: args.limit,
    };
    let EmbedSummary {
        embedded,
        skipped,
        failed,
    } = app.run(cfg).await?;

    println!(
        "done: embedded {embedded}, skipped {skipped}, failed {failed} \
         → collection {collection}",
    );
    Ok(())
}

/// Live one-line progress: shows the in-flight sticker while the model works,
/// then overwrites it with the outcome.
fn print_progress(p: EmbedProgress) {
    use std::io::Write;
    match p.event {
        EmbedEvent::Start => {
            print!("\r[{:>4}/{}] {} …", p.index, p.total, p.sticker_id);
            let _ = std::io::stdout().flush();
        }
        EmbedEvent::Embedded => {
            println!("\r[{:>4}/{}] {} ✓        ", p.index, p.total, p.sticker_id)
        }
        EmbedEvent::Skipped => {
            println!("\r[{:>4}/{}] {} – skipped ", p.index, p.total, p.sticker_id)
        }
        EmbedEvent::Failed => {
            println!("\r[{:>4}/{}] {} ✗ failed ", p.index, p.total, p.sticker_id)
        }
    }
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
