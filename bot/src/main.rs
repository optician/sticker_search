//! Composition root for the live Telegram bot. Two surfaces:
//! - **inline search**: `@bot <query>` in any chat → ranked stickers as you type,
//!   via the `SearchStickers` query path (same collection the embedder wrote).
//! - **`/add <pack>`** (or sending a sticker): queues the pack for the offline
//!   pipeline and replies with its derived `scrape → caption → embed` status.
//!
//! The bot must run under the **same token that scraped the packs** — inline
//! cached-sticker results reference `file_id`, which is per-bot.

mod logic;

use clap::Parser;
use logic::{add_target_from_text, format_status, inline_entries};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use sticker_core::error::{
    EmbeddingGatewayError, GatewayError, PackStatusError, RepoError, SearchError, VectorStoreError,
};
use sticker_core::ports::{PackRequests, TelegramGateway};
use sticker_core::{
    Normalization, PackStatus, SearchHit, SearchQuery, SearchStickers, normalize_pack_name,
};
use sticker_infra::{BotApiGateway, OllamaEmbeddingGateway, QdrantVectorStore, SqliteRepository};
use teloxide::prelude::*;
use teloxide::types::{FileId, InlineQueryResult, InlineQueryResultCachedSticker};
use thiserror::Error;
use time::OffsetDateTime;

/// Defaults mirror the captioner/embedder so the bot resolves the same
/// collection the vectors were written to.
const DEFAULT_CAPTION_MODEL: &str = "qwen3-vl:8b";
const DEFAULT_PROMPT_VERSION: &str = "v1";
const DEFAULT_EMBED_MODEL: &str = "bge-m3";
const DEFAULT_DIM: usize = 1024;
/// Cosine-score floor for inline results. Below this, hits are noise (the wrong
/// sticker hurts more than no result), so the bot drops them. Override with
/// `--min-score 0` to disable.
const DEFAULT_MIN_SCORE: &str = "0.44";
/// Telegram caps an inline answer at 50 results.
const MAX_INLINE: usize = 50;
/// Embedding a short query is quick; no need for the long caption-load timeout.
const NET_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Metadata SQLite database (written by the scraper).
    #[arg(long, default_value = "stickers/meta.sqlite")]
    db: PathBuf,
    /// Caption model whose set (collection) to search.
    #[arg(long, default_value = DEFAULT_CAPTION_MODEL)]
    caption_model: String,
    /// Prompt version of that set.
    #[arg(long = "prompt-version", default_value = DEFAULT_PROMPT_VERSION)]
    prompt_version: String,
    /// Embedding model — must match what the embedder used.
    #[arg(long, default_value = DEFAULT_EMBED_MODEL)]
    embed_model: String,
    /// Vector dimensionality the embedding model emits.
    #[arg(long, default_value_t = DEFAULT_DIM)]
    dim: usize,
    /// Max inline results per query (clamped to 50).
    #[arg(long, default_value_t = 30)]
    limit: usize,
    /// Drop hits scoring below this cosine score. Defaults to 0.44; pass 0 to
    /// keep every hit.
    #[arg(long, default_value = DEFAULT_MIN_SCORE)]
    min_score: Option<f32>,
    /// Skip query-text normalization (NFKC + lowercase + whitespace collapse)
    /// before embedding — must match the setting the embedder ran with.
    #[arg(long)]
    no_normalize: bool,
    /// Ollama base URL. Falls back to $OLLAMA_HOST, then localhost.
    #[arg(long)]
    ollama_url: Option<String>,
    /// Qdrant base URL.
    #[arg(long, default_value = "http://localhost:6333")]
    qdrant_url: String,
}

/// Everything a handler needs, shared across updates behind an `Arc`. Adapters
/// are cheap to (re)build per update from this, which sidesteps the `Sync`
/// requirements of sharing a live SQLite connection — fine for a personal bot.
struct BotConfig {
    db: PathBuf,
    token: String,
    bot_username: String,
    caption_model: String,
    prompt_version: String,
    embed_model: String,
    dim: usize,
    limit: usize,
    min_score: Option<f32>,
    normalization: Normalization,
    ollama_url: String,
    qdrant_url: String,
}

#[derive(Debug, Error)]
enum AppError {
    #[error("TELEGRAM_BOT_TOKEN must be set (the same bot that scraped the packs)")]
    MissingToken,
    #[error("database error: {0}")]
    Repo(#[from] RepoError),
    #[error("embedding gateway: {0}")]
    EmbedGateway(#[from] EmbeddingGatewayError),
    #[error("vector store: {0}")]
    Store(#[from] VectorStoreError),
    #[error(transparent)]
    Search(#[from] SearchError),
    #[error(transparent)]
    Status(#[from] PackStatusError),
}

type HandlerResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

fn open_repo(cfg: &BotConfig) -> Result<SqliteRepository, RepoError> {
    SqliteRepository::open(&cfg.db)
}

fn embed_gateway(cfg: &BotConfig) -> Result<OllamaEmbeddingGateway, EmbeddingGatewayError> {
    OllamaEmbeddingGateway::new(&cfg.ollama_url, &cfg.embed_model, cfg.dim, NET_TIMEOUT)
}

fn vector_store(cfg: &BotConfig) -> Result<QdrantVectorStore, VectorStoreError> {
    QdrantVectorStore::new(&cfg.qdrant_url, NET_TIMEOUT)
}

/// Run one query through the shared search use-case, building the adapters fresh.
async fn search(cfg: &BotConfig, text: &str) -> Result<Vec<SearchHit>, AppError> {
    let app = SearchStickers::new(
        embed_gateway(cfg)?,
        vector_store(cfg)?,
        open_repo(cfg)?,
        open_repo(cfg)?,
    )
    .with_normalization(cfg.normalization);
    let q = SearchQuery {
        text,
        caption_model: &cfg.caption_model,
        prompt_version: &cfg.prompt_version,
        limit: cfg.limit,
        min_score: cfg.min_score,
    };
    Ok(app.search(q).await?)
}

async fn on_inline(bot: Bot, q: InlineQuery, cfg: Arc<BotConfig>) -> HandlerResult {
    let query = q.query.trim();
    if query.is_empty() {
        bot.answer_inline_query(q.id, Vec::<InlineQueryResult>::new())
            .cache_time(5)
            .await?;
        return Ok(());
    }

    let results: Vec<InlineQueryResult> = match search(&cfg, query).await {
        Ok(hits) => inline_entries(&hits)
            .into_iter()
            .map(|(id, file_id)| InlineQueryResultCachedSticker::new(id, FileId(file_id)).into())
            .collect(),
        Err(e) => {
            tracing::warn!(error = %e, query, "inline search failed");
            Vec::new()
        }
    };
    // Short cache so freshly-indexed packs appear quickly; `is_personal` keeps the
    // cache per-user (results are global here, but it avoids cross-user staleness).
    bot.answer_inline_query(q.id, results)
        .cache_time(3)
        .is_personal(true)
        .await?;
    Ok(())
}

async fn on_message(bot: Bot, msg: Message, cfg: Arc<BotConfig>) -> HandlerResult {
    let uid = msg.from.as_ref().map(|u| u.id.0 as i64).unwrap_or(0);

    if let Some(sticker) = msg.sticker() {
        match sticker.set_name.as_deref() {
            Some(set) => add_pack(&bot, &msg, &cfg, set, uid).await?,
            None => {
                bot.send_message(msg.chat.id, "That sticker isn't part of a pack I can add.")
                    .await?;
            }
        }
        return Ok(());
    }

    let Some(text) = msg.text() else {
        return Ok(());
    };
    if text.starts_with("/start") || text.starts_with("/help") {
        bot.send_message(msg.chat.id, help_text(&cfg.bot_username))
            .await?;
    } else if text.starts_with("/add") {
        match add_target_from_text(text) {
            Some(name) => add_pack(&bot, &msg, &cfg, &name, uid).await?,
            None => {
                bot.send_message(
                    msg.chat.id,
                    "Usage: /add <pack link or name> — or just send me a sticker from the pack.",
                )
                .await?;
            }
        }
    } else {
        bot.send_message(msg.chat.id, help_text(&cfg.bot_username))
            .await?;
    }
    Ok(())
}

/// Queue a pack and reply with its current pipeline status. Existence is checked
/// first so a typo'd name is rejected rather than silently queued forever.
async fn add_pack(
    bot: &Bot,
    msg: &Message,
    cfg: &BotConfig,
    raw_name: &str,
    uid: i64,
) -> HandlerResult {
    let name = normalize_pack_name(raw_name);
    let gateway = BotApiGateway::new(cfg.token.clone());
    match gateway.get_sticker_set(&name).await {
        Ok(_) => {}
        Err(GatewayError::NotFound(_)) => {
            bot.send_message(
                msg.chat.id,
                format!("No pack named “{name}” — check the link."),
            )
            .await?;
            return Ok(());
        }
        // Telegram hiccup: don't lose the request — queue it anyway.
        Err(e) => tracing::warn!(error = %e, pack = %name, "could not verify pack; queuing anyway"),
    }

    open_repo(cfg)?.enqueue(&name, uid, OffsetDateTime::now_utc())?;

    let reply = match pack_report(cfg, &name).await {
        Ok(report) => format_status(&report, &cfg.bot_username),
        Err(e) => {
            tracing::warn!(error = %e, pack = %name, "status lookup failed; sending generic ack");
            format!("📥 Queued “{name}”. It'll be processed on the next batch run.")
        }
    };
    bot.send_message(msg.chat.id, reply).await?;
    Ok(())
}

async fn pack_report(cfg: &BotConfig, name: &str) -> Result<sticker_core::PackReport, AppError> {
    let status = PackStatus::new(open_repo(cfg)?, open_repo(cfg)?, vector_store(cfg)?);
    Ok(status
        .report(
            name,
            &cfg.caption_model,
            &cfg.prompt_version,
            &cfg.embed_model,
        )
        .await?)
}

fn help_text(bot_username: &str) -> String {
    format!(
        "Sticker search.\n\n\
         • Search: type @{bot_username} then your query in any chat — results show as you type.\n\
         • Add a pack: /add <link or name>, or send me a sticker from it. \
         I can't see your installed packs, so point me at the ones you want indexed.",
    )
}

async fn run() -> Result<(), AppError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    let token = std::env::var("TELEGRAM_BOT_TOKEN").map_err(|_| AppError::MissingToken)?;
    let ollama_url = cli
        .ollama_url
        .or_else(|| std::env::var("OLLAMA_HOST").ok())
        .unwrap_or_else(|| "http://localhost:11434".to_string());

    let bot = Bot::new(token.clone());
    let me = bot
        .get_me()
        .await
        .map_err(|e| AppError::Search(SearchError::Repo(RepoError::Storage(e.to_string()))))?;
    let bot_username = me.username().to_string();

    let cfg = Arc::new(BotConfig {
        db: cli.db,
        token,
        bot_username,
        caption_model: cli.caption_model,
        prompt_version: cli.prompt_version,
        embed_model: cli.embed_model,
        dim: cli.dim,
        limit: cli.limit.clamp(1, MAX_INLINE),
        min_score: cli.min_score,
        normalization: if cli.no_normalize {
            Normalization::Off
        } else {
            Normalization::default()
        },
        ollama_url,
        qdrant_url: cli.qdrant_url,
    });

    let collection =
        sticker_core::collection_name(&cfg.caption_model, &cfg.prompt_version, &cfg.embed_model);
    tracing::info!(db = %cfg.db.display(), %collection, "starting bot @{}", cfg.bot_username);

    let handler = dptree::entry()
        .branch(Update::filter_inline_query().endpoint(on_inline))
        .branch(Update::filter_message().endpoint(on_message));

    Dispatcher::builder(bot, handler)
        .dependencies(dptree::deps![cfg])
        .enable_ctrlc_handler()
        .build()
        .dispatch()
        .await;
    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
