//! Domain error types.

use thiserror::Error;

/// Errors from the Telegram source.
#[derive(Debug, Error)]
pub enum GatewayError {
    #[error("sticker set not found: {0}")]
    NotFound(String),
    #[error("rate limited; retry after {0}s")]
    RateLimited(u64),
    #[error("telegram transport error: {0}")]
    Transport(String),
}

/// Errors from the metadata store.
#[derive(Debug, Error)]
pub enum RepoError {
    #[error("repository error: {0}")]
    Storage(String),
}

/// Errors from the image store.
#[derive(Debug, Error)]
pub enum StoreError {
    #[error("image store error: {0}")]
    Io(String),
}

/// Aggregate error for a single pack/sticker operation inside the use-case.
#[derive(Debug, Error)]
pub enum ScrapeError {
    #[error(transparent)]
    Gateway(#[from] GatewayError),
    #[error(transparent)]
    Repo(#[from] RepoError),
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// Errors from the captioning model (e.g. Ollama).
#[derive(Debug, Error)]
pub enum CaptionGatewayError {
    #[error("caption transport error: {0}")]
    Transport(String),
    #[error("caption model returned HTTP {0}")]
    HttpStatus(u16),
    #[error("caption response timed out")]
    Timeout,
    #[error("could not parse caption response: {0}")]
    Parse(String),
    #[error("image decode/encode error: {0}")]
    Image(String),
}

/// Aggregate error for a single sticker captioning operation inside the use-case.
#[derive(Debug, Error)]
pub enum CaptionStickerError {
    #[error(transparent)]
    Gateway(#[from] CaptionGatewayError),
    #[error(transparent)]
    Repo(#[from] RepoError),
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// Errors from the text-embedding model (e.g. Ollama `bge-m3`).
#[derive(Debug, Error)]
pub enum EmbeddingGatewayError {
    #[error("embedding transport error: {0}")]
    Transport(String),
    #[error("embedding model returned HTTP {0}")]
    HttpStatus(u16),
    #[error("embedding response timed out")]
    Timeout,
    #[error("could not parse embedding response: {0}")]
    Parse(String),
}

/// Errors from the vector store (Qdrant over its REST API).
#[derive(Debug, Error)]
pub enum VectorStoreError {
    #[error("vector store transport error: {0}")]
    Transport(String),
    #[error("vector store returned HTTP {0}: {1}")]
    HttpStatus(u16, String),
    #[error("could not parse vector store response: {0}")]
    Parse(String),
}

/// Aggregate error for embedding a single caption. Per-sticker failures are
/// logged and counted; they never abort the run.
#[derive(Debug, Error)]
pub enum EmbedStickerError {
    #[error(transparent)]
    Gateway(#[from] EmbeddingGatewayError),
    #[error(transparent)]
    Store(#[from] VectorStoreError),
    #[error("embedder returned {got} dims but the collection expects {expected}")]
    DimensionMismatch { expected: usize, got: usize },
}

/// Run-level error for the embedding use-case: setup failures that abort before
/// per-sticker work (listing captions, creating the collection).
#[derive(Debug, Error)]
pub enum EmbedError {
    #[error(transparent)]
    Repo(#[from] RepoError),
    #[error(transparent)]
    Store(#[from] VectorStoreError),
}

/// Error for a single query through the `SearchStickers` use-case. A query is
/// one shot (embed → search → resolve), so any failure aborts it; unlike the
/// batch stages there are no per-item counts.
#[derive(Debug, Error)]
pub enum SearchError {
    #[error(transparent)]
    Gateway(#[from] EmbeddingGatewayError),
    #[error(transparent)]
    Store(#[from] VectorStoreError),
    #[error(transparent)]
    Repo(#[from] RepoError),
}

/// Error for deriving a pack's pipeline status. Reads the metadata store and
/// probes the vector store, so either layer can fail it.
#[derive(Debug, Error)]
pub enum PackStatusError {
    #[error(transparent)]
    Repo(#[from] RepoError),
    #[error(transparent)]
    Store(#[from] VectorStoreError),
}

/// Run-level error for the captioning use-case. Carries setup failures that
/// abort before processing (prompt precondition, listing stickers). Per-sticker
/// failures are logged and counted, never wrapped here.
#[derive(Debug, Error)]
pub enum CaptionError {
    #[error(
        "prompt version {version:?} already exists with different text; \
         bump the version when editing the prompt"
    )]
    PromptVersionMismatch { version: String },
    #[error(transparent)]
    Repo(#[from] RepoError),
}
