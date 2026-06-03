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
