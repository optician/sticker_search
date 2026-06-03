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
