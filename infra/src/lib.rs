//! Infrastructure adapters for sticker-search (outer onion ring).
//!
//! Each module implements a port from `sticker-core`:
//! - [`sqlite_repo::SqliteRepository`] → `StickerRepository` + `CaptionRepository`
//!   + `CaptionReader`
//! - [`fs_images::FsImageStore`] → `ImageStore`
//! - [`bot_api::BotApiGateway`] → `TelegramGateway`
//! - [`ollama_caption::OllamaCaptionGateway`] → `CaptionGateway`
//! - [`ollama_embed::OllamaEmbeddingGateway`] → `EmbeddingGateway`
//! - [`qdrant_store::QdrantVectorStore`] → `VectorStore`

pub mod bot_api;
pub mod fs_images;
pub mod ollama_caption;
pub mod ollama_embed;
pub mod qdrant_store;
pub mod sqlite_repo;

pub use bot_api::BotApiGateway;
pub use fs_images::FsImageStore;
pub use ollama_caption::OllamaCaptionGateway;
pub use ollama_embed::OllamaEmbeddingGateway;
pub use qdrant_store::QdrantVectorStore;
pub use sqlite_repo::{CaptionFilter, CaptionSort, CaptionStat, CaptionView, SqliteRepository};
