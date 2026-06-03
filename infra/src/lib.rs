//! Infrastructure adapters for sticker-search (outer onion ring).
//!
//! Each module implements a port from `sticker-core`:
//! - [`sqlite_repo::SqliteRepository`] → `StickerRepository` + `CaptionRepository`
//! - [`fs_images::FsImageStore`] → `ImageStore`
//! - [`bot_api::BotApiGateway`] → `TelegramGateway`
//! - [`ollama_caption::OllamaCaptionGateway`] → `CaptionGateway`

pub mod bot_api;
pub mod fs_images;
pub mod ollama_caption;
pub mod sqlite_repo;

pub use bot_api::BotApiGateway;
pub use fs_images::FsImageStore;
pub use ollama_caption::OllamaCaptionGateway;
pub use sqlite_repo::{CaptionFilter, CaptionSort, CaptionStat, CaptionView, SqliteRepository};
