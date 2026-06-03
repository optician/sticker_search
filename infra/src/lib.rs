//! Infrastructure adapters for sticker-search (outer onion ring).
//!
//! Each module implements a port from `sticker-core`:
//! - [`sqlite_repo::SqliteRepository`] → `StickerRepository`
//! - [`fs_images::FsImageStore`] → `ImageStore`
//! - [`bot_api::BotApiGateway`] → `TelegramGateway`

pub mod bot_api;
pub mod fs_images;
pub mod sqlite_repo;

pub use bot_api::BotApiGateway;
pub use fs_images::FsImageStore;
pub use sqlite_repo::SqliteRepository;
