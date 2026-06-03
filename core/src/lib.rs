//! Domain + application layer for sticker-search (onion core).
//!
//! This crate is pure: it defines entities, ports (traits), errors, and the
//! `ScrapePacks` use-case. It must NOT depend on rusqlite, reqwest, or any other
//! I/O crate — adapters live in `sticker-infra`.

pub mod entities;
pub mod error;
pub mod ports;
pub mod scrape;

pub use entities::{FileData, Pack, RemoteSticker, RemoteStickerSet, Sticker, StickerFormat};
pub use error::{GatewayError, RepoError, ScrapeError, StoreError};
pub use ports::{ImageStore, StickerRepository, TelegramGateway};
pub use scrape::{ScrapePacks, ScrapeSummary};
