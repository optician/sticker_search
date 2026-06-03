//! Domain + application layer for sticker-search (onion core).
//!
//! This crate is pure: it defines entities, ports (traits), errors, and the
//! `ScrapePacks` use-case. It must NOT depend on rusqlite, reqwest, or any other
//! I/O crate — adapters live in `sticker-infra`.

pub mod caption;
pub mod entities;
pub mod error;
pub mod ports;
pub mod scrape;

pub use caption::{CaptionProgress, CaptionRun, CaptionStickers, CaptionSummary, ProgressEvent};
pub use entities::{
    Caption, CaptionFields, CaptionResult, FileData, Pack, Prompt, RemoteSticker, RemoteStickerSet,
    Sticker, StickerFormat,
};
pub use error::{
    CaptionError, CaptionGatewayError, CaptionStickerError, GatewayError, RepoError, ScrapeError,
    StoreError,
};
pub use ports::{
    CaptionGateway, CaptionRepository, ImageStore, StickerRepository, TelegramGateway,
};
pub use scrape::{ScrapePacks, ScrapeSummary};
