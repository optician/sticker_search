//! Domain + application layer for sticker-search (onion core).
//!
//! This crate is pure: it defines entities, ports (traits), errors, and the
//! `ScrapePacks` use-case. It must NOT depend on rusqlite, reqwest, or any other
//! I/O crate — adapters live in `sticker-infra`.

pub mod caption;
pub mod embed;
pub mod entities;
pub mod error;
pub mod ports;
pub mod scrape;

pub use caption::{CaptionProgress, CaptionRun, CaptionStickers, CaptionSummary, ProgressEvent};
pub use embed::{
    EmbedCaptions, EmbedEvent, EmbedProgress, EmbedRun, EmbedSummary, collection_name,
};
pub use entities::{
    Caption, CaptionFields, CaptionResult, DistanceMetric, FileData, Pack, Prompt, RemoteSticker,
    RemoteStickerSet, Sticker, StickerFormat, VectorPayload, VectorPoint,
};
pub use error::{
    CaptionError, CaptionGatewayError, CaptionStickerError, EmbedError, EmbedStickerError,
    EmbeddingGatewayError, GatewayError, RepoError, ScrapeError, StoreError, VectorStoreError,
};
pub use ports::{
    CaptionGateway, CaptionReader, CaptionRepository, EmbeddingGateway, ImageStore,
    StickerRepository, TelegramGateway, VectorStore,
};
pub use scrape::{ScrapePacks, ScrapeSummary};
