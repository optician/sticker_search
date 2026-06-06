//! Domain + application layer for sticker-search (onion core).
//!
//! This crate is pure: it defines entities, ports (traits), errors, and the
//! `ScrapePacks` use-case. It must NOT depend on rusqlite, reqwest, or any other
//! I/O crate — adapters live in `sticker-infra`.

pub mod caption;
pub mod embed;
pub mod entities;
pub mod error;
pub mod normalize;
pub mod pack;
pub mod ports;
pub mod scrape;
pub mod search;

pub use caption::{CaptionProgress, CaptionRun, CaptionStickers, CaptionSummary, ProgressEvent};
pub use embed::{
    EmbedCaptions, EmbedEvent, EmbedProgress, EmbedRun, EmbedSummary, collection_name,
};
pub use entities::{
    Caption, CaptionFields, CaptionResult, DistanceMetric, FileData, Pack, PackReport, PackRequest,
    PackStage, Prompt, RemoteSticker, RemoteStickerSet, ScoredPoint, SearchHit, Sticker,
    StickerFormat, VectorPayload, VectorPoint,
};
pub use error::{
    CaptionError, CaptionGatewayError, CaptionStickerError, EmbedError, EmbedStickerError,
    EmbeddingGatewayError, GatewayError, PackStatusError, RepoError, ScrapeError, SearchError,
    StoreError, VectorStoreError,
};
pub use normalize::{Normalization, normalize_for_embedding};
pub use pack::{PackStatus, normalize_pack_name};
pub use ports::{
    CaptionGateway, CaptionLookup, CaptionReader, CaptionRepository, EmbeddingGateway, ImageStore,
    PackRequests, StickerRepository, TelegramGateway, VectorStore,
};
pub use scrape::{ScrapePacks, ScrapeSummary};
pub use search::{SearchQuery, SearchStickers};
