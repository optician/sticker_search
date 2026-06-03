//! Ports: the interfaces the application depends on. Adapters in `sticker-infra`
//! implement these; tests use in-memory fakes.

use crate::entities::{FileData, Pack, RemoteStickerSet, Sticker};
use crate::error::{GatewayError, RepoError, StoreError};

/// Source of sticker data (Telegram Bot API in production).
///
/// Native `async fn` in trait; the use-case is generic over the implementor
/// (static dispatch), so no `async-trait` and no dyn-safety concerns.
#[allow(async_fn_in_trait)]
pub trait TelegramGateway {
    async fn get_sticker_set(&self, name: &str) -> Result<RemoteStickerSet, GatewayError>;
    async fn download_file(&self, file_id: &str) -> Result<FileData, GatewayError>;
}

/// Persistence of pack/sticker metadata. Sync: rusqlite's `Connection` takes
/// `&self` for writes, so interior mutability lives in the adapter.
pub trait StickerRepository {
    fn find_pack_by_name(&self, name: &str) -> Result<Option<Pack>, RepoError>;
    fn upsert_pack(&self, pack: &Pack) -> Result<(), RepoError>;
    fn find_sticker_by_unique_id(&self, file_unique_id: &str)
        -> Result<Option<Sticker>, RepoError>;
    fn upsert_sticker(&self, sticker: &Sticker) -> Result<(), RepoError>;
}

/// Storage of the downloaded thumbnail images on disk.
pub trait ImageStore {
    /// Whether `<pack>/<file_name>` already exists.
    fn exists(&self, pack: &str, file_name: &str) -> bool;
    /// Save bytes to `<pack>/<file_name>`, returning the path relative to the
    /// stickers root (stored in `Sticker::image_path`).
    fn save(&self, pack: &str, file_name: &str, bytes: &[u8]) -> Result<String, StoreError>;
}
