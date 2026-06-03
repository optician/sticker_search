//! Ports: the interfaces the application depends on. Adapters in `sticker-infra`
//! implement these; tests use in-memory fakes.

use crate::entities::{
    Caption, CaptionResult, FileData, Pack, Prompt, RemoteStickerSet, Sticker,
};
use crate::error::{CaptionGatewayError, GatewayError, RepoError, StoreError};
use uuid::Uuid;

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
    /// All stickers, optionally restricted to a single pack name, ordered by
    /// pack then position. Drives the captioning batch.
    fn list_stickers(&self, pack: Option<&str>) -> Result<Vec<Sticker>, RepoError>;
}

/// Storage of the downloaded thumbnail images on disk.
pub trait ImageStore {
    /// Whether `<pack>/<file_name>` already exists.
    fn exists(&self, pack: &str, file_name: &str) -> bool;
    /// Save bytes to `<pack>/<file_name>`, returning the path relative to the
    /// stickers root (stored in `Sticker::image_path`).
    fn save(&self, pack: &str, file_name: &str, bytes: &[u8]) -> Result<String, StoreError>;
    /// Read the bytes at a path relative to the stickers root (a stored
    /// `Sticker::image_path`).
    fn read(&self, image_path: &str) -> Result<Vec<u8>, StoreError>;
}

/// Source of captions for an image (a local VLM via Ollama in production).
///
/// The adapter owns the model wire format and returns parsed fields; core never
/// sees JSON. `model` names the model so the use-case can key and dedup captions.
#[allow(async_fn_in_trait)]
pub trait CaptionGateway {
    fn model(&self) -> &str;
    /// Caption a single image. `image` is the stored file bytes (e.g. webp); the
    /// adapter normalizes the encoding as the model requires.
    async fn caption(&self, image: &[u8]) -> Result<CaptionResult, CaptionGatewayError>;
}

/// Persistence of captions and the prompts that produced them. Keyed by
/// `(sticker_id, model, prompt_version)`.
pub trait CaptionRepository {
    fn caption_exists(
        &self,
        sticker_id: Uuid,
        model: &str,
        prompt_version: &str,
    ) -> Result<bool, RepoError>;
    fn upsert_caption(&self, caption: &Caption) -> Result<(), RepoError>;
    fn find_prompt(&self, version: &str) -> Result<Option<Prompt>, RepoError>;
    fn upsert_prompt(&self, prompt: &Prompt) -> Result<(), RepoError>;
}
