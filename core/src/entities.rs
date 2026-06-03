//! Domain entities and the gateway transport types.

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use uuid::Uuid;

/// Original format of a sticker. The image we actually store is always a static
/// thumbnail; this records what the source sticker was.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StickerFormat {
    Static,
    Animated,
    Video,
}

impl StickerFormat {
    /// Map Telegram's `is_animated` / `is_video` flags. Video takes precedence:
    /// a `.webm` sticker reports neither in some API versions, but when both are
    /// set we treat it as video.
    pub fn from_flags(is_animated: bool, is_video: bool) -> Self {
        match (is_animated, is_video) {
            (_, true) => Self::Video,
            (true, false) => Self::Animated,
            (false, false) => Self::Static,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Static => "static",
            Self::Animated => "animated",
            Self::Video => "video",
        }
    }
}

/// A persisted sticker pack. Keyed by `name` (the `t.me/addstickers/<name>`),
/// with a stable UUID for foreign references.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pack {
    pub id: Uuid,
    pub name: String,
    pub title: String,
    #[serde(with = "time::serde::rfc3339")]
    pub fetched_at: OffsetDateTime,
}

/// A persisted sticker. `id` is intended to double as the future vector-DB key.
/// `file_unique_id` is the stable idempotency key across runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sticker {
    pub id: Uuid,
    pub pack_id: Uuid,
    pub file_unique_id: String,
    pub file_id: String,
    pub emoji: Option<String>,
    pub format: StickerFormat,
    pub width: u32,
    pub height: u32,
    pub position: u32,
    /// Path of the stored thumbnail image, relative to the stickers root.
    pub image_path: String,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
}

/// A sticker set as returned by the gateway, before we mint UUIDs / persist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteStickerSet {
    pub name: String,
    pub title: String,
    pub stickers: Vec<RemoteSticker>,
}

/// A single sticker as returned by the gateway.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteSticker {
    pub file_unique_id: String,
    pub file_id: String,
    /// Preferred download id: the static thumbnail. Falls back to `file_id`.
    pub thumb_file_id: Option<String>,
    pub emoji: Option<String>,
    pub is_animated: bool,
    pub is_video: bool,
    pub width: u32,
    pub height: u32,
}

/// Raw bytes of a downloaded file plus the extension derived from its Telegram
/// file path (e.g. `webp`, `jpg`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileData {
    pub bytes: Vec<u8>,
    pub ext: String,
}

/// The parsed caption a `CaptionGateway` returns for one image, before the
/// use-case attaches the sticker key, model, prompt version, and timestamp.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptionFields {
    /// Literal scene description.
    pub scene: String,
    /// Verbatim on-image text (OCR); empty string when the image has none.
    #[serde(default)]
    pub on_image_text: String,
    /// Emotional tone.
    pub tone: String,
    /// Situations someone would send this sticker in.
    #[serde(default)]
    pub situations: Vec<String>,
}

/// What a `CaptionGateway` returns: the parsed fields plus the model's raw JSON
/// (persisted as `Caption::raw` for debugging / reprocessing).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptionResult {
    pub fields: CaptionFields,
    pub raw: String,
}

/// A persisted caption, keyed by `(sticker_id, model, prompt_version)` so model
/// picks and prompt iterations coexist rather than overwriting each other.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Caption {
    pub sticker_id: Uuid,
    pub model: String,
    pub prompt_version: String,
    pub scene: String,
    pub on_image_text: String,
    pub tone: String,
    pub situations: Vec<String>,
    /// The model's raw JSON response, kept for debugging / reprocessing.
    pub raw: String,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
}

/// A captioning prompt, stored once per version. The integrity guard in the
/// use-case ensures a `version` string maps to exactly one `text`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Prompt {
    pub version: String,
    pub text: String,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
}
