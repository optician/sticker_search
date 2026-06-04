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

/// A caption enriched with the sticker- and pack-level context the embedder
/// folds into one document. `Caption` stays a faithful mirror of the `captions`
/// table; the emoji lives on `stickers` and the name/title on `packs`, so they
/// ride alongside here rather than polluting `Caption`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedDoc {
    pub caption: Caption,
    /// The sticker's sender-assigned emoji (`stickers.emoji`); an intent signal.
    pub emoji: Option<String>,
    /// Pack slug (`packs.name`), e.g. `hamstersad`.
    pub pack_name: String,
    /// Human-readable pack title (`packs.title`), e.g. `Sad Hamster`.
    pub pack_title: String,
}

impl EmbedDoc {
    /// The single text string fed to the embedder. Composes the caption's fields
    /// plus the emoji and pack context into one document so query text and stored
    /// captions share a vector space.
    ///
    /// Order: scene, verbatim on-image text (OCR — often the meme's whole point),
    /// tone, emoji (another intent signal, beside tone), situations, then the pack
    /// as weakest context at the tail. Empty optional fields are dropped so the
    /// embedder never sees dangling labels. Parts are joined with `". "`.
    pub fn embed_text(&self) -> String {
        let c = &self.caption;
        let mut parts = vec![c.scene.clone()];
        if !c.on_image_text.is_empty() {
            parts.push(format!("text: {}", c.on_image_text));
        }
        parts.push(format!("tone: {}", c.tone));
        if let Some(emoji) = self.emoji.as_deref().filter(|e| !e.is_empty()) {
            parts.push(format!("emoji: {emoji}"));
        }
        if !c.situations.is_empty() {
            parts.push(format!("situations: {}", c.situations.join(", ")));
        }
        parts.push(format!("pack: {} ({})", self.pack_title, self.pack_name));
        parts.join(". ")
    }
}

/// Distance metric a vector collection scores with. Cosine is the default for
/// normalized text embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    #[default]
    Cosine,
    Dot,
    Euclid,
}

/// Provenance stored alongside each vector so a hit resolves back to its sticker
/// and the exact `(caption model, prompt, embed model)` that produced it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorPayload {
    pub sticker_id: Uuid,
    pub caption_model: String,
    pub prompt_version: String,
    pub embed_model: String,
}

/// One point to store: the sticker UUID as the id, its vector, and provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorPoint {
    pub id: Uuid,
    pub vector: Vec<f32>,
    pub payload: VectorPayload,
}

/// A single search result from the vector store: a point id (sticker UUID) and
/// its similarity score. The store returns these ranked best-first.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredPoint {
    pub id: Uuid,
    pub score: f32,
}

/// A resolved search hit: the vector score plus the sticker and the caption that
/// produced its vector. Returned by the `SearchStickers` use-case in rank order
/// so a caller can both display the image and judge *why* it matched.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub score: f32,
    pub sticker: Sticker,
    pub caption: Caption,
}

/// A user's request to index a pack, recorded by the bot. The offline pipeline
/// drains these by name; the *status* of a request is derived from the pipeline's
/// own data (see [`crate::pack::PackStatus`]), never written back here. First
/// request for a name wins — re-requests don't overwrite `requested_by`/`at`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackRequest {
    pub name: String,
    /// Telegram user id that asked for it.
    pub requested_by: i64,
    pub requested_at: OffsetDateTime,
}

/// How far a requested pack has progressed through `scrape → caption → embed`.
/// Derived on demand from the pipeline's own data, so nothing needs to write
/// progress back. Each stage means *every* sticker has reached at least it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackStage {
    /// Requested, but no stickers stored yet (not scraped, or scrape failed).
    Queued,
    /// Stickers stored, but not all captioned for the asked (model, prompt).
    Scraped,
    /// All captioned, but not all embedded into the collection.
    Captioned,
    /// Every sticker embedded — searchable.
    Ready,
}

/// A derived status report for one requested pack: its stage plus the counts the
/// stage is computed from, so a caller can show "captioned 12/50".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackReport {
    pub name: String,
    pub stage: PackStage,
    pub sticker_count: usize,
    pub captioned_count: usize,
    pub embedded_count: usize,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn caption(scene: &str, ocr: &str, tone: &str, situations: &[&str]) -> Caption {
        Caption {
            sticker_id: Uuid::nil(),
            model: "qwen".into(),
            prompt_version: "v1".into(),
            scene: scene.into(),
            on_image_text: ocr.into(),
            tone: tone.into(),
            situations: situations.iter().map(|s| s.to_string()).collect(),
            raw: String::new(),
            created_at: OffsetDateTime::UNIX_EPOCH,
        }
    }

    fn doc(caption: Caption, emoji: Option<&str>) -> EmbedDoc {
        EmbedDoc {
            caption,
            emoji: emoji.map(Into::into),
            pack_name: "hamstersad".into(),
            pack_title: "Sad Hamster".into(),
        }
    }

    #[test]
    fn embed_text_composes_all_fields_in_order() {
        let d = doc(
            caption(
                "a chicken on a pan",
                "ЗАПАХЛО",
                "humorous",
                &["cooking", "panic"],
            ),
            Some("🥹"),
        );
        assert_eq!(
            d.embed_text(),
            "a chicken on a pan. text: ЗАПАХЛО. tone: humorous. emoji: 🥹. \
             situations: cooking, panic. pack: Sad Hamster (hamstersad)",
        );
    }

    #[test]
    fn embed_text_drops_empty_ocr_situations_and_emoji() {
        // No OCR, no situations, no emoji — but tone and pack are always present.
        let d = doc(caption("a plain cat", "", "calm", &[]), None);
        assert_eq!(
            d.embed_text(),
            "a plain cat. tone: calm. pack: Sad Hamster (hamstersad)",
        );
    }

    #[test]
    fn embed_text_drops_empty_string_emoji() {
        let d = doc(caption("a plain cat", "", "calm", &[]), Some(""));
        assert_eq!(
            d.embed_text(),
            "a plain cat. tone: calm. pack: Sad Hamster (hamstersad)",
        );
    }
}
