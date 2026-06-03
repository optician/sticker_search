//! Telegram Bot API `TelegramGateway` adapter (reqwest).
//!
//! JSON → domain mapping lives in [`parse_sticker_set`] so it can be unit-tested
//! without the network.

use serde::Deserialize;
use sticker_core::entities::{FileData, RemoteSticker, RemoteStickerSet};
use sticker_core::error::GatewayError;
use sticker_core::ports::TelegramGateway;

pub struct BotApiGateway {
    client: reqwest::Client,
    token: String,
    base: String,
}

impl BotApiGateway {
    pub fn new(token: impl Into<String>) -> Self {
        Self::with_base(token, "https://api.telegram.org")
    }

    /// Override the base URL (used to point at a mock server in tests).
    pub fn with_base(token: impl Into<String>, base: impl Into<String>) -> Self {
        Self { client: reqwest::Client::new(), token: token.into(), base: base.into() }
    }

    fn method_url(&self, method: &str) -> String {
        format!("{}/bot{}/{}", self.base, self.token, method)
    }

    fn file_url(&self, file_path: &str) -> String {
        format!("{}/file/bot{}/{}", self.base, self.token, file_path)
    }
}

fn transport<E: std::fmt::Display>(e: E) -> GatewayError {
    GatewayError::Transport(e.to_string())
}

// ---- wire types ----

#[derive(Deserialize)]
struct ApiResponse<T> {
    ok: bool,
    result: Option<T>,
    description: Option<String>,
    parameters: Option<ApiParams>,
}

#[derive(Deserialize)]
struct ApiParams {
    retry_after: Option<u64>,
}

#[derive(Deserialize)]
struct RawSet {
    name: String,
    title: String,
    stickers: Vec<RawSticker>,
}

#[derive(Deserialize)]
struct RawSticker {
    file_id: String,
    file_unique_id: String,
    #[serde(default)]
    emoji: Option<String>,
    #[serde(default)]
    is_animated: bool,
    #[serde(default)]
    is_video: bool,
    #[serde(default)]
    width: u32,
    #[serde(default)]
    height: u32,
    #[serde(default)]
    thumbnail: Option<RawThumb>,
}

#[derive(Deserialize)]
struct RawThumb {
    file_id: String,
}

#[derive(Deserialize)]
struct RawFile {
    file_path: Option<String>,
}

fn parse_sticker_set(raw: RawSet) -> RemoteStickerSet {
    let stickers = raw
        .stickers
        .into_iter()
        .map(|s| RemoteSticker {
            file_unique_id: s.file_unique_id,
            file_id: s.file_id,
            thumb_file_id: s.thumbnail.map(|t| t.file_id),
            emoji: s.emoji,
            is_animated: s.is_animated,
            is_video: s.is_video,
            width: s.width,
            height: s.height,
        })
        .collect();
    RemoteStickerSet { name: raw.name, title: raw.title, stickers }
}

fn ext_of(file_path: &str) -> String {
    std::path::Path::new(file_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("bin")
        .to_string()
}

impl TelegramGateway for BotApiGateway {
    async fn get_sticker_set(&self, name: &str) -> Result<RemoteStickerSet, GatewayError> {
        let resp = self
            .client
            .get(self.method_url("getStickerSet"))
            .query(&[("name", name)])
            .send()
            .await
            .map_err(transport)?;
        let status = resp.status().as_u16();
        let body: ApiResponse<RawSet> = resp.json().await.map_err(transport)?;

        if body.ok {
            let raw = body
                .result
                .ok_or_else(|| GatewayError::Transport("getStickerSet: missing result".into()))?;
            return Ok(parse_sticker_set(raw));
        }
        if status == 429 {
            let after = body.parameters.and_then(|p| p.retry_after).unwrap_or(0);
            return Err(GatewayError::RateLimited(after));
        }
        let desc = body.description.unwrap_or_default();
        if status == 400 || desc.contains("STICKERSET_INVALID") || desc.contains("not found") {
            return Err(GatewayError::NotFound(name.to_string()));
        }
        Err(GatewayError::Transport(desc))
    }

    async fn download_file(&self, file_id: &str) -> Result<FileData, GatewayError> {
        let resp = self
            .client
            .get(self.method_url("getFile"))
            .query(&[("file_id", file_id)])
            .send()
            .await
            .map_err(transport)?;
        let body: ApiResponse<RawFile> = resp.json().await.map_err(transport)?;
        if !body.ok {
            return Err(GatewayError::Transport(body.description.unwrap_or_default()));
        }
        let file_path = body
            .result
            .and_then(|r| r.file_path)
            .ok_or_else(|| GatewayError::Transport("getFile: missing file_path".into()))?;
        let ext = ext_of(&file_path);

        let bytes = self
            .client
            .get(self.file_url(&file_path))
            .send()
            .await
            .map_err(transport)?
            .bytes()
            .await
            .map_err(transport)?;
        Ok(FileData { bytes: bytes.to_vec(), ext })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
      "ok": true,
      "result": {
        "name": "packA",
        "title": "Pack A",
        "sticker_type": "regular",
        "stickers": [
          {
            "width": 512, "height": 512, "emoji": "😀",
            "is_animated": false, "is_video": true,
            "thumbnail": { "file_id": "THUMB1", "file_unique_id": "tu1" },
            "file_id": "FILE1", "file_unique_id": "u1"
          },
          {
            "width": 100, "height": 100,
            "is_animated": true, "is_video": false,
            "file_id": "FILE2", "file_unique_id": "u2"
          }
        ]
      }
    }"#;

    #[test]
    fn parses_sticker_set_and_maps_fields() {
        let resp: ApiResponse<RawSet> = serde_json::from_str(SAMPLE).unwrap();
        let set = parse_sticker_set(resp.result.unwrap());

        assert_eq!(set.name, "packA");
        assert_eq!(set.stickers.len(), 2);

        let a = &set.stickers[0];
        assert_eq!(a.file_unique_id, "u1");
        assert_eq!(a.file_id, "FILE1");
        assert_eq!(a.thumb_file_id.as_deref(), Some("THUMB1"));
        assert_eq!(a.emoji.as_deref(), Some("😀"));
        assert!(a.is_video);

        let b = &set.stickers[1];
        assert_eq!(b.thumb_file_id, None, "missing thumbnail → None (falls back to file_id later)");
        assert!(b.is_animated);
        assert_eq!(b.emoji, None);
    }

    #[test]
    fn derives_extension_from_file_path() {
        assert_eq!(ext_of("thumbnails/file_3.webp"), "webp");
        assert_eq!(ext_of("videos/file_7.webm"), "webm");
        assert_eq!(ext_of("noext"), "bin");
    }
}
