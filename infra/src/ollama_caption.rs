//! Ollama-backed `CaptionGateway`. Sends the thumbnail (normalized to PNG) plus
//! a configured prompt to `/api/generate` and parses the structured JSON reply.

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::Deserialize;
use std::time::Duration;
use sticker_core::entities::{CaptionFields, CaptionResult};
use sticker_core::error::CaptionGatewayError;
use sticker_core::ports::CaptionGateway;

pub struct OllamaCaptionGateway {
    client: reqwest::Client,
    base_url: String,
    model: String,
    prompt: String,
}

impl OllamaCaptionGateway {
    /// `base_url` like `http://localhost:11434`; `prompt` is the caption prompt
    /// text (its version is tracked separately by the use-case).
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        prompt: impl Into<String>,
        timeout: Duration,
    ) -> Result<Self, CaptionGatewayError> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| CaptionGatewayError::Transport(e.to_string()))?;
        Ok(Self {
            client,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            prompt: prompt.into(),
        })
    }
}

impl CaptionGateway for OllamaCaptionGateway {
    fn model(&self) -> &str {
        &self.model
    }

    async fn caption(&self, image: &[u8]) -> Result<CaptionResult, CaptionGatewayError> {
        let png = to_png(image)?;
        let body = serde_json::json!({
            "model": self.model,
            "prompt": self.prompt,
            "images": [BASE64.encode(&png)],
            "stream": false,
            "format": "json",
        });

        let resp = self
            .client
            .post(format!("{}/api/generate", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    CaptionGatewayError::Timeout
                } else {
                    CaptionGatewayError::Transport(e.to_string())
                }
            })?;

        if !resp.status().is_success() {
            return Err(CaptionGatewayError::HttpStatus(resp.status().as_u16()));
        }

        let text = resp
            .text()
            .await
            .map_err(|e| CaptionGatewayError::Transport(e.to_string()))?;
        parse_caption_response(&text)
    }
}

/// Decode any supported image (webp/png/…) and re-encode as PNG, which the
/// model's vision path reliably accepts.
fn to_png(bytes: &[u8]) -> Result<Vec<u8>, CaptionGatewayError> {
    let img =
        image::load_from_memory(bytes).map_err(|e| CaptionGatewayError::Image(e.to_string()))?;
    let mut out = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .map_err(|e| CaptionGatewayError::Image(e.to_string()))?;
    Ok(out)
}

/// The Ollama `/api/generate` envelope we care about. The structured caption
/// arrives in `response`, but thinking models (qwen3-vl) emit it into `thinking`
/// and leave `response` empty — so we fall back to `thinking`.
#[derive(Deserialize)]
struct OllamaEnvelope {
    #[serde(default)]
    response: String,
    #[serde(default)]
    thinking: String,
}

/// Pure parser, unit-tested without a server. Picks the populated channel, then
/// deserializes the caption JSON. Kept separate from the HTTP call on purpose.
fn parse_caption_response(body: &str) -> Result<CaptionResult, CaptionGatewayError> {
    let env: OllamaEnvelope = serde_json::from_str(body)
        .map_err(|e| CaptionGatewayError::Parse(format!("ollama envelope: {e}")))?;

    let raw = if !env.response.trim().is_empty() {
        env.response
    } else {
        env.thinking
    };
    if raw.trim().is_empty() {
        return Err(CaptionGatewayError::Parse(
            "both response and thinking were empty".into(),
        ));
    }

    let fields: CaptionFields = serde_json::from_str(&raw)
        .map_err(|e| CaptionGatewayError::Parse(format!("caption json: {e}")))?;
    Ok(CaptionResult { fields, raw })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_from_response_channel() {
        let body = r#"{"response":"{\"scene\":\"a cat\",\"on_image_text\":\"hi\",\"tone\":\"cute\",\"situations\":[\"greeting\"]}","thinking":""}"#;
        let r = parse_caption_response(body).unwrap();
        assert_eq!(r.fields.scene, "a cat");
        assert_eq!(r.fields.on_image_text, "hi");
        assert_eq!(r.fields.situations, vec!["greeting".to_string()]);
        assert!(r.raw.contains("a cat"));
    }

    #[test]
    fn falls_back_to_thinking_channel() {
        // qwen3-vl puts the JSON in `thinking` and leaves `response` empty.
        let body = r#"{"response":"","thinking":"{\"scene\":\"chicken\",\"on_image_text\":\"ЗАПАХЛО\",\"tone\":\"humorous\",\"situations\":[]}"}"#;
        let r = parse_caption_response(body).unwrap();
        assert_eq!(r.fields.scene, "chicken");
        assert_eq!(r.fields.on_image_text, "ЗАПАХЛО", "Cyrillic preserved");
    }

    #[test]
    fn defaults_optional_fields() {
        // on_image_text and situations may be omitted.
        let body = r#"{"response":"{\"scene\":\"x\",\"tone\":\"y\"}"}"#;
        let r = parse_caption_response(body).unwrap();
        assert_eq!(r.fields.on_image_text, "");
        assert!(r.fields.situations.is_empty());
    }

    #[test]
    fn errors_on_missing_required_key() {
        // `scene` is required; its absence is a parse error, not a silent default.
        let body = r#"{"response":"{\"tone\":\"y\"}"}"#;
        assert!(matches!(
            parse_caption_response(body),
            Err(CaptionGatewayError::Parse(_))
        ));
    }

    #[test]
    fn errors_when_both_channels_empty() {
        let body = r#"{"response":"  ","thinking":""}"#;
        assert!(matches!(
            parse_caption_response(body),
            Err(CaptionGatewayError::Parse(_))
        ));
    }

    #[test]
    fn errors_on_malformed_envelope() {
        assert!(matches!(
            parse_caption_response("not json"),
            Err(CaptionGatewayError::Parse(_))
        ));
    }

    #[test]
    fn to_png_normalizes_a_generated_image() {
        // Build a tiny RGBA image, encode as webp, confirm to_png round-trips it.
        let img = image::RgbaImage::from_pixel(2, 2, image::Rgba([10, 20, 30, 255]));
        let dynimg = image::DynamicImage::ImageRgba8(img);
        let mut webp = Vec::new();
        dynimg
            .write_to(
                &mut std::io::Cursor::new(&mut webp),
                image::ImageFormat::WebP,
            )
            .unwrap();

        let png = to_png(&webp).unwrap();
        // PNG magic number.
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
    }

    #[test]
    fn to_png_rejects_non_image_bytes() {
        assert!(matches!(
            to_png(b"definitely not an image"),
            Err(CaptionGatewayError::Image(_))
        ));
    }
}
