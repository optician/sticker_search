//! Ollama-backed `EmbeddingGateway`. Sends caption text to `/api/embed` and
//! returns the dense vector. Mirrors `OllamaCaptionGateway`'s reqwest style.

use serde::Deserialize;
use std::time::Duration;
use sticker_core::error::EmbeddingGatewayError;
use sticker_core::ports::EmbeddingGateway;

pub struct OllamaEmbeddingGateway {
    client: reqwest::Client,
    base_url: String,
    model: String,
    dim: usize,
}

impl OllamaEmbeddingGateway {
    /// `base_url` like `http://localhost:11434`; `dim` is the model's known
    /// vector dimensionality (e.g. 1024 for `bge-m3`), used to create the
    /// collection and reject mismatched responses.
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        dim: usize,
        timeout: Duration,
    ) -> Result<Self, EmbeddingGatewayError> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| EmbeddingGatewayError::Transport(e.to_string()))?;
        Ok(Self {
            client,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            dim,
        })
    }
}

impl EmbeddingGateway for OllamaEmbeddingGateway {
    fn model(&self) -> &str {
        &self.model
    }

    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingGatewayError> {
        let body = serde_json::json!({ "model": self.model, "input": text });

        let resp = self
            .client
            .post(format!("{}/api/embed", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    EmbeddingGatewayError::Timeout
                } else {
                    EmbeddingGatewayError::Transport(e.to_string())
                }
            })?;

        if !resp.status().is_success() {
            return Err(EmbeddingGatewayError::HttpStatus(resp.status().as_u16()));
        }

        let text = resp
            .text()
            .await
            .map_err(|e| EmbeddingGatewayError::Transport(e.to_string()))?;
        parse_embed_response(&text)
    }
}

/// The `/api/embed` envelope: a batch of vectors under `embeddings`. We send one
/// input, so we take the first row.
#[derive(Deserialize)]
struct EmbedEnvelope {
    #[serde(default)]
    embeddings: Vec<Vec<f32>>,
}

/// Pure parser, unit-tested without a server.
fn parse_embed_response(body: &str) -> Result<Vec<f32>, EmbeddingGatewayError> {
    let env: EmbedEnvelope = serde_json::from_str(body)
        .map_err(|e| EmbeddingGatewayError::Parse(format!("embed envelope: {e}")))?;
    env.embeddings
        .into_iter()
        .next()
        .filter(|v| !v.is_empty())
        .ok_or_else(|| EmbeddingGatewayError::Parse("no embedding in response".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_first_embedding_row() {
        let body = r#"{"model":"bge-m3","embeddings":[[0.1,-0.2,0.3]]}"#;
        let v = parse_embed_response(body).unwrap();
        assert_eq!(v, vec![0.1, -0.2, 0.3]);
    }

    #[test]
    fn errors_on_empty_batch() {
        assert!(matches!(
            parse_embed_response(r#"{"embeddings":[]}"#),
            Err(EmbeddingGatewayError::Parse(_)),
        ));
    }

    #[test]
    fn errors_on_empty_vector() {
        assert!(matches!(
            parse_embed_response(r#"{"embeddings":[[]]}"#),
            Err(EmbeddingGatewayError::Parse(_)),
        ));
    }

    #[test]
    fn errors_on_malformed_envelope() {
        assert!(matches!(
            parse_embed_response("not json"),
            Err(EmbeddingGatewayError::Parse(_)),
        ));
    }
}
