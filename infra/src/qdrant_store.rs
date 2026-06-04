//! Qdrant-backed `VectorStore` over its REST API (port 6333), using reqwest in
//! the same hand-rolled style as the Ollama gateways — no gRPC/tonic stack.
//!
//! Collection-per-set: each `(caption_model, prompt_version, embed_model)` maps
//! to one collection; points are keyed by the sticker UUID.

use serde_json::{Value, json};
use std::time::Duration;
use sticker_core::entities::{DistanceMetric, VectorPoint};
use sticker_core::error::VectorStoreError;
use sticker_core::ports::VectorStore;
use uuid::Uuid;

pub struct QdrantVectorStore {
    client: reqwest::Client,
    base_url: String,
}

impl QdrantVectorStore {
    /// `base_url` like `http://localhost:6333`.
    pub fn new(base_url: impl Into<String>, timeout: Duration) -> Result<Self, VectorStoreError> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| VectorStoreError::Transport(e.to_string()))?;
        Ok(Self { client, base_url: base_url.into().trim_end_matches('/').to_string() })
    }

    async fn get(&self, path: &str) -> Result<reqwest::Response, VectorStoreError> {
        self.client
            .get(format!("{}{}", self.base_url, path))
            .send()
            .await
            .map_err(transport)
    }
}

fn transport(e: reqwest::Error) -> VectorStoreError {
    if e.is_timeout() {
        VectorStoreError::Transport("request timed out".into())
    } else {
        VectorStoreError::Transport(e.to_string())
    }
}

/// Qdrant's name for each distance metric.
fn metric_str(m: DistanceMetric) -> &'static str {
    match m {
        DistanceMetric::Cosine => "Cosine",
        DistanceMetric::Dot => "Dot",
        DistanceMetric::Euclid => "Euclid",
    }
}

/// Body for `PUT /collections/{name}`.
fn create_collection_body(dim: usize, metric: DistanceMetric) -> Value {
    json!({ "vectors": { "size": dim, "distance": metric_str(metric) } })
}

/// Body for `PUT /collections/{name}/points`: a single point with the sticker
/// UUID as its id and the provenance as payload.
fn upsert_body(point: &VectorPoint) -> Value {
    json!({
        "points": [{
            "id": point.id.to_string(),
            "vector": point.vector,
            "payload": {
                "sticker_id": point.payload.sticker_id.to_string(),
                "caption_model": point.payload.caption_model,
                "prompt_version": point.payload.prompt_version,
                "embed_model": point.payload.embed_model,
            },
        }],
    })
}

impl VectorStore for QdrantVectorStore {
    async fn ensure_collection(
        &self,
        collection: &str,
        dim: usize,
        metric: DistanceMetric,
    ) -> Result<(), VectorStoreError> {
        // Idempotent: only create when absent, so re-runs don't clobber data.
        let existing = self.get(&format!("/collections/{collection}")).await?;
        if existing.status().is_success() {
            return Ok(());
        }

        let resp = self
            .client
            .put(format!("{}/collections/{}", self.base_url, collection))
            .json(&create_collection_body(dim, metric))
            .send()
            .await
            .map_err(transport)?;
        check_success(resp).await.map(|_| ())
    }

    async fn has_vector(
        &self,
        collection: &str,
        point_id: Uuid,
    ) -> Result<bool, VectorStoreError> {
        let resp = self.get(&format!("/collections/{collection}/points/{point_id}")).await?;
        match resp.status().as_u16() {
            200 => Ok(true),
            404 => Ok(false),
            code => Err(VectorStoreError::HttpStatus(code, body_text(resp).await)),
        }
    }

    async fn upsert(
        &self,
        collection: &str,
        point: &VectorPoint,
    ) -> Result<(), VectorStoreError> {
        let resp = self
            .client
            .put(format!("{}/collections/{}/points?wait=true", self.base_url, collection))
            .json(&upsert_body(point))
            .send()
            .await
            .map_err(transport)?;
        check_success(resp).await.map(|_| ())
    }
}

async fn check_success(resp: reqwest::Response) -> Result<reqwest::Response, VectorStoreError> {
    let status = resp.status();
    if status.is_success() {
        Ok(resp)
    } else {
        Err(VectorStoreError::HttpStatus(status.as_u16(), body_text(resp).await))
    }
}

async fn body_text(resp: reqwest::Response) -> String {
    resp.text().await.unwrap_or_else(|_| "<no body>".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sticker_core::entities::VectorPayload;

    #[test]
    fn metric_maps_to_qdrant_names() {
        assert_eq!(metric_str(DistanceMetric::Cosine), "Cosine");
        assert_eq!(metric_str(DistanceMetric::Dot), "Dot");
        assert_eq!(metric_str(DistanceMetric::Euclid), "Euclid");
    }

    #[test]
    fn create_collection_body_carries_size_and_distance() {
        let body = create_collection_body(1024, DistanceMetric::Cosine);
        assert_eq!(body["vectors"]["size"], 1024);
        assert_eq!(body["vectors"]["distance"], "Cosine");
    }

    #[test]
    fn upsert_body_keys_point_by_uuid_with_vector_and_payload() {
        let id = Uuid::nil();
        let point = VectorPoint {
            id,
            vector: vec![0.5, -0.5],
            payload: VectorPayload {
                sticker_id: id,
                caption_model: "qwen".into(),
                prompt_version: "v1".into(),
                embed_model: "bge-m3".into(),
            },
        };
        let body = upsert_body(&point);
        let p = &body["points"][0];
        assert_eq!(p["id"], id.to_string(), "uuid string id");
        assert_eq!(p["vector"][0], 0.5);
        assert_eq!(p["payload"]["caption_model"], "qwen");
        assert_eq!(p["payload"]["embed_model"], "bge-m3");
        assert_eq!(p["payload"]["sticker_id"], id.to_string());
    }
}
