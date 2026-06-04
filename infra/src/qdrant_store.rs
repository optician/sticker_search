//! Qdrant-backed `VectorStore` over its REST API (port 6333), using reqwest in
//! the same hand-rolled style as the Ollama gateways — no gRPC/tonic stack.
//!
//! Collection-per-set: each `(caption_model, prompt_version, embed_model)` maps
//! to one collection; points are keyed by the sticker UUID.

use serde_json::{Value, json};
use std::time::Duration;
use sticker_core::entities::{DistanceMetric, ScoredPoint, VectorPoint};
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
        Ok(Self {
            client,
            base_url: base_url.into().trim_end_matches('/').to_string(),
        })
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

/// Body for `POST /collections/{name}/points/search`. Payload/vectors aren't
/// needed — the point id is the sticker UUID, which is all the read path uses.
fn search_body(query_vector: &[f32], limit: usize, score_threshold: Option<f32>) -> Value {
    let mut body = json!({
        "vector": query_vector,
        "limit": limit,
        "with_payload": false,
        "with_vector": false,
    });
    if let Some(t) = score_threshold {
        body["score_threshold"] = json!(t);
    }
    body
}

/// Parse a Qdrant search response (`{ "result": [{ "id", "score" }, ...] }`)
/// into ranked points. A malformed id/score is a parse error, not a silent drop.
fn parse_search_results(body: &Value) -> Result<Vec<ScoredPoint>, VectorStoreError> {
    let parse = || -> Option<Result<Vec<ScoredPoint>, VectorStoreError>> {
        let items = body.get("result")?.as_array()?;
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            let id_str = item.get("id")?.as_str()?;
            let score = item.get("score")?.as_f64()? as f32;
            let id = match Uuid::parse_str(id_str) {
                Ok(id) => id,
                Err(e) => {
                    return Some(Err(VectorStoreError::Parse(format!(
                        "point id {id_str:?}: {e}"
                    ))));
                }
            };
            out.push(ScoredPoint { id, score });
        }
        Some(Ok(out))
    };
    parse().unwrap_or_else(|| {
        Err(VectorStoreError::Parse(format!(
            "unexpected search response shape: {body}"
        )))
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

    async fn has_vector(&self, collection: &str, point_id: Uuid) -> Result<bool, VectorStoreError> {
        let resp = self
            .get(&format!("/collections/{collection}/points/{point_id}"))
            .await?;
        match resp.status().as_u16() {
            200 => Ok(true),
            404 => Ok(false),
            code => Err(VectorStoreError::HttpStatus(code, body_text(resp).await)),
        }
    }

    async fn upsert(&self, collection: &str, point: &VectorPoint) -> Result<(), VectorStoreError> {
        let resp = self
            .client
            .put(format!(
                "{}/collections/{}/points?wait=true",
                self.base_url, collection
            ))
            .json(&upsert_body(point))
            .send()
            .await
            .map_err(transport)?;
        check_success(resp).await.map(|_| ())
    }

    async fn search(
        &self,
        collection: &str,
        query_vector: &[f32],
        limit: usize,
        score_threshold: Option<f32>,
    ) -> Result<Vec<ScoredPoint>, VectorStoreError> {
        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/search",
                self.base_url, collection
            ))
            .json(&search_body(query_vector, limit, score_threshold))
            .send()
            .await
            .map_err(transport)?;
        let resp = check_success(resp).await?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| VectorStoreError::Parse(e.to_string()))?;
        parse_search_results(&body)
    }
}

async fn check_success(resp: reqwest::Response) -> Result<reqwest::Response, VectorStoreError> {
    let status = resp.status();
    if status.is_success() {
        Ok(resp)
    } else {
        Err(VectorStoreError::HttpStatus(
            status.as_u16(),
            body_text(resp).await,
        ))
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
    fn search_body_carries_vector_limit_and_omits_threshold_when_none() {
        let body = search_body(&[0.1, 0.2], 5, None);
        assert_eq!(body["vector"].as_array().unwrap().len(), 2);
        assert_eq!(body["limit"], 5);
        assert_eq!(body["with_payload"], false);
        assert!(
            body.get("score_threshold").is_none(),
            "no threshold key when None"
        );
    }

    #[test]
    fn search_body_includes_threshold_when_set() {
        let body = search_body(&[0.1], 5, Some(0.42));
        // f32 → JSON loses exactness; compare with tolerance.
        let t = body["score_threshold"].as_f64().unwrap();
        assert!((t - 0.42).abs() < 1e-6, "threshold ~0.42, got {t}");
    }

    #[test]
    fn parse_search_results_reads_ranked_id_and_score() {
        let id = Uuid::nil();
        let body = json!({
            "result": [
                { "id": id.to_string(), "score": 0.9 },
                { "id": id.to_string(), "score": 0.7 },
            ],
            "status": "ok",
        });
        let got = parse_search_results(&body).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0], ScoredPoint { id, score: 0.9 });
        assert_eq!(got[1].score, 0.7);
    }

    #[test]
    fn parse_search_results_empty_is_ok() {
        let body = json!({ "result": [] });
        assert!(parse_search_results(&body).unwrap().is_empty());
    }

    #[test]
    fn parse_search_results_bad_id_is_a_parse_error() {
        let body = json!({ "result": [ { "id": "not-a-uuid", "score": 0.5 } ] });
        assert!(matches!(
            parse_search_results(&body),
            Err(VectorStoreError::Parse(_))
        ));
    }

    #[test]
    fn parse_search_results_missing_result_key_is_a_parse_error() {
        let body = json!({ "status": "ok" });
        assert!(matches!(
            parse_search_results(&body),
            Err(VectorStoreError::Parse(_))
        ));
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
