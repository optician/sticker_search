//! The `EmbedCaptions` use-case: turn the captions from one
//! `(caption_model, prompt_version)` into vectors and store them in a per-set
//! collection, keyed by the sticker UUID. Mirrors `CaptionStickers`.

use crate::entities::{Caption, DistanceMetric, VectorPayload, VectorPoint};
use crate::error::{EmbedError, EmbedStickerError};
use crate::ports::{CaptionReader, EmbeddingGateway, VectorStore};
use uuid::Uuid;

/// Counts reported at the end of a run. Never aborts on per-sticker failure.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EmbedSummary {
    pub embedded: u32,
    pub skipped: u32,
    pub failed: u32,
}

/// Per-run knobs. The run embeds the captions written by one
/// `(caption_model, prompt_version)`; `force` re-embeds points already present;
/// `limit` caps the run.
#[derive(Debug, Clone, Copy)]
pub struct EmbedRun<'a> {
    pub caption_model: &'a str,
    pub prompt_version: &'a str,
    pub force: bool,
    pub limit: Option<usize>,
}

/// What happened to one caption, reported to the progress hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedEvent {
    /// About to call the embedding model.
    Start,
    Embedded,
    Skipped,
    Failed,
}

/// A progress update emitted before and after each caption in a run.
#[derive(Debug, Clone, Copy)]
pub struct EmbedProgress {
    pub index: usize,
    pub total: usize,
    pub sticker_id: Uuid,
    pub event: EmbedEvent,
}

fn ignore_progress(_: EmbedProgress) {}

/// Qdrant collection names allow `[a-zA-Z0-9_-]`; model tags carry `:` and `.`
/// (`qwen3-vl:32b`). Map every other character to `_` so a set maps to one
/// deterministic, legal collection name.
fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Deterministic collection name for a `(caption_model, prompt_version,
/// embed_model)` set. Public so the query side and CLI resolve the same name.
pub fn collection_name(caption_model: &str, prompt_version: &str, embed_model: &str) -> String {
    format!(
        "stickers__{}__{}__{}",
        sanitize(caption_model),
        sanitize(prompt_version),
        sanitize(embed_model),
    )
}

/// Orchestrates embedding. Generic over the ports (static dispatch).
pub struct EmbedCaptions<G, R, V> {
    gateway: G,
    captions: R,
    store: V,
    report: fn(EmbedProgress),
}

impl<G, R, V> EmbedCaptions<G, R, V>
where
    G: EmbeddingGateway,
    R: CaptionReader,
    V: VectorStore,
{
    pub fn new(gateway: G, captions: R, store: V) -> Self {
        Self {
            gateway,
            captions,
            store,
            report: ignore_progress,
        }
    }

    pub fn on_progress(mut self, report: fn(EmbedProgress)) -> Self {
        self.report = report;
        self
    }

    pub fn gateway(&self) -> &G {
        &self.gateway
    }

    pub fn store(&self) -> &V {
        &self.store
    }

    /// Embed every caption for the selected set. Run-level setup failures
    /// (listing captions, creating the collection) abort and surface as `Err`;
    /// per-sticker failures are logged and counted into the summary.
    pub async fn run(&self, cfg: EmbedRun<'_>) -> Result<EmbedSummary, EmbedError> {
        let captions = self
            .captions
            .list_captions(cfg.caption_model, cfg.prompt_version)?;
        let collection =
            collection_name(cfg.caption_model, cfg.prompt_version, self.gateway.model());
        self.store
            .ensure_collection(&collection, self.gateway.dim(), DistanceMetric::Cosine)
            .await?;

        let total = cfg.limit.unwrap_or(captions.len()).min(captions.len());

        let mut summary = EmbedSummary::default();
        for (i, caption) in captions.iter().take(total).enumerate() {
            let report = |event| {
                (self.report)(EmbedProgress {
                    index: i + 1,
                    total,
                    sticker_id: caption.sticker_id,
                    event,
                });
            };
            report(EmbedEvent::Start);
            match self.embed_one(&collection, caption, cfg).await {
                Ok(Outcome::Embedded) => {
                    summary.embedded += 1;
                    report(EmbedEvent::Embedded);
                }
                Ok(Outcome::Skipped) => {
                    summary.skipped += 1;
                    report(EmbedEvent::Skipped);
                }
                Err(e) => {
                    summary.failed += 1;
                    tracing::warn!(sticker = %caption.sticker_id, error = %e, "embed failed");
                    report(EmbedEvent::Failed);
                }
            }
        }
        Ok(summary)
    }

    async fn embed_one(
        &self,
        collection: &str,
        caption: &Caption,
        cfg: EmbedRun<'_>,
    ) -> Result<Outcome, EmbedStickerError> {
        let id = caption.sticker_id;
        if !cfg.force && self.store.has_vector(collection, id).await? {
            return Ok(Outcome::Skipped);
        }

        let vector = self.gateway.embed(&caption.embed_text()).await?;
        let expected = self.gateway.dim();
        if vector.len() != expected {
            return Err(EmbedStickerError::DimensionMismatch {
                expected,
                got: vector.len(),
            });
        }

        let point = VectorPoint {
            id,
            vector,
            payload: VectorPayload {
                sticker_id: id,
                caption_model: cfg.caption_model.to_string(),
                prompt_version: cfg.prompt_version.to_string(),
                embed_model: self.gateway.model().to_string(),
            },
        };
        self.store.upsert(collection, &point).await?;
        Ok(Outcome::Embedded)
    }
}

/// Outcome of processing a single caption.
enum Outcome {
    Embedded,
    Skipped,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{EmbeddingGatewayError, RepoError, VectorStoreError};
    use std::cell::{Cell, RefCell};
    use std::collections::{HashMap, HashSet};
    use time::OffsetDateTime;

    fn caption(sticker_id: Uuid, model: &str, version: &str, scene: &str) -> Caption {
        Caption {
            sticker_id,
            model: model.into(),
            prompt_version: version.into(),
            scene: scene.into(),
            on_image_text: String::new(),
            tone: "neutral".into(),
            situations: vec![],
            raw: String::new(),
            created_at: OffsetDateTime::UNIX_EPOCH,
        }
    }

    // ---- fakes ----

    struct FakeGateway {
        model: String,
        dim: usize,
        /// When set, `embed` returns a vector of this length instead of `dim`.
        bad_len: Option<usize>,
        fail_on: RefCell<HashSet<String>>,
        calls: Cell<u32>,
    }

    impl FakeGateway {
        fn new(model: &str, dim: usize) -> Self {
            Self {
                model: model.into(),
                dim,
                bad_len: None,
                fail_on: RefCell::new(HashSet::new()),
                calls: Cell::new(0),
            }
        }
        fn fail_on_text(self, text: &str) -> Self {
            self.fail_on.borrow_mut().insert(text.into());
            self
        }
        fn calls(&self) -> u32 {
            self.calls.get()
        }
    }

    impl EmbeddingGateway for FakeGateway {
        fn model(&self) -> &str {
            &self.model
        }
        fn dim(&self) -> usize {
            self.dim
        }
        async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingGatewayError> {
            self.calls.set(self.calls.get() + 1);
            if self.fail_on.borrow().contains(text) {
                return Err(EmbeddingGatewayError::Transport("boom".into()));
            }
            let len = self.bad_len.unwrap_or(self.dim);
            // Encode the text length so a test could distinguish vectors.
            Ok(vec![text.len() as f32; len])
        }
    }

    struct FakeCaptions {
        rows: Vec<Caption>,
    }

    impl CaptionReader for FakeCaptions {
        fn list_captions(
            &self,
            model: &str,
            prompt_version: &str,
        ) -> Result<Vec<Caption>, RepoError> {
            Ok(self
                .rows
                .iter()
                .filter(|c| c.model == model && c.prompt_version == prompt_version)
                .cloned()
                .collect())
        }
    }

    #[derive(Default)]
    struct FakeStore {
        collections: RefCell<HashMap<String, (usize, DistanceMetric)>>,
        points: RefCell<HashMap<(String, Uuid), VectorPoint>>,
        ensure_calls: Cell<u32>,
        fail_upsert: bool,
    }

    impl VectorStore for FakeStore {
        async fn ensure_collection(
            &self,
            collection: &str,
            dim: usize,
            metric: DistanceMetric,
        ) -> Result<(), VectorStoreError> {
            self.ensure_calls.set(self.ensure_calls.get() + 1);
            self.collections
                .borrow_mut()
                .insert(collection.into(), (dim, metric));
            Ok(())
        }
        async fn has_vector(
            &self,
            collection: &str,
            point_id: Uuid,
        ) -> Result<bool, VectorStoreError> {
            Ok(self
                .points
                .borrow()
                .contains_key(&(collection.into(), point_id)))
        }
        async fn upsert(
            &self,
            collection: &str,
            point: &VectorPoint,
        ) -> Result<(), VectorStoreError> {
            if self.fail_upsert {
                return Err(VectorStoreError::Transport("down".into()));
            }
            self.points
                .borrow_mut()
                .insert((collection.into(), point.id), point.clone());
            Ok(())
        }
        async fn search(
            &self,
            _collection: &str,
            _query_vector: &[f32],
            _limit: usize,
            _score_threshold: Option<f32>,
        ) -> Result<Vec<crate::entities::ScoredPoint>, VectorStoreError> {
            // The embedder never searches; unused in these tests.
            Ok(vec![])
        }
    }

    fn run_cfg<'a>() -> EmbedRun<'a> {
        EmbedRun {
            caption_model: "qwen",
            prompt_version: "v1",
            force: false,
            limit: None,
        }
    }

    fn app(
        gw: FakeGateway,
        rows: Vec<Caption>,
        store: FakeStore,
    ) -> EmbedCaptions<FakeGateway, FakeCaptions, FakeStore> {
        EmbedCaptions::new(gw, FakeCaptions { rows }, store)
    }

    // ---- tests ----

    #[test]
    fn collection_name_is_deterministic_and_sanitized() {
        assert_eq!(
            collection_name("qwen3-vl:32b", "v1", "bge-m3"),
            "stickers__qwen3-vl_32b__v1__bge-m3",
        );
    }

    #[tokio::test]
    async fn fresh_caption_is_embedded_and_stored_with_provenance() {
        let id = Uuid::new_v4();
        let store = FakeStore::default();
        let app = app(
            FakeGateway::new("bge-m3", 4),
            vec![caption(id, "qwen", "v1", "a cat")],
            store,
        );

        let summary = app.run(run_cfg()).await.unwrap();

        assert_eq!(
            summary,
            EmbedSummary {
                embedded: 1,
                ..Default::default()
            }
        );
        let coll = collection_name("qwen", "v1", "bge-m3");
        let points = app.store().points.borrow();
        let p = points.get(&(coll.clone(), id)).expect("point stored");
        assert_eq!(p.vector.len(), 4, "vector has the model's dim");
        assert_eq!(
            p.payload,
            VectorPayload {
                sticker_id: id,
                caption_model: "qwen".into(),
                prompt_version: "v1".into(),
                embed_model: "bge-m3".into(),
            },
        );
        // Collection created up front with the gateway's dim + cosine.
        assert_eq!(
            app.store().collections.borrow().get(&coll),
            Some(&(4usize, DistanceMetric::Cosine)),
        );
    }

    #[tokio::test]
    async fn existing_vector_is_skipped_without_embedding() {
        let id = Uuid::new_v4();
        let coll = collection_name("qwen", "v1", "bge-m3");
        let store = FakeStore::default();
        store.points.borrow_mut().insert(
            (coll, id),
            VectorPoint {
                id,
                vector: vec![0.0; 4],
                payload: VectorPayload {
                    sticker_id: id,
                    caption_model: "qwen".into(),
                    prompt_version: "v1".into(),
                    embed_model: "bge-m3".into(),
                },
            },
        );
        let app = app(
            FakeGateway::new("bge-m3", 4),
            vec![caption(id, "qwen", "v1", "a cat")],
            store,
        );

        let summary = app.run(run_cfg()).await.unwrap();

        assert_eq!(
            summary,
            EmbedSummary {
                skipped: 1,
                ..Default::default()
            }
        );
        assert_eq!(app.gateway().calls(), 0, "skip avoids the model call");
    }

    #[tokio::test]
    async fn force_re_embeds_existing_vector() {
        let id = Uuid::new_v4();
        let coll = collection_name("qwen", "v1", "bge-m3");
        let store = FakeStore::default();
        store.points.borrow_mut().insert(
            (coll, id),
            VectorPoint {
                id,
                vector: vec![0.0; 4],
                payload: VectorPayload {
                    sticker_id: id,
                    caption_model: "qwen".into(),
                    prompt_version: "v1".into(),
                    embed_model: "bge-m3".into(),
                },
            },
        );
        let app = app(
            FakeGateway::new("bge-m3", 4),
            vec![caption(id, "qwen", "v1", "a cat")],
            store,
        );

        let cfg = EmbedRun {
            force: true,
            ..run_cfg()
        };
        let summary = app.run(cfg).await.unwrap();

        assert_eq!(
            summary,
            EmbedSummary {
                embedded: 1,
                ..Default::default()
            }
        );
        assert_eq!(app.gateway().calls(), 1);
    }

    #[tokio::test]
    async fn only_the_selected_caption_set_is_embedded() {
        let a = caption(Uuid::new_v4(), "qwen", "v1", "match");
        let other_model = caption(Uuid::new_v4(), "llava", "v1", "skip");
        let other_ver = caption(Uuid::new_v4(), "qwen", "v2", "skip");
        let app = app(
            FakeGateway::new("bge-m3", 4),
            vec![a, other_model, other_ver],
            FakeStore::default(),
        );

        let summary = app.run(run_cfg()).await.unwrap();

        assert_eq!(
            summary,
            EmbedSummary {
                embedded: 1,
                ..Default::default()
            }
        );
    }

    #[tokio::test]
    async fn one_failing_embed_does_not_abort_the_run() {
        let good = caption(Uuid::new_v4(), "qwen", "v1", "good");
        let bad = caption(Uuid::new_v4(), "qwen", "v1", "bad");
        let gw = FakeGateway::new("bge-m3", 4).fail_on_text(&bad.embed_text());
        let app = app(gw, vec![good, bad], FakeStore::default());

        let summary = app.run(run_cfg()).await.unwrap();

        assert_eq!(
            summary,
            EmbedSummary {
                embedded: 1,
                failed: 1,
                ..Default::default()
            }
        );
        assert_eq!(app.store().points.borrow().len(), 1);
    }

    #[tokio::test]
    async fn dimension_mismatch_is_a_per_sticker_failure() {
        let id = Uuid::new_v4();
        let mut gw = FakeGateway::new("bge-m3", 4);
        gw.bad_len = Some(3); // model returns 3 dims, collection expects 4
        let app = app(
            gw,
            vec![caption(id, "qwen", "v1", "a cat")],
            FakeStore::default(),
        );

        let summary = app.run(run_cfg()).await.unwrap();

        assert_eq!(
            summary,
            EmbedSummary {
                failed: 1,
                ..Default::default()
            }
        );
        assert!(app.store().points.borrow().is_empty(), "nothing stored");
    }

    #[tokio::test]
    async fn limit_caps_the_run() {
        let rows = vec![
            caption(Uuid::new_v4(), "qwen", "v1", "one"),
            caption(Uuid::new_v4(), "qwen", "v1", "two"),
            caption(Uuid::new_v4(), "qwen", "v1", "three"),
        ];
        let app = app(FakeGateway::new("bge-m3", 4), rows, FakeStore::default());

        let cfg = EmbedRun {
            limit: Some(1),
            ..run_cfg()
        };
        let summary = app.run(cfg).await.unwrap();

        assert_eq!(
            summary,
            EmbedSummary {
                embedded: 1,
                ..Default::default()
            }
        );
    }

    #[tokio::test]
    async fn upsert_failure_aborts_run_setup_is_fine() {
        // A store whose upsert fails turns each sticker into a counted failure,
        // not a run-level abort.
        let store = FakeStore {
            fail_upsert: true,
            ..Default::default()
        };
        let app = app(
            FakeGateway::new("bge-m3", 4),
            vec![caption(Uuid::new_v4(), "qwen", "v1", "x")],
            store,
        );

        let summary = app.run(run_cfg()).await.unwrap();

        assert_eq!(
            summary,
            EmbedSummary {
                failed: 1,
                ..Default::default()
            }
        );
    }
}
