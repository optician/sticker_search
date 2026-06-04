//! The `SearchStickers` use-case: a text query becomes ranked sticker hits.
//!
//! The read counterpart to `EmbedCaptions`. It embeds the query with the *same*
//! `EmbeddingGateway` the captions were embedded with (the invariant that makes
//! the vector space shared), searches the matching collection, and resolves each
//! ranked point id (a sticker UUID) back to its sticker and the caption that
//! produced its vector. Generic over the ports (static dispatch).

use crate::embed::collection_name;
use crate::entities::{ScoredPoint, SearchHit};
use crate::error::SearchError;
use crate::ports::{CaptionLookup, EmbeddingGateway, StickerRepository, VectorStore};

/// One query's inputs. `caption_model` + `prompt_version` select which caption
/// set (and thus which collection) to search; they must name a set that was
/// embedded with this use-case's gateway.
#[derive(Debug, Clone, Copy)]
pub struct SearchQuery<'a> {
    pub text: &'a str,
    pub caption_model: &'a str,
    pub prompt_version: &'a str,
    pub limit: usize,
    pub min_score: Option<f32>,
}

/// Orchestrates a query. Holds the gateway, the vector store, and the two read
/// ports the resolution needs.
pub struct SearchStickers<E, V, S, C> {
    gateway: E,
    store: V,
    stickers: S,
    captions: C,
}

impl<E, V, S, C> SearchStickers<E, V, S, C>
where
    E: EmbeddingGateway,
    V: VectorStore,
    S: StickerRepository,
    C: CaptionLookup,
{
    pub fn new(gateway: E, store: V, stickers: S, captions: C) -> Self {
        Self { gateway, store, stickers, captions }
    }

    pub fn gateway(&self) -> &E {
        &self.gateway
    }

    pub fn store(&self) -> &V {
        &self.store
    }

    /// Embed the query, search the collection for its caption set, and resolve
    /// hits to stickers + captions in rank order. A hit whose sticker or caption
    /// row is missing (index/DB drift) is skipped with a warning rather than
    /// failing the whole query.
    pub async fn search(&self, q: SearchQuery<'_>) -> Result<Vec<SearchHit>, SearchError> {
        let collection =
            collection_name(q.caption_model, q.prompt_version, self.gateway.model());
        let vector = self.gateway.embed(q.text).await?;
        let hits = self.store.search(&collection, &vector, q.limit, q.min_score).await?;

        let mut out = Vec::with_capacity(hits.len());
        for ScoredPoint { id, score } in hits {
            let Some(sticker) = self.stickers.find_sticker_by_id(id)? else {
                tracing::warn!(sticker = %id, "search hit has no sticker row; skipping");
                continue;
            };
            let Some(caption) =
                self.captions.find_caption(id, q.caption_model, q.prompt_version)?
            else {
                tracing::warn!(sticker = %id, "search hit has no caption row; skipping");
                continue;
            };
            out.push(SearchHit { score, sticker, caption });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::{Caption, Pack, Sticker, StickerFormat};
    use crate::error::{EmbeddingGatewayError, RepoError, VectorStoreError};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use time::OffsetDateTime;
    use uuid::Uuid;

    // ---- entity builders ----

    fn sticker(id: Uuid) -> Sticker {
        Sticker {
            id,
            pack_id: Uuid::nil(),
            file_unique_id: format!("u-{id}"),
            file_id: format!("f-{id}"),
            emoji: Some("🐔".into()),
            format: StickerFormat::Static,
            width: 512,
            height: 512,
            position: 0,
            image_path: format!("packA/{id}.webp"),
            created_at: OffsetDateTime::UNIX_EPOCH,
        }
    }

    fn caption(id: Uuid, model: &str, version: &str, scene: &str) -> Caption {
        Caption {
            sticker_id: id,
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
        /// Text the gateway should fail on (simulates a model/transport error).
        fail: Option<String>,
    }

    impl FakeGateway {
        fn new(model: &str) -> Self {
            Self { model: model.into(), dim: 4, fail: None }
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
            if self.fail.as_deref() == Some(text) {
                return Err(EmbeddingGatewayError::Transport("boom".into()));
            }
            Ok(vec![text.len() as f32; self.dim])
        }
    }

    /// A recorded search call: `(collection, query_vector, limit, threshold)`.
    type SearchCall = (String, Vec<f32>, usize, Option<f32>);

    /// Records the last search call and returns a preset ranked list.
    #[derive(Default)]
    struct FakeStore {
        ranked: Vec<ScoredPoint>,
        last: RefCell<Option<SearchCall>>,
        fail: bool,
    }

    impl VectorStore for FakeStore {
        async fn ensure_collection(
            &self,
            _collection: &str,
            _dim: usize,
            _metric: crate::entities::DistanceMetric,
        ) -> Result<(), VectorStoreError> {
            Ok(())
        }
        async fn has_vector(
            &self,
            _collection: &str,
            _point_id: Uuid,
        ) -> Result<bool, VectorStoreError> {
            Ok(false)
        }
        async fn upsert(
            &self,
            _collection: &str,
            _point: &crate::entities::VectorPoint,
        ) -> Result<(), VectorStoreError> {
            Ok(())
        }
        async fn search(
            &self,
            collection: &str,
            query_vector: &[f32],
            limit: usize,
            score_threshold: Option<f32>,
        ) -> Result<Vec<ScoredPoint>, VectorStoreError> {
            if self.fail {
                return Err(VectorStoreError::Transport("down".into()));
            }
            *self.last.borrow_mut() =
                Some((collection.into(), query_vector.to_vec(), limit, score_threshold));
            Ok(self.ranked.clone())
        }
    }

    /// `find_sticker_by_id` is the only method the use-case calls; the rest are
    /// trivial so the fake can satisfy the whole port without panicking.
    #[derive(Default)]
    struct FakeStickers {
        by_id: HashMap<Uuid, Sticker>,
    }

    impl StickerRepository for FakeStickers {
        fn find_pack_by_name(&self, _name: &str) -> Result<Option<Pack>, RepoError> {
            Ok(None)
        }
        fn upsert_pack(&self, _pack: &Pack) -> Result<(), RepoError> {
            Ok(())
        }
        fn find_sticker_by_unique_id(
            &self,
            _file_unique_id: &str,
        ) -> Result<Option<Sticker>, RepoError> {
            Ok(None)
        }
        fn find_sticker_by_id(&self, id: Uuid) -> Result<Option<Sticker>, RepoError> {
            Ok(self.by_id.get(&id).cloned())
        }
        fn upsert_sticker(&self, _sticker: &Sticker) -> Result<(), RepoError> {
            Ok(())
        }
        fn list_stickers(&self, _pack: Option<&str>) -> Result<Vec<Sticker>, RepoError> {
            Ok(vec![])
        }
    }

    #[derive(Default)]
    struct FakeCaptions {
        by_id: HashMap<Uuid, Caption>,
    }

    impl CaptionLookup for FakeCaptions {
        fn find_caption(
            &self,
            sticker_id: Uuid,
            _model: &str,
            _prompt_version: &str,
        ) -> Result<Option<Caption>, RepoError> {
            Ok(self.by_id.get(&sticker_id).cloned())
        }
    }

    fn app(
        gw: FakeGateway,
        store: FakeStore,
        stickers: FakeStickers,
        captions: FakeCaptions,
    ) -> SearchStickers<FakeGateway, FakeStore, FakeStickers, FakeCaptions> {
        SearchStickers::new(gw, store, stickers, captions)
    }

    fn query(text: &str) -> SearchQuery<'_> {
        SearchQuery {
            text,
            caption_model: "qwen",
            prompt_version: "v1",
            limit: 10,
            min_score: None,
        }
    }

    // ---- tests ----

    #[tokio::test]
    async fn resolves_hits_in_rank_order_with_sticker_and_caption() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let store = FakeStore {
            ranked: vec![
                ScoredPoint { id: a, score: 0.9 },
                ScoredPoint { id: b, score: 0.7 },
            ],
            ..Default::default()
        };
        let stickers = FakeStickers {
            by_id: HashMap::from([(a, sticker(a)), (b, sticker(b))]),
        };
        let captions = FakeCaptions {
            by_id: HashMap::from([
                (a, caption(a, "qwen", "v1", "a chicken")),
                (b, caption(b, "qwen", "v1", "a dog")),
            ]),
        };
        let app = app(FakeGateway::new("bge-m3"), store, stickers, captions);

        let hits = app.search(query("bird")).await.unwrap();

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].sticker.id, a, "best-first order preserved");
        assert_eq!(hits[0].score, 0.9);
        assert_eq!(hits[0].caption.scene, "a chicken");
        assert_eq!(hits[1].sticker.id, b);
        assert_eq!(hits[1].score, 0.7);
    }

    #[tokio::test]
    async fn searches_the_collection_for_the_caption_set_and_embed_model() {
        let app = app(
            FakeGateway::new("bge-m3"),
            FakeStore::default(),
            FakeStickers::default(),
            FakeCaptions::default(),
        );

        app.search(query("hello")).await.unwrap();

        let last = app.store().last.borrow();
        let (collection, vector, limit, threshold) = last.as_ref().unwrap();
        // Third component is the gateway's model, mirroring the embedder.
        assert_eq!(collection, &collection_name("qwen", "v1", "bge-m3"));
        assert_eq!(vector.len(), 4, "query embedded by the same gateway");
        assert_eq!(*limit, 10);
        assert_eq!(*threshold, None);
    }

    #[tokio::test]
    async fn passes_limit_and_min_score_through_to_the_store() {
        let app = app(
            FakeGateway::new("bge-m3"),
            FakeStore::default(),
            FakeStickers::default(),
            FakeCaptions::default(),
        );

        let q = SearchQuery { limit: 3, min_score: Some(0.42), ..query("x") };
        app.search(q).await.unwrap();

        let last = app.store().last.borrow();
        let (_, _, limit, threshold) = last.as_ref().unwrap();
        assert_eq!(*limit, 3);
        assert_eq!(*threshold, Some(0.42));
    }

    #[tokio::test]
    async fn hit_missing_its_sticker_row_is_skipped_not_fatal() {
        let present = Uuid::new_v4();
        let orphan = Uuid::new_v4();
        let store = FakeStore {
            ranked: vec![
                ScoredPoint { id: orphan, score: 0.9 },
                ScoredPoint { id: present, score: 0.5 },
            ],
            ..Default::default()
        };
        let stickers = FakeStickers { by_id: HashMap::from([(present, sticker(present))]) };
        let captions = FakeCaptions {
            by_id: HashMap::from([(present, caption(present, "qwen", "v1", "kept"))]),
        };
        let app = app(FakeGateway::new("bge-m3"), store, stickers, captions);

        let hits = app.search(query("x")).await.unwrap();

        assert_eq!(hits.len(), 1, "orphan hit dropped, query still succeeds");
        assert_eq!(hits[0].sticker.id, present);
    }

    #[tokio::test]
    async fn hit_missing_its_caption_row_is_skipped_not_fatal() {
        let id = Uuid::new_v4();
        let store = FakeStore {
            ranked: vec![ScoredPoint { id, score: 0.9 }],
            ..Default::default()
        };
        // Sticker exists, but no caption for this (model, version).
        let stickers = FakeStickers { by_id: HashMap::from([(id, sticker(id))]) };
        let app = app(
            FakeGateway::new("bge-m3"),
            store,
            stickers,
            FakeCaptions::default(),
        );

        let hits = app.search(query("x")).await.unwrap();

        assert!(hits.is_empty(), "captionless hit dropped");
    }

    #[tokio::test]
    async fn embedding_failure_aborts_the_query() {
        let gw = FakeGateway { fail: Some("bad".into()), ..FakeGateway::new("bge-m3") };
        let app = app(gw, FakeStore::default(), FakeStickers::default(), FakeCaptions::default());

        let err = app.search(query("bad")).await.unwrap_err();

        assert!(matches!(err, SearchError::Gateway(_)));
    }

    #[tokio::test]
    async fn store_failure_aborts_the_query() {
        let store = FakeStore { fail: true, ..Default::default() };
        let app = app(FakeGateway::new("bge-m3"), store, FakeStickers::default(), FakeCaptions::default());

        let err = app.search(query("x")).await.unwrap_err();

        assert!(matches!(err, SearchError::Store(_)));
    }
}
