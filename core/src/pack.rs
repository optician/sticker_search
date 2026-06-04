//! Pack-name normalization and the `PackStatus` use-case.
//!
//! `PackStatus` answers "how far along is this pack?" for the bot's `/add`
//! status replies. The queue (`PackRequests`) only records *that* a pack was
//! asked for; the *stage* is derived here from the pipeline's own data — stickers
//! in the metadata store, captions for the asked `(model, prompt)`, and vectors
//! in the collection — so the offline batch never has to write progress back.

use crate::embed::collection_name;
use crate::entities::{PackReport, PackStage};
use crate::error::PackStatusError;
use crate::ports::{CaptionRepository, StickerRepository, VectorStore};

/// Extract the bare pack name from a share link or `tg://` URL, accepting:
/// `crazy_klutzy`, `https://t.me/addstickers/crazy_klutzy`,
/// `t.me/addstickers/crazy_klutzy`, `tg://addstickers?set=crazy_klutzy`.
/// Anything that isn't a recognized link is returned trimmed as-is.
pub fn normalize_pack_name(raw: &str) -> String {
    let s = raw.trim();
    if s.contains("addstickers") {
        if let Some(rest) = s.rsplit_once("addstickers/").map(|(_, r)| r) {
            // https://t.me/addstickers/<name>[/?#...]
            return rest
                .split(['/', '?', '#'])
                .next()
                .unwrap_or(rest)
                .to_string();
        }
        if let Some(rest) = s.split_once("set=").map(|(_, r)| r) {
            // tg://addstickers?set=<name>[&#...]
            return rest.split(['&', '#']).next().unwrap_or(rest).to_string();
        }
    }
    s.to_string()
}

/// Derives a [`PackReport`] for one pack against a fixed
/// `(caption_model, prompt_version, embed_model)` set. Generic over the read
/// ports (static dispatch), mirroring `SearchStickers`.
pub struct PackStatus<S, C, V> {
    stickers: S,
    captions: C,
    store: V,
}

impl<S, C, V> PackStatus<S, C, V>
where
    S: StickerRepository,
    C: CaptionRepository,
    V: VectorStore,
{
    pub fn new(stickers: S, captions: C, store: V) -> Self {
        Self {
            stickers,
            captions,
            store,
        }
    }

    /// Report `name`'s stage. The stage is the *lowest* a sticker has reached:
    /// the pack is `Ready` only when every sticker is embedded, `Captioned` only
    /// when every sticker is captioned, and so on. A pack with no stored stickers
    /// is `Queued` (not scraped yet, or scrape failed). The vector store is probed
    /// only once captioning is complete, so a half-captioned pack costs no Qdrant
    /// calls.
    pub async fn report(
        &self,
        name: &str,
        caption_model: &str,
        prompt_version: &str,
        embed_model: &str,
    ) -> Result<PackReport, PackStatusError> {
        let stickers = self.stickers.list_stickers(Some(name))?;
        let sticker_count = stickers.len();
        if sticker_count == 0 {
            return Ok(report(name, PackStage::Queued, 0, 0, 0));
        }

        let mut captioned = 0;
        for s in &stickers {
            if self
                .captions
                .caption_exists(s.id, caption_model, prompt_version)?
            {
                captioned += 1;
            }
        }
        if captioned < sticker_count {
            return Ok(report(
                name,
                PackStage::Scraped,
                sticker_count,
                captioned,
                0,
            ));
        }

        let collection = collection_name(caption_model, prompt_version, embed_model);
        let mut embedded = 0;
        for s in &stickers {
            if self.store.has_vector(&collection, s.id).await? {
                embedded += 1;
            }
        }
        let stage = if embedded == sticker_count {
            PackStage::Ready
        } else {
            PackStage::Captioned
        };
        Ok(report(name, stage, sticker_count, captioned, embedded))
    }
}

fn report(
    name: &str,
    stage: PackStage,
    sticker_count: usize,
    captioned_count: usize,
    embedded_count: usize,
) -> PackReport {
    PackReport {
        name: name.to_string(),
        stage,
        sticker_count,
        captioned_count,
        embedded_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::{Pack, ScoredPoint, Sticker, StickerFormat, VectorPoint};
    use crate::error::{RepoError, VectorStoreError};
    use rstest::rstest;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use time::OffsetDateTime;
    use uuid::Uuid;

    #[rstest]
    #[case("crazy_klutzy", "crazy_klutzy")]
    #[case("  crazy_klutzy  ", "crazy_klutzy")]
    #[case("https://t.me/addstickers/crazy_klutzy", "crazy_klutzy")]
    #[case("t.me/addstickers/crazy_klutzy", "crazy_klutzy")]
    #[case("https://t.me/addstickers/crazy_klutzy?foo=bar", "crazy_klutzy")]
    #[case("tg://addstickers?set=crazy_klutzy&mode=x", "crazy_klutzy")]
    fn normalizes_links_and_ids(#[case] input: &str, #[case] expected: &str) {
        assert_eq!(normalize_pack_name(input), expected);
    }

    // ---- fakes ----

    fn sticker(n: u32) -> Sticker {
        Sticker {
            id: Uuid::new_v4(),
            pack_id: Uuid::nil(),
            file_unique_id: format!("u{n}"),
            file_id: format!("f{n}"),
            emoji: None,
            format: StickerFormat::Static,
            width: 512,
            height: 512,
            position: n,
            image_path: format!("packA/{n}.webp"),
            created_at: OffsetDateTime::UNIX_EPOCH,
        }
    }

    /// Stickers for one pack name; everything else is unused boilerplate.
    #[derive(Default)]
    struct FakeStickers {
        by_pack: std::collections::HashMap<String, Vec<Sticker>>,
    }

    impl StickerRepository for FakeStickers {
        fn find_pack_by_name(&self, _name: &str) -> Result<Option<Pack>, RepoError> {
            Ok(None)
        }
        fn upsert_pack(&self, _pack: &Pack) -> Result<(), RepoError> {
            Ok(())
        }
        fn find_sticker_by_unique_id(&self, _u: &str) -> Result<Option<Sticker>, RepoError> {
            Ok(None)
        }
        fn find_sticker_by_id(&self, _id: Uuid) -> Result<Option<Sticker>, RepoError> {
            Ok(None)
        }
        fn upsert_sticker(&self, _s: &Sticker) -> Result<(), RepoError> {
            Ok(())
        }
        fn list_stickers(&self, pack: Option<&str>) -> Result<Vec<Sticker>, RepoError> {
            Ok(pack
                .and_then(|p| self.by_pack.get(p))
                .cloned()
                .unwrap_or_default())
        }
    }

    /// Set of sticker ids that count as captioned for the asked (model, version).
    struct FakeCaptions {
        captioned: HashSet<Uuid>,
    }

    impl CaptionRepository for FakeCaptions {
        fn caption_exists(
            &self,
            sticker_id: Uuid,
            _model: &str,
            _prompt_version: &str,
        ) -> Result<bool, RepoError> {
            Ok(self.captioned.contains(&sticker_id))
        }
        fn upsert_caption(&self, _c: &crate::entities::Caption) -> Result<(), RepoError> {
            Ok(())
        }
        fn find_prompt(&self, _v: &str) -> Result<Option<crate::entities::Prompt>, RepoError> {
            Ok(None)
        }
        fn upsert_prompt(&self, _p: &crate::entities::Prompt) -> Result<(), RepoError> {
            Ok(())
        }
    }

    /// Set of sticker ids present in the collection, and a flag to record that
    /// the store was queried at all (to assert we skip it when not captioned).
    #[derive(Default)]
    struct FakeStore {
        embedded: HashSet<Uuid>,
        probed: RefCell<bool>,
    }

    impl VectorStore for FakeStore {
        async fn ensure_collection(
            &self,
            _c: &str,
            _d: usize,
            _m: crate::entities::DistanceMetric,
        ) -> Result<(), VectorStoreError> {
            Ok(())
        }
        async fn has_vector(&self, _c: &str, id: Uuid) -> Result<bool, VectorStoreError> {
            *self.probed.borrow_mut() = true;
            Ok(self.embedded.contains(&id))
        }
        async fn upsert(&self, _c: &str, _p: &VectorPoint) -> Result<(), VectorStoreError> {
            Ok(())
        }
        async fn search(
            &self,
            _c: &str,
            _v: &[f32],
            _l: usize,
            _t: Option<f32>,
        ) -> Result<Vec<ScoredPoint>, VectorStoreError> {
            Ok(vec![])
        }
    }

    fn app(
        stickers: Vec<Sticker>,
        captioned: HashSet<Uuid>,
        embedded: HashSet<Uuid>,
    ) -> PackStatus<FakeStickers, FakeCaptions, FakeStore> {
        let mut by_pack = std::collections::HashMap::new();
        if !stickers.is_empty() {
            by_pack.insert("packA".to_string(), stickers);
        }
        PackStatus::new(
            FakeStickers { by_pack },
            FakeCaptions { captioned },
            FakeStore {
                embedded,
                ..Default::default()
            },
        )
    }

    async fn report_for(app: &PackStatus<FakeStickers, FakeCaptions, FakeStore>) -> PackReport {
        app.report("packA", "qwen", "v1", "bge-m3").await.unwrap()
    }

    #[tokio::test]
    async fn no_stickers_is_queued() {
        let r = report_for(&app(vec![], HashSet::new(), HashSet::new())).await;
        assert_eq!(r.stage, PackStage::Queued);
        assert_eq!(r.sticker_count, 0);
    }

    #[tokio::test]
    async fn partially_captioned_is_scraped_and_skips_the_vector_store() {
        let s = vec![sticker(0), sticker(1)];
        let captioned = HashSet::from([s[0].id]);
        let app = app(s, captioned, HashSet::new());

        let r = report_for(&app).await;
        assert_eq!(r.stage, PackStage::Scraped);
        assert_eq!(r.captioned_count, 1);
        assert_eq!(r.sticker_count, 2);
        assert!(
            !*app.store.probed.borrow(),
            "vector store not probed until captioning is done"
        );
    }

    #[tokio::test]
    async fn fully_captioned_not_yet_embedded_is_captioned() {
        let s = vec![sticker(0), sticker(1)];
        let captioned = HashSet::from([s[0].id, s[1].id]);
        let embedded = HashSet::from([s[0].id]); // only one embedded
        let r = report_for(&app(s, captioned, embedded)).await;
        assert_eq!(r.stage, PackStage::Captioned);
        assert_eq!(r.captioned_count, 2);
        assert_eq!(r.embedded_count, 1);
    }

    #[tokio::test]
    async fn all_embedded_is_ready() {
        let s = vec![sticker(0), sticker(1)];
        let ids = HashSet::from([s[0].id, s[1].id]);
        let r = report_for(&app(s, ids.clone(), ids)).await;
        assert_eq!(r.stage, PackStage::Ready);
        assert_eq!(r.embedded_count, 2);
    }
}
