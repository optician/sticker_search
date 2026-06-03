//! The `ScrapePacks` use-case: fetch sticker sets and persist their metadata +
//! thumbnail images through the ports.

use crate::entities::{Pack, RemoteSticker, Sticker, StickerFormat};
use crate::error::ScrapeError;
use crate::ports::{ImageStore, StickerRepository, TelegramGateway};
use time::OffsetDateTime;
use uuid::Uuid;

/// Counts reported at the end of a run. Never aborts on partial failure.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ScrapeSummary {
    pub packs_ok: u32,
    pub packs_failed: u32,
    pub downloaded: u32,
    pub skipped_existing: u32,
    pub failed: u32,
}

/// Orchestrates scraping. Generic over the three ports (static dispatch).
/// `now` is an injectable clock so tests are deterministic.
pub struct ScrapePacks<G, R, S> {
    gateway: G,
    repo: R,
    images: S,
    now: fn() -> OffsetDateTime,
}

impl<G, R, S> ScrapePacks<G, R, S>
where
    G: TelegramGateway,
    R: StickerRepository,
    S: ImageStore,
{
    pub fn new(gateway: G, repo: R, images: S) -> Self {
        Self::with_clock(gateway, repo, images, OffsetDateTime::now_utc)
    }

    pub fn with_clock(gateway: G, repo: R, images: S, now: fn() -> OffsetDateTime) -> Self {
        Self { gateway, repo, images, now }
    }

    pub fn gateway(&self) -> &G {
        &self.gateway
    }

    pub fn repo(&self) -> &R {
        &self.repo
    }

    pub fn images(&self) -> &S {
        &self.images
    }

    /// Scrape each named pack. Aggregates outcomes into `ScrapeSummary` rather
    /// than short-circuiting.
    pub async fn run(&self, pack_names: &[String]) -> ScrapeSummary {
        let mut summary = ScrapeSummary::default();
        for name in pack_names {
            match self.scrape_pack(name, &mut summary).await {
                Ok(()) => summary.packs_ok += 1,
                Err(e) => {
                    summary.packs_failed += 1;
                    tracing::warn!(pack = %name, error = %e, "pack failed");
                }
            }
        }
        summary
    }

    /// A pack "fails" only if its set can't be fetched or its row can't be
    /// written. A failing individual sticker is counted, not propagated.
    async fn scrape_pack(
        &self,
        name: &str,
        summary: &mut ScrapeSummary,
    ) -> Result<(), ScrapeError> {
        let set = self.gateway.get_sticker_set(name).await?;
        let pack_id = match self.repo.find_pack_by_name(&set.name)? {
            Some(p) => p.id,
            None => Uuid::new_v4(),
        };
        self.repo.upsert_pack(&Pack {
            id: pack_id,
            name: set.name.clone(),
            title: set.title.clone(),
            fetched_at: (self.now)(),
        })?;

        for (position, remote) in set.stickers.iter().enumerate() {
            match self.scrape_sticker(&set.name, pack_id, position as u32, remote).await {
                Ok(Outcome::Downloaded) => summary.downloaded += 1,
                Ok(Outcome::Skipped) => summary.skipped_existing += 1,
                Err(e) => {
                    summary.failed += 1;
                    tracing::warn!(
                        pack = %name,
                        sticker = %remote.file_unique_id,
                        error = %e,
                        "sticker failed"
                    );
                }
            }
        }
        Ok(())
    }

    async fn scrape_sticker(
        &self,
        pack_name: &str,
        pack_id: Uuid,
        position: u32,
        remote: &RemoteSticker,
    ) -> Result<Outcome, ScrapeError> {
        let existing = self.repo.find_sticker_by_unique_id(&remote.file_unique_id)?;
        // Reuse the UUID and creation time of a known sticker; mint fresh otherwise.
        let id = existing.as_ref().map(|s| s.id).unwrap_or_else(Uuid::new_v4);
        let created_at = existing
            .as_ref()
            .map(|s| s.created_at)
            .unwrap_or_else(|| (self.now)());

        // Skip the download only when we already have both the row and the file.
        let (image_path, outcome) = match existing
            .as_ref()
            .filter(|ex| self.images.exists(pack_name, file_name_of(&ex.image_path)))
        {
            Some(ex) => (ex.image_path.clone(), Outcome::Skipped),
            None => {
                let download_id =
                    remote.thumb_file_id.as_deref().unwrap_or(&remote.file_id);
                let file = self.gateway.download_file(download_id).await?;
                let file_name = format!("{id}.{}", file.ext);
                let path = self.images.save(pack_name, &file_name, &file.bytes)?;
                (path, Outcome::Downloaded)
            }
        };

        self.repo.upsert_sticker(&Sticker {
            id,
            pack_id,
            file_unique_id: remote.file_unique_id.clone(),
            file_id: remote.file_id.clone(),
            emoji: remote.emoji.clone(),
            format: StickerFormat::from_flags(remote.is_animated, remote.is_video),
            width: remote.width,
            height: remote.height,
            position,
            image_path,
            created_at,
        })?;
        Ok(outcome)
    }
}

/// Outcome of processing a single sticker.
enum Outcome {
    Downloaded,
    Skipped,
}

/// Last path segment of a stored `image_path` (the `<uuid>.<ext>` file name).
fn file_name_of(image_path: &str) -> &str {
    image_path.rsplit('/').next().unwrap_or(image_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::{FileData, Pack, RemoteSticker, RemoteStickerSet, Sticker, StickerFormat};
    use crate::error::{GatewayError, RepoError, StoreError};
    use std::cell::{Cell, RefCell};
    use std::collections::{HashMap, HashSet};
    use uuid::Uuid;

    fn fixed_now() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH
    }

    fn remote(uid: &str, fid: &str) -> RemoteSticker {
        RemoteSticker {
            file_unique_id: uid.into(),
            file_id: fid.into(),
            thumb_file_id: Some(format!("{fid}_thumb")),
            emoji: Some("😀".into()),
            is_animated: false,
            is_video: false,
            width: 512,
            height: 512,
        }
    }

    fn set_with(name: &str, stickers: Vec<RemoteSticker>) -> RemoteStickerSet {
        RemoteStickerSet { name: name.into(), title: "Title".into(), stickers }
    }

    // ---- fakes ----

    struct FakeGateway {
        sets: HashMap<String, RemoteStickerSet>,
        fail_downloads: HashSet<String>,
        download_calls: Cell<u32>,
    }

    impl FakeGateway {
        fn new(sets: HashMap<String, RemoteStickerSet>) -> Self {
            Self { sets, fail_downloads: HashSet::new(), download_calls: Cell::new(0) }
        }
        fn fail_download(mut self, file_id: &str) -> Self {
            self.fail_downloads.insert(file_id.into());
            self
        }
        fn download_calls(&self) -> u32 {
            self.download_calls.get()
        }
    }

    impl TelegramGateway for FakeGateway {
        async fn get_sticker_set(&self, name: &str) -> Result<RemoteStickerSet, GatewayError> {
            self.sets
                .get(name)
                .cloned()
                .ok_or_else(|| GatewayError::NotFound(name.into()))
        }
        async fn download_file(&self, file_id: &str) -> Result<FileData, GatewayError> {
            self.download_calls.set(self.download_calls.get() + 1);
            if self.fail_downloads.contains(file_id) {
                return Err(GatewayError::Transport("boom".into()));
            }
            Ok(FileData { bytes: vec![1, 2, 3], ext: "webp".into() })
        }
    }

    #[derive(Default)]
    struct FakeRepo {
        packs: RefCell<Vec<Pack>>,
        stickers: RefCell<Vec<Sticker>>,
    }

    impl StickerRepository for FakeRepo {
        fn find_pack_by_name(&self, name: &str) -> Result<Option<Pack>, RepoError> {
            Ok(self.packs.borrow().iter().find(|p| p.name == name).cloned())
        }
        fn upsert_pack(&self, pack: &Pack) -> Result<(), RepoError> {
            let mut v = self.packs.borrow_mut();
            match v.iter_mut().find(|p| p.name == pack.name) {
                Some(e) => *e = pack.clone(),
                None => v.push(pack.clone()),
            }
            Ok(())
        }
        fn find_sticker_by_unique_id(&self, uid: &str) -> Result<Option<Sticker>, RepoError> {
            Ok(self
                .stickers
                .borrow()
                .iter()
                .find(|s| s.file_unique_id == uid)
                .cloned())
        }
        fn upsert_sticker(&self, s: &Sticker) -> Result<(), RepoError> {
            let mut v = self.stickers.borrow_mut();
            match v.iter_mut().find(|x| x.file_unique_id == s.file_unique_id) {
                Some(e) => *e = s.clone(),
                None => v.push(s.clone()),
            }
            Ok(())
        }
        fn list_stickers(&self, _pack: Option<&str>) -> Result<Vec<Sticker>, RepoError> {
            Ok(self.stickers.borrow().clone())
        }
    }

    #[derive(Default)]
    struct FakeImages {
        existing: RefCell<HashSet<String>>,
    }

    impl FakeImages {
        fn mark(&self, pack: &str, file_name: &str) {
            self.existing.borrow_mut().insert(format!("{pack}/{file_name}"));
        }
    }

    impl ImageStore for FakeImages {
        fn exists(&self, pack: &str, file_name: &str) -> bool {
            self.existing.borrow().contains(&format!("{pack}/{file_name}"))
        }
        fn save(&self, pack: &str, file_name: &str, _bytes: &[u8]) -> Result<String, StoreError> {
            let path = format!("{pack}/{file_name}");
            self.existing.borrow_mut().insert(path.clone());
            Ok(path)
        }
        fn read(&self, _image_path: &str) -> Result<Vec<u8>, StoreError> {
            unreachable!("scraper never reads images")
        }
    }

    fn uc(
        gw: FakeGateway,
        repo: FakeRepo,
        images: FakeImages,
    ) -> ScrapePacks<FakeGateway, FakeRepo, FakeImages> {
        ScrapePacks::with_clock(gw, repo, images, fixed_now)
    }

    // ---- application tests ----

    #[tokio::test]
    async fn fresh_sticker_is_downloaded_and_persisted() {
        let mut sets = HashMap::new();
        sets.insert("packA".into(), set_with("packA", vec![remote("u1", "f1")]));
        let app = uc(FakeGateway::new(sets), FakeRepo::default(), FakeImages::default());

        let s = app.run(&["packA".into()]).await;

        assert_eq!(s, ScrapeSummary { packs_ok: 1, downloaded: 1, ..Default::default() });
        assert_eq!(app.repo().packs.borrow().len(), 1);
        let stickers = app.repo().stickers.borrow();
        assert_eq!(stickers.len(), 1);
        assert_eq!(stickers[0].file_id, "f1");
        assert_eq!(stickers[0].position, 0);
        assert_eq!(stickers[0].image_path, format!("packA/{}.webp", stickers[0].id));
    }

    #[tokio::test]
    async fn existing_unique_id_reuses_uuid_and_skips_download() {
        let mut sets = HashMap::new();
        sets.insert("packA".into(), set_with("packA", vec![remote("u1", "f1")]));
        let repo = FakeRepo::default();
        let pack_id = Uuid::new_v4();
        let sticker_id = Uuid::new_v4();
        repo.upsert_pack(&Pack {
            id: pack_id,
            name: "packA".into(),
            title: "Title".into(),
            fetched_at: fixed_now(),
        })
        .unwrap();
        repo.upsert_sticker(&Sticker {
            id: sticker_id,
            pack_id,
            file_unique_id: "u1".into(),
            file_id: "stale".into(),
            emoji: None,
            format: StickerFormat::Static,
            width: 512,
            height: 512,
            position: 0,
            image_path: format!("packA/{sticker_id}.webp"),
            created_at: fixed_now(),
        })
        .unwrap();
        let images = FakeImages::default();
        images.mark("packA", &format!("{sticker_id}.webp"));

        let app = uc(FakeGateway::new(sets), repo, images);
        let s = app.run(&["packA".into()]).await;

        assert_eq!(s.skipped_existing, 1);
        assert_eq!(s.downloaded, 0);
        assert_eq!(app.gateway().download_calls(), 0);
        let stickers = app.repo().stickers.borrow();
        assert_eq!(stickers.len(), 1);
        assert_eq!(stickers[0].id, sticker_id, "uuid must be preserved");
        assert_eq!(stickers[0].file_id, "f1", "volatile file_id refreshed");
    }

    #[tokio::test]
    async fn one_bad_sticker_does_not_abort_the_pack() {
        let mut sets = HashMap::new();
        sets.insert(
            "packA".into(),
            set_with("packA", vec![remote("u1", "f1"), remote("u2", "f2")]),
        );
        let gw = FakeGateway::new(sets).fail_download("f2_thumb");
        let app = uc(gw, FakeRepo::default(), FakeImages::default());

        let s = app.run(&["packA".into()]).await;

        assert_eq!(s.packs_ok, 1);
        assert_eq!(s.downloaded, 1);
        assert_eq!(s.failed, 1);
        assert_eq!(app.repo().stickers.borrow().len(), 1);
    }

    #[tokio::test]
    async fn one_bad_pack_does_not_abort_the_run() {
        let mut sets = HashMap::new();
        sets.insert("good".into(), set_with("good", vec![remote("u1", "f1")]));
        let app = uc(FakeGateway::new(sets), FakeRepo::default(), FakeImages::default());

        let s = app.run(&["missing".into(), "good".into()]).await;

        assert_eq!(s.packs_ok, 1);
        assert_eq!(s.packs_failed, 1);
        assert_eq!(s.downloaded, 1);
    }

    #[rstest::rstest]
    #[case(false, false, StickerFormat::Static)]
    #[case(true, false, StickerFormat::Animated)]
    #[case(false, true, StickerFormat::Video)]
    #[case(true, true, StickerFormat::Video)]
    fn format_from_flags(
        #[case] animated: bool,
        #[case] video: bool,
        #[case] expected: StickerFormat,
    ) {
        assert_eq!(StickerFormat::from_flags(animated, video), expected);
    }
}
