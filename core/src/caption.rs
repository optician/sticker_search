//! The `CaptionStickers` use-case: caption each stored thumbnail through a VLM
//! gateway and persist the result, keyed by `(sticker, model, prompt_version)`.

use crate::entities::{Caption, Prompt, Sticker};
use crate::error::{CaptionError, CaptionStickerError};
use crate::ports::{CaptionGateway, CaptionRepository, ImageStore, StickerRepository};
use time::OffsetDateTime;
use uuid::Uuid;

/// Counts reported at the end of a run. Never aborts on per-sticker failure.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CaptionSummary {
    pub captioned: u32,
    pub skipped: u32,
    pub failed: u32,
}

/// Per-run knobs. `pack` scopes the batch; `force` re-captions stickers already
/// captioned by this `(model, prompt_version)`; `limit` caps the run.
#[derive(Debug, Default, Clone, Copy)]
pub struct CaptionRun<'a> {
    pub pack: Option<&'a str>,
    pub force: bool,
    pub limit: Option<usize>,
}

/// What happened to one sticker, reported to the progress hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressEvent {
    /// About to call the (slow) model for this sticker.
    Start,
    Captioned,
    Skipped,
    Failed,
}

/// A progress update emitted before and after each sticker in a run. `index` is
/// 1-based within the run; `total` is how many stickers this run will touch.
#[derive(Debug, Clone, Copy)]
pub struct CaptionProgress<'a> {
    pub index: usize,
    pub total: usize,
    pub sticker_id: Uuid,
    pub image_path: &'a str,
    pub event: ProgressEvent,
}

/// Default progress hook: does nothing.
fn ignore_progress(_: CaptionProgress) {}

/// Orchestrates captioning. Generic over the ports (static dispatch); `R` carries
/// both the sticker and caption repositories (one adapter implements both).
/// `now` is an injectable clock so tests are deterministic.
pub struct CaptionStickers<G, R, S> {
    gateway: G,
    repo: R,
    images: S,
    now: fn() -> OffsetDateTime,
    report: fn(CaptionProgress),
}

impl<G, R, S> CaptionStickers<G, R, S>
where
    G: CaptionGateway,
    R: StickerRepository + CaptionRepository,
    S: ImageStore,
{
    pub fn new(gateway: G, repo: R, images: S) -> Self {
        Self::with_clock(gateway, repo, images, OffsetDateTime::now_utc)
    }

    pub fn with_clock(gateway: G, repo: R, images: S, now: fn() -> OffsetDateTime) -> Self {
        Self { gateway, repo, images, now, report: ignore_progress }
    }

    /// Install a progress hook, called once with `Start` before each sticker and
    /// once with its outcome after. The composition root owns presentation.
    pub fn on_progress(mut self, report: fn(CaptionProgress)) -> Self {
        self.report = report;
        self
    }

    pub fn gateway(&self) -> &G {
        &self.gateway
    }

    pub fn repo(&self) -> &R {
        &self.repo
    }

    /// Caption every selected sticker. Run-level setup failures (prompt
    /// precondition, listing stickers) abort and surface as `Err`; per-sticker
    /// failures are logged and counted into the summary.
    pub async fn run(
        &self,
        prompt: &Prompt,
        cfg: CaptionRun<'_>,
    ) -> Result<CaptionSummary, CaptionError> {
        self.register_prompt(prompt)?;

        let stickers = self.repo.list_stickers(cfg.pack)?;
        let total = cfg.limit.unwrap_or(stickers.len()).min(stickers.len());

        let mut summary = CaptionSummary::default();
        for (i, sticker) in stickers.iter().take(total).enumerate() {
            let report = |event| {
                (self.report)(CaptionProgress {
                    index: i + 1,
                    total,
                    sticker_id: sticker.id,
                    image_path: &sticker.image_path,
                    event,
                });
            };
            report(ProgressEvent::Start);
            match self.caption_one(sticker, prompt, cfg.force).await {
                Ok(Outcome::Captioned) => {
                    summary.captioned += 1;
                    report(ProgressEvent::Captioned);
                }
                Ok(Outcome::Skipped) => {
                    summary.skipped += 1;
                    report(ProgressEvent::Skipped);
                }
                Err(e) => {
                    summary.failed += 1;
                    tracing::warn!(sticker = %sticker.id, error = %e, "caption failed");
                    report(ProgressEvent::Failed);
                }
            }
        }
        Ok(summary)
    }

    /// Ensure the prompt is on record. A version that already exists with
    /// *different* text is a programmer error (edited without bumping) and
    /// aborts the run. First sighting of a version is inserted as-is.
    fn register_prompt(&self, prompt: &Prompt) -> Result<(), CaptionError> {
        match self.repo.find_prompt(&prompt.version)? {
            Some(existing) if existing.text != prompt.text => {
                Err(CaptionError::PromptVersionMismatch { version: prompt.version.clone() })
            }
            Some(_) => Ok(()),
            None => {
                self.repo.upsert_prompt(prompt)?;
                Ok(())
            }
        }
    }

    async fn caption_one(
        &self,
        sticker: &Sticker,
        prompt: &Prompt,
        force: bool,
    ) -> Result<Outcome, CaptionStickerError> {
        let model = self.gateway.model();
        if !force && self.repo.caption_exists(sticker.id, model, &prompt.version)? {
            return Ok(Outcome::Skipped);
        }

        let bytes = self.images.read(&sticker.image_path)?;
        let result = self.gateway.caption(&bytes).await?;

        self.repo.upsert_caption(&Caption {
            sticker_id: sticker.id,
            model: model.to_string(),
            prompt_version: prompt.version.clone(),
            scene: result.fields.scene,
            on_image_text: result.fields.on_image_text,
            tone: result.fields.tone,
            situations: result.fields.situations,
            raw: result.raw,
            created_at: (self.now)(),
        })?;
        Ok(Outcome::Captioned)
    }
}

/// Outcome of processing a single sticker.
enum Outcome {
    Captioned,
    Skipped,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::{CaptionFields, CaptionResult, StickerFormat};
    use crate::error::{CaptionGatewayError, RepoError};
    use std::cell::{Cell, RefCell};
    use uuid::Uuid;

    fn fixed_now() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH
    }

    fn prompt(version: &str, text: &str) -> Prompt {
        Prompt { version: version.into(), text: text.into(), created_at: fixed_now() }
    }

    fn sticker(path: &str) -> Sticker {
        Sticker {
            id: Uuid::new_v4(),
            pack_id: Uuid::new_v4(),
            file_unique_id: format!("u-{path}"),
            file_id: "f".into(),
            emoji: None,
            format: StickerFormat::Static,
            width: 512,
            height: 512,
            position: 0,
            image_path: path.into(),
            created_at: fixed_now(),
        }
    }

    // ---- fakes ----

    struct FakeGateway {
        model: String,
        fail_on: RefCell<std::collections::HashSet<Vec<u8>>>,
        calls: Cell<u32>,
    }

    impl FakeGateway {
        fn new(model: &str) -> Self {
            Self {
                model: model.into(),
                fail_on: RefCell::new(std::collections::HashSet::new()),
                calls: Cell::new(0),
            }
        }
        fn fail_on_bytes(self, bytes: &[u8]) -> Self {
            self.fail_on.borrow_mut().insert(bytes.to_vec());
            self
        }
        fn calls(&self) -> u32 {
            self.calls.get()
        }
    }

    impl CaptionGateway for FakeGateway {
        fn model(&self) -> &str {
            &self.model
        }
        async fn caption(&self, image_png: &[u8]) -> Result<CaptionResult, CaptionGatewayError> {
            self.calls.set(self.calls.get() + 1);
            if self.fail_on.borrow().contains(image_png) {
                return Err(CaptionGatewayError::Transport("boom".into()));
            }
            // Echo the input path bytes into the scene so tests can assert mapping.
            let scene = String::from_utf8_lossy(image_png).to_string();
            Ok(CaptionResult {
                fields: CaptionFields {
                    scene,
                    on_image_text: "TEXT".into(),
                    tone: "humorous".into(),
                    situations: vec!["s1".into()],
                },
                raw: "{\"raw\":true}".into(),
            })
        }
    }

    #[derive(Default)]
    struct FakeRepo {
        stickers: Vec<Sticker>,
        pack_of: std::collections::HashMap<Uuid, String>,
        captions: RefCell<Vec<Caption>>,
        prompts: RefCell<Vec<Prompt>>,
    }

    impl FakeRepo {
        fn with_stickers(stickers: Vec<Sticker>) -> Self {
            Self { stickers, ..Default::default() }
        }
    }

    impl StickerRepository for FakeRepo {
        fn find_pack_by_name(&self, _: &str) -> Result<Option<crate::entities::Pack>, RepoError> {
            Ok(None)
        }
        fn upsert_pack(&self, _: &crate::entities::Pack) -> Result<(), RepoError> {
            Ok(())
        }
        fn find_sticker_by_unique_id(&self, _: &str) -> Result<Option<Sticker>, RepoError> {
            Ok(None)
        }
        fn find_sticker_by_id(&self, id: Uuid) -> Result<Option<Sticker>, RepoError> {
            Ok(self.stickers.iter().find(|s| s.id == id).cloned())
        }
        fn upsert_sticker(&self, _: &Sticker) -> Result<(), RepoError> {
            Ok(())
        }
        fn list_stickers(&self, pack: Option<&str>) -> Result<Vec<Sticker>, RepoError> {
            Ok(self
                .stickers
                .iter()
                .filter(|s| match pack {
                    None => true,
                    Some(p) => self.pack_of.get(&s.pack_id).is_some_and(|n| n == p),
                })
                .cloned()
                .collect())
        }
    }

    impl CaptionRepository for FakeRepo {
        fn caption_exists(
            &self,
            sticker_id: Uuid,
            model: &str,
            prompt_version: &str,
        ) -> Result<bool, RepoError> {
            Ok(self.captions.borrow().iter().any(|c| {
                c.sticker_id == sticker_id && c.model == model && c.prompt_version == prompt_version
            }))
        }
        fn upsert_caption(&self, caption: &Caption) -> Result<(), RepoError> {
            let mut v = self.captions.borrow_mut();
            match v.iter_mut().find(|c| {
                c.sticker_id == caption.sticker_id
                    && c.model == caption.model
                    && c.prompt_version == caption.prompt_version
            }) {
                Some(e) => *e = caption.clone(),
                None => v.push(caption.clone()),
            }
            Ok(())
        }
        fn find_prompt(&self, version: &str) -> Result<Option<Prompt>, RepoError> {
            Ok(self.prompts.borrow().iter().find(|p| p.version == version).cloned())
        }
        fn upsert_prompt(&self, prompt: &Prompt) -> Result<(), RepoError> {
            let mut v = self.prompts.borrow_mut();
            match v.iter_mut().find(|p| p.version == prompt.version) {
                Some(e) => *e = prompt.clone(),
                None => v.push(prompt.clone()),
            }
            Ok(())
        }
    }

    struct FakeImages;

    impl ImageStore for FakeImages {
        fn exists(&self, _: &str, _: &str) -> bool {
            false
        }
        fn save(&self, _: &str, _: &str, _: &[u8]) -> Result<String, crate::error::StoreError> {
            unreachable!("captioner never saves")
        }
        fn read(&self, image_path: &str) -> Result<Vec<u8>, crate::error::StoreError> {
            Ok(image_path.as_bytes().to_vec())
        }
    }

    fn uc(gw: FakeGateway, repo: FakeRepo) -> CaptionStickers<FakeGateway, FakeRepo, FakeImages> {
        CaptionStickers::with_clock(gw, repo, FakeImages, fixed_now)
    }

    // ---- tests ----

    #[tokio::test]
    async fn fresh_sticker_is_captioned_and_persisted() {
        let s = sticker("packA/a.png");
        let id = s.id;
        let app = uc(FakeGateway::new("qwen"), FakeRepo::with_stickers(vec![s]));

        let summary = app.run(&prompt("v1", "describe"), CaptionRun::default()).await.unwrap();

        assert_eq!(summary, CaptionSummary { captioned: 1, ..Default::default() });
        let caps = app.repo().captions.borrow();
        assert_eq!(caps.len(), 1);
        let c = &caps[0];
        assert_eq!(c.sticker_id, id);
        assert_eq!(c.model, "qwen");
        assert_eq!(c.prompt_version, "v1");
        assert_eq!(c.scene, "packA/a.png", "image bytes reached the gateway");
        assert_eq!(c.on_image_text, "TEXT");
        assert_eq!(c.situations, vec!["s1".to_string()]);
        assert_eq!(c.raw, "{\"raw\":true}");
        // prompt was registered
        assert_eq!(app.repo().prompts.borrow().len(), 1);
    }

    #[tokio::test]
    async fn existing_caption_same_model_and_version_is_skipped() {
        let s = sticker("packA/a.png");
        let repo = FakeRepo::with_stickers(vec![s.clone()]);
        repo.upsert_prompt(&prompt("v1", "describe")).unwrap();
        repo.upsert_caption(&Caption {
            sticker_id: s.id,
            model: "qwen".into(),
            prompt_version: "v1".into(),
            scene: "old".into(),
            on_image_text: String::new(),
            tone: "x".into(),
            situations: vec![],
            raw: String::new(),
            created_at: fixed_now(),
        })
        .unwrap();
        let app = uc(FakeGateway::new("qwen"), repo);

        let summary = app.run(&prompt("v1", "describe"), CaptionRun::default()).await.unwrap();

        assert_eq!(summary, CaptionSummary { skipped: 1, ..Default::default() });
        assert_eq!(app.gateway().calls(), 0, "skip avoids the model call");
        assert_eq!(app.repo().captions.borrow()[0].scene, "old", "not overwritten");
    }

    #[tokio::test]
    async fn force_recaptions_existing() {
        let s = sticker("packA/a.png");
        let repo = FakeRepo::with_stickers(vec![s.clone()]);
        repo.upsert_caption(&Caption {
            sticker_id: s.id,
            model: "qwen".into(),
            prompt_version: "v1".into(),
            scene: "old".into(),
            on_image_text: String::new(),
            tone: "x".into(),
            situations: vec![],
            raw: String::new(),
            created_at: fixed_now(),
        })
        .unwrap();
        let app = uc(FakeGateway::new("qwen"), repo);

        let cfg = CaptionRun { force: true, ..Default::default() };
        let summary = app.run(&prompt("v1", "describe"), cfg).await.unwrap();

        assert_eq!(summary, CaptionSummary { captioned: 1, ..Default::default() });
        assert_eq!(app.gateway().calls(), 1);
        assert_eq!(app.repo().captions.borrow()[0].scene, "packA/a.png", "overwritten");
    }

    #[tokio::test]
    async fn different_model_coexists_with_existing_caption() {
        let s = sticker("packA/a.png");
        let repo = FakeRepo::with_stickers(vec![s.clone()]);
        repo.upsert_caption(&Caption {
            sticker_id: s.id,
            model: "other".into(),
            prompt_version: "v1".into(),
            scene: "from-other".into(),
            on_image_text: String::new(),
            tone: "x".into(),
            situations: vec![],
            raw: String::new(),
            created_at: fixed_now(),
        })
        .unwrap();
        let app = uc(FakeGateway::new("qwen"), repo);

        let summary = app.run(&prompt("v1", "describe"), CaptionRun::default()).await.unwrap();

        assert_eq!(summary, CaptionSummary { captioned: 1, ..Default::default() });
        assert_eq!(app.repo().captions.borrow().len(), 2, "both models' rows kept");
    }

    #[tokio::test]
    async fn one_bad_sticker_does_not_abort_the_run() {
        let good = sticker("packA/good.png");
        let bad = sticker("packA/bad.png");
        let gw = FakeGateway::new("qwen").fail_on_bytes(b"packA/bad.png");
        let app = uc(gw, FakeRepo::with_stickers(vec![good, bad]));

        let summary = app.run(&prompt("v1", "describe"), CaptionRun::default()).await.unwrap();

        assert_eq!(summary, CaptionSummary { captioned: 1, failed: 1, ..Default::default() });
        assert_eq!(app.repo().captions.borrow().len(), 1);
    }

    #[tokio::test]
    async fn prompt_version_mismatch_aborts_before_processing() {
        let s = sticker("packA/a.png");
        let repo = FakeRepo::with_stickers(vec![s]);
        repo.upsert_prompt(&prompt("v1", "ORIGINAL")).unwrap();
        let app = uc(FakeGateway::new("qwen"), repo);

        let err = app.run(&prompt("v1", "EDITED"), CaptionRun::default()).await.unwrap_err();

        assert!(matches!(err, CaptionError::PromptVersionMismatch { version } if version == "v1"));
        assert_eq!(app.gateway().calls(), 0);
        assert!(app.repo().captions.borrow().is_empty());
    }

    use std::sync::atomic::{AtomicUsize, Ordering};
    static STARTS: AtomicUsize = AtomicUsize::new(0);
    static DONES: AtomicUsize = AtomicUsize::new(0);
    fn count_events(p: CaptionProgress) {
        match p.event {
            ProgressEvent::Start => {
                STARTS.fetch_add(1, Ordering::SeqCst);
            }
            _ => {
                DONES.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    #[tokio::test]
    async fn reports_one_start_and_one_outcome_per_sticker() {
        STARTS.store(0, Ordering::SeqCst);
        DONES.store(0, Ordering::SeqCst);
        let stickers = vec![sticker("a.png"), sticker("b.png"), sticker("c.png")];
        let app = uc(FakeGateway::new("qwen"), FakeRepo::with_stickers(stickers))
            .on_progress(count_events);

        app.run(&prompt("v1", "describe"), CaptionRun::default()).await.unwrap();

        assert_eq!(STARTS.load(Ordering::SeqCst), 3, "one Start per sticker");
        assert_eq!(DONES.load(Ordering::SeqCst), 3, "one outcome per sticker");
    }

    #[tokio::test]
    async fn limit_caps_the_run() {
        let stickers = vec![sticker("a.png"), sticker("b.png"), sticker("c.png")];
        let app = uc(FakeGateway::new("qwen"), FakeRepo::with_stickers(stickers));

        let cfg = CaptionRun { limit: Some(1), ..Default::default() };
        let summary = app.run(&prompt("v1", "describe"), cfg).await.unwrap();

        assert_eq!(summary, CaptionSummary { captioned: 1, ..Default::default() });
    }
}
