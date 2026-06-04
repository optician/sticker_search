//! SQLite-backed `StickerRepository` (rusqlite, bundled).

use rusqlite::{Connection, OptionalExtension, Row, params};
use std::sync::{Mutex, MutexGuard};
use sticker_core::entities::{
    Caption, EmbedDoc, Pack, PackRequest, Prompt, Sticker, StickerFormat,
};
use sticker_core::error::RepoError;
use sticker_core::ports::{
    CaptionLookup, CaptionReader, CaptionRepository, PackRequests, StickerRepository,
};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;
use uuid::Uuid;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS packs (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL UNIQUE,
  title       TEXT NOT NULL,
  fetched_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stickers (
  id              TEXT PRIMARY KEY,
  pack_id         TEXT NOT NULL REFERENCES packs(id),
  file_unique_id  TEXT NOT NULL UNIQUE,
  file_id         TEXT NOT NULL,
  emoji           TEXT,
  format          TEXT NOT NULL,
  width           INTEGER NOT NULL,
  height          INTEGER NOT NULL,
  position        INTEGER NOT NULL,
  image_path      TEXT NOT NULL,
  created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prompts (
  version    TEXT PRIMARY KEY,
  text       TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS captions (
  sticker_id     TEXT NOT NULL REFERENCES stickers(id),
  model          TEXT NOT NULL,
  prompt_version TEXT NOT NULL REFERENCES prompts(version),
  scene          TEXT NOT NULL,
  on_image_text  TEXT NOT NULL,
  tone           TEXT NOT NULL,
  situations     TEXT NOT NULL,
  raw            TEXT NOT NULL,
  created_at     TEXT NOT NULL,
  PRIMARY KEY (sticker_id, model, prompt_version)
);
CREATE TABLE IF NOT EXISTS pack_requests (
  name         TEXT PRIMARY KEY,
  requested_by INTEGER NOT NULL,
  requested_at TEXT NOT NULL
);
"#;

/// rusqlite's `Connection` is `Send` but `!Sync`. The live bot drives the async
/// query path on a multi-threaded runtime, where handler futures must be `Send`
/// and hold `&SqliteRepository` across `.await` — which needs the repository to be
/// `Sync`. A `Mutex` around the connection provides that; the bot is low-traffic,
/// so lock contention is a non-issue.
pub struct SqliteRepository {
    db: Mutex<Connection>,
}

impl SqliteRepository {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, RepoError> {
        let conn = Connection::open(path).map_err(storage)?;
        Self::init(conn)
    }

    pub fn open_in_memory() -> Result<Self, RepoError> {
        let conn = Connection::open_in_memory().map_err(storage)?;
        Self::init(conn)
    }

    fn init(conn: Connection) -> Result<Self, RepoError> {
        conn.execute_batch(SCHEMA).map_err(storage)?;
        Ok(Self {
            db: Mutex::new(conn),
        })
    }

    /// Lock the connection. A poisoned lock (a previous holder panicked) becomes a
    /// storage error rather than a panic, honoring the crate's no-panic rule.
    fn lock(&self) -> Result<MutexGuard<'_, Connection>, RepoError> {
        self.db.lock().map_err(storage)
    }
}

fn storage<E: std::fmt::Display>(e: E) -> RepoError {
    RepoError::Storage(e.to_string())
}

fn parse_format(s: &str) -> StickerFormat {
    match s {
        "animated" => StickerFormat::Animated,
        "video" => StickerFormat::Video,
        _ => StickerFormat::Static,
    }
}

fn rfc3339(dt: OffsetDateTime) -> Result<String, RepoError> {
    dt.format(&Rfc3339).map_err(storage)
}

fn parse_uuid(s: &str) -> Result<Uuid, RepoError> {
    Uuid::parse_str(s).map_err(storage)
}

fn parse_time(s: &str) -> Result<OffsetDateTime, RepoError> {
    OffsetDateTime::parse(s, &Rfc3339).map_err(storage)
}

/// The 11 sticker columns, in `SELECT` order, as raw SQL types.
type StickerCols = (
    String,
    String,
    String,
    String,
    Option<String>,
    String,
    u32,
    u32,
    u32,
    String,
    String,
);

/// Sticker columns, `s.`-qualified so they're unambiguous when joined to `packs`.
/// Every query that selects them aliases the table as `s`.
const STICKER_SELECT: &str = "s.id, s.pack_id, s.file_unique_id, s.file_id, s.emoji, s.format,
                              s.width, s.height, s.position, s.image_path, s.created_at";

fn sticker_cols(r: &Row) -> rusqlite::Result<StickerCols> {
    Ok((
        r.get(0)?,
        r.get(1)?,
        r.get(2)?,
        r.get(3)?,
        r.get(4)?,
        r.get(5)?,
        r.get(6)?,
        r.get(7)?,
        r.get(8)?,
        r.get(9)?,
        r.get(10)?,
    ))
}

fn build_sticker(c: StickerCols) -> Result<Sticker, RepoError> {
    Ok(Sticker {
        id: parse_uuid(&c.0)?,
        pack_id: parse_uuid(&c.1)?,
        file_unique_id: c.2,
        file_id: c.3,
        emoji: c.4,
        format: parse_format(&c.5),
        width: c.6,
        height: c.7,
        position: c.8,
        image_path: c.9,
        created_at: parse_time(&c.10)?,
    })
}

/// The 9 caption columns, in `SELECT` order. `c.`-qualified so they're
/// unambiguous when the captions table is joined to `stickers`.
const CAPTION_SELECT: &str = "c.sticker_id, c.model, c.prompt_version, c.scene, c.on_image_text,
                              c.tone, c.situations, c.raw, c.created_at";

type CaptionCols = (
    String,
    String,
    String,
    String,
    String,
    String,
    String,
    String,
    String,
);

fn caption_cols(r: &Row) -> rusqlite::Result<CaptionCols> {
    Ok((
        r.get(0)?,
        r.get(1)?,
        r.get(2)?,
        r.get(3)?,
        r.get(4)?,
        r.get(5)?,
        r.get(6)?,
        r.get(7)?,
        r.get(8)?,
    ))
}

fn build_caption(c: CaptionCols) -> Result<Caption, RepoError> {
    Ok(Caption {
        sticker_id: parse_uuid(&c.0)?,
        model: c.1,
        prompt_version: c.2,
        scene: c.3,
        on_image_text: c.4,
        tone: c.5,
        situations: serde_json::from_str(&c.6).map_err(storage)?,
        raw: c.7,
        created_at: parse_time(&c.8)?,
    })
}

/// The embedder's row: the 9 caption columns plus the sticker emoji and the
/// pack name/title the document folds in. Selected as `{CAPTION_SELECT}, s.emoji,
/// p.name, p.title`, so the three extras land at indices 9, 10, 11.
type EmbedDocCols = (CaptionCols, Option<String>, String, String);

fn embed_doc_cols(r: &Row) -> rusqlite::Result<EmbedDocCols> {
    Ok((caption_cols(r)?, r.get(9)?, r.get(10)?, r.get(11)?))
}

fn build_embed_doc(c: EmbedDocCols) -> Result<EmbedDoc, RepoError> {
    Ok(EmbedDoc {
        caption: build_caption(c.0)?,
        emoji: c.1,
        pack_name: c.2,
        pack_title: c.3,
    })
}

impl StickerRepository for SqliteRepository {
    fn find_pack_by_name(&self, name: &str) -> Result<Option<Pack>, RepoError> {
        let row: Option<(String, String, String, String)> = self
            .lock()?
            .query_row(
                "SELECT id, name, title, fetched_at FROM packs WHERE name = ?1",
                params![name],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .optional()
            .map_err(storage)?;

        row.map(|(id, name, title, fetched_at)| {
            Ok(Pack {
                id: parse_uuid(&id)?,
                name,
                title,
                fetched_at: parse_time(&fetched_at)?,
            })
        })
        .transpose()
    }

    fn upsert_pack(&self, pack: &Pack) -> Result<(), RepoError> {
        self.lock()?
            .execute(
                "INSERT INTO packs (id, name, title, fetched_at)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(name) DO UPDATE SET
                   title = excluded.title,
                   fetched_at = excluded.fetched_at",
                params![
                    pack.id.to_string(),
                    pack.name,
                    pack.title,
                    rfc3339(pack.fetched_at)?
                ],
            )
            .map_err(storage)?;
        Ok(())
    }

    fn find_sticker_by_unique_id(&self, uid: &str) -> Result<Option<Sticker>, RepoError> {
        let row = self
            .lock()?
            .query_row(
                &format!("SELECT {STICKER_SELECT} FROM stickers s WHERE s.file_unique_id = ?1"),
                params![uid],
                sticker_cols,
            )
            .optional()
            .map_err(storage)?;

        row.map(build_sticker).transpose()
    }

    fn find_sticker_by_id(&self, id: Uuid) -> Result<Option<Sticker>, RepoError> {
        let row = self
            .lock()?
            .query_row(
                &format!("SELECT {STICKER_SELECT} FROM stickers s WHERE s.id = ?1"),
                params![id.to_string()],
                sticker_cols,
            )
            .optional()
            .map_err(storage)?;

        row.map(build_sticker).transpose()
    }

    fn list_stickers(&self, pack: Option<&str>) -> Result<Vec<Sticker>, RepoError> {
        let sql = match pack {
            Some(_) => format!(
                "SELECT {STICKER_SELECT} FROM stickers s
                 JOIN packs p ON p.id = s.pack_id
                 WHERE p.name = ?1 ORDER BY s.pack_id, s.position"
            ),
            None => {
                format!("SELECT {STICKER_SELECT} FROM stickers s ORDER BY s.pack_id, s.position")
            }
        };
        let conn = self.lock()?;
        let mut stmt = conn.prepare(&sql).map_err(storage)?;
        let rows = match pack {
            Some(name) => stmt.query_map(params![name], sticker_cols),
            None => stmt.query_map([], sticker_cols),
        }
        .map_err(storage)?;

        let mut out = Vec::new();
        for row in rows {
            out.push(build_sticker(row.map_err(storage)?)?);
        }
        Ok(out)
    }

    fn upsert_sticker(&self, s: &Sticker) -> Result<(), RepoError> {
        self.lock()?
            .execute(
                "INSERT INTO stickers
                   (id, pack_id, file_unique_id, file_id, emoji, format,
                    width, height, position, image_path, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
                 ON CONFLICT(file_unique_id) DO UPDATE SET
                   file_id    = excluded.file_id,
                   emoji      = excluded.emoji,
                   format     = excluded.format,
                   width      = excluded.width,
                   height     = excluded.height,
                   position   = excluded.position,
                   image_path = excluded.image_path",
                params![
                    s.id.to_string(),
                    s.pack_id.to_string(),
                    s.file_unique_id,
                    s.file_id,
                    s.emoji,
                    s.format.as_str(),
                    s.width,
                    s.height,
                    s.position,
                    s.image_path,
                    rfc3339(s.created_at)?,
                ],
            )
            .map_err(storage)?;
        Ok(())
    }
}

impl CaptionRepository for SqliteRepository {
    fn caption_exists(
        &self,
        sticker_id: Uuid,
        model: &str,
        prompt_version: &str,
    ) -> Result<bool, RepoError> {
        let found: Option<i64> = self
            .lock()?
            .query_row(
                "SELECT 1 FROM captions
                 WHERE sticker_id = ?1 AND model = ?2 AND prompt_version = ?3",
                params![sticker_id.to_string(), model, prompt_version],
                |r| r.get(0),
            )
            .optional()
            .map_err(storage)?;
        Ok(found.is_some())
    }

    fn upsert_caption(&self, c: &Caption) -> Result<(), RepoError> {
        let situations = serde_json::to_string(&c.situations).map_err(storage)?;
        self.lock()?
            .execute(
                "INSERT INTO captions
                   (sticker_id, model, prompt_version, scene, on_image_text,
                    tone, situations, raw, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                 ON CONFLICT(sticker_id, model, prompt_version) DO UPDATE SET
                   scene         = excluded.scene,
                   on_image_text = excluded.on_image_text,
                   tone          = excluded.tone,
                   situations    = excluded.situations,
                   raw           = excluded.raw,
                   created_at    = excluded.created_at",
                params![
                    c.sticker_id.to_string(),
                    c.model,
                    c.prompt_version,
                    c.scene,
                    c.on_image_text,
                    c.tone,
                    situations,
                    c.raw,
                    rfc3339(c.created_at)?,
                ],
            )
            .map_err(storage)?;
        Ok(())
    }

    fn find_prompt(&self, version: &str) -> Result<Option<Prompt>, RepoError> {
        let row: Option<(String, String, String)> = self
            .lock()?
            .query_row(
                "SELECT version, text, created_at FROM prompts WHERE version = ?1",
                params![version],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()
            .map_err(storage)?;

        row.map(|(version, text, created_at)| {
            Ok(Prompt {
                version,
                text,
                created_at: parse_time(&created_at)?,
            })
        })
        .transpose()
    }

    fn upsert_prompt(&self, p: &Prompt) -> Result<(), RepoError> {
        // First sighting wins: the use-case guards against version/text drift, so
        // an existing version keeps its original text and created_at.
        self.lock()?
            .execute(
                "INSERT INTO prompts (version, text, created_at) VALUES (?1, ?2, ?3)
                 ON CONFLICT(version) DO NOTHING",
                params![p.version, p.text, rfc3339(p.created_at)?],
            )
            .map_err(storage)?;
        Ok(())
    }
}

impl CaptionReader for SqliteRepository {
    fn list_captions(&self, model: &str, prompt_version: &str) -> Result<Vec<EmbedDoc>, RepoError> {
        let conn = self.lock()?;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {CAPTION_SELECT}, s.emoji, p.name, p.title
                 FROM captions c
                 JOIN stickers s ON s.id = c.sticker_id
                 JOIN packs p    ON p.id = s.pack_id
                 WHERE c.model = ?1 AND c.prompt_version = ?2
                 ORDER BY s.pack_id, s.position"
            ))
            .map_err(storage)?;
        let rows = stmt
            .query_map(params![model, prompt_version], embed_doc_cols)
            .map_err(storage)?;

        let mut out = Vec::new();
        for row in rows {
            out.push(build_embed_doc(row.map_err(storage)?)?);
        }
        Ok(out)
    }
}

impl CaptionLookup for SqliteRepository {
    fn find_caption(
        &self,
        sticker_id: Uuid,
        model: &str,
        prompt_version: &str,
    ) -> Result<Option<Caption>, RepoError> {
        let row = self
            .lock()?
            .query_row(
                &format!(
                    "SELECT {CAPTION_SELECT} FROM captions c
                     WHERE c.sticker_id = ?1 AND c.model = ?2 AND c.prompt_version = ?3"
                ),
                params![sticker_id.to_string(), model, prompt_version],
                caption_cols,
            )
            .optional()
            .map_err(storage)?;

        row.map(build_caption).transpose()
    }
}

impl PackRequests for SqliteRepository {
    fn enqueue(&self, name: &str, requested_by: i64, at: OffsetDateTime) -> Result<(), RepoError> {
        // First request for a name wins; a repeat `/add` is a no-op.
        self.lock()?
            .execute(
                "INSERT INTO pack_requests (name, requested_by, requested_at)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(name) DO NOTHING",
                params![name, requested_by, rfc3339(at)?],
            )
            .map_err(storage)?;
        Ok(())
    }

    fn list_requests(&self) -> Result<Vec<PackRequest>, RepoError> {
        let conn = self.lock()?;
        let mut stmt = conn
            .prepare(
                "SELECT name, requested_by, requested_at
                 FROM pack_requests ORDER BY requested_at, name",
            )
            .map_err(storage)?;
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })
            .map_err(storage)?;
        let mut out = Vec::new();
        for row in rows {
            let (name, requested_by, requested_at) = row.map_err(storage)?;
            out.push(PackRequest {
                name,
                requested_by,
                requested_at: parse_time(&requested_at)?,
            });
        }
        Ok(out)
    }
}

/// One row of `captioner stats`: how many captions a `(model, prompt_version)`
/// pair has produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptionStat {
    pub model: String,
    pub prompt_version: String,
    pub count: u64,
}

/// A caption joined to its sticker's pack + image path, for display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptionView {
    pub sticker_id: Uuid,
    pub pack: String,
    pub image_path: String,
    pub model: String,
    pub prompt_version: String,
    pub scene: String,
    pub on_image_text: String,
    pub tone: String,
    pub situations: Vec<String>,
    pub created_at: OffsetDateTime,
}

/// Result ordering for [`SqliteRepository::query_captions`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CaptionSort {
    /// Pack name, then sticker position — stable reading order (default).
    #[default]
    PackPosition,
    /// Most recently captioned first.
    DateDesc,
    /// Oldest captioned first.
    DateAsc,
}

/// Optional filters for [`SqliteRepository::query_captions`]. All `None` lists
/// everything.
#[derive(Debug, Default, Clone)]
pub struct CaptionFilter {
    pub pack: Option<String>,
    pub model: Option<String>,
    pub prompt_version: Option<String>,
    pub sticker_id: Option<Uuid>,
    /// Substring matched (case-insensitively per SQLite `LIKE`) against both
    /// `scene` and `on_image_text`.
    pub text: Option<String>,
    pub limit: Option<usize>,
    pub sort: CaptionSort,
}

/// Read-side queries used by the `captioner` inspection subcommands. These are
/// inherent (not part of a core port): they serve the composition root's CLI,
/// not the captioning use-case.
impl SqliteRepository {
    pub fn caption_stats(&self) -> Result<Vec<CaptionStat>, RepoError> {
        let conn = self.lock()?;
        let mut stmt = conn
            .prepare(
                "SELECT model, prompt_version, COUNT(*)
                 FROM captions GROUP BY model, prompt_version
                 ORDER BY model, prompt_version",
            )
            .map_err(storage)?;
        let rows = stmt
            .query_map([], |r| {
                Ok(CaptionStat {
                    model: r.get(0)?,
                    prompt_version: r.get(1)?,
                    count: r.get::<_, i64>(2)? as u64,
                })
            })
            .map_err(storage)?;
        rows.collect::<rusqlite::Result<Vec<_>>>().map_err(storage)
    }

    /// Distinct pack names that have at least one caption (for filter menus).
    pub fn caption_packs(&self) -> Result<Vec<String>, RepoError> {
        let conn = self.lock()?;
        let mut stmt = conn
            .prepare(
                "SELECT DISTINCT p.name
                 FROM captions c
                 JOIN stickers s ON s.id = c.sticker_id
                 JOIN packs p    ON p.id = s.pack_id
                 ORDER BY p.name",
            )
            .map_err(storage)?;
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .map_err(storage)?;
        rows.collect::<rusqlite::Result<Vec<_>>>().map_err(storage)
    }

    pub fn list_prompts(&self) -> Result<Vec<Prompt>, RepoError> {
        let conn = self.lock()?;
        let mut stmt = conn
            .prepare("SELECT version, text, created_at FROM prompts ORDER BY version")
            .map_err(storage)?;
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })
            .map_err(storage)?;
        let mut out = Vec::new();
        for row in rows {
            let (version, text, created_at) = row.map_err(storage)?;
            out.push(Prompt {
                version,
                text,
                created_at: parse_time(&created_at)?,
            });
        }
        Ok(out)
    }

    pub fn query_captions(&self, f: &CaptionFilter) -> Result<Vec<CaptionView>, RepoError> {
        let mut sql = String::from(
            "SELECT s.id, p.name, s.image_path, c.model, c.prompt_version,
                    c.scene, c.on_image_text, c.tone, c.situations, c.created_at
             FROM captions c
             JOIN stickers s ON s.id = c.sticker_id
             JOIN packs p    ON p.id = s.pack_id
             WHERE 1 = 1",
        );
        let mut args: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        let sticker_id = f.sticker_id.map(|id| id.to_string());
        if let Some(v) = &f.pack {
            sql.push_str(" AND p.name = ?");
            args.push(Box::new(v.clone()));
        }
        if let Some(v) = &f.model {
            sql.push_str(" AND c.model = ?");
            args.push(Box::new(v.clone()));
        }
        if let Some(v) = &f.prompt_version {
            sql.push_str(" AND c.prompt_version = ?");
            args.push(Box::new(v.clone()));
        }
        if let Some(v) = &sticker_id {
            sql.push_str(" AND c.sticker_id = ?");
            args.push(Box::new(v.clone()));
        }
        if let Some(v) = &f.text {
            sql.push_str(" AND (c.scene LIKE ? OR c.on_image_text LIKE ?)");
            let like = format!("%{v}%");
            args.push(Box::new(like.clone()));
            args.push(Box::new(like));
        }
        sql.push_str(match f.sort {
            CaptionSort::PackPosition => " ORDER BY p.name, s.position",
            CaptionSort::DateDesc => " ORDER BY c.created_at DESC",
            CaptionSort::DateAsc => " ORDER BY c.created_at ASC",
        });
        if let Some(n) = f.limit {
            sql.push_str(" LIMIT ?");
            args.push(Box::new(n as i64));
        }

        let conn = self.lock()?;
        let mut stmt = conn.prepare(&sql).map_err(storage)?;
        let param_refs: Vec<&dyn rusqlite::ToSql> = args.iter().map(|b| b.as_ref()).collect();
        let rows = stmt
            .query_map(rusqlite::params_from_iter(param_refs), |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, String>(5)?,
                    r.get::<_, String>(6)?,
                    r.get::<_, String>(7)?,
                    r.get::<_, String>(8)?,
                    r.get::<_, String>(9)?,
                ))
            })
            .map_err(storage)?;

        let mut out = Vec::new();
        for row in rows {
            let c = row.map_err(storage)?;
            out.push(CaptionView {
                sticker_id: parse_uuid(&c.0)?,
                pack: c.1,
                image_path: c.2,
                model: c.3,
                prompt_version: c.4,
                scene: c.5,
                on_image_text: c.6,
                tone: c.7,
                situations: serde_json::from_str(&c.8).map_err(storage)?,
                created_at: parse_time(&c.9)?,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack() -> Pack {
        Pack {
            id: Uuid::new_v4(),
            name: "packA".into(),
            title: "Title".into(),
            fetched_at: OffsetDateTime::UNIX_EPOCH,
        }
    }

    fn sticker(pack_id: Uuid, file_id: &str) -> Sticker {
        Sticker {
            id: Uuid::new_v4(),
            pack_id,
            file_unique_id: "u1".into(),
            file_id: file_id.into(),
            emoji: Some("😀".into()),
            format: StickerFormat::Static,
            width: 512,
            height: 512,
            position: 0,
            image_path: "packA/x.webp".into(),
            created_at: OffsetDateTime::UNIX_EPOCH,
        }
    }

    #[test]
    fn roundtrips_pack_and_sticker() {
        let repo = SqliteRepository::open_in_memory().unwrap();
        let p = pack();
        repo.upsert_pack(&p).unwrap();
        let s = sticker(p.id, "f1");
        repo.upsert_sticker(&s).unwrap();

        assert_eq!(repo.find_pack_by_name("packA").unwrap(), Some(p));
        assert_eq!(repo.find_sticker_by_unique_id("u1").unwrap(), Some(s));
        assert_eq!(repo.find_sticker_by_unique_id("nope").unwrap(), None);
    }

    #[test]
    fn upsert_sticker_is_idempotent_on_unique_id() {
        let repo = SqliteRepository::open_in_memory().unwrap();
        let p = pack();
        repo.upsert_pack(&p).unwrap();

        let first = sticker(p.id, "old");
        repo.upsert_sticker(&first).unwrap();
        // Same file_unique_id, refreshed file_id, but a *different* UUID supplied:
        // the stored UUID must NOT change (the use-case passes the existing one,
        // but the conflict clause guarantees it regardless).
        let mut second = first.clone();
        second.file_id = "new".into();
        repo.upsert_sticker(&second).unwrap();

        let got = repo.find_sticker_by_unique_id("u1").unwrap().unwrap();
        assert_eq!(got.id, first.id, "uuid stable");
        assert_eq!(got.file_id, "new", "mutable field refreshed");

        let count: u32 = repo
            .db
            .lock()
            .unwrap()
            .query_row("SELECT COUNT(*) FROM stickers", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1, "no duplicate row");
    }

    fn sticker_n(pack_id: Uuid, n: u32) -> Sticker {
        Sticker {
            id: Uuid::new_v4(),
            pack_id,
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

    fn caption(sticker_id: Uuid, model: &str, version: &str) -> Caption {
        Caption {
            sticker_id,
            model: model.into(),
            prompt_version: version.into(),
            scene: "scene".into(),
            on_image_text: "ЗАПАХЛО".into(),
            tone: "humorous".into(),
            situations: vec!["a".into(), "b".into()],
            raw: "{}".into(),
            created_at: OffsetDateTime::UNIX_EPOCH,
        }
    }

    #[test]
    fn list_stickers_all_and_by_pack() {
        let repo = SqliteRepository::open_in_memory().unwrap();
        let a = pack();
        repo.upsert_pack(&a).unwrap();
        let b = Pack {
            id: Uuid::new_v4(),
            name: "packB".into(),
            ..a.clone()
        };
        repo.upsert_pack(&b).unwrap();
        repo.upsert_sticker(&sticker_n(a.id, 1)).unwrap();
        repo.upsert_sticker(&sticker_n(a.id, 0)).unwrap();
        repo.upsert_sticker(&sticker_n(b.id, 5)).unwrap();

        let all = repo.list_stickers(None).unwrap();
        assert_eq!(all.len(), 3);

        let only_a = repo.list_stickers(Some("packA")).unwrap();
        assert_eq!(only_a.len(), 2);
        assert!(only_a.iter().all(|s| s.pack_id == a.id));
        assert_eq!(only_a[0].position, 0, "ordered by position");
        assert_eq!(only_a[1].position, 1);

        assert!(repo.list_stickers(Some("nope")).unwrap().is_empty());
    }

    #[test]
    fn caption_upsert_is_idempotent_and_keyed_by_model_and_version() {
        let repo = SqliteRepository::open_in_memory().unwrap();
        let p = pack();
        repo.upsert_pack(&p).unwrap();
        let s = sticker_n(p.id, 0);
        repo.upsert_sticker(&s).unwrap();
        repo.upsert_prompt(&Prompt {
            version: "v1".into(),
            text: "describe".into(),
            created_at: OffsetDateTime::UNIX_EPOCH,
        })
        .unwrap();

        assert!(!repo.caption_exists(s.id, "qwen", "v1").unwrap());
        repo.upsert_caption(&caption(s.id, "qwen", "v1")).unwrap();
        assert!(repo.caption_exists(s.id, "qwen", "v1").unwrap());
        // Different model or version is a distinct caption.
        assert!(!repo.caption_exists(s.id, "other", "v1").unwrap());
        assert!(!repo.caption_exists(s.id, "qwen", "v2").unwrap());

        // Re-upserting the same key overwrites, never duplicates.
        let mut updated = caption(s.id, "qwen", "v1");
        updated.scene = "new-scene".into();
        repo.upsert_caption(&updated).unwrap();
        let (count, scene): (u32, String) = repo
            .db
            .lock()
            .unwrap()
            .query_row(
                "SELECT COUNT(*), MAX(scene) FROM captions WHERE sticker_id = ?1",
                params![s.id.to_string()],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(count, 1, "no duplicate row");
        assert_eq!(scene, "new-scene", "overwritten");
    }

    #[test]
    fn prompt_first_sighting_wins() {
        let repo = SqliteRepository::open_in_memory().unwrap();
        repo.upsert_prompt(&Prompt {
            version: "v1".into(),
            text: "ORIGINAL".into(),
            created_at: OffsetDateTime::UNIX_EPOCH,
        })
        .unwrap();
        // A second insert of the same version is ignored (the use-case enforces
        // version/text integrity; the store just preserves the first text).
        repo.upsert_prompt(&Prompt {
            version: "v1".into(),
            text: "EDITED".into(),
            created_at: OffsetDateTime::UNIX_EPOCH,
        })
        .unwrap();

        let got = repo.find_prompt("v1").unwrap().unwrap();
        assert_eq!(got.text, "ORIGINAL");
        assert_eq!(repo.find_prompt("missing").unwrap(), None);
    }

    /// A repo with one pack, two stickers, two prompt versions, and captions
    /// from two models, for the read-side query tests.
    fn seeded() -> SqliteRepository {
        let repo = SqliteRepository::open_in_memory().unwrap();
        let p = pack();
        repo.upsert_pack(&p).unwrap();
        let s0 = sticker_n(p.id, 0);
        let s1 = sticker_n(p.id, 1);
        repo.upsert_sticker(&s0).unwrap();
        repo.upsert_sticker(&s1).unwrap();
        for v in ["v1", "v2"] {
            repo.upsert_prompt(&Prompt {
                version: v.into(),
                text: format!("text-{v}"),
                created_at: OffsetDateTime::UNIX_EPOCH,
            })
            .unwrap();
        }
        // s0: qwen/v1 (scene mentions chicken), s1: qwen/v1, s0: llava/v1
        let mut c = caption(s0.id, "qwen", "v1");
        c.scene = "a chicken on a pan".into();
        repo.upsert_caption(&c).unwrap();
        repo.upsert_caption(&caption(s1.id, "qwen", "v1")).unwrap();
        repo.upsert_caption(&caption(s0.id, "llava", "v1")).unwrap();
        repo
    }

    #[test]
    fn caption_stats_groups_by_model_and_version() {
        let stats = seeded().caption_stats().unwrap();
        assert_eq!(
            stats,
            vec![
                CaptionStat {
                    model: "llava".into(),
                    prompt_version: "v1".into(),
                    count: 1
                },
                CaptionStat {
                    model: "qwen".into(),
                    prompt_version: "v1".into(),
                    count: 2
                },
            ]
        );
    }

    #[test]
    fn query_captions_filters_and_searches() {
        let repo = seeded();

        // No filter: every caption row.
        assert_eq!(
            repo.query_captions(&CaptionFilter::default())
                .unwrap()
                .len(),
            3
        );

        // By model.
        let qwen = CaptionFilter {
            model: Some("qwen".into()),
            ..Default::default()
        };
        assert_eq!(repo.query_captions(&qwen).unwrap().len(), 2);

        // Text search hits the one scene mentioning "chicken".
        let chicken = CaptionFilter {
            text: Some("chicken".into()),
            ..Default::default()
        };
        let hits = repo.query_captions(&chicken).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].pack, "packA");
        assert!(hits[0].scene.contains("chicken"));
        assert_eq!(hits[0].situations, vec!["a".to_string(), "b".to_string()]);

        // Limit caps results.
        let one = CaptionFilter {
            limit: Some(1),
            ..Default::default()
        };
        assert_eq!(repo.query_captions(&one).unwrap().len(), 1);
    }

    #[test]
    fn list_prompts_returns_all_versions() {
        let prompts = seeded().list_prompts().unwrap();
        assert_eq!(
            prompts
                .iter()
                .map(|p| p.version.as_str())
                .collect::<Vec<_>>(),
            ["v1", "v2"]
        );
    }

    #[test]
    fn caption_packs_lists_distinct_packs_with_captions() {
        assert_eq!(seeded().caption_packs().unwrap(), vec!["packA".to_string()]);
    }

    #[test]
    fn list_captions_returns_only_the_matching_set_in_position_order() {
        let repo = seeded();
        // qwen/v1 captioned s0 (position 0) and s1 (position 1); llava/v1 only s0.
        let qwen = repo.list_captions("qwen", "v1").unwrap();
        assert_eq!(qwen.len(), 2);
        assert!(
            qwen.iter()
                .all(|d| d.caption.model == "qwen" && d.caption.prompt_version == "v1")
        );
        assert_eq!(
            qwen[0].caption.scene, "a chicken on a pan",
            "s0 (position 0) first"
        );

        assert_eq!(repo.list_captions("llava", "v1").unwrap().len(), 1);
        assert!(repo.list_captions("qwen", "v2").unwrap().is_empty());
    }

    #[test]
    fn list_captions_enriches_with_emoji_and_pack_for_the_document() {
        let repo = SqliteRepository::open_in_memory().unwrap();
        let p = pack(); // name "packA", title "Title"
        repo.upsert_pack(&p).unwrap();
        repo.upsert_prompt(&Prompt {
            version: "v1".into(),
            text: "t".into(),
            created_at: OffsetDateTime::UNIX_EPOCH,
        })
        .unwrap();
        let mut s = sticker_n(p.id, 0);
        s.emoji = Some("🥹".into());
        repo.upsert_sticker(&s).unwrap();
        repo.upsert_caption(&caption(s.id, "qwen", "v1")).unwrap();

        let docs = repo.list_captions("qwen", "v1").unwrap();
        let d = &docs[0];
        assert_eq!(d.emoji.as_deref(), Some("🥹"));
        assert_eq!(d.pack_name, "packA");
        assert_eq!(d.pack_title, "Title");
        // The composed document carries emoji and pack through to the embedder.
        assert_eq!(
            d.embed_text(),
            "scene. text: ЗАПАХЛО. tone: humorous. emoji: 🥹. \
             situations: a, b. pack: Title (packA)",
        );
    }

    #[test]
    fn find_sticker_by_id_roundtrips_and_misses() {
        let repo = SqliteRepository::open_in_memory().unwrap();
        let p = pack();
        repo.upsert_pack(&p).unwrap();
        let s = sticker_n(p.id, 0);
        repo.upsert_sticker(&s).unwrap();

        assert_eq!(repo.find_sticker_by_id(s.id).unwrap().as_ref(), Some(&s));
        assert_eq!(repo.find_sticker_by_id(Uuid::new_v4()).unwrap(), None);
    }

    #[test]
    fn find_caption_resolves_one_set_and_misses_other_keys() {
        let repo = seeded();
        // seeded(): s0 has qwen/v1 (chicken scene) and llava/v1; s1 has qwen/v1.
        let s0 = repo.list_stickers(None).unwrap()[0].id;

        let hit = repo.find_caption(s0, "qwen", "v1").unwrap().unwrap();
        assert_eq!(hit.sticker_id, s0);
        assert_eq!(hit.model, "qwen");
        assert_eq!(hit.scene, "a chicken on a pan");
        assert_eq!(
            hit.situations,
            vec!["a".to_string(), "b".to_string()],
            "JSON column parsed"
        );

        // Same sticker, different model — a different caption.
        assert_eq!(
            repo.find_caption(s0, "llava", "v1").unwrap().unwrap().model,
            "llava"
        );
        // No caption for this (model, version).
        assert!(repo.find_caption(s0, "qwen", "v2").unwrap().is_none());
        // No caption for an unknown sticker.
        assert!(
            repo.find_caption(Uuid::new_v4(), "qwen", "v1")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn pack_requests_enqueue_is_idempotent_and_lists_oldest_first() {
        let repo = SqliteRepository::open_in_memory().unwrap();
        let t0 = OffsetDateTime::UNIX_EPOCH;
        let t1 = t0 + time::Duration::hours(1);

        repo.enqueue("packB", 42, t1).unwrap();
        repo.enqueue("packA", 7, t0).unwrap();
        // Repeat request keeps the first requester/time.
        repo.enqueue("packA", 999, t1).unwrap();

        let reqs = repo.list_requests().unwrap();
        assert_eq!(reqs.len(), 2, "no duplicate row for packA");
        assert_eq!(reqs[0].name, "packA", "oldest request first");
        assert_eq!(reqs[0].requested_by, 7, "first requester preserved");
        assert_eq!(reqs[0].requested_at, t0, "first time preserved");
        assert_eq!(reqs[1].name, "packB");
    }

    #[test]
    fn query_captions_sorts_by_date_desc() {
        let repo = SqliteRepository::open_in_memory().unwrap();
        let p = pack();
        repo.upsert_pack(&p).unwrap();
        let old = sticker_n(p.id, 0);
        let fresh = sticker_n(p.id, 1);
        repo.upsert_sticker(&old).unwrap();
        repo.upsert_sticker(&fresh).unwrap();
        repo.upsert_prompt(&Prompt {
            version: "v1".into(),
            text: "t".into(),
            created_at: OffsetDateTime::UNIX_EPOCH,
        })
        .unwrap();
        let mut c_old = caption(old.id, "qwen", "v1");
        c_old.created_at = OffsetDateTime::UNIX_EPOCH;
        let mut c_fresh = caption(fresh.id, "qwen", "v1");
        c_fresh.created_at = OffsetDateTime::UNIX_EPOCH + time::Duration::days(1);
        repo.upsert_caption(&c_old).unwrap();
        repo.upsert_caption(&c_fresh).unwrap();

        let desc = CaptionFilter {
            sort: CaptionSort::DateDesc,
            ..Default::default()
        };
        let got = repo.query_captions(&desc).unwrap();
        assert_eq!(got[0].sticker_id, fresh.id, "freshest first");
        assert_eq!(got[1].sticker_id, old.id);
    }
}
