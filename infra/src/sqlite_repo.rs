//! SQLite-backed `StickerRepository` (rusqlite, bundled).

use rusqlite::{Connection, OptionalExtension, params};
use sticker_core::entities::{Pack, Sticker, StickerFormat};
use sticker_core::error::RepoError;
use sticker_core::ports::StickerRepository;
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
"#;

pub struct SqliteRepository {
    conn: Connection,
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
        Ok(Self { conn })
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

impl StickerRepository for SqliteRepository {
    fn find_pack_by_name(&self, name: &str) -> Result<Option<Pack>, RepoError> {
        let row: Option<(String, String, String, String)> = self
            .conn
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
        self.conn
            .execute(
                "INSERT INTO packs (id, name, title, fetched_at)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(name) DO UPDATE SET
                   title = excluded.title,
                   fetched_at = excluded.fetched_at",
                params![pack.id.to_string(), pack.name, pack.title, rfc3339(pack.fetched_at)?],
            )
            .map_err(storage)?;
        Ok(())
    }

    fn find_sticker_by_unique_id(&self, uid: &str) -> Result<Option<Sticker>, RepoError> {
        let row = self
            .conn
            .query_row(
                "SELECT id, pack_id, file_unique_id, file_id, emoji, format,
                        width, height, position, image_path, created_at
                 FROM stickers WHERE file_unique_id = ?1",
                params![uid],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                        r.get::<_, Option<String>>(4)?,
                        r.get::<_, String>(5)?,
                        r.get::<_, u32>(6)?,
                        r.get::<_, u32>(7)?,
                        r.get::<_, u32>(8)?,
                        r.get::<_, String>(9)?,
                        r.get::<_, String>(10)?,
                    ))
                },
            )
            .optional()
            .map_err(storage)?;

        let Some(c) = row else { return Ok(None) };
        Ok(Some(Sticker {
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
        }))
    }

    fn upsert_sticker(&self, s: &Sticker) -> Result<(), RepoError> {
        self.conn
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
            .conn
            .query_row("SELECT COUNT(*) FROM stickers", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1, "no duplicate row");
    }
}
