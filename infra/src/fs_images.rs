//! Filesystem-backed `ImageStore`. Lays out files as `<root>/<pack>/<file_name>`
//! and returns the `<pack>/<file_name>` path for `Sticker::image_path`.

use sticker_core::error::StoreError;
use sticker_core::ports::ImageStore;
use std::path::PathBuf;

pub struct FsImageStore {
    root: PathBuf,
}

impl FsImageStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

fn io<E: std::fmt::Display>(e: E) -> StoreError {
    StoreError::Io(e.to_string())
}

impl ImageStore for FsImageStore {
    fn exists(&self, pack: &str, file_name: &str) -> bool {
        self.root.join(pack).join(file_name).is_file()
    }

    fn save(&self, pack: &str, file_name: &str, bytes: &[u8]) -> Result<String, StoreError> {
        let dir = self.root.join(pack);
        std::fs::create_dir_all(&dir).map_err(io)?;
        std::fs::write(dir.join(file_name), bytes).map_err(io)?;
        Ok(format!("{pack}/{file_name}"))
    }

    fn read(&self, image_path: &str) -> Result<Vec<u8>, StoreError> {
        std::fs::read(self.root.join(image_path)).map_err(io)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saves_then_reports_existing() {
        let tmp = std::env::temp_dir().join(format!("stickers-test-{}", std::process::id()));
        let store = FsImageStore::new(&tmp);

        assert!(!store.exists("packA", "a.webp"));
        let rel = store.save("packA", "a.webp", b"bytes").unwrap();
        assert_eq!(rel, "packA/a.webp");
        assert!(store.exists("packA", "a.webp"));
        assert_eq!(std::fs::read(tmp.join("packA/a.webp")).unwrap(), b"bytes");

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn reads_back_by_relative_path() {
        let tmp = std::env::temp_dir().join(format!("stickers-read-test-{}", std::process::id()));
        let store = FsImageStore::new(&tmp);

        let rel = store.save("packA", "a.webp", b"payload").unwrap();
        assert_eq!(store.read(&rel).unwrap(), b"payload");
        assert!(store.read("packA/missing.webp").is_err());

        std::fs::remove_dir_all(&tmp).ok();
    }
}
