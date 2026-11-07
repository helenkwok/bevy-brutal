//! Cache directory handling and atomic file writes.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::{Error, Result};

/// Environment variable that overrides the default cache root.
pub const CACHE_ENV: &str = "GEO_FETCH_CACHE";

/// Root of the on-disk cache. See the crate docs for the layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheDir(PathBuf);

impl CacheDir {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self(root.into())
    }

    /// `$GEO_FETCH_CACHE` if set, otherwise [`CacheDir::default_root`].
    pub fn from_env() -> Self {
        match std::env::var_os(CACHE_ENV) {
            Some(v) if !v.is_empty() => Self(PathBuf::from(v)),
            _ => Self(Self::default_root()),
        }
    }

    /// `data/geo` next to the repository checkout (`<repo>/../data/geo`), matching the other
    /// fetch tools (`tools/fetch_earth_data.sh` uses `/workspace/data/earth`). Downloaded data is
    /// kept out of the source tree so it never ends up in a commit or the source tarball.
    pub fn default_root() -> PathBuf {
        // crates/geo-fetch -> crates -> <repo> -> <repo parent>
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let parent = manifest.ancestors().nth(3).unwrap_or(manifest);
        parent.join("data").join("geo")
    }

    pub fn root(&self) -> &Path {
        &self.0
    }

    pub fn osm_dir(&self) -> PathBuf {
        self.0.join("osm")
    }

    pub fn terrain_dir(&self) -> PathBuf {
        self.0.join("terrain")
    }
}

pub(crate) fn io_err(path: &Path) -> impl FnOnce(std::io::Error) -> Error + '_ {
    move |source| Error::Io { path: path.display().to_string(), source }
}

pub(crate) fn create_parent(path: &Path) -> Result<()> {
    if let Some(p) = path.parent() {
        fs::create_dir_all(p).map_err(io_err(p))?;
    }
    Ok(())
}

/// Path of the temporary file a download goes to before it is renamed into place.
pub(crate) fn part_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".part");
    PathBuf::from(s)
}

/// Write `bytes` to `path` via `<path>.part` + rename, so a crash never leaves a truncated file
/// that a later run would treat as cached.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    create_parent(path)?;
    let part = part_path(path);
    let mut f = fs::File::create(&part).map_err(io_err(&part))?;
    f.write_all(bytes).map_err(io_err(&part))?;
    f.sync_all().map_err(io_err(&part))?;
    drop(f);
    fs::rename(&part, path).map_err(io_err(path))
}

/// A cached file counts as present only if it exists and is non-empty.
pub(crate) fn is_cached(path: &Path) -> bool {
    fs::metadata(path).map(|m| m.is_file() && m.len() > 0).unwrap_or(false)
}

/// Seconds since the Unix epoch, for `.source.txt` sidecars.
pub(crate) fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_and_default_root() {
        let c = CacheDir::new("/tmp/geo");
        assert_eq!(c.osm_dir(), PathBuf::from("/tmp/geo/osm"));
        assert_eq!(c.terrain_dir(), PathBuf::from("/tmp/geo/terrain"));
        let d = CacheDir::default_root();
        assert!(d.ends_with("data/geo"), "{d:?}");
        // Must be outside the crate and outside the repo checkout.
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap();
        assert!(!d.starts_with(repo), "{d:?} is inside {repo:?}");
    }

    #[test]
    fn part_path_appends_suffix() {
        assert_eq!(part_path(Path::new("a/b.osm.pbf")), PathBuf::from("a/b.osm.pbf.part"));
    }
}
