//! Elevation tiles from the AWS "Terrain Tiles" open dataset (Tilezen joerd output),
//! <https://registry.opendata.aws/terrain-tiles/>.
//!
//! Bucket: `https://elevation-tiles-prod.s3.amazonaws.com/` (public, no account, no request
//! signing). Formats, from tilezen/joerd `docs/formats.md`:
//!
//! | format | URL below the bucket | content |
//! |---|---|---|
//! | `terrarium` | `terrarium/{z}/{x}/{y}.png` | 256² PNG, Web Mercator, `h = R*256 + G + B/256 - 32768` m |
//! | `normal` | `normal/{z}/{x}/{y}.png` | 256² PNG, RGB = surface normal, A = quantised height |
//! | `geotiff` | `geotiff/{z}/{x}/{y}.tif` | 512² float GeoTIFF, Web Mercator (covers the `z-1` 256² footprint) |
//! | `skadi` | `skadi/{N00}/{N00E000}.hgt.gz` | 1°×1° SRTM-style HGT, 3601², big-endian i16, void = -32768 |
//!
//! Zoom 0..=15 (`terrarium`/`normal`/`geotiff`); x/y are standard slippy-map tile indices.
//! Licence: attribution only; see [`crate::attribution::TERRAIN_SOURCES`].

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::attribution;
use crate::bbox::MAX_ZOOM;
use crate::cache::{is_cached, write_atomic};
use crate::{BBox, CacheDir, Error, Fetch, Result, TileId};

pub const BUCKET_URL: &str = "https://elevation-tiles-prod.s3.amazonaws.com/";

/// Default cap on tiles per request, so a typo in a bbox or zoom cannot hammer the bucket.
pub const DEFAULT_MAX_TILES: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Format {
    #[default]
    Terrarium,
    Normal,
    Geotiff,
    Skadi,
}

impl Format {
    pub const ALL: [Format; 4] = [Format::Terrarium, Format::Normal, Format::Geotiff, Format::Skadi];

    pub fn name(self) -> &'static str {
        match self {
            Format::Terrarium => "terrarium",
            Format::Normal => "normal",
            Format::Geotiff => "geotiff",
            Format::Skadi => "skadi",
        }
    }

    fn ext(self) -> &'static str {
        match self {
            Format::Terrarium | Format::Normal => "png",
            Format::Geotiff => "tif",
            Format::Skadi => "hgt.gz",
        }
    }
}

impl std::str::FromStr for Format {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Format::ALL
            .into_iter()
            .find(|f| f.name() == s)
            .ok_or_else(|| Error::BadArg(format!("unknown terrain format {s:?} (terrarium|normal|geotiff|skadi)")))
    }
}

/// One downloadable terrain object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Key {
    /// Slippy-map tile (terrarium / normal / geotiff).
    Tile(TileId),
    /// 1° cell by its south-west corner (skadi).
    Cell { lat: i32, lon: i32 },
}

/// `N43E007`-style SRTM cell name for the cell whose south-west corner is (`lat`, `lon`).
pub fn skadi_name(lat: i32, lon: i32) -> String {
    let (ns, ew) = (if lat < 0 { 'S' } else { 'N' }, if lon < 0 { 'W' } else { 'E' });
    format!("{ns}{:02}{ew}{:03}", lat.unsigned_abs(), lon.unsigned_abs())
}

/// Path below the bucket root (and below `<cache>/terrain/aws/`).
pub fn rel_path(format: Format, key: Key) -> Result<String> {
    match (format, key) {
        (Format::Skadi, Key::Cell { lat, lon }) => {
            let name = skadi_name(lat, lon);
            Ok(format!("skadi/{}/{name}.hgt.gz", &name[..3]))
        }
        (Format::Skadi, Key::Tile(_)) | (_, Key::Cell { .. }) => {
            Err(Error::BadArg(format!("{} does not use {key:?} keys", format.name())))
        }
        (f, Key::Tile(t)) => {
            if t.z > MAX_ZOOM || t.x >= 1u32 << t.z || t.y >= 1u32 << t.z {
                return Err(Error::BadArg(format!("tile {t:?} out of range (z 0..={MAX_ZOOM})")));
            }
            Ok(format!("{}/{}/{}/{}.{}", f.name(), t.z, t.x, t.y, f.ext()))
        }
    }
}

pub fn url(format: Format, key: Key) -> Result<String> {
    Ok(format!("{BUCKET_URL}{}", rel_path(format, key)?))
}

pub fn cache_path(cache: &CacheDir, format: Format, key: Key) -> Result<PathBuf> {
    Ok(cache.terrain_dir().join("aws").join(rel_path(format, key)?))
}

/// Every key covering `bbox` for `zooms` (ignored for skadi), in a stable order.
pub fn keys_for(format: Format, bbox: &BBox, zooms: std::ops::RangeInclusive<u8>) -> Result<Vec<Key>> {
    if format == Format::Skadi {
        return Ok(bbox.degree_cells().into_iter().map(|(lat, lon)| Key::Cell { lat, lon }).collect());
    }
    if *zooms.end() > MAX_ZOOM || zooms.is_empty() {
        return Err(Error::BadArg(format!("zoom range {zooms:?} must be within 0..={MAX_ZOOM}")));
    }
    Ok(zooms.flat_map(|z| bbox.tiles(z)).map(Key::Tile).collect())
}

#[derive(Clone, Debug)]
pub struct TileRequest {
    pub format: Format,
    pub bbox: BBox,
    pub zooms: std::ops::RangeInclusive<u8>,
    /// Parallel downloads (AWS copes with more; keep it polite by default).
    pub jobs: usize,
    pub max_tiles: usize,
    pub refresh: bool,
}

impl TileRequest {
    pub fn new(format: Format, bbox: BBox, zooms: std::ops::RangeInclusive<u8>) -> Self {
        Self { format, bbox, zooms, jobs: 4, max_tiles: DEFAULT_MAX_TILES, refresh: false }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TileReport {
    pub requested: usize,
    pub downloaded: usize,
    pub cached: usize,
    pub bytes: u64,
    /// Keys the bucket answered 404/403 for (no data published there).
    pub missing: Vec<Key>,
}

/// Download every tile for `req` into the cache, skipping ones already cached.
/// Fails on the first transport error; 403/404 are recorded in [`TileReport::missing`].
pub fn fetch_tiles(fetch: &dyn Fetch, cache: &CacheDir, req: &TileRequest) -> Result<TileReport> {
    let keys = keys_for(req.format, &req.bbox, req.zooms.clone())?;
    if keys.len() > req.max_tiles {
        return Err(Error::TooMany { what: "terrain request", count: keys.len(), limit: req.max_tiles });
    }
    let report = Mutex::new(TileReport { requested: keys.len(), ..Default::default() });
    let first_err: Mutex<Option<Error>> = Mutex::new(None);
    let next = AtomicUsize::new(0);
    let work = || {
        loop {
            if first_err.lock().unwrap().is_some() {
                return;
            }
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(&key) = keys.get(i) else { return };
            if let Err(e) = fetch_one(fetch, cache, req, key, &report) {
                first_err.lock().unwrap().get_or_insert(e);
            }
        }
    };
    std::thread::scope(|s| {
        for _ in 0..req.jobs.clamp(1, 16) {
            s.spawn(work);
        }
    });
    if let Some(e) = first_err.into_inner().unwrap() {
        return Err(e);
    }
    write_terrain_attribution(cache)?;
    let mut r = report.into_inner().unwrap();
    r.missing.sort();
    Ok(r)
}

fn fetch_one(
    fetch: &dyn Fetch,
    cache: &CacheDir,
    req: &TileRequest,
    key: Key,
    report: &Mutex<TileReport>,
) -> Result<()> {
    let path = cache_path(cache, req.format, key)?;
    if !req.refresh && is_cached(&path) {
        report.lock().unwrap().cached += 1;
        return Ok(());
    }
    match fetch.get(&url(req.format, key)?) {
        Ok(bytes) => {
            write_atomic(&path, &bytes)?;
            let mut r = report.lock().unwrap();
            r.downloaded += 1;
            r.bytes += bytes.len() as u64;
            Ok(())
        }
        Err(Error::Status { status: 403 | 404, .. }) => {
            report.lock().unwrap().missing.push(key);
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Write `<cache>/terrain/ATTRIBUTION.txt`.
pub fn write_terrain_attribution(cache: &CacheDir) -> Result<()> {
    write_atomic(&cache.terrain_dir().join("ATTRIBUTION.txt"), attribution::terrain_full().as_bytes())
}

/// Decode one Terrarium pixel to metres: `R*256 + G + B/256 - 32768`.
pub fn decode_terrarium(r: u8, g: u8, b: u8) -> f32 {
    (r as f32 * 256.0 + g as f32 + b as f32 / 256.0) - 32768.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetch::mock::MockFetcher;

    #[test]
    fn urls_and_paths() {
        let t = Key::Tile(TileId { z: 10, x: 906, y: 617 });
        assert_eq!(
            url(Format::Terrarium, t).unwrap(),
            "https://elevation-tiles-prod.s3.amazonaws.com/terrarium/10/906/617.png"
        );
        assert_eq!(
            url(Format::Normal, t).unwrap(),
            "https://elevation-tiles-prod.s3.amazonaws.com/normal/10/906/617.png"
        );
        assert_eq!(
            url(Format::Geotiff, t).unwrap(),
            "https://elevation-tiles-prod.s3.amazonaws.com/geotiff/10/906/617.tif"
        );
        let c = Key::Cell { lat: -35, lon: 138 };
        assert_eq!(
            url(Format::Skadi, c).unwrap(),
            "https://elevation-tiles-prod.s3.amazonaws.com/skadi/S35/S35E138.hgt.gz"
        );
        assert_eq!(rel_path(Format::Skadi, Key::Cell { lat: 43, lon: 7 }).unwrap(), "skadi/N43/N43E007.hgt.gz");
        assert_eq!(rel_path(Format::Skadi, Key::Cell { lat: 0, lon: -1 }).unwrap(), "skadi/N00/N00W001.hgt.gz");
        assert_eq!(
            cache_path(&CacheDir::new("/c"), Format::Terrarium, t).unwrap(),
            PathBuf::from("/c/terrain/aws/terrarium/10/906/617.png")
        );
        assert!(url(Format::Skadi, t).is_err());
        assert!(url(Format::Terrarium, c).is_err());
        assert!(url(Format::Terrarium, Key::Tile(TileId { z: 16, x: 0, y: 0 })).is_err());
        assert!(url(Format::Terrarium, Key::Tile(TileId { z: 2, x: 4, y: 0 })).is_err());
        assert_eq!("geotiff".parse::<Format>().unwrap(), Format::Geotiff);
        assert!("google".parse::<Format>().is_err());
    }

    #[test]
    fn terrarium_decode() {
        // Worked example from joerd docs/formats.md: rgb(137, 219, 68) -> 2523.265625 m.
        assert_eq!(decode_terrarium(137, 219, 68), 2523.0 + 68.0 / 256.0);
        assert_eq!(decode_terrarium(128, 0, 0), 0.0);
        assert_eq!(decode_terrarium(85, 8, 0), -11000.0);
    }

    #[test]
    fn key_listing() {
        let b: BBox = "7.40,43.72,7.44,43.75".parse().unwrap();
        let k = keys_for(Format::Terrarium, &b, 0..=2).unwrap();
        assert_eq!(k.len(), 3);
        assert!(keys_for(Format::Terrarium, &b, 0..=16).is_err());
        assert_eq!(keys_for(Format::Skadi, &b, 0..=0).unwrap(), vec![Key::Cell { lat: 43, lon: 7 }]);
    }

    #[test]
    fn fetch_tiles_with_mock() {
        let b: BBox = "7.40,43.72,7.44,43.75".parse().unwrap();
        let mut f = MockFetcher::default();
        let keys = keys_for(Format::Terrarium, &b, 0..=3).unwrap();
        for k in &keys[..keys.len() - 1] {
            f = f.with(&url(Format::Terrarium, *k).unwrap(), b"png".to_vec());
        }
        let dir = std::env::temp_dir().join(format!("geo-fetch-test-terrain-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let c = CacheDir::new(&dir);
        let req = TileRequest::new(Format::Terrarium, b, 0..=3);
        let r = fetch_tiles(&f, &c, &req).unwrap();
        assert_eq!(r.requested, 4);
        assert_eq!((r.downloaded, r.cached, r.bytes), (3, 0, 9));
        assert_eq!(r.missing, vec![keys[3]]);
        assert!(c.terrain_dir().join("ATTRIBUTION.txt").exists());
        assert!(cache_path(&c, Format::Terrarium, keys[0]).unwrap().exists());

        let r2 = fetch_tiles(&f, &c, &req).unwrap();
        assert_eq!((r2.downloaded, r2.cached), (0, 3));

        let mut small = req.clone();
        small.max_tiles = 2;
        assert!(matches!(fetch_tiles(&f, &c, &small), Err(Error::TooMany { count: 4, limit: 2, .. })));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
