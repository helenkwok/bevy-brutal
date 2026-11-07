//! OpenStreetMap extracts from Geofabrik (<https://download.geofabrik.de/>).
//!
//! Data licence: ODbL 1.0, "© OpenStreetMap contributors" (see [`crate::attribution`]).
//! Only the public `download.geofabrik.de` server is used, never the login-only
//! `osm-internal` mirror (that one carries personal metadata).
//!
//! URL scheme (documented by Geofabrik, stable for years):
//! - region index: `https://download.geofabrik.de/index-v1.json` (GeoJSON, ~4 MB, one feature per
//!   region with `properties.id`, `parent`, `name`, `urls.pbf` and a (Multi)Polygon boundary)
//! - extract: `https://download.geofabrik.de/<path>-latest.osm.pbf`, e.g. `europe/monaco`
//! - checksum: the same URL + `.md5` (`"<hex>  <file name>"`)

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use md5::{Digest, Md5};

use crate::attribution;
use crate::cache::{create_parent, io_err, is_cached, part_path, unix_now, write_atomic};
use crate::{BBox, CacheDir, Error, Fetch, Result};

pub const GEOFABRIK_BASE: &str = "https://download.geofabrik.de/";
pub const INDEX_URL: &str = "https://download.geofabrik.de/index-v1.json";

/// One downloadable region from the Geofabrik index.
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    /// Short id, e.g. `monaco`, `south-australia`.
    pub id: String,
    pub name: String,
    pub parent: Option<String>,
    /// Path below [`GEOFABRIK_BASE`] without the `-latest.osm.pbf` suffix, e.g. `europe/monaco`.
    pub path: String,
    /// Bounding box of the region's boundary polygon (Geofabrik polygons include a small buffer).
    pub bbox: Option<BBox>,
}

/// Check a region path like `europe/monaco` or `australia-oceania/australia/south-australia`:
/// lowercase ASCII letters, digits, `-` and `_` segments separated by `/`.
pub fn validate_region_path(path: &str) -> Result<()> {
    let ok = !path.is_empty()
        && path.split('/').all(|seg| {
            !seg.is_empty()
                && seg.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
        });
    if ok {
        Ok(())
    } else {
        Err(Error::BadArg(format!("bad Geofabrik region path {path:?} (want e.g. europe/monaco)")))
    }
}

/// `https://download.geofabrik.de/<path>-latest.osm.pbf`
pub fn extract_url(path: &str) -> Result<String> {
    validate_region_path(path)?;
    Ok(format!("{GEOFABRIK_BASE}{path}-latest.osm.pbf"))
}

/// `<cache>/osm/geofabrik/<path>-latest.osm.pbf`
pub fn extract_path(cache: &CacheDir, path: &str) -> Result<PathBuf> {
    validate_region_path(path)?;
    Ok(cache.osm_dir().join("geofabrik").join(format!("{path}-latest.osm.pbf")))
}

pub fn index_path(cache: &CacheDir) -> PathBuf {
    cache.osm_dir().join("geofabrik").join("index-v1.json")
}

/// Parse `index-v1.json`. Features without a public `urls.pbf` are skipped.
pub fn parse_index(json: &[u8]) -> Result<Vec<Region>> {
    let v: serde_json::Value = serde_json::from_slice(json).map_err(|e| Error::Index(e.to_string()))?;
    let feats = v.get("features").and_then(|f| f.as_array()).ok_or_else(|| Error::Index("no features array".into()))?;
    let mut out = Vec::with_capacity(feats.len());
    for f in feats {
        let p = &f["properties"];
        let (Some(id), Some(pbf)) = (p["id"].as_str(), p["urls"]["pbf"].as_str()) else { continue };
        let Some(rel) = pbf.strip_prefix(GEOFABRIK_BASE).and_then(|r| r.strip_suffix("-latest.osm.pbf")) else {
            continue;
        };
        if validate_region_path(rel).is_err() {
            continue;
        }
        out.push(Region {
            id: id.to_string(),
            name: p["name"].as_str().unwrap_or(id).to_string(),
            parent: p["parent"].as_str().map(str::to_string),
            path: rel.to_string(),
            bbox: geometry_bbox(&f["geometry"]),
        });
    }
    if out.is_empty() {
        return Err(Error::Index("index has no downloadable regions".into()));
    }
    Ok(out)
}

/// Bounding box of every coordinate in a GeoJSON geometry.
fn geometry_bbox(g: &serde_json::Value) -> Option<BBox> {
    fn walk(v: &serde_json::Value, b: &mut [f64; 4]) {
        let Some(a) = v.as_array() else { return };
        if let (Some(x), Some(y)) = (a.first().and_then(|x| x.as_f64()), a.get(1).and_then(|y| y.as_f64())) {
            b[0] = b[0].min(x);
            b[1] = b[1].min(y);
            b[2] = b[2].max(x);
            b[3] = b[3].max(y);
        } else {
            a.iter().for_each(|c| walk(c, b));
        }
    }
    let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    walk(&g["coordinates"], &mut b);
    BBox::new(b[0].max(-180.0), b[1].max(-90.0), b[2].min(180.0), b[3].min(90.0)).ok()
}

/// Find a region by id (`monaco`) or by path (`europe/monaco`).
pub fn find_region<'a>(regions: &'a [Region], key: &str) -> Option<&'a Region> {
    regions.iter().find(|r| r.path == key).or_else(|| regions.iter().find(|r| r.id == key))
}

/// The smallest region (by bbox area) whose boundary bbox contains `bbox`.
///
/// This is a bbox test, not a polygon test, so a coastal or border bbox can pick a region whose
/// polygon does not quite cover it. Pass an explicit region when that matters.
pub fn smallest_region_for(regions: &[Region], bbox: &BBox) -> Option<Region> {
    regions
        .iter()
        .filter(|r| r.bbox.is_some_and(|rb| rb.contains(bbox)))
        .min_by(|a, b| a.bbox.unwrap().area_deg2().total_cmp(&b.bbox.unwrap().area_deg2()))
        .cloned()
}

/// Load the cached region index, downloading it if missing (or if `refresh`).
pub fn load_index(fetch: &dyn Fetch, cache: &CacheDir, refresh: bool) -> Result<Vec<Region>> {
    let p = index_path(cache);
    if refresh || !is_cached(&p) {
        let bytes = fetch.get(INDEX_URL)?;
        parse_index(&bytes)?; // do not cache garbage
        write_atomic(&p, &bytes)?;
    }
    let bytes = fs::read(&p).map_err(io_err(&p))?;
    parse_index(&bytes)
}

/// What [`fetch_extract`] produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Extract {
    pub path: PathBuf,
    pub url: String,
    pub md5: String,
    pub bytes: u64,
    /// False when the cached file was reused.
    pub downloaded: bool,
}

/// Options for [`fetch_extract`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ExtractOptions {
    /// Re-download even if a verified copy is cached (Geofabrik rebuilds `-latest` daily).
    pub refresh: bool,
}

/// Parse a Geofabrik `.md5` file: `"<32 hex>  <name>"`.
pub fn parse_md5_file(text: &str) -> Result<String> {
    let h = text.split_whitespace().next().unwrap_or("").to_ascii_lowercase();
    if h.len() == 32 && h.bytes().all(|b| b.is_ascii_hexdigit()) {
        Ok(h)
    } else {
        Err(Error::BadArg(format!("unexpected .md5 contents {text:?}")))
    }
}

/// Download `<path>-latest.osm.pbf` into the cache and verify it against Geofabrik's `.md5`.
/// A cached copy is reused if its `.md5` sidecar matches its contents.
pub fn fetch_extract(fetch: &dyn Fetch, cache: &CacheDir, region_path: &str, opt: ExtractOptions) -> Result<Extract> {
    let url = extract_url(region_path)?;
    let dest = extract_path(cache, region_path)?;
    let md5_path = sidecar(&dest, "md5");

    if !opt.refresh && is_cached(&dest) && is_cached(&md5_path) {
        let expected = parse_md5_file(&fs::read_to_string(&md5_path).map_err(io_err(&md5_path))?)?;
        let (actual, bytes) = md5_file(&dest)?;
        if actual == expected {
            return Ok(Extract { path: dest, url, md5: actual, bytes, downloaded: false });
        }
    }

    let md5_text = String::from_utf8_lossy(&fetch.get(&format!("{url}.md5"))?).into_owned();
    let expected = parse_md5_file(&md5_text)?;

    create_parent(&dest)?;
    let part = part_path(&dest);
    let mut hasher = HashingWriter {
        inner: std::io::BufWriter::new(fs::File::create(&part).map_err(io_err(&part))?),
        md5: Md5::new(),
    };
    let bytes = fetch.get_to(&url, &mut hasher)?;
    hasher.inner.flush().map_err(io_err(&part))?;
    let HashingWriter { inner, md5 } = hasher;
    inner
        .into_inner()
        .map_err(|e| Error::Io { path: part.display().to_string(), source: e.into_error() })?
        .sync_all()
        .map_err(io_err(&part))?;
    let actual = hex(&md5.finalize());
    if actual != expected {
        let _ = fs::remove_file(&part);
        return Err(Error::Checksum { path: dest.display().to_string(), expected, actual });
    }
    fs::rename(&part, &dest).map_err(io_err(&dest))?;
    write_atomic(&md5_path, md5_text.as_bytes())?;
    let src = format!(
        "url: {url}\nfetched_unix: {}\nmd5: {actual}\nbytes: {bytes}\nlicence: ODbL 1.0\ncredit: {}\n",
        unix_now(),
        attribution::OSM_SHORT
    );
    write_atomic(&sidecar(&dest, "source.txt"), src.as_bytes())?;
    write_osm_attribution(cache)?;
    Ok(Extract { path: dest, url, md5: actual, bytes, downloaded: true })
}

/// Write `<cache>/osm/ATTRIBUTION.txt` so the data folder carries its own licence note.
pub fn write_osm_attribution(cache: &CacheDir) -> Result<()> {
    let text = format!(
        "{}\n{}\n\n{}\n",
        attribution::OSM_FULL,
        attribution::GEOFABRIK_SOURCE,
        attribution::OSM_DERIVED_DB_NOTICE
    );
    write_atomic(&cache.osm_dir().join("ATTRIBUTION.txt"), text.as_bytes())
}

fn sidecar(p: &Path, ext: &str) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

struct HashingWriter<W: Write> {
    inner: W,
    md5: Md5,
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.md5.update(&buf[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn md5_file(p: &Path) -> Result<(String, u64)> {
    let mut f = fs::File::open(p).map_err(io_err(p))?;
    let mut h = HashingWriter { inner: std::io::sink(), md5: Md5::new() };
    let n = std::io::copy(&mut f, &mut h).map_err(io_err(p))?;
    Ok((hex(&h.md5.finalize()), n))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetch::mock::MockFetcher;

    const INDEX: &str = r#"{"type":"FeatureCollection","features":[
      {"type":"Feature","properties":{"id":"europe","name":"Europe","urls":{"pbf":"https://download.geofabrik.de/europe-latest.osm.pbf"}},
       "geometry":{"type":"Polygon","coordinates":[[[-35,30],[50,30],[50,82],[-35,82],[-35,30]]]}},
      {"type":"Feature","properties":{"id":"monaco","parent":"europe","name":"Monaco","urls":{"pbf":"https://download.geofabrik.de/europe/monaco-latest.osm.pbf","pbf-internal":"https://osm-internal.download.geofabrik.de/europe/monaco-latest-internal.osm.pbf"}},
       "geometry":{"type":"MultiPolygon","coordinates":[[[[7.40,43.72],[7.45,43.72],[7.45,43.76],[7.40,43.76],[7.40,43.72]]]]}},
      {"type":"Feature","properties":{"id":"internal-only","urls":{"pbf":"https://osm-internal.download.geofabrik.de/x-latest.osm.pbf"}},"geometry":null}
    ]}"#;

    fn tmp(name: &str) -> CacheDir {
        let d = std::env::temp_dir().join(format!("geo-fetch-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        CacheDir::new(d)
    }

    #[test]
    fn url_and_path_builders() {
        assert_eq!(extract_url("europe/monaco").unwrap(), "https://download.geofabrik.de/europe/monaco-latest.osm.pbf");
        assert_eq!(
            extract_url("australia-oceania/australia").unwrap(),
            "https://download.geofabrik.de/australia-oceania/australia-latest.osm.pbf"
        );
        let c = CacheDir::new("/c");
        assert_eq!(
            extract_path(&c, "europe/monaco").unwrap(),
            PathBuf::from("/c/osm/geofabrik/europe/monaco-latest.osm.pbf")
        );
        assert_eq!(index_path(&c), PathBuf::from("/c/osm/geofabrik/index-v1.json"));
        for bad in ["", "../etc", "europe//monaco", "Europe/Monaco", "a/b?c", "/abs", "x/"] {
            assert!(extract_url(bad).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn index_parsing_and_selection() {
        let r = parse_index(INDEX.as_bytes()).unwrap();
        assert_eq!(r.len(), 2, "internal-only region must be skipped");
        let m = find_region(&r, "monaco").unwrap();
        assert_eq!(m.path, "europe/monaco");
        assert_eq!(m.parent.as_deref(), Some("europe"));
        assert_eq!(m.bbox, Some(BBox { west: 7.40, south: 43.72, east: 7.45, north: 43.76 }));
        assert_eq!(find_region(&r, "europe/monaco").unwrap().id, "monaco");
        let inside: BBox = "7.41,43.73,7.44,43.75".parse().unwrap();
        assert_eq!(smallest_region_for(&r, &inside).unwrap().id, "monaco");
        let paris: BBox = "2.2,48.8,2.4,48.9".parse().unwrap();
        assert_eq!(smallest_region_for(&r, &paris).unwrap().id, "europe");
        let adelaide: BBox = "138.5,-35.0,138.7,-34.8".parse().unwrap();
        assert!(smallest_region_for(&r, &adelaide).is_none());
    }

    #[test]
    fn md5_parse() {
        assert_eq!(
            parse_md5_file("CC6D095B83818C70084742BE2CA95FB4  monaco-latest.osm.pbf\n").unwrap(),
            "cc6d095b83818c70084742be2ca95fb4"
        );
        assert!(parse_md5_file("<html>").is_err());
    }

    #[test]
    fn fetch_extract_verifies_and_caches() {
        let body = b"fake pbf bytes".to_vec();
        let sum = hex(&Md5::digest(&body));
        let url = "https://download.geofabrik.de/europe/monaco-latest.osm.pbf";
        let f = MockFetcher::default()
            .with(url, body.clone())
            .with(&format!("{url}.md5"), format!("{sum}  monaco-latest.osm.pbf\n"));
        let c = tmp("osm");
        let e = fetch_extract(&f, &c, "europe/monaco", ExtractOptions::default()).unwrap();
        assert!(e.downloaded);
        assert_eq!(e.bytes, body.len() as u64);
        assert_eq!(fs::read(&e.path).unwrap(), body);
        assert!(c.osm_dir().join("ATTRIBUTION.txt").exists());
        let src = fs::read_to_string(sidecar(&e.path, "source.txt")).unwrap();
        assert!(src.contains("licence: ODbL 1.0") && src.contains(url));
        assert!(!part_path(&e.path).exists());

        // Second call: served from cache, no new requests.
        let n = f.requests().len();
        let e2 = fetch_extract(&f, &c, "europe/monaco", ExtractOptions::default()).unwrap();
        assert!(!e2.downloaded);
        assert_eq!(f.requests().len(), n);
        let _ = fs::remove_dir_all(c.root());
    }

    #[test]
    fn fetch_extract_rejects_bad_checksum() {
        let url = "https://download.geofabrik.de/europe/monaco-latest.osm.pbf";
        let f = MockFetcher::default()
            .with(url, b"corrupt".to_vec())
            .with(&format!("{url}.md5"), "00000000000000000000000000000000  monaco-latest.osm.pbf");
        let c = tmp("osm-bad");
        let err = fetch_extract(&f, &c, "europe/monaco", ExtractOptions::default()).unwrap_err();
        assert!(matches!(err, Error::Checksum { .. }), "{err}");
        let dest = extract_path(&c, "europe/monaco").unwrap();
        assert!(!dest.exists() && !part_path(&dest).exists());
        let _ = fs::remove_dir_all(c.root());
    }

    #[test]
    fn load_index_caches() {
        let f = MockFetcher::default().with(INDEX_URL, INDEX);
        let c = tmp("index");
        assert_eq!(load_index(&f, &c, false).unwrap().len(), 2);
        assert_eq!(load_index(&f, &c, false).unwrap().len(), 2);
        assert_eq!(f.requests(), vec![INDEX_URL.to_string()]);
        let _ = fs::remove_dir_all(c.root());
    }
}
