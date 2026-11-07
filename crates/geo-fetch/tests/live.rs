//! Live-network smoke tests. Off by default (CI never runs them):
//! `cargo test -p geo-fetch --features live-tests -- --test-threads 1`
//! Downloads ~1 MB (Monaco extract + a few dozen terrain tiles) into a temp dir, then decodes
//! the extract into layers and stitches a z14 height grid.
#![cfg(feature = "live-tests")]

use geo_fetch::terrain::{self, Format, TileRequest};
use geo_fetch::{BBox, CacheDir, HttpFetcher, grid, osm, osm_extract};

fn tmp(name: &str) -> CacheDir {
    let d = std::env::temp_dir().join(format!("geo-fetch-live-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    CacheDir::new(d)
}

#[test]
fn monaco_extract() {
    let c = tmp("osm");
    let e = osm::fetch_extract(&HttpFetcher::default(), &c, "europe/monaco", Default::default()).unwrap();
    assert!(e.downloaded && e.bytes > 100_000);
    let head = std::fs::read(&e.path).unwrap();
    // u32 BE header length, then BlobHeader field 1 (tag 0x0a, len 9) = "OSMHeader".
    assert_eq!(&head[4..15], b"\x0a\x09OSMHeader");
    // Decode it into layers.
    let spec = osm_extract::ExtractSpec::default();
    let (f, rep) = osm_extract::extract(&e.path, &spec).unwrap();
    let count = |l| rep.counts.iter().find(|c| c.0 == l).unwrap().1;
    assert!(count(osm_extract::Layer::Roads) > 1000 && count(osm_extract::Layer::Buildings) > 500, "{rep:?}");
    let dir = c.osm_dir().join("derived/monaco");
    let src = osm_extract::SourceInfo { pbf: e.path.clone(), url: Some(e.url), md5: Some(e.md5) };
    let files = osm_extract::write_layers(&f, &rep, &spec, &src, &dir).unwrap();
    assert!(files.license.exists() && files.attribution.exists() && files.manifest.exists());
    let _ = std::fs::remove_dir_all(c.root());
}

#[test]
fn monaco_terrain() {
    let c = tmp("terrain");
    let b: BBox = "7.40,43.72,7.44,43.75".parse().unwrap();
    let r =
        terrain::fetch_tiles(&HttpFetcher::default(), &c, &TileRequest::new(Format::Terrarium, b, 10..=11)).unwrap();
    assert_eq!(r.downloaded, r.requested);
    assert!(r.missing.is_empty());
    // Stitch z14 into a local grid; Monaco spans sea level to ~160 m (Mont Agel is outside).
    let spec = grid::GridSpec::new(b, 14);
    let (g, rep) = grid::build_grid(Some(&HttpFetcher::default()), &c, &spec).unwrap();
    assert!(rep.missing_tiles.is_empty() && rep.filled_samples == 0, "{rep:?}");
    let (lo, hi, _) = g.stats();
    assert!(lo < 5.0 && hi > 100.0 && hi < 800.0, "{lo}..{hi}");
    let _ = std::fs::remove_dir_all(c.root());
}
