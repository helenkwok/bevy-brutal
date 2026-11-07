//! Fetch and cache open geodata for bevy-brutal: OpenStreetMap extracts from Geofabrik and
//! elevation tiles from the AWS Terrain Tiles open dataset (Tilezen/Mapzen "joerd").
//!
//! Only sources that allow commercial use with attribution are supported. There are no Google
//! (or other scrape-only) sources here, by design. See `README.md` for licence notes: the code
//! is MIT OR Apache-2.0, the downloaded **data is not** (OSM is ODbL 1.0, terrain is a mix of
//! public-domain and attribution licences). [`attribution`] has the credit strings.
//!
//! All network access goes through the [`Fetch`] trait, so callers (and tests) can swap the
//! HTTP client. The default client, [`HttpFetcher`], is behind the `net` feature.
//!
//! Cache layout (under [`CacheDir`]):
//! ```text
//! <cache>/osm/geofabrik/index-v1.json                     Geofabrik region index (GeoJSON)
//! <cache>/osm/geofabrik/<region path>-latest.osm.pbf      e.g. europe/monaco-latest.osm.pbf
//! <cache>/osm/geofabrik/<region path>-latest.osm.pbf.md5
//! <cache>/osm/geofabrik/<region path>-latest.osm.pbf.source.txt   URL, fetch time, md5
//! <cache>/osm/ATTRIBUTION.txt
//! <cache>/terrain/aws/terrarium/<z>/<x>/<y>.png
//! <cache>/terrain/aws/geotiff/<z>/<x>/<y>.tif
//! <cache>/terrain/aws/skadi/<N00>/<N00E000>.hgt.gz
//! <cache>/terrain/ATTRIBUTION.txt
//! <cache>/terrain/grids/<name>/{height.npy,height.json,height16.png,hillshade.png,ATTRIBUTION.txt}
//! <cache>/osm/derived/<name>/{roads,buildings,water,landuse}.geojson   (ODbL; ship the folder whole)
//! <cache>/osm/derived/<name>/{manifest.json,ATTRIBUTION.txt,LICENSE.txt}
//! ```
//!
//! [`grid`] stitches Terrarium tiles into a local-ENU height grid (via `coords::Enu`).
//! `osm_extract` (feature `pbf`) turns a `.osm.pbf` into roads, buildings, water and landuse layers.

pub mod attribution;
pub mod bbox;
mod cache;
pub mod derived;
mod fetch;
pub mod grid;
pub mod osm;
#[cfg(feature = "pbf")]
pub mod osm_extract;
pub mod terrain;

pub use bbox::{BBox, TileId};
pub use cache::CacheDir;
pub use fetch::Fetch;
#[cfg(feature = "net")]
pub use fetch::HttpFetcher;

/// Errors from this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid bounding box: {0}")]
    BadBBox(String),
    #[error("invalid argument: {0}")]
    BadArg(String),
    #[error("HTTP {status} for {url}")]
    Status { url: String, status: u16 },
    #[error("network error for {url}: {msg}")]
    Net { url: String, msg: String },
    #[error("checksum mismatch for {path}: expected {expected}, got {actual}")]
    Checksum { path: String, expected: String, actual: String },
    #[error("Geofabrik index: {0}")]
    Index(String),
    #[error("no Geofabrik region covers {0}")]
    NoRegion(String),
    #[error("{what}: {count} tiles exceeds the limit of {limit} (raise it explicitly if you mean it)")]
    TooMany { what: &'static str, count: usize, limit: usize },
    #[error("decode error: {0}")]
    Decode(String),
    #[error("I/O error at {path}: {source}")]
    Io { path: String, source: std::io::Error },
}

pub type Result<T> = std::result::Result<T, Error>;

/// User-Agent sent with every request (Geofabrik and AWS ask clients to identify themselves).
pub const USER_AGENT: &str = concat!("bevy-brutal-geo-fetch/", env!("CARGO_PKG_VERSION"));
