# geo-fetch

Fetch and cache open geodata for bevy-brutal:

- **OpenStreetMap** extracts (`.osm.pbf`) from [Geofabrik](https://download.geofabrik.de/), md5-verified.
- **Elevation** tiles from the [AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/) open dataset (Tilezen joerd output).
- **Height grids**: Terrarium tiles decoded and stitched into one f32 grid in a local ENU frame (`geo_fetch::grid`, `geo-fetch terrain-grid`).
- **OSM layers**: roads, buildings, water and landuse extracted from a `.osm.pbf` into GeoJSON, in a separate ODbL folder (`geo_fetch::osm_extract`, `geo-fetch osm-extract`).
- **Attribution** strings for both (`geo_fetch::attribution`, `geo-fetch credits`).

Only sources that allow commercial use with attribution are supported. **There are no Google sources
(Street View, satellite tiles, 3D meshes) and none should be added**: their terms forbid bulk download
and baking into game assets. See `out/quality/ALEPH_licence_audit.md`. This crate is a clean
rewrite; no code from Aleph (or any other downloader) was copied.

## Licence: code vs data

- The **crate source** is MIT OR Apache-2.0, like the rest of the workspace.
- The **data it downloads is not**, and nothing here relicenses it:
  - OSM data: **ODbL 1.0**, "© OpenStreetMap contributors". Rendered images and meshes shown in
    the game are a *Produced Work* and need attribution only. If you **ship an OSM-derived
    database file** (for example a baked roads or buildings file under `assets/osm/`), that file must
    be offered under ODbL with `attribution::OSM_DERIVED_DB_NOTICE` next to it. Keep such files in their own
    folder with their own LICENSE. This does not affect the Rust code.
  - Terrain: a mix of public-domain (USGS, NOAA) and attribution licences (Copernicus EU-DEM,
    Geoscience Australia CC BY 4.0, Kartverket CC BY 4.0, LINZ CC BY 3.0 NZ, OGL Canada, UK EA OGL,
    INEGI, ArcticDEM). There is no share-alike. Ship the full `attribution::terrain_full()` list.
- Rust dependencies are permissive: ureq, rustls, ring, serde_json, md-5, clap, thiserror, png,
  osmpbf, rayon (MIT/Apache/ISC/BSD), plus webpki-roots (Mozilla CA list, CDLA-Permissive-2.0), and
  the workspace crate `coords`.

### Shipping OSM-derived layers (ODbL)

`osm-extract` writes each extract into its own folder, `<cache>/osm/derived/<name>/`, which holds the
layer GeoJSON files, `manifest.json`, `ATTRIBUTION.txt` (credit + derived-database notice) and
`LICENSE.txt` (ODbL 1.0 pointer + notice). To ship one, copy the **whole folder** as is, e.g. to
`assets/osm/<name>/`, keeping `ATTRIBUTION.txt` and `LICENSE.txt` next to the data. Don't merge it
into MIT/Apache-licensed asset folders. Put "© OpenStreetMap contributors" in the game credits
(`geo-fetch credits --osm`). Height grids have no share-alike; ship `ATTRIBUTION.txt` from the grid folder.

Not legal advice.

## CLI

```sh
cargo run -p geo-fetch -- --help
cargo run -p geo-fetch -- osm --region monaco                 # id, or a path like europe/monaco
cargo run -p geo-fetch -- osm --bbox 138.5,-35.0,138.7,-34.8 --dry-run   # -> south-australia
cargo run -p geo-fetch -- regions australia                   # list Geofabrik regions
cargo run -p geo-fetch -- terrain --bbox 138.5,-35.0,138.7,-34.8 --zoom 8..11
cargo run -p geo-fetch -- terrain --bbox 7.40,43.72,7.44,43.75 --format skadi
cargo run -p geo-fetch -- terrain-grid --bbox 138.5,-35.0,138.7,-34.8 --zoom 12           # -> terrain/grids/<name>/
cargo run -p geo-fetch -- terrain-grid --bbox 138.5,-35.0,138.7,-34.8 --zoom 14 --spacing 30 --height-ref enu-up
cargo run -p geo-fetch -- osm-extract --region monaco                                    # -> osm/derived/monaco/
cargo run -p geo-fetch -- osm-extract --bbox 138.5,-35.0,138.7,-34.8 --name adelaide --layers roads,buildings
cargo run -p geo-fetch -- credits            # paste into CREDITS.md (--osm / --terrain / --db-notice)
```

A bbox is `west,south,east,north` in degrees. Boxes that cross the antimeridian are rejected; split
them in two. If a bbox starts with `-`, write `--bbox=-1.5,...`. Terrain requests over 1024 tiles fail
unless you pass `--max-tiles`. `--jobs` (default 4) sets the number of parallel downloads. `--refresh`
re-downloads data that is already cached. `--dry-run` prints the URLs and cache paths without
downloading anything.

`terrain-grid` flags: `--zoom` (source zoom, default 12), `--spacing <m>` (default: the source pixel
size at the bbox centre, floored to whole metres), `--origin lat,lon` (ENU origin, default bbox
centre), `--height-ref dem|enu-up`, `--fallback-zooms N` (default 4), `--no-fill`, `--no-supersample`,
`--offline` (cache only; missing tiles become holes), `--out`, `--name`, `--jobs`, `--max-tiles`.

`osm-extract` flags: `--region` or `--pbf <file>` (one or the other), `--bbox` (filter; with no
`--region`/`--pbf` it also picks the region), `--layers roads,buildings,water,landuse`, `--offline`,
`--out`, `--name`.

### Height grid details

- Every sample sits on a regular lattice in the ENU tangent plane at the origin, `(x0 + col·s, y0 + row·s, 0)`
  with row 0 = **south**. It is mapped to lat/lon with `coords::Enu::plane_to_geodetic`, and from there
  to global Web Mercator pixel space at the chosen zoom. Then it is sampled bilinearly. Because
  sampling happens in the global mosaic, there are **no tile seams**. The lattice snaps to multiples
  of the spacing, so grids with the same origin and spacing share sample positions.
- If the spacing is coarser than the source pixels, each sample averages k×k taps (`--no-supersample` turns this off).
- **Holes**: missing, 404 or undecodable tiles fall back to coarser zooms (up to `--fallback-zooms`).
  Anything still missing is filled by iterative neighbour averaging (`--no-fill` leaves NaN).
- Heights are DEM values (orthometric, about EGM96). `enu-up` subtracts the earth-curvature drop, so
  `(x, y, h)` is a true ENU point. Over 20 km the difference is about 8 m at the edges.

### OSM extract details

- osmpbf reads the file in 3 parallel passes (rayon): relations, then the ways they need, then the nodes.
- Classification is per layer (`osm_extract::classify`), with a per-layer tag whitelist (`Layer::kept_tags`,
  listed in `manifest.json`).
  - **roads**: `highway=<road/path class>` lines (not `area=yes`).
  - **buildings**: `building=*` except `no`.
  - **water**: `natural=water|coastline`, `waterway=riverbank|dock` (areas) or river/stream/canal/… lines, `landuse=reservoir|basin`.
  - **landuse**: `landuse=*` (except reservoir/basin), else selected `leisure=*` (parks, pitches, …), else selected `natural=*` (wood, scrub, …).
- Closed area ways become Polygons. `type=multipolygon` relations are assembled from their member
  ways into MultiPolygons, with RFC 7946 winding. Old-style multipolygons (tags on the outer way) and
  relations with members outside the extract are counted and skipped.
- The bbox is an **intersect filter, not a clip**: features whose bounds touch the bbox are kept whole.
- Coordinates are WGS84 lon/lat (7 decimals). Use the `suggested_enu_origin_lat_lon` in `manifest.json`,
  which is the bbox centre and the same default origin as `terrain-grid`, with `Geometry::to_enu`
  (or `coords::Enu`) to put OSM on a height grid built from the same `--bbox`.

## Cache layout

The root is `--cache-dir`, else `$GEO_FETCH_CACHE`, else `<repo>/../data/geo`
(`/workspace/data/geo` here). This matches `tools/fetch_earth_data.sh` → `/workspace/data/earth`
and keeps downloaded data outside the source tree.

```text
<cache>/osm/ATTRIBUTION.txt                                 OSM credit + ODbL derived-DB notice
<cache>/osm/geofabrik/index-v1.json                         Geofabrik region index (GeoJSON, ~4 MB)
<cache>/osm/geofabrik/<path>-latest.osm.pbf                 e.g. europe/monaco-latest.osm.pbf
<cache>/osm/geofabrik/<path>-latest.osm.pbf.md5             Geofabrik checksum file
<cache>/osm/geofabrik/<path>-latest.osm.pbf.source.txt      URL, fetch time (unix), md5, bytes, licence
<cache>/terrain/ATTRIBUTION.txt                             full terrain source list
<cache>/terrain/aws/terrarium/<z>/<x>/<y>.png               mirrors the bucket paths 1:1
<cache>/terrain/aws/normal/<z>/<x>/<y>.png
<cache>/terrain/aws/geotiff/<z>/<x>/<y>.tif
<cache>/terrain/aws/skadi/<N00>/<N00E000>.hgt.gz
<cache>/terrain/grids/<name>/height.npy                     f32 [rows, cols], row 0 = south, NaN = hole
<cache>/terrain/grids/<name>/height.json                    frame (ENU origin, ECEF axes), lattice, stats, report
<cache>/terrain/grids/<name>/height16.png                   16-bit grey, h = min + v/65535·(max−min); north up
<cache>/terrain/grids/<name>/hillshade.png                  8-bit preview, north up
<cache>/terrain/grids/<name>/ATTRIBUTION.txt
<cache>/osm/derived/<name>/{roads,buildings,water,landuse}.geojson   ODbL: ship this folder as a unit
<cache>/osm/derived/<name>/manifest.json                    source pbf + md5, bbox rule, counts, kept tags
<cache>/osm/derived/<name>/ATTRIBUTION.txt
<cache>/osm/derived/<name>/LICENSE.txt
```

Downloads go to `<file>.part` first and are then renamed into place, so an interrupted run never
leaves a truncated file that a later run would treat as cached. On later runs, a cached extract is
reused only if it still matches its `.md5`. A tile is reused if its file exists and is not empty.

## Sources and URL schemes

| Data | URL | Notes |
|---|---|---|
| Geofabrik index | `https://download.geofabrik.de/index-v1.json` | `properties.{id,parent,name,urls.pbf}` + boundary polygon |
| Geofabrik extract | `https://download.geofabrik.de/<path>-latest.osm.pbf` (+ `.md5`) | Rebuilt daily. The login-only `osm-internal` mirror is never used |
| Terrarium | `https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png` | 256² PNG, Web Mercator, z 0–15, `h = R·256 + G + B/256 − 32768` m (`terrain::decode_terrarium`) |
| Normal | `…/normal/{z}/{x}/{y}.png` | RGB = normal, A = quantised height |
| GeoTIFF | `…/geotiff/{z}/{x}/{y}.tif` | 512² float, covers the z−1 footprint of a 256² tile |
| Skadi | `…/skadi/{N00}/{N00E000}.hgt.gz` | 1°×1° SRTM-style HGT, 3601², big-endian i16, void −32768 |

`--bbox` region choice for OSM picks the **smallest Geofabrik region whose polygon bbox contains the
query bbox**. Because it compares bboxes, not polygons, a bbox on a coast or border can pick a
region that does not quite cover it. Pass `--region` when that matters.

## Library

```rust
use geo_fetch::{BBox, CacheDir, HttpFetcher, osm, terrain};
let cache = CacheDir::from_env();
let http = HttpFetcher::default();                    // feature "net" (default)
let e = osm::fetch_extract(&http, &cache, "europe/monaco", Default::default())?;
let bbox: BBox = "138.5,-35.0,138.7,-34.8".parse()?;
let r = terrain::fetch_tiles(&http, &cache, &terrain::TileRequest::new(terrain::Format::Terrarium, bbox, 8..=11))?;

// Height grid (feature "net" only needed for downloading; pass None for cache-only).
let mut spec = geo_fetch::grid::GridSpec::new(bbox, 12);
spec.height_ref = geo_fetch::grid::HeightRef::EnuUp;
let (grid, report) = geo_fetch::grid::build_grid(Some(&http), &cache, &spec)?;
let h = grid.sample(100.0, -250.0);                    // metres east/north of the origin
let files = geo_fetch::grid::write_grid(&grid, &report, &spec, &geo_fetch::grid::grid_dir(&cache, "adelaide"))?;

// OSM layers (feature "pbf", default).
use geo_fetch::osm_extract::{self, ExtractSpec};
let spec = ExtractSpec { bbox: Some(bbox), ..Default::default() };
let (features, rep) = osm_extract::extract(&e.path, &spec)?;
let credits = geo_fetch::attribution::credits_block(geo_fetch::attribution::Used { osm: true, terrain: true });
```

All network access goes through the `Fetch` trait, so a runtime streamer can plug in its own
client, and the unit tests use an in-memory mock. Features: `net` (ureq client, default), `pbf`
(osmpbf + rayon, default), `cli` (the binary, default), and `live-tests`. The `grid` module works without
`net`: you can build from cache, or from your own `Fetch`.

## Tests

```sh
cargo test -p geo-fetch                                   # offline: URL/path builders, attribution, tile maths, mock HTTP,
                                                          # grid seams/holes/supersample vs analytic surfaces, npy/png
                                                          # round trips, OSM classify/rings + a hand-encoded PBF end to end
cargo test -p coords                                      # WGS84 geodetic/ECEF/ENU
cargo test -p geo-fetch --features live-tests --test live # hits Geofabrik + AWS (~1 MB, Monaco): fetch, grid, extract
```

CI (`cargo test --workspace`) never touches the network.

## TODO (next steps, deliberately not half-done here)

- [x] PBF parsing into roads, buildings, water and landuse in a separate ODbL folder (`osm-extract`).
- [x] Terrarium decode and mosaic into a local ENU height grid with holes and seams handled (`terrain-grid`).
- [ ] **Meshing**: turn height grids into chunks or heightfields for `crates/terrain` / `crates/planet`, with skirts and LOD.
- [ ] **Draping**: put OSM features on the height grid (road ribbons, building extrusion from `height` / `building:levels`).
- [ ] Geometric clipping of OSM features to the bbox (only an intersect filter for now). Old-style multipolygons.
- [ ] A compact binary output for the game (e.g. FlatGeobuf, or pre-projected ENU f32) next to or instead of GeoJSON. GeoJSON for a large city is about 75 MB.
- [ ] More layers: railways, power, barriers, POIs.
- [ ] Geoid (EGM96) → ellipsoidal heights, to get exact ECEF placement on `crates/planet` (error up to about ±100 m globally; about +20 m at Adelaide; it is a near-constant offset over a city).
- [ ] Polygon-accurate Geofabrik region choice: point-in-polygon on `index-v1.json` instead of bbox containment.
- [ ] Optional `.osc` incremental updates (`urls.updates`) instead of re-downloading `-latest` each day.
- [ ] Retry/backoff and an `If-Modified-Since` check for `--refresh`, plus a download progress indicator for multi-GB extracts.
- [ ] Imagery sources from the audit, also attribution-only: Sentinel-2 (Copernicus Data Space; needs an account/token) and Landsat (USGS; needs an account). These belong in separate modules.
- [ ] Runtime streaming: an async `Fetch` implementation for in-engine tile paging.
