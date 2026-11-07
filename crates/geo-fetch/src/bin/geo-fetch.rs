//! `geo-fetch`: download OSM extracts (Geofabrik) and AWS Terrain Tiles into the local cache.
//! Run `geo-fetch --help` or `geo-fetch <command> --help`.

use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use geo_fetch::attribution::{self, Used};
use geo_fetch::grid::{self, GridSpec, HeightRef};
use geo_fetch::osm_extract::{self, ExtractSpec, Layer, SourceInfo};
use geo_fetch::terrain::{self, Format, TileRequest};
use geo_fetch::{BBox, CacheDir, HttpFetcher, osm};

#[derive(Parser)]
#[command(
    name = "geo-fetch",
    version,
    about = "Fetch and cache open geodata: OSM extracts (Geofabrik, ODbL) and AWS Terrain Tiles elevation.",
    after_help = "Cache root: --cache-dir, else $GEO_FETCH_CACHE, else <repo>/../data/geo.\n\
                  Downloaded data is NOT MIT/Apache: run `geo-fetch credits` for the required attribution."
)]
struct Cli {
    /// Cache root (overrides $GEO_FETCH_CACHE).
    #[arg(long, global = true, value_name = "DIR")]
    cache_dir: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Download a Geofabrik .osm.pbf extract (md5-verified) for a region or the smallest region covering a bbox.
    Osm {
        /// Region id or path from the Geofabrik index, e.g. `monaco`, `europe/monaco`, `south-australia`.
        #[arg(long, conflicts_with = "bbox", required_unless_present = "bbox")]
        region: Option<String>,
        /// west,south,east,north in degrees; picks the smallest Geofabrik region whose bbox contains it.
        #[arg(long, allow_hyphen_values = true)]
        bbox: Option<BBox>,
        /// Re-download the region index and the extract even if cached.
        #[arg(long)]
        refresh: bool,
        /// Resolve the region and print URL + cache path without downloading the extract.
        #[arg(long)]
        dry_run: bool,
    },
    /// List Geofabrik regions (id, path, name), optionally filtered by a substring.
    Regions {
        filter: Option<String>,
        #[arg(long)]
        refresh: bool,
    },
    /// Download AWS Terrain Tiles covering a bbox for a zoom range.
    Terrain {
        /// west,south,east,north in degrees.
        #[arg(long, allow_hyphen_values = true)]
        bbox: BBox,
        /// Zoom level `N` or range `A..B` / `A-B` (0..=15). Ignored for --format skadi.
        #[arg(long, default_value = "10", value_parser = parse_zooms)]
        zoom: RangeInclusive<u8>,
        /// terrarium (PNG, default) | normal (PNG) | geotiff (512² float) | skadi (1° HGT).
        #[arg(long, default_value = "terrarium")]
        format: Format,
        /// Parallel downloads (1..=16).
        #[arg(long, default_value_t = 4)]
        jobs: usize,
        /// Refuse requests larger than this many tiles.
        #[arg(long, default_value_t = terrain::DEFAULT_MAX_TILES)]
        max_tiles: usize,
        /// Re-download tiles that are already cached.
        #[arg(long)]
        refresh: bool,
        /// Print the tile count, first URLs and cache paths without downloading.
        #[arg(long)]
        dry_run: bool,
    },
    /// Stitch Terrarium tiles into one f32 height grid in a local ENU frame (downloads missing tiles).
    ///
    /// Writes height.npy (f32, row 0 = south), height.json (frame, lattice, stats, provenance),
    /// height16.png + hillshade.png (north-up previews) and ATTRIBUTION.txt into
    /// <cache>/terrain/grids/<name>/ unless --out is given.
    TerrainGrid {
        /// west,south,east,north in degrees.
        #[arg(long, allow_hyphen_values = true)]
        bbox: BBox,
        /// Terrarium zoom to sample (0..=15).
        #[arg(long, default_value_t = 12, value_parser = clap::value_parser!(u8).range(0..=15))]
        zoom: u8,
        /// Lattice spacing in metres (default: source pixel size at the origin, floored to whole metres).
        #[arg(long)]
        spacing: Option<f64>,
        /// ENU origin as lat,lon in degrees (default: bbox centre).
        #[arg(long, allow_hyphen_values = true, value_parser = parse_latlon)]
        origin: Option<(f64, f64)>,
        /// dem (heights as published, ≈ MSL) | enu-up (minus tangent-plane curvature drop).
        #[arg(long, default_value = "dem")]
        height_ref: HeightRef,
        /// Coarser zooms to try for missing tiles before neighbour filling.
        #[arg(long, default_value_t = 4)]
        fallback_zooms: u8,
        /// Leave holes as NaN instead of filling them.
        #[arg(long)]
        no_fill: bool,
        /// Plain bilinear even when the spacing is coarser than the source pixels.
        #[arg(long)]
        no_supersample: bool,
        /// Use cached tiles only (no network); absent tiles become holes.
        #[arg(long)]
        offline: bool,
        /// Output directory (default <cache>/terrain/grids/<name>).
        #[arg(long, value_name = "DIR")]
        out: Option<PathBuf>,
        /// Output name below <cache>/terrain/grids/ (default derived from zoom, spacing and bbox).
        #[arg(long)]
        name: Option<String>,
        /// Parallel downloads (1..=16).
        #[arg(long, default_value_t = 4)]
        jobs: usize,
        /// Refuse requests needing more than this many tiles per zoom.
        #[arg(long, default_value_t = terrain::DEFAULT_MAX_TILES)]
        max_tiles: usize,
    },
    /// Decode an OSM extract into roads / buildings / water / landuse GeoJSON (an ODbL derived database).
    ///
    /// Writes <layer>.geojson, manifest.json, ATTRIBUTION.txt and LICENSE.txt into
    /// <cache>/osm/derived/<name>/ unless --out is given. Ship that folder whole (see README).
    OsmExtract {
        /// Region id or path (downloaded if not cached), e.g. `monaco`, `south-australia`.
        #[arg(long, conflicts_with = "pbf")]
        region: Option<String>,
        /// Use this .osm.pbf file instead of a Geofabrik region.
        #[arg(long, value_name = "FILE")]
        pbf: Option<PathBuf>,
        /// west,south,east,north: keep features intersecting this box. Without --region/--pbf it
        /// also picks the smallest Geofabrik region covering it.
        #[arg(long, allow_hyphen_values = true)]
        bbox: Option<BBox>,
        /// Comma-separated layers (default all): roads,buildings,water,landuse.
        #[arg(long, value_delimiter = ',')]
        layers: Option<Vec<Layer>>,
        /// Do not download; the extract must already be cached.
        #[arg(long)]
        offline: bool,
        /// Output directory (default <cache>/osm/derived/<name>).
        #[arg(long, value_name = "DIR")]
        out: Option<PathBuf>,
        /// Output name below <cache>/osm/derived/ (default: region file stem, plus bbox if given).
        #[arg(long)]
        name: Option<String>,
    },
    /// Print the attribution block to paste into CREDITS.md (both datasets unless one is chosen).
    Credits {
        #[arg(long)]
        osm: bool,
        #[arg(long)]
        terrain: bool,
        /// Also print the notice to ship next to OSM-derived database files (ODbL).
        #[arg(long)]
        db_notice: bool,
    },
}

fn parse_zooms(s: &str) -> Result<RangeInclusive<u8>, String> {
    let s = s.trim();
    let (a, b) = s.split_once("..").or_else(|| s.split_once('-')).unwrap_or((s, s));
    let p = |v: &str| v.trim().parse::<u8>().map_err(|e| format!("bad zoom {v:?}: {e}"));
    let (a, b) = (p(a)?, p(b)?);
    if a > b || b > geo_fetch::bbox::MAX_ZOOM {
        return Err(format!("zoom range {a}..{b} must be ascending and within 0..=15"));
    }
    Ok(a..=b)
}

fn parse_latlon(s: &str) -> Result<(f64, f64), String> {
    let (a, b) = s.split_once(',').ok_or_else(|| format!("{s:?}: expected lat,lon"))?;
    let p = |v: &str| v.trim().parse::<f64>().map_err(|e| format!("{v:?}: {e}"));
    let (lat, lon) = (p(a)?, p(b)?);
    if !(-85.0..=85.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
        return Err(format!("{s:?}: lat must be within ±85, lon within ±180"));
    }
    Ok((lat, lon))
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("geo-fetch: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Region id / path / bbox -> Geofabrik region path. A plain path (`europe/monaco`) works without
/// the index; ids and bboxes need it (cached after the first download).
fn resolve_region(
    http: &HttpFetcher,
    cache: &CacheDir,
    region: Option<String>,
    bbox: Option<BBox>,
    refresh: bool,
) -> geo_fetch::Result<String> {
    Ok(match (region, bbox) {
        (Some(r), _) if r.contains('/') => r,
        (Some(r), _) => {
            let idx = osm::load_index(http, cache, refresh)?;
            osm::find_region(&idx, &r)
                .map(|x| x.path.clone())
                .ok_or_else(|| geo_fetch::Error::NoRegion(format!("region {r:?} (try `geo-fetch regions {r}`)")))?
        }
        (None, Some(b)) => {
            let idx = osm::load_index(http, cache, refresh)?;
            let r = osm::smallest_region_for(&idx, &b).ok_or_else(|| geo_fetch::Error::NoRegion(b.to_string()))?;
            eprintln!("bbox {b} -> region {} ({})", r.id, r.name);
            r.path
        }
        (None, None) => return Err(geo_fetch::Error::BadArg("need --region, --pbf or --bbox".into())),
    })
}

fn mib(b: u64) -> String {
    format!("{:.2} MiB", b as f64 / 1048576.0)
}

fn run(cli: Cli) -> geo_fetch::Result<()> {
    let cache = cli.cache_dir.map(CacheDir::new).unwrap_or_else(CacheDir::from_env);
    let http = HttpFetcher::default();
    match cli.cmd {
        Cmd::Osm { region, bbox, refresh, dry_run } => {
            let path = resolve_region(&http, &cache, region, bbox, refresh)?;
            if dry_run {
                println!("url:  {}", osm::extract_url(&path)?);
                println!("path: {}", osm::extract_path(&cache, &path)?.display());
                return Ok(());
            }
            eprintln!("osm: {}", osm::extract_url(&path)?);
            let e = osm::fetch_extract(&http, &cache, &path, osm::ExtractOptions { refresh })?;
            println!(
                "{} {} ({}, md5 {})",
                if e.downloaded { "downloaded" } else { "cached" },
                e.path.display(),
                mib(e.bytes),
                e.md5
            );
            println!("credit: {}", attribution::OSM_SHORT);
        }
        Cmd::Regions { filter, refresh } => {
            let idx = osm::load_index(&http, &cache, refresh)?;
            let f = filter.map(|s| s.to_lowercase());
            for r in idx.iter().filter(|r| {
                f.as_ref().is_none_or(|f| {
                    r.id.contains(f.as_str())
                        || r.path.contains(f.as_str())
                        || r.name.to_lowercase().contains(f.as_str())
                })
            }) {
                let bb = r
                    .bbox
                    .map(|b| format!("{:.3},{:.3},{:.3},{:.3}", b.west, b.south, b.east, b.north))
                    .unwrap_or_default();
                println!("{:<28} {:<52} {:<32} {bb}", r.id, r.path, r.name);
            }
        }
        Cmd::Terrain { bbox, zoom, format, jobs, max_tiles, refresh, dry_run } => {
            let mut req = TileRequest::new(format, bbox, zoom.clone());
            req.jobs = jobs;
            req.max_tiles = max_tiles;
            req.refresh = refresh;
            if dry_run {
                let keys = terrain::keys_for(format, &bbox, zoom)?;
                println!("{} {} object(s) for {bbox}", keys.len(), format.name());
                for k in keys.iter().take(8) {
                    println!(
                        "{}  ->  {}",
                        terrain::url(format, *k)?,
                        terrain::cache_path(&cache, format, *k)?.display()
                    );
                }
                if keys.len() > 8 {
                    println!("... ({} more)", keys.len() - 8);
                }
                return Ok(());
            }
            let r = terrain::fetch_tiles(&http, &cache, &req)?;
            println!(
                "{} tiles: {} downloaded ({}), {} cached, {} missing -> {}",
                r.requested,
                r.downloaded,
                mib(r.bytes),
                r.cached,
                r.missing.len(),
                cache.terrain_dir().join("aws").join(format.name()).display()
            );
            println!("credit: {}", attribution::TERRAIN_SHORT);
        }
        Cmd::TerrainGrid {
            bbox,
            zoom,
            spacing,
            origin,
            height_ref,
            fallback_zooms,
            no_fill,
            no_supersample,
            offline,
            out,
            name,
            jobs,
            max_tiles,
        } => {
            let spec = GridSpec {
                spacing_m: spacing,
                origin,
                height_ref,
                fallback_zooms,
                fill_holes: !no_fill,
                supersample: !no_supersample,
                jobs,
                max_tiles,
                ..GridSpec::new(bbox, zoom)
            };
            let fetch: Option<&dyn geo_fetch::Fetch> = if offline { None } else { Some(&http) };
            let (g, rep) = grid::build_grid(fetch, &cache, &spec)?;
            let dir = out.unwrap_or_else(|| {
                grid::grid_dir(&cache, &name.unwrap_or_else(|| grid::default_name(&spec, g.spacing_m)))
            });
            let files = grid::write_grid(&g, &rep, &spec, &dir)?;
            let (lo, hi, mean) = g.stats();
            let (x0, y0, x1, y1) = g.extent();
            println!(
                "grid {}x{} @ {} m (source px {:.2} m, {}x{} area taps), origin {:.6},{:.6}",
                g.width,
                g.height,
                g.spacing_m,
                rep.source_pixel_m,
                rep.supersample,
                rep.supersample,
                g.origin.lat_deg,
                g.origin.lon_deg
            );
            println!(
                "extent E {x0:.0}..{x1:.0} m, N {y0:.0}..{y1:.0} m; height {lo:.1}..{hi:.1} m (mean {mean:.1}, {})",
                height_ref.name()
            );
            let used: Vec<String> = rep.tiles_used.iter().map(|(z, n)| format!("z{z}:{n}")).collect();
            println!(
                "tiles {} ({} downloaded), {} missing, {} bad; {} fallback + {} filled samples, {} NaN",
                used.join(" "),
                rep.downloaded,
                rep.missing_tiles.len(),
                rep.bad_tiles.len(),
                rep.fallback_samples,
                rep.filled_samples,
                rep.nan_samples
            );
            for p in [&files.npy, &files.meta, &files.png16, &files.hillshade, &files.attribution] {
                println!("  {:>10}  {}", grid::file_size(p), p.display());
            }
            println!("credit: {}", attribution::TERRAIN_SHORT);
        }
        Cmd::OsmExtract { region, pbf, bbox, layers, offline, out, name } => {
            let src = match pbf {
                Some(p) => SourceInfo { pbf: p, ..Default::default() },
                None => {
                    let path = resolve_region(&http, &cache, region, bbox, false)?;
                    let file = osm::extract_path(&cache, &path)?;
                    if offline {
                        if !file.exists() {
                            return Err(geo_fetch::Error::BadArg(format!(
                                "{} is not cached (drop --offline)",
                                file.display()
                            )));
                        }
                        SourceInfo { pbf: file, url: Some(osm::extract_url(&path)?), md5: None }
                    } else {
                        let e = osm::fetch_extract(&http, &cache, &path, osm::ExtractOptions::default())?;
                        eprintln!(
                            "{} {} ({})",
                            if e.downloaded { "downloaded" } else { "cached" },
                            e.path.display(),
                            mib(e.bytes)
                        );
                        SourceInfo { pbf: e.path, url: Some(e.url), md5: Some(e.md5) }
                    }
                }
            };
            let spec = ExtractSpec { bbox, layers: layers.unwrap_or_else(|| Layer::ALL.to_vec()) };
            let t0 = std::time::Instant::now();
            let (features, rep) = osm_extract::extract(&src.pbf, &spec)?;
            let name = name.unwrap_or_else(|| {
                let stem = src.pbf.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                let stem = stem.trim_end_matches(".osm.pbf").trim_end_matches("-latest").to_owned();
                match bbox {
                    Some(b) => format!("{stem}_{}_{}_{}_{}", b.west, b.south, b.east, b.north),
                    None => stem,
                }
            });
            let dir = out.unwrap_or_else(|| osm_extract::derived_dir(&cache, &name));
            let files = osm_extract::write_layers(&features, &rep, &spec, &src, &dir)?;
            println!(
                "read {} ways, {} multipolygons, {} nodes in {:.1}s; dropped {} incomplete, {} unclosed, {} outside bbox",
                rep.ways_read,
                rep.relations_read,
                rep.nodes_needed,
                t0.elapsed().as_secs_f64(),
                rep.incomplete,
                rep.unclosed,
                rep.outside_bbox
            );
            for (layer, p) in &files.layers {
                let n = rep.counts.iter().find(|c| c.0 == *layer).map(|c| c.1).unwrap_or(0);
                println!("  {:>8} features {:>12}  {}", n, grid::file_size(p), p.display());
            }
            for p in [&files.manifest, &files.attribution, &files.license] {
                println!("  {:>8}          {:>12}  {}", "", grid::file_size(p), p.display());
            }
            println!("licence: ODbL 1.0 (derived database) — ship the folder with its ATTRIBUTION.txt and LICENSE.txt");
        }
        Cmd::Credits { osm, terrain, db_notice } => {
            let used = if osm || terrain { Used { osm, terrain } } else { Used { osm: true, terrain: true } };
            print!("{}", attribution::credits_block(used));
            if db_notice {
                println!("\n{}", attribution::OSM_DERIVED_DB_NOTICE);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn zoom_parsing() {
        assert_eq!(parse_zooms("10").unwrap(), 10..=10);
        assert_eq!(parse_zooms("8..12").unwrap(), 8..=12);
        assert_eq!(parse_zooms("8-12").unwrap(), 8..=12);
        assert!(parse_zooms("12..8").is_err());
        assert!(parse_zooms("16").is_err());
    }

    #[test]
    fn negative_bbox_parses() {
        let c = Cli::try_parse_from(["geo-fetch", "terrain", "--bbox", "-1.5,-0.5,0.5,0.5", "--zoom", "3"]).unwrap();
        assert!(matches!(c.cmd, Cmd::Terrain { .. }));
        let c = Cli::try_parse_from([
            "geo-fetch",
            "terrain-grid",
            "--bbox=138.5,-35.0,138.7,-34.8",
            "--zoom",
            "12",
            "--origin=-34.9,138.6",
            "--height-ref",
            "enu-up",
        ])
        .unwrap();
        assert!(matches!(
            c.cmd,
            Cmd::TerrainGrid { zoom: 12, origin: Some((-34.9, 138.6)), height_ref: HeightRef::EnuUp, .. }
        ));
        assert!(Cli::try_parse_from(["geo-fetch", "terrain-grid", "--bbox=0,0,1,1", "--zoom", "16"]).is_err());
        assert!(parse_latlon("91,0").is_err());
        let c = Cli::try_parse_from([
            "geo-fetch",
            "osm-extract",
            "--region",
            "monaco",
            "--layers",
            "roads,water",
            "--bbox=7.40,43.72,7.44,43.75",
        ])
        .unwrap();
        match c.cmd {
            Cmd::OsmExtract { layers, .. } => assert_eq!(layers, Some(vec![Layer::Roads, Layer::Water])),
            _ => panic!(),
        }
        assert!(Cli::try_parse_from(["geo-fetch", "osm-extract", "--region", "monaco", "--pbf", "x.pbf"]).is_err());
        assert!(Cli::try_parse_from(["geo-fetch", "osm-extract", "--layers", "google"]).is_err());
    }
}
