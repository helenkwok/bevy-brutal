//! Decode Terrarium tiles and resample them into one height grid in a local ENU frame.
//!
//! Pipeline ([`build_grid`]):
//! 1. Pick a local East-North-Up frame ([`coords::Enu`]) at the bbox centre (or a caller origin)
//!    and a square lattice of `spacing_m` metres that covers the bbox. Lattice points sit at
//!    integer multiples of the spacing from the origin, so two grids that share an origin and
//!    spacing share sample positions exactly.
//! 2. Make sure the Terrarium tiles under the grid are cached (download them when a [`Fetch`] is
//!    given), decode them, and stitch them into one Web-Mercator mosaic. Sampling happens in the
//!    mosaic's global pixel space, so tile seams are not special: bilinear taps simply read the
//!    neighbouring tile's pixels.
//! 3. For every lattice point, map ENU → WGS84 → Mercator pixel and sample. When the grid spacing
//!    is coarser than the source pixels, `k×k` sub-taps are averaged per cell ("a pixel shows the
//!    average of its footprint", no aliasing); otherwise plain bilinear.
//! 4. Holes (tiles the bucket does not have, or that fail to decode) fall back to coarser zooms
//!    `z-1, z-2, …`, and anything still empty is filled by iterative neighbour averaging. Both are
//!    counted in [`GridReport`].
//!
//! Heights are metres as published by the Terrain Tiles (orthometric, ≈ EGM96 / mean sea level,
//! sea floor negative). With [`HeightRef::EnuUp`] the tangent-plane curvature drop is subtracted
//! so `(east, north, height)` is (to first order) a true ENU position.
//!
//! Memory layout: row-major, **row 0 = south**, column 0 = west; sample `(col, row)` sits at
//! `(x0 + col·spacing, y0 + row·spacing)` metres east/north of the origin. Previews written by
//! [`write_grid`] are flipped to the usual north-up image orientation.

use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use coords::{Enu, Geodetic};

use crate::bbox::MAX_MERCATOR_LAT;
use crate::cache::{create_parent, io_err, write_atomic};
use crate::terrain::{self, Format, Key, TileRequest, decode_terrarium};
use crate::{BBox, CacheDir, Error, Fetch, Result, TileId, attribution};

/// Terrarium tiles are 256×256.
pub const TILE_PX: u32 = 256;
/// Default cap on output samples (64 Mi samples = 256 MiB of f32).
pub const DEFAULT_MAX_SAMPLES: usize = 64 << 20;
/// Largest supersampling factor per axis.
const MAX_SUPERSAMPLE: usize = 8;

/// What the stored height means.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HeightRef {
    /// DEM height as published (≈ metres above mean sea level). The planet's curvature is not in
    /// the grid; the renderer/planet layer adds it.
    #[default]
    Dem,
    /// DEM height minus the tangent-plane curvature drop: the ENU up coordinate.
    EnuUp,
}

impl HeightRef {
    pub fn name(self) -> &'static str {
        match self {
            HeightRef::Dem => "dem",
            HeightRef::EnuUp => "enu_up",
        }
    }
}

impl std::str::FromStr for HeightRef {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "dem" => Ok(HeightRef::Dem),
            "enu_up" | "enu-up" => Ok(HeightRef::EnuUp),
            _ => Err(Error::BadArg(format!("unknown height reference {s:?} (dem|enu-up)"))),
        }
    }
}

/// Inputs for [`build_grid`].
#[derive(Clone, Debug)]
pub struct GridSpec {
    pub bbox: BBox,
    /// Terrarium zoom to sample (0..=15).
    pub zoom: u8,
    /// Lattice spacing in metres; `None` = the source pixel size at the origin latitude, floored
    /// to whole metres (or to 0.1 m below 1 m).
    pub spacing_m: Option<f64>,
    /// ENU origin `(lat, lon)` in degrees; `None` = bbox centre. Height is always 0 (ellipsoid).
    pub origin: Option<(f64, f64)>,
    /// How many coarser zooms to try for holes before neighbour filling.
    pub fallback_zooms: u8,
    /// Fill remaining holes by neighbour averaging (otherwise they stay NaN).
    pub fill_holes: bool,
    /// Area-average when the spacing is coarser than the source pixels.
    pub supersample: bool,
    pub height_ref: HeightRef,
    /// Per-zoom cap on tiles (download and decode).
    pub max_tiles: usize,
    pub max_samples: usize,
    /// Parallel downloads.
    pub jobs: usize,
}

impl GridSpec {
    pub fn new(bbox: BBox, zoom: u8) -> Self {
        Self {
            bbox,
            zoom,
            spacing_m: None,
            origin: None,
            fallback_zooms: 4,
            fill_holes: true,
            supersample: true,
            height_ref: HeightRef::Dem,
            max_tiles: terrain::DEFAULT_MAX_TILES,
            max_samples: DEFAULT_MAX_SAMPLES,
            jobs: 4,
        }
    }

    pub fn origin_geodetic(&self) -> Geodetic {
        let (lat, lon) =
            self.origin.unwrap_or(((self.bbox.south + self.bbox.north) / 2.0, (self.bbox.west + self.bbox.east) / 2.0));
        Geodetic::new(lat, lon, 0.0)
    }
}

/// Ground size of one Terrarium pixel at `lat_deg` and zoom `z`, metres (WGS84 equatorial radius,
/// Web-Mercator scale `cos φ`).
pub fn source_pixel_m(lat_deg: f64, z: u8) -> f64 {
    2.0 * std::f64::consts::PI * coords::geodetic::WGS84_A * lat_deg.to_radians().cos()
        / (TILE_PX as f64 * (1u64 << z) as f64)
}

/// Default spacing for a zoom: the source pixel size floored to whole metres (0.1 m below 1 m).
pub fn default_spacing_m(lat_deg: f64, z: u8) -> f64 {
    let p = source_pixel_m(lat_deg, z);
    if p >= 1.0 { p.floor() } else { ((p * 10.0).floor() / 10.0).max(0.1) }
}

/// A resampled height grid in a local ENU frame. See the module docs for the layout.
#[derive(Clone, Debug, PartialEq)]
pub struct HeightGrid {
    pub origin: Geodetic,
    pub width: usize,
    pub height: usize,
    pub spacing_m: f64,
    /// East of column 0, metres.
    pub x0: f64,
    /// North of row 0 (the southern row), metres.
    pub y0: f64,
    pub height_ref: HeightRef,
    /// `width * height` heights, row 0 = south. NaN only if hole filling was disabled.
    pub heights: Vec<f32>,
}

impl HeightGrid {
    pub fn enu(&self) -> Enu {
        Enu::new(self.origin)
    }

    #[inline]
    pub fn get(&self, col: usize, row: usize) -> f32 {
        self.heights[row * self.width + col]
    }

    /// ENU (east, north) of a sample.
    pub fn position(&self, col: usize, row: usize) -> (f64, f64) {
        (self.x0 + col as f64 * self.spacing_m, self.y0 + row as f64 * self.spacing_m)
    }

    /// Extent `(east_min, north_min, east_max, north_max)` of the sample lattice.
    pub fn extent(&self) -> (f64, f64, f64, f64) {
        let (x1, y1) = self.position(self.width - 1, self.height - 1);
        (self.x0, self.y0, x1, y1)
    }

    /// Bilinear height at ENU `(east, north)`; `None` outside the lattice.
    pub fn sample(&self, east: f64, north: f64) -> Option<f32> {
        let u = (east - self.x0) / self.spacing_m;
        let v = (north - self.y0) / self.spacing_m;
        if !(0.0..=(self.width - 1) as f64).contains(&u) || !(0.0..=(self.height - 1) as f64).contains(&v) {
            return None;
        }
        let (c0, r0) = (
            (u.floor() as usize).min(self.width.saturating_sub(2)),
            (v.floor() as usize).min(self.height.saturating_sub(2)),
        );
        let (c1, r1) = ((c0 + 1).min(self.width - 1), (r0 + 1).min(self.height - 1));
        let (fu, fv) = ((u - c0 as f64) as f32, (v - r0 as f64) as f32);
        let a = self.get(c0, r0) * (1.0 - fu) + self.get(c1, r0) * fu;
        let b = self.get(c0, r1) * (1.0 - fu) + self.get(c1, r1) * fu;
        Some(a * (1.0 - fv) + b * fv)
    }

    /// `(min, max, mean)` over finite samples.
    pub fn stats(&self) -> (f32, f32, f32) {
        let (mut lo, mut hi, mut sum, mut n) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64, 0usize);
        for &h in self.heights.iter().filter(|h| h.is_finite()) {
            lo = lo.min(h);
            hi = hi.max(h);
            sum += h as f64;
            n += 1;
        }
        (lo, hi, if n > 0 { (sum / n as f64) as f32 } else { f32::NAN })
    }
}

/// What [`build_grid`] did, for logs and the metadata sidecar.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GridReport {
    /// `(zoom, tiles decoded)` for the main zoom and every fallback zoom that was needed.
    pub tiles_used: Vec<(u8, usize)>,
    /// Tiles that were absent (bucket 403/404, or not cached when offline).
    pub missing_tiles: Vec<TileId>,
    /// Tiles that were present but failed to decode (treated as missing).
    pub bad_tiles: Vec<TileId>,
    /// Samples taken from a coarser zoom.
    pub fallback_samples: usize,
    /// Samples filled by neighbour averaging.
    pub filled_samples: usize,
    /// Samples still NaN (only when `fill_holes` is off).
    pub nan_samples: usize,
    /// Sub-taps per axis used for area averaging (1 = plain bilinear).
    pub supersample: usize,
    pub source_pixel_m: f64,
    pub downloaded: usize,
}

/// Decode one Terrarium PNG (256² RGB/RGBA, 8-bit) into row-major metres.
pub fn decode_terrarium_png(bytes: &[u8]) -> Result<Vec<f32>> {
    let bad = |m: String| Error::Decode(format!("terrarium png: {m}"));
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().map_err(|e| bad(e.to_string()))?;
    let size = reader.output_buffer_size().ok_or_else(|| bad("image too large".into()))?;
    let mut buf = vec![0; size];
    let info = reader.next_frame(&mut buf).map_err(|e| bad(e.to_string()))?;
    if (info.width, info.height) != (TILE_PX, TILE_PX) {
        return Err(bad(format!("{}x{}, expected 256x256", info.width, info.height)));
    }
    let ch = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        c => return Err(bad(format!("colour type {c:?}, expected RGB/RGBA"))),
    };
    Ok(buf[..info.buffer_size()].chunks_exact(ch).map(|p| decode_terrarium(p[0], p[1], p[2])).collect())
}

/// Encode metres back into a Terrarium PNG (RGB). Used by tests and handy for re-tiling.
pub fn encode_terrarium_png(heights: &[f32]) -> Result<Vec<u8>> {
    assert_eq!(heights.len(), (TILE_PX * TILE_PX) as usize);
    let mut rgb = Vec::with_capacity(heights.len() * 3);
    for &h in heights {
        let v = ((h as f64 + 32768.0) * 256.0).round().clamp(0.0, 16_777_215.0) as u32;
        rgb.extend_from_slice(&[(v >> 16) as u8, (v >> 8) as u8, v as u8]);
    }
    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, TILE_PX, TILE_PX);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header().map_err(|e| Error::Decode(e.to_string()))?;
    w.write_image_data(&rgb).map_err(|e| Error::Decode(e.to_string()))?;
    w.finish().map_err(|e| Error::Decode(e.to_string()))?;
    Ok(out)
}

/// Continuous Web-Mercator pixel coordinates at zoom `z` (pixel `i`'s centre is at `i + 0.5`).
pub fn mercator_px(lon_deg: f64, lat_deg: f64, z: u8) -> (f64, f64) {
    let world = TILE_PX as f64 * (1u64 << z) as f64;
    let lat = lat_deg.clamp(-MAX_MERCATOR_LAT, MAX_MERCATOR_LAT).to_radians();
    let x = (lon_deg + 180.0) / 360.0 * world;
    let y = (1.0 - (lat.tan() + 1.0 / lat.cos()).ln() / std::f64::consts::PI) / 2.0 * world;
    (x, y)
}

/// Decoded tiles of one zoom stitched into a single array; missing pixels are NaN.
struct Mosaic {
    z: u8,
    /// Global pixel index of the mosaic's first column/row.
    px0: i64,
    py0: i64,
    w: usize,
    h: usize,
    px: Vec<f32>,
}

impl Mosaic {
    #[inline]
    fn at(&self, gx: i64, gy: i64) -> f32 {
        let (x, y) = (gx - self.px0, gy - self.py0);
        if x < 0 || y < 0 || x >= self.w as i64 || y >= self.h as i64 {
            return f32::NAN;
        }
        self.px[y as usize * self.w + x as usize]
    }

    fn sample_at(&self, g: Geodetic) -> f32 {
        self.sample(g.lon_deg, g.lat_deg)
    }

    /// NaN-aware bilinear sample at a geodetic position: missing taps are dropped and the rest
    /// renormalised, so data extends at most half a pixel into a hole.
    fn sample(&self, lon: f64, lat: f64) -> f32 {
        let (x, y) = mercator_px(lon, lat, self.z);
        let (u, v) = (x - 0.5, y - 0.5);
        let (i, j) = (u.floor(), v.floor());
        let (fu, fv) = (u - i, v - j);
        let (i, j) = (i as i64, j as i64);
        let taps = [
            (self.at(i, j), (1.0 - fu) * (1.0 - fv)),
            (self.at(i + 1, j), fu * (1.0 - fv)),
            (self.at(i, j + 1), (1.0 - fu) * fv),
            (self.at(i + 1, j + 1), fu * fv),
        ];
        let (mut acc, mut wsum) = (0.0f64, 0.0f64);
        for (h, w) in taps {
            if h.is_finite() && w > 0.0 {
                acc += h as f64 * w;
                wsum += w;
            }
        }
        if wsum > 1e-9 { (acc / wsum) as f32 } else { f32::NAN }
    }
}

/// Planned lattice (before sampling).
struct Lattice {
    enu: Enu,
    spacing: f64,
    x0: f64,
    y0: f64,
    w: usize,
    h: usize,
}

impl Lattice {
    fn plan(spec: &GridSpec) -> Result<Self> {
        let origin = spec.origin_geodetic();
        let enu = Enu::new(origin);
        let spacing = spec.spacing_m.unwrap_or_else(|| default_spacing_m(origin.lat_deg, spec.zoom));
        if !(spacing.is_finite() && spacing > 0.0) {
            return Err(Error::BadArg(format!("spacing {spacing} m must be > 0")));
        }
        // ENU footprint of the bbox, from a dense walk around its edges.
        let b = &spec.bbox;
        let (mut lo, mut hi) = ((f64::INFINITY, f64::INFINITY), (f64::NEG_INFINITY, f64::NEG_INFINITY));
        for (lon, lat) in perimeter(b, 64) {
            let p = enu.geodetic_to_enu(Geodetic::new(lat, lon, 0.0));
            lo = (lo.0.min(p.x), lo.1.min(p.y));
            hi = (hi.0.max(p.x), hi.1.max(p.y));
        }
        let snap_lo = |v: f64| (v / spacing).floor() * spacing;
        let (x0, y0) = (snap_lo(lo.0), snap_lo(lo.1));
        let w = ((hi.0 - x0) / spacing).ceil() as usize + 1;
        let h = ((hi.1 - y0) / spacing).ceil() as usize + 1;
        if w.saturating_mul(h) > spec.max_samples {
            return Err(Error::TooMany {
                what: "height grid samples",
                count: w.saturating_mul(h),
                limit: spec.max_samples,
            });
        }
        Ok(Self { enu, spacing, x0, y0, w, h })
    }

    fn geodetic(&self, col: usize, row: usize) -> Geodetic {
        self.enu.plane_to_geodetic(self.x0 + col as f64 * self.spacing, self.y0 + row as f64 * self.spacing)
    }

    /// WGS84 box covering every lattice point (walks the lattice perimeter).
    fn geo_bounds(&self) -> BBox {
        let (mut w, mut s, mut e, mut n) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        let steps = 64usize;
        let mut push = |g: Geodetic| {
            w = w.min(g.lon_deg);
            e = e.max(g.lon_deg);
            s = s.min(g.lat_deg);
            n = n.max(g.lat_deg);
        };
        for k in 0..=steps {
            let c = k * (self.w - 1) / steps;
            let r = k * (self.h - 1) / steps;
            push(self.geodetic(c, 0));
            push(self.geodetic(c, self.h - 1));
            push(self.geodetic(0, r));
            push(self.geodetic(self.w - 1, r));
        }
        BBox { west: w, south: s, east: e, north: n }
    }
}

fn perimeter(b: &BBox, steps: usize) -> impl Iterator<Item = (f64, f64)> + '_ {
    (0..=steps).flat_map(move |k| {
        let t = k as f64 / steps as f64;
        let lon = b.west + t * (b.east - b.west);
        let lat = b.south + t * (b.north - b.south);
        [(lon, b.south), (lon, b.north), (b.west, lat), (b.east, lat)]
    })
}

/// Expand a WGS84 box by `px` source pixels at zoom `z` (so bilinear taps at the edge have data),
/// clamped to valid coordinates.
fn pad_bbox(b: &BBox, z: u8, px: f64) -> BBox {
    // A Mercator pixel spans 360/(256·2^z) degrees of longitude and at most that much latitude
    // (latitude pixels shrink by cos φ), so one pad works for both axes.
    let deg = px * 360.0 / (TILE_PX as f64 * (1u64 << z) as f64);
    BBox {
        west: (b.west - deg).max(-180.0),
        east: (b.east + deg).min(180.0),
        south: (b.south - deg).max(-MAX_MERCATOR_LAT),
        north: (b.north + deg).min(MAX_MERCATOR_LAT),
    }
}

/// Ensure tiles are cached (if `fetch` is given) and stitch them into a mosaic.
fn load_mosaic(
    fetch: Option<&dyn Fetch>,
    cache: &CacheDir,
    bounds: &BBox,
    z: u8,
    spec: &GridSpec,
    report: &mut GridReport,
) -> Result<Mosaic> {
    let padded = pad_bbox(bounds, z, 2.0);
    let count = padded.tile_count(z);
    if count > spec.max_tiles {
        return Err(Error::TooMany { what: "terrain-grid tiles", count, limit: spec.max_tiles });
    }
    if let Some(f) = fetch {
        let mut req = TileRequest::new(Format::Terrarium, padded, z..=z);
        req.jobs = spec.jobs;
        req.max_tiles = spec.max_tiles;
        report.downloaded += terrain::fetch_tiles(f, cache, &req)?.downloaded;
    }
    let (xs, ys) = padded.tile_range(z);
    let (tw, th) = ((xs.end() - xs.start() + 1) as usize, (ys.end() - ys.start() + 1) as usize);
    let t = TILE_PX as usize;
    let (w, h) = (tw * t, th * t);
    let mut px = vec![f32::NAN; w * h];
    let mut used = 0;
    for ty in ys.clone() {
        for tx in xs.clone() {
            let id = TileId { z, x: tx, y: ty };
            let path = terrain::cache_path(cache, Format::Terrarium, Key::Tile(id))?;
            let tile = match fs::read(&path) {
                Ok(bytes) => match decode_terrarium_png(&bytes) {
                    Ok(v) => v,
                    Err(_) => {
                        report.bad_tiles.push(id);
                        continue;
                    }
                },
                Err(_) => {
                    report.missing_tiles.push(id);
                    continue;
                }
            };
            used += 1;
            let (ox, oy) = ((tx - xs.start()) as usize * t, (ty - ys.start()) as usize * t);
            for r in 0..t {
                px[(oy + r) * w + ox..(oy + r) * w + ox + t].copy_from_slice(&tile[r * t..(r + 1) * t]);
            }
        }
    }
    report.tiles_used.push((z, used));
    Ok(Mosaic { z, px0: *xs.start() as i64 * t as i64, py0: *ys.start() as i64 * t as i64, w, h, px })
}

/// Sample one lattice cell: centre position plus optional `k×k` sub-taps (area average).
fn sample_cell(m: &Mosaic, g: Geodetic, k: usize, spacing: f64) -> f32 {
    if k <= 1 {
        return m.sample(g.lon_deg, g.lat_deg);
    }
    // Within one cell a linear lat/lon offset is exact to well under a millimetre.
    let dlat = (spacing / coords::geodetic::meridional_radius(g.lat_deg)).to_degrees();
    let dlon =
        (spacing / (coords::geodetic::prime_vertical_radius(g.lat_deg) * g.lat_deg.to_radians().cos())).to_degrees();
    let (mut acc, mut n) = (0.0f64, 0usize);
    for j in 0..k {
        let fy = (j as f64 + 0.5) / k as f64 - 0.5;
        for i in 0..k {
            let fx = (i as f64 + 0.5) / k as f64 - 0.5;
            let h = m.sample(g.lon_deg + fx * dlon, g.lat_deg + fy * dlat);
            if h.is_finite() {
                acc += h as f64;
                n += 1;
            }
        }
    }
    if n > 0 { (acc / n as f64) as f32 } else { f32::NAN }
}

/// Run `f(row) -> Vec<f32>` over rows on all cores and return the row-major result.
fn par_rows(h: usize, w: usize, f: impl Fn(usize) -> Vec<f32> + Sync) -> Vec<f32> {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(h.max(1));
    let chunk = h.div_ceil(threads);
    let mut out = vec![0.0f32; w * h];
    std::thread::scope(|s| {
        for (ci, slab) in out.chunks_mut(chunk * w).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (k, row) in slab.chunks_mut(w).enumerate() {
                    row.copy_from_slice(&f(ci * chunk + k));
                }
            });
        }
    });
    out
}

/// Fill NaN samples by repeated averaging of finite 8-neighbours (grows inward from the hole
/// edges). Returns the number filled. Leaves the grid untouched if it has no finite sample.
pub fn fill_holes(heights: &mut [f32], w: usize, h: usize) -> usize {
    let mut holes: Vec<usize> = (0..heights.len()).filter(|&i| !heights[i].is_finite()).collect();
    let total = holes.len();
    if total == heights.len() {
        return 0;
    }
    while !holes.is_empty() {
        let updates: Vec<(usize, f32)> = holes
            .iter()
            .filter_map(|&i| {
                let (c, r) = ((i % w) as i64, (i / w) as i64);
                let (mut acc, mut n) = (0.0f32, 0);
                for dr in -1..=1 {
                    for dc in -1..=1 {
                        let (cc, rr) = (c + dc, r + dr);
                        if (dc, dr) != (0, 0) && cc >= 0 && rr >= 0 && cc < w as i64 && rr < h as i64 {
                            let v = heights[rr as usize * w + cc as usize];
                            if v.is_finite() {
                                acc += v;
                                n += 1;
                            }
                        }
                    }
                }
                (n > 0).then(|| (i, acc / n as f32))
            })
            .collect();
        for &(i, v) in &updates {
            heights[i] = v;
        }
        holes.retain(|&i| !heights[i].is_finite());
    }
    total
}

/// Build a height grid for `spec`. With `fetch = Some(..)` missing tiles are downloaded into the
/// cache first; with `None` only cached tiles are used (absent ones become holes).
pub fn build_grid(fetch: Option<&dyn Fetch>, cache: &CacheDir, spec: &GridSpec) -> Result<(HeightGrid, GridReport)> {
    if spec.zoom > crate::bbox::MAX_ZOOM {
        return Err(Error::BadArg(format!("zoom {} must be within 0..={}", spec.zoom, crate::bbox::MAX_ZOOM)));
    }
    let lat = Lattice::plan(spec)?;
    let bounds = lat.geo_bounds();
    let mut report =
        GridReport { source_pixel_m: source_pixel_m(lat.enu.origin().lat_deg, spec.zoom), ..Default::default() };
    report.supersample = if spec.supersample {
        ((lat.spacing / report.source_pixel_m) - 0.25).ceil().clamp(1.0, MAX_SUPERSAMPLE as f64) as usize
    } else {
        1
    };

    let main = load_mosaic(fetch, cache, &bounds, spec.zoom, spec, &mut report)?;
    let k = report.supersample;
    let mut heights = par_rows(lat.h, lat.w, |r| {
        (0..lat.w).map(|c| sample_cell(&main, lat.geodetic(c, r), k, lat.spacing)).collect()
    });
    drop(main);

    // Coarser zooms for holes. Each fallback level is coarser than the grid spacing or close to
    // it, so plain bilinear is enough there.
    let mut z = spec.zoom;
    for _ in 0..spec.fallback_zooms {
        let holes: Vec<usize> = (0..heights.len()).filter(|&i| !heights[i].is_finite()).collect();
        if holes.is_empty() || z == 0 {
            break;
        }
        z -= 1;
        let m = load_mosaic(fetch, cache, &bounds, z, spec, &mut report)?;
        for i in holes {
            let v = m.sample_at(lat.geodetic(i % lat.w, i / lat.w));
            if v.is_finite() {
                heights[i] = v;
                report.fallback_samples += 1;
            }
        }
    }
    if heights.iter().all(|h| !h.is_finite()) {
        return Err(Error::Decode(format!(
            "no terrain data for {} at z{} (missing {} tiles; offline without a cache?)",
            spec.bbox,
            spec.zoom,
            report.missing_tiles.len()
        )));
    }
    if spec.fill_holes {
        report.filled_samples = fill_holes(&mut heights, lat.w, lat.h);
    }
    report.nan_samples = heights.iter().filter(|h| !h.is_finite()).count();
    if spec.height_ref == HeightRef::EnuUp {
        let drops = par_rows(lat.h, lat.w, |r| (0..lat.w).map(|c| lat.geodetic(c, r).h_m as f32).collect());
        for (h, d) in heights.iter_mut().zip(drops) {
            *h -= d;
        }
    }
    report.missing_tiles.sort();
    report.missing_tiles.dedup();
    report.bad_tiles.sort();
    let grid = HeightGrid {
        origin: lat.enu.origin(),
        width: lat.w,
        height: lat.h,
        spacing_m: lat.spacing,
        x0: lat.x0,
        y0: lat.y0,
        height_ref: spec.height_ref,
        heights,
    };
    Ok((grid, report))
}

// ---------------------------------------------------------------------------------------------
// Output

/// Default output directory: `<cache>/terrain/grids/<name>`.
pub fn grid_dir(cache: &CacheDir, name: &str) -> PathBuf {
    cache.terrain_dir().join("grids").join(name)
}

/// Default grid name from the request, e.g. `z12_31m_138.5_-35_138.7_-34.8`.
pub fn default_name(spec: &GridSpec, spacing_m: f64) -> String {
    let b = &spec.bbox;
    let s = if spacing_m.fract() == 0.0 { format!("{spacing_m}") } else { format!("{spacing_m:.2}") };
    format!("z{}_{}m_{}_{}_{}_{}", spec.zoom, s, b.west, b.south, b.east, b.north)
}

/// Files written by [`write_grid`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GridFiles {
    pub dir: PathBuf,
    /// `height.npy`: little-endian f32, shape `(rows, cols)`, row 0 = south.
    pub npy: PathBuf,
    /// `height.json`: frame, lattice, stats, provenance and credit.
    pub meta: PathBuf,
    /// `height16.png`: 16-bit grey, north-up, `h = min + v/65535·(max-min)`.
    pub png16: PathBuf,
    /// `hillshade.png`: 8-bit shaded relief preview, north-up.
    pub hillshade: PathBuf,
    pub attribution: PathBuf,
}

/// Serialise an f32 array as a NumPy `.npy` (v1.0, `<f4`, C order, shape `(rows, cols)`).
pub fn npy_bytes(data: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(data.len(), rows * cols);
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    // magic(6) + version(2) + len(2) + header + '\n' must be a multiple of 64.
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + data.len() * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Parse an `.npy` written by [`npy_bytes`]: returns `(rows, cols, data)`.
pub fn parse_npy(bytes: &[u8]) -> Result<(usize, usize, Vec<f32>)> {
    let bad = |m: &str| Error::Decode(format!("npy: {m}"));
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" || bytes[6] != 1 {
        return Err(bad("not a v1 .npy file"));
    }
    let hl = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(bytes.get(10..10 + hl).ok_or_else(|| bad("truncated header"))?)
        .map_err(|_| bad("header is not UTF-8"))?;
    if !header.contains("'<f4'") || header.contains("'fortran_order': True") {
        return Err(bad("expected little-endian f32 in C order"));
    }
    let shape = header.split("'shape': (").nth(1).and_then(|s| s.split(')').next()).ok_or_else(|| bad("no shape"))?;
    let dims: Vec<usize> = shape
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().map_err(|_| bad("bad shape")))
        .collect::<Result<_>>()?;
    let [rows, cols] = dims[..] else { return Err(bad("expected a 2-D array")) };
    let body = &bytes[10 + hl..];
    if body.len() != rows * cols * 4 {
        return Err(bad("data length does not match shape"));
    }
    Ok((rows, cols, body.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()))
}

fn write_png(path: &Path, w: usize, h: usize, depth: png::BitDepth, data: &[u8]) -> Result<()> {
    create_parent(path)?;
    let f = fs::File::create(path).map_err(io_err(path))?;
    let mut enc = png::Encoder::new(BufWriter::new(f), w as u32, h as u32);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(depth);
    let e = |e: png::EncodingError| Error::Decode(format!("png encode {}: {e}", path.display()));
    let mut wr = enc.write_header().map_err(e)?;
    wr.write_image_data(data).map_err(e)?;
    wr.finish().map_err(e)?;
    Ok(())
}

/// Shaded relief (azimuth 315°, altitude 45°), north-up 8-bit.
pub fn hillshade(g: &HeightGrid) -> Vec<u8> {
    let (w, h) = (g.width, g.height);
    let (az, alt) = (315f64.to_radians(), 45f64.to_radians());
    // Light direction in ENU (azimuth clockwise from north).
    let l = (az.sin() * alt.cos(), az.cos() * alt.cos(), alt.sin());
    let at = |c: i64, r: i64| g.get(c.clamp(0, w as i64 - 1) as usize, r.clamp(0, h as i64 - 1) as usize) as f64;
    let mut out = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let (ci, ri) = (c as i64, r as i64);
            let dzdx = (at(ci + 1, ri) - at(ci - 1, ri)) / (2.0 * g.spacing_m);
            let dzdy = (at(ci, ri + 1) - at(ci, ri - 1)) / (2.0 * g.spacing_m);
            let n = (-dzdx, -dzdy, 1.0);
            let len = (n.0 * n.0 + n.1 * n.1 + 1.0).sqrt();
            let shade = ((n.0 * l.0 + n.1 * l.1 + n.2 * l.2) / len).max(0.0);
            out[(h - 1 - r) * w + c] = (shade * 255.0).round() as u8;
        }
    }
    out
}

/// Write `height.npy`, `height.json`, `height16.png`, `hillshade.png` and `ATTRIBUTION.txt` into `dir`.
pub fn write_grid(grid: &HeightGrid, report: &GridReport, spec: &GridSpec, dir: &Path) -> Result<GridFiles> {
    let files = GridFiles {
        dir: dir.to_path_buf(),
        npy: dir.join("height.npy"),
        meta: dir.join("height.json"),
        png16: dir.join("height16.png"),
        hillshade: dir.join("hillshade.png"),
        attribution: dir.join("ATTRIBUTION.txt"),
    };
    write_atomic(&files.npy, &npy_bytes(&grid.heights, grid.height, grid.width))?;

    let (lo, hi, mean) = grid.stats();
    let range = (hi - lo).max(1e-6);
    let mut png16 = Vec::with_capacity(grid.heights.len() * 2);
    for r in (0..grid.height).rev() {
        for c in 0..grid.width {
            let h = grid.get(c, r);
            let v = if h.is_finite() { (((h - lo) / range) * 65535.0).round() as u16 } else { 0 };
            png16.extend_from_slice(&v.to_be_bytes());
        }
    }
    write_png(&files.png16, grid.width, grid.height, png::BitDepth::Sixteen, &png16)?;
    write_png(&files.hillshade, grid.width, grid.height, png::BitDepth::Eight, &hillshade(grid))?;

    let enu = grid.enu();
    let [e, n, u] = enu.axes();
    let ecef = enu.origin_ecef();
    let (ex0, ey0, ex1, ey1) = grid.extent();
    let tiles = |v: &[TileId]| v.iter().map(|t| format!("{}/{}/{}", t.z, t.x, t.y)).collect::<Vec<_>>();
    let meta = serde_json::json!({
        "format": "geo-fetch height grid v1",
        "generator": crate::USER_AGENT,
        "frame": {
            "kind": "local ENU tangent plane on WGS84",
            "origin_lat_deg": grid.origin.lat_deg,
            "origin_lon_deg": grid.origin.lon_deg,
            "origin_ecef_m": [ecef.x, ecef.y, ecef.z],
            "ecef_convention": "+Z north pole, +X 0E, +Y 90E (matches crates/planet)",
            "east_ecef": [e.x, e.y, e.z],
            "north_ecef": [n.x, n.y, n.z],
            "up_ecef": [u.x, u.y, u.z],
            "projection": "sample (x0 + col*spacing, y0 + row*spacing, 0) in ENU -> WGS84 lat/lon (coords::Enu::plane_to_geodetic)",
        },
        "lattice": {
            "cols": grid.width,
            "rows": grid.height,
            "spacing_m": grid.spacing_m,
            "x0_east_m": grid.x0,
            "y0_north_m": grid.y0,
            "extent_m": [ex0, ey0, ex1, ey1],
            "row_order": "south_to_north (row 0 = south) in height.npy; PNG previews are north-up",
        },
        "height": {
            "units": "m",
            "reference": grid.height_ref.name(),
            "vertical_datum": "as published by AWS Terrain Tiles: orthometric, approximately EGM96 / mean sea level; bathymetry negative",
            "min": lo, "max": hi, "mean": mean,
            "png16_decode": "h = min + v / 65535 * (max - min)",
        },
        "source": {
            "dataset": "AWS Terrain Tiles (Tilezen joerd), terrarium",
            "url_template": format!("{}terrarium/{{z}}/{{x}}/{{y}}.png", terrain::BUCKET_URL),
            "zoom": spec.zoom,
            "source_pixel_m": report.source_pixel_m,
            "supersample": report.supersample,
            "request_bbox": [spec.bbox.west, spec.bbox.south, spec.bbox.east, spec.bbox.north],
            "tiles_used": report.tiles_used.iter().map(|(z, n)| serde_json::json!({"zoom": z, "tiles": n})).collect::<Vec<_>>(),
            "missing_tiles": tiles(&report.missing_tiles),
            "bad_tiles": tiles(&report.bad_tiles),
            "fallback_samples": report.fallback_samples,
            "filled_samples": report.filled_samples,
            "nan_samples": report.nan_samples,
        },
        "files": { "height": "height.npy", "png16": "height16.png", "hillshade": "hillshade.png", "attribution": "ATTRIBUTION.txt" },
        "credit": attribution::TERRAIN_SHORT,
    });
    let mut json = serde_json::to_vec_pretty(&meta).map_err(|e| Error::Decode(e.to_string()))?;
    json.push(b'\n');
    write_atomic(&files.meta, &json)?;
    write_atomic(&files.attribution, attribution::terrain_full().as_bytes())?;
    Ok(files)
}

/// Read a grid back from a directory written by [`write_grid`].
pub fn read_grid(dir: &Path) -> Result<HeightGrid> {
    let meta_path = dir.join("height.json");
    let meta: serde_json::Value = serde_json::from_slice(&fs::read(&meta_path).map_err(io_err(&meta_path))?)
        .map_err(|e| Error::Decode(format!("{}: {e}", meta_path.display())))?;
    let npy_path = dir.join("height.npy");
    let (rows, cols, heights) = parse_npy(&fs::read(&npy_path).map_err(io_err(&npy_path))?)?;
    let num = |p: &str| {
        meta.pointer(p).and_then(|v| v.as_f64()).ok_or_else(|| Error::Decode(format!("height.json: missing {p}")))
    };
    let height_ref = meta.pointer("/height/reference").and_then(|v| v.as_str()).unwrap_or("dem").parse()?;
    Ok(HeightGrid {
        origin: Geodetic::new(num("/frame/origin_lat_deg")?, num("/frame/origin_lon_deg")?, 0.0),
        width: cols,
        height: rows,
        spacing_m: num("/lattice/spacing_m")?,
        x0: num("/lattice/x0_east_m")?,
        y0: num("/lattice/y0_north_m")?,
        height_ref,
        heights,
    })
}

/// Size of a file in bytes (0 if absent), for reports.
pub fn file_size(path: &Path) -> u64 {
    fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetch::mock::MockFetcher;

    /// Analytic terrain in WGS84: a tilted plane plus a bump, so resampling can be checked anywhere.
    fn truth(lon: f64, lat: f64) -> f32 {
        (100.0
            + (lon - 138.6) * 20_000.0
            + (lat + 34.9) * 10_000.0
            + 50.0 * ((lon * 300.0).sin() * (lat * 300.0).cos())) as f32
    }

    fn synth_tile(t: TileId) -> Vec<f32> {
        let n = TILE_PX as usize;
        let world = TILE_PX as f64 * (1u64 << t.z) as f64;
        let mut v = Vec::with_capacity(n * n);
        for r in 0..n {
            for c in 0..n {
                let gx = (t.x as usize * n + c) as f64 + 0.5;
                let gy = (t.y as usize * n + r) as f64 + 0.5;
                let lon = gx / world * 360.0 - 180.0;
                let lat = (std::f64::consts::PI * (1.0 - 2.0 * gy / world)).sinh().atan().to_degrees();
                v.push(truth(lon, lat));
            }
        }
        v
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("geo-fetch-grid-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    fn mock_for(spec: &GridSpec, zooms: &[u8], skip: &[TileId]) -> MockFetcher {
        let mut f = MockFetcher::default();
        let lat = Lattice::plan(spec).unwrap();
        let b = pad_bbox(&lat.geo_bounds(), spec.zoom, 2.0);
        for &z in zooms {
            for t in b.tiles(z) {
                if !skip.contains(&t) {
                    let png = encode_terrarium_png(&synth_tile(t)).unwrap();
                    f = f.with(&terrain::url(Format::Terrarium, Key::Tile(t)).unwrap(), png);
                }
            }
        }
        f
    }

    #[test]
    fn png_round_trip() {
        let t = TileId { z: 12, x: 3625, y: 2470 };
        let h = synth_tile(t);
        let back = decode_terrarium_png(&encode_terrarium_png(&h).unwrap()).unwrap();
        let max_err = h.iter().zip(&back).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(max_err <= 1.0 / 256.0 + 1e-3, "{max_err}");
        assert!(decode_terrarium_png(b"not a png").is_err());
    }

    #[test]
    fn npy_round_trip() {
        let d: Vec<f32> = (0..12).map(|i| i as f32 * 0.5 - 1.0).collect();
        let b = npy_bytes(&d, 3, 4);
        assert_eq!(&b[..6], b"\x93NUMPY");
        assert_eq!((10 + u16::from_le_bytes([b[8], b[9]]) as usize) % 64, 0);
        assert_eq!(parse_npy(&b).unwrap(), (3, 4, d));
    }

    #[test]
    fn mercator_matches_tile_math() {
        let (x, y) = mercator_px(138.6007, -34.9285, 10);
        let t = crate::bbox::lonlat_to_tile(138.6007, -34.9285, 10);
        assert_eq!(((x / 256.0) as u32, (y / 256.0) as u32), (t.x, t.y));
        assert!((source_pixel_m(0.0, 0) - 156_543.03).abs() < 0.01);
        assert_eq!(default_spacing_m(-34.9, 12), 31.0);
    }

    #[test]
    fn lattice_is_aligned_and_covers_bbox() {
        let mut spec = GridSpec::new("138.58,-34.94,138.62,-34.90".parse().unwrap(), 12);
        spec.spacing_m = Some(30.0);
        let l = Lattice::plan(&spec).unwrap();
        assert_eq!((l.x0 / 30.0).fract(), 0.0);
        assert_eq!((l.y0 / 30.0).fract(), 0.0);
        let g = l.geo_bounds();
        assert!(
            g.west <= spec.bbox.west
                && g.east >= spec.bbox.east
                && g.south <= spec.bbox.south
                && g.north >= spec.bbox.north,
            "{g:?}"
        );
        spec.max_samples = 10;
        assert!(matches!(Lattice::plan(&spec), Err(Error::TooMany { .. })));
    }

    #[test]
    fn grid_matches_analytic_surface_across_seams() {
        // Straddles tile boundaries at z12 (several tiles in each direction).
        let spec =
            GridSpec { spacing_m: Some(40.0), ..GridSpec::new("138.55,-34.95,138.65,-34.85".parse().unwrap(), 12) };
        let cache = CacheDir::new(tmp("seams"));
        let f = mock_for(&spec, &[12], &[]);
        let (g, rep) = build_grid(Some(&f), &cache, &spec).unwrap();
        assert!(rep.missing_tiles.is_empty() && rep.filled_samples == 0 && rep.fallback_samples == 0, "{rep:?}");
        assert!(rep.tiles_used[0].1 >= 4, "{rep:?}");
        let enu = g.enu();
        let mut worst = 0.0f32;
        for r in (0..g.height).step_by(7) {
            for c in (0..g.width).step_by(7) {
                let (e, n) = g.position(c, r);
                let p = enu.plane_to_geodetic(e, n);
                worst = worst.max((g.get(c, r) - truth(p.lon_deg, p.lat_deg)).abs());
            }
        }
        // Bilinear/area error on the bump, not seam steps (a seam bug shows up as tens of metres).
        eprintln!("worst resample error {worst} m");
        assert!(worst < 1.0, "worst {worst} m");
        // Offline rebuild from the cache gives the same grid.
        let (g2, rep2) = build_grid(None, &cache, &spec).unwrap();
        assert_eq!(g, g2);
        assert_eq!(rep2.downloaded, 0);
        // Write + read back.
        let dir = cache.root().join("out");
        let files = write_grid(&g, &rep, &spec, &dir).unwrap();
        for p in [&files.npy, &files.meta, &files.png16, &files.hillshade, &files.attribution] {
            assert!(file_size(p) > 0, "{p:?}");
        }
        assert_eq!(read_grid(&dir).unwrap(), g);
        let _ = fs::remove_dir_all(cache.root());
    }

    #[test]
    fn holes_fall_back_to_coarser_zoom_then_fill() {
        let spec =
            GridSpec { spacing_m: Some(40.0), ..GridSpec::new("138.55,-34.95,138.65,-34.85".parse().unwrap(), 12) };
        let lat = Lattice::plan(&spec).unwrap();
        let z12 = pad_bbox(&lat.geo_bounds(), 12, 2.0).tiles(12);
        let hole = z12[z12.len() / 2];
        let cache = CacheDir::new(tmp("holes"));
        let f = mock_for(&spec, &[12, 11], &[hole]);
        let (g, rep) = build_grid(Some(&f), &cache, &spec).unwrap();
        assert_eq!(rep.missing_tiles, vec![hole]);
        assert!(rep.fallback_samples > 0 && rep.filled_samples == 0, "{rep:?}");
        assert!(g.heights.iter().all(|h| h.is_finite()));
        assert_eq!(rep.tiles_used.iter().map(|t| t.0).collect::<Vec<_>>(), vec![12, 11]);

        // No coarser data at all -> neighbour fill.
        let cache2 = CacheDir::new(tmp("holes2"));
        let spec2 = GridSpec { fallback_zooms: 0, ..spec.clone() };
        let (g2, rep2) = build_grid(Some(&mock_for(&spec2, &[12], &[hole])), &cache2, &spec2).unwrap();
        assert!(rep2.filled_samples > 0 && rep2.nan_samples == 0, "{rep2:?}");
        assert!(g2.heights.iter().all(|h| h.is_finite()));
        let (lo, hi, _) = g.stats();
        let (lo2, hi2, _) = g2.stats();
        assert!(lo2 >= lo - 5.0 && hi2 <= hi + 5.0, "fill stays within data range");

        // Fill disabled keeps NaNs and says so.
        let spec3 = GridSpec { fill_holes: false, ..spec2 };
        let (_, rep3) = build_grid(None, &cache2, &spec3).unwrap();
        assert!(rep3.nan_samples > 0);

        // Nothing at all -> error.
        assert!(build_grid(None, &CacheDir::new(tmp("empty")), &spec).is_err());
        let _ = fs::remove_dir_all(cache.root());
        let _ = fs::remove_dir_all(cache2.root());
    }

    #[test]
    fn supersampling_averages_footprint() {
        // 200 m cells over ~31 m source pixels -> 7x7 sub-taps; the 50 m bump (period ~1 km)
        // gets smoothed, the plane does not move.
        let spec =
            GridSpec { spacing_m: Some(200.0), ..GridSpec::new("138.55,-34.95,138.65,-34.85".parse().unwrap(), 12) };
        let cache = CacheDir::new(tmp("ss"));
        let f = mock_for(&spec, &[12], &[]);
        let (g, rep) = build_grid(Some(&f), &cache, &spec).unwrap();
        assert_eq!(rep.supersample, 7);
        let (g1, _) = build_grid(None, &cache, &GridSpec { supersample: false, ..spec.clone() }).unwrap();
        let var = |g: &HeightGrid| {
            let m = g.heights.iter().map(|&h| h as f64).sum::<f64>() / g.heights.len() as f64;
            g.heights.iter().map(|&h| (h as f64 - m).powi(2)).sum::<f64>() / g.heights.len() as f64
        };
        assert!(var(&g) < var(&g1), "area average must not add energy");
        assert!((g.stats().2 - g1.stats().2).abs() < 2.0);
        let _ = fs::remove_dir_all(cache.root());
    }

    #[test]
    fn enu_up_subtracts_curvature() {
        let spec =
            GridSpec { spacing_m: Some(100.0), ..GridSpec::new("138.55,-34.95,138.65,-34.85".parse().unwrap(), 11) };
        let cache = CacheDir::new(tmp("enu"));
        let f = mock_for(&spec, &[11], &[]);
        let (dem, _) = build_grid(Some(&f), &cache, &spec).unwrap();
        let (up, _) = build_grid(None, &cache, &GridSpec { height_ref: HeightRef::EnuUp, ..spec }).unwrap();
        let (c, r) = (0, 0);
        let (e, n) = dem.position(c, r);
        let drop = dem.enu().curvature_drop_m(e, n) as f32;
        assert!(drop > 0.5, "{drop}");
        assert!((dem.get(c, r) - up.get(c, r) - drop).abs() < 1e-3);
        let _ = fs::remove_dir_all(cache.root());
    }
}
