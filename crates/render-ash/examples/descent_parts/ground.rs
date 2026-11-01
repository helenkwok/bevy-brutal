//! Earth round 5 (the grounded Alps): bake real elevation and imagery for the shared disc.
//! Included by the `descent`, `earth` and `quality` examples.
//!
//! - [`bake_ground`]: Copernicus GLO-30 (ESA / AWS open data; lat/lon mosaics written by
//!   `/workspace/data/earth/dem/mosaic.py`: 1" around the vista, 3" box-averaged for the
//!   rest) resampled (Catmull-Rom) onto the mountains world frame through the graft's
//!   azimuthal-equidistant map, two nested levels ([`super::graft::OUTER`] / `INNER`
//!   extents and sizes), as a [`voxel::ground::Ground`] with the default procedural core.
//! - [`Imagery`]: the EOX Sentinel-2 cloudless mosaic (Web Mercator z11 / z13 stitched by
//!   `/workspace/data/earth/s2/mosaic_s2.py`) sampled at a lat/lon; the graft bake turns it into
//!   mountains-layer albedo: decoded to linear, the acquisition's terrain shading divided out
//!   (Lambert under an assumed late-morning summer sun, partial: S2 mosaics mix dates), and
//!   gain-matched to the planet's Blue Marble albedo over the region (so the disc does not
//!   show as a colour patch where the real Earth returns).
#![allow(dead_code)]

use glam::DVec3;
use planet::graft::GraftMap;
use rayon::prelude::*;
use voxel::ground::{Core, Ground, GroundLevel};

pub const DEM_DIR: &str = "/workspace/data/earth/dem";
pub const S2_DIR: &str = "/workspace/data/earth/s2";
/// Earth round 6: the default (smaller, `Core::E6`) procedural core; round 5's file
/// (`ground_alps.grd`, `Core::E5`) stays loadable with `--ground`.
pub const GROUND_FILE: &str = "/workspace/data/earth/ground_alps_e6.grd";
pub const GRAFT_FILE: &str = "/workspace/data/earth/graft_alps_e6.grf";

/// A number from a flat JSON object (the mosaics' headers).
fn json_f(s: &str, k: &str) -> f64 {
    let p = s.find(&format!("\"{k}\"")).unwrap_or_else(|| panic!("json key {k}"));
    let r = &s[p + k.len() + 2..];
    let r = r[r.find(':').expect(":") + 1..].trim_start();
    let e = r.find([',', '}']).unwrap_or(r.len());
    r[..e].trim().parse().unwrap_or_else(|_| panic!("json value {k}"))
}

/// A lat/lon f32 grid (row 0 = `lat0`, rows go south; PixelIsPoint).
pub struct Dem {
    lat0: f64,
    lon0: f64,
    dlat: f64,
    dlon: f64,
    rows: usize,
    cols: usize,
    h: Vec<f32>,
}

impl Dem {
    pub fn load(stem: &str) -> Self {
        let j = std::fs::read_to_string(format!("{stem}.json")).unwrap_or_else(|e| panic!("{stem}.json: {e}"));
        let raw = std::fs::read(format!("{stem}.f32")).unwrap_or_else(|e| panic!("{stem}.f32: {e}"));
        let h: Vec<f32> = raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        let (rows, cols) = (json_f(&j, "rows") as usize, json_f(&j, "cols") as usize);
        assert_eq!(h.len(), rows * cols, "{stem}");
        Self {
            lat0: json_f(&j, "lat0"),
            lon0: json_f(&j, "lon0"),
            dlat: json_f(&j, "dlat"),
            dlon: json_f(&j, "dlon"),
            rows,
            cols,
            h,
        }
    }

    fn at(&self, r: isize, c: isize) -> f64 {
        let r = r.clamp(0, self.rows as isize - 1) as usize;
        let c = c.clamp(0, self.cols as isize - 1) as usize;
        self.h[r * self.cols + c] as f64
    }

    /// Catmull-Rom bicubic height at (lat, lon) (deg).
    pub fn sample(&self, lat: f64, lon: f64) -> f64 {
        let (v, u) = ((self.lat0 - lat) / self.dlat, (lon - self.lon0) / self.dlon);
        let (r, c) = (v.floor(), u.floor());
        let (fv, fu) = (v - r, u - c);
        let w = |t: f64| {
            [
                ((-0.5 * t + 1.0) * t - 0.5) * t,
                (1.5 * t - 2.5) * t * t + 1.0,
                ((-1.5 * t + 2.0) * t + 0.5) * t,
                (0.5 * t - 0.5) * t * t,
            ]
        };
        let (wv, wu) = (w(fv), w(fu));
        let mut s = 0.0;
        for (b, wb) in wv.iter().enumerate() {
            for (a, wa) in wu.iter().enumerate() {
                s += wb * wa * self.at(r as isize - 1 + b as isize, c as isize - 1 + a as isize);
            }
        }
        s
    }

    /// Exactly flat 3x3 neighbourhood (Copernicus flattens water bodies to one height).
    pub fn flat(&self, lat: f64, lon: f64) -> bool {
        let (r, c) =
            (((self.lat0 - lat) / self.dlat).round() as isize, ((lon - self.lon0) / self.dlon).round() as isize);
        let h0 = self.at(r, c);
        (-1..=1).all(|j| (-1..=1).all(|i| self.at(r + j, c + i) == h0))
    }

    pub fn contains(&self, lat: f64, lon: f64) -> bool {
        let (v, u) = ((self.lat0 - lat) / self.dlat, (lon - self.lon0) / self.dlon);
        v >= 1.0 && u >= 1.0 && v < self.rows as f64 - 2.0 && u < self.cols as f64 - 2.0
    }
}

/// Bake the [`Ground`] (both levels, the default core) for the graft map.
pub fn bake_ground(map: &GraftMap) -> Ground {
    let outer = Dem::load(&format!("{DEM_DIR}/dem_outer"));
    let inner = Dem::load(&format!("{DEM_DIR}/dem_inner"));
    let (cx, cz) = map.world_xz;
    let mut levels = Vec::new();
    for (k, &(size, half)) in [super::graft::OUTER, super::graft::INNER].iter().enumerate() {
        let t = std::time::Instant::now();
        let n = size as usize;
        let step = 2.0 * half / n as f64;
        let h: Vec<f32> = (0..n * n)
            .into_par_iter()
            .with_min_len(n)
            .map(|q| {
                let (x, z) = (cx - half + ((q % n) as f64 + 0.5) * step, cz - half + ((q / n) as f64 + 0.5) * step);
                let (lat, lon) = planet::cube::dir_latlon(map.world_to_dir(x, z));
                let d = if k == 1 && inner.contains(lat, lon) { &inner } else { &outer };
                d.sample(lat, lon) as f32
            })
            .collect();
        eprintln!("ground level {k}: {n}^2 at {step:.2} m, {:.1} s", t.elapsed().as_secs_f64());
        levels.push(GroundLevel::new(n, half, h));
    }
    Ground::new(cx, cz, Core::default(), levels)
}

/// Load the ground file, baking (and caching) it first if missing.
pub fn load_or_bake(path: &str, map: &GraftMap) -> Ground {
    if let Ok(g) = Ground::load(path) {
        return g;
    }
    let t = std::time::Instant::now();
    let g = bake_ground(map);
    std::fs::write(path, g.to_bytes()).unwrap_or_else(|e| panic!("{path}: {e}"));
    eprintln!("ground baked: {path} ({:.1} s)", t.elapsed().as_secs_f64());
    g
}

fn srgb_lin(c: u8) -> f64 {
    let c = c as f64 / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

struct Merc {
    z: u32,
    x0: f64,
    y0: f64,
    w: usize,
    h: usize,
    rgb: Vec<u8>,
}

impl Merc {
    fn load(stem: &str) -> Self {
        let j = std::fs::read_to_string(format!("{stem}.json")).unwrap_or_else(|e| panic!("{stem}.json: {e}"));
        let rgb = std::fs::read(format!("{stem}.rgb")).unwrap_or_else(|e| panic!("{stem}.rgb: {e}"));
        let (w, h) = (json_f(&j, "w") as usize, json_f(&j, "h") as usize);
        assert_eq!(rgb.len(), w * h * 3, "{stem}");
        Self { z: json_f(&j, "z") as u32, x0: json_f(&j, "x0"), y0: json_f(&j, "y0"), w, h, rgb }
    }

    fn px(&self, lat: f64, lon: f64) -> (f64, f64) {
        let n = (256u64 << self.z) as f64;
        (
            (lon + 180.0) / 360.0 * n - self.x0 - 0.5,
            (1.0 - lat.to_radians().tan().asinh() / std::f64::consts::PI) / 2.0 * n - self.y0 - 0.5,
        )
    }

    fn inside(&self, lat: f64, lon: f64, margin: f64) -> bool {
        let (u, v) = self.px(lat, lon);
        u >= margin && v >= margin && u < self.w as f64 - 1.0 - margin && v < self.h as f64 - 1.0 - margin
    }

    /// Bilinear linear-light rgb.
    fn sample(&self, lat: f64, lon: f64) -> [f64; 3] {
        let (u, v) = self.px(lat, lon);
        let (i, j) =
            (u.floor().clamp(0.0, (self.w - 2) as f64) as usize, v.floor().clamp(0.0, (self.h - 2) as f64) as usize);
        let (fu, fv) = ((u - i as f64).clamp(0.0, 1.0), (v - j as f64).clamp(0.0, 1.0));
        let p = |a: usize, b: usize, c: usize| srgb_lin(self.rgb[((j + b) * self.w + i + a) * 3 + c]);
        [0, 1, 2].map(|c| {
            let t = p(0, 0, c) + (p(1, 0, c) - p(0, 0, c)) * fu;
            let b = p(0, 1, c) + (p(1, 1, c) - p(0, 1, c)) * fu;
            t + (b - t) * fv
        })
    }
}

/// The Sentinel-2 cloudless imagery (two zooms) and the DEM's water test.
pub struct Imagery {
    z11: Merc,
    z13: Merc,
    dem_in: Dem,
    dem_out: Dem,
    /// Per-channel gain to mountains-layer albedo (BMNG-matched; `GROUND_GAIN=r,g,b`).
    pub gain: [f64; 3],
    /// Assumed acquisition sun (world frame: x east, y up, z south).
    pub sun: DVec3,
    /// Delighting strength (0..1, `GROUND_DELIGHT`).
    pub delight: f64,
}

impl Imagery {
    pub fn load() -> Self {
        let gain = std::env::var("GROUND_GAIN")
            .ok()
            .map(|v| {
                let p: Vec<f64> = v.split(',').map(|t| t.trim().parse().expect("GROUND_GAIN")).collect();
                [p[0], p[1], p[2]]
            })
            // mean land linear albedo, BMNG July / S2 2020, over 44.7-47.5 N 5.5-9.7 E:
            // (1.146, 1.015, 1.045), x the planet's ALBEDO 0.85 / GRAFT_ALB 0.75
            .unwrap_or([1.30, 1.15, 1.18]);
        let delight = std::env::var("GROUND_DELIGHT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.75);
        // ~10:30 solar time in July at 46 N: elevation ~55 deg, azimuth ~130 deg (SE)
        let (az, el) = (130f64.to_radians(), 55f64.to_radians());
        let sun = DVec3::new(az.sin() * el.cos(), el.sin(), -az.cos() * el.cos());
        Self {
            z11: Merc::load(&format!("{S2_DIR}/s2_z11")),
            z13: Merc::load(&format!("{S2_DIR}/s2_z13")),
            dem_in: Dem::load(&format!("{DEM_DIR}/dem_inner")),
            dem_out: Dem::load(&format!("{DEM_DIR}/dem_outer")),
            gain,
            sun,
            delight,
        }
    }

    /// Mountains-layer albedo (linear) and water weight at a texel: `fine` picks z13 where it
    /// covers, the world normal `n` (y up) delights.
    pub fn albedo(&self, lat: f64, lon: f64, n: DVec3, fine: bool, step_m: f64) -> ([f64; 3], f64) {
        let m = if fine && self.z13.inside(lat, lon, 2.0) { &self.z13 } else { &self.z11 };
        // texel-footprint box: 2x2 sub-samples when the texel is larger than the image pixel
        let px_m = 40_075_016.0 * lat.to_radians().cos() / (256u64 << m.z) as f64;
        let c = if step_m > 1.2 * px_m {
            let (dl, dn) = (0.25 * step_m / 111_320.0, 0.25 * step_m / (111_320.0 * lat.to_radians().cos()));
            let mut s = [0.0; 3];
            for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                let v = m.sample(lat + a * dl, lon + b * dn);
                for k in 0..3 {
                    s[k] += 0.25 * v[k];
                }
            }
            s
        } else {
            m.sample(lat, lon)
        };
        let shade = (0.3 + 0.7 * n.dot(self.sun).max(0.0) / self.sun.y).clamp(0.3, 1.6);
        let k = (1.0 / shade).clamp(0.6, 2.0).powf(self.delight);
        let a = [0, 1, 2].map(|i| (c[i] * self.gain[i] * k).min(0.95));
        let d = if fine && self.dem_in.contains(lat, lon) { &self.dem_in } else { &self.dem_out };
        let lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let water = if d.flat(lat, lon) { 1.0 - ((lum - 0.03) / 0.04).clamp(0.0, 1.0) } else { 0.0 };
        (a, water)
    }
}
