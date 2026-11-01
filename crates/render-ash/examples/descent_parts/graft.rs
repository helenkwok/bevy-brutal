//! Shared Alps (Earth round 4): bake the mountains field into the planet renderer's graft
//! textures, and the graft's patch heights. Included by the `descent` and `earth` examples.
//!
//! The bake evaluates the same f64 [`Alpine`] field the mountains scene meshes, on two nested
//! grids around the vista (outer +-170 km at 83 m, inner +-42.5 km at 21 m; texel-sized filter
//! cells, the field's own band-limiting), and ports the mountains material
//! (`quality.slang`, `material()`: rock / scree / meadow / snow weights by height, slope,
//! cavity and convexity, the same constants) at the scale the mountains layer shows it from
//! tens of kilometres: the sub-50 m detail octaves average out and are left out, the macro
//! noises (>= 63 m) are evaluated with this crate's Perlin noise (same wavelengths and
//! amplitudes, different pattern). The sun visibility is a heightfield march toward the baked
//! sun (soft, 0.6 deg penumbra), the AO the mountains mesher's SDF probe at its far-ring scales.
#![allow(dead_code)]

use glam::DVec3;
use planet::graft::{GraftFile, GraftLevel, GraftMap};
use rayon::prelude::*;
use voxel::fields::Alpine;
use voxel::noise::perlin2;

pub const OUTER: (u32, f64) = (4096, 170_000.0);
pub const INNER: (u32, f64) = (4096, 42_500.0);
pub const R_FULL_M: f64 = 120_000.0;
pub const R_OUT_M: f64 = 165_000.0;
/// Shadow rays: max distance (m); relief < 3.4 km and a 29 deg sun cast < 6.2 km.
const SHADOW_M: f64 = 8000.0;

fn ss(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
fn sat(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}
fn lerp3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t)
}
fn mul3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| a[i] * b[i])
}
fn sc3(a: [f64; 3], k: f64) -> [f64; 3] {
    a.map(|v| v * k)
}

/// sRGB encode (linear 0..1 -> 0..255).
pub fn srgb8(c: f64) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

/// Value-noise stand-in for the shader's `vnoised(.).x` (~[-1, 1]).
fn vn(x: f64, z: f64, seed: u64) -> f64 {
    (perlin2(x, z, seed) * 1.4).clamp(-1.0, 1.0)
}

/// The mountains material's far-field albedo (linear): `quality.slang` lines ~905-1075 with the
/// footprint-faded (< 50 m) octaves at their mean.
pub fn albedo(x: f64, z: f64, h: f64, slope: f64, cav: f64, cvx: f64) -> [f64; 3] {
    let m1 = vn(x / 610.0, z / 610.0, 901);
    let m2 = vn(x / 190.0, z / 190.0, 902);
    let m3 = vn(x / 63.0, z / 63.0, 903);
    let macro_ = 0.55 * m1 + 0.3 * m2 + 0.15 * m3;
    let region = vn(x / 420.0, z / 420.0, 904) + 0.5 * vn(x / 130.0, z / 130.0, 905);
    let mut rock = lerp3([0.17, 0.165, 0.155], [0.32, 0.29, 0.25], sat(0.5 + 0.7 * region));
    rock = sc3(rock, 1.0 + 0.22 * macro_);
    let lichen = ss(0.15, 0.55, m3 + 0.4 * m2) * (1.0 - ss(0.55, 0.9, slope));
    rock = lerp3(rock, mul3(rock, [0.72, 0.74, 0.66]), 0.55 * lichen);
    // strata bands: the 80 m set at its mean tone on steep rock
    let steep = ss(0.18, 0.5, slope);
    rock = lerp3(rock, mul3(rock, [1.01, 0.945, 0.88]), steep);
    rock = lerp3(rock, mul3(rock, [0.75, 0.72, 0.70]), sat(cav * 2.0));
    let lowland = ss(1060.0, 880.0, h + 40.0 * m2);
    rock = sc3(rock, 1.0 + (0.72 - 1.0) * lowland);
    // moss on up-facing lowland outcrops (nr.y ~ 1 - slope)
    let moss = lowland * ss(0.30, 0.75, 1.0 - slope) * ss(-0.4, 0.3, m3);
    rock = lerp3(rock, [0.055, 0.075, 0.035], 0.85 * moss);
    let scree = sc3([0.33, 0.315, 0.29], (1.0 + 0.12 * m3) * (1.0 + (0.6 - 1.0) * lowland));
    let moist = sat(0.5 + 1.6 * cav - cvx + 0.5 * m2);
    let mut grass = lerp3([0.120, 0.158, 0.078], [0.068, 0.125, 0.040], moist);
    grass = sc3(grass, 1.0 + 0.18 * m3);
    grass = lerp3(grass, [0.27, 0.23, 0.15], sat((h - 1000.0) / 450.0 + 0.5 * cvx - 0.35 + 0.3 * m1));
    let shrub = ss(-0.1, 0.7, 0.5 * m3 + 0.3 * m2 + 0.4 * cav) * ss(1300.0, 1050.0, h);
    grass = lerp3(grass, [0.07, 0.09, 0.045], 0.45 * shrub);
    let snow = [0.86, 0.88, 0.92];
    let jitter = 60.0 * vn(x / 230.0, z / 230.0, 906) + 40.0 * m2;
    let mut snow_w = ss(1450.0, 1600.0, h + jitter + 120.0 * cav) * ss(0.40, 0.26, slope - 0.08 * cav);
    snow_w = snow_w.max(ss(1800.0, 2000.0, h + jitter) * ss(0.62, 0.46, slope));
    let grass_w = ss(1350.0, 1150.0, h + jitter) * ss(0.34, 0.15, slope - 0.05 * m3) * (1.0 - snow_w);
    let fan = sat(0.3 + 1.6 * cav - cvx + 0.6 * m2);
    let mut scree_w = ss(0.16, 0.30, slope) * ss(0.66, 0.42, slope) * fan * (1.0 - snow_w);
    scree_w *= (1.0 - grass_w) * (1.0 - 0.8 * lowland);
    let rock_w = sat(1.0 - snow_w - grass_w - scree_w);
    [0, 1, 2].map(|i| rock[i] * rock_w + scree[i] * scree_w + grass[i] * grass_w + snow[i] * snow_w)
}

/// Heights on a square grid (texel centres) with a margin, bilinear lookups.
struct Grid {
    n: usize,
    x0: f64,
    z0: f64,
    step: f64,
    h: Vec<f32>,
}
impl Grid {
    fn new(field: &Alpine, cx: f64, cz: f64, half: f64, size: u32, margin_m: f64) -> Self {
        let step = 2.0 * half / size as f64;
        let m = (margin_m / step).ceil() as usize;
        let n = size as usize + 2 * m;
        let (x0, z0) = (cx - half + 0.5 * step - m as f64 * step, cz - half + 0.5 * step - m as f64 * step);
        let h: Vec<f32> = (0..n * n)
            .into_par_iter()
            .with_min_len(n)
            .map(|k| field.height(x0 + (k % n) as f64 * step, z0 + (k / n) as f64 * step, step) as f32)
            .collect();
        Self { n, x0, z0, step, h }
    }
    fn at(&self, i: usize, j: usize) -> f64 {
        self.h[j.min(self.n - 1) * self.n + i.min(self.n - 1)] as f64
    }
    /// Bilinear height at world (x, z); None outside.
    fn sample(&self, x: f64, z: f64) -> Option<f64> {
        let (u, v) = ((x - self.x0) / self.step, (z - self.z0) / self.step);
        if u < 0.0 || v < 0.0 || u >= (self.n - 1) as f64 || v >= (self.n - 1) as f64 {
            return None;
        }
        let (i, j) = (u as usize, v as usize);
        let (fu, fv) = (u - i as f64, v - j as f64);
        let a = self.at(i, j) + (self.at(i + 1, j) - self.at(i, j)) * fu;
        let b = self.at(i, j + 1) + (self.at(i + 1, j + 1) - self.at(i, j + 1)) * fu;
        Some(a + (b - a) * fv)
    }
}

/// Real-imagery albedo for the grounded field (Earth round 5): `(lat, lon, world normal,
/// inner level?, texel m) -> (linear albedo, water weight)`.
pub type ImageryFn<'a> = &'a (dyn Fn(f64, f64, DVec3, bool, f64) -> ([f64; 3], f64) + Sync);

/// Bake both levels for a world-frame sun direction. With `imagery` (a grounded field), the
/// albedo blends from the material port to the imagery by the field's DEM weight, and the
/// albedo alpha carries 1 - water.
pub fn bake(
    field: &Alpine,
    map: &GraftMap,
    lat: f64,
    lon: f64,
    sun_world: DVec3,
    imagery: Option<ImageryFn>,
) -> GraftFile {
    let (cx, cz) = map.world_xz;
    let outer_grid = Grid::new(field, cx, cz, OUTER.1, OUTER.0, SHADOW_M + 1000.0);
    let mut levels = Vec::new();
    for (li, &(size, half)) in [OUTER, INNER].iter().enumerate() {
        let t = std::time::Instant::now();
        let g = if li == 0 { None } else { Some(Grid::new(field, cx, cz, half, size, SHADOW_M + 500.0)) };
        let g = g.as_ref().unwrap_or(&outer_grid);
        let step = g.step;
        let m = ((g.n - size as usize) / 2) as isize;
        let s = sun_world.normalize();
        let sh = DVec3::new(s.x, 0.0, s.z).normalize();
        let tan_el = s.y / s.x.hypot(s.z);
        let n = size as usize;
        let texels: Vec<([u8; 4], [u8; 4])> = (0..n * n)
            .into_par_iter()
            .with_min_len(n)
            .map(|k| {
                let (i, j) = ((k % n) as isize + m, (k / n) as isize + m);
                let hh = |a: isize, b: isize| g.at(a as usize, b as usize);
                let h = hh(i, j);
                let x = g.x0 + i as f64 * step;
                let z = g.z0 + j as f64 * step;
                let gx = (hh(i + 1, j) - hh(i - 1, j)) / (2.0 * step);
                let gz = (hh(i, j + 1) - hh(i, j - 1)) / (2.0 * step);
                let nrm = DVec3::new(-gx, 1.0, -gz).normalize();
                // cavity / convexity / AO: the mountains mesher's SDF probes (density y - h) at
                // its far-ring scales (cell = 0.004-0.008 x ring radius: ~150-400 m seen from
                // the descent's heights)
                let d = |p: DVec3| -> f64 {
                    let hs = g.sample(p.x, p.z).or_else(|| outer_grid.sample(p.x, p.z)).unwrap_or(h);
                    p.y - hs
                };
                let p = DVec3::new(x, h, z);
                let cs = (250.0f64).max(1.5 * step);
                let cav = ((cs - d(p + nrm * cs)) / cs).clamp(0.0, 1.0);
                let cvx = ((d(p - nrm * cs) + cs) / cs).clamp(0.0, 1.0);
                let (mut occ, mut ws) = (0.0, 0.0);
                for (kk, &f) in [0.75, 1.5, 3.0, 6.0f64].iter().enumerate() {
                    let sc = f * cs;
                    let w = 1.0 / (1.0 + kk as f64 * 0.35);
                    occ += w * ((sc - d(p + nrm * sc)) / sc).clamp(0.0, 1.0);
                    ws += w;
                }
                let ao = (1.0 - 1.15 * occ / ws).clamp(0.05, 1.0);
                // sun visibility: min elevation margin (rad) along the ray, soft over +-0.3 deg
                let vis = if nrm.dot(s) > 0.0 {
                    let mut t = step;
                    let mut margin = f64::MAX;
                    while t < SHADOW_M {
                        let (qx, qz) = (x + sh.x * t, z + sh.z * t);
                        let hs = if t < 1500.0 { g.sample(qx, qz) } else { None }.or_else(|| outer_grid.sample(qx, qz));
                        let Some(hs) = hs else { break };
                        margin = margin.min(((h + 2.0 + t * tan_el) - hs) / t);
                        if margin < -0.01 {
                            break;
                        }
                        t += (step * 0.75).max(t * 0.02);
                    }
                    ss(-0.005, 0.005, margin)
                } else {
                    0.0
                };
                let slope = 1.0 - nrm.y;
                let mut a = albedo(x, z, h, slope, cav, cvx);
                let mut water = 0.0;
                let wd = field.ground_weight(x, z);
                if let (Some(img), true) = (imagery, wd > 0.0) {
                    let (la, lo) = planet::cube::dir_latlon(map.world_to_dir(x, z));
                    let (ia, iw) = img(la, lo, nrm, li == 1, step);
                    // imagery leads the height blend: the procedural snow line on half-real
                    // heights would ring the core in white
                    a = lerp3(a, ia, ss(0.0, 0.6, wd));
                    water = iw * wd;
                }
                (
                    [srgb8(a[0]), srgb8(a[1]), srgb8(a[2]), ((1.0 - water) * 255.0).round() as u8],
                    [
                        ((nrm.x * 0.5 + 0.5) * 255.0).round() as u8,
                        ((nrm.z * 0.5 + 0.5) * 255.0).round() as u8,
                        (vis * 255.0).round() as u8,
                        (ao * 255.0).round() as u8,
                    ],
                )
            })
            .collect();
        let mut albedo_px = Vec::with_capacity(n * n * 4);
        let mut aux = Vec::with_capacity(n * n * 4);
        for (a, b) in texels {
            albedo_px.extend_from_slice(&a);
            aux.extend_from_slice(&b);
        }
        eprintln!("graft level {li}: {size}^2 at {:.1} m, {:.1} s", step, t.elapsed().as_secs_f64());
        levels.push(GraftLevel { size, half_m: half, albedo: albedo_px, aux });
    }
    GraftFile {
        lat,
        lon,
        world_xz: map.world_xz,
        r_full_m: R_FULL_M,
        r_out_m: R_OUT_M,
        sun_world: sun_world.to_array(),
        levels,
    }
}

/// Patch spacing (m) of a quadtree level: a face spans ~R pi/2 over 2^level patches of N quads.
pub fn level_spacing(level: u8) -> f64 {
    planet::EARTH_RADIUS_M * std::f64::consts::FRAC_PI_2 / (1u64 << level) as f64 / planet::cube::N as f64
}

/// Graft height blend: the real height `h_real` at direction `d`, the field's at the same point
/// (filtered at the patch's vertex spacing) weighted by the spatial weight x `strength`;
/// `blend` = (r_full, r_out, strength).
pub fn graft_height(
    field: &Alpine,
    map: &GraftMap,
    blend: (f64, f64, f64),
    d: DVec3,
    h_real: f64,
    spacing: f64,
) -> f64 {
    let (r0, r1, strength) = blend;
    if strength <= 0.0 {
        return h_real;
    }
    let (x, z, dist) = map.dir_to_world(d);
    let w = planet::graft::weight(dist, r0, r1) * strength;
    if w <= 0.0 {
        return h_real;
    }
    let hf = field.height(x, z, spacing.max(2.0));
    h_real + (hf - h_real) * w
}
