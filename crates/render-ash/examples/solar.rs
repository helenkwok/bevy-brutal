//! Solar system round 1: the Sun and the eight planets (+ the Moon) at one epoch, true scale.
//!
//! cargo run --release -p render-ash --example solar -- [--shot <name>|all|list] [--size 1920x1080]
//!     [--ss 2] [--out out/quality/steps] [--layers /workspace/data/solar/layers]
//!
//! The scene is [`planet::solar::System`]: the nested frames (galaxy > "sol" system frame, ~1 mm >
//! one planet frame per body), positions from the Standish elements (+ the AA Moon), IAU 2015 radii
//! and rotation, Mallama geometric albedos, all at the DSCOVR/EPIC epoch the Earth rounds use
//! (2026-07-15 11:07:35 UTC). Nothing is rescaled: sizes and distances are the real ones in metres.
//! What stays readable at system scale does so the way a camera sees it: bodies under ~0.75 px are
//! splatted as points of their true flux (p E (r/d)^2 Phi(alpha)), the Sun with a lens PSF (bloom),
//! stars from Yale BSC5 at their true brightness. Orbits / labels go only into the separate
//! `<name>_hud.png` annotation layer (tools/solar_hud.py, from the json).
//!
//! The Earth: far, the system pass's own LOD (Blue Marble July + the dated VIIRS clouds + a Rayleigh
//! veil); above `HANDOFF_PX` (Earth radius, output pixels) the `earth` example renders it (the
//! Earth round 2..7 planet path, same camera through the frame tree, `STARS=0 HDR_OUT=1`) and the
//! two HDR images are composited: system pixels where a nearer body hides the Earth (alpha -1),
//! else earth + earth.transmittance x system (stars, bodies behind the limb).
//!
//! Timings are lavapipe (CPU Vulkan), not a real-GPU number.
use glam::{DMat3, DVec3};
use planet::descent::AU_M;
use planet::solar::{self, BodyId, Epoch, Surface, System};
use rayon::prelude::*;
use render_ash::AshBackend;
use render_ash::system::{MAX_BODIES, SystemBodyGpu, SystemParamsGpu, SystemTextures};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Earth radius (output px) above which the planet renderer takes over.
const HANDOFF_PX: f64 = 24.0;
/// Bodies smaller than this (radius, render px) are splatted on the CPU.
const SPLAT_PX: f64 = 0.75;
const LAYERS: [&str; 11] = ["sun", "mercury", "venus", "earth", "earth_clouds", "moon", "mars", "jupiter", "saturn", "uranus", "neptune"];

fn layer_of(id: BodyId) -> f32 {
    match id {
        BodyId::Sun => 0.0,
        BodyId::Mercury => 1.0,
        BodyId::Venus => 2.0,
        BodyId::Earth => 3.0,
        BodyId::Moon => 5.0,
        BodyId::Mars => 6.0,
        BodyId::Jupiter => 7.0,
        BodyId::Saturn => 8.0,
        BodyId::Uranus => 9.0,
        BodyId::Neptune => 10.0,
    }
}

#[derive(Clone, Debug)]
struct Cam {
    pos: DVec3,
    fwd: DVec3,
    right: DVec3,
    up: DVec3,
    fov_y: f64,
}

fn cam(pos: DVec3, target: DVec3, up_hint: DVec3, fov_y: f64) -> Cam {
    let fwd = (target - pos).normalize();
    let hint = if fwd.cross(up_hint).length() < 1e-6 { DVec3::Y } else { up_hint };
    let right = fwd.cross(hint).normalize();
    let up = right.cross(fwd);
    Cam { pos, fwd, right, up, fov_y }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    /// The real system at the epoch.
    System,
    /// The true-size line-up (synthetic positions along a line, light from a fixed direction).
    LineUp,
}

#[derive(Clone, Debug)]
struct Shot {
    name: String,
    cam: Cam,
    exposure: f64,
    mode: Mode,
    /// occulting disc radius (solar radii, 0 = none) and corona gain
    occulter: f64,
    corona: bool,
    hud: bool,
    note: String,
    /// another epoch for this shot (default: the EPIC epoch)
    epoch: Option<Epoch>,
}

/// Texture-derived per-body calibration (mean linear luminance of each layer).
struct Calib {
    mean: Vec<[f64; 3]>,
}

fn load_png_rgba(p: &Path) -> (u32, u32, Vec<u8>) {
    let dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))));
    let mut r = dec.read_info().unwrap();
    let mut buf = vec![0u8; r.output_buffer_size().unwrap()];
    let info = r.next_frame(&mut buf).unwrap();
    let (w, h) = (info.width, info.height);
    let n = (w * h) as usize;
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf[..n * 4].to_vec(),
        png::ColorType::Rgb => buf[..n * 3].chunks(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect(),
        png::ColorType::Grayscale => buf[..n].iter().flat_map(|&c| [c, c, c, 255]).collect(),
        png::ColorType::GrayscaleAlpha => buf[..n * 2].chunks(2).flat_map(|c| [c[0], c[0], c[0], c[1]]).collect(),
        c => panic!("{}: colour type {c:?}", p.display()),
    };
    (w, h, rgba)
}

fn s2l(c: u8) -> f64 {
    let c = c as f64 / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// Area-weighted (cos lat) mean linear colour of an equirect layer.
fn mean_linear(w: u32, h: u32, px: &[u8]) -> [f64; 3] {
    let lut: Vec<f64> = (0..=255u8).map(s2l).collect();
    let (mut acc, mut wsum) = ([0.0f64; 3], 0.0f64);
    for y in 0..h {
        let lat = (0.5 - (y as f64 + 0.5) / h as f64) * std::f64::consts::PI;
        let cw = lat.cos();
        for x in (0..w).step_by(2) {
            let i = ((y * w + x) * 4) as usize;
            for c in 0..3 {
                acc[c] += lut[px[i + c] as usize] * cw;
            }
            wsum += cw;
        }
    }
    acc.map(|a| a / wsum)
}

fn lum(c: [f64; 3]) -> f64 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Disc-integrated phase law Phi(alpha) (alpha rad), 1 at full phase.
fn phase_law(b: &solar::Body, a: f64) -> f64 {
    let deg = a.to_degrees();
    match b.surface {
        // Allen's lunar / Mercury-like magnitude law
        Surface::Regolith => 10f64.powf(-0.4 * (0.026 * deg + 4.0e-9 * deg.powi(4))),
        Surface::Dusty => 10f64.powf(-0.4 * 0.016 * deg),
        Surface::Star => 1.0,
        _ => ((a.sin() + (std::f64::consts::PI - a) * a.cos()) / std::f64::consts::PI).max(0.0),
    }
}

/// Visible-light disc colour targets (sRGB of the disc mean) and texture contrast, where the map
/// is not a natural-colour product: the ice giants after Irwin et al. 2024 (reprocessed Voyager /
/// HST: both pale green-blue, Neptune only slightly bluer than Uranus; the map's deep blue is the
/// contrast-stretched Voyager look), Venus (near-featureless pale cream in visible light; the map is
/// a UV / enhanced cloud product), Mars (butterscotch, not the map's saturated rust).
fn true_colour(id: BodyId) -> Option<([u8; 3], f64)> {
    match id {
        BodyId::Uranus => Some(([172, 211, 216], 1.0)),
        BodyId::Neptune => Some(([150, 190, 216], 1.0)),
        BodyId::Venus => Some(([232, 220, 190], 0.18)),
        BodyId::Mars => Some(([190, 132, 95], 1.0)),
        _ => None,
    }
}

/// Replace a layer's chroma: linear = target x (luminance / mean luminance).
fn colourize(px: &mut [u8], target: [f64; 3]) {
    let lut: Vec<f64> = (0..=255u8).map(s2l).collect();
    let lums: Vec<f64> = px.chunks(4).map(|p| lum([lut[p[0] as usize], lut[p[1] as usize], lut[p[2] as usize]])).collect();
    let mean = lums.iter().sum::<f64>() / lums.len() as f64;
    let tl = lum(target);
    for (p, l) in px.chunks_mut(4).zip(lums) {
        for c in 0..3 {
            let v = (target[c] / tl * l * tl / mean).clamp(0.0, 1.0);
            let s = if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
            p[c] = (s * 255.0 + 0.5) as u8;
        }
    }
}

/// Pull a layer's linear values toward its mean by `k` (1 = unchanged).
fn soften(px: &mut [u8], mean: [f64; 3], k: f64) {
    let lut: Vec<f64> = (0..=255u8).map(s2l).collect();
    for p in px.chunks_mut(4) {
        for c in 0..3 {
            let v = mean[c] + (lut[p[c] as usize] - mean[c]) * k;
            let v = v.clamp(0.0, 1.0);
            let s = if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
            p[c] = (s * 255.0 + 0.5) as u8;
        }
    }
}

/// Per-body shading constants: (albedo scale, tint, k / L, haze strength, haze H, HG g, haze mix, phase decay).
fn shading(b: &solar::Body, cal: &Calib) -> ([f32; 4], [f32; 4], [f32; 4]) {
    let li = layer_of(b.id) as usize;
    let mean = cal.mean[li];
    let mut tint = [1.0f64; 3];
    if let Some((t, _)) = true_colour(b.id) {
        let target = t.map(s2l);
        // per-channel ratio, normalised so the luminance is set by the albedo below
        let r = [target[0] / mean[0], target[1] / mean[1], target[2] / mean[2]];
        let k = lum(mean) / lum([mean[0] * r[0], mean[1] * r[1], mean[2] * r[2]]);
        tint = r.map(|v| v * k);
    }
    let k = b.shade_k;
    // geometric albedo p = A x the model's full-phase disc integral
    let factor = match b.surface {
        Surface::Regolith | Surface::Dusty => (1.0 - k) * 2.0 / 3.0 + k,
        Surface::Cloud | Surface::Giant => 2.0 / (2.0 * k + 1.0),
        _ => 2.0 / 3.0,
    };
    let scale = match b.surface {
        Surface::Earth => 1.0, // Blue Marble taken as surface reflectance; clouds from VIIRS
        Surface::Star => 1.0,
        _ => b.geometric_albedo / factor / lum(mean),
    };
    let kind = match b.surface {
        Surface::Star => 0.0,
        Surface::Regolith => 1.0,
        Surface::Dusty => 2.0,
        Surface::Cloud => 3.0,
        Surface::Giant => 4.0,
        Surface::Earth => 5.0,
    };
    // haze: (strength, scale height m, HG g, colour mix)
    let (hz, hh, g, mix) = match b.id {
        BodyId::Venus => (1.0, 5.0e3, 0.7, 0.4),
        BodyId::Mars => (0.35, 11.0e3, 0.6, 0.0),
        BodyId::Jupiter => (0.15, 27.0e3, 0.4, 0.6),
        BodyId::Saturn => (0.15, 60.0e3, 0.4, 0.6),
        BodyId::Uranus => (0.25, 28.0e3, 0.3, 1.0),
        BodyId::Neptune => (0.25, 20.0e3, 0.3, 1.0),
        BodyId::Earth => (1.0, 8.0e3, 0.0, 0.0),
        _ => (0.0, 0.0, 0.0, 0.0),
    };
    let shade = [scale as f32, k as f32, hz as f32, kind];
    let tint = [tint[0] as f32, tint[1] as f32, tint[2] as f32, hh as f32];
    let extra = [0.0, g as f32, mix as f32, 1.2];
    (shade, tint, extra)
}

/// Saturn's ring: effective reflectance gain (the slab single-scattering formula times the
/// opposition / multiple-scattering excess), calibrated to the B ring's I/F ~0.45 at low phase.
const RING_GAIN: f32 = 2.2;

struct Frame {
    params: SystemParamsGpu,
    /// (body index, render-px radius, skipped, centre rel camera)
    bodies: Vec<(usize, f64, bool, DVec3)>,
    earth_handoff: bool,
    ty: f64,
    aspect: f64,
}

fn rows(m: DMat3) -> [[f64; 3]; 3] {
    [m.row(0).to_array(), m.row(1).to_array(), m.row(2).to_array()]
}

/// Line-up layout: (position, rotation) per body, Sun limb at the left edge.
fn lineup(sys: &System) -> Vec<(DVec3, DMat3)> {
    let order = [BodyId::Sun, BodyId::Mercury, BodyId::Venus, BodyId::Earth, BodyId::Moon, BodyId::Mars, BodyId::Jupiter, BodyId::Saturn, BodyId::Uranus, BodyId::Neptune];
    let mut x = 0.0f64;
    let mut out = vec![(DVec3::ZERO, DMat3::IDENTITY); sys.bodies.len()];
    for id in order {
        let i = sys.bodies.iter().position(|b| b.0.id == id).unwrap();
        let b = &sys.bodies[i].0;
        let r = if b.ring_m.1 > 0.0 { b.ring_m.1 * 0.6 } else { b.r_eq_m };
        let gap = match id {
            BodyId::Sun => 0.0,
            BodyId::Moon => 4.0e6,
            _ => 9.0e6,
        };
        let cx = if id == BodyId::Sun { -b.r_eq_m + 1.4e7 } else { x + gap + r };
        x = if id == BodyId::Sun { 1.4e7 } else { cx + r };
        // body-fixed axes: pole tilted 25 deg toward the camera (+Z is toward the camera, the
        // line runs along +X), prime meridian facing the camera
        let tilt = if id == BodyId::Saturn { 22.0f64 } else { 12.0 };
        let pole = DVec3::new(0.0, tilt.to_radians().cos(), tilt.to_radians().sin());
        let xb = DVec3::new(0.0, -pole.z, pole.y).cross(pole).normalize();
        let xb = if xb.z < 0.0 { -xb } else { xb };
        let yb = pole.cross(xb);
        // fixed_from_world: rows = body axes in world coordinates
        let m = DMat3::from_cols(xb, yb, pole).transpose();
        out[i] = (DVec3::new(cx, 0.0, 0.0), m);
    }
    out
}

fn build_frame(sys: &System, cal: &Calib, s: &Shot, rw: u32, rh: u32, ss: u32) -> Frame {
    let c = &s.cam;
    let aspect = rw as f64 / rh as f64;
    let ty = (c.fov_y.to_radians() / 2.0).tan();
    let px_rad = 2.0 * ty / rh as f64;
    let layout = (s.mode == Mode::LineUp).then(|| lineup(sys));
    let mut p = SystemParamsGpu {
        right: [c.right.x as f32, c.right.y as f32, c.right.z as f32, (ty * aspect) as f32],
        up: [c.up.x as f32, c.up.y as f32, c.up.z as f32, ty as f32],
        fwd: [c.fwd.x as f32, c.fwd.y as f32, c.fwd.z as f32, s.exposure as f32],
        ..Default::default()
    };
    let sun_pos = layout.as_ref().map_or(sys.bodies[0].2, |l| l[0].0);
    let sun_rel = sun_pos - c.pos;
    p.sun = [sun_rel.x as f32, sun_rel.y as f32, sun_rel.z as f32, solar::SUN_RADIUS_M as f32];
    let mut flags = 0u32;
    if s.corona {
        flags |= 1;
    }
    if s.mode == Mode::LineUp {
        flags |= 2 | 4;
        // fixed light: upper left, slightly from the camera side (a 30 deg phase)
        let l = DVec3::new(-0.45, 0.35, 0.82).normalize();
        p.light = [l.x as f32, l.y as f32, l.z as f32, 1.0];
    }
    p.view = [sys.bodies.len() as f32, flags as f32, AU_M as f32, 0.0];
    p.corona = [(s.occulter * solar::SUN_RADIUS_M / sun_rel.length()).asin() as f32 * (s.occulter > 0.0) as u8 as f32, 1.0, 0.8, 1.0];
    let sp = sys.bodies[0].0.pole_ecliptic(sys.epoch);
    p.sunpole = [sp.x as f32, sp.y as f32, sp.z as f32, 0.06];
    p.limb = [0.45, 0.62, 0.85, px_rad as f32];
    let mut bodies = Vec::new();
    let mut earth_handoff = false;
    for (i, (b, _f, pos, rot)) in sys.bodies.iter().enumerate().take(MAX_BODIES) {
        let (pos, rot) = layout.as_ref().map_or((*pos, *rot), |l| l[i]);
        let rel = pos - c.pos;
        let d = rel.length();
        let r_px = (b.r_eq_m / d).asin() / px_rad;
        let ring_px = (b.ring_m.1 / d).min(1.0).asin() / px_rad;
        let skip = r_px.max(ring_px) < SPLAT_PX;
        let (shade, tint, mut extra) = shading(b, cal);
        let handoff = b.id == BodyId::Earth && s.mode == Mode::System && r_px / ss as f64 > HANDOFF_PX;
        extra[0] = if handoff { 2.0 } else if skip { 1.0 } else { 0.0 };
        earth_handoff |= handoff;
        let r = rows(rot);
        let ring = if b.ring_m.1 > 0.0 { [b.ring_m.0 as f32, b.ring_m.1 as f32, RING_GAIN, 0.0] } else { [0.0, 0.0, 0.0, -1.0] };
        p.bodies[i] = SystemBodyGpu {
            pos: [rel.x as f32, rel.y as f32, rel.z as f32, b.r_eq_m as f32],
            r0: [r[0][0] as f32, r[0][1] as f32, r[0][2] as f32, (b.r_pol_m / b.r_eq_m) as f32],
            r1: [r[1][0] as f32, r[1][1] as f32, r[1][2] as f32, layer_of(b.id)],
            r2: [r[2][0] as f32, r[2][1] as f32, r[2][2] as f32, if b.id == BodyId::Earth { 4.0 } else { -1.0 }],
            shade,
            tint,
            ring,
            extra,
        };
        bodies.push((i, r_px, skip && !handoff, rel));
    }
    Frame { params: p, bodies, earth_handoff, ty, aspect }
}


struct Star {
    dir: DVec3,
    flux: f64,
    rgb: [f32; 3],
}

/// BSC5 (J2000 RA / Dec) -> ecliptic J2000 directions.
fn load_stars(path: &Path) -> Vec<Star> {
    let Ok(txt) = std::fs::read(path) else { return Vec::new() };
    let m = solar::ecliptic_from_equatorial();
    let mut out = Vec::new();
    for l in txt.split(|&b| b == b'\n') {
        let f = |a: usize, b: usize| -> Option<f64> { std::str::from_utf8(l.get(a - 1..b)?).ok()?.trim().parse().ok() };
        let (Some(rh), Some(rm), Some(rs), Some(dd), Some(dm), Some(ds), Some(v)) = (f(76, 77), f(78, 79), f(80, 83), f(85, 86), f(87, 88), f(89, 90), f(103, 107)) else { continue };
        let sign = if l.get(83) == Some(&b'-') { -1.0 } else { 1.0 };
        let ra = (rh + rm / 60.0 + rs / 3600.0) * 15.0;
        let dec = sign * (dd + dm / 60.0 + ds / 3600.0);
        let bv = f(110, 114).unwrap_or(0.6).clamp(-0.4, 2.0);
        let t = ((bv + 0.4) / 2.4) as f32;
        let rgb = [0.75 + 0.5 * t, 0.85 + 0.1 * (1.0 - (2.0 * t - 1.0).abs()), 1.25 - 0.9 * t];
        let y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        out.push(Star { dir: m * planet::cube::latlon_dir(dec, ra), flux: 10f64.powf(-0.4 * (v + 26.74)), rgb: rgb.map(|c| c / y) });
    }
    out
}

/// Project a camera-relative direction to render pixels (None behind / off-frame by > margin).
fn project(c: &Cam, ty: f64, aspect: f64, rw: u32, rh: u32, d: DVec3) -> Option<(f64, f64)> {
    let z = d.dot(c.fwd);
    if z <= 0.0 {
        return None;
    }
    let x = d.dot(c.right) / z / (ty * aspect);
    let y = -d.dot(c.up) / z / ty;
    Some(((x * 0.5 + 0.5) * rw as f64, (y * 0.5 + 0.5) * rh as f64))
}

fn splat(hdr: &mut [[f32; 4]], rw: u32, rh: u32, fx: f64, fy: f64, l: f32, rgb: [f32; 3]) -> bool {
    let (fx, fy) = (fx - 0.5, fy - 0.5);
    let (ix, iy) = (fx.floor() as i64, fy.floor() as i64);
    if ix < -1 || iy < -1 || ix >= rw as i64 || iy >= rh as i64 {
        return false;
    }
    let (ax, ay) = (fx - ix as f64, fy - iy as f64);
    for (dx, dy, wgt) in [(0, 0, (1.0 - ax) * (1.0 - ay)), (1, 0, ax * (1.0 - ay)), (0, 1, (1.0 - ax) * ay), (1, 1, ax * ay)] {
        let (xx, yy) = (ix + dx, iy + dy);
        if xx < 0 || yy < 0 || xx >= rw as i64 || yy >= rh as i64 {
            continue;
        }
        let q = &mut hdr[(yy as u32 * rw + xx as u32) as usize];
        let tr = q[3].max(0.0);
        for (qk, ck) in q.iter_mut().zip(rgb) {
            *qk += l * wgt as f32 * tr * ck;
        }
    }
    true
}

/// Separable Gaussian (three box passes) of an RGB image.
fn gauss(img: &[[f32; 3]], w: usize, h: usize, sigma: f64) -> Vec<[f32; 3]> {
    let r = (((12.0 * sigma * sigma / 3.0 + 1.0).sqrt() - 1.0) / 2.0).round().max(1.0) as usize;
    let boxh = |src: &[[f32; 3]], w: usize, h: usize| -> Vec<[f32; 3]> {
        let mut out = vec![[0.0f32; 3]; w * h];
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let s = &src[y * w..(y + 1) * w];
            let mut acc = [0.0f64; 3];
            let at = |i: i64| s[i.clamp(0, w as i64 - 1) as usize];
            for i in -(r as i64)..=(r as i64) {
                for (a, v) in acc.iter_mut().zip(at(i)) {
                    *a += v as f64;
                }
            }
            let n = (2 * r + 1) as f64;
            for (x, o) in row.iter_mut().enumerate() {
                *o = [(acc[0] / n) as f32, (acc[1] / n) as f32, (acc[2] / n) as f32];
                let (add, sub) = (at(x as i64 + r as i64 + 1), at(x as i64 - r as i64));
                for c in 0..3 {
                    acc[c] += add[c] as f64 - sub[c] as f64;
                }
            }
        });
        out
    };
    let transpose = |src: &[[f32; 3]], w: usize, h: usize| -> Vec<[f32; 3]> {
        let mut out = vec![[0.0f32; 3]; w * h];
        out.par_chunks_mut(h).enumerate().for_each(|(x, col)| {
            for (y, o) in col.iter_mut().enumerate() {
                *o = src[y * w + x];
            }
        });
        out
    };
    let mut a = img.to_vec();
    for _ in 0..3 {
        a = boxh(&a, w, h);
    }
    let mut t = transpose(&a, w, h);
    for _ in 0..3 {
        t = boxh(&t, h, w);
    }
    transpose(&t, h, w)
}

fn aces(x: f32) -> f32 {
    let (a, b, c, d, e) = (2.51, 0.03, 2.43, 0.59, 0.14);
    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}

fn write_png(path: &Path, w: u32, h: u32, rgb: &[u8]) {
    let file = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(rgb).unwrap();
}

fn v3(v: DVec3) -> String {
    format!("[{:.6e},{:.6e},{:.6e}]", v.x, v.y, v.z)
}

/// Close-up camera: phase angle (deg, Sun-body-camera), elevation off the body's equator toward
/// its pole (deg, sign chosen to the sunlit side when `sun_side`), frame fill.
#[allow(clippy::too_many_arguments)] // a shot description: body, phase, elevation, side, fill, fov, aspect
fn closeup(sys: &System, id: BodyId, phase: f64, elev: f64, sun_side: bool, fill: f64, fov: f64, aspect: f64) -> Cam {
    let (b, _, pos, _) = sys.get(id);
    let pole = b.pole_ecliptic(sys.epoch);
    let s = (-*pos).normalize();
    // swing the sunward direction about the pole by the phase angle (stays near the equator),
    // then lift toward the pole (toward the sunlit pole / ring face when `sun_side`)
    // Uranus in 2026 (4 deg from its 2028 solstice) has its pole ~22 deg from the Sun: swinging
    // about the pole would then not change the phase, so swing about the ecliptic normal instead
    let c = if s.dot(pole).abs() > 0.5 {
        let c0 = DMat3::from_axis_angle(DVec3::Z, phase.to_radians()) * s;
        let c0 = (c0 - DVec3::Z * c0.dot(DVec3::Z)).normalize();
        (c0 * elev.to_radians().cos() + DVec3::Z * elev.to_radians().sin()).normalize()
    } else {
        let c0 = DMat3::from_axis_angle(pole, phase.to_radians()) * s;
        let c0 = (c0 - pole * c0.dot(pole)).normalize();
        let sign = if sun_side { s.dot(pole).signum() } else { 1.0 };
        (c0 * elev.to_radians().cos() + pole * sign * elev.to_radians().sin()).normalize()
    };
    let ty = (fov.to_radians() / 2.0).tan();
    let r = if b.ring_m.1 > 0.0 { b.ring_m.1 } else { b.r_eq_m };
    let d = (r / (fill * ty)).max(r / (fill * ty * aspect) * if b.ring_m.1 > 0.0 { 1.0 } else { 0.0 });
    let d = if b.ring_m.1 > 0.0 { r / (fill * ty * aspect) } else { d };
    let p = *pos + c * d;
    // the pole on the camera's side up (Cassini-style: the near ring arc crosses the lower globe)
    let up = if pole.dot(c) < 0.0 && sun_side { -pole } else { pole };
    cam(p, *pos, up, fov)
}

fn shots(sys: &System, aspect: f64) -> Vec<Shot> {
    let ep = sys.epoch;
    let sun = sys.bodies[0].2;
    let earth = sys.get(BodyId::Earth).2;
    let moon = sys.get(BodyId::Moon).2;
    let z = DVec3::Z;
    let mut v = Vec::new();
    let sh = |name: &str, cam: Cam, exposure: f64, note: &str| Shot { name: name.into(), cam, exposure, mode: Mode::System, occulter: 0.0, corona: false, hud: false, note: note.into(), epoch: None };
    // camera azimuth: behind the Sun as seen from the Earth (the Earth beyond the Sun, top of frame)
    let az = (-earth).normalize();
    let elev = |e: f64| (az * e.to_radians().cos() + z * e.to_radians().sin()).normalize();
    let mut ov = sh("SS1_overview", cam(sun + elev(60.0) * 50.0 * AU_M, sun, z, 60.0), 3.0e4, "50 AU from the Sun, 60 deg above the ecliptic, 60 deg fov: the whole system at true scale");
    ov.hud = true;
    ov.corona = true;
    v.push(ov);
    let mut inner = sh("SS1_inner", cam(sun + elev(50.0) * 3.2 * AU_M, sun, z, 60.0), 3.0e4, "3.2 AU, 50 deg above the ecliptic: Mercury to Mars");
    inner.hud = true;
    inner.corona = true;
    v.push(inner);
    // the approach: distance (log) 50 AU -> the EPIC frame, direction slerp, target Sun -> Earth
    let epic_pos = DVec3::new(13.389f64.to_radians().cos() * 22.513f64.to_radians().cos(), 13.389f64.to_radians().cos() * 22.513f64.to_radians().sin(), 13.389f64.to_radians().sin())
        * (planet::EARTH_RADIUS_M + 1_475_183.0e3);
    let d_end = solar::ecliptic_from_ecef(ep, epic_pos).normalize();
    let dist_end = epic_pos.length();
    let up_end = solar::ecliptic_from_ecef(ep, DVec3::Z);
    let d0 = elev(60.0);
    let n = 7;
    for k in 0..n {
        let u = k as f64 / (n - 1) as f64;
        let dist = (50.0 * AU_M).ln() * (1.0 - u) + dist_end.ln() * u;
        let dist = dist.exp();
        let fov = (60.0f64.ln() * (1.0 - u) + 0.6239f64.ln() * u).exp();
        let st = (u / 0.55).clamp(0.0, 1.0);
        let st = st * st * (3.0 - 2.0 * st);
        let target = sun.lerp(earth, st);
        let om = d0.dot(d_end).clamp(-1.0, 1.0).acos();
        let su = u.powf(0.8);
        let dir = ((d0 * ((1.0 - su) * om).sin() + d_end * (su * om).sin()) / om.sin()).normalize();
        let up = z.lerp(up_end, su).normalize();
        let pos = target + dir * dist;
        // auto-exposure keyed to the Earth: 2.4 once it is a >= 2 px disc, (2 / r_px)^2 brighter
        // while smaller (keeps its integrated signal), capped at the overview's 3e4
        let r_px = (planet::EARTH_RADIUS_M / (earth - pos).length()) / (fov.to_radians() / 1080.0);
        let exposure = (2.4 * (2.0 / r_px).max(1.0).powi(2)).min(3.0e4);
        let mut s = sh(&format!("SS1_path_{k:02}"), cam(pos, target, up, fov), exposure, "approach: log distance, slerped direction, target Sun -> Earth; last = the E2_vs_epic geometry");
        s.hud = k < n - 2;
        s.corona = k < 3;
        v.push(s);
    }
    // the Sun: from the Earth's distance, limb darkening; and the occulted corona
    let mut sun_shot = sh("SS1_sun", cam(earth, sun, z, 0.75), 4.0e-5, "the photosphere from 1 AU: limb darkening (I = I0 mu^a, a = 0.45 / 0.62 / 0.85 r/g/b), granulation map at 6% contrast");
    sun_shot.corona = true;
    v.push(sun_shot);
    let mut cor = sh("SS1_corona", cam(earth, sun, z, 3.0), 60.0, "occulting disc 1.2 R_sun, K + F corona (Baumbach), noise streamers: ~1e-6 of the disc, so only an occulted, 1.5e6x brighter exposure shows it");
    cor.occulter = 1.2;
    cor.corona = true;
    v.push(cor);
    // close-ups: exposure normalised to the Earth's (2.4 at 1 AU, p 0.434)
    let fov = 10.0;
    for (id, phase, el, side, fill) in [
        (BodyId::Mercury, 55.0, 10.0, false, 0.8),
        (BodyId::Venus, 70.0, 10.0, false, 0.8),
        (BodyId::Mars, 40.0, 15.0, false, 0.8),
        (BodyId::Jupiter, 35.0, 10.0, false, 0.8),
        (BodyId::Saturn, 40.0, 20.0, true, 0.85),
        (BodyId::Uranus, 30.0, 10.0, false, 0.8),
        (BodyId::Neptune, 30.0, 10.0, false, 0.8),
    ] {
        let (b, _, pos, _) = sys.get(id);
        let r_au = pos.length() / AU_M;
        let e = 2.4 * r_au * r_au * 0.434 / b.geometric_albedo;
        v.push(sh(&format!("SS1_{}", id.name()), closeup(sys, id, phase, el, side, fill, fov, aspect), e, &format!("flyby, phase {phase} deg, true colours from the maps x the Mallama albedo; exposure scaled by r^2 / p to the Earth's")));
    }
    // Saturn at the Cassini Grand Finale (2017-04-26, rings open ~26 deg to the Sun) for comparison:
    // in 2026 the Sun is only ~6.5 deg below the ring plane (after the 2025 equinox) and the rings are dim
    {
        let ep17 = Epoch::utc(2017, 4, 26, 9, 0, 0.0);
        let s17 = System::new(ep17);
        let (b, _, pos, _) = s17.get(BodyId::Saturn);
        let r_au = pos.length() / AU_M;
        let e = 2.4 * r_au * r_au * 0.434 / b.geometric_albedo;
        let mut s = sh("SS1_saturn_2017", closeup(&s17, BodyId::Saturn, 40.0, 20.0, true, 0.85, fov, aspect), e, "Saturn at 2017-04-26 (Cassini Grand Finale epoch, Sun 26 deg above the rings): comparison for the 2026 shot");
        s.epoch = Some(ep17);
        v.push(s);
    }
    // the Moon in front of the Earth from the sunward side (both near full: the epoch is ~1 day after new moon)
    {
        let u = (moon - earth).normalize();
        let side = u.cross(z).normalize();
        // 1.2e9 m from the Earth, 0.82e9 m from the Moon, 5.4e7 m off the line: the Moon ~1.2 deg beside the Earth
        let pos = earth + u * 1.2e9 + side * 5.4e7 + z * 1.0e7;
        let target = earth.lerp(moon, 0.5);
        v.push(sh("SS1_moon_earth", cam(pos, target, up_end, 2.2), 2.4, "from beyond the Moon, near the Sun line: both near full (epoch ~1 day after new moon); the Moon (system pass) beside the Earth (planet renderer, handoff composite), true relative size and albedo (0.12 vs 0.43)"));
    }
    // the true-size line-up
    {
        let pos = DVec3::new(2.7e8, 0.0, 8.0e11);
        let w_m = 5.5e8;
        let fov = 2.0 * ((w_m / aspect / 2.0) / pos.z).atan().to_degrees();
        let mut s = sh("SS1_scale_strip", cam(pos, DVec3::new(2.7e8, 0.0, 0.0), DVec3::Y, fov), 1.6, "true relative sizes (synthetic line-up, fixed light, no shadows): Sun limb at the left");
        s.mode = Mode::LineUp;
        v.push(s);
    }
    v
}

struct EarthRun {
    hdr: Vec<[f32; 4]>,
    wall_ms: f64,
    json: String,
}

fn run_earth(bin: &Path, sys: &System, s: &Shot, w: u32, h: u32, ss: u32, tmp: &Path) -> EarthRun {
    let ep = sys.epoch;
    let pos = sys.to_earth_ecef(s.cam.pos);
    let f = solar::ecef_from_ecliptic(ep, s.cam.fwd);
    let u = solar::ecef_from_ecliptic(ep, s.cam.up);
    let (sla, slo) = solar::earth_subsolar(ep);
    let t = Instant::now();
    let st = std::process::Command::new(bin)
        .args(["--pack", "/workspace/data/earth/earth.pln", "--out"])
        .arg(tmp)
        .args(["--size", &format!("{w}x{h}"), "--ss", &ss.to_string(), "--fov", &format!("{}", s.cam.fov_y), "--name", &s.name])
        .args(["--cam-ecef", &format!("{},{},{},{},{},{},{},{},{}", pos.x, pos.y, pos.z, f.x, f.y, f.z, u.x, u.y, u.z)])
        .env("STARS", "0")
        .env("HDR_OUT", "1")
        .env("EXPOSURE", format!("{}", s.exposure))
        .env("SUN_LAT", format!("{sla}"))
        .env("SUN_LON", format!("{slo}"))
        .output()
        .expect("earth example");
    let wall_ms = t.elapsed().as_secs_f64() * 1e3;
    let out = String::from_utf8_lossy(&st.stdout).to_string() + &String::from_utf8_lossy(&st.stderr);
    assert!(st.status.success(), "earth handoff failed: {out}");
    let bytes = std::fs::read(tmp.join(format!("{}.hdr.bin", s.name))).expect("earth hdr");
    let hdr: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, [f32; 4]>(&bytes).to_vec();
    assert_eq!(hdr.len(), (w * ss * h * ss) as usize);
    let json = out.lines().rev().find(|l| l.starts_with("{\"shot\"")).unwrap_or("null").to_string();
    EarthRun { hdr, wall_ms, json }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let which = arg("--shot").unwrap_or("all".into());
    let out = PathBuf::from(arg("--out").unwrap_or("out/quality/steps".into()));
    let layers_dir = PathBuf::from(arg("--layers").unwrap_or("/workspace/data/solar/layers".into()));
    let (w, h) = arg("--size").and_then(|s| s.split_once('x').map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap()))).unwrap_or((1920u32, 1080u32));
    let ss: u32 = arg("--ss").and_then(|s| s.parse().ok()).unwrap_or(2);
    let (rw, rh) = (w * ss, h * ss);
    let ep = Epoch::epic();
    let sys = System::new(ep);
    let all = shots(&sys, w as f64 / h as f64);
    if which == "list" {
        for s in &all {
            println!("{}", s.name);
        }
        return;
    }
    let list: Vec<Shot> = all.into_iter().filter(|s| which == "all" || which.split(',').any(|n| n == s.name)).collect();
    assert!(!list.is_empty(), "no shot named {which}");
    std::fs::create_dir_all(&out).unwrap();
    let earth_bin = arg("--earth-bin").map(PathBuf::from).unwrap_or_else(|| std::env::current_exe().unwrap().with_file_name("earth"));
    let validate = std::env::var("PLANET_VALIDATE").map(|v| v != "0").unwrap_or(true);
    let mut be = AshBackend::new(render_ash::BackendConfig { max_instances: 16, validation: validate, ..Default::default() }, None).expect("vulkan");
    // textures
    let tl = Instant::now();
    let loaded: Vec<(u32, u32, Vec<u8>)> = LAYERS.par_iter().map(|n| load_png_rgba(&layers_dir.join(format!("{n}.png")))).collect();
    let (lw, lh) = (loaded[0].0, loaded[0].1);
    assert!(loaded.iter().all(|l| l.0 == lw && l.1 == lh), "layers must share one size");
    let mut loaded = loaded;
    for id in BodyId::ALL {
        let l = &mut loaded[layer_of(id) as usize];
        if matches!(id, BodyId::Uranus | BodyId::Neptune)
            && let Some((t, _)) = true_colour(id)
        {
            // ice giants: keep the map's luminance structure, take the chroma from the target (a
            // per-channel ratio would blow the map's near-zero red up into purple fringes)
            colourize(&mut l.2, t.map(s2l));
        }
        if let Some((_, k)) = true_colour(id)
            && k < 1.0
        {
            let m = mean_linear(l.0, l.1, &l.2);
            soften(&mut l.2, m, k);
        }
    }
    let cal = Calib { mean: loaded.iter().map(|l| mean_linear(l.0, l.1, &l.2)).collect() };
    let ring = load_png_rgba(&layers_dir.join("saturn_ring.png"));
    let layer_px: Vec<Vec<u8>> = loaded.into_iter().map(|l| l.2).collect();
    let tex: SystemTextures = be.system_upload(lw, lh, &layer_px, ring.0, ring.1, std::slice::from_ref(&ring.2)).expect("upload");
    drop(layer_px);
    let load_ms = tl.elapsed().as_secs_f64() * 1e3;
    eprintln!("solar: {} layers {lw}x{lh} + ring {}x{}, {:.0} MB, {load_ms:.0} ms", LAYERS.len(), ring.0, ring.1, tex.bytes as f64 / 1048576.0);
    let stars = load_stars(Path::new("/workspace/data/earth/bsc5.dat"));
    let tmp = std::env::temp_dir().join("ss1_earth");
    std::fs::create_dir_all(&tmp).unwrap();
    for s in &list {
        let mut s = s.clone();
        let sys_o = s.epoch.map(System::new);
        let sys = sys_o.as_ref().unwrap_or(&sys);
        let ep = sys.epoch;
        if let Some(e) = std::env::var("EXPOSURE").ok().and_then(|v| v.parse().ok()) {
            s.exposure = e;
        }
        let fr = build_frame(sys, &cal, &s, rw, rh, ss);
        let tr = Instant::now();
        let (mut hdr, t) = be.render_system(&tex, &fr.params, rw, rh).expect("render system");
        let sys_ms = tr.elapsed().as_secs_f64() * 1e3;
        let px_rad = 2.0 * fr.ty / rh as f64;
        let px_sr = px_rad * px_rad;
        // Earth handoff
        let mut earth_json = "null".to_string();
        let mut earth_ms = 0.0;
        if fr.earth_handoff {
            let er = run_earth(&earth_bin, sys, &s, w, h, ss, &tmp);
            earth_ms = er.wall_ms;
            earth_json = er.json;
            let earth = sys.get(BodyId::Earth).2;
            let e_earth = (AU_M / earth.length()).powi(2) as f32;
            // the planet renderer's own sun disc: masked (the system pass draws the Sun)
            let sun_rel = (sys.bodies[0].2 - s.cam.pos).normalize();
            let sun_r = (solar::SUN_RADIUS_M / (sys.bodies[0].2 - s.cam.pos).length()).asin() * 3.0;
            let (c, ty, aspect) = (&s.cam, fr.ty, fr.aspect);
            hdr.par_chunks_mut(rw as usize).enumerate().for_each(|(y, row)| {
                for (x, q) in row.iter_mut().enumerate() {
                    let e = er.hdr[y * rw as usize + x];
                    if q[3] < -0.5 {
                        q[3] = 0.0;
                        continue;
                    }
                    let nx = ((x as f64 + 0.5) / rw as f64) * 2.0 - 1.0;
                    let ny = ((y as f64 + 0.5) / rh as f64) * 2.0 - 1.0;
                    let d = (c.fwd + c.right * (nx * ty * aspect) - c.up * (ny * ty)).normalize();
                    let ek = if e[3] > 0.0 && d.dot(sun_rel) > sun_r.cos() { 0.0 } else { 1.0 };
                    for k in 0..3 {
                        q[k] = e[k] * e_earth * ek + e[3] * q[k];
                    }
                    q[3] *= e[3];
                }
            });
        }
        // sub-pixel bodies: their true flux
        let mut points = Vec::new();
        for &(i, r_px, skip, rel) in &fr.bodies {
            if !skip {
                continue;
            }
            let b = &sys.bodies[i].0;
            let d = rel.length();
            let flux = if b.id == BodyId::Sun {
                (AU_M / d).powi(2)
            } else {
                let to_sun = sys.bodies[0].2 - (s.cam.pos + rel);
                let e = (AU_M / to_sun.length()).powi(2);
                let alpha = to_sun.normalize().dot(-rel.normalize()).clamp(-1.0, 1.0).acos();
                b.geometric_albedo * e * (b.r_eq_m / d).powi(2) * phase_law(b, alpha)
            };
            let li = layer_of(b.id) as usize;
            let mut col = cal.mean[li];
            if let Some((tc, _)) = true_colour(b.id) {
                col = tc.map(s2l);
            }
            if b.id == BodyId::Sun {
                col = [1.0, 0.98, 0.95];
            }
            let y = lum(col).max(1e-6);
            let rgb = col.map(|c| (c / y) as f32);
            if let Some((fx, fy)) = project(&s.cam, fr.ty, fr.aspect, rw, rh, rel) {
                let l = (flux / px_sr * s.exposure) as f32;
                if splat(&mut hdr, rw, rh, fx, fy, l, rgb) {
                    points.push((b.id.name(), flux, r_px));
                }
            }
        }
        let mut nstars = 0;
        for st in &stars {
            if let Some((fx, fy)) = project(&s.cam, fr.ty, fr.aspect, rw, rh, st.dir)
                && fx >= 0.0
                && fy >= 0.0
                && fx < rw as f64
                && fy < rh as f64
            {
                splat(&mut hdr, rw, rh, fx, fy, (st.flux / px_sr * s.exposure) as f32, st.rgb);
                nstars += 1;
            }
        }
        // box downsample (linear), lens PSF, ACES, sRGB
        let tp = Instant::now();
        let (wu, hu) = (w as usize, h as usize);
        let mut lin = vec![[0.0f32; 3]; wu * hu];
        let inv = 1.0 / (ss * ss) as f32;
        lin.par_chunks_mut(wu).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                for sy in 0..ss as usize {
                    for sx in 0..ss as usize {
                        let q = hdr[(y * ss as usize + sy) * rw as usize + x * ss as usize + sx];
                        for c in 0..3 {
                            o[c] += q[c].max(0.0) * inv;
                        }
                    }
                }
            }
        });
        // lens PSF (energy-preserving): two Gaussian cores (sigma 1.5 / 6 output px, 2% / 0.3%) by
        // separable blurs on the image clamped at BRIGHT, plus, for whatever exceeds BRIGHT (the Sun,
        // as a disc or a flux point), narrower cores and a Moffat halo (beta 1.5, r0 2 px, 0.002%)
        // evaluated analytically: blurs of a 1e7-to-1 source would show their square support
        const BRIGHT: f32 = 16.0;
        let mut bright = Vec::new();
        for (i, c) in lin.iter_mut().enumerate() {
            let l = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
            if l > BRIGHT {
                let k = BRIGHT / l;
                bright.push(((i % wu) as f32, (i / wu) as f32, c.map(|v| v * (1.0 - k)), l - BRIGHT));
                *c = c.map(|v| v * k);
            }
        }
        let cores = [(1.5f32, 0.02f32), (6.0, 0.003)];
        let (halo_w, r0) = (2.0e-5f32, 2.0f32);
        let wsum: f32 = cores.iter().map(|p| p.1).sum::<f32>() + halo_w;
        let mut acc: Vec<[f32; 3]> = lin.iter().map(|c| c.map(|v| v * (1.0 - wsum + halo_w))).collect();
        for (sigma, wgt) in cores {
            let g = gauss(&lin, wu, hu, sigma as f64);
            acc.par_iter_mut().zip(g.par_iter()).for_each(|(a, g)| {
                for c in 0..3 {
                    a[c] += g[c] * wgt;
                }
            });
        }
        // the bright part gets a cleaner lens (a 1e9 source through the 0.3% sigma-6 wing would
        // bury Mercury and Venus): cores 2% / 0.05%, halo 0.002%
        let bcores = [(1.5f32, 0.02f32), (6.0, 0.0005)];
        let bsum: f32 = bcores.iter().map(|p| p.1).sum::<f32>() + halo_w;
        let psf = |r2: f32| -> f32 {
            let mut v = 0.0;
            for (sg, wg) in bcores {
                v += wg / (2.0 * std::f32::consts::PI * sg * sg) * (-r2 / (2.0 * sg * sg)).exp();
            }
            v + halo_w * 0.5 / (std::f32::consts::PI * r0 * r0) * (1.0 + r2 / (r0 * r0)).powf(-1.5)
        };
        // per bright pixel: a circular reach where its halo drops below 2e-5, minus the value there (no edge)
        let reach: Vec<f32> = bright.iter().map(|b| (r0 * (b.3 * halo_w * 0.5 / (std::f32::consts::PI * r0 * r0) / 2.0e-5).powf(1.0 / 3.0)).clamp(12.0, 1600.0)).collect();
        let nbright = bright.len();
        if nbright > 0 {
            acc.par_chunks_mut(wu).enumerate().for_each(|(y, row)| {
                for (bi, b) in bright.iter().enumerate() {
                    let r = reach[bi];
                    let dy = y as f32 - b.1;
                    if dy.abs() > r {
                        continue;
                    }
                    let x0 = (b.0 - r).max(0.0) as usize;
                    let x1 = ((b.0 + r) as usize).min(wu - 1);
                    for (x, o) in row.iter_mut().enumerate().take(x1 + 1).skip(x0) {
                        let dx = x as f32 - b.0;
                        let r2 = dx * dx + dy * dy;
                        if r2 > r * r {
                            continue;
                        }
                        let k = if r2 < 0.25 { 1.0 - bsum } else { 0.0 } + (psf(r2) - psf(r * r)).max(0.0);
                        for (oc, bc) in o.iter_mut().zip(b.2) {
                            *oc += bc * k;
                        }
                    }
                }
            });
        }
        let rgb8: Vec<u8> = acc.par_iter().flat_map_iter(|c| c.map(|v| (planet::bake::linear_to_srgb(aces(v)) * 255.0 + 0.5) as u8)).collect();
        let post_ms = tp.elapsed().as_secs_f64() * 1e3;
        write_png(&out.join(format!("{}.png", s.name)), w, h, &rgb8);
        // json: camera, bodies (screen position / size), orbits (for the HUD), timings
        let to_out = |p: (f64, f64)| (p.0 / ss as f64, p.1 / ss as f64);
        let bodies_json: Vec<String> = fr
            .bodies
            .iter()
            .map(|&(i, r_px, skip, rel)| {
                let b = &sys.bodies[i].0;
                let scr = project(&s.cam, fr.ty, fr.aspect, rw, rh, rel).map(to_out);
                let drawn = if fr.earth_handoff && b.id == BodyId::Earth { "planet_renderer" } else if skip { "flux_point" } else { "disc" };
                format!(
                    "{{\"name\":\"{}\",\"dist_m\":{:.6e},\"radius_px\":{:.4},\"screen\":{},\"drawn\":\"{drawn}\"}}",
                    b.id.name(),
                    rel.length(),
                    r_px / ss as f64,
                    scr.map_or("null".into(), |p| format!("[{:.2},{:.2}]", p.0, p.1))
                )
            })
            .collect();
        let orbits_json: Vec<String> = if s.hud && s.mode == Mode::System {
            BodyId::ALL
                .iter()
                .filter(|&&id| id != BodyId::Sun && id != BodyId::Moon)
                .map(|&id| {
                    let pts: Vec<String> = solar::orbit_polyline(id, ep, 720)
                        .iter()
                        .map(|p| project(&s.cam, fr.ty, fr.aspect, rw, rh, *p - s.cam.pos).map(to_out).map_or("null".into(), |q| format!("[{:.1},{:.1}]", q.0, q.1)))
                        .collect();
                    format!("{{\"name\":\"{}\",\"pts\":[{}]}}", id.name(), pts.join(","))
                })
                .collect()
        } else {
            Vec::new()
        };
        let errs = render_ash::validation::message_counts().0;
        let json = format!(
            "{{\"shot\":\"{}\",\"size\":[{w},{h}],\"ss\":{ss},\"epoch_utc_jd\":{:.9},\"mode\":\"{:?}\",\"note\":\"{}\",\"camera\":{{\"pos_system_m\":{},\"fwd\":{},\"up\":{},\"fov_y_deg\":{:.6},\"dist_sun_au\":{:.6},\"dist_earth_km\":{:.1}}},\"exposure\":{:.6e},\"earth_handoff\":{},\"bodies\":[{}],\"flux_points\":{},\"stars_drawn\":{nstars},\"gpu_ms\":{:.1},\"system_wall_ms\":{sys_ms:.0},\"earth_wall_ms\":{earth_ms:.0},\"post_ms\":{post_ms:.0},\"texture_load_ms\":{load_ms:.0},\"texture_mb\":{:.1},\"validation_errors\":{errs},\"timing_device\":\"lavapipe (CPU Vulkan), not a real-GPU timing\",\"earth\":{earth_json},\"orbits\":[{}]}}",
            s.name,
            ep.jd_utc,
            s.mode,
            s.note,
            v3(s.cam.pos),
            v3(s.cam.fwd),
            v3(s.cam.up),
            s.cam.fov_y,
            (s.cam.pos - sys.bodies[0].2).length() / AU_M,
            (s.cam.pos - sys.get(BodyId::Earth).2).length() / 1e3,
            s.exposure,
            fr.earth_handoff,
            bodies_json.join(","),
            points.len(),
            t.gpu_ms,
            tex.bytes as f64 / 1048576.0,
            orbits_json.join(",")
        );
        std::fs::write(out.join(format!("{}.json", s.name)), &json).unwrap();
        println!("{{\"shot\":\"{}\",\"gpu_ms\":{:.1},\"system_wall_ms\":{sys_ms:.0},\"earth_wall_ms\":{earth_ms:.0},\"handoff\":{},\"flux_points\":{},\"stars\":{nstars},\"validation_errors\":{errs}}}", s.name, t.gpu_ms, fr.earth_handoff, points.len());
    }
    be.system_free(tex);
}
