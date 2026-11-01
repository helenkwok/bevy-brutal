//! Orbit -> mountains descent (Earth round 3).
//!
//! cargo run --release -p render-ash --example descent -- [--heights 400000,150000,...] [--only 0,3]
//!     [--pack /workspace/data/earth/earth.pln] [--out out/quality/steps] [--work out/quality/descent]
//!     [--prefix E3_descent] [--composite-only] [--planet-only] [--reuse-mtn]
//!
//! One continuous camera path ([`planet::descent::Descent::alps`]) from 400 km over the Alps down
//! to the mountains scene's default vista (40 m over the valley floor). Every frame renders the
//! same f64 camera twice, as subprocesses of the `earth` and `quality` examples:
//! - the planet (cube-sphere quadtree, NASA textures, Hillaire atmosphere, volumetric clouds) from
//!   the planet-frame pose (`--cam-ecef`), with the terrain under the mountains layer pressed down
//!   (`FLATTEN`, by the ground weight: the real relief morphs flat as the layer fades in), a
//!   clear-sky window over it (`HOLE`) and its sky mask (`SKY_MASK`);
//! - below 140 km, the mountains scene in layer mode (`LAYER`: bent onto the sphere, radial
//!   coverage alpha, all-round rings) from the same pose in world coordinates, with the handoff's
//!   aerial-perspective range, shadow splits and fixed exposure.
//!
//! It composites them in linear light by [`planet::descent::composite`]. The last frame is the
//! mountains render alone. Writes `<out>/<prefix>_NN.png` + `.json` (pose, weights, layer timings,
//! the seam metric) and keeps the layers in `<work>`.
//!
//! Earth round 4 (the shared Alps): the planet layer no longer flattens the real relief under
//! the mountains layer; it grafts the mountains scene's own field in (`GRAFT`, strength
//! [`planet::graft::strength`] by height: complete by 150 km, before the layer starts at 140 km),
//! from a bake cached at `--graft` (default `/workspace/data/earth/graft_alps.grf`, baked on first
//! use: ~1-2 min). `--no-graft` renders round 3's FLATTEN handoff.
#[path = "descent_parts/graft.rs"]
mod graft;
#[path = "descent_parts/ground.rs"]
mod ground;
use planet::descent::{self, Descent, Handoff, handoff};
use std::path::{Path, PathBuf};
use std::process::Command;
use voxel::fields::Alpine;

/// The quality example's vertical FOV (rad) and output size.
const FOV: f64 = 0.82;
const W: u32 = 1920;
const H: u32 = 1080;
/// Mountains-layer radial fade (m from the vista) and the planet's flattened disc (km).
const FADE: (f64, f64) = (60_000.0, 100_000.0);
const FLAT_KM: (f64, f64) = (100.0, 140.0);
/// Flattened planet terrain height (m): below the valley floor, above any sea level the
/// ocean mask could disagree with.
const FLAT_H: f64 = 300.0;
/// Clear-sky window radius (deg): full clearing over the layer's 100 km, clouds back by 2.2x.
const HOLE_DEG: f64 = 0.9;
/// Round 7: the window only has to be clear where the mountains layer draws (frames from ~60 km
/// down); above, it shrinks to `HOLE_FAR_DEG` (default 0.55 deg; 0.9 = round 6) between 140 and
/// 70 km so the 400 / 150 km frames keep their weather around the vista.
fn hole_deg(h_m: f64) -> f64 {
    let far = std::env::var("HOLE_FAR_DEG").ok().and_then(|v| v.parse().ok()).unwrap_or(0.55);
    let t = ((h_m - 70_000.0) / 70_000.0).clamp(0.0, 1.0);
    HOLE_DEG + (far - HOLE_DEG) * t * t * (3.0 - 2.0 * t)
}
/// Default sun: the mountains scene's (azimuth yaw + 2.35, elevation 0.5 rad).
const SUN_AZ: f64 = 2.75 + 2.35;
const SUN_EL: f64 = 0.5;

/// The mountains layer's exposure (quality example `EXPOSURE`, default 1.1) for a handoff:
/// the planet's exposure in the quality example's units. Exactly "1.100000" below 20 km.
fn mountains_exposure(w: &Handoff) -> f64 {
    1.1 * w.exposure / descent::EXPOSURE_GROUND
}

fn arg(k: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned())
}

fn read_png(p: &Path) -> (u32, u32, usize, Vec<u8>) {
    let dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))));
    let mut r = dec.read_info().unwrap();
    let mut buf = vec![0; r.output_buffer_size().unwrap()];
    let info = r.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    let ch = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        c => panic!("{}: unsupported {c:?}", p.display()),
    };
    (info.width, info.height, ch, buf)
}

fn write_png(p: &Path, w: u32, h: u32, rgb: &[u8]) {
    let mut enc = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(p).unwrap()), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(rgb).unwrap();
}

/// `"key":number` out of a child's json.
fn json_num(s: &str, k: &str) -> Option<f64> {
    let i = s.find(&format!("\"{k}\":"))? + k.len() + 3;
    s[i..].split([',', '}']).next()?.trim().parse().ok()
}

fn run(mut c: Command, log: &Path) -> f64 {
    let t = std::time::Instant::now();
    let out = c.output().expect("spawn");
    std::fs::write(log, [format!("{c:?}\n").as_bytes(), out.stdout.as_slice(), out.stderr.as_slice()].concat()).unwrap();
    assert!(out.status.success(), "{:?} failed, see {}", c, log.display());
    t.elapsed().as_secs_f64() * 1e3
}

fn f3(v: glam::DVec3) -> String {
    format!("{:.6},{:.6},{:.6}", v.x, v.y, v.z)
}

struct Seam {
    /// Pixels with a partial layer weight (both renders visible).
    blend_px: usize,
    /// Mean |planet - mountains| (sRGB 0..255) over terrain pixels both layers cover
    /// (layer alpha > 0.5, planet ground), weighted equally: the colour / exposure mismatch the
    /// blend has to hide. NaN when there are none.
    terrain_diff: [f64; 3],
    /// Mean planet / mountains luminance (linear) over those pixels.
    lum: (f64, f64),
    /// `terrain_diff` after the local colour match (what the blend actually shows).
    matched_diff: [f64; 3],
}

/// Separable box blur (radius `r`, two passes ~ a Gaussian of sigma ~0.8 r) of a W x H x 4 plane.
fn blur4(v: &[[f32; 4]], r: usize) -> Vec<[f32; 4]> {
    let (w, h) = (W as usize, H as usize);
    let pass = |src: &[[f32; 4]], horiz: bool| -> Vec<[f32; 4]> {
        let (n, m) = if horiz { (h, w) } else { (w, h) };
        let at = |line: usize, k: usize| if horiz { line * w + k } else { k * w + line };
        let mut dst = vec![[0.0f32; 4]; w * h];
        for line in 0..n {
            let mut acc = [0.0f32; 4];
            for k in 0..=r.min(m - 1) {
                (0..4).for_each(|c| acc[c] += src[at(line, k)][c]);
            }
            for k in 0..m {
                let lo = k.saturating_sub(r);
                let cnt = (k + r).min(m - 1) - lo + 1;
                dst[at(line, k)] = acc.map(|a| a / cnt as f32);
                if k + r + 1 < m {
                    (0..4).for_each(|c| acc[c] += src[at(line, k + r + 1)][c]);
                }
                if k >= r {
                    (0..4).for_each(|c| acc[c] -= src[at(line, k - r)][c]);
                }
            }
        }
        dst
    };
    let mut x = v.to_vec();
    for _ in 0..2 {
        x = pass(&x, true);
        x = pass(&x, false);
    }
    x
}

/// Local colour match (Earth round 4): with the shared Alps both layers show the same terrain,
/// so the mountains layer takes the planet render's local mean colour (its physical aerial
/// perspective and exposure) and keeps its own detail: per pixel, the ratio of the two layers'
/// blurred linear colours over the terrain both cover, applied with strength `k` (the
/// descent's [`descent::colour_match`]: 0 at the vista).
fn colour_match(p: &[u8], pch: usize, sky: &[u8], m: &[u8], k: f64) -> Vec<[f32; 3]> {
    let n = (W * H) as usize;
    let lin = |b: u8| descent::srgb_to_linear(b as f32 / 255.0);
    let wgt = |i: usize| (m[i * 4 + 3] as f32 / 255.0) * (1.0 - sky[i] as f32 / 255.0);
    let pp: Vec<[f32; 4]> = (0..n).map(|i| {
        let q = wgt(i);
        [lin(p[i * pch]) * q, lin(p[i * pch + 1]) * q, lin(p[i * pch + 2]) * q, q]
    }).collect();
    let mm: Vec<[f32; 4]> = (0..n).map(|i| {
        let q = wgt(i);
        [lin(m[i * 4]) * q, lin(m[i * 4 + 1]) * q, lin(m[i * 4 + 2]) * q, q]
    }).collect();
    let r = (W as usize / 80).max(1);
    let (bp, bm) = (blur4(&pp, r), blur4(&mm, r));
    (0..n).map(|i| {
        let (a, b) = (bp[i], bm[i]);
        if a[3] < 1e-3 || k <= 0.0 {
            return [1.0; 3];
        }
        [0, 1, 2].map(|c| {
            let ratio = ((a[c] + 1e-4) / (b[c] + 1e-4)).clamp(0.33, 3.0);
            1.0 + (ratio - 1.0) * k as f32
        })
    }).collect()
}

fn composite(p: &[u8], pch: usize, sky: &[u8], m: Option<&[u8]>, w: &Handoff, gain: Option<&[[f32; 3]]>, out: &mut [u8]) -> Seam {
    let mut seam = Seam { blend_px: 0, terrain_diff: [0.0; 3], lum: (0.0, 0.0), matched_diff: [0.0; 3] };
    let mut n = 0usize;
    let lum = |c: [f32; 3]| 0.2126 * descent::srgb_to_linear(c[0]) + 0.7152 * descent::srgb_to_linear(c[1]) + 0.0722 * descent::srgb_to_linear(c[2]);
    for i in 0..(W * H) as usize {
        let pc = [0, 1, 2].map(|c| p[i * pch + c] as f32 / 255.0);
        let ps = sky[i] as f32 / 255.0;
        let c = match m {
            None => pc,
            Some(m) => {
                let mc0 = [0, 1, 2].map(|c| m[i * 4 + c] as f32 / 255.0);
                let mc = match gain {
                    Some(g) => [0, 1, 2].map(|c| descent::linear_to_srgb((descent::srgb_to_linear(mc0[c]) * g[i][c]).min(1.0))),
                    None => mc0,
                };
                let a = m[i * 4 + 3] as f32 / 255.0;
                let k = descent::layer_weight(a, ps, w);
                if k > 0.0 && k < 1.0 {
                    seam.blend_px += 1;
                }
                if a > 0.5 && ps < 0.5 {
                    n += 1;
                    for c in 0..3 {
                        seam.terrain_diff[c] += (pc[c] - mc0[c]).abs() as f64 * 255.0;
                        seam.matched_diff[c] += (pc[c] - mc[c]).abs() as f64 * 255.0;
                    }
                    seam.lum.0 += lum(pc) as f64;
                    seam.lum.1 += lum(mc0) as f64;
                }
                descent::composite(pc, mc, a, ps, w)
            }
        };
        for k in 0..3 {
            out[i * 3 + k] = (c[k].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
    let nn = n.max(1) as f64;
    seam.terrain_diff = seam.terrain_diff.map(|v| if n == 0 { f64::NAN } else { v / nn });
    seam.matched_diff = seam.matched_diff.map(|v| if n == 0 { f64::NAN } else { v / nn });
    seam.lum = if n == 0 { (f64::NAN, f64::NAN) } else { (seam.lum.0 / nn, seam.lum.1 / nn) };
    seam
}

fn main() {
    let field = Alpine::new(7);
    // the vista's ground exactly as the quality example computes it
    let ground = field.height(7700.0, -4200.0, 0.5);
    let ds = Descent::alps(ground);
    let heights: Vec<f64> = arg("--heights")
        .map(|v| v.split(',').map(|t| t.trim().parse().expect("--heights")).collect())
        .unwrap_or_else(|| vec![400_000.0, 150_000.0, 60_000.0, 30_000.0, 15_000.0, 7_000.0, 3_000.0, 1_200.0, 400.0, 40.0]);
    let only: Option<Vec<usize>> = arg("--only").map(|v| v.split(',').map(|t| t.trim().parse().unwrap()).collect());
    let out = PathBuf::from(arg("--out").unwrap_or("out/quality/steps".into()));
    let work = PathBuf::from(arg("--work").unwrap_or("out/quality/descent".into()));
    let prefix = arg("--prefix").unwrap_or("E4_descent".into());
    let pack = arg("--pack").unwrap_or("/workspace/data/earth/earth.pln".into());
    let composite_only = std::env::args().any(|a| a == "--composite-only");
    // --planet-only: the planet layer alone (debugging the planet side of the handoff)
    let planet_only = std::env::args().any(|a| a == "--planet-only");
    // --reuse-mtn: keep existing mountains layers (re-render only the planet side)
    let reuse_mtn = std::env::args().any(|a| a == "--reuse-mtn");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::create_dir_all(&work).unwrap();
    let no_graft = std::env::args().any(|a| a == "--no-graft");
    // Earth round 5: the shared disc on real elevation / imagery (`--no-ground`: round 4's
    // procedural disc and its bake)
    let no_ground = no_graft || std::env::args().any(|a| a == "--no-ground");
    let ground_path = arg("--ground").unwrap_or(ground::GROUND_FILE.into());
    let map = planet::graft::GraftMap::from_frames(&ds.frames);
    let gfield = if no_ground { field.clone() } else { Alpine::grounded(7, std::sync::Arc::new(ground::load_or_bake(&ground_path, &map))) };
    let graft_default = if no_ground { "/workspace/data/earth/graft_alps.grf" } else { ground::GRAFT_FILE };
    let graft_path = PathBuf::from(arg("--graft").unwrap_or(graft_default.into()));
    if !no_graft && !composite_only && !graft_path.exists() {
        let t = std::time::Instant::now();
        let img = (!no_ground).then(ground::Imagery::load);
        let f = img.as_ref().map(|im| move |la: f64, lo: f64, n: glam::DVec3, fine: bool, st: f64| im.albedo(la, lo, n, fine, st));
        let g = graft::bake(&gfield, &map, 46.10, 7.60, descent::dir_of(SUN_AZ, SUN_EL), f.as_ref().map(|f| f as graft::ImageryFn));
        if let Some(d) = graft_path.parent() {
            std::fs::create_dir_all(d).unwrap();
        }
        std::fs::write(&graft_path, g.to_bytes()).unwrap();
        eprintln!("graft baked: {} ({:.1} s)", graft_path.display(), t.elapsed().as_secs_f64());
    }
    let bin = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
    let (sun_lat, sun_lon) = ds.subsolar(SUN_AZ, SUN_EL);
    let (cx, cz) = ds.frames.world_xz;
    let a = (46.10, 7.60);
    let cirrus_y = ground + 40.0; // the default view's cirrus deck: camera height + 6 km
    eprintln!("vista ground {ground:.2} m, subsolar {sun_lat:.3}, {sun_lon:.3}");
    for (i, &h) in heights.iter().enumerate() {
        if only.as_ref().is_some_and(|o| !o.contains(&i)) {
            continue;
        }
        let last = h <= ds.h_end_m;
        let s = if last { 1.0 } else { ds.s_at_height(h) };
        let pose = ds.pose(s);
        let w = handoff(pose.h_m);
        let name = format!("{prefix}_{i:02}");
        let cam = ds.frames.to_planet(pose.cam);
        let pl = work.join(format!("{name}_planet.png"));
        let mut t_planet = f64::NAN;
        if !composite_only {
            let mut c = Command::new(bin.join("earth"));
            c.args(["--pack", &pack, "--out", work.to_str().unwrap(), "--name", &format!("{name}_planet"), "--size", &format!("{W}x{H}"), "--ss", "2"]);
            c.args(["--fov", &format!("{:.6}", FOV.to_degrees()), "--cam-ecef", &format!("{},{},{}", f3(cam), f3(pose.fwd), f3(pose.up_hint))]);
            c.env("EXPOSURE", format!("{:.6}", w.exposure)).env("SUN_LAT", format!("{sun_lat:.6}")).env("SUN_LON", format!("{sun_lon:.6}"));
            c.env("MIE", "8").env("SKY_MASK", "1");
            c.env("HOLE", format!("{},{},{}", a.0, a.1, hole_deg(pose.h_m)));
            if no_graft {
                c.env("FLATTEN", format!("{},{},{},{},{FLAT_H},{:.6}", a.0, a.1, FLAT_KM.0, FLAT_KM.1, w.ground));
            } else {
                c.env("GRAFT", format!("{},{:.6}", graft_path.display(), planet::graft::strength(pose.h_m)));
                if !no_ground {
                    c.env("GROUND", &ground_path);
                }
            }
            for k in ["GRAFT_ALB", "GRAFT_AO", "GRAFT_TINT", "GRAFT_TINT_KM", "MESO_TOP", "MESO_AMP", "MESO_BIAS", "MESO_FINE", "MESO_GAMMA", "OCEAN_HAZE", "HAZE_ANGSTROM", "HAZE_SSA", "HAZE_MASK_KM", "CLOUD_FEATHER", "CLOUD_FEATHER_MIX", "HOLE_NOISE", "HAZE_DUST", "EPIC_GRADE"] {
                if let Ok(v) = std::env::var(k) {
                    c.env(k, v);
                }
            }
            t_planet = run(c, &work.join(format!("{name}_planet.log")));
        }
        let use_m = (w.ground > 0.0 || w.sky > 0.0) && !planet_only;
        let ml = work.join(format!("{name}_mtn.png"));
        let mut t_mtn = f64::NAN;
        if use_m && !composite_only && !(reuse_mtn && ml.exists()) {
            let mut c = Command::new(bin.join("quality"));
            c.args(["--scene", "mountains"]);
            if last {
                // the vista exactly as the default view specifies it
                c.env("CAM_H", "40");
            } else {
                c.env("CAM_X", format!("{:.6}", pose.world.x)).env("CAM_Z", format!("{:.6}", pose.world.z)).env("CAM_Y", format!("{:.6}", pose.world.y));
            }
            c.env("CAM_YAW", format!("{:.9}", pose.yaw)).env("CAM_PITCH", format!("{:.9}", pose.pitch));
            // the aerial-only parts of the layer fade out toward the vista; at the vista itself
            // (fade 0) the mountains scene renders in its default view, byte for byte
            let fade = descent::vista_fade(pose.h_m);
            if fade > 0.0 {
                c.env("LAYER", format!("{cx},{cz},{},{},{cirrus_y:.6},{fade:.6}", FADE.0, FADE.1));
                c.env("RING_OUTER", format!("{:.0}", pose.ground_dist_m + 110_000.0));
            }
            // Earth round 6: the default view is grounded too (the vista frame = the default
            // mountains view, which now loads the round-6 ground by default)
            if no_ground {
                c.env("GROUND", "off");
            } else {
                c.env("GROUND", &ground_path).env("GROUND_ALB", graft_path.to_str().unwrap());
            }
            c.env("SUN_AZ", format!("{SUN_AZ:.9}")).env("SUN_EL", format!("{SUN_EL:.9}")).env("AP_KM", format!("{:.4}", w.ap_km));
            c.env("SPLITS", w.splits.map(|v| format!("{v:.1}")).join(","));
            // one exposure per frame for both layers: the quality example's 1.1 is the planet's
            // EXPOSURE_GROUND, so the mountains follow the planet's ramp above 20 km
            c.env("EXPOSURE", format!("{:.6}", mountains_exposure(&w)));
            c.env("AUTO_EXPOSURE", "0").env("SHADOW_CACHE", "0").env("QUALITY_FRAMES", "1").env("QUALITY_OUT", ml.to_str().unwrap());
            t_mtn = run(c, &work.join(format!("{name}_mtn.log")));
        }
        let (pw, ph, pch, p) = read_png(&pl);
        let (sw, sh, _, sky) = read_png(&work.join(format!("{name}_planet_sky.png")));
        assert!((pw, ph, sw, sh) == (W, H, W, H));
        let m = use_m.then(|| {
            let (mw, mh, mch, m) = read_png(&ml);
            assert!((mw, mh) == (W, H) && (mch == 4 || mch == 3));
            // the default view (the vista) has no alpha: all coverage
            if mch == 3 { m.chunks(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect() } else { m }
        });
        let mut img = vec![0u8; (W * H * 3) as usize];
        let mk = if no_graft || std::env::var("COLOUR_MATCH").as_deref() == Ok("0") { 0.0 } else { descent::colour_match(pose.h_m) };
        let gain = match &m {
            Some(m) if mk > 0.0 => Some(colour_match(&p, pch, &sky, m, mk)),
            _ => None,
        };
        let seam = composite(&p, pch, &sky, m.as_deref(), &w, gain.as_deref(), &mut img);
        let png = out.join(format!("{name}.png"));
        write_png(&png, W, H, &img);
        let pj = std::fs::read_to_string(work.join(format!("{name}_planet.json"))).unwrap_or_default();
        let mj = if use_m { std::fs::read_to_string(ml.with_extension("json")).unwrap_or_default() } else { String::new() };
        let num = |v: f64| if v.is_finite() { format!("{v:.3}") } else { "null".into() };
        let ll = planet::cube::dir_latlon(cam);
        let json = format!(
            "{{\"frame\":\"{name}\",\"s\":{:.6},\"mountains_exposure\":{:.6},\"height_above_vista_m\":{:.2},\"altitude_m\":{:.2},\"ground_dist_m\":{:.1},\"lat\":{:.6},\"lon\":{:.6},\"world\":[{:.3},{:.3},{:.3}],\"yaw\":{:.6},\"pitch\":{:.6},\
\"handoff\":{{\"ground\":{:.4},\"sky\":{:.4},\"up\":{:.4},\"ap_km\":{:.2},\"planet_exposure\":{:.4},\"splits\":{:?}}},\
\"layers\":{{\"planet\":\"{}\",\"mountains\":{}}},\"seam\":{{\"blend_pixels\":{},\"terrain_abs_diff_srgb\":[{},{},{}],\"terrain_lum_planet_mtn\":[{},{}],\"matched_abs_diff_srgb\":[{},{},{}],\"colour_match\":{:.4}}},\
\"planet_gpu_ms\":{},\"mountains_gpu_frame_ms\":{},\"mountains_build_ms\":{},\"planet_wall_ms\":{},\"mountains_wall_ms\":{},\"mountains_validation_errors\":{},\
\"device\":\"lavapipe (llvmpipe, CPU): timings are not real-GPU numbers\",\"sun\":{{\"az\":{SUN_AZ:.4},\"el\":{SUN_EL},\"subsolar\":[{sun_lat:.4},{sun_lon:.4}]}},\
\"planet_env\":{{\"MIE\":8,\"HOLE\":[{},{},{:.4}],\"FLATTEN_km\":[{},{}],\"FLATTEN_h_m\":{FLAT_H},\"FLATTEN_strength\":{:.4},\"graft\":{},\"graft_strength\":{:.4}}},\"png\":\"{}\"}}",
            pose.s.max(0.0) + 0.0, mountains_exposure(&w), pose.h_m, cam.length() - planet::EARTH_RADIUS_M, pose.ground_dist_m, ll.0, ll.1, pose.world.x, pose.world.y, pose.world.z, pose.yaw, pose.pitch,
            w.ground, w.sky, w.up, w.ap_km, w.exposure, w.splits,
            pl.display(), if use_m { format!("\"{}\"", ml.display()) } else { "null".into() },
            seam.blend_px, num(seam.terrain_diff[0]), num(seam.terrain_diff[1]), num(seam.terrain_diff[2]), num(seam.lum.0), num(seam.lum.1), num(seam.matched_diff[0]), num(seam.matched_diff[1]), num(seam.matched_diff[2]), mk,
            json_num(&pj, "gpu_ms").map_or("null".into(), num), json_num(&mj, "gpu_frame_ms").map_or("null".into(), num), json_num(&mj, "build_ms").map_or("null".into(), num),
            num(t_planet), num(t_mtn), json_num(&mj, "validation_errors").map_or("null".into(), num),
            a.0, a.1, hole_deg(pose.h_m), FLAT_KM.0, FLAT_KM.1, if no_graft { w.ground } else { 0.0 }, if no_graft { "null".into() } else { format!("\"{}\"", graft_path.display()) }, if no_graft { 0.0 } else { planet::graft::strength(pose.h_m) }, png.display()
        );
        std::fs::write(png.with_extension("json"), &json).unwrap();
        println!("{json}");
    }
}
