//! Earth from orbit: cube-sphere quadtree + NASA textures + atmosphere from space.
//!
//! cargo run --release -p render-ash --example earth -- [--shot day|iss|terminator|glint|all]
//!     [--pack /workspace/data/earth/earth.pln] [--size 1920x1080] [--ss 2] [--out out/quality/earth]
//!
//! Writes `<out>/<shot>.png` (+ `.json`): tonemapped sRGB at the output size, rendered at
//! `ss`x supersampling and box-filtered in linear light. Stars (Yale BSC5) are splatted on the
//! CPU where the sky pass reports transmittance toward space, at their physical brightness
//! relative to the sun (so they only show at night-side exposures).
//!
//! Earth round 3 (the orbit -> mountains descent, `descent` example) adds:
//! `--cam-ecef x,y,z,fx,fy,fz,ux,uy,uz` (planet-frame metres, forward, up hint: an exact f64
//! camera instead of a named shot) with `--fov <deg>` and `--name <name>`; `HOLE=lat,lon,r_deg`
//! (a clear-sky window in the cloud coverage), `FLATTEN=lat,lon,r0_km,r1_km,h_m[,k]` (terrain pressed
//! down to <= h_m inside r0, blending back by r1, strength k 0..1: room for the mountains layer; its glaciers, flat,
//! would read as low cloud, so snow albedo fades out over the same disc) and `SKY_MASK=1`
//! (`<name>_sky.png`: the fraction of each pixel's samples that see sky, for compositing).
//!
//! Earth round 4: `GRAFT=<file.grf>,strength` (the shared Alps, [`planet::graft`]): inside the
//! graft disc the terrain is the mountains scene's own field (patch heights from `Alpine`, the
//! baked albedo / normal / shadow / AO maps from the file), the real Earth beyond; strength 0..1
//! by the descent's height. `GRAFT_ALB` / `GRAFT_AO` tune the albedo gain / AO strength.
//!
//! DEM patch (`planet::dem`): `DEM=<geo-fetch terrain-grid dir>[,strength]` places a real DEM
//! (e.g. `/workspace/data/geo/terrain/grids/adelaide_z12`) on the planet: patch heights inside its
//! disc are the DEM's, shading (normals, cast shadows, coastline) goes through the graft bindings,
//! baked per shot for its sun. `DEM_R=r_full_km,r_out_km`, `DEM_TEX=<size>`, `DEM_TINT`, `DEM_ALB`.
//! `OSM=<geo-fetch osm-extract dir>[,cell_m]` (with DEM) drapes OSM roads / buildings / water /
//! landuse colour over the DEM palette (`planet::drape`, 4 m cells by default).
//! `DEM_CLOSE_KM=<half>` streams a third concentric graft tile (default 3 km with OSM, 0 = off)
//! so Fragments-style building objects LOD (`BuildingLod`, ≤ 6.8 m texels) can show; the shader
//! reads it as array layer 2 via `meso2.y`. Shots `DEM_adelaide_*` (use `HOLE=-34.9,138.6,1.5`
//! for a clear sky over the patch); `DEM_adelaide_objects` is a low nadir over Norwood that
//! fills the close tile with individual roofs (Toorak Gardens / Rose Park).
//!
//! Solar system round 1: `STARS=0` (no star splat) and `HDR_OUT=1` (`<name>.hdr.bin`, the exposed
//! linear HDR at the render size) for the `solar` example's Earth handoff; unset = unchanged.
//!
//! Earth round 5: the graft textures upload as BC7 (sRGB albedo + water alpha, linear aux:
//! normal x / z, sun visibility) from `<file.grf>.alb.ktx2` / `.aux.ktx2` (2-layer 2D arrays,
//! mips to 4x4, zstd; encoded and cached on first use, re-encoded when older than the .grf).
//! `GRAFT_TEX=raw` keeps the round-4 RGBA8 upload straight from the .grf.
#[path = "descent_parts/graft.rs"]
mod graft;
use glam::{DVec3, DVec4, Mat4};
use planet::cube::latlon_dir;
use planet::ktx2::Ktx2;
use planet::pack::PlanetPack;
use planet::patch::{self, PVertex, VERTS};
use planet::quadtree::{self, Shape, View};
use rayon::prelude::*;
use render_ash::AshBackend;
use render_ash::planet::{PatchGpu, PlanetFrame, PlanetParamsGpu};
use std::path::{Path, PathBuf};
use std::time::Instant;

const R_KM: f64 = planet::EARTH_RADIUS_M / 1000.0;

/// A shot: camera (lat, lon, altitude km, heading deg from north, pitch deg from the local
/// horizontal, fov_y deg), subsolar point (lat, lon), exposure, night-light gain.
#[derive(Clone, Copy, Debug)]
struct Shot {
    name: &'static str,
    lat: f64,
    lon: f64,
    alt_km: f64,
    heading: f64,
    pitch: f64,
    fov_y: f64,
    sun_lat: f64,
    sun_lon: f64,
    exposure: f64,
    night_gain: f64,
    /// Greenwich sidereal angle (deg) for the star field.
    gmst: f64,
}

fn shots() -> Vec<Shot> {
    vec![
        // ~35,000 km, full disk: Africa, Europe, the Atlantic; mid-July sun ~13 deg off the view axis
        Shot {
            name: "E2_orbit_day",
            lat: 8.0,
            lon: 18.0,
            alt_km: 35_000.0,
            heading: 0.0,
            pitch: -90.0,
            fov_y: 20.5,
            sun_lat: 21.0,
            sun_lon: 9.0,
            exposure: 2.4,
            night_gain: 1.5e-6,
            gmst: 40.0,
        },
        // the exact DSCOVR EPIC geometry of the reference frame epic_1b_20260715111224
        // (2026-07-15 11:07:35 UTC, EPIC API J2000 vectors): sub-spacecraft point 13.39 N 22.51 E,
        // 1,481,554 km from the centre, subsolar point 21.54 N 14.22 E; fov matches the 853 px
        // disc of the 1080 px reference JPG. North up like EPIC's natural-colour product
        Shot {
            name: "E2_vs_epic",
            lat: 13.389,
            lon: 22.513,
            alt_km: 1_475_183.0,
            heading: 0.0,
            pitch: -90.0,
            fov_y: 0.6239,
            sun_lat: 21.537,
            sun_lon: 14.219,
            exposure: 2.4,
            night_gain: 1.5e-6,
            gmst: 100.2,
        },
        // 120 km over the western Alps looking east along the main chain at mid-morning: the
        // footprint (~0.2-0.6 km) is far below the 2.4 km texels, so the procedural sub-texel
        // detail carries the ground (the first leg of the descent toward the mountains scene)
        Shot {
            name: "E2_low_alps",
            lat: 45.6,
            lon: 6.0,
            alt_km: 120.0,
            heading: 70.0,
            pitch: -38.0,
            fov_y: 50.0,
            sun_lat: 21.5,
            sun_lon: -20.0,
            exposure: 2.6,
            night_gain: 1.5e-6,
            gmst: 0.0,
        },
        // ISS, 410 km over the eastern Atlantic looking north-east to Iberia / the Mediterranean
        Shot {
            name: "E2_orbit_iss",
            lat: 29.0,
            lon: -21.0,
            alt_km: 410.0,
            heading: 48.0,
            pitch: -27.0,
            fov_y: 52.0,
            sun_lat: 21.0,
            sun_lon: -45.0,
            exposure: 2.6,
            night_gain: 1.5e-6,
            gmst: 0.0,
        },
        // night over Italy, looking WNW along the dark limb: the sun is 27 deg below the camera's
        // horizontal (full night, cities below) and 55 deg left of the view, ~6.6 deg below the
        // horizon at the limb in its direction, so the twilight arc glows at the left end and
        // fades along the limb; the thin green airglow line runs above the dark limb; stars
        Shot {
            name: "E2_terminator",
            lat: 43.0,
            lon: 12.0,
            alt_km: 420.0,
            heading: 285.0,
            pitch: -10.0,
            fov_y: 60.0,
            sun_lat: -46.76,
            sun_lon: -73.10,
            exposure: 2.0e5,
            night_gain: 1.5e-6,
            gmst: 200.0,
        },
        // the same camera, telephoto (6 deg) on the limb straight toward the sun, which is ~3.6
        // deg below the horizon at the limb's tangent point: the layered sunset line (red-orange
        // troposphere with cloud silhouettes, yellow-white, the blue ozone/Rayleigh layer, dark)
        Shot {
            name: "E2_sunset",
            lat: 43.0,
            lon: 12.0,
            alt_km: 420.0,
            heading: 285.0,
            pitch: -19.4,
            fov_y: 6.0,
            sun_lat: -6.00,
            sun_lon: -105.47,
            exposure: 200.0,
            night_gain: 1.5e-6,
            gmst: 200.0,
        },
        // ISS over the Sahel at 16:30 local: the mesoscale convective systems of the dated VIIRS
        // map near 13 N 1 W (tops > 12 km), ~750 km ahead (24.5 deg below the horizontal), sun
        // 25 deg up in the WNW: side-lit towers casting shadows east
        Shot {
            name: "E2_iss_convection",
            lat: 7.0,
            lon: -5.5,
            alt_km: 410.0,
            heading: 32.0,
            pitch: -22.0,
            fov_y: 52.0,
            sun_lat: 21.5,
            sun_lon: -68.5,
            exposure: 2.4,
            night_gain: 1.5e-6,
            gmst: 0.0,
        },
        // ocean sun glint: morning sun 35 deg up in the east, specular point on Maui (20.9 N,
        // 156.7 W) 535 km ahead, 39.8 deg below the horizontal -> the islands sit in the glint
        Shot {
            name: "E2_glint",
            lat: 20.8,
            lon: -161.8,
            alt_km: 420.0,
            heading: 88.0,
            pitch: -32.0,
            fov_y: 50.0,
            sun_lat: 11.94,
            sun_lon: -99.83,
            exposure: 2.6,
            night_gain: 1.5e-6,
            gmst: 0.0,
        },
        // DEM patch on the planet (`planet::dem`, DEM=<geo-fetch grid dir>): Adelaide, early
        // November, 15:30 local solar time (sun 39 deg up in the north-west, lighting the Mount
        // Lofty scarp). Run with HOLE=-34.9,138.6,1.5 to clear the dated cloud map over the patch.
        Shot {
            name: "DEM_adelaide_60km",
            lat: -35.30,
            lon: 138.42,
            alt_km: 60.0,
            heading: 25.0,
            pitch: -48.0,
            fov_y: 40.0,
            sun_lat: -15.0,
            sun_lon: 86.0,
            exposure: 2.6,
            night_gain: 1.5e-6,
            gmst: 0.0,
        },
        Shot {
            name: "DEM_adelaide_nadir",
            lat: -34.9,
            lon: 138.6,
            alt_km: 30.0,
            heading: 0.0,
            pitch: -90.0,
            fov_y: 40.0,
            sun_lat: -15.0,
            sun_lon: 86.0,
            exposure: 2.6,
            night_gain: 1.5e-6,
            gmst: 0.0,
        },
        Shot {
            name: "DEM_adelaide_6km",
            lat: -34.93,
            lon: 138.40,
            alt_km: 6.0,
            heading: 85.0,
            pitch: -16.0,
            fov_y: 50.0,
            sun_lat: -15.0,
            sun_lon: 86.0,
            exposure: 2.6,
            night_gain: 1.5e-6,
            gmst: 0.0,
        },
        Shot {
            name: "DEM_adelaide_1500m",
            lat: -34.955,
            lon: 138.545,
            alt_km: 1.5,
            heading: 70.0,
            pitch: -7.0,
            fov_y: 55.0,
            sun_lat: -15.0,
            sun_lon: 86.0,
            exposure: 2.6,
            night_gain: 1.5e-6,
            gmst: 0.0,
        },
        // Close graft tile proof: nadir over Toorak Gardens / Rose Park (inside the DEM-centred
        // DEM_CLOSE_KM≈2..3 km tile) so ≤ 2 m texels fill the frame → BuildingLod objects
        // (per-roof variants + outlines). Norwood (138.63) sits ~3.5 km off-centre and misses a 2 km half.
        Shot {
            name: "DEM_adelaide_objects",
            lat: -34.905,
            lon: 138.605,
            alt_km: 1.8,
            heading: 0.0,
            pitch: -90.0,
            fov_y: 40.0,
            sun_lat: -15.0,
            sun_lon: 86.0,
            exposure: 2.6,
            night_gain: 1.5e-6,
            gmst: 0.0,
        },
    ]
}

struct Cam {
    pos: DVec3,
    fwd: DVec3,
    right: DVec3,
    up: DVec3,
}

/// Exact camera from a planet-frame position, forward and up hint (the descent's poses).
fn camera_ecef(pos: DVec3, fwd: DVec3, up_hint: DVec3) -> Cam {
    let fwd = fwd.normalize();
    let right = fwd.cross(up_hint).normalize();
    let up = right.cross(fwd);
    Cam { pos, fwd, right, up }
}

fn camera(s: &Shot) -> Cam {
    let d = latlon_dir(s.lat, s.lon);
    let pos = d * (planet::EARTH_RADIUS_M + s.alt_km * 1000.0);
    let east = DVec3::Z.cross(d).normalize();
    let north = d.cross(east);
    let (h, p) = (s.heading.to_radians(), s.pitch.to_radians());
    let horiz = north * h.cos() + east * h.sin();
    let fwd = (horiz * p.cos() + d * p.sin()).normalize();
    // keep the local vertical as "up" (nadir shots: north is up)
    let ref_up = if fwd.dot(d).abs() > 0.999 { north } else { d };
    let right = fwd.cross(ref_up).normalize();
    let up = right.cross(fwd);
    Cam { pos, fwd, right, up }
}

/// Bake + upload the DEM patch's graft textures for sun direction `sun` (planet frame) and return
/// the five graft parameter vectors (`graft`, `graft_e`, `graft_n`, `graft_l`, `graft_t`).
/// Albedo gain 1 and a full-disc tint (real albedo at LOD 0 over the bake's ~2.4 km low-pass):
/// the palette only adds the DEM's sub-texel variation to the Blue Marble colour.
/// Bake + upload DEM graft levels. Returns `(graft param vectors, close_half_km)` where
/// `close_half_km` is 0 when the close streamed tile is off (shader skips layer 2).
fn dem_graft(
    be: &mut AshBackend,
    tex: &mut render_ash::planet::PlanetTextures,
    pl: &planet::dem::DemPlacement,
    k: f64,
    sun: DVec3,
    planet_height: &(dyn Fn(DVec3) -> f64 + Sync),
) -> ([[f32; 4]; 5], f32) {
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok());
    let size = env("DEM_TEX").unwrap_or(1024.0) as u32;
    // DEM_CLOSE_KM: half extent of the streamed close tile. Default 2 km when an OSM drape is
    // present (objects LOD needs ≤ 6.8 m texels → 2 km @ 2048 = 1.95 m); 0 disables.
    let close_m = match env("DEM_CLOSE_KM") {
        Some(v) if v <= 0.0 => None,
        Some(v) => Some(v * 1000.0),
        None if pl.drape().is_some() => Some(3000.0),
        None => None,
    };
    let t = Instant::now();
    let g = pl.bake_graft_close(sun, size, planet_height, close_m);
    let bake_ms = t.elapsed().as_secs_f64() * 1e3;
    let alb: Vec<Vec<Vec<u8>>> =
        g.levels.iter().map(|l| render_ash::planet::rgba8_mips(l.size, &l.albedo, true)).collect();
    let aux: Vec<Vec<Vec<u8>>> =
        g.levels.iter().map(|l| render_ash::planet::rgba8_mips(l.size, &l.aux, false)).collect();
    be.planet_set_graft(tex, size, &alb, &aux).expect("upload DEM graft");
    let close_km = g.levels.get(2).map(|l| (l.half_m / 1000.0) as f32).unwrap_or(0.0);
    let steps: String =
        g.levels.iter().map(|l| format!("{:.2}m", 2.0 * l.half_m / l.size as f64)).collect::<Vec<_>>().join(", ");
    eprintln!(
        "dem graft: {} x {size}^2 (texels {steps}) baked in {bake_ms:.0} ms, {:.1} MB{}",
        g.levels.len(),
        tex.graft_bytes as f64 / 1048576.0,
        if close_km > 0.0 { format!(", close {close_km:.2} km") } else { String::new() }
    );
    let m = pl.graft_map();
    let (c, e, n) = (m.centre * k, m.east, m.north);
    let r = planet::EARTH_RADIUS_M;
    (
        [
            [c.x as f32, c.y as f32, c.z as f32, (pl.r_full_m / r) as f32],
            [e.x as f32, e.y as f32, e.z as f32, (pl.r_out_m / r) as f32],
            [n.x as f32, n.y as f32, n.z as f32, (g.levels[0].half_m / 1000.0) as f32],
            [(g.levels[1].half_m / 1000.0) as f32, env("DEM_ALB").unwrap_or(1.0) as f32, 0.0, size as f32],
            [
                env("DEM_TINT").unwrap_or(1.0) as f32,
                // inner radius -1 rad: the tint is ~1 over the whole disc
                -1.0,
                0.0,
                (2400.0 * size as f64 / (2.0 * g.levels[0].half_m)).log2() as f32,
            ],
        ],
        close_km,
    )
}

fn planes(c: &Cam, fov_y: f64, aspect: f64) -> [DVec4; 6] {
    let (ty, tx) = ((fov_y / 2.0).tan(), (fov_y / 2.0).tan() * aspect);
    let pl = |n: DVec3, w: f64| {
        let l = n.length();
        DVec4::new(n.x / l, n.y / l, n.z / l, w / l)
    };
    let (f, r, u) = (c.fwd, c.right, c.up);
    [pl(r + f * tx, 0.0), pl(-r + f * tx, 0.0), pl(u + f * ty, 0.0), pl(-u + f * ty, 0.0), pl(f, -1.0), pl(-f, 1e12)]
}

fn aces(x: f32) -> f32 {
    let (a, b, c, d, e) = (2.51, 0.03, 2.43, 0.59, 0.14);
    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}

struct Star {
    dir: DVec3,
    flux: f64,
    rgb: [f32; 3],
}

/// Yale BSC5 (CDS V/50 fixed format): J2000 RA/Dec, V magnitude, B-V colour.
fn load_stars(path: &Path, gmst_deg: f64) -> Vec<Star> {
    let Ok(txt) = std::fs::read(path) else { return Vec::new() };
    let mut out = Vec::new();
    for l in txt.split(|&b| b == b'\n') {
        let f = |a: usize, b: usize| -> Option<f64> { std::str::from_utf8(l.get(a - 1..b)?).ok()?.trim().parse().ok() };
        let (Some(rh), Some(rm), Some(rs), Some(dd), Some(dm), Some(ds), Some(v)) =
            (f(76, 77), f(78, 79), f(80, 83), f(85, 86), f(87, 88), f(89, 90), f(103, 107))
        else {
            continue;
        };
        let sign = if l.get(83) == Some(&b'-') { -1.0 } else { 1.0 };
        let ra = (rh + rm / 60.0 + rs / 3600.0) * 15.0 - gmst_deg;
        let dec = sign * (dd + dm / 60.0 + ds / 3600.0);
        let bv = f(110, 114).unwrap_or(0.6).clamp(-0.4, 2.0);
        // B-V -> approximate linear sRGB (blackbody-ish ramp), luminance-normalised
        let t = ((bv + 0.4) / 2.4) as f32;
        let rgb = [0.75 + 0.5 * t, 0.85 + 0.1 * (1.0 - (2.0 * t - 1.0).abs()), 1.25 - 0.9 * t];
        let y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        out.push(Star { dir: latlon_dir(dec, ra), flux: 10f64.powf(-0.4 * (v + 26.74)), rgb: rgb.map(|c| c / y) });
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let shot = arg("--shot").unwrap_or("all".into());
    let pack_path = PathBuf::from(arg("--pack").unwrap_or("/workspace/data/earth/earth.pln".into()));
    let out = PathBuf::from(arg("--out").unwrap_or("out/quality/earth".into()));
    let (w, h) = arg("--size")
        .and_then(|s| s.split_once('x').map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap())))
        .unwrap_or((1920u32, 1080u32));
    let ss: u32 = arg("--ss").and_then(|s| s.parse().ok()).unwrap_or(2);
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok());
    std::fs::create_dir_all(&out).unwrap();
    // textures: EARTH_TEX=ktx2 (default: <data>/ktx2, BCn + zstd), nosc (<data>/ktx2_nosc, BCn,
    // no supercompression), raw (<data>/ktx2_raw, uncompressed RGBA8 / RG8 / R8 + zstd: the
    // quality reference), pln (the round-1 PLN1 pack, --pack)
    let mode = std::env::var("EARTH_TEX").unwrap_or("ktx2".into());
    // clouds: EARTH_CLOUDS=vol (default: ray-marched volume) | 2d (round-1 layer at CLOUD_KM)
    let cloud_mode = std::env::var("EARTH_CLOUDS").unwrap_or("vol".into());
    let data_dir = pack_path.parent().map(Path::to_path_buf).unwrap_or_default();
    let validate = std::env::var("PLANET_VALIDATE").map(|v| v != "0").unwrap_or(true);
    let mut be = AshBackend::new(
        render_ash::BackendConfig { max_instances: 16, validation: validate, ..Default::default() },
        None,
    )
    .expect("vulkan");
    let t0 = Instant::now();
    let (tex, height, disk_bytes, read_ms, parse_ms) = if mode == "pln" {
        let bytes = std::fs::read(&pack_path).unwrap_or_else(|e| {
            panic!("{}: {e} (tools/fetch_earth_data.sh, then the earth_bake example)", pack_path.display())
        });
        let read_ms = t0.elapsed().as_secs_f64() * 1e3;
        let disk = bytes.len() as u64;
        let pack = PlanetPack::from_bytes(&bytes).expect("verified PLN1 pack");
        drop(bytes);
        let parse_ms = t0.elapsed().as_secs_f64() * 1e3 - read_ms;
        (be.planet_upload(&pack).expect("upload planet textures"), pack.height, disk, read_ms, parse_ms)
    } else {
        let dir = arg("--tex").map(PathBuf::from).unwrap_or(data_dir.join(match mode.as_str() {
            "raw" => "ktx2_raw",
            "nosc" => "ktx2_nosc",
            _ => "ktx2",
        }));
        let read = |n: &str| {
            std::fs::read(dir.join(format!("{n}.ktx2")))
                .unwrap_or_else(|e| panic!("{}/{n}.ktx2: {e} (run the earth_bake example)", dir.display()))
        };
        let files: Vec<Vec<u8>> = planet::bake::KTX2_FILES.iter().map(|n| read(n)).collect();
        let hbytes = read("height");
        let read_ms = t0.elapsed().as_secs_f64() * 1e3;
        let disk = files.iter().map(|f| f.len() as u64).sum::<u64>() + hbytes.len() as u64;
        let tp = Instant::now();
        let k: Vec<Ktx2> = files.iter().map(|b| Ktx2::parse(b).expect("valid KTX2")).collect();
        let hk = Ktx2::parse(&hbytes).expect("valid height.ktx2");
        let height = planet::bake::height_from_ktx2(&hk).expect("R16 height cube");
        let parse_ms = tp.elapsed().as_secs_f64() * 1e3;
        (be.planet_upload_ktx2(&k).expect("upload planet KTX2 textures"), height, disk, read_ms, parse_ms)
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    let tex_json = format!(
        "{{\"mode\":\"{mode}\",\"disk_mb\":{:.1},\"vram_mb\":{:.1},\"read_ms\":{read_ms:.0},\"parse_ms\":{parse_ms:.0},\"decode_ms\":{:.0},\"load_total_ms\":{load_ms:.0},\"images\":[{}]}}",
        disk_bytes as f64 / 1048576.0,
        tex.bytes as f64 / 1048576.0,
        tex.decode_ms,
        tex.images
            .iter()
            .map(|(f, s, m, b)| format!("[\"{f:?}\",{s},{m},{:.1}]", *b as f64 / 1048576.0))
            .collect::<Vec<_>>()
            .join(",")
    );
    eprintln!("textures {tex_json}");
    let shape = Shape { radius: planet::EARTH_RADIUS_M, h_min: -500.0, h_max: 8900.0 };
    let list3 = |k: &str| {
        std::env::var(k).ok().map(|v| v.split(',').map(|t| t.trim().parse::<f64>().expect(k)).collect::<Vec<f64>>())
    };
    let cam_ecef = arg("--cam-ecef").map(|v| {
        let c: Vec<f64> = v.split(',').map(|t| t.trim().parse().expect("--cam-ecef: 9 numbers")).collect();
        assert_eq!(c.len(), 9, "--cam-ecef x,y,z,fx,fy,fz,ux,uy,uz");
        camera_ecef(DVec3::new(c[0], c[1], c[2]), DVec3::new(c[3], c[4], c[5]), DVec3::new(c[6], c[7], c[8]))
    });
    let shot_list: Vec<Shot> = match &cam_ecef {
        Some(c) => {
            let (lat, lon) = planet::cube::dir_latlon(c.pos);
            let d = c.pos.normalize();
            let east = DVec3::Z.cross(d).normalize();
            let north = d.cross(east);
            let name: &'static str = Box::leak(arg("--name").unwrap_or("custom".into()).into_boxed_str());
            vec![Shot {
                name,
                lat,
                lon,
                alt_km: c.pos.length() / 1000.0 - R_KM,
                heading: c.fwd.dot(east).atan2(c.fwd.dot(north)).to_degrees(),
                pitch: c.fwd.dot(d).clamp(-1.0, 1.0).asin().to_degrees(),
                fov_y: arg("--fov").and_then(|v| v.parse().ok()).unwrap_or(47.0),
                sun_lat: 0.0,
                sun_lon: 0.0,
                exposure: 2.6,
                night_gain: 1.5e-6,
                gmst: 0.0,
            }]
        }
        None => shots().into_iter().filter(|s| shot == "all" || s.name.ends_with(&shot) || s.name == shot).collect(),
    };
    // terrain flattening under the descent's mountains layer (FLATTEN=lat,lon,r0_km,r1_km,h_m)
    // optional 6th value: strength 0..1 (the descent's ground weight: the terrain morphs flat as
    // the mountains layer fades in; 0 = the real terrain)
    let flatten = list3("FLATTEN").map(|v| {
        (latlon_dir(v[0], v[1]), v[2] / R_KM, v[3] / R_KM, v[4], v.get(5).copied().unwrap_or(1.0).clamp(0.0, 1.0))
    });
    let height_at = |d: DVec3| -> f64 {
        let h = height.sample(d);
        match flatten {
            Some((c, a0, a1, hf, k)) => {
                let a = d.normalize().dot(c).clamp(-1.0, 1.0).acos();
                let t = ((a - a0) / (a1 - a0)).clamp(0.0, 1.0);
                let w = (1.0 - t * t * (3.0 - 2.0 * t)) * k;
                h - (h - h.min(hf)) * w
            }
            None => h,
        }
    };
    // shared Alps graft (GRAFT=path,strength)
    let graft_env = std::env::var("GRAFT").ok().map(|v| {
        let (f, k) = v.rsplit_once(',').expect("GRAFT=<file>,<strength>");
        let bytes = std::fs::read(f).unwrap_or_else(|e| panic!("{f}: {e}"));
        (
            planet::graft::GraftFile::from_bytes(&bytes).expect("GRF1 graft file"),
            k.parse::<f64>().expect("GRAFT strength").clamp(0.0, 1.0),
            f.to_string(),
        )
    });
    let graft_ctx = graft_env.as_ref().map(|(g, k, _)| {
        let fr =
            planet::descent::Frames::new(&planet::descent::Anchor { lat: g.lat, lon: g.lon, world_xz: g.world_xz });
        (planet::graft::GraftMap::from_frames(&fr), g.r_full_m, g.r_out_m, *k)
    });
    // Earth round 5: GROUND=<file.grd> grounds the grafted field on real elevation
    let field = match std::env::var("GROUND") {
        Ok(f) if f != "off" => voxel::fields::Alpine::grounded(
            7,
            std::sync::Arc::new(voxel::ground::Ground::load(&f).unwrap_or_else(|e| panic!("{e}"))),
        ),
        _ => voxel::fields::Alpine::new(7),
    };
    // DEM patch (`planet::dem`): DEM=<geo-fetch terrain-grid dir>[,strength] puts a real DEM on
    // the planet (patch heights inside its disc, shading through the graft bindings);
    // DEM_R=r_full_km,r_out_km overrides the blend disc, DEM_TEX the graft texture size (1024)
    let dem = std::env::var("DEM").ok().filter(|v| v != "off").map(|v| {
        let (dir, k) = match v.rsplit_once(',') {
            Some((d, k)) if k.parse::<f64>().is_ok() => (d.to_string(), k.parse::<f64>().unwrap()),
            _ => (v.clone(), 1.0),
        };
        let hg = geo_fetch::grid::read_grid(Path::new(&dir)).unwrap_or_else(|e| panic!("DEM {dir}: {e}"));
        let enu = hg.enu();
        let grid = terrain::dem::DemGrid::new(
            hg.width,
            hg.height,
            hg.spacing_m as f32,
            hg.x0 as f32,
            hg.y0 as f32,
            hg.heights,
        )
        .expect("DEM grid shape");
        let mut pl = planet::dem::DemPlacement::new(enu, grid);
        if let Some(r) = list3("DEM_R") {
            pl = pl.with_blend(r[0] * 1000.0, r[1] * 1000.0);
        }
        // OSM=<geo-fetch osm-extract dir>[,cell_m]: drape roads / buildings / water / landuse
        // (planet::drape) over the DEM palette in the graft bake
        if let Some(v) = std::env::var("OSM").ok().filter(|v| v != "off") {
            let (odir, cell) = match v.rsplit_once(',') {
                Some((d, c)) if c.parse::<f64>().is_ok() => (d.to_string(), c.parse::<f64>().unwrap()),
                _ => (v.clone(), 4.0),
            };
            let d = std::sync::Arc::new(osm_drape(&pl, Path::new(&odir), cell));
            pl = pl.with_drape(d);
        }
        eprintln!(
            "dem: {dir} {}x{} @ {} m, disc {:.2}..{:.2} km, strength {k}",
            pl.grid().width,
            pl.grid().height,
            pl.grid().spacing_m,
            pl.r_full_m / 1000.0,
            pl.r_out_m / 1000.0
        );
        (pl, k.clamp(0.0, 1.0))
    });
    assert!(!(dem.is_some() && graft_env.is_some()), "DEM and GRAFT share the graft bindings: set one");
    let height_lvl = |d: DVec3, level: u8| -> f64 {
        let h = height_at(d);
        let h = match graft_ctx {
            Some((map, r0, r1, k)) => graft::graft_height(&field, &map, (r0, r1, k), d, h, graft::level_spacing(level)),
            None => h,
        };
        match &dem {
            Some((pl, k)) => pl.blend_height(d, h, graft::level_spacing(level), *k),
            None => h,
        }
    };
    let mut tex = tex;
    let graft_params = match (&graft_env, &graft_ctx) {
        (Some((g, _, gpath)), Some((map, r0, r1, k))) if *k > 0.0 => {
            let tg = Instant::now();
            let size = g.levels[0].size;
            let raw = std::env::var("GRAFT_TEX").is_ok_and(|v| v == "raw");
            if raw {
                let alb: Vec<Vec<Vec<u8>>> =
                    g.levels.iter().map(|l| render_ash::planet::rgba8_mips(l.size, &l.albedo, true)).collect();
                let aux: Vec<Vec<Vec<u8>>> =
                    g.levels.iter().map(|l| render_ash::planet::rgba8_mips(l.size, &l.aux, false)).collect();
                be.planet_set_graft(&mut tex, size, &alb, &aux).expect("upload graft");
            } else {
                let (fa, alb) = graft_ktx2(g, gpath, "alb");
                let (fx, aux) = graft_ktx2(g, gpath, "aux");
                be.planet_set_graft_fmt(&mut tex, size, (fa, &alb), (fx, &aux)).expect("upload graft (BC7)");
            }
            eprintln!(
                "graft: {} levels {size}^2, {} {:.1} MB, {:.0} ms, strength {k:.4}",
                g.levels.len(),
                if raw { "RGBA8" } else { "BC7" },
                tex.graft_bytes as f64 / 1048576.0,
                tg.elapsed().as_secs_f64() * 1e3
            );
            let (c, e, n) = (map.centre * *k, map.east, map.north);
            Some([
                [c.x as f32, c.y as f32, c.z as f32, (*r0 / planet::EARTH_RADIUS_M) as f32],
                [e.x as f32, e.y as f32, e.z as f32, (*r1 / planet::EARTH_RADIUS_M) as f32],
                [n.x as f32, n.y as f32, n.z as f32, (g.levels[0].half_m / 1000.0) as f32],
                [
                    (g.levels[1].half_m / 1000.0) as f32,
                    env("GRAFT_ALB").unwrap_or(0.75) as f32,
                    env("GRAFT_AO").unwrap_or(1.0) as f32,
                    size as f32,
                ],
                // tint: real albedo cube mip ~20 km (4096 face: 2.4 km texels -> lod 3), graft outer
                // level the same (83 m texels -> lod 8)
                [
                    env("GRAFT_TINT").unwrap_or(1.0) as f32,
                    (env("GRAFT_TINT_KM").unwrap_or(60.0) * 1000.0 / planet::EARTH_RADIUS_M) as f32,
                    3.0,
                    (20_000.0 * size as f64 / (2.0 * g.levels[0].half_m)).log2() as f32,
                ],
            ])
        }
        _ => None,
    };
    let hole = list3("HOLE").map_or([0.0; 4], |v| {
        let c = latlon_dir(v[0], v[1]);
        [c.x as f32, c.y as f32, c.z as f32, v[2].to_radians() as f32]
    });
    for s in shot_list {
        let mut s = s;
        if let Some(v) = env("EXPOSURE") {
            s.exposure = v;
        }
        if let (Some(a), Some(b)) = (env("SUN_LAT"), env("SUN_LON")) {
            (s.sun_lat, s.sun_lon) = (a, b);
        }
        let (rw, rh) = (w * ss, h * ss);
        let aspect = rw as f64 / rh as f64;
        let fov = s.fov_y.to_radians();
        let c = match &cam_ecef {
            Some(c) => Cam { pos: c.pos, fwd: c.fwd, right: c.right, up: c.up },
            None => camera(&s),
        };
        let tsel = Instant::now();
        let view = View {
            cam: c.pos,
            planes: planes(&c, fov, aspect),
            px_per_rad: rh as f64 / (2.0 * (fov / 2.0).tan()),
            max_error_px: env("LOD_PX").unwrap_or(1.5) * ss as f64,
            max_level: 14,
        };
        let sel = quadtree::select(&view, &shape);
        let sel_ms = tsel.elapsed().as_secs_f64() * 1e3;
        let tb = Instant::now();
        let meshes: Vec<patch::PatchMesh> = sel
            .visible
            .par_iter()
            .map(|(k, _)| patch::build(k, planet::EARTH_RADIUS_M, &|d| height_lvl(d, k.level)))
            .collect();
        let mut vertices: Vec<PVertex> = Vec::with_capacity(meshes.len() * VERTS as usize);
        let mut patches = Vec::with_capacity(meshes.len());
        for m in &meshes {
            vertices.extend_from_slice(&m.verts);
            let o = (m.origin - c.pos).as_vec3();
            patches.push(PatchGpu { origin: [o.x, o.y, o.z, m.key.level as f32] });
        }
        let masks: Vec<u8> = sel.visible.iter().map(|x| x.1).collect();
        let build_ms = tb.elapsed().as_secs_f64() * 1e3;
        // camera-relative view-projection (rotation only; positions are already camera-relative)
        let (f, r, u) = (c.fwd.as_vec3(), c.right.as_vec3(), c.up.as_vec3());
        let view_m = Mat4::from_cols(r.extend(0.0), u.extend(0.0), (-f).extend(0.0), glam::Vec4::W).transpose();
        let near = (s.alt_km * 1000.0 * 0.05).clamp(1.0, 100_000.0) as f32;
        let proj = render_ash::quality::reverse_z_infinite(fov as f32, aspect as f32, near);
        let vp = proj * view_m;
        let sun = latlon_dir(s.sun_lat, s.sun_lon);
        // DEM: the graft textures are baked per shot (the cast shadows follow its sun);
        // close_km > 0 streams BuildingLod objects via graft array layer 2 (meso2.y).
        let (graft_params, close_km) = match &dem {
            Some((pl, k)) if *k > 0.0 => {
                let (g, c) = dem_graft(&mut be, &mut tex, pl, *k, sun, &height_at);
                (Some(g), c)
            }
            _ => (graft_params, 0.0),
        };
        let alt_km = (c.pos.length() / 1000.0) - R_KM;
        let ty = (fov / 2.0).tan();
        let p = PlanetParamsGpu {
            view_proj: vp.to_cols_array_2d(),
            cam_km: [(c.pos.x / 1000.0) as f32, (c.pos.y / 1000.0) as f32, (c.pos.z / 1000.0) as f32, alt_km as f32],
            sun: [sun.x as f32, sun.y as f32, sun.z as f32, 0.004_65],
            right: [r.x, r.y, r.z, (ty * aspect) as f32],
            up: [u.x, u.y, u.z, ty as f32],
            fwd: [f.x, f.y, f.z, env("DEBUG").unwrap_or(0.0) as f32],
            planet: [
                R_KM as f32,
                (R_KM + 100.0) as f32,
                (R_KM + env("CLOUD_KM").unwrap_or(6.0)) as f32,
                env("CLOUD_GAMMA").unwrap_or(1.6) as f32,
            ],
            ocean: [
                (0.003 + 0.00512 * env("WIND").unwrap_or(6.0)) as f32,
                0.02,
                0.22,
                env("RELIEF").unwrap_or(1.0) as f32,
            ],
            light: [
                env("NIGHT").unwrap_or(s.night_gain) as f32,
                env("AIRGLOW").unwrap_or(5.0e-10) as f32,
                env("ALBEDO").unwrap_or(0.85) as f32,
                env("CLOUD_SHADOW").unwrap_or(1.0) as f32,
            ],
            atmo: [env("MIE").unwrap_or(1.0) as f32, 1.0, 1.0, env("GROUND_ALB").unwrap_or(0.15) as f32],
            view: [rw as f32, rh as f32, s.exposure as f32, 0.0],
            cloud: [
                env("CLOUD_BASE").unwrap_or(1.0) as f32,
                env("CLOUD_TOP").unwrap_or(9.0) as f32,
                env("CLOUD_EXT").unwrap_or(24.0) as f32,
                if cloud_mode == "2d" { 0.0 } else { 1.0 },
            ],
            cloud2: [
                env("CLOUD_STEPS").unwrap_or(48.0) as f32,
                env("CLOUD_SUN_STEPS").unwrap_or(6.0) as f32,
                env("CLOUD_NOISE_KM").unwrap_or(10.0) as f32,
                env("CLOUD_MS").unwrap_or(2.5) as f32,
            ],
            aero: [
                env("AERO_FT").unwrap_or(6.0e-3) as f32,
                env("AERO_H").unwrap_or(3.0) as f32,
                env("AERO_STRAT").unwrap_or(4.0e-4) as f32,
                env("EARTH_DETAIL").unwrap_or(1.0) as f32,
            ],
            hole,
            // far-footprint cloud structure (round 3; MESO_TOP=0 MESO_AMP=0 MESO_BIAS=0 MESO_FINE=0 MESO_GAMMA=0 = round 2)
            meso: [
                env("MESO_TOP").unwrap_or(2.0) as f32,
                env("MESO_AMP").unwrap_or(4.5) as f32,
                env("MESO_BIAS").unwrap_or(-0.8) as f32,
                env("MESO_FINE").unwrap_or(1.0) as f32,
            ],
            // round 4 (EPIC cloud fraction): far coverage gamma; MESO_GAMMA=0 MESO_FINE=0
            // MESO_BIAS=-0.7 = round 3
            meso2: [env("MESO_GAMMA").unwrap_or(0.9) as f32, close_km, 0.0, 0.0],
            thaw: flatten
                .map_or([0.0; 4], |(c, _, a1, _, k)| [(c.x * k) as f32, (c.y * k) as f32, (c.z * k) as f32, a1 as f32]),
            graft: graft_params.map_or([0.0; 4], |g| g[0]),
            graft_e: graft_params.map_or([0.0; 4], |g| g[1]),
            graft_n: graft_params.map_or([0.0; 4], |g| g[2]),
            graft_l: graft_params.map_or([0.0; 4], |g| g[3]),
            graft_t: graft_params.map_or([0.0; 4], |g| g[4]),
            // round 6: marine haze over open ocean (OCEAN_HAZE=0 = round 5)
            haze: [
                env("OCEAN_HAZE").unwrap_or(0.15) as f32,
                env("HAZE_ANGSTROM").unwrap_or(0.4) as f32,
                env("HAZE_SSA").unwrap_or(0.98) as f32,
                env("HAZE_MASK_KM").unwrap_or(150.0) as f32,
            ],
            // round 7: cloud-edge feather (km, mix), clear-window edge noise, haze dust
            // (CLOUD_FEATHER=0 HOLE_NOISE=0 HAZE_DUST=0 = round 6)
            cloud3: [
                env("CLOUD_FEATHER").unwrap_or(16.0) as f32,
                env("CLOUD_FEATHER_MIX").unwrap_or(0.6) as f32,
                env("HOLE_NOISE").unwrap_or(0.35) as f32,
                env("HAZE_DUST").unwrap_or(0.0) as f32,
            ],
        };
        let (haze, cloud3) = (p.haze, p.cloud3);
        let frame =
            PlanetFrame { width: rw, height: rh, params: p, patches: &patches, masks: &masks, vertices: &vertices };
        let tr = Instant::now();
        let (px, t) = be.render_planet(&tex, &frame).expect("render planet");
        let render_ms = tr.elapsed().as_secs_f64() * 1e3;
        // REPEAT=n: render n-1 more times, report the last (warm) GPU time next to the first (cold)
        let cold_gpu_ms = t.gpu_ms;
        let (px, t) = {
            let mut last = (px, t);
            for _ in 1..std::env::var("REPEAT").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(1) {
                last = be.render_planet(&tex, &frame).expect("render planet");
            }
            last
        };
        // stars: physical flux relative to the sun (illuminance 1), spread over a 2x2 texel kernel
        let mut hdr = px;
        // STARS=0 (solar round 1, the system renderer's Earth handoff): no star splat here, the
        // system pass draws the sky behind Earth
        let stars = if std::env::var("STARS").as_deref() == Ok("0") {
            Vec::new()
        } else {
            load_stars(&data_dir.join("bsc5.dat"), s.gmst)
        };
        let px_sr = (2.0 * ty / rh as f64).powi(2);
        let mut nstars = 0;
        for st in &stars {
            let d = st.dir;
            let z = d.dot(c.fwd);
            if z <= 0.0 {
                continue;
            }
            let x = d.dot(c.right) / z / (ty * aspect);
            let y = -d.dot(c.up) / z / ty;
            if x.abs() >= 1.0 || y.abs() >= 1.0 {
                continue;
            }
            let fx = (x * 0.5 + 0.5) * rw as f64 - 0.5;
            let fy = (y * 0.5 + 0.5) * rh as f64 - 0.5;
            let (ix, iy) = (fx.floor() as i64, fy.floor() as i64);
            let (ax, ay) = (fx - ix as f64, fy - iy as f64);
            let l = (st.flux / px_sr * s.exposure) as f32;
            for (dx, dy, wgt) in
                [(0, 0, (1.0 - ax) * (1.0 - ay)), (1, 0, ax * (1.0 - ay)), (0, 1, (1.0 - ax) * ay), (1, 1, ax * ay)]
            {
                let (xx, yy) = (ix + dx, iy + dy);
                if xx < 0 || yy < 0 || xx >= rw as i64 || yy >= rh as i64 {
                    continue;
                }
                let q = &mut hdr[(yy as u32 * rw + xx as u32) as usize];
                let tr = q[3];
                for (qk, ck) in q.iter_mut().zip(st.rgb) {
                    *qk += l * wgt as f32 * tr * ck;
                }
            }
            nstars += 1;
        }
        // HDR_OUT=1 (solar round 1): also write <name>.hdr.bin, the exposed linear HDR at the
        // render size (rw x rh x RGBA f32 little-endian, alpha = transmittance toward space)
        if std::env::var("HDR_OUT").as_deref() == Ok("1") {
            std::fs::write(out.join(format!("{}.hdr.bin", s.name)), bytemuck::cast_slice::<[f32; 4], u8>(&hdr))
                .unwrap();
        }
        // PROBE="x,y;x,y" (output pixels): print the linear radiance (units of solar illuminance)
        let unexp = 1.0 / s.exposure as f32;
        if let Ok(pr) = std::env::var("PROBE") {
            for q in pr.split(';').filter_map(|q| q.split_once(',')) {
                let (x, y): (u32, u32) = (q.0.parse().unwrap(), q.1.parse().unwrap());
                let v = hdr[((y * ss) * rw + x * ss) as usize];
                eprintln!(
                    "probe {x},{y}: {:.3e} {:.3e} {:.3e} (a {:.3})",
                    v[0] * unexp,
                    v[1] * unexp,
                    v[2] * unexp,
                    v[3]
                );
            }
        }
        // box downsample in linear light, exposure, ACES, sRGB
        let mut rgb8 = vec![0u8; (w * h * 3) as usize];
        // EPIC_GRADE (round 7, non-physical, off by default): a display-linear 3x3 colour grade
        // toward DSCOVR/EPIC's processed look (greyer, darker ocean). "1" = the matrix fitted on
        // E2_vs_epic vs epic_africa.jpg (class means of clear ocean / land / cloud, ridge toward
        // identity; /workspace/e7work/fitgrade.py); a number s blends identity -> fitted by s;
        // nine comma-separated values give a custom row-major matrix. Unset / "0" = no grade.
        let grade = epic_grade();
        let k = 1.0f32; // pre-exposed on the GPU (params.view.z)
        let inv = 1.0 / (ss * ss) as f32;
        rgb8.par_chunks_mut((w * 3) as usize).enumerate().for_each(|(y, row)| {
            for x in 0..w as usize {
                let mut a = [0.0f32; 3];
                for sy in 0..ss as usize {
                    for sx in 0..ss as usize {
                        let q = hdr[(y * ss as usize + sy) * rw as usize + x * ss as usize + sx];
                        for (ac, qc) in a.iter_mut().zip(q) {
                            *ac += aces(qc.max(0.0) * k);
                        }
                    }
                }
                if let Some(g) = &grade {
                    let l = a.map(|v| v * inv);
                    a = [0, 1, 2].map(|i| (g[i][0] * l[0] + g[i][1] * l[1] + g[i][2] * l[2]).clamp(0.0, 1.0) / inv);
                }
                for (o, ac) in row[x * 3..x * 3 + 3].iter_mut().zip(a) {
                    *o = (planet::bake::linear_to_srgb(ac * inv) * 255.0 + 0.5) as u8;
                }
            }
        });
        // SKY_MASK=1: per output pixel, the fraction of its samples whose ray reached space
        // (alpha = transmittance > 0; the ground pass writes 0)
        if std::env::var("SKY_MASK").as_deref() == Ok("1") {
            let mut m = vec![0u8; (w * h) as usize];
            m.par_chunks_mut(w as usize).enumerate().for_each(|(y, row)| {
                for (x, o) in row.iter_mut().enumerate() {
                    let mut n = 0u32;
                    for sy in 0..ss as usize {
                        for sx in 0..ss as usize {
                            n += (hdr[(y * ss as usize + sy) * rw as usize + x * ss as usize + sx][3] > 0.0) as u32;
                        }
                    }
                    *o = ((n * 255 + ss * ss / 2) / (ss * ss)) as u8;
                }
            });
            let file = std::fs::File::create(out.join(format!("{}_sky.png", s.name))).unwrap();
            let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
            enc.set_color(png::ColorType::Grayscale);
            enc.set_depth(png::BitDepth::Eight);
            enc.write_header().unwrap().write_image_data(&m).unwrap();
        }
        let path = out.join(format!("{}.png", s.name));
        let file = std::fs::File::create(&path).unwrap();
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().unwrap().write_image_data(&rgb8).unwrap();
        let hist = quadtree::level_histogram(sel.visible.iter().map(|x| x.0));
        let mut levels: Vec<(u8, u32)> = hist.into_iter().collect();
        levels.sort();
        // strict JSON (Debug would print Rust tuples): arrays of [level, count] / [pass, ms]
        let levels_json = json_array(levels.iter().map(|(l, n)| format!("[{l},{n}]")));
        let breakdown_json = json_array(t.breakdown.iter().map(|(n, ms)| format!("[\"{n}\",{ms:.3}]")));
        let shot_json = format!(
            "{{\"lat\":{},\"lon\":{},\"heading\":{},\"pitch\":{},\"fov_y\":{},\"sun_lat\":{},\"sun_lon\":{},\"exposure\":{},\"clouds\":\"{cloud_mode}\",\"ocean_haze\":{{\"aod550\":{},\"angstrom\":{},\"ssa\":{},\"mask_km\":{},\"dust\":{}}},\"cloud_feather_km\":{},\"cloud_feather_mix\":{},\"hole_noise\":{},\"epic_grade\":{},\"timing_device\":\"lavapipe (CPU Vulkan), not a real-GPU timing\"}}",
            s.lat,
            s.lon,
            s.heading,
            s.pitch,
            s.fov_y,
            s.sun_lat,
            s.sun_lon,
            s.exposure,
            haze[0],
            haze[1],
            haze[2],
            haze[3],
            cloud3[3],
            cloud3[0],
            cloud3[1],
            cloud3[2],
            grade.map_or("null".to_string(), |g| format!("{:?}", g))
        );
        let json = format!(
            "{{\"shot\":\"{}\",\"size\":[{w},{h}],\"ss\":{ss},\"alt_km\":{alt_km:.1},\"patches\":{},\"leaves\":{},\"balance_splits\":{},\"triangles\":{},\"vertices\":{},\"vertex_mb\":{:.1},\"levels\":{levels_json},\"texture_mb\":{:.1},\"select_ms\":{sel_ms:.2},\"mesh_build_ms\":{build_ms:.1},\"gpu_ms_cold\":{cold_gpu_ms:.1},\"gpu_ms\":{:.1},\"gpu_breakdown\":{breakdown_json},\"render_wall_ms\":{render_ms:.0},\"stars_drawn\":{nstars},\"validation_errors\":{},\"textures\":{tex_json},\"params\":{shot_json}}}",
            s.name,
            t.patches,
            sel.leaves.len(),
            sel.balance_splits,
            t.triangles,
            vertices.len(),
            (vertices.len() * 16) as f64 / 1048576.0,
            tex.bytes as f64 / 1048576.0,
            t.gpu_ms,
            render_ash::validation::message_counts().0,
        );
        println!("{json}");
        std::fs::write(out.join(format!("{}.json", s.name)), &json).unwrap();
    }
    be.planet_free(tex);
}

/// Load a `geo-fetch osm-extract` directory and rasterise it in `pl`'s plane (`cell_m` metres a
/// cell) over the DEM grid's rectangle. Tunnels are skipped. ODbL: © OpenStreetMap contributors.
fn osm_drape(pl: &planet::dem::DemPlacement, dir: &Path, cell_m: f64) -> planet::drape::Drape {
    use geo_fetch::derived::{DerivedGeom, read_layer};
    use planet::drape::{Drape, DrapeClass, DrapeFeature, DrapeShape};
    let t = Instant::now();
    let proj = |r: &[[f64; 2]]| -> Vec<[f64; 2]> {
        r.iter()
            .map(|p| {
                let (e, n) = pl.latlon_to_plane(p[1], p[0]);
                [e, n]
            })
            .collect()
    };
    let mut feats = Vec::new();
    // painter's order is per class (DrapeClass::ALL), so layer order here does not matter
    for layer in ["landuse", "water", "buildings", "roads"] {
        let fs = read_layer(dir, layer).unwrap_or_else(|e| panic!("OSM {}: {e}", dir.display()));
        for f in fs {
            let Some(class) = DrapeClass::from_osm(&f.key, &f.kind) else { continue };
            if f.tunnel {
                continue;
            }
            match &f.geom {
                DerivedGeom::Line(l) => feats.push(DrapeFeature {
                    class,
                    shape: DrapeShape::Line {
                        pts: proj(l),
                        width_m: f.width_m.unwrap_or_else(|| class.default_width_m(&f.kind)),
                    },
                }),
                DerivedGeom::Polygon(p) => feats.push(DrapeFeature {
                    class,
                    shape: DrapeShape::Area { rings: p.iter().map(|r| proj(r)).collect() },
                }),
                DerivedGeom::MultiPolygon(m) => feats.extend(m.iter().map(|p| DrapeFeature {
                    class,
                    shape: DrapeShape::Area { rings: p.iter().map(|r| proj(r)).collect() },
                })),
            }
        }
    }
    let n = feats.len();
    // raster over the extract's manifest bbox (projected) clipped to the graft square, faded over
    // OSM_FADE km (default 3) inside its edges; without a bbox, the DEM rectangle + its own fade
    let half = 1.1 * pl.r_out_m;
    let (c0, c1) = pl.centre_plane();
    let fade_m = std::env::var("OSM_FADE").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(3.0) * 1000.0;
    let (rect, fade) = match geo_fetch::derived::read_bbox(dir) {
        Some([w, s, e, n]) => {
            let cs = [(s, w), (s, e), (n, w), (n, e)].map(|(la, lo)| pl.latlon_to_plane(la, lo));
            let x0 = cs.iter().map(|c| c.0).fold(f64::MAX, f64::min).max(c0 - half);
            let x1 = cs.iter().map(|c| c.0).fold(f64::MIN, f64::max).min(c0 + half);
            let y0 = cs.iter().map(|c| c.1).fold(f64::MAX, f64::min).max(c1 - half);
            let y1 = cs.iter().map(|c| c.1).fold(f64::MIN, f64::max).min(c1 + half);
            ((x0, y0, x1, y1), fade_m)
        }
        None => (pl.grid_rect(), 0.0),
    };
    let d = Drape::build(&feats, rect, cell_m).with_edge_fade(fade);
    let g = pl.grid_rect();
    eprintln!(
        "osm drape rect: {:.1}..{:.1} x {:.1}..{:.1} km (DEM {:.1}..{:.1} x {:.1}..{:.1} km), fade {:.1} km",
        rect.0 / 1e3,
        rect.2 / 1e3,
        rect.1 / 1e3,
        rect.3 / 1e3,
        g.0 / 1e3,
        g.2 / 1e3,
        g.1 / 1e3,
        g.3 / 1e3,
        fade / 1e3
    );
    let (w, h) = d.size();
    eprintln!(
        "osm drape: {} -> {n} features, {w}x{h} @ {cell_m} m in {:.1} s (© OpenStreetMap contributors)",
        dir.display(),
        t.elapsed().as_secs_f64()
    );
    d
}

fn json_array(items: impl Iterator<Item = String>) -> String {
    format!("[{}]", items.collect::<Vec<_>>().join(","))
}

/// Earth round 5: the graft's albedo (`which` = "alb", BC7 sRGB) or aux ("aux", BC7 linear) as
/// per-layer BC7 mip chains (4096 .. 4), from `<grf>.<which>.ktx2`; encoded from the .grf and
/// cached when the file is missing or older than the .grf.
fn graft_ktx2(g: &planet::graft::GraftFile, grf: &str, which: &str) -> (planet::pack::TexFormat, Vec<Vec<Vec<u8>>>) {
    use planet::ktx2::{Ktx2, Supercompression, write_array};
    use planet::pack::{TexFormat, Texture, mip_bytes};
    let fmt = if which == "alb" { TexFormat::Bc7Srgb } else { TexFormat::Bc7 };
    let path = format!("{grf}.{which}.ktx2");
    let mtime = |p: &str| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    let fresh = matches!((mtime(&path), mtime(grf)), (Some(a), Some(b)) if a >= b);
    let size = g.levels[0].size;
    let mips = size.trailing_zeros() - 1; // down to 4x4
    if !fresh {
        let t0 = Instant::now();
        let mut data = Vec::new();
        for l in &g.levels {
            assert_eq!(l.size, size, "graft levels share one size");
            let src = if which == "alb" { &l.albedo } else { &l.aux };
            let chain = render_ash::planet::rgba8_mips(size, src, which == "alb");
            for (m, img) in chain.iter().take(mips as usize).enumerate() {
                data.extend(planet::bake::encode_bc7_rgba8(img, (size >> m) as usize));
            }
        }
        let t = Texture { name: which.into(), format: fmt, size, mips, data };
        let bytes = write_array(
            &t,
            g.levels.len() as u32,
            Supercompression::Zstd(12),
            &[("bevy-brutal.graft", grf.as_bytes())],
        )
        .expect("graft KTX2");
        std::fs::write(&path, &bytes).unwrap_or_else(|e| panic!("{path}: {e}"));
        eprintln!(
            "graft: encoded {path} ({:.1} MB on disk) in {:.1} s",
            bytes.len() as f64 / 1048576.0,
            t0.elapsed().as_secs_f64()
        );
    }
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let k = Ktx2::parse(&bytes).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert!(
        k.array && k.images as usize == g.levels.len() && k.size == size && k.format == fmt,
        "{path}: does not match the graft"
    );
    let lm = k.decode_level_major().expect("graft KTX2 decode");
    let mut out = vec![Vec::new(); k.images as usize];
    let mut o = 0;
    for m in 0..k.mips {
        let mb = mip_bytes(fmt, size, m);
        for chain in out.iter_mut() {
            chain.push(lm[o..o + mb].to_vec());
            o += mb;
        }
    }
    (fmt, out)
}

/// EPIC_GRADE parse (see the post-process): `None` = off.
fn epic_grade() -> Option<[[f32; 3]; 3]> {
    const FIT: [[f32; 3]; 3] = [[0.8005, 0.1943, -0.0427], [-0.0544, 1.1512, -0.1265], [0.0245, 0.3964, 0.4739]];
    let v = std::env::var("EPIC_GRADE").ok()?;
    let vals: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    match vals.len() {
        1 if vals[0] != 0.0 => {
            let s = vals[0];
            Some(std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    let id = if i == j { 1.0 } else { 0.0 };
                    id + s * (FIT[i][j] - id)
                })
            }))
        }
        9 => Some(std::array::from_fn(|i| std::array::from_fn(|j| vals[i * 3 + j]))),
        _ => None,
    }
}
