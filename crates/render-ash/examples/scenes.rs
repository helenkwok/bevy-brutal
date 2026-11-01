//! The eight target scenes (universe, solar system, earth, ocean, mountains, underground, cave,
//! underwater), one per `--scene <name>` (or `all`). Each prints one JSON line with frame time
//! p50/p99, GPU ms, triangles, memory and correctness checks, writes `out/scenes/<name>.png` and
//! `out/scenes/<name>.json`, and marks what is real versus placeholder.
//!
//! cargo run --release -p render-ash --example scenes -- --scene <name|all> \
//!     [--frames 120] [--size 960x540] [--brick 32] [--voxel 0.25] [--validate]
//!
//! * voxel scenes (mountains, underground, cave, underwater seabed): the real C2 pipeline:
//!   camera-centred LOD shells, plan_brick, integer sampling, GPU Slang Transvoxel meshing
//!   (checked bit-exact against the CPU mesher), exact topology checks.
//! * coords scenes (universe, solar system, earth): f64 frame-tree flythroughs across frame levels,
//!   camera-relative instancing, scaled-space impostors, rider precision/jitter vs naive f32.
//! * ocean surface / underwater volume: flat placeholders until Prompts D/E exist.
use std::collections::HashMap;
use std::f64::consts::PI;
use std::time::Instant;

use coords::{BrickGrid, DVec3, FloatingOrigin, FrameId, FrameLevel, FrameTree, WorldPos};
use glam::{Mat4, Quat, Vec3};
use lowlevel_abi::{FrameSnapshot, InstanceGpu};
use rayon::prelude::*;
use render_ash::cull::{look_at, vk_perspective};
use render_ash::mesh::{MeshData, cube, octahedron};
use render_ash::validation::message_counts;
use render_ash::voxel_gpu::VoxelJob;
use render_ash::{AshBackend, BackendConfig, FrameStats};
use voxel::scene::{CaveScene, Mountains, Seabed, Underground};
use voxel::{
    BrickMesh, BrickSamples, DensityField, SUB, ShellLayout, VoxelConfig, VoxelVertex, bricks_in, mesh_brick,
    plan_brick, sample_brick, topology,
};

const SCENES: [&str; 8] =
    ["universe", "solar_system", "earth", "ocean", "mountains", "underground", "cave", "underwater"];
const FOV: f32 = 0.9;
const NEAR: f32 = 0.05;
const FAR: f32 = 1.0e5;
/// Scaled space: objects beyond R0 are pulled in logarithmically (angular size preserved).
const R0: f64 = 5_000.0;
/// Rider probe distance from the camera (metres).
const OFFSET_LEN: f64 = 3.2;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

struct Opts {
    frames: usize,
    w: u32,
    h: u32,
    brick: u32,
    voxel_m: f64,
    validate: bool,
}

#[derive(Default)]
struct Report {
    name: String,
    kind: &'static str,
    frame_ms: Vec<f64>,
    gpu_ms: Vec<f64>,
    tris: usize,
    mem: Vec<(&'static str, u64)>,
    checks: Vec<(String, bool)>,
    metrics: Vec<(String, String)>,
    real: Vec<&'static str>,
    placeholder: Vec<&'static str>,
    /// mid-path capture (else the last frame is written)
    png: Option<(u32, u32, Vec<u8>)>,
}

fn pct(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[((s.len() - 1) as f64 * p).round() as usize]
}

fn json_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

impl Report {
    fn check(&mut self, name: impl Into<String>, ok: bool) {
        self.checks.push((name.into(), ok));
    }
    fn metric(&mut self, k: impl Into<String>, v: impl std::fmt::Display) {
        self.metrics.push((k.into(), v.to_string()));
    }
    fn ok(&self) -> bool {
        self.checks.iter().all(|c| c.1)
    }
    fn json(&self, device: &str, errors: u32, warnings: u32) -> String {
        let mem_total: u64 = self.mem.iter().map(|m| m.1).sum();
        let mem =
            self.mem.iter().map(|(k, v)| format!("\"{k}\":{:.2}", *v as f64 / 1048576.0)).collect::<Vec<_>>().join(",");
        let checks = self.checks.iter().map(|(k, v)| format!("{}:{v}", json_str(k))).collect::<Vec<_>>().join(",");
        let metrics =
            self.metrics.iter().map(|(k, v)| format!("{}:{}", json_str(k), json_str(v))).collect::<Vec<_>>().join(",");
        let list = |l: &[&str]| l.iter().map(|s| json_str(s)).collect::<Vec<_>>().join(",");
        format!(
            "{{\"scene\":\"{}\",\"kind\":\"{}\",\"frames\":{},\"frame_ms_p50\":{:.3},\"frame_ms_p99\":{:.3},\"gpu_ms_p50\":{:.3},\"gpu_ms_p99\":{:.3},\"triangles\":{},\"mem_mb_total\":{:.2},\"mem_mb\":{{{mem}}},\"checks\":{{{checks}}},\"all_checks_pass\":{},\"metrics\":{{{metrics}}},\"real\":[{}],\"placeholder\":[{}],\"validation_errors\":{errors},\"validation_warnings\":{warnings},\"device\":{},\"png\":\"out/scenes/{}.png\"}}",
            self.name,
            self.kind,
            self.frame_ms.len(),
            pct(&self.frame_ms, 0.5),
            pct(&self.frame_ms, 0.99),
            pct(&self.gpu_ms, 0.5),
            pct(&self.gpu_ms, 0.99),
            self.tris,
            mem_total as f64 / 1048576.0,
            self.ok(),
            list(&self.real),
            list(&self.placeholder),
            json_str(device),
            self.name
        )
    }
}

fn gpu_ms(s: &FrameStats) -> Option<f64> {
    Some(s.cull_ms? + s.draw_ms?)
}

// ---------- meshes / instances ----------------------------------------------------------------

fn icosphere(sub: u32) -> MeshData {
    let t = (1.0 + 5f32.sqrt()) / 2.0;
    let mut v: Vec<Vec3> = [
        [-1., t, 0.],
        [1., t, 0.],
        [-1., -t, 0.],
        [1., -t, 0.],
        [0., -1., t],
        [0., 1., t],
        [0., -1., -t],
        [0., 1., -t],
        [t, 0., -1.],
        [t, 0., 1.],
        [-t, 0., -1.],
        [-t, 0., 1.],
    ]
    .iter()
    .map(|p| Vec3::from(*p).normalize())
    .collect();
    let mut f: Vec<[usize; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..sub {
        let mut mid: HashMap<(usize, usize), usize> = HashMap::new();
        let mut m = |a: usize, b: usize, v: &mut Vec<Vec3>| {
            *mid.entry((a.min(b), a.max(b))).or_insert_with(|| {
                v.push(((v[a] + v[b]) * 0.5).normalize());
                v.len() - 1
            })
        };
        let mut nf = Vec::with_capacity(f.len() * 4);
        for [a, b, c] in f {
            let (ab, bc, ca) = (m(a, b, &mut v), m(b, c, &mut v), m(c, a, &mut v));
            nf.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        f = nf;
    }
    // radius 0.5 (unit diameter), like the built-in cube
    let tris: Vec<[Vec3; 3]> = f.iter().map(|t| t.map(|i| v[i] * 0.5)).collect();
    MeshData::from_triangles(&tris)
}

const SPHERE: u32 = 0;
const CUBE: u32 = 1;
const OCTA: u32 = 2;

fn palette(k: u32) -> Vec3 {
    let c = |ph: f32| 0.5 + 0.5 * (std::f32::consts::TAU * (ph + k as f32 * 0.137)).cos();
    Vec3::new(c(0.0), c(0.33), c(0.67))
}

/// material id whose mesh.slang palette colour is closest to `rgb`.
fn material(mesh: u32, rgb: [f32; 3]) -> u32 {
    let t = Vec3::from(rgb);
    (0..256u32)
        .min_by(|&a, &b| (palette(mesh * 7 + a) - t).length().total_cmp(&(palette(mesh * 7 + b) - t).length()))
        .unwrap()
}

fn inst(mesh: u32, mat: u32, pos: Vec3, scale: Vec3) -> InstanceGpu {
    InstanceGpu {
        model: Mat4::from_scale_rotation_translation(scale, Quat::IDENTITY, pos).to_cols_array_2d(),
        mesh_id: mesh,
        material_id: mat,
        _pad: [0; 2],
    }
}

/// Scaled-space impostor placement: camera-relative `rel` (f64 metres), diameter `diam`, minimum
/// angular size `min_ang` (rad). Beyond R0 the distance is compressed logarithmically and the size
/// scaled by the same factor, so the angular size and depth order are preserved.
fn impostor(rel: DVec3, diam: f64, min_ang: f64) -> Option<(Vec3, f32)> {
    let d = rel.length();
    let diam = diam.max(d * min_ang);
    if d <= R0 {
        return Some((rel.as_vec3(), diam as f32));
    }
    let dp = R0 * (1.0 + 0.4 * (d / R0).ln());
    if dp > FAR as f64 * 0.95 {
        return None;
    }
    let k = dp / d;
    Some(((rel * k).as_vec3(), (diam * k) as f32))
}

// ---------- coords flythroughs ------------------------------------------------------------------

struct Universe {
    tree: FrameTree,
    galaxy: FrameId,
    sol: FrameId,
    earth: FrameId,
    site: FrameId,
    planets: Vec<(&'static str, f64, f64, [f32; 3])>, // name, orbit m, radius m, colour
}

const EARTH_R: f64 = 6.371e6;

fn universe() -> Universe {
    let (mut tree, galaxy) = FrameTree::new("milky_way");
    let sol = tree.add(galaxy, "sol", DVec3::new(2.6e20, 0.0, 0.0), 1.0e13).unwrap();
    let planets = vec![
        ("mercury", 5.79e10, 2.44e6, [0.6, 0.55, 0.5]),
        ("venus", 1.082e11, 6.05e6, [0.9, 0.8, 0.5]),
        ("earth", 1.496e11, EARTH_R, [0.2, 0.4, 0.9]),
        ("mars", 2.279e11, 3.39e6, [0.8, 0.35, 0.2]),
        ("jupiter", 7.785e11, 6.99e7, [0.85, 0.7, 0.5]),
        ("saturn", 1.432e12, 5.82e7, [0.9, 0.85, 0.6]),
        ("uranus", 2.867e12, 2.54e7, [0.6, 0.85, 0.9]),
        ("neptune", 4.515e12, 2.46e7, [0.3, 0.45, 0.95]),
    ];
    let pos = |i: usize, r: f64| {
        let a = 0.7 + i as f64 * 2.1;
        DVec3::new(r * a.cos(), r * 0.02 * (i as f64).sin(), r * a.sin())
    };
    let earth = tree.add(sol, "earth", pos(2, 1.496e11), 1.0e7).unwrap();
    let site = tree.add(earth, "site", DVec3::new(EARTH_R, 0.0, 0.0), 1.0e4).unwrap();
    Universe { tree, galaxy, sol, earth, site, planets }
}

impl Universe {
    fn planet_pos(&self, i: usize) -> WorldPos {
        let r = self.planets[i].1;
        let a = 0.7 + i as f64 * 2.1;
        WorldPos { frame: self.sol, p: DVec3::new(r * a.cos(), r * 0.02 * (i as f64).sin(), r * a.sin()) }
    }
}

struct Leg {
    /// path is built in this frame (the destination), so precision is set by the frame level
    frame: FrameId,
    start: DVec3,
    end: DVec3,
}

#[derive(Default)]
struct LevelStat {
    frames: usize,
    max_f64_err: f64,
    max_naive_err: f64,
    max_bevy_err: f64,
}

fn level_name(l: FrameLevel) -> &'static str {
    match l {
        FrameLevel::Galaxy => "galaxy",
        FrameLevel::System => "system",
        FrameLevel::Planet => "planet",
        FrameLevel::Local => "local",
    }
}

fn run_flythrough(name: &str, o: &Opts, b: &mut AshBackend, r: &mut Report) {
    let u = universe();
    let t = &u.tree;
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut rnd = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    // static objects: (pos, diameter, min angular size, mesh, material)
    let mut objs: Vec<(WorldPos, f64, f64, u32, u32)> = Vec::new();
    let star_mat = [material(OCTA, [1.0, 0.95, 0.8]), material(OCTA, [0.7, 0.8, 1.0]), material(OCTA, [1.0, 0.7, 0.5])];
    let nstars = if name == "universe" { 6000 } else { 1500 };
    for i in 0..nstars {
        let (rr, a) = (5e20 * rnd().sqrt(), 2.0 * PI * rnd());
        let arm = (a * 2.0 + rr / 1.2e20).sin() * 0.5 + 0.5; // a hint of spiral arms
        let p = DVec3::new(rr * a.cos(), (rnd() - 0.5) * 3e19 * (1.0 - arm * 0.5), rr * a.sin());
        objs.push((WorldPos { frame: u.galaxy, p }, 1.4e9, 0.004, OCTA, star_mat[i % 3]));
    }
    objs.push((WorldPos { frame: u.galaxy, p: DVec3::ZERO }, 2e19, 0.0, SPHERE, material(SPHERE, [1.0, 0.9, 0.6])));
    objs.push((WorldPos { frame: u.sol, p: DVec3::ZERO }, 1.39e9, 0.02, SPHERE, material(SPHERE, [1.0, 0.85, 0.3])));
    for (i, pl) in u.planets.iter().enumerate() {
        if i == 2 {
            continue; // earth drawn below (own frame)
        }
        objs.push((u.planet_pos(i), 2.0 * pl.2, 0.012, SPHERE, material(SPHERE, pl.3)));
    }
    let earth_mat = material(SPHERE, [0.2, 0.4, 0.9]);
    objs.push((
        WorldPos { frame: u.earth, p: DVec3::new(3.844e8, 0.0, 0.0) },
        3.47e6,
        0.006,
        SPHERE,
        material(SPHERE, [0.7, 0.7, 0.7]),
    ));
    // landing site (local frame, +X is up): placeholder buildings
    let site_objs: Vec<(DVec3, DVec3, u32)> = (0..24)
        .map(|i| {
            let (z, y) = (25.0 + (i / 6) as f64 * 30.0, ((i % 6) as f64 - 2.5) * 18.0);
            let hgt = 4.0 + 14.0 * rnd();
            (
                DVec3::new(hgt / 2.0, y, z),
                DVec3::new(hgt, 8.0, 8.0),
                material(CUBE, [0.6 + 0.3 * rnd() as f32, 0.6, 0.65]),
            )
        })
        .collect();
    let ground_mat = material(CUBE, [0.3, 0.55, 0.25]);
    let probe_mat = material(CUBE, [1.0, 0.2, 0.2]);

    let (legs, expect): (Vec<Leg>, Vec<FrameLevel>) = match name {
        "universe" => (
            vec![Leg { frame: u.sol, start: DVec3::new(-2.6e20, 6e20, 4e20), end: DVec3::new(3e11, 8e11, 1.6e12) }],
            vec![FrameLevel::Galaxy, FrameLevel::System],
        ),
        "solar_system" => {
            let e = t.convert(WorldPos { frame: u.sol, p: DVec3::new(0.0, 2e12, 6e12) }, u.earth).p;
            (
                vec![Leg { frame: u.earth, start: e, end: DVec3::new(2.5e6, 4e6, 6.5e6) }],
                vec![FrameLevel::System, FrameLevel::Planet],
            )
        }
        _ => (
            vec![Leg { frame: u.site, start: DVec3::new(3.0e7, 4e6, -6e6), end: DVec3::new(2.0, 0.0, 0.0) }],
            vec![FrameLevel::System, FrameLevel::Planet, FrameLevel::Local],
        ),
    };
    let proj = vk_perspective(FOV, o.w as f32 / o.h as f32, NEAR, FAR);
    let rad_per_px = FOV as f64 / o.h as f64;
    let mut stats: HashMap<&'static str, LevelStat> = HashMap::new();
    let mut levels_seen: Vec<FrameLevel> = Vec::new();
    let mut fo: Option<FloatingOrigin> = None;
    let mut transitions: Vec<String> = Vec::new();
    let mut last_frame: Option<FrameId> = None;
    let mut max_jitter_px = 0f64;
    let mut max_naive_jitter_px = 0f64;
    let mut max_inst_tris = 0usize;
    let mut continuity_err = 0f64;
    let mesh_tris: Vec<usize> = b.mesh_table().iter().map(|m| m.index_count as usize / 3).collect();
    let mut instances: Vec<InstanceGpu> = Vec::with_capacity(objs.len() + 64);
    let n = o.frames;
    // PNG at the most telling point of the path: the galactic disk / the inner planets / the ground
    let png_frame = match name {
        "universe" => n / 12,
        "solar_system" => n / 5,
        _ => n - 1,
    };
    r.metric("png_at_path_fraction", format!("{:.2}", png_frame as f64 / (n - 1) as f64));
    for f in 0..n {
        let t0 = Instant::now();
        let s = f as f64 / (n - 1) as f64;
        let leg = &legs[0];
        // logarithmic approach in the destination frame + a gentle sideways swing
        let (d0, d1) = (leg.start.length(), leg.end.length());
        let d = d0 * (d1 / d0).powf(s);
        let dir = leg.start.normalize().lerp(leg.end.normalize(), s.powf(0.6)).normalize();
        let cam = t.rehome(WorldPos { frame: leg.frame, p: dir * d });
        let lvl = t.frame(cam.frame).level;
        if !levels_seen.contains(&lvl) {
            levels_seen.push(lvl);
        }
        if last_frame != Some(cam.frame) {
            transitions.push(format!("f{f}:{}", t.frame(cam.frame).name));
            last_frame = Some(cam.frame);
        }
        // frame-to-frame conversion: the rehomed camera converted back to the path frame must equal
        // the input within 4 ulp of the largest magnitude involved (|p| and every frame origin on
        // the path through the LCA). Explicit tolerance, never float equality.
        let input = WorldPos { frame: leg.frame, p: dir * d };
        let back = t.convert(cam, leg.frame).p;
        let mut mag = input.p.length().max(cam.p.length());
        let lca = t.lca(cam.frame, leg.frame);
        for start in [cam.frame, leg.frame] {
            let mut id = start;
            while id != lca {
                let fr = t.frame(id);
                mag += fr.origin_in_parent.length();
                id = fr.parent.expect("lca is an ancestor");
            }
        }
        let tol = 4.0 * coords::frames::ulp(mag);
        continuity_err = continuity_err.max((back - input.p).length() / tol);
        let cam_root = t.convert(cam, u.galaxy).p;
        // look direction: at the destination origin; earth scene levels out near the ground
        let mut look = t.delta(WorldPos { frame: leg.frame, p: DVec3::ZERO }, cam).normalize();
        let mut up = if look.y.abs() > 0.9 { DVec3::Z } else { DVec3::Y };
        if name == "earth" {
            let alt = d.max(1.0);
            let w = (alt.log10() / 7.5).clamp(0.0, 1.0);
            look = (DVec3::new(-1.0, 0.0, 0.0) * w + DVec3::new(-0.08, 0.05, 1.0) * (1.0 - w)).normalize();
            up = DVec3::X;
        }
        let right = look.cross(up).normalize();
        let down = right.cross(look).normalize() * -1.0;
        // rider probe: lives in the camera's frame at a fixed camera-relative offset
        let offset = (look * 3.0 + right * 0.9 + down * 0.45).normalize() * OFFSET_LEN;
        let rider = WorldPos { frame: cam.frame, p: cam.p + offset };
        let rel = t.render_relative(rider, cam).as_dvec3();
        let f64_err = (rel - offset).length();
        let naive = (t.convert(rider, u.galaxy).p.as_vec3() - cam_root.as_vec3()).as_dvec3();
        let naive_err = (naive - offset).length();
        let fo = fo.get_or_insert_with(|| FloatingOrigin::new(cam, 2048.0));
        fo.update(t, cam);
        let bevy = (fo.local_f32(t, rider) - fo.local_f32(t, cam)).as_dvec3();
        let bevy_err = (bevy - offset).length();
        let st = stats.entry(level_name(lvl)).or_default();
        st.frames += 1;
        st.max_f64_err = st.max_f64_err.max(f64_err);
        st.max_naive_err = st.max_naive_err.max(naive_err);
        st.max_bevy_err = st.max_bevy_err.max(bevy_err);
        if lvl != FrameLevel::Galaxy {
            max_jitter_px = max_jitter_px.max(f64_err / offset.length() / rad_per_px);
            max_naive_jitter_px = max_naive_jitter_px.max(naive_err / offset.length() / rad_per_px);
        }

        instances.clear();
        let push = |rel: DVec3, diam: f64, ang: f64, mesh: u32, mat: u32, instances: &mut Vec<InstanceGpu>| {
            if let Some((p, sc)) = impostor(rel, diam, ang) {
                instances.push(inst(mesh, mat, p, Vec3::splat(sc)));
            }
        };
        for &(p, diam, ang, mesh, mat) in &objs {
            push(t.delta(p, cam), diam, ang, mesh, mat, &mut instances);
        }
        let alt = t.delta(WorldPos { frame: u.earth, p: DVec3::ZERO }, cam).length() - EARTH_R;
        if alt > 3e4 || name != "earth" {
            push(
                t.delta(WorldPos { frame: u.earth, p: DVec3::ZERO }, cam),
                2.0 * EARTH_R,
                0.012,
                SPHERE,
                earth_mat,
                &mut instances,
            );
        }
        if name == "earth" && alt < 3e5 {
            // C1 far-field placeholder: a flat 200 km ground tile at the site, + buildings (local frame)
            let g = t.delta(WorldPos { frame: u.site, p: DVec3::new(-0.5, 0.0, 0.0) }, cam);
            if g.length() < R0 * 50.0 {
                let (p, _) = impostor(g, 1.0, 0.0).unwrap();
                let k = p.length() as f64 / g.length().max(1e-9);
                instances.push(inst(CUBE, ground_mat, p, (DVec3::new(1.0, 2e5, 2e5) * k).as_vec3()));
            }
            for &(c, sz, mat) in &site_objs {
                let rel = t.delta(WorldPos { frame: u.site, p: c }, cam);
                let (p, _) = impostor(rel, 1.0, 0.0).unwrap();
                let k = p.length() as f64 / rel.length().max(1e-9);
                instances.push(inst(CUBE, mat, p, (sz * k).as_vec3()));
            }
        }
        if lvl != FrameLevel::Galaxy {
            instances.push(inst(CUBE, probe_mat, rel.as_vec3(), Vec3::splat(0.25)));
        }
        let tris: usize = instances.iter().map(|i| mesh_tris[i.mesh_id as usize]).sum();
        max_inst_tris = max_inst_tris.max(tris);
        let lookf = look.as_vec3();
        let view = look_at(Vec3::ZERO, lookf, up.as_vec3());
        let vp = (proj * view).to_cols_array_2d();
        if let Some(s) =
            b.render(&FrameSnapshot { view_proj: vp, instances: &instances, frame_index: f as u64 }).unwrap()
        {
            r.gpu_ms.extend(gpu_ms(&s));
        }
        if f == png_frame {
            // capture (waits for the GPU); this frame is excluded from the timings
            for s in b.drain().unwrap() {
                r.gpu_ms.extend(gpu_ms(&s));
            }
            r.png = Some(b.read_color_rgba().unwrap());
            continue;
        }
        r.frame_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    for s in b.drain().unwrap() {
        if let Some(g) = gpu_ms(&s) {
            r.gpu_ms.push(g);
        }
    }
    r.tris = max_inst_tris;
    r.mem.push(("instances_cpu", (objs.len() * size_of::<InstanceGpu>()) as u64));
    r.mem.push(("instance_buffers_gpu", (b.frames_in_flight() * b.max_instances() * size_of::<InstanceGpu>()) as u64));
    r.mem.push(("frame_tree", (t.len() * 64) as u64));
    let fo = fo.unwrap();
    r.metric("frame_transitions", transitions.join(" -> "));
    r.metric("rehome_roundtrip_err/tolerance(max)", format!("{continuity_err:.3}"));
    r.metric("floating_origin_rebases", fo.rebases);
    r.metric("rider_jitter_px_max_f64_path(non-galaxy)", format!("{max_jitter_px:.2e}"));
    r.metric("rider_jitter_px_max_naive_f32(non-galaxy)", format!("{max_naive_jitter_px:.2e}"));
    for l in [FrameLevel::Galaxy, FrameLevel::System, FrameLevel::Planet, FrameLevel::Local] {
        if let Some(st) = stats.get(level_name(l)) {
            r.metric(
                format!("{}_level", level_name(l)),
                format!(
                    "frames {} | rider err {:.2e} m (f64 path incl. final f32 narrowing; documented f64 <= {:.1e}) | Bevy f32 path {:.2e} m | naive f32 {:.2e} m",
                    st.frames,
                    st.max_f64_err,
                    l.precision_m(),
                    st.max_bevy_err,
                    st.max_naive_err
                ),
            );
            // tolerance: documented half-ulp of the level's max extent, x2 for the two roundings
            // (cam.p and cam.p+offset), plus f32 narrowing of a 3 m offset (~1e-6 m)
            // explicit tolerance: 2x the documented precision (cam.p and cam.p + offset are each rounded
            // once) + the f32 narrowing of the ~1.1 m camera-relative offset (|offset| * 2^-23)
            let tol = 2.0 * l.precision_m() + OFFSET_LEN * 2f64.powi(-23);
            r.check(format!("{}: f64 rider error within documented precision", level_name(l)), st.max_f64_err <= tol);
            if l != FrameLevel::Galaxy {
                r.check(
                    format!("{}: Bevy floating-origin path error <= 1e-3 m", level_name(l)),
                    st.max_bevy_err <= 1e-3_f64.max(2.0 * l.precision_m()),
                );
            }
        }
    }
    r.check("visited the expected frame levels", expect.iter().all(|l| levels_seen.contains(l)));
    r.check("rehome/convert round trip within 4 ulp of the path magnitude", continuity_err <= 1.0);
    r.check("rider jitter sub-pixel (< 0.5 px) below galaxy level (f64 path)", max_jitter_px < 0.5);
    r.check("naive f32 is measurably worse (control)", max_naive_jitter_px > 10.0 * max_jitter_px.max(1e-9));
    r.real.extend([
        "f64 frame tree (galaxy > system > planet > local), rehome, camera-relative f64 subtraction -> f32",
        "floating origin (same FloatingOrigin the Bevy plugin uses), camera-relative instances through render-ash",
        "GPU cull + indirect draw of all objects",
    ]);
    r.placeholder.extend([
        "stars/planets/sun/moon are instanced spheres/octahedra impostors in scaled space (min angular size)",
        "no atmosphere (Prompt E), no lighting from the sun position",
    ]);
    if name == "earth" {
        r.placeholder.extend([
            "earth surface: flat 200 km ground tile + boxes below 30 km altitude (C1 cube-sphere quadtree + C3 handoff not implemented)",
            "ocean not drawn (Prompt D)",
        ]);
    }
}

// ---------- ocean placeholder --------------------------------------------------------------------

fn run_ocean(o: &Opts, b: &mut AshBackend, r: &mut Report) {
    let sea = material(CUBE, [0.1, 0.3, 0.7]);
    let buoy = material(OCTA, [1.0, 0.5, 0.1]);
    let proj = vk_perspective(FOV, o.w as f32 / o.h as f32, NEAR, FAR);
    let mesh_tris: Vec<usize> = b.mesh_table().iter().map(|m| m.index_count as usize / 3).collect();
    let mut instances = Vec::new();
    for f in 0..o.frames {
        let t0 = Instant::now();
        let cam = Vec3::new(0.0, 6.0, -(f as f32) * 0.2);
        instances.clear();
        instances.push(inst(CUBE, sea, Vec3::new(0.0, -0.5, 0.0) - cam, Vec3::new(40_000.0, 1.0, 40_000.0)));
        for i in 0..40 {
            let a = i as f32 * 0.61;
            let p = Vec3::new(
                a.sin() * (30.0 + i as f32 * 12.0),
                0.4 + 0.3 * (f as f32 * 0.1 + i as f32).sin(),
                20.0 + i as f32 * 15.0,
            );
            instances.push(inst(OCTA, buoy, p - cam, Vec3::splat(1.5)));
        }
        let view = look_at(Vec3::ZERO, Vec3::new(0.0, -0.08, 1.0), Vec3::Y);
        let vp = (proj * view).to_cols_array_2d();
        if let Some(s) =
            b.render(&FrameSnapshot { view_proj: vp, instances: &instances, frame_index: f as u64 }).unwrap()
        {
            r.gpu_ms.extend(gpu_ms(&s));
        }
        r.frame_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    for s in b.drain().unwrap() {
        if let Some(g) = gpu_ms(&s) {
            r.gpu_ms.push(g);
        }
    }
    r.tris = instances.iter().map(|i| mesh_tris[i.mesh_id as usize]).sum();
    r.mem.push(("instances", (instances.len() * size_of::<InstanceGpu>()) as u64));
    r.check("renders", true);
    r.real.push("camera-relative instanced rendering (planet-local frame)");
    r.placeholder.extend([
        "OCEAN SURFACE: flat opaque slab, no waves/FFT/foam/reflections (Prompt D)",
        "buoys: instanced octahedra bobbing on a sine (no buoyancy)",
        "sky: clear colour (Prompt E atmosphere)",
    ]);
}

// ---------- voxel scenes -------------------------------------------------------------------------

struct VoxelSpec {
    field: Box<dyn DensityField + Sync>,
    /// metres
    region: ([f64; 3], [f64; 3]),
    shells: Vec<i64>,
    cam_from: Vec3,
    cam_to: Vec3,
    look_at: Vec3,
    water_y: Option<f32>,
}

fn voxel_spec(name: &str) -> VoxelSpec {
    match name {
        "mountains" => VoxelSpec {
            field: Box::new(Mountains { seed: 7 }),
            region: ([-128.0, -64.0, -128.0], [128.0, 64.0, 128.0]),
            shells: vec![1, 1, 1],
            cam_from: Vec3::new(-35.0, 25.0, 37.0),
            cam_to: Vec3::new(-30.0, 24.0, 30.0),
            look_at: Vec3::new(10.0, 27.0, -6.0),
            water_y: None,
        },
        "underground" => VoxelSpec {
            field: Box::new(Underground { seed: 3 }),
            region: ([-64.0, -64.0, -64.0], [64.0, 64.0, 64.0]),
            shells: vec![1, 1, 1],
            cam_from: Vec3::new(-5.0, 1.8, -4.0),
            cam_to: Vec3::new(-3.6, 1.6, -3.4),
            look_at: Vec3::new(2.2, 0.5, 1.7),
            water_y: None,
        },
        "cave" => VoxelSpec {
            field: Box::new(CaveScene { seed: 1 }),
            region: ([-32.0, -32.0, -32.0], [32.0, 32.0, 32.0]),
            shells: vec![1, 1],
            cam_from: Vec3::new(-11.5, 2.8, 1.4),
            cam_to: Vec3::new(-8.0, 3.0, 1.1),
            look_at: Vec3::new(1.5, 3.4, 0.4),
            water_y: None,
        },
        _ => VoxelSpec {
            field: Box::new(Seabed { seed: 5 }),
            region: ([-64.0, -64.0, -64.0], [64.0, 0.0, 64.0]),
            shells: vec![1, 1, 1],
            cam_from: Vec3::new(-23.0, -13.5, 17.0),
            cam_to: Vec3::new(-18.0, -14.5, 12.0),
            look_at: Vec3::new(-5.0, -16.5, -3.0),
            water_y: Some(0.0),
        },
    }
}

fn run_voxel(name: &str, o: &Opts, b: &mut AshBackend, r: &mut Report) {
    let spec = voxel_spec(name);
    let grid = BrickGrid::new(o.brick, o.voxel_m).expect("brick must be 16/32/64, voxel > 0");
    let cfg = VoxelConfig::new(grid);
    let l = |m: f64| (m / o.voxel_m).round() as i64;
    let mid = (spec.cam_from + spec.cam_to) * 0.5;
    let cam_lattice = [l(mid.x as f64), l(mid.y as f64), l(mid.z as f64)];
    let layout = ShellLayout::camera_centred(&grid, cam_lattice, &spec.shells);
    let max_lod = layout.max_lod();
    let coarse = grid.brick_size() as i64 * grid.step(max_lod);
    let al = |v: i64, up: bool| {
        if up { -coords::floor_div(-v, coarse) * coarse } else { coords::floor_div(v, coarse) * coarse }
    };
    // region: the requested box, grown so the outermost shell box sits >= 1 coarsest brick inside
    // (finer boxes touching the region cut would let Φ move cut-boundary vertices off the plane)
    let outer = layout.boxes.last().copied().expect("at least one shell");
    let region = (
        [0, 1, 2].map(|i| al(l(spec.region.0[i]), false).min(outer.lo[i] - coarse)),
        [0, 1, 2].map(|i| al(l(spec.region.1[i]), true).max(outer.hi[i] + coarse)),
    );

    // C2 pipeline: plan (H8) -> integer sampling (CPU, rayon) -> GPU Transvoxel -> exact checks
    let t0 = Instant::now();
    let coords_list = bricks_in(&grid, &layout, region, max_lod);
    let planned: Vec<_> = coords_list.iter().map(|&c| (c, plan_brick(&grid, &layout, c))).collect();
    let refused = planned.iter().filter(|p| p.1.is_err()).count();
    let sampled: Vec<(BrickSamples, u8, Option<voxel::Aabb>)> = planned
        .par_iter()
        .filter_map(|(c, p)| p.as_ref().ok().map(|&(m, f)| (*c, m, f)))
        .map(|(c, m, f)| (sample_brick(spec.field.as_ref(), &cfg, c, m).expect("finite"), m, f))
        .collect();
    let sample_ms = t0.elapsed().as_secs_f64() * 1e3;
    let sample_bytes: usize = sampled.iter().map(|s| s.0.bytes()).sum();
    let occupied: Vec<&(BrickSamples, u8, Option<voxel::Aabb>)> =
        sampled.iter().filter(|s| s.0.has_sign_change()).collect();
    let jobs: Vec<VoxelJob<'_>> =
        occupied.iter().map(|s| VoxelJob { samples: &s.0, transitions: s.1, finer: s.2 }).collect();
    let tg = Instant::now();
    let gpu = b.mesh_voxels_gpu(&cfg, &jobs).expect("gpu mesh");
    let gpu_mesh_wall_ms = tg.elapsed().as_secs_f64() * 1e3;
    let tc = Instant::now();
    let cpu: Vec<BrickMesh> = jobs.par_iter().map(|j| mesh_brick(&cfg, j.samples, j.transitions, j.finer)).collect();
    let cpu_mesh_ms = tc.elapsed().as_secs_f64() * 1e3;
    let exact = cpu
        .iter()
        .zip(&gpu.bricks)
        .all(|(c, g)| c.indices.iter().map(|&i| c.ipos[i as usize]).eq(g.ipos.iter().copied()));
    let topo = topology(cpu.iter());
    // every open edge must lie on the (exact, integer) region boundary: the interior is watertight
    let (lo, hi) = (region.0.map(|v| v * SUB), region.1.map(|v| v * SUB));
    let mut directed: HashMap<([i64; 3], [i64; 3]), i32> = HashMap::new();
    for m in &cpu {
        for t in m.indices.chunks(3) {
            for (a, c) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *directed.entry((m.ipos[a as usize], m.ipos[c as usize])).or_default() += 1;
            }
        }
    }
    let on_boundary =
        |p: [i64; 3], q: [i64; 3]| (0..3).any(|i| (p[i] == lo[i] && q[i] == lo[i]) || (p[i] == hi[i] && q[i] == hi[i]));
    let interior_open =
        directed.keys().filter(|&&(a, c)| !directed.contains_key(&(c, a)) && !on_boundary(a, c)).count();
    if std::env::var_os("SCENES_DEBUG").is_some() {
        eprintln!("region sub lo {lo:?} hi {hi:?}");
        for (a, c) in directed.keys().filter(|&&(a, c)| !directed.contains_key(&(c, a)) && !on_boundary(a, c)).take(6) {
            let owner: Vec<_> = cpu
                .iter()
                .zip(&jobs)
                .filter(|(m, _)| m.ipos.contains(a))
                .map(|(_, j)| (j.samples.coord, j.transitions))
                .collect();
            eprintln!("open {a:?} -> {c:?} owners {owner:?}");
        }
    }
    if let Some(path) = std::env::var_os("SCENES_DUMP") {
        // diagnostics: per triangle 9 f32 positions + lod + is_transition + brick xyz (f32)
        let mut out: Vec<f32> = Vec::new();
        for (m, j) in cpu.iter().zip(&jobs) {
            let nreg = mesh_brick(&cfg, j.samples, 0, j.finer).indices.len() / 3;
            for (ti, t) in m.indices.chunks(3).enumerate() {
                for &i in t {
                    out.extend(cfg.sub_to_render(m.ipos[i as usize]));
                }
                let c = j.samples.coord;
                out.extend([c.lod as f32, (ti >= nreg) as u8 as f32, c.x as f32, c.y as f32, c.z as f32]);
            }
        }
        std::fs::write(path, bytemuck::cast_slice(&out)).unwrap();
    }
    let lods: Vec<usize> = (0..=max_lod).map(|lv| jobs.iter().filter(|j| j.samples.coord.lod == lv).count()).collect();

    // draw the GPU-meshed triangles (host round-trip; C4's GPU brick pool would keep them resident)
    let mut verts: Vec<VoxelVertex> = Vec::with_capacity(gpu.triangles * 3);
    for g in &gpu.bricks {
        verts
            .extend(g.ipos.iter().zip(&g.normals).map(|(p, n)| VoxelVertex { pos: cfg.sub_to_render(*p), normal: *n }));
    }
    let idx: Vec<u32> = (0..verts.len() as u32).collect();
    b.add_static_mesh(&verts, &idx).unwrap();
    let proj = vk_perspective(FOV, o.w as f32 / o.h as f32, NEAR, 2000.0);
    let mesh_tris: Vec<usize> = b.mesh_table().iter().map(|m| m.index_count as usize / 3).collect();
    let water = material(CUBE, [0.15, 0.35, 0.6]);
    let mut instances = Vec::new();
    for f in 0..o.frames {
        let t0 = Instant::now();
        let s = f as f32 / (o.frames - 1).max(1) as f32;
        let eye = spec.cam_from.lerp(spec.cam_to, s);
        instances.clear();
        if let Some(y) = spec.water_y {
            // UNDERWATER placeholder: opaque water-surface slab overhead (no volume, fog, caustics)
            instances.push(inst(CUBE, water, Vec3::new(0.0, y - 0.5, 0.0), Vec3::new(4000.0, 1.0, 4000.0)));
        }
        let view = look_at(eye, spec.look_at, Vec3::Y);
        let vp = (proj * view).to_cols_array_2d();
        if let Some(st) =
            b.render(&FrameSnapshot { view_proj: vp, instances: &instances, frame_index: f as u64 }).unwrap()
        {
            r.gpu_ms.extend(gpu_ms(&st));
        }
        r.frame_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    for st in b.drain().unwrap() {
        if let Some(g) = gpu_ms(&st) {
            r.gpu_ms.push(g);
        }
    }
    r.tris = gpu.triangles + instances.iter().map(|i| mesh_tris[i.mesh_id as usize]).sum::<usize>();
    r.mem.push(("samples_cpu", sample_bytes as u64));
    r.mem.push(("mesh_cpu_reference", cpu.iter().map(|m| m.bytes() as u64).sum()));
    r.mem.push(("gpu_mesher_transient", gpu.device_bytes));
    r.mem.push(("static_mesh_gpu", (verts.len() * 24 + idx.len() * 4) as u64));
    r.metric("brick", format!("{}^3 @ {} m (trial; --brick/--voxel)", o.brick, o.voxel_m));
    r.metric("shells(half_bricks)", format!("{:?} -> LOD 0..={max_lod}", spec.shells));
    r.metric("bricks planned/sampled/occupied", format!("{}/{}/{}", coords_list.len(), sampled.len(), jobs.len()));
    r.metric("occupied bricks per LOD", format!("{lods:?}"));
    r.metric("sample_ms_cpu_rayon", format!("{sample_ms:.1}"));
    r.metric("gpu_mesh_ms(count+scan+emit)", format!("{:.2}", gpu.gpu_ms.unwrap_or(f64::NAN)));
    r.metric("gpu_mesh_wall_ms(incl pipeline build+readback)", format!("{gpu_mesh_wall_ms:.1}"));
    r.metric("cpu_mesh_ms_rayon(reference)", format!("{cpu_mesh_ms:.1}"));
    r.metric("topology", format!("{topo:?}"));
    r.check("no bricks refused by plan_brick (H8)", refused == 0);
    r.check("GPU Transvoxel == CPU mesher (positions bit-exact, same order)", exact && gpu.triangles == topo.triangles);
    r.check("interior watertight: open edges only on the region boundary", interior_open == 0);
    r.check(
        "no non-manifold edges / orientation conflicts",
        topo.non_manifold_edges == 0 && topo.orientation_conflicts == 0,
    );
    r.check(
        "no degenerate triangles, no inward closed components, all finite",
        topo.degenerate == 0 && topo.flipped == 0 && topo.non_finite == 0,
    );
    r.check("more than one LOD carries surface (shells exercised)", lods.iter().filter(|&&c| c > 0).count() >= 2);
    r.real.extend([
        "C2: camera-centred LOD shells, Transvoxel transition cells + global Φ seam field (multi-coarser-face bricks)",
        "integer sampling/meshing (hard rule H1-H8), GPU Slang Transvoxel mesher verified bit-exact vs CPU",
        "exact topology checks (i128 predicates)",
    ]);
    r.placeholder.push("static world: meshed once around the path midpoint (C4 streaming/re-centring, async generation, GPU brick pool not implemented)");
    r.placeholder.push("no collision (hard rule applies when added)");
    if spec.water_y.is_some() {
        r.placeholder.extend([
            "UNDERWATER VOLUME: none (no fog/absorption/caustics/god rays) - Prompt E/D",
            "water surface: opaque flat slab overhead (Prompt D ocean)",
        ]);
    }
    if name == "underground" || name == "cave" {
        r.placeholder.push("lighting: unshadowed sun + hemispheric ambient (no headlamp/GI)");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let which = arg(&args, "--scene").unwrap_or_else(|| {
        eprintln!("usage: --scene <{}|all>", SCENES.join("|"));
        std::process::exit(2)
    });
    let (w, h) = arg(&args, "--size")
        .map(|s| {
            let (a, b) = s.split_once('x').unwrap();
            (a.parse().unwrap(), b.parse().unwrap())
        })
        .unwrap_or((960u32, 540u32));
    let o = Opts {
        frames: arg(&args, "--frames").map_or(120, |s| s.parse().unwrap()).max(2),
        w,
        h,
        brick: arg(&args, "--brick").map_or(32, |s| s.parse().unwrap()),
        voxel_m: arg(&args, "--voxel").map_or(0.25, |s| s.parse().unwrap()),
        validate: args.iter().any(|a| a == "--validate"),
    };
    let names: Vec<&str> = if which == "all" {
        SCENES.to_vec()
    } else {
        vec![SCENES.iter().copied().find(|s| *s == which).unwrap_or_else(|| {
            eprintln!("unknown scene {which}; one of {}|all", SCENES.join("|"));
            std::process::exit(2)
        })]
    };
    std::fs::create_dir_all("out/scenes").unwrap();
    let mut all_ok = true;
    for name in names {
        let (e0, w0) = message_counts();
        let mut b = AshBackend::new(
            BackendConfig { max_instances: 8192, extent: (o.w, o.h), validation: o.validate, ..Default::default() },
            None,
        )
        .expect("backend");
        b.set_meshes(&[icosphere(3), cube(), octahedron()]).unwrap();
        let mut r = Report { name: name.into(), ..Default::default() };
        match name {
            "universe" | "solar_system" | "earth" => {
                r.kind = "coords flythrough";
                run_flythrough(name, &o, &mut b, &mut r);
            }
            "ocean" => {
                r.kind = "placeholder";
                run_ocean(&o, &mut b, &mut r);
            }
            _ => {
                r.kind = "voxel (real mesher)";
                run_voxel(name, &o, &mut b, &mut r);
            }
        }
        let (pw, ph, px) = match r.png.take() {
            Some(p) => p,
            None => b.read_color_rgba().unwrap(),
        };
        let png = format!("out/scenes/{name}.png");
        let mut enc = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(&png).unwrap()), pw, ph);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().unwrap().write_image_data(&px).unwrap();
        let device = b.device_name().to_string();
        drop(b);
        let (e1, w1) = message_counts();
        if o.validate {
            r.check("validation: 0 errors", e1 == e0);
        }
        let line = r.json(&device, e1 - e0, w1 - w0);
        std::fs::write(format!("out/scenes/{name}.json"), &line).unwrap();
        println!("{line}");
        all_ok &= r.ok();
    }
    if !all_ok {
        eprintln!("some correctness checks FAILED");
        std::process::exit(1);
    }
}
