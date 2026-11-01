//! Visual quality pass: `cargo run -p render-ash --release --example quality -- --scene cave|mountains`
//! Writes out/quality/<scene>.png (1920x1080, 2x2 supersampled) and out/quality/<scene>.json.
//!
//! Pipeline: LOD shells around the camera -> LOD-aware band-limited integer sampling (CPU,
//! rayon; bricks provably above/below the terrain skipped) -> GPU Transvoxel (bit-exact with
//! the CPU) -> weld + island removal -> per-vertex analytic-gradient normals, multi-scale SDF AO,
//! cavity/convexity -> (mountains) far heightfield rings to ~24 km -> quality.slang (cascaded
//! sun shadows / headlamp spot shadow, PBR-ish triplanar materials, fog, sky, ACES).
#[path = "quality_parts/veg.rs"]
mod veg;

use std::collections::HashMap;
use std::time::Instant;

use coords::{BrickCoord, BrickGrid};
use glam::{DVec3, Mat4, Vec3};
use rayon::prelude::*;
use render_ash::quality::{InstancedSet, QParamsGpu, QVertex, QualityFrame, ShadowView, reverse_z_infinite, sun_cascade, sun_cascade_clip};
use render_ash::validation::message_counts;
use render_ash::voxel_gpu::VoxelJob;
use render_ash::{AshBackend, BackendConfig};
use voxel::fields::{Alpine, CaveNetwork, cave_path};
use voxel::{BrickMesh, DensityField, SUB, ShellLayout, VoxelConfig, bricks_in, plan_brick, remove_islands, sample_brick_filtered, sample_brick_smooth, filter_level};

const W: u32 = 1920;
const H: u32 = 1080;
const SS: u32 = 2;
const FOV: f32 = 0.82;

#[derive(Default)]
struct Stats {
    bricks_planned: usize,
    bricks_skipped: usize,
    bricks_sampled: usize,
    bricks_meshed: usize,
    sample_ms: f64,
    gpu_mesh_ms: f64,
    attr_ms: f64,
    voxel_tris: usize,
    ring_tris: usize,
    islands_removed: usize,
    island_tris_removed: usize,
    interior_open_edges: usize,
    topo_clean: bool,
    lods: Vec<(u8, usize)>,
}

/// Opt-in TurboQuant study hook (`BRUTAL_TQ`, unset = off = the default render): `<b>` = TQ b-bit
/// lossy samples decoded on the GPU (`tq.slang`), `<b>h` = TQ hybrid with exact face planes,
/// `band` = the mesh-exact narrow band (+-1 voxel). See starter.md "Learning from turbovec and TurboQuant".
#[derive(Clone, Copy, PartialEq)]
enum TqMode {
    Off,
    Tq(u32),
    Hybrid(u32),
    Band,
}

fn tq_mode() -> TqMode {
    match std::env::var("BRUTAL_TQ").as_deref() {
        Err(_) | Ok("") | Ok("0") | Ok("off") => TqMode::Off,
        Ok("band") => TqMode::Band,
        Ok(v) => {
            let (num, h) = v.strip_suffix('h').map_or((v, false), |n| (n, true));
            match num.parse::<u32>() {
                Ok(b @ 1..=4) if h => TqMode::Hybrid(b),
                Ok(b @ 1..=4) => TqMode::Tq(b),
                _ => panic!("BRUTAL_TQ: expected 1..4, 1h..4h or band, got {v:?}"),
            }
        }
    }
}

struct Mesh {
    world: Vec<DVec3>,
    level_cell: Vec<f64>,
    flags: Vec<f32>,
    indices: Vec<u32>,
}

/// Voxel near field: returns welded vertices (world f64 + filter cell) and indices.
#[allow(clippy::too_many_arguments)]
fn voxel_near(
    b: &mut AshBackend,
    field: &dyn DensityField,
    cfg: &VoxelConfig,
    layout: &ShellLayout,
    region: ([i64; 3], [i64; 3]),
    skip: &(dyn Fn(BrickCoord) -> bool + Sync),
    min_island: usize,
    smooth: bool,
    st: &mut Stats,
) -> Mesh {
    let grid = &cfg.grid;
    let all = bricks_in(grid, layout, region, layout.max_lod());
    st.bricks_planned = all.len();
    let todo: Vec<BrickCoord> = all.into_par_iter().filter(|&c| !skip(c)).collect();
    st.bricks_skipped = st.bricks_planned - todo.len();
    st.bricks_sampled = todo.len();
    let mut soup: Vec<[i64; 3]> = Vec::new();
    let mut lod_tris: HashMap<u8, usize> = HashMap::new();
    for chunk in todo.chunks(384) {
        let t0 = Instant::now();
        let mut smp: Vec<_> = chunk
            .par_iter()
            .filter_map(|&c| {
                let (mask, finer) = plan_brick(grid, layout, c).expect("H8 layout");
                // cave: per-shell level filter (as approved); mountains: the continuous filter
                let s = if smooth { sample_brick_smooth(field, cfg, layout, c, mask) } else { sample_brick_filtered(field, cfg, layout, c, mask) }.expect("finite");
                s.has_sign_change().then_some((s, mask, finer))
            })
            .collect();
        st.sample_ms += t0.elapsed().as_secs_f64() * 1e3;
        let tq = tq_mode();
        if tq != TqMode::Off && !smp.is_empty() {
            let rot = asset::tq::Rotation::new();
            match tq {
                TqMode::Tq(bits) => {
                    let blobs: Vec<Vec<u8>> = smp.par_iter().map(|s| asset::tq::encode(&s.0, bits, &rot)).collect();
                    let refs: Vec<&[u8]> = blobs.iter().map(Vec::as_slice).collect();
                    let dec = b.decode_tq_gpu(&refs, &rot).expect("tq gpu decode");
                    let m = smp[0].0.d.len();
                    for (k, s) in smp.iter_mut().enumerate() {
                        s.0.d.copy_from_slice(&dec.samples[k * m..(k + 1) * m]);
                    }
                    eprintln!("BRUTAL_TQ {bits}: {} bricks, {} KiB blobs (raw {} KiB), GPU decode {:.2} ms (lavapipe)", smp.len(), dec.upload_bytes / 1024, dec.raw_bytes / 1024, dec.gpu_ms.unwrap_or(f64::NAN));
                }
                TqMode::Hybrid(bits) => smp.par_iter_mut().for_each(|s| {
                    let blob = asset::tq::encode(&s.0, bits, &rot);
                    s.0.d = asset::tq::decode_hybrid(&asset::tq::TqView::parse(&blob).unwrap(), &rot, &s.0.d);
                }),
                TqMode::Band => smp.par_iter_mut().for_each(|s| s.0.d = asset::tq::mesh_band(&s.0, s.1, 1024)),
                TqMode::Off => {}
            }
        }
        let jobs: Vec<VoxelJob<'_>> = smp.iter().map(|s| VoxelJob { samples: &s.0, transitions: s.1, finer: s.2 }).collect();
        let t1 = Instant::now();
        let g = b.mesh_voxels_gpu(cfg, &jobs).expect("gpu mesh");
        st.gpu_mesh_ms += t1.elapsed().as_secs_f64() * 1e3;
        for (bt, j) in g.bricks.iter().zip(&jobs) {
            if !bt.ipos.is_empty() {
                st.bricks_meshed += 1;
                *lod_tris.entry(j.samples.coord.lod).or_default() += bt.ipos.len() / 3;
            }
            soup.extend_from_slice(&bt.ipos);
        }
    }
    st.lods = lod_tris.into_iter().collect();
    st.lods.sort();
    // weld by exact integer position
    let mut m = BrickMesh::default();
    let mut weld: HashMap<[i64; 3], u32> = HashMap::with_capacity(soup.len() / 4);
    for p in soup {
        let n = m.ipos.len() as u32;
        let i = *weld.entry(p).or_insert_with(|| {
            m.ipos.push(p);
            n
        });
        m.indices.push(i);
    }
    m.vertices = vec![Default::default(); m.ipos.len()];
    let (lo, hi) = (region.0.map(|v| v * SUB), region.1.map(|v| v * SUB));
    let mut directed: HashMap<(u32, u32), i32> = HashMap::new();
    for t in m.indices.chunks(3) {
        for (a, c) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            *directed.entry((a, c)).or_default() += 1;
        }
    }
    let on_boundary = |p: [i64; 3], q: [i64; 3]| (0..3).any(|i| (p[i] == lo[i] && q[i] == lo[i]) || (p[i] == hi[i] && q[i] == hi[i]));
    // manifold + consistently oriented: every directed edge used once, no degenerate triangles
    let degenerate = m.indices.chunks(3).filter(|t| t[0] == t[1] || t[1] == t[2] || t[0] == t[2]).count();
    st.topo_clean = degenerate == 0 && directed.values().all(|&c| c == 1);
    st.interior_open_edges = directed.keys().filter(|&&(a, c)| !directed.contains_key(&(c, a)) && !on_boundary(m.ipos[a as usize], m.ipos[c as usize])).count();
    let isl = remove_islands(&mut m, min_island);
    st.islands_removed = isl.removed_components;
    st.island_tris_removed = isl.removed_triangles;
    st.voxel_tris = m.indices.len() / 3;
    let k = grid.voxel_m() / SUB as f64;
    let world: Vec<DVec3> = m.ipos.iter().map(|p| DVec3::new(p[0] as f64 * k, p[1] as f64 * k, p[2] as f64 * k)).collect();
    let level_cell: Vec<f64> = m
        .ipos
        .par_iter()
        .map(|p| {
            let q = p.map(|c| c.div_euclid(SUB));
            if smooth { layout.smooth_filter_cell(grid.voxel_m(), q) } else { grid.voxel_m() * (1u64 << filter_level(layout, q)) as f64 }
        })
        .collect();
    let n = world.len();
    Mesh { world, level_cell, flags: vec![0.0; n], indices: m.indices }
}

/// Field-gradient normal, multi-scale SDF AO, cavity and convexity for every vertex.
/// Soft visibility of a point light from `p` by sphere-tracing the field (air = density > 0).
fn soft_visibility(d: &dyn Fn(DVec3) -> f64, p: DVec3, light: DVec3, start: f64) -> f64 {
    let to = light - p;
    let dist = to.length();
    let dir = to / dist;
    let (mut t, mut res) = (start, 1.0f64);
    for _ in 0..96 {
        if t >= dist - 0.2 {
            break;
        }
        let h = d(p + dir * t);
        if h < 0.002 {
            return 0.0;
        }
        res = res.min(10.0 * h / t);
        t += h.clamp(0.03, 1.0);
    }
    res.clamp(0.0, 1.0)
}

/// `light`: optional point light whose soft visibility is written to `attr.w` (cave lantern).
fn attributes(field: &dyn DensityField, m: &Mesh, cam: DVec3, ao_scales: &[f64], cav_scale: f64, light: Option<DVec3>) -> Vec<QVertex> {
    (0..m.world.len())
        .into_par_iter()
        .map(|i| {
            let p = m.world[i];
            let cell = m.level_cell[i];
            let d = |q: DVec3| field.density_lod([q.x, q.y, q.z], cell);
            let h = (0.5 * cell).max(0.02);
            let g = DVec3::new(
                d(p + DVec3::X * h) - d(p - DVec3::X * h),
                d(p + DVec3::Y * h) - d(p - DVec3::Y * h),
                d(p + DVec3::Z * h) - d(p - DVec3::Z * h),
            );
            let n = if g.length_squared() > 0.0 { g.normalize() } else { DVec3::Y };
            let (mut occ, mut wsum) = (0.0, 0.0);
            for (k, &s) in ao_scales.iter().enumerate() {
                let s = s.max(cell * 1.5);
                let w = 1.0 / (1.0 + k as f64 * 0.35);
                occ += w * ((s - d(p + n * s)) / s).clamp(0.0, 1.0);
                wsum += w;
            }
            let ao = (1.0 - 1.15 * occ / wsum).clamp(0.05, 1.0);
            let cs = cav_scale.max(cell * 1.5);
            let cavity = ((cs - d(p + n * cs)) / cs).clamp(0.0, 1.0);
            let convex = ((d(p - n * cs) + cs) / cs).clamp(0.0, 1.0);
            let w = match light {
                Some(l) if (l - p).dot(n) > 0.0 => soft_visibility(&d, p + n * (2.0 * cell), l, 2.0 * cell),
                Some(_) => 0.0,
                None => m.flags[i] as f64,
            };
            let r = p - cam;
            QVertex { pos: [r.x as f32, r.y as f32, r.z as f32], nrm: [n.x as f32, n.y as f32, n.z as f32], attr: [ao as f32, cavity as f32, convex as f32, w as f32] }
        })
        .collect()
}

/// Far field (C1 stand-in): nested square heightfield rings from the voxel region edge outward.
/// Ring k spans [R_k, 2 R_k] with spacing s_k = ratio * R_k (both double per ring, grids aligned,
/// so every coarse boundary vertex is also a fine one). Boundary vertices use the coarser ring's
/// filter and odd fine boundary vertices are snapped to their neighbours' average: no T-junction
/// cracks. Inside the voxel region (1.5 cell overlap) the ring is lowered so the voxel surface
/// wins. Cells outside the view wedge (+ shadow-caster margin) are skipped.
#[allow(clippy::too_many_arguments)]
fn far_rings(a: &Alpine, inner: f64, ratio: f64, outer: f64, centre: (f64, f64), cam: DVec3, yaw: f64, half_fov: f64, min_cell: f64) -> Mesh {
    let mut m = Mesh { world: Vec::new(), level_cell: Vec::new(), flags: Vec::new(), indices: Vec::new() };
    let mut r = inner;
    let mut s = ratio * inner;
    let keep = |x: f64, z: f64| {
        let (dx, dz) = (x - cam.x, z - cam.z);
        if dx.hypot(dz) < 1200.0 {
            return true;
        }
        let ang = dx.atan2(-dz);
        let d = (ang - yaw + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
        d.abs() < half_fov + 0.45
    };
    while r < outer {
        let r_out = r * 2.0;
        let n = (2.0 * r_out / s).round() as i64;
        let ro = n as f64 * s / 2.0;
        let mut cells = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let (lx, lz) = (-ro + (i as f64 + 0.5) * s, -ro + (j as f64 + 0.5) * s);
                if lx.abs().max(lz.abs()) < r - 1.5 * s || !keep(lx + centre.0, lz + centre.1) {
                    continue;
                }
                cells.push((i, j));
            }
        }
        let mut vid: HashMap<(i64, i64), u32> = HashMap::new();
        let mut order = Vec::new();
        for &(i, j) in &cells {
            for v in [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)] {
                let next = (m.world.len() + order.len()) as u32;
                vid.entry(v).or_insert_with(|| {
                    order.push(v);
                    next
                });
            }
        }
        let pts: Vec<(DVec3, f64)> = order
            .par_iter()
            .map(|&(i, j)| {
                let (lx, lz) = (-ro + i as f64 * s, -ro + j as f64 * s);
                let (x, z) = (lx + centre.0, lz + centre.1);
                let edge = i == 0 || j == 0 || i == n || j == n;
                // continuous filter: s at the inner square, 2 s at the outer one (== the next
                // ring's inner value), never below the voxel field's outermost filter: no steps
                let t = ((lx.abs().max(lz.abs()) - r) / r).clamp(0.0, 1.0);
                let lv = (s.log2() + t) * 16.0;
                let cell = (lv.round() / 16.0).exp2().max(min_cell);
                let mut y = if edge && ((i % 2 == 1 && (j == 0 || j == n)) || (j % 2 == 1 && (i == 0 || i == n))) {
                    let (x0, z0, x1, z1) = if j == 0 || j == n { (x - s, z, x + s, z) } else { (x, z - s, x, z + s) };
                    0.5 * (a.height(x0, z0, cell) + a.height(x1, z1, cell))
                } else {
                    a.height(x, z, cell)
                };
                let inside = (r - lx.abs().max(lz.abs())) / s;
                if inside > 0.0 {
                    y -= 0.6 * s * inside.clamp(0.0, 1.0) + 1.0;
                }
                (DVec3::new(x, y, z), cell)
            })
            .collect();
        for (p, cell) in pts {
            m.world.push(p);
            m.level_cell.push(cell);
            m.flags.push(1.0);
        }
        for (i, j) in cells {
            let v = |a: i64, b: i64| vid[&(a, b)];
            // CCW seen from above (+y)
            m.indices.extend([v(i, j), v(i, j + 1), v(i + 1, j + 1), v(i, j), v(i + 1, j + 1), v(i + 1, j)]);
        }
        r = r_out;
        s *= 2.0;
    }
    m
}

/// Sphere radius of the descent's curvature (the planet crate's Earth radius).
fn planet_radius_m() -> f64 {
    planet::EARTH_RADIUS_M
}

fn srgb_to_lin(v: u8) -> f32 {
    let c = v as f32 / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}
fn lin_to_srgb(c: f32) -> u8 {
    let s = if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

/// Display-referred log-average luminance below which the mountains scene is auto-exposed.
const AUTO_KEY: f32 = 0.075;
/// Brighten-only auto-exposure ceiling (was 2.5: a dark forest close-up was lifted into a milky,
/// low-contrast veil; at 2.0 it stays a shaded scene).
const AUTO_MAX: f32 = 2.0;

/// Log-average of the linear (display) luminance of an RGBA8 sRGB image (every 7th pixel).
fn log_avg_luminance(px: &[u8]) -> f32 {
    let lut: Vec<f32> = (0..=255).map(srgb_to_lin).collect();
    let (mut s, mut n) = (0.0f64, 0u64);
    for c in px.as_chunks::<4>().0.iter().step_by(7) {
        let l = 0.2126 * lut[c[0] as usize] + 0.7152 * lut[c[1] as usize] + 0.0722 * lut[c[2] as usize];
        s += ((l + 1e-4) as f64).ln();
        n += 1;
    }
    (s / n.max(1) as f64).exp() as f32
}

fn downsample(px: &[u8], w: u32, h: u32, f: u32, keep_alpha: bool) -> Vec<u8> {
    let lut: Vec<f32> = (0..=255).map(srgb_to_lin).collect();
    let (ow, oh) = (w / f, h / f);
    let mut out = vec![0u8; (ow * oh * 4) as usize];
    out.par_chunks_mut((ow * 4) as usize).enumerate().for_each(|(y, row)| {
        for x in 0..ow as usize {
            // layer mode (Earth round 4): colour = the alpha-weighted mean of the samples (straight
            // alpha: the terrain's own colour, no sky mixed in at silhouettes, so the composite
            // has no halo); the default view (all alpha 255) is the plain mean, as before
            let at = |dx: usize, dy: usize| (((y * f as usize + dy) * w as usize) + x * f as usize + dx) * 4;
            let mut wsum = 0.0f32;
            if keep_alpha {
                for dy in 0..f as usize {
                    for dx in 0..f as usize {
                        wsum += px[at(dx, dy) + 3] as f32;
                    }
                }
            }
            for ch in 0..3 {
                let mut s = 0.0;
                for dy in 0..f as usize {
                    for dx in 0..f as usize {
                        let c = lut[px[at(dx, dy) + ch] as usize];
                        s += if keep_alpha && wsum > 0.0 { c * px[at(dx, dy) + 3] as f32 } else { c };
                    }
                }
                row[x * 4 + ch] = lin_to_srgb(if keep_alpha && wsum > 0.0 { s / wsum } else { s / (f * f) as f32 });
            }
            row[x * 4 + 3] = if keep_alpha {
                let a: u32 = (0..f as usize * f as usize).map(|k| px[(((y * f as usize + k / f as usize) * w as usize) + x * f as usize + k % f as usize) * 4 + 3] as u32).sum();
                ((a as f32 / (f * f) as f32) + 0.5) as u8
            } else {
                255
            };
        }
    });
    out
}

fn write_png(path: &str, w: u32, h: u32, px: &[u8]) {
    let mut enc = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(path).unwrap()), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(px).unwrap();
}

fn v4(v: Vec3, w: f32) -> [f32; 4] {
    [v.x, v.y, v.z, w]
}

struct View {
    cam: DVec3,
    fwd: Vec3,
    params: QParamsGpu,
    shadows: Vec<ShadowView>,
    /// per cascade: world light-space grid cell of the snapped centre and the slice (near, far)
    cells: Vec<([i64; 3], f32, f32)>,
    spot: bool,
    shadow_res: u32,
    /// Instanced vegetation / clutter and the finite camera matrix for its GPU cull.
    veg: Option<(veg::VegSet, Mat4)>,
    chunks: Vec<render_ash::quality::TerrainChunk>,
    /// receiver slice view-projection per cascade (caster culling)
    slices: Vec<Mat4>,
    veg_ms: f64,
    atmosphere: bool,
}

/// Content fingerprint (64-bit multiply-rotate over 8-byte words; not cryptographic: it guards a
/// shadow cache against a changed scene, every byte counts).
#[derive(Default)]
struct Fp(u64);
impl Fp {
    fn bytes(&mut self, b: &[u8]) {
        let mut h = self.0 ^ (b.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let (words, rest) = b.as_chunks::<8>();
        for w in words {
            h = (h ^ u64::from_le_bytes(*w)).wrapping_mul(0xFF51_AFD7_ED55_8CCD).rotate_left(29);
        }
        for &x in rest {
            h = (h ^ x as u64).wrapping_mul(0xC4CE_B9FE_1A85_EC53).rotate_left(23);
        }
        self.0 = h;
    }
}

fn camera_matrices(fwd: Vec3, aspect: f32) -> (Mat4, Mat4) {
    let view = render_ash::cull::look_at(Vec3::ZERO, fwd, Vec3::Y);
    let proj = reverse_z_infinite(FOV, aspect, 0.05);
    let vp = proj * view;
    // clip (x, y, 1, 1) -> direction: inverse of view * (finite-far version of proj)
    let pf = glam::camera::rh::proj::vulkan::perspective(FOV, aspect, 0.05, 1000.0);
    (vp, (pf * view).inverse())
}

const SPLITS: [f32; 4] = [120.0, 600.0, 2500.0, 9000.0];

/// Cascade split distances: [`SPLITS`], or `SPLITS=a,b,c,d` (m; the orbit -> mountains descent
/// widens them with the camera's height above the terrain).
fn splits() -> [f32; 4] {
    std::env::var("SPLITS").ok().and_then(|v| v.split(',').map(|t| t.trim().parse::<f32>().ok()).collect::<Option<Vec<_>>>()).and_then(|v| <[f32; 4]>::try_from(v).ok()).unwrap_or(SPLITS)
}

/// Earth round 3 "composite layer" mode for the orbit -> mountains descent:
/// `LAYER=cx,cz,r0,r1,cirrus_y[,fade]` (m). The flat world is bent onto the sphere around (cx, cz)
/// for rasterisation, the output alpha is terrain coverage x a radial fade r0..r1 from (cx, cz)
/// (sky 0), the cirrus deck is fixed at cirrus_y + 6 km, the voxel shells are centred on the
/// ground under the camera (which may be kilometres up) and the far rings go all round. `fade`
/// (default 1; `planet::descent::vista_fade`) scales the aerial-only parts (bend, shell drop,
/// ring extent / azimuth) toward the default view as the camera nears the vista. Unset = the
/// default view, unchanged.
fn layer() -> Option<[f32; 6]> {
    let v: Vec<f32> = std::env::var("LAYER").ok()?.split(',').map(|t| t.trim().parse().ok()).collect::<Option<_>>()?;
    match v.len() {
        5 => Some([v[0], v[1], v[2], v[3], v[4], 1.0]),
        6 => Some([v[0], v[1], v[2], v[3], v[4], v[5].clamp(0.0, 1.0)]),
        _ => None,
    }
}

/// Far cascades in clipmap mode (round 10; `SHADOW_CLIP=0` = round 9 frustum-slice cascades).
fn shadow_clip() -> bool {
    std::env::var("SHADOW_CLIP").as_deref() != Ok("0")
}

/// Sun cascades for a camera at `o + d` looking along `fwd`, as matrices relative to the
/// geometry origin `o` (the scene is built camera-relative to `o`; `d` moves the eye on a camera
/// path without rebuilding it). Returns the shadow views, per cascade (grid cell, near, far) for
/// the cache keys, and the caster-culling receiver region per cascade.
/// - c0/c1: [`sun_cascade`] fitted to the camera frustum slice; receivers = the slice, i.e. view
///   depth <= radial distance and >= 0.55 x it inside the FOV, over the shader's cascade choice +
///   blend band (radial distance in (0.8 split[c-1], split[c]]).
/// - c2/c3 (clip mode): [`sun_cascade_clip`] squares around the camera with a 1/8-map grid;
///   receivers = the whole square, so a cached map is complete in every view direction.
type Cascades = (Vec<ShadowView>, Vec<([i64; 3], f32, f32)>, Vec<Mat4>);

fn cascades(o: DVec3, d: Vec3, fwd: Vec3, sun: Vec3, res: u32, clip: bool) -> Cascades {
    let aspect = W as f32 / H as f32;
    let c = o + d.as_dvec3();
    let tr = Mat4::from_translation(-d);
    let view_m = render_ash::cull::look_at(d, d + fwd, Vec3::Y);
    let (mut shadows, mut cells, mut slices) = (Vec::new(), Vec::new(), Vec::new());
    let mut near_d = 0.05;
    let sp = splits();
    for (i, &f) in sp.iter().enumerate() {
        let far = clip && i >= 2;
        let (m, texel, cell) = if far { sun_cascade_clip(c, f, sun, res, 6000.0, res / 8) } else { sun_cascade(c, fwd, FOV, aspect, near_d, f, sun, res, 6000.0) };
        let m = m * tr;
        shadows.push(ShadowView { view_proj: m, texel });
        cells.push((cell, near_d, f));
        slices.push(if far { m } else { render_ash::cull::vk_perspective(FOV, aspect, if i == 0 { 0.05 } else { sp[i - 1] * 0.8 * 0.55 }, f) * view_m });
        near_d = f * 0.8;
    }
    (shadows, cells, slices)
}

/// Shadow cache keys (0 = always render): c2/c3 hash (cell, slice, map size, caster fingerprint,
/// near/far, sun); c0/c1 use camera-distance LOD and never cache.
fn cache_keys_for(cells: &[([i64; 3], f32, f32)], res: u32, geo: u64, sun: [f32; 3]) -> Vec<u64> {
    cells
        .iter()
        .enumerate()
        .map(|(c, &(cell, n, f))| {
            if c < 2 {
                return 0;
            }
            let mut h = Fp::default();
            h.bytes(bytemuck::cast_slice(&[cell[0], cell[1], cell[2], c as i64, res as i64, geo as i64]));
            h.bytes(bytemuck::cast_slice(&[n, f, sun[0], sun[1], sun[2]]));
            h.0 | 1
        })
        .collect()
}

/// Earth round 5: build and upload the ground map from the graft bake's albedo levels (same
/// grids, outer first) and the ground's DEM weight: RGBA8 words (sRGB albedo, a = 7-bit
/// imagery weight | 0x80 water), box mips down to 64^2. Returns (address, levels, extents).
fn ground_map(b: &mut AshBackend, g: &voxel::ground::Ground, grf: &str) -> (u64, [[u32; 4]; 2], [f32; 4]) {
    use rayon::prelude::*;
    let t = Instant::now();
    let bytes = std::fs::read(grf).unwrap_or_else(|e| panic!("{grf}: {e}"));
    let gf = planet::graft::GraftFile::from_bytes(&bytes).expect("GRF1 graft file");
    assert_eq!(gf.levels.len(), 2, "ground map: two graft levels");
    let ss = |a: f64, b: f64, x: f64| {
        let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let lin = |c: u8| {
        let c = c as f32 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    let enc = |c: f32| {
        let c = c.clamp(0.0, 1.0);
        let s = if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
        (s * 255.0 + 0.5) as u32
    };
    let mut words: Vec<u32> = Vec::new();
    let mut lv = [[0u32; 4]; 2];
    for (k, l) in gf.levels.iter().enumerate() {
        let n = l.size as usize;
        let step = 2.0 * l.half_m / n as f64;
        // linear rgb, weight, water
        let mut cur: Vec<[f32; 5]> = (0..n * n)
            .into_par_iter()
            .map(|q| {
                let (x, z) = (gf.world_xz.0 - l.half_m + ((q % n) as f64 + 0.5) * step, gf.world_xz.1 - l.half_m + ((q / n) as f64 + 0.5) * step);
                let a = &l.albedo[q * 4..q * 4 + 4];
                // the graft bake's imagery weight (it leads the height blend)
                let w = ss(0.0, 0.6, g.weight(x, z)) as f32;
                [lin(a[0]), lin(a[1]), lin(a[2]), w, if a[3] < 128 { 1.0 } else { 0.0 }]
            })
            .collect();
        lv[k] = [n as u32, 0, words.len() as u32, 0];
        let mut m = n;
        loop {
            words.extend(cur.iter().map(|c| enc(c[0]) | enc(c[1]) << 8 | enc(c[2]) << 16 | (((c[3].clamp(0.0, 1.0) * 127.0).round() as u32) | if c[4] >= 0.5 { 128 } else { 0 }) << 24));
            lv[k][1] += 1;
            if m <= 64 {
                break;
            }
            let h = m / 2;
            cur = (0..h * h)
                .into_par_iter()
                .map(|q| {
                    let (i, j) = (q % h, q / h);
                    let mut o = [0f32; 5];
                    for (a, bb) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let c = cur[(2 * j + bb) * m + 2 * i + a];
                        for t in 0..5 {
                            o[t] += 0.25 * c[t];
                        }
                    }
                    o
                })
                .collect();
            m = h;
        }
    }
    let addr = b.quality_set_ground_map(&words).expect("upload ground map");
    eprintln!("ground map: {:.1} MB, {:.0} ms", words.len() as f64 * 4.0 / 1048576.0, t.elapsed().as_secs_f64() * 1e3);
    (addr, lv, [gf.world_xz.0 as f32, gf.world_xz.1 as f32, gf.levels[0].half_m as f32, gf.levels[1].half_m as f32])
}

/// Earth round 6: the default mountains view's ground (DEM + core) and imagery (the descent's
/// graft bake); both are data outside the source tree, baked by the `descent` example.
const DEFAULT_GROUND: &str = "/workspace/data/earth/ground_alps_e6.grd";
const DEFAULT_GROUND_ALB: &str = "/workspace/data/earth/graft_alps_e6.grf";

fn mountains(b: &mut AshBackend, st: &mut Stats) -> (Vec<QVertex>, Vec<u32>, View) {
    // Earth round 5: GROUND=<file.grd> grounds the field on real elevation, GROUND_ALB=<graft.grf>
    // gives the matching imagery albedo (ground map). Earth round 6: the default view is
    // grounded by default (DEFAULT_GROUND / DEFAULT_GROUND_ALB when present; a missing file
    // falls back to the procedural field with a warning); GROUND=off = the round-5 default view.
    let ground_file = match std::env::var("GROUND") {
        Ok(f) if f == "off" => None,
        Ok(f) => Some(f),
        Err(_) if std::path::Path::new(DEFAULT_GROUND).exists() => Some(DEFAULT_GROUND.to_string()),
        Err(_) => {
            eprintln!("warning: {DEFAULT_GROUND} missing: procedural (round-5) mountains (descent bakes it)");
            None
        }
    };
    let ground = ground_file.map(|f| std::sync::Arc::new(voxel::ground::Ground::load(&f).unwrap_or_else(|e| panic!("{e}"))));
    let alb_file = match std::env::var("GROUND_ALB") {
        Ok(f) if f == "off" => None,
        Ok(f) => Some(f),
        Err(_) if std::path::Path::new(DEFAULT_GROUND_ALB).exists() => Some(DEFAULT_GROUND_ALB.to_string()),
        Err(_) => None,
    };
    let field = match &ground {
        Some(g) => Alpine::grounded(7, g.clone()),
        None => Alpine::new(7),
    };
    let gmap = ground.as_ref().and_then(|g| alb_file.as_ref().map(|f| ground_map(b, g, f)));
    // default vista: valley floor at ~630 m looking SSE up a valley at a range of several
    // peaks 2.5-12 km away (chosen by a skyline-scored search; CAM_* env vars override)
    let (cx, cz) = (std::env::var("CAM_X").ok().and_then(|v| v.parse().ok()).unwrap_or(7700.0), std::env::var("CAM_Z").ok().and_then(|v| v.parse().ok()).unwrap_or(-4200.0));
    let ground = field.height(cx, cz, 0.5);
    // CAM_Y: absolute height (m) instead of CAM_H above the ground (the descent path)
    let cam_y = std::env::var("CAM_Y").ok().and_then(|v| v.parse::<f64>().ok());
    let cam = DVec3::new(cx, cam_y.unwrap_or_else(|| ground + std::env::var("CAM_H").ok().and_then(|v| v.parse().ok()).unwrap_or(40.0)), cz);
    let lay = layer();
    let yaw: f32 = std::env::var("CAM_YAW").ok().and_then(|v| v.parse().ok()).unwrap_or(2.75);
    let pitch: f32 = std::env::var("CAM_PITCH").ok().and_then(|v| v.parse().ok()).unwrap_or(0.08);
    let fwd = Vec3::new(yaw.sin() * pitch.cos(), pitch.sin(), -yaw.cos() * pitch.cos()).normalize();
    let grid = BrickGrid::new(32, 0.5).unwrap();
    let cfg = VoxelConfig::new(grid);
    let l = |v: f64| (v / grid.voxel_m()).floor() as i64;
    // layer mode: shells around the ground under the camera, so an aerial camera still gets the
    // near field below it (the default view: around the camera itself)
    let shell_y = match lay {
        Some(l) if l[5] >= 1.0 => ground.min(cam.y),
        Some(l) => cam.y - (cam.y - ground.min(cam.y)) * l[5] as f64,
        None => cam.y,
    };
    let layout = ShellLayout::camera_centred(&grid, [l(cam.x), l(shell_y), l(cam.z)], &[3, 3, 6, 6]);
    let outer = *layout.boxes.last().unwrap();
    let coarse = 32 * grid.step(layout.max_lod());
    let al = |v: i64, up: bool| if up { (v + coarse - 1).div_euclid(coarse) * coarse } else { v.div_euclid(coarse) * coarse };
    // the region spans the field's rigorous height bounds: peaks can never be cut at its top
    let (hb_lo, hb_hi) = field.height_bounds();
    let region = (
        [al(outer.lo[0] - coarse, false), al(l(hb_lo), false), al(outer.lo[2] - coarse, false)],
        [al(outer.hi[0] + coarse, true), al(l(hb_hi), true), al(outer.hi[2] + coarse, true)],
    );
    // conservative skip (Lipschitz + overhang bound; tested in voxel q5)
    let skip = |c: BrickCoord| {
        let o = grid.brick_origin(c);
        let e = (32 * grid.step(c.lod)) as f64 * grid.voxel_m();
        let lo = [0, 1, 2].map(|i| o[i] as f64 * grid.voxel_m());
        !field.may_cross(lo, [lo[0] + e, lo[1] + e, lo[2] + e], grid.voxel_m() * (1u64 << c.lod) as f64)
    };
    let mut near = voxel_near(b, &field, &cfg, &layout, region, &skip, 400, true, st);
    if std::env::var_os("QUALITY_SPIKES").is_some() {
        // spike-triangle scan of the voxel near field (voxel::meshq), by horizontal distance
        // from the camera; `SPIKE_AT=x,z,r` adds a disc (e.g. one outcrop)
        let pos: Vec<[f64; 3]> = near.world.iter().map(|v| [v.x, v.y, v.z]).collect();
        let hd = |c: [f64; 3]| (c[0] - cam.x).hypot(c[2] - cam.z);
        for (lo, hi) in [(0.0, 60.0), (60.0, 250.0), (250.0, 1000.0), (1000.0, f64::MAX), (0.0, f64::MAX)] {
            let s = voxel::meshq::spike_counts(&pos, &near.indices, |c| (lo..hi).contains(&hd(c)));
            let hi_s = if hi < 1e9 { format!("{hi}") } else { "inf".into() };
            eprintln!("spikes {lo:>4}..{hi_s:<5} m: tris {:>8} folds {:>6} creases {:>6} needles {:>6}", s.tris, s.folds, s.creases, s.needles);
        }
        if let Some([x, z, r]) = std::env::var("SPIKE_AT").ok().and_then(|v| v.split(',').map(|t| t.parse::<f64>().ok()).collect::<Option<Vec<_>>>()).and_then(|v| <[f64; 3]>::try_from(v).ok()) {
            let s = voxel::meshq::spike_counts(&pos, &near.indices, |c| (c[0] - x).hypot(c[2] - z) < r);
            eprintln!("spikes at ({x}, {z}) r {r}: tris {} folds {} creases {} needles {}", s.tris, s.folds, s.creases, s.needles);
        }
    }
    let t = Instant::now();
    let half = (region.1[0] - region.0[0]).min(region.1[2] - region.0[2]) as f64 * 0.5 * grid.voxel_m();
    let centre = DVec3::new((region.0[0] + region.1[0]) as f64 * 0.25, 0.0, (region.0[2] + region.1[2]) as f64 * 0.25);
    let edge_cell = layout.smooth_filter_cell(grid.voxel_m(), region.0);
    // layer mode: rings to RING_OUTER (m) in every direction (an aerial view sees all around)
    let default_rings = (26000.0, (FOV * W as f32 / H as f32 * 0.5) as f64);
    let (ring_outer, ring_fov) = match lay {
        Some(l) => {
            let full = (std::env::var("RING_OUTER").ok().and_then(|v| v.parse().ok()).unwrap_or(104000.0), std::f64::consts::PI);
            let f = l[5] as f64;
            if f >= 1.0 { full } else { (default_rings.0 + (full.0 - default_rings.0) * f, default_rings.1 + (full.1 - default_rings.1) * f) }
        }
        None => default_rings,
    };
    let ring = far_rings(&field, half, 0.004, ring_outer, (centre.x, centre.z), cam, yaw as f64, ring_fov, edge_cell);
    st.ring_tris = ring.indices.len() / 3;
    let base = near.world.len() as u32;
    near.world.extend(ring.world);
    near.level_cell.extend(ring.level_cell);
    near.flags.extend(ring.flags);
    near.indices.extend(ring.indices.iter().map(|i| i + base));
    if std::env::var_os("QUALITY_DIAG").is_some() {
        // per azimuth: max elevation angle of mesh vertices vs of the field itself
        for k in -4..=4 {
            let az = yaw as f64 + k as f64 * 0.15;
            let (dx, dz) = (az.sin(), -az.cos());
            let mut fmax = (f64::MIN, 0.0, 0.0);
            let mut t = 60.0;
            while t < 26000.0 {
                let h = field.height(cam.x + dx * t, cam.z + dz * t, (t * 0.004).max(1.0));
                let e = (h - cam.y) / t;
                if e > fmax.0 { fmax = (e, t, h); }
                t *= 1.01;
            }
            let mut mmax = (f64::MIN, 0.0, 0.0, 0.0);
            for (i, p) in near.world.iter().enumerate() {
                let r = *p - cam;
                let d = r.x.hypot(r.z);
                if d < 30.0 { continue; }
                let a = r.x.atan2(-r.z);
                let da = (a - az + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
                if da.abs() < 0.004 && r.y / d > mmax.0 { mmax = (r.y / d, d, p.y, near.flags[i] as f64); }
            }
            eprintln!("az {k:+}: field max elev {:.4} at {:.0} m (h {:.0}) | mesh max elev {:.4} at {:.0} m (h {:.0}, ring {})", fmax.0, fmax.1, fmax.2, mmax.0, mmax.1, mmax.2, mmax.3);
        }
    }
    let mut verts = attributes(&field, &near, cam, &[2.0, 6.0, 16.0, 40.0, 110.0], 3.0, None);
    // jointed-outcrop material info for voxel vertices, in interpolation-safe form:
    // attr.w = -rock weight (otherwise 0 on voxel terrain) and |nrm| = 1 + joint-crack
    // proximity (the fragment shader normalises the normal anyway)
    verts.par_iter_mut().enumerate().for_each(|(i, v)| {
        if near.flags[i] != 0.0 {
            return;
        }
        let p = near.world[i];
        if let Some(t) = field.tor_info([p.x, p.y, p.z], near.level_cell[i]) {
            v.attr[3] = -(t.rock as f32);
            let k = 1.0 + t.crack as f32;
            v.nrm = v.nrm.map(|c| c * k);
        }
    });
    st.attr_ms = t.elapsed().as_secs_f64() * 1e3;

    let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    // SUN_AZ: absolute sun azimuth (rad, same convention as CAM_YAW) instead of relative to the view
    let (sy, se) = (std::env::var("SUN_AZ").ok().and_then(|v| v.parse().ok()).unwrap_or(yaw + env("SUN_REL_YAW", 2.35)), env("SUN_EL", 0.5));
    let sun = Vec3::new(sy.sin() * se.cos(), se.sin(), -sy.cos() * se.cos()).normalize();
    let aspect = W as f32 / H as f32;
    let (vp, inv) = camera_matrices(fwd, aspect);
    let res = 2048;
    let splits = splits();
    let (shadows, cells, slices) = cascades(cam, Vec3::ZERO, fwd, sun, res, shadow_clip());
    let mut p = QParamsGpu { view_proj: vp.to_cols_array_2d(), inv_dir: inv.to_cols_array_2d(), ..Default::default() };
    p.cam_world = [cam.x as f32, cam.y as f32, cam.z as f32, FOV / H as f32 / SS as f32];
    p.sun_dir = v4(sun, 5.5);
    p.sun_color = [1.0, 0.87, 0.72, 0.42];
    p.sky_zenith = [0.10, 0.22, 0.52, 0.0];
    p.sky_horizon = [0.62, 0.72, 0.86, 0.000_085];
    p.fog = [1.0 / 2600.0, 300.0, 0.92, 0.9];
    p.fog_color = [0.50, 0.60, 0.76, 0.9];
    p.shadow_info = [0.0, 0.0, 1.0, 0.0];
    p.splits = splits;
    // Hillaire atmosphere (ATMO=0 falls back to the round-1 analytic haze). Mie x MIE (default
    // 8: a hazy alpine summer day, ~60 km visibility), ground albedo 0.12, aerial perspective to
    // 40 km. Exposure 1.1 keeps the sky out of the ACES toe (which over-saturates blue).
    let atmosphere = std::env::var("ATMO").as_deref() != Ok("0");
    if atmosphere {
        p.atmo = [env("MIE", 8.0), 0.12, env("AP_KM", 40.0), 1.0];
        // daylight grade (quality.slang `grade`): log contrast 0.10 around mid grey + a slightly
        // cool, luminance-preserving white balance; GRADE=0 disables
        let g = std::env::var("GRADE").as_deref() != Ok("0");
        // ENV_SH=0: legacy ambient (up/side irradiance lerp + fixed bounce) instead of the SH environment
        if std::env::var("ENV_SH").as_deref() == Ok("0") {
            p.lantern_pos[3] = 1.0;
        }
        p.atmo2 = [if g { env("GRADE_K", 0.10) } else { 0.0 }, 0.0047, 1.0, if g { env("GRADE_WB", 0.015) } else { 0.0 }];
        p.sun_color = [1.0, 1.0, 1.0, env("EXPOSURE", 1.1)];
    }
    if let Some((addr, lv, f)) = gmap {
        (p.gmap, p.gmap_lv, p.gmap_f) = (addr, lv, f);
    }
    if let Some([lx, lz, r0, r1, cy, fade]) = lay {
        p.lamp_pos = [lx, lz, (0.5 / planet_radius_m()) as f32 * fade, 1.0];
        p.lamp_dir = [r0, r1, cy, 0.0];
    }
    if std::env::var_os("QUALITY_DEBUG_AP").is_some() {
        p.lantern_color[3] = 3.0;
    }
    if std::env::var_os("QUALITY_DEBUG_RING").is_some() {
        p.lantern_color[3] = 1.0;
    }
    if std::env::var_os("QUALITY_DEBUG_ENV").is_some() {
        p.lantern_color[3] = 4.0;
    }
    if std::env::var_os("QUALITY_DEBUG_SHADOW").is_some() {
        p.lantern_color[3] = 2.0;
    }
    let tv = Instant::now();
    let veg = (std::env::var("VEG").as_deref() != Ok("0")).then(|| {
        let surf = veg::Surface::new(&near.world, &near.indices, cam, 7300.0);
        let mut set = veg::build(&field, &surf, cam, yaw as f64, (FOV * W as f32 / H as f32 * 0.5) as f64);
        // route the instances through the asset format (rANS, Morton-sorted, 1/128 m):
        // encode in 4096-instance chunks, verify, decode in parallel and upload the decoded set
        if std::env::var("BRUTAL_VEG_ASSET").as_deref() != Ok("0") {
            use rayon::prelude::*;
            let t0 = Instant::now();
            let mut w = asset::Writer::new(grid);
            for ch in set.records.chunks(4096) {
                w.add_instances(ch);
            }
            let bytes = asset::AlignedBytes::from_slice(&w.finish());
            let t1 = Instant::now();
            let f = asset::File::open(bytes.as_slice(), asset::Limits::default()).expect("veg asset verifies");
            let c = [cam.x, cam.y, cam.z];
            let dec: Vec<lowlevel_abi::InstanceGpu> = (0..f.chunks().len())
                .into_par_iter()
                .flat_map_iter(|i| f.decode_instances(i).expect("veg asset decodes").into_iter().map(move |r| r.to_gpu(c)))
                .collect();
            let t2 = Instant::now();
            assert_eq!(dec.len(), set.instances.len());
            eprintln!(
                "veg asset: {} instances -> {} bytes ({:.2} B/inst, {:.1}:1 vs InstanceGpu), encode {:.1} ms, verify+decode {:.1} ms",
                dec.len(),
                bytes.as_slice().len(),
                bytes.as_slice().len() as f64 / dec.len().max(1) as f64,
                (dec.len() * std::mem::size_of::<lowlevel_abi::InstanceGpu>()) as f64 / bytes.as_slice().len().max(1) as f64,
                (t1 - t0).as_secs_f64() * 1e3,
                (t2 - t1).as_secs_f64() * 1e3
            );
            set.instances = dec;
        }
        // ground under the canopy: shade / litter darkening in the terrain's ambient occlusion
        let mut ao: Vec<f32> = verts.iter().map(|v| v.attr[0]).collect();
        veg::canopy_ao(&set.crowns, &near.world, &mut ao, cam);
        veg::rock_contact_ao(&set.rocks, &near.world, &mut ao, cam);
        for (v, a) in verts.iter_mut().zip(ao) {
            v.attr[0] = a;
        }
        eprintln!("veg: species spruce/fir/larch/snag {:?}, saplings {}, outliers {}, shrubs {}, boulders talus/erratic/stone {:?}", set.species, set.saplings, set.outliers, set.counts[3], set.boulder_kinds);
        let cull_vp = render_ash::cull::vk_perspective(FOV, aspect, 0.5, 40000.0) * render_ash::cull::look_at(Vec3::ZERO, fwd, Vec3::Y);
        (set, cull_vp)
    });
    let veg_ms = tv.elapsed().as_secs_f64() * 1e3;
    // spatial terrain chunks so each sun cascade draws only its casters
    let (indices, chunks) = render_ash::quality::terrain_chunks(&verts, &near.indices, 80.0);
    eprintln!("terrain chunks: {}", chunks.len());
    (verts, indices, View { cam, fwd, params: p, shadows, cells, spot: false, shadow_res: res, veg, veg_ms, atmosphere, chunks, slices })
}

fn cave(b: &mut AshBackend, st: &mut Stats) -> (Vec<QVertex>, Vec<u32>, View) {
    let field = CaveNetwork { seed: 3 };
    let c0 = cave_path(0.0);
    let eye_y = c0[1] - 1.05 + 1.7;
    let cam = DVec3::new(c0[0], eye_y, 0.0);
    let target = cave_path(-11.0);
    let fwd = (Vec3::new(target[0] as f32, (target[1] - 0.25) as f32, target[2] as f32) - cam.as_vec3()).normalize();
    let grid = BrickGrid::new(32, 0.0625).unwrap();
    let cfg = VoxelConfig::new(grid);
    let l = |v: f64| (v / grid.voxel_m()).floor() as i64;
    let layout = ShellLayout::camera_centred(&grid, [l(cam.x), l(cam.y), l(cam.z)], &[2, 2, 2, 2]);
    let outer = *layout.boxes.last().unwrap();
    let coarse = 32 * grid.step(layout.max_lod());
    let al = |v: i64, up: bool| if up { (v + coarse - 1).div_euclid(coarse) * coarse } else { v.div_euclid(coarse) * coarse };
    let region = (
        [al(l(-28.0).min(outer.lo[0] - coarse), false), al(l(-14.0).min(outer.lo[1] - coarse), false), al(l(-64.0).min(outer.lo[2] - coarse), false)],
        [al(l(30.0).max(outer.hi[0] + coarse), true), al(l(18.0).max(outer.hi[1] + coarse), true), al(l(14.0).max(outer.hi[2] + coarse), true)],
    );
    let skip = |_c: BrickCoord| false;
    let near = voxel_near(b, &field, &cfg, &layout, region, &skip, 300, false, st);
    let t = Instant::now();
    let lp = cave_path(-14.5);
    let lantern_w = DVec3::new(lp[0] - 0.4, lp[1] - 1.05 + 0.35, lp[2]);
    let verts = attributes(&field, &near, cam, &[0.12, 0.3, 0.7, 1.6, 3.5], 0.25, Some(lantern_w));
    st.attr_ms = t.elapsed().as_secs_f64() * 1e3;
    let aspect = W as f32 / H as f32;
    let (vp, inv) = camera_matrices(fwd, aspect);
    // headlamp: just above-right of the eye, aimed along the view, slightly down
    let right = fwd.cross(Vec3::Y).normalize();
    let lamp = right * 0.12 + Vec3::Y * 0.10;
    let ldir = (fwd + Vec3::new(0.0, -0.06, 0.0)).normalize();
    let outer_a: f32 = 0.78;
    let proj = glam::camera::rh::proj::vulkan::perspective(outer_a * 2.0, 1.0, 0.05, 60.0);
    let lv = render_ash::cull::look_at(lamp, lamp + ldir, Vec3::Y);
    let shadows = vec![ShadowView { view_proj: proj * lv, texel: 0.004 }];
    let lantern = lantern_w - cam;
    let mut p = QParamsGpu { view_proj: vp.to_cols_array_2d(), inv_dir: inv.to_cols_array_2d(), ..Default::default() };
    p.cam_world = [cam.x as f32, cam.y as f32, cam.z as f32, FOV / H as f32 / SS as f32];
    p.sun_color = [0.0, 0.0, 0.0, 0.8];
    p.sky_zenith = [0.0, 0.0, 0.0, 1.0];
    p.sky_horizon = [0.0, 0.0, 0.0, 0.035];
    p.fog_color = [0.002, 0.0022, 0.0025, 0.035];
    p.lamp_pos = v4(lamp, 40.0);
    p.lamp_dir = v4(ldir, outer_a.cos());
    p.lamp_color = [22.0, 20.5, 18.0, 0.30f32.cos()];
    p.lantern_pos = [lantern.x as f32, lantern.y as f32, lantern.z as f32, 16.0];
    p.lantern_color = [42.0, 20.0, 7.0, 0.0];
    p.shadow_info = [0.0, 0.0, 1.0, 1.0];
    p.splits = [1e9; 4];
    (verts, near.indices, View { cam, fwd, params: p, shadows, cells: Vec::new(), spot: true, shadow_res: 2048, veg: None, veg_ms: 0.0, atmosphere: false, chunks: Vec::new(), slices: Vec::new() })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let scene = args.iter().position(|a| a == "--scene").and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| "mountains".into());
    let validate = !args.iter().any(|a| a == "--no-validation");
    std::fs::create_dir_all("out/quality").unwrap();
    let (e0, w0) = message_counts();
    let mut b = AshBackend::new(BackendConfig { max_instances: 16, extent: (64, 64), validation: validate, ..Default::default() }, None).expect("backend");
    let mut st = Stats::default();
    let t_all = Instant::now();
    let (verts, indices, view) = match scene.as_str() {
        "cave" => cave(&mut b, &mut st),
        "mountains" => mountains(&mut b, &mut st),
        s => panic!("unknown scene {s} (cave | mountains)"),
    };
    let build_ms = t_all.elapsed().as_secs_f64() * 1e3;
    let atlas = view.veg.as_ref().map_or(Vec::new(), |v| render_ash::impostor::atlas_bytes(&v.0.atlas));
    if let Some((v, _)) = view.veg.as_ref() {
        eprintln!("impostor atlas: {} variants x {}x{} frames x {}^2 texels, {:.1} MB, hash {:016x}", v.atlas.len(), render_ash::impostor::IMP_FRAMES, render_ash::impostor::IMP_FRAMES, render_ash::impostor::IMP_RES, atlas.len() as f64 / 1e6, render_ash::impostor::atlas_hash(&v.atlas));
    }
    // static shadow cascade caching: the far cascades (c2, c3) draw camera-independent casters
    // (terrain + impostor cards), so their key = world-space light matrix (grid cell, slice,
    // sun, map size) + a fingerprint of the caster geometry; c0/c1 (distance LOD) never cache.
    // SHADOW_CACHE=0 redraws every cascade every frame.
    let t_fp = Instant::now();
    let geo = {
        let mut h = Fp::default();
        h.bytes(bytemuck::cast_slice(&verts));
        h.bytes(bytemuck::cast_slice(&indices));
        h.bytes(bytemuck::cast_slice(&view.chunks.iter().flat_map(|c| [c.first_index, c.index_count]).collect::<Vec<u32>>()));
        if let Some((v, _)) = view.veg.as_ref() {
            h.bytes(bytemuck::cast_slice(&v.vertices));
            h.bytes(bytemuck::cast_slice(&v.indices));
            h.bytes(bytemuck::cast_slice(&v.instances));
            h.bytes(bytemuck::cast_slice(&v.lod));
            h.bytes(&atlas);
        }
        h.0
    };
    let cache_keys: Vec<u64> = if std::env::var("SHADOW_CACHE").as_deref() == Ok("0") || view.spot {
        Vec::new()
    } else {
        cache_keys_for(&view.cells, view.shadow_res, geo, [view.params.sun_dir[0], view.params.sun_dir[1], view.params.sun_dir[2]])
    };
    eprintln!("shadow cache keys {:?} (caster fingerprint {:.0} ms)", cache_keys.iter().map(|k| *k != 0).collect::<Vec<_>>(), t_fp.elapsed().as_secs_f64() * 1e3);
    let mut frame = QualityFrame {
        width: W * SS,
        height: H * SS,
        vertices: &verts,
        indices: &indices,
        params: view.params,
        shadow_res: view.shadow_res,
        shadows: &view.shadows,
        spot_shadow: view.spot,
        atmosphere: view.atmosphere,
        terrain_chunks: &view.chunks,
        shadow_slices: &view.slices,
        depth_prepass: std::env::var("PREPASS").as_deref() != Ok("0"),
        // round 11: card-depth impostors in the foliage prepass (IMP_CARD=0: round-10 exact depth)
        impostor_card_depth: std::env::var("IMP_CARD").as_deref() != Ok("0"),
        shadow_cache_keys: &cache_keys,
        instanced: view.veg.as_ref().map(|(v, cull_vp)| InstancedSet {
            vertices: &v.vertices,
            indices: &v.indices,
            meshes: &v.meshes,
            instances: &v.instances,
            cull_view_proj: *cull_vp,
            lod: &v.lod,
            impostor_first: veg::MESH_IMP0,
            atlas: Some((atlas.as_slice(), v.atlas.len() as u32)),
        }),
    };
    // first frame warms up (pipeline compile on lavapipe); report the second
    let (warm, _) = b.render_quality(&frame).expect("render");
    // Brighten-only auto exposure (mountains): a frame mostly in terrain / forest shade would
    // render murky at the vista's exposure. Measure the log-average display luminance of the
    // warm-up frame; below the key the exposure is raised (two damped steps, at most x2.0).
    // Frames at or above the key (the default vista: 0.128) keep their exposure exactly.
    // AUTO_EXPOSURE=0 disables it; the cave keeps its fixed exposure.
    if scene == "mountains" && std::env::var("AUTO_EXPOSURE").as_deref() != Ok("0") {
        let base = frame.params.sun_color[3];
        let mut px = warm;
        for step in 0..2 {
            let k = log_avg_luminance(&px);
            let f = (frame.params.sun_color[3] / base * (AUTO_KEY / k).powf(0.85)).clamp(1.0, AUTO_MAX);
            eprintln!("auto exposure step {step}: log-average luminance {k:.4} -> exposure x{f:.3}");
            if (f * base - frame.params.sun_color[3]).abs() < 1e-4 * base {
                break;
            }
            frame.params.sun_color[3] = base * f;
            px = b.render_quality(&frame).expect("render").0;
        }
    }
    // lavapipe timings are noisy (shared CPU): render QUALITY_FRAMES (default 3) timed frames and
    // report the one with the median GPU time
    let nf: usize = std::env::var("QUALITY_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(3).max(1);
    let mut runs: Vec<(Vec<u8>, render_ash::quality::QualityTimings, f64)> = (0..nf)
        .map(|_| {
            let tw = Instant::now();
            let (px, t) = b.render_quality(&frame).expect("render");
            (px, t, tw.elapsed().as_secs_f64() * 1e3)
        })
        .collect();
    eprintln!("gpu frame ms per run: {:?}", runs.iter().map(|r| (r.1.gpu_ms * 10.0).round() / 10.0).collect::<Vec<_>>());
    eprintln!("cached shadow cascades per run: {:?}", runs.iter().map(|r| r.1.cached_cascades).collect::<Vec<_>>());
    // Camera-path benchmark (CAM_PATH=<frames>, mountains): walk at PATH_SPEED m/s (5) while
    // turning at PATH_TURN rad/s (0.785 = 45 deg/s), 30 fps steps. The scene stays built around
    // its origin; the eye moves through the view matrix and the cascades are refitted per
    // frame. Shading distances (LOD, fog) are still measured from the origin, so path frames are
    // a cost / cache benchmark, not reference images. Prints per-frame GPU / shadow ms and the
    // far-cascade cache hits. It also prints a CPU-only hit-rate simulation of the same path over
    // PATH_SIM frames (default 600 = 20 s) for both cascade modes.
    if let Some(np) = std::env::var("CAM_PATH").ok().and_then(|v| v.parse::<usize>().ok()).filter(|_| !view.spot) {
        let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let (speed, turn, dt) = (env("PATH_SPEED", 5.0), env("PATH_TURN", 0.785), 1.0 / 30.0);
        let (yaw0, pitch) = (view.fwd.x.atan2(-view.fwd.z), view.fwd.y.asin());
        let sun = Vec3::from_slice(&frame.params.sun_dir[..3]).normalize();
        let pose = |k: usize| {
            let t = k as f32 * dt;
            let yaw = yaw0 + turn * t;
            let fwd = Vec3::new(yaw.sin() * pitch.cos(), pitch.sin(), -yaw.cos() * pitch.cos()).normalize();
            // integrated heading (closed form of the arc for turn != 0)
            let d = if turn.abs() < 1e-6 {
                Vec3::new(yaw0.sin(), 0.0, -yaw0.cos()) * speed * t
            } else {
                Vec3::new(-(yaw.cos() - yaw0.cos()), 0.0, -(yaw.sin() - yaw0.sin())) * (speed / turn)
            };
            (d, fwd)
        };
        let sim_n = env("PATH_SIM", 600.0) as usize;
        let sun3 = [frame.params.sun_dir[0], frame.params.sun_dir[1], frame.params.sun_dir[2]];
        for (name, clip) in [("round-9 slice", false), ("round-10 clip", true)] {
            let mut prev: Vec<u64> = Vec::new();
            let mut hits = [0usize; 4];
            for k in 0..sim_n {
                let (d, fwd) = pose(k);
                let (_, cells, _) = cascades(view.cam, d, fwd, sun, view.shadow_res, clip);
                let keys = cache_keys_for(&cells, view.shadow_res, geo, sun3);
                for c in 2..4 {
                    if k > 0 && keys[c] == prev[c] {
                        hits[c] += 1;
                    }
                }
                prev = keys;
            }
            eprintln!("path sim {name}: {sim_n} frames, far-cascade hit rate c2 {:.1} % c3 {:.1} %", 100.0 * hits[2] as f64 / (sim_n - 1).max(1) as f64, 100.0 * hits[3] as f64 / (sim_n - 1).max(1) as f64);
        }
        let no_cache = cache_keys.is_empty();
        let aspect = W as f32 / H as f32;
        let mut rows = Vec::new();
        for k in 0..np {
            let (d, fwd) = pose(k);
            let (shadows, cells, slices) = cascades(view.cam, d, fwd, sun, view.shadow_res, shadow_clip());
            let keys = if no_cache { Vec::new() } else { cache_keys_for(&cells, view.shadow_res, geo, sun3) };
            let (vp, inv) = camera_matrices(fwd, aspect);
            let tr = Mat4::from_translation(-d);
            let mut f = frame;
            f.params.view_proj = (vp * tr).to_cols_array_2d();
            f.params.inv_dir = inv.to_cols_array_2d();
            f.shadows = &shadows;
            f.shadow_slices = &slices;
            f.shadow_cache_keys = &keys;
            if let Some(set) = f.instanced.as_mut() {
                set.cull_view_proj = render_ash::cull::vk_perspective(FOV, aspect, 0.5, 40000.0) * render_ash::cull::look_at(Vec3::ZERO, fwd, Vec3::Y) * tr;
            }
            let (_, t) = b.render_quality(&f).expect("path frame");
            rows.push((t.gpu_ms, t.shadow_ms, t.cached_cascades));
        }
        let n = rows.len().max(1) as f64;
        let mean = |i: usize| rows.iter().map(|r| [r.0, r.1][i]).sum::<f64>() / n;
        let cached: u32 = rows.iter().map(|r| r.2).sum();
        eprintln!("path frames gpu ms: {:?}", rows.iter().map(|r| r.0.round()).collect::<Vec<_>>());
        eprintln!(
            "path render: {} frames, mean gpu frame {:.1} ms, mean shadow {:.1} ms, far cascades cached {} of {} ({:.1} %), mode {}",
            rows.len(),
            mean(0),
            mean(1),
            cached,
            2 * rows.len(),
            100.0 * cached as f64 / (2.0 * n),
            if no_cache { "cold (SHADOW_CACHE=0)" } else if shadow_clip() { "clip" } else { "round-9 slice" }
        );
    }
    runs.sort_by(|a, b| a.1.gpu_ms.total_cmp(&b.1.gpu_ms));
    let (px, t, wall_ms) = runs.swap_remove(nf / 2);
    let img = downsample(&px, W * SS, H * SS, SS, layer().is_some());
    // QUALITY_OUT: output path (the descent writes its layers next to its frames)
    let png = std::env::var("QUALITY_OUT").unwrap_or(format!("out/quality/{scene}.png"));
    write_png(&png, W, H, &img);
    let (e1, w1) = message_counts();
    let json = format!(
        "{{\"scene\":\"{scene}\",\"device\":\"{}\",\"render\":\"{}x{} -> {}x{} (2x2 SSAA)\",\"triangles\":{},\"voxel_triangles\":{},\"ring_triangles\":{},\"bricks_planned\":{},\"bricks_skipped_by_bounds\":{},\"bricks_sampled\":{},\"bricks_meshed\":{},\"lod_triangles\":{:?},\"islands_removed\":{},\"island_triangles_removed\":{},\"interior_open_edges\":{},\"topology_clean\":{},\"sample_ms\":{:.0},\"gpu_mesh_wall_ms\":{:.0},\"attributes_ms\":{:.0},\"build_ms\":{:.0},\"gpu_shadow_ms\":{:.1},\"gpu_main_ms\":{:.1},\"gpu_frame_ms\":{:.1},\"frame_wall_ms\":{:.1},\"validation_errors\":{},\"validation_warnings\":{},\"camera\":[{:.1},{:.1},{:.1}],\"forward\":[{:.3},{:.3},{:.3}],\"veg_instances\":{},\"veg_visible_gpu\":{},\"veg_impostor_draws\":{},\"veg_crossfade_draws\":{},\"veg_view_draws\":{:?},\"veg_view_triangles\":{:?},\"shadow_terrain_triangles\":{:?},\"veg_counts_tree_boulder_tuft_shrub\":{:?},\"veg_triangles\":{},\"veg_build_ms\":{:.0},\"png\":\"{png}\"}}",
        b.ctx.device_name,
        W * SS,
        H * SS,
        W,
        H,
        indices.len() / 3,
        st.voxel_tris,
        st.ring_tris,
        st.bricks_planned,
        st.bricks_skipped,
        st.bricks_sampled,
        st.bricks_meshed,
        st.lods,
        st.islands_removed,
        st.island_tris_removed,
        st.interior_open_edges,
        st.topo_clean,
        st.sample_ms,
        st.gpu_mesh_ms,
        st.attr_ms,
        build_ms,
        t.shadow_ms,
        t.main_ms,
        t.gpu_ms,
        wall_ms,
        e1 - e0,
        w1 - w0,
        view.cam.x,
        view.cam.y,
        view.cam.z,
        view.fwd.x,
        view.fwd.y,
        view.fwd.z,
        view.veg.as_ref().map_or(0, |v| v.0.instances.len()),
        t.visible_instances,
        t.impostor_instances,
        t.fading_instances,
        t.view_instances,
        t.view_triangles,
        t.shadow_terrain_triangles,
        view.veg.as_ref().map_or([0; 4], |v| v.0.counts),
        view.veg.as_ref().map_or(0, |v| v.0.triangles),
        view.veg_ms
    );
    if std::env::var_os("QUALITY_MESH_STATS").is_some() {
        let mut v: Vec<(usize, (u32, u64))> = t.camera_mesh_draws.iter().copied().enumerate().filter(|x| x.1.0 > 0).collect();
        v.sort_by_key(|x| std::cmp::Reverse(x.1.1));
        for (m, (n, tri)) in v.iter().take(16) {
            eprintln!("camera mesh {m:3}: {n:7} draws {tri:9} tris");
        }
    }
    for (l, ms) in &t.breakdown {
        eprintln!("gpu {l:>22}: {ms:8.1} ms");
    }
    println!("{json}");
    std::fs::write(png.strip_suffix(".png").map_or(format!("out/quality/{scene}.json"), |b| format!("{b}.json")), json + "\n").unwrap();
}
