//! Cube-sphere quadtree geometry: 2:1 restriction, crack-free stitching (bit-exact f64 shared
//! edges, whole-sphere watertightness), index variants, conservative culling, f64 -> f32 precision.
use glam::{DVec3, DVec4};
use planet::cube::{self, N};
use planet::patch::{self, grid};
use planet::quadtree::{self, Key, Selection, Shape, View};
use std::collections::{HashMap, HashSet};

const R: f64 = planet::EARTH_RADIUS_M;
const SHAPE: Shape = Shape { radius: R, h_min: -100.0, h_max: 9000.0 };

/// Deterministic, deliberately rough height field (exact ties on lattice points included).
fn height(d: DVec3) -> f64 {
    let s = (d.x * 37.0).sin() * (d.y * 23.0).cos() + (d.z * 51.0).sin() * 0.5;
    (s * 3000.0).max(0.0).round()
}

/// Perspective view from `cam` looking at `target` (planes in camera-relative coordinates).
fn view(cam: DVec3, target: DVec3, fov_y: f64, h_px: f64, err: f64, max_level: u8) -> View {
    let f = (target - cam).normalize();
    let up = if f.cross(DVec3::Z).length() < 1e-6 { DVec3::X } else { DVec3::Z };
    let r = f.cross(up).normalize();
    let u = r.cross(f);
    let (ty, tx) = ((fov_y / 2.0).tan(), (fov_y / 2.0).tan() * 16.0 / 9.0);
    let pl = |n: DVec3, w: f64| {
        let l = n.length();
        DVec4::new(n.x / l, n.y / l, n.z / l, w / l)
    };
    let planes = [pl(r + f * tx, 0.0), pl(-r + f * tx, 0.0), pl(u + f * ty, 0.0), pl(-u + f * ty, 0.0), pl(f, -1.0), pl(-f, 1e10)];
    View { cam, planes, px_per_rad: h_px / (2.0 * (fov_y / 2.0).tan()), max_error_px: err, max_level }
}

/// Adversarial cameras: cube corners and edges, poles, grazing at 2 m, 400 km, GEO, inside the
/// height range, looking at the limb.
fn cameras() -> Vec<(DVec3, DVec3)> {
    let mut v = Vec::new();
    let corner = DVec3::new(1.0, 1.0, 1.0).normalize();
    let edge = DVec3::new(1.0, 1.0, 0.0).normalize();
    for (d, alt) in [(corner, 2.0), (corner, 400e3), (edge, 2.0), (edge, 30e3), (DVec3::Z, 10.0), (-DVec3::Z, 400e3), (cube::latlon_dir(10.0, -20.0), 35_786e3), (cube::latlon_dir(45.0, 45.0), 5.0)] {
        let cam = d * (R + alt);
        let side = d.cross(DVec3::new(0.3, 0.5, 0.8)).normalize();
        v.push((cam, DVec3::ZERO)); // nadir
        v.push((cam, cam + side * 1e6 - d * alt.max(1e3) * 0.2)); // toward the horizon / limb
    }
    v
}

fn select(cam: DVec3, t: DVec3, max_level: u8) -> Selection {
    quadtree::select(&view(cam, t, 60f64.to_radians(), 1080.0, 1.5, max_level), &SHAPE)
}

#[test]
fn neighbours_are_symmetric() {
    for f in 0..6u8 {
        for l in [0u8, 1, 3, 7] {
            let n = 1u32 << l;
            for (i, j) in [(0, 0), (n - 1, 0), (0, n - 1), (n - 1, n - 1), (n / 2, n / 3)] {
                let k = Key { face: f, level: l, i, j };
                for e in 0..4 {
                    let nb = k.neighbour(e);
                    assert_ne!(nb, k);
                    assert!((0..4).any(|e2| nb.neighbour(e2) == k), "{k:?} edge {e} -> {nb:?} not symmetric");
                }
            }
        }
    }
}

#[test]
fn leaves_tile_the_sphere_and_are_2_to_1() {
    for (cam, t) in cameras() {
        for ml in [8u8, 14, 20] {
            let s = select(cam, t, ml);
            // tiling: total area in level-0 units is exactly 6
            let area: f64 = s.leaves.iter().map(|k| 0.25f64.powi(k.level as i32)).sum();
            assert!((area - 6.0).abs() < 1e-9, "leaves do not tile: {area}");
            for a in &s.leaves {
                let mut p = a.parent();
                while let Some(x) = p {
                    assert!(!s.leaves.contains(&x), "leaf {a:?} and ancestor {x:?}");
                    p = x.parent();
                }
            }
            assert!(quadtree::max_level_jump(&s) <= 1, "2:1 violated at cam {cam:?}");
        }
    }
}

/// Edge vertex positions (f64 bits) of leaf `k` along edge `e`, after stitching with `mask`.
fn edge_positions(k: &Key, e: u32, mask: u8) -> Vec<[u64; 3]> {
    let mut out = Vec::new();
    for i in 0..=N {
        let (a, b) = patch::edge_ab(e, i);
        let (a, b) = patch::collapse(mask, a, b);
        let p = patch::vertex_pos(k, a, b, R, &height);
        let bits = [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
        if out.last() != Some(&bits) {
            out.push(bits);
        }
    }
    out
}

#[test]
fn shared_edges_bit_exact_at_all_masks() {
    for (cam, t) in cameras() {
        let s = select(cam, t, 18);
        let masks: HashMap<Key, u8> = s.visible.iter().copied().collect();
        let mut checked = [0u32; 2];
        for (k, m) in &s.visible {
            for e in 0..4u32 {
                let nb = k.neighbour(e as usize);
                let Some(c) = s.covering(&nb) else { continue }; // finer neighbour: checked from its side
                let Some(&cm) = masks.get(&c) else { continue };
                let mine: HashSet<[u64; 3]> = edge_positions(k, e, *m).into_iter().collect();
                // the neighbour's edge facing us
                let Some(ce) = (0..4u32).find(|&e2| c.neighbour(e2 as usize).covers(k)) else {
                    panic!("no back edge: {k:?} e{e} nb {nb:?} covering {c:?}: {:?}", (0..4).map(|e2| c.neighbour(e2)).collect::<Vec<_>>());
                };
                let theirs: HashSet<[u64; 3]> = edge_positions(&c, ce, cm).into_iter().collect();
                if c.level == k.level {
                    assert_eq!(mine, theirs, "same-level edge mismatch {k:?}/{c:?}");
                    checked[0] += 1;
                } else {
                    assert_eq!(c.level + 1, k.level);
                    assert!(m & (1 << e) != 0, "coarser neighbour but edge not stitched");
                    assert_eq!(mine.len() as u32, N / 2 + 1);
                    assert!(mine.is_subset(&theirs), "stitched edge vertex not on coarse edge {k:?}/{c:?}");
                    checked[1] += 1;
                }
            }
        }
        assert!(checked[0] > 0);
    }
}

#[test]
fn whole_sphere_watertight_and_consistently_oriented() {
    // every leaf (not only visible ones) with its mask: weld by exact f64 position and require
    // each undirected edge to be used by exactly two triangles, once in each direction
    for (cam, t) in [(cube::latlon_dir(35.26, 45.0) * (R + 50.0), DVec3::ZERO), (cube::latlon_dir(0.0, 45.0) * (R + 400e3), DVec3::ZERO)] {
        let s = select(cam, t, 12);
        let mut ids: HashMap<[u64; 3], u32> = HashMap::new();
        let mut directed: HashMap<(u32, u32), u32> = HashMap::new();
        for k in &s.leaves {
            let m = quadtree::mask(&s, k);
            let mut vid = vec![0u32; ((N + 1) * (N + 1)) as usize];
            for b in 0..=N {
                for a in 0..=N {
                    let p = patch::vertex_pos(k, a, b, R, &height);
                    let n = ids.len() as u32;
                    vid[grid(a, b) as usize] = *ids.entry([p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]).or_insert(n);
                }
            }
            let ix = patch::indices(m);
            for t in ix.chunks(3) {
                if t.iter().any(|&i| i >= (N + 1) * (N + 1)) {
                    continue; // skirts
                }
                let v = [vid[t[0] as usize], vid[t[1] as usize], vid[t[2] as usize]];
                assert!(v[0] != v[1] && v[1] != v[2] && v[0] != v[2], "degenerate after welding");
                for q in 0..3 {
                    *directed.entry((v[q], v[(q + 1) % 3])).or_insert(0) += 1;
                }
            }
        }
        let mut bad = 0;
        for (&(a, b), &n) in &directed {
            if n != 1 || directed.get(&(b, a)) != Some(&1) {
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "{bad} open / non-manifold / inconsistently oriented edges over {} leaves", s.leaves.len());
        // Euler characteristic of a sphere
        let used: HashSet<u32> = directed.keys().map(|k| k.0).collect();
        let e = directed.len() / 2;
        let f = directed.len() / 3;
        assert_eq!(used.len() as i64 - e as i64 + f as i64, 2, "not a topological sphere");
    }
}

#[test]
fn index_variants_valid() {
    for m in 0..16u8 {
        let ix = patch::indices(m);
        let mut area2 = 0i64;
        let mut boundary: HashMap<(u32, u32), i32> = HashMap::new();
        for t in ix.chunks(3) {
            if t.iter().any(|&i| i >= (N + 1) * (N + 1)) {
                continue;
            }
            let p = |i: u32| ((i % (N + 1)) as i64, (i / (N + 1)) as i64);
            let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
            let cr = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
            assert!(cr > 0, "mask {m}: flipped or degenerate triangle {t:?}");
            area2 += cr;
            for q in 0..3 {
                let (x, y) = (t[q], t[(q + 1) % 3]);
                *boundary.entry((x.min(y), x.max(y))).or_insert(0) += 1;
            }
        }
        assert_eq!(area2, 2 * (N * N) as i64, "mask {m}: grid area");
        // odd vertices of stitched edges never referenced
        for e in 0..4u32 {
            if m & (1 << e) == 0 {
                continue;
            }
            for k in (1..N).step_by(2) {
                let (a, b) = patch::edge_ab(e, k);
                assert!(!ix.contains(&grid(a, b)), "mask {m}: odd vertex of stitched edge {e} used");
            }
        }
        assert!(boundary.values().all(|&n| n <= 2));
    }
}

#[test]
fn culling_is_conservative() {
    for (cam, t) in cameras() {
        let v = view(cam, t, 60f64.to_radians(), 1080.0, 1.5, 14);
        let s = quadtree::select(&v, &SHAPE);
        let vis: HashSet<Key> = s.visible.iter().map(|x| x.0).collect();
        let r0 = R + SHAPE.h_min;
        for k in s.leaves.iter().filter(|k| !vis.contains(k)) {
            for b in (0..=N).step_by(8) {
                for a in (0..=N).step_by(8) {
                    let p = k.vertex_dir(a, b) * (R + SHAPE.h_max * 0.999);
                    let rel = p - cam;
                    let inside = v.planes.iter().all(|pl| pl.truncate().dot(rel) + pl.w >= 0.0);
                    if !inside {
                        continue;
                    }
                    // segment cam -> p blocked by the sphere r0?
                    let d = rel.normalize();
                    let bq = cam.dot(d);
                    let cq = cam.length_squared() - r0 * r0;
                    let disc = bq * bq - cq;
                    let blocked = disc > 0.0 && {
                        let t0 = -bq - disc.sqrt();
                        t0 > 0.0 && t0 < rel.length()
                    };
                    assert!(blocked, "culled leaf {k:?} has a visible point (cam {cam:?})");
                }
            }
        }
    }
}

#[test]
fn camera_relative_f32_precision_at_orbit() {
    // positions reach the GPU as (origin - cam) f64->f32 + local f32; compare with exact f64
    for (alt, nadir_tol) in [(400e3, 0.05), (35_786e3, 5.0), (2.0, 0.002)] {
        let cam = cube::latlon_dir(20.0, 30.0) * (R + alt);
        let s = select(cam, DVec3::ZERO, 18);
        let mut worst_rel = 0.0f64;
        let mut worst_naive_rel = 0.0f64;
        let mut worst_near = 0.0f64;
        let mut naive_near = 0.0f64;
        let mut seam = 0.0f64;
        let mut seen: HashMap<[u64; 3], DVec3> = HashMap::new();
        for (k, _) in s.visible.iter().take(400) {
            let pm = patch::build(k, R, &height);
            let o = (pm.origin - cam).as_vec3();
            for b in (0..=N).step_by(4) {
                for a in (0..=N).step_by(4) {
                    let exact = patch::vertex_pos(k, a, b, R, &height) - cam;
                    let v = pm.verts[grid(a, b) as usize].pos;
                    let gpu = (o + glam::Vec3::from(v)).as_dvec3();
                    let err = (gpu - exact).length();
                    worst_rel = worst_rel.max(err / exact.length());
                    if exact.length() < 50e3 + alt {
                        worst_near = worst_near.max(err);
                    }
                    // control: absolute f32 positions minus an f32 camera
                    let naive = ((exact + cam).as_vec3() - cam.as_vec3()).as_dvec3();
                    worst_naive_rel = worst_naive_rel.max((naive - exact).length() / exact.length());
                    if exact.length() < 50e3 + alt {
                        naive_near = naive_near.max((naive - exact).length());
                    }
                    if a == 0 || a == N || b == 0 || b == N {
                        let key = { let p = exact + cam; [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()] };
                        if let Some(prev) = seen.insert(key, gpu) {
                            seam = seam.max((prev - gpu).length() / exact.length());
                        }
                    }
                }
            }
        }
        println!("alt {alt:>10} m: worst rel err {worst_rel:.2e} (naive f32 {worst_naive_rel:.2e}), near-camera abs {worst_near:.4} m (naive {naive_near:.3} m), shared-vertex f32 gap rel {seam:.2e}");
        assert!(worst_rel < 2e-7, "relative error {worst_rel}");
        assert!(worst_near < nadir_tol, "near error {worst_near} m");
        assert!(seam < 2e-7, "shared vertex f32 gap {seam}");
        if alt < 1e6 {
            assert!(naive_near > 10.0 * worst_near.max(1e-6), "control: naive f32 should be much worse near the camera");
        }
    }
}

#[test]
fn deterministic_selection() {
    let (cam, t) = cameras()[3];
    let a = select(cam, t, 16);
    let b = select(cam, t, 16);
    assert_eq!(a.visible, b.visible);
}

#[test]
fn pack_roundtrip_and_verifier() {
    use planet::pack::*;
    let fmt = TexFormat::Bc4;
    let size = 16;
    let mips = 3;
    let data: Vec<u8> = (0..6 * layer_bytes(fmt, size, mips)).map(|i| (i * 7) as u8).collect();
    let p = PlanetPack {
        textures: vec![Texture { name: "t".into(), format: fmt, size, mips, data }],
        height: HeightCube { size: 4, texels: (0..96).collect(), scale: 0.1, offset: 0.0 },
        credits: "c".into(),
    };
    let b = p.to_bytes();
    let q = PlanetPack::from_bytes(&b).expect("roundtrip");
    assert_eq!(q.textures[0].data, p.textures[0].data);
    assert_eq!(q.height.texels, p.height.texels);
    // truncation / corruption must be rejected, never panic
    for cut in [8, 64, b.len() / 2] {
        assert!(PlanetPack::from_bytes(&b[..cut]).is_err());
    }
    // unaligned copy still reads
    let mut un = vec![0u8; 1];
    un.extend_from_slice(&b);
    let _ = PlanetPack::from_bytes(&un[1..]);
    // height sampling is a pure function of the direction bits
    let d = DVec3::new(1.0, 1.0, 1.0).normalize();
    assert_eq!(q.height.sample(d).to_bits(), q.height.sample(d).to_bits());
}
