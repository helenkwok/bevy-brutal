//! Acceptance tests for C2 (near-field voxel bricks) under the robustness hard rule.
use coords::{BrickCoord, BrickGrid};
use voxel::scene::{Adversarial, BlobScene, CaveScene, Plane};
use voxel::*;

fn cfg(n: u32, v: f64) -> VoxelConfig {
    VoxelConfig::new(BrickGrid::new(n, v).unwrap())
}

/// Region (LOD0 lattice) aligned to `align` lattice units, covering +-`m` metres.
fn region(c: &VoxelConfig, m: f64, align: i64) -> ([i64; 3], [i64; 3]) {
    let r = ((m / c.grid.voxel_m()).ceil() as i64 + align - 1) / align * align;
    ([-r; 3], [r; 3])
}

fn blob() -> BlobScene {
    BlobScene { radius: 3.0, seed: 7 }
}

/// Fraction of triangles whose geometric normal agrees with the interpolated vertex normals.
fn orientation_agreement(w: &VoxelWorld) -> f64 {
    let (mut good, mut total) = (0usize, 0usize);
    for m in w.bricks.values() {
        for t in m.indices.as_chunks::<3>().0 {
            let [a, b, c] = t.map(|i| i as usize);
            let n = tri_normal(m.ipos[a], m.ipos[b], m.ipos[c]);
            let g =
                [0, 1, 2].map(|k| (m.vertices[a].normal[k] + m.vertices[b].normal[k] + m.vertices[c].normal[k]) as f64);
            total += 1;
            if n[0] as f64 * g[0] + n[1] as f64 * g[1] + n[2] as f64 * g[2] > 0.0 {
                good += 1;
            }
        }
    }
    good as f64 / total as f64
}

fn assert_closed(w: &VoxelWorld, what: &str) -> Topology {
    let t = topology(w.bricks.values());
    assert!(t.is_closed(), "{what}: {t:?}");
    assert!(t.triangles > 0, "{what}: empty");
    t
}

#[test]
fn a1_watertight_uniform_lod_all_brick_sizes() {
    for n in [16, 32, 64] {
        for lod in [0u8, 1] {
            let c = cfg(n, 0.125);
            let w = build_world(&blob(), &c, &UniformLod(lod), region(&c, 5.0, n as i64 * 2), lod).unwrap();
            assert!(w.bricks.len() > 1 || n == 64, "n={n}: want a multi-brick case");
            let t = assert_closed(&w, &format!("n={n} lod={lod}"));
            assert!(t.triangles > 1000);
        }
    }
}

#[test]
fn a2_watertight_across_slab_seams_every_face() {
    for n in [16u32, 32] {
        let c = cfg(n, 0.25);
        for axis in 0..3 {
            for (lb, la) in [(0u8, 1u8), (1, 0), (1, 2), (2, 1)] {
                let lay = SlabLayout { axis, split: 0, lod_below: lb, lod_above: la };
                let w = build_world(&blob(), &c, &lay, region(&c, 5.0, n as i64 * 4), 2).unwrap();
                let lods: std::collections::BTreeSet<u8> = w.bricks.keys().map(|b| b.lod).collect();
                assert_eq!(lods.len(), 2, "both LODs must carry surface");
                assert_closed(&w, &format!("n={n} axis={axis} {lb}|{la}"));
            }
        }
    }
}

/// Camera-centred shells: the LOD boxes' faces, edges and corners all cut the surface, so coarse
/// bricks touch finer ones across faces, along edges only and at corners only.
#[test]
fn a3_watertight_camera_centred_shells() {
    let c = cfg(16, 0.25);
    let mut kinds = [0usize; 3]; // coarse bricks with finer neighbours via face / edge-only / corner-only
    for cam in [[0i64, 0, 0], [5, -3, 7], [-20, 11, 2], [13, 13, 13]] {
        for half in [vec![1i64], vec![1, 1]] {
            let big = BlobScene { radius: if half.len() == 1 { 9.0 } else { 20.0 }, seed: 3 };
            let lay = ShellLayout::camera_centred(&c.grid, cam, &half);
            let w = build_world(&big, &c, &lay, region(&c, big.radius * 1.5, 16 << half.len()), lay.max_lod()).unwrap();
            let lods: std::collections::BTreeSet<u8> = w.bricks.keys().map(|b| b.lod).collect();
            assert_eq!(lods.len(), half.len() + 1, "cam {cam:?} shells {half:?}: every LOD carries surface ({lods:?})");
            assert_closed(&w, &format!("cam {cam:?} shells {half:?}"));
            for b in w.bricks.keys() {
                let (mask, finer) = plan_brick(&c.grid, &lay, *b).unwrap();
                if let Some(r) = finer {
                    let o = c.grid.brick_origin(*b);
                    let ext = 16i64 << b.lod;
                    // classify by how many axes the brick overlaps the finer box's range in
                    let overlap = (0..3).filter(|&i| o[i] < r.hi[i] && o[i] + ext > r.lo[i]).count();
                    let touching = (0..3).all(|i| o[i] <= r.hi[i] && o[i] + ext >= r.lo[i]);
                    if touching {
                        match overlap {
                            2 => kinds[0] += (mask != 0) as usize,
                            1 => kinds[1] += 1,
                            0 => kinds[2] += 1,
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    assert!(kinds.iter().all(|&k| k > 0), "face / edge-only / corner-only cases all exercised: {kinds:?}");
}

#[test]
fn a4_negative_control_without_seam_handling_cracks() {
    let c = cfg(16, 0.25);
    let lay = ShellLayout::camera_centred(&c.grid, [0, 0, 0], &[1]);
    let w = build_world(&BlobScene { radius: 9.0, seed: 3 }, &c, &lay, region(&c, 14.0, 32), 1).unwrap();
    let mut no_transitions = Vec::new();
    let mut no_phi = Vec::new();
    for b in w.bricks.keys() {
        let (mask, finer) = plan_brick(&c.grid, &lay, *b).unwrap();
        let s = sample_brick(&BlobScene { radius: 9.0, seed: 3 }, &c, *b, mask).unwrap();
        no_transitions.push(mesh_brick(&c, &s, 0, finer));
        no_phi.push(mesh_brick(&c, &s, mask, None));
    }
    assert!(topology(no_transitions.iter()).boundary_edges > 0, "no transition cells must crack");
    assert!(topology(no_phi.iter()).boundary_edges > 0, "no displacement must crack");
    // the exact "flipped" criterion must fire: reversed winding everywhere -> inward volume,
    // one reversed triangle -> orientation conflicts.
    let mut all: Vec<BrickMesh> = w.bricks.values().cloned().collect();
    let good = topology(all.iter());
    assert!(good.is_closed() && good.flipped == 0, "{good:?}");
    for m in &mut all {
        for t in m.indices.chunks_mut(3) {
            t.swap(1, 2);
        }
    }
    let inv = topology(all.iter());
    assert!(inv.flipped > 0 && inv.flipped == inv.axis_reference_triangles, "{inv:?}");
    let first = all.iter_mut().find(|m| !m.indices.is_empty()).unwrap();
    first.indices.swap(1, 2);
    assert!(topology(all.iter()).orientation_conflicts > 0);
}

#[test]
fn a5_no_degenerate_or_flipped_and_normals_agree() {
    let c = cfg(16, 0.25);
    for lay in [
        Box::new(UniformLod(0)) as Box<dyn LodLayout>,
        Box::new(SlabLayout { axis: 1, split: 0, lod_below: 1, lod_above: 0 }),
        Box::new(ShellLayout::camera_centred(&c.grid, [3, 3, 3], &[1, 1])),
    ] {
        let w = build_world(&BlobScene { radius: 9.0, seed: 3 }, &c, lay.as_ref(), region(&c, 14.0, 64), 2).unwrap();
        let t = topology(w.bricks.values());
        assert!(t.is_closed() && t.closed_components > 0 && t.unreferenced_components == 0, "{t:?}");
        assert_eq!(t.negative_volume_components, 0, "solid blob: outward volume {t:?}");
        let a = orientation_agreement(&w);
        assert!(a > 0.995, "normal agreement {a}");
    }
}

#[test]
fn a6_deterministic_across_runs_and_thread_counts() {
    let c = cfg(16, 0.25);
    let lay = ShellLayout::camera_centred(&c.grid, [3, -2, 1], &[1, 1]);
    let r = region(&c, 14.0, 64);
    let f = BlobScene { radius: 9.0, seed: 3 };
    let a = world_hash(&build_world(&f, &c, &lay, r, 2).unwrap());
    let b = world_hash(&build_world(&f, &c, &lay, r, 2).unwrap());
    let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let s = one.install(|| world_hash(&build_world(&f, &c, &lay, r, 2).unwrap()));
    assert_eq!(a, b);
    assert_eq!(a, s);
}

/// Fuzz (hard rule): adversarial lattice densities (exact zeros = ties at cell corners, ±1
/// quantum, sub-quantum, 1e3 vs 1e-9 ratios), random per lattice point, across uniform, slab and
/// shell layouts. Every case must be closed, with no degenerate or flipped triangles, and deterministic.
#[test]
fn a7_fuzz_adversarial_densities() {
    let c = cfg(16, 0.25);
    let mut cases = 0;
    let mut tris = 0;
    for seed in 0..24u64 {
        let f = Adversarial { seed, voxel_m: 0.25, radius_m: 7.0 };
        let cam = [(seed as i64 % 5) * 7 - 14, (seed as i64 % 3) * 9 - 9, (seed as i64 % 7) * 4 - 12];
        let layouts: Vec<Box<dyn LodLayout>> = vec![
            Box::new(UniformLod(0)),
            Box::new(UniformLod(1)),
            Box::new(SlabLayout {
                axis: (seed % 3) as usize,
                split: 0,
                lod_below: (seed % 2) as u8,
                lod_above: 1 - (seed % 2) as u8,
            }),
            Box::new(ShellLayout::camera_centred(&c.grid, cam, &[1])),
            Box::new(ShellLayout::camera_centred(&c.grid, cam, &[1, 1])),
        ];
        for (li, lay) in layouts.iter().enumerate() {
            let w = build_world(&f, &c, lay.as_ref(), region(&c, 10.0, 64), 2).unwrap();
            let t = topology(w.bricks.values());
            assert!(t.is_closed() && t.open_components == 0, "seed {seed} layout {li}: {t:?}");
            if li < 2 {
                // regular cells only: the per-triangle exact edge-direction check must also hold
                assert_eq!(t.edge_dir_disagreements, 0, "seed {seed} layout {li}: {t:?}");
            }
            tris += t.triangles;
            if li == 4 {
                let w2 = build_world(&f, &c, lay.as_ref(), region(&c, 10.0, 64), 2).unwrap();
                assert_eq!(world_hash(&w), world_hash(&w2), "seed {seed}: determinism");
            }
            cases += 1;
        }
    }
    assert_eq!(cases, 120);
    assert!(tris > 1_000_000, "fuzz must produce substantial geometry, got {tris}");
}

#[test]
fn h2_exact_zero_samples_are_outside() {
    // y = 1.0 is a lattice plane at 0.25 m: every sample on it is exactly 0.
    let c = cfg(16, 0.25);
    let w = build_world(&Plane { h: 1.0 }, &c, &UniformLod(0), ([-32, -16, -32], [32, 48, 32]), 0).unwrap();
    let t = topology(w.bricks.values());
    assert!(t.triangles > 0 && t.is_clean() && t.non_manifold_edges == 0 && t.orientation_conflicts == 0, "{t:?}");
    for m in w.bricks.values() {
        for &p in &m.ipos {
            // just below y = 1.0 (t clamped to 255/256 of the edge), never on or above it
            assert!(p[1] < 4 * SUB && p[1] > 3 * SUB, "{p:?}");
        }
    }
    struct NegZero;
    impl DensityField for NegZero {
        fn density(&self, p: [f64; 3]) -> f64 {
            if p[1] == 1.0 { -0.0 } else { p[1] - 1.0 }
        }
    }
    let w2 = build_world(&NegZero, &c, &UniformLod(0), ([-32, -16, -32], [32, 48, 32]), 0).unwrap();
    assert_eq!(world_hash(&w), world_hash(&w2));
}

#[test]
fn h3_edge_parameter_is_exact_and_clamped() {
    assert_eq!(edge_t(-1, 1), 128);
    assert_eq!(edge_t(-1, 0), 255); // tie at the far corner: clamped inside the edge
    assert_eq!(edge_t(0, -1), 1);
    assert_eq!(edge_t(-DQ_MAX, 1), 255);
    assert_eq!(edge_t(1, -DQ_MAX), 1);
    // antisymmetry: the same edge seen from either end gives the mirrored parameter
    for (a, b) in [(-3, 5), (-700, 2), (-1, 1_000_000), (4, -9)] {
        assert_eq!(edge_t(a, b) + edge_t(b, a), 256, "{a} {b}");
    }
    // i32 bounds used by the GPU port: 256 * DQ_MAX * 2 + 2 * DQ_MAX < 2^31
    assert!(2 * T_STEPS * DQ_MAX as i64 + 2 * DQ_MAX as i64 * 2 < i32::MAX as i64);
}

#[test]
fn h6_non_finite_density_is_an_error() {
    struct Bad;
    impl DensityField for Bad {
        fn density(&self, p: [f64; 3]) -> f64 {
            if p == [1.0, 1.0, 1.0] { f64::NAN } else { p[1] - 0.6 }
        }
    }
    let c = cfg(16, 0.25);
    let e = build_world(&Bad, &c, &UniformLod(0), ([0; 3], [16; 3]), 0).err();
    assert_eq!(e, Some(VoxelError::NonFinite { at: [4, 4, 4] }));
}

#[test]
fn h8_unsupported_layouts_are_refused() {
    let c = cfg(16, 0.25);
    let jump = SlabLayout { axis: 0, split: 0, lod_below: 0, lod_above: 2 };
    assert!(matches!(build_world(&blob(), &c, &jump, region(&c, 5.0, 64), 2), Err(VoxelError::LodJump { .. })));
    // L-shaped (non-box) finer region
    struct LShape;
    impl LodLayout for LShape {
        fn lod_at(&self, p: [i64; 3]) -> u8 {
            if p[0] < 0 || p[2] < 0 { 0 } else { 1 }
        }
        fn finer_box(&self, _: u8) -> Option<Aabb> {
            None
        }
    }
    let b = BrickCoord { x: 0, y: 0, z: 0, lod: 1 };
    assert!(matches!(plan_brick(&c.grid, &LShape, b), Err(VoxelError::NotABox { .. })));
    // A finer box ending less than one cell from a face shared with a coarser brick: the seam
    // displacement would move that face's vertices, so it is refused.
    struct Pinched;
    impl LodLayout for Pinched {
        fn lod_at(&self, p: [i64; 3]) -> u8 {
            if p[0] < 32 && p[1] >= 32 {
                0
            } else if p[0] >= 32 {
                2
            } else {
                1
            }
        }
        fn finer_box(&self, lod: u8) -> Option<Aabb> {
            match lod {
                1 => Some(Aabb { lo: [-BIG, 32, -BIG], hi: [32, BIG, BIG] }),
                _ => Some(Aabb { lo: [-BIG; 3], hi: [32, BIG, BIG] }),
            }
        }
    }
    assert!(matches!(plan_brick(&c.grid, &Pinched, b), Err(VoxelError::PhiReachesCoarser { .. })));
    // ...whereas a one-brick margin is fine (Φ support is one cell, the brick is 16 cells wide)
    let ok = ShellLayout::camera_centred(&c.grid, [0, 0, 0], &[1, 1]);
    for b in bricks_in(&c.grid, &ok, ([-256; 3], [256; 3]), 2) {
        plan_brick(&c.grid, &ok, b).unwrap();
    }
}

#[test]
fn shell_layout_is_nested_aligned_and_has_margin() {
    let g = BrickGrid::new(32, 0.25).unwrap();
    let lay = ShellLayout::camera_centred(&g, [1000, -77, 5], &[2, 2, 1]);
    for (l, b) in lay.boxes.iter().enumerate() {
        let next_ext = 32i64 << (l + 1);
        assert!(
            b.lo.iter().chain(&b.hi).all(|v| v.rem_euclid(next_ext) == 0),
            "box {l} aligned to LOD{} bricks",
            l + 1
        );
        assert!(b.contains([1000, -77, 5]));
        if l > 0 {
            let p = lay.boxes[l - 1];
            let ext = 32i64 << l;
            assert!((0..3).all(|i| b.lo[i] <= p.lo[i] - ext && b.hi[i] >= p.hi[i] + ext), "margin");
        }
    }
}

#[test]
fn cave_scene_has_overhang_and_cave() {
    let c = cfg(32, 0.25);
    let s = CaveScene { seed: 1 };
    assert!(s.density([0.0, 9.2, 8.0]) < 0.0 && s.density([0.0, 6.0, 8.0]) > 0.0);
    assert!(s.density([1.5, 3.8, 0.5]) > 0.0 && s.density([1.5, 9.0, 0.5]) < 0.0);
    let w = build_world(&s, &c, &UniformLod(0), ([-64, -16, -48], [64, 56, 64]), 0).unwrap();
    let t = topology(w.bricks.values());
    assert!(t.is_clean() && t.non_manifold_edges == 0 && t.orientation_conflicts == 0, "{t:?}");
    assert!(w.bricks.len() < w.sampled, "sparse: some bricks empty");
}
