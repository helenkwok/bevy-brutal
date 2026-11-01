//! Quality-pass tests: the sawtooth root cause (C0 crease aliasing under marching cubes at
//! coarse LOD) and its fix (LOD-aware band-limited sampling), seam safety of that filtering,
//! and island removal.
use coords::BrickGrid;
use voxel::fields::Crease;
use voxel::*;

fn cfg(n: u32, v: f64) -> VoxelConfig {
    VoxelConfig::new(BrickGrid::new(n, v).unwrap())
}

/// Highest mesh surface point on the vertical line through (x, z), if any.
fn top_at(m: &BrickMesh, x: f32, z: f32) -> Option<f32> {
    let mut best: Option<f32> = None;
    for t in m.indices.as_chunks::<3>().0 {
        let [a, b, c] = [t[0], t[1], t[2]].map(|i| m.vertices[i as usize].pos);
        let d = (b[0] - a[0]) * (c[2] - a[2]) - (c[0] - a[0]) * (b[2] - a[2]);
        if d.abs() < 1e-12 {
            continue;
        }
        let u = ((x - a[0]) * (c[2] - a[2]) - (c[0] - a[0]) * (z - a[2])) / d;
        let v = ((b[0] - a[0]) * (z - a[2]) - (x - a[0]) * (b[2] - a[2])) / d;
        if u >= -1e-6 && v >= -1e-6 && u + v <= 1.0 + 1e-6 {
            let y = a[1] + u * (b[1] - a[1]) + v * (c[1] - a[1]);
            best = Some(best.map_or(y, |q| q.max(y)));
        }
    }
    best
}

/// Peak-to-peak variation of the meshed crest height along the (straight, level) crest line.
fn crest_zigzag(soften: bool, lod: u8) -> (f32, f32) {
    let c = cfg(16, 0.25);
    let f = Crease { h0: 6.0, slope: 1.0, angle: std::f64::consts::FRAC_PI_6, soften };
    let lay = UniformLod(lod);
    let r = 64 << lod;
    let region = ([-r, -16, -r], [r, 16 << (lod + 1), r]);
    let w =
        if soften { build_world_filtered(&f, &c, &lay, region, lod) } else { build_world(&f, &c, &lay, region, lod) }
            .unwrap();
    let m = w.merged();
    let (s, co) = std::f32::consts::FRAC_PI_6.sin_cos();
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for k in 0..400 {
        let t = -6.0 + 12.0 * k as f32 / 400.0;
        let y = top_at(&m, t * co, t * s).expect("crest covered");
        lo = lo.min(y);
        hi = hi.max(y);
    }
    (hi - lo, (lo + hi) / 2.0)
}

#[test]
fn q1_sawtooth_root_cause_is_c0_crease_aliasing_and_filtering_fixes_it() {
    let cell = |lod: u8| 0.25 * (1u32 << lod) as f32;
    for lod in [1u8, 2] {
        let (raw, _) = crest_zigzag(false, lod);
        let (filt, mid) = crest_zigzag(true, lod);
        eprintln!("lod {lod} cell {} m: raw crest zigzag {raw:.3} m, filtered {filt:.4} m (crest {mid:.2})", cell(lod));
        // the mechanism: a C0 crease diagonal to the lattice zigzags at cell frequency with an
        // amplitude that is a sizeable fraction of a cell - no seam or brick boundary involved
        assert!(raw > 0.15 * cell(lod), "raw crease should alias: {raw}");
        // the fix: crease softened to >= 3 cells at the sampled LOD
        assert!(filt < 0.1 * cell(lod) && raw > 5.0 * filt, "filtered crest should be smooth: {filt} vs raw {raw}");
    }
}

/// A field whose band-limited value depends violently on the sampling cell size: if two
/// bricks sampled a shared point at different filter levels, the mesh would crack.
struct LodDependent(f64);
impl DensityField for LodDependent {
    fn density(&self, p: [f64; 3]) -> f64 {
        self.density_lod(p, 0.25)
    }
    fn density_lod(&self, p: [f64; 3], cell_m: f64) -> f64 {
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        r - self.0 - 1.7 * cell_m.log2() * (p[0] * 0.9).sin() * (p[2] * 0.7).cos()
    }
}

#[test]
fn q2_filtered_sampling_is_watertight_across_lod_shells() {
    let c = cfg(16, 0.25);
    let mut all = std::collections::BTreeSet::new();
    for (cam, radius) in [([0i64, 0, 0], 19.0), ([7, -5, 3], 9.0), ([-30, 12, 21], 14.0), ([3, 3, -9], 19.0)] {
        let lay = ShellLayout::camera_centred(&c.grid, cam, &[1, 1]);
        let r = 96;
        let f = LodDependent(radius);
        let w = build_world_filtered(&f, &c, &lay, ([-r; 3], [r; 3]), lay.max_lod()).unwrap();
        let lods: std::collections::BTreeSet<u8> = w.bricks.keys().map(|b| b.lod).collect();
        assert!(lods.len() >= 2, "surface crosses an LOD transition: {lods:?}");
        all.extend(lods);
        let t = topology(w.bricks.values());
        assert!(t.is_closed() && t.is_clean(), "cam {cam:?}: {t:?}");
        // and unfiltered sampling of the same field is a different (also closed) surface
        assert_ne!(world_hash(&w), world_hash(&build_world(&f, &c, &lay, ([-r; 3], [r; 3]), lay.max_lod()).unwrap()));
    }
    assert_eq!(all.len(), 3, "LOD 0, 1, 2 all exercised: {all:?}");
    // filter level is the coarsest LOD touching the point (pure function of the point)
    let lay = SlabLayout { axis: 0, split: 0, lod_below: 0, lod_above: 1 };
    assert_eq!(filter_level(&lay, [0, 5, 5]), 1);
    assert_eq!(filter_level(&lay, [-1, 5, 5]), 0);
}

struct GroundAndRock;
impl DensityField for GroundAndRock {
    fn density(&self, p: [f64; 3]) -> f64 {
        let ground = p[1] + 0.3 * (p[0] * 0.4).sin();
        let rock = ((p[0] - 2.0).powi(2) + (p[1] - 6.0).powi(2) + p[2].powi(2)).sqrt() - 1.6; // floating
        // sealed air bubbles; the largest closed cavity is kept (it may hold the camera)
        let big = 2.0 - ((p[0] + 3.0).powi(2) + (p[1] + 5.0).powi(2) + p[2].powi(2)).sqrt();
        let small = 1.0 - ((p[0] - 4.0).powi(2) + (p[1] + 4.0).powi(2) + (p[2] - 3.0).powi(2)).sqrt();
        ground.min(rock).max(big).max(small)
    }
}

#[test]
fn q3_island_removal_drops_floating_rock_and_sealed_bubbles_only() {
    let c = cfg(16, 0.25);
    let r = 48;
    let w = build_world(&GroundAndRock, &c, &UniformLod(0), ([-r; 3], [r; 3]), 0).unwrap();
    let mut m = w.merged();
    let before = topology([&m]);
    assert!(before.closed_components >= 2, "{before:?}");
    let st = remove_islands(&mut m, 8);
    let after = topology([&m]);
    assert_eq!(before.closed_components, 3, "{before:?}");
    assert_eq!(st.removed_components, 2, "rock + small bubble: {st:?}");
    assert_eq!((after.closed_components, after.negative_volume_components), (1, 1), "{after:?}");
    assert!(after.open_components >= 1 && after.is_clean() && after.non_manifold_edges == 0);
    assert!(m.vertices.iter().all(|v| v.pos[1] < 2.0), "floating rock gone");
}

/// Negative control for q2: filtering each brick at its *own* LOD (instead of the per-point
/// `filter_level`) makes shared samples disagree, and the mesh cracks at LOD transitions.
#[test]
fn q2b_negative_control_per_brick_filtering_cracks() {
    struct AtCell<'a>(&'a LodDependent, f64);
    impl DensityField for AtCell<'_> {
        fn density(&self, p: [f64; 3]) -> f64 {
            self.0.density_lod(p, self.1)
        }
    }
    let c = cfg(16, 0.25);
    let lay = ShellLayout::camera_centred(&c.grid, [0, 0, 0], &[1, 1]);
    let f = LodDependent(19.0);
    let r = 96;
    let mut meshes = Vec::new();
    for b in bricks_in(&c.grid, &lay, ([-r; 3], [r; 3]), lay.max_lod()) {
        let (mask, finer) = plan_brick(&c.grid, &lay, b).unwrap();
        let smp = sample_brick(&AtCell(&f, 0.25 * (1u32 << b.lod) as f64), &c, b, mask).unwrap();
        meshes.push(mesh_brick(&c, &smp, mask, finer));
    }
    let t = topology(&meshes);
    assert!(t.boundary_edges > 0, "per-brick filter levels must crack: {t:?}");
}

/// The continuous filter: watertight under the violently cell-dependent field, never below
/// the sampling cell, and no steps between LOD shells (neighbouring lattice points differ by at
/// most one 1/16-octave quantum, also across shell boundaries).
#[test]
fn q4_smooth_filter_is_watertight_continuous_and_conservative() {
    let c = cfg(16, 0.25);
    for (cam, radius) in [([0i64, 0, 0], 19.0), ([7, -5, 3], 9.0), ([-30, 12, 21], 14.0)] {
        let lay = ShellLayout::camera_centred(&c.grid, cam, &[1, 1]);
        let r = 96;
        let f = LodDependent(radius);
        let mut meshes = Vec::new();
        for b in bricks_in(&c.grid, &lay, ([-r; 3], [r; 3]), lay.max_lod()) {
            let (mask, finer) = plan_brick(&c.grid, &lay, b).unwrap();
            meshes.push(mesh_brick(&c, &sample_brick_smooth(&f, &c, &lay, b, mask).unwrap(), mask, finer));
        }
        let t = topology(&meshes);
        assert!(t.is_closed() && t.is_clean(), "cam {cam:?}: {t:?}");
    }
    let lay = ShellLayout::camera_centred(&c.grid, [3, -2, 5], &[1, 2, 2]);
    let q = 2f64.powf(1.0 / 16.0) * 1.000_001;
    let (mut lo, mut hi) = (f64::MAX, 0.0f64);
    for axis in 0..3 {
        let mut prev: Option<f64> = None;
        for t in -400..400 {
            let mut p = [3 + t / 7, -2 + t / 11, 5 + t / 5];
            p[axis] = t;
            let cell = lay.smooth_filter_cell(0.25, p);
            let lod = lay.lod_at(p).max(filter_level(&lay, p).saturating_sub(1));
            assert!(cell >= 0.25 * (1u64 << lod) as f64 - 1e-12, "conservative at {p:?}: {cell} lod {lod}");
            if let Some(pc) = prev {
                // one lattice step changes the filter by at most a few quanta (no shell steps)
                assert!(cell / pc <= q * q * q && pc / cell <= q * q * q, "jump {pc} -> {cell} at {p:?}");
            }
            prev = Some(cell);
            lo = lo.min(cell);
            hi = hi.max(cell);
        }
    }
    assert!(hi / lo >= 6.0, "the sweep crossed several shells: {lo} .. {hi}");
}

/// Peaks are not truncated: the region spanning `Alpine::height_bounds` and the conservative
/// brick skip `Alpine::may_cross` keep the summit. Around the highest peak of an 18 km square,
/// the meshed maximum matches the field's (within a cell), and every skipped brick is provably
/// empty (sampled anyway: no sign change).
#[test]
fn q5_alpine_peaks_are_not_truncated_and_skips_are_conservative() {
    use voxel::fields::Alpine;
    let a = Alpine::new(7);
    let (lo_b, hi_b) = a.height_bounds();
    // the highest summit on a coarse grid, refined
    let mut best = (f64::MIN, 0.0, 0.0);
    for j in -45..=45 {
        for i in -45..=45 {
            let (x, z) = (i as f64 * 200.0, j as f64 * 200.0);
            let h = a.height(x, z, 32.0);
            assert!(h > lo_b && h < hi_b, "height bound violated at ({x},{z}): {h} not in {lo_b}..{hi_b}");
            if h > best.0 {
                best = (h, x, z);
            }
        }
    }
    for _ in 0..3 {
        let (_, cx, cz) = best;
        for j in -10..=10 {
            for i in -10..=10 {
                let (x, z) = (cx + i as f64 * 20.0, cz + j as f64 * 20.0);
                let h = a.height(x, z, 1.0);
                if h > best.0 {
                    best = (h, x, z);
                }
            }
        }
    }
    let c = cfg(32, 1.0);
    let lat = |v: f64| (v / c.grid.voxel_m()).floor() as i64;
    let cam = [lat(best.1 - 300.0), lat(best.0), lat(best.2 - 300.0)];
    let lay = ShellLayout::camera_centred(&c.grid, cam, &[2, 2]);
    let ext = 32 << lay.max_lod();
    let al = |v: i64, up: bool| if up { (v + ext - 1).div_euclid(ext) * ext } else { v.div_euclid(ext) * ext };
    let region = (
        [al(lat(best.1) - 256, false), al(lat(lo_b), false), al(lat(best.2) - 256, false)],
        [al(lat(best.1) + 256, true), al(lat(hi_b), true), al(lat(best.2) + 256, true)],
    );
    let (mut skipped, mut top) = (0usize, f64::MIN);
    for b in bricks_in(&c.grid, &lay, region, lay.max_lod()) {
        let o = c.grid.brick_origin(b);
        let e = (32i64 << b.lod) as f64;
        let lo = [o[0] as f64, o[1] as f64, o[2] as f64];
        let cell = (1u64 << b.lod) as f64;
        let (mask, finer) = plan_brick(&c.grid, &lay, b).unwrap();
        let smp = sample_brick_smooth(&a, &c, &lay, b, mask).unwrap();
        if !a.may_cross(lo, [lo[0] + e, lo[1] + e, lo[2] + e], cell) {
            skipped += 1;
            assert!(!smp.has_sign_change(), "skipped brick {b:?} holds surface");
            continue;
        }
        let m = mesh_brick(&c, &smp, mask, finer);
        for v in &m.ipos {
            top = top.max(v[1] as f64 / SUB as f64 * c.grid.voxel_m());
        }
    }
    assert!(skipped > 0, "the skip test was exercised");
    // reference: the field band-limited exactly as the mesher sampled it there
    let mut summit = f64::MIN;
    for j in -40..=40 {
        for i in -40..=40 {
            let (x, z) = (best.1 + i as f64 * 1.0, best.2 + j as f64 * 1.0);
            let fc = lay.smooth_filter_cell(c.grid.voxel_m(), [lat(x), lat(best.0), lat(z)]);
            summit = summit.max(a.height(x, z, fc));
        }
    }
    eprintln!(
        "summit {summit:.1} m at ({:.0},{:.0}); meshed top {top:.1} m; bounds {lo_b:.0}..{hi_b:.0}; {skipped} bricks skipped",
        best.1, best.2
    );
    // within one sampling cell at the summit's LOD (the mesh interpolates between lattice points)
    assert!((top - summit).abs() < 4.0, "meshed top {top} vs field summit {summit}");
}

/// The column height (base + gullies + crags) stays within the Lipschitz bound `may_cross`
/// relies on (SLOPE_MAX = 7 m/m, asserted with a 1.5x margin): 0.1 m central differences at many points across an 18 km
/// square, biased to the high crag zone, at several filter levels.
#[test]
fn q6_alpine_column_slope_within_bound_with_crags() {
    use voxel::fields::Alpine;
    let a = Alpine::new(7);
    let mut worst = 0.0f64;
    let mut high = 0usize;
    let mut s = 0x1234_5678u64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..40_000 {
        let (x, z) = (rnd() * 18000.0 - 9000.0, rnd() * 18000.0 - 9000.0);
        let cell = [0.25, 1.0, 4.0, 16.0][(rnd() * 4.0) as usize % 4];
        if a.height(x, z, 32.0) > 1050.0 {
            high += 1;
        }
        let t = rnd() * std::f64::consts::TAU;
        let (dx, dz) = (0.1 * t.cos(), 0.1 * t.sin());
        let g = (a.height(x + dx, z + dz, cell) - a.height(x - dx, z - dz, cell)).abs() / 0.2;
        worst = worst.max(g);
    }
    eprintln!("worst column slope {worst:.2} m/m ({high} samples in the crag zone)");
    assert!(high > 1000, "the crag zone was exercised");
    assert!(worst < 7.0 / 1.5, "column slope {worst} too close to SLOPE_MAX = 7");
}
