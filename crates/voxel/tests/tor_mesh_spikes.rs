//! Robustness (mesh level): outcrops far from the world origin mesh without spike triangles.
//! Round-10 close-up regression: the tor joint frame was a rotation about the world origin by
//! a spatially varying angle, so ~8.7 km out the 7.5 / 10 m joint blocks were squeezed to ~1 m
//! and every metre carried an independent 0..11 m lift: the mesher emitted a field of vertical
//! teeth (1230 folded triangles within 60 m of the reported outcrop in the GPU near field).
//! CPU-meshes the reported outcrop and the worst far site of the round-10 scan at three LODs
//! (filtered sampling) and counts convex folds (knife-edge spikes) / needles with `voxel::meshq`.
use coords::BrickGrid;
use voxel::fields::Alpine;
use voxel::meshq::spike_counts;
use voxel::*;

fn scan(f: &Alpine, cx: f64, cz: f64, lod: u8) -> meshq::SpikeCounts {
    let v = 0.5;
    let c = VoxelConfig::new(BrickGrid::new(16, v).unwrap());
    let h = f.height(cx, cz, v);
    let q = |m: f64| (m / v).floor() as i64;
    let r = 36.0;
    let region = ([q(cx - r), q(h - 24.0), q(cz - r)], [q(cx + r), q(h + 32.0), q(cz + r)]);
    let w = build_world_filtered(f, &c, &UniformLod(lod), region, lod).unwrap();
    let m = w.merged();
    // weld by exact integer position (bricks are meshed independently)
    let mut weld = std::collections::HashMap::new();
    let mut pos = Vec::new();
    let idx: Vec<u32> = m
        .indices
        .iter()
        .map(|&i| {
            let p = m.ipos[i as usize];
            *weld.entry(p).or_insert_with(|| {
                pos.push(p.map(|c| c as f64 * v / SUB as f64));
                pos.len() as u32 - 1
            })
        })
        .collect();
    spike_counts(&pos, &idx, |p| (p[0] - cx).hypot(p[2] - cz) < r - 4.0)
}

#[test]
fn far_outcrops_mesh_without_spike_triangles() {
    let f = Alpine::new(7);
    // the reported close-up outcrop (~120 m from the camera) and the worst far site
    for (cx, cz) in [(7780.2, -3505.8), (-4735.5, -8852.0)] {
        assert!(f.tor_mask(cx, cz) > 0.3, "site ({cx}, {cz}) is an outcrop");
        for lod in 0..3u8 {
            let s = scan(&f, cx, cz, lod);
            eprintln!("({cx}, {cz}) lod {lod}: {s:?}");
            assert!(s.tris > 2000, "meshed");
            // fixed field: <= 0.09 % (sharp joint-block corners at 0.5 m); cp60 field:
            // 2.2 - 3.6 % at lod 0 / 1, 0.22 - 0.9 % at lod 2 (and 3x the triangles)
            assert!(s.folds * 2000 <= 3 * s.tris, "convex folded (spike) triangles > 0.15 %: {s:?}");
            assert!(s.needles * 20 <= s.tris, "needle triangles: {s:?}");
        }
    }
}
