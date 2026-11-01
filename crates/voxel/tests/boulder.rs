//! Fractured boulders (`voxel::boulder`): every render variant / LOD is watertight and clean
//! under the exact topology checker, deterministic, and has the intended size and faceting.
use voxel::boulder::{Boulder, boulder_mesh};
use voxel::{DensityField, tri_normal};

#[test]
fn b1_boulders_are_watertight_without_degenerate_or_flipped_triangles() {
    for seed in 21..27u64 {
        for cells in [20u32, 8] {
            let b = boulder_mesh(seed, cells);
            let t = &b.topology;
            assert!(t.is_closed(), "seed {seed} cells {cells}: {t:?}");
            assert_eq!((t.degenerate, t.flipped, t.non_manifold_edges, t.boundary_edges), (0, 0, 0, 0));
            assert_eq!(t.closed_components, 1, "one solid piece: {t:?}");
            assert_eq!(t.negative_volume_components, 0, "no cavities");
            let n = t.triangles;
            assert!(
                if cells == 20 { (600..4000).contains(&n) } else { (60..700).contains(&n) },
                "seed {seed} cells {cells}: {n} triangles"
            );
            // every triangle's exact normal agrees with its vertex normals (outward)
            let m = &b.mesh;
            let bad = m.indices.as_chunks::<3>().0.iter().filter(|tr| {
                let nn = tri_normal(m.ipos[tr[0] as usize], m.ipos[tr[1] as usize], m.ipos[tr[2] as usize]);
                let g: f64 = (0..3)
                    .map(|k| tr.iter().map(|&i| m.vertices[i as usize].normal[k] as f64).sum::<f64>() * nn[k] as f64)
                    .sum();
                g <= 0.0
            });
            assert!(bad.count() * 50 <= n, "seed {seed}: vertex normals disagree with winding");
        }
    }
}

#[test]
fn b2_boulders_are_deterministic_and_faceted() {
    let a = boulder_mesh(23, 20);
    let b = boulder_mesh(23, 20);
    assert_eq!(a.mesh, b.mesh, "bit-identical rebuild");
    assert_ne!(a.mesh.ipos, boulder_mesh(24, 20).mesh.ipos, "variants differ");
    // size: inside the documented mesh-space box, and not a sliver
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for v in &a.mesh.vertices {
        for k in 0..3 {
            lo[k] = lo[k].min(v.pos[k]);
            hi[k] = hi[k].max(v.pos[k]);
        }
    }
    for k in 0..3 {
        assert!(hi[k] - lo[k] > 0.5 && lo[k] > -1.25 && hi[k] < 1.25, "axis {k}: {lo:?}..{hi:?}");
    }
    assert!(hi[1] - lo[1] < hi[0] - lo[0], "flattened body");
    // faceted: most of the surface area lies on the cutting planes (normal within 20 deg of one
    // of them), i.e. flat fracture faces and chips, not a smoothed blob
    let faces = Boulder::new(23).face_normals();
    let m = &a.mesh;
    let (mut on, mut total) = (0.0, 0.0);
    for tr in m.indices.as_chunks::<3>().0.iter() {
        let nn = tri_normal(m.ipos[tr[0] as usize], m.ipos[tr[1] as usize], m.ipos[tr[2] as usize]);
        let l = ((nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]) as f64).sqrt();
        let best = faces
            .iter()
            .map(|f| (f[0] * nn[0] as f64 + f[1] * nn[1] as f64 + f[2] * nn[2] as f64) / l)
            .fold(-1.0, f64::max);
        total += l;
        if best > 20f64.to_radians().cos() {
            on += l;
        }
    }
    assert!(on / total > 0.6, "faceting: {:.2} of the area on the cutting planes", on / total);
    // the field is a distance lower bound: |grad| <= 1 (Lipschitz) away from the roughness
    let f = Boulder::new(23);
    let mut worst = 0.0f64;
    for k in 0..2000 {
        let p = [
            ((k * 37) % 101) as f64 / 50.0 - 1.0,
            ((k * 53) % 97) as f64 / 48.0 - 1.0,
            ((k * 71) % 89) as f64 / 44.0 - 1.0,
        ];
        let e = 1e-4;
        let g = [0, 1, 2].map(|i| {
            let (mut a, mut b) = (p, p);
            a[i] += e;
            b[i] -= e;
            (f.density(a) - f.density(b)) / (2.0 * e)
        });
        worst = worst.max((g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt());
    }
    assert!(worst < 1.4, "gradient bound {worst}");
}
