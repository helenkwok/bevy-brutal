//! TurboQuant-style rotated quantisation of brick density samples vs the lossless Lorenzo coder,
//! on the real Alpine near field (same world as `asset_bench`). Also: rotated scalar vs
//! octahedral normals (d = 3). See `starter.md`, "Learning from turbovec and TurboQuant".
//! `cargo run --release -p asset --example tq_bench`
use asset::schema::Writer;
use asset::tq::{self, D, Rotation, TqView};
use coords::{BrickCoord, BrickGrid};
use rayon::prelude::*;
use std::time::Instant;
use voxel::fields::Alpine;
use voxel::*;

fn best_of<T>(n: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::MAX;
    let mut out = None;
    for _ in 0..n {
        let t = Instant::now();
        let r = f();
        best = best.min(t.elapsed().as_secs_f64());
        out = Some(r);
    }
    (best, out.unwrap())
}

/// Edge-crossing displacement between exact and decoded samples over every lattice edge of the
/// brick's cells (corners 0..=n): (sum |dt| in voxels, crossings compared, max |dt|, topology
/// changes = crossings gained or lost).
fn crossings(n: usize, exact: &[i32], dec: &[i32], hist: &mut Vec<f64>) -> (f64, usize, f64, usize) {
    let m = n + 3;
    let at = |v: &[i32], i: usize, j: usize, k: usize| v[((k + 1) * m + (j + 1)) * m + (i + 1)] as f64;
    let (mut sum, mut cnt, mut mx, mut topo) = (0.0, 0usize, 0.0f64, 0usize);
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                for (di, dj, dk) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] {
                    let (i2, j2, k2) = (i + di, j + dj, k + dk);
                    if i2 > n || j2 > n || k2 > n {
                        continue;
                    }
                    let (a, b) = (at(exact, i, j, k), at(exact, i2, j2, k2));
                    let (c, d) = (at(dec, i, j, k), at(dec, i2, j2, k2));
                    let x0 = (a < 0.0) != (b < 0.0);
                    let x1 = (c < 0.0) != (d < 0.0);
                    if x0 != x1 {
                        topo += 1;
                    } else if x0 {
                        let dt = (a / (a - b) - c / (c - d)).abs();
                        sum += dt;
                        cnt += 1;
                        mx = mx.max(dt);
                        hist.push(dt);
                    }
                }
            }
        }
    }
    (sum, cnt, mx, topo)
}

/// Per-block uniform scalar quantiser at `bits` (min / max f32 per 512 samples, no rotation):
/// the obvious fixed-rate competitor (BC4-like).
fn uniform_block(vals: &[i32], bits: u32) -> Vec<i32> {
    let levels = (1u32 << bits) as f64;
    let mut out = vec![0; vals.len()];
    for (b, blk) in vals.chunks(D).enumerate() {
        let (lo, hi) = blk.iter().fold((f64::MAX, f64::MIN), |(l, h), &x| (l.min(x as f64), h.max(x as f64)));
        let (lo, hi) = (lo as f32 as f64, hi as f32 as f64);
        let step = (hi - lo) / levels;
        for (j, &x) in blk.iter().enumerate() {
            let q = if step > 0.0 { ((x as f64 - lo) / step).floor().clamp(0.0, levels - 1.0) } else { 0.0 };
            out[b * D + j] = (lo + (q + 0.5) * step + 0.5).floor() as i32;
        }
    }
    out
}

/// TQ without the rotation (ablation): Lloyd-Max on the normalised, unrotated coordinates.
fn tq_unrotated(vals: &[i32], bits: u32) -> Vec<i32> {
    let c = tq::centroids(bits);
    let bnd: Vec<f64> = c.windows(2).map(|w| 0.5 * (w[0] as f64 + w[1] as f64)).collect();
    let mut out = vec![0; vals.len()];
    for (b, blk) in vals.chunks(D).enumerate() {
        let mean = blk.iter().map(|&x| x as f64).sum::<f64>() / blk.len() as f64;
        let r: Vec<f64> = blk.iter().map(|&x| x as f64 - mean).collect();
        let norm = r.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        let q: Vec<f64> =
            r.iter().map(|x| c[bnd.partition_point(|&t| t < x / norm * (D as f64).sqrt())] as f64).collect();
        let (num, den) = r.iter().zip(&q).fold((0.0, 0.0), |(n, d), (x, y)| (n + x * y, d + y * y));
        let a = if den > 0.0 { num / den } else { 0.0 };
        for (j, y) in q.iter().enumerate() {
            out[b * D + j] = (mean + a * y + 0.5).floor() as i32;
        }
    }
    out
}

fn samples_file_bytes(list: &[BrickSamples]) -> usize {
    let mut w = Writer::new(BrickGrid::new(32, 0.5).unwrap());
    for s in list {
        w.add_samples(s);
    }
    w.finish().len()
}

fn oct_encode(n: [f32; 3], bits: u32) -> [f32; 3] {
    let l1 = n[0].abs() + n[1].abs() + n[2].abs();
    let (mut x, mut y) = (n[0] / l1, n[1] / l1);
    if n[2] < 0.0 {
        let (ox, oy) = (x, y);
        x = (1.0 - oy.abs()) * ox.signum();
        y = (1.0 - ox.abs()) * oy.signum();
    }
    let q = ((1u32 << bits) - 1) as f32;
    let qx = ((x * 0.5 + 0.5) * q).round() / q * 2.0 - 1.0;
    let qy = ((y * 0.5 + 0.5) * q).round() / q * 2.0 - 1.0;
    let z = 1.0 - qx.abs() - qy.abs();
    let (mut ux, mut uy) = (qx, qy);
    if z < 0.0 {
        ux = (1.0 - qy.abs()) * qx.signum();
        uy = (1.0 - qx.abs()) * qy.signum();
    }
    let l = (ux * ux + uy * uy + z * z).sqrt();
    [ux / l, uy / l, z / l]
}

fn main() {
    let field = Alpine::new(7);
    let grid = BrickGrid::new(32, 0.5).unwrap();
    let mut cfg = VoxelConfig::new(grid);
    let (cx, cz) = (7700.0, -4200.0);
    let gy = field.height(cx, cz, 0.5);
    cfg.set_origin([cx, gy + 40.0, cz]);
    let l = |v: f64| (v / grid.voxel_m()).floor() as i64;
    let layout = ShellLayout::camera_centred(&grid, [l(cx), l(gy), l(cz)], &[3, 3, 3]);
    let outer = *layout.boxes.last().unwrap();
    let coarse = 32 << layout.max_lod();
    let al = |v: i64, up: bool| {
        if up { (v + coarse - 1).div_euclid(coarse) * coarse } else { v.div_euclid(coarse) * coarse }
    };
    let region = (
        [al(outer.lo[0], false), al(l(gy) - 300, false), al(outer.lo[2], false)],
        [al(outer.hi[0], true), al(l(gy) + 700, true), al(outer.hi[2], true)],
    );
    let world = build_world_filtered(&field, &cfg, &layout, region, layout.max_lod()).unwrap();
    let coords: Vec<BrickCoord> = world.bricks.keys().copied().collect();
    let mut both: Vec<(BrickSamples, u8, Option<voxel::Aabb>)> = coords
        .par_iter()
        .map(|c| {
            let (mask, finer) = plan_brick(&grid, &layout, *c).unwrap();
            (sample_brick_smooth(&field, &cfg, &layout, *c, mask).unwrap(), mask, finer)
        })
        .collect();
    both.sort_by_key(|(s, _, _)| (s.coord.lod, s.coord.x, s.coord.y, s.coord.z));
    let plans: Vec<(u8, Option<voxel::Aabb>)> = both.iter().map(|b| (b.1, b.2)).collect();
    let smp: Vec<BrickSamples> = both.into_iter().map(|b| b.0).collect();
    let ns: usize = smp.iter().map(|s| s.d.len()).sum();
    // corner samples only (TQ covers `d`; transition faces stay lossless either way)
    let corners: Vec<BrickSamples> = smp
        .iter()
        .map(|s| BrickSamples { coord: s.coord, n: s.n, d: s.d.clone(), faces: Default::default() })
        .collect();
    let lossless = samples_file_bytes(&corners);
    let empty = samples_file_bytes(&[]);
    let dmin = smp.iter().flat_map(|s| s.d.iter()).min().unwrap();
    let dmax = smp.iter().flat_map(|s| s.d.iter()).max().unwrap();
    println!(
        "bricks {} (meshed), corner samples {} ({:.2} MB as i32), value range [{dmin}, {dmax}]",
        smp.len(),
        ns,
        ns as f64 * 4e-6
    );
    println!(
        "lossless Lorenzo+rANS (current): {:.3} MB, {:.3} bits/sample",
        (lossless - empty) as f64 / 1e6,
        (lossless - empty) as f64 * 8.0 / ns as f64
    );
    let cm_per_vox = |c: BrickCoord| grid.voxel_m() * grid.step(c.lod) as f64 * 100.0;

    let rot = Rotation::new();
    println!(
        "\nmode | bits/sample | MB | RMSE (density units) | rel. MSE (vs block var) | edge-crossing shift mean / p99 / max (cm) | topology changes per 1k crossings"
    );
    let report = |label: &str, bytes: f64, decs: &[Vec<i32>]| {
        let (mut se, mut sv) = (0.0f64, 0.0f64);
        let (mut sum_cm, mut cnt, mut mx_cm, mut topo) = (0.0, 0usize, 0.0f64, 0usize);
        let mut hist = Vec::new();
        for (s, dec) in smp.iter().zip(decs) {
            for blk in 0..s.d.len().div_ceil(D) {
                let r = &s.d[blk * D..((blk + 1) * D).min(s.d.len())];
                let m = r.iter().map(|&x| x as f64).sum::<f64>() / r.len() as f64;
                for (j, x) in r.iter().enumerate() {
                    sv += (*x as f64 - m).powi(2);
                    se += (*x as f64 - dec[blk * D + j] as f64).powi(2);
                }
            }
            let cmv = cm_per_vox(s.coord);
            let mut h = Vec::new();
            let (a, b, c, d) = crossings(s.n, &s.d, dec, &mut h);
            sum_cm += a * cmv;
            cnt += b;
            mx_cm = mx_cm.max(c * cmv);
            topo += d;
            hist.extend(h.into_iter().map(|x| x * cmv));
        }
        hist.sort_by(|a, b| a.total_cmp(b));
        let p99 = if hist.is_empty() { 0.0 } else { hist[(hist.len() * 99 / 100).min(hist.len() - 1)] };
        println!(
            "{label} | {:.3} | {:.3} | {:.1} | {:.5} | {:.2} / {:.2} / {:.1} | {:.1}",
            bytes * 8.0 / ns as f64,
            bytes / 1e6,
            (se / ns as f64).sqrt(),
            se / sv,
            sum_cm / cnt.max(1) as f64,
            p99,
            mx_cm,
            topo as f64 * 1000.0 / cnt.max(1) as f64
        );
    };
    for bits in 1..=4u32 {
        let (te, blobs) = best_of(3, || smp.iter().map(|s| tq::encode(s, bits, &rot)).collect::<Vec<_>>());
        let (tep, _) = best_of(3, || smp.par_iter().map(|s| tq::encode(s, bits, &rot)).collect::<Vec<_>>());
        let (tv, _) = best_of(5, || blobs.iter().map(|b| TqView::parse(b).unwrap().blocks).sum::<u32>());
        let (td, decs) =
            best_of(3, || blobs.iter().map(|b| TqView::parse(b).unwrap().decode(&rot)).collect::<Vec<_>>());
        let (tdp, _) =
            best_of(3, || blobs.par_iter().map(|b| TqView::parse(b).unwrap().decode(&rot)).collect::<Vec<_>>());
        let bytes: usize = blobs.iter().map(Vec::len).sum();
        println!(
            "  [TQ {bits}-bit timing: encode {:.1} M samples/s (1 core) / {:.1} (rayon); verify {:.2} GB/s; decode {:.1} M samples/s (1 core) / {:.1} (rayon)]",
            ns as f64 / te / 1e6,
            ns as f64 / tep / 1e6,
            bytes as f64 / tv / 1e9,
            ns as f64 / td / 1e6,
            ns as f64 / tdp / 1e6
        );
        report(&format!("TQ {bits}-bit (rotated, data-oblivious)"), bytes as f64, &decs);
        // hybrid: + lossless face-plane correction through the existing Lorenzo coder
        let res: Vec<BrickSamples> = smp
            .iter()
            .zip(&blobs)
            .map(|(s, b)| BrickSamples {
                coord: s.coord,
                n: s.n,
                d: tq::face_residual(&TqView::parse(b).unwrap(), &rot, &s.d),
                faces: Default::default(),
            })
            .collect();
        let hb = bytes + samples_file_bytes(&res) - empty;
        let hdec: Vec<Vec<i32>> =
            smp.iter().zip(&blobs).map(|(s, b)| tq::decode_hybrid(&TqView::parse(b).unwrap(), &rot, &s.d)).collect();
        report(&format!("TQ {bits}-bit hybrid (face planes exact)"), hb as f64, &hdec);
        let ub = smp.iter().map(|s| s.d.len().div_ceil(D) * (8 + D * bits as usize / 8)).sum::<usize>();
        let udec: Vec<Vec<i32>> = smp.iter().map(|s| uniform_block(&s.d, bits)).collect();
        report(&format!("uniform per-block min/max {bits}-bit (no rotation)"), ub as f64, &udec);
        let rdec: Vec<Vec<i32>> = smp.iter().map(|s| tq_unrotated(&s.d, bits)).collect();
        report(&format!("TQ {bits}-bit WITHOUT rotation (ablation)"), bytes as f64, &rdec);
    }
    // ---- narrow band: clamp to +-T quanta (1 voxel = 1024) before coding. Sign is preserved, so
    // only crossings with an endpoint beyond T move. Lossless baseline is clamped too (fair).
    for t in [2048i32, 4096, 16384] {
        let cl: Vec<BrickSamples> = smp
            .iter()
            .map(|s| BrickSamples {
                coord: s.coord,
                n: s.n,
                d: s.d.iter().map(|&x| x.clamp(-t, t)).collect(),
                faces: Default::default(),
            })
            .collect();
        let lb = samples_file_bytes(&cl) - empty;
        report(
            &format!("band +-{} vox: clamp + lossless Lorenzo", t / 1024),
            lb as f64,
            &cl.iter().map(|s| s.d.clone()).collect::<Vec<_>>(),
        );
        for bits in [2u32, 4] {
            let blobs: Vec<Vec<u8>> = cl.iter().map(|s| tq::encode(s, bits, &rot)).collect();
            let bytes: usize = blobs.iter().map(Vec::len).sum();
            let decs: Vec<Vec<i32>> = blobs.iter().map(|b| TqView::parse(b).unwrap().decode(&rot)).collect();
            report(&format!("band +-{} vox: clamp + TQ {bits}-bit", t / 1024), bytes as f64, &decs);
            let ub = cl.iter().map(|s| s.d.len().div_ceil(D) * (8 + D * bits as usize / 8)).sum::<usize>();
            let udec: Vec<Vec<i32>> = cl.iter().map(|s| uniform_block(&s.d, bits)).collect();
            report(&format!("band +-{} vox: clamp + uniform {bits}-bit", t / 1024), ub as f64, &udec);
        }
    }
    // ---- mesh-exact narrow band (lossless for the mesh by construction; verified below)
    for t in [1024i32, 4096] {
        let bd: Vec<BrickSamples> = smp
            .iter()
            .zip(&plans)
            .map(|(s, p)| BrickSamples { coord: s.coord, n: s.n, d: tq::mesh_band(s, p.0, t), faces: s.faces.clone() })
            .collect();
        let same = smp
            .iter()
            .zip(&bd)
            .zip(&plans)
            .filter(|((a, b), p)| mesh_brick(&cfg, a, p.0, p.1) == mesh_brick(&cfg, b, p.0, p.1))
            .count();
        let corners_only: Vec<BrickSamples> = bd
            .iter()
            .map(|s| BrickSamples { coord: s.coord, n: s.n, d: s.d.clone(), faces: Default::default() })
            .collect();
        let lb = samples_file_bytes(&corners_only) - empty;
        let changed: usize =
            smp.iter().zip(&bd).map(|(a, b)| a.d.iter().zip(&b.d).filter(|(x, y)| x != y).count()).sum();
        report(
            &format!("mesh-exact band, far samples clamped to +-{} vox, + lossless Lorenzo", t / 1024),
            lb as f64,
            &bd.iter().map(|s| s.d.clone()).collect::<Vec<_>>(),
        );
        println!(
            "    meshes bit-identical: {same}/{} bricks; samples altered {:.1}%",
            smp.len(),
            changed as f64 * 100.0 / ns as f64
        );
    }
    report(
        "lossless Lorenzo (reference)",
        (lossless - empty) as f64,
        &smp.iter().map(|s| s.d.clone()).collect::<Vec<_>>(),
    );

    // ---- normals, d = 3: rotated scalar (uniform Lloyd-Max: at d = 3 a rotated unit vector's
    // coordinates are exactly uniform on [-1, 1], Archimedes) vs octahedral
    let normals: Vec<[f32; 3]> = world
        .bricks
        .values()
        .flat_map(|m| m.vertices.iter().map(|v| v.normal))
        .filter(|n| n.iter().map(|x| x * x).sum::<f32>() > 0.5)
        .collect();
    // fixed rotation (Gram-Schmidt of a fixed matrix)
    let a = [[0.36f32, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]];
    let ang = |a: [f32; 3], b: [f32; 3]| (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0).acos().to_degrees();
    println!("\nnormals (d = 3), {} mesh normals: mean / max angular error (deg)", normals.len());
    for bits in [8u32, 10] {
        let (mut s, mut mx) = (0.0f64, 0.0f32);
        for n in &normals {
            let e = ang(*n, oct_encode(*n, bits));
            s += e as f64;
            mx = mx.max(e);
        }
        println!("  octahedral 2 x {bits} = {} bits: {:.3} / {:.3}", 2 * bits, s / normals.len() as f64, mx);
    }
    for bits in [5u32, 6, 7] {
        let q = (1u32 << bits) as f32;
        let (mut s, mut mx) = (0.0f64, 0.0f32);
        for n in &normals {
            let y = [0, 1, 2].map(|r| a[r][0] * n[0] + a[r][1] * n[1] + a[r][2] * n[2]);
            let yq = y.map(|c| (((c * 0.5 + 0.5) * q).floor().clamp(0.0, q - 1.0) + 0.5) / q * 2.0 - 1.0);
            let x = [0, 1, 2].map(|r| a[0][r] * yq[0] + a[1][r] * yq[1] + a[2][r] * yq[2]);
            let l = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
            let e = ang(*n, x.map(|c| c / l));
            s += e as f64;
            mx = mx.max(e);
        }
        println!("  rotated scalar 3 x {bits} = {} bits: {:.3} / {:.3}", 3 * bits, s / normals.len() as f64, mx);
    }
}
