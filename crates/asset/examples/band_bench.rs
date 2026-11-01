//! Mesh-exact band in the asset writer (format minor 2): end-to-end file size and decode speed,
//! band on vs off, on the real Alpine near field (asset_bench world, 3 LOD shells), plus a
//! bit-identity check of every brick's mesh (all LODs, all transition masks).
//! `cargo run --release -p asset --example band_bench`
use asset::fb::{AlignedBytes, Limits};
use asset::schema::{File, Writer};
use coords::BrickGrid;
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
    let coords = bricks_in(&grid, &layout, region, layout.max_lod());
    let mut src: Vec<(BrickSamples, u8, Option<Aabb>)> = coords
        .par_iter()
        .filter_map(|c| {
            let (mask, finer) = plan_brick(&grid, &layout, *c).unwrap();
            let s = sample_brick_smooth(&field, &cfg, &layout, *c, mask).unwrap();
            s.has_sign_change().then_some((s, mask, finer))
        })
        .collect();
    src.sort_by_key(|(s, _, _)| (s.coord.lod, s.coord.x, s.coord.y, s.coord.z));
    let ns: usize = src.iter().map(|s| s.0.d.len() + s.0.faces.iter().map(Vec::len).sum::<usize>()).sum();
    let with_t = src.iter().filter(|s| s.1 != 0).count();
    let lods: std::collections::BTreeSet<u8> = src.iter().map(|s| s.0.coord.lod).collect();
    println!(
        "{} bricks with a surface ({with_t} with transition faces, LODs {lods:?}), {ns} samples incl. faces ({:.1} MB as i32)",
        src.len(),
        ns as f64 * 4e-6
    );
    let only: Vec<BrickSamples> = src
        .iter()
        .map(|s| BrickSamples { coord: s.0.coord, n: s.0.n, d: s.0.d.clone(), faces: s.0.faces.clone() })
        .collect();

    let (te_full, full) = best_of(3, || {
        let mut w = Writer::new(grid);
        for s in &only {
            w.add_samples(s);
        }
        w.finish()
    });
    let (te_band, band) = best_of(3, || {
        let mut w = Writer::new(grid);
        w.add_samples_banded_par(&only, 1024);
        w.finish()
    });
    let (te_band1, _) = best_of(3, || {
        let mut w = Writer::new(grid);
        for s in &only {
            w.add_samples_banded(s, 1024);
        }
        w.finish()
    });
    let (fa, ba) = (AlignedBytes::from_slice(&full), AlignedBytes::from_slice(&band));
    let ff = File::open(fa.as_slice(), Limits::default()).unwrap();
    let fb = File::open(ba.as_slice(), Limits::default()).unwrap();
    let n = src.len();
    let (td_full, _) = best_of(5, || (0..n).map(|i| ff.decode_samples(i).unwrap().unwrap().d.len()).sum::<usize>());
    let (td_band, _) =
        best_of(5, || (0..n).map(|i| fb.decode_mesh_samples(i).unwrap().unwrap().coord().lod as usize).sum::<usize>());
    let (tdp_full, _) =
        best_of(5, || (0..n).into_par_iter().map(|i| ff.decode_samples(i).unwrap().unwrap().d.len()).sum::<usize>());
    let (tdp_band, _) = best_of(5, || {
        (0..n).into_par_iter().map(|i| fb.decode_mesh_samples(i).unwrap().unwrap().coord().lod as usize).sum::<usize>()
    });
    let (tv_full, _) = best_of(5, || File::open_deep(fa.as_slice(), Limits::default()).is_ok());
    let (tv_band, _) = best_of(5, || File::open_deep(ba.as_slice(), Limits::default()).is_ok());
    let ms = |t: f64| t * 1e3;
    println!("| | band off (full density) | band on (t = 1 voxel) |");
    println!("|---|---|---|");
    println!(
        "| file bytes | {} ({:.3} MB) | {} ({:.3} MB) |",
        full.len(),
        full.len() as f64 / 1e6,
        band.len(),
        band.len() as f64 / 1e6
    );
    println!(
        "| bits / sample (incl. faces) | {:.3} | {:.3} |",
        full.len() as f64 * 8.0 / ns as f64,
        band.len() as f64 * 8.0 / ns as f64
    );
    println!(
        "| ratio vs raw i32 | {:.2}:1 | {:.2}:1 |",
        ns as f64 * 4.0 / full.len() as f64,
        ns as f64 * 4.0 / band.len() as f64
    );
    println!(
        "| encode (write file), ms | {:.1} (serial) | {:.1} (band serial) / {:.1} (band rayon) |",
        ms(te_full),
        ms(te_band1),
        ms(te_band)
    );
    println!(
        "| decode all bricks, 1 thread, ms | {:.1} ({:.0} M samples/s) | {:.1} ({:.0} M samples/s, incl. band check) |",
        ms(td_full),
        ns as f64 / td_full / 1e6,
        ms(td_band),
        ns as f64 / td_band / 1e6
    );
    println!("| decode all bricks, rayon, ms | {:.1} | {:.1} |", ms(tdp_full), ms(tdp_band));
    println!("| open_deep (verify + decode all), ms | {:.1} | {:.1} |", ms(tv_full), ms(tv_band));
    // bit identity of every brick mesh
    let same = src
        .par_iter()
        .enumerate()
        .filter(|(i, (s, m, f))| {
            fb.decode_mesh_samples(*i).unwrap().unwrap().mesh(&cfg, *f) == mesh_brick(&cfg, s, *m, *f)
        })
        .count();
    let refused =
        (0..n).filter(|&i| matches!(fb.decode_samples(i), Some(Err(asset::rans::CodecError::Banded)))).count();
    println!(
        "meshes bit-identical from the banded file: {same}/{n} (incl. {with_t} with transitions); decode_samples refused {refused}/{n}"
    );
}
