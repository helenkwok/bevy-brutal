//! Asset-format measurements on the real Alpine mountains near field (camera of the quality
//! scene): compression ratio vs the raw GPU layout, bits per triangle by stream, decode
//! throughput (1 core and rayon), the streaming budget at 60 fps, verifier cost, samples and
//! instances. Dumps the bricks to `/tmp/asset_bricks.bin` for `tools/draco_compare.py`.
//!
//! `cargo run --release -p asset --example asset_bench`
use asset::mesh::{self, ConnMode, MeshOpts};
use asset::rans::Counts;
use asset::schema::{File, NUM_MODELS, Writer};
use asset::{AlignedBytes, Instance, Limits};
use coords::{BrickCoord, BrickGrid};
use rayon::prelude::*;
use std::io::Write as _;
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
    let t = Instant::now();
    let world = build_world_filtered(&field, &cfg, &layout, region, layout.max_lod()).unwrap();
    let bricks: Vec<(BrickCoord, &BrickMesh)> = world.bricks.iter().map(|(c, m)| (*c, m)).collect();
    let (nv, nt): (usize, usize) = (world.vertices(), world.triangles());
    let raw_bytes: usize = bricks.iter().map(|(_, m)| m.vertices.len() * 24 + m.indices.len() * 4).sum();
    println!(
        "world: {} bricks, {nv} vertices, {nt} triangles, raw GPU layout {:.2} MB (24 B/vertex + 12 B/triangle), built in {:.1} s",
        bricks.len(),
        raw_bytes as f64 / 1e6,
        t.elapsed().as_secs_f64()
    );
    let mut lods = [0usize; 8];
    for (c, m) in &bricks {
        lods[c.lod as usize] += m.indices.len() / 3;
    }
    println!("triangles per LOD: {:?}", &lods[..=layout.max_lod() as usize]);

    // ---- stream breakdown (ideal code length per model) for the default mode
    let names = [
        "conn", "start", "sflags", "ref", "lat_ax", "lat_off", "frac_ax", "frac_off", "nu", "nv", "dir", "seq", "", "",
        "", "", "", "", "", "", "cand", "dsgn",
    ];
    for (label, opts) in [
        ("traversal q0 (lossless)", MeshOpts::default()),
        ("traversal q2 (7.8 mm)", MeshOpts { qbits: 2, ..Default::default() }),
        ("sequential q0", MeshOpts { conn: ConnMode::Sequential, ..Default::default() }),
    ] {
        let logs: Vec<_> = bricks.par_iter().map(|(c, m)| mesh::encode(&grid, *c, m, opts)).collect();
        let mut counts = Counts::new(NUM_MODELS);
        let mut raw_bits = 0usize;
        for (_, log) in &logs {
            counts.add_log(log);
            raw_bits += log.bits.finished().len() * 8;
        }
        let models = counts.models();
        let mut s = String::new();
        for (k, n) in names.iter().enumerate() {
            let b = models[k].cost_bits(&counts.0[k]);
            // tagged tokens t > 1 carry t - 1 raw bits (exact for tagged models; 0 for plain symbols)
            let tagged = !matches!(k, 0..=2 | 10 | 20 | 21);
            let rb: f64 = if tagged {
                counts.0[k].iter().enumerate().map(|(t, c)| *c as f64 * t.saturating_sub(1) as f64).sum()
            } else {
                0.0
            };
            if b > 0.0 {
                s += &format!("{n} {:.2}+{:.2}  ", b / nt as f64, rb / nt as f64);
            }
        }
        println!("{label}: bits/triangle by model (rANS + raw): {s}| raw bits {:.2}", raw_bits as f64 / nt as f64);
    }

    // ---- files
    let mut rows = Vec::new();
    for (label, opts) in [
        ("traversal q0 lossless", MeshOpts::default()),
        ("traversal q2 (7.8 mm)", MeshOpts { qbits: 2, ..Default::default() }),
        ("traversal q0, 8-bit normals", MeshOpts { normal_bits: 8, ..Default::default() }),
        ("sequential q0 lossless", MeshOpts { conn: ConnMode::Sequential, ..Default::default() }),
    ] {
        let (enc_s, buf) = best_of(3, || {
            let mut w = Writer::new(grid);
            w.add_meshes(bricks.iter().map(|(c, m)| (*c, *m)), opts);
            w.finish()
        });
        let buf = AlignedBytes::from_slice(&buf);
        let (ver_s, f) = best_of(20, || File::open(buf.as_slice(), Limits::default()).unwrap());
        // decode straight into a preallocated "staging" vertex / index buffer
        let mut offs = Vec::with_capacity(bricks.len());
        let (mut vo, mut io) = (0usize, 0usize);
        for b in f.bricks() {
            let (h, ..) = b.mesh.unwrap();
            offs.push((vo, io));
            vo += h.vertex_count as usize;
            io += h.index_count as usize;
        }
        let mut verts = vec![VoxelVertex::default(); vo];
        let mut idx = vec![0u32; io];
        let decode_into = |i: usize, v: &mut [VoxelVertex], ix: &mut [u32], base: u32| {
            let b = f.bricks()[i];
            let d = f.decode_mesh(i).unwrap().unwrap();
            d.write_gpu(&cfg, b.coord, b.mesh.unwrap().0.qbits, v, ix, base);
        };
        let (one_s, _) = best_of(5, || {
            for (i, &(a, b)) in offs.iter().enumerate() {
                let (na, nb) = (
                    f.bricks()[i].mesh.unwrap().0.vertex_count as usize,
                    f.bricks()[i].mesh.unwrap().0.index_count as usize,
                );
                decode_into(i, &mut verts[a..a + na], &mut idx[b..b + nb], a as u32);
            }
        });
        // parallel: split the staging buffers into disjoint per-brick slices
        let (par_s, _) = best_of(5, || {
            let mut vs: Vec<&mut [VoxelVertex]> = Vec::with_capacity(offs.len());
            let mut is: Vec<&mut [u32]> = Vec::with_capacity(offs.len());
            let (mut vr, mut ir) = (&mut verts[..], &mut idx[..]);
            for b in f.bricks() {
                let h = b.mesh.unwrap().0;
                let (a, rest) = vr.split_at_mut(h.vertex_count as usize);
                vs.push(a);
                vr = rest;
                let (a, rest) = ir.split_at_mut(h.index_count as usize);
                is.push(a);
                ir = rest;
            }
            vs.into_par_iter().zip(is).enumerate().for_each(|(i, (v, ix))| decode_into(i, v, ix, offs[i].0 as u32));
        });
        let cores = rayon::current_num_threads();
        let per_brick_us = one_s / bricks.len() as f64 * 1e6;
        rows.push(format!(
            "| {label} | {:.3} MB | {:.2}:1 | {:.2} | {:.1} | {:.0} / {:.0} | {:.0} / {:.0} | {:.1} | {:.0} | {:.0} |",
            buf.as_slice().len() as f64 / 1e6,
            raw_bytes as f64 / buf.as_slice().len() as f64,
            buf.as_slice().len() as f64 * 8.0 / nt as f64,
            enc_s * 1e3,
            raw_bytes as f64 / one_s / 1e6,
            buf.as_slice().len() as f64 / one_s / 1e6,
            raw_bytes as f64 / par_s / 1e6,
            buf.as_slice().len() as f64 / par_s / 1e6,
            per_brick_us,
            2000.0 / per_brick_us,
            2000.0 / (par_s / bricks.len() as f64 * 1e6),
        ));
        let _ = cores;
        println!("{label}: verify (open) {:.1} us for {} bricks", ver_s * 1e6, f.bricks().len());
    }
    println!(
        "\n| mode | file | ratio vs raw | bits/tri (all) | encode ms | decode MB/s out/in, 1 core | decode MB/s out/in, {} threads | us/brick (1 core) | bricks per 2 ms (1 core) | bricks per 2 ms ({} thr) |",
        rayon::current_num_threads(),
        rayon::current_num_threads()
    );
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for r in rows {
        println!("{r}");
    }

    // ---- raw zero-copy path for comparison (no decode: the verifier cost only)
    let mut w = Writer::new(grid);
    for (c, m) in &bricks {
        w.add_raw(*c, m);
    }
    let rawfile = AlignedBytes::from_slice(&w.finish());
    let (ver_s, f) = best_of(20, || File::open(rawfile.as_slice(), Limits::default()).unwrap());
    let n_ok = (0..f.bricks().len()).filter(|&i| f.raw_gpu(i).is_some()).count();
    println!(
        "\nraw (uncompressed, 256-aligned) file: {:.2} MB ({:.1} % padding overhead), open+verify {:.1} us, {n_ok} zero-copy GPU views",
        rawfile.as_slice().len() as f64 / 1e6,
        (rawfile.as_slice().len() as f64 / raw_bytes as f64 - 1.0) * 100.0,
        ver_s * 1e6
    );

    // ---- density samples of the meshed bricks (the brick "terrain" payload)
    let coords: Vec<BrickCoord> = bricks.iter().map(|(c, _)| *c).take(256).collect();
    let smp: Vec<_> = coords
        .par_iter()
        .map(|c| {
            let (mask, _) = plan_brick(&grid, &layout, *c).unwrap();
            sample_brick_smooth(&field, &cfg, &layout, *c, mask).unwrap()
        })
        .collect();
    let sraw: usize = smp.iter().map(|s| s.bytes()).sum();
    let mut w = Writer::new(grid);
    for s in &smp {
        w.add_samples(s);
    }
    let sbuf = w.finish();
    let f = File::open(&sbuf, Limits::default()).unwrap();
    let (s1, _) = best_of(3, || (0..smp.len()).for_each(|i| drop(f.decode_samples(i).unwrap().unwrap())));
    println!(
        "samples: {} bricks, raw i32 {:.2} MB -> {:.3} MB ({:.1}:1, lossless), decode {:.0} MB/s out (1 core), {:.0} us/brick",
        smp.len(),
        sraw as f64 / 1e6,
        sbuf.len() as f64 / 1e6,
        sraw as f64 / sbuf.len() as f64,
        sraw as f64 / s1 / 1e6,
        s1 / smp.len() as f64 * 1e6
    );

    // ---- instances: forest-like clustered placement on the real height field
    let mut inst = Vec::new();
    let mut s = 1u64;
    let mut rnd = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    for j in 0..400 {
        for i in 0..400 {
            let (x, z) = (cx - 1000.0 + i as f64 * 5.0 + rnd() * 4.0, cz - 1000.0 + j as f64 * 5.0 + rnd() * 4.0);
            if voxel::noise::perlin2(x * 0.004, z * 0.004, 11) < -0.05 {
                continue;
            }
            let h = 8.0 + 20.0 * rnd() * rnd();
            inst.push(Instance {
                pos: [x, field.height(x, z, 0.5) - 0.3, z],
                yaw: (rnd() * std::f64::consts::TAU) as f32,
                height: h as f32,
                width: (h * (0.32 + 0.12 * rnd())) as f32,
                mesh: if rnd() < 0.1 { 3 } else { (rnd() * 3.0) as u8 },
                material: (rnd() * 1023.0) as u16,
            });
        }
    }
    let mut w = Writer::new(grid);
    for ch in inst.chunks(4096) {
        w.add_instances(ch);
    }
    let ibuf = w.finish();
    let f = File::open(&ibuf, Limits::default()).unwrap();
    let cam = [cx, gy, cz];
    let (s1, n) = best_of(3, || {
        (0..f.chunks().len())
            .map(|i| {
                f.decode_instances(i).unwrap().iter().map(|x| std::hint::black_box(x.to_gpu(cam))).fold(0, |n, _| n + 1)
            })
            .sum::<usize>()
    });
    let gpu = inst.len() * size_of::<lowlevel_abi::InstanceGpu>();
    println!(
        "instances: {} -> {:.1} B each ({:.1}:1 vs 80 B InstanceGpu, {:.1}:1 vs a 36 B f64-pos+yaw+scale record), decode+matrix {:.1} M/s (1 core)",
        inst.len(),
        ibuf.len() as f64 / inst.len() as f64,
        gpu as f64 / ibuf.len() as f64,
        (inst.len() * 36) as f64 / ibuf.len() as f64,
        n as f64 / s1 / 1e6
    );

    // ---- dump for Draco: per brick origin (m), extent (m), nv, nt, f32 pos (world-ish metres
    // relative to the brick origin), f32 normals, u32 indices
    let mut out = std::io::BufWriter::new(std::fs::File::create("/tmp/asset_bricks.bin").unwrap());
    out.write_all(&(bricks.len() as u32).to_le_bytes()).unwrap();
    for (c, m) in &bricks {
        let o = grid.brick_origin(*c);
        let ext = (32i64 << c.lod) as f32 * 0.5;
        let k = grid.voxel_m() / SUB as f64;
        out.write_all(&ext.to_le_bytes()).unwrap();
        out.write_all(&(m.ipos.len() as u32).to_le_bytes()).unwrap();
        out.write_all(&((m.indices.len() / 3) as u32).to_le_bytes()).unwrap();
        for p in &m.ipos {
            for a in 0..3 {
                out.write_all(&(((p[a] - o[a] * SUB) as f64 * k) as f32).to_le_bytes()).unwrap();
            }
        }
        for v in &m.vertices {
            out.write_all(bytemuck::bytes_of(&v.normal)).unwrap();
        }
        out.write_all(bytemuck::cast_slice(&m.indices)).unwrap();
    }
    println!("dumped /tmp/asset_bricks.bin");
}
