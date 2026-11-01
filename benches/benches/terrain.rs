use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use terrain::*;

fn bench(c: &mut Criterion) {
    let p = NoiseParams::default();
    let req = ChunkRequest { coord: [3, -2], lod: 0, seed: 42 };
    let mut hm = vec![0.0; PADDED * PADDED];
    let mut g = c.benchmark_group("terrain");
    g.bench_function("noise_scalar", |b| b.iter(|| heightmap_scalar(black_box(&req), &p, &mut hm)));
    g.bench_function("noise_simd", |b| b.iter(|| heightmap_simd(black_box(&req), &p, &mut hm)));
    let mut arena = mem::FrameArena::with_capacity(CHUNK_SAMPLES * CHUNK_SAMPLES * size_of::<TerrainVertex>() + 128);
    heightmap_simd(&req, &p, &mut hm);
    g.bench_function("mesh", |b| {
        b.iter(|| {
            arena.reset();
            black_box(mesh_chunk(&req, black_box(&hm), &arena).unwrap().aabb_max);
        })
    });
    for lod in 0..=3u8 {
        let req = ChunkRequest { lod, ..req };
        g.bench_function(format!("chunk_full_lod{lod}"), |b| {
            b.iter(|| {
                arena.reset();
                heightmap_simd(black_box(&req), &p, &mut hm);
                black_box(mesh_chunk(&req, &hm, &arena).unwrap().aabb_max);
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
