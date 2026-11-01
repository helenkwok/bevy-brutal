use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hot::particles::{fill_test_data, integrate_scalar, integrate_simd, integrate_simd_par};
use lowlevel_abi::{ParticleBuffers, ParticleParams};

// Non-zero gravity on every axis: velocities converge to g/drag instead of decaying into
// subnormals over millions of bench iterations (which made the scalar path ~5x slower).
const PRM: ParticleParams = ParticleParams { dt: 1.0 / 60.0, gravity: [0.5, -9.81, 0.3], drag: 0.1 };

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("particles");
    for n in [10_000usize, 100_000, 1_000_000] {
        g.throughput(Throughput::Elements(n as u64));
        let mut base = ParticleBuffers::default();
        fill_test_data(&mut base, n, 42);
        // Give every particle a long life so the work stays constant across iterations.
        base.life.iter_mut().take(n).for_each(|l| *l = 1e9);

        let mut b = base.clone();
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bch, _| {
            bch.iter(|| black_box(integrate_scalar(&mut b.view_mut(), black_box(PRM))))
        });
        let mut b = base.clone();
        g.bench_with_input(BenchmarkId::new("simd_1t", n), &n, |bch, _| {
            bch.iter(|| black_box(integrate_simd(&mut b.view_mut(), black_box(PRM))))
        });
        let mut b = base.clone();
        g.bench_with_input(BenchmarkId::new("simd_mt", n), &n, |bch, _| {
            bch.iter(|| black_box(integrate_simd_par(&mut b.view_mut(), black_box(PRM), 16 * 1024)))
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
