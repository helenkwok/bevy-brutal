//! Semi-implicit Euler particle integration over SoA buffers.
//!
//! v' = (v + g*dt) * (1 - drag*dt);  p' = p + v'*dt;  life' = life - dt
//! Dead particles (life <= 0) are frozen: neither position nor velocity changes.

use lowlevel_abi::{HotSystem, LANES, ParticleParams, ParticlesSoA};
use wide::f32x8;

/// Scalar reference implementation: the oracle for the SIMD path. Returns the live count.
pub fn integrate_scalar(p: &mut ParticlesSoA<'_>, prm: ParticleParams) -> usize {
    p.assert_valid();
    let ParticleParams { dt, gravity: [gx, gy, gz], drag } = prm;
    let damp = 1.0 - drag * dt;
    let mut live = 0;
    for i in 0..p.len() {
        if p.life[i] <= 0.0 {
            continue;
        }
        let vx = (p.vx[i] + gx * dt) * damp;
        let vy = (p.vy[i] + gy * dt) * damp;
        let vz = (p.vz[i] + gz * dt) * damp;
        p.vx[i] = vx;
        p.vy[i] = vy;
        p.vz[i] = vz;
        p.px[i] += vx * dt;
        p.py[i] += vy * dt;
        p.pz[i] += vz * dt;
        p.life[i] -= dt;
        if p.life[i] > 0.0 {
            live += 1;
        }
    }
    live
}

#[inline(always)]
fn load(s: &[f32; LANES]) -> f32x8 {
    f32x8::from(*s)
}

#[inline(always)]
fn store(dst: &mut [f32; LANES], v: f32x8) {
    *dst = v.to_array();
}

/// View a lane-padded column as `[f32; LANES]` blocks (padding is asserted at the API boundary).
#[inline(always)]
fn lanes(s: &mut [f32]) -> core::slice::IterMut<'_, [f32; LANES]> {
    let (blocks, rest) = s.as_chunks_mut::<LANES>();
    debug_assert!(rest.is_empty(), "column not padded to LANES");
    blocks.iter_mut()
}

/// Integrate one lane-aligned block. All slices have the same len, a multiple of LANES.
#[allow(clippy::too_many_arguments)]
fn integrate_block(
    px: &mut [f32],
    py: &mut [f32],
    pz: &mut [f32],
    vx: &mut [f32],
    vy: &mut [f32],
    vz: &mut [f32],
    life: &mut [f32],
    prm: ParticleParams,
) -> usize {
    let dt = f32x8::splat(prm.dt);
    let gx = f32x8::splat(prm.gravity[0] * prm.dt);
    let gy = f32x8::splat(prm.gravity[1] * prm.dt);
    let gz = f32x8::splat(prm.gravity[2] * prm.dt);
    let damp = f32x8::splat(1.0 - prm.drag * prm.dt);
    let zero = f32x8::ZERO;
    let mut live = 0usize;

    let it = lanes(px).zip(lanes(py)).zip(lanes(pz)).zip(lanes(vx)).zip(lanes(vy)).zip(lanes(vz)).zip(lanes(life));
    for ((((((px, py), pz), vx), vy), vz), life) in it {
        let l = load(life);
        let alive = l.simd_gt(zero); // all-ones lanes where alive
        // Same operation order as the scalar ref: (v + g*dt) * damp, p + v*dt.
        let nvx = (load(vx) + gx) * damp;
        let nvy = (load(vy) + gy) * damp;
        let nvz = (load(vz) + gz) * damp;
        store(vx, alive.bitselect(nvx, load(vx)));
        store(vy, alive.bitselect(nvy, load(vy)));
        store(vz, alive.bitselect(nvz, load(vz)));
        store(px, alive.bitselect(load(px) + nvx * dt, load(px)));
        store(py, alive.bitselect(load(py) + nvy * dt, load(py)));
        store(pz, alive.bitselect(load(pz) + nvz * dt, load(pz)));
        let nl = alive.bitselect(l - dt, l);
        store(life, nl);
        live += nl.simd_gt(zero).to_bitmask().count_ones() as usize;
    }
    live
}

/// SIMD single-threaded integration with `wide::f32x8`. Returns the live count.
pub fn integrate_simd(p: &mut ParticlesSoA<'_>, prm: ParticleParams) -> usize {
    p.assert_valid();
    #[cfg(feature = "tracy")]
    let _z = tracy_client::span!("hot::particles");
    integrate_block(p.px, p.py, p.pz, p.vx, p.vy, p.vz, p.life, prm)
}

/// SIMD + rayon over chunks of `chunk` particles (rounded to LANES). Returns the live count.
#[cfg(feature = "parallel")]
pub fn integrate_simd_par(p: &mut ParticlesSoA<'_>, prm: ParticleParams, chunk: usize) -> usize {
    use rayon::prelude::*;
    p.assert_valid();
    #[cfg(feature = "tracy")]
    let _z = tracy_client::span!("hot::particles_par");
    let c = chunk.max(LANES).next_multiple_of(LANES);
    p.px.par_chunks_mut(c)
        .zip(p.py.par_chunks_mut(c))
        .zip(p.pz.par_chunks_mut(c))
        .zip(p.vx.par_chunks_mut(c))
        .zip(p.vy.par_chunks_mut(c))
        .zip(p.vz.par_chunks_mut(c))
        .zip(p.life.par_chunks_mut(c))
        .map(|((((((px, py), pz), vx), vy), vz), life)| integrate_block(px, py, pz, vx, vy, vz, life, prm))
        .sum()
}

/// `HotSystem` wrapper used by the bridge.
#[derive(Default)]
pub struct ParticleIntegrator;

impl HotSystem for ParticleIntegrator {
    type Input<'a> = ParticleParams;
    type Output<'a> = (ParticlesSoA<'a>, &'a mut usize);
    fn run(&mut self, prm: ParticleParams, (mut soa, live): Self::Output<'_>) {
        #[cfg(feature = "parallel")]
        {
            *live = integrate_simd_par(&mut soa, prm, 16 * 1024);
        }
        #[cfg(not(feature = "parallel"))]
        {
            *live = integrate_simd(&mut soa, prm);
        }
    }
}

/// Deterministic test/bench data (xorshift, no rand dependency).
pub fn fill_test_data(buf: &mut lowlevel_abi::ParticleBuffers, n: usize, seed: u64) {
    buf.reset(n);
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32 // [0,1)
    };
    for i in 0..n {
        buf.px[i] = next() * 100.0 - 50.0;
        buf.py[i] = next() * 100.0;
        buf.pz[i] = next() * 100.0 - 50.0;
        buf.vx[i] = next() * 10.0 - 5.0;
        buf.vy[i] = next() * 10.0;
        buf.vz[i] = next() * 10.0 - 5.0;
        // some particles die during the test, some are already dead
        buf.life[i] = next() * 2.0 - 0.2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lowlevel_abi::ParticleBuffers;

    fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs() / x.abs().max(1.0)).fold(0.0, f32::max)
    }

    fn check(n: usize, seed: u64, prm: ParticleParams, steps: usize) {
        let mut a = ParticleBuffers::default();
        fill_test_data(&mut a, n, seed);
        let mut b = a.clone();
        let mut c = a.clone();
        for _ in 0..steps {
            let la = integrate_scalar(&mut a.view_mut(), prm);
            let lb = integrate_simd(&mut b.view_mut(), prm);
            #[cfg(feature = "parallel")]
            let lc = integrate_simd_par(&mut c.view_mut(), prm, 64);
            #[cfg(not(feature = "parallel"))]
            let lc = integrate_simd(&mut c.view_mut(), prm);
            assert_eq!(la, lb, "live count scalar vs simd");
            assert_eq!(la, lc, "live count scalar vs par");
        }
        for (name, x, y, z) in [
            ("px", &a.px, &b.px, &c.px),
            ("py", &a.py, &b.py, &c.py),
            ("pz", &a.pz, &b.pz, &c.pz),
            ("vx", &a.vx, &b.vx, &c.vx),
            ("vy", &a.vy, &b.vy, &c.vy),
            ("vz", &a.vz, &b.vz, &c.vz),
            ("life", &a.life, &b.life, &c.life),
        ] {
            let e1 = max_rel_err(x, y);
            let e2 = max_rel_err(x, z);
            assert!(e1 <= 1e-5 && e2 <= 1e-5, "{name}: rel err simd={e1} par={e2}");
        }
    }

    #[test]
    fn simd_matches_scalar() {
        let prm = ParticleParams { dt: 1.0 / 60.0, gravity: [0.0, -9.81, 0.0], drag: 0.1 };
        for (n, seed) in [(8, 1), (13, 2), (1000, 3), (100_003, 4)] {
            check(n, seed, prm, 120);
        }
    }

    #[test]
    fn simd_matches_scalar_edge_dt() {
        for dt in [1e-6, 1.0 / 240.0, 1.0 / 30.0] {
            let prm = ParticleParams { dt, gravity: [1.0, -20.0, 0.5], drag: 0.9 };
            check(4097, 99, prm, 30);
        }
    }

    #[test]
    fn dead_particles_frozen() {
        let mut b = ParticleBuffers::default();
        b.reset(8);
        b.vx[0] = 5.0; // life = 0 -> dead
        let prm = ParticleParams { dt: 0.1, gravity: [0.0, -10.0, 0.0], drag: 0.0 };
        assert_eq!(integrate_simd(&mut b.view_mut(), prm), 0);
        assert_eq!((b.px[0], b.vx[0], b.vy[0]), (0.0, 5.0, 0.0));
    }
}
