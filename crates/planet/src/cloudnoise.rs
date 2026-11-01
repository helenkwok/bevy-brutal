//! Tileable 3D noise for the volumetric clouds (Schneider / Hillaire style), N^3 RGBA8:
//! - R: Perlin-Worley: gradient-noise fBm (period 4) remapped by an inverted-Worley fBm, i.e.
//!   billowy, connected cumulus-like blobs;
//! - G, B, A: inverted-Worley fBm at periods 4, 8, 16 cells (erosion octaves, small cauliflower
//!   detail at the cloud edges).
//!
//! Every channel tiles with period N (cells wrap), so the shader samples with REPEAT. Pure
//! function of (N, x, y, z): deterministic across runs and threads.
use rayon::prelude::*;

fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}
fn h3(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    hash(
        (x as u32).wrapping_mul(0x8da6_b343)
            ^ (y as u32).wrapping_mul(0xd816_3841)
            ^ (z as u32).wrapping_mul(0xcb1a_b31f)
            ^ seed.wrapping_mul(0x9e37_79b9),
    )
}
fn unit(h: u32) -> f32 {
    (h >> 8) as f32 / 16_777_216.0
}

/// Inverted Worley F1 in [0, 1] (1 at feature points), `period` cells per unit, wrapping.
fn worley(p: [f32; 3], period: i32, seed: u32) -> f32 {
    let q = p.map(|v| v * period as f32);
    let c = q.map(|v| v.floor() as i32);
    let mut d2 = f32::MAX;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (cx, cy, cz) = (c[0] + dx, c[1] + dy, c[2] + dz);
                let (wx, wy, wz) = (cx.rem_euclid(period), cy.rem_euclid(period), cz.rem_euclid(period));
                let h = h3(wx, wy, wz, seed);
                let f = [unit(h), unit(hash(h ^ 0x68e3_1da4)), unit(hash(h ^ 0x1b56_c4e9))];
                let d = [cx as f32 + f[0] - q[0], cy as f32 + f[1] - q[1], cz as f32 + f[2] - q[2]];
                d2 = d2.min(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            }
        }
    }
    1.0 - d2.sqrt().min(1.0)
}

fn grad(h: u32, d: [f32; 3]) -> f32 {
    // 12 edge directions of the cube
    match h % 12 {
        0 => d[0] + d[1],
        1 => -d[0] + d[1],
        2 => d[0] - d[1],
        3 => -d[0] - d[1],
        4 => d[0] + d[2],
        5 => -d[0] + d[2],
        6 => d[0] - d[2],
        7 => -d[0] - d[2],
        8 => d[1] + d[2],
        9 => -d[1] + d[2],
        10 => d[1] - d[2],
        _ => -d[1] - d[2],
    }
}
/// Periodic gradient noise in about [-1, 1].
fn perlin(p: [f32; 3], period: i32, seed: u32) -> f32 {
    let q = p.map(|v| v * period as f32);
    let c = q.map(|v| v.floor() as i32);
    let f = [q[0] - c[0] as f32, q[1] - c[1] as f32, q[2] - c[2] as f32];
    let s = f.map(|t| t * t * t * (t * (t * 6.0 - 15.0) + 10.0));
    let mut acc = [0.0f32; 8];
    for (i, a) in acc.iter_mut().enumerate() {
        let o = [(i & 1) as i32, ((i >> 1) & 1) as i32, ((i >> 2) & 1) as i32];
        let h = h3(
            (c[0] + o[0]).rem_euclid(period),
            (c[1] + o[1]).rem_euclid(period),
            (c[2] + o[2]).rem_euclid(period),
            seed,
        );
        *a = grad(h, [f[0] - o[0] as f32, f[1] - o[1] as f32, f[2] - o[2] as f32]);
    }
    let l = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x0 = l(acc[0], acc[1], s[0]);
    let x1 = l(acc[2], acc[3], s[0]);
    let x2 = l(acc[4], acc[5], s[0]);
    let x3 = l(acc[6], acc[7], s[0]);
    l(l(x0, x1, s[1]), l(x2, x3, s[1]), s[2])
}
fn worley_fbm(p: [f32; 3], period: i32, seed: u32) -> f32 {
    worley(p, period, seed) * 0.625 + worley(p, period * 2, seed + 1) * 0.25 + worley(p, period * 4, seed + 2) * 0.125
}
fn remap(v: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    c + (v - a) / (b - a) * (d - c)
}

/// Texel (x, y, z) of the N^3 texture, RGBA in [0, 1].
pub fn texel(n: usize, x: usize, y: usize, z: usize) -> [f32; 4] {
    let p = [(x as f32 + 0.5) / n as f32, (y as f32 + 0.5) / n as f32, (z as f32 + 0.5) / n as f32];
    let pf = perlin(p, 4, 7) * 0.5 + perlin(p, 8, 8) * 0.25 + perlin(p, 16, 9) * 0.125;
    let pf = (pf * 0.9 + 0.5).clamp(0.0, 1.0);
    let wf = worley_fbm(p, 4, 11);
    let pw = remap(pf, wf - 1.0, 1.0, 0.0, 1.0).clamp(0.0, 1.0);
    [pw, worley_fbm(p, 4, 21), worley_fbm(p, 8, 31), worley_fbm(p, 16, 41)]
}

/// The whole texture, x fastest, RGBA8.
pub fn generate(n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n * n * n * 4];
    out.par_chunks_mut(n * 4).enumerate().for_each(|(row, o)| {
        let (y, z) = (row % n, row / n);
        for x in 0..n {
            let t = texel(n, x, y, z);
            for k in 0..4 {
                o[x * 4 + k] = (t[k].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tiles_and_spans() {
        let n = 32;
        let a = generate(n);
        // wrap: a texel at x = 0 and the would-be texel at x = n agree (period 1 in p)
        for k in 0..4 {
            let p0 = texel(n, 0, 5, 9)[k];
            let p = [(n as f32 + 0.5) / n as f32, 5.5 / n as f32, 9.5 / n as f32];
            let pn = match k {
                0 => {
                    let pf = (perlin(p, 4, 7) * 0.5 + perlin(p, 8, 8) * 0.25 + perlin(p, 16, 9) * 0.125) * 0.9 + 0.5;
                    remap(pf.clamp(0.0, 1.0), worley_fbm(p, 4, 11) - 1.0, 1.0, 0.0, 1.0).clamp(0.0, 1.0)
                }
                1 => worley_fbm(p, 4, 21),
                2 => worley_fbm(p, 8, 31),
                _ => worley_fbm(p, 16, 41),
            };
            assert!((p0 - pn).abs() < 1e-4, "channel {k} does not tile: {p0} vs {pn}");
        }
        // every channel uses a good part of its range (no flat / saturated channel)
        for k in 0..4 {
            let v: Vec<u8> = a.iter().skip(k).step_by(4).copied().collect();
            let (lo, hi) = (*v.iter().min().unwrap(), *v.iter().max().unwrap());
            let mean = v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
            assert!(hi as i32 - lo as i32 > 120 && mean > 40.0 && mean < 215.0, "channel {k}: {lo}..{hi} mean {mean}");
        }
        assert_eq!(a, generate(n), "deterministic");
    }
}
