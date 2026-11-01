//! Gradient (Perlin-style) noise with quintic fade, 2D and 3D, plus helpers for band-limited
//! fractal sums. Deterministic pure f64 functions (H7): no global state, no float-mode tricks.

#[inline]
fn hash(mut h: u64) -> u64 {
    h ^= h >> 31;
    h = h.wrapping_mul(0x7FB5_D329_728E_A185);
    h ^= h >> 27;
    h = h.wrapping_mul(0x81DA_DEF4_BC2D_D44D);
    h ^ (h >> 33)
}

#[inline]
fn h2(x: i64, z: i64, seed: u64) -> u64 {
    hash(seed ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (z as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
}

#[inline]
fn h3(x: i64, y: i64, z: i64, seed: u64) -> u64 {
    hash(seed ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ (z as u64).wrapping_mul(0x1656_67B1_9E37_79F9))
}

/// Uniform in [0, 1) from a lattice hash (for jitter / per-cell randomness).
#[inline]
pub fn rand2(x: i64, z: i64, seed: u64) -> f64 {
    (h2(x, z, seed) >> 11) as f64 / (1u64 << 53) as f64
}

#[inline]
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

const G2: [[f64; 2]; 8] = [
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, 1.0],
    [0.0, -1.0],
    [std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2],
    [-std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2],
    [std::f64::consts::FRAC_1_SQRT_2, -std::f64::consts::FRAC_1_SQRT_2],
    [-std::f64::consts::FRAC_1_SQRT_2, -std::f64::consts::FRAC_1_SQRT_2],
];

/// 2D gradient noise, roughly in [-1, 1].
pub fn perlin2(x: f64, z: f64, seed: u64) -> f64 {
    let (fx, fz) = (x.floor(), z.floor());
    let (ix, iz) = (fx as i64, fz as i64);
    let (tx, tz) = (x - fx, z - fz);
    let g = |dx: i64, dz: i64| {
        let v = G2[(h2(ix + dx, iz + dz, seed) & 7) as usize];
        v[0] * (tx - dx as f64) + v[1] * (tz - dz as f64)
    };
    let (u, w) = (fade(tx), fade(tz));
    let a = g(0, 0) + (g(1, 0) - g(0, 0)) * u;
    let b = g(0, 1) + (g(1, 1) - g(0, 1)) * u;
    (a + (b - a) * w) * 1.41
}

const G3: [[f64; 3]; 12] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
];

/// 3D gradient noise, roughly in [-1, 1].
pub fn perlin3(p: [f64; 3], seed: u64) -> f64 {
    let f = p.map(f64::floor);
    let i = f.map(|v| v as i64);
    let t = [p[0] - f[0], p[1] - f[1], p[2] - f[2]];
    let g = |dx: i64, dy: i64, dz: i64| {
        let v = G3[(h3(i[0] + dx, i[1] + dy, i[2] + dz, seed) % 12) as usize];
        v[0] * (t[0] - dx as f64) + v[1] * (t[1] - dy as f64) + v[2] * (t[2] - dz as f64)
    };
    let u = t.map(fade);
    let l = |a: f64, b: f64, s: f64| a + (b - a) * s;
    let x00 = l(g(0, 0, 0), g(1, 0, 0), u[0]);
    let x10 = l(g(0, 1, 0), g(1, 1, 0), u[0]);
    let x01 = l(g(0, 0, 1), g(1, 0, 1), u[0]);
    let x11 = l(g(0, 1, 1), g(1, 1, 1), u[0]);
    l(l(x00, x10, u[1]), l(x01, x11, u[1]), u[2]) * 1.1
}

/// Weight of an octave with wavelength `lambda_m` when sampling at cell size `cell_m`:
/// 1 well above 4 cells, 0 below 2 cells (band-limiting so coarse LODs do not alias).
#[inline]
pub fn octave_weight(lambda_m: f64, cell_m: f64) -> f64 {
    let t = ((lambda_m / cell_m - 2.0) / 2.0).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Wider fade for terrain seen at long range: 1 above 6 cells, 0 below 3. Octaves with only 2-4
/// samples per wavelength mesh as stair-stepped stripes that read as vertical smears on cliffs.
#[inline]
pub fn octave_weight_wide(lambda_m: f64, cell_m: f64) -> f64 {
    let t = ((lambda_m / cell_m - 3.0) / 3.0).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Smooth |x| with a rounded crease of half-width `e` (C1; == |x| for |x| >= e).
#[inline]
pub fn soft_abs(x: f64, e: f64) -> f64 {
    if e <= 0.0 || x.abs() >= e { x.abs() } else { 0.5 * (x * x / e + e) }
}

/// Band-limited 3D fbm: `octaves` from base wavelength `lambda_m`, gain 0.5, lacunarity ~2.
pub fn fbm3(p: [f64; 3], lambda_m: f64, octaves: u32, cell_m: f64, seed: u64) -> f64 {
    let (mut s, mut a, mut l) = (0.0, 1.0, lambda_m);
    for o in 0..octaves {
        let w = octave_weight(l, cell_m);
        if w <= 0.0 {
            break;
        }
        // rotate each octave a little (decorrelates the lattice axes)
        let k = 1.0 / l;
        let q = [p[0] * k + 0.31 * o as f64, p[1] * k - 0.17 * o as f64, p[2] * k + 0.53 * o as f64];
        let r = [0.8 * q[0] - 0.6 * q[2], q[1], 0.6 * q[0] + 0.8 * q[2]];
        s += a * w * perlin3(if o % 2 == 1 { r } else { q }, seed.wrapping_add(o as u64 * 7919));
        a *= 0.5;
        l *= 0.497;
    }
    s
}

/// [`fbm3`] with the [`octave_weight_wide`] fade.
pub fn fbm3_wide(p: [f64; 3], lambda_m: f64, octaves: u32, cell_m: f64, seed: u64) -> f64 {
    let (mut s, mut a, mut l) = (0.0, 1.0, lambda_m);
    for o in 0..octaves {
        let w = octave_weight_wide(l, cell_m);
        if w <= 0.0 {
            break;
        }
        // rotate each octave a little (decorrelates the lattice axes)
        let k = 1.0 / l;
        let q = [p[0] * k + 0.31 * o as f64, p[1] * k - 0.17 * o as f64, p[2] * k + 0.53 * o as f64];
        let r = [0.8 * q[0] - 0.6 * q[2], q[1], 0.6 * q[0] + 0.8 * q[2]];
        s += a * w * perlin3(if o % 2 == 1 { r } else { q }, seed.wrapping_add(o as u64 * 7919));
        a *= 0.5;
        l *= 0.497;
    }
    s
}

/// 2D gradient noise with analytic derivatives: `(v, dv/dx, dv/dz)` (same values as [`perlin2`]).
pub fn perlin2d(x: f64, z: f64, seed: u64) -> (f64, f64, f64) {
    let (fx, fz) = (x.floor(), z.floor());
    let (ix, iz) = (fx as i64, fz as i64);
    let (tx, tz) = (x - fx, z - fz);
    let gr = |dx: i64, dz: i64| G2[(h2(ix + dx, iz + dz, seed) & 7) as usize];
    let (g00, g10, g01, g11) = (gr(0, 0), gr(1, 0), gr(0, 1), gr(1, 1));
    let dot = |g: [f64; 2], dx: f64, dz: f64| g[0] * (tx - dx) + g[1] * (tz - dz);
    let (v00, v10, v01, v11) = (dot(g00, 0.0, 0.0), dot(g10, 1.0, 0.0), dot(g01, 0.0, 1.0), dot(g11, 1.0, 1.0));
    let (u, w) = (fade(tx), fade(tz));
    let du = 30.0 * tx * tx * (tx * (tx - 2.0) + 1.0);
    let dw = 30.0 * tz * tz * (tz * (tz - 2.0) + 1.0);
    let a = v00 + (v10 - v00) * u;
    let b = v01 + (v11 - v01) * u;
    let v = a + (b - a) * w;
    let dax = g00[0] + (g10[0] - g00[0]) * u + (v10 - v00) * du;
    let dbx = g01[0] + (g11[0] - g01[0]) * u + (v11 - v01) * du;
    let daz = g00[1] + (g10[1] - g00[1]) * u;
    let dbz = g01[1] + (g11[1] - g01[1]) * u;
    let dx = dax + (dbx - dax) * w;
    let dz = daz + (dbz - daz) * w + (b - a) * dw;
    (v * 1.41, dx * 1.41, dz * 1.41)
}
