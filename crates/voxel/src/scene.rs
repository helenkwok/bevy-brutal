//! Density fields: the cave/overhang demo scene and a closed test blob. Pure f64 functions of
//! world position (R4).
use crate::DensityField;

#[inline]
fn hash3(x: i64, y: i64, z: i64, seed: u64) -> f64 {
    let mut h = seed ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= (z as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h >> 11) as f64 * (2.0 / (1u64 << 53) as f64) - 1.0
}

/// Trilinear value noise in [-1, 1] with smoothstep fade.
pub fn vnoise3(p: [f64; 3], seed: u64) -> f64 {
    let f = p.map(f64::floor);
    let i = f.map(|v| v as i64);
    let t = [0, 1, 2].map(|k| {
        let u = p[k] - f[k];
        u * u * (3.0 - 2.0 * u)
    });
    let l = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let h = |dx, dy, dz| hash3(i[0] + dx, i[1] + dy, i[2] + dz, seed);
    let x00 = l(h(0, 0, 0), h(1, 0, 0), t[0]);
    let x10 = l(h(0, 1, 0), h(1, 1, 0), t[0]);
    let x01 = l(h(0, 0, 1), h(1, 0, 1), t[0]);
    let x11 = l(h(0, 1, 1), h(1, 1, 1), t[0]);
    l(l(x00, x10, t[1]), l(x01, x11, t[1]), t[2])
}

fn fbm3(p: [f64; 3], seed: u64) -> f64 {
    let (mut a, mut f, mut s) = (0.5, 1.0, 0.0);
    for o in 0..3 {
        s += a * vnoise3(p.map(|c| c * f), seed + o);
        a *= 0.5;
        f *= 2.03;
    }
    s
}

fn round_box(p: [f64; 3], c: [f64; 3], half: [f64; 3], r: f64) -> f64 {
    let q = [0, 1, 2].map(|i| (p[i] - c[i]).abs() - (half[i] - r));
    let out = q.map(|v| v.max(0.0));
    (out[0] * out[0] + out[1] * out[1] + out[2] * out[2]).sqrt() + q[0].max(q[1]).max(q[2]).min(0.0) - r
}

fn capsule(p: [f64; 3], a: [f64; 3], b: [f64; 3], r: f64) -> f64 {
    let pa = [0, 1, 2].map(|i| p[i] - a[i]);
    let ba = [0, 1, 2].map(|i| b[i] - a[i]);
    let h = ((pa[0] * ba[0] + pa[1] * ba[1] + pa[2] * ba[2]) / (ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2]))
        .clamp(0.0, 1.0);
    let d = [0, 1, 2].map(|i| pa[i] - ba[i] * h);
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() - r
}

fn sphere(p: [f64; 3], c: [f64; 3], r: f64) -> f64 {
    let d = [0, 1, 2].map(|i| p[i] - c[i]);
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() - r
}

/// Rolling ground + a rock mass with an overhanging lip and a cave tunnel/cavern through it.
/// Roughly x in [-16, 16], y in [-4, 14], z in [-12, 16] metres.
pub struct CaveScene {
    pub seed: u64,
}

impl DensityField for CaveScene {
    fn density(&self, p: [f64; 3]) -> f64 {
        let [x, y, z] = p;
        let ground =
            y - (1.2 * (0.21 * x).sin() + 1.0 * (0.17 * z).cos() + 0.8 * fbm3([x * 0.3, 0.0, z * 0.3], self.seed));
        let disp = 0.9 * fbm3(p.map(|c| c * 0.45), self.seed + 17);
        let rock = round_box(p, [0.0, 4.0, 0.0], [7.0, 6.0, 5.0], 1.5) + disp;
        let lip = round_box(p, [0.0, 9.2, 6.0], [6.5, 1.4, 3.5], 0.8) + 0.7 * disp;
        let solid = ground.min(rock).min(lip);
        let tn = 0.5 * fbm3(p.map(|c| c * 0.6), self.seed + 31);
        let tunnel = capsule(p, [-14.0, 2.2, 1.5], [14.0, 3.2, -0.5], 2.0) + tn;
        let cavern = sphere(p, [1.5, 3.8, 0.5], 3.6) + tn;
        let mouth = capsule(p, [-3.0, 3.0, 2.0], [-3.0, 3.0, 9.0], 1.7) + tn; // opens under the lip
        solid.max(-tunnel.min(cavern).min(mouth))
    }
}

/// A closed noisy blob with a through-tunnel (genus 1), well inside `[-r*1.6, r*1.6]^3`.
pub struct BlobScene {
    pub radius: f64,
    pub seed: u64,
}

impl DensityField for BlobScene {
    fn density(&self, p: [f64; 3]) -> f64 {
        let r = self.radius;
        let blob = sphere(p, [0.0; 3], r) + 0.15 * r * fbm3(p.map(|c| c * 2.0 / r), self.seed);
        let tunnel = capsule(p, [-2.0 * r, 0.1 * r, 0.0], [2.0 * r, -0.1 * r, 0.05 * r], 0.3 * r);
        blob.max(-tunnel)
    }
}

/// Analytic plane `d = y - h` (used to hit exact zeros on lattice points, R1).
pub struct Plane {
    pub h: f64,
}

impl DensityField for Plane {
    fn density(&self, p: [f64; 3]) -> f64 {
        p[1] - self.h
    }
}

/// Fuzz field (adversarial densities on lattice points): exact zeros (ties at corners), ±1
/// quantum, sub-quantum values (quantise to 0) and extreme magnitude ratios. Strictly positive
/// outside a ball, so the surface is closed inside the meshed region.
pub struct Adversarial {
    pub seed: u64,
    pub voxel_m: f64,
    pub radius_m: f64,
}

impl DensityField for Adversarial {
    fn density(&self, p: [f64; 3]) -> f64 {
        if (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() > self.radius_m {
            return 1.0;
        }
        let i = p.map(|c| (c / self.voxel_m).round() as i64);
        let q = self.voxel_m / 1024.0; // one density quantum
        let h = hash3(i[0], i[1], i[2], self.seed);
        let k = ((h + 1.0) * 6.0) as usize; // 0..=12
        [0.0, -0.0, q, -q, 0.4 * q, -0.4 * q, 1e3, -1e3, 1e-9, -1e-9, 0.5, -0.5, -q][k.min(12)]
    }
}

/// Ridged mountains (up to ~70 m) with a natural arch, as a density field.
pub struct Mountains {
    pub seed: u64,
}

impl DensityField for Mountains {
    fn density(&self, p: [f64; 3]) -> f64 {
        let [x, y, z] = p;
        let mut h = 0.0;
        let (mut a, mut f) = (38.0, 0.012);
        for o in 0..4 {
            let r = 1.0 - vnoise3([x * f, 0.37 * o as f64, z * f], self.seed + o).abs();
            h += a * r * r;
            a *= 0.45;
            f *= 2.1;
        }
        let ground = y - (h - 18.0);
        let arch =
            round_box(p, [10.0, 34.0, -6.0], [16.0, 3.5, 4.0], 2.0) + 1.2 * fbm3(p.map(|c| c * 0.2), self.seed + 9);
        let hole = capsule(p, [10.0, 26.0, -14.0], [10.0, 26.0, 2.0], 7.5);
        ground.min(arch.max(-hole)) + 0.6 * fbm3(p.map(|c| c * 0.15), self.seed + 5)
    }
}

/// Inside the rock: a network of tunnels and two caverns (the camera sits in one).
pub struct Underground {
    pub seed: u64,
}

impl DensityField for Underground {
    fn density(&self, p: [f64; 3]) -> f64 {
        let n = 0.6 * fbm3(p.map(|c| c * 0.35), self.seed);
        let t1 = capsule(p, [-30.0, 0.0, 0.0], [30.0, 2.0, 0.0], 2.6);
        let t2 = capsule(p, [4.0, -6.0, -30.0], [-4.0, 4.0, 30.0], 2.2);
        let t3 = capsule(p, [-20.0, 8.0, -20.0], [20.0, -4.0, 20.0], 1.8);
        let c1 = sphere(p, [0.0, 1.0, 0.0], 7.0);
        let c2 = sphere(p, [18.0, -2.0, 14.0], 5.0);
        let air = t1.min(t2).min(t3).min(c1).min(c2) + n;
        // pillars inside the main cavern
        let pillar = capsule(p, [2.5, -8.0, 2.0], [2.0, 9.0, 1.5], 1.1) + 0.3 * n;
        (-air).min(pillar)
    }
}

/// Seabed for the underwater scene: dunes, boulders and a rock arch, about 20 m below sea level.
pub struct Seabed {
    pub seed: u64,
}

impl DensityField for Seabed {
    fn density(&self, p: [f64; 3]) -> f64 {
        let [x, y, z] = p;
        let dunes = 0.9 * (0.3 * x + 0.8 * (0.05 * z).sin()).sin() + 1.5 * fbm3([x * 0.06, 0.0, z * 0.06], self.seed);
        let ground = y - (-20.0 + dunes);
        let rock =
            round_box(p, [-6.0, -15.0, -4.0], [5.0, 5.0, 4.0], 1.5) + 1.0 * fbm3(p.map(|c| c * 0.3), self.seed + 3);
        let arch_hole = capsule(p, [-6.0, -17.0, -10.0], [-6.0, -17.0, 10.0], 3.0);
        let boulders = sphere(p, [8.0, -19.5, 6.0], 2.5).min(sphere(p, [12.0, -19.8, -3.0], 1.6))
            + 0.4 * fbm3(p.map(|c| c * 0.8), self.seed + 4);
        ground.min(rock.max(-arch_hole)).min(boulders)
    }
}
