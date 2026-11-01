//! Earth round 5, "grounded Alps": real elevation (Copernicus GLO-30) under the procedural
//! [`crate::fields::Alpine`] field. A [`Ground`] holds the DEM resampled onto the mountains
//! world frame (x east, z south, metres) in two nested square levels around the vista, each
//! with a box-filtered mip pyramid, and a *procedural core* around the vista where the
//! hand-tuned field is kept (the default view and the last descent frames stay the scene they
//! were). `Alpine::grounded` blends its base height to the DEM by [`Ground::weight`]; the
//! gullies, crags, tors and overhangs then run on top of the real relief.
//!
//! Sampling is a uniform cubic B-spline (C2, no facets) on the mip whose texel matches the
//! sampled cell (trilinear across mips), so `height(x, z, cell)` is band-limited the way the
//! field's own octaves are (`octave_weight_wide`: wavelengths below ~3-6 cells fade).
//!
//! File `GRD1` (little endian): magic, f64 cx, cz, core (yaw, disc r0, r1, wedge r0, r1, half
//! angle a0, a1), u32 level count, per level (u32 size, f64 half extent), then the levels'
//! f32 heights (texel centres, row = +z), outer level first.

/// The procedural core: a disc around the vista plus a wedge along the default view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Core {
    /// Default view yaw (rad): forward = (sin yaw, -cos yaw) in (x, z).
    pub yaw: f64,
    /// Disc: fully procedural inside `disc.0`, DEM beyond `disc.1` (m).
    pub disc: (f64, f64),
    /// Wedge radial extent (m) and half angle (rad), each as (full, gone).
    pub wedge_r: (f64, f64),
    pub wedge_a: (f64, f64),
}

impl Core {
    /// Earth round 5's core: the default view's whole footprint (rings and fog to 26 km) stays
    /// procedural, so the default view was the round-4 scene byte for byte.
    pub const E5: Core = Core { yaw: 2.75, disc: (5_000.0, 9_000.0), wedge_r: (28_000.0, 40_000.0), wedge_a: (0.78, 1.05) };
    /// Earth round 6 (default): only the vista's near field (the valley, the forests and the
    /// big peak on the right, all within ~5 km) stays procedural; real relief (the Monte Rosa /
    /// Matterhorn side, 14-27 km out) shows from ~10 km. Wide ramps (4.5-10 km disc, 8-14 km
    /// wedge) so the height lerp reads as foothills, not a step.
    pub const E6: Core = Core { yaw: 2.75, disc: (4_500.0, 10_000.0), wedge_r: (8_000.0, 14_000.0), wedge_a: (0.78, 1.05) };
}

impl Default for Core {
    fn default() -> Self {
        // default view hfov/2 = atan(tan(0.41) 16/9) = 0.66 rad
        Self::E6
    }
}

fn ss(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// One square level: `size`^2 texel centres over `[c - half, c + half]^2`, plus mips.
#[derive(Clone, Debug)]
pub struct GroundLevel {
    pub size: usize,
    pub half_m: f64,
    /// mips[0] = the level (size^2), each next one half the side (2x2 box).
    pub mips: Vec<Vec<f32>>,
}

impl GroundLevel {
    pub fn new(size: usize, half_m: f64, h: Vec<f32>) -> Self {
        assert_eq!(h.len(), size * size);
        let mut mips = vec![h];
        let mut n = size;
        while n > 64 && n.is_multiple_of(2) {
            let p = mips.last().expect("mip");
            let m = n / 2;
            let mut q = vec![0f32; m * m];
            for j in 0..m {
                for i in 0..m {
                    let s = |a: usize, b: usize| p[(2 * j + b) * n + 2 * i + a];
                    q[j * m + i] = 0.25 * (s(0, 0) + s(1, 0) + s(0, 1) + s(1, 1));
                }
            }
            mips.push(q);
            n = m;
        }
        Self { size, half_m, mips }
    }

    pub fn step(&self) -> f64 {
        2.0 * self.half_m / self.size as f64
    }

    /// Cubic B-spline sample of mip `l` at level-local (dx, dz) (m from the centre).
    fn bspline(&self, l: usize, dx: f64, dz: f64) -> f64 {
        let n = self.size >> l;
        let t = self.step() * (1u64 << l) as f64;
        let u = (dx + self.half_m) / t - 0.5;
        let v = (dz + self.half_m) / t - 0.5;
        let (iu, iv) = (u.floor(), v.floor());
        let (fu, fv) = (u - iu, v - iv);
        let w = |f: f64| {
            let g = 1.0 - f;
            [g * g * g / 6.0, (4.0 - 6.0 * f * f + 3.0 * f * f * f) / 6.0, (1.0 + 3.0 * f + 3.0 * f * f - 3.0 * f * f * f) / 6.0, f * f * f / 6.0]
        };
        let (wu, wv) = (w(fu), w(fv));
        let m = &self.mips[l];
        let cl = |k: f64| (k.max(0.0) as usize).min(n - 1);
        let mut s = 0.0;
        for (b, wb) in wv.iter().enumerate() {
            let row = cl(iv - 1.0 + b as f64) * n;
            let mut r = 0.0;
            for (a, wa) in wu.iter().enumerate() {
                r += wa * m[row + cl(iu - 1.0 + a as f64)] as f64;
            }
            s += wb * r;
        }
        s
    }

    /// Band-limited height for filter `cell` (m): trilinear across mips.
    fn sample(&self, dx: f64, dz: f64, cell: f64) -> f64 {
        let lod = (cell.max(1e-3) / self.step()).log2().clamp(0.0, (self.mips.len() - 1) as f64);
        let l0 = lod.floor() as usize;
        let f = lod - l0 as f64;
        let a = self.bspline(l0, dx, dz);
        if f <= 1e-9 || l0 + 1 >= self.mips.len() { a } else { a + (self.bspline(l0 + 1, dx, dz) - a) * f }
    }
}

/// Real elevation around the vista, in the mountains world frame.
#[derive(Clone, Debug)]
pub struct Ground {
    /// World (x, z) of the vista (the levels' centre).
    pub cx: f64,
    pub cz: f64,
    pub core: Core,
    /// Outer level first.
    pub levels: Vec<GroundLevel>,
    /// Height range of level 0 data (m).
    pub h_min: f64,
    pub h_max: f64,
}

const MAGIC: &[u8; 4] = b"GRD1";

impl Ground {
    pub fn new(cx: f64, cz: f64, core: Core, levels: Vec<GroundLevel>) -> Self {
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        for l in &levels {
            for &h in &l.mips[0] {
                lo = lo.min(h as f64);
                hi = hi.max(h as f64);
            }
        }
        Self { cx, cz, core, levels, h_min: lo, h_max: hi }
    }

    /// DEM weight in [0, 1]: 0 in the procedural core, 1 beyond it; fades to 0 past the outer
    /// level's extent too (the field is procedural again where there is no data).
    pub fn weight(&self, x: f64, z: f64) -> f64 {
        let (dx, dz) = (x - self.cx, z - self.cz);
        let d = dx.hypot(dz);
        let c = &self.core;
        let disc = 1.0 - ss(c.disc.0, c.disc.1, d);
        let wedge = if d < c.wedge_r.1 {
            let (fx, fz) = (c.yaw.sin(), -c.yaw.cos());
            let cosa = if d > 0.0 { ((dx * fx + dz * fz) / d).clamp(-1.0, 1.0) } else { 1.0 };
            (1.0 - ss(c.wedge_r.0, c.wedge_r.1, d)) * (1.0 - ss(c.wedge_a.0, c.wedge_a.1, cosa.acos()))
        } else {
            0.0
        };
        let core = disc.max(wedge);
        let edge = match self.levels.first() {
            Some(l) => 1.0 - ss(0.975, 0.998, dx.abs().max(dz.abs()) / l.half_m),
            None => 0.0,
        };
        (1.0 - core) * edge
    }

    /// Band-limited DEM height (m) at world (x, z) for filter `cell`; nested levels blend over
    /// 86-97 % of the inner level's half extent.
    pub fn height(&self, x: f64, z: f64, cell: f64) -> f64 {
        let (dx, dz) = (x - self.cx, z - self.cz);
        let mut h = 0.0;
        for (k, l) in self.levels.iter().enumerate() {
            let e = dx.abs().max(dz.abs()) / l.half_m;
            let w = if k == 0 { 1.0 } else { 1.0 - ss(0.86, 0.97, e) };
            if w <= 0.0 {
                continue;
            }
            let s = l.sample(dx, dz, cell);
            h = if k == 0 { s } else { h + (s - h) * w };
        }
        h
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        let c = &self.core;
        for v in [self.cx, self.cz, c.yaw, c.disc.0, c.disc.1, c.wedge_r.0, c.wedge_r.1, c.wedge_a.0, c.wedge_a.1] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&(self.levels.len() as u32).to_le_bytes());
        for l in &self.levels {
            b.extend_from_slice(&(l.size as u32).to_le_bytes());
            b.extend_from_slice(&l.half_m.to_le_bytes());
        }
        for l in &self.levels {
            for &h in &l.mips[0] {
                b.extend_from_slice(&h.to_le_bytes());
            }
        }
        b
    }

    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() < 4 + 9 * 8 + 4 || &b[..4] != MAGIC {
            return None;
        }
        let mut o = 4;
        let f64r = |o: &mut usize| {
            let v = f64::from_le_bytes(b.get(*o..*o + 8)?.try_into().ok()?);
            *o += 8;
            Some(v)
        };
        let mut h = [0.0; 9];
        for v in &mut h {
            *v = f64r(&mut o)?;
        }
        let n = u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?) as usize;
        o += 4;
        let mut dims = Vec::new();
        for _ in 0..n {
            let s = u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?) as usize;
            o += 4;
            dims.push((s, f64r(&mut o)?));
        }
        let mut levels = Vec::new();
        for (size, half) in dims {
            let len = size * size * 4;
            let raw = b.get(o..o + len)?;
            o += len;
            let v: Vec<f32> = raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
            levels.push(GroundLevel::new(size, half, v));
        }
        let core = Core { yaw: h[2], disc: (h[3], h[4]), wedge_r: (h[5], h[6]), wedge_a: (h[7], h[8]) };
        Some(Self::new(h[0], h[1], core, levels))
    }

    /// Load a `GRD1` file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let b = std::fs::read(path)?;
        let mut g = Self::from_bytes(&b).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{path}: not a GRD1 ground file")))?;
        // GROUND_CORE="disc0,disc1,wedge0,wedge1[,angle0,angle1]" (m, rad) overrides the baked
        // core extents; GROUND_CORE=e5 / e6 selects a round's core.
        match std::env::var("GROUND_CORE").ok().as_deref().map(str::trim) {
            Some("e5") => g.core = Core::E5,
            Some("e6") => g.core = Core::E6,
            Some(s) => {
                if let Some(v) = s.split(',').map(|t| t.trim().parse::<f64>().ok()).collect::<Option<Vec<_>>>()
                    && (v.len() == 4 || v.len() == 6)
                {
                    g.core.disc = (v[0], v[1]);
                    g.core.wedge_r = (v[2], v[3]);
                    if v.len() == 6 {
                        g.core.wedge_a = (v[4], v[5]);
                    }
                }
            }
            None => {}
        }
        Ok(g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(size: usize, half: f64, f: impl Fn(f64, f64) -> f64) -> GroundLevel {
        let t = 2.0 * half / size as f64;
        let h = (0..size * size).map(|k| f(-half + ((k % size) as f64 + 0.5) * t, -half + ((k / size) as f64 + 0.5) * t) as f32).collect();
        GroundLevel::new(size, half, h)
    }

    /// The cubic B-spline reproduces linear functions exactly (away from the clamped border),
    /// at every mip; mips stop at 64.
    #[test]
    fn bspline_is_exact_on_planes_and_mips_stop_at_64() {
        let l = plane(256, 12_800.0, |x, z| 500.0 + 0.1 * x - 0.05 * z);
        assert_eq!(l.mips.len(), 3);
        for cell in [10.0, 100.0, 150.0, 400.0] {
            for (x, z) in [(0.0, 0.0), (1234.5, -987.0), (-5000.0, 4000.0)] {
                let h = l.sample(x, z, cell);
                assert!((h - (500.0 + 0.1 * x - 0.05 * z)).abs() < 1e-3, "cell {cell} at ({x}, {z}): {h}");
            }
        }
    }

    #[test]
    fn weight_core_edge_and_file_round_trip() {
        let core = Core::default();
        let g = Ground::new(7700.0, -4200.0, core, vec![plane(128, 170_000.0, |x, _| x * 0.01), plane(128, 42_500.0, |_, z| z * 0.01)]);
        // procedural at the vista and along the default view inside the wedge, DEM elsewhere
        assert_eq!(g.weight(7700.0, -4200.0), 0.0);
        let (fx, fz) = (core.yaw.sin(), -core.yaw.cos());
        assert_eq!(g.weight(7700.0 + 7_000.0 * fx, -4200.0 + 7_000.0 * fz), 0.0);
        assert_eq!(g.weight(7700.0 + 20_000.0 * fx, -4200.0 + 20_000.0 * fz), 1.0);
        assert_eq!(g.weight(7700.0 - 20_000.0 * fx, -4200.0 - 20_000.0 * fz), 1.0);
        // ramps: partly procedural between the disc's ends, monotone outward
        let w = |d: f64| g.weight(7700.0 - d * fx, -4200.0 - d * fz);
        assert!(w(6_000.0) > 0.0 && w(6_000.0) < w(8_000.0) && w(8_000.0) < 1.0);
        let e5 = Ground::new(7700.0, -4200.0, Core::E5, g.levels.clone());
        assert_eq!(e5.weight(7700.0 + 20_000.0 * fx, -4200.0 + 20_000.0 * fz), 0.0);
        // fades out before the outer level's edge (no data beyond)
        assert_eq!(g.weight(7700.0 + 169_900.0, -4200.0), 0.0);
        assert!(g.h_min < g.h_max);
        let b = g.to_bytes();
        let r = Ground::from_bytes(&b).expect("GRD1");
        assert_eq!((r.cx, r.cz, r.core, r.levels.len()), (g.cx, g.cz, g.core, 2));
        for (x, z) in [(0.0, 0.0), (30_000.0, -60_000.0), (7000.0, 9000.0)] {
            assert_eq!(r.height(x, z, 50.0), g.height(x, z, 50.0));
        }
        assert!(Ground::from_bytes(&b[..b.len() - 1]).is_none());
        assert!(Ground::from_bytes(b"GRD0").is_none());
    }
}
