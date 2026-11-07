//! Bounded stellar projection/selection and explicitly procedural reference population.
use glam::DVec3;
/// GPU submission/procedural cap, NOT an observed catalogue input-scan limit.
pub const MAX_STARS: usize = 65_536;
pub const AU: f64 = 149_597_870_700.0;
pub const LY: f64 = 9.460_730_472_580_8e15;
/// Matches the existing solar System frame anchor; not a measured galactocentric position.
pub const SOL: DVec3 = DVec3::new(2.46e20, 0.0, 6.0e17);
pub const SIGMA: f64 = 0.7;
pub const SUPPORT: f64 = 4.0 * SIGMA;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Star {
    pub id: u64,
    pub pos: DVec3,
    /// Inverse-square amplitude at 1 AU. Synthetic mode uses illustrative Lsun;
    /// GCNS uses a G-band relative-flux proxy, never bolometric luminosity.
    pub lum: f64,
    pub rgb: [f64; 3],
}
pub struct Population {
    pub stars: Vec<Star>,
}
impl Population {
    /// Synthetic four-arm disk + bulge + 256 nearby tracers. Log-uniform 0.1..1e5 Lsun
    /// tracer weights are illustrative, NOT an IMF, observational catalogue or galaxy luminosity.
    pub fn seeded(seed: u64, count: usize) -> Self {
        let mut rng = seed;
        let mut rnd = || {
            rng = rng.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut stars = Vec::with_capacity(count.min(MAX_STARS));
        if count > 0 {
            stars.push(Star { id: 0, pos: SOL, lum: 1.0, rgb: [1.0, 1.0, 1.0] });
        }
        for id in 1..count.min(MAX_STARS) {
            let (u, v, w, t) = (rnd(), rnd(), rnd(), rnd());
            let (pos, hot) = if id <= 256 {
                let r = (20.0 + 80.0 * u.cbrt()) * LY;
                let z = 2.0 * v - 1.0;
                let a = w * std::f64::consts::TAU;
                (SOL + DVec3::new((1.0 - z * z).sqrt() * a.cos(), (1.0 - z * z).sqrt() * a.sin(), z) * r, t)
            } else if u < 0.18 {
                let r = 4e19 * (-v.max(1e-9).ln()).sqrt();
                let z = w * 2.0 - 1.0;
                let a = t * std::f64::consts::TAU;
                (DVec3::new((1.0 - z * z).sqrt() * a.cos(), (1.0 - z * z).sqrt() * a.sin(), z) * r, 0.1 + 0.25 * rnd())
            } else {
                // Smooth centrally concentrated radial distribution, not a measured density law.
                let r = 5e20 * v.powf(1.3);
                let arm = (id % 4) as f64 * std::f64::consts::FRAC_PI_2;
                let scatter = (-2.0 * w.max(1e-9).ln()).sqrt() * (t * std::f64::consts::TAU).cos();
                let angle = if u < 0.40 {
                    w * std::f64::consts::TAU
                } else {
                    arm + (r / 6e19 + 0.15).ln() / 0.32 + scatter * (0.23 + 0.18 * (1.0 - r / 5e20))
                };
                (DVec3::new(r * angle.cos(), r * angle.sin(), (t - 0.5) * 1e19), 0.25 + 0.75 * rnd())
            };
            let rgb =
                if hot < 0.35 { [1.0, 0.55 + hot, 0.25 + hot] } else { [1.2 - hot * 0.35, 0.9, 0.65 + hot * 0.65] };
            let y = rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722;
            stars.push(Star { id: id as u64, pos, lum: 10f64.powf(-1.0 + 6.0 * rnd()), rgb: rgb.map(|c| c / y) });
        }
        Self { stars }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub origin: DVec3,
    pub pos: DVec3,
    pub right: DVec3,
    pub up: DVec3,
    pub forward: DVec3,
    pub fov: f64,
    pub width: u32,
    pub height: u32,
}
impl Camera {
    pub fn look_at(pos: DVec3, target: DVec3, up: DVec3, fov: f64, width: u32, height: u32) -> Option<Self> {
        if !pos.is_finite()
            || !target.is_finite()
            || !fov.is_finite()
            || !(0.01..3.0).contains(&fov)
            || width == 0
            || width > 4096
            || height == 0
            || height > 4096
        {
            return None;
        }
        let forward = (target - pos).try_normalize()?;
        let right = up.cross(forward).try_normalize()?;
        Some(Self { origin: DVec3::ZERO, pos, right, up: forward.cross(right), forward, fov, width, height })
    }
    pub fn relative(&self, s: &Star) -> DVec3 {
        (s.pos - self.origin) - self.pos
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Source {
    pub id: u64,
    pub center: [f64; 2],
    pub flux: f64,
    pub rgb: [f64; 3],
}
impl Source {
    pub fn bounds(&self) -> (i32, i32, i32, i32) {
        (
            (self.center[0] - SUPPORT - 0.5).ceil() as i32,
            (self.center[0] + SUPPORT - 0.5).floor() as i32,
            (self.center[1] - SUPPORT - 0.5).ceil() as i32,
            (self.center[1] + SUPPORT - 0.5).floor() as i32,
        )
    }
    pub fn weight(&self, x: i32, y: i32) -> f64 {
        let dx = x as f64 + 0.5 - self.center[0];
        let dy = y as f64 + 0.5 - self.center[1];
        (-0.5 * (dx * dx + dy * dy) / (SIGMA * SIGMA)).exp()
    }
    pub fn norm(&self) -> f64 {
        let (x0, x1, y0, y1) = self.bounds();
        let weight = |p: f64, c: f64| (-0.5 * ((p + 0.5 - c) / SIGMA).powi(2)).exp();
        let x: f64 = (x0..=x1).map(|x| weight(x as f64, self.center[0])).sum();
        let y: f64 = (y0..=y1).map(|y| weight(y as f64, self.center[1])).sum();
        x * y
    }
    /// Full support normalized BEFORE clipping; edge loss is not artificially recovered.
    pub fn energy(&self, w: u32, h: u32) -> f64 {
        let (x0, x1, y0, y1) = self.bounds();
        let mut sum = 0.0;
        for y in y0.max(0)..=y1.min(h as i32 - 1) {
            for x in x0.max(0)..=x1.min(w as i32 - 1) {
                sum += self.weight(x, y);
            }
        }
        sum / self.norm() * self.flux
    }
}
#[derive(Clone, Copy, Debug)]
struct Ranked(Source);
impl PartialEq for Ranked {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == std::cmp::Ordering::Equal
    }
}
impl Eq for Ranked {}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Ranked {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        o.0.flux.total_cmp(&self.0.flux).then(self.0.id.cmp(&o.0.id))
    }
}
pub struct Selection {
    heap: std::collections::BinaryHeap<Ranked>,
    pub sources: Vec<Source>,
    pub candidates: usize,
    pub rejected: usize,
}
impl Default for Selection {
    fn default() -> Self {
        Self::new()
    }
}
impl Selection {
    pub fn new() -> Self {
        Self {
            heap: std::collections::BinaryHeap::with_capacity(MAX_STARS),
            sources: Vec::with_capacity(MAX_STARS),
            candidates: 0,
            rejected: 0,
        }
    }
    pub fn update(&mut self, stars: &[Star], cam: &Camera, limit: usize) {
        self.sources.clear();
        self.heap.clear();
        let limit = limit.min(MAX_STARS);
        self.rejected = 0;
        self.candidates = 0;
        if !cam.origin.is_finite()
            || !cam.pos.is_finite()
            || !cam.forward.is_finite()
            || !cam.right.is_finite()
            || !cam.up.is_finite()
            || !cam.fov.is_finite()
            || !(0.01..3.0).contains(&cam.fov)
            || cam.width == 0
            || cam.width > 4096
            || cam.height == 0
            || cam.height > 4096
        {
            return;
        }

        let focal = cam.height as f64 / (2.0 * (cam.fov * 0.5).tan());
        for s in stars {
            let r = cam.relative(s);
            let d2 = r.length_squared();
            let z = r.dot(cam.forward);
            if !r.is_finite()
                || !d2.is_finite()
                || d2 < 1.0
                || z <= 0.0
                || !s.lum.is_finite()
                || s.lum <= 0.0
                || s.rgb.iter().any(|c| !c.is_finite() || *c < 0.0)
            {
                self.rejected += 1;
                continue;
            }
            let center = [
                cam.width as f64 * 0.5 + r.dot(cam.right) / z * focal,
                cam.height as f64 * 0.5 - r.dot(cam.up) / z * focal,
            ];
            if center.iter().any(|x| !x.is_finite()) {
                self.rejected += 1;
                continue;
            }
            if center[0] < -SUPPORT
                || center[1] < -SUPPORT
                || center[0] > cam.width as f64 + SUPPORT
                || center[1] > cam.height as f64 + SUPPORT
            {
                self.rejected += 1;
                continue;
            }
            let flux = s.lum * (AU * AU / d2);
            if flux.is_finite() && flux > 0.0 {
                self.candidates += 1;
                let next = Ranked(Source { id: s.id, center, flux, rgb: s.rgb });
                if self.heap.len() < limit {
                    self.heap.push(next);
                } else if limit > 0 && next < *self.heap.peek().unwrap() {
                    *self.heap.peek_mut().unwrap() = next;
                }
            }
        }
        self.sources.extend(self.heap.drain().map(|x| x.0));
        self.sources.sort_unstable_by(|a, b| b.flux.total_cmp(&a.flux).then(a.id.cmp(&b.id)));
        self.sources.truncate(limit.min(MAX_STARS));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    #[test]
    fn seed_and_bounds() {
        let a = Population::seeded(42, 100000);
        let b = Population::seeded(42, 100000);
        assert_eq!(a.stars, b.stars);
        assert_eq!(a.stars.len(), MAX_STARS);
        assert_eq!(a.stars.iter().filter(|s| s.id == 0).count(), 1);
        assert!(a.stars.iter().all(|s| s.pos.is_finite() && s.lum > 0.0));
    }
    fn camera() -> Camera {
        Camera::look_at(DVec3::ZERO, DVec3::Z, DVec3::Y, 1.0, 128, 128).unwrap()
    }
    #[test]
    fn flux_cap_and_invalid() {
        let c = camera();
        let stars: Vec<_> = (0..10).map(|id| Star { id, pos: DVec3::Z * AU, lum: 1.0, rgb: [1.0; 3] }).collect();
        let mut sel = Selection::new();
        sel.update(&stars, &c, 3);
        assert_eq!(sel.sources.len(), 3);
        assert_eq!(sel.sources.iter().map(|s| s.id).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert!((sel.sources[0].flux - 1.0).abs() < 1e-12);
        let mut s = stars[0];
        s.pos *= 2.0;
        sel.update(&[s], &c, 3);
        assert!((sel.sources[0].flux - 0.25).abs() < 1e-12);
        s.pos = DVec3::ZERO;
        sel.update(&[s], &c, 3);
        assert!(sel.sources.is_empty());
        s.pos = -DVec3::Z;
        sel.update(&[s], &c, 3);
        assert!(sel.sources.is_empty());
        sel.update(&stars, &c, 0);
        assert!(sel.sources.is_empty());
    }
    #[test]
    fn discrete_psf_and_clipping() {
        for center in [[64.0, 64.0], [64.25, 64.75], [64.499, 64.001]] {
            let p = Source { id: 0, center, flux: 1.0, rgb: [1.0; 3] };
            assert!((p.energy(128, 128) - 1.0).abs() < 1e-12);
        }
        let p = Source { id: 0, center: [0.0, 64.0], flux: 1.0, rgb: [1.0; 3] };
        assert!((p.energy(128, 128) - 0.5).abs() < 1e-12);
    }
    #[test]
    fn translation_parallax_rotation_flux() {
        let stars = [Star { id: 0, pos: DVec3::new(0.0, 0.0, AU), lum: 1.0, rgb: [1.0; 3] }];
        let mut a = Selection::new();
        a.update(&stars, &camera(), MAX_STARS);
        let x = a.sources[0].center[0];
        let c = Camera::look_at(DVec3::X * AU * 0.1, DVec3::X * AU * 0.1 + DVec3::Z, DVec3::Y, 1.0, 128, 128).unwrap();
        a.update(&stars, &c, MAX_STARS);
        assert!(a.sources[0].center[0] < x - 1.0);
        let c = Camera::look_at(DVec3::ZERO, DVec3::new(0.1, 0.0, 1.0), DVec3::Y, 1.0, 128, 128).unwrap();
        a.update(&stars, &c, MAX_STARS);
        assert!((a.sources[0].flux - 1.0).abs() < 1e-12);
    }
    #[test]
    fn considers_records_beyond_gpu_cap() {
        let mut stars = vec![Star { id: 1, pos: DVec3::Z * AU, lum: 1.0, rgb: [1.0; 3] }; MAX_STARS];
        stars.push(Star { id: 9007199254740993, pos: DVec3::Z * AU, lum: 10.0, rgb: [1.0; 3] });
        let mut s = Selection::new();
        s.update(&stars, &camera(), 1);
        assert_eq!(s.sources[0].id, 9007199254740993);
        assert_eq!(s.candidates, MAX_STARS + 1);
    }
    #[test]
    fn sol_frame_boundary() {
        use coords::frames::{FrameTree, WorldPos};
        let (mut tree, g) = FrameTree::new("galaxy");
        let s = tree.add(g, "sol", SOL, 1e13).unwrap();
        let star = WorldPos { frame: s, p: DVec3::ZERO };
        for x in [1e13 - 1e7, 1e13 + 1e7] {
            let cam = WorldPos { frame: s, p: DVec3::new(x, 1e10, 0.0) };
            assert_eq!(tree.rehome(cam).frame, if x < 1e13 { s } else { g });
            let a = tree.delta(star, cam);
            let b = tree.delta(star, tree.rehome(cam));
            assert!((a.normalize() - b.normalize()).length() < 1e-8);
            assert!((a.length_squared() / b.length_squared() - 1.0).abs() < 1e-8);
            let mut ca = Camera::look_at(cam.p, cam.p - DVec3::X, DVec3::Z, 1.0, 1280, 720).unwrap();
            ca.origin = SOL;
            let mut cb = ca;
            cb.pos = tree.convert(tree.rehome(cam), s).p;
            let star = Star { id: 0, pos: SOL, lum: 1.0, rgb: [1.0; 3] };
            let mut sa = Selection::new();
            let mut sb = Selection::new();
            sa.update(&[star], &ca, 1);
            sb.update(&[star], &cb, 1);
            for i in 0..2 {
                assert!((sa.sources[0].center[i] - sb.sources[0].center[i]).abs() < 1e-4);
            }
            assert!((sa.sources[0].flux / sb.sources[0].flux - 1.0).abs() < 1e-8);
        }
    }
}
