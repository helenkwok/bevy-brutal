//! Nested f64 reference frames: **Galaxy > System > Planet > Local**.
//!
//! Every world position is a [`WorldPos`]: an f64 [`DVec3`] in metres plus the [`FrameId`] it is
//! expressed in. Each frame level bounds the coordinate magnitudes inside it, which bounds the
//! representation error ([`FrameLevel::precision_m`]):
//!
//! | level  | max extent | f64 precision (half ULP at max extent) |
//! |--------|-----------:|-----------------------------------------|
//! | Galaxy | 1e21 m (~100 kly) | ~65 km  |
//! | System | 1e13 m (~67 AU)   | ~1 mm   |
//! | Planet | 1e7 m             | ~1 nm   |
//! | Local  | 1e4 m             | ~1 pm   |
//!
//! Differences are formed at the **lowest common ancestor** of the two frames, so two points in
//! the same planet frame subtract with planet-level precision even if the planet sits 1e20 m from
//! the galaxy origin. The camera-relative result is computed in f64 and only then narrowed to f32
//! ([`FrameTree::render_relative`]). Frames are translation-only (no rotation yet): orientation
//! (planet spin / tilt) is a documented extension.
//!
//! The brick lattice ([`crate::BrickGrid`]) is the **planet-local voxel layer**: lattice positions
//! are metres in a Planet (or Local) frame.
use glam::{DVec3, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FrameLevel {
    Galaxy = 0,
    System = 1,
    Planet = 2,
    Local = 3,
}

/// Size of one ULP of an f64 at magnitude `x`.
pub fn ulp(x: f64) -> f64 {
    let b = x.abs().to_bits();
    f64::from_bits(b + 1) - f64::from_bits(b)
}

impl FrameLevel {
    /// Documented maximum |coordinate| inside a frame of this level, metres.
    pub const fn max_extent_m(self) -> f64 {
        match self {
            FrameLevel::Galaxy => 1e21,
            FrameLevel::System => 1e13,
            FrameLevel::Planet => 1e7,
            FrameLevel::Local => 1e4,
        }
    }

    /// Worst-case rounding error of one coordinate stored in this frame (half an ULP at the max extent).
    pub fn precision_m(self) -> f64 {
        0.5 * ulp(self.max_extent_m())
    }

    pub fn child(self) -> Option<FrameLevel> {
        match self {
            FrameLevel::Galaxy => Some(FrameLevel::System),
            FrameLevel::System => Some(FrameLevel::Planet),
            FrameLevel::Planet => Some(FrameLevel::Local),
            FrameLevel::Local => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FrameId(pub u32);

#[derive(Clone, Debug)]
pub struct Frame {
    pub name: String,
    pub level: FrameLevel,
    pub parent: Option<FrameId>,
    /// This frame's origin in its parent's coordinates.
    pub origin_in_parent: DVec3,
    /// Positions closer than this to the origin prefer this frame ([`FrameTree::rehome`]).
    pub influence_m: f64,
}

#[derive(Debug, PartialEq)]
pub enum FrameError {
    LocalHasNoChildren,
    OutOfExtent { level: FrameLevel, magnitude: f64 },
    Influence(f64),
}

/// A position: f64 metres in `frame`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldPos {
    pub frame: FrameId,
    pub p: DVec3,
}

#[derive(Clone, Debug)]
pub struct FrameTree {
    frames: Vec<Frame>,
}

impl FrameTree {
    /// A tree with a single galaxy root.
    pub fn new(galaxy_name: &str) -> (Self, FrameId) {
        let root = Frame { name: galaxy_name.into(), level: FrameLevel::Galaxy, parent: None, origin_in_parent: DVec3::ZERO, influence_m: f64::INFINITY };
        (Self { frames: vec![root] }, FrameId(0))
    }

    pub fn root(&self) -> FrameId {
        FrameId(0)
    }

    pub fn frame(&self, id: FrameId) -> &Frame {
        &self.frames[id.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    fn check(level: FrameLevel, v: DVec3) -> Result<(), FrameError> {
        let m = v.abs().max_element();
        if m.is_finite() && m <= level.max_extent_m() { Ok(()) } else { Err(FrameError::OutOfExtent { level, magnitude: m }) }
    }

    /// Add a child frame one level below `parent`, origin given in parent coordinates.
    pub fn add(&mut self, parent: FrameId, name: &str, origin_in_parent: DVec3, influence_m: f64) -> Result<FrameId, FrameError> {
        let pl = self.frame(parent).level;
        let level = pl.child().ok_or(FrameError::LocalHasNoChildren)?;
        Self::check(pl, origin_in_parent)?;
        if !(influence_m > 0.0 && influence_m <= level.max_extent_m()) {
            return Err(FrameError::Influence(influence_m));
        }
        self.frames.push(Frame { name: name.into(), level, parent: Some(parent), origin_in_parent, influence_m });
        Ok(FrameId(self.frames.len() as u32 - 1))
    }

    /// Move a frame (e.g. a planet along its orbit).
    pub fn set_origin(&mut self, id: FrameId, origin_in_parent: DVec3) -> Result<(), FrameError> {
        let pl = self.frame(self.frame(id).parent.expect("root cannot move")).level;
        Self::check(pl, origin_in_parent)?;
        self.frames[id.0 as usize].origin_in_parent = origin_in_parent;
        Ok(())
    }

    fn ancestors(&self, id: FrameId) -> Vec<FrameId> {
        let mut v = vec![id];
        let mut c = id;
        while let Some(p) = self.frame(c).parent {
            v.push(p);
            c = p;
        }
        v
    }

    /// Lowest common ancestor.
    pub fn lca(&self, a: FrameId, b: FrameId) -> FrameId {
        let aa = self.ancestors(a);
        let bb = self.ancestors(b);
        *aa.iter().find(|f| bb.contains(f)).expect("same tree")
    }

    /// Origin of `id` expressed in ancestor `anc` (summed smallest-first).
    fn origin_in(&self, id: FrameId, anc: FrameId) -> DVec3 {
        let mut acc = DVec3::ZERO;
        let mut c = id;
        while c != anc {
            let f = self.frame(c);
            acc += f.origin_in_parent;
            c = f.parent.expect("anc is an ancestor");
        }
        acc
    }

    /// Express `x` in frame `to`.
    pub fn convert(&self, x: WorldPos, to: FrameId) -> WorldPos {
        if x.frame == to {
            return x;
        }
        let l = self.lca(x.frame, to);
        let off = self.origin_in(x.frame, l) - self.origin_in(to, l);
        WorldPos { frame: to, p: off + x.p }
    }

    /// `a - b` in f64, formed at their common ancestor (frames are translation-only, so the
    /// vector is the same in every frame).
    pub fn delta(&self, a: WorldPos, b: WorldPos) -> DVec3 {
        if a.frame == b.frame {
            return a.p - b.p;
        }
        let l = self.lca(a.frame, b.frame);
        (self.origin_in(a.frame, l) - self.origin_in(b.frame, l)) + (a.p - b.p)
    }

    /// Camera-relative render position: subtract in f64, narrow once.
    pub fn render_relative(&self, x: WorldPos, camera: WorldPos) -> Vec3 {
        self.delta(x, camera).as_vec3()
    }

    /// Re-express `x` in the deepest frame whose influence contains it (keeps magnitudes inside the
    /// level's documented extent). Ascends out of frames it has left.
    pub fn rehome(&self, x: WorldPos) -> WorldPos {
        let mut cur = x;
        // ascend while outside the current frame's influence
        while let Some(parent) = self.frame(cur.frame).parent {
            if cur.p.length() <= self.frame(cur.frame).influence_m {
                break;
            }
            cur = self.convert(cur, parent);
        }
        // descend into the nearest child whose influence contains the point
        loop {
            let here = cur.frame;
            let best = (0..self.frames.len() as u32)
                .map(FrameId)
                .filter(|&c| self.frame(c).parent == Some(here))
                .map(|c| (c, cur.p - self.frame(c).origin_in_parent))
                .filter(|(c, d)| d.length() <= self.frame(*c).influence_m)
                .min_by(|a, b| a.1.length().total_cmp(&b.1.length()));
            match best {
                Some((c, d)) => cur = WorldPos { frame: c, p: d },
                None => return cur,
            }
        }
    }
}

/// Floating render origin with hysteresis: the origin jumps to the camera when the camera is more
/// than `threshold_m` away. f32 positions relative to the origin then stay within
/// `threshold_m` + (object distance) of zero.
#[derive(Clone, Copy, Debug)]
pub struct FloatingOrigin {
    pub origin: WorldPos,
    pub threshold_m: f64,
    pub rebases: u64,
}

impl FloatingOrigin {
    pub fn new(at: WorldPos, threshold_m: f64) -> Self {
        Self { origin: at, threshold_m, rebases: 0 }
    }

    /// Returns `Some(shift)` (new origin - old origin, metres) when it rebased. Callers holding f32
    /// origin-relative positions subtract `shift` from them.
    pub fn update(&mut self, tree: &FrameTree, camera: WorldPos) -> Option<DVec3> {
        let d = tree.delta(camera, self.origin);
        if d.length() <= self.threshold_m && camera.frame == self.origin.frame {
            return None;
        }
        self.origin = camera;
        self.rebases += 1;
        Some(d)
    }

    /// f32 position of `x` relative to the current origin.
    pub fn local_f32(&self, tree: &FrameTree, x: WorldPos) -> Vec3 {
        tree.render_relative(x, self.origin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A galaxy with two systems, each with planets, each with local frames, at realistic magnitudes.
    fn universe() -> (FrameTree, [FrameId; 6]) {
        let (mut t, g) = FrameTree::new("milky way");
        let s1 = t.add(g, "sol", DVec3::new(2.6e20, 1.0e17, -3.0e19), 1e13).unwrap();
        let s2 = t.add(g, "alpha", DVec3::new(2.6e20 + 4.1e16, 3.0e16, -3.0e19), 1e13).unwrap();
        let earth = t.add(s1, "earth", DVec3::new(1.496e11, 0.0, 2.0e9), 1e7).unwrap();
        let mars = t.add(s1, "mars", DVec3::new(-2.0e11, 1.0e9, 5.0e10), 1e7).unwrap();
        let site = t.add(earth, "site", DVec3::new(6.371e6, 1234.5, -77.25), 1e4).unwrap();
        let site2 = t.add(earth, "site2", DVec3::new(-3.0e6, 5.5e6, 1.0e6), 1e4).unwrap();
        (t, [s1, s2, earth, mars, site, site2])
    }

    #[test]
    fn documented_precision_per_level() {
        // nm in planet frames, mm in system frames (the design targets); explicit bounds.
        assert!(FrameLevel::Planet.precision_m() <= 1e-9, "{}", FrameLevel::Planet.precision_m());
        assert!(FrameLevel::System.precision_m() <= 1e-3, "{}", FrameLevel::System.precision_m());
        assert!(FrameLevel::Local.precision_m() <= 1e-12);
        assert!(FrameLevel::Galaxy.precision_m() <= 1e5);
        // Empirically: a 1 nm step near the planet extent survives, a 1 mm step near the system extent survives.
        let p = 9.99e6_f64;
        assert!(((p + 1e-8) - p - 1e-8).abs() <= FrameLevel::Planet.precision_m() * 2.0);
        let s = 9.99e12_f64;
        assert!(((s + 1e-2) - s - 1e-2).abs() <= FrameLevel::System.precision_m() * 2.0);
    }

    #[test]
    fn same_planet_differences_have_planet_precision_regardless_of_galaxy_offset() {
        let (t, [_, _, earth, _, site, site2]) = universe();
        let a = WorldPos { frame: site, p: DVec3::new(10.0, 2.0, 3.0) };
        let b = WorldPos { frame: earth, p: DVec3::new(6.371e6 + 10.0 + 1e-6, 1234.5 + 2.0, -77.25 + 3.0) };
        // exact answer: a - b = (-1e-6, 0, 0)
        let d = t.delta(a, b);
        assert!((d - DVec3::new(-1e-6, 0.0, 0.0)).abs().max_element() <= 4.0 * FrameLevel::Planet.precision_m(), "{d:?}");
        // between two local frames of the same planet: LCA = planet
        assert_eq!(t.lca(site, site2), earth);
    }

    #[test]
    fn frame_to_frame_round_trips() {
        let (t, ids) = universe();
        let [s1, s2, earth, mars, site, site2] = ids;
        let x = WorldPos { frame: site, p: DVec3::new(123.456, -9.75, 4000.125) };
        for &to in &[t.root(), s1, s2, earth, mars, site2] {
            let y = t.convert(x, to);
            let back = t.convert(y, site);
            // error bounded by the precision of the coarsest frame on the path (the LCA / target)
            let lvl = t.frame(t.lca(site, to)).level.min(t.frame(to).level);
            let tol = 8.0 * lvl.precision_m().max(FrameLevel::Local.precision_m());
            let err = (back.p - x.p).abs().max_element();
            assert!(err <= tol, "via {}: err {err} tol {tol}", t.frame(to).name);
            // delta agrees with convert
            let d1 = t.delta(x, WorldPos { frame: to, p: DVec3::ZERO });
            assert!((d1 - y.p).abs().max_element() <= tol, "delta vs convert via {}", t.frame(to).name);
        }
    }

    #[test]
    fn render_relative_is_precise_near_the_camera_at_any_level() {
        let (t, [s1, _, earth, _, site, _]) = universe();
        // camera in the system frame 1 km from a local-frame object near Earth
        let obj = WorldPos { frame: site, p: DVec3::new(1.0, 2.0, 3.0) };
        let obj_sys = t.convert(obj, s1);
        let cam = WorldPos { frame: s1, p: obj_sys.p + DVec3::new(1000.0, 0.0, 0.0) };
        let r = t.render_relative(obj, cam);
        // f32 at 1 km: ~6e-5 m ULP; f64 system-level error ~1e-5 m at 1.5e11 m
        assert!((r - Vec3::new(-1000.0, 0.0, 0.0)).abs().max_element() < 2e-4, "{r:?}");
        // camera in the planet frame: same answer
        let cam_p = t.convert(cam, earth);
        assert!((t.render_relative(obj, cam_p) - r).abs().max_element() < 2e-4);
    }

    #[test]
    fn rehome_picks_deepest_containing_frame() {
        let (t, [s1, _, earth, _, site, _]) = universe();
        let near_site = WorldPos { frame: earth, p: DVec3::new(6.371e6 + 5.0, 1234.5, -77.25) };
        assert_eq!(t.rehome(near_site).frame, site);
        let in_space = WorldPos { frame: earth, p: DVec3::new(5e7, 0.0, 0.0) }; // left Earth's 1e7 m influence
        let h = t.rehome(in_space);
        assert_eq!(h.frame, s1);
        let back = t.convert(h, earth);
        assert!((back.p - in_space.p).abs().max_element() <= 4.0 * FrameLevel::System.precision_m());
        assert!(h.p.abs().max_element() <= FrameLevel::System.max_extent_m());
    }

    #[test]
    fn extents_are_enforced() {
        let (mut t, g) = FrameTree::new("g");
        assert!(matches!(t.add(g, "far", DVec3::new(2e21, 0.0, 0.0), 1e13), Err(FrameError::OutOfExtent { .. })));
        let s = t.add(g, "s", DVec3::ZERO, 1e13).unwrap();
        let p = t.add(s, "p", DVec3::ZERO, 1e7).unwrap();
        let l = t.add(p, "l", DVec3::ZERO, 1e4).unwrap();
        assert_eq!(t.add(l, "x", DVec3::ZERO, 1.0), Err(FrameError::LocalHasNoChildren));
    }

    #[test]
    fn rebasing_keeps_f32_positions_accurate_over_planet_scale_travel() {
        let (mut t, g) = FrameTree::new("g");
        let s = t.add(g, "s", DVec3::new(1e20, 0.0, 0.0), 1e13).unwrap();
        let p = t.add(s, "p", DVec3::new(1e11, 0.0, 0.0), 1e7).unwrap();
        let _ = &mut t;
        let start = WorldPos { frame: p, p: DVec3::new(-9.0e6, 0.0, 0.0) };
        let mut fo = FloatingOrigin::new(start, 1000.0);
        // objects riding along 10 m and 100 m ahead of the camera, held as f32 origin-relative positions
        let offsets = [DVec3::new(10.3, 0.0, 0.0), DVec3::new(100.7, 1.1, 0.0)];
        let mut held = offsets.map(|o| o.as_vec3());
        let mut cam = start;
        let mut cam_f32 = Vec3::ZERO; // camera relative to origin, f32
        let (mut worst, mut naive_worst) = (0.0f32, 0.0f32);
        let step = DVec3::new(1_234.567_891, 0.001, 0.0);
        for _ in 0..14_000 {
            // ~17,000 km of travel
            cam.p += step;
            for h in &mut held {
                *h += step.as_vec3();
            }
            cam_f32 += step.as_vec3();
            if let Some(shift) = fo.update(&t, cam) {
                let s32 = shift.as_vec3();
                for h in &mut held {
                    *h -= s32;
                }
                cam_f32 -= s32;
                // re-derive from f64 truth at each rebase (what the plugin does): no drift accumulation
                for (h, o) in held.iter_mut().zip(offsets) {
                    *h = fo.local_f32(&t, WorldPos { frame: cam.frame, p: cam.p + o });
                }
                cam_f32 = fo.local_f32(&t, cam);
            }
            for (h, o) in held.iter().zip(offsets) {
                let exact = o.as_vec3();
                worst = worst.max(((*h - cam_f32) - exact).abs().max_element());
                // naive: f32 absolute planet coordinates
                let ca = cam.p.as_vec3();
                let oa = (cam.p + o).as_vec3();
                naive_worst = naive_worst.max(((oa - ca) - exact).abs().max_element());
            }
        }
        assert!(fo.rebases > 10_000, "rebased {} times", fo.rebases);
        assert!(worst < 1e-3, "rebased f32 error {worst} m");
        assert!(naive_worst > 0.1, "naive f32 error {naive_worst} m should be large (test is meaningful)");
    }
}
