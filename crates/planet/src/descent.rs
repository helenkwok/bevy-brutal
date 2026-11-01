//! Orbit -> mountains descent (Earth round 3): nested f64 frames, the camera path and the
//! planet -> mountains renderer handoff schedule.
//!
//! **Frames.** `galaxy > system ("sol") > planet ("earth", ECEF, Z = north pole, X = 0 E) >
//! local ("alps")`, all [`coords::FrameTree`] f64 frames. The local frame's origin is the sea-level
//! point under the mountains scene's default vista; its axes are the planet frame's (the tree is
//! translation-only), and [`Frames`] adds the ENU rotation. The mountains scene ("world") is
//! y-up, x = east, z = south (yaw 0 looks north, yaw pi/2 east, exactly the quality example's
//! `CAM_YAW`), with the vista at world `(cx, *, cz)`. A world point maps to
//! `local = E (x - cx) + U y - N (z - cz)` (the flat tangent plane); the mountains renderer's
//! layer mode bends that plane onto the sphere for rasterisation (`y -= d^2 / 2R`, see
//! [`bend_error_m`]).
//!
//! **Path.** The camera height above the vista's ground falls log-linearly in the path
//! parameter `s` (constant "zoom rate", 400 km -> 40 m), on a great circle behind the vista
//! (opposite its view heading) at `glide x height` ground distance. It looks at a point in
//! the peaks ahead of the vista, and over the last ~3 km of height turns smoothly into the
//! vista's exact view. Every quantity is a smooth function of `s` (C-infinity away from the
//! smoothstep ends, C1 everywhere).
//!
//! **Handoff.** [`handoff`] gives, per height, the weight of the mountains layer over the planet
//! render for terrain ([`Handoff::ground`], 0 above 140 km, 1 below 30 km: at 60 km the planet's
//! 2.4 km cube maps already show the Valais glaciers as flat cloud-like smears, so the layer
//! carries most of the ground there) and for sky
//! ([`Handoff::sky`]: the planet's sky down to 3 km, the mountains' own sky below 0.3 km), the
//! camera up-vector blend, the aerial-perspective range, shadow splits and the exposure ramp.
use coords::{FrameId, FrameTree, WorldPos};
use glam::DVec3;

use crate::EARTH_RADIUS_M;
use crate::cube::latlon_dir;

/// Astronomical unit (m): the planet frame's offset in the system frame.
pub const AU_M: f64 = 1.495_978_707e11;

/// Where the mountains scene sits on the planet.
#[derive(Clone, Copy, Debug)]
pub struct Anchor {
    /// Latitude / longitude (deg) of the vista's ground point.
    pub lat: f64,
    pub lon: f64,
    /// World x, z of the vista (the quality example's default camera).
    pub world_xz: (f64, f64),
}

/// The nested frame tree plus the local frame's ENU basis (planet-frame unit vectors).
#[derive(Clone, Debug)]
pub struct Frames {
    pub tree: FrameTree,
    pub system: FrameId,
    pub planet: FrameId,
    pub local: FrameId,
    pub east: DVec3,
    pub north: DVec3,
    pub up: DVec3,
    pub world_xz: (f64, f64),
}

impl Frames {
    pub fn new(a: &Anchor) -> Self {
        let (mut tree, galaxy) = FrameTree::new("milky way");
        // the Sun ~26 kly from the galactic centre, the Earth 1 AU from the Sun: the descent's
        // differences are all formed in the planet frame, so these offsets cost no precision
        let system = tree.add(galaxy, "sol", DVec3::new(2.46e20, 0.0, 6.0e17), 1.0e13).expect("system frame");
        let planet = tree.add(system, "earth", DVec3::new(AU_M, 0.0, 0.0), 1.0e7).expect("planet frame");
        let up = latlon_dir(a.lat, a.lon);
        let east = DVec3::Z.cross(up).normalize();
        let north = up.cross(east);
        let local = tree.add(planet, "alps", up * EARTH_RADIUS_M, 1.0e4).expect("local frame");
        Self { tree, system, planet, local, east, north, up, world_xz: a.world_xz }
    }

    /// Mountains world position (flat tangent plane) -> a local-frame position.
    pub fn world_to_local(&self, w: DVec3) -> WorldPos {
        let (cx, cz) = self.world_xz;
        WorldPos { frame: self.local, p: self.east * (w.x - cx) + self.up * w.y - self.north * (w.z - cz) }
    }

    /// Any position -> mountains world coordinates (via the local frame, f64 throughout).
    pub fn to_world(&self, x: WorldPos) -> DVec3 {
        let v = self.tree.convert(x, self.local).p;
        let (cx, cz) = self.world_xz;
        DVec3::new(v.dot(self.east) + cx, v.dot(self.up), -v.dot(self.north) + cz)
    }

    /// Any position -> the planet frame (ECEF metres).
    pub fn to_planet(&self, x: WorldPos) -> DVec3 {
        self.tree.convert(x, self.planet).p
    }

    /// Direction: world -> planet frame, and back.
    pub fn world_dir_to_planet(&self, d: DVec3) -> DVec3 {
        self.east * d.x + self.up * d.y - self.north * d.z
    }
    pub fn planet_dir_to_world(&self, d: DVec3) -> DVec3 {
        DVec3::new(d.dot(self.east), d.dot(self.up), -d.dot(self.north))
    }
}

/// Position error (m) of the layer mode's parabolic bend (`y -= d^2 / 2R` on the flat world) against
/// the exact sphere, for a terrain point `d` metres from the curvature centre at height `y`:
/// (horizontal, vertical). The exact point is at geodesic distance `d` (the world's horizontal
/// distances are taken as geodesic) and radius `R + y`, expressed in the tangent frame.
pub fn bend_error_m(d: f64, y: f64) -> (f64, f64) {
    let r = EARTH_RADIUS_M;
    let th = d / r;
    let (ex, ey) = ((r + y) * th.sin(), (r + y) * th.cos() - r);
    ((ex - d).abs(), (ey - (y - d * d / (2.0 * r))).abs())
}

/// World yaw / pitch (the quality example's convention) of a world-frame direction.
pub fn yaw_pitch(fw: DVec3) -> (f64, f64) {
    let f = fw.normalize();
    (f.x.atan2(-f.z), f.y.clamp(-1.0, 1.0).asin())
}

/// World-frame direction of a yaw / pitch.
pub fn dir_of(yaw: f64, pitch: f64) -> DVec3 {
    DVec3::new(yaw.sin() * pitch.cos(), pitch.sin(), -yaw.cos() * pitch.cos())
}

/// The descent: vista pose, start height and the path shape.
#[derive(Clone, Debug)]
pub struct Descent {
    pub frames: Frames,
    /// Ground height (m) under the vista and the vista's height above it.
    pub ground_m: f64,
    pub h_end_m: f64,
    pub h_start_m: f64,
    pub yaw: f64,
    pub pitch: f64,
    /// Ground distance behind the vista per metre of height.
    pub glide: f64,
    /// World look-at target (in the peaks ahead of the vista).
    pub target: DVec3,
}

/// One camera pose on the path.
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub s: f64,
    /// Height above the vista's ground (m).
    pub h_m: f64,
    /// Camera position, planet frame.
    pub cam: WorldPos,
    /// Planet-frame forward and the up hint both renderers use for the roll.
    pub fwd: DVec3,
    pub up_hint: DVec3,
    /// The same camera in mountains world coordinates (+ yaw / pitch).
    pub world: DVec3,
    pub yaw: f64,
    pub pitch: f64,
    /// Ground distance from the vista (m).
    pub ground_dist_m: f64,
}

fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Spherical interpolation of unit vectors (exactly `b` at t >= 1, `a` at t <= 0).
pub fn slerp(a: DVec3, b: DVec3, t: f64) -> DVec3 {
    if t <= 0.0 {
        return a;
    }
    if t >= 1.0 {
        return b;
    }
    let c = a.dot(b).clamp(-1.0, 1.0);
    let th = c.acos();
    if th < 1e-9 {
        return (a + (b - a) * t).normalize();
    }
    ((a * ((1.0 - t) * th).sin() + b * (t * th).sin()) / th.sin()).normalize()
}

impl Descent {
    /// The round-3 descent onto the quality example's default vista: world (7700, ground + 40,
    /// -4200), yaw 2.75, pitch 0.08, anchored in the Valais Alps (46.10 N, 7.60 E). `ground_m`
    /// is the mountains field's height under the vista.
    pub fn alps(ground_m: f64) -> Self {
        let a = Anchor { lat: 46.10, lon: 7.60, world_xz: (7700.0, -4200.0) };
        let (yaw, pitch) = (2.75, 0.08);
        let f = dir_of(yaw, 0.0);
        let target = DVec3::new(a.world_xz.0 + f.x * 6000.0, 1500.0, a.world_xz.1 + f.z * 6000.0);
        Self { frames: Frames::new(&a), ground_m, h_end_m: 40.0, h_start_m: 400_000.0, yaw, pitch, glide: 1.25, target }
    }

    /// Height (m above the vista ground) at path parameter s in [0, 1]: log-linear.
    pub fn height_at(&self, s: f64) -> f64 {
        let s = s.clamp(0.0, 1.0);
        (self.h_start_m.ln() + (self.h_end_m.ln() - self.h_start_m.ln()) * s).exp()
    }

    /// Inverse of [`Self::height_at`].
    pub fn s_at_height(&self, h_m: f64) -> f64 {
        ((h_m.ln() - self.h_start_m.ln()) / (self.h_end_m.ln() - self.h_start_m.ln())).clamp(0.0, 1.0)
    }

    /// Weight of the vista's own view direction (0 = look at the target), by height.
    pub fn view_weight(&self, h_m: f64) -> f64 {
        1.0 - smoothstep(self.h_end_m.ln(), 3000f64.ln(), h_m.ln())
    }

    pub fn pose(&self, s: f64) -> Pose {
        let fr = &self.frames;
        let s = s.clamp(0.0, 1.0);
        let h = if s >= 1.0 { self.h_end_m } else { self.height_at(s) };
        let g = self.glide * (h - self.h_end_m);
        // great circle from the vista's nadir, opposite the view heading
        let back = self.yaw + std::f64::consts::PI;
        let b = fr.east * back.sin() + fr.north * back.cos();
        let th = g / EARTH_RADIUS_M;
        let dir = fr.up * th.cos() + b * th.sin();
        let cam_p = dir * (EARTH_RADIUS_M + self.ground_m + h);
        let cam = WorldPos { frame: fr.planet, p: cam_p };
        let tgt = fr.to_planet(fr.world_to_local(self.target));
        let look = (tgt - cam_p).normalize();
        let end = fr.world_dir_to_planet(dir_of(self.yaw, self.pitch));
        let fwd = slerp(look, end, self.view_weight(h));
        let wu = handoff(h).up;
        let up_hint = if wu >= 1.0 { fr.up } else { (dir * (1.0 - wu) + fr.up * wu).normalize() };
        let world = if s >= 1.0 {
            DVec3::new(fr.world_xz.0, self.ground_m + self.h_end_m, fr.world_xz.1)
        } else {
            fr.to_world(cam)
        };
        let (yaw, pitch) = if self.view_weight(h) >= 1.0 { (self.yaw, self.pitch) } else { yaw_pitch(fr.planet_dir_to_world(fwd)) };
        Pose { s, h_m: h, cam, fwd, up_hint, world, yaw, pitch, ground_dist_m: g }
    }

    /// Subsolar point (lat, lon, deg) of a sun at world azimuth `az` (yaw convention) and
    /// elevation `el` (rad): the planet renderer's sun for the mountains scene's sun.
    pub fn subsolar(&self, az: f64, el: f64) -> (f64, f64) {
        crate::cube::dir_latlon(self.frames.world_dir_to_planet(dir_of(az, el)))
    }
}

/// Per-height renderer handoff.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Handoff {
    /// Mountains-layer weight on its terrain (x its coverage alpha) and on the sky.
    pub ground: f64,
    pub sky: f64,
    /// Up-hint blend: 0 = the camera's own vertical, 1 = the local frame's (the mountains
    /// renderer's fixed +y). 1 wherever `ground > 0`, so both layers share the roll.
    pub up: f64,
    /// Mountains aerial-perspective range (km): 40 (the default view) near the ground.
    pub ap_km: f64,
    /// Planet-render exposure (the mountains layer gets the same, in its units: see the descent
    /// example's `mountains_exposure`).
    pub exposure: f64,
    /// Mountains sun-shadow cascade splits (m): the default splits near the ground.
    pub splits: [f64; 4],
}

/// Default mountains cascade splits (quality example `SPLITS`).
pub const SPLITS_M: [f64; 4] = [120.0, 600.0, 2500.0, 9000.0];
/// Planet exposure at orbit and the mountains-matched exposure (sun illuminance 5.5 x
/// exposure 1.1 in the quality example's units = 6.05 in the planet's solar units).
pub const EXPOSURE_ORBIT: f64 = 2.6;
pub const EXPOSURE_GROUND: f64 = 6.05;

pub fn handoff(h_m: f64) -> Handoff {
    let lh = h_m.max(1.0).ln();
    let ground = 1.0 - smoothstep(30_000f64.ln(), 140_000f64.ln(), lh);
    let sky = 1.0 - smoothstep(300f64.ln(), 3000f64.ln(), lh);
    let up = 1.0 - smoothstep(140_000f64.ln(), 400_000f64.ln(), lh);
    let ap_km = 40.0 + 80.0 * smoothstep(200f64.ln(), 3000f64.ln(), lh);
    let e = smoothstep(20_000f64.ln(), 150_000f64.ln(), lh);
    let exposure = (EXPOSURE_GROUND.ln() + (EXPOSURE_ORBIT.ln() - EXPOSURE_GROUND.ln()) * e).exp();
    let k = [1.5, 3.0, 6.0, 12.0];
    let splits = [0, 1, 2, 3].map(|i| (k[i] * h_m).clamp(SPLITS_M[i], 250_000.0));
    Handoff { ground, sky, up, ap_km, exposure, splits }
}

/// Vista fade of the mountains layer's aerial-only deviations (the sphere bend, voxel shells
/// centred on the ground, all-round far rings to 100+ km) by height above the vista ground: 1 at
/// 400 m and above, 0 at the vista's 40 m (log-smoothstep), so the path ends continuously in exactly
/// the default mountains view.
pub fn vista_fade(h_m: f64) -> f64 {
    smoothstep(40f64.ln(), 400f64.ln(), h_m.max(1.0).ln())
}

/// Strength of the descent's local colour match (Earth round 4: the mountains layer takes the
/// planet render's local mean colour, its physical aerial perspective, over the shared terrain):
/// 1 from 30 km up (where the layer's edge and its far field sit tens of km out, the disc seam of
/// round 3), 0 at 5 km and below (near terrain keeps the mountains scene's own shading and haze),
/// so the low frames and the vista are the mountains renders.
pub fn colour_match(h_m: f64) -> f64 {
    smoothstep(5000f64.ln(), 30_000f64.ln(), h_m.max(1.0).ln())
}

/// sRGB (0..1) <-> linear.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// Mountains-layer weight of one pixel: its coverage alpha `a` (terrain x radial fade, 0 on its
/// sky) times the ground weight, plus, on the rest, the sky weight: first only where the planet
/// render sees sky (`planet_sky` = sky fraction of the pixel), and as the sky weight reaches 1
/// everywhere, so the last frame is exactly the mountains render (the planet's distant ground
/// beyond the layer cannot leak into the existing scene).
pub fn layer_weight(a: f32, planet_sky: f32, w: &Handoff) -> f32 {
    let sky = w.sky as f32;
    (a * w.ground as f32 + (1.0 - a) * sky * (planet_sky + (1.0 - planet_sky) * sky)).clamp(0.0, 1.0)
}

/// Composite one display pixel (sRGB 0..1): planet render `p`, mountains layer `m`, blended in
/// linear light by [`layer_weight`]. Exactly `m` at weight 1 and `p` at 0.
pub fn composite(p: [f32; 3], m: [f32; 3], a: f32, planet_sky: f32, w: &Handoff) -> [f32; 3] {
    let k = layer_weight(a, planet_sky, w);
    if k >= 1.0 {
        return m;
    }
    if k <= 0.0 {
        return p;
    }
    [0, 1, 2].map(|i| linear_to_srgb(srgb_to_linear(p[i]) * (1.0 - k) + srgb_to_linear(m[i]) * k))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layer's aerial deviations vanish exactly at the vista and are full from 400 m up
    /// (before: always full, so the last frame differed from the default view, PSNR 26 dB).
    #[test]
    fn vista_fade_ends_in_the_default_view() {
        assert_eq!(vista_fade(40.0), 0.0);
        assert_eq!(vista_fade(10.0), 0.0);
        assert_eq!(vista_fade(400.0), 1.0);
        assert_eq!(vista_fade(400_000.0), 1.0);
        let mut last = 0.0;
        let mut h = 40.0;
        while h < 500.0 {
            let f = vista_fade(h);
            assert!(f >= last && f - last < 0.02, "monotone, continuous at {h} m");
            last = f;
            h *= 1.01;
        }
        // every descent frame above the vista keeps the full aerial layer
        for &h in &[400.0, 1200.0, 3000.0, 400_000.0] {
            assert_eq!(vista_fade(h), 1.0);
        }
    }

    #[test]
    fn colour_match_vanishes_at_the_vista() {
        assert_eq!(colour_match(40.0), 0.0);
        assert_eq!(colour_match(5000.0), 0.0);
        assert_eq!(colour_match(30_000.0), 1.0);
        assert_eq!(colour_match(400_000.0), 1.0);
        let (mut last, mut h) = (0.0, 40.0);
        while h < 500_000.0 {
            let k = colour_match(h);
            assert!(k >= last && k - last < 0.02, "monotone, continuous at {h} m");
            last = k;
            h *= 1.01;
        }
        // the whole match happens where the mountains layer's ground weight is full or rising
        assert!(handoff(5000.0).ground == 1.0);
    }

    fn d() -> Descent {
        Descent::alps(630.0)
    }

    #[test]
    fn nested_frames_round_trip_and_render_precision() {
        let ds = d();
        let fr = &ds.frames;
        // world -> local -> planet -> back: sub-micrometre at the vista and 100 km out; through
        // the system frame (1.5e11 m magnitudes) the error is that level's documented resolution
        let sys_eps = coords::FrameLevel::System.precision_m();
        for w in [DVec3::new(7700.0, 670.0, -4200.0), DVec3::new(7712.25, 1888.5, -4180.75), DVec3::new(-90_000.0, 3200.0, 60_000.0)] {
            let l = fr.world_to_local(w);
            let pl = fr.tree.convert(l, fr.planet);
            assert!((fr.to_world(pl) - w).length() < 1e-6, "{w:?} via planet");
            let sys = fr.tree.convert(l, fr.system);
            let e = (fr.to_world(sys) - w).length();
            assert!(e < sys_eps && e < 1e-4, "{w:?} via system: {e} m");
        }
        // camera at the vista, a vertex 3 m away: the f32 render-relative vector (formed in f64
        // at the frames' common ancestor) is exact to well under a millimetre, although the
        // system frame alone would put the camera 1.5e11 m from its origin
        let p = ds.pose(1.0);
        let v = fr.world_to_local(p.world + DVec3::new(2.0, -1.5, 1.5));
        let rr = fr.tree.render_relative(v, p.cam);
        let want = fr.world_dir_to_planet(DVec3::new(2.0, -1.5, 1.5));
        assert!((rr.as_dvec3() - want).length() < 1e-6, "{rr:?} vs {want:?}");
        // the same difference formed in the galaxy frame (1e20 m magnitudes) would be garbage:
        // f64 there has ~16 km ULPs. The tree never does that.
        let g = fr.tree.root();
        let (a, b) = (fr.tree.convert(v, g).p, fr.tree.convert(p.cam, g).p);
        assert!(((a - b) - want).length() > 1.0);
    }

    #[test]
    fn path_endpoints_are_exact() {
        let ds = d();
        let p0 = ds.pose(0.0);
        let alt0 = fr_alt(&ds, &p0);
        assert!((alt0 - (400_000.0 + 630.0)).abs() < 1e-6);
        let p1 = ds.pose(1.0);
        assert_eq!(p1.world, DVec3::new(7700.0, 670.0, -4200.0));
        assert_eq!((p1.yaw, p1.pitch), (2.75, 0.08));
        assert_eq!(p1.up_hint, ds.frames.up);
        // and the f64 path lands there continuously: s = 1 - 1e-9 is within a millimetre
        let pe = ds.pose(1.0 - 1e-9);
        assert!((pe.world - p1.world).length() < 1e-3, "{:?}", pe.world - p1.world);
        assert!((pe.yaw - 2.75).abs() < 1e-9 && (pe.pitch - 0.08).abs() < 1e-9);
    }

    fn fr_alt(ds: &Descent, p: &Pose) -> f64 {
        ds.frames.to_planet(p.cam).length() - EARTH_RADIUS_M
    }

    #[test]
    fn path_is_continuous_and_descends() {
        let ds = d();
        let n = 4000;
        let mut prev = ds.pose(0.0);
        let mut prev_step = 0.0f64;
        for i in 1..=n {
            let p = ds.pose(i as f64 / n as f64);
            assert!(p.h_m < prev.h_m, "height must fall monotonically");
            let step = (ds.frames.to_planet(p.cam) - ds.frames.to_planet(prev.cam)).length();
            // log-linear height: the step shrinks with the height; it never jumps (ratio to the
            // previous step stays within a few percent: no kinks)
            if i > 1 {
                let r = step / prev_step;
                assert!((0.95..1.0).contains(&r), "step ratio {r} at s {}", p.s);
            }
            // orientation turns smoothly: < 0.25 deg per 1/4000 of the path, no flips
            let ang = p.fwd.dot(prev.fwd).clamp(-1.0, 1.0).acos().to_degrees();
            assert!(ang < 0.25, "turn {ang} deg at s {}", p.s);
            assert!(p.up_hint.dot(prev.up_hint) > 0.99999);
            // world and planet descriptions agree (the mountains camera is the planet camera)
            let w = ds.frames.to_world(p.cam);
            assert!((w - p.world).length() < 1e-6 || i == n);
            prev_step = step;
            prev = p;
        }
    }

    #[test]
    fn handoff_schedule() {
        let mut last = handoff(500_000.0);
        assert_eq!((last.ground, last.sky, last.up), (0.0, 0.0, 0.0));
        assert!((last.exposure - EXPOSURE_ORBIT).abs() < 1e-12);
        let mut h = 500_000.0;
        while h > 30.0 {
            let w = handoff(h);
            for (a, b) in [(w.ground, last.ground), (w.sky, last.sky), (w.up, last.up)] {
                assert!((0.0..=1.0).contains(&a) && a >= b, "weights grow monotonically as the camera descends");
                assert!(a - b < 0.02, "and continuously ({b} -> {a} at {h} m)");
            }
            assert!(w.ground == 0.0 || w.up == 1.0, "the mountains layer only shows once both layers share the roll");
            assert!(w.sky == 0.0 || w.ground == 1.0, "its sky only shows once its terrain is fully in");
            assert!(w.splits.iter().zip(&SPLITS_M).all(|(s, m)| s >= m));
            last = w;
            h *= 0.99;
        }
        let g = handoff(40.0);
        assert_eq!((g.ground, g.sky, g.up, g.ap_km), (1.0, 1.0, 1.0, 40.0));
        assert_eq!(g.splits, SPLITS_M);
        assert!((g.exposure - EXPOSURE_GROUND).abs() < 1e-12);
    }

    #[test]
    fn layer_bend_matches_the_sphere() {
        // parabolic bend vs the exact sphere over the layer's extent (rings to ~200 km, terrain
        // to 4.5 km): vertical < 2 m everywhere; horizontal ~ d y / R (< 50 m at 104 km / 3 km
        // height = 0.03 deg seen from 100 km), far below the layer's 60..100 km fade band
        for d in [0.0, 10e3, 26e3, 60e3, 104e3, 200e3] {
            for y in [0.0, 1500.0, 4500.0] {
                let (eh, ev) = bend_error_m(d, y);
                assert!(ev < 2.0 + 0.0 * eh, "vertical {ev} m at {d} m / {y} m");
                assert!(eh <= d * y / EARTH_RADIUS_M + d.powi(3) / (6.0 * EARTH_RADIUS_M * EARTH_RADIUS_M) + 1e-6);
            }
        }
        assert!(bend_error_m(104e3, 3000.0).0 < 50.0);
    }

    #[test]
    fn composite_limits() {
        let p = [0.2, 0.4, 0.6];
        let m = [0.7, 0.5, 0.3];
        let none = handoff(500_000.0);
        let all = handoff(40.0);
        assert_eq!(composite(p, m, 1.0, 0.0, &none), p);
        assert_eq!(composite(p, m, 1.0, 0.0, &all), m);
        // sky over the planet's sky: the mountains' own sky only at the bottom of the descent
        assert_eq!(composite(p, m, 0.0, 1.0, &all), m);
        assert_eq!(composite(p, m, 0.0, 1.0, &handoff(10_000.0)), p);
        // the planet's distant ground where the layer has no terrain: kept while the planet owns
        // the sky, replaced at the end of the descent (the last frame is the mountains render)
        assert_eq!(composite(p, m, 0.0, 0.0, &handoff(10_000.0)), p);
        assert_eq!(composite(p, m, 0.0, 0.0, &all), m);
        let mid = handoff(950.0);
        assert!(mid.sky > 0.3 && mid.sky < 0.7);
        assert!(layer_weight(0.0, 0.0, &mid) < layer_weight(0.0, 1.0, &mid));
        // half blend is in linear light (brighter than the sRGB midpoint of 0.2 / 0.7)
        let w = Handoff { ground: 0.5, ..all };
        let c = composite(p, m, 1.0, 0.0, &w);
        assert!(c[0] > 0.45 && c[0] < 0.7);
    }

    #[test]
    fn sun_maps_to_a_subsolar_point() {
        let ds = d();
        // the scene's default sun: azimuth yaw + 2.35 (WNW), 0.5 rad up -> a subsolar point to the
        // west, with the planet's sun direction equal to the world one
        let (az, el) = (2.75 + 2.35, 0.5);
        let (la, lo) = ds.subsolar(az, el);
        let sp = latlon_dir(la, lo);
        let w = ds.frames.planet_dir_to_world(sp);
        assert!((w - dir_of(az, el)).length() < 1e-12);
        assert!(lo < 7.6, "sun in the west: afternoon");
    }
}
