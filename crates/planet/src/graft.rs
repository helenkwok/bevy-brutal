//! Earth round 4, "shared Alps": the planet renderer shows the mountains scene's own terrain.
//!
//! Round 3 cross-faded two unrelated terrains between 140 and 30 km (the real Valais from the
//! NASA cube maps under the procedural `Alpine` field of the mountains layer). Here the planet
//! layer *is* the mountains terrain inside a disc around the vista: patch heights come from the
//! same f64 field (at the patch's vertex spacing), and the surface shading reads a baked
//! albedo / normal / sun-visibility map of that field ([`GraftFile`], two nested levels). As the
//! mountains layer fades in over identical geometry, the handoff becomes a change of renderer
//! over the same mountains instead of a dissolve between two mountain ranges.
//!
//! **Mapping.** World `(x, z)` (mountains frame, x east, z south, vista at `world_xz`) <-> planet
//! direction by the azimuthal-equidistant projection about the vista's nadir: geodesic distance =
//! the world's horizontal distance from the vista, bearing preserved. That is the convention the
//! mountains layer's sphere bend uses (`descent::bend_error_m`), so both renderers put a world
//! point within ~d y / R + d^3 / 6R^2 of each other (< 50 m at 100 km, ~1 px from 60 km up).
//!
//! **Blend.** Spatially full inside `r_full_m`, smoothstep to the real Earth at `r_out_m`
//! (heights, albedo and normals lerp); in time by [`strength`]: 0 at 400 km (the orbit frame
//! shows the real Alps), 1 from 150 km down (log-smoothstep), so the real relief morphs into the
//! shared terrain before the mountains layer appears (140 km).
use glam::DVec3;

use crate::EARTH_RADIUS_M;
use crate::descent::Frames;

fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Graft strength by camera height above the vista ground (m): 0 at >= 400 km, 1 at <= 150 km.
pub fn strength(h_m: f64) -> f64 {
    1.0 - smoothstep(150_000f64.ln(), 400_000f64.ln(), h_m.max(1.0).ln())
}

/// Azimuthal-equidistant map about the vista (f64).
#[derive(Clone, Copy, Debug)]
pub struct GraftMap {
    pub centre: DVec3,
    pub east: DVec3,
    pub north: DVec3,
    pub world_xz: (f64, f64),
}

impl GraftMap {
    pub fn from_frames(f: &Frames) -> Self {
        Self { centre: f.up, east: f.east, north: f.north, world_xz: f.world_xz }
    }

    /// Planet direction -> world (x, z) and the geodesic distance from the vista (m).
    pub fn dir_to_world(&self, d: DVec3) -> (f64, f64, f64) {
        let d = d.normalize();
        let (e, n) = (d.dot(self.east), d.dot(self.north));
        let s = e.hypot(n);
        // atan2 of the sine / cosine: exact near the centre (acos loses ~1e-8 rad there)
        let ang = s.atan2(d.dot(self.centre));
        let dist = ang * EARTH_RADIUS_M;
        let (ue, un) = if s > 0.0 { (e / s, n / s) } else { (0.0, 0.0) };
        (self.world_xz.0 + dist * ue, self.world_xz.1 - dist * un, dist)
    }

    /// World (x, z) -> planet direction (inverse of [`Self::dir_to_world`]).
    pub fn world_to_dir(&self, x: f64, z: f64) -> DVec3 {
        let (de, dn) = (x - self.world_xz.0, -(z - self.world_xz.1));
        let dist = de.hypot(dn);
        if dist == 0.0 {
            return self.centre;
        }
        let ang = dist / EARTH_RADIUS_M;
        let t = (self.east * de + self.north * dn) / dist;
        (self.centre * ang.cos() + t * ang.sin()).normalize()
    }
}

/// Spatial graft weight at geodesic distance `d` (m): 1 inside `r0`, 0 beyond `r1`.
pub fn weight(d: f64, r0: f64, r1: f64) -> f64 {
    1.0 - smoothstep(r0, r1, d)
}

/// One baked level: `size`^2 texels covering world `[-half, half]^2` around the vista (texel
/// centres), u along +x (east), v along +z (south).
#[derive(Clone, Debug)]
pub struct GraftLevel {
    pub size: u32,
    pub half_m: f64,
    /// RGBA8, sRGB-encoded rgb = albedo, a = 255.
    pub albedo: Vec<u8>,
    /// RGBA8 UNORM: r, g = world-normal x, z * 0.5 + 0.5; b = sun visibility (cast shadow);
    /// a = 255.
    pub aux: Vec<u8>,
}

/// The bake as stored on disk (`GRF1`, little endian): anchor, blend radii, levels outer first.
#[derive(Clone, Debug)]
pub struct GraftFile {
    pub lat: f64,
    pub lon: f64,
    pub world_xz: (f64, f64),
    pub r_full_m: f64,
    pub r_out_m: f64,
    /// Sun direction (world frame) the visibility was baked for.
    pub sun_world: [f64; 3],
    pub levels: Vec<GraftLevel>,
}

const MAGIC: &[u8; 4] = b"GRF1";

impl GraftFile {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        for v in [self.lat, self.lon, self.world_xz.0, self.world_xz.1, self.r_full_m, self.r_out_m, self.sun_world[0], self.sun_world[1], self.sun_world[2]] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&(self.levels.len() as u32).to_le_bytes());
        for l in &self.levels {
            b.extend_from_slice(&l.size.to_le_bytes());
            b.extend_from_slice(&l.half_m.to_le_bytes());
        }
        for l in &self.levels {
            b.extend_from_slice(&l.albedo);
            b.extend_from_slice(&l.aux);
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
            let s = u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?);
            o += 4;
            dims.push((s, f64r(&mut o)?));
        }
        let mut levels = Vec::new();
        for (size, half_m) in dims {
            let len = (size as usize).pow(2) * 4;
            let albedo = b.get(o..o + len)?.to_vec();
            o += len;
            let aux = b.get(o..o + len)?.to_vec();
            o += len;
            levels.push(GraftLevel { size, half_m, albedo, aux });
        }
        Some(Self { lat: h[0], lon: h[1], world_xz: (h[2], h[3]), r_full_m: h[4], r_out_m: h[5], sun_world: [h[6], h[7], h[8]], levels })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descent::{Descent, bend_error_m};

    #[test]
    fn map_round_trips_and_matches_the_layer_bend() {
        let ds = Descent::alps(630.0);
        let m = GraftMap::from_frames(&ds.frames);
        for (x, z) in [(7700.0, -4200.0), (7712.5, -4190.25), (-90_000.0, 60_000.0), (150_000.0, -160_000.0)] {
            let d = m.world_to_dir(x, z);
            let (x2, z2, dist) = m.dir_to_world(d);
            assert!((x2 - x).abs() < 1e-6 && (z2 - z).abs() < 1e-6, "({x}, {z}) -> ({x2}, {z2})");
            assert!((dist - (x - 7700.0f64).hypot(z + 4200.0)).abs() < 1e-6);
        }
        // near the vista the projection is the local frame's tangent plane (to the curvature
        // drop): a point 10 km east lands on the frames' own world -> planet mapping
        let w = glam::DVec3::new(17_700.0, 0.0, -4200.0);
        let flat = ds.frames.to_planet(ds.frames.world_to_local(w)).normalize();
        let ae = m.world_to_dir(w.x, w.z);
        let err_m = flat.angle_between(ae) * EARTH_RADIUS_M;
        // the tangent plane's horizontal stretch d^3 / 3R^2 at 10 km ~ 8 mm
        assert!(err_m < 0.02, "{err_m} m");
        // and the layer's parabolic bend agrees with this map to < 50 m over its whole extent
        assert!(bend_error_m(100e3, 3000.0).0 < 50.0);
    }

    #[test]
    fn strength_schedule() {
        assert_eq!(strength(400_000.0), 0.0);
        assert_eq!(strength(500_000.0), 0.0);
        assert_eq!(strength(150_000.0), 1.0);
        assert_eq!(strength(40.0), 1.0);
        let mut h = 400_000.0;
        let mut last = 0.0;
        while h > 140_000.0 {
            let s = strength(h);
            assert!(s >= last && s - last < 0.02);
            last = s;
            h *= 0.995;
        }
        // the graft is complete before the mountains layer's ground weight leaves 0 (140 km)
        assert_eq!(strength(140_000.0), 1.0);
        assert_eq!(crate::descent::handoff(140_000.0).ground, 0.0);
    }

    #[test]
    fn file_round_trip() {
        let l = GraftLevel { size: 2, half_m: 100.0, albedo: (0..16).collect(), aux: (16..32).collect() };
        let f = GraftFile { lat: 46.1, lon: 7.6, world_xz: (7700.0, -4200.0), r_full_m: 1.0, r_out_m: 2.0, sun_world: [0.1, 0.2, 0.3], levels: vec![l.clone(), l] };
        let g = GraftFile::from_bytes(&f.to_bytes()).unwrap();
        assert_eq!(g.levels.len(), 2);
        assert_eq!(g.levels[1].aux, f.levels[1].aux);
        assert_eq!((g.lat, g.world_xz, g.sun_world), (f.lat, f.world_xz, f.sun_world));
        assert!(GraftFile::from_bytes(&f.to_bytes()[..50]).is_none());
    }
}
