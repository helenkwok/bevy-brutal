//! Coordinate frames shared by terrain, voxels and (later) ocean/atmosphere.
//!
//! * **World**: f64 metres in a nested frame hierarchy (galaxy > system > planet > local), see [`frames`].
//!   The brick lattice below is the planet-local voxel layer.
//! * **Lattice**: global integer sample index `i64` at a fixed spacing. All geometry that must be
//!   watertight across bricks/chunks is derived from lattice integers, never from accumulated floats.
//! * **Brick**: `brick_size^3` voxels (16 / 32 / 64) of `voxel_m` metres, addressed by `BrickCoord`.
//!   LOD `l` bricks use spacing `voxel_m * 2^l` and cover `2^l` times the extent.
//! * **Geodetic**: WGS84 lat/lon/height, ECEF, and local East-North-Up tangent frames, see [`geodetic`].
//! * **Render**: f32 metres relative to a camera/render origin (`to_render`), so precision does not
//!   degrade far from the world origin.
//!
//! No dependencies; integer conversions use floor semantics (correct for negative coordinates).
#![forbid(unsafe_code)]

pub mod frames;
pub mod geodetic;
pub use frames::{FloatingOrigin, Frame, FrameError, FrameId, FrameLevel, FrameTree, WorldPos};
pub use geodetic::{Enu, Geodetic};
pub use glam::{DVec3, Vec3};

/// Floor division (rounds toward -inf). `b > 0`.
#[inline]
pub const fn floor_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    if (a % b != 0) && ((a < 0) != (b < 0)) { q - 1 } else { q }
}

/// Non-negative remainder matching `floor_div`. `b > 0`.
#[inline]
pub const fn floor_mod(a: i64, b: i64) -> i64 {
    a - floor_div(a, b) * b
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BrickCoord {
    pub x: i64,
    pub y: i64,
    pub z: i64,
    pub lod: u8,
}

#[derive(Debug, PartialEq)]
pub enum GridError {
    BrickSize(u32),
    Spacing(f64),
}

/// Brick layout: runtime-configurable brick size and finest voxel spacing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrickGrid {
    brick: u32,
    voxel_m: f64,
}

impl BrickGrid {
    pub const ALLOWED_SIZES: [u32; 3] = [16, 32, 64];

    pub fn new(brick: u32, voxel_m: f64) -> Result<Self, GridError> {
        if !Self::ALLOWED_SIZES.contains(&brick) {
            return Err(GridError::BrickSize(brick));
        }
        if !(voxel_m.is_finite() && voxel_m > 0.0) {
            return Err(GridError::Spacing(voxel_m));
        }
        Ok(Self { brick, voxel_m })
    }

    pub fn brick_size(&self) -> u32 {
        self.brick
    }

    /// Finest (LOD0) voxel spacing in metres.
    pub fn voxel_m(&self) -> f64 {
        self.voxel_m
    }

    /// Lattice step of a LOD in LOD0 voxel units (`2^lod`).
    pub fn step(&self, lod: u8) -> i64 {
        1i64 << lod
    }

    /// Spacing of a LOD in metres.
    pub fn spacing_m(&self, lod: u8) -> f64 {
        self.voxel_m * self.step(lod) as f64
    }

    /// Edge length of a brick at `lod`, metres.
    pub fn brick_extent_m(&self, lod: u8) -> f64 {
        self.spacing_m(lod) * self.brick as f64
    }

    /// LOD0 lattice index of the brick's minimum corner.
    pub fn brick_origin(&self, b: BrickCoord) -> [i64; 3] {
        let s = self.brick as i64 * self.step(b.lod);
        [b.x * s, b.y * s, b.z * s]
    }

    /// Brick at `lod` containing LOD0 lattice point `p`, plus `p`'s offset inside it (LOD0 units).
    pub fn brick_of(&self, p: [i64; 3], lod: u8) -> (BrickCoord, [i64; 3]) {
        let s = self.brick as i64 * self.step(lod);
        let b = BrickCoord { x: floor_div(p[0], s), y: floor_div(p[1], s), z: floor_div(p[2], s), lod };
        (b, [floor_mod(p[0], s), floor_mod(p[1], s), floor_mod(p[2], s)])
    }

    /// World position (f64 metres) of LOD0 lattice point `p`. Exact for |p| < 2^53.
    pub fn lattice_to_world(&self, p: [i64; 3]) -> [f64; 3] {
        p.map(|v| v as f64 * self.voxel_m)
    }

    /// LOD0 lattice cell containing world point `w` (floor).
    pub fn world_to_lattice(&self, w: [f64; 3]) -> [i64; 3] {
        w.map(|v| (v / self.voxel_m).floor() as i64)
    }
}

/// Camera-relative render position. The subtraction is done in f64 *before* narrowing, so two
/// producers that start from the same lattice point get bit-identical f32 results.
#[inline]
pub fn to_render(world: [f64; 3], origin: [f64; 3]) -> [f32; 3] {
    [(world[0] - origin[0]) as f32, (world[1] - origin[1]) as f32, (world[2] - origin[2]) as f32]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_div_mod_negative() {
        for a in -100i64..100 {
            for b in [1i64, 3, 16, 64] {
                let (q, r) = (floor_div(a, b), floor_mod(a, b));
                assert_eq!(q * b + r, a);
                assert!((0..b).contains(&r));
                assert_eq!(q, (a as f64 / b as f64).floor() as i64);
            }
        }
    }

    #[test]
    fn grid_validation() {
        assert!(BrickGrid::new(32, 0.25).is_ok());
        assert_eq!(BrickGrid::new(24, 0.25), Err(GridError::BrickSize(24)));
        assert!(BrickGrid::new(16, 0.0).is_err() && BrickGrid::new(16, f64::NAN).is_err());
    }

    #[test]
    fn brick_round_trip_all_configs() {
        for size in BrickGrid::ALLOWED_SIZES {
            for v in [0.125, 0.25, 0.5] {
                let g = BrickGrid::new(size, v).unwrap();
                for lod in 0..4 {
                    for p in [[0i64, 0, 0], [-1, -1, -1], [1000, -37, 5], [-(1 << 40), 7, 1 << 40]] {
                        let (b, off) = g.brick_of(p, lod);
                        let o = g.brick_origin(b);
                        assert_eq!([o[0] + off[0], o[1] + off[1], o[2] + off[2]], p);
                        assert!(off.iter().all(|&c| (0..size as i64 * g.step(lod)).contains(&c)));
                    }
                }
                assert_eq!(g.brick_extent_m(1), 2.0 * size as f64 * v);
            }
        }
    }

    #[test]
    fn world_lattice_and_render() {
        let g = BrickGrid::new(32, 0.25).unwrap();
        assert_eq!(g.world_to_lattice([-0.01, 0.0, 0.26]), [-1, 0, 1]);
        assert_eq!(g.lattice_to_world([4, -8, 1]), [1.0, -2.0, 0.25]);
        // far from origin: render-relative keeps sub-mm precision
        let far = g.lattice_to_world([40_000_001, 0, 0]); // 10,000 km + 0.25 m
        let r = to_render(far, [10_000_000.0, 0.0, 0.0]);
        assert_eq!(r, [0.25, 0.0, 0.0]);
    }
}
