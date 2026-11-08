//! `planet`: a cube-sphere planet for rendering from orbit down to the ground.
//!
//! - [`cube`]: the six faces, the integer face lattice and the tangent-warped face -> sphere map
//!   (exact on face edges, so neighbouring faces produce bit-identical edge directions).
//! - [`quadtree`]: per-face quadtree keyed by integers (face, level, i, j); screen-space-error
//!   selection from an f64 camera, frustum + horizon culling, 2:1 restriction (balance) and the
//!   per-patch neighbour mask used for stitching.
//! - [`patch`]: patch meshes (N x N quads + skirts) in f64, emitted as f32 offsets from an f64 patch
//!   origin, plus the 16 stitched index variants.
//! - [`descent`]: the orbit -> mountains descent (nested f64 frames, camera path, renderer handoff).
//! - [`dem`]: real-world DEM grids (`terrain::dem`, `geo-fetch`) placed on the planet: ENU / WGS84
//!   -> planet frame in f64, camera-relative f32, height blend into the cube-sphere patches and a
//!   graft bake for the shading.
//! - [`graft`]: the shared Alps (round 4): the mountains terrain inside the planet render.
//! - [`bake`] / [`pack`]: Earth source imagery -> cube-map textures (BC7 / BC5 / BC4 with mips) and
//!   a CPU height cube in the asset container (`PLN1`).
pub mod bake;
pub mod cloudnoise;
pub mod cube;
pub mod dem;
pub mod descent;
pub mod drape;
pub mod graft;
pub mod ktx2;
pub mod pack;
pub mod patch;
pub mod quadtree;
pub mod solar;

/// Mean Earth radius (m).
pub const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Bounded close-graft tile streaming.
pub mod stream;

/// Procedural emissive universe demonstration (not a stellar catalogue).
pub mod universe;

/// External observed GCNS records (no bundled data).
pub mod gcns;

/// Independent CNS5 / DESI positional diagnostics (not merged or photometric stars).
pub mod observed;
