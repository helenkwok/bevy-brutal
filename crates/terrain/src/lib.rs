//! Chunked terrain: deterministic fBm value-noise heightmaps (scalar ref + `wide::f32x8`)
//! and grid meshing into a per-frame arena. GPU upload lives in `render-ash` (stubbed).
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use mem::FrameArena;
use wide::{f32x8, i32x8};

/// Samples per chunk edge at LOD0 (65 -> 64 quads).
pub const CHUNK_SAMPLES: usize = 65;
/// Heightmap is padded by one sample on each side for central-difference normals.
pub const PADDED: usize = CHUNK_SAMPLES + 2;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ChunkRequest {
    pub coord: [i32; 2],
    pub lod: u8,
    pub seed: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct NoiseParams {
    pub octaves: u32,
    pub base_freq: f32,
    pub amplitude: f32,
    pub lacunarity: f32,
    pub gain: f32,
}

impl Default for NoiseParams {
    fn default() -> Self {
        Self { octaves: 5, base_freq: 1.0 / 128.0, amplitude: 40.0, lacunarity: 2.0, gain: 0.5 }
    }
}

// ---- scalar reference --------------------------------------------------------------------

#[inline]
fn hash(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    (h & 0x00ff_ffff) as f32 * (1.0 / 16_777_216.0) // [0,1)
}

#[inline]
fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let (fx, fy) = (x.floor(), y.floor());
    let (ix, iy) = (fx as i32, fy as i32);
    let (tx, ty) = (smooth(x - fx), smooth(y - fy));
    let a = hash(ix, iy, seed);
    let b = hash(ix + 1, iy, seed);
    let c = hash(ix, iy + 1, seed);
    let d = hash(ix + 1, iy + 1, seed);
    let top = a + (b - a) * tx;
    let bot = c + (d - c) * tx;
    top + (bot - top) * ty
}

/// Scalar fBm height at world position (x, z). The oracle for the SIMD path.
pub fn height_scalar(x: f32, z: f32, seed: u32, p: &NoiseParams) -> f32 {
    let (mut f, mut a, mut h) = (p.base_freq, p.amplitude, 0.0);
    for o in 0..p.octaves {
        h += (value_noise(x * f, z * f, seed.wrapping_add(o)) * 2.0 - 1.0) * a;
        f *= p.lacunarity;
        a *= p.gain;
    }
    h
}

// ---- SIMD ----------------------------------------------------------------------------------

#[inline]
fn hash8(x: i32x8, y: i32x8, seed: u32) -> f32x8 {
    // wrapping u32 math emulated on i32 lanes (two's complement, identical bits)
    let mut h = (x * i32x8::splat(0x8da6_b343u32 as i32))
        ^ (y * i32x8::splat(0xd816_3841u32 as i32))
        ^ i32x8::splat(seed.wrapping_mul(0xcb1a_b31f) as i32);
    h ^= lsr(h, 13);
    h *= i32x8::splat(0x5bd1_e995u32 as i32);
    h ^= lsr(h, 15);
    (h & i32x8::splat(0x00ff_ffff)).round_float() * f32x8::splat(1.0 / 16_777_216.0)
}

/// Logical shift right on i32 lanes (wide's `>>` on i32x8 is arithmetic).
#[inline]
fn lsr(v: i32x8, n: i32) -> i32x8 {
    (v >> n) & i32x8::splat(((u32::MAX) >> n) as i32)
}

#[inline]
fn smooth8(t: f32x8) -> f32x8 {
    t * t * (f32x8::splat(3.0) - f32x8::splat(2.0) * t)
}

fn value_noise8(x: f32x8, y: f32x8, seed: u32) -> f32x8 {
    let (fx, fy) = (x.floor(), y.floor());
    let (ix, iy) = (fx.round_int(), fy.round_int()); // already integral, so round == trunc
    let (tx, ty) = (smooth8(x - fx), smooth8(y - fy));
    let one = i32x8::splat(1);
    let a = hash8(ix, iy, seed);
    let b = hash8(ix + one, iy, seed);
    let c = hash8(ix, iy + one, seed);
    let d = hash8(ix + one, iy + one, seed);
    let top = a + (b - a) * tx;
    let bot = c + (d - c) * tx;
    top + (bot - top) * ty
}

/// SIMD fBm for 8 points.
pub fn height_simd(x: f32x8, z: f32x8, seed: u32, p: &NoiseParams) -> f32x8 {
    let (mut f, mut a, mut h) = (p.base_freq, p.amplitude, f32x8::ZERO);
    for o in 0..p.octaves {
        let n = value_noise8(x * f32x8::splat(f), z * f32x8::splat(f), seed.wrapping_add(o));
        h += (n * f32x8::splat(2.0) - f32x8::ONE) * f32x8::splat(a);
        f *= p.lacunarity;
        a *= p.gain;
    }
    h
}

/// World-space spacing between samples for a LOD.
pub fn spacing(lod: u8) -> f32 {
    (1u32 << lod) as f32
}

fn origin(req: &ChunkRequest) -> (f32, f32) {
    let size = (CHUNK_SAMPLES - 1) as f32 * spacing(req.lod);
    (req.coord[0] as f32 * size, req.coord[1] as f32 * size)
}

/// Fill a PADDED x PADDED heightmap (row-major, z rows) with the scalar reference.
pub fn heightmap_scalar(req: &ChunkRequest, p: &NoiseParams, out: &mut [f32]) {
    assert_eq!(out.len(), PADDED * PADDED);
    let (ox, oz) = origin(req);
    let s = spacing(req.lod);
    for j in 0..PADDED {
        for i in 0..PADDED {
            let x = ox + (i as f32 - 1.0) * s;
            let z = oz + (j as f32 - 1.0) * s;
            out[j * PADDED + i] = height_scalar(x, z, req.seed, p);
        }
    }
}

/// Fill a PADDED x PADDED heightmap with the SIMD path (8 samples per step, scalar row tail).
pub fn heightmap_simd(req: &ChunkRequest, p: &NoiseParams, out: &mut [f32]) {
    assert_eq!(out.len(), PADDED * PADDED);
    #[cfg(feature = "tracy")]
    let _z = tracy_client::span!("terrain::heightmap");
    let (ox, oz) = origin(req);
    let s = spacing(req.lod);
    let lane = f32x8::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let (rows, _) = out.as_chunks_mut::<PADDED>();
    for (j, row) in rows.iter_mut().enumerate() {
        let z = f32x8::splat(oz + (j as f32 - 1.0) * s);
        let (blocks, tail) = row.as_chunks_mut::<8>();
        for (b, c) in blocks.iter_mut().enumerate() {
            let i0 = b * 8;
            let x = f32x8::splat(ox) + (f32x8::splat(i0 as f32 - 1.0) + lane) * f32x8::splat(s);
            *c = height_simd(x, z, req.seed, p).to_array();
        }
        let i0 = blocks.len() * 8;
        for (k, v) in tail.iter_mut().enumerate() {
            let i = i0 + k;
            *v = height_scalar(ox + (i as f32 - 1.0) * s, oz + (j as f32 - 1.0) * s, req.seed, p);
        }
    }
}

// ---- meshing -------------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
pub struct TerrainVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
}

pub struct ChunkMeshCpu<'a> {
    pub vertices: &'a mut [TerrainVertex],
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
}

/// Index buffer shared by every chunk at every LOD (same grid topology). Generate once.
pub fn grid_indices() -> Vec<u32> {
    let n = CHUNK_SAMPLES as u32;
    let mut idx = Vec::with_capacity(((n - 1) * (n - 1) * 6) as usize);
    for z in 0..n - 1 {
        for x in 0..n - 1 {
            let i = z * n + x;
            idx.extend_from_slice(&[i, i + n, i + 1, i + 1, i + n, i + n + 1]);
        }
    }
    idx
}

/// Mesh a padded heightmap into the arena. Returns `None` if the arena is full.
pub fn mesh_chunk<'a>(req: &ChunkRequest, heights: &[f32], arena: &'a FrameArena) -> Option<ChunkMeshCpu<'a>> {
    assert_eq!(heights.len(), PADDED * PADDED);
    let (ox, oz) = origin(req);
    let s = spacing(req.lod);
    let verts = arena.alloc_slice_fill(CHUNK_SAMPLES * CHUNK_SAMPLES, TerrainVertex::zeroed())?;
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    let h = |i: usize, j: usize| heights[j * PADDED + i];
    for j in 0..CHUNK_SAMPLES {
        for i in 0..CHUNK_SAMPLES {
            let (pi, pj) = (i + 1, j + 1);
            let y = h(pi, pj);
            let dx = (h(pi + 1, pj) - h(pi - 1, pj)) / (2.0 * s);
            let dz = (h(pi, pj + 1) - h(pi, pj - 1)) / (2.0 * s);
            let inv = 1.0 / (dx * dx + 1.0 + dz * dz).sqrt();
            let pos = [ox + i as f32 * s, y, oz + j as f32 * s];
            for k in 0..3 {
                lo[k] = lo[k].min(pos[k]);
                hi[k] = hi[k].max(pos[k]);
            }
            verts[j * CHUNK_SAMPLES + i] = TerrainVertex { pos, normal: [-dx * inv, inv, -dz * inv] };
        }
    }
    Some(ChunkMeshCpu { vertices: verts, aabb_min: lo, aabb_max: hi })
}

// ---- LOD seams: interface (strategy decided, implementation is Prompt C1) ---------------------

/// LOD of the four edge neighbours, in order -X, +X, -Z, +Z. `None` = no neighbour loaded.
pub type NeighbourLods = [Option<u8>; 4];

#[derive(Debug, PartialEq, Eq)]
pub enum SeamError {
    /// The strategy cannot join a chunk to a neighbour at a different LOD.
    LodMismatch { edge: usize, lod: u8, neighbour: u8 },
}

/// How chunk meshes are made crack-free across LOD boundaries.
///
/// **Decided (20 Oct 2026):** far field = 2:1 restricted cube-sphere quadtree with *shader edge
/// snapping* (the cull pass writes a per-chunk neighbour-level mask; one shared index buffer; odd
/// edge vertices snap onto the coarser neighbour's edge from identical integer inputs) plus a CDLOD
/// morph (Prompt C1); near field = voxel bricks with Transvoxel (Prompt C2, `crates/voxel`).
/// Skirts are only a fallback for far chunks whose neighbours have not streamed in yet.
/// Only [`SingleLod`] exists in this crate today; see starter.md "Prompt C1".
pub trait SeamStrategy {
    fn name(&self) -> &'static str;
    /// Check (and, for real strategies, patch) a chunk mesh given its neighbours' LODs. May append
    /// vertices to `extra_vertices` and return chunk-specific indices; `None` = use the shared grid.
    fn resolve(&self, req: &ChunkRequest, neighbours: NeighbourLods, extra_vertices: &mut Vec<TerrainVertex>) -> Result<Option<Vec<u32>>, SeamError>;
}

/// All chunks at one LOD: neighbouring chunks share edge samples bit-for-bit, so the shared grid
/// index buffer is already watertight. Errors on any LOD mismatch instead of producing cracks.
#[derive(Clone, Copy, Debug, Default)]
pub struct SingleLod;

impl SeamStrategy for SingleLod {
    fn name(&self) -> &'static str {
        "single-lod"
    }

    fn resolve(&self, req: &ChunkRequest, neighbours: NeighbourLods, _extra: &mut Vec<TerrainVertex>) -> Result<Option<Vec<u32>>, SeamError> {
        for (edge, n) in neighbours.iter().enumerate() {
            if let Some(n) = *n
                && n != req.lod
            {
                return Err(SeamError::LodMismatch { edge, lod: req.lod, neighbour: n });
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN_4X4_SEED42: u64 = 0x5f56_9ace_b6ff_4376;

    #[test]
    fn simd_matches_scalar() {
        let p = NoiseParams::default();
        for req in [
            ChunkRequest { coord: [0, 0], lod: 0, seed: 42 },
            ChunkRequest { coord: [-3, 7], lod: 2, seed: 7 },
            ChunkRequest { coord: [1000, -1000], lod: 1, seed: 1 },
        ] {
            let mut a = vec![0.0; PADDED * PADDED];
            let mut b = vec![0.0; PADDED * PADDED];
            heightmap_scalar(&req, &p, &mut a);
            heightmap_simd(&req, &p, &mut b);
            // Same op order on both paths and no FMA contraction: expected bit-identical; the
            // prompt's acceptance bound is 1e-4.
            let err = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
            assert!(err <= 1e-4, "{req:?}: max abs err {err}");
            let identical = a.iter().zip(&b).filter(|(x, y)| x.to_bits() == y.to_bits()).count();
            assert_eq!(identical, a.len(), "{req:?}: SIMD and scalar differ in bits");
        }
    }

    /// Golden hash of a quantized 4x4-chunk heightmap (seed 42, LOD0, mm resolution). Changing the
    /// noise, its parameters or the sample layout must update this constant deliberately.
    #[test]
    fn golden_hash_4x4_seed42() {
        let p = NoiseParams::default();
        let mut hm = vec![0.0; PADDED * PADDED];
        let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a
        for cz in 0..4 {
            for cx in 0..4 {
                heightmap_simd(&ChunkRequest { coord: [cx, cz], lod: 0, seed: 42 }, &p, &mut hm);
                for v in &hm {
                    for b in ((v * 1000.0).round() as i32).to_le_bytes() {
                        h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
                    }
                }
            }
        }
        assert_eq!(h, GOLDEN_4X4_SEED42, "golden terrain hash changed: {h:#018x}");
    }

    #[test]
    fn deterministic_and_nontrivial() {
        let p = NoiseParams::default();
        let req = ChunkRequest { coord: [2, 3], lod: 0, seed: 42 };
        let mut a = vec![0.0; PADDED * PADDED];
        let mut b = vec![0.0; PADDED * PADDED];
        heightmap_simd(&req, &p, &mut a);
        heightmap_simd(&req, &p, &mut b);
        assert_eq!(a, b);
        let (lo, hi) = a.iter().fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(v), h.max(v)));
        assert!(hi - lo > 1.0, "terrain is flat: {lo}..{hi}");
        assert!(lo >= -p.amplitude * 2.0 && hi <= p.amplitude * 2.0);
    }

    #[test]
    fn neighbouring_chunks_share_edges() {
        let p = NoiseParams::default();
        let mut a = vec![0.0; PADDED * PADDED];
        let mut b = vec![0.0; PADDED * PADDED];
        heightmap_scalar(&ChunkRequest { coord: [0, 0], lod: 0, seed: 5 }, &p, &mut a);
        heightmap_scalar(&ChunkRequest { coord: [1, 0], lod: 0, seed: 5 }, &p, &mut b);
        for j in 1..=CHUNK_SAMPLES {
            assert_eq!(a[j * PADDED + CHUNK_SAMPLES], b[j * PADDED + 1], "seam at row {j}");
        }
    }

    #[test]
    fn single_lod_seam_strategy() {
        let req = ChunkRequest { coord: [0, 0], lod: 0, seed: 1 };
        let mut extra = Vec::new();
        assert_eq!(SingleLod.resolve(&req, [Some(0), None, Some(0), Some(0)], &mut extra), Ok(None));
        assert_eq!(
            SingleLod.resolve(&req, [Some(0), Some(1), None, None], &mut extra),
            Err(SeamError::LodMismatch { edge: 1, lod: 0, neighbour: 1 })
        );
    }

    /// Adjacent chunks' meshes put bit-identical vertices on the shared edge (single LOD seam test).
    #[test]
    fn adjacent_chunk_meshes_share_edge_vertices() {
        let p = NoiseParams::default();
        let arena = FrameArena::with_capacity(3 * (CHUNK_SAMPLES * CHUNK_SAMPLES * 24 + 64));
        let mesh = |c: [i32; 2]| {
            let req = ChunkRequest { coord: c, lod: 0, seed: 9 };
            let mut hm = vec![0.0; PADDED * PADDED];
            heightmap_simd(&req, &p, &mut hm);
            mesh_chunk(&req, &hm, &arena).unwrap().vertices.to_vec()
        };
        let (a, b, c) = (mesh([0, 0]), mesh([1, 0]), mesh([0, 1]));
        let n = CHUNK_SAMPLES;
        for k in 0..n {
            assert_eq!(a[k * n + n - 1].pos.map(f32::to_bits), b[k * n].pos.map(f32::to_bits), "x seam row {k}");
            assert_eq!(a[(n - 1) * n + k].pos.map(f32::to_bits), c[k].pos.map(f32::to_bits), "z seam col {k}");
            // padded central differences: normals agree too (no lighting seam)
            assert_eq!(a[k * n + n - 1].normal, b[k * n].normal);
        }
    }

    #[test]
    fn meshing() {
        let p = NoiseParams::default();
        let req = ChunkRequest { coord: [0, 0], lod: 0, seed: 42 };
        let mut hm = vec![0.0; PADDED * PADDED];
        heightmap_simd(&req, &p, &mut hm);
        let arena = FrameArena::with_capacity(CHUNK_SAMPLES * CHUNK_SAMPLES * 24 + 64);
        let m = mesh_chunk(&req, &hm, &arena).unwrap();
        assert_eq!(m.vertices.len(), CHUNK_SAMPLES * CHUNK_SAMPLES);
        for v in m.vertices.iter() {
            let len = (v.normal[0].powi(2) + v.normal[1].powi(2) + v.normal[2].powi(2)).sqrt();
            assert!((len - 1.0).abs() < 1e-4 && v.normal[1] > 0.0);
        }
        assert_eq!(grid_indices().len(), 64 * 64 * 6);
        assert_eq!(m.aabb_max[0], 64.0);
    }
}
