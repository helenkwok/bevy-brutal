//! Near-field sparse voxel bricks (Prompt C2): density field -> quantised sparse bricks ->
//! Transvoxel meshing (regular + transition cells) with watertight LOD seams, including
//! camera-centred LOD **shells** (bricks touching finer bricks across faces, edges or corners).
//!
//! The pipeline is split so meshing is a pure integer function (identical on CPU and GPU):
//! 1. **Sampling** (CPU, f64 field): every lattice sample is quantised once to `i32`
//!    ([`VoxelConfig::quantise`]). Bricks store `(n+3)^3` samples (one-sample apron for normals),
//!    plus the `(2n+1)^2` fine samples of each transition face.
//! 2. **Meshing** ([`mesh_brick`], and `render_ash::voxel_gpu` in Slang): integers only.
//!
//! ## Robustness hard rule (as implemented; acceptance tests in `tests/voxel.rs`)
//! * **H1 integer grid:** all meshing and all topology / orientation / degeneracy tests run on
//!   integer coordinates: global sub-grid positions, [`SUB`] = 256 steps per LOD0 voxel
//!   (brick-local `i32` on the GPU). Vertices are snapped to that sub-grid before any test.
//! * **H2 sign:** inside iff the quantised density `< 0`. Exact zero, `-0.0` and sub-quantum values
//!   are outside. Case codes come only from this test.
//! * **H3 exact interpolation:** the edge parameter is `t = round(256 d_a / (d_a - d_b))` in
//!   integer arithmetic, clamped to `[1, 255]`, with endpoints in canonical (lower lattice point
//!   first) order. So a shared edge yields the same integer vertex in every brick that sees it,
//!   and distinct edges never produce coincident vertices.
//! * **H4 exact predicates:** degeneracy = integer cross product `== 0` (i128). Orientation =
//!   exact edge-direction consistency plus the exact (i128) sign of axis-determined triangles
//!   (see [`Topology::flipped`]). No float equality anywhere.
//! * **H5 bit-identical seams:** seam vertices are computed from identical inputs on both sides
//!   (same lattice samples, same integer formula). Render positions are formed once from the
//!   global integer (`(p - origin) * voxel/256`, f64 then f32), so they are bit-identical too.
//! * **H6 finite input:** non-finite density is an error, never meshed.
//! * **H7 determinism:** output depends only on lattice samples; bricks are keyed by coordinate.
//! * **H8 refuse, don't crack:** unsupported layouts are errors: LOD difference > 1 between any of
//!   the 26 neighbours, a finer region that is not a box ([`LodLayout::finer_box`]), or a brick
//!   whose seam displacement would reach the boundary it shares with a coarser brick.
//!
//! ## LOD seams (multi-face / shells)
//! For a brick at LOD `L` with the finer region `R` (an axis-aligned box), every LOD-`L` vertex is
//! displaced by a single **global** integer field `Φ(p)`. Per axis it is a tent around each of
//! `R`'s planes in that axis (width one coarse cell), times a fall-off for the other two axes
//! ([`phi`]). On a face of `R` it moves coarse vertices inward by `S/4` (S = coarse cell).
//! Along edges and corners of `R` it moves them diagonally, so the transition cells of
//! neighbouring faces meet on diagonal planes (Lengyel's 1-3 transition-cell shrink).
//! Bricks touching `R` only along an edge or a corner get the same displacement, so every pair
//! of same-LOD neighbours agrees on its shared face (Φ is one global function). Transition
//! cells: the full-resolution side uses the undisplaced fine lattice (identical to the finer
//! neighbour), and the half-resolution side reuses the displaced coarse vertices.
#![forbid(unsafe_code)]

pub mod boulder;
pub mod fields;
pub mod ground;
pub mod meshq;
pub mod noise;
pub mod scene;
pub mod tables;

use std::collections::{BTreeMap, HashMap};

use bytemuck::{Pod, Zeroable};
use coords::{BrickCoord, BrickGrid, floor_div};
use rayon::prelude::*;
use tables::*;

/// Sub-grid steps per LOD0 voxel (vertex snapping grid, H1).
pub const SUB: i64 = 256;
/// Edge parameter quantisation steps (H3).
pub const T_STEPS: i64 = 256;
/// Density quanta per LOD0 voxel length (density is a signed distance-like value in metres).
pub const DENSITY_QUANTA_PER_VOXEL: f64 = 1024.0;
/// Quantised density clamp (keeps all GPU integer arithmetic within i32).
pub const DQ_MAX: i32 = 1 << 20;
/// Highest LOD that may have transitions (keeps Φ products within i32 on the GPU).
pub const MAX_LOD: u8 = 6;

/// A signed density field in world metres (planet-local frame): negative = solid.
pub trait DensityField: Sync {
    fn density(&self, p: [f64; 3]) -> f64;
    /// Band-limited density for sampling at cell size `cell_m` (LOD-aware filtering). Fields
    /// may soften creases / drop sub-cell octaves here; default is the unfiltered density.
    /// Used by [`sample_brick_filtered`], which picks `cell_m` from [`filter_level`], a pure
    /// function of the lattice point, so every brick sampling a shared point gets the same
    /// value and the watertightness argument (H2) is unchanged.
    fn density_lod(&self, p: [f64; 3], cell_m: f64) -> f64 {
        let _ = cell_m;
        self.density(p)
    }
}

/// Axis-aligned box of LOD0 lattice points, half-open `[lo, hi)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Aabb {
    pub lo: [i64; 3],
    pub hi: [i64; 3],
}

impl Aabb {
    pub fn contains(&self, p: [i64; 3]) -> bool {
        (0..3).all(|i| p[i] >= self.lo[i] && p[i] < self.hi[i])
    }
}

/// Effectively-infinite lattice bound for half-space layouts.
pub const BIG: i64 = 1 << 40;

/// Which LOD owns the space around each LOD0 lattice point.
pub trait LodLayout: Sync {
    fn lod_at(&self, p: [i64; 3]) -> u8;
    /// The region with LOD `< lod`, if any. It must be a box (H8).
    fn finer_box(&self, lod: u8) -> Option<Aabb>;
}

pub struct UniformLod(pub u8);
impl LodLayout for UniformLod {
    fn lod_at(&self, _: [i64; 3]) -> u8 {
        self.0
    }
    fn finer_box(&self, _: u8) -> Option<Aabb> {
        None
    }
}

/// Two LODs split by the plane `p[axis] == split` (aligned to the coarser brick size).
pub struct SlabLayout {
    pub axis: usize,
    pub split: i64,
    pub lod_below: u8,
    pub lod_above: u8,
}
impl LodLayout for SlabLayout {
    fn lod_at(&self, p: [i64; 3]) -> u8 {
        if p[self.axis] < self.split { self.lod_below } else { self.lod_above }
    }
    fn finer_box(&self, lod: u8) -> Option<Aabb> {
        let (fine_below, fine) =
            if self.lod_below < self.lod_above { (true, self.lod_below) } else { (false, self.lod_above) };
        if lod <= fine || lod > self.lod_below.max(self.lod_above) {
            return None;
        }
        let (mut lo, mut hi) = ([-BIG; 3], [BIG; 3]);
        if fine_below {
            hi[self.axis] = self.split;
        } else {
            lo[self.axis] = self.split;
        }
        Some(Aabb { lo, hi })
    }
}

/// Camera-centred nested LOD boxes: `boxes[l]` = region with LOD `<= l`. Outside the last box is
/// `boxes.len()`. Built so each box is aligned to the next level's bricks and contains the
/// previous one with at least one brick of margin.
#[derive(Clone, Debug)]
pub struct ShellLayout {
    pub boxes: Vec<Aabb>,
}

impl ShellLayout {
    /// `half_bricks[l]`: requested half-width of LOD `l`'s box, in LOD-`l` bricks.
    pub fn camera_centred(grid: &BrickGrid, camera_lattice: [i64; 3], half_bricks: &[i64]) -> Self {
        let n = grid.brick_size() as i64;
        let mut boxes: Vec<Aabb> = Vec::new();
        for (l, &hb) in half_bricks.iter().enumerate() {
            let ext = n << l; // brick extent at LOD l
            let ext_next = n << (l + 1);
            let r = hb * ext;
            let mut lo = camera_lattice.map(|c| c - r);
            let mut hi = camera_lattice.map(|c| c + r);
            if let Some(prev) = boxes.last() {
                for i in 0..3 {
                    lo[i] = lo[i].min(prev.lo[i] - ext);
                    hi[i] = hi[i].max(prev.hi[i] + ext);
                }
            }
            let lo = lo.map(|v| floor_div(v, ext_next) * ext_next);
            let hi = hi.map(|v| -floor_div(-v, ext_next) * ext_next);
            boxes.push(Aabb { lo, hi });
        }
        Self { boxes }
    }

    /// Continuous filter cell size (metres) for LOD-aware band-limiting: `voxel * 2^(L + s)` for
    /// a point in shell `L`, with `s` in [0, 1] ramping from the shell's inner box (0) to its
    /// outer box (1), quantised to 1/16 octave. At every box boundary both shells give the same
    /// value (2^(L-1+1) == 2^L exactly), so the filter has no steps between LOD shells, and it is
    /// a pure function of the lattice point (watertight, H2) and >= the sampling cell there.
    pub fn smooth_filter_cell(&self, voxel_m: f64, p: [i64; 3]) -> f64 {
        let outside = |b: &Aabb| (0..3).map(|i| (b.lo[i] - p[i]).max(p[i] - b.hi[i]).max(0)).max().unwrap_or(0) as f64;
        let inside = |b: &Aabb| (0..3).map(|i| (p[i] - b.lo[i]).min(b.hi[i] - p[i])).min().unwrap_or(0).max(0) as f64;
        let l = self.boxes.iter().position(|b| (0..3).all(|i| p[i] >= b.lo[i] && p[i] <= b.hi[i]));
        let level = match l {
            None => self.boxes.len() as f64,
            Some(l) => {
                let b_out = inside(&self.boxes[l]);
                let a_in = match l {
                    0 => {
                        let b = &self.boxes[0];
                        let c: [i64; 3] = [0, 1, 2].map(|i| (b.lo[i] + b.hi[i]) / 2);
                        (0..3).map(|i| (p[i] - c[i]).abs()).max().unwrap_or(0) as f64
                    }
                    _ => outside(&self.boxes[l - 1]),
                };
                l as f64 + if a_in + b_out > 0.0 { a_in / (a_in + b_out) } else { 1.0 }
            }
        };
        voxel_m * ((level * 16.0).round() / 16.0).exp2()
    }

    pub fn max_lod(&self) -> u8 {
        self.boxes.len() as u8
    }
}

impl LodLayout for ShellLayout {
    fn lod_at(&self, p: [i64; 3]) -> u8 {
        self.boxes.iter().position(|b| b.contains(p)).unwrap_or(self.boxes.len()) as u8
    }
    fn finer_box(&self, lod: u8) -> Option<Aabb> {
        (lod >= 1).then(|| self.boxes.get(lod as usize - 1).copied()).flatten()
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct VoxelVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
}

/// One meshed brick. `ipos` (global sub-grid integers) is the truth; `vertices` are derived
/// render-space f32 (relative to [`VoxelConfig::origin_sub`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BrickMesh {
    pub ipos: Vec<[i64; 3]>,
    pub vertices: Vec<VoxelVertex>,
    /// Per vertex: crossing-edge axis (bits 0-1) and +1 if the outside endpoint is the higher one (bit 2).
    pub dirs: Vec<u8>,
    pub indices: Vec<u32>,
}

impl BrickMesh {
    pub fn bytes(&self) -> usize {
        self.vertices.len() * size_of::<VoxelVertex>() + self.indices.len() * 4
    }
}

#[derive(Debug, PartialEq)]
pub enum VoxelError {
    NonFinite {
        at: [i64; 3],
    },
    LodJump {
        brick: BrickCoord,
        neighbour: u8,
    },
    NotABox {
        brick: BrickCoord,
    },
    /// The LOD-seam displacement of this brick would reach a face/edge/corner it shares with a
    /// coarser brick (finer region closer than one cell to it).
    PhiReachesCoarser {
        brick: BrickCoord,
    },
    LodTooHigh {
        brick: BrickCoord,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct VoxelConfig {
    pub grid: BrickGrid,
    /// Render origin on the global sub-grid (H5). Set it from the camera with [`Self::set_origin`].
    pub origin_sub: [i64; 3],
}

impl VoxelConfig {
    pub fn new(grid: BrickGrid) -> Self {
        Self { grid, origin_sub: [0; 3] }
    }

    /// Snap a planet-frame world position to the sub-grid and use it as the render origin.
    pub fn set_origin(&mut self, world: [f64; 3]) {
        let k = SUB as f64 / self.grid.voxel_m();
        self.origin_sub = world.map(|c| (c * k).round() as i64);
    }

    /// Render origin in world metres (exact).
    pub fn origin_world(&self) -> [f64; 3] {
        let k = self.grid.voxel_m() / SUB as f64;
        self.origin_sub.map(|c| c as f64 * k)
    }

    /// Quantise one density sample (H2): metres -> `i32` quanta of `voxel / 1024`, clamped.
    pub fn quantise(&self, d: f64) -> i32 {
        let q = (d * DENSITY_QUANTA_PER_VOXEL / self.grid.voxel_m()).round();
        q.clamp(-(DQ_MAX as f64), DQ_MAX as f64) as i32
    }

    /// Render-space f32 position of a global sub-grid point (H5).
    pub fn sub_to_render(&self, p: [i64; 3]) -> [f32; 3] {
        let k = self.grid.voxel_m() / SUB as f64;
        [0, 1, 2].map(|i| ((p[i] - self.origin_sub[i]) as f64 * k) as f32)
    }
}

/// Quantised samples of one brick at its LOD.
pub struct BrickSamples {
    pub coord: BrickCoord,
    pub n: usize,
    /// `(n+3)^3` samples, x fastest, index `(i+1, j+1, k+1)` for cell corner `(i, j, k)`.
    pub d: Vec<i32>,
    /// Per face: `(2n+1)^2` fine samples on the face plane (u fastest), empty if not a transition face.
    pub faces: [Vec<i32>; 6],
}

impl BrickSamples {
    #[inline]
    pub fn at(&self, i: i64, j: i64, k: i64) -> i32 {
        let m = self.n as i64 + 3;
        self.d[(((k + 1) * m + (j + 1)) * m + (i + 1)) as usize]
    }

    pub fn bytes(&self) -> usize {
        (self.d.len() + self.faces.iter().map(Vec::len).sum::<usize>()) * 4
    }

    /// True if any cell corner (excluding the apron) differs in sign, or any transition face does.
    pub fn has_sign_change(&self) -> bool {
        let n = self.n as i64;
        let first = self.at(0, 0, 0) < 0;
        let mut any = false;
        'o: for k in 0..=n {
            for j in 0..=n {
                for i in 0..=n {
                    if (self.at(i, j, k) < 0) != first {
                        any = true;
                        break 'o;
                    }
                }
            }
        }
        any || self.faces.iter().any(|f| f.iter().any(|&v| (v < 0) != first))
    }
}

/// Face index: axis * 2 + side (0 = low/min face, 1 = high/max face).
pub const FACES: usize = 6;

/// Face frame: (normal axis, u axis, v axis). u/v are the cyclic successors of the axis.
#[inline]
pub fn face_axes(f: usize) -> (usize, usize, usize) {
    let a = f / 2;
    (a, (a + 1) % 3, (a + 2) % 3)
}

/// How samples are band-limited.
#[derive(Clone, Copy)]
enum Filter<'a> {
    None,
    /// Per point: the coarsest LOD touching it ([`filter_level`]).
    Level(&'a dyn LodLayout),
    /// Per point: [`ShellLayout::smooth_filter_cell`] (no steps between shells).
    Smooth(&'a ShellLayout),
}

fn sample(field: &dyn DensityField, cfg: &VoxelConfig, filter: Filter<'_>, p: [i64; 3]) -> Result<i32, VoxelError> {
    let w = cfg.grid.lattice_to_world(p);
    let d = match filter {
        Filter::None => field.density(w),
        Filter::Level(l) => field.density_lod(w, cfg.grid.voxel_m() * (1u64 << filter_level(l, p)) as f64),
        Filter::Smooth(l) => field.density_lod(w, l.smooth_filter_cell(cfg.grid.voxel_m(), p)),
    };
    if d.is_finite() { Ok(cfg.quantise(d)) } else { Err(VoxelError::NonFinite { at: p }) }
}

/// Filter level of LOD0 lattice point `p`: the coarsest layout LOD among the 8 lattice cells
/// touching `p`. A pure function of `p` (and the layout): a point on a face shared by a fine
/// and a coarse brick is filtered at the coarse level by both, so shared samples agree.
pub fn filter_level(layout: &dyn LodLayout, p: [i64; 3]) -> u8 {
    let mut l = 0;
    for d in 0..8 {
        l = l.max(layout.lod_at([p[0] - (d & 1), p[1] - ((d >> 1) & 1), p[2] - ((d >> 2) & 1)]));
    }
    l
}

/// Sample a brick (with apron) and the fine samples of its transition faces.
pub fn sample_brick(
    field: &dyn DensityField,
    cfg: &VoxelConfig,
    coord: BrickCoord,
    transitions: u8,
) -> Result<BrickSamples, VoxelError> {
    sample_brick_with(field, cfg, Filter::None, coord, transitions)
}

/// [`sample_brick`] band-limited by the continuous [`ShellLayout::smooth_filter_cell`].
pub fn sample_brick_smooth(
    field: &dyn DensityField,
    cfg: &VoxelConfig,
    layout: &ShellLayout,
    coord: BrickCoord,
    transitions: u8,
) -> Result<BrickSamples, VoxelError> {
    sample_brick_with(field, cfg, Filter::Smooth(layout), coord, transitions)
}

/// [`sample_brick`] with LOD-aware band-limiting: each sample uses
/// `field.density_lod(p, cell(filter_level(layout, p)))`.
pub fn sample_brick_filtered(
    field: &dyn DensityField,
    cfg: &VoxelConfig,
    layout: &dyn LodLayout,
    coord: BrickCoord,
    transitions: u8,
) -> Result<BrickSamples, VoxelError> {
    sample_brick_with(field, cfg, Filter::Level(layout), coord, transitions)
}

fn sample_brick_with(
    field: &dyn DensityField,
    cfg: &VoxelConfig,
    layout: Filter<'_>,
    coord: BrickCoord,
    transitions: u8,
) -> Result<BrickSamples, VoxelError> {
    let grid = &cfg.grid;
    let n = grid.brick_size() as usize;
    let m = n as i64 + 3;
    let s = grid.step(coord.lod);
    let o = grid.brick_origin(coord);
    let mut d = Vec::with_capacity((m * m * m) as usize);
    for k in -1..m - 1 {
        for j in -1..m - 1 {
            for i in -1..m - 1 {
                d.push(sample(field, cfg, layout, [o[0] + i * s, o[1] + j * s, o[2] + k * s])?);
            }
        }
    }
    let mut faces: [Vec<i32>; 6] = Default::default();
    for (f, face) in faces.iter_mut().enumerate() {
        if transitions & (1 << f) == 0 {
            continue;
        }
        let (a, ua, va) = face_axes(f);
        let fm = 2 * n as i64 + 1;
        let hs = s / 2;
        face.reserve((fm * fm) as usize);
        for sv in 0..fm {
            for su in 0..fm {
                let mut p = o;
                p[a] += if f % 2 == 1 { n as i64 * s } else { 0 };
                p[ua] += su * hs;
                p[va] += sv * hs;
                face.push(sample(field, cfg, layout, p)?);
            }
        }
    }
    Ok(BrickSamples { coord, n, d, faces })
}

/// Integer edge parameter (H3): `round(T_STEPS * da / (da - db))` clamped to `[1, T_STEPS-1]`.
/// Requires a sign change (`(da < 0) != (db < 0)`).
#[inline]
pub fn edge_t(da: i32, db: i32) -> i64 {
    let (mut num, mut den) = (T_STEPS * da as i64, da as i64 - db as i64);
    if den < 0 {
        num = -num;
        den = -den;
    }
    // num >= 0 here (signs differ, zero counts as outside)
    ((2 * num + den) / (2 * den)).clamp(1, T_STEPS - 1)
}

/// The global LOD-seam displacement Φ (see module docs), in sub-grid units.
/// `lo`/`hi`: the finer box planes (sub-grid units); `log2s`: log2 of the coarse cell in sub-units.
/// Staged shifts keep every intermediate within i32 for the GPU port (same formula there).
#[inline]
pub fn phi(p: [i64; 3], lo: [i64; 3], hi: [i64; 3], log2s: u32) -> [i64; 3] {
    let s = 1i64 << log2s;
    let g = [0, 1, 2].map(|b| (s - (lo[b] - p[b]).max(0).max(p[b] - hi[b])).max(0));
    let mut out = [0i64; 3];
    for (a, o) in out.iter_mut().enumerate() {
        let (b, c) = ((a + 1) % 3, (a + 2) % 3);
        let q = |t: i64| (((t * g[b]) >> log2s) * g[c]) >> (log2s + 2);
        let tp = (s - (p[a] - hi[a]).abs()).max(0);
        let tm = (s - (p[a] - lo[a]).abs()).max(0);
        *o = q(tp) - q(tm);
    }
    out
}

/// Pure-integer brick mesher (H1-H5). `finer`: the layout's finer box for this brick's LOD.
pub fn mesh_brick(cfg: &VoxelConfig, smp: &BrickSamples, transitions: u8, finer: Option<Aabb>) -> BrickMesh {
    let grid = cfg.grid;
    let b = smp.coord;
    let mut m = Mesher {
        cfg,
        smp,
        origin: grid.brick_origin(b),
        s: grid.step(b.lod),
        n: grid.brick_size() as i64,
        phi_box: finer.map(|r| (r.lo.map(|v| v * SUB), r.hi.map(|v| v * SUB))),
        log2s: (grid.step(b.lod) * SUB).trailing_zeros(),
        out: BrickMesh::default(),
        edge_vertex: HashMap::new(),
    };
    m.regular();
    for f in 0..FACES {
        if transitions & (1 << f) != 0 {
            m.transition(f);
        }
    }
    m.out
}

struct Mesher<'a> {
    cfg: &'a VoxelConfig,
    smp: &'a BrickSamples,
    origin: [i64; 3],
    s: i64,
    n: i64,
    phi_box: Option<([i64; 3], [i64; 3])>,
    log2s: u32,
    out: BrickMesh,
    edge_vertex: HashMap<([i64; 3], [i64; 3]), u32>,
}

impl Mesher<'_> {
    /// Gradient (central difference of quantised samples) at brick corner (i, j, k).
    fn grad(&self, i: i64, j: i64, k: i64) -> [f32; 3] {
        let s = self.smp;
        [
            (s.at(i + 1, j, k) - s.at(i - 1, j, k)) as f32,
            (s.at(i, j + 1, k) - s.at(i, j - 1, k)) as f32,
            (s.at(i, j, k + 1) - s.at(i, j, k - 1)) as f32,
        ]
    }

    /// Emit (or reuse) the vertex on lattice edge (pa, pb). `ga/gb`: endpoint gradients.
    #[allow(clippy::too_many_arguments)]
    fn vertex(
        &mut self,
        pa: [i64; 3],
        da: i32,
        ga: [f32; 3],
        pb: [i64; 3],
        db: i32,
        gb: [f32; 3],
        displace: bool,
    ) -> u32 {
        let (a, b, da, db, ga, gb) = if pa < pb { (pa, pb, da, db, ga, gb) } else { (pb, pa, db, da, gb, ga) };
        if let Some(&v) = self.edge_vertex.get(&(a, b)) {
            return v;
        }
        let t = edge_t(da, db);
        let mut ip = [0, 1, 2].map(|i| a[i] * SUB + t * (b[i] - a[i]) * (SUB / T_STEPS));
        if displace && let Some((lo, hi)) = self.phi_box {
            let d = phi(ip, lo, hi, self.log2s);
            ip = [ip[0] + d[0], ip[1] + d[1], ip[2] + d[2]];
        }
        let tf = t as f32 / T_STEPS as f32;
        let g = [0, 1, 2].map(|i| ga[i] + (gb[i] - ga[i]) * tf);
        let gl = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
        let normal = if gl > 0.0 { g.map(|c| c / gl) } else { [0.0, 1.0, 0.0] };
        let axis = (0..3).find(|&i| a[i] != b[i]).unwrap() as u8;
        let outside_high = (db >= 0) as u8; // b is the higher endpoint
        let v = self.out.ipos.len() as u32;
        self.out.ipos.push(ip);
        self.out.vertices.push(VoxelVertex { pos: self.cfg.sub_to_render(ip), normal });
        self.out.dirs.push(axis | (outside_high << 2));
        self.edge_vertex.insert((a, b), v);
        v
    }

    fn regular(&mut self) {
        let (n, s, o) = (self.n, self.s, self.origin);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let corner = |c: usize| (i + (c & 1) as i64, j + ((c >> 1) & 1) as i64, k + ((c >> 2) & 1) as i64);
                    let mut case = 0usize;
                    let mut cd = [0i32; 8];
                    for (c, d) in cd.iter_mut().enumerate() {
                        let (x, y, z) = corner(c);
                        *d = self.smp.at(x, y, z);
                        if *d < 0 {
                            case |= 1 << c;
                        }
                    }
                    if case == 0 || case == 255 {
                        continue;
                    }
                    let (counts, tri) = REGULAR_CELL_DATA[REGULAR_CELL_CLASS[case] as usize];
                    let mut cv = [0u32; 12];
                    for (q, slot) in cv.iter_mut().enumerate().take((counts >> 4) as usize) {
                        let e = REGULAR_VERTEX_DATA[case][q] & 0xFF;
                        let (c0, c1) = (((e >> 4) & 0xF) as usize, (e & 0xF) as usize);
                        let (x0, y0, z0) = corner(c0);
                        let (x1, y1, z1) = corner(c1);
                        let p0 = [o[0] + x0 * s, o[1] + y0 * s, o[2] + z0 * s];
                        let p1 = [o[0] + x1 * s, o[1] + y1 * s, o[2] + z1 * s];
                        let (g0, g1) = (self.grad(x0, y0, z0), self.grad(x1, y1, z1));
                        *slot = self.vertex(p0, cd[c0], g0, p1, cd[c1], g1, true);
                    }
                    for t in 0..(counts & 0xF) as usize {
                        self.out.indices.extend([
                            cv[tri[3 * t] as usize],
                            cv[tri[3 * t + 1] as usize],
                            cv[tri[3 * t + 2] as usize],
                        ]);
                    }
                }
            }
        }
    }

    fn transition(&mut self, f: usize) {
        let (axis, ua, va) = face_axes(f);
        let high = f % 2 == 1;
        let (n, s, o) = (self.n, self.s, self.origin);
        let hs = s / 2;
        let fm = 2 * n + 1;
        // Spiral bit order of the 9 full-res samples (Lengyel, Fig. 4.16).
        const BIT_ORDER: [usize; 9] = [0, 1, 2, 5, 8, 7, 6, 3, 4];
        for cv in 0..n {
            for cu in 0..n {
                let mut sp = [[0i64; 3]; 9];
                let mut sd = [0i32; 9];
                let mut sg = [[0f32; 3]; 9];
                // coarse-corner brick index of face sample (su, sv) in {0, 2}
                let corner_ijk = |su: i64, sv: i64| {
                    let mut c = [0i64; 3];
                    c[axis] = if high { n } else { 0 };
                    c[ua] = cu + su / 2;
                    c[va] = cv + sv / 2;
                    c
                };
                for q in 0..9 {
                    let (su, sv) = ((q % 3) as i64, (q / 3) as i64);
                    let mut p = o;
                    p[axis] += if high { n * s } else { 0 };
                    p[ua] += cu * s + su * hs;
                    p[va] += cv * s + sv * hs;
                    sp[q] = p;
                    sd[q] = self.smp.faces[f][((2 * cv + sv) * fm + (2 * cu + su)) as usize];
                    // normals: average of the coarse-corner gradients spanning this fine sample
                    let (u0, u1) = (su / 2 * 2, (su + 1) / 2 * 2);
                    let (v0, v1) = (sv / 2 * 2, (sv + 1) / 2 * 2);
                    let mut g = [0f32; 3];
                    for (uu, vv) in [(u0, v0), (u1, v0), (u0, v1), (u1, v1)] {
                        let c = corner_ijk(uu, vv);
                        let gc = self.grad(c[0], c[1], c[2]);
                        for i in 0..3 {
                            g[i] += 0.25 * gc[i];
                        }
                    }
                    sg[q] = g;
                }
                let mut case = 0usize;
                for (bit, &q) in BIT_ORDER.iter().enumerate() {
                    if sd[q] < 0 {
                        case |= 1 << bit;
                    }
                }
                if case == 0 || case == 511 {
                    continue;
                }
                let cls = TRANSITION_CELL_CLASS[case];
                let (counts, tri) = TRANSITION_CELL_DATA[(cls & 0x7F) as usize];
                let mut tv = [0u32; 12];
                for (q, slot) in tv.iter_mut().enumerate().take((counts >> 4) as usize) {
                    let e = TRANSITION_VERTEX_DATA[case][q] & 0xFF;
                    let (e0, e1) = (((e >> 4) & 0xF) as usize, (e & 0xF) as usize);
                    // 9..=12 are the half-res copies of full-res corners 0, 2, 6, 8.
                    let half = |x: usize| [0usize, 2, 6, 8].get(x.wrapping_sub(9)).copied();
                    *slot = match (half(e0), half(e1)) {
                        (None, None) => self.vertex(sp[e0], sd[e0], sg[e0], sp[e1], sd[e1], sg[e1], false),
                        (Some(a), Some(b)) => self.vertex(sp[a], sd[a], sg[a], sp[b], sd[b], sg[b], true),
                        _ => unreachable!("transition edges never join the two faces"),
                    };
                }
                // Winding: table winding flipped by the class bit and the face frame handedness.
                let flip = (cls & 0x80 != 0) ^ high;
                for t in 0..(counts & 0xF) as usize {
                    let (a, b, c) = (tv[tri[3 * t] as usize], tv[tri[3 * t + 1] as usize], tv[tri[3 * t + 2] as usize]);
                    if flip { self.out.indices.extend([a, c, b]) } else { self.out.indices.extend([a, b, c]) }
                }
            }
        }
    }
}

/// Neighbourhood checks (H8) and the transition mask for brick `b`.
pub fn plan_brick(grid: &BrickGrid, layout: &dyn LodLayout, b: BrickCoord) -> Result<(u8, Option<Aabb>), VoxelError> {
    let o = grid.brick_origin(b);
    let ext = grid.brick_size() as i64 * grid.step(b.lod);
    let finer = layout.finer_box(b.lod);
    let s = grid.step(b.lod);
    let (mut has_finer, mut mask) = (false, 0u8);
    let mut coarser_dirs: Vec<[i64; 3]> = Vec::new();
    for dz in -1i64..=1 {
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                if (dx, dy, dz) == (0, 0, 0) {
                    continue;
                }
                let d = [dx, dy, dz];
                // a point just across the face / edge / corner, at its centre
                let p = [0, 1, 2].map(|i| match d[i] {
                    -1 => o[i] - 1,
                    1 => o[i] + ext,
                    _ => o[i] + ext / 2,
                });
                let nl = layout.lod_at(p);
                if nl.abs_diff(b.lod) > 1 {
                    return Err(VoxelError::LodJump { brick: b, neighbour: nl });
                }
                if nl + 1 == b.lod {
                    has_finer = true;
                    if !finer.is_some_and(|r| r.contains(p)) {
                        return Err(VoxelError::NotABox { brick: b });
                    }
                    let nonzero = d.iter().filter(|&&c| c != 0).count();
                    if nonzero == 1 {
                        let a = d.iter().position(|&c| c != 0).unwrap();
                        mask |= 1 << (2 * a + (d[a] > 0) as usize);
                    }
                } else if nl == b.lod + 1 {
                    coarser_dirs.push(d);
                }
            }
        }
    }
    if has_finer && b.lod > MAX_LOD {
        return Err(VoxelError::LodTooHigh { brick: b });
    }
    // Φ is zero wherever some axis is >= one cell outside the finer box; the boundary shared with
    // a coarser brick must be entirely in that zone (it is the full-res side of *its* seam).
    if let Some(r) = finer.filter(|_| b.lod >= 1) {
        for d in coarser_dirs {
            let far = (0..3).any(|i| {
                let (lo, hi) = match d[i] {
                    -1 => (o[i], o[i]),
                    1 => (o[i] + ext, o[i] + ext),
                    _ => (o[i], o[i] + ext),
                };
                (r.lo[i] - hi).max(lo - r.hi[i]) >= s
            });
            if !far {
                return Err(VoxelError::PhiReachesCoarser { brick: b });
            }
        }
    }
    // Φ applies to every LOD-L vertex near the finer box (also edge/corner-only neighbours).
    Ok((mask, finer.filter(|_| b.lod >= 1)))
}

/// A meshed sparse brick set.
#[derive(Default)]
pub struct VoxelWorld {
    pub bricks: BTreeMap<BrickCoord, BrickMesh>,
    /// Bricks sampled (including empty/full ones that were dropped).
    pub sampled: usize,
    /// Total sample bytes held while meshing occupied bricks.
    pub sample_bytes: usize,
}

impl VoxelWorld {
    pub fn triangles(&self) -> usize {
        self.bricks.values().map(|m| m.indices.len() / 3).sum()
    }
    pub fn vertices(&self) -> usize {
        self.bricks.values().map(|m| m.vertices.len()).sum()
    }
    pub fn mesh_bytes(&self) -> usize {
        self.bricks.values().map(BrickMesh::bytes).sum()
    }
    /// Concatenate all bricks into one mesh (for upload/rendering).
    pub fn merged(&self) -> BrickMesh {
        let mut out = BrickMesh::default();
        for m in self.bricks.values() {
            let base = out.vertices.len() as u32;
            out.ipos.extend_from_slice(&m.ipos);
            out.vertices.extend_from_slice(&m.vertices);
            out.dirs.extend_from_slice(&m.dirs);
            out.indices.extend(m.indices.iter().map(|i| i + base));
        }
        out
    }
}

/// All bricks at each LOD whose region the layout assigns to that LOD, inside `region`.
pub fn bricks_in(
    grid: &BrickGrid,
    layout: &dyn LodLayout,
    region: ([i64; 3], [i64; 3]),
    max_lod: u8,
) -> Vec<BrickCoord> {
    let mut todo = Vec::new();
    for lod in 0..=max_lod {
        let ext = grid.brick_size() as i64 * grid.step(lod);
        let lo = region.0.map(|v| floor_div(v, ext));
        let hi = region.1.map(|v| floor_div(v - 1, ext));
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    let b = BrickCoord { x, y, z, lod };
                    let o = grid.brick_origin(b);
                    if layout.lod_at([o[0] + ext / 2, o[1] + ext / 2, o[2] + ext / 2]) == lod {
                        todo.push(b);
                    }
                }
            }
        }
    }
    todo
}

/// Sample + mesh every brick of `region` (LOD0 lattice, half-open) at the LOD the layout
/// assigns, keeping only bricks with triangles (sparse). Parallel; deterministic (H7).
pub fn build_world(
    field: &dyn DensityField,
    cfg: &VoxelConfig,
    layout: &dyn LodLayout,
    region: ([i64; 3], [i64; 3]),
    max_lod: u8,
) -> Result<VoxelWorld, VoxelError> {
    build_world_with(field, cfg, layout, region, max_lod, false)
}

/// [`build_world`] using [`sample_brick_filtered`] (LOD-aware band-limited sampling).
pub fn build_world_filtered(
    field: &dyn DensityField,
    cfg: &VoxelConfig,
    layout: &dyn LodLayout,
    region: ([i64; 3], [i64; 3]),
    max_lod: u8,
) -> Result<VoxelWorld, VoxelError> {
    build_world_with(field, cfg, layout, region, max_lod, true)
}

fn build_world_with(
    field: &dyn DensityField,
    cfg: &VoxelConfig,
    layout: &dyn LodLayout,
    region: ([i64; 3], [i64; 3]),
    max_lod: u8,
    filtered: bool,
) -> Result<VoxelWorld, VoxelError> {
    let todo = bricks_in(&cfg.grid, layout, region, max_lod);
    let results: Vec<Result<Option<MeshedBrick>, VoxelError>> = todo
        .par_iter()
        .map(|&b| {
            let (mask, finer) = plan_brick(&cfg.grid, layout, b)?;
            let smp =
                sample_brick_with(field, cfg, if filtered { Filter::Level(layout) } else { Filter::None }, b, mask)?;
            if !smp.has_sign_change() {
                return Ok(None);
            }
            let bytes = smp.bytes();
            let m = mesh_brick(cfg, &smp, mask, finer);
            Ok((!m.indices.is_empty()).then_some((b, m, bytes)))
        })
        .collect();
    let mut w = VoxelWorld { sampled: todo.len(), ..Default::default() };
    for r in results {
        if let Some((b, m, bytes)) = r? {
            w.sample_bytes += bytes;
            w.bricks.insert(b, m);
        }
    }
    Ok(w)
}

/// (coord, mesh, sample bytes) of one occupied brick.
type MeshedBrick = (BrickCoord, BrickMesh, usize);

/// Exact topology / geometry report over meshes welded by integer position (H1, H4).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Topology {
    pub triangles: usize,
    pub welded_vertices: usize,
    /// Edges used by exactly one triangle (holes / cracks, or the region boundary).
    pub boundary_edges: usize,
    /// Edges used by more than two triangles.
    pub non_manifold_edges: usize,
    /// Edges traversed in the same direction by both triangles (inconsistent winding).
    pub orientation_conflicts: usize,
    /// Zero-area triangles (exact integer cross product == 0), including repeated vertices.
    pub degenerate: usize,
    /// Exact flip test on *axis-determined* triangles: all three vertices on parallel lattice edges
    /// with the same inside->outside direction (identical `dirs`). Their projection orthogonal to
    /// that axis is three corners of a cell square (Φ moves each by <= s/4 per axis), so the exact
    /// i128 normal component along the axis is non-zero and its sign is fixed by the tables alone,
    /// independent of the edge parameters. Counted when that component is <= 0. Combined with
    /// `orientation_conflicts == 0` (orientation propagates across every shared edge) this is the
    /// "no flipped triangles" criterion; `unreferenced_components` must be 0 for full coverage.
    pub flipped: usize,
    pub axis_reference_triangles: usize,
    /// Components without an axis-determined triangle (e.g. the 8-triangle octahedra around an
    /// isolated sample, whose corner triangles have a t-independent positive proxy).
    pub unreferenced_components: usize,
    /// Unreferenced components not certified by the exact proxy, i.e. some triangle disagrees
    /// (dot < 0) or none agrees (dot > 0). An inverted component turns every agreeing triangle into
    /// a disagreeing one, so "none < 0 and some > 0" certifies orientation. Must be 0.
    pub unverified_components: usize,
    pub closed_components: usize,
    pub open_components: usize,
    /// Closed components with negative exact signed volume: cavities (air enclosed by solid) when
    /// correctly oriented. Informational.
    pub negative_volume_components: usize,
    /// Per-triangle proxy: exact i128 dot of the normal with the summed inside->outside directions of
    /// its three crossing edges is < 0. Exact and meaningful for regular cells (asserted == 0 for
    /// uniform layouts). Transition triangles near corner ties can legitimately have a normal
    /// nearly perpendicular to all three (in-plane) crossing edges, so it is informational there.
    pub edge_dir_disagreements: usize,
    /// Proxy dot == 0 (transition caps lying in the full-resolution face plane).
    pub tangential: usize,
    pub non_finite: usize,
}

impl Topology {
    pub fn is_closed(&self) -> bool {
        self.boundary_edges == 0 && self.non_manifold_edges == 0 && self.orientation_conflicts == 0 && self.is_clean()
    }
    /// No degenerate, flipped or non-finite output (allows open boundaries at region cuts).
    pub fn is_clean(&self) -> bool {
        self.degenerate == 0 && self.flipped == 0 && self.unverified_components == 0 && self.non_finite == 0
    }
}

/// Exact i128 normal of an integer triangle.
pub fn tri_normal(a: [i64; 3], b: [i64; 3], c: [i64; 3]) -> [i128; 3] {
    let e1 = [0, 1, 2].map(|i| (b[i] - a[i]) as i128);
    let e2 = [0, 1, 2].map(|i| (c[i] - a[i]) as i128);
    [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]]
}

pub fn topology<'a>(meshes: impl IntoIterator<Item = &'a BrickMesh>) -> Topology {
    let mut ids: HashMap<[i64; 3], u32> = HashMap::new();
    let mut pos: Vec<[i64; 3]> = Vec::new();
    let mut t = Topology::default();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut reference: Vec<bool> = Vec::new();
    let mut proxy_pos: Vec<i128> = Vec::new();
    for m in meshes {
        t.non_finite += m.vertices.iter().filter(|v| !v.pos.iter().chain(&v.normal).all(|c| c.is_finite())).count();
        let w: Vec<u32> = m
            .ipos
            .iter()
            .map(|p| {
                let n = ids.len() as u32;
                *ids.entry(*p).or_insert_with(|| {
                    pos.push(*p);
                    n
                })
            })
            .collect();
        for tri in m.indices.as_chunks::<3>().0 {
            let [ia, ib, ic] = tri.map(|i| i as usize);
            t.triangles += 1;
            let nrm = tri_normal(m.ipos[ia], m.ipos[ib], m.ipos[ic]);
            if nrm == [0; 3] {
                t.degenerate += 1;
                continue;
            }
            let mut r = [0i128; 3];
            for &v in &[ia, ib, ic] {
                let d = m.dirs[v];
                r[(d & 3) as usize] += if d & 4 != 0 { 1 } else { -1 };
            }
            let proxy = (nrm[0] * r[0] + nrm[1] * r[1] + nrm[2] * r[2]).signum();
            match proxy {
                -1 => t.edge_dir_disagreements += 1,
                0 => t.tangential += 1,
                _ => {}
            }
            proxy_pos.push(proxy);
            let referenced = m.dirs[ia] == m.dirs[ib] && m.dirs[ib] == m.dirs[ic];
            if referenced {
                let d = m.dirs[ia];
                let sgn = if d & 4 != 0 { 1 } else { -1 };
                t.axis_reference_triangles += 1;
                if nrm[(d & 3) as usize] * sgn <= 0 {
                    t.flipped += 1;
                }
            }
            reference.push(referenced);
            tris.push([w[ia], w[ib], w[ic]]);
        }
    }
    t.welded_vertices = ids.len();
    let mut directed: HashMap<(u32, u32), u32> = HashMap::new();
    for &[a, b, c] in &tris {
        for e in [(a, b), (b, c), (c, a)] {
            *directed.entry(e).or_default() += 1;
        }
    }
    let mut seen = std::collections::HashSet::new();
    for (&(a, b), &n) in &directed {
        if !seen.insert((a.min(b), a.max(b))) {
            continue;
        }
        let r = directed.get(&(b, a)).copied().unwrap_or(0);
        match (n, r) {
            (1, 1) => {}
            (1, 0) | (0, 1) => t.boundary_edges += 1,
            (2, 0) | (0, 2) => t.orientation_conflicts += 1,
            _ => t.non_manifold_edges += 1,
        }
    }
    // Connected components over shared edges (union-find), exact signed volume of closed ones.
    let mut parent: Vec<u32> = (0..tris.len() as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    let mut edge_owner: HashMap<(u32, u32), u32> = HashMap::new();
    for (ti, &[a, b, c]) in tris.iter().enumerate() {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            let k = (u.min(v), u.max(v));
            if let Some(&o) = edge_owner.get(&k) {
                let (x, y) = (find(&mut parent, o), find(&mut parent, ti as u32));
                parent[x as usize] = y;
            } else {
                edge_owner.insert(k, ti as u32);
            }
        }
    }
    // (signed volume x6, open, has axis reference, any proxy < 0, any proxy > 0)
    let mut comp: HashMap<u32, (i128, bool, bool, bool, bool)> = HashMap::new();
    for (ti, &[a, b, c]) in tris.iter().enumerate() {
        let root = find(&mut parent, ti as u32);
        let (pa, pb, pc) = (pos[a as usize], pos[b as usize], pos[c as usize]);
        let cr = [
            pb[1] as i128 * pc[2] as i128 - pb[2] as i128 * pc[1] as i128,
            pb[2] as i128 * pc[0] as i128 - pb[0] as i128 * pc[2] as i128,
            pb[0] as i128 * pc[1] as i128 - pb[1] as i128 * pc[0] as i128,
        ];
        let vol6 = pa[0] as i128 * cr[0] + pa[1] as i128 * cr[1] + pa[2] as i128 * cr[2];
        let open = [(a, b), (b, c), (c, a)]
            .iter()
            .any(|&(u, v)| directed.get(&(v, u)).copied().unwrap_or(0) != 1 || directed[&(u, v)] != 1);
        let e = comp.entry(root).or_insert((0, false, false, false, false));
        e.0 += vol6;
        e.1 |= open;
        e.2 |= reference[ti];
        e.3 |= proxy_pos[ti] < 0;
        e.4 |= proxy_pos[ti] > 0;
    }
    for (vol, open, referenced, any_neg, any_pos) in comp.values() {
        t.unreferenced_components += !referenced as usize;
        t.unverified_components += (!referenced && (*any_neg || !*any_pos)) as usize;
        if *open {
            t.open_components += 1;
        } else {
            t.closed_components += 1;
            t.negative_volume_components += (*vol < 0) as usize;
        }
    }
    t
}

/// FNV-1a over brick coords, integer positions and indices (determinism checks).
pub fn world_hash(w: &VoxelWorld) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
        }
    };
    for (c, m) in &w.bricks {
        eat(&c.x.to_le_bytes());
        eat(&c.y.to_le_bytes());
        eat(&c.z.to_le_bytes());
        eat(&[c.lod]);
        eat(bytemuck::cast_slice(&m.ipos));
        eat(bytemuck::cast_slice(&m.indices));
    }
    h
}

/// What [`remove_islands`] did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct IslandStats {
    pub components: usize,
    pub removed_components: usize,
    pub removed_triangles: usize,
}

/// Remove disconnected islands from a welded mesh (connectivity over shared integer edges):
/// every closed component with positive exact signed volume (a solid lump floating in air),
/// every closed cavity that is not the largest one (sealed air bubbles, never visible), and
/// every component with fewer than `min_tris` triangles. Open components (cut by the meshed
/// region) and the main surface are kept. Vertices are compacted; order is deterministic.
pub fn remove_islands(m: &mut BrickMesh, min_tris: usize) -> IslandStats {
    let mut weld: HashMap<[i64; 3], u32> = HashMap::new();
    let wid: Vec<u32> = m
        .ipos
        .iter()
        .map(|p| {
            let n = weld.len() as u32;
            *weld.entry(*p).or_insert(n)
        })
        .collect();
    let tris: Vec<[u32; 3]> = m
        .indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|t| [wid[t[0] as usize], wid[t[1] as usize], wid[t[2] as usize]])
        .collect();
    let mut parent: Vec<u32> = (0..tris.len() as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    let mut owner: HashMap<(u32, u32), u32> = HashMap::new();
    let mut directed: HashMap<(u32, u32), u32> = HashMap::new();
    for (ti, &[a, b, c]) in tris.iter().enumerate() {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            *directed.entry((u, v)).or_default() += 1;
            let k = (u.min(v), u.max(v));
            if let Some(&o) = owner.get(&k) {
                let (x, y) = (find(&mut parent, o), find(&mut parent, ti as u32));
                parent[x as usize] = y;
            } else {
                owner.insert(k, ti as u32);
            }
        }
    }
    // root -> (tris, signed volume x6, open)
    let mut comp: HashMap<u32, (usize, i128, bool)> = HashMap::new();
    let roots: Vec<u32> = (0..tris.len() as u32).map(|t| find(&mut parent, t)).collect();
    for (ti, &[a, b, c]) in tris.iter().enumerate() {
        let (pa, pb, pc) = (
            m.ipos[m.indices[3 * ti] as usize],
            m.ipos[m.indices[3 * ti + 1] as usize],
            m.ipos[m.indices[3 * ti + 2] as usize],
        );
        let cr = [
            pb[1] as i128 * pc[2] as i128 - pb[2] as i128 * pc[1] as i128,
            pb[2] as i128 * pc[0] as i128 - pb[0] as i128 * pc[2] as i128,
            pb[0] as i128 * pc[1] as i128 - pb[1] as i128 * pc[0] as i128,
        ];
        let open = [(a, b), (b, c), (c, a)].iter().any(|&(u, v)| directed.get(&(v, u)).copied().unwrap_or(0) != 1);
        let e = comp.entry(roots[ti]).or_insert((0, 0, false));
        e.0 += 1;
        e.1 += pa[0] as i128 * cr[0] + pa[1] as i128 * cr[1] + pa[2] as i128 * cr[2];
        e.2 |= open;
    }
    let biggest_cavity = comp.iter().filter(|(_, c)| !c.2 && c.1 < 0).min_by_key(|(r, c)| (c.1, **r)).map(|(r, _)| *r);
    let drop: std::collections::HashSet<u32> = comp
        .iter()
        .filter(|(r, c)| c.0 < min_tris || (!c.2 && c.1 > 0) || (!c.2 && c.1 < 0 && Some(**r) != biggest_cavity))
        .map(|(r, _)| *r)
        .collect();
    let mut st = IslandStats { components: comp.len(), removed_components: drop.len(), removed_triangles: 0 };
    let mut remap = vec![u32::MAX; m.ipos.len()];
    let mut out = BrickMesh::default();
    for (ti, t) in m.indices.as_chunks::<3>().0.iter().enumerate() {
        if drop.contains(&roots[ti]) {
            st.removed_triangles += 1;
            continue;
        }
        for &i in t {
            if remap[i as usize] == u32::MAX {
                remap[i as usize] = out.ipos.len() as u32;
                out.ipos.push(m.ipos[i as usize]);
                out.vertices.push(m.vertices[i as usize]);
                if let Some(&d) = m.dirs.get(i as usize) {
                    out.dirs.push(d);
                }
            }
            out.indices.push(remap[i as usize]);
        }
    }
    *m = out;
    st
}
