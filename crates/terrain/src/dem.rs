//! Real-world heightfields (DEM grids, e.g. `geo-fetch terrain-grid` output) as chunked,
//! LOD'd, skirted meshes.
//!
//! # Frame
//! A [`DemGrid`] is a regular lattice in a local East-North-Up tangent frame: sample `(col, row)`
//! sits at `(x0 + col·spacing, y0 + row·spacing)` metres east/north of the frame origin, row 0 =
//! south. Meshes use this crate's Y-up convention: **`x = east`, `y = height`, `z = −north`**
//! (right-handed, so +Z points south). Positions are f32 metres relative to the grid origin, which
//! is fine for grids of a few tens of km; place the origin in a larger frame (ECEF / planet) with
//! an f64 transform outside this crate.
//!
//! # Chunks and LOD
//! Every chunk has the same topology as the noise chunks: [`CHUNK_SAMPLES`]² vertices (64×64
//! quads). A chunk at LOD `l` samples the grid every `2^l` samples, so it covers `64·2^l` grid
//! cells per side and its vertices are a subset of the LOD `l−1` vertices (nested, CDLOD-ready).
//! Chunk `(x, y)` at LOD `l` starts at grid sample `(x·64·2^l, y·64·2^l)`; `y` counts northward.
//! Vertices past the grid's last row/column are clamped onto it (zero-area quads), so edge chunks
//! keep the fixed vertex count and can use one shared index buffer ([`dem_chunk_indices`]).
//! LOD [`DemTerrain::root_lod`] is a single chunk covering the whole grid.
//!
//! Each chunk carries its **geometric error**: the max |full-resolution height − chunk surface|
//! over all grid samples it covers, using the exact triangulation of the shared index buffer.
//! [`DemTerrain::select`] refines a quadtree until the projected error is below a pixel budget,
//! then enforces 2:1 balance (edge neighbours differ by at most one LOD).
//!
//! # Skirts
//! [`DEM_SKIRT_VERTS`] extra vertices duplicate the four chunk edges, dropped by
//! [`DemTerrain::skirt_depth`] (max of the chunk's and its parent's error, plus a margin). With
//! 2:1 balance the vertical gap at an LOD boundary is at most the coarser chunk's error, so the
//! curtains close every crack. Skirts are the fallback in starter.md's seam decision; shader edge
//! snapping / morphing can replace them later without changing the vertex layout.

use crate::{CHUNK_SAMPLES, TerrainVertex};

/// Quads per chunk edge.
pub const DEM_CHUNK_QUADS: usize = CHUNK_SAMPLES - 1;
/// Skirt vertices per chunk (one curtain row under each of the four edges).
pub const DEM_SKIRT_VERTS: usize = 4 * CHUNK_SAMPLES;
/// Vertices per chunk: grid first (row-major, `j` northward), then the four skirt rows.
pub const DEM_CHUNK_VERTS: usize = CHUNK_SAMPLES * CHUNK_SAMPLES + DEM_SKIRT_VERTS;
/// Indices of the surface part of [`dem_chunk_indices`] (the rest are skirts).
pub const DEM_SURFACE_INDICES: usize = DEM_CHUNK_QUADS * DEM_CHUNK_QUADS * 6;
/// Extra skirt depth beyond the error bound, metres.
pub const SKIRT_MARGIN_M: f32 = 1.0;

#[derive(Debug, PartialEq)]
pub enum DemError {
    /// Fewer than 2×2 samples, `heights.len() != width·height`, or a non-positive spacing.
    BadShape { width: usize, height: usize, len: usize },
}

impl std::fmt::Display for DemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DemError::BadShape { width, height, len } => {
                write!(f, "bad DEM shape: {width}x{height} with {len} heights (need >= 2x2, matching, spacing > 0)")
            }
        }
    }
}

impl std::error::Error for DemError {}

/// A height lattice in a local ENU frame (see the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct DemGrid {
    pub width: usize,
    pub height: usize,
    pub spacing_m: f32,
    /// East of column 0, metres from the frame origin.
    pub x0_m: f32,
    /// North of row 0 (the southern row), metres from the frame origin.
    pub y0_m: f32,
    /// `width·height` heights in metres, row-major, row 0 = south. Non-finite values become 0.
    pub heights: Vec<f32>,
}

impl DemGrid {
    pub fn new(
        width: usize,
        height: usize,
        spacing_m: f32,
        x0_m: f32,
        y0_m: f32,
        mut heights: Vec<f32>,
    ) -> Result<Self, DemError> {
        if width < 2 || height < 2 || heights.len() != width * height || spacing_m.is_nan() || spacing_m <= 0.0 {
            return Err(DemError::BadShape { width, height, len: heights.len() });
        }
        for h in heights.iter_mut().filter(|h| !h.is_finite()) {
            *h = 0.0;
        }
        Ok(Self { width, height, spacing_m, x0_m, y0_m, heights })
    }

    #[inline]
    pub fn get(&self, col: usize, row: usize) -> f32 {
        self.heights[row * self.width + col]
    }

    /// Mesh-space position of a sample (`x = east`, `y = h`, `z = −north`).
    #[inline]
    pub fn position(&self, col: usize, row: usize) -> [f32; 3] {
        [self.x0_m + col as f32 * self.spacing_m, self.get(col, row), -(self.y0_m + row as f32 * self.spacing_m)]
    }

    /// `(min, max)` height.
    pub fn height_range(&self) -> (f32, f32) {
        self.heights.iter().fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(v), h.max(v)))
    }
}

/// A chunk address: LOD and chunk column/row (`y` northward) at that LOD.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DemChunkId {
    pub lod: u8,
    pub x: u32,
    pub y: u32,
}

impl DemChunkId {
    pub fn parent(self) -> Self {
        Self { lod: self.lod + 1, x: self.x / 2, y: self.y / 2 }
    }

    /// The four children (some may lie outside the grid; see [`DemTerrain::exists`]).
    pub fn children(self) -> [Self; 4] {
        assert!(self.lod > 0, "LOD0 chunks have no children");
        let (l, x, y) = (self.lod - 1, self.x * 2, self.y * 2);
        [
            Self { lod: l, x, y },
            Self { lod: l, x: x + 1, y },
            Self { lod: l, x, y: y + 1 },
            Self { lod: l, x: x + 1, y: y + 1 },
        ]
    }

    /// Grid stride between this chunk's vertices.
    pub fn stride(self) -> usize {
        1 << self.lod
    }

    /// First grid sample `(col, row)` of the chunk.
    pub fn first_sample(self) -> (usize, usize) {
        let span = DEM_CHUNK_QUADS * self.stride();
        (self.x as usize * span, self.y as usize * span)
    }
}

/// Mesh-space axis-aligned box.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    /// Euclidean distance from `p` to the box (0 inside).
    pub fn distance(&self, p: [f32; 3]) -> f32 {
        let d2: f32 = (0..3)
            .map(|k| {
                let d = (self.min[k] - p[k]).max(0.0).max(p[k] - self.max[k]);
                d * d
            })
            .sum();
        d2.sqrt()
    }
}

/// Camera inputs for [`DemTerrain::select`].
#[derive(Copy, Clone, Debug)]
pub struct LodView {
    /// Eye in mesh space (`x = east`, `y = up`, `z = −north`).
    pub eye: [f32; 3],
    /// Vertical field of view, radians.
    pub fov_y: f32,
    pub viewport_height_px: f32,
    /// Refine while a chunk's projected geometric error exceeds this many pixels.
    pub max_error_px: f32,
}

impl LodView {
    /// Pixels per metre at 1 m distance.
    pub fn pixels_per_radian(&self) -> f32 {
        self.viewport_height_px / (2.0 * (self.fov_y * 0.5).tan())
    }
}

struct Level {
    chunks_x: u32,
    chunks_y: u32,
    /// Geometric error per chunk, row-major by chunk.
    error: Vec<f32>,
    /// Height range per chunk (surface only).
    range: Vec<(f32, f32)>,
}

/// A [`DemGrid`] with per-chunk error and bounds at every LOD.
pub struct DemTerrain {
    grid: DemGrid,
    levels: Vec<Level>,
}

/// Shared index buffer for every DEM chunk: surface triangles, then skirt triangles.
/// Front faces are counter-clockwise with geometric normals +Y (same as [`crate::grid_indices`]);
/// skirt triangles face outward.
pub fn dem_chunk_indices() -> Vec<u32> {
    let n = CHUNK_SAMPLES as u32;
    let mut idx = Vec::with_capacity(DEM_SURFACE_INDICES + 4 * DEM_CHUNK_QUADS * 6);
    // j is northward, i.e. toward −Z, so this is the mirror of `grid_indices`. Diagonal v10–v01.
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let v00 = j * n + i;
            let (v10, v01, v11) = (v00 + 1, v00 + n, v00 + n + 1);
            idx.extend_from_slice(&[v00, v10, v01, v10, v11, v01]);
        }
    }
    for (e, edge) in edge_rings().iter().enumerate() {
        let base = n * n + e as u32 * n;
        for k in 0..n - 1 {
            let (a, b) = (edge[k as usize], edge[k as usize + 1]);
            let (sa, sb) = (base + k, base + k + 1);
            // edge rings run counter-clockwise seen from above, so (a, sa, b) faces outward
            idx.extend_from_slice(&[a, sa, b, b, sa, sb]);
        }
    }
    idx
}

/// Surface vertex indices along the four edges, each running counter-clockwise seen from +Y
/// (south edge west→east, east edge south→north, north edge east→west, west edge north→south).
fn edge_rings() -> [[u32; CHUNK_SAMPLES]; 4] {
    let n = CHUNK_SAMPLES;
    let at = |i: usize, j: usize| (j * n + i) as u32;
    [
        std::array::from_fn(|k| at(k, 0)),
        std::array::from_fn(|k| at(n - 1, k)),
        std::array::from_fn(|k| at(n - 1 - k, n - 1)),
        std::array::from_fn(|k| at(0, n - 1 - k)),
    ]
}

impl DemTerrain {
    pub fn new(grid: DemGrid) -> Self {
        let span = grid.width.max(grid.height) - 1;
        let mut root = 0u8;
        while DEM_CHUNK_QUADS << root < span {
            root += 1;
        }
        let levels = (0..=root).map(|l| Self::build_level(&grid, l)).collect();
        Self { grid, levels }
    }

    pub fn grid(&self) -> &DemGrid {
        &self.grid
    }

    /// The LOD at which one chunk covers the whole grid.
    pub fn root_lod(&self) -> u8 {
        (self.levels.len() - 1) as u8
    }

    /// `(chunks_x, chunks_y)` at `lod`.
    pub fn chunks_at(&self, lod: u8) -> (u32, u32) {
        let l = &self.levels[lod as usize];
        (l.chunks_x, l.chunks_y)
    }

    pub fn exists(&self, id: DemChunkId) -> bool {
        (id.lod as usize) < self.levels.len() && {
            let l = &self.levels[id.lod as usize];
            id.x < l.chunks_x && id.y < l.chunks_y
        }
    }

    fn slot(&self, id: DemChunkId) -> usize {
        debug_assert!(self.exists(id), "{id:?} outside the grid");
        let l = &self.levels[id.lod as usize];
        (id.y * l.chunks_x + id.x) as usize
    }

    /// Max |full-res height − chunk surface| over the chunk's samples, metres (0 at LOD0).
    pub fn geometric_error(&self, id: DemChunkId) -> f32 {
        self.levels[id.lod as usize].error[self.slot(id)]
    }

    /// Skirt drop: covers the gap to a neighbour one LOD coarser (2:1 balance).
    pub fn skirt_depth(&self, id: DemChunkId) -> f32 {
        let parent = if id.lod < self.root_lod() { self.geometric_error(id.parent()) } else { 0.0 };
        self.geometric_error(id).max(parent) + SKIRT_MARGIN_M
    }

    /// Surface bounds of the chunk (skirts excluded).
    pub fn chunk_aabb(&self, id: DemChunkId) -> Aabb {
        let g = &self.grid;
        let (c0, r0) = id.first_sample();
        let span = DEM_CHUNK_QUADS * id.stride();
        let (c1, r1) = ((c0 + span).min(g.width - 1), (r0 + span).min(g.height - 1));
        let (lo, hi) = self.levels[id.lod as usize].range[self.slot(id)];
        let (p0, p1) = (g.position(c0, r0), g.position(c1, r1));
        Aabb { min: [p0[0], lo, p1[2]], max: [p1[0], hi, p0[2]] }
    }

    /// Grid sample under chunk vertex `(i, j)` (clamped to the grid).
    #[inline]
    fn sample_of(&self, id: DemChunkId, i: usize, j: usize) -> (usize, usize) {
        let (c0, r0) = id.first_sample();
        let s = id.stride();
        ((c0 + i * s).min(self.grid.width - 1), (r0 + j * s).min(self.grid.height - 1))
    }

    /// Write the chunk's [`DEM_CHUNK_VERTS`] vertices into `out` (for arenas / staging).
    pub fn mesh_chunk_into(&self, id: DemChunkId, out: &mut [TerrainVertex]) {
        assert_eq!(out.len(), DEM_CHUNK_VERTS);
        let g = &self.grid;
        let n = CHUNK_SAMPLES;
        let s = id.stride();
        for j in 0..n {
            for i in 0..n {
                let (c, r) = self.sample_of(id, i, j);
                // central differences at this LOD's spacing (clamped at the grid border)
                let (cl, cr) = (c.saturating_sub(s), (c + s).min(g.width - 1));
                let (rs, rn) = (r.saturating_sub(s), (r + s).min(g.height - 1));
                let dhde = (g.get(cr, r) - g.get(cl, r)) / ((cr - cl).max(1) as f32 * g.spacing_m);
                let dhdn = (g.get(c, rn) - g.get(c, rs)) / ((rn - rs).max(1) as f32 * g.spacing_m);
                // surface y = h(x, −z): normal ∝ (−∂h/∂x, 1, −∂h/∂z) with ∂h/∂z = −∂h/∂n
                let inv = 1.0 / (dhde * dhde + 1.0 + dhdn * dhdn).sqrt();
                out[j * n + i] = TerrainVertex { pos: g.position(c, r), normal: [-dhde * inv, inv, dhdn * inv] };
            }
        }
        let depth = self.skirt_depth(id);
        for (e, ring) in edge_rings().iter().enumerate() {
            for (k, &v) in ring.iter().enumerate() {
                let mut sv = out[v as usize];
                sv.pos[1] -= depth;
                out[n * n + e * n + k] = sv;
            }
        }
    }

    /// Mesh one chunk into a new buffer.
    pub fn mesh_chunk(&self, id: DemChunkId) -> Vec<TerrainVertex> {
        let mut v = vec![TerrainVertex { pos: [0.0; 3], normal: [0.0; 3] }; DEM_CHUNK_VERTS];
        self.mesh_chunk_into(id, &mut v);
        v
    }

    /// Quadtree LOD selection from the root, then 2:1 balance. Returns leaf chunks sorted by id;
    /// together they tile the grid exactly once.
    pub fn select(&self, view: &LodView) -> Vec<DemChunkId> {
        let k = view.pixels_per_radian();
        let mut out = Vec::new();
        let mut stack = vec![DemChunkId { lod: self.root_lod(), x: 0, y: 0 }];
        while let Some(id) = stack.pop() {
            let d = self.chunk_aabb(id).distance(view.eye).max(1.0);
            if id.lod > 0 && self.geometric_error(id) * k / d > view.max_error_px {
                stack.extend(id.children().into_iter().filter(|c| self.exists(*c)));
            } else {
                out.push(id);
            }
        }
        self.balance(out)
    }

    /// Split leaves until every edge neighbour is at most one LOD coarser.
    pub fn balance(&self, leaves: Vec<DemChunkId>) -> Vec<DemChunkId> {
        use std::collections::BTreeSet;
        let mut set: BTreeSet<DemChunkId> = leaves.into_iter().collect();
        let leaf_at = |set: &BTreeSet<DemChunkId>, col: i64, row: i64| -> Option<DemChunkId> {
            if col < 0 || row < 0 || col >= self.grid.width as i64 - 1 || row >= self.grid.height as i64 - 1 {
                return None;
            }
            (0..=self.root_lod()).find_map(|lod| {
                let span = (DEM_CHUNK_QUADS << lod) as i64;
                let id = DemChunkId { lod, x: (col / span) as u32, y: (row / span) as u32 };
                set.contains(&id).then_some(id)
            })
        };
        loop {
            let mut split = None;
            'scan: for &id in &set {
                let (c0, r0) = id.first_sample();
                let span = (DEM_CHUNK_QUADS * id.stride()) as i64;
                let (c0, r0) = (c0 as i64, r0 as i64);
                // middle of the edges clipped to the grid
                let mc = (span.min(self.grid.width as i64 - 1 - c0) / 2).max(0);
                let mr = (span.min(self.grid.height as i64 - 1 - r0) / 2).max(0);
                // one probe cell just outside the middle of each edge; a neighbour coarser than
                // this leaf covers the whole edge, so the middle finds it
                for (c, r) in [(c0 - 1, r0 + mr), (c0 + span, r0 + mr), (c0 + mc, r0 - 1), (c0 + mc, r0 + span)] {
                    if let Some(nb) = leaf_at(&set, c, r)
                        && nb.lod > id.lod + 1
                    {
                        split = Some(nb);
                        break 'scan;
                    }
                }
            }
            match split {
                Some(nb) => {
                    set.remove(&nb);
                    set.extend(nb.children().into_iter().filter(|c| self.exists(*c)));
                }
                None => return set.into_iter().collect(),
            }
        }
    }

    fn build_level(g: &DemGrid, lod: u8) -> Level {
        let span = DEM_CHUNK_QUADS << lod;
        let chunks_x = (g.width - 1).div_ceil(span).max(1) as u32;
        let chunks_y = (g.height - 1).div_ceil(span).max(1) as u32;
        let mut error = vec![0.0f32; (chunks_x * chunks_y) as usize];
        let mut range = vec![(f32::MAX, f32::MIN); error.len()];
        let s = 1usize << lod;
        for cy in 0..chunks_y {
            for cx in 0..chunks_x {
                let slot = (cy * chunks_x + cx) as usize;
                let (c0, r0) = (cx as usize * span, cy as usize * span);
                let (mut err, mut lo, mut hi) = (0.0f32, f32::MAX, f32::MIN);
                for j in 0..DEM_CHUNK_QUADS {
                    let (ra, rb) = ((r0 + j * s).min(g.height - 1), (r0 + (j + 1) * s).min(g.height - 1));
                    for i in 0..DEM_CHUNK_QUADS {
                        let (ca, cb) = ((c0 + i * s).min(g.width - 1), (c0 + (i + 1) * s).min(g.width - 1));
                        let (h00, h10, h01, h11) = (g.get(ca, ra), g.get(cb, ra), g.get(ca, rb), g.get(cb, rb));
                        for h in [h00, h10, h01, h11] {
                            lo = lo.min(h);
                            hi = hi.max(h);
                        }
                        if lod == 0 || ca == cb || ra == rb {
                            continue;
                        }
                        let (du, dv) = ((cb - ca) as f32, (rb - ra) as f32);
                        for r in ra..=rb {
                            let fv = (r - ra) as f32 / dv;
                            for c in ca..=cb {
                                let fu = (c - ca) as f32 / du;
                                // exact triangulation of `dem_chunk_indices` (diagonal v10–v01)
                                let t = if fu + fv <= 1.0 {
                                    h00 + fu * (h10 - h00) + fv * (h01 - h00)
                                } else {
                                    h11 + (1.0 - fu) * (h01 - h11) + (1.0 - fv) * (h10 - h11)
                                };
                                err = err.max((g.get(c, r) - t).abs());
                            }
                        }
                    }
                }
                error[slot] = err;
                range[slot] = (lo, hi);
            }
        }
        Level { chunks_x, chunks_y, error, range }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hills(w: usize, h: usize, spacing: f32) -> DemGrid {
        let heights = (0..h)
            .flat_map(|r| {
                (0..w).map(move |c| {
                    let (x, y) = (c as f32 * 0.05, r as f32 * 0.037);
                    120.0 * x.sin() * y.cos() + 40.0 * (x * 3.1 + y * 1.7).sin() + 0.002 * (c * r) as f32
                })
            })
            .collect();
        DemGrid::new(w, h, spacing, -100.0, -50.0, heights).unwrap()
    }

    fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    }
    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[test]
    fn rejects_bad_shapes() {
        assert!(DemGrid::new(1, 5, 1.0, 0.0, 0.0, vec![0.0; 5]).is_err());
        assert!(DemGrid::new(3, 3, 1.0, 0.0, 0.0, vec![0.0; 8]).is_err());
        assert!(DemGrid::new(3, 3, 0.0, 0.0, 0.0, vec![0.0; 9]).is_err());
        let g = DemGrid::new(2, 2, 1.0, 0.0, 0.0, vec![f32::NAN, 1.0, 2.0, 3.0]).unwrap();
        assert_eq!(g.get(0, 0), 0.0);
    }

    #[test]
    fn levels_and_root() {
        let t = DemTerrain::new(hills(593, 718, 31.0)); // the Adelaide z12 shape
        assert_eq!(t.root_lod(), 4);
        assert_eq!(t.chunks_at(0), (10, 12));
        assert_eq!(t.chunks_at(4), (1, 1));
        for l in 0..=t.root_lod() {
            let (cx, cy) = t.chunks_at(l);
            for y in 0..cy {
                for x in 0..cx {
                    let e = t.geometric_error(DemChunkId { lod: l, x, y });
                    if l == 0 {
                        assert_eq!(e, 0.0);
                    } else {
                        assert!(e > 0.0 && e < 400.0, "lod {l}: {e}");
                    }
                }
            }
        }
    }

    /// Surface faces +Y with CCW winding, skirts face outward.
    #[test]
    fn winding() {
        let flat = DemGrid::new(200, 150, 10.0, 0.0, 0.0, vec![5.0; 200 * 150]).unwrap();
        let t = DemTerrain::new(flat);
        let id = DemChunkId { lod: 0, x: 0, y: 0 };
        let v = t.mesh_chunk(id);
        let idx = dem_chunk_indices();
        assert_eq!(idx.len(), DEM_SURFACE_INDICES + 4 * DEM_CHUNK_QUADS * 6);
        let centre = [320.0, 5.0, -320.0];
        for (k, tri) in idx.chunks(3).enumerate() {
            let [a, b, c] = [0, 1, 2].map(|q| v[tri[q] as usize].pos);
            let nrm = cross(sub(b, a), sub(c, a));
            if k * 3 < DEM_SURFACE_INDICES {
                assert!(nrm[1] > 0.0 && nrm[0].abs() < 1e-3 && nrm[2].abs() < 1e-3, "surface tri {k}: {nrm:?}");
            } else {
                let out = sub(a, centre);
                assert!(dot(nrm, [out[0], 0.0, out[2]]) > 0.0, "skirt tri {k} faces inward: {nrm:?}");
                assert!(nrm[1].abs() < 1e-3);
            }
        }
        for p in &v {
            assert_eq!(p.normal, [0.0, 1.0, 0.0]);
        }
    }

    /// Positions follow `x = east`, `z = −north`; slopes give consistent normals.
    #[test]
    fn frame_and_normals() {
        // h rises 0.5 m per metre northward
        let (w, h) = (100, 100);
        let g = DemGrid::new(w, h, 2.0, 10.0, 20.0, (0..w * h).map(|k| (k / w) as f32).collect()).unwrap();
        assert_eq!(g.position(3, 4), [16.0, 4.0, -28.0]);
        let t = DemTerrain::new(g);
        let v = t.mesh_chunk(DemChunkId { lod: 0, x: 0, y: 0 });
        let n = v[10 * CHUNK_SAMPLES + 10].normal;
        // uphill is −Z, so the normal leans toward +Z (south)
        let s = 0.5f32;
        let inv = 1.0 / (1.0 + s * s).sqrt();
        assert!((n[0]).abs() < 1e-6 && (n[1] - inv).abs() < 1e-6 && (n[2] - s * inv).abs() < 1e-6, "{n:?}");
    }

    #[test]
    fn same_lod_neighbours_share_edges() {
        let t = DemTerrain::new(hills(400, 300, 20.0));
        for lod in 0..2u8 {
            let a = t.mesh_chunk(DemChunkId { lod, x: 0, y: 0 });
            let b = t.mesh_chunk(DemChunkId { lod, x: 1, y: 0 });
            let c = t.mesh_chunk(DemChunkId { lod, x: 0, y: 1 });
            let n = CHUNK_SAMPLES;
            for k in 0..n {
                assert_eq!(a[k * n + n - 1], b[k * n], "lod {lod} east seam {k}");
                assert_eq!(a[(n - 1) * n + k], c[k], "lod {lod} north seam {k}");
            }
        }
    }

    /// The error bound holds against the actual coarse triangles, so skirts close LOD cracks.
    #[test]
    fn skirts_cover_one_level_cracks() {
        let t = DemTerrain::new(hills(400, 300, 20.0));
        let g = t.grid();
        // fine LOD0 chunk (1,0) next to coarse LOD1 chunk (1,0) on its east: shared edge at col 128
        let fine = DemChunkId { lod: 0, x: 1, y: 0 };
        let coarse = DemChunkId { lod: 1, x: 1, y: 0 };
        let col = 128;
        let fine_v = t.mesh_chunk(fine);
        let n = CHUNK_SAMPLES;
        let mut worst = 0.0f32;
        for j in 0..n {
            let r = j; // fine chunk rows 0..=64
            // coarse edge: straight segments between even rows
            let (ra, rb) = (r / 2 * 2, (r / 2 * 2 + 2).min(g.height - 1));
            let f = (r - ra) as f32 / (rb - ra) as f32;
            let line = g.get(col, ra) * (1.0 - f) + g.get(col, rb) * f;
            assert_eq!(fine_v[j * n + n - 1].pos[1], g.get(col, r));
            worst = worst.max((g.get(col, r) - line).abs());
        }
        assert!(worst > 0.0);
        assert!(worst <= t.geometric_error(coarse) + 1e-3);
        assert!(worst < t.skirt_depth(fine) && worst < t.skirt_depth(coarse));
    }

    fn assert_tiles_once(t: &DemTerrain, leaves: &[DemChunkId]) {
        let g = t.grid();
        let mut cover = vec![0u8; (g.width - 1) * (g.height - 1)];
        for id in leaves {
            let (c0, r0) = id.first_sample();
            let span = DEM_CHUNK_QUADS * id.stride();
            for r in r0..(r0 + span).min(g.height - 1) {
                for c in c0..(c0 + span).min(g.width - 1) {
                    cover[r * (g.width - 1) + c] += 1;
                }
            }
        }
        assert!(cover.iter().all(|&k| k == 1), "selection does not tile the grid exactly once");
    }

    #[test]
    fn select_refines_near_eye_and_balances() {
        let t = DemTerrain::new(hills(593, 718, 31.0));
        let view = LodView { eye: [5000.0, 400.0, -8000.0], fov_y: 1.0, viewport_height_px: 720.0, max_error_px: 2.0 };
        let leaves = t.select(&view);
        assert_tiles_once(&t, &leaves);
        let near = leaves
            .iter()
            .min_by(|a, b| t.chunk_aabb(**a).distance(view.eye).total_cmp(&t.chunk_aabb(**b).distance(view.eye)));
        assert_eq!(near.unwrap().lod, 0);
        assert!(leaves.iter().any(|l| l.lod >= 2), "far field should be coarse");
        // 2:1 balance: probe across every edge
        for &a in &leaves {
            for &b in &leaves {
                let (aa, bb) = (t.chunk_aabb(a), t.chunk_aabb(b));
                let touch_x = (aa.max[0] == bb.min[0] || bb.max[0] == aa.min[0])
                    && aa.min[2] < bb.max[2]
                    && bb.min[2] < aa.max[2];
                let touch_z = (aa.max[2] == bb.min[2] || bb.max[2] == aa.min[2])
                    && aa.min[0] < bb.max[0]
                    && bb.min[0] < aa.max[0];
                if touch_x || touch_z {
                    assert!(a.lod.abs_diff(b.lod) <= 1, "{a:?} next to {b:?}");
                }
            }
        }
        // a huge budget keeps the root
        let coarse = t.select(&LodView { max_error_px: 1e9, ..view });
        assert_eq!(coarse, vec![DemChunkId { lod: 4, x: 0, y: 0 }]);
    }
}
