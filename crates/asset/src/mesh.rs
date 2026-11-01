//! Draco-style brick-mesh codec (Draco bitstream spec v2.2, <https://google.github.io/draco/spec/>;
//! Rossignac, "Edgebreaker", IEEE TVCG 1999; Touma & Gotsman, "Triangle mesh compression", GI 1998).
//!
//! **Positions** are the brick-local sub-grid integers (`ipos - brick_origin * SUB`), the exact
//! grid the mesher works on, so the default (`qbits = 0`) is *lossless* and a vertex shared by
//! two bricks decodes to bit-identical `ipos` and therefore bit-identical f32 render positions.
//! With `qbits > 0` the *global* coordinate is rounded to a multiple of `2^qbits` (a pure
//! function of the global point, so seams stay bit-identical) and the error is `<= 2^(qbits-1)`.
//! Every vertex lies on a lattice edge, so each coordinate is split into lattice index and
//! in-cell fraction: the two off-axis coordinates have fraction 0 (cheap), the crossing-axis
//! coordinate carries the edge parameter. Lattice indices are predicted by the parallelogram
//! rule `a + b - o` across the traversal gate (Draco's MESH_PREDICTION_PARALLELOGRAM).
//!
//! **Connectivity** (`Traversal`): an Edgebreaker-like depth-first gate traversal. Each reached
//! triangle `(b, a, tip)` is one of 8 symbols `tip_is_new | left_child << 1 | right_child << 2`
//! (children are claimed when pushed, so every triangle is coded exactly once and handles /
//! non-manifold bricks need no special split symbols); an old tip is a back-reference
//! `next_id - 1 - id`, usually small. The decoder needs only a stack: no edge map, no hashing.
//! `Sequential` keeps the exact index / vertex order (Draco's sequential connectivity).
//!
//! **Normals**: octahedral (Cigolle et al., JCGT 2014) with `normal_bits` per component,
//! predicted from the area-weighted geometric normal of the decoded faces (attributes are
//! decoded after connectivity, as in Draco, so full adjacency is available), residual wrapped.
//!
//! All symbols go through one rANS stream with static per-file models ([`crate::rans`]).

use crate::rans::{CodecError, Decoder, SymLog};
use coords::{BrickCoord, BrickGrid};
use voxel::{BrickMesh, SUB, VoxelConfig, VoxelVertex};

pub const M_CONN: u8 = 0;
pub const M_START: u8 = 1;
pub const M_SFLAGS: u8 = 2;
pub const M_REF: u8 = 3;
pub const M_LAT_AX: u8 = 4;
pub const M_LAT_OFF: u8 = 5;
pub const M_FRAC_AX: u8 = 6;
pub const M_FRAC_OFF: u8 = 7;
pub const M_NU: u8 = 8;
pub const M_NV: u8 = 9;
pub const M_DIR: u8 = 10;
pub const M_SEQ: u8 = 11;
pub const M_CAND: u8 = 20;
pub const M_DSGN: u8 = 21;

/// Candidate-list size for old tips (index `MAX_CAND` = escape to an explicit back-reference).
const MAX_CAND: usize = 12;
const OPEN_CAP: usize = 7;
/// Connectivity symbol: open gate to an already-coded triangle (non-manifold input only).
const SKIP: u8 = 8;

/// Per-vertex open (unpaired) directed edges of the triangles decoded so far: the boundary of
/// the decoded region. A tip that closes against an already-decoded neighbour lies on such an
/// edge at `a` or `b`, so it is named by a small candidate index instead of a back-reference
/// (what Edgebreaker's L / R / E symbols resolve implicitly). Encoder and decoder run this
/// identically; lists are bounded, so a full list just costs an escape.
#[derive(Clone, Copy, Default)]
struct Small {
    n: u8,
    v: [u32; OPEN_CAP],
}

impl Small {
    fn pos(&self, x: u32) -> Option<usize> {
        self.v[..self.n as usize].iter().position(|&y| y == x)
    }
    fn remove(&mut self, i: usize) {
        self.n -= 1;
        self.v[i] = self.v[self.n as usize];
    }
    fn full(&self) -> bool {
        self.n as usize == OPEN_CAP
    }
    fn push(&mut self, x: u32) {
        self.v[self.n as usize] = x;
        self.n += 1;
    }
    fn items(&self) -> &[u32] {
        &self.v[..self.n as usize]
    }
}

struct OpenEdges {
    out: Vec<Small>,
    inn: Vec<Small>,
}

impl OpenEdges {
    fn new(nv: usize) -> Self {
        OpenEdges { out: vec![Small::default(); nv], inn: vec![Small::default(); nv] }
    }
    fn is_open(&self, a: u32, b: u32) -> bool {
        self.out[a as usize].pos(b).is_some()
    }
    fn add_tri(&mut self, t: [u32; 3]) {
        for k in 0..3 {
            let (p, q) = (t[k] as usize, t[(k + 1) % 3] as usize);
            // twin (q, p) open? then the pair closes
            if let (Some(i), Some(j)) = (self.out[q].pos(p as u32), self.inn[p].pos(q as u32)) {
                self.out[q].remove(i);
                self.inn[p].remove(j);
            } else if !self.out[p].full() && !self.inn[q].full() {
                self.out[p].push(q as u32);
                self.inn[q].push(p as u32);
            }
        }
    }
    /// Candidates for the tip of `(b, a, tip)`: both-sided first, then left (w, a), then right (b, w).
    fn candidates(&self, a: u32, b: u32, out: &mut [u32; MAX_CAND]) -> usize {
        let (ia, ob) = (self.inn[a as usize].items(), self.out[b as usize].items());
        let mut n = 0;
        let mut add = |w: u32, out: &mut [u32; MAX_CAND]| {
            if n < MAX_CAND && w != a && w != b && !out[..n].contains(&w) {
                out[n] = w;
                n += 1;
            }
        };
        for &w in ia {
            if ob.contains(&w) {
                add(w, out);
            }
        }
        for &w in ia.iter().chain(ob) {
            add(w, out);
        }
        // tips whose neighbours are still pending on the stack: other open edges at a and b
        for &w in self.out[a as usize].items().iter().chain(self.inn[b as usize].items()) {
            add(w, out);
        }
        n
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConnMode {
    Sequential = 0,
    Traversal = 1,
}

#[derive(Clone, Copy, Debug)]
pub struct MeshOpts {
    pub conn: ConnMode,
    /// Position quantum = `2^qbits` sub-grid units (0 = lossless). At most 8.
    pub qbits: u8,
    /// Octahedral bits per component (2..=16).
    pub normal_bits: u8,
}

impl Default for MeshOpts {
    fn default() -> Self {
        MeshOpts { conn: ConnMode::Traversal, qbits: 0, normal_bits: 10 }
    }
}

/// Fixed-size per-brick header (stored as inline FlatBuffers-style fields).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct MeshHeader {
    pub vertex_count: u32,
    pub index_count: u32,
    pub lod: u8,
    pub qbits: u8,
    pub conn: u8,
    pub normal_bits: u8,
    /// bit 0: dirs present; bit 1: geometric normal predictor is flipped.
    pub flags: u8,
    /// In-cell fractions are multiples of `2^fshift` (coarse LODs: the edge parameter has
    /// `T_STEPS` steps per cell), found by the encoder; lossless. Schema field added in v1.0
    /// development with default 0, so files without it decode unchanged.
    pub fshift: u8,
}

/// Decoded brick: quantised brick-local positions (units of `2^qbits` sub-grid steps).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Decoded {
    pub local: Vec<[i32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub dirs: Vec<u8>,
    pub indices: Vec<u32>,
}

fn origin_sub(grid: &BrickGrid, coord: BrickCoord) -> [i64; 3] {
    grid.brick_origin(coord).map(|c| c * SUB)
}

#[inline]
fn quant(g: i64, q: u8) -> i64 {
    if q == 0 { g } else { ((g + (1i64 << (q - 1))) >> q) << q }
}

/// Global sub-grid point after quantisation (what the decoder will reproduce).
pub fn quantise_global(g: [i64; 3], qbits: u8) -> [i64; 3] {
    g.map(|c| quant(c, qbits))
}

// -------------------------------------------------------------------------- octahedral normals

fn oct_encode(n: [f64; 3], c: i32) -> [i32; 2] {
    let s = n[0].abs() + n[1].abs() + n[2].abs();
    if s.is_nan() || s <= 0.0 || !s.is_finite() {
        return [0, 0];
    }
    let (mut x, mut y) = (n[0] / s, n[1] / s);
    if n[2] < 0.0 {
        let sx = if x >= 0.0 { 1.0 } else { -1.0 };
        let sy = if y >= 0.0 { 1.0 } else { -1.0 };
        (x, y) = ((1.0 - y.abs()) * sx, (1.0 - x.abs()) * sy);
    }
    let cf = c as f64;
    [(x * cf).round() as i32, (y * cf).round() as i32].map(|v| v.clamp(-c, c))
}

fn oct_decode(e: [i32; 2], c: i32) -> [f32; 3] {
    let (mut x, mut y) = (e[0] as f64 / c as f64, e[1] as f64 / c as f64);
    let z = 1.0 - x.abs() - y.abs();
    if z < 0.0 {
        let sx = if x >= 0.0 { 1.0 } else { -1.0 };
        let sy = if y >= 0.0 { 1.0 } else { -1.0 };
        (x, y) = ((1.0 - y.abs()) * sx, (1.0 - x.abs()) * sy);
    }
    let l = (x * x + y * y + z * z).sqrt();
    [(x / l) as f32, (y / l) as f32, (z / l) as f32]
}

#[inline]
fn wrap(v: i32, c: i32) -> i32 {
    let m = 2 * c + 1;
    (v + c).rem_euclid(m) - c
}

fn geom_normals(local: &[[i32; 3]], tris: &[[u32; 3]]) -> Vec<[i64; 3]> {
    let mut acc = vec![[0i64; 3]; local.len()];
    for t in tris {
        let [a, b, c] = t.map(|i| local[i as usize].map(i64::from));
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        for &i in t {
            let s = &mut acc[i as usize];
            for k in 0..3 {
                s[k] = s[k].saturating_add(n[k]);
            }
        }
    }
    acc
}

/// Normal predictor: the area-weighted geometric normal of the vertex's decoded faces (Draco's
/// geometric normal predictor). Blending in the parents' decoded normals was measured worse
/// (+0.1 to +0.3 bits/triangle). Pure integer + f64 arithmetic on decoded data: reproducible.
fn pred_oct(g: [i64; 3], flip: bool, c: i32) -> [i32; 2] {
    let s = if flip { -1.0 } else { 1.0 };
    oct_encode(g.map(|v| v as f64 * s), c)
}

/// Bit 2 of `dirs` ("outside endpoint is the higher one") is the sign of the density gradient
/// along the crossing axis, i.e. of the normal's component there.
#[inline]
fn sign_pred(n: [f32; 3], dir: u8) -> u8 {
    let ax = (dir & 3) as usize;
    (ax < 3 && n[ax] > 0.0) as u8
}

// ------------------------------------------------------------------------------- positions

const NO_PRED: u32 = u32::MAX;

#[inline]
fn split_pred(p: i64, cell: i64, axis_role: bool) -> i64 {
    if axis_role { p.div_euclid(cell) } else { (p + cell / 2).div_euclid(cell) }
}

fn predict(local: &[[i32; 3]], v: usize, par: [u32; 3]) -> [i64; 3] {
    if par[0] != NO_PRED {
        let [a, b, o] = par.map(|i| local[i as usize]);
        [0, 1, 2].map(|k| a[k] as i64 + b[k] as i64 - o[k] as i64)
    } else if v > 0 {
        local[v - 1].map(i64::from)
    } else {
        [0; 3]
    }
}

fn log_position(log: &mut SymLog, q: [i32; 3], pred: [i64; 3], axis: u8, cell: i64, fshift: u8) {
    for k in 0..3 {
        let role = axis as usize == k;
        let qk = q[k] as i64;
        let (l, f) = (qk.div_euclid(cell), qk.rem_euclid(cell));
        let lp = split_pred(pred[k], cell, role);
        log.sval(if role { M_LAT_AX } else { M_LAT_OFF }, (l - lp) as i32);
        if cell > 1 {
            debug_assert_eq!(f & ((1 << fshift) - 1), 0);
            log.uval(if role { M_FRAC_AX } else { M_FRAC_OFF }, (f >> fshift) as u32);
        }
    }
}

fn read_position(d: &mut Decoder, pred: [i64; 3], axis: u8, cell: i64, fshift: u8) -> Result<[i32; 3], CodecError> {
    let mut out = [0i32; 3];
    for k in 0..3 {
        let role = axis as usize == k;
        let lp = split_pred(pred[k], cell, role);
        let l = lp + d.sval(if role { M_LAT_AX } else { M_LAT_OFF })? as i64;
        let f = if cell > 1 { (d.uval(if role { M_FRAC_AX } else { M_FRAC_OFF })? as i64) << fshift } else { 0 };
        if f >= cell {
            return Err(CodecError::Corrupt);
        }
        let v = l.checked_mul(cell).and_then(|x| x.checked_add(f)).ok_or(CodecError::Corrupt)?;
        out[k] = i32::try_from(v).map_err(|_| CodecError::Corrupt)?;
        if out[k].unsigned_abs() > LOCAL_MAX {
            return Err(CodecError::Corrupt);
        }
    }
    Ok(out)
}

/// Brick-local coordinates stay well inside i32 (a LOD-6 brick is 2^21 sub-units wide).
const LOCAL_MAX: u32 = 1 << 28;

// ---------------------------------------------------------------------------------- encoder

/// Encode one brick into a symbol log (pass 1). The log is rANS-coded later with the file's
/// shared models (pass 2).
pub fn encode(grid: &BrickGrid, coord: BrickCoord, m: &BrickMesh, opts: MeshOpts) -> (MeshHeader, SymLog) {
    assert!(opts.qbits <= 8 && (2..=16).contains(&opts.normal_bits));
    let org = origin_sub(grid, coord);
    let nv = m.ipos.len();
    let has_dirs = m.dirs.len() == nv;
    let qb = opts.qbits;
    let local_in: Vec<[i32; 3]> = m
        .ipos
        .iter()
        .map(|&g| {
            let qg = quantise_global(g, qb);
            [0, 1, 2].map(|k| {
                let l = (qg[k] - org[k]) >> qb;
                assert!(l.unsigned_abs() <= LOCAL_MAX as u64, "vertex far outside its brick");
                l as i32
            })
        })
        .collect();
    let tris_in: Vec<[u32; 3]> = m.indices.as_chunks::<3>().0.to_vec();
    let mut log = SymLog::default();

    // connectivity -> vertex order (new -> old), triangles in new ids, parallelogram parents
    let mut order: Vec<u32> = Vec::with_capacity(nv);
    let mut new_id = vec![u32::MAX; nv];
    let mut par: Vec<[u32; 3]> = Vec::with_capacity(nv);
    let mut tris: Vec<[u32; 3]> = Vec::with_capacity(tris_in.len());
    match opts.conn {
        ConnMode::Sequential => {
            order.extend(0..nv as u32);
            par.resize(nv, [NO_PRED; 3]);
            let mut hi: i64 = 0;
            for &i in &m.indices {
                log.sval(M_SEQ, (i as i64 - hi) as i32);
                hi = hi.max(i as i64 + 1);
            }
            tris = tris_in.clone();
        }
        ConnMode::Traversal => {
            use std::collections::HashMap;
            let mut edge: HashMap<(u32, u32), (u32, u8)> = HashMap::with_capacity(tris_in.len() * 3);
            for (t, tri) in tris_in.iter().enumerate() {
                for k in 0..3 {
                    edge.entry((tri[k], tri[(k + 1) % 3])).or_insert((t as u32, k as u8));
                }
            }
            let mut claimed = vec![false; tris_in.len()];
            let mut open = OpenEdges::new(nv);
            let mut cand = [0u32; MAX_CAND];
            // gate: (a, b, o) in new ids, plus the neighbour (tri, rotation) it leads to
            let mut stack: Vec<([u32; 3], u32, u8)> = Vec::new();
            let assign =
                |old: u32, p: [u32; 3], order: &mut Vec<u32>, par: &mut Vec<[u32; 3]>, new_id: &mut Vec<u32>| -> u32 {
                    let id = order.len() as u32;
                    new_id[old as usize] = id;
                    order.push(old);
                    par.push(p);
                    id
                };
            let neighbour = |claimed: &[bool], t: u32, a: u32, b: u32| -> Option<(u32, u8)> {
                // across directed edge (a, b) (old ids) lies the triangle holding (b, a)
                edge.get(&(b, a)).copied().filter(|&(n, _)| n != t && !claimed[n as usize])
            };
            for t0 in 0..tris_in.len() {
                if claimed[t0] {
                    continue;
                }
                claimed[t0] = true;
                let tri = tris_in[t0];
                let mut ids = [0u32; 3];
                for k in 0..3 {
                    let old = tri[k];
                    if new_id[old as usize] == u32::MAX {
                        log.sym(M_START, 0);
                        ids[k] = assign(old, [NO_PRED; 3], &mut order, &mut par, &mut new_id);
                    } else {
                        log.sym(M_START, 1);
                        log.uval(M_REF, order.len() as u32 - 1 - new_id[old as usize]);
                        ids[k] = new_id[old as usize];
                    }
                }
                tris.push(ids);
                open.add_tri(ids);
                let mut flags = 0u8;
                let mut kids = Vec::new();
                for k in 0..3 {
                    if let Some(nb) = neighbour(&claimed, t0 as u32, tri[k], tri[(k + 1) % 3]) {
                        flags |= 1 << k;
                        kids.push(([ids[k], ids[(k + 1) % 3], ids[(k + 2) % 3]], nb.0, nb.1));
                    }
                }
                log.sym(M_SFLAGS, flags);
                stack.extend(kids.into_iter().rev());
                while let Some(([a, b, o], t, rot)) = stack.pop() {
                    if tris.len() == tris_in.len() {
                        break;
                    }
                    // claim-on-visit: a gate whose edge got closed meanwhile is stale (both
                    // sides know this from the open-edge state); a still-open gate to a triangle
                    // that is already coded (non-manifold input) is an explicit skip
                    if !open.is_open(a, b) {
                        continue;
                    }
                    if claimed[t as usize] {
                        log.sym(M_CONN, SKIP);
                        continue;
                    }
                    claimed[t as usize] = true;
                    let tt = tris_in[t as usize];
                    let r = rot as usize;
                    let (ob, oa, otip) = (tt[r], tt[(r + 1) % 3], tt[(r + 2) % 3]);
                    debug_assert_eq!((new_id[ob as usize], new_id[oa as usize]), (b, a));
                    let tip_new = new_id[otip as usize] == u32::MAX;
                    let left = neighbour(&claimed, t, oa, otip);
                    let right = neighbour(&claimed, t, otip, ob);
                    log.sym(M_CONN, tip_new as u8 | (left.is_some() as u8) << 1 | (right.is_some() as u8) << 2);
                    let tip = if tip_new {
                        assign(otip, [a, b, o], &mut order, &mut par, &mut new_id)
                    } else {
                        let id = new_id[otip as usize];
                        let nc = open.candidates(a, b, &mut cand);
                        match cand[..nc].iter().position(|&w| w == id) {
                            Some(k) => log.sym(M_CAND, k as u8),
                            None => {
                                log.sym(M_CAND, MAX_CAND as u8);
                                log.uval(M_REF, order.len() as u32 - 1 - id);
                            }
                        }
                        id
                    };
                    tris.push([b, a, tip]);
                    open.add_tri([b, a, tip]);
                    if let Some(nb) = left {
                        stack.push(([a, tip, b], nb.0, nb.1));
                    }
                    if let Some(nb) = right {
                        stack.push(([tip, b, a], nb.0, nb.1));
                    }
                }
            }
            for old in 0..nv as u32 {
                if new_id[old as usize] == u32::MAX {
                    assign(old, [NO_PRED; 3], &mut order, &mut par, &mut new_id);
                }
            }
        }
    }

    // attributes, in new-vertex order
    let cell = (SUB << coord.lod) >> qb;
    let local: Vec<[i32; 3]> = order.iter().map(|&o| local_in[o as usize]).collect();
    let dirs: Vec<u8> = if has_dirs { order.iter().map(|&o| m.dirs[o as usize]).collect() } else { Vec::new() };
    assert!(dirs.iter().all(|&d| d < 64), "dirs use 6 bits at most");
    let log2_cell = cell.trailing_zeros() as u8;
    let fshift = local
        .iter()
        .flatten()
        .map(|&c| (c as i64).rem_euclid(cell))
        .filter(|&f| f != 0)
        .map(|f| f.trailing_zeros() as u8)
        .min()
        .unwrap_or(log2_cell)
        .min(log2_cell);
    for v in 0..nv {
        let axis = if has_dirs {
            // the sign bit (bit 2) is coded after the normals, predicted from them
            log.sym(M_DIR, dirs[v] & !4);
            dirs[v] & 3
        } else {
            3
        };
        log_position(&mut log, local[v], predict(&local, v, par[v]), axis, cell, fshift);
    }
    let c = (1i32 << (opts.normal_bits - 1)) - 1;
    let geo = geom_normals(&local, &tris);
    let normals: Vec<[f64; 3]> =
        order.iter().map(|&o| m.vertices.get(o as usize).map_or([0.0; 3], |v| v.normal.map(f64::from))).collect();
    let dot: f64 = (0..nv).map(|v| (0..3).map(|k| geo[v][k] as f64 * normals[v][k]).sum::<f64>().signum()).sum();
    let flip = dot < 0.0;
    let mut dec_n = Vec::with_capacity(nv);
    for v in 0..nv {
        let e = oct_encode(normals[v], c);
        let p = pred_oct(geo[v], flip, c);
        log.sval(M_NU, wrap(e[0] - p[0], c));
        log.sval(M_NV, wrap(e[1] - p[1], c));
        dec_n.push(oct_decode(e, c));
    }
    if has_dirs {
        for v in 0..nv {
            log.sym(M_DSGN, (dirs[v] >> 2) & 1 ^ sign_pred(dec_n[v], dirs[v]));
        }
    }
    let hdr = MeshHeader {
        vertex_count: nv as u32,
        index_count: m.indices.len() as u32,
        lod: coord.lod,
        qbits: qb,
        conn: opts.conn as u8,
        normal_bits: opts.normal_bits,
        flags: has_dirs as u8 | (flip as u8) << 1,
        fshift,
    };
    (hdr, log)
}

// ---------------------------------------------------------------------------------- decoder

/// Decode one brick. Total: corrupt input gives an error, never a panic or unbounded allocation
/// (counts are validated against [`crate::fb::Limits`] by the caller before this runs).
pub fn decode(h: &MeshHeader, d: &mut Decoder) -> Result<Decoded, CodecError> {
    let nv = h.vertex_count as usize;
    let ni = h.index_count as usize;
    if !ni.is_multiple_of(3) || h.qbits > 8 || !(2..=16).contains(&h.normal_bits) || h.lod > voxel::MAX_LOD + 4 {
        return Err(CodecError::Corrupt);
    }
    let nt = ni / 3;
    let mut tris: Vec<[u32; 3]> = Vec::with_capacity(nt);
    let mut par: Vec<[u32; 3]> = Vec::with_capacity(nv);
    match h.conn {
        0 => {
            par.resize(nv, [NO_PRED; 3]);
            let mut hi: i64 = 0;
            let mut cur = [0u32; 3];
            for i in 0..ni {
                let v = hi + d.sval(M_SEQ)? as i64;
                if !(0..nv as i64).contains(&v) {
                    return Err(CodecError::Corrupt);
                }
                cur[i % 3] = v as u32;
                hi = hi.max(v + 1);
                if i % 3 == 2 {
                    tris.push(cur);
                }
            }
        }
        1 => {
            let mut next: u32 = 0;
            let mut pushed = 0usize; // start triangles + gates pushed (<= 4 per triangle)
            let mut stack: Vec<[u32; 3]> = Vec::new();
            let mut open = OpenEdges::new(nv);
            let mut cand = [0u32; MAX_CAND];
            let new_vertex = |p: [u32; 3], next: &mut u32, par: &mut Vec<[u32; 3]>| -> Result<u32, CodecError> {
                if *next as usize >= nv {
                    return Err(CodecError::Corrupt);
                }
                par.push(p);
                *next += 1;
                Ok(*next - 1)
            };
            let read_ref = |d: &mut Decoder, next: u32| -> Result<u32, CodecError> {
                let r = d.uval(M_REF)?;
                if r >= next {
                    return Err(CodecError::Corrupt);
                }
                Ok(next - 1 - r)
            };
            while tris.len() < nt {
                let Some([a, b, o]) = stack.pop() else {
                    pushed += 1;
                    let mut ids = [0u32; 3];
                    for id in &mut ids {
                        *id = match d.sym(M_START)? {
                            0 => new_vertex([NO_PRED; 3], &mut next, &mut par)?,
                            1 => read_ref(d, next)?,
                            _ => return Err(CodecError::BadSymbol),
                        };
                    }
                    tris.push(ids);
                    open.add_tri(ids);
                    let flags = d.sym(M_SFLAGS)?;
                    if flags > 7 {
                        return Err(CodecError::BadSymbol);
                    }
                    for k in (0..3).rev() {
                        if flags & (1 << k) != 0 {
                            stack.push([ids[k], ids[(k + 1) % 3], ids[(k + 2) % 3]]);
                            pushed += 1;
                        }
                    }
                    if pushed > 4 * nt {
                        return Err(CodecError::Corrupt);
                    }
                    continue;
                };
                if a.max(b).max(o) >= next {
                    return Err(CodecError::Corrupt);
                }
                if !open.is_open(a, b) {
                    continue;
                }
                let s = d.sym(M_CONN)?;
                if s == SKIP {
                    continue;
                }
                if s > 7 {
                    return Err(CodecError::BadSymbol);
                }
                let tip = if s & 1 != 0 {
                    new_vertex([a, b, o], &mut next, &mut par)?
                } else {
                    let nc = open.candidates(a, b, &mut cand);
                    match d.sym(M_CAND)? as usize {
                        k if k < nc => cand[k],
                        MAX_CAND => read_ref(d, next)?,
                        _ => return Err(CodecError::Corrupt),
                    }
                };
                tris.push([b, a, tip]);
                open.add_tri([b, a, tip]);
                if s & 2 != 0 {
                    stack.push([a, tip, b]);
                    pushed += 1;
                }
                if s & 4 != 0 {
                    stack.push([tip, b, a]);
                    pushed += 1;
                }
                if pushed > 4 * nt {
                    return Err(CodecError::Corrupt);
                }
            }
            while (next as usize) < nv {
                new_vertex([NO_PRED; 3], &mut next, &mut par)?;
            }
        }
        _ => return Err(CodecError::Corrupt),
    }
    let has_dirs = h.flags & 1 != 0;
    let cell = (SUB << h.lod) >> h.qbits;
    let mut local: Vec<[i32; 3]> = Vec::with_capacity(nv);
    let mut dirs = Vec::with_capacity(if has_dirs { nv } else { 0 });
    if h.fshift as u32 > cell.trailing_zeros() {
        return Err(CodecError::Corrupt);
    }
    for (v, pv) in par.iter().enumerate() {
        let axis = if has_dirs {
            let s = d.sym(M_DIR)?;
            if s & 4 != 0 {
                return Err(CodecError::BadSymbol);
            }
            dirs.push(s);
            s & 3
        } else {
            3
        };
        let p = predict(&local, v, *pv);
        local.push(read_position(d, p, axis, cell, h.fshift)?);
    }
    let c = (1i32 << (h.normal_bits - 1)) - 1;
    let flip = h.flags & 2 != 0;
    let geo = geom_normals(&local, &tris);
    let mut normals = Vec::with_capacity(nv);
    for g in &geo {
        let p = pred_oct(*g, flip, c);
        let ru = d.sval(M_NU)?;
        let rv = d.sval(M_NV)?;
        if ru.unsigned_abs() > c as u32 || rv.unsigned_abs() > c as u32 {
            return Err(CodecError::Corrupt);
        }
        normals.push(oct_decode([wrap(p[0] + ru, c), wrap(p[1] + rv, c)], c));
    }
    for (dv, n) in dirs.iter_mut().zip(&normals) {
        let b = d.sym(M_DSGN)?;
        if b > 1 {
            return Err(CodecError::BadSymbol);
        }
        *dv |= (b ^ sign_pred(*n, *dv)) << 2;
    }
    let indices = tris.into_flattened();
    Ok(Decoded { local, normals, dirs, indices })
}

impl Decoded {
    /// Global sub-grid positions (the mesher's `ipos`).
    pub fn ipos(&self, grid: &BrickGrid, coord: BrickCoord, qbits: u8) -> Vec<[i64; 3]> {
        let org = origin_sub(grid, coord);
        self.local.iter().map(|l| [0, 1, 2].map(|k| org[k] + ((l[k] as i64) << qbits))).collect()
    }

    /// Write render vertices (exactly [`VoxelConfig::sub_to_render`], so bit-identical to the
    /// mesher's f32 output) and rebased indices straight into GPU-mapped slices.
    pub fn write_gpu(
        &self,
        cfg: &VoxelConfig,
        coord: BrickCoord,
        qbits: u8,
        verts: &mut [VoxelVertex],
        idx: &mut [u32],
        base: u32,
    ) {
        let org = origin_sub(&cfg.grid, coord);
        for ((out, l), n) in verts.iter_mut().zip(&self.local).zip(&self.normals) {
            let g = [0, 1, 2].map(|k| org[k] + ((l[k] as i64) << qbits));
            *out = VoxelVertex { pos: cfg.sub_to_render(g), normal: *n };
        }
        for (o, &i) in idx.iter_mut().zip(&self.indices) {
            *o = i + base;
        }
    }

    pub fn to_brick_mesh(&self, cfg: &VoxelConfig, coord: BrickCoord, qbits: u8) -> BrickMesh {
        let mut vertices = vec![VoxelVertex::default(); self.local.len()];
        let mut indices = vec![0; self.indices.len()];
        self.write_gpu(cfg, coord, qbits, &mut vertices, &mut indices, 0);
        BrickMesh { ipos: self.ipos(&cfg.grid, coord, qbits), vertices, dirs: self.dirs.clone(), indices }
    }
}

/// Canonical triangle set keyed by global position (rotation-normalised, sorted): equal iff two
/// meshes have the same oriented triangles over the same points, whatever the vertex order.
pub fn canonical_tris(ipos: &[[i64; 3]], indices: &[u32]) -> Vec<[[i64; 3]; 3]> {
    let mut v: Vec<[[i64; 3]; 3]> = indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|t| {
            let p = t.map(|i| ipos[i as usize]);
            let r = (0..3).min_by_key(|&k| p[k]).unwrap_or(0);
            [p[r], p[(r + 1) % 3], p[(r + 2) % 3]]
        })
        .collect();
    v.sort_unstable();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oct_roundtrip_error_small() {
        let c = (1 << 9) - 1;
        let mut worst = 0.0f64;
        for i in 0..2000 {
            let t = i as f64 * 0.7311;
            let n = [t.sin() * (t * 0.37).cos(), (t * 1.3).cos(), t.cos() * (t * 0.37).sin() - 0.2];
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            let n = n.map(|v| v / l);
            let d = oct_decode(oct_encode(n, c), c);
            let dot: f64 = (0..3).map(|k| n[k] * d[k] as f64).sum();
            worst = worst.max(dot.min(1.0).acos().to_degrees());
        }
        assert!(worst < 0.25, "10-bit oct error {worst} deg");
    }

    #[test]
    fn wrap_is_inverse() {
        let c = 511;
        for a in -c..=c {
            for p in [-c, -3, 0, 7, c] {
                assert_eq!(wrap(p + wrap(a - p, c), c), a);
            }
        }
    }
}
