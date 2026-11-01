//! Lossless coding of quantised density samples ([`voxel::BrickSamples`]): a 3D Lorenzo
//! predictor (the parallelogram rule generalised to a lattice: `sum over the 7 causal cube
//! corners with signs (-1)^(k+1)`), residuals through the shared rANS stream. Transition-face
//! fine samples use the 2D Lorenzo predictor. Bricks of clamped (far inside / outside) samples
//! cost almost nothing.
//!
//! **Band-clamped payloads** (format minor 2, `SampleBlob.band = t > 0`): [`mesh_band`] keeps
//! every sample the mesher can read exactly and clamps the rest to `+-t`. Such a payload is
//! *not* density: it is only valid for meshing at iso 0 with its own transition mask. The
//! reader hands it out only as [`MeshSamples`]; [`crate::schema::File::decode_samples`]
//! (true density) refuses it with [`CodecError::Banded`].

use crate::rans::{CodecError, Decoder, SymLog};
use coords::BrickCoord;
use voxel::BrickSamples;

pub const M_SAMPLE: u8 = 12;
pub const M_SFACE: u8 = 13;
pub const MAX_N: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct SampleHeader {
    pub n: u16,
    pub face_mask: u8,
    /// 0: full density. `t >= 1`: band-clamped with threshold `t` ([`mesh_band`]).
    pub band: u32,
}

#[inline]
fn lorenzo3(d: &[i32], m: usize, x: usize, y: usize, z: usize) -> i64 {
    let g = |dx: usize, dy: usize, dz: usize| -> i64 {
        if x < dx || y < dy || z < dz { 0 } else { d[(x - dx) + m * ((y - dy) + m * (z - dz))] as i64 }
    };
    g(1, 0, 0) + g(0, 1, 0) + g(0, 0, 1) - g(1, 1, 0) - g(1, 0, 1) - g(0, 1, 1) + g(1, 1, 1)
}

#[inline]
fn lorenzo2(d: &[i32], m: usize, u: usize, v: usize) -> i64 {
    let g = |du: usize, dv: usize| -> i64 { if u < du || v < dv { 0 } else { d[(u - du) + m * (v - dv)] as i64 } };
    g(1, 0) + g(0, 1) - g(1, 1)
}

fn resid(v: i32, p: i64) -> i32 {
    // samples are clamped to +-DQ_MAX and predictions to 7 * DQ_MAX, so this fits easily
    (v as i64 - p) as i32
}

pub fn encode(s: &BrickSamples) -> (SampleHeader, SymLog) {
    let m = s.n + 3;
    assert_eq!(s.d.len(), m * m * m);
    let mut log = SymLog::default();
    for z in 0..m {
        for y in 0..m {
            for x in 0..m {
                log.sval(M_SAMPLE, resid(s.d[x + m * (y + m * z)], lorenzo3(&s.d, m, x, y, z)));
            }
        }
    }
    let f = 2 * s.n + 1;
    let mut mask = 0u8;
    for (k, face) in s.faces.iter().enumerate() {
        if face.is_empty() {
            continue;
        }
        assert_eq!(face.len(), f * f);
        mask |= 1 << k;
        for v in 0..f {
            for u in 0..f {
                log.sval(M_SFACE, resid(face[u + f * v], lorenzo2(face, f, u, v)));
            }
        }
    }
    (SampleHeader { n: s.n as u16, face_mask: mask, band: 0 }, log)
}

/// Encode the mesh-exact band of `s` (threshold `t >= 1` quanta; the transition mask is the set
/// of faces `s` carries).
pub fn encode_banded(s: &BrickSamples, t: i32) -> (SampleHeader, SymLog) {
    let b = BrickSamples { coord: s.coord, n: s.n, d: mesh_band(s, face_mask(s), t), faces: s.faces.clone() };
    let (mut h, log) = encode(&b);
    h.band = t as u32;
    (h, log)
}

/// Transition mask implied by the faces a brick carries.
pub fn face_mask(s: &BrickSamples) -> u8 {
    (0..6).filter(|&f| !s.faces[f].is_empty()).fold(0, |m, f| m | (1 << f))
}

/// A band-clamped (or full) payload that is **only valid for meshing** at iso 0 with the
/// transition mask it was stored with. Off-surface values are not density: anything that needs
/// true density (collision, SDF queries, AO from samples, re-meshing at another LOD / iso /
/// transition mask, edits / digging) must use [`crate::schema::File::decode_samples`] on a full
/// payload, or regenerate the brick from the field.
pub struct MeshSamples {
    inner: BrickSamples,
    band: u32,
}

impl MeshSamples {
    pub fn coord(&self) -> BrickCoord {
        self.inner.coord
    }
    /// The transition mask this payload may be meshed with (and only this one).
    pub fn transitions(&self) -> u8 {
        face_mask(&self.inner)
    }
    /// 0 = full density underneath, else the clamp threshold.
    pub fn band(&self) -> u32 {
        self.band
    }
    /// Mesh with the stored transition mask (bit-identical to meshing the full samples).
    pub fn mesh(&self, cfg: &voxel::VoxelConfig, finer: Option<voxel::Aabb>) -> voxel::BrickMesh {
        voxel::mesh_brick(cfg, &self.inner, self.transitions(), finer)
    }
    /// The raw arrays, for handing to a mesher that takes [`BrickSamples`] (the GPU Transvoxel
    /// mesher). The caller must pass [`Self::transitions`] as the mask and use the result for
    /// meshing only; named so every such use is visible in review / grep.
    pub fn samples_for_meshing_only(&self) -> &BrickSamples {
        &self.inner
    }
    /// Full payloads only: the true density (banded ones return `None`).
    pub fn into_density(self) -> Option<BrickSamples> {
        (self.band == 0).then_some(self.inner)
    }
}

/// Decoded payload, typed by what it means.
pub enum SamplePayload {
    Full(BrickSamples),
    Banded(MeshSamples),
}

impl SamplePayload {
    /// Meshing view (valid for both kinds).
    pub fn for_meshing(self) -> MeshSamples {
        match self {
            SamplePayload::Full(s) => MeshSamples { inner: s, band: 0 },
            SamplePayload::Banded(m) => m,
        }
    }
}

/// Decode any payload and type it; a banded payload must be a fixed point of [`mesh_band`]
/// (anything else is corrupt or forged).
pub fn decode_payload(h: &SampleHeader, coord: BrickCoord, d: &mut Decoder) -> Result<SamplePayload, CodecError> {
    let s = decode(h, coord, d)?;
    if h.band == 0 {
        return Ok(SamplePayload::Full(s));
    }
    if h.band > voxel::DQ_MAX as u32 || h.face_mask != face_mask(&s) || mesh_band(&s, h.face_mask, h.band as i32) != s.d {
        return Err(CodecError::Corrupt);
    }
    Ok(SamplePayload::Banded(MeshSamples { inner: s, band: h.band }))
}


fn decode(h: &SampleHeader, coord: BrickCoord, d: &mut Decoder) -> Result<BrickSamples, CodecError> {
    let n = h.n as usize;
    if n == 0 || n > MAX_N || h.face_mask >= 64 {
        return Err(CodecError::Corrupt);
    }
    let m = n + 3;
    let lim = voxel::DQ_MAX as i64;
    let mut out = vec![0i32; m * m * m];
    for z in 0..m {
        for y in 0..m {
            for x in 0..m {
                let v = lorenzo3(&out, m, x, y, z) + d.sval(M_SAMPLE)? as i64;
                if v.abs() > lim {
                    return Err(CodecError::Corrupt);
                }
                out[x + m * (y + m * z)] = v as i32;
            }
        }
    }
    let f = 2 * n + 1;
    let mut faces: [Vec<i32>; 6] = Default::default();
    for (k, face) in faces.iter_mut().enumerate() {
        if h.face_mask & (1 << k) == 0 {
            continue;
        }
        *face = vec![0; f * f];
        for v in 0..f {
            for u in 0..f {
                let x = lorenzo2(face, f, u, v) + d.sval(M_SFACE)? as i64;
                if x.abs() > lim {
                    return Err(CodecError::Corrupt);
                }
                face[u + f * v] = x as i32;
            }
        }
    }
    Ok(BrickSamples { coord, n, d: out, faces })
}

/// Mesh-exact narrow band (the lesson from the TQ study: MSE-optimal coding spends its bits on
/// samples the mesher never reads). Only for meshing-only consumers of the samples: every sample the mesher can read for this brick (corners of cells with
/// a sign change, whole planes of transition faces, then a 1-sample dilation for the central-
/// difference gradients) stays exact; every other sample is clamped to +-t (sign preserved).
/// `t >= 1` (0 would turn negative samples into non-negative ones).
pub fn mesh_band(s: &BrickSamples, mask: u8, t: i32) -> Vec<i32> {
    assert!(t >= 1, "mesh_band: t must be >= 1 to preserve signs");
    let n = s.n as i64;
    let m = n + 3;
    let idx = |i: i64, j: i64, k: i64| (((k + 1) * m + (j + 1)) * m + (i + 1)) as usize;
    let mut keep = vec![false; s.d.len()];
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let c: Vec<bool> = (0..8).map(|q| s.at(i + (q & 1), j + ((q >> 1) & 1), k + (q >> 2)) < 0).collect();
                if c.iter().any(|&x| x != c[0]) {
                    for q in 0..8 {
                        keep[idx(i + (q & 1), j + ((q >> 1) & 1), k + (q >> 2))] = true;
                    }
                }
            }
        }
    }
    for f in 0..6 {
        if mask & (1 << f) != 0 {
            let (a, ua, va) = voxel::face_axes(f);
            for v in 0..=n {
                for u in 0..=n {
                    let mut c = [0i64; 3];
                    c[a] = if f % 2 == 1 { n } else { 0 };
                    c[ua] = u;
                    c[va] = v;
                    keep[idx(c[0], c[1], c[2])] = true;
                }
            }
        }
    }
    let mut dil = keep.clone();
    for k in -1..=n + 1 {
        for j in -1..=n + 1 {
            for i in -1..=n + 1 {
                if keep[idx(i, j, k)] {
                    for (di, dj, dk) in (0..27).map(|q| (q % 3 - 1, (q / 3) % 3 - 1, q / 9 - 1)) {
                        let (a, b, c) = (i + di, j + dj, k + dk);
                        if (-1..=n + 1).contains(&a) && (-1..=n + 1).contains(&b) && (-1..=n + 1).contains(&c) {
                            dil[idx(a, b, c)] = true;
                        }
                    }
                }
            }
        }
    }
    s.d.iter().zip(&dil).map(|(&x, &k)| if k { x } else { x.clamp(-t, t) }).collect()
}

