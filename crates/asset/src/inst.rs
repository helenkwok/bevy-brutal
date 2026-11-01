//! Vegetation / prop instances, Draco-style: positions quantised to 1/128 m relative to a chunk
//! origin (integers, so chunk-independent and deterministic), Morton-sorted in XZ so neighbours
//! in the stream are neighbours on the ground, delta-predicted; yaw quantised to 12 bits (raw,
//! it is uniformly random); height as `round(64 * log2 h)` (1.09 % relative step, error <= 0.55 %), width as a
//! ratio of height (1/256), both delta-predicted per mesh; mesh id through its own model;
//! material: the low 10 bits raw (per-instance random), the high 6 bits (class / species tag,
//! highly skewed) through `M_IMAT` (format minor 1, `mat_bits = 16`). Chunks written before
//! minor 1 have no `mat_bits` field: the default 10 means "10 bits raw, high bits zero".

use crate::rans::{CodecError, Decoder, SymLog};
use lowlevel_abi::InstanceGpu;

pub const M_IX: u8 = 14;
pub const M_IY: u8 = 15;
pub const M_IZ: u8 = 16;
pub const M_IH: u8 = 17;
pub const M_IW: u8 = 18;
pub const M_IMESH: u8 = 19;
pub const M_IMAT: u8 = 22;
/// Material bits of chunks this writer produces (old chunks: 10).
pub const MAT_BITS: u8 = 16;
pub const POS_Q: f64 = 128.0;
pub const YAW_BITS: u32 = 12;
pub const MAX_MESH: u8 = 63;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    /// World (planet-frame) position, metres, f64.
    pub pos: [f64; 3],
    pub yaw: f32,
    /// Vertical scale (tree height in metres for unit meshes).
    pub height: f32,
    /// Horizontal scale.
    pub width: f32,
    pub mesh: u8,
    pub material: u16,
}

impl Instance {
    /// Camera-relative f32 model matrix (`scale * rot_y(yaw)`, translation), the
    /// `Mat4::from_scale_rotation_translation` layout the instanced pass consumes.
    pub fn to_gpu(&self, cam: [f64; 3]) -> InstanceGpu {
        let (s, c) = (self.yaw as f64).sin_cos();
        let (w, h) = (self.width as f64, self.height as f64);
        let t = [0, 1, 2].map(|k| (self.pos[k] - cam[k]) as f32);
        let model = [
            [(c * w) as f32, 0.0, (-s * w) as f32, 0.0],
            [0.0, h as f32, 0.0, 0.0],
            [(s * w) as f32, 0.0, (c * w) as f32, 0.0],
            [t[0], t[1], t[2], 1.0],
        ];
        InstanceGpu { model, mesh_id: self.mesh as u32, material_id: self.material as u32, _pad: [0; 2] }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct InstHeader {
    pub count: u32,
    /// Chunk origin in 1/128 m units.
    pub origin: [i64; 3],
    /// 10 (pre-minor-1 chunks: material 0..1023 raw) or 16 (low 10 raw + high 6 modelled).
    pub mat_bits: u8,
}

fn morton2(x: u32, z: u32) -> u64 {
    fn spread(v: u32) -> u64 {
        let mut v = v as u64;
        v = (v | (v << 16)) & 0x0000_FFFF_0000_FFFF;
        v = (v | (v << 8)) & 0x00FF_00FF_00FF_00FF;
        v = (v | (v << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
        v = (v | (v << 2)) & 0x3333_3333_3333_3333;
        (v | (v << 1)) & 0x5555_5555_5555_5555
    }
    spread(x) | (spread(z) << 1)
}

struct Q {
    p: [i64; 3],
    yaw: u32,
    h: i32,
    w: i32,
    mesh: u8,
    mat: u16,
}

fn quantise(i: &Instance) -> Q {
    let yaw = ((i.yaw as f64 / std::f64::consts::TAU * (1u64 << YAW_BITS) as f64).round() as i64)
        .rem_euclid(1 << YAW_BITS) as u32;
    let h = (i.height.max(1e-3) as f64).log2() * 64.0;
    let w = (i.width as f64 / i.height.max(1e-3) as f64 * 256.0).round();
    Q {
        p: i.pos.map(|c| (c * POS_Q).round() as i64),
        yaw,
        h: h.round() as i32,
        w: w.clamp(0.0, 1e6) as i32,
        mesh: i.mesh.min(MAX_MESH),
        mat: i.material,
    }
}

/// Encode a chunk (any order in; Morton order out).
pub fn encode(items: &[Instance]) -> (InstHeader, SymLog) {
    encode_bits(items, MAT_BITS)
}

/// `mat_bits = 10` reproduces a pre-minor-1 chunk (high material bits dropped): compatibility
/// tests only.
#[doc(hidden)]
pub fn encode_bits(items: &[Instance], mat_bits: u8) -> (InstHeader, SymLog) {
    assert!(mat_bits == 10 || mat_bits == 16);
    let mut q: Vec<Q> = items.iter().map(quantise).collect();
    let origin = [0, 1, 2].map(|k| q.iter().map(|x| x.p[k]).min().unwrap_or(0));
    for x in &q {
        assert!((0..3).all(|k| x.p[k] - origin[k] < 1 << 31), "instance chunk wider than 16 km");
    }
    q.sort_by_key(|x| (morton2((x.p[0] - origin[0]) as u32, (x.p[2] - origin[2]) as u32), x.p[1], x.mesh, x.yaw));
    let mut log = SymLog::default();
    let mut prev = origin;
    let mut last_hw = [(0i32, 0i32); MAX_MESH as usize + 1];
    for x in &q {
        log.sval(M_IX, (x.p[0] - prev[0]) as i32);
        log.sval(M_IZ, (x.p[2] - prev[2]) as i32);
        log.sval(M_IY, (x.p[1] - prev[1]) as i32);
        prev = x.p;
        log.sym(M_IMESH, x.mesh);
        let (ph, pw) = last_hw[x.mesh as usize];
        log.sval(M_IH, x.h - ph);
        log.sval(M_IW, x.w - pw);
        last_hw[x.mesh as usize] = (x.h, x.w);
        log.bits.write(x.yaw, YAW_BITS);
        log.bits.write(x.mat as u32 & 1023, 10);
        if mat_bits == 16 {
            log.sym(M_IMAT, (x.mat >> 10) as u8);
        }
    }
    (InstHeader { count: q.len() as u32, origin, mat_bits }, log)
}

pub fn decode(h: &InstHeader, d: &mut Decoder, out: &mut Vec<Instance>) -> Result<(), CodecError> {
    let mut prev = h.origin;
    let mut last_hw = [(0i32, 0i32); MAX_MESH as usize + 1];
    out.reserve(h.count as usize);
    for _ in 0..h.count {
        let dx = d.sval(M_IX)? as i64;
        let dz = d.sval(M_IZ)? as i64;
        let dy = d.sval(M_IY)? as i64;
        let p = [prev[0].checked_add(dx), prev[1].checked_add(dy), prev[2].checked_add(dz)];
        let [Some(x), Some(y), Some(z)] = p else { return Err(CodecError::Corrupt) };
        prev = [x, y, z];
        let mesh = d.sym(M_IMESH)?;
        if mesh > MAX_MESH {
            return Err(CodecError::BadSymbol);
        }
        let (ph, pw) = last_hw[mesh as usize];
        let hq = ph.checked_add(d.sval(M_IH)?).ok_or(CodecError::Corrupt)?;
        let wq = pw.checked_add(d.sval(M_IW)?).ok_or(CodecError::Corrupt)?;
        if !(-64 * 16..=64 * 16).contains(&hq) || !(0..=1_000_000).contains(&wq) {
            return Err(CodecError::Corrupt);
        }
        last_hw[mesh as usize] = (hq, wq);
        let yaw = d.bits.read(YAW_BITS)?;
        let mut mat = d.bits.read(10)? as u16;
        if h.mat_bits == 16 {
            let hi = d.sym(M_IMAT)?;
            if hi > 63 {
                return Err(CodecError::BadSymbol);
            }
            mat |= (hi as u16) << 10;
        }
        let height = (hq as f64 / 64.0).exp2();
        out.push(Instance {
            pos: prev.map(|c| c as f64 / POS_Q),
            yaw: (yaw as f64 / (1u64 << YAW_BITS) as f64 * std::f64::consts::TAU) as f32,
            height: height as f32,
            width: (wq as f64 / 256.0 * height) as f32,
            mesh,
            material: mat,
        });
    }
    Ok(())
}
