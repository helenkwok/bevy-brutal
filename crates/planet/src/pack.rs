//! `PLN1`: planet texture pack in the asset container (`asset::fb`, verified zero-copy reads).
//!
//! Root table: 0 `u32` version, 1 `[Texture]`, 2 `Height`, 3 credits (UTF-8 bytes).
//! Texture: 0 `u32` format ([`TexFormat`]), 1 `u32` face size, 2 `u32` mip count, 3 block data
//! (256-aligned; layer-major: for each cube layer +X..-Z, mips 0..n), 4 name (UTF-8).
//! Height: 0 `u32` face size, 1 `u16` texels (layer-major, row-major, Vulkan cube convention),
//! 2 `f64` metres per unit, 3 `f64` offset (m).
use asset::fb::{self, Field, Limits, TableB, Verifier, VerifyError};
use glam::DVec3;

pub const VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TexFormat {
    Bc7Srgb = 1,
    Bc7 = 2,
    Bc5 = 3,
    Bc4 = 4,
    /// Uncompressed formats: the reference ("raw") KTX2 variant and the R16 height cube.
    Rgba8Srgb = 5,
    Rgba8 = 6,
    Rg8 = 7,
    R8 = 8,
    R16 = 9,
}

impl TexFormat {
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            1 => Self::Bc7Srgb,
            2 => Self::Bc7,
            3 => Self::Bc5,
            4 => Self::Bc4,
            5 => Self::Rgba8Srgb,
            6 => Self::Rgba8,
            7 => Self::Rg8,
            8 => Self::R8,
            9 => Self::R16,
            _ => return None,
        })
    }
    /// 4x4 block-compressed (BCn)?
    pub fn is_block(self) -> bool {
        matches!(self, Self::Bc7Srgb | Self::Bc7 | Self::Bc5 | Self::Bc4)
    }
    /// Bytes per 4x4 block (BCn) or per texel (uncompressed).
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Bc4 => 8,
            Self::Bc7Srgb | Self::Bc7 | Self::Bc5 => 16,
            Self::Rgba8Srgb | Self::Rgba8 => 4,
            Self::Rg8 | Self::R16 => 2,
            Self::R8 => 1,
        }
    }
    /// Channels stored.
    pub fn channels(self) -> usize {
        match self {
            Self::Bc4 | Self::R8 | Self::R16 => 1,
            Self::Bc5 | Self::Rg8 => 2,
            _ => 4,
        }
    }
    pub fn is_srgb(self) -> bool {
        matches!(self, Self::Bc7Srgb | Self::Rgba8Srgb)
    }
}

/// Bytes of one mip level of one face (BCn levels are padded to whole 4x4 blocks).
pub fn mip_bytes(fmt: TexFormat, size: u32, mip: u32) -> usize {
    if fmt.is_block() {
        let s = (size >> mip).max(4) as usize;
        (s / 4) * (s / 4) * fmt.block_bytes()
    } else {
        let s = (size >> mip).max(1) as usize;
        s * s * fmt.block_bytes()
    }
}

pub fn layer_bytes(fmt: TexFormat, size: u32, mips: u32) -> usize {
    (0..mips).map(|m| mip_bytes(fmt, size, m)).sum()
}

#[derive(Clone, Debug)]
pub struct Texture {
    pub name: String,
    pub format: TexFormat,
    pub size: u32,
    pub mips: u32,
    pub data: Vec<u8>,
}

/// CPU height cube (Vulkan cube convention), used for the mesh displacement.
#[derive(Clone, Debug)]
pub struct HeightCube {
    pub size: u32,
    pub texels: Vec<u16>,
    pub scale: f64,
    pub offset: f64,
}

impl HeightCube {
    fn texel(&self, face: usize, x: i64, y: i64) -> f64 {
        let s = self.size as i64;
        let (x, y) = (x.clamp(0, s - 1), y.clamp(0, s - 1));
        self.texels[(face as i64 * s * s + y * s + x) as usize] as f64 * self.scale + self.offset
    }
    /// Bilinear height (m) in direction `d`. A pure function of the bits of `d` (deterministic face
    /// selection), so equal directions always give equal heights.
    pub fn sample(&self, d: DVec3) -> f64 {
        let (f, s, t) = crate::cube::vk_cube_st(d);
        let n = self.size as f64;
        let (x, y) = (s * n - 0.5, t * n - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let h00 = self.texel(f, x0, y0);
        let h10 = self.texel(f, x0 + 1, y0);
        let h01 = self.texel(f, x0, y0 + 1);
        let h11 = self.texel(f, x0 + 1, y0 + 1);
        (h00 * (1.0 - fx) + h10 * fx) * (1.0 - fy) + (h01 * (1.0 - fx) + h11 * fx) * fy
    }
}

#[derive(Clone, Debug)]
pub struct PlanetPack {
    pub textures: Vec<Texture>,
    pub height: HeightCube,
    pub credits: String,
}

impl PlanetPack {
    pub fn texture(&self, name: &str) -> Option<&Texture> {
        self.textures.iter().find(|t| t.name == name)
    }
    pub fn gpu_bytes(&self) -> usize {
        self.textures.iter().map(|t| t.data.len()).sum()
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        let tex: Vec<TableB> = self
            .textures
            .iter()
            .map(|t| {
                TableB::default()
                    .with(0, Field::U32(t.format as u32))
                    .with(1, Field::U32(t.size))
                    .with(2, Field::U32(t.mips))
                    .with(3, Field::Bytes(t.data.clone(), 256))
                    .with(4, Field::Bytes(t.name.as_bytes().to_vec(), 1))
            })
            .collect();
        let h = TableB::default()
            .with(0, Field::U32(self.height.size))
            .with(1, Field::Bytes(bytemuck::cast_slice(&self.height.texels).to_vec(), 256))
            .with(2, Field::F64(self.height.scale))
            .with(3, Field::F64(self.height.offset));
        let root = TableB::default()
            .with(0, Field::U32(VERSION))
            .with(1, Field::Tables(tex))
            .with(2, Field::Table(Box::new(h)))
            .with(3, Field::Bytes(self.credits.as_bytes().to_vec(), 1));
        fb::finish(&root)
    }
    /// Verified read (bounds, alignment, sizes consistent with format / size / mips).
    pub fn from_bytes(buf: &[u8]) -> Result<Self, VerifyError> {
        let bad = |field| VerifyError::BadField { table: "PLN1", field };
        let mut v = Verifier::new(buf, Limits { max_bytes: usize::MAX, ..Limits::default() });
        let root = v.root()?;
        v.scalar(&root, 0, 4, 4)?;
        let ver = root.u32(0, 0);
        if ver != VERSION {
            return Err(VerifyError::UnsupportedVersion { major: ver as u16 });
        }
        let n = v.tables(&root, 1)?;
        let mut textures = Vec::with_capacity(n);
        for i in 0..n {
            let t = root.tables_get(1, i).ok_or(bad(1))?;
            let t = v.table_at(t.pos, 1)?;
            for f in 0..3 {
                v.scalar(&t, f, 4, 4)?;
            }
            let format = TexFormat::from_u32(t.u32(0, 0)).ok_or(bad(10))?;
            let (size, mips) = (t.u32(1, 0), t.u32(2, 0));
            if size == 0 || !size.is_power_of_two() || mips == 0 || mips > 16 || (size >> (mips - 1)) < 4 {
                return Err(bad(11));
            }
            let data = v.bytes(&t, 3, 1, 256)?.ok_or(bad(13))?;
            if data.len() != 6 * layer_bytes(format, size, mips) {
                return Err(bad(13));
            }
            let name = v.bytes(&t, 4, 1, 1)?.map(|b| String::from_utf8_lossy(b).into_owned()).unwrap_or_default();
            textures.push(Texture { name, format, size, mips, data: data.to_vec() });
        }
        let h = v.table(&root, 2, 0)?.ok_or(bad(2))?;
        v.scalar(&h, 0, 4, 4)?;
        v.scalar(&h, 2, 8, 8)?;
        v.scalar(&h, 3, 8, 8)?;
        let size = h.u32(0, 0);
        let hb = v.bytes(&h, 1, 2, 256)?.ok_or(bad(21))?;
        if size == 0 || hb.len() != 6 * size as usize * size as usize * 2 {
            return Err(bad(21));
        }
        let height = HeightCube { size, texels: bytemuck::pod_collect_to_vec(hb), scale: h.f64(2, 1.0), offset: h.f64(3, 0.0) };
        let credits = v.bytes(&root, 3, 1, 1)?.map(|b| String::from_utf8_lossy(b).into_owned()).unwrap_or_default();
        Ok(PlanetPack { textures, height, credits })
    }
}
