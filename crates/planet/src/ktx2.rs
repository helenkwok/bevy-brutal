//! KTX2 (Khronos Texture 2.0) cube maps: writer and verifying reader for the Earth textures.
//!
//! - Formats: BC7 (sRGB / UNORM), BC5, BC4, and the uncompressed reference formats RGBA8 (sRGB /
//!   UNORM), RG8, R8 and R16 (the CPU height cube); `vkFormat` is the Vulkan enum, so the loader
//!   creates the image straight from the header.
//! - Supercompression: none or Zstandard (scheme 2), one zstd frame per mip level. The GPU still
//!   sees the BCn blocks: zstd is a disk / streaming layer that is undone into the staging buffer.
//! - Layout per the KTX2 spec: level index (byteOffset, byteLength, uncompressedByteLength) for
//!   level 0 (largest) .. n-1; the level data is stored smallest first ("mip padding" order);
//!   inside a level the six faces +X, -X, +Y, -Y, +Z, -Z follow each other (Vulkan layer order).
//! - Data Format Descriptor: one Khronos basic block (colour model, primaries BT.709, transfer
//!   sRGB / linear, block size, one sample per channel). Key/value data: `KTXwriter` plus caller
//!   keys (sorted, as the spec requires). Files are checked with the Khronos `ktx validate`.
//!
//! Earth round 5: also 2D arrays (faceCount 1, layerCount n, e.g. the graft's two nested
//! levels) through [`write_array`]; [`Ktx2::images`] is 6 for a cube, n for an array.
//!
//! [`Ktx2::parse`] validates every offset / length against the buffer and the format before
//! anything is read (a corrupt or truncated file is an error, never a panic or an OOB read).
use crate::pack::{TexFormat, Texture, mip_bytes};

pub const IDENTIFIER: [u8; 12] = [0xAB, b'K', b'T', b'X', b' ', b'2', b'0', 0xBB, 0x0D, 0x0A, 0x1A, 0x0A];
pub const SUPERCOMPRESSION_NONE: u32 = 0;
pub const SUPERCOMPRESSION_ZSTD: u32 = 2;
const HEADER: usize = 80;
const LEVEL_ENTRY: usize = 24;

/// Vulkan `VkFormat` value of a texture format.
pub fn vk_format(f: TexFormat) -> u32 {
    match f {
        TexFormat::Bc7Srgb => 146,
        TexFormat::Bc7 => 145,
        TexFormat::Bc5 => 141,
        TexFormat::Bc4 => 139,
        TexFormat::Rgba8Srgb => 43,
        TexFormat::Rgba8 => 37,
        TexFormat::Rg8 => 16,
        TexFormat::R8 => 9,
        TexFormat::R16 => 70,
    }
}
pub fn from_vk_format(v: u32) -> Option<TexFormat> {
    [TexFormat::Bc7Srgb, TexFormat::Bc7, TexFormat::Bc5, TexFormat::Bc4, TexFormat::Rgba8Srgb, TexFormat::Rgba8, TexFormat::Rg8, TexFormat::R8, TexFormat::R16]
        .into_iter()
        .find(|&f| vk_format(f) == v)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Supercompression {
    None,
    /// Zstandard at this level (1..=22).
    Zstd(i32),
}

#[derive(Debug, PartialEq, Eq)]
pub enum Ktx2Error {
    BadIdentifier,
    Truncated(&'static str),
    Unsupported(&'static str),
    Inconsistent(&'static str),
    Zstd(String),
}
impl std::fmt::Display for Ktx2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KTX2: {self:?}")
    }
}
impl std::error::Error for Ktx2Error {}

/// Bytes of one level (all six faces).
pub fn level_bytes(fmt: TexFormat, size: u32, level: u32) -> usize {
    6 * mip_bytes(fmt, size, level)
}

fn dfd(fmt: TexFormat) -> Vec<u8> {
    // (colour model, block dims - 1, bytes per block / texel, samples: (bit offset, bit length, channel, upper))
    let bc = |m: u8, ch: &[(u16, u8, u8)], bytes: u8| (m, [3u8, 3, 0, 0], bytes, ch.iter().map(|&(o, l, c)| (o, l, c, u32::MAX)).collect::<Vec<_>>());
    let raw = |ch: &[u8], bits: u8| {
        let up = if bits == 8 { 255u32 } else { 65535 };
        (1u8, [0u8; 4], (ch.len() as u8) * bits / 8, ch.iter().enumerate().map(|(i, &c)| ((i as u16) * bits as u16, bits - 1, c, up)).collect::<Vec<_>>())
    };
    let (model, dims, bytes, mut samples) = match fmt {
        TexFormat::Bc7Srgb | TexFormat::Bc7 => bc(134, &[(0, 127, 0)], 16),
        TexFormat::Bc5 => bc(132, &[(0, 63, 0), (64, 63, 1)], 16),
        TexFormat::Bc4 => bc(131, &[(0, 63, 0)], 8),
        TexFormat::Rgba8Srgb | TexFormat::Rgba8 => raw(&[0, 1, 2, 15], 8),
        TexFormat::Rg8 => raw(&[0, 1], 8),
        TexFormat::R8 => raw(&[0], 8),
        TexFormat::R16 => raw(&[0], 16),
    };
    if fmt.is_srgb() && !fmt.is_block() {
        samples[3].2 |= 0x10; // KHR_DF_SAMPLE_DATATYPE_LINEAR: alpha is linear in an sRGB format
    }
    let transfer = if fmt.is_srgb() { 2u32 } else { 1 };
    let block = 24 + 16 * samples.len() as u32;
    let mut w = vec![
        0u32,                                            // vendorId 0 (Khronos), descriptorType 0 (basic)
        2 | (block << 16),                               // versionNumber 2, descriptorBlockSize
        model as u32 | (1 << 8) | (transfer << 16),      // model, primaries BT.709, transfer, flags 0
        u32::from_le_bytes(dims),
        bytes as u32,                                    // bytesPlane0 (sized also when supercompressed, spec >= 2.0.4)
        0,
    ];
    for (off, len, ch, up) in samples {
        w.push(off as u32 | ((len as u32) << 16) | ((ch as u32) << 24));
        w.push(0); // sample position 0
        w.push(0); // sampleLower
        w.push(up);
    }
    let mut out = (4 + 4 * w.len() as u32).to_le_bytes().to_vec();
    for x in w {
        out.extend(x.to_le_bytes());
    }
    out
}

/// Serialise a cube texture (layer-major data, as [`Texture`]) as KTX2.
pub fn write(t: &Texture, sc: Supercompression, kv: &[(&str, &[u8])]) -> Result<Vec<u8>, Ktx2Error> {
    write_images(t, 6, false, sc, kv)
}

/// Serialise a 2D array texture of `layers` images (layer-major data, as [`Texture`] but with
/// `layers` images instead of six faces) as KTX2 (faceCount 1, layerCount `layers`).
pub fn write_array(t: &Texture, layers: u32, sc: Supercompression, kv: &[(&str, &[u8])]) -> Result<Vec<u8>, Ktx2Error> {
    if layers == 0 {
        return Err(Ktx2Error::Inconsistent("array layer count"));
    }
    write_images(t, layers as usize, true, sc, kv)
}

fn write_images(t: &Texture, images: usize, array: bool, sc: Supercompression, kv: &[(&str, &[u8])]) -> Result<Vec<u8>, Ktx2Error> {
    let mut levels: Vec<Vec<u8>> = Vec::with_capacity(t.mips as usize);
    let layer: usize = (0..t.mips).map(|m| mip_bytes(t.format, t.size, m)).sum();
    if t.data.len() != images * layer {
        return Err(Ktx2Error::Inconsistent("texture data size"));
    }
    for m in 0..t.mips {
        let off: usize = (0..m).map(|k| mip_bytes(t.format, t.size, k)).sum();
        let mb = mip_bytes(t.format, t.size, m);
        let mut lv = Vec::with_capacity(images * mb);
        for f in 0..images {
            lv.extend_from_slice(&t.data[f * layer + off..f * layer + off + mb]);
        }
        levels.push(lv);
    }
    let uncompressed: Vec<usize> = levels.iter().map(|l| l.len()).collect();
    if let Supercompression::Zstd(q) = sc {
        use rayon::prelude::*;
        levels = levels
            .into_par_iter()
            .map(|l| {
                let mut c = zstd::bulk::Compressor::new(q).map_err(|e| Ktx2Error::Zstd(e.to_string()))?;
                if l.len() > (16 << 20) {
                    c.multithread(std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(4)).map_err(|e| Ktx2Error::Zstd(e.to_string()))?;
                }
                c.compress(&l).map_err(|e| Ktx2Error::Zstd(e.to_string()))
            })
            .collect::<Result<_, _>>()?;
    }
    let supercompressed = sc != Supercompression::None;
    let n = t.mips as usize;
    let dfd = dfd(t.format);
    let dfd_off = HEADER + LEVEL_ENTRY * n;
    // key/value data, sorted by key (byte order), each entry padded to 4
    let mut keys: Vec<(String, Vec<u8>)> = kv.iter().map(|(k, v)| (k.to_string(), v.to_vec())).collect();
    keys.push(("KTXwriter".into(), format!("bevy-brutal planet::ktx2 {}", env!("CARGO_PKG_VERSION")).into_bytes()));
    keys.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    let mut kvd = Vec::new();
    for (k, v) in &keys {
        let mut e = k.as_bytes().to_vec();
        e.push(0);
        e.extend_from_slice(v);
        if k == "KTXwriter" && v.last() != Some(&0) {
            e.push(0); // KTXwriter value is a NUL-terminated UTF-8 string
        }
        kvd.extend((e.len() as u32).to_le_bytes());
        kvd.extend(e);
        while kvd.len() % 4 != 0 {
            kvd.push(0);
        }
    }
    let kvd_off = dfd_off + dfd.len();
    let mut out = Vec::with_capacity(kvd_off + kvd.len() + levels.iter().map(|l| l.len() + 16).sum::<usize>());
    out.extend_from_slice(&IDENTIFIER);
    let type_size = if t.format == TexFormat::R16 { 2u32 } else { 1 };
    let (layer_count, face_count) = if array { (images as u32, 1) } else { (0, 6) };
    for v in [vk_format(t.format), type_size, t.size, t.size, 0, layer_count, face_count, t.mips, if supercompressed { SUPERCOMPRESSION_ZSTD } else { SUPERCOMPRESSION_NONE }] {
        out.extend(v.to_le_bytes());
    }
    for v in [dfd_off as u32, dfd.len() as u32, if kvd.is_empty() { 0 } else { kvd_off as u32 }, kvd.len() as u32] {
        out.extend(v.to_le_bytes());
    }
    out.extend(0u64.to_le_bytes()); // sgdByteOffset
    out.extend(0u64.to_le_bytes()); // sgdByteLength
    let index_pos = out.len();
    out.resize(index_pos + LEVEL_ENTRY * n, 0);
    out.extend(&dfd);
    out.extend(&kvd);
    // level alignment: lcm(block bytes, 4) uncompressed; 1 when supercompressed
    let align = if supercompressed { 1 } else { lcm(t.format.block_bytes(), 4) };
    for m in (0..n).rev() {
        while out.len() % align != 0 {
            out.push(0);
        }
        let off = out.len();
        out.extend(&levels[m]);
        let e = index_pos + LEVEL_ENTRY * m;
        out[e..e + 8].copy_from_slice(&(off as u64).to_le_bytes());
        out[e + 8..e + 16].copy_from_slice(&(levels[m].len() as u64).to_le_bytes());
        out[e + 16..e + 24].copy_from_slice(&(uncompressed[m] as u64).to_le_bytes());
    }
    Ok(out)
}

fn lcm(a: usize, b: usize) -> usize {
    let g = |mut a: usize, mut b: usize| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    a / g(a, b) * b
}

#[derive(Clone, Copy, Debug)]
pub struct Level {
    pub offset: usize,
    pub length: usize,
    pub uncompressed: usize,
}

/// A parsed, verified KTX2 cube map (or 2D array) borrowing the file bytes.
#[derive(Debug)]
pub struct Ktx2<'a> {
    pub format: TexFormat,
    pub size: u32,
    pub mips: u32,
    /// Images per level: 6 for a cube map, the layer count for a 2D array.
    pub images: u32,
    /// faceCount 1, layerCount >= 1 (a 2D array) rather than a cube.
    pub array: bool,
    pub supercompression: u32,
    pub levels: Vec<Level>,
    pub kv: Vec<(String, &'a [u8])>,
    bytes: &'a [u8],
}

fn u32_at(b: &[u8], o: usize) -> Result<u32, Ktx2Error> {
    b.get(o..o + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap())).ok_or(Ktx2Error::Truncated("u32"))
}
fn u64_at(b: &[u8], o: usize) -> Result<usize, Ktx2Error> {
    let v = b.get(o..o + 8).map(|s| u64::from_le_bytes(s.try_into().unwrap())).ok_or(Ktx2Error::Truncated("u64"))?;
    usize::try_from(v).map_err(|_| Ktx2Error::Inconsistent("offset overflow"))
}

impl<'a> Ktx2<'a> {
    pub fn parse(b: &'a [u8]) -> Result<Self, Ktx2Error> {
        if b.len() < HEADER || b[..12] != IDENTIFIER {
            return Err(if b.len() < 12 || b[..12] != IDENTIFIER { Ktx2Error::BadIdentifier } else { Ktx2Error::Truncated("header") });
        }
        let h = |i: usize| u32_at(b, 12 + 4 * i);
        let format = from_vk_format(h(0)?).ok_or(Ktx2Error::Unsupported("vkFormat"))?;
        let (w, hh, depth, layers, faces, mips, sc) = (h(2)?, h(3)?, h(4)?, h(5)?, h(6)?, h(7)?, h(8)?);
        let (images, array) = match (layers, faces) {
            (0, 6) => (6, false),
            (n @ 1..=64, 1) => (n, true),
            _ => return Err(Ktx2Error::Unsupported("only square cube maps (faceCount 6, no array) or 2D arrays (faceCount 1, 1-64 layers)")),
        };
        if w != hh || depth != 0 {
            return Err(Ktx2Error::Unsupported("only square images, no depth"));
        }
        if w == 0 || !w.is_power_of_two() || mips == 0 || mips > 16 || (w >> (mips - 1)) == 0 || (format.is_block() && (w >> (mips - 1)) < 4) {
            return Err(Ktx2Error::Inconsistent("size / levelCount"));
        }
        if sc != SUPERCOMPRESSION_NONE && sc != SUPERCOMPRESSION_ZSTD {
            return Err(Ktx2Error::Unsupported("supercompression scheme"));
        }
        let (kvd_off, kvd_len) = (u32_at(b, 56)? as usize, u32_at(b, 60)? as usize);
        let mut levels = Vec::with_capacity(mips as usize);
        for m in 0..mips as usize {
            let e = HEADER + LEVEL_ENTRY * m;
            let l = Level { offset: u64_at(b, e)?, length: u64_at(b, e + 8)?, uncompressed: u64_at(b, e + 16)? };
            let want = images as usize * mip_bytes(format, w, m as u32);
            if l.offset.checked_add(l.length).is_none_or(|end| end > b.len()) {
                return Err(Ktx2Error::Truncated("level data"));
            }
            if l.uncompressed != want || (sc == SUPERCOMPRESSION_NONE && l.length != want) {
                return Err(Ktx2Error::Inconsistent("level size does not match format / size"));
            }
            levels.push(l);
        }
        let mut kv = Vec::new();
        if kvd_len > 0 {
            let kvd = b.get(kvd_off..kvd_off.checked_add(kvd_len).ok_or(Ktx2Error::Truncated("kvd"))?).ok_or(Ktx2Error::Truncated("kvd"))?;
            let mut o = 0;
            while o + 4 <= kvd.len() {
                let n = u32_at(kvd, o)? as usize;
                let e = kvd.get(o + 4..o + 4 + n).ok_or(Ktx2Error::Truncated("kv entry"))?;
                let z = e.iter().position(|&c| c == 0).ok_or(Ktx2Error::Inconsistent("kv key not NUL-terminated"))?;
                kv.push((String::from_utf8_lossy(&e[..z]).into_owned(), &e[z + 1..]));
                o += 4 + n.div_ceil(4) * 4;
            }
        }
        Ok(Ktx2 { format, size: w, mips, images, array, supercompression: sc, levels, kv, bytes: b })
    }
    /// Value of a key (without a trailing NUL).
    pub fn value(&self, key: &str) -> Option<&'a [u8]> {
        self.kv.iter().find(|(k, _)| k == key).map(|(_, v)| v.strip_suffix(&[0]).unwrap_or(v))
    }
    /// Bytes of the level data in the file (compressed when supercompressed).
    pub fn file_level(&self, m: usize) -> &'a [u8] {
        let l = self.levels[m];
        &self.bytes[l.offset..l.offset + l.length]
    }
    /// Total uncompressed bytes (all levels) = GPU bytes.
    pub fn gpu_bytes(&self) -> usize {
        self.levels.iter().map(|l| l.uncompressed).sum()
    }
    /// Level `m` (six faces +X..-Z, or the array layers) into `out` (exactly `levels[m].uncompressed` bytes):
    /// copy, or zstd-decode directly into the destination (e.g. a mapped staging buffer).
    pub fn level_into(&self, m: usize, out: &mut [u8]) -> Result<(), Ktx2Error> {
        let l = self.levels[m];
        if out.len() != l.uncompressed {
            return Err(Ktx2Error::Inconsistent("destination size"));
        }
        let src = self.file_level(m);
        if self.supercompression == SUPERCOMPRESSION_NONE {
            out.copy_from_slice(src);
            return Ok(());
        }
        let n = zstd::bulk::decompress_to_buffer(src, out).map_err(|e| Ktx2Error::Zstd(e.to_string()))?;
        if n != l.uncompressed {
            return Err(Ktx2Error::Inconsistent("zstd level decoded to the wrong size"));
        }
        Ok(())
    }
    /// All levels decoded, level-major (level 0 first), as uploaded by the renderer.
    pub fn decode_level_major(&self) -> Result<Vec<u8>, Ktx2Error> {
        let mut out = vec![0u8; self.gpu_bytes()];
        let mut rest = out.as_mut_slice();
        for m in 0..self.mips as usize {
            let (a, b) = rest.split_at_mut(self.levels[m].uncompressed);
            self.level_into(m, a)?;
            rest = b;
        }
        Ok(out)
    }
    /// Convert to the layer-major [`Texture`] layout (CPU use: the height cube, tests).
    pub fn to_texture(&self, name: &str) -> Result<Texture, Ktx2Error> {
        let lm = self.decode_level_major()?;
        let layer: usize = (0..self.mips).map(|m| mip_bytes(self.format, self.size, m)).sum();
        let n = self.images as usize;
        let mut data = vec![0u8; n * layer];
        let mut lo = 0;
        for m in 0..self.mips {
            let mb = mip_bytes(self.format, self.size, m);
            let off: usize = (0..m).map(|k| mip_bytes(self.format, self.size, k)).sum();
            for f in 0..n {
                data[f * layer + off..f * layer + off + mb].copy_from_slice(&lm[lo + f * mb..lo + (f + 1) * mb]);
            }
            lo += n * mb;
        }
        Ok(Texture { name: name.into(), format: self.format, size: self.size, mips: self.mips, data })
    }
}
