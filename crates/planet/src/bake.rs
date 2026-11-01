//! Earth bake: NASA / Natural Earth equirectangular sources -> cube-map textures (Vulkan layer
//! order, BC7 / BC5 / BC4, box-filtered mips down to 4x4) and the CPU height cube.
//!
//! Textures:
//! - `albedo` BC7 sRGB: Blue Marble NG surface reflectance (RGB) + water fraction (A, linear)
//!   from the Natural Earth land mask (lakes cut out).
//! - `slope` BC5: east / north surface slope from the GEBCO_08-derived topography (blurred by
//!   ~1 source px against its 25 m quantisation), stored as `sign(s) sqrt(|s| / SLOPE_MAX)`.
//! - `night` BC4: Black Marble 2016 (VIIRS DNB) grayscale.
//! - `aux` BC7 linear: R cloud reflectance (Blue Marble clouds), G ice fraction (Natural Earth
//!   glaciers + ice shelves), B ocean depth / 8000 m (GEBCO_08 bathymetry), A 1.
use crate::cube::{dir_latlon, vk_cube_dir};
use crate::pack::{HeightCube, PlanetPack, TexFormat, Texture};
use glam::DVec3;
use rayon::prelude::*;
use std::path::Path;

/// Slope magnitude stored at full scale.
pub const SLOPE_MAX: f64 = 0.6;
/// Topography scale of the source PNG (0..255 -> 0..6400 m).
pub const TOPO_M: f64 = 6400.0;

/// One equirectangular channel set, `ch` interleaved channels: u8 as decoded (exactly the
/// `v / 255.0` the f32 path stores, at a quarter of the memory), f32 once filtered ([`Equi::blur`]).
pub struct Equi {
    pub w: usize,
    pub h: usize,
    pub ch: usize,
    pub data: Vec<f32>,
    pub bytes: Vec<u8>,
}

impl Equi {
    pub fn from_u8(w: usize, h: usize, ch: usize, src: &[u8], src_ch: usize, take: &[usize]) -> Self {
        let mut bytes = vec![0u8; w * h * ch];
        bytes.par_chunks_mut(w * ch).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                for (k, &c) in take.iter().enumerate() {
                    row[x * ch + k] = src[(y * w + x) * src_ch + c];
                }
            }
        });
        Equi { w, h, ch, data: Vec::new(), bytes }
    }
    #[inline]
    fn at(&self, i: usize) -> f32 {
        if self.data.is_empty() { self.bytes[i] as f32 / 255.0 } else { self.data[i] }
    }
    /// Bilinear sample at a direction (longitude wraps, latitude clamps); channel `c`.
    pub fn sample(&self, d: DVec3, c: usize) -> f32 {
        let (lat, lon) = dir_latlon(d);
        let x = (lon + 180.0) / 360.0 * self.w as f64 - 0.5;
        let y = (90.0 - lat) / 180.0 * self.h as f64 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = ((x - x0) as f32, (y - y0) as f32);
        let w = self.w as i64;
        let xi = |x: i64| x.rem_euclid(w) as usize;
        let yi = |y: i64| y.clamp(0, self.h as i64 - 1) as usize;
        let (x0, y0) = (x0 as i64, y0 as i64);
        let t = |x: i64, y: i64| self.at((yi(y) * self.w + xi(x)) * self.ch + c);
        let a = t(x0, y0) * (1.0 - fx) + t(x0 + 1, y0) * fx;
        let b = t(x0, y0 + 1) * (1.0 - fx) + t(x0 + 1, y0 + 1) * fx;
        a * (1.0 - fy) + b * fy
    }
    /// Separable [1 2 1]/4 blur, `passes` times (channel 0 only; x wraps).
    pub fn blur(&mut self, passes: usize) {
        let (w, h) = (self.w, self.h);
        assert_eq!(self.ch, 1);
        if self.data.is_empty() {
            self.data = self.bytes.par_iter().map(|&b| b as f32 / 255.0).collect();
            self.bytes = Vec::new();
        }
        for _ in 0..passes {
            let src = self.data.clone();
            self.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                for x in 0..w {
                    let l = src[y * w + (x + w - 1) % w];
                    let r = src[y * w + (x + 1) % w];
                    row[x] = 0.25 * l + 0.5 * src[y * w + x] + 0.25 * r;
                }
            });
            let src = self.data.clone();
            self.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                let (u, d) = (y.saturating_sub(1), (y + 1).min(h - 1));
                for x in 0..w {
                    row[x] = 0.25 * src[u * w + x] + 0.5 * src[y * w + x] + 0.25 * src[d * w + x];
                }
            });
        }
    }
}

pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}
pub fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// Texel (x, y) direction of Vulkan cube face `f` at size `n`, sub-sample (sx, sy) of `ss`.
fn texel_dir(f: usize, n: usize, x: usize, y: usize, sx: usize, sy: usize, ss: usize) -> DVec3 {
    let s = (x as f64 + (sx as f64 + 0.5) / ss as f64) / n as f64;
    let t = (y as f64 + (sy as f64 + 0.5) / ss as f64) / n as f64;
    vk_cube_dir(f, s, t).normalize()
}

/// Render face `f` of a cube texture with `ss x ss` supersampling (linear values).
pub fn face<F: Fn(DVec3) -> [f32; 4] + Sync>(f: usize, n: usize, ss: usize, texel: &F) -> Vec<[f32; 4]> {
    let mut out = vec![[0.0f32; 4]; n * n];
    out.par_chunks_mut(n).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let mut acc = [0.0f32; 4];
            for sy in 0..ss {
                for sx in 0..ss {
                    let v = texel(texel_dir(f, n, x, y, sx, sy, ss));
                    for k in 0..4 {
                        acc[k] += v[k];
                    }
                }
            }
            let inv = 1.0 / (ss * ss) as f32;
            *o = acc.map(|a| a * inv);
        }
    });
    out
}

fn downsample(src: &[[f32; 4]], n: usize) -> Vec<[f32; 4]> {
    let m = n / 2;
    let mut out = vec![[0.0f32; 4]; m * m];
    out.par_chunks_mut(m).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let p = |dx: usize, dy: usize| src[(2 * y + dy) * n + 2 * x + dx];
            let (a, b, c, d) = (p(0, 0), p(1, 0), p(0, 1), p(1, 1));
            *o = [0, 1, 2, 3].map(|k| 0.25 * (a[k] + b[k] + c[k] + d[k]));
        }
    });
    out
}

/// Encode one mip (`n x n`, linear values) to `fmt` (BCn blocks or raw texels); sRGB formats
/// store RGB sRGB-encoded (alpha linear).
fn encode(fmt: TexFormat, img: &[[f32; 4]], n: usize) -> Vec<u8> {
    let srgb = fmt.is_srgb();
    let comps = fmt.channels();
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    if fmt == TexFormat::R16 {
        return img.iter().flat_map(|p| (((p[0].clamp(0.0, 1.0) * 65535.0) + 0.5) as u16).to_le_bytes()).collect();
    }
    let bytes: Vec<u8> = img
        .par_iter()
        .flat_map_iter(|p| {
            let v: [u8; 4] = if srgb {
                [q(linear_to_srgb(p[0])), q(linear_to_srgb(p[1])), q(linear_to_srgb(p[2])), q(p[3])]
            } else {
                p.map(q)
            };
            v.into_iter().take(comps)
        })
        .collect();
    if !fmt.is_block() {
        return bytes;
    }
    let rows = 64.min(n);
    let chunks: Vec<Vec<u8>> = bytes
        .par_chunks(n * comps * rows)
        .map(|c| {
            let h = (c.len() / (n * comps)) as u32;
            let (w, stride) = (n as u32, (n * comps) as u32);
            match fmt {
                TexFormat::Bc4 => {
                    intel_tex_2::bc4::compress_blocks(&intel_tex_2::RSurface { data: c, width: w, height: h, stride })
                }
                TexFormat::Bc5 => {
                    intel_tex_2::bc5::compress_blocks(&intel_tex_2::RgSurface { data: c, width: w, height: h, stride })
                }
                _ => {
                    let s = bc7_settings();
                    intel_tex_2::bc7::compress_blocks(
                        &s,
                        &intel_tex_2::RgbaSurface { data: c, width: w, height: h, stride },
                    )
                }
            }
        })
        .collect();
    chunks.concat()
}

/// BC7-encode one `n x n` RGBA8 image (bytes as stored: sRGB-encoded rgb for a BC7 sRGB
/// target, linear for BC7 UNORM; the encoder treats both alike). Earth round 5: the graft.
pub fn encode_bc7_rgba8(bytes: &[u8], n: usize) -> Vec<u8> {
    assert!(n >= 4 && n.is_multiple_of(4) && bytes.len() == n * n * 4, "BC7 needs a multiple-of-4 RGBA8 image");
    let rows = 64.min(n);
    let s = bc7_settings();
    let chunks: Vec<Vec<u8>> = bytes
        .par_chunks(n * 4 * rows)
        .map(|c| {
            intel_tex_2::bc7::compress_blocks(
                &s,
                &intel_tex_2::RgbaSurface {
                    data: c,
                    width: n as u32,
                    height: (c.len() / (n * 4)) as u32,
                    stride: (n * 4) as u32,
                },
            )
        })
        .collect();
    chunks.concat()
}

/// BC7 encoder preset: `BAKE_BC7=basic|fast|very_fast` (default very_fast, the round-1 setting).
fn bc7_settings() -> intel_tex_2::bc7::EncodeSettings {
    match std::env::var("BAKE_BC7").as_deref() {
        Ok("basic") => intel_tex_2::bc7::alpha_basic_settings(),
        Ok("fast") => intel_tex_2::bc7::alpha_fast_settings(),
        _ => intel_tex_2::bc7::alpha_very_fast_settings(),
    }
}

/// Bake a cube texture from a texel function (linear values in [0, 1]).
pub fn bake_cube<F: Fn(DVec3) -> [f32; 4] + Sync>(
    name: &str,
    fmt: TexFormat,
    size: usize,
    ss: usize,
    texel: &F,
) -> Texture {
    bake_cube_multi(name, &[fmt], size, ss, texel).pop().unwrap()
}

/// Bake once, encode to several formats (same texels, same box-filtered mips down to 4x4): the
/// BCn and the uncompressed reference variants differ only by the block encoding.
pub fn bake_cube_multi<F: Fn(DVec3) -> [f32; 4] + Sync>(
    name: &str,
    fmts: &[TexFormat],
    size: usize,
    ss: usize,
    texel: &F,
) -> Vec<Texture> {
    let mips = size.trailing_zeros() - 1; // down to 4x4
    let mut data = vec![Vec::new(); fmts.len()];
    for f in 0..6 {
        let mut img = face(f, size, ss, texel);
        let mut n = size;
        for m in 0..mips {
            for (d, &fmt) in data.iter_mut().zip(fmts) {
                d.extend(encode(fmt, &img, n));
            }
            if m + 1 < mips {
                img = downsample(&img, n);
                n /= 2;
            }
        }
    }
    fmts.iter()
        .zip(data)
        .map(|(&format, data)| Texture { name: name.into(), format, size: size as u32, mips, data })
        .collect()
}

/// Sources loaded from a `fetch_earth_data.sh` directory.
pub struct Sources {
    pub albedo: Equi,
    pub land_ice: Equi,
    pub topo: Equi,
    pub bath: Equi,
    pub night: Equi,
    pub clouds: Equi,
    /// Which cloud map `clouds` came from: `viirs` (dated, `clouds_viirs.png`, the default when
    /// present) or `bmng` (Blue Marble clouds); `BAKE_CLOUD_SRC` overrides.
    pub clouds_src: &'static str,
    /// Cloud-top height (km x 20 / 255; `cth_viirs.png`, same VIIRS pass), only with `viirs`.
    pub cth: Option<Equi>,
}

fn load(dir: &Path, name: &str) -> image::DynamicImage {
    let p = dir.join(name);
    let fail = |e: &dyn std::fmt::Display| -> ! { panic!("{}: {e} (run tools/fetch_earth_data.sh)", p.display()) };
    let mut r =
        image::ImageReader::open(&p).unwrap_or_else(|e| fail(&e)).with_guessed_format().unwrap_or_else(|e| fail(&e));
    r.no_limits();
    r.decode().unwrap_or_else(|e| fail(&e))
}

impl Sources {
    pub fn load(dir: &Path) -> Self {
        let rgb = |n: &str| {
            let i = load(dir, n).into_rgb8();
            (i.width() as usize, i.height() as usize, i.into_raw())
        };
        let (w, h, a) = rgb("bmng_july.jpg");
        let albedo = Equi::from_u8(w, h, 3, &a, 3, &[0, 1, 2]);
        drop(a);
        let (w, h, m) = rgb("mask.png");
        let land_ice = Equi::from_u8(w, h, 2, &m, 3, &[0, 1]);
        drop(m);
        let (w, h, t) = rgb("topo.png");
        let mut topo = Equi::from_u8(w, h, 1, &t, 3, &[0]);
        drop(t);
        topo.blur(2);
        let (w, h, b) = rgb("bath.png");
        let bath = Equi::from_u8(w, h, 1, &b, 3, &[0]);
        let (w, h, n) = rgb("night.tif");
        let night = Equi::from_u8(w, h, 1, &n, 3, &[0]);
        let clouds_src = match std::env::var("BAKE_CLOUD_SRC").ok().as_deref() {
            Some("bmng") => "bmng",
            Some("viirs") => "viirs",
            Some(o) => panic!("BAKE_CLOUD_SRC={o}: expected viirs | bmng"),
            None if dir.join("clouds_viirs.png").exists() => "viirs",
            None => "bmng",
        };
        let (w, h, c) = rgb(if clouds_src == "viirs" { "clouds_viirs.png" } else { "clouds.tif" });
        let clouds = Equi::from_u8(w, h, 1, &c, 3, &[0]);
        let cth = (clouds_src == "viirs" && dir.join("cth_viirs.png").exists()).then(|| {
            let (w, h, c) = rgb("cth_viirs.png");
            Equi::from_u8(w, h, 1, &c, 3, &[0])
        });
        Sources { albedo, land_ice, topo, bath, night, clouds, clouds_src, cth }
    }
}

/// East / north unit tangents at `d` (planet frame, +Z north); at the poles east = +Y.
pub fn east_north(d: DVec3) -> (DVec3, DVec3) {
    let e = DVec3::Z.cross(d);
    let e = if e.length_squared() < 1e-20 { DVec3::Y } else { e.normalize() };
    (e, d.cross(e))
}

/// Encoded slope value of a slope component.
pub fn slope_enc(s: f64) -> f32 {
    (s.signum() * (s.abs() / SLOPE_MAX).min(1.0).sqrt()) as f32 * 0.5 + 0.5
}

pub struct BakeConfig {
    pub albedo: usize,
    pub slope: usize,
    pub night: usize,
    pub aux: usize,
    /// KTX2 `clouds` face size (PLN1 keeps them in `aux`).
    pub clouds: usize,
    pub height: usize,
    pub ss: usize,
}

impl Default for BakeConfig {
    fn default() -> Self {
        BakeConfig { albedo: 4096, slope: 4096, night: 4096, aux: 2048, clouds: 4096, height: 1024, ss: 2 }
    }
}

pub const CREDITS: &str = "Blue Marble Next Generation (July 2004), R. Stöckli, E. Vermote, N. Saleous, R. Simmon and D. Herring, NASA Earth Observatory / GSFC (public domain). \
Topography and bathymetry: Jesse Allen, NASA Earth Observatory, from GEBCO_08 (British Oceanographic Data Centre). \
Night lights: Black Marble 2016, NASA Earth Observatory, Suomi NPP VIIRS DNB (M. Román, NASA GSFC). \
Clouds: Blue Marble clouds, NASA GSFC / MODIS (PLN1, BAKE_CLOUD_SRC=bmng), or NOAA-20 VIIRS Corrected Reflectance 2026-07-15, NASA GIBS / EOSDIS (KTX2 default). Land, lakes, glaciers, ice shelves: Natural Earth 1:10m (public domain).";

/// One KTX2 output set of [`bake_ktx2`].
pub struct Ktx2Set {
    pub dir: std::path::PathBuf,
    /// Uncompressed reference formats instead of BCn.
    pub raw: bool,
    pub sc: crate::ktx2::Supercompression,
}

/// KTX2 file names of the Earth set (also the renderer's binding order).
pub const KTX2_FILES: [&str; 5] = ["albedo", "slope", "night", "clouds", "masks"];

/// Bake the Earth cube maps to KTX2 sets (one sampling pass per texture, encoded per set):
/// `albedo` BC7 sRGB (rgb + water a), `slope` BC5, `night` BC4, `clouds` BC5 (4096: r coverage of the cloud
/// volume, g cloud-top height km x 20 / 255, 0 = unknown), `masks` BC5 (r ice, g ocean depth), `height` R16 (CPU displacement, one
/// level, decimetres; keys `planet.height_scale_m` / `planet.height_offset_m`). Raw sets use
/// RGBA8 sRGB / RG8 / R8 / RG8 / RG8. Returns per-file (name, set, file bytes, GPU bytes).
pub fn bake_ktx2(
    src: &Sources,
    cfg: &BakeConfig,
    sets: &[Ktx2Set],
    log: &mut dyn FnMut(&str),
) -> std::io::Result<Vec<(String, usize, usize, usize)>> {
    use crate::ktx2::write;
    let t0 = std::time::Instant::now();
    for s in sets {
        std::fs::create_dir_all(&s.dir)?;
    }
    let mut report = Vec::new();
    let mut emit = |name: &str, texs: Vec<Texture>, log: &mut dyn FnMut(&str)| -> std::io::Result<()> {
        for (i, (t, s)) in texs.iter().zip(sets).enumerate() {
            let kv: Vec<(&str, &[u8])> = if name == "albedo" || name == "height" {
                vec![("planet.credits", CREDITS.as_bytes())]
            } else {
                vec![]
            };
            let mut kv = kv;
            if name == "clouds" {
                kv.push(("planet.clouds_source", src.clouds_src.as_bytes()));
            }
            if name == "height" {
                kv.push(("planet.height_scale_m", b"0.1"));
                kv.push(("planet.height_offset_m", b"0"));
            }
            let bytes = write(t, s.sc, &kv).map_err(std::io::Error::other)?;
            std::fs::write(s.dir.join(format!("{name}.ktx2")), &bytes)?;
            log(&format!(
                "{name} -> {} ({:?} {:?}): file {:.1} MB, GPU {:.1} MB, {:.1} s",
                s.dir.display(),
                t.format,
                s.sc,
                bytes.len() as f64 / 1048576.0,
                t.data.len() as f64 / 1048576.0,
                t0.elapsed().as_secs_f64()
            ));
            report.push((name.to_string(), i, bytes.len(), t.data.len()));
        }
        Ok(())
    };
    let pick = |bc: TexFormat, raw: TexFormat| sets.iter().map(|s| if s.raw { raw } else { bc }).collect::<Vec<_>>();
    let ss = cfg.ss;
    let t = bake_cube_multi("albedo", &pick(TexFormat::Bc7Srgb, TexFormat::Rgba8Srgb), cfg.albedo, ss, &|d| {
        let c = |k| srgb_to_linear(src.albedo.sample(d, k));
        [c(0), c(1), c(2), 1.0 - src.land_ice.sample(d, 0)]
    });
    emit("albedo", t, log)?;
    let r = crate::EARTH_RADIUS_M;
    let step = std::f64::consts::FRAC_PI_2 / cfg.slope as f64 * 0.75;
    let t = bake_cube_multi("slope", &pick(TexFormat::Bc5, TexFormat::Rg8), cfg.slope, ss, &|d| {
        let (e, n) = east_north(d);
        let h = |p: DVec3| src.topo.sample(p.normalize(), 0) as f64 * TOPO_M;
        let ge = (h(d + e * step) - h(d - e * step)) / (2.0 * step * r);
        let gn = (h(d + n * step) - h(d - n * step)) / (2.0 * step * r);
        [slope_enc(ge), slope_enc(gn), 0.0, 0.0]
    });
    emit("slope", t, log)?;
    let t = bake_cube_multi("night", &pick(TexFormat::Bc4, TexFormat::R8), cfg.night, ss, &|d| {
        [src.night.sample(d, 0), 0.0, 0.0, 0.0]
    });
    emit("night", t, log)?;
    // r coverage, g cloud-top height (0 = unknown; at least one code above 0 where known)
    let t = bake_cube_multi("clouds", &pick(TexFormat::Bc5, TexFormat::Rg8), cfg.clouds, ss, &|d| {
        let top = src.cth.as_ref().map_or(0.0, |c| c.sample(d, 0).max(1.5 / 255.0));
        [src.clouds.sample(d, 0), top, 0.0, 0.0]
    });
    emit("clouds", t, log)?;
    let t = bake_cube_multi("masks", &pick(TexFormat::Bc5, TexFormat::Rg8), cfg.aux, ss, &|d| {
        let water = 1.0 - src.land_ice.sample(d, 0);
        let depth = if water > 0.0 { 1.0 - src.bath.sample(d, 0) } else { 0.0 };
        [src.land_ice.sample(d, 1), depth, 0.0, 0.0]
    });
    emit("masks", t, log)?;
    let h = height_cube(src, cfg);
    let data: Vec<u8> = h.texels.iter().flat_map(|v| v.to_le_bytes()).collect();
    let t = Texture { name: "height".into(), format: TexFormat::R16, size: h.size, mips: 1, data };
    emit("height", sets.iter().map(|_| t.clone()).collect(), log)?;
    Ok(report)
}

/// CPU height cube (decimetres, land only), as stored in PLN1 and `height.ktx2`.
pub fn height_cube(src: &Sources, cfg: &BakeConfig) -> HeightCube {
    let n = cfg.height;
    let mut texels = vec![0u16; 6 * n * n];
    for f in 0..6 {
        let img = face(f, n, cfg.ss, &|d| {
            let land = src.land_ice.sample(d, 0);
            [src.topo.sample(d, 0) * land, 0.0, 0.0, 0.0]
        });
        for (i, p) in img.iter().enumerate() {
            texels[f * n * n + i] = (p[0] as f64 * TOPO_M * 10.0).round().clamp(0.0, 65535.0) as u16;
        }
    }
    HeightCube { size: n as u32, texels, scale: 0.1, offset: 0.0 }
}

/// Height cube from `height.ktx2` (R16, one level; scale / offset from its keys).
pub fn height_from_ktx2(k: &crate::ktx2::Ktx2) -> Result<HeightCube, crate::ktx2::Ktx2Error> {
    use crate::ktx2::Ktx2Error;
    if k.format != TexFormat::R16 || k.mips != 1 {
        return Err(Ktx2Error::Inconsistent("height.ktx2 must be R16, one level"));
    }
    let num =
        |key: &str, d: f64| k.value(key).and_then(|v| std::str::from_utf8(v).ok()?.trim().parse().ok()).unwrap_or(d);
    let data = k.decode_level_major()?;
    let texels = data.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect();
    Ok(HeightCube {
        size: k.size,
        texels,
        scale: num("planet.height_scale_m", 0.1),
        offset: num("planet.height_offset_m", 0.0),
    })
}

pub fn bake(src: &Sources, cfg: &BakeConfig, log: &mut dyn FnMut(&str)) -> PlanetPack {
    let t0 = std::time::Instant::now();
    let mut textures = Vec::new();
    let albedo = bake_cube("albedo", TexFormat::Bc7Srgb, cfg.albedo, cfg.ss, &|d| {
        let c = |k| srgb_to_linear(src.albedo.sample(d, k));
        [c(0), c(1), c(2), 1.0 - src.land_ice.sample(d, 0)]
    });
    log(&format!("albedo {} MB, {:.1} s", albedo.data.len() >> 20, t0.elapsed().as_secs_f64()));
    textures.push(albedo);
    let r = crate::EARTH_RADIUS_M;
    let step = std::f64::consts::FRAC_PI_2 / cfg.slope as f64 * 0.75;
    let slope = bake_cube("slope", TexFormat::Bc5, cfg.slope, cfg.ss, &|d| {
        let (e, n) = east_north(d);
        let h = |p: DVec3| src.topo.sample(p.normalize(), 0) as f64 * TOPO_M;
        let ge = (h(d + e * step) - h(d - e * step)) / (2.0 * step * r);
        let gn = (h(d + n * step) - h(d - n * step)) / (2.0 * step * r);
        [slope_enc(ge), slope_enc(gn), 0.0, 0.0]
    });
    log(&format!("slope {} MB, {:.1} s", slope.data.len() >> 20, t0.elapsed().as_secs_f64()));
    textures.push(slope);
    let night = bake_cube("night", TexFormat::Bc4, cfg.night, cfg.ss, &|d| [src.night.sample(d, 0), 0.0, 0.0, 0.0]);
    log(&format!("night {} MB, {:.1} s", night.data.len() >> 20, t0.elapsed().as_secs_f64()));
    textures.push(night);
    let aux = bake_cube("aux", TexFormat::Bc7, cfg.aux, cfg.ss, &|d| {
        let water = 1.0 - src.land_ice.sample(d, 0);
        let depth = if water > 0.0 { 1.0 - src.bath.sample(d, 0) } else { 0.0 };
        [src.clouds.sample(d, 0), src.land_ice.sample(d, 1), depth, 1.0]
    });
    log(&format!("aux {} MB, {:.1} s", aux.data.len() >> 20, t0.elapsed().as_secs_f64()));
    textures.push(aux);
    let height = height_cube(src, cfg);
    log(&format!("height {} MB, {:.1} s", (height.texels.len() * 2) >> 20, t0.elapsed().as_secs_f64()));
    PlanetPack { textures, height, credits: CREDITS.into() }
}
