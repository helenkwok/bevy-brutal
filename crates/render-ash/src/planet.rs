//! Planet-from-orbit renderer (`earth` scene): cube-sphere patches + BC cube-map textures +
//! atmosphere ray-marched from space (see `shaders/planet.slang`). Offscreen, returns linear
//! HDR RGBA (f32, alpha = transmittance toward space for the star pass, 0 on the planet).
//!
//! GPU-friendly layout: all patch vertices in one buffer (`VERTS` per patch slot), one shared
//! index buffer with the 16 stitching variants, one `vkCmdDrawIndexedIndirect` with a command per
//! visible patch (vertex offset = slot; the shader derives the patch from the vertex index, so no
//! `firstInstance` is needed).
use crate::context::{GpuBuffer, GpuImage, image_barrier, memory_barrier};
use crate::{AshBackend, BackendError};
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use planet::ktx2::Ktx2;
use planet::pack::{PlanetPack, TexFormat, Texture};
use planet::patch::{PVertex, VERTS};

const PLANET_VS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/planet_vs.spv"));
const PLANET_FS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/planet_fs.spv"));
const SKY_VS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/planet_sky_vs.spv"));
const SKY_FS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/planet_sky_fs.spv"));
const LUT_CS: [&[u8]; 4] = [
    include_bytes!(concat!(env!("OUT_DIR"), "/planet_transmittance.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/planet_multiscatter.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/planet_skyground.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/planet_irradiance.spv")),
];
/// Workgroups per LUT kernel: transmittance 256x64, multi-scatter 32x32, sky-at-ground 32x32,
/// irradiance 64.
const LUT_GROUPS: [(u32, u32); 4] = [(32, 8), (4, 4), (4, 4), (1, 1)];
/// float4 entries of the LUT buffer (see OFF_* in planet.slang).
const LUT_FLOAT4S: u64 = 18432 + 64;
const COLOR: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
const DEPTH: vk::Format = vk::Format::D32_SFLOAT;
const STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::from_raw(
    vk::ShaderStageFlags::VERTEX.as_raw()
        | vk::ShaderStageFlags::FRAGMENT.as_raw()
        | vk::ShaderStageFlags::COMPUTE.as_raw(),
);

/// == `PParams` in planet.slang.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PlanetParamsGpu {
    pub view_proj: [[f32; 4]; 4],
    pub cam_km: [f32; 4],
    pub sun: [f32; 4],
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub fwd: [f32; 4],
    pub planet: [f32; 4],
    pub ocean: [f32; 4],
    pub light: [f32; 4],
    pub atmo: [f32; 4],
    pub view: [f32; 4],
    /// x volume base (km), y default top at full cover (km), z extinction (1/km), w mode (0 = 2D layer, 1 = volume)
    pub cloud: [f32; 4],
    /// x view steps, y sun steps, z shape-noise period (km), w multiple-scattering gain
    pub cloud2: [f32; 4],
    /// x free-tropospheric aerosol scattering at sea level (1/km), y its scale height (km),
    /// z stratospheric (Junge layer, 20 +- 7 km) aerosol scattering at the peak (1/km), w
    /// procedural sub-texel land-detail gain below a 2.5 km pixel footprint (0 = off).
    /// All zero = Hillaire's clean boundary layer only.
    pub aero: [f32; 4],
    /// Clear-sky window (Earth round 3 descent): xyz unit direction of its centre, w its full-clear
    /// angular radius (rad; the coverage returns over w..2.2 w). w = 0: off.
    pub hole: [f32; 4],
    /// Far-footprint cloud structure (Earth round 3): optical-depth factors applied in the
    /// uniform-column regime only (pixel footprints > 0.6 km, full beyond 2.5 km; the ISS-scale
    /// regime is untouched). x = log2 range of the cloud-top-height factor (deep convective tops
    /// thicker, low decks thinner), y = log2 amplitude of a 256..32 km fBm, its octaves faded in
    /// by footprint. 0, 0 = off (round 2). z = log2 brightness bias; w = fine structure (round 4: a
    /// 16 km octave, higher persistence; 0 = round 3).
    pub meso: [f32; 4],
    /// Snow-free disc (Earth round 3 descent, the companion of the terrain flattening under the
    /// mountains layer: flattened glaciers read as low cloud): xyz centre direction scaled by the
    /// strength 0..1 (the flattening's), w outer angular radius (rad). Reflectance is capped at a
    /// valley forest's, the ice mask and slope-map relief fade out, fully inside 0.7 w. w = 0: off.
    pub thaw: [f32; 4],
    /// Shared Alps (Earth round 4, `planet::graft`): xyz the vista's nadir direction scaled by the
    /// graft strength 0..1, w the full-graft angular radius (rad). w = 0: off.
    pub graft: [f32; 4],
    /// xyz the vista's east, w the graft's outer angular radius (the real Earth beyond).
    pub graft_e: [f32; 4],
    /// xyz the vista's north, w the outer graft level's half extent (km).
    pub graft_n: [f32; 4],
    /// x inner level half extent (km), y albedo gain, z AO strength, w texels per level side.
    pub graft_l: [f32; 4],
    /// Round-4 far-footprint clouds: x = the coverage gamma at multi-km footprints (eased in over
    /// the meso regime's 0.6 .. 2.5 km; 0 = `planet.w` everywhere, round 3).
    pub meso2: [f32; 4],
    /// Large-scale tint of the graft toward the real Earth near its edge: x strength, y inner
    /// radius (rad; full by `graft.w`), z real-albedo LOD, w graft-albedo LOD.
    pub graft_t: [f32; 4],
    /// Round-6 marine haze over open ocean: x vertical AOD at 550 nm (0 = off, = round 5), y
    /// Angstrom exponent, z single-scattering albedo, w water-mask texel size (km).
    pub haze: [f32; 4],
    /// Round 7: x cloud-edge feather scale (km, 0 = off, = round 6), y feather mix, z
    /// clear-window edge noise (0 = round 6's circle), w marine-haze dust absorption (0 = off).
    pub cloud3: [f32; 4],
}

/// == `PPatch`: patch origin minus camera (m, rounded once from f64), w = level.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PatchGpu {
    pub origin: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PushGpu {
    p: u64,
    v: u64,
    patches: u64,
    atmo: u64,
}

/// One frame: visible patches (vertices of patch `k` at `vertices[k * VERTS..]`, stitching mask
/// per patch) and parameters.
pub struct PlanetFrame<'a> {
    pub width: u32,
    pub height: u32,
    pub params: PlanetParamsGpu,
    pub patches: &'a [PatchGpu],
    pub masks: &'a [u8],
    pub vertices: &'a [PVertex],
}

#[derive(Clone, Debug, Default)]
pub struct PlanetTimings {
    pub gpu_ms: f64,
    pub breakdown: Vec<(&'static str, f64)>,
    pub patches: u32,
    pub triangles: u64,
}

struct CubeImage {
    image: vk::Image,
    view: vk::ImageView,
    alloc: Option<Allocation>,
}

/// One cube map to upload: a PLN1 texture (layer-major) or a parsed KTX2 file (level-major,
/// raw or zstd levels decoded straight into the staging buffer).
pub enum CubeSrc<'a> {
    Pln(&'a Texture),
    Ktx2(&'a Ktx2<'a>),
}
impl CubeSrc<'_> {
    fn meta(&self) -> (TexFormat, u32, u32, usize) {
        match self {
            CubeSrc::Pln(t) => (t.format, t.size, t.mips, t.data.len()),
            CubeSrc::Ktx2(k) => (k.format, k.size, k.mips, k.gpu_bytes()),
        }
    }
}

/// Descriptor binding `i` of set 0 = view of image `image` with component swizzle `swizzle`
/// (PLN1 keeps clouds / ice / depth in one BC7 image: the masks binding swizzles g, b -> r, g).
#[derive(Clone, Copy, Debug)]
pub struct TexBinding {
    pub image: usize,
    pub swizzle: [vk::ComponentSwizzle; 4],
}
pub const IDENTITY: [vk::ComponentSwizzle; 4] = [vk::ComponentSwizzle::IDENTITY; 4];

/// Cloud-noise 3D texture size (texels per side, RGBA8; see `planet::cloudnoise`).
pub const CLOUD_NOISE_N: u32 = 128;

/// Uploaded planet textures + their descriptor set (persist across frames).
/// Bindings: 0 albedo, 1 slope, 2 night, 3 clouds (r coverage, g cloud-top height km x 20 / 255, 0 = unknown), 4 masks (r ice, g ocean depth),
/// 5 cloud noise (3D, repeat).
pub struct PlanetTextures {
    cubes: Vec<CubeImage>,
    views: Vec<vk::ImageView>,
    noise: CubeImage,
    sampler: vk::Sampler,
    noise_sampler: vk::Sampler,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    /// GPU bytes of the cube-map texture data (as stored: BCn or raw, all mips).
    pub bytes: u64,
    /// Per image: (format, face size, mips, bytes).
    pub images: Vec<(TexFormat, u32, u32, u64)>,
    /// CPU ms spent filling staging (memcpy or zstd decode), summed over images.
    pub decode_ms: f64,
    index: GpuBuffer,
    variants: [(u32, u32); 16],
    /// Bindings 6 / 7: shared-Alps graft albedo (sRGB) and aux arrays (1x1 dummies until
    /// [`AshBackend::planet_set_graft`]).
    graft: [CubeImage; 2],
    /// GPU bytes of the graft arrays (all mips).
    pub graft_bytes: u64,
}

/// One 2D-array level chain: per layer, per mip, tightly packed RGBA8.
pub type ArrayMips = Vec<Vec<Vec<u8>>>;

/// Box-filtered RGBA8 mip chain of one square layer (`srgb`: rgb averaged in linear light).
pub fn rgba8_mips(size: u32, base: &[u8], srgb: bool) -> Vec<Vec<u8>> {
    let to_lin: Vec<f32> = (0..256).map(|i| descent_lin(i as f32 / 255.0)).collect();
    let mut out = vec![base.to_vec()];
    let mut s = size as usize;
    while s > 1 {
        let prev = out.last().unwrap();
        let n = s / 2;
        let mut m = vec![0u8; n * n * 4];
        for j in 0..n {
            for i in 0..n {
                for c in 0..4 {
                    let px = |a: usize, b: usize| prev[((2 * j + b) * s + 2 * i + a) * 4 + c];
                    let v = if srgb && c < 3 {
                        let l = (to_lin[px(0, 0) as usize]
                            + to_lin[px(1, 0) as usize]
                            + to_lin[px(0, 1) as usize]
                            + to_lin[px(1, 1) as usize])
                            * 0.25;
                        (descent_srgb(l) * 255.0 + 0.5) as u8
                    } else {
                        ((px(0, 0) as u32 + px(1, 0) as u32 + px(0, 1) as u32 + px(1, 1) as u32 + 2) / 4) as u8
                    };
                    m[(j * n + i) * 4 + c] = v;
                }
            }
        }
        out.push(m);
        s = n;
    }
    out
}
fn descent_lin(c: f32) -> f32 {
    planet::descent::srgb_to_linear(c)
}
fn descent_srgb(c: f32) -> f32 {
    planet::descent::linear_to_srgb(c)
}

fn vk_format(f: TexFormat) -> vk::Format {
    match f {
        TexFormat::Bc7Srgb => vk::Format::BC7_SRGB_BLOCK,
        TexFormat::Bc7 => vk::Format::BC7_UNORM_BLOCK,
        TexFormat::Bc5 => vk::Format::BC5_UNORM_BLOCK,
        TexFormat::Bc4 => vk::Format::BC4_UNORM_BLOCK,
        TexFormat::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
        TexFormat::Rgba8 => vk::Format::R8G8B8A8_UNORM,
        TexFormat::Rg8 => vk::Format::R8G8_UNORM,
        TexFormat::R8 => vk::Format::R8_UNORM,
        TexFormat::R16 => vk::Format::R16_UNORM,
    }
}

/// IEEE half -> f32.
pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            // subnormal
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            s | (e2 << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

impl AshBackend {
    /// Upload the PLN1 pack's cube maps (BC, all mips): aux feeds both the clouds and (swizzled)
    /// the masks binding.
    pub fn planet_upload(&mut self, pack: &PlanetPack) -> Result<PlanetTextures, BackendError> {
        let mut srcs = Vec::new();
        for n in ["albedo", "slope", "night", "aux"] {
            srcs.push(CubeSrc::Pln(pack.texture(n).ok_or(BackendError::BadConfig("planet pack lacks a texture"))?));
        }
        use vk::ComponentSwizzle as C;
        let masks = TexBinding { image: 3, swizzle: [C::G, C::B, C::ZERO, C::ONE] };
        // clouds: r coverage; g (cloud-top height) = 0 = unknown -> the shader's coverage model
        let clouds = TexBinding { image: 3, swizzle: [C::R, C::ZERO, C::ZERO, C::ONE] };
        let b = |image| TexBinding { image, swizzle: IDENTITY };
        self.planet_upload_cubes(&srcs, &[b(0), b(1), b(2), clouds, masks])
    }

    /// Upload KTX2 cube maps in binding order (albedo, slope, night, clouds, masks).
    pub fn planet_upload_ktx2(&mut self, files: &[Ktx2<'_>]) -> Result<PlanetTextures, BackendError> {
        if files.len() != 5 {
            return Err(BackendError::BadConfig("planet KTX2 set needs albedo, slope, night, clouds, masks"));
        }
        let srcs: Vec<CubeSrc> = files.iter().map(CubeSrc::Ktx2).collect();
        let b: Vec<TexBinding> = (0..5).map(|image| TexBinding { image, swizzle: IDENTITY }).collect();
        self.planet_upload_cubes(&srcs, &b)
    }

    /// Create + fill one cube image per source (formats straight from the PLN1 / KTX2 header),
    /// then the descriptor set: `bindings[i]` -> binding i; the cloud noise is binding 5.
    pub fn planet_upload_cubes(
        &mut self,
        srcs: &[CubeSrc<'_>],
        bindings: &[TexBinding],
    ) -> Result<PlanetTextures, BackendError> {
        if srcs.iter().any(|s| s.meta().0.is_block()) && !self.ctx.texture_bc {
            return Err(BackendError::Unsupported("textureCompressionBC"));
        }
        if bindings.len() != 5 || bindings.iter().any(|b| b.image >= srcs.len()) {
            return Err(BackendError::BadConfig("planet texture bindings"));
        }
        let d = self.ctx.device.clone();
        let mut cubes = Vec::new();
        let mut images = Vec::new();
        let mut bytes = 0u64;
        let mut decode_ms = 0.0;
        for src in srcs {
            let (format, size, mips, len) = src.meta();
            let fmt = vk_format(format);
            let ici = vk::ImageCreateInfo::default()
                .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
                .image_type(vk::ImageType::TYPE_2D)
                .format(fmt)
                .extent(vk::Extent3D { width: size, height: size, depth: 1 })
                .mip_levels(mips)
                .array_layers(6)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mips,
                base_array_layer: 0,
                layer_count: 6,
            };
            let (image, alloc) = self.planet_image(&ici, "planet_cube")?;
            let mut staging = GpuBuffer::new(
                &mut self.ctx,
                "planet_staging",
                len as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
            )?;
            let t0 = std::time::Instant::now();
            let mut regions = Vec::new();
            let region = |off: u64, mip: u32, layer: u32| vk::BufferImageCopy {
                buffer_offset: off,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: mip,
                    base_array_layer: layer,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D {
                    width: (size >> mip).max(if format.is_block() { 4 } else { 1 }),
                    height: (size >> mip).max(if format.is_block() { 4 } else { 1 }),
                    depth: 1,
                },
            };
            let fill: Result<(), BackendError> = match src {
                CubeSrc::Pln(t) => {
                    staging.mapped_mut()[..len].copy_from_slice(&t.data);
                    let mut off = 0u64;
                    for layer in 0..6 {
                        for mip in 0..mips {
                            regions.push(region(off, mip, layer));
                            off += planet::pack::mip_bytes(format, size, mip) as u64;
                        }
                    }
                    Ok(())
                }
                CubeSrc::Ktx2(k) => {
                    // level-major as in the file: split the mapped staging into per-level slices and
                    // decode the levels in parallel (zstd straight into GPU-visible memory)
                    use rayon::prelude::*;
                    let mut slices = Vec::new();
                    let mut rest = &mut staging.mapped_mut()[..len];
                    let mut off = 0u64;
                    for m in 0..mips as usize {
                        let lb = k.levels[m].uncompressed;
                        let (a, b) = rest.split_at_mut(lb);
                        slices.push((m, a));
                        rest = b;
                        let mb = planet::pack::mip_bytes(format, size, m as u32) as u64;
                        for layer in 0..6 {
                            regions.push(region(off + layer as u64 * mb, m as u32, layer));
                        }
                        off += lb as u64;
                    }
                    slices
                        .into_par_iter()
                        .try_for_each(|(m, out)| k.level_into(m, out))
                        .map_err(|_| BackendError::BadConfig("corrupt KTX2 level"))
                }
            };
            decode_ms += t0.elapsed().as_secs_f64() * 1e3;
            let r = fill.and_then(|_| self.planet_copy(&staging, image, range, &regions));
            staging.destroy(&mut self.ctx);
            if let Err(e) = r {
                // SAFETY: the image is unused (upload failed, nothing references it).
                unsafe { d.destroy_image(image, None) };
                let _ = self.ctx.allocator.free(alloc);
                return Err(e);
            }
            bytes += len as u64;
            images.push((format, size, mips, len as u64));
            cubes.push(CubeImage { image, view: vk::ImageView::null(), alloc: Some(alloc) });
        }
        let mut views = Vec::new();
        for b in bindings {
            let (format, _, mips, _) = srcs[b.image].meta();
            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mips,
                base_array_layer: 0,
                layer_count: 6,
            };
            let comps = vk::ComponentMapping { r: b.swizzle[0], g: b.swizzle[1], b: b.swizzle[2], a: b.swizzle[3] };
            let vci = vk::ImageViewCreateInfo::default()
                .image(cubes[b.image].image)
                .view_type(vk::ImageViewType::CUBE)
                .format(vk_format(format))
                .components(comps)
                .subresource_range(range);
            // SAFETY: the image is bound; the view matches its format, layers and mips.
            views.push(unsafe { d.create_image_view(&vci, None)? });
        }
        // cloud noise: 3D RGBA8, one level (sampled at explicit LOD 0; octaves fade by footprint)
        let n = CLOUD_NOISE_N;
        let ici = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_3D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D { width: n, height: n, depth: n })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let (nimg, nalloc) = self.planet_image(&ici, "planet_cloud_noise")?;
        let data = planet::cloudnoise::generate(n as usize);
        let mut staging = GpuBuffer::new(
            &mut self.ctx,
            "planet_noise_staging",
            data.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;
        staging.mapped_mut()[..data.len()].copy_from_slice(&data);
        let nrange = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let reg = [vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: n, height: n, depth: n },
        }];
        let r = self.planet_copy(&staging, nimg, nrange, &reg);
        staging.destroy(&mut self.ctx);
        r?;
        let vci = vk::ImageViewCreateInfo::default()
            .image(nimg)
            .view_type(vk::ImageViewType::TYPE_3D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(nrange);
        // SAFETY: the noise image is bound and filled; the view matches it.
        let nview = unsafe { d.create_image_view(&vci, None)? };
        let noise = CubeImage { image: nimg, view: nview, alloc: Some(nalloc) };
        let mut sci = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(vk::LOD_CLAMP_NONE);
        if let Some(a) = self.ctx.max_anisotropy {
            sci = sci.anisotropy_enable(true).max_anisotropy(a.min(16.0));
        }
        // SAFETY: device valid; sci initialised (anisotropy only when the feature is enabled).
        let sampler = unsafe { d.create_sampler(&sci, None)? };
        let nsci = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .max_lod(0.0);
        // SAFETY: device valid; nsci initialised.
        let noise_sampler = unsafe { d.create_sampler(&nsci, None)? };
        let dummy =
            |srgb: bool| -> ArrayMips { vec![vec![if srgb { vec![0, 0, 0, 255] } else { vec![128, 128, 255, 255] }]] };
        let g0 = self.planet_array_image(vk::Format::R8G8B8A8_SRGB, 1, &dummy(true))?;
        let g1 = self.planet_array_image(vk::Format::R8G8B8A8_UNORM, 1, &dummy(false))?;
        let binds: Vec<vk::DescriptorSetLayoutBinding> = (0..8)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            })
            .collect();
        // SAFETY: device valid; create infos reference live locals.
        let (set_layout, pool, set) = unsafe {
            let sl =
                d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds), None)?;
            let ps = [vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 8 }];
            let pool =
                d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&ps), None)?;
            let sls = [sl];
            let set = d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&sls),
            )?[0];
            let mut infos: Vec<[vk::DescriptorImageInfo; 1]> = views
                .iter()
                .map(|&v| {
                    [vk::DescriptorImageInfo {
                        sampler,
                        image_view: v,
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    }]
                })
                .collect();
            infos.push([vk::DescriptorImageInfo {
                sampler: noise_sampler,
                image_view: nview,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }]);
            for g in [&g0, &g1] {
                infos.push([vk::DescriptorImageInfo {
                    sampler,
                    image_view: g.view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }]);
            }
            let writes: Vec<vk::WriteDescriptorSet> = infos
                .iter()
                .enumerate()
                .map(|(b, i)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(b as u32)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(i)
                })
                .collect();
            d.update_descriptor_sets(&writes, &[]);
            (sl, pool, set)
        };
        let (ix, variants) = planet::patch::index_variants();
        let mut index = GpuBuffer::new(
            &mut self.ctx,
            "planet_indices",
            (ix.len() * 4) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER,
            MemoryLocation::CpuToGpu,
        )?;
        index.mapped_mut()[..ix.len() * 4].copy_from_slice(bytemuck::cast_slice(&ix));
        Ok(PlanetTextures {
            cubes,
            views,
            noise,
            sampler,
            noise_sampler,
            set_layout,
            pool,
            set,
            bytes,
            images,
            decode_ms,
            index,
            variants,
            graft: [g0, g1],
            graft_bytes: 0,
        })
    }

    /// RGBA8 2D array (layers x mips as given), SHADER_READ_ONLY, with a 2D-array view.
    fn planet_array_image(
        &mut self,
        fmt: vk::Format,
        size: u32,
        layers: &ArrayMips,
    ) -> Result<CubeImage, BackendError> {
        let mips = layers[0].len() as u32;
        let nl = layers.len() as u32;
        let ici = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(fmt)
            .extent(vk::Extent3D { width: size, height: size, depth: 1 })
            .mip_levels(mips)
            .array_layers(nl)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let (image, alloc) = self.planet_image(&ici, "planet_graft")?;
        let len: usize = layers.iter().flatten().map(|m| m.len()).sum();
        let mut staging = GpuBuffer::new(
            &mut self.ctx,
            "planet_graft_staging",
            len as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;
        let mut regions = Vec::new();
        let mut off = 0usize;
        for (l, chain) in layers.iter().enumerate() {
            for (m, data) in chain.iter().enumerate() {
                staging.mapped_mut()[off..off + data.len()].copy_from_slice(data);
                let e = (size >> m).max(1);
                regions.push(vk::BufferImageCopy {
                    buffer_offset: off as u64,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: m as u32,
                        base_array_layer: l as u32,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D::default(),
                    image_extent: vk::Extent3D { width: e, height: e, depth: 1 },
                });
                off += data.len();
            }
        }
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: mips,
            base_array_layer: 0,
            layer_count: nl,
        };
        let r = self.planet_copy(&staging, image, range, &regions);
        staging.destroy(&mut self.ctx);
        if let Err(e) = r {
            // SAFETY: the image is unused (upload failed, nothing references it).
            unsafe { self.ctx.device.destroy_image(image, None) };
            let _ = self.ctx.allocator.free(alloc);
            return Err(e);
        }
        let vci = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
            .format(fmt)
            .subresource_range(range);
        // SAFETY: the image is bound and filled; the view matches its format, layers and mips.
        let view = unsafe { self.ctx.device.create_image_view(&vci, None)? };
        Ok(CubeImage { image, view, alloc: Some(alloc) })
    }

    /// Replace the shared-Alps graft arrays (bindings 6 / 7): `albedo` / `aux` = per layer
    /// (outer, inner) the RGBA8 mip chain of a `size`^2 level (see [`rgba8_mips`]).
    pub fn planet_set_graft(
        &mut self,
        t: &mut PlanetTextures,
        size: u32,
        albedo: &ArrayMips,
        aux: &ArrayMips,
    ) -> Result<(), BackendError> {
        self.planet_set_graft_fmt(t, size, (TexFormat::Rgba8Srgb, albedo), (TexFormat::Rgba8, aux))
    }

    /// [`Self::planet_set_graft`] with explicit formats (Earth round 5: BC7 sRGB albedo + BC7
    /// aux from KTX2; mip data as stored, BC blocks for the block formats).
    pub fn planet_set_graft_fmt(
        &mut self,
        t: &mut PlanetTextures,
        size: u32,
        albedo: (TexFormat, &ArrayMips),
        aux: (TexFormat, &ArrayMips),
    ) -> Result<(), BackendError> {
        if (albedo.0.is_block() || aux.0.is_block()) && !self.ctx.texture_bc {
            return Err(BackendError::Unsupported("textureCompressionBC"));
        }
        if albedo.1.first().map(|c| c.len()) != aux.1.first().map(|c| c.len()) {
            return Err(BackendError::BadConfig("graft albedo / aux mip counts differ"));
        }
        let a = self.planet_array_image(vk_format(albedo.0), size, albedo.1)?;
        let b = self.planet_array_image(vk_format(aux.0), size, aux.1)?;
        self.ctx.wait_idle();
        let d = self.ctx.device.clone();
        let infos = [
            [vk::DescriptorImageInfo {
                sampler: t.sampler,
                image_view: a.view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }],
            [vk::DescriptorImageInfo {
                sampler: t.sampler,
                image_view: b.view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }],
        ];
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(k, i)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(t.set)
                    .dst_binding(6 + k as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(i)
            })
            .collect();
        // SAFETY: GPU idle (no frame uses the set); the views are live.
        unsafe { d.update_descriptor_sets(&writes, &[]) };
        let old = std::mem::replace(&mut t.graft, [a, b]);
        for mut g in old {
            // SAFETY: GPU idle; the set no longer references the old views.
            unsafe {
                d.destroy_image_view(g.view, None);
                d.destroy_image(g.image, None);
            }
            if let Some(al) = g.alloc.take() {
                let _ = self.ctx.allocator.free(al);
            }
        }
        t.graft_bytes = (albedo.1.iter().chain(aux.1).flatten().map(|m| m.len()).sum::<usize>()) as u64;
        Ok(())
    }

    pub(crate) fn planet_image(
        &mut self,
        ici: &vk::ImageCreateInfo,
        name: &str,
    ) -> Result<(vk::Image, Allocation), BackendError> {
        let d = &self.ctx.device;
        // SAFETY: device valid; ici fully initialised; BC formats only when enabled (checked by the caller).
        let image = unsafe { d.create_image(ici, None)? };
        // SAFETY: image was just created from this device.
        let req = unsafe { d.get_image_memory_requirements(image) };
        let alloc = match self.ctx.allocator.allocate(&AllocationCreateDesc {
            name,
            requirements: req,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(a) => a,
            Err(e) => {
                // SAFETY: unbound, unused image.
                unsafe { self.ctx.device.destroy_image(image, None) };
                return Err(e.into());
            }
        };
        // SAFETY: memory satisfies `req`; image unbound.
        unsafe { self.ctx.device.bind_image_memory(image, alloc.memory(), alloc.offset())? };
        Ok((image, alloc))
    }

    /// staging -> image (all regions), leaving it SHADER_READ_ONLY.
    pub(crate) fn planet_copy(
        &mut self,
        staging: &GpuBuffer,
        image: vk::Image,
        range: vk::ImageSubresourceRange,
        regions: &[vk::BufferImageCopy],
    ) -> Result<(), BackendError> {
        let d = self.ctx.device.clone();
        let buf = staging.buffer;
        self.ctx.immediate_submit(|cmd| {
            use vk::AccessFlags2 as A;
            use vk::PipelineStageFlags2 as S;
            let barrier = |old, new, src: (S, A), dst: (S, A)| {
                let b = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(src.0)
                    .src_access_mask(src.1)
                    .dst_stage_mask(dst.0)
                    .dst_access_mask(dst.1)
                    .old_layout(old)
                    .new_layout(new)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(range)];
                // SAFETY: cmd recording; image live with all mips/layers in `range`.
                unsafe { d.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&b)) };
            };
            barrier(
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                (S::NONE, A::NONE),
                (S::COPY, A::TRANSFER_WRITE),
            );
            // SAFETY: staging holds the bytes laid out as `regions`; image in TRANSFER_DST.
            unsafe { d.cmd_copy_buffer_to_image(cmd, buf, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, regions) };
            barrier(
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                (S::COPY, A::TRANSFER_WRITE),
                (S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ),
            );
        })
    }

    pub fn planet_free(&mut self, mut t: PlanetTextures) {
        self.ctx.wait_idle();
        let d = self.ctx.device.clone();
        // SAFETY: GPU idle; nothing references these objects any more.
        unsafe {
            d.destroy_descriptor_pool(t.pool, None);
            d.destroy_descriptor_set_layout(t.set_layout, None);
            d.destroy_sampler(t.sampler, None);
            d.destroy_sampler(t.noise_sampler, None);
            for &v in &t.views {
                d.destroy_image_view(v, None);
            }
            d.destroy_image_view(t.noise.view, None);
            d.destroy_image(t.noise.image, None);
            for c in &t.cubes {
                d.destroy_image(c.image, None);
            }
        }
        if let Some(a) = t.noise.alloc.take() {
            let _ = self.ctx.allocator.free(a);
        }
        for c in &mut t.cubes {
            if let Some(a) = c.alloc.take() {
                let _ = self.ctx.allocator.free(a);
            }
        }
        for g in &mut t.graft {
            // SAFETY: GPU idle; nothing references the graft arrays any more.
            unsafe {
                d.destroy_image_view(g.view, None);
                d.destroy_image(g.image, None);
            }
            if let Some(a) = g.alloc.take() {
                let _ = self.ctx.allocator.free(a);
            }
        }
        t.index.destroy(&mut self.ctx);
    }

    /// Render one planet frame offscreen; returns linear HDR RGBA f32 (row-major) + timings.
    pub fn render_planet(
        &mut self,
        tex: &PlanetTextures,
        f: &PlanetFrame<'_>,
    ) -> Result<(Vec<[f32; 4]>, PlanetTimings), BackendError> {
        if f.patches.len() != f.masks.len() || f.vertices.len() != f.patches.len() * VERTS as usize {
            return Err(BackendError::BadConfig("planet frame: patches / masks / vertices mismatch"));
        }
        self.ctx.wait_idle();
        let d = self.ctx.device.clone();
        let sda = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER;
        let up =
            |ctx: &mut crate::context::VkContext, name: &str, b: &[u8], usage| -> Result<GpuBuffer, BackendError> {
                let mut g = GpuBuffer::new(ctx, name, b.len().max(16) as u64, usage, MemoryLocation::CpuToGpu)?;
                g.mapped_mut()[..b.len()].copy_from_slice(b);
                Ok(g)
            };
        let pb = up(&mut self.ctx, "planet_params", bytemuck::bytes_of(&f.params), sda)?;
        let vb = up(&mut self.ctx, "planet_vertices", bytemuck::cast_slice(f.vertices), sda)?;
        let patch_b = up(&mut self.ctx, "planet_patches", bytemuck::cast_slice(f.patches), sda)?;
        let mut cmds = Vec::with_capacity(f.patches.len());
        let mut tris = 0u64;
        for (k, &m) in f.masks.iter().enumerate() {
            let (first, count) = tex.variants[m as usize & 15];
            tris += count as u64 / 3;
            cmds.push(vk::DrawIndexedIndirectCommand {
                index_count: count,
                instance_count: 1,
                first_index: first,
                vertex_offset: (k as u32 * VERTS) as i32,
                first_instance: 0,
            });
        }
        // SAFETY: DrawIndexedIndirectCommand is 5 plain u32/i32 fields (repr C, no padding).
        let cmd_bytes = unsafe {
            std::slice::from_raw_parts(
                cmds.as_ptr() as *const u8,
                cmds.len() * size_of::<vk::DrawIndexedIndirectCommand>(),
            )
        };
        let ind = up(&mut self.ctx, "planet_indirect", cmd_bytes, vk::BufferUsageFlags::INDIRECT_BUFFER)?;
        let atmo = GpuBuffer::new(&mut self.ctx, "planet_luts", LUT_FLOAT4S * 16, sda, MemoryLocation::GpuOnly)?;
        let ext = vk::Extent2D { width: f.width, height: f.height };
        let mut color = GpuImage::new(
            &mut self.ctx,
            "planet_color",
            ext,
            COLOR,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
        )?;
        let mut depth = GpuImage::new(
            &mut self.ctx,
            "planet_depth",
            ext,
            DEPTH,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
        )?;
        let rb_bytes = f.width as u64 * f.height as u64 * 8;
        let rb = GpuBuffer::new(
            &mut self.ctx,
            "planet_readback",
            rb_bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
        )?;
        let range = [vk::PushConstantRange { stage_flags: STAGES, offset: 0, size: size_of::<PushGpu>() as u32 }];
        let sls = [tex.set_layout];
        // SAFETY: device valid; create info references live locals.
        let layout = unsafe {
            d.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&sls).push_constant_ranges(&range),
                None,
            )?
        };
        let surf = graphics(&d, layout, PLANET_VS, PLANET_FS, true)?;
        let sky = graphics(&d, layout, SKY_VS, SKY_FS, false)?;
        let luts: Vec<vk::Pipeline> = LUT_CS.iter().map(|s| compute(&d, layout, s)).collect::<Result<_, _>>()?;
        // SAFETY: device valid.
        let qp = unsafe {
            d.create_query_pool(
                &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(8),
                None,
            )?
        };
        let pc = PushGpu { p: pb.address, v: vb.address, patches: patch_b.address, atmo: atmo.address };
        let labels = ["atmosphere luts", "surface patches", "sky + limb", "readback copy"];
        let r = self.ctx.immediate_submit(|cmd| {
            use vk::AccessFlags2 as A;
            use vk::PipelineStageFlags2 as S;
            // SAFETY: all handles live for the submit; layouts tracked by the barriers below.
            unsafe {
                d.cmd_reset_query_pool(cmd, qp, 0, 8);
                d.cmd_write_timestamp2(cmd, S::TOP_OF_PIPE, qp, 0);
                d.cmd_push_constants(cmd, layout, STAGES, 0, bytemuck::bytes_of(&pc));
                for (k, (&pipe, &(gx, gy))) in luts.iter().zip(LUT_GROUPS.iter()).enumerate() {
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
                    d.cmd_dispatch(cmd, gx, gy, 1);
                    let dst = if k + 1 < luts.len() {
                        (S::COMPUTE_SHADER, A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE)
                    } else {
                        (S::VERTEX_SHADER | S::FRAGMENT_SHADER, A::SHADER_STORAGE_READ)
                    };
                    memory_barrier(&d, cmd, (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE), dst);
                }
                d.cmd_write_timestamp2(cmd, S::ALL_COMMANDS, qp, 1);
                image_barrier(
                    &d,
                    cmd,
                    color.image,
                    vk::ImageAspectFlags::COLOR,
                    (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                    (S::NONE, A::NONE),
                    (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE),
                );
                image_barrier(
                    &d,
                    cmd,
                    depth.image,
                    vk::ImageAspectFlags::DEPTH,
                    (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL),
                    (S::NONE, A::NONE),
                    (
                        S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                        A::DEPTH_STENCIL_ATTACHMENT_WRITE | A::DEPTH_STENCIL_ATTACHMENT_READ,
                    ),
                );
                let catt = [vk::RenderingAttachmentInfo::default()
                    .image_view(color.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } })];
                let datt = vk::RenderingAttachmentInfo::default()
                    .image_view(depth.view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
                    });
                let ri = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: ext })
                    .layer_count(1)
                    .color_attachments(&catt)
                    .depth_attachment(&datt);
                d.cmd_begin_rendering(cmd, &ri);
                d.cmd_set_viewport(
                    cmd,
                    0,
                    &[vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: ext.width as f32,
                        height: ext.height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );
                d.cmd_set_scissor(cmd, 0, &[vk::Rect2D { offset: vk::Offset2D::default(), extent: ext }]);
                d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, layout, 0, &[tex.set], &[]);
                d.cmd_push_constants(cmd, layout, STAGES, 0, bytemuck::bytes_of(&pc));
                if !cmds.is_empty() {
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, surf);
                    d.cmd_bind_index_buffer(cmd, tex.index.buffer, 0, vk::IndexType::UINT32);
                    d.cmd_draw_indexed_indirect(
                        cmd,
                        ind.buffer,
                        0,
                        cmds.len() as u32,
                        size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                    );
                }
                d.cmd_write_timestamp2(cmd, S::ALL_COMMANDS, qp, 2);
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, sky);
                d.cmd_draw(cmd, 3, 1, 0, 0);
                d.cmd_end_rendering(cmd);
                d.cmd_write_timestamp2(cmd, S::ALL_COMMANDS, qp, 3);
                image_barrier(
                    &d,
                    cmd,
                    color.image,
                    vk::ImageAspectFlags::COLOR,
                    (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
                    (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE),
                    (S::COPY, A::TRANSFER_READ),
                );
                let region = [vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D::default(),
                    image_extent: vk::Extent3D { width: f.width, height: f.height, depth: 1 },
                }];
                d.cmd_copy_image_to_buffer(cmd, color.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, rb.buffer, &region);
                memory_barrier(&d, cmd, (S::COPY, A::TRANSFER_WRITE), (S::HOST, A::HOST_READ));
                d.cmd_write_timestamp2(cmd, S::ALL_COMMANDS, qp, 4);
            }
        });
        let mut t = PlanetTimings { patches: f.patches.len() as u32, triangles: tris, ..Default::default() };
        if r.is_ok()
            && let Some(period) = self.ctx.timestamp_period_ns
        {
            let mut ts = [0u64; 5];
            // SAFETY: queries 0..5 were written by the completed submit.
            if unsafe {
                d.get_query_pool_results(qp, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
            }
            .is_ok()
            {
                let ms = |a: u64, b: u64| b.saturating_sub(a) as f64 * period as f64 * 1e-6;
                t.gpu_ms = ms(ts[0], ts[4]);
                t.breakdown = labels.iter().enumerate().map(|(k, &l)| (l, ms(ts[k], ts[k + 1]))).collect();
            }
        }
        let px: Vec<[f32; 4]> = if r.is_ok() {
            let raw: &[u16] = bytemuck::cast_slice(&rb.mapped()[..rb_bytes as usize]);
            raw.as_chunks::<4>().0.iter().map(|c| c.map(f16_to_f32)).collect()
        } else {
            Vec::new()
        };
        // SAFETY: the submit completed (immediate_submit waits); nothing references these objects.
        unsafe {
            d.destroy_query_pool(qp, None);
            d.destroy_pipeline(surf, None);
            d.destroy_pipeline(sky, None);
            for p in luts {
                d.destroy_pipeline(p, None);
            }
            d.destroy_pipeline_layout(layout, None);
        }
        for mut b in [pb, vb, patch_b, ind, atmo, rb] {
            b.destroy(&mut self.ctx);
        }
        color.destroy(&mut self.ctx);
        depth.destroy(&mut self.ctx);
        r?;
        Ok((px, t))
    }
}

fn graphics(
    d: &ash::Device,
    layout: vk::PipelineLayout,
    vs: &[u8],
    fs: &[u8],
    surface: bool,
) -> Result<vk::Pipeline, BackendError> {
    // SAFETY: device valid; create infos reference live locals; modules destroyed after creation.
    unsafe {
        let vsm = crate::pass::shader_module(d, vs)?;
        let fsm = crate::pass::shader_module(d, fs)?;
        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vsm).name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fsm)
                .name(c"main"),
        ];
        let vi = vk::PipelineVertexInputStateCreateInfo::default();
        let ia = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let vp = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        // skirts are single-sided walls: no culling (far-side patches are horizon-culled on the CPU)
        let rs = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let ms = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let ds = if surface {
            vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::GREATER)
        } else {
            vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(false)
                .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL)
        };
        let blend = [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA)];
        let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
        let dyns = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dy = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyns);
        let cf = [COLOR];
        let mut rendering =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&cf).depth_attachment_format(DEPTH);
        let gci = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vi)
            .input_assembly_state(&ia)
            .viewport_state(&vp)
            .rasterization_state(&rs)
            .multisample_state(&ms)
            .depth_stencil_state(&ds)
            .color_blend_state(&cb)
            .dynamic_state(&dy)
            .layout(layout)
            .push_next(&mut rendering);
        let r = d.create_graphics_pipelines(vk::PipelineCache::null(), &[gci], None).map_err(|(_, e)| e);
        d.destroy_shader_module(vsm, None);
        d.destroy_shader_module(fsm, None);
        Ok(r?[0])
    }
}

fn compute(d: &ash::Device, layout: vk::PipelineLayout, spv: &[u8]) -> Result<vk::Pipeline, BackendError> {
    // SAFETY: device valid; the module is destroyed after pipeline creation.
    unsafe {
        let m = crate::pass::shader_module(d, spv)?;
        let ci = vk::ComputePipelineCreateInfo::default()
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(m)
                    .name(c"main"),
            )
            .layout(layout);
        let r = d.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None).map_err(|(_, e)| e);
        d.destroy_shader_module(m, None);
        Ok(r?[0])
    }
}

#[cfg(test)]
mod layout_tests {
    use super::PlanetParamsGpu;
    use std::mem::{offset_of, size_of};

    /// `PlanetParamsGpu` is read through a buffer-device-address pointer as Slang's `PParams`:
    /// the field order and float4 packing must agree, or every later field silently shifts.
    #[test]
    fn params_layout_matches_slang() {
        assert_eq!(offset_of!(PlanetParamsGpu, cloud2), 64 + 11 * 16);
        assert_eq!(offset_of!(PlanetParamsGpu, aero), 64 + 12 * 16);
        assert_eq!(offset_of!(PlanetParamsGpu, hole), 64 + 13 * 16);
        assert_eq!(offset_of!(PlanetParamsGpu, meso), 64 + 14 * 16);
        assert_eq!(offset_of!(PlanetParamsGpu, thaw), 64 + 15 * 16);
        assert_eq!(offset_of!(PlanetParamsGpu, graft), 64 + 16 * 16);
        assert_eq!(offset_of!(PlanetParamsGpu, graft_l), 64 + 19 * 16);
        assert_eq!(offset_of!(PlanetParamsGpu, haze), 64 + 22 * 16);
        assert_eq!(offset_of!(PlanetParamsGpu, cloud3), 64 + 23 * 16);
        assert_eq!(size_of::<PlanetParamsGpu>(), 64 + 24 * 16);
        let src = include_str!("../shaders/planet.slang");
        let body = &src[src.find("struct PParams {").unwrap()..];
        let body = &body[..body.find("};").unwrap()];
        let names: Vec<&str> = body
            .lines()
            .filter_map(|l| l.trim().strip_prefix("float4 "))
            .map(|l| l.split([';', '[']).next().unwrap().trim())
            .collect();
        assert_eq!(
            names,
            [
                "view_proj",
                "cam_km",
                "sun",
                "right",
                "up",
                "fwd",
                "planet",
                "ocean",
                "light",
                "atmo",
                "view",
                "cloud",
                "cloud2",
                "aero",
                "hole",
                "meso",
                "thaw",
                "graft",
                "graft_e",
                "graft_n",
                "graft_l",
                "meso2",
                "graft_t",
                "haze",
                "cloud3"
            ]
        );
    }
}
