//! Solar-system renderer (round 1): one full-screen fragment pass over every body of the system
//! frame (see `shaders/system.slang`). Offscreen, returns linear HDR RGBA f32 (rgb radiance in
//! solar illuminance at 1 AU per sr x exposure; alpha = transmittance toward space, 0 on a body,
//! -1 where a body hides the planet renderer's occluder).
//!
//! Textures: one 2D array of equirectangular sRGB layers (all the same size, full mip chains) and a
//! 1-row radial ring profile array (rgb colour + opacity).
use crate::context::{GpuBuffer, GpuImage, image_barrier, memory_barrier};
use crate::{AshBackend, BackendError};
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocation;

const SYSTEM_VS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/system_vs.spv"));
const SYSTEM_FS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/system_fs.spv"));
const COLOR: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
/// Bodies per frame (== `bodies[16]` in system.slang).
pub const MAX_BODIES: usize = 16;

/// == `SBody` in system.slang.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SystemBodyGpu {
    pub pos: [f32; 4],
    pub r0: [f32; 4],
    pub r1: [f32; 4],
    pub r2: [f32; 4],
    pub shade: [f32; 4],
    pub tint: [f32; 4],
    pub ring: [f32; 4],
    pub extra: [f32; 4],
}

/// == `SParams` in system.slang.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SystemParamsGpu {
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub fwd: [f32; 4],
    pub sun: [f32; 4],
    pub view: [f32; 4],
    pub corona: [f32; 4],
    pub sunpole: [f32; 4],
    pub light: [f32; 4],
    pub limb: [f32; 4],
    pub bodies: [SystemBodyGpu; MAX_BODIES],
}

struct ArrayImage {
    image: vk::Image,
    view: vk::ImageView,
    alloc: Option<Allocation>,
}

/// Uploaded body / ring textures and their descriptor set.
pub struct SystemTextures {
    images: [ArrayImage; 2],
    sampler: vk::Sampler,
    ring_sampler: vk::Sampler,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    /// Bytes uploaded (all mips).
    pub bytes: u64,
}

/// One frame's timings (lavapipe: CPU Vulkan).
#[derive(Clone, Debug, Default)]
pub struct SystemTimings {
    pub gpu_ms: f64,
}

/// sRGB8 -> linear and back (for the mip box filter).
fn s2l(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}
fn l2s(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

/// Full RGBA8 mip chain of a `w` x `h` image (2x2 box, rgb in linear light when `srgb`, alpha
/// linear; odd sizes clamp).
pub fn mip_chain(w: u32, h: u32, base: &[u8], srgb: bool) -> Vec<(u32, u32, Vec<u8>)> {
    let lut: Vec<f32> = (0..=255u8).map(|c| if srgb { s2l(c) } else { c as f32 / 255.0 }).collect();
    let mut out = vec![(w, h, base.to_vec())];
    while out.last().map(|m| m.0 > 1 || m.1 > 1).unwrap_or(false) {
        let (pw, ph, prev) = out.last().unwrap();
        let (pw, ph) = (*pw, *ph);
        let (nw, nh) = ((pw / 2).max(1), (ph / 2).max(1));
        let mut next = vec![0u8; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let mut acc = [0.0f32; 4];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (sx, sy) = ((2 * x + dx).min(pw - 1), (2 * y + dy).min(ph - 1));
                    let i = ((sy * pw + sx) * 4) as usize;
                    for c in 0..3 {
                        acc[c] += lut[prev[i + c] as usize];
                    }
                    acc[3] += prev[i + 3] as f32 / 255.0;
                }
                let o = ((y * nw + x) * 4) as usize;
                for c in 0..3 {
                    next[o + c] = if srgb { l2s(acc[c] / 4.0) } else { (acc[c] / 4.0 * 255.0 + 0.5) as u8 };
                }
                next[o + 3] = (acc[3] / 4.0 * 255.0 + 0.5) as u8;
            }
        }
        out.push((nw, nh, next));
    }
    out
}

impl AshBackend {
    /// RGBA8 2D array: `layers` (w x h RGBA8 each), full mip chains.
    fn system_array(
        &mut self,
        fmt: vk::Format,
        w: u32,
        h: u32,
        layers: &[Vec<u8>],
        srgb: bool,
        name: &str,
    ) -> Result<(ArrayImage, u64), BackendError> {
        let chains: Vec<Vec<(u32, u32, Vec<u8>)>> = {
            use rayon::prelude::*;
            layers.par_iter().map(|l| mip_chain(w, h, l, srgb)).collect()
        };
        let mips = chains[0].len() as u32;
        let nl = layers.len() as u32;
        let ici = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(fmt)
            .extent(vk::Extent3D { width: w, height: h, depth: 1 })
            .mip_levels(mips)
            .array_layers(nl)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let (image, alloc) = self.planet_image(&ici, name)?;
        let len: usize = chains.iter().flatten().map(|m| m.2.len()).sum();
        let mut staging = GpuBuffer::new(
            &mut self.ctx,
            "system_staging",
            len as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;
        let mut regions = Vec::new();
        let mut off = 0usize;
        for (l, chain) in chains.iter().enumerate() {
            for (m, (mw, mh, data)) in chain.iter().enumerate() {
                staging.mapped_mut()[off..off + data.len()].copy_from_slice(data);
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
                    image_extent: vk::Extent3D { width: *mw, height: *mh, depth: 1 },
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
        Ok((ArrayImage { image, view, alloc: Some(alloc) }, len as u64))
    }

    /// Upload the body layers (`w` x `h` sRGB RGBA8 each) and the ring profiles (`rw` x `rh`
    /// RGBA8: sRGB colour, linear opacity) and build the descriptor set.
    pub fn system_upload(
        &mut self,
        w: u32,
        h: u32,
        layers: &[Vec<u8>],
        rw: u32,
        rh: u32,
        rings: &[Vec<u8>],
    ) -> Result<SystemTextures, BackendError> {
        if layers.is_empty()
            || rings.is_empty()
            || layers.iter().any(|l| l.len() != (w * h * 4) as usize)
            || rings.iter().any(|l| l.len() != (rw * rh * 4) as usize)
        {
            return Err(BackendError::BadConfig("system textures: layer sizes"));
        }
        let (body, b0) = self.system_array(vk::Format::R8G8B8A8_SRGB, w, h, layers, true, "system_bodies")?;
        // ring: colour is sRGB-encoded but the shader decodes it itself (alpha must stay linear)
        let (ring, b1) = self.system_array(vk::Format::R8G8B8A8_UNORM, rw, rh, rings, false, "system_rings")?;
        let d = self.ctx.device.clone();
        let mut sci = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(vk::LOD_CLAMP_NONE);
        if let Some(a) = self.ctx.max_anisotropy {
            sci = sci.anisotropy_enable(true).max_anisotropy(a.min(16.0));
        }
        let rsci = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(0.0);
        let binds: Vec<vk::DescriptorSetLayoutBinding> = (0..2)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            })
            .collect();
        // SAFETY: device valid; create infos reference live locals (anisotropy only when enabled).
        let (sampler, ring_sampler, set_layout, pool, set) = unsafe {
            let sampler = d.create_sampler(&sci, None)?;
            let ring_sampler = d.create_sampler(&rsci, None)?;
            let sl =
                d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds), None)?;
            let ps = [vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 2 }];
            let pool =
                d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&ps), None)?;
            let sls = [sl];
            let set = d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&sls),
            )?[0];
            let infos = [
                [vk::DescriptorImageInfo {
                    sampler,
                    image_view: body.view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }],
                [vk::DescriptorImageInfo {
                    sampler: ring_sampler,
                    image_view: ring.view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }],
            ];
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
            (sampler, ring_sampler, sl, pool, set)
        };
        Ok(SystemTextures { images: [body, ring], sampler, ring_sampler, set_layout, pool, set, bytes: b0 + b1 })
    }

    pub fn system_free(&mut self, mut t: SystemTextures) {
        self.ctx.wait_idle();
        let d = self.ctx.device.clone();
        // SAFETY: GPU idle; nothing references these objects any more.
        unsafe {
            d.destroy_descriptor_pool(t.pool, None);
            d.destroy_descriptor_set_layout(t.set_layout, None);
            d.destroy_sampler(t.sampler, None);
            d.destroy_sampler(t.ring_sampler, None);
            for i in &t.images {
                d.destroy_image_view(i.view, None);
                d.destroy_image(i.image, None);
            }
        }
        for i in &mut t.images {
            if let Some(a) = i.alloc.take() {
                let _ = self.ctx.allocator.free(a);
            }
        }
    }

    /// Render one frame offscreen at `width` x `height`; returns linear HDR RGBA f32 (row-major).
    pub fn render_system(
        &mut self,
        tex: &SystemTextures,
        params: &SystemParamsGpu,
        width: u32,
        height: u32,
    ) -> Result<(Vec<[f32; 4]>, SystemTimings), BackendError> {
        self.ctx.wait_idle();
        let d = self.ctx.device.clone();
        let sda = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER;
        let bytes = bytemuck::bytes_of(params);
        let mut pb = GpuBuffer::new(&mut self.ctx, "system_params", bytes.len() as u64, sda, MemoryLocation::CpuToGpu)?;
        pb.mapped_mut()[..bytes.len()].copy_from_slice(bytes);
        let ext = vk::Extent2D { width, height };
        let mut color = GpuImage::new(
            &mut self.ctx,
            "system_color",
            ext,
            COLOR,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
        )?;
        let rb_bytes = width as u64 * height as u64 * 16;
        let rb = GpuBuffer::new(
            &mut self.ctx,
            "system_readback",
            rb_bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
        )?;
        let range = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::FRAGMENT, offset: 0, size: 8 }];
        let sls = [tex.set_layout];
        // SAFETY: device valid; create info references live locals.
        let layout = unsafe {
            d.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&sls).push_constant_ranges(&range),
                None,
            )?
        };
        let pipe = pipeline(&d, layout)?;
        // SAFETY: device valid.
        let qp = unsafe {
            d.create_query_pool(
                &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2),
                None,
            )?
        };
        let addr = pb.address;
        let r = self.ctx.immediate_submit(|cmd| {
            use vk::AccessFlags2 as A;
            use vk::PipelineStageFlags2 as S;
            // SAFETY: all handles live for the submit; layouts tracked by the barriers below.
            unsafe {
                d.cmd_reset_query_pool(cmd, qp, 0, 2);
                d.cmd_write_timestamp2(cmd, S::TOP_OF_PIPE, qp, 0);
                image_barrier(
                    &d,
                    cmd,
                    color.image,
                    vk::ImageAspectFlags::COLOR,
                    (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                    (S::NONE, A::NONE),
                    (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE),
                );
                let catt = [vk::RenderingAttachmentInfo::default()
                    .image_view(color.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .store_op(vk::AttachmentStoreOp::STORE)];
                let ri = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: ext })
                    .layer_count(1)
                    .color_attachments(&catt);
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
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipe);
                d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, layout, 0, &[tex.set], &[]);
                d.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::FRAGMENT, 0, &addr.to_le_bytes());
                d.cmd_draw(cmd, 3, 1, 0, 0);
                d.cmd_end_rendering(cmd);
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
                    image_extent: vk::Extent3D { width, height, depth: 1 },
                }];
                d.cmd_copy_image_to_buffer(cmd, color.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, rb.buffer, &region);
                memory_barrier(&d, cmd, (S::COPY, A::TRANSFER_WRITE), (S::HOST, A::HOST_READ));
                d.cmd_write_timestamp2(cmd, S::ALL_COMMANDS, qp, 1);
            }
        });
        let mut t = SystemTimings::default();
        if r.is_ok()
            && let Some(period) = self.ctx.timestamp_period_ns
        {
            let mut ts = [0u64; 2];
            // SAFETY: queries 0..2 were written by the completed submit.
            if unsafe {
                d.get_query_pool_results(qp, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
            }
            .is_ok()
            {
                t.gpu_ms = ts[1].saturating_sub(ts[0]) as f64 * period as f64 * 1e-6;
            }
        }
        let px: Vec<[f32; 4]> = if r.is_ok() {
            bytemuck::cast_slice::<u8, [f32; 4]>(&rb.mapped()[..rb_bytes as usize]).to_vec()
        } else {
            Vec::new()
        };
        // SAFETY: the submit completed (immediate_submit waits); nothing references these objects.
        unsafe {
            d.destroy_query_pool(qp, None);
            d.destroy_pipeline(pipe, None);
            d.destroy_pipeline_layout(layout, None);
        }
        for mut b in [pb, rb] {
            b.destroy(&mut self.ctx);
        }
        color.destroy(&mut self.ctx);
        r?;
        Ok((px, t))
    }
}

fn pipeline(d: &ash::Device, layout: vk::PipelineLayout) -> Result<vk::Pipeline, BackendError> {
    // SAFETY: device valid; create infos reference live locals; modules destroyed after creation.
    unsafe {
        let vsm = crate::pass::shader_module(d, SYSTEM_VS)?;
        let fsm = crate::pass::shader_module(d, SYSTEM_FS)?;
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
        let rs = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let ms = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let ds = vk::PipelineDepthStencilStateCreateInfo::default();
        let blend = [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA)];
        let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
        let dyns = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dy = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyns);
        let cf = [COLOR];
        let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&cf);
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

#[cfg(test)]
mod layout_tests {
    use super::{MAX_BODIES, SystemBodyGpu, SystemParamsGpu};
    use std::mem::size_of;

    /// The Rust structs are read through a BDA pointer as Slang's `SParams` / `SBody`: same float4
    /// field order and count, or every later field silently shifts.
    #[test]
    fn params_layout_matches_slang() {
        assert_eq!(size_of::<SystemBodyGpu>(), 8 * 16);
        assert_eq!(size_of::<SystemParamsGpu>(), 9 * 16 + MAX_BODIES * 8 * 16);
        let src = include_str!("../shaders/system.slang");
        let fields = |name: &str| -> Vec<String> {
            let body = &src[src.find(&format!("struct {name} {{")).unwrap()..];
            let body = &body[..body.find("};").unwrap()];
            body.lines()
                .filter_map(|l| l.trim().strip_prefix("float4 "))
                .map(|l| l.split([';', '[']).next().unwrap().trim().to_string())
                .collect()
        };
        assert_eq!(fields("SBody"), ["pos", "r0", "r1", "r2", "shade", "tint", "ring", "extra"]);
        assert_eq!(fields("SParams"), ["right", "up", "fwd", "sun", "view", "corona", "sunpole", "light", "limb"]);
        assert!(src.contains(&format!("SBody bodies[{MAX_BODIES}];")));
        // and the mip chain helper keeps rgb in linear light
        let m = super::mip_chain(2, 1, &[255, 255, 255, 255, 0, 0, 0, 255], true);
        assert_eq!(m.len(), 2);
        assert_eq!(m[1].2[0], super::l2s(0.5));
    }
}
