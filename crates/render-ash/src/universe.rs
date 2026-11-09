//! Persistent bounded additive HDR splats. Explicit `free` before the backend is dropped.
//! Live present path: HDR → fixed-exposure tonemap → R8G8B8A8 SDR (no per-frame HDR host readback).
use crate::context::{GpuBuffer, GpuImage, image_barrier, memory_barrier};
use crate::{AshBackend, BackendError};
use ash::vk;
use gpu_allocator::MemoryLocation;
use planet::universe::{MAX_STARS, SIGMA, Source};
const UNIVERSE_VS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/universe_vs.spv"));
const UNIVERSE_FS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/universe_fs.spv"));
const TONEMAP_VS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/universe_tonemap_vs.spv"));
const TONEMAP_FS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/universe_tonemap_fs.spv"));
const COLOR: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
const SDR: vk::Format = vk::Format::R8G8B8A8_UNORM;
/// Fixed deterministic present exposure (matches GCNS stills scale). Not a dynamic grade.
pub const PRESENT_EXPOSURE: f32 = 300.0;
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Splat {
    /// xy centre in pixels, z instrument sigma pixels, w reciprocal full discrete PSF sum.
    pub position: [f32; 4],
    /// Integrated display-channel flux in caller-declared units; w unused.
    /// GCNS uses achromatic relative G-band flux, not bolometric solar irradiance.
    pub flux: [f32; 4],
}
impl Splat {
    pub fn from_source(s: &Source) -> Self {
        Self {
            position: [s.center[0] as f32, s.center[1] as f32, SIGMA as f32, (1.0 / s.norm()) as f32],
            flux: [(s.flux * s.rgb[0]) as f32, (s.flux * s.rgb[1]) as f32, (s.flux * s.rgb[2]) as f32, 0.0],
        }
    }
}
#[derive(Default, Debug)]
pub struct UniverseTiming {
    pub gpu_ms: f64,
    pub submit_ms: f64,
}
pub struct UniversePass {
    input: GpuBuffer,
    color: GpuImage,
    sdr: GpuImage,
    readback: GpuBuffer,
    sdr_readback: Option<GpuBuffer>,
    layout: vk::PipelineLayout,
    pipe: vk::Pipeline,
    tonemap_layout: vk::PipelineLayout,
    tonemap_pipe: vk::Pipeline,
    tonemap_dsl: vk::DescriptorSetLayout,
    tonemap_pool: vk::DescriptorPool,
    tonemap_set: vk::DescriptorSet,
    sampler: vk::Sampler,
    query: vk::QueryPool,
    width: u32,
    height: u32,
    initialized: bool,
    sdr_initialized: bool,
}
fn extent_ok(w: u32, h: u32) -> Result<(), BackendError> {
    if w == 0 || h == 0 || w > 4096 || h > 4096 {
        Err(BackendError::BadConfig("universe extent must be 1..4096"))
    } else {
        Ok(())
    }
}
fn hdr_target(be: &mut AshBackend, w: u32, h: u32) -> Result<(GpuImage, GpuBuffer), BackendError> {
    let mut color = GpuImage::new(
        &mut be.ctx,
        "universe_hdr",
        vk::Extent2D { width: w, height: h },
        COLOR,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED,
        vk::ImageAspectFlags::COLOR,
    )?;
    match GpuBuffer::new(
        &mut be.ctx,
        "universe_readback",
        w as u64 * h as u64 * 16,
        vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuToCpu,
    ) {
        Ok(rb) => Ok((color, rb)),
        Err(e) => {
            color.destroy(&mut be.ctx);
            Err(e)
        }
    }
}
fn sdr_target(be: &mut AshBackend, w: u32, h: u32) -> Result<GpuImage, BackendError> {
    GpuImage::new(
        &mut be.ctx,
        "universe_sdr",
        vk::Extent2D { width: w, height: h },
        SDR,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::ImageAspectFlags::COLOR,
    )
}
fn destroy_tonemap(d: &ash::Device, p: &UniversePass) {
    // SAFETY: caller waited idle; objects unused.
    unsafe {
        d.destroy_pipeline(p.tonemap_pipe, None);
        d.destroy_pipeline_layout(p.tonemap_layout, None);
        d.destroy_descriptor_pool(p.tonemap_pool, None);
        d.destroy_descriptor_set_layout(p.tonemap_dsl, None);
        d.destroy_sampler(p.sampler, None);
    }
}
fn create_tonemap(
    d: &ash::Device,
    hdr_view: vk::ImageView,
) -> Result<
    (vk::DescriptorSetLayout, vk::DescriptorPool, vk::DescriptorSet, vk::Sampler, vk::PipelineLayout, vk::Pipeline),
    BackendError,
> {
    // SAFETY: device valid; all creates paired with destroy on error paths below.
    unsafe {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let dsl =
            d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?;
        let pool_sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLED_IMAGE, descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLER, descriptor_count: 1 },
        ];
        let pool = match d
            .create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes), None)
        {
            Ok(p) => p,
            Err(e) => {
                d.destroy_descriptor_set_layout(dsl, None);
                return Err(e.into());
            }
        };
        let set = match d.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(std::slice::from_ref(&dsl)),
        ) {
            Ok(s) => s[0],
            Err(e) => {
                d.destroy_descriptor_pool(pool, None);
                d.destroy_descriptor_set_layout(dsl, None);
                return Err(e.into());
            }
        };
        let sampler = match d.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
            None,
        ) {
            Ok(s) => s,
            Err(e) => {
                d.destroy_descriptor_pool(pool, None);
                d.destroy_descriptor_set_layout(dsl, None);
                return Err(e.into());
            }
        };
        let img_info = [vk::DescriptorImageInfo::default()
            .image_view(hdr_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let samp_info = [vk::DescriptorImageInfo::default().sampler(sampler)];
        d.update_descriptor_sets(
            &[
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&img_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&samp_info),
            ],
            &[],
        );
        let range = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::FRAGMENT, offset: 0, size: 16 }];
        let layouts = [dsl];
        let layout = match d.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&range),
            None,
        ) {
            Ok(l) => l,
            Err(e) => {
                d.destroy_sampler(sampler, None);
                d.destroy_descriptor_pool(pool, None);
                d.destroy_descriptor_set_layout(dsl, None);
                return Err(e.into());
            }
        };
        let pipe = match tonemap_pipeline(d, layout) {
            Ok(p) => p,
            Err(e) => {
                d.destroy_pipeline_layout(layout, None);
                d.destroy_sampler(sampler, None);
                d.destroy_descriptor_pool(pool, None);
                d.destroy_descriptor_set_layout(dsl, None);
                return Err(e);
            }
        };
        Ok((dsl, pool, set, sampler, layout, pipe))
    }
}
fn rebind_hdr(d: &ash::Device, set: vk::DescriptorSet, hdr_view: vk::ImageView) {
    let img_info = [vk::DescriptorImageInfo::default()
        .image_view(hdr_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
    // SAFETY: set live; sampled-image binding 0.
    unsafe {
        d.update_descriptor_sets(
            &[vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&img_info)],
            &[],
        );
    }
}
impl UniversePass {
    pub fn new(be: &mut AshBackend, width: u32, height: u32) -> Result<Self, BackendError> {
        extent_ok(width, height)?;
        let mut input = GpuBuffer::new(
            &mut be.ctx,
            "universe_stars",
            (MAX_STARS * 32) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
        )?;
        let (mut color, mut readback) = match hdr_target(be, width, height) {
            Ok(t) => t,
            Err(e) => {
                input.destroy(&mut be.ctx);
                return Err(e);
            }
        };
        let mut sdr = match sdr_target(be, width, height) {
            Ok(t) => t,
            Err(e) => {
                input.destroy(&mut be.ctx);
                color.destroy(&mut be.ctx);
                readback.destroy(&mut be.ctx);
                return Err(e);
            }
        };
        let d = be.ctx.device.clone();
        let range = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::VERTEX, offset: 0, size: 16 }];
        // SAFETY: device valid; range contains one valid 16-byte vertex push block.
        let layout = match unsafe {
            d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&range), None)
        } {
            Ok(l) => l,
            Err(e) => {
                input.destroy(&mut be.ctx);
                color.destroy(&mut be.ctx);
                sdr.destroy(&mut be.ctx);
                readback.destroy(&mut be.ctx);
                return Err(e.into());
            }
        };
        let pipe = match pipeline(&d, layout) {
            Ok(p) => p,
            Err(e) => {
                // SAFETY: no pipeline references this unused layout.
                unsafe { d.destroy_pipeline_layout(layout, None) };
                input.destroy(&mut be.ctx);
                color.destroy(&mut be.ctx);
                sdr.destroy(&mut be.ctx);
                readback.destroy(&mut be.ctx);
                return Err(e);
            }
        };
        let (tonemap_dsl, tonemap_pool, tonemap_set, sampler, tonemap_layout, tonemap_pipe) =
            match create_tonemap(&d, color.view) {
                Ok(t) => t,
                Err(e) => {
                    // SAFETY: unused.
                    unsafe {
                        d.destroy_pipeline(pipe, None);
                        d.destroy_pipeline_layout(layout, None);
                    };
                    input.destroy(&mut be.ctx);
                    color.destroy(&mut be.ctx);
                    sdr.destroy(&mut be.ctx);
                    readback.destroy(&mut be.ctx);
                    return Err(e);
                }
            };
        // SAFETY: device valid; a two-timestamp query pool needs no additional features.
        let query = match unsafe {
            d.create_query_pool(
                &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2),
                None,
            )
        } {
            Ok(q) => q,
            Err(e) => {
                // SAFETY: all objects unused by GPU.
                unsafe {
                    d.destroy_pipeline(tonemap_pipe, None);
                    d.destroy_pipeline_layout(tonemap_layout, None);
                    d.destroy_descriptor_pool(tonemap_pool, None);
                    d.destroy_descriptor_set_layout(tonemap_dsl, None);
                    d.destroy_sampler(sampler, None);
                    d.destroy_pipeline(pipe, None);
                    d.destroy_pipeline_layout(layout, None);
                };
                input.destroy(&mut be.ctx);
                color.destroy(&mut be.ctx);
                sdr.destroy(&mut be.ctx);
                readback.destroy(&mut be.ctx);
                return Err(e.into());
            }
        };
        Ok(Self {
            input,
            color,
            sdr,
            readback,
            sdr_readback: None,
            layout,
            pipe,
            tonemap_layout,
            tonemap_pipe,
            tonemap_dsl,
            tonemap_pool,
            tonemap_set,
            sampler,
            query,
            width,
            height,
            initialized: false,
            sdr_initialized: false,
        })
    }
    pub fn resize(&mut self, be: &mut AshBackend, w: u32, h: u32) -> Result<(), BackendError> {
        extent_ok(w, h)?;
        if (w, h) == (self.width, self.height) {
            return Ok(());
        }
        let (color, rb) = hdr_target(be, w, h)?;
        let sdr = sdr_target(be, w, h)?;
        be.ctx.wait_idle();
        self.color.destroy(&mut be.ctx);
        self.sdr.destroy(&mut be.ctx);
        self.readback.destroy(&mut be.ctx);
        if let Some(mut b) = self.sdr_readback.take() {
            b.destroy(&mut be.ctx);
        }
        self.color = color;
        self.sdr = sdr;
        self.readback = rb;
        rebind_hdr(&be.ctx.device, self.tonemap_set, self.color.view);
        self.width = w;
        self.height = h;
        self.initialized = false;
        self.sdr_initialized = false;
        Ok(())
    }
    pub fn extent(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    pub fn sdr_image(&self) -> vk::Image {
        self.sdr.image
    }
    fn validate_splats(stars: &[Splat]) -> Result<(), BackendError> {
        if stars.len() > MAX_STARS {
            return Err(BackendError::BadConfig("universe splat capacity exceeded"));
        }
        if stars.iter().any(|s| {
            s.position.iter().chain(s.flux.iter()).any(|x| !x.is_finite())
                || s.position[2] < 0.25
                || s.position[2] > 8.0
                || s.position[3] <= 0.0
                || s.position[3] > 16.0
                || s.position[..2].iter().any(|x| x.abs() > 8192.0)
                || s.flux.iter().any(|x| *x < 0.0 || *x > 1e30)
        }) {
            return Err(BackendError::BadConfig("invalid universe splat"));
        }
        Ok(())
    }
    /// Offscreen HDR path with host readback (existing stills / tests). Calls device wait_idle.
    pub fn render(
        &mut self,
        be: &mut AshBackend,
        stars: &[Splat],
    ) -> Result<(&[[f32; 4]], UniverseTiming), BackendError> {
        Self::validate_splats(stars)?;
        be.ctx.wait_idle();
        let bytes = bytemuck::cast_slice(stars);
        self.input.mapped_mut()[..bytes.len()].copy_from_slice(bytes);
        let d = be.ctx.device.clone();
        let ext = vk::Extent2D { width: self.width, height: self.height };
        let mut push = [0u8; 16];
        push[..8].copy_from_slice(&self.input.address.to_le_bytes());
        push[8..12].copy_from_slice(&(self.width as f32).to_le_bytes());
        push[12..].copy_from_slice(&(self.height as f32).to_le_bytes());
        let initialized = self.initialized;
        be.ctx.immediate_submit(|cmd| {
            use vk::AccessFlags2 as A;
            use vk::PipelineStageFlags2 as S;
            // SAFETY: all handles remain live through the synchronous submit; image layouts and host
            // writes are synchronized explicitly, target is cleared and bounded instances read valid input.
            unsafe {
                d.cmd_reset_query_pool(cmd, self.query, 0, 2);
                d.cmd_write_timestamp2(cmd, S::TOP_OF_PIPE, self.query, 0);
                memory_barrier(&d, cmd, (S::HOST, A::HOST_WRITE), (S::VERTEX_SHADER, A::SHADER_READ));
                let hdr_old = if !initialized {
                    vk::ImageLayout::UNDEFINED
                } else if self.sdr_initialized {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                } else {
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                };
                image_barrier(
                    &d,
                    cmd,
                    self.color.image,
                    vk::ImageAspectFlags::COLOR,
                    (hdr_old, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                    (
                        if initialized { S::ALL_COMMANDS } else { S::NONE },
                        if initialized { A::MEMORY_READ | A::MEMORY_WRITE } else { A::NONE },
                    ),
                    (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE),
                );
                let att = [vk::RenderingAttachmentInfo::default()
                    .image_view(self.color.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } })];
                d.cmd_begin_rendering(
                    cmd,
                    &vk::RenderingInfo::default()
                        .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: ext })
                        .layer_count(1)
                        .color_attachments(&att),
                );
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
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipe);
                d.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::VERTEX, 0, &push);
                d.cmd_draw(cmd, 6, stars.len() as u32, 0, 0);
                d.cmd_end_rendering(cmd);
                image_barrier(
                    &d,
                    cmd,
                    self.color.image,
                    vk::ImageAspectFlags::COLOR,
                    (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
                    (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE),
                    (S::COPY, A::TRANSFER_READ),
                );
                let region = [vk::BufferImageCopy {
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_extent: vk::Extent3D { width: ext.width, height: ext.height, depth: 1 },
                    ..Default::default()
                }];
                d.cmd_copy_image_to_buffer(
                    cmd,
                    self.color.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.readback.buffer,
                    &region,
                );
                memory_barrier(&d, cmd, (S::COPY, A::TRANSFER_WRITE), (S::HOST, A::HOST_READ));
                d.cmd_write_timestamp2(cmd, S::ALL_COMMANDS, self.query, 1);
            }
        })?;
        self.initialized = true;
        self.sdr_initialized = false;
        let mut timing = UniverseTiming::default();
        if let Some(period) = be.ctx.timestamp_period_ns {
            let mut ts = [0u64; 2];
            // SAFETY: both queries written in the completed submit; destination has two u64 slots.
            unsafe {
                d.get_query_pool_results(
                    self.query,
                    0,
                    &mut ts,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )?;
            }
            timing.gpu_ms = ts[1].saturating_sub(ts[0]) as f64 * period as f64 * 1e-6;
        }
        let px = bytemuck::cast_slice(&self.readback.mapped()[..self.width as usize * self.height as usize * 16]);
        Ok((px, timing))
    }
    /// Live path: HDR splat + fixed-exposure tonemap to SDR. No HDR host copy. Fence via immediate_submit
    /// (single-queue; not the mesh backend's multi-FIF timeline). No device_wait_idle per frame.
    pub fn render_live(
        &mut self,
        be: &mut AshBackend,
        stars: &[Splat],
        exposure: f32,
    ) -> Result<UniverseTiming, BackendError> {
        Self::validate_splats(stars)?;
        if !exposure.is_finite() || exposure <= 0.0 || exposure > 1e12 {
            return Err(BackendError::BadConfig("invalid present exposure"));
        }
        // SDR may still be TRANSFER_READ by an in-flight present/test blit. Wait that timeline
        // value before layout-transitioning SDR (not device_wait_idle).
        if self.sdr_initialized {
            be.wait_transfer_src_consumers()?;
        }
        let t0 = std::time::Instant::now();
        let bytes = bytemuck::cast_slice(stars);
        self.input.mapped_mut()[..bytes.len()].copy_from_slice(bytes);
        let d = be.ctx.device.clone();
        let ext = vk::Extent2D { width: self.width, height: self.height };
        let mut push = [0u8; 16];
        push[..8].copy_from_slice(&self.input.address.to_le_bytes());
        push[8..12].copy_from_slice(&(self.width as f32).to_le_bytes());
        push[12..].copy_from_slice(&(self.height as f32).to_le_bytes());
        let mut exp_push = [0u8; 16];
        exp_push[..4].copy_from_slice(&exposure.to_le_bytes());
        let sdr_from =
            if self.sdr_initialized { vk::ImageLayout::TRANSFER_SRC_OPTIMAL } else { vk::ImageLayout::UNDEFINED };
        let hdr_initial = if !self.initialized {
            vk::ImageLayout::UNDEFINED
        } else if self.sdr_initialized {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        } else {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        };
        be.ctx.immediate_submit(|cmd| {
            use vk::AccessFlags2 as A;
            use vk::PipelineStageFlags2 as S;
            // SAFETY: live through fence wait in immediate_submit; layouts transitioned explicitly.
            unsafe {
                d.cmd_reset_query_pool(cmd, self.query, 0, 2);
                d.cmd_write_timestamp2(cmd, S::TOP_OF_PIPE, self.query, 0);
                memory_barrier(&d, cmd, (S::HOST, A::HOST_WRITE), (S::VERTEX_SHADER, A::SHADER_READ));
                image_barrier(
                    &d,
                    cmd,
                    self.color.image,
                    vk::ImageAspectFlags::COLOR,
                    (hdr_initial, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                    (
                        if self.initialized { S::ALL_COMMANDS } else { S::NONE },
                        if self.initialized { A::MEMORY_READ | A::MEMORY_WRITE } else { A::NONE },
                    ),
                    (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE),
                );
                let att = [vk::RenderingAttachmentInfo::default()
                    .image_view(self.color.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } })];
                d.cmd_begin_rendering(
                    cmd,
                    &vk::RenderingInfo::default()
                        .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: ext })
                        .layer_count(1)
                        .color_attachments(&att),
                );
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
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipe);
                d.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::VERTEX, 0, &push);
                d.cmd_draw(cmd, 6, stars.len() as u32, 0, 0);
                d.cmd_end_rendering(cmd);
                image_barrier(
                    &d,
                    cmd,
                    self.color.image,
                    vk::ImageAspectFlags::COLOR,
                    (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE),
                    (S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ),
                );
                image_barrier(
                    &d,
                    cmd,
                    self.sdr.image,
                    vk::ImageAspectFlags::COLOR,
                    (sdr_from, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                    (
                        if self.sdr_initialized { S::COPY } else { S::NONE },
                        if self.sdr_initialized { A::TRANSFER_READ } else { A::NONE },
                    ),
                    (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE),
                );
                let satt = [vk::RenderingAttachmentInfo::default()
                    .image_view(self.sdr.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } })];
                d.cmd_begin_rendering(
                    cmd,
                    &vk::RenderingInfo::default()
                        .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: ext })
                        .layer_count(1)
                        .color_attachments(&satt),
                );
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
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.tonemap_pipe);
                d.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.tonemap_layout,
                    0,
                    &[self.tonemap_set],
                    &[],
                );
                d.cmd_push_constants(cmd, self.tonemap_layout, vk::ShaderStageFlags::FRAGMENT, 0, &exp_push);
                d.cmd_draw(cmd, 3, 1, 0, 0);
                d.cmd_end_rendering(cmd);
                image_barrier(
                    &d,
                    cmd,
                    self.sdr.image,
                    vk::ImageAspectFlags::COLOR,
                    (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
                    (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE),
                    (S::ALL_TRANSFER, A::TRANSFER_READ),
                );
                d.cmd_write_timestamp2(cmd, S::ALL_COMMANDS, self.query, 1);
            }
        })?;
        self.initialized = true;
        self.sdr_initialized = true;
        let mut timing = UniverseTiming { submit_ms: t0.elapsed().as_secs_f64() * 1000.0, ..Default::default() };
        if let Some(period) = be.ctx.timestamp_period_ns {
            let mut ts = [0u64; 2];
            // SAFETY: queries written; wait flag.
            unsafe {
                d.get_query_pool_results(
                    self.query,
                    0,
                    &mut ts,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )?;
            }
            timing.gpu_ms = ts[1].saturating_sub(ts[0]) as f64 * period as f64 * 1e-6;
        }
        Ok(timing)
    }
    /// Explicit optional capture of the SDR present target (verification only).
    pub fn read_sdr_rgba(&mut self, be: &mut AshBackend) -> Result<Vec<u8>, BackendError> {
        if !self.sdr_initialized {
            return Err(BackendError::NothingRendered);
        }
        let size = self.width as u64 * self.height as u64 * 4;
        if self.sdr_readback.as_ref().is_none_or(|b| b.size < size) {
            if let Some(mut old) = self.sdr_readback.take() {
                old.destroy(&mut be.ctx);
            }
            self.sdr_readback = Some(GpuBuffer::new(
                &mut be.ctx,
                "universe_sdr_readback",
                size,
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
            )?);
        }
        let rb = self.sdr_readback.as_ref().unwrap().buffer;
        let img = self.sdr.image;
        let d = be.ctx.device.clone();
        let (w, h) = (self.width, self.height);
        be.ctx.immediate_submit(|cmd| {
            use vk::AccessFlags2 as A;
            use vk::PipelineStageFlags2 as S;
            image_barrier(
                &d,
                cmd,
                img,
                vk::ImageAspectFlags::COLOR,
                (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
                (S::ALL_COMMANDS, A::MEMORY_WRITE),
                (S::COPY, A::TRANSFER_READ),
            );
            let region = [vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_extent: vk::Extent3D { width: w, height: h, depth: 1 },
                ..Default::default()
            }];
            // SAFETY: SDR in TRANSFER_SRC; buffer sized.
            unsafe {
                d.cmd_copy_image_to_buffer(cmd, img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, rb, &region);
            }
            memory_barrier(&d, cmd, (S::COPY, A::TRANSFER_WRITE), (S::HOST, A::HOST_READ));
        })?;
        Ok(self.sdr_readback.as_ref().unwrap().mapped()[..size as usize].to_vec())
    }
    pub fn free(mut self, be: &mut AshBackend) {
        be.ctx.wait_idle();
        // SAFETY: synchronous rendering completed, no commands reference these objects.
        unsafe {
            be.ctx.device.destroy_pipeline(self.pipe, None);
            be.ctx.device.destroy_pipeline_layout(self.layout, None);
            be.ctx.device.destroy_query_pool(self.query, None);
        }
        destroy_tonemap(&be.ctx.device, &self);
        self.input.destroy(&mut be.ctx);
        self.color.destroy(&mut be.ctx);
        self.sdr.destroy(&mut be.ctx);
        self.readback.destroy(&mut be.ctx);
        if let Some(mut b) = self.sdr_readback.take() {
            b.destroy(&mut be.ctx);
        }
    }
}
fn pipeline(d: &ash::Device, layout: vk::PipelineLayout) -> Result<vk::Pipeline, BackendError> {
    // SAFETY: device valid; create infos reference live locals; modules destroyed after creation.
    unsafe {
        let vsm = crate::pass::shader_module(d, UNIVERSE_VS)?;
        let fsm = match crate::pass::shader_module(d, UNIVERSE_FS) {
            Ok(m) => m,
            Err(e) => {
                d.destroy_shader_module(vsm, None);
                return Err(e);
            }
        };
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
        let blend = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
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
        let r = d.create_graphics_pipelines(vk::PipelineCache::null(), &[gci], None).map_err(|(partial, e)| {
            for p in partial {
                d.destroy_pipeline(p, None);
            }
            e
        });
        d.destroy_shader_module(vsm, None);
        d.destroy_shader_module(fsm, None);
        Ok(r?[0])
    }
}
fn tonemap_pipeline(d: &ash::Device, layout: vk::PipelineLayout) -> Result<vk::Pipeline, BackendError> {
    // SAFETY: device valid; modules destroyed after create.
    unsafe {
        let vsm = crate::pass::shader_module(d, TONEMAP_VS)?;
        let fsm = match crate::pass::shader_module(d, TONEMAP_FS) {
            Ok(m) => m,
            Err(e) => {
                d.destroy_shader_module(vsm, None);
                return Err(e);
            }
        };
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
        let blend = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
        let dyns = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dy = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyns);
        let cf = [SDR];
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
        let r = d.create_graphics_pipelines(vk::PipelineCache::null(), &[gci], None).map_err(|(partial, e)| {
            for p in partial {
                d.destroy_pipeline(p, None);
            }
            e
        });
        d.destroy_shader_module(vsm, None);
        d.destroy_shader_module(fsm, None);
        Ok(r?[0])
    }
}
