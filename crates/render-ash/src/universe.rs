//! Persistent bounded additive HDR splats. Explicit `free` before the backend is dropped.
use crate::context::{GpuBuffer, GpuImage, image_barrier, memory_barrier};
use crate::{AshBackend, BackendError};
use ash::vk;
use gpu_allocator::MemoryLocation;
use planet::universe::{MAX_STARS, SIGMA, Source};
const UNIVERSE_VS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/universe_vs.spv"));
const UNIVERSE_FS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/universe_fs.spv"));
const COLOR: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
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
}
pub struct UniversePass {
    input: GpuBuffer,
    color: GpuImage,
    readback: GpuBuffer,
    layout: vk::PipelineLayout,
    pipe: vk::Pipeline,
    query: vk::QueryPool,
    width: u32,
    height: u32,
    initialized: bool,
}
fn extent_ok(w: u32, h: u32) -> Result<(), BackendError> {
    if w == 0 || h == 0 || w > 4096 || h > 4096 {
        Err(BackendError::BadConfig("universe extent must be 1..4096"))
    } else {
        Ok(())
    }
}
fn target(be: &mut AshBackend, w: u32, h: u32) -> Result<(GpuImage, GpuBuffer), BackendError> {
    let mut color = GpuImage::new(
        &mut be.ctx,
        "universe_hdr",
        vk::Extent2D { width: w, height: h },
        COLOR,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
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
        let (mut color, mut readback) = match target(be, width, height) {
            Ok(t) => t,
            Err(e) => {
                input.destroy(&mut be.ctx);
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
                    d.destroy_pipeline(pipe, None);
                    d.destroy_pipeline_layout(layout, None)
                };
                input.destroy(&mut be.ctx);
                color.destroy(&mut be.ctx);
                readback.destroy(&mut be.ctx);
                return Err(e.into());
            }
        };
        Ok(Self { input, color, readback, layout, pipe, query, width, height, initialized: false })
    }
    pub fn resize(&mut self, be: &mut AshBackend, w: u32, h: u32) -> Result<(), BackendError> {
        extent_ok(w, h)?;
        if (w, h) == (self.width, self.height) {
            return Ok(());
        }
        let (color, rb) = target(be, w, h)?;
        be.ctx.wait_idle();
        self.color.destroy(&mut be.ctx);
        self.readback.destroy(&mut be.ctx);
        self.color = color;
        self.readback = rb;
        self.width = w;
        self.height = h;
        self.initialized = false;
        Ok(())
    }
    /// Borrows the persistent mapped readback until the next mutable pass operation.
    /// Flux is unexposed and clipped at the viewport without edge renormalization.
    pub fn render(
        &mut self,
        be: &mut AshBackend,
        stars: &[Splat],
    ) -> Result<(&[[f32; 4]], UniverseTiming), BackendError> {
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
                image_barrier(
                    &d,
                    cmd,
                    self.color.image,
                    vk::ImageAspectFlags::COLOR,
                    (
                        if initialized { vk::ImageLayout::TRANSFER_SRC_OPTIMAL } else { vk::ImageLayout::UNDEFINED },
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    ),
                    (if initialized { S::COPY } else { S::NONE }, if initialized { A::TRANSFER_READ } else { A::NONE }),
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
    pub fn free(mut self, be: &mut AshBackend) {
        be.ctx.wait_idle();
        // SAFETY: synchronous rendering completed, no commands reference these objects.
        unsafe {
            be.ctx.device.destroy_pipeline(self.pipe, None);
            be.ctx.device.destroy_pipeline_layout(self.layout, None);
            be.ctx.device.destroy_query_pool(self.query, None);
        }
        self.input.destroy(&mut be.ctx);
        self.color.destroy(&mut be.ctx);
        self.readback.destroy(&mut be.ctx);
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
            // Vulkan may return owned partial handles even when pipeline creation fails.
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
