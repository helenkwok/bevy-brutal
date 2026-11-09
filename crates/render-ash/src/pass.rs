//! The GPU-driven instanced pass and the `AshBackend` that owns it.
//!
//! Per frame, all on the GPU and all in the frame slot's own buffers:
//! 1. `cull`: one thread per instance; frustum test; visible ones take a rank in their mesh
//!    (`atomicAdd(counts[mesh])`) -> `slots[i]` (or INVALID).
//! 2. `scan`: multi-level exclusive prefix sum of `counts` -> `offsets` (tiles of 1024; tile totals
//!    are scanned recursively, then propagated back down).
//! 3. `scatter`: one thread per instance, `visible[offsets[mesh] + slots[i]] = i`; `compact`: one
//!    thread per mesh, non-empty meshes append a `VkDrawIndexedIndirectCommand`.
//! 4. `vkCmdDrawIndexedIndirectCount` into the slot's own colour/depth targets.
//!
//! Frame overlap: every resource a frame touches on the GPU (scratch, targets, stats readback, query
//! pool, command pool, uploaded frame/instance data) belongs to one of `frames_in_flight` slots. A slot
//! is only reused after the host has waited on the timeline value of its previous frame, and no frame
//! records a barrier whose first scope reaches back into another frame, so frames N and N+1 have no
//! GPU-side dependency on each other.
use std::io::Cursor;

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use gpu_allocator::MemoryLocation;
use lowlevel_abi::{FrameSnapshot, InstanceGpu, RenderBackend};

use crate::context::{GpuBuffer, GpuImage, SubgroupInfo, VkContext, WindowHandles, image_barrier, memory_barrier};
use crate::cull::frustum_planes;
use crate::mesh::{MeshData, MeshInfoGpu, MeshVertex, builtin_meshes};
use crate::surface::{
    SURFACE_VERTEX_BYTES, StaticMesh, SurfacePalette, TerrainGpu, TerrainGpuConfig, TerrainSlotKey, UploadOutcome,
    UploadStats, upload_static,
};
use crate::{BackendError, MAX_FRAMES_IN_FLIGHT, MIN_FRAMES_IN_FLIGHT};

/// Capacity of the mesh table (and of per-slot counts/offsets/draws). The per-mesh scan is a
/// 2-level scan at this size (65536 / 1024 = 64 tile totals).
pub const MAX_MESHES: usize = 1 << 16;
/// Elements per scan workgroup (256 threads x 4). Must match TILE in shaders/scan.slang.
pub const SCAN_TILE: u32 = 1024;
const COLOR_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
/// t0 start, t1 after cull/count, t2 after scan, t3 after scatter+args, t4 after draw.
const QUERIES: u32 = 5;
const DRAW_STRIDE: u32 = size_of::<vk::DrawIndexedIndirectCommand>() as u32; // 20
const STATS_DRAWS_OFFSET: u64 = 16;
const CLEAR: [f32; 4] = [0.05, 0.06, 0.09, 1.0];

// Slang -> SPIR-V, compiled by build.rs (shaders/*.slang). All entry points are emitted as "main".
const CULL_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cull.spv"));
const SCAN_BLOCKS_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scan_blocks.spv"));
const SCAN_BLOCKS_WAVE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scan_blocks_wave.spv"));
const SCAN_ADD_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scan_add.spv"));
const SCATTER_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scatter.spv"));
const COMPACT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/compact.spv"));
const SURFACE_VS_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/surface_vs.spv"));
const SURFACE_FS_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/surface_fs.spv"));
const VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mesh_vs.spv"));
const FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mesh_fs.spv"));

/// Mirrors `Frame` in shaders/common.slang (176 bytes).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FrameGpu {
    view_proj: [[f32; 4]; 4],
    planes: [[f32; 4]; 6],
    instance_count: u32,
    mesh_count: u32,
    /// Surface palette height range in metres (`hi <= lo`: noise palette); see `SurfacePalette`.
    surface_range: [f32; 2],
}

/// Mirrors `Push` in shaders/common.slang (9 buffer device addresses, 72 bytes).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PushGpu {
    frame: u64,
    instances: u64,
    meshes: u64,
    counts: u64,
    offsets: u64,
    slots: u64,
    visible: u64,
    draws: u64,
    draw_count: u64,
}

/// Mirrors `ScanPush` in shaders/scan.slang (32 bytes).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ScanPushGpu {
    input: u64,
    output: u64,
    sums: u64,
    n: u32,
    has_sums: u32,
}

// Layout contract with shaders/*.slang (offsets verified with spirv-dis).
const _: () = assert!(size_of::<FrameGpu>() == 176 && size_of::<PushGpu>() == 72 && size_of::<ScanPushGpu>() == 32);
const _: () = assert!(size_of::<InstanceGpu>() == 80 && size_of::<MeshInfoGpu>() == 32 && DRAW_STRIDE == 20);

/// Element counts per scan level: `[n, ceil(n/T), ceil(n/T^2), ...]` down to a single tile.
pub(crate) fn scan_levels(n: u32) -> Vec<u32> {
    let mut v = vec![n];
    while let Some(&last) = v.last()
        && last > SCAN_TILE
    {
        v.push(last.div_ceil(SCAN_TILE));
    }
    v
}

/// CPU reference: wrapping exclusive prefix sum.
pub fn cpu_exclusive_scan(data: &[u32]) -> Vec<u32> {
    let mut acc = 0u32;
    data.iter()
        .map(|&x| {
            let r = acc;
            acc = acc.wrapping_add(x);
            r
        })
        .collect()
}

/// Which tile kernel the GPU scan uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScanMode {
    /// Subgroup arithmetic when the device supports it (see `ScanPath`), else shared memory.
    #[default]
    Auto,
    /// Require the subgroup path; backend creation fails with `Unsupported` if unavailable.
    Subgroup,
    /// Force the shared-memory (Hillis-Steele) fallback.
    SharedMemory,
}

impl ScanMode {
    /// `BRUTAL_SCAN=subgroup|shared|auto` (default auto).
    pub fn from_env() -> Self {
        match std::env::var("BRUTAL_SCAN").as_deref() {
            Ok("subgroup") => Self::Subgroup,
            Ok("shared") => Self::SharedMemory,
            _ => Self::Auto,
        }
    }
}

/// The scan path actually selected at runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanPath {
    /// Subgroup kernel with full subgroups. `required_size` is set when subgroupSizeControl allowed
    /// pinning it; otherwise the size may vary per dispatch (ALLOW_VARYING) and the kernel adapts.
    Subgroup {
        required_size: Option<u32>,
    },
    SharedMemory,
}

/// Decide whether (and how) the subgroup scan kernel can run on this device.
pub fn plan_subgroup_scan(sg: &SubgroupInfo) -> Result<Option<u32>, &'static str> {
    const WG: u32 = 256;
    if !sg.compute_arithmetic() {
        return Err("no subgroup arithmetic in compute");
    }
    if !sg.compute_full_subgroups {
        return Err("computeFullSubgroups unsupported (kernel needs full subgroups)");
    }
    if sg.size_control && sg.required_size_stages.contains(vk::ShaderStageFlags::COMPUTE) && sg.min_size > 0 {
        let mut r = sg.max_size;
        while r >= sg.min_size && r > 0 {
            if r.is_power_of_two() && WG.is_multiple_of(r) && WG <= r * sg.max_compute_workgroup_subgroups {
                return Ok(Some(r));
            }
            r /= 2;
        }
    }
    // No pinned size: with REQUIRE_FULL_SUBGROUPS the workgroup must be a multiple of maxSubgroupSize.
    if sg.max_size > 0 && sg.max_size.is_power_of_two() && WG.is_multiple_of(sg.max_size) {
        return Ok(None);
    }
    Err("no subgroup size divides the 256-thread workgroup")
}

#[derive(Clone, Copy, Debug)]
pub struct BackendConfig {
    pub max_instances: usize,
    pub extent: (u32, u32),
    /// Enable VK_LAYER_KHRONOS_validation (+ sync validation) if installed.
    pub validation: bool,
    /// Frames that may be in flight at once (2..=3). Each gets its own scratch + targets.
    pub frames_in_flight: usize,
    /// Scan tile kernel selection (env `BRUTAL_SCAN`).
    pub scan: ScanMode,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            max_instances: 1 << 20,
            extent: (1280, 720),
            validation: cfg!(debug_assertions) || std::env::var("BRUTAL_VK_VALIDATION").is_ok_and(|v| v == "1"),
            // BRUTAL_FRAMES_IN_FLIGHT=3 switches apps using the default config; invalid values fall back to 2.
            frames_in_flight: std::env::var("BRUTAL_FRAMES_IN_FLIGHT").ok().and_then(|v| v.parse().ok()).unwrap_or(2),
            scan: ScanMode::from_env(),
        }
    }
}

/// Results of one completed frame, read back after its timeline value was reached.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct FrameStats {
    pub frame: u64,
    pub instances: u32,
    pub visible: u32,
    pub draws: u32,
    /// Whole compute chain (cull+count, scan, scatter+args) = t0..t3.
    pub cull_ms: Option<f64>,
    pub count_ms: Option<f64>,
    pub scan_ms: Option<f64>,
    pub scatter_ms: Option<f64>,
    pub draw_ms: Option<f64>,
    /// Absolute GPU timestamps (ns) of the frame's first and last query, for overlap analysis.
    pub gpu_begin_ns: Option<f64>,
    pub gpu_end_ns: Option<f64>,
    /// Frames submitted but not yet completed right after this frame's submit (including itself),
    /// from the timeline counter. >= 2 means this frame was queued while an earlier one was running.
    pub in_flight_at_submit: u64,
    /// Timeline counter value observed right after this frame's submit.
    pub completed_at_submit: u64,
}

/// Everything one frame touches on the GPU.
struct FrameSlot {
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    queries: vk::QueryPool,
    frame_buf: GpuBuffer,
    inst_buf: GpuBuffer,
    counts: GpuBuffer,
    offsets: GpuBuffer,
    scan_scratch: Vec<GpuBuffer>,
    inst_slots: GpuBuffer,
    visible: GpuBuffer,
    draws: GpuBuffer,
    draw_count: GpuBuffer,
    stats_buf: GpuBuffer,
    color: GpuImage,
    depth: GpuImage,
    /// Timeline value signalled by this slot's last submit (0 = never submitted).
    submitted: u64,
    instances: u32,
    in_flight_at_submit: u64,
    completed_at_submit: u64,
}

impl FrameSlot {
    fn buffers_mut(&mut self) -> impl Iterator<Item = &mut GpuBuffer> {
        [
            &mut self.frame_buf,
            &mut self.inst_buf,
            &mut self.counts,
            &mut self.offsets,
            &mut self.inst_slots,
            &mut self.visible,
            &mut self.draws,
            &mut self.draw_count,
            &mut self.stats_buf,
        ]
        .into_iter()
        .chain(self.scan_scratch.iter_mut())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Pipelines {
    compute_layout: vk::PipelineLayout,
    scan_layout: vk::PipelineLayout,
    gfx_layout: vk::PipelineLayout,
    cull: vk::Pipeline,
    /// Tile kernel in use: the subgroup one or the shared-memory fallback.
    scan_blocks: vk::Pipeline,
    scan_blocks_shared: vk::Pipeline,
    scan_blocks_wave: Option<vk::Pipeline>,
    scan_path: ScanPath,
    scan_add: vk::Pipeline,
    scatter: vk::Pipeline,
    compact: vk::Pipeline,
    gfx: vk::Pipeline,
    surface: vk::Pipeline,
}

/// Record a multi-level exclusive scan of `n` u32 at `input` into `output` (may alias).
/// `scratch[k]` holds the tile totals of level k (needs `scan_levels(n)[k+1]` u32).
/// Ends with a compute->compute barrier; the caller adds the barrier to its consumer.
pub(crate) fn record_scan(
    d: &ash::Device,
    p: &Pipelines,
    cmd: vk::CommandBuffer,
    input: u64,
    output: u64,
    n: u32,
    scratch: &[u64],
) {
    use vk::AccessFlags2 as A;
    use vk::PipelineStageFlags2 as S;
    if n == 0 {
        return;
    }
    let sizes = scan_levels(n);
    assert!(scratch.len() + 1 >= sizes.len(), "scan scratch too small");
    let rw = (S::COMPUTE_SHADER, A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE);
    let push = |pc: &ScanPushGpu| {
        // SAFETY: cmd is recording with the scan layout's 32-byte compute push range.
        unsafe { d.cmd_push_constants(cmd, p.scan_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(pc)) };
    };
    // SAFETY: cmd is recording; pipelines/layouts are live; every address points at a buffer with room
    // for the element counts in `sizes` (asserted above / by the caller's allocation).
    unsafe {
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.scan_blocks);
        for (k, &len) in sizes.iter().enumerate() {
            let (inp, out) = if k == 0 { (input, output) } else { (scratch[k - 1], scratch[k - 1]) };
            let has_sums = k + 1 < sizes.len();
            push(&ScanPushGpu {
                input: inp,
                output: out,
                sums: if has_sums { scratch[k] } else { 0 },
                n: len,
                has_sums: has_sums as u32,
            });
            d.cmd_dispatch(cmd, len.div_ceil(SCAN_TILE), 1, 1);
            memory_barrier(d, cmd, rw, rw);
        }
        if sizes.len() > 1 {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.scan_add);
            for k in (0..sizes.len() - 1).rev() {
                let out = if k == 0 { output } else { scratch[k - 1] };
                push(&ScanPushGpu { input: out, output: out, sums: scratch[k], n: sizes[k], has_sums: 1 });
                d.cmd_dispatch(cmd, sizes[k].div_ceil(SCAN_TILE), 1, 1);
                memory_barrier(d, cmd, rw, rw);
            }
        }
    }
}

/// GPU-driven instanced renderer implementing `RenderBackend`.
pub struct AshBackend {
    slots: Vec<FrameSlot>,
    timeline: vk::Semaphore,
    frame_number: u64,
    pub(crate) pipes: Pipelines,
    mesh_buf: GpuBuffer,
    vertex_buf: Option<GpuBuffer>,
    index_buf: Option<GpuBuffer>,
    meshes: Vec<MeshInfoGpu>,
    readback: Option<GpuBuffer>,
    pub extent: vk::Extent2D,
    max_instances: usize,
    last_stats: Option<FrameStats>,
    /// Test hook: extra (semaphore, value) the next submit waits on (see `gate_next_submit`).
    gate: Option<(vk::Semaphore, u64)>,
    terrain: Option<TerrainGpu>,
    surface_palette: SurfacePalette,
    /// Quality-frame shadow maps kept between frames (cascade caching).
    pub(crate) qshadow: Option<crate::quality::QShadowCache>,
    /// Earth round 5: the quality pass's ground map (real imagery albedo), see
    /// `AshBackend::quality_set_ground_map`.
    pub(crate) qground: Option<crate::context::GpuBuffer>,
    statics: Vec<StaticMesh>,
    #[cfg(feature = "windowed")]
    swapchain: Option<crate::swapchain::Swapchain>,
    #[cfg(feature = "windowed")]
    needs_recreate: bool,
    /// Timeline value that must complete before rewriting an external TRANSFER_SRC image last consumed by blit/copy.
    transfer_src_read_until: u64,
    /// Offscreen R8G8B8A8 target used by `submit_transfer_src_read` (blit dst).
    transfer_probe: Option<GpuImage>,
    /// Declared last: dropped after everything above has been destroyed in `Drop`.
    pub ctx: VkContext,
}

pub(crate) fn shader_module(d: &ash::Device, spv: &[u8]) -> Result<vk::ShaderModule, BackendError> {
    let words = ash::util::read_spv(&mut Cursor::new(spv)).map_err(BackendError::Shader)?;
    // SAFETY: `words` is valid SPIR-V produced by slangc from shaders/*.slang (validated with spirv-val).
    Ok(unsafe { d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)? })
}

fn create_pipelines(d: &ash::Device, wave: Option<Option<u32>>) -> Result<Pipelines, BackendError> {
    let range = |stage, size: usize| [vk::PushConstantRange { stage_flags: stage, offset: 0, size: size as u32 }];
    let cpush = range(vk::ShaderStageFlags::COMPUTE, size_of::<PushGpu>());
    let spush = range(vk::ShaderStageFlags::COMPUTE, size_of::<ScanPushGpu>());
    let gpush = range(vk::ShaderStageFlags::VERTEX, size_of::<PushGpu>());
    // SAFETY: device valid; create infos reference live arrays. Shader modules are destroyed after
    // pipeline creation, which is allowed.
    unsafe {
        let layout = |r: &[vk::PushConstantRange]| {
            d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(r), None)
        };
        let compute_layout = layout(&cpush)?;
        let scan_layout = layout(&spush)?;
        let gfx_layout = layout(&gpush)?;

        let srcs = [
            (CULL_SPV, compute_layout),
            (SCAN_BLOCKS_SPV, scan_layout),
            (SCAN_ADD_SPV, scan_layout),
            (SCATTER_SPV, compute_layout),
            (COMPACT_SPV, compute_layout),
        ];
        let mods = srcs.iter().map(|(spv, _)| shader_module(d, spv)).collect::<Result<Vec<_>, _>>()?;
        let infos: Vec<_> = mods
            .iter()
            .zip(&srcs)
            .map(|(&m, &(_, l))| {
                vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(m)
                            .name(c"main"),
                    )
                    .layout(l)
            })
            .collect();
        let cps = d.create_compute_pipelines(vk::PipelineCache::null(), &infos, None).map_err(|(_, e)| e)?;
        for m in mods {
            d.destroy_shader_module(m, None);
        }

        // Subgroup scan kernel: full subgroups always; pinned size when allowed, else varying.
        let (scan_blocks_wave, scan_path) = match wave {
            None => (None, ScanPath::SharedMemory),
            Some(required) => {
                let m = shader_module(d, SCAN_BLOCKS_WAVE_SPV)?;
                let mut req = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                    .required_subgroup_size(required.unwrap_or(0));
                let mut flags = vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS;
                if required.is_none() {
                    flags |= vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE;
                }
                let mut stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(m)
                    .name(c"main")
                    .flags(flags);
                if required.is_some() {
                    stage = stage.push_next(&mut req);
                }
                let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(scan_layout);
                let p = d.create_compute_pipelines(vk::PipelineCache::null(), &[info], None).map_err(|(_, e)| e);
                d.destroy_shader_module(m, None);
                (Some(p?[0]), ScanPath::Subgroup { required_size: required })
            }
        };

        let gfx = gfx_pipeline(d, gfx_layout, VERT_SPV, FRAG_SPV)?;
        let surface = gfx_pipeline(d, gfx_layout, SURFACE_VS_SPV, SURFACE_FS_SPV)?;
        Ok(Pipelines {
            compute_layout,
            scan_layout,
            gfx_layout,
            cull: cps[0],
            scan_blocks: scan_blocks_wave.unwrap_or(cps[1]),
            scan_blocks_shared: cps[1],
            scan_blocks_wave,
            scan_path,
            scan_add: cps[2],
            scatter: cps[3],
            compact: cps[4],
            gfx,
            surface,
        })
    }
}

/// Graphics pipeline over pos+normal vertices (24-byte stride) into COLOR_FORMAT + DEPTH_FORMAT.
fn gfx_pipeline(
    d: &ash::Device,
    gfx_layout: vk::PipelineLayout,
    vs_spv: &[u8],
    fs_spv: &[u8],
) -> Result<vk::Pipeline, BackendError> {
    // SAFETY: device valid; create infos reference live locals; modules destroyed after creation.
    unsafe {
        let vs = shader_module(d, vs_spv)?;
        let fs = shader_module(d, fs_spv)?;
        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vs).name(c"main"),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(fs).name(c"main"),
        ];
        let bindings = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<MeshVertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let attrs = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 12,
            },
        ];
        let vi = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attrs);
        let ia = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let vp = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let rs = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let ms = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let ds = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);
        let blend = [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA)];
        let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
        let dyns = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dy = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyns);
        let color_formats = [COLOR_FORMAT];
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(DEPTH_FORMAT);
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
            .layout(gfx_layout)
            .push_next(&mut rendering);
        let gfx = d.create_graphics_pipelines(vk::PipelineCache::null(), &[gci], None).map_err(|(_, e)| e);
        d.destroy_shader_module(vs, None);
        d.destroy_shader_module(fs, None);
        Ok(gfx?[0])
    }
}

fn targets(ctx: &mut VkContext, extent: vk::Extent2D) -> Result<(GpuImage, GpuImage), BackendError> {
    let color = GpuImage::new(
        ctx,
        "color",
        extent,
        COLOR_FORMAT,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::ImageAspectFlags::COLOR,
    )?;
    let depth = GpuImage::new(
        ctx,
        "depth",
        extent,
        DEPTH_FORMAT,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::ImageAspectFlags::DEPTH,
    )?;
    Ok((color, depth))
}

fn new_slot(ctx: &mut VkContext, max_instances: usize, extent: vk::Extent2D) -> Result<FrameSlot, BackendError> {
    let sda = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER;
    let pci = vk::CommandPoolCreateInfo::default()
        .queue_family_index(ctx.queue_family)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    // SAFETY: device valid; create infos fully initialised.
    let (pool, cmd, queries) = unsafe {
        let pool = ctx.device.create_command_pool(&pci, None)?;
        let cmd = ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1),
        )?[0];
        let queries = ctx.device.create_query_pool(
            &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(QUERIES),
            None,
        )?;
        (pool, cmd, queries)
    };
    let mesh_bytes = (MAX_MESHES * 4) as u64;
    let draw_bytes = MAX_MESHES as u64 * DRAW_STRIDE as u64;
    let scan_scratch = scan_levels(MAX_MESHES as u32)[1..]
        .iter()
        .map(|&len| GpuBuffer::new(ctx, "scan_scratch", len as u64 * 4, sda, MemoryLocation::GpuOnly))
        .collect::<Result<Vec<_>, _>>()?;
    let (color, depth) = targets(ctx, extent)?;
    Ok(FrameSlot {
        pool,
        cmd,
        queries,
        frame_buf: GpuBuffer::new(ctx, "frame", size_of::<FrameGpu>() as u64, sda, MemoryLocation::CpuToGpu)?,
        inst_buf: GpuBuffer::new(
            ctx,
            "instances",
            (max_instances * size_of::<InstanceGpu>()) as u64,
            sda,
            MemoryLocation::CpuToGpu,
        )?,
        counts: GpuBuffer::new(
            ctx,
            "counts",
            mesh_bytes,
            sda | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?,
        offsets: GpuBuffer::new(ctx, "offsets", mesh_bytes, sda, MemoryLocation::GpuOnly)?,
        scan_scratch,
        inst_slots: GpuBuffer::new(ctx, "instance_slots", (max_instances * 4) as u64, sda, MemoryLocation::GpuOnly)?,
        visible: GpuBuffer::new(ctx, "visible", (max_instances * 4) as u64, sda, MemoryLocation::GpuOnly)?,
        draws: GpuBuffer::new(
            ctx,
            "draws",
            draw_bytes,
            sda | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuOnly,
        )?,
        draw_count: GpuBuffer::new(
            ctx,
            "draw_count",
            16,
            sda | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?,
        stats_buf: GpuBuffer::new(
            ctx,
            "stats",
            STATS_DRAWS_OFFSET + draw_bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
        )?,
        color,
        depth,
        submitted: 0,
        instances: 0,
        in_flight_at_submit: 0,
        completed_at_submit: 0,
    })
}

impl AshBackend {
    /// Headless backend rendering into an offscreen target (default 1280x720).
    pub fn new_headless(max_instances: usize) -> Result<Self, BackendError> {
        Self::new(BackendConfig { max_instances, ..Default::default() }, None)
    }

    /// Backend presenting to a window (feature `windowed`). `extent` should be the window's physical size.
    #[cfg(feature = "windowed")]
    pub fn new_windowed(handles: WindowHandles, cfg: BackendConfig) -> Result<Self, BackendError> {
        let mut b = Self::new(cfg, Some(handles))?;
        let sc = crate::swapchain::Swapchain::new(&b.ctx, b.extent, vk::SwapchainKHR::null(), b.slots.len())?;
        let ext = sc.extent;
        b.swapchain = Some(sc);
        if ext != b.extent {
            b.recreate_targets(ext)?;
        }
        Ok(b)
    }

    pub fn new(cfg: BackendConfig, window: Option<WindowHandles>) -> Result<Self, BackendError> {
        if !(MIN_FRAMES_IN_FLIGHT..=MAX_FRAMES_IN_FLIGHT).contains(&cfg.frames_in_flight) {
            return Err(BackendError::BadConfig("frames_in_flight must be 2 or 3"));
        }
        let mut ctx = VkContext::new(window, cfg.validation)?;
        let max_instances = cfg.max_instances.max(1);
        let extent = vk::Extent2D { width: cfg.extent.0.max(1), height: cfg.extent.1.max(1) };
        let wave = match cfg.scan {
            ScanMode::SharedMemory => None,
            ScanMode::Auto => plan_subgroup_scan(&ctx.subgroup).ok(),
            ScanMode::Subgroup => Some(plan_subgroup_scan(&ctx.subgroup).map_err(BackendError::Unsupported)?),
        };
        let pipes = create_pipelines(&ctx.device, wave)?;
        let slots = (0..cfg.frames_in_flight)
            .map(|_| new_slot(&mut ctx, max_instances, extent))
            .collect::<Result<Vec<_>, _>>()?;
        let mut tci =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
        // SAFETY: device valid; timeline semaphores are enabled.
        let timeline =
            unsafe { ctx.device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut tci), None)? };
        let mesh_buf = GpuBuffer::new(
            &mut ctx,
            "mesh_table",
            (MAX_MESHES * size_of::<MeshInfoGpu>()) as u64,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu,
        )?;
        let mut b = Self {
            slots,
            timeline,
            frame_number: 0,
            pipes,
            mesh_buf,
            vertex_buf: None,
            index_buf: None,
            meshes: Vec::new(),
            readback: None,
            extent,
            max_instances,
            last_stats: None,
            gate: None,
            terrain: None,
            surface_palette: SurfacePalette::Noise,
            qshadow: None,
            qground: None,
            statics: Vec::new(),
            #[cfg(feature = "windowed")]
            swapchain: None,
            #[cfg(feature = "windowed")]
            needs_recreate: false,
            transfer_src_read_until: 0,
            transfer_probe: None,
            ctx,
        };
        b.set_meshes(&builtin_meshes())?;
        Ok(b)
    }

    pub fn device_name(&self) -> &str {
        &self.ctx.device_name
    }

    pub fn max_instances(&self) -> usize {
        self.max_instances
    }

    /// Scan tile kernel selected at runtime.
    pub fn scan_path(&self) -> ScanPath {
        self.pipes.scan_path
    }

    pub fn frames_in_flight(&self) -> usize {
        self.slots.len()
    }

    pub fn mesh_table(&self) -> &[MeshInfoGpu] {
        &self.meshes
    }

    /// Stats of the most recently retired frame (collected when its slot was reused or on `finish`).
    pub fn last_stats(&self) -> Option<FrameStats> {
        self.last_stats
    }

    /// Current timeline counter (= number of the last completed frame).
    pub fn completed_frame(&self) -> Result<u64, BackendError> {
        // SAFETY: timeline semaphore is live.
        Ok(unsafe { self.ctx.device.get_semaphore_counter_value(self.timeline)? })
    }

    /// Enable the terrain residency (slots + staging ring) with the shared chunk index list.
    pub fn enable_terrain(&mut self, cfg: TerrainGpuConfig, grid_indices: &[u32]) -> Result<(), BackendError> {
        self.ctx.wait_idle();
        if let Some(mut t) = self.terrain.take() {
            t.destroy(&mut self.ctx);
        }
        self.terrain = Some(TerrainGpu::new(&mut self.ctx, cfg, grid_indices)?);
        Ok(())
    }

    /// Stage one chunk's vertices (`TerrainVertex`/24-byte pos+normal, exactly `verts_per_slot`)
    /// for `coord` (`[i32; 2]` = LOD 0, or a `TerrainSlotKey`). The copy is recorded into the next
    /// `render()`. Returns where it went and what was evicted.
    pub fn upload_terrain_chunk<V: bytemuck::Pod>(
        &mut self,
        coord: impl Into<TerrainSlotKey>,
        vertices: &[V],
    ) -> Result<UploadOutcome, BackendError> {
        const { assert!(size_of::<V>() == SURFACE_VERTEX_BYTES) };
        let completed = self.completed_frame()?;
        let next = self.frame_number + 1;
        let t = self.terrain.as_mut().ok_or(BackendError::BadConfig("terrain not enabled"))?;
        t.upload(coord.into(), bytemuck::cast_slice(vertices), next, completed)
    }

    pub fn terrain_is_resident(&self, coord: impl Into<TerrainSlotKey>) -> bool {
        let k = coord.into();
        self.terrain.as_ref().is_some_and(|t| t.is_resident(k))
    }

    /// Chunks to draw from the next frame on; returns how many are resident (others are skipped).
    pub fn set_terrain_draw<K: Into<TerrainSlotKey> + Copy>(&mut self, coords: &[K]) -> usize {
        self.terrain.as_mut().map_or(0, |t| t.set_draw(coords.iter().map(|&c| c.into())))
    }

    /// Terrain residency config (slots, `verts_per_slot`, staging), if enabled.
    pub fn terrain_config(&self) -> Option<TerrainGpuConfig> {
        self.terrain.as_ref().map(|t| t.cfg)
    }

    /// Colouring of the surface pass (terrain slots + static meshes) from the next frame on.
    pub fn set_surface_palette(&mut self, palette: SurfacePalette) {
        self.surface_palette = palette;
    }

    pub fn surface_palette(&self) -> SurfacePalette {
        self.surface_palette
    }

    pub fn terrain_stats(&self) -> Option<UploadStats> {
        self.terrain.as_ref().map(|t| t.stats)
    }

    /// Add a static world-space mesh (24-byte pos+normal vertices), drawn every frame. Waits idle.
    pub fn add_static_mesh<V: bytemuck::Pod>(&mut self, vertices: &[V], indices: &[u32]) -> Result<(), BackendError> {
        const { assert!(size_of::<V>() == SURFACE_VERTEX_BYTES) };
        if indices.is_empty() {
            return Ok(());
        }
        self.ctx.wait_idle();
        let vb = upload_static(
            &mut self.ctx,
            "static_vertices",
            bytemuck::cast_slice(vertices),
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        let ib = upload_static(
            &mut self.ctx,
            "static_indices",
            bytemuck::cast_slice(indices),
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;
        self.statics.push(StaticMesh { vertices: vb, indices: ib, index_count: indices.len() as u32 });
        Ok(())
    }

    pub fn clear_static_meshes(&mut self) {
        self.ctx.wait_idle();
        for mut m in std::mem::take(&mut self.statics) {
            m.vertices.destroy(&mut self.ctx);
            m.indices.destroy(&mut self.ctx);
        }
    }

    /// Test/diagnostic hook: make the next submitted frame wait (at TOP_OF_PIPE / all commands) on
    /// timeline `semaphore` reaching `value`. The caller must signal it, and must not render more than
    /// `frames_in_flight - 1` further frames before doing so (slot reuse would wait on the gated frame).
    #[doc(hidden)]
    pub fn gate_next_submit(&mut self, semaphore: vk::Semaphore, value: u64) {
        self.gate = Some((semaphore, value));
    }

    fn slot_of(&self, frame: u64) -> usize {
        (frame % self.slots.len() as u64) as usize
    }

    /// Replace the mesh set. Packs all meshes into one vertex + one index buffer (device-local,
    /// uploaded through a staging buffer). Waits for the GPU to go idle.
    pub fn set_meshes(&mut self, meshes: &[MeshData]) -> Result<(), BackendError> {
        if meshes.len() > MAX_MESHES {
            return Err(BackendError::TooManyMeshes { requested: meshes.len(), max: MAX_MESHES });
        }
        self.ctx.wait_idle();
        let mut verts: Vec<MeshVertex> = Vec::new();
        let mut idx: Vec<u32> = Vec::new();
        let mut table = Vec::with_capacity(meshes.len());
        for m in meshes {
            table.push(MeshInfoGpu {
                index_count: m.indices.len() as u32,
                first_index: idx.len() as u32,
                vertex_offset: verts.len() as i32,
                _pad: 0,
                sphere: m.bounding_sphere(),
            });
            verts.extend_from_slice(&m.vertices);
            idx.extend_from_slice(&m.indices);
        }
        let vbytes: &[u8] = bytemuck::cast_slice(&verts);
        let ibytes: &[u8] = bytemuck::cast_slice(&idx);
        let ctx = &mut self.ctx;
        let vb = GpuBuffer::new(
            ctx,
            "vertices",
            vbytes.len().max(4) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;
        let ib = GpuBuffer::new(
            ctx,
            "indices",
            ibytes.len().max(4) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;
        let mut staging = GpuBuffer::new(
            ctx,
            "staging",
            (vbytes.len() + ibytes.len()).max(4) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;
        staging.mapped_mut()[..vbytes.len()].copy_from_slice(vbytes);
        staging.mapped_mut()[vbytes.len()..vbytes.len() + ibytes.len()].copy_from_slice(ibytes);
        let d = ctx.device.clone();
        let (vlen, ilen) = (vbytes.len() as u64, ibytes.len() as u64);
        ctx.immediate_submit(|cmd| {
            // SAFETY: cmd is recording; buffers are live and large enough for the regions.
            unsafe {
                if vlen > 0 {
                    d.cmd_copy_buffer(
                        cmd,
                        staging.buffer,
                        vb.buffer,
                        &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: vlen }],
                    );
                }
                if ilen > 0 {
                    d.cmd_copy_buffer(
                        cmd,
                        staging.buffer,
                        ib.buffer,
                        &[vk::BufferCopy { src_offset: vlen, dst_offset: 0, size: ilen }],
                    );
                }
            }
            memory_barrier(
                &d,
                cmd,
                (vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_WRITE),
                (
                    vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT | vk::PipelineStageFlags2::INDEX_INPUT,
                    vk::AccessFlags2::VERTEX_ATTRIBUTE_READ | vk::AccessFlags2::INDEX_READ,
                ),
            );
        })?;
        staging.destroy(ctx);
        if let Some(mut old) = self.vertex_buf.replace(vb) {
            old.destroy(&mut self.ctx);
        }
        if let Some(mut old) = self.index_buf.replace(ib) {
            old.destroy(&mut self.ctx);
        }
        let tb: &[u8] = bytemuck::cast_slice(&table);
        self.mesh_buf.mapped_mut()[..tb.len()].copy_from_slice(tb);
        self.meshes = table;
        Ok(())
    }

    /// Run the GPU exclusive scan (the same kernels the frame uses) on `data` and return the result.
    /// Waits for the GPU to go idle; for tests and benchmarking.
    pub fn scan_u32(&mut self, data: &[u32]) -> Result<Vec<u32>, BackendError> {
        self.scan_u32_timed(data).map(|(v, _)| v)
    }

    /// `scan_u32` plus the GPU time of the scan dispatches alone (timestamp queries), in ms.
    pub fn scan_u32_timed(&mut self, data: &[u32]) -> Result<(Vec<u32>, Option<f64>), BackendError> {
        if data.is_empty() {
            return Ok((Vec::new(), None));
        }
        let n = u32::try_from(data.len()).map_err(|_| BackendError::BadConfig("scan input > u32::MAX elements"))?;
        self.ctx.wait_idle();
        let sda = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER;
        let bytes = data.len() as u64 * 4;
        let mut input = GpuBuffer::new(&mut self.ctx, "scan_in", bytes, sda, MemoryLocation::CpuToGpu)?;
        let mut output = GpuBuffer::new(&mut self.ctx, "scan_out", bytes, sda, MemoryLocation::GpuToCpu)?;
        let mut scratch = scan_levels(n)[1..]
            .iter()
            .map(|&len| GpuBuffer::new(&mut self.ctx, "scan_scratch", len as u64 * 4, sda, MemoryLocation::GpuOnly))
            .collect::<Result<Vec<_>, _>>()?;
        input.mapped_mut()[..bytes as usize].copy_from_slice(bytemuck::cast_slice(data));
        let addrs: Vec<u64> = scratch.iter().map(|b| b.address).collect();
        let (d, p, ia, oa) = (self.ctx.device.clone(), self.pipes, input.address, output.address);
        // SAFETY: device valid; create info fully initialised. Destroyed below after the GPU is idle.
        let qp = unsafe {
            d.create_query_pool(
                &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2),
                None,
            )?
        };
        let r = self.ctx.immediate_submit(|cmd| {
            use vk::PipelineStageFlags2 as S;
            // SAFETY: cmd recording; query pool live with 2 queries.
            unsafe {
                d.cmd_reset_query_pool(cmd, qp, 0, 2);
                d.cmd_write_timestamp2(cmd, S::TOP_OF_PIPE, qp, 0);
            }
            // host upload (CpuToGpu, written before submit) -> scan reads: covered by submit's host
            // write visibility guarantee.
            record_scan(&d, &p, cmd, ia, oa, n, &addrs);
            // SAFETY: cmd recording.
            unsafe { d.cmd_write_timestamp2(cmd, S::COMPUTE_SHADER, qp, 1) };
            memory_barrier(
                &d,
                cmd,
                (S::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE),
                (S::HOST, vk::AccessFlags2::HOST_READ),
            );
        });
        let gpu_ms = match (&r, self.ctx.timestamp_period_ns) {
            (Ok(()), Some(period)) => {
                let mut t = [0u64; 2];
                // SAFETY: the submit completed (immediate_submit waits), both queries were written.
                unsafe {
                    d.get_query_pool_results(qp, 0, &mut t, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?
                };
                Some(t[1].saturating_sub(t[0]) as f64 * period as f64 / 1e6)
            }
            _ => None,
        };
        // SAFETY: the GPU is done with the pool (immediate_submit waited).
        unsafe { d.destroy_query_pool(qp, None) };
        let out = r.map(|_| (bytemuck::cast_slice::<u8, u32>(&output.mapped()[..bytes as usize]).to_vec(), gpu_ms));
        input.destroy(&mut self.ctx);
        output.destroy(&mut self.ctx);
        for b in &mut scratch {
            b.destroy(&mut self.ctx);
        }
        out
    }

    fn wait_timeline(&self, value: u64) -> Result<(), BackendError> {
        let sems = [self.timeline];
        let vals = [value];
        // SAFETY: timeline semaphore is live; `value` was (or will be) signalled by a queued submit.
        unsafe {
            self.ctx
                .device
                .wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&sems).values(&vals), u64::MAX)?
        };
        Ok(())
    }

    /// Read stats + timestamps of the frame last submitted from slot `si`. Its timeline value must be reached.
    fn collect(&self, si: usize) -> Result<FrameStats, BackendError> {
        let s = &self.slots[si];
        let bytes = s.stats_buf.mapped();
        let draws = u32::from_ne_bytes(bytes[0..4].try_into().unwrap()).min(MAX_MESHES as u32);
        let mut visible = 0u32;
        for i in 0..draws as usize {
            let o = STATS_DRAWS_OFFSET as usize + i * DRAW_STRIDE as usize + 4; // instance_count
            visible += u32::from_ne_bytes(bytes[o..o + 4].try_into().unwrap());
        }
        let mut st = FrameStats {
            frame: s.submitted,
            instances: s.instances,
            visible,
            draws,
            in_flight_at_submit: s.in_flight_at_submit,
            completed_at_submit: s.completed_at_submit,
            ..Default::default()
        };
        if let Some(period) = self.ctx.timestamp_period_ns {
            let mut t = [0u64; QUERIES as usize];
            // SAFETY: the query pool is live and all queries were written by the completed submit.
            unsafe {
                self.ctx.device.get_query_pool_results(
                    s.queries,
                    0,
                    &mut t,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )?
            };
            let period = period as f64;
            let ms = |a: u64, b: u64| Some(b.saturating_sub(a) as f64 * period / 1e6);
            st.count_ms = ms(t[0], t[1]);
            st.scan_ms = ms(t[1], t[2]);
            st.scatter_ms = ms(t[2], t[3]);
            st.cull_ms = ms(t[0], t[3]);
            st.draw_ms = ms(t[3], t[4]);
            st.gpu_begin_ns = Some(t[0] as f64 * period);
            st.gpu_end_ns = Some(t[4] as f64 * period);
        }
        Ok(st)
    }

    /// Wait for every in-flight frame; return their stats oldest first (the last one is also
    /// stored as `last_stats`).
    pub fn drain(&mut self) -> Result<Vec<FrameStats>, BackendError> {
        if self.frame_number == 0 {
            return Err(BackendError::NothingRendered);
        }
        self.wait_timeline(self.frame_number)?;
        let first = self.frame_number.saturating_sub(self.slots.len() as u64 - 1).max(1);
        let mut out = Vec::new();
        for f in first..=self.frame_number {
            let si = self.slot_of(f);
            if self.slots[si].submitted == f {
                out.push(self.collect(si)?);
            }
        }
        self.last_stats = out.last().copied();
        Ok(out)
    }

    /// Wait for the last submitted frame and return its stats.
    pub fn finish(&mut self) -> Result<FrameStats, BackendError> {
        if self.frame_number == 0 {
            return Err(BackendError::NothingRendered);
        }
        self.wait_timeline(self.frame_number)?;
        let st = self.collect(self.slot_of(self.frame_number))?;
        self.last_stats = Some(st);
        Ok(st)
    }

    /// Render one frame. Returns the stats of the frame retired to make room (if any).
    pub fn render(&mut self, snap: &FrameSnapshot<'_>) -> Result<Option<FrameStats>, BackendError> {
        #[cfg(feature = "tracy")]
        let _z = tracy_client::span!("ash::render");
        let n = snap.instances.len();
        if n > self.max_instances {
            return Err(BackendError::TooManyInstances { requested: n, capacity: self.max_instances });
        }
        let fnum = self.frame_number + 1;
        let si = self.slot_of(fnum);

        // Frame pacing: before reusing slot `si`, wait until its previous frame (fnum - F) completed.
        // This host wait is the *only* synchronisation between frames sharing a slot.
        let mut retired = None;
        if self.slots[si].submitted > 0 {
            #[cfg(feature = "tracy")]
            let _w = tracy_client::span!("ash::wait_timeline");
            self.wait_timeline(self.slots[si].submitted)?;
            let st = self.collect(si)?;
            self.last_stats = Some(st);
            retired = Some(st);
        }

        #[cfg(feature = "windowed")]
        let image = self.acquire_image(si)?;
        #[cfg(not(feature = "windowed"))]
        let image: Option<u32> = None;

        // CPU side is now just frame constants + instance upload (counts/offsets are GPU work).
        let frame = FrameGpu {
            view_proj: snap.view_proj,
            planes: frustum_planes(Mat4::from_cols_array_2d(&snap.view_proj)).map(|p| p.to_array()),
            instance_count: n as u32,
            mesh_count: self.meshes.len() as u32,
            surface_range: self.surface_palette.gpu_range(),
        };
        {
            let slot = &mut self.slots[si];
            slot.frame_buf.mapped_mut()[..size_of::<FrameGpu>()].copy_from_slice(bytemuck::bytes_of(&frame));
            let ib: &[u8] = bytemuck::cast_slice(snap.instances);
            slot.inst_buf.mapped_mut()[..ib.len()].copy_from_slice(ib);
            slot.instances = n as u32;
        }

        self.record(si, n as u32, image)?;
        if let Some(t) = self.terrain.as_mut() {
            // Copies recorded into frame `fnum`; drawn slots are read by it (eviction waits on this value).
            t.pending.clear();
            t.mark_drawn(fnum);
        }

        // Submit: signal timeline = fnum (+ render_finished for present); wait acquire if presenting.
        let cbi = [vk::CommandBufferSubmitInfo::default().command_buffer(self.slots[si].cmd)];
        let mut waits: Vec<vk::SemaphoreSubmitInfo> = Vec::new();
        if let Some((sem, value)) = self.gate.take() {
            waits.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(sem)
                    .value(value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
        }
        #[allow(unused_mut)]
        let mut signals = vec![
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.timeline)
                .value(fnum)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        ];
        #[cfg(feature = "windowed")]
        if let (Some(i), Some(sc)) = (image, self.swapchain.as_ref()) {
            waits.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(sc.acquire[si])
                    .stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER),
            );
            signals.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(sc.render_finished[i as usize])
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
        }
        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&cbi)
            .wait_semaphore_infos(&waits)
            .signal_semaphore_infos(&signals);
        // SAFETY: cmd is fully recorded; all referenced resources belong to slot `si` (or are immutable
        // between `set_meshes` calls) and stay alive until timeline value `fnum` is reached.
        unsafe { self.ctx.device.queue_submit2(self.ctx.queue, &[submit], vk::Fence::null())? };
        let completed = self.completed_frame()?;
        let slot = &mut self.slots[si];
        slot.submitted = fnum;
        slot.completed_at_submit = completed;
        slot.in_flight_at_submit = fnum.saturating_sub(completed);
        self.frame_number = fnum;

        #[cfg(feature = "windowed")]
        if let (Some(i), Some(sc)) = (image, self.swapchain.as_ref())
            && !sc.present(self.ctx.queue, i)?
        {
            self.needs_recreate = true;
        }
        let _ = image;
        Ok(retired)
    }

    fn record(&self, si: usize, n: u32, image: Option<u32>) -> Result<(), BackendError> {
        let d = &self.ctx.device;
        let slot = &self.slots[si];
        let cmd = slot.cmd;
        let mesh_count = self.meshes.len() as u32;
        let push = PushGpu {
            frame: slot.frame_buf.address,
            instances: slot.inst_buf.address,
            meshes: self.mesh_buf.address,
            counts: slot.counts.address,
            offsets: slot.offsets.address,
            slots: slot.inst_slots.address,
            visible: slot.visible.address,
            draws: slot.draws.address,
            draw_count: slot.draw_count.address,
        };
        let scratch: Vec<u64> = slot.scan_scratch.iter().map(|b| b.address).collect();
        use vk::AccessFlags2 as A;
        use vk::PipelineStageFlags2 as S;
        let rw = (S::COMPUTE_SHADER, A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE);

        // SAFETY: (whole block) the slot's previous submit completed (host timeline wait in `render`),
        // so its pool, query pool, buffers and images are idle and can be reset/overwritten. All handles
        // are live for the lifetime of `self`. There is deliberately NO barrier against earlier frames:
        // everything below touches only slot `si`'s resources (plus the read-only mesh table / vertex /
        // index buffers, which only change under `wait_idle` in `set_meshes`). Within the frame each
        // producer/consumer pair has its own barrier (commented inline).
        unsafe {
            d.reset_command_pool(slot.pool, vk::CommandPoolResetFlags::empty())?;
            d.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            d.cmd_reset_query_pool(cmd, slot.queries, 0, QUERIES);

            // Terrain uploads staged since the last frame: staging ring -> vertex slots.
            // SAFETY (slot reuse): each target slot's previous reader frame has completed on the
            // timeline (`TerrainGpu::upload` only picks slots with last_used <= completed value),
            // and each staging range is tagged with this frame, so it is not reused before it retires.
            if let Some(t) = self.terrain.as_ref().filter(|t| !t.pending.is_empty()) {
                let slot_bytes = t.cfg.slot_bytes();
                let regions: Vec<vk::BufferCopy> = t
                    .pending
                    .iter()
                    .map(|&(src, s, size)| vk::BufferCopy { src_offset: src, dst_offset: s as u64 * slot_bytes, size })
                    .collect();
                d.cmd_copy_buffer(cmd, t.ring.buf.buffer, t.vertices.buffer, &regions);
                memory_barrier(
                    d,
                    cmd,
                    (S::COPY, A::TRANSFER_WRITE),
                    (S::VERTEX_ATTRIBUTE_INPUT, A::VERTEX_ATTRIBUTE_READ),
                );
            }

            // Zero per-mesh counts and the draw count.
            if mesh_count > 0 {
                d.cmd_fill_buffer(cmd, slot.counts.buffer, 0, mesh_count as u64 * 4, 0);
            }
            d.cmd_fill_buffer(cmd, slot.draw_count.buffer, 0, vk::WHOLE_SIZE, 0);
            // fill (transfer write) -> compute atomics / loads
            memory_barrier(d, cmd, (S::CLEAR, A::TRANSFER_WRITE), rw);
            d.cmd_write_timestamp2(cmd, S::TOP_OF_PIPE, slot.queries, 0);

            // 1. Cull + count: one thread per instance.
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipes.cull);
            d.cmd_push_constants(
                cmd,
                self.pipes.compute_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );
            d.cmd_dispatch(cmd, n.div_ceil(64), 1, 1);
            // counts/slots written -> scan reads counts, scatter reads slots
            memory_barrier(d, cmd, rw, rw);
            d.cmd_write_timestamp2(cmd, S::COMPUTE_SHADER, slot.queries, 1);

            // 2. Exclusive scan counts -> offsets (ends with compute->compute barrier).
            record_scan(d, &self.pipes, cmd, slot.counts.address, slot.offsets.address, mesh_count, &scratch);
            d.cmd_write_timestamp2(cmd, S::COMPUTE_SHADER, slot.queries, 2);

            // 3. Scatter (per instance) and indirect args (per mesh). Independent writes
            //    (visible vs. draws/draw_count), shared read-only inputs: no barrier between them.
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipes.scatter);
            d.cmd_push_constants(
                cmd,
                self.pipes.compute_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );
            d.cmd_dispatch(cmd, n.div_ceil(64), 1, 1);
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipes.compact);
            d.cmd_dispatch(cmd, mesh_count.div_ceil(64), 1, 1);
            d.cmd_write_timestamp2(cmd, S::COMPUTE_SHADER, slot.queries, 3);
            // compute writes -> indirect args, vertex-shader reads of visible list, stats copy
            memory_barrier(
                d,
                cmd,
                (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                (
                    S::DRAW_INDIRECT | S::VERTEX_SHADER | S::COPY,
                    A::INDIRECT_COMMAND_READ | A::SHADER_STORAGE_READ | A::TRANSFER_READ,
                ),
            );

            // Attachments of this slot: discard old contents (UNDEFINED). Their last use was frame
            // fnum - F, already waited on by the host, so the first scope is empty (NONE): no
            // dependency on the frame(s) currently in flight.
            image_barrier(
                d,
                cmd,
                slot.color.image,
                vk::ImageAspectFlags::COLOR,
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                (S::NONE, A::NONE),
                (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE),
            );
            image_barrier(
                d,
                cmd,
                slot.depth.image,
                vk::ImageAspectFlags::DEPTH,
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL),
                (S::NONE, A::NONE),
                (
                    S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                    A::DEPTH_STENCIL_ATTACHMENT_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ),
            );

            let color_att = [vk::RenderingAttachmentInfo::default()
                .image_view(slot.color.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: CLEAR } })];
            let depth_att = vk::RenderingAttachmentInfo::default()
                .image_view(slot.depth.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } });
            let area = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: self.extent };
            d.cmd_begin_rendering(
                cmd,
                &vk::RenderingInfo::default()
                    .render_area(area)
                    .layer_count(1)
                    .color_attachments(&color_att)
                    .depth_attachment(&depth_att),
            );
            if mesh_count > 0
                && let (Some(vb), Some(ib)) = (&self.vertex_buf, &self.index_buf)
            {
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipes.gfx);
                let vp = vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: self.extent.width as f32,
                    height: self.extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                d.cmd_set_viewport(cmd, 0, &[vp]);
                d.cmd_set_scissor(cmd, 0, &[area]);
                d.cmd_bind_vertex_buffers(cmd, 0, &[vb.buffer], &[0]);
                d.cmd_bind_index_buffer(cmd, ib.buffer, 0, vk::IndexType::UINT32);
                d.cmd_push_constants(
                    cmd,
                    self.pipes.gfx_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::bytes_of(&push),
                );
                d.cmd_draw_indexed_indirect_count(
                    cmd,
                    slot.draws.buffer,
                    0,
                    slot.draw_count.buffer,
                    0,
                    mesh_count,
                    DRAW_STRIDE,
                );
            }
            // World-space surfaces (terrain slots, static meshes) with the surface pipeline.
            let terrain = self.terrain.as_ref().filter(|t| !t.draw.is_empty());
            if terrain.is_some() || !self.statics.is_empty() {
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipes.surface);
                let vp = vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: self.extent.width as f32,
                    height: self.extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                d.cmd_set_viewport(cmd, 0, &[vp]);
                d.cmd_set_scissor(cmd, 0, &[area]);
                d.cmd_push_constants(
                    cmd,
                    self.pipes.gfx_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::bytes_of(&push),
                );
                if let Some(t) = terrain {
                    d.cmd_bind_vertex_buffers(cmd, 0, &[t.vertices.buffer], &[0]);
                    d.cmd_bind_index_buffer(cmd, t.indices.buffer, 0, vk::IndexType::UINT32);
                    for &s in &t.draw {
                        d.cmd_draw_indexed(cmd, t.index_count, 1, 0, (s * t.cfg.verts_per_slot) as i32, 0);
                    }
                }
                for m in &self.statics {
                    d.cmd_bind_vertex_buffers(cmd, 0, &[m.vertices.buffer], &[0]);
                    d.cmd_bind_index_buffer(cmd, m.indices.buffer, 0, vk::IndexType::UINT32);
                    d.cmd_draw_indexed(cmd, m.index_count, 1, 0, 0, 0);
                }
            }
            d.cmd_end_rendering(cmd);
            d.cmd_write_timestamp2(cmd, S::ALL_GRAPHICS, slot.queries, 4);

            // Color -> transfer source (for readback / blit to swapchain).
            image_barrier(
                d,
                cmd,
                slot.color.image,
                vk::ImageAspectFlags::COLOR,
                (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
                (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE),
                (S::ALL_TRANSFER, A::TRANSFER_READ),
            );

            // Stats readback: draw count + the used draw commands -> this slot's host-visible buffer.
            d.cmd_copy_buffer(
                cmd,
                slot.draw_count.buffer,
                slot.stats_buf.buffer,
                &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: 4 }],
            );
            if mesh_count > 0 {
                let size = mesh_count as u64 * DRAW_STRIDE as u64;
                d.cmd_copy_buffer(
                    cmd,
                    slot.draws.buffer,
                    slot.stats_buf.buffer,
                    &[vk::BufferCopy { src_offset: 0, dst_offset: STATS_DRAWS_OFFSET, size }],
                );
            }
            memory_barrier(d, cmd, (S::COPY, A::TRANSFER_WRITE), (S::HOST, A::HOST_READ));

            #[cfg(feature = "windowed")]
            if let (Some(i), Some(sc)) = (image, self.swapchain.as_ref()) {
                self.record_blit(cmd, slot.color.image, self.extent, sc.images[i as usize], sc.extent);
            }
            let _ = image;

            d.end_command_buffer(cmd)?;
        }
        Ok(())
    }

    #[cfg(feature = "windowed")]
    fn record_blit(
        &self,
        cmd: vk::CommandBuffer,
        src: vk::Image,
        src_extent: vk::Extent2D,
        dst: vk::Image,
        dst_extent: vk::Extent2D,
    ) {
        self.record_blit_internal(cmd, src, src_extent, dst, dst_extent, false);
    }

    fn record_blit_internal(
        &self,
        cmd: vk::CommandBuffer,
        src: vk::Image,
        src_extent: vk::Extent2D,
        dst: vk::Image,
        dst_extent: vk::Extent2D,
        probe_dst: bool,
    ) {
        use vk::AccessFlags2 as A;
        use vk::PipelineStageFlags2 as S;
        let d = &self.ctx.device;
        // The acquire semaphore is waited at ALL_TRANSFER; chain the layout transition to that wait
        // (first scope = the semaphore wait, not earlier frames' work).
        image_barrier(
            d,
            cmd,
            dst,
            vk::ImageAspectFlags::COLOR,
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
            (S::ALL_TRANSFER, A::NONE),
            (S::ALL_TRANSFER, A::TRANSFER_WRITE),
        );
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let off = |e: vk::Extent2D| {
            [vk::Offset3D { x: 0, y: 0, z: 0 }, vk::Offset3D { x: e.width as i32, y: e.height as i32, z: 1 }]
        };
        let region = [vk::ImageBlit {
            src_subresource: layers,
            src_offsets: off(src_extent),
            dst_subresource: layers,
            dst_offsets: off(dst_extent),
        }];
        // SAFETY: cmd recording; src is in TRANSFER_SRC_OPTIMAL (barrier in `record`), dst in TRANSFER_DST_OPTIMAL.
        unsafe {
            d.cmd_blit_image(
                cmd,
                src,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &region,
                vk::Filter::LINEAR,
            )
        };
        if probe_dst {
            image_barrier(
                d,
                cmd,
                dst,
                vk::ImageAspectFlags::COLOR,
                (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
                (S::ALL_TRANSFER, A::TRANSFER_WRITE),
                (S::NONE, A::NONE),
            );
        } else {
            image_barrier(
                d,
                cmd,
                dst,
                vk::ImageAspectFlags::COLOR,
                (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR),
                (S::ALL_TRANSFER, A::TRANSFER_WRITE),
                (S::NONE, A::NONE),
            );
        }
    }

    #[cfg(feature = "windowed")]
    fn acquire_image(&mut self, si: usize) -> Result<Option<u32>, BackendError> {
        if self.swapchain.is_none() {
            return Ok(None);
        }
        if self.needs_recreate {
            self.recreate_swapchain()?;
        }
        let r = self.swapchain.as_ref().map(|sc| sc.acquire(si)).transpose()?.flatten();
        if r.is_none() {
            self.needs_recreate = true; // out of date: render offscreen only this frame
        }
        Ok(r)
    }

    #[cfg(feature = "windowed")]
    fn recreate_swapchain(&mut self) -> Result<(), BackendError> {
        self.ctx.wait_idle();
        let Some(mut old) = self.swapchain.take() else { return Ok(()) };
        let new = crate::swapchain::Swapchain::new(&self.ctx, self.extent, old.swapchain, self.slots.len());
        old.destroy(&self.ctx);
        let new = new?;
        let ext = new.extent;
        self.swapchain = Some(new);
        self.needs_recreate = false;
        if ext != self.extent || self.slots.iter().any(|s| s.color.extent != ext) {
            self.recreate_targets(ext)?;
        }
        Ok(())
    }

    /// Recreate every slot's colour/depth target at `extent` (GPU idle first).
    fn recreate_targets(&mut self, extent: vk::Extent2D) -> Result<(), BackendError> {
        self.ctx.wait_idle();
        if let Some(mut rb) = self.readback.take() {
            rb.destroy(&mut self.ctx);
        }
        for i in 0..self.slots.len() {
            let (c, d) = targets(&mut self.ctx, extent)?;
            let mut old_c = std::mem::replace(&mut self.slots[i].color, c);
            let mut old_d = std::mem::replace(&mut self.slots[i].depth, d);
            old_c.destroy(&mut self.ctx);
            old_d.destroy(&mut self.ctx);
        }
        self.extent = extent;
        Ok(())
    }

    /// Blit an external R8G8B8A8 image already in TRANSFER_SRC_OPTIMAL to the swapchain and present.
    /// Reuses AshBackend frames-in-flight + acquire/present semaphores (feature `windowed`).
    #[cfg(feature = "windowed")]
    pub fn present_transfer_src(&mut self, src: vk::Image, src_extent: vk::Extent2D) -> Result<(), BackendError> {
        if self.swapchain.is_none() {
            return Err(BackendError::NoSurface);
        }
        if src_extent.width == 0 || src_extent.height == 0 {
            return Err(BackendError::BadConfig("present src extent"));
        }
        let fnum = self.frame_number + 1;
        let si = self.slot_of(fnum);
        if self.slots[si].submitted > 0 {
            self.wait_timeline(self.slots[si].submitted)?;
        }
        let image = self.acquire_image(si)?;
        let Some(i) = image else {
            return Ok(());
        };
        let d = self.ctx.device.clone();
        let cmd = self.slots[si].cmd;
        let pool = self.slots[si].pool;
        // SAFETY: slot wait completed; command pool exclusive to this slot.
        unsafe {
            d.reset_command_pool(pool, vk::CommandPoolResetFlags::empty())?;
            d.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        let dst_extent = self.swapchain.as_ref().unwrap().extent;
        let dst = self.swapchain.as_ref().unwrap().images[i as usize];
        self.record_blit(cmd, src, src_extent, dst, dst_extent);
        // SAFETY: recording complete.
        unsafe { d.end_command_buffer(cmd)? };

        let cbi = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
        let waits = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.swapchain.as_ref().unwrap().acquire[si])
            .stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)];
        let signals = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.timeline)
                .value(fnum)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.swapchain.as_ref().unwrap().render_finished[i as usize])
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        ];
        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&cbi)
            .wait_semaphore_infos(&waits)
            .signal_semaphore_infos(&signals);
        // SAFETY: cmd recorded; resources live until timeline fnum.
        unsafe { d.queue_submit2(self.ctx.queue, &[submit], vk::Fence::null())? };
        self.slots[si].submitted = fnum;
        self.frame_number = fnum;
        self.transfer_src_read_until = fnum;
        if !self.swapchain.as_ref().unwrap().present(self.ctx.queue, i)? {
            self.needs_recreate = true;
        }
        Ok(())
    }

    /// Wait until any queued TRANSFER_READ of an external present/test src image has completed.
    /// Timeline wait only — not `device_wait_idle`.
    pub fn wait_transfer_src_consumers(&self) -> Result<(), BackendError> {
        if self.transfer_src_read_until > 0 {
            self.wait_timeline(self.transfer_src_read_until)?;
        }
        Ok(())
    }

    /// Headless/windowed-shared: submit a TRANSFER_READ of `src` (copy into a tiny staging buffer) on the
    /// frame timeline, matching `present_transfer_src` consumer ordering without needing a swapchain.
    pub fn submit_transfer_src_read(&mut self, src: vk::Image, src_extent: vk::Extent2D) -> Result<(), BackendError> {
        if src_extent.width == 0 || src_extent.height == 0 {
            return Err(BackendError::BadConfig("transfer src extent"));
        }
        let fnum = self.frame_number + 1;
        let si = self.slot_of(fnum);
        if self.slots[si].submitted > 0 {
            self.wait_timeline(self.slots[si].submitted)?;
        }
        // Persistent blit destination matching present_transfer_src's TRANSFER_READ of src via BLIT.
        let need = src_extent;
        let recreate = self.transfer_probe.as_ref().is_none_or(|img| img.extent != need);
        if recreate {
            if let Some(mut old) = self.transfer_probe.take() {
                self.ctx.wait_idle();
                old.destroy(&mut self.ctx);
            }
            self.transfer_probe = Some(GpuImage::new(
                &mut self.ctx,
                "transfer_src_probe",
                need,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                vk::ImageAspectFlags::COLOR,
            )?);
        }
        let dst = self.transfer_probe.as_ref().unwrap().image;
        let d = self.ctx.device.clone();
        let cmd = self.slots[si].cmd;
        let pool = self.slots[si].pool;
        // SAFETY: slot wait completed; exclusive pool.
        unsafe {
            d.reset_command_pool(pool, vk::CommandPoolResetFlags::empty())?;
            d.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        // Same blit path as present_transfer_src (src stays TRANSFER_SRC_OPTIMAL).
        self.record_blit_internal(cmd, src, src_extent, dst, need, true);
        // SAFETY: recording complete.
        unsafe { d.end_command_buffer(cmd)? };
        let cbi = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
        let signals = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.timeline)
            .value(fnum)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let submit = vk::SubmitInfo2::default().command_buffer_infos(&cbi).signal_semaphore_infos(&signals);
        // SAFETY: cmd recorded; src live until timeline fnum.
        unsafe { d.queue_submit2(self.ctx.queue, &[submit], vk::Fence::null())? };
        self.slots[si].submitted = fnum;
        self.frame_number = fnum;
        self.transfer_src_read_until = fnum;
        Ok(())
    }

    /// Copy the most recent frame's colour target (RGBA8, row-major, tightly packed) to the host.
    pub fn read_color_rgba(&mut self) -> Result<(u32, u32, Vec<u8>), BackendError> {
        if self.frame_number == 0 {
            return Err(BackendError::NothingRendered);
        }
        self.ctx.wait_idle();
        let (w, h) = (self.extent.width, self.extent.height);
        let size = w as u64 * h as u64 * 4;
        if self.readback.as_ref().is_none_or(|b| b.size < size) {
            if let Some(mut old) = self.readback.take() {
                old.destroy(&mut self.ctx);
            }
            self.readback = Some(GpuBuffer::new(
                &mut self.ctx,
                "readback",
                size,
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
            )?);
        }
        let rb = self.readback.as_ref().unwrap().buffer;
        let img = self.slots[self.slot_of(self.frame_number)].color.image;
        let d = self.ctx.device.clone();
        self.ctx.immediate_submit(|cmd| {
            use vk::AccessFlags2 as A;
            use vk::PipelineStageFlags2 as S;
            // The last frame left the image in TRANSFER_SRC_OPTIMAL; order this copy after it.
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
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D { width: w, height: h, depth: 1 },
            }];
            // SAFETY: cmd recording; image in TRANSFER_SRC_OPTIMAL; buffer large enough (w*h*4 bytes).
            unsafe { d.cmd_copy_image_to_buffer(cmd, img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, rb, &region) };
            memory_barrier(&d, cmd, (S::COPY, A::TRANSFER_WRITE), (S::HOST, A::HOST_READ));
        })?;
        let px = self.readback.as_ref().unwrap().mapped()[..size as usize].to_vec();
        Ok((w, h, px))
    }
}

impl RenderBackend for AshBackend {
    type Error = BackendError;

    fn resize(&mut self, width: u32, height: u32) -> Result<(), BackendError> {
        let ext = vk::Extent2D { width: width.max(1), height: height.max(1) };
        if ext == self.extent {
            return Ok(());
        }
        #[cfg(feature = "windowed")]
        if self.swapchain.is_some() {
            self.extent = ext;
            return self.recreate_swapchain().and_then(|_| {
                if self.slots.iter().any(|s| s.color.extent != self.extent) {
                    self.recreate_targets(self.extent)
                } else {
                    Ok(())
                }
            });
        }
        self.recreate_targets(ext)
    }

    fn submit(&mut self, frame: &FrameSnapshot<'_>) -> Result<(), BackendError> {
        self.render(frame).map(|_| ())
    }
}

impl Drop for AshBackend {
    fn drop(&mut self) {
        self.ctx.wait_idle();
        // SAFETY: device idle, so none of these objects are in use; each is destroyed exactly once and
        // before `ctx` (which drops after this fn returns) destroys the device.
        unsafe {
            let d = &self.ctx.device;
            let p = self.pipes;
            for pl in [p.cull, p.scan_blocks_shared, p.scan_add, p.scatter, p.compact, p.gfx, p.surface]
                .into_iter()
                .chain(p.scan_blocks_wave)
            {
                d.destroy_pipeline(pl, None);
            }
            for l in [p.compute_layout, p.scan_layout, p.gfx_layout] {
                d.destroy_pipeline_layout(l, None);
            }
            d.destroy_semaphore(self.timeline, None);
            for s in &self.slots {
                d.destroy_command_pool(s.pool, None);
                d.destroy_query_pool(s.queries, None);
            }
        }
        #[cfg(feature = "windowed")]
        if let Some(mut sc) = self.swapchain.take() {
            sc.destroy(&self.ctx);
        }
        if let Some(mut probe) = self.transfer_probe.take() {
            probe.destroy(&mut self.ctx);
        }
        let mut slots = std::mem::take(&mut self.slots);
        for s in &mut slots {
            for b in s.buffers_mut() {
                b.destroy(&mut self.ctx);
            }
            s.color.destroy(&mut self.ctx);
            s.depth.destroy(&mut self.ctx);
        }
        self.mesh_buf.destroy(&mut self.ctx);
        if let Some(mut t) = self.terrain.take() {
            t.destroy(&mut self.ctx);
        }
        if let Some(c) = self.qshadow.take() {
            c.destroy(&mut self.ctx);
        }
        if let Some(mut b) = self.qground.take() {
            b.destroy(&mut self.ctx);
        }
        for mut m in std::mem::take(&mut self.statics) {
            m.vertices.destroy(&mut self.ctx);
            m.indices.destroy(&mut self.ctx);
        }
        for b in [self.vertex_buf.take(), self.index_buf.take(), self.readback.take()].into_iter().flatten() {
            let mut b = b;
            b.destroy(&mut self.ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_level_plan() {
        assert_eq!(scan_levels(0), [0]);
        assert_eq!(scan_levels(1024), [1024]);
        assert_eq!(scan_levels(1025), [1025, 2]);
        assert_eq!(scan_levels(1 << 20), [1 << 20, 1024]);
        assert_eq!(scan_levels((1 << 20) + 1), [(1 << 20) + 1, 1025, 2]);
    }

    #[test]
    fn subgroup_scan_plan() {
        use crate::context::SubgroupInfo;
        let ops = vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC;
        let base = SubgroupInfo {
            size: 32,
            supported_stages: vk::ShaderStageFlags::COMPUTE,
            supported_operations: ops,
            min_size: 8,
            max_size: 64,
            max_compute_workgroup_subgroups: 32,
            required_size_stages: vk::ShaderStageFlags::COMPUTE,
            size_control: true,
            compute_full_subgroups: true,
        };
        assert_eq!(plan_subgroup_scan(&base), Ok(Some(64)));
        // 256 threads at size 8 needs 32 subgroups per workgroup: allowed at exactly 32.
        assert_eq!(plan_subgroup_scan(&SubgroupInfo { max_size: 8, ..base }), Ok(Some(8)));
        assert_eq!(
            plan_subgroup_scan(&SubgroupInfo { max_size: 8, max_compute_workgroup_subgroups: 16, ..base }),
            Ok(None)
        );
        // no size control: varying sizes, size-agnostic kernel
        assert_eq!(plan_subgroup_scan(&SubgroupInfo { size_control: false, ..base }), Ok(None));
        assert!(plan_subgroup_scan(&SubgroupInfo { compute_full_subgroups: false, ..base }).is_err());
        assert!(
            plan_subgroup_scan(&SubgroupInfo { supported_operations: vk::SubgroupFeatureFlags::BASIC, ..base })
                .is_err()
        );
        assert!(
            plan_subgroup_scan(&SubgroupInfo { supported_stages: vk::ShaderStageFlags::FRAGMENT, ..base }).is_err()
        );
    }

    #[test]
    fn cpu_scan_wraps() {
        assert_eq!(cpu_exclusive_scan(&[3, 1, 4]), [0, 3, 4]);
        assert_eq!(cpu_exclusive_scan(&[u32::MAX, 2, 0]), [0, u32::MAX, 1]);
    }
}
