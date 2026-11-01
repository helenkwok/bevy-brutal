//! GPU Transvoxel brick mesher (Prompt C2): `shaders/voxel.slang`, a pure-i32 port of
//! [`voxel::mesh_brick`]. Count pass -> the frame's shared multi-level exclusive scan (in place) ->
//! emit pass, all on the GPU. Output is a non-indexed triangle soup per brick in the CPU mesher's
//! cell/triangle order with brick-local sub-grid `i32` positions, so it compares bit-exactly against
//! the CPU (`tests/gpu.rs::gpu_transvoxel_matches_cpu_exactly`). Normals are f32 sample gradients.
//!
//! Robustness (hard rule H1-H8): densities are the CPU's quantised `i32` samples; `edge_t`, the
//! seam displacement Φ and edge orientation use the same integer formulas; there are no float
//! comparisons in the geometry path. The finer box is translated to brick-local sub units and
//! clamped to `[-2E, 3E]` (`E` = brick extent), which leaves Φ unchanged inside the brick because
//! Φ only depends on plane distances up to one coarse cell `s <= E/16`.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;
use voxel::tables::*;
use voxel::{Aabb, BrickSamples, SUB, VoxelConfig};

use crate::context::{GpuBuffer, memory_barrier};
use crate::pass::{record_scan, scan_levels, shader_module};
use crate::{AshBackend, BackendError};

const VOXEL_COUNT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/voxel_count.spv"));
const VOXEL_EMIT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/voxel_emit.spv"));
const DESC_INTS: usize = 17;
/// Largest brick edge the i32 bounds allow (64 cells x 2^6 step x 256 sub = 2^20).
pub const GPU_MAX_BRICK: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VoxPushGpu {
    tables: u64,
    bricks: u64,
    samples: u64,
    counts: u64,
    out: u64,
    n: u32,
    cpb: u32,
    pad: [u32; 2],
}

/// One brick to mesh: samples from [`voxel::sample_brick`] with the same mask and finer box the
/// CPU mesher would get from [`voxel::plan_brick`].
pub struct VoxelJob<'a> {
    pub samples: &'a BrickSamples,
    pub transitions: u8,
    pub finer: Option<Aabb>,
}

/// GPU mesher output for one brick: 3 vertices per triangle.
#[derive(Clone, Debug, Default)]
pub struct GpuBrickTris {
    /// Global sub-grid positions (brick origin added back on the host in i64).
    pub ipos: Vec<[i64; 3]>,
    pub normals: Vec<[f32; 3]>,
}

#[derive(Debug, Default)]
pub struct GpuVoxelBatch {
    pub bricks: Vec<GpuBrickTris>,
    pub triangles: usize,
    /// GPU time of count + scan (first submit) and emit (second submit), ms.
    pub gpu_ms: Option<f64>,
    /// Device bytes used (samples + descriptors + counts + output).
    pub device_bytes: u64,
}

/// The Transvoxel tables flattened to u32 at the offsets `voxel.slang` expects.
pub fn pack_tables() -> Vec<u32> {
    let mut t = Vec::with_capacity(12312);
    t.extend(REGULAR_CELL_CLASS.iter().map(|&v| v as u32));
    for (c, tri) in REGULAR_CELL_DATA.iter() {
        t.push(*c as u32);
        t.extend(tri.iter().map(|&v| v as u32));
    }
    debug_assert_eq!(t.len(), 512);
    t.extend(REGULAR_VERTEX_DATA.iter().flatten().map(|&v| v as u32));
    debug_assert_eq!(t.len(), 3584);
    t.extend(TRANSITION_CELL_CLASS.iter().map(|&v| v as u32));
    for (c, tri) in TRANSITION_CELL_DATA.iter() {
        t.push(*c as u32);
        t.extend(tri.iter().map(|&v| v as u32));
    }
    debug_assert_eq!(t.len(), 6168);
    t.extend(TRANSITION_VERTEX_DATA.iter().flatten().map(|&v| v as u32));
    t
}

/// Brick descriptor + sample layout (host side, shared by the run and the tests).
fn describe(cfg: &VoxelConfig, jobs: &[VoxelJob<'_>]) -> Result<(Vec<i32>, Vec<i32>), BackendError> {
    let n = cfg.grid.brick_size() as usize;
    if n > GPU_MAX_BRICK {
        return Err(BackendError::BadConfig("GPU voxel mesher: brick size > 64 exceeds the i32 bounds"));
    }
    let mut desc = Vec::with_capacity(jobs.len() * DESC_INTS);
    let mut samples: Vec<i32> = Vec::new();
    for j in jobs {
        let b = j.samples.coord;
        if b.lod > voxel::MAX_LOD || j.samples.n != n {
            return Err(BackendError::BadConfig("GPU voxel mesher: brick LOD/size out of range"));
        }
        let s = cfg.grid.step(b.lod);
        let o = cfg.grid.brick_origin(b).map(|v| v * SUB);
        let ext = n as i64 * s * SUB;
        let clamp = |v: i64, oi: i64| (v.saturating_mul(SUB) - oi).clamp(-2 * ext, 3 * ext) as i32;
        let (lo, hi) = match j.finer {
            Some(r) => ([0, 1, 2].map(|i| clamp(r.lo[i], o[i])), [0, 1, 2].map(|i| clamp(r.hi[i], o[i]))),
            None => ([0; 3], [0; 3]),
        };
        desc.extend([s as i32, (s * SUB).trailing_zeros() as i32, j.transitions as i32, j.finer.is_some() as i32]);
        desc.extend(lo);
        desc.extend(hi);
        desc.push(samples.len() as i32);
        samples.extend_from_slice(&j.samples.d);
        for f in 0..6 {
            if j.transitions & (1 << f) != 0 {
                desc.push(samples.len() as i32);
                samples.extend_from_slice(&j.samples.faces[f]);
            } else {
                desc.push(0);
            }
        }
    }
    if samples.len() > i32::MAX as usize {
        return Err(BackendError::BadConfig("GPU voxel mesher: batch too large"));
    }
    Ok((desc, samples))
}

impl AshBackend {
    /// Mesh `jobs` on the GPU (waits for the GPU; pipelines are built per call).
    pub fn mesh_voxels_gpu(&mut self, cfg: &VoxelConfig, jobs: &[VoxelJob<'_>]) -> Result<GpuVoxelBatch, BackendError> {
        if jobs.is_empty() {
            return Ok(GpuVoxelBatch::default());
        }
        let n = cfg.grid.brick_size() as usize;
        let (desc, samples) = describe(cfg, jobs)?;
        let tables = pack_tables();
        let cpb = (n * n * n + 6 * n * n) as u32;
        let cells = u32::try_from(jobs.len() as u64 * cpb as u64)
            .map_err(|_| BackendError::BadConfig("too many voxel cells"))?;
        if jobs.len() > 65_535 {
            return Err(BackendError::BadConfig("GPU voxel mesher: > 65535 bricks per batch"));
        }
        self.ctx.wait_idle();
        let d = self.ctx.device.clone();
        let sda = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER;
        let mut bufs: Vec<GpuBuffer> = Vec::new();
        let upload = |ctx: &mut crate::VkContext, name: &str, data: &[u8]| -> Result<GpuBuffer, BackendError> {
            let mut b = GpuBuffer::new(ctx, name, data.len() as u64, sda, MemoryLocation::CpuToGpu)?;
            b.mapped_mut()[..data.len()].copy_from_slice(data);
            Ok(b)
        };
        bufs.push(upload(&mut self.ctx, "vox_tables", bytemuck::cast_slice(&tables))?);
        bufs.push(upload(&mut self.ctx, "vox_bricks", bytemuck::cast_slice(&desc))?);
        bufs.push(upload(&mut self.ctx, "vox_samples", bytemuck::cast_slice(&samples))?);
        bufs.push(GpuBuffer::new(
            &mut self.ctx,
            "vox_counts",
            cells as u64 * 4,
            sda | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuOnly,
        )?);
        for &len in &scan_levels(cells)[1..] {
            bufs.push(GpuBuffer::new(&mut self.ctx, "vox_scan", len as u64 * 4, sda, MemoryLocation::GpuOnly)?);
        }
        // last count + last offset -> total triangles
        let tail = GpuBuffer::new(
            &mut self.ctx,
            "vox_tail",
            8,
            sda | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
        )?;
        let scratch: Vec<u64> = bufs[4..].iter().map(|b| b.address).collect();
        let counts_buf = (bufs[3].buffer, bufs[3].address);
        let mut push = VoxPushGpu {
            tables: bufs[0].address,
            bricks: bufs[1].address,
            samples: bufs[2].address,
            counts: counts_buf.1,
            out: 0,
            n: n as u32,
            cpb,
            pad: [0; 2],
        };
        let range = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: size_of::<VoxPushGpu>() as u32,
        }];
        // SAFETY: device valid; create infos reference live data; modules destroyed after creation.
        let (layout, count_p, emit_p, qp) = unsafe {
            let layout =
                d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&range), None)?;
            let mods = [shader_module(&d, VOXEL_COUNT_SPV)?, shader_module(&d, VOXEL_EMIT_SPV)?];
            let infos = mods.map(|m| {
                vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(m)
                            .name(c"main"),
                    )
                    .layout(layout)
            });
            let ps = d.create_compute_pipelines(vk::PipelineCache::null(), &infos, None).map_err(|(_, e)| e);
            for m in mods {
                d.destroy_shader_module(m, None);
            }
            let ps = ps?;
            let qp = d.create_query_pool(
                &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(4),
                None,
            )?;
            (layout, ps[0], ps[1], qp)
        };
        use vk::AccessFlags2 as A;
        use vk::PipelineStageFlags2 as S;
        let rw = (S::COMPUTE_SHADER, A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE);
        let groups = cpb.div_ceil(64);
        let pipes = self.pipes;
        let dispatch = |cmd: vk::CommandBuffer, p: vk::Pipeline, pc: &VoxPushGpu| {
            // SAFETY: cmd recording; pipeline/layout live; all addresses point at live buffers sized above.
            unsafe {
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p);
                d.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(pc));
                d.cmd_dispatch(cmd, groups, jobs.len() as u32, 1);
            }
        };
        let last = (cells as u64 - 1) * 4;
        let tail_buf = tail.buffer;
        let r1 = self.ctx.immediate_submit(|cmd| {
            // SAFETY: cmd recording; query pool has 4 queries.
            unsafe {
                d.cmd_reset_query_pool(cmd, qp, 0, 4);
                d.cmd_write_timestamp2(cmd, S::TOP_OF_PIPE, qp, 0);
            }
            dispatch(cmd, count_p, &push);
            memory_barrier(&d, cmd, rw, rw);
            // last count before the in-place scan overwrites it
            memory_barrier(&d, cmd, rw, (S::COPY, A::TRANSFER_READ));
            // SAFETY: both buffers live; regions in bounds (cells >= 1).
            unsafe {
                d.cmd_copy_buffer(
                    cmd,
                    counts_buf.0,
                    tail_buf,
                    &[vk::BufferCopy { src_offset: last, dst_offset: 0, size: 4 }],
                )
            };
            memory_barrier(&d, cmd, (S::COPY, A::TRANSFER_READ), rw);
            record_scan(&d, &pipes, cmd, counts_buf.1, counts_buf.1, cells, &scratch);
            memory_barrier(&d, cmd, rw, (S::COPY, A::TRANSFER_READ));
            // SAFETY: as above.
            unsafe {
                d.cmd_copy_buffer(
                    cmd,
                    counts_buf.0,
                    tail_buf,
                    &[vk::BufferCopy { src_offset: last, dst_offset: 4, size: 4 }],
                );
                d.cmd_write_timestamp2(cmd, S::ALL_COMMANDS, qp, 1);
            }
            memory_barrier(&d, cmd, (S::COPY, A::TRANSFER_WRITE), (S::HOST, A::HOST_READ));
        });
        let mut out_buf: Option<GpuBuffer> = None;
        let result = r1.and_then(|()| {
            let tw: &[u32] = bytemuck::cast_slice(&tail.mapped()[..8]);
            let total = tw[0] as u64 + tw[1] as u64;
            let ob = GpuBuffer::new(&mut self.ctx, "vox_out", (total * 3 * 24).max(16), sda, MemoryLocation::GpuToCpu)?;
            push.out = ob.address;
            let r2 = self.ctx.immediate_submit(|cmd| {
                // SAFETY: cmd recording.
                unsafe { d.cmd_write_timestamp2(cmd, S::TOP_OF_PIPE, qp, 2) };
                if total > 0 {
                    dispatch(cmd, emit_p, &push);
                }
                // SAFETY: cmd recording.
                unsafe { d.cmd_write_timestamp2(cmd, S::COMPUTE_SHADER, qp, 3) };
                memory_barrier(&d, cmd, rw, (S::HOST, A::HOST_READ));
            });
            out_buf = Some(ob);
            r2?;
            let gpu_ms = match self.ctx.timestamp_period_ns {
                Some(period) => {
                    let mut t = [0u64; 4];
                    // SAFETY: both submits completed and wrote all 4 queries.
                    unsafe {
                        d.get_query_pool_results(
                            qp,
                            0,
                            &mut t,
                            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                        )?
                    };
                    let p = period as f64 / 1e6;
                    Some((t[1].saturating_sub(t[0]) + t[3].saturating_sub(t[2])) as f64 * p)
                }
                None => None,
            };
            Ok((total, gpu_ms))
        });
        // Read back per brick: recompute per-brick triangle ranges from the CPU-side layout of cells.
        let out = result.and_then(|(total, gpu_ms)| {
            let ob = out_buf.as_ref().expect("output buffer");
            let words: &[u32] = bytemuck::cast_slice(&ob.mapped()[..(total * 3 * 24) as usize]);
            // brick b's triangles are contiguous, starting at the scanned offset of its first cell
            let starts = self.read_brick_starts(counts_buf.0, jobs.len(), cpb)?;
            let mut bricks = Vec::with_capacity(jobs.len());
            for (bi, j) in jobs.iter().enumerate() {
                let (a, b) =
                    (starts[bi] as usize, if bi + 1 < jobs.len() { starts[bi + 1] as usize } else { total as usize });
                let o = cfg.grid.brick_origin(j.samples.coord).map(|v| v * SUB);
                let mut t =
                    GpuBrickTris { ipos: Vec::with_capacity((b - a) * 3), normals: Vec::with_capacity((b - a) * 3) };
                for v in a * 3..b * 3 {
                    let w = &words[v * 6..v * 6 + 6];
                    t.ipos.push([0, 1, 2].map(|i| w[i] as i32 as i64 + o[i]));
                    t.normals.push([0, 1, 2].map(|i| f32::from_bits(w[3 + i])));
                }
                bricks.push(t);
            }
            let device_bytes = bufs.iter().map(|b| b.size).sum::<u64>() + ob.size;
            Ok(GpuVoxelBatch { bricks, triangles: total as usize, gpu_ms, device_bytes })
        });
        self.ctx.wait_idle();
        // SAFETY: GPU idle; each object destroyed once.
        unsafe {
            d.destroy_pipeline(count_p, None);
            d.destroy_pipeline(emit_p, None);
            d.destroy_pipeline_layout(layout, None);
            d.destroy_query_pool(qp, None);
        }
        for mut b in bufs.into_iter().chain(out_buf).chain(std::iter::once(tail)) {
            b.destroy(&mut self.ctx);
        }
        out
    }

    /// Scanned offset of each brick's first cell (copied from the GpuOnly counts buffer).
    fn read_brick_starts(&mut self, counts: vk::Buffer, bricks: usize, cpb: u32) -> Result<Vec<u32>, BackendError> {
        let sda = vk::BufferUsageFlags::TRANSFER_DST;
        let mut st = GpuBuffer::new(&mut self.ctx, "vox_starts", bricks as u64 * 4, sda, MemoryLocation::GpuToCpu)?;
        let regions: Vec<vk::BufferCopy> = (0..bricks)
            .map(|b| vk::BufferCopy { src_offset: b as u64 * cpb as u64 * 4, dst_offset: b as u64 * 4, size: 4 })
            .collect();
        let d = self.ctx.device.clone();
        let r = self.ctx.immediate_submit(|cmd| {
            // SAFETY: both buffers live, regions in bounds; prior submits completed (immediate_submit waits).
            unsafe { d.cmd_copy_buffer(cmd, counts, st.buffer, &regions) };
            memory_barrier(
                &d,
                cmd,
                (vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_WRITE),
                (vk::PipelineStageFlags2::HOST, vk::AccessFlags2::HOST_READ),
            );
        });
        let out = r.map(|()| bytemuck::cast_slice::<u8, u32>(&st.mapped()[..bricks * 4]).to_vec());
        st.destroy(&mut self.ctx);
        out
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn tables_pack_to_shader_offsets() {
        assert_eq!(super::pack_tables().len(), 12312);
    }
}
