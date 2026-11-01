//! GPU decoder for TurboQuant-style brick blobs ([`asset::tq`], `shaders/tq.slang`): the verified
//! blob bytes are uploaded as is and decoded on the GPU, one workgroup per 512-sample block, with
//! no CPU decode. Bit-exact with [`asset::tq::TqView::decode`] (tested). A prototype for the
//! "Learning from turbovec and TurboQuant" study in `starter.md`; not used by the default renderer.

use ash::vk;
use asset::tq::{Rotation, TqView};
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use crate::context::{GpuBuffer, memory_barrier};
use crate::pass::shader_module;
use crate::{AshBackend, BackendError};

const TQ_DECODE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tq_decode.spv"));

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TqPushGpu {
    tables: u64,
    blobs: u64,
    out: u64,
    blob_words: u32,
    blocks: u32,
    samples: u32,
    bits: u32,
}

#[derive(Debug, Default)]
pub struct GpuTqDecode {
    /// Decoded samples, `samples` per blob, in blob order.
    pub samples: Vec<i32>,
    /// GPU time of the decode dispatch, ms (timestamp queries).
    pub gpu_ms: Option<f64>,
    /// Bytes uploaded (compressed blobs) and the raw `i32` size they replace.
    pub upload_bytes: u64,
    pub raw_bytes: u64,
}

impl AshBackend {
    /// Verify (CPU, [`TqView::parse`]) then decode `blobs` on the GPU. All blobs must share
    /// bits and shape. Corrupt blobs are rejected before anything is uploaded.
    pub fn decode_tq_gpu(&mut self, blobs: &[&[u8]], rot: &Rotation) -> Result<GpuTqDecode, BackendError> {
        if blobs.is_empty() {
            return Ok(GpuTqDecode::default());
        }
        let views: Vec<TqView<'_>> = blobs
            .iter()
            .map(|b| TqView::parse(b))
            .collect::<Result<_, _>>()
            .map_err(|_| BackendError::BadConfig("TQ blob failed verification"))?;
        let v0 = &views[0];
        if views.iter().any(|v| v.bits != v0.bits || v.samples != v0.samples || v.words.len() != v0.words.len()) {
            return Err(BackendError::BadConfig("TQ blobs differ in bits/shape"));
        }
        if blobs.len() > 65_535 || v0.blocks > 65_535 {
            return Err(BackendError::BadConfig("TQ batch too large"));
        }
        let tables = rot.gpu_tables(v0.bits);
        let mut words: Vec<u32> = Vec::with_capacity(blobs.len() * v0.words.len());
        for v in &views {
            words.extend_from_slice(v.words);
        }
        let out_bytes = blobs.len() as u64 * v0.samples as u64 * 4;
        self.ctx.wait_idle();
        let d = self.ctx.device.clone();
        let sda = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER;
        let upload = |ctx: &mut crate::VkContext, name: &str, data: &[u8]| -> Result<GpuBuffer, BackendError> {
            let mut b = GpuBuffer::new(ctx, name, data.len() as u64, sda, MemoryLocation::CpuToGpu)?;
            b.mapped_mut()[..data.len()].copy_from_slice(data);
            Ok(b)
        };
        let mut bufs = vec![upload(&mut self.ctx, "tq_tables", bytemuck::cast_slice(&tables))?];
        bufs.push(upload(&mut self.ctx, "tq_blobs", bytemuck::cast_slice(&words))?);
        bufs.push(GpuBuffer::new(&mut self.ctx, "tq_out", out_bytes, sda, MemoryLocation::GpuToCpu)?);
        let push = TqPushGpu {
            tables: bufs[0].address,
            blobs: bufs[1].address,
            out: bufs[2].address,
            blob_words: v0.words.len() as u32,
            blocks: v0.blocks,
            samples: v0.samples,
            bits: v0.bits,
        };
        let range = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: size_of::<TqPushGpu>() as u32,
        }];
        // SAFETY: device valid; create infos reference live data; module destroyed after creation.
        let (layout, pipe, qp) = unsafe {
            let layout =
                d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&range), None)?;
            let m = shader_module(&d, TQ_DECODE_SPV)?;
            let info = [vk::ComputePipelineCreateInfo::default()
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(m)
                        .name(c"main"),
                )
                .layout(layout)];
            let ps = d.create_compute_pipelines(vk::PipelineCache::null(), &info, None).map_err(|(_, e)| e);
            d.destroy_shader_module(m, None);
            let ps = ps?;
            let qp = d.create_query_pool(
                &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2),
                None,
            )?;
            (layout, ps[0], qp)
        };
        use vk::AccessFlags2 as A;
        use vk::PipelineStageFlags2 as S;
        let (gx, gy) = (v0.blocks, blobs.len() as u32);
        let r = self.ctx.immediate_submit(|cmd| {
            // SAFETY: cmd recording; pipeline/layout/query pool live; addresses point at live buffers sized above.
            unsafe {
                d.cmd_reset_query_pool(cmd, qp, 0, 2);
                d.cmd_write_timestamp2(cmd, S::TOP_OF_PIPE, qp, 0);
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
                d.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&push));
                d.cmd_dispatch(cmd, gx, gy, 1);
                d.cmd_write_timestamp2(cmd, S::COMPUTE_SHADER, qp, 1);
            }
            memory_barrier(&d, cmd, (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE), (S::HOST, A::HOST_READ));
        });
        let out = r.and_then(|()| {
            let gpu_ms = match self.ctx.timestamp_period_ns {
                Some(period) => {
                    let mut t = [0u64; 2];
                    // SAFETY: the submit completed and wrote both queries.
                    unsafe {
                        d.get_query_pool_results(
                            qp,
                            0,
                            &mut t,
                            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                        )?
                    };
                    Some(t[1].saturating_sub(t[0]) as f64 * period as f64 / 1e6)
                }
                None => None,
            };
            let samples: Vec<i32> = bytemuck::cast_slice(&bufs[2].mapped()[..out_bytes as usize]).to_vec();
            Ok(GpuTqDecode { samples, gpu_ms, upload_bytes: (words.len() * 4) as u64, raw_bytes: out_bytes })
        });
        self.ctx.wait_idle();
        // SAFETY: GPU idle; each object destroyed once.
        unsafe {
            d.destroy_pipeline(pipe, None);
            d.destroy_pipeline_layout(layout, None);
            d.destroy_query_pool(qp, None);
        }
        for mut b in bufs {
            b.destroy(&mut self.ctx);
        }
        out
    }
}
