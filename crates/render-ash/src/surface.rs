//! World-space surface meshes drawn after the instanced pass, with the `surface.slang` pipeline:
//!
//! * **Terrain residency** (Prompt C; LOD'd DEM chunks since 3 Nov 2026): `slots` fixed-size
//!   regions in one device-local vertex buffer, all sharing one index buffer, keyed by
//!   [`TerrainSlotKey`] `(lod, x, y)` (noise chunks use `[i32; 2]` = LOD 0). `verts_per_slot` is
//!   the chunk's fixed vertex count (`65²` noise, `terrain::dem::DEM_CHUNK_VERTS` with skirts). Chunks arrive through a fixed **staging ring**
//!   (host-visible, default 32 MiB) and are copied into their slot at the start of the next frame's
//!   command buffer. LRU eviction never hands out a slot whose last reader frame has not retired.
//! * **Static meshes** (e.g. voxel output): uploaded once with `wait_idle`, drawn every frame.
//!
//! Timeline invariants (frame `f` signals timeline value `f`):
//! * a staging range tagged `f` is reused only after `completed >= f` (its copy ran in frame `f`);
//! * a vertex slot with `last_used = f` (drawn or written by frame `f`) is rewritten only after
//!   `completed >= f`.
use std::collections::{HashMap, VecDeque};

use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::BackendError;
use crate::context::{GpuBuffer, VkContext};

/// pos: [f32; 3] + normal: [f32; 3] (== terrain::TerrainVertex == mesh::MeshVertex).
pub const SURFACE_VERTEX_BYTES: usize = 24;

/// Residency key of one terrain chunk: LOD level plus chunk coordinates at that level.
///
/// Noise chunks (`[i32; 2]`) convert to LOD 0; DEM chunks (`terrain::dem::DemChunkId`) map
/// field-for-field (`y` counts northward there). Keys of different LODs never alias, so a parent
/// and its children can be resident at once (needed while a quadtree refines or coarsens).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TerrainSlotKey {
    pub lod: u8,
    pub x: i32,
    pub y: i32,
}

impl TerrainSlotKey {
    pub const fn new(lod: u8, x: i32, y: i32) -> Self {
        Self { lod, x, y }
    }
}

impl From<[i32; 2]> for TerrainSlotKey {
    fn from(c: [i32; 2]) -> Self {
        Self { lod: 0, x: c[0], y: c[1] }
    }
}

/// A noise-chunk coordinate equals the LOD 0 key with the same `x, y`.
impl PartialEq<[i32; 2]> for TerrainSlotKey {
    fn eq(&self, c: &[i32; 2]) -> bool {
        *self == Self::from(*c)
    }
}

impl From<(u8, i32, i32)> for TerrainSlotKey {
    fn from((lod, x, y): (u8, i32, i32)) -> Self {
        Self { lod, x, y }
    }
}

/// Surface-pass colouring of terrain/static surfaces (fragment palette in `surface.slang`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum SurfacePalette {
    /// Tuned for the ±40 m noise terrain: sand below 0 m, grass, snow from 18–28 m, rock on slopes.
    #[default]
    Noise,
    /// Real-world heights in metres (DEM): water at/below `0 m` (sea level, orthometric heights),
    /// then a hypsometric ramp normalised over `[lo_m, hi_m]` (coast → dry plains → pasture →
    /// scrub/forest → bare high ground) with rock on steep slopes. No snow: absolute heights such
    /// as Adelaide's −5…604 m stay green/brown instead of turning white above 28 m.
    HeightRange { lo_m: f32, hi_m: f32 },
}

impl SurfacePalette {
    /// `HeightRange` spanning a grid's height range (the low end clamped to sea level).
    pub fn for_heights(min_m: f32, max_m: f32) -> Self {
        let lo = min_m.max(0.0);
        Self::HeightRange { lo_m: lo, hi_m: max_m.max(lo + 1.0) }
    }

    /// `(lo, hi)` as sent to the shader in `Frame`; `hi <= lo` selects the noise palette.
    pub(crate) fn gpu_range(self) -> [f32; 2] {
        match self {
            Self::Noise => [0.0, 0.0],
            Self::HeightRange { lo_m, hi_m } if hi_m > lo_m => [lo_m, hi_m],
            // degenerate range: keep the DEM palette with a 1 m span instead of falling back to snow
            Self::HeightRange { lo_m, .. } => [lo_m, lo_m + 1.0],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TerrainGpuConfig {
    /// Resident chunk slots.
    pub slots: u32,
    /// Vertices per chunk: 65 * 65 for noise chunks, `terrain::dem::DEM_CHUNK_VERTS` (65² + 4·65
    /// skirt vertices) for DEM chunks. Must match the shared index buffer.
    pub verts_per_slot: u32,
    /// Staging ring size in bytes (fixed; never grows).
    pub staging_bytes: u64,
}

impl Default for TerrainGpuConfig {
    fn default() -> Self {
        Self { slots: 256, verts_per_slot: 65 * 65, staging_bytes: 32 << 20 }
    }
}

impl TerrainGpuConfig {
    /// Config for DEM chunks with `verts_per_slot` vertices each (pass `DEM_CHUNK_VERTS`).
    pub fn dem(slots: u32, verts_per_slot: u32) -> Self {
        Self { slots, verts_per_slot, ..Default::default() }
    }

    pub fn slot_bytes(&self) -> u64 {
        self.verts_per_slot as u64 * SURFACE_VERTEX_BYTES as u64
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UploadStats {
    pub uploads: u64,
    pub evictions: u64,
    pub bytes: u64,
    /// Max bytes of staging live (uploaded but whose frame had not retired) at any upload.
    pub max_staging_live: u64,
    /// Max number of chunk uploads in flight (staged, not yet retired) at any upload.
    pub max_uploads_in_flight: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub enum UploadOutcome {
    AlreadyResident { slot: u32 },
    Staged { slot: u32, evicted: Option<TerrainSlotKey> },
}

/// Ring allocation bookkeeping (pure, unit-tested). Allocations are tagged with the frame whose
/// command buffer reads them and freed from the tail once that frame retired.
#[derive(Debug)]
pub(crate) struct RingAlloc {
    cap: u64,
    head: u64,
    /// (frame, start, end) in allocation order.
    live: VecDeque<(u64, u64, u64)>,
}

impl RingAlloc {
    pub(crate) fn new(cap: u64) -> Self {
        Self { cap, head: 0, live: VecDeque::new() }
    }

    fn retire(&mut self, completed: u64) {
        while self.live.front().is_some_and(|&(f, _, _)| f <= completed) {
            self.live.pop_front();
        }
        if self.live.is_empty() {
            self.head = 0;
        }
    }

    pub(crate) fn live_bytes(&self) -> u64 {
        self.live.iter().map(|&(_, s, e)| e - s).sum()
    }

    /// Allocate `size` bytes for frame `frame`; `None` if only unretired data is in the way.
    pub(crate) fn alloc(&mut self, size: u64, frame: u64, completed: u64) -> Option<u64> {
        self.retire(completed);
        if size > self.cap {
            return None;
        }
        let start = match self.live.front().map(|&(_, s, _)| s) {
            None => 0,
            // Not wrapped iff head > tail (head == tail with live data means wrapped and full).
            Some(tail) if self.head > tail => {
                if self.head + size <= self.cap {
                    self.head
                } else if size <= tail {
                    0 // wrap
                } else {
                    return None;
                }
            }
            Some(tail) => {
                if self.head + size <= tail {
                    self.head
                } else {
                    return None;
                }
            }
        };
        self.head = start + size;
        self.live.push_back((frame, start, start + size));
        Some(start)
    }
}

pub(crate) struct StagingRing {
    pub(crate) buf: GpuBuffer,
    pub(crate) alloc: RingAlloc,
}

pub(crate) struct TerrainGpu {
    pub(crate) cfg: TerrainGpuConfig,
    pub(crate) vertices: GpuBuffer,
    pub(crate) indices: GpuBuffer,
    pub(crate) index_count: u32,
    pub(crate) ring: StagingRing,
    slot_of: HashMap<TerrainSlotKey, u32>,
    coord_of: Vec<Option<TerrainSlotKey>>,
    /// Last frame that read (drew) or wrote (copied into) the slot.
    last_used: Vec<u64>,
    free: Vec<u32>,
    /// Copies to record into the next frame: (staging offset, slot, bytes).
    pub(crate) pending: Vec<(u64, u32, u64)>,
    pub(crate) draw: Vec<u32>,
    pub(crate) stats: UploadStats,
    in_flight: VecDeque<u64>,
}

impl TerrainGpu {
    pub(crate) fn new(ctx: &mut VkContext, cfg: TerrainGpuConfig, grid_indices: &[u32]) -> Result<Self, BackendError> {
        if cfg.verts_per_slot == 0 || cfg.slots == 0 {
            return Err(BackendError::BadConfig("terrain slots and verts_per_slot must be > 0"));
        }
        if let Some(&max) = grid_indices.iter().max()
            && max >= cfg.verts_per_slot
        {
            return Err(BackendError::BadConfig("terrain index buffer references vertices past verts_per_slot"));
        }
        if cfg.slot_bytes() > cfg.staging_bytes {
            return Err(BackendError::BadConfig("terrain staging ring smaller than one chunk"));
        }
        let vbytes = cfg.slots as u64 * cfg.slot_bytes();
        let vertices = GpuBuffer::new(
            ctx,
            "terrain_vertices",
            vbytes.max(4),
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;
        let indices = upload_static(
            ctx,
            "terrain_indices",
            bytemuck::cast_slice(grid_indices),
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;
        let ring = StagingRing {
            buf: GpuBuffer::new(
                ctx,
                "terrain_staging",
                cfg.staging_bytes,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
            )?,
            alloc: RingAlloc::new(cfg.staging_bytes),
        };
        Ok(Self {
            cfg,
            vertices,
            indices,
            index_count: grid_indices.len() as u32,
            ring,
            slot_of: HashMap::with_capacity(cfg.slots as usize),
            coord_of: vec![None; cfg.slots as usize],
            last_used: vec![0; cfg.slots as usize],
            free: (0..cfg.slots).rev().collect(),
            pending: Vec::new(),
            draw: Vec::new(),
            stats: UploadStats::default(),
            in_flight: VecDeque::new(),
        })
    }

    pub(crate) fn is_resident(&self, coord: TerrainSlotKey) -> bool {
        self.slot_of.contains_key(&coord)
    }

    /// Stage `bytes` (a whole chunk's vertices) for `coord`, to be copied by frame `next_frame`.
    pub(crate) fn upload(
        &mut self,
        coord: TerrainSlotKey,
        bytes: &[u8],
        next_frame: u64,
        completed: u64,
    ) -> Result<UploadOutcome, BackendError> {
        let slot_bytes = self.cfg.slot_bytes();
        if bytes.len() as u64 != slot_bytes {
            return Err(BackendError::BadConfig("terrain chunk vertex data must fill exactly one slot"));
        }
        if let Some(&slot) = self.slot_of.get(&coord) {
            return Ok(UploadOutcome::AlreadyResident { slot });
        }
        // Pick a slot: free list first, else the least-recently-used slot whose last reader retired
        // and that is not drawn in the upcoming frame.
        let mut evicted = None;
        let slot = match self.free.pop() {
            Some(s) => s,
            None => {
                let victim = (0..self.cfg.slots)
                    .filter(|&s| self.last_used[s as usize] <= completed && !self.draw.contains(&s))
                    .min_by_key(|&s| self.last_used[s as usize])
                    .ok_or(BackendError::Unsupported("no terrain slot can be evicted yet (all in flight)"))?;
                // Debug-asserted invariant: the victim's last reader frame has retired.
                debug_assert!(self.last_used[victim as usize] <= completed, "evicting a slot still in flight");
                if let Some(old) = self.coord_of[victim as usize].take() {
                    self.slot_of.remove(&old);
                    evicted = Some(old);
                }
                self.stats.evictions += 1;
                victim
            }
        };
        let Some(off) = self.ring.alloc.alloc(slot_bytes, next_frame, completed) else {
            self.free.push(slot);
            if let Some(old) = evicted {
                // Undo: the old data is still intact in the slot (nothing was copied).
                self.slot_of.insert(old, slot);
                self.coord_of[slot as usize] = Some(old);
                self.stats.evictions -= 1;
                self.free.pop();
            }
            return Err(BackendError::Unsupported("terrain staging ring full; render a frame and retry"));
        };
        self.ring.buf.mapped_mut()[off as usize..(off + slot_bytes) as usize].copy_from_slice(bytes);
        self.pending.push((off, slot, slot_bytes));
        self.slot_of.insert(coord, slot);
        self.coord_of[slot as usize] = Some(coord);
        self.last_used[slot as usize] = next_frame;
        while self.in_flight.front().is_some_and(|&f| f <= completed) {
            self.in_flight.pop_front();
        }
        self.in_flight.push_back(next_frame);
        self.stats.uploads += 1;
        self.stats.bytes += slot_bytes;
        self.stats.max_staging_live = self.stats.max_staging_live.max(self.ring.alloc.live_bytes());
        self.stats.max_uploads_in_flight = self.stats.max_uploads_in_flight.max(self.in_flight.len());
        Ok(UploadOutcome::Staged { slot, evicted })
    }

    /// Set the chunks drawn from the next frame on (non-resident coords are skipped).
    pub(crate) fn set_draw(&mut self, coords: impl IntoIterator<Item = TerrainSlotKey>) -> usize {
        self.draw.clear();
        self.draw.extend(coords.into_iter().filter_map(|c| self.slot_of.get(&c).copied()));
        self.draw.len()
    }

    /// Called while recording frame `frame`: marks drawn slots used by it.
    pub(crate) fn mark_drawn(&mut self, frame: u64) {
        for &s in &self.draw {
            self.last_used[s as usize] = frame;
        }
    }

    pub(crate) fn destroy(&mut self, ctx: &mut VkContext) {
        self.vertices.destroy(ctx);
        self.indices.destroy(ctx);
        self.ring.buf.destroy(ctx);
    }
}

pub(crate) struct StaticMesh {
    pub(crate) vertices: GpuBuffer,
    pub(crate) indices: GpuBuffer,
    pub(crate) index_count: u32,
}

/// Device-local buffer filled through a temporary staging buffer (GPU idle on return).
pub(crate) fn upload_static(
    ctx: &mut VkContext,
    name: &str,
    bytes: &[u8],
    usage: vk::BufferUsageFlags,
) -> Result<GpuBuffer, BackendError> {
    let size = bytes.len().max(4) as u64;
    let dst = GpuBuffer::new(ctx, name, size, usage | vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuOnly)?;
    let mut staging =
        GpuBuffer::new(ctx, "static_staging", size, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu)?;
    staging.mapped_mut()[..bytes.len()].copy_from_slice(bytes);
    let d = ctx.device.clone();
    let (src, dbuf) = (staging.buffer, dst.buffer);
    let r = ctx.immediate_submit(|cmd| {
        // SAFETY: cmd recording; both buffers live and `size` bytes long.
        unsafe { d.cmd_copy_buffer(cmd, src, dbuf, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size }]) };
        crate::context::memory_barrier(
            &d,
            cmd,
            (vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_WRITE),
            (
                vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT | vk::PipelineStageFlags2::INDEX_INPUT,
                vk::AccessFlags2::VERTEX_ATTRIBUTE_READ | vk::AccessFlags2::INDEX_READ,
            ),
        );
    });
    staging.destroy(ctx);
    r.map(|_| dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_keys_keep_lods_apart() {
        let a: TerrainSlotKey = [3, -2].into();
        assert_eq!(a, TerrainSlotKey::new(0, 3, -2));
        assert!(a == [3, -2] && TerrainSlotKey::new(1, 3, -2) != [3, -2]);
        assert_ne!(TerrainSlotKey::new(1, 3, -2), a, "same x,y at another LOD is another chunk");
        let mut m = HashMap::new();
        for l in 0..4u8 {
            m.insert(TerrainSlotKey::from((l, 0, 0)), l);
        }
        assert_eq!(m.len(), 4);
    }

    #[test]
    fn palette_ranges() {
        assert_eq!(SurfacePalette::Noise.gpu_range(), [0.0, 0.0]);
        // Adelaide z12: -4.7..604 m -> sea-level clamped
        assert_eq!(SurfacePalette::for_heights(-4.7, 604.0).gpu_range(), [0.0, 604.0]);
        let flat = SurfacePalette::HeightRange { lo_m: 10.0, hi_m: 10.0 }.gpu_range();
        assert!(flat[1] > flat[0], "degenerate range must not select the snow palette");
    }

    #[test]
    fn ring_wraps_and_respects_unretired_frames() {
        let mut r = RingAlloc::new(100);
        assert_eq!(r.alloc(40, 1, 0), Some(0));
        assert_eq!(r.alloc(40, 2, 0), Some(40));
        assert_eq!(r.alloc(40, 3, 0), None, "frames 1,2 unretired: full");
        assert_eq!(r.alloc(40, 3, 1), Some(0), "frame 1 retired: wrap into its space");
        assert_eq!(r.alloc(30, 3, 1), None, "would overrun frame 2's data");
        assert_eq!(r.live_bytes(), 80);
        assert_eq!(r.alloc(30, 4, 3), Some(0), "everything retired: reset");
        assert_eq!(r.alloc(101, 5, 4), None, "larger than the ring");
    }

    #[test]
    fn ring_never_overlaps_live_ranges() {
        let mut r = RingAlloc::new(1000);
        let mut seed = 12345u64;
        let mut completed = 0;
        for frame in 1..2000u64 {
            for _ in 0..3 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let size = 1 + (seed >> 33) % 300;
                if r.alloc(size, frame, completed).is_some() {
                    let v: Vec<_> = r.live.iter().copied().collect();
                    for (i, a) in v.iter().enumerate() {
                        assert!(a.2 <= 1000);
                        for b in &v[i + 1..] {
                            assert!(a.2 <= b.1 || b.2 <= a.1, "overlap {a:?} {b:?}");
                        }
                    }
                }
            }
            completed = frame.saturating_sub(2); // 2 frames in flight
        }
    }
}
