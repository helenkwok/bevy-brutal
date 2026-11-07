//! ash/Vulkan 1.3 backend: a GPU-driven instanced mesh pass.
//!
//! Per frame (all GPU work, all buffers per frame slot): upload instances -> compute cull + per-mesh
//! count -> multi-level GPU exclusive scan of the counts -> scatter (compacted visible list) +
//! indirect-args generation -> `vkCmdDrawIndexedIndirectCount` with dynamic rendering into the slot's
//! offscreen target (blitted to the swapchain with feature `windowed`). Frames are paced with a
//! timeline semaphore and have no GPU dependencies on each other, so 2-3 can be in flight.
//!
//! Unsafe is allowed in this crate; every block carries a `// SAFETY:` comment.

pub mod context;
pub mod cull;
pub mod grass;
pub mod impostor;
pub mod mesh;
mod pass;
pub mod planet;
pub mod quality;
pub mod scene;
pub mod surface;
#[cfg(feature = "windowed")]
pub mod swapchain;
pub mod system;
pub mod tq_gpu;
pub mod validation;
pub mod voxel_gpu;

use ash::vk;

pub use context::{SubgroupInfo, VkContext, WindowHandles};
pub use pass::{
    AshBackend, BackendConfig, FrameStats, MAX_MESHES, SCAN_TILE, ScanMode, ScanPath, cpu_exclusive_scan,
    plan_subgroup_scan,
};
pub use surface::{SURFACE_VERTEX_BYTES, SurfacePalette, TerrainGpuConfig, TerrainSlotKey, UploadOutcome, UploadStats};

/// Valid range for `BackendConfig::frames_in_flight`.
pub const MIN_FRAMES_IN_FLIGHT: usize = 2;
pub const MAX_FRAMES_IN_FLIGHT: usize = 3;

#[derive(Debug)]
pub enum BackendError {
    Loading(ash::LoadingError),
    Vk(vk::Result),
    Alloc(gpu_allocator::AllocationError),
    Shader(std::io::Error),
    NoSuitableDevice,
    NoSurface,
    Unsupported(&'static str),
    TooManyInstances { requested: usize, capacity: usize },
    TooManyMeshes { requested: usize, max: usize },
    NothingRendered,
    BadConfig(&'static str),
}
impl From<vk::Result> for BackendError {
    fn from(e: vk::Result) -> Self {
        Self::Vk(e)
    }
}
impl From<gpu_allocator::AllocationError> for BackendError {
    fn from(e: gpu_allocator::AllocationError) -> Self {
        Self::Alloc(e)
    }
}

/// Bounded procedural stellar HDR splats.
pub mod universe;
