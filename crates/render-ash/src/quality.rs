//! Quality-pass offscreen terrain renderer (`shaders/quality.slang`): shadow pass (sun cascades
//! or a headlamp spot map), sky, and lit terrain into an RGBA8 target, read back to the host.
//! BDA only: vertices are pulled through a pointer and the shadow maps are depth images copied
//! into a float buffer that the fragment shader filters manually, so there are no descriptors.
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use gpu_allocator::MemoryLocation;

use lowlevel_abi::InstanceGpu;

use crate::context::{GpuBuffer, GpuImage, image_barrier, memory_barrier};
use crate::mesh::MeshInfoGpu;
use crate::pass::{SCAN_TILE, record_scan, shader_module};
use crate::{AshBackend, BackendError};

const TERRAIN_VS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quality_terrain_vs.spv"));
const TERRAIN_FS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quality_terrain_fs.spv"));
const SHADOW_VS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quality_shadow_vs.spv"));
const SKY_VS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quality_sky_vs.spv"));
const SKY_FS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quality_sky_fs.spv"));
const INST_VS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quality_inst_vs.spv"));
const INST_SHADOW_VS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quality_inst_shadow_vs.spv"));
const INST_SHADOW_ALPHA_VS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quality_inst_shadow_alpha_vs.spv"));
const INST_SHADOW_ALPHA_FS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quality_inst_shadow_alpha_fs.spv"));
const INST_FS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quality_inst_fs.spv"));
const INST_DEPTH_FS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quality_inst_depth_fs.spv"));
const IMP_VS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quality_imp_vs.spv"));
const IMP_FS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quality_imp_fs.spv"));
const IMP_FAR_FS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quality_imp_far_fs.spv"));
const IMP_DEPTH_FS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quality_imp_depth_fs.spv"));
const IMP_EQ_FS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quality_imp_eq_fs.spv"));
const IMP_SHADOW_VS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quality_imp_shadow_vs.spv"));
const IMP_SHADOW_FAR_FS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quality_imp_shadow_far_fs.spv"));
const IMP_SHADOW_FS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quality_imp_shadow_fs.spv"));
const QCULL_CS: [&[u8]; 3] = [
    include_bytes!(concat!(env!("OUT_DIR"), "/qcull_cull.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/qcull_scatter.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/qcull_draws.spv")),
];
/// How far (m) terrain shadows are assumed to reach when culling casters per cascade (matches the
/// light frustum's `depth_back` extent toward the sun).
const SHADOW_SWEEP_M: f32 = 6000.0;
/// Views the instance cull handles: the camera plus up to 4 sun cascades (== MAX_VIEWS).
const MAX_VIEWS: usize = 5;
/// `QCullFrame.lod` "no coarser level".
pub const LOD_NONE: u32 = u32::MAX;
/// LOD chain terminator "draw nothing beyond the switch" (dithered out in the camera view).
pub const LOD_VANISH: u32 = u32::MAX - 1;
/// `lod[mesh].w` marker: the mesh is alpha-tested (needle / leaf cards), drawn with the
/// discard pipelines in every view (a NaN bit pattern, never a valid exact-depth distance).
pub const LOD_ALPHA: u32 = 0x7FC0_A1FA;
const ATMO_CS: [&[u8]; 5] = [
    include_bytes!(concat!(env!("OUT_DIR"), "/quality_atmo_transmittance.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/quality_atmo_multiscatter.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/quality_atmo_skyview.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/quality_atmo_aerial.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/quality_atmo_irradiance.spv")),
];
/// Dispatch sizes (workgroups) of the atmosphere kernels, in order: transmittance 256x64,
/// multi-scattering 32x32, sky-view 192x108, aerial-perspective 32x32 columns, irradiance.
const ATMO_GROUPS: [(u32, u32); 5] = [(32, 8), (4, 4), (24, 14), (4, 4), (1, 1)];
/// float4 count of the LUT buffer (== OFF_IRR + 2 irradiance + 9 SH coefficients, rounded up,
/// in quality.slang).
const ATMO_FLOAT4S: u64 = 70912 + 12;

/// Terrain vertex (== `QVertex` in quality.slang, 40 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct QVertex {
    /// Camera-relative metres.
    pub pos: [f32; 3],
    pub nrm: [f32; 3],
    /// ao, cavity, convexity, flags.
    pub attr: [f32; 4],
}

/// Frame parameters (== `QParams` in quality.slang).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct QParamsGpu {
    pub view_proj: [[f32; 4]; 4],
    pub inv_dir: [[f32; 4]; 4],
    pub cam_world: [f32; 4],
    pub sun_dir: [f32; 4],
    pub sun_color: [f32; 4],
    pub sky_zenith: [f32; 4],
    pub sky_horizon: [f32; 4],
    pub fog: [f32; 4],
    pub fog_color: [f32; 4],
    pub lamp_pos: [f32; 4],
    pub lamp_dir: [f32; 4],
    pub lamp_color: [f32; 4],
    pub lantern_pos: [f32; 4],
    pub lantern_color: [f32; 4],
    pub shadow_info: [f32; 4],
    pub splits: [f32; 4],
    pub texel: [f32; 4],
    pub cascade_vp: [[[f32; 4]; 4]; 4],
    /// x Mie density scale, y ground albedo, z aerial-perspective range (km), w enabled.
    pub atmo: [f32; 4],
    /// y sun angular radius (rad), z Rayleigh scale.
    pub atmo2: [f32; 4],
    /// Earth round 5: ground map device address ([`AshBackend::quality_set_ground_map`]), 0 = off.
    pub gmap: u64,
    pub gmap_pad: [u32; 2],
    /// Per level (outer, inner): texels per side, mip count, word offset of mip 0, 0.
    pub gmap_lv: [[u32; 4]; 2],
    /// x, z world centre (m), outer / inner half extent (m).
    pub gmap_f: [f32; 4],
}

impl Default for QParamsGpu {
    fn default() -> Self {
        Zeroable::zeroed()
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct QPushGpu {
    p: u64,
    v: u64,
    shadow: u64,
    cascade: u32,
    pad: u32,
    /// Instanced geometry: instance array and the GPU-compacted visible list.
    inst: u64,
    visible: u64,
    /// Atmosphere LUT buffer (0 when disabled).
    atmo: u64,
    /// Impostor atlas: header (`float4` centre, radius per variant) and texels.
    imp: u64,
    imp_tex: u64,
}
const _: () = assert!(size_of::<QPushGpu>() == 72);

/// One cull view (== `View` in qcull.slang, 112 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct QViewGpu {
    planes: [[f32; 4]; 6],
    /// 0 camera (distance LOD + dither crossfade), 1 hard distance switch, 2 coarsest level.
    mode: u32,
    lod_scale: f32,
    _p: [u32; 2],
}

/// == `QCullFrame` in qcull.slang (640 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct QCullFrameGpu {
    inst: u64,
    meshes: u64,
    lod: u64,
    counts: u64,
    offsets: u64,
    slots: u64,
    visible: u64,
    draws: u64,
    n: u32,
    mesh_count: u32,
    view_count: u32,
    _p: u32,
    views: [QViewGpu; MAX_VIEWS],
}
const _: () = assert!(size_of::<QViewGpu>() == 112 && size_of::<QCullFrameGpu>() == 640);
const PUSH_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::from_raw(
    vk::ShaderStageFlags::VERTEX.as_raw()
        | vk::ShaderStageFlags::FRAGMENT.as_raw()
        | vk::ShaderStageFlags::COMPUTE.as_raw(),
);

/// Reverse-Z infinite perspective, Vulkan clip conventions (NDC y down, z in (0, 1], near -> 1).
pub fn reverse_z_infinite(fov_y: f32, aspect: f32, near: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    Mat4::from_cols_array(&[f / aspect, 0.0, 0.0, 0.0, 0.0, -f, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, near, 0.0])
}

/// Orthographic projection, Vulkan clip conventions, z in [0, 1] over [near, far] (LESS).
pub fn vk_ortho(half_w: f32, half_h: f32, near: f32, far: f32) -> Mat4 {
    let d = far - near;
    Mat4::from_cols_array(&[
        1.0 / half_w,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.0 / half_h,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.0 / d,
        0.0,
        0.0,
        0.0,
        -near / d,
        1.0,
    ])
}

/// Light-space depth grid (m) the cascade centre snaps to along the sun direction (so a
/// world-anchored cascade's depth values do not drift with the camera either).
pub const CASCADE_DEPTH_SNAP: f64 = 16.0;

/// Stable sun cascade: an ortho light view enclosing the bounding sphere of the view-frustum
/// slice `[n, f]`, its centre snapped to whole texels of a grid anchored in *world* light space
/// (`origin` = camera world position; the matrix is camera-relative). Returns (view_proj,
/// texel m, snapped grid cell): two frames with the same cell, slice, sun and map size have the
/// same world-space light matrix, so a cached shadow map of static casters stays valid
/// ([`QualityFrame::shadow_cache_keys`]). The radius is closed-form (rotation-invariant).
#[allow(clippy::too_many_arguments)] // flat math helper: camera frustum slice + light + map size
pub fn sun_cascade(
    origin: glam::DVec3,
    cam_fwd: Vec3,
    fov_y: f32,
    aspect: f32,
    n: f32,
    f: f32,
    sun: Vec3,
    res: u32,
    depth_back: f32,
) -> (Mat4, f32, [i64; 3]) {
    let t = (fov_y * 0.5).tan() as f64;
    let (n64, f64_) = (n as f64, f as f64);
    // bounding sphere of the slice: centre on the axis at the corner-centroid, radius to the far corners
    let cd = 0.5 * (n64 + f64_);
    let rad = (((f64_ - cd) * (f64_ - cd)) + (t * f64_) * (t * f64_) * (1.0 + (aspect as f64) * (aspect as f64)))
        .sqrt()
        .max(((cd - n64) * (cd - n64) + (t * n64) * (t * n64) * (1.0 + (aspect as f64) * (aspect as f64))).sqrt());
    let c = cam_fwd.as_dvec3() * cd;
    let rad32 = rad as f32;
    let texel = 2.0 * rad32 / res as f32;
    let up = if sun.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    // light basis (rows of look_at(0, -sun, up)) in f64; world-space light coordinates of the centre
    let view0 = crate::cull::look_at(Vec3::ZERO, -sun, up);
    let row = |k: usize| glam::DVec3::new(view0.col(0)[k] as f64, view0.col(1)[k] as f64, view0.col(2)[k] as f64);
    let (rx, ry, rz) = (row(0), row(1), row(2));
    let w = origin + c;
    let tx = texel as f64;
    let cell = [
        (rx.dot(w) / tx).round() as i64,
        (ry.dot(w) / tx).round() as i64,
        (rz.dot(w) / CASCADE_DEPTH_SNAP).round() as i64,
    ];
    let cw = rx * (cell[0] as f64 * tx) + ry * (cell[1] as f64 * tx) + rz * (cell[2] as f64 * CASCADE_DEPTH_SNAP);
    let centre = (cw - origin).as_vec3();
    let eye = centre + sun * (rad32 + depth_back);
    let view = crate::cull::look_at(eye, centre, up);
    (vk_ortho(rad32, rad32, 0.0, 2.0 * rad32 + depth_back * 2.0) * view, texel, cell)
}

/// Clipmap-style sun cascade that survives camera rotation and short moves (round 10):
/// an orthographic square of half-size `R = f / (1 - grid / res)` around a centre snapped to a
/// coarse world light-space grid of `grid` texels (`g = grid * texel`, also used for the depth
/// snap). Every point within `f` of the camera stays inside the map, in every direction, while
/// the camera moves by less than `g / 2` per axis from the cell centre. So the snapped cell (the
/// cache key) is independent of the view direction and changes only after a move of about
/// `g = 2 R grid / res`. Returns (camera-relative view-projection, texel m, cell). At wide
/// aspect ratios this is no larger than [`sun_cascade`]'s slice sphere: at 16:9 and 60 deg
/// FOV the slice radius is ~1.2 f, here 1.14 f with `grid = res / 8`.
pub fn sun_cascade_clip(
    cam: glam::DVec3,
    f: f32,
    sun: Vec3,
    res: u32,
    depth_back: f32,
    grid: u32,
) -> (Mat4, f32, [i64; 3]) {
    let k = grid.min(res / 2) as f64 / res as f64;
    let rad = f as f64 / (1.0 - k);
    let texel = 2.0 * rad / res as f64;
    let g = texel * grid.max(1) as f64;
    let up = if sun.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let view0 = crate::cull::look_at(Vec3::ZERO, -sun, up);
    let row = |k: usize| glam::DVec3::new(view0.col(0)[k] as f64, view0.col(1)[k] as f64, view0.col(2)[k] as f64);
    let (rx, ry, rz) = (row(0), row(1), row(2));
    let cell = [(rx.dot(cam) / g).round() as i64, (ry.dot(cam) / g).round() as i64, (rz.dot(cam) / g).round() as i64];
    let cw = rx * (cell[0] as f64 * g) + ry * (cell[1] as f64 * g) + rz * (cell[2] as f64 * g);
    let centre = (cw - cam).as_vec3();
    let r32 = rad as f32;
    let eye = centre + sun * (r32 + depth_back);
    let view = crate::cull::look_at(eye, centre, up);
    (vk_ortho(r32, r32, 0.0, 2.0 * r32 + depth_back * 2.0) * view, texel as f32, cell)
}

/// A shadow view: `view_proj` (camera-relative) and the world size of one texel.
#[derive(Clone, Copy, Debug)]
pub struct ShadowView {
    pub view_proj: Mat4,
    pub texel: f32,
}

/// A contiguous index range of the terrain with its bounding sphere (camera-relative), so each
/// sun cascade draws only the terrain that can cast into it (see [`terrain_chunks`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainChunk {
    pub first_index: u32,
    pub index_count: u32,
    pub sphere: [f32; 4],
}

/// Reorder a triangle list into spatial chunks (XZ bins of `base` m, doubling every octave of
/// camera distance past `4 * base`, so far rings do not make thousands of tiny chunks) and
/// return the new index list with the chunk table. The triangle set is unchanged (only its
/// order: depth ties between coincident triangles could resolve differently).
pub fn terrain_chunks(vertices: &[QVertex], indices: &[u32], base: f32) -> (Vec<u32>, Vec<TerrainChunk>) {
    use std::collections::BTreeMap;
    let mut bins: BTreeMap<(i32, i32, i32), Vec<u32>> = BTreeMap::new();
    for (t, tri) in indices.as_chunks::<3>().0.iter().enumerate() {
        let c = tri.iter().map(|&i| Vec3::from_array(vertices[i as usize].pos)).sum::<Vec3>() / 3.0;
        let d = c.x.hypot(c.z);
        let lvl = (d / (4.0 * base)).max(1.0).log2().floor().clamp(0.0, 12.0) as i32;
        let cell = base * (1 << lvl) as f32;
        bins.entry((lvl, (c.x / cell).floor() as i32, (c.z / cell).floor() as i32)).or_default().push(t as u32);
    }
    let mut out = Vec::with_capacity(indices.len());
    let mut chunks = Vec::with_capacity(bins.len());
    for tris in bins.values() {
        let first = out.len() as u32;
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for &t in tris {
            for k in 0..3 {
                let i = indices[t as usize * 3 + k];
                out.push(i);
                let p = Vec3::from_array(vertices[i as usize].pos);
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        let c = (lo + hi) * 0.5;
        chunks.push(TerrainChunk {
            first_index: first,
            index_count: out.len() as u32 - first,
            sphere: [c.x, c.y, c.z, (hi - lo).length() * 0.5 * 1.001 + 0.01],
        });
    }
    (out, chunks)
}

fn sphere_in(planes: &[glam::Vec4; 6], p: Vec3, r: f32) -> bool {
    planes.iter().all(|pl| pl.truncate().dot(p) + pl.w >= -r)
}

/// Index ranges of `chunks` whose sphere is inside all `planes` (adjacent ones merged).
pub fn cull_chunks(chunks: &[TerrainChunk], planes: &[glam::Vec4; 6]) -> Vec<(u32, u32)> {
    cull_chunks_by(chunks, |p, r| sphere_in(planes, p, r))
}

/// Shadow casters of one cascade: chunks inside the light frustum `planes` whose shadow (the
/// sphere swept `len` m away from the sun, `sun` = unit direction toward it) reaches the
/// cascade's receiver slice `slice` (camera frustum between the split distances). Without
/// the sweep test the far cascades' bounding boxes take in the whole dense near field.
/// Conservative: the sweep is sampled every `len / 16` with the sphere inflated by that step.
pub fn cull_casters(
    chunks: &[TerrainChunk],
    planes: &[glam::Vec4; 6],
    slice: &[glam::Vec4; 6],
    sun: Vec3,
    len: f32,
) -> Vec<(u32, u32)> {
    let step = len / 16.0;
    cull_chunks_by(chunks, |p, r| {
        sphere_in(planes, p, r)
            && (0..=16).any(|k| sphere_in(slice, p - sun * (k as f32 * step), r + step * 0.5 + 1e-3))
    })
}

fn cull_chunks_by(chunks: &[TerrainChunk], keep: impl Fn(Vec3, f32) -> bool) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    for c in chunks {
        let (p, r) = (Vec3::new(c.sphere[0], c.sphere[1], c.sphere[2]), c.sphere[3]);
        if keep(p, r) {
            match out.last_mut() {
                Some(l) if l.0 + l.1 == c.first_index => l.1 += c.index_count,
                _ => out.push((c.first_index, c.index_count)),
            }
        }
    }
    out
}

#[derive(Clone, Copy)]
pub struct QualityFrame<'a> {
    pub width: u32,
    pub height: u32,
    pub vertices: &'a [QVertex],
    pub indices: &'a [u32],
    pub params: QParamsGpu,
    pub shadow_res: u32,
    pub shadows: &'a [ShadowView],
    /// Spot (perspective) shadow map instead of sun cascades.
    pub spot_shadow: bool,
    /// Optional GPU-instanced geometry (trees, boulders, grass tufts).
    pub instanced: Option<InstancedSet<'a>>,
    /// Compute the Hillaire atmosphere LUTs this frame (needs `params.atmo.w = 1`).
    pub atmosphere: bool,
    /// Terrain chunks ([`terrain_chunks`]) for per-cascade culling; empty = every shadow view
    /// draws the whole terrain.
    pub terrain_chunks: &'a [TerrainChunk],
    /// Per shadow view: the receiver slice's view-projection (camera frustum between the split
    /// distances, camera-relative) for [`cull_casters`]; empty = cull by the light frustum only.
    pub shadow_slices: &'a [Mat4],
    /// Foliage depth prepass: the instance meshes are first drawn depth-only (alpha cards and
    /// LOD dither discard exactly as in the shading pass); the terrain then draws as usual
    /// (early-Z rejects it behind the vegetation) and the instance meshes shade with depth
    /// `EQUAL` and no writes (one shaded fragment per pixel). Impostors and sky are drawn after,
    /// as before. Same image (same vertex shaders and discard code) except exact depth ties
    /// between coplanar instance triangles (last drawn instead of first drawn wins).
    pub depth_prepass: bool,
    /// Every camera-view impostor (also the exact-distance and crossfading ones) uses the card's
    /// depth (`impFarFS`, with the LOD dither): early depth testing, no per-pixel depth. With
    /// `depth_prepass` the impostors also join the prepass (`impDepthFS`: dither + coverage)
    /// and then shade with depth EQUAL (`impEqFS`), one shaded fragment per pixel.
    pub impostor_card_depth: bool,
    /// Per shadow view: a cache key (0 = always render). A cascade whose key equals the one it
    /// was last rendered with by this backend is not redrawn: its depth from the previous frame
    /// is reused (static shadow cascade caching). The caller must make the key cover everything
    /// the cascade's depth depends on (world-space light matrix: [`sun_cascade`]'s cell, slice,
    /// sun and map size; the caster geometry); views whose content depends on the camera (the
    /// distance-LOD near cascades) must pass 0.
    pub shadow_cache_keys: &'a [u64],
}

/// Shadow maps kept between quality frames (see [`QualityFrame::shadow_cache_keys`]).
pub(crate) struct QShadowCache {
    buf: GpuBuffer,
    res: u32,
    count: usize,
    keys: [u64; 4],
}

impl QShadowCache {
    pub(crate) fn destroy(mut self, ctx: &mut crate::context::VkContext) {
        self.buf.destroy(ctx);
    }
}

/// GPU-instanced meshes for the quality frame. Instances are culled *per view* (camera + every
/// sun cascade) on the GPU by `qcull.slang`, which also picks each instance's LOD from its camera
/// distance and the `lod` chain (count -> scan (`scan.slang`) -> scatter -> one fixed-slot
/// indirect draw per (view, mesh)). Meshes `impostor_first..` are octahedral impostor cards
/// (drawn with the impostor pipelines, see [`crate::impostor`]). Vertices are pulled
/// (`QVertex`, `attr.w` = material code >= 2); model matrices are camera-relative (built from f64
/// world positions on the host).
#[derive(Clone, Copy)]
pub struct InstancedSet<'a> {
    pub vertices: &'a [QVertex],
    pub indices: &'a [u32],
    pub meshes: &'a [MeshInfoGpu],
    pub instances: &'a [InstanceGpu],
    /// Finite-far camera view-projection for the cull planes (the reverse-Z infinite matrix has
    /// no far plane to normalise).
    pub cull_view_proj: Mat4,
    /// Per mesh: (next coarser mesh or [`LOD_NONE`], switch distance m as f32 bits, crossfade
    /// width m as f32 bits, exact-depth distance m as f32 bits (impostors: nearer camera-view
    /// draws write per-pixel depth, farther ones the card's), 0 = none). Empty = no LOD.
    pub lod: &'a [[u32; 4]],
    /// First impostor mesh (`meshes.len()` when there are none).
    pub impostor_first: u32,
    /// Impostor atlas ([`crate::impostor::atlas_bytes`]) and its variant count.
    pub atlas: Option<(&'a [u8], u32)>,
}

#[derive(Clone, Debug, Default)]
pub struct QualityTimings {
    pub shadow_ms: f64,
    pub main_ms: f64,
    pub gpu_ms: f64,
    /// Instance draws surviving the GPU cull in the camera view (crossfading instances count
    /// twice), read back after the frame.
    pub visible_instances: u32,
    /// Of those, impostor draws.
    pub impostor_instances: u32,
    /// Of those, draws inside a LOD crossfade band (dithered).
    pub fading_instances: u32,
    /// Instance draws per cull view (camera, cascades 0..).
    pub view_instances: [u32; MAX_VIEWS],
    /// Triangles drawn by instances per cull view.
    pub view_triangles: [u64; MAX_VIEWS],
    /// Terrain triangles drawn into each shadow view.
    pub shadow_terrain_triangles: Vec<u64>,
    /// Shadow cascades reused from the previous frame (not redrawn).
    pub cached_cascades: u32,
    /// Camera-view (view 0) per mesh: (instance draws, triangles), crossfading draws counted
    /// twice (GPU-independent workload metric).
    pub camera_mesh_draws: Vec<(u32, u64)>,
    /// GPU time per recorded stage (label, ms), in submission order; sums to `gpu_ms`.
    pub breakdown: Vec<(&'static str, f64)>,
}

/// Timestamp queries of one quality frame (start + labelled marks).
const MAX_MARKS: u32 = 32;

struct Pipes {
    layout: vk::PipelineLayout,
    terrain: vk::Pipeline,
    shadow: vk::Pipeline,
    sky: vk::Pipeline,
    inst: vk::Pipeline,
    inst_fade: vk::Pipeline,
    inst_shadow: vk::Pipeline,
    inst_shadow_alpha: vk::Pipeline,
    imp: vk::Pipeline,
    imp_far: vk::Pipeline,
    pre_imp: vk::Pipeline,
    imp_eq: vk::Pipeline,
    imp_shadow: vk::Pipeline,
    imp_shadow_far: vk::Pipeline,
    pre_inst: vk::Pipeline,
    pre_inst_alpha: vk::Pipeline,
    atmo: Vec<vk::Pipeline>,
    qcull: Vec<vk::Pipeline>,
}

const COLOR: vk::Format = vk::Format::R8G8B8A8_UNORM;
const DEPTH: vk::Format = vk::Format::D32_SFLOAT;

enum Kind {
    Terrain,
    Shadow,
    Sky,
    Inst,
    InstFade,
    InstShadowAlpha,
    InstShadow,
    Imp,
    ImpFar,
    ImpShadow,
    ImpShadowFar,
    PreInst,
    PreInstAlpha,
    /// card-depth impostors in the foliage prepass (dither + coverage, no colour)
    PreImp,
    /// card-depth impostors after the prepass (depth EQUAL, no writes, no discards)
    ImpEq,
}

/// `eq`: a shading pass after the depth prepass (depth `EQUAL`, no writes).
fn pipeline(d: &ash::Device, layout: vk::PipelineLayout, kind: Kind, eq: bool) -> Result<vk::Pipeline, BackendError> {
    // SAFETY: device valid; create infos reference live locals; modules destroyed after creation.
    unsafe {
        let (vs_spv, fs_spv): (&[u8], Option<&[u8]>) = match kind {
            Kind::Terrain => (TERRAIN_VS, Some(TERRAIN_FS)),
            Kind::Shadow => (SHADOW_VS, None),
            Kind::Sky => (SKY_VS, Some(SKY_FS)),
            Kind::Inst => (INST_VS, Some(TERRAIN_FS)),
            Kind::InstFade => (INST_VS, Some(INST_FS)),
            Kind::InstShadow => (INST_SHADOW_VS, None),
            Kind::InstShadowAlpha => (INST_SHADOW_ALPHA_VS, Some(INST_SHADOW_ALPHA_FS)),
            Kind::Imp => (IMP_VS, Some(IMP_FS)),
            Kind::ImpFar => (IMP_VS, Some(IMP_FAR_FS)),
            Kind::ImpShadow => (IMP_SHADOW_VS, Some(IMP_SHADOW_FS)),
            Kind::ImpShadowFar => (IMP_SHADOW_VS, Some(IMP_SHADOW_FAR_FS)),
            Kind::PreInst => (INST_VS, None),
            Kind::PreInstAlpha => (INST_VS, Some(INST_DEPTH_FS)),
            Kind::PreImp => (IMP_VS, Some(IMP_DEPTH_FS)),
            Kind::ImpEq => (IMP_VS, Some(IMP_EQ_FS)),
        };
        let vs = shader_module(d, vs_spv)?;
        let fs = match fs_spv {
            Some(s) => Some(shader_module(d, s)?),
            None => None,
        };
        let mut stages = vec![
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vs).name(c"main"),
        ];
        if let Some(fs) = fs {
            stages.push(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fs)
                    .name(c"main"),
            );
        }
        let vi = vk::PipelineVertexInputStateCreateInfo::default();
        let ia = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let vp = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let rs = match kind {
            Kind::Terrain => vk::PipelineRasterizationStateCreateInfo::default().cull_mode(vk::CullModeFlags::BACK),
            Kind::Inst
            | Kind::InstFade
            | Kind::PreInst
            | Kind::PreInstAlpha
            | Kind::Imp
            | Kind::ImpFar
            | Kind::ImpShadow
            | Kind::ImpShadowFar
            | Kind::PreImp
            | Kind::ImpEq => vk::PipelineRasterizationStateCreateInfo::default().cull_mode(vk::CullModeFlags::NONE),
            Kind::Shadow | Kind::InstShadow | Kind::InstShadowAlpha => {
                vk::PipelineRasterizationStateCreateInfo::default()
                    .cull_mode(vk::CullModeFlags::NONE)
                    .depth_bias_enable(true)
                    .depth_bias_constant_factor(2.0)
                    .depth_bias_slope_factor(2.5)
            }
            Kind::Sky => vk::PipelineRasterizationStateCreateInfo::default().cull_mode(vk::CullModeFlags::NONE),
        }
        .polygon_mode(vk::PolygonMode::FILL)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
        let ms = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let ds = match kind {
            Kind::ImpEq => vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(false)
                .depth_compare_op(vk::CompareOp::EQUAL),
            Kind::Inst | Kind::InstFade if eq => vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(false)
                .depth_compare_op(vk::CompareOp::EQUAL),
            Kind::Terrain
            | Kind::Inst
            | Kind::InstFade
            | Kind::Imp
            | Kind::ImpFar
            | Kind::PreInst
            | Kind::PreInstAlpha
            | Kind::PreImp => vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::GREATER),
            Kind::Shadow | Kind::InstShadow | Kind::InstShadowAlpha | Kind::ImpShadow | Kind::ImpShadowFar => {
                vk::PipelineDepthStencilStateCreateInfo::default()
                    .depth_test_enable(true)
                    .depth_write_enable(true)
                    .depth_compare_op(vk::CompareOp::LESS)
            }
            Kind::Sky => vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(false)
                .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL),
        };
        let blend = [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA)];
        // Shadow passes render with no colour attachment at all.
        let depth_only = matches!(
            kind,
            Kind::Shadow | Kind::InstShadow | Kind::InstShadowAlpha | Kind::ImpShadow | Kind::ImpShadowFar
        );
        // The depth prepass draws inside the main pass, which has one colour attachment: without
        // dynamicRenderingUnusedAttachments the pipeline must declare that attachment too
        // (VUID-vkCmdDrawIndexedIndirect-colorAttachmentCount-06179), so keep it with writes masked off.
        let prepass = matches!(kind, Kind::PreInst | Kind::PreInstAlpha | Kind::PreImp);
        let no_writes =
            [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::empty())];
        let attachments: &[vk::PipelineColorBlendAttachmentState] = match (depth_only, prepass) {
            (true, _) => &[],
            (false, true) => &no_writes,
            (false, false) => &blend,
        };
        let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(attachments);
        let dyns = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dy = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyns);
        let color_formats = [COLOR];
        let mut rendering = vk::PipelineRenderingCreateInfo::default().depth_attachment_format(DEPTH);
        if !depth_only {
            rendering = rendering.color_attachment_formats(&color_formats);
        }
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
        d.destroy_shader_module(vs, None);
        if let Some(fs) = fs {
            d.destroy_shader_module(fs, None);
        }
        Ok(r?[0])
    }
}

fn pipes(d: &ash::Device, eq: bool) -> Result<Pipes, BackendError> {
    let range = [vk::PushConstantRange { stage_flags: PUSH_STAGES, offset: 0, size: size_of::<QPushGpu>() as u32 }];
    // SAFETY: device valid.
    let layout = unsafe {
        d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&range), None)?
    };
    Ok(Pipes {
        layout,
        terrain: pipeline(d, layout, Kind::Terrain, eq)?,
        shadow: pipeline(d, layout, Kind::Shadow, eq)?,
        sky: pipeline(d, layout, Kind::Sky, eq)?,
        inst: pipeline(d, layout, Kind::Inst, eq)?,
        inst_fade: pipeline(d, layout, Kind::InstFade, eq)?,
        inst_shadow: pipeline(d, layout, Kind::InstShadow, eq)?,
        inst_shadow_alpha: pipeline(d, layout, Kind::InstShadowAlpha, eq)?,
        imp: pipeline(d, layout, Kind::Imp, eq)?,
        imp_far: pipeline(d, layout, Kind::ImpFar, eq)?,
        pre_imp: pipeline(d, layout, Kind::PreImp, eq)?,
        imp_eq: pipeline(d, layout, Kind::ImpEq, eq)?,
        imp_shadow: pipeline(d, layout, Kind::ImpShadow, eq)?,
        imp_shadow_far: pipeline(d, layout, Kind::ImpShadowFar, eq)?,
        pre_inst: pipeline(d, layout, Kind::PreInst, eq)?,
        pre_inst_alpha: pipeline(d, layout, Kind::PreInstAlpha, eq)?,
        atmo: ATMO_CS.iter().map(|spv| compute_pipeline(d, layout, spv)).collect::<Result<_, _>>()?,
        qcull: QCULL_CS.iter().map(|spv| compute_pipeline(d, layout, spv)).collect::<Result<_, _>>()?,
    })
}

fn compute_pipeline(d: &ash::Device, layout: vk::PipelineLayout, spv: &[u8]) -> Result<vk::Pipeline, BackendError> {
    // SAFETY: device valid; the module is destroyed after pipeline creation.
    unsafe {
        let m = shader_module(d, spv)?;
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

impl AshBackend {
    /// Earth round 5: upload the quality pass's ground map (RGBA8 words: sRGB albedo, a = 7-bit
    /// imagery weight | 0x80 water; per level its mips, see `QParamsGpu::gmap_lv`) and return its
    /// device address for `QParamsGpu::gmap`. Replaces any previous map.
    pub fn quality_set_ground_map(&mut self, words: &[u32]) -> Result<u64, BackendError> {
        self.ctx.wait_idle();
        if let Some(mut b) = self.qground.take() {
            b.destroy(&mut self.ctx);
        }
        let sda = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER;
        let mut b = GpuBuffer::new(
            &mut self.ctx,
            "q_ground",
            size_of_val(words).max(16) as u64,
            sda,
            MemoryLocation::CpuToGpu,
        )?;
        b.mapped_mut()[..size_of_val(words)].copy_from_slice(bytemuck::cast_slice(words));
        let a = b.address;
        self.qground = Some(b);
        Ok(a)
    }

    /// Render one quality frame offscreen and return tightly packed sRGB RGBA8 pixels.
    pub fn render_quality(&mut self, f: &QualityFrame<'_>) -> Result<(Vec<u8>, QualityTimings), BackendError> {
        if f.vertices.is_empty() || f.indices.is_empty() || f.shadows.is_empty() || f.shadows.len() > 4 {
            return Err(BackendError::BadConfig("quality frame needs geometry and 1..=4 shadow views"));
        }
        self.ctx.wait_idle();
        let d = self.ctx.device.clone();
        let p = pipes(&d, f.depth_prepass)?;
        let sda = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER;
        let mut params = f.params;
        params.shadow_info[0] = f.shadow_res as f32;
        params.shadow_info[1] = f.shadows.len() as f32;
        params.shadow_info[3] = if f.spot_shadow { 1.0 } else { 0.0 };
        for (i, s) in f.shadows.iter().enumerate() {
            params.cascade_vp[i] = s.view_proj.to_cols_array_2d();
            params.texel[i] = s.texel;
        }
        let mut vb =
            GpuBuffer::new(&mut self.ctx, "q_vertices", size_of_val(f.vertices) as u64, sda, MemoryLocation::CpuToGpu)?;
        vb.mapped_mut()[..size_of_val(f.vertices)].copy_from_slice(bytemuck::cast_slice(f.vertices));
        let mut ib = GpuBuffer::new(
            &mut self.ctx,
            "q_indices",
            size_of_val(f.indices) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER,
            MemoryLocation::CpuToGpu,
        )?;
        ib.mapped_mut()[..size_of_val(f.indices)].copy_from_slice(bytemuck::cast_slice(f.indices));
        let mut pb =
            GpuBuffer::new(&mut self.ctx, "q_params", size_of::<QParamsGpu>() as u64, sda, MemoryLocation::CpuToGpu)?;
        pb.mapped_mut()[..size_of::<QParamsGpu>()].copy_from_slice(bytemuck::bytes_of(&params));
        let res = f.shadow_res;
        let smap_bytes = res as u64 * res as u64 * 4;
        // the shadow-map buffer persists across frames (cached cascades keep their depth)
        let (sb, cached_keys) = match self.qshadow.take() {
            Some(c) if c.res == res && c.count == f.shadows.len() => (c.buf, c.keys),
            other => {
                if let Some(c) = other {
                    c.destroy(&mut self.ctx);
                }
                (
                    GpuBuffer::new(
                        &mut self.ctx,
                        "q_shadow",
                        smap_bytes * f.shadows.len() as u64,
                        sda | vk::BufferUsageFlags::TRANSFER_DST,
                        MemoryLocation::GpuOnly,
                    )?,
                    [0; 4],
                )
            }
        };
        let key = |c: usize| f.shadow_cache_keys.get(c).copied().unwrap_or(0);
        let reuse: Vec<bool> = (0..f.shadows.len()).map(|c| key(c) != 0 && cached_keys[c] == key(c)).collect();
        let ext = vk::Extent2D { width: f.width, height: f.height };
        let sext = vk::Extent2D { width: res, height: res };
        let mut simg = GpuImage::new(
            &mut self.ctx,
            "q_shadow_depth",
            sext,
            DEPTH,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::DEPTH,
        )?;
        let mut color = GpuImage::new(
            &mut self.ctx,
            "q_color",
            ext,
            COLOR,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
        )?;
        let mut depth = GpuImage::new(
            &mut self.ctx,
            "q_depth",
            ext,
            DEPTH,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
        )?;
        let rb_bytes = f.width as u64 * f.height as u64 * 4;
        let rb = GpuBuffer::new(
            &mut self.ctx,
            "q_readback",
            rb_bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
        )?;
        let inst = match &f.instanced {
            Some(set) if !set.instances.is_empty() && !set.meshes.is_empty() => {
                // views: the camera (distance LOD + crossfade), then each sun cascade: the two
                // nearest switch LOD by distance (no crossfade), the far ones draw the coarsest
                // level (impostors)
                let mut views = vec![QViewGpu {
                    planes: crate::cull::frustum_planes(set.cull_view_proj).map(|p| p.to_array()),
                    mode: 0,
                    lod_scale: 1.0,
                    _p: [0; 2],
                }];
                if !f.spot_shadow {
                    for (c, sv) in f.shadows.iter().enumerate() {
                        views.push(QViewGpu {
                            planes: crate::cull::frustum_planes(sv.view_proj).map(|p| p.to_array()),
                            mode: if c < 2 { 1 } else { 2 },
                            lod_scale: if c == 1 { 2.0 } else { 1.0 },
                            _p: [0; 2],
                        });
                    }
                }
                Some(InstBufs::new(&mut self.ctx, set, &views)?)
            }
            _ => None,
        };
        let atmo = if f.atmosphere {
            Some(GpuBuffer::new(&mut self.ctx, "q_atmo", ATMO_FLOAT4S * 16, sda, MemoryLocation::GpuOnly)?)
        } else {
            None
        };
        let atmo_addr = atmo.as_ref().map_or(0, |b| b.address);
        // SAFETY: device valid.
        let qp = unsafe {
            d.create_query_pool(
                &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(MAX_MARKS),
                None,
            )?
        };
        let marks = std::cell::RefCell::new(Vec::<&'static str>::new());
        // SAFETY (callers): cmd recording; at most MAX_MARKS - 1 marks per frame.
        let mark = |cmd: vk::CommandBuffer, label: &'static str| {
            let mut m = marks.borrow_mut();
            m.push(label);
            assert!(m.len() < MAX_MARKS as usize);
            // SAFETY: cmd recording; the query was reset at the start of the submit.
            unsafe { d.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, qp, m.len() as u32) };
        };

        let (inst_addr, vis_addr, inst_vb) =
            inst.as_ref().map_or((0, 0, 0), |i| (i.inst.address, i.visible.address, i.vb.address));
        let (imp_addr, imp_tex) = inst
            .as_ref()
            .and_then(|i| i.atlas.as_ref().map(|a| (a.address, a.address + i.atlas_header)))
            .unwrap_or((0, 0));
        let push_v = |cmd: vk::CommandBuffer, cascade: u32, v: u64| {
            let pc = QPushGpu {
                p: pb.address,
                v,
                shadow: sb.address,
                cascade,
                pad: 0,
                inst: inst_addr,
                visible: vis_addr,
                atmo: atmo_addr,
                imp: imp_addr,
                imp_tex,
            };
            // SAFETY: cmd recording; layout declares this range for these stages.
            unsafe { d.cmd_push_constants(cmd, p.layout, PUSH_STAGES, 0, bytemuck::bytes_of(&pc)) };
        };
        // instanced draws of cull view `view`: the mesh range, then the impostor range, each one
        // fixed-slot indirect command per mesh (instance_count may be 0)
        // Instanced draws of cull view `view`, one fixed-slot indirect command per (slot, mesh),
        // instance_count possibly 0. Per view the slots are [whole meshes | crossfading meshes],
        // each split into the mesh range and the impostor range (`imp_first..`). `pipes` =
        // [whole meshes, crossfading meshes, far impostors, exact-depth / crossfading impostors];
        // `ranges`: bit 0 meshes, bit 1 impostors.
        let draw_inst = |cmd: vk::CommandBuffer, pipes: [vk::Pipeline; 4], cascade: u32, view: u32, ranges: u32| {
            if let Some(i) = inst.as_ref() {
                let (m, k) = (i.mesh_count, i.imp_first);
                let base = (2 * view * m) as u64;
                // SAFETY: cmd recording inside a rendering scope; buffers live for the submit; the
                // draws were written by the cull chain (barrier recorded); multiDrawIndirect is enabled.
                unsafe {
                    d.cmd_bind_index_buffer(cmd, i.ib.buffer, 0, vk::IndexType::UINT32);
                    for (bit, pipe, first, count) in
                        [(1, pipes[0], 0, k), (1, pipes[1], m, k), (2, pipes[2], k, m - k), (2, pipes[3], m + k, m - k)]
                    {
                        if count == 0 || ranges & bit == 0 {
                            continue;
                        }
                        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipe);
                        push_v(cmd, cascade, inst_vb);
                        d.cmd_draw_indexed_indirect(cmd, i.draws.buffer, (base + first as u64) * 20, count, 20);
                    }
                    d.cmd_bind_index_buffer(cmd, ib.buffer, 0, vk::IndexType::UINT32);
                }
            }
        };
        let push = |cmd: vk::CommandBuffer, cascade: u32| {
            let pc = QPushGpu {
                p: pb.address,
                v: vb.address,
                shadow: sb.address,
                cascade,
                pad: 0,
                inst: inst_addr,
                visible: vis_addr,
                atmo: atmo_addr,
                imp: imp_addr,
                imp_tex,
            };
            // SAFETY: cmd recording; layout declares this range for these stages.
            unsafe { d.cmd_push_constants(cmd, p.layout, PUSH_STAGES, 0, bytemuck::bytes_of(&pc)) };
        };
        let viewport = |cmd: vk::CommandBuffer, e: vk::Extent2D| {
            // SAFETY: cmd recording, dynamic viewport/scissor declared.
            unsafe {
                d.cmd_set_viewport(
                    cmd,
                    0,
                    &[vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: e.width as f32,
                        height: e.height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );
                d.cmd_set_scissor(cmd, 0, &[vk::Rect2D { offset: vk::Offset2D::default(), extent: e }]);
            }
        };
        let n_idx = f.indices.len() as u32;
        // terrain casters per cascade (CPU cull of the chunk spheres against the light frustum)
        let sun = Vec3::from_slice(&f.params.sun_dir[..3]).normalize_or_zero();
        let shadow_ranges: Vec<Vec<(u32, u32)>> = f
            .shadows
            .iter()
            .enumerate()
            .map(|(c, sv)| {
                let planes = crate::cull::frustum_planes(sv.view_proj);
                match f.shadow_slices.get(c) {
                    Some(&slice) if !f.spot_shadow => cull_casters(
                        f.terrain_chunks,
                        &planes,
                        &crate::cull::frustum_planes(slice),
                        sun,
                        SHADOW_SWEEP_M,
                    ),
                    _ => cull_chunks(f.terrain_chunks, &planes),
                }
            })
            .collect();
        let shadow_tris: Vec<u64> = shadow_ranges.iter().map(|r| r.iter().map(|x| x.1 as u64 / 3).sum()).collect();
        let split = std::cell::RefCell::new(0usize);
        let r = self.ctx.immediate_submit(|cmd| {
            use vk::AccessFlags2 as A;
            use vk::PipelineStageFlags2 as S;
            // SAFETY: all handles live for the duration of the submit; layouts tracked by the barriers below.
            unsafe {
                d.cmd_reset_query_pool(cmd, qp, 0, MAX_MARKS);
                d.cmd_write_timestamp2(cmd, S::TOP_OF_PIPE, qp, 0);
                if let Some(i) = inst.as_ref() {
                    // qcull: per-view cull + LOD (count) -> scan -> scatter -> fixed-slot draws
                    let rw = (S::COMPUTE_SHADER, A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE);
                    d.cmd_fill_buffer(cmd, i.counts.buffer, 0, vk::WHOLE_SIZE, 0);
                    memory_barrier(&d, cmd, (S::CLEAR, A::TRANSFER_WRITE), rw);
                    let fp = i.frame.address;
                    let vm = 2 * i.views * i.mesh_count;
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.qcull[0]);
                    d.cmd_push_constants(cmd, p.layout, PUSH_STAGES, 0, bytemuck::bytes_of(&fp));
                    d.cmd_dispatch(cmd, i.n.div_ceil(64), i.views, 1);
                    memory_barrier(&d, cmd, rw, rw);
                    record_scan(&d, &self.pipes, cmd, i.counts.address, i.offsets.address, vm, &[]);
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.qcull[1]);
                    d.cmd_push_constants(cmd, p.layout, PUSH_STAGES, 0, bytemuck::bytes_of(&fp));
                    d.cmd_dispatch(cmd, i.n.div_ceil(64), i.views, 1);
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.qcull[2]);
                    d.cmd_dispatch(cmd, vm.div_ceil(64), 1, 1);
                    memory_barrier(
                        &d,
                        cmd,
                        rw,
                        (
                            S::DRAW_INDIRECT | S::VERTEX_SHADER | S::FRAGMENT_SHADER | S::COPY,
                            A::INDIRECT_COMMAND_READ | A::SHADER_STORAGE_READ | A::TRANSFER_READ,
                        ),
                    );
                    mark(cmd, "instance cull+lod");
                }
                if atmo.is_some() {
                    // transmittance -> multi-scattering -> sky-view -> aerial perspective -> irradiance;
                    // each reads the previous LUTs (BDA), so a compute->compute barrier between them
                    let rw = (S::COMPUTE_SHADER, A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE);
                    push(cmd, 0);
                    for (k, (&pipe, &(gx, gy))) in p.atmo.iter().zip(ATMO_GROUPS.iter()).enumerate() {
                        if k > 0 {
                            memory_barrier(&d, cmd, rw, rw);
                        }
                        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
                        d.cmd_dispatch(cmd, gx, gy, 1);
                    }
                    memory_barrier(
                        &d,
                        cmd,
                        (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                        (S::FRAGMENT_SHADER | S::VERTEX_SHADER, A::SHADER_STORAGE_READ),
                    );
                    mark(cmd, "atmosphere luts");
                }
                d.cmd_bind_index_buffer(cmd, ib.buffer, 0, vk::IndexType::UINT32);
                for c in 0..f.shadows.len() as u32 {
                    if reuse[c as usize] {
                        mark(
                            cmd,
                            ["shadow c0 cached", "shadow c1 cached", "shadow c2 cached", "shadow c3 cached"]
                                [c as usize],
                        );
                        continue;
                    }
                    image_barrier(
                        &d,
                        cmd,
                        simg.image,
                        vk::ImageAspectFlags::DEPTH,
                        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL),
                        (S::COPY, A::TRANSFER_READ),
                        (
                            S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                            A::DEPTH_STENCIL_ATTACHMENT_WRITE | A::DEPTH_STENCIL_ATTACHMENT_READ,
                        ),
                    );
                    let datt = vk::RenderingAttachmentInfo::default()
                        .image_view(simg.view)
                        .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
                        });
                    let ri = vk::RenderingInfo::default()
                        .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: sext })
                        .layer_count(1)
                        .depth_attachment(&datt);
                    d.cmd_begin_rendering(cmd, &ri);
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, p.shadow);
                    viewport(cmd, sext);
                    push(cmd, c);
                    if f.terrain_chunks.is_empty() {
                        d.cmd_draw_indexed(cmd, n_idx, 1, 0, 0, 0);
                    } else {
                        for &(first, count) in &shadow_ranges[c as usize] {
                            d.cmd_draw_indexed(cmd, count, 1, first, 0, 0);
                        }
                    }
                    mark(
                        cmd,
                        ["shadow c0 terrain", "shadow c1 terrain", "shadow c2 terrain", "shadow c3 terrain"]
                            [c as usize],
                    );
                    if !f.spot_shadow {
                        // near cascades: exact-depth impostor casters (beyond the mesh LOD range); far
                        // cascades: coverage-only cards
                        let ip = if c < 2 { p.imp_shadow } else { p.imp_shadow_far };
                        // class-1 mesh slots in shadow views = alpha-tested casters (needle cards)
                        draw_inst(cmd, [p.inst_shadow, p.inst_shadow_alpha, ip, ip], c, 1 + c, 3);
                        mark(
                            cmd,
                            [
                                "shadow c0 instances",
                                "shadow c1 instances",
                                "shadow c2 instances",
                                "shadow c3 instances",
                            ][c as usize],
                        );
                    }
                    d.cmd_end_rendering(cmd);
                    image_barrier(
                        &d,
                        cmd,
                        simg.image,
                        vk::ImageAspectFlags::DEPTH,
                        (vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
                        (S::LATE_FRAGMENT_TESTS, A::DEPTH_STENCIL_ATTACHMENT_WRITE),
                        (S::COPY, A::TRANSFER_READ),
                    );
                    let region = [vk::BufferImageCopy {
                        buffer_offset: c as u64 * smap_bytes,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::DEPTH,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D::default(),
                        image_extent: vk::Extent3D { width: res, height: res, depth: 1 },
                    }];
                    d.cmd_copy_image_to_buffer(
                        cmd,
                        simg.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        sb.buffer,
                        &region,
                    );
                    mark(cmd, ["shadow c0 copy", "shadow c1 copy", "shadow c2 copy", "shadow c3 copy"][c as usize]);
                }
                memory_barrier(&d, cmd, (S::COPY, A::TRANSFER_WRITE), (S::FRAGMENT_SHADER, A::SHADER_STORAGE_READ));
                let shadow_end = marks.borrow().len();
                image_barrier(
                    &d,
                    cmd,
                    color.image,
                    vk::ImageAspectFlags::COLOR,
                    (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                    (S::TOP_OF_PIPE, A::NONE),
                    (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE),
                );
                image_barrier(
                    &d,
                    cmd,
                    depth.image,
                    vk::ImageAspectFlags::DEPTH,
                    (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL),
                    (S::TOP_OF_PIPE, A::NONE),
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
                    .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } })];
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
                viewport(cmd, ext);
                if f.depth_prepass {
                    // depth-only instance meshes (whole: no FS; crossfading / alpha cards: the
                    // discard-only FS). The terrain then draws normally (early-Z rejects it behind
                    // the vegetation) and the instance meshes shade with EQUAL: one shaded fragment
                    // per pixel for the foliage, which is where the overdraw is
                    if inst.is_some() {
                        draw_inst(
                            cmd,
                            [p.pre_inst, p.pre_inst_alpha, p.pre_imp, p.pre_imp],
                            0,
                            0,
                            if f.impostor_card_depth { 3 } else { 1 },
                        );
                        mark(cmd, "prepass instances");
                    }
                }
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, p.terrain);
                push(cmd, 0);
                d.cmd_draw_indexed(cmd, n_idx, 1, 0, 0, 0);
                mark(cmd, "main terrain");
                if inst.is_some() {
                    draw_inst(cmd, [p.inst, p.inst_fade, p.imp_far, p.imp], 0, 0, 1);
                    mark(cmd, "main instance meshes");
                    let imps = match (f.impostor_card_depth, f.depth_prepass) {
                        (true, true) => [p.imp_eq, p.imp_eq],
                        (true, false) => [p.imp_far, p.imp_far],
                        _ => [p.imp_far, p.imp],
                    };
                    draw_inst(cmd, [p.inst, p.inst_fade, imps[0], imps[1]], 0, 0, 2);
                    mark(cmd, "main impostors");
                }
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, p.sky);
                d.cmd_draw(cmd, 3, 1, 0, 0);
                d.cmd_end_rendering(cmd);
                mark(cmd, "main sky");
                *split.borrow_mut() = shadow_end;
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
                if let Some(i) = inst.as_ref() {
                    d.cmd_copy_buffer(
                        cmd,
                        i.draws.buffer,
                        i.readback.buffer,
                        &[vk::BufferCopy {
                            src_offset: 0,
                            dst_offset: 0,
                            size: (2 * i.views * i.mesh_count) as u64 * 20,
                        }],
                    );
                }
                memory_barrier(&d, cmd, (S::COPY, A::TRANSFER_WRITE), (S::HOST, A::HOST_READ));
            }
        });
        let mut t = QualityTimings::default();
        if r.is_ok()
            && let Some(period) = self.ctx.timestamp_period_ns
        {
            let marks = marks.borrow();
            let mut ts = vec![0u64; marks.len() + 1];
            // SAFETY: queries 0..=marks.len() were written by the completed submit.
            if unsafe {
                d.get_query_pool_results(qp, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
            }
            .is_ok()
            {
                let ms = |a: u64, b: u64| b.saturating_sub(a) as f64 * period as f64 * 1e-6;
                let se = *split.borrow();
                let last = marks.len();
                // shadow = start .. last shadow mark (instance cull + atmosphere LUTs included, as before)
                t.shadow_ms = ms(ts[0], ts[se]);
                t.main_ms = ms(ts[se], ts[last]);
                t.gpu_ms = ms(ts[0], ts[last]);
                t.breakdown = marks.iter().enumerate().map(|(k, &l)| (l, ms(ts[k], ts[k + 1]))).collect();
            }
        }
        t.shadow_terrain_triangles =
            if f.terrain_chunks.is_empty() { vec![n_idx as u64 / 3; f.shadows.len()] } else { shadow_tris };
        let px = rb.mapped()[..rb_bytes as usize].to_vec();
        if let Some(i) = inst.as_ref() {
            let bytes = i.readback.mapped();
            let mc = i.mesh_count as usize;
            // DrawCmd.instance_count is the second u32 of each 20-byte command
            let slot = |k: usize| u32::from_le_bytes(bytes[k * 20 + 4..k * 20 + 8].try_into().unwrap());
            // (view, mesh): whole + crossfading slot
            let cnt = |v: usize, m: usize| slot(2 * v * mc + m) + slot(2 * v * mc + mc + m);
            for v in 0..i.views as usize {
                t.view_instances[v] = (0..mc).map(|m| cnt(v, m)).sum();
                t.view_triangles[v] = (0..mc).map(|m| cnt(v, m) as u64 * (i.mesh_tris[m] as u64)).sum();
            }
            t.visible_instances = t.view_instances[0];
            t.impostor_instances = (i.imp_first as usize..mc).map(|m| cnt(0, m)).sum();
            t.fading_instances = (0..mc).map(|m| slot(mc + m)).sum();
            t.camera_mesh_draws = (0..mc).map(|m| (cnt(0, m), cnt(0, m) as u64 * i.mesh_tris[m] as u64)).collect();
        }
        // SAFETY: the submit completed (immediate_submit waits); nothing references these objects.
        unsafe {
            d.destroy_query_pool(qp, None);
            d.destroy_pipeline(p.terrain, None);
            d.destroy_pipeline(p.shadow, None);
            d.destroy_pipeline(p.sky, None);
            d.destroy_pipeline(p.inst, None);
            d.destroy_pipeline(p.inst_shadow, None);
            d.destroy_pipeline(p.inst_shadow_alpha, None);
            d.destroy_pipeline(p.inst_fade, None);
            d.destroy_pipeline(p.imp, None);
            d.destroy_pipeline(p.imp_far, None);
            d.destroy_pipeline(p.pre_imp, None);
            d.destroy_pipeline(p.imp_eq, None);
            d.destroy_pipeline(p.imp_shadow_far, None);
            d.destroy_pipeline(p.imp_shadow, None);
            d.destroy_pipeline(p.pre_inst, None);
            d.destroy_pipeline(p.pre_inst_alpha, None);
            for &a in p.atmo.iter().chain(&p.qcull) {
                d.destroy_pipeline(a, None);
            }
            d.destroy_pipeline_layout(p.layout, None);
        }
        for mut b in [vb, ib, pb, rb] {
            b.destroy(&mut self.ctx);
        }
        // keep the shadow maps for the next frame; after a failed submit nothing is trusted
        let mut keys = [0u64; 4];
        if r.is_ok() {
            for (c, k) in keys.iter_mut().enumerate().take(f.shadows.len()) {
                *k = key(c);
            }
        }
        self.qshadow = Some(QShadowCache { buf: sb, res, count: f.shadows.len(), keys });
        t.cached_cascades = reuse.iter().filter(|&&x| x).count() as u32;
        if let Some(i) = inst {
            i.destroy(&mut self.ctx);
        }
        if let Some(mut a) = atmo {
            a.destroy(&mut self.ctx);
        }
        for i in [&mut simg, &mut color, &mut depth] {
            i.destroy(&mut self.ctx);
        }
        r?;
        Ok((px, t))
    }
}

/// GPU buffers of an [`InstancedSet`] plus the per-view cull scratch.
struct InstBufs {
    vb: GpuBuffer,
    ib: GpuBuffer,
    meshes: GpuBuffer,
    inst: GpuBuffer,
    lod: GpuBuffer,
    frame: GpuBuffer,
    counts: GpuBuffer,
    offsets: GpuBuffer,
    slots: GpuBuffer,
    visible: GpuBuffer,
    draws: GpuBuffer,
    readback: GpuBuffer,
    atlas: Option<GpuBuffer>,
    atlas_header: u64,
    n: u32,
    mesh_count: u32,
    imp_first: u32,
    views: u32,
    mesh_tris: Vec<u32>,
}

impl InstBufs {
    fn new(
        ctx: &mut crate::context::VkContext,
        set: &InstancedSet<'_>,
        views: &[QViewGpu],
    ) -> Result<Self, BackendError> {
        let sda = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER;
        let n = set.instances.len() as u32;
        let mc = set.meshes.len() as u32;
        let nv = views.len() as u32;
        if views.is_empty()
            || views.len() > MAX_VIEWS
            || 2 * nv * mc > SCAN_TILE
            || set.impostor_first > mc
            || n >= 1 << 24
        {
            return Err(BackendError::BadConfig(
                "instanced set: 1..=5 views, 2 x views x meshes <= 1024, < 2^24 instances",
            ));
        }
        if !set.lod.is_empty() && set.lod.len() != mc as usize {
            return Err(BackendError::BadConfig("instanced set: lod table must have one entry per mesh"));
        }
        let up = |ctx: &mut crate::context::VkContext,
                  name: &str,
                  bytes: &[u8],
                  usage: vk::BufferUsageFlags|
         -> Result<GpuBuffer, BackendError> {
            let mut b = GpuBuffer::new(ctx, name, bytes.len().max(16) as u64, usage, MemoryLocation::CpuToGpu)?;
            b.mapped_mut()[..bytes.len()].copy_from_slice(bytes);
            Ok(b)
        };
        let gpu = |ctx: &mut crate::context::VkContext, name: &str, bytes: u64, usage: vk::BufferUsageFlags| {
            GpuBuffer::new(ctx, name, bytes.max(16), usage, MemoryLocation::GpuOnly)
        };
        let td = vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
        let lod: Vec<[u32; 4]> =
            if set.lod.is_empty() { vec![[LOD_NONE, 0, 0, 0]; mc as usize] } else { set.lod.to_vec() };
        let mut b = Self {
            vb: up(ctx, "qi_vertices", bytemuck::cast_slice(set.vertices), sda)?,
            ib: up(ctx, "qi_indices", bytemuck::cast_slice(set.indices), vk::BufferUsageFlags::INDEX_BUFFER)?,
            meshes: up(ctx, "qi_meshes", bytemuck::cast_slice(set.meshes), sda)?,
            inst: up(ctx, "qi_instances", bytemuck::cast_slice(set.instances), sda)?,
            lod: up(ctx, "qi_lod", bytemuck::cast_slice(&lod), sda)?,
            frame: GpuBuffer::new(ctx, "qi_frame", size_of::<QCullFrameGpu>() as u64, sda, MemoryLocation::CpuToGpu)?,
            counts: gpu(ctx, "qi_counts", (2 * nv * mc) as u64 * 4, sda | td)?,
            offsets: gpu(ctx, "qi_offsets", (2 * nv * mc) as u64 * 4, sda)?,
            slots: gpu(ctx, "qi_slots", (nv * n) as u64 * 16, sda)?,
            visible: gpu(ctx, "qi_visible", (nv * n) as u64 * 8, sda)?,
            draws: gpu(ctx, "qi_draws", (2 * nv * mc) as u64 * 20, sda | td | vk::BufferUsageFlags::INDIRECT_BUFFER)?,
            readback: GpuBuffer::new(
                ctx,
                "qi_readback",
                (2 * nv * mc) as u64 * 20,
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
            )?,
            atlas: match set.atlas {
                Some((bytes, _)) if !bytes.is_empty() => Some(up(ctx, "qi_atlas", bytes, sda)?),
                _ => None,
            },
            atlas_header: set.atlas.map_or(0, |(_, v)| (v as usize * crate::impostor::HEADER_BYTES) as u64),
            n,
            mesh_count: mc,
            imp_first: set.impostor_first,
            views: nv,
            mesh_tris: set.meshes.iter().map(|m| m.index_count / 3).collect(),
        };
        let mut vs = [QViewGpu::zeroed(); MAX_VIEWS];
        vs[..views.len()].copy_from_slice(views);
        let fr = QCullFrameGpu {
            inst: b.inst.address,
            meshes: b.meshes.address,
            lod: b.lod.address,
            counts: b.counts.address,
            offsets: b.offsets.address,
            slots: b.slots.address,
            visible: b.visible.address,
            draws: b.draws.address,
            n,
            mesh_count: mc,
            view_count: nv,
            _p: 0,
            views: vs,
        };
        b.frame.mapped_mut()[..size_of::<QCullFrameGpu>()].copy_from_slice(bytemuck::bytes_of(&fr));
        Ok(b)
    }

    fn destroy(self, ctx: &mut crate::context::VkContext) {
        for mut b in [
            self.vb,
            self.ib,
            self.meshes,
            self.inst,
            self.lod,
            self.frame,
            self.counts,
            self.offsets,
            self.slots,
            self.visible,
            self.draws,
            self.readback,
        ]
        .into_iter()
        .chain(self.atlas)
        {
            b.destroy(ctx);
        }
    }
}

#[cfg(test)]
mod chunk_tests {
    use super::*;

    #[test]
    fn terrain_chunks_keep_every_triangle_and_bound_their_vertices() {
        // a 60 x 60 grid of 50 m quads around the camera
        let n = 60u32;
        let v: Vec<QVertex> = (0..=n)
            .flat_map(|j| {
                (0..=n).map(move |i| QVertex {
                    pos: [(i as f32 - 30.0) * 50.0, ((i * 7 + j * 3) % 11) as f32, (j as f32 - 30.0) * 50.0],
                    ..Default::default()
                })
            })
            .collect();
        let mut idx = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let a = j * (n + 1) + i;
                idx.extend([a, a + n + 1, a + 1, a + 1, a + n + 1, a + n + 2]);
            }
        }
        let (out, chunks) = terrain_chunks(&v, &idx, 150.0);
        let key = |x: &[u32]| {
            let mut t: Vec<[u32; 3]> = x.as_chunks::<3>().0.to_vec();
            t.sort();
            t
        };
        assert_eq!(key(&out), key(&idx), "same triangles, same winding");
        assert!(chunks.len() > 10);
        let mut next = 0;
        for c in &chunks {
            assert_eq!(c.first_index, next);
            next += c.index_count;
            let ctr = Vec3::new(c.sphere[0], c.sphere[1], c.sphere[2]);
            for &i in &out[c.first_index as usize..(c.first_index + c.index_count) as usize] {
                assert!(Vec3::from_array(v[i as usize].pos).distance(ctr) <= c.sphere[3]);
            }
        }
        assert_eq!(next as usize, out.len());
        // a frustum containing everything keeps one merged range; an empty one none
        let all = [glam::Vec4::new(0.0, 0.0, 0.0, 1.0); 6];
        assert_eq!(cull_chunks(&chunks, &all), vec![(0, out.len() as u32)]);
        let none = [glam::Vec4::new(1.0, 0.0, 0.0, -1e6); 6];
        assert!(cull_chunks(&chunks, &none).is_empty());
    }
}

#[cfg(test)]
mod clip_cascade_tests {
    use super::*;
    use glam::DVec3;

    fn ndc(m: Mat4, rel: Vec3) -> Vec3 {
        let c = m * rel.extend(1.0);
        c.truncate() / c.w
    }

    #[test]
    fn clip_cascade_covers_every_direction_and_keeps_its_cell_while_walking() {
        let (res, f, grid) = (2048u32, 2500.0f32, 256u32);
        for sun in
            [Vec3::new(0.4, 0.48, -0.78).normalize(), Vec3::new(-0.2, 0.9, 0.1).normalize(), Vec3::new(0.0, 1.0, 0.0)]
        {
            let cam0 = DVec3::new(7799.3, 562.6, -3614.2);
            let (m0, texel, cell0) = sun_cascade_clip(cam0, f, sun, res, 6000.0, grid);
            assert!(texel * res as f32 * 0.5 <= 1.15 * f, "half-size {} for f {f}", texel * res as f32 * 0.5);
            let g = texel as f64 * grid as f64;
            let mut changes = 0;
            let mut prev = cell0;
            let mut cam = cam0;
            for step in 0..4000 {
                // walk 0.25 m steps on a curving path with some climbing
                let a = step as f64 * 0.002;
                cam += DVec3::new(a.cos(), 0.15 * (a * 3.0).sin(), a.sin()) * 0.25;
                let (m, _, cell) = sun_cascade_clip(cam, f, sun, res, 6000.0, grid);
                if cell != prev {
                    changes += 1;
                    prev = cell;
                }
                // every point within f of the camera, in every direction, is inside the map
                for k in 0..64 {
                    let (u, v) = ((k as f32 * 0.618_034).fract(), (k as f32 + 0.5) / 64.0);
                    let (ph, ct) = (u * std::f32::consts::TAU, 2.0 * v - 1.0);
                    let st = (1.0 - ct * ct).sqrt();
                    let p = ndc(m, Vec3::new(st * ph.cos(), ct, st * ph.sin()) * f * 0.999);
                    assert!(p.x.abs() <= 1.0 && p.y.abs() <= 1.0 && (0.0..=1.0).contains(&p.z), "{p:?} at step {step}");
                }
                // same cell: the same world-space map (compare at a world point near the camera)
                if cell == cell0 {
                    let q = Vec3::new(31.0, -7.0, 12.0);
                    let a = ndc(m, q);
                    let b = ndc(m0, q + (cam - cam0).as_vec3());
                    assert!((a - b).abs().max_element() < 2e-4, "{a:?} vs {b:?}");
                }
            }
            // ~1000 m walked: the cell changes about once per g (= 2 R / 8 ~ 714 m) of travel per axis
            let walked = 1000.0;
            assert!(changes as f64 <= 3.0 * walked / g + 3.0, "{changes} cell changes, g {g:.0} m");
        }
    }
}

#[cfg(test)]
mod layout_tests {
    use super::QParamsGpu;
    use std::mem::{offset_of, size_of};

    /// `QParamsGpu` is read as Slang's `QParams` through a device address (scalar layout): the
    /// round-5 ground-map fields must sit where the shader expects them.
    #[test]
    fn qparams_layout_matches_slang() {
        assert_eq!(offset_of!(QParamsGpu, atmo), 128 + 15 * 16 + 256);
        assert_eq!(offset_of!(QParamsGpu, gmap), 128 + 15 * 16 + 256 + 32);
        assert_eq!(offset_of!(QParamsGpu, gmap_lv), offset_of!(QParamsGpu, gmap) + 16);
        assert_eq!(offset_of!(QParamsGpu, gmap_f), offset_of!(QParamsGpu, gmap_lv) + 32);
        assert_eq!(size_of::<QParamsGpu>(), offset_of!(QParamsGpu, gmap_f) + 16);
        let src = include_str!("../shaders/quality.slang");
        let body = &src[src.find("struct QParams {").unwrap()..];
        let body = &body[..body.find("};").unwrap()];
        let tail: Vec<&str> = body
            .lines()
            .map(|l| l.split("//").next().unwrap().trim())
            .filter(|l| !l.is_empty())
            .skip_while(|l| !l.starts_with("float4 atmo;"))
            .collect();
        assert_eq!(
            tail,
            ["float4 atmo;", "float4 atmo2;", "uint* gmap;", "uint2 gmap_pad;", "uint4 gmap_lv[2];", "float4 gmap_f;"]
        );
    }
}
