# Bevy + BRUTAL-style low-level Rust: starter kit

_Versions checked on crates.io, 19 Oct 2026. Bevy 0.20 was released 8 Oct 2026._

> **This file now ships with a real workspace** in this directory. `cargo check --workspace --all-targets`,
> `cargo test --workspace` and `cargo clippy --workspace --all-targets -- -D warnings` are clean on
> rustc 1.99.1 against bevy 0.20.0 (88 tests, including GPU tests on lavapipe with validation; also green with `BRUTAL_SCAN=shared` and `--all-features` clippy). See [§0 Build status](#0-build-status) for what is real and what is stubbed.

## 0. Build status

| Crate | State |
|---|---|
| `lowlevel-abi` | Real: `ParticleBuffers`/`ParticlesSoA` (lane-padded SoA), `InstanceGpu` (80 B, Pod), `FrameSnapshot`, `RenderBackend`, `HotSystem`. |
| `mem` | Real: `FrameArena` (64-B aligned bump arena on `UnsafeCell` storage, **Miri-clean**), `SlotPool`. |
| `hot` | Real: `integrate_scalar` (oracle), `integrate_simd` (`wide::f32x8`), `integrate_simd_par` (rayon), `ParticleIntegrator: HotSystem`. Tests check SIMD vs scalar to 1e-5 relative error. |
| `terrain` | Real: deterministic fBm value noise (scalar + `f32x8`), padded 67x67 heightmaps, grid meshing into `FrameArena`, shared index buffer. Tests cover SIMD vs scalar (bit-identical), golden hash, determinism, single-LOD seams and normals. `SeamStrategy` trait stub (`SingleLod` only; LOD seams on hold pending discussion). Prompt C single-LOD part **done**: chunk_full 0.055 ms with `x86-64-v3`. |
| `render-ash` | **Real GPU-driven instanced pass (Prompt B, done + round 2):** all-GPU Slang chain: cull+count → multi-level prefix sum → scatter + indirect-args → `vkCmdDrawIndexedIndirectCount`. Frames in flight are configurable (2–3) and overlap-safe, with per-slot scratch, targets, stats and queries, and no inter-frame barriers. Also: timeline-semaphore pacing, timestamp queries, offscreen target + readback, resize, and a swapchain/present path behind `windowed`. Validation + sync validation are clean (headless and Xvfb windowed with resize). See [Prompt B status](#prompt-b-status--implemented-19-oct-2026-executed-in-this-repo-results-on-lavapipe). **Round 3:** subgroup (`WavePrefixSum`) tile scan with runtime capability query + shared-memory fallback (`BRUTAL_SCAN`). Also a world-space `surface` pipeline (Slang) for terrain chunks (staging ring + slot residency with LRU eviction) and static meshes (voxels). |
| `coords` | Real (new): floor div/mod, `BrickGrid` (runtime brick size 16/32/64 + voxel spacing, LOD steps), lattice ↔ world ↔ brick, camera-relative f32 (`to_render`). |
| `voxel` | Real (new): sparse voxel bricks, density fields (cave/overhang scene, test blob), CPU **Transvoxel** (regular + transition cells, Lengyel's MIT tables) with watertight one-level LOD seams, robustness rules R1–R6, topology checker, bench. See [voxel status](#near-field-sparse-voxel-bricks-status). |
| `asset` | Real (new, 21 Oct 2026): brick/mesh/instance streaming format. FlatBuffers-style zero-copy container (vtables, optional fields, 256-B aligned GPU blobs, fuzzed verifier, `major` version), Draco-style compression (lossless sub-grid positions, bit-exact seams, Edgebreaker-like connectivity 2.85 bits/tri, octahedral normals, rANS): **11.4:1 lossless** vs the raw GPU layout (12.2:1 on the round-4 terrain), ~290–300 MB/s/core decode. Miri-clean. Format minor 1: 16-bit instance materials; the quality example's vegetation now streams through it. See [Learning from FlatBuffers and Draco](#learning-from-flatbuffers-and-draco-the-asset-crate--21-oct-2026). |
| quality: impostors + boulders | Real (21 Oct 2026, round 5): octahedral tree impostors with GPU per-instance, per-view LOD (`qcull.slang`), terrain-chunk shadow-caster culling, procedurally fractured boulders (`voxel::boulder`). See [Tree impostors, GPU LOD and boulders](#tree-impostors-gpu-lod-and-boulders--21-oct-2026). |
| quality: blade grass, card trees | Real (21 Oct 2026, round 6): GPU-LOD blade-grass tiles (`render_ash::grass`), needle-card conifers + leaf-card shrubs with alpha-tested shadows, faceted boulders, neutral shade. See [Blade grass, needle-card trees, faceted boulders, shade balance](#blade-grass-needle-card-trees-faceted-boulders-shade-balance--21-oct-2026). |
| `ffi` | Stub: one `unsafe extern "C"` libm binding with a safe wrapper (edition-2024 pattern). |
| `bridge` | Real: `BrutalParticlesPlugin` (FixedUpdate: ECS -> SoA -> SIMD -> write back, despawns dead particles) and `BrutalRenderPlugin<B: RenderBackend>` (PostUpdate after `TransformSystems::Propagate`: gather `InstanceGpu`, submit to a NonSend backend). Integration test on `MinimalPlugins` with a manual fixed clock. |
| `app` | Real: headless by default (runs `FRAMES` frames with `PARTICLES` entities, builds 16 terrain chunks, renders the particles as instances through the ash backend if a Vulkan 1.3 driver exists). Features: `vulkan` (default), `windowed` (winit window, no bevy_render, ash swapchain; verified under Xvfb incl. live resize via `BRUTAL_RESIZE_AT=N`), `tracy`. Every feature combination passes `cargo check`. |
| `benches` | Real: criterion `particles` (scalar / simd_1t / simd_mt at 10k/100k/1M) and `terrain`. Budgets live in `budgets.toml`. |

Measured on this shared 8-vCPU box (noisy, not the reference machine), criterion with a 3 s measurement:

| Bench | 10k | 100k | 1M |
|---|---|---|---|
| particles scalar | 26.0 µs | 257 µs | 2.64 ms |
| particles simd_1t (`f32x8`) | 9.1 µs | 91.7 µs | 0.94 ms |
| particles simd_mt (rayon, 16k chunks) | 9.0 µs | 33.5 µs | **0.165 ms** (budget 0.60) |

Terrain LOD0 chunk (65x65), before Prompt C: heightmap scalar 262 µs, SIMD 140 µs, SIMD + mesh 158 µs (over the 0.12 ms budget). **After Prompt C (`.cargo/config.toml` sets `target-cpu=x86-64-v3`): SIMD heightmap ~43 µs, mesh ~12 µs, chunk_full ~55 µs at every LOD (budget 0.12 ms met).** The headless app ran 100k particles + instance gather + ash upload at about 395 fps on lavapipe *before the backend drew anything*. With the real pass it runs ~21 fps (120 frames, F = 3), dominated by lavapipe rasterizing ~26k live particle instances at 1280x720.

Run:
```bash
cargo test --workspace
cargo bench -p benches --benches -- --warm-up-time 1 --measurement-time 3
PARTICLES=100000 FRAMES=300 cargo run --release -p app
cargo run --release -p app --features tracy        # with the Tracy profiler open
cargo +nightly miri test -p mem
cargo run --release -p render-ash --example instanced_stress -- --instances 10000,100000,1000000 --png out/instanced.png
cargo run --release -p render-ash --example instanced_stress -- --instances 10000 --frames 300 --validate --fif 3
BRUTAL_FRAMES_IN_FLIGHT=3 PARTICLES=20000 FRAMES=150 BRUTAL_RESIZE_AT=60 xvfb-run -a cargo run -p app --features windowed
BRUTAL_SCAN=shared cargo test -p render-ash        # force the shared-memory scan fallback (auto|subgroup|shared)
cargo run --release -p render-ash --example scan_bench
cargo run --release -p render-ash --example terrain_flyover -- --frames 240 --validate --png out/terrain.png
cargo run --release -p render-ash --example voxel_scene -- --brick 32 --voxel 0.25 --validate --png out/voxel.png
cargo run --release -p voxel --example voxel_bench
cargo run --release -p render-ash --example quality -- --scene cave         # -> out/quality/cave.png + .json
cargo run --release -p render-ash --example quality -- --scene mountains    # -> out/quality/mountains.png + .json
cargo test --release -p voxel --test quality                                # q1-q5 (sawtooth, seams, islands, peaks)
cargo test -p asset && cargo +nightly miri test -p asset --lib              # asset format (ASSET_FUZZ_ITERS=60000 for a long fuzz)
cargo run --release -p asset --example asset_bench && /workspace/.venv-draco/bin/python tools/draco_compare.py
```
Tooling used: rustc 1.99.1, `slangc` 2026.20 (release binary in /opt/slang), Vulkan validation layers 1.4.309, Mesa lavapipe (llvmpipe, LLVM 19), spirv-tools, Xvfb.

## 1. Workspace layout

```
bevy-brutal/
├── Cargo.toml                 # workspace, shared deps, lints, profiles
├── .cargo/config.toml         # target-cpu=x86-64-v3 (AVX2+FMA) for x86_64 Linux; delete or RUSTFLAGS="" for baseline
├── crates/
│   ├── app/                   # bin `game`: App::new(), plugins, features vulkan/windowed/tracy
│   ├── bridge/                # Bevy plugins: ECS -> SoA / InstanceGpu -> low-level crates
│   ├── lowlevel-abi/          # NO bevy dep. POD types, SoA views, traits (the contract)
│   ├── render-ash/            # ash + gpu-allocator backend, shaders/*.slang, examples/, tests/ (unsafe allowed)
│   ├── hot/                   # SIMD hot systems: particles (later: culling, skinning)
│   ├── terrain/               # SIMD heightmap gen + meshing into mem::FrameArena
│   ├── coords/                # coordinate frames: lattice / brick / world f64 / render-relative f32
│   ├── voxel/                 # sparse voxel bricks + CPU Transvoxel (LOD seams)
│   ├── asset/                 # BRK1 streaming format: FlatBuffers-style container + Draco-style codecs
│   ├── mem/                   # arenas, slot pools (unsafe allowed, Miri)
│   └── ffi/                   # native bindings (Jolt/FMOD/etc.), bindgen output (unsafe allowed)
├── benches/                   # criterion benches (benches/*.rs) + budgets.toml
├── tools/compile_shaders.sh   # regenerate checked-in SPIR-V from Slang (tools/profile: planned)
└── .agents/                   # (planned) agent prompts from §2, PERF.md, SAFETY.md
```

Rules:
- `lowlevel-abi`, `hot`, `terrain`, `mem`, `render-ash`, `ffi` **never depend on Bevy**. That lets agents build and bench them in seconds, and Bevy upgrades can't break them.
- Only `bridge` knows about both sides.
- `#![forbid(unsafe_code)]` in `app`, `bridge`, `lowlevel-abi`, `hot`, `terrain`, `coords`, `voxel`. In the workspace, `[workspace.lints]` denies `unsafe_op_in_unsafe_fn` and `clippy::undocumented_unsafe_blocks` for everything, so `mem`, `render-ash` and `ffi` need a `// SAFETY:` comment on every block.

### Root Cargo.toml sketch

```toml
[workspace]
resolver = "3"
members = ["crates/*", "benches"]

[workspace.package]
edition = "2024"

[workspace.dependencies]
bevy              = { version = "0.20.0", default-features = false }
ash               = "0.38.0"            # 0.38.0+1.3.281
ash-window        = "0.13.0"
raw-window-handle = "0.6.2"
gpu-allocator     = { version = "0.28.0", default-features = false, features = ["std", "vulkan"] }  # "std" (or "hashbrown") is REQUIRED with default-features = false
bytemuck          = { version = "1.25.2", features = ["derive"] }
glam              = { version = "0.34.1", features = ["bytemuck"] }  # NB: bevy_math pins its own glam; convert at the bridge
wide              = "1.7.1"
tracy-client      = { version = "0.19.0", default-features = false }
tracing-tracy     = "0.12.0"
criterion         = "0.8.2"
rayon             = "1"

[profile.release]
debug = "line-tables-only"   # Tracy symbols
[profile.bench]
inherits = "release"
lto = "thin"
```

`app/Cargo.toml`: there are two modes.
```toml
# Mode A: keep Bevy's renderer, add custom passes / raw Vulkan features
bevy = { workspace = true, features = ["default", "raw_vulkan_init", "trace_tracy"] }
# Mode B (Flo/Tiny Glade style; what this repo does): no bevy_render, you own the swapchain
bevy = { workspace = true, features = ["std", "multi_threaded", "bevy_log"] }
[features]
windowed = ["bevy/bevy_window", "bevy/bevy_winit", "bevy/x11", "bevy/wayland"]
tracy    = ["bevy/trace_tracy", "bridge/tracy", "terrain/tracy", "render-ash?/tracy"]
```
Notes from building it: `bevy_log` is needed for the `info!`/`error!` macros. `bevy_transform` and `bevy_time` are always on. `MinimalPlugins` does **not** include `TransformPlugin`, so add it yourself when headless. In the low-level crates, Tracy is `tracy-client` with `default-features = false` (every span is a no-op) plus a crate feature `tracy = ["tracy-client/enable"]`. Spans are behind `#[cfg(feature = "tracy")]`. Bevy's `trace_tracy` starts the client through tracing-tracy, so `span!` has a running client.

### The contract (`lowlevel-abi`)

```rust
#![forbid(unsafe_code)]
use bytemuck::{Pod, Zeroable};

/// SoA particle storage. All slices have equal len. Capacity is padded to a multiple of 8 (one AVX lane group).
pub struct ParticlesSoA<'a> {
    pub px: &'a mut [f32], pub py: &'a mut [f32], pub pz: &'a mut [f32],
    pub vx: &'a mut [f32], pub vy: &'a mut [f32], pub vz: &'a mut [f32],
    pub life: &'a mut [f32],
}

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
pub struct InstanceGpu { pub model: [[f32; 4]; 4], pub mesh_id: u32, pub material_id: u32, pub _pad: [u32; 2] }

/// Snapshot the ECS hands to the renderer each frame. Plain data, no Bevy types.
pub struct FrameSnapshot<'a> {
    pub view_proj: [[f32; 4]; 4],
    pub instances: &'a [InstanceGpu],
    pub frame_index: u64,
}

pub trait RenderBackend {
    type Error: core::fmt::Debug;
    fn resize(&mut self, w: u32, h: u32) -> Result<(), Self::Error>;
    /// Must not allocate on the host after warm-up. Must not block longer than `budget`.
    fn submit(&mut self, frame: &FrameSnapshot<'_>) -> Result<(), Self::Error>;
}

pub trait HotSystem {
    type Input<'a>; type Output<'a>;
    fn run(&mut self, input: Self::Input<'_>, out: Self::Output<'_>);
}
```

### The bridge (Bevy 0.20 side, safe code only). Excerpt from `crates/bridge/src/lib.rs`

```rust
use bevy::prelude::*;
use bevy::transform::TransformSystems;

impl Plugin for BrutalParticlesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ParticleSettings>().init_resource::<ParticleSoA>()
            .configure_sets(FixedUpdate,
                (BrutalSystems::ExtractParticles, BrutalSystems::Simulate, BrutalSystems::WriteBack).chain())
            .add_systems(FixedUpdate, (
                extract_particles.in_set(BrutalSystems::ExtractParticles),   // Query -> SoA (reused Vecs)
                simulate_particles.in_set(BrutalSystems::Simulate),          // hot::ParticleIntegrator (f32x8 + rayon)
                write_back_particles.in_set(BrutalSystems::WriteBack),       // SoA -> Transform/Particle, despawn dead
            ));
    }
}

impl<B: RenderBackend + 'static> Plugin for BrutalRenderPlugin<B> {
    fn build(&self, app: &mut App) {
        app.init_resource::<InstanceScratch>().init_resource::<ViewProj>()
            .add_systems(PostUpdate,
                (gather_instances, submit_frame::<B>).chain().after(TransformSystems::Propagate));
    }
}

pub fn gather_instances(q: Query<(&GlobalTransform, &MeshRef)>, mut scratch: ResMut<InstanceScratch>) {
    scratch.0.clear();
    scratch.0.extend(q.iter().map(|(t, m)| InstanceGpu {
        model: t.to_matrix().to_cols_array_2d(), mesh_id: m.mesh_id, material_id: m.material_id, _pad: [0; 2],
    }));
}

pub fn submit_frame<B: RenderBackend + 'static>(
    scratch: Res<InstanceScratch>, view_proj: Res<ViewProj>,
    backend: Option<NonSendMut<B>>, mut frame: Local<u64>,
) {
    let Some(mut backend) = backend else { return };
    *frame += 1;
    let snap = FrameSnapshot { view_proj: view_proj.0.to_cols_array_2d(), instances: &scratch.0, frame_index: *frame };
    if let Err(e) = backend.submit(&snap) { bevy::log::error!("render backend submit failed: {e:?}"); }
}
// app: world.insert_non_send(render_ash::AshBackend::new_headless(1 << 20)?);
```

**Bevy 0.20 API corrections, verified by compiling against bevy 0.20.0:**
- `GlobalTransform::to_matrix()` ✅ exists. `TransformSystems::Propagate` ✅ exists, imported as `bevy::transform::TransformSystems`.
- Non-send resources: `App::insert_non_send` / `World::non_send` / `World::non_send_mut`. The older `insert_non_send_resource` / `non_send_resource` names **don't exist** in 0.20. The `NonSend` / `NonSendMut` system params are unchanged.
- App exit is a message: `MessageWriter<AppExit>` + `exit.write(AppExit::Success)`.
- Extraction moved into a new **`bevy_extract`** crate in 0.20 (`ExtractSchedule`, `Extract<P>`, `ExtractComponent`, `ExtractResource`). `bevy::render` still re-exports them, and `RenderApp` stays in `bevy_render`.
- `RawHandleWrapper::get_display_handle()` / `get_window_handle()` are **safe** fns returning `RawDisplayHandle` / `RawWindowHandle`. Only `get_handle()` is `unsafe`.
- The earlier sketch's `Single<(&GlobalTransform, &Projection), With<Camera>>` needs `bevy_camera` (Mode A). In Mode B the game writes a plain `ViewProj(Mat4)` resource instead.
- `wide` 1.7: `f32x8::blend` is deprecated. Use `mask.bitselect(if_one, if_zero)` for full-lane masks from `simd_gt`.
- glam 0.34 removed `Mat4::perspective_rh` / `look_at_rh` in favour of `glam::camera::rh::proj::{vulkan,directx,opengl}::perspective` and `glam::camera::rh::view::look_at_mat4`. In glam 0.33 (bevy_math 0.20) they're deprecated, and Bevy re-exports the DirectX-style one as `bevy::math::proj`. For Vulkan, flip Y yourself or use glam's `vulkan` module.
- Rust 1.99 clippy prefers `slice.as_chunks_mut::<8>()` (gives `&mut [f32; 8]`, so `f32x8::from(*block)` with no unwrap) over `chunks_exact_mut(8)`.

For Mode A (keep `bevy_render`), use Bevy's own extraction instead: `app.sub_app_mut(RenderApp).add_systems(ExtractSchedule, extract_fn)` with `Extract<Query<..>>` params. Then add a render-graph node that runs your pass through wgpu, or through raw Vulkan via `raw_vulkan_init` / wgpu-hal `as_hal`.

Window handle for Mode B: query the `RawHandleWrapper` component on the primary `Window` entity, call `ash_window::enumerate_required_extensions(wrapper.get_display_handle())` before creating the instance (`VkContext::new(Some(display), ..)` does this), and then `ash_window::create_surface(&entry, &instance, display, window, None)`.

### Where unsafe is allowed
| Crate | unsafe | Rule |
|---|---|---|
| app, bridge, lowlevel-abi | forbidden | — |
| hot, terrain | forbidden for now (safe `wide` + `as_chunks_mut` was enough) | if added later: only in `simd/` modules with `// SAFETY:`, scalar oracle + randomized comparison tests |
| mem | yes | Miri must pass on the unit tests |
| render-ash, ffi | yes | `// SAFETY:` names the Vulkan VUID or C contract; validation layers clean in debug |

---

## 2. Agent prompts (copy into `.agents/`)

### Prompt A: SIMD particle integration (`crates/hot/src/particles.rs`)

```
TASK: Extend `ParticleIntegrator: HotSystem` in crates/hot (a baseline exists). Semi-implicit Euler with gravity,
linear drag and lifetime decay over SoA particle storage.

INPUTS:  ParticlesSoA<'_> (len N, padded to a multiple of 8), dt: f32 (0 < dt <= 1/30),
         gravity: [f32;3], drag: f32 in [0,1).
OUTPUTS: in-place update of p*, v*, life. Write the indices of particles whose life <= 0 to
         `dead: &mut Vec<u32>` (pre-reserved; must not reallocate when capacity >= N).
         Return the count of live particles.

BUDGET:  N = 1_000_000 particles: <= 0.60 ms on 8 threads (rayon chunks of 16k) and <= 2.5 ms single-thread,
         on the reference machine (Ryzen 7 7700 / AVX2). Zero heap allocations in `run` after warm-up.

CONSTRAINTS:
- Use `wide::f32x8` (stable) over `as_chunks_mut::<8>()` blocks. A `std::simd` path may sit behind the `nightly` cfg.
- Beware subnormals: with drag and zero gravity on an axis, velocities decay into denormals and run ~5x slower.
  Bench with non-zero gravity on every axis, or flush tiny values to zero, and document which.
- Baseline already in repo (crates/hot/src/particles.rs): 1M simd_mt = 0.165 ms. Do not regress it.
- Remainder lanes: N is padded, so do not write a scalar tail. debug_assert the padding.
- The scalar reference `integrate_scalar` must stay in the file and is the oracle.
SAFETY RULES:
- Prefer safe `chunks_exact(8)` + `f32x8::from(<[f32;8]>)`. Any `unsafe` (get_unchecked, aligned loads)
  needs a `// SAFETY:` comment stating the bound or alignment invariant and the line establishing it.
- No `unsafe` outside `hot/src/simd/`. clippy::undocumented_unsafe_blocks must pass.
BENCH HARNESS: benches/particles.rs with criterion groups `scalar`, `simd_1t`, `simd_mt` at N = 10k/100k/1M.
  Use `Throughput::Elements(N)` and a fixed seed for the data. Add a Tracy span `hot::particles`.
ACCEPTANCE:
1. proptest: SIMD vs scalar max abs error <= 1e-5 relative over 10k random cases (incl. dt edge values).
2. `cargo bench -p benches -- particles` meets the budget. Paste the criterion output.
3. Allocation counter test (a counting global allocator in tests) shows 0 allocations in `run`.
4. `cargo clippy --workspace --all-targets -- -D warnings` is clean.
5. Diff touches only crates/hot, benches/particles.rs and benches/budgets.toml.
```

### Prompt B: GPU-driven instanced mesh pass (`crates/render-ash/src/pass.rs`)

```
TASK: Add a GPU-driven instanced opaque pass to the ash backend. The CPU uploads instances. A compute
shader does frustum culling and writes VkDrawIndexedIndirectCommand + a compacted instance index buffer.
Draw with vkCmdDrawIndexedIndirectCount.

INPUTS:  FrameSnapshot.instances: &[InstanceGpu] (up to 500k), view_proj, a mesh table (mesh_id ->
         first_index, index_count, vertex_offset, bounding sphere) already resident in one big vertex/index buffer.
OUTPUTS: color + depth attachments rendered with dynamic rendering (Vulkan 1.3, no VkRenderPass).
         Expose `GpuTimings { cull_ms, draw_ms }` via timestamp queries.

BUDGET:  500k instances / 64 meshes at 1440p: CPU record + upload <= 0.30 ms. GPU cull <= 0.15 ms,
         draw <= 3.0 ms on RTX 3060 class. 0.3 ms on Steam Deck is the stretch target. No per-frame VkDeviceMemory
         allocation: use gpu-allocator with ring-buffered CpuToGpu upload buffers, frames_in_flight = 2.

CONSTRAINTS: ash 0.38, gpu-allocator 0.28 (vulkan). Shaders in Slang compiled to SPIR-V at build time
  (build.rs). Use synchronization2 barriers only. Descriptor indexing or buffer device address are both OK.
  Document the choice.
SAFETY RULES:
- Every `unsafe {}` gets a `// SAFETY:` comment naming the handles' lifetimes and the sync that makes the access
  legal (e.g. "fence[f] waited in begin_frame, so buffer[f] is not in use by GPU").
- Destruction order lives in one `Drop` impl, commented. Never destroy in-flight resources.
- Debug builds run with VK_LAYER_KHRONOS_validation + sync validation. Zero messages is required.
BENCH HARNESS: example `render-ash/examples/instanced_stress.rs` (headless offscreen target, N via CLI).
  It prints 300-frame p50/p99 CPU and GPU timings as JSON. Tracy spans `ash::record`, `ash::upload` and
  tracy GPU zones if available. A script in tools/profile compares the JSON against benches/budgets.toml.
ACCEPTANCE:
1. Validation + sync validation clean over 300 frames (log attached).
2. Image test: renders the 1k-instance scene. Its PNG matches the golden within 0.5% pixel diff.
3. The culling test puts instances fully outside the frustum and verifies draw count 0 from a readback.
4. Budgets met at p99 (JSON attached). No host allocations after frame 3 (counting allocator).
5. Public API is only `InstancedPass::new/resize/record` + `RenderBackend::submit` wiring.
```

#### Prompt B status: ✅ implemented (19 Oct 2026). Executed in this repo; results on lavapipe.

**What exists** (`crates/render-ash/`):
- `shaders/*.slang` (**Slang**, compiled by `build.rs` with `slangc` v2026.20 to SPIR-V 1.6 with `-fvk-use-scalar-layout`; checked-in `shaders/spv/*.spv` fallback when slangc is absent, regenerated with `tools/compile_shaders.sh`). Buffers are reached only through **buffer device address** (Slang pointers), so there are no descriptor sets. The main push constant holds 9 addresses (72 bytes) and the scan push constant is 32 bytes. Matrices are stored as 4 `float4` columns to avoid Slang's matrix-layout ambiguity. Struct offsets were checked with `spirv-dis` and are asserted at compile time in Rust. The per-frame chain:
  1. `cull.slang` (count pass): one thread per instance, world-space bounding sphere vs 6 normalized planes. A visible instance takes a rank in its mesh with `counts[mesh].increment()` and writes it to `slots[i]` (`INVALID` if culled or the mesh id is bad).
  2. `scan.slang` (prefix sum): `scanBlocks` does a 1024-element tile per 256-thread workgroup (4 per thread, Hillis–Steele in groupshared) and writes each tile total. The host recurses on the totals until one tile remains, then `addOffsets` propagates them back down (**reduce-then-scan / scan-then-propagate, multi-level**: 2 levels up to 1M elements, 3 up to 1G; wrapping u32). It produces `offsets` = exclusive scan of `counts`. In the frame it runs over the mesh count (up to `MAX_MESHES` = 65,536, 2 levels). `AshBackend::scan_u32()` runs the same kernels on arbitrary data for tests and benches.
  3. `scatter.slang`: one thread per instance, `visible[offsets[mesh] + slots[i]] = i`. `compact.slang` (indirect args): one thread per mesh; non-empty meshes append a `VkDrawIndexedIndirectCommand` with `firstInstance = offsets[mesh]` and bump the draw count. The two kernels write disjoint buffers, so there is no barrier between them.
  4. `mesh.slang`: vertex shader uses `SV_VulkanInstanceID` (= `gl_InstanceIndex`, which includes `firstInstance`; `SV_InstanceID` would not) to index `visible`, then Lambert shading.
  The CPU no longer builds a per-mesh histogram: `render()` only writes frame constants and uploads instances.
- `pass.rs` → `AshBackend` with **`BackendConfig::frames_in_flight` = 2 or 3** (default 2, env `BRUTAL_FRAMES_IN_FLIGHT`, anything else is `BadConfig`). Each **frame slot owns everything its frame touches on the GPU**: command pool, timestamp query pool, frame-constants + instance upload buffers, `counts`, `offsets`, scan scratch, `slots`, `visible`, `draws`, `draw_count`, stats readback, and its **own colour + depth target**. A slot is reused only after the host waits on the **timeline semaphore** for its previous frame (N − F). That wait is the only inter-frame synchronisation: there is **no full barrier at frame start any more**, and the attachment transitions use `srcStage = NONE`. So no barrier's first scope reaches into another in-flight frame, and frames N and N+1 have no GPU-side dependency (the mesh table and vertex/index buffers are read-only between `set_meshes` calls, which `wait_idle`). Intra-frame hazards each have a sync2 barrier: fill→count, count→scan, between scan levels, scan→scatter/args, compute→indirect/vertex/copy, colour→transfer, copy→host. `resize()` recreates every slot's targets. `read_color_rgba()` reads the latest frame's target. `FrameStats` adds `count_ms / scan_ms / scatter_ms` (`cull_ms` = whole compute chain), absolute `gpu_begin_ns / gpu_end_ns`, and `in_flight_at_submit / completed_at_submit` read from the timeline right after each submit. `drain()` retires all in-flight frames.
- `swapchain.rs` (feature **`windowed`**): surface from `raw-window-handle` 0.6, FIFO swapchain. The slot's offscreen target is blitted to the acquired image (pipelines never depend on the surface format). There's one acquire semaphore per slot (a `Vec` sized to F), one render-finished semaphore per image, and the swapchain is recreated on out-of-date/suboptimal results and on resize. In the `app`, `--features windowed` creates the backend once Bevy's winit window exists and forwards `WindowResized`.
- `cull.rs`: CPU reference with identical math. It classifies each instance as Inside / Outside / Ambiguous (within float tolerance of a plane). `cpu_exclusive_scan()` is the scan reference.
- `validation.rs`: a counting debug messenger. VK_LAYER_KHRONOS_validation + **synchronization validation** (VK_EXT_validation_features) are enabled in debug builds or with `BRUTAL_VK_VALIDATION=1`.

**Tests** (`cargo test -p render-ash`, validation + sync validation on): 8 GPU tests plus 1 messenger self-check plus 5 unit tests, all passing; `cargo clippy --workspace --all-targets [--all-features] -- -D warnings` is clean.
- **GPU scan == CPU scan**, exactly, for n = 0, 1, 2, 3, 7, 255, 1000, 1023, 1024, 1025, 3089, 65,537, 300,001, **1,048,576** (2 levels, 1024 workgroups), **1,048,577** (3 levels), plus 6 random sizes up to 200k. Values include occasional huge u32s, so the sums wrap and both sides must agree on overflow. A unit test pins the level plan.
- GPU visible count vs CPU reference over 8 randomized scenes (1 to 200k instances, 5% invalid mesh ids, random rotation and non-uniform scale): the GPU count always lies in `[inside, inside + ambiguous]` and matches exactly when nothing is boundary-ambiguous (3/8). In the others, at most 4 instances sit in the tolerance band. A camera looking away gives 0 visible and 0 draws.
- **Many meshes:** 3,000 meshes (> one scan tile, so the in-frame scan runs 2 levels) with 100k / 2.5k / 1 instances also match the CPU reference (3,000 and 1,314 non-empty draws).
- **Overlap proof** (`frames_overlap_and_stay_correct`, F = 2 and F = 3): frame 1 is gated on a host-controlled timeline semaphore, then frames 2..F are rendered. Each `render()` returns while the backend timeline is still 0, so frame N+1 (and N+2 at F = 3) was recorded and submitted before frame N retired, and nothing but slot reuse blocks. The gate is then released and 9 frames with **9 different scenes** are each checked against the CPU reference. Shared or clobbered scratch would fail this. The test also asserts that pacing never exceeds F in flight. Sync validation reports 0 errors over the whole run.
- Offscreen readback is non-empty (48,828 / 49,152 pixels lit). After resize the readback has the new size. Timestamps report each pass for every retired frame. Bad `frames_in_flight` (0, 1, 4) is rejected.
- **Validation: 0 errors, 0 warnings** in every test; in `--validate` stress runs (10k × 120 frames + 100k × 20 frames) at F = 2 and F = 3; and in 150-frame **windowed** runs under Xvfb at F = 2 and F = 3, each with a live resize to 960x540 (swapchain plus every slot's targets recreated). `tests/validation_selfcheck.rs` proves the messenger catches a deliberately invalid call, so the zero is meaningful.

**Perf** (`cargo run --release -p render-ash --example instanced_stress -- --png out/instanced.png [--fif 3]`), 1280x720. **Lavapipe = llvmpipe CPU rasterizer on a shared 8-vCPU box; not representative of a GPU.**

| instances | visible | fps before → after (F=2 / F=3) | GPU compute p50 before → after | after: count / scan / scatter+args | GPU draw p50 before → after |
|---|---|---|---|---|---|
| 10k | 5,280 | 91 → 93.5 / 95.6 | 0.29 → 0.58 ms | 0.25 / 0.08 / 0.24 ms | 10.3 → 9.4 ms |
| 100k | 49,503 | 16 → 15.8 / 17.3 | 2.0 → 2.3 ms | 1.95 / 0.11 / 0.28 ms | 60.3 → 58.6 ms |
| 1M | 478,761 | 1.9 → 2.1 / 2.1 | 17.1 → 18.6 ms | 15.1 / 0.11 / 3.4 ms | 463 → 445 ms |

How to read it: fps is unchanged within noise, because lavapipe spends more than 95% of each frame rasterizing on the same CPU cores. The compute chain is now 4 dispatches plus barriers instead of 2. Each dispatch has a fixed cost on llvmpipe, so it costs about +0.3 ms at 10k and +8% at 1M. The scan itself is about 0.1 ms. In return, the per-frame CPU O(N) histogram is gone (at 100k, command recording is ~25–40 µs and `vkQueueSubmit2` ~5–10 µs; the 8 MB instance memcpy wasn't timed separately), and the pass can overlap on hardware. Standalone `scan_u32` of 1M elements is 7–9 ms *including* host upload and readback (debug build); GPU time is not separately reported.
**Lavapipe caveat on overlap:** the bench reports `max_in_flight` = 2 at 10k and 1 at 100k/1M, with 0 GPU-timestamp overlaps. Lavapipe executes submissions in order on one device thread, and its timeline reports completions late, so it shows pipelining only when the CPU is ahead. The structural property (no cross-frame dependencies, sync-validation clean, gated test) holds on any driver. Actual GPU-side overlap needs a real GPU to measure, and `gpu_overlapping_pairs` in the JSON is there to show it. Rendered frame: `out/instanced.png`.

**Gaps / honest caveats:**
- Sync validation **cannot see buffer-device-address accesses** (it only tracks descriptor-bound resources). The BDA barriers (all scratch buffers) are written by hand and reviewed, and the gated multi-scene test checks per-slot isolation functionally, but the validator does not verify them. Image, attachment, indirect and copy hazards *are* checked; those are the per-slot targets.
- GPU overlap is demonstrated structurally, not as measured concurrency: lavapipe serializes. One queue only; no async-compute queue.
- The scan is a multi-pass reduce-then-scan (2 dispatches per level + 1 add pass), not single-pass decoupled-lookback. Round 3 added a subgroup tile scan (below); decoupled lookback is still open.
- The atomic rank in `cull.slang` makes the order *within* a mesh's region nondeterministic, and `compact.slang` appends draws in nondeterministic order. Counts and images are deterministic; draw-order-dependent effects (e.g. blending) would need a deterministic rank (a second scan over per-instance flags).
- Memory: everything is per slot, so at F = 3 and 1M max instances that's about 3 × (80 MB instances + 8 MB slots/visible + ~3 MB of mesh-sized buffers (counts, offsets, draws, stats) + 2 colour/depth targets). Instances are still read from host-visible memory (fine on lavapipe and ReBAR; staging would suit discrete GPUs better).
- No GPU budgets were measured on real hardware, and the prompt's "RTX 3060 / Steam Deck" budgets remain unverified. No golden-image comparison yet (only the non-empty check). No Tracy GPU zones. Windowed was verified on X11 (Xvfb + lavapipe) only.

#### Round 3 (20 Oct 2026): subgroup scan ✅

**Changes**
- `context.rs`: `SubgroupInfo` comes from `VkPhysicalDeviceVulkan11Properties` (subgroup size, supported stages and operations) and `Vulkan13Properties` (min/max size, `maxComputeWorkgroupSubgroups`, `requiredSubgroupSizeStages`). `subgroupSizeControl` and `computeFullSubgroups` are enabled at device creation when supported. It's printed on the device line of the GPU tests.
- `scan.slang`: a new `scanBlocksWave` entry point covers the same 1024-element tile (256 threads × 4).
  - Each thread does a local serial scan, then `WavePrefixSum`/`WaveActiveSum` across the subgroup, then subgroup totals go into groupshared and subgroup 0 scans them in chunks of `WaveGetLaneCount()` with a carry.
  - Element ownership uses the logical index `SubgroupId * lanes + lane` (SubgroupId read through `spirv_asm`). So nothing assumes how `LocalInvocationIndex` maps onto subgroups, and the code is **size-agnostic**: any power-of-two size that divides 256 works.
- `pass.rs`: `ScanMode { Auto, Subgroup, SharedMemory }` (`BackendConfig::scan`, env `BRUTAL_SCAN=auto|subgroup|shared`). `plan_subgroup_scan()` decides as follows:
  - It needs compute-stage `BASIC|ARITHMETIC` plus `computeFullSubgroups`.
  - With `subgroupSizeControl` and COMPUTE in `requiredSubgroupSizeStages`, it pins the largest valid power-of-two size via `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` + `REQUIRE_FULL_SUBGROUPS`.
  - Otherwise it uses `ALLOW_VARYING_SUBGROUP_SIZE` + full subgroups, as long as `256 % maxSubgroupSize == 0`.
  - Otherwise `Auto` falls back to the shared-memory kernel, and a forced `Subgroup` returns `BackendError::Unsupported`.
  - The shared-memory `scanBlocks` is always built. `AshBackend::scan_path()` reports the choice. `scan_u32_timed()` returns GPU time.
- Lavapipe selects `Subgroup { required_size: Some(8) }`.

**Tests:**
- GPU scan == CPU scan exactly for every prior size (0 … 1,048,577 plus 6 random sizes) on **both** paths.
- The 3,000-mesh cull matches the CPU on both paths.
- The full render-ash suite passes with `BRUTAL_SCAN=shared` too.
- A unit test covers the plan for pinned, varying, unsupported and non-dividing sizes.
- Validation + sync validation: 0 errors in all tests, and in windowed Xvfb runs (F = 3, live resize to 960x540) on both paths.
- Clippy is clean.

**Scan time, before → after** (`scan_bench`, GPU timestamps p50, release, lavapipe; *lavapipe runs subgroups as 8-wide SIMD on the CPU, so the ratio is not a GPU prediction*):

| n | shared (before) | subgroup (after) |
|---|---|---|
| 3,000 | 0.193 ms | 0.129 ms |
| 65,536 | 0.432 ms | 0.180 ms |
| 262,144 | 1.349 ms | 0.401 ms |
| 1,048,576 | 4.621 ms | 1.285 ms (~3.6x) |
| 4,194,304 | 12.868 ms | 4.518 ms |

In-frame (`instanced_stress --validate`, F = 2) scan_ms goes 0.109 → 0.069 (10k), 0.129 → 0.093 (100k), 0.371 → 0.100 (1M). The whole compute chain at 10k drops 0.70 → 0.57 ms. fps is raster-bound, so it barely moves (86 → 93 at 10k).

**Gaps:** no decoupled lookback (still `2·levels + 1` dispatches); not exercised on a device without subgroup arithmetic or with true varying sizes (only via the plan unit test); no real-GPU numbers.


### Prompt C, rewritten as C1–C4 (spec received 20 Oct 2026)

**Context:** the game has to show the universe, solar systems, Earth, ocean, mountains, underground, caves and underwater in detail, at ≥ 60 fps.

**Seam decision (settled by the user):**
- Far field: edge-snap.
- Near field: Transvoxel.
- Skirts only as a fallback for far chunks whose neighbours haven't streamed in yet.
- Ocean (D) and atmosphere (E) stay separate systems.

**Brick size:** 32³ @ 0.25 m is a **trial**. Both values are runtime config (`--brick`/`--voxel`); see the comparison table below.

The **ROBUSTNESS HARD RULE** is part of every C sub-prompt, verbatim:

```
HARD RULE (C1-C4):
 - All meshing and collision tests run on brick-local integer/quantised grid coordinates.
 - Snap vertices to the brick grid (or a fixed sub-grid) before any orientation or intersection test.
 - Exact predicates (integer or Shewchuk-style adaptive) or explicit tolerances; never raw f32 equality.
 - Shared seam vertices are computed from identical inputs on both sides -> bit-identical.
 - Acceptance includes: watertight across LOD transitions, no degenerate or flipped triangles,
   determinism, and a fuzz test on adversarial densities (near-zero values, ties at cell corners).
```

#### C1: far field, cube-sphere quadtree with edge snapping + CDLOD morph (spec; not implemented)

```
TASK: Planet far field: 6-face cube-sphere, per-face quadtree restricted to 2:1 between edge
neighbours; one shared N x N index buffer for every chunk; crack-free LOD via shader edge snapping;
CDLOD vertex morph so LOD changes never pop.
INPUTS:  coords frames (planet frame, f64 camera), planet radius, height source (terrain crate SIMD fBm,
         later baked/streamed tiles), chunk grid N = 65 (64 quads), max level ~ 20 (1 m quads at R = 6371 km).
OUTPUTS: quadtree selection per frame (CPU, f64 camera distance) -> chunk list; GPU: the cull pass
         (render-ash cull.slang) frustum/horizon-culls chunks AND writes a 4 x 2-bit neighbour-level
         mask per visible chunk; the vertex shader snaps odd edge vertices onto the coarser
         neighbour's edge (index-space: v_odd := average of its two even neighbours, from identical
         integer sample coords), then applies the CDLOD morph (morph factor from the f64->f32
         camera-relative distance, clamped to [0,1]); chunk vertices stored camera-relative to the
         chunk origin (planet frame f64 origin -> f32 offset).
BUDGET (60 fps, RTX 3060 @1440p / Steam Deck @800p): selection <= 0.15 ms CPU; cull+mask <= 0.1 ms GPU;
         far-field draw <= 1.5 / 3 ms; <= 2,000 visible chunks; 1 shared index buffer.
HARD RULE applied: quadtree keys are integers (face, level, i, j); restriction and neighbour lookups are
         integer; edge snapping is done on integer sample indices (both chunks evaluate the height at
         the same integer (face, level, i, j) sample -> bit-identical edge positions); the f32 position
         is formed once from the f64 chunk origin. Skirts: only for a chunk whose neighbour is not
         resident yet (flag in the mask), never as the seam strategy.
TESTS:  2:1 restriction invariant after random camera paths (property test); crack test: for every pair
         of edge-adjacent visible chunks, welded edge positions are identical (bit-exact, CPU mirror of
         the shader snap) at all 4 mask combinations; morph continuity: max vertex delta between
         consecutive frames bounded for a camera sweep; fuzz: random heights incl. exact ties/zeros
         at edge samples -> no degenerate/flipped triangles (exact i64 predicates on the quantised
         sample grid); determinism; validation + sync validation 0.
ACCEPTANCE: tests green, budgets met (timestamp JSON), scenes earth + mountains (far ring) use it,
         out/scenes/earth.png shows the curved horizon from 2 m to orbit with no cracks.
```

#### C2: near field, sparse voxel bricks + Transvoxel on the GPU (Slang) ✅ implemented (CPU + GPU)

```
TASK: Near-field sparse voxel bricks (SDF/density), meshed with Transvoxel ON THE GPU (Slang), watertight
LOD seams. Camera-centred LOD shells must work, including bricks with several coarser/finer faces,
edges and corners.
INPUTS:  DensityField (f64 planet-local -> metres, negative = solid), LodLayout (camera-centred
         ShellLayout; must expose a box-shaped finer region per LOD), VoxelConfig { BrickGrid { brick
         16|32|64, voxel_m }, render origin (sub-grid units) }.
OUTPUTS: per brick: quantised samples (i32, with apron) -> GPU count/scan/emit -> triangle soup with
         brick-local i32 sub-grid positions + f32 normals; CPU reference mesher with identical output;
         exact topology report.
BUDGET (60 fps): remesh of 16 bricks (32^3) per frame <= 1 ms GPU; sampling off the main thread (C4).
HARD RULE applied: H1-H8 below.
TESTS / ACCEPTANCE: A1-A7 below + GPU == CPU bit-exact + validation 0.
```

**Hard rule as implemented (H1–H8)**, in the `crates/voxel` module docs. These replace the reconstructed R1–R6.

| | Rule | Implementation |
|---|---|---|
| H1 | integer lattice | Brick-local and global lattice coordinates are `i64` (CPU) and `i32` brick-local (GPU). Vertices live on a fixed sub-grid of **256 per LOD0 voxel** (`SUB`). Global `ipos: [i64; 3]` is the truth. |
| H2 | quantised density | `round(d · 1024 / voxel_m)`, clamped to ±2²⁰, as `i32`. Inside iff `q < 0`, so exact 0 and −0.0 are outside. Non-finite input is an error. |
| H3 | snapped vertices | `edge_t = round(256·da/(da−db))`, clamped to [1, 255], in integers. Position = `a·256 + t·(b−a)`. No float before the final render cast. |
| H4 | exact predicates | Degenerate means the i128 cross product is 0. Flipped is the exact sign of the axis component on *axis-determined* triangles (see below), plus exact edge-orientation consistency. No float equality anywhere. |
| H5 | single narrowing | Render f32 = `((p − origin_sub) · voxel/256)` in f64, then f32 once. |
| H6 | identical seam inputs | A vertex's identity is its canonical lattice edge (lexicographic endpoints). Both sides read the same quantised samples and use the same integer `edge_t` and Φ, so shared vertices are bit-identical. |
| H7 | determinism | Same output for any brick order and thread count (tested with 1 thread vs the rayon pool). GPU == CPU bit-exact. |
| H8 | refuse what it can't join | LOD jump > 1 (26-neighbourhood) → `LodJump`. A finer region that isn't a box → `NotABox`. A finer box within one coarse cell of a face shared with a coarser brick → `PhiReachesCoarser`. |

R1–R6 → H1–H8: R1 → H2; R2 (f64 `t`) → H3 (integer `t`); R3 → H2; R4 → H7; R5 → H5. R6 was a *refusal* of edge/corner transitions; it is now **handled** by the global Φ field, and only the H8 cases are refused.

**Multi-face, edge and corner LOD transitions (shells).** Face-only Transvoxel cracks along brick edges and corners where 2–3 transition faces meet, and at bricks whose finer neighbour is only edge- or corner-adjacent. The fix is one global displacement field Φ.
- Φ is defined from the finer box R of the brick's LOD: a tent around each of R's planes, times a fall-off along the other two axes.
- It uses staged integer shifts `((t·g_b)>>log2s)·g_c >> (log2s+2)`, so the transition width is S/4. The corner Jacobian stays > 0, and every intermediate fits in i32 for the GPU.
- It is applied to every LOD-L vertex near R (face, edge-only and corner-only neighbours) and to half-res transition vertices. Full-res transition vertices are not displaced.
- Because Φ is a single function of the integer position, every brick that produces a vertex displaces it identically.
- `ShellLayout::camera_centred(grid, cam, half_bricks)` builds nested boxes aligned to the next LOD's bricks with ≥ 1 brick of margin.

**Exact "no flipped triangles" check.**
- A triangle whose 3 vertices lie on **parallel lattice edges with the same inside→outside direction** projects onto 3 corners of a cell square (Φ moves each by ≤ s/4 per axis). So its exact i128 normal component on that axis is non-zero, and its sign depends only on the tables, not on `t`.
- Orientation then propagates through `orientation_conflicts == 0`.
- Components with no such triangle (e.g. 8-triangle octahedra around an isolated sample in the fuzz) are certified by the exact edge-direction proxy: none < 0 and at least one > 0. An inverted component would flip every agreeing triangle. `unverified_components` must be 0.
- Why not signed volume: a correctly oriented cavity (air inside rock) has negative volume. The underground scene showed that, so volume is informational only.

**Acceptance status: all ✅**
- 13 voxel tests (`cargo test -p voxel`) and the GPU test `gpu_transvoxel_matches_cpu_exactly`.
- A1: closed 2-manifold at uniform LOD for 16/32/64.
- A2: slab seams on all 6 faces, LOD 0|1, 1|0, 1|2, 2|1.
- A3: camera-centred shells `[1]`, `[1,1]`, 4 camera positions, with face, edge-only and corner-only coarse-to-fine neighbours all present. Closed, ≥ 2 LODs carry surface.
- A4: negative controls. No transitions → cracks; no Φ → cracks; reversed winding → `flipped == axis_reference_triangles`; one reversed triangle → orientation conflict.
- A5: clean, and normals agree with the triangle orientation for > 99.5% of triangles.
- A6: determinism, including 1 thread vs many.
- **A7 fuzz:** 24 seeds × 5 layouts (uniform LOD0, uniform LOD1, slab, shells `[1]`, shells `[1,1]`) on the adversarial field. Hashed lattice values from {0, −0, ±1 quantum, ±0.4 quantum, ±1e3, ±1e-9, ±0.5, …}, i.e. near-zero values and ties at cell corners. Every run is closed, has 0 degenerate, 0 flipped and 0 unverified, and is deterministic; > 1M triangles in total. Uniform layouts also have 0 per-triangle edge-direction disagreements.
- H2/H3/H6/H8 unit tests: exact zero plane, `edge_t` table, NaN rejection, the three refusals plus "1-brick margin shells are accepted".

**GPU mesher** (`render-ash/shaders/voxel.slang` + `src/voxel_gpu.rs`, `AshBackend::mesh_voxels_gpu`):
- Pure i32 (no shaderInt64). Brick-local positions: ≤ 64 cells × 2⁶ step × 256 = 2²⁰. Φ intermediates ≤ s² ≤ 2²⁸; `edge_t` ≤ 2²⁹. The finer box is clamped to [−2E, 3E], which provably leaves Φ unchanged inside the brick.
- One thread per cell, in CPU order: regular cells `(k·n+j)·n+i`, then `n³ + f·n² + cv·n + cu`.
- `countMain` → the frame's shared multi-level exclusive scan (subgroup or shared-memory path), in place → `emitMain` writes non-indexed triangles at the scanned offsets. Tables are packed into one u32 buffer.
- **Result:** GPU positions are **bit-identical** to the CPU, in the same order. Normals match exactly (max |Δn| = 0) on uniform, slab, shells, fuzz-slab and fuzz-shells (up to 180k triangles). Topology of the GPU output alone is closed. 0 validation errors; both scan paths pass.
- On lavapipe (CPU-emulated GPU) the GPU mesher is *slower* than the rayon CPU mesher: 21–53 ms vs 5–15 ms per scene. Real-GPU numbers are still to be measured.

**Brick-size × spacing comparison** (v2 integer pipeline; `cargo run --release -p voxel --example voxel_bench`; cave scene region 32×18×28 m, uniform LOD0, single thread, this noisy box). 32³ @ 0.25 m stays the trial default.

| brick | voxel m | brick edge m | bricks sampled | occupied | sample µs/brick | mesh µs/occupied | tris total | tris/occupied | KiB density/brick | KiB mesh/occupied | total ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16³ | 0.125 | 2 | 2016 | 532 | 1614 | 104 | 298,844 | 562 | 26.8 | 14.0 | 3315 |
| 16³ | 0.25 | 4 | 280 | 124 | 1616 | 107 | 74,152 | 598 | 26.8 | 14.9 | 467 |
| 16³ | 0.5 | 8 | 48 | 36 | 1652 | 106 | 19,884 | 552 | 26.8 | 13.8 | 83 |
| 32³ | 0.125 | 4 | 280 | 124 | 10066 | 516 | 298,844 | 2410 | 167.5 | 58.2 | 2887 |
| 32³ | 0.25 | 8 | 48 | 36 | 10592 | 520 | 79,736 | 2215 | 167.5 | 53.7 | 528 |
| 32³ | 0.5 | 16 | 8 | 8 | 10043 | 519 | 19,884 | 2486 | 167.5 | 60.6 | 84 |
| 64³ | 0.125 | 8 | 48 | 36 | 72851 | 2840 | 321,152 | 8921 | 1174.9 | 212.5 | 3603 |
| 64³ | 0.25 | 16 | 8 | 8 | 72292 | 2947 | 79,736 | 9967 | 1174.9 | 238.3 | 602 |
| 64³ | 0.5 | 32 | 8 | 8 | 72303 | 2260 | 53,342 | 6668 | 1174.9 | 162.0 | 597 |

Compared with v1:
- Meshing is now **4–5× faster**. Integer `edge_t` and sample-gradient normals replace the 6 field evaluations per vertex.
- Density per brick grew by the apron, to (n+3)³.
- Sampling now dominates completely: ~0.25 µs per scalar f64 sample, so ~10 ms per 32³ brick on one core. **That cost is what has to move to SIMD/GPU (C4) before per-frame remeshing is possible.**

#### C3: altitude-based quadtree → voxel handoff with a morph band (spec; not implemented)

```
TASK: Hand the ground over from the C1 far field to C2 voxel bricks as the camera descends (and back), with a
morph band so neither popping nor double surfaces show.
INPUTS:  f64 camera altitude above the planet (planet frame), C1 chunk set, C2 ShellLayout, handoff
         altitudes h_in < h_out (hysteresis), morph band width.
OUTPUTS: per frame: which C1 chunks are suppressed (covered by resident voxel bricks), the voxel shell
         extent, and a per-vertex morph weight in the band (voxel surface morphs from the heightfield
         surface: the voxel density near the band = max(height SDF, voxel SDF) blended in integer
         quanta; the heightfield is lowered by the same amount so the two never z-fight).
BUDGET:  <= 0.05 ms CPU per frame; no frame with a hole while bricks stream (C1 chunk kept until all
         covering bricks are resident).
HARD RULE applied: the handoff boundary is a set of whole C1 quadtree cells == whole voxel bricks at a
         chosen LOD (integer alignment of the brick lattice to the quadtree grid on each cube face);
         the boundary ring is meshed from identical quantised heights on both sides (voxel density
         sampled from the same integer height samples); band weights are integer (quantised) so both
         sides compute the same seam; no f32 comparisons decide coverage.
TESTS:  descend/ascend sweeps: no frame with uncovered pixels in the band (coverage test on a CPU
         raster of the boundary), no double surface (depth-order test), seam ring watertight
         (welded topology like voxel::topology), hysteresis (no flicker at constant altitude),
         fuzz on adversarial heights at the boundary, determinism.
ACCEPTANCE: earth scene descends from orbit to 2 m with no pop or crack; mountains scene mixes far
         field + near voxels.
```

#### C4: streaming and budgets for 60 fps (spec; not implemented)

```
TASK: Keep C1/C2 within frame budget while moving: async generation amortised across frames, a GPU brick
pool, priorities.
INPUTS:  camera f64 + velocity, C1 selection, C2 ShellLayout (re-centred when the camera crosses a brick
         boundary), budgets.
OUTPUTS: job system (rayon/worker threads): sample + quantise bricks off the main thread (SIMD density:
         f64x4/f32x8 or a GPU density pass); GPU brick pool = fixed-size slots (samples + mesh output),
         reusable staging ring (render-ash StagingRing), slot LRU retired by timeline value (as
         TerrainGpu); GPU meshing batched per frame (mesh_voxels_gpu without host round-trip:
         indirect draw from the pool); priority = distance/screen error, deadline-aware.
BUDGET (60 fps, 16.6 ms): main thread <= 0.3 ms for streaming bookkeeping; <= 16 brick meshes (32^3)
         + 8 C1 chunks uploaded per frame; GPU meshing <= 1 ms/frame; pool memory fixed (e.g. 256 MiB);
         0 allocations per frame after warm-up (counting allocator).
HARD RULE applied: a brick is drawn only when its own mesh and the meshes of every neighbour it shares
         a seam with are generated from the same layout epoch (otherwise keep the old set, or skirt
         the far field only); re-centring changes the layout only at brick-aligned positions (integer);
         results deterministic regardless of job completion order (tested by shuffled completion).
TESTS:  flythroughs of all 8 scenes: p99 frame time <= 16.6 ms on the reference GPU; no cracked frame
         (topology of the drawn set each frame); pool never evicts a slot still in flight (timeline);
         shuffled-completion determinism; budget JSON.
ACCEPTANCE: the eight scenes at >= 60 fps (p99) on the reference GPU with C1-C4 + D/E enabled.
```

#### C0 (earlier round, 20 Oct 2026): single-LOD heightmap, SIMD + staging ring ✅

The original Prompt C (single-LOD part) is kept as the C1/C4 groundwork:
- SIMD fBm heightmap: chunk_full is 0.055 ms per LOD0 chunk with x86-64-v3, against a 0.12 ms budget.
- `StagingRing` (32 MiB, timeline-retired).
- `TerrainGpu` slots plus LRU eviction gated by the timeline value.
- A shared index buffer.
- `terrain_flyover` example.
- `SeamStrategy` now documents the decided strategy; the only impl is still `SingleLod`.

Measured: upload p50 0.011 ms / p99 0.071 ms; flyover at 45.6 fps on lavapipe. Validation is clean.

Gaps:
- "16 chunks/frame on the pool ≤ 0.5 ms" is not measured.
- There is no counting allocator.
- There is no transfer queue (lavapipe).
- Normals are not oct-encoded.

### Coords (`crates/coords`): f64 frame hierarchy ✅ (spec received 20 Oct 2026)

Decision: **f64 (DVec3), not fixed-point**, in a nested translation-only frame tree: galaxy > system > planet > local. The brick lattice (`BrickGrid`, i64 lattice/brick coordinates, `floor_div`/`floor_mod`) is kept as the **planet-local voxel layer**.

| Level | Max extent (enforced by `FrameTree::add` + `influence_m`) | Documented precision = ½ ulp(extent) |
|---|---|---|
| Galaxy | 1e21 m | 6.6e4 m |
| System | 1e13 m | **9.8e-4 m (~mm)** |
| Planet | 1e7 m | **9.3e-10 m (~nm)** |
| Local | 1e4 m | 9.1e-13 m |

API:
- `FrameTree` operations:
  - `add(parent, name, origin, influence)` validates the extent; influence must be ≤ the child level's extent;
  - `lca`;
  - `convert`;
  - `delta(a, b)` = a − b, formed at the lowest common ancestor;
  - `render_relative(x, cam)`: **f64 subtraction, then one f32 narrowing**;
  - `rehome`: ascends out of frames it has left, then descends into the deepest child whose influence contains it.
- `FloatingOrigin { threshold_m }`: `update` rebases when the camera is more than the threshold away, or changes frame. `local_f32` gives positions relative to that origin.

Bevy plugin (`bridge::floating`):
- `FloatingOriginPlugin { threshold_m }` (default 2048 m) adds `rebase_origin` + `sync_transforms` to PostUpdate, before transform propagation.
- Components `WorldPosition(WorldPos)` and `FloatingCamera`; resources `Universe(FrameTree)` and `FloatingState`.
- `gather_instances` writes **camera-relative** instance translations to render-ash: f64 `render_relative` for entities with a `WorldPosition`, else origin-relative f32 minus the camera.
- The plugin is opt-in; `app` doesn't enable it yet, because doing so also needs a camera-relative view matrix there.

Tests: 11 coords + 2 bridge. Every comparison uses an explicit tolerance; nothing is ULP-fragile.
- Documented precision per level.
- Precision of two points on the same planet.
- Frame-to-frame round trips (local ↔ planet ↔ system ↔ galaxy).
- `render_relative` precision.
- Rehome up and down.
- Extent and influence refusals.
- Rebasing over a ~17,000 km path: rebased f32 error < 1e-3 m vs naive f32 > 0.1 m (non-integer offsets, so rounding doesn't cancel).
- Bevy integration (CaptureBackend): ≥ 1000 rebases, rider error < 1e-4 m, instances camera-relative.

Flythrough results (scenes below): rider = a probe 3.2 m from the camera, stored in the camera's frame.

| Scene | Frames visited | Rider error (f64 path + final f32) | Bevy floating-origin f32 path | Naive f32 (root frame) | Rehome round trip (err / 4-ulp tol) |
|---|---|---|---|---|---|
| universe | milky_way → sol | galaxy 3.2 m (≤ 6.6e4 documented: the probe collapses onto the camera); system 4.1e-4 m (≤ 9.8e-4) | system 4.1e-4 m | 3.2 m (collapses) | 0.06 |
| solar_system | sol → earth | system 3.8e-4 m; planet 1.2e-7 m | ≤ 3.8e-4 m | system 1.0e3 m (1.9e5 px) | 0.13 |
| earth | sol → earth → site | system 1.3e-5 m; planet 1.2e-7 m; local 1.4e-7 m | local 1.0e-4 m | 3.2 m | 0.13 |

Jitter below galaxy level, f64 path: **≤ 0.077 px**, worst at system level, i.e. the documented ~mm at 1e12 m on an object 3 m away. Earth descent: 0.0025 px. The naive f32 path jitters 600–190,000 px.

Implication: an object a few metres from a camera that sits in a *system* or *galaxy* frame jitters at the documented mm (or 66 km) level. Spaceships and cockpits need a **local frame that moves with the ship**: `FrameTree::set_origin` exists for this, and rehome/rebase handle the rest.

### Prompt D: Ocean (spec only; not implemented)

```
TASK: FFT ocean on a spherical planet: Phillips/JONSWAP spectrum -> GPU FFT (Slang compute) of height +
choppy displacement + slope, rendered on a camera-relative clipmap (or projected grid) wrapped onto the
sphere, with shoreline interaction against voxel terrain and an underwater post-process.

INPUTS:  coords frames (planet centre, radius R, camera f64 world pos; render origin = camera), wind (speed,
         dir, fetch), 3 cascades (e.g. 250 m / 37 m / 5 m tiles), N = 256 per cascade, time t (f64 -> wrapped f32).
         Shore: signed distance to terrain (baked per voxel brick / terrain chunk: bathymetry depth + SDF texture).
OUTPUTS: per-cascade displacement (RGBA16F) + slope/foam (RGBA16F) storage images, mipped; ocean surface draw
         into the HDR colour + depth; underwater mask; foam/jacobian; CPU query API wave_height(world_pos)
         (from a 2-frame-old readback) for buoyancy.

BUDGET (60 fps, RTX 3060 @1440p / Steam Deck @800p): spectrum+FFT 3x256^2 <= 0.35 / 0.9 ms; surface draw
         <= 0.8 / 1.6 ms; underwater post (fog + caustics + god rays approx) <= 0.5 / 1.0 ms; CPU <= 0.1 ms
         (record + query). Memory <= 24 MiB.
ROBUSTNESS: R-D1 no precision loss far from origin: vertex positions = camera-relative, sphere curvature
         applied as f32 offset from the local tangent frame (no f32 absolute planet coords). R-D2 time
         wraps per-cascade on the spectrum period (no phase drift after hours). R-D3 clipmap rings are
         seam-free (snap to cascade texel grid, morph between rings, shared-edge test). R-D4 no NaN for
         any wind 0..40 m/s (wind=0 -> flat). R-D5 deterministic for a seed (CPU reference FFT matches).
         R-D6 horizon: grid never shows a gap at the limb from orbit (fallback to sphere impostor).
TESTS: GPU FFT == CPU reference FFT (rustfft) within 1e-4 rel; Parseval energy check; spectrum
         symmetry (real output); clipmap watertightness (welded mesh, like voxel topology()); camera
         at 10,000 km from origin renders the same image as at origin (golden diff); shoreline: foam/depth
         attenuation = 0 at depth >= threshold, waves damped to <=10% at depth 0; wave_height() vs GPU within 2 cm;
         validation + sync validation 0.
ACCEPTANCE: budgets met (timestamp JSON), all tests green, out/ocean.png (shore + open sea) and
         out/underwater.png, zero allocations per frame after warm-up.
SCENES (see "The eight scenes"): makes `ocean` and `underwater` (surface from below + underwater post)
         real; adds the ocean to `earth` (orbit -> shore) and `solar_system` (orbit LOD); shore/pools in
         `cave` and `mountains`.
```

### Prompt E: Atmosphere (spec only; not implemented)

```
TASK: Physically based atmosphere (Hillaire 2020 LUT pipeline, Bruneton-compatible parameters): transmittance
LUT, multiple-scattering LUT, sky-view LUT, aerial-perspective froxel volume; works from ground to space with
planet shadow; integrated with coords frames.

INPUTS:  planet R_ground, R_top, Rayleigh/Mie/ozone coefficients + scale heights, Mie g, sun dir + illuminance,
         camera f64 world position (altitude computed in f64 from planet centre, R-E1), depth buffer, HDR colour.
OUTPUTS: transmittance LUT 256x64, multi-scatter 32x32, sky-view 192x108 (per frame), aerial perspective
         32x32x32 froxels (per frame), sky + aerial-perspective applied to HDR colour; sun transmittance for
         lighting (feeds terrain/voxel/ocean shading); from space: atmosphere rim composited over the planet.

BUDGET (60 fps, RTX 3060 @1440p / Steam Deck @800p): transmittance+multi-scatter only on parameter change
         (<= 1 ms once); per frame sky-view + froxels <= 0.25 / 0.6 ms; apply pass <= 0.3 / 0.7 ms. CPU <= 0.05 ms.
ROBUSTNESS: R-E1 altitude/view-zenith from f64 camera relative to planet centre (no f32 cancellation at
         R ~ 6.36e6 m); R-E2 ray/sphere intersections use the stable quadratic form, never NaN at the horizon
         or inside the ground sphere (clamp altitude >= 0); R-E3 continuous transition ground <-> space (sky-view
         LUT parameterisation switches to ray-marching beyond R_top without a pop: test luminance continuity);
         R-E4 planet shadow: ray segments in Earth shadow get no single scattering (sun occluded by ground sphere);
         R-E5 energy: transmittance in [0,1], monotonic with path length.
TESTS: LUTs vs CPU reference integrator within 1% (rel) at sample points; transmittance(top, any dir)=1 beyond
         atmosphere; continuity sweep altitude 0 -> 1000 km (max frame-to-frame luminance delta bounded);
         sunset at ground has red-shifted sky near sun (ratio check); twilight: shadowed segment in-scatter = 0
         test; froxel aerial perspective matches ray-marched reference at 5 depths; validation + sync validation 0.
ACCEPTANCE: budgets met, tests green, PNGs: out/sky_noon.png, out/sky_sunset.png, out/orbit.png (limb +
         planet shadow), shading of terrain/voxels/ocean uses the sun transmittance.
SCENES (see "The eight scenes"): makes the sky + aerial perspective real in `earth` (ground -> space,
         continuity R-E3), `ocean`, `mountains` (distant ridges) and the limb/planet shadow in
         `solar_system`; sun transmittance for shading in `cave` (mouth), `underwater` and all voxel
         scenes; no effect on `universe`/`underground`.
```

### The eight scenes (`examples/scenes.rs`) ✅ runner; real vs placeholder marked

```bash
cargo run --release -p render-ash --example scenes -- --scene <universe|solar_system|earth|ocean|mountains|underground|cave|underwater|all> \
    [--frames 120] [--size 960x540] [--brick 32] [--voxel 0.25] [--validate]
```
Each scene:
- prints one JSON line (also written to `out/scenes/<name>.json`; `all.jsonl` holds the last full run) with frame time p50/p99, GPU ms (cull chain + draw, timestamp queries), triangles, memory and named correctness checks, plus `real` and `placeholder` lists;
- writes `out/scenes/<name>.png`;
- exits non-zero if any check fails.

Measured with `--scene all --validate --frames 120` on lavapipe (CPU rasteriser, 960×540, this shared 8-vCPU box). **These are not GPU numbers.** The 60 fps target is for real GPUs; the voxel scenes here are raster-bound on llvmpipe at 90–350k triangles.

| Scene | Real | Placeholder | frame ms p50 / p99 | GPU ms p50 / p99 | triangles | mem MB | checks | val. errors |
|---|---|---|---|---|---|---|---|---|
| `universe` | coords flythrough galaxy → system (f64 frame tree, rehome, floating origin, camera-relative instancing, GPU cull/indirect) | 6,000 stars, the galactic core and the sun are instanced impostors in log-compressed scaled space with a minimum angular size; no atmosphere/lighting | 2.48 / 6.61 | 2.19 / 7.19 | 62,092 | 1.7 | 8/8 ✅ | 0 |
| `solar_system` | coords flythrough system → planet (8 planets at real orbital radii, Earth frame, moon) | sun/planets/moon are impostor spheres (minimum angular size) | 1.29 / 2.82 | 1.04 / 4.43 | 26,092 | 1.4 | 9/9 ✅ | 0 |
| `earth` | coords flythrough system → planet → local, from 3e7 m down to 2 m above a landing site | **surface: flat 200 km tile + boxes below 30 km** (this flythrough is unchanged; the orbital Earth is the separate `earth` example, see *Earth round 1*); no ocean (D) or atmosphere (E) here | 2.06 / 4.52 | 1.75 / 4.77 | 26,392 | 1.4 | 11/11 ✅ | 0 |
| `ocean` | camera-relative instancing in the planet-local frame | **OCEAN SURFACE: flat opaque slab** (D); buoys bob on a sine; no sky (E) | 1.02 / 2.12 | 0.78 / 4.57 | 332 | 0.0 | 2/2 ✅ | 0 |
| `mountains` | **real C2 mesher**: ridged terrain + arch, shells [1,1,1] → LOD 0–3 (27/26/60/30 occupied bricks), GPU Slang Transvoxel bit-exact vs CPU, exact topology | no far field beyond the voxel region (C1/C3); static world (no C4 streaming); no atmosphere/aerial perspective (E) | 13.78 / 24.21 | 13.30 / 26.71 | 349,326 | 177.1 | 7/7 ✅ | 0 |
| `underground` | **real C2 mesher**: tunnels, two caverns, a pillar; shells → LOD 0–2 carry surface | unshadowed sun + hemispheric ambient (no headlamp/GI); static world | 7.53 / 9.80 | 7.11 / 9.68 | 90,126 | 71.6 | 7/7 ✅ | 0 |
| `cave` | **real C2 mesher**: cave scene (tunnel, cavern, overhang lip), shells [1,1] → LOD 0–2 | as underground | 8.69 / 11.60 | 8.30 / 11.19 | 163,260 | 97.3 | 7/7 ✅ | 0 |
| `underwater` | **real C2 mesher** for the seabed (dunes, rock arch, boulders), shells → LOD 0–3 | **UNDERWATER VOLUME: none** (no fog/absorption/caustics/god rays); **water surface: opaque slab** overhead (D/E) | 11.46 / 16.06 | 11.01 / 15.12 | 239,040 | 164.7 | 7/7 ✅ | 0 |

Voxel-scene checks, all passing:
- no layout refusals (H8);
- **GPU Transvoxel == CPU (bit-exact, same order)**;
- **interior watertight**: every open edge lies exactly (in integers) on the region cut. The region is grown so the outermost shell sits ≥ 1 coarse brick inside it;
- 0 non-manifold edges and 0 orientation conflicts;
- 0 degenerate, 0 flipped, 0 unverified, 0 non-finite;
- ≥ 2 LODs carry surface.

Coords-scene checks, all passing:
- rider error within the documented precision per level (tolerance: 2 × precision + |offset|·2⁻²³);
- Bevy floating-origin path ≤ 1e-3 m;
- the expected frame levels are visited;
- rehome/convert round trip ≤ 4 ulp of the path magnitude;
- jitter < 0.5 px below galaxy level;
- naive f32 measurably worse (control).

Per-scene voxel metrics are in the JSON: bricks planned/sampled/occupied, per-LOD counts, sampling ms, GPU mesh ms, CPU reference mesh ms, full topology. For example, mountains: 426 planned / 143 occupied bricks, 349k tris; sampling 645 ms on rayon, the one-off cost C4 must amortise; GPU mesh 53 ms on lavapipe.

**D/E → scenes they make real**

| Scene | Prompt D (ocean) makes real | Prompt E (atmosphere) makes real |
|---|---|---|
| universe | — | — (galaxy-scale impostors stay; E has no effect beyond planets) |
| solar_system | ocean albedo/specular on the planet impostor (orbit LOD of D) | planet limb + rim from orbit, planet shadow |
| earth | ocean surface from orbit to the shore (R-D6 horizon, sphere impostor fallback) | sky from ground to space, aerial perspective, sun transmittance for all shading |
| ocean | **the whole scene**: FFT cascades, foam, shoreline, buoyancy queries replace the slab | sky, sun/sky lighting on the water, horizon haze |
| mountains | shoreline where valleys flood (if sea level intersects) | aerial perspective on the distant ridges, sunset lighting |
| underground | — | — (only the cave-mouth light) |
| cave | water in the cave mouth/pools (shore SDF against voxels) | light through the cave mouth (sun transmittance) |
| underwater | **the water surface from below + the underwater post** (fog, absorption, caustics, god rays) replace the slab | sun transmittance into the water |

### Quality pass: cave + mountains (`examples/quality.rs`, `shaders/quality.slang`) ✅ 20 Oct 2026

Renders 3840x2160 and downsamples 2x2 in linear space to **1920x1080** (`out/quality/{cave,mountains}.png`,
stats in the matching `.json`). Validation: **0 errors, 0 warnings** on both. Shading in Slang: GGX
PBR-ish materials by slope / height / curvature, triplanar textures with detail normals (amplitudes in
metres), per-vertex multi-scale SDF AO + cavity/convexity, reverse-Z infinite projection, ACES.

| Scene | Triangles | Build (CPU) | GPU frame @ 3840x2160, lavapipe | Notes |
|---|---|---|---|---|
| cave | 183k (voxel) | 2.8 s | **~218 ms** (shadow 18, main 198) | headlamp spot + its shadow map, lantern with SDF soft visibility, fog |
| mountains | 4.18M (1.33M voxel + 2.85M rings) | 18 s (attributes 11.5 s) | **~1.0 s** (shadow 0.67 s, main 0.33 s) | 4 sun cascades to 9 km, sky, height fog/aerial perspective |

Lavapipe numbers are CPU rasterisation (8 cores); they say nothing about a real GPU.

**Fields** (`crates/voxel/src/fields.rs`, noise in `noise.rs`): `Alpine` (domain-warped ridged multifractal
with crest modulation and slope-magnitude damping, uplift massif mask, slope-aligned erosion gullies, strata
ledges and overhang bulges on steep faces) and `CaveNetwork` (spline main tunnel + chamber, spaghetti worms,
flutes, strata, stalactites/stalagmites). All band-limited by the sampling cell (`density_lod`).
Near field: voxel shells (32-voxel bricks, 0.5 m mountains / 0.0625 m cave) meshed on the GPU; mountains
far field: heightfield rings (cell = 0.004 x radius) to 26 km, culled to the view wedge.

**Bugs, root causes, fixes, tests** (`crates/voxel/tests/quality.rs`):
- *Sawtooth on ridges:* marching-cubes aliasing of the C0 ridged crease at coarse cells. Fix: the crease
  is softened to >= 3 cells of the sampling LOD (`crease_width`). Test q1: crest zigzag 0.338 m -> 0.039 m.
- *Cracks from per-brick filtering:* the filter level is now a pure function of the lattice point
  (`filter_level`) or continuous (`smooth_filter_cell`). Tests q2/q4 (watertight), q2b (negative control).
- *Floating blobs:* union-find island removal over shared edges (`remove_islands`). Test q3.
- *Mountains "clipped flat at a constant height" (long wall with a straight top):* the clamp suspects
  (region top, brick skip) were real latent risks but not the cause; `QUALITY_DIAG` showed mesh max == field
  max per azimuth. The root cause was the **field**: ridged noise `1 - |n|` puts every crest of an octave at
  the same height (r = 1 along the whole zero contour of n), and with the 5 km octave dominant (higher octaves
  over-damped: damping factor ~0.09 by octave 5) every range was a long flat-topped wall. A massif mask
  multiplying relief added a constant-height escarpment rim. Fixes: crest modulation of the first three
  octaves by independent smooth noise (peaks + saddles), weaker damping (0.15), gain 0.56, uplift instead
  of relief-mask multiplication. Clamp hardening: the region spans `Alpine::height_bounds()` (rigorous),
  and the brick skip is the conservative `Alpine::may_cross` (Lipschitz + overhang bound, both filter ends).
  **Test q5**: around the highest summit of an 18 km square, the meshed top matches the band-limited field
  summit (2105.3 vs 2105.6 m), every height lies inside `height_bounds`, and all 636 skipped bricks are
  sampled anyway and have no sign change.
- *Hard vertical lighting seam half-way across the wall:* not the voxel/ring boundary (the ring debug
  view showed the wall was ring geometry). It was the last cascade (2.7–26 km, ~15 m texels) + a
  grazing-angle terminator: the sun-facing half was unshadowed, the grazing half got coarse self-shadow
  acne. Fixes: splits 120 / 600 / 2500 / 9000 m, cascades blended over the last 20% of each split,
  shadows fade out beyond the last split (fog dominates there), normal offset grows at grazing angles.
- *Vertical smeared streaks on steep faces:* octaves meshed with only 2–4 samples per wavelength
  stair-step into stripes. Alpine now fades octaves over 3–6 cells (`octave_weight_wide`; the cave keeps
  the original 2–4 fade). Cliffs get real triplanar rock with dipping, warped strata at 24 m and 6.5 m
  (albedo bands + ledge detail normal), slope-driven rock/scree/grass/snow blend, snow above ~1500 m
  mostly on flatter ground.
- *Cave lantern light through rock:* per-vertex SDF sphere-traced soft visibility (`attr.w`).
- The cave render is pinned: it uses the per-shell `Filter::Level` path it was approved with
  (pixel-identical after the mountains work); the mountains use `Filter::Smooth`.

#### Quality pass round 2 (21 Oct 2026): progress log (newest last; images in `out/quality/steps/`)
- `00_start.png`: round-1 result.
- Step 1, distant detail (`01a_detail`, `01b_gullies`): shader sub-mesh "eroded" relief normals (octaves the
  mesh cannot represent, lambda < ~4 mesh cells, and the pixel can resolve), macro albedo fields (610/190/63 m),
  strata at 80/24/6.5 m, lichen, scree fans, meadow moisture/dryness and soft shrub patches, materials picked by
  the *effective* (relief-perturbed) slope. Field: 420 m and 170 m gully octaves, domain-warped kernels,
  direction jitter and incision modulation (no corduroy). `GULLY_MAX` derived from the octave table; q5 green.
- Step 2, blue streak (`02a_ambient`, `02b_albedo`): root cause was the shadow ambient = zenith blue x 1.05
  (B/R ~ 5) with no ground bounce, so high-albedo scree/rock in sun shadow went blue-white. Now hemisphere sky
  irradiance (B/R ~ 2) + green-brown bounce from the sunlit valley on faces that see the ground; rock/scree
  albedo lowered to realistic 0.17-0.33.

- Step 3, trees + ground cover (`03a_trees`, `03b_trees`): `examples/quality_parts/veg.rs` builds procedural
  conifers (LOD0 12 whorls x 10 branch tips + underside + trunk, LOD1 6 whorls, LOD2 2-cone stand-in for an
  impostor), boulders and grass tufts; placement by treeline (~1080-1290 m with noise), slope, forest clusters
  and valley-floor meadows. ~83k instances / 2.0M triangles. GPU path: the instanced pass's own kernels
  (`cull.slang` frustum cull + count -> scan -> `scatter.slang` + `compact.slang` indirect args) via the new
  crate-internal `AshBackend::record_cull_chain`, then `vkCmdDrawIndexedIndirectCount` in the main pass and
  every sun cascade (`instVS` / `instShadowVS`, lit by the same `terrainFS` with foliage translucency).
  LOD is chosen at placement by distance (still frame); impostors are not implemented.
- Step 4, atmosphere (`04a_atmo`, `04b_mie8`, `04c_mie3_fixed`, `04d_mie10`, `04e_final`): Hillaire 2020
  physically based sky in `quality.slang` compute (`atmo*CS`, run each frame into one BDA buffer after the
  cull chain): transmittance LUT 256x64, multiple-scattering LUT 32x32, sky-view LUT 192x108 (non-linear
  horizon mapping), aerial-perspective froxels 32^3 out to 40 km, plus a sun/sky irradiance reduction.
  Rayleigh + Mie (Cornette-Shanks g=0.8, absorption 0.444e-3/km) + ozone, units km, RG 6360. Terrain gets sun
  x transmittance, sky irradiance from the LUT, and `col*T + inscatter`; the sky is the sky-view LUT, a
  limb-darkened sun disk and cirrus at 6 km lit by the LUT. Knobs: `MIE` (default 5 = mildly hazy alpine day),
  `EXPOSURE` (0.42), `ATMO=0` (round-1 analytic haze), `QUALITY_DEBUG_AP=1` (r = T, g = inscatter).
  Note: sunlit distant rock is about as bright as the in-scatter, so haze reads mostly as contrast/colour loss,
  not brightening; MIE 3 vs 10 differ only subtly (physically consistent). Cave does not use it (pixel-identical).
- Round 2 gates: clippy -D warnings (default + all-features) clean, 67 tests pass (also `BRUTAL_SCAN=shared`),
  validation 0 (mountains + cave), SPIR-V fallback regenerated and pixel-identical to slangc output.
  Lavapipe 4K GPU: mountains ~1.78 s (shadow ~1.0 s with instanced casters in 4 cascades, main ~0.78 s),
  build ~23 s; cave ~0.27 s. These are CPU-rasteriser numbers, not real-GPU fps.
- (Superseded by round 3 below.)

#### Quality pass round 3 (21 Oct 2026): sky/haze, rock faces, ecological trees (`steps/05*`-`07*`)
- Before: `steps/04e_final.png`. After: `out/quality/mountains.png` (= `steps/07g_final.png`).
- Step 5, sky + haze (`05a_exposure_haze`, `05b_sky_final`): the LUT physics was checked numerically
  (python replica: horizon in-scatter ~0.05 / unit sun illuminance, B/R ~1.2-2.7; AP T ~0.8 / 0.6 at
  12 / 20 km; km units consistent: terrain metres x 0.001, camera 0.67 km). The over-saturated deep-blue
  sky was exposure: 0.42 put the whole frame in the ACES toe (0.01 -> 0.004 vs 0.1 -> 0.126), crushing
  red relative to blue and hiding the in-scatter. Now `EXPOSURE` 1.1 and `MIE` 8 (~60 km visibility):
  pale horizon band, zenith-to-horizon gradient, successive ridges lighter/bluer/lower contrast.
  The AP froxels already cover 40 km (12 km sits at slice ~17.5 of 32).
- Step 6, rock faces (`06a`-`06d`): `buttress()` / `buttress_tri()`: ridged, fall-line-stretched (x2.4),
  domain-warped relief at 110/48/21/9/4 m on the x/z projections, footprint-faded only (no mesh-cell fade,
  so mid-distance faces get it), gated to steep rocky ground (`steep_face`). Strong normal tilt (0.42),
  groove/crest albedo, water/varnish streaks down the fall line (17 m / 6.5 m wide), and a new
  `Mat.occ` cavity occlusion on sky + bounce light (cave/instanced set occ = 1: cave pixel-identical).
- Step 7, trees (`07a_ecology`, `07b_groups`, `07c_bounce`, `07d`/`07e_outcrop`, `07f_cloudshadow`):
  `veg::Surface` bins the actual render mesh (voxel near field + far rings) on a multi-level XZ grid;
  every tree / boulder / tuft is dropped onto the highest drawn triangle (trunks sunk by radius x slope),
  no more floating/sinking from field-vs-mesh differences. Density is ecological: local slope (gullies
  included, none past ~35 deg), regional slope + locally gentle = benches/ledges, concavity at 25 m
  (gullies denser, treeline climbs; spurs exposed), ragged treeline (1.1 km / 260 m noise) with a
  krummholz belt (trees shrink, survive only in 35 m clumps), hard ceiling at 1330 m (shader grass
  limit), and a contrast curve so stands are groups or open ground (no sprinkle of singles). Evaluated
  in parallel (rayon): veg build ~2.0 s incl. mesh binning. Shader: physical ground bounce (valley
  albedo x (sun + sky) x ground view factor), forest-floor outcrops (< ~1000 m) mossy/darker with no
  bright talus. Cloud shadows: the sky's 6 km cirrus field sampled along the sun ray (0.3 max).
- Gates: clippy -D warnings (default + all-features) clean; 67 tests pass (also `BRUTAL_SCAN=shared`);
  validation 0 (mountains + cave); cave pixel-identical; SPIR-V fallback regenerated, pixel-identical.
  Lavapipe 4K GPU: mountains ~1.6-1.9 s (shadow ~0.9 s, main ~0.75 s; ~80k instances, 1.8M veg tris);
  cave ~0.22 s. CPU-rasteriser numbers, not real-GPU fps.
- Known weaknesses: zenith still fairly saturated (ACES per-channel); faces in full shade show
  structure mostly via occlusion (no directional sky); foreground outcrop geometry is blobby voxel
  rock; small shaded rock facets read blue-grey; cloud shadows are broad (cirrus); no tree impostors
  (distant LOD2 cones in every cascade dominate the shadow cost); LOD fixed at placement.

**Mountains camera:** default vista (7700, -4200), 40 m above the valley floor (~673 m), yaw 2.75,
pitch 0.08, chosen by a skyline-scored search (peaks counted, distance spread 2.5–12 km, clear
foreground). Override with `CAM_X/CAM_Z/CAM_H/CAM_YAW/CAM_PITCH/SUN_REL_YAW/SUN_EL`. Debug:
`QUALITY_DIAG=1` (field vs mesh max elevation per azimuth), `QUALITY_DEBUG_RING=1` (tints ring
geometry), `QUALITY_DEBUG_SHADOW=1` (cascade index x shadow term).

**Honest assessment:** the cave holds up well. The mountains now read as a range: several peaks at
different distances, a valley, cast shadows, snowcaps and aerial perspective, with no flat top and no
seam. Still short of photoreal: the sun-facing slopes at 3–5 km are smooth and slightly plasticky (ring
geometry at 12–20 m cells, no texture detail at that footprint); the gully pattern is regular in places;
grass is a flat green carpet with no trees or vegetation scale cues; the shaded foreground outcrops go blue
under the sky ambient; no clouds' shadows, no multiple-scattering atmosphere. One frame takes ~1 s GPU +
18 s build on lavapipe: it's an offline still, not real time.

### Mountains round 4: forest edges, jagged silhouettes, jointed outcrops ✅ (21 Oct 2026)

Step images are in `out/quality/steps/`:
- before: `07g_final.png`;
- forest: `08a`–`08f_forest`;
- crags: `09a`–`09d_crags`;
- outcrop: `10a`–`10h_outcrop`;
- vegetation via the asset format: `11a_vegasset`;
- final: `11_final.png`, the same file as `out/quality/mountains.png`.

**Forest edges (`quality_parts/veg.rs`, `quality.slang`).**
- Stand boundaries are domain-warped (110 m at 300 m) and frayed by 60 m and 22 m octaves.
- Clearings at 120 m and tree-fall gaps at 30 m.
- Avalanche tracks: gullies on steep regional slopes, kept clear down the fall line.
- Saplings at edges and in tracks, rare outliers in the meadows, and an alder/mugo shrub fringe
  to 650 m.
- Interior trees are taller and edge trees wider.
- Species mix by elevation and patch: spruce, fir (lower), larch (gold, higher and in tracks),
  and grey-brown snags.
- Tags go in the instance `material_id` bits 10–14 (species | crowding).
- Shader: per-species albedo, irregular whorl banding (fades by 1.4 km) and self-occlusion
  scaled by crowding.
- CPU canopy AO darkens the ground under and around every crown (`veg::canopy_ao`).
- Trees are excluded from outcrops.

**Silhouettes (`voxel::fields::Alpine`).**
- New `crags()` column term: five band-limited ridged octaves, 430 m / 95 m down to 13 m /
  2.6 m.
- Creases are softened to ≥ 3 cells, as for the base. Crest lines are chopped into teeth and
  notches by an independent modulation.
- Gated by elevation (1050–1650 m) and region, so ridgelines above ~1.4 km are jagged.
- Steep faces also get vertical joint ribs and chimneys (3D, stretched ridged noise).
- Bounds: `CRAG_MAX` goes into `height_bounds`; the ribs go into `OVERHANG_MAX`.

**Robustness bug found and fixed while doing this.** The documented `SLOPE_MAX = 6` Lipschitz
bound (which `may_cross` relies on to skip bricks) was false. Measured column slopes reached
**17–27 m/m**, from two gully defects:
1. The 3×3 Gaussian kernel neighbourhood dropped kernels of weight ~0.2 when `floor(w/λ)`
   changed, giving metre-high steps. It is now a compact `(1 − d²/(1.5λ)²)³` window over 5×5,
   continuous by construction.
2. The stripe direction came from the 3 m local gradient, which flips across every ridge crease.
   That cut 20 m-deep slots along all crests at fine LODs. The direction now comes from the
   broad slope: base filtered at ≥ 24 m and differenced over ±24 m.

With both fixes and the crags, new test **q6** measures ≤ 4.4 m/m over 40 000 samples (cells 0.25–16 m) and
asserts a 1.5× margin to the new `SLOPE_MAX = 7`. q5 (summit not truncated, skips conservative)
still passes.

**Foreground outcrop: jointed tors (`Alpine::tor_mask` / `tor_parts`).**
- About 40 m patches of bedrock below 1250 m, 1.4 % coverage. The seed salt was chosen from
  1700–1900 so one sits in the vista foreground.
- 3 m smooth column mound, plus a steep 3D block core of up to 11 m.
- The steep core is in the 3D term (counted in `OVERHANG_MAX`), so its steepness is safe for
  `may_cross`.
- Two near-orthogonal vertical joint sets (7.5 m and 10 m, with strike varying over 600 m) and
  dipping 3.4 m bedding. They cut V-cracks of width ≥ 1.5 cells.
- The core is decided **per joint block** from the mask at the block centre, so walls follow
  joint planes and tops step per block.
- Inside the crack windows the core blends to the smooth mask. On a joint plane the block weight
  is exactly 0, so the field is continuous, and at coarse LOD it falls back to the smooth core.
- Materials get exact per-vertex info from the CPU, in interpolation-safe form:
  `attr.w = −rock weight` and `|nrm| = 1 + crack proximity`.
- The shader renders grey-brown blocks with lichen, dark open joints (albedo and occlusion), and
  moss only on flat tops.
- The near LOD2 shell grew from 4 to 6 bricks (`[3, 3, 6, 6]`), so the outcrop at ~150–200 m is
  meshed at 1 m rather than 2–4 m cells. That adds 0.27 M voxel triangles.

**Vegetation through the asset format.** `VegSet` keeps `asset::Instance` records. The example
encodes them in 4096-instance chunks, then verifies, decodes (rayon) and uploads the decoded set.
This is the default; set `BRUTAL_VEG_ASSET=0` to bypass it.

**Tree impostors: not done.** Octahedral impostors need an offscreen bake pass (atlas of
albedo, normal and depth per view), a new pipeline and a shader path in the ash renderer. That was
not feasible this round alongside the rest. Distant trees are still LOD2 cones, in every shadow
cascade.

**Numbers (final frame).** Lavapipe, 4K rendered with 2×2 SSAA to 1080p:

| | value |
|---|---|
| triangles | 4.46 M (1.61 M voxel + 2.85 M rings) + 2.32 M vegetation |
| instances | 78.6 k (61 k visible after GPU cull) |
| **GPU frame** | **1.83 s** (shadow 0.98 s, main 0.85 s) |
| CPU build | 43 s (attributes 27 s, sampling 6.9 s, vegetation 5.5 s) |
| validation | 0 errors, 0 warnings |
| cave | pixel-identical (byte-identical PNG), 0.22 s |

The CPU build was ~23 s in round 3. The broad-slope gully fix costs 4 extra `base()`
evaluations per column, run by every density call in the AO taps; crags and tors add more.

**Gates:**
- clippy `-D warnings`, default and `--all-features`: clean;
- **88 tests** pass (also with `BRUTAL_SCAN=shared`);
- SPIR-V fallback regenerated (`tools/compile_shaders.sh`).

**Weaknesses (honest):**
- The foreground outcrop is still small in frame (~60 px tall at 1080p) and sits in terrain
  shadow. Its faces read mostly through the joint darkening, not sunlit relief, and shaded rock
  still goes slightly lilac under the blue sky ambient.
- Mid-distance tors read as flat-topped boxes.
- Crags give jagged ridgelines on the near massif (~3–5 km, voxel geometry). On the far rings
  (≥ 16 m cells) only the 430 m and 180 m octaves survive band-limiting.
- The whorl banding is a shader effect, not geometry.
- Snags are still slightly cool in tone.
- No impostors, so distant cones still dominate the shadow cost.
- The CPU build almost doubled.
- Avalanche tracks are subtle at this camera.

### Learning from FlatBuffers and Draco: the `asset` crate ✅ (21 Oct 2026)

`crates/asset` is the on-disk / streaming format for terrain bricks (quantised density samples),
meshed brick geometry and vegetation instances. Its container is FlatBuffers-style and zero-copy,
and its payloads use Draco-style compression. It is a hand-rolled equivalent, not a dependency on
either project. Measurements below are on the real Alpine near field at the mountains camera
(359 bricks, 0.91 M triangles, LOD 0–2), from `cargo run --release -p asset --example asset_bench`.

**What we read** (sources):
- FlatBuffers internals: <https://flatbuffers.dev/internals/>. The root `uoffset_t`, the
  `file_identifier` at byte 4, tables with an `soffset_t` to a vtable of `u16`
  (vtable size, object size, field offsets; 0 = absent → default), forward `u32` offsets,
  count-prefixed vectors, scalars aligned to their size, little-endian.
- Schema evolution: <https://flatbuffers.dev/evolution/>. Append fields only, deprecate rather
  than remove, no version numbers by design.
- Schema attributes: <https://flatbuffers.dev/schema/>. `force_align` on vectors "only works for
  generated C++ code".
- The Rust verifier: <https://docs.rs/flatbuffers/25.12.19/flatbuffers/struct.VerifierOptions.html>
  (`max_depth`, `max_tables`, `max_apparent_size`).
- Benchmarks: <https://flatbuffers.dev/benchmarks/>. Decode is ~0 because access is in place.
- Draco bitstream spec v2.2: <https://google.github.io/draco/spec/>, and the README
  <https://github.com/google/draco>.
  - Sequential connectivity: delta + zigzag + rANS.
  - Edgebreaker CLERS with topology splits, and the valence variant.
  - Parallelogram and constrained multi-parallelogram prediction. Attributes are decoded after
    connectivity, so full adjacency is available.
  - Octahedral normals with a geometric-normal predictor and canonicalised integer transforms.
  - Tagged and raw rANS symbols; `-cl 0..10`; default quantisation 11 bits for positions.
- Duda, "Asymmetric numeral systems", arXiv:1311.2540 (2013).
- Giesen, ryg_rans (<https://github.com/rygorous/ryg_rans>) and "Interleaved entropy coders",
  arXiv:1402.3392.
- Rossignac, "Edgebreaker", IEEE TVCG 5(1), 1999.
- Touma & Gotsman, "Triangle mesh compression", Graphics Interface 1998 (parallelogram rule).
- Cigolle et al., "A survey of efficient representations for independent unit vectors",
  JCGT 3(2), 2014 (octahedral).
- Ibarria, Lindstrom, Rossignac, Szymczak, "Out-of-core compression and decompression of large
  n-dimensional scalar fields", Eurographics 2003 (Lorenzo predictor).

**Container (`fb.rs`, `schema.rs`).** We kept the FlatBuffers layout essentially as is:
- root offset + `BRK1` identifier;
- tables → vtables of `u16` field offsets, so absent fields read as defaults;
- forward `u32` offsets only, so the graph is acyclic and the verifier terminates by construction;
- count-prefixed vectors;
- every scalar aligned to its own size;
- O(1) field access straight from the bytes.

The schema is written out as FlatBuffers IDL in `schema.rs`: `Root { major, minor, voxel_m,
brick_size, models, bricks: [Brick], inst_chunks: [InstChunk] }`, where `Brick { key, mesh:
MeshBlob, samples: SampleBlob, raw: RawMesh }` and every field is optional.

Why hand-rolled rather than the `flatbuffers` crate:
1. We need **256-byte-aligned vectors**, so `RawMesh` vertex/index blobs and future GPU-ready blobs
   can be bound or copied with no realignment. `force_align` on vectors is C++-only.
2. The crate needs `flatc` codegen, which is not in this toolchain. Checked-in generated code is
   another artefact to keep in sync.
3. Our payloads are mostly opaque compressed blobs plus a few header scalars, so the generated
   accessors buy little. The container core is ~350 lines.

**Raw blobs are true zero-copy GPU views.** `File::raw_gpu` returns `&[VoxelVertex]` / `&[u32]`
borrowed from the file buffer. Test a8 checks pointer identity and 256-byte alignment, using an
mmap-like `AlignedBytes`. The padding cost is 0.6 % on the uncompressed file.

**The verifier** (`File::open`) runs against `Limits` (depth, table count, total size,
vertices/indices/instances per blob, the same idea as `VerifierOptions`). It checks:
- every table, vtable, offset, vector, alignment and inline field the schema reaches;
- the semantic header limits;
- the `major` version.

`File::open_deep` also decodes every payload. The decoders are total: they range-check every
index, reference, count and coordinate, and the rANS decoder must finish in its initial state with
every byte consumed. So a flip inside compressed data is caught, not silently decoded.

**Evolution.**
- New fields are appended. `MeshBlob.fshift` was added this way during development: files
  without it decode with the default.
- Readers ignore fields beyond their vtable. Test a7 writes a "future" file with unknown fields in
  every table and reads it, and hand-builds an "old" file missing optional fields.
- **Deliberate departure:** we *do* store `major`. FlatBuffers evolves only at the field level and
  cannot say "this blob's predictor changed meaning". An incompatible major is rejected, and
  `minor` is informational.

**Compression (`mesh.rs`, `rans.rs`): what we took from Draco, and what we changed.**

*Positions*
- We quantise to the **brick-local integer sub-grid the mesher already works on**
  (`ipos − brick_origin·SUB`, 1/256 voxel). The default (`qbits = 0`) is therefore **lossless**.
- A vertex shared by two bricks decodes to bit-identical `ipos`, and through the mesher's own
  `sub_to_render` to bit-identical f32. Test a3 checks this over thousands of seam vertices, and
  a2 checks that the decoded world's topology matches the original.
- With `qbits > 0`, the *global* coordinate is rounded to `2^qbits`. That is a pure function of
  the global point, so seams stay bit-identical, and the error is ≤ 2^(qbits−1) (test a3,
  q = 0/2/4).
- Draco quantises per mesh over its own bounding box, so neighbouring bricks would not agree on
  seam vertices.
- Voxel-specific prediction: every vertex lies on a lattice edge, so each coordinate splits into
  a lattice index and an in-cell fraction.
  - The two off-axis fractions are 0, nearly free.
  - The crossing-axis fraction is the edge parameter, coded raw. It is real information:
    density, not geometry.
  - Lattice indices use Draco's **parallelogram** predictor `a + b − o` across the traversal gate.
  - Coarse LODs drop the known-zero low bits (`fshift`).

*Connectivity*
- An Edgebreaker-like depth-first gate traversal with **claim-on-visit**. Each triangle is one of
  8 symbols: tip new, left child, right child.
- An old tip is named by an index into a candidate list built from the decoder's own open
  boundary edges at the gate. This plays the role of Edgebreaker's L/R/E, which resolve such
  tips implicitly.
- Rare "S"-like cases escape to a back-reference.
- Handles and non-manifold bricks need no split symbols: a stale gate is detected from the open
  edges, and a SKIP symbol covers non-manifold input.
- The decoder needs only a stack and per-vertex open-edge lists, with no hashing.
- Result: **2.85 bits/triangle** (conn 1.86 + cand 0.56 + escapes 0.43), against Edgebreaker's
  ~2 bits/triangle.
- Getting there took two steps from a naive DFS:
  - first try (explicit back-references): 7.0 bits/triangle;
  - candidate lists: 5.3;
  - claim-on-visit, which removed most "S" returns: 2.85.
- Sequential mode keeps the exact index order (Draco's sequential connectivity) and costs
  18.4 bits/triangle.

*Normals*
- Octahedral, 10 bits per component (Draco's default range).
- Predicted from the **geometric normal** of the decoded faces, as in Draco, because attributes
  are decoded after connectivity. The residual wraps around.
- We also tried blending in the parents' decoded normals; it was 0.1–0.3 bits/triangle worse, so
  we dropped it.
- The error bound is < 0.25° (test a1).
- The `dirs` sign bit is predicted from the decoded normal: 0.02 bits/triangle.

*Entropy coding*
- One byte-wise **rANS** stream per brick (12-bit probabilities, Duda/Giesen), plus a raw-bit side
  stream for Draco-style TAGGED values (bit-length token through rANS, mantissa raw).
- **Static models are fitted once per file**, a shared dictionary of 22 models × 64 symbols
  (2.8 KB). Draco stores tables per mesh, which would cost more than a small brick's payload.

*What we did not adopt, and why*
- Valence / multi-parallelogram connectivity: the gains are small on these near-regular MC
  meshes, and they cost decode speed.
- Draco's per-mesh bounding-box quantisation: it would break seams.
- Interleaved/SIMD rANS (Giesen): not done yet. Decode parallelism comes from rayon across
  bricks; see the weaknesses below.
- FlatBuffers' generated accessors: see above.

**Density samples** (`samples.rs`): a 3D Lorenzo predictor (2D on transition faces), lossless.
256 bricks: 45.0 MB → 8.24 MB (**5.5:1**), decode 368 MB/s out per core.

**Instances** (`inst.rs`):
- Position is quantised to 1/128 m on an integer chunk grid, Morton-sorted in XZ and
  delta-predicted.
- Yaw: 12 bits raw. Height: `round(64·log2 h)` (≤ 0.55 % error). Width: as a ratio of height.
  Height and width are delta-predicted per mesh.
- Mesh id goes through its own model; material is 10 bits raw.
- Result: 87.7 k forest-like instances on the real height field cost **9.0 B each**: 8.9:1 against
  the 80 B `InstanceGpu`, 4.0:1 against a 36 B f64 record.
- Decode plus camera-relative matrix: 12–14 M instances/s per core.

**Measured** (8-vCPU shared box; raw = 24 B/vertex + 12 B/triangle GPU layout = 22.61 MB):

| mode | file | ratio vs raw | bits/tri | decode MB/s (raw out), 1 core | 8 threads | µs/brick, 1 core | bricks per 2 ms: 1 core / 8 thr |
|---|---|---|---|---|---|---|---|
| traversal, lossless (q0) | 1.976 MB | **11.4:1** | **17.3** | 286 | 1933–2275 | 220 | 9 / 61–72 |
| traversal, q2 (7.8 mm, seams exact) | 1.900 MB | 11.9:1 | 16.7 | 288 | 1968 | 219 | 9 / 63 |
| traversal, lossless pos, 8-bit normals | 1.736 MB | 13.0:1 | 15.2 | 286 | 2221 | 220 | 9 / 71 |
| sequential (exact index order) | 3.947 MB | 5.7:1 | 34.6 | 286 | 2210 | 220 | 9 / 70 |
| Draco qp 11 / qn 10, cl 7 (max err 26 mm) | 2.196 MB | 10.3:1 | 19.3 | 109* | | | |
| Draco qp 11 / qn 10, cl 10 | 2.135 MB | 10.6:1 | 18.7 | 85* | | | |
| Draco qp 14 / qn 10, cl 10 (max err 3.3 mm) | 2.604 MB | 8.7:1 | 22.8 | 80* | | | |
| Draco qp 16 / qn 10, cl 10 (max err 0.8 mm) | 2.751 MB | 8.2:1 | 24.1 | 77* | | | |

How to read the comparison:
- Draco 1.5.x ran through DracoPy 2.2.0, per brick (the same streaming unit), with the
  quantisation box pinned to the brick (`tools/draco_compare.py`).
- \* Draco decode times include DracoPy's numpy conversion, so they understate Draco's C++
  decoder.
- **Our lossless stream is smaller than Draco at 11 bits, which is lossy up to 26 mm here, and
  24 % smaller than Draco at 14 bits.** Draco at any setting also breaks bit-exact seams.

Bits/triangle by stream, lossless:
- connectivity 2.85;
- lattice positions 1.70;
- edge-parameter fractions 4.76 (the information floor; `q2` cuts it to 3.95);
- normals 6.77;
- dirs 0.86.

Further numbers:
- Verify (`File::open`) on the 359-brick file: 45–57 µs.
- Fuzz: 60 000 mutants (bit flips, hostile u32s, truncation, splices, payload-only flips) plus
  15 000 random buffers. 0 panics; 68 % rejected structurally; **10 000/10 000** payload-only
  flips caught by the rANS integrity checks.
- **Streaming budget at 60 fps:** ~220 µs per average brick (2.5 k triangles) on one core, so a
  2 ms slice decodes ~9 bricks (~23 k triangles) per frame per core: 540 bricks/s/core. Across 8
  rayon workers, ~60–70 bricks per 2 ms, i.e. ~4 k bricks/s.
- Input bandwidth is ~25 MB/s per core, so disk or network never limits this.

**Tests** (`cargo test -p asset`: 10 unit + 10 integration on a real Alpine world):
- a1 lossless round trip, both connectivity modes, f32 bit-identical, normals < 0.25°;
- a2 decoded world topology unchanged;
- a3 seam bit-identity + quantum error bound at q = 0/2/4;
- a4 determinism (twice, serial vs rayon, 1 thread);
- a5 samples lossless;
- a6 instance error bounds;
- a7 schema evolution in both directions + major rejection;
- a8 zero-copy 256-aligned GPU views;
- a9 fuzz (`ASSET_FUZZ_ITERS` to scale);
- a10 16-bit instance materials + legacy (minor 0) chunks.

**Miri** (`cargo +nightly miri test -p asset --lib`) is clean under default Stacked Borrows: 10
tests including a full synthetic round trip, the aligned buffer (the crate's only `unsafe`) and a
mutation fuzz. Under Miri, `Writer::finish` runs serially, because Stacked Borrows flags
crossbeam-epoch inside rayon's deque, not our code. With rayon on, the suite is also clean under
`-Zmiri-tree-borrows`.

**Weaknesses (honest):**
- Decode is scalar per brick: ~11 M triangles/s/core. Interleaved rANS or a two-pass, SIMD
  attribute decode is the next step.
- Normals are a third of the bits. Recomputing them on the GPU from samples would remove them.
- Draco was compared only through Python.
- The bench meshes come from the CPU mesher at three LODs, not the renderer's full eight-LOD
  shell.
- Instances are lossy by design (1/128 m, 12-bit yaw, 1/64-octave heights): routing the quality
  scene's vegetation through the format changes the final frame by mean 0.28/255 (thin tree edges).

**Format minor 1 (round 4).** Routing the vegetation through the format exposed a real limitation:
instance materials were clamped to 10 bits, which silently dropped the species/crowding tags (bits
10–14), so every tree came back as spruce. Fixed the FlatBuffers way, without a major bump: the
InstChunk gains an appended `mat_bits: ubyte = 10` field. New chunks write 16 (low 10 bits raw, high
6 bits through a new rANS model `M_IMAT` = 22, so `NUM_MODELS` = 23). Old chunks (field absent) still
mean 10 bits. The reader accepts model tables of at least `BASE_MODELS` = 22 (missing ones are empty
and fail to decode if referenced). Test **a10** covers 16-bit round trip, a legacy chunk in the same
file, and rejection of a bad `mat_bits`.

**Re-measured on the round-4 terrain** (crags, outcrops, gully fix; 333 bricks, 0.72 M triangles,
raw 17.75 MB):

| mode | ratio vs raw | bits/tri | decode MB/s out, 1 core / 8 thr | µs/brick | bricks per 2 ms, 1 core / 8 thr |
|---|---|---|---|---|---|
| traversal q0 lossless | **12.15:1** | 16.32 | 301 / 2323 | 177 | 11 / 87 |
| traversal q2 (7.8 mm) | 12.65:1 | 15.68 | 303 / 1990 | 176 | 11 / 75 |
| traversal q0, 8-bit normals | 13.90:1 | 14.27 | 289 / 2261 | 185 | 11 / 85 |
| sequential q0 | 5.97:1 | 33.21 | 314 / 2151 | 170 | 12 / 81 |
| Draco qp11 qn10 cl10 (26 mm) | 11.17:1 | 17.76 | 89* | | |
| Draco qp14 qn10 cl10 (3.3 mm) | 8.93:1 | 22.19 | 85* | | |
| Draco qp16 qn10 cl10 (0.8 mm) | 8.31:1 | 23.85 | 79* | | |

Other re-measured numbers:
- Verify: 42–52 µs.
- Samples: 5.5:1 lossless at 372 MB/s.
- Synthetic instances: 8.9 B each.
- Quality-scene vegetation, with species tags: 78 591 instances in 766 KB, i.e. 9.75 B each and
  8.2:1 vs `InstanceGpu`. Encode takes 16 ms; verify plus parallel decode takes 3 ms.


---

## 3. Existing pieces: verification status

| Claim | Status | Source |
|---|---|---|
| **Flo** replaces Bevy's wgpu renderer with ash/Vulkan | ✅ Verified. Repo wkwan/flo. Not on crates.io, explicitly "fork and adapt". Single-threaded. Triangle/cube/instancing/GLTF/raytracing examples, ~4x FPS vs wgpu on Steam Deck in its own (admittedly unequal) benches. The author says it was largely built with Claude Code, starting from simple examples. | https://github.com/wkwan/flo · https://www.reddit.com/r/bevy/comments/1nfk252/ |
| **Tiny Glade**: Bevy ECS + custom Vulkan renderer | ✅ Verified. "relies on the ECS, scheduling and a few other thingamabobs from Bevy". Uses a modified Bevy with a custom Vulkan renderer: compute-heavy, GPU-driven culling/indirect draws, software RTGI. | https://pouncelight.games/tiny-glade/info/ · https://80.lv/articles/exclusive-tiny-glade-developers-discuss-bevy-proceduralism-publishers-cozy-games · https://www.youtube.com/watch?v=jusWW2pPnA0 |
| Bevy GPU-driven rendering & mesh shaders | ✅ Verified. GPU-driven rendering landed in 0.16. **Experimental mesh shaders in 0.20** (`WgpuFeatures::EXPERIMENTAL_MESH_SHADER`, `draw_mesh_tasks`, not on web). 0.20 also has Solari/DLSS. | https://bevy.org/news/bevy-0-16/ · https://bevy.org/news/bevy-0-20/ |
| Raw Vulkan escape hatch inside Bevy | ✅ `raw_vulkan_init` feature / `RawVulkanInitSettings` lets you enable extra Vulkan features/extensions under wgpu | https://docs.rs/bevy/latest/bevy/render/renderer/raw_vulkan_init/index.html |
| Bevy MCP servers | ✅ **bevy_brp_mcp** 0.23.0 (Bevy 0.20, Bevy Remote Protocol: entities/components/resources, launch, logs; screenshots/input via `bevy_brp_extras`). ✅ **PyBevy** has its own `pybevy mcp` server (Python API on Bevy, beta). | https://github.com/natepiano/bevy_brp · https://github.com/pybevy/pybevy/blob/main/docs/mcp.md |
| Deterministic agent-control plugin | ✅ **Bevy Agent** (Brian Sunter): reset/step/snapshot/restore/branch over JSON-RPC/HTTP/WS, `AgentTick` schedule. **Targets Bevy 0.18**, so expect porting work. Ships a SKILL.md. | https://briansunter.com/projects/bevy-agent · https://briansunter.github.io/bevy-agent/agent-skill.html |
| Bevy agent skills | ✅ Community skills exist (e.g. bfollington/terma `bevy`, sunsided gamedev-bevy). Quality varies. | https://agentskills.so/skills/bfollington-terma-bevy |
| **Specula** = agent-native engine | ❌ **Wrong.** Specula is a TLA+ formal-spec bug-finding tool. Real agent-native engines: **Euca Engine** (Rust, own ECS, deterministic fork/replay, wgpu, *not* Bevy) and **Limina** (MCP-driven 3D engine). | https://github.com/specula-org/Specula · https://eucaengine.com/ · https://www.liminaengine.com/ |
| **BRUTAL** | ✅ Exists, but it's **C#/.NET, not Rust** (Ahwoo, MIT-based licence, repo created Oct 2026). Interop, unmanaged memory, allocation-free collections, jobs, Vulkan+VMA, Jolt/Box2D/FMOD bindings. Powers Kitten Space Agency and Ludeon's Waylith (RimWorld). No editor, renderer or scene graph. Use it as a *philosophy* reference. | https://github.com/ahwootech/BRUTAL · https://ahwoo.com/brutal |

### AVX2 (x86-64-v3) default and how to opt out

`.cargo/config.toml` sets `-C target-cpu=x86-64-v3` (AVX2 + FMA + BMI2: Haswell / Zen 1 and newer, including Steam Deck) for `x86_64-unknown-linux-gnu`. This is **kept as the default**. On the baseline target, `wide::f32x8` splits into SSE2 halves and `i32x8` multiplies (the noise hash) are emulated, about 2.7× slower. Binaries built this way **do not run on CPUs without AVX2** (SIGILL).

Opt out (verified):
- `RUSTFLAGS="" cargo build …`: an empty `RUSTFLAGS` overrides `target.*.rustflags`. Checked with `cargo build -v`: the flag disappears.
- Or `RUSTFLAGS="-C target-cpu=x86-64-v2"` (SSE4.2) / `-C target-cpu=native`.
- Or delete `.cargo/config.toml`.
- **Doesn't work:** `--config 'target.x86_64-unknown-linux-gnu.rustflags=[]'`. Cargo merges config arrays, so the flag stays.

To ship to non-AVX2 machines, either build two binaries (v1/v3) and pick at launch, or use runtime dispatch (`multiversion` / `is_x86_feature_detected!`) in the hot crates.

Results are bit-identical either way. Rust never contracts `a*b+c` into FMA, and the terrain golden hash plus all integer voxel paths are target-independent.

### Custom renderer + Bevy: how others do it
- **No bevy_render (Mode B):** Flo. **vkbevy** (IceSentry, Bevy rendering contributor) is a minimal demo of "Bevy with ash instead of wgpu": https://github.com/IceSentry/vkbevy. Pattern: disable render features, take `RawHandleWrapper` from the window entity, create the surface with ash-window, and run the renderer as a NonSend resource or on its own thread fed by a snapshot (triple buffer).
- **Keep bevy_render (Mode A):** add `ExtractSchedule` systems + render-graph nodes. Drop to raw Vulkan via `raw_vulkan_init` / wgpu-hal when one pass needs it. This is the lower-risk first step.
- **Threaded game/render split + render graph with auto-barriers in pure Rust/ash:** ursus (reference code, small project): https://github.com/PrograMistV1/ursus

### SIMD in Rust: practical choice
- **`wide`** (1.7.1, stable): `f32x8`/`f32x4` with runtime-agnostic codegen. This is the default for hot crates.
- **`std::simd`**: still nightly-only (`portable_simd`). Keep it behind a cfg.
- **glam** (0.34.1): SIMD *within* one Vec3A/Vec4/Mat4 (AoS). Great for gameplay math, but not for throughput over 1M elements. Use SoA + `wide` there and convert at boundaries. Bevy's `bevy_math` re-exports its own glam version, so don't assume your glam types are the same.
- **Layout:** SoA (or AoSoA blocks of 8) padded to the lane width. Use `chunks_exact`, rayon over 16k chunks, and keep a scalar oracle plus proptest. Check codegen with `cargo asm`. Enable `-C target-cpu=x86-64-v3` per-target only if you ship AVX2-min, or use `multiversion` for runtime dispatch.

### Next steps
1. Capture a Tracy profile of a representative scene (`bevy/trace_tracy`) to find the real hotspots.
2. Done: Prompt B (+ round 2 overlap/GPU scan, round 3 subgroup scan); Prompt C single-LOD (perf, staging ring, eviction); coords crate (interpreted); voxel bricks + CPU Transvoxel with watertight slab LOD seams.
3. **Decisions needed:**
   - terrain LOD seam strategy (skirts vs stitching vs Transvoxel-style; `SeamStrategy` is the hook);
   - final brick size and voxel spacing (see the comparison table);
   - the real coords, C1–C4 and voxel specs, plus the "robustness hard rule" text, to check against the interpretations here;
   - the 8-scene list, to map Prompts D/E.
4. Then: voxel GPU (Slang) sampling + SIMD density; clipmap LOD corners (2- and 3-face transitions); Prompt D (ocean); Prompt E (atmosphere). Measure Prompt B/C on real GPUs (overlap, subgroup sizes 32/64, transfer queue). Remaining Prompt B gaps: decoupled-lookback scan, deterministic in-mesh order, async compute, golden-image test.
5. Wire `bevy_brp_mcp` into the app so agents can launch, screenshot and inspect between iterations.


## Tree impostors, GPU LOD and boulders (21 Oct 2026)

Images: `out/quality/steps/11_final.png` (before), `14_final.png` (after),
`14_closeup_impostors_vs_meshes.png` (from `13b_closeup`), `14_closeup_boulders.png` (from `13d_erratic`);
the intermediate steps are `12a`–`12g` (impostors) and `13a`–`13e` (boulders, outcrops).

**Impostor atlas** (`render-ash/src/impostor.rs`): per tree species (spruce, fir, larch, dead snag) the LOD0
mesh is baked at init on the CPU (rayon, deterministic) into a hemi-octahedral atlas of 9x9 views x 128 px,
3x3 supersampled. Texel = 2 x u32: alpha8, height fraction 7 + bark bit, depth16 | octahedral normal 8+8,
AO8. 4 variants = 42.5 MB. Per-variant header: bounding sphere and AABB half-extents.
- The card is fitted to the projected AABB (x1.06 main, x1.01 shadow), and shadow cards are pushed 0.35 r away
  from the sun.
- `impFS` blends the 3 nearest frames (barycentric on the octahedral grid), refines with parallax from the
  depth channel and writes exact `SV_Depth`. It is used only up to `IMP_EXACT` (~1 km), because `SV_Depth`
  forces late-Z on lavapipe.
- `impFarFS` uses card depth (early-Z stays on). Shading runs through the terrain `shade()`: the same
  atmosphere, sun, sky, shadows and aerial perspective as the meshes.
- Shadows: `impShadowFS` (exact depth, nearest frame) in cascades c0/c1, `impShadowFarFS` (coverage only,
  single fetch) in c2/c3.
- Alpha is dithered (IGN noise). `discard` needs the `shaderDemoteToHelperInvocation` feature, which is now
  enabled and required.

**GPU LOD** (`shaders/qcull.slang`; replaces `pass.rs` `CullChain`): views are 0 = camera and 1..4 = cascades.
- Per instance: sphere-vs-frustum cull, then a walk of the mesh's LOD chain
  `lod[mesh] = (next, switch_m, fade_m, exact_m)`, by camera distance (`lod_scale` per view).
- Camera view: inside the fade band the instance is emitted twice with complementary dither codes
  (bit 7 = upper, bits 0-6 = threshold), which gives a smooth crossfade with no popping band.
- c0/c1 switch hard (c1 at 2x scale), c2/c3 take the coarsest level (impostor).
- Per view: 2M fixed draw slots, [0, M) whole and [M, 2M) crossfading or exact-depth impostor. Counts go
  through `record_scan`, then scatter (`idx | code << 24`), then `drawsMain`, then
  `vkCmdDrawIndexedIndirect`.
- Chain per tree: LOD0, then LOD1 at 260 m (fade 50), then the impostor at 900 m (fade 140). Each LOD chain
  can be overridden with `VEG_LOD1_AT`, `VEG_IMP_AT`, `VEG_*_FADE` and `VEG_IMP_EXACT`.
- The LOD is no longer fixed at placement.

**Terrain shadow casters:** the near terrain index buffer is reordered into 822 XZ chunks (80 m bins,
doubling per distance octave, with bounding spheres). Per cascade, the CPU keeps a chunk only if it is inside
the light frustum and its sun-swept sphere (6000 m) reaches that cascade's receiver slice. c3 drops from 3.8 M
to 1.85 M triangles. This fixes the old behaviour where cascades reused the camera-visible instance list, so
off-screen casters were missing.

**Boulders** (`voxel/src/boulder.rs`, `quality_parts/veg.rs`):
- Shape: an SDF of 14 jittered face planes, 7 chipped edges and 1 fracture plane, with a flattened scale and
  small Perlin roughness. Meshed with Transvoxel (`UniformLod`, merged), with `keep_largest` components and
  then `topology()`.
- 6 variants x 3 LODs (20 / 9 / 5 cells across), with GPU LOD switches at 60 m and 200 m. Tests b1
  (watertight, 1 component, no degenerates, normals agree with winding, at both resolutions) and b2
  (determinism, variants differ, sizes, faceting: > 60 % of the area within 20° of the cutting planes).
- Burial: each variant is buried 22–42 % below its mesh y = 0, with vertex AO darkening toward the burial
  line. Triangles entirely deep underground are dropped. Each instance also sinks by slope x size.
- Placement on a 2.5 m grid to 700 m:
  - talus below steep ground (the steepest of 25/50/80 m up the fall line > 0.85–1.5) and in rings around
    tors, with power-law sizes;
  - rare large meadow/forest erratics (1.4–3 m);
  - a sparse sprinkle elsewhere, fewest on flat grass.
- Terrain contact AO in a ring of 1.5 footprint radii (`rock_contact_ao`). Shadows come from the normal
  instance shadow path.
- Material: `mountain_material(..., boulder = 1)` gives the local bedrock with no meadow/scree/snow blending,
  plus crustose lichen on sky-facing faces and moss cushions on top faces (more in moist and forest ground,
  via the crowd bits).
- Instances never overlap the eye (free cameras no longer start inside a crown).

**Outcrops:** at coarse LODs the joint lattice fades and the tor core fell back to a flat `TOR_BLOCK`
plateau (flat-topped boxes at mid distance). The fallback is now modulated by band-limited noise
(x0.48–1.0, so the bound holds).

**Timings** (lavapipe, 1920x1080 with 2x2 SSAA, mountains scene, median of 3 frames):

| | before (`11_final`) | after (`14_final`) |
|---|---|---|
| shadow pass | ~980 ms | **653 ms** (c0 80, c1 167, c2 217, c3 172 incl. ~8 ms copies) |
| main pass | ~850 ms | 974 ms (terrain 557, meshes 284, impostors 132) |
| frame | ~1830 ms | 1627 ms |

- About half of the shadow cut comes from terrain caster culling and half from impostors in c2/c3.
- Main is slower on lavapipe. Measured contributors: the mesh-to-mesh dither crossfade double draws
  (~40 ms; `VEG_LOD1_FADE=0` gives 974 -> ~935 ms) and exact-depth (`SV_Depth`, late-Z) impostors near
  the switch (`VEG_IMP_EXACT=0` saves ~50 ms but shows seams). With impostors disabled in the main view,
  main is still ~1000 ms. The extra cost is mostly the boulders, the shifted LOD distribution and the
  crossfades, not the far impostors. Real-GPU behaviour is not measured.


## Blade grass, needle-card trees, faceted boulders, shade balance (21 Oct 2026)

Images (`out/quality/steps/`):
- close-up before / after, same camera (`CAM_X=7799 CAM_Z=-3614 CAM_H=6 CAM_YAW=3.14 CAM_PITCH=-0.1`):
  `14_closeup_boulders.png` / `16_closeup_after.png` (crops of `13d_erratic` / `16b_trees`);
- default view: `16_final_default.png`;
- impostor close-up (`/workspace/rc.sh` camera): `16_closeup_impostors_vs_meshes.png`;
- steps: `15a` (shade + boulders), `15b`/`15c` (grass), `16a`/`16b` (card trees), `16c`, `16d`.

### Reference: CK42BB/procedural-grass-threejs (MIT, (c) 2026 Kingsley)

It is a Three.js skill (GLSL / WGSL / TSL snippets). Nothing was copied verbatim; the techniques were
reimplemented in Slang on our GPU-driven path.
- **Blade geometry:** a tapered triangle strip along a quadratic Bezier (root, mid control, bent tip),
  `segments` cross-sections + a tip vertex, about 80 % taper. 3 segments far, 4–5 mid, 6–8 hero.
- **Per-blade variation:** instance (position, yaw) + (width scale, height scale, tilt, colour variation).
  Random yaw, with some alignment to the wind recommended.
- **Wind:** three layers, each scaled by height² so roots stay fixed:
  - a global sinusoidal sway along the wind direction;
  - gust fronts (a noise envelope rolling across the field);
  - per-blade turbulence from a position hash.
  The normal is tilted by the displacement ("bent normal").
- **Colour / translucency:** base→tip gradient, a dry colour by per-blade variation, ±15 % brightness
  jitter, root AO (`mix(1 - ao, 1, smoothstep(0, 0.3, t))`), half-Lambert wrap diffuse. SSS approximation:
  `pow(max(dot(-V, L), 0), 3)`, stronger at the tips; a faint specular sheen.
- **Density / LOD:** jittered-grid placement with slope / altitude rejection and noise density, distance
  rings (e.g. 20 / 45 / 80 m at density 1 / 0.4 / 0.1, fewer segments), or a shader distance fade with
  `discard`. Alpha test rather than blend. Grass rarely casts shadows (or uses a simplified caster).
- **Clumping:** Voronoi-like clump centres, `clumpFactor = 1 - min(d / clumpRadius, 1)` raised to
  `2 * clumpiness`, plus per-type presets (meadow: 0.9 m blades, density 35, clumpiness 0.4).
- **Seasonal colour:** spring→summer→autumn lerp with a growing dry amount.

### What we built from it

**Blade grass** (`crates/render-ash/src/grass.rs`, material code 5):
- Tiles: 0.5 m tiles of Bezier blades, deterministic. Levels:

  | level | blades / tile | segments | width multiplier |
  |---|---|---|---|
  | 0 | 38 (152 per m²) | 4 | 1 |
  | 1 | 18 | 2 | 1.5 |
  | 2 | 9 | 1 | 2.9 |

  Widths grow as counts drop, so the covered area stays within ±40 %. Each level keeps a subset of the same
  blades.
- Shape: 3 clumps per tile (75 % of blades): shared height multiplier, tint and outward fan / lean.
  "Rounded" normals: the edge vertices are tilted ±35° around the blade axis, so flat blades shade as
  curved.
- Vertex attributes: root AO, height fraction and a per-blade random in `attr.z`.
- Placement: unrotated tiles on the 0.5 m lattice to ~47 m on meadow (slope < 0.7, below ~1380 m), patchy
  edges, height from a smooth world field. Tiles carry yaw only, so they are sunk by slope × 0.24 to stay on
  the ground.
- GPU LOD in `qcull`: LOD0 to 9 m, LOD1 to 20 m, LOD2 to 40 m, then the new `LOD_VANISH` chain end
  dithers the blades out over 12 m in the camera view. Shadow views: hard cut in c0/c1, never in c2/c3.
  Blades cast and receive shadows through the normal instance passes.
- Static wind, deterministic for the golden image (`grass_wind`, in `inst_rel`, so shadows match): the
  reference's three layers frozen at one instant. A steady down-wind lean, gust fronts (a noise envelope
  sliding along the wind) × a travelling wave, and per-blade flutter from the blade random. All ∝ hf², and
  the blade is shortened to keep its length.
- Shading: the terrain's own meadow palette from the same macro fields, so the fade into the terrain shows
  no band. Per-blade / clump hue, lighter toward the tip, dry straw tips (October), translucency through
  `shade()`.
- Beyond ~12–38 m the blade normal blends to the ground normal and the glow fades. This matches the tone
  of the terrain meadow the blades dither into.
- The terrain under the blades (< 30–46 m) is a darker, browner thatch layer.
- Tests: `grass_patches_are_deterministic` (bit-identical across calls and threads, variants differ) and
  `grass_density_budget_and_geometry` (triangles per level ≤ 360 / 100 / 30, 152 blades per m², roots in
  the tile, heights bounded, no degenerate triangles, unit normals, area ratio).

**Boulders:**
- Geometry: lower SDF roughness and more small chips. The render mesh is unwelded, and each triangle takes
  the exact normal of the fracture plane it lies on (`Boulder::facet_normal`), giving flat facets and crisp
  edges. Triangles on the chamfers keep their face normal.
- Material: the outcrop palette (grey-brown, rock type over hundreds of metres, per-block tone), with fine
  grain only and no relief bump. Lichen is small, sparse rosettes (~5–10 %, pale grey-green, some ochre)
  that fade into their mean tint below ~2 px. Moss only on near-horizontal tops, subtle.
- The earlier "yellow-olive melted lumps" in the close-up were mostly the **shrubs** (lobed alder / willow
  domes). Those are now leaf-card shrubs: 7 arching stems with crossed broadleaf cards, with the lobed mesh
  kept beyond 60 m.

**Shade balance:** shaded grass rendered teal (e.g. RGB 33,49,42 in the close-up). The cause: green albedo
lit only by the sky-view LUT's blue irradiance, with no bounce on up-facing ground. From a valley, much of
the dome is sunlit mountainside and forest. So the LUT irradiance keeps its luminance but only 62 % of its
saturation (×0.85), and up-facing surfaces get a horizon bounce from sunlit terrain (grey-green,
0.12 × sun × sin(elevation)). Result: shade reads neutral to slightly warm-green (40,47,30-ish in grass).
The cave is unaffected (separate branch).

**Near trees** (material code 6, `LOD_ALPHA`):
- Geometry: LOD0 conifers have a tapered 7-sided bark trunk to the leader (visible below the crown), and
  `tiers + 6` whorls of `0.8 × seg` drooping shoots. Each shoot rises off the trunk, sags with a
  species droop, and spruce tips turn up.
- Cards: each shoot carries 3 crossed needle cards (a flat spray ± 0.9 rad) plus 2 side shoots with one
  card each, and a leader pair. About 4–5 k triangles per tree.
- Shading: crown-volume normals (outward + up + some card normal, not flipped as two-sided); AO toward the
  trunk from the card coordinate and toward the lower crown; species colour variation as before.
- Alpha: the needle mask (twig + forward-leaning needle rows inside a tapered envelope; broadleaf blobs for
  shrubs) is procedural from the card UVs in `attr.x / attr.z`. Below ~2 px it switches to a stable
  64 %-coverage dither.
- `qcull` routes `LOD_ALPHA` meshes to the discard slot class in every view: `instFS` in the main view, and
  the new `InstShadowAlpha` pipeline (cut-out shadows) in the cascades.
- LOD chain per species: cards to 80 m (fade 25), whorled cone crown to 260 m, low crown to 900 m, then the
  impostor.
- The impostor atlas is **re-baked from the card LOD0**: `impostor::card_alpha` / `card_ao` mirror the
  shader exactly.

**Timings** (lavapipe, default view 1920x1080 with 2x2 SSAA, median of 3 frames)

| | 14_final | 16_final_default |
|---|---|---|
| shadow | 653 ms | 730 ms |
| main | 974 ms | 1117 ms (terrain 585, instance meshes 374, impostors 157) |
| frame | 1627 ms | 1847 ms |

- Close-up camera, single frame: main instance meshes 273 ms with the cone trees + blade grass (`15b`),
  899 ms with the card trees (`16b`).
- The alpha-tested cards and the discard pipeline (no early-Z on lavapipe) are the dominant cost of this
  round. Blade grass cost is small in the default view, where the camera is 40 m up and blades are ≥ 40 m
  away.

## Close-up round 7: exposure, grass ecology, alpine shrubs, card-tree LODs (21 Oct 2026)

Step images `out/quality/steps/17*`, `18*`, final `19_*`; checkpoints `cp25`..`cp29`.

- **Brighten-only auto exposure** (`examples/quality.rs`, mountains only, `AUTO_EXPOSURE=0`
  disables): the warm-up frame's log-average display luminance is measured; below the key
  0.09 the exposure is raised in two damped steps (`(key/k)^0.85`, at most x2.5).
  Deterministic. The default vista (key 0.12) keeps exposure x1.000 and is pixel-identical
  to round 6; the cave is untouched. Sunlit close-up variant: `SUN_EL=0.8` (the sun is
  camera-relative: `SUN_REL_YAW`, `SUN_EL`).
- **Grass** (`src/grass.rs`, `veg.rs`): four species per tile kind (`seed % 4`): fine,
  seed-head stems (5.2), coarse tussock (5.1), forbs with flower heads (5.3); 8 variants,
  quarter-turn yaw (square tiles still abut: no lattice). Tiles sheared onto the meshed
  ground plane in `inst_rel` (dH/dx, dH/dz from surface finite differences, 6+6 bits in the
  grass material word, 4 random bits): no floating or buried tiles, no slope sink. Density =
  slope (0.5..1.0 cut) x treeline x multi-scale patch noise (38 / 9 / 2.3 m) x canopy
  thinning; cleared inside boulders / trunks with tall coarse tussocks in a ring at their
  bases. Height multi-scale (40 m, 4 m, per tile). Last level fades 27..47 m (was 34..46),
  terrain "under" darkening matched and softened.
- **Shrubs**: dense twig clusters of small crossed cards (18 stems x 5 forks; inner stems
  steep and short = occluded core), green alder (broadleaf pattern, dull olive with
  yellow-brown patches) and mugo / juniper (needle pattern, dark / blue-grey); far level =
  same generator with fewer wider cards (replaces the lobed dome) from 60 m. Whorl banding no
  longer applied to shrubs.
- **Tree LODs**: the cone crowns are gone. Levels 1 (60..260 m) and 2 (260..650 m) are the
  same card-tree generator at lower detail (fewer whorls / shoots, no side shoots, wider
  cards), all alpha-tested; impostors from 650 m (was 900). Trunk ends just above the crown
  base in levels 1/2 and at 0.86 (shaded dark inside the crown) in level 0. `card_alpha`
  blends the exact needle / leaf pattern into dithered coverage (0.78) smoothly by
  `fwidth` (no switch band, less speckle). Near card trees from 60 m (was 80). The cone
  generator `conifer()` was removed.

Timings (lavapipe, 3840x2160 SSAA, ms; shadow / main / frame):

| view | round 6 | round 7 |
|---|---|---|
| default (median of 3) | 730 / 1117 / 1847 | 702 / 1317 / 2019 |
| close-up `16_closeup_after` camera | only instance meshes recorded: 899 | 694 / 2088 / 2782 |
| impostor close-up | 775 / 1431 / 2206 (17e) | 778 / 1627 / 2405 |

Known weaknesses: the mid card levels cost ~+200 ms main in the default view (more alpha
tested fragments; `VEG_IMP_AT` can pull impostors in); in full sun at x1.3 the meadow grass
reads warm / yellow; flower heads and seed heads are barely visible at the close-up scale;
grass gaps show the terrain's meadow shading, not distinct soil / moss; the nearest shrub
sprays read a little stiff and radial.

## Learning from turbovec and TurboQuant (21 Oct 2026)

Sources were read in full: the paper arXiv 2504.19874v1 (Zandieh, Daliri, Hadian, Mirrokni,
"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate") and turbovec
(github.com/RyanCodrai/turbovec at c3b043b, 6 Oct 2026, MIT (c) 2026 Ryan Codrai). The notes
are in `/workspace/notes/turboquant_paper.md`, the bench output in `notes/tq_bench.txt`, and
the step images `out/quality/steps/20_*`. Checkpoints `cp31`..`cp35`.

**What the sources say**
- **TurboQuant.** A data-oblivious (online) vector quantiser.
  - A random rotation (QR of a Gaussian matrix) makes each coordinate of a unit vector follow
    a scaled Beta distribution (Lemma 1). In high d this tends to N(0, 1/d), and the
    coordinates become nearly independent. So a precomputed per-coordinate Lloyd-Max scalar
    codebook is near-optimal (Algorithm 1). The norm is stored separately.
  - Theorem 1 gives D_mse <= (sqrt(3) pi / 2) 4^-b, about 0.36, 0.117, 0.03 and 0.009 at
    b = 1..4.
  - For inner products, Q_prod (Algorithm 2) applies Q_mse at b-1 bits, then a 1-bit QJL sign
    sketch of the residual. This makes the estimate unbiased (Theorem 2: 1.57/d .. 0.047/d).
  - A lower bound of 4^-b (Theorem 3) puts TurboQuant within about 2.7x of optimal (about
    1.45x at b = 1).
  - Entropy coding the indices would gain about 5%; the authors chose not to.
  - Experiments use one A100 (KV cache, nearest-neighbour search). The GPU is used for
    throughput; the method is not GPU-specific.
- **turbovec.**
  - **CPU-only** SIMD (NEON, AVX-512BW, AVX2, scalar). There is no GPU code and no QJL
    stage; it uses a per-vector length renormalisation ||v|| / <u, x_hat> instead.
  - The rotation is 2 rounds of {ChaCha8 permutation, ±1 signs, Walsh-Hadamard}. It replaced
    dense QR in 0.9.0 because the QR result was not bit-reproducible.
  - It uses a Lloyd-Max codebook for Beta((d-1)/2, (d-1)/2), optional TQ+ per-coordinate
    calibration, FastScan-style nibble lookup-table scoring and a staged bit-plane search.
  - CRCs cover headers only.
  - So the premise "all these use GPU" is only half true.

**Mapping onto this engine**
- **Density/SDF bricks** (`d`: 42875 i32 per n = 32 brick) are the only high-dimensional
  payload, so they were prototyped (below).
- **Normals** have d = 3. There the Beta law is *uniform* and the coordinates are dependent,
  so the theory gives nothing. Measured on 382k real mesh normals (mean / max angular error):
  - octahedral 2x8 = 16 bits: 0.351 / 0.946 deg;
  - rotated scalar 3x5 = 15 bits: 1.345 / 2.992 deg;
  - 3x7 = 21 bits: 0.339 / 0.756 deg, against octahedral 2x10 = 20 bits: 0.087 / 0.234 deg.

  Octahedral wins at every rate. Not adopted.
- **Instance transforms** (9 B: a few correlated scalars) and **vertex attributes**: low d
  and already grid-quantised. Not applicable.
- **Seam positions, transition faces and anything feeding the robustness hard rule** must stay
  bit-exact. Lossy VQ is never applied there (measured below: it opens 230k seam edges).
- **Nearest-neighbour / inner-product search** (impostor view selection, material
  similarity): our searches cover at most a few dozen directions or materials and run on the
  GPU in constant time. No vector-database-scale problem exists. Not adopted.
- **What does carry over:** fixed-rate random-access decode, which is GPU-friendly where rANS
  is not; a structured, bit-reproducible rotation (turbovec's lesson); and verify-before-decode.

**Prototype**
- **CPU codec, `asset::tq`.**
  - 512-sample blocks: mean, then a normalised residual, then a fixed 2-round
    {SplitMix Fisher-Yates permutation, signs, unnormalised Walsh-Hadamard} rotation, then
    the Max (1960) Gaussian Lloyd-Max codebook at 1..4 bits.
  - A per-block least-squares scale `a` replaces QJL, because we need MSE rather than
    unbiased inner products.
  - Reconstruction is `floor(mean + a·v + 0.5)`, clamped to ±DQ_MAX.
  - The blob has a header plus CRC32; `TqView::parse` verifies everything, then gives
    zero-copy slices.
  - Hybrid variant: face planes kept exact through a Lorenzo-coded residual.
- **GPU decoder, `shaders/tq.slang` + `render-ash::tq_gpu`.**
  - One 256-thread workgroup per block, with a groupshared WHT. It reads the blob words
    straight from the upload.
  - `precise` blocks FMA contraction, so it is bit-exact with the CPU (tested on real
    bricks plus a full-range adversarial brick, 1..4 bits).
- **Opt-in render hook:** `BRUTAL_TQ=<b>|<b>h|band` in `examples/quality.rs`. Unset means
  the default, unchanged.

**Measured.** Alpine near field, 333 meshed bricks, 14.28M corner samples; `tq_bench`. The
shift column is the edge-crossing shift (mean / p99) in cm; topology is crossings gained or
lost per 1k crossings.

| format | bits/sample | rel. MSE | shift mean / p99 (cm) | topology /1k |
|---|---|---|---|---|
| **current: Lorenzo + rANS, lossless** | **5.50** | 0 | 0 / 0 | 0 |
| TQ 2-bit | 2.14 | 0.117 | 33.8 / 155 | 11493 |
| TQ 4-bit | 4.14 | 0.0095 | 29.2 / 143 | 3196 |
| TQ 4-bit hybrid (exact faces) | 9.36 | 0.0076 | 23.1 / 136 | 2443 |
| per-block uniform min/max 4-bit (no rotation) | 4.14 | 0.0057 | 22.8 / 107 | 3704 |
| TQ 4-bit without rotation (ablation) | 4.14 | 0.0089 | 23.4 / 111 | 4093 |
| **mesh-exact band ±1 vox + Lorenzo** | **2.04** | (far field only) | **0 / 0** | **0** |

- The relative MSE reproduces Theorem 1 almost exactly: 0.365, 0.117, 0.034 and 0.0095 at
  b = 1..4.
- But MSE is the wrong objective here. A brick's energy sits in large values far from the
  surface (up to ±450 voxels), while the mesher reads only signs and values next to the
  zero crossing.
- On smooth fields the rotation also hurts. It whitens spatial structure the plain block
  quantiser exploits: the uniform quantiser beats TQ in MSE at every rate.
- Throughput. CPU (this box) at 4-bit: encode 58 M samples/s on 1 core, 433 M/s with rayon;
  decode 122 M/s on 1 core, 968 M/s with rayon; verify 0.5 GB/s. GPU decode on **lavapipe**
  (a CPU emulation, not a real GPU) runs at about 31-44 M samples/s, e.g. 107 ms for 109
  bricks (4.7M samples).
- Bytes: a 4-bit blob is 22.2 KiB per brick against 167.5 KiB as raw i32 (7.5x less upload
  per chunk, e.g. 2363 vs 18255 KiB). Samples are transient in this engine (resident data is
  meshes), so the saving is staging bandwidth, not resident VRAM.
- Renders: mountains (`out/quality/steps/20_*`, diffs `20_tq4_diff.png`, `20_tq4h_diff.png`).

  | mode | pixels differing from default | interior_open_edges | islands removed | voxel tris |
  |---|---|---|---|---|
  | `BRUTAL_TQ=4` | 27.6% (max 178) | 230,759 | 33,714 | 2.72M (default 1.60M) |
  | `BRUTAL_TQ=4h` | 27.3% | 0 | 23,574 | 2.64M |
  | `BRUTAL_TQ=band` | 0 (pixel-identical) | 0 | 0 | 1.60M |

  The cave is never touched; its default render is pixel-identical to `/tmp/cave_ref.png`.
- Tests:
  - `asset` (6):
    - Lloyd-Max fixed points;
    - scaled-orthogonal, deterministic rotation;
    - error bounds plus hybrid faces exact;
    - the verifier rejects 2000 single-bit flips, truncation and bad bits;
    - `mesh_band` keeps meshes bit-identical (shell plus adversarial layouts);
    - (the mesh-exact band is also verified on all 333 Alpine bricks in `tq_bench`).
  - `render-ash` GPU: `gpu_tq_decode_matches_cpu_exactly` (bit-exact, corrupt blob
    rejected, 0 validation errors).

**What we adopt and what we don't**
- **Not adopted:** TQ for density bricks, normals, instances or search. Lossy rotated VQ
  breaks seams (pure) or adds noise islands (hybrid) even at 4 bits, and the hybrid is
  larger than the current lossless coder. On d = 3 data octahedral coding is strictly
  better.
- **Kept as an opt-in prototype:** `asset::tq` and `tq.slang`, as the reference for fixed-rate
  GPU-decoded payloads. It also carries the techniques worth keeping:
  - a structured, bit-reproducible rotation (permutation, signs, WHT; no dense QR, no FMA);
  - CRC-verified zero-copy blobs;
  - `precise` for CPU/GPU bit-exact float decode.
- **Recommended, from what the study showed:** the **mesh-exact narrow band**
  (`asset::tq::mesh_band`). Samples the mesher never reads are clamped to ±1 voxel, while
  every corner of a sign-change cell, every transition-face plane and their 1-sample dilation
  stay exact.
  - It is lossless for the mesh by construction and tested.
  - Same coder, 5.50 -> 2.04 bits/sample (2.7x smaller).
  - Pixel-identical render.
  - Caveat: valid only while meshing is the samples' sole consumer (anything that later
    reads far-field density, such as AO from samples, collision or SDF queries, would need
    the unclamped data). It has not been wired into `Writer` yet.

Checks (this round):
- clippy `-D warnings` is clean, default and `--all-features`.
- 101 tests pass (95 + 6 new), also with `BRUTAL_SCAN=shared`.
- Validation: 0 errors on the cave and on mountains in every `BRUTAL_TQ` mode.
- The cave is pixel-identical to `/tmp/cave_ref.png`.
- The SPIR-V fallback was regenerated (`tq_decode.spv` added; the existing modules are
  byte-identical).

Reproduce: `cargo run --release -p asset --example tq_bench`;
`BRUTAL_TQ=4 cargo run --release -p render-ash --example quality -- --scene mountains`.

## Mesh-exact band in the asset writer (format minor 2, 21 Oct 2026)

**Format.** `SampleBlob` gets an appended field `band: uint = 0` (slot 4), and `schema::MINOR`
goes 1 -> 2 (major unchanged).
- `band = 0` or absent means full density. Full payloads omit the field, so their bytes are
  identical to minor 1.
- `band = t >= 1` means the payload is [`samples::mesh_band`] with threshold `t`. Every
  sample the mesher can read stays exact: all 8 corners of every cell with a sign change,
  whole face planes of transition faces, and a 1-sample dilation of both (for the
  central-difference gradients). Everything else is clamped to ±t, so signs are preserved.
- The transition mask is the payload's own `face_mask`.
- Writer: `add_samples` (full), `add_samples_banded(s, t)` and `add_samples_banded_par`.
- The reader verifies the flag on decode: a banded payload must be a fixed point of
  `mesh_band` (also `t <= DQ_MAX`, and the mask must match the faces). Anything else is
  `Corrupt`; `open_deep` runs this check.
- Old readers (minor <= 1) ignore unknown fields, so they would treat a banded payload as
  density. **Only write banded files for minor-2 consumers.**
- Backward compatibility is pinned by a golden minor-1 file written by the cp35 code
  (`crates/asset/tests/data/minor1_samples.brk`).

**Safety caveat, enforced by types.** Off-surface values in a banded payload are not density.
- `File::decode_samples(i)` returns **true density** only. On a banded payload it returns
  `Err(CodecError::Banded)` without decoding. Collision, SDF queries, AO from samples,
  re-meshing at another LOD, iso or transition mask, and edits or digging must use it.
  On `Banded` they must request a full payload or regenerate the brick from the field.
- `File::decode_mesh_samples(i)` returns `MeshSamples` (full or banded). Its inner samples are
  private. The only operations are:
  - `mesh(cfg, finer)`, which always uses the stored transition mask;
  - `samples_for_meshing_only()`, for the GPU mesher. The name is grep-able, and the caller
    must pass `transitions()` as the mask;
  - `into_density()`, which is `Some` only for full payloads.
- `File::decode_payload(i)` returns the typed `SamplePayload::{Full, Banded}`.

**Coverage tests.**
- `asset` `a11`: the golden minor-1 file reads unchanged; banded payloads are refused as
  density and mesh bit-identically; mislabelled payloads are rejected.
- `asset` `a12`: Alpine, blob and adversarial-fuzz fields on multi-LOD shell layouts. Every
  brick, with or without transitions, meshes bit-identically from the banded file, and the
  welded topology equals the full-density topology.
- `render-ash` `gpu_mesher_on_banded_payload_matches_full`: the GPU Transvoxel output on
  banded payloads equals the output on full ones (88 bricks, 24 with transitions).
- `band_bench`: all 332 real bricks (LODs 0-3, 55 with transition faces) mesh bit-identically.

**Measured.** `cargo run --release -p asset --example band_bench` on the Alpine near field
(3 shells), samples only, faces included, 14.47M samples (57.9 MB as i32).

| | band off | band on (t = 1 voxel) |
|---|---|---|
| file | 9.995 MB, 5.53 bits/sample (5.8:1) | **3.836 MB, 2.12 bits/sample (15.1:1)** |
| encode | 196 ms (serial) | 269 ms serial / 72 ms rayon |
| decode all, 1 thread | 165 ms (88 M samples/s) | 246 ms (59 M samples/s, incl. the band fixed-point check) |
| decode all, rayon | 22.9 ms | 34.1 ms |
| `open_deep` | 165 ms | 248 ms |

The band costs about 50% more decode time, all of it in the integrity check (recomputing the
band). The rANS decode itself is not slower. For trusted local caches the check could be made
optional; for now it is always on.

## Mountains round 8: distant meadow, grade, stones, spruce (21 Oct 2026)

Images: before `out/quality/steps/21a_base_*` (identical to `19_*`); after `22_final_*`
(`_default`, `_closeup_full`, `_sunlit_full`, and the `_closeup` / `_sunlit` crops = rows
380-1080, the `19_closeup_after` / `19_closeup_sunlit` framing). Steps: 21c (spruce, stones,
sward v1), 21d (streak fix, auto key), 21e (palette, stone albedo), 21f (WB 0.03, impostor test).
Close-up camera: `CAM_X=7799 CAM_Z=-3614 CAM_H=6 CAM_YAW=3.14 CAM_PITCH=-0.1` (+`SUN_EL=0.8`).
`/workspace/r3.sh <prefix>` renders all three.

1. **Mid-distance meadow** (`mountain_material`, quality.slang): procedural sward texture
   bridging blades to terrain shading: three anisotropic value-noise octaves (0.25 / 0.8 /
   2.5 m, stretched ~2.5:1 along a FIXED wind frame) varying tone, straw hue and dark "gaps",
   each faded by pixel footprint (no shimmer, no band), weighted in by `smoothstep(18, 40, dist)`
   under the blades. Plus a `Mat.sheen` / `Mat.lean` term in `shade()`: forward-scatter grazing
   sheen and a view-vs-lean grain modulation (15-45 m fade-in). Pitfall found: rotating the
   texture frame by a spatially varying direction at world coords ~8 km spins the pattern
   into whorls; the frame must be constant (the varying `fl` only drives sheen/grain).
2. **Grade**: the close-up "veil" was the auto exposure pinned at its x2.5 cap (log-avg 0.021),
   not fog/aerial (~1 % at 300 m). Now `AUTO_MAX` 2.0, `AUTO_KEY` 0.09 -> 0.075, and a
   `grade()` before ACES (log contrast `GRADE_K` 0.10, luminance-preserving cool WB
   `GRADE_WB` 0.03; `GRADE=0` disables; cave bypasses it). Meadow palette lost its yellow
   (dry (0.120, 0.158, 0.078), dry patch (0.17, 0.175, 0.112), blade dry tips (1.22, 1.24, 1.05)).
   The default vista (exposure x1.0) changes only by the grade.
3. **Stones**: scattered stones (kind 2) use the faceted boulder meshes with sizes
   `0.3 + 1.1 r^2.2` m, 1.25x taller than wide, half-buried; darker bedrock albedo
   (0.075-0.15), ground-line stain (contact darkening), duller lichen.
4. **Spruce**: `SPRUCE` crown 13 tiers / 11 segs; `card_conifer` adds 9 tiers at LOD0, a
   narrow spire (top 12 %), lower branches sweeping down (`0.70 (1-f)^1.4`), upturned tips,
   pendant side shoots, a leader; darker interior (card self-occlusion). LOD1/2 crowns share
   `Crown`; the impostor atlas is re-baked from LOD0 each run (hash db1dc499fe5366eb).
5. **Impostors earlier: not adopted.** `VEG_IMP_AT` 480 / 350 move only ~1000 of ~60 000 trees
   in these views (most are already impostors or inside 350 m); no measurable gain within
   lavapipe noise. The mid-distance cost is the 60-260 m cone-crown LOD, not the impostor switch.

Timings (lavapipe, 1920x1080 2x2 SSAA, shadow / main / frame ms, single runs on a shared
box, +-5-10 % noise): default 712/1371/2083 -> 643/1344/1987; close-up 759/2054/2813 ->
685/2343/3028; sunlit 666/2054/2720 -> 655/2368/3024. The close-ups cost ~+300 ms main for the
denser spruce LOD0 cards. Weaknesses: stones are still smallish and read grey rather than
strongly faceted at 10 m; sward at 20-40 m is a noise texture (reads as mown/combed, not
clumped blades); close-up is now darker/cooler (exposure capped); spruce LOD0 is more expensive.

## Mountains round 9: foliage prepass, shadow cache, spruce shoot clusters, SH ambient (22 Oct 2026)

Research notes with sources, plus the adopted and rejected techniques: `notes/perf_research.md`.

**Performance**
- **Foliage depth prepass** (`QualityFrame::depth_prepass`, `PREPASS=0` disables):
  - instance meshes are drawn depth-only first (`PreInst` has no FS; `PreInstAlpha` uses
    `instDepthFS`, with the same discards as `instFS`);
  - the terrain then draws normally, and is early-Z rejected behind the vegetation;
  - instances then shade with depth `EQUAL` and no write;
  - default view pixel-identical; close-up 9 px differ (by at most 10/255) from coplanar card ties.
- **Static shadow cascade cache** (`QualityFrame::shadow_cache_keys`, `AshBackend::qshadow`,
  `SHADOW_CACHE=0` disables):
  - `sun_cascade(origin: DVec3, ...)` is world-anchored: the centre snaps to texels in world light
    space, the depth to `CASCADE_DEPTH_SNAP` = 16 m, and the radius is closed-form;
  - c2/c3 are keyed on (cell, slice, res, scene fingerprint, n/f, sun) and are re-rendered only when
    the key changes; c0/c1 always redraw because of camera-distance LOD;
  - this changes the shadow snap vs round 8 (~272k px, max 34/255).
- `QualityTimings::{cached_cascades, camera_mesh_draws}`; `QUALITY_MESH_STATS=1` prints the top
  camera meshes. Mesh ids: trees `sp*3+lod` (0-11), boulders 12-29, grass 30-53, shrubs 54-57,
  impostors 58-61.
- Rejected: impostors from 400 m (main +17%), Hi-Z two-pass, meshlets / compute raster, and VRS.
  See the notes for why.

**Timings** (lavapipe on this box, not real-GPU fps; shadow / main / frame ms, ±5-10 %). "Warm" is
the example's normal run: a warm-up frame first, so c2/c3 hit the cache in every timed run.
"Cold" is `SHADOW_CACHE=0`.

| view | 22_final (round 8) | 23 cold | 23 warm (2 of 4 cascades cached) |
|---|---|---|---|
| default | 643 / 1344 / 1987 | 634 / 1270 / 1904 | 309 / 1204 / 1513 |
| close-up | 685 / 2343 / 3028 | 670 / 1537 / 2208 | 363 / 1592 / 1955 |
| sunlit close-up | 655 / 2368 / 3024 | 610 / 1552 / 2167 | 299 / 1564 / 1862 |
| cave | frame 245 | n/a (spot shadow, uncached) | 21 / 207 / 228, pixel-identical |

GPU-independent counts are unchanged from round 8: no LOD or placement change.

| view | triangles | camera instance draws | camera veg tris | impostor draws |
|---|---|---|---|---|
| default | 4.45 M | 61,277 | 1.13 M | 58,779 |
| close-up | 4.42 M | 48,548 | 2.45 M | 38,148 |

- The prepass rasterises camera instances twice: one depth-only pass, one shading pass.
- Cached cascades skip 0.99 M of 1.84 M (default) and 1.16 M of 2.61 M (close-up) shadow
  instance triangles, plus the c2/c3 terrain.
- The close-up triangle mass is spruce LOD2 (750k tris, 1820 draws) and fir LOD2 (481k), not the
  60-260 m LOD1.

**Visual**
- **Spruce/fir/larch cards (`u_off` 4 → `spruce_card` in `quality.slang`, mirrored by
  `impostor::spruce_alpha`):**
  - shoot-cluster pattern: branch axis plus alternating forward-angled bottle-brush side shoots
    every 0.08, ragged needle-grain edge, per-shoot tone, darker inner shoots;
  - coverage ~0.75, with gaps in every row (test `spruce_card_is_clustered_with_gaps`);
  - once shoots are sub-pixel, the alpha dither tends to 0.9 and the tone to 1;
  - `card_facing_keep` fades edge-on cards, which killed the X artefacts;
  - the impostor atlas re-bakes from LOD0.
- **Ambient:**
  - L2 SH irradiance from the sky-view LUT (`atmoIrradianceCS`, 9 coefficients at `OFF_IRR+2`,
    `ATMO_FLOAT4S` +12);
  - sky above ~10-20° elevation, a sunlit mountainside / forest wall below it (`ENV_SLOPE_ALB`),
    valley floor below 0° (`ENV_GROUND_ALB`, albedo x (sun + sky) / π), which is the ground bounce;
  - sky chroma ×0.35 (`ENV_SKY_SAT`), because the art LUT's zenith irradiance had linear B/R ≈ 3.4;
  - multi-bounce AO (Jimenez 2016);
  - shade grass moved from hue 129° (teal) to 103° (green), with no veil;
  - `ENV_SH=0` gives the legacy ambient. `QUALITY_DEBUG_ENV=1` shows the irradiance (up / toward
    the sun / away from the sun / down quarters, ×4). `GRADE_WB` now defaults to 0.015 (was 0.03,
    measured nearly neutral).
- Images:
  - `out/quality/steps/23_final_{default,closeup,sunlit}.png` (+ `_full`);
  - side-by-side vs 22: `23_vs_22_{default,closeup,sunlit}.png`;
  - shade before/after: `23_shade_lighting_before_after.png`.
- Not done: stone colour/faceting and the combed meadow at 20-40 m.
- Known weaknesses:
  - the deep forest interior at right in the close-up is still near-black (no canopy-transmitted
    light);
  - the denser spruce casts larger shadows, so the sunlit meadow patch left of centre in the
    close-up shrank;
  - the shadow cache misses on any camera rotation.
- Gates:
  - 106 tests pass (104 + `quality_depth_prepass_and_shadow_cache` + spruce card test), also with
    `BRUTAL_SCAN=shared`;
  - clippy -D warnings clean (default + all-features);
  - validation 0 errors on every view;
  - cave pixel-identical to `/tmp/cave_ref.png`;
  - SPIR-V fallback rebuilt (adds `quality_inst_depth_fs.spv`).

## Mountains round 10: forest light, stones, meadow, camera-following shadow cache (22 Oct 2026)

**Forest interior light** (`shade()` in `quality.slang`; live foliage = crown meshes + needle/leaf
cards, not snags):
- Sky-visibility floor, `CANOPY_SKY_FLOOR` 0.08 × (0.3 + 0.7 crowd): a closed stand transmits some
  open-sky diffuse light; a solitary tree keeps its dark interior and tier shading.
- In-crown scattered sun, `CANOPY_SCATTER` 0.05: shadowed foliage gets sun × l.y × (1 − sh),
  green-tinted, stronger toward the crown top.
- Sun flecks: conifer cards stop ~88 % of the sun in the shadow map (`CANOPY_SHADOW_KEEP`
  dither), so crown edges let dappled light through.
- Higher values (0.16 floor, 0.30 scatter, 0.72 keep) were tried and rejected: they flattened and
  yellowed the solitary spruce.
- Close-up: forest at right RGB (3.5, 4.3, 2.4) → (7.1, 12.4, 4.2), now readable dark-green crowns
  and trunk; frame mean 65.9 → 69.4.

**Stones** (`inst_material` boulder branch, `veg.rs::boulder`):
- Cliff palette: warm grey granite/gneiss to brown-grey schist, with a per-block warm/cool strata
  tint.
- Rust streaks on ~25 % of blocks; crustose lichen (grey-green / ochre / a few black) at most
  ~half opacity, never on edges; a damp soil stain at the base.
- Facets read through a per-fracture-plane tone (hash of the flat face normal) and lighter worn
  chamfer edges. The mesh builder flags chamfer triangles with `attr.y + 2`.

**Meadow 20–60 m** (`mountain_material`):
- The fixed 2.5:1 streak frame that read as a combed carpet is gone. Octaves are now mildly
  anisotropic (1.35:1) pairs ~65° apart, or isotropic.
- Added isotropic tussock clumps, dry patches, and sparse late flower speckle (yarrow / hawkbit)
  that fades by footprint.
- The footprint is the ground-plane geometric mean `foot / sqrt(|n·v|)`, so grazing views no
  longer under-filter into horizontal smears.
- Octaves are evaluated only where their footprint weight is non-zero (same image; far terrain
  skips them).

**Camera-following shadow cache** (`render_ash::quality::sun_cascade_clip`; example
`cascades()`; `SHADOW_CLIP=0` = round 9 slices):
- c2/c3 are orthographic squares of half-size R = f/(1 − 1/8) around a centre snapped to a world
  light-space grid of res/8 texels (g ≈ 714 m for c2, 2.6 km for c3; same grid for depth).
- Every point within f of the camera is covered in every direction while the camera stays inside
  the cell. So the key does not depend on view direction and changes only after a move of about g.
  At 16:9 this is smaller than round 9's slice sphere (~1.14 f vs ~1.2 f).
- Casters for c2/c3 are culled against the whole square, so a cached map is complete under any
  rotation. The cost: a cold c2/c3 redraw draws more casters.
- Test `clip_cascade_covers_every_direction_and_keeps_its_cell_while_walking`: coverage on a 1 km
  walk, same cell ⇒ same world map, cell changes ≲ 1 per g.
- c0/c1 still fit the frustum slice and redraw every frame (camera-distance LOD).

**Camera-path benchmark**:
- `CAM_PATH=<frames>`: walk 5 m/s while turning 45°/s, 30 fps steps (`PATH_SPEED`, `PATH_TURN`).
  The eye moves via the view matrix over the scene built at the origin; shading distances stay
  origin-based, so it is a cost / cache benchmark, not a reference image.
- It also prints a CPU-only key simulation of `PATH_SIM` frames for both modes.
- Default view, 60 frames, lavapipe, measured before the restart and before the octave gating:

| mode | mean frame ms | mean shadow ms | far cascades cached |
|---|---|---|---|
| round 10 clip, warm | 1405 | 293 | 120 / 120 (100 %) |
| round 9 slice, warm | 1694 | 574 | 2 / 120 (1.7 %) |
| cold (`SHADOW_CACHE=0`) | 1910 | 800 | 0 % |

- CPU simulation, c2 / c3 hit rate:
  - 600 frames walk + turn: round 9 0 % / 0 %, clip 100 % / 100 %;
  - 9000 frames (1.5 km) straight walk: round 9 92.2 % / 97.3 %, clip 100 % / 100 % (a couple of
    crossings in 9000 frames).

**Timings** (lavapipe = CPU rasteriser, **not real-GPU fps**). After the 14:42 machine restart the
box runs ~15 % slower than this morning, so 23 was re-rendered from cp52 in the same session.
Values are shadow / main / frame ms, ±5–10 % noise (one cold run spiked to 4 s).

| view | 23 warm (round 9) | 24 warm | 23 cold | 24 cold |
|---|---|---|---|---|
| default | 352 / 1397 / 1748 | 335 / 1323 / 1658 | 820 / 1366 / 2186 | 836 / 1264 / 2101 |
| close-up | 393 / 1739 / 2133 | 344 / 1656 / 2000 | 746 / 1728 / 2474 | 1034 / 1703 / 2737 |
| sunlit | 330 / 1954 / 2284 | 330 / 1669 / 1999 | 724 / 1692 / 2416 | 824 / 1726 / 2550 |
| cave | | 21 / 208 / 228, pixel-identical to `/tmp/cave_ref.png` | | |

- A cold round-10 frame is 0–10 % slower: c2/c3 draw casters all around the camera. Warm frames
  are ≈ round 9 for a static camera, and on a moving or turning camera the clip cache saves
  ~280 ms of shadow per frame vs round 9.
- Triangle / instance counts are unchanged from round 9 (no LOD or placement change).

**Images**:
- `out/quality/steps/24_final_{default,closeup,sunlit}.png` (+ `_full`);
- `24_vs_23_{default,closeup,sunlit}.png` (left 23, right 24);
- `24_forest_light_before_after.png`.

**Weaknesses**:
- Auto exposure (a log-average) is dominated by the dark forest. Brightening the forest lowered
  the sunlit close-up exposure ×1.55 → ×1.14, so that frame is ~10 % darker overall (mean 103 →
  92) even though its shade and forest are brighter. A trimmed / percentile meter would fix it;
  not done.
- The close-up still hits the ×2.0 exposure ceiling. The sunlit meadow patch did not grow back.
- Stones in direct sun can read slightly beige.
- The 20–60 m sward still shows soft large-scale smudges, though not a combed grain; the flower
  speckle is a texture, not geometry.
- The path benchmark moves the eye over a scene built for one camera: the far rings and
  vegetation exist only in the build-time FOV wedge.

**Gates**: tests (+ `BRUTAL_SCAN=shared`), clippy -D warnings (default + all-features), validation 0
errors on every view, cave pixel-identical, SPIR-V fallback regenerated with slangc 2026.20.
Toolchain reinstalled after the restart: rustup 1.99.1; apt `mesa-vulkan-drivers`
`vulkan-validationlayers` `vulkan-tools` `spirv-tools`; slang 2026.20 in `/opt/slang`, symlinked
from `/usr/local/bin/slangc`.

## Mountains round 10b: the distorted outcrop (torn tor geometry) (22 Oct 2026)

**Symptom.** In the close-up (and in round 9) the outcrop ~120 m ahead showed a hollow, scooped face and in front of it a field of sharp teeth shaded with meadow colour. `VEG=0` kept it, so it was terrain. The cause was in the voxel near field (the `Alpine` tor term), not in boulders, scree or cards.

**Root cause.** The tor joint frame was a rotation about the *world origin* by a spatially varying strike, `u = R(th(p)) * p` with `th = 0.6 + 0.5 perlin(p / 600)`. Its derivative contains `|p| * grad(th)`. About 8.7 km from the origin the 7.5 m and 10 m joints were squeezed to roughly 1 m, which is sub-cell. The dip term `y + dip(p) * u0` also tilted the bedding by up to about 75°. Every metre was then a new joint block with its own 0–11 m lift (`TOR_BLOCK`), and the mesher turned that into comb teeth. Near the origin the frame is nearly right, which is why tors there looked fine.

I proved it by ablation (`TOR_DBG` bitmask, removed afterwards): setting block→smooth removed the teeth, while crack-off and step-off did not. A surface-height profile through the outcrop jumped 0 ↔ 6 m every 0.5 m.

Secondary true discontinuities, found by the new bisection probe:
- the block id was `round(q)`, so it switched halfway between cracks;
- the step window faded with the LOD;
- at cells ≥ 2.2 m the `lift()` lower edge went negative, cut off by the `t <= 0` early return;
- blocks kept their lift where the mask reached 0;
- the steep-rock strata ledge was a 1.6 m sawtooth.

**Fix (at source, `crates/voxel/src/fields.rs`).**
- The joint frame (strike, dip) is fixed per 500 m domain (`TOR_DOM`), in coordinates relative to the domain anchor. On each boundary it is blended 50/50 with the neighbour domain over `TOR_DOM_BAND` = 40 m, so most points evaluate a single frame (`tor_parts` → `tor_frame`).
- Block id is `floor(q)`. Per-block lift and step are blended 50/50 with the neighbouring block on each joint plane, so the field is continuous and a crack never stands above both blocks it separates.
- The bedding step is windowed to 0 on every plane.
- Lift lower edge `max(0.3 - sharp, 0)`; block lift times `smoothstep(0, 0.1, t)`.
- The ledge underside drops back over at least 0.42 m (`e = clamp(1.5 cell / 7, 0.06, 0.35)`).

**Regression tests.**
- `voxel/tests/tor_continuity.rs`: random walks in outcrop and steep-rock columns, locally and across a ±12 km world square, at 9 LOD cells (0.25–8 m). Each 10 cm segment is bisected 36 times; a change above 5 mm at about 1e-12 m is a jump. Results: 2659 jumps on the cp60 field (up to 12 m), 0 now (worst residual 5e-11 m).
- `voxel/tests/tor_mesh_spikes.rs`: CPU-meshes the reported outcrop and a far site at 0.5, 1 and 2 m. It asserts convex folded triangles stay at or below 0.15 %. Results: cp60 2.2–3.6 % at lod 0/1 with 3× the triangles; now at most 0.09 %.
- `voxel::meshq` (new, unit-tested) is the spike metric. `fold` = a convex edge whose adjacent normals are more than 120° apart (knife-edge fin or tooth). `crease` = the same but concave (V-slot). `needle` = lmax² / 2A > 25.
- The example prints the scan with `QUALITY_SPIKES=1` (bands by distance from the camera, plus `SPIKE_AT=x,z,r`). `/workspace/spikescan.sh <bin> <tag>` runs 5 cameras.
- `asset` a12: the alpine case only requires that transitions are exercised (at least 1; the count is terrain luck). Synthetic scenes still need at least 3.

**Spike scan (GPU near field, convex folds, whole mesh / within 60 m of the camera), cp60 → now.**

| view | folds | folds < 60 m | tris |
|---|---|---|---|
| default | 114 → 0 | 0 → 0 | 1.603 M → 1.595 M |
| close-up | 1232 → 0 (outcrop disc r 60: 628 → 0) | 376 → 0 | 1.476 M → 1.446 M |
| origin (0, 0) | 36 → 14 | 0 → 12 | 1.429 M → 1.426 M |
| west (−4719, −8843) | 12903 → 48 | 10876 → 44 | 2.082 M → 1.841 M |
| crags (8090, −822) | 46 → 34 | 7 → 19 | 2.195 M → 2.191 M |

What remains (about 0.02 % of triangles near the camera) is sharp corners of real joint blocks at 0.25–0.5 m cells, not teeth. Near the origin the joint frame changed (domain-anchored), so there are 12 new block-corner folds where there were none.

**Images.** `out/quality/steps/25_final_{default,closeup,sunlit}(_full).png` and `25_rock_before_after.png` (2× crop of (60, 330)–(700, 650), round 10 vs fixed). The round-10 re-render with the cp60 field is pixel-identical to `24_final`. Versus 24_final, pixels changed by more than 20: default 0.6 %, close-up and sunlit 2.2 %, all on outcrops. The cave is pixel-identical.

**Timings (lavapipe, same session; not real-GPU numbers).** GPU frame ms, round 10 → now: default 1585 → 1563, close-up 1886 → 1914, sunlit 2071 → 1814. Lavapipe frame-to-frame noise is about ±10 %. CPU field sampling for the close-up: 5569 → 5607 ms (+0.7 %; the domain blend evaluates 2–4 frames only in the 40 m bands).

## Mountains round 11: cheaper far trees (LOD2 260–650 m, impostor path) (22 Oct 2026)

**Adopted (both on by default).**
1. **1-segment LOD2 crown** (`veg.rs`: `shoot_n(segs)`, `card_conifer(detail = 2)` uses `lod2_segs()`, default 1, env `VEG_LOD2_SEGS=1..3`). Each LOD2 card is one wide quad instead of a 3-segment bent strip. Spruce LOD2 goes from about 412 to about 156 tris per instance. The silhouette, card count and the instance/LOD/crossfade logic don't change, so coverage stays the same.
2. **Card-depth impostors in the foliage depth prepass** (`QualityFrame::impostor_card_depth`, default on; `IMP_CARD=0` restores the round-10 path). New `impDepthFS` does depth only: `lod_keep` dither plus the `imp_trace` coverage test, writing card depth. New `impEqFS` shades with depth `EQUAL`, no depth write and no discard. New pipelines are `Kind::PreImp` / `Kind::ImpEq`, and `impFarFS` now also honours `lod_keep`. This removes the `SV_Depth` late-Z overdraw of `impFS`. Impostors write card depth from the switch distance on; the 650 m switch and 140 m crossfade are unchanged.
- Regression test `render-ash/tests/gpu.rs::quality_card_depth_impostor_prepass`: a cone forest becomes impostors at 30 m. With card depth, prepass ON and OFF give the same image, and coverage matches the exact-depth path.

**Metrics (GPU-independent, round 10 → 11).** Camera veg triangles: default 1.131 M → 0.770 M (−32 %); close-up and sunlit 2.454 M → 1.669 M (−32 %). The following are unchanged: instances (61278 / 48550), impostor instances (58779 / 38148), crossfade instances (2609 / 11571) and draw calls (one indirect draw per mesh per pass, so (c) has nothing to merge).

**Timings (lavapipe, same session, median GPU frame ms; not real-GPU fps; ±10 % noise).**

| view | warm r10 → r11 | cold (SHADOW_CACHE=0) r10 → r11 | main instance meshes (warm) | main impostors (warm) |
|---|---|---|---|---|
| default | 1569 → 1125 (−28 %) | 2341 → 1668 (−29 %) | 424 → 142 | 166 → 56 |
| close-up | 1948 → 1409 (−28 %) | 2499 → 2026 (−19 %) | 826 → 357 | 11 → 6 |
| sunlit | 2013 → 1400 (−30 %) | 2387 → 1855 (−22 %) | 832 → 402 | 10 → 6 |
| cave | unaffected (no trees), pixel-identical | | | |

**Measured and rejected.**
- 2-segment LOD2: close-up 2.06 M tris and 1792 ms, against 1697 ms for 1 segment, which looks the same.
- Earlier impostor switch with the exact-depth impostor FS: 260 m gives 2040 ms close-up and 400 m gives 2010 ms; main gets slower although tris drop to 1.22–1.36 M. 1-segment + 400 m without the card prepass: default 1690 ms (impostor FS 314 ms).
- **400 m switch with the new card-depth prepass**: close-up 1355 vs 1537 ms (−12 %) and 1.256 M tris; default is the same (1137 vs 1113). **Rejected on images.** The 300–450 m stands turn visibly flatter and lighter (pixels above 32: 1.2 %), see `out/quality/steps/26_rejected_imp400_vs_650.png`.
- (c) instance merging / cluster draws: draws are already one indirect draw per mesh per pass, and the per-instance cost is vertex work that LOD2 already cut.
- (d) distance thinning plus scale-up: not built. With (a) and (b), LOD2 is no longer the bottleneck. Thinning changes the canopy texture, which risks Helen's bar on "thinner", for little remaining win.
- Not tried: single-frame or 2-frame impostor blend, no-parallax, lower-res atlas mips. After the prepass the impostor shading cost is only 6–56 ms.

**Visual check.** Versus round 10 (same session): mean |Δ| is 0.16 (default), 0.30 (close-up) and 0.37 (sunlit). Pixels with Δ > 8: 0.33 / 0.97 / 0.94 %. Mean luminance is equal within 0.06. The changed pixels are slightly darker, not brighter, so there is no thinning. They are card-edge noise across the far stands. Camera sweep +0/+30/+60/+90 m: the step-to-step image change is identical old vs new (39.4 / 48.1 / 37.1), so the crossfade adds no new popping (`26_lod_sweep.png`, old/new row pairs).

**Images.** `out/quality/steps/26_final_{default,closeup,sunlit}(_full).png`, `26_vs_25_{default,closeup,sunlit}.png`, `26_far_forest_before_after.png` (2× crop of (0, 280)–(800, 560) of the close-up: round 10 / round 11 / |Δ|×8), `26_lod_sweep.png`.

**Checks.** The SPIR-V fallback was rebuilt with `tools/compile_shaders.sh`, adding `quality_imp_depth_fs` and `quality_imp_eq_fs`; it is byte-identical to the build.rs slangc output. `cargo test --release --workspace`: 111 passed, also with `BRUTAL_SCAN=shared`. Clippy `-D warnings` is clean (default and all-features). Validation: 0 errors. The cave is pixel-identical to `/tmp/cave_ref.png`. The spike scan (`spikescan.sh`, 5 views) is identical to round 10b.

**Weaknesses.** At LOD2 the 1-segment cards are flat, so the droop curve of each shoot is lost. At 260 m and beyond this was not visible in the checked views, but a long lens or higher resolution could show flatter crowns right at the LOD1→2 switch. Card-depth impostors intersect terrain and each other at card planes, not per pixel; this was invisible at ≥ 650 m in the checked views. The prepass made the depth prepass itself a little more expensive in the default view (268 → 333 ms), and it is now the biggest instance pass there.

## Earth round 1: Earth from orbit (cube-sphere quadtree, NASA data, atmosphere from space) (22 Oct 2026)

**Images** (`out/quality/steps/`, 1920×1080, rendered at 2× supersampling and box-filtered in linear light):
- `E1_orbit_day.png`: full disk from 35,000 km over Africa, mid-July sun 13° off the view axis.
- `E1_orbit_iss.png`: 410 km over the eastern Atlantic, looking NE at the Moroccan coast and the limb.
- `E1_terminator.png`: night over Italy looking WNW at the sunset line, with the sun 13° below the visible horizon. It shows city lights, the twilight band, the 95 km airglow line and BSC5 stars.
- `E1_glint.png`: sun glint with Maui and the Big Island silhouetted in it; morning sun 35° up.
- `E1_vs_old_earth.png`: the old `earth` scene frame next to `E1_orbit_day`.
- `E1_vs_epic_reference.png`: `E1_orbit_day` next to a real DSCOVR EPIC frame of the same hemisphere and season (2026-07-15 11:07 UTC). This was the calibration target.

**Run.**
```
tools/fetch_earth_data.sh /workspace/data/earth            # sources + SHA256SUMS + mask.png (not committed)
cargo run --release -p planet --example earth_bake -- /workspace/data/earth /workspace/data/earth/earth.pln
cargo run --release -p render-ash --example earth -- --shot all --out out/quality/steps   # [--size WxH] [--ss N]
```
The `earth` example takes `--shot day|iss|terminator|glint|all` and `--pack`.

Env overrides:
- `EXPOSURE`, `SUN_LAT`/`SUN_LON`;
- `LOD_PX` (default 1.5 px);
- `CLOUD_KM`, `CLOUD_GAMMA`, `CLOUD_SHADOW`, `WIND` (m/s);
- `MIE`, `GROUND_ALB`, `ALBEDO`, `NIGHT`, `AIRGLOW`, `RELIEF`;
- `REPEAT=n` (reports cold and warm GPU ms);
- `PROBE="x,y;..."` prints linear radiance in units of solar illuminance;
- `DEBUG`: 1 albedo, 2 water/ice, 3 slope, 4 aux, 5 night, -1 atmosphere only.

Each shot also writes a JSON with patches, triangles, vertex MB, LOD level histogram, texture MB, GPU ms breakdown, stars drawn and validation errors.

### What was built
1. **`crates/planet` (geometry, CPU, f64).**
   - **Cube-sphere quadtree.** Integer lattice with N=32 quads per patch and 22 levels. An exact odd tangent warp makes shared face-edge vertices bit-identical.
   - **Neighbours.** Neighbour keys fold across cube edges in integer arithmetic.
   - **Selection.** Screen-space error: the chord sagitta s²/8R plus a relief term, over distance, times px/rad. Default 1.5 px at output resolution.
   - **Culling.** Conservative horizon/limb cone culling plus frustum culling against bounding spheres. Culled nodes stay leaves, so the tiling is always complete.
   - **2:1 balance.** A loop after selection enforces it.
   - **Stitching.** Index-space stitching, 16 index variants selected by a 4-bit coarser-neighbour mask, plus skirts (max(0.25·spacing, 20 m)) as a second line of defence.
   - **Precision.** Vertices are f32 offsets from an f64 patch origin. Origin minus camera is formed in f64 on the CPU, so no f32 ever holds an absolute planet coordinate.
   - **Pack format.** `PLN1` uses the asset::fb container, verified and fuzz-truncation safe. It holds BC cube maps with full mip chains plus a u16 height cube for the CPU displacement.
2. **Baked data** (`earth_bake`, 69 s, 364 MB pack). Vulkan cube convention; planet frame +Z north, +X at lon 0.

   | texture | format | face size | contents | MB |
   |---|---|---|---|---|
   | albedo | BC7 sRGB | 4096² | BMNG July reflectance; alpha = water fraction | 127 |
   | slope | BC5 | 4096² | east/north slope, from GEBCO_08 topography | 127 |
   | night | BC4 | 4096² | Black Marble 2016 | 63 |
   | aux | BC7 | 2048² | R clouds, G ice, B ocean depth | 31 |
   | height cube | u16 (CPU) | 1024² | displacement | 12 |

   - **Encoding.** intel_tex_2 (ISPC). BC7 avoids BC1 banding on the dark ocean.
   - **Water mask.** Natural Earth land minus lakes (`tools/earth_mask.py`), because the bathymetry cannot separate shallow sea from land.
   - **Ice.** Natural Earth glaciers and ice shelves, plus BMNG sea-ice brightness.
3. **`render-ash/src/planet.rs` + `shaders/planet.slang`.** A separate renderer from the mountains `quality` path; mountains and cave are untouched.
   - **Draws.** All patch vertices sit in one buffer. One `vkCmdDrawIndexedIndirect` command per patch selects the stitch variant. The patch index is `vertex_id / VERTS` (vertexOffset = patch·VERTS).
   - **Target.** RGBA16F colour + D32 reverse-Z. Combined image samplers, anisotropic when available. Enables BC and anisotropy if the device supports them (`ctx.texture_bc`, `ctx.max_anisotropy`).
   - **Atmosphere from space (R-E1/E2/E4).**
     - LUTs: Hillaire transmittance, multiple scattering, sky-from-ground and irradiance.
     - Every pixel ray-marches the atmosphere segment it crosses, split at the perigee with quadratic steps, so the camera can be outside the atmosphere.
     - Altitude comes from the f64 camera.
     - Stable sphere intersection (R-E2).
     - Planet shadow (R-E4): each sample's sun visibility goes soft against its own horizon, giving terminator reddening and dark rays through the shadow.
     - Ozone, and an airglow layer at 95 km integrated on its own sub-march.
   - **Ocean.**
     - Cook-Torrance glint: Cox-Munk Beckmann with m² = 0.003 + 0.00512·W, Smith G and Fresnel F0 = 0.02.
     - Mesoscale wind variability (40–150 km noise on m²).
     - Sky reflection from the sky-from-ground LUT.
     - Water-leaving reflectance (0.0022, 0.0075, 0.026), keeping BMNG shallows where brighter.
     - Sea ice from the aux mask and BMNG brightness.
   - **Clouds.**
     - 2D layer at R + 6 km from the Blue Marble cloud map, as nadir coverage → optical depth → opacity 1 − exp(−τ/|μ|).
     - Lit by the sun transmittance at cloud height, with an HG forward lobe and sky ambient.
     - Sub-texel procedural detail plus relief normals, faded in only where the pixel footprint resolves them.
     - Ground shadows taken where the sun ray crosses the layer.
     - Limb clouds composited in the sky pass.
   - **Night.** Black Marble at its physical brightness: L/E = 1.5e-6 · v^2.2 per sr, sodium/LED colour, dimmed by clouds overhead.
   - **Output.** The shader pre-exposes before writing (see bugs). Alpha = transmittance toward space.
4. **Stars, sun, exposure.**
   - Yale BSC5 stars are splatted on the CPU at physical flux 10^(−0.4(V+26.74)) relative to the sun, bilinear over 2×2 pixels, weighted by the sky transmittance. So they are invisible at daylight exposure and appear at night exposure, with nothing faked.
   - Sun disc is limb-darkened.
   - Fixed per-shot exposure and ACES. Daylight shots use 2.4–2.6, calibrated against EPIC. The night shot uses 2e5, a long night-photo exposure. There is no auto-exposure and no veil.

### Calibration (physics against data, not taste)
- **Aerosol.** The Hillaire default aerosol has AOD ≈ 0.005, which made the ocean too saturated and the red channel ~3× too low against MODIS TOA. Now AOD ≈ 0.1 (Mie ×22) with g = 0.75.
- **Rayleigh and ozone.** Coefficients are moved from Hillaire's monochromatic 680/550/440 nm to the sRGB primaries' effective 615/550/465 nm: Rayleigh β ∝ λ⁻⁴, giving (8.67, 13.56, 26.6)e-3/km; ozone (Chappuis) (2.74, 1.88, 0.17)e-3/km. That removed a lilac limb and a pink Sahara.
- **Probes.** Ocean blue path reflectance π·L ≈ 0.10 at nadir (MODIS 469 nm ≈ 0.1). Sahara 0.42 / 0.24 red/blue TOA (MODIS ≈ 0.45 / 0.25–0.3).
- **Cloud mapping.** Nadir coverage a_n = ((c − 0.04)/0.92)^1.6. The first mapping made mid-grey clouds opaque.
- **Airglow gain.** 1.2e-9: a visible limb line without a green veil seen straight down.

### Bugs found and fixed (root causes, with regression tests)
| symptom | root cause | fix | guard |
|---|---|---|---|
| shattered globe | VS read `SV_InstanceID` (always 0 with one indirect command per patch) for the patch origin | patch = `vertex_id / VERTS` | `tests/planet_gpu.rs` (disk area and quadrant balance; mutation fails it) |
| grid over the disk | shading used the rasterised chord point, up to s²/8R ≈ 2 km below the sphere for level-0 patches, which changed the dense-air path | re-intersect the view ray with the sphere through the interpolated height | — (visual; LOD-independent by construction) |
| black dot at nadir | `pow(1 − nv, 5)` with nv = saturate()+1e-4 > 1 gives NaN | `max(saturate(), 1e-4)` | GPU test: all outputs finite with exact nadir in view |
| dotted seam on cube-face edges | lavapipe forms implicit cube-map derivatives from per-lane face (s,t), so quads straddling faces fetch a far-too-coarse mip | `SampleGrad` with screen derivatives of the (continuous) direction | — (visual; costs ~3× surface-pass time on lavapipe) |
| banded, green night ground | night radiances ~1e-6 fell into RGBA16F subnormals (step 6e-8) | pre-exposure (`params.view.z`) before the write, clamp 6e4 | GPU test: city radiance within 10 % with colour ratios (mutation fails it) |
| concentric airglow bands | 10 km layer sampled by 16 march steps of tens of km | separate 24-step integration over the shell crossing(s) | — |

### Numbers (lavapipe = llvmpipe CPU rasteriser on this shared 8-vCPU box; **not real-GPU fps**)
Render size 3840×2160 (2× SS of 1080p). GPU-independent counts are in the first block of columns.

| shot | alt km | patches (leaves) | LOD levels | triangles | vertex MB | GPU ms cold / warm | LUTs | surface | sky | readback |
|---|---|---|---|---|---|---|---|---|---|---|
| orbit_day | 35,000 | 5 (6) | L0 ×5 | 11,520 | 0.1 | 2393 / 2326 | 12 | 2271 | 0.7 | 42 |
| orbit_iss | 410 | 19 (72) | L2–5 | 42,864 | 0.4 | 3021 / 3082 | 12 | 3023 | 1.8 | 46 |
| terminator | 420 | 20 (66) | L2–5 | 44,976 | 0.4 | 1955 / 2076 | 10 | 2025 | 1.7 | 39 |
| glint | 420 | 24 (66) | L2–5 | 54,048 | 0.4 | 3668 / 3378 | 12 | 3328 | 0.7 | 37 |

- **Texture memory.** 352 MB of GPU textures (BC, with mips) plus a 12 MB CPU height cube.
- **CPU.** Selection 0.06–0.2 ms; patch mesh build 0.8 ms for 5–24 patches (rayon).
- **Cost.** The surface pass is per-pixel fragment work: a 16–24-step atmosphere march, cloud detail noise and explicit-gradient cube fetches. On lavapipe it scales with pixels covered, not triangles. A real GPU should be orders of magnitude faster, but that is not measured here.

### Checks
- **Tests.** `cargo test --release --workspace`: 121 passed, also with `BRUTAL_SCAN=shared`. That includes:
  - the 9 planet geometry tests (adversarial cameras at cube corners/edges, poles, 2 m grazing, 400 km and GEO; 2:1 property; bit-exact shared edges at all 16 masks; whole-sphere watertight with Euler characteristic 2; conservative culling; f64 camera-relative precision: 2.65 cm at 400 km vs 0.40 m naive f32, 0.2 mm at 2 m);
  - the new `planet_gpu` test.
- **Clippy.** `-D warnings` clean, default and all-features.
- **Validation.** 0 errors in all four shots and in the tests.
- **SPIR-V.** The fallback was rebuilt with `tools/compile_shaders.sh`, adding the 8 planet entries. They are byte-identical to the build.rs output; existing SPIR-V is unchanged.
- **Identity.** The cave is byte-identical to `/tmp/cave_ref.png`. Mountains default, closeup(_full) and sunlit(_full) are pixel-identical to `26_final_*`.

### Data sources and licences (data lives outside the repo; `tools/fetch_earth_data.sh` fetches it and writes SHA256SUMS)
| data | source | licence / credit |
|---|---|---|
| Blue Marble Next Generation, July 2004, 21600×10800, no relief shading | NASA Earth Observatory / Visible Earth (R. Stöckli, E. Vermote, N. Saleous, R. Simmon, D. Herring) | NASA imagery, not copyrighted; credit NASA Earth Observatory |
| Topography 0–6400 m and bathymetry, 21600×10800 | NASA Earth Observatory (Jesse Allen), derived from GEBCO_08 (BODC) | NASA EO credit; GEBCO data are free to use with attribution to the GEBCO Compilation Group |
| Black Marble 2016 night lights, grey 3 km | NASA Earth Observatory, Suomi NPP VIIRS DNB (M. Román, NASA GSFC) | NASA imagery, not copyrighted; credit NASA Earth Observatory |
| Blue Marble clouds 8192×4096 | NASA GSFC (R. Stöckli), MODIS | NASA imagery, not copyrighted |
| Land, lakes, glaciated areas, Antarctic ice shelves 1:10m | Natural Earth | public domain |
| Yale Bright Star Catalogue, 5th rev. ed. (V/50) | Hoffleit & Warren 1991, via CDS Strasbourg | freely distributed by CDS; cite Hoffleit & Warren (1991) |
| DSCOVR EPIC natural colour 2026-07-15 (comparison only, `data/earth/ref/`) | NASA EPIC team, epic.gsfc.nasa.gov | NASA imagery, not copyrighted |

The pack embeds `planet::bake::CREDITS`. None of the data is in the source tarball.

### Honest weaknesses
- **Resolution at ISS altitude is data-limited.** BMNG is 1.85 km/px at source (2.4 km/texel baked) and clouds ~5 km. At 400 km the foreground ground and islands are soft. Procedural cloud detail hides it for clouds only; there is no ground detail synthesis yet. A tiled 500 m BMNG plus virtual texturing is the next step.
- **Clouds.**
  - The layer is a 2D shell with relief normals: there is no volume, no true self-shadowing, no cloud-side silhouettes at the limb, and a single height (6 km).
  - The map is one static NASA composite, so it doesn't match the date.
  - Cloud shadows are physically placed but subtle.
- **Terminator shot.** At the night exposure the twilight band is near saturation and coloured only at its edges, as a real camera would be. With the sun 13° down it is a thin band. A day/night terminator line on the ground with both sides well exposed would need HDR compositing (not done, by design).
- **Air colour.** The Rayleigh/ozone band shift and AOD 0.1 are justified approximations, not a spectral render. The aerosol type is one global constant (no dust over the Sahara, no maritime/continental variation).
- **Geometry pipeline.** Patch meshes are built on the CPU each frame (cheap: under 1 ms). There is no GPU vertex/cull pass and no mesh caching across frames yet. Relief is displacement plus a slope-texture normal; from orbit that is invisible except via shading.
- **Performance.**
  - lavapipe only: 2–3.7 s per 4K-SS frame, dominated by per-pixel atmosphere and cloud work.
  - `SampleGrad` is ~3× slower than implicit sampling on lavapipe; a real GPU likely does not need it, but that is untested.
  - No temporal reuse and no aerial-perspective LUT froxels for the surface.
- **Not wired in.** Integration into the coords flythrough (`earth` scene in `scenes`) and the Bevy app is not done; this is an offline-capable example renderer.
- **Exposure.** Hand-set per shot (calibrated), not metered.

## Earth round 2: volumetric clouds, VIIRS coverage, KTX2, sunset line (23 Oct 2026)

**Images** (`out/quality/steps/`, 1920×1080 unless noted, rendered at 2× supersampling and box-filtered in linear light; every `.png` has a matching `.json` with patches, GPU ms breakdown, texture load stats, and shot params; timings are lavapipe and say so):
- `E2_orbit_day.png`: full disk from 35,000 km over Africa; mid-July sun.
- `E2_orbit_iss.png`: ISS 410 km over the eastern Atlantic looking NE.
- `E2_iss_convection.png`: ISS over the Sahel looking at the MCS towers of the dated VIIRS map (~13°N 1°W).
- `E2_terminator.png`: night over Italy looking WNW; sun 55° left of the view so the twilight arc glows at the left limb and the thin green airglow line runs the dark limb; cities and BSC5 stars.
- `E2_sunset.png`: telephoto (6°) on the limb toward the sun, 3.6° below the horizon at the tangent: red-orange troposphere with cloud silhouettes, yellow-white, blue Rayleigh/ozone.
- `E2_glint.png`: sun glint with Maui / Big Island silhouettes.
- `E2_low_alps.png`: 120 km over the western Alps (procedural sub-texel land detail engaged).
- `E2_vs_epic.png` + `E2_vs_epic_reference.png`: exact DSCOVR EPIC geometry of `epic_1b_20260715111224` (2026-07-15 11:07:35 UTC) next to the real EPIC frame.
- `E2_vs_E1.png`: E1 vs E2 for day / ISS / terminator / glint.
- `E2_ktx2_before_after.png`: raw RGBA8 vs KTX2 BC7/BC5/BC4 (+ zstd) crops at the same exposure, plus |diff|×8.

**Run.**
```
tools/fetch_earth_data.sh /workspace/data/earth   # BMNG + Black Marble + VIIRS GIBS tiles + earth_clouds.py
cargo run --release -p planet --example earth_bake -- /workspace/data/earth /workspace/data/earth/earth.pln
# also writes data/earth/ktx2/{albedo,slope,night,clouds,masks}.ktx2 (+ ktx2_nosc, ktx2_raw variants)
cargo run --release -p render-ash --example earth -- --shot all --out out/quality/steps
```
`--shot` accepts any of the `E2_*` names above (or `all`).

Env overrides (in addition to E1's):
- `EARTH_TEX=ktx2|nosc|raw|pln` — texture path (default `ktx2`). `pln` is the old PLN1 pack (pre-VIIRS aux clouds). `raw` is uncompressed RGBA8/RG8/R8. `nosc` is KTX2 without zstd.
- `EARTH_CLOUDS=vol|2d` — volumetric ray march (default) or the E1 2D shell.
- `CLOUD_BASE`, `CLOUD_TOP`, `CLOUD_EXT`, `CLOUD_STEPS`, `CLOUD_SUN_STEPS`, `CLOUD_NOISE_KM`, `CLOUD_MS` — volume model.
- `BAKE_CLOUD_SRC=viirs|bmng` — which coverage map the baker uses (default viirs if `clouds_viirs.png` exists).
- `AERO_FT`, `AERO_H`, `AERO_STRAT` — free-troposphere / Junge-layer aerosols (for the red-orange low limb).
- `EARTH_DETAIL` — procedural sub-texel land-detail gain below a ~3.5 km footprint (default 1; 0 = off).
- `AIRGLOW` — default now 5e-10 (was 1.2e-9) so the green line stays thin and faint.

### Was KTX2 considered before?
No. Searching the pre-round-2 tree (`cpE20_start`: `starter.md`, `notes/perf_research.md`, and the whole source tree excluding what this round added) finds zero mentions of KTX2, Basis/UASTC, BCn container formats, or zstd supercompression. Round 1 shipped BC7/BC5/BC4 inside the custom PLN1 pack only.

### What was built
1. **VIIRS GIBS coverage + cloud-top height (same day as the EPIC reference).**
   - `tools/fetch_earth_data.sh` pulls NOAA-20 VIIRS Corrected Reflectance True Colour and `VIIRS_NOAA20_Cloud_Top_Height_Day` for `VIIRS_DATE` (default 2026-07-15), plus the MODIS/VIIRS CTH colormap.
   - `tools/earth_clouds.py` builds `clouds_viirs.png` (coverage in the BMNG convention `s = 0.04 + 0.92·c^(1/1.6)`) and `cth_viirs.png` (km×20 in 8-bit). Coverage: Gumley-unstretched blue reflectance above the linearised BMNG July clear sky, times a whiteness term; glint/haze removed by a local-floor subtraction; no-data filled from BMNG clouds. CTH: exact colormap decode, push-pull fill. Area-weighted mean coverage ≈ 0.25.
   - Baker (`crates/planet`): clouds is now BC5 (R = coverage, G = top km×20/255, 0 = unknown). `planet.clouds_source` KVD on the KTX2. Credits name NASA EOSDIS GIBS / Worldview.
2. **Volumetric clouds (`planet.slang`).**
   - Columns from the map: coverage `c` = nadir opacity; top from CTH (or `base + (default_top − base)·c^1.5` when unknown); thin cover becomes a sheet under its top.
   - Two density regimes with the same mean column opacity `τ_c = −ln(1−0.995c) + 3·smoothstep(0.7,1,c)`: uniform when the footprint is large; patchy solid cells (area ≈ c) from two rotated Perlin-Worley octaves when close, with turret heights and Worley erosion that fades out by a 0.35 km footprint. LOD seam: smoothstep 0.6–2.5 km.
   - Energy-conserving march (coarse grid jittered per pixel + up to 4 substeps of ≤0.15 km with a per-substep column lookup); sun march with doubling steps; 3 multi-scatter octaves; dual-lobe HG; sky ambient + ground bounce.
   - Atmosphere split at the first cloud sample: `L = L1 + T1·(Lc + Tc·(L2 + T2·Ls))`. Ground shadows: 8-step sun march. Limb chord in `skyFS`.
3. **Sunset line.**
   - Green airglow: σ = 2.2 km below / 3.5 km above 95 km (FWHM ~7 km), gain 5e-10 → a thin faint line, not a band. Optional faint OI 630 nm red fringe at 250 km.
   - Free-tropospheric + Junge-layer aerosols (`aero`) so the low troposphere at the limb goes red-orange under a grazing sun. `E2_sunset` is the telephoto that shows the layers; `E2_terminator` keeps the night exposure with the twilight arc off-azimuth.
4. **KTX2 path (`crates/planet/src/ktx2.rs`).**
   - Own writer/parser; zstd scheme 2; DFD details; Khronos validator warning 7010 accepted. BCn: BC7 colour, BC5 normals + clouds, BC4 height/masks. Loaded directly via ash (`EARTH_TEX`).
   - Layout regression test: `PlanetParamsGpu` field order matches Slang `PParams` (including the new `aero`).
5. **Procedural sub-texel land detail** below a ~3.5 km footprint (four octaves 2.4→0.18 km), albedo in log space and a height-field normal whose amplitude follows the texture's own relief. `EARTH_DETAIL=0` disables it. The orbit→mountains flythrough handoff is **not** wired; `E2_low_alps` is the first descent leg only.

### Numbers (lavapipe = llvmpipe CPU Vulkan on this shared 8-vCPU box; **not real-GPU**)
Final E2 shots at 1920×1080 ss 2 (render 3840×2160), `EARTH_TEX=ktx2`, `EARTH_CLOUDS=vol`:

| shot | alt km | GPU ms | VRAM MB | validation |
|---|---|---|---|---|
| orbit_day | 35,000 | 4840 | 480 | 0 |
| orbit_iss | 410 | 4022 | 480 | 0 |
| iss_convection | 410 | 6015 | 480 | 0 |
| terminator | 420 | 2878 | 480 | 0 |
| sunset | 420 | 3386 | 480 | 0 |
| glint | 420 | 4397 | 480 | 0 |
| low_alps | 120 | 8190 | 480 | 0 |
| vs_epic (1080² ss 2) | 1.475e6 | 4295 | 480 | 0 |

**KTX2 before / after** (warm page cache, ss 1, day + ISS; lavapipe software-decodes BC so frame times are not representative of a real GPU):

| mode | disk MB | VRAM MB | load total ms (warm) | gpu_ms day | gpu_ms iss |
|---|---|---|---|---|---|
| ktx2 (default, zstd) | 194.0 | 480 | ~1335 | ~1500 | ~1300 |
| nosc (KTX2, no zstd) | 492.0 | 480 | ~1300 | ~1500 | ~1250 |
| raw (RGBA8/RG8/R8) | 349.5 | 1216 | ~2600 | ~1050 | ~920 |
| pln (old PLN1, pre-VIIRS) | 364.0 | 352 | ~1400 | ~9000 | ~8000 |

- PSNR of KTX2 vs raw at the same exposure: **45.8 dB** (day), **41.8 dB** (ISS). Mean |Δ| ≈ 0.35/255; max |Δ| hits 127 at a few cloud/coast edges (frac > 32 ≈ 0.13 %). `ktx2` and `nosc` are byte-identical in the framebuffer (zstd is disk-only).
- Disk win of KTX2+zstd vs raw: **194 vs 350 MB** (−45 %). VRAM win: **480 vs 1216 MB** (−61 %). Load is dominated by the lavapipe BC decode (~500 ms); on a GPU with native BC the decode step goes away.
- EPIC match: cloud-mask IoU **0.45** (E1-equivalent 2D BMNG was **0.19**); pixel agree **75.6 %** (was 63.8 %). Ours is still sharper and more saturated than EPIC's haze.

### Checks
- **Tests.** `cargo test --release --workspace`: **125 passed**, also with `BRUTAL_SCAN=shared`. Includes the new `PlanetParamsGpu` ↔ Slang layout test and the KTX2 round-trip tests (needs `KTX_TOOL` + its `lib`).
- **Clippy.** `-D warnings` clean, default and `--all-features`.
- **Validation.** 0 errors in every E2 shot and in the tests.
- **SPIR-V.** Fallback rebuilt with `tools/compile_shaders.sh` (planet_* entries updated for the volume / airglow / aero / detail changes).
- **Identity.** Cave byte-identical to `/tmp/cave_ref.png`. Mountains default / closeup(_full) / sunlit(_full) byte-identical to `26_final_*`.

### Data sources added this round
| data | source | licence / credit |
|---|---|---|
| NOAA-20 VIIRS Corrected Reflectance True Colour + Cloud Top Height Day, 2026-07-15 | NASA EOSDIS GIBS / Worldview | NASA imagery, not copyrighted; credit NASA EOSDIS GIBS / Worldview |
| MODIS/VIIRS Cloud Top Height colormap XML | NASA GIBS | NASA |

### Honest weaknesses
- **Volumetric clouds still look soft / snowfield-ish from full-disk.** The ISS convection shot shows towers and self-shadowing; from 35,000 km the volume collapses toward a bright sheet because the footprint is kilometres across and the uniform regime dominates. Thin decks can still terrace under ss 1.
- **EPIC match is coverage, not look.** Cloud systems line up; EPIC is hazier, greyer, less saturated. No spectral render, no per-pixel aerosol map.
- **Airglow vs twilight exposure.** A single exposure cannot show both the faint green line and an unsaturated red-orange troposphere; that is why there are two shots (`E2_terminator`, `E2_sunset`).
- **KTX2 PSNR dropped** vs the pre-VIIRS numbers (52–59 dB → 42–46 dB) because BC5 now carries a sharp CTH channel; the error is concentrated on cloud edges and is invisible at 1:1.
- **Procedural land detail** is a soft albedo/normal crinkle, not a replacement for 500 m imagery; it only engages below a few hundred km.
- **Orbit → mountains descent is not wired.** `E2_low_alps` is a static low-orbit shot toward the Alps; handing the camera into the mountains `quality` scene is future work.
- **lavapipe only.** All ms above are CPU Vulkan; a real GPU is expected to be orders of magnitude faster, especially once BC sampling is hardware.

## Earth round 3: orbit → mountains descent, full-disk cloud structure (25 Oct 2026)

**Images** (`out/quality/steps/`, 1920×1080, ss 2; each `.png` has a `.json`; all timings lavapipe = llvmpipe CPU Vulkan, **not real-GPU**):
- `E3_descent_00.png` … `E3_descent_09.png`: one continuous camera path from 400 km over the Alps to the mountains vista (40 m over the valley floor). Heights 400 / 150 / 60 / 30 / 15 / 7 / 3 / 1.2 / 0.4 km / 40 m.
- `E3_descent_sheet.png` (+ `.json`): labelled contact sheet with heights, handoff weights and seam metrics.
- `E3_descent_fade120_00.png`: an extra pose at 120 m (vista fade 0.47), between frames 08 and 09.
- `E3_vs_epic.png`, `E3_orbit_day.png`, `E3_low_alps.png`: the E2 cameras with the round-3 far-footprint cloud structure.
- `E3_epic_panel.png`: E2 | E3 | DSCOVR EPIC, with crops and statistics in the `.json`.

**Run.**
```
cargo run --release -p render-ash --example descent -- [--only 2,3] [--reuse-mtn] [--heights 120 --prefix X]
# layers + per-layer logs in out/quality/descent/, composites + json in out/quality/steps/
cargo run --release -p render-ash --example earth -- --shot E2_vs_epic   # MESO_TOP/AMP/BIAS=0 = round 2 exactly
```

### What was built
1. **Descent path (`crates/planet/src/descent.rs`).**
   - Nested f64 frames: system → planet (ECEF) → the mountains' local flat y-up frame at Valais 46.10 N 7.60 E. The round trips are tested to < 1 µm (planet frame) and to the system level's own precision.
   - `Descent::alps` is log-linear in height: 400 km → 40 m on a great circle behind the vista. It looks into the peaks and turns into the vista's exact yaw/pitch over the last ~3 km.
2. **Handoff (`handoff`, `layer_weight`, `composite`, `vista_fade`).**
   - Per height:
     - mountains-layer ground weight: 0 above 140 km, 1 below 30 km (log-smoothstep);
     - sky weight: over 3 → 0.3 km;
     - up-vector blend: over 400 → 140 km, so both layers share the roll wherever the layer shows;
     - aerial-perspective range, shadow splits, and one exposure per frame (planet 2.6 → 6.05 over 150 → 20 km; the mountains layer gets the same value in its units).
   - Composite in linear light: `k = a·ground + (1−a)·sky·(ps + (1−ps)·sky)`. At sky = 1 the frame is exactly the mountains render.
   - The planet layer gets `FLATTEN` (terrain pressed to 300 m inside 100 km, back by 140 km) scaled by the ground weight, so the real relief morphs flat only as the layer fades in. It also gets `HOLE` (clear-sky window, 0.9°) and `thaw` (snow-free disc).
   - The mountains layer runs in `LAYER` mode: bent onto the sphere, radial coverage alpha over 60–100 km, all-round rings to 110 km beyond the camera, voxel shells on the ground under the camera.
   - `vista_fade` (1 at ≥ 400 m → 0 at 40 m) scales the bend, shell drop and ring extent/azimuth. At fade 0 the layer is the default view, so **`E3_descent_09` is byte-identical to the default mountains render**. Before this it was 26 dB PSNR off.
3. **Planet-side hooks (`planet.slang`, `PlanetParamsGpu`).**
   - New params: `hole` (window, applied centrally in `cloud_map`, so the volume, its shadows and the 2D path all see it), `meso` and `thaw`. The layout test now covers 16 float4s.
   - `thaw` exists because flattened glaciers read as low cloud. Inside the disc, reflectance is capped at a valley forest's, the ice mask goes to 0 and the slope-map normals flatten, all scaled by the flatten strength.
4. **Full-disk cloud structure (`meso`, far regime only).** Footprint > 0.6 km, full by 2.5 km, so ISS-scale towers are unchanged.
   - Optical depth is scaled by 2^(meso.x·(smoothstep(2.5, 12.5, top_km) − 0.45)) (VIIRS cloud-top height) × 2^(meso.y·fBm). The fBm runs 256 → 32 km, 2:1 anisotropic along a slowly turning axis, with octaves faded by footprint.
   - In-scatter is scaled by the two-stream reflectance ratio R(0.15·τ·f)/R(0.15·τ), R(x) = x/(2+x), times 2^(meso.z·w). Without this, the march saturates in the top optical depths and a thicker or thinner column renders the same.
   - Defaults: `MESO_TOP=2.0 MESO_AMP=4.5 MESO_BIAS=-0.7`. All 0 = round 2, byte for byte (checked).

### Numbers (lavapipe, not real-GPU)
- **Descent planet layer:** 4.4–8.6 s GPU per frame (1920×1080 ss 2), 0 validation errors.
- **Descent mountains layer:**

  | Mode | Build | GPU frame | Validation |
  |---|---|---|---|
  | LAYER, rings to 110 km | 69–158 s | 1.9–3.7 s | 0 |
  | Vista (default view) | 44 s | 1.5 s | 0 |

- **Seam metric at 60 km** (`terrain_lum_planet_mtn` in the json, mean terrain luminance in the blend region):
  - planet 0.111 vs mountains 0.112; mean abs sRGB diff 19/18/17;
  - before the morph and the earlier ground ramp: 0.147 vs 0.112, diff 28/24/20.
- **Cloud statistics on the EPIC geometry** (1024 px disk, ~12 km/px; "interior" = all-cloud 15×15 windows):

  | | interior L p10/50/90 | fine | med | cloud frac (L>0.45) |
  |---|---|---|---|---|
  | E2 | 0.627/0.738/0.764 | 0.0132 | 0.0459 | 0.288 |
  | E3 | 0.651/0.748/0.819 | 0.0152 | 0.0582 | 0.188 |
  | EPIC | 0.585/0.701/0.830 | 0.0218 | 0.0512 | 0.266 |

  The cloud-frac drop is mostly the threshold metric dropping the now-greyer thin cloud.
- **E3 vs E2 regions:**
  - ISS (`low_alps`): 4.2% of pixels changed, all at the far horizon; PSNR 39.5 dB.
  - Orbit day: 24.6 dB, which is intended.

### Checks
- **New tests:**
  - `planet_gpu::cloud_window_clears_and_far_deck_gets_structure`, on a uniform 90% deck:
    - the window centre equals a cloud-free planet;
    - meso = 0 is bit-identical;
    - meso raises the relative luminance spread 0.0065 → 0.17 at an 11 km footprint, with the mean within 35%.
  - `descent::vista_fade_ends_in_the_default_view`;
  - the updated handoff monotonicity/continuity and layout tests.
- **Gates:** `cargo test --release --workspace` gives 134 passed, 0 failed. The same 134 pass with `BRUTAL_SCAN=shared`. `cargo clippy --release --workspace --all-targets [--all-features] -- -D warnings` is clean. Vulkan validation errors: 0 on every render.
- **Identity:** cave, mountains default / closeup_full / sunlit_full, and E2_orbit_day / E2_low_alps (meso off) are byte-identical to the refs.
- **SPIR-V:** the fallback was rebuilt with `tools/compile_shaders.sh`.
- **Environment:** `pkg-config libwayland-dev libxkbcommon-dev libudev-dev libasound2-dev` had to be reinstalled for `--all-features` clippy (the box had been reset).

### Honest weaknesses
- **Two different Alps.** Above ~60 km you see the real Valais (NASA textures); below 30 km you see the procedural `Alpine` field, and nothing ties the two together. 140 → 30 km is a cross-fade between unrelated terrains, with ghosted real glaciers over procedural ridges at 60 km. A grounded fix needs the mountains field conditioned on real DEM data.
- **The layer's edge is visible** at 30–60 km as a disc boundary where the radial alpha (60–100 km) meets the bluer, hazier planet ground. The planet's aerial perspective and the mountains' AP don't match at that range.
- **The clear-sky window is artificial:** a 100 km cloud-free circle around the vista from 150 km down. The mountains' own cirrus replaces it lower.
- **Composite halos:** the layer alpha is not premultiplied, so silhouettes against the planet show thin halos in the 30–60 km frames.
- **The procedural terrain reads as rounded lumps at 7–15 km;** its 2.4 km-class macro shape is weaker than real Alpine relief.
- **Clouds:** the full disk is less of a uniform sheet (grey zones, streaks, brighter cores), but EPIC still has more fine structure (fine 0.015 vs 0.022). Sahel convective cells keep hard bright edges, and the ocean is darker and bluer than EPIC's hazy grey. The fBm is synthetic: real cloud texture at 10–30 km needs higher-resolution imagery.
- **Shadows near the vista (not fully diagnosed):** the right foreground slope is lit at 400 m and 120 m but in shadow in the default view at 40 m. It changes over the last ~100 m. Suspects are the handoff's height-scaled cascade splits (180 m vs 120 m first split at 120 m) and the converging voxel shells; the isolating render was not completed.
- **lavapipe only;** the mountains layer builds take 45–160 s per frame on the CPU, so this is a frame sequence, not a real-time flight.

## Earth round 4: shared Alps, EPIC cloud fraction (28 Oct 2026)

**Images** (`out/quality/steps/`, 1920×1080, ss 2; each `.png` has a `.json`; all timings lavapipe = llvmpipe CPU Vulkan, **not real-GPU**):
- `E4_descent_00.png` … `E4_descent_09.png`: the round-3 camera path (400 / 150 / 60 / 30 / 15 / 7 / 3 / 1.2 / 0.4 km / 40 m) with the shared Alps.
- `E4_descent_sheet.png` (+ `.json`): labelled contact sheet (handoff weights, graft strength, seam metrics).
- `E4_vs_epic.png`, `E4_orbit_day.png`, `E4_low_alps.png`: the E2 cameras with the round-4 cloud defaults.
- `E4_epic_panel.png` (+ `.json`): E3 | E4 | DSCOVR EPIC, crops and statistics.
- `E4_seam_before_after.png` (+ `.json`): E3 (left) vs E4 (right) at 150 / 60 / 30 / 15 / 3 km, with seam numbers.

**Run.**
```
cargo run --release -p render-ash --example descent -- [--only 2,3] [--reuse-mtn] [--no-graft] [--graft <file.grf>]
# first run bakes /workspace/data/earth/graft_alps.grf (256 MB, ~52 s; outside the source tarball)
cargo run --release -p render-ash --example earth -- --shot E2_vs_epic
# MESO_FINE=0 MESO_GAMMA=0 MESO_BIAS=-0.7 = round 3 byte for byte; plus MESO_TOP=0 MESO_AMP=0 MESO_BIAS=0 = round 2
```

### What was built
1. **Shared Alps ("graft", `crates/planet/src/graft.rs`, `examples/descent_parts/graft.rs`).** The mountains render has to stay byte-identical, so the planet adopts the mountains field rather than the other way round. Inside a disc around the vista the planet's terrain *is* `Alpine { seed: 7 }`:
   - **Map:** azimuthal-equidistant about the vista (46.10 N 7.60 E, world (7700, −4200)), f64, via the round-3 nested frames (`GraftMap::from_frames`). It agrees with the layer's `y −= d²/2R` bend to < 50 m at 100 km (tested).
   - **Geometry:** the planet's patch heights are `Alpine::height(x, z, cell = patch vertex spacing)` (the field's own band-limiting), weight 1 inside 120 km, smoothstep back to the real DEM by 165 km.
   - **Time:** strength 0 at ≥ 400 km → 1 at ≤ 150 km (log-smoothstep), so the graft is complete before the mountains layer's ground weight starts at 140 km. FLATTEN and thaw are no longer used (`--no-graft` restores round 3).
   - **Bake:** two nested 4096² levels (±170 km at 83 m, ±42.5 km at 20.75 m): sRGB albedo from a port of `quality.slang`'s mountain material (rock / scree / grass / snow by height, slope, cavity, convexity; sub-50 m octaves at their mean), plus an aux map with the world normal (x, z), heightfield sun visibility (8 km march, soft ±0.005 rad) and AO. File `GRF1` (tested round trip).
   - **Shader (`planet.slang`, bindings 6/7, params `graft`, `graft_e`, `graft_n`, `graft_l`, `graft_t`):** explicit-LOD lookups (non-uniform branch), inner level blended in over 86–97 % of its extent. Albedo × 0.75 (calibrated on the 60 km seam), the vista frame's normal (what the mountains layer shades with), baked shadow and AO in the land term. Toward the disc edge (60 → 120 km) the graft's ~20 km low-pass colour is tinted to the Blue Marble's, so the region does not read as a colour disc where real terrain returns.
2. **Layer compositing.**
   - **No halos:** in LAYER mode the quality example's downsample now averages colour weighted by alpha (straight alpha: the terrain's own colour at silhouettes, no mountains sky mixed in). The default view (alpha 255) is the plain mean as before.
   - **Local colour match (`descent::colour_match`):** since both layers now show the same terrain, the mountains layer takes the planet's local mean colour (its physical aerial perspective) and keeps its own detail: the per-pixel ratio of the two layers' blurred linear colours over the terrain both cover (radius 24 px, 2 box passes). Strength 1 at ≥ 30 km, 0 at ≤ 5 km, so the low frames and the vista are the mountains' own.
3. **Clouds (EPIC fraction).**
   - `meso2.x` (`MESO_GAMMA`, default 0.9): the coverage gamma eases from 1.6 to 0.9 over the meso regime (footprint 0.6 → 2.5 km). Thin decks the map's mips spread out keep their opacity. The ISS regime is unchanged.
   - `meso.w` (`MESO_FINE`, default 1.0): a fifth (16 km) fBm octave, persistence 0.6 → 0.85, octaves faded in from 2 footprints instead of 3.
   - `MESO_BIAS` −0.7 → −0.8, so interiors stay below a bright sheet.

### Numbers (lavapipe, not real-GPU)
- **Graft:** bake 52.4 s; file 256 MB; GPU 341 MB (2 RGBA8 arrays × 2 layers × 4096², full mips), upload with CPU mips ~1.0 s.
- **Descent planet layer:** 4.4–8.2 s GPU per frame; mountains layer build 69–81 s, GPU 1.6–2.5 s; validation errors 0 everywhere.
- **Seam** (mean |planet − mountains| sRGB over terrain both cover; json `seam`):

  | frame | E3 | E4 raw | E4 after colour match |
  |---|---|---|---|
  | 02 (60 km) | 19.4 / 18.4 / 16.5 | 11.0 / 10.0 / 12.4 | 8.9 / 7.2 / 4.5 |
  | 03 (30 km) | 18.1 / 21.3 / 13.6 | 11.4 / 9.3 / 10.3 | 10.5 / 7.8 / 3.5 |
  | 04 (15 km) | 16.9 / 20.7 / 13.5 | 12.8 / 10.8 / 12.0 | 11.3 / 8.2 / 5.5 |
  | 05 (7 km) | 24.4 / 26.4 / 15.3 | 14.0 / 13.2 / 13.7 | 13.5 / 12.4 / 12.7 |
  | 06 (3 km) | 38.8 / 36.1 / 20.8 | 20.2 / 18.5 / 15.8 | (no match) |
  | 08 (400 m) | 30.8 / 31.5 / 28.4 | 22.0 / 20.8 / 14.7 | (no match) |

  Residuals after the match are mostly the detail difference: the planet at 21–83 m texels versus the mountains' metre-scale material.
- **Clouds on the EPIC geometry:**

  | | cloud frac | cloud L p10/50/90 | interior L p10/50/90 | fine | med |
  |---|---|---|---|---|---|
  | E3 | 0.188 | 0.467/0.571/0.771 | 0.651/0.748/0.819 | 0.0152 | 0.0582 |
  | E4 | 0.261 | 0.469/0.580/0.769 | 0.615/0.733/0.819 | 0.0241 | 0.0512 |
  | EPIC | 0.266 | 0.472/0.575/0.748 | 0.585/0.701/0.830 | 0.0218 | 0.0512 |

  Sweep (bias −0.7..0, fine 0..1, gamma 1.6..0.9) in the session log: raising the bias alone lifts the fraction only by making interiors brighter (p50 0.83 at bias 0, a sheet). The far-regime gamma lifts it while interiors stay put.

### Checks
- **Identity:**
  - `E4_descent_09` is byte-identical to `E3_descent_09` (the default mountains view);
  - cave and mountains default / closeup_full / sunlit_full are byte-identical to the round-3 refs;
  - the earth example with the round-3 settings reproduces `E3_vs_epic` / `E3_orbit_day` / `E3_low_alps`, and with meso off `E2_*`, byte for byte (graft off = no change).
- **New tests:**
  - `graft::map_round_trips_and_matches_the_layer_bend`, `strength_schedule`, `file_round_trip`;
  - `descent::colour_match_vanishes_at_the_vista`;
  - the layout test now covers 22 float4s.
- **Gates** (logs `/workspace/checkpoints/e4_*.log`):
  - `cargo test --release --workspace`: 138 passed, 0 failed; the same with `BRUTAL_SCAN=shared`: 138 / 0;
  - clippy `-D warnings`, default and `--all-features`: clean (`graft_height` now takes its blend as one `(r_full, r_out, strength)` tuple; re-rendered `E4_descent_03`, byte-identical);
  - SPIR-V fallback regenerated (`tools/compile_shaders.sh`, spirv-val clean; only the `planet_*` modules changed);
  - Vulkan validation: 0 errors in every planet and mountains layer, the cave and the mountains identity renders.

### Honest weaknesses
- **The real Alps are gone inside the disc.** From 150 km down, a 330 km disc around Valais is the procedural field: Lake Geneva, the real glaciers and the Swiss plateau are replaced by procedural mountains with little snow. That is the price of keeping the mountains render byte-identical. A grounded version would condition `Alpine` on the DEM (it would change the mountains scene).
- **At 150 km the disc still reads as a texture region:** uniform lumpy relief everywhere, versus the real Earth's plains and valleys beyond. The colour tint hides the colour edge, not the relief change.
- **The 7–15 km "rounded lumps" are not fixed;** they are the field's own macro shape, now on both layers.
- **The colour match is a compositing trick:** it borrows the planet's local mean colour, so the planet's coarser shading (21–83 m) sets the mid-frequency tone of the 30–60 km frames. Below 5 km the layers still differ by 14–22 sRGB levels (the planet is darker; mountain AO/shadow at metre scale), hidden by the full ground weight.
- **The clear-sky HOLE is still there.**
- **Clouds:** the fraction and fine structure now match EPIC, but EPIC's hazy grey ocean and softer Sahel cells are still missing (the E4 ocean stays dark blue). Interiors are slightly brighter at p10/p50 (0.615/0.733 vs 0.585/0.701).
- **Graft memory:** 341 MB GPU and a 256 MB cache file for one vista; KTX2/BC7 for the albedo would cut it ~4×.

## Earth round 5: grounded Alps (real DEM + Sentinel-2 in the shared disc), BC7 graft, hole fix (29 Oct 2026)

**Images** (`out/quality/steps/`, 1920×1080, ss 2; each `.png` has a `.json`; all timings lavapipe = llvmpipe CPU Vulkan, **not real-GPU**):
- `E5_descent_00.png` … `E5_descent_09.png`: the round-3/4 camera path (400 / 150 / 60 / 30 / 15 / 7 / 3 / 1.2 / 0.4 km / 40 m) on the grounded field, BC7 graft. `E5_descent_09` is byte-identical to `E4_descent_09` (the default mountains view).
- `E5_descent_sheet.png` (+ `.json`): labelled contact sheet (handoff weights, graft strength, seam metrics, timings).
- `E5_crop_lakes.png` (150 km), `E5_crop_glaciers.png` (60 km), `E5_crop_plateau.png` (60 km, horizon), `E5_crop_core.png` (15 km): E4 (left) vs E5 (right), same crop.
- `E5_seam_02.png`, `E5_seam_03.png`, `E5_seam_04.png`: planet layer (left) vs mountains layer (right) at 60 / 30 / 15 km; json has E5 and E4 seam numbers.
- `E5_bc7_crops.png` (+ `.json`): raw RGBA8 vs BC7 graft, planet layer only, the worst 256² window of frames 01 / 02 / 04 at 2×, with PSNR and VRAM / disk.
- `E5_hole.png` (+ `.json`): frames 00 / 01 before (scaled coverage) and after (eroded coverage).
- `E5_core_option.png` (+ `.json`): **option, not shipped**: the default view with a smaller procedural core (`GROUND_CORE=5000,9000,9000,13000`).

**Run.**
```
cargo run --release -p render-ash --example descent -- [--only 2,3] [--reuse-mtn] [--no-ground] [--ground <file.grd>] [--graft <file.grf>]
# needs the data under /workspace/data/earth (outside the source tarball):
#   dem/  Copernicus GLO-30 tiles + mosaics (dem/mosaic.py; ~930 MB)
#   s2/   EOX Sentinel-2 cloudless z11/z13 mosaics (s2/mosaic_s2.py; ~314 MB)
# fetch / mosaic scripts (copies, run inside those dirs): tools/earth_e5/{fetch_dem.sh,dem_mosaic.py,fetch_s2.py,mosaic_s2.py}
# first run bakes ground_alps.grd (128 MB) and graft_alps_e5.grf (256 MB, ~47 s),
# first earth/planet run encodes graft_alps_e5.grf.{alb,aux}.ktx2 (BC7, 36 + 33 MB, ~5 s)
GRAFT_TEX=raw ...            # round-4 RGBA8 graft upload (no KTX2)
cargo run --release -p render-ash --example descent -- --no-ground   # round 4's procedural disc
```
Env: `GROUND` (grd path, set by descent for its children), `GROUND_ALB` / `GROUND_GAIN` / `GROUND_DELIGHT` (imagery albedo gain, Blue Marble gain match, shading divide strength), `S2_LAYER` (2020 default; 2016 = the CC BY layer), `GROUND_CORE=disc0,disc1,wedge0,wedge1` (m; procedural core override), `GRAFT_TEX=raw`, `BAKE_BC7=basic|fast` (slower, better encoder presets; default very_fast).

### What was built
1. **Grounded field (`crates/voxel/src/ground.rs`, `fields.rs`).** The mountains render must stay byte-identical, so the procedural `Alpine { seed: 7 }` stays the scene near the vista and real elevation takes over outside it: `base = lerp(alpine, dem, w)`, then the gullies, crags, tors and overhangs run on top of the real relief.
   - **DEM:** Copernicus GLO-30 (1″ around the vista, 3″ box-averaged beyond), resampled (Catmull-Rom) onto the mountains world frame through the graft's azimuthal-equidistant map, two nested levels like the graft (±170 km / ±42.5 km), file `GRD1`. Sampling is a cubic B-spline on the mip matching the cell (band-limited like the field's own octaves, no facets).
   - **Procedural core:** the default view's footprint is kept procedural: a disc (procedural inside 5 km, DEM beyond 9 km) plus a wedge along yaw 2.75 (28–40 km, half angle 0.78–1.05 rad). Weight fades out at the outer level's edge (0.975–0.998 of the extent).
   - Without `GROUND` / with `--no-ground` the field is exactly round 4.
2. **Real imagery in the graft bake (`descent_parts/ground.rs`, `graft.rs`).** EOX Sentinel-2 cloudless (Web Mercator z11 / z13 stitched), decoded to linear, the acquisition's terrain shading partly divided out (Lambert, assumed late-morning summer sun), and gain-matched to the planet's Blue Marble over the region. The albedo blends from the material port to the imagery by `smoothstep(0, 0.6, w)`; graft albedo alpha = 1 − water (lakes get water shading in `planet.slang`). The quality pass gets the same albedo as a ground map (`QParams.gmap`, RGBA8: sRGB + 7-bit weight | water bit), so both layers show the same lakes, glaciers and fields; vegetation is suppressed where the imagery takes over.
3. **Softer disc / lumps.** Outside the core the 7–15 km "rounded lumps" are now the real relief; the round-4 disc edge (procedural → DEM at 120–165 km) is gone because the procedural field only lives in the core.
4. **BC7 graft (`planet::ktx2`, `planet::bake::encode_bc7_rgba8`, `planet_set_graft_fmt`).** The graft uploads as two 2-layer 2D-array KTX2 files: BC7 sRGB albedo (+ water alpha) and BC7 linear aux (normal x / z, sun visibility), mips to 4×4, zstd 12, cached next to the `.grf` and re-encoded when older. `ktx2.rs` gained 2D arrays (`write_array`, `faceCount 1, layerCount n`). BC4 / BC5 were not used: aux needs three channels and the graft has no height texture. `GRAFT_TEX=raw` keeps the RGBA8 path.
5. **Clear-sky hole (`planet.slang` `cloud_map`).** Inside the clear-sky window ramp coverage is now *eroded* (`saturate((c − (1 − k)) / k)`): thin cover clears first and thick cells keep their shape, instead of all coverage being scaled down into a uniform faint patch.

### Numbers (lavapipe, not real-GPU)
- **Graft VRAM / disk:**

  | | VRAM | disk | upload |
  |---|---|---|---|
  | E4 / `GRAFT_TEX=raw` (RGBA8, mips to 1×1) | 341.3 MiB | 256 MiB `.grf` | ~1200 ms |
  | E5 BC7 (mips to 4×4) | **85.3 MiB** | + 36.0 + 33.0 MiB `.ktx2` (the `.grf` stays as the encode source) | ~400–460 ms (+ ~5 s one-off encode) |

- **BC7 quality** (planet layer, full frame, raw vs BC7): frame 01 PSNR 51.5 dB / mean |Δ| 0.22 / p99 4; 02 47.0 dB / 0.50 / 6; 04 46.3 dB / 0.58 / 6 (sRGB levels). Not visible in the worst-window crops.
- **Bakes:** ground (`.grd`, 128 MB) 1.3 s from the mosaics; graft bake 46.8 s; quality-pass ground map 170.7 MB, built in 1.66 s.
- **Descent:** planet layer 6.9–9.4 s GPU per frame; mountains layer GPU 1.8–3.1 s, build 46–100 s (one 340 s outlier at frame 07 while another render ran); validation errors 0 everywhere.
- **Seam** (mean |planet − mountains| sRGB R/G/B over terrain both cover):

  | frame | E4 raw | E4 matched | E5 raw | E5 matched |
  |---|---|---|---|---|
  | 02 (60 km) | 11.0 / 10.0 / 12.4 | 8.9 / 7.2 / 4.5 | 21.0 / 18.6 / 16.8 | 17.5 / 15.0 / 9.8 |
  | 03 (30 km) | 11.4 / 9.3 / 10.3 | 10.5 / 7.8 / 3.5 | 18.5 / 13.7 / 12.9 | 15.7 / 12.1 / 5.7 |
  | 04 (15 km) | 12.8 / 10.8 / 12.0 | 11.3 / 8.2 / 5.5 | 20.8 / 15.7 / 14.4 | 17.7 / 13.0 / 7.9 |
  | 05 (7 km) | 14.0 / 13.2 / 13.7 | 13.5 / 12.4 / 12.7 | 16.9 / 14.3 / 14.2 | 16.3 / 13.6 / 13.2 |
  | 06 (3 km) | 20.2 / 18.5 / 15.8 | – | 20.9 / 18.5 / 15.9 | – |
  | 08 (400 m) | 22.0 / 20.8 / 14.7 | – | 22.0 / 20.8 / 14.7 | – |

  The raw difference roughly doubled at 60–15 km: real imagery has far more local contrast (glaciers next to dark forest) and the two renderers shade it differently (planet at 21–83 m texels, mountains at metre scale with its own AO / shadows). Mean luminance agrees (e.g. 0.133 vs 0.148 at 60 km) and side by side (`E5_seam_*`) the layers show the same lakes, valleys and glaciers.
- **Hole:** bright-pixel (L > 0.6) fraction in the frame's centre third at 400 km 5.7 % → 18.5 %; whole frame 17.3 % → 18.7 %.

### Checks
- **Identity:** `E5_descent_09` = `E4_descent_09` byte for byte; cave and mountains default / closeup_full / sunlit_full byte-identical to the round-3 refs; earth `E2_orbit_day` / `E2_low_alps` (round-2 settings) byte-identical. (log `/workspace/checkpoints/e5_ident.log`)
- **New tests:** `ktx2::round_trip_2d_array` (validated with `ktx validate` when `KTX_TOOL` is set); `voxel::ground` tests (`bspline_is_exact_on_planes_and_mips_stop_at_64`, `weight_core_edge_and_file_round_trip`); `quality::layout_tests::qparams_layout_matches_slang` (gmap fields at 656 / 672 / 704, size 720).
- **Gates** (logs `/workspace/checkpoints/e5_*.log`): `cargo test --release --workspace` 142 passed, 0 failed; the same with `BRUTAL_SCAN=shared` 142 / 0 (`KTX_TOOL` set); clippy `-D warnings`, default and `--all-features`: clean; Vulkan validation 0 errors in every planet / mountains layer and identity render.
- SPIR-V fallback regenerated (`tools/compile_shaders.sh`), spirv-val clean on every module.

### Honest weaknesses
- **The procedural core is still visible from 150 km to 15 km:** a smooth olive-grey patch with rounded lumps around the vista (`E5_crop_lakes`, `E5_crop_core`). It exists so the default view stays byte-identical; inside it the 7–15 km lumps are *not* softened. `E5_core_option` shows a smaller core (real Monte Rosa side massif from ~9–13 km), but then frame 08 / the default view would change; that choice is Helen's.
- **Seam numbers got worse (≈ 2× E4 raw at 60–15 km)** though the content now matches. The colour match hides most of it at ≥ 30 km; at 15–7 km the planet is visibly flatter and darker than the mountains layer.
- **Imagery artefacts:** the core reads as a pale bluish-grey smudge in the middle of the real Alps at 150–60 km (`E5_descent_01/02`); the de-shading is approximate (S2 mosaics mix dates and sun angles), so some of the imagery's own shadows stay baked in and get shaded again.
- **Licence:** the default S2 cloudless **2020** layer is **CC BY-NC-SA 4.0** (non-commercial). Attribution: "Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH (Contains modified Copernicus Sentinel data 2020)". `S2_LAYER=2016` selects the CC BY 4.0 layer (older, more seasonal colour). DEM: Copernicus GLO-30 (© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA).
- **Memory:** the quality pass's ground map is 170.7 MB uncompressed; the graft `.grf` (256 MiB) still sits on disk next to the KTX2 cache.
- **Clouds** at 400–150 km still have hard, plate-like edges (unchanged from E4).
- **Not done:** EPIC's hazy grey ocean (the ocean stays dark blue). The hole is reduced, not gone: at 150 km the area around the vista is still clearer than the surroundings.

## Earth round 6: smaller procedural core (real Monte Rosa side), marine haze over open ocean (30 Oct 2026)

**Images** (`out/quality/steps/`, 1920×1080, ss 2; each `.png` has a `.json`; all timings lavapipe = llvmpipe CPU Vulkan, **not real-GPU**):
- `E6_descent_00.png` … `E6_descent_09.png`: the same camera path (400 / 150 / 60 / 30 / 15 / 7 / 3 / 1.2 / 0.4 km / 40 m), smaller core, new graft bake, ocean haze. **`E6_descent_09` is the new default vista** (it changed on purpose; see "New baseline").
- `E6_descent_sheet.png`: labelled contact sheet.
- `E6_vista_before_after.png`: E5 (left) vs E6 (right) at 40 m. Left of the procedural peak the real DEM + Sentinel-2 snow massifs (toward Monte Rosa) now show where E5 had procedural hills. `E6_crop_monterosa.png` is the same at full resolution.
- `E6_crop_core.png` (15 km), `E6_crop_core_05.png` (7 km), `E6_crop_core_07.png` (1.2 km): E5 vs E6, same crop. The core footprint is now a small disc.
- `E6_seam_02/03/04.png`: planet layer vs mountains layer at 60 / 30 / 15 km.
- `E6_vs_epic.png`, `E6_orbit_day.png`, `E6_low_alps.png`: the round-2 earth shots (`--shot E2_*`) with round-6 defaults (haze on).
- `E6_epic_panel.png` (+ `.json`): E5 (no haze) | E6 | NASA EPIC. The json holds clear-ocean colour stats by disk-radius band.

**Run.**
```
cargo run --release -p render-ash --example descent -- --prefix E6_descent
# first run bakes data/earth/ground_alps_e6.grd (128 MB, ~1 s) and graft_alps_e6.grf (256 MB, ~39 s),
# first earth run encodes graft_alps_e6.grf.{alb,aux}.ktx2 (BC7, 36 + 33 MB)
cargo run --release -p render-ash --example earth -- --pack /workspace/data/earth/earth.pln --out <dir> --shot E2_vs_epic
cargo run --release -p render-ash --example quality -- --scene mountains   # now grounded by default
```
Env:
- `GROUND_CORE=e6` (default), `GROUND_CORE=e5` (round 5), or `GROUND_CORE=disc0,disc1,wedge0,wedge1[,ang0,ang1]` (m / rad).
  - With `e5` you also need `--ground ground_alps.grd --graft graft_alps_e5.grf` for the round-5 albedo. The graft bake depends on the core, so each core has its own `.grd` + `.grf` pair.
- `GROUND=off` / `GROUND_ALB=off` (quality) or `--no-ground` (descent): the procedural round-4 field.
- `OCEAN_HAZE=<AOD at 550 nm>` (default 0.15; `0` = round 5), `HAZE_ANGSTROM` (0.4), `HAZE_SSA` (0.98), `HAZE_MASK_KM` (150: scale of the open-ocean mask).

### What was built
1. **Smaller procedural core (`crates/voxel/src/ground.rs`).** `Core::E6`:
   - disc: procedural inside 4.5 km, DEM beyond 10 km (was 5 → 9 km);
   - wedge along yaw 2.75: 8 → 14 km (was 28 → 40 km);
   - same half angle (0.78–1.05 rad).

   The wider disc ramp keeps the edge soft. The default vista still has the procedural foreground and peak (they sit inside ~7 km), but beyond ~14 km along the view the real relief and imagery take over: the snowy massifs on the left toward Monte Rosa (bearing ~132°, ~27 km). The Matterhorn (~162°, ~14.5 km) is behind the procedural peak at 40 m and shows as the pyramid in frames 06 / 07.

   Before choosing, candidates were compared at the vista (`/workspace/e6work/cand_A/B.png`): A was 5–9 km / 9–13 km (round 5's `E5_core_option`), and B (shipped) has the softer ramp.

   `descent` now passes `GROUND`/`GROUND_ALB` to every mountains frame. `quality` defaults to `ground_alps_e6.grd` + `graft_alps_e6.grf` when present (a warning and a procedural fallback when missing).
2. **Marine haze (`planet.slang` `ocean_haze`, `PParams.haze`).** A thin boundary-layer aerosol slab over open water, applied to the surface radiance before night lights:
   - **Optical depth:** τ(λ) = AOD·(λ/550)^−α·m, where the open-ocean mask m = smoothstep(0.55, 0.95, water) is sampled at a ~150 km mip. Lakes and coasts get little haze.
   - **Phase function:** two-term Henyey–Greenstein (0.85·HG(0.7) + 0.15·HG(−0.35)), so the EPIC backscatter geometry gets some light.
   - **Single scattering:** Es·ω·p·μs/(μs+μv)·(1−e^(−τ(1/μs+1/μv))).
   - **Skylight term:** ω/2·Esky/π·(1−Tv).
   - **Result:** L·Tv + both terms.

   `OCEAN_HAZE=0` is byte-identical to round 5. Cost: +3–6 % planet GPU time (vs_epic 6.50 → 6.83 s, orbit_day 7.05 → 7.26 s, low_alps 9.62 → 10.18 s; lavapipe).

### New baseline
- **Mountains (`quality --scene mountains`) changed by design.** New refs: `/workspace/checkpoints/e6_mtn_ref_{default,closeup_full,sunlit_full}.png`. Re-renders match them byte for byte.
- With `GROUND=off GROUND_ALB=off` the three views are still byte-identical to the round-3 refs (`e3_mtn_ref_*`). This proves that only the ground input changed.
- **Cave:** byte-identical to `e3_cave_ref.png`.
- **Earth:** `E2_vs_epic` / `E2_orbit_day` / `E2_low_alps` with `MESO_*=0 OCEAN_HAZE=0` are byte-identical to round 2.
- The round-5 vista (round 5 rendered the 40 m frame and `quality` ungrounded) is exactly the `GROUND=off` render above. Grounding with the e5 files (`GROUND=ground_alps.grd GROUND_ALB=graft_alps_e5.grf`) is *not* identical to it: real terrain shows in the distance even with the old core.
- Script: `/workspace/e6work/ident.sh`, log `/workspace/checkpoints/e6_ident.log`.

### Numbers (lavapipe, not real-GPU)
- **Bakes:** `ground_alps_e6.grd` 1.0 s; `graft_alps_e6.grf` 39.2 s (level 0 7.1 s, level 1 24.8 s); BC7 cache 36.0 + 33.2 MiB.
- **Descent:**
  - planet layer: 7.6–16.5 s GPU per frame (frames 01/02 at 16 s, slower than E5's 7–9 s, because other renders ran in parallel; not a regression in the shader path);
  - mountains layer: GPU 1.7–2.6 s, build 42–101 s;
  - validation: 0 errors everywhere.
- **Seam** (mean |planet − mountains| sRGB R/G/B, raw):

  | frame | E5 | E6 |
  |---|---|---|
  | 02 (60 km) | 21.0 / 18.6 / 16.8 | 21.5 / 19.1 / 17.1 |
  | 03 (30 km) | 18.5 / 13.7 / 12.9 | 19.7 / 14.8 / 13.5 |
  | 04 (15 km) | 20.8 / 15.7 / 14.4 | 22.7 / 17.4 / 15.5 |
  | 05 (7 km) | 16.9 / 14.3 / 14.2 | 18.3 / 15.6 / 15.5 |
  | 09 (40 m) | 26.5 / 25.1 / 17.1 | 26.9 / 25.4 / 17.9 |

  The numbers are slightly worse: more of the frame is now real imagery, which has more local contrast.
- **Ocean vs EPIC** (`E6_epic_panel.json`, clear-ocean mean, disk centre r < 0.5):

  | | sRGB | L | saturation |
  |---|---|---|---|
  | E5 | 42 / 58 / 92 | 0.224 | 0.197 |
  | E6 | 53 / 68 / 100 | 0.264 | 0.182 |
  | EPIC | 40 / 52 / 66 | 0.197 | 0.104 |

  Near the limb (r 0.8–0.95): E5 L 0.290 / sat 0.193, E6 0.320 / 0.174, EPIC 0.299 / 0.092.

### Checks
- **Identity:** see "New baseline" (log `/workspace/checkpoints/e6_ident.log`).
- **Tests:**
  - `weight_core_edge_and_file_round_trip` updated for E6: weight 0 at 7 km along the view, 1 at 20 km, monotone ramp behind; with `Core::E5` the weight at 20 km is 0.
  - The planet layout test covers `haze` at 64 + 22·16, with size 64 + 23·16.
- **Gates** (logs `/workspace/checkpoints/e6_*.log`):
  - `cargo test --release --workspace`: 142 passed, 0 failed.
  - The same with `BRUTAL_SCAN=shared` (`KTX_TOOL` set): 142 / 0.
  - clippy `-D warnings`, default and `--all-features`: clean.
- **SPIR-V:** fallback regenerated, spirv-val clean on all 48 modules.

### Honest weaknesses
- **The ocean is not EPIC-grey.** Physical haze (AOD 0.15, ω 0.98) mostly *brightens* the ocean: L +18 %, now 34 % above EPIC at the disk centre. Saturation only drops 0.197 → 0.182 against EPIC's 0.104.
  - The remaining gap does not look ocean-specific: our land has about 2× EPIC's linear blue too (0.111 vs 0.055).
  - A rough single-scattering Rayleigh estimate puts clear-ocean TOA reflectance at several times higher in blue than in red; EPIC's JPEG is much flatter.
  - So most of EPIC's grey looks like its colour processing (or absorbing aerosol). It is not something a scattering haze reproduces.
  - Closing it would need a non-physical display grade or blue-absorbing (dust-like) aerosol. Neither was shipped.
  - Visually E6's ocean reads softer and milkier toward the limb, still clearly blue.
- **The procedural peak and foreground are still in the vista.** The real massifs on the left are lower-detail: 30 m DEM, softer and flatter than the procedural ridge, with very uniform snow albedo from S2. The Matterhorn is hidden at 40 m.
- **Core edge visible** at 15–7 km as a green procedural basin among grey-white real slopes (`E6_crop_core`, `E6_crop_core_05`).
- **Seam slightly worse** than E5 (see table).
- **Two data sets on disk:** e5 + e6 grd/grf = 2 × 384 MB plus the KTX2 caches.
- **Unchanged from E5:**
  - plate-like cloud edges at 400–150 km;
  - the clear-sky hole is reduced, not gone;
  - licences: S2 cloudless 2020 is CC BY-NC-SA 4.0 (`S2_LAYER=2016` for CC BY 4.0); Copernicus GLO-30 attribution as in round 5.

## Earth round 7: softer cloud edges, smaller clear window at 400–150 km, optional EPIC grade (30 Oct 2026)

**Images** (`out/quality/steps/`, 1920×1080, ss 2; each `.png` has a `.json`; all timings lavapipe = llvmpipe CPU Vulkan, **not real-GPU**):
- `E7_descent_00.png` … `E7_descent_09.png` + `E7_descent_sheet.png`: the same camera path as E6. Frames 07–09 (1.2 km → 40 m vista) are byte-identical to E6; 02–06 differ only in a few cloud pixels (mean |Δ| 0.02–2.2 sRGB levels).
- `E7_cloud_edges_00.png` / `_01.png`: E6 vs E7 crop at 400 / 150 km (deck edges, window rim).
- `E7_hole.png` (+ `.json`): E6 vs E7 full frames 00/01 with the clear-window statistic.
- `E7_vs_epic.png`, `E7_orbit_day.png`, `E7_low_alps.png`: the round-2 earth shots with round-7 defaults. `vs_epic` and `orbit_day` are byte-identical to E6 (full-disk footprints are outside the new features); `low_alps` (cloud edges at a few hundred metres per pixel) changed.
- `E7_epic_grade_panel.png` (+ `.json`): E7 default | `HAZE_DUST=3` | `EPIC_GRADE=0.5` | `EPIC_GRADE=1` | NASA EPIC. The json holds clear-ocean colour by disk band and the cloud statistics for each.

**Run.** As round 6. New env (all in `PParams.cloud3`, read by `earth` and forwarded by `descent`):
- `CLOUD_FEATHER` (km, default 16; `0` = round 6 for every round-7 cloud change), `CLOUD_FEATHER_MIX` (0.6).
- `HOLE_NOISE` (0.35; `0` = round circle). `descent`: `HOLE_FAR_DEG` (0.55; `0.9` = round 6) is the window radius at ≥ 140 km, ramping to `HOLE_DEG` 0.9 at ≤ 70 km where the mountains layer needs it.
- `HAZE_DUST` (0 = off): blue/green absorption on the ocean haze single-scattering albedo (ω − dust·(0, 0.03, 0.09)).
- `EPIC_GRADE` (**non-physical, off by default**): `1` = fitted grade, a number s = identity→fit by s, or nine comma-separated values (row-major display-linear matrix).

### What was built
1. **Cloud edges (`planet.slang`, only for footprints < 5 km):**
   - *Feather:* coverage is blended toward a ~16 km mip with a 4–9 km value-noise weight, so map edges get a ragged fringe instead of a one-texel step.
   - *Column shaping:* `CCol.ce` = min(coverage, coarse 32 km coverage thinned toward the window). Deck thickness and how far tops sink now follow `ce` (and sink up to `ce` 0.95 instead of 0.6), so decks taper over ~10–30 km toward their edges instead of ending in walls.
   - *Fringes:* in the cell model fewer cells pass where `ce` is low, the threshold is widened (soft), and density is scaled down to 35 % at the edge: broken, translucent fringes.
2. **Clear window:** the rim angle is wobbled by two value-noise octaves (`HOLE_NOISE`), coverage inside the ramp is broken up with a mean-preserving noise before erosion, and the window shrinks to 0.55° above 140 km.
3. **EPIC grade (`earth.rs` post, before sRGB encode):** 3×3 matrix fitted by `/workspace/e7work/fitgrade.py` on luminance-binned class means (clear ocean ×3, land ×2, cloud ×1 weight) between `E6_vs_epic` and `epic_africa.jpg`, ridge λ 0.02 toward identity:
   ```
   [ 0.8005  0.1943 -0.0427 ]
   [-0.0544  1.1512 -0.1265 ]
   [ 0.0245  0.3964  0.4739 ]
   ```
   It is a display grade, not physics; kept off so Helen can choose.

### Numbers (lavapipe, not real-GPU)
- **Clear window** (`E7_hole.json`: bright L > 0.6 fraction, centre third vs frame below 25 %; ratio < 1 = clear patch at the vista):

  | frame | E6 centre / frame / ratio | E7 |
  |---|---|---|
  | 00 (400 km) | 18.4 % / 24.1 % / 0.76 | 47.3 % / 26.5 % / 1.78 |
  | 01 (150 km) | 4.0 % / 5.4 % / 0.74 | 8.7 % / 7.5 % / 1.16 |

  The frame-wide bright fraction rises only 2 points (no bright sheet).
- **Ocean vs EPIC** (clear ocean, disk centre r < 0.5, sRGB mean / saturation): E7 default 53 / 68 / 100, 0.182 · `HAZE_DUST=1` 53 / 68 / 99, 0.179 · `HAZE_DUST=3` 53 / 67 / 98, 0.173 · `EPIC_GRADE=0.5` 53 / 65 / 92, 0.152 · `EPIC_GRADE=1` 52 / 62 / 82, 0.118 · EPIC 40 / 52 / 66, 0.105. Limb band (0.8–0.95): default 68 / 83 / 112, 0.174; grade 1 66 / 77 / 95, 0.111; EPIC 67 / 78 / 91, 0.092.
- **EPIC clouds** (`cloudstats.py`): cloud fraction default 0.267 (EPIC 0.266), local std 0.092 (EPIC 0.071) — unchanged from E6. With `EPIC_GRADE=1` the fraction reads 0.295 because the greyer image passes the metric's low-saturation cloud mask more often; the cloud pixels themselves barely change (L p50 0.578).
- **GPU:** vs_epic 7.0 s, orbit_day 7.6 s (E6 6.8 / 7.3); **low_alps 20.5 s (E6 10.2 s)**; descent planet layer 15.6–24.5 s per frame (frames 00/01 20.3 / 24.5 s). Mountains layer GPU 1.8–3.5 s, build 47–106 s. Validation 0 errors everywhere.
- **Seam** (frames 02–09): same as E6 to ±0.01 (e.g. 02: 21.5 / 19.1 / 17.1).

### Checks
- **Identity** (`/workspace/e7work/ident.sh`, log `/workspace/checkpoints/e7_ident.log`): cave == `e3_cave_ref.png`; `GROUND=off` mountains == `e3_mtn_ref_*`; default mountains == `e6_mtn_ref_*` (E6 baseline kept); E2 shots with `MESO_*=0 OCEAN_HAZE=0 CLOUD_FEATHER=0 HOLE_NOISE=0 HAZE_DUST=0` (no grade) == round 2. All byte-identical.
- **Layout test:** `cloud3` at 64 + 23·16, size 64 + 24·16.
- **Gates** (logs `/workspace/checkpoints/e7_*.log`): `cargo test --release --workspace` 142 passed / 0 failed; the same with `BRUTAL_SCAN=shared` (`KTX_TOOL` set) 142 / 0; clippy `-D warnings`, default and `--all-features`: clean.
- **SPIR-V:** fallback regenerated, spirv-val clean on all 48 modules (`e7_spv.log`).

### Honest weaknesses
- **Cloud edges are softer, not soft.** The walls are now tapered slopes with ragged, partly translucent fringes, but at 150 km deck edges still read as crisp outlines with a lit rim; the underlying 5 km coverage map simply has hard clear/overcast boundaries.
- **The window is gone statistically because the real weather is overcast:** with a 0.55° window the Alps at 150 km are mostly under the VIIRS deck, so the vista is no longer visible in frames 00/01 (it reappears at 60 km where the window widens). A visible, still-unnatural cleared lobe remains in frame 01.
- **Cost:** the extra coarse sample per column step and the translucent fringes (rays march further) roughly double the planet pass on cloud-edge-heavy low views (low_alps: `CLOUD_FEATHER=0` 10.2 s vs default 21.0 s, re-measured back to back). Not optimised; `CLOUD_FEATHER=0` restores round-6 cost.
- **Dust aerosol does almost nothing** (b −1 to −2 levels at 3× strength): physically plausible dust cannot make EPIC's grey.
- **The grade is a cosmetic fit:** it greys the ocean (saturation 0.18 → 0.12, EPIC 0.105) but the ocean stays ~30 % brighter than EPIC in red/green, and land shifts a little greener/duller. It also mutes clouds' faint blue. Use it as a look, not as a radiometric fix.

## Solar system round 1: Sun + eight planets + Moon at true scale, approach + handoff to the Earth path (30 Oct 2026)

```bash
export LD_LIBRARY_PATH=/workspace/tools_ext/KTX-Software-4.4.2-Linux-x86_64/lib:$LD_LIBRARY_PATH
cargo build --release -p render-ash --example earth --example solar   # solar calls the earth binary for the handoff
./target/release/examples/solar --shot all            # or --shot list / --shot SS1_saturn,SS1_mars
python3 tools/solar_hud.py out/quality/steps SS1_overview SS1_inner SS1_path_00 SS1_path_01 SS1_path_02 SS1_path_03 SS1_path_04
# needs /workspace/data/solar/layers/*.png (11 equirect 2048x1024 layers + saturn_ring.png, outside the tarball)
# and /workspace/data/earth/earth.pln (+ stars) as before
```

**Scene.** `planet::solar::System` at the EPIC epoch (2026-07-15 11:07:35 UTC): nested f64 frames (galaxy > "sol" system frame > one body-fixed frame per body; the Earth's body frame is the planet renderer's ECEF). Positions: Standish Table 1 (1800–2050) + the AA low-precision Moon; radii, poles, prime meridians: IAU WGCCRE 2015; geometric albedos: Mallama et al. 2017; Saturn's rings 69 700–141 020 km (D ring inner edge to F ring). Everything is real metres; the GPU only ever sees camera-relative f32 (the `coords` solar_system probe shows the precision budget).

**Scale choice: no rescaling.** No log distances, no inflated planets. At system distance a planet is far below a pixel, so it is drawn the way a camera records it: bodies under 0.75 render px are CPU-splatted points carrying their true flux p·E·(r/d)²·Φ(α) (Φ: Allen's lunar law for regolith bodies, 0.016 mag/deg for Mars, Lambert for the rest); the Sun goes through a lens PSF. That is why `SS1_overview` / `SS1_inner` are "a Sun and stars": they are honest. Readability comes from the separate `*_hud.png` annotation layer (orbit polylines at the epoch, markers, labels, caption, from the json; tools/solar_hud.py), never from lying in the render. `SS1_scale_strip` is the one synthetic layout (bodies placed side by side at true relative size and true colour) and says so.

**Rendering (render-ash `system.slang` / `system.rs`).** One full-screen pass, analytic ray–ellipsoid per body (≤ 16), f32 RGBA HDR target, timestamps + readback.
- Sun: limb darkening I = I0·μ^a, a = 0.45 / 0.62 / 0.85 (r/g/b); granulation map at 6% contrast; K+F corona (Baumbach) with noise streamers, visible only with an occulter (`SS1_corona`, 1.2 R☉ disc, 1.5·10⁶× the disc exposure).
- Planets: equirect layers (R8G8B8A8_SRGB, decoded by the sampler — an earlier double decode made Mars brick-red and Neptune navy, fixed), albedo scaled so the disc mean matches the Mallama geometric albedo; a lunar-Lambert / Lambert BRDF by surface class; a thin haze shell (Rayleigh-blue for the Earth, white elsewhere) for the limb.
- True-colour retints (the Solar System Scope maps are processed): Venus softened to 18% contrast toward (232,220,190); Mars toward (190,132,95); Uranus (172,211,216) and Neptune (150,190,216) re-coloured by luminance-preserving chroma replacement (a per-channel ratio blew the map's near-zero red into purple fringes). Neptune is the 2023-reprocessed "pale" colour (Irwin et al. 2024), not the Voyager-era deep blue.
- Saturn's rings: radial profile (opacity from the texture's alpha; RGB rebaked from Cassini-like natural-colour knots: C 140,132,124 / B 214,190,160 / Cassini Division 150,140,130 / A 196→188 / F 200,196,190 — the original RGB was dark blue), Chandrasekhar slab single scattering (lit face I/F = ω·P/4·μ0/(μ0+μ)·(1−e^{−τ(1/μ+1/μ0)}), diffuse transmission on the unlit face), radiance = E/π·I/F, P = 1.6·e^{−α/0.55}+0.25, ω_eff 2.2 calibrated to B-ring I/F ≈ 0.45 at low phase; globe shadow on the rings and ring shadow on the globe.
- Post: stars (Yale BSC5 into the ecliptic frame, true magnitudes), lens PSF, ACES, sRGB. PSF: Gaussian cores σ 1.5 / 6 px (2% / 0.3%) by separable blurs on the image clamped at 16; above 16 (the Sun) the excess gets cores 2% / 0.05% and a Moffat halo (β 1.5, r0 2 px, 0.002%) evaluated analytically per bright pixel in a circular, tapered window. (The first version blurred the 10⁷:1 Sun with box-Gaussians and showed their square support.)

**Earth handoff.** Below 24 px radius the Earth is the system pass's own LOD (Blue Marble July + the dated VIIRS clouds + a Rayleigh veil). Above, `solar` runs the `earth` example with the same camera expressed in ECEF (`--cam-ecef`, `--fov`, `STARS=0 HDR_OUT=1`, the ephemeris subsolar point) and composites the two HDR images: system pixels where a nearer body hides the Earth (alpha −1), otherwise earth + earth.transmittance × system. `SS1_path_06` is the E2_vs_epic camera: it differs from round 7's `E2_vs_epic` by a mean 0.52/255, from the subsolar point (ephemeris 21.475°N 14.613°E vs the EPIC metadata 21.537°N 14.219°E that E2 uses; ~0.4° in longitude, about 1.6 min of rotation) and the camera roll.
- New opt-ins in `earth.rs` (defaults byte-identical): `STARS=0` skips the star load; `HDR_OUT=1` writes `<out>/<name>.hdr.bin` (raw f32 RGBA, render size, before PROBE).

**Path.** `SS1_path_00..06`: log distance 50 AU → the EPIC frame (1.48·10⁶ km), fov 60° → 0.624°, target Sun → Earth (smoothstep over the first 55%), direction slerp; auto-exposure keyed to the Earth (2.4 once it is ≥ 2 px, (2/r_px)² brighter while smaller, capped at 3·10⁴). 05/06 are handoff composites; 00–04 have HUD versions.

**Shots** (`out/quality/steps/SS1_*.png` + `.json`; sheets `SS1_path_sheet.png`, `SS1_planets_sheet.png`): overview (50 AU, 60° above the ecliptic), inner (3.2 AU), path_00..06, sun (from 1 AU, fov 0.75°), corona, close-ups of Mercury … Neptune (phase 30–70°, exposure ∝ r²/p normalised to the Earth's), saturn_2017 (Cassini Grand Finale epoch, Sun 26° above the rings — the 2026 shot has the Sun only ~6.5° below the ring plane just after the 2025 equinox, so its rings are honestly dim), moon_earth (from beyond the Moon near the Sun line, ~1 day after new moon: both near full), scale_strip.

**Numbers (lavapipe, CPU Vulkan — not a GPU number; 3840×2160 render, ss 2, 1920×1080 out).** System pass 143–218 ms GPU (≈ 250–330 ms wall incl. readback); texture upload 117 MB in ~0.35–0.6 s; post (stars + PSF + tonemap) ~70–90 ms; handoff Earth render 3.4–9.3 s wall (the round-7 planet path). Vulkan validation: 0 errors in every shot.

**Identity.** Cave vs e3_cave_ref, GROUND=off vs e3 mountain refs, default mountains vs e6 refs, and `earth` defaults (E2_vs_epic / orbit_day / low_alps) vs the pre-SS1 binary (`/workspace/checkpoints/ss1_base`, rendered from the cpE71_final tree): **all byte-identical** (`cmp`), validation 0 in each (`/workspace/checkpoints/ss1_ident.log`). The round-2 E2 reference PNGs no longer exist on the box, so the "all features off == round 2" check of round 7 could not be repeated.

**Sources / licences.** Planet and Sun maps: Solar System Scope (CC BY 4.0, processed as above); Earth: NASA Blue Marble NG + VIIRS (public domain); ephemeris: JPL / Standish; orientation: IAU WGCCRE 2015; albedos: Mallama, Hilton & Krobusek 2017; Neptune/Uranus colour: Irwin et al. 2024 (MNRAS); stars: Yale BSC5.

**Known gaps (round 1).** No planetary satellites besides the Moon; no Uranus/Neptune/Jupiter rings; giant-planet atmospheres are textures + a haze shell, not scattering; no zodiacal light; the Sun's corona is a model, not a dated image; no eclipse shadows from moons on planets; the overview is CPU-splat + HUD rather than a real-time renderer; Earth below the handoff uses its own simple LOD (Blue Marble + clouds), so the colour step at 24 px is not checked.

**Box reinstall notes.** apt on the box is broken: fetch .debs with curl and `dpkg -i`; `/usr/bin/time` is absent; slangc in /usr/local/bin, spirv-val and lavapipe installed; `tools/compile_shaders.sh` regenerates the `shaders/spv` fallback (includes `system_vs` / `system_fs`).

**Gates** (logs `/workspace/checkpoints/ss1_*.log`): `cargo test --release --workspace` 149 passed / 0 failed, the same with `BRUTAL_SCAN=shared` (`KTX_TOOL` set); clippy `-D warnings` default and `--all-features`: clean (fixed four over-precise ephemeris check literals and six lints in `solar.rs`); `tools/compile_shaders.sh` OK (spv fallback regenerated); Vulkan validation 0 in all 21 SS1 shots and every identity render. New test: `system::layout_tests::params_layout_matches_slang`.

**Saturn 2026 shadow.** With the Sun 6.5° from the ring plane the globe's shadow on the rings is long: in `SS1_saturn` it cuts the left ansa with a straight edge out to the F ring. That is geometry, not clipping (compare `SS1_saturn_2017`, Sun 26° up, short shadow behind the globe).


## Solar system round 2: visibility layer, deep exposures, calibrated Earth handoff (1 Nov 2026)

```bash
export LD_LIBRARY_PATH=/workspace/tools_ext/KTX-Software-4.4.2-Linux-x86_64/lib:$LD_LIBRARY_PATH
cargo build --release -p render-ash --example earth --example solar
./target/release/examples/solar --shot SS2            # SS1 / all = the round-1 shots, unchanged
python3 tools/solar_vis.py out/quality/steps SS2_overview SS2_overview_deep SS2_inner SS2_inner_deep \
  SS2_path_00 SS2_path_01 SS2_path_02 SS2_path_03 SS2_path_04
EARTH_DISC=1 ./target/release/examples/solar --shot SS2_handoff_r24_lod,SS2_handoff_r24_planet   # disc-mean probe
```

**Goal.** Make the true-scale system readable without faking it. The physical image (true scale, true
brightness, EPIC_GRADE off) is untouched; everything that helps you read it sits on top as marked annotation
or in a panel below.

**Visibility layer (`<shot>_vis.png`, 1920×1380 = the 1080 p image + a 300 px panel).**
- Loupes: one per body under 12 px, a *true render* through a long lens, not an icon. The camera moves along
  the same line of sight, so phase and lighting are unchanged and the f32 shader keeps its precision. Exposure
  is 2.4·r², i.e. the irradiance is normalised to 1 AU and the albedo kept, so Neptune and Mercury share one
  scale. Each caption gives V mag, angular size, magnification, phase and distance. Large phase angles are
  real: from 50 AU, Jupiter (5.2 AU from the Sun, 47.6 AU from the camera) is at 115°, so it shows a crescent.
- Inner-system zoom inset: the same camera with a narrower fov (Mars's orbit fills it), its own orbits, a box
  on the main image and connecting lines.
- HUD v2 (PIL, 2× supersampled): orbits clipped to the image, label collision avoidance with leaders,
  "Earth + Moon" merged when they are closer than the label, and a scale bar in AU / km / light-time. The scale
  is taken at the distance of the body nearest the image centre (the Sun on overviews, Mercury, the Moon or the Earth on the
  approach) and labelled with that body.
- Deep exposures: a perfect occulter over the Sun plus a long exposure (coronagraph/LASCO style).
  `SS2_overview_deep` (occulter 5 px, ×300) shows Jupiter and Saturn as clear points and the BSC5 star field.
  `SS2_inner_deep` (occulter 14 px, ×30; at ×300 the K+F corona swamped the frame) shows the inner planets as
  points around a corona glow.

**Earth handoff calibration (`Shot::earth_cal`, SS2 shots only).** A disc-mean probe (`EARTH_DISC=1`) from
the EPIC direction at 24 px measured radiance per unit exposure:

| Earth source | disc mean | vs planet renderer |
|---|---|---|
| planet renderer (the Earth rounds' look) | 0.0513 | reference |
| system LOD (SS1) | 0.0708 | **+38 %**: SS1 had a visible brightness step at 24 px |
| flux point, Mallama p = 0.434 | 0.131 | +155 % |

The planet renderer is taken as the reference Earth. The LOD's Earth tint is now an output gain
(`system.slang`, kind 5; it multiplies the final radiance, so clouds and veil scale too). It is set to
[0.766, 0.715, 0.714]. The flux point uses the planet renderer's effective albedo (gain 0.392, so p ≈ 0.17) and its
disc colour. The handoff now cross-fades between 20 and 28 px instead of switching at 24 px. Result
(`SS2_handoff_sheet.png`): LOD vs planet within +4.5 % at 12 px, +0.9 % at 22 px, 0.0 % at 24 px and −0.2 % at 26 px;
the shipped cross-fade stays within 0.7 % across the band, against +38 % before. SS1 shots keep the old
values (tint 1, so this is exact).

**Shots / files** (`out/quality/steps/`): `SS2_{overview,inner,path_00..04}` (+ `_vis`, `_zoom`,
`_loupe_<body>`), `SS2_{overview,inner}_deep`, and `SS2_handoff_r{12,18,22,24,26,32}_{lod_ss1,lod,planet,auto}`.
Sheets: `SS2_vis_sheet.png`, `SS2_before_after.png` (SS1 HUD vs SS2 vis) and `SS2_handoff_sheet.png`. Report:
`out/quality/SS2_report.md`.

**Numbers (lavapipe).** System pass 149–185 ms GPU per main view; `--shot SS2` (33 shots, 52 loupe/zoom
renders, 24 Earth-renderer runs) takes 50 s wall. Validation errors are 0 on all 33.

**Identity.** `--shot SS1`: 21/21 PNGs are byte-identical to the pre-SS2 binary (`/workspace/checkpoints/ss2_base/out`,
`cmp`). One fix was needed: SS1_sun's camera sits at the Earth's centre, which gave a NaN radius, now treated as 0.

**Gates** (`/workspace/checkpoints/ss2_gates.log`): `cargo fmt --check`, clippy `-D warnings` (default and
`--all-features`), and `cargo test --release --workspace` with 149 passed / 0 failed.

**Open decision: absolute Earth albedo.** The planet renderer's Earth (exposure 2.4, tuned by eye to EPIC)
has an effective geometric albedo of about 0.17 at this phase. Mallama/Bond-consistent is 0.434. SS2 makes everything
consistent with the renderer, so the Earth is about 1 mag fainter than physical. The loupe captions still
print the Mallama V. The SS1 note "Moon albedo 0.12 vs Earth 0.43" is really ~0.12 vs ~0.17 as drawn. To make the
Earth physical, the whole Earth path (the planet renderer and the EPIC comparison exposure) would need re-scaling
by ~2.5×.

**Known gaps (round 2).** The point gain was calibrated at one phase (EPIC, ~5°), so the point↔disc match at
large phase angles is unverified. The Moon loupes show a small dark spot on the far side, most likely Tsiolkovsky
crater (not verified against the map). Uranus and Neptune stay faint even at ×300 (true magnitude ~8–10 at 50 AU),
so the loupes carry them. Stars are BSC5 (V ≤ 6.5) only, so deep frames lack fainter stars.
