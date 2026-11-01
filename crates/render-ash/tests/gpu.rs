//! GPU tests on whatever Vulkan 1.3 device is present (lavapipe on CI/headless boxes), with the
//! Khronos validation layer + synchronization validation enabled when installed.
//! Tests are serialized (one Vulkan device at a time) and each asserts zero validation errors.
use std::sync::Mutex;

use lowlevel_abi::{FrameSnapshot, RenderBackend};
use render_ash::cull::cpu_cull;
use render_ash::scene::{demo_view_proj, random_scene};
use render_ash::validation::message_counts;
use render_ash::{AshBackend, BackendConfig, FrameStats, ScanMode, ScanPath, cpu_exclusive_scan};

static GPU: Mutex<()> = Mutex::new(());

fn backend(w: u32, h: u32, max_instances: usize) -> Option<AshBackend> {
    backend_fif(w, h, max_instances, 2)
}

fn backend_fif(w: u32, h: u32, max_instances: usize, frames_in_flight: usize) -> Option<AshBackend> {
    backend_cfg(BackendConfig { max_instances, extent: (w, h), validation: true, frames_in_flight, ..Default::default() })
}

fn backend_cfg(cfg: BackendConfig) -> Option<AshBackend> {
    match AshBackend::new(cfg, None) {
        Ok(b) => {
            eprintln!(
                "device: {} | validation: {} | sync validation: {} | scan: {:?} | subgroup: {:?}",
                b.device_name(),
                b.ctx.validation_enabled,
                b.ctx.sync_validation,
                b.scan_path(),
                b.ctx.subgroup
            );
            Some(b)
        }
        Err(e) => {
            eprintln!("SKIP: no usable Vulkan 1.3 device: {e:?}");
            None
        }
    }
}

fn assert_no_validation_errors() {
    let (errors, warnings) = message_counts();
    assert_eq!(errors, 0, "validation errors reported (see stderr)");
    eprintln!("validation: {errors} errors, {warnings} warnings");
}

#[test]
fn gpu_visible_count_matches_cpu_reference() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut b) = backend(320, 240, 200_000) else { return };
    let meshes = b.mesh_table().to_vec();
    let mut exact = 0;
    let cases = [(1usize, 1u64), (7, 2), (1000, 3), (4096, 4), (20_000, 5), (65_537, 6), (200_000, 7), (50_000, 8)];
    for (frame, &(n, seed)) in cases.iter().enumerate() {
        let inst = random_scene(n, seed, meshes.len() as u32, 0.05);
        let vp = demo_view_proj(n, 320.0 / 240.0);
        b.render(&FrameSnapshot { view_proj: vp.to_cols_array_2d(), instances: &inst, frame_index: frame as u64 }).unwrap();
        let st = b.finish().unwrap();
        let reference = cpu_cull(&inst, &meshes, vp);
        eprintln!("n={n:>6} gpu_visible={:>6} cpu_inside={:>6} ambiguous={} draws={}", st.visible, reference.inside, reference.ambiguous, st.draws);
        assert!(reference.accepts(st.visible as usize), "n={n}: GPU {} vs CPU {:?}", st.visible, reference);
        assert!(st.draws as usize <= meshes.len());
        assert_eq!(st.instances as usize, n);
        if reference.ambiguous == 0 {
            assert_eq!(st.visible as usize, reference.inside);
            exact += 1;
        }
    }
    // A camera looking away from everything sees nothing.
    let inst = random_scene(5000, 9, meshes.len() as u32, 0.0);
    let away = render_ash::cull::vk_perspective(1.0, 1.0, 0.1, 50.0)
        * render_ash::cull::look_at(glam::Vec3::new(0.0, 0.0, 200.0), glam::Vec3::new(0.0, 0.0, 400.0), glam::Vec3::Y);
    b.render(&FrameSnapshot { view_proj: away.to_cols_array_2d(), instances: &inst, frame_index: 99 }).unwrap();
    let st = b.finish().unwrap();
    assert_eq!((st.visible, st.draws), (0, 0));
    eprintln!("{exact}/{} scenes matched exactly (no boundary-ambiguous instances)", cases.len());
    drop(b);
    assert_no_validation_errors();
}

#[test]
fn offscreen_readback_is_nonempty_and_resize_works() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut b) = backend(256, 192, 50_000) else { return };
    let inst = random_scene(20_000, 42, b.mesh_table().len() as u32, 0.0);
    // Several frames to exercise slot reuse + timeline pacing.
    for f in 0..5 {
        let vp = demo_view_proj(inst.len(), 256.0 / 192.0);
        b.render(&FrameSnapshot { view_proj: vp.to_cols_array_2d(), instances: &inst, frame_index: f }).unwrap();
    }
    let st = b.finish().unwrap();
    assert!(st.visible > 0);
    let (w, h, px) = b.read_color_rgba().unwrap();
    assert_eq!((w, h, px.len()), (256, 192, 256 * 192 * 4));
    let clear = [13u8, 15, 23]; // CLEAR color in UNORM8 (0.05, 0.06, 0.09)
    let lit = px.as_chunks::<4>().0.iter().filter(|p| p[..3].iter().zip(clear).any(|(a, b)| a.abs_diff(b) > 3)).count();
    eprintln!("non-clear pixels: {lit} / {}", w * h);
    assert!(lit > (w * h / 20) as usize, "image is (nearly) empty: {lit} lit pixels");

    b.resize(400, 300).unwrap();
    let vp = demo_view_proj(inst.len(), 400.0 / 300.0);
    b.render(&FrameSnapshot { view_proj: vp.to_cols_array_2d(), instances: &inst, frame_index: 10 }).unwrap();
    let (w, h, px) = b.read_color_rgba().unwrap();
    assert_eq!((w, h, px.len()), (400, 300, 400 * 300 * 4));
    drop(b);
    assert_no_validation_errors();
}

#[test]
fn timestamps_report_cull_and_draw_times() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut b) = backend(320, 240, 100_000) else { return };
    if b.ctx.timestamp_period_ns.is_none() {
        eprintln!("SKIP: device has no timestamp support");
        return;
    }
    let inst = random_scene(100_000, 3, b.mesh_table().len() as u32, 0.0);
    let vp = demo_view_proj(inst.len(), 320.0 / 240.0);
    let mut retired = Vec::new();
    for f in 0..6 {
        if let Some(s) = b.render(&FrameSnapshot { view_proj: vp.to_cols_array_2d(), instances: &inst, frame_index: f }).unwrap() {
            retired.push(s);
        }
    }
    retired.push(b.finish().unwrap());
    assert_eq!(retired.len(), 5, "frames 1..=4 retired by pacing + final finish");
    for s in &retired {
        let (c, d) = (s.cull_ms.unwrap(), s.draw_ms.unwrap());
        eprintln!("frame {}: cull {c:.3} ms, draw {d:.3} ms, visible {}", s.frame, s.visible);
        assert!(c > 0.0 && d > 0.0 && c < 10_000.0 && d < 10_000.0);
    }
    // Pacing retires frame N-2 when submitting frame N; frame 5 is still in flight when `finish`
    // waits for (and reports) frame 6. All frames saw the same scene.
    assert_eq!(retired.iter().map(|s| s.frame).collect::<Vec<_>>(), [1, 2, 3, 4, 6]);
    assert!(retired.windows(2).all(|w| w[1].visible == w[0].visible));
    drop(b);
    assert_no_validation_errors();
}

#[test]
fn rejects_too_many_instances() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut b) = backend(64, 64, 100) else { return };
    let inst = random_scene(101, 1, 4, 0.0);
    let r = b.submit(&FrameSnapshot { view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(), instances: &inst, frame_index: 0 });
    assert!(matches!(r, Err(render_ash::BackendError::TooManyInstances { .. })));
    drop(b);
    assert_no_validation_errors();
}

/// Small deterministic xorshift so the tests do not need a rand dependency.
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Backend with an explicit scan mode; `None` if the device lacks the subgroup path (reported).
fn scan_backend(mode: ScanMode, w: u32, h: u32, max_instances: usize) -> Option<AshBackend> {
    let cfg = BackendConfig { max_instances, extent: (w, h), validation: true, frames_in_flight: 2, scan: mode };
    match AshBackend::new(cfg, None) {
        Err(render_ash::BackendError::Unsupported(why)) if mode == ScanMode::Subgroup => {
            eprintln!("SKIP subgroup path: {why}");
            None
        }
        _ => backend_cfg(cfg),
    }
}

#[test]
fn gpu_scan_matches_cpu_scan() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    for mode in [ScanMode::Subgroup, ScanMode::SharedMemory] {
        let Some(mut b) = scan_backend(mode, 16, 16, 16) else { continue };
        match mode {
            ScanMode::SharedMemory => assert_eq!(b.scan_path(), ScanPath::SharedMemory),
            _ => assert!(matches!(b.scan_path(), ScanPath::Subgroup { .. })),
        }
        scan_sizes_match(&mut b);
        drop(b);
        assert_no_validation_errors();
    }
}

fn scan_sizes_match(b: &mut AshBackend) {
    let tile = render_ash::SCAN_TILE as usize;
    // 0, 1, non-power-of-two, exactly one tile, one-past (2 levels), multi-workgroup, 1M (2 levels,
    // 1024 tiles), 1M+1 (3 levels).
    let mut sizes = vec![0, 1, 2, 3, 7, 255, 1000, tile - 1, tile, tile + 1, 3 * tile + 17, 65_537, 300_001, 1 << 20, (1 << 20) + 1];
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..6 {
        sizes.push((xorshift(&mut rng) % 200_000) as usize);
    }
    for &n in &sizes {
        // Mix small values with occasional huge ones so the u32 sums wrap (GPU and CPU must agree).
        let data: Vec<u32> = (0..n)
            .map(|_| {
                let r = xorshift(&mut rng);
                if r.is_multiple_of(97) { (r >> 32) as u32 } else { (r % 1000) as u32 }
            })
            .collect();
        let (gpu, gpu_ms) = b.scan_u32_timed(&data).unwrap();
        let cpu = cpu_exclusive_scan(&data);
        assert_eq!(gpu.len(), n);
        if let Some(i) = gpu.iter().zip(&cpu).position(|(a, b)| a != b) {
            panic!("{:?} n={n}: first mismatch at {i}: gpu {} cpu {}", b.scan_path(), gpu[i], cpu[i]);
        }
        eprintln!("scan {:?} n={n:>8}: ok (GPU {:.3} ms)", b.scan_path(), gpu_ms.unwrap_or(f64::NAN));
    }
}

#[test]
fn many_meshes_cull_matches_cpu_reference() {
    // > SCAN_TILE meshes, so the per-frame per-mesh scan runs 2 levels inside the real pass; on
    // both scan paths.
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    for mode in [ScanMode::Subgroup, ScanMode::SharedMemory] {
        let Some(b) = scan_backend(mode, 320, 240, 100_000) else { continue };
        many_meshes_case(b);
    }
}

fn many_meshes_case(mut b: AshBackend) {
    let base = render_ash::mesh::builtin_meshes();
    let meshes: Vec<_> = (0..3000).map(|i| base[i % base.len()].clone()).collect();
    b.set_meshes(&meshes).unwrap();
    let table = b.mesh_table().to_vec();
    assert_eq!(table.len(), 3000);
    for (f, &(n, seed)) in [(100_000usize, 11u64), (2500, 12), (1, 13)].iter().enumerate() {
        let inst = random_scene(n, seed, 3000, 0.02);
        let vp = demo_view_proj(n, 320.0 / 240.0);
        b.render(&FrameSnapshot { view_proj: vp.to_cols_array_2d(), instances: &inst, frame_index: f as u64 }).unwrap();
        let st = b.finish().unwrap();
        let r = cpu_cull(&inst, &table, vp);
        let used = {
            let mut seen = vec![false; 3000];
            // upper bound on draws: distinct meshes among visible/ambiguous instances
            for i in &inst {
                if (i.mesh_id as usize) < seen.len() {
                    seen[i.mesh_id as usize] = true;
                }
            }
            seen.iter().filter(|&&x| x).count()
        };
        eprintln!("{:?} meshes=3000 n={n:>6} gpu_visible={} cpu_inside={} ambiguous={} draws={}", b.scan_path(), st.visible, r.inside, r.ambiguous, st.draws);
        assert!(r.accepts(st.visible as usize), "n={n}: GPU {} vs CPU {r:?}", st.visible);
        assert!(st.draws as usize <= used);
        if n >= 2500 {
            assert!(st.draws > 1024, "expected many non-empty meshes, got {}", st.draws);
        }
    }
    drop(b);
    assert_no_validation_errors();
}

/// Proves frame overlap, deterministically and on any driver:
///  * frame N is gated on a host-controlled timeline semaphore, then frames N+1 .. N+F-1 are rendered.
///    `render()` returns for each of them while N is provably not retired (the backend's timeline is
///    still < N), i.e. frame N+1 was recorded and submitted before N retired and nothing in
///    `render()` serialises on the previous frame (only on slot reuse, F frames back);
///  * every frame used a different scene and is checked against the CPU reference after the gate
///    opens, which would fail if in-flight frames shared (and clobbered) scratch buffers;
///  * an ungated run never exceeds F frames in flight.
///
/// Sync validation is on for all of it and must report 0 errors.
#[test]
fn frames_overlap_and_stay_correct() {
    use ash::vk;
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    for fif in [2usize, 3] {
        let Some(mut b) = backend_fif(320, 240, 120_000, fif) else { return };
        assert_eq!(b.frames_in_flight(), fif);
        let meshes = b.mesh_table().to_vec();
        let scenes: Vec<_> = (0..9u64).map(|k| random_scene(60_000 + 7_000 * k as usize, 100 + k, meshes.len() as u32, 0.03)).collect();
        let vp = |n: usize| demo_view_proj(n, 320.0 / 240.0);
        let mut stats: Vec<FrameStats> = Vec::new();
        let render = |b: &mut AshBackend, k: usize| {
            let s = &scenes[k];
            b.render(&FrameSnapshot { view_proj: vp(s.len()).to_cols_array_2d(), instances: s, frame_index: k as u64 }).unwrap()
        };

        // Gated phase: frame 1 waits on `gate`.
        let mut tci = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
        // SAFETY: device is live; timeline semaphores are enabled by the backend.
        let gate = unsafe { b.ctx.device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut tci), None).unwrap() };
        b.gate_next_submit(gate, 1);
        stats.extend(render(&mut b, 0));
        for k in 1..fif {
            stats.extend(render(&mut b, k));
            let done = b.completed_frame().unwrap();
            assert_eq!(done, 0, "frame {} submitted, frame 1 must still be pending (timeline = {done})", k + 1);
        }
        let held: Vec<u64> = (1..=fif as u64).collect();
        // SAFETY: semaphore is live; 1 > current value 0.
        unsafe { b.ctx.device.signal_semaphore(&vk::SemaphoreSignalInfo::default().semaphore(gate).value(1)).unwrap() };

        // Ungated phase.
        for k in fif..scenes.len() {
            stats.extend(render(&mut b, k));
        }
        stats.extend(b.drain().unwrap());
        // SAFETY: drain() waited for every submit, including the one that waited on `gate`.
        unsafe { b.ctx.device.destroy_semaphore(gate, None) };

        assert_eq!(stats.iter().map(|s| s.frame).collect::<Vec<_>>(), (1..=9).collect::<Vec<_>>());
        for (s, inst) in stats.iter().zip(&scenes) {
            let r = cpu_cull(inst, &meshes, vp(inst.len()));
            assert_eq!(s.instances as usize, inst.len());
            assert!(r.accepts(s.visible as usize), "frame {}: GPU {} vs CPU {r:?}", s.frame, s.visible);
        }
        for s in &stats[..fif] {
            assert_eq!(s.completed_at_submit, 0, "frames {held:?} were all submitted while frame 1 was pending");
            assert_eq!(s.in_flight_at_submit, s.frame);
        }
        let in_flight: Vec<u64> = stats.iter().map(|s| s.in_flight_at_submit).collect();
        assert!(in_flight.iter().all(|&x| x as usize <= fif), "pacing must cap frames in flight at F: {in_flight:?}");
        let gpu_overlap = stats
            .windows(2)
            .filter(|w| matches!((w[1].gpu_begin_ns, w[0].gpu_end_ns), (Some(b1), Some(e0)) if b1 < e0))
            .count();
        eprintln!("F={fif}: frames {held:?} in flight together | in_flight_at_submit={in_flight:?} | GPU timestamp overlap {gpu_overlap}/8 (driver-dependent)");
        drop(b);
        assert_no_validation_errors();
    }
}

#[test]
fn rejects_bad_frames_in_flight() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    for fif in [0usize, 1, 4] {
        let r = AshBackend::new(BackendConfig { max_instances: 16, extent: (16, 16), validation: false, frames_in_flight: fif, ..Default::default() }, None);
        assert!(matches!(r, Err(render_ash::BackendError::BadConfig(_))), "fif={fif}");
    }
}

/// Terrain residency: a slot read by an unretired frame is never evicted (frame 1 is gated, so it
/// provably has not retired), eviction prefers LRU slots outside the draw list, and the drawn
/// terrain shows up in the readback. Validation + sync validation clean.
#[test]
fn terrain_eviction_waits_for_retired_frames() {
    use ash::vk;
    use render_ash::{TerrainGpuConfig, UploadOutcome};
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut b) = backend(160, 120, 16) else { return };
    b.enable_terrain(TerrainGpuConfig { slots: 3, staging_bytes: 1 << 20, ..Default::default() }, &terrain::grid_indices()).unwrap();
    let p = terrain::NoiseParams::default();
    let chunk = |c: [i32; 2]| {
        let req = terrain::ChunkRequest { coord: c, lod: 0, seed: 3 };
        let mut hm = vec![0.0; terrain::PADDED * terrain::PADDED];
        terrain::heightmap_simd(&req, &p, &mut hm);
        let arena = mem::FrameArena::with_capacity(terrain::CHUNK_SAMPLES * terrain::CHUNK_SAMPLES * 24 + 64);
        terrain::mesh_chunk(&req, &hm, &arena).unwrap().vertices.to_vec()
    };
    let vp = (render_ash::cull::vk_perspective(1.2, 160.0 / 120.0, 0.5, 1000.0)
        * render_ash::cull::look_at(glam::Vec3::new(64.0, 140.0, -50.0), glam::Vec3::new(64.0, 0.0, 64.0), glam::Vec3::Y))
    .to_cols_array_2d();
    let frame = |b: &mut AshBackend, i: u64| b.render(&FrameSnapshot { view_proj: vp, instances: &[], frame_index: i }).unwrap();
    let (a, bb, c, d) = ([0, 0], [1, 0], [0, 1], [1, 1]);
    for k in [a, bb, c] {
        assert!(matches!(b.upload_terrain_chunk(k, &chunk(k)).unwrap(), UploadOutcome::Staged { evicted: None, .. }));
    }
    assert!(matches!(b.upload_terrain_chunk(a, &chunk(a)).unwrap(), UploadOutcome::AlreadyResident { .. }));
    assert_eq!(b.set_terrain_draw(&[a]), 1);

    let mut tci = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
    // SAFETY: device live; timeline semaphores enabled.
    let gate = unsafe { b.ctx.device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut tci), None).unwrap() };
    b.gate_next_submit(gate, 1);
    frame(&mut b, 0); // frame 1: copies a, b, c; draws a. Held by the gate.
    assert!(b.upload_terrain_chunk(d, &chunk(d)).is_err(), "all slots were written by the unretired frame 1");
    frame(&mut b, 1); // frame 2 (other slot, no wait): draws a
    assert_eq!(b.completed_frame().unwrap(), 0);
    assert!(b.upload_terrain_chunk(d, &chunk(d)).is_err(), "frame 1 still pending");
    // SAFETY: semaphore live; 1 > 0.
    unsafe { b.ctx.device.signal_semaphore(&vk::SemaphoreSignalInfo::default().semaphore(gate).value(1)).unwrap() };
    b.drain().unwrap();
    // SAFETY: drain waited for the submit that waited on the gate.
    unsafe { b.ctx.device.destroy_semaphore(gate, None) };

    match b.upload_terrain_chunk(d, &chunk(d)).unwrap() {
        UploadOutcome::Staged { evicted: Some(e), .. } => assert!(e == bb || e == c, "evicted {e:?}; must not evict drawn chunk a"),
        o => panic!("expected an eviction, got {o:?}"),
    }
    assert!(b.terrain_is_resident(a) && b.terrain_is_resident(d));
    assert_eq!(b.set_terrain_draw(&[a, bb, c, d]), 3);
    for i in 2..5 {
        frame(&mut b, i);
    }
    let (_, _, px) = b.read_color_rgba().unwrap();
    let clear = [13u8, 15, 23];
    let lit = px.as_chunks::<4>().0.iter().filter(|q| q[..3].iter().zip(clear).any(|(x, y)| x.abs_diff(y) > 3)).count();
    eprintln!("terrain: lit pixels {lit}/{}; stats {:?}", 160 * 120, b.terrain_stats());
    assert!(lit > 160 * 120 / 20, "terrain not visible ({lit} lit)"); // 3 chunks seen from 180 m: ~10%
    let st = b.terrain_stats().unwrap();
    assert_eq!((st.uploads, st.evictions), (4, 1));
    drop(b);
    assert_no_validation_errors();
}

/// Prompt C2: the Slang Transvoxel mesher reproduces the CPU integer mesher exactly (positions
/// bit-identical, same triangle order; normals within 1e-4) on uniform, slab, camera-centred
/// shell and adversarial-fuzz layouts, and its output is watertight on its own.
#[test]
fn gpu_transvoxel_matches_cpu_exactly() {
    use render_ash::voxel_gpu::VoxelJob;
    use voxel::scene::{Adversarial, BlobScene};
    use voxel::*;
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut b) = backend(64, 64, 16) else { return };
    let cfg = VoxelConfig::new(coords::BrickGrid::new(16, 0.25).unwrap());
    type Case = (&'static str, Box<dyn DensityField>, Box<dyn LodLayout>);
    let cases: Vec<Case> = vec![
        ("uniform", Box::new(BlobScene { radius: 9.0, seed: 3 }), Box::new(UniformLod(0))),
        ("slab", Box::new(BlobScene { radius: 9.0, seed: 3 }), Box::new(SlabLayout { axis: 1, split: 0, lod_below: 1, lod_above: 0 })),
        ("shells", Box::new(BlobScene { radius: 9.0, seed: 3 }), Box::new(ShellLayout::camera_centred(&cfg.grid, [5, -3, 7], &[1, 1]))),
        ("fuzz-slab", Box::new(Adversarial { seed: 1, voxel_m: 0.25, radius_m: 7.0 }), Box::new(SlabLayout { axis: 1, split: 0, lod_below: 1, lod_above: 0 })),
        ("fuzz-shells", Box::new(Adversarial { seed: 4, voxel_m: 0.25, radius_m: 7.0 }), Box::new(ShellLayout::camera_centred(&cfg.grid, [0, 0, 0], &[1]))),
    ];
    let r = 14.0 / 0.25;
    let region = ([-(r as i64) / 64 * 64 - 64; 3], [(r as i64) / 64 * 64 + 64; 3]);
    for (name, field, lay) in &cases {
        let mut samples = Vec::new();
        let mut plans = Vec::new();
        for bc in bricks_in(&cfg.grid, lay.as_ref(), region, 2) {
            let (mask, finer) = plan_brick(&cfg.grid, lay.as_ref(), bc).unwrap();
            let s = sample_brick(field.as_ref(), &cfg, bc, mask).unwrap();
            if s.has_sign_change() {
                samples.push(s);
                plans.push((mask, finer));
            }
        }
        let jobs: Vec<VoxelJob<'_>> =
            samples.iter().zip(&plans).map(|(s, &(m, f))| VoxelJob { samples: s, transitions: m, finer: f }).collect();
        let gpu = b.mesh_voxels_gpu(&cfg, &jobs).unwrap();
        let (mut tris, mut max_dn) = (0usize, 0f32);
        let mut soup = Vec::new();
        for (j, g) in jobs.iter().zip(&gpu.bricks) {
            let cpu = mesh_brick(&cfg, j.samples, j.transitions, j.finer);
            let pos: Vec<[i64; 3]> = cpu.indices.iter().map(|&i| cpu.ipos[i as usize]).collect();
            assert_eq!(pos, g.ipos, "{name}: brick {:?} positions differ", j.samples.coord);
            for (&i, gn) in cpu.indices.iter().zip(&g.normals) {
                let cn = cpu.vertices[i as usize].normal;
                for k in 0..3 {
                    max_dn = max_dn.max((cn[k] - gn[k]).abs());
                }
            }
            tris += cpu.indices.len() / 3;
            let n = g.ipos.len() as u32;
            soup.push(BrickMesh {
                ipos: g.ipos.clone(),
                vertices: g.normals.iter().map(|&nm| VoxelVertex { pos: [0.0; 3], normal: nm }).collect(),
                // edge directions from the CPU mesh (positions are proven identical just above)
                dirs: cpu.indices.iter().map(|&i| cpu.dirs[i as usize]).collect(),
                indices: (0..n).collect(),
            });
        }
        assert_eq!(gpu.triangles, tris, "{name}");
        assert!(max_dn < 1e-4, "{name}: normal deviation {max_dn}");
        let t = topology(soup.iter());
        assert!(t.boundary_edges == 0 && t.non_manifold_edges == 0 && t.orientation_conflicts == 0, "{name}: {t:?}");
        assert!(t.degenerate == 0 && t.flipped == 0 && t.unverified_components == 0, "{name}: {t:?}");
        eprintln!(
            "{name}: {} bricks, {tris} tris, GPU {:.2} ms, {} KiB device, max |dn| {max_dn:.2e}, closed",
            jobs.len(),
            gpu.gpu_ms.unwrap_or(f64::NAN),
            gpu.device_bytes / 1024
        );
    }
    drop(b);
    assert_no_validation_errors();
}

/// TurboQuant study (starter.md "Learning from turbovec and TurboQuant"): the Slang TQ decoder,
/// reading the verified blob words straight from the upload, reproduces the CPU reference
/// decoder bit-exactly at 1..=4 bits (real sampled bricks + a full-range adversarial brick), and
/// corrupt blobs are rejected before upload.
#[test]
fn gpu_tq_decode_matches_cpu_exactly() {
    use asset::tq::{self, Rotation, TqView};
    use voxel::scene::BlobScene;
    use voxel::*;
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut b) = backend(64, 64, 16) else { return };
    let cfg = VoxelConfig::new(coords::BrickGrid::new(16, 0.25).unwrap());
    let field = BlobScene { radius: 9.0, seed: 3 };
    let mut bricks: Vec<Vec<i32>> = bricks_in(&cfg.grid, &UniformLod(0), ([-64; 3], [64; 3]), 0)
        .into_iter()
        .map(|bc| sample_brick(&field, &cfg, bc, 0).unwrap())
        .filter(|s| s.has_sign_change())
        .map(|s| s.d)
        .collect();
    // adversarial: full +-DQ_MAX range, alternating extremes (exercises the clamp)
    let m = 19usize.pow(3);
    bricks.push((0..m).map(|k| if (k * 2_654_435_761) % 7 < 3 { DQ_MAX } else { -DQ_MAX + (k as i32 % 1000) }).collect());
    let rot = Rotation::new();
    for bits in 1..=4 {
        let blobs: Vec<Vec<u8>> = bricks.iter().map(|d| tq::encode_values(d, 16, bits, &rot)).collect();
        let refs: Vec<&[u8]> = blobs.iter().map(Vec::as_slice).collect();
        let gpu = b.decode_tq_gpu(&refs, &rot).unwrap();
        let cpu: Vec<i32> = blobs.iter().flat_map(|bl| TqView::parse(bl).unwrap().decode(&rot)).collect();
        assert_eq!(gpu.samples.len(), cpu.len());
        let diff = gpu.samples.iter().zip(&cpu).filter(|(a, b)| a != b).count();
        assert_eq!(diff, 0, "bits {bits}: {diff} samples differ GPU vs CPU");
        eprintln!(
            "tq {bits}-bit: {} bricks, {} samples bit-exact, upload {} KiB vs raw {} KiB, GPU {:.3} ms",
            bricks.len(),
            cpu.len(),
            gpu.upload_bytes / 1024,
            gpu.raw_bytes / 1024,
            gpu.gpu_ms.unwrap_or(f64::NAN)
        );
    }
    let mut bad = tq::encode_values(&bricks[0], 16, 4, &rot);
    bad[100] ^= 0x10;
    assert!(b.decode_tq_gpu(&[&bad], &rot).is_err(), "corrupt blob must be rejected");
    assert_no_validation_errors();
}

/// The GPU Transvoxel mesher on band-clamped payloads (asset format minor 2, read back through
/// `MeshSamples`) emits exactly what it emits on full density, transitions included.
#[test]
fn gpu_mesher_on_banded_payload_matches_full() {
    use asset::fb::{AlignedBytes, Limits};
    use asset::schema::{File, Writer};
    use render_ash::voxel_gpu::VoxelJob;
    use voxel::scene::BlobScene;
    use voxel::*;
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut b) = backend(64, 64, 16) else { return };
    let cfg = VoxelConfig::new(coords::BrickGrid::new(16, 0.25).unwrap());
    let lay = ShellLayout::camera_centred(&cfg.grid, [5, -3, 7], &[1, 1]);
    let field = BlobScene { radius: 9.0, seed: 3 };
    let mut src = Vec::new();
    for bc in bricks_in(&cfg.grid, &lay, ([-128; 3], [128; 3]), 2) {
        let (mask, finer) = plan_brick(&cfg.grid, &lay, bc).unwrap();
        let s = sample_brick_smooth(&field, &cfg, &lay, bc, mask).unwrap();
        if s.has_sign_change() {
            src.push((s, mask, finer));
        }
    }
    let mut w = Writer::new(cfg.grid);
    for (s, _, _) in &src {
        w.add_samples_banded(s, 1024);
    }
    let buf = AlignedBytes::from_slice(&w.finish());
    let f = File::open_deep(buf.as_slice(), Limits::default()).unwrap();
    let banded: Vec<_> = (0..src.len()).map(|i| f.decode_mesh_samples(i).unwrap().unwrap()).collect();
    let full_jobs: Vec<VoxelJob<'_>> = src.iter().map(|(s, m, fi)| VoxelJob { samples: s, transitions: *m, finer: *fi }).collect();
    let band_jobs: Vec<VoxelJob<'_>> =
        banded.iter().zip(&src).map(|(m, (_, _, fi))| VoxelJob { samples: m.samples_for_meshing_only(), transitions: m.transitions(), finer: *fi }).collect();
    let (a, c) = (b.mesh_voxels_gpu(&cfg, &full_jobs).unwrap(), b.mesh_voxels_gpu(&cfg, &band_jobs).unwrap());
    assert_eq!(a.triangles, c.triangles);
    for (x, y) in a.bricks.iter().zip(&c.bricks) {
        assert_eq!(x.ipos, y.ipos);
        assert_eq!(x.normals, y.normals);
    }
    let with_t = src.iter().filter(|s| s.1 != 0).count();
    assert!(with_t > 0);
    eprintln!("GPU mesher: {} bricks ({with_t} with transitions), {} tris, banded == full", src.len(), a.triangles);
    assert_no_validation_errors();
}

/// Round 9 performance paths of the quality renderer on a small synthetic scene (bumpy ground,
/// 40 boxes with a crossfading LOD chain, four sun cascades): the foliage depth prepass gives the
/// same image as plain forward shading, and the static shadow cascade cache reuses exactly the
/// keyed cascades, redraws on a key change, and never changes the image.
#[test]
fn quality_depth_prepass_and_shadow_cache() {
    use glam::{DVec3, Mat4, Vec3};
    use lowlevel_abi::InstanceGpu;
    use render_ash::mesh::MeshInfoGpu;
    use render_ash::quality::{InstancedSet, LOD_NONE, QParamsGpu, QVertex, QualityFrame, ShadowView, reverse_z_infinite, sun_cascade};
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut b) = backend(64, 64, 16) else { return };
    let (w, h) = (192u32, 128u32);
    // ground: 80 x 80 quads of 2 m around the camera, 1.6 m below it
    let n = 80u32;
    let gy = |x: f32, z: f32| -1.6 + 0.4 * (x * 0.21).sin() * (z * 0.17).cos();
    let mut verts = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let (x, z) = ((i as f32 - 40.0) * 2.0, (j as f32 - 40.0) * 2.0);
            verts.push(QVertex { pos: [x, gy(x, z), z], nrm: [0.0, 1.0, 0.0], attr: [1.0, 0.0, 0.0, 0.0] });
        }
    }
    let mut idx = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let a = j * (n + 1) + i;
            idx.extend([a, a + n + 1, a + 1, a + 1, a + n + 1, a + n + 2]);
        }
    }
    // instanced mesh 0 (box) -> mesh 1 (smaller box) at 18 m, 10 m crossfade
    let mut iv = Vec::new();
    let mut ii = Vec::new();
    let mut meshes = Vec::new();
    for (k, s) in [0.5f32, 0.4].into_iter().enumerate() {
        let base = iv.len() as u32;
        let first = ii.len() as u32;
        for c in 0..8u32 {
            let p = [if c & 1 != 0 { s } else { -s }, if c & 2 != 0 { 2.0 * s } else { 0.0 }, if c & 4 != 0 { s } else { -s }];
            iv.push(QVertex { pos: p, nrm: Vec3::from_array(p).normalize().to_array(), attr: [1.0, 0.0, 0.0, 2.0 + k as f32] });
        }
        for f in [[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]] {
            ii.extend(f.map(|v| base + v));
        }
        meshes.push(MeshInfoGpu { index_count: 36, first_index: first, vertex_offset: 0, _pad: 0, sphere: [0.0, s, 0.0, 2.3 * s] });
    }
    let lod = vec![[1, 18f32.to_bits(), 10f32.to_bits(), 0], [LOD_NONE, 0, 0, 0]];
    let inst: Vec<InstanceGpu> = (0..40)
        .map(|k| {
            let (x, z) = (((k * 37) % 23) as f32 - 11.0, -3.0 - k as f32 * 0.9);
            let m = Mat4::from_translation(Vec3::new(x, gy(x, z) - 0.05, z)) * Mat4::from_rotation_y(k as f32 * 0.7);
            InstanceGpu { model: m.to_cols_array_2d(), mesh_id: 0, material_id: 0, _pad: [0; 2] }
        })
        .collect();
    let fov = 1.0f32;
    let aspect = w as f32 / h as f32;
    let fwd = Vec3::new(0.0, -0.15, -1.0).normalize();
    let view = render_ash::cull::look_at(Vec3::ZERO, fwd, Vec3::Y);
    let vp = reverse_z_infinite(fov, aspect, 0.1) * view;
    let cull_vp = glam::camera::rh::proj::vulkan::perspective(fov, aspect, 0.1, 500.0) * view;
    let sun = Vec3::new(0.5, 0.6, 0.3).normalize();
    let origin = DVec3::new(1234.5, 300.0, -876.25);
    let (res, splits) = (512u32, [12.0f32, 40.0, 120.0, 400.0]);
    let mut shadows = Vec::new();
    let mut cells = Vec::new();
    let mut nd = 0.1;
    for f in splits {
        let (m, texel, cell) = sun_cascade(origin, fwd, fov, aspect, nd, f, sun, res, 200.0);
        shadows.push(ShadowView { view_proj: m, texel });
        cells.push(cell);
        nd = f * 0.8;
    }
    let mut p = QParamsGpu { view_proj: vp.to_cols_array_2d(), inv_dir: cull_vp.inverse().to_cols_array_2d(), ..Default::default() };
    p.cam_world = [origin.x as f32, origin.y as f32, origin.z as f32, fov / h as f32];
    p.sun_dir = [sun.x, sun.y, sun.z, 5.0];
    p.sun_color = [1.0, 0.9, 0.8, 0.6];
    p.sky_zenith = [0.2, 0.35, 0.7, 0.0];
    p.sky_horizon = [0.6, 0.7, 0.8, 0.0];
    p.fog_color = [0.5, 0.6, 0.7, 0.5];
    p.shadow_info[2] = 1.0;
    p.splits = splits;
    let render = |b: &mut AshBackend, prepass: bool, keys: &[u64]| {
        let frame = QualityFrame {
            width: w,
            height: h,
            vertices: &verts,
            indices: &idx,
            params: p,
            shadow_res: res,
            shadows: &shadows,
            spot_shadow: false,
            instanced: Some(InstancedSet { vertices: &iv, indices: &ii, meshes: &meshes, instances: &inst, cull_view_proj: cull_vp, lod: &lod, impostor_first: 2, atlas: None }),
            atmosphere: false,
            terrain_chunks: &[],
            shadow_slices: &[],
            depth_prepass: prepass,
            impostor_card_depth: false,
            shadow_cache_keys: keys,
        };
        b.render_quality(&frame).expect("quality frame")
    };
    let (reference, t0) = render(&mut b, false, &[]);
    assert!(t0.fading_instances > 0 && t0.visible_instances > t0.fading_instances, "the scene exercises whole and crossfading (dithered) draws: {t0:?}");
    assert!(reference.chunks(4).any(|px| px[..3] != reference[..3]), "non-uniform image");
    let (img, t) = render(&mut b, true, &[]);
    assert_eq!(t.cached_cascades, 0);
    let diff = img.chunks(4).zip(reference.chunks(4)).filter(|(a, b)| a != b).count();
    assert_eq!(diff, 0, "depth prepass changes {diff} pixels");
    let keys = [0, 0, 0x5eed_0002, 0x5eed_0003];
    let (img, t) = render(&mut b, true, &keys);
    assert_eq!(t.cached_cascades, 0, "nothing cached yet");
    assert!(img == reference);
    let (img, t) = render(&mut b, true, &keys);
    assert_eq!(t.cached_cascades, 2, "the two keyed cascades are reused");
    assert!(img == reference, "a cached cascade gives the same image");
    let (img, t) = render(&mut b, false, &[0, 0, 0x5eed_0002, 0x5eed_0004]);
    assert_eq!(t.cached_cascades, 1, "a changed key redraws that cascade only");
    assert!(img == reference);
    let (img, t) = render(&mut b, true, &[]);
    assert_eq!(t.cached_cascades, 0, "no keys: every cascade redrawn");
    assert!(img == reference);
    // world anchoring: a camera move inside one texel keeps the cell and the world-space matrix
    let (m0, texel, c0) = sun_cascade(origin, fwd, fov, aspect, 120.0 * 0.8, 400.0, sun, res, 200.0);
    let mut moved = None;
    for k in 1..400 {
        let o = origin + DVec3::new(0.013 * k as f64 * texel as f64, 0.0, 0.0);
        let (m1, _, c1) = sun_cascade(o, fwd, fov, aspect, 120.0 * 0.8, 400.0, sun, res, 200.0);
        if c1 == c0 {
            // world point X: camera-relative X - origin / X - o; m0 * (X - origin) == m1 * (X - o)
            let w0 = m0;
            let w1 = m1 * Mat4::from_translation((origin - o).as_vec3());
            assert!(w0.abs_diff_eq(w1, 2e-4), "same cell -> same world light matrix");
        } else {
            moved = Some(k);
            break;
        }
    }
    assert!(moved.is_some(), "moving the camera by a texel changes the cell");
    assert!(cells.iter().all(|c| c.iter().all(|&x| x != i64::MIN)));
    // the closed-form radius does not depend on the view direction
    let (_, t2, _) = sun_cascade(origin, Vec3::new(0.7, 0.1, 0.3).normalize(), fov, aspect, 120.0 * 0.8, 400.0, sun, res, 200.0);
    assert_eq!(texel, t2);
    drop(b);
    assert_no_validation_errors();
}

/// Round 11: card-depth impostors (`impostor_card_depth`) in the foliage depth prepass
/// (`impDepthFS`, then `impEqFS` with depth EQUAL and no discards). The prepass must not change
/// the image (no holes where the two shaders' coverage could differ, no lost crossfade dither),
/// and card depth must keep the forest's coverage of the exact-depth path.
#[test]
fn quality_card_depth_impostor_prepass() {
    use glam::{Mat4, Vec3};
    use lowlevel_abi::InstanceGpu;
    use render_ash::mesh::MeshInfoGpu;
    use render_ash::quality::{InstancedSet, LOD_NONE, QParamsGpu, QVertex, QualityFrame, ShadowView, reverse_z_infinite, sun_cascade};
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut b) = backend(64, 64, 512) else { return };
    let (w, h) = (256u32, 128u32);
    // ground plane
    let mut verts = Vec::new();
    let mut idx = Vec::new();
    let n = 40u32;
    for j in 0..=n {
        for i in 0..=n {
            let (x, z) = ((i as f32 - 20.0) * 6.0, -(j as f32) * 6.0 + 6.0);
            verts.push(QVertex { pos: [x, -2.0, z], nrm: [0.0, 1.0, 0.0], attr: [1.0, 0.0, 0.0, 0.0] });
        }
    }
    for j in 0..n {
        for i in 0..n {
            let a = j * (n + 1) + i;
            idx.extend([a, a + n + 1, a + 1, a + 1, a + n + 1, a + n + 2]);
        }
    }
    // mesh 0: a closed 12-sided cone "tree" (unit height); mesh 1: its impostor card
    let (mut iv, mut ii) = (Vec::new(), Vec::new());
    let sides = 12u32;
    iv.push(QVertex { pos: [0.0, 1.0, 0.0], nrm: [0.0, 1.0, 0.0], attr: [1.0, 1.0, 0.0, 2.0] });
    for k in 0..sides {
        let a = k as f32 / sides as f32 * std::f32::consts::TAU;
        let d = Vec3::new(a.cos(), 0.35, a.sin()).normalize();
        iv.push(QVertex { pos: [0.3 * a.cos(), 0.1, 0.3 * a.sin()], nrm: d.to_array(), attr: [0.6, 0.1, 0.0, 2.0] });
    }
    iv.push(QVertex { pos: [0.0, 0.1, 0.0], nrm: [0.0, -1.0, 0.0], attr: [0.3, 0.1, 0.0, 2.0] });
    for k in 0..sides {
        let (a, c) = (1 + k, 1 + (k + 1) % sides);
        ii.extend([0, c, a, sides + 1, a, c]);
    }
    let var = render_ash::impostor::bake_variant(&iv, &ii);
    let cone_sphere = [0.0, 0.5, 0.0, 0.6];
    let mut meshes = vec![MeshInfoGpu { index_count: ii.len() as u32, first_index: 0, vertex_offset: 0, _pad: 0, sphere: cone_sphere }];
    let base = iv.len() as i32;
    let first = ii.len() as u32;
    for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        iv.push(QVertex { pos: [x, y, 0.0], nrm: [0.0, 0.0, 1.0], attr: [0.0, 0.0, 0.0, 2.0] });
    }
    ii.extend([0, 1, 2, 0, 2, 3]);
    meshes.push(MeshInfoGpu { index_count: 6, first_index: first, vertex_offset: base, _pad: 0, sphere: [var.centre.x, var.centre.y, var.centre.z, var.radius] });
    let atlas = render_ash::impostor::atlas_bytes(std::slice::from_ref(&var));
    // cone -> impostor at 30 m with a 12 m crossfade; exact-depth impostors to 60 m
    let lod = vec![[1, 30f32.to_bits(), 12f32.to_bits(), 0], [LOD_NONE, 0, 0, 60f32.to_bits()]];
    // a dense, overlapping stand from 12 m to ~90 m
    let inst: Vec<InstanceGpu> = (0..400)
        .map(|k| {
            let (x, z) = (((k * 37) % 41) as f32 * 1.7 - 34.0 + 0.3 * (k % 7) as f32, -12.0 - (k / 5) as f32 * 1.0);
            let s = 6.0 + (k % 5) as f32;
            let m = Mat4::from_translation(Vec3::new(x, -2.0, z)) * Mat4::from_rotation_y(k as f32 * 0.7) * Mat4::from_scale(Vec3::splat(s));
            InstanceGpu { model: m.to_cols_array_2d(), mesh_id: 0, material_id: 0, _pad: [0; 2] }
        })
        .collect();
    let fov = 1.0f32;
    let aspect = w as f32 / h as f32;
    let fwd = Vec3::new(0.0, -0.05, -1.0).normalize();
    let view = render_ash::cull::look_at(Vec3::ZERO, fwd, Vec3::Y);
    let vp = reverse_z_infinite(fov, aspect, 0.1) * view;
    let cull_vp = glam::camera::rh::proj::vulkan::perspective(fov, aspect, 0.1, 500.0) * view;
    let sun = Vec3::new(0.5, 0.6, 0.3).normalize();
    let origin = glam::DVec3::new(100.0, 50.0, -40.0);
    let (res, splits) = (512u32, [20.0f32, 60.0, 150.0, 400.0]);
    let mut shadows = Vec::new();
    let mut nd = 0.1;
    for f in splits {
        let (m, texel, _) = sun_cascade(origin, fwd, fov, aspect, nd, f, sun, res, 200.0);
        shadows.push(ShadowView { view_proj: m, texel });
        nd = f * 0.8;
    }
    let mut p = QParamsGpu { view_proj: vp.to_cols_array_2d(), inv_dir: cull_vp.inverse().to_cols_array_2d(), ..Default::default() };
    p.cam_world = [origin.x as f32, origin.y as f32, origin.z as f32, fov / h as f32];
    p.sun_dir = [sun.x, sun.y, sun.z, 5.0];
    p.sun_color = [1.0, 0.9, 0.8, 0.6];
    p.sky_zenith = [0.2, 0.35, 0.7, 0.0];
    p.sky_horizon = [0.6, 0.7, 0.8, 0.0];
    p.fog_color = [0.5, 0.6, 0.7, 0.5];
    p.shadow_info[2] = 1.0;
    p.splits = splits;
    let render = |b: &mut AshBackend, instances: &[InstanceGpu], prepass: bool, card: bool| {
        let frame = QualityFrame {
            width: w,
            height: h,
            vertices: &verts,
            indices: &idx,
            params: p,
            shadow_res: res,
            shadows: &shadows,
            spot_shadow: false,
            instanced: Some(InstancedSet { vertices: &iv, indices: &ii, meshes: &meshes, instances, cull_view_proj: cull_vp, lod: &lod, impostor_first: 1, atlas: Some((&atlas, 1)) }),
            atmosphere: false,
            terrain_chunks: &[],
            shadow_slices: &[],
            depth_prepass: prepass,
            impostor_card_depth: card,
            shadow_cache_keys: &[],
        };
        b.render_quality(&frame).expect("quality frame")
    };
    let far: Vec<InstanceGpu> = inst.iter().map(|i| InstanceGpu { model: Mat4::from_translation(Vec3::new(0.0, 0.0, 5000.0)).mul_mat4(&Mat4::from_cols_array_2d(&i.model)).to_cols_array_2d(), ..*i }).collect();
    let (bare, _) = render(&mut b, &far[..1], true, true);
    let (card_pre, t) = render(&mut b, &inst, true, true);
    assert!(t.impostor_instances > 100 && t.fading_instances > 20, "the stand exercises whole, crossfading and impostor draws: {t:?}");
    let (card_nopre, _) = render(&mut b, &inst, false, true);
    let (exact_pre, _) = render(&mut b, &inst, true, false);
    let differ = |a: &[u8], c: &[u8]| a.chunks(4).zip(c.chunks(4)).filter(|(x, y)| x[..3] != y[..3]).count();
    let px = (w * h) as usize;
    // the prepass only changes depth-tie winners (none expected between distinct cards)
    let d = differ(&card_pre, &card_nopre);
    assert!(d * 1000 <= px, "card-depth impostor prepass changes {d} of {px} pixels");
    // coverage (pixels where the trees replace the bare ground) matches the exact-depth path
    let (cov_card, cov_exact) = (differ(&card_pre, &bare), differ(&exact_pre, &bare));
    eprintln!("card-depth impostors: prepass diff {d} px, coverage card {cov_card} / exact {cov_exact} of {px} px, {t:?}");
    assert!(cov_exact * 10 > px, "the stand covers > 10 % of the frame ({cov_exact} px)");
    assert!(cov_card.abs_diff(cov_exact) * 50 <= cov_exact, "coverage card {cov_card} vs exact {cov_exact} px");
    drop(b);
    assert_no_validation_errors();
}
