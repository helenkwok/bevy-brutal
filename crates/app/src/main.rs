//! Game entry point. Headless by default (runs N frames and exits); `--features windowed` opens a
//! winit window without bevy_render (Mode B: the ash backend owns presentation; swapchain TODO).
//!
//! Env: PARTICLES (default 100000), FRAMES (default 300).
#![forbid(unsafe_code)]

use std::time::Instant;

use bevy::prelude::*;
use bridge::{BrutalParticlesPlugin, BrutalRenderPlugin, MeshRef, Particle, ParticleSoA};

#[cfg(feature = "vulkan")]
type Backend = render_ash::AshBackend;
#[cfg(not(feature = "vulkan"))]
type Backend = NullBackend;

/// Backend that drops frames, for builds without `vulkan`.
#[cfg(not(feature = "vulkan"))]
pub struct NullBackend;
#[cfg(not(feature = "vulkan"))]
impl lowlevel_abi::RenderBackend for NullBackend {
    type Error = ();
    fn resize(&mut self, _: u32, _: u32) -> Result<(), ()> {
        Ok(())
    }
    fn submit(&mut self, _: &lowlevel_abi::FrameSnapshot<'_>) -> Result<(), ()> {
        Ok(())
    }
}

#[derive(Resource)]
struct RunConfig {
    particles: usize,
    frames: u32,
    started: Instant,
}

fn env_or<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() -> AppExit {
    let cfg = RunConfig { particles: env_or("PARTICLES", 100_000), frames: env_or("FRAMES", 300), started: Instant::now() };
    let mut app = App::new();

    #[cfg(feature = "windowed")]
    app.add_plugins(DefaultPlugins);
    #[cfg(not(feature = "windowed"))]
    app.add_plugins((
        MinimalPlugins.set(bevy::app::ScheduleRunnerPlugin::run_loop(std::time::Duration::ZERO)),
        TransformPlugin,
        bevy::log::LogPlugin::default(),
    ));

    app.insert_resource(Time::<Fixed>::from_hz(60.0))
        .insert_resource(cfg)
        .add_plugins((BrutalParticlesPlugin, BrutalRenderPlugin::<Backend>::default()))
        .add_systems(Startup, (init_backend, spawn_particles, build_terrain, setup_camera))
        .add_systems(Update, frame_counter);
    #[cfg(feature = "windowed")]
    app.add_systems(PreUpdate, init_windowed_backend).add_systems(Update, (resize_backend, scripted_resize));

    app.run()
}

/// Camera for the low-level renderer (Vulkan clip space: y down, depth 0..1).
fn setup_camera(mut vp: ResMut<bridge::ViewProj>) {
    // bevy_math 0.20 re-exports glam's DirectX-style RH projection (depth 0..1, +Y up) as
    // `bevy::math::proj`; Mat4::perspective_rh/look_at_rh are deprecated in glam 0.33.
    let mut proj = bevy::math::proj::perspective(60f32.to_radians(), 16.0 / 9.0, 0.1, 500.0);
    proj.y_axis.y = -proj.y_axis.y; // +Y up -> Vulkan +Y down
    let view = Transform::from_xyz(0.0, 45.0, 90.0).looking_at(Vec3::new(0.0, 25.0, 0.0), Vec3::Y).to_matrix().inverse();
    vp.0 = proj * view;
}

/// Windowed: create the ash backend once winit has created the primary window and its raw handles.
#[cfg(feature = "windowed")]
fn init_windowed_backend(world: &mut World) {
    use bevy::window::{PrimaryWindow, RawHandleWrapper};
    if world.contains_non_send::<Backend>() {
        return;
    }
    let mut q = world.query_filtered::<(&RawHandleWrapper, &Window), With<PrimaryWindow>>();
    let Ok((h, win)) = q.single(world) else { return };
    let handles = render_ash::WindowHandles { display: h.get_display_handle(), window: h.get_window_handle() };
    let cfg = render_ash::BackendConfig {
        max_instances: 1 << 20,
        extent: (win.resolution.physical_width(), win.resolution.physical_height()),
        ..Default::default()
    };
    match render_ash::AshBackend::new_windowed(handles, cfg) {
        Ok(b) => {
            info!("ash windowed backend up on '{}' ({}x{}, validation: {})", b.device_name(), b.extent.width, b.extent.height, b.ctx.validation_enabled);
            world.insert_non_send(b);
        }
        Err(e) => {
            error!("failed to create windowed Vulkan backend: {e:?}");
            world.write_message(AppExit::error());
        }
    }
}

/// `BRUTAL_RESIZE_AT=N`: resize the primary window to 960x540 at frame N (exercises swapchain recreation).
#[cfg(feature = "windowed")]
fn scripted_resize(mut n: Local<u32>, mut windows: Query<&mut Window, With<bevy::window::PrimaryWindow>>) {
    *n += 1;
    if std::env::var("BRUTAL_RESIZE_AT").ok().and_then(|v| v.parse::<u32>().ok()) == Some(*n)
        && let Ok(mut w) = windows.single_mut()
    {
        info!("scripted resize -> 960x540 at frame {}", *n);
        w.resolution.set(960.0, 540.0);
    }
}

#[cfg(feature = "windowed")]
fn resize_backend(
    mut resized: MessageReader<bevy::window::WindowResized>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    backend: Option<NonSendMut<Backend>>,
) {
    use lowlevel_abi::RenderBackend;
    if resized.read().last().is_none() {
        return;
    }
    let (Some(mut b), Ok(win)) = (backend, windows.single()) else { return };
    let (w, h) = (win.resolution.physical_width(), win.resolution.physical_height());
    match b.resize(w, h) {
        Ok(()) => info!("backend resized to {}x{}", b.extent.width, b.extent.height),
        Err(e) => error!("resize to {w}x{h} failed: {e:?}"),
    }
}

/// Windowed builds create the backend in `init_windowed_backend` once the window exists.
#[cfg(feature = "windowed")]
fn init_backend(_world: &mut World) {}

#[cfg(not(feature = "windowed"))]
fn init_backend(world: &mut World) {
    #[cfg(feature = "vulkan")]
    match render_ash::AshBackend::new_headless(1 << 20) {
        Ok(b) => {
            info!("ash backend up on '{}'", b.ctx.device_name);
            world.insert_non_send(b);
        }
        Err(e) => warn!("no Vulkan backend ({e:?}); running without renderer"),
    }
    #[cfg(not(feature = "vulkan"))]
    world.insert_non_send(NullBackend);
}

fn spawn_particles(mut commands: Commands, cfg: Res<RunConfig>) {
    let n = cfg.particles;
    commands.spawn_batch((0..n).map(|i| {
        let f = i as f32;
        (
            Transform::from_xyz((f * 0.37).sin() * 20.0, 50.0 + (f * 0.11).cos() * 10.0, (f * 0.53).cos() * 20.0),
            Particle { velocity: Vec3::new((f * 1.3).sin(), 5.0, (f * 0.7).cos()), life: 2.0 + (i % 300) as f32 / 60.0 },
            MeshRef { mesh_id: (i % 4) as u32, material_id: 0 },
        )
    }));
    info!("spawned {n} particles");
}

fn build_terrain() {
    use terrain::*;
    let p = NoiseParams::default();
    let arena = mem::FrameArena::with_capacity(16 * CHUNK_SAMPLES * CHUNK_SAMPLES * size_of::<TerrainVertex>() + 1024);
    let mut hm = vec![0.0; PADDED * PADDED];
    let t = Instant::now();
    let mut verts = 0;
    for z in -2..2 {
        for x in -2..2 {
            let req = ChunkRequest { coord: [x, z], lod: 0, seed: 42 };
            heightmap_simd(&req, &p, &mut hm);
            verts += mesh_chunk(&req, &hm, &arena).expect("arena sized for 16 chunks").vertices.len();
        }
    }
    info!("terrain: 16 chunks, {verts} verts in {:.2?} (GPU upload TODO)", t.elapsed());
}

fn frame_counter(
    mut n: Local<u32>,
    cfg: Res<RunConfig>,
    soa: Res<ParticleSoA>,
    instances: Res<bridge::InstanceScratch>,
    mut exit: MessageWriter<AppExit>,
) {
    *n += 1;
    if *n == cfg.frames {
        let secs = cfg.started.elapsed().as_secs_f64();
        info!(
            "{} frames in {secs:.2}s ({:.0} fps); live particles {}, instances gathered {}",
            *n,
            *n as f64 / secs,
            soa.live_after_step,
            instances.0.len()
        );
        exit.write(AppExit::Success);
    }
}
