//! Bevy 0.20 integration. Safe code only: copies ECS data into plain-data SoA buffers, runs the
//! low-level hot systems, writes results back, and hands a `FrameSnapshot` to any `RenderBackend`.
#![forbid(unsafe_code)]

pub mod floating;

use std::marker::PhantomData;

use bevy::prelude::*;
use bevy::transform::TransformSystems;
use hot::particles::ParticleIntegrator;
use lowlevel_abi::{FrameSnapshot, HotSystem, InstanceGpu, ParticleBuffers, ParticleParams, RenderBackend};

// ---------- components & resources --------------------------------------------------------

/// Simulated particle. Position lives in `Transform::translation`.
#[derive(Component, Clone, Copy, Debug)]
pub struct Particle {
    pub velocity: Vec3,
    pub life: f32,
}

/// Marks an entity for the low-level renderer.
#[derive(Component, Clone, Copy, Debug)]
pub struct MeshRef {
    pub mesh_id: u32,
    pub material_id: u32,
}

#[derive(Resource, Clone, Copy, Debug)]
pub struct ParticleSettings {
    pub gravity: Vec3,
    pub drag: f32,
    /// Despawn particles whose life reaches zero.
    pub despawn_dead: bool,
}
impl Default for ParticleSettings {
    fn default() -> Self {
        Self { gravity: Vec3::new(0.0, -9.81, 0.0), drag: 0.1, despawn_dead: true }
    }
}

/// SoA scratch buffers + the entity for each row. Reused every frame (no shrink).
#[derive(Resource, Default)]
pub struct ParticleSoA {
    pub buffers: ParticleBuffers,
    pub entities: Vec<Entity>,
    pub live_after_step: usize,
}

/// Camera matrix for the low-level renderer (set by the game; no bevy_camera dependency here).
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct ViewProj(pub Mat4);

/// Instances gathered this frame, in `InstanceGpu` layout.
#[derive(Resource, Default)]
pub struct InstanceScratch(pub Vec<InstanceGpu>);

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum BrutalSystems {
    ExtractParticles,
    Simulate,
    WriteBack,
}

// ---------- particle plugin -----------------------------------------------------------------

pub struct BrutalParticlesPlugin;

impl Plugin for BrutalParticlesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ParticleSettings>()
            .init_resource::<ParticleSoA>()
            .configure_sets(
                FixedUpdate,
                (BrutalSystems::ExtractParticles, BrutalSystems::Simulate, BrutalSystems::WriteBack).chain(),
            )
            .add_systems(
                FixedUpdate,
                (
                    extract_particles.in_set(BrutalSystems::ExtractParticles),
                    simulate_particles.in_set(BrutalSystems::Simulate),
                    write_back_particles.in_set(BrutalSystems::WriteBack),
                ),
            );
    }
}

/// ECS -> SoA. Padding rows stay at life = 0 so the integrator ignores them.
pub fn extract_particles(q: Query<(Entity, &Transform, &Particle)>, mut soa: ResMut<ParticleSoA>) {
    #[cfg(feature = "tracy")]
    let _z = tracy_client::span!("bridge::extract_particles");
    let n = q.iter().len();
    let ParticleSoA { buffers: b, entities, .. } = &mut *soa;
    b.reset(n);
    entities.clear();
    for (i, (e, t, p)) in q.iter().enumerate() {
        let pos = t.translation;
        b.px[i] = pos.x;
        b.py[i] = pos.y;
        b.pz[i] = pos.z;
        b.vx[i] = p.velocity.x;
        b.vy[i] = p.velocity.y;
        b.vz[i] = p.velocity.z;
        b.life[i] = p.life;
        entities.push(e);
    }
}

/// Run the SIMD integrator (hot crate) over the SoA buffers.
pub fn simulate_particles(
    time: Res<Time<Fixed>>,
    settings: Res<ParticleSettings>,
    mut soa: ResMut<ParticleSoA>,
    mut integrator: Local<ParticleIntegrator>,
) {
    let prm = ParticleParams { dt: time.delta_secs(), gravity: settings.gravity.to_array(), drag: settings.drag };
    let ParticleSoA { buffers, live_after_step, .. } = &mut *soa;
    integrator.run(prm, (buffers.view_mut(), live_after_step));
}

/// SoA -> ECS. Optionally despawns dead particles.
pub fn write_back_particles(
    mut commands: Commands,
    settings: Res<ParticleSettings>,
    soa: Res<ParticleSoA>,
    mut q: Query<(&mut Transform, &mut Particle)>,
) {
    let b = &soa.buffers;
    for (i, &e) in soa.entities.iter().enumerate() {
        let Ok((mut t, mut p)) = q.get_mut(e) else { continue };
        t.translation = Vec3::new(b.px[i], b.py[i], b.pz[i]);
        p.velocity = Vec3::new(b.vx[i], b.vy[i], b.vz[i]);
        p.life = b.life[i];
        if settings.despawn_dead && p.life <= 0.0 {
            commands.entity(e).despawn();
        }
    }
}

// ---------- render plugin -------------------------------------------------------------------

/// Gathers `(GlobalTransform, MeshRef)` into `InstanceScratch` after transform propagation and
/// submits a snapshot to backend `B`, stored as a NonSend resource (it owns Vulkan handles).
/// If no `B` resource is present, instances are still gathered (useful for tests/headless).
pub struct BrutalRenderPlugin<B>(PhantomData<fn() -> B>);

impl<B> Default for BrutalRenderPlugin<B> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<B: RenderBackend + 'static> Plugin for BrutalRenderPlugin<B> {
    fn build(&self, app: &mut App) {
        app.init_resource::<InstanceScratch>()
            .init_resource::<ViewProj>()
            .add_systems(
                PostUpdate,
                (gather_instances, submit_frame::<B>).chain().after(TransformSystems::Propagate),
            );
    }
}

pub fn gather_instances(
    q: Query<(&GlobalTransform, &MeshRef, Option<&floating::WorldPosition>)>,
    uni: Option<Res<floating::Universe>>,
    st: Option<Res<floating::FloatingState>>,
    mut scratch: ResMut<InstanceScratch>,
) {
    #[cfg(feature = "tracy")]
    let _z = tracy_client::span!("bridge::gather_instances");
    scratch.0.clear();
    let (uni, st) = (uni.as_deref(), st.as_deref());
    scratch.0.extend(q.iter().map(|(t, m, wp)| {
        let mut model = t.to_matrix();
        // Camera-relative (f64 subtraction for WorldPosition entities); identity when no floating origin.
        model.w_axis = floating::camera_relative(uni, st, wp, t.translation()).extend(1.0);
        InstanceGpu { model: model.to_cols_array_2d(), mesh_id: m.mesh_id, material_id: m.material_id, _pad: [0; 2] }
    }));
}

pub fn submit_frame<B: RenderBackend + 'static>(
    scratch: Res<InstanceScratch>,
    view_proj: Res<ViewProj>,
    backend: Option<NonSendMut<B>>,
    mut frame: Local<u64>,
) {
    let Some(mut backend) = backend else { return };
    *frame += 1;
    let snap = FrameSnapshot { view_proj: view_proj.0.to_cols_array_2d(), instances: &scratch.0, frame_index: *frame };
    if let Err(e) = backend.submit(&snap) {
        bevy::log::error!("render backend submit failed: {e:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct CountingBackend {
        frames: u64,
        last_instances: usize,
    }
    impl RenderBackend for CountingBackend {
        type Error = ();
        fn resize(&mut self, _: u32, _: u32) -> Result<(), ()> {
            Ok(())
        }
        fn submit(&mut self, f: &FrameSnapshot<'_>) -> Result<(), ()> {
            self.frames += 1;
            self.last_instances = f.instances.len();
            Ok(())
        }
    }

    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, TransformPlugin, BrutalParticlesPlugin))
            .add_plugins(BrutalRenderPlugin::<CountingBackend>::default())
            .insert_non_send(CountingBackend { frames: 0, last_instances: 0 });
        // Deterministic fixed step: advance virtual time by exactly one fixed tick per update.
        app.insert_resource(Time::<Fixed>::from_hz(60.0));
        app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(1.0 / 60.0)));
        app
    }

    #[test]
    fn particles_fall_and_die_and_frames_submit() {
        let mut app = test_app();
        let e = app
            .world_mut()
            .spawn((Transform::from_xyz(0.0, 10.0, 0.0), Particle { velocity: Vec3::ZERO, life: 0.5 }, MeshRef { mesh_id: 0, material_id: 0 }))
            .id();
        for _ in 0..10 {
            app.world_mut().spawn((Transform::default(), Particle { velocity: Vec3::X, life: 100.0 }));
        }
        for _ in 0..10 {
            app.update();
        }
        let y = app.world().get::<Transform>(e).expect("still alive").translation.y;
        assert!(y < 10.0, "particle should fall, y = {y}");
        for _ in 0..40 {
            app.update();
        }
        assert!(app.world().get_entity(e).is_err(), "particle with life 0.5s should be despawned");
        let be = app.world().non_send::<CountingBackend>();
        assert_eq!(be.frames, 50);
        assert_eq!(be.last_instances, 0);
        assert_eq!(app.world().resource::<ParticleSoA>().live_after_step, 10);
    }

    #[derive(Default)]
    struct CaptureBackend {
        last: Vec<InstanceGpu>,
    }
    impl RenderBackend for CaptureBackend {
        type Error = ();
        fn resize(&mut self, _: u32, _: u32) -> Result<(), ()> {
            Ok(())
        }
        fn submit(&mut self, f: &FrameSnapshot<'_>) -> Result<(), ()> {
            self.last = f.instances.to_vec();
            Ok(())
        }
    }

    #[test]
    fn floating_origin_rebases_and_instances_are_camera_relative() {
        use coords::{DVec3, FrameTree, WorldPos};
        use floating::*;
        let (mut tree, g) = FrameTree::new("g");
        let s = tree.add(g, "sys", DVec3::new(3e20, 0.0, 0.0), 1e13).unwrap();
        let p = tree.add(s, "planet", DVec3::new(1.5e11, 0.0, 0.0), 1e7).unwrap();
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, TransformPlugin, FloatingOriginPlugin { threshold_m: 1000.0 }))
            .add_plugins(BrutalRenderPlugin::<CaptureBackend>::default())
            .insert_non_send(CaptureBackend::default())
            .insert_resource(Universe(tree));
        let start = DVec3::new(-9.0e6, 3.0, 0.0);
        let cam = app.world_mut().spawn((Transform::default(), FloatingCamera, WorldPosition(WorldPos { frame: p, p: start }))).id();
        // object 5.3 m ahead of the camera's start, plus one that rides along 10.7 m ahead
        let fixed = app.world_mut().spawn((Transform::default(), MeshRef { mesh_id: 0, material_id: 0 }, WorldPosition(WorldPos { frame: p, p: start + DVec3::new(5.3, 0.0, 0.0) }))).id();
        let rider = app.world_mut().spawn((Transform::default(), MeshRef { mesh_id: 1, material_id: 0 }, WorldPosition(WorldPos { frame: p, p: start + DVec3::new(10.7, 0.0, 0.0) }))).id();
        let step = DVec3::new(1777.25, 0.5, 0.0);
        let mut cam_p = start;
        let mut worst = 0.0f32;
        for i in 0..2000 {
            cam_p += step;
            app.world_mut().get_mut::<WorldPosition>(cam).unwrap().0.p = cam_p;
            app.world_mut().get_mut::<WorldPosition>(rider).unwrap().0.p = cam_p + DVec3::new(10.7, 0.0, 0.0);
            app.update();
            let last = &app.world().non_send::<CaptureBackend>().last;
            let r = last.iter().find(|x| x.mesh_id == 1).unwrap().model[3];
            worst = worst.max((r[0] - 10.7).abs().max(r[1].abs()));
            let t = app.world().get::<Transform>(rider).unwrap().translation;
            assert!(t.length() < 1000.0 + 20.0, "origin-relative f32 stays small at step {i}: {t}");
            let f = last.iter().find(|x| x.mesh_id == 0).unwrap().model[3];
            let exact = (start + DVec3::new(5.3, 0.0, 0.0)) - cam_p;
            let tol = 1e-6 * exact.length() as f32 + 1e-3; // f32 relative rounding of a large camera-relative vector
            assert!((f[0] - exact.x as f32).abs() <= tol, "fixed object camera-relative x {} vs {}", f[0], exact.x);
        }
        let st = app.world().resource::<FloatingState>();
        assert!(st.origin.unwrap().rebases >= 1000, "rebases {}", st.origin.unwrap().rebases);
        assert!(worst < 1e-4, "rider camera-relative error {worst} m");
        let _ = fixed;
    }
}
