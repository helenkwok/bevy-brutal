//! Floating origin over the f64 frame hierarchy (`coords::frames`).
//!
//! * `WorldPosition` (f64, frame-tagged) is the truth for anything that can be far from the origin.
//! * `Transform::translation` of those entities is *derived*: f32 metres relative to the current
//!   floating origin, recomputed from f64 when the entity moves or the origin rebases. So the
//!   f32 values never drift.
//! * The origin rebases to the camera when the camera is more than `threshold_m` from it, or
//!   when the camera changes frame (after `FrameTree::rehome`).
//! * `gather_instances` sends render-ash **camera-relative** translations, subtracted in f64 for
//!   entities with a `WorldPosition`. The game's `ViewProj` must therefore treat the camera as
//!   sitting at the origin (rotation + projection only).
//!
//! Entities with a `WorldPosition` must be hierarchy roots: their `Transform` is overwritten.
use bevy::prelude::*;
use bevy::transform::TransformSystems;
use coords::{DVec3, FloatingOrigin, FrameTree, WorldPos};

#[derive(Component, Clone, Copy, Debug)]
pub struct WorldPosition(pub WorldPos);

/// The (single) camera entity that drives rebasing.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct FloatingCamera;

/// The frame hierarchy.
#[derive(Resource, Clone, Debug)]
pub struct Universe(pub FrameTree);

#[derive(Resource, Clone, Debug, Default)]
pub struct FloatingState {
    pub origin: Option<FloatingOrigin>,
    /// Camera position this frame (after rehome).
    pub camera: Option<WorldPos>,
    /// Shift applied this frame, if it rebased.
    pub last_shift: Option<DVec3>,
}

pub struct FloatingOriginPlugin {
    pub threshold_m: f64,
}

impl Default for FloatingOriginPlugin {
    fn default() -> Self {
        Self { threshold_m: 2048.0 }
    }
}

#[derive(Resource, Clone, Copy)]
struct Threshold(f64);

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct FloatingOriginSystems;

impl Plugin for FloatingOriginPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FloatingState>()
            .insert_resource(Threshold(self.threshold_m))
            .add_systems(
                PostUpdate,
                (rebase_origin, sync_transforms).chain().in_set(FloatingOriginSystems).before(TransformSystems::Propagate),
            );
    }
}

fn rebase_origin(uni: Res<Universe>, th: Res<Threshold>, mut st: ResMut<FloatingState>, mut cam: Query<&mut WorldPosition, With<FloatingCamera>>) {
    st.last_shift = None;
    let Ok(mut cam) = cam.single_mut() else {
        st.camera = None;
        return;
    };
    let homed = uni.0.rehome(cam.0);
    if homed.frame != cam.0.frame {
        cam.0 = homed;
    }
    st.camera = Some(cam.0);
    match st.origin.as_mut() {
        None => {
            st.origin = Some(FloatingOrigin::new(cam.0, th.0));
            st.last_shift = Some(DVec3::ZERO);
        }
        Some(fo) => st.last_shift = fo.update(&uni.0, cam.0),
    }
}

fn sync_transforms(uni: Res<Universe>, st: Res<FloatingState>, mut q: Query<(Ref<WorldPosition>, &mut Transform)>) {
    let Some(fo) = st.origin else { return };
    let all = st.last_shift.is_some();
    for (wp, mut t) in &mut q {
        if all || wp.is_changed() {
            t.translation = Vec3::from_array(fo.local_f32(&uni.0, wp.0).to_array());
        }
    }
}

/// Camera-relative translation for an instance: f64 subtraction when it has a `WorldPosition`,
/// else (origin-relative f32) minus (camera origin-relative f32).
pub fn camera_relative(uni: Option<&Universe>, st: Option<&FloatingState>, wp: Option<&WorldPosition>, global: Vec3) -> Vec3 {
    let (Some(uni), Some(st)) = (uni, st) else { return global };
    let (Some(cam), Some(fo)) = (st.camera, st.origin) else { return global };
    match wp {
        Some(wp) => Vec3::from_array(uni.0.render_relative(wp.0, cam).to_array()),
        None => global - Vec3::from_array(fo.local_f32(&uni.0, cam).to_array()),
    }
}
