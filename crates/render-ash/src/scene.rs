//! Deterministic random scenes for tests, benches and the stress example.
use glam::{Mat4, Quat, Vec3};
use lowlevel_abi::InstanceGpu;

use crate::cull::{look_at, vk_perspective};

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.next_f32()
    }
}

/// Half-extent of the cube the scene's instances occupy (constant density ~ 1 per 8 m^3).
pub fn scene_radius(n: usize) -> f32 {
    (n.max(1) as f32).cbrt()
}

/// `n` instances with random position, rotation and (slightly non-uniform) scale.
/// `bad_mesh_fraction` of them get an out-of-range mesh id (must be skipped by culling).
pub fn random_scene(n: usize, seed: u64, mesh_count: u32, bad_mesh_fraction: f32) -> Vec<InstanceGpu> {
    let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let ext = scene_radius(n);
    (0..n)
        .map(|i| {
            let pos = Vec3::new(r.range(-ext, ext), r.range(-ext, ext), r.range(-ext, ext));
            let axis = Vec3::new(r.range(-1., 1.), r.range(-1., 1.), r.range(-1., 1.)).normalize_or(Vec3::Y);
            let rot = Quat::from_axis_angle(axis, r.range(0.0, std::f32::consts::TAU));
            let s = r.range(0.4, 1.2);
            let scale = Vec3::new(s, s * r.range(0.7, 1.3), s);
            let mesh_id =
                if r.next_f32() < bad_mesh_fraction { mesh_count + 7 } else { (i as u32) % mesh_count.max(1) };
            InstanceGpu {
                model: Mat4::from_scale_rotation_translation(scale, rot, pos).to_cols_array_2d(),
                mesh_id,
                material_id: (i as u32 / 4) % 5,
                _pad: [0; 2],
            }
        })
        .collect()
}

/// Camera just inside the scene cube looking across it; a minority of instances are visible,
/// so culling has real work to do and many instances straddle the frustum planes.
pub fn demo_view_proj(n: usize, aspect: f32) -> Mat4 {
    let ext = scene_radius(n);
    let eye = Vec3::new(ext * 0.2, ext * 0.3, ext * 0.85 + 2.0);
    vk_perspective(60f32.to_radians(), aspect, 0.1, ext * 6.0 + 10.0) * look_at(eye, Vec3::ZERO, Vec3::Y)
}
