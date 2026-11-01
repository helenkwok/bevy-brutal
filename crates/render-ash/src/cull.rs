//! CPU reference frustum culling. Same math as `shaders/cull.slang`; used as the test oracle.
use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};
use lowlevel_abi::InstanceGpu;

use crate::mesh::MeshInfoGpu;

/// Normalized frustum planes (Gribb-Hartmann) for Vulkan clip space (z in [0, 1]).
/// Order: left, right, bottom, top, near, far. Inside when dot(n, p) + d >= 0.
pub fn frustum_planes(view_proj: Mat4) -> [Vec4; 6] {
    let (r0, r1, r2, r3) = (view_proj.row(0), view_proj.row(1), view_proj.row(2), view_proj.row(3));
    [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2].map(|p| p / p.xyz().length())
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Visibility {
    Inside,
    Outside,
    /// Within float tolerance of a plane: the GPU may decide either way.
    Ambiguous,
}

/// World-space bounding sphere of an instance (center, radius) - identical to the shader.
pub fn world_sphere(inst: &InstanceGpu, mesh: &MeshInfoGpu) -> (Vec3, f32) {
    let m = Mat4::from_cols_array_2d(&inst.model);
    let c = (m * Vec4::from((Vec3::from_slice(&mesh.sphere[..3]), 1.0))).xyz();
    let s = m.x_axis.xyz().length().max(m.y_axis.xyz().length()).max(m.z_axis.xyz().length());
    (c, mesh.sphere[3] * s)
}

pub fn classify(planes: &[Vec4; 6], center: Vec3, radius: f32) -> Visibility {
    let eps = 1e-3 * radius.max(1.0) * (1.0 + center.length() * 1e-3);
    let mut ambiguous = false;
    for p in planes {
        let d = p.xyz().dot(center) + p.w + radius;
        if d < -eps {
            return Visibility::Outside;
        }
        if d <= eps {
            ambiguous = true;
        }
    }
    if ambiguous { Visibility::Ambiguous } else { Visibility::Inside }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CullReference {
    pub inside: usize,
    pub ambiguous: usize,
}

impl CullReference {
    /// True if a GPU visible count is consistent with this reference.
    pub fn accepts(&self, gpu_visible: usize) -> bool {
        gpu_visible >= self.inside && gpu_visible <= self.inside + self.ambiguous
    }
}

/// Count visible instances on the CPU. Instances with an out-of-range mesh id are skipped (as on GPU).
pub fn cpu_cull(instances: &[InstanceGpu], meshes: &[MeshInfoGpu], view_proj: Mat4) -> CullReference {
    let planes = frustum_planes(view_proj);
    let mut r = CullReference::default();
    for inst in instances {
        let Some(mesh) = meshes.get(inst.mesh_id as usize) else { continue };
        let (c, rad) = world_sphere(inst, mesh);
        match classify(&planes, c, rad) {
            Visibility::Inside => r.inside += 1,
            Visibility::Ambiguous => r.ambiguous += 1,
            Visibility::Outside => {}
        }
    }
    r
}

/// Vulkan-convention perspective (RH view space, NDC z in [0, 1], NDC y down).
/// glam 0.34 moved camera helpers to `glam::camera::{rh,lh}::{proj,view}`.
pub fn vk_perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    glam::camera::rh::proj::vulkan::perspective(fov_y, aspect, near, far)
}

/// Right-handed look-at view matrix.
pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    glam::camera::rh::view::look_at_mat4(eye, target, up)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planes_classify_points() {
        let vp = vk_perspective(1.0, 1.0, 0.1, 100.0) * look_at(Vec3::ZERO, -Vec3::Z, Vec3::Y);
        let pl = frustum_planes(vp);
        assert_eq!(classify(&pl, Vec3::new(0.0, 0.0, -10.0), 0.5), Visibility::Inside);
        assert_eq!(classify(&pl, Vec3::new(0.0, 0.0, 10.0), 0.5), Visibility::Outside);
        assert_eq!(classify(&pl, Vec3::new(0.0, 0.0, -200.0), 0.5), Visibility::Outside);
        assert_eq!(classify(&pl, Vec3::new(100.0, 0.0, -10.0), 0.5), Visibility::Outside);
    }
}
