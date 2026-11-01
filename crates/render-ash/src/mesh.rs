//! Mesh data: vertex format, GPU mesh table entry, and built-in procedural meshes.
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
pub struct MeshVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
}

/// Mesh table entry, mirrors `MeshInfo` in shaders/common.slang (32 bytes, scalar layout).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
pub struct MeshInfoGpu {
    pub index_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub _pad: u32,
    /// xyz = local bounding-sphere center, w = radius.
    pub sphere: [f32; 4],
}

#[derive(Clone, Debug, Default)]
pub struct MeshData {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
}

impl MeshData {
    /// Flat-shaded mesh from a triangle list (CCW).
    pub fn from_triangles(tris: &[[Vec3; 3]]) -> Self {
        let mut m = MeshData::default();
        for t in tris {
            let n = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
            for p in t {
                m.indices.push(m.vertices.len() as u32);
                m.vertices.push(MeshVertex { pos: p.to_array(), normal: n.to_array() });
            }
        }
        m
    }

    /// Bounding sphere centered at the vertex centroid.
    pub fn bounding_sphere(&self) -> [f32; 4] {
        let n = self.vertices.len().max(1) as f32;
        let c = self.vertices.iter().fold(Vec3::ZERO, |a, v| a + Vec3::from(v.pos)) / n;
        let r = self.vertices.iter().map(|v| (Vec3::from(v.pos) - c).length()).fold(0.0, f32::max);
        [c.x, c.y, c.z, r]
    }
}

fn quad(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> [[Vec3; 3]; 2] {
    [[a, b, c], [a, c, d]]
}

/// Unit cube centered at the origin (side 1).
pub fn cube() -> MeshData {
    let h = 0.5;
    let v = |x: f32, y: f32, z: f32| Vec3::new(x * h, y * h, z * h);
    let faces = [
        quad(v(-1., -1., 1.), v(1., -1., 1.), v(1., 1., 1.), v(-1., 1., 1.)),
        quad(v(1., -1., -1.), v(-1., -1., -1.), v(-1., 1., -1.), v(1., 1., -1.)),
        quad(v(1., -1., 1.), v(1., -1., -1.), v(1., 1., -1.), v(1., 1., 1.)),
        quad(v(-1., -1., -1.), v(-1., -1., 1.), v(-1., 1., 1.), v(-1., 1., -1.)),
        quad(v(-1., 1., 1.), v(1., 1., 1.), v(1., 1., -1.), v(-1., 1., -1.)),
        quad(v(-1., -1., -1.), v(1., -1., -1.), v(1., -1., 1.), v(-1., -1., 1.)),
    ];
    MeshData::from_triangles(&faces.concat())
}

pub fn octahedron() -> MeshData {
    let (px, nx, py, ny, pz, nz) = (Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z);
    let s = 0.6;
    let tris = [
        [px, py, pz],
        [pz, py, nx],
        [nx, py, nz],
        [nz, py, px],
        [pz, ny, px],
        [nx, ny, pz],
        [nz, ny, nx],
        [px, ny, nz],
    ]
    .map(|t| t.map(|p| p * s));
    MeshData::from_triangles(&tris)
}

pub fn tetrahedron() -> MeshData {
    let s = 0.6;
    let a = Vec3::new(1., 1., 1.) * s;
    let b = Vec3::new(1., -1., -1.) * s;
    let c = Vec3::new(-1., 1., -1.) * s;
    let d = Vec3::new(-1., -1., 1.) * s;
    MeshData::from_triangles(&[[a, b, c], [a, d, b], [a, c, d], [b, d, c]])
}

/// Square pyramid, base on y = -0.5, apex at y = 0.5.
pub fn pyramid() -> MeshData {
    let apex = Vec3::new(0., 0.5, 0.);
    let (a, b, c, d) = (
        Vec3::new(-0.5, -0.5, 0.5),
        Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(-0.5, -0.5, -0.5),
    );
    let mut t = vec![[a, b, apex], [b, c, apex], [c, d, apex], [d, a, apex]];
    t.extend(quad(d, c, b, a));
    MeshData::from_triangles(&t)
}

/// The 4 meshes the backend uploads at startup (mesh ids 0..=3).
pub fn builtin_meshes() -> Vec<MeshData> {
    vec![cube(), octahedron(), tetrahedron(), pyramid()]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layouts_and_spheres() {
        assert_eq!(size_of::<MeshInfoGpu>(), 32);
        assert_eq!(size_of::<MeshVertex>(), 24);
        let c = cube();
        assert_eq!(c.indices.len(), 36);
        let s = c.bounding_sphere();
        assert!((s[3] - 0.75f32.sqrt()).abs() < 1e-5);
    }
}

#[cfg(test)]
mod winding_tests {
    use super::*;
    /// Back-face culling is on with CCW front faces: every triangle of the (convex, origin-centered)
    /// built-in meshes must face away from the center.
    #[test]
    fn outward_ccw_winding() {
        for (k, m) in builtin_meshes().iter().enumerate() {
            for t in m.indices.as_chunks::<3>().0 {
                let p = |i: u32| Vec3::from(m.vertices[i as usize].pos);
                let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
                let n = (b - a).cross(c - a);
                assert!(n.dot((a + b + c) / 3.0) > 0.0, "mesh {k} has an inward-facing triangle");
            }
        }
    }
}
