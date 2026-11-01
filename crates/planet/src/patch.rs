//! Patch meshes: `(N+1)^2` grid vertices plus `4 (N+1)` skirt vertices, positions computed in f64
//! in the planet frame and stored as f32 offsets from an f64 patch origin (the camera-relative
//! origin is formed per frame in f64, so no f32 ever holds an absolute planet coordinate).
//!
//! Stitching is in index space: on an edge whose neighbour is one level coarser the odd edge
//! vertices are collapsed onto the previous even one, so the fine edge consists of exactly the
//! coarse edge's segments between bit-identical (integer-lattice) vertices. Skirts hang below every
//! edge as a second line of defence against sub-pixel float dropouts at T-free shared edges.
use crate::cube::N;
use crate::quadtree::Key;
use glam::DVec3;

/// Vertices per patch.
pub const VERTS: u32 = (N + 1) * (N + 1) + 4 * (N + 1);

/// GPU vertex (16 bytes): offset from the patch origin (m) and the surface height (m).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PVertex {
    pub pos: [f32; 3],
    pub height: f32,
}

pub fn grid(a: u32, b: u32) -> u32 {
    a + b * (N + 1)
}

/// Skirt vertex `k` (0..=N) of edge `e` (-u, +u, -v, +v).
pub fn skirt(e: u32, k: u32) -> u32 {
    (N + 1) * (N + 1) + e * (N + 1) + k
}

/// Grid coordinates of edge `e` vertex `k`.
pub fn edge_ab(e: u32, k: u32) -> (u32, u32) {
    match e {
        0 => (0, k),
        1 => (N, k),
        2 => (k, 0),
        _ => (k, N),
    }
}

/// One patch: f64 origin and f32 vertices.
pub struct PatchMesh {
    pub key: Key,
    pub origin: DVec3,
    pub verts: Vec<PVertex>,
    pub h_min: f32,
    pub h_max: f32,
}

/// Exact f64 planet-frame position of grid vertex (a, b): `dir * (radius + height(dir))`.
pub fn vertex_pos(k: &Key, a: u32, b: u32, radius: f64, height: &impl Fn(DVec3) -> f64) -> DVec3 {
    let d = k.vertex_dir(a, b);
    d * (radius + height(d))
}

/// Skirt depth below the edge vertices (m): a quarter of the vertex spacing, at least 20 m.
pub fn skirt_depth(spacing: f64) -> f64 {
    (spacing * 0.25).max(20.0)
}

pub fn build(k: &Key, radius: f64, height: &impl Fn(DVec3) -> f64) -> PatchMesh {
    let origin = k.vertex_dir(N / 2, N / 2) * radius;
    let mut verts = vec![PVertex::default(); VERTS as usize];
    let (mut h_min, mut h_max) = (f32::MAX, f32::MIN);
    let mut hs = vec![0.0f64; ((N + 1) * (N + 1)) as usize];
    for b in 0..=N {
        for a in 0..=N {
            let d = k.vertex_dir(a, b);
            let h = height(d);
            hs[grid(a, b) as usize] = h;
            let p = d * (radius + h) - origin;
            verts[grid(a, b) as usize] = PVertex { pos: p.as_vec3().to_array(), height: h as f32 };
            h_min = h_min.min(h as f32);
            h_max = h_max.max(h as f32);
        }
    }
    let spacing = (k.vertex_dir(0, 0) - k.vertex_dir(1, 0)).length() * radius;
    let depth = skirt_depth(spacing);
    for e in 0..4 {
        for i in 0..=N {
            let (a, b) = edge_ab(e, i);
            let d = k.vertex_dir(a, b);
            let h = hs[grid(a, b) as usize];
            let p = d * (radius + h - depth) - origin;
            verts[skirt(e, i) as usize] = PVertex { pos: p.as_vec3().to_array(), height: (h - depth) as f32 };
        }
    }
    PatchMesh { key: *k, origin, verts, h_min, h_max }
}

/// Collapse target of grid vertex (a, b) under stitching mask `m`.
pub fn collapse(m: u8, a: u32, b: u32) -> (u32, u32) {
    if m & 1 != 0 && a == 0 && b % 2 == 1 {
        return (0, b - 1);
    }
    if m & 2 != 0 && a == N && b % 2 == 1 {
        return (N, b - 1);
    }
    if m & 4 != 0 && b == 0 && a % 2 == 1 {
        return (a - 1, 0);
    }
    if m & 8 != 0 && b == N && a % 2 == 1 {
        return (a - 1, N);
    }
    (a, b)
}

/// Triangle list of mask `m` (stitched grid + skirts), degenerate triangles dropped. Grid
/// triangles are counter-clockwise in (u, v) = outward facing.
pub fn indices(m: u8) -> Vec<u32> {
    let mut out = Vec::with_capacity((N * N * 6 + 4 * N * 6) as usize);
    let g = |a: u32, b: u32| {
        let (a, b) = collapse(m, a, b);
        grid(a, b)
    };
    let tri = |out: &mut Vec<u32>, t: [u32; 3]| {
        if t[0] != t[1] && t[1] != t[2] && t[0] != t[2] {
            out.extend_from_slice(&t);
        }
    };
    for b in 0..N {
        for a in 0..N {
            let (v00, v10, v01, v11) = (g(a, b), g(a + 1, b), g(a, b + 1), g(a + 1, b + 1));
            tri(&mut out, [v00, v10, v11]);
            tri(&mut out, [v00, v11, v01]);
        }
    }
    // skirts: for each edge segment, a quad down to the skirt row (odd vertices collapsed the same
    // way as the grid edge, so the skirt top follows the stitched edge exactly)
    for e in 0..4u32 {
        let stitched = m & (1 << e) != 0;
        let c = |k: u32| if stitched && k % 2 == 1 { k - 1 } else { k };
        for k in 0..N {
            let (k0, k1) = (c(k), c(k + 1));
            if k0 == k1 {
                continue;
            }
            let (a0, b0) = edge_ab(e, k0);
            let (a1, b1) = edge_ab(e, k1);
            let (e0, e1, s0, s1) = (grid(a0, b0), grid(a1, b1), skirt(e, k0), skirt(e, k1));
            tri(&mut out, [e0, s0, s1]);
            tri(&mut out, [e0, s1, e1]);
        }
    }
    out
}

/// All 16 variants concatenated: (indices, [(first, count); 16]).
pub fn index_variants() -> (Vec<u32>, [(u32, u32); 16]) {
    let mut all = Vec::new();
    let mut r = [(0u32, 0u32); 16];
    for (m, slot) in r.iter_mut().enumerate() {
        let ix = indices(m as u8);
        *slot = (all.len() as u32, ix.len() as u32);
        all.extend(ix);
    }
    (all, r)
}
