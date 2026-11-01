//! Procedural fractured boulders, meshed under the robustness hard rule.
//!
//! A boulder is the signed distance (lower bound, 1-Lipschitz) of a convex polytope: ~14 jittered
//! face planes on an anisotropic (elongated, flattened) body, a few *chip* planes that cut its
//! corners and edges off at shallower offsets, and one deep *fracture* plane (a large flat
//! split face), plus a little surface roughness. It is sampled and meshed by the same
//! Transvoxel path as the terrain (integer sub-grid vertices, quantised densities), so the
//! exact [`crate::topology`] checker certifies every variant closed, manifold, consistently
//! oriented and free of degenerate / flipped triangles.
//!
//! Mesh space: roughly `[-1, 1] x [-0.6, 0.6] x [-0.8, 0.8]` m before the instance scale; the
//! renderer buries the lower part in the ground.
use coords::BrickGrid;

use crate::noise::{perlin3, rand2};
use std::collections::HashMap;

use crate::{BrickMesh, DensityField, Topology, UniformLod, VoxelConfig, build_world, topology};

/// One boulder's field (planes are normalised; `scale` is the body's axis lengths).
#[derive(Clone, Debug)]
pub struct Boulder {
    planes: Vec<([f64; 3], f64)>,
    scale: [f64; 3],
    seed: u64,
}

fn norm(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|c| c / l)
}

impl Boulder {
    pub fn new(seed: u64) -> Self {
        let r = |k: i64, j: i64| rand2(k, j, seed ^ 0xb0u64);
        // body: elongated, flattened, a bit asymmetric
        let scale = [0.85 + 0.25 * r(0, 0), 0.45 + 0.2 * r(0, 1), 0.6 + 0.25 * r(0, 2)];
        let mut planes = Vec::new();
        // faces: Fibonacci directions, jittered, offsets 0.86..1.0
        let n = 14;
        for k in 0..n {
            let y = 1.0 - 2.0 * (k as f64 + 0.5) / n as f64;
            let rr = (1.0 - y * y).sqrt();
            let a = k as f64 * 2.399_963 + 6.0 * r(1, k);
            let d = norm([rr * a.cos() + 0.35 * (r(2, k) - 0.5), y + 0.35 * (r(3, k) - 0.5), rr * a.sin() + 0.35 * (r(4, k) - 0.5)]);
            planes.push((d, 0.86 + 0.14 * r(5, k)));
        }
        // chips: corners / edges knocked off (shallower random cuts), a few big and many small
        for k in 0..13 {
            let d = norm([r(6, k) - 0.5, (r(7, k) - 0.35) * 0.9, r(8, k) - 0.5]);
            planes.push((d, if k < 6 { 0.78 + 0.1 * r(9, k) } else { 0.86 + 0.06 * r(9, k) }));
        }
        // fracture: one deep flat split on a side
        let a = std::f64::consts::TAU * r(10, 0);
        planes.push((norm([a.cos(), 0.25 * (r(10, 1) - 0.5), a.sin()]), 0.5 + 0.18 * r(10, 2)));
        Self { planes, scale, seed }
    }

    /// Axis lengths of the body (mesh space).
    pub fn scale(&self) -> [f64; 3] {
        self.scale
    }

    /// Mesh-space unit normal of the cutting plane that bounds the body at `p` (the active
    /// plane of the polytope max): renderers use it for crisp, flat fracture facets.
    pub fn facet_normal(&self, p: [f64; 3]) -> [f64; 3] {
        let q = [p[0] / self.scale[0], p[1] / self.scale[1], p[2] / self.scale[2]];
        let (n, _) = self.planes.iter().map(|(n, h)| (n, n[0] * q[0] + n[1] * q[1] + n[2] * q[2] - h)).fold((&self.planes[0].0, f64::MIN), |a, b| if b.1 > a.1 { b } else { a });
        norm([n[0] / self.scale[0], n[1] / self.scale[1], n[2] / self.scale[2]])
    }

    /// Mesh-space unit normals of the cutting planes (faces, chips, fracture).
    pub fn face_normals(&self) -> Vec<[f64; 3]> {
        self.planes.iter().map(|(n, _)| norm([n[0] / self.scale[0], n[1] / self.scale[1], n[2] / self.scale[2]])).collect()
    }
}

impl DensityField for Boulder {
    fn density(&self, p: [f64; 3]) -> f64 {
        // planes act on the unit body (p / scale); multiplying by the smallest scale keeps the
        // result a lower bound of the true distance (Lipschitz <= 1)
        let q = [p[0] / self.scale[0], p[1] / self.scale[1], p[2] / self.scale[2]];
        let smin = self.scale.iter().copied().fold(f64::MAX, f64::min);
        let d = self.planes.iter().map(|(n, h)| n[0] * q[0] + n[1] * q[1] + n[2] * q[2] - h).fold(f64::MIN, f64::max) * smin;
        // weathering roughness: small, so faces stay readable as flat fractures
        d + 0.004 * perlin3([p[0] * 7.0, p[1] * 7.0, p[2] * 7.0], self.seed + 1) + 0.002 * perlin3([p[0] * 19.0, p[1] * 19.0, p[2] * 19.0], self.seed + 2)
    }
}

/// A meshed boulder and its exact topology report.
pub struct BoulderMesh {
    pub mesh: BrickMesh,
    pub topology: Topology,
}

/// Mesh boulder `seed` with cells of `2 / cells_across` m (LOD0 ~20, far LOD ~8).
pub fn boulder_mesh(seed: u64, cells_across: u32) -> BoulderMesh {
    let b = Boulder::new(seed);
    let voxel = 2.0 / cells_across as f64;
    let cfg = VoxelConfig::new(BrickGrid::new(16, voxel).expect("boulder grid"));
    let half = ((1.25 / voxel).ceil() as i64 + 15) / 16 * 16;
    let w = build_world(&b, &cfg, &UniformLod(0), ([-half; 3], [half; 3]), 0).expect("boulder meshes");
    // the roughness can leave a sub-cell speck of solid beside the surface: keep the body only
    let mut mesh = w.merged();
    keep_largest(&mut mesh);
    let topology = topology(std::iter::once(&mesh));
    BoulderMesh { mesh, topology }
}

/// Keep only the largest connected component (welded by integer position), compacting vertices
/// in first-use order (deterministic).
fn keep_largest(m: &mut BrickMesh) {
    let mut id: HashMap<[i64; 3], usize> = HashMap::new();
    let weld: Vec<usize> = m.ipos.iter().map(|p| { let n = id.len(); *id.entry(*p).or_insert(n) }).collect();
    let mut parent: Vec<usize> = (0..id.len()).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for t in m.indices.as_chunks::<3>().0.iter() {
        let a = find(&mut parent, weld[t[0] as usize]);
        for &k in &t[1..] {
            let b = find(&mut parent, weld[k as usize]);
            parent[b] = a;
        }
    }
    let mut size: HashMap<usize, usize> = HashMap::new();
    let roots: Vec<usize> = m.indices.as_chunks::<3>().0.iter().map(|t| find(&mut parent, weld[t[0] as usize])).collect();
    for &r in &roots {
        *size.entry(r).or_default() += 1;
    }
    let Some(best) = size.iter().max_by_key(|(r, n)| (**n, std::cmp::Reverse(**r))).map(|(r, _)| *r) else { return };
    let mut remap: HashMap<u32, u32> = HashMap::new();
    let mut out = BrickMesh::default();
    for (t, &r) in m.indices.as_chunks::<3>().0.iter().zip(&roots) {
        if r != best {
            continue;
        }
        for &i in t {
            let j = *remap.entry(i).or_insert_with(|| {
                out.ipos.push(m.ipos[i as usize]);
                out.vertices.push(m.vertices[i as usize]);
                out.dirs.push(m.dirs[i as usize]);
                (out.ipos.len() - 1) as u32
            });
            out.indices.push(j);
        }
    }
    *m = out;
}
