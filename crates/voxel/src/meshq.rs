//! Mesh-quality scan for torn / spiky geometry ("spike triangles").
//!
//! A density jump makes the mesher fold the surface back on itself within a cell or two: thin
//! fins and teeth whose two sides meet at a knife edge, plus long needle triangles. Counted per
//! triangle: `fold` = the triangle has a *convex* edge where the adjacent face's normal is more
//! than 120 deg from its own (a knife-edge ridge, fin or tooth: the spike); `crease` = the same
//! but concave (a narrow V-slot such as a joint crack, legitimate); `needle` = longest
//! edge^2 / (2 area) > 25 (an equilateral triangle is 1.15). Needles also occur on a continuous
//! field (LOD transition cells, grazing cuts), so compare counts. Winding as the mesher's:
//! cross(b - a, c - a) points to air.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpikeCounts {
    pub tris: usize,
    pub folds: usize,
    pub creases: usize,
    pub needles: usize,
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Per-triangle flags (`[fold, needle, crease]`) of an indexed mesh with welded vertices.
pub fn spike_flags(pos: &[[f64; 3]], indices: &[u32]) -> Vec<[bool; 3]> {
    let nt = indices.len() / 3;
    let mut nrm = vec![[0.0f64; 3]; nt];
    let mut flags = vec![[false; 3]; nt];
    for t in 0..nt {
        let [a, b, c] = [0, 1, 2].map(|k| pos[indices[3 * t + k] as usize]);
        let n = cross(sub(b, a), sub(c, a));
        let a2 = dot(n, n).sqrt();
        if a2 > 0.0 {
            nrm[t] = n.map(|v| v / a2);
        }
        let l = [
            dot(sub(b, a), sub(b, a)),
            dot(sub(c, b), sub(c, b)),
            dot(sub(a, c), sub(a, c)),
        ];
        let lmax = l[0].max(l[1]).max(l[2]);
        flags[t][1] = a2 <= 0.0 || lmax / a2 > 25.0;
    }
    let mut edge: HashMap<(u32, u32), usize> = HashMap::with_capacity(indices.len());
    for t in 0..nt {
        for k in 0..3 {
            let (i, j) = (indices[3 * t + k], indices[3 * t + (k + 1) % 3]);
            edge.insert((i, j), t);
        }
    }
    for t in 0..nt {
        for k in 0..3 {
            let (i, j) = (indices[3 * t + k], indices[3 * t + (k + 1) % 3]);
            if let Some(&u) = edge.get(&(j, i))
                && dot(nrm[t], nrm[u]) < -0.5
            {
                // the neighbour's far vertex on the solid side of this face: convex edge
                let w = (0..3)
                    .map(|m| indices[3 * u + m])
                    .find(|&v| v != i && v != j)
                    .unwrap_or(i);
                let a = pos[indices[3 * t] as usize];
                if dot(nrm[t], sub(pos[w as usize], a)) < 0.0 {
                    flags[t][0] = true;
                } else {
                    flags[t][2] = true;
                }
            }
        }
    }
    flags
}

/// Spike counts over the triangles selected by `keep` (called with the triangle's centroid).
pub fn spike_counts(
    pos: &[[f64; 3]],
    indices: &[u32],
    keep: impl Fn([f64; 3]) -> bool,
) -> SpikeCounts {
    let flags = spike_flags(pos, indices);
    let mut s = SpikeCounts::default();
    for (t, f) in flags.iter().enumerate() {
        let [a, b, c] = [0, 1, 2].map(|k| pos[indices[3 * t + k] as usize]);
        if !keep([0, 1, 2].map(|i| (a[i] + b[i] + c[i]) / 3.0)) {
            continue;
        }
        s.tris += 1;
        s.folds += f[0] as usize;
        s.needles += f[1] as usize;
        s.creases += f[2] as usize;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_grid_has_no_spikes_and_a_folded_fin_is_found() {
        // 4x4 flat grid
        let mut pos = Vec::new();
        for z in 0..5 {
            for x in 0..5 {
                pos.push([x as f64, 0.0, z as f64]);
            }
        }
        let v = |x: u32, z: u32| z * 5 + x;
        let mut idx = Vec::new();
        for z in 0..4 {
            for x in 0..4 {
                idx.extend([
                    v(x, z),
                    v(x, z + 1),
                    v(x + 1, z + 1),
                    v(x, z),
                    v(x + 1, z + 1),
                    v(x + 1, z),
                ]);
            }
        }
        assert_eq!(
            spike_counts(&pos, &idx, |_| true),
            SpikeCounts {
                tris: 32,
                folds: 0,
                creases: 0,
                needles: 0
            }
        );
        // two triangles sharing an edge, folded nearly flat onto each other: behind the first
        // face (its solid side) a thin fin, in front of it (its air side) a thin slot
        let fin = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 1.0, -0.05],
        ];
        let s = spike_counts(&fin, &[0, 1, 2, 1, 0, 3], |_| true);
        assert_eq!((s.tris, s.folds, s.creases), (2, 2, 0));
        let slot = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 1.0, 0.05],
        ];
        let s = spike_counts(&slot, &[0, 1, 2, 1, 0, 3], |_| true);
        assert_eq!((s.tris, s.folds, s.creases), (2, 0, 2));
        // a needle
        let nd = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 0.1, 0.0]];
        assert_eq!(spike_counts(&nd, &[0, 1, 2], |_| true).needles, 1);
    }
}
