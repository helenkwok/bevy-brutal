//! Procedural blade-grass patches (techniques from CK42BB/procedural-grass-threejs, MIT,
//! reimplemented; see starter.md "Blade grass"): each patch is a 0.5 m x 0.5 m tile of tapered,
//! quadratic-Bezier-curved blades grouped in clumps (shared height / lean / tint per clump),
//! with "rounded" normals (the two edge vertices tilted outward so a flat blade shades like a
//! curved one), base AO and height fraction for dry tips and translucency in the shader, a
//! per-blade random in `attr.z` and material code 5. Tiles are placed unrotated on a 1 m grid
//! (0.5 m) so neighbouring tiles abut with no seam; the GPU picks the level per instance (LOD0 near,
//! fewer and wider blades farther, then dithered out into the terrain's meadow shading).
use glam::Vec3;

use crate::quality::QVertex;

/// Material code of blade grass (`attr.w`).
pub const GRASS_CODE: f32 = 5.0;
/// Levels: (blades per tile, Bezier segments, width multiplier). Width grows as the count drops
/// so the covered area (and the average colour) stays about the same across levels.
pub const GRASS_LODS: [(u32, u32, f32); 3] = [(38, 4, 1.0), (18, 2, 1.5), (9, 1, 2.9)];
/// Tile edge (m): small so an unrotated tile on a meadow slope stays within a few cm of the
/// ground at its edges (instances carry yaw only).
pub const TILE_M: f32 = 0.5;
/// Tallest blade (m, before the instance's height scale): bounds for culling spheres.
pub const MAX_BLADE_M: f32 = 0.62;

fn hash(a: u32, b: u32, c: u32) -> f32 {
    let mut h = a.wrapping_mul(0x9E37_79B9) ^ b.wrapping_mul(0x85EB_CA6B) ^ c.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    (h >> 8) as f32 / (1u32 << 24) as f32
}

/// Tile species (`seed % 4`): fine meadow grass, fine grass with tall seed-head stems,
/// coarse tussock (wide, stiff, blue-green blades in one or two dense clumps) and forbs
/// (fine grass with a few flower stalks). Material sub-codes in `attr.w`: 5.0 fine blade,
/// 5.1 coarse blade, 5.2 straw stem / seed head, 5.3 flower head.
pub const KIND_FINE: u32 = 0;
pub const KIND_SEED: u32 = 1;
pub const KIND_COARSE: u32 = 2;
pub const KIND_FORB: u32 = 3;
/// Flower heads per tile and level (forb tiles only; 2 triangles each).
const HEADS: [u32; 3] = [3, 2, 1];
/// Tall seed-head stems per tile and level (seed tiles; they replace blades).
const STEMS: [u32; 3] = [5, 3, 2];

/// Flower heads in a `grass_patch(seed, lod)` tile.
pub fn flower_heads(seed: u32, lod: usize) -> u32 {
    if seed % 4 == KIND_FORB { HEADS[lod] } else { 0 }
}

/// One blade-grass tile at `lod` (0..3) for `seed` (species `seed % 4`, see [`KIND_FINE`]):
/// (vertices, indices), mesh space with the ground at y = 0 and the tile spanning
/// [-TILE_M / 2, TILE_M / 2] in x and z. Deterministic.
pub fn grass_patch(seed: u32, lod: usize) -> (Vec<QVertex>, Vec<u32>) {
    let (count, segs, wmul) = GRASS_LODS[lod];
    let kind = seed % 4;
    let r = |k: u32, j: u32| hash(seed, k, j);
    let t2 = TILE_M * 0.5;
    let coarse = kind == KIND_COARSE;
    // clumps: 3 centres per tile (a tussock: 1-2 tight ones)
    let nclump = if coarse { 1 + (r(0, 9) < 0.5) as u32 } else { 3 };
    let clumps: Vec<(f32, f32, f32, f32, f32)> = (0..nclump)
        .map(|c| {
            let sp = if coarse { 0.45 } else { 0.8 };
            let (x, z) = ((r(c, 1) * 2.0 - 1.0) * t2 * sp, (r(c, 2) * 2.0 - 1.0) * t2 * sp);
            let height = if coarse { 0.95 + 0.3 * r(c, 3) } else { 0.55 + 0.75 * r(c, 3) };
            let tint = r(c, 4);
            let lean = r(c, 5) * std::f32::consts::TAU;
            (x, z, height, tint, lean)
        })
        .collect();
    let mut v = Vec::with_capacity((count * (2 * segs + 3)) as usize);
    let mut idx = Vec::with_capacity((count * (2 * segs + 1) * 3) as usize);
    // blade k is the same plant at every level (LOD1/2 keep a subset, wider)
    let stride = GRASS_LODS[0].0 / count;
    let stems = if kind == KIND_SEED { STEMS[lod] } else { 0 };
    let heads = flower_heads(seed, lod);
    for b in 0..count {
        let k = b * stride;
        // seed stems / flower stalks: the first blades of the tile (kept at every level)
        let stem = b < stems;
        let stalk = b < heads;
        let in_clump = !stem && !stalk && r(k, 10) < if coarse { 0.92 } else { 0.75 };
        let c = clumps[(r(k, 11) * nclump as f32) as usize % nclump as usize];
        let (mut x, mut z);
        if in_clump {
            // gaussian-ish offset around the clump centre (sum of two uniforms)
            let a = r(k, 12) * std::f32::consts::TAU;
            let d = if coarse { 0.05 } else { 0.075 } * (r(k, 13) + r(k, 14)) * (r(k, 15) * 0.6 + 0.4);
            x = c.0 + a.cos() * d;
            z = c.1 + a.sin() * d;
        } else {
            x = (r(k, 16) - 0.5) * TILE_M;
            z = (r(k, 17) - 0.5) * TILE_M;
        }
        // stay inside the tile (wrapped, so the tiling stays dense up to the edges)
        x = (x + t2).rem_euclid(TILE_M) - t2;
        z = (z + t2).rem_euclid(TILE_M) - t2;
        let hmul = if in_clump { c.2 } else { 0.5 + 0.4 * r(k, 18) };
        let (h, w, code) = if stem {
            // thin straw stem, taller than the leaves, a seed head (wider spindle) at the top
            (0.40 + 0.16 * r(k, 19), (0.0045 + 0.002 * r(k, 20)) * wmul.sqrt(), 5.2)
        } else if stalk {
            (0.22 + 0.14 * r(k, 19), (0.004 + 0.002 * r(k, 20)) * wmul.sqrt(), 5.2)
        } else if coarse {
            ((0.20 + 0.22 * r(k, 19)) * hmul, (0.015 + 0.009 * r(k, 20)) * wmul, 5.1)
        } else {
            ((0.16 + 0.26 * r(k, 19)) * hmul, (0.010 + 0.008 * r(k, 20)) * wmul, 5.0)
        };
        let h = h.min(MAX_BLADE_M - 0.02);
        // facing: random, clump blades fan outward from the centre a little
        let face = r(k, 21) * std::f32::consts::TAU;
        let out = if in_clump { (z - c.1).atan2(x - c.0) } else { face };
        let lean_dir = if in_clump { out * 0.6 + c.4 * 0.4 } else { face };
        // stems stand stiffer; tussock blades arch outward more
        let bend = if stem || stalk {
            0.25
        } else if coarse {
            1.3
        } else {
            1.0
        };
        let curve = (0.12 + 0.35 * r(k, 22)) * h * bend;
        let d = Vec3::new(lean_dir.cos(), 0.0, lean_dir.sin());
        let side = Vec3::new(face.cos(), 0.0, face.sin()).cross(Vec3::Y).normalize();
        let side = if side.length_squared() < 0.5 { Vec3::X } else { side };
        let root = Vec3::new(x, 0.0, z);
        let rnd = 0.65 * (if in_clump { c.3 } else { r(k, 23) }) + 0.35 * r(k, 24);
        // quadratic Bezier p0 = root, p1 = root + up * h * 0.6, p2: tip leaning by `curve`
        let p1 = root + Vec3::Y * h * 0.6;
        let p2 = root + Vec3::Y * h * (1.0 - 0.25 * curve / h) + d * curve * 1.6;
        let bez = |t: f32| root * (1.0 - t) * (1.0 - t) + p1 * 2.0 * (1.0 - t) * t + p2 * t * t;
        let dbez = |t: f32| (p1 - root) * 2.0 * (1.0 - t) + (p2 - p1) * 2.0 * t;
        let base = v.len() as u32;
        for s in 0..=segs {
            let t = s as f32 / (segs as f32 + 0.6);
            let p = bez(t);
            let tan = dbez(t).normalize();
            let n = side.cross(tan).normalize();
            let n = if n.y < 0.0 { -n } else { n };
            // leaves taper; a stem keeps its width and swells into the seed head near the top
            let half = if stem {
                w * (1.0 + 1.8 * smooth(0.62, 0.8, t))
            } else if stalk {
                w
            } else {
                w * (1.0 - 0.85 * t)
            };
            // rounded normals: edges tilted +-35 deg around the blade axis
            let (nl, nr) = ((n - side * 0.7).normalize(), (n + side * 0.7).normalize());
            let ao = 0.25 + 0.75 * (t / 0.35).min(1.0).powf(0.8);
            for (sg, nn) in [(-1.0f32, nl), (1.0, nr)] {
                let q = p + side * half * sg;
                v.push(QVertex { pos: q.to_array(), nrm: nn.to_array(), attr: [ao, t, rnd, code] });
            }
        }
        let tip = bez(1.0);
        let tn = side.cross(dbez(1.0).normalize()).normalize();
        let tn = if tn.y < 0.0 { -tn } else { tn };
        v.push(QVertex { pos: tip.to_array(), nrm: tn.to_array(), attr: [1.0, 1.0, rnd, code] });
        for s in 0..segs {
            let (a, bb, cc, dd) = (base + 2 * s, base + 2 * s + 1, base + 2 * s + 2, base + 2 * s + 3);
            idx.extend([a, bb, cc, bb, dd, cc]);
        }
        let (l, rr) = (base + 2 * segs, base + 2 * segs + 1);
        idx.extend([l, rr, base + 2 * segs + 2]);
        if stalk {
            // flower head: a small upward-facing four-point star (2 triangles) at the stalk top,
            // colour picked in the shader from the per-plant random (5.3)
            let rad = (0.018 + 0.012 * r(k, 25)) * wmul.sqrt().min(1.4);
            let a0 = r(k, 26) * std::f32::consts::TAU;
            let hb = v.len() as u32;
            let up = Vec3::new(0.15 * d.x, 1.0, 0.15 * d.z).normalize();
            for q in 0..4 {
                let a = a0 + q as f32 * std::f32::consts::FRAC_PI_2;
                let p = tip + Vec3::new(a.cos() * rad, 0.004 * (q % 2) as f32, a.sin() * rad);
                v.push(QVertex { pos: p.to_array(), nrm: up.to_array(), attr: [1.0, 1.0, rnd, 5.3] });
            }
            idx.extend([hb, hb + 1, hb + 2, hb, hb + 2, hb + 3]);
        }
    }
    (v, idx)
}

fn smooth(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(v: &[QVertex], i: &[u32]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for x in bytemuck::cast_slice::<QVertex, u8>(v).iter().chain(bytemuck::cast_slice::<u32, u8>(i)) {
            h = (h ^ *x as u64).wrapping_mul(0x100_0000_01b3);
        }
        h
    }

    /// Bit-identical on every call and from any thread; variants differ.
    #[test]
    fn grass_patches_are_deterministic() {
        for lod in 0..3 {
            let (a, ai) = grass_patch(7, lod);
            let d = digest(&a, &ai);
            let threads: Vec<u64> = (0..4)
                .map(|_| {
                    std::thread::spawn(move || {
                        let (v, i) = grass_patch(7, lod);
                        digest(&v, &i)
                    })
                })
                .map(|t| t.join().unwrap())
                .collect();
            assert!(threads.iter().all(|&t| t == d));
            let (b, bi) = grass_patch(8, lod);
            assert_ne!(digest(&b, &bi), d);
        }
    }

    /// Density / triangle budget per level, geometry sanity: roots inside the tile, heights
    /// bounded, no degenerate triangles, normals unit and facing up-ish, covered area roughly
    /// preserved across levels (so the dithered LOD switches do not change the tone).
    #[test]
    fn grass_density_budget_and_geometry() {
        let budget_tris = [360usize, 100, 30];
        let mut area = [0.0f32; 3];
        for lod in 0..3 {
            for seed in 0..8 {
                let (v, i) = grass_patch(seed, lod);
                let tris = i.len() / 3;
                assert!(tris <= budget_tris[lod], "lod {lod}: {tris} triangles");
                assert_eq!(tris as u32, GRASS_LODS[lod].0 * (2 * GRASS_LODS[lod].1 + 1) + 2 * flower_heads(seed, lod));
                for q in &v {
                    assert!(q.pos[0].abs() <= TILE_M * 0.5 + 0.3 && q.pos[2].abs() <= TILE_M * 0.5 + 0.3);
                    assert!(q.pos[1] >= -1e-6 && q.pos[1] <= MAX_BLADE_M);
                    let n = Vec3::from_array(q.nrm);
                    assert!((n.length() - 1.0).abs() < 1e-3 && n.y > -0.2);
                }
                for t in i.as_chunks::<3>().0 {
                    let p = t.map(|k| Vec3::from_array(v[k as usize].pos));
                    let a = (p[1] - p[0]).cross(p[2] - p[0]).length() * 0.5;
                    assert!(a > 1e-9, "degenerate triangle");
                    area[lod] += a;
                }
                // roots (attr.y == 0) inside the tile
                assert!(
                    v.iter()
                        .filter(|q| q.attr[1] == 0.0)
                        .all(|q| q.pos[0].abs() <= TILE_M * 0.5 + 0.06 && q.pos[2].abs() <= TILE_M * 0.5 + 0.06)
                );
            }
        }
        // blades per m^2: 152 near (38 per 0.25 m^2 tile), decreasing; covered area within 40 %
        assert!((GRASS_LODS[0].0 as f32 / (TILE_M * TILE_M) - 152.0).abs() < 1.0);
        assert!(GRASS_LODS.windows(2).all(|w| w[1].0 < w[0].0));
        for lod in 1..3 {
            let r = area[lod] / area[0];
            assert!((0.6..1.4).contains(&r), "lod {lod} area ratio {r}");
        }
    }
}
