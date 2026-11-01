//! Hemi-octahedral impostor atlases, baked on the CPU (deterministic, rayon over frames).
//!
//! Every variant (a mesh in unit mesh space) is rendered orthographically from
//! `IMP_FRAMES x IMP_FRAMES` view directions on the upper hemisphere. Frame `(i, j)` looks along
//! the hemi-octahedral direction at grid *node* `(i, j) / (N - 1) * 2 - 1` (vertex-centred, so
//! the shader blends the 3 nodes of the containing grid triangle; the centre node is the
//! zenith). Each frame covers the variant's bounding sphere (`centre`, `radius`) with
//! `IMP_RES x IMP_RES` texels, 3 x 3 supersampled, storing a small G-buffer instead of colour,
//! so the shader runs the *same* material and lighting code as the meshes:
//!
//! - word 0: alpha (coverage) 8 | material bit (bark) 1 + height fraction 7 | depth 16, where
//!   depth is the front-most surface's offset along the frame direction, `/ radius`, `[-1, 1]`;
//! - word 1: octahedral normal 8 + 8 (mesh space, facing the viewer) | ao 8 | unused 8.
//!
//! Atlas buffer layout (what the GPU reads): per variant a header of two `float4`s, (centre,
//! radius) and (AABB half extents, 0) — the card is fitted to the box's projection, so it covers
//! the silhouette and not the whole bounding sphere — then texels `[variant][frame j * N +
//! i][y * R + x]` as `uint2`.
use glam::{Vec2, Vec3};
use rayon::prelude::*;

use crate::quality::QVertex;

pub const IMP_FRAMES: u32 = 9;
pub const IMP_RES: u32 = 128;
const SS: u32 = 3;

/// Direction (unit, y >= 0) for hemi-octahedral coordinates `(a, b)` in `[-1, 1]^2`.
pub fn hemi_oct_dir(a: f32, b: f32) -> Vec3 {
    let x = (a + b) * 0.5;
    let z = (a - b) * 0.5;
    let y = 1.0 - x.abs() - z.abs();
    Vec3::new(x, y.max(0.0), z).normalize()
}

/// Inverse of [`hemi_oct_dir`]; directions below the horizon are clamped onto it.
pub fn hemi_oct_encode(d: Vec3) -> Vec2 {
    let d = Vec3::new(d.x, d.y.max(0.0), d.z);
    let s = d.x.abs() + d.y + d.z.abs();
    let (x, z) = if s > 0.0 { (d.x / s, d.z / s) } else { (1.0, 0.0) };
    Vec2::new(x + z, x - z)
}

/// Direction of frame node `(i, j)`.
pub fn frame_dir(i: u32, j: u32) -> Vec3 {
    let k = (IMP_FRAMES - 1) as f32;
    hemi_oct_dir(i as f32 / k * 2.0 - 1.0, j as f32 / k * 2.0 - 1.0)
}

/// Image basis (right, up) of a frame looking back along `d` (same formula in the shader).
pub fn frame_basis(d: Vec3) -> (Vec3, Vec3) {
    let r = if d.y > 0.9999 { Vec3::X } else { Vec3::new(d.z, 0.0, -d.x).normalize() };
    (r, d.cross(r))
}

fn oct_encode(n: Vec3) -> [u8; 2] {
    let n = n / (n.x.abs() + n.y.abs() + n.z.abs()).max(1e-12);
    let p = if n.y >= 0.0 { Vec2::new(n.x, n.z) } else { Vec2::new((1.0 - n.z.abs()) * n.x.signum(), (1.0 - n.x.abs()) * n.z.signum()) };
    let q = |v: f32| ((v * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
    [q(p.x), q(p.y)]
}

/// Decode of the octahedral normal (y-up variant used by the atlas; mirrors the shader).
pub fn oct_decode(a: u8, b: u8) -> Vec3 {
    let p = Vec2::new(a as f32 / 255.0 * 2.0 - 1.0, b as f32 / 255.0 * 2.0 - 1.0);
    let y = 1.0 - p.x.abs() - p.y.abs();
    let (x, z) = if y >= 0.0 { (p.x, p.y) } else { ((1.0 - p.y.abs()) * p.x.signum(), (1.0 - p.x.abs()) * p.y.signum()) };
    Vec3::new(x, y, z).normalize()
}

/// One baked variant.
#[derive(Clone, Debug, PartialEq)]
pub struct ImpostorVariant {
    pub centre: Vec3,
    pub radius: f32,
    /// Half extents of the mesh AABB (centred on `centre`).
    pub half: Vec3,
    /// `N * N * R * R` texels, frame-major.
    pub texels: Vec<[u32; 2]>,
}

/// Bounding sphere used for baking: the AABB centre and the farthest vertex from it; plus the
/// AABB half extents.
pub fn bake_sphere(verts: &[QVertex]) -> (Vec3, f32, Vec3) {
    let (lo, hi) = verts.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), v| {
        let p = Vec3::from_array(v.pos);
        (lo.min(p), hi.max(p))
    });
    let c = (lo + hi) * 0.5;
    let r = verts.iter().map(|v| Vec3::from_array(v.pos).distance(c)).fold(0.0f32, f32::max);
    (c, r * 1.01, (hi - lo) * 0.5 * 1.01)
}

#[derive(Clone, Copy)]
struct Sample {
    t: f32,
    n: Vec3,
    ao: f32,
    hf: f32,
    bark: bool,
}

/// Bake all frames of one mesh (`indices` index `verts`; `attr.x` = ao, `attr.y` = height
/// fraction, `attr.w` = material code, 3 = bark).
/// Needle / leaf card alpha (material code 6), the exact mirror of `card_alpha` in
/// quality.slang (without its sub-pixel branch: the bake supersamples). `u` across the card
/// (twig at 0.5; `u >= 2`: broadleaf), `v` along the shoot.
pub fn card_alpha(u: f32, v: f32) -> bool {
    if u > 3.5 {
        return spruce_alpha(u - 4.0, v);
    }
    let leaf = u > 1.5;
    let u = if leaf { u - 2.0 } else { u };
    let a = (u - 0.5).abs() * 2.0;
    let env = 0.95 * (1.0 - v.clamp(0.0, 1.0).powf(2.5)) * (v / 0.05).clamp(0.0, 1.0);
    if a > env {
        return false;
    }
    if a < 0.07 && v < 0.97 {
        return true;
    }
    if leaf {
        let q = Vec2::new(a * 2.2, (v * 7.0 + if u < 0.5 { 0.0 } else { 0.5 }).fract());
        return (q - Vec2::new(1.1 * 0.8, 0.5)).length() < 0.62;
    }
    (v * 22.0 - a * 1.6).rem_euclid(1.0) < 0.58
}

/// GLSL/HLSL `frac` (x - floor x; Rust's `fract` truncates toward zero).
fn fract_gl(x: f32) -> f32 {
    x - x.floor()
}

/// `hash2` of quality.slang.
fn hash2(p: Vec2) -> f32 {
    let mut p = (p * Vec2::new(0.1031, 0.1030)).fract_gl();
    p += Vec2::splat(p.dot(Vec2::new(p.y, p.x) + Vec2::splat(33.33)));
    fract_gl((p.x + p.y) * p.x)
}

/// Spruce shoot-cluster card (`u` in [0, 1] after removing the offset 4): mirror of
/// `spruce_keep` in quality.slang (axis + alternate side shoots as bottle-brushes, ragged needle
/// edge, open gaps between the shoots).
pub fn spruce_alpha(u: f32, v: f32) -> bool {
    let sat = |x: f32| x.clamp(0.0, 1.0);
    let x = (u - 0.5) * 0.62;
    let env = 0.31 * (1.0 - sat(v).powf(3.0)) * sat(v / 0.04);
    if x.abs() > env {
        return false;
    }
    let mut best = x.abs() / (0.06 * (1.0 - 0.5 * sat(v)));
    let fi = (v / 0.08).floor();
    for k in -3..=0 {
        let i = fi + k as f32;
        let side = if fract_gl(i * 0.5) < 0.25 { 1.0 } else { -1.0 };
        let v0 = i * 0.08 + 0.02;
        let len = 0.38 * (1.0 - 0.62 * sat(v0));
        let dir = Vec2::new(side * 0.82, 0.572);
        let pa = Vec2::new(x, v - v0);
        let h = sat(pa.dot(dir) / len);
        let d = (pa - dir * (len * h)).length();
        best = best.min(d / (0.055 * (1.0 - 0.45 * h)));
    }
    let n = hash2(Vec2::new((u * 64.0).floor(), (v * 96.0).floor()) + Vec2::splat(7.0));
    best < 0.85 + 0.55 * n
}

/// Card AO (== the instance VS for code 6): darker toward the trunk and the lower crown.
pub fn card_ao(v: f32, hf: f32) -> f32 {
    let t = (v / 0.85).clamp(0.0, 1.0);
    (0.32 + 0.68 * t * t * (3.0 - 2.0 * t)) * (0.6 + 0.4 * hf)
}

pub fn bake_variant(verts: &[QVertex], indices: &[u32]) -> ImpostorVariant {
    let (c, r, half) = bake_sphere(verts);
    let n = IMP_FRAMES;
    let res = IMP_RES;
    let sres = res * SS;
    let frames: Vec<Vec<[u32; 2]>> = (0..n * n)
        .into_par_iter()
        .map(|f| {
            let d = frame_dir(f % n, f / n);
            let (rt, up) = frame_basis(d);
            let mut buf: Vec<Option<Sample>> = vec![None; (sres * sres) as usize];
            let proj: Vec<(Vec2, f32)> = verts
                .iter()
                .map(|v| {
                    let p = Vec3::from_array(v.pos) - c;
                    let x = (p.dot(rt) / r * 0.5 + 0.5) * sres as f32;
                    let y = (0.5 - p.dot(up) / r * 0.5) * sres as f32;
                    (Vec2::new(x, y), p.dot(d) / r)
                })
                .collect();
            for t in indices.as_chunks::<3>().0.iter() {
                let [a, b, cc] = [t[0] as usize, t[1] as usize, t[2] as usize];
                let (pa, pb, pc) = (proj[a].0, proj[b].0, proj[cc].0);
                let area = (pb - pa).perp_dot(pc - pa);
                if area.abs() < 1e-12 {
                    continue;
                }
                let lo = pa.min(pb).min(pc).floor().max(Vec2::ZERO);
                let hi = pa.max(pb).max(pc).ceil().min(Vec2::splat(sres as f32 - 1.0));
                for sy in lo.y as u32..=hi.y as u32 {
                    for sx in lo.x as u32..=hi.x as u32 {
                        let q = Vec2::new(sx as f32 + 0.5, sy as f32 + 0.5);
                        let w0 = (pb - q).perp_dot(pc - q) / area;
                        let w1 = (pc - q).perp_dot(pa - q) / area;
                        let w2 = 1.0 - w0 - w1;
                        if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                            continue;
                        }
                        let tt = w0 * proj[a].1 + w1 * proj[b].1 + w2 * proj[cc].1;
                        let slot = &mut buf[(sy * sres + sx) as usize];
                        if slot.is_some_and(|s| s.t >= tt) {
                            continue;
                        }
                        let (va, vb, vc) = (&verts[a], &verts[b], &verts[cc]);
                        let code = [va.attr[3], vb.attr[3], vc.attr[3]][if w0 >= w1 && w0 >= w2 { 0 } else if w1 >= w2 { 1 } else { 2 }];
                        let lerp = |k: usize| va.attr[k] * w0 + vb.attr[k] * w1 + vc.attr[k] * w2;
                        let card = code > 5.5;
                        if card && !card_alpha(lerp(0), lerp(2)) {
                            continue;
                        }
                        let mut nn = Vec3::from_array(va.nrm) * w0 + Vec3::from_array(vb.nrm) * w1 + Vec3::from_array(vc.nrm) * w2;
                        if !card && nn.dot(d) < 0.0 {
                            nn = -nn; // two-sided: the side facing the viewer (cards keep volume normals)
                        }
                        *slot = Some(Sample {
                            t: tt,
                            n: nn.normalize_or(d),
                            ao: if card { card_ao(lerp(2), lerp(1)) } else { lerp(0) },
                            hf: lerp(1),
                            bark: (code - 3.0).abs() < 0.5,
                        });
                    }
                }
            }
            (0..res * res)
                .map(|k| {
                    let (x, y) = (k % res, k / res);
                    let (mut hits, mut tmax, mut nsum, mut ao, mut hf, mut bark) = (0u32, -1.0f32, Vec3::ZERO, 0.0f32, 0.0f32, 0u32);
                    for j in 0..SS {
                        for i in 0..SS {
                            if let Some(s) = buf[((y * SS + j) * sres + x * SS + i) as usize] {
                                hits += 1;
                                tmax = tmax.max(s.t);
                                nsum += s.n;
                                ao += s.ao;
                                hf += s.hf;
                                bark += s.bark as u32;
                            }
                        }
                    }
                    if hits == 0 {
                        return [0, 0];
                    }
                    let h = hits as f32;
                    let alpha = (h / (SS * SS) as f32 * 255.0).round() as u32;
                    let hf7 = ((hf / h).clamp(0.0, 1.0) * 127.0).round() as u32;
                    let isbark = (2 * bark > hits) as u32;
                    let depth = ((tmax.clamp(-1.0, 1.0) * 0.5 + 0.5) * 65535.0).round() as u32;
                    let o = oct_encode(nsum.normalize_or(d));
                    let ao8 = ((ao / h).clamp(0.0, 1.0) * 255.0).round() as u32;
                    [alpha | (hf7 | isbark << 7) << 8 | depth << 16, o[0] as u32 | (o[1] as u32) << 8 | ao8 << 16]
                })
                .collect()
        })
        .collect();
    ImpostorVariant { centre: c, radius: r, half, texels: frames.concat() }
}

/// Header bytes per variant in [`atlas_bytes`].
pub const HEADER_BYTES: usize = 32;

/// GPU atlas buffer: header (2 `float4` per variant) followed by all texels.
pub fn atlas_bytes(variants: &[ImpostorVariant]) -> Vec<u8> {
    let mut out = Vec::with_capacity(variants.len() * HEADER_BYTES + variants.iter().map(|v| v.texels.len() * 8).sum::<usize>());
    for v in variants {
        for f in [v.centre.x, v.centre.y, v.centre.z, v.radius, v.half.x, v.half.y, v.half.z, 0.0] {
            out.extend_from_slice(&f.to_le_bytes());
        }
    }
    for v in variants {
        out.extend_from_slice(bytemuck::cast_slice(&v.texels));
    }
    out
}

/// FNV-1a over the texels (determinism checks / reporting).
pub fn atlas_hash(variants: &[ImpostorVariant]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in atlas_bytes(variants) {
        h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad_mesh() -> (Vec<QVertex>, Vec<u32>) {
        // unit cube centred at (0, 0.5, 0), 12 triangles, outward normals
        let mut v = Vec::new();
        let mut idx = Vec::new();
        for axis in 0..3 {
            for s in [-1.0f32, 1.0] {
                let mut n = Vec3::ZERO;
                n[axis] = s;
                let (u, w) = (Vec3::from_array([0, 1, 2].map(|k| ((k == (axis + 1) % 3) as u8) as f32)), Vec3::from_array([0, 1, 2].map(|k| ((k == (axis + 2) % 3) as u8) as f32)));
                let base = v.len() as u32;
                for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                    let p = n * 0.5 + u * (a * 0.5) + w * (b * 0.5) + Vec3::Y * 0.5;
                    v.push(QVertex { pos: p.to_array(), nrm: n.to_array(), attr: [0.8, p.y, 0.0, 2.0] });
                }
                idx.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        }
        (v, idx)
    }

    #[test]
    fn hemi_oct_round_trips_on_frame_nodes() {
        for j in 0..IMP_FRAMES {
            for i in 0..IMP_FRAMES {
                let d = frame_dir(i, j);
                assert!(d.y >= 0.0 && (d.length() - 1.0).abs() < 1e-5);
                let e = hemi_oct_encode(d);
                let k = (IMP_FRAMES - 1) as f32;
                let want = Vec2::new(i as f32 / k * 2.0 - 1.0, j as f32 / k * 2.0 - 1.0);
                // the four horizon corners (|a| = |b| = 1) are each other's images: compare dirs
                assert!(hemi_oct_dir(e.x, e.y).distance(d) < 1e-4, "frame ({i},{j}): {e} vs {want}");
            }
        }
        assert!(frame_dir(IMP_FRAMES / 2, IMP_FRAMES / 2).distance(Vec3::Y) < 1e-6, "centre node is the zenith");
        let (r, u) = frame_basis(Vec3::new(0.3, 0.2, -0.9).normalize());
        assert!(r.dot(u).abs() < 1e-5 && r.y.abs() < 1e-6);
    }

    #[test]
    fn atlas_bake_is_deterministic_and_correct_for_a_cube() {
        let (v, i) = quad_mesh();
        let a = bake_variant(&v, &i);
        let b = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| bake_variant(&v, &i));
        assert_eq!(a, b, "bit-identical across thread counts");
        assert_eq!(atlas_hash(std::slice::from_ref(&a)), atlas_hash(std::slice::from_ref(&b)));
        let r = IMP_RES as usize;
        // top (zenith) frame: the cube's top face, a centred square of side 1 / (2 radius)
        let top = (IMP_FRAMES / 2 * IMP_FRAMES + IMP_FRAMES / 2) as usize;
        let frame = &a.texels[top * r * r..(top + 1) * r * r];
        let covered = frame.iter().filter(|t| t[0] & 255 > 127).count() as f32 / (r * r) as f32;
        let want = (1.0 / (2.0 * a.radius)).powi(2);
        assert!((covered - want).abs() < 0.02, "coverage {covered} vs {want}");
        // centre texel: depth = top face height above the centre (0.5 / radius), normal +y
        let t = frame[r / 2 * r + r / 2];
        let depth = ((t[0] >> 16) as f32 / 65535.0) * 2.0 - 1.0;
        assert!((depth - 0.5 / a.radius).abs() < 2e-3, "depth {depth}");
        let n = oct_decode((t[1] & 255) as u8, ((t[1] >> 8) & 255) as u8);
        assert!(n.distance(Vec3::Y) < 0.02, "normal {n}");
        // empty corners
        assert_eq!(frame[0], [0, 0]);
    }
}

#[cfg(test)]
mod spruce_card_tests {
    use super::*;

    /// The spruce shoot-cluster card is dense but open: ~75 % coverage inside its envelope
    /// (== the dither mean of `card_alpha` in quality.slang, so the far fade has no band), gaps
    /// along every row (sky / interior through the spray), and the axis always covered.
    #[test]
    fn spruce_card_is_clustered_with_gaps() {
        let n = 240;
        let (mut inside, mut on) = (0u32, 0u32);
        for j in 0..n {
            let v = (j as f32 + 0.5) / n as f32;
            let (mut row_on, mut row_off) = (0, 0);
            for i in 0..n {
                let u = (i as f32 + 0.5) / n as f32;
                let x = (u - 0.5) * 0.62;
                let env = 0.31 * (1.0 - v.powf(3.0)) * (v / 0.04).clamp(0.0, 1.0);
                if x.abs() > env {
                    assert!(!card_alpha(u + 4.0, v), "nothing outside the envelope");
                    continue;
                }
                inside += 1;
                if card_alpha(u + 4.0, v) {
                    on += 1;
                    row_on += 1;
                } else {
                    row_off += 1;
                }
            }
            if (0.1..0.85).contains(&v) {
                assert!(row_on > 0 && row_off > 0, "row v={v}: clusters and gaps");
            }
        }
        let cov = on as f32 / inside as f32;
        assert!((0.70..0.80).contains(&cov), "coverage {cov}");
        for j in 1..20 {
            assert!(spruce_alpha(0.5, j as f32 * 0.045), "the branch axis is covered");
        }
    }
}
