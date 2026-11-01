//! Vegetation and ground clutter for the mountains quality scene: procedural conifer meshes
//! per species (spruce, fir, larch, dead snag; LOD0 / LOD1 meshes + an octahedral impostor card,
//! the level picked per instance and view on the GPU by `qcull.slang`), boulders and
//! grass tufts, placed on the f64 field by slope / altitude (treeline) / forest-cluster /
//! moisture density. Instances are camera-relative f32 matrices built from f64 positions and are
//! culled + compacted on the GPU by the instanced-pass chain (cull/scan/scatter/compact).
use glam::{DVec3, Mat4, Quat, Vec3};
use lowlevel_abi::InstanceGpu;
use rayon::prelude::*;
use render_ash::mesh::MeshInfoGpu;
use render_ash::quality::QVertex;
use std::collections::HashMap;
use voxel::fields::Alpine;
use voxel::noise::{perlin2, rand2};

/// Tree species with their own geometry: mesh `3 * species + level` (0 = branches with
/// alpha-tested needle cards, 1 = whorled cone crown, 2 = low-poly crown), impostor card =
/// `MESH_IMP0 + species` (baked from level 0).
pub const TREE_SPECIES: u32 = 4;
pub const TREE_LODS: u32 = 3;
/// Fractured boulders (`voxel::boulder`): `BOULDER_VARIANTS` shapes x `BOULDER_LODS` levels,
/// mesh `MESH_BOULDER + lod * BOULDER_VARIANTS + variant`; LOD picked on the GPU.
pub const MESH_BOULDER: u32 = TREE_SPECIES * TREE_LODS;
pub const BOULDER_VARIANTS: u32 = 6;
pub const BOULDER_LODS: u32 = 3;
/// Transvoxel cells across the 2 m mesh-space body per boulder LOD, and the switch distances.
const BOULDER_CELLS: [u32; 3] = [20, 9, 5];
const BOULDER_LOD_AT: [(f32, f32); 2] = [(60.0, 14.0), (200.0, 40.0)];
/// Blade-grass tiles (`render_ash::grass`): `GRASS_VARIANTS` x 3 levels, mesh
/// `MESH_GRASS + lod * GRASS_VARIANTS + variant`; the last level dithers out (LOD_VANISH).
pub const MESH_GRASS: u32 = MESH_BOULDER + BOULDER_VARIANTS * BOULDER_LODS;
pub const GRASS_VARIANTS: u32 = 8;
/// Grass level switches (m, fade m): LOD0 -> 1, 1 -> 2, 2 -> nothing (terrain meadow shading).
pub const GRASS_LOD_AT: [(f32, f32); 3] = [(9.0, 3.0), (20.0, 6.0), (37.0, 20.0)];
/// Shrubs: `MESH_SHRUB` = green alder (small broadleaf cards, near), `+ 1` its far level
/// (same silhouette, fewer wider cards); `+ 2` / `+ 3` = mugo pine / juniper (needle cards).
pub const MESH_SHRUB: u32 = MESH_GRASS + GRASS_VARIANTS * 3;
pub const MESH_IMP0: u32 = MESH_SHRUB + 4;
pub const MESH_COUNT: u32 = MESH_IMP0 + TREE_SPECIES;
/// LOD switch distances (m) and dither crossfade widths: needle cards -> cone crown, cone ->
/// low crown, low crown -> impostor.
pub const CARD_AT: f32 = 60.0;
pub const CARD_FADE: f32 = 20.0;
const SHRUB_CARD_AT: f32 = 60.0;
pub const LOD1_AT: f32 = 260.0;
pub const LOD1_FADE: f32 = 50.0;
pub const IMP_AT: f32 = 650.0;
pub const IMP_FADE: f32 = 140.0;
/// Impostors nearer than this write per-pixel (reconstructed) depth; farther ones the card's.
pub const IMP_EXACT: f32 = IMP_AT + 0.5 * IMP_FADE + 30.0;
/// Species tags (instance `material_id` bits 10-12; the shader tints by them).
pub const SP_SPRUCE: u32 = 0;
pub const SP_FIR: u32 = 1;
pub const SP_LARCH: u32 = 2;
pub const SP_SNAG: u32 = 3;
pub const SP_ALDER: u32 = 4;
pub const SP_MUGO: u32 = 5;
const FOLIAGE: f32 = 2.0;
const BARK: f32 = 3.0;
const ROCK: f32 = 4.0;

#[derive(Default)]
pub struct VegSet {
    pub vertices: Vec<QVertex>,
    pub indices: Vec<u32>,
    pub meshes: Vec<MeshInfoGpu>,
    pub instances: Vec<InstanceGpu>,
    /// the same instances in world space for the asset format (`asset::inst`)
    pub records: Vec<asset::Instance>,
    /// trees, boulders, tufts, shrubs
    pub counts: [usize; 4],
    /// triangles if every instance drew its placed (finest) mesh
    pub triangles: usize,
    /// per mesh: (next coarser mesh or LOD_NONE, switch m bits, fade m bits, 0) for qcull
    pub lod: Vec<[u32; 4]>,
    /// Earth round 5: the grounded field's DEM weight; no vegetation where the real imagery
    /// (weight > 0.5) shows the ground (its forests are in the albedo)
    pub ground: Option<std::sync::Arc<voxel::ground::Ground>>,
    /// impostor atlas (render_ash::impostor), one variant per tree species
    pub atlas: Vec<render_ash::impostor::ImpostorVariant>,
    /// (world base, crown radius, height) of every tree and shrub: canopy ground occlusion
    pub crowns: Vec<(DVec3, f32, f32)>,
    /// trees per species (spruce, fir, larch, snag) and saplings / outliers placed
    pub species: [usize; 4],
    pub saplings: usize,
    pub outliers: usize,
    /// boulders placed as talus / erratics / scattered stones
    pub boulder_kinds: [usize; 3],
    /// (world base, footprint radius) of every boulder: contact occlusion on the terrain
    pub rocks: Vec<(DVec3, f32)>,
}

fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[derive(Default)]
struct Mb {
    v: Vec<QVertex>,
    i: Vec<u32>,
}

impl Mb {
    fn vert(&mut self, p: Vec3, n: Vec3, ao: f32, hf: f32, code: f32) -> u32 {
        self.v.push(QVertex { pos: p.to_array(), nrm: n.normalize_or(Vec3::Y).to_array(), attr: [ao, hf, 0.0, code] });
        (self.v.len() - 1) as u32
    }
    fn tri(&mut self, a: u32, b: u32, c: u32) {
        self.i.extend([a, b, c]);
    }
}

/// Crown shape of one conifer mesh.
#[derive(Clone, Copy)]
struct Crown {
    tiers: usize,
    /// branch tips per whorl
    seg: usize,
    seed: u64,
    /// whorl radius ~ (1 - f)^exp: < 1 full / columnar (fir), > 1 open cone (larch)
    exp: f32,
    r0: f32,
    droop: f32,
    /// probability that a branch wedge is missing (sparse, open crowns)
    gap: f32,
}

/// Norway spruce: narrow cone (radius ~linear in depth below the top), a thin spire, branches
/// sweeping down in the lower crown with upturned tips, hanging second-order shoots.
const SPRUCE: Crown = Crown { tiers: 13, seg: 11, seed: 11, exp: 1.05, r0: 0.20, droop: 0.045, gap: 0.0 };

/// Dead standing tree: tapering bare trunk with a broken top and short broken branch stubs
/// (bark only; the shader's snag tint makes it silver-grey).
fn snag(stubs: usize, sides: usize, seed: u64) -> Mb {
    let mut m = Mb::default();
    let top = 0.9 + 0.08 * rand2(0, 1, seed) as f32;
    let prism = |m: &mut Mb, a: Vec3, b: Vec3, ra: f32, rb: f32, ha: f32, hb: f32| {
        let ax = (b - a).normalize();
        let s = if ax.y.abs() > 0.9 { Vec3::X } else { Vec3::Y };
        let u = ax.cross(s).normalize();
        let w = ax.cross(u);
        let ring = |m: &mut Mb, c: Vec3, r: f32, hf: f32, ao: f32| -> Vec<u32> {
            (0..sides)
                .map(|k| {
                    let t = k as f32 / sides as f32 * std::f32::consts::TAU;
                    let d = u * t.cos() + w * t.sin();
                    m.vert(c + d * r, d, ao, hf, BARK)
                })
                .collect()
        };
        let r0 = ring(m, a, ra, ha, 0.5);
        let r1 = ring(m, b, rb, hb, 0.7);
        for k in 0..sides {
            let k1 = (k + 1) % sides;
            m.tri(r0[k], r1[k], r0[k1]);
            m.tri(r0[k1], r1[k], r1[k1]);
        }
        let cap = m.vert(b + ax * rb * 0.3, ax, 0.7, hb, BARK);
        for k in 0..sides {
            m.tri(r1[k], cap, r1[(k + 1) % sides]);
        }
    };
    // trunk in 3 slightly kinked sections
    let mut prev = Vec3::ZERO;
    for k in 0..3 {
        let y = top * (k + 1) as f32 / 3.0;
        let off = Vec3::new(0.012 * (rand2(k, 2, seed) as f32 - 0.5), 0.0, 0.012 * (rand2(k, 3, seed) as f32 - 0.5));
        let next = Vec3::Y * y + off * k as f32;
        let (ra, rb) = (0.034 * (1.0 - 0.8 * prev.y / top), 0.034 * (1.0 - 0.8 * y / top));
        prism(&mut m, prev, next, ra, rb.max(0.006), prev.y, y);
        prev = next;
    }
    for k in 0..stubs {
        let f = 0.25 + 0.7 * (k as f32 + rand2(k as i64, 4, seed) as f32) / stubs as f32;
        let a = rand2(k as i64, 5, seed) as f32 * std::f32::consts::TAU;
        let len = (0.05 + 0.13 * rand2(k as i64, 6, seed) as f32) * (1.0 - 0.6 * f);
        let base = Vec3::Y * (f * top);
        let dir = Vec3::new(a.cos(), -0.15 - 0.3 * rand2(k as i64, 7, seed) as f32, a.sin()).normalize();
        let tr = 0.034 * (1.0 - 0.8 * f) * 0.5;
        prism(&mut m, base, base + dir * len, tr.max(0.005), tr * 0.4 + 0.002, f * top, f * top);
    }
    m
}

/// Card material code (alpha-tested needle / leaf cards, `attr` = [u, hf, v, CARD]).
const CARD: f32 = 6.0;

/// One drooping shoot of `len` from `start` toward `d` with two crossed needle cards (flat
/// spray + a card tilted ~65 deg about the shoot axis for volume), 3 segments each. Normals
/// are crown-volume normals (outward from the trunk + up + a little of the card's own),
/// `u_off` = 2 selects the broadleaf pattern.
#[allow(clippy::too_many_arguments)]
fn shoot(
    m: &mut Mb,
    start: Vec3,
    d: Vec3,
    len: f32,
    rise: f32,
    sag: f32,
    tipup: f32,
    width: f32,
    out: Vec3,
    u_off: f32,
    cards: &[(f32, f32)],
) {
    shoot_n(m, start, d, len, rise, sag, tipup, width, out, u_off, cards, 3);
}

/// [`shoot`] with `segs` segments along each card (2 triangles each).
#[allow(clippy::too_many_arguments)]
fn shoot_n(
    m: &mut Mb,
    start: Vec3,
    d: Vec3,
    len: f32,
    rise: f32,
    sag: f32,
    tipup: f32,
    width: f32,
    out: Vec3,
    u_off: f32,
    cards: &[(f32, f32)],
    segs: u32,
) {
    let side = d.cross(Vec3::Y).normalize_or(Vec3::X);
    let pt = |s: f32| start + d * len * s + Vec3::Y * (rise * s - sag * s * s + tipup * s.powi(4));
    let dpt = |s: f32| (d * len + Vec3::Y * (rise - 2.0 * sag * s + 4.0 * tipup * s.powi(3))).normalize_or(d);
    for &(wm, tilt) in cards {
        let w = width * wm;
        let base = m.v.len() as u32;
        for k in 0..=segs {
            let s = k as f32 / segs as f32;
            let p = pt(s);
            let tg = dpt(s);
            let cn0 = tg.cross(side).normalize_or(Vec3::Y);
            let across = (side * tilt.cos() + cn0 * tilt.sin()).normalize_or(side);
            let ww = w * (0.45 + 0.55 * (1.0 - (1.0 - s) * (1.0 - s)));
            let cn = tg.cross(across).normalize_or(Vec3::Y);
            let cn = if cn.y < 0.0 { -cn } else { cn };
            let vn = (out * 0.65 + Vec3::Y * 0.55 + cn * 0.35).normalize();
            for (u, sg) in [(0.0f32, -0.5f32), (1.0, 0.5)] {
                let q = p + across * ww * sg;
                m.v.push(QVertex {
                    pos: q.to_array(),
                    nrm: vn.to_array(),
                    attr: [u + u_off, q.y.clamp(0.0, 1.0), s, CARD],
                });
            }
        }
        for k in 0..segs {
            let (a, b, c, e) = (base + 2 * k, base + 2 * k + 1, base + 2 * k + 2, base + 2 * k + 3);
            m.tri(a, b, c);
            m.tri(b, e, c);
        }
    }
}

/// Segments per LOD2 card (260 m to the impostor switch). Round 11: 1 (straight cards, the
/// shoot droop is < 1 px there; 412 -> 156 tris per spruce, 460 -> 172 per fir; same cards,
/// coverage and silhouette). `VEG_LOD2_SEGS=3` restores the round-10 bent cards.
fn lod2_segs() -> u32 {
    std::env::var("VEG_LOD2_SEGS").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
}

/// Near-LOD conifer (unit height, base at y = 0) with real branch structure: a tapered bark
/// trunk to the leader (visible below the crown and through gaps), whorls of drooping shoots
/// (rise from the trunk, sag, spruce tips turn up) each carrying two crossed alpha-tested
/// needle cards, a leader card pair; AO toward the trunk comes from the card coordinate.
///
/// `detail` 0 = near (full), 1 = mid (80-260 m: fewer whorls and shoots, no side shoots,
/// wider cards so the crown keeps its coverage and outline), 2 = far (260 m to the impostor
/// switch: a sparse set of wide cards). All levels share the generator and seed, so the
/// silhouette, whorl spacing and tone match from the near tree through to the impostor.
/// The trunk ends inside the upper crown (thin, shaded dark there) so it does not show as a
/// line through the top.
fn card_conifer(c: Crown, sp: u32, crown_base: f32, detail: u32) -> Mb {
    let mut m = Mb::default();
    let seed = c.seed + 100;
    let r = |a: i64, b: i64| rand2(a, b, seed) as f32;
    let sides = [7usize, 5, 4][detail as usize];
    let top = crown_base + 0.3;
    let far_rings = [(0.0, 0.014), (crown_base, 0.009), (top, 0.0012)];
    let rings: &[(f32, f32)] = if detail > 0 {
        &far_rings
    } else {
        &[(0.0f32, 0.014f32), (0.04, 0.010), (0.3, 0.0075), (0.6, 0.0048), (0.78, 0.0022), (0.86, 0.0006)]
    };
    let mut prev: Option<Vec<u32>> = None;
    for &(y, rad) in rings {
        let ring: Vec<u32> = (0..sides)
            .map(|k| {
                let t = k as f32 / sides as f32 * std::f32::consts::TAU;
                let d = Vec3::new(t.cos(), 0.0, t.sin());
                // inside the crown the stem is in the shade of the whorls
                let ao = (0.35 + 0.5 * y)
                    * (1.0 - 0.6 * smoothstep(crown_base as f64, crown_base as f64 + 0.25, y as f64) as f32);
                m.vert(d * rad + Vec3::Y * y, d, ao, y, BARK)
            })
            .collect();
        if let Some(p) = prev {
            for k in 0..sides {
                let k1 = (k + 1) % sides;
                m.tri(p[k], ring[k], p[k1]);
                m.tri(p[k1], ring[k], ring[k1]);
            }
        }
        prev = Some(ring);
    }
    let spruce = sp == SP_SPRUCE;
    // conifer cards (spruce and fir) carry the shoot-cluster pattern (card u offset 4)
    let uo = 4.0;
    let tiers = match detail {
        0 => c.tiers + if spruce { 9 } else { 6 },
        1 => c.tiers + 1,
        _ => (c.tiers * 6).div_ceil(10).max(5),
    };
    let per = match detail {
        0 => ((c.seg as f32 * 0.8).round() as usize).max(5),
        1 => ((c.seg as f32 * 0.65).round() as usize).max(5),
        _ => 4,
    };
    // coverage compensation: card width grows as the card count drops
    let wk = [1.0f32, 1.25, 2.1][detail as usize];
    for t in 0..tiers {
        let f = t as f32 / tiers as f32;
        let yb = crown_base + (0.93 - crown_base) * f;
        let mut len = (c.r0 * (1.0 - f).powf(c.exp) + 0.02) * 1.12;
        // spruce spire: the top ~12 % carries only short shoots around the leader
        if spruce {
            len *= 0.45 + 0.55 * smoothstep(0.0, 0.12, (1.0 - f) as f64) as f32;
        }
        let rot = r(t as i64, 0) * std::f32::consts::TAU;
        for b in 0..per {
            if c.gap > 0.0 && r(t as i64, 10 + b as i64) < c.gap {
                continue;
            }
            let ang = rot + (b as f32 + 0.4 * (r(t as i64, 20 + b as i64) - 0.5)) / per as f32 * std::f32::consts::TAU;
            let dh = Vec3::new(ang.cos(), 0.0, ang.sin());
            // spruce: lower branches sweep down from the trunk (up to ~35 deg), upper ones level
            let d = if spruce { (dh - Vec3::Y * (0.70 * (1.0 - f).powf(1.4) + 0.04)).normalize() } else { dh };
            let l = len * (0.8 + 0.35 * r(t as i64, 30 + b as i64));
            let sag = (c.droop * 5.0 + 0.10) * l * (1.0 - 0.5 * f);
            let tipup = if spruce { (0.16 + 0.08 * (1.0 - f)) * l } else { 0.03 * l };
            let start = Vec3::Y * (yb + 0.012 * r(t as i64, 40 + b as i64)) + d * 0.006;
            // main shoot: a flat spray and two cards tilted +-0.9 rad (volume from any side)
            if detail > 0 {
                let segs = if detail == 1 { 3 } else { lod2_segs() };
                shoot_n(
                    &mut m,
                    start,
                    d,
                    l,
                    0.12 * l,
                    sag,
                    tipup,
                    l * 0.62 * wk,
                    dh,
                    uo,
                    if detail == 1 { &[(1.0, 0.0), (0.8, 0.9), (0.8, -0.9)] } else { &[(1.0, 0.0), (0.85, 1.1)] },
                    segs,
                );
                continue;
            }
            shoot(
                &mut m,
                start,
                d,
                l,
                0.12 * l,
                sag,
                tipup,
                l * 0.62,
                dh,
                uo,
                &[(1.0, 0.0), (0.75, 0.9), (0.75, -0.9)],
            );
            // side shoots (second-order branching). Spruce: three per branch that hang down from
            // it (a curtain of pendant shoots, the comb habit) instead of a flat fern-like fan
            let sides: &[(i64, f32)] = if spruce { &[(0, 0.28), (1, 0.5), (2, 0.72)] } else { &[(0, 0.33), (1, 0.62)] };
            for &(k, sf) in sides {
                let side = if k % 2 == 0 { 1.0 } else { -1.0 } * (0.6 + 0.3 * r(t as i64, 50 + b as i64 + 7 * k));
                let sp0 = start + d * l * sf + Vec3::Y * (0.12 * l * sf - sag * sf * sf);
                if spruce {
                    let sd = (d + d.cross(Vec3::Y) * side * 0.55 - Vec3::Y * 1.1).normalize();
                    shoot(
                        &mut m,
                        sp0,
                        sd,
                        l * 0.38 * (1.0 - 0.3 * sf),
                        0.0,
                        0.0,
                        0.0,
                        l * 0.30,
                        dh,
                        uo,
                        &[(1.0, 0.0), (0.7, 1.2)],
                    );
                } else {
                    let sd = (d + d.cross(Vec3::Y) * side).normalize();
                    shoot(&mut m, sp0, sd, l * 0.45, 0.02 * l, sag * 0.3, 0.0, l * 0.32, d, uo, &[(1.0, 0.0)]);
                }
            }
        }
    }
    // leader: a crossed card pair up the top
    for k in 0..2 {
        let a = k as f32 * std::f32::consts::FRAC_PI_2;
        let d = Vec3::new(a.cos(), 0.0, a.sin()) * 0.15 + Vec3::Y;
        let (y0, ll, w) = if sp == SP_SPRUCE { (0.86, 0.15, 0.03) } else { (0.9, 0.1, 0.035) };
        shoot(&mut m, Vec3::Y * y0, d.normalize(), ll, 0.0, 0.0, 0.0, w, Vec3::Y, uo, &[(1.0, 0.0)]);
    }
    m
}

/// Shrub (unit height and footprint radius ~0.5, base at y = 0): a dome of stems from a
/// common root, each forking into twigs that carry small crossed cards (short cards, a few cm
/// to ~0.3 m after scaling, so the leaf / needle pattern reads as many small leaves, not as
/// a card); inner twigs short and low (dense, occluded core), outer ones arching out.
/// `needle` selects the needle pattern (mugo pine / juniper) instead of broadleaf (alder).
/// `far`: the same stems with one wider twig each (mid / far level, same silhouette).
fn card_shrub(seed: u64, needle: bool, far: bool) -> Mb {
    let mut m = Mb::default();
    let u_off = if needle { 0.0 } else { 2.0 };
    let stems = if far { 8 } else { 18 };
    for k in 0..stems as i64 {
        let r = |j: i64| rand2(k, j, seed) as f32;
        let a = (k as f32 + 0.6 * r(0)) / stems as f32 * std::f32::consts::TAU;
        let out = Vec3::new(a.cos(), 0.0, a.sin());
        // alternate inner (steep, short) and outer (arching, long) stems
        let inner = k % 3 == 0;
        let up = if inner { 1.6 + 0.6 * r(1) } else { 0.55 + 0.5 * r(1) };
        let d = (out + Vec3::Y * up).normalize();
        let l = if inner { 0.6 + 0.35 * r(2) } else { 0.55 + 0.25 * r(2) };
        let sag = if needle { 0.08 } else { 0.18 } * l;
        let start = out * 0.04;
        let pt = |t: f32| start + d * l * t - Vec3::Y * sag * t * t;
        let forks: &[(f32, f32)] = if far {
            &[(0.3, 0.6), (0.6, -0.4)]
        } else {
            &[(0.2, 0.9), (0.38, -0.8), (0.55, 0.5), (0.72, -0.4), (0.88, 0.0)]
        };
        for (fi, &(t, turn)) in forks.iter().enumerate() {
            let tw =
                (out * turn.cos() + out.cross(Vec3::Y) * turn.sin() * 0.8 + Vec3::Y * (0.8 + 0.6 * r(10 + fi as i64)))
                    .normalize();
            let tl =
                if far { 0.5 * l } else { (0.28 + 0.12 * r(20 + fi as i64)) * l * if fi == 4 { 1.3 } else { 1.0 } };
            let width = if far { 0.24 } else { 0.13 + 0.04 * r(30 + fi as i64) } * if needle { 0.85 } else { 1.0 };
            shoot(&mut m, pt(t), tw, tl, 0.05 * tl, sag * 0.4, 0.0, width, out, u_off, &[(1.0, 0.0), (0.8, 1.2)]);
        }
    }
    m
}

/// Crowns of a species (spruce, fir, larch, snag): needle-card branches, cone crown, low crown.
fn tree_lods(sp: u32) -> [Mb; 3] {
    let lods = |c: Crown, base: f32| {
        [card_conifer(c, sp, base, 0), card_conifer(c, sp, base, 1), card_conifer(c, sp, base, 2)]
    };
    match sp {
        SP_SPRUCE => lods(SPRUCE, 0.14),
        SP_FIR => {
            // silver fir: fuller columnar crown, blunt top, branches held level, dense
            let c = Crown { tiers: 14, seg: 9, seed: 15, exp: 0.6, r0: 0.19, droop: 0.012, ..SPRUCE };
            lods(c, 0.12)
        }
        SP_LARCH => {
            // larch: open, sparse conical crown, short whorls with gaps (trunk shows through)
            let c = Crown { tiers: 10, seg: 7, seed: 19, exp: 1.05, r0: 0.22, droop: 0.05, gap: 0.22 };
            lods(c, 0.2)
        }
        _ => [snag(11, 5, 23), snag(11, 5, 23), snag(5, 4, 23)],
    }
}

/// Impostor card: 4 corners at +-1 in its own plane (the impostor VS orients it), `attr.x` =
/// atlas variant.
fn impostor_card(variant: u32) -> Mb {
    let mut m = Mb::default();
    for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        m.v.push(QVertex { pos: [x, y, 0.0], nrm: [0.0, 0.0, 1.0], attr: [variant as f32, 0.0, 0.0, FOLIAGE] });
    }
    m.i.extend([0, 1, 2, 0, 2, 3]);
    m
}

/// Render mesh of a fractured boulder (`voxel::boulder`, watertight Transvoxel output) at
/// `cells` resolution. Mesh y = 0 is the burial line (`bury` of the height below it); vertex
/// AO darkens toward and below it (contact with the ground), height fraction in `attr.y`.
/// Triangles entirely deep below the line are dropped (inside the terrain).
fn boulder(seed: u64, cells: u32) -> Mb {
    let field = voxel::boulder::Boulder::new(seed);
    let bm = voxel::boulder::boulder_mesh(seed, cells).mesh;
    let bury = 0.22 + 0.2 * rand2(seed as i64, 7, 77) as f32;
    let (ylo, yhi) = bm.vertices.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(v.pos[1]), b.max(v.pos[1])));
    let y0 = ylo + bury * (yhi - ylo);
    let top = yhi - y0;
    let deep = -0.3 * (yhi - ylo);
    // unwelded, flat-shaded: each triangle takes the normal of the fracture plane it lies on
    // (exact plane of the polytope, so a facet is one crisp flat face with sharp edges);
    // triangles on the chamfer between planes keep their own face normal
    let mut m = Mb::default();
    for t in bm.indices.as_chunks::<3>().0.iter() {
        let p = t.map(|i| Vec3::from_array(bm.vertices[i as usize].pos));
        if p.iter().all(|q| q.y - y0 < deep) {
            continue;
        }
        let vn: Vec3 = t.iter().map(|&i| Vec3::from_array(bm.vertices[i as usize].normal)).sum();
        let mut fnrm = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or(Vec3::Y);
        if fnrm.dot(vn) < 0.0 {
            fnrm = -fnrm;
        }
        let c = (p[0] + p[1] + p[2]) / 3.0;
        let pn = field.facet_normal([c.x as f64, c.y as f64, c.z as f64]);
        let pn = Vec3::new(pn[0] as f32, pn[1] as f32, pn[2] as f32);
        let facet = pn.dot(fnrm) > 0.85;
        let n = if facet { pn } else { fnrm };
        let ids = p.map(|q| {
            let y = q.y - y0;
            let contact = smoothstep(-0.05, 0.4, (y / top) as f64) as f32;
            let ao = (0.35 + 0.65 * contact) * (0.8 + 0.2 * (n.y * 0.5 + 0.5));
            // attr.y: height fraction, + 2 on chamfer triangles between fracture planes (the
            // worn edges; the shader decodes the flag)
            let hf = (y / top).clamp(0.0, 1.0) + if facet { 0.0 } else { 2.0 };
            m.vert(Vec3::new(q.x, y, q.z), n, ao, hf, ROCK)
        });
        m.tri(ids[0], ids[1], ids[2]);
    }
    m
}

impl VegSet {
    /// All meshes in `MESH_*` order, the GPU LOD table and the CPU-baked impostor atlas.
    fn add_meshes(&mut self) {
        let mut lod0 = Vec::new();
        for sp in 0..TREE_SPECIES {
            let [a, b, c] = tree_lods(sp);
            lod0.push((a.v.clone(), a.i.clone()));
            self.add_mesh(a);
            self.add_mesh(b);
            self.add_mesh(c);
        }
        let rocks: Vec<Mb> = (0..BOULDER_LODS * BOULDER_VARIANTS)
            .into_par_iter()
            .map(|k| boulder(21 + (k % BOULDER_VARIANTS) as u64, BOULDER_CELLS[(k / BOULDER_VARIANTS) as usize]))
            .collect();
        for r in rocks {
            self.add_mesh(r);
        }
        for k in 0..3 * GRASS_VARIANTS {
            let (v, i) = render_ash::grass::grass_patch(32 + k % GRASS_VARIANTS, (k / GRASS_VARIANTS) as usize);
            self.add_mesh(Mb { v, i });
            // the static wind pose moves tips up to ~0.15 m and the slope shear the tile corners
            // up to ~0.25 m: widen the culling sphere
            self.meshes.last_mut().unwrap().sphere[3] += 0.45;
        }
        for (needle, seed) in [(false, 41), (true, 43)] {
            self.add_mesh(card_shrub(seed, needle, false));
            self.add_mesh(card_shrub(seed, needle, true));
        }
        self.atlas = lod0.par_iter().map(|(v, i)| render_ash::impostor::bake_variant(v, i)).collect();
        for sp in 0..TREE_SPECIES {
            self.add_mesh(impostor_card(sp));
            let v = &self.atlas[sp as usize];
            // the card is culled with the baked bounding sphere (mesh space)
            self.meshes.last_mut().unwrap().sphere = [v.centre.x, v.centre.y, v.centre.z, v.radius];
        }
        assert_eq!(self.meshes.len() as u32, MESH_COUNT);
        // VEG_LOD1_AT / VEG_IMP_AT / *_FADE / VEG_IMP_EXACT override the distances (experiments)
        let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let none = [render_ash::quality::LOD_NONE, 0, 0, 0];
        self.lod = vec![none; MESH_COUNT as usize];
        for v in 0..BOULDER_VARIANTS {
            for (l, (at, fade)) in BOULDER_LOD_AT.iter().enumerate() {
                let m = MESH_BOULDER + l as u32 * BOULDER_VARIANTS + v;
                self.lod[m as usize] = [m + BOULDER_VARIANTS, at.to_bits(), fade.to_bits(), 0];
            }
        }
        for v in 0..GRASS_VARIANTS {
            for (l, (at, fade)) in GRASS_LOD_AT.iter().enumerate() {
                let m = MESH_GRASS + l as u32 * GRASS_VARIANTS + v;
                let next = if l == 2 { render_ash::quality::LOD_VANISH } else { m + GRASS_VARIANTS };
                self.lod[m as usize] = [next, at.to_bits(), fade.to_bits(), 0];
            }
        }
        for sp in 0..TREE_SPECIES {
            self.lod[(MESH_IMP0 + sp) as usize][3] = env("VEG_IMP_EXACT", IMP_EXACT).to_bits();
        }
        let (l1, im) = (env("VEG_LOD1_AT", LOD1_AT), env("VEG_IMP_AT", IMP_AT));
        let (f1, fi) = (env("VEG_LOD1_FADE", LOD1_FADE), env("VEG_IMP_FADE", IMP_FADE));
        let (ca, cf) = (env("VEG_CARD_AT", CARD_AT), env("VEG_CARD_FADE", CARD_FADE));
        let alpha = render_ash::quality::LOD_ALPHA;
        for sp in 0..TREE_SPECIES {
            let t = TREE_LODS * sp;
            // the snag has no cards: its level 0 is plain bark
            let w = if sp == SP_SNAG { 0 } else { alpha };
            self.lod[t as usize] = [t + 1, ca.to_bits(), cf.to_bits(), w];
            self.lod[(t + 1) as usize] = [t + 2, l1.to_bits(), f1.to_bits(), w];
            self.lod[(t + 2) as usize] = [MESH_IMP0 + sp, im.to_bits(), fi.to_bits(), w];
        }
        for k in [0, 2] {
            self.lod[(MESH_SHRUB + k) as usize] = [MESH_SHRUB + k + 1, SHRUB_CARD_AT.to_bits(), 15f32.to_bits(), alpha];
            self.lod[(MESH_SHRUB + k + 1) as usize] = [render_ash::quality::LOD_NONE, 0, 0, alpha];
        }
    }

    fn add_mesh(&mut self, m: Mb) {
        let c = m.v.iter().map(|v| Vec3::from_array(v.pos)).sum::<Vec3>() / m.v.len() as f32;
        let r = m.v.iter().map(|v| Vec3::from_array(v.pos).distance(c)).fold(0.0f32, f32::max);
        self.meshes.push(MeshInfoGpu {
            index_count: m.i.len() as u32,
            first_index: self.indices.len() as u32,
            vertex_offset: self.vertices.len() as i32,
            _pad: 0,
            sphere: [c.x, c.y, c.z, r],
        });
        self.indices.extend(m.i);
        self.vertices.extend(m.v);
    }

    /// `tag`: species (bits 0-2) | crowding 0..3 (bits 3-4), stored above the 10 random bits.
    #[allow(clippy::too_many_arguments)]
    fn push(&mut self, mesh: u32, world: DVec3, cam: DVec3, yaw: f32, scale: Vec3, rnd: f64, tag: u32) {
        self.push_raw(mesh, world, cam, yaw, scale, (rnd * 1023.0) as u32 | tag << 10);
    }

    /// `push` with the 16-bit material word given directly (grass: random + ground gradient).
    fn push_raw(&mut self, mesh: u32, world: DVec3, cam: DVec3, yaw: f32, scale: Vec3, material_id: u32) {
        if self.ground.as_ref().is_some_and(|g| g.weight(world.x, world.z) > 0.5) {
            return;
        }
        let rel = (world - cam).as_vec3();
        let model = Mat4::from_scale_rotation_translation(scale, Quat::from_rotation_y(yaw), rel);
        // never place an instance around the eye (a free camera inside a crown sees a wall)
        let sph = self.meshes[mesh as usize].sphere;
        if model.transform_point3(Vec3::new(sph[0], sph[1], sph[2])).length() < sph[3] * scale.max_element() + 1.5 {
            return;
        }
        self.instances.push(InstanceGpu { model: model.to_cols_array_2d(), mesh_id: mesh, material_id, _pad: [0; 2] });
        debug_assert!((scale.x - scale.z).abs() <= 1e-6 * scale.x.abs().max(1.0));
        self.records.push(asset::Instance {
            pos: [world.x, world.y, world.z],
            yaw,
            height: scale.y,
            width: scale.x,
            mesh: mesh as u8,
            material: material_id as u16,
        });
        self.counts[match mesh {
            m if m < MESH_BOULDER => 0,
            m if m < MESH_GRASS => 1,
            m if m < MESH_SHRUB => 2,
            _ => 3,
        }] += 1;
        self.triangles += self.meshes[mesh as usize].index_count as usize / 3;
    }
}

/// Is (x, z) inside the horizontal view wedge (plus margin) and within [dmin, dmax]?
fn in_view(dx: f64, dz: f64, yaw: f64, half_fov: f64, dmin: f64, dmax: f64) -> Option<f64> {
    let d = dx.hypot(dz);
    if d < dmin || d > dmax {
        return None;
    }
    let ang = dx.atan2(-dz);
    let da = (ang - yaw + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
    (da.abs() < half_fov + 0.06).then_some(d)
}

/// The rendered terrain surface (voxel near field + far rings, exactly as meshed), binned in XZ
/// on a multi-level grid (level L: 4 * 2^L m; a triangle goes to the level its extent fits) so
/// instances can be dropped onto the triangles that are actually drawn: no floating or sinking
/// from filter / chord differences between the analytic field and the mesh.
pub struct Surface {
    world: Vec<DVec3>,
    tris: Vec<[u32; 3]>,
    levels: Vec<HashMap<(i64, i64), Vec<u32>>>,
}

const BIN0: f64 = 4.0;

impl Surface {
    pub fn new(world: &[DVec3], indices: &[u32], cam: DVec3, radius: f64) -> Surface {
        let mut tris = Vec::new();
        let mut levels: Vec<HashMap<(i64, i64), Vec<u32>>> = (0..12).map(|_| HashMap::new()).collect();
        for t in indices.as_chunks::<3>().0 {
            let [a, b, c] = [world[t[0] as usize], world[t[1] as usize], world[t[2] as usize]];
            let (x0, x1) = (a.x.min(b.x).min(c.x), a.x.max(b.x).max(c.x));
            let (z0, z1) = (a.z.min(b.z).min(c.z), a.z.max(b.z).max(c.z));
            // only what the vegetation can stand on (and faces seen from above)
            if (x0.max(cam.x.min(x1)) - cam.x).hypot(z0.max(cam.z.min(z1)) - cam.z) > radius {
                continue;
            }
            if (b - a).cross(c - a).y.abs() < 1e-9 {
                continue;
            }
            let e = (x1 - x0).max(z1 - z0);
            let l = ((e / BIN0).log2().ceil().max(0.0) as usize).min(11);
            let sz = BIN0 * (1u64 << l) as f64;
            let id = tris.len() as u32;
            tris.push([t[0], t[1], t[2]]);
            for j in (z0 / sz).floor() as i64..=(z1 / sz).floor() as i64 {
                for i in (x0 / sz).floor() as i64..=(x1 / sz).floor() as i64 {
                    levels[l].entry((i, j)).or_default().push(id);
                }
            }
        }
        Surface { world: world.to_vec(), tris, levels }
    }

    /// Highest drawn surface point above (x, z): (height, unit normal y).
    pub fn height(&self, x: f64, z: f64) -> Option<(f64, f64)> {
        let mut best: Option<(f64, f64)> = None;
        for (l, bins) in self.levels.iter().enumerate() {
            let sz = BIN0 * (1u64 << l) as f64;
            let Some(list) = bins.get(&((x / sz).floor() as i64, (z / sz).floor() as i64)) else { continue };
            for &t in list {
                let [a, b, c] = self.tris[t as usize].map(|k| self.world[k as usize]);
                let det = (b.x - a.x) * (c.z - a.z) - (c.x - a.x) * (b.z - a.z);
                let u = ((x - a.x) * (c.z - a.z) - (c.x - a.x) * (z - a.z)) / det;
                let v = ((b.x - a.x) * (z - a.z) - (x - a.x) * (b.z - a.z)) / det;
                if u < -1e-9 || v < -1e-9 || u + v > 1.0 + 1e-9 {
                    continue;
                }
                let y = a.y + u * (b.y - a.y) + v * (c.y - a.y);
                if best.is_none_or(|(by, _)| y > by) {
                    let n = (b - a).cross(c - a).normalize();
                    best = Some((y, n.y.abs()));
                }
            }
        }
        best
    }
}

/// Build meshes and place instances around `cam` looking along `yaw`.
pub fn build(field: &Alpine, surf: &Surface, cam: DVec3, yaw: f64, half_fov: f64) -> VegSet {
    let mut s = VegSet { ground: field.ground.clone(), ..Default::default() };
    s.add_meshes();
    let sd = field.seed + 4000;
    let fh = |x: f64, z: f64, cell: f64| field.height(x, z, cell);
    // Ecological tree density at (x, z), d from the camera. Returns (density, size factor,
    // local slope) or None. Uses the field filtered like the mesh at that distance:
    //  - local slope (+-max(4 m, cell) differences, gullies included): no trees on steep rock or
    //    scree (~24 deg full, none past ~35 deg);
    //  - regional slope (unfiltered base gradient): benches / ledges on steep ground get clumps;
    //  - concavity (mean of 4 neighbours at 25 m minus centre): gullies / hollows wetter and
    //    sheltered (denser, the treeline climbs), convex spurs exposed;
    //  - ragged treeline (1.1 km / 260 m noise + shelter) with a krummholz belt above it where
    //    trees shrink and survive only in tight clumps; hard ceiling below the shader's grass limit.
    // Ecological tree density at (x, z), d from the camera. Returns the stand state or None.
    // Uses the field filtered like the mesh at that distance:
    //  - local slope (+-max(4 m, cell) differences, gullies included): no trees on steep rock or
    //    scree (~24 deg full, none past ~35 deg);
    //  - regional slope (unfiltered base gradient): benches / ledges on steep ground get clumps;
    //  - concavity (mean of 4 neighbours at 25 m minus centre): gullies / hollows wetter and
    //    sheltered (denser, the treeline climbs), convex spurs exposed;
    //  - ragged treeline (1.1 km / 260 m noise + shelter) with a krummholz belt above it where
    //    trees shrink and survive only in tight clumps; hard ceiling below the shader's grass limit;
    //  - round 4: domain-warped stand boundaries frayed by 60 m / 22 m noise, clearings (120 m)
    //    and tree-fall gaps (30 m), and avalanche tracks: the deeper gullies on steep regional
    //    slopes (couloirs), kept clear of mature trees down the fall line into the runout.
    let tree = |x: f64, z: f64, d: f64| -> Option<Stand> {
        // no trees rooted on jointed outcrops (bare rock; the fringe keeps a few)
        if field.tor_mask(x, z) > 0.25 {
            return None;
        }
        let cell = (0.004 * d).max(0.5);
        let c = field.column(x, z, cell);
        let h = c[0];
        let tl0 =
            1170.0 + 75.0 * perlin2(x / 1100.0, z / 1100.0, sd + 1) + 35.0 * perlin2(x / 260.0, z / 260.0, sd + 4);
        if h > tl0 + 170.0 || h > 1330.0 {
            return None;
        }
        let g_reg = (c[1] * c[1] + c[2] * c[2]).sqrt();
        let l1 = cell.max(4.0);
        let (hx0, hx1, hz0, hz1) = (fh(x - l1, z, cell), fh(x + l1, z, cell), fh(x, z - l1, cell), fh(x, z + l1, cell));
        let g = ((hx1 - hx0).hypot(hz1 - hz0)) / (2.0 * l1);
        // deep in the forest (low valley) outcrops are mossy and fissured: a few trees on ledges
        let lowland = smoothstep(1000.0, 880.0, h);
        let sl = smoothstep(0.72 + 0.2 * lowland, 0.45 + 0.1 * lowland, g);
        if sl <= 0.0 {
            return None;
        }
        let l2 = cell.max(25.0);
        let conc = 0.25 * (fh(x - l2, z, cell) + fh(x + l2, z, cell) + fh(x, z - l2, cell) + fh(x, z + l2, cell)) - h;
        let gully = smoothstep(0.5, 5.0, conc);
        let spur = smoothstep(-1.0, -6.0, conc);
        let tl = tl0 + 45.0 * gully - 30.0 * spur;
        let u = (h - tl) / 110.0;
        let alt = smoothstep(0.7, -0.5, u);
        // stands: domain-warped (no round blobs), edges frayed by finer octaves
        let (wx, wz) = (
            x + 110.0 * perlin2(x / 300.0, z / 300.0, sd + 80),
            z + 110.0 * perlin2(x / 300.0 + 4.1, z / 300.0 - 2.3, sd + 81),
        );
        let n_st = perlin2(wx / 650.0, wz / 650.0, sd + 2)
            + 0.5 * perlin2(wx / 170.0, wz / 170.0, sd + 3)
            + 0.28 * perlin2(x / 60.0, z / 60.0, sd + 82)
            + 0.14 * perlin2(x / 22.0, z / 22.0, sd + 83);
        let clus = smoothstep(-0.18, 0.3, n_st);
        let clump = perlin2(x / 35.0, z / 35.0, sd + 5) + 0.6 * gully;
        let kz = smoothstep(-0.5, 0.5, u);
        let tight = smoothstep(-0.1 + 0.7 * kz, 0.25 + 0.7 * kz, clump);
        let stand = lerp(clus, tight, kz);
        // benches: locally gentle ground inside steep terrain hosts clumps; valley floor meadows
        let bench = smoothstep(0.55, 0.95, g_reg) * smoothstep(0.35, 0.18, g);
        let floor = 0.25 + 0.75 * smoothstep(0.04, 0.16, g_reg.max(g));
        let mut dens = alt
            * sl
            * (stand * floor + 0.5 * bench * smoothstep(0.0, 0.4, clump)).min(1.0)
            * (0.85 + 0.3 * gully - 0.25 * spur);
        // clearings and tree-fall gaps inside the forest
        let clear = smoothstep(0.42, 0.62, perlin2(x / 120.0 + 7.7, z / 120.0, sd + 84)) * 0.95
            + smoothstep(0.5, 0.7, perlin2(x / 30.0, z / 30.0 + 3.3, sd + 85)) * 0.7;
        dens *= 1.0 - clear.min(1.0);
        // avalanche tracks: couloirs (40 m concavity) on steep regional slopes, only some of them
        let l3 = cell.max(40.0);
        let conc3 = 0.25 * (fh(x - l3, z, cell) + fh(x + l3, z, cell) + fh(x, z - l3, cell) + fh(x, z + l3, cell)) - h;
        let prone = smoothstep(0.38, 0.62, g_reg) * smoothstep(-0.1, 0.3, perlin2(x / 700.0, z / 700.0, sd + 86));
        let aval = (smoothstep(2.0, 5.5, conc3) * prone).min(1.0);
        dens *= 1.0 - 0.97 * aval;
        let size = (0.45 + 0.55 * smoothstep(0.6, -0.6, u)) * (0.9 + 0.2 * gully);
        // contrast: groups or open ground with feathered edges, not a uniform sprinkle of singles
        let core = smoothstep(0.14, 0.66, dens);
        Some(Stand { dens: core.clamp(0.0, 1.0), raw: dens, size, g, h, u, aval, kz, sl })
    };
    // species: larch toward the treeline, on sunny edges and avalanche margins (golden in
    // October); fir in the lower valley; snags (dead, grey) scattered, more at the treeline
    let species = |x: f64, z: f64, st: &Stand, r: f64| -> u32 {
        let larch_p = 0.02
            + 0.2 * smoothstep(1050.0, 1230.0, st.h)
            + 0.35 * smoothstep(0.3, 0.6, perlin2(x / 220.0, z / 220.0, sd + 87))
            + 0.15 * st.aval;
        let fir_p = 0.35 * smoothstep(1050.0, 900.0, st.h);
        let snag_p = 0.018 + 0.05 * st.kz + 0.03 * smoothstep(0.3, 0.7, perlin2(x / 90.0, z / 90.0, sd + 88));
        if r < snag_p {
            SP_SNAG
        } else if r < snag_p + larch_p.min(0.7) {
            SP_LARCH
        } else if r < snag_p + larch_p.min(0.7) + fir_p {
            SP_FIR
        } else {
            SP_SPRUCE
        }
    };
    // trees: jittered grid, 7 m spacing to 1.5 km, 11 m to 7 km; evaluated in parallel
    type Found = (u32, DVec3, f32, Vec3, f64, u32, u8);
    for (sp, dmin, dmax, salt) in [(7.0, 15.0, 1500.0, 0u64), (11.0, 1500.0, 7000.0, 1)] {
        let n = (dmax / sp) as i64 + 1;
        let (ci, cj) = ((cam.x / sp).floor() as i64, (cam.z / sp).floor() as i64);
        let found: Vec<Found> = (cj - n..=cj + n)
            .into_par_iter()
            .flat_map_iter(|j| {
                let mut row = Vec::new();
                for i in ci - n..=ci + n {
                    let x = (i as f64 + rand2(i, j, sd + salt)) * sp;
                    let z = (j as f64 + rand2(i, j, sd + 10 + salt)) * sp;
                    let Some(d) = in_view(x - cam.x, z - cam.z, yaw, half_fov, dmin, dmax) else { continue };
                    let rnd = rand2(i, j, sd + 20 + salt);
                    let Some(st) = tree(x, z, d) else { continue };
                    // kind: 0 mature, 1 sapling (forest edge band / avalanche margins), 2 outlier
                    let edge = smoothstep(0.04, 0.2, st.raw) * smoothstep(0.5, 0.25, st.dens);
                    let kind = if rnd <= st.dens * 0.95 {
                        0u8
                    } else if rnd <= st.dens * 0.95 + 0.1 * edge + 0.08 * st.aval * st.sl {
                        1
                    } else if rnd > 0.9965 && st.sl > 0.6 && st.u < 0.2 {
                        2
                    } else {
                        continue;
                    };
                    let Some((y, _)) = surf.height(x, z) else { continue };
                    let r = rand2(i, j, sd + 30 + salt);
                    // age: interior trees tall and slender, edge / open-grown trees shorter,
                    // broader and branched to the ground; saplings 15-45 % size
                    let interior = smoothstep(0.55, 0.9, st.dens);
                    let age = match kind {
                        1 => 0.15 + 0.3 * r,
                        2 => 0.7 + 0.35 * r,
                        _ => (0.55 + 0.45 * r) * (0.8 + 0.25 * interior),
                    };
                    let hs = (13.0 + 17.0 * age) * st.size * if kind == 1 { 0.55 } else { 1.0 } * 1.05;
                    let sp_id = species(x, z, &st, rand2(i, j, sd + 45 + salt));
                    let shape = match sp_id {
                        SP_FIR => 1.12,
                        SP_LARCH => 0.85,
                        SP_SNAG => 0.55,
                        _ => 1.0,
                    };
                    // slender spruce .. broad fir; far trees stand for small groups (a bit wider)
                    let wide = if salt == 1 { 1.2 } else { 1.0 }
                        * (0.66 + 0.5 * rand2(i, j, sd + 40))
                        * shape
                        * (1.0 + 0.22 * (1.0 - interior))
                        * if kind == 2 { 1.15 } else { 1.0 };
                    // LOD0 of the species; the GPU picks LOD1 / the impostor per view by distance
                    let mesh = TREE_LODS * sp_id;
                    let yaw_i = (rand2(i, j, sd + 50) * std::f64::consts::TAU) as f32;
                    // trunk base sunk by its radius x slope (downhill side touches the ground)
                    let sink = 0.15 + st.g * 0.022 * hs * wide;
                    let crowd = (st.dens * 3.0).round() as u32;
                    row.push((
                        mesh,
                        DVec3::new(x, y - sink, z),
                        yaw_i,
                        Vec3::new((hs * wide) as f32, hs as f32, (hs * wide) as f32),
                        r,
                        sp_id | crowd << 3,
                        kind,
                    ));
                }
                row
            })
            .collect();
        for (mesh, p, yaw_i, scale, r, tag, kind) in found {
            s.push(mesh, p, cam, yaw_i, scale, r, tag);
            s.crowns.push((p, scale.x * 0.42, scale.y));
            s.species[(tag & 7) as usize] += 1;
            s.saplings += (kind == 1) as usize;
            s.outliers += (kind == 2) as usize;
        }
    }
    // shrub fringe: 2.5 m grid to 650 m along forest edges, in avalanche tracks (alder / willow
    // thickets, autumn yellow-brown) and the krummholz belt (dark mugo pine)
    {
        let sp = 2.5;
        let n = (650.0 / sp) as i64 + 1;
        let (ci, cj) = ((cam.x / sp).floor() as i64, (cam.z / sp).floor() as i64);
        let found: Vec<(DVec3, f32, Vec3, f64, u32)> = (cj - n..=cj + n)
            .into_par_iter()
            .flat_map_iter(|j| {
                let mut row = Vec::new();
                for i in ci - n..=ci + n {
                    let x = (i as f64 + rand2(i, j, sd + 90)) * sp;
                    let z = (j as f64 + rand2(i, j, sd + 91)) * sp;
                    let Some(d) = in_view(x - cam.x, z - cam.z, yaw, half_fov, 6.0, 650.0) else { continue };
                    let rnd = rand2(i, j, sd + 92);
                    if rnd > 0.5 {
                        continue; // no shrub density below exceeds 0.5
                    }
                    let Some(st) = tree(x, z, d) else { continue };
                    let edge = smoothstep(0.03, 0.22, st.raw) * smoothstep(0.7, 0.3, st.dens);
                    let dens = (0.22 * edge + 0.4 * st.aval * st.sl + 0.3 * st.kz * smoothstep(0.0, 0.3, st.raw))
                        * smoothstep(0.2, 0.5, st.sl);
                    if rnd > dens {
                        continue;
                    }
                    let Some((y, _)) = surf.height(x, z) else { continue };
                    let r = rand2(i, j, sd + 93);
                    let mugo = st.kz > 0.4 || rand2(i, j, sd + 94) < 0.3;
                    let (w, h) = if mugo { (1.6 + 2.4 * r, 0.8 + 1.0 * r) } else { (1.8 + 2.6 * r, 1.3 + 1.8 * r) };
                    let yaw_i = (rand2(i, j, sd + 95) * std::f64::consts::TAU) as f32;
                    row.push((
                        DVec3::new(x, y - 0.1, z),
                        yaw_i,
                        Vec3::new(w as f32, h as f32, (w * (0.8 + 0.3 * rand2(i, j, sd + 96))) as f32),
                        r,
                        if mugo { SP_MUGO } else { SP_ALDER },
                    ));
                }
                row
            })
            .collect();
        for (p, yaw_i, scale, r, tag) in found {
            s.push(MESH_SHRUB + if tag == SP_MUGO { 2 } else { 0 }, p, cam, yaw_i, scale, r, tag);
            s.crowns.push((p, scale.x * 0.5, scale.y));
        }
    }
    // boulders: 2.5 m grid to 700 m.
    //  - talus: below steep ground (cliffs / crags / outcrops up-slope within ~25-80 m) on
    //    slopes it can rest on; dense, many small blocks, a few big ones (power-law sizes);
    //  - erratics: rare, large, on gentle meadow / forest ground;
    //  - stones: a sparse sprinkle elsewhere, fewest on flat grass.
    // Sunk by the slope (downhill side touches the ground) on top of each variant's burial.
    {
        let sp = 2.5;
        let n = (700.0 / sp) as i64 + 1;
        let (ci, cj) = ((cam.x / sp).floor() as i64, (cam.z / sp).floor() as i64);
        type Rock = (u32, DVec3, f32, f32, f64, u32, usize);
        let found: Vec<Rock> = (cj - n..=cj + n)
            .into_par_iter()
            .flat_map_iter(|j| {
                let mut row = Vec::new();
                for i in ci - n..=ci + n {
                    let x = (i as f64 + rand2(i, j, sd + 60)) * sp;
                    let z = (j as f64 + rand2(i, j, sd + 61)) * sp;
                    let Some(d) = in_view(x - cam.x, z - cam.z, yaw, half_fov, 8.0, 700.0) else { continue };
                    let pick = rand2(i, j, sd + 63);
                    if pick > 0.4 {
                        continue; // no density below exceeds 0.4
                    }
                    let cell = (0.004 * d).max(0.5);
                    let c = field.column(x, z, cell);
                    let g = (c[1] * c[1] + c[2] * c[2]).sqrt();
                    if g > 1.15 {
                        continue; // on the cliff itself nothing rests
                    }
                    // steepest ground up-slope (fall line from the broad gradient)
                    let (ux, uz) = if g > 1e-3 { (c[1] / g, c[2] / g) } else { (0.0, 0.0) };
                    let mut up = 0.0f64;
                    for dl in [25.0, 50.0, 80.0] {
                        let cu = field.column(x + ux * dl, z + uz * dl, cell.max(4.0));
                        up = up.max((cu[1] * cu[1] + cu[2] * cu[2]).sqrt());
                    }
                    let tor = field.tor_mask(x, z);
                    let talus = (smoothstep(0.85, 1.5, up) * smoothstep(0.12, 0.35, g)
                        + 0.8 * smoothstep(0.05, 0.25, tor) * smoothstep(0.6, 0.3, tor))
                    .min(1.0);
                    let erratic = smoothstep(0.35, 0.15, g)
                        * smoothstep(0.55, 0.75, perlin2(x / 140.0, z / 140.0, sd + 62) * 0.5 + 0.5);
                    let (dens_t, dens_e, dens_s) =
                        (0.32 * talus, 0.006 * erratic, 0.004 + 0.012 * smoothstep(0.2, 0.6, g));
                    let kind = if pick < dens_t {
                        0
                    } else if pick < dens_t + dens_e {
                        1
                    } else if pick < dens_t + dens_e + dens_s {
                        2
                    } else {
                        continue;
                    };
                    let Some((y, _)) = surf.height(x, z) else { continue };
                    let r = rand2(i, j, sd + 64);
                    let sz = match kind {
                        0 => 0.28 + 1.5 * r.powf(2.6) + 0.5 * talus * r.powi(8),
                        1 => 1.4 + 1.6 * r,
                        // scattered stones: power law, many 0.3-0.6 m, a few to ~1.4 m
                        _ => 0.3 + 1.1 * r.powf(2.2),
                    } as f32;
                    let yaw_i = (rand2(i, j, sd + 65) * std::f64::consts::TAU) as f32;
                    let var = (rand2(i, j, sd + 66) * BOULDER_VARIANTS as f64) as u32 % BOULDER_VARIANTS;
                    // half-buried: scattered stones sit deeper (frost-heaved, grassed around)
                    let sink =
                        (g * 0.55 + 0.06 * rand2(i, j, sd + 67) + if kind == 2 { 0.06 } else { 0.0 }) * sz as f64;
                    // moss affinity (crowding bits): forest floor / moist hollows
                    let moist = ((1.0 - smoothstep(900.0, 1250.0, c[0])) * 3.0).round() as u32;
                    row.push((var, DVec3::new(x, y - sink, z), yaw_i, sz, r, moist << 3, kind));
                }
                row
            })
            .collect();
        if std::env::var("VEG_DUMP_ROCKS").is_ok() {
            for f in &found {
                eprintln!("rock kind {} at {:.1} {:.1} {:.1} size {:.2}", f.6, f.1.x, f.1.y, f.1.z, f.3);
            }
        }
        for (var, p, yaw_i, sz, r, tag, kind) in found {
            // scattered stones a little taller than wide (blocky, not lens-like once half-buried)
            let sc = if kind == 2 { Vec3::new(sz, sz * 1.25, sz) } else { Vec3::splat(sz) };
            s.push(MESH_BOULDER + var, p, cam, yaw_i, sc, r, tag);
            s.rocks.push((p, sz * 0.95));
            s.boulder_kinds[kind] += 1;
        }
    }
    // blade grass: 0.5 m tiles (quarter-turn yaw) on the 0.5 m lattice to 48 m (seamless coverage),
    // sheared onto the local ground plane in the vertex shader (gradient in the material bits:
    // no floating or buried tile edges). Density from slope / treeline times a multi-scale
    // patch field (bare and short-turf gaps at 40 m / 9 m / 2.3 m), thinned to litter under
    // crowns, cleared around boulders and trunks with tall coarse tussocks hugging their bases.
    // Species by smooth fields: fine grass, seed-head stands, coarse tussock, sparse forbs.
    // Height multi-scale (40 m stands, 4 m drifts, per tile).
    {
        let t = render_ash::grass::TILE_M as f64;
        let reach = GRASS_LOD_AT[2].0 as f64 + 0.5 * GRASS_LOD_AT[2].1 as f64 + 1.0;
        let n = (reach / t) as i64 + 1;
        let (ci, cj) = ((cam.x / t).floor() as i64, (cam.z / t).floor() as i64);
        // obstacles near the camera: (x, z, clear radius, tussock ring width)
        let cs = 4.0;
        let mut grid: HashMap<Cell, Vec<Obstacle>> = HashMap::new();
        let mut canopy: HashMap<Cell, Vec<(f64, f64, f64)>> = HashMap::new();
        for (p, r) in &s.rocks {
            if (p.x - cam.x).hypot(p.z - cam.z) < reach + 8.0 {
                grid.entry(((p.x / cs).floor() as i64, (p.z / cs).floor() as i64)).or_default().push((
                    p.x,
                    p.z,
                    *r as f64 * 0.92,
                    0.35 + 0.25 * *r as f64,
                ));
            }
        }
        for (p, cr, h) in &s.crowns {
            if (p.x - cam.x).hypot(p.z - cam.z) < reach + 12.0 {
                let trunk = (0.012 * *h as f64).clamp(0.08, 0.45);
                grid.entry(((p.x / cs).floor() as i64, (p.z / cs).floor() as i64)).or_default().push((
                    p.x,
                    p.z,
                    trunk + 0.12,
                    0.45,
                ));
                canopy.entry(((p.x / cs).floor() as i64, (p.z / cs).floor() as i64)).or_default().push((
                    p.x,
                    p.z,
                    *cr as f64 * 1.15,
                ));
            }
        }
        let tiles: Vec<(DVec3, f32, u32, u32, u32)> = (cj - n..=cj + n)
            .into_par_iter()
            .flat_map_iter(|j| {
                let mut row = Vec::new();
                for i in ci - n..=ci + n {
                    let x = (i as f64 + 0.5) * t;
                    let z = (j as f64 + 0.5) * t;
                    let (dx, dz) = (x - cam.x, z - cam.z);
                    let dh = dx.hypot(dz);
                    if dh > reach {
                        continue;
                    }
                    // view wedge with a margin of the tile size (near tiles at the frustum edge)
                    let ang = dx.atan2(-dz);
                    let da =
                        (ang - yaw + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
                    if da.abs() > half_fov + 0.06 + (1.0 / dh.max(0.5)).min(3.2) {
                        continue;
                    }
                    let c = field.column(x, z, 0.5);
                    let slope = (c[1] * c[1] + c[2] * c[2]).sqrt();
                    // nearest obstacle edge (m; negative inside) and its tussock ring width
                    let (gi, gj) = ((x / cs).floor() as i64, (z / cs).floor() as i64);
                    let (mut edge, mut ring) = (f64::MAX, 0.0);
                    let mut cover = 0.0f64;
                    for gj2 in gj - 2..=gj + 2 {
                        for gi2 in gi - 2..=gi + 2 {
                            for &(ox, oz, rr, rw) in grid.get(&(gi2, gj2)).map_or(&[][..], |v| v.as_slice()) {
                                let e = (x - ox).hypot(z - oz) - rr;
                                if e < edge {
                                    (edge, ring) = (e, rw);
                                }
                            }
                            for &(ox, oz, cr) in canopy.get(&(gi2, gj2)).map_or(&[][..], |v| v.as_slice()) {
                                let q = 1.0 - (x - ox).hypot(z - oz) / cr;
                                cover += q.max(0.0).powf(0.7);
                            }
                        }
                    }
                    if edge < 0.0 {
                        continue;
                    }
                    let hug = 1.0 - smoothstep(0.0, ring, edge);
                    let big = perlin2(x / 38.0, z / 38.0, sd + 80);
                    let mid = perlin2(x / 9.0, z / 9.0, sd + 72);
                    let small = perlin2(x / 2.3, z / 2.3, sd + 81);
                    let patch = smoothstep(-0.6, 0.15, 0.35 * big + 0.55 * mid + 0.45 * small);
                    let shade = smoothstep(0.45, 1.6, cover);
                    let dens = smoothstep(1.0, 0.5, slope)
                        * smoothstep(1380.0, 1220.0, c[0])
                        * (0.42 + 0.58 * patch)
                        * (1.0 - 0.85 * shade);
                    let dens = dens.max(hug * smoothstep(1.0, 0.5, slope) * (1.0 - 0.6 * shade));
                    if dens < 0.18 + 0.14 * rand2(i, j, sd + 73) {
                        continue;
                    }
                    // the meshed ground plane under the tile (finite differences of the surface)
                    let Some((y, _)) = surf.height(x, z) else { continue };
                    let e = 0.25;
                    let hx = |o: f64| surf.height(x + o, z).map_or(y, |h| h.0);
                    let hz = |o: f64| surf.height(x, z + o).map_or(y, |h| h.0);
                    let (gx, gz) = (
                        ((hx(e) - hx(-e)) / (2.0 * e)).clamp(-1.0, 1.0),
                        ((hz(e) - hz(-e)) / (2.0 * e)).clamp(-1.0, 1.0),
                    );
                    let stand = perlin2(x / 40.0, z / 40.0, sd + 84);
                    let drift = perlin2(x / 4.0, z / 4.0, sd + 76);
                    let hs = (0.45 + 0.6 * dens.min(1.0))
                        * (0.8 + 0.3 * stand)
                        * (0.78 + 0.22 * drift)
                        * (0.85 + 0.3 * rand2(i, j, sd + 77))
                        * (1.0 + 0.55 * hug)
                        * (1.0 - 0.35 * shade);
                    // species
                    let tus = perlin2(x / 6.0, z / 6.0, sd + 82);
                    let seedf = perlin2(x / 11.0, z / 11.0, sd + 83);
                    let rr = rand2(i, j, sd + 78);
                    let kind = if hug > 0.35 || tus > 0.42 - 0.3 * small.max(0.0) {
                        render_ash::grass::KIND_COARSE
                    } else if rr < 0.035 + 0.05 * smoothstep(0.1, 0.5, mid) && shade < 0.3 {
                        render_ash::grass::KIND_FORB
                    } else if seedf > 0.15 && rr < 0.75 {
                        render_ash::grass::KIND_SEED
                    } else {
                        render_ash::grass::KIND_FINE
                    };
                    let var = kind + 4 * (rand2(i, j, sd + 74) < 0.5) as u32;
                    // material: 4 random bits, ground gradient 6 + 6 bits (see quality.slang inst_rel)
                    let q = |g: f64| ((g * 0.5 + 0.5) * 63.0).round() as u32;
                    let mat = (rand2(i, j, sd + 75) * 15.99) as u32 | q(gx) << 4 | q(gz) << 10;
                    // a quarter-turn (square tiles still abut seamlessly): 4x the variants
                    let turn = (rand2(i, j, sd + 79) * 4.0) as u32 % 4;
                    row.push((DVec3::new(x, y, z), hs as f32, var, mat, turn));
                }
                row
            })
            .collect();
        for (p, hs, var, mat, turn) in tiles {
            s.push_raw(
                MESH_GRASS + var,
                p,
                cam,
                turn as f32 * std::f32::consts::FRAC_PI_2,
                Vec3::new(1.0, hs, 1.0),
                mat,
            );
        }
    }
    s
}

/// Grid cell of the grass obstacle / canopy lookups.
type Cell = (i64, i64);
/// Grass obstacle: (x, z, clear radius, tussock ring width).
type Obstacle = (f64, f64, f64, f64);

/// Stand state at a point (see `tree` in [`build`]).
struct Stand {
    /// contrast-shaped density of mature trees 0..1
    dens: f64,
    /// raw (pre-contrast) density: > 0 marks the forest fringe
    raw: f64,
    size: f64,
    /// local slope
    g: f64,
    h: f64,
    /// treeline coordinate (> 0 above it)
    u: f64,
    /// avalanche track 0..1
    aval: f64,
    /// krummholz belt 0..1
    kz: f64,
    /// slope suitability 0..1
    sl: f64,
}

/// Boulder contact occlusion on the terrain: ambient darkening in a ring around each block
/// (where grass meets rock), fading out at ~1.5 footprint radii.
pub fn rock_contact_ao(rocks: &[(DVec3, f32)], world: &[DVec3], ao: &mut [f32], cam: DVec3) {
    let cs = 4.0;
    let mut grid: HashMap<(i64, i64), Vec<u32>> = HashMap::new();
    for (k, r) in rocks.iter().enumerate() {
        grid.entry(((r.0.x / cs).floor() as i64, (r.0.z / cs).floor() as i64)).or_default().push(k as u32);
    }
    world.par_iter().zip(ao.par_iter_mut()).for_each(|(p, a)| {
        if (p.x - cam.x).hypot(p.z - cam.z) > 720.0 {
            return;
        }
        let (gi, gj) = ((p.x / cs).floor() as i64, (p.z / cs).floor() as i64);
        let mut occ = 0.0f64;
        for j in gj - 2..=gj + 2 {
            for i in gi - 2..=gi + 2 {
                let Some(list) = grid.get(&(i, j)) else { continue };
                for &k in list {
                    let (b, r) = rocks[k as usize];
                    let reach = 1.5 * r as f64 + 0.3;
                    let dh = (p.x - b.x).hypot(p.z - b.z);
                    if dh >= reach || (p.y - b.y).abs() > 2.0 * r as f64 + 1.0 {
                        continue;
                    }
                    let t = 1.0 - dh / reach;
                    occ += t * t;
                }
            }
        }
        *a *= (1.0 / (1.0 + 1.1 * occ)) as f32;
    });
}

/// Canopy ground occlusion: darkens the ambient term (vertex AO) of terrain under and around
/// crowns (shade, litter, no grass), by crown cover within ~1.4 crown radii.
pub fn canopy_ao(crowns: &[(DVec3, f32, f32)], world: &[DVec3], ao: &mut [f32], cam: DVec3) {
    let cs = 8.0;
    let mut grid: HashMap<(i64, i64), Vec<u32>> = HashMap::new();
    for (k, c) in crowns.iter().enumerate() {
        grid.entry(((c.0.x / cs).floor() as i64, (c.0.z / cs).floor() as i64)).or_default().push(k as u32);
    }
    world.par_iter().zip(ao.par_iter_mut()).for_each(|(p, a)| {
        if (p.x - cam.x).hypot(p.z - cam.z) > 7200.0 {
            return;
        }
        let (gi, gj) = ((p.x / cs).floor() as i64, (p.z / cs).floor() as i64);
        let mut occ = 0.0f64;
        for j in gj - 2..=gj + 2 {
            for i in gi - 2..=gi + 2 {
                let Some(list) = grid.get(&(i, j)) else { continue };
                for &k in list {
                    let (b, r, h) = crowns[k as usize];
                    let r = r as f64;
                    let dh = (p.x - b.x).hypot(p.z - b.z);
                    let reach = 1.9 * r + 0.8;
                    if dh >= reach || (p.y - b.y).abs() > h as f64 {
                        continue;
                    }
                    let t = 1.0 - dh / reach;
                    occ += t * t * (0.35 + 0.65 * (h as f64 / 25.0).min(1.0));
                }
            }
        }
        *a *= (1.0 / (1.0 + 1.3 * occ)) as f32;
    });
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}
