//! Per-face quadtree: integer keys, screen-space-error selection from an f64 camera, frustum and
//! horizon culling, 2:1 restriction and the stitching mask.
use crate::cube::{self, FACES, LMAX, M, N};
use glam::DVec3;
use std::collections::{HashMap, HashSet};

/// A quadtree node: face, level and patch indices `i` (along u), `j` (along v) in `0..2^level`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Key {
    pub face: u8,
    pub level: u8,
    pub i: u32,
    pub j: u32,
}

/// Edge order used by masks and neighbour lookups: -u, +u, -v, +v.
pub const EDGES: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

impl Key {
    pub const fn root(face: u8) -> Self {
        Key { face, level: 0, i: 0, j: 0 }
    }
    /// Patch side in centred lattice units (`2M / 2^level`).
    pub fn size_c(&self) -> i64 {
        (2 * M) >> self.level
    }
    /// Lattice step between patch vertices, in face lattice units `U` (`M / (N 2^level)`).
    pub fn step(&self) -> i64 {
        (M / N as i64) >> self.level
    }
    /// Face lattice origin (U, V) of vertex (0, 0).
    pub fn origin_uv(&self) -> (i64, i64) {
        let s = M >> self.level;
        (self.i as i64 * s, self.j as i64 * s)
    }
    pub fn children(&self) -> [Key; 4] {
        let (f, l, i, j) = (self.face, self.level + 1, self.i * 2, self.j * 2);
        [
            Key { face: f, level: l, i, j },
            Key { face: f, level: l, i: i + 1, j },
            Key { face: f, level: l, i, j: j + 1 },
            Key { face: f, level: l, i: i + 1, j: j + 1 },
        ]
    }
    pub fn parent(&self) -> Option<Key> {
        (self.level > 0).then(|| Key { face: self.face, level: self.level - 1, i: self.i / 2, j: self.j / 2 })
    }
    /// Same-level neighbour across edge `e` (see [`EDGES`]), folding over cube edges onto the
    /// adjacent face. Pure integer arithmetic on the patch centre.
    pub fn neighbour(&self, e: usize) -> Key {
        let sz = self.size_c();
        let (di, dj) = EDGES[e];
        let cu = -M + self.i as i64 * sz + sz / 2 + di * sz;
        let cv = -M + self.j as i64 * sz + sz / 2 + dj * sz;
        let p = if cu.abs() > M || cv.abs() > M {
            let (n, u, v) = FACES[self.face as usize];
            let (n, u, v) = (n.as_i64vec3(), u.as_i64vec3(), v.as_i64vec3());
            // fold: the overflow along u (or v) turns into depth along -n on the next face
            let (ex, axis, other, oc) = if cu.abs() > M {
                (cu.abs() - M, u * cu.signum(), v, cv)
            } else {
                (cv.abs() - M, v * cv.signum(), u, cu)
            };
            n * (M - ex) + axis * M + other * oc
        } else {
            let q = cube::cube_point(self.face as usize, cu, cv);
            glam::I64Vec3::new(q[0], q[1], q[2])
        };
        // decompose on the face whose normal component is +M
        let f = FACES.iter().position(|(n, _, _)| p.dot(n.as_i64vec3()) == M).expect("folded point on a face");
        let (_, u, v) = FACES[f];
        let (cu, cv) = (p.dot(u.as_i64vec3()), p.dot(v.as_i64vec3()));
        let ix = (cu + M - sz / 2) / sz;
        let jx = (cv + M - sz / 2) / sz;
        Key { face: f as u8, level: self.level, i: ix as u32, j: jx as u32 }
    }
    /// Unit direction of patch vertex `(a, b)` in `0..=N`.
    pub fn vertex_dir(&self, a: u32, b: u32) -> DVec3 {
        let (u0, v0) = self.origin_uv();
        let s = self.step();
        cube::lattice_dir(self.face as usize, u0 + a as i64 * s, v0 + b as i64 * s)
    }
    /// Is `self` equal to or an ancestor of `k`?
    pub fn covers(&self, k: &Key) -> bool {
        self.face == k.face
            && self.level <= k.level
            && (k.i >> (k.level - self.level)) == self.i
            && (k.j >> (k.level - self.level)) == self.j
    }
}

/// Camera and planet description for one selection.
#[derive(Clone, Copy, Debug)]
pub struct View {
    /// Camera position in the planet frame (m, f64).
    pub cam: DVec3,
    /// World-space frustum planes (xyz normal pointing inside, w offset), relative to the camera:
    /// a point `p` (camera-relative) is inside iff `dot(n, p) + w >= 0` for all planes.
    pub planes: [glam::DVec4; 6],
    /// Pixels per radian at the image centre (`height / (2 tan(fov_y / 2))`).
    pub px_per_rad: f64,
    /// Split while the projected patch error exceeds this many pixels.
    pub max_error_px: f64,
    pub max_level: u8,
}

/// Planet shape bounds used for culling and error estimates.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub radius: f64,
    /// Lowest / highest surface height relative to `radius` (m).
    pub h_min: f64,
    pub h_max: f64,
}

/// Bounding data of a patch: centre direction, cone half-angle, bounding sphere (planet frame).
#[derive(Clone, Copy, Debug)]
pub struct Bounds {
    pub dir: DVec3,
    pub cos_half: f64,
    pub half: f64,
    pub centre: DVec3,
    pub radius: f64,
    /// Vertex spacing on the sphere (m).
    pub spacing: f64,
}

pub fn bounds(k: &Key, s: &Shape) -> Bounds {
    let c = k.vertex_dir(N / 2, N / 2);
    let corners = [k.vertex_dir(0, 0), k.vertex_dir(N, 0), k.vertex_dir(0, N), k.vertex_dir(N, N)];
    let mids = [k.vertex_dir(N / 2, 0), k.vertex_dir(N / 2, N), k.vertex_dir(0, N / 2), k.vertex_dir(N, N / 2)];
    let cos_half = corners.iter().chain(&mids).map(|d| d.dot(c)).fold(1.0f64, f64::min);
    let half = cos_half.clamp(-1.0, 1.0).acos();
    let (r0, r1) = (s.radius + s.h_min, s.radius + s.h_max);
    // the patch lies in the cone around `c` (half-angle `half`) between radii r0 and r1; a sphere
    // centred on the axis between the lowest (rim at r0) and highest (axis at r1) extent, radius =
    // the farthest of the four extreme points (axis / rim at r0 / r1)
    let (sh, ch) = (half.sin(), cos_half);
    let z = 0.5 * (r0 * ch + r1);
    let pts = [(0.0, r1), (r1 * sh, r1 * ch), (r0 * sh, r0 * ch), (0.0, r0)];
    let radius = pts.iter().map(|&(x, y)| x.hypot(y - z)).fold(0.0, f64::max) * 1.0001 + 1.0;
    let centre = c * z;
    let spacing = (k.vertex_dir(0, 0) - k.vertex_dir(1, 1)).length() * s.radius / std::f64::consts::SQRT_2;
    Bounds { dir: c, cos_half, half, centre, radius, spacing }
}

/// Horizon test: can any point of the cone (half-angle `b.half`, radius <= `r1`) be seen from the
/// camera past the occluding sphere of radius `r0`? Conservative (never culls a visible patch).
pub fn above_horizon(b: &Bounds, cam: DVec3, s: &Shape) -> bool {
    let d = cam.length();
    let r0 = s.radius + s.h_min;
    let r1 = s.radius + s.h_max;
    if d <= r0 {
        return true;
    }
    // a point at radius r1 and angle g from the camera direction is visible iff
    // g <= acos(r0 / d) + acos(r0 / r1)
    let lim = (r0 / d).acos() + (r0 / r1).min(1.0).acos();
    let g = (cam / d).dot(b.dir).clamp(-1.0, 1.0).acos();
    g - b.half <= lim
}

pub fn in_frustum(b: &Bounds, v: &View) -> bool {
    let rel = b.centre - v.cam;
    v.planes.iter().all(|p| p.truncate().dot(rel) + p.w >= -b.radius)
}

/// Projected geometric error (px) of a patch: chord sagitta of one vertex spacing plus a terrain
/// term (unresolved relief ~ 2 % of the spacing), over the distance to the bounding sphere.
pub fn error_px(b: &Bounds, v: &View, s: &Shape) -> f64 {
    let e = b.spacing * b.spacing / (8.0 * s.radius) + 0.02 * b.spacing;
    let dist = ((b.centre - v.cam).length() - b.radius).max(1.0);
    e / dist * v.px_per_rad
}

/// Selected leaves: every key that tiles the sphere, with visibility.
#[derive(Default, Debug, Clone)]
pub struct Selection {
    /// All leaves (they tile the sphere exactly once).
    pub leaves: HashSet<Key>,
    /// Visible leaves with their stitching mask (bit e set = neighbour across edge e is coarser).
    pub visible: Vec<(Key, u8)>,
    /// Leaves added by the 2:1 balance.
    pub balance_splits: u32,
}

impl Selection {
    /// The leaf covering `k` (k itself or an ancestor), if the region of `k` is not subdivided
    /// finer than `k`.
    pub fn covering(&self, k: &Key) -> Option<Key> {
        let mut c = Some(*k);
        while let Some(x) = c {
            if self.leaves.contains(&x) {
                return Some(x);
            }
            c = x.parent();
        }
        None
    }
}

/// Select leaves: split by screen-space error until `max_error_px` or `max_level`; culled nodes
/// stay unsplit leaves (they keep the tiling complete for the balance). Then enforce the 2:1
/// restriction and compute visibility + masks.
pub fn select(v: &View, s: &Shape) -> Selection {
    let mut sel = Selection::default();
    let mut stack: Vec<Key> = (0..6).map(Key::root).collect();
    while let Some(k) = stack.pop() {
        let b = bounds(&k, s);
        let vis = above_horizon(&b, v.cam, s) && in_frustum(&b, v);
        if vis && k.level < v.max_level.min(LMAX as u8) && error_px(&b, v, s) > v.max_error_px {
            stack.extend(k.children());
        } else {
            sel.leaves.insert(k);
        }
    }
    balance(&mut sel);
    let mut visible: Vec<(Key, u8)> = sel
        .leaves
        .iter()
        .filter(|k| {
            let b = bounds(k, s);
            above_horizon(&b, v.cam, s) && in_frustum(&b, v)
        })
        .map(|k| (*k, mask(&sel, k)))
        .collect();
    visible.sort();
    sel.visible = visible;
    sel
}

/// 2:1 restriction: no two edge-adjacent leaves differ by more than one level. Splits the coarser
/// leaf until stable.
pub fn balance(sel: &mut Selection) {
    let mut work: Vec<Key> = sel.leaves.iter().copied().collect();
    work.sort();
    while let Some(k) = work.pop() {
        if !sel.leaves.contains(&k) || k.level < 2 {
            continue;
        }
        for e in 0..4 {
            let n = k.neighbour(e);
            if let Some(c) = sel.covering(&n)
                && c.level + 1 < k.level
            {
                sel.leaves.remove(&c);
                for ch in c.children() {
                    sel.leaves.insert(ch);
                    work.push(ch);
                }
                sel.balance_splits += 1;
                work.push(k);
                break;
            }
        }
    }
}

/// Stitching mask of leaf `k`: bit e = the neighbour across edge e is one level coarser.
pub fn mask(sel: &Selection, k: &Key) -> u8 {
    let mut m = 0u8;
    for e in 0..4 {
        if let Some(c) = sel.covering(&k.neighbour(e))
            && c.level < k.level
        {
            m |= 1 << e;
        }
    }
    m
}

/// Check the 2:1 property over all leaves; returns the worst level difference found.
pub fn max_level_jump(sel: &Selection) -> u8 {
    let mut worst = 0;
    for k in &sel.leaves {
        for e in 0..4 {
            if let Some(c) = sel.covering(&k.neighbour(e)) {
                worst = worst.max(k.level.saturating_sub(c.level));
            }
        }
    }
    worst
}

/// Leaf counts per level (for stats).
pub fn level_histogram(keys: impl Iterator<Item = Key>) -> HashMap<u8, u32> {
    let mut h = HashMap::new();
    for k in keys {
        *h.entry(k.level).or_insert(0) += 1;
    }
    h
}
