//! Cube faces, the integer lattice and the face -> sphere map.
//!
//! Face `f` has an outward normal `n` and in-face axes `u`, `v` with `u x v = n` (counter-clockwise
//! in (u, v) is counter-clockwise seen from outside). Face coordinates are integers `U, V` in
//! `0..=M` with `M = N << LMAX`; `c = 2U - M` in `-M..=M` is the centred integer coordinate. The
//! cube point is `M n + cu u + cv v` (an integer 3-vector), so points on shared edges and corners
//! of two or three faces are the *same integers* whichever face computes them.
use glam::{DVec3, IVec3};

/// Quads per patch side.
pub const N: u32 = 32;
/// Deepest quadtree level supported by the lattice (1 quad = 2^-LMAX of a face patch side / N).
pub const LMAX: u32 = 22;
/// Face lattice size: `N << LMAX`.
pub const M: i64 = (N as i64) << LMAX;

/// (normal, u, v) as integer axis vectors; `u x v = n`.
pub const FACES: [(IVec3, IVec3, IVec3); 6] = [
    (IVec3::new(1, 0, 0), IVec3::new(0, 1, 0), IVec3::new(0, 0, 1)),
    (IVec3::new(-1, 0, 0), IVec3::new(0, 0, 1), IVec3::new(0, 1, 0)),
    (IVec3::new(0, 1, 0), IVec3::new(0, 0, 1), IVec3::new(1, 0, 0)),
    (IVec3::new(0, -1, 0), IVec3::new(1, 0, 0), IVec3::new(0, 0, 1)),
    (IVec3::new(0, 0, 1), IVec3::new(1, 0, 0), IVec3::new(0, 1, 0)),
    (IVec3::new(0, 0, -1), IVec3::new(0, 1, 0), IVec3::new(1, 0, 0)),
];

/// Integer cube point of face `f` at centred coordinates `(cu, cv)` (each in `-M..=M`).
pub fn cube_point(f: usize, cu: i64, cv: i64) -> [i64; 3] {
    let (n, u, v) = FACES[f];
    let c = |k: usize| M * n[k] as i64 + cu * u[k] as i64 + cv * v[k] as i64;
    [c(0), c(1), c(2)]
}

/// Odd, exact-at-the-ends tangent warp of a centred coordinate `c / M` in [-1, 1]: equal-angle
/// spacing (cell areas within 1.4x instead of 5.2x of the plain cube). `w(+-1) = +-1` exactly and
/// `w(-x) = -w(x)` bit-exactly, which makes shared face edges bit-identical.
pub fn warp(c: i64) -> f64 {
    if c.abs() >= M {
        return c.signum() as f64;
    }
    let a = (c.abs() as f64 / M as f64) * std::f64::consts::FRAC_PI_4;
    c.signum() as f64 * a.tan()
}

/// Unit sphere direction of an integer cube point: per world axis, the warped coordinate (the axis
/// at `+-M` stays `+-1`), normalised with the components summed in x, y, z order. Depends only on
/// the integer point, never on which face produced it.
pub fn point_dir(p: [i64; 3]) -> DVec3 {
    let w = DVec3::new(warp(p[0]), warp(p[1]), warp(p[2]));
    let l = (w.x * w.x + w.y * w.y + w.z * w.z).sqrt();
    w / l
}

/// Direction of face `f` lattice coordinates `(U, V)` in `0..=M`.
pub fn lattice_dir(f: usize, uu: i64, vv: i64) -> DVec3 {
    point_dir(cube_point(f, 2 * uu - M, 2 * vv - M))
}

/// Face (mesh convention, [`FACES`]) and centred face coordinates in [-1, 1] (before unwarping) of
/// a direction: the inverse of [`point_dir`] up to rounding.
pub fn dir_face(d: DVec3) -> (usize, f64, f64) {
    let a = d.abs();
    let (ax, s) = if a.x >= a.y && a.x >= a.z {
        (0, d.x)
    } else if a.y >= a.z {
        (1, d.y)
    } else {
        (2, d.z)
    };
    let f = ax * 2 + usize::from(s < 0.0);
    let (n, u, v) = FACES[f];
    let dn = d.dot(n.as_dvec3());
    let unwarp = |x: f64| x.atan() / std::f64::consts::FRAC_PI_4;
    (f, unwarp(d.dot(u.as_dvec3()) / dn), unwarp(d.dot(v.as_dvec3()) / dn))
}

/// Vulkan cube-map face (layer order +X, -X, +Y, -Y, +Z, -Z) and (s, t) in [0, 1] of a direction
/// (Vulkan spec 16.5.4, "Cube Map Face Selection"). Ties go to x, then y.
pub fn vk_cube_st(d: DVec3) -> (usize, f64, f64) {
    let a = d.abs();
    let (face, sc, tc, ma) = if a.x >= a.y && a.x >= a.z {
        if d.x >= 0.0 { (0, -d.z, -d.y, a.x) } else { (1, d.z, -d.y, a.x) }
    } else if a.y >= a.z {
        if d.y >= 0.0 { (2, d.x, d.z, a.y) } else { (3, d.x, -d.z, a.y) }
    } else if d.z >= 0.0 {
        (4, d.x, -d.y, a.z)
    } else {
        (5, -d.x, -d.y, a.z)
    };
    (face, 0.5 * (sc / ma + 1.0), 0.5 * (tc / ma + 1.0))
}

/// Inverse of [`vk_cube_st`]: the (unnormalised) direction of Vulkan cube face `face` at (s, t).
pub fn vk_cube_dir(face: usize, s: f64, t: f64) -> DVec3 {
    let (sc, tc) = (2.0 * s - 1.0, 2.0 * t - 1.0);
    match face {
        0 => DVec3::new(1.0, -tc, -sc),
        1 => DVec3::new(-1.0, -tc, sc),
        2 => DVec3::new(sc, 1.0, tc),
        3 => DVec3::new(sc, -1.0, -tc),
        4 => DVec3::new(sc, -tc, 1.0),
        _ => DVec3::new(-sc, -tc, -1.0),
    }
}

/// Planet frame: +Z = north pole, +X = (lat 0, lon 0), +Y = (lat 0, lon 90 E).
pub fn latlon_dir(lat_deg: f64, lon_deg: f64) -> DVec3 {
    let (la, lo) = (lat_deg.to_radians(), lon_deg.to_radians());
    DVec3::new(la.cos() * lo.cos(), la.cos() * lo.sin(), la.sin())
}

/// (lat, lon) in degrees of a direction.
pub fn dir_latlon(d: DVec3) -> (f64, f64) {
    let d = d.normalize();
    (d.z.clamp(-1.0, 1.0).asin().to_degrees(), d.y.atan2(d.x).to_degrees())
}
