//! Independent observed positional diagnostics, never stellar luminosities or a merged census.
//! Raw fields, native epochs, uncertainty, terms and query selection remain in external sidecars.
use crate::universe::{Camera, MAX_STARS, SUPPORT};
use glam::DVec3;
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashSet},
    io::{self, Read},
};

pub const MAX_RECORDS: usize = 100_000;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Cns5,
    Desi,
}
#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub id: u64,
    pub cross_id: Option<u64>,
    pub pos: DVec3,
    /// Native per-source coordinate epoch; None is unknown, not J2016.
    pub epoch: Option<f64>,
    /// Measured parallax (mas) for CNS5; spectroscopic redshift for DESI.
    pub quantity: f64,
    pub error: f64,
    /// Direct corrected Gaia G only; never a Hipparcos/IR conversion.
    pub direct_g: Option<f64>,
}
pub struct Catalogue {
    pub kind: Kind,
    pub raw_rows: usize,
    pub points: Vec<Point>,
    pub raw_sha256: [u8; 32],
}
fn invalid() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid OBSdiag1 snapshot")
}
fn optional(x: f64) -> io::Result<Option<f64>> {
    if x.is_nan() {
        Ok(None)
    } else if x.is_finite() {
        Ok(Some(x))
    } else {
        Err(invalid())
    }
}
/// Read at most 100k CPU points; GPU cap is separate. Exact source IDs never pass through f64.
pub fn read(r: &mut impl Read) -> io::Result<Catalogue> {
    let mut h = [0u8; 64];
    r.read_exact(&mut h)?;
    if &h[..8] != b"OBSdiag1" {
        return Err(invalid());
    }
    let u = |i| u64::from_le_bytes(h[i..i + 8].try_into().unwrap());
    let kind = match u(8) {
        1 => Kind::Cns5,
        2 => Kind::Desi,
        _ => return Err(invalid()),
    };
    let (raw, count) = (u(16), u(24));
    if raw == 0 || raw > MAX_RECORDS as u64 || count > raw {
        return Err(invalid());
    }
    let mut points = Vec::with_capacity(count as usize);
    let mut ids = HashSet::with_capacity(count as usize);
    for _ in 0..count {
        let mut b = [0u8; 72];
        r.read_exact(&mut b)?;
        let id = u64::from_le_bytes(b[..8].try_into().unwrap());
        let cross = u64::from_le_bytes(b[8..16].try_into().unwrap());
        let f = |i| f64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        let p = Point {
            id,
            cross_id: (cross != 0).then_some(cross),
            pos: DVec3::new(f(16), f(24), f(32)),
            epoch: optional(f(40))?,
            quantity: f(48),
            error: f(56),
            direct_g: optional(f(64))?,
        };
        if !ids.insert(id)
            || !p.pos.is_finite()
            || !(1.0..1e28).contains(&p.pos.length())
            || !p.quantity.is_finite()
            || !p.error.is_finite()
            || p.error < 0.
            || p.epoch.is_some_and(|e| !(1800.0..=2200.0).contains(&e))
            || p.direct_g.is_some_and(|g| !(-30.0..60.0).contains(&g))
        {
            return Err(invalid());
        }
        match kind {
            Kind::Cns5 if p.epoch.is_none() || p.quantity <= 0. || p.error <= 0. || p.error / p.quantity > 0.1 => {
                return Err(invalid());
            }
            Kind::Desi
                if id == 0 || !(0.01..0.5).contains(&p.quantity) || p.cross_id.is_some() || p.direct_g.is_some() =>
            {
                return Err(invalid());
            }
            _ => {}
        }
        points.push(p);
    }
    let mut extra = [0u8; 1];
    if r.read(&mut extra)? != 0 {
        return Err(invalid());
    }
    Ok(Catalogue { kind, raw_rows: raw as usize, points, raw_sha256: h[32..64].try_into().unwrap() })
}
/// Equal diagnostic marker. No luminosity, inverse-square factor, RGB or apparent flux.
#[derive(Clone, Copy, Debug)]
pub struct Marker {
    pub id: u64,
    pub center: [f64; 2],
}
struct Ranked {
    distance: f64,
    marker: Marker,
}
impl PartialEq for Ranked {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Ranked {}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Ranked {
    fn cmp(&self, o: &Self) -> Ordering {
        self.distance.total_cmp(&o.distance).then(self.marker.id.cmp(&o.marker.id))
    }
}
/// Scan ALL points; bounded nearest-visible heap, stable ID tie-break. Zero limit still counts visibility.
/// Native positions are not propagated or falsely relabelled to a common epoch.
pub fn select_markers(points: &[Point], cam: &Camera, limit: usize) -> (Vec<Marker>, usize) {
    let limit = limit.min(MAX_STARS);
    let mut heap: BinaryHeap<Ranked> = BinaryHeap::with_capacity(limit);
    let mut visible = 0;
    if !cam.origin.is_finite()
        || !cam.pos.is_finite()
        || !cam.forward.is_finite()
        || !cam.up.is_finite()
        || !cam.right.is_finite()
        || !cam.fov.is_finite()
        || !(0.01..3.).contains(&cam.fov)
        || cam.width == 0
        || cam.height == 0
        || cam.width > 4096
        || cam.height > 4096
    {
        return (Vec::new(), 0);
    }
    let focal = cam.height as f64 / (2. * (cam.fov * 0.5).tan());
    for p in points {
        let delta = (p.pos - cam.origin) - cam.pos;
        let z = delta.dot(cam.forward);
        let distance = delta.length_squared();
        if !delta.is_finite() || !distance.is_finite() || distance <= 0. || z <= 0. {
            continue;
        }
        let x = cam.width as f64 * 0.5 + delta.dot(cam.right) * focal / z;
        let y = cam.height as f64 * 0.5 - delta.dot(cam.up) * focal / z;
        if !x.is_finite()
            || !y.is_finite()
            || x < -SUPPORT
            || y < -SUPPORT
            || x > cam.width as f64 + SUPPORT
            || y > cam.height as f64 + SUPPORT
        {
            continue;
        }
        visible += 1;
        if limit == 0 {
            continue;
        }
        let ranked = Ranked { distance, marker: Marker { id: p.id, center: [x, y] } };
        if heap.len() < limit {
            heap.push(ranked);
        } else if heap.peek().is_some_and(|worst| ranked.cmp(worst) == Ordering::Less) {
            heap.pop();
            heap.push(ranked);
        }
    }
    (heap.into_sorted_vec().into_iter().map(|r| r.marker).collect(), visible)
}
