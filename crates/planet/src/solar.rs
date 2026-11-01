//! Solar system (round 1): ephemeris, body data, orientation and the nested frames.
//!
//! **Positions.** Planets from the JPL "Approximate Positions of the Planets" Keplerian elements
//! (E. M. Standish, Table 1, valid 1800-2050; heliocentric, mean ecliptic and equinox of J2000),
//! the Earth from the Earth-Moon barycentre minus the Moon's share, the Moon from the low-precision
//! series of the Astronomical Almanac (section D), precessed to J2000. Checked against JPL Horizons
//! (DE441) at the EPIC epoch in the tests: planets to <= 0.12 deg heliocentric (Saturn, the worst; the
//! inner planets to ~0.01 deg), the Moon to ~0.3 deg geocentric.
//!
//! **Orientation.** IAU WGCCRE (2015) pole (RA, Dec) and prime-meridian angle W for the Sun, the
//! planets and the Moon (secular terms; the small periodic terms are left out). The Earth uses
//! IAU 1976 precession + GMST (IAU 1982), the same ECEF the planet renderer uses (Z = north pole,
//! X = 0 deg E); nutation (<= 20") and polar motion are left out.
//!
//! **Frames.** `galaxy > system ("sol", heliocentric, ecliptic J2000 axes, X = equinox, Z = ecliptic
//! north) > one planet frame per body`. The system frame's documented extent is 1e13 m (66 AU,
//! past Neptune's aphelion) at ~1 mm precision; every camera in the solar-system shots lives there and
//! every body-minus-camera offset is formed in f64 before it is rounded to f32 for the GPU. Planet
//! frames are translation-only (as [`coords::FrameTree`] is); the body-fixed rotation is
//! [`Body::fixed_from_ecliptic`].
use coords::{FrameId, FrameTree, WorldPos};
use glam::{DMat3, DVec3};

use crate::descent::AU_M;

/// Seconds TT - UTC in 2026 (TAI - UTC = 37 s, TT - TAI = 32.184 s).
pub const TT_MINUS_UTC_S: f64 = 69.184;
/// Mean obliquity of the ecliptic at J2000 (deg, IAU 1976).
pub const OBLIQUITY_J2000_DEG: f64 = 23.439_291_1;
/// Earth / Moon mass ratio (DE430).
pub const EARTH_MOON_MASS_RATIO: f64 = 81.300_568_9;
/// Solar radius (m, IAU 2015 nominal).
pub const SUN_RADIUS_M: f64 = 6.957e8;

/// An instant: Julian date in UTC (UT1 taken = UTC, |UT1-UTC| < 0.9 s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Epoch {
    pub jd_utc: f64,
}

impl Epoch {
    /// From a civil UTC date (proleptic Gregorian).
    pub fn utc(y: i32, mo: u32, d: u32, h: u32, mi: u32, s: f64) -> Self {
        let (y, m) = if mo <= 2 { (y - 1, mo + 12) } else { (y, mo) };
        let a = (y as f64 / 100.0).floor();
        let b = 2.0 - a + (a / 4.0).floor();
        let jd0 = (365.25 * (y as f64 + 4716.0)).floor() + (30.6001 * (m as f64 + 1.0)).floor() + d as f64 + b - 1524.5;
        Self { jd_utc: jd0 + (h as f64 + mi as f64 / 60.0 + s / 3600.0) / 24.0 }
    }
    /// The DSCOVR/EPIC image the Earth rounds validate against: 2026-07-15 11:07:35 UTC.
    pub fn epic() -> Self {
        Self::utc(2026, 7, 15, 11, 7, 35.0)
    }
    /// Days TT from J2000.0.
    pub fn d_tt(self) -> f64 {
        self.jd_utc + TT_MINUS_UTC_S / 86400.0 - 2_451_545.0
    }
    /// Julian centuries TT from J2000.0.
    pub fn t_tt(self) -> f64 {
        self.d_tt() / 36525.0
    }
}

/// The bodies rendered in round 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BodyId {
    Sun,
    Mercury,
    Venus,
    Earth,
    Moon,
    Mars,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
}

impl BodyId {
    pub const ALL: [BodyId; 10] = [
        BodyId::Sun,
        BodyId::Mercury,
        BodyId::Venus,
        BodyId::Earth,
        BodyId::Moon,
        BodyId::Mars,
        BodyId::Jupiter,
        BodyId::Saturn,
        BodyId::Uranus,
        BodyId::Neptune,
    ];
    pub fn name(self) -> &'static str {
        match self {
            BodyId::Sun => "sun",
            BodyId::Mercury => "mercury",
            BodyId::Venus => "venus",
            BodyId::Earth => "earth",
            BodyId::Moon => "moon",
            BodyId::Mars => "mars",
            BodyId::Jupiter => "jupiter",
            BodyId::Saturn => "saturn",
            BodyId::Uranus => "uranus",
            BodyId::Neptune => "neptune",
        }
    }
}

/// How a body's surface scatters light (the renderer's shading model).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    /// Emissive photosphere with limb darkening.
    Star,
    /// Regolith: lunar-Lambert (Lommel-Seeliger + Lambert, McEwen 1991).
    Regolith,
    /// Thin-atmosphere rock (Mars): regolith + a thin dust-haze limb.
    Dusty,
    /// Cloud deck under a thick haze (Venus): Minnaert + a bright forward-scattering limb.
    Cloud,
    /// Gas giant: Minnaert limb darkening.
    Giant,
    /// The Earth far LOD (ocean / land / cloud maps + a Rayleigh limb); near, the planet renderer.
    Earth,
}

/// Physical data of one body.
#[derive(Clone, Copy, Debug)]
pub struct Body {
    pub id: BodyId,
    /// Equatorial and polar radius (m, IAU 2015; 1-bar level for the giants).
    pub r_eq_m: f64,
    pub r_pol_m: f64,
    /// Geometric albedo (V).
    pub geometric_albedo: f64,
    pub surface: Surface,
    /// Pole (ICRF RA, Dec, deg) as `a + b T` (T Julian centuries TT) and the prime meridian
    /// `W = w0 + w1 d` (d days TT).
    pub pole_ra: (f64, f64),
    pub pole_dec: (f64, f64),
    pub w: (f64, f64),
    /// Minnaert exponent k (giants / cloud) or the lunar-Lambert L (regolith).
    pub shade_k: f64,
    /// Ring inner / outer radius (m), 0 = none.
    pub ring_m: (f64, f64),
}

/// The body table. Sources: IAU WGCCRE 2015 (Archinal et al. 2018) radii and rotation,
/// Mallama et al. 2017 geometric albedos.
pub fn body(id: BodyId) -> Body {
    let b = |r_eq: f64, r_pol: f64, p: f64, s: Surface, ra: (f64, f64), dec: (f64, f64), w: (f64, f64), k: f64| Body {
        id,
        r_eq_m: r_eq * 1e3,
        r_pol_m: r_pol * 1e3,
        geometric_albedo: p,
        surface: s,
        pole_ra: ra,
        pole_dec: dec,
        w,
        shade_k: k,
        ring_m: (0.0, 0.0),
    };
    match id {
        BodyId::Sun => b(695_700.0, 695_700.0, 0.0, Surface::Star, (286.13, 0.0), (63.87, 0.0), (84.176, 14.184_4), 0.0),
        BodyId::Mercury => b(2_440.53, 2_438.26, 0.142, Surface::Regolith, (281.0103, -0.0328), (61.4155, -0.0049), (329.5988, 6.138_510_8), 0.8),
        BodyId::Venus => b(6_051.8, 6_051.8, 0.689, Surface::Cloud, (272.76, 0.0), (67.16, 0.0), (160.20, -1.481_368_8), 0.75),
        BodyId::Earth => b(6_378.137, 6_356.752, 0.434, Surface::Earth, (0.0, -0.641), (90.0, -0.557), (190.147, 360.985_623_5), 1.0),
        BodyId::Moon => b(1_737.4, 1_737.4, 0.12, Surface::Regolith, (269.9949, 0.0031), (66.5392, 0.0130), (38.3213, 13.176_358_15), 0.9),
        BodyId::Mars => b(3_396.19, 3_376.20, 0.170, Surface::Dusty, (317.269_202, -0.109_275_47), (54.432_516, -0.058_271_05), (176.049_863, 350.891_982_443_297), 0.6),
        BodyId::Jupiter => b(71_492.0, 66_854.0, 0.538, Surface::Giant, (268.056_595, -0.006_499), (64.495_303, 0.002_413), (284.95, 870.536), 0.88),
        BodyId::Saturn => Body {
            // the ring profile's span, calibrated on its features (B inner edge 92,000 km, Cassini
            // division 117,580 km, Encke gap 133,590 km, A outer edge 136,775 km, F ring 140,180 km):
            // 69,700 km (D ring) to 141,020 km, ~34.8 km per texel of the 2048-texel profile
            ring_m: (69_700.0e3, 141_020.0e3),
            ..b(60_268.0, 54_364.0, 0.499, Surface::Giant, (40.589, -0.036), (83.537, -0.004), (38.90, 810.793_902_4), 0.9)
        },
        BodyId::Uranus => b(25_559.0, 24_973.0, 0.488, Surface::Giant, (257.311, 0.0), (-15.175, 0.0), (203.81, -501.160_092_8), 0.9),
        BodyId::Neptune => b(24_764.0, 24_341.0, 0.442, Surface::Giant, (299.36, 0.0), (43.46, 0.0), (249.978, 541.139_775_7), 0.9),
    }
}

/// Standish Table 1 elements: (a AU, e, I, L, long. perihelion, long. node) and rates per century.
type Elements = ([f64; 6], [f64; 6]);
fn elements(id: BodyId) -> Option<Elements> {
    Some(match id {
        BodyId::Mercury => (
            [0.387_099_27, 0.205_635_93, 7.004_979_02, 252.250_323_50, 77.457_796_28, 48.330_765_93],
            [0.000_000_37, 0.000_019_06, -0.005_947_49, 149_472.674_111_75, 0.160_476_89, -0.125_340_81],
        ),
        BodyId::Venus => (
            [0.723_335_66, 0.006_776_72, 3.394_676_05, 181.979_099_50, 131.602_467_18, 76.679_842_55],
            [0.000_003_90, -0.000_041_07, -0.000_788_90, 58_517.815_387_29, 0.002_683_29, -0.277_694_18],
        ),
        // the Earth-Moon barycentre
        BodyId::Earth => (
            [1.000_002_61, 0.016_711_23, -0.000_015_31, 100.464_571_66, 102.937_681_93, 0.0],
            [0.000_005_62, -0.000_043_92, -0.012_946_68, 35_999.372_449_81, 0.323_273_64, 0.0],
        ),
        BodyId::Mars => (
            [1.523_710_34, 0.093_394_10, 1.849_691_42, -4.553_432_05, -23.943_629_59, 49.559_538_91],
            [0.000_018_47, 0.000_078_82, -0.008_131_31, 19_140.302_684_99, 0.444_410_88, -0.292_573_43],
        ),
        BodyId::Jupiter => (
            [5.202_887_00, 0.048_386_24, 1.304_396_95, 34.396_440_51, 14.728_479_83, 100.473_909_09],
            [-0.000_116_07, -0.000_132_53, -0.001_837_14, 3_034.746_127_75, 0.212_526_68, 0.204_691_06],
        ),
        BodyId::Saturn => (
            [9.536_675_94, 0.053_861_79, 2.485_991_87, 49.954_244_23, 92.598_878_31, 113.662_424_48],
            [-0.001_250_60, -0.000_509_91, 0.001_936_09, 1_222.493_622_01, -0.418_972_16, -0.288_677_94],
        ),
        BodyId::Uranus => (
            [19.189_164_64, 0.047_257_44, 0.772_637_83, 313.238_104_51, 170.954_276_30, 74.016_925_03],
            [-0.001_961_76, -0.000_043_97, -0.002_429_39, 428.482_027_85, 0.408_052_81, 0.042_405_89],
        ),
        BodyId::Neptune => (
            [30.069_922_76, 0.008_590_48, 1.770_043_47, -55.120_029_69, 44.964_762_27, 131.784_225_74],
            [0.000_262_91, 0.000_051_05, 0.000_353_72, 218.459_453_25, -0.322_414_64, -0.005_086_64],
        ),
        _ => return None,
    })
}

/// Elements at `t` centuries TT: (a AU, e, I rad, mean anomaly rad, arg. perihelion rad, node rad).
fn elements_at(el: &Elements, t: f64) -> [f64; 6] {
    let e: [f64; 6] = std::array::from_fn(|i| el.0[i] + el.1[i] * t);
    let m = (e[3] - e[4]).to_radians().rem_euclid(std::f64::consts::TAU);
    let m = if m > std::f64::consts::PI { m - std::f64::consts::TAU } else { m };
    [e[0], e[1], e[2].to_radians(), m, (e[4] - e[5]).to_radians(), e[5].to_radians()]
}

/// Ecliptic position (m) at eccentric anomaly `ea` on the orbit of elements `e` (see [`elements_at`]).
fn orbit_point(e: &[f64; 6], ea: f64) -> DVec3 {
    let (a, ecc, inc, _, w, node) = (e[0], e[1], e[2], e[3], e[4], e[5]);
    let (xp, yp) = (a * (ea.cos() - ecc), a * (1.0 - ecc * ecc).sqrt() * ea.sin());
    let (cw, sw, co, so, ci, si) = (w.cos(), w.sin(), node.cos(), node.sin(), inc.cos(), inc.sin());
    DVec3::new(
        (cw * co - sw * so * ci) * xp + (-sw * co - cw * so * ci) * yp,
        (cw * so + sw * co * ci) * xp + (-sw * so + cw * co * ci) * yp,
        (sw * si) * xp + (cw * si) * yp,
    ) * AU_M
}

/// Heliocentric position (m, ecliptic J2000) from Keplerian elements at `t` centuries TT.
fn kepler(el: &Elements, t: f64) -> DVec3 {
    let e = elements_at(el, t);
    let (ecc, m) = (e[1], e[3]);
    let mut ea = m + ecc * m.sin();
    for _ in 0..30 {
        let d = (ea - ecc * ea.sin() - m) / (1.0 - ecc * ea.cos());
        ea -= d;
        if d.abs() < 1e-15 {
            break;
        }
    }
    orbit_point(&e, ea)
}

/// Geocentric Moon (m, ecliptic J2000), Astronomical Almanac low-precision series.
pub fn moon_geocentric(ep: Epoch) -> DVec3 {
    let t = ep.t_tt();
    let s = |a: f64, b: f64| (a + b * t).to_radians().sin();
    let c = |a: f64, b: f64| (a + b * t).to_radians().cos();
    let lon = 218.32 + 481_267.881 * t + 6.29 * s(135.0, 477_198.87) - 1.27 * s(259.3, -413_335.36) + 0.66 * s(235.7, 890_534.22) + 0.21 * s(269.9, 954_397.74)
        - 0.19 * s(357.5, 35_999.05)
        - 0.11 * s(186.5, 966_404.03);
    let lat = 5.13 * s(93.3, 483_202.02) + 0.28 * s(228.2, 960_400.89) - 0.28 * s(318.3, 6_003.15) - 0.17 * s(217.6, -407_332.21);
    let par = 0.9508 + 0.0518 * c(135.0, 477_198.87) + 0.0095 * c(259.3, -413_335.36) + 0.0078 * c(235.7, 890_534.22) + 0.0028 * c(269.9, 954_397.74);
    let r = 6_378_140.0 / par.to_radians().sin();
    // of date -> J2000: general precession in longitude
    let lon = (lon - 1.396_971_3 * t).to_radians();
    let lat = lat.to_radians();
    DVec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()) * r
}

/// Heliocentric position (m, ecliptic J2000) of a body.
pub fn position(id: BodyId, ep: Epoch) -> DVec3 {
    let t = ep.t_tt();
    match id {
        BodyId::Sun => DVec3::ZERO,
        BodyId::Earth => {
            let emb = kepler(&elements(BodyId::Earth).unwrap(), t);
            emb - moon_geocentric(ep) / (1.0 + EARTH_MOON_MASS_RATIO)
        }
        BodyId::Moon => {
            let emb = kepler(&elements(BodyId::Earth).unwrap(), t);
            let m = moon_geocentric(ep);
            emb + m * (EARTH_MOON_MASS_RATIO / (1.0 + EARTH_MOON_MASS_RATIO))
        }
        _ => kepler(&elements(id).unwrap(), t),
    }
}

/// The osculating orbit ellipse at the epoch (heliocentric ecliptic J2000, m), `n` points evenly in
/// eccentric anomaly; empty for the Sun and the Moon. (The Earth's is the Earth-Moon barycentre's.)
pub fn orbit_polyline(id: BodyId, ep: Epoch, n: usize) -> Vec<DVec3> {
    match elements(id) {
        Some(el) => {
            let e = elements_at(&el, ep.t_tt());
            (0..n).map(|i| orbit_point(&e, std::f64::consts::TAU * i as f64 / n as f64)).collect()
        }
        None => Vec::new(),
    }
}

/// ICRF equatorial -> ecliptic J2000.
pub fn ecliptic_from_equatorial() -> DMat3 {
    DMat3::from_rotation_x(-OBLIQUITY_J2000_DEG.to_radians())
}

/// Unit vector of (RA, Dec) in deg.
fn radec(ra: f64, dec: f64) -> DVec3 {
    let (a, d) = (ra.to_radians(), dec.to_radians());
    DVec3::new(d.cos() * a.cos(), d.cos() * a.sin(), d.sin())
}

/// IAU 1976 precession matrix: J2000 mean equator -> mean equator of date.
pub fn precession(ep: Epoch) -> DMat3 {
    let t = ep.t_tt();
    let as_ = |x: f64| (x / 3600.0).to_radians();
    let zeta = as_(2306.2181 * t + 0.30188 * t * t + 0.017998 * t * t * t);
    let z = as_(2306.2181 * t + 1.09468 * t * t + 0.018203 * t * t * t);
    let th = as_(2004.3109 * t - 0.42665 * t * t - 0.041833 * t * t * t);
    DMat3::from_rotation_z(z) * DMat3::from_rotation_y(-th) * DMat3::from_rotation_z(zeta)
}

/// Greenwich mean sidereal time (deg, IAU 1982, UT1 = UTC).
pub fn gmst_deg(ep: Epoch) -> f64 {
    let d = ep.jd_utc - 2_451_545.0;
    let t = d / 36525.0;
    (280.460_618_37 + 360.985_647_366_29 * d + 0.000_387_933 * t * t).rem_euclid(360.0)
}

impl Body {
    /// Rotation taking ecliptic-J2000 vectors to this body's fixed frame (Z = north pole, X = prime
    /// meridian). The Earth's is the planet renderer's ECEF.
    pub fn fixed_from_ecliptic(&self, ep: Epoch) -> DMat3 {
        let eq_from_ecl = ecliptic_from_equatorial().transpose();
        if self.id == BodyId::Earth {
            return DMat3::from_rotation_z(-gmst_deg(ep).to_radians()) * precession(ep) * eq_from_ecl;
        }
        let (t, d) = (ep.t_tt(), ep.d_tt());
        let ra = self.pole_ra.0 + self.pole_ra.1 * t;
        let dec = self.pole_dec.0 + self.pole_dec.1 * t;
        let w = (self.w.0 + self.w.1 * d).rem_euclid(360.0);
        // IAU: body frame = Rz(W) Rx(90 - dec) Rz(90 + ra) applied to ICRF
        let m = DMat3::from_rotation_z(-w.to_radians()) * DMat3::from_rotation_x(-(90.0 - dec).to_radians()) * DMat3::from_rotation_z(-(90.0 + ra).to_radians());
        m * eq_from_ecl
    }

    /// North pole (ecliptic J2000 unit vector).
    pub fn pole_ecliptic(&self, ep: Epoch) -> DVec3 {
        self.fixed_from_ecliptic(ep).transpose() * DVec3::Z
    }

    /// Pole direction in ICRF from the table (for tests).
    pub fn pole_icrf(&self, ep: Epoch) -> DVec3 {
        let t = ep.t_tt();
        radec(self.pole_ra.0 + self.pole_ra.1 * t, self.pole_dec.0 + self.pole_dec.1 * t)
    }
}

/// Subsolar point on the Earth (lat, lon deg) at the epoch.
pub fn earth_subsolar(ep: Epoch) -> (f64, f64) {
    let e = body(BodyId::Earth);
    let s = e.fixed_from_ecliptic(ep) * (-position(BodyId::Earth, ep)).normalize();
    (s.z.asin().to_degrees(), s.y.atan2(s.x).to_degrees())
}

/// Ecliptic-J2000 direction of an ECEF direction at the epoch (and back).
pub fn ecliptic_from_ecef(ep: Epoch, v: DVec3) -> DVec3 {
    body(BodyId::Earth).fixed_from_ecliptic(ep).transpose() * v
}
pub fn ecef_from_ecliptic(ep: Epoch, v: DVec3) -> DVec3 {
    body(BodyId::Earth).fixed_from_ecliptic(ep) * v
}

/// The solar system at one epoch: nested frames + per-body state.
#[derive(Clone, Debug)]
pub struct System {
    pub epoch: Epoch,
    pub tree: FrameTree,
    pub galaxy: FrameId,
    pub system: FrameId,
    /// (body, its planet frame, heliocentric position m, ecliptic -> body-fixed rotation)
    pub bodies: Vec<(Body, FrameId, DVec3, DMat3)>,
}

impl System {
    pub fn new(ep: Epoch) -> Self {
        let (mut tree, galaxy) = FrameTree::new("milky way");
        // the Sun ~8.0 kpc from the galactic centre (the same offset the descent uses)
        let system = tree.add(galaxy, "sol", DVec3::new(2.46e20, 0.0, 6.0e17), 1.0e13).expect("system frame");
        let bodies = BodyId::ALL
            .iter()
            .map(|&id| {
                let b = body(id);
                let p = position(id, ep);
                let f = tree.add(system, id.name(), p, 1.0e7).expect("planet frame");
                (b, f, p, b.fixed_from_ecliptic(ep))
            })
            .collect();
        Self { epoch: ep, tree, galaxy, system, bodies }
    }

    pub fn get(&self, id: BodyId) -> &(Body, FrameId, DVec3, DMat3) {
        self.bodies.iter().find(|b| b.0.id == id).expect("body")
    }

    /// A system-frame position.
    pub fn at(&self, p: DVec3) -> WorldPos {
        WorldPos { frame: self.system, p }
    }

    /// Camera (system frame) -> the Earth's ECEF position (through the frame tree, f64).
    pub fn to_earth_ecef(&self, cam: DVec3) -> DVec3 {
        let (_, f, _, r) = *self.get(BodyId::Earth);
        r * self.tree.convert(self.at(cam), f).p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// JPL Horizons (DE441), 2026-07-15 11:07:35 UT, heliocentric (500@10) ecliptic J2000, km.
    const HORIZONS: [(BodyId, [f64; 3]); 9] = [
        (BodyId::Mercury, [3.037_427_093_644_822e7, -5.784_803_228_383_894e7, -7.513_384_104_503_263e6]),
        (BodyId::Venus, [-6.875_970_197_368_039e7, -8.364_916_000_353_213e7, 2.818_131_475_891_735e6]),
        (BodyId::Earth, [5.850_557_548_987_23e7, -1.403_516_239_939_145e8, 7.923_136_327_385_902e3]),
        (BodyId::Moon, [5.828_531_566_325_65e7, -1.400_631_912_747_495e8, 2.049_050_717_344_135e4]),
        (BodyId::Mars, [1.569_661_014_301_233e8, 1.516_718_151_830_257e8, -6.703_969_521_040_32e5]),
        (BodyId::Jupiter, [-4.521_729_911_314_537e8, 6.478_094_081_978_796e8, 7.425_735_945_254_713e6]),
        (BodyId::Saturn, [1.399_399_600_660_636e9, 2.006_353_584_014_715e8, -5.919_854_054_363_149e7]),
        (BodyId::Uranus, [1.376_784_420_612_789e9, 2.564_217_149_175_563e9, -8.328_498_101_480_484e6]),
        (BodyId::Neptune, [4.465_475_463_122_034e9, 1.698_987_009_171_883e8, -1.064_035_898_283_15e8]),
    ];

    #[test]
    fn epoch_jd() {
        assert!((Epoch::epic().jd_utc - 2_461_236.963_599_537).abs() < 1e-8);
        assert!((Epoch::utc(2000, 1, 1, 12, 0, 0.0).jd_utc - 2_451_545.0).abs() < 1e-9);
    }

    #[test]
    fn planets_match_horizons() {
        let ep = Epoch::epic();
        for (id, km) in HORIZONS {
            let h = DVec3::from_array(km) * 1e3;
            let p = position(id, ep);
            let ang = p.angle_between(h).to_degrees();
            let dr = (p.length() - h.length()).abs() / h.length();
            eprintln!("{:8} angle {ang:.4} deg  dr {dr:.2e}", id.name());
            let lim = match id {
                BodyId::Saturn | BodyId::Jupiter | BodyId::Uranus | BodyId::Neptune => 0.15,
                _ => 0.03,
            };
            assert!(ang < lim, "{id:?} off by {ang} deg");
            assert!(dr < 2e-3, "{id:?} distance off by {dr}");
        }
        // the Moon around the Earth
        let ep = Epoch::epic();
        let m = position(BodyId::Moon, ep) - position(BodyId::Earth, ep);
        let h = (DVec3::from_array(HORIZONS[3].1) - DVec3::from_array(HORIZONS[2].1)) * 1e3;
        let ang = m.angle_between(h).to_degrees();
        eprintln!("moon geocentric angle {ang:.3} deg, dr {:.2e}", (m.length() - h.length()).abs() / h.length());
        assert!(ang < 0.5);
        assert!((m.length() - h.length()).abs() / h.length() < 5e-3);
    }

    #[test]
    fn subsolar_matches_epic() {
        let (lat, lon) = earth_subsolar(Epoch::epic());
        eprintln!("subsolar {lat:.3} N {lon:.3} E");
        // independent: the NOAA / Astronomical Almanac low-precision Sun + GMST gives 21.4757 N,
        // 14.6072 E for this instant
        assert!((lat - 21.4757).abs() < 0.01, "lat {lat}");
        assert!((lon - 14.6072).abs() < 0.02, "lon {lon}");
        // the EPIC metadata of the 2026-07-15 11:07:35 image says 21.537 N, 14.219 E: 0.06 deg / 0.39 deg
        // (~95 s of Earth rotation) away; the metadata's own model/timing, not ours (both
        // computations above agree to 0.006 deg)
        assert!((lat - 21.537).abs() < 0.1 && (lon - 14.219).abs() < 0.5);
    }

    #[test]
    fn poles_and_rotations() {
        let ep = Epoch::epic();
        for id in BodyId::ALL {
            let b = body(id);
            let m = b.fixed_from_ecliptic(ep);
            // orthonormal, right-handed
            assert!((m.determinant() - 1.0).abs() < 1e-12);
            assert!((m * m.transpose() - DMat3::IDENTITY).to_cols_array().iter().all(|x| x.abs() < 1e-12));
            if id != BodyId::Earth {
                let pole = ecliptic_from_equatorial() * b.pole_icrf(ep);
                assert!(b.pole_ecliptic(ep).angle_between(pole) < 1e-9, "{id:?}");
            }
        }
        // obliquities (the IAU north pole is the one north of the invariable plane, so Uranus's
        // 97.77 deg reads as 82.23): Earth 23.44, Mars 25.19, Jupiter 3.13, Saturn 26.73, Uranus 97.77, Neptune 28.32
        // (pole vs the planet's own orbit normal, ~ the ecliptic normal to within the inclination)
        for (id, ob, tol) in [(BodyId::Earth, 23.44, 0.05), (BodyId::Mars, 25.19, 1.9), (BodyId::Jupiter, 3.13, 1.4), (BodyId::Saturn, 26.73, 2.5), (BodyId::Uranus, 180.0 - 97.77, 0.8), (BodyId::Neptune, 28.32, 1.8)] {
            let a = body(id).pole_ecliptic(ep).angle_between(DVec3::Z).to_degrees();
            assert!((a - ob).abs() < tol, "{id:?} tilt {a}");
        }
    }

    #[test]
    fn frames_nest_and_keep_precision() {
        let s = System::new(Epoch::epic());
        assert_eq!(s.tree.len(), 2 + BodyId::ALL.len());
        let (_, ef, ep_, _) = *s.get(BodyId::Earth);
        // a camera 1.5e6 km sunward of the Earth, expressed in the system frame, round-trips through
        // the Earth frame to sub-millimetre
        let cam = ep_ - ep_.normalize() * 1.5e9;
        let back = s.tree.convert(s.tree.convert(s.at(cam), ef), s.system).p;
        assert!((back - cam).length() < 1e-3);
        // Neptune's frame origin is inside the system frame's documented extent
        assert!(s.get(BodyId::Neptune).2.abs().max_element() < coords::FrameLevel::System.max_extent_m());
        // the ECEF of a sunward camera points at the subsolar point
        let e = s.to_earth_ecef(cam).normalize();
        let (lat, lon) = earth_subsolar(s.epoch);
        let sub = crate::cube::latlon_dir(lat, lon);
        assert!(e.angle_between(sub) < 1e-9);
    }

    #[test]
    fn orbit_closes() {
        let ep = Epoch::epic();
        for id in [BodyId::Mercury, BodyId::Earth, BodyId::Neptune] {
            let o = orbit_polyline(id, ep, 256);
            assert_eq!(o.len(), 256);
            let p = position(id, ep);
            let d = o.iter().map(|q| (*q - p).length()).fold(f64::INFINITY, f64::min);
            // the body sits on (near) its drawn orbit
            assert!(d / p.length() < 0.03, "{id:?} {}", d / p.length());
        }
    }
}
