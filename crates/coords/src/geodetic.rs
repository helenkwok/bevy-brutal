//! WGS84 geodetic coordinates, ECEF, and local East-North-Up (ENU) tangent frames.
//!
//! ECEF here matches the planet renderer (`crates/planet`): metres, origin at the Earth's centre,
//! **+Z = north pole, +X = 0° E on the equator, +Y = 90° E**. The planet renderer itself uses a
//! sphere; this module uses the WGS84 ellipsoid, which is what DEMs and OSM coordinates are
//! referenced to. The ellipsoid/sphere difference (up to ~21 km in radius) is a planet-scale
//! concern; inside one [`Enu`] frame everything is consistent.
//!
//! An [`Enu`] frame is the tangent plane at a geodetic origin: +X east, +Y north, +Z up along the
//! ellipsoid normal. It is a rotation + translation of ECEF, so it fits under a Planet frame in
//! [`crate::FrameTree`] once frames grow orientation; until then use [`Enu::origin_ecef`] as the
//! Local frame origin and [`Enu::to_ecef`] / [`Enu::from_ecef`] for directions.
use glam::{DMat3, DVec3};

/// WGS84 semi-major axis, metres.
pub const WGS84_A: f64 = 6_378_137.0;
/// WGS84 flattening.
pub const WGS84_F: f64 = 1.0 / 298.257_223_563;
/// WGS84 first eccentricity squared, `f(2-f)`.
pub const WGS84_E2: f64 = WGS84_F * (2.0 - WGS84_F);
/// WGS84 semi-minor axis, metres.
pub const WGS84_B: f64 = WGS84_A * (1.0 - WGS84_F);

/// Geodetic position on WGS84: latitude/longitude in degrees, height above the ellipsoid in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geodetic {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub h_m: f64,
}

impl Geodetic {
    pub const fn new(lat_deg: f64, lon_deg: f64, h_m: f64) -> Self {
        Self { lat_deg, lon_deg, h_m }
    }

    /// ECEF metres.
    pub fn to_ecef(self) -> DVec3 {
        let (sp, cp) = self.lat_deg.to_radians().sin_cos();
        let (sl, cl) = self.lon_deg.to_radians().sin_cos();
        let n = prime_vertical_radius(self.lat_deg);
        DVec3::new((n + self.h_m) * cp * cl, (n + self.h_m) * cp * sl, (n * (1.0 - WGS84_E2) + self.h_m) * sp)
    }

    /// Inverse of [`Geodetic::to_ecef`] (Bowring start + Newton refinement; sub-micrometre for
    /// points within a few thousand km of the surface). Longitude in `(-180, 180]`.
    pub fn from_ecef(p: DVec3) -> Self {
        let lon = p.y.atan2(p.x);
        let r = (p.x * p.x + p.y * p.y).sqrt();
        if r < 1e-9 {
            // On the polar axis.
            let lat = if p.z >= 0.0 { 90.0 } else { -90.0 };
            return Self { lat_deg: lat, lon_deg: 0.0, h_m: p.z.abs() - WGS84_B };
        }
        // Bowring's initial guess via the parametric latitude.
        let ep2 = WGS84_E2 / (1.0 - WGS84_E2);
        let beta = (WGS84_A * p.z).atan2(WGS84_B * r);
        let (sb, cb) = beta.sin_cos();
        let mut lat = (p.z + ep2 * WGS84_B * sb * sb * sb).atan2(r - WGS84_E2 * WGS84_A * cb * cb * cb);
        let mut h = 0.0;
        for _ in 0..4 {
            let (s, c) = lat.sin_cos();
            let n = WGS84_A / (1.0 - WGS84_E2 * s * s).sqrt();
            h = if c.abs() > 1e-3 { r / c - n } else { p.z / s - n * (1.0 - WGS84_E2) };
            lat = (p.z / r / (1.0 - WGS84_E2 * n / (n + h))).atan();
        }
        Self { lat_deg: lat.to_degrees(), lon_deg: lon.to_degrees(), h_m: h }
    }
}

/// Radius of curvature in the prime vertical (east-west), `N(φ)`, metres.
pub fn prime_vertical_radius(lat_deg: f64) -> f64 {
    let s = lat_deg.to_radians().sin();
    WGS84_A / (1.0 - WGS84_E2 * s * s).sqrt()
}

/// Meridional radius of curvature (north-south), `M(φ)`, metres.
pub fn meridional_radius(lat_deg: f64) -> f64 {
    let s = lat_deg.to_radians().sin();
    WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * s * s).powf(1.5)
}

/// Local East-North-Up tangent frame at a geodetic origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Enu {
    origin: Geodetic,
    origin_ecef: DVec3,
    /// Columns are the east, north and up unit vectors in ECEF.
    basis: DMat3,
}

impl Enu {
    pub fn new(origin: Geodetic) -> Self {
        let (sp, cp) = origin.lat_deg.to_radians().sin_cos();
        let (sl, cl) = origin.lon_deg.to_radians().sin_cos();
        let east = DVec3::new(-sl, cl, 0.0);
        let north = DVec3::new(-sp * cl, -sp * sl, cp);
        let up = DVec3::new(cp * cl, cp * sl, sp);
        Self { origin, origin_ecef: origin.to_ecef(), basis: DMat3::from_cols(east, north, up) }
    }

    pub fn origin(&self) -> Geodetic {
        self.origin
    }

    pub fn origin_ecef(&self) -> DVec3 {
        self.origin_ecef
    }

    /// East, north, up unit vectors in ECEF (the columns of the ENU→ECEF rotation).
    pub fn axes(&self) -> [DVec3; 3] {
        [self.basis.x_axis, self.basis.y_axis, self.basis.z_axis]
    }

    /// ENU metres → ECEF metres.
    pub fn to_ecef(&self, enu: DVec3) -> DVec3 {
        self.origin_ecef + self.basis * enu
    }

    /// ECEF metres → ENU metres.
    pub fn from_ecef(&self, ecef: DVec3) -> DVec3 {
        self.basis.transpose() * (ecef - self.origin_ecef)
    }

    pub fn geodetic_to_enu(&self, g: Geodetic) -> DVec3 {
        self.from_ecef(g.to_ecef())
    }

    pub fn enu_to_geodetic(&self, enu: DVec3) -> Geodetic {
        Geodetic::from_ecef(self.to_ecef(enu))
    }

    /// Geodetic position of the ellipsoid point whose normal passes through the tangent-plane
    /// point `(east, north, 0)`, approximately: the geodetic lat/lon of that plane point. This is
    /// the map projection a local height grid uses (`h_m` is how far the plane sits above the
    /// ellipsoid there, ≈ d²/2R: 7.8 m at 10 km, 31 m at 20 km).
    pub fn plane_to_geodetic(&self, east: f64, north: f64) -> Geodetic {
        self.enu_to_geodetic(DVec3::new(east, north, 0.0))
    }

    /// Height of the tangent plane above the ellipsoid at `(east, north)`: how much a point at
    /// ellipsoid height `h` "drops" below the plane. Subtract it from a DEM height to get the ENU
    /// up coordinate (to first order).
    pub fn curvature_drop_m(&self, east: f64, north: f64) -> f64 {
        self.plane_to_geodetic(east, north).h_m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ecef_known_points() {
        let p = Geodetic::new(0.0, 0.0, 0.0).to_ecef();
        assert!((p - DVec3::new(WGS84_A, 0.0, 0.0)).length() < 1e-6);
        let p = Geodetic::new(0.0, 90.0, 0.0).to_ecef();
        assert!((p - DVec3::new(0.0, WGS84_A, 0.0)).length() < 1e-6);
        let p = Geodetic::new(90.0, 0.0, 0.0).to_ecef();
        assert!((p - DVec3::new(0.0, 0.0, WGS84_B)).length() < 1e-6);
        // Adelaide GPO; reference values from an independent Python evaluation of the closed form.
        let p = Geodetic::new(-34.9285, 138.6007, 50.0).to_ecef();
        assert!((p - DVec3::new(-3_926_883.874, 3_461_928.569, -3_631_395.019)).length() < 1e-2, "{p:?}");
    }

    #[test]
    fn ecef_round_trip() {
        for lat in [-89.9, -60.0, -34.9285, 0.0, 12.5, 43.73, 89.99] {
            for lon in [-179.5, -73.0, 0.0, 7.42, 138.6, 180.0] {
                for h in [-11_000.0, 0.0, 50.0, 8_848.0, 400_000.0] {
                    let g = Geodetic::new(lat, lon, h);
                    let b = Geodetic::from_ecef(g.to_ecef());
                    assert!((b.lat_deg - lat).abs() < 1e-9, "{g:?} -> {b:?}");
                    let dlon = (b.lon_deg - lon).rem_euclid(360.0);
                    assert!(dlon.min(360.0 - dlon) < 1e-9, "{g:?} -> {b:?}");
                    assert!((b.h_m - h).abs() < 1e-5, "{g:?} -> {b:?}");
                }
            }
        }
        let pole = Geodetic::from_ecef(DVec3::new(0.0, 0.0, WGS84_B + 10.0));
        assert_eq!((pole.lat_deg, pole.h_m), (90.0, 10.0));
    }

    #[test]
    fn enu_frame() {
        let enu = Enu::new(Geodetic::new(-34.9285, 138.6007, 0.0));
        let [e, n, u] = enu.axes();
        assert!((e.dot(n)).abs() < 1e-12 && (e.dot(u)).abs() < 1e-12 && (n.dot(u)).abs() < 1e-12);
        assert!((e.cross(n) - u).length() < 1e-12, "right-handed E,N,U");
        assert!(enu.from_ecef(enu.origin_ecef()).length() < 1e-9);
        // 1 km north along the meridian is ~1 km north in ENU.
        let dlat = 1000.0 / meridional_radius(-34.9285);
        let p = enu.geodetic_to_enu(Geodetic::new(-34.9285 + dlat.to_degrees(), 138.6007, 0.0));
        assert!((p.y - 1000.0).abs() < 0.01 && p.x.abs() < 1e-6 && (p.z + 0.078).abs() < 0.01, "{p:?}");
        // round trip
        let q = DVec3::new(-5432.1, 2345.6, 123.4);
        assert!((enu.geodetic_to_enu(enu.enu_to_geodetic(q)) - q).length() < 1e-6);
        // curvature drop ≈ d²/2R
        let d = enu.curvature_drop_m(10_000.0, 0.0);
        assert!((d - 1e8 / (2.0 * prime_vertical_radius(-34.9285))).abs() < 0.05, "{d}");
    }
}
