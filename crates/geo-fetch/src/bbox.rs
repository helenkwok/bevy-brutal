//! WGS84 bounding boxes and Web-Mercator ("slippy map") tile math.

use std::fmt;
use std::str::FromStr;

use crate::{Error, Result};

/// Latitude limit of the square Web-Mercator world (EPSG:3857).
pub const MAX_MERCATOR_LAT: f64 = 85.051_128_779_806_59;
/// Highest zoom published in the AWS Terrain Tiles bucket.
pub const MAX_ZOOM: u8 = 15;

/// Axis-aligned WGS84 box in degrees. Boxes crossing the antimeridian are not supported
/// (split them into two).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BBox {
    pub west: f64,
    pub south: f64,
    pub east: f64,
    pub north: f64,
}

/// A slippy-map tile: `x` grows east, `y` grows south, both in `0..2^z`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileId {
    pub z: u8,
    pub x: u32,
    pub y: u32,
}

impl BBox {
    pub fn new(west: f64, south: f64, east: f64, north: f64) -> Result<Self> {
        let b = Self { west, south, east, north };
        let finite = [west, south, east, north].iter().all(|v| v.is_finite());
        if !finite {
            return Err(Error::BadBBox(format!("{b}: non-finite value")));
        }
        if !(-180.0..=180.0).contains(&west) || !(-180.0..=180.0).contains(&east) {
            return Err(Error::BadBBox(format!("{b}: longitude outside -180..180")));
        }
        if !(-90.0..=90.0).contains(&south) || !(-90.0..=90.0).contains(&north) {
            return Err(Error::BadBBox(format!("{b}: latitude outside -90..90")));
        }
        if west >= east {
            return Err(Error::BadBBox(format!("{b}: west must be < east (antimeridian boxes: split in two)")));
        }
        if south >= north {
            return Err(Error::BadBBox(format!("{b}: south must be < north")));
        }
        Ok(b)
    }

    /// True if `other` lies entirely inside `self` (edges inclusive).
    pub fn contains(&self, other: &BBox) -> bool {
        self.west <= other.west && self.south <= other.south && self.east >= other.east && self.north >= other.north
    }

    /// Area in square degrees (a cheap "smallest region" key, not a geodesic area).
    pub fn area_deg2(&self) -> f64 {
        (self.east - self.west) * (self.north - self.south)
    }

    /// Inclusive tile ranges `(x0..=x1, y0..=y1)` covering the box at zoom `z`.
    /// Latitudes are clamped to the Mercator limit.
    pub fn tile_range(&self, z: u8) -> (std::ops::RangeInclusive<u32>, std::ops::RangeInclusive<u32>) {
        let a = lonlat_to_tile(self.west, self.north, z);
        let b = lonlat_to_tile(self.east, self.south, z);
        (a.x..=b.x, a.y..=b.y)
    }

    pub fn tile_count(&self, z: u8) -> usize {
        let (xs, ys) = self.tile_range(z);
        (xs.end() - xs.start() + 1) as usize * (ys.end() - ys.start() + 1) as usize
    }

    /// All tiles covering the box at zoom `z`, row-major (north to south, west to east).
    pub fn tiles(&self, z: u8) -> Vec<TileId> {
        let (xs, ys) = self.tile_range(z);
        ys.flat_map(|y| xs.clone().map(move |x| TileId { z, x, y })).collect()
    }

    /// 1°×1° cells (south-west corner, integer degrees) covering the box, for SRTM-style tiles.
    pub fn degree_cells(&self) -> Vec<(i32, i32)> {
        let lat0 = self.south.floor() as i32;
        let lat1 = (self.north.ceil() as i32 - 1).max(lat0);
        let lon0 = self.west.floor() as i32;
        let lon1 = (self.east.ceil() as i32 - 1).max(lon0);
        (lat0..=lat1).rev().flat_map(|lat| (lon0..=lon1).map(move |lon| (lat, lon))).collect()
    }
}

impl fmt::Display for BBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{},{},{},{}", self.west, self.south, self.east, self.north)
    }
}

/// Parses `west,south,east,north` in degrees (the GeoJSON / OSM `bbox` order).
impl FromStr for BBox {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        let v: Vec<f64> = s
            .split(',')
            .map(|p| p.trim().parse::<f64>())
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| Error::BadBBox(format!("{s:?}: {e}")))?;
        match v[..] {
            [w, s_, e, n] => BBox::new(w, s_, e, n),
            _ => Err(Error::BadBBox(format!("{s:?}: expected west,south,east,north"))),
        }
    }
}

/// The tile containing a point. Longitude 180 and the Mercator poles map to the last tile.
pub fn lonlat_to_tile(lon: f64, lat: f64, z: u8) -> TileId {
    let n = (1u64 << z) as f64;
    let lat = lat.clamp(-MAX_MERCATOR_LAT, MAX_MERCATOR_LAT).to_radians();
    let fx = (lon + 180.0) / 360.0 * n;
    let fy = (1.0 - (lat.tan() + 1.0 / lat.cos()).ln() / std::f64::consts::PI) / 2.0 * n;
    let max = (1u32 << z) - 1;
    TileId { z, x: (fx.floor().max(0.0) as u32).min(max), y: (fy.floor().max(0.0) as u32).min(max) }
}

impl TileId {
    /// WGS84 bounds of this tile.
    pub fn bbox(&self) -> BBox {
        let n = (1u64 << self.z) as f64;
        let lon = |x: f64| x / n * 360.0 - 180.0;
        let lat = |y: f64| (std::f64::consts::PI * (1.0 - 2.0 * y / n)).sinh().atan().to_degrees();
        BBox {
            west: lon(self.x as f64),
            east: lon(self.x as f64 + 1.0),
            north: lat(self.y as f64),
            south: lat(self.y as f64 + 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_validate() {
        let b: BBox = "7.40,43.72,7.44,43.75".parse().unwrap();
        assert_eq!(b, BBox { west: 7.40, south: 43.72, east: 7.44, north: 43.75 });
        assert_eq!(b.to_string(), "7.4,43.72,7.44,43.75");
        assert!("1,2,3".parse::<BBox>().is_err());
        assert!("a,2,3,4".parse::<BBox>().is_err());
        assert!("10,0,5,1".parse::<BBox>().is_err()); // west > east
        assert!("0,10,1,5".parse::<BBox>().is_err()); // south > north
        assert!("0,0,181,1".parse::<BBox>().is_err());
        assert!("0,-91,1,1".parse::<BBox>().is_err());
    }

    #[test]
    fn known_tiles() {
        assert_eq!(lonlat_to_tile(0.0, 0.0, 0), TileId { z: 0, x: 0, y: 0 });
        assert_eq!(lonlat_to_tile(0.0, 0.0, 1), TileId { z: 1, x: 1, y: 1 });
        assert_eq!(lonlat_to_tile(180.0, -90.0, 2), TileId { z: 2, x: 3, y: 3 });
        // Adelaide CBD (138.6007 E, 34.9285 S) at z10 -> tile 906/618 (checked with the OSM wiki formula in Python).
        assert_eq!(lonlat_to_tile(138.6007, -34.9285, 10), TileId { z: 10, x: 906, y: 618 });
    }

    #[test]
    fn tile_bbox_round_trip() {
        let t = TileId { z: 12, x: 2133, y: 1495 };
        let b = t.bbox();
        let c = lonlat_to_tile((b.west + b.east) / 2.0, (b.south + b.north) / 2.0, 12);
        assert_eq!(c, t);
        assert!(b.north > b.south && b.east > b.west);
    }

    #[test]
    fn tile_cover() {
        let b: BBox = "7.40,43.72,7.44,43.75".parse().unwrap();
        assert_eq!(b.tile_count(0), 1);
        let t = b.tiles(14);
        assert_eq!(t.len(), b.tile_count(14));
        assert!(t.iter().all(|t| t.z == 14));
        // every tile intersects the box
        for tile in &t {
            let tb = tile.bbox();
            assert!(tb.west <= b.east && tb.east >= b.west && tb.south <= b.north && tb.north >= b.south);
        }
    }

    #[test]
    fn degree_cells() {
        let b: BBox = "7.4,43.72,7.44,43.75".parse().unwrap();
        assert_eq!(b.degree_cells(), vec![(43, 7)]);
        let b: BBox = "-1.5,-0.5,0.5,0.5".parse().unwrap();
        assert_eq!(b.degree_cells(), vec![(0, -2), (0, -1), (0, 0), (-1, -2), (-1, -1), (-1, 0)]);
        let b: BBox = "138,-35,139,-34".parse().unwrap();
        assert_eq!(b.degree_cells(), vec![(-35, 138)]);
    }
}
