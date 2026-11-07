//! Read back `geo-fetch osm-extract` layers (`<layer>.geojson`, one feature per line) without the
//! `pbf` feature: lon/lat geometry plus the `key` / `kind` classification and a few tags.
//! The data is ODbL: keep `ATTRIBUTION.txt` / `LICENSE.txt` with anything derived from it.

use std::io::BufRead;
use std::path::Path;

use serde_json::Value;

use crate::{Error, Result};

/// Lon/lat (deg) geometry of a derived feature.
#[derive(Clone, Debug, PartialEq)]
pub enum DerivedGeom {
    Line(Vec<[f64; 2]>),
    /// Outer ring first, then holes.
    Polygon(Vec<Vec<[f64; 2]>>),
    MultiPolygon(Vec<Vec<Vec<[f64; 2]>>>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct DerivedFeature {
    /// OSM key that selected the feature (`highway`, `building`, `landuse`, …).
    pub key: String,
    /// Its value (`primary`, `yes`, `residential`, …).
    pub kind: String,
    /// `width` tag in metres, when it parses as a plain number.
    pub width_m: Option<f64>,
    /// `tunnel` tag set (not `no`): tunnels are not visible from above.
    pub tunnel: bool,
    pub geom: DerivedGeom,
}

fn pt(v: &Value) -> Option<[f64; 2]> {
    let a = v.as_array()?;
    Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?])
}

fn line(v: &Value) -> Option<Vec<[f64; 2]>> {
    v.as_array()?.iter().map(pt).collect()
}

fn poly(v: &Value) -> Option<Vec<Vec<[f64; 2]>>> {
    v.as_array()?.iter().map(line).collect()
}

/// Parse one GeoJSON feature object; `None` for unsupported geometry.
pub fn parse_feature(v: &Value) -> Option<DerivedFeature> {
    let props = v.get("properties")?;
    let s = |k: &str| props.get(k).and_then(Value::as_str);
    let g = v.get("geometry")?;
    let c = g.get("coordinates")?;
    let geom = match g.get("type")?.as_str()? {
        "LineString" => DerivedGeom::Line(line(c)?),
        "Polygon" => DerivedGeom::Polygon(poly(c)?),
        "MultiPolygon" => DerivedGeom::MultiPolygon(c.as_array()?.iter().map(poly).collect::<Option<_>>()?),
        _ => return None,
    };
    Some(DerivedFeature {
        key: s("key")?.to_owned(),
        kind: s("kind").unwrap_or("").to_owned(),
        width_m: s("width").and_then(|w| w.trim().trim_end_matches(" m").trim().parse().ok()),
        tunnel: s("tunnel").is_some_and(|t| t != "no"),
        geom,
    })
}

/// Read `<dir>/<layer>.geojson` (`layer` = roads | buildings | water | landuse), streaming one
/// feature per line (the writer's layout); a missing file is an empty layer.
pub fn read_layer(dir: &Path, layer: &str) -> Result<Vec<DerivedFeature>> {
    let path = dir.join(format!("{layer}.geojson"));
    let io = |e| Error::Io { path: path.display().to_string(), source: e };
    let f = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(io(e)),
    };
    let mut out = Vec::new();
    for l in std::io::BufReader::new(f).lines() {
        let l = l.map_err(io)?;
        let t = l.trim().trim_end_matches(',');
        if !t.starts_with("{\"type\":\"Feature\"") {
            continue;
        }
        let v: Value = serde_json::from_str(t).map_err(|e| Error::Decode(format!("{}: {e}", path.display())))?;
        out.extend(parse_feature(&v));
    }
    Ok(out)
}

/// The `bbox` (`[west, south, east, north]`, degrees) recorded in a derived directory's
/// `manifest.json`, if any.
pub fn read_bbox(dir: &Path) -> Option<[f64; 4]> {
    let v: Value = serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).ok()?).ok()?;
    let b = v.get("bbox")?.as_array()?;
    if b.len() != 4 {
        return None;
    }
    let mut o = [0.0; 4];
    for (o, x) in o.iter_mut().zip(b) {
        *o = x.as_f64()?;
    }
    Some(o)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_line_per_feature() {
        let dir = std::env::temp_dir().join(format!("geo-fetch-derived-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("roads.geojson"),
            concat!(
                "{\"type\":\"FeatureCollection\",\"features\":[\n",
                "{\"type\":\"Feature\",\"id\":\"way/1\",\"properties\":{\"key\":\"highway\",\"kind\":\"primary\",\"width\":\"12\"},\"geometry\":{\"type\":\"LineString\",\"coordinates\":[[138.6,-34.9],[138.61,-34.9]]}},\n",
                "{\"type\":\"Feature\",\"id\":\"way/2\",\"properties\":{\"key\":\"building\",\"kind\":\"yes\",\"tunnel\":\"yes\"},\"geometry\":{\"type\":\"Polygon\",\"coordinates\":[[[0,0],[1,0],[1,1],[0,0]]]}}\n",
                "]}\n"
            ),
        )
        .unwrap();
        let f = read_layer(&dir, "roads").unwrap();
        assert_eq!(f.len(), 2);
        assert_eq!(f[0].kind, "primary");
        assert_eq!(f[0].width_m, Some(12.0));
        assert!(matches!(&f[0].geom, DerivedGeom::Line(l) if l.len() == 2));
        assert!(f[1].tunnel);
        assert!(read_layer(&dir, "water").unwrap().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}
