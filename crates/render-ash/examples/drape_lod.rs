//! CPU sheet for the OSM drape (no GPU): the Sentinel-2-fitted palette vs the old hand palette,
//! and the building layer's screen-coverage LOD tiers (proxy -> merged -> objects) at several
//! sample footprints over the same window.
//!
//! ```text
//! cargo run --release -p render-ash --example drape_lod -- \
//!     --osm /workspace/data/geo/osm/derived/adelaide_wide --at -34.92,138.63 --size 1600 \
//!     --out out/quality/dem/drape_lod/norwood
//! ```
//! Writes `<out>_<panel>.png` (512 px, nearest-upscaled) and `<out>.json` (tier weights).
//! © OpenStreetMap contributors (ODbL); palette: contains modified Copernicus Sentinel data 2024.

use geo_fetch::derived::{DerivedGeom, read_layer};
use planet::drape::{Drape, DrapeClass, DrapeFeature, DrapePalette, DrapeShape};
use std::path::{Path, PathBuf};

const PX: usize = 512;

type Proj<'a> = dyn Fn(&[[f64; 2]]) -> Vec<[f64; 2]> + 'a;

fn main() {
    let mut osm = PathBuf::from("/workspace/data/geo/osm/derived/adelaide_wide");
    let (mut lat, mut lon, mut size) = (-34.92, 138.63, 1600.0);
    let mut out = PathBuf::from("out/quality/dem/drape_lod/norwood");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let v = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--osm" => osm = v.into(),
            "--at" => {
                let p: Vec<f64> = v.split(',').map(|x| x.parse().expect("--at lat,lon")).collect();
                (lat, lon) = (p[0], p[1]);
            }
            "--size" => size = v.parse().expect("--size metres"),
            "--out" => out = v.into(),
            a => panic!("unknown arg {a}"),
        }
        i += 2;
    }
    if let Some(d) = out.parent() {
        std::fs::create_dir_all(d).unwrap();
    }
    // local tangent plane (equirectangular about the window centre; < 1e-4 error over 2 km)
    let (my, mx) = (111_132.0, 111_320.0 * lat.to_radians().cos());
    let proj = |r: &[[f64; 2]]| -> Vec<[f64; 2]> { r.iter().map(|p| [(p[0] - lon) * mx, (p[1] - lat) * my]).collect() };
    let h = 0.5 * size + 200.0;
    let inside = |r: &[[f64; 2]]| r.iter().any(|p| p[0].abs() < h && p[1].abs() < h);
    let feats = load(&osm, &proj, &inside);
    let rect = (-0.5 * size, -0.5 * size, 0.5 * size, 0.5 * size);
    let s2 = Drape::build(&feats, rect, 0.5);
    let old = Drape::build_with(&feats, rect, 0.5, old_palette());
    let mut report = Vec::new();
    let mut panel = |name: &str, d: &Drape, step: f64| {
        let n = ((size / step).round() as usize).clamp(1, PX);
        let cells: Vec<[f64; 3]> = (0..n * n)
            .map(|k| {
                let (x, y) = (k % n, k / n);
                let e = rect.0 + (x as f64 + 0.5) * size / n as f64;
                let nn = rect.3 - (y as f64 + 0.5) * size / n as f64;
                d.composite([0.10, 0.09, 0.07], e, nn, size / n as f64, 1.0).0
            })
            .collect();
        let mut img = vec![0u8; PX * PX * 3];
        for y in 0..PX {
            for x in 0..PX {
                let c = cells[(y * n / PX) * n + x * n / PX];
                for j in 0..3 {
                    img[(y * PX + x) * 3 + j] = planet::dem::srgb8(c[j]);
                }
            }
        }
        let path = PathBuf::from(format!("{}_{name}.png", out.display()));
        let f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        let mut enc = png::Encoder::new(f, PX as u32, PX as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.write_header().unwrap().write_image_data(&img).unwrap();
        let t = d.lod.weights(size / n as f64);
        report.push(format!(
            "{{\"panel\":\"{name}\",\"step_m\":{:.2},\"coverage_px\":{:.2},\"proxy\":{:.3},\"merged\":{:.3},\"objects\":{:.3}}}",
            size / n as f64,
            d.lod.coverage_px(size / n as f64),
            t.proxy,
            t.merged,
            t.objects
        ));
        eprintln!("{}", path.display());
    };
    panel("old_palette_3m", &old, 3.125);
    panel("s2_3m", &s2, 3.125);
    for step in [3.125, 6.25, 12.5, 25.0, 50.0, 100.0] {
        panel(&format!("s2_{step}m"), &s2, step);
    }
    std::fs::write(format!("{}.json", out.display()), format!("[{}]\n", report.join(",\n"))).unwrap();
}

fn load(dir: &Path, proj: &Proj, inside: &dyn Fn(&[[f64; 2]]) -> bool) -> Vec<DrapeFeature> {
    let mut feats = Vec::new();
    for layer in ["landuse", "water", "buildings", "roads"] {
        for f in read_layer(dir, layer).unwrap_or_else(|e| panic!("OSM {}: {e}", dir.display())) {
            let Some(class) = DrapeClass::from_osm(&f.key, &f.kind) else { continue };
            if f.tunnel {
                continue;
            }
            match &f.geom {
                DerivedGeom::Line(l) if inside(l) => feats.push(DrapeFeature {
                    class,
                    shape: DrapeShape::Line {
                        pts: proj(l),
                        width_m: f.width_m.unwrap_or_else(|| class.default_width_m(&f.kind)),
                    },
                }),
                DerivedGeom::Polygon(p) if p.first().is_some_and(|r| inside(r)) => feats.push(DrapeFeature {
                    class,
                    shape: DrapeShape::Area { rings: p.iter().map(|r| proj(r)).collect() },
                }),
                DerivedGeom::MultiPolygon(m) => {
                    feats.extend(m.iter().filter(|p| p.first().is_some_and(|r| inside(r))).map(|p| DrapeFeature {
                        class,
                        shape: DrapeShape::Area { rings: p.iter().map(|r| proj(r)).collect() },
                    }))
                }
                _ => {}
            }
        }
    }
    feats
}

/// The pre-Sentinel hand palette (round 58), for the before / after panel.
fn old_palette() -> DrapePalette {
    let full = [
        [0.150, 0.125, 0.075],
        [0.110, 0.105, 0.090],
        [0.170, 0.165, 0.155],
        [0.070, 0.105, 0.045],
        [0.040, 0.058, 0.032],
        [0.260, 0.230, 0.180],
        [0.030, 0.045, 0.055],
        [0.200, 0.170, 0.150],
        [0.150, 0.135, 0.115],
        [0.090, 0.090, 0.092],
        [0.065, 0.066, 0.070],
    ];
    let b = full[DrapeClass::Building.index()];
    DrapePalette { full, proxy: full, roofs: [b; 5], roof_weights: [1.0; 5] }
}
