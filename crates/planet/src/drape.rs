//! OSM colour drape for a [`crate::dem::DemPlacement`]: vector features (roads, buildings, water,
//! landuse) already projected into the DEM's ENU plane are rasterised once into per-class
//! coverage maps (anti-aliased, `cell_m` metres a cell, with a box mip pyramid), and the graft
//! bake samples them at its own texel footprint to composite a sharper albedo than the ~2.4 km
//! Blue Marble ("the pixel shows the average": a texel far larger than a road gets the road's
//! area fraction, not a hard line).
//!
//! The module is data-source agnostic (no GeoJSON here): callers map their features to
//! [`DrapeClass`] with [`DrapeClass::from_osm`] and to plane coordinates with
//! [`crate::dem::DemPlacement::latlon_to_plane`]. OSM data is ODbL: credit
//! "© OpenStreetMap contributors" wherever the drape is shown.

use rayon::prelude::*;

/// Coverage classes, in compositing order (later classes paint over earlier ones).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DrapeClass {
    /// Farmland, vineyards, orchards, meadows.
    Farm,
    /// Residential / garages landuse (suburb mix of roofs, gardens, street trees).
    Residential,
    /// Industrial, commercial, retail, railway yards.
    Industrial,
    /// Parks, grass, golf, sport pitches, cemeteries.
    Grass,
    /// Wood, forest, scrub, heath, nature reserves.
    Trees,
    /// Bare sand / beach / rock.
    Bare,
    Water,
    Building,
    /// Footways, cycleways, tracks.
    Path,
    /// Residential / service / unclassified streets.
    MinorRoad,
    /// Motorway .. tertiary (+ links).
    MajorRoad,
}

impl DrapeClass {
    pub const ALL: [DrapeClass; 11] = [
        DrapeClass::Farm,
        DrapeClass::Residential,
        DrapeClass::Industrial,
        DrapeClass::Grass,
        DrapeClass::Trees,
        DrapeClass::Bare,
        DrapeClass::Water,
        DrapeClass::Building,
        DrapeClass::Path,
        DrapeClass::MinorRoad,
        DrapeClass::MajorRoad,
    ];

    pub fn index(self) -> usize {
        Self::ALL.iter().position(|c| *c == self).unwrap_or(0)
    }

    /// Classify a geo-fetch derived feature (`key` = selecting OSM key, `kind` = its value).
    /// `None` = not draped (e.g. waterway lines narrower than a texel are still kept as Water).
    pub fn from_osm(key: &str, kind: &str) -> Option<DrapeClass> {
        use DrapeClass::*;
        Some(match key {
            "highway" => match kind {
                "motorway" | "trunk" | "primary" | "secondary" | "tertiary" | "motorway_link" | "trunk_link"
                | "primary_link" | "secondary_link" | "tertiary_link" | "busway" | "bus_guideway" => MajorRoad,
                "footway" | "cycleway" | "bridleway" | "path" | "steps" | "corridor" | "track" | "pedestrian" => Path,
                _ => MinorRoad,
            },
            "building" => Building,
            "natural" | "waterway" if matches!(kind, "water" | "riverbank" | "dock" | "river" | "canal" | "stream") => {
                Water
            }
            "landuse" if matches!(kind, "reservoir" | "basin") => Water,
            "landuse" => match kind {
                "residential" | "garages" => Residential,
                "industrial" | "commercial" | "retail" | "railway" | "construction" | "brownfield" | "port"
                | "military" | "quarry" | "landfill" => Industrial,
                "farmland"
                | "vineyard"
                | "orchard"
                | "meadow"
                | "farmyard"
                | "allotments"
                | "plant_nursery"
                | "greenhouse_horticulture" => Farm,
                "forest" => Trees,
                "grass" | "recreation_ground" | "cemetery" | "village_green" | "flowerbed" | "greenfield" => Grass,
                _ => return None,
            },
            "leisure" => match kind {
                "nature_reserve" => Trees,
                "marina" => Water,
                _ => Grass,
            },
            "natural" => match kind {
                "wood" | "scrub" | "heath" => Trees,
                "grassland" | "fell" => Grass,
                "sand" | "beach" | "bare_rock" | "scree" | "shingle" | "mud" => Bare,
                "wetland" => Grass,
                _ => return None,
            },
            _ => return None,
        })
    }

    /// Linear albedo of the merged / objects tiers ([`DrapePalette::adelaide_s2`] `full` fit).
    pub fn albedo(self) -> [f64; 3] {
        DrapePalette::adelaide_s2().full[self.index()]
    }

    /// Default painted width (m) of a line feature.
    pub fn default_width_m(self, kind: &str) -> f64 {
        match (self, kind) {
            (DrapeClass::MajorRoad, "motorway" | "trunk") => 22.0,
            (DrapeClass::MajorRoad, "primary") => 16.0,
            (DrapeClass::MajorRoad, k) if k.ends_with("_link") => 8.0,
            (DrapeClass::MajorRoad, _) => 12.0,
            (DrapeClass::MinorRoad, "service") => 5.0,
            (DrapeClass::MinorRoad, _) => 8.0,
            (DrapeClass::Path, _) => 2.5,
            (DrapeClass::Water, "river" | "canal") => 15.0,
            (DrapeClass::Water, _) => 4.0,
            _ => 4.0,
        }
    }
}

/// Number of drape classes.
pub const NC: usize = DrapeClass::ALL.len();

/// Per-class linear albedos fitted to Sentinel-2 L2A surface reflectance (`tools/s2_fit.py`):
/// a painter's-order NNLS fit of every 10 m pixel's B04/B03/B02 against its OSM class coverage, so
/// the drape's averaged texels reproduce what the satellite sees ("the pixel shows the average").
/// Contains modified Copernicus Sentinel data 2024; © OpenStreetMap contributors (ODbL).
#[derive(Clone, Debug, PartialEq)]
pub struct DrapePalette {
    /// Buildings as their own layer (merged / objects tiers).
    pub full: [[f64; 3]; NC],
    /// Buildings folded into the landuse beneath (proxy tier): the 10 m residential / industrial
    /// pixels already contain their roofs, so painting buildings again would double-count them.
    /// The `Building` entry is unused (= `full`).
    pub proxy: [[f64; 3]; NC],
    /// Roof colour variants for the objects tier (weighted mean = `full[Building]`).
    pub roofs: [[f64; 3]; 5],
    /// Share of buildings given each variant.
    pub roof_weights: [f64; 5],
}

impl DrapePalette {
    /// Variant (0..5) for a hash: weighted by [`Self::roof_weights`].
    fn pick_roof(&self, h: u64) -> usize {
        let total: f64 = self.roof_weights.iter().sum();
        let mut t = (h >> 11) as f64 / (1u64 << 53) as f64 * total;
        for (i, w) in self.roof_weights.iter().enumerate() {
            if t < *w {
                return i;
            }
            t -= w;
        }
        self.roofs.len() - 1
    }
}

impl DrapePalette {
    /// Adelaide, S2B 2024-12-24 (tile 54HTG, 0.02 % cloud), `adelaide_wide` OSM extract; see
    /// `out/quality/DEM_osm_drape.md` for the fit (RMSE ~0.05 per band over 11.5 M pixels).
    /// Water is the median of open-water pixels (SCL 6, fully OSM water), not the regression.
    pub fn adelaide_s2() -> DrapePalette {
        // order = DrapeClass::ALL
        let full = [
            [0.1023, 0.0850, 0.0594], // Farm (summer stubble / dry pasture)
            [0.1302, 0.1141, 0.0920], // Residential ground (gardens, drives, street trees)
            [0.2044, 0.1719, 0.1379], // Industrial
            [0.0979, 0.0883, 0.0583], // Grass (parks; mostly dry in December)
            [0.0598, 0.0615, 0.0462], // Trees
            [0.2414, 0.2364, 0.1917], // Bare / beach
            [0.0586, 0.0896, 0.0658], // Water (open, inland / river)
            [0.2448, 0.2254, 0.2022], // Building (roof mean)
            [0.1257, 0.1016, 0.0786], // Path
            [0.1073, 0.0963, 0.0774], // MinorRoad (incl. verge / tree canopy at 10 m)
            [0.1028, 0.0915, 0.0780], // MajorRoad
        ];
        let mut proxy = [
            [0.1022, 0.0848, 0.0591],
            [0.1382, 0.1215, 0.0997], // Residential incl. roofs
            [0.2296, 0.1999, 0.1666], // Industrial incl. roofs
            [0.0998, 0.0902, 0.0603],
            [0.0600, 0.0617, 0.0463],
            [0.2417, 0.2361, 0.1914],
            [0.0586, 0.0896, 0.0658],
            [0.0; 3],
            [0.1191, 0.0947, 0.0710],
            [0.1025, 0.0917, 0.0727],
            [0.1018, 0.0905, 0.0771],
        ];
        proxy[DrapeClass::Building.index()] = full[DrapeClass::Building.index()];
        // fully-covered roof pixels: terracotta tile (R/G > 1.2) + luminance quartiles of the
        // rest (dark Colorbond .. galvanised), with their pixel shares; rescaled to the regression
        // mean (weighted) so the objects tier averages to the merged tier
        let q = [
            [0.1903, 0.1393, 0.1132],
            [0.1450, 0.1348, 0.1186],
            [0.2454, 0.2295, 0.2024],
            [0.3278, 0.3068, 0.2732],
            [0.5309, 0.5164, 0.4823],
        ];
        let roof_weights = [0.126, 0.219, 0.219, 0.219, 0.219];
        let b = full[DrapeClass::Building.index()];
        let mut roofs = q;
        let wsum: f64 = roof_weights.iter().sum();
        for i in 0..3 {
            let m = q.iter().zip(roof_weights).map(|(c, w)| c[i] * w).sum::<f64>() / wsum;
            for r in &mut roofs {
                r[i] *= b[i] / m;
            }
        }
        DrapePalette { full, proxy, roofs, roof_weights }
    }
}

/// Fragments-style screen-coverage LOD for the building layer. "Screen" is the sample footprint
/// (`step_m`: the graft texel, chosen per quadtree level by screen error), and coverage is how
/// many footprints one typical building spans, `building_m / step_m`:
/// * **proxy** (< `merged_px`): buildings dissolve into the landuse beneath with the S2 `proxy`
///   palette (roofs already inside those colours), no building layer at all;
/// * **merged** (`merged_px` .. `objects_px`): building coverage as one layer at the mean roof
///   albedo (stable, no per-roof shimmer);
/// * **objects** (≥ `objects_px`): per-building roof variant + outline darkening.
///
/// Tiers cross-fade over `blend_octaves` (log2 of coverage) so a level change never pops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BuildingLod {
    /// Typical building size (m): √ of the extract's median footprint area.
    pub building_m: f64,
    pub merged_px: f64,
    pub objects_px: f64,
    pub blend_octaves: f64,
    /// Albedo factor on outline cells (eave / wall shadow) in the objects tier.
    pub edge_dark: f64,
}

impl Default for BuildingLod {
    fn default() -> Self {
        // Adelaide wide extract: 183 710 buildings, median 185.5 m² -> 13.6 m
        BuildingLod { building_m: 13.6, merged_px: 0.5, objects_px: 2.0, blend_octaves: 0.5, edge_dark: 0.85 }
    }
}

/// Which building tier(s) a footprint uses: weights `(proxy, merged, objects)` summing to 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TierWeights {
    pub proxy: f64,
    pub merged: f64,
    pub objects: f64,
}

impl BuildingLod {
    /// Pixels (footprints) a typical building spans at `step_m`.
    pub fn coverage_px(&self, step_m: f64) -> f64 {
        self.building_m / step_m.max(1e-6)
    }

    pub fn weights(&self, step_m: f64) -> TierWeights {
        let c = self.coverage_px(step_m).log2();
        let h = self.blend_octaves.max(1e-6);
        let up = |th: f64| {
            let t = ((c - th.log2() + h) / (2.0 * h)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        let not_proxy = up(self.merged_px);
        let objects = up(self.objects_px).min(not_proxy);
        TierWeights { proxy: 1.0 - not_proxy, merged: not_proxy - objects, objects }
    }
}

/// Plane-coordinate geometry (metres east / north of the DEM's ENU origin).
#[derive(Clone, Debug)]
pub enum DrapeShape {
    /// Polyline painted `width_m` wide (round joins / caps).
    Line { pts: Vec<[f64; 2]>, width_m: f64 },
    /// Rings filled even-odd (outer + holes; multipolygons = several features or extra rings).
    Area { rings: Vec<Vec<[f64; 2]>> },
}

#[derive(Clone, Debug)]
pub struct DrapeFeature {
    pub class: DrapeClass,
    pub shape: DrapeShape,
}

/// Extra coverage channels after the [`NC`] classes: roof albedo premultiplied by building
/// coverage (R, G, B, scaled by [`ROOF_SCALE`]) and building-outline coverage. Mipped and
/// filtered like the classes, so the objects tier is an average too.
const ROOF: usize = NC;
const EDGE: usize = NC + 3;
const NCH: usize = NC + 4;
/// Roof albedo stored as `albedo / ROOF_SCALE * 255`.
const ROOF_SCALE: f64 = 0.6;

/// One coverage mip level: `w * h` cells per channel, 0..255 = area fraction.
#[derive(Clone, Debug)]
struct Level {
    w: usize,
    h: usize,
    cell: f64,
    cov: Vec<Vec<u8>>,
}

impl Level {
    fn down(&self) -> Option<Level> {
        if self.w < 4 || self.h < 4 {
            return None;
        }
        let (w, h) = (self.w / 2, self.h / 2);
        let cov = self
            .cov
            .par_iter()
            .map(|c| {
                let mut o = vec![0u8; w * h];
                for y in 0..h {
                    for x in 0..w {
                        let s = |dx: usize, dy: usize| c[(2 * y + dy) * self.w + 2 * x + dx] as u32;
                        o[y * w + x] = ((s(0, 0) + s(1, 0) + s(0, 1) + s(1, 1) + 2) / 4) as u8;
                    }
                }
                o
            })
            .collect();
        Some(Level { w, h, cell: self.cell * 2.0, cov })
    }

    /// Bilinear class fractions at plane-relative cell coords.
    fn sample(&self, fx: f64, fy: f64, out: &mut [f64]) {
        let (x, y) = (fx - 0.5, fy - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (tx, ty) = (x - x0, y - y0);
        let at = |c: &Vec<u8>, xi: f64, yi: f64| {
            if xi < 0.0 || yi < 0.0 || xi >= self.w as f64 || yi >= self.h as f64 {
                0.0
            } else {
                c[yi as usize * self.w + xi as usize] as f64 / 255.0
            }
        };
        for (o, c) in out.iter_mut().zip(&self.cov) {
            let a = at(c, x0, y0) + (at(c, x0 + 1.0, y0) - at(c, x0, y0)) * tx;
            let b = at(c, x0, y0 + 1.0) + (at(c, x0 + 1.0, y0 + 1.0) - at(c, x0, y0 + 1.0)) * tx;
            *o = a + (b - a) * ty;
        }
    }
}

/// Rasterised drape (see the module docs).
#[derive(Clone, Debug)]
pub struct Drape {
    /// South-west corner of the raster (plane metres).
    x0: f64,
    y0: f64,
    /// North-east corner of the rectangle passed to [`Drape::build`].
    x1: f64,
    y1: f64,
    /// Width (m) of the fade from the raster's rectangle edges inward ([`Drape::edge_weight`]).
    pub edge_fade_m: f64,
    levels: Vec<Level>,
    /// Feature counts per class (report only).
    pub counts: [usize; DrapeClass::ALL.len()],
    pub palette: DrapePalette,
    pub lod: BuildingLod,
}

/// Supersampling per cell side (coverage = SS² sub-samples).
const SS: usize = 4;

impl Drape {
    /// Rasterise `features` over the plane rectangle `[x0, x1] × [y0, y1]` at `cell_m` metres.
    pub fn build(features: &[DrapeFeature], rect: (f64, f64, f64, f64), cell_m: f64) -> Drape {
        Self::build_with(features, rect, cell_m, DrapePalette::adelaide_s2())
    }

    /// [`Self::build`] with an explicit palette (its roof variants are baked into the raster).
    pub fn build_with(
        features: &[DrapeFeature],
        (x0, y0, x1, y1): (f64, f64, f64, f64),
        cell_m: f64,
        palette: DrapePalette,
    ) -> Drape {
        let w = ((x1 - x0) / cell_m).ceil().max(1.0) as usize;
        let h = ((y1 - y0) / cell_m).ceil().max(1.0) as usize;
        let mut counts = [0usize; DrapeClass::ALL.len()];
        for f in features {
            counts[f.class.index()] += 1;
        }
        // one class at a time: the 1-bit sub-sample buffer is (w·SS)·(h·SS) / 8 bytes
        let per: Vec<Vec<Vec<u8>>> = DrapeClass::ALL
            .par_iter()
            .map(|&cls| {
                let building = cls == DrapeClass::Building;
                let mut r = Raster::new(w, h, x0, y0, cell_m, building);
                for (i, f) in features.iter().enumerate().filter(|(_, f)| f.class == cls) {
                    // stable per-object roof variant (1..=5) from the feature's index
                    r.tag = palette.pick_roof(splitmix(i as u64)) as u8 + 1;
                    match &f.shape {
                        DrapeShape::Line { pts, width_m } => r.line(pts, *width_m),
                        DrapeShape::Area { rings } => r.area(rings),
                    }
                }
                let cov = r.coverage();
                if building {
                    let mut v = r.roof_channels(&cov, &palette.roofs);
                    v.insert(0, cov);
                    v
                } else {
                    vec![cov]
                }
            })
            .collect();
        let mut cov: Vec<Vec<u8>> = Vec::with_capacity(NCH);
        let mut extra = Vec::new();
        for (cls, mut v) in DrapeClass::ALL.iter().zip(per) {
            cov.push(v.remove(0));
            if *cls == DrapeClass::Building {
                extra = v;
            }
        }
        cov.extend(extra);
        debug_assert_eq!(cov.len(), NCH);
        let mut levels = vec![Level { w, h, cell: cell_m, cov }];
        while let Some(l) = levels.last().and_then(Level::down) {
            levels.push(l);
        }
        Drape { x0, y0, x1, y1, edge_fade_m: 0.0, levels, counts, palette, lod: BuildingLod::default() }
    }

    /// Override the class colours (roof variants stay as baked; use [`Self::build_with`]).
    pub fn with_palette(mut self, p: DrapePalette) -> Self {
        self.palette = p;
        self
    }

    /// Override the building screen-coverage LOD.
    pub fn with_lod(mut self, lod: BuildingLod) -> Self {
        self.lod = lod;
        self
    }

    /// Fade the drape out over `m` metres inside its rectangle's edges (see [`Self::edge_weight`]).
    pub fn with_edge_fade(mut self, m: f64) -> Self {
        self.edge_fade_m = m.max(0.0);
        self
    }

    /// Plane rectangle `(x0, y0, x1, y1)` the drape was rasterised over.
    pub fn rect(&self) -> (f64, f64, f64, f64) {
        (self.x0, self.y0, self.x1, self.y1)
    }

    /// 0 outside the rectangle, smoothstep up to 1 at [`Self::edge_fade_m`] inside it.
    pub fn edge_weight(&self, e: f64, n: f64) -> f64 {
        let inside = (e - self.x0).min(self.x1 - e).min(n - self.y0).min(self.y1 - n);
        if self.edge_fade_m <= 0.0 {
            return if inside >= 0.0 { 1.0 } else { 0.0 };
        }
        let t = (inside / self.edge_fade_m).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    pub fn cell_m(&self) -> f64 {
        self.levels[0].cell
    }

    pub fn size(&self) -> (usize, usize) {
        (self.levels[0].w, self.levels[0].h)
    }

    /// Class fractions at plane point `(e, n)` filtered for a sample footprint `step_m`
    /// (trilinear in the mip pyramid).
    pub fn fractions(&self, e: f64, n: f64, step_m: f64) -> [f64; NC] {
        let c = self.channels(e, n, step_m);
        std::array::from_fn(|i| c[i])
    }

    fn channels(&self, e: f64, n: f64, step_m: f64) -> [f64; NCH] {
        let lod = (step_m / self.levels[0].cell).max(1.0).log2().min((self.levels.len() - 1) as f64);
        let k = lod.floor() as usize;
        let t = lod - k as f64;
        let mut a = [0.0; NCH];
        let l = &self.levels[k];
        l.sample((e - self.x0) / l.cell, (n - self.y0) / l.cell, &mut a);
        if t > 0.0 && k + 1 < self.levels.len() {
            let mut b = [0.0; NCH];
            let l = &self.levels[k + 1];
            l.sample((e - self.x0) / l.cell, (n - self.y0) / l.cell, &mut b);
            for (x, y) in a.iter_mut().zip(b) {
                *x += (y - *x) * t;
            }
        }
        a
    }

    /// Composite over `base` (linear albedo) with weight `w` (0..1): each class in
    /// [`DrapeClass::ALL`] order paints over what lies beneath by its area fraction, with the
    /// building layer resolved by [`BuildingLod`] at this `step_m`. Returns the albedo and the
    /// water fraction.
    pub fn composite(&self, base: [f64; 3], e: f64, n: f64, step_m: f64, w: f64) -> ([f64; 3], f64) {
        if w <= 0.0 {
            return (base, 0.0);
        }
        let f = self.channels(e, n, step_m);
        let tw = self.lod.weights(step_m);
        let bi = DrapeClass::Building.index();
        let mut c = base;
        for (i, &a) in f[..NC].iter().enumerate() {
            let mut a = a.clamp(0.0, 1.0) * w;
            let k: [f64; 3] = if i == bi {
                a *= 1.0 - tw.proxy;
                let mean = self.palette.full[bi];
                // objects: coverage-weighted roof variants, outlines darkened
                let obj = if tw.objects > 0.0 && f[bi] > 1e-4 {
                    let edge = (f[EDGE] / f[bi]).clamp(0.0, 1.0);
                    let dark = 1.0 - (1.0 - self.lod.edge_dark) * edge;
                    [0, 1, 2].map(|j| (f[ROOF + j] / f[bi]).clamp(0.0, 1.0) * ROOF_SCALE * dark)
                } else {
                    mean
                };
                let m = tw.merged + tw.objects;
                if m <= 0.0 {
                    continue;
                }
                [0, 1, 2].map(|j| (mean[j] * tw.merged + obj[j] * tw.objects) / m)
            } else {
                let (p, q) = (self.palette.proxy[i], self.palette.full[i]);
                [0, 1, 2].map(|j| p[j] * tw.proxy + q[j] * (1.0 - tw.proxy))
            };
            if a > 0.0 {
                for j in 0..3 {
                    c[j] += (k[j] - c[j]) * a;
                }
            }
        }
        (c, f[DrapeClass::Water.index()].clamp(0.0, 1.0) * w)
    }
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Binary sub-sample rasteriser for one class (SS× the cell resolution), downsampled to coverage.
struct Raster {
    w: usize,
    h: usize,
    sw: usize,
    sh: usize,
    x0: f64,
    y0: f64,
    sub: f64,
    bits: Vec<u64>,
    /// Current feature's tag and, for the building layer, the last tag set per cell.
    tag: u8,
    owner: Option<Vec<u8>>,
}

impl Raster {
    fn new(w: usize, h: usize, x0: f64, y0: f64, cell: f64, owners: bool) -> Self {
        let (sw, sh) = (w * SS, h * SS);
        Raster {
            w,
            h,
            sw,
            sh,
            x0,
            y0,
            sub: cell / SS as f64,
            bits: vec![0; (sw * sh).div_ceil(64)],
            tag: 0,
            owner: owners.then(|| vec![0; w * h]),
        }
    }

    /// Building extras from the coverage + owner map: premultiplied roof R, G, B (variant of the
    /// cell's owning building) and outline coverage (partial cells, or owner differs from a
    /// 4-neighbour: adjacent roofs stay distinct).
    fn roof_channels(&self, cov: &[u8], roofs: &[[f64; 3]]) -> Vec<Vec<u8>> {
        let own = self.owner.as_deref().unwrap_or(&[]);
        let n = self.w * self.h;
        let mut out = vec![vec![0u8; n]; 4];
        for y in 0..self.h {
            for x in 0..self.w {
                let i = y * self.w + x;
                let (c, o) = (cov[i], own.get(i).copied().unwrap_or(0));
                if c == 0 || o == 0 {
                    continue;
                }
                let r = roofs[(o - 1) as usize % roofs.len()];
                for j in 0..3 {
                    out[j][i] = ((r[j] / ROOF_SCALE).clamp(0.0, 1.0) * c as f64).round() as u8;
                }
                let nb = |dx: i64, dy: i64| {
                    let (xx, yy) = (x as i64 + dx, y as i64 + dy);
                    if xx < 0 || yy < 0 || xx >= self.w as i64 || yy >= self.h as i64 {
                        return 0;
                    }
                    let k = yy as usize * self.w + xx as usize;
                    if cov[k] == 0 { 0 } else { own[k] }
                };
                let edge = c < 250 || [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().any(|&(dx, dy)| nb(dx, dy) != o);
                if edge {
                    out[3][i] = c;
                }
            }
        }
        out
    }

    fn to_sub(&self, p: [f64; 2]) -> (f64, f64) {
        ((p[0] - self.x0) / self.sub, (p[1] - self.y0) / self.sub)
    }

    /// Even-odd scanline fill of all rings together (sub-sample centres).
    fn area(&mut self, rings: &[Vec<[f64; 2]>]) {
        let pts: Vec<Vec<(f64, f64)>> = rings.iter().map(|r| r.iter().map(|&p| self.to_sub(p)).collect()).collect();
        let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);
        for r in &pts {
            for p in r {
                ymin = ymin.min(p.1);
                ymax = ymax.max(p.1);
            }
        }
        if !(ymax >= 0.0 && ymin < self.sh as f64) {
            return;
        }
        let ya = (ymin - 0.5).ceil().max(0.0) as usize;
        let yb = ((ymax - 0.5).floor() as i64).min(self.sh as i64 - 1);
        let mut xs = Vec::new();
        for y in ya as i64..=yb {
            let yc = y as f64 + 0.5;
            xs.clear();
            for r in &pts {
                let n = r.len();
                for i in 0..n {
                    let (a, b) = (r[i], r[(i + 1) % n]);
                    if (a.1 <= yc) != (b.1 <= yc) {
                        xs.push(a.0 + (yc - a.1) / (b.1 - a.1) * (b.0 - a.0));
                    }
                }
            }
            xs.sort_by(|a, b| a.total_cmp(b));
            let row = y as usize * self.sw;
            for pair in xs.as_chunks::<2>().0 {
                let xa = (pair[0] - 0.5).ceil().max(0.0) as i64;
                let xb = ((pair[1] - 0.5).floor() as i64).min(self.sw as i64 - 1);
                for x in xa..=xb {
                    self.set(row + x as usize);
                }
            }
        }
    }

    /// Capsules along the polyline (`width_m` wide).
    fn line(&mut self, pts: &[[f64; 2]], width_m: f64) {
        let r = (0.5 * width_m / self.sub).max(0.5);
        let p: Vec<(f64, f64)> = pts.iter().map(|&q| self.to_sub(q)).collect();
        if p.len() == 1 {
            self.capsule(p[0], p[0], r);
        }
        for s in p.windows(2) {
            self.capsule(s[0], s[1], r);
        }
    }

    fn capsule(&mut self, a: (f64, f64), b: (f64, f64), r: f64) {
        let xa = (a.0.min(b.0) - r).floor().max(0.0) as i64;
        let xb = ((a.0.max(b.0) + r).ceil() as i64).min(self.sw as i64 - 1);
        let ya = (a.1.min(b.1) - r).floor().max(0.0) as i64;
        let yb = ((a.1.max(b.1) + r).ceil() as i64).min(self.sh as i64 - 1);
        if xa > xb || ya > yb {
            return;
        }
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let l2 = dx * dx + dy * dy;
        let r2 = r * r;
        for y in ya..=yb {
            let py = y as f64 + 0.5;
            for x in xa..=xb {
                let px = x as f64 + 0.5;
                let t = if l2 > 0.0 { (((px - a.0) * dx + (py - a.1) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
                let (qx, qy) = (a.0 + dx * t - px, a.1 + dy * t - py);
                if qx * qx + qy * qy <= r2 {
                    self.set(y as usize * self.sw + x as usize);
                }
            }
        }
    }

    fn set(&mut self, i: usize) {
        self.bits[i >> 6] |= 1 << (i & 63);
        if let Some(o) = &mut self.owner {
            let (x, y) = (i % self.sw, i / self.sw);
            o[(y / SS) * self.w + x / SS] = self.tag;
        }
    }

    fn get(&self, i: usize) -> u32 {
        ((self.bits[i >> 6] >> (i & 63)) & 1) as u32
    }

    fn coverage(&self) -> Vec<u8> {
        let mut o = vec![0u8; self.w * self.h];
        let n = (SS * SS) as u32;
        for y in 0..self.h {
            for x in 0..self.w {
                let mut s = 0u32;
                for dy in 0..SS {
                    let row = (y * SS + dy) * self.sw + x * SS;
                    s += (row..row + SS).map(|i| self.get(i)).sum::<u32>();
                }
                o[y * self.w + x] = ((s * 255 + n / 2) / n) as u8;
            }
        }
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(x0: f64, y0: f64, s: f64) -> Vec<[f64; 2]> {
        vec![[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s], [x0, y0]]
    }

    #[test]
    fn area_coverage_matches_area() {
        let f = vec![DrapeFeature {
            class: DrapeClass::Building,
            shape: DrapeShape::Area { rings: vec![sq(10.0, 10.0, 20.0)] },
        }];
        let d = Drape::build(&f, (0.0, 0.0, 64.0, 64.0), 4.0);
        let b = DrapeClass::Building.index();
        assert!((d.fractions(20.0, 20.0, 1.0)[b] - 1.0).abs() < 1e-6);
        assert_eq!(d.fractions(50.0, 50.0, 1.0)[b], 0.0);
        // a 64 m footprint sees ~400 / 4096 of the 64 m square
        let f64m = d.fractions(32.0, 32.0, 64.0)[b];
        assert!((f64m - 400.0 / 4096.0).abs() < 0.03, "{f64m}");
    }

    #[test]
    fn holes_are_even_odd() {
        let f = vec![DrapeFeature {
            class: DrapeClass::Water,
            shape: DrapeShape::Area { rings: vec![sq(0.0, 0.0, 40.0), sq(12.0, 12.0, 16.0)] },
        }];
        let d = Drape::build(&f, (0.0, 0.0, 40.0, 40.0), 2.0);
        let w = DrapeClass::Water.index();
        assert!(d.fractions(20.0, 20.0, 1.0)[w] < 1e-6);
        assert!(d.fractions(5.0, 5.0, 1.0)[w] > 0.99);
    }

    #[test]
    fn line_width_sets_fraction() {
        let f = vec![DrapeFeature {
            class: DrapeClass::MajorRoad,
            shape: DrapeShape::Line { pts: vec![[0.0, 32.0], [64.0, 32.0]], width_m: 8.0 },
        }];
        let d = Drape::build(&f, (0.0, 0.0, 64.0, 64.0), 2.0);
        let r = DrapeClass::MajorRoad.index();
        assert!(d.fractions(32.0, 32.0, 1.0)[r] > 0.99);
        // 8 m of a 64 m footprint
        let v = d.fractions(32.0, 32.0, 64.0)[r];
        assert!((v - 0.125).abs() < 0.03, "{v}");
    }

    #[test]
    fn composite_paints_in_order() {
        let f = vec![
            DrapeFeature { class: DrapeClass::Grass, shape: DrapeShape::Area { rings: vec![sq(0.0, 0.0, 32.0)] } },
            DrapeFeature {
                class: DrapeClass::MajorRoad,
                shape: DrapeShape::Line { pts: vec![[0.0, 16.0], [32.0, 16.0]], width_m: 8.0 },
            },
        ];
        let d = Drape::build(&f, (0.0, 0.0, 32.0, 32.0), 1.0);
        let (c, wat) = d.composite([0.5; 3], 16.0, 16.0, 1.0, 1.0);
        assert_eq!(wat, 0.0);
        assert!((c[0] - DrapeClass::MajorRoad.albedo()[0]).abs() < 1e-3);
        let (c, _) = d.composite([0.5; 3], 16.0, 4.0, 1.0, 1.0);
        assert!((c[1] - DrapeClass::Grass.albedo()[1]).abs() < 1e-3);
        let (c, _) = d.composite([0.5; 3], 16.0, 4.0, 1.0, 0.0);
        assert_eq!(c, [0.5; 3]);
    }

    #[test]
    fn osm_classes() {
        assert_eq!(DrapeClass::from_osm("highway", "primary"), Some(DrapeClass::MajorRoad));
        assert_eq!(DrapeClass::from_osm("highway", "residential"), Some(DrapeClass::MinorRoad));
        assert_eq!(DrapeClass::from_osm("natural", "water"), Some(DrapeClass::Water));
        assert_eq!(DrapeClass::from_osm("natural", "coastline"), None);
        assert_eq!(DrapeClass::from_osm("landuse", "farmland"), Some(DrapeClass::Farm));
        assert_eq!(DrapeClass::from_osm("building", "yes"), Some(DrapeClass::Building));
    }

    #[test]
    fn lod_tiers_cross_fade() {
        let l = BuildingLod::default();
        let near = l.weights(1.0);
        assert!(near.objects > 0.99, "{near:?}");
        let far = l.weights(200.0);
        assert!(far.proxy > 0.99, "{far:?}");
        let mid = l.weights(l.building_m);
        assert!(mid.merged > 0.99, "{mid:?}");
        let mut prev = 0.0;
        for k in 0..200 {
            let step = 0.5 * 1.05f64.powi(k);
            let t = l.weights(step);
            assert!((t.proxy + t.merged + t.objects - 1.0).abs() < 1e-9);
            assert!(t.proxy >= prev - 1e-12, "proxy weight must grow with footprint");
            prev = t.proxy;
        }
    }

    #[test]
    fn roof_variants_average_to_mean() {
        let p = DrapePalette::adelaide_s2();
        let b = p.full[DrapeClass::Building.index()];
        let ws: f64 = p.roof_weights.iter().sum();
        for j in 0..3 {
            let m = p.roofs.iter().zip(p.roof_weights).map(|(r, w)| r[j] * w).sum::<f64>() / ws;
            assert!((m - b[j]).abs() < 1e-9);
        }
    }

    #[test]
    fn objects_tier_shows_per_roof_colour() {
        // 40 small roofs, objects tier: roofs differ; merged tier: all the mean
        let f: Vec<_> = (0..40)
            .map(|i| DrapeFeature {
                class: DrapeClass::Building,
                shape: DrapeShape::Area {
                    rings: vec![sq(4.0 + 16.0 * (i % 8) as f64, 4.0 + 16.0 * (i / 8) as f64, 10.0)],
                },
            })
            .collect();
        let d = Drape::build(&f, (0.0, 0.0, 128.0, 96.0), 1.0);
        let mut seen = std::collections::BTreeSet::new();
        for i in 0..40 {
            let (e, n) = (9.0 + 16.0 * (i % 8) as f64, 9.0 + 16.0 * (i / 8) as f64);
            let (c, _) = d.composite([0.0; 3], e, n, 1.0, 1.0);
            seen.insert((c[0] * 1000.0).round() as i64);
            let lod = BuildingLod { objects_px: 1e9, ..BuildingLod::default() };
            let (m, _) = d.clone().with_lod(lod).composite([0.0; 3], e, n, 1.0, 1.0);
            assert!((m[0] - DrapeClass::Building.albedo()[0]).abs() < 1e-3, "{m:?}");
        }
        assert!(seen.len() >= 4, "{seen:?}");
        // proxy tier: no building layer at all
        let (c, _) = d.composite([0.0; 3], 64.0, 48.0, 400.0, 1.0);
        assert_eq!(c, [0.0; 3]);
    }
}
