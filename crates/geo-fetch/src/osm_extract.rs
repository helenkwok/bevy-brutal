//! Decode a Geofabrik `.osm.pbf` into a few game-useful layers and write them as GeoJSON, in a
//! folder that can be shipped as-is under ODbL.
//!
//! Layers ([`Layer`]):
//!
//! | layer | OSM selection | geometry |
//! |---|---|---|
//! | `roads` | `highway=*` road/path classes (no platforms, stops, …) | LineString |
//! | `buildings` | `building=*` (≠ `no`), closed ways + `type=multipolygon` relations | Polygon / MultiPolygon |
//! | `water` | `natural=water`, `waterway=riverbank/dock`, `landuse=reservoir/basin` areas; `waterway=river/stream/canal/drain/ditch` and `natural=coastline` lines | Polygon / MultiPolygon / LineString |
//! | `landuse` | `landuse=*` (not water), parks/pitches/… (`leisure`), vegetation and bare ground (`natural`) | Polygon / MultiPolygon |
//!
//! Only a whitelist of tags is kept per layer (names, road class/lanes/bridges, building heights
//! and levels, …), never user names or edit metadata.
//!
//! Reading uses the `osmpbf` crate (MIT OR Apache-2.0) in three parallel passes over the file:
//! relations (multipolygons and their member ways), ways (tagged ways and member ways, collecting
//! node ids), then nodes (coordinates of exactly those ids). Multipolygon rings are assembled
//! from member ways by matching endpoint node ids; inners are assigned to the smallest containing
//! outer. Winding follows RFC 7946 (outer CCW, holes CW).
//!
//! Output licence: the GeoJSON files are an OSM **derived database** (ODbL 1.0). [`write_layers`]
//! puts `ATTRIBUTION.txt` and `LICENSE.txt` (the ODbL notice) next to them; ship the folder whole.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use coords::{Enu, Geodetic};
use osmpbf::{BlobDecode, BlobReader, RelMemberType};
use rayon::prelude::*;

use crate::cache::{create_parent, io_err, unix_now, write_atomic};
use crate::{BBox, Error, Result, attribution};

/// Output layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Layer {
    Roads,
    Buildings,
    Water,
    Landuse,
}

impl Layer {
    pub const ALL: [Layer; 4] = [Layer::Roads, Layer::Buildings, Layer::Water, Layer::Landuse];

    pub fn name(self) -> &'static str {
        match self {
            Layer::Roads => "roads",
            Layer::Buildings => "buildings",
            Layer::Water => "water",
            Layer::Landuse => "landuse",
        }
    }

    /// Tags copied into GeoJSON properties for this layer (besides `@id`, `key`, `kind`).
    pub fn kept_tags(self) -> &'static [&'static str] {
        match self {
            Layer::Roads => &[
                "name", "ref", "oneway", "lanes", "maxspeed", "surface", "bridge", "tunnel", "layer", "width",
                "service", "access", "junction",
            ],
            Layer::Buildings => &[
                "name",
                "height",
                "min_height",
                "building:levels",
                "building:min_level",
                "roof:shape",
                "roof:height",
                "roof:levels",
                "roof:colour",
                "building:colour",
                "building:material",
            ],
            Layer::Water => &["name", "water", "intermittent", "width", "tunnel", "layer"],
            Layer::Landuse => &["name"],
        }
    }
}

impl std::str::FromStr for Layer {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Layer::ALL
            .into_iter()
            .find(|l| l.name() == s)
            .ok_or_else(|| Error::BadArg(format!("unknown layer {s:?} (roads|buildings|water|landuse)")))
    }
}

/// Keys needed to classify features (always read, even if not kept).
const CLASS_KEYS: &[&str] = &["highway", "building", "natural", "waterway", "landuse", "leisure", "area", "type"];

const ROAD_CLASSES: &[&str] = &[
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
    "living_street",
    "service",
    "pedestrian",
    "track",
    "busway",
    "bus_guideway",
    "road",
    "footway",
    "cycleway",
    "bridleway",
    "path",
    "steps",
    "corridor",
];
const WATER_LINES: &[&str] = &["river", "stream", "canal", "drain", "ditch", "tidal_channel"];
const LANDUSE_LEISURE: &[&str] = &[
    "park",
    "garden",
    "pitch",
    "golf_course",
    "nature_reserve",
    "playground",
    "recreation_ground",
    "sports_centre",
    "stadium",
    "track",
    "dog_park",
    "common",
    "marina",
];
const LANDUSE_NATURAL: &[&str] = &[
    "wood",
    "scrub",
    "grassland",
    "heath",
    "wetland",
    "beach",
    "sand",
    "bare_rock",
    "scree",
    "shingle",
    "glacier",
    "mud",
    "fell",
];

/// One classification of an OSM object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Class {
    pub layer: Layer,
    /// The OSM key that selected it (`highway`, `building`, `natural`, …).
    pub key: &'static str,
    /// Its value (`residential`, `yes`, `water`, …).
    pub kind: String,
    /// True for area features (need a closed way or a multipolygon).
    pub area: bool,
}

fn tag<'a>(tags: &'a [(String, String)], k: &str) -> Option<&'a str> {
    tags.iter().find(|(kk, _)| kk == k).map(|(_, v)| v.as_str())
}

/// Classify an object by its tags. An object can land in more than one layer.
pub fn classify(tags: &[(String, String)]) -> Vec<Class> {
    let mut out = Vec::new();
    let mut push =
        |layer, key: &'static str, kind: &str, area| out.push(Class { layer, key, kind: kind.to_owned(), area });
    if let Some(h) = tag(tags, "highway")
        && ROAD_CLASSES.contains(&h)
        && tag(tags, "area") != Some("yes")
    {
        push(Layer::Roads, "highway", h, false);
    }
    if let Some(b) = tag(tags, "building")
        && b != "no"
    {
        push(Layer::Buildings, "building", b, true);
    }
    match (tag(tags, "natural"), tag(tags, "waterway"), tag(tags, "landuse")) {
        (Some("water"), _, _) => push(Layer::Water, "natural", "water", true),
        (Some("coastline"), _, _) => push(Layer::Water, "natural", "coastline", false),
        (_, Some(w @ ("riverbank" | "dock")), _) => push(Layer::Water, "waterway", w, true),
        (_, Some(w), _) if WATER_LINES.contains(&w) => push(Layer::Water, "waterway", w, false),
        (_, _, Some(l @ ("reservoir" | "basin"))) => push(Layer::Water, "landuse", l, true),
        _ => {}
    }
    if let Some(l) = tag(tags, "landuse")
        && !matches!(l, "reservoir" | "basin")
    {
        push(Layer::Landuse, "landuse", l, true);
    } else if let Some(l) = tag(tags, "leisure")
        && LANDUSE_LEISURE.contains(&l)
    {
        push(Layer::Landuse, "leisure", l, true);
    } else if let Some(n) = tag(tags, "natural")
        && LANDUSE_NATURAL.contains(&n)
    {
        push(Layer::Landuse, "natural", n, true);
    }
    out
}

fn wanted_key(k: &str) -> bool {
    CLASS_KEYS.contains(&k) || Layer::ALL.iter().any(|l| l.kept_tags().contains(&k))
}

/// Lon/lat pair, degrees.
pub type LonLat = [f64; 2];

#[derive(Clone, Debug, PartialEq)]
pub enum Geometry {
    LineString(Vec<LonLat>),
    /// Rings: outer first (CCW), then holes (CW). Rings are closed (first == last).
    Polygon(Vec<Vec<LonLat>>),
    MultiPolygon(Vec<Vec<Vec<LonLat>>>),
}

impl Geometry {
    fn points(&self) -> Box<dyn Iterator<Item = &LonLat> + '_> {
        match self {
            Geometry::LineString(l) => Box::new(l.iter()),
            Geometry::Polygon(p) => Box::new(p.iter().flatten()),
            Geometry::MultiPolygon(m) => Box::new(m.iter().flatten().flatten()),
        }
    }

    /// `(west, south, east, north)`.
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        self.points().fold((f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY), |b, p| {
            (b.0.min(p[0]), b.1.min(p[1]), b.2.max(p[0]), b.3.max(p[1]))
        })
    }

    pub fn intersects(&self, bb: &BBox) -> bool {
        let (w, s, e, n) = self.bounds();
        w <= bb.east && e >= bb.west && s <= bb.north && n >= bb.south
    }

    /// Same shape with every vertex mapped to ENU `[east, north]` metres (on the ellipsoid, h = 0).
    pub fn to_enu(&self, enu: &Enu) -> Geometry {
        let f = |p: &LonLat| {
            let v = enu.geodetic_to_enu(Geodetic::new(p[1], p[0], 0.0));
            [v.x, v.y]
        };
        let ring = |r: &Vec<LonLat>| r.iter().map(f).collect::<Vec<_>>();
        match self {
            Geometry::LineString(l) => Geometry::LineString(ring(l)),
            Geometry::Polygon(p) => Geometry::Polygon(p.iter().map(ring).collect()),
            Geometry::MultiPolygon(m) => {
                Geometry::MultiPolygon(m.iter().map(|p| p.iter().map(ring).collect()).collect())
            }
        }
    }

    fn write_json(&self, out: &mut impl Write) -> std::io::Result<()> {
        fn pt(out: &mut impl Write, p: &LonLat) -> std::io::Result<()> {
            // 7 decimals = OSM's own 100 nanodegree precision.
            write!(out, "[{},{}]", round7(p[0]), round7(p[1]))
        }
        fn line(out: &mut impl Write, l: &[LonLat]) -> std::io::Result<()> {
            out.write_all(b"[")?;
            for (i, p) in l.iter().enumerate() {
                if i > 0 {
                    out.write_all(b",")?;
                }
                pt(out, p)?;
            }
            out.write_all(b"]")
        }
        fn poly(out: &mut impl Write, p: &[Vec<LonLat>]) -> std::io::Result<()> {
            out.write_all(b"[")?;
            for (i, r) in p.iter().enumerate() {
                if i > 0 {
                    out.write_all(b",")?;
                }
                line(out, r)?;
            }
            out.write_all(b"]")
        }
        match self {
            Geometry::LineString(l) => {
                out.write_all(br#"{"type":"LineString","coordinates":"#)?;
                line(out, l)?;
            }
            Geometry::Polygon(p) => {
                out.write_all(br#"{"type":"Polygon","coordinates":"#)?;
                poly(out, p)?;
            }
            Geometry::MultiPolygon(m) => {
                out.write_all(br#"{"type":"MultiPolygon","coordinates":["#)?;
                for (i, p) in m.iter().enumerate() {
                    if i > 0 {
                        out.write_all(b",")?;
                    }
                    poly(out, p)?;
                }
                out.write_all(b"]")?;
            }
        }
        out.write_all(b"}")
    }
}

fn round7(v: f64) -> f64 {
    (v * 1e7).round() / 1e7
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OsmType {
    Way,
    Relation,
}

/// One output feature.
#[derive(Clone, Debug, PartialEq)]
pub struct Feature {
    pub layer: Layer,
    pub osm_type: OsmType,
    pub id: i64,
    pub key: &'static str,
    pub kind: String,
    /// Whitelisted tags for the layer, sorted by key.
    pub tags: Vec<(String, String)>,
    pub geometry: Geometry,
}

impl Feature {
    pub fn osm_ref(&self) -> String {
        match self.osm_type {
            OsmType::Way => format!("way/{}", self.id),
            OsmType::Relation => format!("relation/{}", self.id),
        }
    }
}

/// OSM tags as `(key, value)` pairs.
pub type Tags = Vec<(String, String)>;

/// A multipolygon relation: `(id, tags, members as (way id, is_inner))`.
pub type Multipolygon = (i64, Tags, Vec<(i64, bool)>);

/// Raw objects needed for assembly, independent of the file format (tests build this directly).
#[derive(Clone, Debug, Default)]
pub struct OsmData {
    pub nodes: HashMap<i64, LonLat>,
    /// Node refs of every retained way (tagged ways and multipolygon members).
    pub way_refs: HashMap<i64, Vec<i64>>,
    /// Tagged ways: `(id, tags)` (only classification keys and `Layer::kept_tags` are retained).
    pub ways: Vec<(i64, Vec<(String, String)>)>,
    /// Multipolygon relations: `(id, tags, members as (way id, is_inner))`.
    pub relations: Vec<Multipolygon>,
}

/// Header facts worth recording in the manifest.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PbfHeader {
    pub replication_timestamp: Option<i64>,
    pub writing_program: Option<String>,
    pub source: Option<String>,
}

/// Counters for logs and the manifest.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExtractReport {
    pub header: PbfHeader,
    pub counts: Vec<(Layer, usize)>,
    pub ways_read: usize,
    pub relations_read: usize,
    pub nodes_needed: usize,
    /// Ways/relations dropped because a referenced node or member way is not in the file.
    pub incomplete: usize,
    /// Area ways that are not closed, or relations whose rings did not close.
    pub unclosed: usize,
    /// Dropped by the bbox filter.
    pub outside_bbox: usize,
}

/// What to extract.
#[derive(Clone, Debug)]
pub struct ExtractSpec {
    /// Keep features whose bounds intersect this box (no geometric clipping).
    pub bbox: Option<BBox>,
    pub layers: Vec<Layer>,
}

impl Default for ExtractSpec {
    fn default() -> Self {
        Self { bbox: None, layers: Layer::ALL.to_vec() }
    }
}

fn pbf_err(path: &Path) -> impl Fn(osmpbf::Error) -> Error + '_ {
    move |e| Error::Decode(format!("{}: {e}", path.display()))
}

fn keep_tags<'a>(it: impl Iterator<Item = (&'a str, &'a str)>) -> Vec<(String, String)> {
    it.filter(|(k, _)| wanted_key(k)).map(|(k, v)| (k.to_owned(), v.to_owned())).collect()
}

/// Run `f` on every primitive block of the file in parallel and concatenate the results.
fn par_blocks<T: Send>(
    path: &Path,
    f: impl Fn(&osmpbf::PrimitiveBlock) -> Vec<T> + Sync + Send,
) -> Result<(Vec<T>, Option<PbfHeader>)> {
    let reader = BlobReader::from_path(path).map_err(pbf_err(path))?;
    let parts: Vec<Result<(Vec<T>, Option<PbfHeader>)>> = reader
        .par_bridge()
        .map(|blob| {
            let blob = blob.map_err(pbf_err(path))?;
            Ok(match blob.decode().map_err(pbf_err(path))? {
                BlobDecode::OsmData(block) => (f(&block), None),
                BlobDecode::OsmHeader(h) => (
                    Vec::new(),
                    Some(PbfHeader {
                        replication_timestamp: h.osmosis_replication_timestamp(),
                        writing_program: h.writing_program().map(str::to_owned),
                        source: h.source().map(str::to_owned),
                    }),
                ),
                BlobDecode::Unknown(_) => (Vec::new(), None),
            })
        })
        .collect();
    let (mut all, mut header) = (Vec::new(), None);
    for p in parts {
        let (v, h) = p?;
        all.extend(v);
        header = header.or(h);
    }
    Ok((all, header))
}

/// Read the objects needed for `layers` from a `.osm.pbf`.
pub fn read_pbf(path: &Path, layers: &[Layer]) -> Result<(OsmData, PbfHeader)> {
    let want = |tags: &[(String, String)]| classify(tags).iter().any(|c| layers.contains(&c.layer));

    // Pass 1: multipolygon relations of interest.
    let (relations, header) = par_blocks(path, |b| {
        let mut v = Vec::new();
        for g in b.groups() {
            for r in g.relations() {
                let tags = keep_tags(r.tags());
                if tag(&tags, "type") != Some("multipolygon") || !want(&tags) {
                    continue;
                }
                let members: Vec<(i64, bool)> = r
                    .members()
                    .filter(|m| m.member_type == RelMemberType::Way)
                    .map(|m| (m.member_id, m.role().map(|s| s == "inner").unwrap_or(false)))
                    .collect();
                v.push((r.id(), tags, members));
            }
        }
        v
    })?;
    let member_ids: HashSet<i64> = relations.iter().flat_map(|r| r.2.iter().map(|m| m.0)).collect();

    // Pass 2: tagged ways of interest and member ways.
    type WayRow = (i64, Option<Vec<(String, String)>>, Vec<i64>);
    let (rows, _) = par_blocks(path, |b| {
        let mut v: Vec<WayRow> = Vec::new();
        for g in b.groups() {
            for w in g.ways() {
                let tags = keep_tags(w.tags());
                let tagged = want(&tags);
                if tagged || member_ids.contains(&w.id()) {
                    v.push((w.id(), tagged.then_some(tags), w.refs().collect()));
                }
            }
        }
        v
    })?;
    let mut data = OsmData { relations, ..Default::default() };
    for (id, tags, refs) in rows {
        if let Some(t) = tags {
            data.ways.push((id, t));
        }
        data.way_refs.insert(id, refs);
    }
    let needed: HashSet<i64> = data.way_refs.values().flatten().copied().collect();

    // Pass 3: coordinates of the referenced nodes (plain and dense).
    let (nodes, _) = par_blocks(path, |b| {
        let mut v = Vec::new();
        for g in b.groups() {
            for n in g.dense_nodes() {
                if needed.contains(&n.id()) {
                    v.push((n.id(), [n.lon(), n.lat()]));
                }
            }
            for n in g.nodes() {
                if needed.contains(&n.id()) {
                    v.push((n.id(), [n.lon(), n.lat()]));
                }
            }
        }
        v
    })?;
    data.nodes = nodes.into_iter().collect();
    data.ways.sort_by_key(|w| w.0);
    data.relations.sort_by_key(|r| r.0);
    Ok((data, header.unwrap_or_default()))
}

/// Join way node lists into closed rings by matching endpoints. Returns rings and the number of
/// pieces that could not be closed.
pub fn assemble_rings(mut segs: Vec<Vec<i64>>) -> (Vec<Vec<i64>>, usize) {
    segs.retain(|s| s.len() >= 2);
    let (mut rings, mut open) = (Vec::new(), 0);
    while let Some(mut cur) = segs.pop() {
        loop {
            if cur.len() >= 4 && cur.first() == cur.last() {
                rings.push(cur);
                break;
            }
            let (first, last) = (cur[0], *cur.last().unwrap());
            let Some(j) = segs.iter().position(|s| {
                s[0] == last || *s.last().unwrap() == last || s[0] == first || *s.last().unwrap() == first
            }) else {
                open += 1;
                break;
            };
            let mut s = segs.swap_remove(j);
            if s[0] == last {
                cur.extend_from_slice(&s[1..]);
            } else if *s.last().unwrap() == last {
                s.reverse();
                cur.extend_from_slice(&s[1..]);
            } else if *s.last().unwrap() == first {
                s.pop();
                s.extend_from_slice(&cur);
                cur = s;
            } else {
                s.reverse();
                s.pop();
                s.extend_from_slice(&cur);
                cur = s;
            }
        }
    }
    (rings, open)
}

/// Twice the signed area (shoelace, lon/lat degrees): positive = counter-clockwise.
fn signed_area2(r: &[LonLat]) -> f64 {
    r.windows(2).map(|w| w[0][0] * w[1][1] - w[1][0] * w[0][1]).sum()
}

fn orient(mut r: Vec<LonLat>, ccw: bool) -> Vec<LonLat> {
    if (signed_area2(&r) > 0.0) != ccw {
        r.reverse();
    }
    r
}

fn point_in_ring(p: LonLat, r: &[LonLat]) -> bool {
    let mut inside = false;
    for w in r.windows(2) {
        let (a, b) = (w[0], w[1]);
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0] {
            inside = !inside;
        }
    }
    inside
}

fn coords_of(data: &OsmData, refs: &[i64]) -> Option<Vec<LonLat>> {
    refs.iter().map(|id| data.nodes.get(id).copied()).collect()
}

/// Build features from raw objects.
pub fn build_features(data: &OsmData, spec: &ExtractSpec) -> (Vec<Feature>, ExtractReport) {
    let mut rep =
        ExtractReport { ways_read: data.ways.len(), relations_read: data.relations.len(), ..Default::default() };
    rep.nodes_needed = data.nodes.len();
    let mut out = Vec::new();
    let kept = |layer: Layer, tags: &[(String, String)]| {
        let mut t: Vec<(String, String)> =
            tags.iter().filter(|(k, _)| layer.kept_tags().contains(&k.as_str())).cloned().collect();
        t.sort();
        t
    };

    for (id, tags) in &data.ways {
        let classes: Vec<Class> = classify(tags).into_iter().filter(|c| spec.layers.contains(&c.layer)).collect();
        if classes.is_empty() {
            continue;
        }
        let Some(pts) = data.way_refs.get(id).and_then(|r| coords_of(data, r)) else {
            rep.incomplete += 1;
            continue;
        };
        let closed = pts.len() >= 4 && pts.first() == pts.last();
        for c in classes {
            let geometry = if c.area {
                if !closed {
                    rep.unclosed += 1;
                    continue;
                }
                Geometry::Polygon(vec![orient(pts.clone(), true)])
            } else if pts.len() >= 2 {
                Geometry::LineString(pts.clone())
            } else {
                continue;
            };
            out.push(Feature {
                layer: c.layer,
                osm_type: OsmType::Way,
                id: *id,
                key: c.key,
                kind: c.kind,
                tags: kept(c.layer, tags),
                geometry,
            });
        }
    }

    for (id, tags, members) in &data.relations {
        let classes: Vec<Class> =
            classify(tags).into_iter().filter(|c| c.area && spec.layers.contains(&c.layer)).collect();
        if classes.is_empty() {
            continue;
        }
        let (mut outer, mut inner) = (Vec::new(), Vec::new());
        let mut complete = true;
        for &(wid, is_inner) in members {
            match data.way_refs.get(&wid) {
                Some(r) => (if is_inner { &mut inner } else { &mut outer }).push(r.clone()),
                None => complete = false,
            }
        }
        let (outer_rings, open_o) = assemble_rings(outer);
        let (inner_rings, open_i) = assemble_rings(inner);
        let to_pts = |rs: Vec<Vec<i64>>| rs.iter().map(|r| coords_of(data, r)).collect::<Option<Vec<_>>>();
        let (Some(outers), Some(inners)) = (to_pts(outer_rings), to_pts(inner_rings)) else {
            rep.incomplete += 1;
            continue;
        };
        if !complete {
            // Members outside the extract: keep what closed, but count it.
            rep.incomplete += 1;
        }
        if open_o + open_i > 0 {
            rep.unclosed += 1;
        }
        if outers.is_empty() {
            continue;
        }
        let mut polys: Vec<Vec<Vec<LonLat>>> = outers.into_iter().map(|r| vec![orient(r, true)]).collect();
        for hole in inners {
            // Smallest outer containing the hole's first vertex.
            let best = polys
                .iter()
                .enumerate()
                .filter(|(_, p)| point_in_ring(hole[0], &p[0]))
                .min_by(|a, b| signed_area2(&a.1[0]).abs().total_cmp(&signed_area2(&b.1[0]).abs()))
                .map(|(i, _)| i);
            if let Some(i) = best {
                polys[i].push(orient(hole, false));
            }
        }
        let geometry =
            if polys.len() == 1 { Geometry::Polygon(polys.pop().unwrap()) } else { Geometry::MultiPolygon(polys) };
        for c in classes {
            out.push(Feature {
                layer: c.layer,
                osm_type: OsmType::Relation,
                id: *id,
                key: c.key,
                kind: c.kind,
                tags: kept(c.layer, tags),
                geometry: geometry.clone(),
            });
        }
    }

    if let Some(bb) = &spec.bbox {
        let before = out.len();
        out.retain(|f| f.geometry.intersects(bb));
        rep.outside_bbox = before - out.len();
    }
    out.sort_by_key(|f| (f.layer, f.osm_type, f.id));
    rep.counts = spec.layers.iter().map(|&l| (l, out.iter().filter(|f| f.layer == l).count())).collect();
    (out, rep)
}

/// Read a `.osm.pbf` and build features in one go.
pub fn extract(pbf: &Path, spec: &ExtractSpec) -> Result<(Vec<Feature>, ExtractReport)> {
    let (data, header) = read_pbf(pbf, &spec.layers)?;
    let (f, mut rep) = build_features(&data, spec);
    rep.header = header;
    Ok((f, rep))
}

/// Where the source extract came from, for the manifest.
#[derive(Clone, Debug, Default)]
pub struct SourceInfo {
    pub pbf: PathBuf,
    pub url: Option<String>,
    pub md5: Option<String>,
}

/// Files written by [`write_layers`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerFiles {
    pub dir: PathBuf,
    pub layers: Vec<(Layer, PathBuf)>,
    pub manifest: PathBuf,
    pub attribution: PathBuf,
    pub license: PathBuf,
}

/// Default output directory for derived layers: `<cache>/osm/derived/<name>`.
pub fn derived_dir(cache: &crate::CacheDir, name: &str) -> PathBuf {
    cache.osm_dir().join("derived").join(name)
}

/// Text of `LICENSE.txt` for a derived-layers folder.
pub fn license_text() -> String {
    format!(
        "{}\n\nFull licence text: https://opendatacommons.org/licenses/odbl/1-0/\n\
         Summary: you may share, create and adapt this database, provided you attribute it \
         (\"{}\"), offer any adapted database you distribute under the ODbL (share-alike), and do \
         not apply technical restrictions that prevent others from using it. Images, meshes and \
         other Produced Works made from it need attribution only.\n",
        attribution::OSM_DERIVED_DB_NOTICE,
        attribution::OSM_SHORT
    )
}

/// Write one GeoJSON file per layer plus `manifest.json`, `ATTRIBUTION.txt` and `LICENSE.txt`.
pub fn write_layers(
    features: &[Feature],
    rep: &ExtractReport,
    spec: &ExtractSpec,
    src: &SourceInfo,
    dir: &Path,
) -> Result<LayerFiles> {
    let mut files = LayerFiles {
        dir: dir.to_path_buf(),
        layers: Vec::new(),
        manifest: dir.join("manifest.json"),
        attribution: dir.join("ATTRIBUTION.txt"),
        license: dir.join("LICENSE.txt"),
    };
    for &layer in &spec.layers {
        let path = dir.join(format!("{}.geojson", layer.name()));
        write_geojson(&path, features.iter().filter(|f| f.layer == layer))?;
        files.layers.push((layer, path));
    }
    let origin = spec.bbox.map(|b| [(b.south + b.north) / 2.0, (b.west + b.east) / 2.0]);
    let manifest = serde_json::json!({
        "format": "geo-fetch OSM derived layers v1",
        "generator": crate::USER_AGENT,
        "generated_unix": unix_now(),
        "licence": "ODbL-1.0",
        "attribution": attribution::OSM_SHORT,
        "source": {
            "pbf": src.pbf.display().to_string(),
            "url": src.url,
            "md5": src.md5,
            "osm_replication_timestamp_unix": rep.header.replication_timestamp,
            "writing_program": rep.header.writing_program,
        },
        "bbox": spec.bbox.map(|b| [b.west, b.south, b.east, b.north]),
        "bbox_rule": "features whose bounds intersect the bbox are kept whole (no clipping)",
        "suggested_enu_origin_lat_lon": origin,
        "enu_note": "terrain-grid with the same --bbox uses the bbox centre as its ENU origin; project with coords::Enu",
        "coordinates": "WGS84 lon/lat degrees (RFC 7946), 7 decimals",
        "layers": files.layers.iter().map(|(l, p)| serde_json::json!({
            "name": l.name(),
            "file": p.file_name().map(|n| n.to_string_lossy().into_owned()),
            "features": rep.counts.iter().find(|c| c.0 == *l).map(|c| c.1).unwrap_or(0),
            "kept_tags": l.kept_tags(),
        })).collect::<Vec<_>>(),
        "stats": {
            "ways_read": rep.ways_read,
            "relations_read": rep.relations_read,
            "nodes": rep.nodes_needed,
            "incomplete": rep.incomplete,
            "unclosed": rep.unclosed,
            "outside_bbox": rep.outside_bbox,
        },
    });
    let mut json = serde_json::to_vec_pretty(&manifest).map_err(|e| Error::Decode(e.to_string()))?;
    json.push(b'\n');
    write_atomic(&files.manifest, &json)?;
    let attr = format!(
        "{}\n{}\n\n{}\n",
        attribution::OSM_FULL,
        attribution::GEOFABRIK_SOURCE,
        attribution::OSM_DERIVED_DB_NOTICE
    );
    write_atomic(&files.attribution, attr.as_bytes())?;
    write_atomic(&files.license, license_text().as_bytes())?;
    Ok(files)
}

/// Write a GeoJSON FeatureCollection, one feature per line.
pub fn write_geojson<'a>(path: &Path, features: impl Iterator<Item = &'a Feature>) -> Result<()> {
    create_parent(path)?;
    let tmp = crate::cache::part_path(path);
    let f = fs::File::create(&tmp).map_err(io_err(&tmp))?;
    let mut w = BufWriter::new(f);
    let res = (|| -> std::io::Result<()> {
        w.write_all(b"{\"type\":\"FeatureCollection\",\"features\":[\n")?;
        for (i, f) in features.enumerate() {
            if i > 0 {
                w.write_all(b",\n")?;
            }
            let mut props = serde_json::Map::new();
            props.insert("@id".into(), f.osm_ref().into());
            props.insert("key".into(), f.key.into());
            props.insert("kind".into(), f.kind.clone().into());
            for (k, v) in &f.tags {
                props.insert(k.clone(), v.clone().into());
            }
            write!(w, "{{\"type\":\"Feature\",\"id\":\"{}\",\"properties\":", f.osm_ref())?;
            serde_json::to_writer(&mut w, &props)?;
            w.write_all(b",\"geometry\":")?;
            f.geometry.write_json(&mut w)?;
            w.write_all(b"}")?;
        }
        w.write_all(b"\n]}\n")?;
        w.flush()
    })();
    res.map_err(io_err(&tmp))?;
    drop(w);
    fs::rename(&tmp, path).map_err(io_err(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(kv: &[(&str, &str)]) -> Vec<(String, String)> {
        kv.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn classification() {
        let c = classify(&t(&[("highway", "residential"), ("name", "King William St")]));
        assert_eq!(c, vec![Class { layer: Layer::Roads, key: "highway", kind: "residential".into(), area: false }]);
        assert!(classify(&t(&[("highway", "bus_stop")])).is_empty());
        assert!(classify(&t(&[("highway", "pedestrian"), ("area", "yes")])).is_empty());
        assert!(classify(&t(&[("building", "no")])).is_empty());
        assert_eq!(classify(&t(&[("building", "house")]))[0].layer, Layer::Buildings);
        let w = classify(&t(&[("natural", "water"), ("water", "lake")]));
        assert_eq!((w[0].layer, w[0].area), (Layer::Water, true));
        let r = classify(&t(&[("waterway", "river")]));
        assert_eq!((r[0].layer, r[0].area), (Layer::Water, false));
        let res = classify(&t(&[("landuse", "reservoir")]));
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].layer, Layer::Water);
        assert_eq!(classify(&t(&[("leisure", "park")]))[0].layer, Layer::Landuse);
        assert_eq!(classify(&t(&[("natural", "wood")]))[0].layer, Layer::Landuse);
        // building inside a landuse polygon tag set -> both layers
        let both = classify(&t(&[("building", "yes"), ("landuse", "retail")]));
        assert_eq!(both.iter().map(|c| c.layer).collect::<Vec<_>>(), vec![Layer::Buildings, Layer::Landuse]);
        assert_eq!("water".parse::<Layer>().unwrap(), Layer::Water);
        assert!("google".parse::<Layer>().is_err());
    }

    #[test]
    fn ring_assembly() {
        // Square split into three ways, one reversed.
        let (r, open) = assemble_rings(vec![vec![1, 2], vec![3, 2], vec![3, 4, 1]]);
        assert_eq!(open, 0);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].len(), 5);
        assert_eq!(r[0].first(), r[0].last());
        let mut ids = r[0][..4].to_vec();
        ids.sort();
        assert_eq!(ids, vec![1, 2, 3, 4]);
        // Two separate closed rings + one dangling piece.
        let (r, open) = assemble_rings(vec![vec![1, 2, 3, 1], vec![5, 6, 7, 5], vec![8, 9]]);
        assert_eq!((r.len(), open), (2, 1));
    }

    /// Small world: a road, a building with a courtyard (multipolygon), a lake way, a park, and
    /// a road whose node is missing.
    fn sample() -> OsmData {
        let mut d = OsmData::default();
        let pts: &[(i64, f64, f64)] = &[
            (1, 138.600, -34.900),
            (2, 138.601, -34.900),
            (3, 138.601, -34.901),
            (4, 138.600, -34.901),
            (5, 138.6002, -34.9002),
            (6, 138.6002, -34.9008),
            (7, 138.6008, -34.9008),
            (8, 138.6008, -34.9002),
            (10, 138.590, -34.890),
            (11, 138.610, -34.890),
            (20, 140.0, -30.0),
            (21, 140.01, -30.0),
            (22, 140.01, -30.01),
            (23, 140.0, -30.0),
        ];
        for &(id, lon, lat) in pts {
            d.nodes.insert(id, [lon, lat]);
        }
        d.way_refs.insert(100, vec![10, 11]);
        d.ways.push((100, t(&[("highway", "primary"), ("name", "North Tce"), ("lanes", "4"), ("fixme", "x")])));
        // multipolygon members
        d.way_refs.insert(200, vec![1, 2, 3]);
        d.way_refs.insert(201, vec![3, 4, 1]);
        d.way_refs.insert(202, vec![5, 6, 7, 8, 5]);
        d.relations.push((
            300,
            t(&[("type", "multipolygon"), ("building", "university"), ("building:levels", "3")]),
            vec![(200, false), (201, false), (202, true)],
        ));
        // lake far away (outside the bbox used below)
        d.way_refs.insert(400, vec![20, 21, 22, 23]);
        d.ways.push((400, t(&[("natural", "water"), ("water", "lake")])));
        // unclosed park
        d.way_refs.insert(500, vec![1, 2, 3]);
        d.ways.push((500, t(&[("leisure", "park")])));
        // road with a missing node
        d.way_refs.insert(600, vec![10, 999]);
        d.ways.push((600, t(&[("highway", "service")])));
        d
    }

    #[test]
    fn features_from_data() {
        let d = sample();
        let (f, rep) = build_features(&d, &ExtractSpec::default());
        assert_eq!(rep.incomplete, 1);
        assert_eq!(rep.unclosed, 1);
        let road = f.iter().find(|x| x.id == 100).unwrap();
        assert_eq!(road.tags, t(&[("lanes", "4"), ("name", "North Tce")]), "fixme dropped, sorted");
        let b = f.iter().find(|x| x.id == 300).unwrap();
        assert_eq!((b.layer, b.osm_type, b.kind.as_str()), (Layer::Buildings, OsmType::Relation, "university"));
        let Geometry::Polygon(rings) = &b.geometry else { panic!("{:?}", b.geometry) };
        assert_eq!(rings.len(), 2, "outer + courtyard");
        assert!(signed_area2(&rings[0]) > 0.0 && signed_area2(&rings[1]) < 0.0, "RFC 7946 winding");
        let lake = f.iter().find(|x| x.id == 400).unwrap();
        assert_eq!(lake.layer, Layer::Water);

        let bb: BBox = "138.5,-35.0,138.7,-34.8".parse().unwrap();
        let (f2, rep2) = build_features(&d, &ExtractSpec { bbox: Some(bb), layers: vec![Layer::Roads, Layer::Water] });
        assert_eq!(rep2.outside_bbox, 1);
        assert_eq!(f2.iter().map(|x| x.id).collect::<Vec<_>>(), vec![100]);
        assert_eq!(rep2.counts, vec![(Layer::Roads, 1), (Layer::Water, 0)]);

        let enu = Enu::new(Geodetic::new(-34.9, 138.6, 0.0));
        let Geometry::LineString(l) = road.geometry.to_enu(&enu) else { panic!() };
        assert!((l[0][0] - -914.0).abs() < 2.0 && (l[0][1] - 1109.0).abs() < 2.0, "{l:?}");
    }

    #[test]
    fn writes_valid_geojson_and_licence_files() {
        let d = sample();
        let spec = ExtractSpec::default();
        let (f, rep) = build_features(&d, &spec);
        let dir = std::env::temp_dir().join(format!("geo-fetch-osmx-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let files =
            write_layers(&f, &rep, &spec, &SourceInfo { pbf: "x.osm.pbf".into(), ..Default::default() }, &dir).unwrap();
        for (layer, p) in &files.layers {
            let v: serde_json::Value = serde_json::from_slice(&fs::read(p).unwrap()).unwrap();
            assert_eq!(v["type"], "FeatureCollection");
            let n = v["features"].as_array().unwrap().len();
            assert_eq!(n, rep.counts.iter().find(|c| c.0 == *layer).unwrap().1);
            if *layer == Layer::Roads {
                let p0 = &v["features"][0];
                assert_eq!(p0["properties"]["@id"], "way/100");
                assert_eq!(p0["geometry"]["coordinates"][0][0], 138.59);
            }
        }
        let m: serde_json::Value = serde_json::from_slice(&fs::read(&files.manifest).unwrap()).unwrap();
        assert_eq!(m["licence"], "ODbL-1.0");
        assert!(fs::read_to_string(&files.attribution).unwrap().contains("© OpenStreetMap contributors"));
        assert!(fs::read_to_string(&files.license).unwrap().contains("opendatacommons.org/licenses/odbl/1-0"));
        let _ = fs::remove_dir_all(&dir);
    }

    // --- A tiny hand-encoded .osm.pbf, so the osmpbf path is tested offline. ---

    fn varint(out: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }
    fn zz(v: i64) -> u64 {
        ((v << 1) ^ (v >> 63)) as u64
    }
    fn field_varint(out: &mut Vec<u8>, f: u32, v: u64) {
        varint(out, (f as u64) << 3);
        varint(out, v);
    }
    fn field_bytes(out: &mut Vec<u8>, f: u32, b: &[u8]) {
        varint(out, ((f as u64) << 3) | 2);
        varint(out, b.len() as u64);
        out.extend_from_slice(b);
    }
    fn packed(vals: impl IntoIterator<Item = u64>) -> Vec<u8> {
        let mut b = Vec::new();
        for v in vals {
            varint(&mut b, v);
        }
        b
    }
    fn blob(out: &mut Vec<u8>, kind: &str, payload: &[u8]) {
        let mut blob = Vec::new();
        field_bytes(&mut blob, 1, payload); // raw
        field_varint(&mut blob, 2, payload.len() as u64);
        let mut hdr = Vec::new();
        field_bytes(&mut hdr, 1, kind.as_bytes());
        field_varint(&mut hdr, 3, blob.len() as u64);
        out.extend_from_slice(&(hdr.len() as u32).to_be_bytes());
        out.extend_from_slice(&hdr);
        out.extend_from_slice(&blob);
    }

    /// Encode `sample()` (minus the missing-node road) as a PBF with plain nodes, ways, relations.
    fn sample_pbf() -> Vec<u8> {
        let d = sample();
        let mut strings: Vec<String> = vec![String::new()];
        let mut sid = |s: &str| -> u64 {
            if let Some(i) = strings.iter().position(|x| x == s) {
                return i as u64;
            }
            strings.push(s.to_owned());
            (strings.len() - 1) as u64
        };
        let mut nodes_grp = Vec::new();
        let mut ids: Vec<_> = d.nodes.keys().copied().collect();
        ids.sort();
        for id in ids {
            let [lon, lat] = d.nodes[&id];
            let mut n = Vec::new();
            field_varint(&mut n, 1, zz(id));
            field_varint(&mut n, 8, zz((lat * 1e7).round() as i64));
            field_varint(&mut n, 9, zz((lon * 1e7).round() as i64));
            field_bytes(&mut nodes_grp, 1, &n);
        }
        let mut ways_grp = Vec::new();
        let tagged: HashMap<i64, &Vec<(String, String)>> = d.ways.iter().map(|(i, t)| (*i, t)).collect();
        let mut wids: Vec<_> = d.way_refs.keys().copied().filter(|&i| i != 600).collect();
        wids.sort();
        for id in wids {
            let mut w = Vec::new();
            field_varint(&mut w, 1, id as u64);
            if let Some(tags) = tagged.get(&id) {
                field_bytes(&mut w, 2, &packed(tags.iter().map(|(k, _)| sid(k))));
                field_bytes(&mut w, 3, &packed(tags.iter().map(|(_, v)| sid(v))));
            }
            let refs = &d.way_refs[&id];
            let mut prev = 0;
            field_bytes(
                &mut w,
                8,
                &packed(refs.iter().map(|&r| {
                    let dlt = r - prev;
                    prev = r;
                    zz(dlt)
                })),
            );
            field_bytes(&mut ways_grp, 3, &w);
        }
        let mut rel_grp = Vec::new();
        for (id, tags, members) in &d.relations {
            let mut r = Vec::new();
            field_varint(&mut r, 1, *id as u64);
            field_bytes(&mut r, 2, &packed(tags.iter().map(|(k, _)| sid(k))));
            field_bytes(&mut r, 3, &packed(tags.iter().map(|(_, v)| sid(v))));
            field_bytes(&mut r, 8, &packed(members.iter().map(|m| sid(if m.1 { "inner" } else { "outer" }))));
            let mut prev = 0;
            field_bytes(
                &mut r,
                9,
                &packed(members.iter().map(|m| {
                    let dlt = m.0 - prev;
                    prev = m.0;
                    zz(dlt)
                })),
            );
            field_bytes(&mut r, 10, &packed(members.iter().map(|_| 1)));
            field_bytes(&mut rel_grp, 4, &r);
        }
        let mut st = Vec::new();
        for s in &strings {
            field_bytes(&mut st, 1, s.as_bytes());
        }
        let mut block = Vec::new();
        field_bytes(&mut block, 1, &st);
        for g in [&nodes_grp, &ways_grp, &rel_grp] {
            field_bytes(&mut block, 2, g);
        }
        let mut header = Vec::new();
        field_bytes(&mut header, 4, b"OsmSchema-V0.6");
        field_bytes(&mut header, 16, b"geo-fetch-test");
        field_varint(&mut header, 32, 1_700_000_000);
        let mut file = Vec::new();
        blob(&mut file, "OSMHeader", &header);
        blob(&mut file, "OSMData", &block);
        file
    }

    #[test]
    fn reads_hand_encoded_pbf() {
        let p = std::env::temp_dir().join(format!("geo-fetch-osmx-{}.osm.pbf", std::process::id()));
        fs::write(&p, sample_pbf()).unwrap();
        let (f, rep) = extract(&p, &ExtractSpec::default()).unwrap();
        assert_eq!(rep.header.replication_timestamp, Some(1_700_000_000));
        assert_eq!(rep.header.writing_program.as_deref(), Some("geo-fetch-test"));
        let (expect, _) = build_features(
            &OsmData { ways: sample().ways.into_iter().filter(|w| w.0 != 600).collect(), ..sample() },
            &ExtractSpec::default(),
        );
        assert_eq!(f.len(), expect.len());
        for (a, b) in f.iter().zip(&expect) {
            assert_eq!((a.layer, a.id, &a.kind, &a.tags), (b.layer, b.id, &b.kind, &b.tags));
            let (ab, bb) = (a.geometry.bounds(), b.geometry.bounds());
            assert!((ab.0 - bb.0).abs() < 1e-7 && (ab.3 - bb.3).abs() < 1e-7);
        }
        // Layer selection reaches the reader: only roads requested -> no relation pass output.
        let (data, _) = read_pbf(&p, &[Layer::Roads]).unwrap();
        assert!(data.relations.is_empty() && data.ways.len() == 1);
        let _ = fs::remove_file(&p);
    }
}
