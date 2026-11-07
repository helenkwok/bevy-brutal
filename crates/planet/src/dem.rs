//! Real-world DEM patches (`terrain::dem::DemGrid`, e.g. `geo-fetch terrain-grid` output) placed on
//! the cube-sphere planet.
//!
//! # Frames
//! A DEM grid lives in a [`coords::Enu`] tangent plane on WGS84: sample `(col, row)` is the plane
//! point `(east, north) = (x0 + col·s, y0 + row·s)`, and its geodetic latitude / longitude is
//! [`Enu::plane_to_geodetic`] of that point (the projection `geo-fetch` resampled the tiles with).
//! Two placements of a sample with orthometric height `H` follow from that:
//!
//! * **Planet frame** ([`DemPlacement::plane_to_planet`]): the renderer's sphere (radius
//!   [`EARTH_RADIUS_M`], +Z north, +X 0° E). The planet's textures and height cube are lat/lon
//!   maps with geodetic latitude used as the sphere's latitude, so the DEM goes on the same way:
//!   `latlon_dir(lat, lon) · (R + H)`. Latitude / longitude (and hence imagery, OSM, the coast) stay
//!   exact; lengths change by the sphere / ellipsoid radius ratios (`R/M` north-south, `R/N`
//!   east-west: +0.23 % / −0.22 % at Adelaide, ~±21 m at the edge of a ±9 km grid).
//! * **WGS84 ECEF** ([`DemPlacement::plane_to_ecef`]): the true position, ellipsoidal height
//!   `H + N_geoid` along the local normal. [`DemPlacement::geoid_n_m`] is the (constant, per patch)
//!   geoid undulation; it defaults to 0 (EGM96 is not wired yet, see the TODO in starter.md).
//!
//! Positions are formed in f64 and handed to the GPU as f32 offsets from an f64 origin (a planet
//! patch origin or a [`PlanetChunk::origin`]); the per-frame camera offset `origin − camera` is
//! also formed in f64 ([`camera_offset`]), so no f32 ever holds an absolute planet coordinate.
//!
//! # Blending into the cube-sphere
//! [`DemPlacement::blend_height`] is a height override for [`crate::patch::build`]: inside a disc of
//! geodesic radius `r_full_m` around the grid centre the patch heights are the DEM's (bilinear, from
//! a tent-filtered pyramid matched to the patch's vertex spacing, so coarse patches do not alias),
//! smoothstep back to the planet's own heights at `r_out_m`. The disc is the [`crate::graft`]
//! disc, so the shading side reuses the shared-Alps graft path: [`DemPlacement::bake_graft`]
//! writes the DEM's normals, cast shadows and coastline into a [`GraftFile`] (azimuthal-equidistant
//! about the centre, the same weight function), which the planet shader blends over the 2.4 km
//! slope map. Geometry and shading therefore switch to the DEM over the same disc.
use coords::{Enu, Geodetic};
use glam::{DVec3, Vec3};
use terrain::dem::{DemChunkId, DemGrid, DemTerrain};

use crate::EARTH_RADIUS_M;
use crate::cube::{dir_latlon, latlon_dir};
use crate::graft::{GraftFile, GraftLevel, GraftMap, weight};

fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// One pyramid level: same origin as the grid, spacing `s · 2^k`.
#[derive(Clone, Debug)]
struct Level {
    w: usize,
    h: usize,
    spacing: f64,
    data: Vec<f32>,
}

impl Level {
    fn at(&self, c: usize, r: usize) -> f32 {
        self.data[r * self.w + c]
    }

    /// Next level: every other sample of a separable [1 2 1] / 4 tent (sample positions kept).
    fn down(&self) -> Option<Self> {
        if self.w < 3 || self.h < 3 {
            return None;
        }
        let (w, h) = (self.w.div_ceil(2), self.h.div_ceil(2));
        let tap = |c: isize, r: isize| {
            self.at(c.clamp(0, self.w as isize - 1) as usize, r.clamp(0, self.h as isize - 1) as usize)
        };
        let mut data = Vec::with_capacity(w * h);
        for r in 0..h as isize {
            for c in 0..w as isize {
                let (c2, r2) = (2 * c, 2 * r);
                let mut acc = 0.0f32;
                for (dr, wr) in [(-1, 0.25f32), (0, 0.5), (1, 0.25)] {
                    for (dc, wc) in [(-1, 0.25f32), (0, 0.5), (1, 0.25)] {
                        acc += wr * wc * tap(c2 + dc, r2 + dr);
                    }
                }
                data.push(acc);
            }
        }
        Some(Self { w, h, spacing: self.spacing * 2.0, data })
    }

    /// Bilinear at fractional sample coordinates; `None` outside.
    fn bilinear(&self, u: f64, v: f64) -> Option<f64> {
        if !(0.0..=(self.w - 1) as f64).contains(&u) || !(0.0..=(self.h - 1) as f64).contains(&v) {
            return None;
        }
        let (c0, r0) = ((u.floor() as usize).min(self.w - 2), (v.floor() as usize).min(self.h - 2));
        let (fu, fv) = (u - c0 as f64, v - r0 as f64);
        let a = self.at(c0, r0) as f64 * (1.0 - fu) + self.at(c0 + 1, r0) as f64 * fu;
        let b = self.at(c0, r0 + 1) as f64 * (1.0 - fu) + self.at(c0 + 1, r0 + 1) as f64 * fu;
        Some(a * (1.0 - fv) + b * fv)
    }
}

/// A DEM grid placed on the planet (see the module docs).
#[derive(Clone, Debug)]
pub struct DemPlacement {
    enu: Enu,
    grid: DemGrid,
    levels: Vec<Level>,
    /// Plane coordinates of the blend disc's centre (the grid centre).
    centre_en: (f64, f64),
    map: GraftMap,
    /// Full-DEM radius and outer blend radius (geodesic metres from the centre).
    pub r_full_m: f64,
    pub r_out_m: f64,
    /// Width (m) of the fade from the DEM back to the planet's own terrain inside the grid's
    /// rectangle edges (the disc reaches past the inscribed circle to keep the grid's corners).
    pub edge_fade_m: f64,
    /// Geoid undulation `N` (m) added to the DEM's orthometric heights for [`Self::plane_to_ecef`].
    pub geoid_n_m: f64,
    /// Optional OSM colour drape over the palette in [`Self::bake_graft`] (plane coordinates).
    drape: Option<std::sync::Arc<crate::drape::Drape>>,
}

/// Shadow rays of [`DemPlacement::bake_graft`]: max distance (m).
const SHADOW_M: f64 = 6000.0;

impl DemPlacement {
    /// Place `grid` (lattice in `enu`'s tangent plane) with the default blend: a disc about the
    /// grid centre through its corners (`r_out` = half diagonal, `r_full = 0.85 r_out`) times a
    /// fade of [`Self::edge_fade_m`] (8 samples, at least 600 m) inside the rectangle's edges.
    pub fn new(enu: Enu, grid: DemGrid) -> Self {
        let s = grid.spacing_m as f64;
        let (x0, y0) = (grid.x0_m as f64, grid.y0_m as f64);
        let (x1, y1) = (x0 + (grid.width - 1) as f64 * s, y0 + (grid.height - 1) as f64 * s);
        let centre_en = (0.5 * (x0 + x1), 0.5 * (y0 + y1));
        let r_out = (0.5 * (x1 - x0).hypot(y1 - y0)).max(s);
        let mut levels = vec![Level { w: grid.width, h: grid.height, spacing: s, data: grid.heights.clone() }];
        while let Some(l) = levels.last().and_then(Level::down) {
            levels.push(l);
        }
        let g = enu.plane_to_geodetic(centre_en.0, centre_en.1);
        let up = latlon_dir(g.lat_deg, g.lon_deg);
        let east = DVec3::Z.cross(up).normalize();
        let north = up.cross(east);
        let map = GraftMap { centre: up, east, north, world_xz: (0.0, 0.0) };
        Self {
            enu,
            grid,
            levels,
            centre_en,
            map,
            r_full_m: 0.85 * r_out,
            r_out_m: r_out,
            edge_fade_m: (8.0 * s).max(600.0),
            geoid_n_m: 0.0,
            drape: None,
        }
    }

    /// Override the blend radii (m); `r_out` is clamped to at least `r_full`.
    pub fn with_blend(mut self, r_full_m: f64, r_out_m: f64) -> Self {
        self.r_full_m = r_full_m.max(0.0);
        self.r_out_m = r_out_m.max(self.r_full_m);
        self
    }

    /// Composite `drape` (built in this placement's plane coordinates, see
    /// [`Self::latlon_to_plane`]) over the palette in [`Self::bake_graft`], weighted by the DEM's
    /// rectangle fade; its water also clears the graft alpha (lakes, rivers, the Torrens).
    pub fn with_drape(mut self, drape: std::sync::Arc<crate::drape::Drape>) -> Self {
        self.drape = Some(drape);
        self
    }

    pub fn drape(&self) -> Option<&crate::drape::Drape> {
        self.drape.as_deref()
    }

    /// Plane rectangle of the grid lattice `(x0, y0, x1, y1)` (m).
    pub fn grid_rect(&self) -> (f64, f64, f64, f64) {
        let g = &self.grid;
        let s = g.spacing_m as f64;
        let (x0, y0) = (g.x0_m as f64, g.y0_m as f64);
        (x0, y0, x0 + (g.width - 1) as f64 * s, y0 + (g.height - 1) as f64 * s)
    }

    pub fn enu(&self) -> &Enu {
        &self.enu
    }
    pub fn grid(&self) -> &DemGrid {
        &self.grid
    }
    /// Blend-disc centre: plane coordinates, geodetic position and planet-frame direction.
    pub fn centre_plane(&self) -> (f64, f64) {
        self.centre_en
    }
    pub fn centre_geodetic(&self) -> Geodetic {
        self.enu.plane_to_geodetic(self.centre_en.0, self.centre_en.1)
    }
    pub fn centre_dir(&self) -> DVec3 {
        self.map.centre
    }
    /// The disc's azimuthal-equidistant map (world x east, z south, metres from the centre).
    pub fn graft_map(&self) -> &GraftMap {
        &self.map
    }
    /// Pyramid levels (level 0 = the grid).
    pub fn levels(&self) -> usize {
        self.levels.len()
    }

    /// Geodetic latitude / longitude (deg) of plane point `(east, north)`.
    pub fn plane_to_latlon(&self, east: f64, north: f64) -> (f64, f64) {
        let g = self.enu.plane_to_geodetic(east, north);
        (g.lat_deg, g.lon_deg)
    }

    /// Inverse of [`Self::plane_to_latlon`]: the plane point whose geodetic lat / lon is
    /// `(lat, lon)` (ellipsoid-point start, two Newton steps on the local radii: < 1 µm).
    pub fn latlon_to_plane(&self, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
        let p = self.enu.geodetic_to_enu(Geodetic::new(lat_deg, lon_deg, 0.0));
        let (mut e, mut n) = (p.x, p.y);
        for _ in 0..2 {
            let g = self.enu.plane_to_geodetic(e, n);
            let m = coords::geodetic::meridional_radius(g.lat_deg);
            let nc = coords::geodetic::prime_vertical_radius(g.lat_deg) * g.lat_deg.to_radians().cos();
            let mut dlon = lon_deg - g.lon_deg;
            dlon -= 360.0 * (dlon / 360.0).round();
            let (dx, dy) = (dlon.to_radians() * nc, (lat_deg - g.lat_deg).to_radians() * m);
            // the plane's east / north at (e, n) are the origin's; to first order the geodetic
            // offsets map straight back
            e += dx;
            n += dy;
        }
        (e, n)
    }

    /// Planet-frame direction -> plane point.
    pub fn dir_to_plane(&self, d: DVec3) -> (f64, f64) {
        let (lat, lon) = dir_latlon(d);
        self.latlon_to_plane(lat, lon)
    }

    /// Plane point -> planet-frame unit direction.
    pub fn plane_to_dir(&self, east: f64, north: f64) -> DVec3 {
        let (lat, lon) = self.plane_to_latlon(east, north);
        latlon_dir(lat, lon)
    }

    /// Planet-frame (sphere) position, f64 metres, of plane point `(east, north)` at orthometric
    /// height `h_m`.
    pub fn plane_to_planet(&self, east: f64, north: f64, h_m: f64) -> DVec3 {
        self.plane_to_dir(east, north) * (EARTH_RADIUS_M + h_m)
    }

    /// WGS84 ECEF position (f64 metres) of plane point `(east, north)` at orthometric height `h_m`
    /// (ellipsoidal height `h_m + geoid_n_m` along the local normal).
    pub fn plane_to_ecef(&self, east: f64, north: f64, h_m: f64) -> DVec3 {
        let g = self.enu.plane_to_geodetic(east, north);
        Geodetic::new(g.lat_deg, g.lon_deg, h_m + self.geoid_n_m).to_ecef()
    }

    /// DEM height (m) at plane point `(east, north)`, filtered for a sample spacing of `spacing_m`
    /// (trilinear in the tent pyramid; `spacing_m <=` the grid spacing = bilinear on the grid).
    /// `None` outside the lattice.
    pub fn height(&self, east: f64, north: f64, spacing_m: f64) -> Option<f64> {
        let s0 = self.levels[0].spacing;
        let lod = (spacing_m / s0).max(1.0).log2().min((self.levels.len() - 1) as f64);
        let k = lod.floor() as usize;
        let t = lod - k as f64;
        let at = |l: &Level| {
            let (x0, y0) = (self.grid.x0_m as f64, self.grid.y0_m as f64);
            l.bilinear((east - x0) / l.spacing, (north - y0) / l.spacing)
        };
        // coarse levels can end up to one sample short of the grid's last row / column: fall back
        // to the grid there
        let a = at(&self.levels[k]).or_else(|| if k > 0 { at(&self.levels[0]) } else { None })?;
        if t <= 0.0 || k + 1 >= self.levels.len() {
            return Some(a);
        }
        Some(at(&self.levels[k + 1]).map_or(a, |b| a + (b - a) * t))
    }

    /// Height gradient `(dh/de, dh/dn)` at `(east, north)` (central differences at
    /// `max(spacing_m, grid spacing)`); `None` outside the lattice.
    pub fn gradient(&self, east: f64, north: f64, spacing_m: f64) -> Option<(f64, f64)> {
        let s = spacing_m.max(self.levels[0].spacing);
        let h = |e: f64, n: f64| self.height(e, n, spacing_m).or_else(|| self.height(east, north, spacing_m));
        let (he0, he1) = (h(east - s, north)?, h(east + s, north)?);
        let (hn0, hn1) = (h(east, north - s)?, h(east, north + s)?);
        Some(((he1 - he0) / (2.0 * s), (hn1 - hn0) / (2.0 * s)))
    }

    /// Spatial blend weight (1 inside `r_full`, 0 beyond `r_out`) and geodesic distance (m) of a
    /// planet direction from the disc centre.
    pub fn weight_dir(&self, d: DVec3) -> (f64, f64) {
        let (_, _, dist) = self.map.dir_to_world(d);
        (weight(dist, self.r_full_m, self.r_out_m), dist)
    }

    /// Fade (0..1) from the grid rectangle's edges (inset one sample) inward over
    /// [`Self::edge_fade_m`]: 0 outside the grid.
    pub fn rect_weight(&self, e: f64, n: f64) -> f64 {
        let g = &self.grid;
        let s = g.spacing_m as f64;
        let (x0, y0) = (g.x0_m as f64 + s, g.y0_m as f64 + s);
        let (x1, y1) = (g.x0_m as f64 + (g.width - 2) as f64 * s, g.y0_m as f64 + (g.height - 2) as f64 * s);
        let inside = (e - x0).min(x1 - e).min(n - y0).min(y1 - n);
        smoothstep(0.0, self.edge_fade_m.max(1e-6), inside)
    }

    /// Total DEM weight at direction `d`: disc × rectangle fade.
    pub fn weight_total(&self, d: DVec3) -> f64 {
        let (w, _) = self.weight_dir(d);
        if w <= 0.0 {
            return 0.0;
        }
        let (e, n) = self.dir_to_plane(d);
        w * self.rect_weight(e, n)
    }

    /// Height override for [`crate::patch::build`]: the planet's own height `h_planet` at
    /// direction `d`, blended toward the DEM's (filtered at the patch's vertex spacing) by the disc
    /// weight × `strength` (0..1).
    pub fn blend_height(&self, d: DVec3, h_planet: f64, spacing_m: f64, strength: f64) -> f64 {
        if strength <= 0.0 {
            return h_planet;
        }
        let (w, _) = self.weight_dir(d);
        let w = w * strength.min(1.0);
        if w <= 0.0 {
            return h_planet;
        }
        let (e, n) = self.dir_to_plane(d);
        let w = w * self.rect_weight(e, n);
        if w <= 0.0 {
            return h_planet;
        }
        match self.height(e, n, spacing_m) {
            Some(h) => h_planet + (h - h_planet) * w,
            None => h_planet,
        }
    }

    /// A `terrain::dem` chunk on the planet: every vertex (`x = east`, `y = height`, `z = −north`,
    /// skirts included) through [`Self::plane_to_planet`], stored as f32 offsets from the f64
    /// planet position of the chunk's centre at height 0. Normals are rotated from the tangent
    /// plane into the planet frame.
    pub fn chunk_on_planet(&self, terrain: &DemTerrain, id: DemChunkId) -> PlanetChunk {
        let verts = terrain.mesh_chunk(id);
        let aabb = terrain.chunk_aabb(id);
        let (ce, cn) = (0.5 * (aabb.min[0] + aabb.max[0]) as f64, -0.5 * (aabb.min[2] + aabb.max[2]) as f64);
        let origin = self.plane_to_planet(ce, cn, 0.0);
        let [ex, nx, ux] = self.enu.axes();
        let mut pos = Vec::with_capacity(verts.len());
        let mut normal = Vec::with_capacity(verts.len());
        for v in &verts {
            let p = self.plane_to_planet(v.pos[0] as f64, -(v.pos[2] as f64), v.pos[1] as f64);
            pos.push((p - origin).as_vec3().to_array());
            let n = ex * v.normal[0] as f64 + ux * v.normal[1] as f64 - nx * v.normal[2] as f64;
            normal.push(n.normalize_or_zero().as_vec3().to_array());
        }
        PlanetChunk { id, origin, pos, normal }
    }

    /// Bake the disc's shading into a [`GraftFile`] for the planet shader's graft bindings: albedo
    /// (a height / slope / coast palette; the shader's graft tint transfers the Blue Marble colour
    /// onto it), alpha = 1 − water (DEM height below 0 m), aux = world normal x / z, cast-shadow
    /// sun visibility for `sun_planet` (planet-frame direction), and 1 (no AO).
    ///
    /// Levels (each `size`², concentric about the centre):
    /// * outer `1.1 r_out`, mid `0.5 r_out`;
    /// * optional close tile (`close_half_m`): Fragments-style streamed level so OSM building
    ///   objects LOD (`BuildingLod`, ≤ 6.8 m texels at `objects_px = 2`) can resolve. The planet
    ///   shader reads it as array layer 2 when `meso2.y` holds the half extent in km.
    pub fn bake_graft(&self, sun_planet: DVec3, size: u32, planet_height: &(dyn Fn(DVec3) -> f64 + Sync)) -> GraftFile {
        self.bake_graft_close(sun_planet, size, planet_height, None)
    }

    /// [`Self::bake_graft`] with an optional close half extent (m). `close_half_m` is clamped below
    /// the mid half so the streamed tile is always the finest concentric level.
    pub fn bake_graft_close(
        &self,
        sun_planet: DVec3,
        size: u32,
        planet_height: &(dyn Fn(DVec3) -> f64 + Sync),
        close_half_m: Option<f64>,
    ) -> GraftFile {
        let s = sun_planet.normalize();
        let [ae, an, au] = self.enu.axes();
        let s_enu = DVec3::new(s.dot(ae), s.dot(an), s.dot(au));
        let horiz = DVec3::new(s_enu.x, s_enu.y, 0.0).normalize_or_zero();
        let tan_el = s_enu.z / s_enu.x.hypot(s_enu.y).max(1e-9);
        let mid = 0.5 * self.r_out_m;
        let mut halves = vec![1.1 * self.r_out_m, mid];
        if let Some(c) = close_half_m {
            let c = c.clamp(1.0, mid * 0.95);
            halves.push(c);
        }
        let mut levels = Vec::new();
        for half in halves {
            let n = size as usize;
            let step = 2.0 * half / size as f64;
            let mut albedo = vec![0u8; n * n * 4];
            let mut aux = vec![0u8; n * n * 4];
            let rows: Vec<(Vec<u8>, Vec<u8>)> = {
                use rayon::prelude::*;
                (0..n)
                    .into_par_iter()
                    .map(|j| {
                        let mut ra = Vec::with_capacity(n * 4);
                        let mut rx = Vec::with_capacity(n * 4);
                        for i in 0..n {
                            let x = -half + (i as f64 + 0.5) * step;
                            let z = -half + (j as f64 + 0.5) * step;
                            let (a, b) = self.texel(x, z, step, s_enu, horiz, tan_el, planet_height);
                            ra.extend_from_slice(&a);
                            rx.extend_from_slice(&b);
                        }
                        (ra, rx)
                    })
                    .collect()
            };
            for (j, (ra, rx)) in rows.into_iter().enumerate() {
                albedo[j * n * 4..(j + 1) * n * 4].copy_from_slice(&ra);
                aux[j * n * 4..(j + 1) * n * 4].copy_from_slice(&rx);
            }
            levels.push(GraftLevel { size, half_m: half, albedo, aux });
        }
        let c = self.centre_geodetic();
        GraftFile {
            lat: c.lat_deg,
            lon: c.lon_deg,
            world_xz: (0.0, 0.0),
            r_full_m: self.r_full_m,
            r_out_m: self.r_out_m,
            sun_world: [s.dot(self.map.east), s.dot(self.map.centre), -s.dot(self.map.north)],
            levels,
        }
    }

    /// Height and plane gradient at `(e, n)`: the DEM faded into the planet's own terrain
    /// (`planet_height`, sampled with a `>= 1.2 km` stencil: the cube's ~2.4 km texels) by
    /// [`Self::rect_weight`]. Returns `(h, dh/de, dh/dn, rect weight, planet height)`.
    fn blended(
        &self,
        e: f64,
        n: f64,
        step: f64,
        ph: &(dyn Fn(DVec3) -> f64 + Sync),
        grad: bool,
    ) -> (f64, f64, f64, f64, f64) {
        let wr = self.rect_weight(e, n);
        let dem = if wr > 0.0 {
            match (self.height(e, n, step), if grad { self.gradient(e, n, step) } else { Some((0.0, 0.0)) }) {
                (Some(h), Some(g)) => Some((h, g)),
                _ => None,
            }
        } else {
            None
        };
        let wr = if dem.is_some() { wr } else { 0.0 };
        let hp_at = |e: f64, n: f64| ph(self.plane_to_dir(e, n));
        let need_p = wr < 1.0;
        let hp = if need_p { hp_at(e, n) } else { 0.0 };
        let (pe, pn) = if need_p && grad {
            let sp = step.max(1200.0);
            ((hp_at(e + sp, n) - hp_at(e - sp, n)) / (2.0 * sp), (hp_at(e, n + sp) - hp_at(e, n - sp)) / (2.0 * sp))
        } else {
            (0.0, 0.0)
        };
        let (hd, (de, dn)) = dem.unwrap_or((hp, (pe, pn)));
        let l = |a: f64, b: f64| a + (b - a) * wr;
        (l(hp, hd), l(pe, de), l(pn, dn), wr, hp)
    }

    /// One graft texel at disc-world `(x, z)` (east, south): (albedo sRGBA, aux).
    #[allow(clippy::too_many_arguments)]
    fn texel(
        &self,
        x: f64,
        z: f64,
        step: f64,
        s_enu: DVec3,
        horiz: DVec3,
        tan_el: f64,
        ph: &(dyn Fn(DVec3) -> f64 + Sync),
    ) -> ([u8; 4], [u8; 4]) {
        let (e, n) = self.dir_to_plane(self.map.world_to_dir(x, z));
        let (h, ge, gn, wr, hp) = self.blended(e, n, step, ph, true);
        // world frame: x east, y up, z south
        let nrm = DVec3::new(-ge, 1.0, gn).normalize();
        let n_enu = DVec3::new(-ge, -gn, 1.0).normalize();
        let vis = if n_enu.dot(s_enu) > 0.0 && s_enu.z > 0.0 {
            let mut t = step.max(self.levels[0].spacing);
            let mut margin = f64::MAX;
            while t < SHADOW_M {
                let (qe, qn) = (e + horiz.x * t, n + horiz.y * t);
                let (hs, ..) = self.blended(qe, qn, step, ph, false);
                // the ground drops t² / 2R below the ray's tangent plane
                margin = margin.min(((h + 2.0 + t * tan_el) - (hs - t * t / (2.0 * EARTH_RADIUS_M))) / t);
                if margin < -0.01 {
                    break;
                }
                t += (step * 0.75).max(t * 0.02);
            }
            smoothstep(-0.005, 0.005, margin)
        } else {
            0.0
        };
        // water: the DEM's sub-zero ground, else the planet height cube's sea level
        let mut water = smoothstep(0.0, -1.0, h) * wr + smoothstep(1.0, 0.2, hp) * (1.0 - wr);
        let mut a = palette(h, 1.0 - nrm.y);
        if let Some(d) = &self.drape {
            // the drape's own rectangle fade (wider than the DEM when the OSM extract is), else
            // the DEM's
            let w = if d.edge_fade_m > 0.0 { d.edge_weight(e, n) } else { wr };
            let (c, wd) = d.composite(a, e, n, step, w);
            a = c;
            water = water.max(wd);
        }
        (
            [srgb8(a[0]), srgb8(a[1]), srgb8(a[2]), ((1.0 - water) * 255.0).round() as u8],
            [
                ((nrm.x * 0.5 + 0.5) * 255.0).round() as u8,
                ((nrm.z * 0.5 + 0.5) * 255.0).round() as u8,
                (vis * 255.0).round() as u8,
                255,
            ],
        )
    }
}

/// Dry-temperate DEM palette (linear albedo): coastal plain (tan / grey, urban and cropland),
/// greener foothills, dark eucalypt on steep or high ground. The shader's graft tint rescales it to
/// the imagery's large-scale colour, so only its relative variation matters.
pub fn palette(h: f64, slope: f64) -> [f64; 3] {
    let plain = [0.115, 0.105, 0.085];
    let foot = [0.085, 0.095, 0.060];
    let forest = [0.045, 0.060, 0.035];
    let t1 = smoothstep(60.0, 250.0, h);
    let t2 = smoothstep(0.04, 0.2, slope).max(smoothstep(350.0, 550.0, h)) * smoothstep(40.0, 150.0, h);
    let mix = |a: [f64; 3], b: [f64; 3], t: f64| [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t);
    mix(mix(plain, foot, t1), forest, t2)
}

/// sRGB encode (linear 0..1 -> 0..255).
pub fn srgb8(c: f64) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

/// A DEM chunk on the planet: f32 vertex offsets (`pos`) and planet-frame normals relative to the
/// f64 `origin` (planet frame, metres).
#[derive(Clone, Debug)]
pub struct PlanetChunk {
    pub id: DemChunkId,
    pub origin: DVec3,
    pub pos: Vec<[f32; 3]>,
    pub normal: Vec<[f32; 3]>,
}

impl PlanetChunk {
    /// Camera-relative f32 vertex positions (`origin − camera` formed in f64, rounded once).
    pub fn camera_relative(&self, cam: DVec3) -> impl Iterator<Item = Vec3> + '_ {
        let o = camera_offset(self.origin, cam);
        self.pos.iter().map(move |p| o + Vec3::from_array(*p))
    }
}

/// `origin − camera` in f64, rounded once to f32 (the per-frame patch / chunk offset).
pub fn camera_offset(origin: DVec3, cam: DVec3) -> Vec3 {
    (origin - cam).as_vec3()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch;
    use crate::quadtree::Key;

    /// 641 x 641 at 30 m (+-9.6 km) about Adelaide: a 500 m Gaussian hill 3 km east of the origin
    /// on a 40 m plateau, sea (-5 m) west of x = -6 km.
    fn grid() -> DemPlacement {
        let (n, s) = (641usize, 30.0f32);
        let x0 = -(n as f32 - 1.0) * 0.5 * s;
        let mut h = Vec::with_capacity(n * n);
        for r in 0..n {
            for c in 0..n {
                let (e, nn) = (x0 + c as f32 * s, x0 + r as f32 * s);
                let d2 = (e - 3000.0).powi(2) + nn.powi(2);
                h.push(if e < -6000.0 { -5.0 } else { 40.0 + 500.0 * (-d2 / (2.0 * 1500.0f32.powi(2))).exp() });
            }
        }
        let g = DemGrid::new(n, n, s, x0, x0, h).unwrap();
        DemPlacement::new(Enu::new(Geodetic::new(-34.9, 138.6, 0.0)), g)
    }

    #[test]
    fn latlon_plane_round_trip() {
        let p = grid();
        for (e, n) in [(0.0, 0.0), (9000.0, -9000.0), (-9500.0, 4321.5), (123.4, 9600.0)] {
            let (lat, lon) = p.plane_to_latlon(e, n);
            let (e2, n2) = p.latlon_to_plane(lat, lon);
            assert!((e2 - e).abs() < 1e-6 && (n2 - n).abs() < 1e-6, "({e}, {n}) -> ({e2}, {n2})");
            let (e3, n3) = p.dir_to_plane(p.plane_to_dir(e, n));
            assert!((e3 - e).abs() < 1e-3 && (n3 - n).abs() < 1e-3, "dir ({e}, {n}) -> ({e3}, {n3})");
        }
    }

    #[test]
    fn planet_and_ecef_placement() {
        let p = grid();
        let g = p.centre_geodetic();
        assert!((g.lat_deg + 34.9).abs() < 1e-9 && (g.lon_deg - 138.6).abs() < 1e-9);
        // sphere: lat / lon preserved, radius R + H
        let q = p.plane_to_planet(5000.0, -7000.0, 321.0);
        let (lat, lon) = dir_latlon(q);
        let (la2, lo2) = p.plane_to_latlon(5000.0, -7000.0);
        assert!((lat - la2).abs() < 1e-9 && (lon - lo2).abs() < 1e-9);
        assert!((q.length() - EARTH_RADIUS_M - 321.0).abs() < 1e-6);
        // WGS84: the plane point drops d^2 / 2R onto the ellipsoid, heights along the local normal
        let e = p.plane_to_ecef(8000.0, 6000.0, 0.0);
        let enu = p.enu().from_ecef(e);
        assert!((enu.x - 8000.0).abs() < 0.01 && (enu.y - 6000.0).abs() < 0.01, "{enu:?}");
        assert!((enu.z + p.enu().curvature_drop_m(8000.0, 6000.0)).abs() < 0.01, "{enu:?}");
        let up = p.plane_to_ecef(8000.0, 6000.0, 100.0) - e;
        assert!((up.length() - 100.0).abs() < 1e-6);
        // geoid undulation shifts along the normal
        let mut p2 = p.clone();
        p2.geoid_n_m = -20.0;
        assert!(((p2.plane_to_ecef(0.0, 0.0, 50.0) - p.plane_to_ecef(0.0, 0.0, 30.0)).length()) < 1e-6);
    }

    #[test]
    fn pyramid_and_height() {
        let p = grid();
        assert!(p.levels() >= 9);
        // on a sample: the grid value exactly
        let h0 = p.height(3000.0, 0.0, 0.0).unwrap();
        assert!((h0 - 540.0).abs() < 1e-3, "{h0}");
        // coarser spacing smooths the peak, monotonically
        let mut last = h0;
        for s in [60.0, 120.0, 240.0, 480.0, 960.0] {
            let h = p.height(3000.0, 0.0, s).unwrap();
            assert!(h <= last + 1e-6 && h > 40.0, "{s}: {h}");
            last = h;
        }
        assert!(p.height(20_000.0, 0.0, 0.0).is_none());
        let (ge, gn) = p.gradient(3000.0 + 1500.0, 0.0, 30.0).unwrap();
        // d/dx of the Gaussian at one sigma: -A / sigma e^-1/2
        assert!((ge + 500.0 / 1500.0 * (-0.5f64).exp()).abs() < 3e-3 && gn.abs() < 1e-3, "{ge} {gn}");
    }

    #[test]
    fn rect_fade_keeps_corners_and_meets_the_planet() {
        let p = grid();
        let (x0, y0) = (p.grid().x0_m as f64, p.grid().y0_m as f64);
        assert_eq!(p.rect_weight(0.0, 0.0), 1.0);
        assert_eq!(p.rect_weight(x0 - 10.0, 0.0), 0.0);
        let mid = p.rect_weight(x0 + 0.5 * p.edge_fade_m, 0.0);
        assert!(mid > 0.2 && mid < 0.8, "{mid}");
        // the disc reaches the corners; at the rectangle's edge the blend is the planet height
        assert!(p.r_out_m >= x0.hypot(y0) - 1.0);
        let edge = p.plane_to_dir(x0 + 1.0, 0.0);
        assert!((p.blend_height(edge, 7.0, 20.0, 1.0) - 7.0).abs() < 1e-9);
        assert!(p.weight_total(edge) < 1e-9 && (p.weight_total(p.centre_dir()) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn blend_into_patches() {
        let p = grid();
        let c = p.centre_dir();
        assert_eq!(p.blend_height(c, 7.0, 20.0, 1.0), p.height(0.0, 0.0, 20.0).unwrap());
        assert_eq!(p.blend_height(c, 7.0, 20.0, 0.0), 7.0);
        let far = p.plane_to_dir(0.0, p.r_out_m + 100.0);
        assert_eq!(p.blend_height(far, 7.0, 20.0, 1.0), 7.0);
        // a level-14 planet patch over the hill: inside r_full its vertices sit on the DEM
        let (f, u, v) = crate::cube::dir_face(p.plane_to_dir(3000.0, 0.0));
        let lvl = 14u8;
        let n = 1u32 << lvl;
        let (i, j) = (((u + 1.0) * 0.5 * n as f64) as u32, ((v + 1.0) * 0.5 * n as f64) as u32);
        let key = Key { face: f as u8, level: lvl, i, j };
        let spacing = EARTH_RADIUS_M * std::f64::consts::FRAC_PI_2 / n as f64 / crate::cube::N as f64;
        let mesh = patch::build(&key, EARTH_RADIUS_M, &|d| p.blend_height(d, 0.0, spacing, 1.0));
        let mut worst = 0.0f64;
        for b in 0..=crate::cube::N {
            for a in 0..=crate::cube::N {
                let d = key.vertex_dir(a, b);
                let (en, nn) = p.dir_to_plane(d);
                let want = p.height(en, nn, spacing).unwrap();
                let v = mesh.verts[patch::grid(a, b) as usize];
                let pos = mesh.origin + glam::DVec3::from_array(v.pos.map(f64::from));
                worst = worst.max((pos.length() - EARTH_RADIUS_M - want).abs());
            }
        }
        // f32 offsets over a ~600 m patch: sub-centimetre
        assert!(worst < 0.01, "{worst} m");
    }

    #[test]
    fn chunks_on_planet_are_camera_relative_exact() {
        let p = grid();
        let t = DemTerrain::new(p.grid().clone());
        let id = DemChunkId { lod: 0, x: 4, y: 4 };
        let ch = p.chunk_on_planet(&t, id);
        let verts = t.mesh_chunk(id);
        // camera 100 m above a vertex near the chunk's corner
        let v0 = verts[0];
        let cam = p.plane_to_planet(v0.pos[0] as f64, -(v0.pos[2] as f64), v0.pos[1] as f64 + 100.0);
        let mut worst = 0.0f64;
        for (v, rel) in verts.iter().zip(ch.camera_relative(cam)) {
            let truth = p.plane_to_planet(v.pos[0] as f64, -(v.pos[2] as f64), v.pos[1] as f64) - cam;
            worst = worst.max((truth - rel.as_dvec3()).length());
        }
        assert!(worst < 2e-3, "{worst} m");
        // normals point away from the planet centre
        let up = ch.origin.normalize();
        assert!(ch.normal.iter().take(65 * 65).all(|n| glam::DVec3::from_array(n.map(f64::from)).dot(up) > 0.5));
    }

    #[test]
    fn graft_bake_normals_shadow_coast() {
        let p = grid();
        // low sun (4.6 deg) from the west: the hill's east flank is in shadow (its peak, 2.5 km back,
        // subtends 8.5 deg)
        let [e, _, u] = p.enu().axes();
        let sun = (u * 0.08 - e).normalize();
        let g = p.bake_graft(sun, 128, &|_| 0.0);
        assert_eq!(g.levels.len(), 2);
        let g3 = p.bake_graft_close(sun, 128, &|_| 0.0, Some(3000.0));
        assert_eq!(g3.levels.len(), 3);
        assert!((g3.levels[2].half_m - 3000.0).abs() < 1e-6);
        // production DEM_TEX=2048 + 3 km half → 2.93 m texels (BuildingLod objects ≤ 6.8 m)
        let step_prod = 2.0 * g3.levels[2].half_m / 2048.0;
        assert!(step_prod <= 6.8, "close tile at DEM_TEX=2048 must resolve objects, got {step_prod}");
        let l = &g.levels[0];
        let at = |x: f64, z: f64| {
            let n = l.size as usize;
            let i = ((x + l.half_m) / (2.0 * l.half_m) * n as f64) as usize;
            let j = ((z + l.half_m) / (2.0 * l.half_m) * n as f64) as usize;
            let k = (j * n + i) * 4;
            ([l.albedo[k], l.albedo[k + 1], l.albedo[k + 2], l.albedo[k + 3]], [l.aux[k], l.aux[k + 1], l.aux[k + 2]])
        };
        // flat plateau north-west of the hill: normal up, lit, land
        let (a, x) = at(-3000.0, -6000.0);
        assert!(x[0].abs_diff(128) <= 1 && x[1].abs_diff(128) <= 1 && x[2] == 255 && a[3] == 255, "{a:?} {x:?}");
        // the hill's west flank faces the sun (normal toward -x), east flank shadowed
        let (_, w) = at(3000.0 - 1500.0, 0.0);
        assert!(w[0] < 110 && w[2] == 255, "{w:?}");
        let (_, s) = at(3000.0 + 2500.0, 0.0);
        assert!(s[2] == 0, "{s:?}");
        // sea west of -6 km
        let (sea, _) = at(-7500.0, 0.0);
        assert_eq!(sea[3], 0);
        // file round trip
        assert!(GraftFile::from_bytes(&g.to_bytes()).is_some());
    }
}
