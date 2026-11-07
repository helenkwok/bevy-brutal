# Credits and data licences

Software in this repository is dual-licensed MIT OR Apache-2.0 (Helen Kwok, 2026).
**Data and textures are separate.** Do not assume they are MIT/Apache.

## Inspiration (not a dependency)

- [BRUTAL](https://github.com/ahwootech/BRUTAL) by Ahwoo Limited — design reference only; no BRUTAL code is shipped here. BRUTAL License v1.0 requires credits + logo splash for products that *use* BRUTAL; that does not apply to this independent Rust codebase unless you incorporate BRUTAL itself.

## Engine / libraries

- [Bevy](https://bevyengine.org/) — MIT OR Apache-2.0
- Other Rust crates — see each crate's `license` on crates.io / in `Cargo.lock`

## Embedded / vendored code

- Transvoxel (Eric Lengyel) — MIT (`crates/voxel/LICENSE-Transvoxel`)

## Earth / planet data (fetched by tools; not in the source tarball by default)

| Asset | Source | Licence notes |
|---|---|---|
| Blue Marble / Black Marble / EPIC | NASA Earth Observatory / EPIC | NASA imagery generally not copyrighted; credit NASA |
| GEBCO-derived topo/bathy (via NASA EO) | GEBCO Compilation Group | Free use with GEBCO attribution |
| Copernicus GLO-30 DEM | EU / ESA / DLR / Airbus | Copernicus licence; attribution required |
| Sentinel-2 cloudless **2020** (default) | EOX s2maps.eu | **CC BY-NC-SA 4.0** (non-commercial + share-alike) |
| Sentinel-2 cloudless **2016** (`S2_LAYER=2016`) | EOX | CC BY 4.0 |

For an Apache/MIT-friendly public demo path, prefer `S2_LAYER=2016` or omit Sentinel imagery from the published tree.

## OSM and terrain fetched by `crates/geo-fetch` (not in the source tree; cached under `data/geo`)

| Asset | Source | Licence notes |
|---|---|---|
| OpenStreetMap extracts (`.osm.pbf`) | © OpenStreetMap contributors, via Geofabrik download server | **ODbL 1.0**. Credit "© OpenStreetMap contributors". Any OSM-derived *database file* we ship must be offered under ODbL (`geo_fetch::attribution::OSM_DERIVED_DB_NOTICE`). Rendered visuals need attribution only |
| AWS Terrain Tiles (Terrarium / GeoTIFF / Skadi) | Tilezen joerd, AWS Registry of Open Data | Public domain + attribution licences (USGS, NOAA, Copernicus EU-DEM, Geoscience Australia, Kartverket, LINZ, OGL Canada, UK EA, INEGI, ArcticDEM). Full list: `cargo run -p geo-fetch -- credits --terrain` |
| Sentinel-2 L2A surface reflectance (palette fit only) | Copernicus / ESA, via Element 84 Earth Search STAC on AWS Open Data (`sentinel-cogs`) | Copernicus Sentinel data terms: free incl. commercial use; credit **"Contains modified Copernicus Sentinel data 2024"**. Only fitted per-class albedos (`planet::drape::DrapePalette::adelaide_s2`) are in the tree; the scene clip stays in `data/geo/sentinel2` (`tools/s2_fit.py`). Not the EOX cloudless mosaic (CC BY-NC-SA) |

Derived files from `geo-fetch osm-extract` (`data/geo/osm/derived/<name>/`) are ODbL. If they are ever copied into the tree (e.g. `assets/osm/<name>/`), they keep their own `LICENSE.txt` + `ATTRIBUTION.txt` and are not covered by MIT/Apache. Height grids (`terrain-grid`) carry the terrain attribution only.

No Google data (Street View, satellite, 3D tiles) is used anywhere in this repository.
