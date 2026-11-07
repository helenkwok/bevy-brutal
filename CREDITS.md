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


## Gaia EDR3 Catalogue of Nearby Stars (external local input)

The observed `universe` input is the **full GAVO `gcns.main` mirror of GCNS**, not
synthetic stellar data and not a complete Milky Way map. Source/schema/terms:
https://dc.g-vo.org/gcns/q/cone/info and https://dc.g-vo.org/browse/gcns/q .
The mirror states: “Gaia Catalogue of Nearby Stars © 2023 by Richard Smart is
licensed under CC BY-NC 4.0.” https://creativecommons.org/licenses/by-nc/4.0/ .
This is a **noncommercial restriction**, not MIT/Apache code permission. The raw
catalogue, derived binary, observed samples and local imagery are not committed.
Commercial incorporation/redistribution of this acquired product (or adapted
outputs) requires a suitable permission/licensing assessment; no such clearance
is claimed. ESA illustrations and the paper have their own terms, not inferred
from this table licence. The code's licence does not relicense catalogue data.

Primary citation: Gaia Collaboration, R. L. Smart et al. (2021), “Gaia Early Data
Release 3: The Gaia Catalogue of Nearby Stars”, A&A 649, A6,
https://doi.org/10.1051/0004-6361/202039498 .
Official release overview: https://www.cosmos.esa.int/web/gaia/edr3-gcns .

Gaia acknowledgement:
This work has made use of data from the European Space Agency (ESA) mission
Gaia (https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing
and Analysis Consortium (DPAC,
https://www.cosmos.esa.int/web/gaia/dpac/consortium). Funding for the DPAC has
been provided by national institutions, in particular the institutions
participating in the Gaia Multilateral Agreement.

Distances are GCNS Bayesian posterior estimates/quantiles, not naive 1/parallax.
Photometric display uses an explicitly modelled achromatic Gaia G-band proxy;
it is not calibrated bolometric luminosity or observed RGB. Static ICRS epoch
J2016.0 positions are not propagated to the current date. Ancillary resolved
system/cluster tables are not yet ingested. No extra Sun is inserted in GCNS.
