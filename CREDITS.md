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

## CNS5 independent positional diagnostic (external snapshot, unmerged)

Golovin, Reffert, Just, Jordan, Vani & Jahreiß (2023), “The Fifth Catalogue of
Nearby Stars (CNS5)”, A&A 670 A19, https://www.aanda.org/articles/aa/full_html/2023/02/aa44250-22/aa44250-22.html .
GAVO https://dc.g-vo.org/CNS5 explicitly labels the compilation **CC BY 4.0**.
This does not independently clear upstream Gaia/Hipparcos/infrared product rights.
No commercial redistribution clearance is asserted. Retained current mirror has
5,909 rows, not verified as all 5,931 published objects. Raw field bibcodes and
native epochs are preserved. Existing Gaia/DPAC acknowledgment above also applies.
No CNS5 data, samples or imagery are bundled; no automatic GCNS supplement occurs.

## DESI DR1 selected observed galaxies (external data, positional markers)

DESI Collaboration et al. (2026), “Data Release 1 of the Dark Energy Spectroscopic
Instrument”; release and paper links: https://data.desi.lbl.gov/doc/releases/dr1/ .
Data licence/required attribution: https://data.desi.lbl.gov/doc/acknowledgments/ .
DESI declares **CC BY 4.0**, including commercial reuse with its stated citation,
acknowledgment and change-notice conditions. This is distinct from the code licence.
Changes: selected RA/Dec/redshift field, quality-primary filtering, assumed FLRW
comoving coordinate conversion, equal-weight diagnostic rasterization. No physical
photometry/galaxy shapes or survey completeness is claimed. Inputs include target
coordinates from Legacy Surveys DR9 as documented by DESI; no Gaia stellar VAC is
used by this galaxy path. The original raw subset and attribution snapshots stay
external, not redistributed by the repository.

Required DESI acknowledgment (retained policy text):

> This research used data obtained with the Dark Energy Spectroscopic Instrument
> (DESI). DESI construction and operations is managed by the Lawrence Berkeley
> National Laboratory. This material is based upon work supported by the U.S.
> Department of Energy, Office of Science, Office of High-Energy Physics, under
> Contract No. DE–AC02–05CH11231, and by the National Energy Research Scientific
> Computing Center, a DOE Office of Science User Facility under the same contract.
> Additional support for DESI was provided by the U.S. National Science Foundation
> (NSF), Division of Astronomical Sciences under Contract No. AST-0950945 to the
> NSF’s National Optical-Infrared Astronomy Research Laboratory; the Science and
> Technology Facilities Council of the United Kingdom; the Gordon and Betty Moore
> Foundation; the Heising-Simons Foundation; the French Alternative Energies and
> Atomic Energy Commission (CEA); the National Council of Humanities, Science and
> Technology of Mexico (CONAHCYT); the Ministry of Science and Innovation of Spain
> (MICINN), and by the DESI Member Institutions:
> www.desi.lbl.gov/collaborating-institutions.
>
> The DESI collaboration is honored to be permitted to conduct scientific research
> on I’oligam Du’ag (Kitt Peak), a mountain with particular significance to the
> Tohono O’odham Nation. Any opinions, findings, and conclusions or recommendations
> expressed in this material are those of the author(s) and do not necessarily
> reflect the views of the U.S. National Science Foundation, the U.S. Department of
> Energy, or any of the listed funding agencies.

The subset was obtained through the Astro Data Lab, part of the Community Science
and Data Center (CSDC) at NSF NOIRLab, operated by AURA under a cooperative agreement
with NSF: https://datalab.noirlab.edu/data/desi . SPARCL was not used.
