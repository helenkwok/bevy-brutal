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
