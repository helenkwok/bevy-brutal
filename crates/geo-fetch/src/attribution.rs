//! Credit strings for the data this crate downloads.
//!
//! The texts follow the licensors' own wording:
//! - OSM: <https://www.openstreetmap.org/copyright> and the OSMF attribution guideline.
//! - Terrain: the "Required attribution" list in tilezen/joerd `docs/attribution.md`
//!   (the AWS Terrain Tiles bucket is the joerd output; the "Mapzen" line is only required for
//!   Mapzen's former hosted service, so it is offered separately as optional).
//!
//! Show [`OSM_SHORT`] wherever OSM-derived content is visible (credits screen, map UI), and put
//! [`credits_block`] in CREDITS.md / the game's credits. None of this is legal advice.

/// One-line OSM credit for on-screen use.
pub const OSM_SHORT: &str = "© OpenStreetMap contributors";

/// Full OSM credit for CREDITS files.
pub const OSM_FULL: &str = "Map data © OpenStreetMap contributors, available under the Open Database License (ODbL) 1.0. \
https://www.openstreetmap.org/copyright";

/// Licence notice to ship *next to* any OSM-derived database file you distribute
/// (for example a baked buildings or roads file). ODbL share-alike applies to such files;
/// rendered images and meshes shown in the game are a "Produced Work" and need attribution only.
pub const OSM_DERIVED_DB_NOTICE: &str = "This file contains data derived from OpenStreetMap. \
© OpenStreetMap contributors. It is made available under the Open Database License (ODbL) 1.0: \
https://opendatacommons.org/licenses/odbl/1-0/ . Any rights in individual contents of the database \
are licensed under the Database Contents License: https://opendatacommons.org/licenses/dbcl/1-0/ . \
This file is NOT covered by the MIT OR Apache-2.0 licence of the bevy-brutal source code.";

/// Where the OSM extract came from (Geofabrik's own credit request; their downloads are free).
pub const GEOFABRIK_SOURCE: &str = "OSM extracts downloaded from Geofabrik GmbH, https://download.geofabrik.de/";

/// One-line terrain credit for on-screen use.
pub const TERRAIN_SHORT: &str = "Terrain: AWS Terrain Tiles (Tilezen) — USGS, NOAA, Copernicus EU-DEM, and others";

/// The per-source attribution lines required by the AWS Terrain Tiles / Tilezen joerd data.
pub const TERRAIN_SOURCES: &[&str] = &[
    "ArcticDEM terrain data DEM(s) were created from DigitalGlobe, Inc., imagery and funded under National Science Foundation awards 1043681, 1559691, and 1542736;",
    "Australia terrain data © Commonwealth of Australia (Geoscience Australia) 2017;",
    "Austria terrain data © offene Daten Österreichs – Digitales Geländemodell (DGM) Österreich;",
    "Canada terrain data contains information licensed under the Open Government Licence – Canada;",
    "Europe terrain data produced using Copernicus data and information funded by the European Union - EU-DEM layers;",
    "Global ETOPO1 terrain data U.S. National Oceanic and Atmospheric Administration",
    "Mexico terrain data source: INEGI, Continental relief, 2016;",
    "New Zealand terrain data Copyright 2011 Crown copyright (c) Land Information New Zealand and the New Zealand Government (All rights reserved);",
    "Norway terrain data © Kartverket;",
    "United Kingdom terrain data © Environment Agency copyright and/or database right 2015. All rights reserved;",
    "United States 3DEP (formerly NED) and global GMTED2010 and SRTM terrain data courtesy of the U.S. Geological Survey.",
];

/// Optional extra line some projects add because the tiles were produced by Mapzen.
pub const TERRAIN_MAPZEN: &str = "Mapzen";

/// Where the terrain tiles came from.
pub const TERRAIN_SOURCE: &str = "Elevation tiles from the Terrain Tiles open dataset on AWS (Registry of Open Data), https://registry.opendata.aws/terrain-tiles/";

/// Multi-line terrain credit: a heading plus one `* ` bullet per source.
pub fn terrain_full() -> String {
    let mut s = String::from("Terrain data: AWS Terrain Tiles (Tilezen joerd). Sources:\n");
    for line in TERRAIN_SOURCES {
        s.push_str("* ");
        s.push_str(line);
        s.push('\n');
    }
    s.push_str(TERRAIN_SOURCE);
    s.push('\n');
    s
}

/// Which datasets a credits block should cover.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Used {
    pub osm: bool,
    pub terrain: bool,
}

/// Markdown/plain-text credits block for the datasets in `used`, ready for CREDITS.md.
pub fn credits_block(used: Used) -> String {
    let mut s = String::new();
    if used.osm {
        s.push_str(OSM_FULL);
        s.push('\n');
        s.push_str(GEOFABRIK_SOURCE);
        s.push('\n');
    }
    if used.terrain {
        if !s.is_empty() {
            s.push('\n');
        }
        s.push_str(&terrain_full());
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn osm_strings_exact() {
        assert_eq!(OSM_SHORT, "© OpenStreetMap contributors");
        assert!(OSM_FULL.starts_with("Map data © OpenStreetMap contributors"));
        assert!(OSM_FULL.contains("Open Database License (ODbL) 1.0"));
        assert!(OSM_FULL.ends_with("https://www.openstreetmap.org/copyright"));
        assert!(OSM_DERIVED_DB_NOTICE.contains("https://opendatacommons.org/licenses/odbl/1-0/"));
        assert!(OSM_DERIVED_DB_NOTICE.contains("NOT covered by the MIT OR Apache-2.0"));
    }

    #[test]
    fn terrain_list_matches_joerd() {
        assert_eq!(TERRAIN_SOURCES.len(), 11);
        let joined = TERRAIN_SOURCES.join("\n");
        for needle in [
            "National Science Foundation awards 1043681, 1559691, and 1542736",
            "© Commonwealth of Australia (Geoscience Australia) 2017",
            "Digitales Geländemodell (DGM) Österreich",
            "Open Government Licence – Canada",
            "funded by the European Union - EU-DEM layers",
            "ETOPO1 terrain data U.S. National Oceanic and Atmospheric Administration",
            "INEGI, Continental relief, 2016",
            "Land Information New Zealand",
            "© Kartverket",
            "© Environment Agency copyright and/or database right 2015",
            "3DEP (formerly NED) and global GMTED2010 and SRTM terrain data courtesy of the U.S. Geological Survey.",
        ] {
            assert!(joined.contains(needle), "missing {needle:?}");
        }
        // Sorted like the joerd list (alphabetical by region).
        let mut sorted = TERRAIN_SOURCES.to_vec();
        sorted.sort();
        assert_eq!(sorted, TERRAIN_SOURCES);
    }

    #[test]
    fn terrain_full_format() {
        let t = terrain_full();
        assert_eq!(t.lines().filter(|l| l.starts_with("* ")).count(), 11);
        assert!(t.starts_with("Terrain data: AWS Terrain Tiles (Tilezen joerd). Sources:\n* ArcticDEM"));
        assert!(t.ends_with("https://registry.opendata.aws/terrain-tiles/\n"));
    }

    #[test]
    fn credits_block_selects() {
        assert_eq!(credits_block(Used::default()), "");
        let o = credits_block(Used { osm: true, terrain: false });
        assert_eq!(o, format!("{OSM_FULL}\n{GEOFABRIK_SOURCE}\n"));
        let both = credits_block(Used { osm: true, terrain: true });
        assert_eq!(both, format!("{OSM_FULL}\n{GEOFABRIK_SOURCE}\n\n{}", terrain_full()));
        let t = credits_block(Used { osm: false, terrain: true });
        assert_eq!(t, terrain_full());
    }
}
