# RS1: real observed starfield

This replaces the procedural starfield for navigable travel. It uses Gaia DR3 with Bailer-Jones EDR3 distances (G < 13, ≤ 1 kpc), Hipparcos-2 for bright and saturated stars, and GCNS, which is authoritative within 100 pc. Every rendered star is a catalogue row. Data stays outside git under `/workspace/data/astro/real_starfield/`, with provenance in `MANIFEST.json` there.

- `tools/real_starfield/`: TAP fetch (`tap.py`, `fetch_gaia_esa_chunks.py`), normalisation (`ingest.py` → RSRC0001), checks (`annotate.py`, `band_check.py`, `bright_check.py`), `contact_sheet.py` and `manifest.py`.
- `crates/planet/src/starfield.rs`:
  - crossmatch merge;
  - RSTILE01 octree, brightness-prioritised with 4096 stars per node, 16-B quantised records and node-relative u32 positions in the Sol-local ICRS parsec frame;
  - best-first plan with an exact completeness magnitude;
  - LRU streaming;
  - projection.
- `crates/planet/examples/build_starfield.rs` turns RSRC into `tiles/rs1.rstile`.
- `crates/render-ash/examples/real_starfield.rs` renders the reference views, the all-sky map, the band check and a streaming flight into `out/quality/real_starfield/`. See REPORT.md there.
- `galaxy_travel --tiles <rs1.rstile>` gives interactive flight on the real tiles: `[`/`]` change the base speed gear, up to 163.84 pc/s.

The camera handedness fix landed with RS1: screen-right = forward × up, and the tonemap no longer flips rows. Before it, skies rendered rolled 180°. As a result, GT1 mouse yaw and strafe now match what you see on screen.
