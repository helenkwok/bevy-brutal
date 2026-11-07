# bevy-brutal

Experimental Bevy 0.20 + ash/Vulkan work toward a seamless universe → solar system → Earth → underground / caves experience at 60fps+.

Inspired by [BRUTAL](https://github.com/ahwootech/BRUTAL)'s design ideas; this is an independent Rust codebase and does **not** include BRUTAL source.

See [starter.md](starter.md) for the working notes, and [CREDITS.md](CREDITS.md) for third-party and data licences.

## DEM close-tile streaming (`earth` example)

With `DEM=<terrain-grid dir>` and `OSM=<osm-extract dir>`, the close graft follows
camera ground look-at, not the DEM origin. It keeps the existing 3 km half extent
(`DEM_CLOSE_KM`; 0 disables) and resolution (`DEM_TEX=1024`, or 2048 for 2.93 m texels).
Targets snap to a 1.5 km grid with 1.125 km recenter hysteresis at the default extent.
Sky/outside-DEM targets use permanent outer/mid coverage. The previous complete close
tile remains visible at its old location while its replacement is baking.

- `DEM_BAKE_ROWS=32`, `DEM_BAKE_MS=4`: stop before starting another row when either
  per-render-frame limit is reached. A row is indivisible and can exceed 4 ms; actual
  elapsed time is logged. Zero rows or time pauses baking.
- `DEM_CACHE_TILES=3` (clamped to 0..16): LRU cache of at most three complete CPU close tiles plus **one**
  partial tile (128 MiB raw close data at 2048; separate from far/mid, mips and uploads).
  Zero disables close work; one permits atomic replacement. A new target cancels stale
  partial work; no backlog or prefetch queue. Cache lifetime is one DEM/size/sun setup.
- Steady-state GPU residency is two far/mid layers and at most one close layer.
  Replacement temporarily holds both old/new arrays (up to six layers plus staging). Complete
  current-target tiles alone activate. Uploads/mipmap generation use the existing
  GPU-idle descriptor swap and release the old images safely; their cost is logged
  separately and is **not** covered by the bake time budget.
  Configuration clamps rows to 0..4096 and milliseconds to 0..1000; `DEM_TEX` must
  be a power of two in 1..=4096. Invalid/non-finite budget values use defaults.

The example is an offline renderer, not an interactive game loop. Every actual render
(including `REPEAT=N`) advances the budget; a single-frame screenshot may still show
far/mid while warming. Outer/mid setup and OSM loading remain synchronous. An ordered
route such as `--shot DEM_adelaide_cbd,DEM_adelaide_coast,DEM_adelaide_hills` shares
cached tiles while the sun is unchanged. Keep `EPIC_GRADE` unset for physical defaults.
Downloaded geodata belongs outside the source tree; see `crates/geo-fetch/README.md`.

## License

Except where noted below, this repository is dual-licensed under either of:

* MIT License ([LICENSE-MIT](LICENSE-MIT))
* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you shall be dual-licensed as above, without any
additional terms or conditions.

### Third-party and data

See [NOTICE](NOTICE) and [CREDITS.md](CREDITS.md). In particular, Earth imagery
and DEM data are **not** covered by the MIT/Apache dual licence.
