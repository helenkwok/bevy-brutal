# Performance research: dense forests + grass on a GPU-driven Vulkan renderer (round 9, 22 Oct 2026)

Measured hotspot before this round (lavapipe = llvmpipe software Vulkan, 1920x1080 with 2x2 SSAA,
`quality --scene mountains`): close-up ~3.0 s/frame (shadow ~685 / main ~2343 ms), of which
main terrain 786 ms and main instance meshes 1545 ms; default view main terrain 599 / instance
meshes 588 / impostors 158 ms. The renderer is forward-shaded with no depth prepass: the terrain
was fully shaded and then overdrawn by grass blades and needle cards.

**Lavapipe is not a GPU.** It rasterises on the CPU (8-wide SIMD per core). Its costs are dominated
by fragment shading and per-pixel depth tests. Vertex throughput and triangle setup are also slow,
but relative to fragments they matter less than on a real GPU. Every number here is a lavapipe
number, not real-GPU fps. The GPU-independent metrics (instances, triangles and cascades drawn,
shaded fragments) are reported where they are available.

## Sources and what they say

| Technique | Source | Takeaway for us |
|---|---|---|
| GPU-driven culling + two-phase Hi-Z occlusion | Haar & Aaltonen, "GPU-Driven Rendering Pipelines", SIGGRAPH 2015 Advances, https://advances.realtimerendering.com/s2015/aaltonenhaar_siggraph2015_combined_final_footer_220dpi.pdf ; Aaltonen's thread on Nanite using the same algorithm, https://threadreaderapp.com/thread/1402951008830070790.html | Phase 1 culls against the previous frame's depth pyramid. Phase 2 re-tests what was culled against the refreshed pyramid. A depth prepass of the "best occluders" feeds Hi-Z *and* early-Z. They report only a small overall geometry gain (<10%), with 30-80% of shadow triangles culled. It is best for partial occlusion such as vegetation clouds. |
| Terrain + forest culling | Far Cry 5 terrain talk, GDC 2018, https://gdcvault.com/play/1025480/Terrain-Rendering-in-Far-Cry (notes: https://wolfand11.github.io/blogs/graphics/TerrainRenderingInFarCry5.html) | The pipeline is GPU quadtree, then 8x8 patches, then frustum + conservative Hi-Z occlusion (a best-occluder prepass on console, a software occlusion buffer on PC) + back-face cone culling. A compute shader emits grass instances into one indirect draw. The last grass LOD is simple quads matched to the terrain colour. |
| Procedural grass | Wohllaib, "Procedural Grass in Ghost of Tsushima", GDC 2021, https://www.youtube.com/watch?v=Ibe1JBF5i5Y , slides https://archive.thedatadungeon.com/ghost_of_tsushima_2020/documents/gdc_2021/gdc_2021_procedural_grass_in_got.pdf | Per-tile compute generates blades with distance, frustum and occlusion culling into indirect draws: ~83k of 1M considered blades in ~2.5 ms. Cards were rejected for **overdraw**; occlusion culling was "a small perf win in most shots". |
| GPU placement | van Muijden, "GPU-Based Run-Time Procedural Placement in Horizon Zero Dawn", GDC 2017, https://www.guerrilla-games.com/read/gpu-based-procedural-placement-in-horizon-zero-dawn | Dense deterministic placement on the GPU around the player (we already place deterministically and cull on the GPU). |
| Visibility buffer for foliage | McLaren, "Adventures with Deferred Texturing in Horizon Forbidden West", GDC 2022, https://www.guerrilla-games.com/read/adventures-with-deferred-texturing-in-horizon-forbidden-west ; Burns & Hunt, "The Visibility Buffer", JCGT 2013, https://jcgt.org/published/0002/02/04/paper.pdf | Foliage and alpha-tested geometry draw a visibility-buffer prepass (alpha test included). Shading then runs once per pixel in compute, with software VRS on top. The point is to **remove shading overdraw** from alpha-tested foliage. |
| Nanite foliage | UE 5.7/5.8 docs, "Nanite Foliage", https://dev.epicgames.com/documentation/unreal-engine/nanite-foliage ; Karis, "Nanite: A Deep Dive", SIGGRAPH 2021, https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf | Assemblies instance branch parts, and voxels replace pixel-sized clusters at distance (no billboards). The compute rasteriser for <32 px triangles is ~3x faster than hardware raster for tiny triangles. The 64-bit visibility buffer is written with an atomic max. |
| Billboards / impostor LOD | SpeedTree docs, LOD and billboards, https://docs.speedtree.com/doku.php?id=overview_level-ofdetail , https://docs.speedtree.com/doku.php?id=compiler_billboards | Smooth LOD (fewer cards), then a crossfade to 6-10 azimuthal billboards with normal maps, using alpha-to-coverage / dither for the fade. |
| Shadow caching | UE5 Virtual Shadow Maps docs, https://dev.epicgames.com/documentation/unreal-engine/virtual-shadow-maps-in-unreal-engine ; "Virtual Shadow Maps in Fortnite Battle Royale Chapter 4", https://www.unrealengine.com/tech-blog/virtual-shadow-maps-in-fortnite-battle-royale-chapter-4 ; Schober, "A sampling of UE4 rendering improvements" (GDC 2017), https://gpuopen.com/download/Capsaicin-Cream-GDC2017-ArneSchober.pdf | Pages and cascades are cached between frames, with static geometry separated so that dynamic objects don't invalidate it. A moving sun or WPO foliage invalidates the cache, which is why Fortnite doesn't cache sun pages. |
| VRS | VK_KHR_fragment_shading_rate proposal, https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_fragment_shading_rate.html ; Gears Tactics VRS, https://devblogs.microsoft.com/directx/gears-tactics-vrs/ ; DOOM: The Dark Ages variable-rate compute (GPC 2025), https://static.graphicsprogrammingconference.com/public/2025/talks/variable-rate-compute-shaders-in-doom-the-dark-ages/Fuller-Hammer-variable-rate-compute-shaders-in-doom-the-dark-ages.pdf | Typically 10-19% GPU savings. **Foliage is the worst case** (high-frequency detail). |

## What we adopted, and the measurements

All A/B runs are back to back on the same box. Single-run lavapipe noise is ±5-10%.

1. **Foliage depth prepass with alpha test (adopted).** This follows Haar/Aaltonen's "best occluders
   prepass" and Forbidden West's foliage visibility prepass.
   - The instance meshes are drawn depth-only first. Whole meshes have no fragment shader; crossfading
     and needle/leaf cards run the *same* discard code as the shading FS (`instDepthFS`).
   - The terrain then draws normally, so early-Z rejects it behind the vegetation.
   - The instance meshes then shade with depth `EQUAL` and no writes: one shaded fragment per pixel.
   - Impostors and sky are unchanged.
   - Close-up main: **2490 → 1546 ms (-38%)**. Default: **1374 → 1188 ms (-14%)**.
   - The image is pixel-identical on the default view. 9 of 8.3M pixels differ (by at most 10/255) on
     the close-up, from exact depth ties between coplanar card triangles.
   - Also prepassing the terrain (and shading it with `EQUAL` too): close-up main -30% but default
     +9%, so instance-only is the better trade-off.
2. **Static shadow cascade cache with world-anchored cascades (adopted).**
   - `sun_cascade` now snaps the cascade centre to a texel grid in *world* light space. Before, the
     grid was camera-relative, so the cascades also shimmered when the camera translated. The radius
     is now closed-form, so it no longer depends on camera rotation.
   - The far cascades c2/c3 draw camera-independent casters (terrain + sun-facing impostor cards). They
     get a key: grid cell, slice, sun, map size, plus a 64-bit fingerprint of all caster bytes
     (42 ms CPU, once per scene).
   - The backend keeps the shadow maps between frames and redraws a cascade only when its key changes.
     c0/c1 use camera-distance LOD, so they always redraw.
   - Default view, warm frame: shadow **670 → 280 ms**, frame **1842 → 1444 ms**. The image is
     pixel-identical with and without the cache.
   - Honest caveat: a warm hit needs the same snapped cell. With a static camera that's every frame.
     When walking, the c3 cell (texel ~5-9 m) changes every ~1-2 s at 5 m/s, and c2 (texel ~1.6 m)
     every ~0.3 s. Any camera *rotation* moves the slice centroid and misses. A real engine would
     scroll the map (clipmap-style) or refresh c3 every N frames. Cold frames cost the same as before.
3. **Workload metrics (added).** `QualityTimings::camera_mesh_draws` gives per-mesh camera-view draws
   and triangles (`QUALITY_MESH_STATS=1`).
   - Close-up: **spruce LOD2 at 260-650 m = 750k tris (1820 draws), fir LOD2 = 481k**. LOD1 at
     60-260 m is only 207k + 88k. So "mid-distance card trees" is wrong: the triangle mass is the
     far low-crown LOD.

## What we rejected, and why

- **Impostors earlier (`VEG_IMP_AT` 400).** Camera triangles fall 45% (close-up 2.45M → 1.36M), but main
  time *rises* (default 1152 → 1348 ms; impostor shading 159 → 461 ms).
  - Our impostor fragment shader blends 3 frames with parallax and writes exact depth (late-Z), so on
    a fragment-bound rasteriser it costs more than ~400-triangle crowns.
  - It could pay off if the impostors went into the prepass too (an alpha-only depth FS); not done.
- **Hi-Z / two-pass occlusion culling.** With the prepass, occluded instances no longer cost shading,
  only vertex and raster work. The impostor experiment shows that removing even 45% of triangles
  doesn't help on lavapipe.
  - Haar/Aaltonen themselves report <10% geometry gains.
  - It would matter on a real GPU in occluded forest views. Not worth its complexity (pyramid build,
    second-phase draws) for this target.
- **Meshlets / mesh shaders / compute rasterisation (Nanite-style).** Lavapipe exposes VK_EXT_mesh_shader
  (task/mesh subgroup stages are listed), but a software rasteriser gains nothing from hardware
  meshlet culling. A compute rasteriser for tiny far-crown triangles is a large project, so this is
  deferred.
- **VRS.** Not useful on lavapipe; foliage is also the documented worst case.

## Final numbers (23_final vs 22_final, lavapipe, same box)

Reported as shadow / main / frame ms. "Warm" means the static c2/c3 shadow cascades come from the
cache: 2 of 4 cascades on every timed run, because the example renders a warm-up frame first.
"Cold" means `SHADOW_CACHE=0`, so every cascade is redrawn, as on a frame where the camera has just
moved to a new cell.

See `starter.md` "Mountains round 9" for the table. Highlights: close-up frame 3028 → 2208 ms cold / 1955 ms warm
(-27% / -35%), and the default view 1987 → 1904 cold / 1513 warm. The cave (no vegetation, spot shadow) is
unchanged and pixel-identical.

## Visual work done in the same round (not performance)

- The spruce/fir/larch cards use a shoot-cluster alpha pattern (bottle-brush side shoots every 0.08
  along the branch, ragged needle grain, per-shoot tone, dark interior, sky gaps) instead of the
  fern-frond pattern. Cards fade out when seen edge-on (`card_facing_keep`), which removes the X
  artefacts of crossed cards. The impostor atlas re-bakes from LOD0, and the CPU bake mirrors the
  pattern in `impostor.rs`.
- Ambient light is an L2 SH irradiance built from the sky-view LUT (Ramamoorthi & Hanrahan 2001,
  https://graphics.stanford.edu/papers/envmap/envmap.pdf).
  - Below ~10-20° elevation the environment is replaced by a sunlit mountainside / forest wall, and
    below 0° by the valley floor (albedo x (sun + sky) / π), which gives the ground bounce.
  - It uses multi-bounce AO (Jimenez et al. 2016, "Practical Real-Time Strategies for Accurate
    Indirect Occlusion", Activision tech report
    ATVI-TR-16-01).
  - The art-tuned LUT is far too saturated to light with: zenith irradiance B/R ≈ 3.4 in linear.
    The bake therefore keeps 35% of the sky chroma, which brings B/R to ~2 (clear-sky diffuse is
    roughly 1.6-2).
  - Shade-grass hue moves from 129° (teal) to 103° (green). Fall back with `ENV_SH=0`.
