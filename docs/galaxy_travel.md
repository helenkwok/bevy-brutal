# GT1 galaxy_travel — Sol-neighbourhood continuous flight (GCNS)

Observed Gaia EDR3 GCNS / Smart et al. 2021 only. Nearby catalogue (~100 pc nominal), **not** the full Milky Way. Sol-local ICRS epoch J2016.0 static medians. Unresolved point sources; no invented radii/planets/arms.

## Launch

Interactive (needs a display + GUI session; this agent lacks GUI automation tools — parent runs desktop smoke):

```bash
cargo run -p render-ash --example galaxy_travel --features windowed --release -- \
  --catalogue /workspace/data/astro/gcns/gcns-v1.bin
```

Deterministic headless replay + PNG sheet (no display):

```bash
cargo run -p render-ash --example galaxy_travel --features windowed --release -- \
  --catalogue /workspace/data/astro/gcns/gcns-v1.bin \
  --frames 90 --capture out/quality/GT1_sheet --size 640x360
```

Optional: `--replay inputs.json`, `--record out.json`, `--headless`, `--no-validation`.

## Controls (interactive)

| Input | Action |
| --- | --- |
| W / S | Forward / back |
| A / D | Strafe left / right |
| R / F | Up / down |
| Mouse (LMB grab) | Look |
| Shift / Ctrl | Speed gear ±1 |
| Esc | Release grab; second Esc quits |
| Q | Quit |

Speeds are shown in **pc/s** (base 0.01 pc/s at gear 0, ×2 per gear). Approach stop: 1000 AU before unresolved point singularity.

Default look target: Gaia source_id `5853498713190525696` (nearest GCNS median ≈ Proxima direction/distance).

## Coverage HUD

Beyond ~100 pc: warning `outside nominal survey volume; coverage incomplete`. Stars with medians beyond 100 pc remain visible; they are never deleted. No synthetic fill.

## Present path

UniversePass live: HDR splats → fixed exposure `PRESENT_EXPOSURE=300` Reinhard+sRGB tonemap → R8G8B8A8 SDR. No per-frame HDR host readback. Sync: `immediate_submit` fence (not `device_wait_idle` each frame). Windowed present reuses `AshBackend` swapchain blit (`present_transfer_src`). Screenshot/SDR readback is explicit `--capture` verification only.

## Precision / limits

Whole-root-frame f64 precision limitations remain (large teleports/rebases). Catalogue is Sol-local; travel camera is hypothetical. Not commercial-cleared full-sky. SS2 / Earth / offscreen GCNS stills unchanged.

## Window title status (not in-frame HUD)

Each frame updates the window title with Sol-local `xyz_pc`, yaw/pitch (rad), speed (pc/s), selected count, grab/focus flags, and `OUTSIDE nominal 100pc survey` when beyond the nominal volume. There is **no** drawn on-screen overlay yet; do not treat stderr-only logs as a HUD.

## Sync note (present path)

After `present_transfer_src` blits the SDR image, the next `render_live` **timeline-waits** that consume before layout-transitioning SDR (fixes WRITE_AFTER_READ). Not `device_wait_idle` per frame.

## Coordinate note

Travel uses Sol-local ICRS metres with `Camera.origin = 0` (catalogue native frame). Nested `FrameTree` rebasing is not exercised on this path; whole-root f64 precision limits still apply for huge teleports.
