# GT1 galaxy_travel — launch & limits

**Commit under test:** see `git rev-parse HEAD` (desktop smoke: `3386c9a`).

Observed Gaia EDR3 GCNS only (external `/workspace/data/astro/gcns/gcns-v1.bin`). Nearby survey (~100 pc nominal), **not** full Milky Way. Unresolved point sources.

## Launch

```bash
# Interactive (needs a real display / human or sanctioned Computer session)
cargo run -p render-ash --example galaxy_travel --features windowed --release -- \
  --catalogue /workspace/data/astro/gcns/gcns-v1.bin --size 960x540

# Headless deterministic replay + PNG sheet (no GUI)
cargo run -p render-ash --example galaxy_travel --features windowed --release -- \
  --catalogue /workspace/data/astro/gcns/gcns-v1.bin \
  --frames 90 --capture out/quality/GT1_sheet --size 640x360
```

## Controls

| Input | Action |
| --- | --- |
| W/S A/D R/F | Move |
| LMB | Cursor grab (Confined, else Locked) |
| Mouse | Look via `DeviceEvent::MouseMotion` **relative** deltas (not `CursorMoved`) |
| Shift / Ctrl | Speed gear ±1 |
| Esc | Release grab; second Esc quits |
| Q | Quit |

**Status readout:** dynamic **window title** (xyz pc, yaw/pitch, speed, selected, grab/focus, coverage). No in-frame overlay HUD.

## Verification status (honest)

- Sync: 0 `WRITE_AFTER_READ` after timeline-wait fix (`3386c9a`).
- Desktop smoke (`3386c9a`): validation clean; title; grab OK; Esc; key taps change selected IDs; resize/focus; Q clean.
- **Partial milestone:** absolute Computer pointer warps did **not** change yaw/pitch — expected for relative `DeviceEvent::MouseMotion` (raw XI_RawMotion), not proof look is broken. Held Shift/Ctrl/WASD not verified by tap-only automation. Human relative mouse + held keys still needed for “fully playable”.
- Executor lacks UI tooling; parent has desktop. Do not use shell/X11 automation from this agent.

## Resources / timing (lavapipe sample, 640×360, 90 frames)

| metric | mean | p95 |
| --- | ---: | ---: |
| select_ms | ~9.3 | ~10.3 |
| submit_ms | ~15.7 | ~17.0 |
| total_ms | ~30.8 | ~60.8 |

- Eligible stars: **330881**; GPU select cap **65536**; spatial gather (not full 330k photometric rescan every frame, but per-frame `seen[330881]` bitvec alloc remains).
- No per-frame HDR host readback on live path; SDR capture only with `--capture`.
- Device label: `llvmpipe` ≠ native GPU fps.

## Limits

- Not full galaxy / no synthetic fill / no DESI·CNS5 in travel.
- Sol-local f64 root only (no nested FrameTree rebasing in this path).
- Present sync: timeline wait for transfer-src consumers; single-queue live path (not mesh multi-FIF).
- See also `docs/galaxy_travel.md`.
