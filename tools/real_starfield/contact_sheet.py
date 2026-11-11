#!/usr/bin/env python3
"""Labelled contact sheet for out/quality/real_starfield (PIL from /workspace/data/astro/analysis-deps)."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/workspace/data/astro/analysis-deps")
from PIL import Image, ImageDraw

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "out/quality/real_starfield")
v = json.loads((OUT / "views.json").read_text())
recs = {r["name"]: r for r in v["views"] + v["flight"]}
ORDER = ["sol_orion_annotated", "sol_big_dipper_annotated", "sol_crux_annotated", "sol_galactic_centre",
         "sol_cygnus", "sol_anticentre", "mid_250pc_toward_orion", "out_500pc_l90_back_at_sol",
         "out_500pc_toward_gc", "out_1kpc_l90_back_at_sol", "out_1kpc_above_plane_down",
         "flight_006", "flight_012"]
W, H, LAB, COLS = 640, 360, 34, 3
tiles = [n for n in ORDER if (OUT / f"{n}.png").exists()]
rows = (len(tiles) + COLS - 1) // COLS
sky = OUT / "sol_allsky_galactic_aitoff.png"
SKY_H = W * COLS // 2 if sky.exists() else 0
sheet = Image.new("RGB", (W * COLS, rows * (H + LAB) + SKY_H + LAB * 2), (12, 12, 12))
d = ImageDraw.Draw(sheet)
d.text((8, 6), f"RS1 real observed starfield: {v['tile_stars']:,} catalogue stars (GCNS {v['tile_src_counts']['gcns']:,} / "
       f"Gaia DR3+BJ {v['tile_src_counts']['gaia_dr3_bj']:,} / Hipparcos-2 {v['tile_src_counts']['hipparcos2']:,}); "
       f"exposure {v['exposure']}, EPIC_GRADE off; Vulkan validation errors={v['validation_errors']} warnings={v['validation_warnings']}",
       fill=(230, 230, 230))
d.text((8, 20), "No synthesised stars. Rings on *_annotated = raw Hipparcos positions projected independently.", fill=(160, 160, 160))
y0 = LAB * 2
if SKY_H:
    sheet.paste(Image.open(sky).convert("RGB").resize((W * COLS, SKY_H)), (0, y0))
    d.text((8, y0 + 4), "sol_allsky_galactic_aitoff: all-sky from Sol, galactic Hammer-Aitoff, l=0 centre, l increases left", fill=(120, 255, 120))
    y0 += SKY_H
for i, n in enumerate(tiles):
    x, y = (i % COLS) * W, y0 + (i // COLS) * (H + LAB)
    sheet.paste(Image.open(OUT / f"{n}.png").convert("RGB").resize((W, H)), (x, y + LAB))
    r = recs.get(n.replace("_annotated", ""), {})
    d.text((x + 6, y + 3), n, fill=(120, 255, 120))
    d.text((x + 6, y + 17), f"{r.get('dist_from_sol_pc', 0):.0f} pc from Sol | complete to G={r.get('m_complete')} | "
           f"{r.get('submitted', 0):,} stars drawn | fov {r.get('fov_y_deg')}deg", fill=(200, 200, 200))
sheet.save(OUT / "contact_sheet.png")
print("wrote", OUT / "contact_sheet.png", sheet.size)
