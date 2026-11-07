#!/usr/bin/env python3
"""Assemble drape_lod panels + a Sentinel-2 true-colour reference crop into one sheet.
usage: tools/drape_lod_sheet.py <name> <lat> <lon> [size_m]   (after the drape_lod example)
Contains modified Copernicus Sentinel data 2024; © OpenStreetMap contributors (ODbL)."""
import json, sys
import numpy as np, rasterio
from pyproj import Transformer
from PIL import Image, ImageDraw

name, lat, lon = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
size = float(sys.argv[4]) if len(sys.argv) > 4 else 1600.0
base = f"out/quality/dem/drape_lod/{name}"
clip = "/workspace/data/geo/sentinel2/adelaide/S2B_54HTG_20241224_0_L2A_clip.tif"
PX = 512
with rasterio.open(clip) as src:
    rgb = src.read([1, 2, 3]).astype(np.float32) * 1e-4  # boa offset already applied (Earth Search)
    tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    my, mx = 111132.0, 111320.0 * np.cos(np.radians(lat))
    g = (np.arange(PX) + 0.5) / PX * size - size / 2
    e, n = np.meshgrid(g, -g)
    X, Y = tr.transform(lon + e / mx, lat + n / my)
    r, c = rasterio.transform.rowcol(src.transform, X.ravel(), Y.ravel())
    r, c = np.array(r).reshape(PX, PX), np.array(c).reshape(PX, PX)
    lin = rgb[:, r, c].transpose(1, 2, 0)
srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.clip(lin, 0, 1) ** (1 / 2.4) - 0.055)
s2 = Image.fromarray((np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8))
s2.save(f"{base}_sentinel2.png")
rep = {p["panel"]: p for p in json.load(open(f"{base}.json"))}

def lab(im, t):
    im = im.copy(); d = ImageDraw.Draw(im); d.rectangle([0, 0, PX, 18], fill=(0, 0, 0)); d.text((4, 3), t, fill=(255, 255, 255)); return im

top = [(Image.open(f"{base}_old_palette_3m.png"), "old hand palette (3 m)"),
       (Image.open(f"{base}_s2_3m.png"), "S2-fitted palette, objects tier (3 m)"),
       (s2, "Sentinel-2 L2A 2024-12-24 (10 m)")]
bot = []
for st in ["3.125", "12.5", "50", "100"]:
    p = rep[f"s2_{st}m"]
    bot.append((Image.open(f"{base}_s2_{st}m.png"),
                f"{p['step_m']:.1f} m: {p['coverage_px']:.2f} px/bldg  P{p['proxy']:.2f} M{p['merged']:.2f} O{p['objects']:.2f}"))
W = 4 * PX
sheet = Image.new("RGB", (W, 2 * PX + 22), (20, 20, 20))
for i, (im, t) in enumerate(top):
    sheet.paste(lab(im, t), (i * PX, 0))
for i, (im, t) in enumerate(bot):
    sheet.paste(lab(im, t), (i * PX, PX))
ImageDraw.Draw(sheet).text((4, 2 * PX + 5), f"{name} {lat},{lon} {size:.0f} m  |  (c) OpenStreetMap contributors (ODbL)  |  Contains modified Copernicus Sentinel data 2024", fill=(220, 220, 220))
sheet.save(f"{base}_sheet.png")
print(f"{base}_sheet.png")
