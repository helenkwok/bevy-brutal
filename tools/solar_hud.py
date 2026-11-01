#!/usr/bin/env python3
"""HUD annotation for `solar` shots: <out>/<shot>.png + .json -> <shot>_hud.png.

The render itself is a true-scale, true-brightness image (planets are sub-pixel points at system
distances). This overlay only annotates it, from the json the renderer wrote: orbit polylines
(projected heliocentric orbits at the shot epoch), body markers and labels, and a caption.
Usage: solar_hud.py <out_dir> <shot> [<shot> ...]
"""
import json
import sys
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
COL = {
    "sun": (255, 220, 120), "mercury": (180, 180, 180), "venus": (230, 210, 160), "earth": (110, 170, 255),
    "moon": (170, 170, 170), "mars": (230, 120, 80), "jupiter": (220, 180, 130), "saturn": (230, 200, 140),
    "uranus": (150, 220, 230), "neptune": (110, 150, 255),
}


def hud(out, shot):
    j = json.load(open(f"{out}/{shot}.json"))
    im = Image.open(f"{out}/{shot}.png").convert("RGB")
    W, H = im.size
    sx = W / j["size"][0]
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    fs = max(12, int(H / 60))
    font = ImageFont.truetype(FONT, fs)
    small = ImageFont.truetype(FONT, int(fs * 0.8))
    for o in j.get("orbits") or []:
        c = COL.get(o["name"], (200, 200, 200)) + (90,)
        seg = []
        for p in o["pts"] + [None]:
            if p is None:
                if len(seg) > 1:
                    d.line(seg, fill=c, width=max(1, int(sx)))
                seg = []
            else:
                seg.append((p[0] * sx, p[1] * sx))
    placed = []
    bodies = sorted(j["bodies"], key=lambda b: b["name"] != "sun")
    for b in bodies:
        if b.get("screen") is None:
            continue
        x, y = b["screen"][0] * sx, b["screen"][1] * sx
        if not (0 <= x < W and 0 <= y < H):
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < (2.2 * fs) ** 2 for px, py in placed):
            continue  # crowded (the Moon on the Earth): skip the label
        placed.append((x, y))
        c = COL.get(b["name"], (220, 220, 220))
        r = max(b.get("radius_px", 0) * sx + 5, 7, 40 * sx if b["name"] == "sun" else 0)
        d.ellipse([x - r, y - r, x + r, y + r], outline=c + (200,), width=max(1, int(sx)))
        d.text((x + r + 4, y - fs * 0.6), b["name"].capitalize(), font=font, fill=c + (230,))
    cam = j["camera"]
    cap = (f"{j['shot']}   true scale, true brightness (exposure {j['exposure']:.3g});  orbits, rings and labels are annotation"
           f"   |   {cam.get('dist_sun_au', 0):.2f} AU from the Sun, fov {cam['fov_y_deg']:.3g} deg   |   lavapipe {j['gpu_ms']} ms")
    d.rectangle([0, H - fs * 2, W, H], fill=(0, 0, 0, 150))
    d.text((fs * 0.6, H - fs * 1.6), cap, font=small, fill=(220, 220, 220, 255))
    Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB").save(f"{out}/{shot}_hud.png")


if __name__ == "__main__":
    for s in sys.argv[2:]:
        hud(sys.argv[1], s)
