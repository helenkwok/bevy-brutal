#!/usr/bin/env python3
"""SS2 readability layer for `solar` shots: <out>/<shot>.png + .json (+ loupes / zoom) -> <shot>_vis.png.

The physical render stays untouched in the top 1920x1080. Everything that makes it readable is
annotation or a separate true render:
  * on the image: anti-aliased orbit polylines at the epoch, body markers, labels placed around
    collisions (no label is dropped silently: a body without room is named on its loupe), the
    Sun's position (and the occulter of a deep exposure), a scale bar valid at the distance of the body nearest the centre;
  * a panel below: one loupe per body that is under 12 px on screen (a true render through a long
    lens on the same line of sight, at its real phase; exposure normalised to the irradiance at
    1 AU, so albedo differences stay), each with V magnitude, angular size, phase, distance and
    magnification; and the inner-system zoom (a true render from the same camera, narrower fov)
    with its own orbits, tied to a box on the main image.
Usage: solar_vis.py <out_dir> <shot> [<shot> ...]
"""
import datetime
import json
import math
import sys

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
COL = {
    "sun": (255, 220, 120), "mercury": (190, 190, 190), "venus": (235, 215, 165), "earth": (120, 175, 255),
    "moon": (175, 175, 175), "mars": (235, 125, 85), "jupiter": (225, 185, 135), "saturn": (235, 205, 145),
    "uranus": (155, 225, 232), "neptune": (120, 160, 255),
}
ORDER = ["mercury", "venus", "earth", "moon", "mars", "jupiter", "saturn", "uranus", "neptune"]
SS = 2  # overlay supersampling (anti-aliased lines)
PANEL = 300


def fmt_mag(m):
    if m >= 1e6:
        return f"x{m / 1e6:.3g}M"
    if m >= 1e3:
        return f"x{m / 1e3:.3g}k"
    return f"x{m:.3g}"


def fmt_ang(a):
    if a >= 60:
        return f"{a / 60:.3g}'"
    if a >= 0.1:
        return f"{a:.3g}\""
    return f"{a * 1000:.3g} mas"


def text_o(d, xy, s, font, fill, o=2):
    """Text with a dark outline (legible on stars and on the Sun's halo)."""
    d.text(xy, s, font=font, fill=fill, stroke_width=o, stroke_fill=(0, 0, 0, 200))


def polyline(d, pts, fill, width, ox=0, oy=0, sx=1.0):
    seg = []
    for p in pts + [None]:
        if p is None:
            if len(seg) > 1:
                d.line(seg, fill=fill, width=width, joint="curve")
            seg = []
        else:
            seg.append(((p[0] * sx + ox) * SS, (p[1] * sx + oy) * SS))


def vis(out, shot):
    j = json.load(open(f"{out}/{shot}.json"))
    v = j["vis"]
    W, H = j["size"]
    base = Image.open(f"{out}/{shot}.png").convert("RGBA")
    canvas = Image.new("RGBA", (W, H + PANEL), (8, 8, 10, 255))
    canvas.paste(base, (0, 0))
    ov = Image.new("RGBA", (W * SS, (H + PANEL) * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    fs = 18
    font = ImageFont.truetype(FONT, fs * SS)
    bold = ImageFont.truetype(BOLD, fs * SS)
    small = ImageFont.truetype(FONT, int(fs * 0.78) * SS)
    tiny = ImageFont.truetype(FONT, int(fs * 0.7) * SS)
    # orbits
    for o in j.get("orbits") or []:
        c = COL.get(o["name"], (200, 200, 200))
        pts = [p if p is not None and -50 <= p[0] <= W + 50 and -50 <= p[1] <= H else None for p in o["pts"]]
        polyline(d, pts, c + (110,), SS + 1)
    # the Sun / occulter
    sun = v.get("sun_screen")
    if sun and 0 <= sun[0] < W and 0 <= sun[1] < H:
        if v.get("occ_px", 0) > 0:
            r = v["occ_px"]
            x, y = sun
            d.ellipse([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS], outline=(255, 220, 120, 160), width=SS)
    # markers + labels
    boxes = []  # placed label boxes (main-image px)
    bodies = {b["name"]: b for b in j["bodies"]}
    loupe_names = {lp["name"] for lp in v["loupes"]}
    marks = []
    for b in j["bodies"]:
        if not b.get("screen"):
            continue
        x, y = b["screen"]
        if not (0 <= x < W and 0 <= y < H):
            continue
        name = b["name"]
        c = COL.get(name, (220, 220, 220))
        r = max(b.get("radius_px", 0) + 4, 6)
        if name == "sun":
            r = max(b.get("radius_px", 0) + 10, v.get("occ_px", 0) + 4, 14)
        if name == "moon" and "earth" in bodies and bodies["earth"].get("screen"):
            ex, ey = bodies["earth"]["screen"]
            if (ex - x) ** 2 + (ey - y) ** 2 < 15 ** 2:
                continue  # sits on the Earth's marker: named "Earth + Moon"
        marks.append((name, x, y, r, c))
        d.ellipse([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS], outline=c + (210,), width=SS)
        boxes.append((x - r, y - r, x + r, y + r))

    def free(bx):
        if bx[0] < 2 or bx[1] < 2 or bx[2] > W - 2 or bx[3] > H - 2:
            return False
        return all(bx[2] < o[0] or bx[0] > o[2] or bx[3] < o[1] or bx[1] > o[3] for o in boxes)

    for name, x, y, r, c in marks:
        label = name.capitalize()
        if name == "earth" and "moon" in bodies and not any(m[0] == "moon" for m in marks):
            label = "Earth + Moon"
        tw = d.textlength(label, font=font) / SS
        th = fs * 1.1
        placed = False
        # try around the marker, then further out with a leader
        for dist in (r + 4, r + 22, r + 48):
            for ang in (0, 180, -40, 40, 220, 140, -90, 90):
                a = math.radians(ang)
                cx, cy = x + math.cos(a) * dist, y + math.sin(a) * dist
                lx = cx if math.cos(a) >= -0.1 else cx - tw
                ly = cy - th / 2
                bx = (lx - 2, ly - 2, lx + tw + 2, ly + th + 2)
                if free(bx):
                    boxes.append(bx)
                    if dist > r + 4:
                        d.line([(x + math.cos(a) * r) * SS, (y + math.sin(a) * r) * SS, cx * SS, cy * SS], fill=c + (150,), width=SS)
                    text_o(d, (lx * SS, ly * SS), label, font, c + (240,), o=SS * 2)
                    placed = True
                    break
            if placed:
                break
    # scale bar (perpendicular to the view at the distance of the body nearest the centre)
    ppa = v.get("px_per_au") or 0
    at = (v.get("scale_at") or "sun").capitalize()
    if ppa > 0:
        best = None
        for n in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20):
            if 90 <= n * ppa <= 320:
                best = n
        if best is None:
            best = min((0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20), key=lambda n: abs(n * ppa - 200))
        L = best * ppa
        if 20 < L < W * 0.6:
            x0, y0 = 30, H - 40
            d.line([x0 * SS, y0 * SS, (x0 + L) * SS, y0 * SS], fill=(235, 235, 235, 230), width=2 * SS)
            for xx in (x0, x0 + L):
                d.line([xx * SS, (y0 - 6) * SS, xx * SS, (y0 + 6) * SS], fill=(235, 235, 235, 230), width=2 * SS)
            lm = best * 499.005 / 60.0
            lt = f"{lm:.3g} light-min" if lm >= 1 else f"{lm * 60:.3g} light-s"
            km = best * 1.495978707e8
            kt = (f"{km / 1e9:.3g} billion km" if km >= 1e9 else f"{km / 1e6:.3g} million km" if km >= 1e6
                  else f"{km:,.0f} km")
            text_o(d, (x0 * SS, (y0 - 30) * SS), f"{best:g} AU  =  {kt}  =  {lt}", small, (235, 235, 235, 240), o=SS * 2)
            text_o(d, (x0 * SS, (y0 + 10) * SS), f"at the {at}'s distance", tiny, (170, 170, 170, 220), o=SS)
    # ---- panel ----
    py = H + 10
    x = 12
    z = v.get("zoom")
    if z:
        zi = Image.open(f"{out}/{z['file']}").convert("RGBA")
        zs = z["size"]
        zx, zy = x, H + (PANEL - zs) // 2
        canvas.paste(zi, (zx, zy))
        for o in z.get("orbits") or []:
            c = COL.get(o["name"], (200, 200, 200))
            pts = [p if p is None or (0 <= p[0] <= zs and 0 <= p[1] <= zs) else None for p in o["pts"]]
            polyline(d, pts, c + (120,), SS, zx, zy)
        for b in z["bodies"]:
            bx, by = b["screen"]
            if b["name"] in ("sun", "moon") or not (6 <= bx <= zs - 6 and 6 <= by <= zs - 6):
                continue
            c = COL.get(b["name"], (220, 220, 220))
            d.ellipse([(zx + bx - 4) * SS, (zy + by - 4) * SS, (zx + bx + 4) * SS, (zy + by + 4) * SS], outline=c + (220,), width=SS)
            text_o(d, ((zx + bx + 6) * SS, (zy + by - 8) * SS), b["name"].capitalize(), tiny, c + (230,), o=SS)
        d.rectangle([zx * SS, zy * SS, (zx + zs) * SS, (zy + zs) * SS], outline=(200, 200, 200, 160), width=SS)
        text_o(d, ((zx + 6) * SS, (zy + 4) * SS), f"inner system, same camera, fov {z['fov_y_deg']:.3g} deg", tiny, (220, 220, 220, 240), o=SS)
        cx, cy, hh = z["main_rect"]
        rect = (cx - hh, cy - hh, cx + hh, cy + hh)
        d.rectangle([c * SS for c in rect], outline=(200, 200, 200, 150), width=SS)
        d.line([rect[0] * SS, rect[3] * SS, zx * SS, zy * SS], fill=(200, 200, 200, 90), width=SS)
        d.line([rect[2] * SS, rect[3] * SS, (zx + zs) * SS, zy * SS], fill=(200, 200, 200, 90), width=SS)
        x = zx + zs + 16
    loupes = sorted(v["loupes"], key=lambda lp: lp["screen"][0])
    if loupes:
        n = len(loupes)
        ls = v["loupe_size"]
        avail = W - x - 12
        gap = max(4, min(40, (avail - n * ls) / max(n, 1)))
        span = n * ls + (n - 1) * gap
        x0 = x + max(0, (avail - span) / 2)
        for k, lp in enumerate(loupes):
            lx = int(x0 + k * (ls + gap))
            li = Image.open(f"{out}/{lp['file']}").convert("RGBA")
            canvas.paste(li, (lx, py))
            c = COL.get(lp["name"], (220, 220, 220))
            d.rectangle([lx * SS, py * SS, (lx + ls) * SS, (py + ls) * SS], outline=c + (120,), width=SS)
            # leader to the body on the image
            bx, by = lp["screen"]
            d.line([(lx + ls / 2) * SS, py * SS, bx * SS, by * SS], fill=c + (70,), width=SS)
            ty = py + ls + 4
            text_o(d, (lx * SS, ty * SS), lp["name"].capitalize(), bold, c + (250,), o=SS)
            text_o(d, (lx * SS, (ty + 22) * SS), f"V {lp['v_mag']:+.1f}   {fmt_ang(lp['ang_diam_arcsec'])}   {fmt_mag(lp['magnification'])}", tiny, (225, 225, 225, 240), o=SS)
            text_o(d, (lx * SS, (ty + 38) * SS), f"phase {lp['phase_deg']:.0f} deg   {lp['dist_au']:.3g} AU", tiny, (190, 190, 190, 230), o=SS)
    # caption
    jd = j["epoch_utc_jd"]
    t = datetime.datetime(1970, 1, 1) + datetime.timedelta(days=jd - 2440587.5)
    cam = j["camera"]
    cap = (f"{shot}   {t:%Y-%m-%d %H:%M} UTC   |   {cam['dist_sun_au']:.3g} AU from the Sun, fov {cam['fov_y_deg']:.3g} deg, exposure {j['exposure']:.3g}"
           f"   |   image: true scale + true brightness; lines, markers and labels are annotation; loupes: true renders, long lens, exposure x r^2 (1 AU irradiance)")
    d.rectangle([0, (H + PANEL - 24) * SS, W * SS, (H + PANEL) * SS], fill=(0, 0, 0, 200))
    text_o(d, (10 * SS, (H + PANEL - 20) * SS), cap, tiny, (215, 215, 215, 255), o=0)
    ov = ov.resize((W, H + PANEL), Image.LANCZOS)
    Image.alpha_composite(canvas, ov).convert("RGB").save(f"{out}/{shot}_vis.png")


if __name__ == "__main__":
    for s in sys.argv[2:]:
        vis(sys.argv[1], s)
