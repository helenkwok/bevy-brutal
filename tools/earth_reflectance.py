#!/usr/bin/env python3
"""Earth reflectance probe (read-only): radiance factor R = pi L / E_sun before the tone curve.

Runs the `earth` example on the DSCOVR/EPIC geometry (E2_vs_epic, default look, EPIC_GRADE off) with
HDR_OUT=1, plus two debug passes (DEBUG=2: water fraction / ice mask, DEBUG=4: cloud-map value) at
the same camera, and classifies each pixel:

  clear ocean : water > 0.98, ice < 0.02, cloud map < 0.06   (map 0.04 = no cloud, shader: (m-0.04)/0.92)
  thick cloud : cloud map > 0.85                              (nadir opacity k.c >~ 0.8, tau >~ 2.5 before meso)
  clear land  : water < 0.02, ice < 0.02, cloud map < 0.06
regions (geometry from the camera, ray/sphere):
  disc centre : mu > 0.85 and mu0 > 0.80   (view / sun zenith cosines)
  limb        : 0.08 < mu < 0.25 and mu0 > 0.2

Units: the planet renderer's radiance is in solar irradiance (1 AU) per sr x exposure, so
R = pi * hdr / exposure, with E = 1 (the solar compositor applies the 1/r^2 factor, not this probe).
BRF = R / mu0. Disc mean R over the projected disc = the EPIC "scattering function" P(phase), which
at zero phase is the geometric albedo p.

usage: tools/earth_reflectance.py [--bin target/release/examples/earth] [--out out/quality/steps]
       [--work /tmp/earth_refl] [--skip-render] [--size 1080]
"""
import argparse, json, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

R_KM = 6371.0
ALT_KM = 1_475_183.0
FOV = 0.6239
EXPO = 2.4
SUB = (13.389, 22.513)        # sub-spacecraft point (E2_vs_epic)
SUNS = {"epic": (21.537, 14.219), "zero_phase": SUB}
CERES = {"clear_ocean": (0.08, 0.10), "thick_cloud": (0.5, 0.8)}


def ll(lat, lon):
    a, b = np.radians(lat), np.radians(lon)
    return np.array([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)])


def render(bin_, work, tag, size, env):
    d = os.path.join(work, tag)
    os.makedirs(d, exist_ok=True)
    e = dict(os.environ, STARS="0", HDR_OUT="1", PLANET_VALIDATE="0", **env)
    e.pop("EPIC_GRADE", None)
    subprocess.run([bin_, "--shot", "E2_vs_epic", "--pack", "/workspace/data/earth/earth.pln", "--size",
                    f"{size}x{size}", "--ss", "1", "--out", d], env=e, check=True, capture_output=True)
    return d


def load(d, size):
    return np.fromfile(os.path.join(d, "E2_vs_epic.hdr.bin"), dtype=np.float32).reshape(size, size, 4)


def geometry(size, sun_ll):
    d = R_KM + ALT_KM
    sub, sun = ll(*SUB), ll(*sun_ll)
    C = sub * d
    fwd = -sub
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    ty = np.tan(np.radians(FOV) / 2)
    x = (np.arange(size) + 0.5) / size * 2 - 1
    X, Y = np.meshgrid(x, x)
    rd = fwd + right * (X * ty)[..., None] - up * (Y * ty)[..., None]
    rd /= np.linalg.norm(rd, axis=-1, keepdims=True)
    b = (rd * C).sum(-1)
    disc = b * b - ((C * C).sum() - R_KM * R_KM)
    hit = disc > 0
    t = -b - np.sqrt(np.maximum(disc, 0))
    n = (C + rd * t[..., None]) / R_KM
    mu = np.where(hit, (n * -rd).sum(-1), 0.0)
    mu0 = np.where(hit, (n * sun).sum(-1), 0.0)
    phase = float(np.degrees(np.arccos(np.clip(np.dot(sub, sun), -1, 1))))
    r_px = np.tan(np.arcsin(R_KM / d)) / ty * size / 2
    return hit, mu, mu0, phase, r_px


def lum(a):
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def stats(R, RY, mu0, m):
    if m.sum() == 0:
        return {"n": 0}
    v = RY[m]
    return {"n": int(m.sum()), "median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "median_rgb": [float(np.median(R[..., k][m])) for k in range(3)],
            "median_brf": float(np.median((RY / np.maximum(mu0, 1e-3))[m]))}


def analyse(main, d2, d4, size, sun_ll):
    hit, mu, mu0, phase, r_px = geometry(size, sun_ll)
    R = np.pi * main[..., :3] / EXPO
    RY = lum(R)
    water, ice, cov = d2[..., 0], d2[..., 1], d4[..., 0]
    cls = {"clear_ocean": (water > 0.98) & (ice < 0.02) & (cov < 0.06),
           "thick_cloud": cov > 0.85,
           "clear_land": (water < 0.02) & (ice < 0.02) & (cov < 0.06)}
    centre = hit & (mu > 0.85) & (mu0 > 0.80)
    limb = hit & (mu > 0.08) & (mu < 0.25) & (mu0 > 0.2)
    out = {"phase_deg": phase, "disc_r_px": r_px, "classes": {}}
    for k, c in cls.items():
        out["classes"][k] = {"centre": stats(R, RY, mu0, centre & c), "limb": stats(R, RY, mu0, limb & c)}
    out["classes"]["all"] = {"centre": stats(R, RY, mu0, centre), "limb": stats(R, RY, mu0, limb)}
    oc, cl = out["classes"]["clear_ocean"]["centre"], out["classes"]["thick_cloud"]["centre"]
    out["contrast_cloud_over_ocean"] = cl.get("median", np.nan) / oc.get("median", np.nan)
    # disc mean over the projected disc + atmosphere rim (the flux a distant observer gets), and the
    # EPIC-paper form (pixels with SZA < 76 deg only)
    yy, xx = np.mgrid[0:size, 0:size] + 0.5
    rim = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 < (r_px * 1.03) ** 2
    area = np.pi * r_px * r_px
    out["disc_mean_R"] = {"Y": float(RY[rim].sum() / area), "rgb": [float(R[..., k][rim].sum() / area) for k in range(3)]}
    s76 = hit & (mu0 > np.cos(np.radians(76)))
    out["epic_P_sza76"] = {"Y": float(RY[s76].mean()), "rgb": [float(R[..., k][s76].mean()) for k in range(3)]}
    # cloud fraction proxies
    out["cloud_frac_map_gt_0.5"] = float((cov[hit] > 0.5).mean())
    out["frac_R_gt_0.3_mu0_gt_0.3"] = float((RY[hit & (mu0 > 0.3)] > 0.3).mean())
    return out, dict(R=R, RY=RY, hit=hit, mu=mu, mu0=mu0, centre=centre, limb=limb, **cls)


def false_colour(v, lo=0.02, hi=1.0):
    t = np.clip((np.log10(np.maximum(v, 1e-6)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)), 0, 1)
    stops = np.array([[0, 0, 0], [40, 20, 120], [20, 110, 200], [30, 190, 140], [230, 220, 60], [255, 255, 255]], float)
    p = t * (len(stops) - 1)
    i = np.minimum(p.astype(int), len(stops) - 2)
    f = (p - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# disc-mean references (luminance-ish): EPIC L1B scattering function P (~geometric albedo at 4.5-11.5 deg),
# 23 Aug 2016 daily mean, 551 nm (Song et al. 2018, Remote Sens. 10, 1594, table 3); Mallama V-band p
EPIC_P551 = 0.217
MALLAMA_P = 0.434


def sheet(path, png, a, res, cap):
    S = 600
    f = ImageFont.truetype(FONT, 15)
    fs = ImageFont.truetype(FONT, 13)
    fb = ImageFont.truetype(BOLD, 17)
    tiles = []
    img = Image.open(png).convert("RGB").resize((S, S), Image.LANCZOS)
    tiles.append((img, "planet renderer, tonemapped (default look, EPIC_GRADE off)"))
    fc = false_colour(a["RY"])
    tiles.append((Image.fromarray(fc).resize((S, S), Image.LANCZOS), "radiance factor \u03c0L/E before tonemap (log 0.02 \u2192 1)"))
    base = np.asarray(img.resize((a["RY"].shape[1], a["RY"].shape[0]))).astype(float) * 0.35
    ov = base.copy()
    ov[a["clear_ocean"] & a["hit"]] = [40, 120, 255]
    ov[a["thick_cloud"] & a["hit"]] = [255, 255, 255]
    ov[a["clear_land"] & a["hit"]] = [200, 140, 60]
    c = a["centre"]
    edge = (c ^ np.roll(c, 2, 0)) | (c ^ np.roll(c, 2, 1))
    ov[edge] = [255, 60, 60]
    ov[a["limb"] & ~(a["clear_ocean"] | a["thick_cloud"])] = [255, 160, 0]
    tiles.append((Image.fromarray(ov.astype(np.uint8)).resize((S, S), Image.LANCZOS),
                  "classes: clear ocean / thick cloud / clear land; centre (red), limb (orange)"))
    bp = Image.new("RGB", (S, S), (16, 16, 20))
    d = ImageDraw.Draw(bp)
    C = res["classes"]
    rows = [("clear ocean \u00b7 disc centre", C["clear_ocean"]["centre"], CERES["clear_ocean"], [(0.061, "Exolune")]),
            ("thick cloud \u00b7 disc centre", C["thick_cloud"]["centre"], CERES["thick_cloud"], [(0.645, "Exolune")]),
            ("clear land \u00b7 disc centre", C["clear_land"]["centre"], None, []),
            ("ocean \u00b7 limb (\u03bc 0.08\u20130.25)", C["clear_ocean"]["limb"], None, []),
            ("cloud \u00b7 limb", C["thick_cloud"]["limb"], None, []),
            ("disc mean = P(phase) \u2248 p", {"median": res["disc_mean_R"]["Y"]}, None,
             [(EPIC_P551, "EPIC 551"), (MALLAMA_P, "Mallama p")])]
    X0, X1 = 24, S - 24
    sx = lambda v: X0 + (X1 - X0) * min(v, 1.0)
    d.text((X0, 14), cap, fill=(235, 235, 235), font=fb)
    d.text((X0, 38), "median (yellow), p10\u2013p90 (blue), CERES band (green), references (ticks)", fill=(160, 160, 170), font=fs)
    for i, (name, st, band, refs) in enumerate(rows):
        y = 74 + i * 84
        d.text((X0, y), name, fill=(225, 225, 225), font=f)
        d.rectangle([X0, y + 22, X1, y + 46], outline=(70, 70, 82))
        if band:
            d.rectangle([sx(band[0]), y + 22, sx(band[1]), y + 46], fill=(28, 92, 44))
        if "p10" in st:
            d.rectangle([sx(st["p10"]), y + 32, sx(st["p90"]), y + 36], fill=(120, 140, 210))
        for v, lab in refs:
            d.line([sx(v), y + 18, sx(v), y + 50], fill=(230, 90, 90), width=2)
            d.text((sx(v) + 4, y + 50), lab, fill=(230, 120, 120), font=fs)
        if "median" in st:
            m = st["median"]
            d.rectangle([sx(m) - 2, y + 19, sx(m) + 2, y + 49], fill=(255, 215, 70))
            d.text((X1 - 52, y), f"{m:.3f}", fill=(255, 215, 70), font=f)
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        d.text((sx(v) - 12, S - 24), f"{v:.2f}", fill=(140, 140, 150), font=fs)
    tiles.append((bp, "class medians vs CERES and disc-mean references"))
    W = Image.new("RGB", (2 * S, 2 * S + 56), (0, 0, 0))
    dd = ImageDraw.Draw(W)
    for k, (t, cpt) in enumerate(tiles):
        x, y = (k % 2) * S, (k // 2) * (S + 28)
        W.paste(t, (x, y + 28))
        dd.text((x + 10, y + 6), cpt, fill=(230, 230, 230), font=f)
    W.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="target/release/examples/earth")
    ap.add_argument("--out", default="out/quality/steps")
    ap.add_argument("--work", default="/tmp/earth_refl")
    ap.add_argument("--size", type=int, default=1080)
    ap.add_argument("--skip-render", action="store_true")
    a = ap.parse_args()
    results = {}
    for tag, sun_ll in SUNS.items():
        env = {"SUN_LAT": str(sun_ll[0]), "SUN_LON": str(sun_ll[1])}
        dirs = {}
        for k, extra in (("main", {}), ("d2", {"DEBUG": "2"}), ("d4", {"DEBUG": "4"})):
            dirs[k] = os.path.join(a.work, f"{tag}_{k}")
            if not a.skip_render:
                render(a.bin, a.work, f"{tag}_{k}", a.size, {**env, **extra})
        res, arr = analyse(load(dirs["main"], a.size), load(dirs["d2"], a.size), load(dirs["d4"], a.size), a.size, sun_ll)
        results[tag] = res
        sheet(os.path.join(a.out, f"ER_{tag}_sheet.png"), os.path.join(dirs["main"], "E2_vs_epic.png"), arr, res,
              f"sun: {tag.replace('_', ' ')} \u00b7 phase {res['phase_deg']:.1f}\u00b0 \u00b7 \u03c0L/E (luminance)")
    with open(os.path.join(a.out, "ER_probe.json"), "w") as f:
        json.dump(results, f, indent=1)
    json.dump(results, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
