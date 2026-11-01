#!/usr/bin/env python3
"""Daily cloud-coverage map from NASA GIBS VIIRS true colour (fetch_earth_data.sh step 'viirs').

earth_clouds.py <data dir> [date]: assembles <data>/viirs_<date>/t_<row>_<col>.jpg (8 x 4 tiles,
45 deg each, 2048 px) into a 16384 x 8192 equirectangular coverage map clouds_viirs.png (8-bit).

Corrected reflectance is display-stretched (GIBS / Gumley et al. piecewise-linear enhancement,
reflectance x 255 -> 0..255 through (0,0) (30,110) (60,160) (120,210) (190,240) (255,255)); it is
inverted to reflectance. Cloudiness = blue-band reflectance above the clear-sky surface (Blue
Marble NG July, linearised, the same reference the renderer uses for the ground), normalised by
the headroom to a thick cloud's ~0.75, times a whiteness term (min/max of RGB) so bright but
coloured surfaces the clear reference does not explain (desert haze, smoke, turbid shelf water)
do not read as cloud. Snow / sea ice are bright in the clear reference too, so they cancel.

Sun glint (VIIRS has no glint mask in the true-colour product) shows as smooth grey stripes a
few hundred km wide over the tropical oceans; it, residual haze and clear-reference mismatch are
removed as a low local floor: the ~30 km minimum of the coverage, blurred, is subtracted where it
is low (< ~0.3) and kept where it is high (overcast decks have a high floor too). No-data (polar
night, swath gaps) is filled with the Blue Marble cloud map (clouds.tif), feathered over ~50 km.

Stored in the renderer's cloud-map convention (the Blue Marble one): nadir opacity
a = ((s - 0.04) / 0.92)^1.6, so s = 0.04 + 0.92 a^(1/1.6).
"""
import sys, os
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

data = sys.argv[1] if len(sys.argv) > 1 else "/workspace/data/earth"
date = sys.argv[2] if len(sys.argv) > 2 else "20260715"
d = os.path.join(data, f"viirs_{date}")
W, H, T = 16384, 8192, 2048
img = np.zeros((H, W, 3), np.uint8)
for j in range(4):
    for i in range(8):
        t = np.asarray(Image.open(os.path.join(d, f"t_{j}_{i}.jpg")).convert("RGB"))
        assert t.shape == (T, T, 3), t.shape
        img[j * T:(j + 1) * T, i * T:(i + 1) * T] = t
xs = np.array([0, 110, 160, 210, 240, 255], np.float32)
ys = np.array([0, 30, 60, 120, 190, 255], np.float32) / 255.0
lut = np.interp(np.arange(256, dtype=np.float32), xs, ys).astype(np.float32)
def srgb_lin(c):
    c = c.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)
bm = Image.open(os.path.join(data, "bmng_july.jpg")).convert("RGB").resize((W, H), Image.BILINEAR)
bm = np.asarray(bm)
from PIL import ImageFilter
cov = np.zeros((H, W), np.float32)
nod = np.zeros((H, W), bool)
for y0 in range(0, H, 1024):  # strips (memory)
    s = slice(y0, y0 + 1024)
    v = img[s]
    refl = lut[v]                               # (h, W, 3) reflectance
    nodata = v.max(axis=2) < 6
    clear_b = srgb_lin(bm[s][..., 2])
    rb = refl[..., 2]
    c = np.clip((rb - clear_b - 0.03) / np.maximum(0.75 - clear_b, 0.1), 0.0, 1.0)
    white = refl.min(axis=2) / np.maximum(refl.max(axis=2), 1e-3)
    c *= np.clip((white - 0.55) / 0.3, 0.0, 1.0)
    c[nodata] = 0.0
    cov[s] = c
    nod[s] = nodata
del img, bm
def lowres(a, f):  # f x f block reduction
    return a.reshape(H // f, f, W // f, f)
def up4(a):
    im = Image.fromarray(a.astype(np.float32), "F").resize((W, H), Image.BILINEAR)
    return np.asarray(im, np.float32)
# local floor: min over 4x4 blocks then a 7x7 min at quarter res (~29 km), blurred ~20 km
q = np.where(nod, 1.0, cov)
fl = lowres(q, 4).min(axis=(1, 3))
fl8 = Image.fromarray(np.clip(fl * 255.0, 0, 255).astype(np.uint8), "L").filter(ImageFilter.MinFilter(7))
fl8 = fl8.filter(ImageFilter.GaussianBlur(4))
floor = up4(np.asarray(fl8, np.float32) / 255.0)
k = 1.0 - np.clip((floor - 0.25) / 0.25, 0.0, 1.0)
k = k * k * (3.0 - 2.0 * k)
f = floor * k
cov = np.clip((cov - f) / np.maximum(1.0 - f, 1e-3), 0.0, 1.0)
del floor, k, f, q
# fill no-data with the Blue Marble cloud map, feathered
bmc = np.asarray(Image.open(os.path.join(data, "clouds.tif")).convert("L").resize((W, H), Image.BILINEAR), np.float32) / 255.0
bmc = np.clip((bmc - 0.04) / 0.92, 0.0, 1.0) ** 1.6
m = lowres(nod.astype(np.float32), 4).max(axis=(1, 3))
m8 = Image.fromarray((m * 255).astype(np.uint8), "L").filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(5))
m = np.maximum(up4(np.asarray(m8, np.float32) / 255.0), nod)
cov = cov * (1.0 - m) + bmc * m
del bmc, m
out = (np.clip(0.04 + 0.92 * cov ** (1.0 / 1.6), 0, 1) * 255.0 + 0.5).astype(np.uint8)
Image.fromarray(out, "L").save(os.path.join(data, "clouds_viirs.png"), optimize=False)
lat = (90 - (np.arange(H) + 0.5) * 180 / H) * np.pi / 180
w = np.cos(lat)[:, None]
print(f"clouds_viirs.png {W}x{H}: area-weighted mean coverage {(cov * w).sum() / (w.sum() * W):.3f}, "
      f"fraction > 0.5: {((cov > 0.5) * w).sum() / (w.sum() * W):.3f}")

# ---- cloud-top height (VIIRS_NOAA20_Cloud_Top_Height_Day, same pass): <viirs>/cth/t_<j>_<i>.png
# (8 x 4 tiles, 1024 px = ~4.9 km) + the GIBS colormap (colormap.xml, MODIS_VIIRS_Cloud_Top_Height:
# 50 m bins, one RGB each, transparent = no cloud / fill). Decoded to metres, holes filled by
# push-pull (mean of the valid neighbourhood at the coarsest level that has one), stored as
# cth_viirs.png 8192 x 4096, 8-bit, value = km x 20 (0 .. 12.75 km).
import re
cd = os.path.join(d, "cth")
if os.path.isdir(cd):
    ents = re.findall(r'<ColorMapEntry rgb="(\d+),(\d+),(\d+)" transparent="false" sourceValue="\[(-?[\d.]+),', open(os.path.join(cd, "colormap.xml")).read())
    key = np.full(1 << 24, -1.0, np.float32)
    for r, g, b, v in ents:
        key[(int(r) << 16) | (int(g) << 8) | int(b)] = float(v) + 25.0
    CW, CH, CT = 8192, 4096, 1024
    hm = np.full((CH, CW), -1.0, np.float32)
    miss = 0
    for j in range(4):
        for i in range(8):
            t = np.asarray(Image.open(os.path.join(cd, f"t_{j}_{i}.png")).convert("RGBA"))
            assert t.shape == (CT, CT, 4), t.shape
            k = (t[..., 0].astype(np.int32) << 16) | (t[..., 1].astype(np.int32) << 8) | t[..., 2]
            v = key[k]
            miss += int(((v < 0) & (t[..., 3] > 0)).sum())
            v[t[..., 3] == 0] = -1.0
            hm[j * CT:(j + 1) * CT, i * CT:(i + 1) * CT] = v
    valid = hm >= 0
    # push-pull: pyramid of (sum, count) of valid pixels, holes take the finest non-empty level
    levels = [(np.where(valid, hm, 0.0).astype(np.float64), valid.astype(np.float64))]
    while levels[-1][0].shape[0] > 4:
        s0, c0 = levels[-1]
        hh, ww = s0.shape
        levels.append((s0.reshape(hh // 2, 2, ww // 2, 2).sum(axis=(1, 3)), c0.reshape(hh // 2, 2, ww // 2, 2).sum(axis=(1, 3))))
    s0, c0 = levels[-1]
    fill = np.where(c0 > 0, s0 / np.maximum(c0, 1e-9), np.float64(2000.0))
    for s1, c1 in reversed(levels[:-1]):
        fill = np.repeat(np.repeat(fill, 2, axis=0), 2, axis=1)
        fill = np.where(c1 > 0, s1 / np.maximum(c1, 1e-9), fill)
        if s1.shape[0] < CH:   # light smoothing of the coarse fill between levels
            q = np.pad(fill, 1, mode="edge")
            fill = sum(q[a:a + fill.shape[0], b:b + fill.shape[1]] for a in range(3) for b in range(3)) / 9.0
    hkm = fill.astype(np.float32) / 1000.0
    Image.fromarray(np.clip(hkm * 20.0 + 0.5, 0, 255).astype(np.uint8), "L").save(os.path.join(data, "cth_viirs.png"))
    latc = (90 - (np.arange(CH) + 0.5) * 180 / CH) * np.pi / 180
    wc = np.cos(latc)[:, None] * valid
    print(f"cth_viirs.png {CW}x{CH}: valid {valid.mean():.3f}, unmatched colours {miss}, mean top {(hm * wc).sum() / wc.sum() / 1000:.2f} km, "
          f">8 km {((hm > 8000) * wc).sum() / wc.sum():.3f}")
