#!/usr/bin/env python3
"""Fit planet::drape class albedos to Sentinel-2 L2A surface reflectance.

Data: Copernicus Sentinel-2 L2A COGs from the public AWS Open Data bucket (Element 84 Earth
Search STAC, no account). Free incl. commercial use; credit
"Contains modified Copernicus Sentinel data [year]". No Google sources.

Model (same as Drape::composite): each 10 m pixel = painter's-order blend of OSM class
coverage (rasterised 4x4 supersampled) over an unknown "unmapped" base:
    rho = sum_k v_k c_k + (1 - sum v) c_base,   v_k = f_k * prod_{j>k} (1 - f_j)
solved per band by non-negative least squares on clear land/water pixels (SCL 4,5,6,11 dropped
cloud/shadow). Two fits: `full` (buildings separate: merged/objects tiers) and `proxy`
(buildings folded into the landuse beneath: proxy tier, avoids double-counting roofs that the
10 m residential pixels already contain). Also per-building roof stats for the objects tier.

usage: tools/s2_fit.py [--item S2B_54HTG_20241224_0_L2A] [--osm DIR] [--out DIR]
"""
import argparse, json, math, os, sys, urllib.request
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize
from pyproj import Transformer
from scipy.optimize import nnls  # noqa

CLASSES = ["Farm", "Residential", "Industrial", "Grass", "Trees", "Bare", "Water", "Building",
           "Path", "MinorRoad", "MajorRoad"]
MAJOR = {"motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link", "trunk_link",
         "primary_link", "secondary_link", "tertiary_link", "busway", "bus_guideway"}
PATH = {"footway", "cycleway", "bridleway", "path", "steps", "corridor", "track", "pedestrian"}


def classify(key, kind):  # mirror of DrapeClass::from_osm
    if key == "highway":
        return "MajorRoad" if kind in MAJOR else "Path" if kind in PATH else "MinorRoad"
    if key == "building":
        return "Building"
    if key in ("natural", "waterway") and kind in ("water", "riverbank", "dock", "river", "canal", "stream"):
        return "Water"
    if key == "landuse":
        if kind in ("reservoir", "basin"): return "Water"
        if kind in ("residential", "garages"): return "Residential"
        if kind in ("industrial", "commercial", "retail", "railway", "construction", "brownfield", "port",
                    "military", "quarry", "landfill"): return "Industrial"
        if kind in ("farmland", "vineyard", "orchard", "meadow", "farmyard", "allotments", "plant_nursery",
                    "greenhouse_horticulture"): return "Farm"
        if kind == "forest": return "Trees"
        if kind in ("grass", "recreation_ground", "cemetery", "village_green", "flowerbed", "greenfield"): return "Grass"
        return None
    if key == "leisure":
        return "Trees" if kind == "nature_reserve" else "Water" if kind == "marina" else "Grass"
    if key == "natural":
        if kind in ("wood", "scrub", "heath"): return "Trees"
        if kind in ("grassland", "fell", "wetland"): return "Grass"
        if kind in ("sand", "beach", "bare_rock", "scree", "shingle", "mud"): return "Bare"
    return None


def width(cls, kind):  # mirror of DrapeClass::default_width_m
    if cls == "MajorRoad":
        return 22.0 if kind in ("motorway", "trunk") else 16.0 if kind == "primary" else 8.0 if kind.endswith("_link") else 12.0
    if cls == "MinorRoad": return 5.0 if kind == "service" else 8.0
    if cls == "Path": return 2.5
    if cls == "Water": return 15.0 if kind in ("river", "canal") else 4.0
    return 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", default="S2B_54HTG_20241224_0_L2A")
    ap.add_argument("--osm", default="/workspace/data/geo/osm/derived/adelaide_wide")
    ap.add_argument("--out", default="/workspace/data/geo/sentinel2/adelaide")
    ap.add_argument("--bbox", default="138.43,-35.06,138.77,-34.74")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bbox = [float(v) for v in a.bbox.split(",")]
    item = json.load(urllib.request.urlopen(
        f"https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/{a.item}"))
    year = item["properties"]["datetime"][:4]

    # ---- windowed COG reads (cached as a small GeoTIFF clip) ----
    clip = os.path.join(a.out, f"{a.item}_clip.tif")
    if not os.path.exists(clip):
        bands = []
        for name in ("red", "green", "blue", "scl"):
            href = item["assets"][name]["href"]
            with rasterio.open(href) as src:
                if name == "red":
                    crs = src.crs
                    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                    xs, ys = tr.transform([bbox[0], bbox[2], bbox[0], bbox[2]], [bbox[1], bbox[1], bbox[3], bbox[3]])
                    win = from_bounds(min(xs), min(ys), max(xs), max(ys), src.transform).round_offsets().round_lengths()
                    transform = src.window_transform(win)
                    shape = (int(win.height), int(win.width))
                    bounds = rasterio.windows.bounds(win, src.transform)
                    data = src.read(1, window=win)
                else:
                    data = src.read(1, window=from_bounds(*bounds, src.transform), out_shape=shape,
                                    resampling=rasterio.enums.Resampling.nearest)
                bands.append(data.astype(np.uint16))
                print("read", name, data.shape, file=sys.stderr)
        with rasterio.open(clip, "w", driver="GTiff", width=shape[1], height=shape[0], count=4, dtype="uint16",
                           crs=crs, transform=transform, compress="deflate") as dst:
            for i, b in enumerate(bands):
                dst.write(b, i + 1)
    with rasterio.open(clip) as src:
        raw = src.read()
        crs, transform = src.crs, src.transform
    h, w = raw.shape[1:]
    # Earth Search re-scales baseline >= 04.00 products (earthsearch:boa_offset_applied); else -0.1
    off = 0.0 if item["properties"].get("earthsearch:boa_offset_applied") else -0.1
    rgb = np.clip(raw[:3].astype(np.float32) * 1e-4 + off, 0, 1)
    scl = raw[3]
    valid = np.isin(scl, (4, 5, 6)) & (raw[0] > 0)  # vegetation, not-vegetated, water

    # ---- OSM coverage at 10 m (4x4 supersampled) ----
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    shapes = {c: [] for c in CLASSES}
    bldg = []
    from shapely.geometry import shape as shp
    from shapely.ops import transform as stransform
    proj = lambda x, y, z=None: tr.transform(x, y)
    for layer in ("landuse", "water", "buildings", "roads"):
        gj = json.load(open(os.path.join(a.osm, f"{layer}.geojson")))
        for f in gj["features"]:
            p = f["properties"]
            cls = classify(p.get("key", ""), p.get("kind", ""))
            if cls is None:
                continue
            g = stransform(proj, shp(f["geometry"]))
            if g.geom_type in ("LineString", "MultiLineString"):
                g = g.buffer(width(cls, p.get("kind", "")) / 2)
            if g.is_empty:
                continue
            shapes[cls].append(g)
            if cls == "Building":
                bldg.append(g)
        print("osm", layer, file=sys.stderr)
    S = 4
    from affine import Affine
    tsub = transform * Affine.scale(1 / S)
    frac = np.zeros((len(CLASSES), h, w), np.float32)
    for i, c in enumerate(CLASSES):
        if shapes[c]:
            m = rasterize(((g, 1) for g in shapes[c]), out_shape=(h * S, w * S), transform=tsub, dtype="uint8")
            frac[i] = m.reshape(h, S, w, S).mean(axis=(1, 3))
        print("raster", c, len(shapes[c]), float(frac[i].mean()), file=sys.stderr)

    def visible(fr):
        v = np.empty_like(fr)
        above = np.ones(fr.shape[1:], np.float32)
        for k in range(fr.shape[0] - 1, -1, -1):
            v[k] = fr[k] * above
            above *= 1 - fr[k]
        return v

    rng = np.random.default_rng(1)

    def fit(fr, names):
        v = visible(fr)
        base = 1 - v.sum(0)
        sea = (scl == 6).astype(np.float32)
        # unmapped remainder split into land and open sea (OSM has no sea polygons)
        X = np.concatenate([v, (base * (1 - sea))[None], (base * sea)[None]], 0).reshape(len(names) + 2, -1).T
        y = rgb.reshape(3, -1).T
        idx = np.flatnonzero(valid.ravel())
        idx = rng.choice(idx, min(len(idx), 1_500_000), replace=False)
        X, y = X[idx], y[idx]
        out = {}
        resid = []
        cols = []
        for b in range(3):
            c, _ = nnls(X.astype(np.float64), y[:, b].astype(np.float64))
            cols.append(c)
            resid.append(float(np.sqrt(np.mean((X @ c - y[:, b]) ** 2))))
        cols = np.array(cols).T
        weight = X.sum(0)
        for i, n in enumerate(names + ["Unmapped", "Sea"]):
            out[n] = {"albedo": [round(float(x), 4) for x in cols[i]], "pixel_weight": round(float(weight[i]))}
        return out, resid

    full, rfull = fit(frac, CLASSES)
    # proxy: buildings folded into whatever landuse lies beneath (drop the Building layer)
    keep = [c for c in CLASSES if c != "Building"]
    proxy, rprox = fit(frac[[CLASSES.index(c) for c in keep]], keep)

    # OSM water polygons include dry basins / buffered creeks: take Water from pixels that are
    # both fully OSM-water and SCL water (open water), not from the regression
    wi = CLASSES.index("Water")
    ow = (frac[wi] > 0.95) & (scl == 6)
    open_water = [round(float(x), 4) for x in np.median(rgb[:, ow], axis=1)]
    for f in (full, proxy):
        f["Water"]["albedo_regression"] = f["Water"]["albedo"]
        f["Water"]["albedo"] = open_water

    # ---- per-building roof reflectance (objects tier spread): fully covered pixels only ----
    bi = CLASSES.index("Building")
    pure = (frac[bi] > 0.95) & valid
    roof = rgb[:, pure].T
    lum = roof @ np.array([0.2126, 0.7152, 0.0722])
    qs = [10, 30, 50, 70, 90]
    pct = {f"p{q}": [round(float(x), 4) for x in np.percentile(roof, q, axis=0)] for q in qs}
    # roof "variants": terracotta tile (R/G > 1.2) + luminance quartiles of the rest (dark
    # Colorbond .. galvanised); `variant_weights` = pixel share, used to pick per-building variants
    tile = roof[:, 0] > 1.2 * np.maximum(roof[:, 1], 1e-4)
    rest, rl = roof[~tile], lum[~tile]
    e = np.percentile(rl, [0, 25, 50, 75, 100])
    groups = [roof[tile]] + [rest[(rl >= e[i]) & (rl <= e[i + 1])] for i in range(4)]
    variants = [[round(float(x), 4) for x in g.mean(0)] for g in groups]
    vweights = [round(len(g) / len(roof), 3) for g in groups]
    areas = np.array([g.area for g in bldg])

    res = {
        "source": f"Copernicus Sentinel-2 L2A {a.item} (Earth Search / AWS Open Data), bands B04/B03/B02 surface reflectance",
        "credit": f"Contains modified Copernicus Sentinel data {year}",
        "osm_credit": "© OpenStreetMap contributors (ODbL)",
        "bbox": bbox,
        "valid_pixels": int(valid.sum()),
        "model": "painter's-order NNLS per band over OSM class coverage (see tools/s2_fit.py)",
        "rmse_rgb_full": [round(x, 4) for x in rfull],
        "rmse_rgb_proxy": [round(x, 4) for x in rprox],
        "full": full,
        "proxy": proxy,
        "roof": {"pure_pixels": int(pure.sum()), "percentiles": pct, "variants": variants, "variant_weights": vweights},
        "buildings": {"count": len(areas), "median_area_m2": round(float(np.median(areas)), 1),
                      "p90_area_m2": round(float(np.percentile(areas, 90)), 1)},
    }
    json.dump(res, open(os.path.join(a.out, "drape_palette.json"), "w"), indent=1)
    # small preview for compares: S2 true colour (linear -> sRGB, x1 exposure)
    np.save(os.path.join(a.out, "rgb_preview.npy"), rgb.astype(np.float16))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
