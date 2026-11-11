#!/usr/bin/env python3
"""Independent sanity overlay: project named Hipparcos stars straight from the RAW catalogue
(J1991.25 -> J2016.0 proper motion, unit vectors only) with the camera saved by the renderer and
draw labelled rings on a copy of the engine PNG. Also measures the engine image brightness at each
predicted position (hit = local peak within 3 px is far above the frame median)."""
import csv, json, math, sys
from pathlib import Path
sys.path.insert(0, '/workspace/data/astro/analysis-deps')
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RAW = Path('/workspace/data/astro/real_starfield/raw')
NAMED = {
    'Betelgeuse': 27989, 'Rigel': 24436, 'Bellatrix': 25336, 'Saiph': 27366, 'Alnitak': 26727, 'Alnilam': 26311,
    'Mintaka': 25930, 'Dubhe': 54061, 'Merak': 53910, 'Phecda': 58001, 'Megrez': 59774, 'Alioth': 62956,
    'Mizar': 65378, 'Alkaid': 67301, 'Acrux': 60718, 'Mimosa': 62434, 'Gacrux': 61084, 'Imai': 59747,
    'Rigil Kent': 71683, 'Hadar': 68702, 'Antares': 80763, 'Kaus Aus.': 90185, 'Nunki': 92855, 'Shaula': 85927,
    'Deneb': 102098, 'Sadr': 100453, 'Albireo': 95947, 'Vega': 91262, 'Altair': 97649, 'Aldebaran': 21421,
    'Capella': 24608, 'Elnath': 25428, 'Alcyone': 17702, 'Sirius': 32349, 'Canopus': 30438, 'Procyon': 37279,
}


def hip_dirs():
    want = set(NAMED.values())
    out = {}
    for r in csv.DictReader(open(RAW / 'hip2_newreduction.csv')):
        h = int(r['hip'])
        if h not in want:
            continue
        ra, de = math.radians(float(r['ra'])), math.radians(float(r['dec']))
        p = np.array([math.cos(de) * math.cos(ra), math.cos(de) * math.sin(ra), math.sin(de)])
        ea = np.array([-math.sin(ra), math.cos(ra), 0.0])
        ed = np.array([-math.sin(de) * math.cos(ra), -math.sin(de) * math.sin(ra), math.cos(de)])
        p = p + math.radians(1 / 3.6e6) * 24.75 * (float(r['pm_ra']) * ea + float(r['pm_de']) * ed)
        out[h] = p / np.linalg.norm(p)
    return out


def main(out_dir):
    out_dir = Path(out_dir)
    views = json.load(open(out_dir / 'views.json'))['views']
    dirs = hip_dirs()
    report = []
    for v in views:
        if v['dist_from_sol_pc'] > 1e-6:
            continue
        img = Image.open(out_dir / f"{v['name']}.png").convert('RGB')
        a = np.asarray(img).astype(float).mean(axis=2)
        w, h = v['size']
        f = h / (2 * math.tan(math.radians(v['fov_y_deg']) / 2))
        F, R, U = (np.array(v[k]) for k in ('forward', 'right', 'up'))
        d = ImageDraw.Draw(img)
        med = float(np.median(a))
        rows = []
        for name, hip in NAMED.items():
            p = dirs.get(hip)
            if p is None:
                continue
            z = p @ F
            if z <= 0:
                continue
            x, y = w / 2 + (p @ R) / z * f, h / 2 - (p @ U) / z * f
            if not (6 <= x < w - 6 and 6 <= y < h - 6):
                continue
            xi, yi = int(x), int(y)
            peak = float(a[max(0, yi - 3):yi + 4, max(0, xi - 3):xi + 4].max())
            hit = peak >= max(150.0, med + 100)
            rows.append(dict(name=name, hip=hip, x=round(x, 1), y=round(y, 1), peak=round(peak, 1), hit=hit))
            col = (90, 255, 120) if hit else (255, 70, 70)
            d.ellipse([x - 11, y - 11, x + 11, y + 11], outline=col, width=1)
            d.text((x + 13, y - 6), name, fill=col)
        d.text((8, 8), f"{v['name']}: rings = raw Hipparcos J2016 positions projected independently (green=engine star present)", fill=(200, 200, 200))
        img.save(out_dir / f"{v['name']}_annotated.png")
        report.append(dict(view=v['name'], stars=rows, hits=sum(r['hit'] for r in rows), total=len(rows)))
    (out_dir / 'annotation_check.json').write_text(json.dumps(report, indent=1))
    for r in report:
        print(r['view'], f"{r['hits']}/{r['total']}", [s['name'] for s in r['stars'] if not s['hit']])


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'out/quality/real_starfield')
