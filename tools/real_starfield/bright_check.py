#!/usr/bin/env python3
"""Naked-eye completeness: every Hipparcos main-catalogue star (ESA 1997, raw CSV) with V < VLIM
should have an engine star (engine_bright_g7.csv, dumped from the tiles by real_starfield) within
RADIUS arcsec (after J1991.25 -> J2016.0 proper-motion propagation). Reports misses with their Hipparcos parallax so gaps are explained, not hidden."""
import csv, json, math, sys
from pathlib import Path

RAW = Path("/workspace/data/astro/real_starfield/raw")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "out/quality/real_starfield")
VLIM, RADIUS = 6.0, 60.0


def vec(ra, de):
    a, d = math.radians(ra), math.radians(de)
    return (math.cos(d) * math.cos(a), math.cos(d) * math.sin(a), math.sin(d))


eng = {}
with open(OUT / "engine_bright_g7.csv") as f:
    for r in csv.DictReader(f):
        ra, de = float(r["ra_deg"]), float(r["dec_deg"])
        eng.setdefault((int(ra), int(de + 90)), []).append(vec(ra, de))
cosr = math.cos(math.radians(RADIUS / 3600))
n = found = 0
miss = []
with open(RAW / "hip1_main.csv") as f:
    for r in csv.DictReader(f):
        try:
            v, ra, de = float(r["vmag"]), float(r["ra"]), float(r["de"])
        except ValueError:
            continue
        if v >= VLIM:
            continue
        n += 1
        # hip1 positions are epoch J1991.25; engine positions are J2016.0. Propagate proper motion.
        try:
            dt = 2016.0 - 1991.25
            de += float(r["pmde"]) * dt / 3.6e6
            ra += float(r["pmra"]) * dt / 3.6e6 / max(1e-6, math.cos(math.radians(de)))
        except ValueError:
            pass
        u = vec(ra, de)
        ok = False
        span = 1 if abs(de) < 80 else 360
        for i in range(-span, span + 1):
            for j in (-1, 0, 1):
                for w in eng.get(((int(ra) + i) % 360, int(de + 90) + j), []):
                    if u[0] * w[0] + u[1] * w[1] + u[2] * w[2] >= cosr:
                        ok = True
                        break
                if ok:
                    break
            if ok:
                break
        if ok:
            found += 1
        else:
            miss.append({"hip": int(r["hip"]), "vmag": v, "plx_mas": r["plx"], "e_plx": r["e_plx"], "sptype": r["sptype"]})
res = {"vlim": VLIM, "radius_arcsec": RADIUS, "hipparcos_stars": n, "matched": found, "missing": len(miss),
       "missing_rows": sorted(miss, key=lambda m: m["vmag"])}
(OUT / "bright_check.json").write_text(json.dumps(res, indent=1))
print(f"Hipparcos V<{VLIM}: {found}/{n} have an engine star within {RADIUS}\"; missing {len(miss)}")
for m in res["missing_rows"][:15]:
    print("  miss", m)
