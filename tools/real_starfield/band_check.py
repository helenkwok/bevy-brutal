#!/usr/bin/env python3
"""Independent Milky Way band reference from the raw Hipparcos main catalogue (ESA 1997, Vmag),
computed WITHOUT the engine: surface density vs |galactic latitude| for V<7 (Hipparcos is complete
to V~7.3), with no distance cut and with the RS1 1 kpc cut (1/plx, plx > 1 mas). Compare with
`band_density_from_sol` in out/quality/real_starfield/views.json (engine stars, apparent G).

usage: band_check.py [views.json] -> writes band_check.json next to views.json
"""
import csv, json, math, sys
from pathlib import Path

RAW = Path("/workspace/data/astro/real_starfield/raw")
R_GAL = [  # ICRS -> Galactic (Hipparcos vol.1 / Gaia DR3 doc)
    [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
    [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
    [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669],
]


def abs_b(ra, de):
    a, d = math.radians(ra), math.radians(de)
    u = (math.cos(d) * math.cos(a), math.cos(d) * math.sin(a), math.sin(d))
    z = sum(R_GAL[2][i] * u[i] for i in range(3))
    return abs(math.degrees(math.asin(max(-1.0, min(1.0, z)))))


def ratio(bins):
    sb = [math.sin(math.radians(10 * i)) for i in range(10)]
    dens = [bins[i] / (sb[i + 1] - sb[i]) for i in range(9)]
    hi = sum(bins[6:]) / (1 - sb[6])
    return dens[0] / hi, dens


def main():
    views = Path(sys.argv[1] if len(sys.argv) > 1 else "out/quality/real_starfield/views.json")
    cuts = {"V<7_all_distances": [0] * 9, "V<7_within_1kpc_plx_gt_1mas": [0] * 9}
    rows = 0
    with open(RAW / "hip1_main.csv") as f:
        for r in csv.DictReader(f):
            try:
                v, ra, de = float(r["vmag"]), float(r["ra"]), float(r["de"])
            except ValueError:
                continue
            rows += 1
            if v >= 7.0:
                continue
            k = min(8, int(abs_b(ra, de) // 10))
            cuts["V<7_all_distances"][k] += 1
            try:
                if float(r["plx"]) > 1.0:
                    cuts["V<7_within_1kpc_plx_gt_1mas"][k] += 1
            except ValueError:
                pass
    out = {"source": "ESA Hipparcos main catalogue I/239 (public.hipparcos, gea.esac.esa.int)", "rows_read": rows, "reference": {}}
    for name, bins in cuts.items():
        r, _ = ratio(bins)
        out["reference"][name] = {"counts_per_10deg_absb": bins, "ratio_absb_lt10_over_gt60": round(r, 3)}
        print(f"Hipparcos {name}: ratio |b|<10 / |b|>60 = {r:.2f}; counts {bins}")
    if views.exists():
        eng = json.loads(views.read_text()).get("band_density_from_sol")
        out["engine"] = eng
        for e in eng or []:
            print(f"engine G<{e['mag_lt']}: ratio = {e['ratio_absb_lt10_over_gt60']}; counts {e['counts_per_10deg_absb']}")
    dst = views.parent / "band_check.json"
    dst.write_text(json.dumps(out, indent=1))
    print("wrote", dst)


if __name__ == "__main__":
    main()
