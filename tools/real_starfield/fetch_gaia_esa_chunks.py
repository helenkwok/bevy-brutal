#!/usr/bin/env python3
"""Fetch Gaia DR3 x Bailer-Jones EDR3 distances (<=1 kpc, G<13, parallax>0.7 mas) from the ESA Gaia
archive in HEALPix level-3 source_id ranges (anonymous async jobs are limited to 60 s each).
Chunks that time out are split into 4 level-4 sub-ranges. Output: raw/esa_chunks/*.csv."""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from tap import run_async

BASE = 'https://gea.esac.esa.int/tap-server/tap'
OUT = Path('/workspace/data/astro/real_starfield/raw/esa_chunks')
Q = """SELECT g.source_id, g.ra, g.dec, g.parallax, g.parallax_error, g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.ruwe, d.r_med_geo, d.r_lo_geo, d.r_hi_geo, d.r_med_photogeo, d.r_lo_photogeo, d.r_hi_photogeo
FROM gaiadr3.gaia_source AS g JOIN external.gaiaedr3_distance AS d ON g.source_id = d.source_id
WHERE g.source_id >= {lo} AND g.source_id < {hi} AND g.parallax > 0.7 AND g.phot_g_mean_mag < 13 AND (d.r_med_photogeo <= 1000 OR (d.r_med_photogeo IS NULL AND d.r_med_geo <= 1000))"""
STEP = {3: 2**35 * 4**9, 4: 2**35 * 4**8, 5: 2**35 * 4**7}


def job(level, pix):
    lo, hi = pix * STEP[level], (pix + 1) * STEP[level]
    out = OUT / f'L{level}_{pix:05d}.csv'
    if out.exists():
        return 0
    for attempt in range(4):
        try:
            run_async(BASE, Q.format(lo=lo, hi=hi), str(out) + '.part', maxrec=5_000_000, poll=3)
            os.replace(str(out) + '.part', out)
            return 0
        except Exception as e:
            msg = str(e)
            if 'Maximum execution time' in msg and level < 5:
                return sum(job(level + 1, pix * 4 + k) for k in range(4))
            print(f'retry L{level} {pix}: {msg[:200]}', file=sys.stderr, flush=True)
            time.sleep(20 * (attempt + 1))
    print(f'FAILED L{level} {pix}', file=sys.stderr, flush=True)
    return 1


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(int(os.environ.get('JOBS', '6'))) as ex:
        fails = sum(ex.map(lambda p: job(3, p), range(768)))
    print('failed chunks:', fails)
