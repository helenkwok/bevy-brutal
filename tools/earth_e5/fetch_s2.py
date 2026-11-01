# Sentinel-2 cloudless (EOX) tiles for the Alps graft. Layer via S2_LAYER env.
import math, os, sys, subprocess, concurrent.futures as cf
LAYER = os.environ.get('S2_LAYER', 's2cloudless-2020_3857')
def tile(lat, lon, z):
    n = 2 ** z
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
jobs = []
for z, (la0, la1, lo0, lo1) in [(11, (44.50, 47.70, 5.20, 10.00)), (13, (45.68, 46.52, 7.00, 8.20))]:
    x0, y0 = tile(la1, lo0, z); x1, y1 = tile(la0, lo1, z)
    for y in range(int(y0), int(y1) + 1):
        for x in range(int(x0), int(x1) + 1):
            p = f'{LAYER}/{z}/{y}_{x}.jpg'
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                jobs.append((z, y, x, p))
print(len(jobs), 'tiles to fetch', flush=True)
def get(j):
    z, y, x, p = j
    os.makedirs(os.path.dirname(p), exist_ok=True)
    url = f'https://tiles.maps.eox.at/wmts/1.0.0/{LAYER}/default/g/{z}/{y}/{x}.jpg'
    r = subprocess.run(['curl', '-s', '-f', '--retry', '3', '-A', 'bevy-brutal-research/0.1', '-o', p, url])
    return r.returncode
with cf.ThreadPoolExecutor(4) as ex:
    bad = sum(1 for rc in ex.map(get, jobs) if rc)
print('failed', bad); print('S2_DONE')
