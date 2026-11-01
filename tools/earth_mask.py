#!/usr/bin/env python3
"""earth_mask.py <dir>: rasterises Natural Earth 1:10m polygons (public domain) into a 21600x10800
equirectangular mask, <dir>/mask.png: R = land (lakes cut out), G = ice (glaciers + Antarctic ice
shelves). Needs `pyshp` and Pillow. Rings are drawn with the even-odd rule (holes punched)."""
import sys, os
import shapefile
from PIL import Image, ImageDraw
d = sys.argv[1]; W, H = 21600, 10800
def raster(names, img, val):
    dr = ImageDraw.Draw(img)
    for n in names:
        r = shapefile.Reader(os.path.join(d, 'ne', n))
        for sh in r.iterShapes():
            pts = sh.points; parts = list(sh.parts) + [len(pts)]
            for a, b in zip(parts[:-1], parts[1:]):
                ring = [((x + 180.0) / 360.0 * W, (90.0 - y) / 180.0 * H) for x, y in pts[a:b]]
                if len(ring) < 3: continue
                # rings: exterior clockwise in shapefiles; holes counter-clockwise
                s = sum((ring[i][0] - ring[i - 1][0]) * (ring[i][1] + ring[i - 1][1]) for i in range(len(ring)))
                hole = s > 0  # image y points down: clockwise-in-lat/lon is counter-clockwise here
                dr.polygon(ring, fill=(0 if hole else val))
land = Image.new('L', (W, H), 0); raster(['ne_10m_land'], land, 255)
lakes = Image.new('L', (W, H), 0); raster(['ne_10m_lakes'], lakes, 255)
ice = Image.new('L', (W, H), 0); raster(['ne_10m_glaciated_areas', 'ne_10m_antarctic_ice_shelves_polys'], ice, 255)
from PIL import ImageChops
land = ImageChops.subtract(land, lakes)
Image.merge('RGB', (land, ice, Image.new('L', (W, H), 0))).save(os.path.join(d, 'mask.png'))
print('ok mask.png')
