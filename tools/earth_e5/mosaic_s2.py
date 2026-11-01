# Stitch the EOX S2 cloudless tiles into one Web Mercator raster per zoom: raw RGB8 + json.
import os, json, glob, numpy as np
from PIL import Image
LAYER = os.environ.get('S2_LAYER', 's2cloudless-2020_3857')
for z in (11, 13):
    fs = glob.glob(f'{LAYER}/{z}/*.jpg')
    ys = sorted({int(os.path.basename(f).split('_')[0]) for f in fs}); xs = sorted({int(os.path.basename(f).split('_')[1][:-4]) for f in fs})
    y0, x0 = ys[0], xs[0]; H, W = (ys[-1] - y0 + 1) * 256, (xs[-1] - x0 + 1) * 256
    a = np.zeros((H, W, 3), np.uint8)
    for f in fs:
        y, x = os.path.basename(f)[:-4].split('_'); y, x = int(y), int(x)
        a[(y - y0) * 256:(y - y0 + 1) * 256, (x - x0) * 256:(x - x0 + 1) * 256] = np.asarray(Image.open(f).convert('RGB'))
    a.tofile(f's2_z{z}.rgb')
    json.dump(dict(z=z, x0=x0 * 256, y0=y0 * 256, w=W, h=H, layer=LAYER), open(f's2_z{z}.json', 'w'))
    print(z, W, H)
    Image.fromarray(a[::8, ::8]).save(f's2_z{z}_preview.jpg')
