# Mosaic Copernicus GLO-30 (PixelIsPoint, 1 arcsec, tile top-left = (lat+1, lon)) into two lat/lon grids.
import numpy as np, tifffile, json, os
A = 1 / 3600
def build(name, la_top, la_bot, lo_l, lo_r, step):
    rows = int(round((la_top - la_bot) * 3600)); cols = int(round((lo_r - lo_l) * 3600))
    full = np.zeros((rows, cols), np.float32)
    for la in range(int(np.floor(la_bot)), int(np.ceil(la_top))):
        for lo in range(int(np.floor(lo_l)), int(np.ceil(lo_r))):
            f = f'Copernicus_DSM_COG_10_N{la:02d}_00_E{lo:03d}_00_DEM.tif'
            t = tifffile.imread(f).astype(np.float32)
            # tile row i -> lat la+1 - i/3600 ; mosaic row r -> la_top - r/3600
            r0 = int(round((la_top - (la + 1)) * 3600)); c0 = int(round((lo - lo_l) * 3600))
            rs, cs = max(r0, 0), max(c0, 0)
            re, ce = min(r0 + 3600, rows), min(c0 + 3600, cols)
            if rs < re and cs < ce:
                full[rs:re, cs:ce] = t[rs - r0:re - r0, cs - c0:ce - c0]
    if step > 1:
        r2, c2 = rows // step, cols // step
        full = full[:r2 * step, :c2 * step].reshape(r2, step, c2, step).mean(axis=(1, 3))
    lat0 = la_top - (step - 1) / 2 * A; lon0 = lo_l + (step - 1) / 2 * A
    full.astype('<f4').tofile(name + '.f32')
    h = dict(lat0=lat0, lon0=lon0, dlat=step * A, dlon=step * A, rows=full.shape[0], cols=full.shape[1],
             min=float(full.min()), max=float(full.max()))
    json.dump(h, open(name + '.json', 'w'), indent=1); print(name, h)
build('dem_inner', 46.52, 45.68, 7.0, 8.2, 1)
build('dem_outer', 47.70, 44.50, 5.2, 10.0, 3)
