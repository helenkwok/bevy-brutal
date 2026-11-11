#!/usr/bin/env python3
"""Normalise observed catalogues (GCNS, Gaia DR3 + Bailer-Jones EDR3 distances, Hipparcos) into
RSRC0001 records for the Rust merge/tile builder. NO synthetic rows: every output record is one
input catalogue row; rows that cannot be placed (no G, no distance, plx<=0) are counted and dropped.

Record (56 B, little endian): id u64 | match_gaia u64 | x,y,z f64 pc (Sol-local ICRS, epoch J2016.0)
| gmag f32 (apparent Gaia G or documented transform) | bp_rp f32 (NaN = none) | flags u32 | pad u32.
"""
import csv, hashlib, json, math, struct, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

ROOT = Path('/workspace/data/astro/real_starfield')
RAW = ROOT / 'raw'
NORM = ROOT / 'norm'
GCNS_CSV = Path('/workspace/data/astro/gcns/gcns-main.csv')
REC = struct.Struct('<QQdddffII')
assert REC.size == 56

# flags
F_HIP_POS, F_G_FROM_V, F_G_FROM_HP, F_COL_FROM_VI, F_COL_FROM_BV = 1, 2, 4, 8, 16
F_EPOCH_PROP, F_LOW_PLX_SNR, F_RUWE, F_GEO_DIST, F_COL_RANGE = 32, 64, 128, 256, 512
F_HIP_BEYOND_1KPC = 1024
F_HIP_DIST_BJ = 2048


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fnum(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def unit(ra_deg, dec_deg):
    a, d = math.radians(ra_deg), math.radians(dec_deg)
    return math.cos(d) * math.cos(a), math.cos(d) * math.sin(a), math.sin(d)


class Writer:
    def __init__(self, path):
        self.path, self.f, self.n = path, open(path, 'wb'), 0
        self.f.write(b'RSRC0001' + struct.pack('<Q', 0))

    def put(self, sid, match, xyz, g, bprp, flags):
        assert sid > 0 and all(math.isfinite(c) for c in xyz) and math.isfinite(g)
        self.f.write(REC.pack(sid, match, *xyz, g, float('nan') if bprp is None else bprp, flags, 0))
        self.n += 1

    def close(self):
        self.f.seek(8)
        self.f.write(struct.pack('<Q', self.n))
        self.f.close()


def mamajek():
    rows = []
    cols = None
    for line in open(RAW / 'mamajek_EEM_dwarf.txt'):
        if line.startswith('#SpT'):
            cols = line[1:].split()
            continue
        if cols and line.strip() and not line.startswith('#'):
            parts = line.split()
            if len(parts) < len(cols) - 1 or not parts[0][0].isalpha():
                if rows:
                    break
                continue
            rows.append(dict(zip(cols, parts)))
    out = []
    for r in rows:
        t, bv, br = fnum(r.get('Teff')), fnum(r.get('B-V')), fnum(r.get('Bp-Rp'))
        vi = fnum(r.get('V-Ic'))
        if br is None and vi is not None and -0.4 < vi < 5.0:
            # Hot rows lack Bp-Rp in the table: Gaia EDR3 doc Table 5.7 GBP-GRP = f(V-Ic).
            br = poly(vi, BPRP_VI_JC)
        out.append((t, bv, br))
    return out


def interp(x, xs, ys):
    order = np.argsort(xs)
    xs, ys = np.asarray(xs)[order], np.asarray(ys)[order]
    return float(np.interp(x, xs, ys))


# Wyman, Sloan & Shirley 2013 (JCGT 2:2) multi-lobe fit to CIE 1931 2-degree CMFs.
def _g(l, mu, s1, s2):
    s = np.where(l < mu, s1, s2)
    return np.exp(-0.5 * ((l - mu) / s) ** 2)


def blackbody_rgb(teff):
    lam = np.arange(360.0, 831.0, 1.0)
    xb = 1.056 * _g(lam, 599.8, 37.9, 31.0) + 0.362 * _g(lam, 442.0, 16.0, 26.7) - 0.065 * _g(lam, 501.1, 20.4, 26.2)
    yb = 0.821 * _g(lam, 568.8, 46.9, 40.5) + 0.286 * _g(lam, 530.9, 16.3, 31.1)
    zb = 1.217 * _g(lam, 437.0, 11.8, 36.0) + 0.681 * _g(lam, 459.0, 26.0, 13.8)
    l = lam * 1e-9
    b = 1.0 / (l ** 5 * (np.exp(1.438776877e-2 / (l * teff)) - 1.0))
    X, Y, Z = (b * xb).sum(), (b * yb).sum(), (b * zb).sum()
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    rgb = np.clip(M @ np.array([X, Y, Z]) / Y, 0.0, None)
    y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return (rgb / y).tolist()


LUT_MIN, LUT_MAX, LUT_N = -0.6, 5.0, 255  # code 255 = no colour


def colour_lut(table):
    xs = [br for t, bv, br in table if t and br is not None]
    ts = [math.log10(t) for t, bv, br in table if t and br is not None]
    lut = []
    for i in range(LUT_N):
        br = LUT_MIN + (LUT_MAX - LUT_MIN) * i / (LUT_N - 1)
        teff = 10 ** interp(br, xs, ts)
        lut.append(dict(code=i, bp_rp=round(br, 5), teff=round(teff, 1), rgb=[round(c, 6) for c in blackbody_rgb(teff)]))
    return lut, (min(xs), max(xs))


def poly(x, c):
    return sum(ci * x ** i for i, ci in enumerate(c))


# Gaia EDR3 documentation sec. 5.5.1 (Riello et al. 2021) Tables 5.6/5.7.
G_V_VI = [-0.01597, -0.02809, -0.2483, 0.03656, -0.002939]   # -0.4 < V-Ic < 5.0, sigma 0.0272
G_HP_BV = [-0.02392, -0.4069, 0.04569, -0.0452]              # -0.25 < B-V < 1.9, sigma 0.024
BPRP_VI_JC = [-0.03298, 1.259, -0.1279, 0.01631]        # -0.4 < V-Ic < 5.0, sigma 0.0446
BPRP_VI_HIP = [-0.01612, 1.274, -0.08143]                    # -0.5 < V-I < 3.5, sigma 0.082


def gcns(w, stats):
    with open(GCNS_CSV, newline='') as f:
        for r in csv.DictReader(f):
            stats['rows'] += 1
            g = fnum(r['phot_g_mean_mag'])
            if g is None:
                stats['dropped_no_g'] += 1
                continue
            d = float(r['dist_50']) * 1000.0
            u = unit(float(r['ra']), float(r['dec']))
            bp, rp = fnum(r['phot_bp_mean_mag']), fnum(r['phot_rp_mean_mag'])
            bprp = bp - rp if bp is not None and rp is not None else None
            stats['no_colour'] += bprp is None
            w.put(int(r['source_id']), 0, tuple(c * d for c in u), g, bprp, 0)


def gaia(w, stats, path):
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            stats['rows'] += 1
            g = fnum(r['phot_g_mean_mag'])
            dp, dg = fnum(r['r_med_photogeo']), fnum(r['r_med_geo'])
            flags = 0
            d = dp
            if d is None:
                d, flags = dg, flags | F_GEO_DIST
            if g is None or d is None or not 0 < d <= 1000.0:
                stats['dropped'] += 1
                continue
            plx, e = fnum(r['parallax']), fnum(r['parallax_error'])
            if plx is not None and e and plx / e < 5:
                flags |= F_LOW_PLX_SNR
            ruwe = fnum(r['ruwe'])
            if ruwe is not None and ruwe > 1.4:
                flags |= F_RUWE
            bp, rp = fnum(r['phot_bp_mean_mag']), fnum(r['phot_rp_mean_mag'])
            bprp = bp - rp if bp is not None and rp is not None else None
            stats['no_colour'] += bprp is None
            stats['geo_fallback'] += bool(flags & F_GEO_DIST)
            u = unit(float(r['ra']), float(r['dec']))
            w.put(int(r['source_id']), 0, tuple(c * d for c in u), g, bprp, flags)


def hip(w, stats, table, far_vmax):
    v1 = {}
    with open(RAW / 'hip1_main.csv', newline='') as f:
        for r in csv.DictReader(f):
            v1[int(r['hip'])] = (fnum(r['vmag']), fnum(r['v_i']), fnum(r['b_v']))
    nb = {}
    with open(RAW / 'hip2_gaiadr3_best_neighbour.csv', newline='') as f:
        for r in csv.DictReader(f):
            nb[int(r['original_ext_source_id'])] = int(r['source_id'])
    bj = {}
    with open(RAW / 'hip2_gaiadr3_bj_distances.csv', newline='') as f:
        for r in csv.DictReader(f):
            d = fnum(r['r_med_photogeo'])
            d = d if d is not None else fnum(r['r_med_geo'])
            if d is not None and d > 0:
                bj[int(r['hip'])] = d
    bv_t = [(bv, br) for t, bv, br in table if bv is not None and br is not None]
    dt = 2016.0 - 1991.25
    with open(RAW / 'hip2_newreduction.csv', newline='') as f:
        for r in csv.DictReader(f):
            stats['rows'] += 1
            h = int(r['hip'])
            plx, eplx = fnum(r['plx']), fnum(r['e_plx'])
            snr = plx / eplx if plx is not None and eplx else (math.inf if plx and plx > 0 else -math.inf)
            dist_flags = 0
            if plx is not None and plx > 0 and snr >= 5:
                d = 1000.0 / plx
            elif h in bj:
                # Weak Hipparcos parallax: use the Bailer-Jones EDR3 distance of the Gaia DR3 best neighbour.
                d, dist_flags = bj[h], F_HIP_DIST_BJ
                stats['dist_from_bj'] += 1
            elif plx is not None and plx > 0 and snr >= 1:
                d, dist_flags = 1000.0 / plx, F_LOW_PLX_SNR
            else:
                stats['dropped_distance_unconstrained'] += 1
                continue
            vmag, vi1, bv1 = v1.get(h, (None, None, None))
            vi = fnum(r['v_i']) if fnum(r['v_i']) is not None else vi1
            bv = fnum(r['b_v']) if fnum(r['b_v']) is not None else bv1
            hp = fnum(r['hp_mag'])
            flags = F_HIP_POS | F_EPOCH_PROP | dist_flags
            if vmag is not None and vi is not None and -0.4 < vi < 5.0:
                g, flags = vmag + poly(vi, G_V_VI), flags | F_G_FROM_V
            elif hp is not None and bv is not None and -0.25 < bv < 1.9:
                g, flags = hp + poly(bv, G_HP_BV), flags | F_G_FROM_HP
            else:
                stats['dropped_no_photometry'] += 1
                continue
            if vi is not None and -0.5 < vi < 3.5:
                bprp, flags = poly(vi, BPRP_VI_HIP), flags | F_COL_FROM_VI
            elif bv is not None:
                lo, hi = min(b for b, _ in bv_t), max(b for b, _ in bv_t)
                bprp, flags = interp(min(max(bv, lo), hi), [b for b, _ in bv_t], [c for _, c in bv_t]), flags | F_COL_FROM_BV
                if not lo <= bv <= hi:
                    flags |= F_COL_RANGE
            else:
                bprp = None
                stats['no_colour'] += 1
            if d > 1000.0:
                if g <= far_vmax:
                    flags |= F_HIP_BEYOND_1KPC
                    stats['kept_beyond_1kpc_bright'] += 1
                else:
                    stats['dropped_beyond_1kpc'] += 1
                    continue
            # Rigorous linear space-motion of the unit vector J1991.25 -> J2016.0 (no RV term).
            ra, de = math.radians(float(r['ra'])), math.radians(float(r['dec']))
            pa, pd = float(r['pm_ra']), float(r['pm_de'])  # mas/yr, pm_ra is mu_alpha*
            p = np.array([math.cos(de) * math.cos(ra), math.cos(de) * math.sin(ra), math.sin(de)])
            ea = np.array([-math.sin(ra), math.cos(ra), 0.0])
            ed = np.array([-math.sin(de) * math.cos(ra), -math.sin(de) * math.sin(ra), math.cos(de)])
            k = math.radians(1.0 / 3.6e6) * dt
            p = p + k * (pa * ea + pd * ed)
            p /= np.linalg.norm(p)
            w.put(h, nb.get(h, 0), tuple(float(c) * d for c in p), g, bprp, flags)


def main():
    only = set(sys.argv[1:])  # optional subset: gcns gaia hip
    NORM.mkdir(parents=True, exist_ok=True)
    table = mamajek()
    lut, br_range = colour_lut(table)
    (ROOT / 'colour_lut.json').write_text(json.dumps(dict(
        code_rule=f'code=round((bp_rp-({LUT_MIN}))/({LUT_MAX}-({LUT_MIN}))*{LUT_N-1}) clamped 0..{LUT_N-1}; 255=no colour',
        teff_source='Pecaut & Mamajek 2013 / Mamajek EEM dwarf table v2022.04.16 (Bp-Rp -> Teff; rows without Bp-Rp use GBP-GRP=f(V-Ic) from Gaia EDR3 doc Table 5.7; linear interp in log Teff; clamped beyond table range %s)' % (br_range,),
        rgb_model='Planck spectrum x CIE1931 2deg CMF (Wyman-Sloan-Shirley 2013 analytic fit) -> XYZ -> linear sRGB (D65), negatives clipped, Rec.709 luminance normalised to 1',
        caveats='Observed (reddened) colours, dwarf Teff sequence; chromaticity only, brightness carried by G-band flux',
        lut=lut), indent=1))
    with open(ROOT / 'colour_lut.bin', 'wb') as f:
        for e in lut:
            f.write(struct.pack('<4f', *e['rgb'], e['bp_rp']))
    sp = ROOT / 'ingest_stats.json'
    stats = json.loads(sp.read_text()) if sp.exists() and only else {}
    if not only or 'gcns' in only:
        s = stats['gcns'] = dict(rows=0, dropped_no_g=0, no_colour=0)
        w = Writer(NORM / 'gcns.rsrc'); gcns(w, s); w.close(); s['written'] = w.n
    if not only or 'gaia' in only:
        gavo = RAW / 'gaia_dr3_bj_1kpc_g13.csv'
        paths = [gavo] if gavo.exists() else sorted(RAW.glob('esa_chunks/*.csv'))
        s = stats['gaia'] = dict(rows=0, dropped=0, no_colour=0, geo_fallback=0, inputs=[p.name for p in paths])
        w = Writer(NORM / 'gaia.rsrc')
        for gp in paths:
            gaia(w, s, gp)
        w.close(); s['written'] = w.n
    far_vmax = 6.5
    if not only or 'hip' in only:
        s = stats['hip'] = dict(rows=0, dist_from_bj=0, dropped_distance_unconstrained=0, dropped_no_photometry=0, dropped_beyond_1kpc=0,
                                kept_beyond_1kpc_bright=0, no_colour=0, far_rule=f'Hip rows beyond 1000 pc kept only if G<={far_vmax}', distance_rule='1/plx if plx/e_plx>=5; else Bailer-Jones EDR3 (photogeo, else geo) of the Gaia DR3 best neighbour; else 1/plx if plx/e_plx>=1; else dropped')
        w = Writer(NORM / 'hip.rsrc'); hip(w, s, table, far_vmax); w.close(); s['written'] = w.n
    stats['created'] = datetime.now(ZoneInfo('Australia/Adelaide')).isoformat() + ' Australia/Adelaide'
    stats['files'] = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(NORM.glob('*.rsrc'))}
    (ROOT / 'ingest_stats.json').write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
