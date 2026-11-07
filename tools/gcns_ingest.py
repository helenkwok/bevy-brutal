#!/usr/bin/env python3
"""Retain full observed GCNS externally; derive monochrome G-band point input.
No catalogue is bundled. GAVO's GCNS mirror is CC BY-NC 4.0, not the code licence.
"""
import argparse
import collections
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import struct
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

AU = 149597870700.0  # IAU 2012 exact astronomical unit in metres
PC = AU * 648000 / math.pi  # IAU 2015 parsec definition
ROWS = 331312
QUERY = 'SELECT * FROM gcns.main'
URL = 'https://dc.g-vo.org/tap/sync?'
RAW_URL = URL + urllib.parse.urlencode(dict(REQUEST='doQuery', LANG='ADQL', FORMAT='csv', MAXREC=400000, QUERY=QUERY))
SCHEMA_URL = URL + urllib.parse.urlencode(dict(REQUEST='doQuery', LANG='ADQL', FORMAT='votable/td', QUERY="SELECT * FROM TAP_SCHEMA.columns WHERE table_name = 'gcns.main'"))
SERVICE_URL = 'https://dc.g-vo.org/gcns/q/cone/info'


def convert_row(r):
    sid = int(r['source_id'])  # Never pass source IDs through floating point.
    if not 0 < sid < 2**64:
        raise ValueError('invalid source_id')
    keys = ['ra', 'dec', 'dist_1', 'dist_16', 'dist_50', 'dist_84', 'gcns_prob', 'parallax', 'parallax_error']
    a, d, q1, q16, q50, q84, prob, parallax, error = (float(r[k]) for k in keys)
    if not all(math.isfinite(float(r[k])) for k in keys) or not (0 <= a < 360 and -90 <= d <= 90 and 0 < q1 <= q16 <= q50 <= q84 and q1 <= .1 and .38 <= prob <= 1 and parallax > 8 and error >= 0):
        raise ValueError(f'invalid GCNS astrometry/distance/selection: {sid}')
    if not r['phot_g_mean_mag']:
        return None  # Missing photometry is not fabricated.
    g = float(r['phot_g_mean_mag'])
    if not math.isfinite(g) or not -10 < g < 40:
        raise ValueError('invalid G photometry')
    a, d = math.radians(a), math.radians(d)
    radius = q50 * 1000 * PC
    # Right-handed Sol-local ICRS: +X alpha=0,delta=0; +Y alpha=90; +Z north pole.
    x, y, z = radius*math.cos(d)*math.cos(a), radius*math.cos(d)*math.sin(a), radius*math.sin(d)
    if not math.isfinite(radius):
        raise ValueError('nonfinite inferred radius')
    # Reference-observer G/G=0 Vega-band relative flux. NOT bolometric Lsun or RGB.
    proxy = 10**(-.4*g) * (radius/AU) * (radius/AU)
    if not math.isfinite(proxy) or proxy <= 0:
        raise ValueError('nonfinite G-band proxy')
    return sid, x, y, z, proxy


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fetch(url, path):
    if path.exists():
        raise FileExistsError(f'refusing to overwrite retained original {path}; choose a new --root for a new acquisition')
    tmp = path.with_suffix(path.suffix + '.part')
    try:
        with urllib.request.urlopen(url, timeout=600) as src, tmp.open('wb') as dst:
            total = 0
            while block := src.read(1024*1024):
                total += len(block)
                if total > 700000000:
                    raise ValueError('download exceeds 700 MB bound')
                dst.write(block)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def ingest(root, expected_sha=None):
    raw, schema = root/'gcns-main.csv', root/'schema.vot'
    raw_sha = sha(raw)
    if expected_sha and raw_sha != expected_sha:
        raise ValueError('raw SHA256 does not match requested pin')
    xml = ET.parse(schema).getroot()
    fields = [x.attrib['name'] for x in xml.findall('.//{*}FIELD')]
    columns = [dict(zip(fields, [v.text for v in row])) for row in xml.findall('.//{*}TR')]
    units = {x['column_name']: x['unit'] for x in columns}
    for key, unit in dict(ra='deg', dec='deg', dist_50='kpc', dist_16='kpc', dist_84='kpc', parallax='mas', phot_g_mean_mag='mag').items():
        if units.get(key) != unit:
            raise ValueError(f'schema unit mismatch: {key}')
    nulls, ids, count, eligible, outside, refs = collections.Counter(), set(), 0, 0, 0, []
    target = root/'gcns-v1.bin'
    tmp = root/'gcns-v1.bin.part'
    try:
        with raw.open(newline='') as src, tmp.open('wb') as dst:
            reader = csv.DictReader(src)
            if reader.fieldnames is None or set(reader.fieldnames) != set(units) or len(reader.fieldnames) != 74:
                raise ValueError('GCNS schema/column count mismatch')
            dst.write(b'GCNSv001' + struct.pack('<QQ', ROWS, 0))
            for r in reader:
                count += 1
                if count > ROWS or None in r or None in r.values():
                    raise ValueError('malformed/oversized GCNS table')
                sid = int(r['source_id'])
                if sid in ids:
                    raise ValueError('duplicate source_id')
                ids.add(sid)
                nulls.update(k for k,v in r.items() if v == '')
                for c in columns:
                    value=r[c['column_name']]
                    if value and c.get('datatype') in ('float','double') and not math.isfinite(float(value)):
                        raise ValueError(f"nonfinite optional field {c['column_name']}: {sid}")
                    if value and c.get('datatype') in ('long','short','int'):
                        int(value)  # Validate integer encoding without f64 roundoff.
                row = convert_row(r)
                outside += float(r['dist_50']) > .1
                if row is not None:
                    dst.write(struct.pack('<Qdddd', *row))
                    eligible += 1
                    if len(refs) < 3:
                        refs.append(dict(source_id=str(sid), ra=r['ra'], dec=r['dec'], dist_50_kpc=r['dist_50'], icrs_m=row[1:4]))
            if count != ROWS:
                raise ValueError(f'expected full {ROWS} rows, got {count}')
            dst.seek(16)
            dst.write(struct.pack('<Q', eligible))
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    manifest = dict(release='Gaia EDR3 GCNS / Smart et al. 2021 A&A 649 A6', raw_url=RAW_URL, query=QUERY, filters='none; complete gcns.main', schema_url=SCHEMA_URL, terms_url=SERVICE_URL, licence='GAVO: CC BY-NC 4.0; no commercial redistribution permission assumed', raw_rows=count, eligible_rows=eligible, missing_g=count-eligible, median_beyond_100pc=outside, null_counts=dict(nulls), columns=columns, reference_rows=refs, epoch='J2016.0; ICRS, no equinox assigned and no time propagation', axes='Sol-local right-handed ICRS XYZ; translation into engine anchor is not a measured Sun record', display_model='achromatic G-band relative flux 10^(-0.4 G), not measured RGB or bolometric luminosity; observer translation uses median Bayesian distance and inverse square, no extinction correction/variability/uncertainty sampling', files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [raw,schema,target]+[root/n for n in ['service.html','independent_check.json'] if (root/n).exists()]}, manifest_created=datetime.now(ZoneInfo('Australia/Adelaide')).isoformat()+' Australia/Adelaide', references=['https://www.cosmos.esa.int/web/gaia/edr3-gcns','https://doi.org/10.1051/0004-6361/202039498',SERVICE_URL], constants=dict(AU_m=AU,parsec_m=PC))
    # Raw retrieval time is its retained acquisition mtime; conversion time is separate.
    manifest['raw_retrieved'] = datetime.fromtimestamp(raw.stat().st_mtime, ZoneInfo('Australia/Adelaide')).isoformat()+' Australia/Adelaide'
    m = root/'manifest.json.part'
    m.write_text(json.dumps(manifest,indent=2)+'\n')
    m.replace(root/'manifest.json')
    return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('/workspace/data/astro/gcns'))
    ap.add_argument('--fetch',action='store_true',help='Explicit network fetch; by default convert retained local files only')
    ap.add_argument('--expected-sha256')
    a=ap.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    try:
        if a.fetch:
            for url,name in [(RAW_URL,'gcns-main.csv'),(SCHEMA_URL,'schema.vot'),(SERVICE_URL,'service.html')]:fetch(url,a.root/name)
        m=ingest(a.root,a.expected_sha256)
        print(json.dumps({k:m[k] for k in ['raw_rows','eligible_rows','missing_g','median_beyond_100pc','files']},indent=2))
    except (OSError, ValueError, KeyError, ET.ParseError) as e:
        ap.exit(1,f'GCNS ingest failed: {e}\n')

if __name__=='__main__':main()
