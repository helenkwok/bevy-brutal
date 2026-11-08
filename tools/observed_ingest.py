#!/usr/bin/env python3
"""Offline, pinned CNS5/DESI positional diagnostics; no merge or photometric invention.
Only --fetch performs network access. Raw tables/terms remain outside git.
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

PC = 149597870700.0 * 648000 / math.pi  # IAU 2012 AU, IAU 2015 parsec
MPC = PC * 1e6
MAX_ROWS = 100000
MAX_BYTES = 100000000
RAW = {'cns5': 'cns5-full.vot', 'desi': 'galaxies-field.vot'}
DESI_WHERE = ("FROM desi_dr1.zpix AS z JOIN desi_dr1.photometry AS p ON z.targetid=p.targetid "
              "WHERE p.ra>=150 AND p.ra<160 AND p.dec>=-2 AND p.dec<2 AND z.survey='main' "
              "AND z.program IN ('bright','dark') AND z.spectype='GALAXY' AND z.z>=0.01 AND z.z<0.5")
DESI_QUERY = ('SELECT z.targetid,p.ra,p.dec,p.ref_epoch,z.z,z.zerr,z.zwarn,z.spectype,z.subtype,'
              'z.survey,z.program,z.zcat_primary,z.main_primary,z.zcat_nspec,z.deltachi2,'
              'z.coadd_fiberstatus,z.min_mjd,z.max_mjd ' + DESI_WHERE + ' ORDER BY z.targetid,z.program')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_table(path, expected_sha=None, expected_rows=None, max_rows=MAX_ROWS):
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('raw response exceeds 100 MB')
    if expected_sha and sha(path) != expected_sha:
        raise ValueError('raw checksum mismatch')
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise ValueError('malformed/truncated VOTable') from e
    statuses = [x.get('value') for x in root.iter() if x.tag.split('}')[-1] == 'INFO' and x.get('name') == 'QUERY_STATUS']
    if not statuses or any(s != 'OK' for s in statuses):
        raise ValueError(f'TAP error/truncation/missing status: {statuses}')
    if len(root.findall('.//{*}TABLE')) != 1 or root.find('.//{*}TABLEDATA') is None:
        raise ValueError('exactly one TABLEDATA table required')
    fields = root.findall('.//{*}FIELD')
    names = [f.attrib['name'] for f in fields]
    if len(names) != len(set(names)):
        raise ValueError('duplicate column name')
    schema = {f.attrib['name']: dict(f.attrib) for f in fields}
    nulls = {f.attrib['name']: next((v.get('null') for v in f if v.tag.split('}')[-1] == 'VALUES'), None) for f in fields}
    rows = []
    for tr in root.findall('.//{*}TR'):
        if len(tr) != len(names) or len(rows) >= max_rows:
            raise ValueError('row-width/MAXREC bound violated')
        rows.append({n: None if cell.text in (None, '', nulls[n]) else cell.text for n, cell in zip(names, tr)})
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError('pinned row count mismatch; review acquisition, not a fixed release-size assertion')
    return rows, schema


def number(row, key):
    value = row[key]
    if value is None or value in ('NaN', 'nan', ''):
        return None
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f'infinite field {key}')
    return f


def ident(value, optional=False):
    if value is None and optional:
        return 0
    n = int(value)
    if not 0 <= n < 2**64 or str(n) != value:
        raise ValueError('noncanonical/out-of-range integer ID')
    return n


def position(ra, dec, radius):
    if not (0 <= ra < 360 and -90 <= dec <= 90 and radius > 0 and math.isfinite(radius)):
        raise ValueError('invalid coordinate/distance')
    a, d = math.radians(ra), math.radians(dec)
    return radius*math.cos(d)*math.cos(a), radius*math.cos(d)*math.sin(a), radius*math.sin(d)


def convert_cns5(rows):
    records, ids, stats = [], set(), collections.Counter()
    for r in rows:
        sid, gaia = ident(r['cns5_id']), ident(r['gaia_edr3_id'], optional=True)
        if sid in ids:
            raise ValueError('duplicate CNS5 ID')
        ids.add(sid)
        ra, dec, epoch, parallax, error, g = (number(r, k) for k in ['ra', 'dec', 'epoch', 'parallax', 'parallax_error', 'g_mag'])
        stats['missing_direct_g'] += g is None
        if any(v is None for v in [ra, dec, epoch, parallax, error]) or parallax <= 0 or error <= 0 or error/parallax > .1:
            stats['position_ineligible'] += 1
            continue
        if not 1800 <= epoch <= 2200:
            raise ValueError('invalid native epoch')
        xyz = position(ra, dec, 1000/parallax*PC)
        records.append((sid, gaia, *xyz, epoch, parallax, error, g if g is not None else math.nan))
        stats['eligible_with_direct_g'] += g is not None
    return records, dict(stats, raw=len(rows), eligible=len(records), dropped=len(rows)-len(records))


def comoving_mpc(z):
    """Flat LCDM H0=70, Om=.3, no radiation: a disclosed coordinate assumption.

    Composite Simpson512; checked separately against Astropy, not cz/H0 or dL.
    """
    if not math.isfinite(z) or not 0 <= z <= 10:
        raise ValueError('redshift outside supported 0..10')
    n = 512
    h = z/n
    total = 0.
    for i in range(n+1):
        total += (1 if i in (0, n) else 4 if i % 2 else 2)/math.sqrt(.3*(1+i*h)**3+.7)
    return 299792.458/70*total*h/3


def convert_desi(rows):
    records, ids = [], set()
    for r in rows:
        sid = ident(r['targetid'])
        ra, dec, z, err, ep = (number(r, k) for k in ['ra', 'dec', 'z', 'zerr', 'ref_epoch'])
        if sid == 0 or any(v is None for v in [ra, dec, z]):
            raise ValueError('missing DESI identifier/coordinate/redshift')
        if not (150 <= ra < 160 and -2 <= dec < 2 and .01 <= z < .5 and r['spectype'] == 'GALAXY' and r['survey'] == 'main' and r['program'] in ('bright', 'dark')):
            raise ValueError('row outside documented field selection')
        if r['zcat_primary'] not in ('true', 'false', '1', '0', 't', 'f', 'T', 'F'):
            raise ValueError('invalid primary boolean')
        if r['zcat_primary'] not in ('true', '1', 't', 'T') or int(r['zwarn']) != 0 or int(r['coadd_fiberstatus']) != 0 or err is None or err < 0:
            continue
        if sid in ids:
            raise ValueError('ambiguous duplicate quality-primary DESI target')
        ids.add(sid)
        xyz = position(ra, dec, comoving_mpc(z)*MPC)
        if ep is not None and ep != 0 and not 1800 <= ep <= 2200:
            raise ValueError('invalid reference epoch')
        records.append((sid, 0, *xyz, ep if ep else math.nan, z, err, math.nan))
    return records, dict(raw=len(rows), eligible=len(records), dropped=len(rows)-len(records), raw_unique_ids=len({r['targetid'] for r in rows}))


def atomic(path, data):
    tmp = path.with_suffix(path.suffix+'.part')
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def validate_schema(kind, schema):
    required = {'ra': 'deg', 'dec': 'deg'}
    if kind == 'cns5':
        required.update(epoch='yr', parallax='mas', parallax_error='mas', g_mag='mag')
    for k, unit in required.items():
        if schema.get(k, {}).get('unit') != unit:
            raise ValueError(f'unit/schema drift: {k}')
    for k in (['cns5_id', 'gaia_edr3_id'] if kind == 'cns5' else ['targetid', 'zwarn', 'coadd_fiberstatus']):
        if schema.get(k, {}).get('datatype') not in ('long', 'int'):
            raise ValueError(f'integer schema drift: {k}')


def check_query(kind, path):
    root = ET.parse(path).getroot()
    queries = [e.get('value') for e in root.iter() if e.tag.split('}')[-1] == 'INFO' and e.get('name', '').lower() == 'query']
    expected = 'SELECT * FROM cns5.main' if kind == 'cns5' else DESI_QUERY
    if len(queries) != 1 or ''.join(queries[0].split()).lower() != ''.join(expected.split()).lower():
        raise ValueError('query/selection provenance mismatch; no arbitrary TOP samples')


def match_gcns(rows, path):
    """Exact EDR3 IDs only. Native epochs are irrelevant to this identifier comparison.

    No tiny-angle match across unpropagated epochs, no RV=0 motion assumption,
    no guessed non-Gaia associations, and absolutely no automatic supplement.
    """
    gids = set()
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            sid = ident(r['source_id'])
            if sid == 0 or sid in gids or len(gids) >= 400000:
                raise ValueError('duplicate/oversized GCNS crossmatch input')
            gids.add(sid)
    matched = sum(ident(r['gaia_edr3_id'], optional=True) in gids for r in rows)
    unique = {ident(r['gaia_edr3_id'], optional=True) for r in rows} - {0}
    return dict(merged=False, matched_rows=matched, unresolved_rows=len(rows)-matched,
                no_gaia_id=sum(r['gaia_edr3_id'] is None for r in rows), unique_gaia_ids=len(unique),
                unique_shared_gaia_ids=len(unique & gids), gcns_rows_scanned=len(gids),
                gcns_sha256=sha(path), policy='exact ID overlap, NOT unique gaps; all nonmatches unresolved; no angular/epoch merge')


def ingest(kind, root, gcns=None):
    root = Path(root)
    pin = json.loads((root/'manifest.json').read_text())
    raw = root/RAW[kind]
    rows, schema = load_table(raw, pin['files'][raw.name]['sha256'], pin['raw_rows'])
    validate_schema(kind, schema)
    check_query(kind, raw)
    # A copied snapshot schema is independently retained at acquisition; validate field types/units.
    if (root/'retained-schema.json').exists():
        expected = {x['name']: x for x in json.loads((root/'retained-schema.json').read_text())}
        if set(schema) != set(expected) or any(schema[k].get(a) != expected[k].get(a) for k in schema for a in ('datatype', 'unit', 'arraysize')):
            raise ValueError('retained schema drift')
    records, stats = (convert_cns5 if kind == 'cns5' else convert_desi)(rows)
    digest = sha(raw)
    data = bytearray(struct.pack('<8sQQQ32s', b'OBSdiag1', 1 if kind == 'cns5' else 2, len(rows), len(records), bytes.fromhex(digest)))
    for record in records:
        data.extend(struct.pack('<QQ7d', *record))
    sidecar = dict(kind=kind, raw_sha256=digest, derived_sha256=hashlib.sha256(data).hexdigest(), counts=stats,
                   source_manifest_sha256=sha(root/'manifest.json'), schema=schema, source_acquisition=pin,
                   axes='right-handed ICRS XYZ metres; f64; no epoch propagation; no inserted Sol',
                   mode='independent UNMERGED equal-weight position diagnostic; no stellar luminosity/flux or invented G/RGB',
                   distance_model='S/N>=10 reciprocal measured parallax, not a Bayesian posterior' if kind == 'cns5' else 'radial comoving flat LCDM H0=70 km/s/Mpc Om=.3 Ode=.7 radiation=0; model not measured distance',
                   limitations='CNS5 GAVO snapshot, not certified full5931 or unique GCNS supplement; native epochs retained; upstream rights unresolved' if kind == 'cns5' else 'selected field only; primary+ZWARN0+fibre0+finite nonnegative zerr; OBJTYPE unavailable/not applied; no universal DeltaChi2 cut; redshift-space/lightcone/peculiar-velocity limitations',
                   created=datetime.now(ZoneInfo('Australia/Adelaide')).isoformat()+' Australia/Adelaide')
    if gcns is not None:
        if kind != 'cns5':
            raise ValueError('GCNS matching is only defined for CNS5')
        gcns = Path(gcns)
        gp = json.loads((gcns.parent/'manifest.json').read_text())
        if sha(gcns) != gp['files'][gcns.name]['sha256']:
            raise ValueError('GCNS crossmatch input checksum mismatch')
        sidecar['gcns_matching'] = match_gcns(rows, gcns)
        if sidecar['gcns_matching']['gcns_rows_scanned'] != gp['raw_rows']:
            raise ValueError('GCNS crossmatch input row count mismatch')
    # Validate and serialize both before replacing either output. Each replacement is atomic.
    metadata = (json.dumps(sidecar, indent=2, allow_nan=False)+'\n').encode()
    atomic(root/'diagnostic-v1.json', metadata)
    atomic(root/'diagnostic-v1.bin', data)
    return sidecar


def fetch(kind, root, expected_rows):
    """Explicit re-acquisition to a NEW root only; preserve the old snapshot unchanged."""
    root = Path(root)
    if root.exists():
        raise FileExistsError('fetch requires a new root; never overwrite a retained snapshot')
    endpoint = 'https://dc.g-vo.org/tap/sync' if kind == 'cns5' else 'https://datalab.noirlab.edu/tap/sync'
    query = 'SELECT * FROM cns5.main' if kind == 'cns5' else DESI_QUERY
    url = endpoint+'?'+urllib.parse.urlencode(dict(REQUEST='doQuery', LANG='ADQL', QUERY=query, FORMAT='votable/td' if kind == 'cns5' else 'votable', MAXREC=MAX_ROWS))
    root.mkdir(parents=True)
    raw = root/RAW[kind]
    with urllib.request.urlopen(url, timeout=180) as response:
        data = response.read(MAX_BYTES+1)
    if len(data) > MAX_BYTES:
        raise ValueError('100 MB fetch bound exceeded')
    atomic(raw, data)
    load_table(raw, expected_rows=expected_rows)
    atomic(root/'manifest.json', json.dumps(dict(raw_rows=expected_rows, query=query, url=url, files={raw.name:dict(sha256=sha(raw), bytes=len(data))}, licence='CNS5 compilation BY4, upstream unresolved' if kind == 'cns5' else 'DESI BY4; required acknowledgments https://data.desi.lbl.gov/doc/acknowledgments/'), indent=2).encode())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kind', choices=['cns5', 'desi'], required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--gcns', type=Path, help='CNS5 exact-ID overlap analysis only, never merges catalogues')
    ap.add_argument('--expected-rows', type=int)
    a = ap.parse_args()
    try:
        if a.fetch:
            if a.expected_rows is None or not 0 < a.expected_rows < MAX_ROWS:
                ap.error('--fetch requires independently established --expected-rows below MAXREC100000')
            fetch(a.kind, a.root, a.expected_rows)
        print(json.dumps(ingest(a.kind, a.root, a.gcns)['counts'], indent=2))
    except (OSError, ValueError, KeyError, TypeError) as e:
        ap.exit(1, f'observed ingest failed: {e}\n')


if __name__ == '__main__':
    main()
