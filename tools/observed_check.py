#!/usr/bin/env python3
"""Optional OFFLINE independent arithmetic checks, requires astropy + scipy.
Not an observational accuracy claim, runtime dependency, or default test dependency.
"""
import argparse
import json
import math
from pathlib import Path
import struct
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from observed_ingest import PC, MPC, RAW, load_table, sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--kind', choices=['cns5','desi'], required=True)
    a=ap.parse_args()
    meta=json.loads((a.root/'diagnostic-v1.json').read_text())
    binary=a.root/'diagnostic-v1.bin'
    assert sha(binary)==meta['derived_sha256']
    rows,_=load_table(a.root/RAW[a.kind],meta['raw_sha256'],meta['counts']['raw'])
    if a.kind=='desi':
        rows=[r for r in rows if r['zcat_primary'] in ('true','1','t','T')]
    key='cns5_id' if a.kind=='cns5' else 'targetid'
    raw={int(r[key]):r for r in rows}
    data=binary.read_bytes();count=struct.unpack_from('<Q',data,24)[0]
    records=list(struct.iter_unpack('<QQ7d',data[64:]));assert len(records)==count
    indices=list(range(0,len(records),max(1,len(records)//8)))[:8]
    if a.kind=='cns5':
        indices=indices[:4]+[i for i,r in enumerate(records) if r[5]!=2016][:4]
    checks=[];model=FlatLambdaCDM(H0=70,Om0=.3,Tcmb0=0)
    for i in indices:
        r=records[i];src=raw[r[0]]
        if a.kind=='cns5':
            assert r[6]==float(src['parallax']) and r[5]==float(src['epoch'])
            radius=1000/float(src['parallax'])*PC
        else:
            assert r[6]==float(src['z'])
            radius=float(model.comoving_distance(float(src['z'])).value)*MPC
        independent=SkyCoord(ra=float(src['ra'])*u.deg,dec=float(src['dec'])*u.deg,distance=radius*u.m,frame='icrs').cartesian.xyz.to_value(u.m)
        residual=max(abs(float(v)-w) for v,w in zip(independent,r[2:5]))
        relative=residual/radius
        assert relative<1e-10
        checks.append(dict(id=str(r[0]),native_epoch=None if math.isnan(r[5]) else r[5],relative_xyz_residual=relative,residual_pc=residual/PC))
    report=dict(kind=a.kind,raw_sha256=meta['raw_sha256'],derived_sha256=meta['derived_sha256'],checks=checks,note='independent numerical agreement only, not observational accuracy; no epoch propagation')
    (a.root/'numerical_crosscheck.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
