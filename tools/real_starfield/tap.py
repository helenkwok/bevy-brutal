"""Minimal IVOA TAP async client (stdlib + requests). Records every query for manifests."""
import os, sys, time, requests

def run_async(base, query, out_path, fmt='csv', maxrec=None, poll=10, timeout=6 * 3600):
    data = dict(REQUEST='doQuery', LANG='ADQL', FORMAT=fmt, QUERY=query, PHASE='RUN')
    if maxrec is not None:
        data['MAXREC'] = str(maxrec)
    r = requests.post(base.rstrip('/') + '/async', data=data, allow_redirects=False, timeout=120)
    job = r.headers.get('Location') or r.url
    if r.status_code not in (200, 303) or not job:
        raise RuntimeError(f'submit failed {r.status_code}: {r.text[:500]}')
    if not job.startswith('http'):
        job = base.rstrip('/') + '/async/' + job.rsplit('/', 1)[-1]
    # Some services need explicit RUN.
    requests.post(job + '/phase', data=dict(PHASE='RUN'), timeout=60)
    t0 = time.time()
    while True:
        ph = requests.get(job + '/phase', timeout=60).text.strip()
        if ph in ('COMPLETED', 'ERROR', 'ABORTED'):
            break
        if time.time() - t0 > timeout:
            raise RuntimeError('timeout ' + job)
        if os.environ.get('TAP_VERBOSE', '1') == '1':
            print(f'[tap] {job} {ph} {time.time()-t0:.0f}s', file=sys.stderr, flush=True)
        time.sleep(poll)
    if ph != 'COMPLETED':
        err = requests.get(job + '/error', timeout=60).text
        raise RuntimeError(f'{ph}: {err[:2000]}')
    with requests.get(job + '/results/result', stream=True, timeout=600) as res:
        res.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in res.iter_content(1 << 20):
                f.write(chunk)
    return job

if __name__ == '__main__':
    base, out = sys.argv[1], sys.argv[2]
    q = sys.stdin.read()
    import os
    mr = os.environ.get('TAP_MAXREC')
    print(run_async(base, q, out, maxrec=int(mr) if mr else None))
