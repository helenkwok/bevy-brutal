#!/usr/bin/env python3
"""Compress the asset_bench bricks (/tmp/asset_bricks.bin) with Google Draco (DracoPy) per
brick, the same streaming unit as our format. Quantisation box = the brick box (origin 0,
range = brick extent), so Draco's position grid is comparable to ours. Reports bytes,
bits/triangle, decode speed and max position error.
Run: /workspace/.venv-draco/bin/python tools/draco_compare.py"""
import struct, time, sys
import numpy as np
import DracoPy

def load(path="/tmp/asset_bricks.bin"):
    b = open(path, "rb").read()
    (n,) = struct.unpack_from("<I", b, 0); o = 4; out = []
    for _ in range(n):
        ext, nv, nt = struct.unpack_from("<fII", b, o); o += 12
        p = np.frombuffer(b, np.float32, nv * 3, o).reshape(nv, 3); o += nv * 12
        nn = np.frombuffer(b, np.float32, nv * 3, o).reshape(nv, 3); o += nv * 12
        f = np.frombuffer(b, np.uint32, nt * 3, o).reshape(nt, 3); o += nt * 12
        out.append((ext, p, nn, f))
    return out

bricks = load()
nt = sum(len(f) for _, _, _, f in bricks)
nv = sum(len(p) for _, p, _, _ in bricks)
raw = nv * 24 + nt * 12
print(f"{len(bricks)} bricks, {nv} vertices, {nt} triangles, raw {raw/1e6:.2f} MB")
print("| Draco setting | bytes | ratio vs raw | bits/tri | decode MB/s out (1 core, incl. numpy) | max pos err (mm) |")
print("|---|---|---|---|---|---|")
for qp, qn, cl in [(11, 10, 7), (11, 10, 10), (14, 10, 7), (14, 10, 10), (16, 10, 10), (11, 8, 10)]:
    blobs = []; err = 0.0
    for ext, p, nn, f in bricks:
        # small margin: phi-displaced / boundary vertices may sit marginally outside the box
        lo = p.min(axis=0).astype(np.float64); rng = float((p.max(axis=0) - lo).max())
        blobs.append(DracoPy.encode(p, f, quantization_bits=qp, compression_level=cl, quantization_range=rng, quantization_origin=lo.tolist(), normals=np.ascontiguousarray(nn, np.float64), normal_quantization_bits=qn))
    size = sum(len(x) for x in blobs)
    best = 1e9
    for rep in range(3):
        t = time.perf_counter()
        dec = [DracoPy.decode(x) for x in blobs]
        best = min(best, time.perf_counter() - t)
    # position error: Draco reorders vertices; compare sorted point sets per brick
    for (ext, p, nn, f), d in zip(bricks[:60], dec[:60]):
        q = np.asarray(d.points, np.float64)
        # nearest-neighbour error via rounding-free brute force on a subsample
        idx = np.linspace(0, len(p) - 1, min(len(p), 200)).astype(int)
        for i in idx:
            err = max(err, float(np.min(np.linalg.norm(q - p[i], axis=1))))
    print(f"| qp {qp}, qn {qn}, cl {cl} | {size} | {raw/size:.2f}:1 | {size*8/nt:.2f} | {raw/best/1e6:.0f} | {err*1e3:.2f} |")
