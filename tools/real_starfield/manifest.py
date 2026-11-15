#!/usr/bin/env python3
"""Write the RS1 provenance manifest: sources, exact queries, licences, row counts, cuts, sha256.
Output: /workspace/data/astro/real_starfield/MANIFEST.json (+ copy in out/quality/real_starfield/)."""
import hashlib, json, sys, time
from pathlib import Path

ROOT = Path("/workspace/data/astro/real_starfield")
RAW = ROOT / "raw"
sys.path.insert(0, str(Path(__file__).parent))
from fetch_gaia_esa_chunks import Q as ESA_Q  # noqa: E402


def sha(p, h=None):
    h = h or hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(p):
    with open(p, "rb") as f:
        return sum(1 for _ in f) - 1


def fileinfo(p):
    return {"path": str(p), "bytes": p.stat().st_size, "rows": rows(p), "sha256": sha(p)}


chunks = sorted((RAW / "esa_chunks").glob("*.csv"))
combined = hashlib.sha256()
for c in chunks:
    combined.update(f"{c.name} {sha(c)}\n".encode())
ESA = "https://gea.esac.esa.int/tap-server/tap (ESA Gaia Archive, anonymous async TAP)"
GAIA_LIC = "Gaia data: ESA/Gaia/DPAC, CC BY-NC 3.0 IGO (https://www.cosmos.esa.int/web/gaia-users/license ; commercial use needs ESA authorisation, data.licences@esa.int); cite Gaia Collaboration 2016, 2023 (DR3)"
m = {
    "milestone": "RS1 real observed starfield (no synthesised stars)",
    "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "frame": "Sol-local ICRS cartesian, parsec (x->RA0, z->NCP), epoch J2016.0; f64 in engine",
    "sources": [
        {"name": "Gaia DR3 x Bailer-Jones EDR3 distances", "service": ESA,
         "query": ESA_Q.replace("g.source_id >= {lo} AND g.source_id < {hi} AND ", "") ,
         "chunking": "768 HEALPix level-3 source_id ranges [pix*2^35*4^9, (pix+1)*2^35*4^9), 60 s anonymous job limit; 0 failed chunks",
         "cuts": "parallax > 0.7 mas; phot_g_mean_mag < 13; r_med_photogeo <= 1000 pc (r_med_geo if no photogeo)",
         "files": len(chunks), "bytes": sum(c.stat().st_size for c in chunks), "rows": sum(rows(c) for c in chunks),
         "unique_source_ids": "equal to rows (checked)", "sha256_of_sorted_file_hash_list": combined.hexdigest(),
         "licence": GAIA_LIC + "; Bailer-Jones et al. 2021 AJ 161 147",
         "size_note": "Full Gaia <=1 kpc is ~80M rows (parallax>0.7); G<13 keeps 4.06M (G<14 would be ~7.9M)."},
        {"name": "Hipparcos new reduction (van Leeuwen 2007)", "service": ESA,
         "query": "SELECT * FROM public.hipparcos_newreduction (all columns; text reconstructed from CSV header)",
         **fileinfo(RAW / "hip2_newreduction.csv"), "licence": "ESA Hipparcos; cite van Leeuwen 2007 A&A 474 653"},
        {"name": "Hipparcos main catalogue (ESA 1997) for V, B-V, V-I", "service": ESA,
         "query": "SELECT * FROM public.hipparcos (all columns; text reconstructed from CSV header)",
         **fileinfo(RAW / "hip1_main.csv"), "licence": "ESA 1997, The Hipparcos and Tycho Catalogues, ESA SP-1200"},
        {"name": "Gaia DR3 Hipparcos-2 best neighbour", "service": ESA,
         "query": "SELECT source_id, original_ext_source_id, angular_distance, number_of_neighbours, xm_flag FROM gaiadr3.hipparcos2_best_neighbour",
         **fileinfo(RAW / "hip2_gaiadr3_best_neighbour.csv"), "licence": GAIA_LIC},
        {"name": "Bailer-Jones distances of Hipparcos best neighbours", "service": ESA,
         "query": (RAW / "hip_bj.sql").read_text().strip(),
         **fileinfo(RAW / "hip2_gaiadr3_bj_distances.csv"), "licence": GAIA_LIC + "; Bailer-Jones et al. 2021"},
        {"name": "GCNS (Gaia Catalogue of Nearby Stars, Smart et al. 2021)",
         "service": "https://dc.g-vo.org/tap (GAVO); see /workspace/data/astro/gcns/manifest.json",
         "query": "SELECT * FROM gcns.main", **fileinfo(Path("/workspace/data/astro/gcns/gcns-main.csv")),
         "licence": "GAVO: CC BY-NC 4.0; no commercial redistribution permission assumed"},
        {"name": "Mamajek EEM dwarf colour/Teff table v2022.04.16",
         "url": "http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt",
         **fileinfo(RAW / "mamajek_EEM_dwarf.txt"), "rows_note": "rows = text lines incl. comment header",
         "licence": "no explicit licence in file; citation requested: Pecaut & Mamajek 2013 ApJS 208 9"},
    ],
    "transforms": {
        "G_from_V": "Gaia EDR3 doc 5.5.1 Table 5.7 G-V = f(V-Ic) [-0.01597,-0.02809,-0.2483,0.03656,-0.002939]",
        "G_from_Hp": "Table 5.6 G-Hp = f(B-V) [-0.02392,-0.4069,0.04569,-0.0452]",
        "BpRp_from_VI": "GBP-GRP = f(V-Ic) [-0.03298,1.259,-0.1279,0.01631]",
        "colour": "BP-RP -> Teff (Mamajek) -> Planck x CIE 1931 CMF (Wyman et al. 2013 fit) -> linear sRGB D65, luminance-normalised, 255-entry LUT",
        "brightness": "flux = 10^(-0.4 G_apparent), G_apparent = M + 5 log10(d/10pc), M stored per star; no extinction correction (observed, reddened)",
        "hip_epoch": "J1991.25 -> J2016.0 linear proper motion on the unit vector",
    },
    "ingest_stats": json.loads((ROOT / "ingest_stats.json").read_text()),
    "merge_and_tiles": json.loads((ROOT / "merge_stats.json").read_text()),
    "derived_files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in
                      [ROOT / "norm/gcns.rsrc", ROOT / "norm/gaia.rsrc", ROOT / "norm/hip.rsrc",
                       ROOT / "tiles/rs1.rstile", ROOT / "colour_lut.bin"]},
    "unused": "raw/unused/*.sql: an aborted ESA full-sky query (60 s limit) and a GAVO gedr3dist.litewithdist query whose async job was still queued/executing after 30 min and was abandoned (local poller stopped; no rows used).",
}
for s in m["ingest_stats"].values():
    if isinstance(s, dict) and "inputs" in s and isinstance(s["inputs"], list) and len(s["inputs"]) > 10:
        s["inputs"] = f"{len(s['inputs'])} files raw/esa_chunks/L3_00000..L3_{len(s['inputs'])-1:05d}.csv"
txt = json.dumps(m, indent=1)
(ROOT / "MANIFEST.json").write_text(txt)
out = Path("out/quality/real_starfield")
if out.exists():
    (out / "data_manifest.json").write_text(txt)
print("wrote", ROOT / "MANIFEST.json", len(txt), "bytes")
