# Independent observed catalogue diagnostics

`universe` still defaults to external GCNS. The separate `observed` example is
**not** a merged catalogue, photometric universe scene, or seamless scale handoff.
It has no synthetic fallback. No scientific catalogue or rendered observation is
bundled in git; code licensing does not relicense input data.

## Pinned offline conversion (Python standard library only)

```sh
python3 tools/observed_ingest.py --kind cns5 --root /workspace/data/astro/cns5 \
  --gcns /workspace/data/astro/gcns/gcns-main.csv
python3 tools/observed_ingest.py --kind desi --root /workspace/data/astro/desi
python3 -m unittest discover -s tests/tools
```

Each root contains retained raw VOTable, `manifest.json` with raw SHA256/row pin,
source/schema/terms snapshots and exact queries. Import validates query identity,
TAP QUERY_STATUS including trailing overflow, TABLEDATA shape, field types/units,
retained schema when available, SHA256, count and a 100,000-row/100-MB limit.
Malformed input fails before derived outputs replace existing files. Each derived
file replacement is atomic; `diagnostic-v1.json` contains the binary checksum
for detecting mismatched/torn pairs. Raw originals are never changed by ingest.
The reader validates bounded binary structure, not cryptographic authenticity;
verify the derived checksum against the sidecar when moving/reusing binaries.

`diagnostic-v1.bin`: little-endian `OBSdiag1`, kind u64 (1 CNS5, 2 DESI), raw
count u64, eligible count u64, raw SHA256 bytes (64-byte header); then records
`<QQ7d`: exact ID, optional Gaia ID (0 absent), ICRS XYZ metres, native coordinate
epoch (NaN absent), measured parallax or z, its error, optional **direct** Gaia G
(NaN absent). Runtime records are explicitly typed positions, never `Star`.
Full original quantities/bibcodes/multiplicity/motion/quality fields remain raw.
Output sidecar records schema, source acquisition, selection, assumptions,
checksums, missingness and optional exact-ID GCNS overlap.

For an independently counted future snapshot, explicit network re-acquisition is
available, **to a new root only**, e.g. `--fetch --expected-rows 5909` added to
the CNS5 invocation without `--gcns`. This number is a snapshot pin, not an
assertion all CNS5 releases have that size. The documented TAP GET queries are
in the importer. Fetch requires status/count verification and refuses overflow;
retain official schema/terms snapshots as well before distributing any product.
Default tests and ingest perform no network requests.

## Scientific boundaries

- **CNS5 current GAVO `cns5.main` snapshot:** 5,909 rows, not certified complete
  original 5,931-object publication. The discrepancy remains unresolved. Source:
  https://dc.g-vo.org/CNS5 and https://dc.g-vo.org/tableinfo/cns5.main .
  Native epochs are retained, not propagated or labelled common J2016. Position
  diagnostic accepts positive parallax/error with S/N >=10 and uses disclosed
  reciprocal parallax, **not GCNS Bayesian posterior distances**. All accepted
  records draw equal positional markers, including IR-only objects; no missing
  Gaia G, RGB or physical luminosity is invented. Hipparcos-transformed G and
  deblended photometry remain distinct in raw columns. Catalogue selection and
  errors can place inferred positions beyond nominal 25 pc.
- Optional GCNS overlap uses exact EDR3 IDs across **all** retained GCNS rows.
  It never compares different-epoch positions at tiny angular radii, assumes
  missing RV, promotes nearest neighbours to matches, or merges sources.
  No shared ID means **unresolved**, not a confirmed new unique star. Blended
  components may share Gaia IDs; unique IDs and matched rows are reported
  separately. The earlier exploratory angular candidates are not activation
  criteria. Non-Gaia matching and proper uncertainty-aware epoch propagation
  remain scientific work, not automatically solved by this milestone.
- **DESI DR1 selected observed field:** raw 44,546 rows / 43,295 targets;
  37,871 eligible quality-primary targets. Exact query joins `desi_dr1.zpix`
  to `photometry`, RA [150,160), Dec [-2,2), z [.01,.5), main survey bright/dark,
  Redrock SPECTYPE GALAXY. Raw repeats and flags are retained. Derivative cuts:
  ZCAT_PRIMARY true, ZWARN=0, COADD_FIBERSTATUS=0, finite nonnegative ZERR.
  OBJTYPE is unavailable/not applied; no universal DELTACHI2 cut is invented.
  These flags do not guarantee correct classifications/redshifts.
- DESI coordinates use **model** radial comoving distance, not measured metric
  distances or luminosity distance: flat LCDM H0=70 km/s/Mpc, Omega_m=.3,
  Omega_Lambda=.7, radiation omitted. Composite Simpson integration of
  `(c/H0) integral_0^z dz/sqrt(.3*(1+z)^3+.7)`, not `cz/H0`. This is an explicit
  display assumption, not an inferred DESI cosmology. Peculiar velocities,
  survey selection/gaps, calibration and light-cone vs simultaneous-time
  interpretations remain. Reference epoch 0 is unknown, not J2016 or emission
  time. Actual target coordinates originate from Legacy Surveys DR9; provenance
  remains separate from Gaia-based stellar VACs, which are not ingested here.

## Actual Vulkan positional views

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json
export VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true
cargo run -p render-ash --example observed -- --kind cns5
cargo run -p render-ash --example observed -- --kind desi
```

The example requires active Vulkan **and synchronization validation** and emits
PNG, unexposed RGBA f32 HDR and JSON under ignored `out/quality/observed/<kind>`.
CNS5 has nearby/sky views; DESI has a comoving cone view and the observed sky
field. Camera, epoch caveats, counts, clipping energy and timing accompany each.
PSF weight1 is an equal-marker rasterization convention **not stellar flux**;
there is no inverse-square luminosity path or invented galaxy shape. The existing
HDR pass only rasterizes markers. All loaded positions are examined each view;
nearest visible then exact ID selects at most 65,536 GPU instances. CPU input is
bounded at 100,000 rows; no first-65,536 input truncation. EPIC_GRADE is off.
Lavapipe timing is not a hardware 60fps claim. Shader/SS2/Earth paths are unchanged.

Optional independent scientific checks may use Astropy (and scipy); optional
caption sheets may use Pillow/Matplotlib. None is a runtime/importer dependency.
Prior Matplotlib figures are not Vulkan evidence. Numerical agreement of two
coordinate/distance calculations is **not observational accuracy**.

See CREDITS.md for CNS5/Gaia qualification and DESI's full required acknowledgment.
