//! Robustness: the Alpine density must be continuous at every LOD filter, in particular
//! inside the jointed outcrops (tors). A jump in the field makes the mesher emit spikes and
//! torn faces (round-10 close-up: a field of vertical teeth in front of an outcrop ~120 m away,
//! where per-block lifts / steps switched halfway between joint planes, and on planes whose
//! crack window had faded with the LOD; at far LODs the outcrop lift was cut off at its rim;
//! on steep rock the strata ledge was a 1.6 m sawtooth).
//!
//! Probe: a jump survives as the step shrinks, a steep wall does not. Each 10 cm segment of a
//! random walk is bisected 36 times towards the half with the larger change (down to ~1e-12 m);
//! a continuous field (steepest intended tor wall ~11 m over a 0.25 m crack window) is then
//! left with < 1e-9 m, a discontinuity with its full height, so > 5 mm is a jump. (A fixed tiny
//! step would miss jumps: they live on planes.) Only segments within 4 m of the surface count.
use voxel::DensityField;
use voxel::fields::Alpine;

fn rnd(i: u64) -> f64 {
    let mut x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 29;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// (jumps, probes, worst jump m, where) over random near-surface probes of outcrop / steep-rock columns in a
/// `half`-m square around (cx, cz), for one LOD cell size. Each probe then walks
/// 32 segments of 10 cm along a random direction, so it crosses joint planes and block boundaries.
fn scan_jumps(f: &Alpine, cx: f64, cz: f64, half: f64, cell: f64, probes: usize) -> (usize, usize, f64, [f64; 3]) {
    let (mut jumps, mut n, mut worst, mut at) = (0usize, 0usize, 0.0f64, [0.0; 3]);
    let mut i = 0u64;
    let mut found = 0;
    while found < probes && i < 400 * probes as u64 {
        i += 1;
        let x = cx + (rnd(i * 7) - 0.5) * 2.0 * half;
        let z = cz + (rnd(i * 7 + 1) - 0.5) * 2.0 * half;
        // outcrop (tor) columns and steep-rock columns (overhang / ledge / joint terms)
        if f.tor_mask(x, z) < 0.05 && f.column(x, z, cell)[3] < 0.2 {
            continue;
        }
        found += 1;
        let h = f.height(x, z, cell);
        let y = h - 4.0 + 18.0 * rnd(i * 7 + 2);
        let (a, b) = (rnd(i * 7 + 3) * std::f64::consts::TAU, rnd(i * 7 + 4) * 2.0 - 1.0);
        let s = (1.0 - b * b).sqrt();
        let dir = [s * a.cos(), b, s * a.sin()];
        let at_t = |t: f64| [x + dir[0] * t, y + dir[1] * t, z + dir[2] * t];
        for k in 0..32 {
            let (mut t0, mut t1) = (0.1 * k as f64, 0.1 * (k + 1) as f64);
            let (mut d0, mut d1) = (f.density_lod(at_t(t0), cell), f.density_lod(at_t(t1), cell));
            if d0.abs().min(d1.abs()) > 4.0 {
                continue;
            }
            n += 1;
            for _ in 0..36 {
                let tm = 0.5 * (t0 + t1);
                let dm = f.density_lod(at_t(tm), cell);
                if (dm - d0).abs() >= (d1 - dm).abs() {
                    (t1, d1) = (tm, dm);
                } else {
                    (t0, d0) = (tm, dm);
                }
            }
            let j = (d1 - d0).abs();
            if j > 5e-3 {
                jumps += 1;
                if std::env::var_os("TOR_DIAG").is_some() {
                    let (p0, p1) = (at_t(t0), at_t(t1));
                    eprintln!(
                        "  jump {j:.3} cell {cell} at {p0:?}: d {d0:.4}->{d1:.4} h {:.4}->{:.4} tor {:.4}->{:.4}",
                        f.height(p0[0], p0[2], cell),
                        f.height(p1[0], p1[2], cell),
                        f.tor_pub(p0, cell),
                        f.tor_pub(p1, cell)
                    );
                }
            }
            if j > worst {
                worst = j;
                at = at_t(t0);
            }
        }
    }
    (jumps, n, worst, at)
}

#[test]
fn alpine_density_is_continuous_in_outcrops_and_steep_rock_at_every_lod() {
    let f = Alpine::new(7);
    let mut total = 0;
    for cell in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0] {
        // the close-up camera's surroundings (the reported outcrop) and a wider world square
        for (cx, cz, half) in [(7799.0, -3614.0, 800.0), (0.0, 0.0, 12000.0)] {
            let (j, n, w, at) = scan_jumps(&f, cx, cz, half, cell, 400);
            eprintln!(
                "cell {cell} around ({cx}, {cz}) +-{half}: {j} jumps in {n} near-surface probes, worst {w:.2e} m at {at:?}"
            );
            total += j;
        }
    }
    assert_eq!(total, 0, "density discontinuities in outcrops");
}
