//! Quality-pass density fields (metres, f64, pure): [`Alpine`] (km-scale eroded mountains),
//! [`CaveNetwork`] (tunnels, chambers, speleothems) and [`Crease`] (a test field for the
//! sawtooth root cause). All implement LOD-aware band-limiting via `density_lod`.
use crate::DensityField;
use crate::noise::{fbm3, fbm3_wide, octave_weight, octave_weight_wide, perlin2, perlin2d, perlin3, rand2, soft_abs};
use crate::ground::Ground;
use std::cell::RefCell;
use std::sync::Arc;

fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Crease half-width (metres) used when sampling at cell size `cell_m`: never sharper than
/// 3 cells, so marching cubes can represent the crest without cell-frequency zigzag
/// (chord error ~ s^2 / (8 r) with crest radius r = width / slope).
#[inline]
pub fn crease_width(cell_m: f64, min_m: f64) -> f64 {
    (3.0 * cell_m).max(min_m)
}

/// A straight ridge `y = h0 - slope*|u|` along a direction rotated `angle` from +x (C0 crease),
/// optionally softened by [`crease_width`] when sampled with `density_lod`.
pub struct Crease {
    pub h0: f64,
    pub slope: f64,
    pub angle: f64,
    pub soften: bool,
}

impl Crease {
    pub fn height(&self, x: f64, z: f64, cell_m: f64) -> f64 {
        let (s, c) = self.angle.sin_cos();
        let u = -s * x + c * z; // distance across the crest line
        let e = if self.soften { crease_width(cell_m, 0.0) } else { 0.0 };
        self.h0 - self.slope * soft_abs(u, e)
    }
}

impl DensityField for Crease {
    fn density(&self, p: [f64; 3]) -> f64 {
        p[1] - self.height(p[0], p[2], 0.0)
    }
    fn density_lod(&self, p: [f64; 3], cell_m: f64) -> f64 {
        p[1] - self.height(p[0], p[2], cell_m)
    }
}

const CACHE: usize = 1 << 13;
type Col = (u64, u64, u64, [f64; 4]);
thread_local! {
    static COLS: RefCell<Vec<Col>> = RefCell::new(vec![(u64::MAX, u64::MAX, u64::MAX, [0.0; 4]); CACHE]);
}

const BASE_M: f64 = 150.0;
const LOW_M: f64 = 260.0;
const UPLIFT_M: f64 = 700.0;
const RELIEF_M: f64 = 2300.0;
/// |perlin2| bound used for `low` (gradient noise with the 1.41 scale stays within +-1.5).
const LOW_MAX: f64 = 0.28 * 1.5;
/// Sum of the gully octave amplitudes (each term is in [-1, 1]).
/// Gully octaves (wavelength m, depth m): 420 m and 170 m spur/gully systems shape whole faces.
const GULLY_OCTAVES: [(f64, f64); 6] = [(420.0, 24.0), (170.0, 10.0), (64.0, 3.5), (27.0, 1.4), (11.0, 0.6), (4.6, 0.25)];
const GULLY_MAX: f64 = {
    let mut s = 0.0;
    let mut i = 0;
    while i < GULLY_OCTAVES.len() {
        s += GULLY_OCTAVES[i].1;
        i += 1;
    }
    s
};
/// Outward bound of the 3D overhang term (bulge 9 * fbm(<=2) + ledge 1.6 + rough 1.3 * 2
/// + vertical joint ribs 2.4 * 2 + outcrop block lift 11, joint cracks 2.6, block steps 1.8).
const OVERHANG_MAX: f64 = 9.0 * 2.0 + 1.6 + 1.3 * 2.0 + 2.4 * 2.0 + TOR_BLOCK + TOR_CRACK + TOR_STEP;
/// Jointed outcrops (tors) below the treeline: column lift, joint-crack depth and per-block
/// step amplitude (m).
const TOR_LIFT: f64 = 3.0;
/// Steep-sided 3D block lift of the outcrop core (in the 3D term: arbitrary steepness is safe
/// because `may_cross` adds OVERHANG_MAX independent of the slope bound).
const TOR_BLOCK: f64 = 11.0;
/// Seed salt of the outcrop mask (art-directed: of salts 1700..1900, one of the two that put
/// an outcrop in the quality-scene foreground; coverage ~1.4 % of the terrain).
const TOR_SALT: u64 = 1876;
const TOR_CRACK: f64 = 2.6;
const TOR_STEP: f64 = 1.8;
/// Outcrop joint-frame domain size and boundary blend band (m), see `tor_parts`.
const TOR_DOM: f64 = 500.0;
const TOR_DOM_BAND: f64 = 40.0;
/// Crag octaves (wavelength m, height m) for the jagged high ridgelines: ridged teeth that are
/// broken into pinnacles and notches. Each term is in [0, 1], so the crags add [0, CRAG_MAX].
const CRAG_OCTAVES: [(f64, f64); 5] = [(430.0, 95.0), (180.0, 46.0), (75.0, 18.0), (31.0, 7.0), (13.0, 2.6)];
const CRAG_MAX: f64 = 95.0 + 46.0 + 18.0 + 7.0 + 2.6;
/// Lipschitz bound on the band-limited column height (m/m), measured worst ~2.3 for the
/// ridged base (0.1 m central differences over 18 x 18 km) plus <= ~2.4 from the gullies,
/// plus the crags; `q6` measures <= 4.4 m/m (with crags, broad-slope gullies) and asserts a
/// 1.5x margin.
const SLOPE_MAX: f64 = 7.0;

/// Outcrop material info at a point: `rock` weight in [0, 1], `crack` (1 on a joint plane,
/// 0 beyond 0.45 m).
#[derive(Clone, Copy, Debug)]
pub struct TorInfo {
    pub rock: f64,
    pub crack: f64,
}

struct TorParts {
    delta: f64,
    info: TorInfo,
}

/// Km-scale alpine terrain: domain-warped ridged multifractal (soft creases, band-limited),
/// slope-aligned gully erosion, strata ledges and overhangs on steep faces.
///
/// Earth round 5: optionally *grounded* on real elevation ([`crate::ground::Ground`]): the base
/// height blends to the DEM outside the ground's procedural core; everything else (gullies,
/// crags, tors, overhangs) runs on top. `Alpine::new(seed)` is the ungrounded field.
#[derive(Clone)]
pub struct Alpine {
    pub seed: u64,
    pub ground: Option<Arc<Ground>>,
}

/// Lipschitz bound used by `may_cross` for a grounded field (real cliffs plus the core blend).
const SLOPE_MAX_GROUNDED: f64 = 12.0;

impl Alpine {
    pub fn new(seed: u64) -> Self {
        Self { seed, ground: None }
    }

    pub fn grounded(seed: u64, ground: Arc<Ground>) -> Self {
        Self { seed, ground: Some(ground) }
    }

    /// DEM weight at (x, z) (0 when ungrounded).
    pub fn ground_weight(&self, x: f64, z: f64) -> f64 {
        self.ground.as_ref().map_or(0.0, |g| g.weight(x, z))
    }

    fn base(&self, x: f64, z: f64, cell: f64) -> f64 {
        let Some(g) = &self.ground else {
            return self.base_alpine(x, z, cell);
        };
        let w = g.weight(x, z);
        if w <= 0.0 {
            return self.base_alpine(x, z, cell);
        }
        let d = g.height(x, z, cell);
        if w >= 1.0 { d } else { self.base_alpine(x, z, cell) + (d - self.base_alpine(x, z, cell)) * w }
    }

    fn base_alpine(&self, x: f64, z: f64, cell: f64) -> f64 {
        let sd = self.seed;
        let warp = |lam: f64, s: u64| {
            let mut v = 0.0;
            let (mut a, mut l) = (1.0, lam);
            for o in 0..3 {
                v += a * octave_weight_wide(l, cell) * perlin2(x / l + 3.1 * o as f64, z / l - 1.7 * o as f64, s + o);
                a *= 0.45;
                l *= 0.5;
            }
            v
        };
        let u = x + 1100.0 * warp(6000.0, sd + 11);
        let v = z + 1100.0 * warp(6000.0, sd + 23);
        // massif mask: where the big ranges are (low frequency), so there are real lowlands too
        let mass = smoothstep(-0.6, 0.6, perlin2(x / 14000.0, z / 14000.0, sd + 5) + 0.4 * perlin2(x / 5000.0, z / 5000.0, sd + 6));
        // ridged multifractal (Musgrave) with derivative dampening (slopes erode: detail is
        // suppressed where the accumulated gradient is large, valley floors stay smooth).
        // Creases softened to >= 3 cells at the sampled LOD (sawtooth fix).
        let (mut sum, mut amp, mut w, mut lam) = (0.0, 1.0, 1.0, 5200.0);
        // accumulated slope magnitude (continuous: |d|n|/dn| does not flip across creases, unlike
        // the signed gradient, which made the damping jump and the terrain discontinuous)
        let mut gm = 0.0f64;
        let mut norm = 0.0;
        for o in 0..11u64 {
            let wt = octave_weight_wide(lam, cell);
            norm += amp;
            if wt <= 0.0 {
                break;
            }
            let (s, c) = (0.61 * o as f64 + 0.3).sin_cos();
            // first octaves mildly stretched: ranges have a grain, but no long flat crests
            let st = if o < 2 { 0.8 } else { 1.0 };
            let (ru, rv) = ((c * u - s * v) * st, s * u + c * v);
            let (n, nx, nz) = perlin2d(ru / lam + 0.37 * o as f64, rv / lam, sd + 100 + o);
            let e = crease_width(cell, 1.5) * 2.0 / lam;
            let r = 1.0 - soft_abs(n, e);
            // Ridged noise puts every crest of an octave at the same height (r = 1 along the
            // whole zero contour of n): with the 5 km octave dominant that made a long
            // flat-topped wall. Independent non-ridged crest modulation on the big octaves turns
            // each crest line into peaks and saddles (stays <= 1, so the bound holds).
            let crest = if o < 3 {
                let m = perlin2(u / (2600.0 * 0.6f64.powi(o as i32)) + 5.3, v / (2600.0 * 0.6f64.powi(o as i32)), sd + 300 + o);
                0.3 + 0.7 * smoothstep(-0.55, 0.55, m)
            } else {
                1.0
            };
            let sig = r * r * w * crest;
            // damping from the slope of *previous* octaves only, measured by the smooth flank
            // magnitude 2 r w |grad n| / lam (ignores the crease kink, so it does not swing
            // across the crease width and raise crest "spines")
            let damp = 1.0 / (1.0 + 0.15 * (1950.0 * gm).powi(2));
            sum += sig * amp * wt * damp;
            gm += amp * wt * 2.0 * r * w * (nx * nx * st * st + nz * nz).sqrt() / lam;
            w = (sig * 2.2).clamp(0.0, 1.0);
            amp *= 0.56;
            lam *= 0.5;
        }
        // sum <= norm (r^2 w damp wt <= 1 per octave), so h is in [0, 1]
        let h = (sum / norm.max(1e-9)).clamp(0.0, 1.0);
        // power curve: sharp peaks, broad valleys. The massif mask adds *uplift* and only mildly
        // scales relief: multiplying relief by a mask made a long escarpment whose rim sat at a
        // near-constant height (the "range clipped flat" wall).
        let low = 0.18 * perlin2(x / 3000.0, z / 3000.0, sd + 9) + 0.1 * perlin2(x / 900.0, z / 900.0, sd + 10) * octave_weight_wide(900.0, cell);
        BASE_M + LOW_M * low + UPLIFT_M * mass + RELIEF_M * h.powf(1.6) * (0.55 + 0.45 * mass)
    }

    #[doc(hidden)]
    pub fn base_pub(&self, x: f64, z: f64, cell: f64) -> f64 {
        self.base(x, z, cell)
    }

    /// Slope-aligned gullies (cheap analytic erosion): per octave, a sum of Gaussian-weighted
    /// stripe kernels whose stripes run down the local slope; amplitude scaled by steepness.
    fn gullies(&self, x: f64, z: f64, gx: f64, gz: f64, cell: f64) -> f64 {
        let slope = (gx * gx + gz * gz).sqrt();
        if slope < 1e-6 {
            return 0.0;
        }
        let strength = smoothstep(0.12, 0.7, slope);
        let (dx0, dz0) = (gx / slope, gz / slope);
        let mut h = 0.0;
        for (o, &(lam, amp)) in GULLY_OCTAVES.iter().enumerate() {
            let wt = octave_weight_wide(lam, cell);
            if wt <= 0.0 {
                break;
            }
            let so = self.seed + 900 + 7 * o as u64;
            // domain warp (no lattice regularity) and per-region direction jitter: gullies
            // braid, converge and wander instead of running as parallel corduroy
            let wx = x + 0.45 * lam * perlin2(x / (1.7 * lam), z / (1.7 * lam), so + 1);
            let wz = z + 0.45 * lam * perlin2(x / (1.7 * lam) + 5.2, z / (1.7 * lam) - 3.1, so + 2);
            let rot = 0.55 * perlin2(x / (2.6 * lam), z / (2.6 * lam), so + 3);
            let (sn, cs) = rot.sin_cos();
            let (dx, dz) = (dx0 * cs - dz0 * sn, dx0 * sn + dz0 * cs);
            let (px, pz) = (-dz, dx); // across-slope direction
            // some slopes are deeply incised, others smooth
            let incise = 0.35 + 0.9 * smoothstep(-0.45, 0.55, perlin2(x / (3.3 * lam) - 1.7, z / (3.3 * lam) + 2.9, so + 4));
            let (cx, cz) = ((wx / lam).floor() as i64, (wz / lam).floor() as i64);
            let (mut acc, mut ws) = (0.0, 0.0);
            // compactly supported window of radius 1.5 lam over a 5 x 5 neighbourhood: any
            // kernel outside it is >= 2 lam away (weight exactly 0) and the point's own cell
            // kernel is within sqrt(2) lam, so ws > 0 and the sum is continuous. (The previous
            // Gaussian over 3 x 3 dropped kernels of weight ~0.2 when floor(w / lam) changed:
            // metre-high steps in the terrain, slopes up to 17 m/m against SLOPE_MAX.)
            for j in -2..=2 {
                for i in -2..=2 {
                    let (ix, iz) = (cx + i, cz + j);
                    let ox = (ix as f64 + rand2(ix, iz, so)) * lam;
                    let oz = (iz as f64 + rand2(ix, iz, so + 5)) * lam;
                    let (ddx, ddz) = (wx - ox, wz - oz);
                    let q = (1.0 - (ddx * ddx + ddz * ddz) / (2.25 * lam * lam)).max(0.0);
                    let w = q * q * q;
                    // per-kernel frequency and phase jitter: irregular, braided gullies
                    let fj = 0.8 + 0.5 * rand2(ix, iz, so + 6);
                    let pj = std::f64::consts::TAU * rand2(ix, iz, so + 3);
                    let ph = (ddx * px + ddz * pz) / lam * std::f64::consts::TAU * 1.3 * fj + pj;
                    acc += w * ph.cos();
                    ws += w;
                }
            }
            // V-shaped gullies, rounded spurs; |term| <= 1 so |gullies| <= sum of amplitudes
            let g = acc / ws.max(1e-12);
            h += amp * wt * strength * incise.min(1.0) * (1.0 - 0.5 * (1.0 - g) * (1.0 - g));
        }
        h
    }

    /// Jagged crest detail above ~1100 m: per octave a ridged noise (crease softened to the
    /// sampled LOD like the base) whose crest lines are chopped into teeth by an independent
    /// modulation, so ridgelines read as pinnacles and notches instead of smooth arcs.
    /// Returns a value in [0, CRAG_MAX]; band-limited, so far rings get only the big teeth.
    fn crags(&self, x: f64, z: f64, b: f64, cell: f64) -> f64 {
        let sd = self.seed + 1400;
        let mask = smoothstep(1050.0, 1650.0, b) * (0.55 + 0.45 * smoothstep(-0.4, 0.4, perlin2(x / 2300.0, z / 2300.0, sd)));
        if mask <= 0.0 {
            return 0.0;
        }
        let mut h = 0.0;
        for (o, &(lam, amp)) in CRAG_OCTAVES.iter().enumerate() {
            let wt = octave_weight_wide(lam, cell);
            if wt <= 0.0 {
                break;
            }
            let so = sd + 10 * (o as u64 + 1);
            let (sn, cs) = (1.13 * o as f64 + 0.4).sin_cos();
            let (ru, rv) = (cs * x - sn * z, sn * x + cs * z);
            let n = perlin2(ru / lam, rv / lam, so);
            let e = crease_width(cell, 0.4) * 2.0 / lam;
            let r = (1.0 - soft_abs(n, e)).max(0.0);
            // teeth: the crest is notched by a shorter-wavelength modulation
            let m = smoothstep(-0.2, 0.3, perlin2(ru / (0.45 * lam) + 7.7, rv / (0.45 * lam), so + 1));
            h += amp * wt * r * r * (0.1 + 0.9 * m);
        }
        mask * h
    }

    /// Outcrop (tor) mask in [0, 1]: ~40 m patches of jointed bedrock poking out of the
    /// lower slopes and forest floor, in some regions only. Smooth (Lipschitz ~0.1 /m).
    pub fn tor_mask(&self, x: f64, z: f64) -> f64 {
        let sd = self.seed + TOR_SALT;
        let reg = smoothstep(-0.1, 0.35, perlin2(x / 700.0, z / 700.0, sd));
        if reg <= 0.0 {
            return 0.0;
        }
        let n = perlin2(x / 85.0, z / 85.0, sd + 1) + 0.3 * perlin2(x / 31.0, z / 31.0, sd + 2);
        reg * smoothstep(0.42, 0.68, n)
    }

    /// Jointed outcrop geometry (returned as a distance change: positive carves air).
    /// Two near-orthogonal vertical joint sets (spacing 7.5 / 10 m, strike varying over
    /// hundreds of metres) and dipping bedding (3.4 m) cut V-cracks (width >= 1.5 cells).
    /// The steep 3D core (up to TOR_BLOCK) is decided *per joint block* from the mask at the
    /// block centre, so walls follow joint planes (rectilinear fracture faces, stepped tops),
    /// blended 50/50 with the neighbouring block on each joint plane (continuous, and a crack
    /// never stands above both blocks); where the joints fade (coarse LOD) the core falls back
    /// to the smooth mask. Bounded by TOR_BLOCK + TOR_CRACK + TOR_STEP.
    fn tor(&self, p: [f64; 3], gate: f64, cell: f64) -> f64 {
        self.tor_parts(p, gate, cell).map_or(0.0, |t| t.delta)
    }

    /// Per-point outcrop description for materials (exact, from the same joint lattice as the
    /// geometry): `None` outside outcrops.
    pub fn tor_info(&self, p: [f64; 3], cell: f64) -> Option<TorInfo> {
        let h = self.column(p[0], p[2], cell)[0];
        self.tor_parts(p, smoothstep(1250.0, 1150.0, h), cell).map(|t| t.info)
    }

    fn tor_parts(&self, p: [f64; 3], gate: f64, cell: f64) -> Option<TorParts> {
        let t = self.tor_mask(p[0], p[2]) * gate;
        if t <= 0.0 {
            return None;
        }
        // The joint frame (strike, dip) is fixed per TOR_DOM domain, in coordinates relative to
        // the domain anchor, and blended 50/50 with the neighbouring domain on its boundary
        // (weight 0 from TOR_DOM_BAND inwards, so mostly one frame is evaluated).
        // (Round 10 root cause of the "field of spikes": the frame was a rotation about the
        // *world origin* by an angle varying over 600 m, u = R(th(p)) p, whose derivative has a
        // |p| grad(th) term: ~8.7 km out, the 7.5 / 10 m joints were squeezed to ~1 m (sub-cell)
        // and the dipping bedding tilted by up to ~75 deg, so every metre was a new block with
        // an independent 0..11 m lift: comb teeth on every outcrop far from the origin.)
        let axis = |v: f64| {
            let q = v / TOR_DOM;
            let i = q.floor();
            let fr = q - i;
            let nb = if fr < 0.5 { i - 1.0 } else { i + 1.0 };
            (i, nb, 0.5 * (1.0 - smoothstep(0.0, TOR_DOM_BAND, fr.min(1.0 - fr) * TOR_DOM)))
        };
        let ((ix, nx, ax), (iz, nz, az)) = (axis(p[0]), axis(p[2]));
        let (mut delta, mut rock, mut crack) = (0.0, 0.0, 0.0);
        for (dx, wx) in [(ix, 1.0 - ax), (nx, ax)] {
            for (dz, wz) in [(iz, 1.0 - az), (nz, az)] {
                let wgt = wx * wz;
                if wgt > 0.0 {
                    let f = self.tor_frame(p, t, gate, cell, ((dx + 0.5) * TOR_DOM, (dz + 0.5) * TOR_DOM));
                    delta += wgt * f.delta;
                    rock += wgt * f.info.rock;
                    crack += wgt * f.info.crack;
                }
            }
        }
        Some(TorParts { delta, info: TorInfo { rock, crack } })
    }

    /// The jointed outcrop in one domain's joint frame (anchor `o`), for mask value `t > 0`.
    fn tor_frame(&self, p: [f64; 3], t: f64, gate: f64, cell: f64, o: (f64, f64)) -> TorParts {
        let sd = self.seed + 1750;
        let th = 0.6 + 0.5 * perlin2(o.0 / 600.0, o.1 / 600.0, sd);
        let (sn, cs) = th.sin_cos();
        let (s2, c2) = (th + 1.42).sin_cos();
        let dip = 0.18 * perlin2(o.0 / 400.0 + 3.0, o.1 / 400.0, sd + 1);
        let (lx, lz) = (p[0] - o.0, p[2] - o.1);
        let u = [cs * lx + sn * lz, c2 * lx + s2 * lz, p[1] + dip * (cs * lx + sn * lz)];
        let sp = [7.5, 10.0, 3.4];
        let dep = [1.0, 1.0, 0.55];
        // Joint planes (cracks) lie at integer q = u / sp; a block is the slab between two of them,
        // so its id is floor(q) and changes exactly on a crack plane. Per-block values (core lift,
        // random step) are blended across the nearest vertical plane to the neighbouring block,
        // 50/50 on the plane, so the field is continuous there and a crack never stands above
        // both blocks it separates. The bedding (k = 2) step part is windowed to 0 on every plane.
        // (Round 10 fixes: the id was round(q), switching halfway between cracks; the step
        // window faded with the LOD; and the crack fell back to the *smooth* mask lift, which
        // left thin fins up to ~8 m tall along joint lines between low rim blocks, plus up to
        // 1.3 m ridges around lowered blocks: the "field of spikes" in the round-10 close-up.)
        let (mut crack, mut winr, mut wtv, mut id) = (0.0f64, 1.0f64, 1.0f64, [0i64; 3]);
        let (mut nb, mut a) = ([0i64; 2], [0.0f64; 2]);
        let mut near = 0.0f64;
        let w = crease_width(cell, 0.5) * 0.5;
        for k in 0..3 {
            let wt = octave_weight_wide(2.0 * sp[k], cell);
            let q = u[k] / sp[k];
            id[k] = q.floor() as i64;
            let fr = q - q.floor();
            let dist = fr.min(1.0 - fr) * sp[k];
            let c = 1.0 - smoothstep(0.0, w, dist);
            winr *= 1.0 - c;
            if k < 2 {
                wtv = wtv.min(wt);
                // the nearest plane's neighbour; window < half a block so the side never flips
                // while the weight is non-zero
                nb[k] = if fr < 0.5 { id[k] - 1 } else { id[k] + 1 };
                a[k] = 0.5 * (1.0 - smoothstep(0.0, w.min(0.45 * sp[k]), dist));
            }
            if wt <= 0.0 {
                continue;
            }
            crack = crack.max(dep[k] * wt * c);
            near = near.max(wt * (1.0 - smoothstep(0.0, 0.45, dist)));
        }
        let det = cs * s2 - sn * c2;
        let hw = 0.09 * (1.5 * cell).max(1.0);
        // lower edge clamped to >= 0: with sharp > 0.3 (cells >= ~2.2 m) the lift was already
        // non-zero at mask 0, where the `t <= 0` early return cut it off (rim jump up to ~1.8 m
        // around every outcrop at far LODs)
        let lift = |m: f64, sharp: f64| smoothstep((0.3 - sharp).max(0.0), 0.3 + sharp, m) * (0.6 + 0.4 * smoothstep(0.3, 0.9, m));
        // block (i, j): lift from the mask at its centre (solve the 2 x 2 joint basis), random step
        let blk = |i: i64, j: i64| {
            let (a0, a1) = ((i as f64 + 0.5) * sp[0], (j as f64 + 0.5) * sp[1]);
            let (xc, zc) = ((a0 * s2 - a1 * sn) / det + o.0, (cs * a1 - c2 * a0) / det + o.1);
            let tc = self.tor_mask(xc, zc) * gate;
            (lift(tc, 0.02) * (0.7 + 0.3 * rand2(i, j, sd + 11)), rand2(i, j, sd + 7) - 0.5)
        };
        let (mut block, mut rstep) = (0.0, 0.0);
        for (di, wi) in [(id[0], 1.0 - a[0]), (nb[0], a[0])] {
            for (dj, wj) in [(id[1], 1.0 - a[1]), (nb[1], a[1])] {
                if wi * wj > 0.0 {
                    let (bl, r) = blk(di, dj);
                    block += wi * wj * bl;
                    rstep += wi * wj * r;
                }
            }
        }
        // the smooth fallback (coarse LODs, where the joint lattice fades) would be an even
        // TOR_BLOCK plateau, i.e. flat-topped boxes at mid distance: give it uneven tops
        // (<= 1, so the bound holds), the 13 m term only where the LOD resolves it
        let uneven = (perlin2(p[0] / 31.0, p[2] / 31.0, sd + 13) + 0.6 * octave_weight_wide(26.0, cell) * perlin2(p[0] / 13.0, p[2] / 13.0, sd + 14)).clamp(-1.0, 1.0);
        let smooth = lift(t, hw) * (0.74 + 0.26 * uneven);
        // blocks end with the mask: a block whose centre is inside the outcrop still had its
        // full lift where the mask (and the `t <= 0` early return) reached 0 (rim jump)
        let block = block * smoothstep(0.0, 0.1, t);
        let cw = octave_weight_wide(85.0, cell) * (smooth + wtv * (block - smooth));
        let core = TOR_BLOCK * cw;
        // per-block random steps; bedding gives stacked slabs
        let r = rstep + 0.4 * (rand2(id[0] ^ id[2], id[1], sd + 9) - 0.5) * winr;
        let joints = t.max(smooth) * (TOR_CRACK * crack + TOR_STEP * 2.0 / 1.4 * r * octave_weight_wide(15.0, cell)).clamp(-TOR_STEP, TOR_CRACK + TOR_STEP);
        TorParts { delta: joints - core, info: TorInfo { rock: (cw / 0.6).clamp(0.0, 1.0).max(t.min(1.0) * 0.5), crack: near } }
    }

    #[doc(hidden)]
    pub fn tor_pub(&self, p: [f64; 3], cell: f64) -> f64 {
        let h = self.column(p[0], p[2], cell)[0];
        self.tor(p, smoothstep(1250.0, 1150.0, h), cell)
    }

    #[doc(hidden)]
    pub fn crags_pub(&self, x: f64, z: f64, cell: f64) -> f64 {
        self.crags(x, z, self.base(x, z, cell), cell)
    }

    /// (height, dH/dx, dH/dz, steepness) of the column, band-limited for `cell`.
    pub fn column(&self, x: f64, z: f64, cell: f64) -> [f64; 4] {
        // grounded and ungrounded fields must not share cache entries
        let salt = if self.ground.is_some() { 0x6772_6f75_6e64_0000 } else { 0 };
        let key = (x.to_bits(), z.to_bits(), cell.to_bits() ^ salt);
        let slot = ((key.0.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ key.1.wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ key.2) >> 51) as usize & (CACHE - 1);
        if let Some(v) = COLS.with(|c| {
            let c = c.borrow();
            let e = &c[slot];
            (e.0 == key.0 && e.1 == key.1 && e.2 == key.2).then_some(e.3)
        }) {
            return v;
        }
        let h = 3.0f64.max(cell);
        let b = self.base(x, z, cell);
        let gx = (self.base(x + h, z, cell) - self.base(x - h, z, cell)) / (2.0 * h);
        let gz = (self.base(x, z + h, cell) - self.base(x, z - h, cell)) / (2.0 * h);
        // Gully stripes follow the *broad* slope: base filtered at >= 24 m and differenced over
        // +-24 m, so its crease is rounded over >= 72 m. With the local gradient the stripe
        // direction flipped across every crest within a few metres (20 m deep slots on the
        // ridgelines at fine LODs, column slopes up to 27 m/m against SLOPE_MAX).
        let (bc, bh) = (cell.max(24.0), 24.0);
        let bgx = (self.base(x + bh, z, bc) - self.base(x - bh, z, bc)) / (2.0 * bh);
        let bgz = (self.base(x, z + bh, bc) - self.base(x, z - bh, bc)) / (2.0 * bh);
        let tor = self.tor_mask(x, z) * smoothstep(1250.0, 1150.0, b);
        // on real relief the crag teeth stay on steep ground (not on glaciers and plateaus)
        let crag_k = match &self.ground {
            Some(g) => 1.0 - g.weight(x, z) * (1.0 - 0.5 * smoothstep(0.35, 0.9, bgx.hypot(bgz))),
            None => 1.0,
        };
        let hh = b + self.gullies(x, z, bgx, bgz, cell) + crag_k * self.crags(x, z, b, cell) + TOR_LIFT * tor;
        let steep = smoothstep(0.75, 1.5, (gx * gx + gz * gz).sqrt());
        let v = [hh, gx, gz, steep];
        COLS.with(|c| c.borrow_mut()[slot] = (key.0, key.1, key.2, v));
        v
    }

    pub fn height(&self, x: f64, z: f64, cell: f64) -> f64 {
        self.column(x, z, cell)[0]
    }

    /// Rigorous (lo, hi) bounds of the solid surface height anywhere, any filter level:
    /// meshed regions must span them vertically or peaks get cut flat at the region top.
    pub fn height_bounds(&self) -> (f64, f64) {
        let (lo, hi) = (BASE_M - LOW_M * LOW_MAX - GULLY_MAX - OVERHANG_MAX, BASE_M + LOW_M * LOW_MAX + UPLIFT_M + RELIEF_M + GULLY_MAX + CRAG_MAX + TOR_LIFT + OVERHANG_MAX);
        match &self.ground {
            Some(g) => (lo.min(g.h_min - GULLY_MAX - OVERHANG_MAX), hi.max(g.h_max + GULLY_MAX + CRAG_MAX + TOR_LIFT + OVERHANG_MAX)),
            None => (lo, hi),
        }
    }

    /// Conservative test whether the surface can cross the world box `lo..hi` (metres),
    /// sampled at filters in `[cell, 2 cell]` (the smooth filter's range inside one LOD shell):
    /// samples the column height at both ends on a grid over the footprint and
    /// widens the range by the Lipschitz bound times the half-diagonal plus the 3D overhang
    /// bound. `false` means the box is provably all air or all solid (safe to skip).
    pub fn may_cross(&self, lo: [f64; 3], hi: [f64; 3], cell: f64) -> bool {
        let k = 8;
        let (mut hmin, mut hmax) = (f64::MAX, f64::MIN);
        for j in 0..=k {
            for i in 0..=k {
                let x = lo[0] + (hi[0] - lo[0]) * i as f64 / k as f64;
                let z = lo[2] + (hi[2] - lo[2]) * j as f64 / k as f64;
                for c in [cell, 2.0 * cell] {
                    let h = self.height(x, z, c);
                    hmin = hmin.min(h);
                    hmax = hmax.max(h);
                }
            }
        }
        let sp = ((hi[0] - lo[0]) / k as f64).hypot((hi[2] - lo[2]) / k as f64) * 0.5;
        let m = if self.ground.is_some() { SLOPE_MAX_GROUNDED } else { SLOPE_MAX } * sp + OVERHANG_MAX + 2.0;
        !(lo[1] > hmax + m || hi[1] < hmin - m)
    }

    fn eval(&self, p: [f64; 3], cell: f64) -> f64 {
        let [h, gx, gz, steep] = self.column(p[0], p[2], cell);
        let mut d = (p[1] - h) / (1.0 + gx * gx + gz * gz).sqrt();
        if steep > 0.0 && d.abs() < 60.0 {
            // overhangs / bulges and strata ledges on steep rock
            let bulge = 9.0 * fbm3_wide(p, 46.0, 4, cell, self.seed + 300);
            let strata_y = p[1] / 7.0 + 0.8 * perlin2(p[0] / 310.0, p[2] / 310.0, self.seed + 7);
            let f = strata_y - strata_y.floor();
            // the ledge underside drops back to 0 over >= 0.42 m (>= 1.5 cells): the bare
            // sawtooth was a 1.6 m density jump at every stratum boundary
            let e = (1.5 * cell / 7.0).clamp(0.06, 0.35);
            let ledge = 1.6 * octave_weight_wide(7.0, cell) * smoothstep(0.55, 0.95, f) * (1.0 - smoothstep(1.0 - e, 1.0, f));
            let rough = 1.3 * fbm3_wide(p, 9.0, 3, cell, self.seed + 333);
            // vertical joints: ribs and chimneys (stretched ridged noise, crease softened)
            let jw = octave_weight_wide(14.0, cell);
            let joint = if jw > 0.0 {
                let n = perlin3([p[0] / 14.0, p[1] / 55.0, p[2] / 14.0], self.seed + 350);
                2.4 * jw * (1.0 - 2.0 * soft_abs(n, crease_width(cell, 0.3) * 2.0 / 14.0)).max(-1.0)
            } else {
                0.0
            };
            d -= steep * (bulge + ledge + rough + joint);
        }
        if d.abs() < 18.0 {
            d += self.tor(p, smoothstep(1250.0, 1150.0, h), cell);
        }
        d
    }
}

impl DensityField for Alpine {
    fn density(&self, p: [f64; 3]) -> f64 {
        self.eval(p, 0.25)
    }
    fn density_lod(&self, p: [f64; 3], cell_m: f64) -> f64 {
        self.eval(p, cell_m)
    }
}

/// A limestone cave: a meandering main tunnel (spline worm through the camera) opening into
/// a chamber, noise "spaghetti" side worms, rough fluted walls, strata, stalactites and
/// stalagmites. Air is `density > 0`.
pub struct CaveNetwork {
    pub seed: u64,
}

/// Main tunnel centreline at depth `z` (tunnel runs toward -z from the camera at z = 0).
pub fn cave_path(z: f64) -> [f64; 3] {
    let t = -z;
    [2.2 * (t / 9.0).sin() + 0.8 * (t / 3.7).sin(), 1.35 - 0.035 * t.max(0.0) + 0.25 * (t / 5.3).sin(), z]
}
pub const CHAMBER_C: [f64; 3] = [3.0, 2.2, -40.0];
pub const CHAMBER_R: [f64; 3] = [13.0, 6.5, 11.0];
pub const CHAMBER_FLOOR: f64 = -1.6;

fn smin(a: f64, b: f64, k: f64) -> f64 {
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    b + (a - b) * h - k * h * (1.0 - h)
}

impl CaveNetwork {
    fn tunnel(&self, p: [f64; 3], cell: f64) -> f64 {
        // distance to centreline sampled near z (path is nearly straight per metre)
        let mut best = f64::MAX;
        let mut by = 0.0;
        let z0 = p[2].clamp(-34.0, 8.0);
        for k in -6..=6 {
            let z = z0 + k as f64 * 0.5;
            let c = cave_path(z);
            let d = [p[0] - c[0], (p[1] - c[1]) * 1.3, p[2] - z];
            let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if l < best {
                best = l;
                by = c[1];
            }
        }
        let r = 2.1 + 0.45 * perlin2(p[2] / 6.0, 0.5, self.seed + 1) * octave_weight(6.0, cell);
        let t = best - r;
        // flat-ish sediment floor
        let floor = by - 1.05 + 0.08 * perlin2(p[0] / 1.7, p[2] / 1.7, self.seed + 2) * octave_weight(1.7, cell);
        t.max(floor - p[1])
    }

    fn chamber(&self, p: [f64; 3]) -> f64 {
        let q = [0, 1, 2].map(|i| (p[i] - CHAMBER_C[i]) / CHAMBER_R[i]);
        let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
        ((l - 1.0) * 6.5).max(CHAMBER_FLOOR - p[1])
    }

    fn ceiling(&self, x: f64, z: f64) -> Option<(f64, f64)> {
        // (ceiling y, floor y) of the chamber or tunnel at this column
        let q = [(x - CHAMBER_C[0]) / CHAMBER_R[0], (z - CHAMBER_C[2]) / CHAMBER_R[2]];
        let r2 = q[0] * q[0] + q[1] * q[1];
        if r2 < 0.85 {
            return Some((CHAMBER_C[1] + CHAMBER_R[1] * (1.0 - r2).sqrt(), CHAMBER_FLOOR));
        }
        if (-30.0..6.0).contains(&z) {
            let c = cave_path(z);
            if (x - c[0]).abs() < 1.3 {
                return Some((c[1] + 2.1 / 1.3 - 0.1, c[1] - 1.05));
            }
        }
        None
    }

    /// Solid speleothems (negative inside): cones hanging from ceilings / rising from floors.
    fn speleothems(&self, p: [f64; 3], cell: f64) -> f64 {
        let g = 1.6;
        let (cx, cz) = ((p[0] / g).floor() as i64, (p[2] / g).floor() as i64);
        let mut best = f64::MAX;
        for j in -1..=1 {
            for i in -1..=1 {
                let (ix, iz) = (cx + i, cz + j);
                let r0 = rand2(ix, iz, self.seed + 40);
                if r0 < 0.35 {
                    continue;
                }
                let x = (ix as f64 + 0.15 + 0.7 * rand2(ix, iz, self.seed + 41)) * g;
                let z = (iz as f64 + 0.15 + 0.7 * rand2(ix, iz, self.seed + 42)) * g;
                // keep the camera's spot clear (z = 0 on the main tunnel)
                if x.hypot(z) < 3.0 {
                    continue;
                }
                let Some((ceil, floor)) = self.ceiling(x, z) else { continue };
                let dxz = ((p[0] - x).powi(2) + (p[2] - z).powi(2)).sqrt();
                let ang = (p[2] - z).atan2(p[0] - x);
                let big = ceil - floor > 6.0;
                // irregular cross-section and drip rings: radius modulated around and along
                let wob = 1.0
                    + 0.22 * perlin2(ang.cos() * 1.3 + ix as f64 * 3.1, p[1] * 1.7, self.seed + 45) * octave_weight(0.3, cell)
                    + 0.10 * perlin2(ang.sin() * 2.0 + iz as f64, p[1] * 6.0, self.seed + 46) * octave_weight(0.12, cell);
                // stalactite
                let len = (if big { 1.2 + 3.8 * rand2(ix, iz, self.seed + 43) } else { 0.3 + 0.9 * rand2(ix, iz, self.seed + 43) }).max(cell * 3.0);
                let rb = (len * (0.15 + 0.10 * rand2(ix, iz, self.seed + 44))).max(cell * 1.5);
                let tip = ceil - len;
                let t = ((p[1] - tip) / len).clamp(0.0, 1.5);
                let rad = rb * t.powf(1.5) * wob + cell * 0.5;
                let dy = (tip - p[1]).max(0.0);
                best = best.min(if dy > 0.0 { dxz.hypot(dy) - cell * 0.5 } else { (dxz - rad) * 0.8 });
                // stalagmite: fewer, stubby, rounded tops
                if r0 > 0.72 {
                    let len2 = (0.45 * len).max(cell * 3.0);
                    let tip2 = floor + len2;
                    let t2 = ((tip2 - p[1]) / len2).clamp(0.0, 1.5);
                    let rad2 = 1.7 * rb * t2.powf(0.55) * wob + cell * 0.5;
                    let dy2 = (p[1] - tip2).max(0.0);
                    best = best.min(if dy2 > 0.0 { dxz.hypot(dy2) - rb * 0.25 } else { (dxz - rad2) * 0.8 });
                }
            }
        }
        best
    }

    fn eval(&self, p: [f64; 3], cell: f64) -> f64 {
        let sd = self.seed;
        let mut air = smin(self.tunnel(p, cell), self.chamber(p), 2.5);
        // spaghetti side worms (zero-set intersection of two 3D noises), masked to a region
        let lam = 15.0;
        let q = [p[0] / lam, p[1] / lam * 1.6, p[2] / lam];
        let (n1, n2) = (perlin3(q, sd + 10), perlin3([q[0] + 17.3, q[1] - 4.1, q[2] + 9.2], sd + 11));
        let worm = ((n1 * n1 + n2 * n2).sqrt() - 0.085) * lam * 0.55;
        let mask = smoothstep(-0.1, 0.25, perlin3([p[0] / 40.0, 0.3, p[2] / 40.0], sd + 12));
        air = smin(air, worm + (1.0 - mask) * 8.0, 1.5);
        // rough walls: bumps, vertical flowstone fluting, horizontal strata
        let bumps = 0.45 * fbm3(p, 3.2, 5, cell, sd + 20);
        // fractured rock: ridged creases (soft-abs, >= 3 cells) at ~0.7 m
        let fr = perlin3([p[0] / 0.7, p[1] / 0.45, p[2] / 0.7], sd + 23);
        let fracture = 0.07 * octave_weight(0.45, cell) * (1.0 - soft_abs(fr, crate::fields::crease_width(cell, 0.0) / 0.35));
        let flute = 0.16 * octave_weight(0.8, cell) * perlin3([p[0] / 0.8, p[1] / 4.0, p[2] / 0.8], sd + 21);
        let sy = p[1] / 0.75 + 0.6 * perlin2(p[0] / 9.0, p[2] / 9.0, sd + 22);
        let strata = 0.13 * octave_weight(0.75, cell) * smoothstep(0.6, 0.95, sy - sy.floor());
        air += bumps + flute - strata - fracture;
        // speleothems are solid: remove them from the air
        let sp = self.speleothems(p, cell);
        air = air.max(-sp);
        -air
    }
}

impl DensityField for CaveNetwork {
    fn density(&self, p: [f64; 3]) -> f64 {
        self.eval(p, 0.05)
    }
    fn density_lod(&self, p: [f64; 3], cell_m: f64) -> f64 {
        self.eval(p, cell_m)
    }
}
