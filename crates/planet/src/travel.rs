//! GT1 Sol-neighbourhood continuous flight helpers: spatial selection, coverage, free-flight replay.
//!
//! Catalogue positions remain Sol-local ICRS epoch J2016.0 static medians. The travel camera is a
//! hypothetical observer; whole-root-frame f64 precision limits still apply (see docs/galaxy_travel.md).
use crate::gcns::PARSEC_M;
use crate::universe::{AU, Camera, Selection, Star};
use glam::DVec3;

/// GCNS nominal ~100 pc probabilistic horizon (Smart et al. 2021); not a hard deletion wall.
pub const NOMINAL_SURVEY_PC: f64 = 100.0;

/// Gaia EDR3 source_id for the nearest retained GCNS median (~Proxima Cen direction/distance).
/// Verified present in the external GCNSv001 binary used by this workspace.
pub const DEFAULT_LOOK_SOURCE_ID: u64 = 5_853_498_713_190_525_696;

/// Stop free-flight before unresolved point singularity (1000 AU).
pub const APPROACH_STOP_AU: f64 = 1000.0;

/// Base cruise speed: 0.01 pc / s at gear 0 (displayed units: pc/s).
pub const BASE_SPEED_PC_S: f64 = 0.01;

#[inline]
pub fn default_look_source_id() -> u64 {
    DEFAULT_LOOK_SOURCE_ID
}

#[inline]
pub fn approach_stop_m() -> f64 {
    APPROACH_STOP_AU * AU
}

#[derive(Clone, Debug)]
pub struct Coverage {
    pub distance_pc: f64,
    pub outside_nominal: bool,
    pub warning: &'static str,
}

impl Coverage {
    pub fn at(pos_sol_local_m: DVec3) -> Self {
        let distance_pc = pos_sol_local_m.length() / PARSEC_M;
        let outside_nominal = distance_pc > NOMINAL_SURVEY_PC;
        Self {
            distance_pc,
            outside_nominal,
            warning: if outside_nominal { "outside nominal survey volume; coverage incomplete" } else { "" },
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SelectStats {
    pub input_records: usize,
    pub candidates_considered: usize,
    pub culled: usize,
    pub selected: usize,
    pub dropped_by_cap: usize,
    pub select_ms: f64,
}

/// Uniform grid over Sol-local metres. All input records remain eligible; GPU cap applies only after photometry.
pub struct StarIndex {
    stars: Vec<Star>,
    cell: f64,
    origin: DVec3,
    dims: [i32; 3],
    /// Prefix lengths then flat index lists (CSR).
    offsets: Vec<u32>,
    indices: Vec<u32>,
}

impl StarIndex {
    pub fn build(stars: &[Star], cell_m: f64) -> Self {
        let cell = cell_m.max(1.0);
        let stars = stars.to_vec();
        if stars.is_empty() {
            return Self { stars, cell, origin: DVec3::ZERO, dims: [1, 1, 1], offsets: vec![0], indices: Vec::new() };
        }
        let mut mn = stars[0].pos;
        let mut mx = stars[0].pos;
        for s in &stars {
            mn = mn.min(s.pos);
            mx = mx.max(s.pos);
        }
        // Pad one cell so edge stars stay interior to the grid AABB.
        let origin = mn - DVec3::splat(cell);
        let extent = (mx - origin) + DVec3::splat(cell);
        let dims = [
            ((extent.x / cell).ceil() as i32).max(1),
            ((extent.y / cell).ceil() as i32).max(1),
            ((extent.z / cell).ceil() as i32).max(1),
        ];
        let ncells = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
        let mut counts = vec![0u32; ncells];
        let cell_of = |p: DVec3| -> usize {
            let ix = ((p.x - origin.x) / cell).floor() as i32;
            let iy = ((p.y - origin.y) / cell).floor() as i32;
            let iz = ((p.z - origin.z) / cell).floor() as i32;
            let ix = ix.clamp(0, dims[0] - 1) as usize;
            let iy = iy.clamp(0, dims[1] - 1) as usize;
            let iz = iz.clamp(0, dims[2] - 1) as usize;
            (iz * dims[1] as usize + iy) * dims[0] as usize + ix
        };
        for s in &stars {
            counts[cell_of(s.pos)] += 1;
        }
        let mut offsets = Vec::with_capacity(ncells + 1);
        offsets.push(0);
        for c in &counts {
            offsets.push(offsets.last().unwrap() + c);
        }
        let mut cursor = offsets.clone();
        // Last sentinel unused as write cursor; truncate write region.
        cursor.truncate(ncells);
        let mut indices = vec![0u32; stars.len()];
        for (i, s) in stars.iter().enumerate() {
            let c = cell_of(s.pos);
            let slot = cursor[c] as usize;
            indices[slot] = i as u32;
            cursor[c] += 1;
        }
        Self { stars, cell, origin, dims, offsets, indices }
    }

    pub fn len(&self) -> usize {
        self.stars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stars.is_empty()
    }

    pub fn stars(&self) -> &[Star] {
        &self.stars
    }

    /// Exact photometric agreement with [`Selection::update`] over the full catalogue for the same camera/cap.
    pub fn select(&self, cam: &Camera, limit: usize, out: &mut Selection) -> SelectStats {
        let t0 = std::time::Instant::now();
        let mut ids = Vec::with_capacity(self.stars.len().min(131_072));
        let culled = self.gather_candidate_ids(cam, &mut ids);
        out.update_indexed(&self.stars, &ids, cam, limit);
        let selected = out.sources.len();
        let dropped = out.candidates.saturating_sub(selected);
        SelectStats {
            input_records: self.stars.len(),
            candidates_considered: ids.len(),
            culled,
            selected,
            dropped_by_cap: dropped,
            select_ms: t0.elapsed().as_secs_f64() * 1000.0,
        }
    }

    fn gather_candidate_ids(&self, cam: &Camera, out: &mut Vec<u32>) -> usize {
        out.clear();
        if self.stars.is_empty()
            || !cam.origin.is_finite()
            || !cam.pos.is_finite()
            || !cam.forward.is_finite()
            || !cam.right.is_finite()
            || !cam.up.is_finite()
            || !cam.fov.is_finite()
            || !(0.01..3.0).contains(&cam.fov)
            || cam.width == 0
            || cam.height == 0
        {
            return 0;
        }
        let cam_world = cam.origin + cam.pos;
        let focal = cam.height as f64 / (2.0 * (cam.fov * 0.5).tan());
        let sx = cam.width as f64;
        let sy = cam.height as f64;
        let margin = crate::universe::SUPPORT + 2.0;
        let always_r = (30.0 * PARSEC_M).max(self.cell * 2.0);
        let always_r2 = always_r * always_r;
        let half = DVec3::splat(self.cell * 0.5);
        let diag = half.length();
        let mut seen = vec![false; self.stars.len()];
        for iz in 0..self.dims[2] {
            for iy in 0..self.dims[1] {
                for ix in 0..self.dims[0] {
                    let ci = ((iz * self.dims[1] + iy) * self.dims[0] + ix) as usize;
                    let start = self.offsets[ci] as usize;
                    let end = self.offsets[ci + 1] as usize;
                    if start == end {
                        continue;
                    }
                    let center = self.origin
                        + DVec3::new(
                            (ix as f64 + 0.5) * self.cell,
                            (iy as f64 + 0.5) * self.cell,
                            (iz as f64 + 0.5) * self.cell,
                        );
                    let dist = (center - cam_world).length();
                    let near = dist <= always_r + diag;
                    let keep = near || cell_may_be_visible(cam, cam_world, center, diag, focal, sx, sy, margin);
                    if !keep {
                        continue;
                    }
                    for &idx in &self.indices[start..end] {
                        let i = idx as usize;
                        if !seen[i] {
                            seen[i] = true;
                            out.push(idx);
                        }
                    }
                }
            }
        }
        for (i, s) in self.stars.iter().enumerate() {
            if seen[i] {
                continue;
            }
            if (s.pos - cam_world).length_squared() <= always_r2 {
                seen[i] = true;
                out.push(i as u32);
            }
        }
        self.stars.len().saturating_sub(out.len())
    }
}

#[allow(clippy::too_many_arguments)]
fn cell_may_be_visible(
    cam: &Camera,
    cam_world: DVec3,
    center: DVec3,
    radius: f64,
    focal: f64,
    sx: f64,
    sy: f64,
    margin: f64,
) -> bool {
    // Conservative: test expanded sphere samples (center ± radius along axes + diagonals).
    let offsets = [
        DVec3::ZERO,
        DVec3::X * radius,
        DVec3::NEG_X * radius,
        DVec3::Y * radius,
        DVec3::NEG_Y * radius,
        DVec3::Z * radius,
        DVec3::NEG_Z * radius,
        DVec3::ONE.normalize() * radius,
        DVec3::new(1.0, 1.0, -1.0).normalize() * radius,
        DVec3::new(1.0, -1.0, 1.0).normalize() * radius,
        DVec3::new(-1.0, 1.0, 1.0).normalize() * radius,
    ];
    for o in offsets {
        let p = center + o;
        let r = p - cam_world;
        let z = r.dot(cam.forward);
        if z <= 0.0 {
            // Behind camera: still keep if sphere crosses the near plane.
            if r.length() < radius * 2.0 {
                return true;
            }
            continue;
        }
        let u = sx * 0.5 + r.dot(cam.right) / z * focal;
        let v = sy * 0.5 - r.dot(cam.up) / z * focal;
        if u.is_finite() && v.is_finite() && u >= -margin && v >= -margin && u <= sx + margin && v <= sy + margin {
            return true;
        }
        // Angular size of cell may straddle the screen even if center sample is out.
        let ang = (radius / z).atan();
        let pix = ang * focal;
        if u + pix >= -margin && u - pix <= sx + margin && v + pix >= -margin && v - pix <= sy + margin {
            return true;
        }
    }
    false
}

#[derive(Clone, Copy, Debug)]
pub struct InputSample {
    pub dt: f64,
    pub forward: f32,
    pub right: f32,
    pub up: f32,
    pub yaw_delta: f64,
    pub pitch_delta: f64,
    /// Speed gear offset; gear 0 => BASE_SPEED_PC_S. Clamped to [-3, 8].
    pub speed_gear: i8,
    pub teleport: Option<DVec3>,
}

#[derive(Clone, Debug)]
pub struct FlightState {
    pub origin: DVec3,
    pub pos: DVec3,
    pub yaw: f64,
    pub pitch: f64,
    approach: Option<DVec3>,
}

impl FlightState {
    pub fn sol_looking_at(target_sol_local: DVec3) -> Self {
        let pos = DVec3::ZERO;
        let dir = (target_sol_local - pos).normalize_or_zero();
        // Use look-at convention matching Camera: build yaw/pitch from forward with Z-up.
        let forward = if dir.length_squared() < 1e-30 { DVec3::X } else { dir };
        let pitch = forward.z.asin().clamp(-1.5, 1.5);
        let yaw = forward.y.atan2(forward.x);
        Self { origin: DVec3::ZERO, pos, yaw, pitch, approach: Some(target_sol_local) }
    }

    pub fn set_approach_limit(&mut self, target: Option<DVec3>) {
        self.approach = target;
    }

    pub fn speed_m_s(gear: i8) -> f64 {
        let g = gear.clamp(-3, 8) as i32;
        BASE_SPEED_PC_S * PARSEC_M * 2f64.powi(g)
    }

    pub fn integrate(&mut self, s: &InputSample) {
        if let Some(p) = s.teleport
            && p.is_finite()
        {
            self.pos = p;
        }
        if !s.dt.is_finite() || s.dt <= 0.0 || s.dt > 1.0 {
            return;
        }
        if !s.yaw_delta.is_finite() || !s.pitch_delta.is_finite() {
            return;
        }
        if !s.forward.is_finite() || !s.right.is_finite() || !s.up.is_finite() {
            return;
        }
        self.yaw = wrap_angle(self.yaw + s.yaw_delta);
        self.pitch = (self.pitch + s.pitch_delta).clamp(-1.5, 1.5);
        let (f, r, u) = self.basis();
        let speed = Self::speed_m_s(s.speed_gear);
        let mut delta = (f * s.forward as f64 + r * s.right as f64 + u * s.up as f64) * speed * s.dt;
        if !delta.is_finite() {
            return;
        }
        let mut next = self.pos + delta;
        if let Some(target) = self.approach {
            let stop = approach_stop_m();
            let to = next - target;
            let d = to.length();
            if d.is_finite() && d < stop {
                // Push back onto the stop sphere along the approach vector.
                if d < 1e-30 {
                    next = target - f * stop;
                } else {
                    next = target + to * (stop / d);
                }
                delta = next - self.pos;
            }
        }
        if next.is_finite() {
            self.pos = next;
        }
        let _ = delta;
    }

    pub fn basis(&self) -> (DVec3, DVec3, DVec3) {
        let cp = self.pitch.cos();
        let forward = DVec3::new(self.yaw.cos() * cp, self.yaw.sin() * cp, self.pitch.sin()).normalize_or_zero();
        let world_up = DVec3::Z;
        // Right-handed travel basis matching Camera::look_at (right = up × forward).
        let right = world_up.cross(forward).normalize_or_zero();
        let up = forward.cross(right).normalize_or_zero();
        (forward, right, up)
    }

    pub fn camera(&self, width: u32, height: u32) -> Option<Camera> {
        let (forward, _right, up) = self.basis();
        if !self.pos.is_finite() || !forward.is_finite() || forward.length_squared() < 1e-30 {
            return None;
        }
        let mut c = Camera::look_at(self.pos, self.pos + forward, up, 60f64.to_radians(), width, height)?;
        c.origin = self.origin;
        Some(c)
    }
}

fn wrap_angle(a: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let mut x = a % tau;
    if x > std::f64::consts::PI {
        x -= tau;
    } else if x < -std::f64::consts::PI {
        x += tau;
    }
    x
}

/// Interactive key latch. Cleared on focus loss so movement cannot stick after alt-tab.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FlightKeys {
    pub w: bool,
    pub s: bool,
    pub a: bool,
    pub d: bool,
    pub r: bool,
    pub f: bool,
    pub shift: bool,
    pub ctrl: bool,
}

impl FlightKeys {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn on_focus(&mut self, focused: bool) {
        if !focused {
            self.clear();
        }
    }

    pub fn speed_gear(&self) -> i8 {
        if self.shift {
            1
        } else if self.ctrl {
            -1
        } else {
            0
        }
    }

    pub fn to_axes(&self) -> (f32, f32, f32) {
        (
            (self.w as i8 - self.s as i8) as f32,
            (self.d as i8 - self.a as i8) as f32,
            (self.r as i8 - self.f as i8) as f32,
        )
    }
}

/// One-line status for window title (not an in-frame HUD).
pub fn status_title(
    pos_sol_local_m: glam::DVec3,
    yaw: f64,
    pitch: f64,
    speed_pc_s: f64,
    selected: usize,
    grabbed: bool,
    focused: bool,
) -> String {
    let cov = Coverage::at(pos_sol_local_m);
    let grab = if grabbed { "grab" } else { "free" };
    let focus = if focused { "focus" } else { "unfocus" };
    let warn = if cov.outside_nominal { " | OUTSIDE nominal 100pc survey" } else { "" };
    format!(
        "GT1 GCNS | xyz_pc=({:.3},{:.3},{:.3}) | yaw={:.2} pitch={:.2} | {:.4} pc/s | sel={} | {}/{}{}",
        pos_sol_local_m.x / PARSEC_M,
        pos_sol_local_m.y / PARSEC_M,
        pos_sol_local_m.z / PARSEC_M,
        yaw,
        pitch,
        speed_pc_s,
        selected,
        grab,
        focus,
        warn
    )
}
