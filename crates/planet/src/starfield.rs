//! Real observed starfield (RS1): catalogue merge/dedupe, the `RSTILE01` octree tile format and
//! distance/brightness-prioritised streaming selection.
//!
//! Every star in a tile file is one row of an observed catalogue (GCNS, Gaia DR3 with Bailer-Jones
//! EDR3 distances, or Hipparcos new reduction); nothing here invents positions, magnitudes or
//! colours. Test fixtures in `tests/starfield.rs` are synthetic and never written to the data dir.
//!
//! Frame: Sol-local right-handed ICRS (+X towards RA 0/Dec 0, +Z north pole), parsecs, catalogue
//! epoch J2016.0 (Hipparcos rows propagated linearly from J1991.25 by the ingest script).
//! Brightness: Gaia G-band (Vega) apparent magnitude as catalogued (no extinction correction); the
//! tile stores the "pseudo absolute" magnitude `M = G - 5 log10(d / 10 pc)` so the apparent
//! magnitude from any observer is `M + 5 log10(d_obs / 10 pc)` (inverse square only).
use crate::gcns::PARSEC_M;
use crate::universe::{Camera, SUPPORT, Source};
use glam::DVec3;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::{self, Read, Write};

pub const SRC_GCNS: u8 = 0;
pub const SRC_GAIA: u8 = 1;
pub const SRC_HIP: u8 = 2;
pub const SRC_NAMES: [&str; 3] = ["gcns", "gaia_dr3_bj", "hipparcos2"];

/// Ingest flags (see tools/real_starfield/ingest.py).
pub mod flag {
    pub const HIP_POS: u32 = 1;
    pub const G_FROM_V: u32 = 2;
    pub const G_FROM_HP: u32 = 4;
    pub const COL_FROM_VI: u32 = 8;
    pub const COL_FROM_BV: u32 = 16;
    pub const EPOCH_PROP: u32 = 32;
    pub const LOW_PLX_SNR: u32 = 64;
    pub const RUWE: u32 = 128;
    pub const GEO_DIST: u32 = 256;
    pub const COL_RANGE: u32 = 512;
    pub const HIP_BEYOND_1KPC: u32 = 1024;
    pub const HIP_DIST_BJ: u32 = 2048;
    /// Set by the merge: Hipparcos row replaced a saturated (G < threshold) Gaia row.
    pub const HIP_REPLACED_GAIA: u32 = 1 << 16;
}

fn invalid(msg: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// One normalised catalogue row (RSRC0001, 56 bytes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawRecord {
    pub id: u64,
    /// Hipparcos rows: Gaia DR3 hipparcos2_best_neighbour source_id (0 = none).
    pub match_gaia: u64,
    pub pos_pc: DVec3,
    pub gmag: f32,
    /// NaN = no measured colour.
    pub bp_rp: f32,
    pub flags: u32,
}

pub fn read_rsrc(r: &mut impl Read) -> io::Result<Vec<RawRecord>> {
    let mut h = [0u8; 16];
    r.read_exact(&mut h)?;
    if &h[..8] != b"RSRC0001" {
        return Err(invalid("bad RSRC magic"));
    }
    let n = u64::from_le_bytes(h[8..16].try_into().unwrap());
    if n > 50_000_000 {
        return Err(invalid("RSRC count too large"));
    }
    let mut out = Vec::with_capacity(n as usize);
    let mut ids = HashSet::with_capacity(n as usize);
    let mut b = [0u8; 56];
    for _ in 0..n {
        r.read_exact(&mut b)?;
        let u = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        let f = |i: usize| f64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        let g = |i: usize| f32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        let rec = RawRecord {
            id: u(0),
            match_gaia: u(8),
            pos_pc: DVec3::new(f(16), f(24), f(32)),
            gmag: g(40),
            bp_rp: g(44),
            flags: u32::from_le_bytes(b[48..52].try_into().unwrap()),
        };
        if rec.id == 0
            || !ids.insert(rec.id)
            || !rec.pos_pc.is_finite()
            || rec.pos_pc.length() < 1e-3
            || !rec.gmag.is_finite()
            || !(-5.0..30.0).contains(&rec.gmag)
            || rec.bp_rp.is_infinite()
        {
            return Err(invalid("invalid RSRC record"));
        }
        out.push(rec);
    }
    let mut extra = [0u8; 1];
    if r.read(&mut extra)? != 0 {
        return Err(invalid("trailing RSRC bytes"));
    }
    Ok(out)
}

pub fn write_rsrc(w: &mut impl Write, recs: &[RawRecord]) -> io::Result<()> {
    w.write_all(b"RSRC0001")?;
    w.write_all(&(recs.len() as u64).to_le_bytes())?;
    for r in recs {
        w.write_all(&r.id.to_le_bytes())?;
        w.write_all(&r.match_gaia.to_le_bytes())?;
        for c in r.pos_pc.to_array() {
            w.write_all(&c.to_le_bytes())?;
        }
        w.write_all(&r.gmag.to_le_bytes())?;
        w.write_all(&r.bp_rp.to_le_bytes())?;
        w.write_all(&r.flags.to_le_bytes())?;
        w.write_all(&0u32.to_le_bytes())?;
    }
    Ok(())
}

/// A merged, deduplicated star ready for tiling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Merged {
    pub src: u8,
    pub id: u64,
    pub pos_pc: DVec3,
    pub gmag: f32,
    pub bp_rp: f32,
    pub flags: u32,
}
impl Merged {
    fn from_raw(src: u8, r: &RawRecord) -> Self {
        Self { src, id: r.id, pos_pc: r.pos_pc, gmag: r.gmag, bp_rp: r.bp_rp, flags: r.flags }
    }
    /// Inverse-square pseudo absolute G magnitude (no extinction term).
    pub fn abs_mag(&self) -> f64 {
        self.gmag as f64 - 5.0 * (self.pos_pc.length() / 10.0).log10()
    }
}

#[derive(Clone, Debug)]
pub struct MergeParams {
    /// GCNS is authoritative inside this Sol distance (non-GCNS Gaia/faint Hip rows dropped).
    pub gcns_radius_pc: f64,
    /// Hipparcos replaces Gaia when the Hipparcos-derived G is brighter than this (Gaia saturation).
    pub hip_saturation_g: f32,
    /// Positional fallback for Hipparcos rows without a usable best-neighbour id.
    pub fallback_radius_arcsec: f64,
    pub fallback_max_dg: f32,
}
impl Default for MergeParams {
    fn default() -> Self {
        Self { gcns_radius_pc: 100.0, hip_saturation_g: 6.0, fallback_radius_arcsec: 2.0, fallback_max_dg: 1.0 }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MergeStats {
    pub gcns_in: usize,
    pub gaia_in: usize,
    pub hip_in: usize,
    pub gaia_dup_of_gcns: usize,
    pub gaia_inside_gcns_volume_not_in_gcns: usize,
    pub hip_dup_of_gcns_by_id: usize,
    pub hip_dup_of_gaia_by_id: usize,
    pub hip_replaced_saturated_gaia: usize,
    pub hip_bright_matched_gaia_outside_sample_kept: usize,
    pub hip_faint_matched_gaia_outside_sample_dropped: usize,
    pub hip_dup_of_gcns_positional: usize,
    pub hip_dup_of_gaia_positional: usize,
    pub hip_replaced_saturated_gaia_positional: usize,
    pub hip_unmatched_faint_inside_gcns_volume_dropped: usize,
    pub hip_unmatched_kept: usize,
    pub out_by_src: [usize; 3],
}

impl MergeStats {
    pub fn to_json(&self) -> String {
        let f = [
            ("gcns_in", self.gcns_in),
            ("gaia_in", self.gaia_in),
            ("hip_in", self.hip_in),
            ("gaia_dup_of_gcns", self.gaia_dup_of_gcns),
            ("gaia_inside_gcns_volume_not_in_gcns", self.gaia_inside_gcns_volume_not_in_gcns),
            ("hip_dup_of_gcns_by_id", self.hip_dup_of_gcns_by_id),
            ("hip_dup_of_gaia_by_id", self.hip_dup_of_gaia_by_id),
            ("hip_replaced_saturated_gaia", self.hip_replaced_saturated_gaia),
            ("hip_bright_matched_gaia_outside_sample_kept", self.hip_bright_matched_gaia_outside_sample_kept),
            ("hip_faint_matched_gaia_outside_sample_dropped", self.hip_faint_matched_gaia_outside_sample_dropped),
            ("hip_dup_of_gcns_positional", self.hip_dup_of_gcns_positional),
            ("hip_dup_of_gaia_positional", self.hip_dup_of_gaia_positional),
            ("hip_replaced_saturated_gaia_positional", self.hip_replaced_saturated_gaia_positional),
            ("hip_unmatched_faint_inside_gcns_volume_dropped", self.hip_unmatched_faint_inside_gcns_volume_dropped),
            ("hip_unmatched_kept", self.hip_unmatched_kept),
        ];
        let body: Vec<String> = f.iter().map(|(k, v)| format!("\"{k}\": {v}")).collect();
        format!(
            "{{{}, \"out_by_src\": {{\"gcns\": {}, \"gaia_dr3_bj\": {}, \"hipparcos2\": {}}}}}",
            body.join(", "),
            self.out_by_src[0],
            self.out_by_src[1],
            self.out_by_src[2]
        )
    }
}

/// Angular hash for the positional fallback (sorted keys; no per-cell allocation).
struct AngIndex {
    cell: f64,
    keys: Vec<(u64, u32)>,
}
fn ang_key(u: DVec3, cell: f64) -> (i64, i64, i64) {
    ((u.x / cell).floor() as i64, (u.y / cell).floor() as i64, (u.z / cell).floor() as i64)
}
fn pack_key(k: (i64, i64, i64)) -> u64 {
    let m = |v: i64| (v + (1 << 20)) as u64 & 0x1f_ffff;
    (m(k.0) << 42) | (m(k.1) << 21) | m(k.2)
}
impl AngIndex {
    fn build(stars: &[Merged], cell: f64) -> Self {
        let mut keys: Vec<(u64, u32)> =
            stars.iter().enumerate().map(|(i, s)| (pack_key(ang_key(s.pos_pc.normalize(), cell)), i as u32)).collect();
        keys.sort_unstable();
        Self { cell, keys }
    }
    fn near(&self, u: DVec3, out: &mut Vec<u32>) {
        out.clear();
        let k = ang_key(u, self.cell);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let key = pack_key((k.0 + dx, k.1 + dy, k.2 + dz));
                    let start = self.keys.partition_point(|e| e.0 < key);
                    out.extend(self.keys[start..].iter().take_while(|e| e.0 == key).map(|e| e.1));
                }
            }
        }
    }
}

/// Deterministic merge. Precedence: GCNS (inside its volume) > Gaia (G >= saturation) > Hipparcos;
/// Hipparcos wins over Gaia only for saturated bright stars, and fills stars Gaia lacks entirely.
pub fn merge(gcns: &[RawRecord], gaia: &[RawRecord], hip: &[RawRecord], p: &MergeParams) -> (Vec<Merged>, MergeStats) {
    let mut st = MergeStats { gcns_in: gcns.len(), gaia_in: gaia.len(), hip_in: hip.len(), ..Default::default() };
    let gcns_ids: HashSet<u64> = gcns.iter().map(|r| r.id).collect();
    let mut out: Vec<Merged> = gcns.iter().map(|r| Merged::from_raw(SRC_GCNS, r)).collect();
    let n_gcns = out.len();
    let mut gaia_slot: HashMap<u64, usize> = HashMap::with_capacity(gaia.len());
    for r in gaia {
        if gcns_ids.contains(&r.id) {
            st.gaia_dup_of_gcns += 1;
        } else if r.pos_pc.length() < p.gcns_radius_pc {
            st.gaia_inside_gcns_volume_not_in_gcns += 1;
        } else {
            gaia_slot.insert(r.id, out.len());
            out.push(Merged::from_raw(SRC_GAIA, r));
        }
    }
    let mut removed = vec![false; out.len()];
    let cell = (p.fallback_radius_arcsec / 3600.0).to_radians();
    let index = AngIndex::build(&out, cell);
    let mut near = Vec::new();
    let mut hip_out = Vec::new();
    for r in hip {
        let h = Merged::from_raw(SRC_HIP, r);
        let bright = h.gmag < p.hip_saturation_g;
        if r.match_gaia != 0 {
            if gcns_ids.contains(&r.match_gaia) {
                st.hip_dup_of_gcns_by_id += 1;
                continue;
            }
            if let Some(&slot) = gaia_slot.get(&r.match_gaia) {
                if bright && !removed[slot] {
                    removed[slot] = true;
                    st.hip_replaced_saturated_gaia += 1;
                    hip_out.push(Merged { flags: h.flags | flag::HIP_REPLACED_GAIA, ..h });
                } else {
                    st.hip_dup_of_gaia_by_id += 1;
                }
                continue;
            }
            if bright {
                st.hip_bright_matched_gaia_outside_sample_kept += 1;
                hip_out.push(h);
            } else {
                st.hip_faint_matched_gaia_outside_sample_dropped += 1;
            }
            continue;
        }
        // Positional fallback: nearest kept GCNS/Gaia row within radius and |dG| limit.
        let u = h.pos_pc.normalize();
        index.near(u, &mut near);
        let best = near
            .iter()
            .copied()
            .filter(|&i| !removed[i as usize])
            .map(|i| (i, out[i as usize].pos_pc.normalize().angle_between(u)))
            .filter(|&(i, a)| a <= cell && (out[i as usize].gmag - h.gmag).abs() <= p.fallback_max_dg)
            .min_by(|a, b| a.1.total_cmp(&b.1).then(out[a.0 as usize].id.cmp(&out[b.0 as usize].id)));
        if let Some((i, _)) = best {
            let i = i as usize;
            if i < n_gcns {
                st.hip_dup_of_gcns_positional += 1;
            } else if bright {
                removed[i] = true;
                st.hip_replaced_saturated_gaia_positional += 1;
                hip_out.push(Merged { flags: h.flags | flag::HIP_REPLACED_GAIA, ..h });
            } else {
                st.hip_dup_of_gaia_positional += 1;
            }
            continue;
        }
        if h.pos_pc.length() < p.gcns_radius_pc && !bright {
            st.hip_unmatched_faint_inside_gcns_volume_dropped += 1;
            continue;
        }
        st.hip_unmatched_kept += 1;
        hip_out.push(h);
    }
    let mut merged: Vec<Merged> =
        out.into_iter().zip(removed).filter(|(_, rm)| !rm).map(|(s, _)| s).chain(hip_out).collect();
    merged.sort_by(|a, b| a.src.cmp(&b.src).then(a.id.cmp(&b.id)));
    for s in &merged {
        st.out_by_src[s.src as usize] += 1;
    }
    (merged, st)
}

// ---------------------------------------------------------------------------------------------
// RSTILE01 file format: little-endian, 64-byte aligned sections, fixed-size records, zero-copy.
// ---------------------------------------------------------------------------------------------

pub const MAGIC: [u8; 8] = *b"RSTILE01";
pub const Q_MAX: f64 = u32::MAX as f64;
pub const MAG_MIN: f32 = -16.0;
pub const MAG_SCALE: f32 = 40.0 / 65535.0;
pub const NO_COLOUR: u8 = 255;
/// Fraction of the root size by which the root cube is shifted (not a dyadic fraction).
pub const ROOT_OFFSET_FRAC: f64 = 0.012_345_678_9;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub header_bytes: u32,
    pub node_count: u32,
    pub lut_count: u32,
    pub star_count: u64,
    pub root_min: [f64; 3],
    pub root_size: f64,
    pub mag_min: f32,
    pub mag_scale: f32,
    pub colour_min: f32,
    pub colour_max: f32,
    pub nodes_off: u64,
    pub stars_off: u64,
    pub ids_off: u64,
    pub lut_off: u64,
    pub src_counts: [u64; 3],
    pub _reserved: [u64; 15],
}
const _: () = assert!(std::mem::size_of::<TileHeader>() == 256);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileNode {
    pub min: [f64; 3],
    pub size: f64,
    /// Index of the first child (children contiguous, octant order); 0 = leaf.
    pub first_child: u32,
    pub child_mask: u8,
    pub depth: u8,
    pub _pad: u16,
    pub star_offset: u32,
    pub star_count: u32,
    /// Brightest / faintest pseudo-absolute magnitude stored in this node (conservative bounds).
    pub m_own_min: f32,
    pub m_own_max: f32,
    /// Brightest magnitude among all descendants (+inf if none).
    pub m_desc_min: f32,
    pub _pad2: u32,
}
const _: () = assert!(std::mem::size_of::<TileNode>() == 64);

/// 16-byte GPU-friendly star record: node-relative u32 position, u16 magnitude, u8 colour code.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileStar {
    pub q: [u32; 3],
    pub mag_q: u16,
    pub colour: u8,
    /// bits 0-1 source; bit 2 Hipparcos photometric transform; bit 3 low parallax S/N;
    /// bit 4 Hipparcos beyond 1 kpc; bit 5 replaced saturated Gaia; bit 6 RUWE>1.4;
    /// bit 7 Hipparcos row placed at the Bailer-Jones distance of its Gaia DR3 neighbour.
    pub flags: u8,
}
const _: () = assert!(std::mem::size_of::<TileStar>() == 16);

impl TileStar {
    pub fn src(&self) -> u8 {
        self.flags & 3
    }
}
#[inline]
pub fn decode_mag(q: u16) -> f64 {
    MAG_MIN as f64 + q as f64 * MAG_SCALE as f64
}
#[inline]
pub fn decode_pos(node: &TileNode, q: [u32; 3]) -> DVec3 {
    let s = node.size / Q_MAX;
    DVec3::new(node.min[0] + q[0] as f64 * s, node.min[1] + q[1] as f64 * s, node.min[2] + q[2] as f64 * s)
}
/// Maximum per-axis absolute position error for a star stored in `node` (pc).
pub fn pos_error_bound(node: &TileNode) -> f64 {
    0.5 * node.size / Q_MAX + node.size * 1e-15
}
pub fn colour_code(bp_rp: f32, lo: f32, hi: f32) -> u8 {
    if bp_rp.is_nan() {
        return NO_COLOUR;
    }
    (((bp_rp.clamp(lo, hi) - lo) / (hi - lo)) * 254.0).round() as u8
}
pub fn colour_value(code: u8, lo: f32, hi: f32) -> Option<f32> {
    (code != NO_COLOUR).then(|| lo + (hi - lo) * code as f32 / 254.0)
}

#[derive(Clone, Debug)]
pub struct BuildParams {
    pub node_capacity: usize,
    pub max_depth: u8,
    pub colour_min: f32,
    pub colour_max: f32,
}
impl Default for BuildParams {
    fn default() -> Self {
        Self { node_capacity: 4096, max_depth: 24, colour_min: -0.6, colour_max: 5.0 }
    }
}

fn align64(v: &mut Vec<u8>) {
    while !v.len().is_multiple_of(64) {
        v.push(0);
    }
}

fn tile_flags(s: &Merged) -> u8 {
    let mut f = s.src & 3;
    if s.flags & (flag::G_FROM_V | flag::G_FROM_HP) != 0 {
        f |= 4;
    }
    if s.flags & flag::LOW_PLX_SNR != 0 {
        f |= 8;
    }
    if s.flags & flag::HIP_BEYOND_1KPC != 0 {
        f |= 16;
    }
    if s.flags & flag::HIP_REPLACED_GAIA != 0 {
        f |= 32;
    }
    if s.flags & flag::RUWE != 0 {
        f |= 64;
    }
    if s.flags & flag::HIP_DIST_BJ != 0 {
        f |= 128;
    }
    f
}

/// Build an RSTILE01 byte image. `lut` is the 255-entry colour LUT (rgb + bp_rp in w).
pub fn build_tiles(stars: &[Merged], lut: &[[f32; 4]], p: &BuildParams) -> io::Result<Vec<u8>> {
    if stars.is_empty() || p.node_capacity == 0 || lut.len() != 255 {
        return Err(invalid("empty input, zero capacity or bad LUT"));
    }
    let mut maxabs = 0.0f64;
    for s in stars {
        maxabs = maxabs.max(s.pos_pc.abs().max_element());
    }
    // Root offset so Sol (the default observer) never lies on an octree split plane at any depth:
    // otherwise eight nodes per level touch the observer and all get the unbounded (-inf) key.
    let root_size = 2.0 * (maxabs * 1.06).max(1.0).log2().ceil().exp2();
    let root_min = [-root_size * (0.5 - ROOT_OFFSET_FRAC); 3];
    // Quantised magnitudes first so every bound below is computed on the decoded values.
    let mut mq = Vec::with_capacity(stars.len());
    for s in stars {
        let m = s.abs_mag();
        let q = ((m - MAG_MIN as f64) / MAG_SCALE as f64).round();
        if !(0.0..=65535.0).contains(&q) {
            return Err(invalid("pseudo absolute magnitude outside tile range"));
        }
        mq.push(q as u16);
    }
    let mut order: Vec<u32> = (0..stars.len() as u32).collect();
    order.sort_by(|&a, &b| {
        mq[a as usize]
            .cmp(&mq[b as usize])
            .then(stars[a as usize].src.cmp(&stars[b as usize].src))
            .then(stars[a as usize].id.cmp(&stars[b as usize].id))
    });
    let mut nodes: Vec<TileNode> =
        vec![TileNode { min: root_min, size: root_size, m_desc_min: f32::INFINITY, ..Default::default() }];
    let mut recs: Vec<TileStar> = Vec::with_capacity(stars.len());
    let mut ids: Vec<u64> = Vec::with_capacity(stars.len());
    let mut queue = std::collections::VecDeque::new();
    queue.push_back((0usize, order));
    let margin = 2e-3f32; // conservative bound slack (f32 rounding of decoded magnitudes)
    while let Some((ni, list)) = queue.pop_front() {
        let node = nodes[ni];
        let keep =
            if list.len() <= p.node_capacity || node.depth >= p.max_depth { list.len() } else { p.node_capacity };
        let (own, rest) = list.split_at(keep);
        nodes[ni].star_offset = recs.len() as u32;
        nodes[ni].star_count = own.len() as u32;
        if let (Some(&f), Some(&l)) = (own.first(), own.last()) {
            nodes[ni].m_own_min = decode_mag(mq[f as usize]) as f32 - margin;
            nodes[ni].m_own_max = decode_mag(mq[l as usize]) as f32 + margin;
        } else {
            nodes[ni].m_own_min = f32::INFINITY;
            nodes[ni].m_own_max = f32::NEG_INFINITY;
        }
        for &i in own {
            let s = &stars[i as usize];
            let mut q = [0u32; 3];
            for (a, qa) in q.iter_mut().enumerate() {
                let t = ((s.pos_pc[a] - node.min[a]) / node.size * Q_MAX).round();
                if !(0.0..=Q_MAX).contains(&t) {
                    return Err(invalid("star outside node during quantisation"));
                }
                *qa = t as u32;
            }
            recs.push(TileStar {
                q,
                mag_q: mq[i as usize],
                colour: colour_code(s.bp_rp, p.colour_min, p.colour_max),
                flags: tile_flags(s),
            });
            ids.push(s.id);
        }
        if rest.is_empty() {
            continue;
        }
        nodes[ni].m_desc_min = decode_mag(mq[rest[0] as usize]) as f32 - margin;
        let half = node.size * 0.5;
        let centre = [node.min[0] + half, node.min[1] + half, node.min[2] + half];
        let mut parts: [Vec<u32>; 8] = Default::default();
        for &i in rest {
            let pp = stars[i as usize].pos_pc;
            let o = (pp.x >= centre[0]) as usize
                | ((pp.y >= centre[1]) as usize) << 1
                | ((pp.z >= centre[2]) as usize) << 2;
            parts[o].push(i);
        }
        let first = nodes.len();
        let mut mask = 0u8;
        for (o, part) in parts.into_iter().enumerate() {
            if part.is_empty() {
                continue;
            }
            mask |= 1 << o;
            let cmin = [
                if o & 1 != 0 { centre[0] } else { node.min[0] },
                if o & 2 != 0 { centre[1] } else { node.min[1] },
                if o & 4 != 0 { centre[2] } else { node.min[2] },
            ];
            nodes.push(TileNode {
                min: cmin,
                size: half,
                depth: node.depth + 1,
                m_desc_min: f32::INFINITY,
                ..Default::default()
            });
            queue.push_back((nodes.len() - 1, part));
        }
        nodes[ni].first_child = first as u32;
        nodes[ni].child_mask = mask;
    }
    let mut src_counts = [0u64; 3];
    for s in stars {
        src_counts[s.src as usize] += 1;
    }
    let mut buf = vec![0u8; 256];
    let nodes_off = buf.len() as u64;
    buf.extend_from_slice(bytemuck::cast_slice(&nodes));
    align64(&mut buf);
    let lut_off = buf.len() as u64;
    buf.extend_from_slice(bytemuck::cast_slice(lut));
    align64(&mut buf);
    let stars_off = buf.len() as u64;
    buf.extend_from_slice(bytemuck::cast_slice(&recs));
    align64(&mut buf);
    let ids_off = buf.len() as u64;
    buf.extend_from_slice(bytemuck::cast_slice(&ids));
    let h = TileHeader {
        magic: MAGIC,
        version: 1,
        header_bytes: 256,
        node_count: nodes.len() as u32,
        lut_count: lut.len() as u32,
        star_count: recs.len() as u64,
        root_min,
        root_size,
        mag_min: MAG_MIN,
        mag_scale: MAG_SCALE,
        colour_min: p.colour_min,
        colour_max: p.colour_max,
        nodes_off,
        stars_off,
        ids_off,
        lut_off,
        src_counts,
        _reserved: [0; 15],
    };
    buf[..256].copy_from_slice(bytemuck::bytes_of(&h));
    Ok(buf)
}

/// Random-access backing for streaming (file `pread` or in-memory image).
pub trait Backing: Sync {
    fn read_at(&self, off: u64, buf: &mut [u8]) -> io::Result<()>;
    fn len(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
impl Backing for Vec<u8> {
    fn read_at(&self, off: u64, buf: &mut [u8]) -> io::Result<()> {
        let s = usize::try_from(off).map_err(|_| invalid("offset"))?;
        let e = s.checked_add(buf.len()).ok_or_else(|| invalid("offset"))?;
        buf.copy_from_slice(self.get(s..e).ok_or_else(|| invalid("read past end"))?);
        Ok(())
    }
    fn len(&self) -> u64 {
        self.as_slice().len() as u64
    }
}
pub struct FileBacking {
    file: std::fs::File,
    len: u64,
}
impl FileBacking {
    pub fn open(path: &std::path::Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len();
        Ok(Self { file, len })
    }
}
impl Backing for FileBacking {
    fn read_at(&self, off: u64, buf: &mut [u8]) -> io::Result<()> {
        use std::os::unix::fs::FileExt;
        self.file.read_exact_at(buf, off)
    }
    fn len(&self) -> u64 {
        self.len
    }
}

/// Header + node table + LUT (resident, small). Star/id blocks stream per node.
pub struct TileIndex {
    pub header: TileHeader,
    pub nodes: Vec<TileNode>,
    pub lut: Vec<[f32; 4]>,
}
impl TileIndex {
    pub fn open(b: &impl Backing) -> io::Result<Self> {
        let mut hb = [0u8; 256];
        b.read_at(0, &mut hb)?;
        let header: TileHeader = bytemuck::pod_read_unaligned(&hb);
        let n = header.node_count as u64;
        let sc = header.star_count;
        let ok = header.magic == MAGIC
            && header.version == 1
            && header.header_bytes == 256
            && header.lut_count == 255
            && n > 0
            && header.root_size.is_finite()
            && header.root_size > 0.0
            && header.nodes_off == 256
            && header.nodes_off + n * 64 <= header.lut_off
            && header.lut_off + 255 * 16 <= header.stars_off
            && header.stars_off + sc * 16 <= header.ids_off
            && header.ids_off + sc * 8 == b.len()
            && [header.lut_off, header.stars_off, header.ids_off].iter().all(|o| o % 64 == 0)
            && header.src_counts.iter().sum::<u64>() == sc;
        if !ok {
            return Err(invalid("bad RSTILE01 header"));
        }
        let mut nodes = vec![TileNode::default(); n as usize];
        b.read_at(header.nodes_off, bytemuck::cast_slice_mut(&mut nodes))?;
        let mut lut = vec![[0f32; 4]; 255];
        b.read_at(header.lut_off, bytemuck::cast_slice_mut(&mut lut))?;
        let mut total = 0u64;
        for (i, nd) in nodes.iter().enumerate() {
            let kids = nd.child_mask.count_ones() as u64;
            let child_ok = if nd.child_mask == 0 {
                nd.first_child == 0
            } else {
                nd.first_child as u64 > i as u64 && nd.first_child as u64 + kids <= n
            };
            if !child_ok
                || nd.star_offset as u64 + nd.star_count as u64 > sc
                || !nd.size.is_finite()
                || nd.size <= 0.0
                || !nd.min.iter().all(|v| v.is_finite())
            {
                return Err(invalid("bad RSTILE01 node"));
            }
            total += nd.star_count as u64;
        }
        if total != sc || lut.iter().flatten().any(|v| !v.is_finite() || *v < -10.0) {
            return Err(invalid("RSTILE01 node star totals / LUT mismatch"));
        }
        Ok(Self { header, nodes, lut })
    }
    pub fn children(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        let nd = &self.nodes[i];
        (0..nd.child_mask.count_ones() as usize).map(move |k| nd.first_child as usize + k)
    }
}

fn aabb_distance(nd: &TileNode, p: DVec3) -> f64 {
    let mut d2 = 0.0;
    for a in 0..3 {
        let lo = nd.min[a];
        let hi = lo + nd.size;
        let v = if p[a] < lo {
            lo - p[a]
        } else if p[a] > hi {
            p[a] - hi
        } else {
            0.0
        };
        d2 += v * v;
    }
    d2.sqrt()
}
#[inline]
fn dist_mod(d_pc: f64) -> f64 {
    5.0 * (d_pc.max(1e-9) / 10.0).log10()
}

/// Minimum distance from the observer at which a star is drawn (unresolved-point limit).
pub const MIN_DRAW_AU: f64 = 1000.0;
#[inline]
pub fn min_draw_pc() -> f64 {
    MIN_DRAW_AU * crate::universe::AU / PARSEC_M
}

#[derive(Clone, Debug, Default)]
pub struct Plan {
    pub nodes: Vec<u32>,
    pub stars: u64,
    /// Every star with apparent magnitude < `m_complete` (from the plan position) is in `nodes`.
    pub m_complete: f64,
}

#[derive(PartialEq)]
struct Key(f64, u32);
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Key {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        o.0.total_cmp(&self.0).then(o.1.cmp(&self.1))
    }
}

impl TileIndex {
    /// Brightness-first best-first traversal under a star budget. Returns the exact completeness
    /// magnitude: the brightest possible apparent magnitude of anything left unloaded.
    pub fn plan(&self, cam_pc: DVec3, star_budget: u64) -> Plan {
        let key = |i: usize| {
            let nd = &self.nodes[i];
            // Nodes containing the observer have no useful distance bound: they are loaded first
            // (key -inf) so they never define the completeness magnitude unless the budget is tiny.
            let d = aabb_distance(nd, cam_pc);
            if d <= 0.0 {
                f64::NEG_INFINITY
            } else {
                // Stars closer than MIN_DRAW_AU are never drawn, so the bound distance is clamped there.
                nd.m_own_min.min(nd.m_desc_min) as f64 + dist_mod(d.max(min_draw_pc()))
            }
        };
        let mut heap = BinaryHeap::new();
        heap.push(Key(key(0), 0));
        let mut plan = Plan { m_complete: f64::INFINITY, ..Default::default() };
        while let Some(Key(k, i)) = heap.pop() {
            let nd = &self.nodes[i as usize];
            if !plan.nodes.is_empty() && plan.stars + nd.star_count as u64 > star_budget {
                plan.m_complete = k;
                break;
            }
            plan.nodes.push(i);
            plan.stars += nd.star_count as u64;
            for c in self.children(i as usize) {
                heap.push(Key(key(c), c as u32));
            }
        }
        plan
    }
}

/// Resident node block (raw records exactly as stored; decode happens per frame).
pub struct Block {
    pub stars: Vec<TileStar>,
    pub ids: Vec<u64>,
    last_used: u64,
}

#[derive(Clone, Debug, Default)]
pub struct StreamStats {
    pub nodes_planned: usize,
    pub nodes_loaded_now: usize,
    pub nodes_evicted_now: usize,
    pub bytes_read_now: u64,
    pub bytes_read_total: u64,
    pub resident_stars: u64,
    pub m_complete: f64,
    pub plan_ms: f64,
    pub load_ms: f64,
}

pub struct TileStream<B: Backing> {
    pub index: TileIndex,
    backing: B,
    pub cache: HashMap<u32, Block>,
    frame: u64,
    pub resident_budget: u64,
    bytes_total: u64,
}
impl<B: Backing> TileStream<B> {
    pub fn new(backing: B, resident_budget: u64) -> io::Result<Self> {
        let index = TileIndex::open(&backing)?;
        Ok(Self { index, backing, cache: HashMap::new(), frame: 0, resident_budget, bytes_total: 0 })
    }
    /// Plan for `cam_pc`, load missing node blocks, evict least-recently-used blocks over budget.
    pub fn update(&mut self, cam_pc: DVec3, star_budget: u64) -> io::Result<(Plan, StreamStats)> {
        self.frame += 1;
        let t0 = std::time::Instant::now();
        let plan = self.index.plan(cam_pc, star_budget);
        let plan_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = std::time::Instant::now();
        let mut st =
            StreamStats { nodes_planned: plan.nodes.len(), m_complete: plan.m_complete, plan_ms, ..Default::default() };
        for &n in &plan.nodes {
            if let Some(b) = self.cache.get_mut(&n) {
                b.last_used = self.frame;
                continue;
            }
            let nd = self.index.nodes[n as usize];
            let c = nd.star_count as usize;
            let mut stars = vec![TileStar::default(); c];
            let mut ids = vec![0u64; c];
            let so = self.index.header.stars_off + nd.star_offset as u64 * 16;
            let io = self.index.header.ids_off + nd.star_offset as u64 * 8;
            self.backing.read_at(so, bytemuck::cast_slice_mut(&mut stars))?;
            self.backing.read_at(io, bytemuck::cast_slice_mut(&mut ids))?;
            st.bytes_read_now += c as u64 * 24;
            st.nodes_loaded_now += 1;
            self.cache.insert(n, Block { stars, ids, last_used: self.frame });
        }
        let mut resident: u64 = self.cache.values().map(|b| b.stars.len() as u64).sum();
        if resident > self.resident_budget {
            let mut lru: Vec<(u64, u32)> =
                self.cache.iter().filter(|(_, b)| b.last_used != self.frame).map(|(k, b)| (b.last_used, *k)).collect();
            lru.sort_unstable();
            for (_, k) in lru {
                if resident <= self.resident_budget {
                    break;
                }
                if let Some(b) = self.cache.remove(&k) {
                    resident -= b.stars.len() as u64;
                    st.nodes_evicted_now += 1;
                }
            }
        }
        self.bytes_total += st.bytes_read_now;
        st.bytes_read_total = self.bytes_total;
        st.resident_stars = resident;
        st.load_ms = t1.elapsed().as_secs_f64() * 1e3;
        Ok((plan, st))
    }
}

/// A decoded star in camera-independent form (pc, pseudo-absolute magnitude).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decoded {
    pub src: u8,
    pub id: u64,
    pub pos_pc: DVec3,
    pub abs_mag: f64,
    pub colour: u8,
    pub flags: u8,
}
pub fn decode_block(nd: &TileNode, b: &Block, out: &mut Vec<Decoded>) {
    out.extend(b.stars.iter().zip(&b.ids).map(|(s, &id)| Decoded {
        src: s.src(),
        id,
        pos_pc: decode_pos(nd, s.q),
        abs_mag: decode_mag(s.mag_q),
        colour: s.colour,
        flags: s.flags,
    }));
}

/// Apparent G magnitude from `cam_pc` (inverse square, no extinction change).
#[inline]
pub fn apparent_mag(d: &Decoded, cam_pc: DVec3) -> f64 {
    d.abs_mag + dist_mod((d.pos_pc - cam_pc).length())
}

#[derive(Clone, Debug, Default)]
pub struct ProjectStats {
    pub considered: usize,
    pub fainter_than_complete: usize,
    pub too_close: usize,
    pub off_screen: usize,
    pub candidates: usize,
    pub submitted: usize,
    pub dropped_by_cap: usize,
    pub m_cut: f64,
    /// Submitted stars per source: [GCNS, Gaia DR3+BJ, Hipparcos-2].
    pub by_src: [usize; 3],
}

/// Unique id for a projected star. GCNS and Gaia rows are both Gaia DR3 `source_id`s (< 2^63,
/// deduplicated against each other by the merge); Hipparcos rows get bit 63 set on the HIP number.
/// (An earlier `id ^ src << 62` tag collided with real source_ids >= 2^62 and mislabelled sources.)
pub fn render_id(src: u8, id: u64) -> u64 {
    if src == SRC_HIP { id | (1 << 63) } else { id }
}

/// Project resident stars brighter than `min(m_complete, m_limit)` and keep the `cap` brightest
/// by flux (flux = 10^(-0.4 m), the GCNS G-band display convention). Deterministic order.
pub fn project(
    stars: &[Decoded],
    cam: &Camera,
    m_complete: f64,
    m_limit: f64,
    cap: usize,
    lut: &[[f32; 4]],
) -> (Vec<Source>, ProjectStats) {
    let mut st = ProjectStats { considered: stars.len(), m_cut: m_complete.min(m_limit), ..Default::default() };
    let focal = cam.height as f64 / (2.0 * (cam.fov * 0.5).tan());
    let min_d = MIN_DRAW_AU * crate::universe::AU;
    let mut out: Vec<(Source, u8)> = Vec::new();
    for s in stars {
        let rel = s.pos_pc * PARSEC_M - (cam.origin + cam.pos);
        let d = rel.length();
        if d < min_d {
            st.too_close += 1;
            continue;
        }
        let m = s.abs_mag + dist_mod(d / PARSEC_M);
        if m >= st.m_cut {
            st.fainter_than_complete += 1;
            continue;
        }
        let z = rel.dot(cam.forward);
        if z <= 0.0 {
            st.off_screen += 1;
            continue;
        }
        let c = [
            cam.width as f64 * 0.5 + rel.dot(cam.right) / z * focal,
            cam.height as f64 * 0.5 - rel.dot(cam.up) / z * focal,
        ];
        if !(c[0].is_finite() && c[1].is_finite())
            || c[0] < -SUPPORT
            || c[1] < -SUPPORT
            || c[0] > cam.width as f64 + SUPPORT
            || c[1] > cam.height as f64 + SUPPORT
        {
            st.off_screen += 1;
            continue;
        }
        let rgb = if s.colour == NO_COLOUR {
            [1.0; 3]
        } else {
            let l = lut[s.colour as usize];
            [l[0] as f64, l[1] as f64, l[2] as f64]
        };
        out.push((Source { id: render_id(s.src, s.id), center: c, flux: 10f64.powf(-0.4 * m), rgb }, s.src));
    }
    st.candidates = out.len();
    let order = |a: &(Source, u8), b: &(Source, u8)| b.0.flux.total_cmp(&a.0.flux).then(a.0.id.cmp(&b.0.id));
    if out.len() > cap {
        out.select_nth_unstable_by(cap, order);
        out.truncate(cap);
    }
    out.sort_unstable_by(order);
    st.submitted = out.len();
    st.dropped_by_cap = st.candidates - st.submitted;
    for (_, src) in &out {
        st.by_src[(*src as usize).min(2)] += 1;
    }
    (out.into_iter().map(|(s, _)| s).collect(), st)
}
