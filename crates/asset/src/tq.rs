//! TurboQuant-style fixed-rate lossy coding of brick density samples (prototype, opt-in).
//!
//! Technique from Zandieh, Daliri, Hadian, Mirrokni, "TurboQuant: Online Vector Quantization
//! with Near-optimal Distortion Rate", arXiv:2504.19874 (Algorithm 1, `TurboQuant_mse`): a
//! data-oblivious random rotation makes every coordinate of a unit vector follow a known
//! (Beta -> Gaussian N(0, 1/d)) distribution whatever the input, so one precomputed Lloyd-Max
//! scalar codebook quantises every coordinate. The rotation follows turbovec
//! (<https://github.com/RyanCodrai/turbovec>, MIT, `src/rotation.rs`): rounds of a seeded
//! permutation, +-1 sign flips and a Walsh-Hadamard transform, instead of a dense QR rotation,
//! so the transform is deterministic bit-for-bit.
//!
//! Our adaptation (see `starter.md`, "Learning from turbovec and TurboQuant"):
//! - a brick's `(n+3)^3` samples are cut into blocks of [`D`] = 512 consecutive samples (the last
//!   block padded with its mean). The rotation is data-oblivious, so the spatial tiling does not
//!   matter to the quantiser;
//! - per block: the mean (f32) is removed, the residual is normalised, rotated by [`Rotation`]
//!   (2 rounds, **unnormalised** WHT: the total gain is exactly 512 = 2^9), and each coordinate
//!   is quantised against the standard-Gaussian Lloyd-Max codebook ([`centroids`], Max 1960);
//! - reconstruction `x = mean + a * v`, where `v` is the inverse transform of the centroids
//!   (adds and subtracts only) and `a` (f32) is the **least-squares** scale fitted by the encoder
//!   (it absorbs the norm, the transform gain and the Lloyd-Max shrinkage, as turbovec's
//!   per-vector `||v|| / <u, x^>` does for inner products). No QJL stage: we need MSE, not
//!   unbiased inner products;
//! - samples are rounded back to `i32` with `floor(x + 0.5)`.
//!
//! Every decode operation is an IEEE add/sub, one multiply and one add, in a fixed order, so the
//! GPU decoder (`render-ash`, `shaders/tq.slang`) reproduces the CPU bit-exactly.
//!
//! **Lossy: never used for seam-defining data by default.** [`decode_hybrid`] restores the six
//! brick face planes (`i, j, k` in `{0, n}`) exactly from a lossless correction, so brick seams stay
//! bit-identical.
//!
//! Blob layout (little endian, `u32` words): 8-word header (`TQB1`, version, bits, n, samples,
//! blocks, crc32, 0), then `blocks x (mean, a)` f32 pairs, then the bit-packed codes (code `j` at
//! bit `j * bits`, LSB first, may straddle words). The CRC covers the whole blob with the CRC word
//! zeroed; [`TqView::parse`] verifies everything before any decode.
use voxel::BrickSamples;

/// Block length (power of two).
pub const D: usize = 512;
/// Rotation rounds (turbovec uses 2; one round leaves blocks too structured).
pub const ROUNDS: usize = 2;
const SEED: u64 = 0x5475_7262_6f51_7631;
const MAGIC: u32 = u32::from_le_bytes(*b"TQB1");
const VERSION: u32 = 1;
const HEADER_WORDS: usize = 8;

/// Positive half of the standard-Gaussian Lloyd-Max centroids (Max 1960, Table I) for 1..=4 bits.
const POS: [&[f32]; 4] = [
    &[0.7979],
    &[0.4528, 1.5104],
    &[0.2451, 0.7560, 1.3439, 2.1520],
    &[0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6181, 2.0690, 2.7326],
];

/// Sorted centroids for `bits` (1..=4): `2^bits` values, symmetric.
pub fn centroids(bits: u32) -> Vec<f32> {
    let p = POS[bits as usize - 1];
    p.iter().rev().map(|c| -c).chain(p.iter().copied()).collect()
}

/// Decision boundaries (midpoints, `2^bits - 1`).
fn boundaries(c: &[f32]) -> Vec<f64> {
    c.windows(2).map(|w| 0.5 * (w[0] as f64 + w[1] as f64)).collect()
}

struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The fixed orthogonal-up-to-2^9 transform: per round a Fisher-Yates permutation
/// (`out[i] = in[perm[i]]`), sign flips, an unnormalised Walsh-Hadamard transform.
pub struct Rotation {
    pub perm: [[u16; D]; ROUNDS],
    /// bit `i` of word `i / 32`: coordinate `i` negated
    pub signs: [[u32; D / 32]; ROUNDS],
}

impl Rotation {
    pub fn new() -> Self {
        let mut rng = SplitMix(SEED);
        let mut perm = [[0u16; D]; ROUNDS];
        let mut signs = [[0u32; D / 32]; ROUNDS];
        for r in 0..ROUNDS {
            let p = &mut perm[r];
            for (i, v) in p.iter_mut().enumerate() {
                *v = i as u16;
            }
            for i in (1..D).rev() {
                let j = (rng.next() % (i as u64 + 1)) as usize;
                p.swap(i, j);
            }
            for i in 0..D {
                if rng.next() & 1 == 1 {
                    signs[r][i / 32] |= 1 << (i % 32);
                }
            }
        }
        Rotation { perm, signs }
    }

    /// Tables for `tq.slang`: permutations, sign words, then the `2^bits` centroids (f32 bits).
    pub fn gpu_tables(&self, bits: u32) -> Vec<u32> {
        let mut t: Vec<u32> = self.perm.iter().flatten().map(|&p| p as u32).collect();
        t.extend(self.signs.iter().flatten());
        debug_assert_eq!(t.len(), GPU_CENT);
        t.extend(centroids(bits).iter().map(|c| c.to_bits()));
        t
    }

    fn neg(&self, r: usize, i: usize) -> bool {
        self.signs[r][i / 32] >> (i % 32) & 1 == 1
    }

    /// Forward transform (encoder, f64): `z = H D2 P2 H D1 P1 u`, gain 512.
    pub fn forward(&self, u: &[f64; D]) -> [f64; D] {
        let mut v = *u;
        for r in 0..ROUNDS {
            let mut t = [0.0f64; D];
            for i in 0..D {
                let x = v[self.perm[r][i] as usize];
                t[i] = if self.neg(r, i) { -x } else { x };
            }
            wht(&mut t);
            v = t;
        }
        v
    }

    /// Transposed transform (decoder, f32, the canonical reconstruction): per round in
    /// reverse: WHT, signs, inverse permutation. Exactly what `tq.slang` computes.
    pub fn inverse_f32(&self, z: &[f32; D]) -> [f32; D] {
        let mut v = *z;
        for r in (0..ROUNDS).rev() {
            wht(&mut v);
            let mut t = [0.0f32; D];
            for i in 0..D {
                let x = if self.neg(r, i) { -v[i] } else { v[i] };
                t[self.perm[r][i] as usize] = x;
            }
            v = t;
        }
        v
    }
}

impl Default for Rotation {
    fn default() -> Self {
        Self::new()
    }
}

/// In-place unnormalised Walsh-Hadamard transform, butterflies in a fixed order (stage `h`,
/// pair `(j, j + h)`: `(a + b, a - b)`); every pair of a stage is independent, so any parallel
/// schedule gives identical results.
fn wht<T: Copy + std::ops::Add<Output = T> + std::ops::Sub<Output = T>>(v: &mut [T; D]) {
    let mut h = 1;
    while h < D {
        for i in (0..D).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (v[j], v[j + h]);
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

/// Reconstruct one sample: `floor(mean + a * v + 0.5)` clamped to the density range
/// `[-DQ_MAX, DQ_MAX]` (exact in f32, so the GPU's float->int conversion is always defined).
/// f32 ops in this order, no fusing (`tq.slang` marks them `precise`).
#[inline]
pub fn reconstruct(mean: f32, a: f32, v: f32) -> i32 {
    let p = a * v;
    let x = mean + p;
    let r = (x + 0.5).floor();
    let m = voxel::DQ_MAX as f32;
    r.clamp(-m, m) as i32
}

/// u32 offsets of the GPU decoder's table buffer ([`Rotation::gpu_tables`], `tq.slang`).
pub const GPU_PERM: usize = 0;
pub const GPU_SIGNS: usize = ROUNDS * D;
pub const GPU_CENT: usize = GPU_SIGNS + ROUNDS * D / 32;

/// Errors from [`TqView::parse`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TqError {
    Truncated,
    Misaligned,
    BadMagic,
    BadVersion(u32),
    BadBits(u32),
    BadShape,
    BadSize,
    BadCrc,
    BadParam,
}

/// A verified, zero-copy view of a TQ blob.
pub struct TqView<'a> {
    pub bits: u32,
    pub n: u32,
    pub samples: u32,
    pub blocks: u32,
    /// `(mean, a)` per block
    pub params: &'a [[f32; 2]],
    pub codes: &'a [u32],
    /// whole blob as words (the GPU decoder reads it as is)
    pub words: &'a [u32],
}

fn code_words(blocks: usize, bits: u32) -> usize {
    (blocks * D * bits as usize).div_ceil(32)
}

/// Total blob size in bytes for `samples` samples at `bits`.
pub fn blob_bytes(samples: usize, bits: u32) -> usize {
    let blocks = samples.div_ceil(D);
    (HEADER_WORDS + 2 * blocks + code_words(blocks, bits)) * 4
}

impl<'a> TqView<'a> {
    /// Verify `bytes` completely (structure, sizes, CRC, finite parameters) and view it in place.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, TqError> {
        if bytes.len() < HEADER_WORDS * 4 {
            return Err(TqError::Truncated);
        }
        if !bytes.len().is_multiple_of(4) {
            return Err(TqError::BadSize);
        }
        let words: &[u32] = bytemuck::try_cast_slice(bytes).map_err(|_| TqError::Misaligned)?;
        let h = &words[..HEADER_WORDS];
        if h[0] != MAGIC {
            return Err(TqError::BadMagic);
        }
        if h[1] != VERSION {
            return Err(TqError::BadVersion(h[1]));
        }
        let bits = h[2];
        if !(1..=4).contains(&bits) {
            return Err(TqError::BadBits(bits));
        }
        let (n, samples, blocks) = (h[3], h[4], h[5]);
        if !(1..=128).contains(&n)
            || samples as u64 != (n as u64 + 3).pow(3)
            || blocks as usize != (samples as usize).div_ceil(D)
            || h[7] != 0
        {
            return Err(TqError::BadShape);
        }
        if words.len() * 4 != blob_bytes(samples as usize, bits) {
            return Err(TqError::BadSize);
        }
        if crc_words(words) != h[6] {
            return Err(TqError::BadCrc);
        }
        let p = &words[HEADER_WORDS..HEADER_WORDS + 2 * blocks as usize];
        let params: &[[f32; 2]] = bytemuck::cast_slice(p);
        if params.iter().flatten().any(|x| !x.is_finite() || x.abs() > 1.0e30) {
            return Err(TqError::BadParam);
        }
        let codes = &words[HEADER_WORDS + 2 * blocks as usize..];
        Ok(TqView { bits, n, samples, blocks, params, codes, words })
    }

    /// Code `j` (global index over all blocks).
    #[inline]
    pub fn code(&self, j: usize) -> u32 {
        // (a block's codes start word-aligned: D * bits is a multiple of 32)
        let bit = j * self.bits as usize;
        let (w, s) = (bit / 32, bit % 32);
        let mut c = self.codes[w] >> s;
        if s + self.bits as usize > 32 {
            c |= self.codes[w + 1] << (32 - s);
        }
        c & ((1 << self.bits) - 1)
    }

    /// Decode all samples (CPU reference).
    pub fn decode(&self, rot: &Rotation) -> Vec<i32> {
        let c = centroids(self.bits);
        let mut out = vec![0i32; self.samples as usize];
        for b in 0..self.blocks as usize {
            let mut z = [0.0f32; D];
            for (j, zj) in z.iter_mut().enumerate() {
                *zj = c[self.code(b * D + j) as usize];
            }
            let v = rot.inverse_f32(&z);
            let [mean, a] = self.params[b];
            for (j, &vj) in v.iter().enumerate() {
                if let Some(o) = out.get_mut(b * D + j) {
                    *o = reconstruct(mean, a, vj);
                }
            }
        }
        out
    }
}

/// IEEE CRC-32 over the blob words with the CRC word (index 6) taken as 0.
fn crc_words(words: &[u32]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let t = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 == 1 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *e = c;
        }
        t
    });
    let mut c = !0u32;
    for (i, w) in words.iter().enumerate() {
        let w = if i == 6 { 0 } else { *w };
        for byte in w.to_le_bytes() {
            c = t[((c ^ byte as u32) & 0xFF) as usize] ^ (c >> 8);
        }
    }
    !c
}

/// Encode a brick's corner samples (`s.d`; transition faces are not covered) at `bits` (1..=4).
pub fn encode(s: &BrickSamples, bits: u32, rot: &Rotation) -> Vec<u8> {
    encode_values(&s.d, s.n as u32, bits, rot)
}

/// Encode `(n+3)^3` values.
pub fn encode_values(vals: &[i32], n: u32, bits: u32, rot: &Rotation) -> Vec<u8> {
    assert!((1..=4).contains(&bits) && vals.len() as u64 == (n as u64 + 3).pow(3));
    let blocks = vals.len().div_ceil(D);
    let c = centroids(bits);
    let bnd = boundaries(&c);
    let gscale = 1.0 / (D as f64).sqrt();
    let mut params = Vec::with_capacity(blocks);
    let mut codes = vec![0u32; code_words(blocks, bits)];
    for b in 0..blocks {
        let real = &vals[b * D..((b + 1) * D).min(vals.len())];
        let mean = (real.iter().map(|&x| x as f64).sum::<f64>() / real.len() as f64) as f32;
        let mut r = [0.0f64; D];
        for (j, x) in real.iter().enumerate() {
            r[j] = *x as f64 - mean as f64;
        }
        let norm = r.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mut idx = [0u32; D];
        let mut a = 0.0f32;
        if norm > 1e-9 {
            let u = r.map(|x| x / norm);
            let z = rot.forward(&u);
            let mut zc = [0.0f32; D];
            for j in 0..D {
                let g = z[j] * gscale;
                let k = bnd.partition_point(|&t| t < g);
                idx[j] = k as u32;
                zc[j] = c[k];
            }
            let v = rot.inverse_f32(&zc);
            // least-squares scale over the real samples
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for j in 0..real.len() {
                num += r[j] * v[j] as f64;
                den += v[j] as f64 * v[j] as f64;
            }
            if den > 0.0 {
                a = (num / den) as f32;
            }
        }
        params.push([mean, a]);
        for (j, &k) in idx.iter().enumerate() {
            let bit = (b * D + j) * bits as usize;
            let (w, s) = (bit / 32, bit % 32);
            codes[w] |= k << s;
            if s + bits as usize > 32 {
                codes[w + 1] |= k >> (32 - s);
            }
        }
    }
    let mut words = vec![MAGIC, VERSION, bits, n, vals.len() as u32, blocks as u32, 0, 0];
    for p in &params {
        words.push(p[0].to_bits());
        words.push(p[1].to_bits());
    }
    words.extend_from_slice(&codes);
    words[6] = crc_words(&words);
    bytemuck::cast_slice(&words).to_vec()
}

/// Is sample index `k` of an `(n+3)^3` brick on a face plane (`i`, `j` or `k` in `{0, n}`)?
/// Those samples alone define the vertex positions on the brick's faces (seams).
pub fn on_face_plane(k: usize, n: usize) -> bool {
    let m = n + 3;
    let (i, j, l) = (k % m, (k / m) % m, k / (m * m));
    [i, j, l].iter().any(|&c| c == 1 || c == n + 1)
}

/// Hybrid decode: TQ everywhere, then the face planes restored exactly from `exact`
/// (in the format this is a lossless, Lorenzo-coded correction; see the bench).
pub fn decode_hybrid(view: &TqView<'_>, rot: &Rotation, exact: &[i32]) -> Vec<i32> {
    let mut out = view.decode(rot);
    let n = view.n as usize;
    for (k, o) in out.iter_mut().enumerate() {
        if on_face_plane(k, n) {
            *o = exact[k];
        }
    }
    out
}

/// The hybrid's lossless correction field: `exact - tq` on the face planes, 0 elsewhere.
pub fn face_residual(view: &TqView<'_>, rot: &Rotation, exact: &[i32]) -> Vec<i32> {
    let dec = view.decode(rot);
    let n = view.n as usize;
    dec.iter()
        .zip(exact)
        .enumerate()
        .map(|(k, (d, e))| if on_face_plane(k, n) { e.wrapping_sub(*d) } else { 0 })
        .collect()
}

/// Moved to [`crate::samples::mesh_band`] (the stored, versioned form is `SampleBlob.band`).
pub use crate::samples::mesh_band;

#[cfg(test)]
mod tests {
    use super::*;

    fn erf(x: f64) -> f64 {
        // Abramowitz & Stegun 7.1.26 (|err| < 1.5e-7)
        let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
        let y = 1.0
            - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t + 0.254_829_592)
                * t
                * (-x * x).exp();
        if x >= 0.0 { y } else { -y }
    }

    /// The tabulated centroids are Lloyd-Max fixed points for N(0, 1): each centroid is the
    /// conditional mean of its cell (within the table's 4-digit precision).
    #[test]
    fn tq_codebooks_are_lloyd_max_fixed_points() {
        let pdf = |x: f64| (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let cdf = |x: f64| 0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2));
        for bits in 1..=4 {
            let c = centroids(bits);
            let b = boundaries(&c);
            for k in 0..c.len() {
                let lo = if k == 0 { -1e9 } else { b[k - 1] };
                let hi = if k == c.len() - 1 { 1e9 } else { b[k] };
                // E[X | lo < X < hi] = (pdf(lo) - pdf(hi)) / (cdf(hi) - cdf(lo))
                let m = (pdf(lo) - pdf(hi)) / (cdf(hi) - cdf(lo));
                assert!((m - c[k] as f64).abs() < 2e-3, "bits {bits} k {k}: {m} vs {}", c[k]);
            }
        }
    }

    /// Orthogonality up to the exact gain 512 and the forward/transpose pairing.
    #[test]
    fn tq_rotation_is_scaled_orthogonal_and_deterministic() {
        let r = Rotation::new();
        let r2 = Rotation::new();
        assert!(r.perm == r2.perm && r.signs == r2.signs);
        let mut u = [0.0f64; D];
        u[3] = 1.0;
        let z = r.forward(&u);
        let n2: f64 = z.iter().map(|x| x * x).sum();
        assert!((n2 - 512.0f64 * 512.0).abs() < 1e-6);
        // a unit basis vector is spread out: no coordinate keeps more than a quarter of the norm
        assert!(z.iter().all(|x| x.abs() < 0.25 * 512.0));
        assert!(z.iter().filter(|x| x.abs() > 1e-9).count() > D / 2);
        let zf: [f32; D] = z.map(|x| x as f32);
        let back = r.inverse_f32(&zf);
        for (j, b) in back.iter().enumerate() {
            let want = if j == 3 { 512.0 * 512.0 } else { 0.0 };
            assert!((*b as f64 - want).abs() < 1.0, "{j}: {b}");
        }
    }

    fn field(n: usize, seed: u64) -> Vec<i32> {
        let m = n + 3;
        let mut rng = SplitMix(seed);
        let (a, b) = ((rng.next() % 1000) as f64 / 300.0, (rng.next() % 1000) as f64 / 300.0);
        (0..m * m * m)
            .map(|k| {
                let (i, j, l) = ((k % m) as f64, ((k / m) % m) as f64, (k / (m * m)) as f64);
                let h = 12.0 + 4.0 * (i / 5.0 + a).sin() + 3.0 * (l / 7.0 + b).cos();
                (((j - h) * 4000.0) as i32).clamp(-60000, 60000)
            })
            .collect()
    }

    /// Round trip: deterministic, error bounds per bit width (relative to the block spread),
    /// hybrid restores the face planes exactly.
    #[test]
    fn tq_round_trip_error_bounds_and_hybrid_faces() {
        let rot = Rotation::new();
        let n = 16;
        let vals = field(n, 7);
        let mut prev = f64::MAX;
        for bits in 1..=4 {
            let blob = encode_values(&vals, n as u32, bits, &rot);
            assert_eq!(blob, encode_values(&vals, n as u32, bits, &rot));
            assert_eq!(blob.len(), blob_bytes(vals.len(), bits));
            let v = TqView::parse(&blob).unwrap();
            let dec = v.decode(&rot);
            // relative MSE against the per-block variance: the paper's D_mse for unit vectors
            // (0.36, 0.117, 0.03, 0.009 at 1..4 bits) with margin for d = 512 and the LS scale
            let (mut err, mut var) = (0.0f64, 0.0f64);
            for b in 0..v.blocks as usize {
                let r = &vals[b * D..((b + 1) * D).min(vals.len())];
                let m = r.iter().map(|&x| x as f64).sum::<f64>() / r.len() as f64;
                for (j, x) in r.iter().enumerate() {
                    var += (*x as f64 - m).powi(2);
                    err += (*x as f64 - dec[b * D + j] as f64).powi(2);
                }
            }
            let rel = err / var;
            let bound = [0.40, 0.13, 0.036, 0.011][bits as usize - 1];
            assert!(rel < bound, "bits {bits}: relative MSE {rel}");
            assert!(rel < prev);
            prev = rel;
            let hyb = decode_hybrid(&v, &rot, &vals);
            for (k, (h, x)) in hyb.iter().zip(&vals).enumerate() {
                if on_face_plane(k, n) {
                    assert_eq!(h, x);
                }
            }
            let res = face_residual(&v, &rot, &vals);
            let patched: Vec<i32> = dec.iter().zip(&res).map(|(d, r)| d.wrapping_add(*r)).collect();
            assert_eq!(patched, hyb);
        }
    }

    /// The verifier rejects corruption anywhere (CRC over the whole blob) and bad headers.
    #[test]
    fn tq_verifier_rejects_corrupt_blobs() {
        let rot = Rotation::new();
        let vals = field(8, 3);
        let blob = encode_values(&vals, 8, 3, &rot);
        let words: Vec<u32> = bytemuck::cast_slice(&blob).to_vec();
        let ok = |w: &[u32]| TqView::parse(bytemuck::cast_slice(w)).is_ok();
        assert!(ok(&words));
        let mut rng = SplitMix(99);
        for _ in 0..2000 {
            let mut w = words.clone();
            let bit = (rng.next() % (w.len() as u64 * 32)) as usize;
            w[bit / 32] ^= 1 << (bit % 32);
            assert!(!ok(&w), "flip at bit {bit} accepted");
        }
        for cut in [0usize, 4, 31, blob.len() - 4] {
            assert!(TqView::parse(&blob[..cut]).is_err());
        }
        let mut w = words.clone();
        w[2] = 7;
        assert_eq!(TqView::parse(bytemuck::cast_slice(&w)).err(), Some(TqError::BadBits(7)));
    }

    #[test]
    fn tq_mesh_band_keeps_meshes_bit_identical() {
        use voxel::scene::{Adversarial, BlobScene};
        use voxel::*;
        let cfg = VoxelConfig::new(coords::BrickGrid::new(16, 0.25).unwrap());
        let cases: Vec<(Box<dyn DensityField>, Box<dyn LodLayout>)> = vec![
            (
                Box::new(BlobScene { radius: 9.0, seed: 3 }),
                Box::new(ShellLayout::camera_centred(&cfg.grid, [5, -3, 7], &[1, 1])),
            ),
            (
                Box::new(Adversarial { seed: 4, voxel_m: 0.25, radius_m: 7.0 }),
                Box::new(SlabLayout { axis: 1, split: 0, lod_below: 1, lod_above: 0 }),
            ),
        ];
        let (mut bricks, mut altered) = (0, 0);
        for (field, lay) in &cases {
            for bc in bricks_in(&cfg.grid, lay.as_ref(), ([-128; 3], [128; 3]), 2) {
                let (mask, finer) = plan_brick(&cfg.grid, lay.as_ref(), bc).unwrap();
                let s = sample_brick(field.as_ref(), &cfg, bc, mask).unwrap();
                if !s.has_sign_change() {
                    continue;
                }
                for t in [1, 1024] {
                    let b = BrickSamples { coord: s.coord, n: s.n, d: mesh_band(&s, mask, t), faces: s.faces.clone() };
                    altered += s.d.iter().zip(&b.d).filter(|(x, y)| x != y).count();
                    assert_eq!(mesh_brick(&cfg, &s, mask, finer), mesh_brick(&cfg, &b, mask, finer), "{bc:?} t {t}");
                }
                bricks += 1;
            }
        }
        assert!(bricks > 20 && altered > 0, "{bricks} bricks, {altered} altered");
    }
}
