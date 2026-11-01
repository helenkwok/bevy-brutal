//! Static-model rANS entropy coder (Duda 2013, "Asymmetric numeral systems"; byte-wise
//! renormalisation as in F. Giesen's ryg_rans) plus a raw bit stream for the "extra bits" of
//! tagged values (Draco's TAGGED_SYMBOLS: an rANS-coded bit length, then the remaining bits raw).
//!
//! One rANS state carries symbols of many models (contexts): decode order mirrors encode order
//! reversed, so the encoder logs `(model, symbol)` during the forward pass and codes the log
//! backwards at the end. Models are *static* per file (built from the symbol counts of every
//! payload in the file, stored once like a dictionary), so decoding builds each lookup table
//! once per file, not per brick, and the per-symbol work is a mask, a table load and a
//! multiply-add.
//!
//! Every decode path is total: corrupt input yields `Err`, never a panic or an out-of-bounds
//! access, and a successful decode must end exactly at the initial state with all bytes consumed.

/// Probability precision: frequencies of a model sum to `1 << PROB_BITS`.
pub const PROB_BITS: u32 = 12;
pub const PROB_SCALE: u32 = 1 << PROB_BITS;
/// Largest alphabet of any model (tagged values use 33 tokens).
pub const MAX_ALPHABET: usize = 64;
const RANS_L: u32 = 1 << 23;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecError {
    /// rANS input ended early or did not end at the initial state.
    Truncated,
    BadState,
    /// A decoded symbol is not valid where it was read.
    BadSymbol,
    /// A model's frequencies do not sum to `PROB_SCALE`, or a model is used that has no symbols.
    BadModel,
    /// Decoded data violates a structural invariant (index out of range, overflow, ...).
    Corrupt,
    /// A count or size exceeds the configured limits.
    Limit,
    /// The sample payload is band-clamped (mesh-only, format minor 2): true density was
    /// requested but is not stored. Use `File::decode_mesh_samples` or regenerate the brick.
    Banded,
}

/// A static frequency model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model {
    pub freq: [u16; MAX_ALPHABET],
    cum: [u32; MAX_ALPHABET + 1],
}

impl Default for Model {
    fn default() -> Self {
        Model { freq: [0; MAX_ALPHABET], cum: [0; MAX_ALPHABET + 1] }
    }
}

impl Model {
    /// Normalise counts to `PROB_SCALE`; every symbol that occurred keeps a frequency >= 1.
    /// All-zero counts give the empty model (no symbol can be coded with it).
    pub fn from_counts(counts: &[u64; MAX_ALPHABET]) -> Model {
        let total: u64 = counts.iter().sum();
        let mut m = Model::default();
        if total == 0 {
            return m;
        }
        let mut f = [0i64; MAX_ALPHABET];
        for i in 0..MAX_ALPHABET {
            if counts[i] > 0 {
                f[i] = ((counts[i] as u128 * PROB_SCALE as u128 + total as u128 / 2) / total as u128).max(1) as i64;
            }
        }
        let mut sum: i64 = f.iter().sum();
        // fix rounding: take from / give to the most frequent symbols (never below 1)
        while sum != PROB_SCALE as i64 {
            let mut best = 0;
            for i in 0..MAX_ALPHABET {
                if f[i] > f[best] {
                    best = i;
                }
            }
            if sum > PROB_SCALE as i64 {
                // largest symbol that can still give
                let mut k = usize::MAX;
                for i in 0..MAX_ALPHABET {
                    if f[i] > 1 && (k == usize::MAX || f[i] > f[k]) {
                        k = i;
                    }
                }
                let d = (sum - PROB_SCALE as i64).min(f[k] - 1);
                f[k] -= d;
                sum -= d;
            } else {
                f[best] += PROB_SCALE as i64 - sum;
                sum = PROB_SCALE as i64;
            }
        }
        for (q, v) in m.freq.iter_mut().zip(f.iter()) {
            *q = *v as u16;
        }
        m.rebuild();
        m
    }

    /// Validate stored frequencies (sum must be `PROB_SCALE`, or all zero = unused model).
    pub fn from_freqs(freq: &[u16]) -> Result<Model, CodecError> {
        if freq.len() != MAX_ALPHABET {
            return Err(CodecError::BadModel);
        }
        let sum: u32 = freq.iter().map(|&f| f as u32).sum();
        if sum != 0 && sum != PROB_SCALE {
            return Err(CodecError::BadModel);
        }
        let mut m = Model::default();
        m.freq.copy_from_slice(freq);
        m.rebuild();
        Ok(m)
    }

    fn rebuild(&mut self) {
        let mut c = 0u32;
        for i in 0..MAX_ALPHABET {
            self.cum[i] = c;
            c += self.freq[i] as u32;
        }
        self.cum[MAX_ALPHABET] = c;
    }

    pub fn is_empty(&self) -> bool {
        self.cum[MAX_ALPHABET] == 0
    }

    /// Ideal code length of `counts` under this model, in bits (for size estimates).
    pub fn cost_bits(&self, counts: &[u64; MAX_ALPHABET]) -> f64 {
        let mut b = 0.0;
        for (c, q) in counts.iter().zip(self.freq.iter()) {
            if *c > 0 {
                b += *c as f64 * (PROB_SCALE as f64 / (*q).max(1) as f64).log2();
            }
        }
        b
    }
}

/// A model with its decode lookup table (`PROB_SCALE` entries).
pub struct DecModel {
    freq: [u16; MAX_ALPHABET],
    cum: [u32; MAX_ALPHABET + 1],
    sym: Box<[u8]>,
}

impl DecModel {
    pub fn new(m: &Model) -> DecModel {
        let mut sym = vec![0u8; if m.is_empty() { 0 } else { PROB_SCALE as usize }].into_boxed_slice();
        if !m.is_empty() {
            for s in 0..MAX_ALPHABET {
                let (a, b) = (m.cum[s] as usize, m.cum[s + 1] as usize);
                sym[a..b].fill(s as u8);
            }
        }
        DecModel { freq: m.freq, cum: m.cum, sym }
    }
}

/// Symbol counts per model (pass 1 of encoding).
#[derive(Clone)]
pub struct Counts(pub Vec<[u64; MAX_ALPHABET]>);

impl Counts {
    pub fn new(models: usize) -> Counts {
        Counts(vec![[0; MAX_ALPHABET]; models])
    }
    pub fn add_log(&mut self, log: &SymLog) {
        for &(m, s) in &log.syms {
            self.0[m as usize][s as usize] += 1;
        }
    }
    pub fn models(&self) -> Vec<Model> {
        self.0.iter().map(Model::from_counts).collect()
    }
}

/// Forward-pass log of symbols plus the raw-bit stream.
#[derive(Default, Clone, Debug)]
pub struct SymLog {
    pub syms: Vec<(u8, u8)>,
    pub bits: BitWriter,
}

impl SymLog {
    #[inline]
    pub fn sym(&mut self, model: u8, s: u8) {
        self.syms.push((model, s));
    }
    /// Tagged unsigned value: token = bit length (0..=32), then `len - 1` raw bits.
    #[inline]
    pub fn uval(&mut self, model: u8, v: u32) {
        let tok = if v == 0 { 0 } else { 32 - v.leading_zeros() };
        self.syms.push((model, tok as u8));
        if tok > 1 {
            self.bits.write(v & ((1u32 << (tok - 1)) - 1), tok - 1);
        }
    }
    #[inline]
    pub fn sval(&mut self, model: u8, v: i32) {
        self.uval(model, zigzag(v));
    }

    /// rANS-encode the log with `models`. Panics if a symbol has zero frequency (an encoder bug:
    /// models are always built from counts that include this log).
    pub fn encode(&self, models: &[Model]) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.syms.len() / 2 + 8);
        let mut x = RANS_L;
        for &(mi, s) in self.syms.iter().rev() {
            let m = &models[mi as usize];
            let f = m.freq[s as usize] as u32;
            assert!(f > 0, "symbol {s} of model {mi} has zero frequency");
            let c = m.cum[s as usize];
            let x_max = ((RANS_L >> PROB_BITS) << 8) * f;
            while x >= x_max {
                out.push(x as u8);
                x >>= 8;
            }
            x = ((x / f) << PROB_BITS) + (x % f) + c;
        }
        out.extend_from_slice(&[(x >> 24) as u8, (x >> 16) as u8, (x >> 8) as u8, x as u8]);
        out.reverse();
        out
    }
}

#[inline]
pub fn zigzag(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}
#[inline]
pub fn unzigzag(u: u32) -> i32 {
    ((u >> 1) as i32) ^ -((u & 1) as i32)
}

/// rANS decoder over a byte slice, plus the raw-bit reader.
pub struct Decoder<'a> {
    x: u32,
    buf: &'a [u8],
    pos: usize,
    models: &'a [DecModel],
    pub bits: BitReader<'a>,
}

impl<'a> Decoder<'a> {
    pub fn new(rans: &'a [u8], bits: &'a [u8], models: &'a [DecModel]) -> Result<Decoder<'a>, CodecError> {
        if rans.len() < 4 {
            return Err(CodecError::Truncated);
        }
        let x = u32::from_le_bytes([rans[0], rans[1], rans[2], rans[3]]);
        if !(RANS_L..RANS_L << 8).contains(&x) {
            return Err(CodecError::BadState);
        }
        Ok(Decoder { x, buf: rans, pos: 4, models, bits: BitReader::new(bits) })
    }

    #[inline]
    pub fn sym(&mut self, model: u8) -> Result<u8, CodecError> {
        let m = self.models.get(model as usize).ok_or(CodecError::BadModel)?;
        if m.sym.is_empty() {
            return Err(CodecError::BadModel);
        }
        let slot = self.x & (PROB_SCALE - 1);
        let s = m.sym[slot as usize];
        let f = m.freq[s as usize] as u32;
        self.x = f * (self.x >> PROB_BITS) + slot - m.cum[s as usize];
        while self.x < RANS_L {
            let b = *self.buf.get(self.pos).ok_or(CodecError::Truncated)?;
            self.pos += 1;
            self.x = (self.x << 8) | b as u32;
        }
        Ok(s)
    }

    #[inline]
    pub fn uval(&mut self, model: u8) -> Result<u32, CodecError> {
        let tok = self.sym(model)? as u32;
        match tok {
            0 => Ok(0),
            1 => Ok(1),
            2..=32 => Ok((1u32 << (tok - 1)) | self.bits.read(tok - 1)?),
            _ => Err(CodecError::BadSymbol),
        }
    }
    #[inline]
    pub fn sval(&mut self, model: u8) -> Result<i32, CodecError> {
        Ok(unzigzag(self.uval(model)?))
    }

    /// A complete, uncorrupted stream ends at the encoder's initial state with all input used.
    pub fn finish(&self) -> Result<(), CodecError> {
        if self.x != RANS_L || self.pos != self.buf.len() {
            return Err(CodecError::BadState);
        }
        self.bits.finish()
    }
}

/// LSB-first bit writer.
#[derive(Default, Clone, Debug)]
pub struct BitWriter {
    pub bytes: Vec<u8>,
    acc: u64,
    n: u32,
}

impl BitWriter {
    #[inline]
    pub fn write(&mut self, v: u32, nbits: u32) {
        debug_assert!(nbits <= 32);
        if nbits == 0 {
            return;
        }
        self.acc |= ((v as u64) & ((1u64 << nbits) - 1)) << self.n;
        self.n += nbits;
        while self.n >= 8 {
            self.bytes.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }
    pub fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.bytes.push(self.acc as u8);
        }
        self.bytes
    }
    pub fn finished(&self) -> Vec<u8> {
        self.clone().finish()
    }
}

/// LSB-first bit reader; reading past the end is an error.
pub struct BitReader<'a> {
    buf: &'a [u8],
    pos: usize,
    acc: u64,
    n: u32,
}

impl<'a> BitReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        BitReader { buf, pos: 0, acc: 0, n: 0 }
    }
    #[inline]
    pub fn read(&mut self, nbits: u32) -> Result<u32, CodecError> {
        if nbits == 0 {
            return Ok(0);
        }
        if nbits > 32 {
            return Err(CodecError::BadSymbol);
        }
        if self.n < nbits {
            self.refill();
            if self.n < nbits {
                return Err(CodecError::Truncated);
            }
        }
        let v = (self.acc & ((1u64 << nbits) - 1)) as u32;
        self.acc = if nbits == 64 { 0 } else { self.acc >> nbits };
        self.n -= nbits;
        Ok(v)
    }
    /// Top the accumulator up to >= 57 bits (or to the end of the buffer).
    #[inline]
    fn refill(&mut self) {
        if let Some(w) = self.buf.get(self.pos..self.pos + 8) {
            let word = u64::from_le_bytes([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]);
            let take = (63 - self.n) / 8;
            self.acc |= word << self.n;
            if take < 8 {
                // keep only the bytes actually consumed
                let keep = self.n + take * 8;
                self.acc &= if keep >= 64 { u64::MAX } else { (1u64 << keep) - 1 };
            }
            self.pos += take as usize;
            self.n += take * 8;
        } else {
            while self.n <= 56 {
                let Some(&b) = self.buf.get(self.pos) else { break };
                self.pos += 1;
                self.acc |= (b as u64) << self.n;
                self.n += 8;
            }
        }
    }
    /// All bytes consumed and fewer than 8 padding bits left, all zero.
    pub fn finish(&self) -> Result<(), CodecError> {
        if self.pos != self.buf.len() || self.n >= 8 || self.acc != 0 {
            return Err(CodecError::BadState);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s >> 33
    }

    #[test]
    fn roundtrip_mixed_models_and_values() {
        let mut s = 1u64;
        let mut log = SymLog::default();
        let mut want = Vec::new();
        for i in 0..2000 {
            if i % 3 == 0 {
                let v = (lcg(&mut s) % 7) as u8;
                log.sym(0, v);
                want.push((0, v as i64));
            } else {
                let v = (lcg(&mut s) as i64 % 5000) - 2500;
                log.sval(1, v as i32);
                want.push((1, v));
            }
        }
        let mut c = Counts::new(2);
        c.add_log(&log);
        let models = c.models();
        let rans = log.encode(&models);
        let bits = log.bits.finished();
        let dm: Vec<DecModel> = models.iter().map(DecModel::new).collect();
        let mut d = Decoder::new(&rans, &bits, &dm).unwrap();
        for (m, v) in want {
            if m == 0 {
                assert_eq!(d.sym(0).unwrap() as i64, v);
            } else {
                assert_eq!(d.sval(1).unwrap() as i64, v);
            }
        }
        d.finish().unwrap();
    }

    #[test]
    fn extreme_values_and_single_symbol_models() {
        let mut log = SymLog::default();
        for v in [0, 1, -1, i32::MAX, i32::MIN, 12345, -77777] {
            log.sval(0, v);
        }
        for _ in 0..100 {
            log.sym(1, 5); // single-symbol model: 0 bits per symbol
        }
        let mut c = Counts::new(2);
        c.add_log(&log);
        let models = c.models();
        let rans = log.encode(&models);
        let bits = log.bits.finished();
        let dm: Vec<DecModel> = models.iter().map(DecModel::new).collect();
        let mut d = Decoder::new(&rans, &bits, &dm).unwrap();
        for v in [0, 1, -1, i32::MAX, i32::MIN, 12345, -77777] {
            assert_eq!(d.sval(0).unwrap(), v);
        }
        for _ in 0..100 {
            assert_eq!(d.sym(1).unwrap(), 5);
        }
        d.finish().unwrap();
        assert!(rans.len() < 40, "single-symbol model should cost ~0 bits: {}", rans.len());
    }

    #[test]
    fn normalisation_keeps_rare_symbols_and_sums_exactly() {
        let mut counts = [0u64; MAX_ALPHABET];
        counts[0] = 1_000_000_000;
        for (i, c) in counts.iter_mut().enumerate().skip(1).take(40) {
            *c = i as u64;
        }
        let m = Model::from_counts(&counts);
        assert_eq!(m.freq.iter().map(|&f| f as u32).sum::<u32>(), PROB_SCALE);
        assert!(m.freq[1..41].iter().all(|&f| f >= 1));
        assert!(Model::from_freqs(&m.freq).is_ok());
        let mut bad = m.freq;
        bad[0] += 1;
        assert_eq!(Model::from_freqs(&bad), Err(CodecError::BadModel));
    }

    #[test]
    fn garbage_never_panics() {
        let mut counts = [0u64; MAX_ALPHABET];
        for (i, c) in counts.iter_mut().enumerate().take(33) {
            *c = 1 + i as u64;
        }
        let dm = vec![DecModel::new(&Model::from_counts(&counts)), DecModel::new(&Model::default())];
        let mut s = 7u64;
        for len in 0..64 {
            let buf: Vec<u8> = (0..len).map(|_| lcg(&mut s) as u8).collect();
            if let Ok(mut d) = Decoder::new(&buf, &buf, &dm) {
                for _ in 0..200 {
                    let _ = d.uval(0);
                    assert!(d.sym(1).is_err());
                    assert!(d.sym(9).is_err());
                }
                let _ = d.finish();
            }
        }
    }
}
