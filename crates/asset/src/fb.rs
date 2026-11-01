//! FlatBuffers-style container (see <https://flatbuffers.dev/internals/>), hand-rolled so that
//! GPU-uploadable vectors can be aligned to 256 bytes (FlatBuffers' `force_align` on vectors
//! "only works for generated C++ code", <https://flatbuffers.dev/schema/>) and so the build
//! does not depend on `flatc` code generation.
//!
//! Wire format (all little-endian, every scalar aligned to its own size):
//! - `u32` at 0: offset of the root table; 4 bytes at 4: file identifier (as FlatBuffers).
//! - table: `i32 soffset` at its start; `vtable = table - soffset`. Inline fields follow.
//! - vtable: `u16 vtable_bytes, u16 table_bytes, u16 field_offset[n]` (0 = field absent ->
//!   default). Readers built against an older schema see fewer fields (ignore new ones), newer
//!   readers get defaults for fields old writers never knew: forward and backward compatible.
//! - offsets to tables / vectors are `u32`, relative to the offset's own position, and always
//!   point *forward*: the object graph is acyclic by construction, so the verifier terminates.
//! - vector: `u32 count` immediately followed by the elements; the element data can be aligned
//!   to any power of two up to 256 (the offset points at the count, as in FlatBuffers).
//!
//! Access is zero-copy: a field is found through the vtable in O(1); vectors of POD elements
//! are borrowed straight from the buffer (mmap / staging memory) when it is aligned.

pub const IDENT: [u8; 4] = *b"BRK1";
pub const MAX_ALIGN: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyError {
    TooSmall,
    BadIdentifier,
    OutOfBounds { at: usize },
    Misaligned { at: usize },
    BadVtable { at: usize },
    BackwardOffset { at: usize },
    TooDeep,
    TooManyTables,
    TooLarge,
    /// The field is present but has the wrong size / an impossible value.
    BadField { table: &'static str, field: u16 },
    UnsupportedVersion { major: u16 },
    /// A payload failed to decode (corrupt compressed data).
    Payload(crate::rans::CodecError),
}

/// Limits for the verifier (cf. `flatbuffers::VerifierOptions`: max_depth, max_tables,
/// max_apparent_size).
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub max_depth: usize,
    pub max_tables: usize,
    pub max_bytes: usize,
    pub max_vertices: u32,
    pub max_indices: u32,
    pub max_instances: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Limits { max_depth: 16, max_tables: 1 << 20, max_bytes: 1 << 31, max_vertices: 1 << 22, max_indices: 3 << 22, max_instances: 1 << 22 }
    }
}

// ------------------------------------------------------------------------------------ builder

/// Inline-field value of a table under construction.
#[derive(Clone, Debug)]
pub enum Field {
    U8(u8),
    U16(u16),
    U32(u32),
    I64(i64),
    F64(f64),
    /// Inline struct: raw little-endian bytes and their alignment.
    Struct(Vec<u8>, usize),
    Table(Box<TableB>),
    /// Vector of bytes (elements already serialised), element data aligned to `align`.
    Bytes(Vec<u8>, usize),
    Tables(Vec<TableB>),
}

impl Field {
    fn inline_size_align(&self) -> (usize, usize) {
        match self {
            Field::U8(_) => (1, 1),
            Field::U16(_) => (2, 2),
            Field::U32(_) => (4, 4),
            Field::I64(_) | Field::F64(_) => (8, 8),
            Field::Struct(b, a) => (b.len(), *a),
            Field::Table(_) | Field::Bytes(..) | Field::Tables(_) => (4, 4),
        }
    }
}

/// A table under construction: field id -> value.
#[derive(Clone, Debug, Default)]
pub struct TableB {
    pub fields: Vec<Option<Field>>,
}

impl TableB {
    pub fn set(&mut self, id: u16, f: Field) -> &mut Self {
        let i = id as usize;
        if self.fields.len() <= i {
            self.fields.resize(i + 1, None);
        }
        self.fields[i] = Some(f);
        self
    }
    pub fn with(mut self, id: u16, f: Field) -> Self {
        self.set(id, f);
        self
    }
}

fn align_up(v: usize, a: usize) -> usize {
    (v + a - 1) & !(a - 1)
}

fn pad_to(buf: &mut Vec<u8>, a: usize) {
    let n = align_up(buf.len(), a);
    buf.resize(n, 0);
}

/// Serialise `root` (parent first, children after, offsets patched forward).
pub fn finish(root: &TableB) -> Vec<u8> {
    let mut buf = vec![0u8; 8];
    buf[4..8].copy_from_slice(&IDENT);
    let pos = write_table(&mut buf, root);
    buf[0..4].copy_from_slice(&(pos as u32).to_le_bytes());
    pad_to(&mut buf, 8);
    buf
}

fn write_table(buf: &mut Vec<u8>, t: &TableB) -> usize {
    let n = t.fields.iter().rposition(Option::is_some).map_or(0, |i| i + 1);
    // inline layout: soffset (4), then fields sorted by alignment (largest first)
    let mut order: Vec<usize> = (0..n).filter(|&i| t.fields[i].is_some()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(t.fields[i].as_ref().map_or(0, |f| f.inline_size_align().1)));
    let max_align = order.iter().map(|&i| t.fields[i].as_ref().map_or(1, |f| f.inline_size_align().1)).max().unwrap_or(4).max(4);
    let mut rel = vec![0usize; n];
    let mut off = 4usize;
    // fields are laid out relative to a table start aligned to max_align
    for &i in &order {
        let (sz, al) = t.fields[i].as_ref().map_or((0, 1), Field::inline_size_align);
        off = align_up(off, al);
        rel[i] = off;
        off += sz;
    }
    let obj_size = align_up(off, 4);
    // vtable right before the table
    pad_to(buf, 2);
    let vt = buf.len();
    let vt_bytes = 4 + 2 * n;
    buf.extend_from_slice(&(vt_bytes as u16).to_le_bytes());
    buf.extend_from_slice(&(obj_size as u16).to_le_bytes());
    for (i, r) in rel.iter().enumerate() {
        let v = if t.fields[i].is_some() { *r as u16 } else { 0 };
        buf.extend_from_slice(&v.to_le_bytes());
    }
    pad_to(buf, max_align);
    let tpos = buf.len();
    buf.resize(tpos + obj_size, 0);
    buf[tpos..tpos + 4].copy_from_slice(&((tpos - vt) as i32).to_le_bytes());
    let mut deferred: Vec<(usize, &Field)> = Vec::new();
    for &i in &order {
        let p = tpos + rel[i];
        match t.fields[i].as_ref() {
            Some(Field::U8(v)) => buf[p] = *v,
            Some(Field::U16(v)) => buf[p..p + 2].copy_from_slice(&v.to_le_bytes()),
            Some(Field::U32(v)) => buf[p..p + 4].copy_from_slice(&v.to_le_bytes()),
            Some(Field::I64(v)) => buf[p..p + 8].copy_from_slice(&v.to_le_bytes()),
            Some(Field::F64(v)) => buf[p..p + 8].copy_from_slice(&v.to_le_bytes()),
            Some(Field::Struct(b, _)) => buf[p..p + b.len()].copy_from_slice(b),
            Some(f) => deferred.push((p, f)),
            None => {}
        }
    }
    for (p, f) in deferred {
        let child = match f {
            Field::Table(c) => write_table(buf, c),
            Field::Bytes(b, align) => {
                let data = align_up(buf.len() + 4, (*align).clamp(4, MAX_ALIGN));
                buf.resize(data, 0);
                buf[data - 4..data].copy_from_slice(&(b.len() as u32).to_le_bytes());
                buf.extend_from_slice(b);
                data - 4
            }
            Field::Tables(ts) => {
                pad_to(buf, 4);
                let v = buf.len();
                buf.extend_from_slice(&(ts.len() as u32).to_le_bytes());
                buf.resize(v + 4 + 4 * ts.len(), 0);
                for (k, c) in ts.iter().enumerate() {
                    let slot = v + 4 + 4 * k;
                    let cp = write_table(buf, c);
                    buf[slot..slot + 4].copy_from_slice(&((cp - slot) as u32).to_le_bytes());
                }
                v
            }
            _ => unreachable!(),
        };
        buf[p..p + 4].copy_from_slice(&((child - p) as u32).to_le_bytes());
    }
    tpos
}

// ------------------------------------------------------------------------------------- reader

#[inline]
fn rd_u16(b: &[u8], p: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(p..p.checked_add(2)?)?.try_into().ok()?))
}
#[inline]
fn rd_u32(b: &[u8], p: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(p..p.checked_add(4)?)?.try_into().ok()?))
}

/// Zero-copy view of a table. Accessors are O(1) through the vtable and bounds-checked (a
/// malformed buffer yields defaults / `None`, never UB); run [`Verifier`] first to *reject*
/// malformed buffers instead.
#[derive(Clone, Copy, Debug)]
pub struct Table<'a> {
    pub buf: &'a [u8],
    pub pos: usize,
}

impl<'a> Table<'a> {
    pub fn root(buf: &'a [u8]) -> Option<Table<'a>> {
        let p = rd_u32(buf, 0)? as usize;
        (buf.get(4..8)? == IDENT).then_some(Table { buf, pos: p })
    }
    fn vtable(&self) -> Option<usize> {
        let so = rd_u32(self.buf, self.pos)? as i32 as i64;
        let vt = self.pos as i64 - so;
        (vt >= 0).then_some(vt as usize)
    }
    /// Absolute position of field `id`, if present.
    pub fn field(&self, id: u16) -> Option<usize> {
        let vt = self.vtable()?;
        let vt_bytes = rd_u16(self.buf, vt)? as usize;
        let slot = 4 + 2 * id as usize;
        if slot + 2 > vt_bytes {
            return None; // older writer: field unknown to it -> default
        }
        let o = rd_u16(self.buf, vt + slot)? as usize;
        (o != 0).then_some(self.pos + o)
    }
    pub fn u8(&self, id: u16, d: u8) -> u8 {
        self.field(id).and_then(|p| self.buf.get(p).copied()).unwrap_or(d)
    }
    pub fn u16(&self, id: u16, d: u16) -> u16 {
        self.field(id).and_then(|p| rd_u16(self.buf, p)).unwrap_or(d)
    }
    pub fn u32(&self, id: u16, d: u32) -> u32 {
        self.field(id).and_then(|p| rd_u32(self.buf, p)).unwrap_or(d)
    }
    pub fn i64(&self, id: u16, d: i64) -> i64 {
        self.field(id).and_then(|p| Some(i64::from_le_bytes(self.buf.get(p..p + 8)?.try_into().ok()?))).unwrap_or(d)
    }
    pub fn f64(&self, id: u16, d: f64) -> f64 {
        self.field(id).and_then(|p| Some(f64::from_le_bytes(self.buf.get(p..p + 8)?.try_into().ok()?))).unwrap_or(d)
    }
    pub fn struct_bytes(&self, id: u16, size: usize) -> Option<&'a [u8]> {
        let p = self.field(id)?;
        self.buf.get(p..p.checked_add(size)?)
    }
    fn deref(&self, id: u16) -> Option<usize> {
        let p = self.field(id)?;
        let o = rd_u32(self.buf, p)? as usize;
        p.checked_add(o)
    }
    pub fn table(&self, id: u16) -> Option<Table<'a>> {
        Some(Table { buf: self.buf, pos: self.deref(id)? })
    }
    /// Byte vector (zero-copy).
    pub fn bytes(&self, id: u16) -> Option<&'a [u8]> {
        let v = self.deref(id)?;
        let n = rd_u32(self.buf, v)? as usize;
        self.buf.get(v + 4..(v + 4).checked_add(n)?)
    }
    /// Vector of POD elements, borrowed in place if the bytes are aligned for `T`.
    pub fn pod<T: bytemuck::Pod>(&self, id: u16) -> Option<&'a [T]> {
        bytemuck::try_cast_slice(self.bytes(id)?).ok()
    }
    pub fn tables_len(&self, id: u16) -> usize {
        self.deref(id).and_then(|v| rd_u32(self.buf, v)).unwrap_or(0) as usize
    }
    pub fn tables_get(&self, id: u16, i: usize) -> Option<Table<'a>> {
        let v = self.deref(id)?;
        let slot = v.checked_add(4)?.checked_add(i.checked_mul(4)?)?;
        let o = rd_u32(self.buf, slot)? as usize;
        Some(Table { buf: self.buf, pos: slot.checked_add(o)? })
    }
}

// ----------------------------------------------------------------------------------- verifier

/// Structural verifier: every table, vtable, offset and vector the schema reaches is checked
/// for bounds, alignment and forward direction, with depth / table-count limits.
pub struct Verifier<'a> {
    pub buf: &'a [u8],
    pub limits: Limits,
    tables: usize,
}

impl<'a> Verifier<'a> {
    pub fn new(buf: &'a [u8], limits: Limits) -> Self {
        Verifier { buf, limits, tables: 0 }
    }

    pub fn root(&mut self) -> Result<Table<'a>, VerifyError> {
        if self.buf.len() < 8 {
            return Err(VerifyError::TooSmall);
        }
        if self.buf.len() > self.limits.max_bytes {
            return Err(VerifyError::TooLarge);
        }
        if self.buf[4..8] != IDENT {
            return Err(VerifyError::BadIdentifier);
        }
        let p = rd_u32(self.buf, 0).ok_or(VerifyError::TooSmall)? as usize;
        if p < 8 {
            return Err(VerifyError::BackwardOffset { at: 0 });
        }
        self.table_at(p, 0)
    }

    fn check(&self, p: usize, n: usize, align: usize) -> Result<(), VerifyError> {
        let end = p.checked_add(n).ok_or(VerifyError::OutOfBounds { at: p })?;
        if end > self.buf.len() {
            return Err(VerifyError::OutOfBounds { at: p });
        }
        if !p.is_multiple_of(align) {
            return Err(VerifyError::Misaligned { at: p });
        }
        Ok(())
    }

    /// Verify the table header + vtable at `p`.
    pub fn table_at(&mut self, p: usize, depth: usize) -> Result<Table<'a>, VerifyError> {
        if depth > self.limits.max_depth {
            return Err(VerifyError::TooDeep);
        }
        self.tables += 1;
        if self.tables > self.limits.max_tables {
            return Err(VerifyError::TooManyTables);
        }
        self.check(p, 4, 4)?;
        let so = rd_u32(self.buf, p).ok_or(VerifyError::OutOfBounds { at: p })? as i32 as i64;
        let vt = p as i64 - so;
        if vt < 0 {
            return Err(VerifyError::BadVtable { at: p });
        }
        let vt = vt as usize;
        self.check(vt, 4, 2)?;
        let vt_bytes = rd_u16(self.buf, vt).ok_or(VerifyError::BadVtable { at: vt })? as usize;
        let obj = rd_u16(self.buf, vt + 2).ok_or(VerifyError::BadVtable { at: vt })? as usize;
        if vt_bytes < 4 || !vt_bytes.is_multiple_of(2) || obj < 4 {
            return Err(VerifyError::BadVtable { at: vt });
        }
        self.check(vt, vt_bytes, 2)?;
        self.check(p, obj, 4)?;
        for k in 0..(vt_bytes - 4) / 2 {
            let o = rd_u16(self.buf, vt + 4 + 2 * k).ok_or(VerifyError::BadVtable { at: vt })? as usize;
            if o != 0 && (o < 4 || o >= obj) {
                return Err(VerifyError::BadVtable { at: vt });
            }
        }
        Ok(Table { buf: self.buf, pos: p })
    }

    /// Inline scalar/struct field of `size` bytes must lie inside the table and be aligned.
    pub fn scalar(&self, t: &Table<'a>, id: u16, size: usize, align: usize) -> Result<(), VerifyError> {
        if let Some(p) = t.field(id) {
            let vt = t.vtable().ok_or(VerifyError::BadVtable { at: t.pos })?;
            let obj = rd_u16(self.buf, vt + 2).unwrap_or(0) as usize;
            if p + size > t.pos + obj {
                return Err(VerifyError::OutOfBounds { at: p });
            }
            self.check(p, size, align)?;
        }
        Ok(())
    }

    fn offset_target(&self, t: &Table<'a>, id: u16) -> Result<Option<usize>, VerifyError> {
        let Some(p) = t.field(id) else { return Ok(None) };
        self.scalar(t, id, 4, 4)?;
        let o = rd_u32(self.buf, p).ok_or(VerifyError::OutOfBounds { at: p })? as usize;
        if o == 0 {
            return Err(VerifyError::BackwardOffset { at: p });
        }
        Ok(Some(p.checked_add(o).ok_or(VerifyError::OutOfBounds { at: p })?))
    }

    pub fn table(&mut self, t: &Table<'a>, id: u16, depth: usize) -> Result<Option<Table<'a>>, VerifyError> {
        match self.offset_target(t, id)? {
            None => Ok(None),
            Some(c) => self.table_at(c, depth + 1).map(Some),
        }
    }

    /// Byte vector whose element data is aligned to `align` and holds a multiple of `elem` bytes.
    pub fn bytes(&self, t: &Table<'a>, id: u16, elem: usize, align: usize) -> Result<Option<&'a [u8]>, VerifyError> {
        let Some(v) = self.offset_target(t, id)? else { return Ok(None) };
        self.check(v, 4, 4)?;
        let n = rd_u32(self.buf, v).ok_or(VerifyError::OutOfBounds { at: v })? as usize;
        self.check(v + 4, n, align.max(1))?;
        if elem > 1 && !n.is_multiple_of(elem) {
            return Err(VerifyError::BadField { table: "vector", field: id });
        }
        Ok(Some(&self.buf[v + 4..v + 4 + n]))
    }

    /// Vector of tables; returns the element count after checking the offset slots.
    pub fn tables(&self, t: &Table<'a>, id: u16) -> Result<usize, VerifyError> {
        let Some(v) = self.offset_target(t, id)? else { return Ok(0) };
        self.check(v, 4, 4)?;
        let n = rd_u32(self.buf, v).ok_or(VerifyError::OutOfBounds { at: v })? as usize;
        if n > self.limits.max_tables {
            return Err(VerifyError::TooManyTables);
        }
        self.check(v + 4, n.checked_mul(4).ok_or(VerifyError::TooLarge)?, 4)?;
        for k in 0..n {
            let slot = v + 4 + 4 * k;
            if rd_u32(self.buf, slot).unwrap_or(0) == 0 {
                return Err(VerifyError::BackwardOffset { at: slot });
            }
        }
        Ok(n)
    }
}

// ------------------------------------------------------------------------------ aligned bytes

/// Owned byte buffer aligned to [`MAX_ALIGN`] (like an mmap'd file or a mapped staging buffer),
/// so in-place POD views of 256-aligned vectors are always possible.
pub struct AlignedBytes {
    words: Vec<Align256>,
    len: usize,
}

#[repr(C, align(256))]
#[derive(Clone, Copy)]
struct Align256([u8; 256]);

impl AlignedBytes {
    pub fn from_slice(b: &[u8]) -> Self {
        let mut words = vec![Align256([0; 256]); b.len().div_ceil(256)];
        let dst: &mut [u8] = bytemuck_bytes_mut(&mut words);
        dst[..b.len()].copy_from_slice(b);
        AlignedBytes { words, len: b.len() }
    }
    pub fn as_slice(&self) -> &[u8] {
        &bytemuck_bytes(&self.words)[..self.len]
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        let len = self.len;
        &mut bytemuck_bytes_mut(&mut self.words)[..len]
    }
}

fn bytemuck_bytes(w: &[Align256]) -> &[u8] {
    // SAFETY: Align256 is repr(C) over [u8; 256] (no padding, every bit pattern valid), so a
    // slice of n of them is exactly 256 * n initialised bytes; lifetime and mutability follow `w`.
    unsafe { std::slice::from_raw_parts(w.as_ptr().cast::<u8>(), w.len() * 256) }
}

fn bytemuck_bytes_mut(w: &mut [Align256]) -> &mut [u8] {
    // SAFETY: as above; the exclusive borrow of `w` is transferred to the returned slice.
    unsafe { std::slice::from_raw_parts_mut(w.as_mut_ptr().cast::<u8>(), w.len() * 256) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_offsets_and_defaults() {
        let child = TableB::default().with(0, Field::U32(77));
        let root = TableB::default()
            .with(0, Field::U16(3))
            .with(2, Field::F64(1.5))
            .with(3, Field::Bytes(vec![1, 2, 3, 4, 5, 6, 7, 8], 256))
            .with(4, Field::Table(Box::new(child.clone())))
            .with(5, Field::Tables(vec![child.clone(), child]));
        let buf = AlignedBytes::from_slice(&finish(&root));
        let b = buf.as_slice();
        let mut v = Verifier::new(b, Limits::default());
        let r = v.root().unwrap();
        assert_eq!(r.u16(0, 9), 3);
        assert_eq!(r.u8(1, 42), 42, "absent field -> default");
        assert_eq!(r.u32(99, 5), 5, "field beyond the vtable (newer schema) -> default");
        assert_eq!(r.f64(2, 0.0), 1.5);
        let bytes = v.bytes(&r, 3, 1, 256).unwrap().unwrap();
        assert_eq!(bytes, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(bytes.as_ptr() as usize % 256, 0, "256-aligned vector in an aligned buffer");
        let pod: &[u32] = r.pod(3).unwrap();
        assert_eq!(pod, &[0x04030201, 0x08070605]);
        let c = v.table(&r, 4, 0).unwrap().unwrap();
        assert_eq!(c.u32(0, 0), 77);
        assert_eq!(v.tables(&r, 5).unwrap(), 2);
        assert_eq!(r.tables_get(5, 1).unwrap().u32(0, 0), 77);
    }

    #[test]
    fn verifier_rejects_basic_corruption() {
        let root = TableB::default().with(0, Field::Bytes(vec![9; 40], 16));
        let good = finish(&root);
        assert!(Verifier::new(&good, Limits::default()).root().is_ok());
        assert_eq!(Verifier::new(&good[..4], Limits::default()).root().unwrap_err(), VerifyError::TooSmall);
        let mut bad = good.clone();
        bad[4] = b'X';
        assert_eq!(Verifier::new(&bad, Limits::default()).root().unwrap_err(), VerifyError::BadIdentifier);
        let mut bad = good.clone();
        bad[0..4].copy_from_slice(&(good.len() as u32 + 100).to_le_bytes());
        assert!(Verifier::new(&bad, Limits::default()).root().is_err());
    }
}
