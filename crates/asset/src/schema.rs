//! The `BRK1` file schema, written with [`crate::fb`]. In FlatBuffers IDL it would read:
//!
//! ```text
//! file_identifier "BRK1";
//! struct Key { x: long; y: long; z: long; lod: ubyte; }            // 32 bytes, align 8
//! struct Origin { x: long; y: long; z: long; }
//! table MeshBlob  { vertex_count: uint; index_count: uint; qbits: ubyte; conn: ubyte;
//!                   normal_bits: ubyte = 10; flags: ubyte; rans: [ubyte]; bits: [ubyte];
//!                   fshift: ubyte = 0; /* appended */ }
//! table SampleBlob{ n: ushort; face_mask: ubyte; rans: [ubyte]; bits: [ubyte];
//!                   band: uint = 0; /* appended, minor 2: >0 = mesh-only band clamp */ }
//! table RawMesh   { vertices: [ubyte] (align 256); indices: [ubyte] (align 256); } // GPU-ready
//! table Brick     { key: Key; mesh: MeshBlob; samples: SampleBlob; raw: RawMesh; }
//! table InstChunk { count: uint; origin: Origin; rans: [ubyte]; bits: [ubyte];
//!                   mat_bits: ubyte = 10; /* appended, minor 1 */ }
//! table Root      { major: ushort = 1; minor: ushort; voxel_m: double; brick_size: ushort;
//!                   models: [ushort] (>= BASE_MODELS x 64 rANS frequencies; minor 1
//!                   writes NUM_MODELS = 23);
//!                   bricks: [Brick]; inst_chunks: [InstChunk]; }
//! ```
//!
//! Evolution follows FlatBuffers' rules (<https://flatbuffers.dev/evolution/>): new fields are
//! only appended, never removed or renumbered (deprecated fields keep their slot), every field
//! is optional with a default. Unlike FlatBuffers we also store a `major` version: a reader
//! rejects files whose *meaning* changed incompatibly (e.g. a new predictor for an existing
//! blob), which field-level evolution cannot express. `minor` is informational.

use crate::fb::{self, Field, Limits, Table, TableB, Verifier, VerifyError};
use crate::inst::{self, InstHeader, Instance};
use crate::mesh::{self, Decoded, MeshHeader, MeshOpts};
use crate::rans::{CodecError, Counts, DecModel, Decoder, MAX_ALPHABET, Model, SymLog};
use crate::samples::{self, SampleHeader};
use coords::{BrickCoord, BrickGrid};
use rayon::prelude::*;
use voxel::{BrickMesh, BrickSamples, VoxelVertex};

pub const MAJOR: u16 = 1;
pub const MINOR: u16 = 2;
pub const NUM_MODELS: usize = 23;
/// Models every file has (minor 0 wrote 22); later ones are optional and default to uniform.
pub const BASE_MODELS: usize = 22;

pub mod root {
    pub const MAJOR: u16 = 0;
    pub const MINOR: u16 = 1;
    pub const VOXEL_M: u16 = 2;
    pub const BRICK_SIZE: u16 = 3;
    pub const MODELS: u16 = 4;
    pub const BRICKS: u16 = 5;
    pub const INST_CHUNKS: u16 = 6;
}
pub mod brick {
    pub const KEY: u16 = 0;
    pub const MESH: u16 = 1;
    pub const SAMPLES: u16 = 2;
    pub const RAW: u16 = 3;
}
pub mod meshb {
    pub const VERTEX_COUNT: u16 = 0;
    pub const INDEX_COUNT: u16 = 1;
    pub const QBITS: u16 = 2;
    pub const CONN: u16 = 3;
    pub const NORMAL_BITS: u16 = 4;
    pub const FLAGS: u16 = 5;
    pub const RANS: u16 = 6;
    pub const BITS: u16 = 7;
    /// appended after the first files were written (default 0): evolution in practice
    pub const FSHIFT: u16 = 8;
}
pub mod sampleb {
    pub const N: u16 = 0;
    pub const FACE_MASK: u16 = 1;
    pub const RANS: u16 = 2;
    pub const BITS: u16 = 3;
    /// appended in minor 2 (default 0 = full density; t >= 1 = mesh-exact band, mesh-only)
    pub const BAND: u16 = 4;
}
pub mod raw {
    pub const VERTICES: u16 = 0;
    pub const INDICES: u16 = 1;
}
pub mod instc {
    pub const COUNT: u16 = 0;
    pub const ORIGIN: u16 = 1;
    pub const RANS: u16 = 2;
    pub const BITS: u16 = 3;
    /// appended in minor 1
    pub const MAT_BITS: u16 = 4;
}

const KEY_BYTES: usize = 32;

fn key_bytes(c: BrickCoord) -> Vec<u8> {
    let mut b = Vec::with_capacity(KEY_BYTES);
    for v in [c.x, c.y, c.z] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.push(c.lod);
    b.resize(KEY_BYTES, 0);
    b
}

fn read_i64s<const N: usize>(b: &[u8]) -> [i64; N] {
    std::array::from_fn(|k| i64::from_le_bytes(b[8 * k..8 * k + 8].try_into().unwrap_or([0; 8])))
}

// ----------------------------------------------------------------------------------- writer

struct PendingBrick {
    coord: BrickCoord,
    mesh: Option<(MeshHeader, SymLog)>,
    samples: Option<(SampleHeader, SymLog)>,
    raw: Option<(Vec<u8>, Vec<u8>)>,
}

/// Two-pass writer: logs are collected per brick, then one set of static models is fitted to
/// the whole file (a shared dictionary: per-brick tables would cost more than small bricks),
/// then every log is rANS-coded in parallel.
pub struct Writer {
    grid: BrickGrid,
    bricks: Vec<PendingBrick>,
    insts: Vec<(InstHeader, SymLog)>,
    /// Write an extra (unknown to v1.0 readers) field into every table: schema-evolution tests.
    pub test_future_fields: bool,
}

impl Writer {
    pub fn new(grid: BrickGrid) -> Self {
        Writer { grid, bricks: Vec::new(), insts: Vec::new(), test_future_fields: false }
    }

    fn entry(&mut self, coord: BrickCoord) -> &mut PendingBrick {
        if self.bricks.last().is_none_or(|b| b.coord != coord) {
            self.bricks.push(PendingBrick { coord, mesh: None, samples: None, raw: None });
        }
        let last = self.bricks.len() - 1;
        &mut self.bricks[last]
    }

    pub fn add_mesh(&mut self, coord: BrickCoord, m: &BrickMesh, opts: MeshOpts) {
        let e = mesh::encode(&self.grid, coord, m, opts);
        self.entry(coord).mesh = Some(e);
    }

    /// Encode many bricks in parallel (pass 1).
    pub fn add_meshes<'a>(&mut self, items: impl IntoIterator<Item = (BrickCoord, &'a BrickMesh)>, opts: MeshOpts) {
        let items: Vec<_> = items.into_iter().collect();
        let grid = self.grid;
        let enc: Vec<_> = items.par_iter().map(|(c, m)| (*c, mesh::encode(&grid, *c, m, opts))).collect();
        for (c, e) in enc {
            self.entry(c).mesh = Some(e);
        }
    }

    /// Full density (readable by every consumer).
    pub fn add_samples(&mut self, s: &BrickSamples) {
        let e = samples::encode(s);
        self.entry(s.coord).samples = Some(e);
    }

    /// Mesh-only payload: the mesh-exact band ([`samples::mesh_band`], threshold `t >= 1`
    /// quanta, mask = the faces `s` carries). Readers get it only as
    /// [`samples::MeshSamples`]. Requires format minor 2 to be read correctly; older readers
    /// ignore the flag, so only write it for minor-2 consumers.
    pub fn add_samples_banded(&mut self, s: &BrickSamples, t: i32) {
        let e = samples::encode_banded(s, t);
        self.entry(s.coord).samples = Some(e);
    }

    /// Full samples mislabelled as banded (tests: the reader must reject them).
    #[doc(hidden)]
    pub fn add_samples_mislabelled(&mut self, s: &BrickSamples, t: u32) {
        let (mut h, log) = samples::encode(s);
        h.band = t;
        self.entry(s.coord).samples = Some((h, log));
    }

    /// [`Self::add_samples_banded`] for many bricks in parallel.
    pub fn add_samples_banded_par(&mut self, items: &[BrickSamples], t: i32) {
        let enc: Vec<_> = items.par_iter().map(|s| (s.coord, samples::encode_banded(s, t))).collect();
        for (c, e) in enc {
            self.entry(c).samples = Some(e);
        }
    }

    /// Uncompressed GPU-layout copy (vertex / index blobs aligned to 256 bytes).
    pub fn add_raw(&mut self, coord: BrickCoord, m: &BrickMesh) {
        let v = bytemuck::cast_slice::<VoxelVertex, u8>(&m.vertices).to_vec();
        let i = bytemuck::cast_slice::<u32, u8>(&m.indices).to_vec();
        self.entry(coord).raw = Some((v, i));
    }

    pub fn add_instances(&mut self, items: &[Instance]) {
        self.insts.push(inst::encode(items));
    }

    /// A pre-minor-1 instance chunk (10-bit materials, no `mat_bits` field): tests only.
    #[doc(hidden)]
    pub fn add_instances_legacy(&mut self, items: &[Instance]) {
        self.insts.push(inst::encode_bits(items, 10));
    }

    pub fn finish(self) -> Vec<u8> {
        let mut counts = Counts::new(NUM_MODELS);
        for b in &self.bricks {
            for log in [b.mesh.as_ref().map(|m| &m.1), b.samples.as_ref().map(|s| &s.1)].into_iter().flatten() {
                counts.add_log(log);
            }
        }
        for (_, log) in &self.insts {
            counts.add_log(log);
        }
        let models = counts.models();
        let mut mbytes = Vec::with_capacity(NUM_MODELS * MAX_ALPHABET * 2);
        for m in &models {
            for f in m.freq {
                mbytes.extend_from_slice(&f.to_le_bytes());
            }
        }
        let fut = self.test_future_fields;
        let future = |t: TableB| if fut { t.with(15, Field::U32(0xDEAD_BEEF)) } else { t };
        let build = |b: &PendingBrick| {
            let mut t = TableB::default().with(brick::KEY, Field::Struct(key_bytes(b.coord), 8));
            if let Some((h, log)) = &b.mesh {
                let mb = TableB::default()
                    .with(meshb::VERTEX_COUNT, Field::U32(h.vertex_count))
                    .with(meshb::INDEX_COUNT, Field::U32(h.index_count))
                    .with(meshb::QBITS, Field::U8(h.qbits))
                    .with(meshb::CONN, Field::U8(h.conn))
                    .with(meshb::NORMAL_BITS, Field::U8(h.normal_bits))
                    .with(meshb::FLAGS, Field::U8(h.flags))
                    .with(meshb::RANS, Field::Bytes(log.encode(&models), 4))
                    .with(meshb::BITS, Field::Bytes(log.bits.finished(), 4))
                    .with(meshb::FSHIFT, Field::U8(h.fshift));
                t.set(brick::MESH, Field::Table(Box::new(future(mb))));
            }
            if let Some((h, log)) = &b.samples {
                let sb = TableB::default()
                    .with(sampleb::N, Field::U16(h.n))
                    .with(sampleb::FACE_MASK, Field::U8(h.face_mask))
                    .with(sampleb::RANS, Field::Bytes(log.encode(&models), 4))
                    .with(sampleb::BITS, Field::Bytes(log.bits.finished(), 4));
                // full payloads omit the field (default 0): byte-identical to minor-1 blobs
                let sb = if h.band != 0 { sb.with(sampleb::BAND, Field::U32(h.band)) } else { sb };
                t.set(brick::SAMPLES, Field::Table(Box::new(sb)));
            }
            if let Some((v, i)) = &b.raw {
                let rb = TableB::default()
                    .with(raw::VERTICES, Field::Bytes(v.clone(), 256))
                    .with(raw::INDICES, Field::Bytes(i.clone(), 256));
                t.set(brick::RAW, Field::Table(Box::new(rb)));
            }
            future(t)
        };
        // Miri's Stacked Borrows flags crossbeam-epoch (rayon's deque) itself; run serially there
        let bricks: Vec<TableB> = if cfg!(miri) {
            self.bricks.iter().map(build).collect()
        } else {
            self.bricks.par_iter().map(build).collect()
        };
        let chunks: Vec<TableB> = self
            .insts
            .iter()
            .map(|(h, log)| {
                let mut o = Vec::new();
                for v in h.origin {
                    o.extend_from_slice(&v.to_le_bytes());
                }
                let mut t = TableB::default()
                    .with(instc::COUNT, Field::U32(h.count))
                    .with(instc::ORIGIN, Field::Struct(o, 8))
                    .with(instc::RANS, Field::Bytes(log.encode(&models), 4))
                    .with(instc::BITS, Field::Bytes(log.bits.finished(), 4));
                // legacy chunks omit the field (its default, 10, is what they mean)
                if h.mat_bits != 10 {
                    t.set(instc::MAT_BITS, Field::U8(h.mat_bits));
                }
                t
            })
            .collect();
        let root = TableB::default()
            .with(root::MAJOR, Field::U16(MAJOR))
            .with(root::MINOR, Field::U16(MINOR + fut as u16))
            .with(root::VOXEL_M, Field::F64(self.grid.voxel_m()))
            .with(root::BRICK_SIZE, Field::U16(self.grid.brick_size() as u16))
            .with(root::MODELS, Field::Bytes(mbytes, 4))
            .with(root::BRICKS, Field::Tables(bricks))
            .with(root::INST_CHUNKS, Field::Tables(chunks));
        fb::finish(&future(root))
    }
}

// ----------------------------------------------------------------------------------- reader

/// A verified file. Opening checks every table / vtable / offset / vector reachable through
/// the schema plus all header counts against [`Limits`]; payload bytes are only touched when a
/// brick is decoded (zero-copy: blob slices borrow the input buffer).
pub struct File<'a> {
    pub buf: &'a [u8],
    pub root: Table<'a>,
    pub major: u16,
    pub minor: u16,
    pub grid: BrickGrid,
    pub models: Vec<DecModel>,
    bricks: Vec<BrickView<'a>>,
    chunks: Vec<ChunkView<'a>>,
}

#[derive(Clone, Copy, Debug)]
pub struct BrickView<'a> {
    pub coord: BrickCoord,
    pub mesh: Option<(MeshHeader, &'a [u8], &'a [u8])>,
    pub samples: Option<(SampleHeader, &'a [u8], &'a [u8])>,
    /// Raw GPU blobs: vertex bytes (24 B each) and index bytes, both 256-aligned in the file.
    pub raw: Option<(&'a [u8], &'a [u8])>,
}

#[derive(Clone, Copy, Debug)]
pub struct ChunkView<'a> {
    pub header: InstHeader,
    pub rans: &'a [u8],
    pub bits: &'a [u8],
}

fn bad(table: &'static str, field: u16) -> VerifyError {
    VerifyError::BadField { table, field }
}

impl<'a> File<'a> {
    pub fn open(buf: &'a [u8], limits: Limits) -> Result<File<'a>, VerifyError> {
        let mut v = Verifier::new(buf, limits);
        let r = v.root()?;
        v.scalar(&r, root::MAJOR, 2, 2)?;
        v.scalar(&r, root::MINOR, 2, 2)?;
        v.scalar(&r, root::VOXEL_M, 8, 8)?;
        v.scalar(&r, root::BRICK_SIZE, 2, 2)?;
        let major = r.u16(root::MAJOR, 1);
        if major != MAJOR {
            return Err(VerifyError::UnsupportedVersion { major });
        }
        let grid = BrickGrid::new(r.u16(root::BRICK_SIZE, 32) as u32, r.f64(root::VOXEL_M, 0.5))
            .map_err(|_| bad("Root", root::BRICK_SIZE))?;
        let mb = v.bytes(&r, root::MODELS, 2, 4)?.ok_or(bad("Root", root::MODELS))?;
        if mb.len() < BASE_MODELS * MAX_ALPHABET * 2 {
            return Err(bad("Root", root::MODELS));
        }
        // a newer writer may append models; this reader only uses the ones it knows, and an
        // older file lacks the newest ones (empty stand-ins: its chunks never reference them, and a
        // reference would fail to decode)
        let present = (mb.len() / (MAX_ALPHABET * 2)).min(NUM_MODELS);
        let mut models = Vec::with_capacity(NUM_MODELS);
        for k in 0..NUM_MODELS {
            let f: Vec<u16> = if k < present {
                mb[k * 128..k * 128 + 128].as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect()
            } else {
                vec![0; MAX_ALPHABET]
            };
            let m: Model = Model::from_freqs(&f).map_err(VerifyError::Payload)?;
            models.push(DecModel::new(&m));
        }
        let nb = v.tables(&r, root::BRICKS)?;
        let mut bricks = Vec::with_capacity(nb);
        for i in 0..nb {
            let pos = r.tables_get(root::BRICKS, i).ok_or(VerifyError::OutOfBounds { at: r.pos })?.pos;
            let t = v.table_at(pos, 1)?;
            v.scalar(&t, brick::KEY, KEY_BYTES, 8)?;
            let kb = t.struct_bytes(brick::KEY, KEY_BYTES).ok_or(bad("Brick", brick::KEY))?;
            let [x, y, z] = read_i64s::<3>(kb);
            let coord = BrickCoord { x, y, z, lod: kb[24] };
            if coord.lod > voxel::MAX_LOD + 4 || [x, y, z].iter().any(|c| c.unsigned_abs() > 1 << 40) {
                return Err(bad("Brick", brick::KEY));
            }
            let mesh = match v.table(&t, brick::MESH, 1)? {
                None => None,
                Some(m) => {
                    for (id, sz) in [
                        (meshb::VERTEX_COUNT, 4),
                        (meshb::INDEX_COUNT, 4),
                        (meshb::QBITS, 1),
                        (meshb::CONN, 1),
                        (meshb::NORMAL_BITS, 1),
                        (meshb::FLAGS, 1),
                        (meshb::FSHIFT, 1),
                    ] {
                        v.scalar(&m, id, sz, sz)?;
                    }
                    let h = MeshHeader {
                        vertex_count: m.u32(meshb::VERTEX_COUNT, 0),
                        index_count: m.u32(meshb::INDEX_COUNT, 0),
                        lod: coord.lod,
                        qbits: m.u8(meshb::QBITS, 0),
                        conn: m.u8(meshb::CONN, 1),
                        normal_bits: m.u8(meshb::NORMAL_BITS, 10),
                        flags: m.u8(meshb::FLAGS, 0),
                        fshift: m.u8(meshb::FSHIFT, 0),
                    };
                    if h.vertex_count > limits.max_vertices
                        || h.index_count > limits.max_indices
                        || !h.index_count.is_multiple_of(3)
                    {
                        return Err(VerifyError::TooLarge);
                    }
                    if h.qbits > 8 || h.conn > 1 || !(2..=16).contains(&h.normal_bits) || h.flags > 3 || h.fshift > 16 {
                        return Err(bad("MeshBlob", meshb::QBITS));
                    }
                    let rans = v.bytes(&m, meshb::RANS, 1, 1)?.unwrap_or(&[]);
                    let bits = v.bytes(&m, meshb::BITS, 1, 1)?.unwrap_or(&[]);
                    Some((h, rans, bits))
                }
            };
            let samples = match v.table(&t, brick::SAMPLES, 1)? {
                None => None,
                Some(s) => {
                    v.scalar(&s, sampleb::N, 2, 2)?;
                    v.scalar(&s, sampleb::FACE_MASK, 1, 1)?;
                    v.scalar(&s, sampleb::BAND, 4, 4)?;
                    let h = SampleHeader {
                        n: s.u16(sampleb::N, 0),
                        face_mask: s.u8(sampleb::FACE_MASK, 0),
                        band: s.u32(sampleb::BAND, 0),
                    };
                    if h.n == 0 || h.n as usize > samples::MAX_N || h.face_mask >= 64 || h.band > voxel::DQ_MAX as u32 {
                        return Err(bad("SampleBlob", sampleb::N));
                    }
                    let rans = v.bytes(&s, sampleb::RANS, 1, 1)?.unwrap_or(&[]);
                    let bits = v.bytes(&s, sampleb::BITS, 1, 1)?.unwrap_or(&[]);
                    Some((h, rans, bits))
                }
            };
            let raw = match v.table(&t, brick::RAW, 1)? {
                None => None,
                Some(rt) => {
                    let vb = v.bytes(&rt, raw::VERTICES, size_of::<VoxelVertex>(), 256)?.unwrap_or(&[]);
                    let ib = v.bytes(&rt, raw::INDICES, 4, 256)?.unwrap_or(&[]);
                    if vb.len() / 24 > limits.max_vertices as usize || ib.len() / 4 > limits.max_indices as usize {
                        return Err(VerifyError::TooLarge);
                    }
                    Some((vb, ib))
                }
            };
            bricks.push(BrickView { coord, mesh, samples, raw });
        }
        let nc = v.tables(&r, root::INST_CHUNKS)?;
        let mut chunks = Vec::with_capacity(nc);
        for i in 0..nc {
            let pos = r.tables_get(root::INST_CHUNKS, i).ok_or(VerifyError::OutOfBounds { at: r.pos })?.pos;
            let t = v.table_at(pos, 1)?;
            v.scalar(&t, instc::COUNT, 4, 4)?;
            v.scalar(&t, instc::ORIGIN, 24, 8)?;
            let count = t.u32(instc::COUNT, 0);
            if count > limits.max_instances {
                return Err(VerifyError::TooLarge);
            }
            let origin = t.struct_bytes(instc::ORIGIN, 24).map_or([0; 3], read_i64s::<3>);
            let rans = v.bytes(&t, instc::RANS, 1, 1)?.unwrap_or(&[]);
            let bits = v.bytes(&t, instc::BITS, 1, 1)?.unwrap_or(&[]);
            v.scalar(&t, instc::MAT_BITS, 1, 1)?;
            let mat_bits = t.u8(instc::MAT_BITS, 10);
            if mat_bits != 10 && mat_bits != 16 {
                return Err(bad("InstChunk", instc::MAT_BITS));
            }
            chunks.push(ChunkView { header: InstHeader { count, origin, mat_bits }, rans, bits });
        }
        Ok(File { buf, root: r, major, minor: r.u16(root::MINOR, 0), grid, models, bricks, chunks })
    }

    /// [`Self::open`] plus a full decode of every payload (what a loader that cannot afford a
    /// late failure would run once; streaming readers decode lazily instead).
    pub fn open_deep(buf: &'a [u8], limits: Limits) -> Result<File<'a>, VerifyError> {
        let f = Self::open(buf, limits)?;
        for i in 0..f.bricks.len() {
            f.decode_mesh(i).transpose().map_err(VerifyError::Payload)?;
            f.decode_payload(i).transpose().map_err(VerifyError::Payload)?;
        }
        for i in 0..f.chunks.len() {
            f.decode_instances(i).map_err(VerifyError::Payload)?;
        }
        Ok(f)
    }

    pub fn bricks(&self) -> &[BrickView<'a>] {
        &self.bricks
    }
    pub fn chunks(&self) -> &[ChunkView<'a>] {
        &self.chunks
    }

    pub fn decode_mesh(&self, i: usize) -> Option<Result<Decoded, CodecError>> {
        let (h, rans, bits) = self.bricks.get(i)?.mesh?;
        Some((|| {
            let mut d = Decoder::new(rans, bits, &self.models)?;
            let m = mesh::decode(&h, &mut d)?;
            d.finish()?;
            Ok(m)
        })())
    }

    /// **True density** of brick `i` (collision, SDF queries, AO, edits, re-meshing at another
    /// LOD / iso). A band-clamped payload is refused with [`CodecError::Banded`] without
    /// decoding: request a full payload or regenerate the brick.
    pub fn decode_samples(&self, i: usize) -> Option<Result<BrickSamples, CodecError>> {
        let (h, _, _) = self.bricks.get(i)?.samples?;
        if h.band != 0 {
            return Some(Err(CodecError::Banded));
        }
        Some(self.decode_payload(i)?.map(|p| match p {
            samples::SamplePayload::Full(s) => s,
            samples::SamplePayload::Banded(_) => unreachable!("band 0 decodes as Full"),
        }))
    }

    /// Decode either kind, typed by meaning.
    pub fn decode_payload(&self, i: usize) -> Option<Result<samples::SamplePayload, CodecError>> {
        let b = self.bricks.get(i)?;
        let (h, rans, bits) = b.samples?;
        Some((|| {
            let mut d = Decoder::new(rans, bits, &self.models)?;
            let s = samples::decode_payload(&h, b.coord, &mut d)?;
            d.finish()?;
            Ok(s)
        })())
    }

    /// Meshing view of brick `i` (full or banded payloads).
    pub fn decode_mesh_samples(&self, i: usize) -> Option<Result<samples::MeshSamples, CodecError>> {
        Some(self.decode_payload(i)?.map(samples::SamplePayload::for_meshing))
    }

    pub fn decode_instances(&self, i: usize) -> Result<Vec<Instance>, CodecError> {
        let c = self.chunks.get(i).ok_or(CodecError::Corrupt)?;
        let mut out = Vec::new();
        let mut d = Decoder::new(c.rans, c.bits, &self.models)?;
        inst::decode(&c.header, &mut d, &mut out)?;
        d.finish()?;
        Ok(out)
    }

    /// Zero-copy GPU view of a raw brick: `&[VoxelVertex]` / `&[u32]` straight out of the file
    /// bytes (requires the buffer itself to be 256-aligned, e.g. mmap or [`fb::AlignedBytes`]).
    pub fn raw_gpu(&self, i: usize) -> Option<(&'a [VoxelVertex], &'a [u32])> {
        let (v, idx) = self.bricks.get(i)?.raw?;
        Some((bytemuck::try_cast_slice(v).ok()?, bytemuck::try_cast_slice(idx).ok()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fb::AlignedBytes;
    use voxel::VoxelConfig;

    /// Closed octahedron around a brick-interior point, vertices on lattice edges (like MC output).
    fn octahedron() -> (BrickCoord, BrickMesh) {
        let c = BrickCoord { x: 3, y: -1, z: 2, lod: 1 };
        let grid = BrickGrid::new(32, 0.5).unwrap();
        let o = grid.brick_origin(c).map(|v| v * voxel::SUB);
        let cell = voxel::SUB * 2;
        let ctr = [o[0] + 10 * cell, o[1] + 7 * cell, o[2] + 12 * cell];
        let mut ipos = Vec::new();
        let mut dirs = Vec::new();
        for (axis, s) in [(0usize, 1i64), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)] {
            let mut p = ctr;
            p[axis] += s * (cell + 37 + axis as i64 * 11);
            ipos.push(p);
            dirs.push(axis as u8 | if s > 0 { 4 } else { 0 });
        }
        let tris = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]];
        let indices: Vec<u32> = tris.iter().flatten().copied().collect();
        let vertices = ipos
            .iter()
            .map(|p| {
                let d = [0, 1, 2].map(|k| (p[k] - ctr[k]) as f32);
                let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                VoxelVertex { pos: [0.0; 3], normal: d.map(|v| v / l) }
            })
            .collect();
        (c, BrickMesh { ipos, vertices, dirs, indices })
    }

    fn tiny_file() -> Vec<u8> {
        let (c, m) = octahedron();
        let mut w = Writer::new(BrickGrid::new(32, 0.5).unwrap());
        w.add_mesh(c, &m, MeshOpts::default());
        w.add_raw(c, &m);
        w.add_instances(&[
            Instance { pos: [1.0, 2.0, 3.0], yaw: 0.5, height: 12.0, width: 4.0, mesh: 1, material: 7 },
            Instance { pos: [9.5, 2.5, -3.0], yaw: 6.0, height: 20.0, width: 6.0, mesh: 2, material: 1000 },
        ]);
        w.finish()
    }

    #[test]
    fn tiny_round_trip() {
        let (c, m) = octahedron();
        let buf = AlignedBytes::from_slice(&tiny_file());
        let f = File::open_deep(buf.as_slice(), Limits::default()).unwrap();
        let cfg = VoxelConfig::new(f.grid);
        let d = f.decode_mesh(0).unwrap().unwrap().to_brick_mesh(&cfg, c, 0);
        assert_eq!(mesh::canonical_tris(&d.ipos, &d.indices), mesh::canonical_tris(&m.ipos, &m.indices));
        assert_eq!(f.decode_instances(0).unwrap().len(), 2);
        let (v, i) = f.raw_gpu(0).unwrap();
        assert_eq!((v.len(), i), (6, &m.indices[..]));
    }

    #[test]
    fn tiny_mutations_never_panic() {
        let good = tiny_file();
        let n = if cfg!(miri) { 120 } else { 3000 };
        let mut s = 0x1234_5678u64;
        for _ in 0..n {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let mut b = good.clone();
            let p = (s >> 33) as usize % b.len();
            b[p] ^= 1 << ((s >> 20) & 7);
            if s & 1 == 0 {
                b.truncate((s >> 40) as usize % (b.len() + 1));
            }
            let _ = File::open_deep(&b, Limits::default());
        }
    }
}
