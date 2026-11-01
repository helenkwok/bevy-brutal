//! `asset`: the on-disk / streaming format for terrain bricks (quantised density samples),
//! meshed brick geometry and vegetation instances.
//!
//! - [`fb`]: a FlatBuffers-style container (offset-based, aligned, little-endian, vtables with
//!   optional fields, read in place from mmap / staging memory, plus a bounds-checking verifier).
//! - [`rans`]: static-model rANS entropy coder (Duda 2013) with a raw-bit side stream.
//! - [`mesh`]: Draco-style brick-mesh compression (lossless integer positions with lattice-aware
//!   parallelogram prediction, Edgebreaker-like connectivity, predicted octahedral normals).
//! - [`samples`]: Lorenzo-predicted lossless density samples.
//! - [`inst`]: quantised, Morton-sorted, delta-predicted instances.
//! - [`schema`]: the `BRK1` schema, [`schema::Writer`] and the verified zero-copy [`schema::File`].
//!
//! See `starter.md`, "Learning from FlatBuffers and Draco", for design and measurements.

pub mod fb;
pub mod inst;
pub mod mesh;
pub mod rans;
pub mod samples;
pub mod schema;
pub mod tq;

pub use fb::{AlignedBytes, Limits, VerifyError};
pub use inst::Instance;
pub use mesh::{ConnMode, MeshOpts};
pub use schema::{File, Writer};
