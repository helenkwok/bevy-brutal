//! Asset-format tests on a real Alpine brick world (the mountains camera): lossless round trips,
//! quantum error bound, seam bit-identity, closed topology after decode, determinism, schema
//! evolution, zero-copy GPU views, and a fuzzed verifier.
use asset::fb::{self, AlignedBytes, Field, Limits, TableB};
use asset::mesh::{self, ConnMode, MeshOpts};
use asset::schema::{self, File, Writer};
use asset::{Instance, VerifyError};
use coords::{BrickCoord, BrickGrid};
use std::collections::HashMap;
use std::sync::OnceLock;
use voxel::fields::Alpine;
use voxel::*;

struct World {
    cfg: VoxelConfig,
    world: VoxelWorld,
    layout: ShellLayout,
}

fn world() -> &'static World {
    static W: OnceLock<World> = OnceLock::new();
    W.get_or_init(|| {
        let field = Alpine::new(7);
        let grid = BrickGrid::new(32, 0.5).unwrap();
        let mut cfg = VoxelConfig::new(grid);
        let (cx, cz) = (7700.0, -4200.0);
        let gy = field.height(cx, cz, 0.5);
        cfg.set_origin([cx, gy + 40.0, cz]);
        let l = |v: f64| (v / grid.voxel_m()).floor() as i64;
        let layout = ShellLayout::camera_centred(&grid, [l(cx), l(gy), l(cz)], &[1, 1]);
        let b0 = layout.boxes[0];
        let coarse = 128;
        let al = |v: i64, up: bool| if up { (v + coarse - 1).div_euclid(coarse) * coarse } else { v.div_euclid(coarse) * coarse };
        let region = ([al(b0.lo[0], false), al(l(gy) - 96, false), al(b0.lo[2], false)], [al(b0.hi[0], true), al(l(gy) + 96, true), al(b0.hi[2], true)]);
        let world = build_world_filtered(&field, &cfg, &layout, region, layout.max_lod()).unwrap();
        assert!(world.bricks.len() >= 8, "test world has {} bricks", world.bricks.len());
        World { cfg, world, layout }
    })
}

fn write(w: &World, opts: MeshOpts) -> Vec<u8> {
    let mut wr = Writer::new(w.cfg.grid);
    wr.add_meshes(w.world.bricks.iter().map(|(c, m)| (*c, m)), opts);
    wr.finish()
}

fn decode_all(f: &File, cfg: &VoxelConfig, q: u8) -> Vec<(BrickCoord, BrickMesh)> {
    (0..f.bricks().len()).map(|i| (f.bricks()[i].coord, f.decode_mesh(i).unwrap().unwrap().to_brick_mesh(cfg, f.bricks()[i].coord, q))).collect()
}

fn angle_deg(a: [f32; 3], b: [f32; 3]) -> f64 {
    let d: f64 = (0..3).map(|k| a[k] as f64 * b[k] as f64).sum();
    let la: f64 = a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let lb: f64 = b.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    (d / (la * lb)).clamp(-1.0, 1.0).acos().to_degrees()
}

#[test]
fn a1_lossless_round_trip_both_connectivity_modes() {
    let w = world();
    for conn in [ConnMode::Traversal, ConnMode::Sequential] {
        let buf = AlignedBytes::from_slice(&write(w, MeshOpts { conn, ..Default::default() }));
        let f = File::open(buf.as_slice(), Limits::default()).unwrap();
        assert_eq!(f.bricks().len(), w.world.bricks.len());
        let mut worst = 0.0f64;
        for (c, d) in decode_all(&f, &w.cfg, 0) {
            let o = &w.world.bricks[&c];
            assert_eq!(d.ipos.len(), o.ipos.len());
            assert_eq!(mesh::canonical_tris(&d.ipos, &d.indices), mesh::canonical_tris(&o.ipos, &o.indices), "{c:?} {conn:?}");
            if conn == ConnMode::Sequential {
                assert_eq!(d.indices, o.indices, "sequential keeps the exact index buffer");
                assert_eq!(d.ipos, o.ipos);
                assert_eq!(d.dirs, o.dirs);
            }
            let orig: HashMap<[i64; 3], (usize, u8)> = o.ipos.iter().enumerate().map(|(i, p)| (*p, (i, o.dirs[i]))).collect();
            for (i, p) in d.ipos.iter().enumerate() {
                let (oi, od) = orig[p];
                assert_eq!(d.dirs[i], od);
                assert_eq!(d.vertices[i].pos.map(f32::to_bits), o.vertices[oi].pos.map(f32::to_bits), "f32 render position bit-identical");
                worst = worst.max(angle_deg(d.vertices[i].normal, o.vertices[oi].normal));
            }
        }
        assert!(worst < 0.25, "10-bit octahedral normal error {worst} deg");
    }
}

#[test]
fn a2_decoded_world_is_still_closed() {
    let w = world();
    let buf = write(w, MeshOpts::default());
    let f = File::open(&buf, Limits::default()).unwrap();
    let dec = decode_all(&f, &w.cfg, 0);
    let t0 = topology(w.world.bricks.values());
    let t1 = topology(dec.iter().map(|(_, m)| m));
    assert!(t0.is_clean() && t1.is_clean());
    assert_eq!((t0.boundary_edges, t0.non_manifold_edges, t0.orientation_conflicts), (t1.boundary_edges, t1.non_manifold_edges, t1.orientation_conflicts));
    let _ = &w.layout;
}

#[test]
fn a3_seams_bit_identical_and_quantum_error_bounded() {
    let w = world();
    for q in [0u8, 2, 4] {
        let buf = write(w, MeshOpts { qbits: q, ..Default::default() });
        let f = File::open(&buf, Limits::default()).unwrap();
        let half = if q == 0 { 0 } else { 1i64 << (q - 1) };
        // original global point -> decoded (f32 bits, ipos) seen by each brick
        let mut seen: HashMap<[i64; 3], ([u32; 3], [i64; 3], usize)> = HashMap::new();
        let mut shared = 0usize;
        for (i, (c, d)) in decode_all(&f, &w.cfg, q).into_iter().enumerate() {
            let o = &w.world.bricks[&c];
            // the decoder emits vertices in its own order: map back through the quantised point
            let mut by_q: HashMap<[i64; 3], usize> = HashMap::new();
            for (k, p) in d.ipos.iter().enumerate() {
                by_q.insert(*p, k);
            }
            for p in &o.ipos {
                let qp = mesh::quantise_global(*p, q);
                let k = by_q[&qp];
                for (dq, pa) in d.ipos[k].iter().zip(p) {
                    assert!((dq - pa).abs() <= half, "q{q}: error {} > {half}", dq - pa);
                }
                let bits = d.vertices[k].pos.map(f32::to_bits);
                match seen.get(p) {
                    Some(&(b, ip, j)) if j != i => {
                        shared += 1;
                        assert_eq!((b, ip), (bits, d.ipos[k]), "q{q}: seam vertex {p:?} differs between bricks");
                    }
                    _ => {
                        seen.insert(*p, (bits, d.ipos[k], i));
                    }
                }
            }
        }
        assert!(shared > 1000, "seams exercised ({shared} shared vertices)");
    }
}

#[test]
fn a4_deterministic_and_parallel_equals_serial() {
    let w = world();
    let a = write(w, MeshOpts::default());
    let b = write(w, MeshOpts::default());
    assert_eq!(a, b);
    let mut s = Writer::new(w.cfg.grid);
    for (c, m) in &w.world.bricks {
        s.add_mesh(*c, m, MeshOpts::default());
    }
    assert_eq!(s.finish(), a, "rayon pass-1 encode is order-preserving");
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    assert_eq!(pool.install(|| write(w, MeshOpts::default())), a);
}

#[test]
fn a5_samples_lossless() {
    let w = world();
    let field = Alpine::new(7);
    let mut wr = Writer::new(w.cfg.grid);
    let coords: Vec<BrickCoord> = w.world.bricks.keys().copied().take(6).collect();
    let mut orig = Vec::new();
    for c in &coords {
        let (mask, _) = plan_brick(&w.cfg.grid, &w.layout, *c).unwrap();
        let s = sample_brick_smooth(&field, &w.cfg, &w.layout, *c, mask).unwrap();
        wr.add_samples(&s);
        orig.push(s);
    }
    let buf = wr.finish();
    let f = File::open(&buf, Limits::default()).unwrap();
    for (i, s) in orig.iter().enumerate() {
        let d = f.decode_samples(i).unwrap().unwrap();
        assert_eq!((d.coord, d.n), (s.coord, s.n));
        assert_eq!(d.d, s.d);
        assert_eq!(d.faces, s.faces);
    }
    let raw: usize = orig.iter().map(|s| s.d.len() * 4 + s.faces.iter().map(|f| f.len() * 4).sum::<usize>()).sum();
    assert!(buf.len() * 3 < raw, "samples compress at least 3:1 ({} vs {raw})", buf.len());
}

fn rand_instances(n: usize, seed: u64) -> Vec<Instance> {
    let mut s = seed;
    let mut r = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let h = 6.0 + 22.0 * r();
            Instance {
                pos: [7000.0 + 900.0 * r(), 600.0 + 300.0 * r(), -4800.0 + 900.0 * r()],
                yaw: (r() * std::f64::consts::TAU) as f32,
                height: h as f32,
                width: (h * (0.3 + 0.15 * r())) as f32,
                mesh: (r() * 3.0) as u8,
                material: (r() * 1023.0) as u16,
            }
        })
        .collect()
}

#[test]
fn a6_instances_bounded_error() {
    let src = rand_instances(5000, 3);
    let mut wr = Writer::new(BrickGrid::new(32, 0.5).unwrap());
    wr.add_instances(&src);
    let buf = wr.finish();
    let f = File::open(&buf, Limits::default()).unwrap();
    let mut out = f.decode_instances(0).unwrap();
    assert_eq!(out.len(), src.len());
    // match by (mesh, material, nearest position)
    let key = |i: &Instance| ((i.pos[0] * 128.0).round() as i64, (i.pos[2] * 128.0).round() as i64, i.mesh, i.material);
    let mut a = src.clone();
    a.sort_by_key(key);
    out.sort_by_key(key);
    for (s, d) in a.iter().zip(&out) {
        for k in 0..3 {
            assert!((s.pos[k] - d.pos[k]).abs() <= 0.5 / 128.0 + 1e-9);
        }
        let dy = ((s.yaw - d.yaw) as f64).rem_euclid(std::f64::consts::TAU);
        assert!(dy.min(std::f64::consts::TAU - dy) <= std::f64::consts::PI / 4096.0 + 1e-6);
        assert!((d.height / s.height - 1.0).abs() < 0.0055, "height within half a 1/64-octave step (2^(1/128) - 1)");
        assert!((d.width / s.width - 1.0).abs() < 0.013);
        assert_eq!((s.mesh, s.material), (d.mesh, d.material));
    }
    let raw = src.len() * size_of::<lowlevel_abi::InstanceGpu>();
    assert!(buf.len() * 6 < raw, "uniformly random synthetic instances (real ones: see the bench) {} B vs {raw} B GPU", buf.len());
}

#[test]
fn a7_schema_evolution_both_directions() {
    let w = world();
    let (c, m) = w.world.bricks.iter().next().unwrap();
    // newer writer (extra fields in every table, minor + 1): this reader ignores them
    let mut wr = Writer::new(w.cfg.grid);
    wr.test_future_fields = true;
    wr.add_mesh(*c, m, MeshOpts::default());
    wr.add_raw(*c, m);
    let buf = AlignedBytes::from_slice(&wr.finish());
    let f = File::open_deep(buf.as_slice(), Limits::default()).unwrap();
    assert_eq!((f.major, f.minor), (schema::MAJOR, schema::MINOR + 1));
    let d = f.decode_mesh(0).unwrap().unwrap().to_brick_mesh(&w.cfg, *c, 0);
    assert_eq!(mesh::canonical_tris(&d.ipos, &d.indices), mesh::canonical_tris(&m.ipos, &m.indices));
    // older writer: no minor, no inst_chunks, no samples, mesh blob without normal_bits
    // (defaults apply); built by hand with the container API
    let (h, log) = mesh::encode(&w.cfg.grid, *c, m, MeshOpts::default());
    let mut counts = asset::rans::Counts::new(schema::NUM_MODELS);
    counts.add_log(&log);
    let models = counts.models();
    let mut mb = Vec::new();
    for md in &models {
        for v in md.freq {
            mb.extend_from_slice(&v.to_le_bytes());
        }
    }
    let mut key = Vec::new();
    for v in [c.x, c.y, c.z] {
        key.extend_from_slice(&v.to_le_bytes());
    }
    key.push(c.lod);
    key.resize(32, 0);
    let blob = TableB::default()
        .with(schema::meshb::VERTEX_COUNT, Field::U32(h.vertex_count))
        .with(schema::meshb::INDEX_COUNT, Field::U32(h.index_count))
        .with(schema::meshb::FLAGS, Field::U8(h.flags))
        .with(schema::meshb::RANS, Field::Bytes(log.encode(&models), 4))
        .with(schema::meshb::BITS, Field::Bytes(log.bits.finished(), 4));
    let brick = TableB::default().with(schema::brick::KEY, Field::Struct(key, 8)).with(schema::brick::MESH, Field::Table(Box::new(blob)));
    let root = TableB::default()
        .with(schema::root::VOXEL_M, Field::F64(0.5))
        .with(schema::root::BRICK_SIZE, Field::U16(32))
        .with(schema::root::MODELS, Field::Bytes(mb, 4))
        .with(schema::root::BRICKS, Field::Tables(vec![brick]));
    let old = fb::finish(&root);
    let f = File::open_deep(&old, Limits::default()).unwrap();
    assert_eq!((f.major, f.minor, f.chunks().len()), (1, 0, 0));
    assert!(f.bricks()[0].samples.is_none() && f.bricks()[0].raw.is_none());
    let d = f.decode_mesh(0).unwrap().unwrap();
    assert_eq!(d.indices.len(), m.indices.len(), "defaults (qbits 0, traversal, 10-bit normals) apply");
    // incompatible major version is rejected
    let mut wr = Writer::new(w.cfg.grid);
    wr.add_mesh(*c, m, MeshOpts::default());
    let mut buf = wr.finish();
    let r = fb::Table::root(&buf).unwrap();
    let p = r.field(schema::root::MAJOR).unwrap();
    buf[p..p + 2].copy_from_slice(&2u16.to_le_bytes());
    assert!(matches!(File::open(&buf, Limits::default()), Err(VerifyError::UnsupportedVersion { major: 2 })));
}

#[test]
fn a8_raw_blobs_are_zero_copy_gpu_views() {
    let w = world();
    let mut wr = Writer::new(w.cfg.grid);
    for (c, m) in w.world.bricks.iter().take(4) {
        wr.add_raw(*c, m);
    }
    let buf = AlignedBytes::from_slice(&wr.finish());
    let f = File::open(buf.as_slice(), Limits::default()).unwrap();
    for (i, (_, m)) in w.world.bricks.iter().take(4).enumerate() {
        let (v, idx) = f.raw_gpu(i).unwrap();
        assert_eq!(v.as_ptr() as usize % 256, 0);
        assert_eq!(idx.as_ptr() as usize % 256, 0);
        assert!(std::ptr::eq(v.as_ptr().cast::<u8>(), f.bricks()[i].raw.unwrap().0.as_ptr()), "borrowed, not copied");
        assert_eq!(v, &m.vertices[..]);
        assert_eq!(idx, &m.indices[..]);
    }
}

/// splitmix64
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn small_file() -> Vec<u8> {
    let w = world();
    let mut wr = Writer::new(w.cfg.grid);
    let mut it = w.world.bricks.iter().filter(|(_, m)| m.indices.len() < 3000);
    let (c, m) = it.next().unwrap();
    wr.add_mesh(*c, m, MeshOpts::default());
    let (c2, m2) = it.next().unwrap();
    wr.add_mesh(*c2, m2, MeshOpts { conn: ConnMode::Sequential, qbits: 1, normal_bits: 8 });
    wr.add_raw(*c2, m2);
    wr.add_instances(&rand_instances(60, 9));
    wr.finish()
}

/// Mutation fuzzing of the whole read path: open (structure + limits) and a deep decode of
/// every payload must never panic, loop or over-allocate, whatever the bytes. Flips in
/// compressed payloads are detected by the rANS final-state / exact-length checks.
#[test]
fn a9_verifier_fuzz_never_panics() {
    let good = small_file();
    assert!(File::open_deep(&good, Limits::default()).is_ok());
    let n = std::env::var("ASSET_FUZZ_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(4000);
    let mut rng = Rng(0xB0B);
    let (mut rejected, mut payload_flips, mut payload_caught) = (0usize, 0usize, 0usize);
    let limits = Limits { max_vertices: 1 << 16, max_indices: 1 << 18, max_instances: 1 << 14, ..Limits::default() };
    for it in 0..n {
        let mut b = good.clone();
        let kind = it % 6;
        match kind {
            0 => {
                for _ in 0..1 + rng.below(4) {
                    let p = rng.below(b.len());
                    b[p] ^= 1 << rng.below(8);
                }
            }
            1 => {
                let p = rng.below(b.len());
                b[p] = [0, 0xFF, 0x7F, 0x80][rng.below(4)];
            }
            2 => b.truncate(rng.below(b.len())),
            3 => {
                // overwrite an aligned u32 (offsets, counts, soffsets) with a hostile value
                let p = rng.below(b.len() / 4) * 4;
                let v: u32 = [0, 1, 4, 0xFFFF_FFFF, 0x8000_0000, b.len() as u32, (b.len() as u32).wrapping_sub(4), rng.next() as u32][rng.below(8)];
                b[p..p + 4].copy_from_slice(&v.to_le_bytes());
            }
            4 => {
                // splice: copy a random chunk over another place
                let (s, d, l) = (rng.below(b.len()), rng.below(b.len()), rng.below(64));
                let l = l.min(b.len() - s).min(b.len() - d);
                let chunk = b[s..s + l].to_vec();
                b[d..d + l].copy_from_slice(&chunk);
            }
            _ => {
                // flip inside a compressed mesh payload only (structure stays valid)
                let f = File::open(&good, Limits::default()).unwrap();
                let (_, rans, _) = f.bricks()[0].mesh.unwrap();
                let off = rans.as_ptr() as usize - good.as_ptr() as usize;
                let p = off + rng.below(rans.len());
                b[p] ^= 1 << rng.below(8);
                payload_flips += 1;
            }
        }
        let r = File::open_deep(&b, limits);
        if r.is_err() {
            rejected += 1;
            if kind == 5 {
                payload_caught += 1;
            }
        }
    }
    // pure garbage, and garbage behind a valid identifier
    for i in 0..n / 4 {
        let len = rng.below(4096);
        let mut b: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        if i % 2 == 0 && b.len() >= 8 {
            b[4..8].copy_from_slice(b"BRK1");
        }
        let _ = File::open_deep(&b, limits);
    }
    eprintln!("fuzz: {n} mutants, {rejected} rejected; payload bit flips caught {payload_caught}/{payload_flips}");
    assert!(rejected > n / 2);
    assert!(payload_caught * 10 >= payload_flips * 9, "rANS integrity checks catch >= 90 % of payload flips");
}

/// Minor 1: full 16-bit instance materials (species / class tags above the 10 random bits)
/// round-trip exactly; a pre-minor-1 chunk (no `mat_bits` field, 10-bit materials) in the same
/// file still decodes, and a bad `mat_bits` value is rejected by the verifier.
#[test]
fn a10_instance_materials_16_bit_and_legacy_chunks() {
    let w = world();
    let mut s = 0x9e37_79b9_u64;
    let mut r = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let items: Vec<Instance> = (0..5000)
        .map(|_| Instance {
            pos: [7000.0 + 900.0 * r(), 600.0 + 300.0 * r(), -4800.0 + 900.0 * r()],
            yaw: (r() * std::f64::consts::TAU) as f32,
            height: (6.0 + 22.0 * r()) as f32,
            width: (3.0 + 4.0 * r()) as f32,
            mesh: (r() * 6.0) as u8,
            // skewed tag (like species) above 10 random bits
            material: ((r() * 1023.0) as u16) | (([0u16, 0, 0, 1, 1, 2, 9, 63][(r() * 8.0) as usize % 8]) << 10),
        })
        .collect();
    let mut wr = Writer::new(w.cfg.grid);
    wr.add_instances(&items);
    let legacy: Vec<Instance> = items.iter().map(|i| Instance { material: i.material & 1023, ..*i }).collect();
    wr.add_instances_legacy(&legacy);
    let buf = wr.finish();
    let f = File::open(&buf, Limits::default()).unwrap();
    assert_eq!(f.chunks().len(), 2);
    assert_eq!((f.chunks()[0].header.mat_bits, f.chunks()[1].header.mat_bits), (16, 10));
    for (k, src) in [&items, &legacy].into_iter().enumerate() {
        let mut a: Vec<(u8, u16)> = src.iter().map(|i| (i.mesh, i.material)).collect();
        let mut b: Vec<(u8, u16)> = f.decode_instances(k).unwrap().iter().map(|i| (i.mesh, i.material)).collect();
        a.sort();
        b.sort();
        assert_eq!(a, b, "chunk {k}: materials round-trip exactly");
    }
    // corrupt mat_bits -> verifier error, never a misdecode
    let r0 = fb::Table::root(&buf).unwrap();
    let c0 = r0.tables_get(schema::root::INST_CHUNKS, 0).unwrap();
    let p = c0.field(schema::instc::MAT_BITS).unwrap();
    let mut bad = buf.clone();
    bad[p] = 12;
    assert!(File::open(&bad, Limits::default()).is_err());
}

/// Format minor 2, band-clamped sample payloads: a golden minor-1 file (written by the cp35
/// code) still reads as full density; banded payloads are refused as density
/// (`CodecError::Banded`), mesh bit-identically through `MeshSamples`, and mislabelled payloads
/// are rejected.
#[test]
fn a11_banded_samples_format_and_type_split() {
    use asset::rans::CodecError;
    use voxel::scene::BlobScene;
    // golden minor-1 file
    let gold = AlignedBytes::from_slice(include_bytes!("data/minor1_samples.brk"));
    let f = File::open_deep(gold.as_slice(), Limits::default()).unwrap();
    assert_eq!((f.major, f.minor), (1, 1));
    let cfg = VoxelConfig::new(f.grid);
    let lay = ShellLayout::camera_centred(&cfg.grid, [0, 0, 0], &[1]);
    let field = BlobScene { radius: 6.0, seed: 3 };
    assert_eq!(f.bricks().len(), 8);
    for (i, b) in f.bricks().iter().enumerate() {
        let (mask, finer) = plan_brick(&cfg.grid, &lay, b.coord).unwrap();
        let s = sample_brick(&field, &cfg, b.coord, mask).unwrap();
        let d = f.decode_samples(i).unwrap().unwrap();
        assert_eq!((d.d.clone(), d.faces.clone()), (s.d.clone(), s.faces.clone()), "minor-1 samples must read unchanged");
        let m = f.decode_mesh_samples(i).unwrap().unwrap();
        assert_eq!((m.band(), m.transitions()), (0, mask));
        assert_eq!(m.mesh(&cfg, finer), mesh_brick(&cfg, &s, mask, finer));
        assert!(m.into_density().is_some());
    }
    // the same bricks written by this writer: full = minor-1 payload bytes, banded = mesh-only
    let mut full = Writer::new(cfg.grid);
    let mut banded = Writer::new(cfg.grid);
    let mut bad = Writer::new(cfg.grid);
    let mut src = Vec::new();
    for b in f.bricks() {
        let (mask, finer) = plan_brick(&cfg.grid, &lay, b.coord).unwrap();
        let s = sample_brick(&field, &cfg, b.coord, mask).unwrap();
        full.add_samples(&s);
        banded.add_samples_banded(&s, 1024);
        bad.add_samples_mislabelled(&s, 1024);
        src.push((s, mask, finer));
    }
    let (fb_, bb) = (AlignedBytes::from_slice(&full.finish()), AlignedBytes::from_slice(&banded.finish()));
    let ff = File::open_deep(fb_.as_slice(), Limits::default()).unwrap();
    let fbd = File::open_deep(bb.as_slice(), Limits::default()).unwrap();
    assert_eq!(ff.minor, schema::MINOR);
    assert_eq!(schema::MINOR, 2);
    for (i, (s, mask, finer)) in src.iter().enumerate() {
        assert_eq!(ff.decode_samples(i).unwrap().unwrap().d, s.d);
        assert_eq!(fbd.bricks()[i].samples.unwrap().0.band, 1024);
        assert!(matches!(fbd.decode_samples(i).unwrap(), Err(CodecError::Banded)), "banded payload must not read as density");
        let m = fbd.decode_mesh_samples(i).unwrap().unwrap();
        assert_eq!((m.band(), m.transitions()), (1024, *mask));
        assert_eq!(m.mesh(&cfg, *finer), mesh_brick(&cfg, s, *mask, *finer), "banded mesh must be bit-identical");
        assert!(m.into_density().is_none());
    }
    let changed = src.iter().enumerate().filter(|(i, (s, _, _))| fbd.decode_mesh_samples(*i).unwrap().unwrap().samples_for_meshing_only().d != s.d).count();
    assert!(changed > 0, "the band must actually clamp something");
    // full data labelled as banded is not a fixed point of the band: rejected
    let bad = AlignedBytes::from_slice(&bad.finish());
    assert!(File::open_deep(bad.as_slice(), Limits::default()).is_err());
    let fbad = File::open(bad.as_slice(), Limits::default()).unwrap();
    assert!((0..fbad.bricks().len()).any(|i| fbad.decode_mesh_samples(i).unwrap().is_err()));
}

/// The band covers everything regular cells, Transvoxel transition cells and LOD seams read:
/// on multi-LOD shell layouts (Alpine mountains camera, a blob scene and the adversarial fuzz
/// field, 3 LOD levels) every brick, with transitions or not, meshes bit-identically from the
/// banded file, and the banded world is still watertight across LOD seams.
#[test]
fn a12_band_bit_identical_across_lods_and_transitions() {
    use voxel::scene::{Adversarial, BlobScene};
    let w = world();
    let grid16 = BrickGrid::new(16, 0.25).unwrap();
    let cfg16 = VoxelConfig::new(grid16);
    type Case<'a> = (&'a str, &'a VoxelConfig, Box<dyn DensityField>, ShellLayout, ([i64; 3], [i64; 3]));
    let cases: Vec<Case<'_>> = vec![
        ("alpine", &w.cfg, Box::new(Alpine::new(7)), w.layout.clone(), {
            w.world.bricks.keys().fold(([i64::MAX; 3], [i64::MIN; 3]), |(lo, hi), c| {
                let o = w.cfg.grid.brick_origin(*c);
                let e = 32 * w.cfg.grid.step(c.lod);
                ([0, 1, 2].map(|k| lo[k].min(o[k])), [0, 1, 2].map(|k| hi[k].max(o[k] + e)))
            })
        }),
        ("blob", &cfg16, Box::new(BlobScene { radius: 9.0, seed: 3 }), ShellLayout::camera_centred(&grid16, [5, -3, 7], &[1, 1, 1]), ([-128; 3], [128; 3])),
        ("fuzz", &cfg16, Box::new(Adversarial { seed: 4, voxel_m: 0.25, radius_m: 7.0 }), ShellLayout::camera_centred(&grid16, [20, 4, -6], &[1, 1]), ([-128; 3], [128; 3])),
    ];
    for (name, cfg, field, lay, region) in &cases {
        let mut src = Vec::new();
        for bc in bricks_in(&cfg.grid, lay, *region, lay.max_lod()) {
            let (mask, finer) = plan_brick(&cfg.grid, lay, bc).unwrap();
            let s = sample_brick_smooth(field.as_ref(), cfg, lay, bc, mask).unwrap();
            if s.has_sign_change() {
                src.push((s, mask, finer));
            }
        }
        let mut wr = Writer::new(cfg.grid);
        for (s, _, _) in &src {
            wr.add_samples_banded(s, 1024);
        }
        let buf = AlignedBytes::from_slice(&wr.finish());
        let f = File::open_deep(buf.as_slice(), Limits::default()).unwrap();
        let (mut with_t, mut lods) = (0, std::collections::BTreeSet::new());
        let mut meshes = Vec::new();
        for (i, (s, mask, finer)) in src.iter().enumerate() {
            let m = f.decode_mesh_samples(i).unwrap().unwrap();
            assert_eq!(m.transitions(), *mask, "{name}");
            let got = m.mesh(cfg, *finer);
            assert_eq!(got, mesh_brick(cfg, s, *mask, *finer), "{name}: brick {:?} mask {mask:#x}", s.coord);
            with_t += (*mask != 0) as usize;
            lods.insert(s.coord.lod);
            meshes.push(got);
        }
        // coverage: the synthetic scenes guarantee >= 3 transition bricks; on the real terrain
        // the count is terrain luck (3 -> 2 when the round-10 outcrop fix reshaped a tor in the
        // fixture box), so there only require that transitions and both LODs are exercised
        let min_t = if *name == "alpine" { 1 } else { 3 };
        assert!(with_t >= min_t && lods.len() >= 2, "{name}: {} bricks, {with_t} with transitions, lods {lods:?}", src.len());
        let t = topology(meshes.iter());
        let full: Vec<BrickMesh> = src.iter().map(|(s, m, f)| mesh_brick(cfg, s, *m, *f)).collect();
        assert_eq!(format!("{t:?}"), format!("{:?}", topology(full.iter())), "{name}");
        // the Alpine test region cuts the terrain open at its outer box; the blob scenes are closed
        if *name != "alpine" {
            assert!(t.boundary_edges == 0 && t.non_manifold_edges == 0 && t.orientation_conflicts == 0, "{name}: {t:?}");
        }
        assert!(t.non_manifold_edges == 0 && t.orientation_conflicts == 0, "{name}: {t:?}");
        eprintln!("{name}: {} bricks ({with_t} with transitions, LODs {lods:?}) bit-identical from the banded file; topology equal to full", src.len());
    }
}
