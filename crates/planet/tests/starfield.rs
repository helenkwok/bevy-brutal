//! RS1 starfield tests. All stars here are SYNTHETIC TEST FIXTURES (deterministic hash noise),
//! used only to exercise merge/quantisation/selection logic; they are never written to data dirs.
use glam::DVec3;
use planet::gcns::PARSEC_M;
use planet::starfield::*;
use planet::universe::Camera;

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn lut() -> Vec<[f32; 4]> {
    (0..255).map(|i| [1.0, 1.0 - i as f32 / 600.0, 1.0 - i as f32 / 300.0, i as f32]).collect()
}

fn fixture_cloud(n: usize, seed: u64, radius: f64) -> Vec<Merged> {
    let mut r = rng(seed);
    (0..n)
        .map(|i| {
            let p = loop {
                let v = DVec3::new(r() * 2.0 - 1.0, r() * 2.0 - 1.0, r() * 2.0 - 1.0);
                if v.length() <= 1.0 && v.length() > 1e-3 {
                    break v * radius;
                }
            };
            Merged {
                src: (i % 3) as u8,
                id: 1 + i as u64,
                pos_pc: p,
                gmag: (2.0 + r() * 14.0) as f32,
                bp_rp: if i % 17 == 0 { f32::NAN } else { (r() * 4.0 - 0.3) as f32 },
                flags: 0,
            }
        })
        .collect()
}

fn decoded_all(idx: &TileIndex, img: &[u8]) -> Vec<Decoded> {
    let mut stream = TileStream::new(img.to_vec(), u64::MAX).unwrap();
    let (plan, _) = stream.update(DVec3::ZERO, u64::MAX).unwrap();
    assert_eq!(plan.stars, idx.header.star_count);
    let mut out = Vec::new();
    for n in &plan.nodes {
        decode_block(&stream.index.nodes[*n as usize], &stream.cache[n], &mut out);
    }
    out
}

#[test]
fn quantisation_round_trip_error_bounds() {
    let stars = fixture_cloud(30_000, 7, 1000.0);
    let img = build_tiles(&stars, &lut(), &BuildParams { node_capacity: 512, ..Default::default() }).unwrap();
    let idx = TileIndex::open(&img).unwrap();
    assert_eq!(idx.header.star_count, 30_000);
    let stream = {
        let mut s = TileStream::new(img.clone(), u64::MAX).unwrap();
        s.update(DVec3::ZERO, u64::MAX).unwrap();
        s
    };
    let by_id: std::collections::HashMap<(u8, u64), &Merged> = stars.iter().map(|s| ((s.src, s.id), s)).collect();
    let mut worst_pos_ratio = 0.0f64;
    let mut worst_mag = 0.0f64;
    let mut seen = 0;
    for (n, b) in &stream.cache {
        let nd = &idx.nodes[*n as usize];
        let bound = pos_error_bound(nd);
        let mut d = Vec::new();
        decode_block(nd, b, &mut d);
        for s in d {
            let o = by_id[&(s.src, s.id)];
            let err = (s.pos_pc - o.pos_pc).abs().max_element();
            assert!(err <= bound, "pos err {err} > bound {bound}");
            worst_pos_ratio = worst_pos_ratio.max(err / bound);
            let me = (s.abs_mag - o.abs_mag()).abs();
            assert!(me <= 0.5 * MAG_SCALE as f64 + 1e-9, "mag err {me}");
            worst_mag = worst_mag.max(me);
            match colour_value(s.colour, -0.6, 5.0) {
                None => assert!(o.bp_rp.is_nan()),
                Some(c) => assert!((c - o.bp_rp.clamp(-0.6, 5.0)).abs() <= 0.5 * 5.6 / 254.0 + 1e-6),
            }
            assert!(s.pos_pc.distance(o.pos_pc) * PARSEC_M / planet::universe::AU < 1.0, "sub-AU at 1 kpc root");
            seen += 1;
        }
    }
    assert_eq!(seen, 30_000);
    assert!(worst_pos_ratio > 0.0 && worst_mag > 0.0);
}

#[test]
fn octree_brightness_priority_invariants() {
    let stars = fixture_cloud(20_000, 11, 500.0);
    let img = build_tiles(&stars, &lut(), &BuildParams { node_capacity: 256, ..Default::default() }).unwrap();
    let idx = TileIndex::open(&img).unwrap();
    for (i, nd) in idx.nodes.iter().enumerate() {
        for c in idx.children(i) {
            let ch = &idx.nodes[c];
            assert!(ch.m_own_min >= nd.m_own_max - 0.01, "child brighter than parent's faintest");
            assert!(nd.m_desc_min <= ch.m_own_min.min(ch.m_desc_min) + 1e-6);
            assert_eq!(ch.size, nd.size * 0.5);
        }
        if nd.child_mask != 0 {
            assert_eq!(nd.star_count, 256);
        }
    }
}

#[test]
fn selection_matches_brute_force() {
    let stars = fixture_cloud(40_000, 3, 1000.0);
    let img = build_tiles(&stars, &lut(), &BuildParams { node_capacity: 300, ..Default::default() }).unwrap();
    let idx = TileIndex::open(&img).unwrap();
    let all = decoded_all(&idx, &img);
    for (k, (cam, budget)) in [
        (DVec3::ZERO, 5_000u64),
        (DVec3::new(500.0, 0.0, 0.0), 5_000),
        (DVec3::new(-300.0, 700.0, 100.0), 1_000),
        (DVec3::new(1.0, 2.0, 3.0), 40_000),
        (DVec3::new(2000.0, 0.0, 0.0), 3_000),
    ]
    .into_iter()
    .enumerate()
    {
        let mut stream = TileStream::new(img.to_vec(), u64::MAX).unwrap();
        let (plan, st) = stream.update(cam, budget).unwrap();
        assert!(plan.stars <= budget.max(idx.nodes[0].star_count as u64));
        assert_eq!(st.nodes_planned, plan.nodes.len());
        let mut loaded = Vec::new();
        for n in &plan.nodes {
            decode_block(&stream.index.nodes[*n as usize], &stream.cache[n], &mut loaded);
        }
        let key = |d: &Decoded| (d.src, d.id);
        let near = |d: &Decoded| (d.pos_pc - cam).length() >= min_draw_pc();
        let mut got: Vec<_> =
            loaded.iter().filter(|d| near(d) && apparent_mag(d, cam) < plan.m_complete).map(key).collect();
        let mut want: Vec<_> =
            all.iter().filter(|d| near(d) && apparent_mag(d, cam) < plan.m_complete).map(key).collect();
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(got, want, "case {k}: streamed set != brute force at m_complete {}", plan.m_complete);
        if budget >= 40_000 {
            assert_eq!(plan.m_complete, f64::INFINITY, "full budget loads everything");
        }
        // Projection agrees with brute-force projection over the full catalogue.
        let c = Camera::look_at(cam * PARSEC_M, (cam + DVec3::X) * PARSEC_M, DVec3::Z, 1.2, 320, 200).unwrap();
        let (a, _) = project(&loaded, &c, plan.m_complete, 99.0, 1 << 20, &idx.lut);
        let (b, _) = project(&all, &c, plan.m_complete, 99.0, 1 << 20, &idx.lut);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.id, y.id);
            assert_eq!(x.center, y.center);
        }
        // GPU cap keeps exactly the brightest.
        let (capped, ps) = project(&loaded, &c, plan.m_complete, 99.0, 50, &idx.lut);
        assert_eq!(capped.len(), 50.min(a.len()));
        assert_eq!(ps.dropped_by_cap, a.len() - capped.len());
        for (x, y) in capped.iter().zip(&a) {
            assert_eq!(x.id, y.id);
        }
    }
}

#[test]
fn streaming_lru_eviction_and_reload() {
    let stars = fixture_cloud(20_000, 5, 1000.0);
    let img = build_tiles(&stars, &lut(), &BuildParams { node_capacity: 200, ..Default::default() }).unwrap();
    let mut s = TileStream::new(img, 3_000).unwrap();
    let (_, a) = s.update(DVec3::new(-800.0, 0.0, 0.0), 2_500).unwrap();
    assert!(a.nodes_loaded_now > 0);
    let (_, b) = s.update(DVec3::new(800.0, 0.0, 0.0), 2_500).unwrap();
    assert!(b.nodes_evicted_now > 0 && b.resident_stars <= 3_000);
    let (_, c) = s.update(DVec3::new(800.0, 0.0, 0.0), 2_500).unwrap();
    assert_eq!(c.nodes_loaded_now, 0, "steady camera reloads nothing");
}

fn raw(id: u64, m: u64, pos: DVec3, g: f32) -> RawRecord {
    RawRecord { id, match_gaia: m, pos_pc: pos, gmag: g, bp_rp: 1.0, flags: 0 }
}

#[test]
fn crossmatch_dedupe_rules() {
    let dir = |ra: f64, dec: f64, d: f64| {
        let (a, b) = (ra.to_radians(), dec.to_radians());
        DVec3::new(b.cos() * a.cos(), b.cos() * a.sin(), b.sin()) * d
    };
    let arcsec = 1.0 / 3600.0;
    let gcns = vec![raw(10, 0, dir(10.0, 10.0, 50.0), 9.0), raw(11, 0, dir(20.0, 0.0, 8.0), 1.5)];
    let gaia = vec![
        raw(10, 0, dir(10.0, 10.0, 51.0), 9.0),  // dup of GCNS by source_id
        raw(20, 0, dir(30.0, 5.0, 60.0), 10.0),  // inside 100 pc but not GCNS -> dropped
        raw(21, 0, dir(40.0, 5.0, 400.0), 4.0),  // saturated, Hip by id replaces
        raw(22, 0, dir(50.0, 5.0, 400.0), 8.0),  // well measured, Hip by id dropped
        raw(23, 0, dir(60.0, 5.0, 700.0), 5.0),  // saturated, Hip positional replaces
        raw(24, 0, dir(70.0, 5.0, 700.0), 9.0),  // unsaturated, Hip positional dropped
        raw(25, 0, dir(80.0, 5.0, 700.0), 12.0), // |dG| too large for positional match
    ];
    let hip = vec![
        raw(1, 10, dir(10.0, 10.0, 50.5), 9.1),                // matches GCNS id -> drop
        raw(2, 21, dir(40.0, 5.0, 380.0), 3.9),                // replaces Gaia 21
        raw(3, 22, dir(50.0, 5.0, 380.0), 8.1),                // dropped for Gaia 22
        raw(4, 999, dir(90.0, 5.0, 300.0), 2.0),               // bright, Gaia match out of sample -> keep
        raw(5, 998, dir(91.0, 5.0, 300.0), 7.0),               // faint, Gaia match out of sample -> drop
        raw(6, 0, dir(60.0, 5.0 + 0.5 * arcsec, 650.0), 5.2),  // positional replace of 23
        raw(7, 0, dir(70.0, 5.0 + 1.0 * arcsec, 650.0), 9.2),  // positional dup of 24
        raw(8, 0, dir(80.0, 5.0 + 0.5 * arcsec, 650.0), 9.0),  // dG=3 -> no match -> keep
        raw(9, 0, dir(20.0, 0.0 + 0.3 * arcsec, 8.1), 1.4),    // positional dup of GCNS 11
        raw(12, 0, dir(100.0, 0.0, 30.0), 0.0),                // Sirius-like: bright, no Gaia -> keep
        raw(13, 0, dir(110.0, 0.0, 30.0), 8.0),                // faint unmatched inside GCNS volume -> drop
        raw(14, 0, dir(60.0, 5.0 + 3.0 * arcsec, 650.0), 5.0), // 3" > 2" radius -> keep (no match)
    ];
    let (m, st) = merge(&gcns, &gaia, &hip, &MergeParams::default());
    let ids = |src: u8| m.iter().filter(|s| s.src == src).map(|s| s.id).collect::<Vec<_>>();
    assert_eq!(ids(SRC_GCNS), vec![10, 11]);
    assert_eq!(ids(SRC_GAIA), vec![22, 24, 25]);
    assert_eq!(ids(SRC_HIP), vec![2, 4, 6, 8, 12, 14]);
    assert_eq!(st.gaia_dup_of_gcns, 1);
    assert_eq!(st.gaia_inside_gcns_volume_not_in_gcns, 1);
    assert_eq!(st.hip_dup_of_gcns_by_id, 1);
    assert_eq!(st.hip_replaced_saturated_gaia, 1);
    assert_eq!(st.hip_dup_of_gaia_by_id, 1);
    assert_eq!(st.hip_bright_matched_gaia_outside_sample_kept, 1);
    assert_eq!(st.hip_faint_matched_gaia_outside_sample_dropped, 1);
    assert_eq!(st.hip_replaced_saturated_gaia_positional, 1);
    assert_eq!(st.hip_dup_of_gaia_positional, 1);
    assert_eq!(st.hip_dup_of_gcns_positional, 1);
    assert_eq!(st.hip_unmatched_faint_inside_gcns_volume_dropped, 1);
    assert_eq!(st.hip_unmatched_kept, 3);
    assert_eq!(st.out_by_src, [2, 3, 6]);
    assert!(m.iter().filter(|s| s.id == 2 || s.id == 6).all(|s| s.flags & flag::HIP_REPLACED_GAIA != 0));
    // Every output row is an input row (no synthesis): (src,id) unique and present in inputs.
    let mut keys: Vec<_> = m.iter().map(|s| (s.src, s.id)).collect();
    keys.dedup();
    assert_eq!(keys.len(), m.len());
}

#[test]
fn rsrc_roundtrip_and_rejects() {
    let recs = vec![raw(5, 0, DVec3::new(1.0, 2.0, 3.0), 4.0), raw(9007199254740993, 7, DVec3::X * 900.0, 12.5)];
    let mut b = Vec::new();
    write_rsrc(&mut b, &recs).unwrap();
    assert_eq!(read_rsrc(&mut &b[..]).unwrap(), recs);
    assert!(read_rsrc(&mut &b[..b.len() - 1]).is_err());
    let mut dup = Vec::new();
    write_rsrc(&mut dup, &[recs[0], recs[0]]).unwrap();
    assert!(read_rsrc(&mut &dup[..]).is_err());
    let mut bad = b.clone();
    bad[0] = b'X';
    assert!(read_rsrc(&mut &bad[..]).is_err());
}

#[test]
fn tile_header_rejects_corruption() {
    let stars = fixture_cloud(2_000, 9, 100.0);
    let img = build_tiles(&stars, &lut(), &BuildParams { node_capacity: 100, ..Default::default() }).unwrap();
    assert!(TileIndex::open(&img).is_ok());
    let mut a = img.clone();
    a[0] = 0;
    assert!(TileIndex::open(&a).is_err());
    let mut t = img.clone();
    t.pop();
    assert!(TileIndex::open(&t).is_err());
    let mut c = img.clone();
    // Corrupt first node star_count.
    c[256 + 44] ^= 0xff;
    assert!(TileIndex::open(&c).is_err());
}

#[test]
fn completeness_grows_with_budget() {
    let stars = fixture_cloud(40_000, 3, 1000.0);
    let img = build_tiles(&stars, &lut(), &BuildParams { node_capacity: 300, ..Default::default() }).unwrap();
    let idx = TileIndex::open(&img).unwrap();
    for cam in [DVec3::ZERO, DVec3::new(500.0, 0.0, 0.0)] {
        // Not strictly monotone in general (child keys can undercut a parent's), but more budget
        // must eventually buy a fainter completeness limit, and the full budget loads everything.
        let small = idx.plan(cam, 2_000).m_complete;
        let large = idx.plan(cam, 30_000).m_complete;
        assert!(large > small + 1.0, "{small} -> {large}");
        assert_eq!(idx.plan(cam, 40_000).m_complete, f64::INFINITY);
    }
}

#[test]
fn project_ids_unique_and_sources_counted_for_high_source_ids() {
    // Real Gaia DR3 source_ids reach ~6.9e18 > 2^62; source labels must not come from id bits.
    use planet::starfield::{Decoded, SRC_GAIA, SRC_GCNS, SRC_HIP, project, render_id};
    let big = (1u64 << 62) + 12345;
    let mk =
        |src, id, x: f64| Decoded { src, id, pos_pc: DVec3::new(x, 0.0, 0.0), abs_mag: 0.0, colour: 255, flags: 0 };
    let stars = vec![
        mk(SRC_GCNS, big, 10.0),
        mk(SRC_GAIA, big ^ (1 << 62), 11.0),
        mk(SRC_GAIA, big + 1, 12.0),
        mk(SRC_HIP, 32349, 13.0),
    ];
    let c = Camera::look_at(DVec3::ZERO, DVec3::X * PARSEC_M, DVec3::Z, 1.0, 64, 64).unwrap();
    let lut = vec![[1.0f32; 4]; 256];
    let (out, st) = project(&stars, &c, 99.0, 99.0, 100, &lut);
    assert_eq!(st.by_src, [1, 2, 1]);
    let mut ids: Vec<u64> = out.iter().map(|s| s.id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 4);
    assert!(ids.contains(&render_id(SRC_HIP, 32349)) && render_id(SRC_HIP, 32349) >> 63 == 1);
}
