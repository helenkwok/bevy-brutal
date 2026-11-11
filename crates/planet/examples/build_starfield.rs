//! RS1: merge normalised observed catalogues and write the RSTILE01 octree tile file.
//!
//!   python3 tools/real_starfield/ingest.py          # TAP-fetched raw CSV -> norm/*.rsrc
//!   cargo run -p planet --release --example build_starfield -- [/workspace/data/astro/real_starfield]
//!
//! Inputs/outputs live outside git. No rows are synthesised: every tile star is one catalogue row.
use planet::starfield::*;
use std::path::PathBuf;

fn load(p: &std::path::Path) -> Vec<RawRecord> {
    let f =
        std::fs::File::open(p).unwrap_or_else(|e| panic!("{}: {e} (run tools/real_starfield/ingest.py)", p.display()));
    read_rsrc(&mut std::io::BufReader::with_capacity(1 << 20, f)).unwrap()
}

fn main() {
    let root = PathBuf::from(std::env::args().nth(1).unwrap_or("/workspace/data/astro/real_starfield".into()));
    let t0 = std::time::Instant::now();
    let gcns = load(&root.join("norm/gcns.rsrc"));
    let gaia = load(&root.join("norm/gaia.rsrc"));
    let hip = load(&root.join("norm/hip.rsrc"));
    let lut_bytes = std::fs::read(root.join("colour_lut.bin")).expect("colour_lut.bin");
    assert_eq!(lut_bytes.len(), 255 * 16);
    let lut: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&lut_bytes);
    let mp = MergeParams::default();
    let (merged, st) = merge(&gcns, &gaia, &hip, &mp);
    let t_merge = t0.elapsed().as_secs_f64();
    let bp = BuildParams::default();
    let img = build_tiles(&merged, &lut, &bp).unwrap();
    let idx = TileIndex::open(&img).unwrap();
    let out = root.join("tiles");
    std::fs::create_dir_all(&out).unwrap();
    let path = out.join("rs1.rstile");
    std::fs::write(&path, &img).unwrap();
    let max_depth = idx.nodes.iter().map(|n| n.depth).max().unwrap();
    let leaves = idx.nodes.iter().filter(|n| n.child_mask == 0).count();
    let maxd = merged.iter().map(|s| s.pos_pc.length()).fold(0.0, f64::max);
    let worst_bound = idx.nodes.iter().map(pos_error_bound).fold(0.0, f64::max);
    let no_col = merged.iter().filter(|s| s.bp_rp.is_nan()).count();
    let mut by_mag = [0usize; 8];
    for s in &merged {
        let b = ((s.gmag.max(-2.0) + 2.0) / 3.0) as usize;
        by_mag[b.min(7)] += 1;
    }
    let json = format!(
        "{{\n \"inputs\": {{\"gcns\": {}, \"gaia\": {}, \"hip\": {}}},\n \"merge_params\": {{\"gcns_radius_pc\": {}, \"hip_saturation_g\": {}, \"fallback_radius_arcsec\": {}, \"fallback_max_dg\": {}}},\n \"merge_stats\": {},\n \"output_stars\": {},\n \"output_by_src\": {{\"gcns\": {}, \"gaia_dr3_bj\": {}, \"hipparcos2\": {}}},\n \"no_colour\": {},\n \"g_hist_bins_3mag_from_-2\": {:?},\n \"max_distance_pc\": {:.1},\n \"tile_file\": {:?},\n \"tile_bytes\": {},\n \"nodes\": {},\n \"leaves\": {},\n \"max_depth\": {},\n \"node_capacity\": {},\n \"root_min_pc\": {:?},\n \"root_size_pc\": {},\n \"worst_position_error_bound_pc\": {:e},\n \"mag_quantum\": {:e},\n \"merge_s\": {:.2},\n \"total_s\": {:.2}\n}}\n",
        gcns.len(),
        gaia.len(),
        hip.len(),
        mp.gcns_radius_pc,
        mp.hip_saturation_g,
        mp.fallback_radius_arcsec,
        mp.fallback_max_dg,
        st.to_json(),
        merged.len(),
        st.out_by_src[0],
        st.out_by_src[1],
        st.out_by_src[2],
        no_col,
        by_mag,
        maxd,
        path.display().to_string(),
        img.len(),
        idx.nodes.len(),
        leaves,
        max_depth,
        bp.node_capacity,
        idx.header.root_min,
        idx.header.root_size,
        worst_bound,
        MAG_SCALE,
        t_merge,
        t0.elapsed().as_secs_f64()
    );
    std::fs::write(root.join("merge_stats.json"), &json).unwrap();
    println!("{json}");
}
