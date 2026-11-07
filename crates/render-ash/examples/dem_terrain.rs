//! Real-world DEM chunks (`terrain::dem`) through the render-ash terrain slots, headless.
//!
//! cargo run --release -p render-ash --example dem_terrain -- \
//!     [--grid /workspace/data/geo/terrain/grids/adelaide_z12] [--name adelaide_z12] [--out out/quality/dem] \
//!     [--tau 2.0] [--size 1280x720] [--fov 50] [--eye E,N,H] [--target E,N,H] [--slots 256] \
//!     [--fly 120] [--budget 16] [--validate]
//!
//! Slots are keyed `(lod, x, y)` (`TerrainSlotKey` from `DemChunkId`), `verts_per_slot =
//! DEM_CHUNK_VERTS` (65² surface + 4·65 skirt vertices) with the shared `dem_chunk_indices()`.
//! Writes, into `--out`:
//! * `<name>_gpu.png`               selected LOD set, `SurfacePalette::HeightRange` over the grid's heights
//! * `<name>_gpu_noise_palette.png` same frame with the old noise palette (snow above 28 m), for comparison
//! * `<name>_gpu.json`              selection, residency/upload stats, palette pixel stats, flyover stats
//!
//! `--fly N` then flies N frames south-west → north-east over the grid (5 → 60 % of the diagonal), re-selecting
//! the LOD set every frame and streaming at most `--budget` new chunks per frame; a selected chunk
//! that is not resident yet is covered by its nearest resident ancestor (descendants of a drawn
//! ancestor are dropped, so nothing overlaps). Coarse levels (the top 3) are uploaded first.
//! Eye/target are ENU metres (east, north, height) from the grid origin, as in `terrain::dem_preview`.
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use glam::Vec3;
use lowlevel_abi::FrameSnapshot;
use render_ash::cull::{look_at, vk_perspective};
use render_ash::validation::message_counts;
use render_ash::{AshBackend, BackendConfig, BackendError, SurfacePalette, TerrainGpuConfig, TerrainSlotKey};
use terrain::dem::{DEM_CHUNK_VERTS, DemChunkId, DemGrid, DemTerrain, LodView, dem_chunk_indices};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn vec3_arg(args: &[String], name: &str, default: [f32; 3]) -> [f32; 3] {
    arg(args, name).map_or(default, |s| {
        let v: Vec<f32> = s.split(',').map(|x| x.trim().parse().expect("number")).collect();
        [v[0], v[1], v[2]]
    })
}

/// ENU (east, north, up) -> mesh space (x = east, y = up, z = -north).
fn enu(v: [f32; 3]) -> Vec3 {
    Vec3::new(v[0], v[2], -v[1])
}

fn key(id: DemChunkId) -> TerrainSlotKey {
    TerrainSlotKey::new(id.lod, id.x as i32, id.y as i32)
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn write_png(path: &Path, w: u32, h: u32, rgba: &[u8]) {
    let mut enc = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(path).unwrap()), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(rgba).unwrap();
    eprintln!("wrote {}", path.display());
}

/// (surface pixels, near-white "snow" pixels among them). Clear colour = sky.
fn palette_stats(px: &[u8]) -> (usize, usize) {
    let clear = [13u8, 15, 23];
    let (mut surf, mut white) = (0, 0);
    for q in px.as_chunks::<4>().0 {
        if q[..3].iter().zip(clear).all(|(x, y)| x.abs_diff(y) <= 2) {
            continue;
        }
        surf += 1;
        let (mx, mn) = (q[..3].iter().max().unwrap(), q[..3].iter().min().unwrap());
        if *mn > 170 && mx - mn < 30 {
            white += 1;
        }
    }
    (surf, white)
}

struct Streamer {
    uploads_ok: usize,
    ring_full: usize,
    mesh_ms: Vec<f64>,
    upload_ms: Vec<f64>,
    buf: Vec<terrain::TerrainVertex>,
}

impl Streamer {
    /// Upload up to `budget` of `ids` that are not resident. Stops early when the ring is full.
    fn stream(&mut self, b: &mut AshBackend, t: &DemTerrain, ids: &[DemChunkId], budget: usize) -> usize {
        let mut n = 0;
        for &id in ids {
            if n == budget {
                break;
            }
            if b.terrain_is_resident(key(id)) {
                continue;
            }
            let m = Instant::now();
            t.mesh_chunk_into(id, &mut self.buf);
            self.mesh_ms.push(m.elapsed().as_secs_f64() * 1e3);
            let u = Instant::now();
            match b.upload_terrain_chunk(key(id), &self.buf) {
                Ok(_) => {
                    self.upload_ms.push(u.elapsed().as_secs_f64() * 1e3);
                    self.uploads_ok += 1;
                    n += 1;
                }
                Err(BackendError::Unsupported(_)) => {
                    // staging ring full or every slot in flight: retry next frame
                    self.ring_full += 1;
                    break;
                }
                Err(e) => panic!("upload {id:?}: {e:?}"),
            }
        }
        n
    }
}

/// Draw list for `selected`: each chunk if resident, else its nearest resident ancestor; drawn
/// ancestors suppress their descendants. Returns (keys, chunks with no resident cover).
fn cover(b: &AshBackend, t: &DemTerrain, selected: &[DemChunkId]) -> (Vec<TerrainSlotKey>, usize) {
    let mut draw: BTreeSet<DemChunkId> = BTreeSet::new();
    let mut holes = 0;
    for &id in selected {
        let mut c = id;
        loop {
            if b.terrain_is_resident(key(c)) {
                draw.insert(c);
                break;
            }
            if c.lod >= t.root_lod() {
                holes += 1;
                break;
            }
            c = c.parent();
        }
    }
    let has_drawn_ancestor = |id: DemChunkId| {
        let mut c = id;
        while c.lod < t.root_lod() {
            c = c.parent();
            if draw.contains(&c) {
                return true;
            }
        }
        false
    };
    let keys = draw.iter().copied().filter(|&id| !has_drawn_ancestor(id)).map(key).collect();
    (keys, holes)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let grid_dir =
        PathBuf::from(arg(&args, "--grid").unwrap_or_else(|| "/workspace/data/geo/terrain/grids/adelaide_z12".into()));
    let out = PathBuf::from(arg(&args, "--out").unwrap_or_else(|| "out/quality/dem".into()));
    let name = arg(&args, "--name")
        .unwrap_or_else(|| grid_dir.file_name().map_or("dem".into(), |s| s.to_string_lossy().into_owned()));
    let tau: f32 = arg(&args, "--tau").map_or(2.0, |s| s.parse().unwrap());
    let fov_deg: f32 = arg(&args, "--fov").map_or(50.0, |s| s.parse().unwrap());
    let slots: u32 = arg(&args, "--slots").map_or(256, |s| s.parse().unwrap());
    let fly: usize = arg(&args, "--fly").map_or(120, |s| s.parse().unwrap());
    let budget: usize = arg(&args, "--budget").map_or(16, |s| s.parse().unwrap());
    let (w, h) = arg(&args, "--size")
        .map(|s| {
            let (a, b) = s.split_once('x').unwrap();
            (a.parse().unwrap(), b.parse().unwrap())
        })
        .unwrap_or((1280u32, 720u32));
    let eye_enu = vec3_arg(&args, "--eye", [-3000.0, -9000.0, 900.0]);
    let target_enu = vec3_arg(&args, "--target", [8000.0, -4000.0, 250.0]);
    let validate = args.iter().any(|a| a == "--validate");
    std::fs::create_dir_all(&out).unwrap();

    let hg = geo_fetch::grid::read_grid(&grid_dir).unwrap_or_else(|e| panic!("{}: {e}", grid_dir.display()));
    let grid = DemGrid::new(hg.width, hg.height, hg.spacing_m as f32, hg.x0 as f32, hg.y0 as f32, hg.heights).unwrap();
    let (hmin, hmax) = grid.height_range();
    let (gx0, gy0) = (grid.x0_m, grid.y0_m);
    let (gx1, gy1) = (gx0 + (grid.width - 1) as f32 * grid.spacing_m, gy0 + (grid.height - 1) as f32 * grid.spacing_m);
    let t_build = Instant::now();
    let t = DemTerrain::new(grid);
    let build_ms = t_build.elapsed().as_secs_f64() * 1e3;

    let mut b = AshBackend::new(
        BackendConfig { max_instances: 1, extent: (w, h), validation: validate, ..Default::default() },
        None,
    )
    .expect("backend");
    let cfg = TerrainGpuConfig::dem(slots, DEM_CHUNK_VERTS as u32);
    b.enable_terrain(cfg, &dem_chunk_indices()).unwrap();
    let palette = SurfacePalette::for_heights(hmin, hmax);
    b.set_surface_palette(palette);
    eprintln!(
        "{name}: {}x{} h {hmin:.1}..{hmax:.1} m, root LOD {} | device {} | {w}x{h} | slots {slots} x {} verts ({:.1} MiB) | palette {palette:?}",
        t.grid().width,
        t.grid().height,
        t.root_lod(),
        b.device_name(),
        DEM_CHUNK_VERTS,
        slots as f64 * cfg.slot_bytes() as f64 / (1 << 20) as f64,
    );

    let fov = fov_deg.to_radians();
    let proj = vk_perspective(fov, w as f32 / h as f32, 5.0, 100_000.0);
    let mut st = Streamer {
        uploads_ok: 0,
        ring_full: 0,
        mesh_ms: Vec::new(),
        upload_ms: Vec::new(),
        buf: vec![bytemuck::Zeroable::zeroed(); DEM_CHUNK_VERTS],
    };
    let mut frame_index = 0u64;
    let mut render = |b: &mut AshBackend, vp: [[f32; 4]; 4]| {
        b.render(&FrameSnapshot { view_proj: vp, instances: &[], frame_index }).unwrap();
        frame_index += 1;
    };

    // --- Still: the dem_preview view, everything resident before the shot.
    let eye = enu(eye_enu);
    let view = LodView { eye: eye.to_array(), fov_y: fov, viewport_height_px: h as f32, max_error_px: tau };
    let selected = t.select(&view);
    assert!(selected.len() <= slots as usize, "{} selected chunks > {slots} slots", selected.len());
    let mut per_lod = vec![0usize; t.root_lod() as usize + 1];
    for id in &selected {
        per_lod[id.lod as usize] += 1;
    }
    let vp = (proj * look_at(eye, enu(target_enu), Vec3::Y)).to_cols_array_2d();
    let keys: Vec<TerrainSlotKey> = selected.iter().map(|&id| key(id)).collect();
    let t_still = Instant::now();
    let mut still_frames = 0;
    while keys.iter().any(|&k| !b.terrain_is_resident(k)) {
        st.stream(&mut b, &t, &selected, usize::MAX);
        b.set_terrain_draw(&keys);
        render(&mut b, vp);
        still_frames += 1;
        assert!(still_frames < 64, "still view never became resident");
    }
    // one more frame so the last copies are drawn, then settle the ring
    for _ in 0..2 {
        assert_eq!(b.set_terrain_draw(&keys), keys.len());
        render(&mut b, vp);
    }
    let still_ms = t_still.elapsed().as_secs_f64() * 1e3;
    let (iw, ih, px) = b.read_color_rgba().unwrap();
    write_png(&out.join(format!("{name}_gpu.png")), iw, ih, &px);
    let (surf, white) = palette_stats(&px);

    b.set_surface_palette(SurfacePalette::Noise);
    for _ in 0..2 {
        render(&mut b, vp);
    }
    let (_, _, px_noise) = b.read_color_rgba().unwrap();
    write_png(&out.join(format!("{name}_gpu_noise_palette.png")), iw, ih, &px_noise);
    let (surf_n, white_n) = palette_stats(&px_noise);
    b.set_surface_palette(palette);
    eprintln!(
        "still: {} chunks {per_lod:?} | resident after {still_frames} frames ({still_ms:.0} ms) | snow-white px: dem palette {:.2} %, noise palette {:.2} %",
        selected.len(),
        100.0 * white as f64 / surf.max(1) as f64,
        100.0 * white_n as f64 / surf_n.max(1) as f64,
    );

    // --- Flyover: SW -> NE over the first 60 % of the diagonal at 1.2 km, looking ahead and down.
    let (mut frame_ms, mut sel_ms, mut drawn, mut hole_frames, mut max_holes) = (Vec::new(), Vec::new(), 0usize, 0, 0);
    let mut coarse: Vec<DemChunkId> = Vec::new();
    for l in t.root_lod().saturating_sub(2)..=t.root_lod() {
        let (cx, cy) = t.chunks_at(l);
        coarse.extend((0..cy).flat_map(|y| (0..cx).map(move |x| DemChunkId { lod: l, x, y })));
    }
    coarse.sort_by_key(|id| std::cmp::Reverse(id.lod));
    let stats0 = b.terrain_stats().unwrap();
    let t_fly = Instant::now();
    for f in 0..fly {
        let ft = Instant::now();
        let s = f as f32 / fly.max(2).saturating_sub(1) as f32;
        let (e, n) = (gx0 + (gx1 - gx0) * (0.05 + 0.55 * s), gy0 + (gy1 - gy0) * (0.05 + 0.55 * s));
        let eye = enu([e, n, 1200.0]);
        let at = enu([e + 4000.0, n + 4000.0, 150.0]);
        let sv = Instant::now();
        let mut sel =
            t.select(&LodView { eye: eye.to_array(), fov_y: fov, viewport_height_px: h as f32, max_error_px: tau });
        sel_ms.push(sv.elapsed().as_secs_f64() * 1e3);
        // coarse cover first, then nearest-first refinement
        sel.sort_by(|a, c| {
            t.chunk_aabb(*a).distance(eye.to_array()).total_cmp(&t.chunk_aabb(*c).distance(eye.to_array()))
        });
        let mut left = budget;
        left -= st.stream(&mut b, &t, &coarse, left);
        st.stream(&mut b, &t, &sel, left);
        let (keys, holes) = cover(&b, &t, &sel);
        if holes > 0 {
            hole_frames += 1;
            max_holes = max_holes.max(holes);
        }
        drawn += b.set_terrain_draw(&keys);
        render(&mut b, (proj * look_at(eye, at, Vec3::Y)).to_cols_array_2d());
        frame_ms.push(ft.elapsed().as_secs_f64() * 1e3);
    }
    b.drain().unwrap();
    let fly_s = t_fly.elapsed().as_secs_f64();
    if fly > 0 {
        let (iw, ih, px) = b.read_color_rgba().unwrap();
        write_png(&out.join(format!("{name}_gpu_fly_last.png")), iw, ih, &px);
    }
    let stats = b.terrain_stats().unwrap();
    let (errors, warnings) = message_counts();
    let json = format!(
        "{{\"name\":\"{name}\",\"grid\":\"{}\",\"device\":\"{}\",\"size\":[{w},{h}],\"height_m\":[{hmin:.2},{hmax:.2}],\"root_lod\":{},\"build_ms\":{build_ms:.1},\
\"slots\":{slots},\"verts_per_slot\":{DEM_CHUNK_VERTS},\"slot_bytes\":{},\"palette\":\"{palette:?}\",\
\"still\":{{\"eye_enu\":{eye_enu:?},\"target_enu\":{target_enu:?},\"fov_deg\":{fov_deg},\"tau_px\":{tau},\"chunks\":{},\"per_lod\":{per_lod:?},\"frames_to_resident\":{still_frames},\"ms\":{still_ms:.0},\
\"surface_px\":{surf},\"white_px_dem_palette\":{white},\"white_px_noise_palette\":{white_n}}},\
\"fly\":{{\"frames\":{fly},\"budget\":{budget},\"fps\":{:.1},\"frame_ms_p50\":{:.2},\"frame_ms_p99\":{:.2},\"select_ms_p50\":{:.3},\"avg_drawn\":{:.1},\"hole_frames\":{hole_frames},\"max_holes\":{max_holes},\
\"uploads\":{},\"evictions\":{}}},\
\"mesh_ms_p50\":{:.3},\"upload_call_ms_p50\":{:.3},\"ring_full_retries\":{},\"max_uploads_in_flight\":{},\"max_staging_live_mib\":{:.2},\"validation_errors\":{errors},\"validation_warnings\":{warnings}}}",
        grid_dir.display(),
        b.device_name(),
        t.root_lod(),
        cfg.slot_bytes(),
        selected.len(),
        fly as f64 / fly_s.max(1e-9),
        pct(&mut frame_ms.clone(), 0.5),
        pct(&mut frame_ms, 0.99),
        pct(&mut sel_ms, 0.5),
        drawn as f64 / fly.max(1) as f64,
        stats.uploads - stats0.uploads,
        stats.evictions - stats0.evictions,
        pct(&mut st.mesh_ms, 0.5),
        pct(&mut st.upload_ms, 0.5),
        st.ring_full,
        stats.max_uploads_in_flight,
        stats.max_staging_live as f64 / (1 << 20) as f64,
    );
    std::fs::write(out.join(format!("{name}_gpu.json")), &json).unwrap();
    println!("{json}");
    drop(b);
    if validate && errors > 0 {
        std::process::exit(1);
    }
}
