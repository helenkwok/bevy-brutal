//! Headless terrain streaming flyover (Prompt C, single LOD).
//!
//! cargo run --release -p render-ash --example terrain_flyover -- \
//!     [--frames 240] [--radius 4] [--slots 128] [--size 1280x720] [--png out/terrain.png] [--validate]
//!
//! A camera flies a fixed closed spline; each frame the chunks within `radius` (Chebyshev, in
//! chunks) of the camera are requested, at most 16 new ones are generated (SIMD heightmap + mesh
//! into a per-frame FrameArena) and staged through the 32 MiB ring, then drawn. Prints one JSON
//! line: per-chunk generate time, per-chunk upload-call time, frame times, max uploads in flight,
//! evictions, validation counts.
use std::time::Instant;

use glam::Vec3;
use lowlevel_abi::FrameSnapshot;
use mem::FrameArena;
use render_ash::cull::{look_at, vk_perspective};
use render_ash::validation::message_counts;
use render_ash::{AshBackend, BackendConfig, TerrainGpuConfig};
use terrain::{
    CHUNK_SAMPLES, ChunkRequest, NoiseParams, PADDED, TerrainVertex, grid_indices, heightmap_simd, mesh_chunk,
};

const CHUNK_M: f32 = (CHUNK_SAMPLES - 1) as f32;
const MAX_NEW_PER_FRAME: usize = 16;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * p).round() as usize]
}

/// Closed flight path (radius ~300 m), altitude above the terrain's max amplitude.
fn camera(t: f32) -> (Vec3, Vec3) {
    let a = t * std::f32::consts::TAU;
    let eye = Vec3::new(300.0 * a.cos(), 70.0 + 10.0 * (2.0 * a).sin(), 220.0 * a.sin());
    let ahead = Vec3::new(300.0 * (a + 0.15).cos(), 25.0, 220.0 * (a + 0.15).sin());
    (eye, ahead)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let frames: usize = arg(&args, "--frames").map_or(240, |s| s.parse().unwrap());
    let radius: i32 = arg(&args, "--radius").map_or(4, |s| s.parse().unwrap());
    let slots: u32 = arg(&args, "--slots").map_or(128, |s| s.parse().unwrap());
    let (w, h) = arg(&args, "--size")
        .map(|s| {
            let (a, b) = s.split_once('x').unwrap();
            (a.parse().unwrap(), b.parse().unwrap())
        })
        .unwrap_or((1280u32, 720u32));
    let png = arg(&args, "--png");
    let validate = args.iter().any(|a| a == "--validate");

    let mut b = AshBackend::new(
        BackendConfig { max_instances: 1, extent: (w, h), validation: validate, ..Default::default() },
        None,
    )
    .expect("backend");
    b.enable_terrain(TerrainGpuConfig { slots, ..Default::default() }, &grid_indices()).unwrap();
    eprintln!(
        "device: {} | {w}x{h} | slots {slots} | radius {radius} | validation {}",
        b.device_name(),
        b.ctx.validation_enabled
    );

    let p = NoiseParams::default();
    let mut arena = FrameArena::with_capacity(
        MAX_NEW_PER_FRAME * (CHUNK_SAMPLES * CHUNK_SAMPLES * size_of::<TerrainVertex>() + 64),
    );
    let mut hm = vec![0.0f32; PADDED * PADDED];
    let (mut gen_ms, mut up_ms, mut frame_ms) = (Vec::new(), Vec::new(), Vec::new());
    let proj = vk_perspective(1.0, w as f32 / h as f32, 0.5, 2000.0);
    let t0 = Instant::now();
    for f in 0..frames {
        let ft = Instant::now();
        let (eye, at) = camera(f as f32 / frames as f32);
        let (cx, cz) = ((eye.x / CHUNK_M).floor() as i32, (eye.z / CHUNK_M).floor() as i32);
        let mut wanted: Vec<[i32; 2]> = Vec::new();
        for dz in -radius..=radius {
            for dx in -radius..=radius {
                wanted.push([cx + dx, cz + dz]);
            }
        }
        // nearest first, so the 16-per-frame budget fills the view centre first
        wanted.sort_by_key(|c| (c[0] - cx).abs().max((c[1] - cz).abs()));
        arena.reset();
        let mut new = 0;
        for &c in &wanted {
            if new == MAX_NEW_PER_FRAME || b.terrain_is_resident(c) {
                continue;
            }
            let req = ChunkRequest { coord: c, lod: 0, seed: 42 };
            let g = Instant::now();
            heightmap_simd(&req, &p, &mut hm);
            let m = mesh_chunk(&req, &hm, &arena).expect("arena sized for 16 chunks");
            gen_ms.push(g.elapsed().as_secs_f64() * 1e3);
            let u = Instant::now();
            b.upload_terrain_chunk(c, m.vertices).expect("upload");
            up_ms.push(u.elapsed().as_secs_f64() * 1e3);
            new += 1;
        }
        b.set_terrain_draw(&wanted);
        let vp = proj * look_at(eye, at, Vec3::Y);
        b.render(&FrameSnapshot { view_proj: vp.to_cols_array_2d(), instances: &[], frame_index: f as u64 }).unwrap();
        frame_ms.push(ft.elapsed().as_secs_f64() * 1e3);
    }
    b.drain().unwrap();
    let secs = t0.elapsed().as_secs_f64();
    let st = b.terrain_stats().unwrap();
    let (errors, warnings) = message_counts();
    println!(
        "{{\"frames\":{frames},\"fps\":{:.1},\"frame_ms_p50\":{:.2},\"chunk_gen_ms_p50\":{:.4},\"chunk_gen_ms_p99\":{:.4},\"upload_call_ms_p50\":{:.4},\"upload_call_ms_p99\":{:.4},\"uploads\":{},\"evictions\":{},\"max_uploads_in_flight\":{},\"max_staging_live_mib\":{:.2},\"validation_errors\":{errors},\"validation_warnings\":{warnings},\"device\":\"{}\"}}",
        frames as f64 / secs,
        pct(&mut frame_ms, 0.5),
        pct(&mut gen_ms, 0.5),
        pct(&mut gen_ms, 0.99),
        pct(&mut up_ms, 0.5),
        pct(&mut up_ms, 0.99),
        st.uploads,
        st.evictions,
        st.max_uploads_in_flight,
        st.max_staging_live as f64 / (1 << 20) as f64,
        b.device_name()
    );

    if let Some(path) = png {
        // Beauty shot: low over the terrain looking across ridges.
        let eye = Vec3::new(-40.0, 62.0, -60.0);
        let at = Vec3::new(160.0, 5.0, 140.0);
        let (cx, cz) = ((at.x / CHUNK_M).floor() as i32, (at.z / CHUNK_M).floor() as i32);
        let wanted: Vec<[i32; 2]> = (-4..=4).flat_map(|dz| (-4..=4).map(move |dx| [cx + dx, cz + dz])).collect();
        let vp = (proj * look_at(eye, at, Vec3::Y)).to_cols_array_2d();
        for f in 0..8 {
            arena.reset();
            let missing: Vec<[i32; 2]> =
                wanted.iter().copied().filter(|&c| !b.terrain_is_resident(c)).take(MAX_NEW_PER_FRAME).collect();
            for c in missing {
                let req = ChunkRequest { coord: c, lod: 0, seed: 42 };
                heightmap_simd(&req, &p, &mut hm);
                let m = mesh_chunk(&req, &hm, &arena).unwrap();
                b.upload_terrain_chunk(c, m.vertices).unwrap();
            }
            b.set_terrain_draw(&wanted);
            b.render(&FrameSnapshot { view_proj: vp, instances: &[], frame_index: (frames + f) as u64 }).unwrap();
        }
        let (w, h, px) = b.read_color_rgba().unwrap();
        if let Some(dir) = std::path::Path::new(&path).parent() {
            std::fs::create_dir_all(dir).unwrap();
        }
        let mut enc = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(&path).unwrap()), w, h);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().unwrap().write_image_data(&px).unwrap();
        eprintln!("wrote {path}");
    }
    let (errors, _) = message_counts();
    if validate && errors > 0 {
        std::process::exit(1);
    }
}
