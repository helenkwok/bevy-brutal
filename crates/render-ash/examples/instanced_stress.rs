//! Headless stress run of the GPU-driven instanced pass.
//!
//! cargo run --release -p render-ash --example instanced_stress -- \
//!     [--instances 10000,100000,1000000] [--frames 60,30,10] [--size 1280x720] \
//!     [--png out/instanced.png] [--validate] [--fif 2|3] [--scan auto|subgroup|shared]
//!
//! Prints one JSON line per instance count: wall fps, CPU ms per `render()` (timeline wait for the
//! slot + upload + record + submit), p50/p99 GPU compute-chain (cull+count, scan, scatter+args) and
//! draw ms from timestamp queries, max frames in flight, and how many consecutive frame pairs
//! overlapped in GPU time.
use std::time::Instant;

use lowlevel_abi::FrameSnapshot;
use render_ash::scene::{demo_view_proj, random_scene};
use render_ash::validation::message_counts;
use render_ash::{AshBackend, BackendConfig, FrameStats};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let counts: Vec<usize> = arg(&args, "--instances")
        .unwrap_or_else(|| "10000,100000,1000000".into())
        .split(',')
        .map(|s| s.trim().replace('_', "").parse().expect("bad --instances"))
        .collect();
    let frames: Vec<usize> =
        arg(&args, "--frames").unwrap_or_else(|| "60,30,10".into()).split(',').map(|s| s.trim().parse().expect("bad --frames")).collect();
    let (w, h) = arg(&args, "--size")
        .map(|s| {
            let (a, b) = s.split_once('x').expect("--size WxH");
            (a.parse().unwrap(), b.parse().unwrap())
        })
        .unwrap_or((1280u32, 720u32));
    let png = arg(&args, "--png");
    let validate = args.iter().any(|a| a == "--validate");
    let scan = match arg(&args, "--scan").as_deref() {
        Some("subgroup") => render_ash::ScanMode::Subgroup,
        Some("shared") => render_ash::ScanMode::SharedMemory,
        _ => render_ash::ScanMode::Auto,
    };
    let fif: usize = arg(&args, "--fif").map(|s| s.parse().expect("bad --fif")).unwrap_or(2);

    let max = *counts.iter().max().unwrap();
    let mut b = AshBackend::new(BackendConfig { max_instances: max, extent: (w, h), validation: validate, frames_in_flight: fif, scan }, None).expect("backend");
    eprintln!(
        "device: {} | {w}x{h} | frames in flight: {fif} | scan: {:?} | validation: {} (sync: {}) | timestamps: {}",
        b.device_name(),
        b.scan_path(),
        b.ctx.validation_enabled,
        b.ctx.sync_validation,
        b.ctx.timestamp_period_ns.is_some()
    );
    let meshes = b.mesh_table().len() as u32;

    for (k, &n) in counts.iter().enumerate() {
        let nframes = frames.get(k).or(frames.last()).copied().unwrap_or(10).max(3);
        let inst = random_scene(n, 1234 + n as u64, meshes, 0.0);
        let vp = demo_view_proj(n, w as f32 / h as f32).to_cols_array_2d();
        // warm-up (pipelines, first-touch of buffers)
        for f in 0..2 {
            b.render(&FrameSnapshot { view_proj: vp, instances: &inst, frame_index: f }).unwrap();
        }
        b.drain().unwrap();

        let mut stats: Vec<FrameStats> = Vec::new();
        let mut cpu_ms = Vec::new();
        let t0 = Instant::now();
        for f in 0..nframes {
            let t = Instant::now();
            let r = b.render(&FrameSnapshot { view_proj: vp, instances: &inst, frame_index: f as u64 }).unwrap();
            let el = t.elapsed();
            cpu_ms.push(el.as_secs_f64() * 1e3);
            if let Some(s) = r {
                stats.push(s);
            }
        }
        stats.extend(b.drain().unwrap());
        let secs = t0.elapsed().as_secs_f64();
        let col = |f: fn(&FrameStats) -> Option<f64>| stats.iter().filter_map(f).collect::<Vec<f64>>();
        let (mut cull, mut draw) = (col(|s| s.cull_ms), col(|s| s.draw_ms));
        let (mut count, mut scan, mut scatter) = (col(|s| s.count_ms), col(|s| s.scan_ms), col(|s| s.scatter_ms));
        let max_in_flight = stats.iter().map(|s| s.in_flight_at_submit).max().unwrap_or(0);
        let gpu_overlap = stats
            .windows(2)
            .filter(|w| matches!((w[1].gpu_begin_ns, w[0].gpu_end_ns), (Some(b1), Some(e0)) if b1 < e0))
            .count();
        let last = *stats.last().unwrap();
        println!(
            "{{\"instances\":{n},\"frames\":{nframes},\"fif\":{fif},\"width\":{w},\"height\":{h},\"fps\":{:.2},\"cpu_render_ms_p50\":{:.3},\"gpu_cull_ms_p50\":{:.3},\"gpu_cull_ms_p99\":{:.3},\"count_ms_p50\":{:.3},\"scan_ms_p50\":{:.3},\"scatter_args_ms_p50\":{:.3},\"gpu_draw_ms_p50\":{:.3},\"gpu_draw_ms_p99\":{:.3},\"max_in_flight\":{max_in_flight},\"gpu_overlapping_pairs\":\"{gpu_overlap}/{}\",\"visible\":{},\"draws\":{},\"device\":\"{}\"}}",
            nframes as f64 / secs,
            pct(&mut cpu_ms, 0.5),
            pct(&mut cull, 0.5),
            pct(&mut cull, 0.99),
            pct(&mut count, 0.5),
            pct(&mut scan, 0.5),
            pct(&mut scatter, 0.5),
            pct(&mut draw, 0.5),
            pct(&mut draw, 0.99),
            stats.len().saturating_sub(1),
            last.visible,
            last.draws,
            b.device_name()
        );

        if k == 0
            && let Some(path) = &png
        {
            let (w, h, px) = b.read_color_rgba().unwrap();
            if let Some(dir) = std::path::Path::new(path).parent() {
                std::fs::create_dir_all(dir).unwrap();
            }
            let file = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
            let mut enc = png::Encoder::new(file, w, h);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            enc.write_header().unwrap().write_image_data(&px).unwrap();
            eprintln!("wrote {path} ({w}x{h}, {n} instances, {} visible)", last.visible);
        }
    }
    let (errors, warnings) = message_counts();
    eprintln!("validation messages: {errors} errors, {warnings} warnings");
    if validate && errors > 0 {
        std::process::exit(1);
    }
}
