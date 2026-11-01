//! Near-field voxel bricks: cave + overhang scene meshed with Transvoxel (LOD0 near the camera,
//! LOD1 behind, transition cells on the seam), drawn through the surface pipeline.
//!
//! cargo run --release -p render-ash --example voxel_scene -- \
//!     [--brick 32] [--voxel 0.25] [--size 1280x720] [--png out/voxel.png] [--validate]
use std::time::Instant;

use coords::BrickGrid;
use glam::Vec3;
use lowlevel_abi::FrameSnapshot;
use render_ash::cull::{look_at, vk_perspective};
use render_ash::validation::message_counts;
use render_ash::{AshBackend, BackendConfig};
use voxel::scene::CaveScene;
use voxel::{SlabLayout, VoxelConfig, build_world, topology};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let brick: u32 = arg(&args, "--brick").map_or(32, |s| s.parse().unwrap());
    let voxel_m: f64 = arg(&args, "--voxel").map_or(0.25, |s| s.parse().unwrap());
    let (w, h) = arg(&args, "--size")
        .map(|s| {
            let (a, b) = s.split_once('x').unwrap();
            (a.parse().unwrap(), b.parse().unwrap())
        })
        .unwrap_or((1280u32, 720u32));
    let png = arg(&args, "--png").unwrap_or_else(|| "out/voxel.png".into());
    let validate = args.iter().any(|a| a == "--validate");

    let grid = BrickGrid::new(brick, voxel_m).expect("brick must be 16/32/64, voxel > 0");
    let cfg = VoxelConfig::new(grid);
    // LOD0 for z >= 0 (towards the camera), LOD1 behind; split aligned to LOD1 bricks.
    let layout = SlabLayout { axis: 2, split: 0, lod_below: 1, lod_above: 0 };
    let l = |m: f64| (m / voxel_m).round() as i64;
    let region = ([l(-16.0), l(-4.0), l(-12.0)], [l(16.0), l(14.0), l(16.0)]);
    let scene = CaveScene { seed: 1 };
    let t0 = Instant::now();
    let world = build_world(&scene, &cfg, &layout, region, 1).expect("voxel build");
    let build_ms = t0.elapsed().as_secs_f64() * 1e3;
    let topo = topology(world.bricks.values());
    let merged = world.merged();

    let mut b = AshBackend::new(BackendConfig { max_instances: 1, extent: (w, h), validation: validate, ..Default::default() }, None).expect("backend");
    b.add_static_mesh(&merged.vertices, &merged.indices).unwrap();
    let proj = vk_perspective(0.9, w as f32 / h as f32, 0.1, 500.0);
    let vp = (proj * look_at(Vec3::new(-15.0, 7.5, 22.0), Vec3::new(-1.0, 4.0, 1.0), Vec3::Y)).to_cols_array_2d();
    let mut gpu_ms = 0.0;
    for f in 0..4 {
        if let Some(s) = b.render(&FrameSnapshot { view_proj: vp, instances: &[], frame_index: f }).unwrap() {
            gpu_ms = s.draw_ms.unwrap_or(f64::NAN);
        }
    }
    let (pw, ph, px) = b.read_color_rgba().unwrap();
    if let Some(dir) = std::path::Path::new(&png).parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut enc = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(&png).unwrap()), pw, ph);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&px).unwrap();
    let (errors, warnings) = message_counts();
    let lods: Vec<usize> = (0..2).map(|l| world.bricks.keys().filter(|k| k.lod == l).count()).collect();
    println!(
        "{{\"brick\":{brick},\"voxel_m\":{voxel_m},\"bricks_sampled\":{},\"bricks_occupied\":{},\"lod0\":{},\"lod1\":{},\"tris\":{},\"verts\":{},\"build_ms_parallel\":{build_ms:.1},\"draw_ms\":{gpu_ms:.3},\"non_manifold\":{},\"orientation_conflicts\":{},\"degenerate\":{},\"non_finite\":{},\"open_edges_region_cut\":{},\"validation_errors\":{errors},\"validation_warnings\":{warnings},\"device\":\"{}\",\"png\":\"{png}\"}}",
        world.sampled,
        world.bricks.len(),
        lods[0],
        lods[1],
        world.triangles(),
        world.vertices(),
        topo.non_manifold_edges,
        topo.orientation_conflicts,
        topo.degenerate,
        topo.non_finite,
        topo.boundary_edges,
        b.device_name()
    );
    if validate && errors > 0 {
        std::process::exit(1);
    }
}
