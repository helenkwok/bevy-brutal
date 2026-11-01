//! Brick-size x voxel-spacing comparison on the cave scene (LOD0 everywhere, single thread per
//! brick timings). `cargo run --release -p voxel --example voxel_bench [-- --json]`
use std::time::Instant;

use coords::{BrickCoord, BrickGrid, floor_div};
use voxel::scene::CaveScene;
use voxel::*;

fn main() {
    let json = std::env::args().any(|a| a == "--json");
    let scene = CaveScene { seed: 1 };
    let (lo_m, hi_m) = ([-16.0, -4.0, -12.0], [16.0, 14.0, 16.0]);
    if !json {
        println!(
            "| brick | voxel m | brick m | bricks sampled | occupied | sample us/brick | mesh us/brick | tris total | tris/occupied | KiB density/brick | KiB mesh/occupied | total ms (1 thread) |"
        );
        println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    }
    for n in [16u32, 32, 64] {
        for v in [0.125f64, 0.25, 0.5] {
            let grid = BrickGrid::new(n, v).unwrap();
            let cfg = VoxelConfig::new(grid);
            let ext = n as i64;
            let lo = lo_m.map(|m: f64| floor_div((m / v).floor() as i64, ext));
            let hi = hi_m.map(|m: f64| floor_div((m / v).ceil() as i64 - 1, ext));
            let (mut sampled, mut occupied, mut tris, mut mesh_bytes) = (0usize, 0usize, 0usize, 0usize);
            let (mut t_sample, mut t_mesh) = (0f64, 0f64);
            let mut dens_bytes = 0;
            let t0 = Instant::now();
            for z in lo[2]..=hi[2] {
                for y in lo[1]..=hi[1] {
                    for x in lo[0]..=hi[0] {
                        let b = BrickCoord { x, y, z, lod: 0 };
                        let a = Instant::now();
                        let s = sample_brick(&scene, &cfg, b, 0).unwrap();
                        let m0 = Instant::now();
                        sampled += 1;
                        dens_bytes = s.bytes();
                        t_sample += (m0 - a).as_secs_f64();
                        if !s.has_sign_change() {
                            continue;
                        }
                        let m = mesh_brick(&cfg, &s, 0, None);
                        t_mesh += m0.elapsed().as_secs_f64();
                        if !m.indices.is_empty() {
                            occupied += 1;
                            tris += m.indices.len() / 3;
                            mesh_bytes += m.vertices.len() * 24 + m.indices.len() * 4;
                        }
                    }
                }
            }
            let total = t0.elapsed().as_secs_f64() * 1e3;
            let occ = occupied.max(1) as f64;
            if json {
                println!(
                    "{{\"brick\":{n},\"voxel_m\":{v},\"sampled\":{sampled},\"occupied\":{occupied},\"sample_us_per_brick\":{:.1},\"mesh_us_per_occupied\":{:.1},\"tris\":{tris},\"density_bytes_per_brick\":{dens_bytes},\"mesh_bytes_per_occupied\":{:.0},\"total_ms\":{total:.1}}}",
                    t_sample * 1e6 / sampled as f64,
                    t_mesh * 1e6 / occ,
                    mesh_bytes as f64 / occ
                );
            } else {
                println!(
                    "| {n}^3 | {v} | {} | {sampled} | {occupied} | {:.0} | {:.0} | {tris} | {:.0} | {:.1} | {:.1} | {total:.0} |",
                    n as f64 * v,
                    t_sample * 1e6 / sampled as f64,
                    t_mesh * 1e6 / occ,
                    tris as f64 / occ,
                    dens_bytes as f64 / 1024.0,
                    mesh_bytes as f64 / occ / 1024.0
                );
            }
        }
    }
}
