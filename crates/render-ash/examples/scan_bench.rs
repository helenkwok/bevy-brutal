//! GPU scan timing, subgroup vs shared-memory tile kernel (same multi-level driver).
//!
//! cargo run --release -p render-ash --example scan_bench -- [--reps 20]
//!
//! Prints one JSON line per (path, n): p50/min GPU ms of the scan dispatches alone (timestamps),
//! and Melem/s. Verifies every result against the CPU scan.
use render_ash::{AshBackend, BackendConfig, ScanMode, cpu_exclusive_scan};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps: usize = args.iter().position(|a| a == "--reps").and_then(|i| args.get(i + 1)).map_or(20, |s| s.parse().unwrap());
    let sizes = [3000usize, 65_536, 262_144, 1 << 20, 4 << 20];
    for mode in [ScanMode::SharedMemory, ScanMode::Subgroup] {
        let cfg = BackendConfig { max_instances: 16, extent: (16, 16), validation: false, frames_in_flight: 2, scan: mode };
        let mut b = match AshBackend::new(cfg, None) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("{mode:?}: unavailable: {e:?}");
                continue;
            }
        };
        for &n in &sizes {
            let data: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(2_654_435_761) % 1000).collect();
            let want = cpu_exclusive_scan(&data);
            let mut t = Vec::new();
            for _ in 0..reps {
                let (got, ms) = b.scan_u32_timed(&data).unwrap();
                assert!(got == want, "{mode:?} n={n} mismatch");
                t.extend(ms);
            }
            t.sort_by(f64::total_cmp);
            let p50 = t[t.len() / 2];
            println!(
                "{{\"path\":\"{:?}\",\"n\":{n},\"gpu_ms_p50\":{p50:.3},\"gpu_ms_min\":{:.3},\"melem_per_s\":{:.0},\"device\":\"{}\"}}",
                b.scan_path(),
                t[0],
                n as f64 / p50 / 1e3,
                b.device_name()
            );
        }
    }
}
