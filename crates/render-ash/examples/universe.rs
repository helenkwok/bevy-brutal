//! Observed external GCNS input by default; synthetic U1 only with explicit --procedural.
use glam::DVec3;
use planet::universe::{Camera, LY, MAX_STARS, Population, SOL, Selection};
use render_ash::{
    AshBackend, BackendConfig,
    universe::{Splat, UniversePass},
};
use std::{io::Write, path::PathBuf, time::Instant};
struct Shot {
    name: &'static str,
    cam: Camera,
    exposure: f64,
    frame: String,
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let procedural = args.iter().any(|a| a == "--procedural");
    let default_out = if procedural { "out/quality/U1_reference" } else { "out/quality/GCNS" };
    let out = PathBuf::from(arg("--out").unwrap_or(default_out.into()));
    std::fs::create_dir_all(&out).unwrap();
    let (w, h) = arg("--size")
        .and_then(|v| v.split_once('x').map(|(w, h)| (w.parse().unwrap(), h.parse().unwrap())))
        .unwrap_or((1280, 720));
    let which = arg("--shot").unwrap_or("all".into());
    let (pop, raw_rows, dataset) = if procedural {
        (
            Population::seeded(0x554e495645525345, MAX_STARS),
            MAX_STARS,
            "explicit procedural reference; NOT observed data",
        )
    } else {
        let path = arg("--catalogue").unwrap_or("/workspace/data/astro/gcns/gcns-v1.bin".into());
        let result = std::fs::File::open(&path).and_then(|f| planet::gcns::read(&mut std::io::BufReader::new(f)));
        let c = result.unwrap_or_else(|e| {
            eprintln!("GCNS input {path}: {e}. Run tools/gcns_ingest.py first. No procedural fallback.");
            std::process::exit(1)
        });
        (
            Population { stars: c.stars },
            c.raw_rows,
            "Gaia EDR3 GCNS / Smart et al. 2021; nearby catalogue, NOT full galaxy; ICRS epoch J2016.0",
        )
    };
    let seed_json = if procedural { "\"554e495645525345\"" } else { "null" };
    let make = |pos, target, origin| {
        let mut c = Camera::look_at(pos, target, DVec3::Z, 60f64.to_radians(), w, h).unwrap();
        c.origin = origin;
        c
    };
    let shots = if procedural {
        let mut shots = vec![
            Shot {
                name: "U1_overview",
                cam: make(DVec3::new(0.0, -7e20, 6e20), DVec3::ZERO, DVec3::ZERO),
                exposure: 3e15,
                frame: "galaxy".into(),
            },
            Shot {
                name: "U1_disk_oblique",
                cam: make(DVec3::new(0.0, -8e20, 1.7e20), DVec3::ZERO, DVec3::ZERO),
                exposure: 3e15,
                frame: "galaxy".into(),
            },
            Shot {
                name: "U1_overview_deep",
                cam: make(DVec3::new(0.0, -7e20, 6e20), DVec3::ZERO, DVec3::ZERO),
                exposure: 3e16,
                frame: "galaxy".into(),
            },
            Shot {
                name: "U1_parallax_A",
                cam: make(DVec3::new(-30.0, -60.0, 20.0) * LY, DVec3::new(0.0, 50.0, 0.0) * LY, SOL),
                exposure: 1e10,
                frame: "sol-relative".into(),
            },
            Shot {
                name: "U1_parallax_B",
                cam: make(DVec3::new(-20.0, -60.0, 20.0) * LY, DVec3::new(10.0, 50.0, 0.0) * LY, SOL),
                exposure: 1e10,
                frame: "sol-relative".into(),
            },
        ];
        for shot in &mut shots {
            if shot.name.starts_with("U1_overview") {
                shot.cam.fov = 75f64.to_radians();
            }
        }
        use coords::frames::{FrameTree, WorldPos};
        let (mut tree, galaxy) = FrameTree::new("galaxy");
        let system = tree.add(galaxy, "sol", SOL, 1e13).unwrap();
        for (name, d) in [
            ("U1_sol_00", 4e18),
            ("U1_sol_01", 1e17),
            ("U1_sol_02", 1e15),
            ("U1_sol_03", 1.01e13),
            ("U1_sol_04", 0.99e13),
        ] {
            let cam = tree.rehome(WorldPos { frame: system, p: DVec3::new(0.0, -d, d * 0.02) });
            let p = tree.convert(cam, system).p;
            shots.push(Shot {
                name,
                cam: make(p, DVec3::ZERO, SOL),
                exposure: 1e10,
                frame: tree.frame(cam.frame).name.clone(),
            });
        }
        shots
    } else {
        let pc = planet::gcns::PARSEC_M;
        let mut nearby = make(DVec3::new(0.0, -180.0, 80.0) * pc, DVec3::ZERO, DVec3::ZERO);
        nearby.fov = 75f64.to_radians();
        vec![
            Shot { name: "GCNS_nearby", cam: nearby, exposure: 3e2, frame: "Sol-local ICRS J2016.0".into() },
            Shot {
                name: "GCNS_parallax_A",
                cam: make(DVec3::ZERO, DVec3::X, DVec3::ZERO),
                exposure: 3e2,
                frame: "Sol-local ICRS J2016.0".into(),
            },
            Shot {
                name: "GCNS_parallax_B",
                cam: make(DVec3::Y * pc, DVec3::Y * pc + DVec3::X * pc, DVec3::ZERO),
                exposure: 3e2,
                frame: "Sol-local ICRS J2016.0".into(),
            },
        ]
    };
    let mut be = AshBackend::new(
        BackendConfig { max_instances: 16, extent: (w, h), validation: true, ..Default::default() },
        None,
    )
    .unwrap();
    let mut pass = UniversePass::new(&mut be, w, h).unwrap();
    let mut selected = Selection::new();
    let mut splats = Vec::with_capacity(MAX_STARS);
    let mut rgb = Vec::with_capacity(w as usize * h as usize * 3);
    let device_json = format!("{:?}", be.ctx.device_name);
    for s in shots.into_iter().filter(|s| which == "all" || which.split(',').any(|x| s.name == x)) {
        let t = Instant::now();
        selected.update(&pop.stars, &s.cam, MAX_STARS);
        let select_ms = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        splats.clear();
        splats.extend(selected.sources.iter().map(Splat::from_source));
        let prepare_ms = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let (hdr, gpu) = pass.render(&mut be, &splats).unwrap();
        let wall = t.elapsed().as_secs_f64() * 1000.0;
        assert!(hdr.iter().flatten().all(|v| v.is_finite()));
        let energy: Vec<f64> = (0..3).map(|c| hdr.iter().map(|p| p[c] as f64).sum()).collect();
        let cpu: f64 = selected.sources.iter().map(|p| p.energy(w, h) * p.rgb[0]).sum();
        let mut raw = std::fs::File::create(out.join(format!("{}.hdr.bin", s.name))).unwrap();
        raw.write_all(bytemuck::cast_slice(hdr)).unwrap();
        rgb.clear();
        for p in hdr {
            let a = [p[0] as f64 * s.exposure, p[1] as f64 * s.exposure, p[2] as f64 * s.exposure];
            let peak = a.into_iter().fold(0.0, f64::max);
            for v in a {
                let v = v / (1.0 + peak);
                let srgb = if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
                rgb.push((srgb.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            }
        }
        let f = std::fs::File::create(out.join(format!("{}.png", s.name))).unwrap();
        let mut png = png::Encoder::new(f, w, h);
        png.set_color(png::ColorType::Rgb);
        png.set_depth(png::BitDepth::Eight);
        png.write_header().unwrap().write_image_data(&rgb).unwrap();
        let errors = render_ash::validation::message_counts().0;
        let bright = selected
            .sources
            .iter()
            .take(32)
            .map(|s| format!("{{\"id\":\"{}\",\"pixel\":{:?},\"flux\":{}}}", s.id, s.center, s.flux))
            .collect::<Vec<_>>()
            .join(",");
        let dropped = selected.candidates - selected.sources.len();
        let origin = s.cam.origin.to_array();
        let forward = s.cam.forward.to_array();
        let up = s.cam.up.to_array();
        let fov = s.cam.fov;
        let sol_json = selected
            .sources
            .iter()
            .find(|s| s.id == 0)
            .map(|s| format!("{{\"pixel\":{:?},\"flux\":{}}}", s.center, s.flux))
            .unwrap_or("null".into());
        let json = format!(
            "{{\"shot\":\"{}\",\"population\":\"{dataset}\",\"seed\":{seed_json},\"raw_catalogue_rows\":{raw_rows},\"selection_policy\":\"scan every loaded record; brightest visible flux first, exact u64 source_id tie-break; cap65536; no hard median100pc cut\",\"photometry_note\":\"GCNS: monochrome G-band relative Vega-zero flux, not RGB or bolometric luminosity; no extinction, variability or epoch propagation; translated observer inverse-square approximation from posterior median distances\",\"stars\":{},\"selected\":{},\"candidates\":{},\"dropped_visible_by_gpu_cap\":{dropped},\"size\":[{},{}],\"exposure\":{},\"epic_grade\":null,\"frame\":\"{}\",\"camera_relative_m\":[{},{},{}],\"flux_rgb\":{:?},\"cpu_red_flux_clipped\":{},\"select_ms\":{},\"prepare_ms\":{},\"gpu_ms\":{},\"render_wall_ms\":{},\"validation_errors\":{},\"device_name\":{device_json},\"timing_note\":\"offscreen measurement, CPU Vulkan is not hardware 60fps proof\",\"camera_origin_m\":{origin:?},\"camera_forward\":{forward:?},\"camera_up\":{up:?},\"fov_y_rad\":{fov},\"sol_anchor\":{sol_json},\"bright_sources\":[{bright}],\"hdr_format\":\"RGBA f32 LE integrated flux per pixel, unexposed\"}}",
            s.name,
            pop.stars.len(),
            selected.sources.len(),
            selected.candidates,
            w,
            h,
            s.exposure,
            s.frame,
            s.cam.pos.x,
            s.cam.pos.y,
            s.cam.pos.z,
            energy,
            cpu,
            select_ms,
            prepare_ms,
            gpu.gpu_ms,
            wall,
            errors
        );
        std::fs::write(out.join(format!("{}.json", s.name)), &json).unwrap();
        eprintln!("{json}");
    }
    pass.free(&mut be);
    drop(be);
    assert_eq!(render_ash::validation::message_counts().0, 0);
}
