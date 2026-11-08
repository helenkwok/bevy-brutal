//! Real CNS5/DESI positional diagnostic. Equal marker weights are NOT apparent photometry.
//! No synthetic fallback, catalogue merge, galaxy shapes, or change to GCNS default.
use glam::DVec3;
use planet::{
    gcns::PARSEC_M,
    observed::{self, Kind},
    universe::{Camera, MAX_STARS, Source},
};
use render_ash::{
    AshBackend, BackendConfig,
    universe::{Splat, UniversePass},
};
use std::{
    io::{self, Write},
    path::PathBuf,
    time::Instant,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let kind = arg("--kind").unwrap_or("cns5".into());
    if !["cns5", "desi"].contains(&kind.as_str()) {
        return Err("--kind must be cns5 or desi".into());
    }
    let path = PathBuf::from(arg("--catalogue").unwrap_or(format!("/workspace/data/astro/{kind}/diagnostic-v1.bin")));
    let out = PathBuf::from(arg("--out").unwrap_or(format!("out/quality/observed/{kind}")));
    let c = observed::read(&mut io::BufReader::new(std::fs::File::open(&path)?))?;
    if (kind == "cns5") != (c.kind == Kind::Cns5) {
        return Err("catalogue kind mismatch".into());
    }
    std::fs::create_dir_all(&out)?;
    let (w, h) = (1280, 720);
    let make = |pos, target, up, fov: f64| Camera::look_at(pos, target, up, fov.to_radians(), w, h).unwrap();
    let pc = PARSEC_M;
    let mpc = pc * 1e6;
    let direction = DVec3::new(155f64.to_radians().cos(), 155f64.to_radians().sin(), 0.);
    let tangent = DVec3::new(-direction.y, direction.x, 0.);
    let shots = if c.kind == Kind::Cns5 {
        vec![
            ("nearby", make(DVec3::new(0., -65., 22.) * pc, DVec3::ZERO, DVec3::Z, 55.)),
            ("sky", make(DVec3::ZERO, DVec3::X, DVec3::Z, 60.)),
        ]
    } else {
        let mid = direction * 1000. * mpc;
        vec![
            ("cone", make(mid + DVec3::Z * 2200. * mpc, mid, tangent, 30.)),
            ("sky", make(DVec3::ZERO, direction, DVec3::Z, 7.)),
        ]
    };
    let label = if c.kind == Kind::Cns5 {
        "CNS5 current GAVO snapshot, not certified full published5931; UNMERGED native per-source epochs; SNR10 reciprocal-parallax coordinates; ALL equal position markers, including IR-only sources, NOT G-band brightness. No motion propagation. Upstream rights unresolved."
    } else {
        "DESI DR1 observed selected field150<=RA<160,-2<=Dec<2,0.01<=z<0.5; primary ZWARN0 fibre0 finite zerr; OBJTYPE unavailable/not applied. Equal GALAXY-classified diagnostic markers, NOT luminosity/shapes. FlatLCDM H0=70 Om=0.3 Ode=0.7 radiation0 comoving coordinates; lightcone/redshift-space caveats; no common J2016 epoch."
    };
    let mut be = AshBackend::new(
        BackendConfig { max_instances: 16, extent: (w, h), validation: true, ..Default::default() },
        None,
    )
    .map_err(|e| io::Error::other(format!("{e:?}")))?;
    assert!(
        be.ctx.validation_enabled && be.ctx.sync_validation,
        "actual Vulkan and synchronization validation required"
    );
    let mut pass = UniversePass::new(&mut be, w, h).map_err(|e| io::Error::other(format!("{e:?}")))?;
    let sha = c.raw_sha256.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        for (name, cam) in shots {
            let start = Instant::now();
            let (markers, visible) = observed::select_markers(&c.points, &cam, MAX_STARS);
            let select_ms = start.elapsed().as_secs_f64() * 1000.;
            // Explicit rasterization boundary: generic display-channel PSF weight of ONE,
            // never a Star, stellar luminosity or inverse-square function of distance.
            let sources: Vec<_> =
                markers.iter().map(|m| Source { id: m.id, center: m.center, flux: 1., rgb: [1.; 3] }).collect();
            let splats: Vec<_> = sources.iter().map(Splat::from_source).collect();
            let start = Instant::now();
            let (hdr, timing) = pass.render(&mut be, &splats).map_err(|e| io::Error::other(format!("{e:?}")))?;
            let render_ms = start.elapsed().as_secs_f64() * 1000.;
            assert!(hdr.iter().flatten().all(|v| v.is_finite()));
            let gpu_energy: f64 = hdr.iter().map(|v| v[0] as f64).sum();
            let cpu_energy: f64 = sources.iter().map(|s| s.energy(w, h)).sum();
            assert!(
                gpu_energy > 0. && (gpu_energy / cpu_energy - 1.).abs() < 0.005,
                "diagnostic marker energy mismatch"
            );
            let mut raw = std::fs::File::create(out.join(format!("{name}.hdr.bin")))?;
            raw.write_all(bytemuck::cast_slice(hdr))?;
            let exposure = 1.5;
            let mut rgb = Vec::with_capacity(w as usize * h as usize * 3);
            for pixel in hdr {
                for channel in pixel.iter().take(3) {
                    let x = *channel as f64 * exposure;
                    let x = x / (1. + x);
                    let x = if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1. / 2.4) - 0.055 };
                    rgb.push((x.clamp(0., 1.) * 255. + 0.5) as u8);
                }
            }
            let f = std::fs::File::create(out.join(format!("{name}.png")))?;
            let mut png = png::Encoder::new(f, w, h);
            png.set_color(png::ColorType::Rgb);
            png.set_depth(png::BitDepth::Eight);
            png.write_header()?.write_image_data(&rgb)?;
            let errors = render_ash::validation::message_counts().0;
            assert_eq!(errors, 0);
            let json = format!(
                "{{\"kind\":\"{kind}\",\"view\":\"{name}\",\"label\":\"{label}\",\"raw_sha256\":\"{sha}\",\"raw_rows\":{},\"loaded_eligible\":{},\"visible\":{visible},\"submitted\":{},\"dropped_by_cap\":{},\"cap\":{MAX_STARS},\"policy\":\"all loaded records scanned; nearest visible then exact ID; equal marker weight1\",\"exposure\":{exposure},\"camera_m\":{:?},\"forward\":{:?},\"up\":{:?},\"fov_y_rad\":{},\"size\":[{w},{h}],\"raw_data_sidecar\":\"diagnostic-v1.json beside input\",\"cpu_energy\":{cpu_energy},\"gpu_energy\":{gpu_energy},\"select_ms\":{select_ms},\"gpu_ms\":{},\"render_wall_ms\":{render_ms},\"validation_errors\":{errors},\"sync_validation\":true,\"device\":{:?},\"timing_note\":\"offscreen; lavapipe is not 60fps hardware proof\",\"epic_grade\":false}}",
                c.raw_rows,
                c.points.len(),
                markers.len(),
                visible - markers.len(),
                cam.pos.to_array(),
                cam.forward.to_array(),
                cam.up.to_array(),
                cam.fov,
                timing.gpu_ms,
                be.ctx.device_name
            );
            std::fs::write(out.join(format!("{name}.json")), &json)?;
            eprintln!("{json}");
        }
        Ok(())
    })();
    pass.free(&mut be);
    drop(be);
    result?;
    assert_eq!(render_ash::validation::message_counts().0, 0);
    Ok(())
}
