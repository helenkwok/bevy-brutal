//! RS1 stills + streaming flight from the observed real-star tile file (RSTILE01).
//!
//!   cargo run -p render-ash --release --example real_starfield -- \
//!     --tiles /workspace/data/astro/real_starfield/tiles/rs1.rstile --out out/quality/real_starfield
//!
//! Every splat is one catalogue row (GCNS / Gaia DR3+Bailer-Jones / Hipparcos). Stars are drawn
//! only if brighter than the streaming completeness magnitude, so tile boundaries never show as
//! density steps. Exposure is the fixed physical GT1 present exposure (EPIC_GRADE off).
use glam::DVec3;
use planet::gcns::PARSEC_M;
use planet::starfield::{Decoded, FileBacking, TileStream, apparent_mag, decode_block, project};
use planet::universe::{Camera, Source};
use render_ash::{
    AshBackend, BackendConfig,
    universe::{PRESENT_EXPOSURE, Splat, UniversePass, splats_with_halo},
};
use std::path::PathBuf;
use std::time::Instant;

/// ICRS -> Galactic rotation (Hipparcos convention, ESA 1997 vol.1 sec 1.5.3).
const R_GAL: [[f64; 3]; 3] = [
    [-0.054_875_560_416_215_4, -0.873_437_090_234_885, -0.483_835_015_548_713_2],
    [0.494_109_427_875_583_7, -0.444_829_629_960_011_2, 0.746_982_244_497_219],
    [-0.867_666_149_019_004_7, -0.198_076_373_431_201_5, 0.455_983_776_175_066_9],
];
fn radec(ra: f64, dec: f64) -> DVec3 {
    let (a, d) = (ra.to_radians(), dec.to_radians());
    DVec3::new(d.cos() * a.cos(), d.cos() * a.sin(), d.sin())
}
fn gal(l: f64, b: f64) -> DVec3 {
    let g = radec(l, b); // same spherical->cartesian form in the galactic frame
    // icrs = R^T g
    DVec3::new(
        R_GAL[0][0] * g.x + R_GAL[1][0] * g.y + R_GAL[2][0] * g.z,
        R_GAL[0][1] * g.x + R_GAL[1][1] * g.y + R_GAL[2][1] * g.z,
        R_GAL[0][2] * g.x + R_GAL[1][2] * g.y + R_GAL[2][2] * g.z,
    )
}

struct View {
    name: String,
    pos_pc: DVec3,
    look: DVec3,
    up: DVec3,
    fov_deg: f64,
    note: String,
}

fn views() -> Vec<View> {
    let n = DVec3::Z;
    let gn = gal(0.0, 90.0);
    let v = |name: &str, pos: DVec3, look: DVec3, up: DVec3, fov: f64, note: &str| View {
        name: name.into(),
        pos_pc: pos,
        look: look.normalize(),
        up,
        fov_deg: fov,
        note: note.into(),
    };
    let to_orion = radec(83.8, -1.2);
    let gc = gal(0.0, 0.0);
    let l90 = gal(90.0, 0.0);
    vec![
        v("sol_orion", DVec3::ZERO, to_orion, n, 50.0, "From Sol: Orion, celestial north up"),
        v("sol_big_dipper", DVec3::ZERO, radec(178.0, 55.0), n, 50.0, "From Sol: Ursa Major / Big Dipper"),
        v("sol_crux", DVec3::ZERO, radec(197.0, -61.0), n, 50.0, "From Sol: Crux + Alpha/Beta Centauri"),
        v(
            "sol_galactic_centre",
            DVec3::ZERO,
            gc,
            gn,
            60.0,
            "From Sol: towards Galactic centre (Sgr), galactic north up",
        ),
        v("sol_cygnus", DVec3::ZERO, gal(80.0, 0.0), gn, 60.0, "From Sol: Cygnus / Milky Way, galactic north up"),
        v("sol_anticentre", DVec3::ZERO, gal(180.0, 0.0), gn, 60.0, "From Sol: Galactic anticentre (Taurus/Auriga)"),
        v(
            "mid_250pc_toward_orion",
            to_orion * 250.0,
            to_orion,
            n,
            60.0,
            "250 pc from Sol towards Orion OB1, looking on",
        ),
        v("out_500pc_l90_back_at_sol", l90 * 500.0, -l90, gn, 70.0, "500 pc out (l=90,b=0), looking back at Sol"),
        v("out_500pc_toward_gc", gc * 500.0, gc, gn, 70.0, "500 pc towards Galactic centre, looking on"),
        v(
            "out_1kpc_l90_back_at_sol",
            l90 * 1000.0,
            -l90,
            gn,
            70.0,
            "1 kpc out (l=90,b=0) at the sample edge, looking back",
        ),
        v(
            "out_1kpc_above_plane_down",
            gal(0.0, 90.0) * 1000.0,
            -gn,
            gal(0.0, 0.0),
            70.0,
            "1 kpc above the plane (NGP), looking down at Sol",
        ),
    ]
}

fn write_png(path: &std::path::Path, w: u32, h: u32, rgba: &[u8]) {
    let f = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(f, w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(rgba).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let tiles = PathBuf::from(arg("--tiles").unwrap_or("/workspace/data/astro/real_starfield/tiles/rs1.rstile".into()));
    let out = PathBuf::from(arg("--out").unwrap_or("out/quality/real_starfield".into()));
    let (w, h): (u32, u32) = arg("--size")
        .and_then(|v| {
            let (a, b) = v.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((1280, 720));
    let budget: u64 = arg("--budget").and_then(|v| v.parse().ok()).unwrap_or(2_000_000);
    let resident: u64 = arg("--resident").and_then(|v| v.parse().ok()).unwrap_or(3_000_000);
    let cap: usize = arg("--cap").and_then(|v| v.parse().ok()).unwrap_or(1_000_000);
    let mag_limit: f64 = arg("--mag-limit").and_then(|v| v.parse().ok()).unwrap_or(99.0);
    let only = arg("--only");
    let flight_steps: usize = arg("--flight").and_then(|v| v.parse().ok()).unwrap_or(0);
    std::fs::create_dir_all(&out).unwrap();
    let mut stream = TileStream::new(FileBacking::open(&tiles).expect("tile file"), resident).expect("RSTILE01");
    let hdr = stream.index.header;
    eprintln!(
        "tiles {}: stars={} nodes={} src_counts={:?} root_size={}pc",
        tiles.display(),
        hdr.star_count,
        hdr.node_count,
        hdr.src_counts,
        hdr.root_size
    );
    let mut be = AshBackend::new(
        BackendConfig { max_instances: 16, extent: (w, h), validation: true, ..Default::default() },
        None,
    )
    .expect("AshBackend");
    let device = be.ctx.device_name.clone();
    let mut pass = UniversePass::with_capacity(&mut be, w, h, cap + cap / 4).unwrap();
    let lut = stream.index.lut.clone();
    let mut records = Vec::new();
    let mut decoded: Vec<Decoded> = Vec::new();
    let mut render_one = |stream: &mut TileStream<FileBacking>,
                          be: &mut AshBackend,
                          pass: &mut UniversePass,
                          name: &str,
                          pos_pc: DVec3,
                          look: DVec3,
                          up: DVec3,
                          fov_deg: f64,
                          note: &str,
                          save: bool| {
        let t0 = Instant::now();
        let (plan, st) = stream.update(pos_pc, budget).unwrap();
        let t1 = Instant::now();
        decoded.clear();
        for n in &plan.nodes {
            decode_block(&stream.index.nodes[*n as usize], &stream.cache[n], &mut decoded);
        }
        let decode_ms = t1.elapsed().as_secs_f64() * 1e3;
        let cam =
            Camera::look_at(pos_pc * PARSEC_M, (pos_pc + look) * PARSEC_M, up, fov_deg.to_radians(), w, h).unwrap();
        let t2 = Instant::now();
        let (sources, ps) = project(&decoded, &cam, plan.m_complete, mag_limit, cap, &lut);
        let project_ms = t2.elapsed().as_secs_f64() * 1e3;
        let mut splats: Vec<Splat> = Vec::with_capacity(sources.len() + 4096);
        let halos = splats_with_halo(&sources, PRESENT_EXPOSURE as f64, &mut splats);
        assert!(splats.len() <= pass.capacity(), "core+halo splats exceed GPU buffer");
        let timing = pass.render_live(be, &splats, PRESENT_EXPOSURE).unwrap();
        let by_src = ps.by_src;
        if save {
            let rgba = pass.read_sdr_rgba(be).unwrap();
            write_png(&out.join(format!("{name}.png")), w, h, &rgba);
        }
        let total_ms = t0.elapsed().as_secs_f64() * 1e3;
        let rec = format!(
            "{{\"name\":\"{name}\",\"note\":\"{note}\",\"pos_pc\":{:?},\"dist_from_sol_pc\":{:.3},\"forward\":{:?},\"right\":{:?},\"up\":{:?},\"fov_y_deg\":{fov_deg},\"size\":[{w},{h}],\"m_complete\":{},\"mag_limit\":{mag_limit},\"nodes_planned\":{},\"nodes_loaded_now\":{},\"nodes_evicted_now\":{},\"bytes_read_now\":{},\"bytes_read_total\":{},\"resident_stars\":{},\"planned_stars\":{},\"in_view_brighter_than_cut\":{},\"submitted\":{},\"halo_splats\":{halos},\"dropped_by_cap\":{},\"submitted_by_src\":{{\"gcns\":{},\"gaia_dr3_bj\":{},\"hipparcos2\":{}}},\"plan_ms\":{:.2},\"load_ms\":{:.2},\"decode_ms\":{decode_ms:.2},\"project_ms\":{project_ms:.2},\"submit_ms\":{:.2},\"gpu_ms\":{:.2},\"total_ms\":{total_ms:.2}}}",
            pos_pc.to_array(),
            pos_pc.length(),
            cam.forward.to_array(),
            cam.right.to_array(),
            cam.up.to_array(),
            if plan.m_complete.is_finite() { format!("{:.3}", plan.m_complete) } else { "null".into() },
            st.nodes_planned,
            st.nodes_loaded_now,
            st.nodes_evicted_now,
            st.bytes_read_now,
            st.bytes_read_total,
            st.resident_stars,
            plan.stars,
            ps.candidates,
            ps.submitted,
            ps.dropped_by_cap,
            by_src[0],
            by_src[1],
            by_src[2],
            st.plan_ms,
            st.load_ms,
            timing.submit_ms,
            timing.gpu_ms,
        );
        eprintln!("{rec}");
        rec
    };
    for v in views() {
        if only.as_deref().is_some_and(|o| !o.split(',').any(|x| x == v.name)) {
            continue;
        }
        records.push(render_one(
            &mut stream,
            &mut be,
            &mut pass,
            &v.name,
            v.pos_pc,
            v.look,
            v.up,
            v.fov_deg,
            &v.note,
            true,
        ));
    }
    // Streaming flight: Sol -> 1 kpc along l=90 (looking ahead), recording per-step I/O.
    let mut flight = Vec::new();
    if flight_steps > 0 {
        let dir = gal(90.0, 0.0);
        for i in 0..=flight_steps {
            let d = 1000.0 * i as f64 / flight_steps as f64;
            let name = format!("flight_{i:03}");
            let save = i % (flight_steps / 4).max(1) == 0;
            flight.push(render_one(
                &mut stream,
                &mut be,
                &mut pass,
                &name,
                dir * d,
                dir,
                gal(0.0, 90.0),
                70.0,
                "flight Sol->1kpc l=90",
                save,
            ));
        }
    }
    // Milky Way band check from Sol: full-sky star surface density vs |galactic latitude| for
    // observed stars brighter than each cut (well inside the completeness magnitude).
    let (plan, _) = stream.update(DVec3::ZERO, budget).unwrap();
    decoded.clear();
    for n in &plan.nodes {
        decode_block(&stream.index.nodes[*n as usize], &stream.cache[n], &mut decoded);
    }
    // Dump every engine star brighter than G=7 seen from Sol (RA/Dec from tile positions) for the
    // independent naked-eye completeness check in tools/real_starfield/bright_check.py.
    {
        let mut csv = String::from("src,id,ra_deg,dec_deg,g_apparent\n");
        for d in &decoded {
            let r = d.pos_pc.length();
            let m = apparent_mag(d, DVec3::ZERO);
            if r > 0.0 && m < 7.0 {
                let u = d.pos_pc / r;
                csv += &format!(
                    "{},{},{:.6},{:.6},{:.3}\n",
                    d.src,
                    d.id,
                    u.y.atan2(u.x).to_degrees().rem_euclid(360.0),
                    u.z.asin().to_degrees(),
                    m
                );
            }
        }
        std::fs::write(out.join("engine_bright_g7.csv"), csv).unwrap();
    }
    let mut band_json = Vec::new();
    for band_mag in [7.0f64, 9.0] {
        let mut bins = [0usize; 9];
        for d in &decoded {
            let r = d.pos_pc.length();
            if r < 1e-9 || apparent_mag(d, DVec3::ZERO) >= band_mag.min(plan.m_complete) {
                continue;
            }
            let u = d.pos_pc / r;
            let zg = R_GAL[2][0] * u.x + R_GAL[2][1] * u.y + R_GAL[2][2] * u.z; // sin b
            let b = zg.clamp(-1.0, 1.0).asin().to_degrees().abs();
            bins[((b / 10.0) as usize).min(8)] += 1;
        }
        let dens = |i: usize| {
            let (b1, b2) = ((i * 10) as f64, (i * 10 + 10) as f64);
            let sr = 4.0 * std::f64::consts::PI * (b2.to_radians().sin() - b1.to_radians().sin());
            bins[i] as f64 / (sr * (180.0 / std::f64::consts::PI).powi(2))
        };
        let ratio = dens(0)
            / ((dens(6) * (70f64.to_radians().sin() - 60f64.to_radians().sin())
                + dens(7) * (80f64.to_radians().sin() - 70f64.to_radians().sin())
                + dens(8) * (1.0 - 80f64.to_radians().sin()))
                / (1.0 - 60f64.to_radians().sin()));
        let per: Vec<String> = (0..9).map(|i| format!("{:.4}", dens(i))).collect();
        eprintln!("Milky Way band (G<{band_mag}, from Sol): |b|<10 / |b|>60 density = {ratio:.2}; counts {bins:?}");
        band_json.push(format!(
            "{{\"mag_lt\":{band_mag},\"ratio_absb_lt10_over_gt60\":{ratio:.3},\"counts_per_10deg_absb\":{bins:?},\"per_deg2\":[{}]}}",
            per.join(",")
        ));
    }
    // All-sky view from Sol, Galactic Hammer-Aitoff (equal-area, l=0 centre, l increasing left,
    // north up). Same observed stars, flux, colour LUT, halo PSF and tonemap as the views; only
    // the projection differs. This is the direct Milky Way band check.
    let mut allsky_note = String::from("skipped (--only)");
    if only.is_none() {
        let (aw, ah) = (2048u32, 1024u32);
        let mut sky_pass = UniversePass::with_capacity(&mut be, aw, ah, cap + cap / 4).unwrap();
        let m_cut = plan.m_complete.min(mag_limit);
        let mut src: Vec<Source> = Vec::new();
        for d in &decoded {
            let r = d.pos_pc.length();
            if r * 206_264.806 < planet::starfield::MIN_DRAW_AU {
                continue;
            }
            let m = apparent_mag(d, DVec3::ZERO);
            if m >= m_cut {
                continue;
            }
            let u = d.pos_pc / r;
            let g = DVec3::new(
                R_GAL[0][0] * u.x + R_GAL[0][1] * u.y + R_GAL[0][2] * u.z,
                R_GAL[1][0] * u.x + R_GAL[1][1] * u.y + R_GAL[1][2] * u.z,
                R_GAL[2][0] * u.x + R_GAL[2][1] * u.y + R_GAL[2][2] * u.z,
            );
            let (l, b) = (g.y.atan2(g.x), g.z.clamp(-1.0, 1.0).asin());
            let den = (1.0 + b.cos() * (l * 0.5).cos()).sqrt();
            let x = 2.0 * std::f64::consts::SQRT_2 * b.cos() * (l * 0.5).sin() / den;
            let y = std::f64::consts::SQRT_2 * b.sin() / den;
            let px = aw as f64 * 0.5 - x / (2.0 * std::f64::consts::SQRT_2) * (aw as f64 * 0.49);
            let py = ah as f64 * 0.5 - y / std::f64::consts::SQRT_2 * (ah as f64 * 0.49);
            let rgb = if d.colour == planet::starfield::NO_COLOUR {
                [1.0; 3]
            } else {
                let c = lut[d.colour as usize];
                [c[0] as f64, c[1] as f64, c[2] as f64]
            };
            src.push(Source {
                id: planet::starfield::render_id(d.src, d.id),
                center: [px, py],
                flux: 10f64.powf(-0.4 * m),
                rgb,
            });
        }
        src.sort_unstable_by(|a, b| b.flux.total_cmp(&a.flux).then(a.id.cmp(&b.id)));
        src.truncate(cap);
        let mut splats = Vec::with_capacity(src.len() + 8192);
        splats_with_halo(&src, PRESENT_EXPOSURE as f64, &mut splats);
        sky_pass.render_live(&mut be, &splats, PRESENT_EXPOSURE).unwrap();
        let rgba = sky_pass.read_sdr_rgba(&mut be).unwrap();
        write_png(&out.join("sol_allsky_galactic_aitoff.png"), aw, ah, &rgba);
        sky_pass.free(&mut be);
        allsky_note = format!(
            "{{\"file\":\"sol_allsky_galactic_aitoff.png\",\"size\":[{aw},{ah}],\"projection\":\"Hammer-Aitoff galactic, l=0 centre, l increases left\",\"m_cut\":{m_cut:.3},\"stars\":{}}}",
            src.len()
        );
        eprintln!("all-sky: {allsky_note}");
    }
    pass.free(&mut be);
    drop(be);
    let (errors, warnings) = render_ash::validation::message_counts();
    let json = format!(
        "{{\n\"tiles\":{:?},\n\"tile_stars\":{},\n\"tile_src_counts\":{{\"gcns\":{},\"gaia_dr3_bj\":{},\"hipparcos2\":{}}},\n\"device\":{:?},\n\"timing_note\":\"software rasteriser (lavapipe) timings are NOT hardware fps evidence\",\n\"exposure\":{},\n\"epic_grade\":false,\n\"budget_stars\":{budget},\n\"resident_budget\":{resident},\n\"gpu_cap\":{cap},\n\"validation_errors\":{errors},\n\"validation_warnings\":{warnings},\n\"allsky\":{},\n\"band_density_from_sol\":[{}],\n\"views\":[\n{}\n],\n\"flight\":[\n{}\n]\n}}\n",
        tiles.display().to_string(),
        hdr.star_count,
        hdr.src_counts[0],
        hdr.src_counts[1],
        hdr.src_counts[2],
        device,
        PRESENT_EXPOSURE,
        if allsky_note.starts_with('{') { allsky_note.clone() } else { format!("{allsky_note:?}") },
        band_json.join(","),
        records.join(",\n"),
        flight.join(",\n")
    );
    let name = if flight_steps > 0 && records.is_empty() { "flight.json" } else { "views.json" };
    std::fs::write(out.join(name), json).unwrap();
    eprintln!("validation errors={errors} warnings={warnings}");
    assert_eq!(errors, 0, "Vulkan validation errors");
}
