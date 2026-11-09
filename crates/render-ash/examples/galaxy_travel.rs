//! GT1: windowed continuous flight through observed GCNS (external file only).
//!
//! Launch (interactive, needs a display + cursor grab):
//!   cargo run -p render-ash --example galaxy_travel --features windowed -- \
//!     --catalogue /workspace/data/astro/gcns/gcns-v1.bin
//!
//! Deterministic headless replay + PNG sheet (no display required):
//!   cargo run -p render-ash --example galaxy_travel --features windowed --release -- \
//!     --catalogue /workspace/data/astro/gcns/gcns-v1.bin \
//!     --replay /path/to/inputs.json --frames 120 --capture out/quality/GT1_sheet \
//!     --size 640x360
//!
//! Controls (interactive): W/S forward/back, A/D strafe, R/F up/down, mouse look,
//! Shift/Ctrl speed gears (±1), Esc release grab / quit on second Esc, Q quit.
//! Status: dynamic **window title** (xyz pc, yaw/pitch, speed, sel, grab/focus, coverage warning).
//! There is no in-frame overlay HUD yet — title is the live readout.
//! Focus loss clears key latches (no sticky WASD) and releases cursor grab.
//! Units: speed in pc/s; positions Sol-local ICRS J2016 metres.
//! Catalogue: Gaia EDR3 GCNS / Smart et al. 2021 — nearby survey, NOT full Milky Way.
#![allow(clippy::too_many_arguments, clippy::collapsible_if)]
use glam::DVec3;
use planet::gcns::PARSEC_M;
use planet::travel::{
    Coverage, FlightKeys, FlightState, InputSample, NOMINAL_SURVEY_PC, StarIndex, default_look_source_id, status_title,
};
use planet::universe::MAX_STARS;
use render_ash::{
    AshBackend, BackendConfig,
    universe::{PRESENT_EXPOSURE, Splat, UniversePass},
};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug)]
struct SampleRec {
    sample: InputSample,
    selected_ids: Vec<u64>,
    pos: [f64; 3],
    coverage_outside: bool,
}

fn parse_args() -> Args {
    let args: Vec<_> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let flag = |k: &str| args.iter().any(|a| a == k);
    let (w, h) = arg("--size")
        .and_then(|v| {
            let (w, h) = v.split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?))
        })
        .unwrap_or((1280, 720));
    Args {
        catalogue: PathBuf::from(arg("--catalogue").unwrap_or("/workspace/data/astro/gcns/gcns-v1.bin".into())),
        replay: arg("--replay").map(PathBuf::from),
        record: arg("--record").map(PathBuf::from),
        capture: arg("--capture").map(PathBuf::from),
        frames: arg("--frames").and_then(|v| v.parse().ok()).unwrap_or(0),
        width: w,
        height: h,
        headless: flag("--headless") || arg("--replay").is_some(),
        validation: !flag("--no-validation"),
    }
}
struct Args {
    catalogue: PathBuf,
    replay: Option<PathBuf>,
    record: Option<PathBuf>,
    capture: Option<PathBuf>,
    frames: usize,
    width: u32,
    height: u32,
    headless: bool,
    validation: bool,
}

fn load_catalogue(path: &Path) -> (StarIndex, usize) {
    let f = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("GCNS input {}: {e}. Run tools/gcns_ingest.py first. No procedural fallback.", path.display());
        std::process::exit(1);
    });
    let c = planet::gcns::read(&mut std::io::BufReader::new(f)).unwrap_or_else(|e| {
        eprintln!("GCNS parse {}: {e}", path.display());
        std::process::exit(1);
    });
    let index = StarIndex::build(&c.stars, 5.0 * PARSEC_M);
    (index, c.raw_rows)
}

fn look_target(index: &StarIndex) -> DVec3 {
    let id = default_look_source_id();
    index.stars().iter().find(|s| s.id == id).map(|s| s.pos).unwrap_or(DVec3::X * PARSEC_M)
}

fn write_png(path: &Path, w: u32, h: u32, rgba: &[u8]) {
    let f = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(f, w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(rgba).unwrap();
}

fn run_replay(args: &Args, index: &StarIndex, raw_rows: usize) {
    let samples: Vec<InputSample> = if let Some(path) = &args.replay {
        let text = std::fs::read_to_string(path).expect("replay file");
        parse_replay_json(&text)
    } else {
        // Built-in deterministic path toward the default look source, then a short strafe + pitch.
        let mut v = Vec::new();
        for i in 0..args.frames.max(90) {
            let gear = if i < 30 {
                2
            } else if i < 60 {
                1
            } else {
                0
            };
            v.push(InputSample {
                dt: 1.0 / 60.0,
                forward: if i < 70 { 1.0 } else { 0.0 },
                right: if (40..55).contains(&i) { 0.5 } else { 0.0 },
                up: 0.0,
                yaw_delta: if (20..40).contains(&i) { 0.01 } else { 0.0 },
                pitch_delta: if (50..65).contains(&i) { -0.008 } else { 0.0 },
                speed_gear: gear,
                teleport: None,
            });
        }
        v
    };
    let mut be = AshBackend::new(
        BackendConfig {
            max_instances: 16,
            extent: (args.width, args.height),
            validation: args.validation,
            ..Default::default()
        },
        None,
    )
    .expect("AshBackend");
    let device = be.ctx.device_name.clone();
    let mut pass = UniversePass::new(&mut be, args.width, args.height).unwrap();
    let mut flight = FlightState::sol_looking_at(look_target(index));
    let mut sel = planet::universe::Selection::new();
    let mut splats = Vec::with_capacity(MAX_STARS);
    let mut records = Vec::new();
    let mut select_ms = Vec::new();
    let mut submit_ms = Vec::new();
    let mut total_ms = Vec::new();
    let capture_dir = args.capture.clone();
    if let Some(dir) = &capture_dir {
        std::fs::create_dir_all(dir).unwrap();
    }
    let n = if args.frames > 0 { args.frames.min(samples.len()) } else { samples.len() };
    for (fi, sample) in samples.into_iter().take(n).enumerate() {
        let t0 = Instant::now();
        flight.integrate(&sample);
        let cam = flight.camera(args.width, args.height).expect("camera");
        let stats = index.select(&cam, MAX_STARS, &mut sel);
        select_ms.push(stats.select_ms);
        splats.clear();
        splats.extend(sel.sources.iter().map(Splat::from_source));
        let timing = pass.render_live(&mut be, &splats, PRESENT_EXPOSURE).unwrap();
        submit_ms.push(timing.submit_ms);
        let cov = Coverage::at(flight.pos);
        records.push(SampleRec {
            sample,
            selected_ids: sel.sources.iter().map(|s| s.id).collect(),
            pos: flight.pos.to_array(),
            coverage_outside: cov.outside_nominal,
        });
        if let Some(dir) = &capture_dir {
            // Capture every 15th frame + first/last for the sheet (explicit verification mode).
            if fi == 0 || fi + 1 == n || fi % 15 == 0 {
                let rgba = pass.read_sdr_rgba(&mut be).unwrap();
                assert!(rgba.iter().any(|&b| b > 0), "frame {fi} empty SDR");
                assert!(rgba.len() == (args.width * args.height * 4) as usize);
                write_png(&dir.join(format!("frame_{fi:04}.png")), args.width, args.height, &rgba);
            }
        }
        total_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        if fi == 0 || fi + 1 == n || fi % 30 == 0 {
            eprintln!(
                "frame={fi} selected={} candidates={} culled={} outside={} pos_pc={:.4} select_ms={:.2} submit_ms={:.2} gpu_ms={:.2}",
                stats.selected,
                stats.candidates_considered,
                stats.culled,
                cov.outside_nominal,
                cov.distance_pc,
                stats.select_ms,
                timing.submit_ms,
                timing.gpu_ms
            );
        }
    }
    // Second pass: identical replay must match IDs/positions within tolerance.
    let mut flight2 = FlightState::sol_looking_at(look_target(index));
    let mut sel2 = planet::universe::Selection::new();
    for (i, rec) in records.iter().enumerate() {
        flight2.integrate(&rec.sample);
        let cam = flight2.camera(args.width, args.height).unwrap();
        index.select(&cam, MAX_STARS, &mut sel2);
        assert!((flight2.pos - DVec3::from_array(rec.pos)).length() < 1e-6, "pos diverge @ {i}");
        let ids: Vec<_> = sel2.sources.iter().map(|s| s.id).collect();
        assert_eq!(ids, rec.selected_ids, "selected ids diverge @ {i}");
        assert_eq!(Coverage::at(flight2.pos).outside_nominal, rec.coverage_outside);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let p95 = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.total_cmp(b));
        s[((s.len() as f64) * 0.95) as usize].min(*s.last().unwrap_or(&0.0))
    };
    let meta = format!(
        "{{\n  \"dataset\": \"Gaia EDR3 GCNS / Smart et al. 2021; Sol-local ICRS J2016.0 static medians\",\n  \"raw_rows\": {raw_rows},\n  \"eligible\": {},\n  \"nominal_survey_pc\": {NOMINAL_SURVEY_PC},\n  \"look_source_id\": \"{}\",\n  \"exposure\": {},\n  \"epic_grade\": null,\n  \"device_name\": {:?},\n  \"device_note\": \"label lavapipe vs native hardware from device_name\",\n  \"frames\": {},\n  \"size\": [{}, {}],\n  \"select_ms_mean\": {},\n  \"select_ms_p95\": {},\n  \"submit_ms_mean\": {},\n  \"submit_ms_p95\": {},\n  \"total_ms_mean\": {},\n  \"total_ms_p95\": {},\n  \"hdr_host_readback_per_frame\": false,\n  \"full_catalogue_rescan_per_frame\": false,\n  \"sync_note\": \"live path uses immediate_submit fence (no device_wait_idle per frame); single-queue not mesh multi-FIF\",\n  \"coverage_warning\": \"outside nominal survey volume; coverage incomplete\",\n  \"not_full_milky_way\": true,\n  \"final_pos_m\": {:?},\n  \"final_outside_nominal\": {},\n  \"final_selected_top\": {:?}\n}}\n",
        index.len(),
        default_look_source_id(),
        PRESENT_EXPOSURE,
        device,
        n,
        args.width,
        args.height,
        mean(&select_ms),
        p95(&select_ms),
        mean(&submit_ms),
        p95(&submit_ms),
        mean(&total_ms),
        p95(&total_ms),
        records.last().map(|r| r.pos).unwrap_or([0.0; 3]),
        records.last().map(|r| r.coverage_outside).unwrap_or(false),
        records.last().map(|r| r.selected_ids.iter().take(8).cloned().collect::<Vec<_>>()).unwrap_or_default()
    );
    let out_meta = capture_dir
        .as_ref()
        .map(|d| d.join("GT1_replay.json"))
        .unwrap_or_else(|| PathBuf::from("out/quality/GT1_replay.json"));
    if let Some(parent) = out_meta.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&out_meta, &meta).unwrap();
    eprintln!("{meta}");
    if let Some(path) = &args.record {
        write_replay_json(path, &records);
    }
    // Contact-sheet from captured frames if any.
    if let Some(dir) = &capture_dir {
        make_contact_sheet(dir, args.width, args.height);
    }
    pass.free(&mut be);
    drop(be);
    assert_eq!(render_ash::validation::message_counts().0, 0);
}

fn parse_replay_json(text: &str) -> Vec<InputSample> {
    // Minimal hand parser: array of objects with dt,forward,right,up,yaw_delta,pitch_delta,speed_gear.
    let mut out = Vec::new();
    for chunk in text.split('{').skip(1) {
        let Some(end) = chunk.find('}') else { continue };
        let body = &chunk[..end];
        if !body.contains("dt") {
            continue;
        }
        let num = |k: &str| {
            body.split(',').find_map(|p| {
                let p = p.trim();
                p.strip_prefix(k)?.strip_prefix(':').map(|v| v.trim().trim_matches(',').parse::<f64>().ok())?
            })
        };
        let Some(dt) = num("\"dt\"") else { continue };
        let teleport = if let Some(idx) = body.find("\"teleport\"") {
            let slice = &body[idx..];
            slice.find('[').and_then(|a| {
                let rest = &slice[a + 1..];
                let b = rest.find(']')?;
                let nums: Vec<f64> = rest[..b].split(',').filter_map(|x| x.trim().parse().ok()).collect();
                (nums.len() == 3 && nums.iter().all(|v| v.is_finite())).then(|| DVec3::new(nums[0], nums[1], nums[2]))
            })
        } else {
            None
        };
        out.push(InputSample {
            dt,
            forward: num("\"forward\"").unwrap_or(0.0) as f32,
            right: num("\"right\"").unwrap_or(0.0) as f32,
            up: num("\"up\"").unwrap_or(0.0) as f32,
            yaw_delta: num("\"yaw_delta\"").unwrap_or(0.0),
            pitch_delta: num("\"pitch_delta\"").unwrap_or(0.0),
            speed_gear: num("\"speed_gear\"").unwrap_or(0.0) as i8,
            teleport,
        });
    }
    assert!(!out.is_empty(), "replay JSON contained no samples");
    out
}

fn write_replay_json(path: &Path, records: &[SampleRec]) {
    let mut s = String::from("[\n");
    for (i, r) in records.iter().enumerate() {
        if i > 0 {
            s.push_str(",\n");
        }
        s.push_str(&format!(
            "  {{\"dt\":{},\"forward\":{},\"right\":{},\"up\":{},\"yaw_delta\":{},\"pitch_delta\":{},\"speed_gear\":{},\"pos\":{:?},\"outside\":{},\"ids\":{:?}}}",
            r.sample.dt,
            r.sample.forward,
            r.sample.right,
            r.sample.up,
            r.sample.yaw_delta,
            r.sample.pitch_delta,
            r.sample.speed_gear,
            r.pos,
            r.coverage_outside,
            r.selected_ids.iter().take(32).cloned().collect::<Vec<_>>()
        ));
    }
    s.push_str("\n]\n");
    std::fs::write(path, s).unwrap();
}

fn make_contact_sheet(dir: &Path, fw: u32, fh: u32) {
    let mut frames = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("png")
                && p.file_name().unwrap().to_string_lossy().starts_with("frame_")
        })
        .collect::<Vec<_>>();
    frames.sort();
    if frames.is_empty() {
        return;
    }
    let cols = frames.len().clamp(1, 4);
    let rows = frames.len().div_ceil(cols);
    let mut rgba = vec![0u8; (cols as u32 * fw * rows as u32 * fh * 4) as usize];
    for (i, path) in frames.iter().enumerate() {
        let file = std::io::BufReader::new(std::fs::File::open(path).unwrap());
        let dec = png::Decoder::new(file);
        let mut reader = dec.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        let (c, r) = (i % cols, i / cols);
        for y in 0..fh as usize {
            for x in 0..fw as usize {
                let si = (y * fw as usize + x) * 4;
                if si + 3 >= info.buffer_size() {
                    break;
                }
                let di = (((r * fh as usize + y) * (cols * fw as usize)) + (c * fw as usize + x)) * 4;
                rgba[di..di + 4].copy_from_slice(&buf[si..si + 4]);
            }
        }
    }
    write_png(&dir.join("GT1_contact_sheet.png"), cols as u32 * fw, rows as u32 * fh, &rgba);
    eprintln!("wrote {}", dir.join("GT1_contact_sheet.png").display());
}

#[cfg(feature = "windowed")]
fn run_interactive(args: &Args, index: &StarIndex, _raw_rows: usize) {
    use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
    use std::sync::Arc;
    use winit::application::ApplicationHandler;
    use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::keyboard::{KeyCode, PhysicalKey};
    use winit::window::{CursorGrabMode, Window, WindowId};

    struct App {
        args: Args,
        index: StarIndex,
        window: Option<Arc<Window>>,
        backend: Option<AshBackend>,
        pass: Option<UniversePass>,
        flight: FlightState,
        keys: FlightKeys,
        mouse_grabbed: bool,
        yaw_acc: f64,
        pitch_acc: f64,
        last: Instant,
        focus: bool,
        splats: Vec<Splat>,
        sel: planet::universe::Selection,
        frames: u64,
        quit: bool,
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_some() {
                return;
            }
            let attrs = Window::default_attributes()
                .with_title("GT1 galaxy_travel — GCNS Sol neighbourhood (not full Milky Way)")
                .with_inner_size(winit::dpi::PhysicalSize::new(self.args.width, self.args.height));
            let window = Arc::new(event_loop.create_window(attrs).expect("window"));
            let wh = render_ash::WindowHandles {
                display: window.display_handle().unwrap().as_raw(),
                window: window.window_handle().unwrap().as_raw(),
            };
            let mut be = AshBackend::new_windowed(
                wh,
                BackendConfig {
                    max_instances: 16,
                    extent: (self.args.width, self.args.height),
                    validation: self.args.validation,
                    ..Default::default()
                },
            )
            .expect("windowed backend");
            let pass = UniversePass::new(&mut be, self.args.width, self.args.height).unwrap();
            self.window = Some(window);
            self.backend = Some(be);
            self.pass = Some(pass);
            self.last = Instant::now();
        }
        fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
            match event {
                WindowEvent::CloseRequested => {
                    self.quit = true;
                    event_loop.exit();
                }
                WindowEvent::Focused(f) => {
                    self.focus = f;
                    self.keys.on_focus(f);
                    if !f {
                        self.mouse_grabbed = false;
                        if let Some(w) = &self.window {
                            let _ = w.set_cursor_grab(CursorGrabMode::None);
                            w.set_cursor_visible(true);
                        }
                        eprintln!("focus lost: keys cleared, cursor released");
                    } else {
                        eprintln!("focus gained");
                    }
                }
                WindowEvent::Resized(size) => {
                    if size.width > 0 && size.height > 0 {
                        self.args.width = size.width;
                        self.args.height = size.height;
                        if let (Some(be), Some(pass)) = (self.backend.as_mut(), self.pass.as_mut()) {
                            let _ = lowlevel_abi::RenderBackend::resize(be, size.width, size.height);
                            let _ = pass.resize(be, size.width, size.height);
                        }
                    }
                }
                WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                    if let Some(w) = &self.window {
                        match w
                            .set_cursor_grab(CursorGrabMode::Confined)
                            .or_else(|_| w.set_cursor_grab(CursorGrabMode::Locked))
                        {
                            Ok(()) => {
                                w.set_cursor_visible(false);
                                self.mouse_grabbed = true;
                                eprintln!("cursor grab OK (Confined/Locked); Esc releases");
                            }
                            Err(e) => {
                                self.mouse_grabbed = false;
                                eprintln!("cursor grab FAILED: {e:?} (pointer may stay free)");
                            }
                        }
                    }
                }
                WindowEvent::KeyboardInput { event: KeyEvent { physical_key, state, .. }, .. } => {
                    let down = state == ElementState::Pressed;
                    match physical_key {
                        PhysicalKey::Code(KeyCode::KeyW) => self.keys.w = down,
                        PhysicalKey::Code(KeyCode::KeyS) => self.keys.s = down,
                        PhysicalKey::Code(KeyCode::KeyA) => self.keys.a = down,
                        PhysicalKey::Code(KeyCode::KeyD) => self.keys.d = down,
                        PhysicalKey::Code(KeyCode::KeyR) => self.keys.r = down,
                        PhysicalKey::Code(KeyCode::KeyF) => self.keys.f = down,
                        PhysicalKey::Code(KeyCode::ShiftLeft) | PhysicalKey::Code(KeyCode::ShiftRight) => {
                            self.keys.shift = down
                        }
                        PhysicalKey::Code(KeyCode::ControlLeft) | PhysicalKey::Code(KeyCode::ControlRight) => {
                            self.keys.ctrl = down
                        }
                        PhysicalKey::Code(KeyCode::Escape) if down => {
                            if self.mouse_grabbed {
                                self.mouse_grabbed = false;
                                if let Some(w) = &self.window {
                                    let _ = w.set_cursor_grab(CursorGrabMode::None);
                                    w.set_cursor_visible(true);
                                }
                                eprintln!("Esc: cursor released");
                            } else {
                                self.quit = true;
                                event_loop.exit();
                            }
                        }
                        PhysicalKey::Code(KeyCode::KeyQ) if down => {
                            self.quit = true;
                            event_loop.exit();
                        }
                        _ => {}
                    }
                }
                WindowEvent::RedrawRequested => self.frame(),
                _ => {}
            }
        }
        fn device_event(
            &mut self,
            _event_loop: &ActiveEventLoop,
            _device: winit::event::DeviceId,
            event: winit::event::DeviceEvent,
        ) {
            if let winit::event::DeviceEvent::MouseMotion { delta } = event {
                if self.mouse_grabbed && self.focus {
                    self.yaw_acc += delta.0 * 0.002;
                    self.pitch_acc -= delta.1 * 0.002;
                }
            }
        }
        fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        }
    }
    impl App {
        fn frame(&mut self) {
            if self.quit {
                return;
            }
            let Some(be) = self.backend.as_mut() else { return };
            let Some(pass) = self.pass.as_mut() else { return };
            let now = Instant::now();
            let mut dt = now.duration_since(self.last).as_secs_f64();
            self.last = now;
            if !self.focus {
                dt = 0.0; // pause motion while unfocused; still present last frame
            }
            dt = dt.clamp(0.0, 0.1);
            let gear = self.keys.speed_gear();
            let (forward, right, up) = if self.focus { self.keys.to_axes() } else { (0.0, 0.0, 0.0) };
            let sample = InputSample {
                dt,
                forward,
                right,
                up,
                yaw_delta: if self.focus { std::mem::take(&mut self.yaw_acc) } else { 0.0 },
                pitch_delta: if self.focus { std::mem::take(&mut self.pitch_acc) } else { 0.0 },
                speed_gear: gear,
                teleport: None,
            };
            self.flight.integrate(&sample);
            let cam = match self.flight.camera(self.args.width, self.args.height) {
                Some(c) => c,
                None => return,
            };
            let stats = self.index.select(&cam, MAX_STARS, &mut self.sel);
            self.splats.clear();
            self.splats.extend(self.sel.sources.iter().map(Splat::from_source));
            if pass.render_live(be, &self.splats, PRESENT_EXPOSURE).is_err() {
                return;
            }
            let (w, h) = pass.extent();
            let _ = be.present_transfer_src(pass.sdr_image(), ash::vk::Extent2D { width: w, height: h });
            self.frames += 1;
            let speed = FlightState::speed_m_s(gear) / PARSEC_M;
            let title = status_title(
                self.flight.pos,
                self.flight.yaw,
                self.flight.pitch,
                speed,
                stats.selected,
                self.mouse_grabbed,
                self.focus,
            );
            if let Some(w) = &self.window {
                w.set_title(&title);
            }
            if self.frames % 120 == 1 {
                let cov = Coverage::at(self.flight.pos);
                eprintln!(
                    "t={} selected={} culled={} speed={:.4} pc/s outside={} grab={} | {}",
                    self.frames, stats.selected, stats.culled, speed, cov.outside_nominal, self.mouse_grabbed, title
                );
            }
        }
    }
    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        flight: FlightState::sol_looking_at(look_target(index)),
        args: Args {
            catalogue: args.catalogue.clone(),
            replay: None,
            record: None,
            capture: None,
            frames: 0,
            width: args.width,
            height: args.height,
            headless: false,
            validation: args.validation,
        },
        index: StarIndex::build(index.stars(), 5.0 * PARSEC_M),
        window: None,
        backend: None,
        pass: None,
        keys: FlightKeys::default(),
        mouse_grabbed: false,
        yaw_acc: 0.0,
        pitch_acc: 0.0,
        last: Instant::now(),
        focus: true,
        splats: Vec::new(),
        sel: planet::universe::Selection::new(),
        frames: 0,
        quit: false,
    };
    event_loop.run_app(&mut app).unwrap();
    if let (Some(mut be), Some(pass)) = (app.backend.take(), app.pass.take()) {
        pass.free(&mut be);
    }
}

fn main() {
    let args = parse_args();
    let (index, raw_rows) = load_catalogue(&args.catalogue);
    eprintln!(
        "GT1 galaxy_travel: {} eligible / {raw_rows} raw; look_source_id={}; nominal {} pc; exposure={}",
        index.len(),
        default_look_source_id(),
        NOMINAL_SURVEY_PC,
        PRESENT_EXPOSURE
    );
    if args.headless || args.replay.is_some() || args.frames > 0 || args.capture.is_some() {
        run_replay(&args, &index, raw_rows);
    } else {
        #[cfg(feature = "windowed")]
        run_interactive(&args, &index, raw_rows);
        #[cfg(not(feature = "windowed"))]
        {
            eprintln!("rebuild with --features windowed for interactive; use --replay/--frames for headless");
            std::process::exit(2);
        }
    }
}
