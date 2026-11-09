//! GT1 travel: spatial index exactness, coverage, flight/replay determinism.
use glam::DVec3;
use planet::gcns::PARSEC_M;
use planet::travel::{
    Coverage, FlightState, InputSample, NOMINAL_SURVEY_PC, StarIndex, approach_stop_m, default_look_source_id,
};
use planet::universe::{Camera, MAX_STARS, Selection, Star};

fn star(id: u64, pos: DVec3, lum: f64) -> Star {
    Star { id, pos, lum, rgb: [1.0; 3] }
}

fn cam_at(pos: DVec3, target: DVec3, w: u32, h: u32) -> Camera {
    let up =
        if (target - pos).normalize_or_zero().cross(DVec3::Z).length_squared() < 1e-12 { DVec3::Y } else { DVec3::Z };
    let mut c = Camera::look_at(pos, target, up, 60f64.to_radians(), w, h)
        .unwrap_or_else(|| panic!("look_at failed pos={pos:?} target={target:?}"));
    c.origin = DVec3::ZERO;
    c
}

#[test]
fn spatial_matches_bruteforce_caps_and_cameras() {
    let mut stars = Vec::new();
    for i in 0..2000u64 {
        let a = i as f64 * 0.017;
        let r = (0.2 + (i % 97) as f64 * 0.4) * PARSEC_M;
        stars.push(star(
            i + 1,
            DVec3::new(r * a.cos(), r * a.sin(), ((i % 11) as f64 - 5.0) * 0.05 * PARSEC_M),
            1.0 + (i % 7) as f64,
        ));
    }
    // Pathological cell-edge star.
    stars.push(star(9_007_199_254_740_993, DVec3::new(5.0 * PARSEC_M, 0.0, 0.0), 50.0));
    let index = StarIndex::build(&stars, 2.0 * PARSEC_M);
    assert_eq!(index.len(), stars.len());

    let cameras = [
        cam_at(DVec3::ZERO, DVec3::X, 320, 180),
        cam_at(DVec3::Y * 3.0 * PARSEC_M, DVec3::Y * 3.0 * PARSEC_M + DVec3::X, 320, 180),
        cam_at(DVec3::X * 120.0 * PARSEC_M, DVec3::X * 119.0 * PARSEC_M, 128, 128),
        cam_at(DVec3::new(5.0, 0.001, 0.0) * PARSEC_M, DVec3::new(4.0, 0.001, 0.0) * PARSEC_M, 256, 256),
    ];
    for cam in cameras {
        for cap in [0usize, 1, 17, 256, MAX_STARS] {
            let mut brute = Selection::new();
            brute.update(&stars, &cam, cap);
            let mut spatial = Selection::new();
            let stats = index.select(&cam, cap, &mut spatial);
            assert_eq!(spatial.sources.len(), brute.sources.len(), "cap={cap}");
            assert_eq!(
                spatial.sources.iter().map(|s| s.id).collect::<Vec<_>>(),
                brute.sources.iter().map(|s| s.id).collect::<Vec<_>>(),
                "cap={cap} ids"
            );
            for (a, b) in spatial.sources.iter().zip(&brute.sources) {
                assert!((a.flux / b.flux - 1.0).abs() < 1e-12);
                assert!((a.center[0] - b.center[0]).abs() < 1e-9);
                assert!((a.center[1] - b.center[1]).abs() < 1e-9);
            }
            assert!(stats.candidates_considered >= spatial.candidates || cap == 0);
            assert!(stats.culled + stats.candidates_considered <= stars.len() + stats.culled); // sanity
        }
    }
}

#[test]
fn full_catalogue_eligibility_not_truncated_to_gpu_cap() {
    let path = "/workspace/data/astro/gcns/gcns-v1.bin";
    let c = planet::gcns::read(&mut std::io::BufReader::new(std::fs::File::open(path).unwrap())).unwrap();
    assert!(c.stars.len() > MAX_STARS);
    let index = StarIndex::build(&c.stars, 5.0 * PARSEC_M);
    assert_eq!(index.len(), c.stars.len());
    let cam = cam_at(DVec3::ZERO, DVec3::X * PARSEC_M, 128, 128);
    let mut brute = Selection::new();
    brute.update(&c.stars, &cam, 64);
    let mut spatial = Selection::new();
    let stats = index.select(&cam, 64, &mut spatial);
    assert_eq!(
        spatial.sources.iter().map(|s| s.id).collect::<Vec<_>>(),
        brute.sources.iter().map(|s| s.id).collect::<Vec<_>>()
    );
    assert!(stats.input_records == c.stars.len());
    // Beyond nominal volume retained.
    let beyond = c.stars.iter().filter(|s| s.pos.length() > NOMINAL_SURVEY_PC * PARSEC_M).count();
    assert!(beyond > 30_000);
}

#[test]
fn coverage_warning_and_look_source() {
    assert!(!Coverage::at(DVec3::ZERO).outside_nominal);
    assert!(Coverage::at(DVec3::X * 150.0 * PARSEC_M).outside_nominal);
    assert!(Coverage::at(DVec3::X * 150.0 * PARSEC_M).warning.contains("outside nominal survey volume"));
    let path = "/workspace/data/astro/gcns/gcns-v1.bin";
    let c = planet::gcns::read(&mut std::io::BufReader::new(std::fs::File::open(path).unwrap())).unwrap();
    let id = default_look_source_id();
    assert!(c.stars.iter().any(|s| s.id == id));
}

#[test]
fn flight_replay_deterministic_and_safe_near_approach() {
    let mut a = FlightState::sol_looking_at(DVec3::X * PARSEC_M);
    let mut b = FlightState::sol_looking_at(DVec3::X * PARSEC_M);
    let samples = [
        InputSample {
            dt: 1.0 / 60.0,
            forward: 1.0,
            right: 0.0,
            up: 0.0,
            yaw_delta: 0.01,
            pitch_delta: -0.005,
            speed_gear: 0,
            teleport: None,
        },
        InputSample {
            dt: 1.0 / 60.0,
            forward: 0.0,
            right: -1.0,
            up: 0.5,
            yaw_delta: 0.0,
            pitch_delta: 0.0,
            speed_gear: 1,
            teleport: None,
        },
        InputSample {
            dt: 0.0,
            forward: 1.0,
            right: 1.0,
            up: 1.0,
            yaw_delta: 1.0,
            pitch_delta: 1.0,
            speed_gear: 0,
            teleport: None,
        }, // invalid dt ignored
        InputSample {
            dt: 1.0 / 60.0,
            forward: 0.0,
            right: 0.0,
            up: 0.0,
            yaw_delta: 0.0,
            pitch_delta: 0.0,
            speed_gear: 0,
            teleport: Some(DVec3::X * 2.0 * PARSEC_M),
        },
        InputSample {
            dt: 1.0 / 30.0,
            forward: 1.0,
            right: 0.0,
            up: 0.0,
            yaw_delta: -0.02,
            pitch_delta: 0.01,
            speed_gear: -1,
            teleport: None,
        },
    ];
    for s in &samples {
        a.integrate(s);
        b.integrate(s);
    }
    assert!((a.pos - b.pos).length() < 1e-9);
    assert!((a.yaw - b.yaw).abs() < 1e-12);
    assert!((a.pitch - b.pitch).abs() < 1e-12);
    let cam = a.camera(320, 180).unwrap();
    assert!(cam.pos.is_finite() && cam.forward.is_finite());

    // Near-zero approach clamp.
    let target = DVec3::X * PARSEC_M;
    let mut f = FlightState::sol_looking_at(target);
    f.pos = target - DVec3::X * (approach_stop_m() * 0.5);
    let before = f.pos;
    f.integrate(&InputSample {
        dt: 1.0,
        forward: 1.0,
        right: 0.0,
        up: 0.0,
        yaw_delta: 0.0,
        pitch_delta: 0.0,
        speed_gear: 5,
        teleport: None,
    });
    // Without an approach target set, free flight continues; with set_approach_limit it stops.
    f.set_approach_limit(Some(target));
    f.pos = before;
    f.integrate(&InputSample {
        dt: 1.0,
        forward: 1.0,
        right: 0.0,
        up: 0.0,
        yaw_delta: 0.0,
        pitch_delta: 0.0,
        speed_gear: 5,
        teleport: None,
    });
    assert!(f.pos.distance(target) >= approach_stop_m() * 0.99);
}

#[test]
fn invalid_speeds_and_nan_inputs_rejected() {
    let mut f = FlightState::sol_looking_at(DVec3::X);
    let bad = InputSample {
        dt: f64::NAN,
        forward: f32::INFINITY,
        right: f32::NAN,
        up: 0.0,
        yaw_delta: f64::NAN,
        pitch_delta: 0.0,
        speed_gear: 0,
        teleport: Some(DVec3::NAN),
    };
    let pos = f.pos;
    f.integrate(&bad);
    assert_eq!(f.pos, pos);
}

#[test]
fn full_gcns_spatial_agrees_sol_moving_outside_caps() {
    let path = "/workspace/data/astro/gcns/gcns-v1.bin";
    let c = planet::gcns::read(&mut std::io::BufReader::new(std::fs::File::open(path).unwrap())).unwrap();
    let index = StarIndex::build(&c.stars, 5.0 * PARSEC_M);
    let cameras = [
        ("sol", cam_at(DVec3::ZERO, DVec3::X * PARSEC_M, 160, 90)),
        ("local", cam_at(DVec3::new(2.0, -1.0, 0.5) * PARSEC_M, DVec3::new(3.0, -1.0, 0.5) * PARSEC_M, 160, 90)),
        ("outside", cam_at(DVec3::X * 150.0 * PARSEC_M, DVec3::X * 149.0 * PARSEC_M, 128, 72)),
        ("cell_edge", cam_at(DVec3::new(5.0, 0.001, 0.0) * PARSEC_M, DVec3::new(4.0, 0.001, 0.0) * PARSEC_M, 128, 128)),
    ];
    for (name, cam) in cameras {
        for cap in [0usize, 1, MAX_STARS] {
            let mut brute = Selection::new();
            brute.update(&c.stars, &cam, cap);
            let mut spatial = Selection::new();
            let stats = index.select(&cam, cap, &mut spatial);
            assert_eq!(stats.input_records, c.stars.len(), "{name} cap={cap}");
            assert_eq!(spatial.sources.len(), brute.sources.len(), "{name} cap={cap} len");
            let ids_s: Vec<_> = spatial.sources.iter().map(|s| s.id).collect();
            let ids_b: Vec<_> = brute.sources.iter().map(|s| s.id).collect();
            assert_eq!(ids_s, ids_b, "{name} cap={cap} ids");
            for (a, b) in spatial.sources.iter().zip(&brute.sources) {
                assert!((a.flux / b.flux - 1.0).abs() < 1e-9, "{name} flux");
            }
            if name == "outside" {
                assert!(Coverage::at(cam.pos).outside_nominal);
                // Must not go empty solely due to being outside nominal 100pc.
                if cap > 0 {
                    assert!(!spatial.sources.is_empty(), "outside retained measured stars");
                }
            }
        }
    }
}

#[test]
fn replay_all_selected_ids_match_across_full_path() {
    let path = "/workspace/data/astro/gcns/gcns-v1.bin";
    let c = planet::gcns::read(&mut std::io::BufReader::new(std::fs::File::open(path).unwrap())).unwrap();
    let index = StarIndex::build(&c.stars, 5.0 * PARSEC_M);
    let target = c.stars.iter().find(|s| s.id == default_look_source_id()).unwrap().pos;
    let mut flight = FlightState::sol_looking_at(target);
    let mut sel = Selection::new();
    let mut recorded: Vec<(DVec3, f64, f64, Vec<u64>)> = Vec::new();
    let samples: Vec<InputSample> = (0..45)
        .map(|i| InputSample {
            dt: 1.0 / 60.0,
            forward: if i < 30 { 1.0 } else { 0.0 },
            right: if (10..20).contains(&i) { 0.4 } else { 0.0 },
            up: 0.0,
            yaw_delta: if (5..15).contains(&i) { 0.012 } else { 0.0 },
            pitch_delta: if (20..28).contains(&i) { -0.01 } else { 0.0 },
            speed_gear: if i < 15 { 2 } else { 0 },
            teleport: if i == 35 { Some(DVec3::X * 120.0 * PARSEC_M) } else { None },
        })
        .collect();
    for s in &samples {
        flight.integrate(s);
        let cam = flight.camera(160, 90).unwrap();
        index.select(&cam, MAX_STARS, &mut sel);
        recorded.push((flight.pos, flight.yaw, flight.pitch, sel.sources.iter().map(|x| x.id).collect()));
    }
    // Replay identical inputs.
    let mut flight2 = FlightState::sol_looking_at(target);
    let mut sel2 = Selection::new();
    for (i, s) in samples.iter().enumerate() {
        flight2.integrate(s);
        let cam = flight2.camera(160, 90).unwrap();
        index.select(&cam, MAX_STARS, &mut sel2);
        assert!((flight2.pos - recorded[i].0).length() < 1e-6, "pos @{i}");
        assert!((flight2.yaw - recorded[i].1).abs() < 1e-12);
        assert!((flight2.pitch - recorded[i].2).abs() < 1e-12);
        let ids: Vec<_> = sel2.sources.iter().map(|x| x.id).collect();
        assert_eq!(ids, recorded[i].3, "all selected ids @{i}");
    }
    assert!(Coverage::at(recorded.last().unwrap().0).outside_nominal);
}

#[test]
fn focus_loss_clears_sticky_keys() {
    use planet::travel::FlightKeys;
    let mut k = FlightKeys { w: true, shift: true, a: true, ..FlightKeys::default() };
    k.on_focus(false);
    assert_eq!(k, FlightKeys::default());
    k.w = true;
    k.on_focus(true);
    assert!(k.w);
    let (f, r, u) = FlightKeys { w: true, d: true, r: true, ..FlightKeys::default() }.to_axes();
    assert_eq!((f, r, u), (1.0, 1.0, 1.0));
    assert_eq!(FlightKeys { shift: true, ..FlightKeys::default() }.speed_gear(), 1);
    assert_eq!(FlightKeys { ctrl: true, ..FlightKeys::default() }.speed_gear(), -1);
}

#[test]
fn status_title_includes_coverage_and_pose() {
    use planet::travel::status_title;
    let t = status_title(DVec3::X * 150.0 * PARSEC_M, 0.5, -0.25, 0.02, 12, true, true);
    assert!(t.contains("OUTSIDE"));
    assert!(t.contains("xyz_pc="));
    assert!(t.contains("grab"));
    let t2 = status_title(DVec3::ZERO, 0.0, 0.0, 0.01, 1, false, false);
    assert!(!t2.contains("OUTSIDE"));
    assert!(t2.contains("free/unfocus"));
}
