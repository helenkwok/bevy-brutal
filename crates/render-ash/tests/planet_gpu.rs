//! Planet renderer on the present Vulkan device (lavapipe headless) with a tiny synthetic pack.
//! Regression guards for Earth round 1 bugs:
//! - patch origins: every patch must use its own camera-relative origin (the VS once read the
//!   instance id, always 0 with one indirect command per patch -> a shattered globe);
//! - NaN-free shading at exact nadir (pow of a negative base in the Fresnel term);
//! - night radiances ~1e-6 must survive the RGBA16F target (pre-exposure), with the city colour
//!   ratios intact (they used to quantise into f16 subnormals -> banding and a green cast).
mod common;

use glam::{DVec3, Mat4};
use planet::bake::bake_cube;
use planet::pack::{HeightCube, PlanetPack, TexFormat};
use planet::patch::{self, VERTS};
use planet::quadtree::{self, Shape, View};
use render_ash::planet::{PatchGpu, PlanetFrame, PlanetParamsGpu, PlanetTextures};
use render_ash::validation::message_counts;
use render_ash::{AshBackend, BackendConfig};

const R: f64 = planet::EARTH_RADIUS_M;

fn pack() -> PlanetPack {
    pack_with_clouds(0.0)
}

/// `cloud`: the aux map's r = the cloud map's uniform coverage.
fn pack_with_clouds(cloud: f32) -> PlanetPack {
    let albedo = bake_cube("albedo", TexFormat::Bc7Srgb, 32, 1, &|d: DVec3| {
        [0.25, 0.25, 0.25, if d.z < 0.0 { 1.0 } else { 0.0 }]
    });
    let slope = bake_cube("slope", TexFormat::Bc5, 32, 1, &|_| [0.5, 0.5, 0.0, 1.0]);
    let night = bake_cube("night", TexFormat::Bc4, 32, 1, &|_| [0.5, 0.0, 0.0, 1.0]);
    let aux = bake_cube("aux", TexFormat::Bc7, 32, 1, &|_| [cloud, 0.0, 0.5, 1.0]);
    PlanetPack {
        textures: vec![albedo, slope, night, aux],
        height: HeightCube { size: 8, texels: vec![0; 6 * 64], scale: 0.1, offset: 0.0 },
        credits: "test".into(),
    }
}

/// Camera at `alt_km` above (0, 0) looking at the planet centre; sun along `sun`. 96x96, fov 40.
fn render(
    be: &mut AshBackend,
    tex: &PlanetTextures,
    pk: &PlanetPack,
    alt_km: f64,
    sun: DVec3,
    exposure: f32,
    night: f32,
) -> Vec<[f32; 4]> {
    render_with(be, tex, pk, alt_km, sun, exposure, night, &|_| {})
}

#[allow(clippy::too_many_arguments)]
fn render_with(
    be: &mut AshBackend,
    tex: &PlanetTextures,
    pk: &PlanetPack,
    alt_km: f64,
    sun: DVec3,
    exposure: f32,
    night: f32,
    edit: &dyn Fn(&mut PlanetParamsGpu),
) -> Vec<[f32; 4]> {
    let n = 96u32;
    let fov = 40f64.to_radians();
    let cam = DVec3::X * (R + alt_km * 1000.0);
    let (fwd, up) = (-DVec3::X, DVec3::Z);
    let right = fwd.cross(up);
    let t = (fov / 2.0).tan();
    let pl = |v: DVec3, w: f64| {
        let l = v.length();
        glam::DVec4::new(v.x / l, v.y / l, v.z / l, w / l)
    };
    let planes = [
        pl(right + fwd * t, 0.0),
        pl(-right + fwd * t, 0.0),
        pl(up + fwd * t, 0.0),
        pl(-up + fwd * t, 0.0),
        pl(fwd, -1.0),
        pl(-fwd, 1e12),
    ];
    let view = View { cam, planes, px_per_rad: n as f64 / (2.0 * t), max_error_px: 1.0, max_level: 10 };
    let sel = quadtree::select(&view, &Shape { radius: R, h_min: 0.0, h_max: 0.0 });
    assert!(sel.visible.len() > 1, "need several patches to exercise per-patch origins");
    let mut vertices = Vec::new();
    let mut patches = Vec::new();
    for (k, _) in &sel.visible {
        let m = patch::build(k, R, &|d| pk.height.sample(d));
        assert_eq!(m.verts.len(), VERTS as usize);
        vertices.extend_from_slice(&m.verts);
        let o = (m.origin - cam).as_vec3();
        patches.push(PatchGpu { origin: [o.x, o.y, o.z, k.level as f32] });
    }
    let masks: Vec<u8> = sel.visible.iter().map(|x| x.1).collect();
    let (f, r, u) = (fwd.as_vec3(), right.as_vec3(), up.as_vec3());
    let view_m = Mat4::from_cols(r.extend(0.0), u.extend(0.0), (-f).extend(0.0), glam::Vec4::W).transpose();
    let vp = render_ash::quality::reverse_z_infinite(fov as f32, 1.0, 1000.0) * view_m;
    let rk = (R / 1000.0) as f32;
    let s = sun.normalize().as_vec3();
    let c = cam / 1000.0;
    let mut params = PlanetParamsGpu {
        view_proj: vp.to_cols_array_2d(),
        cam_km: [c.x as f32, c.y as f32, c.z as f32, alt_km as f32],
        sun: [s.x, s.y, s.z, 0.004_65],
        right: [r.x, r.y, r.z, t as f32],
        up: [u.x, u.y, u.z, t as f32],
        fwd: [f.x, f.y, f.z, 0.0],
        planet: [rk, rk + 100.0, rk + 6.0, 1.6],
        ocean: [0.003 + 0.00512 * 6.0, 0.02, 0.22, 1.0],
        light: [night, 0.0, 1.0, 1.0],
        atmo: [10.0, 1.0, 1.0, 0.15],
        view: [n as f32, n as f32, exposure, 0.0],
        cloud: [1.0, 9.0, 24.0, 0.0],
        cloud2: [48.0, 6.0, 18.0, 1.0],
        aero: [0.0; 4],
        hole: [0.0; 4],
        meso: [0.0; 4],
        thaw: [0.0; 4],
        graft: [0.0; 4],
        graft_e: [0.0; 4],
        graft_n: [0.0; 4],
        graft_l: [0.0; 4],
        meso2: [0.0; 4],
        graft_t: [0.0; 4],
        haze: [0.0; 4],
        cloud3: [0.0; 4],
    };
    edit(&mut params);
    let frame = PlanetFrame { width: n, height: n, params, patches: &patches, masks: &masks, vertices: &vertices };
    let (px, timings) = be.render_planet(tex, &frame).expect("render");
    assert_eq!(timings.patches as usize, sel.visible.len());
    px
}

#[test]
fn planet_renders_round_nan_free_and_resolves_night_radiance() {
    let mut be = match AshBackend::new(
        BackendConfig { max_instances: 16, extent: (64, 64), validation: true, ..Default::default() },
        None,
    ) {
        Ok(b) => {
            common::check_validation(&b.ctx);
            b
        }
        Err(e) => {
            common::skip_or_fail("AshBackend::new", &e);
            return;
        }
    };
    let pk = pack();
    let tex = be.planet_upload(&pk).expect("upload");
    let (e0, _) = message_counts();
    let n = 96usize;

    // day side, sun behind the camera (exact nadir at the image centre)
    let day = render(&mut be, &tex, &pk, 20_000.0, DVec3::new(1.0, 0.0, 0.05), 3.0, 0.0);
    assert!(day.iter().all(|p| p.iter().all(|v| v.is_finite())), "non-finite output");
    // disk (alpha 0 = planet) vs the expected angular radius asin(R / d)
    let disk: Vec<(usize, usize)> = (0..n * n).filter(|&i| day[i][3] == 0.0).map(|i| (i % n, i / n)).collect();
    let d = R + 20_000_000.0;
    let r_px = (R / d).asin().tan() / 20f64.to_radians().tan() * n as f64 / 2.0;
    let area = disk.len() as f64;
    let expect = std::f64::consts::PI * r_px * r_px;
    assert!((area / expect - 1.0).abs() < 0.04, "disk area {area} vs {expect:.0} px");
    let c = n as f64 / 2.0;
    let quads = [0usize, 1, 2, 3].map(|q| {
        disk.iter()
            .filter(|(x, y)| ((*x as f64 + 0.5 > c) as usize) + 2 * ((*y as f64 + 0.5 > c) as usize) == q)
            .count() as f64
    });
    for q in quads {
        assert!((q / (area / 4.0) - 1.0).abs() < 0.03, "disk quadrants unbalanced: {quads:?}");
    }
    let centre = day[(n / 2) * n + n / 2];
    assert!(centre[0] > 0.01 && centre[2] > centre[0], "lit nadir must be bright and bluish: {centre:?}");

    // night side: sun behind the planet, city map 0.5 everywhere, physical gain, pre-exposed
    let exposure = 2.0e5f32;
    let gain = 1.5e-6f32;
    let night = render(&mut be, &tex, &pk, 20_000.0, DVec3::new(-1.0, 0.0, 0.0), exposure, gain);
    let p = night[(n / 2) * n + n / 2];
    let city = gain * 0.5f32.powf(2.2);
    let got = [p[0] / exposure, p[1] / exposure, p[2] / exposure];
    assert!(got[0] > 0.7 * city && got[0] < 1.1 * city, "night radiance {got:?} vs city {city:.3e}");
    let (rg, rb) = (got[1] / got[0], got[2] / got[0]);
    assert!(
        (rg / 0.72 - 1.0).abs() < 0.1 && (rb / 0.42 - 1.0).abs() < 0.25,
        "city colour ratios lost: g/r {rg:.3} b/r {rb:.3}"
    );

    be.planet_free(tex);
    let (e1, _) = message_counts();
    assert_eq!(e1 - e0, 0, "validation errors during planet rendering");
}

/// Earth round 3 cloud regressions on a uniform 90 % deck (volumetric mode):
/// - the clear-sky window (`hole`) must clear every cloud path at its centre (the first version
///   only cleared the volume's columns; the ground's cloud shadow still read the raw map), so the
///   nadir matches a cloud-free planet there;
/// - from 1500 km (pixel footprints ~11 km, EPIC full-disk scale: the 256 .. 64 km octaves are
///   resolved; at ~150 km footprints every octave is faded out by design) the far-footprint
///   structure (`meso`) must break the uniform deck into texture (it rendered as one flat sheet),
///   while all-zero `meso` stays bit-identical to round 2.
#[test]
fn cloud_window_clears_and_far_deck_gets_structure() {
    let mut be = match AshBackend::new(
        BackendConfig { max_instances: 16, extent: (64, 64), validation: true, ..Default::default() },
        None,
    ) {
        Ok(b) => {
            common::check_validation(&b.ctx);
            b
        }
        Err(e) => {
            common::skip_or_fail("AshBackend::new", &e);
            return;
        }
    };
    let (e0, _) = message_counts();
    let n = 96usize;
    let sun = DVec3::new(1.0, 0.0, 0.05);
    let clear_pk = pack();
    let clear_tex = be.planet_upload(&clear_pk).expect("upload");
    let pk = pack_with_clouds(0.9);
    let tex = be.planet_upload(&pk).expect("upload");
    // volumetric clouds with the earth example's noise period and multiple-scattering gain
    let vol = |p: &mut PlanetParamsGpu| {
        p.cloud[3] = 1.0;
        p.cloud2 = [48.0, 6.0, 10.0, 2.5];
    };
    let lum = |c: [f32; 4]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let mid = (n / 2) * n + n / 2;

    let clear = render_with(&mut be, &clear_tex, &clear_pk, 20_000.0, sun, 3.0, 0.0, &vol);
    let deck = render_with(&mut be, &tex, &pk, 20_000.0, sun, 3.0, 0.0, &vol);
    assert!(lum(deck[mid]) > 1.5 * lum(clear[mid]), "the deck must be bright: {:?} vs {:?}", deck[mid], clear[mid]);
    // window of 10 deg around the nadir (+X)
    let hole = render_with(&mut be, &tex, &pk, 20_000.0, sun, 3.0, 0.0, &|p| {
        vol(p);
        p.hole = [1.0, 0.0, 0.0, 10f32.to_radians()];
    });
    let rel = (lum(hole[mid]) - lum(clear[mid])).abs() / lum(clear[mid]);
    assert!(rel < 0.02, "window centre must be cloud-free: {:?} vs clear {:?}", hole[mid], clear[mid]);

    // structure: luminance spread over the whole view (all planet from 1500 km), meso off vs on
    let spread = |px: &[[f32; 4]]| {
        let v: Vec<f32> = px.iter().map(|&c| lum(c)).collect();
        let m = v.iter().sum::<f32>() / v.len() as f32;
        ((v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32).sqrt() / m, m)
    };
    let deck = render_with(&mut be, &tex, &pk, 1500.0, sun, 3.0, 0.0, &vol);
    let zero = render_with(&mut be, &tex, &pk, 1500.0, sun, 3.0, 0.0, &|p| {
        vol(p);
        p.meso = [0.0; 4];
    });
    assert!(zero.iter().zip(&deck).all(|(a, b)| a == b), "meso = 0 must be round 2 exactly");
    let meso = render_with(&mut be, &tex, &pk, 1500.0, sun, 3.0, 0.0, &|p| {
        vol(p);
        p.meso = [2.0, 4.5, -0.7, 0.0];
    });
    assert!(meso.iter().all(|p| p.iter().all(|v| v.is_finite())), "non-finite output");
    let (s0, m0) = spread(&deck);
    let (s1, m1) = spread(&meso);
    eprintln!("deck spread {s0:.4} (mean {m0:.4}) -> with meso {s1:.4} (mean {m1:.4})");
    assert!(s1 > 2.0 * s0 + 0.02, "far deck must gain structure: relative spread {s0:.4} -> {s1:.4}");
    assert!((m1 / m0 - 1.0).abs() < 0.35, "and keep its mean brightness: {m0} -> {m1}");
    be.planet_free(tex);
    be.planet_free(clear_tex);
    drop(be); // teardown errors (leaked objects at vkDestroyDevice) count too
    let (e1, _) = message_counts();
    assert_eq!(e1, e0, "validation errors");
}

/// The close layer UV origin moves independently of the fixed DEM/far graft origin.
#[test]
fn close_graft_offset_changes_only_near_coverage() {
    let mut be = match AshBackend::new(
        BackendConfig { max_instances: 16, extent: (96, 96), validation: true, ..Default::default() },
        None,
    ) {
        Ok(b) => {
            common::check_validation(&b.ctx);
            b
        }
        Err(e) => {
            common::skip_or_fail("AshBackend::new", &e);
            return;
        }
    };
    let pk = pack();
    let mut tex = be.planet_upload(&pk).unwrap();
    let layer = |c: [u8; 4]| render_ash::planet::rgba8_mips(8, &c.repeat(64), true);
    let alb = vec![layer([120, 120, 120, 255]), layer([120, 120, 120, 255]), layer([240, 30, 30, 255])];
    let aux = vec![render_ash::planet::rgba8_mips(8, &[128u8, 128, 255, 255].repeat(64), false); 3];
    be.planet_set_graft(&mut tex, 8, &alb, &aux).unwrap();
    let edit = |p: &mut PlanetParamsGpu, offset: f32, half: f32| {
        p.graft = [1.0, 0.0, 0.0, 0.05];
        p.graft_e = [0.0, 1.0, 0.0, 0.1];
        p.graft_n = [0.0, 0.0, 1.0, 500.0];
        p.graft_l = [250.0, 1.0, 0.0, 8.0];
        p.meso2 = [0.0, half, offset, 0.0];
    };
    let near = render_with(&mut be, &tex, &pk, 2.0, DVec3::X, 1.0, 0.0, &|p| edit(p, 0.0, 5.0));
    let away = render_with(&mut be, &tex, &pk, 2.0, DVec3::X, 1.0, 0.0, &|p| edit(p, 100.0, 5.0));
    let far = render_with(&mut be, &tex, &pk, 2.0, DVec3::X, 1.0, 0.0, &|p| edit(p, 0.0, 0.0));
    let mean = |px: &[[f32; 4]]| px.iter().map(|q| q[0] as f64 - q[1] as f64).sum::<f64>() / px.len() as f64;
    assert!(mean(&near) > mean(&away) + 0.01, "close tile did not move: {} {}", mean(&near), mean(&away));
    assert_eq!(away, far, "outside close tile must preserve far/mid coverage");
    be.planet_free(tex);
    drop(be);
    assert_eq!(message_counts().0, 0, "validation errors including resource retirement");
}
