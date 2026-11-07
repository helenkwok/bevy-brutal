mod common;
use planet::universe::Source;
use render_ash::{
    AshBackend, BackendConfig,
    universe::{Splat, UniversePass},
};
#[test]
fn hdr_energy_capacity_resize_and_retirement() {
    let mut be = match AshBackend::new(
        BackendConfig { max_instances: 16, extent: (96, 96), validation: true, ..Default::default() },
        None,
    ) {
        Ok(b) => b,
        Err(e) => {
            common::skip_or_fail("AshBackend::new", &e);
            return;
        }
    };
    common::check_validation(&be.ctx);
    let mut pass = UniversePass::new(&mut be, 128, 128).unwrap();
    for (w, h, center) in
        [(128, 128, [64.0, 64.0]), (128, 128, [64.3, 64.7]), (256, 192, [64.25, 64.75]), (96, 96, [0.0, 48.0])]
    {
        pass.resize(&mut be, w, h).unwrap();
        let s = Source { id: 0, center, flux: 1.0, rgb: [1.0, 0.5, 0.25] };
        let (px, _) = pass.render(&mut be, &[Splat::from_source(&s)]).unwrap();
        assert!(px.iter().flatten().all(|x| x.is_finite()));
        let sum: f64 = px.iter().map(|p| p[0] as f64).sum();
        assert!((sum / s.energy(w, h) - 1.0).abs() < 0.002, "energy {sum}");
    }
    let source = Source { id: 0, center: [48.0, 48.0], flux: 1e-6, rgb: [1.0; 3] };
    let full = vec![Splat::from_source(&source); 65536];
    let (px, _) = pass.render(&mut be, &full).unwrap();
    let sum: f64 = px.iter().map(|p| p[0] as f64).sum();
    assert!((sum / 0.065536 - 1.0).abs() < 0.005, "full-capacity additive energy {sum}");
    assert!(pass.resize(&mut be, 0, 128).is_err());
    let mut invalid = Splat::from_source(&source);
    invalid.position[0] = f32::NAN;
    assert!(pass.render(&mut be, &[invalid]).is_err());
    invalid = Splat::from_source(&source);
    invalid.flux[0] = 1e35;
    assert!(pass.render(&mut be, &[invalid]).is_err());
    let bad = vec![Splat::default(); 65537];
    assert!(pass.render(&mut be, &bad).is_err());
    let (px, _) = pass.render(&mut be, &[]).unwrap();
    assert!(px.iter().all(|p| p[..3] == [0.0; 3]));
    pass.free(&mut be);
    drop(be);
    assert_eq!(render_ash::validation::message_counts().0, 0);
}
