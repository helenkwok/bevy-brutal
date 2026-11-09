mod common;
use planet::universe::Source;
use render_ash::{
    AshBackend, BackendConfig,
    universe::{PRESENT_EXPOSURE, Splat, UniversePass},
};
#[test]
fn live_tonemap_continuous_frames_resize_no_hdr_readback() {
    let mut be = match AshBackend::new(
        BackendConfig { max_instances: 16, extent: (160, 90), validation: true, ..Default::default() },
        None,
    ) {
        Ok(b) => b,
        Err(e) => {
            common::skip_or_fail("AshBackend::new", &e);
            return;
        }
    };
    common::check_validation(&be.ctx);
    let mut pass = UniversePass::new(&mut be, 160, 90).unwrap();
    let s = Source { id: 1, center: [80.0, 45.0], flux: 1.0, rgb: [1.0, 0.8, 0.6] };
    let splat = Splat::from_source(&s);
    let mut nonempty = 0u32;
    let mut w = 160u32;
    let mut h = 90u32;
    for i in 0..8u32 {
        if i == 3 {
            w = 128;
            h = 72;
            pass.resize(&mut be, w, h).unwrap();
        }
        let t = pass.render_live(&mut be, &[splat], PRESENT_EXPOSURE).unwrap();
        assert!(t.submit_ms.is_finite());
        let rgba = pass.read_sdr_rgba(&mut be).unwrap();
        assert_eq!(rgba.len(), (w * h * 4) as usize);
        if rgba.iter().any(|&b| b > 0) {
            nonempty += 1;
        }
    }
    assert!(nonempty >= 3, "expected lit SDR frames, got {nonempty}");
    let (hdr, _) = pass.render(&mut be, &[splat]).unwrap();
    assert!(hdr.iter().flatten().all(|x| x.is_finite()));
    pass.free(&mut be);
    drop(be);
    assert_eq!(render_ash::validation::message_counts().0, 0);
}

#[test]
fn live_after_transfer_src_consumer_has_no_sync_hazard() {
    let mut be = match AshBackend::new(
        BackendConfig { max_instances: 16, extent: (64, 64), validation: true, ..Default::default() },
        None,
    ) {
        Ok(b) => b,
        Err(e) => {
            common::skip_or_fail("AshBackend::new", &e);
            return;
        }
    };
    common::check_validation(&be.ctx);
    if !be.ctx.validation_enabled {
        eprintln!("SKIP: validation layer not active");
        return;
    }
    let mut pass = UniversePass::new(&mut be, 64, 64).unwrap();
    let s = Source { id: 1, center: [32.0, 32.0], flux: 1.0, rgb: [1.0; 3] };
    let splat = Splat::from_source(&s);
    pass.render_live(&mut be, &[splat], PRESENT_EXPOSURE).unwrap();
    let (w, h) = pass.extent();
    // Mimic present_transfer_src: async TRANSFER_READ of SDR without waiting.
    be.submit_transfer_src_read(pass.sdr_image(), ash::vk::Extent2D { width: w, height: h }).unwrap();
    // Next live frame must not race the prior TRANSFER_READ (desktop bug).
    let before = render_ash::validation::message_counts().0;
    pass.render_live(&mut be, &[splat], PRESENT_EXPOSURE).unwrap();
    let after = render_ash::validation::message_counts().0;
    assert_eq!(
        after, before,
        "WRITE_AFTER_READ hazard on SDR after transfer consumer (errors {} -> {})",
        before, after
    );
    // Continuous present-like cadence.
    for _ in 0..4 {
        be.submit_transfer_src_read(pass.sdr_image(), ash::vk::Extent2D { width: w, height: h }).unwrap();
        let b = render_ash::validation::message_counts().0;
        pass.render_live(&mut be, &[splat], PRESENT_EXPOSURE).unwrap();
        assert_eq!(render_ash::validation::message_counts().0, b);
    }
    pass.free(&mut be);
    drop(be);
    assert_eq!(render_ash::validation::message_counts().0, 0);
}
