//! Synthetic fixtures only, never observed catalogue redistribution.
use glam::DVec3;
use planet::{
    observed::{self, Kind, Point, select_markers},
    universe::{Camera, MAX_STARS},
};
fn point(id: u64, pos: DVec3) -> Point {
    Point { id, cross_id: None, pos, epoch: Some(2000.0), quantity: 100.0, error: 1.0, direct_g: None }
}
fn binary(kind: u64) -> Vec<u8> {
    let mut b = b"OBSdiag1".to_vec();
    for x in [kind, 2, 1] {
        b.extend(x.to_le_bytes());
    }
    b.extend([7; 32]);
    b.extend(9007199254740993u64.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    for x in [1e17f64, 0., 0., 2000., 100., 1., f64::NAN] {
        b.extend(x.to_le_bytes());
    }
    b
}
#[test]
fn typed_snapshot_exact_ids_and_corruption() {
    let b = binary(1);
    let c = observed::read(&mut &b[..]).unwrap();
    assert_eq!(c.kind, Kind::Cns5);
    assert_eq!(c.raw_rows, 2);
    assert_eq!(c.points[0].id, 9007199254740993);
    assert_eq!(c.points[0].epoch, Some(2000.));
    assert_eq!(c.points[0].direct_g, None);
    assert!(observed::read(&mut &b[..b.len() - 1]).is_err());
    let mut bad = b.clone();
    bad.push(0);
    assert!(observed::read(&mut &bad[..]).is_err());
    let mut bad = b.clone();
    bad[16..24].copy_from_slice(&100001u64.to_le_bytes());
    assert!(observed::read(&mut &bad[..]).is_err());
    assert!(observed::read(&mut &binary(3)[..]).is_err());
}
#[test]
fn all_records_scanned_capacity_deterministic_and_no_inverse_square() {
    let cam = Camera::look_at(DVec3::ZERO, DVec3::X, DVec3::Z, 1., 640, 480).unwrap();
    let mut pts = Vec::new();
    for id in 0..70000 {
        pts.push(point(id, DVec3::X * (100000. - id as f64)));
    }
    let (markers, visible) = select_markers(&pts, &cam, MAX_STARS);
    assert_eq!(visible, 70000);
    assert_eq!(markers.len(), MAX_STARS);
    assert_eq!(markers[0].id, 69999);
    assert!(markers.iter().any(|m| m.id == 69999));
    assert_eq!(markers[0].center, [320., 240.]);
    assert_eq!(select_markers(&pts, &cam, 0).0.len(), 0);
    let p = [point(1, DVec3::X), point(2, DVec3::X * 2.)];
    let (m, _) = select_markers(&p, &cam, 2);
    assert_eq!(m[0].center, m[1].center);
}
#[test]
fn geometry_culling_translation_and_native_epoch() {
    let cam = Camera::look_at(DVec3::ZERO, DVec3::X, DVec3::Z, 1., 640, 480).unwrap();
    let p = [point(1, DVec3::ZERO), point(2, -DVec3::X), point(3, DVec3::new(10., 1., 0.))];
    let (m, n) = select_markers(&p, &cam, 10);
    assert_eq!(n, 1);
    assert_eq!(m.len(), 1);
    let moved = Camera::look_at(DVec3::Y, DVec3::Y + DVec3::X, DVec3::Z, 1., 640, 480).unwrap();
    assert_ne!(m[0].center, select_markers(&p[2..], &moved, 10).0[0].center);
    let invalid = Camera { fov: f64::NAN, ..cam };
    assert_eq!(select_markers(&p, &invalid, 10).1, 0);
}
#[test]
fn duplicate_nonfinite_and_kind_specific_fields_rejected() {
    let b = binary(1);
    let mut duplicate = b.clone();
    duplicate[24..32].copy_from_slice(&2u64.to_le_bytes());
    duplicate.extend(&b[64..]);
    assert!(observed::read(&mut &duplicate[..]).is_err());
    for (offset, value) in [(80, f64::NAN), (104, f64::INFINITY), (112, -1.), (120, -1.)] {
        let mut bad = b.clone();
        bad[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        assert!(observed::read(&mut &bad[..]).is_err());
    }
    let mut desi = binary(2);
    desi[104..112].copy_from_slice(&f64::NAN.to_le_bytes());
    desi[112..120].copy_from_slice(&0.1f64.to_le_bytes());
    desi[120..128].copy_from_slice(&0.0001f64.to_le_bytes());
    let c = observed::read(&mut &desi[..]).unwrap();
    assert_eq!(c.points[0].epoch, None);
    desi[128..136].copy_from_slice(&20f64.to_le_bytes());
    assert!(observed::read(&mut &desi[..]).is_err());
}
