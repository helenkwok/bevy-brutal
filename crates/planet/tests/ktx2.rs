//! KTX2 writer / reader: round trip for every format x supercompression, corrupt-file rejection,
//! and (when `KTX_TOOL` names the Khronos `ktx` binary) `ktx validate` on every written file.
use planet::ktx2::{Ktx2, Ktx2Error, Supercompression, write, write_array};
use planet::pack::{TexFormat, Texture, layer_bytes};

fn synth(fmt: TexFormat, size: u32) -> Texture {
    let mips = if fmt.is_block() { size.trailing_zeros() - 1 } else { size.trailing_zeros() + 1 };
    let n = 6 * layer_bytes(fmt, size, mips);
    // deterministic, poorly compressible-but-not-random bytes
    let data = (0..n).map(|i| ((i as u32).wrapping_mul(2_654_435_761) >> 13) as u8 ^ (i / 97) as u8).collect();
    Texture { name: "t".into(), format: fmt, size, mips, data }
}

const FORMATS: [TexFormat; 9] = [
    TexFormat::Bc7Srgb,
    TexFormat::Bc7,
    TexFormat::Bc5,
    TexFormat::Bc4,
    TexFormat::Rgba8Srgb,
    TexFormat::Rgba8,
    TexFormat::Rg8,
    TexFormat::R8,
    TexFormat::R16,
];

#[test]
fn round_trip_all_formats() {
    let dir = std::env::temp_dir().join("planet_ktx2_test");
    std::fs::create_dir_all(&dir).unwrap();
    for fmt in FORMATS {
        for sc in [Supercompression::None, Supercompression::Zstd(3)] {
            let t = synth(fmt, 64);
            let bytes = write(&t, sc, &[("planet.test", b"1")]).unwrap();
            let k = Ktx2::parse(&bytes).unwrap();
            assert_eq!((k.format, k.size, k.mips), (fmt, 64, t.mips));
            assert_eq!(k.value("planet.test"), Some(&b"1"[..]));
            assert!(k.value("KTXwriter").is_some());
            assert_eq!(k.gpu_bytes(), t.data.len());
            assert_eq!(k.to_texture("t").unwrap().data, t.data, "{fmt:?} {sc:?}");
            let p = dir.join(format!("{fmt:?}_{}.ktx2", if sc == Supercompression::None { "none" } else { "zstd" }));
            std::fs::write(&p, &bytes).unwrap();
            if let Ok(tool) = std::env::var("KTX_TOOL") {
                // warnings are errors except 7010 (custom key: allowed by the spec, merely flagged)
                let o = std::process::Command::new(&tool)
                    .args(["validate", "-f", "mini-json"])
                    .arg(&p)
                    .output()
                    .expect("run ktx validate");
                let out = String::from_utf8_lossy(&o.stdout);
                let ids: Vec<&str> =
                    out.match_indices("\"id\":").map(|(i, _)| &out[i + 5..(i + 10).min(out.len())]).collect();
                assert!(
                    o.status.success() && ids.iter().all(|id| id.starts_with("7010")),
                    "ktx validate {}: {out}",
                    p.display()
                );
            }
        }
    }
}

#[test]
fn rejects_corrupt_files() {
    let t = synth(TexFormat::Bc7, 32);
    let good = write(&t, Supercompression::Zstd(3), &[]).unwrap();
    assert!(Ktx2::parse(&good).is_ok());
    // bad identifier, every truncation, oversized level offsets, bad face count, unknown vkFormat
    let mut b = good.clone();
    b[1] = b'Q';
    assert_eq!(Ktx2::parse(&b).unwrap_err(), Ktx2Error::BadIdentifier);
    for cut in [0, 11, 40, 79, 100, good.len() / 2, good.len() - 1] {
        assert!(Ktx2::parse(&good[..cut]).is_err(), "truncated at {cut}");
    }
    let mut b = good.clone();
    b[80..88].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(Ktx2::parse(&b).is_err());
    let mut b = good.clone();
    b[36..40].copy_from_slice(&1u32.to_le_bytes());
    assert!(Ktx2::parse(&b).is_err());
    let mut b = good.clone();
    b[12..16].copy_from_slice(&7u32.to_le_bytes());
    assert!(Ktx2::parse(&b).is_err());
    // a corrupt zstd frame is an error, not a panic or a short level
    let k = Ktx2::parse(&good).unwrap();
    let off = k.levels[0].offset;
    let mut b = good.clone();
    for x in &mut b[off + 4..off + 40] {
        *x ^= 0x5a;
    }
    let k = Ktx2::parse(&b).unwrap();
    let mut out = vec![0u8; k.levels[0].uncompressed];
    assert!(k.level_into(0, &mut out).is_err());
    // uncompressed level with a wrong length is rejected at parse time
    let raw = write(&t, Supercompression::None, &[]).unwrap();
    let mut b = raw.clone();
    let len = u64::from_le_bytes(b[88..96].try_into().unwrap());
    b[88..96].copy_from_slice(&(len - 16).to_le_bytes());
    assert!(Ktx2::parse(&b).is_err());
}

#[test]
fn round_trip_2d_array() {
    // Earth round 5: the graft's two nested levels as a 2-layer 2D array (BC7 sRGB / BC7)
    let dir = std::env::temp_dir().join("planet_ktx2_test");
    std::fs::create_dir_all(&dir).unwrap();
    for fmt in [TexFormat::Bc7Srgb, TexFormat::Bc7, TexFormat::Rgba8Srgb] {
        for sc in [Supercompression::None, Supercompression::Zstd(3)] {
            let mut t = synth(fmt, 64);
            let per = t.data.len() / 6;
            t.data.truncate(2 * per);
            let bytes = write_array(&t, 2, sc, &[]).unwrap();
            let k = Ktx2::parse(&bytes).unwrap();
            assert_eq!((k.format, k.size, k.mips, k.images, k.array), (fmt, 64, t.mips, 2, true));
            assert_eq!(k.gpu_bytes(), t.data.len());
            assert_eq!(k.to_texture("t").unwrap().data, t.data, "{fmt:?} {sc:?}");
            let p =
                dir.join(format!("array_{fmt:?}_{}.ktx2", if sc == Supercompression::None { "none" } else { "zstd" }));
            std::fs::write(&p, &bytes).unwrap();
            if let Ok(tool) = std::env::var("KTX_TOOL") {
                let o = std::process::Command::new(&tool)
                    .args(["validate", "-f", "mini-json"])
                    .arg(&p)
                    .output()
                    .expect("run ktx validate");
                let out = String::from_utf8_lossy(&o.stdout);
                let ids: Vec<&str> =
                    out.match_indices("\"id\":").map(|(i, _)| &out[i + 5..(i + 10).min(out.len())]).collect();
                assert!(
                    o.status.success() && ids.iter().all(|id| id.starts_with("7010")),
                    "ktx validate {}: {out}",
                    p.display()
                );
            }
            // a cube parser view of the same bytes would need 6 images: wrong data size is rejected
            assert!(write_array(&t, 3, sc, &[]).is_err());
        }
    }
}
