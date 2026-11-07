//! GCNS binary input; test fixtures below are synthetic, not redistributed observations.
use crate::universe::Star;
use glam::DVec3;
use std::io::{self, Read};
/// IAU 2012 AU and IAU 2015 parsec definition (648000/pi AU).
pub const PARSEC_M: f64 = 149_597_870_700.0 * 648_000.0 / std::f64::consts::PI;
pub struct Catalogue {
    pub stars: Vec<Star>,
    pub raw_rows: usize,
}
fn invalid() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid GCNSv001 catalogue")
}
/// Sol-local right-handed ICRS, static epoch J2016.0; no synthetic Sun insertion.
/// Full raw rows/uncertainties/photometry remain in the external retained CSV.
pub fn read(r: &mut impl Read) -> io::Result<Catalogue> {
    let mut h = [0u8; 24];
    r.read_exact(&mut h)?;
    if &h[..8] != b"GCNSv001" {
        return Err(invalid());
    }
    let raw = u64::from_le_bytes(h[8..16].try_into().unwrap());
    let count = u64::from_le_bytes(h[16..24].try_into().unwrap());
    if raw == 0 || raw > 400000 || count > raw {
        return Err(invalid());
    }
    let mut stars = Vec::with_capacity(count as usize);
    let mut ids = std::collections::HashSet::with_capacity(count as usize);
    for _ in 0..count {
        let mut b = [0u8; 40];
        r.read_exact(&mut b)?;
        let id = u64::from_le_bytes(b[..8].try_into().unwrap());
        let f = |i| f64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        let pos = DVec3::new(f(8), f(16), f(24));
        let lum = f(32);
        if id == 0
            || !ids.insert(id)
            || !pos.is_finite()
            || pos.length_squared() < 1.0
            || !lum.is_finite()
            || lum <= 0.0
        {
            return Err(invalid());
        }
        stars.push(Star { id, pos, lum, rgb: [1.0; 3] });
    }
    let mut extra = [0u8; 1];
    if r.read(&mut extra)? != 0 {
        return Err(invalid());
    }
    Ok(Catalogue { stars, raw_rows: raw as usize })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_id_and_invalid_inputs() {
        let mut bytes = b"GCNSv001".to_vec();
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(9007199254740993u64.to_le_bytes());
        for v in [1e17f64, 0.0, 0.0, 2.0] {
            bytes.extend(v.to_le_bytes());
        }
        let c = read(&mut &bytes[..]).unwrap();
        assert_eq!(c.stars[0].id, 9007199254740993);
        assert_eq!(c.raw_rows, 1);
        assert!(read(&mut &bytes[..bytes.len() - 1]).is_err());
        bytes[0] = 0;
        assert!(read(&mut &bytes[..]).is_err());
    }
}
