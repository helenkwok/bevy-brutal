//! The contract between Bevy (via `bridge`) and the low-level crates.
//! Plain data only: no Bevy types, no allocation policy, no unsafe.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};

/// SIMD lane width all SoA buffers are padded to (one AVX `f32x8`).
pub const LANES: usize = 8;

/// Round `n` up to a multiple of [`LANES`].
#[inline]
pub const fn padded_len(n: usize) -> usize {
    n.div_ceil(LANES) * LANES
}

/// Owned SoA particle storage. Every column has the same, `LANES`-padded length.
/// Padding particles have `life = 0` and are ignored by consumers that check `live`.
#[derive(Default, Clone, Debug)]
pub struct ParticleBuffers {
    pub px: Vec<f32>,
    pub py: Vec<f32>,
    pub pz: Vec<f32>,
    pub vx: Vec<f32>,
    pub vy: Vec<f32>,
    pub vz: Vec<f32>,
    pub life: Vec<f32>,
    /// Number of real (non-padding) particles.
    pub live: usize,
}

impl ParticleBuffers {
    /// Clear and resize to hold `n` particles (padded). Does not shrink capacity.
    pub fn reset(&mut self, n: usize) {
        let len = padded_len(n);
        for col in self.columns_mut() {
            col.clear();
            col.resize(len, 0.0);
        }
        self.live = n;
    }

    #[inline]
    pub fn len_padded(&self) -> usize {
        self.px.len()
    }

    fn columns_mut(&mut self) -> [&mut Vec<f32>; 7] {
        [&mut self.px, &mut self.py, &mut self.pz, &mut self.vx, &mut self.vy, &mut self.vz, &mut self.life]
    }

    /// Borrow as a mutable SoA view.
    pub fn view_mut(&mut self) -> ParticlesSoA<'_> {
        ParticlesSoA {
            px: &mut self.px,
            py: &mut self.py,
            pz: &mut self.pz,
            vx: &mut self.vx,
            vy: &mut self.vy,
            vz: &mut self.vz,
            life: &mut self.life,
        }
    }
}

/// Borrowed SoA view. All slices have equal length, a multiple of [`LANES`].
pub struct ParticlesSoA<'a> {
    pub px: &'a mut [f32],
    pub py: &'a mut [f32],
    pub pz: &'a mut [f32],
    pub vx: &'a mut [f32],
    pub vy: &'a mut [f32],
    pub vz: &'a mut [f32],
    pub life: &'a mut [f32],
}

impl ParticlesSoA<'_> {
    pub fn len(&self) -> usize {
        self.px.len()
    }
    pub fn is_empty(&self) -> bool {
        self.px.is_empty()
    }
    /// Panics unless all columns are equal length and lane-padded. Call at API boundaries.
    pub fn assert_valid(&self) {
        let n = self.px.len();
        assert_eq!(n % LANES, 0, "SoA length {n} not padded to {LANES}");
        for (name, l) in [
            ("py", self.py.len()),
            ("pz", self.pz.len()),
            ("vx", self.vx.len()),
            ("vy", self.vy.len()),
            ("vz", self.vz.len()),
            ("life", self.life.len()),
        ] {
            assert_eq!(l, n, "column {name} has len {l}, expected {n}");
        }
    }
}

/// Simulation parameters for particle integration.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParticleParams {
    pub dt: f32,
    pub gravity: [f32; 3],
    /// Linear drag per second, in [0, 1).
    pub drag: f32,
}

/// One GPU instance. `#[repr(C)]`, 80 bytes, std430-compatible.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
pub struct InstanceGpu {
    pub model: [[f32; 4]; 4],
    pub mesh_id: u32,
    pub material_id: u32,
    pub _pad: [u32; 2],
}

/// Per-frame snapshot handed to a render backend. Plain data, borrowed from the bridge.
pub struct FrameSnapshot<'a> {
    pub view_proj: [[f32; 4]; 4],
    pub instances: &'a [InstanceGpu],
    pub frame_index: u64,
}

/// A render backend consumes snapshots. Implementations must not allocate on the host
/// after warm-up and must not block longer than their documented budget.
pub trait RenderBackend {
    type Error: core::fmt::Debug;
    fn resize(&mut self, width: u32, height: u32) -> Result<(), Self::Error>;
    fn submit(&mut self, frame: &FrameSnapshot<'_>) -> Result<(), Self::Error>;
}

/// A hot system: pure function from inputs to outputs over borrowed plain data.
pub trait HotSystem {
    type Input<'a>;
    type Output<'a>;
    fn run(&mut self, input: Self::Input<'_>, out: Self::Output<'_>);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn instance_layout() {
        assert_eq!(core::mem::size_of::<InstanceGpu>(), 80);
    }
    #[test]
    fn padding() {
        assert_eq!(padded_len(0), 0);
        assert_eq!(padded_len(1), 8);
        assert_eq!(padded_len(16), 16);
        let mut b = ParticleBuffers::default();
        b.reset(13);
        assert_eq!(b.len_padded(), 16);
        b.view_mut().assert_valid();
    }
}
