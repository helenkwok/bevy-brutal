//! Native bindings live here (Jolt, FMOD, etc. via bindgen/cxx). STUB: one libm binding that
//! shows the pattern: raw `extern` declaration + safe wrapper with a documented contract.

mod sys {
    // SAFETY (declaration): matches C99 `float cosf(float)` from libm, which std already links on all
    // tier-1 targets. Edition 2024 requires `unsafe extern`; `safe fn` marks it callable without
    // `unsafe` because it has no pointer arguments or global-state preconditions.
    unsafe extern "C" {
        pub safe fn cosf(x: f32) -> f32;
    }
}

/// Safe wrapper. Real bindings: wrap handles in RAII types and document the C contract per fn.
#[inline]
pub fn native_cos(x: f32) -> f32 {
    sys::cosf(x)
}

#[cfg(test)]
mod tests {
    #[test]
    fn libm_roundtrip() {
        assert!((super::native_cos(0.0) - 1.0).abs() < 1e-6);
        assert!((super::native_cos(std::f32::consts::PI) + 1.0).abs() < 1e-6);
    }
}
