//! Shared helpers for the GPU integration tests.

/// True on CI (`CI=true` or `GITHUB_ACTIONS=true`), where a Vulkan device (lavapipe) is always
/// provisioned, so a failed device/backend init is a real regression rather than a missing GPU.
pub fn gpu_required() -> bool {
    ["CI", "GITHUB_ACTIONS"].iter().any(|k| std::env::var(k).is_ok_and(|v| v.eq_ignore_ascii_case("true")))
}

/// Skip locally when no usable Vulkan device exists; panic on CI so the job cannot go green
/// without running the GPU assertions.
#[track_caller]
pub fn skip_or_fail(what: &str, err: &dyn std::fmt::Debug) {
    if gpu_required() {
        panic!("{what} failed on CI (a Vulkan device is expected): {err:?}");
    }
    eprintln!("SKIP: {what} failed (no usable Vulkan 1.3 device?): {err:?}");
}
