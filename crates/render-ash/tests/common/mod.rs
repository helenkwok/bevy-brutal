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

/// True when `BRUTAL_REQUIRE_VALIDATION=1` (CI sets it after installing the Khronos layer).
pub fn validation_required() -> bool {
    std::env::var("BRUTAL_REQUIRE_VALIDATION").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// With `BRUTAL_REQUIRE_VALIDATION=1`, panic unless the Khronos validation layer *and*
/// synchronization validation are active, so "zero validation errors" cannot pass vacuously.
#[track_caller]
pub fn check_validation(ctx: &render_ash::VkContext) {
    if validation_required() {
        assert!(ctx.validation_enabled, "BRUTAL_REQUIRE_VALIDATION=1 but VK_LAYER_KHRONOS_validation is not active");
        assert!(ctx.sync_validation, "BRUTAL_REQUIRE_VALIDATION=1 but synchronization validation is not enabled");
    }
}
