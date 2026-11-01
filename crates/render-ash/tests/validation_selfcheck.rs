//! Proves the counting debug messenger is live: a deliberately invalid call must be reported.
//! Separate test binary (own process) so the global error counter of tests/gpu.rs stays clean.
mod common;

use ash::vk;
use render_ash::VkContext;
use render_ash::validation::message_counts;

#[test]
fn messenger_reports_invalid_usage() {
    let ctx = match VkContext::new(None, true) {
        Ok(c) if c.validation_enabled => {
            common::check_validation(&c);
            c
        }
        Ok(c) => {
            common::check_validation(&c); // panics when BRUTAL_REQUIRE_VALIDATION=1
            return eprintln!("SKIP: validation layer not installed");
        }
        // CI provisions lavapipe, so a missing device fails there; the layer itself is optional.
        Err(e) => return common::skip_or_fail("VkContext::new", &e),
    };
    let before = message_counts().0;
    // size = 0 violates VUID-VkBufferCreateInfo-size-00912.
    let bci = vk::BufferCreateInfo::default().size(0).usage(vk::BufferUsageFlags::STORAGE_BUFFER);
    // SAFETY: intentionally invalid parameters to trigger a validation error; the buffer (if the
    // driver returns one) is destroyed immediately and never used.
    unsafe {
        if let Ok(b) = ctx.device.create_buffer(&bci, None) {
            ctx.device.destroy_buffer(b, None);
        }
    }
    let after = message_counts().0;
    assert!(after > before, "validation layer did not report an obviously invalid call");
}
