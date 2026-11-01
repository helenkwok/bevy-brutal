//! VK_EXT_debug_utils messenger that prints and counts validation messages.
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

use ash::vk;

static ERRORS: AtomicU32 = AtomicU32::new(0);
static WARNINGS: AtomicU32 = AtomicU32::new(0);

/// (errors, warnings) reported by the validation layer since process start.
pub fn message_counts() -> (u32, u32) {
    (ERRORS.load(Ordering::Relaxed), WARNINGS.load(Ordering::Relaxed))
}

/// # Safety
/// Called by the Vulkan loader with a valid callback-data pointer (or null) for the duration of the call.
pub unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: the loader guarantees `data` (when non-null) and its `p_message` string are valid,
    // NUL-terminated and alive for the duration of this call; we copy the text out before returning.
    let msg = unsafe { data.as_ref().and_then(|d| d.message_as_c_str()) }
        .map(|m| m.to_string_lossy().into_owned())
        .unwrap_or_default();
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        ERRORS.fetch_add(1, Ordering::Relaxed);
        eprintln!("[vk ERROR {ty:?}] {msg}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        WARNINGS.fetch_add(1, Ordering::Relaxed);
        eprintln!("[vk WARN {ty:?}] {msg}");
    }
    vk::FALSE
}
