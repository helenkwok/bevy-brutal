//! Vulkan instance/device creation, validation messenger, and small GPU resource helpers.
use std::ffi::{CStr, c_char};
use std::mem::ManuallyDrop;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::BackendError;

/// Raw handles of the window to present to (from Bevy's `RawHandleWrapper`).
#[derive(Copy, Clone, Debug)]
pub struct WindowHandles {
    pub display: RawDisplayHandle,
    pub window: RawWindowHandle,
}

pub struct Surface {
    pub loader: ash::khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
}

/// Subgroup capabilities of the selected device (VkPhysicalDeviceSubgroupProperties via
/// Vulkan11Properties, plus the Vulkan 1.3 subgroup-size-control properties/features).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SubgroupInfo {
    /// Default subgroup size.
    pub size: u32,
    pub supported_stages: vk::ShaderStageFlags,
    pub supported_operations: vk::SubgroupFeatureFlags,
    pub min_size: u32,
    pub max_size: u32,
    pub max_compute_workgroup_subgroups: u32,
    /// Stages that accept VkPipelineShaderStageRequiredSubgroupSizeCreateInfo.
    pub required_size_stages: vk::ShaderStageFlags,
    /// `subgroupSizeControl` feature (enabled on the device when supported).
    pub size_control: bool,
    /// `computeFullSubgroups` feature (enabled on the device when supported).
    pub compute_full_subgroups: bool,
}

impl SubgroupInfo {
    /// Subgroup arithmetic (WavePrefixSum / WaveActiveSum) + ballot/basic usable in compute shaders.
    pub fn compute_arithmetic(&self) -> bool {
        self.supported_stages.contains(vk::ShaderStageFlags::COMPUTE)
            && self
                .supported_operations
                .contains(vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC)
    }
}

/// Core Vulkan objects. Destruction order is in `Drop`: allocator -> device -> surface -> messenger -> instance.
pub struct VkContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue_family: u32,
    pub queue: vk::Queue,
    pub device_name: String,
    /// Nanoseconds per timestamp tick; `None` if the queue has no timestamp support.
    pub timestamp_period_ns: Option<f32>,
    pub validation_enabled: bool,
    /// Synchronization validation enabled via VK_EXT_validation_features.
    pub sync_validation: bool,
    pub subgroup: SubgroupInfo,
    pub surface: Option<Surface>,
    /// `textureCompressionBC` enabled (supported by the device).
    pub texture_bc: bool,
    /// `samplerAnisotropy` enabled, with the device's max anisotropy.
    pub max_anisotropy: Option<f32>,
    pub allocator: ManuallyDrop<Allocator>,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

impl VkContext {
    /// Create an instance + device. `window` adds surface/swapchain support; `validation` enables
    /// VK_LAYER_KHRONOS_validation (+ synchronization validation when the layer exposes
    /// VK_EXT_validation_features) and a counting debug messenger, if the layer is installed.
    pub fn new(window: Option<WindowHandles>, validation: bool) -> Result<Self, BackendError> {
        // SAFETY: loads the system Vulkan loader. The Entry is stored in `Self` and outlives every
        // object created from it.
        let entry = unsafe { ash::Entry::load() }.map_err(BackendError::Loading)?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"bevy-brutal")
            .engine_name(c"bevy-brutal")
            .api_version(vk::API_VERSION_1_3);

        let mut exts: Vec<*const c_char> = match window {
            Some(w) => ash_window::enumerate_required_extensions(w.display)?.to_vec(),
            None => Vec::new(),
        };
        let layer = c"VK_LAYER_KHRONOS_validation";
        // SAFETY: entry is a valid loaded Entry.
        let layers_avail = unsafe { entry.enumerate_instance_layer_properties()? };
        let validation = validation && layers_avail.iter().any(|l| l.layer_name_as_c_str().is_ok_and(|n| n == layer));
        let mut layers: Vec<*const c_char> = Vec::new();
        let mut sync_validation = false;
        if validation {
            layers.push(layer.as_ptr());
            exts.push(ash::ext::debug_utils::NAME.as_ptr());
            // SAFETY: entry is valid; `layer` is a NUL-terminated layer name.
            let layer_exts = unsafe { entry.enumerate_instance_extension_properties(Some(layer))? };
            if layer_exts
                .iter()
                .any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == ash::ext::validation_features::NAME))
            {
                exts.push(ash::ext::validation_features::NAME.as_ptr());
                sync_validation = true;
            }
        }
        let enables = [vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION];
        let mut vfeatures = vk::ValidationFeaturesEXT::default().enabled_validation_features(&enables);
        let mut ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&exts);
        if sync_validation {
            ci = ci.push_next(&mut vfeatures);
        }
        // SAFETY: `ci` and everything it references (app_info, name arrays of 'static CStrs, vfeatures)
        // are alive for the duration of the call.
        let instance = unsafe { entry.create_instance(&ci, None)? };

        let debug = if validation {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let mci = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(crate::validation::debug_callback));
            // SAFETY: debug_utils was enabled on this instance; the callback is a 'static fn.
            let m = unsafe { loader.create_debug_utils_messenger(&mci, None)? };
            Some((loader, m))
        } else {
            None
        };

        // Partially-built context so early returns clean up via Drop-like helper.
        let fail = |instance: &ash::Instance,
                    debug: &Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
                    surface: &Option<Surface>| {
            // SAFETY: no device exists yet; children (messenger, surface) are destroyed before the instance.
            unsafe {
                if let Some(s) = surface {
                    s.loader.destroy_surface(s.surface, None);
                }
                if let Some((l, m)) = debug {
                    l.destroy_debug_utils_messenger(*m, None);
                }
                instance.destroy_instance(None);
            }
        };

        let surface = match window {
            Some(w) => {
                // SAFETY: the caller guarantees the window/display handles are valid and outlive the
                // backend (Bevy keeps the winit window alive while the Window entity exists).
                match unsafe { ash_window::create_surface(&entry, &instance, w.display, w.window, None) } {
                    Ok(s) => Some(Surface { loader: ash::khr::surface::Instance::new(&entry, &instance), surface: s }),
                    Err(e) => {
                        fail(&instance, &debug, &None);
                        return Err(e.into());
                    }
                }
            }
            None => None,
        };

        let Some((physical_device, queue_family, device_name)) = Self::pick_device(&instance, surface.as_ref()) else {
            fail(&instance, &debug, &surface);
            return Err(BackendError::NoSuitableDevice);
        };

        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(queue_family).queue_priorities(&prio)];
        let dev_exts: Vec<*const c_char> =
            if surface.is_some() { vec![ash::khr::swapchain::NAME.as_ptr()] } else { vec![] };
        // Optional features: subgroup size control / full compute subgroups (used by the subgroup scan).
        let (size_control, full_subgroups) = {
            let mut q13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut q2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut q13);
            // SAFETY: valid physical device; q2's pNext chain points at a live local.
            unsafe { instance.get_physical_device_features2(physical_device, &mut q2) };
            let (a, b) = (q13.subgroup_size_control == vk::TRUE, q13.compute_full_subgroups == vk::TRUE);
            (a, b)
        };
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true)
            // `discard` in the quality pass (LOD dither, impostor alpha) is demote-to-helper in SPIR-V 1.6
            .shader_demote_to_helper_invocation(true)
            .subgroup_size_control(size_control)
            .compute_full_subgroups(full_subgroups);
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .draw_indirect_count(true)
            .buffer_device_address(true)
            .scalar_block_layout(true);
        // Optional core features: BC texture compression (planet textures) and anisotropic filtering.
        // SAFETY: valid physical device handle.
        let core = unsafe { instance.get_physical_device_features(physical_device) };
        let (texture_bc, anisotropy) = (core.texture_compression_bc == vk::TRUE, core.sampler_anisotropy == vk::TRUE);
        let mut f2 = vk::PhysicalDeviceFeatures2::default().features(
            vk::PhysicalDeviceFeatures::default()
                .multi_draw_indirect(true)
                .texture_compression_bc(texture_bc)
                .sampler_anisotropy(anisotropy),
        );
        let dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .enabled_extension_names(&dev_exts)
            .push_next(&mut f2)
            .push_next(&mut f13)
            .push_next(&mut f12);
        // SAFETY: physical_device came from `instance`; dci and its pNext chain are alive for the
        // call; pick_device verified every enabled feature is supported.
        let device = match unsafe { instance.create_device(physical_device, &dci, None) } {
            Ok(d) => d,
            Err(e) => {
                fail(&instance, &debug, &surface);
                return Err(e.into());
            }
        };
        // SAFETY: family `queue_family`, index 0 was requested in `qci`.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        // SAFETY: valid physical device handle.
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        // SAFETY: valid physical device handle.
        let qprops = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let timestamp_period_ns = (qprops[queue_family as usize].timestamp_valid_bits > 0
            && props.limits.timestamp_compute_and_graphics == vk::TRUE)
            .then_some(props.limits.timestamp_period);

        let subgroup = {
            let mut p11 = vk::PhysicalDeviceVulkan11Properties::default();
            let mut p13 = vk::PhysicalDeviceVulkan13Properties::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut p11).push_next(&mut p13);
            // SAFETY: valid physical device (API >= 1.3, checked in pick_device); pNext chain is live.
            unsafe { instance.get_physical_device_properties2(physical_device, &mut p2) };
            SubgroupInfo {
                size: p11.subgroup_size,
                supported_stages: p11.subgroup_supported_stages,
                supported_operations: p11.subgroup_supported_operations,
                min_size: p13.min_subgroup_size,
                max_size: p13.max_subgroup_size,
                max_compute_workgroup_subgroups: p13.max_compute_workgroup_subgroups,
                required_size_stages: p13.required_subgroup_size_stages,
                size_control,
                compute_full_subgroups: full_subgroups,
            }
        };

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            queue_family,
            queue,
            device_name,
            timestamp_period_ns,
            validation_enabled: validation,
            sync_validation,
            subgroup,
            surface,
            texture_bc,
            max_anisotropy: anisotropy.then_some(props.limits.max_sampler_anisotropy),
            allocator: ManuallyDrop::new(allocator),
            debug,
        })
    }

    fn pick_device(instance: &ash::Instance, surface: Option<&Surface>) -> Option<(vk::PhysicalDevice, u32, String)> {
        // SAFETY: instance is valid for the duration of this function.
        let devices = unsafe { instance.enumerate_physical_devices() }.ok()?;
        let mut best: Option<(i32, vk::PhysicalDevice, u32, String)> = None;
        for pd in devices {
            // SAFETY: pd was enumerated from this instance.
            let props = unsafe { instance.get_physical_device_properties(pd) };
            if props.api_version < vk::API_VERSION_1_3 {
                continue;
            }
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f13).push_next(&mut f12);
            // SAFETY: pd is valid; f2's pNext chain points at live locals.
            unsafe { instance.get_physical_device_features2(pd, &mut f2) };
            let multi_draw_indirect = f2.features.multi_draw_indirect;
            let ok = f13.dynamic_rendering == vk::TRUE
                && f13.synchronization2 == vk::TRUE
                && f13.shader_demote_to_helper_invocation == vk::TRUE
                && f12.timeline_semaphore == vk::TRUE
                && f12.draw_indirect_count == vk::TRUE
                && f12.buffer_device_address == vk::TRUE
                && f12.scalar_block_layout == vk::TRUE
                && multi_draw_indirect == vk::TRUE;
            if !ok {
                continue;
            }
            // SAFETY: pd is valid.
            let qf = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            let Some(qi) = qf.iter().enumerate().position(|(i, q)| {
                q.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                    && surface.is_none_or(|s| {
                        // SAFETY: pd, queue index and surface are valid handles from this instance.
                        unsafe { s.loader.get_physical_device_surface_support(pd, i as u32, s.surface) }
                            .unwrap_or(false)
                    })
            }) else {
                continue;
            };
            let score = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                _ => 0,
            };
            let name = props.device_name_as_c_str().map(CStr::to_string_lossy).unwrap_or_default().into_owned();
            if best.as_ref().is_none_or(|b| score > b.0) {
                best = Some((score, pd, qi as u32, name));
            }
        }
        best.map(|(_, pd, qi, n)| (pd, qi, n))
    }

    /// Record and synchronously execute a one-off command buffer (uploads, readbacks).
    pub fn immediate_submit(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<(), BackendError> {
        let d = &self.device;
        let pci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.queue_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        // SAFETY: device valid; all objects created here are destroyed before returning, after the
        // fence wait guarantees the GPU is done with them.
        unsafe {
            let pool = d.create_command_pool(&pci, None)?;
            let cmd = d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1),
            )?[0];
            d.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            record(cmd);
            d.end_command_buffer(cmd)?;
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cbi = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let res = d
                .queue_submit2(self.queue, &[vk::SubmitInfo2::default().command_buffer_infos(&cbi)], fence)
                .and_then(|_| d.wait_for_fences(&[fence], true, u64::MAX));
            d.destroy_fence(fence, None);
            d.destroy_command_pool(pool, None);
            res?;
        }
        Ok(())
    }

    pub fn wait_idle(&self) {
        // SAFETY: device is valid.
        unsafe {
            let _ = self.device.device_wait_idle();
        }
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        // SAFETY: wait idle so no GPU work references our objects; the allocator frees its memory
        // before the device dies; device before surface/messenger/instance. Nothing is used after.
        unsafe {
            let _ = self.device.device_wait_idle();
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            if let Some(s) = self.surface.take() {
                s.loader.destroy_surface(s.surface, None);
            }
            if let Some((l, m)) = self.debug.take() {
                l.destroy_debug_utils_messenger(m, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

// ---------- resources ----------------------------------------------------------------------

/// A buffer with its allocation and (if requested) device address.
pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub alloc: Option<Allocation>,
    pub size: vk::DeviceSize,
    pub address: vk::DeviceAddress,
}

impl GpuBuffer {
    pub fn new(
        ctx: &mut VkContext,
        name: &str,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<Self, BackendError> {
        let size = size.max(16);
        let bci = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: device valid; bci fully initialised.
        let buffer = unsafe { ctx.device.create_buffer(&bci, None)? };
        // SAFETY: buffer was just created from this device.
        let req = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
        let alloc = match ctx.allocator.allocate(&AllocationCreateDesc {
            name,
            requirements: req,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(a) => a,
            Err(e) => {
                // SAFETY: buffer is unbound and unused.
                unsafe { ctx.device.destroy_buffer(buffer, None) };
                return Err(e.into());
            }
        };
        // SAFETY: memory comes from our allocator on this device and satisfies `req`; buffer unbound.
        unsafe { ctx.device.bind_buffer_memory(buffer, alloc.memory(), alloc.offset())? };
        let address = if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            // SAFETY: buffer has SHADER_DEVICE_ADDRESS usage, is bound, and the feature is enabled.
            unsafe { ctx.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer)) }
        } else {
            0
        };
        Ok(Self { buffer, alloc: Some(alloc), size, address })
    }

    /// Host-mapped bytes (only for CpuToGpu / GpuToCpu buffers).
    pub fn mapped_mut(&mut self) -> &mut [u8] {
        self.alloc.as_mut().and_then(|a| a.mapped_slice_mut()).expect("buffer is not host-visible")
    }

    pub fn mapped(&self) -> &[u8] {
        self.alloc.as_ref().and_then(|a| a.mapped_slice()).expect("buffer is not host-visible")
    }

    pub fn destroy(&mut self, ctx: &mut VkContext) {
        if let Some(a) = self.alloc.take() {
            let _ = ctx.allocator.free(a);
        }
        // SAFETY: callers wait for the GPU to be idle before destroying (see AshBackend::drop/resize).
        unsafe { ctx.device.destroy_buffer(self.buffer, None) };
        self.buffer = vk::Buffer::null();
    }
}

/// A 2D image + view + allocation.
pub struct GpuImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub alloc: Option<Allocation>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl GpuImage {
    pub fn new(
        ctx: &mut VkContext,
        name: &str,
        extent: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
    ) -> Result<Self, BackendError> {
        let ici = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: device valid; ici fully initialised.
        let image = unsafe { ctx.device.create_image(&ici, None)? };
        // SAFETY: image was just created from this device.
        let req = unsafe { ctx.device.get_image_memory_requirements(image) };
        let alloc = ctx.allocator.allocate(&AllocationCreateDesc {
            name,
            requirements: req,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        // SAFETY: memory satisfies `req`; image unbound.
        unsafe { ctx.device.bind_image_memory(image, alloc.memory(), alloc.offset())? };
        let vci = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(subresource(aspect));
        // SAFETY: image is bound and the view matches its format/aspect.
        let view = unsafe { ctx.device.create_image_view(&vci, None)? };
        Ok(Self { image, view, alloc: Some(alloc), format, extent })
    }

    pub fn destroy(&mut self, ctx: &mut VkContext) {
        // SAFETY: callers ensure the GPU is idle; view destroyed before image; memory freed last.
        unsafe {
            ctx.device.destroy_image_view(self.view, None);
            ctx.device.destroy_image(self.image, None);
        }
        if let Some(a) = self.alloc.take() {
            let _ = ctx.allocator.free(a);
        }
    }
}

pub fn subresource(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// Record a global memory barrier (sync2).
pub fn memory_barrier(
    d: &ash::Device,
    cmd: vk::CommandBuffer,
    src: (vk::PipelineStageFlags2, vk::AccessFlags2),
    dst: (vk::PipelineStageFlags2, vk::AccessFlags2),
) {
    let b = [vk::MemoryBarrier2::default()
        .src_stage_mask(src.0)
        .src_access_mask(src.1)
        .dst_stage_mask(dst.0)
        .dst_access_mask(dst.1)];
    // SAFETY: cmd is in the recording state (callers only use this between begin/end).
    unsafe { d.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b)) };
}

/// Record an image layout transition (sync2).
#[allow(clippy::too_many_arguments)]
pub fn image_barrier(
    d: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    layouts: (vk::ImageLayout, vk::ImageLayout),
    src: (vk::PipelineStageFlags2, vk::AccessFlags2),
    dst: (vk::PipelineStageFlags2, vk::AccessFlags2),
) {
    let b = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src.0)
        .src_access_mask(src.1)
        .dst_stage_mask(dst.0)
        .dst_access_mask(dst.1)
        .old_layout(layouts.0)
        .new_layout(layouts.1)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource(aspect))];
    // SAFETY: cmd is recording; `image` is a live image created with a compatible aspect.
    unsafe { d.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&b)) };
}
