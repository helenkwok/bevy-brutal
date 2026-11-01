//! Swapchain + present path (feature `windowed`). The pass renders into its offscreen color target,
//! which is blitted to the acquired swapchain image; so pipelines never depend on the surface format.
use ash::vk;

use crate::context::VkContext;
use crate::BackendError;

pub struct Swapchain {
    pub loader: ash::khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    /// Signalled by acquire, waited by the frame's submit. One per frame slot.
    pub acquire: Vec<vk::Semaphore>,
    /// Signalled by the frame's submit, waited by present. One per swapchain image.
    pub render_finished: Vec<vk::Semaphore>,
}

impl Swapchain {
    pub fn new(
        ctx: &VkContext,
        desired: vk::Extent2D,
        old: vk::SwapchainKHR,
        frames_in_flight: usize,
    ) -> Result<Self, BackendError> {
        let s = ctx.surface.as_ref().ok_or(BackendError::NoSurface)?;
        let pd = ctx.physical_device;
        // SAFETY: surface and physical device are valid handles of the same instance.
        let (caps, formats) = unsafe {
            (
                s.loader.get_physical_device_surface_capabilities(pd, s.surface)?,
                s.loader.get_physical_device_surface_formats(pd, s.surface)?,
            )
        };
        if !caps.supported_usage_flags.contains(vk::ImageUsageFlags::TRANSFER_DST) {
            return Err(BackendError::Unsupported("swapchain images cannot be TRANSFER_DST"));
        }
        let fmt = formats
            .iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_UNORM || f.format == vk::Format::R8G8B8A8_UNORM)
            .or(formats.first())
            .copied()
            .ok_or(BackendError::Unsupported("surface reports no formats"))?;
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: desired.width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: desired.height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        let mut count = caps.min_image_count + 1;
        if caps.max_image_count > 0 {
            count = count.min(caps.max_image_count);
        }
        let ci = vk::SwapchainCreateInfoKHR::default()
            .surface(s.surface)
            .min_image_count(count)
            .image_format(fmt.format)
            .image_color_space(fmt.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(old);
        let loader = ash::khr::swapchain::Device::new(&ctx.instance, &ctx.device);
        // SAFETY: VK_KHR_swapchain was enabled on the device (surface present); ci references live data.
        let swapchain = unsafe { loader.create_swapchain(&ci, None)? };
        // SAFETY: swapchain was just created.
        let images = unsafe { loader.get_swapchain_images(swapchain)? };
        let sem = || {
            // SAFETY: device valid.
            unsafe { ctx.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
        };
        let acquire = (0..frames_in_flight).map(|_| sem()).collect::<Result<Vec<_>, _>>()?;
        let render_finished = images.iter().map(|_| sem()).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { loader, swapchain, images, format: fmt.format, extent, acquire, render_finished })
    }

    /// Acquire the next image, signalling `acquire[slot]`. `Ok(None)` means out of date: recreate.
    pub fn acquire(&self, slot: usize) -> Result<Option<u32>, BackendError> {
        // SAFETY: swapchain and semaphore are live; the semaphore is unsignalled because the previous
        // submit that waited on it (same slot) completed (timeline wait in begin_frame).
        match unsafe { self.loader.acquire_next_image(self.swapchain, u64::MAX, self.acquire[slot], vk::Fence::null()) } {
            Ok((i, _suboptimal)) => Ok(Some(i)),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Present `image`. Returns false if the swapchain must be recreated.
    pub fn present(&self, queue: vk::Queue, image: u32) -> Result<bool, BackendError> {
        let waits = [self.render_finished[image as usize]];
        let scs = [self.swapchain];
        let idx = [image];
        let pi = vk::PresentInfoKHR::default().wait_semaphores(&waits).swapchains(&scs).image_indices(&idx);
        // SAFETY: image was acquired and the submit signalling `waits` has been queued.
        match unsafe { self.loader.queue_present(queue, &pi) } {
            Ok(false) => Ok(true),
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        // SAFETY: called after device_wait_idle; semaphores and swapchain are no longer in use.
        unsafe {
            for s in self.acquire.iter().chain(&self.render_finished) {
                ctx.device.destroy_semaphore(*s, None);
            }
            self.loader.destroy_swapchain(self.swapchain, None);
        }
    }
}
