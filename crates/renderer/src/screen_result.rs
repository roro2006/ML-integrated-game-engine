//! A screen-space result that group 0 binds for the renderer's whole life:
//! ambient occlusion's and the contact shadows' (C4).
//!
//! Such a result lives outside the render graph's pool, like the shadow
//! atlas, because the bind group holding it outlives every frame. When its
//! effect is off, no pass writes it; instead it is cleared once to a value
//! the lit shaders read as "no effect", so the frame pays nothing and the
//! image is exactly as without the effect.

/// One persistent result texture, its size, and whether it holds the
/// "no effect" value.
#[derive(Debug)]
pub(crate) struct ScreenResult {
    label: &'static str,
    format: wgpu::TextureFormat,
    texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    size: (u32, u32),
    /// What "no effect" is, in this result's channels.
    neutral: wgpu::Color,
    /// Whether the texture holds `neutral` rather than a pass's output.
    neutral_now: bool,
}

impl ScreenResult {
    /// A result of `size`, clamped to what the device can create, cleared to
    /// `neutral`.
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &'static str,
        format: wgpu::TextureFormat,
        size: (u32, u32),
        neutral: wgpu::Color,
    ) -> Self {
        let size = clamp_to_device(device, size);
        let (texture, view) = create_texture(device, label, format, size);
        let mut result = Self {
            label,
            format,
            texture,
            view,
            size,
            neutral,
            neutral_now: false,
        };
        result.make_neutral(device, queue);
        result
    }

    /// The texture's size: what a pass writing it dispatches over.
    pub(crate) fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Recreates the texture at `size` if that differs, returning whether it
    /// did, so the caller rebuilds group 0. A new texture holds zeros, not
    /// the neutral value, until a pass writes it or it is made neutral.
    pub(crate) fn resize(&mut self, device: &wgpu::Device, size: (u32, u32)) -> bool {
        let size = clamp_to_device(device, size);
        if size == self.size {
            return false;
        }
        (self.texture, self.view) = create_texture(device, self.label, self.format, size);
        self.size = size;
        self.neutral_now = false;
        true
    }

    /// Records that a pass writes the texture this frame.
    pub(crate) fn mark_written(&mut self) {
        self.neutral_now = false;
    }

    /// Clears the texture to its neutral value unless it already holds it.
    /// Its own small submission: this happens when the effect is turned off
    /// or the output is resized while it is off, never per frame.
    pub(crate) fn make_neutral(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.neutral_now {
            return;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(self.label),
        });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(self.label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.neutral),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        queue.submit(std::iter::once(encoder.finish()));
        self.neutral_now = true;
    }

    /// The texture, for tests that read it back.
    #[cfg(test)]
    pub(crate) fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }
}

/// `size` within what the device can create: an output too large to draw
/// skips its frames, but the result group 0 binds must still exist.
fn clamp_to_device(device: &wgpu::Device, (width, height): (u32, u32)) -> (u32, u32) {
    let max = device.limits().max_texture_dimension_2d;
    (width.clamp(1, max), height.clamp(1, max))
}

fn create_texture(
    device: &wgpu::Device,
    label: &'static str,
    format: wgpu::TextureFormat,
    (width, height): (u32, u32),
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        // Written by a compute pass, read by the lit shaders, cleared as an
        // attachment while the effect is off, and copied out by tests.
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}
