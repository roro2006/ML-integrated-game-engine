//! Screen-space contact shadows (C4): a full-resolution march from every
//! visible point toward the directional light through the depth prepass,
//! into the persistent result the lit shaders read through group 0 and
//! combine with the cascades by taking the darker.
//!
//! The method is in `contact_shadow_trace.wgsl`. It reads the prepass's
//! normal target to start each march clear of its own surface, so a frame
//! with contact shadows has the prepass write normals, as occlusion does.
//! The result is a [`ScreenResult`]: off, the pass is not declared and the
//! result is cleared once to "lit", so the image is exactly as without it.

use crate::block_on::block_on;
use crate::frame::{ContactShadows, RenderFrame};
use crate::screen_result::ScreenResult;
use crate::shaders::CONTACT_SHADOW_TRACE;

/// The result's format: the smallest every device can write as storage.
pub(crate) const CONTACT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Bytes of `ContactParams`.
pub(crate) const CONTACT_PARAMS_SIZE: usize = 176;
const WORKGROUP: u32 = 8;
/// "Lit": what the lit shaders read when contact shadows are off.
const LIT: wgpu::Color = wgpu::Color::WHITE;

/// This frame's contact shadow settings and the unit direction to the light,
/// or `None` when it traces none: turned off, a setting that would do nothing
/// or is not finite, a light with no direction, or a camera that is not a
/// perspective one, which the march cannot reconstruct.
pub(crate) fn frame_settings(frame: &RenderFrame) -> Option<(ContactShadows, [f32; 3])> {
    let settings = frame.contact_shadows?;
    let usable = settings.length.is_finite()
        && settings.length > 0.0
        && settings.thickness.is_finite()
        && settings.thickness > 0.0;
    // The march's length is in metres along this direction, so it must be a
    // unit vector; `DirectionalLight::direction` is public and may not be.
    let to_light = glam::Vec3::from_array(frame.light.direction).try_normalize()?;
    (usable && crate::shadow::view_depths(&frame.view_proj).is_some())
        .then_some((settings, to_light.to_array()))
}

/// `ContactParams` for one frame. `to_light` is the light's unit direction,
/// and `size` the full output size.
pub(crate) fn params_bytes(
    view_proj: &[[f32; 4]; 4],
    inverse_view_proj: &[[f32; 4]; 4],
    to_light: [f32; 3],
    settings: &ContactShadows,
    size: (u32, u32),
) -> [u8; CONTACT_PARAMS_SIZE] {
    let (width, height) = (size.0.max(1) as f32, size.1.max(1) as f32);
    let mut floats = [0.0_f32; CONTACT_PARAMS_SIZE / 4];
    for (offset, matrix) in [(0, view_proj), (16, inverse_view_proj)] {
        for (column, values) in matrix.iter().enumerate() {
            floats[offset + column * 4..offset + column * 4 + 4].copy_from_slice(values);
        }
    }
    floats[32..36].copy_from_slice(&[to_light[0], to_light[1], to_light[2], settings.length]);
    floats[36..40].copy_from_slice(&[settings.thickness, 0.0, 0.0, 0.0]);
    floats[40..44].copy_from_slice(&[width, height, 1.0 / width, 1.0 / height]);
    let mut out = [0_u8; CONTACT_PARAMS_SIZE];
    for (chunk, value) in out.chunks_exact_mut(4).zip(floats) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    out
}

/// Compiles `source` into the march pipeline, returning the validation
/// message instead of panicking.
pub(crate) fn create_pipeline(device: &wgpu::Device, source: &str) -> Result<wgpu::ComputePipeline, String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(CONTACT_SHADOW_TRACE),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("contact-shadow-trace"),
        layout: None,
        module: &module,
        entry_point: Some("contact_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    match block_on(scope.pop()) {
        Some(error) => Err(error.to_string()),
        None => Ok(pipeline),
    }
}

/// The contact shadow result group 0 binds, and what produces it.
#[derive(Debug)]
pub(crate) struct ContactShadowResources {
    pub(crate) pipeline: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    /// The output's size.
    result: ScreenResult,
    /// Whether this frame traces, decided before recording.
    active: bool,
}

impl ContactShadowResources {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &str,
        output_size: (u32, u32),
    ) -> Result<Self, String> {
        let pipeline = create_pipeline(device, source)?;
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("contact-shadow-params"),
            size: CONTACT_PARAMS_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let result = ScreenResult::new(device, queue, "contact-shadows", CONTACT_FORMAT, output_size, LIT);
        Ok(Self {
            pipeline,
            params,
            result,
            active: false,
        })
    }

    /// Readies this frame: resizes the result with the output, uploads the
    /// parameters, and clears the result to "lit" when contact shadows are
    /// off and it holds anything else. Call before the surface is acquired.
    /// Returns whether the result was recreated, so group 0 must be too.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &RenderFrame,
        output_size: (u32, u32),
    ) -> bool {
        let recreated = self.result.resize(device, output_size);
        let settings = frame_settings(frame);
        self.active = settings.is_some();
        match settings {
            Some((settings, to_light)) => {
                let inverse = crate::uniforms::inverse(&frame.view_proj).unwrap_or(crate::uniforms::IDENTITY);
                queue.write_buffer(
                    &self.params,
                    0,
                    &params_bytes(&frame.view_proj, &inverse, to_light, &settings, output_size),
                );
                self.result.mark_written();
            }
            None => self.result.make_neutral(device, queue),
        }
        recreated
    }

    /// Whether this frame traces contact shadows, so declares their pass.
    pub(crate) fn active(&self) -> bool {
        self.active
    }

    /// The result group 0 binds.
    pub(crate) fn view(&self) -> &wgpu::TextureView {
        &self.result.view
    }

    /// The march, from depth and normals into the result.
    pub(crate) fn record(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
        normals: &wgpu::TextureView,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("contact-shadow-bind-group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(normals),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.result.view),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("contact-shadows"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (width, height) = self.result.size();
        pass.dispatch_workgroups(width.div_ceil(WORKGROUP), height.div_ceil(WORKGROUP), 1);
    }

    /// The result texture, for tests that read it back.
    #[cfg(test)]
    pub(crate) fn texture(&self) -> &wgpu::Texture {
        self.result.texture()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_parameters_land_where_the_shader_reads_them() {
        let mut view_proj = crate::uniforms::IDENTITY;
        view_proj[3] = [1.0, 2.0, 3.0, 1.0];
        let mut inverse = crate::uniforms::IDENTITY;
        inverse[3] = [5.0, 6.0, 7.0, 1.0];
        let settings = ContactShadows {
            length: 0.4,
            thickness: 0.15,
        };
        let bytes = params_bytes(&view_proj, &inverse, [0.0, 0.6, 0.8], &settings, (200, 100));
        let float = |index: usize| {
            f32::from_le_bytes([
                bytes[index * 4],
                bytes[index * 4 + 1],
                bytes[index * 4 + 2],
                bytes[index * 4 + 3],
            ])
        };
        assert_eq!(float(12), 1.0, "view_proj translation, column 3");
        assert_eq!(float(16 + 13), 6.0, "inv_view_proj translation, column 3");
        assert_eq!(
            [float(32), float(33), float(34), float(35)],
            [0.0, 0.6, 0.8, 0.4],
            "to the light, then the march length"
        );
        assert_eq!(float(36), 0.15, "thickness");
        assert_eq!(
            [float(40), float(41), float(42), float(43)],
            [200.0, 100.0, 0.005, 0.01]
        );
    }

    #[test]
    fn unusable_settings_and_cameras_trace_nothing() {
        let perspective =
            glam::camera::rh::proj::directx::perspective(1.0, 1.5, 0.1, 100.0).to_cols_array_2d();
        let mut frame = RenderFrame::new(perspective, crate::Color::BLACK);
        assert_eq!(frame_settings(&frame), None, "off by default");

        frame.contact_shadows = Some(ContactShadows::default());
        let (settings, _) = frame_settings(&frame).expect("on");
        assert_eq!(settings, ContactShadows::default());

        for (length, thickness) in [(0.0, 0.2), (f32::NAN, 0.2), (0.3, -1.0), (0.3, f32::INFINITY)] {
            frame.contact_shadows = Some(ContactShadows { length, thickness });
            assert_eq!(
                frame_settings(&frame),
                None,
                "length {length}, thickness {thickness}"
            );
        }

        frame.contact_shadows = Some(ContactShadows::default());
        frame.light.direction = [0.0, 2.0, 0.0];
        assert_eq!(
            frame_settings(&frame).map(|(_, to_light)| to_light),
            Some([0.0, 1.0, 0.0]),
            "the march runs along a unit direction, so its length is metres"
        );
        for direction in [[0.0; 3], [f32::NAN, 1.0, 0.0]] {
            frame.light.direction = direction;
            assert_eq!(frame_settings(&frame), None, "no direction: {direction:?}");
        }
        frame.light.direction = [0.0, 1.0, 0.0];

        frame.contact_shadows = Some(ContactShadows::default());
        frame.view_proj = crate::uniforms::IDENTITY;
        assert_eq!(frame_settings(&frame), None, "no perspective camera");
    }
}
