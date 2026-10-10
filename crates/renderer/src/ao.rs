//! Ground-truth ambient occlusion (C4): a half-resolution trace against the
//! depth prepass and its normal target, a depth-aware spatial denoise, and
//! the persistent result the lit shaders read through group 0.
//!
//! The result lives outside the render graph's pool, like the shadow atlas,
//! because group 0 binds it for the whole renderer's life. With occlusion
//! off the trace and denoise are not declared at all, and the result is
//! cleared once to "fully open", so ambient light is exactly as before and
//! the frame pays nothing.
//!
//! `gtao.wgsl` holds the method; [`side_visibility`] and [`open_slice`] are
//! the CPU mirrors of its slice integral that the tests hold it to.

use crate::block_on::block_on;
use crate::frame::{AmbientOcclusion, RenderFrame};
use crate::graph::TransientDesc;
use crate::shaders::GTAO;

/// The format of both occlusion targets: visibility, and the traced point's
/// distance from the camera for the depth-aware filters. `rgba16float` is
/// the smallest float format every device can write as a storage texture.
pub(crate) const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Bytes of `AoParams`.
pub(crate) const AO_PARAMS_SIZE: usize = 112;
const WORKGROUP: u32 = 8;
/// The largest radius traced on screen, as a fraction of the output height.
/// Close to the camera a metre covers much of the screen; past this the
/// samples spread too thin to find small creases, and stop hitting cache.
const MAX_RADIUS_SCREEN_FRACTION: f32 = 0.1;

/// The occlusion targets' size for an output of `width` by `height`: half,
/// rounded up so the last column and row of pixels still have a texel.
pub(crate) fn half_size(width: u32, height: u32) -> (u32, u32) {
    (width.div_ceil(2).max(1), height.div_ceil(2).max(1))
}

/// The raw trace's descriptor: written by the trace, read by the denoise.
pub(crate) fn raw_desc(width: u32, height: u32) -> TransientDesc {
    let (width, height) = half_size(width, height);
    TransientDesc {
        format: AO_FORMAT,
        width,
        height,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
    }
}

/// This frame's occlusion settings, or `None` when it traces none: turned
/// off, a setting that would do nothing or is not finite, or a camera that
/// is not a perspective one, which the trace cannot reconstruct.
pub(crate) fn frame_settings(frame: &RenderFrame) -> Option<AmbientOcclusion> {
    let settings = frame.ambient_occlusion?;
    let usable = settings.radius.is_finite()
        && settings.radius > 0.0
        && settings.strength.is_finite()
        && settings.strength > 0.0;
    (usable && crate::shadow::view_depths(&frame.view_proj).is_some()).then_some(AmbientOcclusion {
        strength: settings.strength.min(1.0),
        ..settings
    })
}

/// `AoParams` for one frame. `size` is the full output size.
pub(crate) fn params_bytes(
    inverse_view_proj: &[[f32; 4]; 4],
    eye: [f32; 3],
    settings: &AmbientOcclusion,
    size: (u32, u32),
) -> [u8; AO_PARAMS_SIZE] {
    let (width, height) = (size.0.max(1) as f32, size.1.max(1) as f32);
    let mut floats = [0.0_f32; AO_PARAMS_SIZE / 4];
    for (column, values) in inverse_view_proj.iter().enumerate() {
        floats[column * 4..column * 4 + 4].copy_from_slice(values);
    }
    floats[16..20].copy_from_slice(&[eye[0], eye[1], eye[2], settings.radius]);
    floats[20..24].copy_from_slice(&[width, height, 1.0 / width, 1.0 / height]);
    floats[24..28].copy_from_slice(&[settings.strength, height * MAX_RADIUS_SCREEN_FRACTION, 0.0, 0.0]);
    let mut out = [0_u8; AO_PARAMS_SIZE];
    for (chunk, value) in out.chunks_exact_mut(4).zip(floats) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    out
}

/// Cosine-weighted visibility of one side of a slice, from the normal's
/// angle `n` to the horizon `h`, both from the view vector. Mirror of
/// `side_visibility` in `gtao.wgsl`.
#[cfg(test)]
pub(crate) fn side_visibility(h: f32, n: f32) -> f32 {
    0.25 * (-(2.0 * h - n).cos() + n.cos() + 2.0 * h * n.sin())
}

/// What a slice integrates to with both horizons at the tangent. Mirror of
/// `open_slice` in `gtao.wgsl`.
#[cfg(test)]
pub(crate) fn open_slice(n: f32) -> f32 {
    n.cos() + n * n.sin()
}

/// The trace and denoise pipelines, built together so a hot reload installs
/// both or neither.
#[derive(Debug)]
pub(crate) struct AoPipelines {
    trace: wgpu::ComputePipeline,
    denoise: wgpu::ComputePipeline,
}

/// Compiles `source` into the trace and denoise pipelines, returning the
/// validation message instead of panicking.
pub(crate) fn create_pipelines(device: &wgpu::Device, source: &str) -> Result<AoPipelines, String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(GTAO),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = |label, entry_point| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            // Each entry point's own bindings: the two share no layout.
            layout: None,
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    };
    let pipelines = AoPipelines {
        trace: pipeline("gtao-trace", "gtao_main"),
        denoise: pipeline("gtao-denoise", "denoise_main"),
    };
    match block_on(scope.pop()) {
        Some(error) => Err(error.to_string()),
        None => Ok(pipelines),
    }
}

/// The occlusion result group 0 binds, and what produces it.
#[derive(Debug)]
pub(crate) struct AmbientOcclusionResources {
    pub(crate) pipelines: AoPipelines,
    params: wgpu::Buffer,
    texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    /// The result's size: half the output's.
    size: (u32, u32),
    /// Whether the result holds "fully open" rather than a trace.
    open: bool,
    /// Whether this frame traces, decided before recording.
    active: bool,
}

impl AmbientOcclusionResources {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &str,
        output_size: (u32, u32),
    ) -> Result<Self, String> {
        let pipelines = create_pipelines(device, source)?;
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ao-params"),
            size: AO_PARAMS_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let size = result_size(device, output_size);
        let (texture, view) = result_texture(device, size);
        let resources = Self {
            pipelines,
            params,
            texture,
            view,
            size,
            open: true,
            active: false,
        };
        resources.clear(device, queue);
        Ok(resources)
    }

    /// Readies this frame: resizes the result with the output, uploads the
    /// parameters, and clears the result to "fully open" when occlusion has
    /// just been turned off. Call before the surface is acquired. Returns
    /// whether the result was recreated, so group 0 must be too.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &RenderFrame,
        output_size: (u32, u32),
    ) -> bool {
        let size = result_size(device, output_size);
        let recreated = size != self.size;
        if recreated {
            (self.texture, self.view) = result_texture(device, size);
            self.size = size;
            self.open = false;
        }
        let settings = frame_settings(frame);
        self.active = settings.is_some();
        match settings {
            Some(settings) => {
                let inverse = crate::uniforms::inverse(&frame.view_proj).unwrap_or(crate::uniforms::IDENTITY);
                queue.write_buffer(
                    &self.params,
                    0,
                    &params_bytes(&inverse, frame.camera_position, &settings, output_size),
                );
                self.open = false;
            }
            None if !self.open => {
                self.clear(device, queue);
                self.open = true;
            }
            None => {}
        }
        recreated
    }

    /// Whether this frame traces occlusion, so declares its passes.
    pub(crate) fn active(&self) -> bool {
        self.active
    }

    /// The half-resolution trace, from depth and normals into `raw`.
    pub(crate) fn record_trace(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
        normals: &wgpu::TextureView,
        raw: &wgpu::TextureView,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gtao-trace-bind-group"),
            layout: &self.pipelines.trace.get_bind_group_layout(0),
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
                    resource: wgpu::BindingResource::TextureView(raw),
                },
            ],
        });
        self.dispatch(encoder, "ao", &self.pipelines.trace, &bind_group);
    }

    /// The denoise, from `raw` into the result group 0 binds.
    pub(crate) fn record_denoise(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        raw: &wgpu::TextureView,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gtao-denoise-bind-group"),
            layout: &self.pipelines.denoise.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(raw),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.view),
                },
            ],
        });
        self.dispatch(encoder, "ao-denoise", &self.pipelines.denoise, &bind_group);
    }

    fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(
            self.size.0.div_ceil(WORKGROUP),
            self.size.1.div_ceil(WORKGROUP),
            1,
        );
    }

    /// Fills the result with visibility 1 at distance 0, which the lit
    /// shaders read as "no occlusion" (ambient_occlusion.wgsl). Its own small
    /// submission: this happens on a toggle or a resize, never per frame.
    fn clear(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ao-clear"),
        });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ao-clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
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
    }

    /// The result texture, for tests that read it back.
    #[cfg(test)]
    pub(crate) fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }
}

/// The result's size for an output size, within what the device can create:
/// an output too large to draw skips its frames, but the result must exist.
fn result_size(device: &wgpu::Device, (width, height): (u32, u32)) -> (u32, u32) {
    let max = device.limits().max_texture_dimension_2d;
    let (width, height) = half_size(width, height);
    (width.min(max), height.min(max))
}

fn result_texture(device: &wgpu::Device, (width, height): (u32, u32)) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ambient-occlusion"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: AO_FORMAT,
        // Written by the denoise, read by the lit shaders, cleared as an
        // attachment when occlusion is off, and copied out by tests.
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

#[cfg(test)]
mod tests {
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

    use super::*;

    #[test]
    fn the_targets_are_half_the_output_rounded_up_and_never_empty() {
        assert_eq!(half_size(1920, 1080), (960, 540));
        assert_eq!(half_size(1281, 721), (641, 361));
        assert_eq!(half_size(1, 1), (1, 1));
        assert_eq!(
            half_size(0, 0),
            (1, 1),
            "a minimized window still has a texture to bind"
        );
        let raw = raw_desc(1921, 1080);
        assert_eq!((raw.width, raw.height), (961, 540));
        assert_eq!(raw.format, AO_FORMAT);
        assert!(raw.usage.contains(wgpu::TextureUsages::STORAGE_BINDING));
        assert!(raw.usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
    }

    #[test]
    fn an_open_slice_divides_to_exactly_one_whatever_the_normal() {
        for n in [-1.2, -0.5, 0.0, 0.3, 1.0_f32] {
            let open = side_visibility(n + FRAC_PI_2, n) + side_visibility(n - FRAC_PI_2, n);
            assert!((open / open_slice(n) - 1.0).abs() < 1e-5, "n = {n}: {open}");
        }
        assert!(
            (open_slice(0.0) - 1.0).abs() < 1e-6,
            "facing the viewer, a slice is the whole of it"
        );
    }

    /// The paper's outer integral: over every slice angle, the projected
    /// normal's length times its open slice gives pi, the cosine-weighted
    /// hemisphere's normalisation. Holds for a surface seen at any angle.
    #[test]
    fn the_open_slices_of_a_tilted_surface_integrate_to_the_hemisphere() {
        let view = glam::Vec3::Z;
        for tilt in [0.0, 0.4, 1.0, 1.4_f32] {
            let normal = glam::Vec3::new(tilt.sin(), 0.0, tilt.cos());
            let steps = 4096;
            let total: f32 = (0..steps)
                .map(|step| {
                    let phi = (step as f32 + 0.5) / steps as f32 * std::f32::consts::PI;
                    let direction = glam::Vec3::new(phi.cos(), phi.sin(), 0.0);
                    let axis = direction.cross(view);
                    let projected = normal - axis * normal.dot(axis);
                    let length = projected.length();
                    let n = projected.dot(direction).signum()
                        * (projected.dot(view) / length).clamp(-1.0, 1.0).acos();
                    length * open_slice(n) * std::f32::consts::PI / steps as f32
                })
                .sum();
            assert!(
                (total / std::f32::consts::PI - 1.0).abs() < 1e-3,
                "tilt {tilt}: {total}"
            );
        }
    }

    #[test]
    fn a_horizon_at_the_normal_hides_its_half_of_the_slice() {
        // A wall rising to the normal on one side, seen across the corner
        // it makes with the floor.
        let blocked = side_visibility(0.0, 0.0) + side_visibility(-FRAC_PI_2, 0.0);
        assert!((blocked - 0.5).abs() < 1e-5, "{blocked}");
        // Halfway down that side, cosine weighting leaves a quarter of it:
        // the part near the normal counts most.
        let halfway = side_visibility(FRAC_PI_4, 0.0);
        assert!((halfway - 0.25).abs() < 1e-5, "{halfway}");
        // A horizon at the normal on both sides is a closed pit.
        assert!(side_visibility(0.0, 0.0).abs() < 1e-6);
    }

    #[test]
    fn the_parameters_land_where_the_shader_reads_them() {
        let mut inverse = crate::uniforms::IDENTITY;
        inverse[3] = [5.0, 6.0, 7.0, 1.0];
        let settings = AmbientOcclusion {
            radius: 0.75,
            strength: 0.5,
        };
        let bytes = params_bytes(&inverse, [1.0, 2.0, 3.0], &settings, (200, 100));
        let float = |index: usize| {
            f32::from_le_bytes([
                bytes[index * 4],
                bytes[index * 4 + 1],
                bytes[index * 4 + 2],
                bytes[index * 4 + 3],
            ])
        };
        assert_eq!(float(12), 5.0, "inv_view_proj translation, column 3");
        assert_eq!(
            [float(16), float(17), float(18), float(19)],
            [1.0, 2.0, 3.0, 0.75]
        );
        assert_eq!(
            [float(20), float(21), float(22), float(23)],
            [200.0, 100.0, 0.005, 0.01]
        );
        assert_eq!(float(24), 0.5, "strength");
        assert!((float(25) - 10.0).abs() < 1e-5, "a tenth of the height");
    }

    #[test]
    fn unusable_settings_and_cameras_trace_nothing() {
        let perspective =
            glam::camera::rh::proj::directx::perspective(1.0, 1.5, 0.1, 100.0).to_cols_array_2d();
        let mut frame = RenderFrame::new(perspective, crate::Color::BLACK);
        assert_eq!(frame_settings(&frame), None, "off by default");

        frame.ambient_occlusion = Some(AmbientOcclusion::default());
        assert_eq!(frame_settings(&frame), Some(AmbientOcclusion::default()));

        for (radius, strength) in [(0.0, 1.0), (f32::NAN, 1.0), (1.0, 0.0), (1.0, f32::INFINITY)] {
            frame.ambient_occlusion = Some(AmbientOcclusion { radius, strength });
            assert_eq!(
                frame_settings(&frame),
                None,
                "radius {radius}, strength {strength}"
            );
        }

        frame.ambient_occlusion = Some(AmbientOcclusion {
            radius: 1.0,
            strength: 4.0,
        });
        assert_eq!(
            frame_settings(&frame).map(|settings| settings.strength),
            Some(1.0)
        );

        frame.view_proj = crate::uniforms::IDENTITY;
        assert_eq!(frame_settings(&frame), None, "no perspective camera");
    }
}
