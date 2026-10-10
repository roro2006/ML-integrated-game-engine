//! The sky (D3): a per-frame sky-view LUT and the fullscreen pass that
//! draws it behind the scene, with the sun's disc.

use super::AtmosphereParams;
use super::luts::{AtmosphereLuts, LUT_FORMAT};
use crate::block_on::block_on;
use crate::frame::Sky;
use crate::shaders::ATMOSPHERE_SKY;

/// Sky-view LUT size: azimuth from the sun (x) by view zenith (y).
pub(crate) const SKY_VIEW_SIZE: (u32, u32) = (192, 108);
/// Ray-march steps per sky-view texel. Must match `atmosphere_sky.wgsl`.
#[cfg(test)]
pub(crate) const SKY_STEPS: u32 = 30;
/// `SkyFrame` in `atmosphere_sky.wgsl`: a matrix and three `vec4`s.
pub(crate) const SKY_FRAME_SIZE: usize = 112;

/// The lowest the camera is taken to be, a metre above the ground: at the
/// ground itself the horizon is singular.
const MIN_ALTITUDE_KM: f32 = 0.001;
const METRES_PER_KM: f32 = 1000.0;
const WORKGROUP: u32 = 8;

/// The camera's distance from the planet's centre, in kilometres, for a
/// world height in metres. The world's y = 0 is the planet's surface.
pub(crate) fn camera_radius(params: &AtmosphereParams, height_metres: f32) -> f32 {
    params.bottom_radius + (height_metres / METRES_PER_KM).max(MIN_ALTITUDE_KM)
}

/// The `SkyFrame` block for one frame.
pub(crate) fn sky_frame_bytes(
    inverse_view_proj: &[[f32; 4]; 4],
    sky: &Sky,
    camera_radius: f32,
) -> [u8; SKY_FRAME_SIZE] {
    let sun = glam::Vec3::from(sky.sun_direction)
        .try_normalize()
        .unwrap_or(glam::Vec3::Y);
    let mut words = [0.0_f32; SKY_FRAME_SIZE / 4];
    for (column, values) in inverse_view_proj.iter().enumerate() {
        words[column * 4..column * 4 + 4].copy_from_slice(values);
    }
    words[16..20].copy_from_slice(&[sun.x, sun.y, sun.z, sky.sun_angular_radius.cos()]);
    words[20..23].copy_from_slice(&sky.sun_illuminance);
    words[24] = camera_radius;
    let mut out = [0_u8; SKY_FRAME_SIZE];
    for (chunk, word) in out.chunks_exact_mut(4).zip(words) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    out
}

/// The sky-view LUT, its per-frame uniform, and both pipelines.
#[derive(Debug)]
pub(crate) struct SkyResources {
    pub(crate) frame_buffer: wgpu::Buffer,
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "read back by the GPU tests; the views keep it alive")
    )]
    pub(crate) sky_view: wgpu::Texture,
    sky_view_view: wgpu::TextureView,
    view_pipeline: wgpu::ComputePipeline,
    draw_pipeline: wgpu::RenderPipeline,
    view_group: wgpu::BindGroup,
    draw_group: wgpu::BindGroup,
}

impl SkyResources {
    /// Compiles `source` (`atmosphere_sky.wgsl`, includes expanded) against
    /// `luts`. The sky draws into `color_format` with `depth_format` loaded.
    pub(crate) fn new(
        device: &wgpu::Device,
        source: &str,
        luts: &AtmosphereLuts,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Result<Self, String> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(ATMOSPHERE_SKY),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let view_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sky-view"),
            layout: None,
            module: &module,
            entry_point: Some("sky_view_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let draw_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("sky_vs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            // On the far plane, behind everything the scene drew.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("sky_fs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(color_format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        if let Some(error) = block_on(scope.pop()) {
            return Err(error.to_string());
        }

        let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky-frame"),
            size: SKY_FRAME_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (width, height) = SKY_VIEW_SIZE;
        let sky_view = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sky-view"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: LUT_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let sky_view_view = sky_view.create_view(&wgpu::TextureViewDescriptor::default());
        let (view_group, draw_group) = bind_groups(
            device,
            &view_pipeline,
            &draw_pipeline,
            &frame_buffer,
            &sky_view_view,
            luts,
        );
        Ok(Self {
            frame_buffer,
            sky_view,
            sky_view_view,
            view_pipeline,
            draw_pipeline,
            view_group,
            draw_group,
        })
    }

    /// Rebinds to new LUTs, after the atmosphere is rebuilt.
    pub(crate) fn rebind(&mut self, device: &wgpu::Device, luts: &AtmosphereLuts) {
        (self.view_group, self.draw_group) = bind_groups(
            device,
            &self.view_pipeline,
            &self.draw_pipeline,
            &self.frame_buffer,
            &self.sky_view_view,
            luts,
        );
    }

    /// Records the sky-view LUT for this frame.
    pub(crate) fn record_view(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sky-view"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.view_pipeline);
        pass.set_bind_group(0, &self.view_group, &[]);
        let (width, height) = SKY_VIEW_SIZE;
        pass.dispatch_workgroups(width.div_ceil(WORKGROUP), height.div_ceil(WORKGROUP), 1);
    }

    /// Draws the sky into `pass`, which has the scene colour and depth bound.
    pub(crate) fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.draw_pipeline);
        pass.set_bind_group(0, &self.draw_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn bind_groups(
    device: &wgpu::Device,
    view_pipeline: &wgpu::ComputePipeline,
    draw_pipeline: &wgpu::RenderPipeline,
    frame_buffer: &wgpu::Buffer,
    sky_view: &wgpu::TextureView,
    luts: &AtmosphereLuts,
) -> (wgpu::BindGroup, wgpu::BindGroup) {
    let view_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sky-view-bind-group"),
        layout: &view_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: luts.params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: frame_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&luts.transmittance_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&luts.multi_scattering_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&luts.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(sky_view),
            },
        ],
    });
    let draw_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sky-bind-group"),
        layout: &draw_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: luts.params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: frame_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&luts.transmittance_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&luts.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(sky_view),
            },
        ],
    });
    (view_group, draw_group)
}

/// CPU mirror of the sky-view parameterisation, for tests.
#[cfg(test)]
pub(crate) mod mirror {
    use super::super::AtmosphereParams;

    fn horizon_zenith(params: &AtmosphereParams, r: f64) -> (f64, f64) {
        let bottom = f64::from(params.bottom_radius);
        let beta = ((r * r - bottom * bottom).max(0.0).sqrt() / r)
            .clamp(-1.0, 1.0)
            .acos();
        (std::f64::consts::PI - beta, beta)
    }

    /// LUT coordinate to (view zenith cosine, cosine of azimuth from sun).
    pub(crate) fn angles(params: &AtmosphereParams, u: f64, v: f64, r: f64) -> (f64, f64) {
        let (horizon, beta) = horizon_zenith(params, r);
        let zenith = if v < 0.5 {
            let coord = 1.0 - 2.0 * v;
            horizon * (1.0 - coord * coord)
        } else {
            let coord = v * 2.0 - 1.0;
            horizon + beta * coord * coord
        };
        (zenith.cos(), -(u * u * 2.0 - 1.0))
    }

    pub(crate) fn uv(params: &AtmosphereParams, r: f64, cos_zenith: f64, cos_azimuth: f64) -> (f64, f64) {
        let (horizon, beta) = horizon_zenith(params, r);
        let zenith = cos_zenith.clamp(-1.0, 1.0).acos();
        let v = if zenith < horizon {
            (1.0 - (1.0 - zenith / horizon).max(0.0).sqrt()) * 0.5
        } else {
            ((zenith - horizon) / beta).clamp(0.0, 1.0).sqrt() * 0.5 + 0.5
        };
        ((-cos_azimuth * 0.5 + 0.5).clamp(0.0, 1.0).sqrt(), v)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{MULTI_SCATTERING_SIZE, distance_to_ground, distance_to_top, medium, reference};
    use super::*;

    #[test]
    fn the_sky_view_mapping_round_trips_above_and_below_the_horizon() {
        let params = AtmosphereParams::default();
        let r = 6360.5;
        for &(u, v) in &[(0.1, 0.1), (0.5, 0.45), (0.9, 0.55), (0.3, 0.9), (0.7, 0.49)] {
            let (cos_zenith, cos_azimuth) = mirror::angles(&params, u, v, r);
            let (u2, v2) = mirror::uv(&params, r, cos_zenith, cos_azimuth);
            assert!(
                (u - u2).abs() < 1e-9 && (v - v2).abs() < 1e-9,
                "({u}, {v}) -> ({u2}, {v2})"
            );
        }
    }

    #[test]
    fn the_middle_row_is_the_limb_and_the_left_column_faces_the_sun() {
        let params = AtmosphereParams::default();
        let r = 6361.0;
        let bottom = f64::from(params.bottom_radius);
        let limb = -((r * r - bottom * bottom).sqrt() / r);
        let (cos_zenith, _) = mirror::angles(&params, 0.5, 0.5, r);
        assert!((cos_zenith - limb).abs() < 1e-9, "{cos_zenith} against {limb}");
        let (_, toward_sun) = mirror::angles(&params, 0.0, 0.3, r);
        assert!((toward_sun - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_frame_block_puts_the_sun_and_camera_where_the_shader_reads_them() {
        let sky = Sky::default();
        let bytes = sky_frame_bytes(&crate::uniforms::IDENTITY, &sky, 6360.25);
        let word = |index: usize| {
            let mut four = [0_u8; 4];
            four.copy_from_slice(&bytes[index * 4..index * 4 + 4]);
            f32::from_le_bytes(four)
        };
        assert_eq!(word(0), 1.0, "the matrix leads");
        assert!((word(19) - sky.sun_angular_radius.cos()).abs() < 1e-7);
        assert_eq!(word(20), sky.sun_illuminance[0]);
        assert_eq!(word(24), 6360.25);
        let sun_length = (word(16).powi(2) + word(17).powi(2) + word(18).powi(2)).sqrt();
        assert!(
            (sun_length - 1.0).abs() < 1e-6,
            "the sun direction is unit length"
        );
    }

    #[test]
    fn the_camera_never_sits_below_a_metre() {
        let params = AtmosphereParams::default();
        assert_eq!(camera_radius(&params, -50.0), 6360.001);
        assert_eq!(camera_radius(&params, 2000.0), 6362.0);
    }

    /// CPU single scattering plus multiple scattering read from `table`
    /// (the GPU's, bilinearly, as the shader samples it), mirroring
    /// `integrate_sky`.
    fn reference_sky(
        params: &AtmosphereParams,
        table: &[[f32; 3]],
        r: f64,
        view: [f64; 3],
        sun: [f64; 3],
    ) -> [f64; 3] {
        let bottom = f64::from(params.bottom_radius);
        let top = f64::from(params.top_radius);
        let pi = std::f64::consts::PI;
        let multi = |height: f64, mu_sun: f64| -> [f64; 3] {
            let (width, rows) = MULTI_SCATTERING_SIZE;
            let x = ((mu_sun * 0.5 + 0.5).clamp(0.0, 1.0) * f64::from(width) - 0.5)
                .clamp(0.0, f64::from(width - 1));
            let y = (((height - bottom) / (top - bottom)).clamp(0.0, 1.0) * f64::from(rows) - 0.5)
                .clamp(0.0, f64::from(rows - 1));
            let (x0, y0) = (x.floor() as u32, y.floor() as u32);
            let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(rows - 1));
            let (fx, fy) = (x - f64::from(x0), y - f64::from(y0));
            let at = |px: u32, py: u32, channel: usize| f64::from(table[(py * width + px) as usize][channel]);
            std::array::from_fn(|channel| {
                let upper = at(x0, y0, channel) * (1.0 - fx) + at(x1, y0, channel) * fx;
                let lower = at(x0, y1, channel) * (1.0 - fx) + at(x1, y1, channel) * fx;
                upper * (1.0 - fy) + lower * fy
            })
        };
        let toward = |height: f64, mu: f64| -> [f64; 3] {
            let horizon = -(1.0 - (bottom / height).powi(2)).max(0.0).sqrt();
            if mu < horizon {
                [0.0; 3]
            } else {
                reference::transmittance(params, height, mu, 256)
            }
        };
        let ground = distance_to_ground(params, r, view[1]);
        let length = ground.unwrap_or_else(|| distance_to_top(params, r, view[1]));
        let cos_angle = view[0] * sun[0] + view[1] * sun[1] + view[2] * sun[2];
        let phase_rayleigh = 3.0 / (16.0 * pi) * (1.0 + cos_angle * cos_angle);
        let g = f64::from(params.mie_g);
        let phase_mie = 3.0 / (8.0 * pi) * (1.0 - g * g) * (1.0 + cos_angle * cos_angle)
            / ((2.0 + g * g) * (1.0 + g * g - 2.0 * g * cos_angle).max(1e-6).powf(1.5));
        let dt = length / f64::from(SKY_STEPS);
        let mut luminance = [0.0; 3];
        let mut throughput = [1.0; 3];
        for step in 0..SKY_STEPS {
            let t = (f64::from(step) + 0.5) * dt;
            let point = [view[0] * t, r + view[1] * t, view[2] * t];
            let height = (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
            let mu_sun = (point[0] * sun[0] + point[1] * sun[1] + point[2] * sun[2]) / height;
            let local = medium(params, height - bottom);
            let sunlight = toward(height, mu_sun);
            let psi = multi(height, mu_sun);
            for channel in 0..3 {
                let single = sunlight[channel]
                    * (local.rayleigh_scattering[channel] * phase_rayleigh
                        + local.mie_scattering * phase_mie);
                let multiple = (local.rayleigh_scattering[channel] + local.mie_scattering) * psi[channel];
                let extinction = local.extinction[channel];
                let step_transmittance = (-extinction * dt).exp();
                let fraction = (1.0 - step_transmittance) / extinction.max(1e-12);
                luminance[channel] += throughput[channel] * (single + multiple) * fraction;
                throughput[channel] *= step_transmittance;
            }
        }
        luminance
    }

    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn the_sky_view_lut_matches_the_cpu_reference_at_noon_and_sunset() {
        use crate::{Color, RenderFrame, Renderer, RendererConfig};
        let params = AtmosphereParams::default();
        for sun_elevation in [1.2_f32, 0.02] {
            let mut renderer = Renderer::headless(32, 32, RendererConfig::default()).expect("headless");
            let mut frame = RenderFrame::new(crate::uniforms::IDENTITY, Color::BLACK);
            let sun = [sun_elevation.cos(), sun_elevation.sin(), 0.0];
            frame.sky = Some(Sky {
                sun_direction: sun,
                ..Sky::default()
            });
            renderer.render_to_image(&frame).expect("renders");
            let sky = &renderer.sky;
            let lut = super::super::luts::tests::read_lut(&renderer, &sky.sky_view, SKY_VIEW_SIZE);
            let multi = super::super::luts::tests::read_lut(
                &renderer,
                &renderer.atmosphere.multi_scattering,
                MULTI_SCATTERING_SIZE,
            );
            let r = f64::from(camera_radius(&params, 0.0));
            let sun = sun.map(f64::from);
            let (width, height) = SKY_VIEW_SIZE;
            for &(x, y) in &[(10, 20), (96, 40), (180, 52), (5, 53), (100, 80)] {
                let u = (f64::from(x) + 0.5) / f64::from(width);
                let v = (f64::from(y) + 0.5) / f64::from(height);
                let (cos_zenith, cos_azimuth) = mirror::angles(&params, u, v, r);
                let sin_zenith = (1.0 - cos_zenith * cos_zenith).max(0.0).sqrt();
                let sin_azimuth = (1.0 - cos_azimuth * cos_azimuth).max(0.0).sqrt();
                let view = [sin_zenith * cos_azimuth, cos_zenith, sin_zenith * sin_azimuth];
                let cpu = reference_sky(&params, &multi, r, view, sun);
                let gpu = lut[(y * width + x) as usize];
                for channel in 0..3 {
                    let error = (f64::from(gpu[channel]) - cpu[channel]).abs() / cpu[channel].abs().max(1e-4);
                    assert!(
                        error < 0.03,
                        "sun {sun_elevation}, texel ({x}, {y}) channel {channel}: gpu {} cpu {}",
                        gpu[channel],
                        cpu[channel]
                    );
                }
            }
        }
    }
}
