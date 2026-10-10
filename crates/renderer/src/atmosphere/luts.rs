//! The atmosphere's lookup tables on the GPU (D3): transmittance and
//! multiple scattering, baked in compute whenever the parameters change.

use super::{AtmosphereParams, MULTI_SCATTERING_SIZE, PARAMS_SIZE, TRANSMITTANCE_SIZE};
use crate::block_on::block_on;
use crate::shaders::ATMOSPHERE_LUTS;

/// Half floats: transmittance lies in 0..1 and the multiple-scattering
/// transfer is small, so 16 bits keep both well inside the 1 percent the
/// tests hold them to, at half the bandwidth of 32.
pub(crate) const LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

const WORKGROUP: u32 = 8;

fn lut(device: &wgpu::Device, label: &str, (width, height): (u32, u32)) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
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
    })
}

/// The baked tables and what bakes them.
#[derive(Debug)]
pub(crate) struct AtmosphereLuts {
    pub(crate) params: AtmosphereParams,
    pub(crate) params_buffer: wgpu::Buffer,
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "read back by the GPU tests; the views keep it alive")
    )]
    pub(crate) transmittance: wgpu::Texture,
    pub(crate) transmittance_view: wgpu::TextureView,
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "read back by the GPU tests; the views keep it alive")
    )]
    pub(crate) multi_scattering: wgpu::Texture,
    pub(crate) multi_scattering_view: wgpu::TextureView,
    /// Linear and clamped, for every lookup into these tables.
    pub(crate) sampler: wgpu::Sampler,
    transmittance_pipeline: wgpu::ComputePipeline,
    transmittance_group: wgpu::BindGroup,
    multi_scattering_pipeline: wgpu::ComputePipeline,
    multi_scattering_group: wgpu::BindGroup,
}

impl AtmosphereLuts {
    /// Compiles `source` (`atmosphere_luts.wgsl`, includes expanded) and
    /// allocates the tables. Call [`bake`](Self::bake) before reading them.
    pub(crate) fn new(device: &wgpu::Device, source: &str, params: AtmosphereParams) -> Result<Self, String> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(ATMOSPHERE_LUTS),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let compute = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                // Layouts come from the shader: each entry point binds only
                // what it uses, which keeps the transmittance table out of
                // the pass that writes it.
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let transmittance_pipeline = compute("transmittance_main");
        let multi_scattering_pipeline = compute("multi_scattering_main");
        if let Some(error) = block_on(scope.pop()) {
            return Err(error.to_string());
        }

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atmosphere-params"),
            size: PARAMS_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let transmittance = lut(device, "atmosphere-transmittance", TRANSMITTANCE_SIZE);
        let multi_scattering = lut(device, "atmosphere-multi-scattering", MULTI_SCATTERING_SIZE);
        let transmittance_view = transmittance.create_view(&wgpu::TextureViewDescriptor::default());
        let multi_scattering_view = multi_scattering.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atmosphere-lut-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..wgpu::SamplerDescriptor::default()
        });

        let transmittance_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atmosphere-transmittance-bake"),
            layout: &transmittance_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&transmittance_view),
                },
            ],
        });
        let multi_scattering_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atmosphere-multi-scattering-bake"),
            layout: &multi_scattering_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&transmittance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&multi_scattering_view),
                },
            ],
        });

        Ok(Self {
            params,
            params_buffer,
            transmittance,
            transmittance_view,
            multi_scattering,
            multi_scattering_view,
            sampler,
            transmittance_pipeline,
            transmittance_group,
            multi_scattering_pipeline,
            multi_scattering_group,
        })
    }

    /// Writes the parameters and records both bakes into `encoder`,
    /// transmittance first.
    pub(crate) fn bake(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        queue.write_buffer(&self.params_buffer, 0, &self.params.bytes());
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("atmosphere-luts"),
            timestamp_writes: None,
        });
        for (pipeline, group, (width, height)) in [
            (
                &self.transmittance_pipeline,
                &self.transmittance_group,
                TRANSMITTANCE_SIZE,
            ),
            (
                &self.multi_scattering_pipeline,
                &self.multi_scattering_group,
                MULTI_SCATTERING_SIZE,
            ),
        ] {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(width.div_ceil(WORKGROUP), height.div_ceil(WORKGROUP), 1);
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::atmosphere::reference;
    use crate::atmosphere::transmittance_uv_to_r_mu;
    use crate::{Renderer, RendererConfig};

    /// IEEE half to single, for reading back `Rgba16Float`.
    pub(crate) fn half_to_f32(bits: u16) -> f32 {
        let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
        let exponent = i32::from((bits >> 10) & 0x1f);
        let mantissa = f32::from(bits & 0x3ff);
        match exponent {
            0 => sign * mantissa * 2f32.powi(-24),
            31 => sign * f32::INFINITY,
            _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
        }
    }

    /// Reads an `Rgba16Float` texture back as rows of RGB.
    pub(crate) fn read_lut(
        renderer: &Renderer,
        texture: &wgpu::Texture,
        (width, height): (u32, u32),
    ) -> Vec<[f32; 3]> {
        let bytes_per_row =
            (width * 8).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        renderer.queue.submit(std::iter::once(encoder.finish()));
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        renderer
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the device finishes");
        receiver.recv().expect("map callback").expect("maps");
        let mapped = buffer.get_mapped_range(..).expect("mapped");
        mapped
            .chunks_exact(bytes_per_row as usize)
            .flat_map(|row| {
                row[..width as usize * 8].chunks_exact(8).map(|texel| {
                    std::array::from_fn(|channel| {
                        half_to_f32(u16::from_le_bytes([texel[channel * 2], texel[channel * 2 + 1]]))
                    })
                })
            })
            .collect()
    }

    fn relative_error(gpu: f32, cpu: f64) -> f64 {
        (f64::from(gpu) - cpu).abs() / cpu.abs().max(1e-6)
    }

    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn the_baked_transmittance_matches_the_cpu_reference_within_one_percent() {
        let renderer = Renderer::headless(16, 16, RendererConfig::default()).expect("headless");
        let luts = &renderer.atmosphere;
        let table = read_lut(&renderer, &luts.transmittance, TRANSMITTANCE_SIZE);
        let (width, height) = TRANSMITTANCE_SIZE;
        for &(x, y) in &[
            (0, 0),
            (40, 3),
            (128, 32),
            (250, 10),
            (255, 63),
            (200, 60),
            (5, 50),
        ] {
            let u = (f64::from(x) + 0.5) / f64::from(width);
            let v = (f64::from(y) + 0.5) / f64::from(height);
            let (r, mu) = transmittance_uv_to_r_mu(&luts.params, u, v);
            let cpu = reference::transmittance(&luts.params, r, mu, 4096);
            let gpu = table[(y * width + x) as usize];
            for channel in 0..3 {
                // Values near zero (grazing rays at the ground) are held in
                // absolute terms; the rest relatively.
                let error = if cpu[channel] < 0.01 {
                    (f64::from(gpu[channel]) - cpu[channel]).abs() * 100.0
                } else {
                    relative_error(gpu[channel], cpu[channel])
                };
                assert!(
                    error < 0.01,
                    "texel ({x}, {y}) channel {channel}: gpu {} cpu {}",
                    gpu[channel],
                    cpu[channel]
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn the_baked_multiple_scattering_matches_the_cpu_reference() {
        let renderer = Renderer::headless(16, 16, RendererConfig::default()).expect("headless");
        let luts = &renderer.atmosphere;
        let table = read_lut(&renderer, &luts.multi_scattering, MULTI_SCATTERING_SIZE);
        let (width, height) = MULTI_SCATTERING_SIZE;
        let bottom = f64::from(luts.params.bottom_radius);
        let top = f64::from(luts.params.top_radius);
        for &(x, y) in &[(31, 0), (24, 2), (16, 0), (31, 20), (20, 31)] {
            let mu_sun = (f64::from(x) + 0.5) / f64::from(width) * 2.0 - 1.0;
            let r = bottom + (f64::from(y) + 0.5) / f64::from(height) * (top - bottom);
            let cpu = reference::multiple_scattering(&luts.params, r, mu_sun);
            let gpu = table[(y * width + x) as usize];
            for channel in 0..3 {
                assert!(
                    relative_error(gpu[channel], cpu[channel]) < 0.02,
                    "texel ({x}, {y}) channel {channel}: gpu {} cpu {}",
                    gpu[channel],
                    cpu[channel]
                );
            }
        }
    }

    #[test]
    fn halves_decode() {
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0x3800), 0.5);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert_eq!(half_to_f32(0x0000), 0.0);
    }
}
