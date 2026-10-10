//! Image-based lighting (C2): the split-sum BRDF integration LUT, baked
//! once at startup, and the environment it integrates against: the sky as
//! a cube, prefiltered by roughness, with its irradiance as harmonics,
//! re-baked when the sun moves.

use crate::block_on::block_on;
use crate::shaders::BRDF_LUT;

/// The LUT's size: n.v across, perceptual roughness down.
pub(crate) const BRDF_LUT_SIZE: u32 = 128;
/// Importance samples per texel. Must match `brdf_lut.wgsl`.
#[cfg(test)]
pub(crate) const BRDF_LUT_SAMPLES: u32 = 1024;
const WORKGROUP: u32 = 8;

/// The baked split-sum table.
#[derive(Debug)]
pub(crate) struct BrdfLut {
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "read back by the GPU tests; the view keeps it alive")
    )]
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
}

impl BrdfLut {
    /// Compiles `source` (`brdf_lut.wgsl`, includes expanded) and allocates
    /// the table. Call [`bake`](Self::bake) before reading it.
    pub(crate) fn new(device: &wgpu::Device, source: &str) -> Result<Self, String> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(BRDF_LUT),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brdf-lut"),
            layout: None,
            module: &module,
            entry_point: Some("brdf_lut_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        if let Some(error) = block_on(scope.pop()) {
            return Err(error.to_string());
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brdf-lut"),
            size: wgpu::Extent3d {
                width: BRDF_LUT_SIZE,
                height: BRDF_LUT_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Storage-capable everywhere; the table only needs two channels.
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brdf-lut-bake"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        Ok(Self {
            texture,
            view,
            pipeline,
            group,
        })
    }

    /// Records the bake into `encoder`.
    pub(crate) fn bake(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("brdf-lut"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        let groups = BRDF_LUT_SIZE.div_ceil(WORKGROUP);
        pass.dispatch_workgroups(groups, groups, 1);
    }
}

/// Environment cube faces, texels per side. Must match `ibl_environment.wgsl`.
pub(crate) const ENVIRONMENT_SIZE: u32 = 64;
/// Prefiltered mips, 64 down to 1: roughness 0 to 1 in sixths.
pub(crate) const PREFILTER_MIPS: u32 = 7;
/// `EnvironmentParams` in `ibl_environment.wgsl`: four `vec4`s.
const ENVIRONMENT_PARAMS_SIZE: u64 = 64;
/// `Environment` in `ibl.wgsl`: a header `vec4` and nine SH `vec4`s.
pub(crate) const ENVIRONMENT_UNIFORM_SIZE: u64 = 160;
const SH_BYTES: u64 = 9 * 16;
/// The sun may move this far (cosine) before the environment is re-baked.
/// About a quarter of a degree: a minute of game time at a typical rate.
const REBAKE_SUN_COSINE: f32 = 0.999_99;

/// What an environment was baked from, to tell when it is out of date.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BakeKey {
    pub(crate) sun: [f32; 3],
    pub(crate) sun_illuminance: [f32; 3],
    pub(crate) camera_radius: f32,
}

impl BakeKey {
    /// Whether an environment baked for `self` still stands for `other`.
    pub(crate) fn still_fits(&self, other: &Self) -> bool {
        let dot = self.sun[0] * other.sun[0] + self.sun[1] * other.sun[1] + self.sun[2] * other.sun[2];
        dot >= REBAKE_SUN_COSINE
            && self.sun_illuminance == other.sun_illuminance
            && (self.camera_radius - other.camera_radius).abs() < 0.01
    }
}

/// The `EnvironmentParams` block. `uniform_radiance` replaces the sky with
/// a constant, which only tests use.
pub(crate) fn environment_params_bytes(key: &BakeKey, uniform_radiance: Option<[f32; 3]>) -> [u8; 64] {
    let mut words = [0.0_f32; 16];
    words[0..3].copy_from_slice(&key.sun);
    words[4..7].copy_from_slice(&key.sun_illuminance);
    words[8] = key.camera_radius;
    if let Some(radiance) = uniform_radiance {
        words[12..15].copy_from_slice(&radiance);
        words[15] = 1.0;
    }
    let mut out = [0_u8; 64];
    for (chunk, word) in out.chunks_exact_mut(4).zip(words) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    out
}

/// The header of the `Environment` uniform: image-based light on or off.
pub(crate) fn environment_header_bytes(enabled: bool) -> [u8; 16] {
    let words = [
        if enabled { 1.0_f32 } else { 0.0 },
        PREFILTER_MIPS as f32,
        0.0,
        0.0,
    ];
    let mut out = [0_u8; 16];
    for (chunk, word) in out.chunks_exact_mut(4).zip(words) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    out
}

fn cube_texture(device: &wgpu::Device, label: &str, mips: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: ENVIRONMENT_SIZE,
            height: ENVIRONMENT_SIZE,
            depth_or_array_layers: 6,
        },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn array_view(texture: &wgpu::Texture, mip: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        base_mip_level: mip,
        mip_level_count: Some(1),
        ..wgpu::TextureViewDescriptor::default()
    })
}

fn cube_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..wgpu::TextureViewDescriptor::default()
    })
}

/// The environment the PBR shader lights with: the sky as a cube, that cube
/// prefiltered by roughness, and its irradiance as harmonics.
#[derive(Debug)]
pub(crate) struct Environment {
    params: wgpu::Buffer,
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "read back by the GPU tests; the views keep it alive")
    )]
    pub(crate) prefiltered: wgpu::Texture,
    /// The prefiltered cube, sampled by `ibl.wgsl`.
    pub(crate) prefiltered_view: wgpu::TextureView,
    sh: wgpu::Buffer,
    /// `Environment` for `ibl.wgsl`: the header, then the SH copied in.
    pub(crate) uniform: wgpu::Buffer,
    cube_pipeline: wgpu::ComputePipeline,
    prefilter_pipeline: wgpu::ComputePipeline,
    sh_pipeline: wgpu::ComputePipeline,
    cube_group: wgpu::BindGroup,
    prefilter_groups: Vec<wgpu::BindGroup>,
    sh_group: wgpu::BindGroup,
    /// What the environment was last baked from; `None` before the first.
    pub(crate) baked: Option<BakeKey>,
}

impl Environment {
    /// Compiles `source` (`ibl_environment.wgsl`, includes expanded) and
    /// binds it to the atmosphere's parameters and this frame's sky view.
    pub(crate) fn new(
        device: &wgpu::Device,
        source: &str,
        atmosphere: &wgpu::Buffer,
        sky_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> Result<Self, String> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(crate::shaders::IBL_ENVIRONMENT),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let compute = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let cube_pipeline = compute("cube_main");
        let prefilter_pipeline = compute("prefilter_main");
        let sh_pipeline = compute("sh_main");
        if let Some(error) = block_on(scope.pop()) {
            return Err(error.to_string());
        }

        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let params = buffer(
            "environment-params",
            ENVIRONMENT_PARAMS_SIZE,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let sh = buffer(
            "environment-sh",
            SH_BYTES,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let uniform = buffer(
            "environment",
            ENVIRONMENT_UNIFORM_SIZE,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        );
        let radiance = cube_texture(device, "environment-radiance", 1);
        let prefiltered = cube_texture(device, "environment-prefiltered", PREFILTER_MIPS);
        let radiance_cube = cube_view(&radiance);
        let prefiltered_view = cube_view(&prefiltered);

        let entry = |binding, resource| wgpu::BindGroupEntry { binding, resource };
        let group = |pipeline: &wgpu::ComputePipeline, label, entries: &[wgpu::BindGroupEntry<'_>]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &pipeline.get_bind_group_layout(0),
                entries,
            })
        };
        let radiance_out = array_view(&radiance, 0);
        let cube_group = group(
            &cube_pipeline,
            "environment-cube",
            &[
                entry(0, atmosphere.as_entire_binding()),
                entry(1, params.as_entire_binding()),
                entry(2, wgpu::BindingResource::TextureView(sky_view)),
                entry(3, wgpu::BindingResource::Sampler(sampler)),
                entry(4, wgpu::BindingResource::TextureView(&radiance_out)),
            ],
        );
        let prefilter_groups = (0..PREFILTER_MIPS)
            .map(|mip| {
                let out = array_view(&prefiltered, mip);
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("environment-prefilter"),
                    layout: &prefilter_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&radiance_cube),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::TextureView(&out),
                        },
                    ],
                })
            })
            .collect();
        let radiance_faces = array_view(&radiance, 0);
        let sh_group = group(
            &sh_pipeline,
            "environment-sh",
            &[
                entry(7, sh.as_entire_binding()),
                entry(8, wgpu::BindingResource::TextureView(&radiance_faces)),
            ],
        );
        Ok(Self {
            params,
            prefiltered,
            prefiltered_view,
            sh,
            uniform,
            cube_pipeline,
            prefilter_pipeline,
            sh_pipeline,
            cube_group,
            prefilter_groups,
            sh_group,
            baked: None,
        })
    }

    /// Readies a bake for `key`: writes its parameters and records that the
    /// environment now stands for it. [`record_bake`](Self::record_bake)
    /// does the GPU work, after this frame's sky-view LUT exists.
    pub(crate) fn prepare_bake(
        &mut self,
        queue: &wgpu::Queue,
        key: BakeKey,
        uniform_radiance: Option<[f32; 3]>,
    ) {
        queue.write_buffer(&self.params, 0, &environment_params_bytes(&key, uniform_radiance));
        self.baked = Some(key);
    }

    /// Records the bake: the cube, its prefiltered mips, and the harmonics
    /// copied into the shading uniform.
    pub(crate) fn record_bake(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("environment"),
                timestamp_writes: None,
            });
            let groups = ENVIRONMENT_SIZE.div_ceil(WORKGROUP);
            pass.set_pipeline(&self.cube_pipeline);
            pass.set_bind_group(0, &self.cube_group, &[]);
            pass.dispatch_workgroups(groups, groups, 6);
            pass.set_pipeline(&self.prefilter_pipeline);
            for (mip, group) in self.prefilter_groups.iter().enumerate() {
                let size = (ENVIRONMENT_SIZE >> mip).max(1);
                pass.set_bind_group(0, group, &[]);
                let groups = size.div_ceil(WORKGROUP);
                pass.dispatch_workgroups(groups, groups, 6);
            }
            pass.set_pipeline(&self.sh_pipeline);
            pass.set_bind_group(0, &self.sh_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.sh, 0, &self.uniform, 16, SH_BYTES);
    }

    /// Writes whether image-based light is on this frame.
    pub(crate) fn set_enabled(&self, queue: &wgpu::Queue, enabled: bool) {
        queue.write_buffer(&self.uniform, 0, &environment_header_bytes(enabled));
    }
}

/// CPU mirror of `brdf_lut.wgsl` in `f64`, for tests.
#[cfg(test)]
pub(crate) mod mirror {
    use super::BRDF_LUT_SAMPLES;
    use std::f64::consts::PI;

    fn visibility(n_dot_v: f64, n_dot_l: f64, alpha: f64) -> f64 {
        let alpha2 = alpha * alpha;
        let view = n_dot_l * (n_dot_v * n_dot_v * (1.0 - alpha2) + alpha2).sqrt();
        let light = n_dot_v * (n_dot_l * n_dot_l * (1.0 - alpha2) + alpha2).sqrt();
        0.5 / (view + light).max(1e-5)
    }

    /// (A, B) for one texel, by the shader's importance sampling with
    /// `samples` points.
    pub(crate) fn texel(n_dot_v: f64, roughness: f64, samples: u32) -> (f64, f64) {
        let alpha = roughness * roughness;
        let view = [(1.0 - n_dot_v * n_dot_v).sqrt(), 0.0, n_dot_v];
        let (mut scale, mut bias) = (0.0, 0.0);
        for index in 0..samples {
            let xi = (
                f64::from(index) / f64::from(samples),
                f64::from(index.reverse_bits()) / 4_294_967_296.0,
            );
            let phi = 2.0 * PI * xi.0;
            let cos_theta = ((1.0 - xi.1) / (1.0 + (alpha * alpha - 1.0) * xi.1)).sqrt();
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
            let half = [sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta];
            let v_dot_h = view[0] * half[0] + view[1] * half[1] + view[2] * half[2];
            let n_dot_l = 2.0 * v_dot_h * half[2] - view[2];
            if n_dot_l <= 0.0 {
                continue;
            }
            let weight =
                4.0 * visibility(n_dot_v, n_dot_l, alpha) * n_dot_l * v_dot_h.max(0.0) / half[2].max(1e-5);
            let fresnel = (1.0 - v_dot_h.max(0.0)).powi(5);
            scale += (1.0 - fresnel) * weight;
            bias += fresnel * weight;
        }
        (scale / f64::from(samples), bias / f64::from(samples))
    }

    pub(crate) fn texel_default(n_dot_v: f64, roughness: f64) -> (f64, f64) {
        texel(n_dot_v, roughness, BRDF_LUT_SAMPLES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_smooth_surface_seen_head_on_reflects_f0_exactly() {
        // Roughness near zero puts every sample on the mirror direction:
        // A = 1 and B = 0, so the reflectance is F0 itself.
        let (scale, bias) = mirror::texel_default(0.999, 0.02);
        assert!(
            (scale - 1.0).abs() < 0.01 && bias.abs() < 0.01,
            "({scale}, {bias})"
        );
    }

    #[test]
    fn energy_is_never_created_and_grazing_angles_gain_fresnel() {
        for n_dot_v in [0.05, 0.3, 0.7, 0.99] {
            for roughness in [0.1, 0.5, 0.9] {
                let (scale, bias) = mirror::texel_default(n_dot_v, roughness);
                assert!(
                    scale >= 0.0 && bias >= 0.0 && scale + bias <= 1.0 + 1e-3,
                    "({scale}, {bias})"
                );
            }
        }
        let grazing = mirror::texel_default(0.05, 0.2).1;
        let head_on = mirror::texel_default(0.95, 0.2).1;
        assert!(
            grazing > head_on * 5.0,
            "Fresnel rises toward grazing: {grazing} against {head_on}"
        );
    }

    #[test]
    fn the_shaders_sample_count_has_converged() {
        // The baked count is within half a percent of four times as many.
        for (n_dot_v, roughness) in [(0.2, 0.3), (0.6, 0.6), (0.9, 0.9)] {
            let (a, b) = mirror::texel_default(n_dot_v, roughness);
            let (a4, b4) = mirror::texel(n_dot_v, roughness, BRDF_LUT_SAMPLES * 4);
            assert!((a + b - a4 - b4).abs() / (a4 + b4) < 0.005);
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn the_baked_lut_matches_the_cpu_integration_within_half_a_percent() {
        use crate::{Renderer, RendererConfig};
        let renderer = Renderer::headless(16, 16, RendererConfig::default()).expect("headless");
        let size = (BRDF_LUT_SIZE, BRDF_LUT_SIZE);
        let table = crate::atmosphere::luts::tests::read_lut(&renderer, &renderer.brdf_lut.texture, size);
        for &(x, y) in &[(3, 3), (20, 100), (64, 64), (127, 10), (100, 127), (5, 120)] {
            let n_dot_v = (f64::from(x) + 0.5) / f64::from(BRDF_LUT_SIZE);
            let roughness = (f64::from(y) + 0.5) / f64::from(BRDF_LUT_SIZE);
            let (scale, bias) = mirror::texel_default(n_dot_v, roughness);
            let gpu = table[(y * BRDF_LUT_SIZE + x) as usize];
            for (gpu, cpu, name) in [(gpu[0], scale, "A"), (gpu[1], bias, "B")] {
                let error = (f64::from(gpu) - cpu).abs() / cpu.abs().max(0.01);
                assert!(error < 0.005, "texel ({x}, {y}) {name}: gpu {gpu} cpu {cpu}");
            }
        }
    }

    /// The direction through a cube texel, as `face_direction` in the shader.
    fn face_direction(face: u32, x: u32, y: u32, size: u32) -> [f64; 3] {
        let u = (f64::from(x) + 0.5) / f64::from(size) * 2.0 - 1.0;
        let v = (f64::from(y) + 0.5) / f64::from(size) * 2.0 - 1.0;
        let d = match face {
            0 => [1.0, -v, -u],
            1 => [-1.0, -v, u],
            2 => [u, 1.0, v],
            3 => [u, -1.0, -v],
            4 => [u, -v, 1.0],
            _ => [-u, -v, -1.0],
        };
        let length = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        d.map(|value| value / length)
    }

    fn sh_basis(d: [f64; 3]) -> [f64; 9] {
        let [x, y, z] = d;
        [
            0.282095,
            0.488603 * y,
            0.488603 * z,
            0.488603 * x,
            1.092548 * x * y,
            1.092548 * y * z,
            0.315392 * (3.0 * z * z - 1.0),
            1.092548 * x * z,
            0.546274 * (x * x - y * y),
        ]
    }

    /// Projects `radiance` (one channel) over the cube as `sh_main` does.
    fn project(radiance: impl Fn([f64; 3]) -> f64, size: u32) -> [f64; 9] {
        let mut sh = [0.0; 9];
        for face in 0..6 {
            for y in 0..size {
                for x in 0..size {
                    let u = (f64::from(x) + 0.5) / f64::from(size) * 2.0 - 1.0;
                    let v = (f64::from(y) + 0.5) / f64::from(size) * 2.0 - 1.0;
                    let solid_angle = 4.0 / (f64::from(size * size) * (1.0 + u * u + v * v).powf(1.5));
                    let direction = face_direction(face, x, y, size);
                    let value = radiance(direction) * solid_angle;
                    for (coefficient, basis) in sh.iter_mut().zip(sh_basis(direction)) {
                        *coefficient += value * basis;
                    }
                }
            }
        }
        sh
    }

    /// Irradiance from harmonics, as `sh_irradiance` in `ibl.wgsl`.
    fn irradiance(sh: &[f64; 9], n: [f64; 3]) -> f64 {
        use std::f64::consts::PI;
        let bands = [
            PI,
            2.0 * PI / 3.0,
            2.0 * PI / 3.0,
            2.0 * PI / 3.0,
            PI / 4.0,
            PI / 4.0,
            PI / 4.0,
            PI / 4.0,
            PI / 4.0,
        ];
        sh.iter()
            .zip(sh_basis(n))
            .zip(bands)
            .map(|((coefficient, basis), band)| coefficient * basis * band)
            .sum::<f64>()
            .max(0.0)
    }

    #[test]
    fn a_uniform_sky_gives_pi_times_its_radiance_from_every_side() {
        let sh = project(|_| 0.5, ENVIRONMENT_SIZE);
        for n in [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.6, 0.0, 0.8],
        ] {
            let e = irradiance(&sh, n);
            assert!(
                (e - std::f64::consts::PI * 0.5).abs() / (std::f64::consts::PI * 0.5) < 0.01,
                "{n:?}: {e}"
            );
        }
    }

    #[test]
    fn a_bright_upper_sky_lights_what_faces_up() {
        let sh = project(|d| if d[1] > 0.0 { 1.0 } else { 0.1 }, ENVIRONMENT_SIZE);
        let up = irradiance(&sh, [0.0, 1.0, 0.0]);
        let side = irradiance(&sh, [1.0, 0.0, 0.0]);
        let down = irradiance(&sh, [0.0, -1.0, 0.0]);
        assert!(up > side && side > down, "{up} {side} {down}");
        // The exact irradiance under a sky of 1 above and 0.1 below, facing
        // up, is pi; band 2 gets within a few percent.
        assert!(
            (up - std::f64::consts::PI).abs() / std::f64::consts::PI < 0.05,
            "{up}"
        );
    }

    #[test]
    fn a_bake_key_holds_until_the_sun_moves_a_quarter_degree() {
        let key = BakeKey {
            sun: [0.0, 1.0, 0.0],
            sun_illuminance: [3.0; 3],
            camera_radius: 6360.002,
        };
        let tilt = |degrees: f32| {
            let (sin, cos) = degrees.to_radians().sin_cos();
            BakeKey {
                sun: [sin, cos, 0.0],
                ..key
            }
        };
        assert!(key.still_fits(&tilt(0.1)));
        assert!(!key.still_fits(&tilt(0.5)));
        assert!(!key.still_fits(&BakeKey {
            sun_illuminance: [2.0; 3],
            ..key
        }));
    }

    fn read_buffer(renderer: &crate::Renderer, source: &wgpu::Buffer, size: u64) -> Vec<f32> {
        let buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(source, 0, &buffer, 0, size);
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
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect()
    }

    /// One mip of one face of the prefiltered cube, as RGB.
    fn read_face(renderer: &crate::Renderer, face: u32, mip: u32) -> Vec<[f32; 3]> {
        let size = (ENVIRONMENT_SIZE >> mip).max(1);
        let bytes_per_row =
            (size * 8).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(bytes_per_row) * u64::from(size),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &renderer.environment.prefiltered,
                mip_level: mip,
                origin: wgpu::Origin3d { x: 0, y: 0, z: face },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d {
                width: size,
                height: size,
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
        let half = crate::atmosphere::luts::tests::half_to_f32;
        mapped
            .chunks_exact(bytes_per_row as usize)
            .flat_map(|row| {
                row[..size as usize * 8].chunks_exact(8).map(move |texel| {
                    std::array::from_fn(|channel| {
                        half(u16::from_le_bytes([texel[channel * 2], texel[channel * 2 + 1]]))
                    })
                })
            })
            .collect()
    }

    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn a_uniform_environment_bakes_to_exact_irradiance_and_flat_mips() {
        let mut renderer =
            crate::Renderer::headless(16, 16, crate::RendererConfig::default()).expect("headless");
        let radiance = [0.25_f32, 0.5, 1.0];
        let key = BakeKey {
            sun: [0.0, 1.0, 0.0],
            sun_illuminance: [1.0; 3],
            camera_radius: 6360.002,
        };
        renderer
            .environment
            .prepare_bake(&renderer.queue, key, Some(radiance));
        let mut encoder = renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        renderer.environment.record_bake(&mut encoder);
        renderer.queue.submit(std::iter::once(encoder.finish()));

        let words = read_buffer(&renderer, &renderer.environment.uniform, ENVIRONMENT_UNIFORM_SIZE);
        for channel in 0..3 {
            let sh: [f64; 9] = std::array::from_fn(|index| f64::from(words[4 + index * 4 + channel]));
            for n in [
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.6, 0.8, 0.0],
            ] {
                let expected = std::f64::consts::PI * f64::from(radiance[channel]);
                let e = irradiance(&sh, n);
                assert!(
                    (e - expected).abs() / expected < 0.01,
                    "channel {channel} {n:?}: {e} against {expected}"
                );
            }
        }
        for mip in [0, 2, 4, 6] {
            for face in 0..6 {
                for texel in read_face(&renderer, face, mip) {
                    for channel in 0..3 {
                        let error = (texel[channel] - radiance[channel]).abs() / radiance[channel];
                        assert!(error < 0.01, "mip {mip} face {face}: {texel:?}");
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn the_environment_re_bakes_only_when_the_sun_moves() {
        use crate::{Color, RenderFrame, Sky};
        let mut renderer =
            crate::Renderer::headless(16, 16, crate::RendererConfig::default()).expect("headless");
        let mut frame = RenderFrame::new(crate::uniforms::IDENTITY, Color::BLACK);
        let bakes = |renderer: &crate::Renderer| renderer.graph_dump().contains(" environment:");
        frame.sky = Some(Sky::default());
        renderer.render_to_image(&frame).expect("renders");
        assert!(
            bakes(&renderer),
            "the first sky bakes:\n{}",
            renderer.graph_dump()
        );
        renderer.render_to_image(&frame).expect("renders");
        assert!(!bakes(&renderer), "an unchanged sky does not");
        let mut moved = Sky::default();
        moved.sun_direction[0] += 0.05;
        frame.sky = Some(moved);
        renderer.render_to_image(&frame).expect("renders");
        assert!(bakes(&renderer), "a moved sun does");
        frame.sky = None;
        renderer.render_to_image(&frame).expect("renders");
        assert!(!bakes(&renderer), "no sky, no bake");
    }
}
