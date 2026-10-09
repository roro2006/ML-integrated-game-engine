//! Bloom: the dual-filter downsample and upsample chain (`bloom.wgsl`).
//!
//! The chain's levels are graph transients, half the output size and halving
//! from there, so they are pooled and resized like any other target. Bloom is
//! scene light: the tonemap pass adds it before exposure, so it brightens and
//! adapts with everything else.

use crate::block_on::block_on;
use crate::frame::Bloom;
use crate::passes::HDR_FORMAT;
use crate::shaders::BLOOM;

/// At most this many levels: past six halvings the blur is wider than any
/// light needs and the levels are a few texels across.
pub(crate) const MAX_LEVELS: u32 = 6;

/// `Params` in `bloom.wgsl`.
pub(crate) const PARAMS_SIZE: usize = 16;

/// The size of each level for an output of `width` by `height`: half the
/// output, then halving, never below 1, up to [`MAX_LEVELS`].
pub(crate) fn level_sizes(width: u32, height: u32) -> Vec<(u32, u32)> {
    let mut sizes = Vec::new();
    let (mut w, mut h) = ((width / 2).max(1), (height / 2).max(1));
    for _ in 0..MAX_LEVELS {
        sizes.push((w, h));
        if w == 1 && h == 1 {
            break;
        }
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    sizes
}

/// Whether a frame's bloom does anything at all.
pub(crate) fn enabled(bloom: &Bloom) -> bool {
    bloom.intensity.is_finite() && bloom.intensity > 0.0
}

/// The `Params` uniform: threshold and knee, sanitised.
pub(crate) fn params_bytes(bloom: &Bloom) -> [u8; PARAMS_SIZE] {
    let finite_or = |value: f32, fallback: f32| if value.is_finite() { value } else { fallback };
    let threshold = finite_or(bloom.threshold, 1.0).max(0.0);
    let knee = finite_or(bloom.knee, 0.0).max(0.0);
    let mut out = [0; PARAMS_SIZE];
    out[0..4].copy_from_slice(&threshold.to_le_bytes());
    out[4..8].copy_from_slice(&knee.to_le_bytes());
    out
}

/// The pipelines, sampler and parameter buffer the chain shares.
#[derive(Debug)]
pub(crate) struct BloomResources {
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) prefilter: wgpu::RenderPipeline,
    pub(crate) downsample: wgpu::RenderPipeline,
    pub(crate) upsample: wgpu::RenderPipeline,
    pub(crate) sampler: wgpu::Sampler,
    pub(crate) params: wgpu::Buffer,
}

impl BloomResources {
    pub(crate) fn new(device: &wgpu::Device, source: &str) -> Result<Self, String> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(PARAMS_SIZE as u64),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(BLOOM),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = |entry: &str, prefilter: bool, blend: Option<wgpu::BlendState>| {
            let constants = [("PREFILTER", if prefilter { 1.0 } else { 0.0 })];
            let targets = [Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })];
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let prefilter = pipeline("downsample", true, None);
        let downsample = pipeline("downsample", false, None);
        let upsample = pipeline("upsample", false, Some(additive));
        if let Some(error) = block_on(scope.pop()) {
            return Err(error.to_string());
        }
        Ok(Self {
            layout,
            prefilter,
            downsample,
            upsample,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("bloom-sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bloom-params"),
                size: PARAMS_SIZE as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        })
    }
}

/// Mirror of `bright_part` in `bloom.wgsl`, for one colour.
#[cfg(test)]
pub(crate) fn bright_part(colour: [f32; 3], threshold: f32, knee: f32) -> [f32; 3] {
    let brightness = colour.iter().copied().fold(0.0, f32::max);
    let knee = knee.max(1e-5);
    let soft = (brightness - threshold + knee).clamp(0.0, 2.0 * knee);
    let soft = soft * soft / (4.0 * knee);
    let weight = soft.max(brightness - threshold) / brightness.max(1e-5);
    colour.map(|channel| channel * weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_halve_from_half_size_and_stop_at_six_or_one_texel() {
        assert_eq!(
            level_sizes(1920, 1080),
            [(960, 540), (480, 270), (240, 135), (120, 67), (60, 33), (30, 16)]
        );
        assert_eq!(level_sizes(160, 120).len(), 6);
        assert_eq!(
            level_sizes(4, 4),
            [(2, 2), (1, 1)],
            "a tiny output stops at one texel"
        );
        assert_eq!(level_sizes(1, 1), [(1, 1)]);
    }

    #[test]
    fn nothing_below_the_knee_blooms_and_everything_above_it_does() {
        assert_eq!(
            bright_part([0.4; 3], 1.0, 0.5),
            [0.0; 3],
            "below threshold - knee"
        );
        let well_above = bright_part([4.0, 2.0, 1.0], 1.0, 0.5);
        assert!(
            (well_above[0] - 3.0).abs() < 1e-5,
            "the threshold is subtracted from the brightest channel"
        );
        assert!(
            (well_above[1] / well_above[0] - 0.5).abs() < 1e-5,
            "and the colour keeps its hue"
        );
        let in_knee = bright_part([1.0; 3], 1.0, 0.5);
        assert!(in_knee[0] > 0.0 && in_knee[0] < 0.5, "a soft start: {in_knee:?}");
    }

    #[test]
    fn bloom_is_off_unless_it_has_a_positive_finite_intensity() {
        assert!(!enabled(&Bloom::default()));
        let on = Bloom {
            intensity: 0.05,
            ..Bloom::default()
        };
        assert!(enabled(&on));
        assert!(!enabled(&Bloom {
            intensity: f32::NAN,
            ..on
        }));
    }

    #[test]
    fn params_are_sanitised() {
        let bytes = params_bytes(&Bloom {
            intensity: 1.0,
            threshold: f32::NAN,
            knee: -2.0,
        });
        assert_eq!(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]), 1.0);
        assert_eq!(f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]), 0.0);
    }
}
