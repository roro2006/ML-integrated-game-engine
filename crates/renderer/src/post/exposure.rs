//! Automatic exposure: a luminance histogram of the HDR scene, averaged and
//! adapted over time on the GPU (`exposure_histogram.wgsl`).
//!
//! The adapted luminance stays on the GPU. After the average pass it is
//! copied into a small uniform the tonemap pass reads, in the same submission,
//! so nothing waits on a readback and the tonemap never reads a storage buffer
//! (which downlevel fragment stages need not support).
//!
//! The CPU decides only how far to adapt this frame, from the frame time and
//! the speeds in [`AutoExposure`]. Infinite speeds adapt instantly, which is
//! also what tests use to be deterministic.

use crate::block_on::block_on;
use crate::frame::AutoExposure;
use crate::shaders::EXPOSURE_HISTOGRAM;

/// Bins in the histogram; must match `BINS` in the shader.
pub(crate) const BINS: usize = 256;

/// `Params` in the shader.
pub(crate) const PARAMS_SIZE: usize = 16;

/// `State` in the shader: the adapted luminance, a flag and padding. Also the
/// size of the tonemap pass's copy.
pub(crate) const STATE_SIZE: usize = 16;

/// The mid-grey the adapted average is mapped to; `EXPOSURE_KEY` in
/// `tonemap.wgsl`.
#[cfg(test)]
pub(crate) const KEY: f32 = 0.18;

/// The longest frame adaptation is allowed to cover. A long stall (a loading
/// hitch, a debugger break) should not snap the exposure.
const MAX_ADAPTATION_STEP_SECONDS: f32 = 0.25;

/// The share of the gap to close this frame at `speed` per second.
pub(crate) fn blend(speed: f32, frame_seconds: f32) -> f32 {
    if speed.is_infinite() && speed > 0.0 {
        return 1.0;
    }
    let positive = |value: f32| value > 0.0;
    if !positive(speed) || !positive(frame_seconds) {
        return 0.0;
    }
    1.0 - (-speed * frame_seconds.min(MAX_ADAPTATION_STEP_SECONDS)).exp()
}

/// The `Params` uniform for one frame.
pub(crate) fn params_bytes(settings: &AutoExposure, frame_seconds: f32) -> [u8; PARAMS_SIZE] {
    let min = settings.min_log2_luminance;
    let range = (settings.max_log2_luminance - min).max(1.0);
    let values = [
        min,
        range,
        blend(settings.brighten_speed, frame_seconds),
        blend(settings.darken_speed, frame_seconds),
    ];
    let mut out = [0; PARAMS_SIZE];
    for (index, value) in values.iter().enumerate() {
        out[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    out
}

/// GPU resources that live as long as the renderer: the histogram, the
/// adapted state, its copy for the tonemap pass, and the two pipelines.
#[derive(Debug)]
pub(crate) struct ExposureResources {
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) build: wgpu::ComputePipeline,
    pub(crate) average: wgpu::ComputePipeline,
    pub(crate) histogram: wgpu::Buffer,
    pub(crate) state: wgpu::Buffer,
    pub(crate) params: wgpu::Buffer,
    /// What the tonemap pass reads: `state` copied after averaging.
    pub(crate) exposure_uniform: wgpu::Buffer,
}

impl ExposureResources {
    pub(crate) fn new(device: &wgpu::Device, source: &str) -> Result<Self, String> {
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("exposure-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage(1),
                storage(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
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
            label: Some("exposure-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(EXPOSURE_HISTOGRAM),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let build = pipeline("build");
        let average = pipeline("average");
        if let Some(error) = block_on(scope.pop()) {
            return Err(error.to_string());
        }

        let buffer = |label, size: usize, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        Ok(Self {
            layout,
            build,
            average,
            // Zeroed at creation, which is the empty histogram and the
            // "nothing measured yet" state the shader expects.
            histogram: buffer("exposure-histogram", BINS * 4, wgpu::BufferUsages::STORAGE),
            state: buffer(
                "exposure-state",
                STATE_SIZE,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            params: buffer(
                "exposure-params",
                PARAMS_SIZE,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            exposure_uniform: buffer(
                "exposure-uniform",
                STATE_SIZE,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
        })
    }
}

/// CPU copies of the shader's maths, for tests.
#[cfg(test)]
pub(crate) mod mirror {
    use crate::frame::AutoExposure;

    /// Mirror of `luminance_bin` in the shader.
    pub(crate) fn luminance_bin(colour: [f32; 3], settings: &AutoExposure) -> usize {
        let luminance = colour[0] * 0.2126 + colour[1] * 0.7152 + colour[2] * 0.0722;
        let lit = luminance > 0.0;
        if !lit {
            return 0;
        }
        let min = settings.min_log2_luminance;
        let range = (settings.max_log2_luminance - min).max(1.0);
        let position = (luminance.log2() - min) / range;
        if position < 0.0 {
            return 0;
        }
        1 + (position.clamp(0.0, 1.0) * 254.0) as usize
    }

    /// Mirror of the shader's `average`, without adaptation: the luminance a
    /// histogram measures.
    pub(crate) fn measured_luminance(histogram: &[u32; super::BINS], settings: &AutoExposure) -> Option<f32> {
        let (weighted, counted) = histogram.iter().enumerate().skip(1).fold(
            (0.0f64, 0.0f64),
            |(weighted, counted), (bin, &count)| {
                (
                    weighted + f64::from(count) * bin as f64,
                    counted + f64::from(count),
                )
            },
        );
        if counted <= 0.0 {
            return None;
        }
        let min = settings.min_log2_luminance;
        let range = (settings.max_log2_luminance - min).max(1.0);
        let position = ((weighted / counted) as f32 - 1.0) / 254.0;
        Some((position * range + min).exp2())
    }
}

#[cfg(test)]
mod tests {
    use super::mirror::*;
    use super::*;

    #[test]
    fn instant_adaptation_closes_the_whole_gap() {
        assert_eq!(
            blend(f32::INFINITY, 0.0),
            1.0,
            "even on a zero-length first frame"
        );
        assert_eq!(blend(f32::INFINITY, 0.016), 1.0);
    }

    #[test]
    fn finite_adaptation_is_exponential_and_capped() {
        let one_frame = blend(2.0, 0.016);
        assert!((one_frame - (1.0 - (-0.032f32).exp())).abs() < 1e-6);
        assert_eq!(
            blend(2.0, 10.0),
            blend(2.0, MAX_ADAPTATION_STEP_SECONDS),
            "a stall is capped"
        );
        assert_eq!(blend(0.0, 0.016), 0.0, "zero speed never moves");
        assert_eq!(blend(f32::NAN, 0.016), 0.0, "nor does a broken one");
        assert_eq!(blend(2.0, -1.0), 0.0, "time does not run backwards");
    }

    #[test]
    fn bins_cover_the_range_in_stops_and_black_is_bin_zero() {
        let settings = AutoExposure::default();
        assert_eq!(luminance_bin([0.0; 3], &settings), 0);
        assert_eq!(luminance_bin([f32::NAN; 3], &settings), 0);
        let below = settings.min_log2_luminance.exp2() * 0.5;
        assert_eq!(
            luminance_bin([below; 3], &settings),
            0,
            "below the range is black"
        );
        assert_eq!(
            luminance_bin([settings.min_log2_luminance.exp2(); 3], &settings),
            1
        );
        assert_eq!(
            luminance_bin([1.0e9; 3], &settings),
            255,
            "above the range clamps"
        );
        assert!(luminance_bin([1.0; 3], &settings) < luminance_bin([2.0; 3], &settings));
    }

    #[test]
    fn a_uniform_scene_measures_its_own_luminance() {
        let settings = AutoExposure::default();
        for luminance in [0.05f32, 1.0, 20.0] {
            let mut histogram = [0u32; BINS];
            histogram[luminance_bin([luminance; 3], &settings)] = 1000;
            histogram[0] = 5000; // black sky, ignored
            let measured = measured_luminance(&histogram, &settings).expect("something measured");
            // One bin is (range / 254) stops wide.
            let bin_stops = (settings.max_log2_luminance - settings.min_log2_luminance) / 254.0;
            assert!(
                (measured.log2() - luminance.log2()).abs() <= bin_stops,
                "{luminance} measured as {measured}"
            );
        }
        assert_eq!(
            measured_luminance(&[0; BINS], &settings),
            None,
            "an all-black frame measures nothing"
        );
    }

    #[test]
    fn params_carry_the_range_and_both_blends() {
        let settings = AutoExposure {
            min_log2_luminance: -8.0,
            max_log2_luminance: 4.0,
            brighten_speed: f32::INFINITY,
            darken_speed: 0.0,
        };
        let bytes = params_bytes(&settings, 0.016);
        let at = |i: usize| f32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
        assert_eq!(at(0), -8.0);
        assert_eq!(at(4), 12.0);
        assert_eq!(at(8), 1.0);
        assert_eq!(at(12), 0.0);
    }
}
