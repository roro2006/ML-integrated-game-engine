//! The tonemap pass: HDR scene light to display values.
//!
//! The curves are mirrored here on the CPU, function for function with
//! `assets/shaders/colour.wgsl`, so their behaviour is unit-tested without a
//! GPU and a GPU test can hold the shader to the mirror.

use crate::block_on::block_on;
use crate::shaders::TONEMAP;
use crate::{DEFAULT_EXPOSURE_EV, PostSettings, Tonemapper};

/// `Post` in `tonemap.wgsl`: exposure (f32), curve (u32), bloom intensity
/// (f32), auto-exposure flag (u32), then lift, inverse gamma and gain (three
/// `vec4<f32>`, rgb used).
pub(crate) const POST_UNIFORM_SIZE: usize = 64;

/// Gamma is clamped away from zero, where `1 / gamma` would be infinite.
const MIN_GAMMA: f32 = 0.01;

/// Exposure is clamped to this many stops either way. Beyond it the image is
/// black or white whatever the scene, and an unbounded value can overflow
/// `f32` in the shader.
const MAX_EXPOSURE_EV: f32 = 16.0;

/// The linear scale for an exposure in stops. A non-finite value means the
/// default exposure, as if the field had never been set; a finite one is
/// clamped to [`MAX_EXPOSURE_EV`].
pub(crate) fn exposure_scale(exposure_ev: f32) -> f32 {
    let ev = if exposure_ev.is_finite() {
        exposure_ev
    } else {
        DEFAULT_EXPOSURE_EV
    };
    ev.clamp(-MAX_EXPOSURE_EV, MAX_EXPOSURE_EV).exp2()
}

/// CPU copies of the shader's curves. Test-only: the runtime path is the
/// shader, and these exist so the curves are tested without a GPU and the
/// shader can be held to them.
#[cfg(test)]
#[expect(
    clippy::excessive_precision,
    reason = "the constants are copied digit for digit from colour.wgsl and their published sources"
)]
pub(crate) mod mirror {
    use super::exposure_scale;
    use crate::{PostSettings, Tonemapper};

    type Vec3 = [f32; 3];
    /// Three columns, so `apply(m, v)` matches WGSL's `m * v`.
    type Mat3 = [Vec3; 3];

    fn apply(m: &Mat3, v: Vec3) -> Vec3 {
        [0, 1, 2].map(|row| m[0][row] * v[0] + m[1][row] * v[1] + m[2][row] * v[2])
    }

    const ACES_INPUT: Mat3 = [
        [0.59719, 0.07600, 0.02840],
        [0.35458, 0.90834, 0.13383],
        [0.04823, 0.01566, 0.83777],
    ];
    const ACES_OUTPUT: Mat3 = [
        [1.60475, -0.10208, -0.00327],
        [-0.53108, 1.10813, -0.07276],
        [-0.07367, -0.00605, 1.07602],
    ];

    /// Mirror of `aces_fitted` in `colour.wgsl`.
    pub(crate) fn aces_fitted(colour: Vec3) -> Vec3 {
        let fitted = apply(&ACES_INPUT, colour).map(|v| {
            let a = v * (v + 0.0245786) - 0.000090537;
            let b = v * (0.983729 * v + 0.4329510) + 0.238081;
            a / b
        });
        apply(&ACES_OUTPUT, fitted).map(|v| v.clamp(0.0, 1.0))
    }

    const AGX_INSET: Mat3 = [
        [0.842479062253094, 0.0423282422610123, 0.0423756549057051],
        [0.0784335999999992, 0.878468636469772, 0.0784336],
        [0.0792237451477643, 0.0791661274605434, 0.879142973793104],
    ];
    const AGX_OUTSET: Mat3 = [
        [1.19687900512017, -0.0528968517574562, -0.0529716355144438],
        [-0.0980208811401368, 1.15190312990417, -0.0980434501171241],
        [-0.0990297440797205, -0.0989611768448433, 1.15107367264116],
    ];
    const AGX_MIN_EV: f32 = -12.47393;
    const AGX_MAX_EV: f32 = 4.026069;

    /// Mirror of `agx` in `colour.wgsl`.
    pub(crate) fn agx(colour: Vec3) -> Vec3 {
        let inset = apply(&AGX_INSET, colour.map(|v| v.max(1e-10)));
        let curved = inset.map(|v| {
            let x = (v.log2().clamp(AGX_MIN_EV, AGX_MAX_EV) - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV);
            let x2 = x * x;
            let x4 = x2 * x2;
            15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232
        });
        apply(&AGX_OUTSET, curved).map(|v| v.max(0.0).powf(2.2).clamp(0.0, 1.0))
    }

    /// The sRGB transfer function, as `linear_to_srgb` in `colour.wgsl` and as the
    /// GPU applies it on an sRGB target.
    pub(crate) fn linear_to_srgb(v: f32) -> f32 {
        if v <= 0.0031308 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    }

    /// What the tonemap pass shows for a linear scene colour, as an 8-bit display
    /// code per channel. The reference the GPU test checks the shader against.
    /// Mirror of `finite_scene` in `colour.wgsl`.
    pub(crate) fn finite_scene(colour: Vec3) -> Vec3 {
        colour.map(|v| if v.is_nan() { 0.0 } else { v.clamp(0.0, 65504.0) })
    }

    pub(crate) fn display_code(scene: Vec3, settings: &PostSettings) -> [u8; 3] {
        let scale = exposure_scale(settings.exposure_ev);
        let exposed = finite_scene(scene).map(|v| v * scale);
        let display = match settings.tonemapper {
            Tonemapper::AcesFitted => aces_fitted(exposed),
            Tonemapper::Agx => agx(exposed),
        };
        let grade = settings.grade;
        // Graded in the encoded space, as the shader does.
        let graded = [0, 1, 2].map(|i| {
            super::grade_channel(
                linear_to_srgb(display[i]),
                grade.lift[i],
                grade.gamma[i],
                grade.gain[i],
            )
        });
        graded.map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
    }
}

/// The shader's `curve` index for a tonemapper; matches `tonemap.wgsl`.
fn operator(tonemapper: Tonemapper) -> u32 {
    match tonemapper {
        Tonemapper::AcesFitted => 0,
        Tonemapper::Agx => 1,
    }
}

/// The `Post` uniform for a frame.
pub(crate) fn post_bytes(settings: &PostSettings) -> [u8; POST_UNIFORM_SIZE] {
    let mut out = [0; POST_UNIFORM_SIZE];
    out[0..4].copy_from_slice(&exposure_scale(settings.exposure_ev).to_le_bytes());
    out[4..8].copy_from_slice(&operator(settings.tonemapper).to_le_bytes());
    let bloom = if super::bloom::enabled(&settings.bloom) {
        settings.bloom.intensity
    } else {
        0.0
    };
    out[8..12].copy_from_slice(&bloom.to_le_bytes());
    out[12..16].copy_from_slice(&u32::from(settings.auto_exposure.is_some()).to_le_bytes());
    let grade = settings.grade;
    let inverse_gamma = grade.gamma.map(|gamma| 1.0 / sanitised_gamma(gamma));
    // A broken lift or gain falls back to neutral rather than reaching the
    // shader, where a NaN would turn pixels black or white.
    let neutral = |values: [f32; 3], fallback: f32| values.map(|v| if v.is_finite() { v } else { fallback });
    let lift = neutral(grade.lift, 0.0);
    let gain = neutral(grade.gain, 1.0);
    for (offset, values) in [(16, lift), (32, inverse_gamma), (48, gain)] {
        for (index, value) in values.iter().enumerate() {
            let at = offset + index * 4;
            out[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    out
}

/// A gamma the shader can divide by: non-finite means 1, and small values
/// are clamped.
fn sanitised_gamma(gamma: f32) -> f32 {
    if gamma.is_finite() {
        gamma.max(MIN_GAMMA)
    } else {
        1.0
    }
}

/// Mirror of `grade` in `colour.wgsl`, on one sRGB-encoded display channel.
#[cfg(test)]
pub(crate) fn grade_channel(encoded: f32, lift: f32, gamma: f32, gain: f32) -> f32 {
    let lifted = gain * (encoded + lift * (1.0 - encoded));
    lifted.max(0.0).powf(1.0 / sanitised_gamma(gamma)).clamp(0.0, 1.0)
}

/// Whether the shader must apply the sRGB encode itself: only when the
/// swapchain format will not.
pub(crate) fn encodes_in_shader(format: wgpu::TextureFormat) -> bool {
    !format.is_srgb()
}

/// Group 0: the HDR scene texture, the `Post` uniform, and the adapted
/// exposure the auto-exposure passes leave.
pub(crate) fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tonemap-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    // Read with `textureLoad`, never filtered.
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(POST_UNIFORM_SIZE as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(super::exposure::STATE_SIZE as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

pub(crate) fn create_post_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("post-uniform"),
        size: POST_UNIFORM_SIZE as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Compiles `source` into the tonemap pipeline for a `format` target.
///
/// Returns the validation message rather than panicking, like the other
/// pipelines, so a hot-reload typo keeps the last good pipeline.
pub(crate) fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    source: &str,
    format: wgpu::TextureFormat,
) -> Result<wgpu::RenderPipeline, String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(TONEMAP),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let constants = [("encode_srgb", if encodes_in_shader(format) { 1.0 } else { 0.0 })];
    let targets = [Some(wgpu::ColorTargetState {
        format,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    })];
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("tonemap-pipeline"),
        layout: Some(layout),
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
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            targets: &targets,
        }),
        multiview_mask: None,
        cache: None,
    });
    match block_on(scope.pop()) {
        Some(error) => Err(error.to_string()),
        None => Ok(pipeline),
    }
}

#[cfg(test)]
mod tests {
    use super::mirror::*;
    use super::*;

    fn close(a: f32, b: f32, tolerance: f32) -> bool {
        (a - b).abs() <= tolerance
    }

    #[test]
    fn aces_matches_its_reference_values() {
        // Worked by hand from the fit: the matrices' rows sum to 1, so grey
        // stays grey and only the rational curve acts on it.
        assert!(close(aces_fitted([0.18; 3])[0], 0.10559, 2e-4));
        assert!(close(aces_fitted([1.0; 3])[0], 0.61913, 2e-4));
        assert_eq!(aces_fitted([0.0; 3]), [0.0; 3], "black stays black");
    }

    #[test]
    fn both_curves_rise_and_stay_within_the_display_range() {
        for curve in [aces_fitted as fn([f32; 3]) -> [f32; 3], agx] {
            let mut previous = -1.0;
            for step in 0..200 {
                let scene = (step as f32 / 10.0 - 12.0).exp2();
                let display = curve([scene; 3])[1];
                assert!((0.0..=1.0).contains(&display), "{display} at {scene}");
                assert!(display >= previous, "falls at {scene}");
                previous = display;
            }
            assert!(
                curve([1.0e4; 3])[0] > 0.95,
                "very bright light reaches near white"
            );
        }
    }

    #[test]
    fn agx_keeps_black_near_black_and_grey_in_the_middle() {
        assert!(agx([0.0; 3])[0] < 0.01);
        let grey = agx([0.18; 3]);
        assert!((0.05..0.4).contains(&grey[0]), "mid grey maps to {}", grey[0]);
        assert!(close(grey[0], grey[2], 1e-3), "grey stays neutral");
    }

    #[test]
    fn the_default_exposure_keeps_mid_grey_where_the_slice_was_authored() {
        // Before tonemapping, 18 percent grey went straight to the display.
        let untonemapped = (linear_to_srgb(0.18) * 255.0).round() as u8;
        let [grey, _, _] = display_code([0.18; 3], &PostSettings::default());
        assert!(
            grey.abs_diff(untonemapped) <= 1,
            "18% grey shows as {grey}, was {untonemapped}"
        );
    }

    #[test]
    fn broken_scene_light_reaches_the_curve_as_something_finite() {
        assert_eq!(finite_scene([f32::NAN, f32::INFINITY, -3.0]), [0.0, 65504.0, 0.0]);
        // Neutral, because ACES mixes channels: an infinite green would brighten
        // red through the input matrix.
        let settings = PostSettings::default();
        assert_eq!(display_code([f32::NAN; 3], &settings), [0; 3], "NaN is black");
        assert_eq!(
            display_code([f32::INFINITY; 3], &settings),
            [255; 3],
            "infinity is white"
        );
    }

    #[test]
    fn the_srgb_encode_matches_the_standard() {
        assert_eq!(linear_to_srgb(0.0), 0.0);
        assert!(close(linear_to_srgb(1.0), 1.0, 1e-6));
        assert!(close(linear_to_srgb(0.214), 0.4999, 1e-3));
        assert!(close(linear_to_srgb(0.002), 0.02584, 1e-5), "the linear toe");
    }

    #[test]
    fn exposure_is_in_stops_and_bounded() {
        assert_eq!(exposure_scale(0.0), 1.0);
        assert_eq!(exposure_scale(1.0), 2.0);
        assert_eq!(exposure_scale(-2.0), 0.25);
        assert_eq!(exposure_scale(100.0), MAX_EXPOSURE_EV.exp2());
        let default = DEFAULT_EXPOSURE_EV.exp2();
        assert_eq!(
            exposure_scale(f32::NAN),
            default,
            "a broken setting means the default"
        );
        assert_eq!(exposure_scale(f32::INFINITY), default);
    }

    fn f32_at(bytes: &[u8], at: usize) -> f32 {
        f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    }

    #[test]
    fn the_post_uniform_carries_scale_curve_and_grade() {
        let bytes = post_bytes(&PostSettings {
            exposure_ev: 1.0,
            tonemapper: Tonemapper::Agx,
            grade: crate::Grade {
                lift: [0.1, 0.0, 0.0],
                gamma: [2.0, 1.0, 1.0],
                gain: [1.0, 1.0, 0.5],
            },
            auto_exposure: None,
            bloom: crate::Bloom {
                intensity: 0.25,
                ..crate::Bloom::default()
            },
        });
        assert_eq!(f32_at(&bytes, 0), 2.0);
        assert_eq!(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]), 1);
        assert_eq!(f32_at(&bytes, 8), 0.25, "bloom intensity");
        assert_eq!(f32_at(&bytes, 16), 0.1, "lift.r");
        assert_eq!(f32_at(&bytes, 32), 0.5, "inverse gamma.r");
        assert_eq!(f32_at(&bytes, 48 + 8), 0.5, "gain.b");
    }

    #[test]
    fn the_default_grade_changes_nothing() {
        for display in [0.0, 0.18, 0.5, 1.0] {
            assert_eq!(grade_channel(display, 0.0, 1.0, 1.0), display);
        }
    }

    #[test]
    fn lift_raises_blacks_gain_scales_whites_and_gamma_bends_mids() {
        assert!(
            (grade_channel(0.0, 0.1, 1.0, 1.0) - 0.1).abs() < 1e-6,
            "lift moves black"
        );
        assert_eq!(grade_channel(1.0, 0.1, 1.0, 1.0), 1.0, "and leaves white");
        assert_eq!(grade_channel(1.0, 0.0, 1.0, 0.5), 0.5, "gain scales white");
        assert!(
            grade_channel(0.25, 0.0, 2.0, 1.0) > 0.25,
            "gamma above 1 brightens mids"
        );
        assert_eq!(
            grade_channel(0.5, 0.0, 0.0, 1.0),
            grade_channel(0.5, 0.0, MIN_GAMMA, 1.0),
            "zero gamma is clamped"
        );
        assert_eq!(
            grade_channel(0.5, 0.0, f32::NAN, 1.0),
            0.5,
            "a broken gamma means 1"
        );
    }

    #[test]
    fn only_a_non_srgb_target_is_encoded_in_the_shader() {
        assert!(!encodes_in_shader(wgpu::TextureFormat::Bgra8UnormSrgb));
        assert!(!encodes_in_shader(wgpu::TextureFormat::Rgba8UnormSrgb));
        assert!(encodes_in_shader(wgpu::TextureFormat::Bgra8Unorm));
    }
}
