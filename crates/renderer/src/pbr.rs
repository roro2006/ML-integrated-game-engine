//! The PBR material: its bind group, its pipelines, and the textures that
//! stand in for empty slots.
//!
//! Material parameters live in a uniform buffer per material, in bind group 2
//! with the material's textures and sampler. Per-draw data stays in group 1,
//! where the base-colour tint and flash already are (ADR-0017), so a material
//! shared by many draws is bound once and its parameters are uploaded once.

use crate::block_on::block_on;
use crate::frame::{AlphaMode, MaterialDesc};
use crate::permutation::PermutationKey;
use crate::shaders::PBR_LIT;
use crate::texture::GpuTexture;

/// `MaterialParams` in `pbr_lit.wgsl`: emissive (vec4) and factors (vec4).
pub(crate) const MATERIAL_PARAMS_SIZE: usize = 32;

/// Packs a material's parameters: emissive rgb with the normal scale in w,
/// then metallic, roughness, alpha cutoff and occlusion strength.
pub(crate) fn material_params_bytes(desc: &MaterialDesc) -> [u8; MATERIAL_PARAMS_SIZE] {
    let cutoff = match desc.alpha {
        AlphaMode::Opaque => 0.0,
        AlphaMode::Mask { cutoff } => cutoff,
    };
    let [r, g, b] = desc.emissive;
    let values = [
        r,
        g,
        b,
        desc.normal_scale,
        desc.metallic,
        desc.roughness,
        cutoff,
        desc.occlusion_strength,
    ];
    let mut out = [0; MATERIAL_PARAMS_SIZE];
    for (index, value) in values.iter().enumerate() {
        out[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    out
}

/// Group 2: parameters, four textures, one sampler.
pub(crate) fn material_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let texture = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("material-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(MATERIAL_PARAMS_SIZE as u64),
                },
                count: None,
            },
            texture(1),
            texture(2),
            texture(3),
            texture(4),
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// How a material slot reads its texture: the slot decides, not the texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SlotSpace {
    Srgb,
    Linear,
}

impl SlotSpace {
    pub(crate) fn view_format(self) -> wgpu::TextureFormat {
        match self {
            Self::Srgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            Self::Linear => wgpu::TextureFormat::Rgba8Unorm,
        }
    }
}

/// A view of `texture` that reads it in the slot's colour space. Textures are
/// created able to be viewed either way (see `texture::upload`).
pub(crate) fn slot_view(texture: &GpuTexture, space: SlotSpace) -> wgpu::TextureView {
    texture.texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("material-slot"),
        format: Some(space.view_format()),
        ..Default::default()
    })
}

/// 1x1 textures for empty slots, holding each slot's neutral value: white
/// base colour, a flat normal, full occlusion with roughness and metallic of 1
/// so the factors decide, and white emission so the emissive factor does.
#[derive(Debug)]
pub(crate) struct DefaultTextures {
    pub(crate) white: GpuTexture,
    pub(crate) flat_normal: GpuTexture,
}

/// The neutral texels: white, and a tangent-space normal pointing straight out.
pub(crate) const WHITE_TEXEL: [u8; 4] = [255, 255, 255, 255];
pub(crate) const FLAT_NORMAL_TEXEL: [u8; 4] = [128, 128, 255, 255];

/// Compiles the PBR pipeline for one permutation.
pub(crate) fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    source: &str,
    key: PermutationKey,
) -> Result<wgpu::RenderPipeline, String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(PBR_LIT),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let constants = key.constants();
    let vertex_buffers = [Some(crate::pipeline::vertex_layout())];
    let targets = [Some(wgpu::ColorTargetState {
        format: crate::passes::HDR_FORMAT,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    })];
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("pbr-pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            buffers: &vertex_buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            // Cut-out foliage is seen from both sides.
            cull_mode: if key.alpha_mask {
                None
            } else {
                Some(wgpu::Face::Back)
            },
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: crate::pipeline::DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
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

/// CPU copy of `brdf.wgsl`, for tests to hold the shader to.
#[cfg(test)]
pub(crate) mod mirror {
    type Vec3 = [f32; 3];

    const PI: f32 = std::f32::consts::PI;

    fn dot(a: Vec3, b: Vec3) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn normalize(v: Vec3) -> Vec3 {
        let length = dot(v, v).sqrt();
        v.map(|c| c / length)
    }

    fn mix(a: Vec3, b: Vec3, t: f32) -> Vec3 {
        [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t)
    }

    /// Mirror of `brdf_direct` plus `ambient_flat`, the whole lit result of
    /// `pbr_lit.wgsl` for an untextured material.
    #[expect(clippy::too_many_arguments, reason = "mirrors the shader's signature")]
    pub(crate) fn shade(
        normal: Vec3,
        to_view: Vec3,
        to_light: Vec3,
        albedo: Vec3,
        metallic: f32,
        roughness: f32,
        irradiance: Vec3,
        ambient: Vec3,
    ) -> Vec3 {
        let roughness = roughness.clamp(0.045, 1.0);
        let f0 = mix([0.04; 3], albedo, metallic);
        let ambient_light = [0, 1, 2].map(|i| (albedo[i] * (1.0 - metallic) + f0[i]) * ambient[i]);

        let n_dot_l = dot(normal, to_light).max(0.0);
        if n_dot_l <= 0.0 {
            return ambient_light;
        }
        let halfway = normalize([0, 1, 2].map(|i| to_view[i] + to_light[i]));
        let n_dot_v = dot(normal, to_view).max(1e-4);
        let n_dot_h = dot(normal, halfway).max(0.0);
        let v_dot_h = dot(to_view, halfway).max(0.0);
        let alpha = roughness * roughness;
        let alpha2 = alpha * alpha;

        let d = {
            let denominator = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
            alpha2 / (PI * denominator * denominator)
        };
        let v = {
            let view = n_dot_l * (n_dot_v * n_dot_v * (1.0 - alpha2) + alpha2).sqrt();
            let light = n_dot_v * (n_dot_l * n_dot_l * (1.0 - alpha2) + alpha2).sqrt();
            0.5 / (view + light).max(1e-5)
        };
        let fresnel = f0.map(|f| f + (1.0 - f) * (1.0 - v_dot_h).powi(5));
        [0, 1, 2].map(|i| {
            let specular = d * v * fresnel[i];
            let diffuse = (1.0 - fresnel[i]) * (1.0 - metallic) * albedo[i] / PI;
            (diffuse + specular) * irradiance[i] * n_dot_l + ambient_light[i]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(bytes: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    }

    #[test]
    fn material_params_land_where_the_shader_reads_them() {
        let bytes = material_params_bytes(&MaterialDesc {
            emissive: [0.1, 0.2, 0.3],
            normal_scale: 0.5,
            metallic: 1.0,
            roughness: 0.25,
            occlusion_strength: 0.75,
            alpha: AlphaMode::Mask { cutoff: 0.4 },
            ..MaterialDesc::default()
        });
        assert_eq!(read_f32(&bytes, 0), 0.1, "emissive.r");
        assert_eq!(read_f32(&bytes, 12), 0.5, "normal scale in emissive.w");
        assert_eq!(read_f32(&bytes, 16), 1.0, "metallic");
        assert_eq!(read_f32(&bytes, 20), 0.25, "roughness");
        assert_eq!(read_f32(&bytes, 24), 0.4, "alpha cutoff");
        assert_eq!(read_f32(&bytes, 28), 0.75, "occlusion strength");
        assert_eq!(
            read_f32(&material_params_bytes(&MaterialDesc::default()), 24),
            0.0
        );
    }

    #[test]
    fn a_white_dielectric_lit_head_on_matches_the_flat_shader() {
        // Light, view and normal aligned, rough enough that specular is small:
        // the PBR result should sit near the flat Lambert result of 1.
        let shaded = mirror::shade(
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0; 3],
            0.0,
            1.0,
            [PI_F32; 3],
            [0.0; 3],
        );
        assert!((shaded[0] - 1.0).abs() < 0.1, "got {}", shaded[0]);
    }

    const PI_F32: f32 = std::f32::consts::PI;

    #[test]
    fn metals_have_no_diffuse_and_tint_their_reflection() {
        let gold = [1.0, 0.78, 0.34];
        // Light from the side, viewed head on: almost no specular reaches the
        // eye, and a metal has no diffuse, so it is nearly dark.
        let side = mirror::shade(
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.8, 0.0, 0.6],
            gold,
            1.0,
            0.2,
            [PI_F32; 3],
            [0.0; 3],
        );
        assert!(side.iter().all(|&channel| channel < 0.05), "{side:?}");
        // Mirror direction: a bright, gold-tinted highlight.
        let highlight = mirror::shade(
            [0.0, 0.0, 1.0],
            [0.6, 0.0, 0.8],
            [-0.6, 0.0, 0.8],
            gold,
            1.0,
            0.2,
            [PI_F32; 3],
            [0.0; 3],
        );
        assert!(
            highlight[0] > highlight[2] * 2.0,
            "the reflection takes the metal's colour: {highlight:?}"
        );
    }

    #[test]
    fn slots_read_in_their_own_colour_space() {
        assert_eq!(SlotSpace::Srgb.view_format(), wgpu::TextureFormat::Rgba8UnormSrgb);
        assert_eq!(SlotSpace::Linear.view_format(), wgpu::TextureFormat::Rgba8Unorm);
    }
}
