//! The GPU side of cascaded shadows (C1): the atlas, the per-cascade
//! uniforms the shadow pass draws with, and the `ShadowData` block the lit
//! shaders sample with. The fitting itself is in `shadow`.

use glam::{Mat4, Vec3};

use crate::block_on::block_on;
use crate::frame::{MAX_CASCADES, ShadowSettings};
use crate::pipeline::vertex_layout;
use crate::shaders::SHADOW_DEPTH;
use crate::shadow::Cascade;

/// The atlas format: the same full-precision depth as the scene.
pub(crate) const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// One cascade's view-projection, the whole of the shadow pass's uniform.
const CASCADE_SIZE: u64 = 64;

/// `ShadowData` as `shadow.wgsl` declares it: five view-projections (320),
/// the cascades' far distances and texel sizes packed four to a `vec4`
/// (two `vec4`s each, 64), and the atlas description (16).
pub(crate) const SHADOW_DATA_SIZE: usize = 400;

/// Slope-scaled bias in the shadow pipeline. The normal offset applied when
/// sampling does most of the work; this removes the acne left on surfaces
/// nearly parallel to the light.
const SLOPE_BIAS: f32 = 1.5;

/// How the cascades sit in the atlas: side by side, each square.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AtlasLayout {
    /// Cascades in the atlas; 0 when shadows are off.
    pub(crate) cascades: u32,
    /// Texels along one side of a cascade.
    pub(crate) resolution: u32,
}

impl AtlasLayout {
    /// The layout for `settings`, shrinking the per-cascade resolution so the
    /// atlas fits `max_dimension`. Shadows off gives a 1x1 atlas, which keeps
    /// the shaders' bindings valid without a second code path.
    pub(crate) fn for_settings(settings: &ShadowSettings, max_dimension: u32) -> Self {
        let cascades = settings.cascades.min(MAX_CASCADES);
        if cascades == 0 || settings.resolution == 0 || max_dimension == 0 {
            return Self {
                cascades: 0,
                resolution: 1,
            };
        }
        Self {
            cascades,
            resolution: settings.resolution.min(max_dimension / cascades).max(1),
        }
    }

    /// The atlas size in texels.
    pub(crate) fn size(self) -> (u32, u32) {
        (self.resolution * self.cascades.max(1), self.resolution)
    }

    /// Cascade `index`'s viewport: x, y, width, height in texels.
    pub(crate) fn viewport(self, index: u32) -> [f32; 4] {
        let side = self.resolution as f32;
        [index as f32 * side, 0.0, side, side]
    }
}

/// Whether a bounding sphere can cast into a cascade: it overlaps the
/// cascade's square seen from the light. Depth is not tested, because the
/// projection already reaches toward the light for casters above the slice.
pub(crate) fn may_cast_into(cascade: &Cascade, resolution: u32, centre: Vec3, radius: f32) -> bool {
    let half_size = cascade.texel_world * resolution as f32 / 2.0;
    if half_size <= 0.0 {
        return false;
    }
    let reach = 1.0 + radius / half_size;
    let clip = cascade.view_proj.project_point3(centre);
    clip.x.abs() <= reach && clip.y.abs() <= reach
}

/// A mesh's local bounds as a world-space sphere under `model`.
pub(crate) fn world_sphere(model: &[[f32; 4]; 4], centre: [f32; 3], half_extents: [f32; 3]) -> (Vec3, f32) {
    let model = Mat4::from_cols_array_2d(model);
    let scale = model
        .x_axis
        .truncate()
        .length()
        .max(model.y_axis.truncate().length())
        .max(model.z_axis.truncate().length());
    (
        model.transform_point3(Vec3::from(centre)),
        Vec3::from(half_extents).length() * scale,
    )
}

fn put_f32(out: &mut [u8], offset: usize, value: f32) {
    if let Some(slot) = out.get_mut(offset..offset + 4) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
}

fn put_mat4(out: &mut [u8], offset: usize, matrix: &Mat4) {
    for (slot, value) in matrix.to_cols_array().into_iter().enumerate() {
        put_f32(out, offset + slot * 4, value);
    }
}

/// The `ShadowData` block for this frame's cascades.
pub(crate) fn shadow_data_bytes(cascades: &[Cascade], layout: AtlasLayout) -> [u8; SHADOW_DATA_SIZE] {
    let mut out = [0_u8; SHADOW_DATA_SIZE];
    let count = cascades.len().min(MAX_CASCADES as usize);
    for (index, cascade) in cascades.iter().take(count).enumerate() {
        put_mat4(&mut out, index * 64, &cascade.view_proj);
        put_f32(&mut out, 320 + index * 4, cascade.far);
        put_f32(&mut out, 352 + index * 4, cascade.texel_world);
    }
    let (width, height) = layout.size();
    put_f32(&mut out, 384, count as f32);
    put_f32(&mut out, 388, layout.resolution as f32);
    put_f32(&mut out, 392, 1.0 / width as f32);
    put_f32(&mut out, 396, 1.0 / height as f32);
    out
}

/// Each cascade's view-projection at `stride` bytes apart, for the shadow
/// pass's dynamic offsets.
pub(crate) fn cascade_bytes(cascades: &[Cascade], stride: u32) -> Vec<u8> {
    let stride = stride as usize;
    let mut out = vec![0_u8; stride * cascades.len()];
    for (index, cascade) in cascades.iter().enumerate() {
        put_mat4(&mut out, index * stride, &cascade.view_proj);
    }
    out
}

/// The atlas texture, its view and its layout.
#[derive(Debug)]
pub(crate) struct Atlas {
    pub(crate) layout: AtlasLayout,
    /// Read back by the GPU tests; the view alone keeps it alive in a frame.
    #[cfg_attr(not(test), expect(dead_code, reason = "only GPU tests read the atlas back"))]
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
}

impl Atlas {
    pub(crate) fn new(device: &wgpu::Device, layout: AtlasLayout) -> Self {
        let (width, height) = layout.size();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow-atlas"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADOW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            layout,
            texture,
            view,
        }
    }
}

/// The comparison sampler shadow lookups use: hardware 2x2 PCF per tap.
pub(crate) fn comparison_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("shadow-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..wgpu::SamplerDescriptor::default()
    })
}

/// Per-cascade uniforms for the shadow pass: one view-projection per
/// cascade, selected with a dynamic offset.
#[derive(Debug)]
pub(crate) struct CascadeUniforms {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) bind_group: wgpu::BindGroup,
    /// Bytes between cascades: the uniform offset alignment.
    pub(crate) stride: u32,
}

impl CascadeUniforms {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let alignment = device.limits().min_uniform_buffer_offset_alignment.max(1);
        let stride = (CASCADE_SIZE as u32).div_ceil(alignment) * alignment;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow-cascades"),
            size: u64::from(stride) * u64::from(MAX_CASCADES),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow-cascade-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(CASCADE_SIZE),
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow-cascade-bind-group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(CASCADE_SIZE),
                }),
            }],
        });
        Self {
            buffer,
            layout,
            bind_group,
            stride,
        }
    }
}

/// Everything the renderer keeps for shadows between frames.
#[derive(Debug)]
pub(crate) struct Shadows {
    pub(crate) atlas: Atlas,
    /// The `ShadowData` uniform, bound in group 0 for the lit shaders.
    pub(crate) data_buffer: wgpu::Buffer,
    pub(crate) sampler: wgpu::Sampler,
    pub(crate) cascade_uniforms: CascadeUniforms,
    pub(crate) pipeline_layout: wgpu::PipelineLayout,
    pub(crate) pipeline: wgpu::RenderPipeline,
    /// This frame's cascades; empty when shadows are off.
    pub(crate) cascades: Vec<Cascade>,
}

impl Shadows {
    pub(crate) fn new(
        device: &wgpu::Device,
        draw_layout: &wgpu::BindGroupLayout,
        source: &str,
        layout: AtlasLayout,
    ) -> Result<Self, String> {
        let cascade_uniforms = CascadeUniforms::new(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow-depth-pipeline-layout"),
            bind_group_layouts: &[Some(&cascade_uniforms.layout), Some(draw_layout)],
            immediate_size: 0,
        });
        let pipeline = create_pipeline(device, &pipeline_layout, source)?;
        Ok(Self {
            atlas: Atlas::new(device, layout),
            data_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shadow-data"),
                size: SHADOW_DATA_SIZE as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            sampler: comparison_sampler(device),
            cascade_uniforms,
            pipeline_layout,
            pipeline,
            cascades: Vec::new(),
        })
    }
}

/// Compiles `source` and builds the shadow depth pipeline. `layout` is the
/// cascade group then the draw group.
pub(crate) fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    source: &str,
) -> Result<wgpu::RenderPipeline, String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(SHADOW_DEPTH),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shadow-depth-pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(vertex_layout())],
        },
        // No culling: open meshes and single-sided walls must still cast.
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: SHADOW_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: 0,
                slope_scale: SLOPE_BIAS,
                clamp: 0.0,
            },
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: None,
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
    use super::*;

    #[test]
    fn the_atlas_shrinks_cascades_to_fit_the_device() {
        let settings = ShadowSettings::default();
        assert_eq!(
            AtlasLayout::for_settings(&settings, 16384),
            AtlasLayout {
                cascades: 4,
                resolution: 2048
            }
        );
        let floor = AtlasLayout::for_settings(&settings, 2048);
        assert_eq!(floor.resolution, 512, "the downlevel floor fits four cascades");
        assert!(floor.size().0 <= 2048);
    }

    #[test]
    fn shadows_off_still_has_a_bindable_atlas() {
        for settings in [
            ShadowSettings {
                cascades: 0,
                ..ShadowSettings::default()
            },
            ShadowSettings {
                resolution: 0,
                ..ShadowSettings::default()
            },
        ] {
            let layout = AtlasLayout::for_settings(&settings, 8192);
            assert_eq!(layout.cascades, 0);
            assert_eq!(layout.size(), (1, 1));
        }
    }

    #[test]
    fn viewports_sit_side_by_side() {
        let layout = AtlasLayout {
            cascades: 3,
            resolution: 1024,
        };
        assert_eq!(layout.size(), (3072, 1024));
        assert_eq!(layout.viewport(2), [2048.0, 0.0, 1024.0, 1024.0]);
    }

    /// A 20 m square seen straight down, centred on the origin.
    fn cascade() -> Cascade {
        let view = glam::camera::rh::view::look_to_mat4(Vec3::ZERO, Vec3::NEG_Y, Vec3::Z);
        let projection = glam::camera::rh::proj::directx::orthographic(-10.0, 10.0, -10.0, 10.0, -50.0, 50.0);
        Cascade {
            view_proj: projection * view,
            far: 10.0,
            texel_world: 20.0 / 1024.0,
        }
    }

    #[test]
    fn a_caster_overlapping_the_cascade_square_is_kept_and_one_beside_it_is_not() {
        let cascade = cascade();
        assert!(may_cast_into(&cascade, 1024, Vec3::ZERO, 1.0));
        assert!(
            may_cast_into(&cascade, 1024, Vec3::new(10.5, 0.0, 0.0), 1.0),
            "overlaps the edge"
        );
        assert!(!may_cast_into(&cascade, 1024, Vec3::new(12.0, 0.0, 0.0), 1.0));
        assert!(
            may_cast_into(&cascade, 1024, Vec3::new(0.0, 40.0, 0.0), 1.0),
            "height toward the light does not cull"
        );
    }

    #[test]
    fn a_scaled_mesh_has_a_scaled_sphere() {
        let model = Mat4::from_scale_rotation_translation(
            Vec3::new(1.0, 3.0, 1.0),
            glam::Quat::IDENTITY,
            Vec3::new(5.0, 0.0, 0.0),
        )
        .to_cols_array_2d();
        let (centre, radius) = world_sphere(&model, [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]);
        assert!((centre - Vec3::new(5.0, 3.0, 0.0)).length() < 1e-5);
        assert!((radius - 3.0_f32.sqrt() * 3.0).abs() < 1e-5);
    }

    fn read(bytes: &[u8], offset: usize) -> f32 {
        let mut word = [0_u8; 4];
        word.copy_from_slice(&bytes[offset..offset + 4]);
        f32::from_le_bytes(word)
    }

    #[test]
    fn shadow_data_lands_at_the_offsets_the_shader_declares() {
        let mut cascades = vec![cascade(), cascade()];
        cascades[1].far = 30.0;
        let layout = AtlasLayout {
            cascades: 2,
            resolution: 1024,
        };
        let bytes = shadow_data_bytes(&cascades, layout);
        assert_eq!(read(&bytes, 64), cascades[1].view_proj.to_cols_array()[0]);
        assert_eq!(read(&bytes, 324), 30.0);
        assert_eq!(read(&bytes, 352), cascades[0].texel_world);
        assert_eq!(read(&bytes, 384), 2.0, "cascade count");
        assert_eq!(read(&bytes, 388), 1024.0);
        assert_eq!(read(&bytes, 392), 1.0 / 2048.0);
    }

    #[test]
    fn cascade_uniforms_sit_one_stride_apart() {
        let cascades = [cascade(), cascade()];
        let bytes = cascade_bytes(&cascades, 256);
        assert_eq!(bytes.len(), 512);
        assert_eq!(read(&bytes, 256), cascades[1].view_proj.to_cols_array()[0]);
    }
}
