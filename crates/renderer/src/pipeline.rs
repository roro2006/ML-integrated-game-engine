//! The lit pipeline, its bind group layouts, and the depth buffer.
//!
//! Pipeline creation is the one place in the renderer where bad *data* — a
//! shader with a typo in it — can reach the GPU API, so everything here is
//! written to return an error instead of panicking. wgpu reports shader and
//! pipeline validation failures through an error scope; without one pushed, the
//! default handler aborts the process, which is precisely the mid-session death
//! that hot reload exists to avoid.

use crate::block_on::block_on;
use crate::shaders::LIT;
use crate::uniforms::{DRAW_DATA_SIZE, GLOBALS_SIZE};

/// Fixed by `docs/plans/m3-renderer.md`, and matched by every render pass that
/// uses the lit pipeline.
pub(crate) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Interleaved: position, normal, uv, colour, tangent, wind. Offsets match
/// `MeshData::interleaved`. A shader may declare a subset: `lit.wgsl` reads
/// the first four and ignores tangent and wind.
const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 6] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 12,
        shader_location: 1,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x2,
        offset: 24,
        shader_location: 2,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 32,
        shader_location: 3,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 44,
        shader_location: 4,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: 60,
        shader_location: 5,
    },
];

pub(crate) fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: (crate::mesh_data::VERTEX_FLOATS * 4) as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &VERTEX_ATTRIBUTES,
    }
}

/// Bind group 0: per-frame `Globals`.
pub(crate) fn globals_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("globals-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                // Declared so a layout/shader mismatch is caught at pipeline
                // creation rather than becoming a per-draw validation cost.
                min_binding_size: wgpu::BufferSize::new(GLOBALS_SIZE as u64),
            },
            count: None,
        }],
    })
}

/// Bind group 1: per-draw `DrawData`, addressed by dynamic offset.
///
/// One buffer and one bind group serve every draw in the frame; only the offset
/// changes between them. The alternative — a bind group per draw — allocates
/// per frame and is the usual reason a simple forward renderer stalls.
pub(crate) fn draw_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("draw-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                // The *binding* is one `DrawData` (DRAW_DATA_SIZE, 160 bytes)
                // even though the stride between draws is padded up to the
                // device's alignment.
                min_binding_size: wgpu::BufferSize::new(DRAW_DATA_SIZE as u64),
            },
            count: None,
        }],
    })
}

/// Compiles `source` and builds the lit pipeline.
///
/// Returns the validation message rather than panicking, so a caller holding a
/// working pipeline can keep drawing with it.
pub(crate) fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    source: &str,
    color_format: wgpu::TextureFormat,
) -> Result<wgpu::RenderPipeline, String> {
    // Both the module and the pipeline are created inside the scope: a shader
    // can parse cleanly and still be rejected against the layout, and both
    // failures need the same treatment.
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(LIT),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let vertex_buffers = [Some(vertex_layout())];
    let targets = [Some(wgpu::ColorTargetState {
        format: color_format,
        // Opaque geometry only; transparency needs sorting the engine does not
        // do yet, and enabling blending here would hide that.
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    })];
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("lit-pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &vertex_buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
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

/// The depth texture's size, or `None` when there is nothing to render to.
///
/// A minimized window reports a zero-sized surface, and a zero extent is a
/// validation error in `create_texture`. Returning `None` keeps that case in one
/// obvious place instead of spread across the callers.
pub(crate) fn depth_size(width: u32, height: u32) -> Option<(u32, u32)> {
    (width > 0 && height > 0).then_some((width, height))
}

/// Creates the per-frame uniform buffer and the bind group that points at it.
pub(crate) fn create_globals_resources(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("globals-uniform"),
        size: GLOBALS_SIZE as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("globals-bind-group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: wgpu::BufferSize::new(GLOBALS_SIZE as u64),
            }),
        }],
    });
    (buffer, bind_group)
}

/// Creates a per-draw uniform buffer holding `capacity` draws, and its bind
/// group.
///
/// The bind group's binding size is one `DrawData`, not the whole buffer: the
/// dynamic offset selects which slot a draw reads, and declaring the small size
/// is what lets one buffer serve every draw in the frame.
pub(crate) fn create_draw_resources(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    stride: u32,
    capacity: u32,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let size = u64::from(stride) * u64::from(capacity.max(1));
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("draw-uniform"),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("draw-bind-group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: wgpu::BufferSize::new(DRAW_DATA_SIZE as u64),
            }),
        }],
    });
    (buffer, bind_group)
}

/// Whether a set of changed shader names means the lit pipeline is stale.
///
/// Separated from the rebuild itself so the decision can be tested without a
/// device, and so an edit to an unrelated shader cannot trigger a needless
/// rebuild once this crate owns more than one pipeline.
pub(crate) fn lit_shader_changed(changed: &[String]) -> bool {
    changed.iter().any(|name| name == LIT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_vertex_layout_matches_the_interleaved_mesh_format() {
        let layout = vertex_layout();
        assert_eq!(layout.array_stride, 64, "16 floats per vertex");
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Vertex);

        let offsets: Vec<_> = layout
            .attributes
            .iter()
            .map(|attribute| (attribute.shader_location, attribute.offset))
            .collect();
        assert_eq!(offsets, vec![(0, 0), (1, 12), (2, 24), (3, 32), (4, 44), (5, 60)]);
        assert_eq!(layout.attributes[2].format, wgpu::VertexFormat::Float32x2);
        assert_eq!(
            layout.attributes[3].format,
            wgpu::VertexFormat::Float32x3,
            "vertex colour is linear RGB"
        );
        assert_eq!(
            layout.attributes[4].format,
            wgpu::VertexFormat::Float32x4,
            "tangent carries handedness in w"
        );
        assert_eq!(layout.attributes[5].format, wgpu::VertexFormat::Float32);
    }

    #[test]
    fn a_zero_sized_surface_has_no_depth_buffer() {
        assert_eq!(depth_size(1280, 720), Some((1280, 720)));
        assert_eq!(depth_size(0, 720), None, "a minimized window must not build one");
        assert_eq!(depth_size(1280, 0), None);
        assert_eq!(depth_size(0, 0), None);
    }

    #[test]
    fn only_an_edit_to_the_lit_shader_rebuilds_the_lit_pipeline() {
        assert!(!lit_shader_changed(&[]));
        assert!(lit_shader_changed(&[LIT.to_owned()]));
        assert!(lit_shader_changed(&["other.wgsl".to_owned(), LIT.to_owned()]));
        assert!(
            !lit_shader_changed(&["debug-lines.wgsl".to_owned()]),
            "an unrelated shader must not rebuild this pipeline"
        );
    }

    #[test]
    fn the_depth_format_is_the_one_the_plan_fixes() {
        assert_eq!(DEPTH_FORMAT, wgpu::TextureFormat::Depth32Float);
    }

    /// The only claim in this module that cannot be checked without a driver:
    /// that a shader which fails to compile comes back as an error instead of
    /// taking the process down. wgpu's default uncaptured-error handler panics,
    /// so this verifies the error scope in `create_pipeline` actually catches
    /// the failure — and, on the way, that the real `lit.wgsl` matches the bind
    /// group layouts above.
    ///
    /// Ignored so CI and headless development need no GPU. Run it on a machine
    /// with a working Vulkan driver:
    /// `cargo test -p renderer -- --ignored`.
    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn a_broken_shader_is_reported_rather_than_panicking() {
        use crate::shaders::ShaderLibrary;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("this test is ignored unless an adapter exists");
        let (device, _queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pipeline-test-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        }))
        .expect("the adapter should give up a device");

        let globals = globals_layout(&device);
        let draw = draw_layout(&device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lit-pipeline-layout"),
            bind_group_layouts: &[Some(&globals), Some(&draw)],
            immediate_size: 0,
        });
        let format = wgpu::TextureFormat::Bgra8UnormSrgb;

        let source = ShaderLibrary::new(None)
            .source(LIT)
            .expect("the lit shader is embedded")
            .into_owned();
        assert!(
            create_pipeline(&device, &layout, &source, format).is_ok(),
            "the shipped lit shader must match the bind group layouts"
        );

        let error = create_pipeline(&device, &layout, "@vertex fn nonsense(", format)
            .expect_err("a truncated shader cannot compile");
        assert!(!error.is_empty(), "a rejection should explain itself");
    }
}
