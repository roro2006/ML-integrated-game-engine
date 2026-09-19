//! Debug line rendering: the vertex format, its GPU buffer, and its pipeline.
//!
//! Debug geometry is an overlay on the scene, not part of it, and the pipeline
//! state here is what makes it behave like one:
//!
//! - **Depth test enabled, depth write disabled.** Testing means a line behind
//!   a wall is hidden by the wall, so the overlay reads as being *in* the
//!   world. Not writing means the line never occludes anything, so a bounds box
//!   drawn around a character cannot hide the character inside it, and two
//!   overlapping debug shapes never z-fight. Enabling writes here would make
//!   debug drawing change the picture it is supposed to be describing.
//! - **No culling.** A line has no winding and therefore no facing; `Cull::Back`
//!   is meaningless for `LineList` and only invites confusion later.
//!
//! The whole pass is one draw call over one vertex buffer, whatever mix of
//! boxes, axes, and rays it contains: colour rides on the vertex and positions
//! arrive in world space, so nothing is per-shape.
//!
//! As in `mesh.rs`, the byte conversion is a free function kept apart from the
//! upload so it can be asserted without a device.

use crate::block_on::block_on;
use crate::pipeline::DEPTH_FORMAT;
use crate::shaders::DEBUG_LINE;

/// One end of a debug line.
///
/// Positions are **world space**: the debug pipeline has no per-draw uniform,
/// so whatever produces a line has already applied its transform.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DebugVertex {
    pub position: [f32; 3],
    /// Linear RGBA, like [`MaterialDesc::base_color`](crate::MaterialDesc). The
    /// surface is sRGB and the GPU converts on write, so these are linear
    /// values, not sRGB ones.
    pub color: [f32; 4],
}

impl DebugVertex {
    pub const fn new(position: [f32; 3], color: [f32; 4]) -> Self {
        Self { position, color }
    }
}

/// Bytes per vertex: `position` (12) + `color` (16).
///
/// Unlike a uniform block this has no 16-byte alignment rule to satisfy — a
/// vertex attribute only needs its offset aligned to its component size — so
/// the layout is packed tight at 28 bytes rather than padded to 32.
pub(crate) const DEBUG_VERTEX_SIZE: usize = 28;

/// Vertices the debug buffer starts out able to hold.
///
/// 512 lines: enough for a screenful of bounds boxes and axis gizmos without
/// reallocating on the frame debug drawing is first switched on. The buffer
/// grows from here and is never shrunk, exactly like the per-draw uniform
/// buffer.
pub(crate) const INITIAL_VERTEX_CAPACITY: u32 = 1024;

const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 2] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 12,
        shader_location: 1,
    },
];

pub(crate) fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: DEBUG_VERTEX_SIZE as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &VERTEX_ATTRIBUTES,
    }
}

/// How many vertices this frame's debug pass should draw, or `None` when there
/// is no pass to record.
///
/// Two separate jobs, deliberately in one pure function so the decision is
/// testable without a device:
///
/// - An empty list means the pass is skipped entirely — no buffer growth, no
///   upload, no pipeline switch. A frame that draws no debug lines must cost
///   nothing.
/// - An **odd** vertex count is rounded down. `LineList` consumes vertices in
///   pairs and wgpu rejects a `draw` whose count is odd, so a caller that
///   somehow submitted a half-finished line would otherwise lose the whole
///   frame to a validation error instead of that one line.
pub(crate) fn draw_vertex_count(vertices: &[DebugVertex]) -> Option<u32> {
    // Saturating rather than wrapping: a list longer than u32::MAX is not
    // reachable through the vertex budget, and clamping beats truncating.
    let count = u32::try_from(vertices.len()).unwrap_or(u32::MAX);
    let pairs = count - count % 2;
    (pairs > 0).then_some(pairs)
}

/// Little-endian vertex bytes, written into a caller-owned buffer.
///
/// `bytemuck` is not a dependency and `unsafe` is forbidden workspace-wide, so
/// this is written out by hand like the uniform blocks in `uniforms.rs`. The
/// buffer is passed in so the allocation survives between frames.
pub(crate) fn pack_vertices(out: &mut Vec<u8>, vertices: &[DebugVertex]) {
    out.clear();
    out.reserve(vertices.len() * DEBUG_VERTEX_SIZE);
    for vertex in vertices {
        for value in vertex.position {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for value in vertex.color {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
}

/// Creates a vertex buffer holding `capacity` debug vertices.
pub(crate) fn create_vertex_buffer(device: &wgpu::Device, capacity: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("debug-line-vertices"),
        // `max(1)`: a zero-sized buffer is a validation error, and the pass is
        // skipped at zero vertices anyway.
        size: u64::from(capacity.max(1)) * DEBUG_VERTEX_SIZE as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Compiles `source` and builds the debug line pipeline.
///
/// `layout` binds group 0 only. Returns the validation message rather than
/// panicking, for the same reason as `pipeline::create_pipeline`: a caller
/// holding a working pipeline can keep drawing with it.
pub(crate) fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    source: &str,
    color_format: wgpu::TextureFormat,
) -> Result<wgpu::RenderPipeline, String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(DEBUG_LINE),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let vertex_buffers = [Some(vertex_layout())];
    let targets = [Some(wgpu::ColorTargetState {
        format: color_format,
        // Opaque, like the lit pass. Alpha-blended debug lines would need
        // back-to-front sorting that nothing does, and the alpha channel is
        // kept in the vertex for a later blended variant rather than being
        // silently ignored in a way that looks broken.
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    })];
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("debug-line-pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &vertex_buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            strip_index_format: None,
            // Winding is meaningless for a line; see the module comment.
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            // Same format and compare as the lit pass, because this shares that
            // pass's depth attachment.
            format: DEPTH_FORMAT,
            // The whole point: test against the scene, never write into it.
            depth_write_enabled: Some(false),
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

/// Whether a set of changed shader names means the debug pipeline is stale.
///
/// The mirror of `pipeline::lit_shader_changed`: an edit to one shader must not
/// rebuild the other's pipeline.
pub(crate) fn debug_shader_changed(changed: &[String]) -> bool {
    changed.iter().any(|name| name == DEBUG_LINE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uniforms::grown_capacity;

    fn read_f32(bytes: &[u8], offset: usize) -> f32 {
        let mut word = [0_u8; 4];
        word.copy_from_slice(&bytes[offset..offset + 4]);
        f32::from_le_bytes(word)
    }

    #[test]
    fn the_vertex_layout_matches_the_packed_vertex() {
        let layout = vertex_layout();
        assert_eq!(layout.array_stride, DEBUG_VERTEX_SIZE as u64);
        assert_eq!(layout.array_stride, 28, "3 position floats plus 4 colour floats");
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Vertex);

        let attributes: Vec<_> = layout
            .attributes
            .iter()
            .map(|attribute| (attribute.shader_location, attribute.offset, attribute.format))
            .collect();
        assert_eq!(
            attributes,
            vec![
                (0, 0, wgpu::VertexFormat::Float32x3),
                (1, 12, wgpu::VertexFormat::Float32x4),
            ]
        );
    }

    #[test]
    fn packed_vertices_land_at_their_declared_offsets() {
        let mut packed = Vec::new();
        pack_vertices(
            &mut packed,
            &[
                DebugVertex::new([1.0, 2.0, 3.0], [0.25, 0.5, 0.75, 1.0]),
                DebugVertex::new([4.0, 5.0, 6.0], [0.0, 0.0, 0.0, 0.5]),
            ],
        );

        assert_eq!(packed.len(), 2 * DEBUG_VERTEX_SIZE);
        assert_eq!(read_f32(&packed, 0), 1.0, "position.x");
        assert_eq!(read_f32(&packed, 8), 3.0, "position.z");
        assert_eq!(read_f32(&packed, 12), 0.25, "colour starts at 12");
        assert_eq!(read_f32(&packed, 24), 1.0, "colour.a");
        // The second vertex starts exactly one stride in, with no padding.
        assert_eq!(read_f32(&packed, DEBUG_VERTEX_SIZE), 4.0);
        assert_eq!(read_f32(&packed, DEBUG_VERTEX_SIZE + 24), 0.5);
    }

    #[test]
    fn packing_reuses_the_callers_allocation() {
        let mut packed = vec![0xAB; 4096];
        pack_vertices(&mut packed, &[]);
        assert!(packed.is_empty(), "a frame with no lines packs to nothing");
        assert!(packed.capacity() >= 4096, "the allocation should survive");
    }

    #[test]
    fn no_lines_means_no_pass() {
        assert_eq!(
            draw_vertex_count(&[]),
            None,
            "an empty frame records no debug pass"
        );
    }

    #[test]
    fn an_odd_vertex_is_dropped_rather_than_failing_the_draw() {
        let vertex = DebugVertex::new([0.0; 3], [1.0; 4]);
        assert_eq!(draw_vertex_count(&[vertex; 2]), Some(2));
        assert_eq!(
            draw_vertex_count(&[vertex; 3]),
            Some(2),
            "LineList rejects an odd count; the half line goes, not the frame"
        );
        assert_eq!(draw_vertex_count(&[vertex; 4]), Some(4));
        assert_eq!(
            draw_vertex_count(&[vertex; 1]),
            None,
            "a single vertex is half a line, so there is still nothing to draw"
        );
    }

    /// The debug buffer grows with the same one-way doubling as the per-draw
    /// uniform buffer, and for the same reason: debug line counts wobble frame
    /// to frame, and a frame that once needed the space will need it again.
    #[test]
    fn the_vertex_buffer_doubles_and_never_shrinks() {
        assert_eq!(
            grown_capacity(INITIAL_VERTEX_CAPACITY, 100),
            INITIAL_VERTEX_CAPACITY
        );
        assert_eq!(grown_capacity(INITIAL_VERTEX_CAPACITY, 1024), 1024);
        assert_eq!(grown_capacity(INITIAL_VERTEX_CAPACITY, 1025), 2048);
        assert_eq!(grown_capacity(1024, 5000), 8192);
    }

    /// `write_buffer` offsets and sizes must be multiples of 4, so a 28-byte
    /// stride has to keep every buffer size a multiple of 4 at any capacity.
    #[test]
    fn buffer_sizes_stay_copy_aligned() {
        for capacity in [0_u64, 1, 2, 3, 1024, 65536] {
            let size = capacity.max(1) * DEBUG_VERTEX_SIZE as u64;
            assert_eq!(size % 4, 0, "capacity {capacity} produces an unaligned buffer");
            assert!(size > 0, "a zero-sized buffer is a validation error");
        }
    }

    #[test]
    fn only_an_edit_to_the_debug_shader_rebuilds_the_debug_pipeline() {
        assert!(!debug_shader_changed(&[]));
        assert!(debug_shader_changed(&[DEBUG_LINE.to_owned()]));
        assert!(
            !debug_shader_changed(&[crate::shaders::LIT.to_owned()]),
            "an edit to the lit shader must not rebuild this pipeline"
        );
    }

    /// The claims that need a driver: that `LineList` with depth-test-on,
    /// depth-write-off and no culling is a pipeline the GPU accepts, and that
    /// the shipped `debug_line.wgsl` matches the shared globals layout — in
    /// particular that declaring the full 112-byte `Globals` struct satisfies
    /// the layout's `min_binding_size`.
    ///
    /// Ignored so CI and headless development need no GPU. Run it on a machine
    /// with a working Vulkan driver:
    /// `cargo test -p renderer -- --ignored`.
    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn the_shipped_debug_shader_builds_against_the_shared_globals_layout() {
        use crate::pipeline::globals_layout;
        use crate::shaders::ShaderLibrary;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("this test is ignored unless an adapter exists");
        let (device, _queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("debug-line-test-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        }))
        .expect("the adapter should give up a device");

        let globals = globals_layout(&device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("debug-line-pipeline-layout"),
            // Group 0 only: debug vertices are already in world space.
            bind_group_layouts: &[Some(&globals)],
            immediate_size: 0,
        });

        let source = ShaderLibrary::new(None)
            .source(DEBUG_LINE)
            .expect("the debug shader is embedded")
            .into_owned();
        assert!(
            create_pipeline(&device, &layout, &source, wgpu::TextureFormat::Bgra8UnormSrgb).is_ok(),
            "the shipped debug shader must match the globals bind group layout"
        );
    }
}
