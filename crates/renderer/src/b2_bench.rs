//! B2 lighting-path benchmark: clustered forward+ against deferred (ADR-0019).
//!
//! Test-only. Both paths draw the scene from `b2_scene` with the production
//! BRDF, the same light set and the same GPU cluster culling, after the same
//! depth prepass, so the per-pass GPU timestamps compare the lighting
//! structure and nothing else. A forward path without the prepass is timed
//! too, as the evidence for keeping the prepass.
//!
//! Run on a chosen GPU with
//! `B2_GPU=low cargo test -p renderer --release b2_bench -- --ignored --nocapture`
//! (`B2_GPU` unset or `high` picks the high-performance adapter). The printed
//! adapter identity, not the preference, says which hardware class was used.

use crate::b2_scene::{self, BenchLight, Summary};
use crate::{GpuPreference, Renderer, RendererConfig, clustered, gpu_timing, pipeline};

const BENCH_SIZE: (u32, u32) = (2560, 1440);
const WARMUP_FRAMES: u32 = 30;
const MEASURED_FRAMES: u32 = 200;
const LIGHT_COUNTS: [u32; 3] = [64, b2_scene::LIGHT_COUNT, 1024];
/// Cluster culling runs one invocation per cluster in groups of this size;
/// it must match `@workgroup_size` in `light_cull.wgsl`.
const CULL_WORKGROUP: u32 = 64;
const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const GBUFFER_FORMATS: [wgpu::TextureFormat; 3] = [
    wgpu::TextureFormat::Rgba16Float,
    wgpu::TextureFormat::Rgb10a2Unorm,
    wgpu::TextureFormat::Rgba8Unorm,
];
/// The most passes any path records, so the query set is sized once.
const MAX_PASSES: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Path {
    /// Clustered forward shading with depth test only: every covered layer is
    /// shaded. The baseline that shows what the prepass buys.
    ForwardNoPrepass,
    /// Depth prepass, then clustered forward shading of visible pixels only.
    Forward,
    /// Depth prepass, G-buffer fill, then one clustered lighting pass.
    Deferred,
}

impl Path {
    const ALL: [Self; 3] = [Self::ForwardNoPrepass, Self::Forward, Self::Deferred];

    fn passes(self) -> &'static [&'static str] {
        match self {
            Self::ForwardNoPrepass => &["light_cull", "forward"],
            Self::Forward => &["depth_prepass", "light_cull", "forward"],
            Self::Deferred => &["depth_prepass", "light_cull", "gbuffer", "deferred_lighting"],
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::ForwardNoPrepass => "forward+ without prepass",
            Self::Forward => "clustered forward+",
            Self::Deferred => "deferred",
        }
    }
}

/// The production prepass (`depth_prepass.wgsl`) with the scene's positions
/// already in clip space: the same position-only, fragment-free work without
/// the uniforms the fixed orthographic scene does not need.
const PREPASS_SOURCE: &str = "@vertex fn vs_main(@location(0) position: vec3<f32>)     -> @builtin(position) @invariant vec4<f32> { return vec4<f32>(position, 1.0); }";

/// Sources for the two prototypes: the production BRDF, the shared light
/// loop, then the path's entry points.
fn lighting_source(entry: &str) -> String {
    [
        include_str!("../../../assets/shaders/brdf.wgsl"),
        include_str!("../../../assets/shaders/prototype_lighting.wgsl"),
        entry,
    ]
    .join("\n")
}

fn module(device: &wgpu::Device, label: &str, source: String) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

fn depth_state(write: bool, compare: wgpu::CompareFunction) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: pipeline::DEPTH_FORMAT,
        depth_write_enabled: Some(write),
        depth_compare: Some(compare),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// Every pipeline either path uses, built with layouts derived from the
/// shaders so the prototypes carry no layout code of their own.
struct Pipelines {
    prepass: wgpu::RenderPipeline,
    cull: wgpu::ComputePipeline,
    /// Shades on top of the prepass: depth `LessEqual`, no depth writes.
    forward: wgpu::RenderPipeline,
    /// Writes its own depth: the no-prepass baseline.
    forward_writing_depth: wgpu::RenderPipeline,
    /// Loops over every light instead of the cluster list; the culling oracle.
    forward_all_lights: wgpu::RenderPipeline,
    gbuffer: wgpu::RenderPipeline,
    deferred: wgpu::RenderPipeline,
}

impl Pipelines {
    fn new(device: &wgpu::Device, hdr_format: wgpu::TextureFormat) -> Self {
        let forward_module = module(
            device,
            "b2-forward",
            lighting_source(include_str!("../../../assets/shaders/clustered_forward.wgsl")),
        );
        let forward = |label, depth, clustered: bool| {
            let constants = [("CLUSTERED", if clustered { 1.0 } else { 0.0 })];
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &forward_module,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(pipeline::vertex_layout())],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(depth),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &forward_module,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    targets: &[Some(hdr_format.into())],
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        let cull_module = module(
            device,
            "b2-light-cull",
            include_str!("../../../assets/shaders/light_cull.wgsl").to_owned(),
        );
        let gbuffer_module = module(
            device,
            "b2-gbuffer",
            include_str!("../../../assets/shaders/gbuffer.wgsl").to_owned(),
        );
        let deferred_module = module(
            device,
            "b2-deferred-lighting",
            lighting_source(include_str!("../../../assets/shaders/deferred_lighting.wgsl")),
        );
        let gbuffer_targets = GBUFFER_FORMATS.map(|format| Some(format.into()));

        Self {
            prepass: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("b2-depth-prepass"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &module(device, "b2-depth-prepass", PREPASS_SOURCE.to_owned()),
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(pipeline::vertex_layout())],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(depth_state(true, wgpu::CompareFunction::Less)),
                multisample: wgpu::MultisampleState::default(),
                fragment: None,
                multiview_mask: None,
                cache: None,
            }),
            cull: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("b2-light-cull"),
                layout: None,
                module: &cull_module,
                entry_point: Some("cs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }),
            forward: forward(
                "b2-forward",
                depth_state(false, wgpu::CompareFunction::LessEqual),
                true,
            ),
            forward_writing_depth: forward(
                "b2-forward-writing-depth",
                depth_state(true, wgpu::CompareFunction::Less),
                true,
            ),
            forward_all_lights: forward(
                "b2-forward-all-lights",
                depth_state(false, wgpu::CompareFunction::LessEqual),
                false,
            ),
            gbuffer: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("b2-gbuffer"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &gbuffer_module,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(pipeline::vertex_layout())],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(depth_state(false, wgpu::CompareFunction::LessEqual)),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &gbuffer_module,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &gbuffer_targets,
                }),
                multiview_mask: None,
                cache: None,
            }),
            deferred: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("b2-deferred-lighting"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &deferred_module,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &deferred_module,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(hdr_format.into())],
                }),
                multiview_mask: None,
                cache: None,
            }),
        }
    }
}

fn filled_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    bytes: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

fn float_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn texture(
    device: &wgpu::Device,
    label: &str,
    size: (u32, u32),
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: usage | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

fn view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// The fixed scene, one light set and the targets, at one resolution.
struct Bench {
    pipelines: Pipelines,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    hdr_texture: wgpu::Texture,
    hdr: wgpu::TextureView,
    depth: wgpu::TextureView,
    gbuffer: [wgpu::TextureView; 3],
    cull_group: wgpu::BindGroup,
    forward_group: wgpu::BindGroup,
    forward_writing_depth_group: wgpu::BindGroup,
    forward_all_lights_group: wgpu::BindGroup,
    deferred_groups: [wgpu::BindGroup; 2],
    cull_workgroups: u32,
}

impl Bench {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: (u32, u32),
        hdr_format: wgpu::TextureFormat,
        quads_per_side: u32,
        lights: &[BenchLight],
    ) -> Self {
        let pipelines = Pipelines::new(device, hdr_format);
        let mesh = b2_scene::scene_mesh(quads_per_side);
        let light_count = u32::try_from(lights.len()).expect("benchmark light counts fit u32");
        let clusters = clustered::cluster_count(clustered::GRID).expect("the benchmark grid is valid");
        let list_words =
            clustered::list_words(clustered::GRID, light_count).expect("the benchmark light lists fit u32");

        let storage = wgpu::BufferUsages::STORAGE;
        let vertices = filled_buffer(
            device,
            queue,
            "b2-vertices",
            &float_bytes(&mesh.interleaved()),
            wgpu::BufferUsages::VERTEX,
        );
        let index_bytes: Vec<u8> = mesh
            .indices
            .iter()
            .flat_map(|index| index.to_le_bytes())
            .collect();
        let indices = filled_buffer(
            device,
            queue,
            "b2-indices",
            &index_bytes,
            wgpu::BufferUsages::INDEX,
        );
        let params = filled_buffer(
            device,
            queue,
            "b2-params",
            &b2_scene::params_bytes([size.0, size.1], light_count, clustered::GRID),
            wgpu::BufferUsages::UNIFORM,
        );
        // A zero-light set still binds a buffer: bindings cannot be empty.
        let mut light_bytes = b2_scene::light_bytes(lights);
        if light_bytes.is_empty() {
            light_bytes.resize(32, 0);
        }
        let lights = filled_buffer(device, queue, "b2-lights", &light_bytes, storage);
        let lists = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("b2-cluster-lists"),
            size: u64::from(list_words) * 4,
            usage: storage,
            mapped_at_creation: false,
        });

        let sampled = wgpu::TextureUsages::TEXTURE_BINDING;
        let hdr_texture = texture(device, "b2-hdr", size, hdr_format, wgpu::TextureUsages::COPY_SRC);
        let hdr = view(&hdr_texture);
        let depth = view(&texture(
            device,
            "b2-depth",
            size,
            pipeline::DEPTH_FORMAT,
            sampled,
        ));
        let gbuffer =
            GBUFFER_FORMATS.map(|format| view(&texture(device, "b2-gbuffer", size, format, sampled)));

        let light_group = |layout: wgpu::BindGroupLayout, label| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: lights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: lists.as_entire_binding(),
                    },
                ],
            })
        };
        let cull_group = light_group(pipelines.cull.get_bind_group_layout(0), "b2-cull");
        let forward_group = light_group(pipelines.forward.get_bind_group_layout(0), "b2-forward");
        let forward_writing_depth_group = light_group(
            pipelines.forward_writing_depth.get_bind_group_layout(0),
            "b2-forward-writing-depth",
        );
        let forward_all_lights_group = light_group(
            pipelines.forward_all_lights.get_bind_group_layout(0),
            "b2-forward-all-lights",
        );
        let deferred_lights = light_group(pipelines.deferred.get_bind_group_layout(0), "b2-deferred-lights");
        let texture_entry = |binding, view| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(view),
        };
        let deferred_textures = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("b2-deferred-gbuffer"),
            layout: &pipelines.deferred.get_bind_group_layout(1),
            entries: &[
                texture_entry(0, &gbuffer[0]),
                texture_entry(1, &gbuffer[1]),
                texture_entry(2, &gbuffer[2]),
                texture_entry(3, &depth),
            ],
        });

        Self {
            pipelines,
            vertices,
            indices,
            index_count: u32::try_from(mesh.indices.len()).expect("the benchmark mesh fits u32 indices"),
            hdr_texture,
            hdr,
            depth,
            gbuffer,
            cull_group,
            forward_group,
            forward_writing_depth_group,
            forward_all_lights_group,
            deferred_groups: [deferred_lights, deferred_textures],
            cull_workgroups: clusters.div_ceil(CULL_WORKGROUP),
        }
    }

    /// Records one frame of `path`. With `queries`, the `i`th pass writes its
    /// begin and end timestamps to the slots `gpu_timing::query_indices(i)`.
    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        path: Path,
        all_lights: bool,
        queries: Option<&wgpu::QuerySet>,
    ) {
        let mut pass_index = 0;
        let mut slots = || {
            let index = pass_index;
            pass_index += 1;
            queries.map(|query_set| {
                let (begin, end) =
                    gpu_timing::query_indices(index).expect("benchmark passes fit the query set");
                (query_set, begin, end)
            })
        };

        if path != Path::ForwardNoPrepass {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("depth_prepass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(self.depth_attachment(wgpu::LoadOp::Clear(1.0))),
                timestamp_writes: render_writes(slots()),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipelines.prepass);
            self.draw_scene(&mut pass);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("light_cull"),
                timestamp_writes: slots().map(|(query_set, begin, end)| wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(begin),
                    end_of_pass_write_index: Some(end),
                }),
            });
            pass.set_pipeline(&self.pipelines.cull);
            pass.set_bind_group(0, &self.cull_group, &[]);
            pass.dispatch_workgroups(self.cull_workgroups, 1, 1);
        }

        match path {
            Path::ForwardNoPrepass | Path::Forward => {
                let (pipeline, group, depth_load) = match (path, all_lights) {
                    (Path::ForwardNoPrepass, _) => (
                        &self.pipelines.forward_writing_depth,
                        &self.forward_writing_depth_group,
                        wgpu::LoadOp::Clear(1.0),
                    ),
                    (_, true) => (
                        &self.pipelines.forward_all_lights,
                        &self.forward_all_lights_group,
                        wgpu::LoadOp::Load,
                    ),
                    (_, false) => (&self.pipelines.forward, &self.forward_group, wgpu::LoadOp::Load),
                };
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("forward"),
                    color_attachments: &[Some(color_attachment(&self.hdr))],
                    depth_stencil_attachment: Some(self.depth_attachment(depth_load)),
                    timestamp_writes: render_writes(slots()),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                self.draw_scene(&mut pass);
            }
            Path::Deferred => {
                {
                    let targets = self.gbuffer.each_ref().map(|view| Some(color_attachment(view)));
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("gbuffer"),
                        color_attachments: &targets,
                        depth_stencil_attachment: Some(self.depth_attachment(wgpu::LoadOp::Load)),
                        timestamp_writes: render_writes(slots()),
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_pipeline(&self.pipelines.gbuffer);
                    self.draw_scene(&mut pass);
                }
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("deferred_lighting"),
                    color_attachments: &[Some(color_attachment(&self.hdr))],
                    depth_stencil_attachment: None,
                    timestamp_writes: render_writes(slots()),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.pipelines.deferred);
                pass.set_bind_group(0, &self.deferred_groups[0], &[]);
                pass.set_bind_group(1, &self.deferred_groups[1], &[]);
                pass.draw(0..3, 0..1);
            }
        }
    }

    fn depth_attachment(&self, load: wgpu::LoadOp<f32>) -> wgpu::RenderPassDepthStencilAttachment<'_> {
        wgpu::RenderPassDepthStencilAttachment {
            view: &self.depth,
            depth_ops: Some(wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }
    }

    fn draw_scene(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

fn render_writes(slots: Option<(&wgpu::QuerySet, u32, u32)>) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
    slots.map(|(query_set, begin, end)| wgpu::RenderPassTimestampWrites {
        query_set,
        beginning_of_pass_write_index: Some(begin),
        end_of_pass_write_index: Some(end),
    })
}

fn color_attachment(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            store: wgpu::StoreOp::Store,
        },
    }
}

/// Maps `buffer`, blocking until the GPU is done, and returns its bytes.
fn read_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device finishes the frame");
    receiver
        .recv()
        .expect("the map callback runs")
        .expect("the readback buffer maps");
    let bytes = buffer
        .get_mapped_range(..)
        .expect("the mapped range is readable")
        .to_vec();
    buffer.unmap();
    bytes
}

/// GPU milliseconds per pass, plus the whole path from the first pass's begin
/// to the last pass's end, over [`MEASURED_FRAMES`] frames.
///
/// Each frame waits for its own timestamps. That serialises frames, which is
/// fine here: the numbers are GPU intervals, not CPU throughput.
fn measure(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bench: &Bench,
    path: Path,
) -> Vec<(&'static str, Summary)> {
    let names = path.passes();
    let query_count = u32::try_from(names.len() * 2).expect("a handful of passes");
    let bytes = u64::from(MAX_PASSES) * 2 * 8;
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("b2-timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: MAX_PASSES * 2,
    });
    let resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("b2-resolve"),
        size: bytes,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("b2-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let period = queue.get_timestamp_period();

    let mut samples = vec![Vec::new(); names.len() + 1];
    for frame in 0..WARMUP_FRAMES + MEASURED_FRAMES {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("b2-frame"),
        });
        bench.record(&mut encoder, path, false, Some(&queries));
        encoder.resolve_query_set(&queries, 0..query_count, &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));
        let ticks: Vec<u64> = read_buffer(device, &readback)
            .chunks_exact(8)
            .take(names.len() * 2)
            .map(|tick| u64::from_le_bytes(tick.try_into().expect("eight-byte timestamp")))
            .collect();
        if frame < WARMUP_FRAMES {
            continue;
        }
        for (index, millis) in gpu_timing::pass_millis(&ticks, period).into_iter().enumerate() {
            samples[index].extend(millis);
        }
        let whole = [ticks[0], ticks[ticks.len() - 1]];
        samples[names.len()].extend(gpu_timing::pass_millis(&whole, period).into_iter().flatten());
    }

    names
        .iter()
        .copied()
        .chain(std::iter::once("total"))
        .zip(samples)
        .filter_map(|(name, values)| b2_scene::summarize(&values).map(|summary| (name, summary)))
        .collect()
}

fn bench_renderer() -> Renderer {
    let gpu_preference = match std::env::var("B2_GPU").as_deref() {
        Ok("low") => GpuPreference::LowPower,
        _ => GpuPreference::HighPerformance,
    };
    Renderer::headless(
        16,
        16,
        RendererConfig {
            gpu_preference,
            ..RendererConfig::default()
        },
    )
    .expect("a headless renderer")
}

#[test]
#[ignore = "benchmark; run with --release -- --ignored --nocapture"]
fn b2_lighting_path_benchmark() {
    let renderer = bench_renderer();
    let (device, queue) = (&renderer.device, &renderer.queue);
    let info = renderer.adapter_info();
    assert!(
        device.features().contains(wgpu::Features::TIMESTAMP_QUERY),
        "{} has no timestamp queries; B2 needs measured GPU time",
        info.name
    );
    println!(
        "B2 benchmark on {} ({}, {}, driver {}), {}x{}, {} back-to-front layers, cluster grid {:?}",
        info.name,
        info.device_kind,
        info.backend,
        info.driver,
        BENCH_SIZE.0,
        BENCH_SIZE.1,
        b2_scene::LAYERS,
        clustered::GRID,
    );
    let runs = LIGHT_COUNTS
        .map(|count| (count, b2_scene::QUADS_PER_SIDE))
        .into_iter()
        .chain([(b2_scene::LIGHT_COUNT, b2_scene::DENSE_QUADS_PER_SIDE)]);
    for (count, quads_per_side) in runs {
        let lights = b2_scene::lights(count);
        let bench = Bench::new(device, queue, BENCH_SIZE, HDR_FORMAT, quads_per_side, &lights);
        let triangles = u64::from(b2_scene::LAYERS) * u64::from(quads_per_side).pow(2) * 2;
        for path in Path::ALL {
            println!("  {count:>5} lights, {triangles} triangles, {}:", path.label());
            let results = measure(device, queue, &bench, path);
            assert_eq!(
                results.len(),
                path.passes().len() + 1,
                "every pass and the total were measured"
            );
            for (name, summary) in results {
                println!(
                    "    {name:<18} median {:>7.3} ms  p95 {:>7.3} ms  mean {:>7.3} ms  (n={})",
                    summary.median, summary.p95, summary.mean, summary.count
                );
            }
        }
    }
}

/// Correctness for the benchmark itself: a timing comparison of two paths is
/// only evidence if they draw the same image.
mod images {
    use super::*;

    const SIZE: (u32, u32) = (320, 180);
    /// Full-precision target so the comparison measures the paths, not
    /// half-float rounding.
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
    const BYTES_PER_PIXEL: u32 = 16;

    fn render(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bench: &Bench,
        path: Path,
        all_lights: bool,
    ) -> Vec<f32> {
        // 320 pixels of 16 bytes is 5120, already a multiple of the 256-byte
        // row alignment, so the rows need no unpadding.
        let bytes_per_row = SIZE.0 * BYTES_PER_PIXEL;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("b2-image-readback"),
            size: u64::from(bytes_per_row) * u64::from(SIZE.1),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("b2-image"),
        });
        bench.record(&mut encoder, path, all_lights, None);
        encoder.copy_texture_to_buffer(
            bench.hdr_texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(SIZE.1),
                },
            },
            wgpu::Extent3d {
                width: SIZE.0,
                height: SIZE.1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        read_buffer(device, &buffer)
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().expect("four-byte float")))
            .collect()
    }

    /// The largest per-channel difference, relative to the brighter value
    /// plus `floor`, so dark pixels are not held to a relative bound.
    fn worst_difference(a: &[f32], b: &[f32], floor: f32) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / (x.abs().max(y.abs()) + floor))
            .fold(0.0, f32::max)
    }

    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn every_path_draws_the_same_lit_image() {
        let renderer = bench_renderer();
        let (device, queue) = (&renderer.device, &renderer.queue);
        let bench = Bench::new(
            device,
            queue,
            SIZE,
            FORMAT,
            b2_scene::QUADS_PER_SIDE,
            &b2_scene::lights(b2_scene::LIGHT_COUNT),
        );

        let reference = render(device, queue, &bench, Path::Forward, true);
        let mean = reference.iter().sum::<f32>() / reference.len() as f32;
        assert!(mean > 0.05, "the scene is lit (mean {mean})");

        let clustered = render(device, queue, &bench, Path::Forward, false);
        let difference = worst_difference(&reference, &clustered, 1e-3);
        assert!(
            difference < 1e-4,
            "culling drops no contributing light: {difference}"
        );

        let no_prepass = render(device, queue, &bench, Path::ForwardNoPrepass, false);
        let difference = worst_difference(&reference, &no_prepass, 1e-3);
        assert!(difference < 1e-4, "the prepass changes no pixel: {difference}");

        // The G-buffer stores albedo in 8 bits and the normal in 10, so
        // deferred may differ by about one albedo step.
        let deferred = render(device, queue, &bench, Path::Deferred, false);
        let difference = worst_difference(&reference, &deferred, 1e-2);
        assert!(
            difference < 0.02,
            "deferred matches forward within G-buffer precision: {difference}"
        );
    }
}
