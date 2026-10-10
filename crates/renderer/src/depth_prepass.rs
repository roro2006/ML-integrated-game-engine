//! The depth prepass pipeline (ADR-0024).
//!
//! Position-only and with no fragment stage, so it writes depth at the
//! lowest cost the hardware has. It uses the lit pipeline layout: the
//! globals for the view-projection and the per-draw model matrix.

use crate::block_on::block_on;
use crate::pipeline::{DEPTH_FORMAT, vertex_layout};
use crate::shaders::DEPTH_PREPASS;

/// Compiles `source` and builds the prepass pipeline, returning the
/// validation message instead of panicking so a hot reload can keep the
/// working pipeline.
pub(crate) fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    source: &str,
) -> Result<wgpu::RenderPipeline, String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(DEPTH_PREPASS),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("depth-prepass-pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(vertex_layout())],
        },
        // The same rasterisation as the opaque forward pipelines, so the
        // two passes cover exactly the same pixels.
        primitive: wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Back),
            ..wgpu::PrimitiveState::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
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
