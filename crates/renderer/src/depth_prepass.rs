//! The depth prepass pipelines (ADR-0024).
//!
//! Three pipelines from one shader, built and hot-reloaded together:
//!
//! - `depth_only`: position-only, with no fragment stage, so it writes depth
//!   at the lowest cost the hardware has. The prepass uses it whenever
//!   ambient occlusion is off.
//! - `normals`: depth plus the world-space normal target that GTAO reads
//!   (ADR-0024 decision 4, the thin G-buffer), for opaque draws.
//! - `masked_normals`: the same for cut-outs, running their alpha test so a
//!   leaf's holes stay open. Without it GTAO would see no foliage at all,
//!   and would occlude the ground behind a leaf onto the leaf.
//!
//! The first two use the lit pipeline layout (globals, draw); the masked one
//! adds the material group, as the PBR pipelines lay it out.

use crate::block_on::block_on;
use crate::pipeline::{DEPTH_FORMAT, vertex_layout};
use crate::shaders::DEPTH_PREPASS;

/// The normal target's format: three 10-bit channels hold an encoded unit
/// vector to about a tenth of a degree, in four bytes.
pub(crate) const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgb10a2Unorm;

/// The normal target's descriptor at a given size.
pub(crate) fn normal_desc(width: u32, height: u32) -> crate::graph::TransientDesc {
    crate::graph::TransientDesc {
        format: NORMAL_FORMAT,
        width,
        height,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    }
}

#[derive(Debug)]
pub(crate) struct PrepassPipelines {
    pub(crate) depth_only: wgpu::RenderPipeline,
    pub(crate) normals: wgpu::RenderPipeline,
    pub(crate) masked_normals: wgpu::RenderPipeline,
}

/// Compiles `source` and builds the three prepass pipelines, returning the
/// validation message instead of panicking so a hot reload can keep the
/// working ones. `masked_layout` adds the material group to `layout`.
pub(crate) fn create_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    masked_layout: &wgpu::PipelineLayout,
    source: &str,
) -> Result<PrepassPipelines, String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(DEPTH_PREPASS),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipelines = PrepassPipelines {
        depth_only: create_pipeline(device, layout, &module, Variant::DepthOnly),
        normals: create_pipeline(device, layout, &module, Variant::Normals),
        masked_normals: create_pipeline(device, masked_layout, &module, Variant::MaskedNormals),
    };
    match block_on(scope.pop()) {
        Some(error) => Err(error.to_string()),
        None => Ok(pipelines),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Variant {
    DepthOnly,
    Normals,
    MaskedNormals,
}

fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    variant: Variant,
) -> wgpu::RenderPipeline {
    let targets = [Some(wgpu::ColorTargetState {
        format: NORMAL_FORMAT,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    })];
    let fragment = |entry_point| wgpu::FragmentState {
        module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        targets: &targets,
    };
    let (label, vertex_entry, fragment) = match variant {
        Variant::DepthOnly => ("depth-prepass-pipeline", "vs_main", None),
        Variant::Normals => (
            "depth-prepass-normals-pipeline",
            "vs_normals",
            Some(fragment("fs_normals")),
        ),
        Variant::MaskedNormals => (
            "depth-prepass-masked-normals-pipeline",
            "vs_normals",
            Some(fragment("fs_masked_normals")),
        ),
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(vertex_entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(vertex_layout())],
        },
        // The same rasterisation as the forward pipelines, so the two passes
        // cover exactly the same pixels: cut-outs are double-sided there too.
        primitive: wgpu::PrimitiveState {
            cull_mode: if variant == Variant::MaskedNormals {
                None
            } else {
                Some(wgpu::Face::Back)
            },
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
        fragment,
        multiview_mask: None,
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_normal_target_is_drawn_into_then_sampled_by_the_occlusion_pass() {
        let desc = normal_desc(1920, 1080);
        assert_eq!(desc.format, wgpu::TextureFormat::Rgb10a2Unorm);
        assert!(desc.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
        assert!(desc.usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
        assert_eq!((desc.width, desc.height), (1920, 1080));
    }
}
