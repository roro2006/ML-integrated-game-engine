//! The frame's passes, declared on the render graph.
//!
//! Three passes, each its own render pass:
//!
//! - `lit` clears the HDR scene target and the depth buffer, then draws the
//!   opaque geometry in linear light.
//! - `tonemap` maps the HDR scene onto the swapchain image (`post::tonemap`).
//! - `debug-lines` loads the swapchain image and depth and draws world-space
//!   lines that test depth without writing it, so the scene hides them and
//!   they never hide the scene. It runs after tonemapping on purpose: a debug
//!   colour should reach the screen as authored, not be exposed and curved
//!   like scene light. It is not declared at all on a frame with no lines, so
//!   debug drawing that is switched off costs nothing.
//!
//! The scene target and the depth buffer are pooled transients. Both carry
//! `TEXTURE_BINDING` as well as `RENDER_ATTACHMENT`: the tonemap pass reads
//! the scene, and later screen-space passes will sample depth.

use crate::graph::resources::{self, TexturePool};
use crate::graph::{RenderGraph, ResourceId, TextureDesc};
use crate::pipeline::DEPTH_FORMAT;
use crate::registry::DrawPlan;
use crate::{RenderError, RenderFrame, Renderer};

/// What each pass records into: the frame's encoder and a view of every
/// resource, indexed by [`ResourceId::index`].
#[derive(Debug)]
pub(crate) struct PassContext {
    encoder: wgpu::CommandEncoder,
    views: Vec<Option<wgpu::TextureView>>,
}

impl PassContext {
    fn view(&self, id: ResourceId) -> Result<wgpu::TextureView, String> {
        self.views
            .get(id.index())
            .cloned()
            .flatten()
            .ok_or_else(|| format!("resource {} has no view", id.index()))
    }
}

/// The HDR scene target's format: half-float linear light, no MSAA.
pub(crate) const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The HDR scene target's descriptor at a given size.
pub(crate) fn hdr_desc(width: u32, height: u32) -> TextureDesc {
    TextureDesc {
        format: HDR_FORMAT,
        width,
        height,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    }
}

/// The depth buffer's descriptor at a given size.
pub(crate) fn depth_desc(width: u32, height: u32) -> TextureDesc {
    TextureDesc {
        format: DEPTH_FORMAT,
        width,
        height,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    }
}

impl Renderer {
    /// Declares this frame's passes, allocates their transients from `pool`,
    /// and records them into one command buffer.
    ///
    /// Uniforms and vertex data must already be written: this only records.
    pub(crate) fn record_frame(
        &self,
        pool: &mut TexturePool<wgpu::Texture>,
        surface_view: wgpu::TextureView,
        frame: &RenderFrame,
        plan: &DrawPlan,
        debug_vertices: Option<u32>,
    ) -> Result<wgpu::CommandBuffer, RenderError> {
        let (width, height) = self.size;
        let mut graph: RenderGraph<'_, PassContext> = RenderGraph::default();
        let surface = graph.import("surface");
        let hdr = graph.create("hdr", hdr_desc(width, height));
        let depth = graph.create("depth", depth_desc(width, height));

        // Depth is stored only when a later pass loads it; otherwise the store
        // is wasted bandwidth, which tiled integrated GPUs feel most.
        let depth_is_read_later = debug_vertices.is_some();
        graph.add_pass("lit", &[], &[hdr, depth], move |context| {
            self.lit_pass(context, hdr, depth, depth_is_read_later, frame, plan)
        });
        graph.add_pass("tonemap", &[hdr], &[surface], move |context| {
            self.tonemap_pass(context, hdr, surface)
        });
        if let Some(count) = debug_vertices {
            graph.add_pass("debug-lines", &[depth, surface], &[surface], move |context| {
                self.debug_line_pass(context, surface, depth, count)
            });
        }

        let graph_error = |error: crate::graph::GraphError| RenderError::Graph(error.to_string());
        let compiled = graph.compile().map_err(graph_error)?;

        pool.begin_frame();
        let mut views = vec![None; compiled.resource_count()];
        let mut set_view = |id: ResourceId, view: Option<wgpu::TextureView>| {
            let entry = views
                .get_mut(id.index())
                .ok_or_else(|| RenderError::Graph(format!("resource {} is out of range", id.index())))?;
            *entry = view;
            Ok::<(), RenderError>(())
        };
        set_view(surface, Some(surface_view))?;
        for (id, name, desc) in compiled.transients() {
            let slot = pool.acquire(desc, |desc| resources::create_texture(&self.device, name, desc));
            let view = pool
                .get(slot)
                .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
            set_view(id, view)?;
        }

        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame-encoder"),
            });
        let mut context = PassContext { encoder, views };
        compiled.execute(&mut context).map_err(graph_error)?;
        Ok(context.encoder.finish())
    }

    fn lit_pass(
        &self,
        context: &mut PassContext,
        scene: ResourceId,
        depth: ResourceId,
        keep_depth: bool,
        frame: &RenderFrame,
        plan: &DrawPlan,
    ) -> Result<(), String> {
        let color_view = context.view(scene)?;
        let depth_view = context.view(depth)?;
        // Scene light, like everything else written here; the tonemap pass
        // maps it to the display along with the geometry.
        let clear = frame.clear_color;
        let mut pass = context.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("lit-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: clear.r,
                        g: clear.g,
                        b: clear.b,
                        a: clear.a,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    // 1.0 is the far plane for the `0..1` depth range
                    // (ADR-0008), and `Less` keeps the nearest fragment.
                    load: wgpu::LoadOp::Clear(1.0),
                    store: if keep_depth {
                        wgpu::StoreOp::Store
                    } else {
                        wgpu::StoreOp::Discard
                    },
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        if plan.draws.is_empty() {
            return Ok(());
        }
        let stride = self.draw_stride as usize;
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.globals_bind_group, &[]);
        for planned in &plan.draws {
            let Some(mesh) = self.meshes.get_by_index(planned.mesh) else {
                continue;
            };
            let Ok(offset) = u32::try_from(planned.slot * stride) else {
                continue;
            };
            pass.set_bind_group(1, &self.draw_bind_group, &[offset]);
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
        Ok(())
    }

    fn tonemap_pass(
        &self,
        context: &mut PassContext,
        hdr: ResourceId,
        surface: ResourceId,
    ) -> Result<(), String> {
        let hdr_view = context.view(hdr)?;
        let surface_view = context.view(surface)?;
        // Built per frame: the scene view comes from the pool and is only
        // stable between resizes. Caching it with the pooled texture is noted
        // as follow-up work in the phase 1 task list.
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tonemap-bind-group"),
            layout: &self.tonemap_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.post_buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = context.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tonemap-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &surface_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    // Every pixel is overwritten, so nothing needs loading.
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.tonemap_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }

    fn debug_line_pass(
        &self,
        context: &mut PassContext,
        surface: ResourceId,
        depth: ResourceId,
        count: u32,
    ) -> Result<(), String> {
        let color_view = context.view(surface)?;
        let depth_view = context.view(depth)?;
        let mut pass = context.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("debug-line-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    // Nothing reads depth after this pass yet.
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.debug_pipeline);
        pass.set_bind_group(0, &self.globals_bind_group, &[]);
        pass.set_vertex_buffer(0, self.debug_vertex_buffer.slice(..));
        pass.draw(0..count, 0..1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scene_target_is_half_float_and_readable_by_the_tonemap_pass() {
        let desc = hdr_desc(64, 32);
        assert_eq!(desc.format, wgpu::TextureFormat::Rgba16Float);
        assert!(desc.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
        assert!(desc.usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
    }

    #[test]
    fn the_depth_buffer_can_be_sampled_as_well_as_drawn_into() {
        let desc = depth_desc(1280, 720);
        assert_eq!(desc.format, wgpu::TextureFormat::Depth32Float);
        assert!(desc.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
        assert!(desc.usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
        assert_eq!((desc.width, desc.height), (1280, 720));
    }
}
