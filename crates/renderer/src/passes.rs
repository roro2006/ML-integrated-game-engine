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
use crate::graph::{RenderGraph, ResourceId, TransientDesc};
use crate::pipeline::DEPTH_FORMAT;
use crate::registry::DrawPlan;
use crate::{RenderError, RenderFrame, Renderer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BloomStep {
    Down { prefilter: bool },
    Up,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExposureStep {
    Histogram,
    Average,
}

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
pub(crate) fn hdr_desc(width: u32, height: u32) -> TransientDesc {
    TransientDesc {
        format: HDR_FORMAT,
        width,
        height,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    }
}

/// The depth buffer's descriptor at a given size.
pub(crate) fn depth_desc(width: u32, height: u32) -> TransientDesc {
    TransientDesc {
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
        timing: &mut Option<crate::gpu_timing::GpuTimer>,
    ) -> Result<(wgpu::CommandBuffer, String), RenderError> {
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
        if frame.post.auto_exposure.is_some() {
            graph.add_pass("exposure-histogram", &[hdr], &[], move |context| {
                self.exposure_pass(context, hdr, ExposureStep::Histogram)
            });
            graph.add_pass("exposure-average", &[hdr], &[], move |context| {
                self.exposure_pass(context, hdr, ExposureStep::Average)
            });
        }
        let bloom = self.declare_bloom(&mut graph, hdr, &frame.post.bloom);
        let tonemap_reads: &[ResourceId] = match bloom {
            Some(first) => &[hdr, first],
            None => &[hdr],
        };
        graph.add_pass("tonemap", tonemap_reads, &[surface], move |context| {
            self.tonemap_pass(context, hdr, bloom, surface)
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
        let graph_dump = compiled.dump();
        let queries = timing
            .as_mut()
            .and_then(|timer| timer.start(compiled.pass_names().collect()));
        compiled
            .execute_profiled(&mut context, |context, index, begin| {
                if let (Some(queries), Some((start, end))) =
                    (&queries, crate::gpu_timing::query_indices(index as u32))
                {
                    context
                        .encoder
                        .write_timestamp(queries, if begin { start } else { end });
                }
            })
            .map_err(graph_error)?;
        if let Some(timer) = timing {
            timer.resolve(&mut context.encoder);
        }
        Ok((context.encoder.finish(), graph_dump))
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
        // Draws stay in the order the engine gave; the pipeline changes only
        // when a draw needs a different one. Group 0 is rebound on every
        // change, since a pipeline switch may invalidate it.
        let mut current: Option<Option<crate::permutation::PermutationKey>> = None;
        for planned in &plan.draws {
            let Some(mesh) = self.meshes.get_by_index(planned.mesh) else {
                continue;
            };
            let Ok(offset) = u32::try_from(planned.slot * stride) else {
                continue;
            };
            let pbr = frame
                .draws
                .get(planned.draw)
                .and_then(|draw| self.materials.get(draw.material))
                .and_then(|material| material.pbr.as_ref());
            let wanted = pbr.map(|binding| binding.key);
            if current != Some(wanted) {
                let pipeline = match wanted {
                    Some(key) => self
                        .pbr_pipelines
                        .get(&key)
                        .ok_or_else(|| format!("no PBR pipeline for {key:?}"))?,
                    None => &self.pipeline,
                };
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.globals_bind_group, &[]);
                current = Some(wanted);
            }
            if let Some(binding) = pbr {
                pass.set_bind_group(2, &binding.bind_group, &[]);
            }
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
        bloom: Option<ResourceId>,
        surface: ResourceId,
    ) -> Result<(), String> {
        let hdr_view = context.view(hdr)?;
        // With bloom off the scene stands in for it: the intensity is 0, so it
        // adds nothing, and no black texture has to be kept for the purpose.
        let bloom_view = match bloom {
            Some(first) => context.view(first)?,
            None => hdr_view.clone(),
        };
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.exposure.exposure_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&bloom_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.bloom.sampler),
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

    /// Declares the bloom chain, if bloom is on, and returns its first level,
    /// which holds the whole chain's light once the upsamples have run.
    fn declare_bloom<'a>(
        &'a self,
        graph: &mut RenderGraph<'a, PassContext>,
        hdr: ResourceId,
        settings: &crate::Bloom,
    ) -> Option<ResourceId> {
        const LEVELS: [&str; 6] = ["bloom-0", "bloom-1", "bloom-2", "bloom-3", "bloom-4", "bloom-5"];
        const DOWN: [&str; 6] = [
            "bloom-down-0",
            "bloom-down-1",
            "bloom-down-2",
            "bloom-down-3",
            "bloom-down-4",
            "bloom-down-5",
        ];
        const UP: [&str; 6] = [
            "bloom-up-0",
            "bloom-up-1",
            "bloom-up-2",
            "bloom-up-3",
            "bloom-up-4",
            "bloom-up-5",
        ];

        if !crate::post::bloom::enabled(settings) {
            return None;
        }
        let (width, height) = self.size;
        let levels: Vec<ResourceId> = crate::post::bloom::level_sizes(width, height)
            .into_iter()
            .zip(LEVELS)
            .map(|((level_width, level_height), name)| {
                graph.create(
                    name,
                    TransientDesc {
                        format: HDR_FORMAT,
                        width: level_width,
                        height: level_height,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    },
                )
            })
            .collect();
        for (index, &level) in levels.iter().enumerate() {
            let source = if index == 0 { hdr } else { levels[index - 1] };
            let prefilter = index == 0;
            graph.add_pass(DOWN[index], &[source], &[level], move |context| {
                self.bloom_pass(context, source, level, BloomStep::Down { prefilter })
            });
        }
        for index in (0..levels.len().saturating_sub(1)).rev() {
            let (smaller, target) = (levels[index + 1], levels[index]);
            graph.add_pass(UP[index], &[smaller, target], &[target], move |context| {
                self.bloom_pass(context, smaller, target, BloomStep::Up)
            });
        }
        levels.first().copied()
    }

    fn bloom_pass(
        &self,
        context: &mut PassContext,
        source: ResourceId,
        target: ResourceId,
        step: BloomStep,
    ) -> Result<(), String> {
        let resources = &self.bloom;
        let source_view = context.view(source)?;
        let target_view = context.view(target)?;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom-bind-group"),
            layout: &resources.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&resources.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.params.as_entire_binding(),
                },
            ],
        });
        let (pipeline, load) = match step {
            BloomStep::Down { prefilter: true } => {
                (&resources.prefilter, wgpu::LoadOp::Clear(wgpu::Color::BLACK))
            }
            BloomStep::Down { prefilter: false } => {
                (&resources.downsample, wgpu::LoadOp::Clear(wgpu::Color::BLACK))
            }
            // Upsamples add onto what the downsample left there.
            BloomStep::Up => (&resources.upsample, wgpu::LoadOp::Load),
        };
        let mut pass = context.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bloom-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }

    fn exposure_pass(
        &self,
        context: &mut PassContext,
        hdr: ResourceId,
        step: ExposureStep,
    ) -> Result<(), String> {
        let resources = &self.exposure;
        let hdr_view = context.view(hdr)?;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("exposure-bind-group"),
            layout: &resources.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.histogram.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.params.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = context.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(match step {
                    ExposureStep::Histogram => "exposure-histogram",
                    ExposureStep::Average => "exposure-average",
                }),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bind_group, &[]);
            match step {
                ExposureStep::Histogram => {
                    let (width, height) = self.size;
                    pass.set_pipeline(&resources.build);
                    pass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
                }
                ExposureStep::Average => {
                    pass.set_pipeline(&resources.average);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
        }
        if step == ExposureStep::Average {
            // Hand the adapted value to the tonemap pass as a uniform.
            context.encoder.copy_buffer_to_buffer(
                &resources.state,
                0,
                &resources.exposure_uniform,
                0,
                crate::post::exposure::STATE_SIZE as u64,
            );
        }
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
