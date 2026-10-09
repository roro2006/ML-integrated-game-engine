//! The render graph: passes declare the resources they read and write, and the
//! graph checks the order, works out how long each resource lives, and runs the
//! passes.
//!
//! Decisions, recorded here because they are easy to second-guess later:
//!
//! - **Declaration order is execution order.** The graph does not reorder
//!   passes. wgpu submits to one queue and this renderer has no async compute,
//!   so reordering would buy nothing and make a frame harder to read. Instead
//!   the graph *validates* the order: a pass that reads a transient resource no
//!   earlier pass wrote is an error. That is the mistake a sorting graph would
//!   report as a cycle, reported here at the pass that makes it.
//! - **No barriers.** wgpu tracks resource state and inserts synchronisation
//!   within a queue itself. The graph's job is ordering and lifetimes; it emits
//!   no barriers and never will while there is one queue.
//! - **Imported resources** (the swapchain image, later a persistent history
//!   buffer) are owned outside the graph and count as written before the first
//!   pass. **Transient resources** are created from a descriptor and handed out
//!   by the pool in `graph/resources.rs`, which uses the lifetimes computed
//!   here.
//!
//! The graph is generic over the context its passes receive, so the ordering
//! and lifetime logic is tested without a device.

pub(crate) mod resources;

use std::collections::HashSet;
use std::fmt::{self, Write as _};

/// A resource declared in one graph. Only meaningful for the graph that
/// returned it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct ResourceId(usize);

/// What a transient texture must be. Identical descriptors are interchangeable,
/// which is what lets the pool reuse a texture between frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TextureDesc {
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) usage: wgpu::TextureUsages,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Origin {
    Imported,
    Transient(TextureDesc),
}

#[derive(Debug)]
struct Resource {
    name: &'static str,
    origin: Origin,
}

/// The first and last position, in execution order, of the passes that use a
/// resource. Inclusive at both ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Lifetime {
    pub(crate) first: usize,
    pub(crate) last: usize,
}

type Execute<'a, C> = Box<dyn FnOnce(&mut C) + 'a>;

struct Pass<'a, C> {
    name: &'static str,
    reads: Vec<ResourceId>,
    writes: Vec<ResourceId>,
    execute: Execute<'a, C>,
}

impl<C> fmt::Debug for Pass<'_, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pass")
            .field("name", &self.name)
            .field("reads", &self.reads)
            .field("writes", &self.writes)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GraphError {
    /// A pass reads a transient resource that no earlier pass writes.
    ReadBeforeWrite {
        pass: &'static str,
        resource: &'static str,
    },
    /// A pass names a resource this graph never declared.
    UnknownResource { pass: &'static str },
    /// Two passes share a name, which would make the dump and per-pass timings
    /// ambiguous.
    DuplicatePass(&'static str),
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadBeforeWrite { pass, resource } => {
                write!(f, "pass `{pass}` reads `{resource}` before any pass writes it")
            }
            Self::UnknownResource { pass } => {
                write!(f, "pass `{pass}` uses a resource from another graph")
            }
            Self::DuplicatePass(name) => write!(f, "two passes are named `{name}`"),
        }
    }
}

/// A frame's passes and resources, as declared.
#[derive(Debug)]
pub(crate) struct RenderGraph<'a, C> {
    resources: Vec<Resource>,
    passes: Vec<Pass<'a, C>>,
}

impl<C> Default for RenderGraph<'_, C> {
    fn default() -> Self {
        Self {
            resources: Vec::new(),
            passes: Vec::new(),
        }
    }
}

impl<'a, C> RenderGraph<'a, C> {
    /// Declares a resource owned outside the graph, such as the swapchain
    /// image. It counts as written before the first pass.
    pub(crate) fn import(&mut self, name: &'static str) -> ResourceId {
        self.declare(name, Origin::Imported)
    }

    /// Declares a transient texture the pool provides for this frame.
    pub(crate) fn create(&mut self, name: &'static str, desc: TextureDesc) -> ResourceId {
        self.declare(name, Origin::Transient(desc))
    }

    fn declare(&mut self, name: &'static str, origin: Origin) -> ResourceId {
        self.resources.push(Resource { name, origin });
        ResourceId(self.resources.len() - 1)
    }

    /// Adds a pass that runs after every pass added before it.
    ///
    /// A pass that loads a target and draws over it, as debug lines do, lists
    /// the target in both `reads` and `writes`.
    pub(crate) fn add_pass(
        &mut self,
        name: &'static str,
        reads: &[ResourceId],
        writes: &[ResourceId],
        execute: impl FnOnce(&mut C) + 'a,
    ) {
        self.passes.push(Pass {
            name,
            reads: reads.to_vec(),
            writes: writes.to_vec(),
            execute: Box::new(execute),
        });
    }

    /// Checks the order and computes each resource's lifetime.
    pub(crate) fn compile(self) -> Result<CompiledGraph<'a, C>, GraphError> {
        let mut names = HashSet::new();
        let mut written: Vec<bool> = self
            .resources
            .iter()
            .map(|resource| resource.origin == Origin::Imported)
            .collect();
        let mut lifetimes: Vec<Option<Lifetime>> = vec![None; self.resources.len()];

        for (position, pass) in self.passes.iter().enumerate() {
            if !names.insert(pass.name) {
                return Err(GraphError::DuplicatePass(pass.name));
            }
            for id in pass.reads.iter().chain(&pass.writes) {
                if id.0 >= self.resources.len() {
                    return Err(GraphError::UnknownResource { pass: pass.name });
                }
            }
            for id in &pass.reads {
                if !written[id.0] {
                    return Err(GraphError::ReadBeforeWrite {
                        pass: pass.name,
                        resource: self.resources[id.0].name,
                    });
                }
            }
            for id in &pass.writes {
                written[id.0] = true;
            }
            for id in pass.reads.iter().chain(&pass.writes) {
                let lifetime = lifetimes[id.0].get_or_insert(Lifetime {
                    first: position,
                    last: position,
                });
                lifetime.last = position;
            }
        }

        Ok(CompiledGraph {
            resources: self.resources,
            passes: self.passes,
            lifetimes,
        })
    }
}

/// A graph whose order has been checked, ready to have its transients
/// allocated and its passes run.
#[derive(Debug)]
pub(crate) struct CompiledGraph<'a, C> {
    resources: Vec<Resource>,
    passes: Vec<Pass<'a, C>>,
    lifetimes: Vec<Option<Lifetime>>,
}

impl<C> CompiledGraph<'_, C> {
    /// When `id` is in use, or `None` if no pass touches it.
    pub(crate) fn lifetime(&self, id: ResourceId) -> Option<Lifetime> {
        self.lifetimes.get(id.0).copied().flatten()
    }

    /// Transient resources that some pass uses, with their descriptors. A
    /// transient no pass touches is never allocated.
    pub(crate) fn transients(&self) -> impl Iterator<Item = (ResourceId, TextureDesc)> + '_ {
        self.resources
            .iter()
            .enumerate()
            .filter_map(|(index, resource)| match resource.origin {
                Origin::Transient(desc) if self.lifetimes[index].is_some() => Some((ResourceId(index), desc)),
                _ => None,
            })
    }

    /// Pass names in execution order.
    pub(crate) fn pass_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.passes.iter().map(|pass| pass.name)
    }

    /// A stable text description of the graph: passes in order with what they
    /// read and write, then each resource with its lifetime.
    pub(crate) fn dump(&self) -> String {
        let name = |id: &ResourceId| self.resources[id.0].name;
        let list = |ids: &[ResourceId]| ids.iter().map(name).collect::<Vec<_>>().join(", ");
        let mut out = String::from("passes:\n");
        for (position, pass) in self.passes.iter().enumerate() {
            let _ = writeln!(
                out,
                "  {position} {}: reads [{}] writes [{}]",
                pass.name,
                list(&pass.reads),
                list(&pass.writes)
            );
        }
        out.push_str("resources:\n");
        for (resource, lifetime) in self.resources.iter().zip(&self.lifetimes) {
            let origin = match resource.origin {
                Origin::Imported => "imported".to_owned(),
                Origin::Transient(desc) => {
                    format!("transient {:?} {}x{}", desc.format, desc.width, desc.height)
                }
            };
            let live = match lifetime {
                Some(lifetime) => format!("passes {}..={}", lifetime.first, lifetime.last),
                None => "unused".to_owned(),
            };
            let _ = writeln!(out, "  {}: {origin}, {live}", resource.name);
        }
        out
    }

    /// Runs every pass in order.
    pub(crate) fn execute(self, context: &mut C) {
        for pass in self.passes {
            (pass.execute)(context);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HDR: TextureDesc = TextureDesc {
        format: wgpu::TextureFormat::Rgba16Float,
        width: 64,
        height: 32,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
    };

    type Log = Vec<&'static str>;

    fn logging(name: &'static str) -> impl FnOnce(&mut Log) {
        move |log: &mut Log| log.push(name)
    }

    /// lit -> tonemap -> debug lines, the shape of the frame after E1.
    fn frame_graph() -> (RenderGraph<'static, Log>, [ResourceId; 3]) {
        let mut graph = RenderGraph::default();
        let surface = graph.import("surface");
        let hdr = graph.create("hdr", HDR);
        let depth = graph.create(
            "depth",
            TextureDesc {
                format: wgpu::TextureFormat::Depth32Float,
                ..HDR
            },
        );
        graph.add_pass("lit", &[], &[hdr, depth], logging("lit"));
        graph.add_pass("tonemap", &[hdr], &[surface], logging("tonemap"));
        graph.add_pass(
            "debug-lines",
            &[depth, surface],
            &[surface],
            logging("debug-lines"),
        );
        (graph, [surface, hdr, depth])
    }

    #[test]
    fn passes_run_in_the_order_they_were_added() {
        let (graph, _) = frame_graph();
        let compiled = graph.compile().expect("a valid frame");
        assert_eq!(
            compiled.pass_names().collect::<Vec<_>>(),
            ["lit", "tonemap", "debug-lines"]
        );

        let mut log = Log::new();
        compiled.execute(&mut log);
        assert_eq!(log, ["lit", "tonemap", "debug-lines"]);
    }

    #[test]
    fn reading_a_transient_before_it_is_written_is_rejected() {
        let mut graph: RenderGraph<'_, Log> = RenderGraph::default();
        let surface = graph.import("surface");
        let hdr = graph.create("hdr", HDR);
        // Declared the wrong way round: tonemap before the pass that fills hdr.
        graph.add_pass("tonemap", &[hdr], &[surface], logging("tonemap"));
        graph.add_pass("lit", &[], &[hdr], logging("lit"));

        assert_eq!(
            graph.compile().err(),
            Some(GraphError::ReadBeforeWrite {
                pass: "tonemap",
                resource: "hdr",
            })
        );
    }

    #[test]
    fn a_pass_reading_its_own_first_write_is_rejected() {
        let mut graph: RenderGraph<'_, Log> = RenderGraph::default();
        let hdr = graph.create("hdr", HDR);
        graph.add_pass("feedback", &[hdr], &[hdr], logging("feedback"));
        assert!(matches!(
            graph.compile(),
            Err(GraphError::ReadBeforeWrite { pass: "feedback", .. })
        ));
    }

    #[test]
    fn imported_resources_count_as_already_written() {
        let mut graph: RenderGraph<'_, Log> = RenderGraph::default();
        let history = graph.import("history");
        let hdr = graph.create("hdr", HDR);
        graph.add_pass("resolve", &[history], &[hdr], logging("resolve"));
        assert!(graph.compile().is_ok());
    }

    #[test]
    fn a_resource_from_another_graph_is_rejected() {
        let mut other: RenderGraph<'_, Log> = RenderGraph::default();
        other.import("a");
        let foreign = other.import("b");

        let mut graph: RenderGraph<'_, Log> = RenderGraph::default();
        graph.import("only");
        graph.add_pass("lit", &[], &[foreign], logging("lit"));
        assert_eq!(
            graph.compile().err(),
            Some(GraphError::UnknownResource { pass: "lit" })
        );
    }

    #[test]
    fn duplicate_pass_names_are_rejected() {
        let mut graph: RenderGraph<'_, Log> = RenderGraph::default();
        let surface = graph.import("surface");
        graph.add_pass("lit", &[], &[surface], logging("a"));
        graph.add_pass("lit", &[], &[surface], logging("b"));
        assert_eq!(graph.compile().err(), Some(GraphError::DuplicatePass("lit")));
    }

    #[test]
    fn lifetimes_span_first_to_last_use() {
        let (graph, [surface, hdr, depth]) = frame_graph();
        let compiled = graph.compile().expect("a valid frame");
        assert_eq!(compiled.lifetime(hdr), Some(Lifetime { first: 0, last: 1 }));
        assert_eq!(compiled.lifetime(depth), Some(Lifetime { first: 0, last: 2 }));
        assert_eq!(compiled.lifetime(surface), Some(Lifetime { first: 1, last: 2 }));
    }

    #[test]
    fn an_unused_transient_has_no_lifetime_and_is_not_allocated() {
        let (mut graph, [_, hdr, depth]) = frame_graph();
        let bloom = graph.create("bloom", HDR);
        let compiled = graph.compile().expect("a valid frame");

        assert_eq!(compiled.lifetime(bloom), None);
        let transients: Vec<_> = compiled.transients().map(|(id, _)| id).collect();
        assert_eq!(
            transients,
            [hdr, depth],
            "imports and unused transients are left out"
        );
    }

    #[test]
    fn the_dump_lists_passes_then_resources() {
        let (graph, _) = frame_graph();
        let dump = graph.compile().expect("a valid frame").dump();
        assert_eq!(
            dump,
            "passes:\n\
             \x20 0 lit: reads [] writes [hdr, depth]\n\
             \x20 1 tonemap: reads [hdr] writes [surface]\n\
             \x20 2 debug-lines: reads [depth, surface] writes [surface]\n\
             resources:\n\
             \x20 surface: imported, passes 1..=2\n\
             \x20 hdr: transient Rgba16Float 64x32, passes 0..=1\n\
             \x20 depth: transient Depth32Float 64x32, passes 0..=2\n"
        );
    }
}
