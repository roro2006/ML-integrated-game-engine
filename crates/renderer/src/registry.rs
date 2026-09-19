//! Handle registries and the per-frame draw plan.
//!
//! Both live here because they are the crate's bookkeeping layer: which handle
//! refers to which GPU resource, and what a frame's draw list turns into once
//! those handles have been resolved. Neither touches `wgpu`, so both are
//! testable on a machine with no GPU — which matters, because "a stale handle
//! must not panic the frame" is a rule about this code and nothing else.

use core::marker::PhantomData;

use crate::frame::{DrawCommand, MaterialHandle, MeshHandle};

/// A runtime-only index into a [`Registry`].
///
/// Implemented by the handle types in `frame.rs` so one registry can serve
/// meshes and materials without either being able to index the other: a
/// `MeshHandle` simply cannot be passed where a `MaterialHandle` is expected.
pub(crate) trait Handle: Copy {
    fn from_index(index: u32) -> Self;
    fn index(self) -> u32;
}

impl Handle for MeshHandle {
    fn from_index(index: u32) -> Self {
        Self::from_index(index)
    }

    fn index(self) -> u32 {
        self.index()
    }
}

impl Handle for MaterialHandle {
    fn from_index(index: u32) -> Self {
        Self::from_index(index)
    }

    fn index(self) -> u32 {
        self.index()
    }
}

/// Append-only storage that mints typed handles.
///
/// Append-only on purpose for M3: nothing unloads a mesh yet, so a handle stays
/// valid for the renderer's lifetime and there is no ABA problem to solve. When
/// unloading arrives it needs generational handles like the ECS uses, which is a
/// deliberate later change rather than something to guess at now.
#[derive(Debug)]
pub(crate) struct Registry<H, T> {
    items: Vec<T>,
    handle: PhantomData<H>,
}

impl<H, T> Default for Registry<H, T> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            handle: PhantomData,
        }
    }
}

impl<H: Handle, T> Registry<H, T> {
    pub(crate) fn insert(&mut self, value: T) -> H {
        // Saturating rather than panicking: four billion registrations is not a
        // reachable state, and a renderer is not a place to abort a session.
        let index = u32::try_from(self.items.len()).unwrap_or(u32::MAX);
        self.items.push(value);
        H::from_index(index)
    }

    pub(crate) fn get(&self, handle: H) -> Option<&T> {
        self.items.get(handle.index() as usize)
    }

    pub(crate) fn get_by_index(&self, index: u32) -> Option<&T> {
        self.items.get(index as usize)
    }
}

/// How much geometry a registered mesh carries.
///
/// A trait rather than a concrete type so the planner below can be exercised
/// without constructing GPU buffers.
pub(crate) trait Geometry {
    fn triangle_count(&self) -> u32;
}

/// One draw that resolved, ready to be recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlannedDraw {
    /// Index into the frame's draw list.
    pub(crate) draw: usize,
    /// Index of the mesh in the mesh registry.
    pub(crate) mesh: u32,
    /// Which per-draw uniform slot this draw occupies. Equal to the draw's
    /// position in the *plan*, not in the frame, so skipped draws leave no gap
    /// and the uniform buffer only grows with draws that are actually recorded.
    pub(crate) slot: usize,
}

/// A frame's draw list with every handle resolved.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct DrawPlan {
    pub(crate) draws: Vec<PlannedDraw>,
    pub(crate) triangles: u32,
    /// Draws whose mesh or material handle referred to nothing.
    pub(crate) skipped: u32,
}

/// Resolves a frame's draw list against the registries.
///
/// An unknown handle is skipped and counted rather than treated as an error. A
/// handle can go stale in ordinary play — content is swapped, an entity is
/// despawned mid-extraction — and dropping one prop is a far better outcome than
/// losing the frame or the process. The count is what makes the loss visible;
/// silently drawing nothing is the failure mode this guards against.
pub(crate) fn plan_draws<M: Geometry, T>(
    draws: &[DrawCommand],
    meshes: &Registry<MeshHandle, M>,
    materials: &Registry<MaterialHandle, T>,
) -> DrawPlan {
    let mut plan = DrawPlan {
        draws: Vec::with_capacity(draws.len()),
        ..DrawPlan::default()
    };
    for (index, draw) in draws.iter().enumerate() {
        let Some(mesh) = meshes.get(draw.mesh) else {
            plan.skipped += 1;
            continue;
        };
        if materials.get(draw.material).is_none() {
            plan.skipped += 1;
            continue;
        }
        plan.triangles = plan.triangles.saturating_add(mesh.triangle_count());
        plan.draws.push(PlannedDraw {
            draw: index,
            mesh: draw.mesh.index(),
            slot: plan.draws.len(),
        });
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    /// Stands in for `GpuMesh`, which cannot be built without a device.
    struct FakeMesh(u32);

    impl Geometry for FakeMesh {
        fn triangle_count(&self) -> u32 {
            self.0
        }
    }

    struct FakeMaterial;

    fn draw(mesh: MeshHandle, material: MaterialHandle) -> DrawCommand {
        DrawCommand {
            mesh,
            material,
            model: IDENTITY,
            normal_matrix: IDENTITY,
        }
    }

    #[test]
    fn handles_are_minted_in_order_and_resolve_to_their_value() {
        let mut meshes: Registry<MeshHandle, FakeMesh> = Registry::default();
        let first = meshes.insert(FakeMesh(2));
        let second = meshes.insert(FakeMesh(10));

        assert_eq!(first.index(), 0);
        assert_eq!(second.index(), 1);
        assert_eq!(meshes.get(first).map(FakeMesh::triangle_count), Some(2));
        assert_eq!(meshes.get(second).map(FakeMesh::triangle_count), Some(10));
        assert!(
            meshes.get(MeshHandle::from_index(2)).is_none(),
            "a handle past the end must resolve to nothing, not panic"
        );
    }

    #[test]
    fn a_plan_counts_triangles_from_the_meshes_it_resolved() {
        let mut meshes = Registry::default();
        let cube = meshes.insert(FakeMesh(12));
        let plane = meshes.insert(FakeMesh(2));
        let mut materials = Registry::default();
        let material = materials.insert(FakeMaterial);

        let draws = [draw(cube, material), draw(plane, material), draw(cube, material)];
        let plan = plan_draws(&draws, &meshes, &materials);

        assert_eq!(plan.draws.len(), 3);
        assert_eq!(plan.triangles, 12 + 2 + 12);
        assert_eq!(plan.skipped, 0);
        assert_eq!(
            plan.draws.iter().map(|planned| planned.slot).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(plan.draws[1].mesh, plane.index());
    }

    #[test]
    fn unknown_mesh_and_material_handles_are_skipped_and_counted() {
        let mut meshes = Registry::default();
        let cube = meshes.insert(FakeMesh(12));
        let mut materials = Registry::default();
        let material = materials.insert(FakeMaterial);

        let draws = [
            draw(MeshHandle::from_index(99), material),
            draw(cube, material),
            draw(cube, MaterialHandle::from_index(42)),
            draw(MeshHandle::from_index(7), MaterialHandle::from_index(7)),
        ];
        let plan = plan_draws(&draws, &meshes, &materials);

        assert_eq!(plan.skipped, 3);
        assert_eq!(plan.draws.len(), 1, "only the resolvable draw survives");
        assert_eq!(plan.triangles, 12, "skipped draws contribute no triangles");
        assert_eq!(plan.draws[0].draw, 1, "the plan points back at the right command");
        assert_eq!(
            plan.draws[0].slot, 0,
            "slots are dense, so skipped draws waste no uniform space"
        );
    }

    #[test]
    fn an_empty_frame_plans_nothing() {
        let meshes: Registry<MeshHandle, FakeMesh> = Registry::default();
        let materials: Registry<MaterialHandle, FakeMaterial> = Registry::default();
        let plan = plan_draws(&[], &meshes, &materials);
        assert_eq!(plan, DrawPlan::default());
    }
}
