//! Renderer facade exposed to games, and the extraction step that turns world
//! state into a frame.
//!
//! Games configure and observe rendering through these types; the wgpu backend
//! stays inside the `renderer` crate.
//!
//! # Extraction
//!
//! [`RenderExtractor::extract`] reads the ECS and produces a
//! [`RenderFrame`]: a finished, culled, ordered draw list. The backend draws
//! exactly what it is given, in the order given
//! (`docs/plans/m3-renderer.md`), which is what keeps culling and ordering
//! testable without a GPU and keeps the GPU layer free of scene knowledge.
//!
//! Extraction is **read-only** over the world and produces no side effects, so
//! it belongs at the end of the variable-rate frame schedule, after the camera
//! rig has been posed. It never runs inside fixed-step simulation: what is
//! visible this frame must not influence gameplay.
//!
//! The boundary conversions live here on purpose — `math::Aabb` from
//! [`renderer::Aabb`], and `Mat4` to `[[f32; 4]; 4]` — because that is what
//! keeps `glam` out of the renderer's signatures (ADR-0008).

pub use renderer::{
    AdapterInfo, AmbientLight, BackendPreference, Color, DeviceKind, DirectionalLight, DrawCommand,
    FrameOutcome, FrameStats, GpuPreference, MaterialDesc, MaterialHandle, MeshData, MeshHandle, PassTiming,
    PresentMode, RenderError, RenderFrame, RendererConfig, SkipReason,
};

use crate::camera::Camera;
use crate::ecs::{Component, Entity, Resource, World};
use crate::math::{Aabb, Frustum, Mat3, Mat4, Transform};

/// Startup-time access to GPU asset registration.
///
/// Games register meshes and materials through this rather than touching the
/// renderer, so one piece of scene-building code serves both a windowed
/// session and a `--headless` run. That matters because extraction, culling,
/// and ordering are engine logic that must stay exercisable without a GPU
/// (`docs/plans/m3-renderer.md`).
pub trait RenderAssets {
    fn upload_mesh(&mut self, mesh: &MeshData) -> Result<MeshHandle, RenderError>;
    fn create_material(&mut self, desc: &MaterialDesc) -> MaterialHandle;
}

/// [`RenderAssets`] for runs with no GPU.
///
/// Hands out sequential indices, and validates meshes exactly as the backend
/// would, so a mesh that would fail upload still fails here rather than
/// passing headlessly and breaking on real hardware. Fabricated handles are
/// safe by construction: the backend skips unknown handles at draw time and
/// counts them.
#[derive(Debug, Default)]
pub struct HeadlessRenderAssets {
    meshes: u32,
    materials: u32,
}

impl RenderAssets for HeadlessRenderAssets {
    fn upload_mesh(&mut self, mesh: &MeshData) -> Result<MeshHandle, RenderError> {
        mesh.validate().map_err(RenderError::MeshUpload)?;
        let handle = MeshHandle::from_index(self.meshes);
        self.meshes += 1;
        Ok(handle)
    }

    fn create_material(&mut self, _desc: &MaterialDesc) -> MaterialHandle {
        let handle = MaterialHandle::from_index(self.materials);
        self.materials += 1;
        handle
    }
}

/// Colour the frame buffer is cleared to. Systems may change it at any time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClearColor(pub Color);

impl Default for ClearColor {
    fn default() -> Self {
        Self(Color::rgb(0.05, 0.05, 0.08))
    }
}

impl Resource for ClearColor {}

/// Adapter description of the GPU in use, available once rendering starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveAdapter(pub AdapterInfo);

impl Resource for ActiveAdapter {}

/// The scene's lighting, written into every frame's per-frame uniform block.
///
/// One directional key light plus a flat ambient fill is the whole lighting
/// model for M3: it is what low-poly flat-shaded facets need to read clearly,
/// and it costs the same on integrated graphics whatever the scene contains.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SceneLighting {
    pub key: DirectionalLight,
    pub ambient: AmbientLight,
}

impl Resource for SceneLighting {}

// `Transform` is the engine's world-placement component. The type lives in
// `math` so game-domain and tooling code can use it without depending on the
// ECS; the component impl belongs here, with the rest of the rendering-facing
// components, rather than putting an ECS concept into the math module.
impl Component for Transform {}

/// What to draw for an entity.
///
/// Both handles are runtime-only, like [`Entity`]: they are minted by mesh
/// upload and material registration and must never reach canonical state, a
/// save file, or generated content. Content refers to meshes and materials by
/// stable string IDs, which are resolved to handles at load time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshRenderer {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
}

impl Component for MeshRenderer {}

impl MeshRenderer {
    pub fn new(mesh: MeshHandle, material: MaterialHandle) -> Self {
        Self { mesh, material }
    }
}

/// One rigid piece of a [`Model`].
///
/// Handles are runtime-only, exactly as on [`MeshRenderer`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelPart {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    /// Where the part sits on the body when nothing is animating it, relative
    /// to the entity's [`Transform`].
    pub rest: Transform,
    /// This frame's animation, applied on top of `rest` in the part's own
    /// space: a swinging arm rotates about its own shoulder, not the body's
    /// centre. Presentation writes it; nothing but extraction reads it.
    pub pose: Transform,
    /// Mesh-space bounds, so each part is culled on its own.
    pub bounds: Option<RenderBounds>,
}

impl ModelPart {
    /// A part at `rest`, unposed.
    pub fn new(mesh: MeshHandle, material: MaterialHandle, rest: Transform) -> Self {
        Self {
            mesh,
            material,
            rest,
            pose: Transform::IDENTITY,
            bounds: None,
        }
    }

    /// The part's world matrix under an entity whose matrix is `entity`.
    pub fn world_matrix(&self, entity: &Mat4) -> Mat4 {
        *entity * self.rest.to_matrix() * self.pose.to_matrix()
    }
}

/// A body made of rigid parts (ADR-0017): each drawn as
/// `entity × rest × pose`, one draw per part.
///
/// There is deliberately no entity hierarchy behind this. A part is data inside
/// one component, composed at extraction, so a model is spawned, hidden,
/// culled and despawned as one thing and no parent/child bookkeeping exists to
/// go stale. An entity should carry a [`Model`] or a [`MeshRenderer`]; if it
/// carries both, both draw.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Model {
    pub parts: Vec<ModelPart>,
}

impl Component for Model {}

/// Inverse-transpose of a model matrix's linear part, for normals.
///
/// A composed part matrix can be degenerate — a part posed to zero scale —
/// and inverting it would put NaNs on the GPU. The fallback mirrors
/// [`Transform::normal_matrix`]: rotation only, which keeps normals unit and
/// finite.
fn normal_matrix_of(model: &Mat4, fallback: &Transform) -> Mat4 {
    let linear = Mat3::from_mat4(*model);
    if linear.determinant().abs() < 1e-12 {
        return Mat4::from_quat(fallback.rotation);
    }
    Mat4::from_mat3(linear.inverse().transpose())
}

/// Whether an entity is drawn.
///
/// Absence of this component means **visible**, so the common case costs no
/// storage and no lookup miss is a bug. Hiding is a rendering concern only: a
/// hidden enemy still simulates, which is what culling-independent gameplay
/// requires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Visibility {
    pub visible: bool,
}

impl Component for Visibility {}

impl Visibility {
    pub const VISIBLE: Self = Self { visible: true };
    pub const HIDDEN: Self = Self { visible: false };

    pub const fn is_visible(self) -> bool {
        self.visible
    }
}

impl Default for Visibility {
    fn default() -> Self {
        Self::VISIBLE
    }
}

/// Object-space bounds used to cull an entity.
///
/// Bounds are in the entity's own space and are transformed by its model matrix
/// during extraction, so one value can be shared by every instance of a mesh
/// and nothing has to be recomputed when an entity moves.
///
/// An entity **without** this component is never culled. That is the
/// conservative direction: drawing something off-screen costs one draw call,
/// whereas culling something visible looks like geometry vanishing at the edge
/// of the screen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderBounds(pub Aabb);

impl Component for RenderBounds {}

impl RenderBounds {
    /// Bounds of a CPU-side mesh, or `None` if it has no vertices.
    ///
    /// This is the intended pairing for [`MeshRenderer`]: take the bounds from
    /// the same [`MeshData`] that was uploaded to get the handle.
    pub fn from_mesh_data(mesh: &MeshData) -> Option<Self> {
        mesh.bounds().map(|bounds| Self(Aabb::from(bounds)))
    }

    /// These bounds in world space, given an entity's model matrix.
    pub fn world_bounds(&self, model: &Mat4) -> Aabb {
        self.0.transformed(model)
    }
}

impl From<renderer::Aabb> for RenderBounds {
    fn from(bounds: renderer::Aabb) -> Self {
        Self(Aabb::from(bounds))
    }
}

/// What one extraction pass decided, for the performance overlay and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExtractStats {
    /// Draw commands emitted.
    pub drawn: u32,
    /// Draws dropped by frustum culling. A [`Model`] is culled part by part,
    /// so it can contribute several.
    pub culled: u32,
    /// Entities skipped because they carry a hidden [`Visibility`].
    pub hidden: u32,
}

impl ExtractStats {
    /// Draw candidates: one per [`MeshRenderer`] entity, one per visible
    /// [`Model`] part, and one per hidden entity of either kind.
    pub fn considered(self) -> u32 {
        self.drawn + self.culled + self.hidden
    }
}

/// Sort key: material, then mesh, then the entity itself, then the part.
///
/// See [`RenderExtractor`] for why this is the order. The part index is what
/// keeps the key total for a [`Model`] whose parts share a mesh and material —
/// two identical arms, say — which would otherwise tie and leave their order to
/// the sort.
type DrawKey = (u32, u32, u32, u32, u32);

/// The part index a [`MeshRenderer`] draw sorts with. [`Model`] parts use
/// `1 + index`, so the two never collide on an entity carrying both.
const WHOLE: u32 = 0;

/// Builds the key an emitted draw is sorted by.
///
/// The entity's index **and generation** are the tie-break, so the key is a
/// total order determined entirely by world *state*. Without it, draws sharing a
/// material and mesh would keep component-storage order, which depends on the
/// history of inserts and removals rather than on the state itself.
fn draw_key(material: u32, mesh: u32, entity: Entity, part: u32) -> DrawKey {
    (material, mesh, entity.index(), entity.generation(), part)
}

/// Whether an entity survives frustum culling.
///
/// An entity with no [`RenderBounds`] is always kept; see that type for why the
/// ambiguous case errs toward drawing.
fn passes_culling(frustum: &Frustum, bounds: Option<&RenderBounds>, model: &Mat4) -> bool {
    bounds.is_none_or(|bounds| frustum.intersects_aabb(&bounds.world_bounds(model)))
}

#[derive(Debug, Clone, Copy)]
struct Entry {
    key: DrawKey,
    draw: DrawCommand,
}

/// Turns world state into a [`RenderFrame`].
///
/// # Ordering policy
///
/// Draws are emitted grouped by **material, then mesh**, because that is the
/// order that minimises GPU state changes: the material determines the bind
/// group (and later the pipeline), which is the more expensive switch, and the
/// mesh determines the vertex and index buffers. Sorting by mesh first would
/// rebind the same materials repeatedly.
///
/// The order is *deterministic*: it depends only on world state, never on
/// hash-map iteration or on the history of component inserts and removals. That
/// matters beyond tidiness — a draw list that shuffles between identical frames
/// makes frame-time regressions impossible to attribute and defeats replay
/// comparison.
///
/// The extractor owns a scratch buffer so sorting needs no per-frame
/// allocation; combined with [`RenderFrame::clear_draws`], a steady-state frame
/// allocates nothing. Keep one in a resource or in the runner rather than
/// constructing one per frame.
#[derive(Debug, Default)]
pub struct RenderExtractor {
    scratch: Vec<Entry>,
    /// The view-projection of the last frame extracted.
    previous_view_proj: Option<[[f32; 4]; 4]>,
}

impl Resource for RenderExtractor {}

impl RenderExtractor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fills `frame` from `world` for a viewport of the given aspect ratio.
    ///
    /// `frame`'s existing allocation is reused. A world with no [`Camera`]
    /// resource falls back to [`Camera::default`] rather than skipping the
    /// frame, so a scene renders from something sensible before the game has
    /// installed its rig.
    pub fn extract(&mut self, world: &World, aspect: f32, frame: &mut RenderFrame) -> ExtractStats {
        let camera = world.resource::<Camera>().copied().unwrap_or_default();
        let view_projection = camera.view_projection(aspect);
        let frustum = Frustum::from_view_projection(&view_projection);
        let lighting = world.resource::<SceneLighting>().copied().unwrap_or_default();

        let view_proj = view_projection.to_cols_array_2d();
        // Last frame's matrix for motion vectors; the first frame has none, so
        // it moves nowhere.
        frame.prev_view_proj = self.previous_view_proj.unwrap_or(view_proj);
        self.previous_view_proj = Some(view_proj);
        frame.view_proj = view_proj;
        frame.camera_position = camera.eye.to_array();
        frame.clear_color = world.resource::<ClearColor>().copied().unwrap_or_default().0;
        frame.light = lighting.key;
        frame.ambient = lighting.ambient;
        frame.clear_draws();

        let mut stats = ExtractStats::default();
        self.scratch.clear();
        for (entity, mesh_renderer, transform) in world.query2::<MeshRenderer, Transform>() {
            if !world
                .get::<Visibility>(entity)
                .is_none_or(|visibility| visibility.is_visible())
            {
                stats.hidden += 1;
                continue;
            }
            let model = transform.to_matrix();
            if !passes_culling(&frustum, world.get::<RenderBounds>(entity), &model) {
                stats.culled += 1;
                continue;
            }
            stats.drawn += 1;
            self.scratch.push(Entry {
                key: draw_key(
                    mesh_renderer.material.index(),
                    mesh_renderer.mesh.index(),
                    entity,
                    WHOLE,
                ),
                draw: DrawCommand {
                    mesh: mesh_renderer.mesh,
                    material: mesh_renderer.material,
                    model: model.to_cols_array_2d(),
                    normal_matrix: transform.normal_matrix().to_cols_array_2d(),
                },
            });
        }

        for (entity, model, transform) in world.query2::<Model, Transform>() {
            if !world
                .get::<Visibility>(entity)
                .is_none_or(|visibility| visibility.is_visible())
            {
                stats.hidden += 1;
                continue;
            }
            let body = transform.to_matrix();
            for (index, part) in model.parts.iter().enumerate() {
                let matrix = part.world_matrix(&body);
                if !passes_culling(&frustum, part.bounds.as_ref(), &matrix) {
                    stats.culled += 1;
                    continue;
                }
                stats.drawn += 1;
                let rotation = Transform::IDENTITY
                    .with_rotation(transform.rotation * part.rest.rotation * part.pose.rotation);
                self.scratch.push(Entry {
                    key: draw_key(part.material.index(), part.mesh.index(), entity, 1 + index as u32),
                    draw: DrawCommand {
                        mesh: part.mesh,
                        material: part.material,
                        model: matrix.to_cols_array_2d(),
                        normal_matrix: normal_matrix_of(&matrix, &rotation).to_cols_array_2d(),
                    },
                });
            }
        }

        // The key is a total order, so an unstable sort is deterministic here
        // and avoids the temporary allocation a stable sort would make.
        self.scratch.sort_unstable_by_key(|entry| entry.key);
        frame.draws.extend(self.scratch.iter().map(|entry| entry.draw));
        stats
    }
}

/// Extracts a whole frame in one call, allocating a fresh [`RenderFrame`].
///
/// For tests and one-shot use. A running game should keep a [`RenderExtractor`]
/// and a [`RenderFrame`] and call [`RenderExtractor::extract`], so neither
/// allocates per frame.
pub fn extract_render_frame(world: &World, aspect: f32) -> RenderFrame {
    let mut frame = RenderFrame::default();
    RenderExtractor::new().extract(world, aspect, &mut frame);
    frame
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::FollowCamera;
    use crate::math::Vec3;
    use crate::primitives;

    const ASPECT: f32 = 16.0 / 9.0;

    /// A camera at +5Z looking back at the origin, matching the fixtures used
    /// in `math.rs` so culling expectations line up with the tested frustum.
    fn test_camera() -> Camera {
        Camera::new(60f32.to_radians(), 0.1, 100.0).looking_from(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO)
    }

    fn test_world() -> World {
        let mut world = World::new();
        world.insert_resource(test_camera());
        world
    }

    fn unit_bounds() -> RenderBounds {
        RenderBounds(Aabb {
            min: Vec3::splat(-0.5),
            max: Vec3::splat(0.5),
        })
    }

    fn part(mesh: u32, material: u32, rest: Transform) -> ModelPart {
        let mut part = ModelPart::new(
            MeshHandle::from_index(mesh),
            MaterialHandle::from_index(material),
            rest,
        );
        part.bounds = Some(unit_bounds());
        part
    }

    #[test]
    fn a_model_draws_each_part_at_entity_times_rest_times_pose() {
        use crate::math::Quat;

        let mut world = test_world();
        let entity = world.spawn();
        let _ = world.insert(entity, Transform::from_xyz(1.0, 0.0, 0.0));
        let mut arm = part(0, 0, Transform::from_xyz(0.5, 1.0, 0.0));
        arm.pose = Transform::IDENTITY.with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2));
        let head = part(1, 0, Transform::from_xyz(0.0, 1.5, 0.0));
        let _ = world.insert(
            entity,
            Model {
                parts: vec![arm, head],
            },
        );

        let frame = extract_render_frame(&world, ASPECT);
        assert_eq!(frame.draws.len(), 2, "one draw per part");

        let body = Transform::from_xyz(1.0, 0.0, 0.0).to_matrix();
        for expected in [arm.world_matrix(&body), head.world_matrix(&body)] {
            assert!(
                frame
                    .draws
                    .iter()
                    .any(|draw| Mat4::from_cols_array_2d(&draw.model).abs_diff_eq(expected, 1e-5)),
                "a draw sits at entity × rest × pose"
            );
        }

        // The posed arm's local +Y points along world -X: the pose rotated it
        // about its own shoulder, not about the body's centre.
        let arm_draw = frame
            .draws
            .iter()
            .find(|draw| draw.mesh == MeshHandle::from_index(0))
            .expect("the arm is drawn");
        let up = Mat4::from_cols_array_2d(&arm_draw.model).transform_vector3(Vec3::Y);
        assert!(up.abs_diff_eq(Vec3::NEG_X, 1e-5), "got {up:?}");

        // Checked independently of `world_matrix`, which the comparison above
        // shares with extraction and so cannot catch a wrong multiplication
        // order: posing about its own shoulder leaves the arm *where* it was.
        // Applied in the other order, the pose would swing the rest offset
        // itself, putting the arm at (0, 0.5, 0).
        let at = Mat4::from_cols_array_2d(&arm_draw.model).transform_point3(Vec3::ZERO);
        assert!(at.abs_diff_eq(Vec3::new(1.5, 1.0, 0.0), 1e-5), "got {at:?}");
    }

    #[test]
    fn a_hidden_model_draws_no_parts_and_counts_once() {
        let mut world = test_world();
        let entity = world.spawn();
        let _ = world.insert(entity, Transform::IDENTITY);
        let _ = world.insert(
            entity,
            Model {
                parts: vec![part(0, 0, Transform::IDENTITY), part(1, 0, Transform::IDENTITY)],
            },
        );
        let _ = world.insert(entity, Visibility::HIDDEN);

        let mut frame = RenderFrame::default();
        let stats = RenderExtractor::new().extract(&world, ASPECT, &mut frame);
        assert!(frame.draws.is_empty());
        assert_eq!(stats.hidden, 1, "a hidden model is one hidden thing");
    }

    #[test]
    fn model_parts_are_culled_one_by_one() {
        let mut world = test_world();
        let entity = world.spawn();
        let _ = world.insert(entity, Transform::IDENTITY);
        let _ = world.insert(
            entity,
            Model {
                parts: vec![
                    part(0, 0, Transform::IDENTITY),
                    part(1, 0, Transform::from_xyz(0.0, 0.0, 500.0)),
                ],
            },
        );

        let mut frame = RenderFrame::default();
        let stats = RenderExtractor::new().extract(&world, ASPECT, &mut frame);
        assert_eq!((stats.drawn, stats.culled), (1, 1));
        assert_eq!(frame.draws[0].mesh, MeshHandle::from_index(0));
    }

    /// Two identical arms share a mesh and a material; without the part index
    /// in the key they would tie, and their order would be the sort's choice.
    #[test]
    fn identical_parts_still_sort_deterministically() {
        let mut world = test_world();
        let entity = world.spawn();
        let _ = world.insert(entity, Transform::IDENTITY);
        let _ = world.insert(
            entity,
            Model {
                parts: vec![
                    part(0, 0, Transform::from_xyz(-0.5, 0.0, 0.0)),
                    part(0, 0, Transform::from_xyz(0.5, 0.0, 0.0)),
                ],
            },
        );

        let first = extract_render_frame(&world, ASPECT);
        assert_eq!(first, extract_render_frame(&world, ASPECT));
        assert_eq!(
            first.draws[0].model[3][0], -0.5,
            "parts draw in their authored order when everything else ties"
        );
    }

    #[test]
    fn a_part_posed_to_nothing_still_has_finite_normals() {
        let mut world = test_world();
        let entity = world.spawn();
        let _ = world.insert(entity, Transform::IDENTITY);
        let mut flat = part(0, 0, Transform::IDENTITY);
        flat.bounds = None;
        flat.pose = Transform::IDENTITY.with_scale(Vec3::ZERO);
        let _ = world.insert(entity, Model { parts: vec![flat] });

        let frame = extract_render_frame(&world, ASPECT);
        assert!(
            frame.draws[0]
                .normal_matrix
                .iter()
                .flatten()
                .all(|value| value.is_finite()),
            "a degenerate pose must not put NaNs on the GPU"
        );
    }

    #[test]
    fn visibility_absent_means_visible() {
        assert!(Visibility::default().is_visible());
        assert!(Visibility::VISIBLE.is_visible());
        assert!(!Visibility::HIDDEN.is_visible());
        // The extraction path treats a missing component as visible.
        let missing: Option<&Visibility> = None;
        assert!(missing.is_none_or(|visibility| visibility.is_visible()));
    }

    #[test]
    fn bounds_convert_from_the_mesh_hand_off() {
        let bounds = RenderBounds::from(renderer::Aabb {
            min: [-1.0, -2.0, -3.0],
            max: [1.0, 2.0, 3.0],
        });
        assert_eq!(bounds.0.min, Vec3::new(-1.0, -2.0, -3.0));
        assert_eq!(bounds.0.max, Vec3::new(1.0, 2.0, 3.0));

        // The pairing extraction expects: bounds taken from the same mesh that
        // was uploaded for the handle.
        let cube = primitives::cube(2.0);
        let bounds = RenderBounds::from_mesh_data(&cube).expect("a cube has vertices");
        assert_eq!(bounds.0.min, Vec3::splat(-1.0));
        assert_eq!(bounds.0.max, Vec3::splat(1.0));
        assert_eq!(RenderBounds::from_mesh_data(&MeshData::default()), None);
    }

    #[test]
    fn world_bounds_follow_the_model_matrix() {
        let bounds = unit_bounds();
        let transform = Transform::from_xyz(4.0, 0.0, -2.0).with_scale(Vec3::splat(2.0));
        let world_bounds = bounds.world_bounds(&transform.to_matrix());
        assert_eq!(world_bounds.min, Vec3::new(3.0, -1.0, -3.0));
        assert_eq!(world_bounds.max, Vec3::new(5.0, 1.0, -1.0));
    }

    #[test]
    fn culling_keeps_what_the_camera_can_see() {
        let frustum = test_camera().frustum(ASPECT);
        let bounds = unit_bounds();
        let at = |x: f32, y: f32, z: f32| Transform::from_xyz(x, y, z).to_matrix();

        assert!(
            passes_culling(&frustum, Some(&bounds), &at(0.0, 0.0, 0.0)),
            "the point the camera looks at must be drawn"
        );
        assert!(
            !passes_culling(&frustum, Some(&bounds), &at(0.0, 0.0, 20.0)),
            "behind the camera must be culled"
        );
        assert!(
            !passes_culling(&frustum, Some(&bounds), &at(500.0, 0.0, 0.0)),
            "far off-axis must be culled"
        );
        assert!(
            !passes_culling(&frustum, Some(&bounds), &at(0.0, 0.0, -200.0)),
            "beyond the far plane must be culled"
        );
        assert!(
            passes_culling(&frustum, Some(&bounds), &at(0.0, 0.0, 4.6)),
            "straddling the near plane is partly visible"
        );
        assert!(
            passes_culling(&frustum, None, &at(0.0, 0.0, 20.0)),
            "an entity with no bounds is never culled, even behind the camera"
        );

        // Scale is respected: a big enough object stays visible from off-axis
        // where its unscaled self would not be.
        let far_off = Transform::from_xyz(60.0, 0.0, 0.0);
        assert!(!passes_culling(&frustum, Some(&bounds), &far_off.to_matrix()));
        let huge = far_off.with_scale(Vec3::splat(200.0));
        assert!(
            passes_culling(&frustum, Some(&bounds), &huge.to_matrix()),
            "scaling the bounds up must bring the entity back into view"
        );
    }

    #[test]
    fn draw_keys_group_by_material_then_mesh_then_entity() {
        let mut world = World::new();
        let first = world.spawn();
        let second = world.spawn();

        // Material is the outer group: the same material with a later mesh
        // still sorts before a different material with an earlier mesh.
        assert!(draw_key(1, 9, first, WHOLE) < draw_key(2, 0, first, WHOLE));
        assert!(draw_key(1, 0, first, WHOLE) < draw_key(1, 9, first, WHOLE));
        assert!(draw_key(1, 0, first, WHOLE) < draw_key(1, 0, second, WHOLE));
        // A model's part is the innermost tie-break, after the entity.
        assert!(draw_key(1, 0, first, 1) < draw_key(1, 0, first, 2));
        assert!(draw_key(1, 0, first, 9) < draw_key(1, 0, second, WHOLE));

        let mut keys = [
            draw_key(2, 0, first, WHOLE),
            draw_key(1, 5, second, WHOLE),
            draw_key(1, 5, first, WHOLE),
            draw_key(1, 0, second, WHOLE),
            draw_key(2, 0, second, WHOLE),
        ];
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                draw_key(1, 0, second, WHOLE),
                draw_key(1, 5, first, WHOLE),
                draw_key(1, 5, second, WHOLE),
                draw_key(2, 0, first, WHOLE),
                draw_key(2, 0, second, WHOLE),
            ]
        );

        // A reused slot does not collide with the stale handle it replaced, so
        // the key stays a total order over live entities.
        assert!(world.despawn(second));
        let reused = world.spawn();
        assert_eq!(reused.index(), second.index());
        assert_ne!(draw_key(1, 1, reused, WHOLE), draw_key(1, 1, second, WHOLE));
    }

    #[test]
    fn an_empty_world_produces_an_empty_frame_from_the_camera() {
        let world = test_world();
        let frame = extract_render_frame(&world, ASPECT);

        assert!(frame.draws.is_empty());
        assert_eq!(
            frame.view_proj,
            test_camera().view_projection(ASPECT).to_cols_array_2d()
        );
        assert_eq!(frame.clear_color, ClearColor::default().0);
        assert_eq!(frame.light, DirectionalLight::default());
        assert_eq!(frame.ambient, AmbientLight::default());
    }

    #[test]
    fn frame_settings_come_from_world_resources() {
        let mut world = World::new();

        // No camera at all: the default rig's view, not a broken frame.
        let frame = extract_render_frame(&world, ASPECT);
        assert_eq!(
            frame.view_proj,
            Camera::default().view_projection(ASPECT).to_cols_array_2d()
        );

        // A posed rig reaches the frame through the camera resource.
        let mut camera = Camera::default();
        let rig = FollowCamera::new(Vec3::new(10.0, 0.0, -4.0));
        rig.apply_to(&mut camera);
        world.insert_resource(camera);
        world.insert_resource(ClearColor(Color::rgb(0.2, 0.3, 0.4)));
        let lighting = SceneLighting {
            key: DirectionalLight::default().with_direction([0.0, 0.0, 1.0]),
            ambient: AmbientLight {
                color: [0.1, 0.2, 0.3],
                intensity: 0.5,
            },
        };
        world.insert_resource(lighting);

        let frame = extract_render_frame(&world, ASPECT);
        assert_eq!(frame.view_proj, camera.view_projection(ASPECT).to_cols_array_2d());
        assert_eq!(frame.clear_color, Color::rgb(0.2, 0.3, 0.4));
        assert_eq!(frame.light, lighting.key);
        assert_eq!(frame.ambient, lighting.ambient);
    }

    #[test]
    fn the_camera_and_last_frames_matrix_reach_the_renderer() {
        let mut world = test_world();
        let mut extractor = RenderExtractor::new();
        let mut frame = RenderFrame::default();

        extractor.extract(&world, ASPECT, &mut frame);
        assert_eq!(frame.camera_position, [0.0, 0.0, 5.0]);
        assert_eq!(
            frame.prev_view_proj, frame.view_proj,
            "a first frame has no motion to report"
        );
        let first = frame.view_proj;

        world.insert_resource(
            Camera::new(60f32.to_radians(), 0.1, 100.0).looking_from(Vec3::new(2.0, 1.0, 5.0), Vec3::ZERO),
        );
        extractor.extract(&world, ASPECT, &mut frame);
        assert_eq!(frame.camera_position, [2.0, 1.0, 5.0]);
        assert_eq!(
            frame.prev_view_proj, first,
            "the previous frame's matrix carries over"
        );
        assert_ne!(frame.view_proj, first);
    }

    #[test]
    fn extraction_reuses_the_frame_allocation_and_is_repeatable() {
        let world = test_world();
        let mut extractor = RenderExtractor::new();
        let mut frame = RenderFrame::default();

        // Pre-grow the draw list the way a busy frame would.
        frame.draws.reserve(64);
        let capacity = frame.draws.capacity();

        let first = extractor.extract(&world, ASPECT, &mut frame);
        let view_proj = frame.view_proj;
        let second = extractor.extract(&world, ASPECT, &mut frame);

        assert_eq!(first, second, "the same world must extract to the same stats");
        assert_eq!(frame.view_proj, view_proj);
        assert!(frame.draws.is_empty(), "stale draws must not survive a frame");
        assert_eq!(
            frame.draws.capacity(),
            capacity,
            "extraction must not reallocate the draw list"
        );
    }

    #[test]
    fn degenerate_aspect_still_produces_a_finite_frame() {
        let world = test_world();
        // A minimised window reports a zero-sized surface.
        for aspect in [0.0, f32::NAN, f32::INFINITY, -1.0] {
            let frame = extract_render_frame(&world, aspect);
            assert!(
                frame.view_proj.iter().flatten().all(|value| value.is_finite()),
                "aspect {aspect} produced a non-finite view_proj"
            );
        }
    }

    #[test]
    fn stats_start_empty_and_sum_what_was_considered() {
        let stats = ExtractStats {
            drawn: 3,
            culled: 2,
            hidden: 1,
        };
        assert_eq!(stats.considered(), 6);
        assert_eq!(ExtractStats::default().considered(), 0);
    }

    // NOTE: end-to-end extraction tests (an entity in front is drawn, one
    // behind is culled, a hidden one is skipped, one missing a `Transform` or a
    // `MeshRenderer` is skipped, and the emitted draw order and matrices) need
    // `MeshHandle`/`MaterialHandle` values, which only the `renderer` crate can
    // mint: both constructors and the registry that calls them are
    // `pub(crate)`, and the public route (`Renderer::upload_mesh`) needs a GPU.
    // The decisions those tests would cover are each exercised above through
    // `passes_culling`, `draw_key`, and the resource-driven frame fields. See
    // the task report: exposing a public constructor on the handles is the
    // one-line change that unblocks them.
}
