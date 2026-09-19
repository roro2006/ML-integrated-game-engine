//! Engine math: transforms, projections, and culling volumes.
//!
//! Conventions are engine-owned (ADR-0008) and every item here obeys them:
//!
//! - Right-handed, **Y up**, **−Z forward**, world units in metres.
//! - Clip-space depth is `0..1` (wgpu/DirectX), not OpenGL's `-1..1`.
//! - Counter-clockwise winding is front-facing.
//!
//! glam supplies the arithmetic but none of the conventions: it offers an
//! `opengl` (`-1..1`), a `directx` (`0..1`, Y up), and a `vulkan` (`0..1`, Y
//! down) projection family, and choosing the wrong one clips or flips the
//! whole scene with no obvious symptom. The helpers below exist so that choice
//! is made once, here, and so game code goes through `engine::math` rather
//! than calling glam directly.

use std::f32::consts::PI;

// `directx` is the right family despite the name: it is the WebGPU convention
// (glam marks it `#[doc(alias = "webgpu")]`), which is what wgpu normalises to
// on every backend. Reaching for `vulkan` because the engine targets
// Vulkan-capable hardware would invert Y in clip space.
use glam::camera::rh::proj::directx;
use glam::camera::rh::view;

pub use glam::{Mat3, Mat4, Quat, Vec2, Vec3, Vec4};

/// Floors applied to degenerate projection input.
///
/// Resizing to a zero-sized window (minimise, or a drag that momentarily
/// reports 0×0) is normal, and a zero aspect ratio or zero near distance would
/// otherwise produce a matrix full of infinities and NaNs. NaN propagates into
/// culling, where it silently drops geometry, so the projection helpers clamp
/// instead of panicking: the frame is degenerate either way, but the matrix
/// stays finite and invertible and the next resize recovers.
const MIN_ASPECT: f32 = 1e-4;
const MIN_FOV: f32 = 1e-4;
const MIN_NEAR: f32 = 1e-4;
const MIN_DEPTH_RANGE: f32 = 1e-4;
const MIN_ORTHO_EXTENT: f32 = 1e-4;

/// Returns `value` clamped to `min..=max`, substituting `min` when `value` is
/// NaN or infinite (`f32::clamp` would propagate a NaN).
fn clamp_finite(value: f32, min: f32, max: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        min
    }
}

/// Returns `value` if it is finite, otherwise `fallback`.
fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

/// Picks an up vector that is usable as a basis hint for `forward`.
///
/// A caller-supplied up that is parallel to the view direction (looking
/// straight down, with up still `Y`) makes the cross product zero and the
/// resulting basis NaN, so fall back to an axis that is not parallel.
fn basis_up(forward: Vec3, up: Vec3) -> Vec3 {
    let up = up.try_normalize().unwrap_or(Vec3::Y);
    if up.cross(forward).length_squared() > 1e-12 {
        up
    } else if forward.cross(Vec3::Y).length_squared() > 1e-12 {
        Vec3::Y
    } else {
        Vec3::Z
    }
}

/// Position, orientation, and scale of something in world space.
///
/// This is the runtime (ECS) representation only. Canonical domain state
/// stores plain numeric fields and never these types (ADR-0008).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            ..Self::IDENTITY
        }
    }

    pub fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        Self::from_translation(Vec3::new(x, y, z))
    }

    #[must_use]
    pub fn with_rotation(self, rotation: Quat) -> Self {
        Self { rotation, ..self }
    }

    #[must_use]
    pub fn with_scale(self, scale: Vec3) -> Self {
        Self { scale, ..self }
    }

    /// Model matrix, applying scale, then rotation, then translation.
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// Inverse-transpose of the model matrix, for transforming normals.
    ///
    /// Under non-uniform scale a normal transformed by the model matrix is no
    /// longer perpendicular to the surface (a squashed sphere lights wrongly),
    /// so normals need the inverse-transpose of the linear part instead.
    ///
    /// Only the 3×3 part is meaningful; it is widened to a `Mat4` because the
    /// per-draw uniform block stores it that way for std140-style alignment
    /// (`docs/plans/m3-renderer.md`). Widening a `Mat3` rather than inverting
    /// the full `Mat4` also keeps the translation row clean, so a shader may
    /// use either `mat3(normal_matrix)` or `normal_matrix * vec4(n, 0.0)`.
    ///
    /// A zero component in `scale` makes the linear part singular; rather than
    /// return infinities, fall back to the rotation alone.
    pub fn normal_matrix(&self) -> Mat4 {
        let linear = Mat3::from_mat4(self.to_matrix());
        if linear.determinant().abs() < 1e-12 {
            return Mat4::from_quat(self.rotation);
        }
        Mat4::from_mat3(linear.inverse().transpose())
    }

    /// Facing direction: −Z rotated into world space.
    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }

    /// +X rotated into world space.
    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    /// +Y rotated into world space.
    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }

    /// Orients this transform so [`forward`](Self::forward) points at `target`.
    ///
    /// A `target` equal to the current translation leaves the rotation
    /// untouched rather than producing NaN, so a follow camera that catches up
    /// to its subject does not corrupt the frame.
    #[must_use]
    pub fn looking_at(self, target: Vec3, up: Vec3) -> Self {
        let Some(forward) = (target - self.translation).try_normalize() else {
            return self;
        };
        // `look_to_rh` builds a world-to-view rotation; the object's own
        // orientation is its inverse, which for a rotation is its transpose.
        let basis = view::look_to_mat3(forward, basis_up(forward, up)).transpose();
        self.with_rotation(Quat::from_mat3(&basis))
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Right-handed perspective projection with a `0..1` depth range.
///
/// View-space −`near` maps to NDC z `0.0` and −`far` to `1.0`, which is what
/// the `Depth32Float` target and `Less` depth compare in
/// `docs/plans/m3-renderer.md` expect.
///
/// Degenerate input is clamped, never rejected: `aspect` is floored at
/// `1e-4` (a zero-sized window reports `0`), `fov_y_radians` is kept inside
/// `(0, PI)`, `near` is floored at `1e-4` (a right-handed perspective divide
/// requires a positive near distance), and `far` is kept at least `1e-4`
/// beyond `near`. Non-finite input collapses to the same floors.
pub fn perspective(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let fov_y = clamp_finite(fov_y_radians, MIN_FOV, PI - MIN_FOV);
    let aspect = clamp_finite(aspect, MIN_ASPECT, f32::MAX);
    let near = clamp_finite(near, MIN_NEAR, f32::MAX);
    let far = clamp_finite(far, near + MIN_DEPTH_RANGE, f32::MAX);
    directx::perspective(fov_y, aspect, near, far)
}

/// Right-handed orthographic projection with a `0..1` depth range.
///
/// View-space −`near` maps to NDC z `0.0` and −`far` to `1.0`, matching
/// [`perspective`].
///
/// Unlike [`perspective`], `near` may be zero or negative: there is no
/// perspective divide, so a box that starts behind the eye is well defined
/// (useful for shadow and UI volumes). Only the degenerate cases are clamped —
/// each of `right`, `top`, and `far` is held at least `1e-4` above `left`,
/// `bottom`, and `near` respectively, and non-finite input falls back to `0`.
pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat4 {
    let left = finite_or(left, 0.0);
    let right = clamp_finite(right, left + MIN_ORTHO_EXTENT, f32::MAX);
    let bottom = finite_or(bottom, 0.0);
    let top = clamp_finite(top, bottom + MIN_ORTHO_EXTENT, f32::MAX);
    let near = finite_or(near, 0.0);
    let far = clamp_finite(far, near + MIN_DEPTH_RANGE, f32::MAX);
    directx::orthographic(left, right, bottom, top, near, far)
}

/// Right-handed view matrix for an eye at `eye` looking at `target`.
///
/// The resulting view space keeps the engine convention: +X right, +Y up, and
/// the scene in front of the camera at negative Z.
///
/// Degenerate input is clamped rather than fatal: a `target` on top of `eye`
/// yields a translation-only view matrix, and an `up` parallel to the view
/// direction is replaced by an axis that is not.
pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    let Some(forward) = (target - eye).try_normalize() else {
        return Mat4::from_translation(-eye);
    };
    view::look_to_mat4(eye, forward, basis_up(forward, up))
}

/// An axis-aligned bounding box, used for culling and spatial queries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// Bounds enclosing every point, or `None` if the iterator is empty.
    pub fn from_points(points: impl IntoIterator<Item = Vec3>) -> Option<Self> {
        let mut points = points.into_iter();
        let first = points.next()?;
        let mut bounds = Self {
            min: first,
            max: first,
        };
        for point in points {
            bounds.min = bounds.min.min(point);
            bounds.max = bounds.max.max(point);
        }
        Some(bounds)
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Smallest box containing both inputs.
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn contains_point(&self, point: Vec3) -> bool {
        point.cmpge(self.min).all() && point.cmple(self.max).all()
    }

    /// Axis-aligned bounds of this box after an **affine** transform.
    ///
    /// The transformed box is generally not axis-aligned, so this returns the
    /// bounds enclosing it — the same result as transforming all eight corners
    /// and re-fitting, but with three absolute-value dot products instead. It
    /// assumes the last row of `matrix` is `(0, 0, 0, 1)`; feeding it a
    /// projection matrix would skip the perspective divide and give nonsense.
    pub fn transformed(&self, matrix: &Mat4) -> Self {
        let center = matrix.transform_point3(self.center());
        let linear = Mat3::from_mat4(*matrix);
        let abs = Mat3::from_cols(linear.x_axis.abs(), linear.y_axis.abs(), linear.z_axis.abs());
        let half_extents = abs * self.half_extents();
        Self {
            min: center - half_extents,
            max: center + half_extents,
        }
    }
}

/// Converts mesh-space bounds from the renderer's hand-off format.
///
/// `renderer::MeshData` stores plain arrays so mesh producers need no math
/// dependency; the conversion belongs here, at the boundary where bounds
/// become cullable.
impl From<renderer::Aabb> for Aabb {
    fn from(bounds: renderer::Aabb) -> Self {
        Self {
            min: Vec3::from_array(bounds.min),
            max: Vec3::from_array(bounds.max),
        }
    }
}

/// The six clip planes of a view-projection matrix.
///
/// Planes are stored as `(normal.xyz, distance)` with unit-length normals
/// pointing **into** the frustum, so a point is inside a plane when
/// `normal · point + distance >= 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frustum {
    /// Left, right, bottom, top, near, far.
    planes: [Vec4; 6],
}

impl Frustum {
    /// Extracts the six planes from a view-projection matrix (Gribb-Hartmann).
    ///
    /// Each plane is a sum or difference of the matrix rows, derived from the
    /// clip-space inequality it enforces. The **near plane is row 2 alone**,
    /// not `row3 + row2`: with a `0..1` depth range the near test is
    /// `z_clip >= 0`, whereas OpenGL's `-1..1` range makes it
    /// `z_clip >= -w`. Using the OpenGL form here would place the near plane
    /// behind the camera and quietly keep geometry that should be culled.
    pub fn from_view_projection(view_projection: &Mat4) -> Self {
        let m = view_projection;
        // glam stores columns, so a row is the same component of each column.
        let row = |i: usize| Vec4::new(m.x_axis[i], m.y_axis[i], m.z_axis[i], m.w_axis[i]);
        let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
        Self {
            planes: [
                normalize_plane(r3 + r0),
                normalize_plane(r3 - r0),
                normalize_plane(r3 + r1),
                normalize_plane(r3 - r1),
                normalize_plane(r2),
                normalize_plane(r3 - r2),
            ],
        }
    }

    /// Whether `bounds` may be visible.
    ///
    /// This is deliberately **conservative**: it rejects a box only when the
    /// box lies wholly outside a single plane. A box in the corner region
    /// outside two planes but outside neither alone is kept, and so is any box
    /// whose coordinates are NaN. Keeping something off-screen costs a draw
    /// call; culling something visible is a bug that shows as geometry
    /// vanishing at the edge of the screen, so every ambiguous case is kept.
    pub fn intersects_aabb(&self, bounds: &Aabb) -> bool {
        let center = bounds.center();
        let half_extents = bounds.half_extents();
        for plane in &self.planes {
            let normal = plane.truncate();
            // Projection of the box onto the plane normal: the "positive
            // vertex" test, without branching per axis.
            let radius = half_extents.dot(normal.abs());
            if normal.dot(center) + plane.w < -radius {
                return false;
            }
        }
        true
    }
}

/// Scales a plane so its normal is unit length.
///
/// A degenerate or non-finite matrix (a zero-sized window before clamping, a
/// matrix built from NaN) can yield a zero-length normal. Normalising that
/// would produce NaN, and NaN comparisons are false, which would silently cull
/// everything; return a plane that accepts every point instead.
fn normalize_plane(plane: Vec4) -> Vec4 {
    let length = plane.truncate().length();
    if length.is_finite() && length > 1e-12 {
        plane / length
    } else {
        Vec4::new(0.0, 0.0, 0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    const EPS: f32 = 1e-5;

    #[track_caller]
    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= EPS,
            "expected {expected}, got {actual}"
        );
    }

    #[track_caller]
    fn assert_vec_close(actual: Vec3, expected: Vec3) {
        assert!(
            actual.abs_diff_eq(expected, EPS),
            "expected {expected:?}, got {actual:?}"
        );
    }

    fn test_bounds(center: Vec3, half: f32) -> Aabb {
        Aabb {
            min: center - Vec3::splat(half),
            max: center + Vec3::splat(half),
        }
    }

    /// A view-projection for a camera at +5Z looking back at the origin.
    fn test_view_projection() -> Mat4 {
        perspective(60f32.to_radians(), 16.0 / 9.0, 0.1, 100.0)
            * look_at(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y)
    }

    #[test]
    fn identity_transform_round_trips() {
        assert_eq!(Transform::default(), Transform::IDENTITY);
        assert_eq!(Transform::IDENTITY.to_matrix(), Mat4::IDENTITY);
        assert_eq!(Transform::IDENTITY.normal_matrix(), Mat4::IDENTITY);

        let transform = Transform::from_xyz(1.0, 2.0, 3.0);
        assert_eq!(transform.translation, Vec3::new(1.0, 2.0, 3.0));
        assert_vec_close(
            transform.to_matrix().transform_point3(Vec3::ZERO),
            transform.translation,
        );

        // Builders compose without disturbing the other fields.
        let built = Transform::from_translation(Vec3::X)
            .with_rotation(Quat::from_rotation_y(FRAC_PI_2))
            .with_scale(Vec3::splat(2.0));
        assert_eq!(built.translation, Vec3::X);
        assert_eq!(built.scale, Vec3::splat(2.0));
    }

    #[test]
    fn identity_axes_follow_the_engine_convention() {
        let transform = Transform::IDENTITY;
        assert_vec_close(transform.forward(), Vec3::NEG_Z);
        assert_vec_close(transform.right(), Vec3::X);
        assert_vec_close(transform.up(), Vec3::Y);
    }

    #[test]
    fn yaw_rotates_forward_about_y() {
        // A +90° yaw in a right-handed, Y-up frame turns −Z toward −X.
        let transform = Transform::IDENTITY.with_rotation(Quat::from_rotation_y(FRAC_PI_2));
        assert_vec_close(transform.forward(), Vec3::NEG_X);
        assert_vec_close(transform.right(), Vec3::NEG_Z);
        assert_vec_close(transform.up(), Vec3::Y);
    }

    #[test]
    fn to_matrix_applies_scale_then_rotation_then_translation() {
        let transform = Transform::from_xyz(10.0, 0.0, 0.0)
            .with_rotation(Quat::from_rotation_y(FRAC_PI_2))
            .with_scale(Vec3::splat(2.0));
        // (1,0,0) scaled to (2,0,0), yawed to (0,0,-2), then translated.
        let point = transform.to_matrix().transform_point3(Vec3::X);
        assert_vec_close(point, Vec3::new(10.0, 0.0, -2.0));
    }

    #[test]
    fn normal_matrix_keeps_normals_perpendicular_under_non_uniform_scale() {
        let transform = Transform::from_xyz(3.0, -1.0, 2.0)
            .with_rotation(Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), 0.7))
            .with_scale(Vec3::new(2.0, 0.5, 3.0));

        // A surface tangent and its normal, perpendicular in object space.
        let normal = Vec3::new(1.0, 1.0, 0.0).normalize();
        let tangent = Vec3::new(1.0, -1.0, 0.0).normalize();
        assert_close(normal.dot(tangent), 0.0);

        let model = transform.to_matrix();
        let moved_tangent = model.transform_vector3(tangent);
        let moved_normal = Mat3::from_mat4(transform.normal_matrix()) * normal;
        assert_close(moved_normal.normalize().dot(moved_tangent.normalize()), 0.0);

        // The naive route is what the normal matrix exists to avoid: prove it
        // really does skew the normal off the surface here.
        let naive = model.transform_vector3(normal).normalize();
        assert!(
            naive.dot(moved_tangent.normalize()).abs() > 0.1,
            "test is toothless: the model matrix happened to preserve perpendicularity"
        );
    }

    #[test]
    fn normal_matrix_survives_zero_scale() {
        let transform = Transform::IDENTITY.with_scale(Vec3::new(1.0, 0.0, 1.0));
        assert!(
            transform
                .normal_matrix()
                .to_cols_array()
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn looking_at_points_forward_at_the_target() {
        let transform = Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y);
        assert_vec_close(transform.forward(), Vec3::NEG_Z);
        assert_vec_close(transform.up(), Vec3::Y);

        let transform = Transform::from_xyz(4.0, 0.0, 0.0).looking_at(Vec3::ZERO, Vec3::Y);
        assert_vec_close(transform.forward(), Vec3::NEG_X);

        // Straight down, with an up vector parallel to the view direction.
        let transform = Transform::from_xyz(0.0, 5.0, 0.0).looking_at(Vec3::ZERO, Vec3::Y);
        assert_vec_close(transform.forward(), Vec3::NEG_Y);

        // Degenerate target keeps the previous orientation instead of NaN.
        let rotation = Quat::from_rotation_y(FRAC_PI_2);
        let transform = Transform::IDENTITY
            .with_rotation(rotation)
            .looking_at(Vec3::ZERO, Vec3::Y);
        assert_eq!(transform.rotation, rotation);
    }

    /// Mandatory: a `-1..1` projection would place the near plane at NDC z
    /// `-1`, and every depth test in the renderer would be wrong.
    #[test]
    fn perspective_maps_near_to_zero_and_far_to_one() {
        let (near, far) = (0.1, 100.0);
        let projection = perspective(60f32.to_radians(), 16.0 / 9.0, near, far);
        let ndc_z = |view_z: f32| {
            let clip = projection * Vec4::new(0.0, 0.0, view_z, 1.0);
            clip.z / clip.w
        };

        assert_close(ndc_z(-near), 0.0);
        assert_close(ndc_z(-far), 1.0);
        // Depth grows with distance, so a `Less` compare keeps the nearest.
        assert!(ndc_z(-2.0) > ndc_z(-1.0));
        assert!((0.0..=1.0).contains(&ndc_z(-1.0)));
    }

    #[test]
    fn orthographic_maps_near_to_zero_and_far_to_one() {
        let (near, far) = (0.5, 40.0);
        let projection = orthographic(-10.0, 10.0, -10.0, 10.0, near, far);
        let ndc = |view_z: f32| {
            let clip = projection * Vec4::new(0.0, 0.0, view_z, 1.0);
            clip.z / clip.w
        };
        assert_close(ndc(-near), 0.0);
        assert_close(ndc(-far), 1.0);

        // Corners of the box map to the corners of NDC xy.
        let corner = projection * Vec4::new(10.0, 10.0, -near, 1.0);
        assert_close(corner.x, 1.0);
        assert_close(corner.y, 1.0);
    }

    #[test]
    fn projections_clamp_degenerate_input_instead_of_panicking() {
        // A minimised window: zero size, zero planes, reversed depth range.
        let cases = [
            perspective(0.0, 0.0, 0.0, 0.0),
            perspective(f32::NAN, f32::NAN, f32::NAN, f32::NAN),
            perspective(60f32.to_radians(), 16.0 / 9.0, 10.0, 1.0),
            orthographic(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            orthographic(f32::NAN, 1.0, 0.0, f32::INFINITY, 5.0, 1.0),
        ];
        for matrix in cases {
            assert!(
                matrix.to_cols_array().iter().all(|value| value.is_finite()),
                "degenerate projection produced a non-finite matrix: {matrix:?}"
            );
        }
        // Negative near is legal for an orthographic box (no perspective divide).
        let projection = orthographic(-1.0, 1.0, -1.0, 1.0, -5.0, 5.0);
        let clip = projection * Vec4::new(0.0, 0.0, 5.0, 1.0);
        assert_close(clip.z / clip.w, 0.0);
    }

    #[test]
    fn look_at_places_points_in_view_space() {
        let eye = Vec3::new(0.0, 0.0, 5.0);
        let view = look_at(eye, Vec3::ZERO, Vec3::Y);

        // The eye is the origin of view space, and the target is 5 m down −Z.
        assert_vec_close(view.transform_point3(eye), Vec3::ZERO);
        assert_vec_close(view.transform_point3(Vec3::ZERO), Vec3::new(0.0, 0.0, -5.0));
        // World +X stays right, world +Y stays up.
        assert_vec_close(
            view.transform_point3(Vec3::new(1.0, 2.0, 0.0)),
            Vec3::new(1.0, 2.0, -5.0),
        );

        // Looking down −X from +X, view-right is world −Z, so a point at world
        // +Z lands on view −X: the basis stays right-handed through the turn.
        let view = look_at(Vec3::new(5.0, 0.0, 0.0), Vec3::ZERO, Vec3::Y);
        assert_vec_close(view.transform_point3(Vec3::ZERO), Vec3::new(0.0, 0.0, -5.0));
        assert_vec_close(
            view.transform_point3(Vec3::new(0.0, 0.0, 5.0)),
            Vec3::new(-5.0, 0.0, -5.0),
        );
        assert_vec_close(
            view.transform_point3(Vec3::new(0.0, 1.0, 0.0)),
            Vec3::new(0.0, 1.0, -5.0),
        );

        // Degenerate eye/target pair stays finite.
        let view = look_at(eye, eye, Vec3::Y);
        assert!(view.to_cols_array().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn aabb_basics() {
        let bounds = Aabb::from_points([
            Vec3::new(1.0, -2.0, 3.0),
            Vec3::new(-1.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 5.0),
        ])
        .expect("three points produce bounds");
        assert_eq!(bounds.min, Vec3::new(-1.0, -2.0, 0.0));
        assert_eq!(bounds.max, Vec3::new(1.0, 2.0, 5.0));
        assert_vec_close(bounds.center(), Vec3::new(0.0, 0.0, 2.5));
        assert_vec_close(bounds.half_extents(), Vec3::new(1.0, 2.0, 2.5));

        assert!(Aabb::from_points([]).is_none());

        assert!(bounds.contains_point(bounds.center()));
        assert!(
            bounds.contains_point(bounds.min),
            "the surface counts as contained"
        );
        assert!(bounds.contains_point(bounds.max));
        assert!(!bounds.contains_point(Vec3::new(0.0, 0.0, -0.1)));

        let other = test_bounds(Vec3::new(10.0, 0.0, 0.0), 1.0);
        let union = bounds.union(&other);
        assert_eq!(union.min, Vec3::new(-1.0, -2.0, -1.0));
        assert_eq!(union.max, Vec3::new(11.0, 2.0, 5.0));
        assert_eq!(union, other.union(&bounds), "union is symmetric");
    }

    #[test]
    fn aabb_transformed_matches_brute_force_corners() {
        let bounds = Aabb {
            min: Vec3::new(-1.0, -2.0, -0.5),
            max: Vec3::new(2.0, 1.0, 3.0),
        };
        let matrix = Transform::from_xyz(4.0, -3.0, 1.5)
            .with_rotation(Quat::from_axis_angle(Vec3::new(0.3, 1.0, -0.6).normalize(), 1.1))
            .with_scale(Vec3::new(2.0, 0.25, 1.5))
            .to_matrix();

        let corners = (0..8).map(|i| {
            let pick = |bit: usize, axis: usize| {
                if i & (1 << bit) == 0 {
                    bounds.min[axis]
                } else {
                    bounds.max[axis]
                }
            };
            matrix.transform_point3(Vec3::new(pick(0, 0), pick(1, 1), pick(2, 2)))
        });
        let expected = Aabb::from_points(corners).expect("eight corners produce bounds");

        let actual = bounds.transformed(&matrix);
        assert_vec_close(actual.min, expected.min);
        assert_vec_close(actual.max, expected.max);

        // Identity is a no-op.
        assert_eq!(bounds.transformed(&Mat4::IDENTITY), bounds);
    }

    #[test]
    fn frustum_keeps_visible_boxes_and_culls_the_rest() {
        let frustum = Frustum::from_view_projection(&test_view_projection());

        assert!(
            frustum.intersects_aabb(&test_bounds(Vec3::ZERO, 0.5)),
            "a box at the point the camera looks at must be kept"
        );
        assert!(
            !frustum.intersects_aabb(&test_bounds(Vec3::new(0.0, 0.0, 20.0), 0.5)),
            "a box behind the camera must be culled"
        );
        assert!(
            frustum.intersects_aabb(&test_bounds(Vec3::new(0.0, 0.0, 4.9), 0.5)),
            "a box straddling the near plane is partly visible and must be kept"
        );
        assert!(
            !frustum.intersects_aabb(&test_bounds(Vec3::new(500.0, 0.0, 0.0), 0.5)),
            "a box far off to the side must be culled"
        );
        assert!(
            !frustum.intersects_aabb(&test_bounds(Vec3::new(0.0, 0.0, -200.0), 0.5)),
            "a box beyond the far plane must be culled"
        );
        assert!(
            !frustum.intersects_aabb(&test_bounds(Vec3::new(0.0, 400.0, 0.0), 0.5)),
            "a box far above must be culled"
        );
        assert!(
            frustum.intersects_aabb(&test_bounds(Vec3::ZERO, 1000.0)),
            "a box enclosing the whole camera must be kept"
        );
    }

    #[test]
    fn frustum_culling_is_conservative() {
        let frustum = Frustum::from_view_projection(&test_view_projection());

        // Sweep the volume the camera can see and confirm nothing visible is
        // dropped: any box containing a point that projects inside the unit
        // cube (with 0..1 depth) must be kept.
        let view_projection = test_view_projection();
        let mut checked = 0;
        for x in -6..=6 {
            for y in -6..=6 {
                for z in -12..=6 {
                    let point = Vec3::new(x as f32, y as f32, z as f32);
                    let clip = view_projection * point.extend(1.0);
                    if clip.w <= 0.0 {
                        continue;
                    }
                    let ndc = clip.truncate() / clip.w;
                    let visible = (-1.0..=1.0).contains(&ndc.x)
                        && (-1.0..=1.0).contains(&ndc.y)
                        && (0.0..=1.0).contains(&ndc.z);
                    if visible {
                        checked += 1;
                        assert!(
                            frustum.intersects_aabb(&test_bounds(point, 0.01)),
                            "culled a visible point at {point:?}"
                        );
                    }
                }
            }
        }
        assert!(
            checked > 20,
            "sweep found too few visible points to be meaningful"
        );
    }

    #[test]
    fn frustum_from_a_degenerate_matrix_keeps_everything() {
        // Zero-length plane normals must not normalise into NaN, because NaN
        // comparisons are false and would cull the entire scene.
        let frustum = Frustum::from_view_projection(&Mat4::ZERO);
        assert!(frustum.intersects_aabb(&test_bounds(Vec3::ZERO, 1.0)));
        assert!(frustum.intersects_aabb(&test_bounds(Vec3::new(1e6, 1e6, 1e6), 1.0)));

        // NaN bounds are kept rather than silently dropped.
        let frustum = Frustum::from_view_projection(&test_view_projection());
        assert!(frustum.intersects_aabb(&test_bounds(Vec3::splat(f32::NAN), 1.0)));
    }

    #[test]
    fn mesh_bounds_convert_from_the_renderer_hand_off() {
        let bounds = Aabb::from(renderer::Aabb {
            min: [-1.0, -2.0, -3.0],
            max: [1.0, 2.0, 3.0],
        });
        assert_eq!(bounds.min, Vec3::new(-1.0, -2.0, -3.0));
        assert_eq!(bounds.max, Vec3::new(1.0, 2.0, 3.0));
    }
}
