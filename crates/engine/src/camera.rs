//! Camera state and the fixed-angle follow rig.
//!
//! [`Camera`] holds the projection parameters and the eye/target pair the view
//! matrix is built from. [`FollowCamera`] is the rig that drives it for the
//! vertical slice: a perspective camera at a **fixed** pitch and yaw that
//! trails a target at a fixed distance and cannot be rotated by the player
//! (ADR-0007).
//!
//! Every matrix here goes through [`crate::math`], never glam directly. The
//! `0..1` depth range and the Y-up clip convention are pinned in that module
//! (ADR-0008); calling glam's projection helpers from here would silently pick
//! a different convention and break clipping in a way that looks like an art
//! bug.
//!
//! # Timing domain
//!
//! The rig belongs to the **variable-rate frame domain**: drive
//! [`FollowCamera::follow`] from a [`FrameStage`](crate::FrameStage) system with
//! [`FrameTime::delta`](crate::time::FrameTime::delta), after the fixed steps
//! that moved the subject. The camera is presentation only, so it should update
//! at display rate to stay smooth on a high-refresh monitor. Nothing inside the
//! fixed-step simulation may read it, or the frame rate would leak into
//! gameplay — the same reason combat windows are counted in ticks (ADR-0007).
//! Smoothing is exponential decay rather than a per-frame lerp factor precisely
//! so that a variable delta still produces frame-rate-independent motion.

use std::f32::consts::{FRAC_PI_2, PI};

use crate::ecs::Resource;
use crate::math::{Frustum, Mat4, Vec3, look_at, perspective};

/// Floors and limits applied to degenerate rig input.
///
/// Content and gameplay code can hand the rig a zero distance, a straight-down
/// pitch, or a NaN target (an entity that has not been placed yet). None of
/// those may reach the view matrix as an unusable basis: NaN propagates into
/// culling, where it silently drops geometry.
const MIN_DISTANCE: f32 = 0.01;
/// Just short of straight down, so the view direction is never parallel to the
/// world up axis.
const MAX_PITCH: f32 = FRAC_PI_2 - 1e-3;

/// Returns `value` if it is finite, otherwise `fallback`.
fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

/// Returns `value` if every component is finite, otherwise `fallback`.
fn finite_vec_or(value: Vec3, fallback: Vec3) -> Vec3 {
    if value.is_finite() { value } else { fallback }
}

/// Perspective camera: projection parameters plus the view it is looking from.
///
/// The view is stored as an eye/target pair rather than a [`Transform`] because
/// that is what a follow rig naturally produces, and because
/// [`crate::math::look_at`] already resolves the degenerate cases (target on
/// top of the eye, up parallel to the view direction).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    /// Vertical field of view in radians.
    pub fov_y_radians: f32,
    pub near: f32,
    pub far: f32,
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
}

impl Resource for Camera {}

impl Camera {
    /// 50°, a moderately long lens. A wide fov exaggerates the perspective of
    /// a fixed-angle view and makes the ground plane read as sloped.
    pub const DEFAULT_FOV_Y: f32 = PI * 50.0 / 180.0;
    /// Close enough for the camera to sit inside a small room, far enough to
    /// keep depth precision usable across the far plane.
    pub const DEFAULT_NEAR: f32 = 0.1;
    pub const DEFAULT_FAR: f32 = 250.0;

    /// A camera at the origin looking down −Z with the given projection.
    ///
    /// Degenerate values are not rejected here; [`crate::math::perspective`]
    /// clamps them when the matrix is built, so a camera configured from
    /// content data can never take the frame down.
    pub fn new(fov_y_radians: f32, near: f32, far: f32) -> Self {
        Self {
            fov_y_radians,
            near,
            far,
            eye: Vec3::ZERO,
            target: Vec3::NEG_Z,
            up: Vec3::Y,
        }
    }

    /// Places the eye and aims it at `target`.
    #[must_use]
    pub fn looking_from(self, eye: Vec3, target: Vec3) -> Self {
        Self { eye, target, ..self }
    }

    /// World-to-view matrix.
    pub fn view_matrix(&self) -> Mat4 {
        look_at(self.eye, self.target, self.up)
    }

    /// View-to-clip matrix for a viewport of the given aspect ratio (width /
    /// height). A zero aspect (a minimised window) is clamped, not rejected.
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        perspective(self.fov_y_radians, aspect, self.near, self.far)
    }

    /// World-to-clip matrix: what the per-frame uniform block carries.
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection_matrix(aspect) * self.view_matrix()
    }

    /// Culling volume for this camera at the given aspect ratio.
    pub fn frustum(&self, aspect: f32) -> Frustum {
        Frustum::from_view_projection(&self.view_projection(aspect))
    }

    /// Direction the camera is looking, or −Z if the eye sits on the target.
    pub fn forward(&self) -> Vec3 {
        (self.target - self.eye).try_normalize().unwrap_or(Vec3::NEG_Z)
    }
}

impl Default for Camera {
    /// The default projection, posed by the default [`FollowCamera`], so a
    /// world that has a camera resource but no rig system still renders a
    /// sensible view of the origin.
    fn default() -> Self {
        let mut camera = Self::new(Self::DEFAULT_FOV_Y, Self::DEFAULT_NEAR, Self::DEFAULT_FAR);
        FollowCamera::default().apply_to(&mut camera);
        camera
    }
}

/// The drawable area, in physical pixels.
///
/// Published by the runner when the surface is created and on every resize,
/// so a game can turn a cursor position into a point in the world. A world
/// with no window — a headless test — has none, and anything that needs one
/// must do without.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
}

impl Resource for Viewport {}

impl Viewport {
    /// Width over height, or `None` for a zero-sized (minimised) surface.
    pub fn aspect(self) -> Option<f32> {
        (self.width > 0 && self.height > 0).then(|| self.width as f32 / self.height as f32)
    }
}

impl Camera {
    /// The ray from the eye through a pixel, as an origin and a unit
    /// direction. `pixel` is in physical pixels from the top-left, which is
    /// what the platform reports for the cursor.
    ///
    /// `None` for a zero-sized viewport or a degenerate camera, rather than a
    /// ray full of NaNs.
    pub fn ray_through(&self, viewport: Viewport, pixel: (f64, f64)) -> Option<(Vec3, Vec3)> {
        let aspect = viewport.aspect()?;
        let x = (pixel.0 as f32 / viewport.width as f32) * 2.0 - 1.0;
        // Pixels grow downward; clip space grows upward.
        let y = 1.0 - (pixel.1 as f32 / viewport.height as f32) * 2.0;
        let inverse = self.view_projection(aspect).inverse();
        if !inverse.is_finite() {
            return None;
        }
        // wgpu's clip space puts depth in 0..1: the near plane is 0.
        let near = inverse.project_point3(Vec3::new(x, y, 0.0));
        let far = inverse.project_point3(Vec3::new(x, y, 1.0));
        let direction = (far - near).try_normalize()?;
        near.is_finite().then_some((near, direction))
    }

    /// Where the ray through a pixel meets the horizontal plane at `height`,
    /// or `None` if it never does — looking level or up, or behind the eye.
    pub fn ground_point(&self, viewport: Viewport, pixel: (f64, f64), height: f32) -> Option<Vec3> {
        let (origin, direction) = self.ray_through(viewport, pixel)?;
        if direction.y.abs() < 1e-6 {
            return None;
        }
        let distance = (height - origin.y) / direction.y;
        (distance > 0.0).then(|| origin + direction * distance)
    }
}

/// Fixed-angle 2.5D follow rig (ADR-0007).
///
/// Pitch and yaw are fixed by the game, not by the player: the camera orbits
/// nothing, it only translates to keep its subject framed. That is what lets
/// low-poly models and animations be authored to read from a single angle.
///
/// The rig owns a smoothed `target` that chases the subject's position; the eye
/// is derived from it, so the whole rig is described by the subject position
/// plus four constants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FollowCamera {
    /// Downward tilt in radians. Positive looks down at the ground.
    pub pitch: f32,
    /// Rotation about world +Y in radians. Zero looks down −Z, so the eye sits
    /// on the +Z side of its subject.
    pub yaw: f32,
    /// Distance from the focus point to the eye, in metres.
    pub distance: f32,
    /// Offset added to the followed position before framing, so the camera
    /// looks at the subject's chest rather than its feet.
    pub focus_offset: Vec3,
    /// Smoothed subject position. Not the point the camera looks at; see
    /// [`focus`](Self::focus).
    pub target: Vec3,
    /// Seconds for the rig to close **half** the remaining distance to its
    /// subject. Zero or negative disables smoothing (the rig snaps).
    ///
    /// A half-life is the honest unit for exponential decay: unlike a "lerp
    /// speed" it means the same thing at any frame rate.
    pub smoothing_half_life: f32,
}

impl Resource for FollowCamera {}

impl FollowCamera {
    /// 50° down, framing roughly two thirds ground and one third horizon.
    pub const DEFAULT_PITCH: f32 = PI * 50.0 / 180.0;
    /// 12 m back reads the player at about a sixth of screen height with the
    /// default 50° lens: enough context to see an incoming enemy group.
    pub const DEFAULT_DISTANCE: f32 = 12.0;
    /// Short enough that the camera never feels detached from the player, long
    /// enough to absorb a dodge's acceleration spike.
    pub const DEFAULT_HALF_LIFE: f32 = 0.12;

    /// A rig at the default angle, already framing `target` with no catch-up
    /// left to do.
    pub fn new(target: Vec3) -> Self {
        Self {
            target: finite_vec_or(target, Vec3::ZERO),
            ..Self::default()
        }
    }

    /// The point the camera looks at: the smoothed subject plus the offset.
    pub fn focus(&self) -> Vec3 {
        finite_vec_or(self.target, Vec3::ZERO) + finite_vec_or(self.focus_offset, Vec3::ZERO)
    }

    /// Direction the camera looks, from the fixed pitch and yaw.
    ///
    /// Yaw `0` and pitch `0` give −Z, matching [`Transform::forward`] for an
    /// unrotated transform (`crate::math`), so the rig's angles read the same
    /// way as any other orientation in the engine.
    ///
    /// [`Transform::forward`]: crate::math::Transform::forward
    pub fn forward(&self) -> Vec3 {
        let pitch = finite_or(self.pitch, 0.0).clamp(-MAX_PITCH, MAX_PITCH);
        let (sin_pitch, cos_pitch) = pitch.sin_cos();
        let (sin_yaw, cos_yaw) = finite_or(self.yaw, 0.0).sin_cos();
        Vec3::new(-sin_yaw * cos_pitch, -sin_pitch, -cos_yaw * cos_pitch)
    }

    /// Where the camera sits: back along [`forward`](Self::forward) from the
    /// focus point.
    pub fn eye(&self) -> Vec3 {
        let distance = finite_or(self.distance, Self::DEFAULT_DISTANCE).max(MIN_DISTANCE);
        self.focus() - self.forward() * distance
    }

    /// Jumps the rig to `target` with no smoothing.
    ///
    /// Use this whenever the subject teleports — spawn, respawn, load, a scene
    /// change — so the camera does not sweep across the level.
    pub fn snap_to(&mut self, target: Vec3) {
        self.target = finite_vec_or(target, self.target);
    }

    /// Advances the smoothed target toward `target` over `delta_seconds`.
    ///
    /// Frame-rate independence comes from exponential decay: the fraction of
    /// the gap that *remains* is `2^(-dt / half_life)`, and because
    /// `2^a · 2^b = 2^(a+b)`, splitting a step into pieces gives the same
    /// result as taking it whole (up to float rounding). A naive
    /// `lerp(current, target, 0.1)` per frame would instead make the camera
    /// lag twice as far behind at 30 fps as at 60.
    ///
    /// Given the same starting state and the same sequence of deltas this is
    /// deterministic, so a replayed frame sequence reproduces the same framing.
    pub fn follow(&mut self, target: Vec3, delta_seconds: f32) {
        let target = finite_vec_or(target, self.target);
        let half_life = finite_or(self.smoothing_half_life, 0.0);
        let delta = finite_or(delta_seconds, 0.0).max(0.0);
        if half_life <= 0.0 {
            self.target = target;
            return;
        }
        // `exp2` of a large negative exponent underflows to 0.0, which is the
        // correct limit (snap), not a NaN.
        let remaining = (-delta / half_life).exp2();
        self.target = target + (finite_vec_or(self.target, target) - target) * remaining;
    }

    /// Poses `camera` from this rig, leaving its projection untouched.
    pub fn apply_to(&self, camera: &mut Camera) {
        camera.eye = self.eye();
        camera.target = self.focus();
        camera.up = Vec3::Y;
    }
}

impl Default for FollowCamera {
    fn default() -> Self {
        Self {
            pitch: Self::DEFAULT_PITCH,
            yaw: 0.0,
            distance: Self::DEFAULT_DISTANCE,
            focus_offset: Vec3::new(0.0, 1.0, 0.0),
            target: Vec3::ZERO,
            smoothing_half_life: Self::DEFAULT_HALF_LIFE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec4;

    const EPS: f32 = 1e-5;

    #[track_caller]
    fn assert_vec_close(actual: Vec3, expected: Vec3) {
        assert!(
            actual.abs_diff_eq(expected, EPS),
            "expected {expected:?}, got {actual:?}"
        );
    }

    #[track_caller]
    fn assert_finite(matrix: Mat4) {
        assert!(
            matrix.to_cols_array().iter().all(|value| value.is_finite()),
            "matrix was not finite: {matrix:?}"
        );
    }

    /// Projects a world point to NDC, returning `None` if it is behind the eye.
    fn project(camera: &Camera, aspect: f32, point: Vec3) -> Option<Vec3> {
        let clip: Vec4 = camera.view_projection(aspect) * point.extend(1.0);
        (clip.w > 0.0).then(|| clip.truncate() / clip.w)
    }

    fn test_camera() -> Camera {
        Camera::new(90f32.to_radians(), 0.1, 100.0).looking_from(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO)
    }

    #[test]
    fn view_matrix_places_world_points_in_view_space() {
        let camera = test_camera();
        let view = camera.view_matrix();

        // The eye is the origin of view space and the target lies 5 m down −Z.
        assert_vec_close(view.transform_point3(camera.eye), Vec3::ZERO);
        assert_vec_close(view.transform_point3(Vec3::ZERO), Vec3::new(0.0, 0.0, -5.0));
        // World +X stays right and world +Y stays up, per the engine convention.
        assert_vec_close(
            view.transform_point3(Vec3::new(2.0, 1.0, 0.0)),
            Vec3::new(2.0, 1.0, -5.0),
        );
        assert_vec_close(camera.forward(), Vec3::NEG_Z);
    }

    #[test]
    fn projection_keeps_the_zero_to_one_depth_convention() {
        let camera = test_camera();
        // The point the camera looks at lands dead centre of the viewport.
        let centre = project(&camera, 16.0 / 9.0, Vec3::ZERO).expect("target is in front");
        assert_vec_close(centre, Vec3::new(0.0, 0.0, centre.z));
        assert!(
            (0.0..=1.0).contains(&centre.z),
            "depth must be in 0..1, was {}",
            centre.z
        );

        // Near maps to 0 and far to 1: a `-1..1` projection would break every
        // depth test in the renderer (ADR-0008).
        let near = project(&camera, 1.0, Vec3::new(0.0, 0.0, 5.0 - camera.near)).expect("at near plane");
        let far = project(&camera, 1.0, Vec3::new(0.0, 0.0, 5.0 - camera.far)).expect("at far plane");
        assert!(near.z.abs() < 1e-4, "near mapped to {}", near.z);
        assert!((far.z - 1.0).abs() < 1e-4, "far mapped to {}", far.z);

        // A 90° vertical fov puts a point one unit right and one unit deep on
        // the right edge of a square viewport, and right of centre on a wide one.
        let edge = project(&camera, 1.0, Vec3::new(1.0, 0.0, 4.0)).expect("in front");
        assert!(
            (edge.x - 1.0).abs() < 1e-4,
            "expected the right edge, got {}",
            edge.x
        );
        let wide = project(&camera, 16.0 / 9.0, Vec3::new(1.0, 0.0, 4.0)).expect("in front");
        assert!(wide.x > 0.0 && wide.x < edge.x, "a wider viewport sees more");

        // Points above the camera project upward, not downward: a Y-flipped
        // clip space renders a correct scene upside down.
        let above = project(&camera, 1.0, Vec3::new(0.0, 1.0, 4.0)).expect("in front");
        assert!(above.y > 0.0, "world +Y must project to clip +Y, got {}", above.y);

        // Behind the eye there is no valid projection at all.
        assert!(project(&camera, 1.0, Vec3::new(0.0, 0.0, 6.0)).is_none());
    }

    #[test]
    fn view_projection_is_projection_times_view() {
        let camera = test_camera();
        let aspect = 16.0 / 9.0;
        assert_eq!(
            camera.view_projection(aspect),
            camera.projection_matrix(aspect) * camera.view_matrix()
        );
        // And the frustum is built from exactly that matrix.
        assert_eq!(
            camera.frustum(aspect),
            Frustum::from_view_projection(&camera.view_projection(aspect))
        );
    }

    #[test]
    fn fixed_rig_sits_above_and_behind_its_target() {
        let rig = FollowCamera {
            pitch: 45f32.to_radians(),
            yaw: 0.0,
            distance: 10.0,
            focus_offset: Vec3::ZERO,
            target: Vec3::new(3.0, 0.0, -2.0),
            smoothing_half_life: 0.0,
        };

        // 45° down and 10 m back: equal parts up and toward +Z.
        let offset = 10.0 * std::f32::consts::FRAC_1_SQRT_2;
        assert_vec_close(rig.focus(), Vec3::new(3.0, 0.0, -2.0));
        assert_vec_close(rig.eye(), Vec3::new(3.0, offset, -2.0 + offset));
        assert_vec_close(rig.forward(), Vec3::new(0.0, -offset, -offset).normalize());

        // The eye stays exactly `distance` from the focus point, whatever the
        // angles, and always looks straight at it.
        for pitch in [0.0, 20f32.to_radians(), 80f32.to_radians()] {
            for yaw in [0.0, FRAC_PI_2, PI, -2.0] {
                let rig = FollowCamera { pitch, yaw, ..rig };
                let to_focus = rig.focus() - rig.eye();
                assert!((to_focus.length() - rig.distance).abs() < 1e-4);
                assert_vec_close(to_focus.normalize(), rig.forward());
            }
        }
    }

    #[test]
    fn yaw_swings_the_eye_around_the_target_without_moving_it() {
        let rig = FollowCamera {
            pitch: 0.0,
            yaw: 0.0,
            distance: 4.0,
            focus_offset: Vec3::ZERO,
            target: Vec3::ZERO,
            smoothing_half_life: 0.0,
        };
        // Yaw 0 looks down −Z, so the eye is on the +Z side.
        assert_vec_close(rig.eye(), Vec3::new(0.0, 0.0, 4.0));
        // +90° yaw turns the view toward −X (matching `Transform::forward`),
        // putting the eye on +X.
        let turned = FollowCamera {
            yaw: FRAC_PI_2,
            ..rig
        };
        assert_vec_close(turned.forward(), Vec3::NEG_X);
        assert_vec_close(turned.eye(), Vec3::new(4.0, 0.0, 0.0));
    }

    #[test]
    fn focus_offset_raises_the_aim_point_and_the_eye_together() {
        let rig = FollowCamera {
            focus_offset: Vec3::new(0.0, 1.5, 0.0),
            target: Vec3::new(1.0, 0.0, 0.0),
            ..FollowCamera::default()
        };
        let flat = FollowCamera {
            focus_offset: Vec3::ZERO,
            ..rig
        };
        assert_vec_close(rig.focus(), flat.focus() + Vec3::new(0.0, 1.5, 0.0));
        assert_vec_close(rig.eye(), flat.eye() + Vec3::new(0.0, 1.5, 0.0));

        let mut camera = Camera::default();
        rig.apply_to(&mut camera);
        assert_vec_close(camera.eye, rig.eye());
        assert_vec_close(camera.target, rig.focus());
        assert_eq!(camera.up, Vec3::Y);
    }

    #[test]
    fn smoothing_converges_on_the_target() {
        let mut rig = FollowCamera {
            smoothing_half_life: 0.1,
            target: Vec3::ZERO,
            ..FollowCamera::default()
        };
        let subject = Vec3::new(10.0, 0.0, 0.0);

        // One half-life closes exactly half the gap: that is what the unit means.
        rig.follow(subject, 0.1);
        assert_vec_close(rig.target, Vec3::new(5.0, 0.0, 0.0));
        rig.follow(subject, 0.1);
        assert_vec_close(rig.target, Vec3::new(7.5, 0.0, 0.0));

        // It keeps approaching monotonically and never overshoots. Once the gap
        // reaches float epsilon it stops shrinking, which is convergence rather
        // than a stall, so a strict decrease is only required while there is
        // still a meaningful gap left to close.
        let mut previous = (subject - rig.target).length();
        for step in 0..200 {
            rig.follow(subject, 1.0 / 60.0);
            let gap = (subject - rig.target).length();
            assert!(
                gap <= previous,
                "smoothing reversed at step {step}: {previous} to {gap}"
            );
            if previous > 1e-4 {
                assert!(gap < previous, "smoothing stalled at {gap}");
            }
            assert!(
                rig.target.x <= subject.x,
                "smoothing overshot to {:?}",
                rig.target
            );
            previous = gap;
        }
        assert!(previous < 1e-3, "did not converge; gap {previous}");

        // A zero delta is a no-op rather than a jump.
        let held = rig.target;
        rig.follow(subject, 0.0);
        assert_eq!(rig.target, held);
    }

    #[test]
    fn smoothing_is_frame_rate_independent() {
        let subject = Vec3::new(4.0, 2.0, -7.0);
        let rig = FollowCamera {
            smoothing_half_life: 0.15,
            target: Vec3::ZERO,
            ..FollowCamera::default()
        };

        let mut coarse = rig;
        coarse.follow(subject, 0.016);

        let mut fine = rig;
        fine.follow(subject, 0.008);
        fine.follow(subject, 0.008);

        // Exponential decay composes exactly, so the two paths agree to float
        // rounding. A per-frame lerp factor would leave these visibly apart.
        assert!(
            coarse.target.abs_diff_eq(fine.target, 1e-5),
            "16ms gave {:?}, 2x8ms gave {:?}",
            coarse.target,
            fine.target
        );

        // The same holds across a whole second of wildly different pacing.
        let mut slow = rig;
        let mut fast = rig;
        for _ in 0..30 {
            slow.follow(subject, 1.0 / 30.0);
        }
        for _ in 0..240 {
            fast.follow(subject, 1.0 / 240.0);
        }
        assert!(
            slow.target.abs_diff_eq(fast.target, 1e-4),
            "30fps gave {:?}, 240fps gave {:?}",
            slow.target,
            fast.target
        );

        // Determinism: replaying the same deltas reproduces the same framing.
        let mut replay = rig;
        for _ in 0..30 {
            replay.follow(subject, 1.0 / 30.0);
        }
        assert_eq!(replay.target, slow.target);
    }

    #[test]
    fn snapping_skips_smoothing_entirely() {
        let mut rig = FollowCamera::default();
        rig.snap_to(Vec3::new(100.0, 0.0, 0.0));
        assert_vec_close(rig.target, Vec3::new(100.0, 0.0, 0.0));

        // A rig configured with no smoothing tracks its subject exactly, which
        // is what a cutscene or a debug camera wants.
        let mut rig = FollowCamera {
            smoothing_half_life: 0.0,
            ..FollowCamera::default()
        };
        rig.follow(Vec3::new(-3.0, 1.0, 2.0), 1.0 / 60.0);
        assert_vec_close(rig.target, Vec3::new(-3.0, 1.0, 2.0));

        let mut rig = FollowCamera {
            smoothing_half_life: -1.0,
            ..FollowCamera::default()
        };
        rig.follow(Vec3::new(5.0, 0.0, 0.0), 1.0 / 60.0);
        assert_vec_close(rig.target, Vec3::new(5.0, 0.0, 0.0));
    }

    #[test]
    fn degenerate_camera_input_stays_finite() {
        // A minimised window reports a zero-sized surface, so aspect is 0.
        let camera = test_camera();
        assert_finite(camera.projection_matrix(0.0));
        assert_finite(camera.view_projection(0.0));
        assert_finite(camera.view_projection(f32::NAN));

        // Content-driven nonsense: inverted depth range, zero fov, NaN planes.
        let broken = Camera {
            fov_y_radians: 0.0,
            near: 10.0,
            far: 1.0,
            eye: Vec3::ZERO,
            target: Vec3::ZERO, // eye sits exactly on the target
            up: Vec3::ZERO,
        };
        assert_finite(broken.view_matrix());
        assert_finite(broken.view_projection(16.0 / 9.0));
        assert_vec_close(broken.forward(), Vec3::NEG_Z);

        let nan = Camera {
            fov_y_radians: f32::NAN,
            near: f32::NAN,
            far: f32::NAN,
            ..test_camera()
        };
        assert_finite(nan.view_projection(16.0 / 9.0));
    }

    #[test]
    fn degenerate_rig_input_stays_finite() {
        // Zero distance would put the eye on the target; NaN would poison the
        // whole view matrix and, through it, culling.
        let cases = [
            FollowCamera {
                distance: 0.0,
                ..FollowCamera::default()
            },
            FollowCamera {
                distance: f32::NAN,
                pitch: f32::NAN,
                yaw: f32::INFINITY,
                ..FollowCamera::default()
            },
            FollowCamera {
                pitch: FRAC_PI_2, // straight down, parallel to world up
                ..FollowCamera::default()
            },
            FollowCamera {
                target: Vec3::splat(f32::NAN),
                focus_offset: Vec3::splat(f32::NAN),
                ..FollowCamera::default()
            },
        ];
        for rig in cases {
            assert!(rig.eye().is_finite(), "eye was {:?} for {rig:?}", rig.eye());
            assert!(rig.focus().is_finite(), "focus was {:?}", rig.focus());
            assert!(rig.forward().is_finite(), "forward was {:?}", rig.forward());
            let mut camera = Camera::default();
            rig.apply_to(&mut camera);
            assert_finite(camera.view_matrix());
            assert_finite(camera.view_projection(16.0 / 9.0));
        }

        // A NaN subject position or delta leaves the rig where it was rather
        // than corrupting it for every later frame.
        let mut rig = FollowCamera::new(Vec3::new(1.0, 2.0, 3.0));
        rig.follow(Vec3::splat(f32::NAN), 1.0 / 60.0);
        assert_vec_close(rig.target, Vec3::new(1.0, 2.0, 3.0));
        rig.follow(Vec3::new(1.0, 2.0, 3.0), f32::NAN);
        assert_vec_close(rig.target, Vec3::new(1.0, 2.0, 3.0));
        rig.follow(Vec3::new(9.0, 9.0, 9.0), -1.0);
        assert_vec_close(rig.target, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn default_camera_frames_the_origin_from_the_rig_angle() {
        let camera = Camera::default();
        let rig = FollowCamera::default();
        assert_vec_close(camera.eye, rig.eye());
        assert_vec_close(camera.target, rig.focus());
        assert_eq!(camera.fov_y_radians, Camera::DEFAULT_FOV_Y);

        // Above and on the +Z side, looking down at the origin.
        assert!(camera.eye.y > 0.0 && camera.eye.z > 0.0);
        assert!(camera.forward().y < 0.0);
        // The subject is on screen from the default pose.
        let ndc = project(&camera, 16.0 / 9.0, Vec3::new(0.0, 1.0, 0.0)).expect("in front");
        assert!(ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0 && (0.0..=1.0).contains(&ndc.z));
    }

    // ---- picking the ground under a cursor --------------------------------

    fn rig_camera() -> Camera {
        let mut camera = Camera::default();
        let mut rig = FollowCamera::default();
        rig.snap_to(Vec3::new(4.0, 0.9, -3.0));
        rig.apply_to(&mut camera);
        camera
    }

    /// The pixel a world point is drawn at, in the platform's top-left
    /// convention.
    fn pixel_of(camera: &Camera, viewport: Viewport, point: Vec3) -> (f64, f64) {
        let aspect = viewport.aspect().expect("a real viewport");
        let ndc = project(camera, aspect, point).expect("in front of the eye");
        (
            f64::from((ndc.x + 1.0) * 0.5 * viewport.width as f32),
            f64::from((1.0 - ndc.y) * 0.5 * viewport.height as f32),
        )
    }

    #[test]
    fn a_cursor_over_a_ground_point_picks_that_point() {
        let camera = rig_camera();
        let viewport = Viewport {
            width: 1280,
            height: 720,
        };
        for point in [
            Vec3::new(4.0, 0.0, -3.0),
            Vec3::new(9.0, 0.0, -6.5),
            Vec3::new(-1.0, 0.0, 1.0),
            Vec3::new(4.5, 0.0, -9.0),
        ] {
            let pixel = pixel_of(&camera, viewport, point);
            let picked = camera
                .ground_point(viewport, pixel, 0.0)
                .expect("the ground is in view");
            assert!(
                (picked - point).length() < 1e-2,
                "the cursor over {point:?} picked {picked:?}"
            );
        }
    }

    #[test]
    fn pixels_grow_downward_and_right() {
        let camera = rig_camera();
        let viewport = Viewport {
            width: 800,
            height: 600,
        };
        let centre = camera
            .ground_point(viewport, (400.0, 300.0), 0.0)
            .expect("centre");
        let lower = camera.ground_point(viewport, (400.0, 500.0), 0.0).expect("lower");
        let right = camera.ground_point(viewport, (700.0, 300.0), 0.0).expect("right");
        // The rig looks down -Z from above and behind, unrotated: lower on
        // screen is nearer the camera (+Z), and right on screen is +X.
        assert!(lower.z > centre.z, "{lower:?} vs {centre:?}");
        assert!(right.x > centre.x, "{right:?} vs {centre:?}");
    }

    #[test]
    fn no_ground_point_without_a_viewport_or_a_plane_in_view() {
        let camera = rig_camera();
        let empty = Viewport { width: 0, height: 0 };
        assert!(camera.ground_point(empty, (0.0, 0.0), 0.0).is_none());
        // A camera looking level never meets the ground through its centre.
        let level = Camera::default().looking_from(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, 2.0, -10.0));
        let viewport = Viewport {
            width: 100,
            height: 100,
        };
        assert!(level.ground_point(viewport, (50.0, 50.0), 0.0).is_none());
        // And one looking down at a plane above its eye finds it behind.
        assert!(rig_camera().ground_point(viewport, (50.0, 50.0), 100.0).is_none());
    }
}
