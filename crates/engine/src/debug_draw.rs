//! Immediate-mode debug line drawing.
//!
//! [`DebugDraw`] is an ECS resource that any system may push lines into during
//! the frame. It accumulates world-space vertices and is drained into the
//! [`RenderFrame`] after extraction, which is what keeps extraction itself
//! read-only over the world: a system that wants to visualise something writes
//! it here, not into the render list.
//!
//! The API takes `engine::math` types — [`Vec3`], [`Aabb`], [`Transform`] — and
//! converts to plain arrays at the renderer boundary, exactly like the matrix
//! conversions in `render.rs`. That is what keeps `glam` out of the renderer's
//! signatures (ADR-0008).
//!
//! # Budget
//!
//! Debug drawing is unmetered by nature: a loop over every entity's bounds is
//! one line of code, and a bug in that loop is another. A vertex budget
//! ([`DEFAULT_VERTEX_BUDGET`]) caps what one frame can accumulate; lines past it
//! are dropped and counted rather than allocated. Debug tooling must not be
//! able to stall the frame it is diagnosing, and a visibly incomplete overlay
//! plus a rising drop count is a far better failure than a frame-time spike
//! that looks like a rendering problem.

use renderer::{DebugVertex, RenderFrame};

use crate::ecs::Resource;
use crate::math::{Aabb, Transform, Vec3, Vec4};

/// Vertices one frame of debug drawing may accumulate before lines are dropped.
///
/// 64k vertices is 32k line segments — far more than is readable on screen, so
/// hitting it means a loop is drawing more than someone intended.
pub const DEFAULT_VERTEX_BUDGET: usize = 64 * 1024;

/// Colour of the +X arm drawn by [`DebugDraw::axes`].
pub const AXIS_X_COLOR: Vec4 = Vec4::new(0.9, 0.2, 0.2, 1.0);
/// Colour of the +Y arm drawn by [`DebugDraw::axes`].
pub const AXIS_Y_COLOR: Vec4 = Vec4::new(0.3, 0.85, 0.3, 1.0);
/// Colour of the +Z arm drawn by [`DebugDraw::axes`].
pub const AXIS_Z_COLOR: Vec4 = Vec4::new(0.3, 0.45, 1.0, 1.0);

/// Linear RGBA colours for debug geometry.
///
/// Colours are [`Vec4`] rather than a new colour type or `renderer::Color`:
/// `Vec4` is already an `engine::math` type, so it composes with the rest of
/// the math API, and `renderer::Color` is `f64` and exists for the clear
/// colour. These are **linear** values; the surface is sRGB and the GPU
/// converts on write.
pub mod colors {
    use crate::math::Vec4;

    pub const WHITE: Vec4 = Vec4::new(1.0, 1.0, 1.0, 1.0);
    pub const RED: Vec4 = Vec4::new(0.9, 0.2, 0.2, 1.0);
    pub const GREEN: Vec4 = Vec4::new(0.3, 0.85, 0.3, 1.0);
    pub const BLUE: Vec4 = Vec4::new(0.3, 0.45, 1.0, 1.0);
    pub const YELLOW: Vec4 = Vec4::new(0.95, 0.85, 0.2, 1.0);
    /// The usual colour for a bounds box, dim enough not to dominate the scene.
    pub const BOUNDS: Vec4 = Vec4::new(0.2, 0.8, 0.9, 1.0);
}

/// Accumulates debug lines for the current frame.
///
/// Register it as a resource and call the methods from any system; the runner
/// drains it into the frame after extraction and the renderer draws the result
/// as an overlay that is occluded by the scene but never occludes it.
#[derive(Debug, Clone)]
pub struct DebugDraw {
    /// Pairs of vertices: every two entries are one `LineList` segment.
    vertices: Vec<DebugVertex>,
    budget: usize,
    dropped_lines: u32,
}

impl DebugDraw {
    pub fn new() -> Self {
        Self::with_vertex_budget(DEFAULT_VERTEX_BUDGET)
    }

    /// A resource with a non-default budget, in vertices.
    pub fn with_vertex_budget(budget: usize) -> Self {
        Self {
            vertices: Vec::new(),
            budget,
            dropped_lines: 0,
        }
    }

    /// Draws a line from `start` to `end` in world space.
    ///
    /// A line is all-or-nothing against the budget: half a segment would be a
    /// vertex the `LineList` pairs with the *next* line, drawing a stray
    /// diagonal across the scene.
    pub fn line(&mut self, start: Vec3, end: Vec3, color: Vec4) {
        if self.vertices.len() + 2 > self.budget {
            self.dropped_lines = self.dropped_lines.saturating_add(1);
            return;
        }
        // The boundary conversion: glam types stop here (ADR-0008).
        let color = color.to_array();
        self.vertices.push(DebugVertex::new(start.to_array(), color));
        self.vertices.push(DebugVertex::new(end.to_array(), color));
    }

    /// Draws the twelve edges of an axis-aligned box.
    ///
    /// The corners are indexed by bit: bit 0 selects `max.x` over `min.x`, bit 1
    /// `max.y`, bit 2 `max.z`. Two corners are joined by an edge exactly when
    /// they differ in one bit, so walking the three bits over the four corners
    /// that have each bit clear produces all twelve edges, once each, without a
    /// hand-written table to get wrong.
    pub fn wire_box(&mut self, bounds: &Aabb, color: Vec4) {
        let corner = |index: usize| {
            Vec3::new(
                if index & 1 == 0 {
                    bounds.min.x
                } else {
                    bounds.max.x
                },
                if index & 2 == 0 {
                    bounds.min.y
                } else {
                    bounds.max.y
                },
                if index & 4 == 0 {
                    bounds.min.z
                } else {
                    bounds.max.z
                },
            )
        };
        for axis in [1_usize, 2, 4] {
            for index in 0..8 {
                if index & axis == 0 {
                    self.line(corner(index), corner(index | axis), color);
                }
            }
        }
    }

    /// Draws a three-armed gizmo showing `transform`'s orientation, with arms
    /// `scale` metres long.
    ///
    /// The arms are the **+X, +Y, and +Z** basis vectors, so the blue arm points
    /// opposite [`Transform::forward`] — the engine looks down −Z (ADR-0008),
    /// and drawing the basis rather than the facing direction is what makes this
    /// gizmo comparable with every other engine's.
    ///
    /// `transform.scale` is deliberately ignored: the gizmo describes
    /// orientation, and folding in a non-uniform scale would skew the arms away
    /// from the axes they are labelling, while a zero scale would erase them.
    pub fn axes(&mut self, transform: &Transform, scale: f32) {
        let origin = transform.translation;
        self.line(origin, origin + transform.right() * scale, AXIS_X_COLOR);
        self.line(origin, origin + transform.up() * scale, AXIS_Y_COLOR);
        self.line(
            origin,
            origin + (transform.rotation * Vec3::Z) * scale,
            AXIS_Z_COLOR,
        );
    }

    /// Moves this frame's vertices into `frame` and leaves the resource empty.
    ///
    /// Called by the runner after extraction, which is what lets extraction stay
    /// read-only over the world. The frame's previous debug lines are dropped
    /// first and the two buffers are swapped rather than copied, so each side
    /// keeps its allocation and neither reallocates per frame.
    ///
    /// The drop counter is **not** reset: it is a cumulative total, so the
    /// engine can log it, or difference it per frame, after draining.
    pub fn drain_into(&mut self, frame: &mut RenderFrame) {
        frame.debug_lines.clear();
        std::mem::swap(&mut frame.debug_lines, &mut self.vertices);
    }

    /// Discards everything accumulated so far without drawing it.
    pub fn clear(&mut self) {
        self.vertices.clear();
    }

    /// Accumulated vertices. Two per line.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Lines dropped for exceeding the budget, since this resource was created.
    ///
    /// Cumulative and monotonic, like any counter: a number that stops rising
    /// means the budget is no longer being hit. Reported rather than logged
    /// here, so the engine's diagnostics layer owns the output.
    pub fn dropped_lines(&self) -> u32 {
        self.dropped_lines
    }

    /// The vertex budget this resource enforces.
    pub fn vertex_budget(&self) -> usize {
        self.budget
    }
}

impl Default for DebugDraw {
    fn default() -> Self {
        Self::new()
    }
}

impl Resource for DebugDraw {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Quat;
    use std::f32::consts::FRAC_PI_2;

    const EPS: f32 = 1e-5;

    #[track_caller]
    fn assert_vec_close(actual: [f32; 3], expected: Vec3) {
        let actual = Vec3::from_array(actual);
        assert!(
            actual.abs_diff_eq(expected, EPS),
            "expected {expected:?}, got {actual:?}"
        );
    }

    fn test_bounds() -> Aabb {
        Aabb {
            min: Vec3::new(-1.0, -2.0, -3.0),
            max: Vec3::new(4.0, 5.0, 6.0),
        }
    }

    #[test]
    fn a_line_becomes_two_vertices_carrying_its_colour() {
        let mut debug = DebugDraw::new();
        debug.line(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, -5.0, -6.0),
            colors::YELLOW,
        );

        assert_eq!(debug.vertex_count(), 2);
        assert!(!debug.is_empty());
        // Positions and colours survive the glam -> array conversion intact.
        assert_eq!(debug.vertices[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(debug.vertices[1].position, [-4.0, -5.0, -6.0]);
        assert_eq!(debug.vertices[0].color, colors::YELLOW.to_array());
        assert_eq!(debug.vertices[1].color, colors::YELLOW.to_array());
    }

    #[test]
    fn wire_box_emits_twelve_edges_matching_the_corners() {
        let bounds = test_bounds();
        let mut debug = DebugDraw::new();
        debug.wire_box(&bounds, colors::BOUNDS);

        assert_eq!(debug.vertex_count(), 24, "twelve edges, two vertices each");

        // Every vertex sits on a corner: each component is a min or a max.
        for vertex in &debug.vertices {
            for axis in 0..3 {
                let value = vertex.position[axis];
                assert!(
                    value == bounds.min[axis] || value == bounds.max[axis],
                    "vertex {vertex:?} is not on a corner of {bounds:?}"
                );
            }
        }

        // Every edge runs along exactly one axis: an edge differing in two
        // components would be a face diagonal, in three a body diagonal.
        let mut edges = Vec::new();
        for pair in debug.vertices.chunks_exact(2) {
            let differing = (0..3)
                .filter(|axis| pair[0].position[*axis] != pair[1].position[*axis])
                .count();
            assert_eq!(differing, 1, "{pair:?} is a diagonal, not an edge");

            // Normalised so the same edge drawn either way round compares equal.
            let mut ends = [pair[0].position, pair[1].position];
            ends.sort_by(|a, b| a.partial_cmp(b).expect("corner coordinates are finite"));
            edges.push(ends);
        }
        edges.sort_by(|a, b| a.partial_cmp(b).expect("corner coordinates are finite"));
        edges.dedup();
        assert_eq!(edges.len(), 12, "each of the twelve edges must be drawn once");
    }

    #[test]
    fn a_degenerate_box_still_draws_rather_than_panicking() {
        let point = Aabb {
            min: Vec3::ZERO,
            max: Vec3::ZERO,
        };
        let mut debug = DebugDraw::new();
        debug.wire_box(&point, colors::WHITE);
        assert_eq!(
            debug.vertex_count(),
            24,
            "zero-extent bounds collapse, not vanish"
        );
    }

    #[test]
    fn axes_are_coloured_and_oriented_by_the_transform() {
        let mut debug = DebugDraw::new();
        debug.axes(&Transform::from_xyz(10.0, 0.0, 0.0), 2.0);

        assert_eq!(debug.vertex_count(), 6, "three arms");
        let origin = Vec3::new(10.0, 0.0, 0.0);
        for arm in debug.vertices.chunks_exact(2) {
            assert_vec_close(arm[0].position, origin);
        }

        // Identity orientation: the arms are the world axes, scaled.
        assert_vec_close(debug.vertices[1].position, origin + Vec3::X * 2.0);
        assert_vec_close(debug.vertices[3].position, origin + Vec3::Y * 2.0);
        assert_vec_close(debug.vertices[5].position, origin + Vec3::Z * 2.0);

        assert_eq!(debug.vertices[1].color, AXIS_X_COLOR.to_array());
        assert_eq!(debug.vertices[3].color, AXIS_Y_COLOR.to_array());
        assert_eq!(debug.vertices[5].color, AXIS_Z_COLOR.to_array());
        // The three arms are distinguishable, which is the whole point.
        assert_ne!(AXIS_X_COLOR, AXIS_Y_COLOR);
        assert_ne!(AXIS_Y_COLOR, AXIS_Z_COLOR);
    }

    #[test]
    fn axes_follow_a_rotated_transform() {
        // A +90° yaw in a right-handed, Y-up frame turns +X toward −Z.
        let transform = Transform::IDENTITY.with_rotation(Quat::from_rotation_y(FRAC_PI_2));
        let mut debug = DebugDraw::new();
        debug.axes(&transform, 1.0);

        assert_vec_close(debug.vertices[1].position, Vec3::NEG_Z);
        assert_vec_close(debug.vertices[3].position, Vec3::Y);
        assert_vec_close(debug.vertices[5].position, Vec3::X);
    }

    #[test]
    fn scale_does_not_skew_the_gizmo() {
        let transform = Transform::IDENTITY.with_scale(Vec3::new(5.0, 0.0, 2.0));
        let mut debug = DebugDraw::new();
        debug.axes(&transform, 1.0);

        // A zero scale component must not erase an arm.
        assert_vec_close(debug.vertices[1].position, Vec3::X);
        assert_vec_close(debug.vertices[3].position, Vec3::Y);
        assert_vec_close(debug.vertices[5].position, Vec3::Z);
    }

    #[test]
    fn the_budget_drops_excess_lines_and_counts_them() {
        // Room for three lines exactly.
        let mut debug = DebugDraw::with_vertex_budget(6);
        assert_eq!(debug.vertex_budget(), 6);

        for _ in 0..3 {
            debug.line(Vec3::ZERO, Vec3::X, colors::WHITE);
        }
        assert_eq!(debug.vertex_count(), 6);
        assert_eq!(debug.dropped_lines(), 0, "nothing is dropped inside the budget");

        for _ in 0..4 {
            debug.line(Vec3::ZERO, Vec3::X, colors::WHITE);
        }
        assert_eq!(
            debug.vertex_count(),
            6,
            "the buffer must not grow past the budget"
        );
        assert_eq!(debug.dropped_lines(), 4);

        // A shape that cannot fit is dropped line by line, never half-drawn:
        // an odd vertex would pair with the next line and draw a stray diagonal.
        debug.wire_box(&test_bounds(), colors::BOUNDS);
        assert_eq!(debug.vertex_count() % 2, 0);
        assert_eq!(debug.dropped_lines(), 16, "twelve more edges dropped");
    }

    #[test]
    fn drain_into_moves_the_vertices_and_leaves_the_resource_empty() {
        let mut debug = DebugDraw::new();
        debug.line(Vec3::ZERO, Vec3::ONE, colors::RED);
        debug.line(Vec3::ZERO, Vec3::NEG_X, colors::GREEN);

        let mut frame = RenderFrame::default();
        debug.drain_into(&mut frame);

        assert_eq!(frame.debug_lines.len(), 4);
        assert_eq!(frame.debug_lines[0].color, colors::RED.to_array());
        assert_eq!(frame.debug_lines[2].color, colors::GREEN.to_array());
        assert!(debug.is_empty(), "the resource is empty after draining");
        assert_eq!(debug.vertex_count(), 0);

        // The next frame replaces the last one's lines rather than appending to
        // them, so a frame that draws nothing shows nothing.
        debug.line(Vec3::ZERO, Vec3::Y, colors::BLUE);
        debug.drain_into(&mut frame);
        assert_eq!(frame.debug_lines.len(), 2);
        assert_eq!(frame.debug_lines[0].color, colors::BLUE.to_array());

        debug.drain_into(&mut frame);
        assert!(
            frame.debug_lines.is_empty(),
            "draining an empty resource clears the frame"
        );
    }

    #[test]
    fn clear_discards_without_drawing() {
        let mut debug = DebugDraw::default();
        debug.wire_box(&test_bounds(), colors::BOUNDS);
        assert_eq!(debug.vertex_count(), 24);

        debug.clear();
        assert!(debug.is_empty());
        assert_eq!(debug.vertex_budget(), DEFAULT_VERTEX_BUDGET);
    }
}
