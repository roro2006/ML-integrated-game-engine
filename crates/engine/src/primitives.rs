//! Procedural low-poly primitive meshes.
//!
//! The starter kit the vertical slice is blocked out with, before authored
//! Blender assets arrive through the bake step (ADR-0011).
//!
//! # Conventions
//!
//! * Right-handed space, Y up, -Z forward, dimensions in metres.
//! * Triangles wind counter-clockwise seen from *outside* the surface, so
//!   back-face culling keeps the outside visible. Every primitive is covered by
//!   a winding test, because an inside-out mesh is silently invisible once
//!   culling is on rather than obviously wrong.
//! * Meshes are centred on the origin; [`plane`] is the exception and lies flat
//!   on XZ at `y = 0` so it can be used directly as ground.
//! * [`cube`], [`plane`], [`cylinder`] and [`cone`] are flat shaded: faces do
//!   not share vertices, so each face gets one constant normal and reads as a
//!   crisp facet. That is the look the art direction wants, and it also keeps
//!   these meshes free of the seam/averaging questions smooth normals raise.
//!   [`uv_sphere`] is the deliberate exception and uses smooth per-vertex
//!   normals (the radial direction), because a faceted sphere at blocking-out
//!   resolution reads as a bad sphere rather than as a stylised one.
//!
//! # Degenerate parameters
//!
//! Blocking-out geometry is often driven by content data, so bad numbers are
//! expected and must not take the frame down. Dimensions are clamped to
//! [`MIN_EXTENT`] (non-finite values, including `NaN`, clamp to it as well) and
//! segment/ring counts to [`MIN_SEGMENTS`]..=[`MAX_SEGMENTS`]. Every function
//! therefore returns a mesh that passes `MeshData::validate`, never a panic.
//!
//! # UV mapping
//!
//! The scheme is per-surface-kind and chosen to be predictable for tiling
//! block-out materials rather than to be atlas-friendly:
//!
//! * Flat faces (cube faces, plane, end caps) use planar projection along the
//!   face normal, spanning the whole `0..=1` range once.
//! * Cylinder and cone sides use cylindrical projection: `u` follows the
//!   azimuth once around, `v` runs 0 at the top to 1 at the bottom.
//! * The sphere uses the usual spherical projection: `u` is azimuth, `v` is
//!   polar angle from the north pole.
//!
//! `v` increases downward throughout, matching the top-left texture origin
//! wgpu samples with.

use std::f32::consts::{PI, TAU};

use renderer::MeshData;

/// Smallest accepted linear dimension, in metres (1 mm).
///
/// Zero-size meshes would produce degenerate triangles that break normal
/// computation and bounding volumes, so sizes clamp up to this instead.
pub const MIN_EXTENT: f32 = 0.001;

/// Fewest radial segments a round primitive is built with: below three there is
/// no enclosed volume at all.
pub const MIN_SEGMENTS: u32 = 3;

/// Upper bound on segments, rings, and plane subdivisions.
///
/// Block-out primitives are never the interesting part of the frame budget;
/// this caps a mistyped parameter at a few hundred thousand vertices instead of
/// letting it exhaust memory.
pub const MAX_SEGMENTS: u32 = 256;

/// A flat-shaded axis-aligned cube centred on the origin.
///
/// `size` is the full edge length in metres (a cube of size 2 spans -1..1 on
/// each axis), clamped to [`MIN_EXTENT`].
///
/// The six faces are fully split: 24 vertices, 12 triangles. Each face carries
/// its own outward normal and a planar `0..=1` UV patch.
pub fn cube(size: f32) -> MeshData {
    let half = clamp_extent(size) / 2.0;
    let mut builder = Builder::new();

    // (normal, u axis, v axis) per face, with `u cross v == normal`, which is
    // what makes the shared quad winding below come out counter-clockwise from
    // outside for every face.
    const FACES: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ];
    // Corner offsets along (u, v), counter-clockwise starting bottom-left.
    const CORNERS: [(f32, f32); 4] = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];

    for (normal, u_axis, v_axis) in FACES {
        let mut corner_indices = [0u32; 4];
        for (slot, (su, sv)) in CORNERS.iter().enumerate() {
            let position = [
                (normal[0] + su * u_axis[0] + sv * v_axis[0]) * half,
                (normal[1] + su * u_axis[1] + sv * v_axis[1]) * half,
                (normal[2] + su * u_axis[2] + sv * v_axis[2]) * half,
            ];
            let uv = [(su + 1.0) / 2.0, (1.0 - sv) / 2.0];
            corner_indices[slot] = builder.vertex(position, normal, uv);
        }
        builder.quad(corner_indices);
    }

    builder.into_mesh()
}

/// A subdivided ground quad lying on the XZ plane at `y = 0`, centred on the
/// origin in X and Z.
///
/// `size` is the full edge length in metres, clamped to [`MIN_EXTENT`].
/// `subdivisions` is the number of extra cuts per axis, so the grid has
/// `(subdivisions + 1)^2` quads; it is capped at [`MAX_SEGMENTS`].
///
/// Grid vertices are shared rather than duplicated per face. Every face normal
/// here is `+Y`, so sharing produces shading identical to per-face normals at a
/// quarter of the vertices. Subdivision exists so vertex-lit or
/// vertex-displaced ground does not have to be one enormous quad.
///
/// UVs are planar over the whole sheet: `u` follows `+X`, `v` follows `+Z`,
/// both spanning `0..=1` once, so tiling is a material-side decision.
pub fn plane(size: f32, subdivisions: u32) -> MeshData {
    let size = clamp_extent(size);
    let cells = subdivisions.min(MAX_SEGMENTS) + 1;
    let half = size / 2.0;
    let step = size / cells as f32;
    let mut builder = Builder::new();

    for row in 0..=cells {
        for column in 0..=cells {
            let u = column as f32 / cells as f32;
            let v = row as f32 / cells as f32;
            let position = [-half + column as f32 * step, 0.0, -half + row as f32 * step];
            builder.vertex(position, [0.0, 1.0, 0.0], [u, v]);
        }
    }

    let stride = cells + 1;
    for row in 0..cells {
        for column in 0..cells {
            let near_left = row * stride + column;
            let far_left = (row + 1) * stride + column;
            builder.quad([near_left, far_left, far_left + 1, near_left + 1]);
        }
    }

    builder.into_mesh()
}

/// A latitude/longitude sphere centred on the origin, with smooth normals.
///
/// `radius` is in metres and clamped to [`MIN_EXTENT`]. `segments` is the
/// number of azimuthal divisions and `rings` the number of latitude bands from
/// pole to pole; both are clamped to [`MIN_SEGMENTS`]..=[`MAX_SEGMENTS`].
///
/// Unlike the other primitives this one is **smooth shaded**: each vertex
/// normal is its own radial direction, so a low segment count still reads as a
/// round object instead of a gem. The cost is the usual UV sphere trade: the
/// seam column is duplicated for UV continuity, and each pole is duplicated per
/// column so the pole vertices get sensible `u` values.
///
/// Pole bands emit one triangle per segment instead of two, because the second
/// triangle of a polar quad collapses to zero area.
///
/// Produces `(rings + 1) * (segments + 1)` vertices and
/// `2 * segments * (rings - 1)` triangles.
pub fn uv_sphere(radius: f32, segments: u32, rings: u32) -> MeshData {
    let radius = clamp_extent(radius);
    let segments = clamp_segments(segments);
    let rings = clamp_segments(rings);
    let mut builder = Builder::new();

    for ring in 0..=rings {
        let v = ring as f32 / rings as f32;
        let (sin_polar, cos_polar) = (v * PI).sin_cos();
        // Snap the poles exactly: sin(PI) is not zero in f32, and a pole that
        // is a hair off-axis gives the adjacent triangles a wobbly normal.
        let (y, ring_radius) = match ring {
            0 => (radius, 0.0),
            r if r == rings => (-radius, 0.0),
            _ => (radius * cos_polar, radius * sin_polar),
        };
        for segment in 0..=segments {
            let u = segment as f32 / segments as f32;
            let (sin_azimuth, cos_azimuth) = (u * TAU).sin_cos();
            let position = [ring_radius * sin_azimuth, y, ring_radius * cos_azimuth];
            builder.vertex(position, normalize(position), [u, v]);
        }
    }

    let stride = segments + 1;
    for ring in 0..rings {
        for segment in 0..segments {
            let top_left = ring * stride + segment;
            let bottom_left = (ring + 1) * stride + segment;
            if ring != 0 {
                builder.triangle(top_left, bottom_left, top_left + 1);
            }
            if ring + 1 != rings {
                builder.triangle(bottom_left, bottom_left + 1, top_left + 1);
            }
        }
    }

    builder.into_mesh()
}

/// A flat-shaded closed cylinder centred on the origin, its axis along Y.
///
/// `radius` and `height` are in metres (height is the full extent, so the caps
/// sit at `±height / 2`) and clamp to [`MIN_EXTENT`]; `segments` clamps to
/// [`MIN_SEGMENTS`]..=[`MAX_SEGMENTS`].
///
/// Each side quad owns four vertices sharing one normal, the radial direction
/// at the quad's mid-angle, which is exactly the quad's plane normal. That is
/// what gives the barrel/pillar silhouette its readable facets. The caps are
/// triangle fans whose rim vertices are shared within a cap only, since a cap's
/// normal is constant.
///
/// Produces `6 * segments + 2` vertices and `4 * segments` triangles.
pub fn cylinder(radius: f32, height: f32, segments: u32) -> MeshData {
    let radius = clamp_extent(radius);
    let half_height = clamp_extent(height) / 2.0;
    let segments = clamp_segments(segments);
    let mut builder = Builder::new();

    for segment in 0..segments {
        let u0 = segment as f32 / segments as f32;
        let u1 = (segment + 1) as f32 / segments as f32;
        let (sin0, cos0) = (u0 * TAU).sin_cos();
        let (sin1, cos1) = (u1 * TAU).sin_cos();
        let (sin_mid, cos_mid) = ((u0 + u1) / 2.0 * TAU).sin_cos();
        let normal = [sin_mid, 0.0, cos_mid];

        let bottom0 = builder.vertex([radius * sin0, -half_height, radius * cos0], normal, [u0, 1.0]);
        let bottom1 = builder.vertex([radius * sin1, -half_height, radius * cos1], normal, [u1, 1.0]);
        let top1 = builder.vertex([radius * sin1, half_height, radius * cos1], normal, [u1, 0.0]);
        let top0 = builder.vertex([radius * sin0, half_height, radius * cos0], normal, [u0, 0.0]);
        builder.quad([bottom0, bottom1, top1, top0]);
    }

    builder.cap(radius, half_height, segments, Facing::Up);
    builder.cap(radius, -half_height, segments, Facing::Down);
    builder.into_mesh()
}

/// A flat-shaded closed cone centred on the origin, apex toward `+Y`.
///
/// `radius` is the base radius and `height` the full apex-to-base extent, both
/// in metres and clamped to [`MIN_EXTENT`]; the apex sits at `+height / 2` and
/// the base cap at `-height / 2`. `segments` clamps to
/// [`MIN_SEGMENTS`]..=[`MAX_SEGMENTS`].
///
/// Every side triangle owns its three vertices and uses its true geometric
/// normal, so the cone reads as a faceted spike — the intended look for
/// blocked-out roofs, trees and spell markers. The apex is therefore duplicated
/// per side triangle, which is also what avoids the pinched-normal artefact a
/// shared apex would create.
///
/// Side UVs are cylindrical with the apex pinned to the middle of its
/// triangle's `u` span; the base cap is planar.
///
/// Produces `4 * segments + 1` vertices and `2 * segments` triangles.
pub fn cone(radius: f32, height: f32, segments: u32) -> MeshData {
    let radius = clamp_extent(radius);
    let half_height = clamp_extent(height) / 2.0;
    let segments = clamp_segments(segments);
    let apex = [0.0, half_height, 0.0];
    let mut builder = Builder::new();

    for segment in 0..segments {
        let u0 = segment as f32 / segments as f32;
        let u1 = (segment + 1) as f32 / segments as f32;
        let (sin0, cos0) = (u0 * TAU).sin_cos();
        let (sin1, cos1) = (u1 * TAU).sin_cos();
        let base0 = [radius * sin0, -half_height, radius * cos0];
        let base1 = [radius * sin1, -half_height, radius * cos1];
        let normal = normalize(cross(subtract(base0, apex), subtract(base1, apex)));

        let a = builder.vertex(apex, normal, [(u0 + u1) / 2.0, 0.0]);
        let b = builder.vertex(base0, normal, [u0, 1.0]);
        let c = builder.vertex(base1, normal, [u1, 1.0]);
        builder.triangle(a, b, c);
    }

    builder.cap(radius, -half_height, segments, Facing::Down);
    builder.into_mesh()
}

/// Which way a disc cap looks, which decides both its normal and its winding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Facing {
    Up,
    Down,
}

/// Accumulates vertices and indices, handing out indices as vertices are added.
#[derive(Debug)]
struct Builder {
    mesh: MeshData,
}

impl Builder {
    fn new() -> Self {
        Self {
            mesh: MeshData::default(),
        }
    }

    fn vertex(&mut self, position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> u32 {
        let index = self.mesh.positions.len() as u32;
        self.mesh.positions.push(position);
        self.mesh.normals.push(normal);
        self.mesh.uvs.push(uv);
        index
    }

    fn triangle(&mut self, a: u32, b: u32, c: u32) {
        self.mesh.indices.extend_from_slice(&[a, b, c]);
    }

    /// Splits a convex quad given counter-clockwise from outside into the two
    /// triangles that share its first corner, preserving that winding.
    fn quad(&mut self, corners: [u32; 4]) {
        let [a, b, c, d] = corners;
        self.triangle(a, b, c);
        self.triangle(a, c, d);
    }

    /// Appends a horizontal disc at `y` as a triangle fan around its centre.
    ///
    /// The fan is wound outward for `facing`, and its rim vertices are shared
    /// within the cap: a cap's normal is constant, so sharing them costs no
    /// shading fidelity.
    fn cap(&mut self, radius: f32, y: f32, segments: u32, facing: Facing) {
        let normal = match facing {
            Facing::Up => [0.0, 1.0, 0.0],
            Facing::Down => [0.0, -1.0, 0.0],
        };
        let centre = self.vertex([0.0, y, 0.0], normal, [0.5, 0.5]);
        let first_rim = self.mesh.positions.len() as u32;

        for segment in 0..segments {
            let (sin, cos) = (segment as f32 / segments as f32 * TAU).sin_cos();
            // Planar projection down the cap normal. The bottom cap mirrors `v`
            // so its texture is not seen mirrored from below.
            let v = match facing {
                Facing::Up => 0.5 - cos / 2.0,
                Facing::Down => 0.5 + cos / 2.0,
            };
            self.vertex([radius * sin, y, radius * cos], normal, [0.5 + sin / 2.0, v]);
        }

        for segment in 0..segments {
            let rim = first_rim + segment;
            let next_rim = first_rim + (segment + 1) % segments;
            match facing {
                Facing::Up => self.triangle(centre, rim, next_rim),
                Facing::Down => self.triangle(centre, next_rim, rim),
            }
        }
    }

    fn into_mesh(self) -> MeshData {
        self.mesh
    }
}

/// Clamps a linear dimension to something a mesh can actually be built from.
///
/// `NaN` falls through `f32::max` to [`MIN_EXTENT`], and infinities are caught
/// explicitly, so no caller input can produce non-finite geometry.
fn clamp_extent(metres: f32) -> f32 {
    if metres.is_finite() {
        metres.max(MIN_EXTENT)
    } else {
        MIN_EXTENT
    }
}

fn clamp_segments(count: u32) -> u32 {
    count.clamp(MIN_SEGMENTS, MAX_SEGMENTS)
}

fn subtract(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalizes a vector, falling back to `+Y` for a zero-length input so a
/// degenerate face still yields a unit normal rather than `NaN`.
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if length > 0.0 {
        [v[0] / length, v[1] / length, v[2] / length]
    } else {
        [0.0, 1.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named_primitives() -> Vec<(String, MeshData)> {
        let mut meshes = vec![
            ("cube(2.0)".to_string(), cube(2.0)),
            ("cube(0.25)".to_string(), cube(0.25)),
            ("plane(4.0, 0)".to_string(), plane(4.0, 0)),
            ("plane(10.0, 7)".to_string(), plane(10.0, 7)),
            ("cone(0.5, 1.5, 5)".to_string(), cone(0.5, 1.5, 5)),
        ];
        for segments in [3, 4, 8, 17] {
            meshes.push((
                format!("uv_sphere(1.0, {segments}, 3)"),
                uv_sphere(1.0, segments, 3),
            ));
            meshes.push((
                format!("uv_sphere(2.0, {segments}, 9)"),
                uv_sphere(2.0, segments, 9),
            ));
            meshes.push((
                format!("cylinder(1.0, 2.0, {segments})"),
                cylinder(1.0, 2.0, segments),
            ));
            meshes.push((format!("cone(1.0, 2.0, {segments})"), cone(1.0, 2.0, segments)));
        }
        meshes
    }

    fn assert_near(actual: f32, expected: f32, what: &str) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "{what}: expected {expected}, got {actual}"
        );
    }

    fn length(v: [f32; 3]) -> f32 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[test]
    fn every_primitive_validates() {
        for (name, mesh) in named_primitives() {
            assert_eq!(mesh.validate(), Ok(()), "{name} should be a valid mesh");
            assert!(mesh.triangle_count() > 0, "{name} should have triangles");
        }
    }

    #[test]
    fn every_normal_is_unit_length() {
        for (name, mesh) in named_primitives() {
            for (index, &normal) in mesh.normals.iter().enumerate() {
                assert_near(length(normal), 1.0, &format!("{name} normal {index} length"));
            }
        }
    }

    /// The load-bearing test: a mesh wound the wrong way is not visibly wrong,
    /// it is invisible once back-face culling is enabled. Comparing each
    /// triangle's geometric normal (edge cross product, which encodes winding)
    /// against its stored vertex normals catches that, and catches zero-area
    /// triangles at the same time.
    #[test]
    fn every_triangle_winds_counter_clockwise_from_outside() {
        for (name, mesh) in named_primitives() {
            for triangle in 0..mesh.triangle_count() {
                let corner = |offset: usize| mesh.positions[mesh.indices[triangle * 3 + offset] as usize];
                let geometric = cross(subtract(corner(1), corner(0)), subtract(corner(2), corner(0)));
                assert!(
                    length(geometric) > 1e-9,
                    "{name} triangle {triangle} is degenerate"
                );
                let geometric = normalize(geometric);
                for offset in 0..3 {
                    let normal = mesh.normals[mesh.indices[triangle * 3 + offset] as usize];
                    assert!(
                        dot(geometric, normal) > 0.0,
                        "{name} triangle {triangle} corner {offset} faces inward \
                         (geometric {geometric:?} vs vertex {normal:?})"
                    );
                }
            }
        }
    }

    #[test]
    fn every_uv_is_inside_the_unit_square() {
        for (name, mesh) in named_primitives() {
            for (index, &[u, v]) in mesh.uvs.iter().enumerate() {
                assert!(
                    (-1e-5..=1.0 + 1e-5).contains(&u) && (-1e-5..=1.0 + 1e-5).contains(&v),
                    "{name} uv {index} is outside 0..=1: [{u}, {v}]"
                );
            }
        }
    }

    #[test]
    fn cube_has_twenty_four_vertices_and_twelve_triangles() {
        let mesh = cube(1.0);
        assert_eq!(mesh.vertex_count(), 24);
        assert_eq!(mesh.triangle_count(), 12);
    }

    #[test]
    fn cube_has_six_distinct_flat_faces() {
        let mesh = cube(2.0);
        for axis_normal in [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ] {
            let matching = mesh.normals.iter().filter(|&&n| n == axis_normal).count();
            assert_eq!(matching, 4, "expected 4 vertices with normal {axis_normal:?}");
        }
    }

    #[test]
    fn parameterised_counts_match_the_documented_formulas() {
        // plane: (subdivisions + 1)^2 quads over a shared (subdivisions + 2)^2 grid.
        let mesh = plane(1.0, 1);
        assert_eq!(mesh.vertex_count(), 9);
        assert_eq!(mesh.triangle_count(), 8);
        let mesh = plane(1.0, 3);
        assert_eq!(mesh.vertex_count(), 25);
        assert_eq!(mesh.triangle_count(), 32);

        // sphere: (rings + 1) * (segments + 1) vertices, 2 * segments * (rings - 1) triangles.
        let mesh = uv_sphere(1.0, 8, 4);
        assert_eq!(mesh.vertex_count(), 45);
        assert_eq!(mesh.triangle_count(), 48);

        // cylinder: 6 * segments + 2 vertices, 4 * segments triangles.
        let mesh = cylinder(1.0, 1.0, 8);
        assert_eq!(mesh.vertex_count(), 50);
        assert_eq!(mesh.triangle_count(), 32);

        // cone: 4 * segments + 1 vertices, 2 * segments triangles.
        let mesh = cone(1.0, 1.0, 8);
        assert_eq!(mesh.vertex_count(), 33);
        assert_eq!(mesh.triangle_count(), 16);
    }

    #[test]
    fn bounds_match_requested_dimensions() {
        let bounds = cube(2.0).bounds().expect("cube has vertices");
        assert_eq!(bounds.min, [-1.0, -1.0, -1.0]);
        assert_eq!(bounds.max, [1.0, 1.0, 1.0]);

        let bounds = plane(4.0, 3).bounds().expect("plane has vertices");
        assert_eq!(bounds.min, [-2.0, 0.0, -2.0]);
        assert_eq!(bounds.max, [2.0, 0.0, 2.0]);

        let bounds = uv_sphere(1.5, 16, 8).bounds().expect("sphere has vertices");
        for axis in 0..3 {
            assert_near(bounds.min[axis], -1.5, "sphere min");
            assert_near(bounds.max[axis], 1.5, "sphere max");
        }

        // 16 segments put vertices exactly on ±X and ±Z.
        let bounds = cylinder(0.5, 2.0, 16).bounds().expect("cylinder has vertices");
        assert_near(bounds.min[0], -0.5, "cylinder min x");
        assert_near(bounds.max[0], 0.5, "cylinder max x");
        assert_near(bounds.min[1], -1.0, "cylinder min y");
        assert_near(bounds.max[1], 1.0, "cylinder max y");
        assert_near(bounds.min[2], -0.5, "cylinder min z");
        assert_near(bounds.max[2], 0.5, "cylinder max z");

        let bounds = cone(0.5, 2.0, 16).bounds().expect("cone has vertices");
        assert_near(bounds.min[0], -0.5, "cone min x");
        assert_near(bounds.max[0], 0.5, "cone max x");
        assert_near(bounds.min[1], -1.0, "cone base y");
        assert_near(bounds.max[1], 1.0, "cone apex y");
    }

    #[test]
    fn plane_lies_flat_on_xz_facing_up() {
        let mesh = plane(3.0, 2);
        assert!(mesh.positions.iter().all(|p| p[1] == 0.0));
        assert!(mesh.normals.iter().all(|&n| n == [0.0, 1.0, 0.0]));
    }

    #[test]
    fn sphere_normals_are_smooth_radial_directions() {
        let radius = 2.0;
        let mesh = uv_sphere(radius, 12, 6);
        for (position, normal) in mesh.positions.iter().zip(&mesh.normals) {
            for axis in 0..3 {
                assert_near(normal[axis], position[axis] / radius, "sphere radial normal");
            }
        }
    }

    #[test]
    fn generation_is_deterministic() {
        assert_eq!(cube(1.25), cube(1.25));
        assert_eq!(plane(8.0, 5), plane(8.0, 5));
        assert_eq!(uv_sphere(0.75, 11, 7), uv_sphere(0.75, 11, 7));
        assert_eq!(cylinder(0.3, 1.8, 9), cylinder(0.3, 1.8, 9));
        assert_eq!(cone(0.3, 1.8, 9), cone(0.3, 1.8, 9));
    }

    #[test]
    fn degenerate_parameters_clamp_instead_of_panicking() {
        let degenerate = [
            ("cube(0.0)", cube(0.0)),
            ("cube(-4.0)", cube(-4.0)),
            ("cube(NaN)", cube(f32::NAN)),
            ("cube(inf)", cube(f32::INFINITY)),
            ("plane(0.0, 0)", plane(0.0, 0)),
            ("plane(NaN, u32::MAX)", plane(f32::NAN, u32::MAX)),
            ("uv_sphere(0.0, 0, 0)", uv_sphere(0.0, 0, 0)),
            ("uv_sphere(-1.0, 2, 1)", uv_sphere(-1.0, 2, 1)),
            ("cylinder(0.0, 0.0, 0)", cylinder(0.0, 0.0, 0)),
            ("cylinder(NaN, -2.0, 1)", cylinder(f32::NAN, -2.0, 1)),
            ("cone(-1.0, inf, 2)", cone(-1.0, f32::INFINITY, 2)),
            ("cone(0.0, 0.0, u32::MAX)", cone(0.0, 0.0, u32::MAX)),
        ];
        for (name, mesh) in degenerate {
            assert_eq!(mesh.validate(), Ok(()), "{name} should still be valid");
            assert!(
                mesh.positions.iter().flatten().all(|c| c.is_finite()),
                "{name} should have finite positions"
            );
        }

        // Clamping is to the documented minimums, not to something arbitrary.
        assert_eq!(
            uv_sphere(1.0, 0, 0).vertex_count(),
            uv_sphere(1.0, 3, 3).vertex_count()
        );
        assert_eq!(
            cone(1.0, 1.0, u32::MAX).triangle_count(),
            2 * MAX_SEGMENTS as usize
        );
        let bounds = cube(0.0).bounds().expect("clamped cube has vertices");
        assert_near(bounds.max[0], MIN_EXTENT / 2.0, "clamped cube half extent");
    }
}
