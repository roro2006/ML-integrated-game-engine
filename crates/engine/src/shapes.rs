//! Flat-shaded low-poly shapes with painted faces: the parts models are built
//! from (ADR-0017).
//!
//! [`crate::primitives`] is the block-out kit — one mesh per kind of thing,
//! uncoloured. This module is the modelling kit: a small, fixed set of
//! parametric shapes that a model recipe composes into a character, each face
//! painted from a [`FaceColors`] by which way it faces. The set is fixed on
//! purpose. A recipe — authored by hand or proposed by an AI — picks shapes and
//! numbers from this list and never supplies vertex data, which is what makes a
//! model something validation can bound.
//!
//! # Orientation is computed, not authored
//!
//! Every shape here is convex, so "outward" for a face is simply away from the
//! shape's centroid. Faces are described as polygons in whatever order is
//! convenient, and `Solid::into_mesh` turns each one to face outward before
//! triangulating it. Hand-written winding is how inside-out meshes happen, and
//! an inside-out mesh is invisible once back-face culling is on rather than
//! obviously wrong; computing it removes the chance.
//!
//! # Conventions
//!
//! As [`crate::primitives`]: right-handed, Y up, -Z forward, metres, flat
//! shaded — no face shares a vertex with another, so each face reads as one
//! crisp facet in one colour. Boxes, wedges and spheres are centred on the
//! origin; a frustum is centred on its axis and vertically, which for an
//! odd-sided one is not the centre of its bounding box. Dimensions clamp to
//! [`MIN_EXTENT`] and side counts to [`MIN_SIDES`]..=[`MAX_SIDES`], so every
//! input yields a valid mesh.

use std::f32::consts::TAU;

use renderer::MeshData;

use crate::math::{Mat3, Transform, Vec3};
use crate::primitives::MIN_EXTENT;

/// Fewest sides a round shape is built with.
pub const MIN_SIDES: u32 = 3;

/// Most sides a round shape is built with. Low-poly by construction: past this,
/// a cylinder stops reading as faceted, and triangle budgets stop meaning much.
pub const MAX_SIDES: u32 = 24;

/// Most subdivision levels a sphere accepts: 0 is an icosahedron (20 faces),
/// 1 is 80 faces.
pub const MAX_SPHERE_DETAIL: u32 = 1;

/// Most rings of facets a dome may have between its crown and its rim.
pub const MAX_DOME_RINGS: u32 = 6;

/// A face counts as "top" or "bottom" when its normal is within about 45
/// degrees of straight up or down.
const CAP_THRESHOLD: f32 = 0.7;

/// One parametric shape. Every variant is convex.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shape {
    /// An axis-aligned box of the given full extents.
    Box { size: Vec3 },
    /// A ramp: full height at the back (+Z), falling to nothing at the front
    /// (-Z). A roof, a blade's edge, a beak.
    Wedge { size: Vec3 },
    /// A vertical tapered column: an n-gon at the bottom and another at the
    /// top. With equal radii it is a cylinder, or a prism when `sides` is
    /// small; with a zero top radius it is a cone.
    Frustum {
        bottom_radius: f32,
        top_radius: f32,
        height: f32,
        sides: u32,
    },
    /// A faceted ball: an icosahedron, subdivided `detail` times.
    Sphere { radius: f32, detail: u32 },
    /// A box whose top differs from its bottom: `bottom` and `top` are full
    /// extents on X and Z. The trapezoid torso, the limb that thins toward the
    /// hand, the flared boot. A top of zero on one axis is a ridge, and on
    /// both a pyramid — a spike.
    Taper {
        bottom: crate::math::Vec2,
        top: crate::math::Vec2,
        height: f32,
    },
    /// A faceted half-sphere, flat side down: `segments` around, `rings` of
    /// facets from rim to crown. Pauldrons, helmets, knee plates, the crown of
    /// a head. One ring is a cone.
    Dome { radius: f32, segments: u32, rings: u32 },
}

/// The colour a face is painted, by which way it faces.
///
/// Three groups rather than one colour per face, because that is the grain a
/// recipe can sensibly describe — a hat's crown and brim, a boot's sole — and
/// because a per-face list would tie a recipe to one tessellation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceColors {
    pub top: [f32; 3],
    pub side: [f32; 3],
    pub bottom: [f32; 3],
}

impl FaceColors {
    /// Every face one colour.
    pub const fn uniform(color: [f32; 3]) -> Self {
        Self {
            top: color,
            side: color,
            bottom: color,
        }
    }

    fn for_normal(&self, normal: Vec3) -> [f32; 3] {
        if normal.y > CAP_THRESHOLD {
            self.top
        } else if normal.y < -CAP_THRESHOLD {
            self.bottom
        } else {
            self.side
        }
    }
}

/// Builds a shape's mesh, painted by `colors`.
pub fn mesh(shape: Shape, colors: FaceColors) -> MeshData {
    let solid = match shape {
        Shape::Box { size } => boxed(size),
        Shape::Wedge { size } => wedge(size),
        Shape::Frustum {
            bottom_radius,
            top_radius,
            height,
            sides,
        } => frustum(bottom_radius, top_radius, height, sides),
        Shape::Sphere { radius, detail } => sphere(radius, detail),
        Shape::Taper { bottom, top, height } => taper(bottom, top, height),
        Shape::Dome {
            radius,
            segments,
            rings,
        } => dome(radius, segments, rings),
    };
    solid.into_mesh(colors)
}

/// Bakes `transform` into a mesh's vertices: positions move, normals turn.
///
/// How a part is set on its slot. Normals go through the inverse-transpose so
/// they stay perpendicular under non-uniform scale, and are renormalised; a
/// degenerate transform falls back to rotation only, as
/// [`Transform::normal_matrix`] does, so no input produces a NaN normal.
pub fn placed(mut mesh: MeshData, transform: &Transform) -> MeshData {
    let matrix = transform.to_matrix();
    let normals = Mat3::from_mat4(transform.normal_matrix());
    for position in &mut mesh.positions {
        *position = matrix.transform_point3(Vec3::from(*position)).to_array();
    }
    for normal in &mut mesh.normals {
        let turned = (normals * Vec3::from(*normal)).normalize_or_zero();
        *normal = if turned == Vec3::ZERO {
            *normal
        } else {
            turned.to_array()
        };
    }
    mesh
}

/// Joins meshes into one, so everything a slot carries costs one draw.
///
/// Colours are carried per vertex; a mesh with none contributes white, so
/// merging a painted part with an unpainted one cannot shift the colours of
/// either.
pub fn merged(meshes: impl IntoIterator<Item = MeshData>) -> MeshData {
    let mut out = MeshData::default();
    for mesh in meshes {
        let base = out.positions.len() as u32;
        let count = mesh.positions.len();
        out.positions.extend_from_slice(&mesh.positions);
        out.normals.extend_from_slice(&mesh.normals);
        out.uvs.extend_from_slice(&mesh.uvs);
        if mesh.colors.len() == count {
            out.colors.extend_from_slice(&mesh.colors);
        } else {
            out.colors.extend(std::iter::repeat_n(renderer::WHITE, count));
        }
        out.indices.extend(mesh.indices.iter().map(|index| index + base));
    }
    out
}

/// Clamps a dimension, sending non-finite values to the minimum.
fn extent(metres: f32) -> f32 {
    if metres.is_finite() {
        metres.max(MIN_EXTENT)
    } else {
        MIN_EXTENT
    }
}

/// A convex polyhedron as corner points and polygons over them.
struct Solid {
    points: Vec<Vec3>,
    faces: Vec<Vec<usize>>,
}

impl Solid {
    fn into_mesh(self, colors: FaceColors) -> MeshData {
        let centroid = self.points.iter().copied().sum::<Vec3>() / self.points.len().max(1) as f32;
        let mut mesh = MeshData::default();

        for face in self.faces {
            // Corners that coincide — a taper's top shrunk to a point — are
            // one corner. Dropping repeats keeps every emitted triangle's
            // area non-zero, so every normal is a real direction; a polygon
            // left with fewer than three corners is no face at all.
            let mut face: Vec<usize> = face;
            face.dedup_by(|a, b| self.points[*a].distance_squared(self.points[*b]) < 1e-12);
            while face.len() > 1
                && self.points[face[0]].distance_squared(self.points[face[face.len() - 1]]) < 1e-12
            {
                face.pop();
            }
            if face.len() < 3 {
                continue;
            }
            let corners: Vec<Vec3> = face.iter().map(|&index| self.points[index]).collect();
            let mut normal = newell(&corners);
            if normal == Vec3::ZERO {
                continue;
            }
            let face_centre = corners.iter().copied().sum::<Vec3>() / corners.len() as f32;
            // Turn the polygon to face away from the centroid: see the module
            // docs for why this is computed rather than authored.
            if normal.dot(face_centre - centroid) < 0.0 {
                face.reverse();
                normal = -normal;
            }

            let color = colors.for_normal(normal);
            let base = mesh.positions.len() as u32;
            for &index in &face {
                mesh.positions.push(self.points[index].to_array());
                mesh.normals.push(normal.to_array());
                mesh.uvs.push([0.0, 0.0]);
                mesh.colors.push(color);
            }
            // A convex polygon fans cleanly from its first corner, keeping the
            // winding the reversal above established.
            for corner in 1..face.len() as u32 - 1 {
                mesh.indices
                    .extend_from_slice(&[base, base + corner, base + corner + 1]);
            }
        }
        mesh
    }
}

/// A polygon's unit normal by Newell's method, which is robust for any planar
/// polygon regardless of which corner is most convex. Zero for a degenerate
/// polygon.
fn newell(corners: &[Vec3]) -> Vec3 {
    let mut normal = Vec3::ZERO;
    for (index, current) in corners.iter().enumerate() {
        let next = corners[(index + 1) % corners.len()];
        normal.x += (current.y - next.y) * (current.z + next.z);
        normal.y += (current.z - next.z) * (current.x + next.x);
        normal.z += (current.x - next.x) * (current.y + next.y);
    }
    normal.normalize_or_zero()
}

fn boxed(size: Vec3) -> Solid {
    let half = Vec3::new(extent(size.x), extent(size.y), extent(size.z)) / 2.0;
    // Corner `i` takes max on x when bit 0 is set, y for bit 1, z for bit 2.
    let points = (0..8)
        .map(|i| {
            Vec3::new(
                if i & 1 == 0 { -half.x } else { half.x },
                if i & 2 == 0 { -half.y } else { half.y },
                if i & 4 == 0 { -half.z } else { half.z },
            )
        })
        .collect();
    let faces = vec![
        vec![0, 1, 3, 2], // -Z
        vec![4, 5, 7, 6], // +Z
        vec![0, 1, 5, 4], // -Y
        vec![2, 3, 7, 6], // +Y
        vec![0, 2, 6, 4], // -X
        vec![1, 3, 7, 5], // +X
    ];
    Solid { points, faces }
}

fn wedge(size: Vec3) -> Solid {
    let half = Vec3::new(extent(size.x), extent(size.y), extent(size.z)) / 2.0;
    let points = vec![
        Vec3::new(-half.x, -half.y, -half.z), // 0 front bottom left
        Vec3::new(half.x, -half.y, -half.z),  // 1 front bottom right
        Vec3::new(-half.x, -half.y, half.z),  // 2 back bottom left
        Vec3::new(half.x, -half.y, half.z),   // 3 back bottom right
        Vec3::new(-half.x, half.y, half.z),   // 4 back top left
        Vec3::new(half.x, half.y, half.z),    // 5 back top right
    ];
    let faces = vec![
        vec![0, 1, 3, 2], // bottom
        vec![2, 3, 5, 4], // back
        vec![0, 1, 5, 4], // slope
        vec![0, 2, 4],    // left
        vec![1, 3, 5],    // right
    ];
    Solid { points, faces }
}

fn frustum(bottom_radius: f32, top_radius: f32, height: f32, sides: u32) -> Solid {
    let sides = sides.clamp(MIN_SIDES, MAX_SIDES) as usize;
    let bottom = extent(bottom_radius);
    // A top radius may be zero — that is a cone — but not negative or NaN.
    let top = if top_radius.is_finite() {
        top_radius.max(0.0)
    } else {
        0.0
    };
    let half = extent(height) / 2.0;

    let ring = |radius: f32, y: f32| {
        (0..sides).map(move |side| {
            let angle = side as f32 / sides as f32 * TAU;
            Vec3::new(radius * angle.sin(), y, radius * angle.cos())
        })
    };

    let mut points: Vec<Vec3> = ring(bottom, -half).collect();
    let mut faces = vec![(0..sides).collect::<Vec<_>>()];

    if top < MIN_EXTENT {
        // A cone: one apex, and triangles rather than quads up the side.
        let apex = points.len();
        points.push(Vec3::new(0.0, half, 0.0));
        for side in 0..sides {
            faces.push(vec![side, (side + 1) % sides, apex]);
        }
    } else {
        let first_top = points.len();
        points.extend(ring(top, half));
        faces.push((first_top..first_top + sides).collect());
        for side in 0..sides {
            let next = (side + 1) % sides;
            faces.push(vec![side, next, first_top + next, first_top + side]);
        }
    }
    Solid { points, faces }
}

/// A dimension that may be zero — a taper's top — but never negative or NaN.
/// Anything below [`MIN_EXTENT`] snaps to zero, so "a point" has one meaning.
fn optional_extent(metres: f32) -> f32 {
    if metres.is_finite() && metres >= MIN_EXTENT {
        metres
    } else {
        0.0
    }
}

fn taper(bottom: crate::math::Vec2, top: crate::math::Vec2, height: f32) -> Solid {
    let low = (extent(bottom.x) / 2.0, extent(bottom.y) / 2.0);
    let high = (optional_extent(top.x) / 2.0, optional_extent(top.y) / 2.0);
    let half = extent(height) / 2.0;
    // Corners in the same order on both rings: -x-z, +x-z, +x+z, -x+z.
    let ring = |(x, z): (f32, f32), y: f32| {
        [(-x, -z), (x, -z), (x, z), (-x, z)].map(|(cx, cz)| Vec3::new(cx, y, cz))
    };
    let mut points = ring(low, -half).to_vec();
    points.extend(ring(high, half));
    let faces = vec![
        vec![0, 1, 2, 3],
        vec![4, 5, 6, 7],
        vec![0, 1, 5, 4],
        vec![1, 2, 6, 5],
        vec![2, 3, 7, 6],
        vec![3, 0, 4, 7],
    ];
    Solid { points, faces }
}

fn dome(radius: f32, segments: u32, rings: u32) -> Solid {
    let radius = extent(radius);
    let segments = segments.clamp(MIN_SIDES, MAX_SIDES) as usize;
    let rings = rings.clamp(1, MAX_DOME_RINGS) as usize;
    // Centred vertically like every other shape: the crown at +r/2, the rim
    // at -r/2.
    let half = radius / 2.0;

    let mut points = vec![Vec3::new(0.0, radius - half, 0.0)];
    for ring in 1..=rings {
        let elevation = std::f32::consts::FRAC_PI_2 * (1.0 - ring as f32 / rings as f32);
        let (rise, reach) = elevation.sin_cos();
        for segment in 0..segments {
            let angle = segment as f32 / segments as f32 * TAU;
            points.push(Vec3::new(
                radius * reach * angle.sin(),
                radius * rise - half,
                radius * reach * angle.cos(),
            ));
        }
    }
    let at = |ring: usize, segment: usize| 1 + (ring - 1) * segments + segment % segments;

    let mut faces = Vec::new();
    for segment in 0..segments {
        faces.push(vec![0, at(1, segment), at(1, segment + 1)]);
    }
    for ring in 1..rings {
        for segment in 0..segments {
            faces.push(vec![
                at(ring, segment),
                at(ring, segment + 1),
                at(ring + 1, segment + 1),
                at(ring + 1, segment),
            ]);
        }
    }
    faces.push((0..segments).map(|segment| at(rings, segment)).collect());
    Solid { points, faces }
}

fn sphere(radius: f32, detail: u32) -> Solid {
    let radius = extent(radius);
    let golden = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let mut points: Vec<Vec3> = [
        (-1.0, golden, 0.0),
        (1.0, golden, 0.0),
        (-1.0, -golden, 0.0),
        (1.0, -golden, 0.0),
        (0.0, -1.0, golden),
        (0.0, 1.0, golden),
        (0.0, -1.0, -golden),
        (0.0, 1.0, -golden),
        (golden, 0.0, -1.0),
        (golden, 0.0, 1.0),
        (-golden, 0.0, -1.0),
        (-golden, 0.0, 1.0),
    ]
    .into_iter()
    .map(|(x, y, z)| Vec3::new(x, y, z).normalize() * radius)
    .collect();

    let mut faces: Vec<[usize; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    for _ in 0..detail.min(MAX_SPHERE_DETAIL) {
        let mut finer = Vec::with_capacity(faces.len() * 4);
        for [a, b, c] in faces {
            let mut midpoint = |from: usize, to: usize| {
                let point = ((points[from] + points[to]) / 2.0).normalize() * radius;
                points.push(point);
                points.len() - 1
            };
            let ab = midpoint(a, b);
            let bc = midpoint(b, c);
            let ca = midpoint(c, a);
            finer.extend_from_slice(&[[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        faces = finer;
    }

    Solid {
        points,
        faces: faces.into_iter().map(|face| face.to_vec()).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RED: [f32; 3] = [0.8, 0.1, 0.1];
    const GREEN: [f32; 3] = [0.1, 0.8, 0.1];
    const BLUE: [f32; 3] = [0.1, 0.1, 0.8];

    fn banded() -> FaceColors {
        FaceColors {
            top: RED,
            side: GREEN,
            bottom: BLUE,
        }
    }

    fn every_shape() -> Vec<Shape> {
        vec![
            Shape::Box {
                size: Vec3::new(1.0, 2.0, 0.5),
            },
            Shape::Wedge {
                size: Vec3::new(1.0, 0.6, 1.4),
            },
            Shape::Frustum {
                bottom_radius: 0.5,
                top_radius: 0.5,
                height: 1.0,
                sides: 8,
            },
            Shape::Frustum {
                bottom_radius: 0.4,
                top_radius: 0.2,
                height: 1.2,
                sides: 5,
            },
            Shape::Frustum {
                bottom_radius: 0.5,
                top_radius: 0.0,
                height: 1.0,
                sides: 6,
            },
            Shape::Frustum {
                bottom_radius: 0.3,
                top_radius: 0.3,
                height: 0.8,
                sides: 3,
            },
            Shape::Sphere {
                radius: 0.5,
                detail: 0,
            },
            Shape::Sphere {
                radius: 0.5,
                detail: 1,
            },
            Shape::Taper {
                bottom: crate::math::Vec2::new(0.4, 0.3),
                top: crate::math::Vec2::new(0.6, 0.35),
                height: 0.6,
            },
            Shape::Taper {
                bottom: crate::math::Vec2::new(0.3, 0.3),
                top: crate::math::Vec2::ZERO,
                height: 0.5,
            },
            Shape::Taper {
                bottom: crate::math::Vec2::new(0.4, 0.2),
                top: crate::math::Vec2::new(0.0, 0.2),
                height: 0.3,
            },
            Shape::Dome {
                radius: 0.3,
                segments: 7,
                rings: 3,
            },
            Shape::Dome {
                radius: 0.3,
                segments: 6,
                rings: 1,
            },
        ]
    }

    fn corners(mesh: &MeshData, tri: &[u32]) -> [Vec3; 3] {
        [0, 1, 2].map(|k| Vec3::from(mesh.positions[tri[k] as usize]))
    }

    fn centroid(mesh: &MeshData) -> Vec3 {
        mesh.positions.iter().map(|&p| Vec3::from(p)).sum::<Vec3>() / mesh.positions.len() as f32
    }

    #[test]
    fn every_shape_validates_and_is_fully_painted() {
        for shape in every_shape() {
            let mesh = mesh(shape, banded());
            assert_eq!(mesh.validate(), Ok(()), "{shape:?}");
            assert_eq!(
                mesh.colors.len(),
                mesh.positions.len(),
                "{shape:?}: every vertex carries a colour"
            );
        }
    }

    /// Orientation is computed; this is what proves the computation. Every
    /// triangle must wind counter-clockwise seen from outside, or it vanishes
    /// under back-face culling.
    #[test]
    fn every_triangle_faces_outward() {
        for shape in every_shape() {
            let mesh = mesh(shape, banded());
            let inside = centroid(&mesh);
            for tri in mesh.indices.chunks(3) {
                let [a, b, c] = corners(&mesh, tri);
                let winding = (b - a).cross(c - a);
                let outward = (a + b + c) / 3.0 - inside;
                assert!(
                    winding.dot(outward) > 0.0,
                    "{shape:?}: a triangle winds inward and would be culled"
                );
            }
        }
    }

    #[test]
    fn every_face_is_flat_and_its_normal_is_true() {
        for shape in every_shape() {
            let mesh = mesh(shape, banded());
            for tri in mesh.indices.chunks(3) {
                let normals: Vec<[f32; 3]> = tri.iter().map(|&i| mesh.normals[i as usize]).collect();
                assert!(
                    normals.iter().all(|n| n == &normals[0]),
                    "{shape:?}: a facet shares one normal"
                );
                let normal = Vec3::from(normals[0]);
                assert!((normal.length() - 1.0).abs() < 1e-4, "{shape:?}: unit normal");

                let [a, b, c] = corners(&mesh, tri);
                let geometric = (b - a).cross(c - a).normalize();
                assert!(
                    geometric.dot(normal) > 0.999,
                    "{shape:?}: the stored normal is the face's real normal"
                );
            }
        }
    }

    #[test]
    fn faces_are_painted_by_which_way_they_face() {
        let mesh = mesh(Shape::Box { size: Vec3::ONE }, banded());
        for (normal, color) in mesh.normals.iter().zip(&mesh.colors) {
            let expected = match normal[1] {
                y if y > 0.5 => RED,
                y if y < -0.5 => BLUE,
                _ => GREEN,
            };
            assert_eq!(color, &expected, "normal {normal:?}");
        }
        assert!(mesh.colors.contains(&RED) && mesh.colors.contains(&BLUE));
    }

    #[test]
    fn triangle_counts_are_what_a_budget_can_count_on() {
        let count = |shape| mesh(shape, FaceColors::uniform(RED)).triangle_count();
        assert_eq!(count(Shape::Box { size: Vec3::ONE }), 12);
        assert_eq!(count(Shape::Wedge { size: Vec3::ONE }), 8);
        let cylinder = |sides| Shape::Frustum {
            bottom_radius: 0.5,
            top_radius: 0.5,
            height: 1.0,
            sides,
        };
        // Two n-gon caps of n-2 triangles each, and n quads up the side.
        assert_eq!(count(cylinder(8)), 2 * 6 + 16);
        let cone = Shape::Frustum {
            bottom_radius: 0.5,
            top_radius: 0.0,
            height: 1.0,
            sides: 8,
        };
        assert_eq!(count(cone), 6 + 8);
        assert_eq!(
            count(Shape::Sphere {
                radius: 1.0,
                detail: 0
            }),
            20
        );
        assert_eq!(
            count(Shape::Sphere {
                radius: 1.0,
                detail: 1
            }),
            80
        );

        let taper = |top_x, top_z| Shape::Taper {
            bottom: crate::math::Vec2::new(0.4, 0.4),
            top: crate::math::Vec2::new(top_x, top_z),
            height: 0.5,
        };
        assert_eq!(count(taper(0.3, 0.3)), 12, "a tapered box has a box's faces");
        assert_eq!(
            count(taper(0.0, 0.3)),
            8,
            "a ridge: two sides become triangles, no top"
        );
        assert_eq!(
            count(taper(0.0, 0.0)),
            6,
            "a pyramid: a square and four triangles"
        );

        let dome = |segments, rings| Shape::Dome {
            radius: 0.5,
            segments,
            rings,
        };
        // A fan to the crown, quads between rings, and the rim's n-gon.
        assert_eq!(count(dome(8, 3)), 8 + 2 * 8 * 2 + 6);
        assert_eq!(count(dome(6, 1)), 6 + 4, "one ring is a cone");
    }

    /// Recipes are data, and data can be wrong. No input may panic, produce a
    /// non-finite vertex, or escape the triangle ceiling.
    #[test]
    fn hostile_parameters_clamp_rather_than_break() {
        let hostile = [
            Shape::Box {
                size: Vec3::new(f32::NAN, -3.0, 0.0),
            },
            Shape::Wedge {
                size: Vec3::new(f32::INFINITY, 0.0, -1.0),
            },
            Shape::Frustum {
                bottom_radius: -1.0,
                top_radius: f32::NAN,
                height: 0.0,
                sides: 0,
            },
            Shape::Frustum {
                bottom_radius: 1.0,
                top_radius: 1.0,
                height: 1.0,
                sides: u32::MAX,
            },
            Shape::Sphere {
                radius: f32::NEG_INFINITY,
                detail: u32::MAX,
            },
            Shape::Taper {
                bottom: crate::math::Vec2::new(f32::NAN, -1.0),
                top: crate::math::Vec2::new(f32::INFINITY, -5.0),
                height: 0.0,
            },
            Shape::Dome {
                radius: f32::NAN,
                segments: u32::MAX,
                rings: u32::MAX,
            },
        ];
        for shape in hostile {
            let mesh = mesh(shape, FaceColors::uniform(RED));
            assert_eq!(mesh.validate(), Ok(()), "{shape:?}");
            assert!(
                mesh.positions.iter().flatten().all(|v| v.is_finite()),
                "{shape:?}: finite geometry"
            );
            assert!(
                // The largest the clamps allow is a dome of MAX_SIDES segments
                // and MAX_DOME_RINGS rings: its crown fan, the quads between
                // rings, and its rim's n-gon.
                mesh.triangle_count()
                    <= (MAX_SIDES + 2 * MAX_SIDES * (MAX_DOME_RINGS - 1) + (MAX_SIDES - 2)) as usize,
                "{shape:?}: bounded"
            );
        }
    }

    #[test]
    fn generation_is_deterministic() {
        for shape in every_shape() {
            assert_eq!(mesh(shape, banded()), mesh(shape, banded()), "{shape:?}");
        }
    }

    #[test]
    fn placing_moves_positions_and_turns_normals() {
        use crate::math::Quat;

        let boxed = mesh(Shape::Box { size: Vec3::ONE }, banded());
        let transform = Transform::from_xyz(2.0, 0.0, 0.0)
            .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2));
        let moved = placed(boxed.clone(), &transform);

        let bounds = moved.bounds().expect("non-empty");
        assert!((bounds.min[0] - 1.5).abs() < 1e-5 && (bounds.max[0] - 2.5).abs() < 1e-5);

        // The top face (+Y) now faces -X, and is still painted as the top was:
        // colour is baked by the face's *authored* orientation, so turning a
        // part does not repaint it.
        let top = boxed.normals.iter().position(|n| n[1] > 0.9).expect("a top face");
        assert!(Vec3::from(moved.normals[top]).abs_diff_eq(Vec3::NEG_X, 1e-5));
        assert_eq!(moved.colors[top], RED);

        // Placing must not break the outward winding the shape was built with.
        let inside = centroid(&moved);
        for tri in moved.indices.chunks(3) {
            let [a, b, c] = corners(&moved, tri);
            assert!((b - a).cross(c - a).dot((a + b + c) / 3.0 - inside) > 0.0);
        }
    }

    #[test]
    fn a_degenerate_placement_keeps_normals_finite() {
        let flat = placed(
            mesh(Shape::Box { size: Vec3::ONE }, banded()),
            &Transform::IDENTITY.with_scale(Vec3::new(1.0, 0.0, 1.0)),
        );
        assert!(flat.normals.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn merging_offsets_indices_and_keeps_every_colour() {
        let red = mesh(Shape::Box { size: Vec3::ONE }, FaceColors::uniform(RED));
        let blue = placed(
            mesh(
                Shape::Sphere {
                    radius: 0.3,
                    detail: 0,
                },
                FaceColors::uniform(BLUE),
            ),
            &Transform::from_xyz(0.0, 1.0, 0.0),
        );
        let mut unpainted = mesh(Shape::Box { size: Vec3::ONE }, banded());
        unpainted.colors.clear();

        let joined = merged([red.clone(), blue.clone(), unpainted.clone()]);
        assert_eq!(joined.validate(), Ok(()));
        assert_eq!(
            joined.triangle_count(),
            red.triangle_count() + blue.triangle_count() + unpainted.triangle_count()
        );
        assert!(joined.colors[..red.positions.len()].iter().all(|c| c == &RED));
        let blue_end = red.positions.len() + blue.positions.len();
        assert!(
            joined.colors[red.positions.len()..blue_end]
                .iter()
                .all(|c| c == &BLUE)
        );
        assert!(
            joined.colors[blue_end..].iter().all(|c| c == &renderer::WHITE),
            "an unpainted mesh merges in white rather than borrowing a neighbour's colours"
        );
    }

    /// A part's offset in a recipe is measured from its shape's centre, so the
    /// centre has to be where the docs say it is.
    #[test]
    fn shapes_are_centred_where_the_docs_say() {
        for shape in every_shape() {
            let mesh = mesh(shape, banded());
            let bounds = mesh.bounds().expect("non-empty");
            let vertical = (bounds.min[1] + bounds.max[1]) / 2.0;
            assert!(vertical.abs() < 1e-4, "{shape:?}: vertically centred");

            // Boxes, wedges and spheres are centred in their bounds. A frustum
            // is centred on its axis: every ring corner appears equally often,
            // so the mean of its vertices sits on the axis even when an odd
            // number of sides leaves its bounding box lopsided.
            let (x, z) = match shape {
                Shape::Frustum { .. } | Shape::Dome { .. } => {
                    let mean = centroid(&mesh);
                    (mean.x, mean.z)
                }
                _ => (
                    (bounds.min[0] + bounds.max[0]) / 2.0,
                    (bounds.min[2] + bounds.max[2]) / 2.0,
                ),
            };
            assert!(x.abs() < 1e-4 && z.abs() < 1e-4, "{shape:?}: centred ({x}, {z})");
        }
    }
}
