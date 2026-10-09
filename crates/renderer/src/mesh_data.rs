//! CPU-side mesh data: the hand-off format between mesh producers
//! (primitives, the asset bake step) and GPU upload.
//!
//! Attributes are stored as plain arrays rather than math-library types so
//! mesh producers need no math dependency, and so the layout matches what is
//! interleaved into a vertex buffer:
//!
//! | Location | Attribute | Format      |
//! |----------|-----------|-------------|
//! | 0        | position  | `Float32x3` |
//! | 1        | normal    | `Float32x3` |
//! | 2        | uv        | `Float32x2` |
//! | 3        | colour    | `Float32x3` |
//! | 4        | tangent   | `Float32x4` |
//! | 5        | wind      | `Float32`   |
//!
//! Tangent and wind were added together in one layout change (owner decision,
//! M17): 64 bytes per vertex. The tangent's `w` is the handedness of the UV
//! mapping. An empty `tangents` list means they are generated at upload from
//! positions, normals and UVs (see `tangents.rs`); the asset cooker will
//! supply MikkTSpace tangents instead (B6). Wind is how strongly a vertex
//! sways, 0 to 1, for foliage (D2); empty means 0, so nothing sways.
//!
//! Colour is linear RGB per vertex (ADR-0017). It is how a low-poly model
//! carries several flat colours in one mesh — a painted face, a trim band —
//! and it is multiplied by the material's tint in the shader. An empty
//! `colors` list means white, so a mesh built without colours renders exactly
//! as it did before colour existed.

use std::fmt;

/// An axis-aligned bounding box in the mesh's own space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    /// Bounds enclosing every point, or `None` if the iterator is empty.
    pub fn from_points(points: impl IntoIterator<Item = [f32; 3]>) -> Option<Self> {
        let mut points = points.into_iter();
        let first = points.next()?;
        let mut bounds = Self {
            min: first,
            max: first,
        };
        for point in points {
            for ((min, max), value) in bounds.min.iter_mut().zip(&mut bounds.max).zip(point) {
                *min = min.min(value);
                *max = max.max(value);
            }
        }
        Some(bounds)
    }

    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) / 2.0,
            (self.min[1] + self.max[1]) / 2.0,
            (self.min[2] + self.max[2]) / 2.0,
        ]
    }

    pub fn half_extents(&self) -> [f32; 3] {
        [
            (self.max[0] - self.min[0]) / 2.0,
            (self.max[1] - self.min[1]) / 2.0,
            (self.max[2] - self.min[2]) / 2.0,
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeshDataError {
    /// Every attribute array must have one entry per vertex.
    MismatchedAttributeLengths {
        positions: usize,
        normals: usize,
        uvs: usize,
        colors: usize,
        tangents: usize,
        wind: usize,
    },
    /// Indices must describe whole triangles.
    IndexCountNotTriangles(usize),
    IndexOutOfRange {
        index: u32,
        vertex_count: usize,
    },
    Empty,
}

impl fmt::Display for MeshDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MismatchedAttributeLengths {
                positions,
                normals,
                uvs,
                colors,
                tangents,
                wind,
            } => write!(
                f,
                "attribute lengths differ: {positions} positions, {normals} normals, {uvs} uvs, \
                 {colors} colours, {tangents} tangents, {wind} wind weights (colours, tangents \
                 and wind may also be empty)"
            ),
            Self::IndexCountNotTriangles(count) => {
                write!(f, "index count {count} is not a multiple of 3")
            }
            Self::IndexOutOfRange { index, vertex_count } => {
                write!(f, "index {index} exceeds vertex count {vertex_count}")
            }
            Self::Empty => f.write_str("mesh has no vertices or no indices"),
        }
    }
}

impl std::error::Error for MeshDataError {}

/// An indexed triangle mesh ready to be interleaved and uploaded.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    /// Linear RGB per vertex, or empty for white. See the module docs.
    pub colors: Vec<[f32; 3]>,
    /// Tangent per vertex, `w` the handedness, or empty to generate them.
    pub tangents: Vec<[f32; 4]>,
    /// Sway strength per vertex, or empty for none.
    pub wind: Vec<f32>,
    pub indices: Vec<u32>,
}

/// Floats per interleaved vertex: position (3), normal (3), uv (2), colour (3),
/// tangent (4), wind (1).
///
/// The one number the pipeline's vertex stride and the shader's layout comment
/// are both derived from, so they cannot drift apart.
pub const VERTEX_FLOATS: usize = 16;

/// The colour of a vertex whose mesh carries none.
pub const WHITE: [f32; 3] = [1.0, 1.0, 1.0];

impl MeshData {
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn bounds(&self) -> Option<Aabb> {
        Aabb::from_points(self.positions.iter().copied())
    }

    /// Checks the invariants GPU upload relies on.
    pub fn validate(&self) -> Result<(), MeshDataError> {
        if self.positions.is_empty() || self.indices.is_empty() {
            return Err(MeshDataError::Empty);
        }
        let count = self.positions.len();
        let optional_ok = |len: usize| len == 0 || len == count;
        if self.normals.len() != count
            || self.uvs.len() != count
            || !optional_ok(self.colors.len())
            || !optional_ok(self.tangents.len())
            || !optional_ok(self.wind.len())
        {
            return Err(MeshDataError::MismatchedAttributeLengths {
                positions: count,
                normals: self.normals.len(),
                uvs: self.uvs.len(),
                colors: self.colors.len(),
                tangents: self.tangents.len(),
                wind: self.wind.len(),
            });
        }
        if !self.indices.len().is_multiple_of(3) {
            return Err(MeshDataError::IndexCountNotTriangles(self.indices.len()));
        }
        let vertex_count = self.positions.len();
        for &index in &self.indices {
            if index as usize >= vertex_count {
                return Err(MeshDataError::IndexOutOfRange { index, vertex_count });
            }
        }
        Ok(())
    }

    /// Interleaves attributes into the vertex layout above: [`VERTEX_FLOATS`]
    /// floats per vertex.
    ///
    /// Tangents are generated here when the mesh carries none, so every mesh
    /// that reaches the GPU has them.
    pub fn interleaved(&self) -> Vec<f32> {
        let generated;
        let tangents = if self.tangents.len() == self.positions.len() {
            &self.tangents
        } else {
            generated = self.generated_tangents();
            &generated
        };
        let mut data = Vec::with_capacity(self.positions.len() * VERTEX_FLOATS);
        for vertex in 0..self.positions.len() {
            let position = self.positions[vertex];
            let normal = self.normals.get(vertex).copied().unwrap_or([0.0, 1.0, 0.0]);
            let uv = self.uvs.get(vertex).copied().unwrap_or([0.0, 0.0]);
            let color = self.colors.get(vertex).copied().unwrap_or(WHITE);
            let tangent = tangents.get(vertex).copied().unwrap_or([1.0, 0.0, 0.0, 1.0]);
            let wind = self.wind.get(vertex).copied().unwrap_or(0.0);
            data.extend_from_slice(&position);
            data.extend_from_slice(&normal);
            data.extend_from_slice(&uv);
            data.extend_from_slice(&color);
            data.extend_from_slice(&tangent);
            data.push(wind);
        }
        data
    }

    fn generated_tangents(&self) -> Vec<[f32; 4]> {
        // Missing normals or UVs (an invalid mesh) still produce one tangent
        // per vertex, so interleaving never reads past an array.
        let count = self.positions.len();
        let normals: Vec<[f32; 3]> = (0..count)
            .map(|vertex| self.normals.get(vertex).copied().unwrap_or([0.0, 1.0, 0.0]))
            .collect();
        let uvs: Vec<[f32; 2]> = (0..count)
            .map(|vertex| self.uvs.get(vertex).copied().unwrap_or([0.0, 0.0]))
            .collect();
        crate::tangents::generate(&self.positions, &normals, &uvs, &self.indices)
    }

    /// Paints every vertex one colour.
    pub fn with_color(mut self, color: [f32; 3]) -> Self {
        self.colors = vec![color; self.positions.len()];
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle() -> MeshData {
        MeshData {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2],
            ..MeshData::default()
        }
    }

    #[test]
    fn validates_a_well_formed_mesh() {
        let mesh = triangle();
        assert_eq!(mesh.validate(), Ok(()));
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.triangle_count(), 1);
    }

    #[test]
    fn rejects_broken_meshes() {
        assert_eq!(MeshData::default().validate(), Err(MeshDataError::Empty));

        let mut mesh = triangle();
        mesh.normals.pop();
        assert!(matches!(
            mesh.validate(),
            Err(MeshDataError::MismatchedAttributeLengths { .. })
        ));

        let mut mesh = triangle();
        mesh.indices.push(0);
        assert_eq!(mesh.validate(), Err(MeshDataError::IndexCountNotTriangles(4)));

        let mut mesh = triangle();
        mesh.indices = vec![0, 1, 7];
        assert_eq!(
            mesh.validate(),
            Err(MeshDataError::IndexOutOfRange {
                index: 7,
                vertex_count: 3
            })
        );
    }

    #[test]
    fn computes_bounds() {
        assert_eq!(MeshData::default().bounds(), None);
        let bounds = triangle().bounds().unwrap();
        assert_eq!(bounds.min, [0.0, 0.0, 0.0]);
        assert_eq!(bounds.max, [1.0, 2.0, 0.0]);
        assert_eq!(bounds.center(), [0.5, 1.0, 0.0]);
        assert_eq!(bounds.half_extents(), [0.5, 1.0, 0.0]);
    }

    #[test]
    fn interleaves_sixteen_floats_per_vertex() {
        let data = triangle().interleaved();
        assert_eq!(data.len(), 3 * VERTEX_FLOATS);
        // u runs along +X and v along +Y on a +Z face. With top-down UVs that
        // puts the top of the image at -Y, a mirrored mapping: tangent +X, w -1.
        assert_eq!(
            &data[0..16],
            &[
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, -1.0, 0.0
            ]
        );
        assert_eq!(
            &data[16..27],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn supplied_tangents_and_wind_are_used_as_given() {
        let mut mesh = triangle();
        mesh.tangents = vec![[0.0, 1.0, 0.0, -1.0]; 3];
        mesh.wind = vec![0.25, 0.5, 1.0];
        assert_eq!(mesh.validate(), Ok(()));
        let data = mesh.interleaved();
        assert_eq!(&data[11..16], &[0.0, 1.0, 0.0, -1.0, 0.25]);
        assert_eq!(data[2 * VERTEX_FLOATS + 15], 1.0, "the last vertex's wind");

        mesh.wind.pop();
        assert!(matches!(
            mesh.validate(),
            Err(MeshDataError::MismatchedAttributeLengths { wind: 2, .. })
        ));
        mesh.wind.clear();
        mesh.tangents.pop();
        assert!(matches!(
            mesh.validate(),
            Err(MeshDataError::MismatchedAttributeLengths { tangents: 2, .. })
        ));
    }

    /// Wave 1 must change nothing on screen: a mesh with no colours is white,
    /// and white times a material's tint is the material, as before.
    #[test]
    fn a_mesh_without_colours_is_white() {
        let data = triangle().interleaved();
        for vertex in data.chunks(VERTEX_FLOATS) {
            assert_eq!(&vertex[8..11], &WHITE);
        }
        assert_eq!(triangle().validate(), Ok(()), "and no colours is valid");
    }

    #[test]
    fn painted_colours_reach_the_vertex_and_must_cover_every_vertex() {
        let painted = triangle().with_color([0.8, 0.2, 0.1]);
        assert_eq!(painted.validate(), Ok(()));
        assert_eq!(&painted.interleaved()[8..11], &[0.8, 0.2, 0.1]);

        let mut partial = triangle();
        partial.colors = vec![[1.0, 0.0, 0.0]];
        assert!(
            matches!(
                partial.validate(),
                Err(MeshDataError::MismatchedAttributeLengths { colors: 1, .. })
            ),
            "some colours but not all is a mistake, not a default"
        );
    }
}
