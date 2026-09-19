//! Uploaded meshes and the conversion from `MeshData` to buffer bytes.
//!
//! The byte conversions are separate from the upload itself so they can be
//! tested without a device: an interleave bug produces geometry that is wrong
//! rather than absent, which is exactly the kind of thing a headless test
//! catches cheaply and a GPU test catches expensively.

use crate::mesh_data::Aabb;
use crate::registry::Geometry;

/// A mesh living in GPU memory, plus the counts the frame stats and culling
/// need.
#[derive(Debug)]
pub(crate) struct GpuMesh {
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) index_buffer: wgpu::Buffer,
    pub(crate) index_count: u32,
    pub(crate) triangle_count: u32,
    /// Bounds in the mesh's own space. Kept here because the engine's frustum
    /// culling needs them and the upload is the only place they are known.
    pub(crate) bounds: Aabb,
}

impl Geometry for GpuMesh {
    fn triangle_count(&self) -> u32 {
        self.triangle_count
    }
}

/// What the engine can ask about an uploaded mesh.
///
/// Returned by `Renderer::mesh_info`, which is how extraction gets the bounds
/// it culls against without the renderer having to know what culling is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshInfo {
    pub index_count: u32,
    pub triangle_count: u32,
    pub bounds: Aabb,
}

/// Little-endian bytes for a vertex buffer.
///
/// Written out by hand rather than transmuted: `bytemuck` is not a dependency,
/// and `unsafe` is forbidden workspace-wide, so this is the honest way to get
/// from `&[f32]` to `&[u8]`.
pub(crate) fn float_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Little-endian bytes for a `Uint32` index buffer.
pub(crate) fn index_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_data::MeshData;

    fn triangle() -> MeshData {
        MeshData {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            colors: Vec::new(),
            indices: vec![0, 1, 2],
        }
    }

    #[test]
    fn vertex_bytes_are_forty_four_per_vertex() {
        let mesh = triangle();
        let bytes = float_bytes(&mesh.interleaved());
        assert_eq!(
            bytes.len(),
            mesh.vertex_count() * 44,
            "the vertex layout is 11 floats per vertex"
        );

        // The second vertex starts at byte 44 with position.x == 1.0.
        let mut word = [0_u8; 4];
        word.copy_from_slice(&bytes[44..48]);
        assert_eq!(f32::from_le_bytes(word), 1.0);
    }

    #[test]
    fn index_bytes_round_trip() {
        let bytes = index_bytes(&[0, 1, 70000]);
        assert_eq!(bytes.len(), 12, "Uint32 indices are four bytes each");

        let mut word = [0_u8; 4];
        word.copy_from_slice(&bytes[8..12]);
        assert_eq!(
            u32::from_le_bytes(word),
            70000,
            "indices above 65535 must survive, which is why the format is Uint32"
        );
    }

    #[test]
    fn empty_attributes_produce_empty_buffers() {
        assert!(float_bytes(&[]).is_empty());
        assert!(index_bytes(&[]).is_empty());
    }
}
