//! Tangent generation for meshes that arrive without tangents.
//!
//! The interim stand-in for MikkTSpace, which the asset cooker (B6) will run
//! offline. Per triangle, the tangent and bitangent are solved from the edge
//! vectors and the UV deltas (Lengyel's method) and summed per vertex, which
//! weights each triangle by its area. Each vertex's tangent is then made
//! perpendicular to its normal, and `w` records the handedness, the sign the
//! shader multiplies `cross(normal, tangent)` by to get the bitangent.
//!
//! A vertex whose triangles have no usable UV mapping (all UVs equal, which
//! is every engine primitive without a texture) gets an arbitrary tangent
//! perpendicular to its normal and `w = 1`, so normal mapping is a no-op on it
//! rather than NaN.

type Vec3 = [f32; 3];

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: Vec3, s: f32) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `v` scaled to unit length, or `None` if it is too short to have a
/// direction.
fn normalized(v: Vec3) -> Option<Vec3> {
    let length = dot(v, v).sqrt();
    (length > 1e-12 && length.is_finite()).then(|| scale(v, 1.0 / length))
}

/// Some unit vector perpendicular to `normal`.
fn any_perpendicular(normal: Vec3) -> Vec3 {
    let axis = if normal[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    normalized(sub(axis, scale(normal, dot(normal, axis)))).unwrap_or([1.0, 0.0, 0.0])
}

/// One tangent per vertex. `indices` must already be validated as in range.
pub(crate) fn generate(
    positions: &[Vec3],
    normals: &[Vec3],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> Vec<[f32; 4]> {
    let count = positions.len();
    let mut tangents = vec![[0.0; 3]; count];
    let mut bitangents = vec![[0.0; 3]; count];

    for triangle in indices.chunks_exact(3) {
        let [i0, i1, i2] = [triangle[0], triangle[1], triangle[2]].map(|index| index as usize);
        if i0.max(i1).max(i2) >= count {
            continue;
        }
        let edge1 = sub(positions[i1], positions[i0]);
        let edge2 = sub(positions[i2], positions[i0]);
        let du1 = uvs[i1][0] - uvs[i0][0];
        let dv1 = uvs[i1][1] - uvs[i0][1];
        let du2 = uvs[i2][0] - uvs[i0][0];
        let dv2 = uvs[i2][1] - uvs[i0][1];
        let area = du1 * dv2 - du2 * dv1;
        if area.abs() < 1e-12 {
            continue;
        }
        let inverse = 1.0 / area;
        let tangent = scale(sub(scale(edge1, dv2), scale(edge2, dv1)), inverse);
        let bitangent = scale(sub(scale(edge2, du1), scale(edge1, du2)), inverse);
        for index in [i0, i1, i2] {
            tangents[index] = add(tangents[index], tangent);
            bitangents[index] = add(bitangents[index], bitangent);
        }
    }

    (0..count)
        .map(|vertex| {
            let normal = normals
                .get(vertex)
                .copied()
                .and_then(normalized)
                .unwrap_or([0.0, 1.0, 0.0]);
            let along_surface = sub(tangents[vertex], scale(normal, dot(normal, tangents[vertex])));
            match normalized(along_surface) {
                Some(tangent) => {
                    let handedness = if dot(cross(normal, tangent), bitangents[vertex]) < 0.0 {
                        -1.0
                    } else {
                        1.0
                    };
                    [tangent[0], tangent[1], tangent[2], handedness]
                }
                None => {
                    let tangent = any_perpendicular(normal);
                    [tangent[0], tangent[1], tangent[2], 1.0]
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 4], b: [f32; 4]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    /// A quad facing +Z, with u running along +X and v along -Y (top-down UVs).
    fn quad(uvs: Vec<[f32; 2]>) -> Vec<[f32; 4]> {
        generate(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            &[[0.0, 0.0, 1.0]; 4],
            &uvs,
            &[0, 1, 2, 0, 2, 3],
        )
    }

    #[test]
    fn the_tangent_follows_increasing_u() {
        let tangents = quad(vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]);
        for tangent in tangents {
            assert!(close(tangent, [1.0, 0.0, 0.0, -1.0]), "{tangent:?}");
        }
    }

    #[test]
    fn mirrored_uvs_flip_the_handedness_not_the_normal() {
        // Same quad with u mirrored: the tangent turns round and so does w.
        let tangents = quad(vec![[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]);
        for tangent in tangents {
            assert!(close(tangent, [-1.0, 0.0, 0.0, 1.0]), "{tangent:?}");
        }
    }

    #[test]
    fn every_tangent_is_unit_length_and_perpendicular_to_its_normal() {
        // A sloped triangle whose UVs are skewed against its edges.
        let normal = normalized([0.0, 1.0, 1.0]).expect("non-zero");
        let tangents = generate(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, -1.0]],
            &[normal; 3],
            &[[0.0, 0.0], [0.7, 0.2], [0.1, 0.9]],
            &[0, 1, 2],
        );
        for tangent in tangents {
            let direction = [tangent[0], tangent[1], tangent[2]];
            assert!((dot(direction, direction) - 1.0).abs() < 1e-5);
            assert!(dot(direction, normal).abs() < 1e-5);
            assert!(tangent[3] == 1.0 || tangent[3] == -1.0);
        }
    }

    #[test]
    fn a_mesh_without_a_uv_mapping_still_gets_valid_tangents() {
        let tangents = quad(vec![[0.0, 0.0]; 4]);
        for tangent in tangents {
            assert!(tangent.iter().all(|value| value.is_finite()));
            let direction = [tangent[0], tangent[1], tangent[2]];
            assert!((dot(direction, direction) - 1.0).abs() < 1e-5);
            assert!(dot(direction, [0.0, 0.0, 1.0]).abs() < 1e-5);
        }
    }

    #[test]
    fn a_normal_along_x_still_gets_a_perpendicular() {
        let tangent = any_perpendicular([1.0, 0.0, 0.0]);
        assert!(dot(tangent, [1.0, 0.0, 0.0]).abs() < 1e-6);
        assert!((dot(tangent, tangent) - 1.0).abs() < 1e-6);
    }
}
