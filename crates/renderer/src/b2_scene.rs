//! Fixed scene, light set and statistics for the B2 lighting-path benchmark (ADR-0019). Deterministic, so every run and both paths see the same work.

use crate::mesh_data::MeshData;

/// Overdraw depth for the benchmark scene.
///
/// Stacking this many layers of quads exercises the depth buffer and
/// overdraw handling without making the scene so large that it stops
/// fitting in the GPU's working set.
pub(crate) const LAYERS: u32 = 8;

/// Grid resolution per layer.
///
/// A 16x16 grid of quads per layer keeps the vertex count manageable while
/// still producing enough geometry to stress the lighting path.
pub(crate) const QUADS_PER_SIDE: u32 = 16;

/// Grid resolution for the dense variant: about 1.6 million triangles, some
/// 22 pixels each at 1440p (and a 210 MB vertex buffer, under the
/// 256 MB buffer limit). Small triangles waste fragment work in 2x2 quads,
/// which costs a forward path more than a deferred one, so the decision is
/// measured with both large and small triangles.
pub(crate) const DENSE_QUADS_PER_SIDE: u32 = 320;

/// Number of lights in the default benchmark light set.
///
/// 256 lights is enough to make the per-light loop the dominant cost in the
/// lighting pass, which is what the benchmark is trying to measure.
pub(crate) const LIGHT_COUNT: u32 = 256;

/// A single light in the benchmark scene.
///
/// The position and radius are packed into one vector to match the layout
/// expected by the GPU shader, and the colour and intensity are packed
/// similarly. Keeping them as plain arrays makes serialisation to bytes
/// trivial.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BenchLight {
    pub(crate) position_radius: [f32; 4],
    pub(crate) color_intensity: [f32; 4],
}

/// A small, fast, deterministic PRNG for generating benchmark data.
///
/// The benchmark needs reproducible random values but does not need
/// cryptographic quality, so a simple xorshift is sufficient. Using a
/// fixed seed guarantees that every run produces identical geometry and
/// lights.
struct XorShift32(u32);

impl XorShift32 {
    /// Advances the state and returns the next 32-bit value.
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Returns a uniform float in [0, 1).
    ///
    /// The top 24 bits of the xorshift output are used, which gives
    /// sufficient precision for benchmark data without the cost of
    /// converting a full 32-bit integer to a float.
    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// Builds the fixed benchmark scene mesh.
///
/// The scene consists of LAYERS stacked grids of quads, each layer
/// tiling the [-1, 1] square in x and y. Layers are emitted from farthest
/// (z = 0.9) to nearest (z = 0.1) so that the draw order matches the
/// depth order. Each quad gets a random colour drawn from a single shared
/// RNG seeded with a fixed value, ensuring the scene is identical on
/// every run.
pub(crate) fn scene_mesh(quads_per_side: u32) -> MeshData {
    let quads_per_layer = quads_per_side * quads_per_side;
    let total_quads = (LAYERS * quads_per_layer) as usize;
    let total_vertices = total_quads * 4;
    let total_indices = total_quads * 6;

    let mut positions = Vec::with_capacity(total_vertices);
    let mut normals = Vec::with_capacity(total_vertices);
    let mut uvs = Vec::with_capacity(total_vertices);
    let mut tangents = Vec::with_capacity(total_vertices);
    let mut wind = Vec::with_capacity(total_vertices);
    let mut colors = Vec::with_capacity(total_vertices);
    let mut indices = Vec::with_capacity(total_indices);

    let mut rng = XorShift32(0x2545_F491);
    let step = 2.0 / quads_per_side as f32;

    for layer in 0..LAYERS {
        let z = 0.9 - 0.8 * layer as f32 / (LAYERS - 1) as f32;

        for row in 0..quads_per_side {
            for col in 0..quads_per_side {
                let x0 = -1.0 + col as f32 * step;
                let x1 = x0 + step;
                let y0 = -1.0 + row as f32 * step;
                let y1 = y0 + step;

                let color = [
                    0.2 + 0.8 * rng.unit(),
                    0.2 + 0.8 * rng.unit(),
                    0.2 + 0.8 * rng.unit(),
                ];

                let base = positions.len() as u32;
                let corners = [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]];

                for corner in corners {
                    positions.push(corner);
                    normals.push([0.0, 0.0, -1.0]);
                    uvs.push([0.0, 0.0]);
                    tangents.push([1.0, 0.0, 0.0, 1.0]);
                    wind.push(0.0);
                    colors.push(color);
                }

                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        }
    }

    MeshData {
        positions,
        normals,
        uvs,
        colors,
        tangents,
        wind,
        indices,
    }
}

/// Generates a deterministic set of benchmark lights.
///
/// Each light's position, radius, colour and intensity are drawn from a
/// fixed-seed RNG so that the light set is identical on every run. The
/// draw order within each light is fixed to match the spec exactly.
pub(crate) fn lights(count: u32) -> Vec<BenchLight> {
    let mut rng = XorShift32(0x9E37_79B9);
    let mut result = Vec::with_capacity(count as usize);

    for _ in 0..count {
        let x = rng.unit() * 2.0 - 1.0;
        let y = rng.unit() * 2.0 - 1.0;
        let z = 0.05 + 0.9 * rng.unit();
        let radius = 0.1 + 0.25 * rng.unit();

        let r = 0.2 + 0.8 * rng.unit();
        let g = 0.2 + 0.8 * rng.unit();
        let b = 0.2 + 0.8 * rng.unit();
        let intensity = 1.0 + 3.0 * rng.unit();

        result.push(BenchLight {
            position_radius: [x, y, z, radius],
            color_intensity: [r, g, b, intensity],
        });
    }

    result
}

/// Serialises a slice of lights into a flat byte buffer.
///
/// Each light occupies 32 bytes: the four floats of position_radius
/// followed by the four floats of color_intensity, each written in
/// little-endian. This layout matches what the GPU expects.
pub(crate) fn light_bytes(lights: &[BenchLight]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(lights.len() * 32);

    for light in lights {
        for &f in &light.position_radius {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        for &f in &light.color_intensity {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
    }

    bytes
}

/// Packs benchmark parameters into a 32-byte little-endian buffer.
///
/// The layout is: extent[0], extent[1], light_count, 0, grid[0], grid[1],
/// grid[2], 0. The two zero slots are reserved for future use.
pub(crate) fn params_bytes(extent: [u32; 2], light_count: u32, grid: [u32; 3]) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    let values = [extent[0], extent[1], light_count, 0, grid[0], grid[1], grid[2], 0];

    for (i, &v) in values.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }

    bytes
}

/// Summary statistics for a set of timing samples.
///
/// Contains the count of valid samples, the arithmetic mean, the median,
/// and the 95th percentile using the nearest-rank method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Summary {
    pub(crate) count: usize,
    pub(crate) mean: f32,
    pub(crate) median: f32,
    pub(crate) p95: f32,
}

/// Computes summary statistics from a slice of timing samples.
///
/// Non-finite values (NaN, infinity) are discarded. If no finite samples
/// remain, returns None. The median is the middle element for odd counts
/// or the average of the two middle elements for even counts. The p95
/// uses the nearest-rank method: rank = ceil(0.95 * n), index = rank - 1.
pub(crate) fn summarize(samples: &[f32]) -> Option<Summary> {
    let finite: Vec<f32> = samples.iter().copied().filter(|s| s.is_finite()).collect();

    if finite.is_empty() {
        return None;
    }

    let mut sorted = finite;
    sorted.sort_by(f32::total_cmp);

    let n = sorted.len();
    let sum: f32 = sorted.iter().sum();
    let mean = sum / n as f32;

    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    };

    let rank = (n * 95).div_ceil(100);
    let p95_index = rank.saturating_sub(1).min(n - 1);
    let p95 = sorted[p95_index];

    Some(Summary {
        count: n,
        mean,
        median,
        p95,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_has_expected_vertex_and_index_counts_and_validates() {
        // Arrange
        let mesh = scene_mesh(QUADS_PER_SIDE);
        let quads = (LAYERS * QUADS_PER_SIDE * QUADS_PER_SIDE) as usize;

        // Act
        let vertex_count = mesh.positions.len();

        // Assert
        assert_eq!(vertex_count, quads * 4);
        assert_eq!(mesh.indices.len(), quads * 6);
        assert_eq!(mesh.validate(), Ok(()));
        for [x, y, _] in &mesh.positions {
            assert!((-1.0..=1.0).contains(x), "x out of range: {x}");
            assert!((-1.0..=1.0).contains(y), "y out of range: {y}");
        }
        let first_z = mesh.positions[0][2];
        assert!((first_z - 0.9).abs() < 1e-6, "first z: {first_z}");
        let last_z = mesh.positions[vertex_count - 1][2];
        assert!((last_z - 0.1).abs() < 1e-6, "last z: {last_z}");
    }

    #[test]
    fn scene_is_deterministic() {
        // Arrange
        let first = scene_mesh(QUADS_PER_SIDE);

        // Act
        let second = scene_mesh(QUADS_PER_SIDE);

        // Assert
        assert_eq!(first, second);
    }

    #[test]
    fn lights_are_deterministic() {
        // Arrange
        let first = lights(LIGHT_COUNT);

        // Act
        let second = lights(LIGHT_COUNT);

        // Assert
        assert_eq!(first, second);
    }

    #[test]
    fn every_light_is_inside_the_volume_with_radius_in_range() {
        // Arrange
        let lights = lights(LIGHT_COUNT);

        // Act
        // (the check below is the whole test)

        // Assert
        for light in &lights {
            let [x, y, z, radius] = light.position_radius;
            assert!((-1.0..=1.0).contains(&x), "x out of range: {x}");
            assert!((-1.0..=1.0).contains(&y), "y out of range: {y}");
            assert!((0.05..=0.95).contains(&z), "z out of range: {z}");
            assert!((0.1..=0.35).contains(&radius), "radius out of range: {radius}");
        }
    }

    #[test]
    fn light_bytes_has_32_bytes_per_light_and_starts_with_first_x() {
        // Arrange
        let lights = lights(4);

        // Act
        let bytes = light_bytes(&lights);

        // Assert
        assert_eq!(bytes.len(), 32 * lights.len());
        let first_x = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(first_x, lights[0].position_radius[0]);
    }

    #[test]
    fn params_bytes_round_trips_its_layout() {
        // Arrange
        let extent = [1280, 720];
        let grid = [16, 9, 24];

        // Act
        let bytes = params_bytes(extent, 256, grid);

        // Assert
        let decoded: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        assert_eq!(decoded, [1280, 720, 256, 0, 16, 9, 24, 0]);
    }

    #[test]
    fn summarize_empty_is_none() {
        // Arrange
        let samples: [f32; 0] = [];

        // Act
        let result = summarize(&samples);

        // Assert
        assert!(result.is_none());
    }

    #[test]
    fn summarize_all_nan_is_none() {
        // Arrange
        let samples = [f32::NAN, f32::NAN, f32::NAN];

        // Act
        let result = summarize(&samples);

        // Assert
        assert!(result.is_none());
    }

    #[test]
    fn summarize_odd_count_median_and_mean() {
        // Arrange
        let samples = [3.0, 1.0, 2.0];

        // Act
        let result = summarize(&samples);

        // Assert
        assert_eq!(
            result,
            Some(Summary {
                count: 3,
                mean: 2.0,
                median: 2.0,
                p95: 3.0,
            })
        );
    }

    #[test]
    fn summarize_even_count_averages_middle_pair() {
        // Arrange
        let samples = [1.0, 2.0, 3.0, 4.0];

        // Act
        let result = summarize(&samples);

        // Assert
        assert_eq!(result.map(|summary| summary.median), Some(2.5));
    }

    #[test]
    fn summarize_p95_is_nearest_rank() {
        // Arrange
        let samples: Vec<f32> = (1..=100).map(|i| i as f32).collect();

        // Act
        let result = summarize(&samples);

        // Assert
        assert_eq!(result.map(|summary| summary.p95), Some(95.0));
    }

    #[test]
    fn summarize_ignores_nan_samples() {
        // Arrange
        let samples = [1.0, f32::NAN, 3.0];

        // Act
        let result = summarize(&samples);

        // Assert
        assert_eq!(result.map(|summary| summary.count), Some(2));
        assert_eq!(result.map(|summary| summary.mean), Some(2.0));
    }
}
