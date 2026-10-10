//! CPU reference for the B2 orthographic cluster culler. The benchmark's
//! GPU source uses precisely these bounds and a conservative sphere/AABB test.

pub(crate) const GRID: [u32; 3] = [16, 9, 16];

pub(crate) fn cluster_count(grid: [u32; 3]) -> Option<u32> {
    if grid.contains(&0) {
        return None;
    }
    grid[0].checked_mul(grid[1])?.checked_mul(grid[2])
}

pub(crate) fn list_words(grid: [u32; 3], lights: u32) -> Option<u32> {
    cluster_count(grid)?.checked_mul(lights.checked_add(1)?)
}

#[cfg(test)]
fn intersects(light: [f32; 4], cell: [u32; 3], grid: [u32; 3]) -> bool {
    let mut distance2 = 0.0;
    for axis in 0..3 {
        let (origin, extent) = if axis == 2 { (0.0, 1.0) } else { (-1.0, 2.0) };
        let size = extent / grid[axis] as f32;
        let low = origin + cell[axis] as f32 * size;
        let delta = light[axis] - light[axis].clamp(low, low + size);
        distance2 += delta * delta;
    }
    distance2 <= light[3] * light[3]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_are_checked_and_zero_lights_still_have_a_count_word() {
        assert_eq!(cluster_count(GRID), Some(2304));
        assert_eq!(list_words(GRID, 0), Some(2304));
        assert_eq!(list_words([1, 1, 1], 100), Some(101));
        assert_eq!(cluster_count([0, 1, 1]), None);
        assert_eq!(cluster_count([u32::MAX, 2, 1]), None);
        assert_eq!(list_words(GRID, u32::MAX), None);
    }

    #[test]
    fn tangent_spheres_and_lights_on_cluster_boundaries_are_not_dropped() {
        assert!(intersects([0.0, 0.0, 0.5, 0.0], [0, 0, 0], [2, 2, 2]));
        assert!(intersects([0.0, 0.0, 0.5, 0.0], [1, 1, 1], [2, 2, 2]));
        assert!(intersects([2.0, 0.0, 0.5, 1.0], [0, 0, 0], [1, 1, 1]));
        assert!(!intersects([2.001, 0.0, 0.5, 1.0], [0, 0, 0], [1, 1, 1]));
    }

    #[test]
    fn every_point_in_a_light_sphere_keeps_the_light_in_its_cluster() {
        let light = [0.2, -0.3, 0.6, 0.35];
        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    let p = [x as f32 / 16.0 - 1.0, y as f32 / 16.0 - 1.0, z as f32 / 32.0];
                    let d: f32 = (0..3).map(|a| (p[a] - light[a]).powi(2)).sum();
                    if d <= light[3] * light[3] {
                        let cell = [
                            ((p[0] + 1.0) * 0.5 * GRID[0] as f32) as u32,
                            ((p[1] + 1.0) * 0.5 * GRID[1] as f32) as u32,
                            (p[2] * GRID[2] as f32) as u32,
                        ];
                        assert!(intersects(light, cell, GRID));
                    }
                }
            }
        }
    }
}
