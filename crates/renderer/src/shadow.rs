//! Cascaded shadow map fitting for the directional light (C1).
//!
//! CPU only. The camera's view range, up to the shadow distance, is split
//! into cascades; each cascade's frustum slice is enclosed in a sphere and
//! rendered with an orthographic projection from the light.
//!
//! Two choices keep the shadows from shimmering as the camera moves:
//!
//! - The sphere's radius depends only on the slice's shape, which a camera
//!   rotation does not change, so a cascade never resizes while the player
//!   turns. It is rounded up to a sixteenth of a metre so float noise cannot
//!   resize it either.
//! - The projection's centre is snapped to whole texels in a light space that
//!   depends only on the light direction. The texel grid is then fixed in the
//!   world, and a moving camera slides the cascade across it by whole texels.

use glam::{Mat4, Vec3};

use crate::frame::{MAX_CASCADES, ShadowSettings};

/// Radii are rounded up to a multiple of this, in metres.
const RADIUS_STEP: f32 = 1.0 / 16.0;

/// The light direction is treated as vertical above this `|y|`, and the light
/// space takes world +Z as its up vector instead of +Y.
const VERTICAL_LIGHT: f32 = 0.99;

/// A frustum whose far plane is not wider than its near plane by at least
/// this fraction is treated as orthographic, which has no eye to measure
/// depth from; such a camera gets no cascades.
const PERSPECTIVE_EPSILON: f32 = 1e-4;

/// The smallest cascade resolution used; the one-texel padding needs more
/// than two texels.
const MIN_RESOLUTION: u32 = 16;

/// One cascade, ready for the shadow pass and for sampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Cascade {
    /// World to light clip space: x and y in `-1..1` across the cascade,
    /// depth `0..1` from the light (ADR-0008's depth range).
    pub(crate) view_proj: Mat4,
    /// Where this cascade ends, as a distance along the camera's view
    /// direction. The shader picks the first cascade whose `far` lies beyond
    /// the fragment.
    pub(crate) far: f32,
    /// The world-space size of one shadow texel, for the normal offset bias.
    pub(crate) texel_world: f32,
}

/// Where each cascade ends, from `near` to `far`, by the practical split
/// scheme: a blend of logarithmic and uniform splits weighted by `lambda`.
///
/// Returns `count` distances in increasing order; the last is exactly `far`.
/// Empty when `count` is 0 or the range is not a positive interval.
pub(crate) fn split_distances(near: f32, far: f32, count: u32, lambda: f32) -> Vec<f32> {
    if count == 0 || !(near > 0.0 && far > near && far.is_finite()) {
        return Vec::new();
    }
    let lambda = lambda.clamp(0.0, 1.0);
    (1..=count)
        .map(|index| {
            if index == count {
                return far;
            }
            let fraction = index as f32 / count as f32;
            let logarithmic = near * (far / near).powf(fraction);
            let uniform = near + (far - near) * fraction;
            lambda * logarithmic + (1.0 - lambda) * uniform
        })
        .collect()
}

/// The camera frustum as eight world-space corners, near plane first, and
/// the view-space depth of each plane.
struct Frustum {
    near_corners: [Vec3; 4],
    far_corners: [Vec3; 4],
    near: f32,
    far: f32,
}

impl Frustum {
    /// From the inverse view-projection of a perspective camera (`0..1`
    /// depth). `None` when the matrix is degenerate or orthographic.
    ///
    /// The eye is not taken from the caller: the planes' view depths follow
    /// from their widths, since a perspective frustum's width grows in
    /// proportion to depth, so `near / far = near_width / far_width`.
    fn new(inverse_view_proj: &Mat4) -> Option<Self> {
        const NDC: [(f32, f32); 4] = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        let corner = |(x, y): (f32, f32), z: f32| inverse_view_proj.project_point3(Vec3::new(x, y, z));
        let near_corners = NDC.map(|xy| corner(xy, 0.0));
        let far_corners = NDC.map(|xy| corner(xy, 1.0));
        let centre = |corners: &[Vec3; 4]| corners.iter().copied().sum::<Vec3>() / 4.0;
        let (near_centre, far_centre) = (centre(&near_corners), centre(&far_corners));
        let length = far_centre.distance(near_centre);
        let near_width = near_corners[0].distance(near_centre);
        let far_width = far_corners[0].distance(far_centre);
        let growth = far_width - near_width;
        if growth <= far_width * PERSPECTIVE_EPSILON {
            return None;
        }
        let near = length * near_width / growth;
        let far = near + length;
        let finite = near_corners
            .iter()
            .chain(&far_corners)
            .all(|corner| corner.is_finite());
        (finite && near > 0.0 && far > near).then_some(Self {
            near_corners,
            far_corners,
            near,
            far,
        })
    }

    /// The eight corners of the slice between view depths `from` and `to`.
    /// View depth is linear along each corner ray, so this is exact.
    fn slice(&self, from: f32, to: f32) -> [Vec3; 8] {
        let at = |depth: f32, index: usize| {
            let t = (depth - self.near) / (self.far - self.near);
            self.near_corners[index].lerp(self.far_corners[index], t)
        };
        std::array::from_fn(|index| at(if index < 4 { from } else { to }, index % 4))
    }
}

/// Fits the cascades for one frame.
///
/// `view_proj` is the camera's (column-major, `0..1` depth) and `to_light`
/// the unit direction toward the light. Returns no
/// cascades when shadows are off or the inputs are degenerate, which the
/// caller treats as "unshadowed", never as an error.
pub(crate) fn fit(view_proj: &[[f32; 4]; 4], to_light: [f32; 3], settings: &ShadowSettings) -> Vec<Cascade> {
    let count = settings.cascades.min(MAX_CASCADES);
    if count == 0 || settings.resolution == 0 {
        return Vec::new();
    }
    let Some(to_light) = Vec3::from(to_light).try_normalize() else {
        return Vec::new();
    };
    let inverse = Mat4::from_cols_array_2d(view_proj).inverse();
    let Some(frustum) = Frustum::new(&inverse) else {
        return Vec::new();
    };
    let shadow_far = frustum.far.min(settings.max_distance);
    let splits = split_distances(frustum.near, shadow_far, count, settings.split_lambda);

    let light_view = light_view(to_light);
    let mut from = frustum.near;
    splits
        .into_iter()
        .map(|to| {
            let cascade = fit_slice(&frustum.slice(from, to), &light_view, settings);
            from = to;
            Cascade { far: to, ..cascade }
        })
        .collect()
}

/// A rotation-only view looking along the light, fixed for a light
/// direction so that the texel grid snapped in it is fixed in the world.
fn light_view(to_light: Vec3) -> Mat4 {
    let up = if to_light.y.abs() > VERTICAL_LIGHT {
        Vec3::Z
    } else {
        Vec3::Y
    };
    glam::camera::rh::view::look_to_mat4(Vec3::ZERO, -to_light, up)
}

fn fit_slice(corners: &[Vec3; 8], light_view: &Mat4, settings: &ShadowSettings) -> Cascade {
    let centre = corners.iter().copied().sum::<Vec3>() / 8.0;
    let radius = corners
        .iter()
        .map(|corner| corner.distance(centre))
        .fold(0.0, f32::max);
    let radius = ((radius / RADIUS_STEP).ceil() * RADIUS_STEP).max(RADIUS_STEP);
    // Snapping moves the centre by up to a texel, so the box is one texel
    // wider than the sphere on each side: r' = r + 2r'/n gives this r'.
    let resolution = settings.resolution.max(MIN_RESOLUTION) as f32;
    let radius = radius * resolution / (resolution - 2.0);

    let texel = 2.0 * radius / resolution;
    let centre = light_view.transform_point3(centre);
    let snapped_x = (centre.x / texel).floor() * texel;
    let snapped_y = (centre.y / texel).floor() * texel;

    // The light looks down -Z, so a point's distance from it is -z. Casters
    // between the light and the slice are kept by reaching `max_distance`
    // further toward the light than the slice itself.
    let distance = -centre.z;
    // WebGPU's convention: depth 0..1, y up.
    let projection = glam::camera::rh::proj::directx::orthographic(
        snapped_x - radius,
        snapped_x + radius,
        snapped_y - radius,
        snapped_y + radius,
        distance - radius - settings.max_distance,
        distance + radius,
    );
    Cascade {
        view_proj: projection * *light_view,
        far: 0.0,
        texel_world: texel,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::camera::rh::proj::directx::perspective;
    use glam::camera::rh::view::look_at_mat4;

    fn camera(eye: Vec3, target: Vec3) -> [[f32; 4]; 4] {
        let view = look_at_mat4(eye, target, Vec3::Y);
        let proj = perspective(60.0_f32.to_radians(), 16.0 / 9.0, 0.1, 200.0);
        (proj * view).to_cols_array_2d()
    }

    fn assert_slices_contained(view_proj: [[f32; 4]; 4], eye: Vec3, cascades: &[Cascade]) {
        let inv = Mat4::from_cols_array_2d(&view_proj).inverse();
        let corners = [
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
        ];
        let near_c = corners.map(|c| inv.project_point3(c));
        let far_c = corners.map(|c| inv.project_point3(Vec3::new(c.x, c.y, 1.0)));

        let near_centre = near_c.iter().fold(Vec3::ZERO, |acc, &v| acc + v) / 4.0;
        let far_centre = far_c.iter().fold(Vec3::ZERO, |acc, &v| acc + v) / 4.0;
        let forward = (far_centre - near_centre).normalize();

        let near_depth = (near_centre - eye).dot(forward);
        let far_depth = (far_centre - eye).dot(forward);

        for (i, cascade) in cascades.iter().enumerate() {
            let from = if i == 0 { near_depth } else { cascades[i - 1].far };
            let to = cascade.far;

            for k in 0..4 {
                for depth in [from, to] {
                    let t = (depth - near_depth) / (far_depth - near_depth);
                    let point = near_c[k].lerp(far_c[k], t);
                    let ndc = cascade.view_proj.project_point3(point);

                    assert!(
                        ndc.x >= -1.0 - 1e-4 && ndc.x <= 1.0 + 1e-4,
                        "Cascade {} point {:?} NDC x {} out of range",
                        i,
                        point,
                        ndc.x
                    );
                    assert!(
                        ndc.y >= -1.0 - 1e-4 && ndc.y <= 1.0 + 1e-4,
                        "Cascade {} point {:?} NDC y {} out of range",
                        i,
                        point,
                        ndc.y
                    );
                    assert!(
                        ndc.z >= -1e-4 && ndc.z <= 1.0 + 1e-4,
                        "Cascade {} point {:?} NDC z {} out of range",
                        i,
                        point,
                        ndc.z
                    );
                }
            }
        }
    }

    #[test]
    fn split_distances_is_increasing_and_ends_at_far() {
        // Arrange
        let near = 0.1;
        let far = 80.0;
        let count = 4;
        let lambda = 0.75;

        // Act
        let distances = split_distances(near, far, count, lambda);

        // Assert
        assert_eq!(distances.len(), 4);
        for i in 1..distances.len() {
            assert!(
                distances[i] > distances[i - 1],
                "Distances must be strictly increasing"
            );
        }
        assert_eq!(distances.last().copied().unwrap_or(0.0), far);
        assert!(distances.first().copied().unwrap_or(0.0) > near);
    }

    #[test]
    fn split_distances_uniform_when_lambda_is_zero() {
        // Arrange
        let near = 1.0;
        let far = 81.0;
        let count = 4;
        let lambda = 0.0;
        let expected = [21.0, 41.0, 61.0, 81.0];

        // Act
        let distances = split_distances(near, far, count, lambda);

        // Assert
        assert_eq!(distances.len(), 4);
        for i in 0..4 {
            assert!(
                (distances[i] - expected[i]).abs() < 1e-4,
                "Distance {} expected {}, got {}",
                i,
                expected[i],
                distances[i]
            );
        }
    }

    #[test]
    fn split_distances_logarithmic_when_lambda_is_one() {
        // Arrange
        let near = 1.0;
        let far = 16.0;
        let count = 4;
        let lambda = 1.0;
        let expected = [2.0, 4.0, 8.0, 16.0];

        // Act
        let distances = split_distances(near, far, count, lambda);

        // Assert
        assert_eq!(distances.len(), 4);
        for i in 0..4 {
            assert!(
                (distances[i] - expected[i]).abs() < 1e-4,
                "Distance {} expected {}, got {}",
                i,
                expected[i],
                distances[i]
            );
        }
    }

    #[test]
    fn split_distances_empty_for_invalid_ranges() {
        // Arrange
        let cases = [
            (1.0, 10.0, 0, 0.5),
            (0.0, 10.0, 4, 0.5),
            (-1.0, 10.0, 4, 0.5),
            (5.0, 5.0, 4, 0.5),
            (5.0, 1.0, 4, 0.5),
            (1.0, f32::INFINITY, 4, 0.5),
        ];

        // Act & Assert
        for (near, far, count, lambda) in cases {
            let distances = split_distances(near, far, count, lambda);
            assert!(
                distances.is_empty(),
                "Expected empty for near={}, far={}, count={}, lambda={}",
                near,
                far,
                count,
                lambda
            );
        }
    }

    #[test]
    fn fit_returns_one_cascade_per_count_with_increasing_far() {
        // Arrange
        let eye = Vec3::new(3.0, 5.0, 12.0);
        let target = Vec3::ZERO;
        let settings = ShadowSettings::default();
        let to_light = [0.35, 0.8, 0.45];
        let view_proj = camera(eye, target);

        // Act
        let cascades = fit(&view_proj, to_light, &settings);

        // Assert
        assert_eq!(cascades.len(), 4);
        for i in 1..cascades.len() {
            assert!(
                cascades[i].far > cascades[i - 1].far,
                "Cascade far must be strictly increasing"
            );
        }
        let Some(last) = cascades.last() else {
            panic!("expected cascades");
        };
        assert!(
            (last.far - 80.0).abs() < 1e-3,
            "Last cascade far should be close to 80.0, got {}",
            last.far
        );
    }

    #[test]
    fn cascades_cover_each_camera_slice() {
        // Arrange
        let eye = Vec3::new(3.0, 5.0, 12.0);
        let target = Vec3::ZERO;
        let settings = ShadowSettings::default();
        let to_light = [0.35, 0.8, 0.45];
        let view_proj = camera(eye, target);

        // Act
        let cascades = fit(&view_proj, to_light, &settings);

        // Assert
        assert_slices_contained(view_proj, eye, &cascades);
    }

    #[test]
    fn cascade_origin_moves_by_whole_texels_when_camera_moves_sub_texel() {
        // Arrange
        let eye = Vec3::new(3.0, 5.0, 12.0);
        let target = Vec3::ZERO;
        let settings = ShadowSettings::default();
        let to_light = [0.35, 0.8, 0.45];
        let view_proj_a = camera(eye, target);
        let cascades_a = fit(&view_proj_a, to_light, &settings);

        let d = 0.3 * cascades_a[0].texel_world;
        let eye2 = eye + Vec3::new(d, 0.0, 0.0);
        let target2 = target + Vec3::new(d, 0.0, 0.0);
        let view_proj_b = camera(eye2, target2);
        let cascades_b = fit(&view_proj_b, to_light, &settings);

        let n = settings.resolution as f32;

        // Act & Assert
        assert_eq!(cascades_a.len(), cascades_b.len());
        for i in 0..cascades_a.len() {
            let a = cascades_a[i].view_proj.project_point3(Vec3::ZERO);
            let b = cascades_b[i].view_proj.project_point3(Vec3::ZERO);

            for axis in [0, 1] {
                let texels = (b[axis] - a[axis]) * n / 2.0;
                assert!(
                    (texels - texels.round()).abs() < 1e-3,
                    "Cascade {} axis {} texels {} not close to integer",
                    i,
                    axis,
                    texels
                );
                assert!(
                    texels.round().abs() <= 1.0,
                    "Cascade {} axis {} texels {} too large",
                    i,
                    axis,
                    texels
                );
            }
        }
    }

    #[test]
    fn texel_world_is_invariant_under_camera_rotation() {
        // Arrange
        let eye = Vec3::new(3.0, 5.0, 12.0);
        let target1 = Vec3::ZERO;
        let target2 = Vec3::new(5.0, 0.0, -3.0);
        let settings = ShadowSettings::default();
        let to_light = [0.35, 0.8, 0.45];

        let view_proj1 = camera(eye, target1);
        let view_proj2 = camera(eye, target2);

        // Act
        let cascades1 = fit(&view_proj1, to_light, &settings);
        let cascades2 = fit(&view_proj2, to_light, &settings);

        // Assert
        assert_eq!(cascades1.len(), cascades2.len());
        for i in 0..cascades1.len() {
            let a = cascades1[i].texel_world;
            let b = cascades2[i].texel_world;
            assert!(
                (a - b).abs() <= 1e-6 * a.abs(),
                "Cascade {} texel_world differs: {} vs {}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn degenerate_inputs_give_no_cascades() {
        // Arrange
        let eye = Vec3::new(3.0, 5.0, 12.0);
        let target = Vec3::ZERO;
        let view_proj = camera(eye, target);
        let to_light = [0.35, 0.8, 0.45];

        // Act & Assert
        let settings_a = ShadowSettings {
            cascades: 0,
            ..ShadowSettings::default()
        };
        assert!(fit(&view_proj, to_light, &settings_a).is_empty());

        let settings_b = ShadowSettings {
            resolution: 0,
            ..ShadowSettings::default()
        };
        assert!(fit(&view_proj, to_light, &settings_b).is_empty());

        let to_light_c = [0.0, 0.0, 0.0];
        assert!(fit(&view_proj, to_light_c, &ShadowSettings::default()).is_empty());

        let view_proj_d = [[0.0; 4]; 4];
        assert!(fit(&view_proj_d, to_light, &ShadowSettings::default()).is_empty());
    }

    #[test]
    fn cascade_count_is_clamped_to_max_cascades() {
        // Arrange
        let eye = Vec3::new(3.0, 5.0, 12.0);
        let target = Vec3::ZERO;
        let settings = ShadowSettings {
            cascades: 99,
            ..ShadowSettings::default()
        };
        let to_light = [0.35, 0.8, 0.45];
        let view_proj = camera(eye, target);

        // Act
        let cascades = fit(&view_proj, to_light, &settings);

        // Assert
        assert_eq!(cascades.len(), MAX_CASCADES as usize);
    }

    #[test]
    fn straight_down_light_gives_finite_contained_cascades() {
        // Arrange
        let eye = Vec3::new(3.0, 5.0, 12.0);
        let target = Vec3::ZERO;
        let settings = ShadowSettings::default();
        let to_light = [0.0, 1.0, 0.0];
        let view_proj = camera(eye, target);

        // Act
        let cascades = fit(&view_proj, to_light, &settings);

        // Assert
        assert!(!cascades.is_empty());
        assert_eq!(cascades.len(), 4);
        for cascade in &cascades {
            for val in cascade.view_proj.to_cols_array() {
                assert!(val.is_finite(), "view_proj element not finite: {}", val);
            }
            assert!(cascade.far.is_finite(), "far not finite");
            assert!(cascade.texel_world.is_finite(), "texel_world not finite");
            assert!(cascade.texel_world > 0.0, "texel_world must be positive");
        }
        assert_slices_contained(view_proj, eye, &cascades);
    }

    #[test]
    fn snapping_never_pushes_a_slice_corner_out_of_its_cascade() {
        // Arrange: a coarse atlas makes a texel large, so a one-texel snap
        // without padding would move a corner visibly outside the box.
        let settings = ShadowSettings {
            resolution: 16,
            ..ShadowSettings::default()
        };
        let to_light = [0.35, 0.8, 0.45];

        for step in 0..64 {
            // Act: the camera slides by fractions of a coarse texel.
            let offset = Vec3::new(step as f32 * 0.37, step as f32 * 0.11, step as f32 * -0.23);
            let eye = Vec3::new(3.0, 5.0, 12.0) + offset;
            let view_proj = camera(eye, offset);
            let cascades = fit(&view_proj, to_light, &settings);

            // Assert
            assert_slices_contained(view_proj, eye, &cascades);
        }
    }

    #[test]
    fn a_caster_between_the_light_and_the_slice_lands_in_the_depth_range() {
        // Arrange
        let eye = Vec3::new(3.0, 5.0, 12.0);
        let settings = ShadowSettings::default();
        let to_light = Vec3::new(0.35, 0.8, 0.45).normalize();
        let cascades = fit(&camera(eye, Vec3::ZERO), to_light.to_array(), &settings);
        let Some(first) = cascades.first() else {
            panic!("shadows are on");
        };

        // Act: a point on the ground in view, and the same point lifted far
        // toward the light, as a tall tree or a cliff above it would be.
        let ground = Vec3::new(0.0, 0.0, 6.0);
        let caster = ground + to_light * (settings.max_distance * 0.5);
        let ground_depth = first.view_proj.project_point3(ground).z;
        let caster_depth = first.view_proj.project_point3(caster).z;

        // Assert
        assert!((0.0..=1.0).contains(&ground_depth), "ground depth {ground_depth}");
        assert!(
            (0.0..=1.0).contains(&caster_depth),
            "a caster above the slice is still drawn into the map: depth {caster_depth}"
        );
        assert!(caster_depth < ground_depth, "and it is nearer the light");
    }
}
