//! A fixed scene for golden-image tests.
//!
//! Built from nothing but this crate, since the engine's primitives sit above
//! it in the dependency order. It exercises what the current passes can do: a
//! lit ground, vertex colours under a tint, a flash overlay, a rotated body,
//! and debug lines partly hidden behind geometry, so a pass that loses depth
//! testing, vertex colour or the flash changes the image.

use crate::{Color, DrawCommand, MaterialDesc, MeshData, RenderFrame, Renderer, RendererConfig, golden};

/// Not a multiple of 64 pixels, so every readback row carries padding.
pub(crate) const WIDTH: u32 = 160;
pub(crate) const HEIGHT: u32 = 120;

type Mat4 = [[f32; 4]; 4];
type Vec3 = [f32; 3];

const IDENTITY: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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

fn normalize(v: Vec3) -> Vec3 {
    let length = dot(v, v).sqrt();
    [v[0] / length, v[1] / length, v[2] / length]
}

/// Column-major product `a * b`, matching how `view_proj` crosses the frame
/// boundary.
fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for (column, out_column) in out.iter_mut().enumerate() {
        for (row, value) in out_column.iter_mut().enumerate() {
            *value = (0..4).map(|k| a[k][row] * b[column][k]).sum();
        }
    }
    out
}

/// Right-handed, `0..1` depth (ADR-0008).
fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y / 2.0).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, near * far / (near - far), 0.0],
    ]
}

fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    let forward = normalize(sub(target, eye));
    let side = normalize(cross(forward, up));
    let up = cross(side, forward);
    [
        [side[0], up[0], -forward[0], 0.0],
        [side[1], up[1], -forward[1], 0.0],
        [side[2], up[2], -forward[2], 0.0],
        [-dot(side, eye), -dot(up, eye), dot(forward, eye), 1.0],
    ]
}

/// A turn about Y, then a translation. Rigid, so the turn alone is also the
/// normal matrix.
fn turned_at(angle: f32, at: Vec3) -> (Mat4, Mat4) {
    let (sin, cos) = angle.sin_cos();
    let rotation = [
        [cos, 0.0, -sin, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [sin, 0.0, cos, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut model = rotation;
    model[3] = [at[0], at[1], at[2], 1.0];
    (model, rotation)
}

fn ground(half: f32) -> MeshData {
    MeshData {
        positions: vec![
            [-half, 0.0, -half],
            [-half, 0.0, half],
            [half, 0.0, half],
            [half, 0.0, -half],
        ],
        normals: vec![[0.0, 1.0, 0.0]; 4],
        uvs: vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        indices: vec![0, 1, 2, 0, 2, 3],
        ..MeshData::default()
    }
}

/// A unit cube resting on the ground, each face painted its own colour.
fn painted_cube() -> MeshData {
    // (normal, u axis, v axis, colour), with `u cross v == normal` so the shared
    // winding is counter-clockwise from outside.
    const FACES: [(Vec3, Vec3, Vec3, Vec3); 6] = [
        (
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.2, 0.2],
        ),
        (
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.9, 0.2],
        ),
        (
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.9, 0.9, 0.9],
        ),
        (
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.1, 0.1, 0.1],
        ),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.3, 0.9]),
        (
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.8, 0.2],
        ),
    ];
    const CORNERS: [(f32, f32); 4] = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];

    let mut mesh = MeshData::default();
    for (normal, u_axis, v_axis, color) in FACES {
        let first = mesh.positions.len() as u32;
        for (su, sv) in CORNERS {
            mesh.positions.push([
                (normal[0] + su * u_axis[0] + sv * v_axis[0]) * 0.5,
                (normal[1] + su * u_axis[1] + sv * v_axis[1]) * 0.5 + 0.5,
                (normal[2] + su * u_axis[2] + sv * v_axis[2]) * 0.5,
            ]);
            mesh.normals.push(normal);
            mesh.uvs.push([(su + 1.0) / 2.0, (1.0 - sv) / 2.0]);
            mesh.colors.push(color);
        }
        mesh.indices
            .extend_from_slice(&[first, first + 1, first + 2, first, first + 2, first + 3]);
    }
    mesh
}

/// Uploads the scene's meshes and materials and returns its frame.
pub(crate) fn build(renderer: &mut Renderer) -> RenderFrame {
    let ground_mesh = renderer.upload_mesh(&ground(4.0)).expect("the ground is valid");
    let cube_mesh = renderer.upload_mesh(&painted_cube()).expect("the cube is valid");

    let ground_material = renderer.create_material(&MaterialDesc {
        base_color: [0.35, 0.45, 0.3, 1.0],
        ..MaterialDesc::default()
    });
    let plain = renderer.create_material(&MaterialDesc::default());
    let tinted_flashing = renderer.create_material(&MaterialDesc {
        base_color: [1.0, 0.7, 0.7, 1.0],
        flash: [1.0, 1.0, 1.0, 0.4],
        ..MaterialDesc::default()
    });

    let view = look_at([3.5, 3.0, 5.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]);
    let projection = perspective(60f32.to_radians(), WIDTH as f32 / HEIGHT as f32, 0.1, 50.0);
    let mut frame = RenderFrame::new(mul(&projection, &view), Color::rgb(0.05, 0.07, 0.12));

    frame.push(DrawCommand {
        mesh: ground_mesh,
        material: ground_material,
        model: IDENTITY,
        normal_matrix: IDENTITY,
    });
    for (material, angle, at) in [
        (plain, 0.0, [-0.9, 0.0, 0.3]),
        (tinted_flashing, 0.6, [1.0, 0.0, -0.4]),
    ] {
        let (model, normal_matrix) = turned_at(angle, at);
        frame.push(DrawCommand {
            mesh: cube_mesh,
            material,
            model,
            normal_matrix,
        });
    }

    // An axis gizmo at the origin, half buried in the plain cube's shadow side,
    // and a long line running behind the second cube so depth testing shows.
    frame.push_debug_line([0.0, 0.05, 0.0], [1.5, 0.05, 0.0], [1.0, 0.0, 0.0, 1.0]);
    frame.push_debug_line([0.0, 0.05, 0.0], [0.0, 1.5, 0.0], [0.0, 1.0, 0.0, 1.0]);
    frame.push_debug_line([0.0, 0.05, 0.0], [0.0, 0.05, 1.5], [0.0, 0.0, 1.0, 1.0]);
    frame.push_debug_line([-3.0, 0.5, -1.5], [3.0, 0.5, -1.5], [1.0, 0.0, 1.0, 1.0]);
    frame
}

fn headless() -> Renderer {
    Renderer::headless(WIDTH, HEIGHT, RendererConfig::default())
        .expect("these tests are ignored unless a GPU adapter exists")
}

#[test]
fn column_major_products_compose_right_to_left() {
    let (translate, _) = turned_at(0.0, [1.0, 2.0, 3.0]);
    let (turn, _) = turned_at(std::f32::consts::FRAC_PI_2, [0.0, 0.0, 0.0]);
    // Turning +X a quarter about Y gives -Z; translating after adds (1, 2, 3).
    let combined = mul(&translate, &turn);
    let x_axis = combined[0];
    assert!((x_axis[0] - 0.0).abs() < 1e-6 && (x_axis[2] + 1.0).abs() < 1e-6);
    assert_eq!(combined[3], [1.0, 2.0, 3.0, 1.0]);
}

#[test]
fn the_camera_looks_at_its_target() {
    let view = look_at([3.5, 3.0, 5.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]);
    let view_proj = mul(&perspective(1.0, 1.0, 0.1, 50.0), &view);
    // The target projects to the centre of the screen.
    let clip: Vec<f32> = (0..4)
        .map(|row| {
            view_proj[0][row] * 0.0 + view_proj[1][row] * 0.5 + view_proj[2][row] * 0.0 + view_proj[3][row]
        })
        .collect();
    assert!((clip[0] / clip[3]).abs() < 1e-5);
    assert!((clip[1] / clip[3]).abs() < 1e-5);
    let depth = clip[2] / clip[3];
    assert!(
        (0.0..1.0).contains(&depth),
        "in front of the camera, depth {depth}"
    );
}

#[test]
fn the_painted_cube_is_a_valid_mesh_on_the_ground() {
    let cube = painted_cube();
    assert_eq!(cube.validate(), Ok(()));
    let bounds = cube.bounds().expect("not empty");
    assert_eq!(bounds.min[1], 0.0, "it rests on the ground");
    assert_eq!(bounds.max[1], 1.0);
    assert_eq!(ground(4.0).validate(), Ok(()));
}

#[test]
#[ignore = "requires a GPU adapter and driver"]
fn a_headless_frame_reads_back_whole_and_in_rgba_order() {
    use crate::post::tonemap::mirror::display_code;

    let mut renderer = headless();
    // Pure scene red and blue: any channel swap in the readback (the window
    // swapchain is BGRA) or a lost row would show here.
    let scene = [3.0, 0.0, 0.5];
    let frame = RenderFrame::new(IDENTITY, Color::rgb(3.0, 0.0, 0.5));
    let image = renderer
        .render_to_image(&frame)
        .expect("a headless frame reads back");
    let [want_r, want_g, want_b] = display_code(scene, &frame.post);

    assert_eq!((image.width, image.height), (WIDTH, HEIGHT));
    assert_eq!(image.rgba8.len(), (WIDTH * HEIGHT * 4) as usize);
    assert!(
        want_r > want_b && want_b > want_g,
        "the expectation itself is lopsided"
    );
    for (x, y) in [(0, 0), (WIDTH - 1, 0), (0, HEIGHT - 1), (WIDTH - 1, HEIGHT - 1)] {
        let [r, g, b, a] = image.pixel(x, y).expect("in bounds");
        assert!(
            r.abs_diff(want_r) <= 1 && g.abs_diff(want_g) <= 1 && b.abs_diff(want_b) <= 1,
            "({x}, {y}) was {:?}, expected {:?}",
            [r, g, b],
            [want_r, want_g, want_b]
        );
        assert_eq!(a, 255, "opaque at ({x}, {y})");
    }
}

#[test]
#[ignore = "requires a GPU adapter and driver"]
fn a_zero_sized_headless_renderer_reports_instead_of_reading() {
    let mut renderer = headless();
    renderer.resize(0, 0);
    assert!(renderer.render_to_image(&RenderFrame::default()).is_err());

    renderer.resize(WIDTH, HEIGHT);
    assert!(
        renderer.render_to_image(&RenderFrame::default()).is_ok(),
        "it recovers when given a size again"
    );
}

#[test]
#[ignore = "requires a GPU adapter and driver"]
fn an_oversized_headless_renderer_is_an_error_not_a_panic() {
    assert!(matches!(
        Renderer::headless(1 << 20, 16, RendererConfig::default()),
        Err(crate::RenderError::UnsupportedSurface(_))
    ));

    let mut renderer = headless();
    renderer.resize(1 << 20, 16);
    assert!(renderer.render_to_image(&RenderFrame::default()).is_err());
}

/// Holds the tonemap shader to its CPU mirror: a frame with no geometry is
/// all clear colour, so every pixel is one known scene value run through
/// exposure, the curve and the sRGB encode. Covers the 18 percent grey
/// acceptance check of E1 among the values.
#[test]
#[ignore = "requires a GPU adapter and driver"]
fn the_tonemap_pass_matches_its_cpu_mirror() {
    use crate::post::tonemap::mirror::display_code;
    use crate::{PostSettings, Tonemapper};

    let mut renderer = headless();
    let scenes: [[f32; 3]; 5] = [
        [0.18, 0.18, 0.18],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [4.0, 4.0, 4.0],
        [2.0, 0.3, 0.05],
    ];
    let warm = crate::Grade {
        lift: [0.05, 0.02, 0.0],
        gamma: [1.2, 1.0, 0.9],
        gain: [1.0, 0.95, 0.8],
    };
    for tonemapper in [Tonemapper::AcesFitted, Tonemapper::Agx] {
        for (exposure_ev, grade) in [
            (0.0, crate::Grade::default()),
            (1.0, warm),
            (-2.0, crate::Grade::default()),
        ] {
            let settings = PostSettings {
                exposure_ev,
                tonemapper,
                grade,
                ..PostSettings::default()
            };
            for scene in scenes {
                let mut frame = RenderFrame::new(
                    IDENTITY,
                    Color::rgb(f64::from(scene[0]), f64::from(scene[1]), f64::from(scene[2])),
                );
                frame.post = settings;
                let image = renderer
                    .render_to_image(&frame)
                    .expect("a headless frame reads back");
                let [r, g, b, _] = image.pixel(WIDTH / 2, HEIGHT / 2).expect("in bounds");
                let expected = display_code(scene, &settings);
                for (got, want) in [r, g, b].into_iter().zip(expected) {
                    assert!(
                        got.abs_diff(want) <= 1,
                        "{tonemapper:?} at {exposure_ev} EV, scene {scene:?}: got {:?}, expected {expected:?}",
                        [r, g, b]
                    );
                }
            }
        }
    }
}

/// E1 auto-exposure: a dim and a bright uniform scene both come out at mid
/// grey once exposure has adapted, here instantly. The bright one follows the
/// dim one, so adaptation is seen going both ways in a running renderer.
#[test]
#[ignore = "requires a GPU adapter and driver"]
fn auto_exposure_brings_any_uniform_scene_to_mid_grey() {
    use crate::post::exposure::KEY;
    use crate::post::tonemap::mirror::display_code;
    use crate::{AutoExposure, PostSettings};

    let mut renderer = headless();
    let auto = AutoExposure {
        brighten_speed: f32::INFINITY,
        darken_speed: f32::INFINITY,
        ..AutoExposure::default()
    };
    let settings = PostSettings {
        exposure_ev: 0.0,
        auto_exposure: Some(auto),
        ..PostSettings::default()
    };
    let manual = PostSettings {
        exposure_ev: 0.0,
        ..PostSettings::default()
    };
    let [expected, ..] = display_code([KEY; 3], &manual);
    for grey in [0.5f64, 8.0, 0.5] {
        let mut frame = RenderFrame::new(IDENTITY, Color::rgb(grey, grey, grey));
        frame.post = settings;
        let image = renderer.render_to_image(&frame).expect("reads back");
        let [r, ..] = image.pixel(WIDTH / 2, HEIGHT / 2).expect("in bounds");
        // One histogram bin is 16/254 of a stop, about 4.5 percent of light.
        assert!(
            r.abs_diff(expected) <= 3,
            "grey {grey} shows as {r}, mid grey is {expected}"
        );
    }
}

/// A swapchain with no sRGB format gets the encode in the shaders instead of
/// in hardware. Both the tonemap pass and the debug lines must do it, exactly
/// once, so the whole reference scene, lines included, must come out the same
/// either way, give or take a rounding step.
#[test]
#[ignore = "requires a GPU adapter and driver"]
fn a_non_srgb_target_is_encoded_by_the_shaders_to_the_same_image() {
    let mut hardware = headless();
    let mut shader = Renderer::headless_with_format(
        WIDTH,
        HEIGHT,
        RendererConfig::default(),
        wgpu::TextureFormat::Rgba8Unorm,
    )
    .expect("these tests are ignored unless a GPU adapter exists");

    let hardware_image = {
        let frame = build(&mut hardware);
        hardware.render_to_image(&frame).expect("reads back")
    };
    let shader_image = {
        let frame = build(&mut shader);
        shader.render_to_image(&frame).expect("reads back")
    };
    if let Err(mismatch) = golden::compare(&hardware_image, &shader_image, 1) {
        panic!("shader encode differs from hardware encode: {mismatch}");
    }
}

/// The reference every later pass change is held to. Re-captured only on
/// purpose, with `UPDATE_GOLDEN=1`, when a change is meant to alter the image.
#[test]
#[ignore = "requires a GPU adapter and driver"]
fn the_reference_scene_matches_its_golden_image() {
    let mut renderer = headless();
    let frame = build(&mut renderer);
    let image = renderer
        .render_to_image(&frame)
        .expect("a headless frame reads back");
    if let Err(message) = golden::check("reference_scene", &image, 0) {
        panic!("{message}");
    }
}

/// PBR acceptance: one camera-facing quad, so the centre pixel has a known
/// normal, view and light direction, and its shade can be predicted.
mod pbr {
    use super::*;
    use crate::pbr::mirror;
    use crate::post::tonemap::mirror::display_code;
    use crate::{ColorSpace, MaterialTextures, Mips, PostSettings, Shading, TextureDesc};

    const PI: f32 = std::f32::consts::PI;

    /// A 4 m quad at the origin facing +Z, with top-down UVs.
    fn quad() -> MeshData {
        MeshData {
            positions: vec![
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            uvs: vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            indices: vec![0, 1, 2, 0, 2, 3],
            ..MeshData::default()
        }
    }

    /// The centre pixel of the quad, lit by `to_light`, with no ambient.
    fn centre(renderer: &mut Renderer, material: &MaterialDesc, to_light: Vec3, exposure_ev: f32) -> [u8; 3] {
        let mesh = renderer.upload_mesh(&quad()).expect("valid");
        let material = renderer.create_material(material);
        let view = look_at(EYE, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let projection = perspective(
            FOV_Y_DEGREES.to_radians(),
            WIDTH as f32 / HEIGHT as f32,
            0.1,
            20.0,
        );
        let mut frame = RenderFrame::new(mul(&projection, &view), Color::BLACK);
        frame.camera_position = EYE;
        frame.light = crate::DirectionalLight {
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            ..crate::DirectionalLight::default()
        }
        .with_direction(to_light);
        frame.ambient.intensity = 0.0;
        frame.post = PostSettings {
            exposure_ev,
            ..PostSettings::default()
        };
        frame.push(DrawCommand {
            mesh,
            material,
            model: IDENTITY,
            normal_matrix: IDENTITY,
        });
        let image = renderer.render_to_image(&frame).expect("reads back");
        let [r, g, b, _] = image.pixel(WIDTH / 2, HEIGHT / 2).expect("in bounds");
        [r, g, b]
    }

    const FOV_Y_DEGREES: f32 = 40.0;
    const EYE: Vec3 = [0.0, 0.0, 3.0];

    /// The direction to the eye from where the centre pixel's ray meets the
    /// quad. The pixel centre sits half a pixel off the axis, and on a smooth
    /// surface lit head on that small tilt already moves the GGX peak by
    /// around ten percent, so the reference must use the real direction.
    fn to_view_at_centre_pixel() -> Vec3 {
        let pixel_x = (WIDTH / 2) as f32 + 0.5;
        let pixel_y = (HEIGHT / 2) as f32 + 0.5;
        let ndc = [
            pixel_x / WIDTH as f32 * 2.0 - 1.0,
            1.0 - pixel_y / HEIGHT as f32 * 2.0,
        ];
        let half_height = (FOV_Y_DEGREES.to_radians() / 2.0).tan();
        let aspect = WIDTH as f32 / HEIGHT as f32;
        // The ray leaves the eye along -Z; it meets z = 0 after EYE.z metres.
        let hit = [
            ndc[0] * half_height * aspect * EYE[2],
            ndc[1] * half_height * EYE[2],
            0.0,
        ];
        normalize(sub(EYE, hit))
    }

    /// Exposure that puts `linear` at mid grey, where a code value is the
    /// finest step of the curve, so the comparison is a tight relative check.
    fn exposure_for(linear: [f32; 3]) -> f32 {
        let brightest = linear.iter().copied().fold(0.0, f32::max).max(1e-6);
        (0.18 / brightest).log2()
    }

    fn assert_close(got: [u8; 3], expected: [u8; 3], what: &str) {
        for (g, e) in got.into_iter().zip(expected) {
            assert!(g.abs_diff(e) <= 2, "{what}: got {got:?}, expected {expected:?}");
        }
    }

    fn centre_of_stacked_quads(
        renderer: &mut Renderer,
        draws: [(&MaterialDesc, [[f32; 4]; 4]); 2],
    ) -> [u8; 3] {
        let mesh = renderer.upload_mesh(&quad()).expect("valid");
        let material_a = renderer.create_material(draws[0].0);
        let material_b = renderer.create_material(draws[1].0);

        let view = look_at(EYE, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let projection = perspective(
            FOV_Y_DEGREES.to_radians(),
            WIDTH as f32 / HEIGHT as f32,
            0.1,
            20.0,
        );
        let mut frame = RenderFrame::new(mul(&projection, &view), Color::BLACK);
        frame.camera_position = EYE;
        frame.light = crate::DirectionalLight {
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            ..crate::DirectionalLight::default()
        }
        .with_direction([0.0, 0.0, 1.0]);
        frame.ambient.intensity = 0.0;
        frame.post = PostSettings {
            exposure_ev: 0.0,
            ..PostSettings::default()
        };

        frame.push(DrawCommand {
            mesh,
            material: material_a,
            model: draws[0].1,
            normal_matrix: IDENTITY,
        });
        frame.push(DrawCommand {
            mesh,
            material: material_b,
            model: draws[1].1,
            normal_matrix: IDENTITY,
        });

        let image = renderer.render_to_image(&frame).expect("reads back");
        let [r, g, b, _] = image.pixel(WIDTH / 2, HEIGHT / 2).expect("in bounds");
        [r, g, b]
    }

    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn a_cut_out_in_front_shows_the_opaque_surface_through_its_holes() {
        let mut renderer = headless();

        let opaque_desc = MaterialDesc {
            base_color: [0.8, 0.0, 0.0, 1.0],
            shading: Shading::Pbr,
            roughness: 0.5,
            alpha: crate::AlphaMode::Opaque,
            ..MaterialDesc::default()
        };

        let mut cut_out_model = IDENTITY;
        cut_out_model[3][2] = 1.0;

        for alpha in [0.0, 1.0] {
            for cut_out_first in [true, false] {
                let cut_out_desc = MaterialDesc {
                    base_color: [0.0, 0.8, 0.0, alpha],
                    shading: Shading::Pbr,
                    roughness: 0.5,
                    alpha: crate::AlphaMode::Mask { cutoff: 0.5 },
                    ..MaterialDesc::default()
                };

                let draws = if cut_out_first {
                    [(&cut_out_desc, cut_out_model), (&opaque_desc, IDENTITY)]
                } else {
                    [(&opaque_desc, IDENTITY), (&cut_out_desc, cut_out_model)]
                };

                let [r, g, b] = centre_of_stacked_quads(&mut renderer, draws);

                if alpha == 0.0 {
                    assert!(
                        r > g + 30 && r > 30,
                        "alpha 0.0, cut_out_first={cut_out_first}: expected red-dominant, got {r}, {g}, {b}"
                    );
                } else {
                    assert!(
                        g > r + 30,
                        "alpha 1.0, cut_out_first={cut_out_first}: expected green-dominant, got {r}, {g}, {b}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn every_permutation_is_compiled_at_startup() {
        assert_eq!(headless().pbr_pipeline_count(), 8);
    }

    /// B4 acceptance: metallic 0 and 1, roughness from smooth to rough, with
    /// the light head on and at 30 degrees, each matching the analytic BRDF.
    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn the_shader_matches_the_analytic_brdf() {
        let mut renderer = headless();
        let albedo = [0.8, 0.5, 0.3];
        let mut failures = Vec::new();
        for to_light in [[0.0, 0.0, 1.0], [0.5, 0.0, 0.866]] {
            let to_light = normalize(to_light);
            for metallic in [0.0, 1.0] {
                for roughness in [0.1, 0.5, 1.0] {
                    let linear = mirror::shade(
                        [0.0, 0.0, 1.0],
                        to_view_at_centre_pixel(),
                        to_light,
                        albedo,
                        metallic,
                        roughness,
                        [PI; 3],
                        [0.0; 3],
                    );
                    let exposure_ev = exposure_for(linear);
                    let settings = PostSettings {
                        exposure_ev,
                        ..PostSettings::default()
                    };
                    let material = MaterialDesc {
                        base_color: [albedo[0], albedo[1], albedo[2], 1.0],
                        shading: Shading::Pbr,
                        metallic,
                        roughness,
                        ..MaterialDesc::default()
                    };
                    let got = centre(&mut renderer, &material, to_light, exposure_ev);
                    let expected = display_code(linear, &settings);
                    if got.iter().zip(expected).any(|(g, e)| g.abs_diff(e) > 2) {
                        failures.push(format!(
                            "metallic {metallic}, roughness {roughness}, light {to_light:?}: got {got:?}, expected {expected:?}"
                        ));
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{}",
            failures.join(
                "
"
            )
        );
    }

    /// A normal map tilting every normal towards +X must shade exactly like
    /// geometry tilted that way, whichever colour space the map was uploaded
    /// in, because the normal slot always reads linear.
    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn a_normal_map_tilts_the_shading_and_is_always_read_linear() {
        let mut renderer = headless();
        // Encoded tilt mostly up the image: green well away from 128, so a
        // bitangent pointing the wrong way cannot pass.
        let texel = [140u8, 200, 210, 255];
        let decoded = [
            140.0 / 255.0 * 2.0 - 1.0,
            200.0 / 255.0 * 2.0 - 1.0,
            210.0 / 255.0 * 2.0 - 1.0,
        ];
        // The quad's tangent is +X and its bitangent, up the image, +Y.
        let normal = normalize(decoded);
        // Light from above, so tilting up brightens and tilting down darkens.
        let to_light = normalize([0.2, 0.6, 0.8]);
        let linear = mirror::shade(
            normal,
            to_view_at_centre_pixel(),
            to_light,
            [0.7; 3],
            0.0,
            1.0,
            [PI; 3],
            [0.0; 3],
        );
        let exposure_ev = exposure_for(linear);
        let settings = PostSettings {
            exposure_ev,
            ..PostSettings::default()
        };
        let flat = centre(
            &mut renderer,
            &MaterialDesc {
                base_color: [0.7, 0.7, 0.7, 1.0],
                shading: Shading::Pbr,
                roughness: 1.0,
                ..MaterialDesc::default()
            },
            to_light,
            exposure_ev,
        );

        for color_space in [ColorSpace::Linear, ColorSpace::Srgb] {
            let map = renderer
                .create_texture(
                    &TextureDesc {
                        width: 2,
                        height: 2,
                        color_space,
                        mips: Mips::Single,
                    },
                    &texel.repeat(4),
                )
                .expect("valid");
            let material = MaterialDesc {
                base_color: [0.7, 0.7, 0.7, 1.0],
                shading: Shading::Pbr,
                roughness: 1.0,
                textures: MaterialTextures {
                    normal: Some(map),
                    ..MaterialTextures::default()
                },
                ..MaterialDesc::default()
            };
            let got = centre(&mut renderer, &material, to_light, exposure_ev);
            assert_close(
                got,
                display_code(linear, &settings),
                &format!("normal map uploaded {color_space:?}"),
            );
            assert_ne!(got, flat, "the map visibly changes the shading");
        }
    }
}

/// Reads the shadow atlas back as depth values, row by row.
fn read_shadow_atlas(renderer: &Renderer) -> (u32, u32, Vec<f32>) {
    let atlas = &renderer.shadows.atlas;
    let (width, height) = atlas.layout.size();
    let bytes_per_row =
        (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shadow-atlas-readback"),
        size: u64::from(bytes_per_row) * u64::from(height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = renderer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &atlas.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::DepthOnly,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    renderer.queue.submit(std::iter::once(encoder.finish()));
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    renderer
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device finishes");
    receiver.recv().expect("map callback").expect("buffer maps");
    let mapped = buffer.get_mapped_range(..).expect("mapped");
    let depths = mapped
        .chunks_exact(bytes_per_row as usize)
        .flat_map(|row| {
            row[..width as usize * 4]
                .chunks_exact(4)
                .map(|texel| f32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]]))
        })
        .collect();
    (width, height, depths)
}

#[test]
#[ignore = "requires a GPU adapter and driver"]
fn the_shadow_pass_draws_the_scene_into_every_cascade_before_the_prepass() {
    let mut renderer = headless();
    let frame = build(&mut renderer);
    renderer.render_to_image(&frame).expect("renders");

    let dump = renderer.graph_dump();
    let shadow = dump.find("shadow").expect("the shadow pass is in the graph");
    let prepass = dump.find("depth-prepass").expect("the prepass is in the graph");
    assert!(shadow < prepass, "shadows are drawn before the prepass:\n{dump}");

    let cascades = renderer.shadows.cascades.len() as u32;
    assert_eq!(cascades, crate::ShadowSettings::default().cascades);
    let (width, height, depths) = read_shadow_atlas(&renderer);
    let resolution = width / cascades;
    for cascade in 0..cascades {
        let covered = (0..height)
            .flat_map(|y| (0..resolution).map(move |x| (y * width + cascade * resolution + x) as usize))
            .filter(|&index| depths[index] < 1.0)
            .count();
        assert!(covered > 0, "cascade {cascade} holds geometry depth");
    }
}

#[test]
#[ignore = "requires a GPU adapter and driver"]
fn shadows_off_records_no_shadow_pass() {
    let mut renderer = headless();
    let mut frame = build(&mut renderer);
    frame.light.shadow.cascades = 0;
    renderer.render_to_image(&frame).expect("renders");
    assert!(renderer.shadows.cascades.is_empty());
    assert!(
        !renderer
            .graph_dump()
            .lines()
            .any(|line| line.contains(": shadow")),
        "no shadow pass:\n{}",
        renderer.graph_dump()
    );
    assert_eq!(renderer.shadows.atlas.layout.size(), (1, 1));
}

#[test]
#[ignore = "requires a GPU adapter and driver"]
fn gpu_pass_timings_arrive_a_few_frames_later_named_after_the_graph() {
    let mut renderer = headless();
    let frame = build(&mut renderer);

    let supports_timestamps = renderer
        .device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);

    for _ in 0..8 {
        renderer.render(&frame).expect("renders");
    }

    if !supports_timestamps {
        assert!(renderer.gpu_timings().is_none());
        return;
    }

    let timings = renderer.gpu_timings().expect("timings arrive a few frames late");
    let names: Vec<&str> = timings.iter().map(|timing| timing.name).collect();

    // The timings follow the graph's pass order exactly: "  0 shadow: ...".
    let graph_dump = renderer.graph_dump();
    let graph_passes: Vec<&str> = graph_dump
        .lines()
        .take_while(|line| !line.starts_with("resources"))
        .filter_map(|line| {
            line.trim_start()
                .split_once(' ')?
                .1
                .split_once(':')
                .map(|(name, _)| name)
        })
        .collect();
    assert_eq!(names, graph_passes, "graph:\n{graph_dump}");
    assert_eq!(&names[..3], ["shadow", "depth-prepass", "lit"]);

    assert!(
        timings
            .iter()
            .any(|timing| matches!(timing.gpu_ms, Some(ms) if ms > 0.0 && ms.is_finite())),
        "expected at least one positive finite GPU timing"
    );
}

/// E1 bloom acceptance: light above the threshold bleeds past its edges,
/// light below it does not, and more light gives a brighter halo.
mod bloom {
    use super::*;
    use crate::{Bloom, Image, PostSettings};

    /// Pixels right of the cube's edge: the cube spans about 26 pixels
    /// around the centre, so 30 to the right is clear of it.
    const HALO: (u32, u32) = (WIDTH / 2 + 30, HEIGHT / 2);

    /// A painted cube facing the camera on black, its material scaled by
    /// `brightness`, drawn with `bloom`.
    fn glowing_cube(renderer: &mut Renderer, brightness: f32, light_intensity: f32, bloom: Bloom) -> Image {
        let mesh = renderer.upload_mesh(&painted_cube()).expect("valid");
        let material = renderer.create_material(&MaterialDesc {
            base_color: [brightness, brightness, brightness, 1.0],
            ..MaterialDesc::default()
        });
        let view = look_at([0.0, 0.5, 4.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]);
        let projection = perspective(60f32.to_radians(), WIDTH as f32 / HEIGHT as f32, 0.1, 50.0);
        let mut frame = RenderFrame::new(mul(&projection, &view), Color::BLACK);
        frame.light.intensity = light_intensity;
        frame.post = PostSettings {
            bloom,
            ..PostSettings::default()
        };
        frame.push(DrawCommand {
            mesh,
            material,
            model: IDENTITY,
            normal_matrix: IDENTITY,
        });
        renderer.render_to_image(&frame).expect("reads back")
    }

    fn brightness_at(image: &Image, (x, y): (u32, u32)) -> u32 {
        let [r, g, b, _] = image.pixel(x, y).expect("in bounds");
        u32::from(r) + u32::from(g) + u32::from(b)
    }

    fn on() -> Bloom {
        Bloom {
            intensity: 0.3,
            threshold: 1.0,
            knee: 0.5,
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn bright_light_bleeds_past_its_edges() {
        let mut renderer = headless();
        let without = glowing_cube(&mut renderer, 8.0, 1.0, Bloom::default());
        let with = glowing_cube(&mut renderer, 8.0, 1.0, on());
        assert_eq!(
            brightness_at(&without, HALO),
            0,
            "nothing is drawn there without bloom"
        );
        assert!(
            brightness_at(&with, HALO) > 0,
            "bloom lights the black around a bright cube"
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn light_below_the_threshold_does_not_bloom() {
        let mut renderer = headless();
        let without = glowing_cube(&mut renderer, 0.3, 1.0, Bloom::default());
        let with = glowing_cube(&mut renderer, 0.3, 1.0, on());
        if let Err(mismatch) = golden::compare(&without, &with, 1) {
            panic!("a dim scene changed under bloom: {mismatch}");
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn four_times_the_light_gives_a_brighter_halo() {
        let mut renderer = headless();
        let dim = glowing_cube(&mut renderer, 8.0, 1.0, on());
        let bright = glowing_cube(&mut renderer, 8.0, 4.0, on());
        let (dim_halo, bright_halo) = (brightness_at(&dim, HALO), brightness_at(&bright, HALO));
        assert!(
            bright_halo > dim_halo,
            "halo {bright_halo} at 4x light, {dim_halo} at 1x"
        );
        assert!(bright_halo < 3 * 255, "and the halo is not clipped to white");
    }
}

/// The M17 showcase: procedural textures on a tiled ground, a row of spheres
/// stepping through metallic and roughness, auto-exposure and bloom on. Built
/// from nothing but this crate, so it needs no assets and no decoder.
mod showcase {
    use super::*;
    use crate::{
        AutoExposure, Bloom, ColorSpace, MaterialTextures, Mips, PostSettings, SamplerDesc, Shading,
        TextureDesc, TextureHandle, Wrap,
    };

    const TEXTURE_SIZE: u32 = 256;

    /// A UV sphere of radius 0.5 resting on the ground, smooth normals, u around
    /// and v top to bottom.
    fn sphere(segments: u32, rings: u32) -> MeshData {
        let mut mesh = MeshData::default();
        for ring in 0..=rings {
            let v = ring as f32 / rings as f32;
            let polar = v * std::f32::consts::PI;
            for segment in 0..=segments {
                let u = segment as f32 / segments as f32;
                let azimuth = u * std::f32::consts::TAU;
                let normal = [
                    polar.sin() * azimuth.cos(),
                    polar.cos(),
                    -polar.sin() * azimuth.sin(),
                ];
                mesh.positions
                    .push([normal[0] * 0.5, normal[1] * 0.5 + 0.5, normal[2] * 0.5]);
                mesh.normals.push(normal);
                mesh.uvs.push([u, v]);
            }
        }
        let stride = segments + 1;
        for ring in 0..rings {
            for segment in 0..segments {
                let a = ring * stride + segment;
                let b = a + stride;
                mesh.indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        mesh
    }

    /// The ground with its UVs repeated `tiles` times.
    fn tiled_ground(half: f32, tiles: f32) -> MeshData {
        let mut mesh = ground(half);
        for uv in &mut mesh.uvs {
            *uv = [uv[0] * tiles, uv[1] * tiles];
        }
        mesh
    }

    /// RGBA8 texels from a function of (x, y) in 0..1.
    fn texels(paint: impl Fn(f32, f32) -> [u8; 4]) -> Vec<u8> {
        let size = TEXTURE_SIZE as f32;
        (0..TEXTURE_SIZE * TEXTURE_SIZE)
            .flat_map(|index| {
                let x = (index % TEXTURE_SIZE) as f32 / size;
                let y = (index / TEXTURE_SIZE) as f32 / size;
                paint(x, y)
            })
            .collect()
    }

    fn texture(
        renderer: &mut Renderer,
        color_space: ColorSpace,
        paint: impl Fn(f32, f32) -> [u8; 4],
    ) -> TextureHandle {
        renderer
            .create_texture(
                &TextureDesc {
                    width: TEXTURE_SIZE,
                    height: TEXTURE_SIZE,
                    color_space,
                    mips: Mips::Generate,
                },
                &texels(paint),
            )
            .expect("valid")
    }

    fn grout(x: f32, y: f32) -> bool {
        (x * 4.0).fract() < 0.04 || (y * 4.0).fract() < 0.04
    }

    fn ground_material(renderer: &mut Renderer) -> crate::MaterialHandle {
        // Stone tiles: two warm greys in a checker, with dark grout.
        let albedo = texture(renderer, ColorSpace::Srgb, |x, y| {
            let checker = ((x * 4.0) as u32 + (y * 4.0) as u32).is_multiple_of(2);
            match (grout(x, y), checker) {
                (true, _) => [40, 38, 35, 255],
                (false, true) => [168, 156, 140, 255],
                (false, false) => [128, 120, 110, 255],
            }
        });
        // Each tile domed, sloping down to the grout: a +Y-up tangent-space map.
        let normal = texture(renderer, ColorSpace::Linear, |x, y| {
            let slope = |t: f32| ((t * 4.0).fract() - 0.5) * 0.8;
            let (nx, ny) = (slope(x), -slope(y));
            let nz = (1.0 - nx * nx - ny * ny).max(0.0).sqrt();
            let encode = |c: f32| ((c * 0.5 + 0.5) * 255.0).round() as u8;
            [encode(nx), encode(ny), encode(nz), 255]
        });
        // Occlusion in the grout, roughness rising across the texture.
        let orm = texture(renderer, ColorSpace::Linear, |x, y| {
            let occlusion = if grout(x, y) { 90 } else { 255 };
            [occlusion, (60.0 + x * 195.0) as u8, 0, 255]
        });
        renderer.create_material(&MaterialDesc {
            shading: Shading::Pbr,
            textures: MaterialTextures {
                base_color: Some(albedo),
                normal: Some(normal),
                orm: Some(orm),
                emissive: None,
            },
            sampler: SamplerDesc {
                wrap: Wrap::Repeat,
                anisotropy: 8,
                ..SamplerDesc::default()
            },
            roughness: 1.0,
            ..MaterialDesc::default()
        })
    }

    pub(super) fn build(renderer: &mut Renderer, width: u32, height: u32) -> RenderFrame {
        let ground_mesh = renderer.upload_mesh(&tiled_ground(8.0, 4.0)).expect("valid");
        let ground_material = ground_material(renderer);

        let eye = [0.0, 2.2, 6.0];
        let view = look_at(eye, [0.0, 0.6, 0.0], [0.0, 1.0, 0.0]);
        let projection = perspective(50f32.to_radians(), width as f32 / height as f32, 0.1, 100.0);
        let mut frame = RenderFrame::new(mul(&projection, &view), Color::rgb(0.25, 0.35, 0.55));
        frame.camera_position = eye;
        frame.light = crate::DirectionalLight {
            color: [1.0, 0.95, 0.85],
            intensity: 3.0,
            ..crate::DirectionalLight::default()
        }
        .with_direction([0.5, 0.7, 0.4]);
        frame.ambient.intensity = 0.4;
        frame.post = PostSettings {
            exposure_ev: 0.0,
            auto_exposure: Some(AutoExposure {
                brighten_speed: f32::INFINITY,
                darken_speed: f32::INFINITY,
                ..AutoExposure::default()
            }),
            bloom: Bloom {
                intensity: 0.05,
                ..Bloom::default()
            },
            ..PostSettings::default()
        };
        frame.push(DrawCommand {
            mesh: ground_mesh,
            material: ground_material,
            model: IDENTITY,
            normal_matrix: IDENTITY,
        });

        // Two rows of five: dielectric in front, metal behind, smooth to rough.
        let sphere_mesh = renderer.upload_mesh(&sphere(48, 24)).expect("valid");
        for (row, metallic) in [(0, 0.0), (1, 1.0)] {
            for column in 0..5 {
                let roughness = 0.1 + column as f32 * 0.225;
                let base_color = if metallic > 0.5 {
                    [0.95, 0.75, 0.35, 1.0]
                } else {
                    [0.7, 0.15, 0.12, 1.0]
                };
                let material = renderer.create_material(&MaterialDesc {
                    base_color,
                    shading: Shading::Pbr,
                    metallic,
                    roughness,
                    ..MaterialDesc::default()
                });
                let (model, normal_matrix) =
                    turned_at(0.0, [column as f32 * 1.3 - 2.6, 0.0, -(row as f32) * 1.4]);
                frame.push(DrawCommand {
                    mesh: sphere_mesh,
                    material,
                    model,
                    normal_matrix,
                });
            }
        }
        frame
    }

    #[test]
    fn the_sphere_is_a_valid_closed_mesh() {
        let sphere = sphere(16, 8);
        assert_eq!(sphere.validate(), Ok(()));
        let bounds = sphere.bounds().expect("not empty");
        assert!((bounds.min[1] - 0.0).abs() < 1e-6 && (bounds.max[1] - 1.0).abs() < 1e-6);
    }

    /// Renders the showcase for a person to look at. Set SHOWCASE_TO to a path
    /// to keep the image (raw RGBA behind the golden header).
    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn the_showcase_renders() {
        let (width, height) = (640, 360);
        let mut renderer = Renderer::headless(width, height, RendererConfig::default())
            .expect("these tests are ignored unless a GPU adapter exists");
        let frame = build(&mut renderer, width, height);
        let image = renderer.render_to_image(&frame).expect("reads back");
        let lit = image
            .rgba8
            .chunks_exact(4)
            .filter(|pixel| pixel[..3].iter().any(|&c| c > 20))
            .count();
        assert!(lit > image.rgba8.len() / 8, "most of the frame is lit, not black");
        if let Some(path) = std::env::var_os("SHOWCASE_TO") {
            std::fs::write(path, golden::encode(&image)).expect("writable");
        }
    }

    /// The M17 frame-rate check: the showcase at 1440p with every phase 1
    /// feature on, timed over many frames with the GPU's work included. 60 fps
    /// means 16.6 ms; the High-tier target GPU should be far inside it.
    #[test]
    #[ignore = "requires a GPU adapter and driver; measures performance"]
    fn the_showcase_holds_60_fps_at_1440p() {
        const FRAMES: u32 = 240;
        let (width, height) = (2560, 1440);
        let mut renderer = Renderer::headless(width, height, RendererConfig::default())
            .expect("these tests are ignored unless a GPU adapter exists");
        let frame = build(&mut renderer, width, height);
        // Warm up: pipelines, pool, first exposure.
        renderer.render_to_image(&frame).expect("reads back");

        let started = std::time::Instant::now();
        for _ in 0..FRAMES {
            renderer.render(&frame).expect("renders");
        }
        // Reading back waits for every queued frame to finish on the GPU.
        renderer.render_to_image(&frame).expect("reads back");
        let per_frame = started.elapsed() / (FRAMES + 1);
        println!(
            "showcase at {width}x{height} on {}: {:.3} ms per frame",
            renderer.adapter_info().name,
            per_frame.as_secs_f64() * 1000.0
        );
        assert!(
            per_frame < std::time::Duration::from_micros(16_667),
            "{per_frame:?} per frame is under 60 fps"
        );
    }
}

/// C1 acceptance: cascaded shadows land where the light geometry says, the
/// open ground shows no acne or cascade seams, and cut-outs cast through
/// their alpha.
mod shadows {
    use super::*;
    use crate::{AlphaMode, Shading};

    /// The pixel a world point lands on, from the frame's view-projection.
    fn pixel_of(view_proj: &Mat4, point: Vec3) -> (u32, u32) {
        let clip: [f32; 4] = std::array::from_fn(|row| {
            (0..4)
                .map(|column| view_proj[column][row] * if column == 3 { 1.0 } else { point[column] })
                .sum()
        });
        let (x, y) = (clip[0] / clip[3], clip[1] / clip[3]);
        (
            ((x * 0.5 + 0.5) * WIDTH as f32) as u32,
            ((0.5 - y * 0.5) * HEIGHT as f32) as u32,
        )
    }

    fn luminance(image: &crate::Image, (x, y): (u32, u32)) -> i32 {
        let [r, g, b, _] = image.pixel(x, y).expect("in bounds");
        i32::from(r) + i32::from(g) + i32::from(b)
    }

    /// A grey ground under a light from -x at 45 degrees, seen from above.
    fn sunlit_ground(renderer: &mut Renderer, half: f32) -> RenderFrame {
        let ground_mesh = renderer.upload_mesh(&ground(half)).expect("valid");
        let grey = renderer.create_material(&MaterialDesc {
            base_color: [0.5, 0.5, 0.5, 1.0],
            ..MaterialDesc::default()
        });
        let view = look_at([1.5, 9.0, 3.0], [1.5, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let projection = perspective(50f32.to_radians(), WIDTH as f32 / HEIGHT as f32, 0.1, 60.0);
        let mut frame = RenderFrame::new(mul(&projection, &view), Color::BLACK);
        frame.light = frame.light.with_direction([-1.0, 1.0, 0.0]);
        frame.push(DrawCommand {
            mesh: ground_mesh,
            material: grey,
            model: IDENTITY,
            normal_matrix: IDENTITY,
        });
        frame
    }

    /// A 2 m tall box over x and z in -0.5..0.5 casts a shadow from x = 0.5
    /// to x = 2.5 under a 45 degree light from -x.
    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn a_box_casts_its_shadow_where_the_light_geometry_puts_it() {
        let mut renderer = headless();
        let mut frame = sunlit_ground(&mut renderer, 10.0);
        let cube = renderer.upload_mesh(&painted_cube()).expect("valid");
        let plain = renderer.create_material(&MaterialDesc::default());
        let mut tall = IDENTITY;
        tall[1][1] = 2.0;
        frame.push(DrawCommand {
            mesh: cube,
            material: plain,
            model: tall,
            normal_matrix: IDENTITY,
        });
        let image = renderer.render_to_image(&frame).expect("renders");

        let at = |x: f32| luminance(&image, pixel_of(&frame.view_proj, [x, 0.0, 0.0]));
        let shadowed = [at(1.0), at(1.8), at(2.3)];
        let lit = [at(2.7), at(3.5), at(5.0)];
        for value in shadowed {
            assert!((value - shadowed[0]).abs() <= 3, "uniform shadow: {shadowed:?}");
        }
        for value in lit {
            assert!((value - lit[0]).abs() <= 3, "uniform light: {lit:?}");
        }
        assert!(
            lit[0] > shadowed[0] + 60,
            "0.2 m inside the edge is shadowed and 0.2 m outside is lit: {shadowed:?} against {lit:?}"
        );
        let behind = luminance(&image, pixel_of(&frame.view_proj, [-2.0, 0.0, 0.0]));
        assert!((behind - lit[0]).abs() <= 3, "the light side of the box is lit");
    }

    /// Every ground pixel has the same normal and light, so any spread in
    /// their colours is acne or a seam between cascades.
    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn open_ground_has_no_acne_or_cascade_seams_out_to_the_horizon() {
        let mut renderer = headless();
        let ground_mesh = renderer.upload_mesh(&ground(80.0)).expect("valid");
        let grey = renderer.create_material(&MaterialDesc {
            base_color: [0.5, 0.5, 0.5, 1.0],
            ..MaterialDesc::default()
        });
        // Low over the ground and looking along it, so the view crosses every
        // cascade; a grazing light is where acne shows first.
        let view = look_at([0.0, 1.7, 0.0], [0.0, 0.0, -20.0], [0.0, 1.0, 0.0]);
        let projection = perspective(60f32.to_radians(), WIDTH as f32 / HEIGHT as f32, 0.1, 100.0);
        let mut frame = RenderFrame::new(mul(&projection, &view), Color::BLACK);
        frame.light = frame.light.with_direction([0.6, 0.25, -0.3]);
        frame.push(DrawCommand {
            mesh: ground_mesh,
            material: grey,
            model: IDENTITY,
            normal_matrix: IDENTITY,
        });
        let image = renderer.render_to_image(&frame).expect("renders");

        let ground_pixels: Vec<i32> = (0..HEIGHT)
            .flat_map(|y| (0..WIDTH).map(move |x| (x, y)))
            .map(|pixel| luminance(&image, pixel))
            .filter(|&value| value > 0)
            .collect();
        assert!(
            ground_pixels.len() > (WIDTH * HEIGHT / 3) as usize,
            "the ground fills the view"
        );
        let (low, high) = ground_pixels
            .iter()
            .fold((i32::MAX, i32::MIN), |(low, high), &value| {
                (low.min(value), high.max(value))
            });
        assert!(high - low <= 3, "ground luminance spans {low}..={high}");
    }

    /// A cut-out board hovering over the ground: fully cut away it casts
    /// nothing, solid it casts its square. The sampled ground is not behind
    /// the board from the camera, so only the shadow can darken it.
    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn a_cut_out_casts_a_shadow_only_where_it_is_solid() {
        let mut values = Vec::new();
        for alpha in [0.0, 1.0] {
            let mut renderer = headless();
            let mut frame = sunlit_ground(&mut renderer, 10.0);
            let board = renderer.upload_mesh(&ground(0.75)).expect("valid");
            let leaf = renderer.create_material(&MaterialDesc {
                base_color: [0.2, 0.6, 0.2, alpha],
                shading: Shading::Pbr,
                alpha: AlphaMode::Mask { cutoff: 0.5 },
                ..MaterialDesc::default()
            });
            // Raised 1 m over (1, -3): the 45 degree light from -x puts its
            // shadow 1 m along +x, around (2, 0, -3), which the camera sees
            // past the board's edge rather than through it.
            let mut raised = IDENTITY;
            raised[3] = [1.0, 1.0, -3.0, 1.0];
            frame.push(DrawCommand {
                mesh: board,
                material: leaf,
                model: raised,
                normal_matrix: IDENTITY,
            });
            let image = renderer.render_to_image(&frame).expect("renders");
            values.push(luminance(&image, pixel_of(&frame.view_proj, [2.0, 0.0, -3.0])));
        }
        assert!(
            values[0] > values[1] + 60,
            "a cut-away board casts no shadow, a solid one does: {values:?}"
        );
    }
}
