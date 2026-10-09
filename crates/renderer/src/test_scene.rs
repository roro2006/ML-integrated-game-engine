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
        colors: Vec::new(),
        indices: vec![0, 1, 2, 0, 2, 3],
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
fn the_clear_colour_reads_back_display_encoded() {
    let mut renderer = headless();
    // Linear 0.214 encodes to sRGB 0.4999, code 127 or 128 depending on rounding.
    let frame = RenderFrame::new(IDENTITY, Color::rgb(0.214, 0.0, 1.0));
    let image = renderer
        .render_to_image(&frame)
        .expect("a headless frame reads back");

    assert_eq!((image.width, image.height), (WIDTH, HEIGHT));
    assert_eq!(image.rgba8.len(), (WIDTH * HEIGHT * 4) as usize);
    for (x, y) in [(0, 0), (WIDTH - 1, 0), (0, HEIGHT - 1), (WIDTH - 1, HEIGHT - 1)] {
        let [r, g, b, a] = image.pixel(x, y).expect("in bounds");
        assert!((127..=128).contains(&r), "red at ({x}, {y}) was {r}");
        assert_eq!((g, b, a), (0, 255, 255), "at ({x}, {y})");
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
