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
    for tonemapper in [Tonemapper::AcesFitted, Tonemapper::Agx] {
        for exposure_ev in [0.0, 1.0, -2.0] {
            let settings = PostSettings {
                exposure_ev,
                tonemapper,
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
        // Encoded tilt: x = 191/255 * 2 - 1, z = 218/255 * 2 - 1.
        let texel = [191u8, 128, 218, 255];
        let decoded = [
            191.0 / 255.0 * 2.0 - 1.0,
            128.0 / 255.0 * 2.0 - 1.0,
            218.0 / 255.0 * 2.0 - 1.0,
        ];
        // The quad's tangent is +X and, with top-down UVs, its bitangent -Y.
        let normal = normalize([decoded[0], -decoded[1], decoded[2]]);
        let to_light = normalize([0.6, 0.0, 0.8]);
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
