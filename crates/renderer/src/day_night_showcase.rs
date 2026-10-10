//! The M18 exit showcase: one small yard lit at noon, at dusk and at night,
//! with every phase 2 pass on. Cascaded and contact shadows, the sky and its
//! environment light, aerial perspective and height fog, ambient occlusion,
//! and at night 64 clustered point lights.
//!
//! The three times are this crate's golden set for phase 2 (G3's stand-in,
//! see `golden`), and the 1440p run is the phase's frame-time check, whose
//! numbers ADR-0025 records as the per-pass budgets.
//!
//! The sun and moon stand where the engine's default time of day puts them
//! (`engine::time_of_day`, 45 degrees north, declination 10 degrees) at
//! 12:00, about 19:20 and 00:00. The renderer sits below the engine, so the
//! directions are written out here rather than computed.

use glam::{Mat4, Vec3};

use crate::ao_tests::{cuboid, pixel_of};
use crate::test_scene::showcase::sphere;
use crate::{
    AlphaMode, AmbientOcclusion, AutoExposure, Bloom, Color, ColorSpace, ContactShadows, DirectionalLight,
    DrawCommand, HeightFog, Image, LocalLight, MaterialDesc, MaterialTextures, MeshData, Mips, PostSettings,
    RenderFrame, Renderer, RendererConfig, Shading, Sky, TextureDesc, golden, sunlight_transmittance,
};

/// The golden images' size: small enough to keep in the repository, not a
/// multiple of 64 so readback rows carry padding.
const GOLDEN_WIDTH: u32 = 320;
const GOLDEN_HEIGHT: u32 = 180;

/// The camera, before the yard is turned (see [`yard`]).
const EYE: Vec3 = Vec3::new(0.0, 1.8, 6.5);
const TARGET: Vec3 = Vec3::new(0.0, 0.5, 0.0);
/// Half the ground's width, and the camera's far plane beyond it: far enough
/// that fog and the air carry the ground to the horizon. The sky is black
/// below the horizon, so a nearer edge would show as a dark line.
const GROUND_HALF: f32 = 1500.0;
const FAR_PLANE: f32 = 2500.0;

fn camera(eye: Vec3, target: Vec3, aspect: f32) -> [[f32; 4]; 4] {
    let view = glam::camera::rh::view::look_at_mat4(eye, target, Vec3::Y);
    let projection =
        glam::camera::rh::proj::directx::perspective(50_f32.to_radians(), aspect, 0.1, FAR_PLANE);
    (projection * view).to_cols_array_2d()
}
/// The engine's ambient at full daylight, and the fraction night leaves.
const DAY_AMBIENT: f32 = 0.4;
const NIGHT_AMBIENT: f32 = 0.15;
/// The lamps' grid: 8 by 8 over the yard, at knee height.
const LAMPS_PER_SIDE: u32 = 8;
const LAMP_SPACING: f32 = 1.1;
const LAMP_HEIGHT: f32 = 0.4;

/// A moment of the day: where the sun is, and what lights the yard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Moment {
    Noon,
    Dusk,
    Night,
}

impl Moment {
    pub(crate) const ALL: [Self; 3] = [Self::Noon, Self::Dusk, Self::Night];

    fn name(self) -> &'static str {
        match self {
            Self::Noon => "noon",
            Self::Dusk => "dusk",
            Self::Night => "night",
        }
    }

    /// Unit direction to the sun.
    fn sun(self) -> Vec3 {
        match self {
            // 55 degrees up, due south (+z).
            Self::Noon => Vec3::new(0.0, 0.819, 0.574),
            // Four degrees up, a little north of west.
            Self::Dusk => Vec3::new(-0.96, 0.07, -0.27).normalize(),
            // Well below the northern horizon.
            Self::Night => Vec3::new(0.0, -0.643, -0.766),
        }
    }

    /// The key light: the sun, reddened by the air it crosses, or by night
    /// the cool, dim moon opposite it. Intensities follow the engine's
    /// twilight hand-over at these elevations.
    fn key(self) -> DirectionalLight {
        let sun = self.sun();
        let (color, intensity, direction) = match self {
            Self::Noon => (sunlight_transmittance(EYE.y, sun.to_array()), 1.0, sun),
            Self::Dusk => (sunlight_transmittance(EYE.y, sun.to_array()), 0.88, sun),
            Self::Night => ([0.62, 0.72, 1.0], 0.06, -sun),
        };
        DirectionalLight {
            color,
            intensity,
            ..DirectionalLight::default()
        }
        .with_direction(direction.to_array())
    }

    fn ambient(self) -> f32 {
        match self {
            Self::Noon => DAY_AMBIENT,
            Self::Dusk => DAY_AMBIENT * 0.94,
            Self::Night => DAY_AMBIENT * NIGHT_AMBIENT,
        }
    }
}

/// The yard is laid out facing +Z and then turned a quarter turn, so the
/// camera looks west: the noon sun (south) stands to its left and throws
/// shadows across the view, and the dusk sun stands ahead of it.
fn yard() -> Mat4 {
    Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2)
}

/// A point of the laid-out yard, where it stands once turned.
fn in_yard(point: Vec3) -> Vec3 {
    yard().transform_point3(point)
}

/// RGBA8 texels of a `size`-square texture from a function of (x, y) in 0..1.
fn texels(size: u32, paint: impl Fn(f32, f32) -> [u8; 4]) -> Vec<u8> {
    (0..size * size)
        .flat_map(|index| {
            paint(
                (index % size) as f32 / size as f32,
                (index / size) as f32 / size as f32,
            )
        })
        .collect()
}

/// A fan of leaves: green where a leaf is, transparent between them.
fn leaf_material(renderer: &mut Renderer) -> crate::MaterialHandle {
    const SIZE: u32 = 128;
    let leaves = texels(SIZE, |x, y| {
        // Five leaves rising from the bottom centre, each a narrow ellipse.
        let (dx, dy) = (x - 0.5, 1.0 - y);
        let inside = (0..5).any(|leaf| {
            let angle = (leaf as f32 - 2.0) * 0.32;
            let (sin, cos) = angle.sin_cos();
            let along = dx * sin + dy * cos;
            let across = dx * cos - dy * sin;
            let half_length = 0.45;
            along > 0.0 && (along / half_length - 1.0).powi(2) + (across / 0.07).powi(2) < 1.0
        });
        if inside {
            [70, (120.0 + 60.0 * y) as u8, 50, 255]
        } else {
            [70, 120, 50, 0]
        }
    });
    let base_color = renderer
        .create_texture(
            &TextureDesc {
                width: SIZE,
                height: SIZE,
                color_space: ColorSpace::Srgb,
                mips: Mips::Generate,
            },
            &leaves,
        )
        .expect("valid");
    renderer.create_material(&MaterialDesc {
        shading: Shading::Pbr,
        textures: MaterialTextures {
            base_color: Some(base_color),
            ..MaterialTextures::default()
        },
        roughness: 0.7,
        alpha: AlphaMode::Mask { cutoff: 0.5 },
        ..MaterialDesc::default()
    })
}

/// A vertical quad, `width` by `height`, standing on the origin and facing +Z.
fn panel(width: f32, height: f32) -> MeshData {
    let half = width / 2.0;
    MeshData {
        positions: vec![
            [-half, 0.0, 0.0],
            [half, 0.0, 0.0],
            [half, height, 0.0],
            [-half, height, 0.0],
        ],
        normals: vec![[0.0, 0.0, 1.0]; 4],
        uvs: vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        indices: vec![0, 1, 2, 0, 2, 3],
        ..MeshData::default()
    }
}

fn place(frame: &mut RenderFrame, mesh: crate::MeshHandle, material: crate::MaterialHandle, model: Mat4) {
    let model = yard() * model;
    let normal_matrix = model.inverse().transpose();
    frame.push(DrawCommand {
        mesh,
        material,
        model: model.to_cols_array_2d(),
        normal_matrix: normal_matrix.to_cols_array_2d(),
    });
}

fn pbr(
    renderer: &mut Renderer,
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
) -> crate::MaterialHandle {
    renderer.create_material(&MaterialDesc {
        shading: Shading::Pbr,
        base_color,
        metallic,
        roughness,
        ..MaterialDesc::default()
    })
}

/// The 64 lamps, warm and a little varied, lighting the yard at night.
fn lamps() -> Vec<LocalLight> {
    let offset = (LAMPS_PER_SIDE - 1) as f32 * LAMP_SPACING / 2.0;
    (0..LAMPS_PER_SIDE * LAMPS_PER_SIDE)
        .map(|index| {
            let (column, row) = (index % LAMPS_PER_SIDE, index / LAMPS_PER_SIDE);
            let warmth = (index % 3) as f32 * 0.1;
            LocalLight {
                position: in_yard(Vec3::new(
                    column as f32 * LAMP_SPACING - offset,
                    LAMP_HEIGHT,
                    row as f32 * LAMP_SPACING - offset,
                ))
                .to_array(),
                range: 2.0,
                color: [1.0, 0.7 + warmth, 0.4 + warmth],
                intensity: 0.12,
                spot: None,
            }
        })
        .collect()
}

/// The ground under a lamp in the front row, and a point midway between
/// lamps, for checking that the night is lit in pools.
fn lamp_pool_and_gap() -> (Vec3, Vec3) {
    let offset = (LAMPS_PER_SIDE - 1) as f32 * LAMP_SPACING / 2.0;
    let x = 3.0 * LAMP_SPACING - offset;
    (
        in_yard(Vec3::new(x, 0.0, offset)),
        in_yard(Vec3::new(
            x - LAMP_SPACING / 2.0,
            0.0,
            offset - LAMP_SPACING / 2.0,
        )),
    )
}

/// The yard at `moment`, for a target of `width` by `height`.
pub(crate) fn build(renderer: &mut Renderer, moment: Moment, width: u32, height: u32) -> RenderFrame {
    let eye = in_yard(EYE);
    let mut frame = RenderFrame::new(
        camera(eye, in_yard(TARGET), width as f32 / height as f32),
        Color::BLACK,
    );
    frame.camera_position = eye.to_array();
    frame.light = moment.key();
    frame.ambient.intensity = moment.ambient();
    frame.sky = Some(Sky {
        sun_direction: moment.sun().to_array(),
        sun_illuminance: [std::f32::consts::PI; 3],
        sun_angular_radius: 0.0093,
    });
    frame.fog = Some(HeightFog::default());
    frame.ambient_occlusion = Some(AmbientOcclusion::default());
    frame.contact_shadows = Some(ContactShadows::default());
    frame.post = PostSettings {
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
    if moment == Moment::Night {
        frame.lights = lamps();
    }

    let ground = renderer
        .upload_mesh(&cuboid(
            Vec3::new(-GROUND_HALF, -0.5, -GROUND_HALF),
            Vec3::new(GROUND_HALF, 0.0, GROUND_HALF),
        ))
        .expect("valid");
    let ground_material = pbr(renderer, [0.45, 0.42, 0.38, 1.0], 0.0, 0.85);
    place(&mut frame, ground, ground_material, Mat4::IDENTITY);

    // Crates: a big one, a small one beside it, a low step in front.
    let unit_box = renderer
        .upload_mesh(&cuboid(Vec3::new(-0.5, 0.0, -0.5), Vec3::new(0.5, 1.0, 0.5)))
        .expect("valid");
    let wood = pbr(renderer, [0.55, 0.36, 0.2, 1.0], 0.0, 0.75);
    for (position, size, turn) in [
        (Vec3::new(-1.6, 0.0, -0.6), Vec3::splat(1.2), 0.3),
        (Vec3::new(-0.75, 0.0, -0.2), Vec3::splat(0.5), -0.2),
        (Vec3::new(-1.0, 0.0, 1.0), Vec3::new(0.9, 0.15, 0.6), 0.0),
    ] {
        let model = Mat4::from_translation(position) * Mat4::from_rotation_y(turn) * Mat4::from_scale(size);
        place(&mut frame, unit_box, wood, model);
    }

    // A polished metal sphere, reflecting the sky.
    let ball = renderer.upload_mesh(&sphere(48, 24)).expect("valid");
    let chrome = pbr(renderer, [0.95, 0.93, 0.88, 1.0], 1.0, 0.15);
    place(
        &mut frame,
        ball,
        chrome,
        Mat4::from_translation(Vec3::new(0.9, 0.0, 0.2)) * Mat4::from_scale(Vec3::splat(1.1)),
    );

    // A cut-out plant: three leaf panels crossed about its stem.
    let leaf_panel = renderer.upload_mesh(&panel(1.0, 1.2)).expect("valid");
    let leaves = leaf_material(renderer);
    for turn in [0.0, 1.05, 2.1] {
        let model = Mat4::from_translation(Vec3::new(2.4, 0.0, -0.8)) * Mat4::from_rotation_y(turn);
        place(&mut frame, leaf_panel, leaves, model);
    }
    frame
}

/// Renders `moment` on a fresh renderer, twice, keeping the second frame:
/// auto-exposure and the environment bake each settle one frame late, and a
/// fresh renderer keeps one moment's throttled bake from reaching the next.
fn settled(moment: Moment, width: u32, height: u32) -> (Renderer, RenderFrame, Image) {
    let mut renderer = Renderer::headless(width, height, RendererConfig::default())
        .expect("these tests are ignored unless a GPU adapter exists");
    let frame = build(&mut renderer, moment, width, height);
    renderer.render_to_image(&frame).expect("renders");
    let image = renderer.render_to_image(&frame).expect("renders");
    (renderer, frame, image)
}

fn luminance(image: &Image, (x, y): (u32, u32)) -> u32 {
    let [r, g, b, _] = image.pixel(x, y).expect("in bounds");
    u32::from(r) + u32::from(g) + u32::from(b)
}

/// Every pass phase 2 added, by graph name. `light-clusters` only runs on a
/// frame with lights, so the night.
fn phase_two_passes(moment: Moment) -> Vec<&'static str> {
    let mut passes = vec![
        "shadow",
        "sky-view",
        "depth-prepass",
        "ao",
        "ao-denoise",
        "contact-shadows",
        "lit",
        "sky",
    ];
    if moment == Moment::Night {
        passes.push("light-clusters");
    }
    passes
}

#[test]
fn every_moment_is_lit_from_above_and_the_lamps_stand_over_the_ground() {
    let lamps = lamps();
    assert_eq!(lamps.len(), 64);
    for (index, lamp) in lamps.iter().enumerate() {
        assert!(
            lamp.position[0].abs() < 5.0 && lamp.position[2].abs() < 5.0,
            "lamp {index} stands within the yard"
        );
    }
    for moment in Moment::ALL {
        assert!((moment.sun().length() - 1.0).abs() < 1e-3, "{moment:?}");
        assert!(moment.key().direction[1] > 0.0, "{moment:?} is lit from above");
    }
}

/// The phase 2 golden set. References are captured on the RX 7900 XT with
/// `UPDATE_GOLDEN=1`; the advisory lavapipe CI job runs this test with a
/// tolerance (`GOLDEN_TOLERANCE`) and keeps what it rendered.
#[test]
#[ignore = "requires a GPU adapter and driver"]
fn the_day_night_showcase_matches_its_golden_images() {
    let mut failures = Vec::new();
    for moment in Moment::ALL {
        let (renderer, frame, image) = settled(moment, GOLDEN_WIDTH, GOLDEN_HEIGHT);
        let names = crate::ao_tests::pass_names(renderer.graph_dump());
        for pass in phase_two_passes(moment) {
            assert!(names.contains(&pass), "{moment:?} runs {pass}: {names:?}");
        }
        if moment == Moment::Night {
            let (pool, gap) = lamp_pool_and_gap();
            let at = |point| luminance(&image, pixel_of(&frame, point, GOLDEN_WIDTH, GOLDEN_HEIGHT));
            assert!(
                at(pool) > at(gap) + 30,
                "the night is lit in pools under the lamps: {} under one, {} between",
                at(pool),
                at(gap)
            );
        }
        if let Err(message) = golden::check(&format!("day_night_{}", moment.name()), &image, 0) {
            failures.push(message);
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// ADR-0025's budgets for the screen-space passes at 1440p on the High tier,
/// in milliseconds, each over the passes whose names start with a prefix.
/// Their cost follows the pixel count, so this small scene measures them
/// fairly; the geometry passes' budgets wait for a content-scale scene.
const SCREEN_SPACE_BUDGETS: [(&str, &[&str], f64); 7] = [
    ("ambient occlusion", &["ao"], 0.50),
    ("contact shadows", &["contact-shadows"], 0.30),
    ("sky", &["sky-view", "sky"], 0.10),
    ("exposure", &["exposure-"], 0.10),
    ("bloom", &["bloom-"], 0.25),
    ("tonemap", &["tonemap"], 0.15),
    ("light clusters", &["light-clusters"], 0.10),
];

/// GPU time over the passes named exactly `prefix`, or starting with it when
/// it ends in '-' or is "ao", which also covers "ao-denoise". `None` when no
/// pass matches, so a renamed pass cannot pass its budget at 0 ms.
fn time_of(timings: &[crate::PassTiming], prefixes: &[&str]) -> Option<f64> {
    let matched: Vec<f64> = timings
        .iter()
        .filter(|timing| {
            prefixes.iter().any(|&prefix| {
                timing.name == prefix
                    || ((prefix.ends_with('-') || prefix == "ao") && timing.name.starts_with(prefix))
            })
        })
        .filter_map(|timing| timing.gpu_ms)
        .map(f64::from)
        .collect();
    (!matched.is_empty()).then(|| matched.iter().sum())
}

/// Each pass's least GPU time over `frames` more frames, in the order the
/// passes run, or `None` without timestamps. The least rather than the last:
/// other work sharing the GPU, such as the rest of the test suite, stretches
/// some frames' timestamps, never shortens them.
fn least_pass_times(
    renderer: &mut Renderer,
    frame: &RenderFrame,
    frames: u32,
) -> Option<Vec<crate::PassTiming>> {
    let mut least: Option<Vec<crate::PassTiming>> = None;
    for _ in 0..frames {
        renderer.render(frame).expect("renders");
        let timings = renderer.gpu_timings()?;
        match &mut least {
            None => least = Some(timings.to_vec()),
            Some(least) => {
                for kept in least.iter_mut() {
                    let latest = timings
                        .iter()
                        .find(|timing| timing.name == kept.name)
                        .and_then(|timing| timing.gpu_ms);
                    kept.gpu_ms = match (kept.gpu_ms, latest) {
                        (Some(kept), Some(latest)) => Some(kept.min(latest)),
                        (kept, latest) => kept.or(latest),
                    };
                }
            }
        }
    }
    least
}

/// The phase 2 frame-time check: the showcase at 1440p with every pass on,
/// timed over many frames with the GPU's work included, at noon and at night
/// with its 64 lights. Prints each pass's least GPU time (so "all passes" is
/// a floor, not a real frame), and holds ADR-0025's screen-space budgets on
/// the reference GPU.
#[test]
#[ignore = "requires a GPU adapter and driver; measures performance"]
fn the_day_night_showcase_holds_60_fps_at_1440p() {
    const FRAMES: u32 = 240;
    let (width, height) = (2560, 1440);
    for moment in [Moment::Noon, Moment::Night] {
        let (mut renderer, frame, _) = settled(moment, width, height);
        let started = std::time::Instant::now();
        for _ in 0..FRAMES {
            renderer.render(&frame).expect("renders");
        }
        // Reading back waits for every queued frame to finish on the GPU.
        renderer.render_to_image(&frame).expect("reads back");
        let per_frame = started.elapsed() / (FRAMES + 1);
        let adapter = renderer.adapter_info().name.clone();
        println!(
            "{moment:?} at {width}x{height} on {adapter}: {:.3} ms per frame",
            per_frame.as_secs_f64() * 1000.0
        );
        assert!(
            per_frame < std::time::Duration::from_micros(16_667),
            "{moment:?}: {per_frame:?} per frame is under 60 fps"
        );
        let Some(timings) = least_pass_times(&mut renderer, &frame, 60) else {
            println!("no timestamp support on {adapter}");
            continue;
        };
        let mut total = 0.0;
        for timing in &timings {
            if let Some(ms) = timing.gpu_ms {
                total += ms;
                println!("  {:<20} {ms:.3} ms", timing.name);
            }
        }
        println!("  {:<20} {total:.3} ms", "all passes");
        if !adapter.ends_with("7900 XT") {
            continue;
        }
        for (name, prefixes, budget) in SCREEN_SPACE_BUDGETS {
            // Light clusters run only on a frame with lights.
            if name == "light clusters" && moment != Moment::Night {
                continue;
            }
            let spent = time_of(&timings, prefixes)
                .unwrap_or_else(|| panic!("{moment:?}: no timed pass for {name} among {prefixes:?}"));
            assert!(
                spent <= budget,
                "{moment:?}: {name} takes {spent:.3} ms, over its {budget} ms budget (ADR-0025)"
            );
        }
    }
}
