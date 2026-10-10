//! C5 acceptance on the GPU: clustered point and spot lights light what
//! they reach and nothing else, the cull runs before the forward pass, and a
//! full cluster reports its overflow.

use glam::{Mat4, Vec3};

use crate::{
    Color, DrawCommand, LocalLight, MaterialDesc, MeshData, RenderFrame, Renderer, RendererConfig, Shading,
    SpotCone,
};

const WIDTH: u32 = 160;
const HEIGHT: u32 = 120;
const IDENTITY: [[f32; 4]; 4] = Mat4::IDENTITY.to_cols_array_2d();

fn headless() -> Renderer {
    Renderer::headless(WIDTH, HEIGHT, RendererConfig::default())
        .expect("these tests are ignored unless a GPU adapter exists")
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

/// The camera's view-projection, from above and behind the origin.
fn camera(aspect: f32) -> [[f32; 4]; 4] {
    let view = glam::camera::rh::view::look_at_mat4(Vec3::new(0.0, 12.0, 6.0), Vec3::ZERO, Vec3::Y);
    let projection = glam::camera::rh::proj::directx::perspective(50_f32.to_radians(), aspect, 0.1, 100.0);
    (projection * view).to_cols_array_2d()
}

/// A white PBR ground seen from above in the dark: no sun, no ambient, so
/// every lit pixel is local light.
fn dark_ground(renderer: &mut Renderer) -> RenderFrame {
    let mesh = renderer.upload_mesh(&ground(30.0)).expect("valid");
    let material = renderer.create_material(&MaterialDesc {
        shading: Shading::Pbr,
        roughness: 0.8,
        ..MaterialDesc::default()
    });
    let (width, height) = renderer.surface_size();
    let mut frame = RenderFrame::new(camera(width as f32 / height as f32), Color::BLACK);
    frame.camera_position = [0.0, 12.0, 6.0];
    frame.light.intensity = 0.0;
    frame.ambient.intensity = 0.0;
    frame.push(DrawCommand {
        mesh,
        material,
        model: IDENTITY,
        normal_matrix: IDENTITY,
    });
    frame
}

fn pixel_of(frame: &RenderFrame, point: Vec3) -> (u32, u32) {
    let ndc = Mat4::from_cols_array_2d(&frame.view_proj).project_point3(point);
    (
        ((ndc.x * 0.5 + 0.5) * WIDTH as f32) as u32,
        ((0.5 - ndc.y * 0.5) * HEIGHT as f32) as u32,
    )
}

fn brightness(image: &crate::Image, (x, y): (u32, u32)) -> u32 {
    let [r, g, b, _] = image.pixel(x, y).expect("in bounds");
    u32::from(r) + u32::from(g) + u32::from(b)
}

fn lamp(position: [f32; 3], range: f32) -> LocalLight {
    LocalLight {
        position,
        range,
        color: [1.0, 1.0, 1.0],
        intensity: 3.0,
        spot: None,
    }
}

/// Pass names in recorded order, from the graph dump's "  0 name: ..." lines.
fn pass_names(dump: &str) -> Vec<&str> {
    dump.lines()
        .take_while(|line| !line.starts_with("resources"))
        .filter_map(|line| {
            line.trim_start()
                .split_once(' ')?
                .1
                .split_once(':')
                .map(|(name, _)| name)
        })
        .collect()
}

#[test]
#[ignore = "requires a GPU adapter and driver"]
fn a_point_light_lights_the_ground_within_its_range_only() {
    let mut renderer = headless();
    let mut frame = dark_ground(&mut renderer);
    let unlit = renderer.render_to_image(&frame).expect("renders");
    assert!(
        !pass_names(renderer.graph_dump()).contains(&"light-clusters"),
        "no lights, no cull:\n{}",
        renderer.graph_dump()
    );

    frame.lights.push(lamp([0.0, 1.0, 0.0], 3.0));
    let image = renderer.render_to_image(&frame).expect("renders");
    let names = pass_names(renderer.graph_dump());
    let cull = names.iter().position(|&name| name == "light-clusters");
    let lit = names.iter().position(|&name| name == "lit");
    assert!(
        matches!((cull, lit), (Some(cull), Some(lit)) if cull < lit),
        "lights are culled before the forward pass: {names:?}"
    );

    let under = brightness(&image, pixel_of(&frame, Vec3::ZERO));
    let aside = brightness(&image, pixel_of(&frame, Vec3::new(1.5, 0.0, 0.0)));
    let beyond = brightness(&image, pixel_of(&frame, Vec3::new(4.0, 0.0, 0.0)));
    assert_eq!(
        brightness(&unlit, pixel_of(&frame, Vec3::ZERO)),
        0,
        "dark without the lamp"
    );
    assert!(
        under > aside && aside > 0,
        "falls off with distance: {under} > {aside} > 0"
    );
    assert_eq!(beyond, 0, "nothing past the range");
}

#[test]
#[ignore = "requires a GPU adapter and driver"]
fn a_spot_light_lights_only_inside_its_cone() {
    let mut renderer = headless();
    let mut frame = dark_ground(&mut renderer);
    // Pointing straight down from 2 m with a 20 degree cone: a pool about
    // 0.73 m in radius.
    frame.lights.push(LocalLight {
        spot: Some(SpotCone {
            direction: [0.0, -1.0, 0.0],
            inner_angle: 15_f32.to_radians(),
            outer_angle: 20_f32.to_radians(),
        }),
        ..lamp([0.0, 2.0, 0.0], 6.0)
    });
    let image = renderer.render_to_image(&frame).expect("renders");
    assert!(
        brightness(&image, pixel_of(&frame, Vec3::ZERO)) > 0,
        "lit inside the cone"
    );
    assert_eq!(
        brightness(&image, pixel_of(&frame, Vec3::new(1.5, 0.0, 0.0))),
        0,
        "dark outside the cone though within range"
    );
}

/// 511 lamps on a grid over the visible ground, each reaching only its own
/// patch.
fn lamp_grid() -> Vec<LocalLight> {
    (0..511)
        .map(|index| {
            let (column, row) = ((index % 23) as f32, (index / 23) as f32);
            lamp([column * 0.9 - 10.0, 0.6, row * 0.9 - 10.0], 1.2)
        })
        .collect()
}

#[test]
#[ignore = "requires a GPU adapter and driver"]
fn with_512_lights_one_outside_the_view_changes_nothing() {
    let mut renderer = headless();
    let mut frame = dark_ground(&mut renderer);
    frame.lights = lamp_grid();
    let inside_only = renderer.render_to_image(&frame).expect("renders");
    let reference = pixel_of(&frame, Vec3::new(0.35, 0.0, 0.35));
    assert!(
        brightness(&inside_only, reference) > 0,
        "the lamps light the reference pixel"
    );

    // Far to the side and beyond the far plane: in no cluster.
    frame.lights.push(lamp([200.0, 1.0, 0.0], 20.0));
    assert_eq!(frame.lights.len(), 512);
    let with_outside = renderer.render_to_image(&frame).expect("renders");
    assert_eq!(
        brightness(&with_outside, reference),
        brightness(&inside_only, reference),
        "the reference pixel is unchanged"
    );
    assert!(with_outside == inside_only, "the whole image is unchanged");

    renderer.light_clusters.collect_blocking(&renderer.device);
    assert_eq!(
        renderer.light_clusters.latest_overflow(),
        Some(0),
        "no cluster is full"
    );
}

#[test]
#[ignore = "requires a GPU adapter and driver"]
fn a_full_cluster_counts_what_it_drops() {
    let mut renderer = headless();
    let mut frame = dark_ground(&mut renderer);
    // Dim enough that all 64 together stay below white.
    frame.lights = (0..64)
        .map(|index| LocalLight {
            intensity: 3.0 / 64.0,
            ..lamp([index as f32 * 0.01, 1.0, 0.0], 4.0)
        })
        .collect();
    let all = renderer.render_to_image(&frame).expect("renders");
    renderer.light_clusters.collect_blocking(&renderer.device);
    assert_eq!(
        renderer.light_clusters.latest_overflow(),
        Some(0),
        "the default cap holds 64"
    );

    frame.cluster_light_cap = 8;
    let capped = renderer.render_to_image(&frame).expect("renders");
    renderer.light_clusters.collect_blocking(&renderer.device);
    let dropped = renderer.light_clusters.latest_overflow().expect("read back");
    assert!(dropped > 0, "a lowered cap drops lights and says so");
    let centre = pixel_of(&frame, Vec3::ZERO);
    assert!(
        brightness(&capped, centre) < brightness(&all, centre),
        "dropped lights no longer light the ground"
    );

    // The next frame's stats carry the reading just collected.
    let stats = match renderer.render(&frame).expect("renders") {
        crate::FrameOutcome::Presented(stats) => stats,
        crate::FrameOutcome::Skipped(reason) => panic!("skipped: {reason:?}"),
    };
    assert_eq!(stats.light_overflow, Some(dropped), "the engine sees the count");

    frame.lights.clear();
    let stats = match renderer.render(&frame).expect("renders") {
        crate::FrameOutcome::Presented(stats) => stats,
        crate::FrameOutcome::Skipped(reason) => panic!("skipped: {reason:?}"),
    };
    assert_eq!(stats.light_overflow, Some(0), "no lights drop nothing, at once");
}

/// Records what the cull and the forward pass cost with 512 lights, where
/// the device can time passes. Prints rather than asserts a budget: budgets
/// are set at the M18 exit (slice 13).
#[test]
#[ignore = "requires a GPU adapter and driver; measures performance"]
fn the_cost_of_512_lights_is_recorded_per_pass() {
    let mut renderer = Renderer::headless(1920, 1080, RendererConfig::default())
        .expect("these tests are ignored unless a GPU adapter exists");
    let mut frame = dark_ground(&mut renderer);
    frame.lights = lamp_grid();
    frame.lights.push(lamp([0.0, 3.0, 0.0], 8.0));
    for _ in 0..12 {
        renderer.render(&frame).expect("renders");
    }
    let Some(timings) = renderer.gpu_timings() else {
        println!("no timestamp support on {}", renderer.adapter_info().name);
        return;
    };
    for timing in timings {
        if matches!(timing.name, "light-clusters" | "lit") {
            println!(
                "512 lights at 1920x1080 on {}: {} {:?} ms",
                renderer.adapter_info().name,
                timing.name,
                timing.gpu_ms
            );
        }
    }
    assert!(
        timings.iter().any(|timing| timing.name == "light-clusters"),
        "the cull is timed by name"
    );
}

/// The emissive term (B4 slot): a surface glows in the dark in its own
/// colour, scaled by its emissive texture, with no light at all.
#[test]
#[ignore = "requires a GPU adapter and driver"]
fn an_emissive_surface_glows_by_factor_times_texture() {
    use crate::{ColorSpace, MaterialTextures, Mips, TextureDesc};

    let mut renderer = headless();
    let frame = dark_ground(&mut renderer);
    let centre = pixel_of(&frame, Vec3::ZERO);
    let glowing = |renderer: &mut Renderer, texel: Option<[u8; 4]>| {
        let emissive = texel.map(|texel| {
            let desc = TextureDesc {
                width: 1,
                height: 1,
                color_space: ColorSpace::Srgb,
                mips: Mips::Single,
            };
            renderer.create_texture(&desc, &texel).expect("valid")
        });
        let material = renderer.create_material(&MaterialDesc {
            shading: Shading::Pbr,
            emissive: [0.6, 0.3, 0.1],
            textures: MaterialTextures {
                emissive,
                ..MaterialTextures::default()
            },
            ..MaterialDesc::default()
        });
        let mut frame = frame.clone();
        frame.draws[0].material = material;
        let image = renderer.render_to_image(&frame).expect("renders");
        image.pixel(centre.0, centre.1).expect("in bounds")
    };
    let [r, g, b, _] = glowing(&mut renderer, None);
    assert!(r > g && g > b && b > 0, "glows in its own colour: {r} {g} {b}");
    assert_eq!(
        glowing(&mut renderer, Some([0, 0, 0, 255]))[..3],
        [0, 0, 0],
        "a black emissive texture turns it off"
    );
}
