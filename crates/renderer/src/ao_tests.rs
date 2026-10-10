//! C4 acceptance on the GPU: GTAO darkens a concave corner and leaves an
//! open plane alone, leaves no halo around a thin object far in front of a
//! wall, sees cut-outs through the masked prepass, costs nothing once turned
//! off, and fits its budget at 1080p.

use glam::{Mat4, Vec3};

use crate::{
    AlphaMode, AmbientOcclusion, Color, DrawCommand, MaterialDesc, MaterialHandle, MeshData, RenderFrame,
    Renderer, RendererConfig, Shading,
};

const IDENTITY: [[f32; 4]; 4] = Mat4::IDENTITY.to_cols_array_2d();

pub(crate) fn headless(width: u32, height: u32) -> Renderer {
    Renderer::headless(width, height, RendererConfig::default())
        .expect("these tests are ignored unless a GPU adapter exists")
}

/// An axis-aligned box with a face per side, normals out.
pub(crate) fn cuboid(low: Vec3, high: Vec3) -> MeshData {
    // Each face: its normal, then its four corners counter-clockwise seen
    // from outside.
    let faces: [(Vec3, [Vec3; 4]); 6] = [
        (
            Vec3::X,
            [
                Vec3::new(high.x, low.y, high.z),
                Vec3::new(high.x, low.y, low.z),
                Vec3::new(high.x, high.y, low.z),
                Vec3::new(high.x, high.y, high.z),
            ],
        ),
        (
            Vec3::NEG_X,
            [
                Vec3::new(low.x, low.y, low.z),
                Vec3::new(low.x, low.y, high.z),
                Vec3::new(low.x, high.y, high.z),
                Vec3::new(low.x, high.y, low.z),
            ],
        ),
        (
            Vec3::Y,
            [
                Vec3::new(low.x, high.y, high.z),
                Vec3::new(high.x, high.y, high.z),
                Vec3::new(high.x, high.y, low.z),
                Vec3::new(low.x, high.y, low.z),
            ],
        ),
        (
            Vec3::NEG_Y,
            [
                Vec3::new(low.x, low.y, low.z),
                Vec3::new(high.x, low.y, low.z),
                Vec3::new(high.x, low.y, high.z),
                Vec3::new(low.x, low.y, high.z),
            ],
        ),
        (
            Vec3::Z,
            [
                Vec3::new(low.x, low.y, high.z),
                Vec3::new(high.x, low.y, high.z),
                Vec3::new(high.x, high.y, high.z),
                Vec3::new(low.x, high.y, high.z),
            ],
        ),
        (
            Vec3::NEG_Z,
            [
                Vec3::new(high.x, low.y, low.z),
                Vec3::new(low.x, low.y, low.z),
                Vec3::new(low.x, high.y, low.z),
                Vec3::new(high.x, high.y, low.z),
            ],
        ),
    ];
    let mut mesh = MeshData::default();
    for (face, (normal, corners)) in faces.iter().enumerate() {
        let base = (face * 4) as u32;
        mesh.positions
            .extend(corners.iter().map(|corner| corner.to_array()));
        mesh.normals.extend([normal.to_array(); 4]);
        mesh.uvs.extend([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]);
        mesh.indices
            .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    mesh
}

/// A vertical quad in the plane z = `z`, facing +Z, for cut-outs.
pub(crate) fn upright_quad(half_width: f32, height: f32, z: f32) -> MeshData {
    MeshData {
        positions: vec![
            [-half_width, 0.0, z],
            [half_width, 0.0, z],
            [half_width, height, z],
            [-half_width, height, z],
        ],
        normals: vec![[0.0, 0.0, 1.0]; 4],
        uvs: vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        indices: vec![0, 1, 2, 0, 2, 3],
        ..MeshData::default()
    }
}

pub(crate) fn camera(eye: Vec3, target: Vec3, aspect: f32) -> [[f32; 4]; 4] {
    let view = glam::camera::rh::view::look_at_mat4(eye, target, Vec3::Y);
    let projection = glam::camera::rh::proj::directx::perspective(50_f32.to_radians(), aspect, 0.1, 100.0);
    (projection * view).to_cols_array_2d()
}

pub(crate) fn draw(
    frame: &mut RenderFrame,
    renderer: &mut Renderer,
    mesh: &MeshData,
    material: MaterialHandle,
) {
    let mesh = renderer.upload_mesh(mesh).expect("valid");
    frame.push(DrawCommand {
        mesh,
        material,
        model: IDENTITY,
        normal_matrix: IDENTITY,
    });
}

/// Ambient light only, so every pixel's brightness follows its occlusion.
fn ambient_only(renderer: &Renderer, eye: Vec3, target: Vec3) -> RenderFrame {
    let (width, height) = renderer.surface_size();
    let mut frame = RenderFrame::new(camera(eye, target, width as f32 / height as f32), Color::BLACK);
    frame.camera_position = eye.to_array();
    frame.light.intensity = 0.0;
    frame.ambient.intensity = 1.0;
    frame
}

const CORNER_EYE: Vec3 = Vec3::new(0.0, 1.6, 2.5);
const CORNER_TARGET: Vec3 = Vec3::new(0.0, 0.2, -0.8);
/// The wall's face, where it meets the ground.
const WALL_Z: f32 = -1.0;

/// A ground plane meeting a wall: a concave corner along the wall's foot.
fn corner(renderer: &mut Renderer) -> RenderFrame {
    let mut frame = ambient_only(renderer, CORNER_EYE, CORNER_TARGET);
    let material = renderer.create_material(&MaterialDesc::default());
    draw(
        &mut frame,
        renderer,
        &cuboid(Vec3::new(-10.0, -0.5, -10.0), Vec3::new(10.0, 0.0, 10.0)),
        material,
    );
    draw(
        &mut frame,
        renderer,
        &cuboid(Vec3::new(-3.0, 0.0, WALL_Z - 0.2), Vec3::new(3.0, 2.0, WALL_Z)),
        material,
    );
    frame
}

/// The occlusion result: visibility per half-resolution texel, row by row.
struct Occlusion {
    width: u32,
    height: u32,
    visibility: Vec<f32>,
}

/// A texture's texels, `bytes_per_texel` each, row by row without padding.
pub(crate) fn read_texels(renderer: &Renderer, texture: &wgpu::Texture, bytes_per_texel: u32) -> Vec<u8> {
    let (width, height) = (texture.width(), texture.height());
    let bytes_per_row = (width * bytes_per_texel).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("screen-result-readback"),
        size: u64::from(bytes_per_row) * u64::from(height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = renderer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        texture.size(),
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
    mapped
        .chunks_exact(bytes_per_row as usize)
        .flat_map(|row| row[..(width * bytes_per_texel) as usize].iter().copied())
        .collect()
}

/// The pixel `point` lands on, in a target of `width` by `height`.
pub(crate) fn pixel_of(frame: &RenderFrame, point: Vec3, width: u32, height: u32) -> (u32, u32) {
    let ndc = Mat4::from_cols_array_2d(&frame.view_proj).project_point3(point);
    let x = ((ndc.x * 0.5 + 0.5) * width as f32) as u32;
    let y = ((0.5 - ndc.y * 0.5) * height as f32) as u32;
    assert!(x < width && y < height, "{point} is off screen");
    (x, y)
}

impl Occlusion {
    fn read(renderer: &Renderer) -> Self {
        let texture = renderer.ambient_occlusion.texture();
        let (width, height) = (texture.width(), texture.height());
        let half = crate::atmosphere::luts::tests::half_to_f32;
        let visibility = read_texels(renderer, texture, 8)
            .chunks_exact(8)
            .map(|texel| half(u16::from_le_bytes([texel[0], texel[1]])))
            .collect();
        Self {
            width,
            height,
            visibility,
        }
    }

    /// Visibility where `point` lands on screen.
    fn at(&self, frame: &RenderFrame, point: Vec3) -> f32 {
        let (x, y) = pixel_of(frame, point, self.width, self.height);
        self.visibility[(y * self.width + x) as usize]
    }
}

/// Pass names in recorded order, from the graph dump's "  0 name: ..." lines.
pub(crate) fn pass_names(dump: &str) -> Vec<&str> {
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
fn a_concave_corner_is_occluded_and_an_open_plane_is_not() {
    let mut renderer = headless(320, 240);
    let mut frame = corner(&mut renderer);
    frame.ambient_occlusion = Some(AmbientOcclusion::default());
    renderer.render_to_image(&frame).expect("renders");
    let occlusion = Occlusion::read(&renderer);

    for point in [
        Vec3::new(0.0, 0.0, WALL_Z + 0.05),
        Vec3::new(0.8, 0.0, WALL_Z + 0.08),
        Vec3::new(-0.5, 0.06, WALL_Z),
    ] {
        let visibility = occlusion.at(&frame, point);
        assert!(
            visibility < 0.7,
            "the corner at {point} is occluded: {visibility}"
        );
    }
    for point in [Vec3::new(0.0, 0.0, 0.8), Vec3::new(-0.7, 0.0, 0.6)] {
        let visibility = occlusion.at(&frame, point);
        assert!(
            (visibility - 1.0).abs() <= 0.02,
            "the open ground at {point} is open: {visibility}"
        );
    }
    let high_wall = occlusion.at(&frame, Vec3::new(0.3, 1.6, WALL_Z));
    assert!(
        (high_wall - 1.0).abs() <= 0.02,
        "the wall well above the corner is open: {high_wall}"
    );
}

#[test]
#[ignore = "requires a GPU adapter and driver"]
fn occlusion_runs_between_the_prepass_and_the_forward_pass_and_turns_off_cleanly() {
    let mut renderer = headless(160, 120);
    let mut frame = corner(&mut renderer);
    let before = renderer.render_to_image(&frame).expect("renders");
    let names = pass_names(renderer.graph_dump());
    assert!(
        !names.iter().any(|name| name.starts_with("ao")),
        "off costs nothing: {names:?}"
    );

    frame.ambient_occlusion = Some(AmbientOcclusion::default());
    let occluded = renderer.render_to_image(&frame).expect("renders");
    let names = pass_names(renderer.graph_dump());
    let position = |wanted: &str| names.iter().position(|&name| name == wanted);
    let order = ["depth-prepass", "ao", "ao-denoise", "lit"].map(position);
    assert!(
        order.iter().all(Option::is_some) && order.windows(2).all(|pair| pair[0] < pair[1]),
        "prepass, trace, denoise, then the forward pass: {names:?}"
    );
    assert_ne!(occluded.rgba8, before.rgba8, "the corner darkens");

    frame.ambient_occlusion = None;
    let after = renderer.render_to_image(&frame).expect("renders");
    assert_eq!(
        after.rgba8, before.rgba8,
        "turned off, the image is exactly as before"
    );
}

#[test]
#[ignore = "requires a GPU adapter and driver"]
fn a_thin_object_far_in_front_of_a_wall_leaves_no_halo_on_it() {
    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 240;
    let mut renderer = headless(WIDTH, HEIGHT);
    let eye = Vec3::new(0.0, 0.0, 4.0);
    let mut frame = ambient_only(&renderer, eye, Vec3::new(0.0, 0.0, -5.0));
    let wall = renderer.create_material(&MaterialDesc::default());
    let pole = renderer.create_material(&MaterialDesc {
        base_color: [1.0, 0.0, 0.0, 1.0],
        ..MaterialDesc::default()
    });
    draw(
        &mut frame,
        &mut renderer,
        &cuboid(Vec3::new(-8.0, -8.0, -3.2), Vec3::new(8.0, 8.0, -3.0)),
        wall,
    );
    // Two metres in front of the wall, twice the occlusion radius.
    draw(
        &mut frame,
        &mut renderer,
        &cuboid(Vec3::new(-0.04, -3.0, -1.04), Vec3::new(0.04, 3.0, -0.96)),
        pole,
    );
    let plain = renderer.render_to_image(&frame).expect("renders");
    frame.ambient_occlusion = Some(AmbientOcclusion::default());
    let occluded = renderer.render_to_image(&frame).expect("renders");

    let is_pole = |x: u32, y: u32| {
        let [r, g, _, _] = plain.pixel(x, y).expect("in bounds");
        r > 0 && g == 0
    };
    assert!((0..WIDTH).any(|x| is_pole(x, HEIGHT / 2)), "the pole is in view");
    let mut changed = 0;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let [a, b] = [&plain, &occluded].map(|image| image.pixel(x, y).expect("in bounds"));
            let difference: u32 = a.iter().zip(b).map(|(&a, b)| u32::from(a.abs_diff(b))).sum();
            if difference <= 3 || is_pole(x, y) {
                continue;
            }
            changed += 1;
            let near_pole = (x.saturating_sub(2)..=(x + 2).min(WIDTH - 1))
                .any(|nx| (y.saturating_sub(2)..=(y + 2).min(HEIGHT - 1)).any(|ny| is_pole(nx, ny)));
            assert!(
                near_pole,
                "the wall darkened at ({x}, {y}), more than 2 px from the pole ({difference})"
            );
        }
    }
    println!("wall pixels changed within 2 px of the pole: {changed}");
}

/// The filters weigh neighbours by distance: a bar hanging in front of the
/// dark corner, farther from it than the radius, stays open inside its
/// silhouette instead of taking on the corner's occlusion.
#[test]
#[ignore = "requires a GPU adapter and driver"]
fn occlusion_does_not_bleed_across_a_silhouette() {
    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 240;
    let mut renderer = headless(WIDTH, HEIGHT);
    let mut frame = corner(&mut renderer);
    let bar = renderer.create_material(&MaterialDesc {
        base_color: [1.0, 0.0, 0.0, 1.0],
        ..MaterialDesc::default()
    });
    // On the line from the eye to the corner, 2.7 m from it and 1.1 m up.
    let centre = CORNER_EYE + (Vec3::new(0.0, 0.0, WALL_Z) - CORNER_EYE) * 0.3;
    draw(
        &mut frame,
        &mut renderer,
        &cuboid(
            centre - Vec3::new(0.2, 0.025, 0.025),
            centre + Vec3::new(0.2, 0.025, 0.025),
        ),
        bar,
    );
    let plain = renderer.render_to_image(&frame).expect("renders");
    frame.ambient_occlusion = Some(AmbientOcclusion::default());
    let occluded = renderer.render_to_image(&frame).expect("renders");
    let behind = Occlusion::read(&renderer).at(&frame, Vec3::new(1.0, 0.0, WALL_Z + 0.03));
    assert!(behind < 0.7, "the corner behind the bar is dark: {behind}");

    let is_bar = |x: u32, y: u32| plain.pixel(x, y).is_some_and(|[r, g, _, _]| r > 0 && g == 0);
    let inside = |x: u32, y: u32| {
        (x.saturating_sub(3)..=x + 3).all(|nx| (y.saturating_sub(3)..=y + 3).all(|ny| is_bar(nx, ny)))
    };
    let mut checked = 0;
    for y in 3..HEIGHT - 3 {
        for x in 3..WIDTH - 3 {
            if !inside(x, y) {
                continue;
            }
            checked += 1;
            let [a, b] = [&plain, &occluded].map(|image| image.pixel(x, y).expect("in bounds"));
            let difference: u32 = a.iter().zip(b).map(|(&a, b)| u32::from(a.abs_diff(b))).sum();
            assert!(
                difference <= 3,
                "the bar darkened at ({x}, {y}), 3 px inside its edge ({difference})"
            );
        }
    }
    assert!(
        checked > 100,
        "the bar covers the corner on screen: {checked} pixels checked"
    );
}

#[test]
#[ignore = "requires a GPU adapter and driver"]
fn cut_outs_occlude_where_they_are_drawn_and_not_through_their_holes() {
    let visibility_at_foot = |alpha: f32| {
        let mut renderer = headless(320, 240);
        let mut frame = ambient_only(&renderer, CORNER_EYE, CORNER_TARGET);
        let ground = renderer.create_material(&MaterialDesc::default());
        draw(
            &mut frame,
            &mut renderer,
            &cuboid(Vec3::new(-10.0, -0.5, -10.0), Vec3::new(10.0, 0.0, 10.0)),
            ground,
        );
        let leaf = renderer.create_material(&MaterialDesc {
            shading: Shading::Pbr,
            base_color: [1.0, 1.0, 1.0, alpha],
            alpha: AlphaMode::Mask { cutoff: 0.5 },
            ..MaterialDesc::default()
        });
        draw(&mut frame, &mut renderer, &upright_quad(3.0, 2.0, WALL_Z), leaf);
        frame.ambient_occlusion = Some(AmbientOcclusion::default());
        renderer.render_to_image(&frame).expect("renders");
        Occlusion::read(&renderer).at(&frame, Vec3::new(0.0, 0.0, WALL_Z + 0.05))
    };
    let solid = visibility_at_foot(1.0);
    assert!(
        solid < 0.7,
        "a solid cut-out occludes the ground at its foot: {solid}"
    );
    let hole = visibility_at_foot(0.0);
    assert!(hole > 0.95, "a fully cut-away one does not: {hole}");
}

/// Records what occlusion costs at 1080p, and what the normal target and
/// the masked prepass add to the prepass, where the device can time passes.
/// The 1 ms budget is held on the reference GPU only.
#[test]
#[ignore = "requires a GPU adapter and driver; measures performance"]
fn occlusion_fits_its_budget_at_1080p() {
    let mut renderer = headless(1920, 1080);
    let mut frame = corner(&mut renderer);
    let crate_material = renderer.create_material(&MaterialDesc::default());
    let leaf = renderer.create_material(&MaterialDesc {
        shading: Shading::Pbr,
        alpha: AlphaMode::Mask { cutoff: 0.5 },
        ..MaterialDesc::default()
    });
    for index in 0..64 {
        let x = (index % 8) as f32 * 0.7 - 2.45;
        let z = (index / 8) as f32 * -0.6 + 1.0;
        let low = Vec3::new(x, 0.0, z);
        draw(
            &mut frame,
            &mut renderer,
            &cuboid(low, low + Vec3::splat(0.3)),
            crate_material,
        );
        let mut sprig = upright_quad(0.2, 0.5, z - 0.2);
        for position in &mut sprig.positions {
            position[0] += x;
        }
        draw(&mut frame, &mut renderer, &sprig, leaf);
    }
    let timings = |renderer: &mut Renderer, frame: &RenderFrame| {
        for _ in 0..12 {
            renderer.render(frame).expect("renders");
        }
        renderer.gpu_timings().map(<[crate::PassTiming]>::to_vec)
    };
    let Some(off) = timings(&mut renderer, &frame) else {
        println!("no timestamp support on {}", renderer.adapter_info().name);
        return;
    };
    frame.ambient_occlusion = Some(AmbientOcclusion::default());
    let on = timings(&mut renderer, &frame).expect("timestamps were supported a moment ago");
    let ms = |timings: &[crate::PassTiming], name: &str| {
        timings
            .iter()
            .find(|timing| timing.name == name)
            .and_then(|timing| timing.gpu_ms)
    };
    let adapter = renderer.adapter_info().name.clone();
    println!(
        "1920x1080 on {adapter}: depth-prepass {:?} ms off, {:?} ms with normals and cut-outs; \
         ao {:?} ms, ao-denoise {:?} ms, lit {:?} ms",
        ms(&off, "depth-prepass"),
        ms(&on, "depth-prepass"),
        ms(&on, "ao"),
        ms(&on, "ao-denoise"),
        ms(&on, "lit"),
    );
    let (Some(trace), Some(denoise)) = (ms(&on, "ao"), ms(&on, "ao-denoise")) else {
        panic!("both occlusion passes are timed by name");
    };
    if adapter.contains("7900") {
        assert!(
            trace + denoise <= 1.0,
            "{trace} + {denoise} ms is over the 1 ms budget"
        );
    }
}
