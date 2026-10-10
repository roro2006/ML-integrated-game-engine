//! C4 contact shadows on the GPU: a small object gains the contact shadow a
//! coarse cascade misses, nothing changes where the cascades already shadow
//! or where nothing stands, an object far in front of a surface casts
//! nothing onto it, cut-outs cast through the masked prepass, and the pass
//! costs nothing once turned off.

use glam::Vec3;

use crate::ao_tests::{camera, cuboid, draw, headless, pass_names, pixel_of, read_texels, upright_quad};
use crate::{
    AlphaMode, AmbientOcclusion, Color, ContactShadows, Image, MaterialDesc, RenderFrame, Renderer, Shading,
    ShadowSettings,
};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

/// The sun at 45 degrees, from -X: shadows fall toward +X, as long as their
/// caster is tall.
fn low_sun() -> [f32; 3] {
    Vec3::new(-1.0, 1.0, 0.0).normalize().to_array()
}

/// Sun and a little ambient, so a shadow is plainly darker than open ground.
fn sunlit(renderer: &Renderer, eye: Vec3, target: Vec3, to_light: [f32; 3]) -> RenderFrame {
    let (width, height) = renderer.surface_size();
    let mut frame = RenderFrame::new(camera(eye, target, width as f32 / height as f32), Color::BLACK);
    frame.camera_position = eye.to_array();
    frame.light = frame.light.with_direction(to_light);
    frame.light.intensity = 1.0;
    frame.ambient.intensity = 0.15;
    frame
}

fn ground(frame: &mut RenderFrame, renderer: &mut Renderer) {
    ground_of(frame, renderer, 10.0);
}

/// A ground plane `half` metres each way from the origin.
fn ground_of(frame: &mut RenderFrame, renderer: &mut Renderer, half: f32) {
    let material = renderer.create_material(&MaterialDesc::default());
    draw(
        frame,
        renderer,
        &cuboid(Vec3::new(-half, -0.5, -half), Vec3::new(half, 0.0, half)),
        material,
    );
}

/// Pixels that differ between two renders of the same size.
fn changed_pixels(a: &Image, b: &Image) -> usize {
    a.rgba8
        .chunks_exact(4)
        .zip(b.rgba8.chunks_exact(4))
        .filter(|(a, b)| a != b)
        .count()
}

fn brightness(image: &Image, (x, y): (u32, u32)) -> u32 {
    let [r, g, b, _] = image.pixel(x, y).expect("in bounds");
    u32::from(r) + u32::from(g) + u32::from(b)
}

/// Whether the contact shadow result is lit where `point` lands.
fn contact_lit(renderer: &Renderer, frame: &RenderFrame, point: Vec3) -> bool {
    let texture = renderer.contact_shadows.texture();
    let (width, height) = (texture.width(), texture.height());
    let (x, y) = pixel_of(frame, point, width, height);
    read_texels(renderer, texture, 4)[((y * width + x) * 4) as usize] > 127
}

const PEBBLE_EYE: Vec3 = Vec3::new(0.15, 0.25, 0.35);
const PEBBLE_TARGET: Vec3 = Vec3::new(0.03, 0.0, 0.0);
/// The pebble's half-size: a 5 cm cube.
const PEBBLE: f32 = 0.025;

/// A 5 cm cube on the ground under a single coarse cascade, whose texels
/// are far larger than the cube.
fn pebble(renderer: &mut Renderer) -> RenderFrame {
    let mut frame = sunlit(renderer, PEBBLE_EYE, PEBBLE_TARGET, low_sun());
    frame.light.shadow = ShadowSettings {
        cascades: 1,
        resolution: 256,
        max_distance: 40.0,
        ..ShadowSettings::default()
    };
    ground(&mut frame, renderer);
    let material = renderer.create_material(&MaterialDesc::default());
    draw(
        &mut frame,
        renderer,
        &cuboid(
            Vec3::new(-PEBBLE, 0.0, -PEBBLE),
            Vec3::new(PEBBLE, 2.0 * PEBBLE, PEBBLE),
        ),
        material,
    );
    frame
}

#[test]
#[ignore = "requires a GPU adapter and driver"]
fn a_small_object_on_a_plane_gains_the_contact_shadow_the_cascade_misses() {
    let mut renderer = headless(WIDTH, HEIGHT);
    let mut frame = pebble(&mut renderer);
    // Beside the pebble on the side away from the sun, inside the 5 cm its
    // 5 cm height throws at 45 degrees; and open ground off to one side.
    let shaded = Vec3::new(PEBBLE + 0.015, 0.0, 0.0);
    let open = Vec3::new(PEBBLE + 0.015, 0.0, 0.1);
    let cascades_only = renderer.render_to_image(&frame).expect("renders");
    frame.contact_shadows = Some(ContactShadows::default());
    let with_contact = renderer.render_to_image(&frame).expect("renders");
    assert!(
        !contact_lit(&renderer, &frame, shaded),
        "the march finds the pebble"
    );
    assert!(contact_lit(&renderer, &frame, open), "and nothing on open ground");

    let at = |image: &Image, point: Vec3| brightness(image, pixel_of(&frame, point, WIDTH, HEIGHT));
    let open_brightness = at(&cascades_only, open);
    let missed = at(&cascades_only, shaded);
    let caught = at(&with_contact, shaded);
    println!("open ground {open_brightness}, beside the pebble: cascades {missed}, with contact {caught}");
    assert!(
        missed * 10 >= open_brightness * 9,
        "the coarse cascade misses the pebble's shadow: {missed} against {open_brightness}"
    );
    assert!(
        caught * 10 <= open_brightness * 6,
        "contact shadows catch it: {caught} against {open_brightness}"
    );
    assert_eq!(
        at(&with_contact, open),
        open_brightness,
        "open ground keeps its light"
    );
}

#[test]
#[ignore = "requires a GPU adapter and driver"]
fn contact_shadows_run_after_the_prepass_and_turn_off_cleanly() {
    let mut renderer = headless(WIDTH, HEIGHT);
    let mut frame = pebble(&mut renderer);
    let before = renderer.render_to_image(&frame).expect("renders");
    let names = pass_names(renderer.graph_dump());
    assert!(
        !names.contains(&"contact-shadows"),
        "off costs nothing: {names:?}"
    );

    frame.contact_shadows = Some(ContactShadows::default());
    let shadowed = renderer.render_to_image(&frame).expect("renders");
    let names = pass_names(renderer.graph_dump());
    let position = |wanted: &str| names.iter().position(|&name| name == wanted);
    let order = ["depth-prepass", "contact-shadows", "lit"].map(position);
    assert!(
        order.iter().all(Option::is_some) && order.windows(2).all(|pair| pair[0] < pair[1]),
        "prepass, march, then the forward pass: {names:?}"
    );
    assert!(
        !names.iter().any(|name| name.starts_with("ao")),
        "contact shadows alone trace no occlusion: {names:?}"
    );
    assert_ne!(shadowed.rgba8, before.rgba8, "the pebble's contact shadow shows");

    frame.contact_shadows = None;
    let after = renderer.render_to_image(&frame).expect("renders");
    assert_eq!(
        after.rgba8, before.rgba8,
        "turned off, the image is exactly as before"
    );
}

/// The result is recreated with the output and group 0 rebuilt with it: after
/// a resize the march still lands where the pebble is, and turning contact
/// shadows off and on again leaves no stale shadow behind.
#[test]
#[ignore = "requires a GPU adapter and driver"]
fn contact_shadows_follow_a_resize_and_a_toggle() {
    let mut renderer = headless(WIDTH, HEIGHT);
    let mut frame = pebble(&mut renderer);
    frame.contact_shadows = Some(ContactShadows::default());
    renderer.render_to_image(&frame).expect("renders");

    renderer.resize(200, 150);
    frame = pebble(&mut renderer);
    let cascades_only = renderer.render_to_image(&frame).expect("renders");
    frame.contact_shadows = Some(ContactShadows::default());
    let with_contact = renderer.render_to_image(&frame).expect("renders");
    let texture = renderer.contact_shadows.texture();
    assert_eq!(
        (texture.width(), texture.height()),
        (200, 150),
        "the contact result follows the resize"
    );

    let shaded = Vec3::new(PEBBLE + 0.015, 0.0, 0.0);
    let open = Vec3::new(PEBBLE + 0.015, 0.0, 0.1);
    assert!(
        !contact_lit(&renderer, &frame, shaded),
        "the march still finds the pebble after a resize"
    );
    assert!(contact_lit(&renderer, &frame, open), "and open ground stays lit");
    // The lit pass reads the result through group 0: a bind group left
    // pointing at the old texture would show no shadow here.
    let at = |image: &Image| brightness(image, pixel_of(&frame, shaded, 200, 150));
    assert!(
        at(&with_contact) * 10 <= at(&cascades_only) * 6,
        "the forward pass sees the resized result: {} with contact, {} without",
        at(&with_contact),
        at(&cascades_only)
    );

    frame.contact_shadows = None;
    renderer.render_to_image(&frame).expect("renders");
    assert!(
        contact_lit(&renderer, &frame, shaded),
        "turned off, the result reads lit everywhere"
    );

    frame.contact_shadows = Some(ContactShadows::default());
    renderer.render_to_image(&frame).expect("renders");
    assert!(
        !contact_lit(&renderer, &frame, shaded),
        "turned back on, the shadow returns"
    );
}

#[test]
#[ignore = "requires a GPU adapter and driver"]
fn contact_shadows_change_nothing_where_the_cascades_already_shadow() {
    let mut renderer = headless(WIDTH, HEIGHT);
    let mut frame = sunlit(
        &renderer,
        Vec3::new(2.5, 2.0, 3.0),
        Vec3::new(0.8, 0.0, 0.0),
        low_sun(),
    );
    ground(&mut frame, &mut renderer);
    let material = renderer.create_material(&MaterialDesc::default());
    // A metre-high block: the cascades shadow the metre of ground beyond it.
    draw(
        &mut frame,
        &mut renderer,
        &cuboid(Vec3::new(-0.5, 0.0, -0.5), Vec3::new(0.5, 1.0, 0.5)),
        material,
    );
    let cascades_only = renderer.render_to_image(&frame).expect("renders");
    frame.contact_shadows = Some(ContactShadows::default());
    let with_contact = renderer.render_to_image(&frame).expect("renders");

    for point in [
        Vec3::new(0.52, 0.0, 0.0),
        Vec3::new(0.7, 0.0, 0.3),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.2, 0.0, -0.3),
    ] {
        let (x, y) = pixel_of(&frame, point, WIDTH, HEIGHT);
        assert_eq!(
            with_contact.pixel(x, y),
            cascades_only.pixel(x, y),
            "the cascade's shadow at {point} is left as it was"
        );
    }
}

#[test]
#[ignore = "requires a GPU adapter and driver"]
fn an_open_plane_does_not_shadow_itself_even_at_a_grazing_angle() {
    let cases = [
        // Low sun across a floor seen almost edge-on.
        (
            Vec3::new(0.0, 0.3, 3.0),
            Vec3::new(0.0, 0.2, -30.0),
            Vec3::new(-1.0, 0.27, 0.3),
        ),
        // Low sun ahead, against the camera: every step lands farther away
        // than the surface it started from.
        (
            Vec3::new(0.0, 0.3, 3.0),
            Vec3::new(0.0, 0.2, -30.0),
            Vec3::new(0.3, 0.27, -1.0),
        ),
        // Low sun from behind the camera, grazing the same floor.
        (
            Vec3::new(0.0, 0.3, 3.0),
            Vec3::new(0.0, 0.2, -30.0),
            Vec3::new(0.2, 0.2, 1.0),
        ),
        // High sun, looking down.
        (Vec3::new(0.0, 4.0, 3.0), Vec3::ZERO, Vec3::new(0.35, 0.8, 0.45)),
    ];
    for (eye, target, to_light) in cases {
        let mut renderer = headless(WIDTH, HEIGHT);
        let mut frame = sunlit(&renderer, eye, target, to_light.normalize().to_array());
        ground(&mut frame, &mut renderer);
        let plain = renderer.render_to_image(&frame).expect("renders");
        frame.contact_shadows = Some(ContactShadows::default());
        let with_contact = renderer.render_to_image(&frame).expect("renders");
        let changed = changed_pixels(&plain, &with_contact);
        assert_eq!(
            changed, 0,
            "eye {eye}, sun toward {to_light}: {changed} pixels of open ground shadowed"
        );
    }
}

/// Far from the camera, depth's precision falls with the square of the
/// distance while the thickness stays 0.2 m: the march must still not find
/// the ground behind itself. The camera's far plane is 100 m.
#[test]
#[ignore = "requires a GPU adapter and driver"]
fn distant_ground_does_not_shadow_itself() {
    for to_light in [Vec3::new(0.3, 0.27, -1.0), Vec3::new(-1.0, 0.4, 0.2)] {
        let mut renderer = headless(WIDTH, HEIGHT);
        let mut frame = sunlit(
            &renderer,
            Vec3::new(0.0, 6.0, 0.0),
            Vec3::new(0.0, 0.0, -70.0),
            to_light.normalize().to_array(),
        );
        ground_of(&mut frame, &mut renderer, 120.0);
        let plain = renderer.render_to_image(&frame).expect("renders");
        frame.contact_shadows = Some(ContactShadows::default());
        let with_contact = renderer.render_to_image(&frame).expect("renders");
        let (_, far_row) = pixel_of(&frame, Vec3::new(0.0, 0.0, -90.0), WIDTH, HEIGHT);
        assert!(far_row > 0, "ground 90 m away is in view");
        let changed = changed_pixels(&plain, &with_contact);
        assert_eq!(
            changed, 0,
            "sun toward {to_light}: {changed} pixels of distant ground shadowed"
        );
    }
}

/// Occlusion and contact shadows share the normal target but nothing else:
/// each turns on and off alone, and either alone leaves the other's result
/// neutral and its passes undeclared.
#[test]
#[ignore = "requires a GPU adapter and driver"]
fn occlusion_and_contact_shadows_switch_independently() {
    let mut renderer = headless(WIDTH, HEIGHT);
    let mut frame = pebble(&mut renderer);
    let neither = renderer.render_to_image(&frame).expect("renders");
    let mut images = Vec::new();
    for (occlusion, contact) in [(true, false), (false, true), (true, true), (false, false)] {
        frame.ambient_occlusion = occlusion.then(AmbientOcclusion::default);
        frame.contact_shadows = contact.then(ContactShadows::default);
        let image = renderer.render_to_image(&frame).expect("renders");
        let names = pass_names(renderer.graph_dump());
        assert_eq!(
            names.contains(&"ao"),
            occlusion,
            "occlusion {occlusion}, contact {contact}: {names:?}"
        );
        assert_eq!(
            names.contains(&"contact-shadows"),
            contact,
            "occlusion {occlusion}, contact {contact}: {names:?}"
        );
        assert_eq!(
            contact_lit(&renderer, &frame, Vec3::new(PEBBLE + 0.015, 0.0, 0.0)),
            !contact,
            "the contact result is neutral unless traced"
        );
        images.push(image);
    }
    assert_ne!(images[0].rgba8, neither.rgba8, "occlusion alone shows");
    assert_ne!(images[1].rgba8, neither.rgba8, "contact shadows alone show");
    assert_ne!(
        images[2].rgba8, images[0].rgba8,
        "both differ from occlusion alone"
    );
    assert_ne!(images[2].rgba8, images[1].rgba8, "and from contact shadows alone");
    assert_eq!(
        images[3].rgba8, neither.rgba8,
        "both off again, exactly as before"
    );
}

#[test]
#[ignore = "requires a GPU adapter and driver"]
fn an_object_far_in_front_of_a_surface_casts_no_contact_shadow_on_it() {
    let mut renderer = headless(WIDTH, HEIGHT);
    let eye = Vec3::new(0.0, 0.0, 4.0);
    // The sun behind and above the camera: a march from the wall heads back
    // toward the pole, passing far behind it on screen.
    let mut frame = sunlit(
        &renderer,
        eye,
        Vec3::new(0.0, 0.0, -5.0),
        Vec3::new(0.15, 0.1, 1.0).normalize().to_array(),
    );
    frame.light.shadow.cascades = 0;
    let wall = renderer.create_material(&MaterialDesc::default());
    draw(
        &mut frame,
        &mut renderer,
        &cuboid(Vec3::new(-8.0, -8.0, -3.2), Vec3::new(8.0, 8.0, -3.0)),
        wall,
    );
    let pole = renderer.create_material(&MaterialDesc {
        base_color: [1.0, 0.0, 0.0, 1.0],
        ..MaterialDesc::default()
    });
    draw(
        &mut frame,
        &mut renderer,
        &cuboid(Vec3::new(-0.04, -3.0, -1.04), Vec3::new(0.04, 3.0, -0.96)),
        pole,
    );
    let plain = renderer.render_to_image(&frame).expect("renders");
    frame.contact_shadows = Some(ContactShadows::default());
    let with_contact = renderer.render_to_image(&frame).expect("renders");
    assert_eq!(
        with_contact.rgba8, plain.rgba8,
        "two metres in front, beyond the thickness, the pole shadows nothing"
    );
}

#[test]
#[ignore = "requires a GPU adapter and driver"]
fn cut_outs_cast_contact_shadows_where_they_are_drawn_and_not_through_their_holes() {
    let lit_at_foot = |alpha: f32| {
        let mut renderer = headless(WIDTH, HEIGHT);
        // The sun low behind the leaf, so its foot shadows the ground in
        // front of it, toward the camera.
        let mut frame = sunlit(
            &renderer,
            Vec3::new(0.0, 0.6, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, -1.0).normalize().to_array(),
        );
        frame.light.shadow.cascades = 0;
        ground(&mut frame, &mut renderer);
        let leaf = renderer.create_material(&MaterialDesc {
            shading: Shading::Pbr,
            base_color: [1.0, 1.0, 1.0, alpha],
            alpha: AlphaMode::Mask { cutoff: 0.5 },
            ..MaterialDesc::default()
        });
        draw(&mut frame, &mut renderer, &upright_quad(1.0, 0.5, -1.0), leaf);
        frame.contact_shadows = Some(ContactShadows::default());
        renderer.render_to_image(&frame).expect("renders");
        contact_lit(&renderer, &frame, Vec3::new(0.0, 0.0, -0.95))
    };
    assert!(
        !lit_at_foot(1.0),
        "a solid cut-out shadows the ground at its foot"
    );
    assert!(lit_at_foot(0.0), "a fully cut-away one does not");
}

/// Records what contact shadows cost at 1080p, where the device can time
/// passes. Slice 13 sets the per-pass budgets from these numbers.
#[test]
#[ignore = "requires a GPU adapter and driver; measures performance"]
fn contact_shadows_are_timed_at_1080p() {
    let mut renderer = headless(1920, 1080);
    let mut frame = sunlit(
        &renderer,
        Vec3::new(0.0, 1.6, 3.0),
        Vec3::new(0.0, 0.0, -2.0),
        low_sun(),
    );
    ground(&mut frame, &mut renderer);
    let material = renderer.create_material(&MaterialDesc::default());
    for index in 0..64 {
        let low = Vec3::new(
            (index % 8) as f32 * 0.7 - 2.45,
            0.0,
            (index / 8) as f32 * -0.6 + 1.0,
        );
        draw(
            &mut frame,
            &mut renderer,
            &cuboid(low, low + Vec3::splat(0.3)),
            material,
        );
    }
    frame.contact_shadows = Some(ContactShadows::default());
    for _ in 0..12 {
        renderer.render(&frame).expect("renders");
    }
    let Some(timings) = renderer.gpu_timings().map(<[crate::PassTiming]>::to_vec) else {
        println!("no timestamp support on {}", renderer.adapter_info().name);
        return;
    };
    let ms = |name: &str| {
        timings
            .iter()
            .find(|timing| timing.name == name)
            .and_then(|timing| timing.gpu_ms)
    };
    println!(
        "1920x1080 on {}: depth-prepass {:?} ms, contact-shadows {:?} ms, lit {:?} ms",
        renderer.adapter_info().name,
        ms("depth-prepass"),
        ms("contact-shadows"),
        ms("lit"),
    );
    assert!(ms("contact-shadows").is_some(), "the pass is timed by name");
}
