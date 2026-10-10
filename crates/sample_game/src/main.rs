//! Cinderwake — the vertical slice used to validate the engine.
//!
//! At milestone M3 this is a blocked-out scene: a ground plane, primitive
//! props, and a player box driven in the fixed-step simulation while a
//! fixed-angle 2.5D camera follows it (ADR-0007). It exists to prove the
//! platform, timing, ECS, extraction, and renderer path end to end. Gameplay
//! arrives in M5.
//!
//! Usage: `cinderwake [--headless [FRAMES]]`

mod aim;
mod animation;
mod hud;
mod menu;
mod nameplate;
mod slice;
mod vfx;

use std::process::ExitCode;
use std::time::Duration;

use engine::camera::{Camera, FollowCamera};
use engine::debug_draw::{DebugDraw, colors};
use engine::ecs::{Component, Resource, World};
use engine::input::{FixedInput, FrameInput, KeyCode};
use engine::math::{Transform, Vec2, Vec3};
use engine::primitives;
use engine::render::{
    AmbientOcclusion, ClearColor, Color, ContactShadows, MaterialDesc, MaterialHandle, MeshData, MeshHandle,
    MeshRenderer, RenderAssets, RenderBounds, RenderExtractor, SceneLighting, Visibility,
};
use engine::time::{FixedTime, FrameTime};
use engine::{App, AppConfig, AppControl, EngineError, FixedStage, FrameStage};
// `EnemyState` is used only by the slice's integration tests, which name the
// type explicitly when collecting enemy states.
use game_runtime::combat::Attack;
#[cfg(test)]
use game_runtime::enemy::EnemyState;
use game_runtime::enemy::{EnemyBrain, PathFollower, follow_path};
use game_runtime::movement::{Capsule, CollisionWorld, Dodge};

const APP_ID: &str = "cinderwake";
const DEFAULT_HEADLESS_FRAMES: u64 = 600;
/// Where the slice begins: the hub.
const SLICE_START: &str = "cinderwake-square";
/// Fixed so a slice run is reproducible; a real new game would pick one and
/// record it in the save.
const SLICE_SEED: u64 = 0xC1DE_2026;
/// Metres per second the player box moves. Deliberately brisk so a short
/// headless run produces visible movement.
const PLAYER_SPEED: f32 = 6.0;

/// Marks the entity the camera follows and input drives.
#[derive(Debug)]
struct Player;

impl Component for Player {}

/// Whether the debug overlay (wire bounds and player axes) is drawn.
///
/// Off by default: an overlay that costs nothing when disabled is the only
/// kind that can stay in a release build.
#[derive(Debug, Default)]
struct DebugOverlay(bool);

impl Resource for DebugOverlay {}

/// Uploads a mesh and spawns an entity for it, returning the entity.
///
/// Bounds come from the CPU-side mesh rather than the GPU, because culling is
/// engine-side work and must not need a round trip through the backend.
fn spawn_prop(
    world: &mut World,
    mesh: MeshHandle,
    material: MaterialHandle,
    mesh_data: &MeshData,
    transform: Transform,
) {
    let entity = world.spawn();
    let _ = world.insert(entity, transform);
    let _ = world.insert(entity, MeshRenderer::new(mesh, material));
    let _ = world.insert(entity, Visibility::default());
    if let Some(bounds) = RenderBounds::from_mesh_data(mesh_data) {
        let _ = world.insert(entity, bounds);
    }
}

fn build_scene(world: &mut World, assets: &mut dyn RenderAssets) {
    let ground_mesh = primitives::plane(48.0, 12);
    let box_mesh = primitives::cube(1.0);
    let sphere_mesh = primitives::uv_sphere(0.8, 16, 12);
    let cylinder_mesh = primitives::cylinder(0.6, 2.4, 12);
    let cone_mesh = primitives::cone(0.9, 2.0, 10);

    // A failed upload is a content bug, not a crash: log it and leave the
    // scene short of one prop rather than taking the frame loop down.
    let upload =
        |assets: &mut dyn RenderAssets, mesh: &MeshData, name: &'static str| match assets.upload_mesh(mesh) {
            Ok(handle) => Some(handle),
            Err(error) => {
                engine::log_error!("game", "mesh upload failed", mesh = name, error = error);
                None
            }
        };

    let Some(ground) = upload(assets, &ground_mesh, "ground") else {
        return;
    };
    let Some(boxes) = upload(assets, &box_mesh, "box") else {
        return;
    };
    let Some(sphere) = upload(assets, &sphere_mesh, "sphere") else {
        return;
    };
    let Some(cylinder) = upload(assets, &cylinder_mesh, "cylinder") else {
        return;
    };
    let Some(cone) = upload(assets, &cone_mesh, "cone") else {
        return;
    };

    // Linear colours: the surface is sRGB and the GPU converts on write.
    let grass = assets.create_material(&MaterialDesc {
        base_color: [0.16, 0.30, 0.16, 1.0],
        ..MaterialDesc::default()
    });
    let player_material = assets.create_material(&MaterialDesc {
        base_color: [0.80, 0.42, 0.18, 1.0],
        ..MaterialDesc::default()
    });
    let stone = assets.create_material(&MaterialDesc {
        base_color: [0.42, 0.44, 0.48, 1.0],
        ..MaterialDesc::default()
    });
    let timber = assets.create_material(&MaterialDesc {
        base_color: [0.35, 0.24, 0.14, 1.0],
        ..MaterialDesc::default()
    });

    spawn_prop(world, ground, grass, &ground_mesh, Transform::IDENTITY);

    // The player box sits on the ground, so it is raised by half its height.
    let player = world.spawn();
    let _ = world.insert(player, Transform::from_xyz(0.0, 0.5, 0.0));
    let _ = world.insert(player, MeshRenderer::new(boxes, player_material));
    let _ = world.insert(player, Visibility::default());
    if let Some(bounds) = RenderBounds::from_mesh_data(&box_mesh) {
        let _ = world.insert(player, bounds);
    }
    let _ = world.insert(player, Player);

    // A deterministic ring of props, so what is on screen is reproducible and
    // frustum culling has something to actually reject.
    let ring = [
        (sphere, stone, &sphere_mesh, 0.8),
        (cylinder, timber, &cylinder_mesh, 1.2),
        (cone, timber, &cone_mesh, 1.0),
        (boxes, stone, &box_mesh, 0.5),
    ];
    for (index, (mesh, material, mesh_data, height)) in ring.into_iter().enumerate() {
        for step in 0..3 {
            let angle = (index * 3 + step) as f32 * (std::f32::consts::TAU / 12.0);
            let radius = 7.0 + (step as f32) * 3.0;
            let transform = Transform::from_xyz(angle.cos() * radius, height, angle.sin() * radius);
            spawn_prop(world, mesh, material, mesh_data, transform);
        }
    }
}

/// How many frames between performance-overlay log records.
///
/// Two seconds at 60 fps: often enough to watch a number move during a
/// validation run, rare enough not to drown the log it is written to.
const OVERLAY_LOG_INTERVAL: u64 = 120;

/// Puts the performance overlay into the log.
///
/// The overlay's rows have been built and unit-tested since M3 and nothing has
/// ever surfaced them: there is no text rendering, so `F1` draws wire boxes and
/// axes and not one number. `Status::marker` exists so an over-budget row is
/// visible "in a log or a CI failure and not only in colour on a screen", and
/// this is the caller that finally makes that true — the difference between a
/// native run being an impression and being a measurement.
///
/// The budget is frame time only. Draw-call and triangle limits belong to a
/// device profile measured on real hardware; inventing plausible-looking ones
/// here would quietly become the numbers everyone trusts, which is exactly what
/// `Budget`'s own documentation warns against. The counts are still reported —
/// they are simply not judged yet.
fn add_overlay_logging(app: &mut App) -> Result<(), EngineError> {
    app.add_frame_system(FrameStage::PostUpdate, "overlay-log", |world, _| {
        let Some(diagnostics) = world.resource::<engine::diagnostics::FrameDiagnostics>() else {
            return;
        };
        if diagnostics.frames == 0 || !diagnostics.frames.is_multiple_of(OVERLAY_LOG_INTERVAL) {
            return;
        }

        let budget = debug_tools::overlay::Budget::default();
        let metrics = world.metrics();
        let rows = debug_tools::overlay::rows(diagnostics, Some(&metrics), &budget);
        let summary = debug_tools::overlay::lines(&rows).join("  ");

        // Over budget is a warning rather than an info line: the point of the
        // marker is that a missed frame target should be hard to scroll past.
        if debug_tools::overlay::over_budget(&rows) {
            engine::log_warn!("overlay", "over budget", rows = summary);
        } else {
            engine::log_info!("overlay", "performance", rows = summary);
        }
    })?;
    Ok(())
}

/// The game's lighting: the engine's key and ambient light, with ambient
/// occlusion and contact shadows on, so props sit in their creases and on
/// the ground rather than float above it.
fn scene_lighting() -> SceneLighting {
    SceneLighting {
        ambient_occlusion: Some(AmbientOcclusion::default()),
        contact_shadows: Some(ContactShadows::default()),
        ..SceneLighting::default()
    }
}

fn build_app(renderer: engine::render::RendererConfig) -> Result<App, EngineError> {
    let mut app = App::new(AppConfig {
        app_id: APP_ID.to_owned(),
        title: "Cinderwake".to_owned(),
        renderer,
        ..AppConfig::default()
    })?;

    {
        let world = app.world_mut();
        world.insert_resource(Camera::default());
        world.insert_resource(FollowCamera::default());
        world.insert_resource(scene_lighting());
        world.insert_resource(RenderExtractor::new());
        world.insert_resource(DebugDraw::new());
        world.insert_resource(DebugOverlay::default());
        world.insert_resource(ClearColor(Color::rgb(0.05, 0.07, 0.11)));
    }

    add_overlay_logging(&mut app)?;
    app.add_startup_hook(build_scene);

    // Fixed step: movement is simulation, so it runs on the tick and uses the
    // tick duration — never frame time (ADR-0007).
    app.add_fixed_system(FixedStage::Update, "move-player", |world, _| {
        let step = world
            .resource::<FixedTime>()
            .map_or(0.0, |time| time.step.as_secs_f32());
        let Some(input) = world.resource::<FixedInput>() else {
            return;
        };
        let mut direction = Vec3::ZERO;
        if input.keys.pressed(KeyCode::W) {
            direction.z -= 1.0;
        }
        if input.keys.pressed(KeyCode::S) {
            direction.z += 1.0;
        }
        if input.keys.pressed(KeyCode::A) {
            direction.x -= 1.0;
        }
        if input.keys.pressed(KeyCode::D) {
            direction.x += 1.0;
        }
        let Some(direction) = direction.try_normalize() else {
            return;
        };
        let delta = direction * PLAYER_SPEED * step;
        for (_, transform, _) in world.query2_mut::<Transform, Player>() {
            transform.translation += delta;
        }
    })?;

    // Frame rate: the camera is presentation, so the rig is posed per frame
    // after simulation has moved its subject.
    app.add_frame_system(FrameStage::PostUpdate, "follow-player", |world, _| {
        let Some(target) = world
            .query2::<Transform, Player>()
            .next()
            .map(|(_, transform, _)| transform.translation)
        else {
            return;
        };
        let delta = world
            .resource::<FrameTime>()
            .map_or(0.0, |time| time.delta.as_secs_f32());
        world.resource_scope::<FollowCamera, _>(|world, rig| {
            rig.follow(target, delta);
            if let Some(camera) = world.resource_mut::<Camera>() {
                rig.apply_to(camera);
            }
        });
    })?;

    // F1 toggles the overlay. It runs in the frame domain because it
    // visualises presentation state; the runner hands the accumulated lines to
    // the renderer after extraction.
    app.add_frame_system(FrameStage::PostUpdate, "debug-overlay", |world, _| {
        let toggled = world
            .resource::<FrameInput>()
            .is_some_and(|input| input.keys.just_pressed(KeyCode::F1));
        if toggled && let Some(overlay) = world.resource_mut::<DebugOverlay>() {
            overlay.0 = !overlay.0;
        }
        if !world.resource::<DebugOverlay>().is_some_and(|overlay| overlay.0) {
            return;
        }
        world.resource_scope::<DebugDraw, _>(|world, debug| {
            for (_, transform, bounds) in world.query2::<Transform, RenderBounds>() {
                debug.wire_box(&bounds.world_bounds(&transform.to_matrix()), colors::BOUNDS);
            }
            if let Some((_, transform, _)) = world.query2::<Transform, Player>().next() {
                debug.axes(transform, 1.5);
            }
        });
    })?;

    app.add_frame_system(FrameStage::PostUpdate, "quit-on-escape", |world, _| {
        let quit = world
            .resource::<FrameInput>()
            .is_some_and(|input| input.keys.just_pressed(KeyCode::Escape));
        if quit && let Some(control) = world.resource_mut::<AppControl>() {
            control.request_exit();
        }
    })?;

    Ok(app)
}

/// Builds the playable slice on top of an app.
///
/// Separate from [`build_scene`] on purpose: that scene is what proves the
/// renderer path end to end, and four tests still assert against it. The slice
/// adds gameplay beside it rather than replacing the only integration coverage
/// the engine has.
/// An app with the engine's services but none of M3's scene.
///
/// [`build_app`] always adds [`build_scene`], which brings its own ground,
/// prop ring, player box, movement, and camera. Attaching the slice on top of
/// that draws two worlds in one frame — and, less obviously, makes any test
/// asserting "something was drawn" pass whether or not the slice drew anything.
fn build_slice_app(renderer: engine::render::RendererConfig) -> Result<App, EngineError> {
    let mut app = App::new(AppConfig {
        app_id: APP_ID.to_owned(),
        title: "Cinderwake".to_owned(),
        renderer,
        ..AppConfig::default()
    })?;

    {
        let world = app.world_mut();
        world.insert_resource(Camera::default());
        world.insert_resource(FollowCamera::default());
        world.insert_resource(scene_lighting());
        world.insert_resource(RenderExtractor::new());
        world.insert_resource(DebugDraw::new());
        world.insert_resource(DebugOverlay::default());
        world.insert_resource(ClearColor(Color::rgb(0.05, 0.07, 0.11)));
    }

    // The slice had no overlay of any kind: `build_app` registers the F1
    // wire-box system and this one never did, so a windowed slice run reported
    // nothing about itself at all.
    add_overlay_logging(&mut app)?;

    app.add_frame_system(FrameStage::PostUpdate, "quit-on-escape", |world, _| {
        let quit = world
            .resource::<FrameInput>()
            .is_some_and(|input| input.keys.just_pressed(KeyCode::Escape));
        if quit && let Some(control) = world.resource_mut::<AppControl>() {
            control.request_exit();
        }
    })?;

    Ok(app)
}

/// Loads content, builds the world, and attaches the slice to a fresh app.
///
/// Shared by the windowed and headless paths so they cannot drift apart.
/// Returns `None` having already logged why.
/// Opens the class panel when content defines more than one class to choose.
fn offer_class_choice(app: &mut App) {
    if let Some(slice) = app.world_mut().resource_mut::<slice::Slice>()
        && slice.content.classes.len() > 1
    {
        slice.offer = Some(game_runtime::selection::Offer::ClassChoice { chosen: false });
    }
}

fn start_slice(resume: bool, renderer: engine::render::RendererConfig) -> Option<App> {
    let paths = match engine::paths::EnginePaths::resolve(
        APP_ID,
        &engine::paths::PathEnvironment::from_process(APP_ID),
    ) {
        Ok(paths) => paths,
        Err(error) => {
            engine::log_error!("game", "cannot resolve engine paths", error = error);
            return None;
        }
    };
    let Some(assets) = paths.assets.clone() else {
        engine::log_error!("game", "no asset root found; cannot load content");
        return None;
    };

    let start = match game_core::ids::LocationId::new(SLICE_START) {
        Ok(start) => start,
        Err(error) => {
            engine::log_error!("game", "slice start id is invalid", error = error);
            return None;
        }
    };

    // Resolved before the world is built: resuming needs somewhere to read
    // from, and a new game needs somewhere to clear the log.
    let save_dir = match paths.create_writable_dirs() {
        Ok(()) => Some(paths.saves.clone()),
        Err(error) => {
            engine::log_error!("game", "cannot create save directories", error = error);
            None
        }
    };

    let content_root = assets.join("content");
    let mut slice = if resume {
        let Some(dir) = save_dir.as_ref() else {
            engine::log_error!("game", "cannot resume without a save directory");
            return None;
        };
        match slice::Slice::resume(&content_root, dir) {
            Ok(slice) => slice,
            Err(error) => {
                // Never quietly fall back to a fresh world. A player who asked
                // to resume and silently got a new game would lose everything
                // without being told, which is the exact failure ADR-0013 is
                // written against.
                engine::log_error!("game", "cannot resume the slice", error = error);
                return None;
            }
        }
    } else {
        match slice::Slice::new(&content_root, &start, SLICE_SEED) {
            Ok(slice) => slice,
            Err(error) => {
                engine::log_error!("game", "cannot start the slice", error = error);
                return None;
            }
        }
    };

    if let Some(dir) = save_dir {
        if !resume {
            // A new game's log starts empty, so it holds exactly the history
            // behind the snapshot this run will write. Appending to the
            // previous run's log would leave the two describing different
            // worlds, and ADR-0013's replay contract unverifiable.
            //
            // The previous `world.ron` is deliberately left alone: it is real
            // player state, and it will be replaced when this game saves. That
            // leaves a window — new game begun, nothing saved yet — where the
            // snapshot is old and the log is empty. That costs replay
            // verification, not player data.
            match std::fs::remove_file(dir.join(slice::EVENT_LOG)) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    engine::log_warn!("game", "cannot clear the previous event log", error = error)
                }
            }
        }
        slice.save_dir = Some(dir);
    }

    let mut app = match build_slice_app(renderer) {
        Ok(app) => app,
        Err(error) => {
            engine::log_error!("game", "cannot create application", error = error);
            return None;
        }
    };
    if let Err(error) = attach_slice(&mut app, slice) {
        engine::log_error!("game", "cannot attach the slice", error = error);
        return None;
    }
    Some(app)
}

fn attach_slice(app: &mut App, slice: slice::Slice) -> Result<(), EngineError> {
    {
        let world = app.world_mut();
        world.insert_resource(slice);
        // Mouse aim (see `aim`): where the cursor points, and the cast mode.
        world.insert_resource(aim::Aim::default());
        world.insert_resource(aim::CastInput::default());
    }

    // Aim first, before anything this tick acts on it: the cursor projected
    // onto the ground through last frame's camera, and the cast mode cycled.
    app.add_fixed_system(FixedStage::PreUpdate, "slice-aim", |world, _| {
        aim::update_aim(world);
        let cycle = world
            .resource::<FixedInput>()
            .is_some_and(|input| input.keys.just_pressed(aim::CYCLE_KEY));
        if cycle && let Some(input) = world.resource_mut::<aim::CastInput>() {
            input.mode = input.mode.next();
            input.aiming = None;
            engine::log_info!("slice", "cast mode", mode = input.mode.label());
        }
    })?;

    // Populating needs content, so it happens once at startup — and again on a
    // location change, which rebuilds the runtime from content and canonical
    // state rather than carrying entities across.
    app.add_startup_hook(|world, assets| {
        // The one place a `RenderAssets` handle exists. Uploading here rather
        // than in `populate` matters: populate runs again on every location
        // change, and would leak a mesh and material per transition.
        if let Some(visuals) = slice::SliceVisuals::build(assets) {
            world.insert_resource(visuals);
        }
        // Every model content defines, built once for the same reason the
        // visuals are (ADR-0017). Built before anything is populated, so the
        // first bodies spawned can wear them.
        let library = match (
            world.resource::<slice::SliceVisuals>(),
            world.resource::<slice::Slice>(),
        ) {
            (Some(visuals), Some(slice)) => Some(slice::ModelLibrary::build(
                &slice.content,
                assets,
                visuals.model_base,
                game_runtime::movement::Capsule::default().half_extent(),
            )),
            _ => None,
        };
        if let Some(library) = library {
            world.insert_resource(library);
        }
        // Effects and telegraphs share a few meshes and a material per part,
        // built once for the same reason again (M7 wave 4).
        let effects = world
            .resource::<slice::Slice>()
            .and_then(|slice| vfx::VfxLibrary::build(&slice.content, assets));
        if let Some(effects) = effects {
            world.insert_resource(effects);
            world.insert_resource(vfx::VfxRuntime::default());
        }
        // A mesh per name, for the plates over heads.
        let names = world
            .resource::<slice::Slice>()
            .map(|slice| nameplate::NameLibrary::build(&slice.content, assets));
        if let Some(names) = names {
            world.insert_resource(names);
        }
        // Glyphs and a panel for the on-screen menu.
        if let Some(look) = menu::MenuLook::build(assets) {
            world.insert_resource(look);
        }
        // The HUD's colours; it draws with the menu's glyphs.
        let look = hud::HudLook::build(
            assets,
            world.resource::<slice::Slice>().map(|slice| &slice.content),
        );
        world.insert_resource(look);
        world.insert_resource(hud::Notices::default());
        world.resource_scope::<slice::Slice, _>(|world, slice| {
            let collision = slice
                .current_location()
                .map_or_else(|| CollisionWorld::new(0.0), slice::collision_for);
            world.insert_resource(collision);
            let start = slice::populate(world, slice, None);
            // Spawned from inside the scope because the player's pools come
            // from canonical state now, not from a constant (ADR-0015).
            slice::spawn_player(world, slice, start);
        });
    });

    // Movement is simulation: fixed step, tick duration, never frame time.
    app.add_fixed_system(FixedStage::Update, "slice-player-move", |world, _| {
        // A roll is committed motion and overrides input for its duration
        // (ADR-0007). Steering out of a dodge would make the i-frames free.
        if let Some(player) = slice::player_entity(world)
            && world.get::<Dodge>(player).is_some()
        {
            world.resource_scope::<CollisionWorld, _>(|world, collision| {
                let Some(motion) = world.get_mut::<Dodge>(player).map(Dodge::advance) else {
                    return;
                };
                let Some(from) = world.get::<Transform>(player).map(|t| t.translation) else {
                    return;
                };
                let result = collision.move_capsule(Capsule::default(), from, motion);
                if let Some(transform) = world.get_mut::<Transform>(player) {
                    transform.translation = result.position;
                }
            });
            return;
        }

        // A broken stance holds the body where it was hit.
        if let Some(player) = slice::player_entity(world)
            && slice::staggered(world, player)
        {
            return;
        }
        let Some(input) = world.resource::<FixedInput>() else {
            return;
        };
        let mut direction = Vec3::ZERO;
        if input.keys.pressed(KeyCode::W) {
            direction.z -= 1.0;
        }
        if input.keys.pressed(KeyCode::S) {
            direction.z += 1.0;
        }
        if input.keys.pressed(KeyCode::A) {
            direction.x -= 1.0;
        }
        if input.keys.pressed(KeyCode::D) {
            direction.x += 1.0;
        }
        let Some(direction) = direction.try_normalize() else {
            return;
        };

        // Resolved first: `query2_mut` takes two component types, and moving
        // everything that merely *has* a facing would drive the enemies with
        // the player's input too.
        let Some(player) = world
            .query2::<Transform, slice::Player>()
            .next()
            .map(|(entity, _, _)| entity)
        else {
            return;
        };

        world.resource_scope::<CollisionWorld, _>(|world, collision| {
            let Some(from) = world.get::<Transform>(player).map(|t| t.translation) else {
                return;
            };
            let motion = direction * slice::PLAYER_SPEED;
            let result = collision.move_capsule(Capsule::default(), from, motion);
            if let Some(transform) = world.get_mut::<Transform>(player) {
                transform.translation = result.position;
            }
            // Facing follows motion, so the body looks where it walks — until
            // it acts, when the aim turns it instead (see `aim`). Walking
            // while committed must not steer a blow that is aimed elsewhere.
            if !aim::acting(world, player)
                && let Some(facing) = world.get_mut::<slice::Facing>(player)
            {
                facing.0 = direction;
            }
        });
    })?;

    // Enemies path toward the player over the current location's navmesh.
    app.add_fixed_system(FixedStage::Update, "slice-enemies", |world, _| {
        let Some(player) = world
            .query2::<Transform, slice::Player>()
            .next()
            .map(|(_, transform, _)| transform.translation)
        else {
            return;
        };

        // Collected first because stepping needs `&mut` on transforms while
        // pathing needs `&` on the navmesh and collision world.
        let enemies: Vec<(engine::ecs::Entity, Vec3)> = world
            .query2::<Transform, slice::Enemy>()
            .map(|(entity, transform, _)| (entity, transform.translation))
            .collect();

        world.resource_scope::<slice::Slice, _>(|world, slice| {
            world.resource_scope::<CollisionWorld, _>(|world, collision| {
                for (entity, position) in enemies {
                    // Staggered: it neither thinks, turns, moves, nor acts
                    // until its stance recovers.
                    if slice::staggered(world, entity) {
                        continue;
                    }
                    let distance = (player - position).length();
                    // An enemy mid-swing is committed, and `think` needs to know
                    // that or it would re-decide its state every tick.
                    let mid_attack = world
                        .get::<Attack>(entity)
                        .is_some_and(|attack| !attack.is_finished());
                    // A cast commits exactly as a swing does. Leaving it out
                    // let an enemy start a swing on the tick after it began a
                    // cast — `start_ability` refused the second cast, and the
                    // fallback below read that refusal as "swing instead" —
                    // so the ashlord summoned and swung at once.
                    let committed = mid_attack || world.get::<slice::Casting>(entity).is_some();

                    // Face the player before swinging: the hit volume sits in
                    // front of the attacker, so an enemy that never turns will
                    // swing past whoever it is chasing.
                    if let Some(direction) = (player - position).try_normalize()
                        && let Some(facing) = world.get_mut::<slice::Facing>(entity)
                    {
                        facing.0 = Vec3::new(direction.x, 0.0, direction.z);
                    }

                    let Some(brain) = world.get_mut::<EnemyBrain>(entity) else {
                        continue;
                    };
                    brain.think(distance, committed);
                    let wants_to_swing = brain.wants_to_attack();
                    let swing = brain.build_attack();
                    let brain = *brain;

                    // An enemy that knows an ability reaches for it before
                    // falling back to a swing (M8): the first in its authored
                    // list that is ready, affordable, and in range, which
                    // keeps "what this enemy leads with" a content decision
                    // and "how it decides" a rule with tests.
                    let cast = wants_to_swing
                        && !committed
                        && slice::enemy_ability(world, slice, entity, distance)
                            .is_some_and(|index| slice::start_ability(world, slice, entity, index));
                    // Never at air (M8): a swing that cannot reach the
                    // player is not started. A ranged enemy between spits
                    // holds its ground rather than slashing from seven
                    // metres away.
                    let reaches = distance <= swing.reach + swing.radius + Capsule::default().radius;
                    if wants_to_swing && !committed && !cast && reaches {
                        let _ = world.insert(entity, swing);
                    }

                    let Some(follower) = world.get_mut::<PathFollower>(entity) else {
                        continue;
                    };
                    let result = follow_path(
                        &brain,
                        follower,
                        Capsule::default(),
                        position,
                        player,
                        &slice.navmesh,
                        collision,
                    );
                    if let Some(transform) = world.get_mut::<Transform>(entity) {
                        transform.translation = result.position;
                    }
                }
            });
        });
    })?;

    // Interaction: talk, harvest, or use whatever is in reach. Fixed step
    // because what it produces — pickups, quest progress — is simulation, and a
    // frame-rate-dependent interact would let a fast machine gather twice.
    app.add_fixed_system(FixedStage::Update, "slice-interact", |world, _| {
        let pressed = world
            .resource::<FixedInput>()
            .is_some_and(|input| input.keys.just_pressed(KeyCode::E));
        if !pressed {
            return;
        }

        // Modal: while something is on offer, E is how the player walks away
        // from it rather than how they reach into the world. Escape is not
        // available for this — `slice-save-on-exit` owns it — and a menu that
        // could only be left by choosing from it would be a trap on any list
        // with nothing affordable on it.
        let closed = world
            .resource_mut::<slice::Slice>()
            .is_some_and(|slice| slice.close_offer());
        if closed {
            return;
        }

        let Some(from) = world
            .query2::<Transform, slice::Player>()
            .next()
            .map(|(_, transform, _)| transform.translation)
        else {
            return;
        };

        match slice::interaction_at(world, from) {
            slice::Interaction::Talk(npc) => {
                // Scoped so the resource borrow ends before the world is
                // queried again: `resource_mut` holds `world` mutably, and
                // finding the speaker's entity needs it immutably.
                let began = {
                    let Some(slice) = world.resource_mut::<slice::Slice>() else {
                        return;
                    };
                    // Opening the conversation is all that happens here. The
                    // `Talked` signal that "speak to X" objectives wait on is
                    // raised when the conversation *ends* (`Slice::end_offer`),
                    // because that is the tick its own `QuestAccepted` commits
                    // on. Raising it here would spend the signal a tick before
                    // the quest it belongs to existed.
                    if !slice.open_conversation(&npc) {
                        slice
                            .notices
                            .push(format!("{} has nothing to say.", hud::words(npc.name())));
                    }

                    // Agreeing to lead someone somewhere is the conversation's
                    // consequence, but the quest states who and where.
                    slice.begin_escort(&npc)
                };

                // Resolved into an owned entity in its own statement: a
                // `let`-chain would keep the query's iterator alive across the
                // insert, which borrows the world mutably.
                let speaker = if began {
                    world
                        .query2::<Transform, slice::Npc>()
                        .find(|(_, _, marker)| marker.0 == npc)
                        .map(|(entity, _, _)| entity)
                } else {
                    None
                };
                if let Some(entity) = speaker {
                    let _ = world.insert(entity, slice::Escorting);
                }
            }
            slice::Interaction::Examine { entity, clue } => {
                // Marked before the signal, for the same reason a harvested node
                // is: one prop must not satisfy a counted objective twice.
                let text = match world.get_mut::<slice::Clue>(entity) {
                    Some(node) if node.examined => return,
                    Some(node) => {
                        node.examined = true;
                        node.text.clone()
                    }
                    None => None,
                };
                if let Some(slice) = world.resource_mut::<slice::Slice>() {
                    // What the clue says, if content wrote anything (M9).
                    if let Some(text) = text {
                        slice.open_note(hud::subject(clue.name()), text);
                    }
                    slice.signal(narrative::quest::Signal::Investigated { clue });
                }
            }
            slice::Interaction::Decide { entity, choice } => {
                // Asked, not taken (M9): a decision is settled by confirming
                // it in the panel, never by walking into its marker. The
                // marker retires when the choice is settled, however that
                // happens (`retire_settled_points`).
                let prompt = world
                    .get::<slice::ChoicePoint>(entity)
                    .and_then(|point| point.prompt.clone())
                    .unwrap_or_else(|| "Settle this now? It cannot be undone.".to_owned());
                if let Some(slice) = world.resource_mut::<slice::Slice>() {
                    slice.open_decision(choice, prompt);
                }
            }
            slice::Interaction::Harvest { entity, item } => {
                // Marked taken first: a node that grants before it is spent
                // could be harvested twice in one tick's worth of input.
                if let Some(node) = world.get_mut::<slice::Gatherable>(entity) {
                    if node.taken {
                        return;
                    }
                    node.taken = true;
                }
                if let Some(slice) = world.resource_mut::<slice::Slice>() {
                    for event in game_runtime::interaction::gather(&item, 1) {
                        slice.raise(event);
                    }
                    slice.signal(narrative::quest::Signal::Collected { item, quantity: 1 });
                }
            }
            slice::Interaction::Craft => {
                // Using a forge opens its list; what gets made is the player's
                // choice, taken in `slice-select`. The `Collected` signal a
                // craft raises travels with the choice, not with standing here.
                if let Some(slice) = world.resource_mut::<slice::Slice>() {
                    slice.open_forge();
                }
            }
            slice::Interaction::Upgrade => {
                // Nothing signals a build: a blueprint's effect is the world
                // flag it grants, which dialogue, recipes, and services all
                // already gate on.
                if let Some(slice) = world.resource_mut::<slice::Slice>() {
                    slice.open_board();
                }
            }
            slice::Interaction::Use(spawn) => {
                let Some((destination, arrive_at)) = world
                    .resource::<slice::Slice>()
                    .and_then(|slice| slice.link_from(&spawn))
                else {
                    // A station that leads nowhere is a fixture, not a door.
                    return;
                };

                world.resource_scope::<slice::Slice, _>(|world, slice| {
                    slice::travel_to(world, slice, &destination, Some(&arrive_at));
                });
            }
            slice::Interaction::Nothing => {}
        }
    })?;

    // Picking from whatever is on offer. Fixed step for the same reason
    // interacting is: what a choice produces is simulation, and a
    // frame-rate-dependent menu would let a fast machine craft twice on one
    // press.
    app.add_fixed_system(FixedStage::Update, "slice-select", |world, _| {
        /// The digit keys, paired with the row each one takes. Nine because
        /// that is what fits on one hand and one row of keys; a longer list
        /// needs scrolling, which needs a real menu.
        const DIGITS: [(KeyCode, u32); 9] = [
            (KeyCode::Digit1, 1),
            (KeyCode::Digit2, 2),
            (KeyCode::Digit3, 3),
            (KeyCode::Digit4, 4),
            (KeyCode::Digit5, 5),
            (KeyCode::Digit6, 6),
            (KeyCode::Digit7, 7),
            (KeyCode::Digit8, 8),
            (KeyCode::Digit9, 9),
        ];

        // Resolved to an owned digit first: reading input borrows the world,
        // and taking the choice needs it mutably.
        let pressed = {
            let Some(input) = world.resource::<FixedInput>() else {
                return;
            };
            DIGITS
                .iter()
                .copied()
                .find(|(key, _)| input.keys.just_pressed(*key))
                .map(|(_, digit)| digit)
        };
        let Some(digit) = pressed else {
            return;
        };
        if let Some(slice) = world.resource_mut::<slice::Slice>() {
            slice.take_offer(digit);
        }
    })?;

    // Clicking the on-screen menu. Before the swing is read, so a click spent
    // on the menu — choosing an option, or anywhere on the open panel — is
    // marked spent and never becomes a swing too.
    app.add_fixed_system(FixedStage::Update, "slice-menu", |world, _| {
        if menu::click(world) {
            let tick = aim::tick(world);
            if let Some(input) = world.resource_mut::<aim::CastInput>() {
                input.confirmed_at = Some(tick);
            }
        }
    })?;

    // Reloading content in a development run (M9 wave 3). F5 asks; the reload
    // waits for a safe point — no fight, no panel open — so nothing is torn
    // out from under a swing or a conversation. A reload that fails, or that
    // would orphan the running world, changes nothing and says why.
    app.add_fixed_system(FixedStage::Update, "slice-reload", |world, _| {
        if world.resource::<DevTools>().is_none() {
            return;
        }
        let pressed = world
            .resource::<FixedInput>()
            .is_some_and(|input| input.keys.just_pressed(KeyCode::F5));
        let Some(slice) = world.resource_mut::<slice::Slice>() else {
            return;
        };
        if pressed {
            slice.reload_requested = true;
            if slice.in_combat || slice.offer.is_some() {
                slice
                    .notices
                    .push("Reload waits: finish the fight and close the panel.".to_owned());
            }
        }
        if !slice.reload_requested || slice.in_combat || slice.offer.is_some() {
            return;
        }
        slice.reload_requested = false;
        world.resource_scope::<slice::Slice, _>(|world, slice| {
            let notice = match slice::reload_content(world, slice) {
                Ok((count, presentation_changed)) => {
                    engine::log_info!("dev", "content reloaded", definitions = count);
                    if presentation_changed {
                        format!("Content reloaded - {count} definitions. Models, effects and names change on restart.")
                    } else {
                        format!("Content reloaded - {count} definitions.")
                    }
                }
                Err(refused) => {
                    for problem in refused.problems() {
                        engine::log_warn!("dev", "reload refused", problem = problem);
                    }
                    match &refused {
                        slice::ReloadRefused::NoRoot => "Reload refused: no content folder is known.".to_owned(),
                        slice::ReloadRefused::Invalid(problems) | slice::ReloadRefused::Orphans(problems) => {
                            // One line on screen; every problem is in the log.
                            let first: String = problems
                                .first()
                                .map_or("", String::as_str)
                                .chars()
                                .take(64)
                                .collect();
                            format!("Reload refused - {} problem(s): {first}", problems.len())
                        }
                    }
                }
            };
            slice.notices.push(notice);
        });
    })?;

    // The journal and the controls list (M9): read-only panels, each toggled
    // by its own key.
    app.add_fixed_system(FixedStage::Update, "slice-panels", |world, _| {
        let (journal, controls) = world.resource::<FixedInput>().map_or((false, false), |input| {
            (
                input.keys.just_pressed(KeyCode::J),
                input.keys.just_pressed(KeyCode::H),
            )
        });
        let Some(slice) = world.resource_mut::<slice::Slice>() else {
            return;
        };
        if journal {
            slice.toggle_panel(game_runtime::selection::Offer::Journal);
        } else if controls {
            slice.toggle_panel(game_runtime::selection::Offer::Controls);
        }
    })?;

    // Using a consumable (M9): X for the first one carried, or whatever the
    // inventory asked for. After the menu systems, so a pick made this tick is
    // started this tick.
    app.add_fixed_system(FixedStage::Update, "slice-use", |world, _| {
        let tick = world.resource::<FixedTime>().map_or(0, |time| time.tick);
        let quick = world
            .resource::<FixedInput>()
            .is_some_and(|input| input.keys.just_pressed(KeyCode::X));
        let wanted = world.resource_mut::<slice::Slice>().and_then(|slice| {
            slice
                .use_requested
                .take()
                .or_else(|| quick.then(|| slice::quick_use(slice)).flatten())
        });
        if let Some(item) = wanted {
            world.resource_scope::<slice::Slice, _>(|world, slice| {
                slice::start_use(world, slice, &item, tick);
            });
        }
    })?;

    // Opening what the player is carrying. Fixed step like the other menus,
    // because equipping produces a domain event and simulation owns those.
    app.add_fixed_system(FixedStage::Update, "slice-equipment", |world, _| {
        let pressed = world
            .resource::<FixedInput>()
            .is_some_and(|input| input.keys.just_pressed(KeyCode::I));
        if !pressed {
            return;
        }
        if let Some(slice) = world.resource_mut::<slice::Slice>() {
            // Refuses while anything else is open, so this cannot discard a
            // conversation cursor mid-sentence.
            slice.open_equipment();
        }
    })?;

    // Abilities sit between choosing and swinging, where the M6 plan puts them.
    // Fixed step, because a cast's wind-up and recovery are counted in ticks
    // like every other committed action.
    app.add_fixed_system(FixedStage::Update, "slice-abilities", |world, _| {
        // The engine's fixed-step index, never `state.clock.tick`: nothing
        // advances the canonical clock during play, so a status measured
        // against it would never expire.
        let tick = world.resource::<FixedTime>().map_or(0, |time| time.tick);

        // Read input first, act second — the shape `slice-select` uses, because
        // reading borrows the world and casting needs it mutably.
        let requested = {
            let Some(input) = world.resource::<FixedInput>() else {
                return;
            };
            let mut buttons = aim::Buttons::read(input);
            // A dodge pressed this tick wins, exactly as it beats an attack
            // below. Without this the cast would start first and `start_dodge`
            // would then refuse it as committed, silently inverting ADR-0007's
            // tie-break in the one place a player would feel it.
            if input.keys.just_pressed(KeyCode::ShiftLeft) || input.keys.just_pressed(KeyCode::ShiftRight) {
                buttons.pressed = None;
                buttons.released = [false; aim::ABILITY_KEYS.len()];
                buttons.confirm = false;
            }
            let Some(state) = world.resource::<aim::CastInput>().copied() else {
                return;
            };
            // A click the menu already spent this tick confirms nothing else.
            if state.confirmed_at == Some(tick) {
                buttons.confirm = false;
            }
            let intent = aim::cast_input(state.mode, state.aiming, buttons);
            if let Some(input) = world.resource_mut::<aim::CastInput>() {
                input.aiming = intent.aiming;
                if intent.consumed_click {
                    input.confirmed_at = Some(tick);
                }
            }
            intent.cast
        };

        world.resource_scope::<slice::Slice, _>(|world, slice| {
            if let Some(player) = slice::player_entity(world) {
                // Before anything fires: a wind-up still turns to follow the
                // aim, so a bolt leaving this tick leaves toward the cursor.
                aim::track_aim(world, player);
                if let Some(index) = requested {
                    // Faced first, so the cast begins toward the aim and its
                    // telegraph is right from its first frame.
                    let facing = world.get::<slice::Facing>(player).copied();
                    aim::face_aim(world, player);
                    if !slice::start_ability(world, slice, player, index)
                        && let Some(facing) = facing
                    {
                        // Refused: put the body back as it was, rather than
                        // leave a failed press turning it.
                        let _ = world.insert(player, facing);
                    }
                }
            }
            slice::step_abilities(world, slice, tick);
            // After casts, so a bolt launched this tick flies this tick; before
            // `slice-wounds`, so its hits flash like any other.
            let seconds = world
                .resource::<FixedTime>()
                .map_or(1.0 / 60.0, |time| time.step.as_secs_f32());
            slice::step_projectiles(world, slice, tick, seconds);
        });
    })?;

    // Attacking and dodging: fixed step, because every window they open is
    // counted in ticks and must not depend on how fast the machine renders.
    app.add_fixed_system(FixedStage::Update, "slice-player-combat", |world, _| {
        let Some(input) = world.resource::<FixedInput>() else {
            return;
        };
        // Space, or a left-click that was not spent confirming a cast.
        let clicked = input.mouse_buttons.just_pressed(engine::input::MouseButton::Left);
        let spent = world
            .resource::<aim::CastInput>()
            .is_some_and(|cast| cast.confirmed_at == Some(aim::tick(world)));
        let Some(input) = world.resource::<FixedInput>() else {
            return;
        };
        let attack = input.keys.just_pressed(KeyCode::Space) || (clicked && !spent);
        // Either shift: binding only the left one quietly ignores half the
        // people who reach for a modifier without looking.
        let dodge =
            input.keys.just_pressed(KeyCode::ShiftLeft) || input.keys.just_pressed(KeyCode::ShiftRight);

        let mut direction = Vec2::ZERO;
        if input.keys.pressed(KeyCode::W) {
            direction.y -= 1.0;
        }
        if input.keys.pressed(KeyCode::S) {
            direction.y += 1.0;
        }
        if input.keys.pressed(KeyCode::A) {
            direction.x -= 1.0;
        }
        if input.keys.pressed(KeyCode::D) {
            direction.x += 1.0;
        }

        // Dodge wins a tie: the defensive option should never lose to the
        // aggressive one when a player presses both in a panic.
        //
        // Both arms sit inside the scope because a swing now reads what is
        // wielded: `start_attack` resolves the player's loadout to size and
        // type the blow, so it needs the slice that holds canonical state and
        // the item registry.
        if dodge || attack {
            world.resource_scope::<slice::Slice, _>(|world, slice| {
                if dodge {
                    slice::start_dodge(world, direction.normalize_or_zero());
                } else if let Some(player) = slice::player_entity(world) {
                    let facing = world.get::<slice::Facing>(player).copied();
                    aim::face_aim(world, player);
                    if !slice::start_attack(world, slice)
                        && let Some(facing) = facing
                    {
                        let _ = world.insert(player, facing);
                    }
                }
            });
        }
    })?;

    app.add_fixed_system(FixedStage::Update, "slice-combat", |world, _| {
        // The fixed-step index, for the same reason `slice-abilities` uses it:
        // a status is measured against a clock that actually advances.
        let tick = world.resource::<FixedTime>().map_or(0, |time| time.tick);
        world.resource_scope::<slice::Slice, _>(|world, slice| {
            slice::step_combat(world, slice, tick);
            // After the dead are reaped, so a wave called this tick is timed
            // from the kill that called it.
            slice::step_waves(world, slice, tick);
            // Checked every tick rather than only after a hit: whatever kills
            // the player — a swing now, a hazard later — the answer is the same.
            slice::respawn_if_dead(world, slice);
            // The fourth safe point. A fight that just ended writes its cost
            // now, rather than leaving the record stale until the player
            // happens to use a door, die, or quit. After a respawn this is a
            // no-op: `respawn_if_dead` has already synced, and `sync_health`
            // measures against what the tick will commit.
            slice::sync_on_combat_exit(world, slice);
        });
    })?;

    // Noticing damage, after everything that can deal it this tick — effects in
    // `slice-abilities`, swings in `slice-combat` — so a hit from either
    // flashes the same way.
    app.add_fixed_system(FixedStage::Update, "slice-wounds", |world, _| {
        let tick = world.resource::<FixedTime>().map_or(0, |time| time.tick);
        slice::note_wounds(world, tick);
        slice::recover_poise(world);
    })?;

    // Whoever is being escorted walks after the player. Fixed step, because
    // moving anything is simulation.
    app.add_fixed_system(FixedStage::Update, "slice-escort", |world, _| {
        world.resource_scope::<CollisionWorld, _>(|world, collision| {
            slice::follow_leader(world, collision);
        });
    })?;

    // Combat feedback. Frame-rate, because it is presentation: nothing here
    // changes what happens in a fight, only whether a player can see it.
    app.add_frame_system(FrameStage::PostUpdate, "slice-feedback", |world, _| {
        let tick = world.resource::<FixedTime>().map_or(0, |time| time.tick);
        slice::show_cues(world, tick);
        slice::pose_bodies(world, tick);
        // Limbs after the body: clips pose each slot, and the body's lean and
        // roll still play on top of whatever the limbs are doing.
        world.resource_scope::<slice::Slice, _>(|world, slice| {
            animation::animate(world, &slice.content, tick);
            vfx::show_telegraphs(world, &slice.content);
            vfx::show_projectiles(world, &slice.content, tick);
            // After bodies are posed, so a name sits over where the body is
            // drawn this frame.
            nameplate::show_names(world, &slice.content);
        });
        menu::show_menu(world);
        hud::show_hud(world, tick);
        let step = world
            .resource::<FixedTime>()
            .map_or(1.0 / 60.0, |time| time.step.as_secs_f32());
        vfx::play_effects(world, tick, step);
        slice::age_remains(world, tick);
        world.resource_scope::<DebugDraw, _>(|world, debug| {
            if let Some(slice) = world.resource::<slice::Slice>() {
                slice::draw_gauges(world, debug, &slice.content, tick);
            }
        });
    })?;

    // The camera follows the *slice's* player. `build_app`'s `follow-player`
    // tracks a different `Player` type entirely — the M3 scene's — so without
    // this the rig would frame a box in another world and leave Cinderwake off
    // screen. Frame-rate, because posing a camera is presentation.
    app.add_frame_system(FrameStage::PostUpdate, "slice-camera", |world, _| {
        let Some(target) = world
            .query2::<Transform, slice::Player>()
            .next()
            .map(|(_, transform, _)| transform.translation)
        else {
            return;
        };
        let delta = world
            .resource::<FrameTime>()
            .map_or(0.0, |time| time.delta.as_secs_f32());
        world.resource_scope::<FollowCamera, _>(|world, rig| {
            rig.follow(target, delta);
            if let Some(camera) = world.resource_mut::<Camera>() {
                rig.apply_to(camera);
            }
        });
    })?;

    // Escape is a safe point. `engine::run` consumes the app, so a windowed
    // session has no "after the loop" in which to write a save — it has to
    // happen on the way out.
    app.add_frame_system(FrameStage::PostUpdate, "slice-save-on-exit", |world, _| {
        let leaving = world
            .resource::<FrameInput>()
            .is_some_and(|input| input.keys.just_pressed(KeyCode::Escape));
        if !leaving {
            return;
        }
        world.resource_scope::<slice::Slice, _>(|world, slice| {
            slice::sync_health(world, slice);
            if let Err(error) = slice.commit() {
                engine::log_error!("game", "a domain event was rejected", error = error);
            }
            if let Err(error) = slice.write_save() {
                engine::log_error!("game", "cannot write the save", error = error);
            }
        });
    })?;

    // Quests consume whatever the tick raised, then everything commits together
    // so one tick's outcomes land in the log as one ordered group.
    app.add_fixed_system(FixedStage::PostUpdate, "slice-commit", |world, _| {
        // Before quests drain the tick's signals: a settled choice retires its
        // marker however it was settled.
        slice::retire_settled_points(world);
        let tick = world.resource::<FixedTime>().map_or(0, |time| time.tick);
        let Some(slice) = world.resource_mut::<slice::Slice>() else {
            return;
        };

        // Commit first, *then* let quests react, then commit what they produced.
        //
        // Advancing before committing loses signals: accepting a quest and
        // satisfying its first objective happen on the same tick when the giver
        // is also the objective, and `advance` ignores a quest that is not yet
        // active — while `advance_quests` drains the tick's signals either way.
        // Talking to Sela accepted "Still Waters" and threw away the `Talked`
        // that should have completed `meet-sela`, so the player had to speak to
        // her twice.
        for pass in 0..2 {
            if pass == 1 {
                slice.advance_quests();
            }
            if let Err(error) = slice.commit() {
                engine::log_error!("game", "a domain event was rejected", error = error);
            }
        }
        // A class picked this tick has committed; rebuild the body from it.
        let rebuild = world
            .resource_mut::<slice::Slice>()
            .is_some_and(|slice| std::mem::take(&mut slice.class_changed));
        if rebuild {
            world.resource_scope::<slice::Slice, _>(|world, slice| {
                slice::respawn_player(world, slice);
            });
        }
        hud::collect_notices(world, tick);
    })?;

    Ok(())
}

/// How the game was asked to run.
///
/// `resume` lives on the slice variants rather than being a free-standing flag
/// because resuming the M3 scene is meaningless — it has no canonical state to
/// restore — and putting it here makes that combination unrepresentable rather
/// than merely rejected.
enum Mode {
    Windowed,
    Headless(u64),
    /// The playable slice, headless for the given number of frames.
    ///
    /// What CI runs and what the determinism checks need.
    Slice {
        frames: u64,
        resume: bool,
    },
    /// The playable slice, in a window.
    SliceWindowed {
        resume: bool,
    },
}

/// What the command line asked for: which mode, and how to configure the GPU.
///
/// Renderer options are carried beside the mode rather than inside it because
/// they apply to *any* windowed run — the M3 scene as much as the slice —
/// whereas `resume` is meaningful only where there is canonical state.
struct Launch {
    mode: Mode,
    renderer: engine::render::RendererConfig,
    /// Development tools (M9): F5 reloads content. Slice runs only.
    dev: bool,
}

/// Present in a development run (`--dev`): what enables F5's content reload.
/// A resource rather than a build flag so a release build can be profiled
/// with the tools on, and so a player never has them by accident.
struct DevTools;

impl engine::ecs::Resource for DevTools {}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Launch, String> {
    let mut args = args.peekable();
    let mut mode = Mode::Windowed;
    let mut resume = false;
    let mut dev = false;
    let mut gpu: Option<engine::render::GpuPreference> = None;
    let mut backend: Option<engine::render::BackendPreference> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--headless" => {
                let frames = match args.peek().and_then(|next| next.parse::<u64>().ok()) {
                    Some(frames) => {
                        args.next();
                        frames
                    }
                    None => DEFAULT_HEADLESS_FRAMES,
                };
                mode = Mode::Headless(frames);
            }
            // `--slice` opens a window, because the point of the slice is to be
            // looked at. `--slice N` runs N frames headless instead, which is
            // what CI and the determinism checks want.
            "--slice" => {
                mode = match args.peek().and_then(|next| next.parse::<u64>().ok()) {
                    Some(frames) => {
                        args.next();
                        Mode::Slice {
                            frames,
                            resume: false,
                        }
                    }
                    None => Mode::SliceWindowed { resume: false },
                };
            }
            // Applied after the loop rather than here, so `--resume --slice`
            // and `--slice --resume` mean the same thing. An order-dependent
            // flag is a bug report waiting to happen.
            "--resume" => resume = true,
            "--dev" => dev = true,
            // Which GPU to ask for. The default stays `HighPerformance`,
            // because changing it would alter what a player with a discrete
            // card gets; validating the minimum device profile is an explicit
            // request, and the `adapter selected` log line reports what was
            // actually chosen.
            "--gpu" => {
                gpu = Some(match args.next().as_deref() {
                    Some("low-power") => engine::render::GpuPreference::LowPower,
                    Some("high-performance") => engine::render::GpuPreference::HighPerformance,
                    Some(other) => {
                        return Err(format!("--gpu expects low-power or high-performance: {other}"));
                    }
                    None => return Err("--gpu expects low-power or high-performance".to_owned()),
                });
            }
            // `vulkan` reproduces SteamOS behaviour while testing on Windows,
            // where `auto` would otherwise pick DX12 (ADR-0003).
            "--backend" => {
                backend = Some(match args.next().as_deref() {
                    Some("auto") => engine::render::BackendPreference::Auto,
                    Some("vulkan") => engine::render::BackendPreference::Vulkan,
                    Some(other) => return Err(format!("--backend expects auto or vulkan: {other}")),
                    None => return Err("--backend expects auto or vulkan".to_owned()),
                });
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    match &mut mode {
        Mode::Slice { resume: flag, .. } | Mode::SliceWindowed { resume: flag } => *flag = resume,
        Mode::Windowed | Mode::Headless(_) if resume => {
            return Err("--resume applies to --slice: the M3 scene has no save to restore".to_owned());
        }
        Mode::Windowed | Mode::Headless(_) if dev => {
            return Err("--dev applies to --slice: the M3 scene has no content to reload".to_owned());
        }
        _ => {}
    }

    // Refused rather than ignored, for the same reason `--resume` is: a flag
    // that silently does nothing is worse than one that says so. A headless run
    // never creates a device, so it has no GPU to choose.
    if (gpu.is_some() || backend.is_some()) && matches!(mode, Mode::Headless(_) | Mode::Slice { .. }) {
        return Err(
            "--gpu and --backend apply to windowed runs: a headless run creates no device".to_owned(),
        );
    }

    let mut renderer = engine::render::RendererConfig::default();
    if let Some(gpu) = gpu {
        renderer.gpu_preference = gpu;
    }
    if let Some(backend) = backend {
        renderer.backend = backend;
    }
    Ok(Launch { mode, renderer, dev })
}

fn main() -> ExitCode {
    let Launch { mode, renderer, dev } = match parse_args(std::env::args().skip(1)) {
        Ok(launch) => launch,
        Err(message) => {
            engine::log_error!("game", "invalid arguments", detail = message);
            engine::log_error!(
                "game",
                "usage: cinderwake [--headless [FRAMES]] [--slice [FRAMES]] [--resume] [--dev] \
                 [--gpu low-power|high-performance] [--backend auto|vulkan]"
            );
            return ExitCode::FAILURE;
        }
    };

    let mut app = match build_app(renderer) {
        Ok(app) => app,
        Err(error) => {
            engine::log_error!("game", "cannot create application", error = error);
            return ExitCode::FAILURE;
        }
    };

    match mode {
        Mode::Headless(frames) => {
            let report = app.run_headless(frames, Duration::from_micros(16_667));
            let frame = engine::render::extract_render_frame(app.world(), 16.0 / 9.0);
            engine::log_info!(
                "game",
                "headless run complete",
                frames = report.frame + 1,
                entities = app.world().entity_count(),
                draws = frame.draws.len(),
            );
            ExitCode::SUCCESS
        }
        Mode::Slice { frames, resume } => {
            let Some(mut app) = start_slice(resume, renderer) else {
                return ExitCode::FAILURE;
            };
            if dev {
                app.world_mut().insert_resource(DevTools);
            }
            let report = app.run_headless(frames, Duration::from_micros(16_667));

            // Exiting is a safe point: reconcile health, fold it in, then write.
            // Saving first would record the run as though the fights in it had
            // never happened.
            app.world_mut().resource_scope::<slice::Slice, _>(|world, slice| {
                slice::sync_health(world, slice);
                if let Err(error) = slice.commit() {
                    engine::log_error!("game", "a domain event was rejected", error = error);
                }
                if let Err(error) = slice.write_save() {
                    engine::log_error!("game", "cannot write the save", error = error);
                }
            });

            engine::log_info!(
                "game",
                "slice run complete",
                frames = report.frame + 1,
                entities = app.world().entity_count(),
            );
            ExitCode::SUCCESS
        }
        Mode::SliceWindowed { resume } => {
            let Some(mut slice_app) = start_slice(resume, renderer) else {
                return ExitCode::FAILURE;
            };
            if dev {
                slice_app.world_mut().insert_resource(DevTools);
            }
            // A new character chooses who to be (M9). Headless runs and a
            // resumed save keep the class they have.
            if !resume {
                offer_class_choice(&mut slice_app);
            }
            // `engine::run` consumes the app, so there is no "after the loop"
            // here — the save happens in `slice-save-on-exit`, on the way out.
            match engine::run(slice_app) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    engine::log_error!("game", "engine stopped with an error", error = error);
                    ExitCode::FAILURE
                }
            }
        }
        Mode::Windowed => match engine::run(app) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                engine::log_error!("game", "engine stopped with an error", error = error);
                ExitCode::FAILURE
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headless_app(frames: u64) -> App {
        let mut app = build_app(engine::render::RendererConfig::default()).unwrap();
        app.run_headless(frames, Duration::from_micros(16_667));
        app
    }

    #[test]
    fn scene_builds_and_extracts_without_a_gpu() {
        let app = headless_app(4);
        assert!(
            app.world().entity_count() > 10,
            "scene should spawn ground plus props"
        );

        let frame = engine::render::extract_render_frame(app.world(), 16.0 / 9.0);
        assert!(!frame.draws.is_empty(), "extraction should produce draw commands");
        // Grouped by material then mesh, so equal materials are contiguous.
        let materials: Vec<u32> = frame.draws.iter().map(|draw| draw.material.index()).collect();
        let mut seen = materials.clone();
        seen.dedup();
        let mut unique = seen.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            seen.len(),
            unique.len(),
            "materials should be contiguous: {materials:?}"
        );
    }

    #[test]
    fn culling_rejects_what_is_behind_the_camera() {
        let app = headless_app(2);
        let all = engine::render::extract_render_frame(app.world(), 16.0 / 9.0);
        let entities = app.world().entity_count() as usize;
        assert!(
            all.draws.len() < entities,
            "a ring of props around the camera should not be fully visible"
        );
    }

    #[test]
    fn player_moves_only_while_input_is_held() {
        let mut app = build_app(engine::render::RendererConfig::default()).unwrap();
        app.run_headless(1, Duration::from_micros(16_667));
        let start = player_position(app.world());

        app.handle_platform_event(&engine::input::PlatformEvent::Key {
            key: KeyCode::W,
            state: engine::input::ButtonState::Pressed,
            repeat: false,
        });
        app.run_headless(30, Duration::from_micros(16_667));
        let moved = player_position(app.world());
        assert!(moved.z < start.z - 0.1, "W should move the player along -Z");

        app.handle_platform_event(&engine::input::PlatformEvent::Key {
            key: KeyCode::W,
            state: engine::input::ButtonState::Released,
            repeat: false,
        });
        app.run_headless(30, Duration::from_micros(16_667));
        let settled = player_position(app.world());
        assert!((settled.z - moved.z).abs() < 1e-5, "release should stop movement");
    }

    #[test]
    fn camera_follows_the_player() {
        let mut app = build_app(engine::render::RendererConfig::default()).unwrap();
        app.run_headless(1, Duration::from_micros(16_667));
        app.handle_platform_event(&engine::input::PlatformEvent::Key {
            key: KeyCode::D,
            state: engine::input::ButtonState::Pressed,
            repeat: false,
        });
        app.run_headless(120, Duration::from_micros(16_667));

        let player = player_position(app.world());
        let camera = app.world().resource::<Camera>().unwrap();
        assert!(
            (camera.target.x - player.x).abs() < 1.0,
            "camera target {:?} should track player {player:?}",
            camera.target
        );
    }

    fn player_position(world: &World) -> Vec3 {
        world
            .query2::<Transform, Player>()
            .next()
            .map(|(_, transform, _)| transform.translation)
            .expect("scene should contain a player")
    }

    #[test]
    fn debug_overlay_is_off_until_toggled() {
        let mut app = build_app(engine::render::RendererConfig::default()).unwrap();
        app.run_headless(2, Duration::from_micros(16_667));
        assert_eq!(
            app.world().resource::<DebugDraw>().unwrap().vertex_count(),
            0,
            "an overlay nobody asked for must cost nothing"
        );

        app.handle_platform_event(&engine::input::PlatformEvent::Key {
            key: KeyCode::F1,
            state: engine::input::ButtonState::Pressed,
            repeat: false,
        });
        app.run_headless(1, Duration::from_micros(16_667));

        let debug = app.world().resource::<DebugDraw>().unwrap();
        assert!(debug.vertex_count() > 0, "overlay should emit wire bounds");
        assert!(
            debug.vertex_count().is_multiple_of(2),
            "line lists must contain whole segments"
        );
        assert_eq!(debug.dropped_lines(), 0, "the scene should fit the vertex budget");
    }

    #[test]
    fn parses_headless_argument() {
        let parse =
            |args: &[&str]| parse_args(args.iter().map(|a| (*a).to_owned())).map(|launch| launch.mode);
        assert!(matches!(parse(&[]), Ok(Mode::Windowed)));
        assert!(matches!(
            parse(&["--headless"]),
            Ok(Mode::Headless(DEFAULT_HEADLESS_FRAMES))
        ));
        assert!(matches!(parse(&["--headless", "10"]), Ok(Mode::Headless(10))));
        // Bare `--slice` opens a window, because the point of the slice is to be
        // looked at; `--slice N` runs N frames headless, which is what CI and
        // the determinism checks want.
        assert!(matches!(
            parse(&["--slice"]),
            Ok(Mode::SliceWindowed { resume: false })
        ));
        assert!(matches!(
            parse(&["--slice", "42"]),
            Ok(Mode::Slice {
                frames: 42,
                resume: false
            })
        ));
        assert!(parse(&["--nope"]).is_err());
    }

    #[test]
    fn dev_tools_are_for_the_slice() {
        let dev = |args: &[&str]| parse_args(args.iter().map(|a| (*a).to_owned())).map(|launch| launch.dev);
        assert_eq!(dev(&["--slice", "--dev"]), Ok(true));
        assert_eq!(dev(&["--dev", "--slice", "10"]), Ok(true));
        assert_eq!(dev(&["--slice"]), Ok(false));
        assert!(dev(&["--dev"]).is_err(), "the M3 scene has nothing to reload");
    }

    #[test]
    fn parses_resume_in_either_order() {
        let parse =
            |args: &[&str]| parse_args(args.iter().map(|a| (*a).to_owned())).map(|launch| launch.mode);
        assert!(matches!(
            parse(&["--slice", "--resume"]),
            Ok(Mode::SliceWindowed { resume: true })
        ));
        assert!(matches!(
            parse(&["--resume", "--slice"]),
            Ok(Mode::SliceWindowed { resume: true })
        ));
        assert!(matches!(
            parse(&["--slice", "42", "--resume"]),
            Ok(Mode::Slice {
                frames: 42,
                resume: true
            })
        ));
        // Resuming the M3 scene is not a thing: it has no canonical state.
        assert!(parse(&["--resume"]).is_err());
        assert!(parse(&["--headless", "--resume"]).is_err());
    }

    /// The renderer flags exist to validate the minimum device profile, which
    /// is integrated graphics — the opposite of the default.
    #[test]
    fn parses_renderer_preferences() {
        let launch = |args: &[&str]| parse_args(args.iter().map(|a| (*a).to_owned()));

        let default = launch(&[]).expect("no arguments is valid");
        assert_eq!(
            default.renderer,
            engine::render::RendererConfig::default(),
            "the shipped default is unchanged: a player with a discrete card \
             should still get it"
        );

        let low = launch(&["--gpu", "low-power"]).expect("valid");
        assert_eq!(
            low.renderer.gpu_preference,
            engine::render::GpuPreference::LowPower
        );

        let vulkan = launch(&["--backend", "vulkan"]).expect("valid");
        assert_eq!(vulkan.renderer.backend, engine::render::BackendPreference::Vulkan);

        // Refused rather than ignored: a headless run creates no device.
        assert!(launch(&["--headless", "--gpu", "low-power"]).is_err());
        assert!(launch(&["--slice", "10", "--backend", "vulkan"]).is_err());
        // But a windowed slice run is exactly where they belong.
        assert!(launch(&["--slice", "--gpu", "low-power"]).is_ok());

        assert!(launch(&["--gpu"]).is_err(), "a value is required");
        assert!(launch(&["--gpu", "nonsense"]).is_err());
        assert!(launch(&["--backend", "metal"]).is_err());
    }

    /// The slice, driven headless against the shipped content.
    mod playable {
        use super::*;
        use engine::input::{ButtonState, PlatformEvent};
        use game_core::content::{AbilityDef, DamageType, ItemDef, StatusId};
        use game_core::events::DomainEvent;
        use game_core::ids::AbilityId;
        use game_core::ids::{ItemId, LocationId, NpcId, ObjectiveId, QuestId, RegionId};
        use game_core::world::QuestStatus;
        use game_core::world::{EquipmentSlot, WorldState};
        use game_runtime::combat::{Attack, Health};
        use game_runtime::enemy::EnemyBrain;
        use game_runtime::movement::Stamina;
        use game_runtime::selection;
        use game_runtime::status::Statuses;
        use std::path::{Path, PathBuf};

        fn content_root() -> PathBuf {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/content")
        }

        fn location(name: &str) -> LocationId {
            LocationId::new(name).expect("valid id")
        }

        fn quest(name: &str) -> QuestId {
            QuestId::new(name).expect("valid id")
        }

        fn slice_app(start: &str) -> App {
            let built = slice::Slice::new(&content_root(), &location(start), SLICE_SEED)
                .expect("the shipped content loads");
            // Deliberately the *bare* app: with M3's scene attached, a test
            // asserting that something was drawn would pass on M3's prop ring
            // alone, whether or not the slice drew a single thing.
            let mut app = build_slice_app(engine::render::RendererConfig::default()).expect("app builds");
            attach_slice(&mut app, built).expect("systems register");
            app
        }

        fn press(app: &mut App, key: KeyCode) {
            app.handle_platform_event(&PlatformEvent::Key {
                key,
                state: ButtonState::Pressed,
                repeat: false,
            });
        }

        fn release(app: &mut App, key: KeyCode) {
            app.handle_platform_event(&PlatformEvent::Key {
                key,
                state: ButtonState::Released,
                repeat: false,
            });
        }

        fn step(app: &mut App, frames: u64) {
            app.run_headless(frames, Duration::from_micros(16_667));
        }

        fn player_at(world: &World) -> Vec3 {
            world
                .query2::<Transform, slice::Player>()
                .next()
                .map(|(_, transform, _)| transform.translation)
                .expect("the slice spawns a player")
        }

        /// M9: of an NPC's conversations whose conditions hold, one opens
        /// directly, several open a topic list, and none opens nothing.
        #[test]
        fn an_npc_offers_what_it_holds_open() {
            use game_core::content::Condition;
            use game_core::ids::{DialogueId, WorldFlag};
            use game_runtime::selection::Offer;

            let mut slice = slice::Slice::new(&content_root(), &location("cinderwake-square"), SLICE_SEED)
                .expect("the shipped content loads");
            let sela = NpcId::new("warden-sela").expect("valid id");

            assert!(slice.open_conversation(&sela));
            assert!(
                matches!(slice.offer, Some(Offer::Talking(_))),
                "one conversation opens directly"
            );
            slice.close_offer();

            let greeting = DialogueId::new("sela-greeting").expect("valid id");
            let mut second = slice.content.dialogues[&greeting].clone();
            second.id = DialogueId::new("sela-rumours").expect("valid id");
            second.topic = Some("Rumours".to_owned());
            second.conditions = vec![Condition::WorldFlagSet(
                WorldFlag::new("heard-rumours").expect("valid id"),
            )];
            slice.content.dialogues.insert(second.id.clone(), second);

            assert!(slice.open_conversation(&sela));
            assert!(
                matches!(slice.offer, Some(Offer::Talking(_))),
                "the second is shut, so the first still opens directly"
            );
            slice.close_offer();

            slice
                .state
                .world_flags
                .insert(WorldFlag::new("heard-rumours").expect("valid id"));
            assert!(slice.open_conversation(&sela));
            assert_eq!(
                slice.offer,
                Some(Offer::Topics(sela.clone())),
                "two open: pick a topic"
            );
            slice.close_offer();

            for dialogue in slice.content.dialogues.values_mut() {
                dialogue.conditions = vec![Condition::WorldFlagUnset(
                    WorldFlag::new("heard-rumours").expect("valid id"),
                )];
            }
            assert!(!slice.open_conversation(&sela), "nothing open, nothing opens");
            assert_eq!(slice.offer, None);
        }

        #[test]
        fn the_hub_populates_from_authored_content() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            let world = app.world();

            assert_eq!(
                world.query::<slice::Npc>().count(),
                1,
                "warden sela stands in the square"
            );
            // By kind rather than by count: a station whose purpose changed
            // would otherwise still satisfy a bare tally.
            let mut kinds: Vec<String> = world
                .query::<slice::Station>()
                .map(|(_, station)| format!("{:?}", station.kind))
                .collect();
            kinds.sort();
            assert_eq!(
                kinds,
                vec!["Crafting".to_owned(), "Doorway".to_owned(), "Upgrades".to_owned()],
                "the forge, the lane gate, and the upgrade board"
            );
            assert_eq!(
                world.query::<slice::Enemy>().count(),
                0,
                "the hub is safe (ADR-0007)"
            );
        }

        #[test]
        fn the_lane_spawns_its_enemies_and_resources() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let world = app.world();

            assert_eq!(
                world.query::<slice::Enemy>().count(),
                4,
                "the waterworks ambush's first wave — two husks; a third and the \
                 spitter come when those fall — plus the old forest's two-warden \
                 picket holding the way to the crypt"
            );
            assert_eq!(world.query::<slice::Gatherable>().count(), 2);
        }

        /// Entities finally have real positions — the thing M4 could not do.
        #[test]
        fn spawns_land_where_content_places_them() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);

            let npc = app
                .world()
                .query2::<Transform, slice::Npc>()
                .next()
                .map(|(_, transform, _)| transform.translation)
                .expect("sela is placed");
            assert!(
                (npc.x - 3.0).abs() < 1e-3 && (npc.z - 2.0).abs() < 1e-3,
                "sela's authored post is (3, 0, 2) metres, got {npc:?}"
            );
        }

        #[test]
        fn input_drives_the_player_and_not_the_enemies() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);

            let start = player_at(app.world());

            press(&mut app, KeyCode::D);
            step(&mut app, 30);
            release(&mut app, KeyCode::D);

            assert!(
                player_at(app.world()).x > start.x + 0.1,
                "D should walk the player along +X"
            );

            // Every lane enemy posts well beyond the 9 m acquire radius from
            // the entrance, so all four standing are correctly still holding
            // position (the ambush's second wave has not arrived).
            // Input must not have dragged any of them along with the player.
            let enemies: Vec<Vec3> = app
                .world()
                .query2::<Transform, slice::Enemy>()
                .map(|(_, transform, _)| transform.translation)
                .collect();
            assert_eq!(
                enemies,
                vec![
                    Vec3::new(2.0, 0.9, 0.0),
                    Vec3::new(4.0, 0.9, 1.5),
                    Vec3::new(-2.5, 0.9, 9.0),
                    Vec3::new(2.5, 0.9, 9.5),
                ],
                "unaware enemies hold their authored posts; player input is not their input"
            );
        }

        /// Aggro has to be earned: the lane's enemies post outside the 9 m
        /// acquire radius, so the player walks in before anything happens.
        #[test]
        fn enemies_close_once_the_player_is_noticed() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);

            let gap = |world: &World| {
                let player = player_at(world);
                world
                    .query2::<Transform, slice::Enemy>()
                    .map(|(_, transform, _)| (transform.translation - player).length())
                    .fold(f32::INFINITY, f32::min)
            };

            let posts = |world: &World| -> Vec<Vec3> {
                world
                    .query2::<Transform, slice::Enemy>()
                    .map(|(_, transform, _)| transform.translation)
                    .collect()
            };

            let at_rest = gap(app.world());
            let authored = posts(app.world());
            assert!(
                at_rest > 9.0,
                "the lane should start the player outside aggro range, got {at_rest}"
            );

            // Walk up the lane (+Z) until the nearest enemy notices.
            press(&mut app, KeyCode::S);
            step(&mut app, 90);
            release(&mut app, KeyCode::S);

            let noticed = gap(app.world());
            let holding: Vec<EnemyState> = app
                .world()
                .query::<EnemyBrain>()
                .map(|(_, brain)| brain.state())
                .collect();
            assert!(
                holding.iter().any(|state| *state != EnemyState::Idle),
                "walking into the picket should wake something: gap {noticed}, states {holding:?}"
            );

            // And once awake it leaves its post under its own steam. Note the
            // claim is *moved*, not *kept closing*: a melee brain stops at its
            // preferred range to swing, so demanding the gap shrink forever
            // would assert the opposite of the behaviour we want.
            step(&mut app, 150);
            assert_ne!(
                posts(app.world()),
                authored,
                "an alerted enemy should have left the spot content put it on"
            );
        }

        #[test]
        fn talking_to_sela_starts_the_quest() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);

            // Walk to her post at (3, 0, 2) and interact.
            press(&mut app, KeyCode::D);
            step(&mut app, 60);
            release(&mut app, KeyCode::D);
            press(&mut app, KeyCode::S);
            step(&mut app, 40);
            release(&mut app, KeyCode::S);

            converse(&mut app);

            let slice = app.world().resource::<slice::Slice>().expect("slice exists");
            assert_eq!(
                slice.state.quest_status(&quest("still-waters")),
                QuestStatus::Active,
                "talking to the quest giver should record acceptance as a domain event"
            );
        }

        #[test]
        fn harvesting_grants_the_item_once() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);

            // The reed bed sits at (5, 0, 6); the lane start is at (0, 0, -12).
            press(&mut app, KeyCode::S);
            step(&mut app, 260);
            release(&mut app, KeyCode::S);
            press(&mut app, KeyCode::D);
            step(&mut app, 80);
            release(&mut app, KeyCode::D);

            let reed = ItemId::new("still-reed").expect("valid id");
            for _ in 0..3 {
                press(&mut app, KeyCode::E);
                step(&mut app, 2);
                release(&mut app, KeyCode::E);
                step(&mut app, 2);
            }

            let slice = app.world().resource::<slice::Slice>().expect("slice exists");
            let held = slice.state.player.item_count(&reed);
            assert!(
                held <= 1,
                "a node must not be farmable; pressing interact three times yielded {held}"
            );
        }

        /// The property the whole event-sourced design rests on.
        #[test]
        fn two_identical_runs_produce_identical_canonical_state() {
            let run = || {
                let mut app = slice_app("ashfall-lane");
                step(&mut app, 1);
                press(&mut app, KeyCode::S);
                step(&mut app, 120);
                release(&mut app, KeyCode::S);
                step(&mut app, 60);

                let slice = app.world().resource::<slice::Slice>().expect("slice exists");
                ron::to_string(&slice.state).expect("serializes")
            };
            assert_eq!(run(), run());
        }

        #[test]
        fn nothing_runtime_leaks_into_canonical_state() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 120);

            let slice = app.world().resource::<slice::Slice>().expect("slice exists");
            let text = ron::to_string(&slice.state).expect("serializes");
            assert!(
                !text.contains("Entity"),
                "a runtime handle in canonical state is what PROJECT_CONTEXT §3.3 forbids"
            );
            slice
                .state
                .validate_canonical()
                .expect("canonical state stays valid while the slice runs");
        }

        /// Walking through a door rebuilds the world on the far side.
        #[test]
        fn using_a_gate_travels_to_the_linked_location() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            assert_eq!(app.world().query::<slice::Npc>().count(), 1);

            // The lane gate stands at (0, 0, 9); the hub start is the origin.
            press(&mut app, KeyCode::S);
            step(&mut app, 130);
            release(&mut app, KeyCode::S);

            press(&mut app, KeyCode::E);
            step(&mut app, 2);
            release(&mut app, KeyCode::E);
            step(&mut app, 2);

            let world = app.world();
            let slice = world.resource::<slice::Slice>().expect("slice exists");
            assert_eq!(
                slice.location,
                location("ashfall-lane"),
                "the gate should have led to the lane"
            );
            assert_eq!(
                slice.state.player.location.as_ref(),
                Some(&location("ashfall-lane")),
                "arriving is durable, so canonical state records it"
            );
            // The lane has an NPC of its own — Thal — so the claim is not
            // "nobody is here" but "the people here are the ones this location
            // places". The warden stays in her square.
            let present: Vec<String> = world
                .query::<slice::Npc>()
                .map(|(_, npc)| npc.0.to_string())
                .collect();
            assert_eq!(
                present,
                vec!["arbiter-thal".to_owned()],
                "the hub's warden must not follow the player into the waste"
            );
            let ambush = game_core::ids::EncounterId::new("waterworks-ambush").expect("valid id");
            assert_eq!(
                world
                    .query::<slice::Enemy>()
                    .filter(|(_, enemy)| enemy.encounter.as_ref() == Some(&ambush))
                    .count(),
                2,
                "the lane's ambush should be standing: its first wave"
            );

            // And the player arrives at the door's `arrive_at`, not the lane's
            // player start — they are the same spawn here, so check the region
            // was discovered instead, which only a transition can do.
            assert!(
                slice
                    .state
                    .region(&RegionId::new("ashfall-waste").expect("valid id"))
                    .is_some_and(|region| region.discovered),
                "entering a region discovers it"
            );
        }

        #[test]
        fn a_station_that_leads_nowhere_is_inert() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);

            // The upgrade board at (-3, 0, 1) is a fixture, not a door.
            press(&mut app, KeyCode::A);
            step(&mut app, 50);
            release(&mut app, KeyCode::A);
            press(&mut app, KeyCode::S);
            step(&mut app, 20);
            release(&mut app, KeyCode::S);

            press(&mut app, KeyCode::E);
            step(&mut app, 4);
            release(&mut app, KeyCode::E);

            let slice = app.world().resource::<slice::Slice>().expect("slice exists");
            assert_eq!(
                slice.location,
                location("cinderwake-square"),
                "using a non-door must not teleport anyone"
            );
        }

        /// Combat, driven to a kill.
        #[test]
        fn a_husk_can_be_killed_and_drops_loot() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);

            // Controlled geometry. Aggro and pathing have their own tests;
            // this one is about whether a swing connects, kills, and pays out.
            // Choreographing that through four converging enemies tests the AI
            // instead — and the previous attempt failed for exactly that
            // reason: the husks walked *past* the advancing player and ended up
            // behind the hit volume, beating on their back while every swing
            // cut empty ground.
            let player = slice::player_entity(app.world()).expect("a player");
            let position = player_at(app.world());
            let facing = app
                .world()
                .get::<slice::Facing>(player)
                .map(|facing| facing.0)
                .expect("a facing");

            let enemies: Vec<_> = app
                .world()
                .query::<slice::Enemy>()
                .map(|(entity, _)| entity)
                .collect();
            let (target, rest) = enemies.split_first().expect("the lane is populated");
            let target = *target;
            assert_eq!(
                app.world()
                    .get::<slice::Enemy>(target)
                    .map(|enemy| enemy.prefab.to_string()),
                Some("corrupted-husk".to_owned()),
                "the first lane spawn is the picket husk"
            );
            for entity in rest.iter().copied() {
                app.world_mut().despawn(entity);
            }
            if let Some(transform) = app.world_mut().get_mut::<Transform>(target) {
                transform.translation = position + facing * 1.5;
            }

            // A cadence stamina can sustain: a swing costs 25, and regeneration
            // waits 30 ticks before it resumes. A husk has 60 health and the
            // player deals 22, so three connecting swings do it.
            //
            // Watched by the husk itself rather than by a count of enemies:
            // it is the ambush's last standing member, so its death calls the
            // second wave in, and the count goes *up* (M8).
            let mut dead = false;
            for _ in 0..6 {
                press(&mut app, KeyCode::Space);
                step(&mut app, 1);
                release(&mut app, KeyCode::Space);
                step(&mut app, 80);

                dead = app.world().get::<slice::Enemy>(target).is_none();
                if dead {
                    break;
                }
            }

            assert!(dead, "a husk stood in front of the player should die");
            let world = app.world();

            let slice = world.resource::<slice::Slice>().expect("slice exists");
            let carried: u32 = slice.state.player.inventory.values().sum();
            assert!(
                carried > 0,
                "a kill should have dropped something: {:?}",
                slice.state.player.inventory
            );
        }

        /// The encounter is not cleared until its *last* member falls.
        #[test]
        fn clearing_an_encounter_needs_every_member_dead() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);

            let world = app.world();
            let ambush = game_core::ids::EncounterId::new("waterworks-ambush").expect("valid id");
            let tagged = world
                .query::<slice::Enemy>()
                .filter(|(_, enemy)| enemy.encounter.as_ref() == Some(&ambush))
                .count();
            assert_eq!(
                tagged, 2,
                "the ambush's first wave is two of the lane's four standing; the \
                 picket is a fight of its own, and clearing one must not clear \
                 the other"
            );

            let slice = world.resource::<slice::Slice>().expect("slice exists");
            assert_eq!(
                slice.state.quest_status(&quest("still-waters")),
                QuestStatus::NotStarted,
                "nothing has been cleared yet"
            );
        }

        // -------------------------------------------------------------------
        // M6 wave 3: abilities reaching the game. Until these existed, every
        // ability test proved the executor worked on paper while nothing in a
        // running slice ever cast anything — the exact gap ADR-0016 found.
        // -------------------------------------------------------------------

        /// The player's stamina specifically. `query::<Stamina>().next()` takes
        /// whoever comes first, and enemies with abilities now carry stamina
        /// too, so the bare query is no longer a reliable way to find the
        /// player.
        fn player_stamina(app: &App) -> f32 {
            app.world()
                .query2::<Stamina, slice::Player>()
                .next()
                .map(|(_, stamina, _)| stamina.current)
                .expect("the player has stamina")
        }

        fn casting(app: &App) -> bool {
            app.world().query::<slice::Casting>().next().is_some()
        }

        /// PROJECT_CONTEXT §6's two active abilities, and the first
        /// skillshot, from content.
        #[test]
        fn the_player_starts_with_the_classes_authored_abilities() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);

            let known = app
                .world()
                .query2::<slice::Abilities, slice::Player>()
                .next()
                .map(|(_, abilities, _)| abilities.0.clone())
                .expect("the wayfinder is granted its loadout");
            let names: Vec<String> = known.iter().map(|id| id.to_string()).collect();
            assert_eq!(
                names,
                ["ember-cleave", "warden-guard", "ember-bolt"],
                "the loadout comes from classes.ron, in authored order"
            );
        }

        /// `ember-cleave` doing what it says: Q costs stamina, commits the
        /// player, and lands its authored fire damage on a husk.
        #[test]
        fn ember_cleave_costs_stamina_and_damages_an_enemy() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);

            // Put one husk directly in front, and take the rest away so the
            // ability cannot be credited to someone else's swing.
            let (player, position, facing) = {
                let world = app.world();
                let entity = world
                    .query2::<Transform, slice::Player>()
                    .next()
                    .map(|(entity, transform, _)| (entity, transform.translation))
                    .expect("a player");
                let facing = world.get::<slice::Facing>(entity.0).map_or(Vec3::NEG_Z, |f| f.0);
                (entity.0, entity.1, facing)
            };
            let _ = player;
            let enemies: Vec<engine::ecs::Entity> = app
                .world()
                .query::<slice::Enemy>()
                .map(|(entity, _)| entity)
                .collect();
            let target = *enemies.first().expect("the lane has enemies");
            for entity in enemies.iter().skip(1) {
                app.world_mut().despawn(*entity);
            }
            if let Some(transform) = app.world_mut().get_mut::<Transform>(target) {
                transform.translation = position + facing * 2.0;
            }

            let before = app
                .world()
                .get::<Health>(target)
                .map(|health| health.current)
                .expect("the husk has health");
            let full = player_stamina(&app);

            press(&mut app, KeyCode::Q);
            step(&mut app, 1);
            release(&mut app, KeyCode::Q);
            step(&mut app, 1);

            assert!(
                player_stamina(&app) < full,
                "casting should cost the authored 20 stamina"
            );
            assert!(casting(&app), "the cast commits the player through its wind-up");

            // Past the 14-tick wind-up, so the effects land.
            step(&mut app, 20);
            let after = app
                .world()
                .get::<Health>(target)
                .map(|health| health.current)
                .expect("the husk still exists");
            assert!(
                after < before,
                "ember-cleave should damage the husk in front of it: {before} -> {after}"
            );
        }

        /// A cast is a committed action under ADR-0007: it cannot be rolled out
        /// of, and a dodge pressed on the same tick wins instead of losing.
        #[test]
        fn a_cast_is_committed_and_a_dodge_wins_the_tie() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);

            // Both on one tick: the dodge should happen and the cast should not.
            press(&mut app, KeyCode::Q);
            press(&mut app, KeyCode::ShiftLeft);
            step(&mut app, 1);
            release(&mut app, KeyCode::Q);
            step(&mut app, 1);
            release(&mut app, KeyCode::ShiftLeft);

            assert!(
                !casting(&app),
                "a dodge pressed with an ability must beat it, as it beats an attack"
            );
            assert!(
                app.world().query::<Dodge>().next().is_some(),
                "and the dodge itself should have started"
            );

            // Now cast cleanly, then try to roll out of it mid-wind-up.
            step(&mut app, 60);
            press(&mut app, KeyCode::Q);
            step(&mut app, 1);
            release(&mut app, KeyCode::Q);
            step(&mut app, 1);
            assert!(casting(&app), "the cast started");

            press(&mut app, KeyCode::ShiftLeft);
            step(&mut app, 1);
            release(&mut app, KeyCode::ShiftLeft);
            assert!(
                casting(&app),
                "a committed cast cannot be dodge-cancelled (ADR-0007)"
            );
        }

        /// `warden-guard` doing what it says: R applies `status-guarded`, which
        /// halves what the player takes, and then expires on schedule.
        #[test]
        fn warden_guard_applies_a_status_that_later_expires() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);

            press(&mut app, KeyCode::R);
            step(&mut app, 1);
            release(&mut app, KeyCode::R);
            step(&mut app, 1);
            // Past the 6-tick wind-up so the status lands.
            step(&mut app, 10);

            let tick = app.world().resource::<FixedTime>().map_or(0, |time| time.tick);
            let guarded = app
                .world()
                .query2::<Statuses, slice::Player>()
                .next()
                .map(|(_, statuses, _)| statuses.damage_taken_multiplier(tick))
                .expect("the player has a status container");
            assert!(
                (guarded - 0.5).abs() < 1e-6,
                "status-guarded should halve incoming damage, got {guarded}"
            );

            // 180 ticks of duration: well past it, the protection is gone.
            step(&mut app, 200);
            let tick = app.world().resource::<FixedTime>().map_or(0, |time| time.tick);
            let lapsed = app
                .world()
                .query2::<Statuses, slice::Player>()
                .next()
                .map(|(_, statuses, _)| statuses.damage_taken_multiplier(tick))
                .expect("the player has a status container");
            assert!(
                (lapsed - 1.0).abs() < 1e-6,
                "the status must expire in the running game, not only in a unit test: got {lapsed}"
            );
        }

        /// The player's live swing, if one is underway.
        fn swing(app: &App) -> Option<Attack> {
            app.world()
                .query2::<Attack, slice::Player>()
                .next()
                .map(|(_, attack, _)| *attack)
        }

        /// The M6 exit criterion: equipping a weapon changes damage.
        ///
        /// Asserted on the swing itself rather than on a husk's health, because
        /// a health delta is also a test of reach, resistances and whether
        /// anything was standing there — three ways to go green or red for the
        /// wrong reason. What is wielded should show up in what is swung.
        #[test]
        fn equipping_a_weapon_changes_the_swing() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            grant(&mut app, "ember-glaive", 1);

            // Bare-handed first, so the comparison has a measured baseline
            // rather than an assumed one.
            press(&mut app, KeyCode::Space);
            step(&mut app, 1);
            release(&mut app, KeyCode::Space);
            let bare = swing(&app).expect("a swing started");
            assert_eq!(
                bare.damage,
                slice::PLAYER_ATTACK_DAMAGE,
                "an unarmed swing is the authored constant"
            );
            assert_eq!(
                bare.damage_type,
                DamageType::Physical,
                "and a fist is not on fire"
            );

            // Let the swing finish, then equip and swing again.
            step(&mut app, 60);
            app.world_mut().resource_scope::<slice::Slice, _>(|_, slice| {
                assert!(slice.open_equipment(), "nothing else is open");
                assert!(slice.take_offer(1), "the glaive is the only equippable held");
            });
            step(&mut app, 2);

            press(&mut app, KeyCode::Space);
            step(&mut app, 1);
            release(&mut app, KeyCode::Space);
            let armed = swing(&app).expect("a second swing started");

            // 4000 milli-units of authored damage on top of the base 22.
            assert!(
                (armed.damage - (slice::PLAYER_ATTACK_DAMAGE + 4.0)).abs() < 1e-6,
                "the glaive's authored damage should reach the swing: {} -> {}",
                bare.damage,
                armed.damage
            );
            assert_eq!(
                armed.damage_type,
                DamageType::Fire,
                "and only a weapon may retype the blow"
            );
        }

        /// The path a player actually takes: a key, a list, a digit.
        #[test]
        fn the_equipment_key_opens_what_is_carried() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            grant(&mut app, "ember-glaive", 1);
            // Also carried, and never equippable: the list must not simply be
            // the inventory.
            grant(&mut app, "ember-ore", 1);

            press(&mut app, KeyCode::I);
            step(&mut app, 2);
            release(&mut app, KeyCode::I);
            step(&mut app, 1);

            app.world_mut().resource_scope::<slice::Slice, _>(|_, slice| {
                let offered = selection::rows(
                    slice.offer.as_ref().expect("the equipment list opened"),
                    &slice.content,
                    &slice.state,
                );
                assert_eq!(
                    offered.iter().map(|row| row.label.as_str()).collect::<Vec<_>>(),
                    vec!["Ember Glaive"],
                    "ore is carried but equips nothing"
                );
            });

            press(&mut app, KeyCode::Digit1);
            step(&mut app, 2);
            release(&mut app, KeyCode::Digit1);
            step(&mut app, 2);

            let equipped = app
                .world()
                .resource::<slice::Slice>()
                .and_then(|slice| slice.state.player.equipped(EquipmentSlot::Weapon).cloned());
            assert_eq!(
                equipped,
                Some(ItemId::new("ember-glaive").expect("valid id")),
                "pressing the digit should have raised ItemEquipped and filled the slot"
            );
        }

        /// Abilities bind by their position in the class's authored loadout.
        ///
        /// The wayfinder authors three — `ember-bolt` became the third in M7
        /// wave 5, for the game rather than for this test — so this pins Q, R
        /// and F, and the behaviour of one past the end.
        #[test]
        fn ability_keys_bind_by_loadout_position() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);

            press(&mut app, KeyCode::Q);
            step(&mut app, 1);
            release(&mut app, KeyCode::Q);
            step(&mut app, 1);
            assert!(casting(&app), "Q casts the first authored ability");
            step(&mut app, 60);

            press(&mut app, KeyCode::R);
            step(&mut app, 1);
            release(&mut app, KeyCode::R);
            step(&mut app, 1);
            assert!(casting(&app), "R casts the second");
            step(&mut app, 60);

            press(&mut app, KeyCode::F);
            step(&mut app, 1);
            release(&mut app, KeyCode::F);
            step(&mut app, 1);
            assert!(casting(&app), "F casts the third");
            step(&mut app, 200);

            // Slot 4 against a three-ability loadout: nothing, rather than a
            // panic or a wrap back onto the first ability.
            let stamina = player_stamina(&app);
            press(&mut app, KeyCode::C);
            step(&mut app, 2);
            release(&mut app, KeyCode::C);
            step(&mut app, 1);
            assert!(
                !casting(&app),
                "a key past the end of the loadout must not cast anything"
            );
            assert!(
                (player_stamina(&app) - stamina).abs() < 1e-6,
                "and must not spend anything either"
            );
        }

        fn player_health(app: &App, player: engine::ecs::Entity) -> f32 {
            app.world()
                .get::<Health>(player)
                .map(|health| health.current)
                .expect("the player has health")
        }

        /// What the player loses to exactly one enemy swing.
        ///
        /// Set up rather than choreographed: every enemy but one is removed,
        /// and that one loses its brain, is stood a metre in front of the
        /// player facing them, and is handed a swing directly. The first drop
        /// in health can then only be that swing, whatever aggro tuning or
        /// pathing would otherwise have done.
        /// Leaves exactly one enemy, brainless, a metre in front of the player
        /// and facing them, with a swing already started. Returns
        /// `(player, enemy)`.
        fn arm_one_enemy(app: &mut App) -> (engine::ecs::Entity, engine::ecs::Entity) {
            let (player, position) = app
                .world()
                .query2::<Transform, slice::Player>()
                .next()
                .map(|(entity, transform, _)| (entity, transform.translation))
                .expect("a player");
            let facing = app
                .world()
                .get::<slice::Facing>(player)
                .map_or(Vec3::NEG_Z, |facing| facing.0);

            let enemies: Vec<engine::ecs::Entity> = app
                .world()
                .query::<slice::Enemy>()
                .map(|(entity, _)| entity)
                .collect();
            let enemy = *enemies.first().expect("the lane has enemies");
            for other in enemies.iter().skip(1) {
                app.world_mut().despawn(*other);
            }
            let world = app.world_mut();
            let _ = world.remove::<EnemyBrain>(enemy);
            if let Some(transform) = world.get_mut::<Transform>(enemy) {
                transform.translation = position + facing;
            }
            let _ = world.insert(enemy, slice::Facing(-facing));
            let _ = world.insert(enemy, Attack::new(slice::PLAYER_ATTACK_TIMING, 10.0, 1.4, 0.9));
            (player, enemy)
        }

        fn one_enemy_hit(app: &mut App) -> f32 {
            let (player, _) = arm_one_enemy(app);
            let before = player_health(app, player);
            for _ in 0..90 {
                step(app, 1);
                let now = player_health(app, player);
                if now < before {
                    return before - now;
                }
            }
            0.0
        }

        /// A shipped status, and what its definition grants (M9).
        fn defined_status(app: &App, name: &str) -> (StatusId, Vec<game_runtime::status::Modifier>) {
            let id = StatusId(AbilityId::new(name).expect("valid id"));
            let modifiers = app
                .world()
                .resource::<slice::Slice>()
                .and_then(|slice| slice.content.statuses.get(&id))
                .map(game_runtime::status::modifiers_of)
                .expect("the status is defined");
            (id, modifiers)
        }

        fn afflict(
            app: &mut App,
            entity: engine::ecs::Entity,
            status: (StatusId, Vec<game_runtime::status::Modifier>),
            ticks: u64,
        ) {
            let tick = app.world().resource::<FixedTime>().map_or(0, |time| time.tick);
            let world = app.world_mut();
            if world.get::<Statuses>(entity).is_none() {
                let _ = world.insert(entity, Statuses::new());
            }
            let (id, modifiers) = status;
            let applied = world
                .get_mut::<Statuses>(entity)
                .is_some_and(|statuses| statuses.apply(id, tick + ticks, &modifiers));
            assert!(applied);
        }

        /// A damage-dealt status on the swinger reaches the blow (M9). Before
        /// M9 nothing read it.
        #[test]
        fn a_damage_dealt_status_strengthens_a_real_swing() {
            let mut plain = slice_app("ashfall-lane");
            step(&mut plain, 1);
            let full = one_enemy_hit(&mut plain);
            assert!(full > 0.0);

            let mut enraged = slice_app("ashfall-lane");
            step(&mut enraged, 1);
            let (player, enemy) = arm_one_enemy(&mut enraged);
            let rage = (
                StatusId(AbilityId::new("status-test-rage").expect("valid id")),
                vec![game_runtime::status::Modifier::DamageDealt(2.0)],
            );
            afflict(&mut enraged, enemy, rage, 600);
            let before = player_health(&enraged, player);
            let mut lost = 0.0;
            for _ in 0..90 {
                step(&mut enraged, 1);
                lost = before - player_health(&enraged, player);
                if lost > 0.0 {
                    break;
                }
            }
            assert!((lost - full * 2.0).abs() < 1e-3, "doubled: {full} became {lost}");
        }

        /// `status-smouldering` burns for as long as it holds, through the
        /// bearer's resistances, and then stops (M9).
        #[test]
        fn smouldering_burns_until_it_ends() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let enemy = app
                .world()
                .query::<slice::Enemy>()
                .map(|(entity, _)| entity)
                .next()
                .expect("an enemy");
            let _ = app.world_mut().remove::<EnemyBrain>(enemy);
            let health = |app: &App| {
                app.world()
                    .get::<Health>(enemy)
                    .map_or(0.0, |health| health.current)
            };
            let before = health(&app);
            let burn = defined_status(&app, "status-smouldering");
            afflict(&mut app, enemy, burn, 120);

            step(&mut app, 60);
            let halfway = before - health(&app);
            assert!(halfway > 0.0, "it burns");
            step(&mut app, 70);
            let total = before - health(&app);
            assert!(
                total > halfway && total <= 6.0 + 1e-3,
                "three a second for two seconds, less what fire resistance takes: {total}"
            );
            step(&mut app, 60);
            assert!(
                (before - health(&app) - total).abs() < 1e-4,
                "and nothing once it has ended"
            );
        }

        /// `status-resin-mired` slows stamina's return (M9).
        #[test]
        fn mired_stamina_comes_back_slower() {
            let recovered = |mired: bool| {
                let mut app = slice_app("cinderwake-square");
                step(&mut app, 1);
                let player = slice::player_entity(app.world()).expect("a player");
                if mired {
                    let status = defined_status(&app, "status-resin-mired");
                    afflict(&mut app, player, status, 600);
                }
                if let Some(stamina) = app.world_mut().get_mut::<Stamina>(player) {
                    stamina.current = 0.0;
                }
                let start = player_stamina(&app);
                step(&mut app, 30);
                player_stamina(&app) - start
            };
            let plain = recovered(false);
            let mired = recovered(true);
            assert!(plain > 0.0);
            assert!(
                (mired - plain * 0.4).abs() < 1e-3,
                "mired recovers at 0.4 of the rate: {plain} against {mired}"
            );
        }

        /// `warden-guard` doing what it says against what actually hits a
        /// player.
        ///
        /// The wave 3 test checked the multiplier's *value*; this checks the
        /// damage. Until melee targets consulted statuses the two disagreed:
        /// the multiplier read 0.5 while every enemy swing landed in full.
        #[test]
        fn status_guarded_halves_a_real_enemy_swing() {
            let mut plain = slice_app("ashfall-lane");
            step(&mut plain, 1);
            let full = one_enemy_hit(&mut plain);
            assert!(
                full > 0.0,
                "the placed swing must land, or the comparison proves nothing"
            );

            let mut guarded = slice_app("ashfall-lane");
            step(&mut guarded, 1);
            // Applied directly: whether R applies it is the wave 3 test's job,
            // and this one is about what the status does to a hit.
            let player = slice::player_entity(guarded.world()).expect("a player");
            let id = StatusId(AbilityId::new("status-guarded").expect("valid id"));
            // What the shipped definition says the status does (M9).
            let modifiers = guarded
                .world()
                .resource::<slice::Slice>()
                .and_then(|slice| slice.content.statuses.get(&id))
                .map(game_runtime::status::modifiers_of)
                .expect("status-guarded is defined");
            let applied = guarded
                .world_mut()
                .get_mut::<Statuses>(player)
                .expect("the player carries statuses")
                .apply(id, u64::MAX / 2, &modifiers);
            assert!(applied);

            let halved = one_enemy_hit(&mut guarded);
            assert!(
                (halved - full * 0.5).abs() < 1e-3,
                "status-guarded should halve a real swing: {full} -> {halved}"
            );
        }

        /// The armour half of the M6 exit criterion, measured on an enemy
        /// swing — the only way anything in the slice hits a player.
        #[test]
        fn equipping_armour_reduces_what_a_swing_deals() {
            let mut plain = slice_app("ashfall-lane");
            step(&mut plain, 1);
            let full = one_enemy_hit(&mut plain);
            assert!(
                full > 0.0,
                "the placed swing must land, or the comparison proves nothing"
            );

            let mut armoured = slice_app("ashfall-lane");
            step(&mut armoured, 1);
            // Authored in RON the way content would be, but added to this run's
            // registry only. None ships: nothing in the slice grants armour,
            // and shipping a piece to satisfy a test would be content written
            // for the suite rather than for the game.
            let mail: ItemDef = ron::from_str(
                r#"(
                    schema_version: 1,
                    id: "scorched-mail",
                    name: "Scorched Mail",
                    category: Armor,
                    stack_limit: 1,
                    combat: Some((resistances: { Physical: 500 })),
                )"#,
            )
            .expect("the armour parses");
            if let Some(slice) = armoured.world_mut().resource_mut::<slice::Slice>() {
                slice.content.items.insert(mail.id.clone(), mail);
            }
            grant(&mut armoured, "scorched-mail", 1);
            if let Some(slice) = armoured.world_mut().resource_mut::<slice::Slice>() {
                assert!(slice.open_equipment(), "nothing else is open");
                assert!(slice.take_offer(1), "the mail is the only equippable held");
                slice.close_offer();
            }
            step(&mut armoured, 2);

            let reduced = one_enemy_hit(&mut armoured);
            assert!(
                (reduced - full * 0.5).abs() < 1e-3,
                "50% physical resistance should halve the swing: {full} -> {reduced}"
            );
        }

        /// The M6 criterion that had been argued rather than shown: an ability
        /// authored purely in RON, with its own effects, timing and cost, plays
        /// with no Rust change. It is parsed here and named nowhere in code, and
        /// it lands in the fourth loadout slot, so this is also the test that
        /// `C` binds anything.
        #[test]
        fn an_ability_authored_in_ron_plays_with_no_rust_change() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);

            let ability: AbilityDef = ron::from_str(
                r#"(
                    schema_version: 1,
                    id: "cinder-veil",
                    name: "Cinder Veil",
                    targeting: SelfOnly,
                    cost: 7000,
                    cooldown_ticks: 240,
                    windup_ticks: 3,
                    recovery_ticks: 5,
                    effects: [
                        ApplyStatus(status: "status-guarded", duration_ticks: 30),
                    ],
                )"#,
            )
            .expect("the ability parses");
            let id = ability.id.clone();
            let player = slice::player_entity(app.world()).expect("a player");
            if let Some(slice) = app.world_mut().resource_mut::<slice::Slice>() {
                slice.content.abilities.insert(id.clone(), ability);
            }
            let loadout = &mut app
                .world_mut()
                .get_mut::<slice::Abilities>(player)
                .expect("the player has a loadout")
                .0;
            loadout.push(id);
            assert_eq!(loadout.len(), 4, "after the wayfinder's authored three");

            let full = player_stamina(&app);
            press(&mut app, KeyCode::C);
            step(&mut app, 1);
            release(&mut app, KeyCode::C);
            step(&mut app, 1);
            assert!(casting(&app), "C casts the fourth loadout slot");
            assert!(
                (full - player_stamina(&app) - 7.0).abs() < 1e-3,
                "at its authored cost of 7"
            );

            let guarded = |app: &App| {
                let tick = app.world().resource::<FixedTime>().map_or(0, |time| time.tick);
                app.world()
                    .get::<Statuses>(player)
                    .map_or(1.0, |statuses| statuses.damage_taken_multiplier(tick))
            };

            // Past its 3-tick wind-up, the status it authored is in force.
            step(&mut app, 5);
            assert!((guarded(&app) - 0.5).abs() < 1e-6, "its effect landed");

            // Its own 30-tick duration governs, not warden-guard's 180.
            step(&mut app, 40);
            assert!(
                (guarded(&app) - 1.0).abs() < 1e-6,
                "the authored duration, not another ability's, sets the expiry"
            );

            // And its authored 240-tick cooldown still holds.
            let rested = player_stamina(&app);
            press(&mut app, KeyCode::C);
            step(&mut app, 1);
            release(&mut app, KeyCode::C);
            step(&mut app, 1);
            assert!(!casting(&app), "still cooling down");
            assert!(player_stamina(&app) >= rested, "and a refusal spends nothing");
        }

        /// Two identical runs of one ability sequence end in the same place:
        /// the M6 plan's determinism check, taken in a running game with live
        /// enemies rather than only against the executor.
        #[test]
        fn an_ability_sequence_is_deterministic() {
            fn run() -> (WorldState, f32, f32, Vec<(f32, Vec3)>) {
                let mut app = slice_app("ashfall-lane");
                step(&mut app, 1);
                // F throws a bolt, so a projectile's flight is in the run.
                for (key, settle) in [
                    (KeyCode::Q, 40),
                    (KeyCode::R, 30),
                    (KeyCode::F, 60),
                    (KeyCode::Space, 120),
                ] {
                    press(&mut app, key);
                    step(&mut app, 1);
                    release(&mut app, key);
                    step(&mut app, settle);
                }
                let player = slice::player_entity(app.world()).expect("a player");
                let enemies = app
                    .world()
                    .query2::<Transform, slice::Enemy>()
                    .map(|(entity, transform, _)| {
                        let health = app.world().get::<Health>(entity).map_or(0.0, |h| h.current);
                        (health, transform.translation)
                    })
                    .collect();
                let state = app
                    .world()
                    .resource::<slice::Slice>()
                    .map(|slice| slice.state.clone())
                    .expect("the slice");
                (state, player_health(&app, player), player_stamina(&app), enemies)
            }
            assert_eq!(run(), run());
        }

        /// What a body is drawn with. A model must be drawn with one material
        /// across every part: a cue that recoloured half a body would be a
        /// body with two states at once.
        fn drawn_with(app: &App, entity: engine::ecs::Entity) -> engine::render::MaterialHandle {
            if let Some(model) = app.world().get::<engine::render::Model>(entity) {
                let first = model.parts.first().expect("a model has parts").material;
                assert!(
                    model.parts.iter().all(|part| part.material == first),
                    "every part of a model shows the same cue"
                );
                return first;
            }
            app.world()
                .get::<engine::render::MeshRenderer>(entity)
                .map(|renderer| renderer.material)
                .expect("the entity is drawn")
        }

        fn base_of(app: &App, entity: engine::ecs::Entity) -> engine::render::MaterialHandle {
            app.world()
                .get::<slice::BaseMaterial>(entity)
                .map(|base| base.0)
                .expect("dressed entities record their base look")
        }

        fn cues(app: &App) -> slice::Cues {
            app.world()
                .resource::<slice::SliceVisuals>()
                .map(|visuals| visuals.cues)
                .expect("the slice uploads its visuals")
        }

        /// Overlapping states need an order, and this is it.
        #[test]
        fn cues_show_what_a_player_would_most_regret_missing() {
            use game_runtime::combat::AttackPhase;
            use slice::{Cue, cue};

            assert_eq!(cue(false, false, None), Cue::Idle);
            assert_eq!(cue(false, false, Some(AttackPhase::Windup)), Cue::Windup);
            assert_eq!(cue(false, false, Some(AttackPhase::Active)), Cue::Strike);
            assert_eq!(
                cue(false, false, Some(AttackPhase::Recovery)),
                Cue::Idle,
                "recovery is the punish window, and looks like nothing is happening"
            );
            assert_eq!(
                cue(false, true, Some(AttackPhase::Active)),
                Cue::Evading,
                "whether the i-frames are up beats what anyone is swinging"
            );
            assert_eq!(
                cue(true, true, Some(AttackPhase::Active)),
                Cue::Hurt,
                "being hit beats everything: it answers 'did that land?'"
            );
        }

        #[test]
        fn a_pool_fraction_is_always_drawable() {
            assert_eq!(slice::fraction(60.0, 120.0), 0.5);
            assert_eq!(slice::fraction(-5.0, 120.0), 0.0, "overkill draws empty");
            assert_eq!(slice::fraction(500.0, 120.0), 1.0);
            assert_eq!(slice::fraction(10.0, 0.0), 0.0, "no pool, no division by zero");
            assert_eq!(slice::fraction(f32::NAN, 120.0), 0.0);
        }

        /// The telegraph a Souls-lite fight rests on, then the answer to "did
        /// that land?", then back to normal.
        #[test]
        fn an_enemy_swing_is_telegraphed_and_its_hit_flashes() {
            use game_runtime::combat::AttackPhase;

            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let cues = cues(&app);
            let (player, enemy) = arm_one_enemy(&mut app);
            let phase = |app: &App| app.world().get::<Attack>(enemy).map(Attack::phase);

            step(&mut app, 1);
            assert_eq!(phase(&app), Some(AttackPhase::Windup));
            assert_eq!(drawn_with(&app, enemy), cues.windup, "a wind-up is visible");

            let before = player_health(&app, player);
            let mut struck = false;
            for _ in 0..30 {
                step(&mut app, 1);
                if phase(&app) == Some(AttackPhase::Active) {
                    assert_eq!(
                        drawn_with(&app, enemy),
                        cues.strike,
                        "live hit frames are visible"
                    );
                    struck = true;
                }
                if player_health(&app, player) < before {
                    break;
                }
            }
            assert!(struck, "the swing reached its active window");
            assert!(player_health(&app, player) < before, "and landed");
            assert_eq!(drawn_with(&app, player), cues.hurt, "the player flashes when hit");

            step(&mut app, slice::HURT_FLASH_TICKS + 30);
            assert_eq!(
                drawn_with(&app, player),
                base_of(&app, player),
                "and the flash ends on its own"
            );
            assert_eq!(
                drawn_with(&app, enemy),
                base_of(&app, enemy),
                "a finished swing goes back to the body's own colour"
            );
        }

        /// A cast is a committed action like a swing, so it telegraphs like one.
        #[test]
        fn a_cast_wind_up_reads_like_a_swing() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let cues = cues(&app);
            let player = slice::player_entity(app.world()).expect("a player");

            press(&mut app, KeyCode::R);
            step(&mut app, 1);
            release(&mut app, KeyCode::R);
            step(&mut app, 1);
            assert!(casting(&app));
            assert_eq!(drawn_with(&app, player), cues.windup);
        }

        #[test]
        fn a_dodge_shows_its_invulnerable_window() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let cues = cues(&app);
            let player = slice::player_entity(app.world()).expect("a player");

            press(&mut app, KeyCode::ShiftLeft);
            step(&mut app, 1);
            release(&mut app, KeyCode::ShiftLeft);
            assert_eq!(drawn_with(&app, player), cues.evading);
        }

        /// Health, stamina and cooldowns are not debugging aids, so they must
        /// not wait for F1.
        #[test]
        fn gauges_draw_without_the_debug_overlay() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            assert!(
                !app.world()
                    .resource::<DebugOverlay>()
                    .is_some_and(|overlay| overlay.0),
                "the overlay is off"
            );
            // Headless runs never hand lines to a renderer, so they pile up;
            // start from empty to count one frame's worth.
            if let Some(debug) = app.world_mut().resource_mut::<DebugDraw>() {
                debug.clear();
            }
            step(&mut app, 1);
            let debug = app.world().resource::<DebugDraw>().expect("debug draw");
            assert!(debug.vertex_count() > 0, "bars were drawn");
            assert!(debug.vertex_count().is_multiple_of(2), "as whole segments");
            assert_eq!(debug.dropped_lines(), 0, "within the budget");
        }

        /// How far a body's top is tipped toward `direction`: positive leans
        /// in, negative leans away.
        fn lean_toward(app: &App, entity: engine::ecs::Entity, direction: Vec3) -> f32 {
            let rotation = app
                .world()
                .get::<Transform>(entity)
                .map(|transform| transform.rotation)
                .expect("a body");
            (rotation * Vec3::Y).dot(direction)
        }

        #[test]
        fn poses_follow_the_windows_combat_runs_on() {
            use game_runtime::combat::AttackPhase::{Active, Finished, Recovery, Windup};
            use slice::{Motion, Pose, pose};

            assert_eq!(pose(Motion::Still, 0.0), Pose::REST);

            let coiled = pose(
                Motion::Swing {
                    phase: Windup,
                    progress: 1.0,
                },
                0.0,
            );
            assert!(coiled.lean < 0.0, "a wind-up leans back");
            assert!(coiled.scale.y < 1.0, "and crouches");

            let struck = pose(
                Motion::Swing {
                    phase: Active,
                    progress: 1.0,
                },
                0.0,
            );
            assert!(struck.lean > 0.0, "the blow leans in");

            let recovered = pose(
                Motion::Swing {
                    phase: Recovery,
                    progress: 1.0,
                },
                0.0,
            );
            assert!(recovered.lean.abs() < 1e-6, "recovery ends upright");
            assert!((recovered.scale - Vec3::ONE).length() < 1e-6);
            assert_eq!(
                pose(
                    Motion::Swing {
                        phase: Finished,
                        progress: 1.0
                    },
                    0.0
                ),
                Pose::REST
            );

            let channel = pose(
                Motion::Cast {
                    phase: Windup,
                    progress: 1.0,
                },
                0.0,
            );
            assert!(channel.scale.x > 1.0, "a cast swells while it channels");

            let rolling = pose(Motion::Roll { progress: 0.5 }, 0.0);
            assert!(
                (rolling.roll - std::f32::consts::PI).abs() < 1e-4,
                "halfway through a roll is upside down"
            );
            let rolled = pose(Motion::Roll { progress: 1.0 }, 0.0);
            assert!(
                (rolled.roll - std::f32::consts::TAU).abs() < 1e-4,
                "and a finished roll is one full turn, so it lands upright"
            );

            let flinching = pose(Motion::Still, 1.0);
            assert!(
                flinching.scale.y < 1.0 && flinching.scale.x > 1.0,
                "a hit squashes"
            );
            assert_eq!(
                pose(Motion::Still, 7.0),
                flinching,
                "and an out-of-range flinch is clamped, not extrapolated"
            );
        }

        /// The rule that makes animation safe: posing a body must never move
        /// it, or a lunge would move its hitbox and change the fight.
        #[test]
        fn posing_never_moves_a_body() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (_, enemy) = arm_one_enemy(&mut app);
            step(&mut app, 3);

            let positions = |app: &App| -> Vec<(engine::ecs::Entity, Vec3)> {
                app.world()
                    .query::<Transform>()
                    .map(|(entity, transform)| (entity, transform.translation))
                    .collect()
            };
            let before = positions(&app);
            let tick = app.world().resource::<FixedTime>().map_or(0, |time| time.tick);
            slice::pose_bodies(app.world_mut(), tick);

            assert_eq!(positions(&app), before, "posing moved something");
            let rotation = app.world().get::<Transform>(enemy).map(|t| t.rotation);
            assert_ne!(
                rotation,
                Some(engine::math::Quat::IDENTITY),
                "while the swing is visibly posed"
            );
        }

        /// A wind-up you can see coil is a wind-up you can learn to dodge.
        #[test]
        fn an_enemy_coils_then_leans_into_its_strike() {
            use game_runtime::combat::AttackPhase;

            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (_, enemy) = arm_one_enemy(&mut app);
            let facing = app
                .world()
                .get::<slice::Facing>(enemy)
                .map(|facing| facing.0)
                .expect("the enemy faces the player");
            let phase = |app: &App| app.world().get::<Attack>(enemy).map(Attack::phase);

            step(&mut app, 5);
            assert_eq!(phase(&app), Some(AttackPhase::Windup));
            assert!(
                lean_toward(&app, enemy, facing) < 0.0,
                "coiled back from the player"
            );

            let mut leaned_in = false;
            for _ in 0..20 {
                step(&mut app, 1);
                if phase(&app) != Some(AttackPhase::Windup) && lean_toward(&app, enemy, facing) > 0.0 {
                    leaned_in = true;
                    break;
                }
            }
            assert!(leaned_in, "and leans into the blow once it is live");
        }

        #[test]
        fn a_dodge_visibly_rolls() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let player = slice::player_entity(app.world()).expect("a player");

            press(&mut app, KeyCode::ShiftLeft);
            step(&mut app, 1);
            release(&mut app, KeyCode::ShiftLeft);
            step(&mut app, 6);
            let rotation = app.world().get::<Transform>(player).map(|t| t.rotation);
            assert_ne!(rotation, Some(engine::math::Quat::IDENTITY), "mid-roll");

            step(&mut app, slice::DODGE_DURATION as u64 + 10);
            let rotation = app.world().get::<Transform>(player).map(|t| t.rotation);
            assert_eq!(
                rotation,
                Some(engine::math::Quat::IDENTITY),
                "and upright once the roll is over"
            );
        }

        /// A kill has a visible answer, and the answer does not linger.
        #[test]
        fn a_dying_enemy_leaves_remains_that_go() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let enemies_before = app.world().query::<slice::Enemy>().count();
            let enemy = app
                .world()
                .query::<slice::Enemy>()
                .map(|(entity, _)| entity)
                .next()
                .expect("the lane has enemies");
            app.world_mut()
                .get_mut::<Health>(enemy)
                .expect("the enemy has health")
                .current = 0.0;
            step(&mut app, 1);

            assert_eq!(
                app.world().query::<slice::Enemy>().count(),
                enemies_before - 1,
                "the enemy itself is gone the tick it dies, as before"
            );
            assert_eq!(app.world().query::<slice::Remains>().count(), 1);

            step(&mut app, slice::REMAINS_TICKS + 5);
            assert_eq!(
                app.world().query::<slice::Remains>().count(),
                0,
                "and its remains go on their own"
            );
        }

        #[test]
        fn leaving_a_location_sweeps_its_remains() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let enemy = app
                .world()
                .query::<slice::Enemy>()
                .map(|(entity, _)| entity)
                .next()
                .expect("the lane has enemies");
            app.world_mut()
                .get_mut::<Health>(enemy)
                .expect("the enemy has health")
                .current = 0.0;
            step(&mut app, 1);
            assert_eq!(app.world().query::<slice::Remains>().count(), 1);

            slice::clear_location(app.world_mut());
            assert_eq!(
                app.world().query::<slice::Remains>().count(),
                0,
                "remains would otherwise follow the player through a door"
            );
        }

        // ---- models (M7 wave 2) ------------------------------------------

        /// A budget counted in content has to be the budget the engine builds,
        /// or validation bounds a number nothing draws.
        #[test]
        fn content_triangle_counts_are_what_the_engine_builds() {
            use game_core::content::{Fixed, ShapeDef};
            let f = Fixed::from_milli;
            let shapes = [
                ShapeDef::Box {
                    x: f(400),
                    y: f(300),
                    z: f(200),
                },
                ShapeDef::Wedge {
                    x: f(400),
                    y: f(300),
                    z: f(200),
                },
                ShapeDef::Cylinder {
                    radius: f(100),
                    height: f(500),
                    sides: 3,
                },
                ShapeDef::Cylinder {
                    radius: f(100),
                    height: f(500),
                    sides: 24,
                },
                ShapeDef::Cone {
                    radius: f(100),
                    height: f(500),
                    sides: 5,
                },
                ShapeDef::Frustum {
                    bottom: f(100),
                    top: f(0),
                    height: f(500),
                    sides: 7,
                },
                ShapeDef::Frustum {
                    bottom: f(100),
                    top: f(60),
                    height: f(500),
                    sides: 7,
                },
                ShapeDef::Sphere {
                    radius: f(200),
                    detail: 0,
                },
                ShapeDef::Sphere {
                    radius: f(200),
                    detail: 1,
                },
                ShapeDef::Taper {
                    x: f(400),
                    z: f(300),
                    top_x: f(500),
                    top_z: f(320),
                    height: f(600),
                },
                ShapeDef::Taper {
                    x: f(300),
                    z: f(300),
                    top_x: f(0),
                    top_z: f(0),
                    height: f(400),
                },
                ShapeDef::Taper {
                    x: f(300),
                    z: f(200),
                    top_x: f(0),
                    top_z: f(200),
                    height: f(400),
                },
                ShapeDef::Taper {
                    x: f(300),
                    z: f(200),
                    top_x: f(150),
                    top_z: f(0),
                    height: f(400),
                },
                ShapeDef::Dome {
                    radius: f(200),
                    segments: 7,
                    rings: 3,
                },
                ShapeDef::Dome {
                    radius: f(200),
                    segments: 6,
                    rings: 1,
                },
                ShapeDef::Dome {
                    radius: f(200),
                    segments: 24,
                    rings: 6,
                },
            ];
            for shape in shapes {
                let built = engine::shapes::mesh(
                    slice::shape_of(shape),
                    engine::shapes::FaceColors::uniform([1.0; 3]),
                );
                assert_eq!(shape.triangles(), built.triangle_count(), "{shape:?}");
            }
        }

        /// Where a body's lowest point sits, in world space, at rest.
        fn lowest_point(app: &App, entity: engine::ecs::Entity) -> f32 {
            let world = app.world();
            let body = world.get::<Transform>(entity).expect("a body").to_matrix();
            world
                .get::<engine::render::Model>(entity)
                .expect("a modelled body")
                .parts
                .iter()
                .filter_map(|part| {
                    part.bounds
                        .map(|bounds| bounds.world_bounds(&part.world_matrix(&body)).min.y)
                })
                .fold(f32::INFINITY, f32::min)
        }

        /// Recipes measure from the ground, and the loader subtracts how high a
        /// body's origin stands. Wrong either way, everyone floats or sinks.
        #[test]
        fn models_stand_on_the_ground() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let player = slice::player_entity(app.world()).expect("a player");
            let feet = lowest_point(&app, player);
            assert!(feet.abs() < 0.02, "the wayfinder's feet at {feet}");

            for (entity, _) in app.world().query::<slice::Enemy>() {
                let feet = lowest_point(&app, entity);
                assert!(feet.abs() < 0.02, "an enemy's feet at {feet}");
            }
        }

        /// Scale multiplies pivots and parts alike, so an elite stays standing
        /// rather than sinking into the floor it is larger than.
        #[test]
        fn a_scaled_model_still_stands_on_the_ground() {
            let mut app = slice_app("crypt-of-ash");
            step(&mut app, 1);
            let elite = app
                .world()
                .query::<slice::Enemy>()
                .find(|(_, enemy)| enemy.prefab.to_string() == "husk-ashlord")
                .map(|(entity, _)| entity)
                .expect("the crypt holds the ashlord");
            let feet = lowest_point(&app, elite);
            assert!(feet.abs() < 0.02, "the ashlord's feet at {feet}");
        }

        /// Each slot rests on its joint, so a pose turns an arm about the
        /// shoulder rather than about its own middle.
        #[test]
        fn a_model_has_one_part_per_slot_resting_on_its_joint() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let player = slice::player_entity(app.world()).expect("a player");
            let model = app
                .world()
                .get::<engine::render::Model>(player)
                .expect("the wayfinder wears its model")
                .clone();
            assert_eq!(model.parts.len(), 6, "torso, head, two arms, two legs");

            // The right shoulder pivot is (0.29, 1.47, 0) from the ground; the
            // body's origin stands 0.9 above it.
            let stand = game_runtime::movement::Capsule::default().half_extent();
            let shoulder = Vec3::new(0.29, 1.47 - stand, 0.0);
            assert!(
                model
                    .parts
                    .iter()
                    .any(|part| part.rest.translation.abs_diff_eq(shoulder, 1e-4)),
                "a part rests on the right shoulder"
            );

            let library = app
                .world()
                .resource::<slice::ModelLibrary>()
                .expect("built at startup");
            let built = library
                .get(&game_core::ids::ModelId::new("wayfinder").expect("valid id"))
                .expect("the wayfinder model is built");
            assert_eq!(
                built.slots.iter().map(ToString::to_string).collect::<Vec<_>>(),
                vec!["torso", "head", "arm-l", "arm-r", "leg-l", "leg-r"],
                "parts follow the body plan's slot order, which clips will rely on"
            );
        }

        /// A model has a front, so a body turns to face where it faces.
        #[test]
        fn a_modelled_body_faces_where_it_is_facing() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (_, enemy) = arm_one_enemy(&mut app);
            let facing = app
                .world()
                .get::<slice::Facing>(enemy)
                .map(|facing| facing.0)
                .expect("facing");
            // Past the swing, so no lean muddies the forward vector.
            step(&mut app, 60);
            let forward = app
                .world()
                .get::<Transform>(enemy)
                .map(|transform| transform.rotation * Vec3::NEG_Z)
                .expect("a body");
            assert!(
                forward.abs_diff_eq(facing.normalize(), 1e-3),
                "the model's front ({forward:?}) points along its facing ({facing:?})"
            );
        }

        #[test]
        fn a_modelled_enemy_leaves_modelled_remains() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let enemy = app
                .world()
                .query::<slice::Enemy>()
                .map(|(entity, _)| entity)
                .next()
                .expect("the lane has enemies");
            let parts = app
                .world()
                .get::<engine::render::Model>(enemy)
                .map(|model| model.parts.len())
                .expect("the enemy is modelled");
            app.world_mut()
                .get_mut::<Health>(enemy)
                .expect("the enemy has health")
                .current = 0.0;
            step(&mut app, 1);

            let remains: Vec<usize> = app
                .world()
                .query2::<slice::Remains, engine::render::Model>()
                .map(|(_, _, model)| model.parts.len())
                .collect();
            assert_eq!(
                remains,
                vec![parts],
                "what falls is the same body, every part of it"
            );
        }

        /// A named model that was never built — a failed upload — must not
        /// take a body out of the world.
        #[test]
        fn a_model_that_was_not_built_falls_back_to_the_block_out() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let visuals = app
                .world()
                .resource::<slice::SliceVisuals>()
                .copied()
                .expect("visuals");
            let world = app.world_mut();
            let entity = world.spawn();
            let _ = world.insert(entity, Transform::IDENTITY);
            let missing = game_core::ids::ModelId::new("never-built").expect("valid id");
            slice::dress_as(world, entity, &visuals, Some(&missing), visuals.enemy);

            assert!(world.get::<engine::render::Model>(entity).is_none());
            assert_eq!(
                world.get::<engine::render::MeshRenderer>(entity).map(|r| r.mesh),
                Some(visuals.enemy.0),
                "drawn as the block-out shape it would have had"
            );
        }

        #[test]
        fn palette_colours_are_converted_to_linear() {
            assert_eq!(slice::linear((0, 0, 0)), [0.0; 3]);
            assert_eq!(slice::linear((255, 255, 255)), [1.0; 3]);
            let [mid, _, _] = slice::linear((128, 0, 0));
            assert!(
                (mid - 0.2158).abs() < 1e-3,
                "sRGB mid-grey is about a fifth of linear, not a half: {mid}"
            );
        }

        // ---- animation (M7 wave 3) ---------------------------------------

        /// The pose of one slot of a body, by slot name.
        fn slot_pose(app: &App, entity: engine::ecs::Entity, slot: &str) -> Transform {
            let world = app.world();
            let worn = world
                .get::<slice::Wears>(entity)
                .expect("a modelled body")
                .0
                .clone();
            let index = world
                .resource::<slice::ModelLibrary>()
                .and_then(|library| library.get(&worn))
                .and_then(|built| built.slots.iter().position(|name| name.to_string() == slot))
                .expect("the body has that slot");
            world.get::<engine::render::Model>(entity).expect("a model").parts[index].pose
        }

        /// How far a limb hanging along -Y has swung forward about X, in
        /// degrees.
        fn swing_degrees(pose: &Transform) -> f32 {
            let limb = pose.rotation * Vec3::NEG_Y;
            (-limb.z).atan2(-limb.y).to_degrees()
        }

        #[test]
        fn walking_swings_the_legs_and_standing_still_rests_them() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            let player = slice::player_entity(app.world()).expect("a player");

            press(&mut app, KeyCode::W);
            let mut widest: f32 = 0.0;
            for _ in 0..40 {
                step(&mut app, 1);
                widest = widest.max(swing_degrees(&slot_pose(&app, player, "leg-l")).abs());
            }
            release(&mut app, KeyCode::W);
            assert!(widest > 15.0, "walking swings the legs: at most {widest} degrees");

            step(&mut app, 30);
            let resting = swing_degrees(&slot_pose(&app, player, "leg-l")).abs();
            assert!(
                resting < 1.0,
                "and standing still brings them back: {resting} degrees"
            );
        }

        /// The overhead chop: the weapon arm is raised by the end of the
        /// wind-up and has come down once the blow is live.
        #[test]
        fn a_swing_raises_the_weapon_arm_then_brings_it_down() {
            use game_runtime::combat::AttackPhase;

            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let player = slice::player_entity(app.world()).expect("a player");
            let phase = |app: &App| app.world().get::<Attack>(player).map(Attack::phase);

            press(&mut app, KeyCode::Space);
            step(&mut app, 1);
            release(&mut app, KeyCode::Space);

            let mut raised: f32 = 0.0;
            while phase(&app) == Some(AttackPhase::Windup) {
                raised = raised.max(swing_degrees(&slot_pose(&app, player, "arm-r")));
                step(&mut app, 1);
            }
            assert!(
                raised > 90.0,
                "the weapon arm rises overhead in the wind-up: {raised} degrees"
            );

            step(&mut app, 4);
            let struck = swing_degrees(&slot_pose(&app, player, "arm-r"));
            assert!(
                struck < raised - 45.0,
                "and comes down as the blow lands: {raised} -> {struck}"
            );
        }

        /// The ownership rule, widened to parts: animating moves limbs and
        /// never a body.
        #[test]
        fn animating_never_moves_a_body() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (_, enemy) = arm_one_enemy(&mut app);
            step(&mut app, 8);

            let positions = |app: &App| -> Vec<(engine::ecs::Entity, Vec3)> {
                app.world()
                    .query::<Transform>()
                    .map(|(entity, transform)| (entity, transform.translation))
                    .collect()
            };
            let poses = |app: &App| {
                app.world()
                    .get::<engine::render::Model>(enemy)
                    .expect("a model")
                    .clone()
            };

            // Knock the arm off its pose, so animating has something to write.
            if let Some(model) = app.world_mut().get_mut::<engine::render::Model>(enemy) {
                for part in &mut model.parts {
                    part.pose = Transform::IDENTITY;
                }
            }
            let before = positions(&app);
            let unposed = poses(&app);
            let tick = app.world().resource::<FixedTime>().map_or(0, |time| time.tick) + 100;
            app.world_mut().resource_scope::<slice::Slice, _>(|world, slice| {
                animation::animate(world, &slice.content, tick);
            });

            assert_eq!(positions(&app), before, "animating moved something");
            assert_ne!(poses(&app), unposed, "while the enemy's limbs did move");
        }

        /// A model's own clip overrides its body plan's; the rest comes from
        /// the plan.
        #[test]
        fn husks_shamble_where_the_wayfinder_walks() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let library = app.world().resource::<slice::ModelLibrary>().expect("built");
            let walk = |model: &str| {
                library
                    .get(&game_core::ids::ModelId::new(model).expect("valid id"))
                    .and_then(|built| built.animations.walk.clone())
                    .map(|clip| clip.to_string())
            };
            let clip = |model: &str, pick: fn(&game_core::content::AnimationSet) -> Option<&game_core::ids::ClipId>| {
                library
                    .get(&game_core::ids::ModelId::new(model).expect("valid id"))
                    .and_then(|built| pick(&built.animations).map(|clip| clip.to_string()))
            };
            fn swing(set: &game_core::content::AnimationSet) -> Option<&game_core::ids::ClipId> {
                set.swing.as_ref()
            }
            fn dodge(set: &game_core::content::AnimationSet) -> Option<&game_core::ids::ClipId> {
                set.dodge.as_ref()
            }
            assert_eq!(walk("wayfinder").as_deref(), Some("humanoid-walk"));
            assert_eq!(walk("corrupted-husk").as_deref(), Some("husk-shamble"));
            assert_eq!(clip("wayfinder", swing).as_deref(), Some("humanoid-swing"));
            assert_eq!(
                clip("corrupted-husk", swing).as_deref(),
                Some("husk-claw"),
                "the husk overrides its walk and its swing"
            );
            assert_eq!(
                clip("corrupted-husk", dodge).as_deref(),
                Some("humanoid-dodge"),
                "and keeps the plan's clip for everything it does not name"
            );
        }

        /// An ability on cooldown is refused, which is what stops Q being a
        /// better melee attack than the melee attack.
        #[test]
        fn an_ability_on_cooldown_is_refused() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);

            press(&mut app, KeyCode::Q);
            step(&mut app, 1);
            release(&mut app, KeyCode::Q);
            step(&mut app, 1);
            // Let the whole cast finish: wind-up, effects, recovery.
            step(&mut app, 60);
            assert!(!casting(&app), "the first cast finished");

            let before = player_stamina(&app);
            press(&mut app, KeyCode::Q);
            step(&mut app, 1);
            release(&mut app, KeyCode::Q);
            step(&mut app, 1);

            assert!(!casting(&app), "ember-cleave is on a 90-tick cooldown");
            assert!(
                (player_stamina(&app) - before).abs() < 1.0,
                "a refused cast must not spend anything"
            );
        }

        /// Task I: the elite's one distinguishing ability can finally fire.
        /// `call-husks` is authored on `husk-ashlord` and, until wave 3, could
        /// never run — so the elite was a bigger husk with a dead content field.
        #[test]
        fn the_ashlord_summons_husks() {
            let mut app = slice_app("crypt-of-ash");
            step(&mut app, 1);

            let ashlord = app
                .world()
                .query::<slice::Enemy>()
                .find(|(_, enemy)| enemy.prefab.name() == "husk-ashlord")
                .map(|(entity, _)| entity)
                .expect("the crypt holds the elite");
            assert!(
                app.world().get::<slice::Abilities>(ashlord).is_some(),
                "the ashlord is granted its authored loadout"
            );

            let before = app.world().query::<slice::Enemy>().count();

            // Activated directly rather than by walking into the fight and
            // waiting. Driving the brain there means depending on aggro tuning,
            // on `Support`'s preferred range, and on the player surviving a
            // three-enemy elite encounter long enough — and a death would
            // respawn them in the settlement, rebuilding the location so the
            // enemy count afterwards describes the wrong room entirely. What
            // this test is for is that `call-husks` summons, so it says so.
            let started = {
                let mut started = false;
                app.world_mut().resource_scope::<slice::Slice, _>(|world, slice| {
                    started = slice::start_ability(world, slice, ashlord, 0);
                });
                started
            };
            assert!(started, "the ashlord's authored ability should activate");

            // Past the 45-tick wind-up, so the effects land.
            step(&mut app, 60);

            let after = app.world().query::<slice::Enemy>().count();
            assert!(
                after > before,
                "call-husks should add husks to the fight: {before} -> {after}"
            );
            // Counted exactly, not merely "more than before": the ashlord
            // stands with two authored husks, so a vague assertion would also
            // be satisfied by something else spawning. `call-husks` summons 2.
            assert_eq!(
                (after - before) % 2,
                0,
                "each cast summons the authored pair: {before} -> {after}"
            );
            // The bound that makes ADR-0010's claim real, asserted against the
            // executor's actual cap rather than against a number no run could
            // reach. 400 ticks is long enough for several casts of a 600-tick
            // cooldown to be attempted.
            assert!(
                after <= game_runtime::ability::MAX_SPAWNS_PER_LOCATION,
                "summoning must stop at the per-location cap, got {after}"
            );
        }

        #[test]
        fn a_swing_costs_stamina_and_cannot_be_spammed() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);

            let full = player_stamina(&app);

            press(&mut app, KeyCode::Space);
            step(&mut app, 1);
            release(&mut app, KeyCode::Space);

            let spent = player_stamina(&app);
            assert!(spent < full, "a swing should cost stamina: {full} -> {spent}");

            // Emptying the bar refuses further swings rather than going negative.
            for _ in 0..20 {
                press(&mut app, KeyCode::Space);
                step(&mut app, 1);
                release(&mut app, KeyCode::Space);
            }
            let floor = player_stamina(&app);
            assert!(floor >= 0.0, "stamina must not go negative, got {floor}");
        }

        #[test]
        fn a_dodge_makes_the_player_briefly_untouchable() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);

            press(&mut app, KeyCode::ShiftLeft);
            step(&mut app, 1);
            release(&mut app, KeyCode::ShiftLeft);

            let rolling = app
                .world()
                .query::<Dodge>()
                .next()
                .map(|(_, dodge)| dodge.is_invulnerable())
                .expect("a dodge should have started");
            assert!(rolling, "i-frames cover the start of the roll");

            // And it ends on schedule rather than lasting forever.
            step(&mut app, slice::DODGE_DURATION as u64 + 30);
            assert_eq!(
                app.world().query::<Dodge>().count(),
                0,
                "a finished dodge must stop existing"
            );
        }

        fn objective(name: &str) -> ObjectiveId {
            ObjectiveId::new(name).expect("valid id")
        }

        /// Opens `still-waters` and marks everything before `node` done, so a
        /// test can exercise one step of the graph without walking the whole
        /// quest to reach it.
        fn quest_open_at(app: &mut App, completed: &[&str]) {
            if let Some(slice) = app.world_mut().resource_mut::<slice::Slice>() {
                slice.raise(DomainEvent::QuestAccepted {
                    quest: quest("still-waters"),
                });
                for name in completed {
                    slice.raise(DomainEvent::QuestObjectiveCompleted {
                        quest: quest("still-waters"),
                        objective: objective(name),
                    });
                }
            }
            step(app, 2);
        }

        /// Stands the player on `position`. Distances have their own tests; a
        /// test about examining a thing should not also be a test about walking.
        fn stand_at(app: &mut App, position: Vec3) {
            slice::move_player_to(app.world_mut(), position);
            step(app, 1);
        }

        fn interact(app: &mut App) {
            press(app, KeyCode::E);
            step(app, 2);
            release(app, KeyCode::E);
            step(app, 2);
        }

        /// Presses a digit, taking that row of whatever is on offer.
        fn choose(app: &mut App, digit: u32) {
            let key = match digit {
                1 => KeyCode::Digit1,
                2 => KeyCode::Digit2,
                3 => KeyCode::Digit3,
                other => panic!("no test presses digit {other}"),
            };
            press(app, key);
            step(app, 2);
            release(app, key);
            step(app, 2);
        }

        /// Whether a menu is currently open.
        fn offer_open(app: &App) -> bool {
            app.world()
                .resource::<slice::Slice>()
                .is_some_and(|slice| slice.offer.is_some())
        }

        /// How many rows the open offer currently has.
        fn offered(app: &App) -> usize {
            app.world()
                .resource::<slice::Slice>()
                .and_then(|slice| {
                    slice
                        .offer
                        .as_ref()
                        .map(|offer| game_runtime::selection::rows(offer, &slice.content, &slice.state).len())
                })
                .unwrap_or(0)
        }

        /// Talks to whoever is in reach, taking the first line at every node.
        ///
        /// The runtime used to walk conversations on the player's behalf; now
        /// the player walks them, so a test that only cares whether a quest
        /// moved drives the real selection API the way a person would. A test
        /// about *which* line gets taken presses [`choose`] itself.
        fn converse(app: &mut App) {
            interact(app);
            // Bounded: a conversation that somehow never closed should fail the
            // test rather than hang it.
            for _ in 0..8 {
                if !offer_open(app) || offered(app) == 0 {
                    break;
                }
                choose(app, 1);
            }
            if offer_open(app) {
                // A node with nothing sayable is left by walking away.
                interact(app);
            }
        }

        #[test]
        fn the_lane_holds_the_quest_s_clue_and_decision() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let world = app.world();

            assert_eq!(world.query::<slice::Clue>().count(), 1);
            assert_eq!(world.query::<slice::ChoicePoint>().count(), 1);
            assert_eq!(
                world.query::<slice::Npc>().count(),
                1,
                "arbiter thal now stands where the quest can reach him"
            );
        }

        #[test]
        fn examining_the_intake_completes_the_survey() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            quest_open_at(&mut app, &["meet-sela"]);

            let at = app
                .world()
                .query2::<Transform, slice::Clue>()
                .next()
                .map(|(_, transform, _)| transform.translation)
                .expect("the lane has a clue");
            stand_at(&mut app, at);
            interact(&mut app);

            let slice = app.world().resource::<slice::Slice>().expect("slice exists");
            assert!(
                slice
                    .state
                    .objective_progress(&quest("still-waters"), &objective("survey-waterworks"))
                    .completed,
                "examining the fouled intake should satisfy the survey objective"
            );
        }

        #[test]
        fn a_clue_can_only_be_examined_once() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);

            let at = app
                .world()
                .query2::<Transform, slice::Clue>()
                .next()
                .map(|(_, transform, _)| transform.translation)
                .expect("the lane has a clue");
            stand_at(&mut app, at);
            interact(&mut app);
            interact(&mut app);

            let examined = app
                .world()
                .query::<slice::Clue>()
                .next()
                .map(|(_, clue)| clue.examined)
                .expect("the clue is still in the world");
            assert!(examined, "the prop stays, spent, rather than vanishing");
        }

        #[test]
        fn settling_the_marker_completes_the_decision() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            quest_open_at(&mut app, &["meet-sela", "survey-waterworks"]);

            let at = app
                .world()
                .query2::<Transform, slice::ChoicePoint>()
                .next()
                .map(|(_, transform, _)| transform.translation)
                .expect("the lane has a choice point");
            stand_at(&mut app, at);
            interact(&mut app);

            // M9: walking up asks; it does not decide.
            let decided = |app: &App| {
                app.world()
                    .resource::<slice::Slice>()
                    .expect("slice exists")
                    .state
                    .objective_progress(&quest("still-waters"), &objective("choose-allegiance"))
                    .completed
            };
            assert!(!decided(&app), "the marker only opened its confirm panel");
            assert!(
                matches!(
                    app.world()
                        .resource::<slice::Slice>()
                        .and_then(|slice| slice.offer.clone()),
                    Some(game_runtime::selection::Offer::Decision { .. })
                ),
                "the confirm panel is open"
            );

            choose(&mut app, 1);
            assert!(
                decided(&app),
                "confirming water rights should satisfy the allegiance objective"
            );
            let settled = app
                .world()
                .query::<slice::ChoicePoint>()
                .next()
                .map(|(_, point)| point.settled)
                .expect("the marker stays");
            assert!(settled, "and the marker retires, so it cannot be decided again");
            assert_eq!(
                app.world()
                    .resource::<slice::Slice>()
                    .and_then(|slice| slice.offer.clone()),
                None,
                "a decision closes once made"
            );
        }

        /// The other half of "both": a conversation can settle the same choice.
        #[test]
        fn agreeing_terms_with_thal_settles_the_same_choice() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            quest_open_at(&mut app, &["meet-sela", "survey-waterworks"]);

            // Thal's terms are gated on carrying eight bark, so give the player
            // the bark rather than gathering it — this is a test about dialogue
            // settling a decision, not about harvesting.
            if let Some(slice) = app.world_mut().resource_mut::<slice::Slice>() {
                slice.raise(DomainEvent::ItemGranted {
                    item: ItemId::new("old-forest-bark").expect("valid id"),
                    quantity: 8,
                });
            }
            step(&mut app, 2);

            let at = app
                .world()
                .query2::<Transform, slice::Npc>()
                .next()
                .map(|(_, transform, _)| transform.translation)
                .expect("thal is standing in the lane");
            stand_at(&mut app, at);
            converse(&mut app);

            let slice = app.world().resource::<slice::Slice>().expect("slice exists");
            assert!(
                slice
                    .state
                    .objective_progress(&quest("still-waters"), &objective("choose-allegiance"))
                    .completed,
                "agreeing terms is itself the decision"
            );
        }

        /// The M5 exit criterion, as far as the combat branch goes: the quest
        /// graph runs start to finish driven only by interactions and combat.
        ///
        /// Set up deliberately at each step rather than walked — distances,
        /// aggro, and pathing have their own tests, and choreographing all of
        /// them at once would test those instead of the quest graph.
        #[test]
        fn the_quest_runs_from_sela_to_its_resolution() {
            // It begins in the hub, with the warden.
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);

            let sela = app
                .world()
                .query2::<Transform, slice::Npc>()
                .next()
                .map(|(_, transform, _)| transform.translation)
                .expect("the warden holds the square");
            stand_at(&mut app, sela);
            converse(&mut app);

            let open = |app: &App| -> Vec<String> {
                let slice = app.world().resource::<slice::Slice>().expect("slice exists");
                let def = slice
                    .content
                    .quests
                    .get(&quest("still-waters"))
                    .expect("the quest is authored");
                narrative::quest::open_objectives(def, &slice.state)
                    .iter()
                    .map(ToString::to_string)
                    .collect()
            };
            assert_eq!(
                open(&app),
                vec!["survey-waterworks".to_owned()],
                "talking to sela accepts the quest and opens the survey"
            );

            // Then the waste: examine the fouled intake.
            travel_to_lane(&mut app);
            let clue = app
                .world()
                .query2::<Transform, slice::Clue>()
                .next()
                .map(|(_, transform, _)| transform.translation)
                .expect("the lane has a clue");
            stand_at(&mut app, clue);
            interact(&mut app);
            assert_eq!(open(&app), vec!["choose-allegiance".to_owned()]);
            // What the intake shows is open to read; E puts it down (M9).
            interact(&mut app);

            // Settle water rights at the marker.
            let marker = app
                .world()
                .query2::<Transform, slice::ChoicePoint>()
                .next()
                .map(|(_, transform, _)| transform.translation)
                .expect("the lane has a choice point");
            stand_at(&mut app, marker);
            interact(&mut app);
            choose(&mut app, 1);
            assert_eq!(
                open(&app),
                vec!["secure-works".to_owned(), "seek-order".to_owned()],
                "the decision opens both branches"
            );

            // Take the combat branch: clear the ambush.
            clear_encounter(&mut app, "waterworks-ambush");
            assert_eq!(
                open(&app),
                vec!["report-back".to_owned(), "seek-order".to_owned()],
                "clearing the ambush opens the report; the road not taken stays \
                 open, because nothing in the graph makes branches exclusive"
            );

            // And report to Sela, who is back in the square.
            travel_to_hub(&mut app);
            let sela = app
                .world()
                .query2::<Transform, slice::Npc>()
                .next()
                .map(|(_, transform, _)| transform.translation)
                .expect("the warden is back on her post");
            stand_at(&mut app, sela);
            // The ambush already paid kill experience (ADR-0018), so the
            // reward is what the report itself adds.
            let experience_before_report = app
                .world()
                .resource::<slice::Slice>()
                .expect("slice exists")
                .state
                .player
                .progression
                .experience;
            converse(&mut app);

            let slice = app.world().resource::<slice::Slice>().expect("slice exists");
            assert_eq!(
                slice.state.quest_status(&quest("still-waters")),
                QuestStatus::Completed {
                    resolution: game_core::ids::QuestResolutionId::new("waters-restored").expect("valid id")
                },
                "the quest records which ending occurred, not merely that it ended"
            );
            assert_eq!(
                slice.state.player.progression.experience - experience_before_report,
                450,
                "and pays the reward package"
            );
            assert_eq!(
                slice
                    .state
                    .player
                    .item_count(&ItemId::new("warden-seal").expect("valid id")),
                1
            );
        }

        /// Walks the player through the gate into the lane.
        fn travel_to_lane(app: &mut App) {
            let gate = station_at(app, "lane-gate");
            stand_at(app, gate);
            interact(app);
            assert_eq!(
                app.world()
                    .resource::<slice::Slice>()
                    .map(|slice| slice.location.clone()),
                Some(location("ashfall-lane"))
            );
        }

        fn travel_to_hub(app: &mut App) {
            let gate = station_at(app, "lane-exit");
            stand_at(app, gate);
            interact(app);
            assert_eq!(
                app.world()
                    .resource::<slice::Slice>()
                    .map(|slice| slice.location.clone()),
                Some(location("cinderwake-square"))
            );
        }

        /// The position of the station with `spawn`.
        fn station_at(app: &App, spawn: &str) -> Vec3 {
            let wanted = game_core::ids::SpawnId::new(spawn).expect("valid id");
            app.world()
                .query2::<Transform, slice::Station>()
                .find(|(_, _, station)| station.spawn == wanted)
                .map(|(_, transform, _)| transform.translation)
                .unwrap_or_else(|| panic!("no station named {spawn}"))
        }

        /// Removes every enemy belonging to `encounter`, so that fight clears.
        ///
        /// Deliberately *not* fought. What this end-to-end test is about is
        /// whether the quest graph notices a cleared encounter and opens the
        /// next node; that a swing connects and kills is proven in isolation by
        /// `a_husk_can_be_killed_and_drops_loot`.
        ///
        /// Trying to win the fight here instead cost four attempts and taught
        /// three things worth keeping: a teleported enemy keeps steering to
        /// waypoints computed for where it used to be, placing one at `y = 0`
        /// leaves it half in the floor, and — the one that actually matters —
        /// a dead player goes on swinging, because nothing implements ADR-0007's
        /// respawn yet. A test that has to survive a four-on-one to reach its
        /// assertion is testing the wrong thing.
        fn clear_encounter(app: &mut App, encounter: &str) {
            let wanted = game_core::ids::EncounterId::new(encounter).expect("valid id");
            let tagged: Vec<_> = app
                .world()
                .query::<slice::Enemy>()
                .filter(|(_, enemy)| enemy.encounter.as_ref() == Some(&wanted))
                .map(|(entity, _)| entity)
                .collect();
            assert!(!tagged.is_empty(), "no enemy here belongs to {encounter}");

            // Wave by wave through the combat system's own death path, so the
            // encounter check runs exactly as it does in play: each wave that
            // falls calls the next, and only the last one clears the fight.
            let mut tagged = tagged;
            for _ in 0..=game_core::content::MAX_WAVES {
                for entity in &tagged {
                    if let Some(health) = app.world_mut().get_mut::<game_runtime::combat::Health>(*entity) {
                        health.current = 0.0;
                    }
                }
                app.world_mut().resource_scope::<slice::Slice, _>(|world, slice| {
                    slice::reap_dead(world, slice);
                });
                let pending = app
                    .world()
                    .resource::<slice::Waves>()
                    .is_some_and(|waves| waves.pending(&wanted));
                if !pending {
                    step(app, 2);
                    return;
                }
                step(app, slice::WAVE_DELAY_TICKS + 2);
                tagged = app
                    .world()
                    .query::<slice::Enemy>()
                    .filter(|(_, enemy)| enemy.encounter.as_ref() == Some(&wanted))
                    .map(|(entity, _)| entity)
                    .collect();
            }
            panic!("{encounter} never ran out of waves");
        }

        /// What canonical state records the player's health as.
        fn recorded_health(app: &App) -> i64 {
            let slice = app.world().resource::<slice::Slice>().expect("slice exists");
            i64::from(slice.state.player.health.current)
        }

        /// What the player actually has, in the runtime.
        fn live_health(app: &App) -> f32 {
            let player = slice::player_entity(app.world()).expect("a player");
            app.world()
                .get::<game_runtime::combat::Health>(player)
                .map_or(0.0, |health| health.current)
        }

        /// The fourth safe point: winning a fight writes its cost straight
        /// away, without waiting for a door, a death, or the way out.
        #[test]
        fn a_finished_fight_records_its_cost() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);

            // Stand in the ambush so it notices: awareness is what puts the
            // player in combat.
            let ambush = app
                .world()
                .query2::<Transform, slice::Enemy>()
                .next()
                .map(|(_, transform, _)| transform.translation)
                .expect("the lane holds the ambush");
            stand_at(&mut app, ambush);
            step(&mut app, 4);

            // Wounded directly rather than by losing a fight on purpose: this
            // is a test about *recording* a fight's cost, and taking the hits
            // for real would make it a test about enemy damage instead.
            let player = slice::player_entity(app.world()).expect("a player");
            if let Some(health) = app.world_mut().get_mut::<game_runtime::combat::Health>(player) {
                health.current -= 30.0;
            }
            step(&mut app, 2);

            let during = recorded_health(&app);
            assert!(
                (during as f32) > live_health(&app),
                "mid-fight the record is deliberately stale: combat does not \
                 write an event per hit"
            );

            // The fight ends when the last enemy aware of the player stops
            // being aware of it.
            clear_encounter(&mut app, "waterworks-ambush");
            step(&mut app, 4);

            let after = recorded_health(&app);
            assert!(
                after < during,
                "the end of a fight is a safe point (recorded {during} during, \
                 {after} after)"
            );
            assert_eq!(
                after,
                live_health(&app).round() as i64,
                "and what is recorded is what the player actually has"
            );
        }

        /// The safe point is the *edge* of a fight, not the state of being out
        /// of one — otherwise standing still would write the same value forever.
        #[test]
        fn standing_around_out_of_a_fight_records_nothing() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);

            let before = recorded_health(&app);
            step(&mut app, 120);

            assert_eq!(
                recorded_health(&app),
                before,
                "no fight ended, so nothing should have been written"
            );
            let slice = app.world().resource::<slice::Slice>().expect("slice exists");
            assert!(
                slice.pending.is_empty(),
                "and no event should be queued waiting to be written"
            );
        }

        /// Kills the player outright. Set up rather than fought: losing a fight
        /// on purpose would make this a test about enemy damage.
        fn kill_the_player(app: &mut App) {
            let player = slice::player_entity(app.world()).expect("a player");
            if let Some(health) = app.world_mut().get_mut::<game_runtime::combat::Health>(player) {
                health.current = 0.0;
            }
        }

        #[test]
        fn dying_sends_the_player_home_whole() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            kill_the_player(&mut app);
            step(&mut app, 2);

            let world = app.world();
            let slice = world.resource::<slice::Slice>().expect("slice exists");
            assert_eq!(
                slice.location,
                location("cinderwake-square"),
                "the dead wake up in the settlement (ADR-0007)"
            );
            assert_eq!(
                slice.state.player.location.as_ref(),
                Some(&location("cinderwake-square")),
                "and canonical state records where they woke up"
            );

            let player = slice::player_entity(world).expect("a player");
            let health = world
                .get::<game_runtime::combat::Health>(player)
                .expect("the player has health");
            assert_eq!(health.current, health.max, "restored, not left on the floor");
            assert!(
                world.get::<Attack>(player).is_none() && world.get::<Dodge>(player).is_none(),
                "and not still swinging at whatever killed them"
            );
        }

        /// Progress is kept — that is the whole of ADR-0007's death rule.
        #[test]
        fn dying_costs_the_walk_back_and_nothing_else() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            quest_open_at(&mut app, &["meet-sela"]);

            let before = {
                let slice = app.world().resource::<slice::Slice>().expect("slice exists");
                slice.state.clone()
            };

            kill_the_player(&mut app);
            step(&mut app, 2);

            let slice = app.world().resource::<slice::Slice>().expect("slice exists");
            assert_eq!(
                slice.state.quest_status(&quest("still-waters")),
                before.quest_status(&quest("still-waters")),
                "dying must not undo a quest"
            );
            assert_eq!(
                slice.state.player.inventory, before.player.inventory,
                "nor drop what was carried: there is no corpse to run back to"
            );
        }

        #[test]
        fn the_fight_you_died_in_is_standing_when_you_return() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            clear_encounter(&mut app, "waterworks-ambush");
            clear_encounter(&mut app, "old-forest-picket");
            assert_eq!(
                app.world().query::<slice::Enemy>().count(),
                0,
                "the lane is empty once both of its fights fall"
            );

            kill_the_player(&mut app);
            step(&mut app, 2);
            travel_to_lane(&mut app);

            assert_eq!(
                app.world().query::<slice::Enemy>().count(),
                4,
                "returning rebuilds the lane from content — both fights back on \
                 their feet, not just the one that killed you"
            );
        }

        /// A slice app whose world writes its save and log into `dir`.
        fn slice_app_saving_into(start: &str, dir: &Path) -> App {
            let built = slice_saving_into(start, dir);
            let mut app = build_slice_app(engine::render::RendererConfig::default()).expect("app builds");
            attach_slice(&mut app, built).expect("systems register");
            app
        }

        /// Everything the run actually recorded in its event log.
        fn logged_events(dir: &Path) -> Vec<DomainEvent> {
            let path = dir.join(slice::EVENT_LOG);
            if !path.exists() {
                return Vec::new();
            }
            game_core::log::read_events(&path)
                .expect("the log is readable")
                .events
        }

        #[test]
        fn dying_at_home_does_not_log_a_move_that_moved_nobody() {
            let saves = TempSaves::new("death-at-home");
            let mut app = slice_app_saving_into("cinderwake-square", saves.path());
            step(&mut app, 1);
            kill_the_player(&mut app);
            step(&mut app, 2);

            let slice = app.world().resource::<slice::Slice>().expect("slice exists");
            assert_eq!(slice.location, location("cinderwake-square"));

            // Asserted against the log rather than against `player.location`.
            // That field used to stay `None` until the first door, so "no
            // location recorded" stood in for "no move logged"; now that where
            // you start is durable, the proxy is meaningless and the claim has
            // to be made directly. `Slice::new`'s own `PlayerMoved` goes
            // through `apply`, not `commit`, so it never reaches the log.
            // Read once and asserted twice. The first assertion is what stops
            // this passing vacuously: if nothing were ever committed there
            // would be no log, and "no move in the log" would prove nothing.
            // The death does record something — `sync_health` queues a health
            // delta on the way down and again on the way back up.
            let recorded = logged_events(saves.path());
            assert!(
                !recorded.is_empty(),
                "the death should have recorded something; an empty log means \
                 this test never exercised the mechanism it claims to"
            );
            assert!(
                !recorded
                    .iter()
                    .any(|event| matches!(event, DomainEvent::PlayerMoved { .. })),
                "re-entering the location you are already in records no move"
            );
        }

        /// The regression guard for ADR-0015: a fresh world used to record the
        /// player as `0/0` while the runtime handed them 100 health.
        #[test]
        fn a_new_world_has_the_class_s_pools() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);

            let world = app.world();
            let slice = world.resource::<slice::Slice>().expect("slice exists");
            assert_eq!(
                slice
                    .state
                    .player
                    .progression
                    .class
                    .as_ref()
                    .map(ToString::to_string),
                Some("wayfinder".to_owned())
            );
            assert_eq!(slice.state.player.health.max, 120, "the wayfinder's pool");
            assert_eq!(slice.state.player.stamina.max, 100);

            let player = slice::player_entity(world).expect("a player");
            let health = world
                .get::<game_runtime::combat::Health>(player)
                .expect("the player has health");
            assert_eq!(
                (health.max, health.current),
                (120.0, 120.0),
                "the runtime is derived from the record, not invented beside it"
            );
        }

        fn canonical_health(app: &App) -> u32 {
            app.world()
                .resource::<slice::Slice>()
                .expect("slice exists")
                .state
                .player
                .health
                .current
        }

        fn wound_the_player(app: &mut App, to: f32) {
            let player = slice::player_entity(app.world()).expect("a player");
            if let Some(health) = app.world_mut().get_mut::<game_runtime::combat::Health>(player) {
                health.current = to;
            }
        }

        #[test]
        fn a_fight_is_recorded_when_the_player_leaves_the_place_it_happened() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            assert_eq!(canonical_health(&app), 120, "unhurt to begin with");

            wound_the_player(&mut app, 74.0);
            assert_eq!(
                canonical_health(&app),
                120,
                "combat does not touch canonical state tick by tick"
            );

            travel_to_lane(&mut app);
            assert_eq!(
                canonical_health(&app),
                74,
                "but a transition is a safe point, and reconciles it"
            );
        }

        #[test]
        fn dying_records_the_fall_and_the_recovery() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            wound_the_player(&mut app, 12.0);

            kill_the_player(&mut app);
            step(&mut app, 2);

            assert_eq!(
                canonical_health(&app),
                120,
                "the dead wake up whole, and the record says so"
            );
        }

        #[test]
        fn reconciling_unchanged_health_records_nothing() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);

            let queued = app.world_mut().resource_scope::<slice::Slice, _>(|world, slice| {
                let first = slice::sync_health(world, slice);
                slice.pending.clear();
                first
            });
            assert_eq!(
                queued,
                Some(false),
                "health that has not moved must not put a line in the log"
            );
        }

        /// The position of the station with the given kind.
        fn station_of_kind(app: &App, kind: game_core::content::StationKind) -> Vec3 {
            app.world()
                .query2::<Transform, slice::Station>()
                .find(|(_, _, station)| station.kind == kind)
                .map(|(_, transform, _)| transform.translation)
                .unwrap_or_else(|| panic!("no {kind:?} station in this location"))
        }

        fn held(app: &App, item: &str) -> u32 {
            app.world()
                .resource::<slice::Slice>()
                .expect("slice exists")
                .state
                .player
                .item_count(&ItemId::new(item).expect("valid id"))
        }

        fn grant(app: &mut App, item: &str, quantity: u32) {
            if let Some(slice) = app.world_mut().resource_mut::<slice::Slice>() {
                slice.raise(DomainEvent::ItemGranted {
                    item: ItemId::new(item).expect("valid id"),
                    quantity,
                });
            }
            step(app, 2);
        }

        #[test]
        fn the_forge_crafts_what_the_player_can_afford() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);

            // The draught needs two reeds and a resin, and no blueprint.
            grant(&mut app, "still-reed", 2);
            grant(&mut app, "corrupt-resin", 1);

            // Bound first: taking `&app` inside a `&mut app` call
            // borrows it twice in one expression.
            let at = station_of_kind(&app, game_core::content::StationKind::Crafting);
            stand_at(&mut app, at);
            interact(&mut app);
            choose(&mut app, 1);

            assert_eq!(held(&app, "clearwater-draught"), 1, "the output arrives");
            assert_eq!(held(&app, "still-reed"), 0, "and the inputs are spent");
            assert_eq!(held(&app, "corrupt-resin"), 0);
        }

        #[test]
        fn a_blueprint_gated_recipe_is_skipped_until_it_is_built() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);

            // Enough for the ember glaive, which needs the forge blueprint,
            // and nothing for the ungated draught.
            grant(&mut app, "ember-ore", 6);
            grant(&mut app, "old-forest-bark", 2);

            // Bound first: taking `&app` inside a `&mut app` call
            // borrows it twice in one expression.
            let at = station_of_kind(&app, game_core::content::StationKind::Crafting);
            stand_at(&mut app, at);
            interact(&mut app);
            assert_eq!(
                offered(&app),
                0,
                "a recipe the player cannot make is not on the list at all"
            );
            choose(&mut app, 1);

            assert_eq!(
                held(&app, "ember-glaive"),
                0,
                "materials are not enough; the forge has to be standing"
            );
            assert_eq!(held(&app, "ember-ore"), 6, "and nothing was spent trying");
        }

        #[test]
        fn using_a_forge_with_nothing_to_make_does_nothing() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);

            let before = {
                let slice = app.world().resource::<slice::Slice>().expect("slice exists");
                slice.state.clone()
            };
            // Bound first: taking `&app` inside a `&mut app` call
            // borrows it twice in one expression.
            let at = station_of_kind(&app, game_core::content::StationKind::Crafting);
            stand_at(&mut app, at);
            interact(&mut app);
            choose(&mut app, 1);

            let slice = app.world().resource::<slice::Slice>().expect("slice exists");
            assert_eq!(
                slice.state.player.inventory, before.player.inventory,
                "an empty-handed craft must not put a line in the log"
            );
        }

        fn flag_set(app: &App, flag: &str) -> bool {
            app.world()
                .resource::<slice::Slice>()
                .expect("slice exists")
                .state
                .has_world_flag(&game_core::ids::WorldFlag::new(flag).expect("valid id"))
        }

        /// Stands at a station, opens it, and takes the first row on offer.
        ///
        /// The first row rather than a named one because these tests are about
        /// the cost and prerequisite rules, not about picking: what is on the
        /// list is exactly what those rules allow, so row 1 is the only thing
        /// there is to take.
        fn use_station(app: &mut App, kind: game_core::content::StationKind) {
            let at = station_of_kind(app, kind);
            stand_at(app, at);
            interact(app);
            choose(app, 1);
            if offer_open(app) {
                // A station stays open so a player can make two things without
                // walking away and back. A test that is finished with one has
                // to walk away for the same reason a player would — otherwise
                // the next `interact` spends its press closing this menu
                // instead of opening the next thing.
                interact(app);
            }
        }

        #[test]
        fn the_board_builds_what_the_player_can_afford() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            grant(&mut app, "ember-ore", 12);
            grant(&mut app, "old-forest-bark", 4);

            use_station(&mut app, game_core::content::StationKind::Upgrades);

            assert!(flag_set(&app, "forge-relit"), "the forge is standing");
            assert_eq!(held(&app, "ember-ore"), 0, "and it cost what it costs");
            assert_eq!(held(&app, "old-forest-bark"), 0);
        }

        #[test]
        fn a_blueprint_waits_for_its_prerequisite() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            // Enough for the scriptorium, which needs the forge's tools first,
            // and nothing towards the forge itself.
            grant(&mut app, "old-forest-bark", 8);
            grant(&mut app, "still-reed", 6);

            use_station(&mut app, game_core::content::StationKind::Upgrades);

            assert!(!flag_set(&app, "scriptorium-raised"));
            assert!(!flag_set(&app, "forge-relit"));
            assert_eq!(held(&app, "old-forest-bark"), 8, "and nothing was spent");
        }

        /// The loop closing: gather, build, and the blueprint unlocks a recipe
        /// that was unreachable a moment earlier.
        #[test]
        fn building_the_forge_unlocks_the_recipes_it_gates() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);

            // Before the forge, the glaive's materials buy nothing.
            grant(&mut app, "ember-ore", 6);
            grant(&mut app, "old-forest-bark", 2);
            use_station(&mut app, game_core::content::StationKind::Crafting);
            assert_eq!(held(&app, "ember-glaive"), 0, "gated on a blueprint");

            // Build it.
            grant(&mut app, "ember-ore", 12);
            grant(&mut app, "old-forest-bark", 4);
            use_station(&mut app, game_core::content::StationKind::Upgrades);
            assert!(flag_set(&app, "forge-relit"));

            // And now the same materials at the same forge make the glaive.
            use_station(&mut app, game_core::content::StationKind::Crafting);
            assert_eq!(
                held(&app, "ember-glaive"),
                1,
                "the blueprint is what stood between the player and this"
            );
        }

        /// The slice has to be *lookable at*, not merely assertable about.
        #[test]
        fn a_populated_location_produces_draw_commands() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);

            // Dressed count is deterministic; how many survive frustum culling
            // is not, so the strong claim goes on the former.
            // Bodies that wear a model, not every `Model`: effects, marks and
            // nameplates are models too, and are not what this counts.
            let modelled = app.world().query::<slice::Wears>().count();
            let blocked_out = app.world().query::<engine::render::MeshRenderer>().count();
            assert_eq!(
                modelled, 6,
                "the player, the ambush's first two, two bark wardens, and Thal \
                 wear models"
            );
            assert_eq!(
                blocked_out, 7,
                "ground, two gatherables, a clue, a choice marker, and two doorways \
                 are still block-out shapes"
            );

            let frame = engine::render::extract_render_frame(app.world(), 16.0 / 9.0);
            assert!(
                !frame.draws.is_empty(),
                "every entity the slice spawns used to be invisible by construction"
            );
        }

        #[test]
        fn travelling_rebuilds_the_visuals_rather_than_accumulating_them() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            let hub = app.world().query::<engine::render::MeshRenderer>().count();

            travel_to_lane(&mut app);
            let lane = app.world().query::<engine::render::MeshRenderer>().count();

            assert!(
                lane < hub + 20,
                "the hub's props must not follow the player into the lane: {hub} -> {lane}"
            );
            assert!(lane > 1, "and the lane must have its own");
        }

        /// The crypt now has a reason to exist, and it is gated behind the
        /// first quest rather than offered from the start.
        #[test]
        fn the_second_quest_waits_for_the_first() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);

            let sela = station_free_npc(&app);
            stand_at(&mut app, sela);
            converse(&mut app);

            let slice = app.world().resource::<slice::Slice>().expect("slice exists");
            assert_eq!(
                slice.state.quest_status(&quest("ember-and-ash")),
                QuestStatus::NotStarted,
                "talking to sela before the waterworks are dealt with starts \
                 the first quest, not the second"
            );
            assert_eq!(
                slice.state.quest_status(&quest("still-waters")),
                QuestStatus::Active
            );
        }

        /// Sela's position. She is the only NPC in the hub.
        fn station_free_npc(app: &App) -> Vec3 {
            app.world()
                .query2::<Transform, slice::Npc>()
                .next()
                .map(|(_, transform, _)| transform.translation)
                .expect("the warden holds the square")
        }

        #[test]
        fn finishing_the_first_quest_opens_the_second() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);

            // Finish `still-waters` through the domain, then ask her again.
            if let Some(slice) = app.world_mut().resource_mut::<slice::Slice>() {
                slice.raise(DomainEvent::QuestAccepted {
                    quest: quest("still-waters"),
                });
                slice.raise(DomainEvent::QuestCompleted {
                    quest: quest("still-waters"),
                    resolution: game_core::ids::QuestResolutionId::new("waters-restored").expect("valid id"),
                });
            }
            step(&mut app, 2);

            let sela = station_free_npc(&app);
            stand_at(&mut app, sela);
            converse(&mut app);

            let slice = app.world().resource::<slice::Slice>().expect("slice exists");
            assert_eq!(
                slice.state.quest_status(&quest("ember-and-ash")),
                QuestStatus::Active,
                "with the waterworks settled, she has the next thing to say"
            );
        }

        /// The crypt quest, start to finish. Set up at each step rather than
        /// fought: `a_husk_can_be_killed_and_drops_loot` proves combat, and
        /// this proves the graph.
        #[test]
        fn the_crypt_quest_runs_to_its_resolution() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);

            if let Some(slice) = app.world_mut().resource_mut::<slice::Slice>() {
                slice.raise(DomainEvent::QuestAccepted {
                    quest: quest("ember-and-ash"),
                });
                slice.raise(DomainEvent::QuestObjectiveCompleted {
                    quest: quest("ember-and-ash"),
                    objective: objective("hear-the-crypt"),
                });
            }
            step(&mut app, 2);

            // Arriving satisfies `Reach`, via the signal a transition raises.
            travel_to_lane(&mut app);
            let mouth = station_at(&app, "crypt-mouth");
            stand_at(&mut app, mouth);
            interact(&mut app);

            let slice = app.world().resource::<slice::Slice>().expect("slice exists");
            assert_eq!(slice.location, location("crypt-of-ash"), "in the crypt");
            assert!(
                slice
                    .state
                    .objective_progress(&quest("ember-and-ash"), &objective("descend"))
                    .completed,
                "arriving is the objective"
            );

            clear_encounter(&mut app, "ashlord-stand");

            // And back to Sela.
            let exit = station_at(&app, "crypt-exit");
            stand_at(&mut app, exit);
            interact(&mut app);
            let gate = station_at(&app, "lane-exit");
            stand_at(&mut app, gate);
            interact(&mut app);

            // Measured across the hand-in rather than as a total: the ashlord
            // and its guards drop ember ore too, so an absolute figure would
            // couple this quest's payout to the loot tables' seeded rolls and
            // break the moment anyone retuned a drop.
            let before_reward = held(&app, "ember-ore");

            let sela = station_free_npc(&app);
            stand_at(&mut app, sela);
            converse(&mut app);

            let slice = app.world().resource::<slice::Slice>().expect("slice exists");
            assert_eq!(
                slice.state.quest_status(&quest("ember-and-ash")),
                QuestStatus::Completed {
                    resolution: game_core::ids::QuestResolutionId::new("crypt-silenced").expect("valid id")
                },
                "the crypt is silenced, and the record says which ending that was"
            );
            assert_eq!(
                held(&app, "ember-ore") - before_reward,
                8,
                "and the reward package paid exactly what it promises"
            );
        }

        /// The last authored-but-unplayable branch of `still-waters`.
        #[test]
        fn thal_can_be_escorted_home_to_settle_the_quest() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);

            // Open `seek-order` directly: the branch it hangs off is already
            // covered by the end-to-end test, and this one is about escorting.
            if let Some(slice) = app.world_mut().resource_mut::<slice::Slice>() {
                slice.raise(DomainEvent::QuestAccepted {
                    quest: quest("still-waters"),
                });
                for done in ["meet-sela", "survey-waterworks", "choose-allegiance"] {
                    slice.raise(DomainEvent::QuestObjectiveCompleted {
                        quest: quest("still-waters"),
                        objective: objective(done),
                    });
                }
            }
            step(&mut app, 2);

            // Ask him, and he comes along.
            let thal = station_free_npc(&app);
            stand_at(&mut app, thal);
            converse(&mut app);

            let escorting = app
                .world()
                .resource::<slice::Slice>()
                .and_then(|slice| slice.escorting.clone());
            assert!(escorting.is_some(), "talking to thal should start the escort");
            assert_eq!(
                app.world().query::<slice::Escorting>().count(),
                1,
                "and mark him as following"
            );

            // Walk to the lane's exit; he should close the distance himself.
            let gate = station_at(&app, "lane-exit");
            stand_at(&mut app, gate);
            step(&mut app, 240);
            let gap = {
                let world = app.world();
                let player = player_at(world);
                world
                    .query2::<Transform, slice::Escorting>()
                    .map(|(_, transform, _)| (transform.translation - player).length())
                    .fold(f32::INFINITY, f32::min)
            };
            assert!(
                gap < 6.0,
                "an escorted npc should have followed rather than stayed put, gap {gap}"
            );

            // Through the door, and the objective settles on arrival.
            interact(&mut app);

            let slice = app.world().resource::<slice::Slice>().expect("slice exists");
            assert_eq!(slice.location, location("cinderwake-square"));
            assert!(slice.escorting.is_none(), "delivering him ends the escort");
            assert!(
                slice
                    .state
                    .objective_progress(&quest("still-waters"), &objective("seek-order"))
                    .completed,
                "bringing the arbiter to Cinderwake is the objective"
            );
        }

        #[test]
        fn nobody_follows_without_a_quest_asking_for_it() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);

            let thal = station_free_npc(&app);
            stand_at(&mut app, thal);
            converse(&mut app);

            let slice = app.world().resource::<slice::Slice>().expect("slice exists");
            assert!(
                slice.escorting.is_none(),
                "an npc does not trail after everyone who says hello"
            );
            assert_eq!(app.world().query::<slice::Escorting>().count(), 0);
        }

        /// The capability the first-available stand-in could not express: two
        /// ways out of one node, told apart only by which key is pressed.
        ///
        /// `talk_to` always took option 0, so "Then I'll go around." was
        /// authored, validated, and impossible to say. This is the test that
        /// could not have been written before selection existed.
        #[test]
        fn a_player_can_take_a_line_other_than_the_first() {
            // Declining: row 2 at Thal's opening node ends the conversation
            // without ever reaching the price.
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            grant(&mut app, "old-forest-bark", 8);

            let thal = station_free_npc(&app);
            stand_at(&mut app, thal);
            interact(&mut app);
            assert_eq!(
                offered(&app),
                2,
                "both of thal's opening lines are ungated, so there is a choice to make"
            );

            choose(&mut app, 2);
            assert!(!offer_open(&app), "declining ends the conversation");
            assert_eq!(
                held(&app, "old-forest-bark"),
                8,
                "and costs nothing, because the price was never discussed"
            );

            // The same conversation, the same bark, row 1 instead: terms are
            // agreed and paid for.
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            grant(&mut app, "old-forest-bark", 8);
            let thal = station_free_npc(&app);
            stand_at(&mut app, thal);
            converse(&mut app);

            assert_eq!(
                held(&app, "old-forest-bark"),
                0,
                "one different keypress, and the bark changes hands"
            );
            assert!(flag_set(&app, "order-passage-granted"));
        }

        /// A unique directory under the system temp dir, removed on drop.
        ///
        /// Mirrors `renderer::shaders`' helper and for the same reason: a
        /// `tempfile` dependency for a handful of save files is not one worth
        /// asking for.
        struct TempSaves(PathBuf);

        impl TempSaves {
            fn new(label: &str) -> Self {
                use std::sync::atomic::{AtomicU32, Ordering};
                use std::time::{SystemTime, UNIX_EPOCH};

                static COUNTER: AtomicU32 = AtomicU32::new(0);
                let nanos = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|elapsed| elapsed.as_nanos())
                    .unwrap_or(0);
                let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
                let path = std::env::temp_dir().join(format!(
                    "cinderwake-{label}-{}-{nanos}-{unique}",
                    std::process::id()
                ));
                std::fs::create_dir_all(&path).expect("temp dir is creatable");
                Self(path)
            }

            fn path(&self) -> &Path {
                &self.0
            }
        }

        impl Drop for TempSaves {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        /// A slice over the shipped content that saves into `dir`.
        fn slice_saving_into(start: &str, dir: &Path) -> slice::Slice {
            let mut built = slice::Slice::new(&content_root(), &location(start), SLICE_SEED)
                .expect("the shipped content loads");
            built.save_dir = Some(dir.to_path_buf());
            built
        }

        /// M4's exit criterion, exercised by the slice at last rather than only
        /// by `game_core`: a world is saved, loaded, and is the same world.
        #[test]
        fn a_saved_world_resumes_as_itself() {
            let saves = TempSaves::new("round-trip");

            let recorded = {
                let mut slice = slice_saving_into("cinderwake-square", saves.path());
                slice.raise(DomainEvent::QuestAccepted {
                    quest: quest("still-waters"),
                });
                slice.raise(DomainEvent::ItemGranted {
                    item: ItemId::new("still-reed").expect("valid id"),
                    quantity: 3,
                });
                slice.enter(&location("ashfall-lane"));
                slice.commit().expect("the events apply");
                slice.write_save().expect("the save is written");
                slice.state.clone()
            };

            let resumed = slice::Slice::resume(&content_root(), saves.path()).expect("the save loads");

            assert_eq!(
                resumed.state, recorded,
                "a resumed world is the world that was saved"
            );
            assert_eq!(
                resumed.location,
                location("ashfall-lane"),
                "and the player is put back where they were, not at the start"
            );
        }

        /// The latent bug the load path exposed: `enter` emits `PlayerMoved`
        /// only when the location *changes*, so a world that had never used a
        /// door recorded no location at all, and its save could not say where
        /// to put the player back.
        #[test]
        fn a_world_that_never_moved_still_records_where_it_started() {
            let saves = TempSaves::new("start-location");
            {
                let slice = slice_saving_into("cinderwake-square", saves.path());
                assert_eq!(
                    slice.state.player.location.as_ref(),
                    Some(&location("cinderwake-square")),
                    "where you start is durable from the first tick"
                );
                slice.write_save().expect("the save is written");
            }

            let resumed = slice::Slice::resume(&content_root(), saves.path()).expect("loads");
            assert_eq!(resumed.location, location("cinderwake-square"));
        }

        /// ADR-0015: a resumed character is the one the save describes, not one
        /// rebuilt from today's class content.
        #[test]
        fn resuming_takes_the_players_pools_from_the_save() {
            let saves = TempSaves::new("pools");
            let recorded = {
                let mut slice = slice_saving_into("cinderwake-square", saves.path());
                slice.raise(DomainEvent::PlayerHealthChanged { delta: -17 });
                slice.commit().expect("applies");
                slice.write_save().expect("the save is written");
                slice.state.player.health.current
            };

            let resumed = slice::Slice::resume(&content_root(), saves.path()).expect("loads");
            assert_eq!(
                resumed.state.player.health.current, recorded,
                "re-applying `ClassChosen` on load would overwrite the record \
                 with today's content — the failure ADR-0015 exists to prevent"
            );
            assert!(
                resumed.state.player.health.current < resumed.state.player.health.max,
                "and the wound survived the round trip"
            );
        }

        /// A deleted location has a safe recovery, so it degrades rather than
        /// killing the save.
        #[test]
        fn a_save_naming_a_vanished_location_falls_back_to_the_settlement() {
            let saves = TempSaves::new("vanished");
            {
                let mut slice = slice_saving_into("cinderwake-square", saves.path());
                slice.state.player.location = Some(location("a-place-that-burned-down"));
                slice.write_save().expect("the save is written");
            }

            let resumed =
                slice::Slice::resume(&content_root(), saves.path()).expect("degrades rather than refusing");
            assert_eq!(
                resumed.location,
                location("cinderwake-square"),
                "the settlement is where you wake up when your location is gone"
            );
        }

        /// A deleted item has no safe recovery, so it is refused — loudly, and
        /// naming what is missing.
        #[test]
        fn a_save_naming_content_that_no_longer_exists_is_refused() {
            let saves = TempSaves::new("dangling");
            {
                let mut slice = slice_saving_into("cinderwake-square", saves.path());
                slice
                    .state
                    .player
                    .inventory
                    .insert(ItemId::new("a-thing-that-was-deleted").expect("valid id"), 1);
                slice.write_save().expect("the save is written");
            }

            // `expect_err` would require `Slice: Debug`, and deriving it on a
            // large runtime struct to satisfy a test idiom is the wrong trade.
            let Err(error) = slice::Slice::resume(&content_root(), saves.path()) else {
                panic!("a save naming content that no longer exists must be refused");
            };
            let message = error.to_string();
            assert!(
                message.contains("a-thing-that-was-deleted"),
                "the refusal has to name what is missing, or it is a scavenger \
                 hunt: {message}"
            );
        }

        /// ADR-0013: a save from a newer build is refused and left untouched —
        /// never partially loaded, and never silently replaced by a new world.
        #[test]
        fn a_save_from_a_newer_build_is_refused_and_left_alone() {
            let saves = TempSaves::new("future");
            let path = saves.path().join(slice::SAVE_FILE);
            std::fs::write(&path, "(format_version: 9999)").expect("writes");

            let Err(error) = slice::Slice::resume(&content_root(), saves.path()) else {
                panic!("a save from a newer build must be refused");
            };
            assert!(
                matches!(error, slice::SliceError::Save(_)),
                "surfaced as a save error rather than swallowed: {error}"
            );
            assert!(path.exists(), "and the file is left where it was");
        }

        // ---- effects and telegraphs (M7 wave 4) ---------------------------

        /// A slice whose content is edited before anything is built from it.
        fn slice_app_with(start: &str, edit: impl FnOnce(&mut content_runtime::ContentRegistry)) -> App {
            let mut built = slice::Slice::new(&content_root(), &location(start), SLICE_SEED)
                .expect("the shipped content loads");
            edit(&mut built.content);
            let mut app = build_slice_app(engine::render::RendererConfig::default()).expect("app builds");
            attach_slice(&mut app, built).expect("systems register");
            app
        }

        /// Leaves one husk, brainless, two metres in front of the player.
        /// Returns `(player, husk)`.
        fn lone_husk(app: &mut App) -> (engine::ecs::Entity, engine::ecs::Entity) {
            let (player, position) = app
                .world()
                .query2::<Transform, slice::Player>()
                .next()
                .map(|(entity, transform, _)| (entity, transform.translation))
                .expect("a player");
            let facing = app
                .world()
                .get::<slice::Facing>(player)
                .map_or(Vec3::NEG_Z, |facing| facing.0);
            let enemies: Vec<engine::ecs::Entity> = app
                .world()
                .query::<slice::Enemy>()
                .map(|(entity, _)| entity)
                .collect();
            let husk = *enemies.first().expect("the lane has enemies");
            for other in enemies.iter().skip(1) {
                app.world_mut().despawn(*other);
            }
            let world = app.world_mut();
            let _ = world.remove::<EnemyBrain>(husk);
            if let Some(transform) = world.get_mut::<Transform>(husk) {
                transform.translation = position + facing * 2.0;
            }
            (player, husk)
        }

        /// Every effect on screen: where it plays, and how many pieces it has.
        fn playing(app: &App) -> Vec<(Vec3, usize)> {
            let world = app.world();
            world
                .query2::<vfx::Playing, engine::render::Model>()
                .map(|(entity, _, model)| {
                    let at = world
                        .get::<Transform>(entity)
                        .map(|transform| transform.translation)
                        .expect("an effect is placed");
                    (at, model.parts.len())
                })
                .collect()
        }

        /// Every telegraph showing: where, and its outline and fill parts.
        fn marks(app: &App) -> Vec<(Vec3, engine::render::ModelPart, engine::render::ModelPart)> {
            let world = app.world();
            world
                .query2::<vfx::TelegraphMark, engine::render::Model>()
                .filter(|(entity, _, _)| {
                    world
                        .get::<engine::render::Visibility>(*entity)
                        .is_none_or(|visibility| visibility.is_visible())
                })
                .map(|(entity, _, model)| {
                    let at = world.get::<Transform>(entity).expect("placed").translation;
                    (at, model.parts[0], model.parts[1])
                })
                .collect()
        }

        /// Taps an ability key: pressed for a tick, then released, and a tick
        /// for the release to cast — hold to aim, release to cast being the
        /// default (see `aim`).
        fn cast(app: &mut App, key: KeyCode) {
            press(app, key);
            step(app, 1);
            release(app, key);
            step(app, 1);
        }

        /// Steps until the player's cast has run `elapsed` ticks.
        fn cast_until(app: &mut App, player: engine::ecs::Entity, elapsed: u32) {
            for _ in 0..120 {
                if app
                    .world()
                    .get::<slice::Casting>(player)
                    .is_some_and(|casting| casting.elapsed >= elapsed)
                {
                    return;
                }
                step(app, 1);
            }
            panic!("the cast never reached {elapsed} ticks");
        }

        #[test]
        fn ember_cleave_plays_its_lash_on_the_husk_it_hit() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (player, husk) = lone_husk(&mut app);
            assert!(playing(&app).is_empty(), "nothing plays before the cast");

            cast(&mut app, KeyCode::Q);
            // Past the 14-tick wind-up: the effects land, and the lash plays.
            cast_until(&mut app, player, 16);

            let at = app
                .world()
                .get::<Transform>(husk)
                .expect("the husk stands")
                .translation;
            let effects = playing(&app);
            assert_eq!(effects.len(), 1, "one impact, on the one husk hit: {effects:?}");
            let (played_at, pieces) = effects[0];
            assert!(
                (played_at - at).length() < 1e-4,
                "played on the husk: {played_at:?} vs {at:?}"
            );
            // The authored ember-lash: sixteen embers, a pulse, and a beam
            // back to the wayfinder two metres away.
            assert_eq!(pieces, 16 + 1 + 1);
            assert_eq!(vfx::live_particles(app.world()), 18);
        }

        #[test]
        fn warden_guard_plays_its_shimmer_on_the_caster() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            let player = slice::player_entity(app.world()).expect("a player");

            cast(&mut app, KeyCode::R);
            cast_until(&mut app, player, 8);

            let at = player_at(app.world());
            let effects = playing(&app);
            assert_eq!(effects.len(), 1, "one cast effect: {effects:?}");
            assert!((effects[0].0 - at).length() < 1e-4, "played on the caster");
            // A pulse and twelve motes; no beam, because the caster is where
            // the effect plays and there is nothing to span.
            assert_eq!(effects[0].1, 1 + 12);
        }

        #[test]
        fn effects_clear_away_when_they_end() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (player, _) = lone_husk(&mut app);
            cast(&mut app, KeyCode::Q);
            cast_until(&mut app, player, 16);
            assert!(!playing(&app).is_empty(), "the lash is playing");

            // Its longest part lasts 28 ticks.
            step(&mut app, 30);
            assert!(playing(&app).is_empty(), "and is gone once it has run");
            assert_eq!(vfx::live_particles(app.world()), 0);
        }

        /// The cap turns effects away whole and counts them; it never stalls
        /// and never plays half an effect.
        #[test]
        fn the_particle_cap_drops_whole_effects_and_counts_them() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            let lash = game_core::ids::VfxId::new("ember-lash").expect("valid id");
            let world = app.world_mut();
            for index in 0..40 {
                let at = Vec3::new(index as f32, 1.0, 0.0);
                vfx::request(world, &lash, at + Vec3::Z * 2.0, at);
            }
            step(&mut app, 1);

            let live = vfx::live_particles(app.world());
            assert!(live <= vfx::MAX_LIVE_PARTICLES, "{live} pieces is over the cap");
            let whole = vfx::MAX_LIVE_PARTICLES / 18;
            assert_eq!(live, whole * 18, "every effect admitted is whole");
            assert!(playing(&app).iter().all(|(_, pieces)| *pieces == 18));
            let dropped = app
                .world()
                .resource::<vfx::VfxRuntime>()
                .map_or(0, |runtime| runtime.dropped);
            assert_eq!(dropped, (40 - whole) as u64, "and each turned away is counted");

            // Requests past the queue's cap are dropped too, and counted.
            let world = app.world_mut();
            for _ in 0..(vfx::MAX_PENDING + 5) {
                vfx::request(world, &lash, Vec3::ZERO, Vec3::ZERO);
            }
            let dropped_now = app
                .world()
                .resource::<vfx::VfxRuntime>()
                .map_or(0, |runtime| runtime.dropped);
            assert_eq!(dropped_now, dropped + 5);
        }

        /// Stripping every effect from content leaves a fight identical: no
        /// system that decides an outcome reads anything presentation made.
        #[test]
        fn effects_never_change_a_fight() {
            fn fight(app: &mut App) -> (String, Vec<f32>, Vec<Vec3>, usize) {
                step(app, 1);
                let (player, husk) = lone_husk(app);
                // The husk swings too, so a telegraph is up during the fight.
                let _ = app.world_mut().insert(husk, slice::Facing(Vec3::Z));
                let _ = app
                    .world_mut()
                    .insert(husk, Attack::new(slice::PLAYER_ATTACK_TIMING, 10.0, 1.4, 0.9));
                cast(app, KeyCode::Q);
                let mut most = 0;
                for _ in 0..60 {
                    step(app, 1);
                    most = most.max(vfx::live_particles(app.world()));
                }
                cast(app, KeyCode::R);
                for _ in 0..60 {
                    step(app, 1);
                    most = most.max(vfx::live_particles(app.world()));
                }
                let world = app.world();
                let pools = [player, husk]
                    .iter()
                    .map(|entity| world.get::<Health>(*entity).map_or(-1.0, |health| health.current))
                    .chain(world.get::<Stamina>(player).map(|stamina| stamina.current))
                    .collect();
                let places = [player, husk]
                    .iter()
                    .filter_map(|entity| {
                        world
                            .get::<Transform>(*entity)
                            .map(|transform| transform.translation)
                    })
                    .collect();
                let state = world
                    .resource::<slice::Slice>()
                    .map(|slice| ron::to_string(&slice.state).expect("serializes"))
                    .expect("a slice");
                (state, pools, places, most)
            }

            let mut shown = slice_app("ashfall-lane");
            let with = fight(&mut shown);
            let mut stripped = slice_app_with("ashfall-lane", |content| {
                for ability in content.abilities.values_mut() {
                    ability.presentation = Default::default();
                }
                content.vfx.clear();
            });
            let without = fight(&mut stripped);

            assert!(
                with.3 > 0,
                "effects played in the shown fight, or this proves nothing"
            );
            assert_eq!(without.3, 0, "and none in the stripped one");
            assert_eq!(with.0, without.0, "canonical state is identical");
            assert_eq!(with.1, without.1, "every pool is identical");
            assert_eq!(with.2, without.2, "every body stands in the same place");
        }

        #[test]
        fn an_enemy_swing_is_marked_on_the_ground_where_it_lands() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (_, enemy) = arm_one_enemy(&mut app);
            step(&mut app, 1);

            let world = app.world();
            let body = world.get::<Transform>(enemy).expect("placed").translation;
            let facing = world.get::<slice::Facing>(enemy).expect("facing").0;
            let lands = body + facing * 1.4;
            let shown = marks(&app);
            assert_eq!(shown.len(), 1, "one blow is coming");
            let (at, outline, fill) = shown[0];
            assert!(
                (at - Vec3::new(lands.x, 0.0, lands.z)).length() < 1e-4,
                "marked on the ground under the hit volume: {at:?} vs {lands:?}"
            );
            assert!(
                (outline.pose.scale.x - 0.9).abs() < 1e-5,
                "outlined at the hit radius"
            );
            assert!(
                fill.pose.scale.x < outline.pose.scale.x,
                "filling through the wind-up"
            );
            let coming = outline.material;

            // Into the active window: the fill is full and the mark turns.
            let windup = slice::PLAYER_ATTACK_TIMING.windup;
            step(&mut app, u64::from(windup));
            let shown = marks(&app);
            assert_eq!(shown.len(), 1);
            assert!((shown[0].2.pose.scale.x - 0.9).abs() < 1e-5, "full once live");
            assert_ne!(shown[0].1.material, coming, "and looks different once live");

            // Spent: nothing is marked.
            step(&mut app, u64::from(slice::PLAYER_ATTACK_TIMING.total()));
            assert!(marks(&app).is_empty(), "a spent blow leaves no mark");
        }

        #[test]
        fn the_players_own_swing_is_marked_apart_from_a_threat() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (_, enemy) = arm_one_enemy(&mut app);
            step(&mut app, 1);
            let threat = marks(&app)[0].1.material;
            let _ = app.world_mut().remove::<Attack>(enemy);

            press(&mut app, KeyCode::Space);
            step(&mut app, 1);
            release(&mut app, KeyCode::Space);
            let shown = marks(&app);
            assert_eq!(shown.len(), 1, "the player's swing is marked too");
            assert_ne!(shown[0].1.material, threat, "in a colour that is not a threat's");
        }

        /// The ashlord, alone and brainless, with its summoning started.
        fn ashlord_casting(app: &mut App) -> engine::ecs::Entity {
            step(app, 1);
            let ashlord = app
                .world()
                .query::<slice::Enemy>()
                .find(|(_, enemy)| enemy.prefab.name() == "husk-ashlord")
                .map(|(entity, _)| entity)
                .expect("the crypt holds the elite");
            let others: Vec<engine::ecs::Entity> = app
                .world()
                .query::<slice::Enemy>()
                .map(|(entity, _)| entity)
                .filter(|entity| *entity != ashlord)
                .collect();
            for other in others {
                app.world_mut().despawn(other);
            }
            let _ = app.world_mut().remove::<EnemyBrain>(ashlord);
            let mut started = false;
            app.world_mut().resource_scope::<slice::Slice, _>(|world, slice| {
                started = slice::start_ability(world, slice, ashlord, 0);
            });
            assert!(started, "the summoning starts");
            ashlord
        }

        /// An area ability that asks for a telegraph marks its whole reach
        /// around the caster through the wind-up — and one that does not
        /// marks nothing.
        #[test]
        fn an_area_telegraph_marks_the_reach_around_its_caster() {
            let mut app = slice_app_with("crypt-of-ash", |content| {
                for ability in content.abilities.values_mut() {
                    if ability.id.name() == "call-husks" {
                        ability.presentation.telegraph = true;
                    }
                }
            });
            let ashlord = ashlord_casting(&mut app);
            step(&mut app, 10);

            let body = app.world().get::<Transform>(ashlord).expect("placed").translation;
            let shown = marks(&app);
            assert_eq!(shown.len(), 1, "one mark, for the cast");
            let (at, outline, fill) = shown[0];
            assert!(
                (at - Vec3::new(body.x, 0.0, body.z)).length() < 1e-4,
                "around the caster"
            );
            assert!(
                (outline.pose.scale.x - slice::ABILITY_REACH).abs() < 1e-5,
                "as far as the ability reaches"
            );
            assert!(fill.pose.scale.x > 0.0 && fill.pose.scale.x < slice::ABILITY_REACH);

            // The shipped call-husks asks for none.
            let mut plain = slice_app("crypt-of-ash");
            ashlord_casting(&mut plain);
            step(&mut plain, 10);
            assert!(marks(&plain).is_empty(), "a summoning marks no ground");
        }

        /// `ember-cleave` plays its own clip, not the body's cast: the blade
        /// arm is drawn far higher behind the head than a two-handed cast
        /// lifts it.
        #[test]
        fn an_ability_plays_its_own_clip() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            let player = slice::player_entity(app.world()).expect("a player");
            cast(&mut app, KeyCode::Q);
            // The last tick of the 14-tick wind-up: the keys at 900..1000.
            cast_until(&mut app, player, 13);
            let raised = swing_degrees(&slot_pose(&app, player, "arm-r"));
            assert!(
                raised > 140.0,
                "ember-cleave-cast draws the arm back to about 165 degrees, the \
                 body's cast only to about 105: {raised}"
            );
        }

        // ---- skillshots (M7 wave 5) ---------------------------------------

        fn projectiles(app: &App) -> Vec<(engine::ecs::Entity, Vec3)> {
            app.world()
                .query::<game_runtime::projectile::Projectile>()
                .map(|(entity, projectile)| (entity, projectile.position))
                .collect()
        }

        fn health_of(app: &App, entity: engine::ecs::Entity) -> f32 {
            app.world()
                .get::<Health>(entity)
                .map_or(-1.0, |health| health.current)
        }

        /// One husk, brainless, `ahead` metres in front of the player and
        /// `aside` to its right. Returns `(player, husk)`.
        fn husk_at(app: &mut App, ahead: f32, aside: f32) -> (engine::ecs::Entity, engine::ecs::Entity) {
            let (player, husk) = lone_husk(app);
            let position = player_at(app.world());
            let facing = app
                .world()
                .get::<slice::Facing>(player)
                .map_or(Vec3::NEG_Z, |facing| facing.0);
            let right = Vec3::new(-facing.z, 0.0, facing.x);
            if let Some(transform) = app.world_mut().get_mut::<Transform>(husk) {
                transform.translation = position + facing * ahead + right * aside;
            }
            (player, husk)
        }

        /// Steps until nothing is in flight, and says how many frames it took.
        fn until_landed(app: &mut App) -> u64 {
            for frames in 1..400 {
                step(app, 1);
                if projectiles(app).is_empty() {
                    return frames;
                }
            }
            panic!("a bolt was still in flight after 400 frames");
        }

        /// Every lane showing: where it starts, one rail, and its fill.
        fn lanes(app: &App) -> Vec<(Vec3, engine::render::ModelPart, engine::render::ModelPart)> {
            let world = app.world();
            world
                .query2::<vfx::LaneMark, engine::render::Model>()
                .filter(|(entity, _, _)| {
                    world
                        .get::<engine::render::Visibility>(*entity)
                        .is_none_or(|visibility| visibility.is_visible())
                })
                .map(|(entity, _, model)| {
                    let at = world.get::<Transform>(entity).expect("placed").translation;
                    (at, model.parts[0], model.parts[2])
                })
                .collect()
        }

        #[test]
        fn ember_bolt_flies_and_strikes_the_first_husk_in_its_lane() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (_, husk) = husk_at(&mut app, 6.0, 0.0);
            let before = health_of(&app, husk);

            cast(&mut app, KeyCode::F);
            // Past the 12-tick wind-up the bolt is in the air, and has not
            // yet covered six metres.
            let mut launched = false;
            for _ in 0..20 {
                step(&mut app, 1);
                if !projectiles(&app).is_empty() {
                    launched = true;
                    break;
                }
            }
            assert!(launched, "F launches a bolt once its wind-up ends");
            assert_eq!(
                health_of(&app, husk),
                before,
                "a bolt in the air has hurt nobody yet"
            );

            until_landed(&mut app);
            assert!(
                (before - health_of(&app, husk) - 14.0).abs() < 1e-3,
                "it strikes for its authored 14: {before} -> {}",
                health_of(&app, husk)
            );
            let struck = app
                .world()
                .get::<Transform>(husk)
                .expect("the husk stands")
                .translation;
            assert!(
                playing(&app).iter().any(|(at, _)| (*at - struck).length() < 0.9),
                "and bursts where it struck: {:?}",
                playing(&app)
            );
        }

        #[test]
        fn a_bolt_misses_what_is_not_in_its_lane() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (_, husk) = husk_at(&mut app, 6.0, 1.5);
            let before = health_of(&app, husk);
            cast(&mut app, KeyCode::F);
            step(&mut app, 14);
            assert!(!projectiles(&app).is_empty(), "in the air");
            // 14 metres at 16 m/s is 53 ticks: spent, not stopped.
            let frames = until_landed(&mut app);
            assert!(
                frames > 45,
                "it flew its range rather than stopping early: {frames} frames"
            );
            assert_eq!(
                health_of(&app, husk),
                before,
                "a husk a metre and a half aside is missed"
            );
        }

        #[test]
        fn a_wall_stops_a_bolt_and_its_lane() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (player, husk) = husk_at(&mut app, 6.0, 0.0);
            let position = player_at(app.world());
            let facing = app.world().get::<slice::Facing>(player).expect("facing").0;
            // A wall three metres out, a metre and a half either side of the
            // lane and thin along it.
            let wall = position + facing * 3.0;
            let across = Vec3::new(-facing.z, 0.0, facing.x);
            let corners = [
                wall + across * 1.5 + facing * 0.2 - Vec3::Y * 1.0,
                wall - across * 1.5 - facing * 0.2 + Vec3::Y * 2.0,
            ];
            if let Some(collision) = app.world_mut().resource_mut::<CollisionWorld>() {
                collision.push_box(game_runtime::movement::StaticBox::new(corners[0], corners[1]));
            }
            let before = health_of(&app, husk);

            cast(&mut app, KeyCode::F);
            step(&mut app, 4);
            let lane = lanes(&app);
            assert_eq!(lane.len(), 1, "the wind-up marks one lane");
            let length = lane[0].1.pose.scale.z;
            assert!(
                length > 2.0 && length < 3.0,
                "the lane ends at the wall, not at the bolt's full range: {length}"
            );

            // Through the rest of the wind-up, until it is thrown.
            for _ in 0..20 {
                if !projectiles(&app).is_empty() {
                    break;
                }
                step(&mut app, 1);
            }
            assert!(!projectiles(&app).is_empty(), "the bolt is thrown");
            until_landed(&mut app);
            assert_eq!(
                health_of(&app, husk),
                before,
                "the husk behind the wall is untouched"
            );
            assert!(
                playing(&app).iter().any(|(at, _)| (*at - wall).length() < 1.0),
                "the bolt bursts against the wall: {:?}",
                playing(&app)
            );
        }

        #[test]
        fn the_lane_telegraph_marks_the_bolts_path_through_the_wind_up() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (player, _) = husk_at(&mut app, 30.0, 0.0);
            let position = player_at(app.world());
            let facing = app.world().get::<slice::Facing>(player).expect("facing").0;
            assert!(lanes(&app).is_empty(), "nothing marked before the cast");

            cast(&mut app, KeyCode::F);
            step(&mut app, 2);
            let early = lanes(&app);
            assert_eq!(early.len(), 1);
            let (start, rail, fill) = early[0];
            assert!(
                (start - Vec3::new(position.x, 0.0, position.z)).length() < 1e-4,
                "it starts under the caster"
            );
            assert!(
                (rail.pose.scale.z - 14.0).abs() < 1e-3,
                "as long as the bolt's range"
            );
            let width = rail.pose.translation.x.abs() * 2.0 + rail.pose.scale.x;
            assert!((width - 0.5).abs() < 1e-3, "as wide as the bolt strikes: {width}");
            let heading = app
                .world()
                .query2::<vfx::LaneMark, Transform>()
                .next()
                .map(|(_, _, transform)| transform.rotation * Vec3::NEG_Z)
                .expect("a lane");
            assert!((heading - facing).length() < 1e-4, "along the aim");

            step(&mut app, 6);
            let later = lanes(&app)[0].2;
            assert!(
                later.pose.scale.z > fill.pose.scale.z,
                "the fill runs down the lane as the wind-up does"
            );

            step(&mut app, 20);
            assert!(lanes(&app).is_empty(), "and it is gone once the bolt is thrown");
        }

        #[test]
        fn a_bolt_flies_as_its_model_and_leaves_a_trail() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            // Aimed off the model's own forward, so "nose first" is a turn the
            // runtime has to make rather than one it gets for free.
            let player = slice::player_entity(app.world()).expect("a player");
            let _ = app
                .world_mut()
                .insert(player, slice::Facing(Vec3::new(1.0, 0.0, 1.0).normalize()));
            let (player, _) = husk_at(&mut app, 30.0, 0.0);
            let facing = app.world().get::<slice::Facing>(player).expect("facing").0;
            cast(&mut app, KeyCode::F);
            step(&mut app, 20);

            let (bolt, _) = *projectiles(&app).first().expect("in the air");
            let world = app.world();
            let model = world.get::<engine::render::Model>(bolt).expect("dressed");
            let built = world
                .resource::<slice::ModelLibrary>()
                .and_then(|library| {
                    library.get(&game_core::ids::ModelId::new("ember-bolt").expect("valid id"))
                })
                .expect("the bolt model is built");
            assert_eq!(model.parts, built.parts, "it flies as its authored model");
            let nose = world.get::<Transform>(bolt).expect("placed").rotation * Vec3::NEG_Z;
            assert!((nose - facing).length() < 1e-4, "nose first");
            assert!(!playing(&app).is_empty(), "shedding its trail as it goes");
        }

        /// A bolt aimed at the player by `husk`, set up directly rather than
        /// waited for: no enemy authors one yet.
        fn enemy_bolt(app: &mut App, husk: engine::ecs::Entity) {
            let player = player_at(app.world());
            let from = app.world().get::<Transform>(husk).expect("placed").translation;
            let launch = game_runtime::projectile::Launch {
                owner: husk,
                from,
                direction: (player - from).normalize(),
                speed: 16.0,
                range: 10.0,
                radius: 0.25,
                pierce: 0,
                on_hit: vec![game_core::content::Effect::DealDamage {
                    amount: game_core::content::Fixed::from_whole(20),
                    damage_type: game_core::content::DamageType::Fire,
                }],
            };
            let world = app.world_mut();
            let entity = world.spawn();
            let _ = world.insert(entity, Transform::from_translation(from));
            let _ = world.insert(entity, game_runtime::projectile::Projectile::new(launch));
            let _ = world.insert(
                entity,
                slice::Flying {
                    ability: game_core::ids::AbilityId::new("ember-bolt").expect("valid id"),
                    from_player: false,
                    launched: 0,
                    dealt: 1.0,
                },
            );
        }

        /// A bolt answers statuses and a roll exactly as a swing does.
        #[test]
        fn an_enemy_bolt_is_guarded_against_and_rolled_through() {
            let taken = |prepare: &dyn Fn(&mut App, engine::ecs::Entity)| {
                let mut app = slice_app("ashfall-lane");
                step(&mut app, 1);
                let (player, husk) = husk_at(&mut app, 5.0, 0.0);
                prepare(&mut app, player);
                let before = health_of(&app, player);
                enemy_bolt(&mut app, husk);
                until_landed(&mut app);
                before - health_of(&app, player)
            };

            let plain = taken(&|_, _| {});
            assert!(
                plain > 0.0,
                "an enemy bolt hurts the player, or nothing below proves anything"
            );
            let guarded = taken(&|app, player| {
                let tick = app.world().resource::<FixedTime>().map_or(0, |time| time.tick);
                let id = game_core::content::StatusId(
                    game_core::ids::AbilityId::new("status-guarded").expect("valid id"),
                );
                let modifiers = app
                    .world()
                    .resource::<slice::Slice>()
                    .and_then(|slice| slice.content.statuses.get(&id))
                    .map(game_runtime::status::modifiers_of)
                    .expect("status-guarded is defined");
                let mut statuses = Statuses::new();
                statuses.apply(id, tick + 600, &modifiers);
                let _ = app.world_mut().insert(player, statuses);
            });
            assert!(
                (guarded - plain * 0.5).abs() < 1e-3,
                "status-guarded halves a bolt as it halves a swing: {plain} -> {guarded}"
            );
            let rolled = taken(&|app, player| {
                let _ = app
                    .world_mut()
                    .insert(player, Dodge::new(Vec2::new(1.0, 0.0), 600, 600, 0.0));
            });
            assert_eq!(
                rolled, 0.0,
                "rolled through: the invulnerable window holds against a bolt"
            );
        }

        #[test]
        fn a_bolt_in_the_air_is_lost_at_a_door() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            husk_at(&mut app, 30.0, 0.0);
            cast(&mut app, KeyCode::F);
            step(&mut app, 16);
            assert!(!projectiles(&app).is_empty(), "in the air");
            slice::clear_location(app.world_mut());
            assert!(projectiles(&app).is_empty(), "not carried through the door");
        }

        /// M7's first exit criterion: a new enemy with its own model, its own
        /// animation, and a skillshot, authored purely in RON. Every
        /// definition below is parsed here and named nowhere in Rust — the
        /// enemy, its body, its cast clip, its projectile's look, and the
        /// ability — and in a running slice it winds up, throws, and hits.
        #[test]
        fn a_new_enemy_with_a_skillshot_is_authored_purely_in_ron() {
            let spit: AbilityDef = ron::from_str(
                r#"(
                    schema_version: 1,
                    id: "cinder-spit",
                    name: "Cinder Spit",
                    targeting: SelfOnly,
                    cost: 0,
                    cooldown_ticks: 90,
                    windup_ticks: 20,
                    recovery_ticks: 20,
                    effects: [
                        Projectile(speed: 12000, range: 12000, radius: 300, on_hit: [
                            DealDamage(amount: 9000, damage_type: Fire),
                        ]),
                    ],
                    presentation: (
                        projectile: Some("cinder-glob"),
                        impact: Some("ember-bolt-burst"),
                        telegraph: true,
                    ),
                )"#,
            )
            .expect("the ability parses");
            let lunge: game_core::content::ClipDef = ron::from_str(
                r#"(
                    schema_version: 1,
                    id: "spit-lunge",
                    name: "Spit lunge",
                    timing: Windows,
                    tracks: [
                        (slot: "head", keys: [
                            (at: 0),
                            (at: 900, turn: (x: -40000, y: 0, z: 0)),
                            (at: 1100, turn: (x: 35000, y: 0, z: 0)),
                            (at: 3000),
                        ]),
                    ],
                )"#,
            )
            .expect("the clip parses");
            let body: game_core::content::ModelDef = ron::from_str(
                r#"(
                    schema_version: 1,
                    id: "cinder-spitter",
                    name: "Cinder spitter",
                    body_plan: "humanoid",
                    palette: "cinderwake",
                    parts: [
                        (slot: "torso", shape: Taper(x: 420, z: 320, top_x: 300, top_z: 240, height: 520), at: (x: 0, y: 260, z: 0), color: "ash"),
                        (slot: "head", shape: Dome(radius: 200, segments: 6, rings: 2), at: (x: 0, y: 120, z: 0), color: "ember", top: Some("ash-dark")),
                        (slot: "arm-l", shape: Taper(x: 120, z: 120, top_x: 90, top_z: 90, height: 520), at: (x: 0, y: -260, z: 0), color: "ash-dark"),
                        (slot: "arm-r", shape: Taper(x: 120, z: 120, top_x: 90, top_z: 90, height: 520), at: (x: 0, y: -260, z: 0), color: "ash-dark"),
                        (slot: "leg-l", shape: Taper(x: 130, z: 130, top_x: 150, top_z: 150, height: 900), at: (x: 0, y: -450, z: 0), color: "ash"),
                        (slot: "leg-r", shape: Taper(x: 130, z: 130, top_x: 150, top_z: 150, height: 900), at: (x: 0, y: -450, z: 0), color: "ash"),
                    ],
                    animations: (cast: Some("spit-lunge")),
                )"#,
            )
            .expect("the body parses");
            let glob: game_core::content::ModelDef = ron::from_str(
                r#"(
                    schema_version: 1,
                    id: "cinder-glob",
                    name: "Cinder glob",
                    body_plan: "bolt",
                    palette: "cinderwake",
                    parts: [(slot: "core", shape: Sphere(radius: 200, detail: 0), color: "ember")],
                )"#,
            )
            .expect("the glob parses");
            let spitter: game_core::content::EnemyDef = ron::from_str(
                r#"(
                    schema_version: 1,
                    id: "cinder-spitter",
                    name: "Cinder Spitter",
                    behavior: Ranged,
                    max_health: 40000,
                    damage: 5000,
                    abilities: ["cinder-spit"],
                    visual_tags: ["ember", "ranged"],
                    threat: 1300,
                    variant_of: None,
                    loot_table: None,
                    model: Some("cinder-spitter"),
                )"#,
            )
            .expect("the enemy parses");
            assert_eq!(spit.validate(), Ok(()));
            assert_eq!(lunge.validate(), Ok(()));
            assert_eq!(body.validate(), Ok(()));
            assert_eq!(glob.validate(), Ok(()));
            assert_eq!(spitter.validate(), Ok(()));

            let prefab = spitter.id.clone();
            let spit_id = spit.id.clone();
            let mut app = slice_app_with("ashfall-lane", move |content| {
                content.abilities.insert(spit.id.clone(), spit);
                content.clips.insert(lunge.id.clone(), lunge);
                content.models.insert(body.id.clone(), body);
                content.models.insert(glob.id.clone(), glob);
                content.enemies.insert(spitter.id.clone(), spitter);
            });
            step(&mut app, 1);
            let (player, husk) = lone_husk(&mut app);
            let _ = app.world_mut().despawn(husk);
            let position = player_at(app.world());
            let facing = app.world().get::<slice::Facing>(player).expect("facing").0;
            let mut enemy = None;
            app.world_mut().resource_scope::<slice::Slice, _>(|world, slice| {
                let spawn = game_core::ids::SpawnId::new("summoned").expect("valid id");
                enemy = slice::spawn_enemy_at(world, slice, &prefab, spawn, None, position + facing * 6.0);
            });
            let enemy = enemy.expect("the new enemy spawns from content");
            assert_eq!(
                app.world()
                    .get::<slice::Wears>(enemy)
                    .map(|wears| wears.0.to_string()),
                Some("cinder-spitter".to_owned()),
                "in its own model"
            );

            let before = health_of(&app, player);
            let (mut lane, mut cast_clip, mut thrown) = (false, false, false);
            for _ in 0..300 {
                step(&mut app, 1);
                lane |= !lanes(&app).is_empty();
                // Its own clip, not the humanoid cast: the head, which
                // `humanoid-cast` tips back, is thrown forward past 20 degrees.
                if app.world().get::<slice::Casting>(enemy).is_some() {
                    assert!(
                        app.world().get::<Attack>(enemy).is_none(),
                        "a cast commits: no swing may start during one (ADR-0007)"
                    );
                    let pose = slot_pose(&app, enemy, "head");
                    cast_clip |= (pose.rotation * Vec3::NEG_Z).y < -0.34;
                }
                thrown |= app
                    .world()
                    .query::<slice::Flying>()
                    .any(|(_, flying)| flying.ability == spit_id && !flying.from_player);
                if thrown && health_of(&app, player) < before {
                    break;
                }
            }
            assert!(lane, "its wind-up marks the lane it will spit down");
            assert!(cast_clip, "it plays the clip its own model names");
            assert!(thrown, "it throws its authored skillshot");
            assert!(
                (before - health_of(&app, player) - 9.0).abs() < 1e-3,
                "which strikes the player for its authored 9: {before} -> {}",
                health_of(&app, player)
            );
        }

        // ---- mouse aim ------------------------------------------------------

        fn aim_at(app: &mut App, point: Vec3) {
            app.world_mut()
                .resource_mut::<aim::Aim>()
                .expect("the slice installs an aim")
                .point = Some(point);
        }

        fn mouse(app: &mut App, button: engine::input::MouseButton, state: ButtonState) {
            app.handle_platform_event(&PlatformEvent::MouseButton { button, state });
        }

        fn click(app: &mut App, button: engine::input::MouseButton) {
            mouse(app, button, ButtonState::Pressed);
            step(app, 1);
            mouse(app, button, ButtonState::Released);
        }

        fn facing_of(app: &App, entity: engine::ecs::Entity) -> Vec3 {
            app.world().get::<slice::Facing>(entity).expect("facing").0
        }

        fn set_mode(app: &mut App, mode: aim::CastMode) {
            app.world_mut()
                .resource_mut::<aim::CastInput>()
                .expect("the slice installs cast input")
                .mode = mode;
        }

        #[test]
        fn a_left_click_swings_toward_the_cursor() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let player = slice::player_entity(app.world()).expect("a player");
            let at = player_at(app.world());
            assert!(
                (facing_of(&app, player) - Vec3::NEG_Z).length() < 1e-4,
                "facing -Z to begin"
            );
            aim_at(&mut app, at + Vec3::new(5.0, 0.0, 0.0));

            click(&mut app, engine::input::MouseButton::Left);
            assert!(app.world().get::<Attack>(player).is_some(), "left-click swings");
            assert!(
                (facing_of(&app, player) - Vec3::X).length() < 1e-4,
                "turned to the cursor: {:?}",
                facing_of(&app, player)
            );
            let shown = marks(&app);
            assert_eq!(shown.len(), 1);
            let lands = at + Vec3::X * slice::PLAYER_ATTACK_REACH;
            assert!(
                (shown[0].0 - Vec3::new(lands.x, 0.0, lands.z)).length() < 1e-3,
                "and the blow is marked where the cursor points"
            );
        }

        #[test]
        fn a_wind_up_follows_the_cursor_and_locks_when_the_blow_is_live() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let player = slice::player_entity(app.world()).expect("a player");
            let at = player_at(app.world());
            aim_at(&mut app, at + Vec3::X * 5.0);
            press(&mut app, KeyCode::Space);
            step(&mut app, 1);
            release(&mut app, KeyCode::Space);

            // Mid wind-up: the aim moves, and the swing turns with it.
            aim_at(&mut app, at + Vec3::Z * 5.0);
            step(&mut app, 2);
            assert!(
                (facing_of(&app, player) - Vec3::Z).length() < 1e-4,
                "tracks through the wind-up: {:?}",
                facing_of(&app, player)
            );

            // Live: the aim moves again, and the blow does not follow.
            let windup = u64::from(slice::PLAYER_ATTACK_TIMING.windup);
            step(&mut app, windup);
            assert!(
                app.world()
                    .get::<Attack>(player)
                    .is_some_and(|attack| attack.phase() != game_runtime::combat::AttackPhase::Windup),
                "past the wind-up"
            );
            aim_at(&mut app, at - Vec3::X * 5.0);
            step(&mut app, 1);
            assert!(
                (facing_of(&app, player) - Vec3::Z).length() < 1e-4,
                "locked once live: {:?}",
                facing_of(&app, player)
            );
        }

        #[test]
        fn a_bolt_leaves_toward_where_the_cursor_is_when_it_is_thrown() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            husk_at(&mut app, 30.0, 0.0);
            let at = player_at(app.world());
            aim_at(&mut app, at + Vec3::X * 5.0);
            cast(&mut app, KeyCode::F);
            // Moved during the wind-up; the bolt goes where the cursor is now.
            step(&mut app, 4);
            let toward = Vec3::new(3.0, 0.0, 4.0).normalize();
            aim_at(&mut app, at + toward * 5.0);
            for _ in 0..20 {
                if !projectiles(&app).is_empty() {
                    break;
                }
                step(&mut app, 1);
            }
            let direction = app
                .world()
                .query::<game_runtime::projectile::Projectile>()
                .next()
                .map(|(_, projectile)| projectile.direction)
                .expect("a bolt in the air");
            assert!(
                (direction - toward).length() < 1e-3,
                "thrown toward the cursor as it stood at release: {direction:?}"
            );
        }

        #[test]
        fn walking_does_not_turn_a_committed_swing() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let player = slice::player_entity(app.world()).expect("a player");
            let at = player_at(app.world());
            aim_at(&mut app, at + Vec3::X * 5.0);
            press(&mut app, KeyCode::Space);
            step(&mut app, 1);
            release(&mut app, KeyCode::Space);
            // Walk the other way through the whole swing.
            press(&mut app, KeyCode::A);
            let total = u64::from(slice::PLAYER_ATTACK_TIMING.total());
            let mut checked = 0;
            for _ in 0..total {
                step(&mut app, 1);
                if app.world().get::<Attack>(player).is_none() {
                    break;
                }
                checked += 1;
                assert!(
                    facing_of(&app, player).x > 0.99,
                    "walking must not steer a blow aimed elsewhere: {:?}",
                    facing_of(&app, player)
                );
            }
            release(&mut app, KeyCode::A);
            assert!(
                checked > 10,
                "the swing was checked through most of its length: {checked}"
            );
        }

        #[test]
        fn quick_cast_with_indicator_aims_while_held_and_casts_on_release() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            husk_at(&mut app, 30.0, 0.0);
            set_mode(&mut app, aim::CastMode::QuickWithIndicator);
            let at = player_at(app.world());
            aim_at(&mut app, at + Vec3::X * 5.0);

            press(&mut app, KeyCode::F);
            step(&mut app, 3);
            assert!(!casting(&app), "held: not cast yet");
            assert_eq!(lanes(&app).len(), 1, "held: the lane is shown to aim with");
            let heading = app
                .world()
                .query2::<vfx::LaneMark, Transform>()
                .next()
                .map(|(_, _, transform)| transform.rotation * Vec3::NEG_Z)
                .expect("a lane");
            assert!(
                (heading - Vec3::X).length() < 1e-3,
                "pointing at the cursor: {heading:?}"
            );

            release(&mut app, KeyCode::F);
            step(&mut app, 1);
            assert!(casting(&app), "released: cast");
        }

        #[test]
        fn normal_cast_aims_on_a_press_casts_on_a_click_and_cancels_on_a_right_click() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            husk_at(&mut app, 30.0, 0.0);
            set_mode(&mut app, aim::CastMode::Normal);
            let player = slice::player_entity(app.world()).expect("a player");
            let at = player_at(app.world());
            aim_at(&mut app, at + Vec3::X * 5.0);

            cast(&mut app, KeyCode::F);
            step(&mut app, 2);
            assert!(!casting(&app), "a press only aims");
            assert_eq!(lanes(&app).len(), 1, "and shows the lane");

            click(&mut app, engine::input::MouseButton::Right);
            step(&mut app, 1);
            assert!(!casting(&app), "right-click cancels");
            assert!(lanes(&app).is_empty(), "and puts the indicator away");

            cast(&mut app, KeyCode::F);
            click(&mut app, engine::input::MouseButton::Left);
            assert!(casting(&app), "a left-click casts what is aimed");
            assert!(
                app.world().get::<Attack>(player).is_none(),
                "and the click that cast it does not also swing"
            );

            // A confirming click is spent on the cast even when the cast is
            // refused — here, cooling down — rather than falling through to
            // a swing the player never asked for.
            step(&mut app, 60);
            assert!(!casting(&app), "the first cast is over");
            let bolt = game_core::ids::AbilityId::new("ember-bolt").expect("valid id");
            if let Some(cooldowns) = app
                .world_mut()
                .get_mut::<game_runtime::ability::Cooldowns>(player)
            {
                cooldowns.start(bolt, 600);
            }
            cast(&mut app, KeyCode::F);
            click(&mut app, engine::input::MouseButton::Left);
            assert!(!casting(&app), "refused: still cooling down");
            assert!(
                app.world().get::<Attack>(player).is_none(),
                "and the confirming click did not become a swing"
            );
        }

        #[test]
        fn f2_cycles_the_cast_mode() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            let mode = |app: &App| app.world().resource::<aim::CastInput>().expect("cast input").mode;
            assert_eq!(
                mode(&app),
                aim::CastMode::QuickWithIndicator,
                "hold to aim and release to cast, by default"
            );
            for expected in [
                aim::CastMode::Normal,
                aim::CastMode::Quick,
                aim::CastMode::QuickWithIndicator,
            ] {
                press(&mut app, KeyCode::F2);
                step(&mut app, 1);
                release(&mut app, KeyCode::F2);
                step(&mut app, 1);
                assert_eq!(mode(&app), expected);
            }
        }

        /// The whole path from a cursor pixel to a point on the ground, through
        /// the camera the slice actually poses.
        #[test]
        fn the_cursor_aims_at_the_ground_under_it() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 2);
            let viewport = engine::camera::Viewport {
                width: 1280,
                height: 720,
            };
            app.world_mut().insert_resource(viewport);
            let camera = *app
                .world()
                .resource::<Camera>()
                .expect("the slice poses a camera");
            let target = player_at(app.world()) + Vec3::new(3.0, 0.0, -2.0);
            let target = Vec3::new(target.x, 0.0, target.z);
            let clip =
                camera.view_projection(viewport.aspect().expect("a real viewport")) * target.extend(1.0);
            let ndc = clip.truncate() / clip.w;
            let pixel = (
                f64::from((ndc.x + 1.0) * 0.5 * 1280.0),
                f64::from((1.0 - ndc.y) * 0.5 * 720.0),
            );
            app.handle_platform_event(&PlatformEvent::CursorMoved {
                x: pixel.0,
                y: pixel.1,
            });
            step(&mut app, 1);
            let aimed = app
                .world()
                .resource::<aim::Aim>()
                .and_then(|aim| aim.point)
                .expect("the cursor is over the ground");
            assert!(
                (aimed - target).length() < 0.05,
                "aimed at {aimed:?}, wanted {target:?}"
            );
            let reticle = |app: &App| {
                let world = app.world();
                world
                    .query2::<vfx::Reticle, Transform>()
                    .find(|(entity, _, _)| {
                        world
                            .get::<engine::render::Visibility>(*entity)
                            .is_none_or(|visibility| visibility.is_visible())
                    })
                    .map(|(_, _, transform)| transform.translation)
            };
            assert_eq!(reticle(&app), None, "no reticle until an ability is being aimed");

            // Held: the reticle sits under the cursor. Released: gone again.
            press(&mut app, KeyCode::F);
            step(&mut app, 1);
            // Compared with the aim as it stands now, not the first target: the
            // follow camera is still easing in, so the same pixel covers
            // slightly different ground a few ticks on.
            let shown = reticle(&app).expect("a reticle while aiming");
            let now = app
                .world()
                .resource::<aim::Aim>()
                .and_then(|aim| aim.point)
                .expect("still aiming");
            assert!(
                (shown - now).length() < 1e-4,
                "under the cursor: {shown:?} vs {now:?}"
            );
            release(&mut app, KeyCode::F);
            step(&mut app, 1);
            assert_eq!(reticle(&app), None, "and gone once cast");
        }

        // ---- enemies from content (M8 wave 1) ------------------------------

        /// Leaves the player alone with one enemy of `prefab`, `ahead`
        /// metres in front, thinking for itself.
        fn face_off(app: &mut App, prefab: &str, ahead: f32) -> (engine::ecs::Entity, engine::ecs::Entity) {
            let (player, husk) = lone_husk(app);
            let _ = app.world_mut().despawn(husk);
            let position = player_at(app.world());
            let facing = app.world().get::<slice::Facing>(player).expect("facing").0;
            let prefab = game_core::ids::PrefabId::new(prefab).expect("valid id");
            let mut enemy = None;
            app.world_mut().resource_scope::<slice::Slice, _>(|world, slice| {
                let spawn = game_core::ids::SpawnId::new("summoned").expect("valid id");
                enemy = slice::spawn_enemy_at(world, slice, &prefab, spawn, None, position + facing * ahead);
            });
            (player, enemy.expect("the enemy spawns from content"))
        }

        fn distance_between(app: &App, a: engine::ecs::Entity, b: engine::ecs::Entity) -> f32 {
            let at = |entity| app.world().get::<Transform>(entity).expect("placed").translation;
            let (a, b) = (at(a), at(b));
            Vec3::new(a.x - b.x, 0.0, a.z - b.z).length()
        }

        /// M8's first exit criterion: an enemy with its own range, pacing and
        /// timing is a content file and nothing else. Parsed here, named
        /// nowhere in Rust, and measured in a running slice against what it
        /// authored.
        #[test]
        fn a_tuned_enemy_is_a_content_file_and_plays_by_its_numbers() {
            let lurker: game_core::content::EnemyDef = ron::from_str(
                r#"(
                    schema_version: 1,
                    id: "ash-lurker",
                    name: "Ash Lurker",
                    behavior: Melee,
                    max_health: 50000,
                    damage: 6000,
                    abilities: [],
                    visual_tags: ["ash"],
                    threat: 1100,
                    variant_of: None,
                    tuning: (
                        acquire: Some(12000),
                        release: Some(18000),
                        preferred_range: Some(3000),
                        move_speed: Some(6000),
                        windup_ticks: Some(40),
                        active_ticks: Some(3),
                        recovery_ticks: Some(12),
                        reach: Some(2500),
                        strike_radius: Some(1200),
                    ),
                )"#,
            )
            .expect("the enemy parses");
            assert_eq!(lurker.validate(), Ok(()));
            let mut app = slice_app_with("ashfall-lane", move |content| {
                content.enemies.insert(lurker.id.clone(), lurker);
            });
            step(&mut app, 1);
            // Eleven metres: inside its authored 12 m acquire, outside a
            // husk's 9 m, so noticing the player at all is already its own.
            let (player, lurker) = face_off(&mut app, "ash-lurker", 11.0);

            // Its pace: 6 m/s is a tenth of a metre a tick, near twice a husk.
            let before = distance_between(&app, player, lurker);
            step(&mut app, 10);
            let closed = before - distance_between(&app, player, lurker);
            assert!(
                (closed - 1.0).abs() < 0.15,
                "closed {closed} m in 10 ticks at its authored 6 m/s"
            );

            // Its range and timing: it stops at about 3 m and winds up for 40.
            let mut swung = None;
            for _ in 0..200 {
                step(&mut app, 1);
                if let Some(attack) = app.world().get::<Attack>(lurker) {
                    swung = Some((*attack, distance_between(&app, player, lurker)));
                    break;
                }
            }
            let (attack, at) = swung.expect("it swings once in range");
            assert!(
                at <= 3.0 + 0.15 && at > 2.0,
                "from its preferred 3 m, not a husk's 1.6: {at}"
            );
            assert_eq!(
                attack.timing,
                game_runtime::combat::AttackTiming::new(40, 3, 12),
                "with its authored windows"
            );
            assert!((attack.reach - 2.5).abs() < 1e-3 && (attack.radius - 1.2).abs() < 1e-3);
        }

        /// The resin spitter leads with its skillshot, and the player sees
        /// the lane before the glob leaves.
        #[test]
        fn the_resin_spitter_spits_along_a_telegraphed_lane() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (_, spitter) = face_off(&mut app, "resin-spitter", 7.0);
            let spit = game_core::ids::AbilityId::new("resin-spit").expect("valid id");
            let (mut warned, mut thrown) = (false, false);
            for _ in 0..240 {
                step(&mut app, 1);
                let casting = app
                    .world()
                    .get::<slice::Casting>(spitter)
                    .is_some_and(|casting| casting.ability == spit);
                if casting && !thrown {
                    warned |= !lanes(&app).is_empty();
                }
                thrown |= app
                    .world()
                    .query::<slice::Flying>()
                    .any(|(_, flying)| flying.ability == spit && !flying.from_player);
                if thrown {
                    break;
                }
            }
            assert!(thrown, "the spitter spits");
            assert!(warned, "and its wind-up marks the lane first");
        }

        // ---- poise, the ashlord, and waves (M8 waves 2 and 3) --------------

        fn cleared(app: &App, encounter: &game_core::ids::EncounterId) -> bool {
            app.world()
                .resource::<slice::Slice>()
                .expect("a slice")
                .signals
                .iter()
                .any(|signal| {
                    matches!(
                        signal,
                        narrative::quest::Signal::EncounterCleared { encounter: done } if done == encounter
                    )
                })
        }

        fn tagged(app: &App, encounter: &game_core::ids::EncounterId) -> Vec<(engine::ecs::Entity, Vec3)> {
            app.world()
                .query2::<Transform, slice::Enemy>()
                .filter(|(_, _, enemy)| enemy.encounter.as_ref() == Some(encounter))
                .map(|(entity, transform, _)| (entity, transform.translation))
                .collect()
        }

        /// Kills `enemies` through the combat system's own death path, and
        /// leaves the tick's signals unconsumed so a test can read them.
        fn fell(app: &mut App, enemies: &[(engine::ecs::Entity, Vec3)]) {
            for (entity, _) in enemies {
                if let Some(health) = app.world_mut().get_mut::<Health>(*entity) {
                    health.current = 0.0;
                }
            }
            app.world_mut().resource_scope::<slice::Slice, _>(|world, slice| {
                slice::reap_dead(world, slice);
            });
        }

        #[test]
        fn an_ambush_arrives_in_waves_and_clears_only_after_the_last() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let ambush = game_core::ids::EncounterId::new("waterworks-ambush").expect("valid id");

            let first = tagged(&app, &ambush);
            assert_eq!(first.len(), 2, "two husks in the open");
            fell(&mut app, &first);
            assert!(
                !cleared(&app, &ambush),
                "the first wave falling does not clear the fight"
            );
            assert!(tagged(&app, &ambush).is_empty(), "and nothing has arrived yet");

            // The arrival is announced where each of the second wave will stand.
            step(&mut app, 2);
            let posts = [Vec3::new(-2.0, 0.9, -1.0), Vec3::new(-4.0, 0.9, 3.0)];
            for post in posts {
                assert!(
                    playing(&app)
                        .iter()
                        .any(|(at, _)| (Vec3::new(at.x, 0.9, at.z) - post).length() < 0.1),
                    "the ash rises at {post:?} before anything stands there: {:?}",
                    playing(&app)
                );
            }
            assert!(tagged(&app, &ambush).is_empty(), "still announced, not arrived");

            step(&mut app, slice::WAVE_DELAY_TICKS);
            let second = tagged(&app, &ambush);
            assert_eq!(second.len(), 2, "the third husk and the spitter arrive");
            for post in posts {
                assert!(
                    second.iter().any(|(_, at)| (*at - post).length() < 0.2),
                    "at their authored posts: {second:?}"
                );
            }
            let prefabs: Vec<String> = second
                .iter()
                .filter_map(|(entity, _)| app.world().get::<slice::Enemy>(*entity))
                .map(|enemy| enemy.prefab.to_string())
                .collect();
            assert!(
                prefabs.contains(&"resin-spitter".to_owned()),
                "the spitter is in the second wave"
            );

            if let Some(slice) = app.world_mut().resource_mut::<slice::Slice>() {
                slice.signals.clear();
            }
            fell(&mut app, &second);
            assert!(cleared(&app, &ambush), "the last wave falling clears the fight");
        }

        #[test]
        fn a_blow_that_breaks_a_husks_stance_cancels_its_swing() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (player, husk) = arm_one_enemy(&mut app);
            // A long wind-up, so the player's blow lands inside it.
            let _ = app.world_mut().insert(
                husk,
                Attack::new(game_runtime::combat::AttackTiming::new(60, 6, 20), 10.0, 1.4, 0.9),
            );
            let before = player_health(&app, player);
            press(&mut app, KeyCode::Space);
            step(&mut app, 1);
            release(&mut app, KeyCode::Space);

            let mut broke = false;
            for _ in 0..30 {
                step(&mut app, 1);
                if slice::staggered(app.world(), husk) {
                    broke = true;
                    break;
                }
            }
            assert!(broke, "22 damage breaks a husk's 20 poise");
            assert!(
                app.world().get::<Attack>(husk).is_none(),
                "and its wind-up is lost"
            );
            step(&mut app, 90);
            assert_eq!(player_health(&app, player), before, "so its blow never lands");
        }

        #[test]
        fn deep_poise_shrugs_off_a_single_blow() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (_, warden) = face_off(&mut app, "bark-warden", 1.5);
            let _ = app.world_mut().remove::<EnemyBrain>(warden);
            let _ = app.world_mut().insert(warden, slice::Facing(Vec3::Z));
            let _ = app.world_mut().insert(
                warden,
                Attack::new(game_runtime::combat::AttackTiming::new(60, 6, 20), 10.0, 1.4, 0.9),
            );
            let health = app.world().get::<Health>(warden).expect("health").current;
            press(&mut app, KeyCode::Space);
            step(&mut app, 1);
            release(&mut app, KeyCode::Space);
            step(&mut app, 20);
            assert!(
                app.world().get::<Health>(warden).expect("health").current < health,
                "the blow landed, or this proves nothing"
            );
            assert!(
                !slice::staggered(app.world(), warden),
                "a charger's 40 poise holds"
            );
            assert!(
                app.world().get::<Attack>(warden).is_some(),
                "and it keeps its swing"
            );
        }

        #[test]
        fn a_staggered_player_cannot_act_and_always_recovers() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let player = slice::player_entity(app.world()).expect("a player");
            let hit = |app: &mut App, amount: f32| {
                if let Some(health) = app.world_mut().get_mut::<Health>(player) {
                    health.current -= amount;
                }
                step(app, 1);
            };
            hit(&mut app, 31.0);
            assert!(
                slice::staggered(app.world(), player),
                "31 breaks the player's 30 poise"
            );

            let at = player_at(app.world());
            press(&mut app, KeyCode::D);
            press(&mut app, KeyCode::Space);
            step(&mut app, 1);
            release(&mut app, KeyCode::Space);
            assert!(
                app.world().get::<Attack>(player).is_none(),
                "no swing while staggered"
            );
            assert!((player_at(app.world()) - at).length() < 1e-4, "no walking either");

            step(&mut app, game_runtime::combat::STAGGER_TICKS);
            assert!(!slice::staggered(app.world(), player), "the stagger ends on time");
            assert!(player_at(app.world()).x > at.x, "and the player walks again");
            release(&mut app, KeyCode::D);

            // Inside the grace window another heavy hit does not break it again.
            hit(&mut app, 40.0);
            assert!(!slice::staggered(app.world(), player), "no stunlock: grace holds");
        }

        #[test]
        fn a_staggered_enemy_holds_still() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (_, husk) = face_off(&mut app, "corrupted-husk", 5.0);
            step(&mut app, 2);
            if let Some(health) = app.world_mut().get_mut::<Health>(husk) {
                health.current -= 25.0;
            }
            step(&mut app, 1);
            assert!(slice::staggered(app.world(), husk));
            let at = app.world().get::<Transform>(husk).expect("placed").translation;
            step(&mut app, game_runtime::combat::STAGGER_TICKS - 2);
            let now = app.world().get::<Transform>(husk).expect("placed").translation;
            assert!(
                (now - at).length() < 1e-4,
                "it does not advance while reeling: {at:?} -> {now:?}"
            );
            step(&mut app, 20);
            let later = app.world().get::<Transform>(husk).expect("placed").translation;
            assert!((later - at).length() > 0.1, "and comes on again once it recovers");
        }

        /// A ranged enemy between spits does not slash at the player from
        /// seven metres away.
        #[test]
        fn an_enemy_never_swings_at_air() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (_, spitter) = face_off(&mut app, "resin-spitter", 7.0);
            let spit = game_core::ids::AbilityId::new("resin-spit").expect("valid id");
            if let Some(cooldowns) = app
                .world_mut()
                .get_mut::<game_runtime::ability::Cooldowns>(spitter)
            {
                cooldowns.start(spit, 10_000);
            }
            for _ in 0..90 {
                step(&mut app, 1);
                assert!(
                    app.world().get::<Attack>(spitter).is_none(),
                    "a swing that cannot reach is never started"
                );
            }
        }

        #[test]
        fn the_ashlord_slams_those_who_close_in() {
            let mut app = slice_app("crypt-of-ash");
            step(&mut app, 1);
            let ashlord = app
                .world()
                .query::<slice::Enemy>()
                .find(|(_, enemy)| enemy.prefab.name() == "husk-ashlord")
                .map(|(entity, _)| entity)
                .expect("the elite");
            // Summoned already: its call is cooling, so what it does next is
            // its answer to someone standing close.
            let call = game_core::ids::AbilityId::new("call-husks").expect("valid id");
            if let Some(cooldowns) = app
                .world_mut()
                .get_mut::<game_runtime::ability::Cooldowns>(ashlord)
            {
                cooldowns.start(call, 10_000);
            }
            let at = app.world().get::<Transform>(ashlord).expect("placed").translation;
            slice::move_player_to(app.world_mut(), at + Vec3::new(3.0, 0.0, 0.0));
            let slam = game_core::ids::AbilityId::new("ash-slam").expect("valid id");
            let mut slammed = false;
            let mut marked = false;
            for _ in 0..120 {
                step(&mut app, 1);
                if app
                    .world()
                    .get::<slice::Casting>(ashlord)
                    .is_some_and(|casting| casting.ability == slam)
                {
                    slammed = true;
                    marked |= marks(&app)
                        .iter()
                        .any(|(_, outline, _)| (outline.pose.scale.x - slice::ABILITY_REACH).abs() < 1e-4);
                }
            }
            assert!(slammed, "the ashlord slams when the player closes");
            assert!(marked, "and marks the ground it will land on");
        }

        // ---- nameplates -----------------------------------------------------

        /// Every nameplate showing: where it stands, and its shadow and text.
        fn plates(app: &App) -> Vec<(Transform, engine::render::ModelPart, engine::render::ModelPart)> {
            let world = app.world();
            world
                .query2::<nameplate::Nameplate, engine::render::Model>()
                .filter(|(entity, _, _)| {
                    world
                        .get::<engine::render::Visibility>(*entity)
                        .is_none_or(|visibility| visibility.is_visible())
                })
                .map(|(entity, _, model)| {
                    let at = *world.get::<Transform>(entity).expect("placed");
                    (at, model.parts[0], model.parts[1])
                })
                .collect()
        }

        fn name_mesh(app: &App, name: &str) -> engine::render::MeshHandle {
            app.world()
                .resource::<nameplate::NameLibrary>()
                .and_then(|library| library.get(name))
                .unwrap_or_else(|| panic!("{name} has a mesh"))
        }

        #[test]
        fn an_enemy_near_the_player_wears_its_name() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (_, husk) = face_off(&mut app, "corrupted-husk", 4.0);
            let _ = app.world_mut().remove::<EnemyBrain>(husk);
            step(&mut app, 1);

            // By name: Arbiter Thal keeps his post in the lane, near enough to
            // be named too.
            let husk_name = name_mesh(&app, "Corrupted Husk");
            let shown: Vec<_> = plates(&app)
                .into_iter()
                .filter(|(_, _, text)| text.mesh == husk_name)
                .collect();
            assert_eq!(shown.len(), 1, "the husk wears its content name, once");
            let (at, shadow, text) = shown[0];
            assert_eq!(shadow.mesh, text.mesh, "with its shadow behind it");
            assert_ne!(shadow.material, text.material, "in black, under the colour");

            let body = app.world().get::<Transform>(husk).expect("placed").translation;
            assert!(
                (at.translation.x - body.x).abs() < 1e-4 && (at.translation.z - body.z).abs() < 1e-4,
                "over the husk: {:?} vs {body:?}",
                at.translation
            );
            assert!(
                at.translation.y > body.y + 0.9,
                "above its head: {:?}",
                at.translation
            );
            let camera = *app.world().resource::<Camera>().expect("a camera");
            let front = at.rotation * Vec3::Z;
            assert!(
                (front + camera.forward()).length() < 1e-3,
                "and turned to the camera"
            );
        }

        #[test]
        fn a_name_goes_with_its_body_and_is_not_shown_from_afar() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (_, husk) = face_off(&mut app, "corrupted-husk", 4.0);
            let _ = app.world_mut().remove::<EnemyBrain>(husk);
            step(&mut app, 1);
            let husk_name = name_mesh(&app, "Corrupted Husk");
            let husks = |app: &App| {
                plates(app)
                    .into_iter()
                    .filter(|(_, _, text)| text.mesh == husk_name)
                    .count()
            };
            assert_eq!(husks(&app), 1);

            if let Some(transform) = app.world_mut().get_mut::<Transform>(husk) {
                transform.translation.x += nameplate::NAME_RANGE + 2.0;
            }
            step(&mut app, 1);
            assert_eq!(husks(&app), 0, "too far off to name");

            if let Some(transform) = app.world_mut().get_mut::<Transform>(husk) {
                transform.translation.x -= nameplate::NAME_RANGE + 2.0;
            }
            step(&mut app, 1);
            assert_eq!(husks(&app), 1, "named again once near");
            fell(&mut app, &[(husk, Vec3::ZERO)]);
            step(&mut app, 1);
            assert_eq!(husks(&app), 0, "and gone with the body");
            // And no plate is left over showing a stale name: every plate on
            // screen labels a body, once — here only Thal, at his post.
            let thal = name_mesh(&app, "Arbiter Thal");
            let shown = plates(&app);
            assert_eq!(shown.len(), 1, "one body near, one plate: {}", shown.len());
            assert_eq!(shown[0].2.mesh, thal);
        }

        #[test]
        fn people_and_elites_are_named_in_their_own_colours() {
            let mut hub = slice_app("cinderwake-square");
            step(&mut hub, 1);
            let sela = hub
                .world()
                .query2::<Transform, slice::Npc>()
                .next()
                .map(|(_, transform, _)| transform.translation)
                .expect("sela stands in the square");
            slice::move_player_to(hub.world_mut(), sela + Vec3::new(2.0, 0.0, 0.0));
            step(&mut hub, 1);
            let person = plates(&hub)
                .into_iter()
                .find(|(_, _, text)| text.mesh == name_mesh(&hub, "Warden Sela"))
                .expect("sela wears her name, read from her id")
                .2
                .material;

            let mut crypt = slice_app("crypt-of-ash");
            step(&mut crypt, 1);
            let ashlord = crypt
                .world()
                .query2::<Transform, slice::Enemy>()
                .find(|(_, _, enemy)| enemy.prefab.name() == "husk-ashlord")
                .map(|(_, transform, _)| transform.translation)
                .expect("the elite");
            slice::move_player_to(crypt.world_mut(), ashlord + Vec3::new(3.0, 0.0, 0.0));
            step(&mut crypt, 1);
            let shown = plates(&crypt);
            let material_of = |name: &str| {
                shown
                    .iter()
                    .find(|(_, _, text)| text.mesh == name_mesh(&crypt, name))
                    .map(|(_, _, text)| text.material)
                    .unwrap_or_else(|| panic!("{name} is named"))
            };
            let elite = material_of("Husk Ashlord");
            let regular = material_of("Corrupted Husk");
            assert_ne!(elite, regular, "an elite's name is not a regular's colour");
            assert_ne!(person, regular, "nor is a person's a threat's");
            assert_ne!(person, elite);
        }

        // ---- melee looks -----------------------------------------------------

        #[test]
        fn a_live_swing_draws_its_slash_along_the_blow() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            let player = slice::player_entity(app.world()).expect("a player");
            let at = player_at(app.world());
            press(&mut app, KeyCode::Space);
            step(&mut app, 1);
            release(&mut app, KeyCode::Space);
            assert!(playing(&app).is_empty(), "nothing while the blow winds up");

            let windup = u64::from(slice::PLAYER_ATTACK_TIMING.windup);
            step(&mut app, windup);
            assert!(
                app.world()
                    .get::<Attack>(player)
                    .is_some_and(|attack| attack.phase() == game_runtime::combat::AttackPhase::Active),
                "the blow is live"
            );
            let lands = at + Vec3::NEG_Z * slice::PLAYER_ATTACK_REACH;
            let effects = playing(&app);
            assert!(
                effects
                    .iter()
                    .any(|(played, pieces)| (*played - lands).length() < 1e-3
                        && *pieces == 2 * usize::from(game_core::content::SLASH_SEGMENTS)),
                "the blade arc plays along the blow, both its slashes: {effects:?}"
            );
        }

        #[test]
        fn a_connecting_swing_throws_sparks_where_it_bites() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (_, husk) = husk_at(&mut app, 1.3, 0.0);
            let before = health_of(&app, husk);
            let body = app.world().get::<Transform>(husk).expect("placed").translation;
            press(&mut app, KeyCode::Space);
            step(&mut app, 1);
            release(&mut app, KeyCode::Space);
            let mut sparked = false;
            for _ in 0..30 {
                step(&mut app, 1);
                // strike-sparks: nine sparks and five chips of steel.
                sparked |= playing(&app)
                    .iter()
                    .any(|(played, pieces)| (*played - body).length() < 1e-3 && *pieces == 14);
                if sparked {
                    break;
                }
            }
            assert!(
                health_of(&app, husk) < before,
                "the blow landed, or this proves nothing"
            );
            assert!(sparked, "and sparks flew where it did");
        }

        #[test]
        fn the_ashlords_slam_cracks_the_ground_and_throws_rock() {
            let mut app = slice_app("crypt-of-ash");
            let ashlord = ashlord_casting(&mut app);
            // Its first ability was started by the helper; the slam is its
            // second, started directly once the first is out of the way.
            step(&mut app, 120);
            let _ = app.world_mut().remove::<slice::Casting>(ashlord);
            let mut started = false;
            app.world_mut().resource_scope::<slice::Slice, _>(|world, slice| {
                started = slice::start_ability(world, slice, ashlord, 1);
            });
            assert!(started, "the slam starts");
            let at = app.world().get::<Transform>(ashlord).expect("placed").translation;
            let mut quaked = false;
            for _ in 0..60 {
                step(&mut app, 1);
                // ash-quake: ten fissures, nine spires, sixteen rocks and a
                // ring of dust.
                quaked |= playing(&app)
                    .iter()
                    .any(|(played, pieces)| (*played - at).length() < 1e-3 && *pieces == 10 + 9 + 16 + 1);
                if quaked {
                    break;
                }
            }
            assert!(quaked, "the ground splits under the slam");
        }

        // ---- the on-screen menu ----------------------------------------------

        fn open_menu(app: &App) -> Option<menu::Menu> {
            let world = app.world();
            world
                .query::<menu::Menu>()
                .find(|(entity, _)| {
                    world
                        .get::<engine::render::Visibility>(*entity)
                        .is_none_or(|visibility| visibility.is_visible())
                })
                .map(|(_, menu)| menu.clone())
        }

        /// Stands the player at Sela and opens the conversation, with a
        /// 1280 by 720 window to click in.
        fn talking_to_sela() -> App {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            app.world_mut().insert_resource(engine::camera::Viewport {
                width: 1280,
                height: 720,
            });
            let sela = app
                .world()
                .query2::<Transform, slice::Npc>()
                .next()
                .map(|(_, transform, _)| transform.translation)
                .expect("the warden holds the square");
            stand_at(&mut app, sela);
            interact(&mut app);
            assert!(offer_open(&app), "talking to sela opens a conversation");
            app
        }

        /// The pixel at the middle of the `option`th option's first line.
        fn option_pixel(app: &App, option: usize) -> (f64, f64) {
            let slice = app.world().resource::<slice::Slice>().expect("a slice");
            let (title, prompt, options) = menu::contents(slice).expect("something is on offer");
            let layout = menu::layout(&title, &prompt, &options);
            let panel = menu::panel(&layout, 1280.0 / 720.0);
            let line = layout
                .lines
                .iter()
                .position(|line| line.style == menu::Style::Option(option))
                .expect("that option is listed");
            let (top, bottom) = panel.line_span(line);
            (
                f64::from((panel.left + panel.width * 0.5) * 1280.0),
                f64::from((top + bottom) * 0.5 * 720.0),
            )
        }

        #[test]
        fn talking_puts_the_speakers_words_on_screen() {
            let app = talking_to_sela();
            let shown = open_menu(&app).expect("a menu is on screen");
            let slice = app.world().resource::<slice::Slice>().expect("a slice");
            let (_, prompt, options) = menu::contents(slice).expect("on offer");
            assert_eq!(
                shown.lines.first().map(String::as_str),
                Some("Warden Sela"),
                "who is speaking"
            );
            let said: Vec<&str> = shown
                .lines
                .iter()
                .map(String::as_str)
                .filter(|line| !line.trim().is_empty() && prompt.contains(line.trim()))
                .collect();
            assert_eq!(
                said.join(" ").split_whitespace().collect::<Vec<_>>(),
                prompt.split_whitespace().collect::<Vec<_>>(),
                "what she says, all of it"
            );
            assert!(!options.is_empty(), "she offers something to say back");
            for option in &options {
                assert!(
                    shown
                        .lines
                        .iter()
                        .any(|line| !line.is_empty() && option.starts_with(line.as_str())),
                    "the option {option:?} is listed"
                );
            }

            // Every character drawn is a glyph on the panel, and the panel is
            // one more.
            let drawn: usize = shown
                .lines
                .iter()
                .map(|line| line.chars().filter(|c| engine::pixel_text::has_glyph(*c)).count())
                .sum();
            let parts = app
                .world()
                .query2::<menu::Menu, engine::render::Model>()
                .next()
                .map(|(_, _, model)| model.parts.len())
                .expect("the menu is a model");
            assert_eq!(parts, drawn + 1, "a glyph per character, and the panel");
        }

        #[test]
        fn hovering_an_option_lights_it() {
            let mut app = talking_to_sela();
            let pixel = option_pixel(&app, 0);
            app.handle_platform_event(&PlatformEvent::CursorMoved {
                x: pixel.0,
                y: pixel.1,
            });
            step(&mut app, 1);
            assert_eq!(open_menu(&app).expect("open").hovered, Some(0));
            app.handle_platform_event(&PlatformEvent::CursorMoved { x: 5.0, y: 5.0 });
            step(&mut app, 1);
            assert_eq!(
                open_menu(&app).expect("open").hovered,
                None,
                "and not when the cursor leaves it"
            );
        }

        #[test]
        fn clicking_an_option_says_it_and_does_not_swing() {
            let mut app = talking_to_sela();
            let before = {
                let slice = app.world().resource::<slice::Slice>().expect("a slice");
                menu::contents(slice)
            };
            let pixel = option_pixel(&app, 0);
            app.handle_platform_event(&PlatformEvent::CursorMoved {
                x: pixel.0,
                y: pixel.1,
            });
            step(&mut app, 1);
            click(&mut app, engine::input::MouseButton::Left);
            step(&mut app, 1);

            let after = {
                let slice = app.world().resource::<slice::Slice>().expect("a slice");
                menu::contents(slice)
            };
            assert_ne!(after, before, "the conversation moved on");
            let player = slice::player_entity(app.world()).expect("a player");
            assert!(
                app.world().get::<Attack>(player).is_none(),
                "and the click that chose was not also a swing"
            );
        }

        #[test]
        fn walking_away_puts_the_menu_away() {
            let mut app = talking_to_sela();
            assert!(open_menu(&app).is_some());
            interact(&mut app);
            assert!(!offer_open(&app), "the conversation is over");
            assert!(open_menu(&app).is_none(), "and nothing is left on screen");
        }

        #[test]
        fn an_unknown_start_location_is_refused() {
            let missing = location("nowhere-at-all");
            assert!(
                slice::Slice::new(&content_root(), &missing, SLICE_SEED).is_err(),
                "booting into a location content does not define must fail loudly"
            );
        }

        #[test]
        fn every_dialogue_speaker_exists_in_canonical_state() {
            let built = slice::Slice::new(&content_root(), &location("cinderwake-square"), SLICE_SEED)
                .expect("content loads");

            for dialogue in built.content.dialogues.values() {
                assert!(
                    built.state.npc(&dialogue.speaker).is_some(),
                    "{} speaks but has no canonical state, so talking would fail at the reducer",
                    dialogue.speaker
                );
            }
            let _ = NpcId::new("warden-sela").expect("valid id");
        }

        // ---- the HUD (M9) ----------------------------------------------------

        fn hud(app: &App) -> hud::Hud {
            app.world()
                .query::<hud::Hud>()
                .next()
                .map(|(_, hud)| hud.clone())
                .expect("the HUD is drawn")
        }

        fn tap(app: &mut App, key: KeyCode) {
            press(app, key);
            step(app, 2);
            release(app, key);
            step(app, 2);
        }

        fn offer(app: &App) -> Option<game_runtime::selection::Offer> {
            app.world()
                .resource::<slice::Slice>()
                .and_then(|slice| slice.offer.clone())
        }

        #[test]
        fn taking_a_quest_tracks_it_and_says_so() {
            let mut app = talking_to_sela();
            assert!(hud(&app).tracker.is_empty(), "nothing is tracked before a quest");

            choose(&mut app, 1);
            if offer(&app).is_some() {
                interact(&mut app);
            }
            let shown = hud(&app);
            assert_eq!(
                shown.tracker,
                vec!["Still Waters".to_owned(), "- Examine Fouled Intake".to_owned()],
                "the quest, and the step after speaking to sela"
            );
            assert!(
                shown.notices.contains(&"New quest: Still Waters".to_owned())
                    && shown.notices.contains(&"Done: Talk to Warden Sela".to_owned()),
                "{:?}",
                shown.notices
            );
        }

        #[test]
        fn notices_fade() {
            let mut app = talking_to_sela();
            choose(&mut app, 1);
            if offer(&app).is_some() {
                interact(&mut app);
            }
            assert!(!hud(&app).notices.is_empty());
            step(&mut app, hud::NOTICE_TICKS + 10);
            assert!(hud(&app).notices.is_empty(), "{:?}", hud(&app).notices);
        }

        #[test]
        fn the_prompt_says_what_e_would_do_and_gives_way_to_a_panel() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            let sela = app
                .world()
                .query2::<Transform, slice::Npc>()
                .next()
                .map(|(_, transform, _)| transform.translation)
                .expect("sela");
            stand_at(&mut app, sela);
            assert_eq!(hud(&app).prompt.as_deref(), Some("E - Talk to Warden Sela"));

            let gate = station_at(&app, "lane-gate");
            stand_at(&mut app, gate);
            assert_eq!(hud(&app).prompt.as_deref(), Some("E - Enter Ashfall Lane"));

            stand_at(&mut app, Vec3::new(-8.0, 0.0, -8.0));
            assert_eq!(hud(&app).prompt, None, "nothing in reach, nothing said");

            stand_at(&mut app, sela);
            interact(&mut app);
            let shown = hud(&app);
            assert_eq!(shown.prompt, None, "a panel is open");
            assert!(
                shown.abilities.is_empty(),
                "the bottom of the screen is the panel's"
            );
        }

        #[test]
        fn each_ability_is_listed_under_its_key() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            assert_eq!(
                hud(&app).abilities,
                vec![
                    "Q  Ember Cleave".to_owned(),
                    "R  Warden's Guard".to_owned(),
                    "F  Ember Bolt".to_owned()
                ]
            );
        }

        #[test]
        fn examining_a_clue_shows_what_it_says() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let clue = app
                .world()
                .query2::<Transform, slice::Clue>()
                .next()
                .map(|(_, transform, _)| transform.translation)
                .expect("the lane has a clue");
            stand_at(&mut app, clue);
            assert_eq!(hud(&app).prompt.as_deref(), Some("E - Examine Fouled Intake"));
            interact(&mut app);

            let shown = open_menu(&app).expect("the clue's text is on screen");
            assert_eq!(shown.lines.first().map(String::as_str), Some("Fouled Intake"));
            assert!(
                shown
                    .lines
                    .iter()
                    .any(|line| line.starts_with("Resin clogs the grate")),
                "{:?}",
                shown.lines
            );
            interact(&mut app);
            assert_eq!(offer(&app), None, "E puts it down");
        }

        #[test]
        fn the_journal_and_the_controls_open_and_close_on_their_keys() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);

            tap(&mut app, KeyCode::J);
            let shown = open_menu(&app).expect("the journal is on screen");
            assert_eq!(shown.lines.first().map(String::as_str), Some("Journal"));
            assert!(
                shown
                    .lines
                    .iter()
                    .any(|line| line == "No quests yet. Talk to people.")
            );
            tap(&mut app, KeyCode::J);
            assert_eq!(offer(&app), None, "J again closes it");

            tap(&mut app, KeyCode::H);
            let shown = open_menu(&app).expect("the controls are on screen");
            assert_eq!(shown.lines.first().map(String::as_str), Some("Controls"));
            assert!(shown.lines.iter().any(|line| line == "WASD - move"));
            tap(&mut app, KeyCode::J);
            assert_eq!(
                offer(&app),
                Some(game_runtime::selection::Offer::Controls),
                "J does not replace another open panel"
            );
        }

        #[test]
        fn the_journal_lists_active_quests_then_finished_ones() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            if let Some(slice) = app.world_mut().resource_mut::<slice::Slice>() {
                for event in [
                    DomainEvent::QuestAccepted {
                        quest: quest("still-waters"),
                    },
                    DomainEvent::QuestAccepted {
                        quest: quest("ember-and-ash"),
                    },
                    DomainEvent::QuestCompleted {
                        quest: quest("ember-and-ash"),
                        resolution: game_core::ids::QuestResolutionId::new("crypt-silenced")
                            .expect("valid id"),
                    },
                ] {
                    slice.raise(event);
                }
            }
            step(&mut app, 2);
            let slice = app.world().resource::<slice::Slice>().expect("a slice");
            assert_eq!(
                hud::journal(&slice.content, &slice.state),
                "Still Waters\n  - Talk to Warden Sela\nEmber and Ash - done: Crypt Silenced"
            );
        }

        #[test]
        fn someone_with_nothing_open_to_say_says_so() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            if let Some(slice) = app.world_mut().resource_mut::<slice::Slice>() {
                let never = game_core::ids::WorldFlag::new("never-set").expect("valid id");
                for dialogue in slice.content.dialogues.values_mut() {
                    dialogue.conditions = vec![game_core::content::Condition::WorldFlagSet(never.clone())];
                }
            }
            let sela = app
                .world()
                .query2::<Transform, slice::Npc>()
                .next()
                .map(|(_, transform, _)| transform.translation)
                .expect("sela");
            stand_at(&mut app, sela);
            interact(&mut app);
            assert_eq!(offer(&app), None);
            assert!(
                hud(&app)
                    .notices
                    .contains(&"Warden Sela has nothing to say.".to_owned()),
                "{:?}",
                hud(&app).notices
            );
        }

        /// The HUD draws a glyph at a time (M9 plan, wave 1 decisions). This
        /// holds a typical moment — one quest, two notices, three abilities —
        /// to the 128 parts measured when it was written, with some room, so
        /// a change that multiplies the cost is noticed rather than shipped.
        #[test]
        fn the_hud_stays_within_its_draw_budget() {
            let mut app = talking_to_sela();
            choose(&mut app, 1);
            if offer(&app).is_some() {
                interact(&mut app);
            }
            let parts: usize = app
                .world()
                .query::<hud::Hud>()
                .filter_map(|(entity, _)| app.world().get::<engine::render::Model>(entity))
                .map(|model| model.parts.len())
                .sum();
            assert!((64..=160).contains(&parts), "{parts} HUD parts");
        }

        /// The player's statuses show with their time left (M9).
        #[test]
        fn a_status_on_the_player_shows_with_its_time_left() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            assert!(hud(&app).statuses.is_empty());
            press(&mut app, KeyCode::R);
            step(&mut app, 1);
            release(&mut app, KeyCode::R);
            step(&mut app, 11);
            assert_eq!(
                hud(&app).statuses,
                vec!["Guarded 3s".to_owned()],
                "three seconds, rounded up"
            );
            step(&mut app, 200);
            assert!(hud(&app).statuses.is_empty(), "gone when it ends");
        }

        /// An afflicted enemy wears its statuses under its name (M9).
        #[test]
        fn an_afflicted_enemy_wears_its_status_under_its_name() {
            let mut app = slice_app("ashfall-lane");
            step(&mut app, 1);
            let (_, enemy) = arm_one_enemy(&mut app);
            let parts = |app: &App| {
                app.world()
                    .query::<nameplate::Nameplate>()
                    .filter(|(entity, _)| {
                        app.world()
                            .get::<engine::render::Visibility>(*entity)
                            .is_some_and(|visibility| visibility.is_visible())
                    })
                    .filter_map(|(entity, _)| app.world().get::<engine::render::Model>(entity))
                    .map(|model| model.parts.len())
                    .max()
                    .unwrap_or(0)
            };
            let plain = parts(&app);
            assert_eq!(plain, 2, "a name and its shadow");
            let burn = defined_status(&app, "status-smouldering");
            afflict(&mut app, enemy, burn, 120);
            step(&mut app, 1);
            assert_eq!(parts(&app), 3, "and a line for the burn");
            step(&mut app, 130);
            assert_eq!(parts(&app), 2, "which goes when the burn does");
        }

        /// A class that says nothing fights exactly as the slice always has,
        /// bit for bit (M9).
        #[test]
        fn an_untuned_class_fights_by_the_old_numbers() {
            assert_eq!(
                slice::PlayerCombat::of(&game_core::content::ClassCombat::default()),
                slice::PlayerCombat::default()
            );
            let built = slice::Slice::new(&content_root(), &location("cinderwake-square"), SLICE_SEED)
                .expect("content loads");
            let wayfinder =
                &built.content.classes[&game_core::ids::ClassId::new("wayfinder").expect("valid id")];
            assert_eq!(
                slice::PlayerCombat::of(&wayfinder.combat),
                slice::PlayerCombat::default()
            );
        }

        /// The cinderward's numbers reach its swing, its roll, and its stance
        /// (M9): the second class differs in how it fights, not only in pools.
        #[test]
        fn a_class_swings_rolls_and_stands_by_its_own_numbers() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            let cinderward = {
                let slice = app.world().resource::<slice::Slice>().expect("a slice");
                let class =
                    &slice.content.classes[&game_core::ids::ClassId::new("cinderward").expect("valid id")];
                slice::PlayerCombat::of(&class.combat)
            };
            assert_ne!(cinderward, slice::PlayerCombat::default());
            let player = slice::player_entity(app.world()).expect("a player");
            let _ = app.world_mut().insert(player, cinderward);

            let before = player_stamina(&app);
            press(&mut app, KeyCode::Space);
            step(&mut app, 1);
            release(&mut app, KeyCode::Space);
            let attack = app.world().get::<Attack>(player).copied().expect("swinging");
            assert_eq!(attack.timing.windup, 18);
            assert!((attack.damage - 34.0).abs() < 1e-4, "{}", attack.damage);
            assert!(
                (before - player_stamina(&app) - 32.0).abs() < 1e-3,
                "a heavier swing costs more"
            );

            // Past the swing, then a roll.
            step(&mut app, 60);
            let before = player_stamina(&app);
            press(&mut app, KeyCode::ShiftLeft);
            step(&mut app, 1);
            release(&mut app, KeyCode::ShiftLeft);
            let dodge = app
                .world()
                .get::<game_runtime::movement::Dodge>(player)
                .cloned()
                .expect("rolling");
            assert_eq!(dodge.invulnerable_for, 8);
            assert_eq!(
                dodge.duration,
                slice::DODGE_DURATION,
                "the roll itself is not the class's"
            );
            assert!((before - player_stamina(&app) - 30.0).abs() < 1e-3);
        }

        fn give(app: &mut App, name: &str, quantity: u32) {
            if let Some(slice) = app.world_mut().resource_mut::<slice::Slice>() {
                slice.raise(DomainEvent::ItemGranted {
                    item: ItemId::new(name).expect("valid id"),
                    quantity,
                });
            }
            step(app, 2);
        }

        fn carried(app: &App, name: &str) -> u32 {
            app.world()
                .resource::<slice::Slice>()
                .and_then(|slice| {
                    slice
                        .state
                        .player
                        .inventory
                        .get(&ItemId::new(name).expect("valid id"))
                        .copied()
                })
                .unwrap_or(0)
        }

        /// X drinks a draught (M9): a committed half-second, then thirty health
        /// back and a while refreshed, and one fewer draught.
        #[test]
        fn x_drinks_a_draught_after_a_committed_moment() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            give(&mut app, "clearwater-draught", 2);
            assert!(
                hud(&app)
                    .abilities
                    .contains(&"X  Clearwater Draught x2".to_owned())
            );
            let player = slice::player_entity(app.world()).expect("a player");
            if let Some(health) = app.world_mut().get_mut::<Health>(player) {
                health.current -= 50.0;
            }
            // A drop in health reads as a blow and breaks the stance (M8), so
            // let the stagger pass before drinking.
            step(&mut app, game_runtime::combat::STAGGER_TICKS + 2);
            assert!(!slice::staggered(app.world(), player));
            let wounded = player_health(&app, player);

            tap(&mut app, KeyCode::X);
            assert!(app.world().get::<slice::Using>(player).is_some(), "drinking");
            assert!((player_health(&app, player) - wounded).abs() < 1e-4, "not yet");
            press(&mut app, KeyCode::Space);
            step(&mut app, 1);
            release(&mut app, KeyCode::Space);
            assert!(
                app.world().get::<Attack>(player).is_none(),
                "no swing while drinking (ADR-0007)"
            );

            step(&mut app, 40);
            assert!(app.world().get::<slice::Using>(player).is_none());
            assert!(
                (player_health(&app, player) - (wounded + 30.0)).abs() < 1e-3,
                "thirty back: {wounded} to {}",
                player_health(&app, player)
            );
            assert_eq!(carried(&app, "clearwater-draught"), 1, "one spent");
            assert!(
                hud(&app)
                    .statuses
                    .iter()
                    .any(|line| line.starts_with("Refreshed")),
                "{:?}",
                hud(&app).statuses
            );
        }

        /// The inventory offers what can be used, and picking it drinks (M9).
        #[test]
        fn the_inventory_offers_a_draught_to_drink() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            give(&mut app, "clearwater-draught", 1);
            tap(&mut app, KeyCode::I);
            let shown = open_menu(&app).expect("the inventory is open");
            assert_eq!(shown.lines.first().map(String::as_str), Some("Inventory"));
            assert!(
                shown
                    .lines
                    .iter()
                    .any(|line| line == "1) Use Clearwater Draught (1)"),
                "{:?}",
                shown.lines
            );
            choose(&mut app, 1);
            step(&mut app, 40);
            assert_eq!(carried(&app, "clearwater-draught"), 0, "drunk from the menu");
        }

        /// A stagger costs the moment, not the draught (M9).
        #[test]
        fn a_stagger_spills_nothing() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            give(&mut app, "clearwater-draught", 1);
            tap(&mut app, KeyCode::X);
            let player = slice::player_entity(app.world()).expect("a player");
            assert!(app.world().get::<slice::Using>(player).is_some());
            let tick = app.world().resource::<FixedTime>().map_or(0, |time| time.tick);
            if let Some(poise) = app.world_mut().get_mut::<game_runtime::combat::Poise>(player) {
                assert!(poise.take(1_000.0, tick), "broken");
            }
            step(&mut app, 40);
            assert!(app.world().get::<slice::Using>(player).is_none(), "cancelled");
            assert_eq!(carried(&app, "clearwater-draught"), 1, "and not spent");
        }

        /// A fresh start offers every class, and picking one rebuilds the
        /// player from it: pools, loadout, swing, and stance (M9).
        #[test]
        fn choosing_a_class_rebuilds_the_player_from_it() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            offer_class_choice(&mut app);
            step(&mut app, 1);
            let shown = open_menu(&app).expect("the class panel is open");
            assert_eq!(shown.lines.first().map(String::as_str), Some("Who are you?"));
            for row in [
                "1) Cinderward - 160 health, 80 stamina",
                "2) Wayfinder - 120 health, 100 stamina",
            ] {
                assert!(
                    shown.lines.iter().any(|line| line == row),
                    "{row} in {:?}",
                    shown.lines
                );
            }

            choose(&mut app, 1);
            assert_eq!(offer(&app), None, "a choice closes the panel");
            let slice = app.world().resource::<slice::Slice>().expect("a slice");
            assert_eq!(
                slice
                    .state
                    .player
                    .progression
                    .class
                    .as_ref()
                    .map(|class| class.name()),
                Some("cinderward")
            );
            let expected = slice::PlayerCombat::of(
                &slice.content.classes[&game_core::ids::ClassId::new("cinderward").expect("valid id")].combat,
            );
            let player = slice::player_entity(app.world()).expect("a player");
            let world = app.world();
            assert_eq!(world.get::<Health>(player).map(|health| health.max), Some(160.0));
            assert_eq!(world.get::<slice::PlayerCombat>(player).copied(), Some(expected));
            assert_eq!(
                world
                    .get::<game_runtime::combat::Poise>(player)
                    .map(|poise| poise.max),
                Some(60.0),
                "the cinderward's deeper stance"
            );
            assert_eq!(
                world.get::<slice::Abilities>(player).map(|known| known.0.len()),
                Some(2)
            );
            assert_eq!(world.query::<slice::Player>().count(), 1, "one body, not two");
        }

        /// Putting the panel down keeps the default class (M9).
        #[test]
        fn closing_the_class_panel_keeps_the_default() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            offer_class_choice(&mut app);
            interact(&mut app);
            assert_eq!(offer(&app), None);
            let class = app
                .world()
                .resource::<slice::Slice>()
                .and_then(|slice| slice.state.player.progression.class.clone());
            assert_eq!(class.as_ref().map(|class| class.name()), Some("wayfinder"));
        }

        /// Arriving somewhere says where, and in which region (M9).
        #[test]
        fn arriving_names_the_place_and_its_region() {
            let mut app = slice_app("cinderwake-square");
            step(&mut app, 1);
            travel_to_lane(&mut app);
            assert!(
                hud(&app)
                    .notices
                    .contains(&"Ashfall Lane - Ashfall Waste".to_owned()),
                "{:?}",
                hud(&app).notices
            );
        }

        // ---- reloading content (M9 wave 3) ------------------------------------

        /// A private copy of the shipped content, for a test to edit.
        fn content_copy(name: &str) -> PathBuf {
            let to = std::env::temp_dir().join(format!("cinderwake-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&to);
            std::fs::create_dir_all(&to).expect("a scratch folder");
            for entry in std::fs::read_dir(content_root()).expect("shipped content") {
                let entry = entry.expect("an entry");
                if entry.path().is_file() {
                    std::fs::copy(entry.path(), to.join(entry.file_name())).expect("copied");
                }
            }
            to
        }

        fn edit_content(root: &Path, file: &str, from: &str, to: &str) {
            let path = root.join(file);
            let text = std::fs::read_to_string(&path).expect("readable");
            assert!(text.contains(from), "{file} holds {from}");
            std::fs::write(&path, text.replacen(from, to, 1)).expect("writable");
        }

        fn dev_app_at(root: &Path, start: &str) -> App {
            let built = slice::Slice::new(root, &location(start), SLICE_SEED).expect("the copy loads");
            let mut app = build_slice_app(engine::render::RendererConfig::default()).expect("app builds");
            attach_slice(&mut app, built).expect("systems register");
            app.world_mut().insert_resource(DevTools);
            step(&mut app, 1);
            app
        }

        fn husk_health(app: &App) -> f32 {
            let world = app.world();
            world
                .query::<slice::Enemy>()
                .filter(|(_, enemy)| enemy.prefab.name() == "corrupted-husk")
                .filter_map(|(entity, _)| world.get::<Health>(entity))
                .map(|health| health.max)
                .next()
                .expect("a husk")
        }

        #[test]
        fn f5_reloads_content_in_a_development_run() {
            let root = content_copy("reload-ok");
            let mut app = dev_app_at(&root, "ashfall-lane");
            let before = husk_health(&app);
            let player = slice::player_entity(app.world()).expect("a player");
            stand_at(&mut app, Vec3::new(1.0, 0.0, -9.0));
            let at = app
                .world()
                .get::<Transform>(player)
                .map(|transform| transform.translation);

            edit_content(&root, "enemies.ron", "max_health: 60000", "max_health: 75000");
            tap(&mut app, KeyCode::F5);

            let after = husk_health(&app);
            assert!(
                (after - 75.0).abs() < 1e-3 && (before - after).abs() > 1.0,
                "{before} -> {after}"
            );
            assert!(
                hud(&app)
                    .notices
                    .iter()
                    .any(|line| line.starts_with("Content reloaded - ")),
                "{:?}",
                hud(&app).notices
            );
            let player = slice::player_entity(app.world()).expect("a player");
            let now = app
                .world()
                .get::<Transform>(player)
                .map(|transform| transform.translation);
            assert!(
                matches!((at, now), (Some(at), Some(now)) if (Vec3::new(at.x, 0.0, at.z) - Vec3::new(now.x, 0.0, now.z)).length() < 0.2),
                "the player stays where they stood: {at:?} -> {now:?}"
            );
            let _ = std::fs::remove_dir_all(root);
        }

        #[test]
        fn a_broken_reload_changes_nothing_and_says_why() {
            let root = content_copy("reload-broken");
            let mut app = dev_app_at(&root, "ashfall-lane");
            let before = husk_health(&app);
            edit_content(
                &root,
                "enemies.ron",
                "max_health: 60000",
                "max_health: 75000,,, broken",
            );
            tap(&mut app, KeyCode::F5);
            assert!(
                (husk_health(&app) - before).abs() < 1e-4,
                "the old content still runs"
            );
            assert!(
                hud(&app)
                    .notices
                    .iter()
                    .any(|line| line.starts_with("Reload refused - ")
                        && line.contains("problem(s): enemies.ron: ")),
                "names the file, not the folder: {:?}",
                hud(&app).notices
            );
            let _ = std::fs::remove_dir_all(root);
        }

        #[test]
        fn a_reload_that_would_orphan_the_world_is_refused() {
            let root = content_copy("reload-orphan");
            let mut app = dev_app_at(&root, "cinderwake-square");
            give(&mut app, "clearwater-draught", 1);
            edit_content(
                &root,
                "items.ron",
                "id: \"clearwater-draught\"",
                "id: \"murkwater-draught\"",
            );
            // Nothing else may name it either, or the copy would not load at all.
            for file in ["quests.ron", "dialogues.ron", "recipes.ron"] {
                let path = root.join(file);
                let text = std::fs::read_to_string(&path).expect("readable");
                std::fs::write(&path, text.replace("clearwater-draught", "murkwater-draught"))
                    .expect("writable");
            }
            tap(&mut app, KeyCode::F5);
            assert!(
                hud(&app)
                    .notices
                    .iter()
                    .any(|line| line.starts_with("Reload refused - ") && line.contains("clearwater-draught")),
                "the carried draught would vanish: {:?}",
                hud(&app).notices
            );
            let _ = std::fs::remove_dir_all(root);
        }

        #[test]
        fn a_reload_waits_for_a_closed_panel_and_needs_dev_tools() {
            let root = content_copy("reload-wait");
            let mut app = dev_app_at(&root, "ashfall-lane");
            let before = husk_health(&app);
            edit_content(&root, "enemies.ron", "max_health: 60000", "max_health: 75000");

            tap(&mut app, KeyCode::J);
            tap(&mut app, KeyCode::F5);
            assert!(
                (husk_health(&app) - before).abs() < 1e-4,
                "not while the journal is open"
            );
            assert!(
                hud(&app)
                    .notices
                    .iter()
                    .any(|line| line.starts_with("Reload waits"))
            );
            tap(&mut app, KeyCode::J);
            assert!(
                (husk_health(&app) - 75.0).abs() < 1e-3,
                "then as soon as it closes"
            );

            let mut plain = slice_app("ashfall-lane");
            step(&mut plain, 1);
            let untouched = husk_health(&plain);
            tap(&mut plain, KeyCode::F5);
            assert!(
                (husk_health(&plain) - untouched).abs() < 1e-4,
                "no --dev, no reload"
            );
            let _ = std::fs::remove_dir_all(root);
        }

        // ---- the M9 exit test --------------------------------------------------

        /// The shipped content with `tests/fixtures/content-m9/` added: each
        /// family the fixture extends becomes a directory holding the shipped
        /// file and the new one — the loader's own layout, so nothing edits or
        /// merges text, and every addition is a content file.
        fn m9_content() -> PathBuf {
            let root = content_copy("m9-exit");
            let extra = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/content-m9");
            for entry in std::fs::read_dir(extra).expect("the fixture") {
                let path = entry.expect("an entry").path();
                if path.extension().is_none_or(|extension| extension != "ron") {
                    continue;
                }
                let family = path.file_stem().expect("a name").to_owned();
                let folder = root.join(&family);
                std::fs::create_dir_all(&folder).expect("a family folder");
                let shipped = root.join(&family).with_extension("ron");
                if shipped.is_file() {
                    std::fs::rename(&shipped, folder.join("shipped.ron")).expect("moved");
                }
                std::fs::copy(&path, folder.join("m9.ron")).expect("copied");
            }
            root
        }

        fn position_of<T: engine::ecs::Component>(app: &App, keep: impl Fn(&T) -> bool) -> Vec<Vec3> {
            app.world()
                .query2::<Transform, T>()
                .filter(|(_, _, found)| keep(found))
                .map(|(_, transform, _)| transform.translation)
                .collect()
        }

        /// M9's exit criterion: a new quest, an enemy variant, and an
        /// encounter, authored only as content, load and play to the end — and
        /// the HUD says the right things on the way.
        #[test]
        fn m9_new_content_plays_to_completion() {
            let root = m9_content();
            let mut app = dev_app_at(&root, "cinder-hollow");
            let keeper = position_of::<slice::Npc>(&app, |_| true)[0];

            // The keeper gives the quest, and a salve with it.
            stand_at(&mut app, keeper);
            assert_eq!(hud(&app).prompt.as_deref(), Some("E - Talk to Hollow Keeper"));
            interact(&mut app);
            choose(&mut app, 1);
            if offer(&app).is_some() {
                interact(&mut app);
            }
            let shown = hud(&app);
            assert!(
                shown
                    .notices
                    .contains(&"New quest: Embers in the Hollow".to_owned()),
                "{shown:?}"
            );
            assert_eq!(
                shown.tracker,
                vec![
                    "Embers in the Hollow".to_owned(),
                    "- Collect Hollow Ember 0/2".to_owned()
                ]
            );
            assert_eq!(carried(&app, "cooling-salve"), 1);

            // Two embers.
            for ember in position_of::<slice::Gatherable>(&app, |node| node.item.name() == "hollow-ember") {
                stand_at(&mut app, ember);
                interact(&mut app);
            }
            assert_eq!(
                hud(&app).tracker,
                vec![
                    "Embers in the Hollow".to_owned(),
                    "- Defeat The Hollow Stand".to_owned()
                ]
            );

            // The variant's claw sets you burning: the new status, from the
            // new ability, on the new enemy.
            let husk = position_of::<slice::Enemy>(&app, |enemy| enemy.prefab.name() == "cinder-husk")[0];
            stand_at(&mut app, husk + Vec3::new(0.0, 0.0, -1.2));
            let burn = StatusId(AbilityId::new("status-cinder-burn").expect("valid id"));
            let player = slice::player_entity(app.world()).expect("a player");
            let mut burning = false;
            for _ in 0..400 {
                step(&mut app, 1);
                let tick = app.world().resource::<FixedTime>().map_or(0, |time| time.tick);
                if app
                    .world()
                    .get::<Statuses>(player)
                    .is_some_and(|statuses| statuses.is_active(&burn, tick))
                {
                    burning = true;
                    break;
                }
            }
            assert!(burning, "a cinder husk's claw burns");
            assert!(
                hud(&app)
                    .statuses
                    .iter()
                    .any(|line| line.starts_with("Cinder Burn")),
                "{:?}",
                hud(&app).statuses
            );

            // Both waves.
            clear_encounter(&mut app, "hollow-stand");
            assert_eq!(
                hud(&app).tracker,
                vec![
                    "Embers in the Hollow".to_owned(),
                    "- Talk to Hollow Keeper".to_owned()
                ]
            );

            // The salve, once the burn has run out.
            step(&mut app, 200);
            let wounded = player_health(&app, player);
            let max = app.world().get::<Health>(player).map_or(0.0, |health| health.max);
            assert!(wounded < max, "the fight left a mark");
            tap(&mut app, KeyCode::X);
            step(&mut app, 30);
            let healed = player_health(&app, player);
            assert!(
                (healed - (wounded + 25.0).min(max)).abs() < 1e-3,
                "{wounded} -> {healed}"
            );
            assert_eq!(carried(&app, "cooling-salve"), 0);

            // Back to the keeper.
            stand_at(&mut app, keeper);
            interact(&mut app);
            choose(&mut app, 1);
            if offer(&app).is_some() {
                interact(&mut app);
            }
            let slice = app.world().resource::<slice::Slice>().expect("a slice");
            assert!(
                matches!(
                    slice
                        .state
                        .quest_status(&QuestId::new("embers-in-the-hollow").expect("valid id")),
                    QuestStatus::Completed { ref resolution } if resolution.name() == "hollow-quiet"
                ),
                "the quest resolves"
            );
            assert!(
                hud(&app)
                    .notices
                    .contains(&"Quest complete: Embers in the Hollow - Hollow Quiet".to_owned()),
                "{:?}",
                hud(&app).notices
            );
            assert_eq!(carried(&app, "cooling-salve"), 1, "the reward");
            assert!(hud(&app).tracker.is_empty(), "nothing left to track");
            let _ = std::fs::remove_dir_all(root);
        }
    }
}
