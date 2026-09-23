//! Visual effects and ground telegraphs (ADR-0017 §6, M7 wave 4).
//!
//! Two things a player reads a fight by, beyond the bodies themselves:
//!
//! - **Effects** — bursts of shards, pulses along the ground, beams — composed
//!   in content ([`game_core::content::VfxDef`]) and played where gameplay says
//!   something happened. Gameplay *requests* an effect; presentation plays it
//!   on a later frame. Nothing flows back: no system in the fixed step reads an
//!   effect, so stripping every one leaves a fight identical.
//! - **Telegraphs** — a mark on the ground under every blow that is coming,
//!   its outline the hit volume and its fill growing through the wind-up, so a
//!   player can see both *where* and *when*. Read from the live [`Attack`] or
//!   [`Casting`] every frame, never authored, so a telegraph cannot disagree
//!   with the blow it warns of.
//!
//! Both are drawn as [`Model`]s — one entity per effect or mark, one part per
//! piece — so they reuse the model draw path whole and need nothing new from
//! the renderer. Pieces are opaque, low-poly, and lit, and read as glowing
//! through their material's flash overlay rather than through transparency.
//!
//! # Bounds
//!
//! Content bounds each effect ([`game_core::content::MAX_BURST_SHARDS`] and
//! the rest). The runtime bounds them together: past [`MAX_LIVE_PARTICLES`] a
//! new effect is dropped whole and counted, and past [`MAX_PENDING`] so is a
//! request. Caps drop; they never stall, and they never touch gameplay.

use std::collections::BTreeMap;
use std::f32::consts::TAU;

use content_runtime::ContentRegistry;
use engine::ecs::{Component, Entity, Resource, World};
use engine::math::{Quat, Transform, Vec3};
use engine::render::{MaterialDesc, MaterialHandle, MeshHandle, Model, ModelPart, RenderAssets, Visibility};
use engine::shapes::{self, FaceColors, Shape};
use game_core::content::{Effect, Fixed, Targeting, VfxPart};
use game_core::ids::VfxId;
use game_runtime::combat::{Attack, AttackPhase};
use game_runtime::movement::{Capsule, CollisionWorld};
use game_runtime::projectile::Projectile;

use crate::slice::{self, Casting, Enemy, Facing, Flying, ModelLibrary};

/// Most pieces on screen at once, across every effect.
///
/// About eleven full-size bursts. Each piece is a draw, so this is a ceiling
/// on what effects can cost the renderer however many land at once.
pub(crate) const MAX_LIVE_PARTICLES: usize = 384;

/// Most effects waiting to be played. Requests are drained every frame, so
/// this is only reached when frames stop running while ticks do.
pub(crate) const MAX_PENDING: usize = 64;

/// Heights above the ground, far enough apart not to fight for depth.
const FILL_LIFT: f32 = 0.02;
const RING_LIFT: f32 = 0.04;
const PULSE_LIFT: f32 = 0.06;

/// A ring's band, as a share of its radius.
const RING_BAND: f32 = 0.08;
const RING_SEGMENTS: u32 = 20;
/// How fast a shard tumbles, in radians per second.
const SHARD_SPIN: f32 = 9.0;
/// A beam shorter than this has nothing to span and is skipped.
const SHORTEST_BEAM: f32 = 0.05;

/// How much each piece's own colour outshines the scene's lighting: enough to
/// read as lit from within, not so much that it loses its facets.
const GLOW: f32 = 0.55;

/// A lane telegraph's rails, as a width in metres.
const RAIL: f32 = 0.06;
/// How often a projectile sheds its trail, in ticks.
pub(crate) const TRAIL_EVERY: u64 = 3;
/// Edge of the block a projectile with no model of its own flies as.
const FALLBACK_BOLT: f32 = 0.25;

fn metres(value: Fixed) -> f32 {
    value.as_milli() as f32 / 1000.0
}

/// A part of an effect, read into runtime terms once at startup.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    Burst {
        count: u8,
        size: f32,
        speed: f32,
        gravity: f32,
    },
    Pulse {
        radius: f32,
    },
    Beam {
        width: f32,
    },
    /// Angles in radians.
    Slash {
        radius: f32,
        arc: f32,
        width: f32,
        tilt: f32,
    },
    Crack {
        count: u8,
        length: f32,
        width: f32,
    },
    Spire {
        count: u8,
        height: f32,
        width: f32,
        spread: f32,
    },
}

/// How much of an eruption's life the outermost spire waits before it rises:
/// the spires go up as a wave, nearest first.
const SPIRE_WAVE: f32 = 0.22;
/// How much of its life one spire takes to punch up out of the ground.
const SPIRE_RISE: f32 = 0.08;
/// How late in an eruption the spires start sinking back.
const SPIRE_SINKING: f32 = 0.72;
/// How far a spire overshoots its height as it bursts up, before settling.
const SPIRE_OVERSHOOT: f32 = 0.12;

/// How much of a slash's life its sweep takes: the rest is fading.
const SLASH_SWEEP: f32 = 0.35;
/// How much of a crack's life it spends racing out.
const CRACK_GROWTH: f32 = 0.2;
/// How late in a crack's life it starts to close.
const CRACK_CLOSING: f32 = 0.6;
/// A fissure's height above the ground: under a pulse, over a telegraph.
const CRACK_LIFT: f32 = 0.03;

#[derive(Debug, Clone, Copy, PartialEq)]
struct BuiltPart {
    kind: Kind,
    ticks: u32,
    material: MaterialHandle,
    /// Its own mesh, for a part whose look is its shape — a spire, painted
    /// rock and glowing tip. Everything else draws with a shared mesh and
    /// takes its colour from `material`.
    mesh: Option<MeshHandle>,
}

/// What a telegraph borrows, by whose blow it is and whether it is live.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TelegraphLook {
    ring: [MaterialHandle; 5],
    fill: [MaterialHandle; 5],
}

/// Which look a warning wears: the player's own blows, a threat, each coming
/// or live, and last the aiming colour of an ability not yet cast.
fn look_index(warning: &Warning) -> usize {
    if warning.aiming {
        4
    } else {
        usize::from(warning.hostile) * 2 + usize::from(warning.live)
    }
}

/// Edge of the reticle under the cursor, in metres.
const RETICLE: f32 = 0.35;

/// Every effect content defines, built once at startup, and the meshes every
/// effect and telegraph shares.
///
/// Meshes are white and shared — one block, one ring, one disc — and colour
/// comes from each part's material, so a new effect costs materials, never
/// meshes.
#[derive(Debug)]
pub(crate) struct VfxLibrary {
    effects: BTreeMap<VfxId, Vec<BuiltPart>>,
    block: MeshHandle,
    ring: MeshHandle,
    disc: MeshHandle,
    telegraph: TelegraphLook,
    /// What a projectile with no model of its own flies as.
    bolt: MaterialHandle,
}

impl Resource for VfxLibrary {}

/// A unit ring on the ground: `RING_SEGMENTS` blocks set edge to edge. Built
/// from blocks because every shape is convex and a ring is not.
fn ring_mesh() -> engine::render::MeshData {
    let white = FaceColors::uniform(renderer_white());
    let angle = TAU / RING_SEGMENTS as f32;
    // Each block's length is its chord plus a little, so neighbours overlap
    // rather than leaving a hairline gap at the outer edge.
    let length = 2.0 * (angle * 0.5).sin() * 1.05;
    let block = shapes::mesh(
        Shape::Box {
            size: Vec3::new(length, 0.02, RING_BAND),
        },
        white,
    );
    shapes::merged((0..RING_SEGMENTS).map(|index| {
        let theta = angle * index as f32;
        let placement = Transform {
            translation: Vec3::new(theta.sin(), 0.0, theta.cos()) * (1.0 - RING_BAND * 0.5),
            rotation: Quat::from_rotation_y(theta),
            scale: Vec3::ONE,
        };
        shapes::placed(block.clone(), &placement)
    }))
}

/// One spire of rock, one metre tall and one wide at its base, its base at
/// the origin: a jagged main spike with a lesser shard leaning off it, and —
/// in `tip` — a molten cap over the top of the main spike. Convex pieces
/// merged, like a model's parts, and painted in their vertices.
fn spire_mesh(rock: [f32; 3], tip: [f32; 3]) -> engine::render::MeshData {
    let spike = |bottom: f32, height: f32, color: [f32; 3], at: Vec3, lean: Quat| {
        shapes::placed(
            shapes::mesh(
                Shape::Taper {
                    bottom: engine::math::Vec2::new(bottom, bottom * 0.85),
                    top: engine::math::Vec2::ZERO,
                    height,
                },
                FaceColors::uniform(color),
            ),
            &Transform {
                // A taper is centred on its height; stand it on its base.
                translation: at + lean * (Vec3::Y * height * 0.5),
                rotation: lean,
                scale: Vec3::ONE,
            },
        )
    };
    shapes::merged([
        spike(1.0, 1.0, rock, Vec3::ZERO, Quat::from_rotation_y(0.3)),
        // The molten cap: from just over halfway up, a hair wider than the
        // spike there so it sits proud of the rock rather than inside it.
        spike(0.47, 0.44, tip, Vec3::Y * 0.56, Quat::from_rotation_y(0.3)),
        // A lesser shard broken off the side, leaning out, and set down so
        // its lowest corner — not its centre — meets the ground.
        grounded(spike(
            0.5,
            0.55,
            rock,
            Vec3::new(0.32, 0.0, 0.1),
            Quat::from_rotation_z(-0.4) * Quat::from_rotation_y(0.9),
        )),
    ])
}

/// `mesh` moved up or down so its lowest point is at `y = 0`.
fn grounded(mut mesh: engine::render::MeshData) -> engine::render::MeshData {
    let lowest = mesh
        .positions
        .iter()
        .map(|position| position[1])
        .fold(f32::INFINITY, f32::min);
    if lowest.is_finite() {
        for position in &mut mesh.positions {
            position[1] -= lowest;
        }
    }
    mesh
}

fn renderer_white() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

impl VfxLibrary {
    /// Builds every effect in `content`, or nothing if the shared meshes
    /// cannot be uploaded.
    pub(crate) fn build(content: &ContentRegistry, assets: &mut dyn RenderAssets) -> Option<Self> {
        let white = FaceColors::uniform(renderer_white());
        let block = shapes::mesh(Shape::Box { size: Vec3::ONE }, white);
        let disc = shapes::mesh(
            Shape::Frustum {
                bottom_radius: 1.0,
                top_radius: 1.0,
                height: 0.01,
                sides: 24,
            },
            white,
        );
        let upload = |assets: &mut dyn RenderAssets, mesh: &engine::render::MeshData, name: &'static str| {
            assets
                .upload_mesh(mesh)
                .map_err(|error| engine::log_error!("vfx", "mesh upload failed", mesh = name, error = error))
                .ok()
        };
        let block = upload(assets, &block, "vfx-block")?;
        let ring = upload(assets, &ring_mesh(), "vfx-ring")?;
        let disc = upload(assets, &disc, "vfx-disc")?;

        fn glow(assets: &mut dyn RenderAssets, [r, g, b]: [f32; 3], amount: f32) -> MaterialHandle {
            assets.create_material(&MaterialDesc {
                base_color: [r, g, b, 1.0],
                flash: [r, g, b, amount],
            })
        }
        // Spires are solid rock: lit like the world, their colour in their
        // vertices, so they shade as things standing in the light do.
        let rock = assets.create_material(&MaterialDesc::default());

        let mut effects = BTreeMap::new();
        for def in content.vfx.values() {
            // Validated at load; a miss means content changed under us.
            let Some(palette) = content.palettes.get(&def.palette) else {
                continue;
            };
            let parts = def
                .parts
                .iter()
                .map(|part| {
                    let color = palette
                        .swatches
                        .get(part.color())
                        .map_or(renderer_white(), |&rgb| slice::linear(rgb));
                    let kind = match part {
                        VfxPart::Burst {
                            count,
                            size,
                            speed,
                            gravity,
                            ..
                        } => Kind::Burst {
                            count: *count,
                            size: metres(*size),
                            speed: metres(*speed),
                            gravity: metres(*gravity),
                        },
                        VfxPart::Pulse { radius, .. } => Kind::Pulse {
                            radius: metres(*radius),
                        },
                        VfxPart::Beam { width, .. } => Kind::Beam {
                            width: metres(*width),
                        },
                        VfxPart::Slash {
                            radius,
                            arc,
                            width,
                            tilt,
                            ..
                        } => Kind::Slash {
                            radius: metres(*radius),
                            arc: metres(*arc).to_radians(),
                            width: metres(*width),
                            tilt: metres(*tilt).to_radians(),
                        },
                        VfxPart::Crack {
                            count, length, width, ..
                        } => Kind::Crack {
                            count: *count,
                            length: metres(*length),
                            width: metres(*width),
                        },
                        VfxPart::Spire {
                            count,
                            height,
                            width,
                            spread,
                            glow: tip,
                            ..
                        } => {
                            let tip = tip
                                .as_ref()
                                .and_then(|swatch| palette.swatches.get(swatch))
                                .map_or(color, |&rgb| slice::linear(rgb));
                            let mesh = assets
                                .upload_mesh(&spire_mesh(color, tip))
                                .map_err(|error| {
                                    engine::log_warn!(
                                        "vfx",
                                        "a spire failed to upload",
                                        effect = def.id,
                                        error = error
                                    )
                                })
                                .ok();
                            return BuiltPart {
                                kind: Kind::Spire {
                                    count: *count,
                                    height: metres(*height),
                                    width: metres(*width),
                                    spread: metres(*spread),
                                },
                                ticks: part.ticks(),
                                material: rock,
                                mesh,
                            };
                        }
                    };
                    BuiltPart {
                        kind,
                        ticks: part.ticks(),
                        material: glow(assets, color, GLOW),
                        mesh: None,
                    }
                })
                .collect();
            effects.insert(def.id.clone(), parts);
        }

        // The same language as the body cues: yellow is coming and red is now
        // for a blow aimed at the player; the player's own blows are a pale
        // blue that brightens, so their own reach never reads as a threat.
        let telegraph = TelegraphLook {
            ring: [
                glow(assets, [0.35, 0.60, 0.95], 0.5),
                glow(assets, [0.80, 0.92, 1.00], 0.7),
                glow(assets, [0.95, 0.80, 0.10], 0.6),
                glow(assets, [1.00, 0.10, 0.05], 0.8),
                // Aiming: plain white, the one colour no threat or blow uses,
                // so an indicator never reads as something already coming.
                glow(assets, [0.92, 0.92, 0.88], 0.8),
            ],
            fill: [
                glow(assets, [0.20, 0.32, 0.50], 0.3),
                glow(assets, [0.45, 0.55, 0.70], 0.4),
                glow(assets, [0.55, 0.40, 0.05], 0.35),
                glow(assets, [0.70, 0.08, 0.04], 0.5),
                glow(assets, [0.50, 0.50, 0.48], 0.4),
            ],
        };

        let bolt = glow(assets, slice::linear((255, 170, 60)), 0.7);

        Some(Self {
            effects,
            block,
            ring,
            disc,
            telegraph,
            bolt,
        })
    }

    /// Pieces `vfx` would draw if played from `from` to `at`.
    fn pieces(&self, vfx: &VfxId, from: Vec3, at: Vec3) -> Vec<Piece> {
        let Some(parts) = self.effects.get(vfx) else {
            return Vec::new();
        };
        let spans = (from - at).length() >= SHORTEST_BEAM;
        let mut pieces = Vec::new();
        for part in parts {
            match part.kind {
                Kind::Burst { count, .. } => {
                    pieces.extend((0..count).map(|index| Piece { part: *part, index }));
                }
                Kind::Pulse { .. } => pieces.push(Piece {
                    part: *part,
                    index: 0,
                }),
                Kind::Beam { .. } if spans => pieces.push(Piece {
                    part: *part,
                    index: 0,
                }),
                // A slash needs a direction to sweep across, which only a
                // span gives it.
                Kind::Slash { .. } if spans => {
                    pieces.extend(
                        (0..game_core::content::SLASH_SEGMENTS).map(|index| Piece { part: *part, index }),
                    );
                }
                Kind::Crack { count, .. } | Kind::Spire { count, .. } => {
                    pieces.extend((0..count).map(|index| Piece { part: *part, index }));
                }
                Kind::Beam { .. } | Kind::Slash { .. } => {}
            }
        }
        pieces
    }

    fn mesh_for(&self, part: &BuiltPart) -> MeshHandle {
        if let Some(mesh) = part.mesh {
            return mesh;
        }
        match part.kind {
            Kind::Burst { .. }
            | Kind::Beam { .. }
            | Kind::Slash { .. }
            | Kind::Crack { .. }
            | Kind::Spire { .. } => self.block,
            Kind::Pulse { .. } => self.ring,
        }
    }
}

/// One effect gameplay asked for.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Request {
    pub vfx: VfxId,
    pub from: Vec3,
    pub at: Vec3,
}

/// Effects waiting to be played, and what the caps turned away.
#[derive(Debug, Default)]
pub(crate) struct VfxRuntime {
    pending: Vec<Request>,
    /// Effects and requests dropped by a cap, ever.
    pub dropped: u64,
    /// Varies each effect's scatter, so two bursts in one place differ.
    next_seed: u32,
}

impl Resource for VfxRuntime {}

/// Asks for `vfx` to play from `from` to `at`.
///
/// The one door from gameplay to effects, and it only ever writes: a request
/// is a note for presentation, and nothing that decides a fight reads it. With
/// no runtime installed — a world without presentation — it does nothing.
pub(crate) fn request(world: &mut World, vfx: &VfxId, from: Vec3, at: Vec3) {
    let Some(runtime) = world.resource_mut::<VfxRuntime>() else {
        return;
    };
    if runtime.pending.len() >= MAX_PENDING {
        runtime.dropped += 1;
        return;
    }
    runtime.pending.push(Request {
        vfx: vfx.clone(),
        from,
        at,
    });
}

/// One piece of a playing effect: which part it belongs to, and which shard.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Piece {
    part: BuiltPart,
    index: u8,
}

/// An effect on screen.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Playing {
    born: u64,
    ends: u64,
    seed: u32,
    /// Where the effect came from, relative to where it plays.
    toward: Vec3,
    pieces: Vec<Piece>,
}

impl Component for Playing {}

/// A small, fast, well-mixed hash to `0..1`: the scatter of one shard.
///
/// Presentation only, so it need not match anything; it need only be stable,
/// which keeps a burst from shimmering between frames.
fn unit(seed: u32, index: u8, salt: u32) -> f32 {
    let mut x = seed
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(u32::from(index).wrapping_mul(0x85EB_CA6B))
        .wrapping_add(salt.wrapping_mul(0xC2B2_AE35));
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    (x >> 8) as f32 / (1u32 << 24) as f32
}

/// Where one piece is, `age` ticks into its part, relative to the effect's
/// origin.
///
/// Pure, so each primitive's motion is testable without a world: a burst
/// throws, falls and shrinks; a pulse spreads; a beam spans and narrows.
/// `floor` is the ground's height relative to the origin, which no piece
/// sinks through; `toward` is where the effect came from.
fn piece_pose(piece: Piece, seed: u32, age: f32, step_seconds: f32, toward: Vec3, floor: f32) -> Transform {
    let ticks = piece.part.ticks.max(1) as f32;
    let progress = (age / ticks).clamp(0.0, 1.0);
    let fading = 1.0 - progress;
    match piece.part.kind {
        Kind::Burst {
            count,
            size,
            speed,
            gravity,
        } => {
            let seconds = age * step_seconds;
            // Spread evenly around, jittered so the ring of shards is not
            // mechanical; thrown between low and steep so some skim and some
            // arc.
            let around =
                (f32::from(piece.index) + unit(seed, piece.index, 1) * 0.6) / f32::from(count.max(1)) * TAU;
            let up = 0.25 + unit(seed, piece.index, 2) * 0.95;
            let direction = Vec3::new(up.cos() * around.cos(), up.sin(), up.cos() * around.sin());
            let pace = speed * (0.6 + 0.4 * unit(seed, piece.index, 3));
            let mut at = direction * pace * seconds;
            at.y -= 0.5 * gravity * seconds * seconds;
            let edge = size * fading;
            at.y = at.y.max(floor + edge * 0.5);
            let axis = Vec3::new(
                unit(seed, piece.index, 4) - 0.5,
                unit(seed, piece.index, 5) - 0.5,
                unit(seed, piece.index, 6) - 0.5,
            )
            .try_normalize()
            .unwrap_or(Vec3::Y);
            Transform {
                translation: at,
                rotation: Quat::from_axis_angle(axis, seconds * SHARD_SPIN),
                scale: Vec3::splat(edge),
            }
        }
        Kind::Pulse { radius } => {
            // Fast out, slowing as it reaches its edge, like a shockwave.
            let reach = radius * (1.0 - fading * fading);
            Transform {
                translation: Vec3::new(0.0, floor + PULSE_LIFT, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::new(reach, fading, reach),
            }
        }
        Kind::Beam { width } => {
            let length = toward.length();
            let Some(along) = toward.try_normalize() else {
                return Transform {
                    scale: Vec3::ZERO,
                    ..Transform::IDENTITY
                };
            };
            let thickness = width * fading;
            Transform {
                translation: toward * 0.5,
                rotation: Quat::from_rotation_arc(Vec3::Z, along),
                scale: Vec3::new(thickness, thickness, length),
            }
        }
        Kind::Slash {
            radius,
            arc,
            width,
            tilt,
        } => {
            // The arc is centred on where the effect came from — the one
            // swinging — and faces where it plays: the blow's path.
            let hidden = Transform {
                scale: Vec3::ZERO,
                ..Transform::IDENTITY
            };
            let Some(forward) = Vec3::new(-toward.x, 0.0, -toward.z).try_normalize() else {
                return hidden;
            };
            let segments = f32::from(game_core::content::SLASH_SEGMENTS);
            let share = (f32::from(piece.index) + 0.5) / segments;
            // Segments appear one after another across the sweep.
            let swept = (progress / SLASH_SWEEP).min(1.0);
            if swept < share {
                return hidden;
            }
            let roll = Quat::from_axis_angle(forward, tilt);
            let angle = -arc * 0.5 + arc * share;
            let out = roll * (Quat::from_rotation_y(angle) * forward);
            let along = roll * (Quat::from_rotation_y(angle) * forward.cross(Vec3::Y)).normalize();
            let normal = along.cross(out).normalize();
            // The newest segments are the widest: the blade's edge is where
            // it is now, and the trail behind it is already thinning.
            let age = (swept - share).max(0.0);
            let thickness = width * (1.0 - progress).max(0.0) * (1.0 - 0.5 * age);
            let chord = 2.0 * radius * (arc / (2.0 * segments)).sin() * 1.15;
            Transform {
                translation: toward + out * radius,
                rotation: Quat::from_mat3(&engine::math::Mat3::from_cols(along, out, normal)),
                scale: Vec3::new(chord, thickness, 0.04),
            }
        }
        Kind::Crack { count, length, width } => {
            let around =
                (f32::from(piece.index) + unit(seed, piece.index, 7) * 0.7) / f32::from(count.max(1)) * TAU;
            let reach = length * (0.6 + 0.4 * unit(seed, piece.index, 8));
            // Races out, holds, then closes again.
            let grown = 1.0 - (1.0 - (progress / CRACK_GROWTH).min(1.0)).powi(2);
            let closing = if progress > CRACK_CLOSING {
                ((1.0 - progress) / (1.0 - CRACK_CLOSING)).max(0.0)
            } else {
                1.0
            };
            let span = reach * grown;
            let direction = Vec3::new(around.cos(), 0.0, around.sin());
            Transform {
                translation: direction * span * 0.5 + Vec3::Y * (floor + CRACK_LIFT),
                rotation: Quat::from_rotation_y(-around),
                scale: Vec3::new(span, 0.02, width * closing),
            }
        }
        Kind::Spire {
            count,
            height,
            width,
            spread,
        } => {
            // Where it stands: scattered through the spread, the ring of
            // them jittered so they do not read as a pattern.
            let around =
                (f32::from(piece.index) + unit(seed, piece.index, 9) * 0.8) / f32::from(count.max(1)) * TAU;
            let out = 0.3 + 0.7 * unit(seed, piece.index, 10);
            let direction = Vec3::new(around.cos(), 0.0, around.sin());
            let tall = height * (0.55 + 0.45 * unit(seed, piece.index, 11));
            let wide = width * (0.7 + 0.5 * unit(seed, piece.index, 12));

            // Up as a wave from the centre, a burst with a little overshoot,
            // a hold, then down again the way it came.
            let rise = ((progress - SPIRE_WAVE * out) / SPIRE_RISE).clamp(0.0, 1.0);
            let burst = 1.0 - (1.0 - rise).powi(3);
            let overshoot = SPIRE_OVERSHOOT * (rise * std::f32::consts::PI).sin();
            let sinking = ((progress - SPIRE_SINKING) / (1.0 - SPIRE_SINKING)).clamp(0.0, 1.0);
            let sink = sinking * sinking * (3.0 - 2.0 * sinking);
            let emerged = (burst * (1.0 - sink)).clamp(0.0, 1.0);

            // It rises by moving up out of the ground, not by growing: what
            // has not come up yet is below the floor, and the floor hides it.
            let lean = Quat::from_axis_angle(
                Vec3::Y.cross(direction).try_normalize().unwrap_or(Vec3::X),
                -(0.2 + 0.25 * unit(seed, piece.index, 13)),
            );
            let spin = Quat::from_rotation_y(unit(seed, piece.index, 14) * TAU);
            Transform {
                translation: direction * spread * out + Vec3::Y * (floor - tall * (1.0 - emerged)),
                rotation: lean * spin,
                scale: Vec3::new(wide, tall * (1.0 + overshoot * (1.0 - sink)), wide),
            }
        }
    }
}

/// Pieces on screen now, across every effect.
pub(crate) fn live_particles(world: &World) -> usize {
    world
        .query2::<Playing, Model>()
        .map(|(_, _, model)| model.parts.len())
        .sum()
}

/// Plays what was requested, moves what is playing, and clears what is done.
pub(crate) fn play_effects(world: &mut World, tick: u64, step_seconds: f32) {
    let pending = match world.resource_mut::<VfxRuntime>() {
        Some(runtime) => std::mem::take(&mut runtime.pending),
        None => return,
    };
    world.resource_scope::<VfxLibrary, _>(|world, library| {
        let mut live = live_particles(world);
        for request in pending {
            let pieces = library.pieces(&request.vfx, request.from, request.at);
            if pieces.is_empty() {
                continue;
            }
            // Whole or not at all: half a burst reads as a different effect.
            if live + pieces.len() > MAX_LIVE_PARTICLES {
                if let Some(runtime) = world.resource_mut::<VfxRuntime>() {
                    runtime.dropped += 1;
                }
                continue;
            }
            live += pieces.len();
            let seed = match world.resource_mut::<VfxRuntime>() {
                Some(runtime) => {
                    runtime.next_seed = runtime.next_seed.wrapping_add(1);
                    runtime.next_seed
                }
                None => 0,
            };
            let longest = pieces.iter().map(|piece| piece.part.ticks).max().unwrap_or(0);
            let parts = pieces
                .iter()
                .map(|piece| {
                    ModelPart::new(
                        library.mesh_for(&piece.part),
                        piece.part.material,
                        Transform::IDENTITY,
                    )
                })
                .collect();
            let entity = world.spawn();
            let _ = world.insert(entity, Transform::from_translation(request.at));
            let _ = world.insert(entity, Model { parts });
            let _ = world.insert(
                entity,
                Playing {
                    born: tick,
                    ends: tick + u64::from(longest),
                    seed,
                    toward: request.from - request.at,
                    pieces,
                },
            );
        }
    });

    let playing: Vec<(Entity, Playing, Vec3)> = world
        .query2::<Playing, Transform>()
        .map(|(entity, playing, transform)| (entity, playing.clone(), transform.translation))
        .collect();
    for (entity, playing, origin) in playing {
        if tick >= playing.ends {
            let _ = world.despawn(entity);
            continue;
        }
        let age = tick.saturating_sub(playing.born) as f32;
        let Some(model) = world.get_mut::<Model>(entity) else {
            continue;
        };
        for (part, piece) in model.parts.iter_mut().zip(&playing.pieces) {
            part.pose = piece_pose(*piece, playing.seed, age, step_seconds, playing.toward, -origin.y);
        }
    }
}

/// The ground a coming blow covers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Ground {
    /// A circle about the warning's `centre`.
    Circle { radius: f32 },
    /// A lane from the warning's `centre` along `direction`: where a
    /// projectile will fly, as wide as it strikes and as long as it will get
    /// before its range runs out or a wall stops it.
    Lane {
        direction: Vec3,
        length: f32,
        width: f32,
    },
}

/// One blow that is coming: where it will land, how big, how soon.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Warning {
    /// On the ground: a circle's centre, or where a lane starts.
    pub centre: Vec3,
    pub ground: Ground,
    /// Through the wind-up, `0..1`; `1` once the blow is live.
    pub progress: f32,
    /// Aimed at the player, rather than by them.
    pub hostile: bool,
    /// The hit volume is live this tick.
    pub live: bool,
    /// Not a blow at all yet: an ability the player is aiming, before the
    /// cast (see `aim`).
    pub aiming: bool,
}

/// Where the wind-up stands, or `None` once the blow is spent.
fn warning_progress(phase: AttackPhase, progress: f32) -> Option<(f32, bool)> {
    match phase {
        AttackPhase::Windup => Some((progress, false)),
        AttackPhase::Active => Some((1.0, true)),
        AttackPhase::Recovery | AttackPhase::Finished => None,
    }
}

/// Every blow that is coming, read from the gameplay that will deliver it.
pub(crate) fn warnings(world: &World, content: &ContentRegistry) -> Vec<Warning> {
    let mut found = Vec::new();
    let on_ground = |at: Vec3| Vec3::new(at.x, 0.0, at.z);
    for (entity, transform, attack) in world.query2::<Transform, Attack>() {
        let (phase, progress) = slice::phase_progress(attack.timing, attack.elapsed());
        let Some((progress, live)) = warning_progress(phase, progress) else {
            continue;
        };
        let facing = world.get::<Facing>(entity).map_or(Vec3::NEG_Z, |facing| facing.0);
        found.push(Warning {
            centre: on_ground(transform.translation + facing * attack.reach),
            ground: Ground::Circle {
                radius: attack.radius,
            },
            progress,
            hostile: world.get::<Enemy>(entity).is_some(),
            live,
            aiming: false,
        });
    }
    for (entity, transform, casting) in world.query2::<Transform, Casting>() {
        let Some(def) = content.abilities.get(&casting.ability) else {
            continue;
        };
        if !def.presentation.telegraph {
            continue;
        }
        let (phase, progress) = slice::phase_progress(casting.timing, casting.elapsed);
        let Some((progress, live)) = warning_progress(phase, progress) else {
            continue;
        };
        let facing = world.get::<Facing>(entity).map_or(Vec3::NEG_Z, |facing| facing.0);
        let Some(ground) = ability_ground(world, def, transform.translation, facing, false) else {
            continue;
        };
        found.push(Warning {
            centre: on_ground(transform.translation),
            ground,
            progress,
            hostile: world.get::<Enemy>(entity).is_some(),
            live,
            aiming: false,
        });
    }

    // The ability the player is aiming, before it is cast: its own shape, in
    // the aiming colour, pointed at the cursor.
    let aimed = world
        .resource::<crate::aim::CastInput>()
        .and_then(|input| input.aiming);
    if let Some(slot) = aimed
        && let Some((player, transform, _)) = world.query2::<Transform, slice::Player>().next()
        && let Some(def) = world
            .get::<slice::Abilities>(player)
            .and_then(|known| known.0.get(slot))
            .and_then(|id| content.abilities.get(id))
    {
        let toward = world
            .resource::<crate::aim::Aim>()
            .and_then(|aim| aim.point)
            .map(|point| point - transform.translation)
            .unwrap_or_else(|| world.get::<Facing>(player).map_or(Vec3::NEG_Z, |facing| facing.0));
        if let Some(ground) = ability_ground(world, def, transform.translation, toward, true) {
            found.push(Warning {
                centre: on_ground(transform.translation),
                ground,
                progress: 0.0,
                hostile: false,
                live: false,
                aiming: true,
            });
        }
    }
    found
}

/// The ground an ability covers, cast from `origin` toward `toward`.
///
/// A projectile's lane takes precedence: it is where the ability's danger
/// actually goes. Otherwise only an area has ground to mark for a telegraph —
/// validation allows nothing else, and this checks again so a telegraph is
/// never drawn for a shape it cannot know. An aiming `preview` also shows a
/// single-target ability's reach, and a small ring for one on the caster, so
/// every ability has something to aim with.
fn ability_ground(
    world: &World,
    def: &game_core::content::AbilityDef,
    origin: Vec3,
    toward: Vec3,
    preview: bool,
) -> Option<Ground> {
    if let Some(Effect::Projectile { range, radius, .. }) = def.projectile() {
        let direction = Vec3::new(toward.x, 0.0, toward.z).try_normalize()?;
        // Bounded as launch bounds it, and cut short where a wall will stop
        // it, so the lane ends where the bolt will.
        let range = metres(*range).clamp(0.0, game_core::content::MAX_PROJECTILE_RANGE_METRES as f32);
        let radius = metres(*radius).clamp(
            game_core::content::MIN_PROJECTILE_RADIUS_MILLI as f32 / 1000.0,
            game_core::content::MAX_PROJECTILE_RADIUS_MILLI as f32 / 1000.0,
        );
        let length = world
            .resource::<CollisionWorld>()
            .and_then(|collision| collision.sweep(Capsule::new(radius, 0.0), origin, direction * range))
            .map_or(range, |time| range * time);
        return Some(Ground::Lane {
            direction,
            length,
            width: radius * 2.0,
        });
    }
    match def.targeting {
        // `AreaAtPoint` lands on the caster until the runtime can aim a point
        // (`fire` passes none), so both mark the ground around it.
        Targeting::AreaAroundSelf | Targeting::AreaAtPoint => Some(Ground::Circle {
            radius: slice::ABILITY_REACH,
        }),
        Targeting::SingleEnemy | Targeting::SingleAlly if preview => Some(Ground::Circle {
            radius: slice::ABILITY_REACH,
        }),
        Targeting::SelfOnly if preview => Some(Ground::Circle { radius: 1.0 }),
        _ => None,
    }
}

/// A circle on the ground, reused from frame to frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TelegraphMark;

impl Component for TelegraphMark {}

/// A lane on the ground, reused from frame to frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LaneMark;

impl Component for LaneMark {}

/// The pooled marks of one kind, spare ones first spawned with `parts`.
fn pooled<C: Component + Copy>(
    world: &mut World,
    marker: C,
    needed: usize,
    parts: &[ModelPart],
) -> Vec<Entity> {
    let mut marks: Vec<Entity> = world.query::<C>().map(|(entity, _)| entity).collect();
    // Query order is storage order; sorting makes which mark shows which
    // warning stable from frame to frame.
    marks.sort();
    while marks.len() < needed {
        let mark = world.spawn();
        let _ = world.insert(mark, marker);
        let _ = world.insert(
            mark,
            Model {
                parts: parts.to_vec(),
            },
        );
        marks.push(mark);
    }
    marks
}

/// Draws a mark under every blow that is coming, and hides the rest.
///
/// Marks are pooled: one is spawned the first time it is needed and hidden,
/// not despawned, when its blow is spent, so a long fight does not churn
/// entities at the rate enemies swing.
pub(crate) fn show_telegraphs(world: &mut World, content: &ContentRegistry) {
    let Some((ring, disc, block, look)) = world
        .resource::<VfxLibrary>()
        .map(|library| (library.ring, library.disc, library.block, library.telegraph))
    else {
        return;
    };
    let warnings = warnings(world, content);
    let circles: Vec<&Warning> = warnings
        .iter()
        .filter(|warning| matches!(warning.ground, Ground::Circle { .. }))
        .collect();
    let lanes: Vec<&Warning> = warnings
        .iter()
        .filter(|warning| matches!(warning.ground, Ground::Lane { .. }))
        .collect();

    let circle_marks = pooled(
        world,
        TelegraphMark,
        circles.len(),
        &[
            ModelPart::new(ring, look.ring[0], Transform::IDENTITY),
            ModelPart::new(disc, look.fill[0], Transform::IDENTITY),
        ],
    );
    let lane_marks = pooled(
        world,
        LaneMark,
        lanes.len(),
        &[
            ModelPart::new(block, look.ring[0], Transform::IDENTITY),
            ModelPart::new(block, look.ring[0], Transform::IDENTITY),
            ModelPart::new(block, look.fill[0], Transform::IDENTITY),
        ],
    );

    for (warning, &mark) in circles.iter().zip(&circle_marks) {
        let Ground::Circle { radius } = warning.ground else {
            continue;
        };
        let which = look_index(warning);
        let _ = world.insert(mark, Transform::from_translation(warning.centre));
        let _ = world.insert(mark, Visibility::VISIBLE);
        if let Some(model) = world.get_mut::<Model>(mark)
            && let [outline, fill] = model.parts.as_mut_slice()
        {
            outline.material = look.ring[which];
            outline.pose = Transform {
                translation: Vec3::Y * RING_LIFT,
                rotation: Quat::IDENTITY,
                scale: Vec3::new(radius, 1.0, radius),
            };
            let filled = radius * warning.progress;
            fill.material = look.fill[which];
            fill.pose = Transform {
                translation: Vec3::Y * FILL_LIFT,
                rotation: Quat::IDENTITY,
                scale: Vec3::new(filled, 1.0, filled),
            };
        }
    }

    for (warning, &mark) in lanes.iter().zip(&lane_marks) {
        let Ground::Lane {
            direction,
            length,
            width,
        } = warning.ground
        else {
            continue;
        };
        let which = look_index(warning);
        // The mark's own -Z runs down the lane.
        let _ = world.insert(
            mark,
            Transform {
                translation: warning.centre,
                rotation: Quat::from_rotation_arc(Vec3::NEG_Z, direction),
                scale: Vec3::ONE,
            },
        );
        let _ = world.insert(mark, Visibility::VISIBLE);
        if let Some(model) = world.get_mut::<Model>(mark)
            && let [left, right, fill] = model.parts.as_mut_slice()
        {
            for (rail, side) in [(left, -1.0), (right, 1.0)] {
                rail.material = look.ring[which];
                rail.pose = Transform {
                    translation: Vec3::new(side * (width - RAIL) * 0.5, RING_LIFT, -length * 0.5),
                    rotation: Quat::IDENTITY,
                    scale: Vec3::new(RAIL, 0.02, length),
                };
            }
            let filled = length * warning.progress;
            fill.material = look.fill[which];
            fill.pose = Transform {
                translation: Vec3::new(0.0, FILL_LIFT, -filled * 0.5),
                rotation: Quat::IDENTITY,
                scale: Vec3::new(width, 0.01, filled),
            };
        }
    }

    for &spare in circle_marks
        .iter()
        .skip(circles.len())
        .chain(lane_marks.iter().skip(lanes.len()))
    {
        let _ = world.insert(spare, Visibility::HIDDEN);
    }

    // The reticle: where the cursor meets the ground, shown only while an
    // ability is being aimed — the rest of the time it is clutter.
    let aiming = world
        .resource::<crate::aim::CastInput>()
        .is_some_and(|input| input.aiming.is_some());
    let aim = world
        .resource::<crate::aim::Aim>()
        .and_then(|aim| aim.point)
        .filter(|_| aiming);
    // Spawned the first time there is an aim, not before: a world with no
    // cursor — a headless one — carries no reticle at all.
    let reticles = pooled(
        world,
        Reticle,
        usize::from(aim.is_some()),
        &[ModelPart::new(ring, look.ring[4], Transform::IDENTITY)],
    );
    let Some(&reticle) = reticles.first() else {
        return;
    };
    match aim {
        Some(point) => {
            let _ = world.insert(
                reticle,
                Transform::from_translation(Vec3::new(point.x, 0.0, point.z)),
            );
            let _ = world.insert(reticle, Visibility::VISIBLE);
            if let Some(model) = world.get_mut::<Model>(reticle)
                && let Some(part) = model.parts.first_mut()
            {
                part.pose = Transform {
                    translation: Vec3::Y * RING_LIFT,
                    rotation: Quat::IDENTITY,
                    scale: Vec3::new(RETICLE, 1.0, RETICLE),
                };
            }
        }
        None => {
            let _ = world.insert(reticle, Visibility::HIDDEN);
        }
    }
}

/// The ring under the cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Reticle;

impl Component for Reticle {}

/// When a projectile last shed its trail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Trailing {
    last: u64,
}

impl Component for Trailing {}

/// Dresses every projectile in flight, turns it nose first, and sheds its
/// trail.
///
/// Presentation only. Its position is the simulation's, written in the fixed
/// step; this sets rotation and model, and asks for trail effects, and nothing
/// the fixed step reads.
pub(crate) fn show_projectiles(world: &mut World, content: &ContentRegistry, tick: u64) {
    let Some((block, bolt)) = world
        .resource::<VfxLibrary>()
        .map(|library| (library.block, library.bolt))
    else {
        return;
    };
    let flying: Vec<(Entity, Flying, Vec3, Vec3)> = world
        .query2::<Projectile, Flying>()
        .map(|(entity, projectile, flying)| {
            (entity, flying.clone(), projectile.position, projectile.direction)
        })
        .collect();
    for (entity, flying, position, direction) in flying {
        let look = content
            .abilities
            .get(&flying.ability)
            .map(|def| &def.presentation);
        if world.get::<Model>(entity).is_none() {
            let built = look
                .and_then(|look| look.projectile.as_ref())
                .and_then(|id| {
                    world
                        .resource::<ModelLibrary>()
                        .and_then(|library| library.get(id))
                })
                .map(|built| built.parts.clone());
            let parts = built.unwrap_or_else(|| {
                let mut part = ModelPart::new(block, bolt, Transform::IDENTITY);
                part.pose = Transform {
                    scale: Vec3::splat(FALLBACK_BOLT),
                    ..Transform::IDENTITY
                };
                vec![part]
            });
            let _ = world.insert(entity, Model { parts });
            let _ = world.insert(entity, Trailing { last: tick });
        }
        if let Some(transform) = world.get_mut::<Transform>(entity) {
            transform.rotation = Quat::from_rotation_arc(Vec3::NEG_Z, direction);
        }
        let due = world
            .get::<Trailing>(entity)
            .is_some_and(|trailing| tick >= trailing.last + TRAIL_EVERY);
        if due && let Some(trail) = look.and_then(|look| look.trail.as_ref()) {
            request(world, trail, position, position);
            let _ = world.insert(entity, Trailing { last: tick });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STEP: f32 = 1.0 / 60.0;

    fn material() -> MaterialHandle {
        let mut assets = engine::render::HeadlessRenderAssets::default();
        assets.create_material(&MaterialDesc::default())
    }

    fn burst(count: u8, gravity: f32, ticks: u32) -> BuiltPart {
        BuiltPart {
            kind: Kind::Burst {
                count,
                size: 0.2,
                speed: 4.0,
                gravity,
            },
            ticks,
            material: material(),
            mesh: None,
        }
    }

    #[test]
    fn a_burst_throws_outward_and_shrinks_away() {
        let part = burst(8, 0.0, 30);
        for index in 0..8 {
            let piece = Piece { part, index };
            let start = piece_pose(piece, 7, 0.0, STEP, Vec3::ZERO, -1.0);
            let middle = piece_pose(piece, 7, 15.0, STEP, Vec3::ZERO, -1.0);
            let end = piece_pose(piece, 7, 30.0, STEP, Vec3::ZERO, -1.0);
            assert!(
                start.translation.length() < 1e-5,
                "every shard starts at the origin"
            );
            assert!(
                middle.translation.length() > 0.5,
                "shard {index} has been thrown: {:?}",
                middle.translation
            );
            assert!((start.scale.x - 0.2).abs() < 1e-5, "full size at first");
            assert!(end.scale.x.abs() < 1e-5, "gone by the end: {:?}", end.scale);
        }
    }

    #[test]
    fn shards_scatter_rather_than_stack() {
        let part = burst(12, 0.0, 30);
        let spots: Vec<Vec3> = (0..12)
            .map(|index| piece_pose(Piece { part, index }, 3, 10.0, STEP, Vec3::ZERO, -1.0).translation)
            .collect();
        for (a, first) in spots.iter().enumerate() {
            for second in &spots[a + 1..] {
                assert!(
                    (*first - *second).length() > 0.05,
                    "two shards in one place: {first:?} and {second:?}"
                );
            }
        }
    }

    #[test]
    fn gravity_pulls_shards_down_but_never_through_the_ground() {
        let heavy = burst(8, 30.0, 120);
        let floor = -0.9;
        for index in 0..8 {
            let piece = Piece { part: heavy, index };
            let late = piece_pose(piece, 5, 110.0, STEP, Vec3::ZERO, floor);
            let lowest = late.translation.y - late.scale.y * 0.5;
            assert!(
                lowest >= floor - 1e-4,
                "shard {index} sank below the ground: {late:?}"
            );
        }
        // And gravity does pull: the same shard sits lower with it than
        // without it.
        let piece = Piece {
            part: heavy,
            index: 0,
        };
        let light = Piece {
            part: burst(8, 0.0, 120),
            index: 0,
        };
        let early = 20.0;
        assert!(
            piece_pose(piece, 5, early, STEP, Vec3::ZERO, -100.0)
                .translation
                .y
                < piece_pose(light, 5, early, STEP, Vec3::ZERO, -100.0)
                    .translation
                    .y
        );
    }

    #[test]
    fn a_pulse_spreads_to_its_radius_on_the_ground() {
        let part = BuiltPart {
            kind: Kind::Pulse { radius: 2.5 },
            ticks: 20,
            material: material(),
            mesh: None,
        };
        let piece = Piece { part, index: 0 };
        let floor = -0.9;
        let start = piece_pose(piece, 0, 0.0, STEP, Vec3::ZERO, floor);
        let nearly = piece_pose(piece, 0, 19.9, STEP, Vec3::ZERO, floor);
        assert!(start.scale.x.abs() < 1e-5, "starts as a point");
        assert!(
            (nearly.scale.x - 2.5).abs() < 0.01,
            "reaches its radius: {:?}",
            nearly.scale
        );
        assert!(
            (start.translation.y - (floor + PULSE_LIFT)).abs() < 1e-5,
            "lies on the ground, not at the origin's height"
        );
    }

    #[test]
    fn a_beam_spans_from_its_source_to_where_it_plays() {
        let part = BuiltPart {
            kind: Kind::Beam { width: 0.2 },
            ticks: 10,
            material: material(),
            mesh: None,
        };
        let toward = Vec3::new(3.0, 0.5, -4.0);
        let pose = piece_pose(Piece { part, index: 0 }, 0, 0.0, STEP, toward, -1.0);
        // The unit block's two end faces, carried by the pose, land on the
        // origin and on the source.
        let matrix = pose.to_matrix();
        let near = matrix.transform_point3(Vec3::new(0.0, 0.0, -0.5));
        let far = matrix.transform_point3(Vec3::new(0.0, 0.0, 0.5));
        assert!(near.length() < 1e-4, "one end at the effect: {near:?}");
        assert!((far - toward).length() < 1e-4, "the other at its source: {far:?}");
    }

    fn slash(arc_degrees: f32) -> BuiltPart {
        BuiltPart {
            kind: Kind::Slash {
                radius: 1.5,
                arc: arc_degrees.to_radians(),
                width: 0.2,
                tilt: 0.0,
            },
            ticks: 30,
            material: material(),
            mesh: None,
        }
    }

    /// Played from a swinger 1.4 m behind the hit centre, facing -Z.
    const SWINGER: Vec3 = Vec3::new(0.0, 0.0, 1.4);

    #[test]
    fn a_slash_sweeps_its_segments_in_order_across_the_blows_path() {
        let part = slash(120.0);
        let segments = game_core::content::SLASH_SEGMENTS;
        let shown = |age: f32| {
            (0..segments)
                .filter(|&index| {
                    piece_pose(Piece { part, index }, 0, age, STEP, SWINGER, -1.0)
                        .scale
                        .y
                        > 0.0
                })
                .count()
        };
        assert_eq!(shown(0.0), 0, "nothing before the sweep begins");
        let early = shown(3.0);
        let later = shown(8.0);
        assert!(
            early > 0 && early < later,
            "segments appear one after another: {early} then {later}"
        );
        assert_eq!(
            shown(10.5),
            usize::from(segments),
            "all of them by the end of the sweep"
        );
        assert_eq!(shown(30.0), 0, "and all thinned away by the end");

        for index in 0..segments {
            let pose = piece_pose(Piece { part, index }, 0, 20.0, STEP, SWINGER, -1.0);
            let from_swinger = pose.translation - SWINGER;
            assert!(
                (from_swinger.length() - 1.5).abs() < 1e-3,
                "segment {index} on the arc's radius: {}",
                from_swinger.length()
            );
            assert!(
                from_swinger.z < 0.0,
                "and in front of the swinger: {from_swinger:?}"
            );
        }
    }

    #[test]
    fn a_slash_with_nowhere_to_face_is_not_drawn() {
        let part = slash(90.0);
        let pose = piece_pose(Piece { part, index: 3 }, 0, 20.0, STEP, Vec3::ZERO, -1.0);
        assert_eq!(pose.scale, Vec3::ZERO);
    }

    #[test]
    fn cracks_race_out_along_the_ground_and_close_again() {
        let part = BuiltPart {
            kind: Kind::Crack {
                count: 8,
                length: 3.0,
                width: 0.08,
            },
            ticks: 60,
            material: material(),
            mesh: None,
        };
        let floor = -0.9;
        for index in 0..8 {
            let piece = Piece { part, index };
            let early = piece_pose(piece, 4, 1.0, STEP, Vec3::ZERO, floor);
            let open = piece_pose(piece, 4, 20.0, STEP, Vec3::ZERO, floor);
            let late = piece_pose(piece, 4, 60.0, STEP, Vec3::ZERO, floor);
            assert!(
                (open.translation.y - (floor + CRACK_LIFT)).abs() < 1e-5,
                "on the ground"
            );
            assert!(
                early.scale.x < open.scale.x,
                "racing out: {} then {}",
                early.scale.x,
                open.scale.x
            );
            assert!(open.scale.x >= 3.0 * 0.6 - 1e-3, "most of its length once grown");
            // One end stays at the centre: a fissure splits out from the blow.
            let direction = open.rotation * Vec3::X;
            let near = open.translation - direction * open.scale.x * 0.5;
            assert!(
                Vec3::new(near.x, 0.0, near.z).length() < 1e-3,
                "fissure {index} starts at the centre: {near:?}"
            );
            assert!(late.scale.z.abs() < 1e-5, "and has closed by the end");
        }
    }

    fn eruption() -> BuiltPart {
        BuiltPart {
            kind: Kind::Spire {
                count: 9,
                height: 2.5,
                width: 0.9,
                spread: 4.5,
            },
            ticks: 100,
            material: material(),
            mesh: None,
        }
    }

    /// The highest point of a posed spire: the tip of a one-metre spike,
    /// wherever its lean has put it.
    fn tip(pose: &Transform) -> f32 {
        (pose.translation + pose.rotation * (Vec3::Y * pose.scale.y)).y
    }

    #[test]
    fn spires_start_below_the_ground_burst_up_and_sink_back() {
        let part = eruption();
        let floor = -0.9;
        for index in 0..9 {
            let piece = Piece { part, index };
            let at = |age: f32| piece_pose(piece, 3, age, STEP, Vec3::ZERO, floor);
            assert!(
                tip(&at(0.0)) <= floor + 1e-4,
                "spire {index} starts wholly underground"
            );
            let up = at(50.0);
            assert!(
                tip(&up) > floor + 2.5 * 0.55 * 0.8,
                "spire {index} stands well clear of the ground once up: tip at {}",
                tip(&up)
            );
            assert!(
                (up.translation.y - floor).abs() < 1e-3,
                "and its base is at the ground, not floating: {}",
                up.translation.y
            );
            assert!(
                tip(&at(100.0)) <= floor + 1e-3,
                "spire {index} has sunk back by the end"
            );
            let out = Vec3::new(up.translation.x, 0.0, up.translation.z).length();
            assert!(out <= 4.5 + 1e-3, "within the spread: {out}");
        }
    }

    /// Up as a wave: the spires nearest the centre break the ground first.
    #[test]
    fn spires_rise_nearest_first() {
        let part = eruption();
        let floor = -0.9;
        let mut by_distance: Vec<(f32, f32)> = (0..9)
            .map(|index| {
                let piece = Piece { part, index };
                let up = piece_pose(piece, 3, 50.0, STEP, Vec3::ZERO, floor);
                let out = Vec3::new(up.translation.x, 0.0, up.translation.z).length();
                // The first age at which it has broken the surface.
                let broke = (0..100)
                    .find(|&age| {
                        tip(&piece_pose(piece, 3, age as f32, STEP, Vec3::ZERO, floor)) > floor + 0.05
                    })
                    .map_or(f32::MAX, |age| age as f32);
                (out, broke)
            })
            .collect();
        by_distance.sort_by(|a, b| a.0.total_cmp(&b.0));
        for pair in by_distance.windows(2) {
            assert!(
                pair[0].1 <= pair[1].1,
                "a nearer spire breaks the ground no later than a farther one: {by_distance:?}"
            );
        }
    }

    #[test]
    fn a_spire_mesh_stands_on_its_base_with_a_glowing_tip() {
        let rock = [0.2, 0.2, 0.2];
        let glow = [1.0, 0.6, 0.1];
        let mesh = spire_mesh(rock, glow);
        let bounds = mesh.bounds().expect("drawn");
        assert!(
            bounds.min[1].abs() < 1e-3,
            "its base is at the origin: {bounds:?}"
        );
        assert!((bounds.max[1] - 1.0).abs() < 0.02, "a metre tall: {bounds:?}");
        let highest = mesh
            .positions
            .iter()
            .zip(&mesh.colors)
            .max_by(|a, b| a.0[1].total_cmp(&b.0[1]))
            .map(|(_, color)| *color)
            .expect("vertices");
        assert_eq!(highest, glow, "the tip is the molten colour");
        assert!(mesh.colors.contains(&rock), "and the body is rock");
        assert!(mesh.triangle_count() < 40, "low-poly: {}", mesh.triangle_count());
    }

    #[test]
    fn a_warning_follows_the_windows() {
        assert_eq!(warning_progress(AttackPhase::Windup, 0.25), Some((0.25, false)));
        assert_eq!(warning_progress(AttackPhase::Active, 0.1), Some((1.0, true)));
        assert_eq!(warning_progress(AttackPhase::Recovery, 0.5), None);
        assert_eq!(warning_progress(AttackPhase::Finished, 1.0), None);
    }
}
