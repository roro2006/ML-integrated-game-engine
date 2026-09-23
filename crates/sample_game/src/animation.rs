//! Playing animation clips on modelled bodies (ADR-0017 §5, M7 wave 3).
//!
//! Each frame, every body that wears a model picks the state it is in — dodging,
//! swinging, casting, walking, or standing — reads that state's clock from the
//! gameplay that owns it, samples the state's clip at that time for each of its
//! slots, and writes the result to its parts' poses. A change of state
//! crossfades over a few ticks rather than snapping.
//!
//! Nothing here writes anything the simulation reads. Poses belong to parts,
//! and parts are drawn and nothing else; the body's translation — its hitbox —
//! is read, for walking, and never written. A test holds that line.

use content_runtime::ContentRegistry;
use engine::ecs::{Component, Entity, World};
use engine::math::{Quat, Transform, Vec3};
use engine::render::Model;
use game_core::content::{ClipDef, ClipTiming, Fixed, KeyDef, TrackDef};
use game_runtime::combat::Attack;
use game_runtime::movement::Dodge;

use crate::slice::{self, Casting, ModelLibrary, Wears};

/// How long a change of state blends, in ticks: a tenth of a second at 60 Hz,
/// long enough to hide a snap and short enough that a wind-up still starts
/// when its attack does.
pub(crate) const BLEND_TICKS: u64 = 6;

/// How far a body must move in one tick to count as walking, in metres.
const WALKING: f32 = 0.004;

/// What a body is doing, for choosing a clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Act {
    Idle,
    Walk,
    Swing,
    Cast,
    Dodge,
}

/// Where a body's animation stands.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Animator {
    act: Act,
    /// The tick the current act began, for the crossfade.
    since: u64,
    /// Each part's pose when the current act began, to blend from.
    from: Vec<Transform>,
    /// How far through its walk cycle the body is, `0..1`.
    stride: f32,
    /// Where the body was, and at which tick, when last looked at.
    last_position: Vec3,
    last_tick: u64,
    /// Whether the body moved over the last tick that passed.
    moving: bool,
}

impl Component for Animator {}

fn metres(value: Fixed) -> f32 {
    value.as_milli() as f32 / 1000.0
}

/// A key as a transform: its turn, offset and scale.
fn key_pose(key: &KeyDef) -> (Vec3, Vec3, Vec3) {
    let turn = Vec3::new(metres(key.turn.x), metres(key.turn.y), metres(key.turn.z));
    let offset = Vec3::new(metres(key.offset.x), metres(key.offset.y), metres(key.offset.z));
    let scale = Vec3::new(metres(key.scale.x), metres(key.scale.y), metres(key.scale.z));
    (turn, offset, scale)
}

fn compose((turn, offset, scale): (Vec3, Vec3, Vec3)) -> Transform {
    let rotation = Quat::from_rotation_z(turn.z.to_radians())
        * Quat::from_rotation_y(turn.y.to_radians())
        * Quat::from_rotation_x(turn.x.to_radians());
    Transform {
        translation: offset,
        rotation,
        scale,
    }
}

/// The slope a monotone curve leaves a key with, given the slopes of the
/// segments either side of it: none at a peak, a trough or a hold, so the
/// curve never overshoots a key or wobbles through a pose held still; the
/// harmonic mean of the two otherwise, which leans toward the gentler side.
fn slope(before: f32, after: f32) -> f32 {
    if before * after <= 0.0 {
        0.0
    } else {
        2.0 * before * after / (before + after)
    }
}

/// A monotone cubic from `(t1, y1)` to `(t2, y2)` at `t`, with `(t0, y0)`
/// before and `(t3, y3)` after (Fritsch–Carlson). Neighbours at the same time
/// as the key they flank — the ends of a one-shot clip — read as flat, so a
/// clip eases into its first key and out of its last.
fn monotone(times: [f32; 4], values: [f32; 4], t: f32) -> f32 {
    let delta = |i: usize| {
        let span = times[i + 1] - times[i];
        if span > 1e-6 {
            (values[i + 1] - values[i]) / span
        } else {
            0.0
        }
    };
    let (d01, d12, d23) = (delta(0), delta(1), delta(2));
    let span = times[2] - times[1];
    if span <= 1e-6 {
        return values[2];
    }
    let (m1, m2) = if d12 == 0.0 {
        (0.0, 0.0)
    } else {
        (slope(d01, d12), slope(d12, d23))
    };
    let s = ((t - times[1]) / span).clamp(0.0, 1.0);
    let (s2, s3) = (s * s, s * s * s);
    (2.0 * s3 - 3.0 * s2 + 1.0) * values[1]
        + (s3 - 2.0 * s2 + s) * span * m1
        + (-2.0 * s3 + 3.0 * s2) * values[2]
        + (s3 - s2) * span * m2
}

/// A slot's pose at `time` on its clip's clock, in milli-units.
///
/// Between keys, turn, offset and scale follow a monotone cubic curve through
/// the keys — turns as degrees, component by component, which is what an
/// author wrote. The curve passes exactly through every key but carries the
/// motion through it, so a swing that coils, strikes and follows through
/// reads as one movement where straight lines between the same keys read as
/// three. It never overshoots a key, so a pose held for a while stays held.
/// Outside the keys a one-shot clip holds its first or last key; a cycling
/// clip wraps from its last key round to its first, and curves across the
/// wrap too.
pub(crate) fn sample(track: &TrackDef, time: f32, cycles: bool, end: f32) -> Transform {
    let keys = &track.keys;
    let Some(first) = keys.first() else {
        return Transform::IDENTITY;
    };
    let last = keys.last().unwrap_or(first);
    let count = keys.len() as isize;
    let at = |key: &KeyDef| key.at.as_milli() as f32;
    // Key `index` and its time: wrapped round a cycle — shifted by whole
    // cycles so times stay in order — and held at the ends of a one-shot.
    let key = |index: isize| -> (f32, (Vec3, Vec3, Vec3)) {
        if cycles {
            let lap = index.div_euclid(count) as f32;
            let key = &keys[index.rem_euclid(count) as usize];
            (at(key) + lap * end, key_pose(key))
        } else {
            let key = &keys[index.clamp(0, count - 1) as usize];
            (at(key), key_pose(key))
        }
    };
    // The curve from key `index` to the next, at `time` on the same clock.
    let curve = |index: isize, time: f32| {
        let points = [key(index - 1), key(index), key(index + 1), key(index + 2)];
        let times = points.map(|(time, _)| time);
        let axis = |pick: &dyn Fn(&(Vec3, Vec3, Vec3)) -> Vec3| {
            let values = points.map(|(_, pose)| pick(&pose));
            Vec3::new(
                monotone(times, values.map(|v| v.x), time),
                monotone(times, values.map(|v| v.y), time),
                monotone(times, values.map(|v| v.z), time),
            )
        };
        compose((axis(&|pose| pose.0), axis(&|pose| pose.1), axis(&|pose| pose.2)))
    };

    if time < at(first) || time >= at(last) {
        if !cycles || count == 1 {
            return compose(key_pose(if time < at(first) { first } else { last }));
        }
        // Wrapping: across the end of the cycle, from the last key to the
        // first as it comes round again.
        let time = if time >= at(last) { time } else { time + end };
        return curve(count - 1, time);
    }
    for (index, pair) in keys.windows(2).enumerate() {
        if time >= at(&pair[0]) && time < at(&pair[1]) {
            return curve(index as isize, time);
        }
    }
    compose(key_pose(last))
}

/// A whole clip at `time`, one pose per slot in `slots`' order; slots the clip
/// has no track for rest.
pub(crate) fn sample_clip(clip: &ClipDef, slots: &[game_core::ids::SlotId], time: f32) -> Vec<Transform> {
    let cycles = matches!(clip.timing, ClipTiming::Loop { .. } | ClipTiming::Stride { .. });
    let end = clip.timing.end() as f32;
    slots
        .iter()
        .map(|slot| {
            clip.tracks
                .iter()
                .find(|track| &track.slot == slot)
                .map_or(Transform::IDENTITY, |track| sample(track, time, cycles, end))
        })
        .collect()
}

/// Where a committed action is on a `Windows` clock: the window it is in, plus
/// how far through that window, in milli-units — wind-up `0..1000`, active
/// `1000..2000`, recovery `2000..3000`. Gameplay's windows set the pace, so a
/// slower attack plays a slower animation and nothing here can change that.
pub(crate) fn windows_clock(timing: game_runtime::combat::AttackTiming, elapsed: u32) -> f32 {
    use game_runtime::combat::AttackPhase;
    let (phase, progress) = slice::phase_progress(timing, elapsed);
    let window = match phase {
        AttackPhase::Windup => 0.0,
        AttackPhase::Active => 1.0,
        AttackPhase::Recovery => 2.0,
        AttackPhase::Finished => return 3000.0,
    };
    (window + progress.clamp(0.0, 1.0)) * 1000.0
}

/// Blends two poses: `t` of the way from `from` to `to`.
fn mix(from: &Transform, to: &Transform, t: f32) -> Transform {
    Transform {
        translation: from.translation.lerp(to.translation, t),
        rotation: from.rotation.slerp(to.rotation, t),
        scale: from.scale.lerp(to.scale, t),
    }
}

/// Poses every modelled body's parts for this frame.
pub(crate) fn animate(world: &mut World, content: &ContentRegistry, tick: u64) {
    world.resource_scope::<ModelLibrary, _>(|world, library| {
        let bodies: Vec<(Entity, game_core::ids::ModelId)> = world
            .query::<Wears>()
            .map(|(entity, wears)| (entity, wears.0.clone()))
            .collect();
        for (entity, model) in bodies {
            let Some(built) = library.get(&model) else {
                continue;
            };
            let Some(position) = world.get::<Transform>(entity).map(|t| t.translation) else {
                continue;
            };

            let mut animator = world.get::<Animator>(entity).cloned().unwrap_or(Animator {
                act: Act::Idle,
                since: tick,
                from: vec![Transform::IDENTITY; built.parts.len()],
                stride: 0.0,
                last_position: position,
                last_tick: tick,
                moving: false,
            });

            // Motion is judged per simulation tick, not per frame: a frame on
            // which no tick ran moved nothing, and reading that as "stopped"
            // would flicker a walking body between walk and idle.
            if tick != animator.last_tick {
                let travelled = Vec3::new(
                    position.x - animator.last_position.x,
                    0.0,
                    position.z - animator.last_position.z,
                )
                .length();
                let ticks = tick.saturating_sub(animator.last_tick).max(1) as f32;
                animator.moving = travelled / ticks > WALKING;
                if let Some(ClipTiming::Stride { metres: stride }) = built
                    .animations
                    .walk
                    .as_ref()
                    .and_then(|id| content.clips.get(id))
                    .map(|clip| clip.timing)
                {
                    animator.stride = (animator.stride + travelled / metres(stride).max(0.1)).fract();
                }
                animator.last_position = position;
                animator.last_tick = tick;
            }

            let (act, clock) = if let Some(dodge) = world.get::<Dodge>(entity) {
                (
                    Act::Dodge,
                    dodge.elapsed() as f32 / dodge.duration.max(1) as f32 * 1000.0,
                )
            } else if let Some(attack) = world.get::<Attack>(entity) {
                (Act::Swing, windows_clock(attack.timing, attack.elapsed()))
            } else if let Some(casting) = world.get::<Casting>(entity) {
                (Act::Cast, windows_clock(casting.timing, casting.elapsed))
            } else if animator.moving {
                (Act::Walk, animator.stride * 1000.0)
            } else {
                let period = built
                    .animations
                    .idle
                    .as_ref()
                    .and_then(|id| content.clips.get(id))
                    .and_then(|clip| match clip.timing {
                        ClipTiming::Loop { ticks } => Some(u64::from(ticks.max(1))),
                        _ => None,
                    })
                    .unwrap_or(1);
                (Act::Idle, (tick % period) as f32 / period as f32 * 1000.0)
            };

            // An ability may bring its own clip (ADR-0017 §8), played in place
            // of the body's cast; it runs on the same windows either way.
            let own = world
                .get::<Casting>(entity)
                .and_then(|casting| content.abilities.get(&casting.ability))
                .and_then(|ability| ability.presentation.clip.as_ref());
            let clip = match act {
                Act::Idle => built.animations.idle.as_ref(),
                Act::Walk => built.animations.walk.as_ref(),
                Act::Swing => built.animations.swing.as_ref(),
                Act::Cast => own.or(built.animations.cast.as_ref()),
                Act::Dodge => built.animations.dodge.as_ref(),
            }
            .and_then(|id| content.clips.get(id));
            let target = clip.map_or_else(
                || vec![Transform::IDENTITY; built.slots.len()],
                |clip| sample_clip(clip, &built.slots, clock),
            );

            let Some(model) = world.get_mut::<Model>(entity) else {
                continue;
            };
            if act != animator.act {
                animator.from = model.parts.iter().map(|part| part.pose).collect();
                animator.act = act;
                animator.since = tick;
            }
            let blend = (tick.saturating_sub(animator.since) as f32 / BLEND_TICKS as f32).min(1.0);
            for (index, part) in model.parts.iter_mut().enumerate() {
                let to = target.get(index).copied().unwrap_or(Transform::IDENTITY);
                let from = animator.from.get(index).copied().unwrap_or(Transform::IDENTITY);
                part.pose = mix(&from, &to, blend);
            }
            let _ = world.insert(entity, animator);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::content::Point;

    fn key(at: i64, x_degrees: i64) -> KeyDef {
        KeyDef {
            at: Fixed::from_milli(at),
            turn: Point::new(Fixed::from_whole(x_degrees), Fixed::ZERO, Fixed::ZERO),
            offset: Point::default(),
            scale: Point::new(Fixed::from_whole(1), Fixed::from_whole(1), Fixed::from_whole(1)),
        }
    }

    fn track(keys: Vec<KeyDef>) -> TrackDef {
        TrackDef {
            slot: game_core::ids::SlotId::new("arm-r").expect("valid id"),
            keys,
        }
    }

    /// The X turn a sampled pose carries, in degrees, measured the way a limb
    /// shows it: a limb hanging along -Y, turned by θ about X, points along
    /// (0, -cos θ, -sin θ).
    fn x_degrees(pose: &Transform) -> f32 {
        let limb = pose.rotation * Vec3::NEG_Y;
        (-limb.z).atan2(-limb.y).to_degrees()
    }

    #[test]
    fn keys_are_hit_exactly_and_interpolated_between() {
        let arm = track(vec![key(0, 0), key(1000, 90), key(2000, 10)]);
        assert!((x_degrees(&sample(&arm, 0.0, false, 3000.0)) - 0.0).abs() < 1e-3);
        assert!((x_degrees(&sample(&arm, 1000.0, false, 3000.0)) - 90.0).abs() < 1e-3);
        assert!((x_degrees(&sample(&arm, 500.0, false, 3000.0)) - 45.0).abs() < 1e-3);
        assert!((x_degrees(&sample(&arm, 1500.0, false, 3000.0)) - 50.0).abs() < 1e-3);
    }

    /// A held pose stays held: the curve never swings past a key, so an arm
    /// raised and held does not drift beyond where it was put.
    #[test]
    fn a_held_pose_does_not_overshoot() {
        let arm = track(vec![key(0, 0), key(900, 150), key(1000, 150), key(1300, 35)]);
        for step in 900..=1000 {
            let held = x_degrees(&sample(&arm, step as f32, false, 3000.0));
            assert!(
                (held - 150.0).abs() < 1e-2,
                "held at 150 through the hold: {held} at {step}"
            );
        }
        for step in 0..=1300 {
            let angle = x_degrees(&sample(&arm, step as f32, false, 3000.0));
            assert!(
                angle <= 150.0 + 1e-2,
                "never past the highest key: {angle} at {step}"
            );
        }
        // And a sharp peak — a fast raise and a slow fall — is the top of the
        // motion, not a point the curve sails over.
        let snap = track(vec![key(0, 0), key(100, 150), key(1000, 0)]);
        for step in 0..=1000 {
            let angle = x_degrees(&sample(&snap, step as f32, false, 3000.0));
            assert!(angle <= 150.0 + 1e-2, "past the peak: {angle} at {step}");
        }
    }

    /// Motion carries through a key rather than changing speed abruptly on
    /// it, which is what makes a swing read as one movement.
    #[test]
    fn motion_carries_through_a_key() {
        let arm = track(vec![key(0, 0), key(500, 20), key(1000, 100)]);
        let at = |time: f32| x_degrees(&sample(&arm, time, false, 3000.0));
        let before = (at(500.0) - at(490.0)) / 10.0;
        let after = (at(510.0) - at(500.0)) / 10.0;
        assert!(before > 0.0 && after > 0.0, "still moving through the key");
        assert!(
            (after - before).abs() / after < 0.15,
            "its pace barely changes across the key: {before} then {after} degrees a milli-unit"
        );
        assert!(
            (at(500.0) - 20.0).abs() < 1e-3,
            "and it still passes exactly through the key"
        );
    }

    #[test]
    fn a_one_shot_clip_holds_its_ends() {
        let arm = track(vec![key(200, 30), key(800, 60)]);
        assert!((x_degrees(&sample(&arm, 0.0, false, 1000.0)) - 30.0).abs() < 1e-3);
        assert!((x_degrees(&sample(&arm, 1000.0, false, 1000.0)) - 60.0).abs() < 1e-3);
    }

    /// A walk has no seam: past its last key it heads back toward its first.
    #[test]
    fn a_cycling_clip_wraps_from_its_last_key_to_its_first() {
        let leg = track(vec![key(0, 20), key(500, -20)]);
        let at_seam = x_degrees(&sample(&leg, 750.0, true, 1000.0));
        assert!(
            at_seam.abs() < 1e-3,
            "halfway from -20 back round to 20 is 0, got {at_seam}"
        );
        let late = x_degrees(&sample(&leg, 999.0, true, 1000.0));
        assert!(
            (late - 20.0).abs() < 0.2,
            "and it arrives back at the first key: {late}"
        );
    }

    /// Gameplay owns time. The same moment of an attack's windows is the same
    /// pose however long those windows are, so retiming an attack in content
    /// retimes its animation and nothing in a clip can do the reverse.
    #[test]
    fn the_windows_clock_follows_the_attack_not_the_clip() {
        use game_runtime::combat::AttackTiming;
        let quick = AttackTiming {
            windup: 10,
            active: 4,
            recovery: 10,
        };
        let slow = AttackTiming {
            windup: 40,
            active: 16,
            recovery: 40,
        };
        assert_eq!(
            windows_clock(quick, 5),
            windows_clock(slow, 20),
            "halfway through each wind-up"
        );
        assert_eq!(
            windows_clock(quick, 10),
            1000.0,
            "the active window starts at 1000"
        );
        assert_eq!(windows_clock(slow, 56), 2000.0, "and recovery at 2000");
        assert_eq!(
            windows_clock(slow, 500),
            3000.0,
            "a finished action rests at the end"
        );
    }
}
