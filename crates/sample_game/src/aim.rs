//! Aiming with the mouse: where the player points, and how a key becomes a
//! cast.
//!
//! The cursor is projected onto the ground at the start of every fixed tick
//! and kept as [`Aim`], an input like any held key. Actions turn to it: a swing
//! or a cast faces the aim the tick it starts and keeps turning to follow it
//! through its wind-up, then locks the tick the blow is live or the effects
//! land. Walking is untouched — the body faces where it moves until it acts.
//!
//! # Cast modes
//!
//! How an ability key turns into a cast is a player preference, cycled with
//! [`CYCLE_KEY`], after the three modes players of lane games expect:
//!
//! | Mode | Press | Then |
//! |---|---|---|
//! | [`CastMode::QuickWithIndicator`], the default | shows the indicator while held | release casts; right-click cancels |
//! | [`CastMode::Normal`] | shows the indicator | left-click casts; right-click or the key again cancels |
//! | [`CastMode::Quick`] | casts at once | — |
//!
//! The indicator and the reticle under the cursor show only while an ability
//! is being aimed: the rest of the time the screen is the fight.
//!
//! The indicator is the ability's own telegraph shape — a lane, a circle, its
//! reach — drawn in the aiming colour, so what you aim with is what the world
//! will be warned with. The rules are one pure function, [`cast_input`], so
//! every mode is testable without a window.

use engine::camera::{Camera, Viewport};
use engine::ecs::{Entity, Resource, World};
use engine::input::{InputState, KeyCode, MouseButton};
use engine::math::Vec3;
use engine::time::FixedTime;
use game_runtime::combat::{Attack, AttackPhase};

use crate::slice::{Casting, Facing};

/// The ability keys, in loadout order: the first authored ability binds to the
/// first key. A table rather than a match chain so a fourth ability in
/// `classes.ron` binds with no Rust change.
pub(crate) const ABILITY_KEYS: [KeyCode; 4] = [KeyCode::Q, KeyCode::R, KeyCode::F, KeyCode::C];

/// Cycles the cast mode.
pub(crate) const CYCLE_KEY: KeyCode = KeyCode::F2;

/// Closer than this, the cursor is on top of the body and has no direction.
const DEAD_ZONE: f32 = 0.05;

/// Where the player is aiming, on the ground.
///
/// Kept, not cleared, when the cursor leaves the window or the view cannot
/// see the ground: the last aim stands, as it would for anyone who lifted
/// their hand off the mouse.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Aim {
    pub point: Option<Vec3>,
}

impl Resource for Aim {}

/// How an ability key becomes a cast. See the module docs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum CastMode {
    /// Hold to aim, release to cast: the default, chosen by the project
    /// owner so every cast is seen before it is thrown.
    #[default]
    QuickWithIndicator,
    Normal,
    Quick,
}

impl CastMode {
    pub(crate) fn next(self) -> Self {
        match self {
            Self::QuickWithIndicator => Self::Normal,
            Self::Normal => Self::Quick,
            Self::Quick => Self::QuickWithIndicator,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Quick => "quick cast",
            Self::QuickWithIndicator => "quick cast with indicator",
            Self::Normal => "normal cast",
        }
    }
}

/// The player's cast mode, and what their ability keys are in the middle of.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CastInput {
    pub mode: CastMode,
    /// The loadout slot being aimed, before it is cast. Only the indicator
    /// modes ever set it.
    pub aiming: Option<usize>,
    /// The tick a left-click confirmed a cast, so the same click does not
    /// also swing.
    pub confirmed_at: Option<u64>,
}

impl Resource for CastInput {}

/// This tick's buttons, as the cast rules need them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Buttons {
    /// The first ability key pressed this tick, by loadout slot.
    pub pressed: Option<usize>,
    /// Which ability keys went up this tick, by slot.
    pub released: [bool; ABILITY_KEYS.len()],
    pub confirm: bool,
    pub cancel: bool,
}

impl Buttons {
    pub(crate) fn read(input: &InputState) -> Self {
        let mut released = [false; ABILITY_KEYS.len()];
        for (slot, key) in ABILITY_KEYS.iter().enumerate() {
            released[slot] = input.keys.just_released(*key);
        }
        Self {
            pressed: ABILITY_KEYS.iter().position(|key| input.keys.just_pressed(*key)),
            released,
            confirm: input.mouse_buttons.just_pressed(MouseButton::Left),
            cancel: input.mouse_buttons.just_pressed(MouseButton::Right),
        }
    }
}

/// What the ability keys ask for this tick.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Intent {
    /// Cast this loadout slot now.
    pub cast: Option<usize>,
    /// The slot being aimed after this tick.
    pub aiming: Option<usize>,
    /// A left-click was spent confirming a cast, and must not also swing.
    pub consumed_click: bool,
}

/// The cast rules, as one pure function of the mode, what was being aimed,
/// and this tick's buttons.
pub(crate) fn cast_input(mode: CastMode, aiming: Option<usize>, buttons: Buttons) -> Intent {
    let hold = Intent {
        aiming,
        ..Intent::default()
    };
    match mode {
        CastMode::Quick => Intent {
            cast: buttons.pressed,
            ..Intent::default()
        },
        CastMode::QuickWithIndicator => {
            if buttons.cancel && aiming.is_some() {
                return Intent::default();
            }
            if let Some(slot) = buttons.pressed {
                // Pressed and let go inside one tick is a tap: cast it.
                if buttons.released[slot] {
                    return Intent {
                        cast: Some(slot),
                        ..Intent::default()
                    };
                }
                return Intent {
                    aiming: Some(slot),
                    ..Intent::default()
                };
            }
            match aiming {
                Some(slot) if buttons.released[slot] => Intent {
                    cast: Some(slot),
                    ..Intent::default()
                },
                _ => hold,
            }
        }
        CastMode::Normal => {
            if buttons.cancel && aiming.is_some() {
                return Intent::default();
            }
            if let Some(slot) = buttons.pressed {
                // The same key again puts the ability away; another key aims
                // that one instead.
                return Intent {
                    aiming: (aiming != Some(slot)).then_some(slot),
                    ..Intent::default()
                };
            }
            match aiming {
                Some(slot) if buttons.confirm => Intent {
                    cast: Some(slot),
                    aiming: None,
                    consumed_click: true,
                },
                _ => hold,
            }
        }
    }
}

/// Projects the cursor onto the ground, if there is a cursor, a window and a
/// camera to project it through.
pub(crate) fn update_aim(world: &mut World) {
    let point = {
        let (Some(input), Some(viewport), Some(camera)) = (
            world.resource::<engine::input::FixedInput>(),
            world.resource::<Viewport>(),
            world.resource::<Camera>(),
        ) else {
            return;
        };
        let Some(cursor) = input.cursor_position else {
            return;
        };
        camera.ground_point(*viewport, cursor, 0.0)
    };
    if let (Some(point), Some(aim)) = (point, world.resource_mut::<Aim>()) {
        aim.point = Some(point);
    }
}

/// Turns `entity` to face the aim, if there is one and it is not underfoot.
pub(crate) fn face_aim(world: &mut World, entity: Entity) {
    let Some(target) = world.resource::<Aim>().and_then(|aim| aim.point) else {
        return;
    };
    let Some(from) = world
        .get::<engine::math::Transform>(entity)
        .map(|t| t.translation)
    else {
        return;
    };
    let Some(direction) = Vec3::new(target.x - from.x, 0.0, target.z - from.z)
        .try_normalize()
        .filter(|_| Vec3::new(target.x - from.x, 0.0, target.z - from.z).length() > DEAD_ZONE)
    else {
        return;
    };
    if let Some(facing) = world.get_mut::<Facing>(entity) {
        facing.0 = direction;
    }
}

/// Whether `entity` is winding up — the window in which aim still steers.
fn winding_up(world: &World, entity: Entity) -> bool {
    world
        .get::<Attack>(entity)
        .is_some_and(|attack| attack.phase() == AttackPhase::Windup)
        || world
            .get::<Casting>(entity)
            .is_some_and(|casting| casting.phase() == AttackPhase::Windup)
}

/// Keeps a winding-up player turned to the aim, so a swing or a bolt goes
/// where the cursor is when it leaves, not where it was when it began. Once
/// the blow is live or the effects have landed, facing is left alone.
pub(crate) fn track_aim(world: &mut World, player: Entity) {
    if winding_up(world, player) {
        face_aim(world, player);
    }
}

/// Whether the player is committed to an action, during which walking does
/// not turn them.
pub(crate) fn acting(world: &World, entity: Entity) -> bool {
    world.get::<Attack>(entity).is_some() || world.get::<Casting>(entity).is_some()
}

/// The fixed-step index, for recording when a click was spent.
pub(crate) fn tick(world: &World) -> u64 {
    world.resource::<FixedTime>().map_or(0, |time| time.tick)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn press(slot: usize) -> Buttons {
        Buttons {
            pressed: Some(slot),
            ..Buttons::default()
        }
    }

    fn release(slot: usize) -> Buttons {
        let mut released = [false; ABILITY_KEYS.len()];
        released[slot] = true;
        Buttons {
            released,
            ..Buttons::default()
        }
    }

    fn click() -> Buttons {
        Buttons {
            confirm: true,
            ..Buttons::default()
        }
    }

    fn cancel() -> Buttons {
        Buttons {
            cancel: true,
            ..Buttons::default()
        }
    }

    #[test]
    fn quick_cast_casts_on_the_press() {
        let intent = cast_input(CastMode::Quick, None, press(2));
        assert_eq!(intent.cast, Some(2));
        assert_eq!(intent.aiming, None, "nothing lingers to aim");
        assert_eq!(
            cast_input(CastMode::Quick, None, click()),
            Intent::default(),
            "a click is a swing"
        );
    }

    #[test]
    fn quick_cast_with_indicator_aims_while_held_and_casts_on_release() {
        let mode = CastMode::QuickWithIndicator;
        let held = cast_input(mode, None, press(1));
        assert_eq!(
            (held.cast, held.aiming),
            (None, Some(1)),
            "held: aiming, not cast"
        );
        let still = cast_input(mode, held.aiming, Buttons::default());
        assert_eq!(
            (still.cast, still.aiming),
            (None, Some(1)),
            "still held: still aiming"
        );
        let let_go = cast_input(mode, still.aiming, release(1));
        assert_eq!((let_go.cast, let_go.aiming), (Some(1), None), "released: cast");

        // Releasing some other key does not cast this one.
        assert_eq!(cast_input(mode, Some(1), release(0)).cast, None);
        // Right-click puts it away uncast.
        assert_eq!(cast_input(mode, Some(1), cancel()), Intent::default());
        // A tap inside one tick is a cast, not a stuck indicator.
        let mut tap = press(0);
        tap.released[0] = true;
        assert_eq!(cast_input(mode, None, tap).cast, Some(0));
    }

    #[test]
    fn normal_cast_aims_on_the_press_and_casts_on_a_click() {
        let mode = CastMode::Normal;
        let aimed = cast_input(mode, None, press(2));
        assert_eq!((aimed.cast, aimed.aiming), (None, Some(2)));
        // Letting go of the key changes nothing: it is not held to aim.
        assert_eq!(cast_input(mode, Some(2), release(2)).aiming, Some(2));

        let confirmed = cast_input(mode, Some(2), click());
        assert_eq!(confirmed.cast, Some(2));
        assert_eq!(confirmed.aiming, None);
        assert!(
            confirmed.consumed_click,
            "the click that casts does not also swing"
        );

        assert_eq!(
            cast_input(mode, Some(2), press(2)).aiming,
            None,
            "the key again puts it away"
        );
        assert_eq!(
            cast_input(mode, Some(2), press(0)).aiming,
            Some(0),
            "another key aims that one"
        );
        assert_eq!(
            cast_input(mode, Some(2), cancel()),
            Intent::default(),
            "right-click cancels"
        );
        let idle_click = cast_input(mode, None, click());
        assert!(
            !idle_click.consumed_click,
            "with nothing aimed, a click is a swing"
        );
    }

    #[test]
    fn the_modes_cycle_through_all_three() {
        let mut mode = CastMode::default();
        assert_eq!(mode, CastMode::QuickWithIndicator, "hold to aim, release to cast");
        let mut seen = vec![mode];
        for _ in 0..3 {
            mode = mode.next();
            seen.push(mode);
        }
        assert_eq!(
            seen,
            vec![
                CastMode::QuickWithIndicator,
                CastMode::Normal,
                CastMode::Quick,
                CastMode::QuickWithIndicator
            ]
        );
    }
}
