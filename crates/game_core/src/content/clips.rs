//! Animation clips (ADR-0017 §5).
//!
//! A clip moves a body's slots — never its parts, never the body itself — by
//! keyframes of turn, offset and scale. Because a clip addresses slots by name,
//! one clip serves every model built on a body plan with those slots; a track
//! for a slot a body does not have is skipped rather than an error.
//!
//! # Gameplay owns time
//!
//! A clip never decides how long anything takes. Its keys are placed on a
//! clock that something else drives:
//!
//! - [`ClipTiming::Windows`]: the three windows of a committed action. Keys run
//!   from `0` to `3000`: `0..1000` is the wind-up, `1000..2000` the active
//!   window, `2000..3000` the recovery. An attack retimed in content is an
//!   animation retimed with it, and no clip can make an attack faster or
//!   slower (ADR-0007's windows stay the only authority).
//! - [`ClipTiming::Span`]: one action of known length, `0..1000` start to end —
//!   a dodge.
//! - [`ClipTiming::Loop`]: a repeating cycle of so many ticks — idling.
//! - [`ClipTiming::Stride`]: one cycle per so many metres travelled — walking,
//!   so the feet keep pace with the ground rather than sliding over it.
//!
//! And whatever a clip does, it does to *presentation*: a slot's pose is read
//! by the renderer and nothing else, so no clip can move a hitbox.

use serde::{Deserialize, Serialize};

use crate::ids::{ClipId, SlotId};

use super::{Fixed, Point};

/// Most tracks one clip may have.
pub const MAX_TRACKS: usize = 16;
/// Most keys one track may have.
pub const MAX_KEYS: usize = 16;
/// The end of a [`ClipTiming::Windows`] clip's clock, in milli-units.
pub const WINDOWS_END_MILLI: i64 = 3_000;
/// The end of every other clip's clock, in milli-units.
pub const CYCLE_END_MILLI: i64 = 1_000;
/// Longest loop, in ticks: a minute at 60 Hz.
pub const MAX_LOOP_TICKS: u32 = 3_600;
/// Shortest and longest stride, in milli-metres.
pub const MIN_STRIDE_MILLI: i64 = 100;
pub const MAX_STRIDE_MILLI: i64 = 10_000;
/// Furthest a key may offset a slot from rest, in metres.
pub const MAX_KEY_OFFSET_METRES: i64 = 1;

/// What drives a clip's clock. See the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum ClipTiming {
    Windows,
    Span,
    Loop { ticks: u32 },
    Stride { metres: Fixed },
}

impl ClipTiming {
    /// Where this clip's clock ends, in milli-units.
    pub fn end(&self) -> i64 {
        match self {
            Self::Windows => WINDOWS_END_MILLI,
            Self::Span | Self::Loop { .. } | Self::Stride { .. } => CYCLE_END_MILLI,
        }
    }
}

fn unit_scale() -> Point {
    let one = Fixed::from_whole(1);
    Point::new(one, one, one)
}

/// A slot's pose at one moment.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyDef {
    /// When, on the clip's clock (see [`ClipTiming`]), in milli-units.
    pub at: Fixed,
    /// Rotation about the slot's pivot, in degrees about X, then Y, then Z.
    #[serde(default)]
    pub turn: Point,
    /// Movement from rest, in metres. Presentation only: it moves the part,
    /// never the body's hitbox.
    #[serde(default)]
    pub offset: Point,
    /// Per-axis scale, in milli-units; absent is unscaled.
    #[serde(default = "unit_scale")]
    pub scale: Point,
}

/// Keys for one slot, in time order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrackDef {
    pub slot: SlotId,
    pub keys: Vec<KeyDef>,
}

/// An animation clip.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClipDef {
    pub schema_version: u32,
    pub id: ClipId,
    pub name: String,
    pub timing: ClipTiming,
    pub tracks: Vec<TrackDef>,
}

/// Which clip plays in each state a body can be in.
///
/// Every entry is optional: a state with no clip leaves the body's slots at
/// rest, and the procedural body pose — lean, roll, flinch — still plays on top
/// of whatever the slots are doing. A body plan names the defaults; a model may
/// override any of them.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnimationSet {
    /// Standing still. A `Loop` clip.
    #[serde(default)]
    pub idle: Option<ClipId>,
    /// Moving. A `Stride` clip.
    #[serde(default)]
    pub walk: Option<ClipId>,
    /// A committed attack, keyed across its windows. A `Windows` clip.
    #[serde(default)]
    pub swing: Option<ClipId>,
    /// A committed cast, keyed across its windows. A `Windows` clip.
    #[serde(default)]
    pub cast: Option<ClipId>,
    /// A dodge, start to end. A `Span` clip.
    #[serde(default)]
    pub dodge: Option<ClipId>,
}

impl AnimationSet {
    /// Every clip named, with the timing its state needs, in a fixed order.
    pub fn uses(&self) -> impl Iterator<Item = (&'static str, &ClipId)> {
        [
            ("idle", &self.idle),
            ("walk", &self.walk),
            ("swing", &self.swing),
            ("cast", &self.cast),
            ("dodge", &self.dodge),
        ]
        .into_iter()
        .filter_map(|(state, clip)| clip.as_ref().map(|clip| (state, clip)))
    }

    /// This set with any gaps filled from `defaults`.
    pub fn or(&self, defaults: &Self) -> Self {
        let pick = |own: &Option<ClipId>, fallback: &Option<ClipId>| own.clone().or_else(|| fallback.clone());
        Self {
            idle: pick(&self.idle, &defaults.idle),
            walk: pick(&self.walk, &defaults.walk),
            swing: pick(&self.swing, &defaults.swing),
            cast: pick(&self.cast, &defaults.cast),
            dodge: pick(&self.dodge, &defaults.dodge),
        }
    }
}

/// Whether a clip's timing suits the state it is used for: a walk keyed in
/// attack windows would play against a clock that never runs.
pub fn timing_suits(state: &str, timing: ClipTiming) -> bool {
    matches!(
        (state, timing),
        ("idle", ClipTiming::Loop { .. })
            | ("walk", ClipTiming::Stride { .. })
            | ("swing" | "cast", ClipTiming::Windows)
            | ("dodge", ClipTiming::Span)
    )
}
