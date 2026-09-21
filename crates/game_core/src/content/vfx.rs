//! Visual effects composed from a small allowlist (ADR-0017 §6).
//!
//! An effect is a few **parts**, each a primitive from a fixed set — a burst of
//! shards, a pulse along the ground, a beam — with a colour from a palette and
//! numbers inside bounds. Nothing here is geometry, a shader, or behaviour:
//! the runtime owns what each primitive looks like and how it moves, and a
//! definition only chooses among them. That is what lets an AI propose an
//! effect (ADR-0010) and validation bound its worst case.
//!
//! # Presentation only
//!
//! Nothing in the simulation reads an effect. An effect plays where gameplay
//! says something happened; it never decides whether it did. Stripping every
//! effect from a fight leaves its outcome identical, and a test holds that.
//!
//! # Where a part plays
//!
//! An effect is played from one point to another — from a caster to what its
//! ability touched, say. A burst plays at the far point, a pulse spreads on the
//! ground beneath it, and a beam spans the two. Played at a single point, the
//! two coincide and a beam has nothing to span, so it is skipped.

use serde::{Deserialize, Serialize};

use crate::ids::{PaletteId, SwatchId, VfxId};

use super::Fixed;

/// Most parts one effect may have.
pub const MAX_VFX_PARTS: usize = 4;
/// Most shards one burst may throw.
pub const MAX_BURST_SHARDS: u8 = 32;
/// Longest any part may last, in ticks: three seconds at 60 Hz. Nothing a
/// player needs to read takes longer, and a lingering effect is clutter.
pub const MAX_VFX_TICKS: u32 = 180;
/// A shard's size lies in `2 cm ..= 1 m`.
pub const MIN_SHARD_MILLI: i64 = 20;
pub const MAX_SHARD_MILLI: i64 = 1_000;
/// Fastest a shard may be thrown, in metres per second.
pub const MAX_SHARD_SPEED_METRES: i64 = 20;
/// Strongest pull on a shard, in metres per second squared: about four times
/// real gravity, which is already more than any shard should need to look
/// heavy.
pub const MAX_GRAVITY_METRES: i64 = 40;
/// A pulse's final radius lies in `10 cm ..= 8 m`.
pub const MIN_PULSE_MILLI: i64 = 100;
pub const MAX_PULSE_MILLI: i64 = 8_000;
/// A beam's width lies in `2 cm ..= 1 m`.
pub const MIN_BEAM_MILLI: i64 = 20;
pub const MAX_BEAM_MILLI: i64 = 1_000;
/// Pieces one slash is drawn with, whatever its arc.
pub const SLASH_SEGMENTS: u8 = 10;
/// A slash's radius lies in `30 cm ..= 6 m`.
pub const MIN_SLASH_RADIUS_MILLI: i64 = 300;
pub const MAX_SLASH_RADIUS_MILLI: i64 = 6_000;
/// A slash sweeps between 20 and 270 degrees.
pub const MIN_SLASH_ARC_DEGREES: i64 = 20;
pub const MAX_SLASH_ARC_DEGREES: i64 = 270;
/// Most fissures one crack may split into.
pub const MAX_CRACKS: u8 = 16;
/// A fissure's length lies in `20 cm ..= 8 m`.
pub const MIN_CRACK_MILLI: i64 = 200;
pub const MAX_CRACK_MILLI: i64 = 8_000;
/// Most rock spires one eruption may throw up.
pub const MAX_SPIRES: u8 = 16;
/// A spire's height lies in `30 cm ..= 6 m`.
pub const MIN_SPIRE_HEIGHT_MILLI: i64 = 300;
pub const MAX_SPIRE_HEIGHT_MILLI: i64 = 6_000;
/// A spire's base lies in `10 cm ..= 2 m` across.
pub const MIN_SPIRE_WIDTH_MILLI: i64 = 100;
pub const MAX_SPIRE_WIDTH_MILLI: i64 = 2_000;
/// Spires rise within `50 cm ..= 8 m` of where the effect plays.
pub const MIN_SPIRE_SPREAD_MILLI: i64 = 500;
pub const MAX_SPIRE_SPREAD_MILLI: i64 = 8_000;

/// One primitive in an effect.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum VfxPart {
    /// Chunky shards thrown up and out from where the effect plays: they
    /// tumble, fall under `gravity`, and shrink away over `ticks`.
    Burst {
        color: SwatchId,
        count: u8,
        /// Edge length of one shard, in metres.
        size: Fixed,
        /// How fast shards leave, in metres per second.
        speed: Fixed,
        /// Downward pull, in metres per second squared; absent is none, and
        /// shards drift outward like embers.
        #[serde(default)]
        gravity: Fixed,
        ticks: u32,
    },
    /// A flat ring on the ground, spreading from nothing to `radius` and
    /// thinning as it goes.
    Pulse {
        color: SwatchId,
        radius: Fixed,
        ticks: u32,
    },
    /// A bar from where the effect came from to where it plays, narrowing to
    /// nothing over `ticks`.
    Beam {
        color: SwatchId,
        width: Fixed,
        ticks: u32,
    },
    /// The path of a blow: an arc of `radius` around where the effect came
    /// from, centred on where it plays, sweeping across `arc` degrees in the
    /// first third of its life and then thinning away. `tilt` rolls the arc
    /// off the level, in degrees, so a slash can fall diagonally.
    Slash {
        color: SwatchId,
        radius: Fixed,
        arc: Fixed,
        width: Fixed,
        #[serde(default)]
        tilt: Fixed,
        ticks: u32,
    },
    /// The ground splitting: `count` fissures racing out from where the
    /// effect plays to about `length` metres, lying on the ground and closing
    /// again at the end.
    Crack {
        color: SwatchId,
        count: u8,
        length: Fixed,
        width: Fixed,
        ticks: u32,
    },
    /// The ground erupting: `count` jagged spires of rock punched up out of
    /// it within `spread` metres, nearest first like a shockwave, leaning away
    /// from the centre; they hold, then sink back into the ground. Solid
    /// geometry, not a drawing on the floor — the ground itself hides what is
    /// still below it. `glow` paints their tips, molten, if set.
    Spire {
        color: SwatchId,
        #[serde(default)]
        glow: Option<SwatchId>,
        count: u8,
        height: Fixed,
        width: Fixed,
        spread: Fixed,
        ticks: u32,
    },
}

impl VfxPart {
    /// The swatch this part is painted with.
    pub fn color(&self) -> &SwatchId {
        match self {
            Self::Burst { color, .. }
            | Self::Pulse { color, .. }
            | Self::Beam { color, .. }
            | Self::Slash { color, .. }
            | Self::Crack { color, .. }
            | Self::Spire { color, .. } => color,
        }
    }

    /// How long this part lasts, in ticks.
    pub fn ticks(&self) -> u32 {
        match self {
            Self::Burst { ticks, .. }
            | Self::Pulse { ticks, .. }
            | Self::Beam { ticks, .. }
            | Self::Slash { ticks, .. }
            | Self::Crack { ticks, .. }
            | Self::Spire { ticks, .. } => *ticks,
        }
    }

    /// Every swatch this part is painted with: its colour, and a spire's
    /// glow.
    pub fn swatches(&self) -> impl Iterator<Item = &SwatchId> {
        let glow = match self {
            Self::Spire { glow, .. } => glow.as_ref(),
            _ => None,
        };
        std::iter::once(self.color()).chain(glow)
    }

    /// How many pieces this part draws: what the runtime's particle cap
    /// counts.
    pub fn particles(&self) -> usize {
        match self {
            Self::Burst { count, .. } | Self::Crack { count, .. } | Self::Spire { count, .. } => {
                usize::from(*count)
            }
            Self::Slash { .. } => usize::from(SLASH_SEGMENTS),
            Self::Pulse { .. } | Self::Beam { .. } => 1,
        }
    }
}

/// A visual effect.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VfxDef {
    pub schema_version: u32,
    pub id: VfxId,
    pub name: String,
    pub palette: PaletteId,
    pub parts: Vec<VfxPart>,
}

impl VfxDef {
    /// Pieces the whole effect draws at once, at most.
    pub fn particles(&self) -> usize {
        self.parts.iter().map(VfxPart::particles).sum()
    }

    /// How long the effect lasts: its longest part.
    pub fn ticks(&self) -> u32 {
        self.parts.iter().map(VfxPart::ticks).max().unwrap_or(0)
    }
}
