//! Abilities and the allowlisted effect primitives.

use serde::{Deserialize, Serialize};

use crate::ids::{AbilityId, ClipId, ModelId, PrefabId, VfxId};

use super::*;

/// The allowlisted effect primitives (PROJECT_CONTEXT §7).
///
/// Extending this set is a deliberate act with tests, never a content change.
/// There is intentionally no "run script" variant: that is the difference
/// between content that can be generated safely and content that cannot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Effect {
    DealDamage {
        amount: Fixed,
        damage_type: DamageType,
    },
    ApplyStatus {
        status: StatusId,
        duration_ticks: u32,
    },
    SpawnEntity {
        prefab: PrefabId,
        count: u8,
    },
    ModifyResource {
        resource: ResourceType,
        amount: Fixed,
    },
    Teleport {
        max_distance: Fixed,
    },
    EmitEvent {
        event: GameEventId,
    },
    /// A skillshot (ADR-0017 §7): a body that flies from the caster along its
    /// aim, stopped by walls, and applies `on_hit` to what it strikes.
    ///
    /// Gameplay, not presentation: it flies in the fixed step and is resolved
    /// deterministically. Bounded in every dimension content can set — see the
    /// `PROJECTILE_*` limits — and `on_hit` is one level deep, holding only
    /// effects that act on what was struck ([`Effect::allowed_on_hit`]), so a
    /// projectile can never seed another.
    Projectile {
        /// Metres per second.
        speed: Fixed,
        /// How far it flies before it is spent, in metres.
        range: Fixed,
        /// How close it must pass to strike, in metres, beyond a body's own.
        radius: Fixed,
        /// How many targets it passes through before stopping; `0` stops at
        /// the first.
        #[serde(default)]
        pierce: u8,
        on_hit: Vec<Effect>,
    },
}

/// Slowest a projectile may fly, in metres per second: anything slower lives
/// long enough to be clutter rather than a threat.
pub const MIN_PROJECTILE_SPEED_METRES: i64 = 2;
/// Fastest a projectile may fly, in metres per second: half a metre a tick at
/// 60 Hz. The sweep resolves any speed exactly, so this is about being able
/// to see it coming — a faster shot is a hitscan in disguise.
pub const MAX_PROJECTILE_SPEED_METRES: i64 = 30;
/// Furthest a projectile may fly, in metres.
pub const MAX_PROJECTILE_RANGE_METRES: i64 = 20;
/// A projectile's strike radius lies in `5 cm ..= 1 m`: any wider and it is an
/// area effect that ignores the area target cap.
pub const MIN_PROJECTILE_RADIUS_MILLI: i64 = 50;
pub const MAX_PROJECTILE_RADIUS_MILLI: i64 = 1_000;
/// Most targets one projectile may pass through.
pub const MAX_PIERCE: u8 = 4;
/// Most effects one projectile may apply on a hit.
pub const MAX_ON_HIT_EFFECTS: usize = 4;

impl Effect {
    /// Whether this effect may run inside a projectile's `on_hit`.
    ///
    /// Only effects that act on what was struck. `Projectile` and
    /// `SpawnEntity` would let one cast seed an unbounded chain, and
    /// `Teleport` moves the caster, which a hit far away has no business doing.
    pub fn allowed_on_hit(&self) -> bool {
        matches!(
            self,
            Self::DealDamage { .. }
                | Self::ApplyStatus { .. }
                | Self::ModifyResource { .. }
                | Self::EmitEvent { .. }
        )
    }

    /// Whether this effect may run when an item is used (M9): only what acts
    /// on the user. Anything aimed at others would make a potion a weapon.
    pub fn allowed_on_use(&self) -> bool {
        matches!(self, Self::ModifyResource { .. } | Self::ApplyStatus { .. })
    }

    /// The variant's name, for error messages.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::DealDamage { .. } => "DealDamage",
            Self::ApplyStatus { .. } => "ApplyStatus",
            Self::SpawnEntity { .. } => "SpawnEntity",
            Self::ModifyResource { .. } => "ModifyResource",
            Self::Teleport { .. } => "Teleport",
            Self::EmitEvent { .. } => "EmitEvent",
            Self::Projectile { .. } => "Projectile",
        }
    }
}

/// Who an ability can be aimed at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Targeting {
    SelfOnly,
    SingleEnemy,
    SingleAlly,
    AreaAroundSelf,
    AreaAtPoint,
}

/// An ability composed from approved primitives.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbilityDef {
    pub schema_version: u32,
    pub id: AbilityId,
    pub name: String,
    pub targeting: Targeting,
    /// Stamina or equivalent spent on use.
    pub cost: Fixed,
    /// Cooldown in fixed simulation ticks, never seconds: ADR-0007's
    /// Souls-lite combat expresses every window in ticks so behaviour does not
    /// drift with frame rate.
    pub cooldown_ticks: u32,
    /// Wind-up before the effects land, in ticks.
    pub windup_ticks: u32,
    /// Recovery during which the actor is committed, in ticks.
    pub recovery_ticks: u32,
    pub effects: Vec<Effect>,
    /// How the ability looks (ADR-0017 §8). Absent is the fallback every
    /// ability had before: the body's own cast clip and a debug burst.
    #[serde(default)]
    pub presentation: AbilityPresentation,
}

/// How an ability looks: presentation only, so none of it can change what the
/// ability does, when it lands, or whom it touches.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbilityPresentation {
    /// The clip the caster plays through the ability's windows, in place of
    /// its body's `cast` clip. A `Windows` clip, so it cannot retime the cast.
    #[serde(default)]
    pub clip: Option<ClipId>,
    /// Plays on the caster the tick the effects land.
    #[serde(default)]
    pub cast: Option<VfxId>,
    /// What a projectile looks like in flight: a model, flown nose first.
    /// Only for an ability that launches one.
    #[serde(default)]
    pub projectile: Option<ModelId>,
    /// Played again and again where a projectile has just been, so it leaves
    /// a trail. Only for an ability that launches one.
    #[serde(default)]
    pub trail: Option<VfxId>,
    /// Plays on everyone the effects changed — damaged, afflicted, healed —
    /// played from the caster, so a beam in it spans caster to target. Played
    /// on what was *changed*, not on what was in reach, so a summon standing
    /// next to the player does not look like it hit them.
    #[serde(default)]
    pub impact: Option<VfxId>,
    /// Marks the ground the ability will land on through its wind-up.
    ///
    /// Only a switch: the shape is read from the ability's targeting and the
    /// runtime's reach, never authored, so a telegraph cannot disagree with
    /// the area it warns of. Valid only for area targeting — a circle — or
    /// for an ability that launches a projectile — a lane along its aim. A
    /// single-target ability picks its target when it lands, so there is no
    /// ground to mark in advance that would be honest.
    #[serde(default)]
    pub telegraph: bool,
}

impl AbilityDef {
    /// The first projectile this ability launches, if any.
    pub fn projectile(&self) -> Option<&Effect> {
        self.effects
            .iter()
            .find(|effect| matches!(effect, Effect::Projectile { .. }))
    }
}
