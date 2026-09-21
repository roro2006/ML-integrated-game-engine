//! Enemies and encounter composition.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::ids::{AbilityId, EncounterId, LootTableId, ModelId, PrefabId};

use super::*;

/// How an enemy behaves, chosen from implemented templates rather than authored
/// as logic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum BehaviorTemplate {
    Melee,
    Ranged,
    Charger,
    Support,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnemyDef {
    pub schema_version: u32,
    pub id: PrefabId,
    pub name: String,
    pub behavior: BehaviorTemplate,
    pub max_health: Fixed,
    pub damage: Fixed,
    pub abilities: Vec<AbilityId>,
    /// Visual tags resolved against the approved asset library; never a path.
    pub visual_tags: BTreeSet<String>,
    /// Difficulty weight this enemy contributes to an encounter budget.
    pub threat: Fixed,
    /// The archetype this is a variant of, if any. Variation without
    /// duplicating a whole definition (ADR-0007's single elite).
    pub variant_of: Option<PrefabId>,
    /// What this drops when killed.
    ///
    /// A property of the enemy rather than of where it stands: the same husk
    /// drops the same things wherever it is placed. Optional because not
    /// everything that can die is worth looting, and `#[serde(default)]` so
    /// content authored before this field still parses.
    #[serde(default)]
    pub loot_table: Option<LootTableId>,
    /// What it looks like (ADR-0017). Absent means the block-out shape the
    /// slice has always drawn, so an enemy without a model still plays.
    #[serde(default)]
    pub model: Option<ModelId>,
    /// Experience for killing one (ADR-0018). Levels are how talent points
    /// arrive, so fights are part of how a build grows.
    #[serde(default)]
    pub experience: u32,
    /// How it plays, beyond which template it thinks with (M8). Every field
    /// is optional and falls back to the template's value, so an enemy that
    /// says nothing plays exactly as its template always has.
    #[serde(default)]
    pub tuning: EnemyTuning,
    /// How its swings look.
    #[serde(default)]
    pub presentation: MeleePresentation,
}

/// How a body's plain swings look — the blows that are not abilities, and so
/// have no `AbilityPresentation` of their own. Presentation only, like every
/// look: neither effect changes what a swing does.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeleePresentation {
    /// Plays the tick a swing goes live, from the swinger toward where it
    /// lands, so a `Slash` in it traces the blow's path.
    #[serde(default)]
    pub swing: Option<crate::ids::VfxId>,
    /// Plays where a swing connects.
    #[serde(default)]
    pub hit: Option<crate::ids::VfxId>,
}

/// An enemy's numbers: how far it notices, where it likes to stand, how fast
/// it moves, and how its swing is timed and shaped.
///
/// Numbers only. The behaviour template still decides *how* the enemy
/// thinks — when it closes, when it backs off — and content cannot describe a
/// decision procedure (ADR-0010). What moved here is only what the templates
/// used to hardcode, so a new enemy that plays differently is a content file.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnemyTuning {
    /// How close the player must come to be noticed, in metres.
    #[serde(default)]
    pub acquire: Option<Fixed>,
    /// How far the player must get to be forgotten, in metres. Never less
    /// than `acquire`: the gap is what stops an enemy flickering between idle
    /// and chasing at the edge.
    #[serde(default)]
    pub release: Option<Fixed>,
    /// How close it wants to be before it acts, in metres.
    #[serde(default)]
    pub preferred_range: Option<Fixed>,
    /// Metres per second.
    #[serde(default)]
    pub move_speed: Option<Fixed>,
    /// Its swing's windows, in fixed ticks (ADR-0007).
    #[serde(default)]
    pub windup_ticks: Option<u32>,
    #[serde(default)]
    pub active_ticks: Option<u32>,
    #[serde(default)]
    pub recovery_ticks: Option<u32>,
    /// How far in front of it a swing lands, in metres.
    #[serde(default)]
    pub reach: Option<Fixed>,
    /// How wide a swing's hit is, in metres.
    #[serde(default)]
    pub strike_radius: Option<Fixed>,
    /// How much damage its stance absorbs before it staggers. Deep poise is
    /// what makes a warden or an elite hard to interrupt.
    #[serde(default)]
    pub poise: Option<Fixed>,
}

/// Furthest an enemy may notice the player from, in metres.
pub const MAX_AGGRO_METRES: i64 = 30;
/// Furthest an enemy may want to stand from the player, in metres.
pub const MAX_PREFERRED_RANGE_METRES: i64 = 20;
/// Fastest an enemy may move, in metres per second: a little over twice the
/// player's walk, which is already a charge.
pub const MAX_ENEMY_SPEED_METRES: i64 = 12;
/// Longest a swing window may be, in ticks: four seconds at 60 Hz.
pub const MAX_WINDOW_TICKS: u32 = 240;
/// Furthest a swing may reach, in metres.
pub const MAX_REACH_METRES: i64 = 10;
/// Widest a swing's hit may be, in metres.
pub const MAX_STRIKE_RADIUS_METRES: i64 = 3;
/// Deepest an enemy's poise may be: enough that it staggers eventually, never
/// so much that it simply cannot.
pub const MAX_POISE: i64 = 500;

/// One group of enemies placed in an encounter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncounterSlot {
    pub enemy: PrefabId,
    pub count: u8,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncounterDef {
    pub schema_version: u32,
    pub id: EncounterId,
    pub name: String,
    /// Upper bound on total threat. M11 validates generated encounters against
    /// this, so it must be explicit rather than implied by the slot list.
    pub threat_budget: Fixed,
    pub slots: Vec<EncounterSlot>,
    /// Played where each enemy of a later wave is about to appear, so a
    /// reinforcement is seen coming rather than popping in (M8). Absent, they
    /// arrive unannounced.
    #[serde(default)]
    pub arrival: Option<crate::ids::VfxId>,
}

/// Longest an encounter may be spread out: the first wave and three more.
pub const MAX_WAVES: u8 = 4;
