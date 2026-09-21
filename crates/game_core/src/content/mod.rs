//! Typed content schemas.
//!
//! The authored data model: what a designer writes in RON, and — just as
//! importantly — the complete vocabulary the AI pipeline will later be
//! constrained to propose (ADR-0010). Content can only compose primitives that
//! exist in this file, which is what makes generated content safe to run: there
//! is no script string, no expression language, and no escape hatch through
//! which a proposal could express behaviour the engine has not already
//! implemented and tested.
//!
//! Keep this set finite and small enough to hold in your head. Every addition
//! is a promise to support that primitive forever, in saves and in replay.
//!
//! # No floating point
//!
//! Every value that feeds simulation is fixed-point ([`Fixed`], milli-units).
//! Damage, costs, and budgets are compared, serialized, and replayed; float
//! results differ subtly across platforms and optimization levels, which would
//! make replay equality — the property the whole event-sourced design rests on
//! — quietly false. Do not "simplify" these to `f32`.
//!
//! # Why variants are not kebab-case
//!
//! Enum variants keep their Rust spelling (`QuestItem`, `TalkTo`) rather than
//! being renamed to kebab-case. RON cannot parse a bare hyphenated identifier:
//! with `rename_all = "kebab-case"` an author would have to write
//! `r#quest-item` in every content file, and a round trip through RON's own
//! serializer would hide that from tests. The project's kebab-case rule governs
//! filenames and *ids* — which are strings here, and stay kebab-case — not the
//! spelling of variants.
//!
//! # Scope
//!
//! This module defines shapes and validates *intrinsic* invariants: a quest
//! graph reaches a completion, a budget is not negative, a stack limit is
//! positive. It does **not** resolve cross-file references — whether
//! `ItemId("ember-ore")` actually exists is `content_runtime`'s job, because
//! only it has seen every file.

mod abilities;
mod classes;
mod clips;
mod dialogue;
mod enemies;
mod items;
mod levels;
mod locations;
mod models;
mod primitives;
mod quests;
mod regions;
mod rewards;
mod statuses;
mod validation;
mod vfx;

#[cfg(test)]
mod tests;

pub use abilities::{
    AbilityDef, AbilityPresentation, Effect, MAX_ON_HIT_EFFECTS, MAX_PIERCE, MAX_PROJECTILE_RADIUS_MILLI,
    MAX_PROJECTILE_RANGE_METRES, MAX_PROJECTILE_SPEED_METRES, MIN_PROJECTILE_RADIUS_MILLI,
    MIN_PROJECTILE_SPEED_METRES, Targeting,
};
pub use classes::{ClassCombat, ClassDef, DODGE_TICKS, MAX_CLASS_COST, MAX_CLASS_DAMAGE};
pub use clips::{
    AnimationSet, CYCLE_END_MILLI, ClipDef, ClipTiming, KeyDef, MAX_KEY_OFFSET_METRES, MAX_KEYS,
    MAX_LOOP_TICKS, MAX_STRIDE_MILLI, MAX_TRACKS, MIN_STRIDE_MILLI, TrackDef, WINDOWS_END_MILLI,
    timing_suits,
};
pub use dialogue::{Condition, DialogueDef, DialogueEffect, DialogueNode, DialogueOption};
pub use enemies::{
    BehaviorTemplate, EncounterDef, EncounterSlot, EnemyDef, EnemyTuning, MAX_AGGRO_METRES,
    MAX_ENEMY_SPEED_METRES, MAX_POISE, MAX_PREFERRED_RANGE_METRES, MAX_REACH_METRES,
    MAX_STRIKE_RADIUS_METRES, MAX_WAVES, MAX_WINDOW_TICKS, MeleePresentation,
};
pub use items::{CombatStats, ItemCategory, ItemDef, ItemUse, MAX_USE_EFFECTS, MAX_USE_TICKS, RecipeDef};
pub use levels::{LevelCurveDef, MAX_LEVEL};
pub use locations::{
    LocationDef, LocationKind, LocationLink, NavPolygon, Placement, Point, PropPlacement, SpawnPoint,
    SpawnRole, StationKind,
};
pub use models::{
    BodyPlanDef, MAX_DOME_RINGS, MAX_EXTENT_METRES, MAX_OFFSET_METRES, MAX_PARTS, MAX_SCALE_MILLI, MAX_SIDES,
    MAX_SLOTS, MAX_SPHERE_DETAIL, MAX_STRETCH_MILLI, MAX_SWATCHES, MAX_TRIANGLES, MAX_TURN_DEGREES,
    MIN_SCALE_MILLI, MIN_SIDES, MIN_STRETCH_MILLI, ModelDef, PaletteDef, PartDef, ShapeDef, SlotDef,
};
pub use primitives::{
    CONTENT_SCHEMA_VERSION, DamageType, FIXED_SCALE, Fixed, GameEventId, ResourceType, StatusId,
};
pub use quests::{Objective, QuestDef, QuestNode, RewardPackage};
pub use regions::{Biome, RegionDef};
pub use rewards::{LootEntry, LootTableDef, SettlementUpgradeDef};
pub use statuses::{
    MAX_STATUS_DPS_MILLI, MAX_STATUS_MODIFIERS, MAX_STATUS_MULTIPLIER_MILLI, MIN_STATUS_MULTIPLIER_MILLI,
    StatusDef, StatusModifier,
};
pub use validation::ContentError;
pub use vfx::{
    MAX_BEAM_MILLI, MAX_BURST_SHARDS, MAX_CRACK_MILLI, MAX_CRACKS, MAX_GRAVITY_METRES, MAX_PULSE_MILLI,
    MAX_SHARD_MILLI, MAX_SHARD_SPEED_METRES, MAX_SLASH_ARC_DEGREES, MAX_SLASH_RADIUS_MILLI,
    MAX_SPIRE_HEIGHT_MILLI, MAX_SPIRE_SPREAD_MILLI, MAX_SPIRE_WIDTH_MILLI, MAX_SPIRES, MAX_VFX_PARTS,
    MAX_VFX_TICKS, MIN_BEAM_MILLI, MIN_CRACK_MILLI, MIN_PULSE_MILLI, MIN_SHARD_MILLI, MIN_SLASH_ARC_DEGREES,
    MIN_SLASH_RADIUS_MILLI, MIN_SPIRE_HEIGHT_MILLI, MIN_SPIRE_SPREAD_MILLI, MIN_SPIRE_WIDTH_MILLI,
    SLASH_SEGMENTS, VfxDef, VfxPart,
};
