//! Items and crafting recipes.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::ids::{ItemId, UpgradeId};

use super::{DamageType, Fixed};

/// What an item is for. Drives stacking and UI grouping, not behaviour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum ItemCategory {
    Material,
    Consumable,
    Weapon,
    Armor,
    QuestItem,
}

/// What an item contributes to combat while it is equipped.
///
/// Optional on [`ItemDef`], because most items have nothing to say here: a
/// material is not a weapon and should not have to declare that it deals no
/// damage. Every field is fixed-point for the reason the module docs give —
/// these values feed simulation, and replay equality must not depend on
/// floating-point behaviour. The runtime converts once, at its own boundary.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CombatStats {
    /// Added to the wielder's attack damage.
    #[serde(default)]
    pub damage: Fixed,
    /// Retypes the wielder's attacks. `None` leaves the attack as it was, which
    /// is what lets armour carry stats without turning a sword into a torch.
    #[serde(default)]
    pub damage_type: Option<DamageType>,
    /// Per-kind damage reduction, as a fraction in milli-units: `250` is 25%.
    ///
    /// A `BTreeMap` rather than a list of pairs so the ordering is total and
    /// the serialized form is stable, which content hashing depends on.
    #[serde(default)]
    pub resistances: BTreeMap<DamageType, Fixed>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ItemDef {
    pub schema_version: u32,
    pub id: ItemId,
    pub name: String,
    pub category: ItemCategory,
    /// Maximum stack size; must be at least one or the item cannot be held.
    pub stack_limit: u32,
    /// Combat properties when equipped, if any.
    ///
    /// `#[serde(default)]` so every item authored before M6 still parses and
    /// `CONTENT_SCHEMA_VERSION` stays 1. Adding a field that every existing
    /// content file would otherwise have to be edited for is exactly the kind
    /// of breaking schema change ADR-0013 exists to avoid.
    #[serde(default)]
    pub combat: Option<CombatStats>,
    /// What using it does (M9). Only a consumable may say, and using one
    /// spends it.
    #[serde(default)]
    pub on_use: Option<ItemUse>,
}

/// What using a consumable does: effects on the user, after a short
/// committed moment in which the user cannot swing, roll, or cast.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ItemUse {
    /// Applied to the user. Only effects that act on the user are allowed:
    /// restoring or spending a resource, and applying a status.
    pub effects: Vec<super::Effect>,
    /// How long using it takes, in fixed ticks.
    pub use_ticks: u32,
}

/// Most effects one item's use may carry.
pub const MAX_USE_EFFECTS: usize = 4;
/// Longest using an item may take, in fixed ticks: two seconds.
pub const MAX_USE_TICKS: u32 = 120;

/// A crafting recipe: inputs consumed, output produced.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeDef {
    pub schema_version: u32,
    pub id: ItemId,
    pub name: String,
    /// Quantities required, keyed by item so the map is order-stable.
    pub inputs: BTreeMap<ItemId, u32>,
    pub output: ItemId,
    pub output_quantity: u32,
    /// Settlement upgrade required to craft this, if any (ADR-0007 blueprints).
    ///
    /// An `UpgradeId`, not an `ItemId`: a blueprint is not a thing you carry,
    /// and typing it correctly lets the loader resolve it against the upgrade
    /// registry instead of comparing names across two id types.
    pub requires_upgrade: Option<UpgradeId>,
}
