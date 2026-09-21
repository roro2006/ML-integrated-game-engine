//! Character classes.
//!
//! ADR-0007 chose distinct classes over one flexible archetype, and a class is
//! content rather than an enum so adding one is an authoring change. This is
//! the family M6 builds on; it exists in M5 because the player's pools have to
//! come from *somewhere* authored, and "somewhere" should not be a constant
//! buried in the runtime.
//!
//! # Pools are authored, not derived
//!
//! `base_health` and `base_stamina` are [`Fixed`] like every other simulation
//! value. They are copied into canonical state when a class is chosen, rather
//! than looked up whenever health is read: a save must reproduce the pools the
//! player actually had, even if the class was retuned afterwards. That is the
//! same rule an attack already follows when it copies its timing at the start
//! of a swing.

use serde::{Deserialize, Serialize};

use crate::ids::{AbilityId, ClassId, LevelCurveId, ModelId};

use super::Fixed;

/// One playable class.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassDef {
    pub schema_version: u32,
    pub id: ClassId,
    pub name: String,
    /// Starting hit points.
    pub base_health: Fixed,
    /// Starting action resource, gating committed attacks and dodges.
    pub base_stamina: Fixed,
    /// Abilities the class begins with. Resolved against the ability registry.
    #[serde(default)]
    pub abilities: Vec<AbilityId>,
    /// What a character of this class looks like (ADR-0017). Absent means the
    /// block-out shape.
    #[serde(default)]
    pub model: Option<ModelId>,
    /// How its swings look.
    #[serde(default)]
    pub presentation: super::MeleePresentation,
    /// How this class fights (M9). Every number is optional; one left out is
    /// the slice's tuned default, so a class that says nothing plays exactly
    /// as the wayfinder always has.
    #[serde(default)]
    pub combat: ClassCombat,
    /// How this class levels (ADR-0018). Without one it stays at level 1.
    #[serde(default)]
    pub level_curve: Option<LevelCurveId>,
}

/// A class's own swing, stance, and roll. Metres and points are fixed-point
/// milli-units; windows are fixed ticks.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassCombat {
    /// Damage of one swing, before equipment.
    #[serde(default)]
    pub damage: Option<Fixed>,
    /// How far in front a swing lands, in metres.
    #[serde(default)]
    pub reach: Option<Fixed>,
    /// How wide a swing's hit is, in metres.
    #[serde(default)]
    pub radius: Option<Fixed>,
    /// Stamina a swing costs.
    #[serde(default)]
    pub cost: Option<Fixed>,
    /// The swing's windows (ADR-0007).
    #[serde(default)]
    pub windup_ticks: Option<u32>,
    #[serde(default)]
    pub active_ticks: Option<u32>,
    #[serde(default)]
    pub recovery_ticks: Option<u32>,
    /// Damage the stance absorbs before it staggers.
    #[serde(default)]
    pub poise: Option<Fixed>,
    /// Stamina a roll costs.
    #[serde(default)]
    pub dodge_cost: Option<Fixed>,
    /// Ticks at the start of a roll that nothing can hit, out of the roll's
    /// [`DODGE_TICKS`].
    #[serde(default)]
    pub dodge_iframes: Option<u32>,
}

/// How long a roll lasts, in fixed ticks. Not a class's to change: the roll's
/// distance and feel are tuned around it, and i-frames are a share of it.
pub const DODGE_TICKS: u32 = 18;
/// Most damage one class swing may deal before equipment.
pub const MAX_CLASS_DAMAGE: i64 = 200;
/// Most stamina one swing or roll may cost.
pub const MAX_CLASS_COST: i64 = 100;

impl ClassDef {
    /// The starting health pool, in whole points.
    ///
    /// Canonical pools are integers; content authors in milli-units like
    /// everything else, and the conversion happens once, here.
    pub fn health_pool(&self) -> u32 {
        whole(self.base_health)
    }

    /// The starting stamina pool, in whole points.
    pub fn stamina_pool(&self) -> u32 {
        whole(self.base_stamina)
    }
}

/// Milli-units to whole points, clamped at zero.
///
/// Validation rejects non-positive values, so the clamp is belt and braces for
/// content that somehow bypassed it rather than an expected path.
fn whole(value: Fixed) -> u32 {
    u32::try_from(value.as_milli() / super::FIXED_SCALE).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class(health: i64, stamina: i64) -> ClassDef {
        ClassDef {
            schema_version: super::super::CONTENT_SCHEMA_VERSION,
            id: ClassId::new("wayfinder").expect("valid id"),
            name: "Wayfinder".to_owned(),
            base_health: Fixed::from_milli(health),
            base_stamina: Fixed::from_milli(stamina),
            abilities: Vec::new(),
            model: None,
            presentation: Default::default(),
            combat: ClassCombat::default(),
            level_curve: None,
        }
    }

    #[test]
    fn pools_convert_to_whole_points() {
        let def = class(120_000, 90_000);
        assert_eq!(def.health_pool(), 120);
        assert_eq!(def.stamina_pool(), 90);
    }

    #[test]
    fn a_class_nobody_could_survive_as_is_rejected() {
        assert!(class(0, 90_000).validate().is_err());
        assert!(class(120_000, 0).validate().is_err());
        assert!(class(-1_000, 90_000).validate().is_err());
        assert!(class(120_000, 90_000).validate().is_ok());
    }

    #[test]
    fn class_combat_is_bounded_and_optional() {
        let mut def = class(120_000, 90_000);
        assert!(def.validate().is_ok(), "a class that says nothing is valid");
        def.combat.damage = Some(Fixed::from_milli(MAX_CLASS_DAMAGE * 1000 + 1));
        assert!(def.validate().is_err(), "too heavy a blow");
        def.combat.damage = Some(Fixed::from_milli(34_000));
        assert!(def.validate().is_ok());
        def.combat.dodge_iframes = Some(DODGE_TICKS + 1);
        assert!(def.validate().is_err(), "i-frames longer than the roll");
        def.combat.dodge_iframes = Some(DODGE_TICKS);
        assert!(def.validate().is_ok());
        def.combat.windup_ticks = Some(0);
        assert!(def.validate().is_err(), "a swing needs a wind-up (ADR-0007)");
    }

    #[test]
    fn classes_round_trip() {
        let def = class(120_000, 90_000);
        let text = ron::to_string(&def).expect("serializes");
        assert_eq!(ron::from_str::<ClassDef>(&text).expect("parses"), def);
    }

    #[test]
    fn abilities_default_to_none() {
        let text = r#"(schema_version: 1, id: "wayfinder", name: "Wayfinder", base_health: 120000, base_stamina: 90000)"#;
        let def = ron::from_str::<ClassDef>(text).expect("parses without an abilities field");
        assert!(def.abilities.is_empty());
    }
}
