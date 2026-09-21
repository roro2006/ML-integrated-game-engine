//! Statuses: what a status id means, as content (M9 wave 2).
//!
//! Until M9 the meaning of a status was a `match` in `game_runtime::status`,
//! and any other id loaded, applied, and silently did nothing. A status is now
//! a definition — a name, a colour, and a short list of modifiers — and an
//! `ApplyStatus` naming one that does not exist stops the content loading.
//!
//! What a status can *do* is still an allowlist in Rust: [`StatusModifier`]
//! names the four kinds of modifier the runtime implements, each bounded
//! here. New statuses built from those kinds are content; a new kind is a code
//! change with tests (ADR-0010).

use serde::{Deserialize, Serialize};

use crate::ids::{PaletteId, SwatchId, VfxId};

use super::{DamageType, Fixed, StatusId};

/// Most modifiers one status may carry.
pub const MAX_STATUS_MODIFIERS: usize = 4;
/// Bounds on a status multiplier, in milli-units: a tenth to ten times.
pub const MIN_STATUS_MULTIPLIER_MILLI: i64 = 100;
pub const MAX_STATUS_MULTIPLIER_MILLI: i64 = 10_000;
/// Most damage per second one damage-over-time modifier may deal, in
/// milli-units: thirty a second takes the slice's player from full to nothing
/// in four seconds, which is the most a single status should ever threaten.
pub const MAX_STATUS_DPS_MILLI: i64 = 30_000;

/// One thing a status does while it holds. Multipliers are fixed-point
/// milli-units, so 500 halves and 2000 doubles.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum StatusModifier {
    /// Scales damage the bearer takes.
    DamageTaken(Fixed),
    /// Scales damage the bearer deals.
    DamageDealt(Fixed),
    /// Scales how fast the bearer's stamina comes back.
    StaminaRegen(Fixed),
    /// Damage every second, of a kind the bearer's resistances answer.
    DamageOverTime { per_second: Fixed, kind: DamageType },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatusDef {
    pub schema_version: u32,
    pub id: StatusId,
    /// What the status chip says.
    pub name: String,
    /// The chip's colour: a swatch in `palette`.
    pub palette: PaletteId,
    pub color: SwatchId,
    pub modifiers: Vec<StatusModifier>,
    /// Played on the bearer when the status is applied, if set.
    #[serde(default)]
    pub vfx: Option<VfxId>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::AbilityId;

    fn status(modifiers: Vec<StatusModifier>) -> StatusDef {
        StatusDef {
            schema_version: super::super::CONTENT_SCHEMA_VERSION,
            id: StatusId(AbilityId::new("status-test").expect("valid id")),
            name: "Test".to_owned(),
            palette: PaletteId::new("cinderwake").expect("valid id"),
            color: SwatchId::new("ember").expect("valid id"),
            modifiers,
            vfx: None,
        }
    }

    fn milli(value: i64) -> Fixed {
        Fixed::from_milli(value)
    }

    #[test]
    fn a_status_must_do_something_and_not_too_much() {
        assert!(
            status(Vec::new()).validate().is_err(),
            "a status with no modifiers"
        );
        assert!(
            status(vec![
                StatusModifier::DamageTaken(milli(500));
                MAX_STATUS_MODIFIERS + 1
            ])
            .validate()
            .is_err(),
            "too many modifiers"
        );
        assert!(
            status(vec![
                StatusModifier::DamageTaken(milli(500));
                MAX_STATUS_MODIFIERS
            ])
            .validate()
            .is_ok()
        );
    }

    #[test]
    fn multipliers_and_burns_are_bounded() {
        for (value, ok) in [
            (MIN_STATUS_MULTIPLIER_MILLI - 1, false),
            (MIN_STATUS_MULTIPLIER_MILLI, true),
            (MAX_STATUS_MULTIPLIER_MILLI, true),
            (MAX_STATUS_MULTIPLIER_MILLI + 1, false),
        ] {
            for modifier in [
                StatusModifier::DamageTaken(milli(value)),
                StatusModifier::DamageDealt(milli(value)),
                StatusModifier::StaminaRegen(milli(value)),
            ] {
                assert_eq!(
                    status(vec![modifier.clone()]).validate().is_ok(),
                    ok,
                    "{modifier:?}"
                );
            }
        }
        for (value, ok) in [
            (0, false),
            (1, true),
            (MAX_STATUS_DPS_MILLI, true),
            (MAX_STATUS_DPS_MILLI + 1, false),
        ] {
            let burn = StatusModifier::DamageOverTime {
                per_second: milli(value),
                kind: DamageType::Fire,
            };
            assert_eq!(status(vec![burn]).validate().is_ok(), ok, "{value} per second");
        }
    }

    #[test]
    fn statuses_round_trip_and_default_their_effect() {
        let text = r#"(schema_version: 1, id: "status-test", name: "Test", palette: "cinderwake", color: "ember", modifiers: [DamageOverTime(per_second: 3000, kind: Fire)])"#;
        let def: StatusDef = ron::from_str(text).expect("parses without a vfx field");
        assert_eq!(def.vfx, None);
        let again: StatusDef = ron::from_str(&ron::to_string(&def).expect("serializes")).expect("parses");
        assert_eq!(again, def);
    }
}
