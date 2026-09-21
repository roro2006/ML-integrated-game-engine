//! Status effects: bounded, allowlisted modifiers with a lifetime in ticks.
//!
//! M6 wave 2 task E. `Effect::ApplyStatus` names a [`StatusId`]; this module
//! decides what that name is allowed to *mean*.
//!
//! # A status is a modifier, not a script
//!
//! The set of things a status can do is [`Modifier`], an allowlist in Rust.
//! Which modifiers a status id grants is content (M9): a `StatusDef` lists
//! them, bounded by the loader, and [`modifiers_of`] converts them once, when
//! the status is applied. New statuses are content; a new *kind* of modifier
//! is a code change with tests — the same principle as the `Effect`
//! allowlist and `BehaviorTemplate`, applied one level down.
//!
//! Before M9 the mapping was a `match` here, and an unknown id applied and
//! did nothing. Now an `ApplyStatus` naming an undefined status does not load
//! (ADR-0010), so a status in play always means what its definition says.
//!
//! # Bounded three ways
//!
//! - **Duration.** Every status carries an absolute expiry tick, so it ends
//!   whether or not anything remembers to tick it (`ability::MAX_STATUS_TICKS`
//!   caps how long it may be asked to last).
//! - **Count.** An entity holds at most [`MAX_ACTIVE_STATUSES`], so repeatedly
//!   applying *distinct* ids cannot grow without limit.
//! - **Magnitude.** Multipliers clamp into [`MIN_MULTIPLIER`]..=[`MAX_MULTIPLIER`],
//!   so no combination of statuses can reach immortality or a one-shot.
//!
//! # Runtime only
//!
//! Active statuses are never serialized. They are rebuilt from content and
//! canonical state on load, exactly like attack phase and current stamina
//! (`docs/plans/m6-combat-ability-runtime.md`).

use engine::ecs::Component;
use game_core::content::{DamageType, FIXED_SCALE, StatusDef, StatusId, StatusModifier};

/// Maximum statuses one entity may hold at once.
///
/// Refreshing an existing status is free, so this bounds only the number of
/// *distinct* ids in play. Eight is far more than the slice authors — it has
/// one — and small enough that iterating them per hit stays trivial.
pub const MAX_ACTIVE_STATUSES: usize = 8;

/// Lower bound on any damage multiplier.
///
/// Not zero: a status that reduced incoming damage to nothing would make a
/// fight unloseable, and ADR-0007's combat is built on trades being real.
pub const MIN_MULTIPLIER: f32 = 0.1;

/// Upper bound on any damage multiplier.
pub const MAX_MULTIPLIER: f32 = 10.0;

/// What a status is permitted to do.
///
/// Extending this enum is a deliberate act with tests, exactly like extending
/// `Effect`. There is intentionally no variant that runs arbitrary logic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Modifier {
    /// Scales damage this entity takes. Below 1.0 is protection.
    DamageTaken(f32),
    /// Scales damage this entity deals.
    DamageDealt(f32),
    /// Scales stamina regeneration.
    StaminaRegen(f32),
    /// Damage applied once per tick while the status holds.
    DamageOverTime {
        /// Amount per tick, before the target's resistances.
        per_tick: f32,
        kind: DamageType,
    },
}

/// Fixed ticks in a second, for turning an authored per-second rate into a
/// per-tick one. The simulation steps at 60 Hz.
pub const TICKS_PER_SECOND: f32 = 60.0;

/// Most modifiers one active status holds. Content is bounded tighter by the
/// loader (`MAX_STATUS_MODIFIERS`); this is the runtime's own backstop.
pub const MAX_MODIFIERS_PER_STATUS: usize = 8;

/// The modifiers a status definition grants, in simulation units.
///
/// Fixed-point content becomes floats here, once, at the boundary: a
/// multiplier of 500 milli is 0.5, and damage per second becomes damage per
/// tick.
pub fn modifiers_of(def: &StatusDef) -> Vec<Modifier> {
    let scale = |value: game_core::content::Fixed| value.as_milli() as f32 / FIXED_SCALE as f32;
    def.modifiers
        .iter()
        .map(|modifier| match modifier {
            StatusModifier::DamageTaken(value) => Modifier::DamageTaken(scale(*value)),
            StatusModifier::DamageDealt(value) => Modifier::DamageDealt(scale(*value)),
            StatusModifier::StaminaRegen(value) => Modifier::StaminaRegen(scale(*value)),
            StatusModifier::DamageOverTime { per_second, kind } => Modifier::DamageOverTime {
                per_tick: scale(*per_second) / TICKS_PER_SECOND,
                kind: *kind,
            },
        })
        .collect()
}

/// One status currently affecting an entity.
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveStatus {
    pub status: StatusId,
    /// Absolute tick at which this status stops applying.
    ///
    /// Absolute rather than a countdown so that expiry is a comparison against
    /// the simulation clock rather than an accumulation — a missed or doubled
    /// tick then cannot silently extend a status.
    pub expires_at: u64,
    /// What it does while it holds, resolved from content when applied.
    pub modifiers: Vec<Modifier>,
}

/// Statuses currently affecting one entity.
///
/// Runtime state: rebuilt on load, never serialized.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Statuses {
    active: Vec<ActiveStatus>,
}

impl Component for Statuses {}

impl Statuses {
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies a status, refreshing any existing instance of the same id.
    ///
    /// Refreshing rather than stacking is the conservative choice: it cannot
    /// accumulate, so the same ability cast repeatedly extends its protection
    /// rather than multiplying it. Returns whether the status was recorded — a
    /// new id is refused once [`MAX_ACTIVE_STATUSES`] are held, so a caller can
    /// tell a refusal from a refresh.
    pub fn apply(&mut self, status: StatusId, expires_at: u64, modifiers: &[Modifier]) -> bool {
        if let Some(held) = self.active.iter_mut().find(|held| held.status == status) {
            // Never shortens: a weaker reapplication of the same status must not
            // cut an existing one short.
            held.expires_at = held.expires_at.max(expires_at);
            return true;
        }
        if self.active.len() >= MAX_ACTIVE_STATUSES {
            return false;
        }
        self.active.push(ActiveStatus {
            status,
            expires_at,
            modifiers: modifiers.iter().copied().take(MAX_MODIFIERS_PER_STATUS).collect(),
        });
        true
    }

    /// Drops every status whose expiry tick has arrived.
    pub fn expire(&mut self, now: u64) {
        self.active.retain(|held| held.expires_at > now);
    }

    /// Whether `status` applies as of `now`.
    ///
    /// Checks the expiry rather than trusting [`Self::expire`] to have run, so a
    /// caller that forgets to tick cannot leave a status in force forever.
    pub fn is_active(&self, status: &StatusId, now: u64) -> bool {
        self.active
            .iter()
            .any(|held| &held.status == status && held.expires_at > now)
    }

    pub fn len(&self) -> usize {
        self.active.len()
    }

    pub fn is_empty(&self) -> bool {
        self.active.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &ActiveStatus> {
        self.active.iter()
    }

    /// Every modifier granted by every status still in force at `now`.
    ///
    /// Order follows application order, which is stable and caller-controlled;
    /// the aggregate multipliers below are products, so order does not change
    /// the result, but a deterministic iteration keeps any future
    /// order-sensitive modifier honest.
    pub fn modifiers(&self, now: u64) -> impl Iterator<Item = &Modifier> + '_ {
        self.active
            .iter()
            .filter(move |held| held.expires_at > now)
            .flat_map(|held| held.modifiers.iter())
    }

    /// Combined multiplier on damage taken, clamped.
    pub fn damage_taken_multiplier(&self, now: u64) -> f32 {
        self.multiplier(now, |modifier| match modifier {
            Modifier::DamageTaken(scale) => Some(*scale),
            _ => None,
        })
    }

    /// Combined multiplier on damage dealt, clamped.
    pub fn damage_dealt_multiplier(&self, now: u64) -> f32 {
        self.multiplier(now, |modifier| match modifier {
            Modifier::DamageDealt(scale) => Some(*scale),
            _ => None,
        })
    }

    /// Combined multiplier on stamina regeneration, clamped.
    pub fn stamina_regen_multiplier(&self, now: u64) -> f32 {
        self.multiplier(now, |modifier| match modifier {
            Modifier::StaminaRegen(scale) => Some(*scale),
            _ => None,
        })
    }

    /// Total damage-over-time owed this tick, before the target's resistances.
    ///
    /// Returned per damage kind so resistances still apply: a burn should be
    /// resisted like any other fire.
    pub fn damage_over_time(&self, now: u64) -> Vec<(f32, DamageType)> {
        self.modifiers(now)
            .filter_map(|modifier| match modifier {
                Modifier::DamageOverTime { per_tick, kind } => Some((per_tick.max(0.0), *kind)),
                _ => None,
            })
            .collect()
    }

    /// Products the selected multipliers and clamps the result.
    ///
    /// Clamping the *product* rather than each factor is what makes the bound
    /// hold under stacking: eight statuses at 0.5 each would otherwise reach
    /// 0.004 and be indistinguishable from immunity.
    fn multiplier(&self, now: u64, select: impl Fn(&Modifier) -> Option<f32>) -> f32 {
        let product = self
            .modifiers(now)
            .filter_map(select)
            // A non-finite authored multiplier would poison the product, so it
            // is dropped rather than propagated.
            .filter(|scale| scale.is_finite())
            .fold(1.0, |total, scale| total * scale);
        product.clamp(MIN_MULTIPLIER, MAX_MULTIPLIER)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::ids::AbilityId;

    fn status(name: &str) -> StatusId {
        StatusId(AbilityId::new(name).expect("valid id"))
    }

    fn guarded() -> StatusId {
        status("status-guarded")
    }

    /// What `status-guarded` grants in the shipped content.
    const GUARDED: [Modifier; 1] = [Modifier::DamageTaken(0.5)];

    fn def(modifiers: Vec<StatusModifier>) -> StatusDef {
        StatusDef {
            schema_version: 1,
            id: guarded(),
            name: "Guarded".to_owned(),
            palette: game_core::ids::PaletteId::new("cinderwake").expect("valid id"),
            color: game_core::ids::SwatchId::new("warden-blue").expect("valid id"),
            modifiers,
            vfx: None,
        }
    }

    #[test]
    fn a_definition_becomes_simulation_units() {
        use game_core::content::Fixed;
        let converted = modifiers_of(&def(vec![
            StatusModifier::DamageTaken(Fixed::from_milli(500)),
            StatusModifier::DamageDealt(Fixed::from_milli(1_500)),
            StatusModifier::StaminaRegen(Fixed::from_milli(400)),
            StatusModifier::DamageOverTime {
                per_second: Fixed::from_milli(3_000),
                kind: DamageType::Fire,
            },
        ]));
        assert_eq!(
            converted,
            vec![
                Modifier::DamageTaken(0.5),
                Modifier::DamageDealt(1.5),
                Modifier::StaminaRegen(0.4),
                Modifier::DamageOverTime {
                    per_tick: 3.0 / TICKS_PER_SECOND,
                    kind: DamageType::Fire
                },
            ]
        );
    }

    /// A status applied with nothing to grant changes nothing.
    #[test]
    fn a_status_with_no_modifiers_changes_nothing() {
        let mut statuses = Statuses::new();
        assert!(statuses.apply(status("status-inert"), 1_000, &[]));
        assert!(
            (statuses.damage_taken_multiplier(0) - 1.0).abs() < 1e-6,
            "nothing granted, nothing changed"
        );
        assert!(statuses.damage_over_time(0).is_empty());
    }

    #[test]
    fn a_status_holds_for_its_duration_and_then_stops() {
        let mut statuses = Statuses::new();
        assert!(statuses.apply(guarded(), 180, &GUARDED));

        for tick in 0..180 {
            statuses.expire(tick);
            assert!(statuses.is_active(&guarded(), tick), "lapsed early at {tick}");
            assert!((statuses.damage_taken_multiplier(tick) - 0.5).abs() < 1e-6);
        }
        statuses.expire(180);
        assert!(!statuses.is_active(&guarded(), 180));
        assert!(statuses.is_empty());
        assert!(
            (statuses.damage_taken_multiplier(180) - 1.0).abs() < 1e-6,
            "an expired status stops modifying"
        );
    }

    /// Reapplying refreshes rather than stacking, so a spammed ability cannot
    /// multiply its own protection.
    #[test]
    fn reapplying_refreshes_rather_than_stacking() {
        let mut statuses = Statuses::new();
        for _ in 0..20 {
            assert!(statuses.apply(guarded(), 300, &GUARDED));
        }
        assert_eq!(statuses.len(), 1, "one id is one entry");
        assert!(
            (statuses.damage_taken_multiplier(0) - 0.5).abs() < 1e-6,
            "twenty casts protect exactly as much as one"
        );
    }

    #[test]
    fn a_refresh_never_shortens_an_existing_status() {
        let mut statuses = Statuses::new();
        statuses.apply(guarded(), 500, &GUARDED);
        statuses.apply(guarded(), 100, &GUARDED);
        assert!(
            statuses.is_active(&guarded(), 400),
            "a shorter reapplication must not cut the longer one short"
        );
    }

    /// The count bound: distinct ids cannot accumulate without limit.
    #[test]
    fn an_entity_cannot_hold_unbounded_statuses() {
        let mut statuses = Statuses::new();
        for index in 0..MAX_ACTIVE_STATUSES {
            assert!(
                statuses.apply(status(&format!("status-{index}")), 1_000, &[]),
                "the first {MAX_ACTIVE_STATUSES} are accepted"
            );
        }
        assert_eq!(statuses.len(), MAX_ACTIVE_STATUSES);

        for index in 0..50 {
            assert!(
                !statuses.apply(status(&format!("status-overflow-{index}")), 1_000, &[]),
                "a new id past the cap is refused"
            );
        }
        assert_eq!(statuses.len(), MAX_ACTIVE_STATUSES, "the cap holds");

        // A refresh of something already held still works at the cap, or a
        // capped entity could never renew what it has.
        assert!(statuses.apply(status("status-0"), 5_000, &[]));
        assert_eq!(statuses.len(), MAX_ACTIVE_STATUSES);
    }

    /// The magnitude bound: stacked protection cannot reach immunity.
    #[test]
    fn stacked_multipliers_cannot_reach_immunity_or_a_one_shot() {
        // Eight distinct halving statuses would multiply to 0.0039.
        let mut statuses = Statuses::new();
        for index in 0..MAX_ACTIVE_STATUSES {
            statuses.apply(status(&format!("status-{index}")), 1_000, &GUARDED);
        }
        let taken = statuses.damage_taken_multiplier(0);
        assert!(
            (taken - MIN_MULTIPLIER).abs() < 1e-6,
            "0.5 to the eighth is clamped to the floor, got {taken}"
        );
        assert!(
            (MIN_MULTIPLIER..=MAX_MULTIPLIER).contains(&taken),
            "multiplier {taken} escaped its bounds"
        );
        assert!(taken > 0.0, "damage taken is never nullified");
    }

    #[test]
    fn multipliers_are_clamped_at_both_ends() {
        let mut statuses = Statuses::new();
        statuses.apply(guarded(), 1_000, &GUARDED);
        let taken = statuses.damage_taken_multiplier(0);
        assert!((MIN_MULTIPLIER..=MAX_MULTIPLIER).contains(&taken));

        // Nothing here grants dealt or regen modifiers, so both are the
        // identity — which is the correct answer, not a missing one.
        assert!((statuses.damage_dealt_multiplier(0) - 1.0).abs() < 1e-6);
        assert!((statuses.stamina_regen_multiplier(0) - 1.0).abs() < 1e-6);

        let mut frenzied = Statuses::new();
        for index in 0..MAX_ACTIVE_STATUSES {
            frenzied.apply(
                status(&format!("status-{index}")),
                1_000,
                &[Modifier::DamageDealt(4.0)],
            );
        }
        assert!(
            (frenzied.damage_dealt_multiplier(0) - MAX_MULTIPLIER).abs() < 1e-6,
            "stacked damage dealt is capped too"
        );
    }

    /// Expired statuses must not contribute, even when `expire` has not run.
    #[test]
    fn modifiers_ignore_lapsed_statuses_without_expiring_them() {
        let mut statuses = Statuses::new();
        statuses.apply(guarded(), 10, &GUARDED);
        assert!((statuses.damage_taken_multiplier(9) - 0.5).abs() < 1e-6);
        assert!(
            (statuses.damage_taken_multiplier(10) - 1.0).abs() < 1e-6,
            "the modifier stops at the expiry tick, not when someone tidies up"
        );
        assert_eq!(statuses.len(), 1, "and the entry is still there, untidied");
    }

    #[test]
    fn damage_over_time_is_reported_per_kind_and_never_negative() {
        assert!(Statuses::new().damage_over_time(0).is_empty());
        let mut statuses = Statuses::new();
        statuses.apply(
            status("status-smouldering"),
            100,
            &[Modifier::DamageOverTime {
                per_tick: 0.05,
                kind: DamageType::Fire,
            }],
        );
        statuses.apply(
            status("status-poisoned-wrongly"),
            100,
            &[Modifier::DamageOverTime {
                per_tick: -3.0,
                kind: DamageType::Corruption,
            }],
        );
        assert_eq!(
            statuses.damage_over_time(0),
            vec![(0.05, DamageType::Fire), (0.0, DamageType::Corruption)],
            "a negative rate would heal; it is floored at nothing"
        );
        assert!(
            statuses.damage_over_time(100).is_empty(),
            "and it stops when the status does"
        );
    }

    #[test]
    fn statuses_iterate_deterministically() {
        let mut statuses = Statuses::new();
        statuses.apply(status("status-a"), 100, &[]);
        statuses.apply(status("status-b"), 100, &[]);
        statuses.apply(status("status-c"), 100, &[]);

        let first: Vec<&StatusId> = statuses.iter().map(|held| &held.status).collect();
        let second: Vec<&StatusId> = statuses.iter().map(|held| &held.status).collect();
        assert_eq!(first, second);
        assert_eq!(
            first.iter().map(|id| id.0.name().to_owned()).collect::<Vec<_>>(),
            ["status-a", "status-b", "status-c"],
            "application order is preserved"
        );
    }
}
