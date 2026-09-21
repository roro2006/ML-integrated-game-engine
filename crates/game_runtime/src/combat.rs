//! Attacks, hit detection, and damage.
//!
//! ADR-0007 chose Souls-lite combat: attacks are *committed*. Once an attack
//! starts, the attacker is locked into wind-up, an active window, and recovery,
//! and cannot simply cancel out of a bad decision. That is the whole feel of
//! it, and it only works if the windows are exact.
//!
//! # Everything is counted in ticks
//!
//! Not seconds, not frames. A 14-tick wind-up is 14 fixed simulation steps on
//! every machine, at every frame rate, in every replay. Expressing a window in
//! seconds would make the dodge timing depend on how fast the player's computer
//! happens to be, which is the bug this project keeps designing away from.
//!
//! # What is durable and what is not
//!
//! Nothing here writes canonical state. Attack phase, current health, and hit
//! bookkeeping are runtime state, rebuilt from content on load. Only *outcomes*
//! — an NPC dying, loot granted, an objective advanced — become domain events,
//! which is what keeps the event log readable instead of one entry per sword
//! swing (`docs/plans/m5-vertical-slice.md`).

use engine::ecs::{Component, Entity};
use engine::math::{Vec2, Vec3};
use game_core::content::{DamageType, ItemDef};
use game_core::ids::ItemId;
use game_core::world::{EquipmentSlot, PlayerState};

use crate::movement::Capsule;

/// A vector flattened to the ground plane.
///
/// Written out rather than using glam's swizzle traits: `engine::math` curates
/// what it re-exports (ADR-0008), and pulling `Vec3Swizzles` through it to save
/// four characters would leak glam's surface one trait at a time.
fn ground(vector: Vec3) -> Vec2 {
    Vec2::new(vector.x, vector.z)
}

/// The timing of one attack, in fixed ticks.
///
/// Authored per ability in content (`AbilityDef` carries the same three
/// windows), and copied here when an attack starts so that retuning content
/// mid-swing cannot change an attack already in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttackTiming {
    /// Ticks before the attack can hit. The window a dodge must beat.
    pub windup: u32,
    /// Ticks during which the hit volume is live.
    pub active: u32,
    /// Ticks after the active window during which the attacker is committed.
    pub recovery: u32,
}

impl AttackTiming {
    pub const fn new(windup: u32, active: u32, recovery: u32) -> Self {
        Self {
            windup,
            active,
            recovery,
        }
    }

    /// Total ticks the attacker is committed for.
    pub const fn total(self) -> u32 {
        self.windup + self.active + self.recovery
    }
}

impl Default for AttackTiming {
    /// A deliberate one-handed swing at 60 Hz: roughly 230 ms wind-up, 100 ms
    /// of live hit, 400 ms of recovery. Slow enough to read and to punish.
    fn default() -> Self {
        Self::new(14, 6, 24)
    }
}

/// Maximum fraction of damage a resistance may remove.
///
/// Not 1.0: an immune target cannot be fought, and ADR-0007's combat rests on
/// trades being real. Equipment and content can therefore make a damage type
/// cheap to take, never free.
pub const MAX_RESISTANCE: f32 = 0.75;

/// A typed quantity of damage.
///
/// `DamageType` has existed in `game_core` since M4 and never reached the
/// runtime, which is why `ember-cleave`'s `Fire` was decorative (ADR-0016).
/// This is the type that carries it across the boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Damage {
    /// Amount before the target's resistances. Never negative.
    pub amount: f32,
    pub kind: DamageType,
}

impl Damage {
    /// Builds a damage value, flooring the amount at zero.
    ///
    /// Clamped at construction rather than at application, so no path through
    /// the runtime can express healing by dealing a negative hit.
    pub fn new(amount: f32, kind: DamageType) -> Self {
        Self {
            amount: if amount.is_finite() { amount.max(0.0) } else { 0.0 },
            kind,
        }
    }
}

/// Per-kind damage reduction, as fractions in `0.0..=MAX_RESISTANCE`.
///
/// Runtime-only and `f32`, per ADR-0016: canonical `Pool` stays `u32` and
/// nothing here reaches the save envelope. Stored as a fixed array rather than
/// a map so lookup is an index, iteration order is fixed, and a resistance set
/// is `Copy` and cheap to carry on every target.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Resistances([f32; 4]);

/// Index of a damage kind within [`Resistances`].
///
/// Written as an explicit match rather than a cast so that reordering
/// `DamageType` cannot silently permute everyone's armour.
const fn slot(kind: DamageType) -> usize {
    match kind {
        DamageType::Physical => 0,
        DamageType::Fire => 1,
        DamageType::Frost => 2,
        DamageType::Corruption => 3,
    }
}

impl Resistances {
    /// No reduction of anything.
    pub const fn none() -> Self {
        Self([0.0; 4])
    }

    /// Sets one kind's reduction, clamped into `0.0..=MAX_RESISTANCE`.
    pub fn with(mut self, kind: DamageType, fraction: f32) -> Self {
        self.set(kind, fraction);
        self
    }

    /// Sets one kind's reduction, clamped. A non-finite value is treated as
    /// zero rather than propagated into every later calculation.
    pub fn set(&mut self, kind: DamageType, fraction: f32) {
        let clamped = if fraction.is_finite() {
            fraction.clamp(0.0, MAX_RESISTANCE)
        } else {
            0.0
        };
        self.0[slot(kind)] = clamped;
    }

    /// Reduction against `kind`.
    pub fn get(&self, kind: DamageType) -> f32 {
        self.0[slot(kind)]
    }

    /// Adds another set, clamping the total.
    ///
    /// Additive rather than multiplicative so that two pieces of armour cannot
    /// compound toward immunity; the clamp then holds regardless of how many
    /// sources contribute.
    pub fn combine(self, other: Self) -> Self {
        let mut total = Self::none();
        for (index, slot) in total.0.iter_mut().enumerate() {
            *slot = (self.0[index] + other.0[index]).clamp(0.0, MAX_RESISTANCE);
        }
        total
    }

    /// The amount actually dealt after this set reduces `damage`.
    pub fn mitigate(&self, damage: Damage) -> f32 {
        (damage.amount * (1.0 - self.get(damage.kind))).max(0.0)
    }
}

/// Which part of an attack is happening.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttackPhase {
    /// Telegraphed, not yet dangerous.
    Windup,
    /// The hit volume is live this tick.
    Active,
    /// Committed but harmless; the punish window.
    Recovery,
    /// Over. The attacker may act again.
    Finished,
}

/// An attack in progress.
///
/// Created when an attack starts and advanced once per fixed tick. It carries
/// its own timing so content changes cannot alter a swing already underway.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Attack {
    pub timing: AttackTiming,
    /// Damage dealt on a connecting hit, before the target's resistances.
    pub damage: f32,
    /// What kind of damage this is.
    ///
    /// `Physical` for a plain weapon swing; an ability sets its own through
    /// [`Attack::with_damage_type`]. Defaulting rather than requiring it at
    /// every construction keeps a swing's call site about timing and reach.
    pub damage_type: DamageType,
    /// How far in front of the attacker the hit volume sits.
    pub reach: f32,
    /// Radius of the hit volume.
    pub radius: f32,
    elapsed: u32,
    /// Set once the attack has connected, so one swing cannot hit twice.
    spent: bool,
}

impl Component for Attack {}

impl Attack {
    pub fn new(timing: AttackTiming, damage: f32, reach: f32, radius: f32) -> Self {
        Self {
            timing,
            damage,
            damage_type: DamageType::Physical,
            reach,
            radius,
            elapsed: 0,
            spent: false,
        }
    }

    /// Retypes the damage this attack deals.
    ///
    /// Consuming builder rather than a fifth constructor argument: an attack's
    /// kind is the exception, not the rule, and four positional floats are
    /// already one too many to read at a glance.
    pub fn with_damage_type(mut self, kind: DamageType) -> Self {
        self.damage_type = kind;
        self
    }

    pub fn phase(&self) -> AttackPhase {
        let timing = self.timing;
        if self.elapsed < timing.windup {
            AttackPhase::Windup
        } else if self.elapsed < timing.windup + timing.active {
            AttackPhase::Active
        } else if self.elapsed < timing.total() {
            AttackPhase::Recovery
        } else {
            AttackPhase::Finished
        }
    }

    /// Whether the hit volume is live and has not already connected.
    ///
    /// A single swing hits at most once: without this, a six-tick active window
    /// would deal its damage six times.
    pub fn can_hit(&self) -> bool {
        self.phase() == AttackPhase::Active && !self.spent
    }

    pub fn is_finished(&self) -> bool {
        self.phase() == AttackPhase::Finished
    }

    /// Whether the attacker is locked into this attack and cannot act.
    pub fn is_committed(&self) -> bool {
        !self.is_finished()
    }

    /// Marks the swing as having connected.
    pub fn mark_spent(&mut self) {
        self.spent = true;
    }

    /// Advances one fixed tick.
    pub fn tick(&mut self) {
        if !self.is_finished() {
            self.elapsed += 1;
        }
    }

    pub fn elapsed(&self) -> u32 {
        self.elapsed
    }

    /// Centre of the hit volume for an attacker at `origin` facing `facing`.
    ///
    /// Flattened to the ground plane: the slice has no vertical aiming, and a
    /// hit volume that drifted with a slope would be invisible and unfair.
    pub fn hit_centre(&self, origin: Vec3, facing: Vec3) -> Vec3 {
        let forward = ground(facing).normalize_or_zero();
        origin + Vec3::new(forward.x, 0.0, forward.y) * self.reach
    }
}

/// Runtime health. Durable *values* live in canonical state; this is the
/// working copy combat mutates between safe points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Health {
    pub current: f32,
    pub max: f32,
}

impl Component for Health {}

impl Health {
    pub fn full(max: f32) -> Self {
        Self { current: max, max }
    }

    pub fn is_dead(&self) -> bool {
        self.current <= 0.0
    }

    /// Applies damage, clamped so health never goes negative.
    ///
    /// Returns whether this blow killed — the moment that becomes a domain
    /// event, unlike the damage itself.
    pub fn apply_damage(&mut self, amount: f32) -> bool {
        if self.is_dead() {
            return false;
        }
        self.current = (self.current - amount.max(0.0)).max(0.0);
        self.is_dead()
    }

    pub fn heal(&mut self, amount: f32) {
        if !self.is_dead() {
            self.current = (self.current + amount.max(0.0)).min(self.max);
        }
    }
}

/// Whether a hit volume overlaps a target capsule.
///
/// Both are treated as circles on the ground plane plus a vertical span, which
/// is what the slice's flat spaces need and is cheap enough to run for every
/// attacker against every target.
pub fn hit_connects(centre: Vec3, radius: f32, target: Capsule, target_position: Vec3) -> bool {
    let vertical_gap = (centre.y - target_position.y).abs();
    if vertical_gap > target.half_extent() + radius {
        return false;
    }
    let planar = (ground(centre) - ground(target_position)).length();
    planar <= radius + target.radius
}

/// What the player's equipment contributes to combat.
///
/// Derived on demand from canonical state plus content, never stored: the
/// durable fact is *which item is in which slot* (`PlayerState::equipment`),
/// and deriving the numbers from it means retuning an item's stats in content
/// takes effect without a migration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Loadout {
    /// Damage added to the wielder's attacks.
    pub bonus_damage: f32,
    /// Retypes the wielder's attacks when set. Only the weapon slot may do
    /// this: armour that turned your sword into a torch would be a surprise.
    pub damage_type: Option<DamageType>,
    /// Combined reduction from everything worn.
    pub resistances: Resistances,
}

impl Default for Loadout {
    /// Bare-handed: no bonus, no retyping, no protection.
    fn default() -> Self {
        Self {
            bonus_damage: 0.0,
            damage_type: None,
            resistances: Resistances::none(),
        }
    }
}

/// Resolves what the player is wearing into combat terms.
///
/// Takes a lookup rather than a content registry so this stays a pure function
/// of canonical state plus content: it is exhaustively testable without
/// building a registry, and `combat` does not need to know `content_runtime`'s
/// types (ADR-0002's dependency direction).
///
/// Iteration follows `PlayerState::equipment`, a `BTreeMap` keyed by slot, so
/// the result does not depend on the order things were equipped in.
pub fn resolve_loadout<'a>(player: &PlayerState, lookup: impl Fn(&ItemId) -> Option<&'a ItemDef>) -> Loadout {
    let mut loadout = Loadout::default();
    for (slot, item) in &player.equipment {
        let Some(stats) = lookup(item).and_then(|def| def.combat.as_ref()) else {
            continue;
        };
        loadout.bonus_damage += crate::ability::units(stats.damage).max(0.0);
        if *slot == EquipmentSlot::Weapon
            && let Some(kind) = stats.damage_type
        {
            loadout.damage_type = Some(kind);
        }
        for (kind, fraction) in &stats.resistances {
            loadout.resistances = loadout
                .resistances
                .combine(Resistances::none().with(*kind, crate::ability::units(*fraction)));
        }
    }
    loadout
}

/// One resolved hit, ready for the caller to turn into consequences.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub target: Entity,
    /// Damage actually dealt, after resistances and any status multiplier —
    /// not what the attack nominally carries.
    pub damage: f32,
    /// The kind that landed, so a caller can react to fire differently from a
    /// sword without re-deriving it from the attacker.
    pub kind: DamageType,
    /// Whether the blow killed, which is the part that becomes a domain event.
    pub lethal: bool,
}

/// Why an attack did not land.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissReason {
    /// The hit volume is not live this tick.
    NotActive,
    /// The swing already connected.
    AlreadySpent,
    /// Nothing was in range.
    OutOfRange,
    /// The target was mid-dodge (ADR-0007's i-frames).
    Invulnerable,
}

/// Who is swinging, and from where.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Attacker<'a> {
    pub attack: &'a Attack,
    pub position: Vec3,
    pub facing: Vec3,
}

/// Who might be hit.
///
/// Grouped rather than passed as loose parameters because two bare `Vec3`s in a
/// row is an invitation to swap attacker and target positions — a bug that
/// compiles, runs, and produces plausible-looking misses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Target {
    pub entity: Entity,
    pub capsule: Capsule,
    pub position: Vec3,
    /// Mid-dodge, so i-frames apply (ADR-0007).
    pub invulnerable: bool,
    /// Per-kind reduction, from equipment and elsewhere.
    pub resistances: Resistances,
    /// Multiplier from active statuses, where below 1.0 is protection.
    ///
    /// Carried as a resolved number rather than as a `Statuses` reference so
    /// this stays a plain value type and the rule below stays pure; the caller
    /// asks `Statuses::damage_taken_multiplier` once and passes the answer.
    pub damage_taken: f32,
}

impl Target {
    /// An unprotected target: no resistances, no status modifiers.
    pub fn new(entity: Entity, capsule: Capsule, position: Vec3, invulnerable: bool) -> Self {
        Self {
            entity,
            capsule,
            position,
            invulnerable,
            resistances: Resistances::none(),
            damage_taken: 1.0,
        }
    }

    /// Gives the target a resistance set.
    pub fn with_resistances(mut self, resistances: Resistances) -> Self {
        self.resistances = resistances;
        self
    }

    /// Gives the target a status multiplier on incoming damage.
    pub fn with_damage_taken(mut self, multiplier: f32) -> Self {
        self.damage_taken = if multiplier.is_finite() {
            multiplier.max(0.0)
        } else {
            1.0
        };
        self
    }
}

/// Resolves one attacker's swing against one candidate target.
///
/// Deliberately small and pure: the caller owns the ECS queries and decides what
/// the outcome means. Keeping the rule itself free of world access is what lets
/// it be tested exhaustively without building a scene.
pub fn resolve_hit(attacker: Attacker<'_>, target: Target, health: &mut Health) -> Result<Hit, MissReason> {
    let attack = attacker.attack;
    if attack.phase() != AttackPhase::Active {
        return Err(MissReason::NotActive);
    }
    if !attack.can_hit() {
        return Err(MissReason::AlreadySpent);
    }
    let centre = attack.hit_centre(attacker.position, attacker.facing);
    if !hit_connects(centre, attack.radius, target.capsule, target.position) {
        return Err(MissReason::OutOfRange);
    }
    // Checked last so a dodge through a swing still counts as *in range* — the
    // distinction matters for feedback and for tuning the i-frame window.
    if target.invulnerable {
        return Err(MissReason::Invulnerable);
    }

    // Status multiplier first, then resistances. The order is deliberate and
    // load-bearing: resistances clamp at `MAX_RESISTANCE`, so applying them
    // last means no stack of buffs can push mitigation past that ceiling.
    let incoming = Damage::new(attack.damage * target.damage_taken, attack.damage_type);
    let dealt = target.resistances.mitigate(incoming);
    let lethal = health.apply_damage(dealt);
    Ok(Hit {
        target: target.entity,
        damage: dealt,
        kind: incoming.kind,
        lethal,
    })
}

/// How long a broken stance holds a body, in ticks: about 0.4 s at 60 Hz.
///
/// Long enough to punish, short enough that a staggered body is back in the
/// fight before the next blow of a combo lands.
pub const STAGGER_TICKS: u64 = 24;

/// How long after a stagger ends before poise can break again, in ticks.
///
/// The anti-stunlock bound. However fast the hits come, a body spends at most
/// `STAGGER_TICKS` of every `STAGGER_TICKS + STAGGER_GRACE_TICKS` staggered,
/// and always gets this long to act — to roll, swing, or run — before it can
/// be broken again.
pub const STAGGER_GRACE_TICKS: u64 = 45;

/// How long poise takes to refill from empty when nothing is hitting it, in
/// ticks.
pub const POISE_REFILL_TICKS: f32 = 150.0;

/// How much punishment a body absorbs before its stance breaks (M8).
///
/// Being hit costs poise equal to the damage it dealt. At zero the body
/// **staggers**: a blow it was still winding up is cancelled, and it cannot
/// act for [`STAGGER_TICKS`]. Its poise then refills at once, and cannot break
/// again for [`STAGGER_GRACE_TICKS`] after the stagger — which is what makes a
/// stunlock impossible rather than merely unlikely. Poise also creeps back
/// while nothing hits it, so light hits spread out never add up to a stagger.
///
/// This is what makes a trade cost something: a husk that swings into your
/// blade loses its swing, and a warden with deep poise does not.
///
/// Runtime state, like health: nothing here is saved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Poise {
    pub max: f32,
    pub current: f32,
    /// The tick the current stagger ends; in the past when not staggered.
    staggered_until: u64,
    /// The tick poise may break again.
    breakable_from: u64,
}

impl Component for Poise {}

impl Poise {
    pub fn new(max: f32) -> Self {
        let max = if max.is_finite() { max.max(0.0) } else { 0.0 };
        Self {
            max,
            current: max,
            staggered_until: 0,
            breakable_from: 0,
        }
    }

    /// Takes a hit of `damage` at `tick`. Returns whether this hit broke the
    /// stance.
    pub fn take(&mut self, damage: f32, tick: u64) -> bool {
        if !damage.is_finite() || damage <= 0.0 || tick < self.breakable_from {
            return false;
        }
        self.current -= damage;
        if self.current > 0.0 {
            return false;
        }
        self.current = self.max;
        self.staggered_until = tick + STAGGER_TICKS;
        self.breakable_from = self.staggered_until + STAGGER_GRACE_TICKS;
        true
    }

    /// Whether the body is staggered at `tick`.
    pub fn is_staggered(&self, tick: u64) -> bool {
        tick < self.staggered_until
    }

    /// One tick of recovery.
    pub fn tick(&mut self) {
        self.current = (self.current + self.max / POISE_REFILL_TICKS).min(self.max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::ecs::World;

    fn timing() -> AttackTiming {
        AttackTiming::new(3, 2, 4)
    }

    fn swing() -> Attack {
        Attack::new(timing(), 25.0, 1.5, 0.8)
    }

    fn target_entity() -> Entity {
        let mut world = World::new();
        world.spawn()
    }

    #[test]
    fn phases_run_in_order_and_end() {
        let mut attack = swing();
        let expected = [
            AttackPhase::Windup,
            AttackPhase::Windup,
            AttackPhase::Windup,
            AttackPhase::Active,
            AttackPhase::Active,
            AttackPhase::Recovery,
            AttackPhase::Recovery,
            AttackPhase::Recovery,
            AttackPhase::Recovery,
        ];
        for (tick, phase) in expected.into_iter().enumerate() {
            assert_eq!(attack.phase(), phase, "wrong phase at tick {tick}");
            assert!(attack.is_committed(), "the attacker is locked in at tick {tick}");
            attack.tick();
        }
        assert_eq!(attack.phase(), AttackPhase::Finished);
        assert!(!attack.is_committed());
    }

    #[test]
    fn a_finished_attack_stops_advancing() {
        let mut attack = swing();
        for _ in 0..50 {
            attack.tick();
        }
        assert_eq!(attack.elapsed(), timing().total(), "elapsed must not run away");
    }

    /// A six-tick active window must not deal its damage six times.
    #[test]
    fn one_swing_connects_at_most_once() {
        let mut attack = swing();
        for _ in 0..timing().windup {
            attack.tick();
        }
        assert!(attack.can_hit());
        attack.mark_spent();
        assert!(!attack.can_hit(), "a spent swing cannot connect again");
        attack.tick();
        assert_eq!(attack.phase(), AttackPhase::Active);
        assert!(!attack.can_hit(), "still spent on the next active tick");
    }

    #[test]
    fn the_hit_volume_sits_in_front_of_the_attacker() {
        let attack = swing();
        let centre = attack.hit_centre(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(
            (centre.z + 1.5).abs() < 1e-5,
            "reach should extend along -Z, got {centre:?}"
        );
        assert!(
            (centre.y - 1.0).abs() < 1e-5,
            "the volume stays at the attacker's height"
        );
    }

    #[test]
    fn a_facing_of_zero_does_not_produce_nan() {
        let attack = swing();
        let centre = attack.hit_centre(Vec3::new(2.0, 1.0, 3.0), Vec3::ZERO);
        assert!(centre.is_finite());
        assert_eq!(centre, Vec3::new(2.0, 1.0, 3.0));
    }

    #[test]
    fn reach_decides_what_connects() {
        let capsule = Capsule::new(0.5, 1.0);
        let centre = Vec3::new(0.0, 1.0, 0.0);
        assert!(hit_connects(centre, 0.8, capsule, Vec3::new(1.0, 1.0, 0.0)));
        assert!(
            !hit_connects(centre, 0.8, capsule, Vec3::new(3.0, 1.0, 0.0)),
            "a target well beyond reach is not hit"
        );
        assert!(
            !hit_connects(centre, 0.8, capsule, Vec3::new(0.5, 9.0, 0.0)),
            "and neither is one far above it"
        );
    }

    // -----------------------------------------------------------------------
    // Typed damage, resistances, and equipment (M6 wave 2, tasks D and F).
    // -----------------------------------------------------------------------

    fn item(id: &str, combat: Option<game_core::content::CombatStats>) -> ItemDef {
        ItemDef {
            schema_version: 1,
            id: ItemId::new(id).expect("valid id"),
            name: id.to_owned(),
            category: game_core::content::ItemCategory::Weapon,
            stack_limit: 1,
            combat,
            on_use: None,
        }
    }

    #[test]
    fn damage_is_floored_at_construction() {
        assert!((Damage::new(-5.0, DamageType::Fire).amount - 0.0).abs() < 1e-6);
        assert!((Damage::new(f32::NAN, DamageType::Fire).amount - 0.0).abs() < 1e-6);
        assert!((Damage::new(3.5, DamageType::Frost).amount - 3.5).abs() < 1e-6);
    }

    /// The bound that keeps a fight winnable: armour reduces, never nullifies.
    #[test]
    fn resistances_cannot_reach_immunity() {
        let absurd = Resistances::none().with(DamageType::Fire, 5.0);
        assert!((absurd.get(DamageType::Fire) - MAX_RESISTANCE).abs() < 1e-6);

        let dealt = absurd.mitigate(Damage::new(100.0, DamageType::Fire));
        assert!(dealt > 0.0, "a resisted hit still hurts, got {dealt}");
        assert!((dealt - 25.0).abs() < 1e-4);

        // Stacking sources cannot beat the ceiling either.
        let stacked = Resistances::none()
            .with(DamageType::Fire, 0.7)
            .combine(Resistances::none().with(DamageType::Fire, 0.7));
        assert!((stacked.get(DamageType::Fire) - MAX_RESISTANCE).abs() < 1e-6);

        // And a negative or non-finite authored value is treated as none.
        assert!(
            (Resistances::none()
                .with(DamageType::Fire, -2.0)
                .get(DamageType::Fire))
            .abs()
                < 1e-6
        );
        assert!(
            (Resistances::none()
                .with(DamageType::Fire, f32::INFINITY)
                .get(DamageType::Fire))
            .abs()
                < 1e-6
        );
    }

    #[test]
    fn resistance_applies_only_to_its_own_kind() {
        let fireproof = Resistances::none().with(DamageType::Fire, 0.5);
        assert!((fireproof.mitigate(Damage::new(10.0, DamageType::Fire)) - 5.0).abs() < 1e-6);
        assert!(
            (fireproof.mitigate(Damage::new(10.0, DamageType::Frost)) - 10.0).abs() < 1e-6,
            "frost is not resisted by fire protection"
        );
    }

    /// `ember-cleave`'s `Fire` reaching the runtime is the point of ADR-0016.
    #[test]
    fn a_typed_hit_is_resisted_by_the_matching_kind() {
        let mut attack = swing().with_damage_type(DamageType::Fire);
        for _ in 0..timing().windup {
            attack.tick();
        }
        let mut health = Health::full(100.0);
        let hit = resolve_hit(
            Attacker {
                attack: &attack,
                position: Vec3::new(0.0, 1.0, 0.0),
                facing: Vec3::new(0.0, 0.0, -1.0),
            },
            Target::new(
                target_entity(),
                Capsule::new(0.5, 1.0),
                Vec3::new(0.0, 1.0, -1.8),
                false,
            )
            .with_resistances(Resistances::none().with(DamageType::Fire, 0.5)),
            &mut health,
        )
        .expect("in range");

        assert_eq!(hit.kind, DamageType::Fire);
        assert!(
            (hit.damage - 12.5).abs() < 1e-4,
            "25 fire against 50% fire resistance is 12.5, got {}",
            hit.damage
        );
        assert!((health.current - 87.5).abs() < 1e-4);
    }

    #[test]
    fn a_status_multiplier_scales_what_lands() {
        let mut attack = swing();
        for _ in 0..timing().windup {
            attack.tick();
        }
        let mut health = Health::full(100.0);
        let hit = resolve_hit(
            Attacker {
                attack: &attack,
                position: Vec3::new(0.0, 1.0, 0.0),
                facing: Vec3::new(0.0, 0.0, -1.0),
            },
            Target::new(
                target_entity(),
                Capsule::new(0.5, 1.0),
                Vec3::new(0.0, 1.0, -1.8),
                false,
            )
            .with_damage_taken(0.5),
            &mut health,
        )
        .expect("in range");

        assert!(
            (hit.damage - 12.5).abs() < 1e-4,
            "warden-guard halves a 25-damage swing, got {}",
            hit.damage
        );
    }

    #[test]
    fn an_unequipped_player_has_a_bare_loadout() {
        let loadout = resolve_loadout(&PlayerState::default(), |_| None);
        assert_eq!(loadout, Loadout::default());
        assert!((loadout.bonus_damage - 0.0).abs() < 1e-6);
        assert_eq!(loadout.damage_type, None);
    }

    /// Task F's exit criterion: equipping a weapon changes damage, equipping
    /// armour changes what is taken.
    #[test]
    fn equipment_changes_damage_dealt_and_taken() {
        use game_core::content::{CombatStats, Fixed};
        use std::collections::BTreeMap;

        let glaive = item(
            "ember-glaive",
            Some(CombatStats {
                damage: Fixed::from_milli(4_000),
                damage_type: Some(DamageType::Fire),
                resistances: BTreeMap::new(),
            }),
        );
        let mail = item(
            "scorched-mail",
            Some(CombatStats {
                damage: Fixed::ZERO,
                // Armour may not retype attacks; this is set to prove it is
                // ignored outside the weapon slot.
                damage_type: Some(DamageType::Frost),
                resistances: [(DamageType::Fire, Fixed::from_milli(250))].into_iter().collect(),
            }),
        );

        let mut player = PlayerState::default();
        player.equipment.insert(EquipmentSlot::Weapon, glaive.id.clone());
        player.equipment.insert(EquipmentSlot::Armor, mail.id.clone());

        let loadout = resolve_loadout(&player, |id| {
            [&glaive, &mail].into_iter().find(|def| &def.id == id)
        });

        assert!(
            (loadout.bonus_damage - 4.0).abs() < 1e-6,
            "the glaive adds its 4000 milli-units"
        );
        assert_eq!(
            loadout.damage_type,
            Some(DamageType::Fire),
            "the weapon retypes attacks; the armour must not"
        );
        assert!((loadout.resistances.get(DamageType::Fire) - 0.25).abs() < 1e-6);
        assert!((loadout.resistances.get(DamageType::Frost) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn an_item_without_combat_stats_contributes_nothing() {
        let ore = item("ember-ore", None);
        let mut player = PlayerState::default();
        player.equipment.insert(EquipmentSlot::Weapon, ore.id.clone());

        let loadout = resolve_loadout(&player, |id| (id == &ore.id).then_some(&ore));
        assert_eq!(loadout, Loadout::default(), "a material is not a weapon");
    }

    /// An equipped item content no longer defines must not panic or poison the
    /// loadout — content can be retired while a save still names it.
    #[test]
    fn an_unknown_equipped_item_is_skipped() {
        let mut player = PlayerState::default();
        player.equipment.insert(
            EquipmentSlot::Weapon,
            ItemId::new("deleted-sword").expect("valid id"),
        );
        assert_eq!(resolve_loadout(&player, |_| None), Loadout::default());
    }

    #[test]
    fn damage_clamps_and_reports_the_killing_blow() {
        let mut health = Health::full(30.0);
        assert!(!health.apply_damage(10.0));
        assert!((health.current - 20.0).abs() < 1e-6);

        assert!(
            health.apply_damage(100.0),
            "the blow that empties health is lethal"
        );
        assert!((health.current - 0.0).abs() < 1e-6, "health never goes negative");
        assert!(health.is_dead());

        assert!(
            !health.apply_damage(10.0),
            "hitting a corpse is not a second kill"
        );
    }

    #[test]
    fn healing_respects_the_ceiling_and_the_dead() {
        let mut health = Health::full(50.0);
        health.apply_damage(20.0);
        health.heal(100.0);
        assert!((health.current - 50.0).abs() < 1e-6);

        health.apply_damage(50.0);
        health.heal(10.0);
        assert!(health.is_dead(), "healing does not resurrect");
    }

    #[test]
    fn a_swing_lands_on_a_target_in_front() {
        let mut attack = swing();
        for _ in 0..timing().windup {
            attack.tick();
        }
        let mut health = Health::full(100.0);
        let hit = resolve_hit(
            Attacker {
                attack: &attack,
                position: Vec3::new(0.0, 1.0, 0.0),
                facing: Vec3::new(0.0, 0.0, -1.0),
            },
            Target::new(
                target_entity(),
                Capsule::new(0.5, 1.0),
                Vec3::new(0.0, 1.0, -1.8),
                false,
            ),
            &mut health,
        )
        .expect("the target is in front and in reach");

        assert!((hit.damage - 25.0).abs() < 1e-6);
        assert!(!hit.lethal);
        assert!((health.current - 75.0).abs() < 1e-6);
    }

    #[test]
    fn misses_explain_themselves() {
        let capsule = Capsule::new(0.5, 1.0);
        let in_front = Vec3::new(0.0, 1.0, -1.8);
        let mut health = Health::full(100.0);

        let attempt = |attack: &Attack, position: Vec3, invulnerable: bool, health: &mut Health| {
            resolve_hit(
                Attacker {
                    attack,
                    position: Vec3::new(0.0, 1.0, 0.0),
                    facing: Vec3::new(0.0, 0.0, -1.0),
                },
                Target::new(target_entity(), capsule, position, invulnerable),
                health,
            )
        };

        let winding_up = swing();
        assert_eq!(
            attempt(&winding_up, in_front, false, &mut health),
            Err(MissReason::NotActive)
        );

        let mut active = swing();
        for _ in 0..timing().windup {
            active.tick();
        }
        assert_eq!(
            attempt(&active, Vec3::new(0.0, 1.0, -8.0), false, &mut health),
            Err(MissReason::OutOfRange)
        );
        assert_eq!(
            attempt(&active, in_front, true, &mut health),
            Err(MissReason::Invulnerable),
            "a dodge through the swing is an i-frame miss, not a range miss"
        );

        let mut spent = active;
        spent.mark_spent();
        assert_eq!(
            attempt(&spent, in_front, false, &mut health),
            Err(MissReason::AlreadySpent)
        );

        assert!((health.current - 100.0).abs() < 1e-6, "no miss may deal damage");
    }

    /// The timing that makes the combat readable: a dodge started before the
    /// active window and lasting through it must beat the attack.
    #[test]
    fn a_well_timed_dodge_beats_a_swing() {
        use crate::movement::Dodge;
        use engine::math::Vec2;

        let mut attack = swing();
        let mut dodge = Dodge::new(Vec2::new(0.0, 1.0), 8, 6, 0.4);
        let mut health = Health::full(100.0);
        let mut landed = false;

        for _ in 0..timing().total() {
            if attack.can_hit() {
                let outcome = resolve_hit(
                    Attacker {
                        attack: &attack,
                        position: Vec3::new(0.0, 1.0, 0.0),
                        facing: Vec3::new(0.0, 0.0, -1.0),
                    },
                    Target::new(
                        target_entity(),
                        Capsule::new(0.5, 1.0),
                        Vec3::new(0.0, 1.0, -1.8),
                        dodge.is_invulnerable(),
                    ),
                    &mut health,
                );
                landed |= outcome.is_ok();
            }
            attack.tick();
            dodge.advance();
        }

        assert!(
            !landed,
            "i-frames covering the active window must prevent the hit"
        );
        assert!((health.current - 100.0).abs() < 1e-6);
    }

    #[test]
    fn a_late_dodge_does_not_save_you() {
        use crate::movement::Dodge;
        use engine::math::Vec2;

        let mut attack = swing();
        // Started too late: invulnerability begins after the active window.
        let mut dodge = Dodge::new(Vec2::new(0.0, 1.0), 8, 0, 0.4);
        let mut health = Health::full(100.0);

        for _ in 0..timing().total() {
            if attack.can_hit()
                && resolve_hit(
                    Attacker {
                        attack: &attack,
                        position: Vec3::new(0.0, 1.0, 0.0),
                        facing: Vec3::new(0.0, 0.0, -1.0),
                    },
                    Target::new(
                        target_entity(),
                        Capsule::new(0.5, 1.0),
                        Vec3::new(0.0, 1.0, -1.8),
                        dodge.is_invulnerable(),
                    ),
                    &mut health,
                )
                .is_ok()
            {
                attack.mark_spent();
            }
            attack.tick();
            dodge.advance();
        }

        assert!((health.current - 75.0).abs() < 1e-6, "the swing should land once");
    }

    mod poise {
        use super::*;

        #[test]
        fn a_hit_under_poise_does_not_stagger() {
            let mut poise = Poise::new(20.0);
            assert!(!poise.take(8.0, 10));
            assert!(!poise.is_staggered(11));
            assert!((poise.current - 12.0).abs() < 1e-6);
        }

        #[test]
        fn breaking_poise_staggers_for_a_bounded_time() {
            let mut poise = Poise::new(20.0);
            assert!(poise.take(22.0, 100), "one heavy blow breaks a light stance");
            assert!(poise.is_staggered(100));
            assert!(poise.is_staggered(100 + STAGGER_TICKS - 1));
            assert!(
                !poise.is_staggered(100 + STAGGER_TICKS),
                "and the stagger ends on time"
            );
            assert!((poise.current - 20.0).abs() < 1e-6, "poise is whole again");
        }

        /// The anti-stunlock bound: hits every tick for ten seconds, and the
        /// body is still free to act most of the time, never staggered twice
        /// without a full grace window between.
        #[test]
        fn a_hit_every_tick_cannot_stunlock() {
            let mut poise = Poise::new(5.0);
            let mut staggered = 0;
            let mut last_stagger_end: Option<u64> = None;
            for tick in 0..600 {
                if poise.take(100.0, tick) {
                    if let Some(end) = last_stagger_end {
                        assert!(
                            tick >= end + STAGGER_GRACE_TICKS,
                            "broken again at {tick}, only {} ticks after the last stagger",
                            tick - end
                        );
                    }
                    last_stagger_end = Some(tick + STAGGER_TICKS);
                }
                if poise.is_staggered(tick) {
                    staggered += 1;
                }
                poise.tick();
            }
            let bound = 600 * STAGGER_TICKS / (STAGGER_TICKS + STAGGER_GRACE_TICKS) + STAGGER_TICKS;
            assert!(
                staggered <= bound,
                "staggered {staggered} of 600 ticks, bound {bound}"
            );
            assert!(staggered > 0, "it did stagger, or this proves nothing");
        }

        #[test]
        fn poise_creeps_back_between_hits() {
            let mut poise = Poise::new(30.0);
            poise.take(20.0, 0);
            for _ in 0..(POISE_REFILL_TICKS as u32) {
                poise.tick();
            }
            assert!(
                (poise.current - 30.0).abs() < 1e-3,
                "full again after a quiet spell"
            );
            assert!(!poise.take(20.0, 200), "so the same hit does not break it");
        }

        #[test]
        fn nonsense_damage_does_nothing() {
            let mut poise = Poise::new(10.0);
            for bad in [0.0, -5.0, f32::NAN, f32::INFINITY] {
                assert!(!poise.take(bad, 0));
            }
            assert!((poise.current - 10.0).abs() < 1e-6);
            assert!(!Poise::new(f32::NAN).max.is_nan());
        }
    }
}
