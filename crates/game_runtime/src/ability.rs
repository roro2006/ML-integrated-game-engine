//! Ability activation, targeting, and the effect executor.
//!
//! This is the module ADR-0016 says the safety story rests on. The `Effect`
//! allowlist has been parsed, validated, reference-resolved, and hashed since
//! M4 — and never executed, which means ADR-0010's claim that a hallucinating
//! model can do no worse than propose an invalid instance has been untestable
//! by construction. Nothing executed the allowlist, so nothing proved the
//! allowlist bounded anything.
//!
//! # Every effect is bounded by construction
//!
//! Each primitive has a limit that does not depend on the content being
//! sensible:
//!
//! - [`Effect::SpawnEntity`] is capped by [`MAX_SPAWNS_PER_LOCATION`] as well as
//!   its own `count`, so `call-husks` on a loop is a no-op rather than an
//!   entity leak.
//! - [`Effect::Teleport`] is clamped to `max_distance` *and* swept through
//!   [`CollisionWorld`], so it cannot become a wall-clip.
//! - [`Effect::ApplyStatus`] is clamped to [`MAX_STATUS_TICKS`] and carries an
//!   absolute expiry tick, so a status cannot outlive its duration.
//! - Area targeting is capped at [`MAX_TARGETS`], so one effect cannot scale
//!   with how many things happen to be standing nearby.
//! - [`Effect::Projectile`] is clamped in speed, range, radius and pierce,
//!   capped at [`MAX_PROJECTILES_PER_LOCATION`] in flight, and carries only
//!   [`Effect::allowed_on_hit`] effects, so it cannot seed another
//!   (see [`crate::projectile`]).
//!
//! These bounds are the whole reason the allowlist is safe to hand to a
//! generator later, so each has its own test that fails loudly.
//!
//! # The executor returns outcomes; the caller applies them
//!
//! Same shape as [`resolve_hit`](crate::combat::resolve_hit). A pure rule is
//! exhaustively testable without building a scene, and it keeps the ECS queries
//! in one place. Nothing here mutates the world, and nothing here writes
//! canonical state: cooldowns, active statuses, and in-flight ability state are
//! runtime state, rebuilt from content on load exactly like attack phase and
//! current stamina already are.
//!
//! # Ticks, never seconds
//!
//! `cooldown_ticks`, `windup_ticks`, and `recovery_ticks` are authored in fixed
//! simulation ticks and are used as authored. Converting to seconds anywhere in
//! here would make ADR-0007's committed-action timing depend on frame rate,
//! which is the bug this project keeps designing away from.

use std::collections::BTreeMap;

use engine::ecs::{Component, Entity};
use engine::math::{Vec2, Vec3};
use game_core::content::{
    AbilityDef, DamageType, Effect, Fixed, GameEventId, ResourceType, StatusId, Targeting,
};
use game_core::ids::{AbilityId, PrefabId};

use crate::combat::AttackTiming;
use crate::movement::{Capsule, CollisionWorld};
use crate::projectile::{Launch, MAX_PROJECTILES_PER_LOCATION};

/// Maximum entities one location may hold from [`Effect::SpawnEntity`].
///
/// Content validation already rejects a zero `count`, and `count` is a `u8`, so
/// a single cast is bounded by the schema. This bounds the *accumulation*: an
/// ability on a 600-tick cooldown summoning two husks is fine, and the same
/// ability reached by a generator that did not think about repetition is not.
/// The cap belongs to the executor rather than the validator because it is a
/// property of the live location, which no content file can see.
pub const MAX_SPAWNS_PER_LOCATION: usize = 24;

/// Maximum duration, in ticks, that [`Effect::ApplyStatus`] may request.
///
/// One minute at 60 Hz. Long enough for anything the slice authors —
/// `warden-guard` asks for 180 — and short enough that a status cannot become
/// permanent through a large authored number.
pub const MAX_STATUS_TICKS: u32 = 3_600;

/// Maximum entities one effect may touch.
///
/// Area targeting would otherwise scale with how crowded a room happens to be,
/// which makes an ability's cost unpredictable and its worst case unbounded.
pub const MAX_TARGETS: usize = 16;

/// Ticks during which an ability's effects are live.
///
/// [`AbilityDef`] authors a wind-up and a recovery but no active window: an
/// ability's effects land on one tick, at the end of its wind-up. Attacks carry
/// a multi-tick active window because a swing sweeps through space; an effect
/// resolves at an instant, so widening this would let one cast fire twice.
const ABILITY_ACTIVE_TICKS: u32 = 1;

/// A vector flattened to the ground plane.
///
/// Written out rather than using glam's swizzle traits, for the reason
/// `combat::ground` gives: `engine::math` curates what it re-exports (ADR-0008).
fn ground(vector: Vec3) -> Vec2 {
    Vec2::new(vector.x, vector.z)
}

/// Converts an authored fixed-point value to the runtime's `f32`.
///
/// Content is fixed-point so that comparison and replay are exact
/// (`game_core::content`); the runtime is `f32` because `Health` and `Stamina`
/// are. This is the single boundary between them for abilities.
///
/// `sample_game::slice::metres` performs the same conversion under a name that
/// suits positions. Unifying them is wave 3's job, when the slice starts
/// calling into here; duplicating the arithmetic in the meantime is how two
/// conventions drift apart.
pub fn units(value: Fixed) -> f32 {
    value.as_milli() as f32 / 1000.0
}

/// Per-entity, per-ability cooldowns, counted in fixed ticks.
///
/// A component rather than a global table because two husks must cool down
/// independently: a shared table would let one enemy's cast silence the other.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Cooldowns {
    /// Ticks remaining per ability. Keyed by [`AbilityId`] in a `BTreeMap` so
    /// iteration is deterministic, which replay equality depends on.
    remaining: BTreeMap<AbilityId, u32>,
}

impl Component for Cooldowns {}

impl Cooldowns {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `ability` may be cast this tick.
    pub fn is_ready(&self, ability: &AbilityId) -> bool {
        self.remaining_for(ability) == 0
    }

    /// Ticks left before `ability` is ready, zero when it already is.
    pub fn remaining_for(&self, ability: &AbilityId) -> u32 {
        self.remaining.get(ability).copied().unwrap_or(0)
    }

    /// Puts `ability` on cooldown for `ticks`.
    ///
    /// A zero-tick cooldown is stored as absent rather than as zero, so the map
    /// holds only what is actually cooling down.
    pub fn start(&mut self, ability: AbilityId, ticks: u32) {
        if ticks == 0 {
            self.remaining.remove(&ability);
        } else {
            self.remaining.insert(ability, ticks);
        }
    }

    /// Advances one fixed tick, dropping anything that has come off cooldown.
    pub fn tick(&mut self) {
        self.remaining.retain(|_, ticks| {
            *ticks = ticks.saturating_sub(1);
            *ticks > 0
        });
    }

    /// How many abilities are currently cooling down.
    pub fn len(&self) -> usize {
        self.remaining.len()
    }

    pub fn is_empty(&self) -> bool {
        self.remaining.is_empty()
    }
}

/// Why an ability did not activate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// Still cooling down, with the ticks left to wait.
    OnCooldown { remaining: u32 },
    /// The actor cannot pay the cost. Reported rather than partially spent: an
    /// ability the player cannot afford must not half-happen, exactly as
    /// `Stamina::try_spend` guarantees for dodges.
    NotEnoughStamina,
    /// The actor is committed to an attack or another ability (ADR-0007).
    Committed,
}

/// An ability that passed its gates and is now committed.
///
/// Carries its own timing, copied at activation, so retuning content mid-cast
/// cannot change a cast already in flight — the same rule
/// [`crate::combat::Attack`] follows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Activation {
    pub ability: AbilityId,
    pub timing: AttackTiming,
    /// Cooldown to start once the cast commits, in ticks.
    pub cooldown_ticks: u32,
}

/// Checks an ability's gates and, if they pass, spends its cost.
///
/// The stamina spend happens here rather than in the caller because "can afford"
/// and "has paid" must not be two separate decisions with a tick between them.
/// Cooldown is *not* started here: it starts when the cast commits, which is the
/// caller's tick boundary.
pub fn activate(
    def: &AbilityDef,
    cooldowns: &Cooldowns,
    stamina: &mut crate::movement::Stamina,
    committed: bool,
) -> Result<Activation, Refusal> {
    if committed {
        return Err(Refusal::Committed);
    }
    let remaining = cooldowns.remaining_for(&def.id);
    if remaining > 0 {
        return Err(Refusal::OnCooldown { remaining });
    }
    if !stamina.try_spend(units(def.cost)) {
        return Err(Refusal::NotEnoughStamina);
    }
    Ok(Activation {
        ability: def.id.clone(),
        timing: AttackTiming::new(def.windup_ticks, ABILITY_ACTIVE_TICKS, def.recovery_ticks),
        cooldown_ticks: def.cooldown_ticks,
    })
}

/// Something an ability could be aimed at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    pub entity: Entity,
    pub position: Vec3,
    /// Whether this is an enemy of the actor. Targeting is expressed in terms
    /// of hostility rather than of entity kind so the same rules serve a player
    /// ability and an enemy one.
    pub hostile: bool,
}

/// Where an ability is aimed from, and at what.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aim<'a> {
    pub actor: Entity,
    pub origin: Vec3,
    pub facing: Vec3,
    /// Single-target range, or area radius.
    pub reach: f32,
    /// Centre for [`Targeting::AreaAtPoint`]; ignored by the other variants.
    pub point: Option<Vec3>,
    /// Everything that might be hit. Order is the caller's and is preserved in
    /// ties, so a deterministic caller gets a deterministic target set.
    pub candidates: &'a [Candidate],
}

/// Resolves a [`Targeting`] variant to the entities an ability affects.
///
/// Capped at [`MAX_TARGETS`]. Ordering is by planar distance, with the caller's
/// candidate order breaking ties: distance alone is not a total order once two
/// things stand equally far away, and an unstable target set would make replay
/// diverge.
pub fn resolve_targets(targeting: Targeting, aim: Aim<'_>) -> Vec<Entity> {
    match targeting {
        Targeting::SelfOnly => vec![aim.actor],
        Targeting::SingleEnemy => nearest(aim, aim.origin, true).into_iter().collect(),
        Targeting::SingleAlly => nearest(aim, aim.origin, false).into_iter().collect(),
        Targeting::AreaAroundSelf => within(aim, aim.origin, true),
        // An aimed area with no point falls back to the actor, rather than
        // silently affecting nothing or defaulting to the world origin.
        Targeting::AreaAtPoint => within(aim, aim.point.unwrap_or(aim.origin), true),
    }
}

/// Candidates of the requested hostility within reach of `centre`, nearest
/// first, capped at [`MAX_TARGETS`].
fn within(aim: Aim<'_>, centre: Vec3, hostile: bool) -> Vec<Entity> {
    let mut found: Vec<(usize, f32, Entity)> = aim
        .candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| candidate.hostile == hostile && candidate.entity != aim.actor)
        .filter_map(|(index, candidate)| {
            let distance = (ground(candidate.position) - ground(centre)).length();
            (distance <= aim.reach).then_some((index, distance, candidate.entity))
        })
        .collect();
    // `total_cmp` rather than `partial_cmp`: a NaN distance from degenerate
    // geometry would otherwise make the sort order undefined.
    found.sort_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)));
    found.truncate(MAX_TARGETS);
    found.into_iter().map(|(_, _, entity)| entity).collect()
}

/// The single nearest candidate of the requested hostility, if any is in reach.
fn nearest(aim: Aim<'_>, centre: Vec3, hostile: bool) -> Option<Entity> {
    within(aim, centre, hostile).into_iter().next()
}

// `ActiveStatus` and `Statuses` moved to `crate::status` in wave 2 task E,
// where the plan assigns them, alongside the modifier allowlist that decides
// what a status id is permitted to mean.

/// What executing one effect asks the caller to do.
///
/// Deliberately descriptive rather than imperative: an outcome names a bounded
/// change, and the caller — which owns the ECS queries — performs it. Nothing
/// here can express a change the executor has not already bounded.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Damage {
        target: Entity,
        amount: f32,
        damage_type: DamageType,
    },
    Status {
        target: Entity,
        status: StatusId,
        /// Absolute tick at which it ends, already clamped to
        /// [`MAX_STATUS_TICKS`].
        expires_at: u64,
    },
    Spawn {
        prefab: PrefabId,
        /// Already reduced to what the location's remaining budget allows.
        count: u8,
        at: Vec3,
    },
    Resource {
        target: Entity,
        resource: ResourceType,
        /// Signed: `warden-guard` spends stamina through this.
        amount: f32,
    },
    Teleport {
        actor: Entity,
        /// Already clamped and collision-resolved: the caller moves the actor
        /// here without further checks.
        to: Vec3,
    },
    Event {
        event: GameEventId,
    },
    /// A projectile to put in flight, already clamped in every dimension.
    Launch(Launch),
}

/// Everything one cast needs to resolve its effects.
#[derive(Debug, Clone, Copy)]
pub struct Cast<'a> {
    pub actor: Entity,
    pub origin: Vec3,
    pub facing: Vec3,
    /// The actor's body, for sweeping a teleport.
    pub capsule: Capsule,
    /// Already-resolved targets, from [`resolve_targets`].
    pub targets: &'a [Entity],
    /// Where an aimed effect points; teleport moves toward it when present.
    pub point: Option<Vec3>,
    pub collision: &'a CollisionWorld,
    /// Entities this location already holds from earlier spawn effects.
    pub live_spawns: usize,
    /// Projectiles already in flight in this location.
    pub live_projectiles: usize,
    /// The current simulation tick, for absolute status expiry.
    pub tick: u64,
}

/// Executes one ability's effects in authored order.
///
/// Spawn budget accumulates across the effect list, so an ability with two
/// spawn effects cannot exceed the cap by splitting the request in two.
pub fn execute(def: &AbilityDef, cast: Cast<'_>) -> Vec<Outcome> {
    let mut outcomes = Vec::new();
    let mut cast = cast;
    for effect in &def.effects {
        let produced = execute_effect(effect, cast, cast.live_spawns);
        for outcome in &produced {
            match outcome {
                Outcome::Spawn { count, .. } => cast.live_spawns += usize::from(*count),
                Outcome::Launch(_) => cast.live_projectiles += 1,
                _ => {}
            }
        }
        outcomes.extend(produced);
    }
    outcomes
}

/// Executes a single effect. One bounded function per variant.
pub fn execute_effect(effect: &Effect, cast: Cast<'_>, live_spawns: usize) -> Vec<Outcome> {
    match effect {
        Effect::DealDamage { amount, damage_type } => deal_damage(*amount, *damage_type, cast),
        Effect::ApplyStatus {
            status,
            duration_ticks,
        } => apply_status(status, *duration_ticks, cast),
        Effect::SpawnEntity { prefab, count } => spawn_entity(prefab, *count, cast, live_spawns)
            .into_iter()
            .collect(),
        Effect::ModifyResource { resource, amount } => modify_resource(*resource, *amount, cast),
        Effect::Teleport { max_distance } => teleport(*max_distance, cast).into_iter().collect(),
        Effect::EmitEvent { event } => vec![Outcome::Event { event: event.clone() }],
        Effect::Projectile {
            speed,
            range,
            radius,
            pierce,
            on_hit,
        } => launch(*speed, *range, *radius, *pierce, on_hit, cast)
            .into_iter()
            .collect(),
    }
}

/// Put a projectile in flight along the actor's aim, bounded in every
/// dimension whatever the content says.
///
/// Clamped here as well as validated, because the executor does not assume
/// the validator ran. `on_hit` is filtered to what may run on a hit, so a
/// nested projectile or summon is dropped rather than obeyed.
fn launch(
    speed: Fixed,
    range: Fixed,
    radius: Fixed,
    pierce: u8,
    on_hit: &[Effect],
    cast: Cast<'_>,
) -> Option<Outcome> {
    if cast.live_projectiles >= MAX_PROJECTILES_PER_LOCATION {
        return None;
    }
    let aim = match cast.point {
        Some(point) => ground(point) - ground(cast.origin),
        None => ground(cast.facing),
    };
    let direction = aim.normalize_or_zero();
    if direction == Vec2::ZERO {
        return None;
    }
    let on_hit: Vec<Effect> = on_hit
        .iter()
        .filter(|effect| effect.allowed_on_hit())
        .take(game_core::content::MAX_ON_HIT_EFFECTS)
        .cloned()
        .collect();
    if on_hit.is_empty() {
        return None;
    }
    let clamp_metres = |value: Fixed, low: f32, high: f32| units(value).clamp(low, high);
    Some(Outcome::Launch(Launch {
        owner: cast.actor,
        from: cast.origin,
        direction: Vec3::new(direction.x, 0.0, direction.y),
        speed: clamp_metres(
            speed,
            game_core::content::MIN_PROJECTILE_SPEED_METRES as f32,
            game_core::content::MAX_PROJECTILE_SPEED_METRES as f32,
        ),
        range: clamp_metres(range, 0.0, game_core::content::MAX_PROJECTILE_RANGE_METRES as f32),
        radius: clamp_metres(
            radius,
            game_core::content::MIN_PROJECTILE_RADIUS_MILLI as f32 / 1000.0,
            game_core::content::MAX_PROJECTILE_RADIUS_MILLI as f32 / 1000.0,
        ),
        pierce: pierce.min(game_core::content::MAX_PIERCE),
        on_hit,
    }))
}

/// Damage every resolved target.
///
/// Negative damage is impossible by validation (`AbilityDef::validate` checks
/// non-negativity), and clamped here anyway: the executor does not assume the
/// validator ran, because a generated candidate reaches it by a different path.
fn deal_damage(amount: Fixed, damage_type: DamageType, cast: Cast<'_>) -> Vec<Outcome> {
    let amount = units(amount).max(0.0);
    cast.targets
        .iter()
        .take(MAX_TARGETS)
        .map(|target| Outcome::Damage {
            target: *target,
            amount,
            damage_type,
        })
        .collect()
}

/// Apply a status to every resolved target, bounded in duration.
///
/// A zero duration produces nothing rather than a status that expires on the
/// tick it began: "applied for no time" is not a state worth representing.
fn apply_status(status: &StatusId, duration_ticks: u32, cast: Cast<'_>) -> Vec<Outcome> {
    let duration = duration_ticks.min(MAX_STATUS_TICKS);
    if duration == 0 {
        return Vec::new();
    }
    let expires_at = cast.tick.saturating_add(u64::from(duration));
    cast.targets
        .iter()
        .take(MAX_TARGETS)
        .map(|target| Outcome::Status {
            target: *target,
            status: status.clone(),
            expires_at,
        })
        .collect()
}

/// Request entities, reduced to what the location can still hold.
///
/// Returns `None` at the cap rather than a zero count: "spawn nothing" is not an
/// outcome the caller should have to handle.
fn spawn_entity(prefab: &PrefabId, count: u8, cast: Cast<'_>, live_spawns: usize) -> Option<Outcome> {
    let budget = MAX_SPAWNS_PER_LOCATION.saturating_sub(live_spawns);
    let allowed = usize::from(count).min(budget);
    if allowed == 0 {
        return None;
    }
    Some(Outcome::Spawn {
        prefab: prefab.clone(),
        // `allowed <= count`, which is a `u8`, so this cannot truncate.
        count: allowed as u8,
        at: cast.point.unwrap_or(cast.origin),
    })
}

/// Change a resource on every resolved target. Signed, unlike damage.
///
/// Lowering health is the exception: it lands on the caster only. On anyone
/// else it would be damage through an untyped door that skips resistances and
/// statuses, so `DealDamage` stays the one way to hurt someone. Validation
/// rejects such an ability outright, and this holds the line anyway, because
/// the executor does not assume the validator ran.
fn modify_resource(resource: ResourceType, amount: Fixed, cast: Cast<'_>) -> Vec<Outcome> {
    let amount = units(amount);
    let self_only = resource == ResourceType::Health && amount < 0.0;
    cast.targets
        .iter()
        .filter(|target| !self_only || **target == cast.actor)
        .take(MAX_TARGETS)
        .map(|target| Outcome::Resource {
            target: *target,
            resource,
            amount,
        })
        .collect()
}

/// Move the actor, clamped to `max_distance` and swept through static geometry.
///
/// Swept rather than checked at the destination: an overlap test at the end
/// point would happily place the actor on the far side of a wall, which is the
/// wall-clip this bound exists to prevent. Reusing `move_capsule` also means a
/// teleport obeys exactly the same collision rules as walking, so there is no
/// second geometry implementation to drift.
fn teleport(max_distance: Fixed, cast: Cast<'_>) -> Option<Outcome> {
    let limit = units(max_distance).max(0.0);
    if limit <= 0.0 {
        return None;
    }
    let desired = match cast.point {
        Some(point) => ground(point) - ground(cast.origin),
        None => ground(cast.facing),
    };
    let direction = desired.normalize_or_zero();
    if direction == Vec2::ZERO {
        return None;
    }
    let distance = if cast.point.is_some() {
        desired.length().min(limit)
    } else {
        limit
    };
    let motion = Vec3::new(direction.x * distance, 0.0, direction.y * distance);
    let result = cast.collision.move_capsule(cast.capsule, cast.origin, motion);
    Some(Outcome::Teleport {
        actor: cast.actor,
        to: result.position,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movement::{Stamina, StaticBox};
    use crate::status::Statuses;
    use engine::ecs::World;

    fn ability_id(name: &str) -> AbilityId {
        AbilityId::new(name).expect("valid id")
    }

    fn prefab(name: &str) -> PrefabId {
        PrefabId::new(name).expect("valid id")
    }

    fn status_id(name: &str) -> StatusId {
        StatusId(ability_id(name))
    }

    fn entities(count: usize) -> Vec<Entity> {
        let mut world = World::new();
        (0..count).map(|_| world.spawn()).collect()
    }

    /// `call-husks` as authored in `assets/content/abilities.ron`.
    fn call_husks() -> AbilityDef {
        AbilityDef {
            schema_version: 1,
            id: ability_id("call-husks"),
            name: "Call Husks".to_owned(),
            targeting: Targeting::AreaAroundSelf,
            cost: Fixed::from_milli(30_000),
            cooldown_ticks: 600,
            windup_ticks: 45,
            recovery_ticks: 40,
            effects: vec![
                Effect::SpawnEntity {
                    prefab: prefab("corrupted-husk"),
                    count: 2,
                },
                Effect::EmitEvent {
                    event: GameEventId(ability_id("event-husks-called")),
                },
            ],
            presentation: Default::default(),
        }
    }

    /// `ember-cleave` as authored.
    fn ember_cleave() -> AbilityDef {
        AbilityDef {
            schema_version: 1,
            id: ability_id("ember-cleave"),
            name: "Ember Cleave".to_owned(),
            targeting: Targeting::SingleEnemy,
            cost: Fixed::from_milli(20_000),
            cooldown_ticks: 90,
            windup_ticks: 14,
            recovery_ticks: 26,
            effects: vec![Effect::DealDamage {
                amount: Fixed::from_milli(18_000),
                damage_type: DamageType::Fire,
            }],
            presentation: Default::default(),
        }
    }

    fn empty_world() -> CollisionWorld {
        CollisionWorld::new(0.0)
    }

    fn cast<'a>(actor: Entity, targets: &'a [Entity], collision: &'a CollisionWorld) -> Cast<'a> {
        Cast {
            actor,
            origin: Vec3::new(0.0, 1.0, 0.0),
            facing: Vec3::new(0.0, 0.0, -1.0),
            capsule: Capsule::new(0.5, 1.0),
            targets,
            point: None,
            collision,
            live_spawns: 0,
            live_projectiles: 0,
            tick: 0,
        }
    }

    // ---------------------------------------------------------------------
    // The three bound tests. These are what make ADR-0010's safety claim real:
    // without them the allowlist is a list of names, not a boundary.
    // ---------------------------------------------------------------------

    /// A summon that ignores its cap is an entity leak, and `call-husks` is on a
    /// 600-tick cooldown precisely because nothing stopped it being otherwise.
    #[test]
    fn spawn_entity_cannot_exceed_the_location_cap() {
        let actor = entities(1)[0];
        let collision = empty_world();
        let husks = call_husks();

        // Room for one: the request for two is reduced, not refused.
        let mut nearly_full = cast(actor, &[], &collision);
        nearly_full.live_spawns = MAX_SPAWNS_PER_LOCATION - 1;
        let outcomes = execute(&husks, nearly_full);
        let spawns: Vec<&Outcome> = outcomes
            .iter()
            .filter(|outcome| matches!(outcome, Outcome::Spawn { .. }))
            .collect();
        assert_eq!(spawns.len(), 1, "one spawn outcome, with a reduced count");
        assert!(
            matches!(spawns[0], Outcome::Spawn { count: 1, .. }),
            "the count must be clamped to the remaining budget, got {:?}",
            spawns[0]
        );

        // At the cap: nothing at all.
        let mut full = cast(actor, &[], &collision);
        full.live_spawns = MAX_SPAWNS_PER_LOCATION;
        let outcomes = execute(&husks, full);
        assert!(
            !outcomes
                .iter()
                .any(|outcome| matches!(outcome, Outcome::Spawn { .. })),
            "a location at its cap spawns nothing, got {outcomes:?}"
        );
        assert!(
            outcomes
                .iter()
                .any(|outcome| matches!(outcome, Outcome::Event { .. })),
            "the ability's other effects still run"
        );
    }

    /// Repetition is the case a cap exists for: casting until the budget is gone
    /// must converge, not accumulate.
    #[test]
    fn repeated_summoning_converges_on_the_cap() {
        let actor = entities(1)[0];
        let collision = empty_world();
        let husks = call_husks();
        let mut live = 0usize;

        for _ in 0..100 {
            let mut this_cast = cast(actor, &[], &collision);
            this_cast.live_spawns = live;
            for outcome in execute(&husks, this_cast) {
                if let Outcome::Spawn { count, .. } = outcome {
                    live += usize::from(count);
                }
            }
        }

        assert_eq!(
            live, MAX_SPAWNS_PER_LOCATION,
            "a hundred casts must stop at the cap, not run away"
        );
    }

    /// One ability splitting its request across two effects must not beat the
    /// cap that a single effect obeys.
    #[test]
    fn spawn_budget_accumulates_within_one_ability() {
        let actor = entities(1)[0];
        let collision = empty_world();
        let greedy = AbilityDef {
            effects: vec![
                Effect::SpawnEntity {
                    prefab: prefab("corrupted-husk"),
                    count: 200,
                },
                Effect::SpawnEntity {
                    prefab: prefab("corrupted-husk"),
                    count: 200,
                },
            ],
            ..call_husks()
        };

        let total: usize = execute(&greedy, cast(actor, &[], &collision))
            .into_iter()
            .filter_map(|outcome| match outcome {
                Outcome::Spawn { count, .. } => Some(usize::from(count)),
                _ => None,
            })
            .sum();

        assert_eq!(
            total, MAX_SPAWNS_PER_LOCATION,
            "two effects share one budget rather than each getting their own"
        );
    }

    /// The wall-clip. A teleport that resolves at its destination rather than
    /// along the way is a way through any wall in the game.
    #[test]
    fn teleport_cannot_pass_through_a_wall() {
        let actor = entities(1)[0];
        let mut collision = CollisionWorld::new(0.0);
        // A wall spanning x, sitting at z = -5, between the actor and where a
        // 10 m blink would otherwise land.
        collision.push_box(StaticBox::new(
            Vec3::new(-10.0, 0.0, -6.0),
            Vec3::new(10.0, 3.0, -5.0),
        ));

        let blink = AbilityDef {
            effects: vec![Effect::Teleport {
                max_distance: Fixed::from_whole(10),
            }],
            ..ember_cleave()
        };

        let outcomes = execute(&blink, cast(actor, &[], &collision));
        let [Outcome::Teleport { to, .. }] = outcomes[..] else {
            panic!("expected one teleport outcome, got {outcomes:?}");
        };
        assert!(
            to.z > -5.0,
            "the actor must stop short of the wall, not appear beyond it: got z={}",
            to.z
        );
        assert!(to.is_finite(), "and must not produce NaN");
    }

    /// The other half of the same bound: distance is clamped even in open
    /// ground, so `max_distance` means what it says.
    #[test]
    fn teleport_is_clamped_to_its_authored_distance() {
        let actor = entities(1)[0];
        let collision = empty_world();
        let blink = AbilityDef {
            effects: vec![Effect::Teleport {
                max_distance: Fixed::from_whole(3),
            }],
            ..ember_cleave()
        };

        // Aimed far beyond the limit.
        let mut aimed = cast(actor, &[], &collision);
        aimed.point = Some(Vec3::new(0.0, 1.0, -50.0));
        let outcomes = execute(&blink, aimed);
        let [Outcome::Teleport { to, .. }] = outcomes[..] else {
            panic!("expected one teleport outcome, got {outcomes:?}");
        };
        let travelled = (ground(to) - ground(Vec3::new(0.0, 1.0, 0.0))).length();
        assert!(
            travelled <= 3.0 + 1e-3,
            "a 3 m teleport aimed 50 m away must travel 3 m, got {travelled}"
        );
    }

    /// A status whose duration nothing enforces is not a bounded effect, it is a
    /// permanent one with a number attached.
    #[test]
    fn a_status_does_not_outlive_its_duration() {
        let actor = entities(1)[0];
        let collision = empty_world();
        let guard = AbilityDef {
            targeting: Targeting::SelfOnly,
            effects: vec![Effect::ApplyStatus {
                status: status_id("status-guarded"),
                duration_ticks: 180,
            }],
            ..ember_cleave()
        };

        let targets = [actor];
        let outcomes = execute(&guard, cast(actor, &targets, &collision));
        let [
            Outcome::Status {
                ref status,
                expires_at,
                ..
            },
        ] = outcomes[..]
        else {
            panic!("expected one status outcome, got {outcomes:?}");
        };
        assert_eq!(expires_at, 180, "applied at tick 0 for 180 ticks");

        let mut statuses = Statuses::new();
        statuses.apply(status.clone(), expires_at, &[]);

        // Run past the duration, expiring each tick as the simulation would.
        for tick in 0..179 {
            statuses.expire(tick);
            assert!(
                statuses.is_active(status, tick),
                "the status must hold for its full duration, lapsed at tick {tick}"
            );
        }
        statuses.expire(180);
        assert!(
            !statuses.is_active(status, 180),
            "the status must be gone the tick it expires"
        );
        assert!(statuses.is_empty(), "and must not be left holding a dead entry");
    }

    /// An authored duration larger than the cap is clamped rather than honoured.
    #[test]
    fn status_duration_is_capped() {
        let actor = entities(1)[0];
        let collision = empty_world();
        let forever = AbilityDef {
            targeting: Targeting::SelfOnly,
            effects: vec![Effect::ApplyStatus {
                status: status_id("status-guarded"),
                duration_ticks: u32::MAX,
            }],
            ..ember_cleave()
        };

        let targets = [actor];
        let outcomes = execute(&forever, cast(actor, &targets, &collision));
        let [Outcome::Status { expires_at, .. }] = outcomes[..] else {
            panic!("expected one status outcome, got {outcomes:?}");
        };
        assert_eq!(
            expires_at,
            u64::from(MAX_STATUS_TICKS),
            "a status may not be authored into permanence"
        );
    }

    // ---------------------------------------------------------------------
    // Cooldowns, cost, and commitment.
    // ---------------------------------------------------------------------

    #[test]
    fn cooldowns_count_down_in_ticks_and_expire() {
        let mut cooldowns = Cooldowns::new();
        let cleave = ability_id("ember-cleave");
        assert!(cooldowns.is_ready(&cleave), "an untouched ability is ready");

        cooldowns.start(cleave.clone(), 3);
        for expected in [3, 2, 1] {
            assert_eq!(cooldowns.remaining_for(&cleave), expected);
            assert!(!cooldowns.is_ready(&cleave));
            cooldowns.tick();
        }
        assert!(cooldowns.is_ready(&cleave), "ready on the tick it reaches zero");
        assert!(cooldowns.is_empty(), "and the entry is dropped, not kept at zero");
    }

    /// Two husks must cool down independently, which is why cooldowns are a
    /// component rather than a global table.
    #[test]
    fn cooldowns_are_per_ability() {
        let mut cooldowns = Cooldowns::new();
        let cleave = ability_id("ember-cleave");
        let guard = ability_id("warden-guard");

        cooldowns.start(cleave.clone(), 2);
        cooldowns.start(guard.clone(), 10);
        cooldowns.tick();
        cooldowns.tick();

        assert!(cooldowns.is_ready(&cleave));
        assert!(
            !cooldowns.is_ready(&guard),
            "one ability's cooldown is not another's"
        );
        assert_eq!(cooldowns.remaining_for(&guard), 8);
    }

    #[test]
    fn activation_refuses_a_cooling_ability_without_spending() {
        let def = ember_cleave();
        let mut cooldowns = Cooldowns::new();
        cooldowns.start(def.id.clone(), 12);
        let mut stamina = Stamina::new(100.0, 1.0, 0);

        assert_eq!(
            activate(&def, &cooldowns, &mut stamina, false),
            Err(Refusal::OnCooldown { remaining: 12 })
        );
        assert!(
            (stamina.current - 100.0).abs() < 1e-6,
            "a refused activation must not cost anything"
        );
    }

    #[test]
    fn activation_refuses_when_stamina_is_short_and_spends_when_it_is_not() {
        let def = ember_cleave();
        let cooldowns = Cooldowns::new();

        let mut poor = Stamina::new(10.0, 1.0, 0);
        assert_eq!(
            activate(&def, &cooldowns, &mut poor, false),
            Err(Refusal::NotEnoughStamina)
        );
        assert!(
            (poor.current - 10.0).abs() < 1e-6,
            "an unaffordable ability must not half-happen"
        );

        let mut rich = Stamina::new(100.0, 1.0, 0);
        let activation = activate(&def, &cooldowns, &mut rich, false).expect("affordable and ready");
        assert!(
            (rich.current - 80.0).abs() < 1e-6,
            "cost is the authored 20.0, converted once"
        );
        assert_eq!(activation.cooldown_ticks, 90);
        assert_eq!(
            activation.timing,
            AttackTiming::new(14, ABILITY_ACTIVE_TICKS, 26),
            "timing comes from content, in ticks, unconverted"
        );
    }

    /// ADR-0007: an ability is a committed action exactly like a swing.
    #[test]
    fn a_committed_actor_cannot_start_an_ability() {
        let def = ember_cleave();
        let cooldowns = Cooldowns::new();
        let mut stamina = Stamina::new(100.0, 1.0, 0);

        assert_eq!(
            activate(&def, &cooldowns, &mut stamina, true),
            Err(Refusal::Committed)
        );
        assert!((stamina.current - 100.0).abs() < 1e-6);
    }

    // ---------------------------------------------------------------------
    // Targeting.
    // ---------------------------------------------------------------------

    fn aim_scene() -> (Vec<Entity>, Vec<Candidate>) {
        let ids = entities(4);
        let candidates = vec![
            Candidate {
                entity: ids[1],
                position: Vec3::new(0.0, 1.0, -2.0),
                hostile: true,
            },
            Candidate {
                entity: ids[2],
                position: Vec3::new(0.0, 1.0, -1.0),
                hostile: true,
            },
            Candidate {
                entity: ids[3],
                position: Vec3::new(0.0, 1.0, -1.5),
                hostile: false,
            },
        ];
        (ids, candidates)
    }

    fn aim(actor: Entity, candidates: &[Candidate], reach: f32) -> Aim<'_> {
        Aim {
            actor,
            origin: Vec3::new(0.0, 1.0, 0.0),
            facing: Vec3::new(0.0, 0.0, -1.0),
            reach,
            point: None,
            candidates,
        }
    }

    #[test]
    fn every_targeting_variant_resolves() {
        let (ids, candidates) = aim_scene();
        let actor = ids[0];

        assert_eq!(
            resolve_targets(Targeting::SelfOnly, aim(actor, &candidates, 5.0)),
            vec![actor]
        );
        assert_eq!(
            resolve_targets(Targeting::SingleEnemy, aim(actor, &candidates, 5.0)),
            vec![ids[2]],
            "the nearest hostile, not the first listed"
        );
        assert_eq!(
            resolve_targets(Targeting::SingleAlly, aim(actor, &candidates, 5.0)),
            vec![ids[3]]
        );
        assert_eq!(
            resolve_targets(Targeting::AreaAroundSelf, aim(actor, &candidates, 5.0)),
            vec![ids[2], ids[1]],
            "every hostile in range, nearest first"
        );

        let mut at_point = aim(actor, &candidates, 0.75);
        at_point.point = Some(Vec3::new(0.0, 1.0, -2.0));
        assert_eq!(
            resolve_targets(Targeting::AreaAtPoint, at_point),
            vec![ids[1]],
            "an aimed area is centred on the point, not the actor"
        );
    }

    #[test]
    fn targeting_respects_reach_and_finds_nothing_when_out_of_range() {
        let (ids, candidates) = aim_scene();
        let actor = ids[0];
        assert!(
            resolve_targets(Targeting::SingleEnemy, aim(actor, &candidates, 0.5)).is_empty(),
            "nothing is in reach of a half-metre ability"
        );
        assert!(resolve_targets(Targeting::AreaAroundSelf, aim(actor, &candidates, 0.5)).is_empty());
    }

    /// An area ability must not scale with how crowded the room is.
    #[test]
    fn area_targeting_is_capped() {
        let mut world = World::new();
        let actor = world.spawn();
        let candidates: Vec<Candidate> = (0..MAX_TARGETS * 3)
            .map(|index| Candidate {
                entity: world.spawn(),
                position: Vec3::new(index as f32 * 0.01, 1.0, -1.0),
                hostile: true,
            })
            .collect();

        let targets = resolve_targets(Targeting::AreaAroundSelf, aim(actor, &candidates, 50.0));
        assert_eq!(targets.len(), MAX_TARGETS, "a crowd cannot widen an ability");
    }

    /// Replay equality depends on this: equal distances must not reorder.
    #[test]
    fn target_order_is_deterministic_in_ties() {
        let mut world = World::new();
        let actor = world.spawn();
        let candidates: Vec<Candidate> = (0..4)
            .map(|_| Candidate {
                entity: world.spawn(),
                position: Vec3::new(0.0, 1.0, -2.0),
                hostile: true,
            })
            .collect();

        let first = resolve_targets(Targeting::AreaAroundSelf, aim(actor, &candidates, 5.0));
        let second = resolve_targets(Targeting::AreaAroundSelf, aim(actor, &candidates, 5.0));
        assert_eq!(first, second);
        let expected: Vec<Entity> = candidates.iter().map(|candidate| candidate.entity).collect();
        assert_eq!(first, expected, "ties keep the caller's order");
    }

    #[test]
    fn an_actor_is_never_caught_in_its_own_area() {
        let mut world = World::new();
        let actor = world.spawn();
        let candidates = vec![Candidate {
            entity: actor,
            position: Vec3::new(0.0, 1.0, 0.0),
            hostile: true,
        }];
        assert!(
            resolve_targets(Targeting::AreaAroundSelf, aim(actor, &candidates, 5.0)).is_empty(),
            "the caster is excluded from hostile target sets"
        );
    }

    // ---------------------------------------------------------------------
    // The authored abilities, executed.
    // ---------------------------------------------------------------------

    #[test]
    fn ember_cleave_deals_its_authored_fire_damage() {
        let ids = entities(2);
        let collision = empty_world();
        let targets = [ids[1]];
        let outcomes = execute(&ember_cleave(), cast(ids[0], &targets, &collision));

        let [
            Outcome::Damage {
                target,
                amount,
                damage_type,
            },
        ] = outcomes[..]
        else {
            panic!("expected one damage outcome, got {outcomes:?}");
        };
        assert_eq!(target, ids[1]);
        assert!((amount - 18.0).abs() < 1e-6, "18000 milli-units is 18.0");
        assert_eq!(damage_type, DamageType::Fire, "the Fire is no longer decorative");
    }

    #[test]
    fn call_husks_spawns_its_prefab_and_emits_its_event() {
        let actor = entities(1)[0];
        let collision = empty_world();
        let outcomes = execute(&call_husks(), cast(actor, &[], &collision));

        assert_eq!(outcomes.len(), 2, "both authored effects run, in order");
        let [
            Outcome::Spawn {
                prefab: ref spawned,
                count,
                at,
            },
            Outcome::Event { ref event },
        ] = outcomes[..]
        else {
            panic!("unexpected outcomes: {outcomes:?}");
        };
        assert_eq!(spawned, &prefab("corrupted-husk"));
        assert_eq!(count, 2);
        assert_eq!(at, Vec3::new(0.0, 1.0, 0.0), "husks arrive at the caster");
        assert_eq!(event, &GameEventId(ability_id("event-husks-called")));
    }

    /// The executor's half of the rule validation also enforces: a proposal
    /// that reaches it by another path still cannot hurt anyone but its caster
    /// through a resource.
    #[test]
    fn negative_health_lands_only_on_the_caster() {
        let ids = entities(2);
        let (actor, bystander) = (ids[0], ids[1]);
        let collision = empty_world();
        let bleed = |milli| AbilityDef {
            targeting: Targeting::AreaAroundSelf,
            effects: vec![Effect::ModifyResource {
                resource: ResourceType::Health,
                amount: Fixed::from_milli(milli),
            }],
            ..ember_cleave()
        };
        let targets = [bystander, actor];

        let hurt: Vec<Entity> = execute(&bleed(-10_000), cast(actor, &targets, &collision))
            .into_iter()
            .map(|outcome| match outcome {
                Outcome::Resource { target, .. } => target,
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        assert_eq!(hurt, vec![actor], "only the caster pays in health");

        let healed = execute(&bleed(10_000), cast(actor, &targets, &collision));
        assert_eq!(healed.len(), 2, "healing reaches everyone it targets");
    }

    #[test]
    fn modify_resource_carries_its_sign() {
        let actor = entities(1)[0];
        let collision = empty_world();
        // `warden-guard`'s second effect: a further stamina cost.
        let drain = AbilityDef {
            targeting: Targeting::SelfOnly,
            effects: vec![Effect::ModifyResource {
                resource: ResourceType::Stamina,
                amount: Fixed::from_milli(-5_000),
            }],
            ..ember_cleave()
        };

        let targets = [actor];
        let outcomes = execute(&drain, cast(actor, &targets, &collision));
        let [Outcome::Resource { resource, amount, .. }] = outcomes[..] else {
            panic!("expected one resource outcome, got {outcomes:?}");
        };
        assert_eq!(resource, ResourceType::Stamina);
        assert!(
            (amount + 5.0).abs() < 1e-6,
            "a drain stays negative: resources are signed, damage is not"
        );
    }

    /// Damage is clamped even though validation rejects a negative amount: the
    /// executor does not assume the validator ran, because generated content
    /// reaches it by a different path.
    #[test]
    fn negative_damage_cannot_heal() {
        let ids = entities(2);
        let collision = empty_world();
        let backwards = AbilityDef {
            effects: vec![Effect::DealDamage {
                amount: Fixed::from_milli(-50_000),
                damage_type: DamageType::Physical,
            }],
            ..ember_cleave()
        };

        let targets = [ids[1]];
        let outcomes = execute(&backwards, cast(ids[0], &targets, &collision));
        let [Outcome::Damage { amount, .. }] = outcomes[..] else {
            panic!("expected one damage outcome, got {outcomes:?}");
        };
        assert!((amount - 0.0).abs() < 1e-6, "damage floors at zero");
    }

    #[test]
    fn an_effect_with_no_targets_produces_nothing() {
        let actor = entities(1)[0];
        let collision = empty_world();
        assert!(
            execute(&ember_cleave(), cast(actor, &[], &collision)).is_empty(),
            "a swing at empty air is not an outcome"
        );
    }

    /// Two identical runs must produce identical outcomes, or replay is a lie.
    #[test]
    fn execution_is_deterministic() {
        let ids = entities(3);
        let collision = empty_world();
        let targets = [ids[1], ids[2]];
        let run = || execute(&ember_cleave(), cast(ids[0], &targets, &collision));
        assert_eq!(run(), run());
    }
}
