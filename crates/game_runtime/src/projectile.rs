//! Projectiles in flight: skillshots (ADR-0017 §7, M7 wave 5).
//!
//! A projectile is simulation. It is launched by the effect executor
//! ([`crate::ability::Outcome::Launch`]), moves in the fixed step, and strikes
//! what it passes by a swept test rather than by checking where it lands, so
//! no speed lets it skip over a body or a wall between two ticks. What it does
//! to what it strikes is its `on_hit` effects, run through the same executor
//! as any other ability, so armour and statuses answer it the same way.
//!
//! # Bounded in every dimension
//!
//! | Bound | Where |
//! |---|---|
//! | Speed, range, radius, pierce | clamped at launch ([`crate::ability`]) |
//! | Live projectiles | [`MAX_PROJECTILES_PER_LOCATION`], checked at launch |
//! | Lifetime | range over speed: every tick spends range |
//! | `on_hit` depth | only [`Effect::allowed_on_hit`] effects survive launch, and are checked again on strike |
//!
//! Each bound has a test below that fails loudly.
//!
//! Not saved: a projectile in flight at a save is lost, as an attack
//! mid-swing is. Like everything else here it is runtime state.

use engine::ecs::{Component, Entity};
use engine::math::{Vec2, Vec3};
use game_core::content::Effect;

use crate::ability::{self, Candidate, Cast, Outcome};
use crate::movement::{Capsule, CollisionWorld};

/// Most projectiles one location may have in flight.
///
/// Bounds the worst case of rapid casting: a projectile on a short cooldown,
/// or several enemies sharing one, cannot fill a room faster than this.
pub const MAX_PROJECTILES_PER_LOCATION: usize = 32;

/// A projectile about to fly, as the executor bounded it.
#[derive(Debug, Clone, PartialEq)]
pub struct Launch {
    pub owner: Entity,
    pub from: Vec3,
    /// Unit length, on the ground plane.
    pub direction: Vec3,
    /// Metres per second.
    pub speed: f32,
    /// Metres.
    pub range: f32,
    /// Metres, beyond a body's own radius.
    pub radius: f32,
    pub pierce: u8,
    /// Only effects allowed on a hit.
    pub on_hit: Vec<Effect>,
}

/// A projectile in flight.
///
/// The entity's `Transform` follows `position` and is set by the caller;
/// `position` is the authority, so nothing outside the fixed step moves it.
#[derive(Debug, Clone, PartialEq)]
pub struct Projectile {
    pub owner: Entity,
    pub position: Vec3,
    pub direction: Vec3,
    pub speed: f32,
    /// Range left to fly, in metres.
    pub remaining: f32,
    pub radius: f32,
    /// Targets it may still pass through before a strike stops it.
    pub pierce_left: u8,
    /// Everyone struck so far, so nobody is struck twice by one projectile.
    pub struck: Vec<Entity>,
    pub on_hit: Vec<Effect>,
}

impl Component for Projectile {}

impl Projectile {
    pub fn new(launch: Launch) -> Self {
        Self {
            owner: launch.owner,
            position: launch.from,
            direction: launch.direction,
            speed: launch.speed,
            remaining: launch.range,
            radius: launch.radius,
            pierce_left: launch.pierce,
            struck: Vec::new(),
            on_hit: launch.on_hit,
        }
    }
}

/// How a flight ended.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum End {
    /// Stopped against static geometry, here.
    Wall(Vec3),
    /// Struck a target with no pierce left, here.
    Stopped(Vec3),
    /// Flew its whole range.
    Spent(Vec3),
}

/// One tick of flight.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Flight {
    /// Who was struck this tick, and where along the flight, in order.
    pub struck: Vec<(Entity, Vec3)>,
    pub end: Option<End>,
}

fn ground(vector: Vec3) -> Vec2 {
    Vec2::new(vector.x, vector.z)
}

/// Moves `projectile` one tick of `seconds`, and reports whom it struck.
///
/// Swept: the path from where it was to where it is going is tested whole, so
/// a fast projectile cannot pass a body between ticks. A wall is found first,
/// and nothing beyond it can be struck. Only hostile candidates are
/// considered — the caller decides hostility, and leaves out anyone who cannot
/// be struck, such as a body inside a dodge's invulnerable window. `body` is a
/// body's own radius.
pub fn fly(
    projectile: &mut Projectile,
    candidates: &[Candidate],
    body: f32,
    collision: &CollisionWorld,
    seconds: f32,
) -> Flight {
    let mut flight = Flight::default();
    let step = (projectile.speed * seconds).min(projectile.remaining).max(0.0);
    let motion = projectile.direction * step;
    let wall = collision.sweep(Capsule::new(projectile.radius, 0.0), projectile.position, motion);
    let mut travel = wall.map_or(step, |time| step * time);

    let along = ground(projectile.direction);
    let reach = projectile.radius + body;
    let mut hits: Vec<(f32, usize, Entity)> = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| candidate.hostile && !projectile.struck.contains(&candidate.entity))
        .filter_map(|(index, candidate)| {
            let offset = ground(candidate.position) - ground(projectile.position);
            let distance = offset.dot(along).clamp(0.0, travel);
            let miss = (offset - along * distance).length();
            (miss <= reach).then_some((distance, index, candidate.entity))
        })
        .collect();
    // Nearest along the path first; ties in the caller's order.
    hits.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));

    let mut stopped = false;
    for (distance, _, entity) in hits {
        projectile.struck.push(entity);
        flight
            .struck
            .push((entity, projectile.position + projectile.direction * distance));
        if projectile.pierce_left == 0 {
            travel = distance;
            stopped = true;
            break;
        }
        projectile.pierce_left -= 1;
    }

    projectile.position += projectile.direction * travel;
    projectile.remaining = (projectile.remaining - travel).max(0.0);
    flight.end = if stopped {
        Some(End::Stopped(projectile.position))
    } else if wall.is_some() {
        Some(End::Wall(projectile.position))
    } else if projectile.remaining <= f32::EPSILON || step <= 0.0 {
        Some(End::Spent(projectile.position))
    } else {
        None
    };
    flight
}

/// What striking `target` at `at` does: the projectile's `on_hit`, run through
/// the effect executor with that target alone.
///
/// Filtered again here, because a `Projectile` component is runtime state that
/// could be built by a path other than launch: nothing but an allowed effect
/// runs on a hit, whatever the component holds.
pub fn strike(
    projectile: &Projectile,
    target: Entity,
    at: Vec3,
    collision: &CollisionWorld,
    tick: u64,
) -> Vec<Outcome> {
    let targets = [target];
    let cast = Cast {
        actor: projectile.owner,
        origin: at,
        facing: projectile.direction,
        capsule: Capsule::default(),
        targets: &targets,
        point: None,
        collision,
        // Saturated, so anything that slipped through could spawn or launch
        // nothing; the filter below should already have stopped it.
        live_spawns: usize::MAX,
        live_projectiles: usize::MAX,
        tick,
    };
    projectile
        .on_hit
        .iter()
        .filter(|effect| effect.allowed_on_hit())
        .flat_map(|effect| ability::execute_effect(effect, cast, usize::MAX))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ability::execute_effect;
    use crate::movement::StaticBox;
    use engine::ecs::World;
    use game_core::content::{DamageType, Fixed, GameEventId, MAX_PIERCE};
    use game_core::ids::{AbilityId, PrefabId};

    const TICK: f32 = 1.0 / 60.0;
    const BODY: f32 = 0.35;

    fn entities(count: usize) -> Vec<Entity> {
        let mut world = World::new();
        (0..count).map(|_| world.spawn()).collect()
    }

    fn damage() -> Effect {
        Effect::DealDamage {
            amount: Fixed::from_whole(10),
            damage_type: DamageType::Fire,
        }
    }

    fn projectile_effect(
        speed: i64,
        range: i64,
        radius_milli: i64,
        pierce: u8,
        on_hit: Vec<Effect>,
    ) -> Effect {
        Effect::Projectile {
            speed: Fixed::from_whole(speed),
            range: Fixed::from_whole(range),
            radius: Fixed::from_milli(radius_milli),
            pierce,
            on_hit,
        }
    }

    /// Launches `effect` from the origin, facing -Z, through the executor.
    fn launched(
        effect: &Effect,
        actor: Entity,
        collision: &CollisionWorld,
        live_projectiles: usize,
    ) -> Option<Launch> {
        let cast = Cast {
            actor,
            origin: Vec3::new(0.0, 1.0, 0.0),
            facing: Vec3::NEG_Z,
            capsule: Capsule::default(),
            targets: &[],
            point: None,
            collision,
            live_spawns: 0,
            live_projectiles,
            tick: 0,
        };
        execute_effect(effect, cast, 0)
            .into_iter()
            .find_map(|outcome| match outcome {
                Outcome::Launch(launch) => Some(launch),
                _ => None,
            })
    }

    fn standing(entity: Entity, z: f32, x: f32) -> Candidate {
        Candidate {
            entity,
            position: Vec3::new(x, 1.0, z),
            hostile: true,
        }
    }

    /// Flies until the flight ends, collecting everyone struck.
    fn fly_out(
        projectile: &mut Projectile,
        candidates: &[Candidate],
        collision: &CollisionWorld,
    ) -> (Vec<Entity>, End, usize) {
        let mut struck = Vec::new();
        for ticks in 1..10_000 {
            let flight = fly(projectile, candidates, BODY, collision, TICK);
            struck.extend(flight.struck.iter().map(|(entity, _)| *entity));
            if let Some(end) = flight.end {
                return (struck, end, ticks);
            }
        }
        panic!("a projectile never ended: the lifetime bound failed");
    }

    // ---- the bounds ----------------------------------------------------------

    #[test]
    fn range_bounds_how_far_it_flies() {
        let actor = entities(1)[0];
        let open = CollisionWorld::new(0.0);
        let launch =
            launched(&projectile_effect(12, 5, 200, 0, vec![damage()]), actor, &open, 0).expect("launches");
        let mut projectile = Projectile::new(launch);
        let (_, end, ticks) = fly_out(&mut projectile, &[], &open);
        let End::Spent(at) = end else {
            panic!("an unobstructed flight is spent, not stopped: {end:?}");
        };
        assert!((at.z + 5.0).abs() < 1e-3, "flew exactly its range: {at:?}");
        assert_eq!(ticks, 25, "5 m at 12 m/s is 25 ticks");

        // Authored past the cap, it still stops at the cap.
        let far = launched(
            &projectile_effect(30, 500, 200, 0, vec![damage()]),
            actor,
            &open,
            0,
        )
        .expect("launches");
        assert!(far.range <= game_core::content::MAX_PROJECTILE_RANGE_METRES as f32);
    }

    #[test]
    fn a_wall_stops_it_and_shields_whoever_is_behind() {
        let [actor, behind] = entities(2)[..] else {
            unreachable!()
        };
        let mut walled = CollisionWorld::new(0.0);
        walled.push_box(StaticBox::new(
            Vec3::new(-2.0, 0.0, -4.0),
            Vec3::new(2.0, 3.0, -3.5),
        ));
        let launch = launched(
            &projectile_effect(30, 20, 200, 0, vec![damage()]),
            actor,
            &walled,
            0,
        )
        .expect("launches");
        let mut projectile = Projectile::new(launch);
        let (struck, end, _) = fly_out(&mut projectile, &[standing(behind, -6.0, 0.0)], &walled);
        let End::Wall(at) = end else {
            panic!("it should end at the wall: {end:?}");
        };
        assert!(
            at.z >= -3.5 + 0.2 - 1e-3,
            "stopped on the near face, not in or past it: {at:?}"
        );
        assert!(struck.is_empty(), "nothing behind a wall is struck");
    }

    #[test]
    fn speed_is_capped_so_it_can_be_seen_coming() {
        let actor = entities(1)[0];
        let open = CollisionWorld::new(0.0);
        let launch = launched(
            &projectile_effect(10_000, 20, 200, 0, vec![damage()]),
            actor,
            &open,
            0,
        )
        .expect("launches");
        assert!(launch.speed <= game_core::content::MAX_PROJECTILE_SPEED_METRES as f32);
        let mut projectile = Projectile::new(launch);
        let before = projectile.position;
        let _ = fly(&mut projectile, &[], BODY, &open, TICK);
        let moved = (projectile.position - before).length();
        assert!(
            moved <= game_core::content::MAX_PROJECTILE_SPEED_METRES as f32 * TICK + 1e-4,
            "moved {moved} m in one tick"
        );
    }

    #[test]
    fn at_full_speed_it_cannot_pass_a_body_between_ticks() {
        let [actor, target] = entities(2)[..] else {
            unreachable!()
        };
        let open = CollisionWorld::new(0.0);
        // 0.5 m a tick, against a slender body standing midway between two
        // tick positions: 0.25 m from each, and only 0.1 m wide with the
        // projectile's radius. Checking where the projectile *is* each tick
        // would never find it; sweeping where it went does.
        let slender = 0.05;
        let launch =
            launched(&projectile_effect(30, 20, 50, 0, vec![damage()]), actor, &open, 0).expect("launches");
        let mut projectile = Projectile::new(launch);
        let candidates = [standing(target, -7.25, 0.0)];
        let mut struck = Vec::new();
        let mut end = None;
        while end.is_none() {
            let flight = fly(&mut projectile, &candidates, slender, &open, TICK);
            struck.extend(flight.struck.iter().map(|(entity, _)| *entity));
            end = flight.end;
        }
        assert_eq!(struck, vec![target]);
        let Some(End::Stopped(at)) = end else {
            panic!("a strike with no pierce stops it: {end:?}");
        };
        assert!(
            at.z > -7.25 - 1e-3 && at.z < -7.25 + 0.1 + 1e-3,
            "stopped where it met the body, not past it: {at:?}"
        );
    }

    #[test]
    fn radius_is_capped_and_decides_who_it_grazes() {
        let [actor, near, far] = entities(3)[..] else {
            unreachable!()
        };
        let open = CollisionWorld::new(0.0);
        // Authored enormous; clamped to a metre, plus a body.
        let launch = launched(
            &projectile_effect(20, 10, 50_000, 4, vec![damage()]),
            actor,
            &open,
            0,
        )
        .expect("launches");
        assert!(launch.radius <= game_core::content::MAX_PROJECTILE_RADIUS_MILLI as f32 / 1000.0);
        let mut projectile = Projectile::new(launch);
        let reach = 1.0 + BODY;
        let (struck, _, _) = fly_out(
            &mut projectile,
            &[
                standing(near, -4.0, reach - 0.05),
                standing(far, -5.0, reach + 0.05),
            ],
            &open,
        );
        assert_eq!(struck, vec![near], "grazed inside the radius, missed outside it");
    }

    #[test]
    fn pierce_bounds_how_many_it_strikes() {
        let actors = entities(10);
        let open = CollisionWorld::new(0.0);
        let line: Vec<Candidate> = actors[1..]
            .iter()
            .enumerate()
            .map(|(index, entity)| standing(*entity, -1.0 - index as f32, 0.0))
            .collect();

        let one = launched(
            &projectile_effect(20, 20, 200, 1, vec![damage()]),
            actors[0],
            &open,
            0,
        )
        .expect("launches");
        let mut projectile = Projectile::new(one);
        let (struck, end, _) = fly_out(&mut projectile, &line, &open);
        assert_eq!(
            struck,
            vec![actors[1], actors[2]],
            "pierce 1: the first, and one more"
        );
        assert!(matches!(end, End::Stopped(_)));

        // Authored far past the cap, it strikes at most the cap plus one.
        let greedy = launched(
            &projectile_effect(20, 20, 200, 200, vec![damage()]),
            actors[0],
            &open,
            0,
        )
        .expect("launches");
        let mut projectile = Projectile::new(greedy);
        let (struck, _, _) = fly_out(&mut projectile, &line, &open);
        assert_eq!(struck.len(), usize::from(MAX_PIERCE) + 1);
        assert_eq!(
            struck,
            actors[1..=usize::from(MAX_PIERCE) + 1].to_vec(),
            "nearest first"
        );
    }

    #[test]
    fn nobody_is_struck_twice_by_one_projectile() {
        let [actor, target] = entities(2)[..] else {
            unreachable!()
        };
        let open = CollisionWorld::new(0.0);
        // Slow and wide with pierce, so the target overlaps it for many ticks.
        let launch = launched(
            &projectile_effect(2, 6, 1_000, 4, vec![damage()]),
            actor,
            &open,
            0,
        )
        .expect("launches");
        let mut projectile = Projectile::new(launch);
        let (struck, _, _) = fly_out(&mut projectile, &[standing(target, -2.0, 0.0)], &open);
        assert_eq!(struck, vec![target]);
    }

    #[test]
    fn the_live_cap_refuses_a_launch() {
        let actor = entities(1)[0];
        let open = CollisionWorld::new(0.0);
        let effect = projectile_effect(20, 10, 200, 0, vec![damage()]);
        assert!(launched(&effect, actor, &open, MAX_PROJECTILES_PER_LOCATION - 1).is_some());
        assert!(
            launched(&effect, actor, &open, MAX_PROJECTILES_PER_LOCATION).is_none(),
            "at the cap, a cast launches nothing"
        );
    }

    #[test]
    fn one_ability_cannot_launch_past_the_cap_by_splitting() {
        let actor = entities(1)[0];
        let open = CollisionWorld::new(0.0);
        let volley = game_core::content::AbilityDef {
            schema_version: 1,
            id: AbilityId::new("volley").expect("valid id"),
            name: "Volley".to_owned(),
            targeting: game_core::content::Targeting::SelfOnly,
            cost: Fixed::ZERO,
            cooldown_ticks: 0,
            windup_ticks: 1,
            recovery_ticks: 1,
            effects: vec![projectile_effect(20, 10, 200, 0, vec![damage()]); 8],
            presentation: Default::default(),
        };
        let cast = Cast {
            actor,
            origin: Vec3::ZERO,
            facing: Vec3::NEG_Z,
            capsule: Capsule::default(),
            targets: &[],
            point: None,
            collision: &open,
            live_spawns: 0,
            live_projectiles: MAX_PROJECTILES_PER_LOCATION - 3,
            tick: 0,
        };
        let launches = ability::execute(&volley, cast)
            .into_iter()
            .filter(|outcome| matches!(outcome, Outcome::Launch(_)))
            .count();
        assert_eq!(launches, 3, "only up to the cap, across the whole ability");
    }

    /// Content validation rejects these; the runtime holds the line anyway,
    /// because a generated candidate reaches it by a different path.
    #[test]
    fn on_hit_is_one_level_deep_whatever_the_content_says() {
        let [actor, target] = entities(2)[..] else {
            unreachable!()
        };
        let open = CollisionWorld::new(0.0);
        let nested = vec![
            projectile_effect(20, 10, 200, 0, vec![damage()]),
            Effect::SpawnEntity {
                prefab: PrefabId::new("corrupted-husk").expect("valid id"),
                count: 5,
            },
            Effect::Teleport {
                max_distance: Fixed::from_whole(10),
            },
            damage(),
            Effect::EmitEvent {
                event: GameEventId(AbilityId::new("event-struck").expect("valid id")),
            },
        ];
        let launch = launched(&projectile_effect(20, 10, 200, 0, nested), actor, &open, 0)
            .expect("launches the allowed part");
        assert!(
            launch.on_hit.iter().all(Effect::allowed_on_hit),
            "no projectile, summon or teleport survives launch: {:?}",
            launch.on_hit
        );
        assert_eq!(launch.on_hit.len(), 2);

        // And a component built around the launch still runs only what is
        // allowed.
        let mut projectile = Projectile::new(launch);
        projectile
            .on_hit
            .push(projectile_effect(20, 10, 200, 0, vec![damage()]));
        projectile.on_hit.push(Effect::SpawnEntity {
            prefab: PrefabId::new("corrupted-husk").expect("valid id"),
            count: 5,
        });
        projectile.on_hit.push(Effect::Teleport {
            max_distance: Fixed::from_whole(10),
        });
        let outcomes = strike(&projectile, target, Vec3::ZERO, &open, 0);
        assert!(
            outcomes
                .iter()
                .all(|outcome| matches!(outcome, Outcome::Damage { .. } | Outcome::Event { .. })),
            "a strike launches, summons and moves nothing: {outcomes:?}"
        );
        assert!(outcomes.iter().any(|outcome| matches!(
            outcome,
            Outcome::Damage { target: struck, .. } if *struck == target
        )));

        // With nothing allowed left, nothing launches at all.
        let hollow = projectile_effect(
            20,
            10,
            200,
            0,
            vec![Effect::Teleport {
                max_distance: Fixed::from_whole(3),
            }],
        );
        assert!(launched(&hollow, actor, &open, 0).is_none());
    }

    #[test]
    fn only_the_hostile_are_struck() {
        let [actor, friend] = entities(2)[..] else {
            unreachable!()
        };
        let open = CollisionWorld::new(0.0);
        let launch = launched(
            &projectile_effect(20, 10, 200, 0, vec![damage()]),
            actor,
            &open,
            0,
        )
        .expect("launches");
        let mut projectile = Projectile::new(launch);
        let friendly = Candidate {
            hostile: false,
            ..standing(friend, -3.0, 0.0)
        };
        let owner = Candidate {
            hostile: false,
            ..standing(actor, 0.0, 0.0)
        };
        let (struck, end, _) = fly_out(&mut projectile, &[owner, friendly], &open);
        assert!(struck.is_empty());
        assert!(matches!(end, End::Spent(_)));
    }

    #[test]
    fn a_strike_runs_the_on_hit_effects_on_the_target_alone() {
        let [actor, target] = entities(2)[..] else {
            unreachable!()
        };
        let open = CollisionWorld::new(0.0);
        let launch = launched(
            &projectile_effect(20, 10, 200, 0, vec![damage()]),
            actor,
            &open,
            0,
        )
        .expect("launches");
        let projectile = Projectile::new(launch);
        let outcomes = strike(&projectile, target, Vec3::new(0.0, 1.0, -3.0), &open, 0);
        assert_eq!(
            outcomes,
            vec![Outcome::Damage {
                target,
                amount: 10.0,
                damage_type: DamageType::Fire,
            }]
        );
    }
}
