//! Headless M2 exit-criteria test: a world can spawn, update, query, despawn,
//! and reject stale handles deterministically.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Position {
    x: i64,
    y: i64,
}
impl Component for Position {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Velocity {
    dx: i64,
    dy: i64,
}
impl Component for Velocity {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Lifetime(u32);
impl Component for Lifetime {}

#[derive(Debug, PartialEq, Eq)]
struct Expired {
    position: Position,
}
impl Event for Expired {}

#[derive(Default)]
struct Tick(u64);
impl Resource for Tick {}

#[derive(Default)]
struct ExpiredLog(Vec<(u64, Position)>);
impl Resource for ExpiredLog {}

/// Deterministic pseudo-random sequence (SplitMix64) so the test needs no RNG crate.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn build_schedule(seed: u64) -> Schedule {
    let mut schedule = Schedule::with_stages("sim", &["spawn", "update", "cleanup", "observe"]).unwrap();

    let mut rng = seed;
    schedule
        .add_system("spawn", "spawner", move |world, commands| {
            let tick = world.resource::<Tick>().map_or(0, |t| t.0);
            let count = splitmix(&mut rng) % 3;
            for _ in 0..count {
                let e = commands.spawn();
                let r = splitmix(&mut rng);
                commands.insert(e, Position { x: tick as i64, y: 0 });
                commands.insert(
                    e,
                    Velocity {
                        dx: (r % 5) as i64 - 2,
                        dy: ((r >> 8) % 5) as i64 - 2,
                    },
                );
                commands.insert(e, Lifetime(1 + (r >> 16) as u32 % 6));
            }
        })
        .unwrap();

    schedule
        .add_system("update", "movement", |world, _| {
            for (_, position, velocity) in world.query2_mut::<Position, Velocity>() {
                position.x += velocity.dx;
                position.y += velocity.dy;
            }
        })
        .unwrap();

    schedule
        .add_system("cleanup", "expire", |world, commands| {
            let mut expired = Vec::new();
            world.for_each2_mut::<Lifetime, Position>(|entity, lifetime, position| {
                lifetime.0 -= 1;
                if lifetime.0 == 0 {
                    commands.despawn(entity);
                    expired.push(*position);
                }
            });
            for position in expired {
                world.send_event(Expired { position }).unwrap();
            }
        })
        .unwrap();

    let mut cursor = EventCursor::default();
    schedule
        .add_system("observe", "record-expired", move |world, _| {
            let tick = world.resource::<Tick>().map_or(0, |t| t.0);
            let positions: Vec<Position> = world
                .resource::<Events<Expired>>()
                .map(|events| events.read(&mut cursor).map(|e| e.position).collect())
                .unwrap_or_default();
            if let Some(log) = world.resource_mut::<ExpiredLog>() {
                log.0.extend(positions.into_iter().map(|p| (tick, p)));
            }
            if let Some(t) = world.resource_mut::<Tick>() {
                t.0 += 1;
            }
        })
        .unwrap();

    schedule
}

type Snapshot = (Vec<(Entity, Position, Lifetime)>, Vec<(u64, Position)>);

fn simulate(seed: u64, ticks: u32) -> (Snapshot, Vec<Entity>) {
    let mut world = World::new();
    world.insert_resource(Tick::default());
    world.insert_resource(ExpiredLog::default());
    world.add_event::<Expired>(EventCadence::FixedTick);
    let mut schedule = build_schedule(seed);
    let mut every_entity_seen = Vec::new();

    for _ in 0..ticks {
        schedule.run(&mut world);
        world.update_events(EventCadence::FixedTick);
        every_entity_seen.extend(world.query::<Position>().map(|(e, _)| e));
    }

    let mut state: Vec<(Entity, Position, Lifetime)> = world
        .query2::<Position, Lifetime>()
        .map(|(e, p, l)| (e, *p, *l))
        .collect();
    state.sort_by_key(|(e, _, _)| *e);
    let log = world.remove_resource::<ExpiredLog>().unwrap().0;
    every_entity_seen.sort();
    every_entity_seen.dedup();
    let stale: Vec<Entity> = every_entity_seen
        .into_iter()
        .filter(|e| !world.is_alive(*e))
        .collect();
    for entity in &stale {
        assert_eq!(
            world.get::<Position>(*entity),
            None,
            "stale handle resolved to data"
        );
        assert!(!world.despawn(*entity), "stale handle despawned something");
    }
    ((state, log), stale)
}

#[test]
fn simulation_is_deterministic_and_rejects_stale_handles() {
    let (first, first_stale) = simulate(42, 200);
    let (second, second_stale) = simulate(42, 200);
    assert_eq!(first, second);
    assert_eq!(first_stale, second_stale);
    assert!(!first.1.is_empty(), "expected some entities to expire");
    assert!(!first_stale.is_empty(), "expected despawned handles to be tested");
    assert!(
        first_stale.iter().any(|e| e.generation() > 0),
        "expected slot reuse with bumped generations"
    );

    let (different, _) = simulate(7, 200);
    assert_ne!(first, different, "different seeds should diverge");
}
