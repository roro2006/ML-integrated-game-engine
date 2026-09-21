//! Domain-to-runtime projection and gameplay systems.
//!
//! The only layer that sees both canonical domain state and the runtime ECS.
//! It projects accepted domain facts into entities, runs gameplay systems on
//! engine primitives, and feeds player actions back as domain events.
//!
//! # Which way the arrow points
//!
//! Projection reads canonical state and writes the ECS — never the reverse.
//! [`project`] takes `&WorldState`, so the signature itself proves it cannot
//! mutate the domain: durable change happens only through domain events and
//! their reducers (`game_core::events`).
//!
//! # Runtime identity stays runtime
//!
//! [`ProjectionMap`] holds the correspondence between domain ids and ECS
//! entities. It is a runtime resource, rebuilt whenever a world is loaded or
//! re-projected, and it is deliberately not serializable: an entity handle in a
//! save would be meaningless on load and is exactly what PROJECT_CONTEXT §3.3
//! forbids.
//!
//! # Scope at M4
//!
//! Entities carry *identity* — which NPC, which player — but not yet position.
//! Spatial placement needs authored location content, which arrives with the
//! playable slice in M5. Inventing placeholder coordinates now would put
//! fictional geometry in the one place that is supposed to be authoritative.

pub mod ability;
pub mod build;
pub mod combat;
pub mod enemy;
pub mod interaction;
pub mod movement;
pub mod navigation;
pub mod projectile;
pub mod selection;
pub mod status;

use std::collections::BTreeMap;

use engine::ecs::{Component, Entity, Resource, World};
use game_core::ids::NpcId;
use game_core::world::WorldState;

/// Marks the entity projected from the player.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayerRef;

impl Component for PlayerRef {}

/// Marks an entity projected from a durable NPC, and names which one.
///
/// The id, not the entity, is the identity: the entity is rebuilt on every
/// load, while the id is what saves, quests, and dialogue refer to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpcRef(pub NpcId);

impl Component for NpcRef {}

/// Correspondence between domain ids and the entities projected from them.
///
/// Runtime-only and intentionally not `Serialize`: see the module docs.
#[derive(Debug, Default)]
pub struct ProjectionMap {
    player: Option<Entity>,
    npcs: BTreeMap<NpcId, Entity>,
}

impl Resource for ProjectionMap {}

impl ProjectionMap {
    /// The entity projected from the player, if the world has been projected.
    pub fn player(&self) -> Option<Entity> {
        self.player
    }

    /// The entity projected from `npc`, if it currently exists.
    pub fn npc(&self, npc: &NpcId) -> Option<Entity> {
        self.npcs.get(npc).copied()
    }

    /// How many NPC entities the projection is currently maintaining.
    pub fn npc_count(&self) -> usize {
        self.npcs.len()
    }
}

/// What one projection pass did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionStats {
    /// Entities created because the domain gained something.
    pub spawned: usize,
    /// Entities removed because the domain no longer warrants them.
    pub despawned: usize,
    /// Entities already present and kept.
    pub retained: usize,
}

/// Projects canonical state into the runtime world.
///
/// Incremental and idempotent: projecting the same state twice spawns nothing
/// the second time, and an NPC that has died since the last pass has its entity
/// removed. That matters because projection runs again after every load and at
/// content-activation safe points, not only once at startup.
pub fn project(state: &WorldState, world: &mut World, map: &mut ProjectionMap) -> ProjectionStats {
    let mut stats = ProjectionStats::default();

    match map.player {
        Some(entity) if world.is_alive(entity) => stats.retained += 1,
        _ => {
            let entity = world.spawn();
            // Insertion cannot fail: the entity was just spawned.
            let _ = world.insert(entity, PlayerRef);
            map.player = Some(entity);
            stats.spawned += 1;
        }
    }

    // An NPC earns an entity while it is alive. Dead NPCs stay in canonical
    // state — quests gate on having killed them — but have nothing to project.
    let living: Vec<&NpcId> = state
        .npcs
        .iter()
        .filter(|(_, npc)| npc.alive)
        .map(|(id, _)| id)
        .collect();

    let stale: Vec<NpcId> = map
        .npcs
        .keys()
        .filter(|id| !state.npcs.get(*id).is_some_and(|npc| npc.alive))
        .cloned()
        .collect();
    for id in stale {
        if let Some(entity) = map.npcs.remove(&id) {
            world.despawn(entity);
            stats.despawned += 1;
        }
    }

    for id in living {
        match map.npcs.get(id) {
            Some(entity) if world.is_alive(*entity) => stats.retained += 1,
            _ => {
                let entity = world.spawn();
                let _ = world.insert(entity, NpcRef(id.clone()));
                map.npcs.insert(id.clone(), entity);
                stats.spawned += 1;
            }
        }
    }

    stats
}

/// Removes every entity this projection created.
///
/// Used when switching worlds: the runtime is rebuilt from canonical state, so
/// nothing may survive from the previous one.
pub fn clear_projection(world: &mut World, map: &mut ProjectionMap) -> usize {
    let mut removed = 0;
    if let Some(entity) = map.player.take()
        && world.despawn(entity)
    {
        removed += 1;
    }
    for (_, entity) in std::mem::take(&mut map.npcs) {
        if world.despawn(entity) {
            removed += 1;
        }
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::events::{DomainEvent, apply};
    use game_core::ids::ContentRevisionId;
    use game_core::world::NpcState;

    fn npc(name: &str) -> NpcId {
        NpcId::new(name).expect("valid id")
    }

    fn world_state() -> WorldState {
        let mut state = WorldState::new(1, ContentRevisionId::new("rev-0001").expect("valid id"));
        state.npcs.insert(npc("warden-sela"), NpcState::default());
        state.npcs.insert(npc("arbiter-thal"), NpcState::default());
        state
    }

    #[test]
    fn projects_player_and_living_npcs() {
        let state = world_state();
        let mut ecs = World::new();
        let mut map = ProjectionMap::default();

        let stats = project(&state, &mut ecs, &mut map);
        assert_eq!(stats.spawned, 3, "one player plus two npcs");
        assert_eq!(stats.despawned, 0);
        assert_eq!(ecs.entity_count(), 3);
        assert!(map.player().is_some());
        assert_eq!(map.npc_count(), 2);
        assert_eq!(ecs.query::<NpcRef>().count(), 2);
    }

    #[test]
    fn projecting_twice_changes_nothing() {
        let state = world_state();
        let mut ecs = World::new();
        let mut map = ProjectionMap::default();

        project(&state, &mut ecs, &mut map);
        let before: Vec<Entity> = ecs.query::<NpcRef>().map(|(entity, _)| entity).collect();

        let stats = project(&state, &mut ecs, &mut map);
        assert_eq!(stats.spawned, 0, "nothing new to project");
        assert_eq!(stats.retained, 3);
        assert_eq!(ecs.entity_count(), 3, "and no duplicates");

        let after: Vec<Entity> = ecs.query::<NpcRef>().map(|(entity, _)| entity).collect();
        assert_eq!(before, after, "entities are reused, not recreated");
    }

    #[test]
    fn a_death_in_the_domain_removes_its_entity() {
        let mut state = world_state();
        let mut ecs = World::new();
        let mut map = ProjectionMap::default();
        project(&state, &mut ecs, &mut map);

        apply(
            &mut state,
            &DomainEvent::NpcDied {
                npc: npc("arbiter-thal"),
            },
        )
        .expect("applies");

        let stats = project(&state, &mut ecs, &mut map);
        assert_eq!(stats.despawned, 1);
        assert_eq!(map.npc_count(), 1);
        assert!(map.npc(&npc("arbiter-thal")).is_none());
        assert_eq!(ecs.query::<NpcRef>().count(), 1);
    }

    #[test]
    fn an_npc_added_to_the_domain_gains_an_entity() {
        let mut state = world_state();
        let mut ecs = World::new();
        let mut map = ProjectionMap::default();
        project(&state, &mut ecs, &mut map);

        state.npcs.insert(npc("smith-orin"), NpcState::default());
        let stats = project(&state, &mut ecs, &mut map);
        assert_eq!(stats.spawned, 1);
        assert_eq!(map.npc_count(), 3);
    }

    /// The M4 exit criterion: projecting must not disturb canonical state, and
    /// no runtime identity may leak into it.
    #[test]
    fn projection_leaves_canonical_state_byte_identical() {
        let state = world_state();
        let before = ron::to_string(&state).expect("serializes");

        let mut ecs = World::new();
        let mut map = ProjectionMap::default();
        project(&state, &mut ecs, &mut map);

        let after = ron::to_string(&state).expect("serializes");
        assert_eq!(before, after, "projection is read-only over the domain");
        assert!(
            !after.contains("Entity"),
            "no runtime handle may appear in canonical state"
        );
    }

    #[test]
    fn clearing_removes_everything_projected() {
        let state = world_state();
        let mut ecs = World::new();
        let mut map = ProjectionMap::default();
        project(&state, &mut ecs, &mut map);

        assert_eq!(clear_projection(&mut ecs, &mut map), 3);
        assert_eq!(ecs.entity_count(), 0);
        assert!(map.player().is_none());
        assert_eq!(map.npc_count(), 0);

        // And a fresh projection rebuilds from canonical state alone.
        let stats = project(&state, &mut ecs, &mut map);
        assert_eq!(stats.spawned, 3);
    }

    #[test]
    fn a_world_reloaded_from_a_save_projects_identically() {
        let state = world_state();
        let text = ron::to_string(&state).expect("serializes");
        let reloaded: WorldState = ron::from_str(&text).expect("parses");

        let mut first_ecs = World::new();
        let mut first_map = ProjectionMap::default();
        let first = project(&state, &mut first_ecs, &mut first_map);

        let mut second_ecs = World::new();
        let mut second_map = ProjectionMap::default();
        let second = project(&reloaded, &mut second_ecs, &mut second_map);

        assert_eq!(first, second);
        assert_eq!(first_map.npc_count(), second_map.npc_count());
    }
}
