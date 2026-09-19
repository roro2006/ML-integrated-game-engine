//! The runtime world: entities, component storages, resources, and event channels.

use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::fmt;

use super::entity::{Entities, Entity};
use super::event::{Event, EventCadence, Events};
use super::storage::{Component, ErasedStorage, SparseSet, downcast_mut, downcast_ref};

/// Marker trait for singleton values stored in a [`World`].
pub trait Resource: 'static {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EcsError {
    /// The entity handle is stale or was never spawned in this world.
    NoSuchEntity(Entity),
    /// `send_event` was called for an event type never registered with `add_event`.
    EventNotRegistered(&'static str),
}

impl fmt::Display for EcsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSuchEntity(entity) => write!(f, "no such entity: {entity:?}"),
            Self::EventNotRegistered(name) => write!(f, "event type not registered: {name}"),
        }
    }
}

impl std::error::Error for EcsError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentMetrics {
    pub name: &'static str,
    pub count: usize,
}

/// Debug-visible counts for a world.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldMetrics {
    pub entities: usize,
    /// Per component type, sorted by type name.
    pub components: Vec<ComponentMetrics>,
    pub resources: usize,
}

struct EventChannel {
    type_id: TypeId,
    cadence: EventCadence,
    update: fn(&mut World),
}

/// Single-threaded ECS world.
///
/// Queries borrow the world, so structural changes cannot happen while a query
/// is alive; record them in [`Commands`](super::Commands) instead.
///
/// Query iteration follows the dense order of the first component type, which
/// is deterministic for a given history of inserts and removals.
#[derive(Default)]
pub struct World {
    entities: Entities,
    storages: Vec<Box<dyn ErasedStorage>>,
    storage_lookup: HashMap<TypeId, usize>,
    resources: HashMap<TypeId, Box<dyn Any>>,
    event_channels: Vec<EventChannel>,
}

impl fmt::Debug for World {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("World")
            .field("entities", &self.entities.len())
            .field("component_types", &self.storages.len())
            .field("resources", &self.resources.len())
            .field("event_channels", &self.event_channels.len())
            .finish()
    }
}

impl World {
    pub fn new() -> Self {
        Self::default()
    }

    // --- Entities -------------------------------------------------------

    pub fn spawn(&mut self) -> Entity {
        self.entities.alloc()
    }

    /// Despawns an entity and drops all of its components.
    /// Returns `false` if the handle is stale.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.entities.is_alive(entity) {
            return false;
        }
        for storage in &mut self.storages {
            storage.remove_entity(entity);
        }
        self.entities.free(entity)
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity)
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    // --- Components -----------------------------------------------------

    /// Inserts or replaces a component, returning the previous value.
    pub fn insert<T: Component>(&mut self, entity: Entity, component: T) -> Result<Option<T>, EcsError> {
        if !self.entities.is_alive(entity) {
            return Err(EcsError::NoSuchEntity(entity));
        }
        Ok(self.storage_or_insert::<T>().insert(entity, component))
    }

    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        self.storage_mut::<T>()?.remove(entity)
    }

    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        self.storage::<T>()?.get(entity)
    }

    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        self.storage_mut::<T>()?.get_mut(entity)
    }

    pub fn has<T: Component>(&self, entity: Entity) -> bool {
        self.storage::<T>()
            .is_some_and(|storage| storage.contains(entity))
    }

    /// Iterates all entities with component `A`.
    pub fn query<A: Component>(&self) -> impl Iterator<Item = (Entity, &A)> {
        self.storage::<A>().into_iter().flat_map(SparseSet::iter)
    }

    /// Iterates all entities with component `A`, mutably.
    pub fn query_mut<A: Component>(&mut self) -> impl Iterator<Item = (Entity, &mut A)> {
        self.storage_mut::<A>().into_iter().flat_map(SparseSet::iter_mut)
    }

    /// Iterates entities that have both `A` and `B`, in `A`'s order.
    pub fn query2<A: Component, B: Component>(&self) -> impl Iterator<Item = (Entity, &A, &B)> {
        self.storage::<A>()
            .zip(self.storage::<B>())
            .into_iter()
            .flat_map(|(a, b)| {
                a.iter()
                    .filter_map(move |(entity, va)| b.get(entity).map(|vb| (entity, va, vb)))
            })
    }

    /// Iterates entities that have both `A` and `B`, with `A` mutable, in `A`'s order.
    ///
    /// # Panics
    /// If `A` and `B` are the same type.
    pub fn query2_mut<A: Component, B: Component>(&mut self) -> impl Iterator<Item = (Entity, &mut A, &B)> {
        self.storage_pair_mut::<A, B>().into_iter().flat_map(|(a, b)| {
            let b: &SparseSet<B> = b;
            a.iter_mut()
                .filter_map(move |(entity, va)| b.get(entity).map(|vb| (entity, va, vb)))
        })
    }

    /// Visits entities that have both `A` and `B`, with both mutable, in `A`'s order.
    ///
    /// # Panics
    /// If `A` and `B` are the same type.
    pub fn for_each2_mut<A: Component, B: Component>(&mut self, mut f: impl FnMut(Entity, &mut A, &mut B)) {
        let Some((a, b)) = self.storage_pair_mut::<A, B>() else {
            return;
        };
        for (entity, va) in a.iter_mut() {
            if let Some(vb) = b.get_mut(entity) {
                f(entity, va, vb);
            }
        }
    }

    fn storage<T: Component>(&self) -> Option<&SparseSet<T>> {
        let index = *self.storage_lookup.get(&TypeId::of::<T>())?;
        downcast_ref(self.storages[index].as_ref())
    }

    fn storage_mut<T: Component>(&mut self) -> Option<&mut SparseSet<T>> {
        let index = *self.storage_lookup.get(&TypeId::of::<T>())?;
        downcast_mut(self.storages[index].as_mut())
    }

    fn storage_or_insert<T: Component>(&mut self) -> &mut SparseSet<T> {
        let index = match self.storage_lookup.get(&TypeId::of::<T>()) {
            Some(&index) => index,
            None => {
                let index = self.storages.len();
                self.storages.push(Box::new(SparseSet::<T>::new()));
                self.storage_lookup.insert(TypeId::of::<T>(), index);
                index
            }
        };
        match downcast_mut(self.storages[index].as_mut()) {
            Some(storage) => storage,
            None => unreachable!("component storage registered under the wrong type"),
        }
    }

    fn storage_pair_mut<A: Component, B: Component>(
        &mut self,
    ) -> Option<(&mut SparseSet<A>, &mut SparseSet<B>)> {
        assert_ne!(
            TypeId::of::<A>(),
            TypeId::of::<B>(),
            "paired query requires two distinct component types, got {} twice",
            type_name::<A>()
        );
        let a = *self.storage_lookup.get(&TypeId::of::<A>())?;
        let b = *self.storage_lookup.get(&TypeId::of::<B>())?;
        let [sa, sb] = self.storages.get_disjoint_mut([a, b]).ok()?;
        Some((downcast_mut(sa.as_mut())?, downcast_mut(sb.as_mut())?))
    }

    // --- Resources ------------------------------------------------------

    /// Inserts or replaces a resource, returning the previous value.
    pub fn insert_resource<R: Resource>(&mut self, resource: R) -> Option<R> {
        self.resources
            .insert(TypeId::of::<R>(), Box::new(resource))
            .and_then(|old| old.downcast::<R>().ok())
            .map(|old| *old)
    }

    pub fn remove_resource<R: Resource>(&mut self) -> Option<R> {
        self.resources
            .remove(&TypeId::of::<R>())
            .and_then(|value| value.downcast::<R>().ok())
            .map(|value| *value)
    }

    pub fn resource<R: Resource>(&self) -> Option<&R> {
        self.resources.get(&TypeId::of::<R>())?.downcast_ref()
    }

    pub fn resource_mut<R: Resource>(&mut self) -> Option<&mut R> {
        self.resources.get_mut(&TypeId::of::<R>())?.downcast_mut()
    }

    pub fn has_resource<R: Resource>(&self) -> bool {
        self.resources.contains_key(&TypeId::of::<R>())
    }

    /// Temporarily removes resource `R` so `f` can use it alongside mutable
    /// world access, then reinserts it. Returns `None` if `R` is absent.
    pub fn resource_scope<R: Resource, T>(&mut self, f: impl FnOnce(&mut World, &mut R) -> T) -> Option<T> {
        let mut resource = self.remove_resource::<R>()?;
        let output = f(self, &mut resource);
        self.insert_resource(resource);
        Some(output)
    }

    // --- Events ---------------------------------------------------------

    /// Registers an event channel with the given rotation cadence.
    /// Re-registering changes the cadence and keeps buffered events.
    pub fn add_event<E: Event>(&mut self, cadence: EventCadence) {
        let type_id = TypeId::of::<E>();
        match self
            .event_channels
            .iter_mut()
            .find(|channel| channel.type_id == type_id)
        {
            Some(channel) => channel.cadence = cadence,
            None => self.event_channels.push(EventChannel {
                type_id,
                cadence,
                update: update_event_channel::<E>,
            }),
        }
        if !self.has_resource::<Events<E>>() {
            self.insert_resource(Events::<E>::default());
        }
    }

    pub fn send_event<E: Event>(&mut self, event: E) -> Result<(), EcsError> {
        self.resource_mut::<Events<E>>()
            .map(|events| events.send(event))
            .ok_or(EcsError::EventNotRegistered(type_name::<E>()))
    }

    /// Rotates every event channel registered with `cadence`, in registration order.
    pub fn update_events(&mut self, cadence: EventCadence) {
        for index in 0..self.event_channels.len() {
            let channel = &self.event_channels[index];
            if channel.cadence == cadence {
                let update = channel.update;
                update(self);
            }
        }
    }

    // --- Diagnostics ----------------------------------------------------

    pub fn metrics(&self) -> WorldMetrics {
        let mut components: Vec<ComponentMetrics> = self
            .storages
            .iter()
            .map(|storage| ComponentMetrics {
                name: storage.component_name(),
                count: storage.len(),
            })
            .collect();
        components.sort_by(|a, b| a.name.cmp(b.name));
        WorldMetrics {
            entities: self.entities.len(),
            components,
            resources: self.resources.len(),
        }
    }
}

fn update_event_channel<E: Event>(world: &mut World) {
    if let Some(events) = world.resource_mut::<Events<E>>() {
        events.update();
    }
}

#[cfg(test)]
mod tests {
    use super::super::event::EventCursor;
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Position(i32);
    impl Component for Position {}

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Velocity(i32);
    impl Component for Velocity {}

    #[derive(Debug, PartialEq)]
    struct Health(u32);
    impl Component for Health {}

    #[derive(Debug, Default, PartialEq)]
    struct Score(u32);
    impl Resource for Score {}

    struct Ping;
    impl Event for Ping {}

    #[test]
    fn component_lifecycle() {
        let mut world = World::new();
        let e = world.spawn();
        assert_eq!(world.insert(e, Position(1)), Ok(None));
        assert_eq!(world.insert(e, Position(2)), Ok(Some(Position(1))));
        assert!(world.has::<Position>(e));
        world.get_mut::<Position>(e).unwrap().0 += 1;
        assert_eq!(world.get::<Position>(e), Some(&Position(3)));
        assert_eq!(world.remove::<Position>(e), Some(Position(3)));
        assert!(!world.has::<Position>(e));
        assert_eq!(world.remove::<Position>(e), None);
    }

    #[test]
    fn despawn_drops_all_components_and_rejects_stale_handle() {
        let mut world = World::new();
        let e = world.spawn();
        world.insert(e, Position(1)).unwrap();
        world.insert(e, Health(10)).unwrap();
        assert!(world.despawn(e));
        assert!(!world.despawn(e));
        assert_eq!(world.entity_count(), 0);
        assert_eq!(world.insert(e, Position(5)), Err(EcsError::NoSuchEntity(e)));

        let reused = world.spawn();
        assert_eq!(reused.index(), e.index());
        assert_eq!(world.get::<Position>(reused), None);
        assert_eq!(world.get::<Health>(reused), None);
        assert_eq!(world.get::<Position>(e), None);
        let metrics = world.metrics();
        assert!(metrics.components.iter().all(|c| c.count == 0));
    }

    #[test]
    fn single_and_paired_queries() {
        let mut world = World::new();
        let a = world.spawn();
        let b = world.spawn();
        let c = world.spawn();
        world.insert(a, Position(0)).unwrap();
        world.insert(b, Position(10)).unwrap();
        world.insert(c, Position(20)).unwrap();
        world.insert(a, Velocity(1)).unwrap();
        world.insert(c, Velocity(3)).unwrap();

        assert_eq!(world.query::<Position>().count(), 3);
        assert_eq!(world.query::<Health>().count(), 0);

        let pairs: Vec<(Entity, i32, i32)> = world
            .query2::<Position, Velocity>()
            .map(|(e, p, v)| (e, p.0, v.0))
            .collect();
        assert_eq!(pairs, vec![(a, 0, 1), (c, 20, 3)]);

        for (_, position, velocity) in world.query2_mut::<Position, Velocity>() {
            position.0 += velocity.0;
        }
        world.for_each2_mut::<Velocity, Position>(|_, velocity, position| {
            velocity.0 *= 2;
            position.0 += velocity.0;
        });
        for (_, position) in world.query_mut::<Position>() {
            position.0 += 100;
        }
        assert_eq!(world.get::<Position>(a), Some(&Position(103)));
        assert_eq!(world.get::<Position>(b), Some(&Position(110)));
        assert_eq!(world.get::<Position>(c), Some(&Position(129)));
    }

    #[test]
    fn paired_query_with_missing_storage_is_empty() {
        let mut world = World::new();
        let e = world.spawn();
        world.insert(e, Position(1)).unwrap();
        assert_eq!(world.query2::<Position, Health>().count(), 0);
        assert_eq!(world.query2_mut::<Position, Health>().count(), 0);
    }

    #[test]
    #[should_panic(expected = "two distinct component types")]
    fn paired_mut_query_rejects_same_type() {
        let mut world = World::new();
        let e = world.spawn();
        world.insert(e, Position(1)).unwrap();
        let _ = world.query2_mut::<Position, Position>().count();
    }

    #[test]
    fn resources_and_scope() {
        let mut world = World::new();
        assert_eq!(world.resource::<Score>(), None);
        assert_eq!(world.insert_resource(Score(1)), None);
        assert_eq!(world.insert_resource(Score(2)), Some(Score(1)));
        world.resource_mut::<Score>().unwrap().0 += 1;

        let e = world.spawn();
        world.insert(e, Health(5)).unwrap();
        let seen = world.resource_scope::<Score, _>(|world, score| {
            for (_, health) in world.query_mut::<Health>() {
                health.0 += score.0;
            }
            score.0 += 10;
            world.has_resource::<Score>()
        });
        assert_eq!(seen, Some(false));
        assert_eq!(world.resource::<Score>(), Some(&Score(13)));
        assert_eq!(world.get::<Health>(e), Some(&Health(8)));
        assert_eq!(world.remove_resource::<Score>(), Some(Score(13)));
        assert_eq!(world.resource_scope::<Score, _>(|_, _| ()), None);
    }

    #[test]
    fn events_rotate_by_cadence() {
        let mut world = World::new();
        assert_eq!(
            world.send_event(Ping),
            Err(EcsError::EventNotRegistered(type_name::<Ping>()))
        );
        world.add_event::<Ping>(EventCadence::FixedTick);
        world.send_event(Ping).unwrap();

        world.update_events(EventCadence::Frame);
        world.update_events(EventCadence::Frame);
        assert_eq!(world.resource::<Events<Ping>>().map(Events::len), Some(1));

        world.update_events(EventCadence::FixedTick);
        world.update_events(EventCadence::FixedTick);
        let mut cursor = EventCursor::default();
        let events = world.resource::<Events<Ping>>().unwrap();
        assert_eq!(events.read(&mut cursor).count(), 0);
        assert_eq!(cursor.missed(), 1);
    }

    #[test]
    fn metrics_report_counts() {
        let mut world = World::new();
        let e = world.spawn();
        world.spawn();
        world.insert(e, Position(1)).unwrap();
        world.insert_resource(Score(0));
        let metrics = world.metrics();
        assert_eq!(metrics.entities, 2);
        assert_eq!(metrics.resources, 1);
        assert_eq!(metrics.components.len(), 1);
        assert_eq!(metrics.components[0].count, 1);
        assert!(metrics.components[0].name.ends_with("Position"));
    }
}
