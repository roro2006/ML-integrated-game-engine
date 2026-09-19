//! Deferred structural mutation.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use super::entity::Entity;
use super::storage::Component;
use super::world::World;

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

/// Placeholder for an entity that a [`Commands`] buffer will spawn on apply.
///
/// Only valid for the buffer and batch that issued it; using it elsewhere, or
/// after that batch was applied or cleared, makes the command a no-op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingEntity {
    buffer: u64,
    batch: u64,
    slot: usize,
}

/// The entity a command operates on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandTarget {
    Entity(Entity),
    Pending(PendingEntity),
}

impl From<Entity> for CommandTarget {
    fn from(entity: Entity) -> Self {
        Self::Entity(entity)
    }
}

impl From<PendingEntity> for CommandTarget {
    fn from(pending: PendingEntity) -> Self {
        Self::Pending(pending)
    }
}

type EntityOp = Box<dyn FnOnce(&mut World, Entity)>;
type WorldOp = Box<dyn FnOnce(&mut World)>;

enum Command {
    Spawn,
    Despawn(CommandTarget),
    Entity(CommandTarget, EntityOp),
    World(WorldOp),
}

/// Outcome of applying a command buffer.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CommandReport {
    /// Entities spawned, in the order their spawn commands were recorded.
    pub spawned: Vec<Entity>,
    /// Commands ignored because their target was stale or unresolvable.
    pub skipped: usize,
}

/// A queue of structural world changes applied later, strictly in recorded order.
///
/// Systems record commands while iterating queries (which borrow the world)
/// and the schedule applies them once the system returns.
pub struct Commands {
    id: u64,
    batch: u64,
    spawns: usize,
    queue: Vec<Command>,
}

impl Default for Commands {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Commands {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Commands")
            .field("queued", &self.queue.len())
            .field("pending_spawns", &self.spawns)
            .finish()
    }
}

impl Commands {
    pub fn new() -> Self {
        Self {
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            batch: 0,
            spawns: 0,
            queue: Vec::new(),
        }
    }

    pub fn spawn(&mut self) -> PendingEntity {
        let pending = PendingEntity {
            buffer: self.id,
            batch: self.batch,
            slot: self.spawns,
        };
        self.spawns += 1;
        self.queue.push(Command::Spawn);
        pending
    }

    pub fn despawn(&mut self, target: impl Into<CommandTarget>) {
        self.queue.push(Command::Despawn(target.into()));
    }

    pub fn insert<T: Component>(&mut self, target: impl Into<CommandTarget>, component: T) {
        self.queue.push(Command::Entity(
            target.into(),
            Box::new(move |world, entity| {
                // Liveness is checked before the op runs, so insertion cannot fail here.
                let _ = world.insert(entity, component);
            }),
        ));
    }

    pub fn remove<T: Component>(&mut self, target: impl Into<CommandTarget>) {
        self.queue.push(Command::Entity(
            target.into(),
            Box::new(|world, entity| {
                world.remove::<T>(entity);
            }),
        ));
    }

    /// Queues an arbitrary world mutation.
    pub fn add(&mut self, op: impl FnOnce(&mut World) + 'static) {
        self.queue.push(Command::World(Box::new(op)));
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Discards queued commands and invalidates outstanding pending entities.
    pub fn clear(&mut self) {
        self.queue.clear();
        self.spawns = 0;
        self.batch += 1;
    }

    /// Applies all queued commands in order and invalidates pending entities.
    pub fn apply(&mut self, world: &mut World) -> CommandReport {
        let mut report = CommandReport {
            spawned: Vec::with_capacity(self.spawns),
            skipped: 0,
        };
        let (id, batch) = (self.id, self.batch);
        for command in self.queue.drain(..) {
            match command {
                Command::Spawn => report.spawned.push(world.spawn()),
                Command::Despawn(target) => {
                    let despawned = resolve(target, id, batch, &report.spawned)
                        .is_some_and(|entity| world.despawn(entity));
                    if !despawned {
                        report.skipped += 1;
                    }
                }
                Command::Entity(target, op) => match resolve(target, id, batch, &report.spawned) {
                    Some(entity) if world.is_alive(entity) => op(world, entity),
                    _ => report.skipped += 1,
                },
                Command::World(op) => op(world),
            }
        }
        self.spawns = 0;
        self.batch += 1;
        report
    }
}

fn resolve(target: CommandTarget, id: u64, batch: u64, spawned: &[Entity]) -> Option<Entity> {
    match target {
        CommandTarget::Entity(entity) => Some(entity),
        CommandTarget::Pending(pending) if pending.buffer == id && pending.batch == batch => {
            spawned.get(pending.slot).copied()
        }
        CommandTarget::Pending(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Tag(u8);
    impl Component for Tag {}

    #[test]
    fn commands_apply_in_recorded_order() {
        let mut world = World::default();
        let a = world.spawn();
        let b = world.spawn();
        let mut commands = Commands::new();
        commands.insert(a, Tag(1));
        commands.remove::<Tag>(a);
        commands.remove::<Tag>(b);
        commands.insert(b, Tag(2));
        let report = commands.apply(&mut world);
        assert_eq!(report.skipped, 0);
        assert_eq!(world.get::<Tag>(a), None);
        assert_eq!(world.get::<Tag>(b), Some(&Tag(2)));
        assert!(commands.is_empty());
    }

    #[test]
    fn pending_entities_resolve_within_batch() {
        let mut world = World::default();
        let mut commands = Commands::new();
        let first = commands.spawn();
        let second = commands.spawn();
        commands.insert(second, Tag(2));
        commands.insert(first, Tag(1));
        let report = commands.apply(&mut world);
        assert_eq!(report.spawned.len(), 2);
        assert_eq!(world.get::<Tag>(report.spawned[0]), Some(&Tag(1)));
        assert_eq!(world.get::<Tag>(report.spawned[1]), Some(&Tag(2)));
    }

    #[test]
    fn stale_pending_entities_are_skipped() {
        let mut world = World::default();
        let mut commands = Commands::new();
        let old = commands.spawn();
        commands.apply(&mut world);

        commands.spawn();
        commands.insert(old, Tag(9));
        let report = commands.apply(&mut world);
        assert_eq!(report.skipped, 1);
        assert_eq!(world.get::<Tag>(report.spawned[0]), None);

        let mut other = Commands::new();
        let foreign = other.spawn();
        commands.spawn();
        commands.insert(foreign, Tag(9));
        assert_eq!(commands.apply(&mut world).skipped, 1);
    }

    #[test]
    fn commands_on_dead_entities_are_skipped() {
        let mut world = World::default();
        let e = world.spawn();
        let mut commands = Commands::new();
        commands.despawn(e);
        commands.insert(e, Tag(1));
        commands.despawn(e);
        let report = commands.apply(&mut world);
        assert_eq!(report.skipped, 2);
        assert!(!world.is_alive(e));
    }

    #[test]
    fn structural_changes_can_be_recorded_during_query() {
        let mut world = World::default();
        for i in 0..4 {
            let e = world.spawn();
            world.insert(e, Tag(i)).unwrap();
        }
        let mut commands = Commands::new();
        for (entity, tag) in world.query::<Tag>() {
            if tag.0 % 2 == 0 {
                commands.despawn(entity);
            }
        }
        commands.apply(&mut world);
        let mut remaining: Vec<u8> = world.query::<Tag>().map(|(_, t)| t.0).collect();
        remaining.sort_unstable();
        assert_eq!(remaining, vec![1, 3]);
    }
}
