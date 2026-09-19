//! Generational runtime entity handles and slot allocation.

use std::collections::VecDeque;
use std::fmt;

/// A transient runtime handle to an entity in a [`World`](super::World).
///
/// Handles are generational: when an entity is despawned its slot may be
/// reused, but handles issued for the previous occupant are rejected.
///
/// Entity handles are runtime-only. They must never be persisted in canonical
/// state, save files, or content; use stable domain IDs there instead.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Entity {
    index: u32,
    generation: u32,
}

impl Entity {
    const fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Slot index. Slots are reused after despawn, so this is not unique on its own.
    pub const fn index(self) -> u32 {
        self.index
    }

    /// Generation of the slot at the time this handle was issued.
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

impl fmt::Debug for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Entity({}v{})", self.index, self.generation)
    }
}

#[derive(Debug, Clone, Copy)]
struct Slot {
    generation: u32,
    alive: bool,
}

/// Allocator for entity slots.
///
/// Freed slots are reused in FIFO order. Allocation is deterministic for a
/// given sequence of spawns and despawns, and recently freed slots are not
/// immediately recycled. A slot whose generation would overflow is retired
/// permanently rather than wrapping, so stale handles can never alias.
#[derive(Debug, Default)]
pub(crate) struct Entities {
    slots: Vec<Slot>,
    free: VecDeque<u32>,
    alive: usize,
}

impl Entities {
    pub(crate) fn alloc(&mut self) -> Entity {
        if let Some(index) = self.free.pop_front() {
            let slot = &mut self.slots[index as usize];
            slot.alive = true;
            self.alive += 1;
            return Entity::new(index, slot.generation);
        }
        let Ok(index) = u32::try_from(self.slots.len()) else {
            panic!("entity index space exhausted");
        };
        self.slots.push(Slot {
            generation: 0,
            alive: true,
        });
        self.alive += 1;
        Entity::new(index, 0)
    }

    /// Frees a live entity's slot. Returns `false` for stale or unknown handles.
    pub(crate) fn free(&mut self, entity: Entity) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        let slot = &mut self.slots[entity.index as usize];
        slot.alive = false;
        self.alive -= 1;
        if let Some(next) = slot.generation.checked_add(1) {
            slot.generation = next;
            self.free.push_back(entity.index);
        }
        true
    }

    pub(crate) fn is_alive(&self, entity: Entity) -> bool {
        self.slots
            .get(entity.index as usize)
            .is_some_and(|slot| slot.alive && slot.generation == entity.generation)
    }

    pub(crate) fn len(&self) -> usize {
        self.alive
    }

    #[cfg(test)]
    fn force_generation(&mut self, index: u32, generation: u32) -> Entity {
        self.slots[index as usize].generation = generation;
        Entity::new(index, generation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_sequential_indices_starting_at_generation_zero() {
        let mut entities = Entities::default();
        let a = entities.alloc();
        let b = entities.alloc();
        assert_eq!((a.index(), a.generation()), (0, 0));
        assert_eq!((b.index(), b.generation()), (1, 0));
        assert_eq!(entities.len(), 2);
    }

    #[test]
    fn reused_slot_gets_new_generation_and_rejects_stale_handle() {
        let mut entities = Entities::default();
        let old = entities.alloc();
        assert!(entities.free(old));
        let new = entities.alloc();
        assert_eq!(new.index(), old.index());
        assert_eq!(new.generation(), old.generation() + 1);
        assert!(!entities.is_alive(old));
        assert!(entities.is_alive(new));
    }

    #[test]
    fn double_free_and_stale_free_are_rejected() {
        let mut entities = Entities::default();
        let e = entities.alloc();
        assert!(entities.free(e));
        assert!(!entities.free(e));
        let reused = entities.alloc();
        assert!(!entities.free(e), "stale handle must not free the new occupant");
        assert!(entities.is_alive(reused));
        assert_eq!(entities.len(), 1);
    }

    #[test]
    fn free_slots_are_reused_in_fifo_order() {
        let mut entities = Entities::default();
        let a = entities.alloc();
        let b = entities.alloc();
        let c = entities.alloc();
        entities.free(b);
        entities.free(a);
        entities.free(c);
        let order: Vec<u32> = (0..3).map(|_| entities.alloc().index()).collect();
        assert_eq!(order, vec![1, 0, 2]);
    }

    #[test]
    fn unknown_index_is_not_alive() {
        let mut entities = Entities::default();
        let e = entities.alloc();
        let foreign = Entity::new(e.index() + 10, 0);
        assert!(!entities.is_alive(foreign));
    }

    #[test]
    fn generation_overflow_retires_slot() {
        let mut entities = Entities::default();
        entities.alloc();
        let last = entities.force_generation(0, u32::MAX);
        assert!(entities.free(last));
        assert!(!entities.is_alive(last));
        let next = entities.alloc();
        assert_eq!(next.index(), 1, "retired slot must not be reused");
    }
}
