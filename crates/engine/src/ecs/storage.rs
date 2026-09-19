//! Sparse-set component storage.

use std::any::Any;

use super::entity::Entity;

/// Marker trait for types stored as ECS components.
///
/// Implement it explicitly (`impl Component for Position {}`) so the set of
/// component types stays intentional and reviewable.
pub trait Component: 'static {}

const EMPTY: u32 = u32::MAX;

/// Dense component values with a sparse entity-index lookup table.
///
/// Iteration follows dense order, which is fully determined by the sequence of
/// inserts and removals (removal swaps the last element into the hole).
pub(crate) struct SparseSet<T> {
    sparse: Vec<u32>,
    entities: Vec<Entity>,
    values: Vec<T>,
}

impl<T> SparseSet<T> {
    pub(crate) fn new() -> Self {
        Self {
            sparse: Vec::new(),
            entities: Vec::new(),
            values: Vec::new(),
        }
    }

    fn dense_index(&self, entity: Entity) -> Option<usize> {
        let dense = *self.sparse.get(entity.index() as usize)?;
        if dense == EMPTY {
            return None;
        }
        let dense = dense as usize;
        (self.entities.get(dense) == Some(&entity)).then_some(dense)
    }

    /// Inserts or replaces the value for `entity`, returning any previous value.
    pub(crate) fn insert(&mut self, entity: Entity, value: T) -> Option<T> {
        if let Some(dense) = self.dense_index(entity) {
            return Some(std::mem::replace(&mut self.values[dense], value));
        }
        let sparse_index = entity.index() as usize;
        if sparse_index >= self.sparse.len() {
            self.sparse.resize(sparse_index + 1, EMPTY);
        }
        // Each entity index appears at most once, so the dense length fits in u32.
        self.sparse[sparse_index] = self.entities.len() as u32;
        self.entities.push(entity);
        self.values.push(value);
        None
    }

    pub(crate) fn remove(&mut self, entity: Entity) -> Option<T> {
        let dense = self.dense_index(entity)?;
        self.sparse[entity.index() as usize] = EMPTY;
        self.entities.swap_remove(dense);
        let value = self.values.swap_remove(dense);
        if let Some(moved) = self.entities.get(dense) {
            self.sparse[moved.index() as usize] = dense as u32;
        }
        Some(value)
    }

    pub(crate) fn get(&self, entity: Entity) -> Option<&T> {
        self.dense_index(entity).map(|dense| &self.values[dense])
    }

    pub(crate) fn get_mut(&mut self, entity: Entity) -> Option<&mut T> {
        self.dense_index(entity).map(|dense| &mut self.values[dense])
    }

    pub(crate) fn contains(&self, entity: Entity) -> bool {
        self.dense_index(entity).is_some()
    }

    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (Entity, &T)> {
        self.entities.iter().copied().zip(self.values.iter())
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (Entity, &mut T)> {
        self.entities.iter().copied().zip(self.values.iter_mut())
    }
}

/// Type-erased view used by the world for despawn cleanup and metrics.
pub(crate) trait ErasedStorage: Any {
    fn remove_entity(&mut self, entity: Entity);
    fn len(&self) -> usize;
    fn component_name(&self) -> &'static str;
}

impl<T: Component> ErasedStorage for SparseSet<T> {
    fn remove_entity(&mut self, entity: Entity) {
        self.remove(entity);
    }

    fn len(&self) -> usize {
        SparseSet::len(self)
    }

    fn component_name(&self) -> &'static str {
        std::any::type_name::<T>()
    }
}

pub(crate) fn downcast_ref<T: Component>(storage: &dyn ErasedStorage) -> Option<&SparseSet<T>> {
    (storage as &dyn Any).downcast_ref()
}

pub(crate) fn downcast_mut<T: Component>(storage: &mut dyn ErasedStorage) -> Option<&mut SparseSet<T>> {
    (storage as &mut dyn Any).downcast_mut()
}

#[cfg(test)]
mod tests {
    use super::super::entity::Entities;
    use super::*;

    #[test]
    fn insert_get_replace() {
        let mut ids = Entities::default();
        let e = ids.alloc();
        let mut set = SparseSet::new();
        assert_eq!(set.insert(e, 1), None);
        assert_eq!(set.get(e), Some(&1));
        assert_eq!(set.insert(e, 2), Some(1));
        assert_eq!(set.len(), 1);
        *set.get_mut(e).unwrap() += 1;
        assert_eq!(set.get(e), Some(&3));
    }

    #[test]
    fn swap_remove_keeps_lookup_consistent() {
        let mut ids = Entities::default();
        let [a, b, c] = [ids.alloc(), ids.alloc(), ids.alloc()];
        let mut set = SparseSet::new();
        set.insert(a, 'a');
        set.insert(b, 'b');
        set.insert(c, 'c');
        assert_eq!(set.remove(a), Some('a'));
        assert_eq!(set.remove(a), None);
        assert_eq!(set.get(b), Some(&'b'));
        assert_eq!(set.get(c), Some(&'c'));
        let order: Vec<char> = set.iter().map(|(_, v)| *v).collect();
        assert_eq!(order, vec!['c', 'b']);
    }

    #[test]
    fn stale_handle_does_not_see_new_occupant_value() {
        let mut ids = Entities::default();
        let old = ids.alloc();
        ids.free(old);
        let new = ids.alloc();
        let mut set = SparseSet::new();
        set.insert(new, 5);
        assert!(!set.contains(old));
        assert_eq!(set.remove(old), None);
        assert_eq!(set.get(new), Some(&5));
    }
}
