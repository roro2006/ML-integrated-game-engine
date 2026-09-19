//! Buffered event channels with explicit two-update lifetime.

use std::fmt;

use super::world::Resource;

/// Marker trait for types sent through [`Events`] channels.
pub trait Event: 'static {}

/// When an event channel's buffers are rotated.
///
/// An event is readable during the update in which it was sent and during the
/// following update of its cadence; after that it is dropped. Frame channels
/// rotate once per rendered frame. Fixed-tick channels rotate once per fixed
/// simulation tick, so events survive frames in which no tick runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventCadence {
    Frame,
    FixedTick,
}

/// A double-buffered event queue stored as a world resource.
pub struct Events<E> {
    previous: Vec<E>,
    current: Vec<E>,
    previous_start: u64,
    current_start: u64,
}

impl<E> Default for Events<E> {
    fn default() -> Self {
        Self {
            previous: Vec::new(),
            current: Vec::new(),
            previous_start: 0,
            current_start: 0,
        }
    }
}

impl<E: Event> Resource for Events<E> {}

impl<E> fmt::Debug for Events<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Events")
            .field("retained", &self.len())
            .field("sent_total", &(self.current_start + self.current.len() as u64))
            .finish()
    }
}

/// Per-reader position in an event channel.
///
/// Readers own their cursor, so several independent readers can consume the
/// same channel without interfering with each other.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct EventCursor {
    next: u64,
    missed: u64,
}

impl EventCursor {
    /// Total events this cursor never read because they expired first.
    pub fn missed(&self) -> u64 {
        self.missed
    }
}

impl<E> Events<E> {
    pub fn send(&mut self, event: E) {
        self.current.push(event);
    }

    /// Rotates buffers: drops the previous update's events and retains the
    /// current ones for one more update.
    pub fn update(&mut self) {
        std::mem::swap(&mut self.previous, &mut self.current);
        self.current.clear();
        self.previous_start = self.current_start;
        self.current_start += self.previous.len() as u64;
    }

    /// Returns all retained events the cursor has not yet read, oldest first,
    /// and advances the cursor past them.
    pub fn read<'a>(&'a self, cursor: &mut EventCursor) -> impl Iterator<Item = &'a E> + use<'a, E> {
        let end = self.current_start + self.current.len() as u64;
        cursor.missed += self.previous_start.saturating_sub(cursor.next);
        let start = cursor.next.clamp(self.previous_start, end);
        cursor.next = end;

        let previous_from = ((start - self.previous_start) as usize).min(self.previous.len());
        let current_from = (start.saturating_sub(self.current_start) as usize).min(self.current.len());
        self.previous[previous_from..]
            .iter()
            .chain(self.current[current_from..].iter())
    }

    /// Number of retained events across both buffers.
    pub fn len(&self) -> usize {
        self.previous.len() + self.current.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Hit(u32);
    impl Event for Hit {}

    fn read_all(events: &Events<Hit>, cursor: &mut EventCursor) -> Vec<u32> {
        events.read(cursor).map(|hit| hit.0).collect()
    }

    #[test]
    fn events_are_readable_in_same_and_next_update_only() {
        let mut events = Events::default();
        events.send(Hit(1));
        let mut same_update = EventCursor::default();
        assert_eq!(read_all(&events, &mut same_update), vec![1]);

        events.update();
        let mut next_update = EventCursor::default();
        assert_eq!(read_all(&events, &mut next_update), vec![1]);

        events.update();
        let mut late = EventCursor::default();
        assert!(read_all(&events, &mut late).is_empty());
        assert_eq!(late.missed(), 1);
    }

    #[test]
    fn cursor_never_rereads() {
        let mut events = Events::default();
        let mut cursor = EventCursor::default();
        events.send(Hit(1));
        assert_eq!(read_all(&events, &mut cursor), vec![1]);
        events.send(Hit(2));
        assert_eq!(read_all(&events, &mut cursor), vec![2]);
        events.update();
        events.send(Hit(3));
        assert_eq!(read_all(&events, &mut cursor), vec![3]);
        assert!(read_all(&events, &mut cursor).is_empty());
        assert_eq!(cursor.missed(), 0);
    }

    #[test]
    fn independent_cursors_each_see_all_events() {
        let mut events = Events::default();
        let (mut a, mut b) = (EventCursor::default(), EventCursor::default());
        events.send(Hit(7));
        events.send(Hit(8));
        assert_eq!(read_all(&events, &mut a), vec![7, 8]);
        events.update();
        assert_eq!(read_all(&events, &mut b), vec![7, 8]);
    }

    #[test]
    fn missed_counts_only_expired_events() {
        let mut events = Events::default();
        let mut cursor = EventCursor::default();
        events.send(Hit(1));
        events.send(Hit(2));
        events.update();
        events.send(Hit(3));
        events.update();
        events.send(Hit(4));
        assert_eq!(read_all(&events, &mut cursor), vec![3, 4]);
        assert_eq!(cursor.missed(), 2);
    }
}
