//! Minimal single-threaded ECS.
//!
//! - Generational [`Entity`] handles with stale-handle rejection.
//! - Sparse-set component storage per component type.
//! - Single and paired queries; iteration order is deterministic for a given
//!   history of structural changes.
//! - [`Commands`] for deferred structural mutation.
//! - Typed resources and double-buffered [`Events`] with explicit cadence.
//! - [`Schedule`]s of ordered stages and systems with per-system metrics.
//!
//! The ECS holds transient runtime state only. It is not a save format, and
//! [`Entity`] handles must never appear in canonical state or content.
//! See `docs/adr/0005-ecs-storage-and-scheduling.md`.

mod command;
mod entity;
mod event;
mod schedule;
mod storage;
mod world;

#[cfg(test)]
mod tests;

pub use command::{CommandReport, CommandTarget, Commands, PendingEntity};
pub use entity::Entity;
pub use event::{Event, EventCadence, EventCursor, Events};
pub use schedule::{Schedule, ScheduleError, SystemMetrics, SystemMetricsView};
pub use storage::Component;
pub use world::{ComponentMetrics, EcsError, Resource, World, WorldMetrics};
