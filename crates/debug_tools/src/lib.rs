//! Developer-facing debug tooling.
//!
//! Performance overlays, entity/component inspection, event-log and replay
//! inspection, and review of generated content candidates. Debug-only: nothing
//! here may become required for gameplay.
//!
//! Milestone M5 onward.
//!
//! # Text in, text out
//!
//! Every tool here is a pure function from state to labelled text. Nothing
//! draws, nothing owns a window, and nothing mutates what it inspects. Three
//! consequences, all deliberate:
//!
//! - **The tools are testable headless**, so a regression in what the overlay
//!   reports fails in CI rather than being noticed on a screen weeks later.
//! - **The same view works in a log, a terminal, or on the HUD**, which matters
//!   because the machine where something goes wrong is often the one without a
//!   debugger attached.
//! - **Nothing in the game can come to depend on a debug tool**, since none of
//!   them produce anything but strings.
//!
//! # Inspection is built on the metrics the ECS already exposes
//!
//! The entity view reads [`WorldMetrics`](engine::ecs::WorldMetrics) rather than
//! enumerating components generically. The ECS is statically typed and has no
//! reflection, which is the right trade for the slice — and adding one to make a
//! debug view prettier would be exactly the speculative machinery
//! The project defers speculative machinery until the vertical slice needs it.

pub mod domain;
pub mod overlay;

pub use domain::{Section, inspect};
pub use overlay::{Budget, Row, Status, over_budget, rows};
