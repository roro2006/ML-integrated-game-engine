//! Engine runtime.
//!
//! Owns the application lifecycle: the frame loop, fixed-timestep simulation,
//! the ECS, input state, diagnostics, and the renderer facade. Games build an
//! [`App`], register systems, and call [`run`].
//!
//! Layering (see `docs/adr/0002-workspace-and-crate-layout.md`):
//! `sample_game -> game_runtime -> engine -> platform | renderer`.

pub mod app;
pub mod camera;
pub mod debug_draw;
pub mod diagnostics;
pub mod ecs;
mod error;
pub mod input;
pub mod math;
pub mod pixel_text;
pub mod primitives;
pub mod render;
mod runner;
pub mod shapes;
pub mod time;
pub mod time_of_day;

/// Engine path services, re-exported from the platform layer.
pub use platform::paths;

// Resolved paths are published to the world so game systems can locate saves
// and assets without touching the environment themselves.
impl ecs::Resource for paths::EnginePaths {}

pub use app::{App, AppConfig, AppControl, FixedStage, FrameReport, FrameStage};
pub use error::EngineError;
pub use runner::run;
