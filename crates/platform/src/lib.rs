//! Platform layer: windowing, normalized input, and path services.
//!
//! This is the lowest engine-owned layer. The windowing backend (winit) is
//! confined to [`window`]; nothing above this crate sees backend types except
//! the neutral `raw-window-handle` traits the renderer needs to create a
//! surface. See `docs/adr/0004-platform-and-input-backend.md`.

pub mod input;
pub mod paths;
mod window;

pub use window::{LoopAction, PlatformError, PlatformHandler, PlatformWindow, WindowConfig, run};
