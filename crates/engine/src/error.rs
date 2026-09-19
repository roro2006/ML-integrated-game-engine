//! Top-level engine errors.

use std::fmt;

use crate::ecs::ScheduleError;
use crate::time::TimeError;
use platform::PlatformError;
use platform::paths::PathError;
use renderer::RenderError;

#[derive(Debug)]
pub enum EngineError {
    Time(TimeError),
    Schedule(ScheduleError),
    Paths(PathError),
    Platform(PlatformError),
    Render(RenderError),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Time(error) => write!(f, "timing configuration error: {error}"),
            Self::Schedule(error) => write!(f, "schedule error: {error}"),
            Self::Paths(error) => write!(f, "path error: {error}"),
            Self::Platform(error) => write!(f, "platform error: {error}"),
            Self::Render(error) => write!(f, "render error: {error}"),
        }
    }
}

impl std::error::Error for EngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Time(error) => Some(error),
            Self::Schedule(error) => Some(error),
            Self::Paths(error) => Some(error),
            Self::Platform(error) => Some(error),
            Self::Render(error) => Some(error),
        }
    }
}

impl From<TimeError> for EngineError {
    fn from(error: TimeError) -> Self {
        Self::Time(error)
    }
}

impl From<ScheduleError> for EngineError {
    fn from(error: ScheduleError) -> Self {
        Self::Schedule(error)
    }
}

impl From<PathError> for EngineError {
    fn from(error: PathError) -> Self {
        Self::Paths(error)
    }
}

impl From<PlatformError> for EngineError {
    fn from(error: PlatformError) -> Self {
        Self::Platform(error)
    }
}

impl From<RenderError> for EngineError {
    fn from(error: RenderError) -> Self {
        Self::Render(error)
    }
}
