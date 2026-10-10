//! Engine-owned structured logging and frame diagnostics.
//!
//! Logging is deliberately small: a global level filter, a stderr sink, and an
//! optional file sink. Messages are static text; variable data goes in
//! `key = value` fields so logs stay greppable and machine-parseable.
//! See `docs/adr/0006-diagnostics-and-math-without-external-crates.md`.
//!
//! ```ignore
//! engine::log_info!("renderer", "adapter selected", name = info.name, backend = info.backend);
//! ```

use std::collections::VecDeque;
use std::fmt::{self, Write as _};
use std::fs::File;
use std::io::{self, LineWriter, Write as _};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::ecs::Resource;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl Level {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Error => "ERROR",
            Self::Warn => "WARN",
            Self::Info => "INFO",
            Self::Debug => "DEBUG",
            Self::Trace => "TRACE",
        }
    }
}

const DEFAULT_LEVEL: Level = if cfg!(debug_assertions) {
    Level::Debug
} else {
    Level::Info
};

/// One log record. Construct through the `log_*!` macros.
pub struct Record<'a> {
    pub level: Level,
    pub target: &'static str,
    pub message: &'static str,
    pub fields: &'a [(&'static str, &'a dyn fmt::Display)],
}

impl fmt::Debug for Record<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Record")
            .field("level", &self.level)
            .field("target", &self.target)
            .field("message", &self.message)
            .field("fields", &self.fields.len())
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct LogConfig {
    pub max_level: Level,
    /// File to (re)create for this run's log output.
    pub file: Option<PathBuf>,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            max_level: DEFAULT_LEVEL,
            file: None,
        }
    }
}

#[derive(Debug)]
pub enum LogInitError {
    AlreadyInitialized,
    File { path: PathBuf, source: io::Error },
}

impl fmt::Display for LogInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyInitialized => f.write_str("logger already initialized"),
            Self::File { path, source } => {
                write!(f, "cannot create log file {}: {source}", path.display())
            }
        }
    }
}

impl std::error::Error for LogInitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::AlreadyInitialized => None,
            Self::File { source, .. } => Some(source),
        }
    }
}

struct Logger {
    max_level: Level,
    file: Option<Mutex<LineWriter<File>>>,
}

static LOGGER: OnceLock<Logger> = OnceLock::new();
static START: OnceLock<Instant> = OnceLock::new();

/// Installs the global logger. Records emitted before this go to stderr at the
/// default level.
pub fn init(config: LogConfig) -> Result<(), LogInitError> {
    START.get_or_init(Instant::now);
    if LOGGER.get().is_some() {
        return Err(LogInitError::AlreadyInitialized);
    }
    let file = match config.file {
        Some(path) => {
            let file = File::create(&path).map_err(|source| LogInitError::File { path, source })?;
            Some(Mutex::new(LineWriter::new(file)))
        }
        None => None,
    };
    LOGGER
        .set(Logger {
            max_level: config.max_level,
            file,
        })
        .map_err(|_| LogInitError::AlreadyInitialized)
}

pub fn enabled(level: Level) -> bool {
    level <= LOGGER.get().map_or(DEFAULT_LEVEL, |logger| logger.max_level)
}

pub fn emit(record: &Record<'_>) {
    let elapsed = START.get_or_init(Instant::now).elapsed();
    let line = format_record(elapsed, record);
    // Logging must never take the engine down; sink write failures are ignored.
    let _ = writeln!(io::stderr().lock(), "{line}");
    if let Some(Logger { file: Some(file), .. }) = LOGGER.get()
        && let Ok(mut writer) = file.lock()
    {
        let _ = writeln!(writer, "{line}");
    }
}

/// Formats `[   1.234s LEVEL target] message key=value key="spaced value"`.
pub fn format_record(elapsed: Duration, record: &Record<'_>) -> String {
    let mut line = String::with_capacity(96);
    let _ = write!(
        line,
        "[{:>9.3}s {:<5} {}] {}",
        elapsed.as_secs_f64(),
        record.level.as_str(),
        record.target,
        record.message
    );
    for (key, value) in record.fields {
        let value = value.to_string();
        if value.is_empty() || value.contains(char::is_whitespace) || value.contains('"') {
            let _ = write!(line, " {key}={value:?}");
        } else {
            let _ = write!(line, " {key}={value}");
        }
    }
    line
}

#[macro_export]
macro_rules! log_event {
    ($level:expr, $target:expr, $message:expr $(, $key:ident = $value:expr)* $(,)?) => {{
        let level = $level;
        if $crate::diagnostics::enabled(level) {
            $crate::diagnostics::emit(&$crate::diagnostics::Record {
                level,
                target: $target,
                message: $message,
                fields: &[$((::core::stringify!($key), &$value as &dyn ::core::fmt::Display)),*],
            });
        }
    }};
}

#[macro_export]
macro_rules! log_error {
    ($($args:tt)*) => { $crate::log_event!($crate::diagnostics::Level::Error, $($args)*) };
}

#[macro_export]
macro_rules! log_warn {
    ($($args:tt)*) => { $crate::log_event!($crate::diagnostics::Level::Warn, $($args)*) };
}

#[macro_export]
macro_rules! log_info {
    ($($args:tt)*) => { $crate::log_event!($crate::diagnostics::Level::Info, $($args)*) };
}

#[macro_export]
macro_rules! log_debug {
    ($($args:tt)*) => { $crate::log_event!($crate::diagnostics::Level::Debug, $($args)*) };
}

#[macro_export]
macro_rules! log_trace {
    ($($args:tt)*) => { $crate::log_event!($crate::diagnostics::Level::Trace, $($args)*) };
}

const FRAME_SAMPLES: usize = 120;

/// Rolling frame statistics, stored as a world resource.
#[derive(Debug, Clone, Default)]
pub struct FrameDiagnostics {
    samples: VecDeque<Duration>,
    pub frames: u64,
    pub fixed_ticks: u64,
    /// Draw calls submitted in the most recent rendered frame.
    pub draw_calls: u32,
    /// Triangles submitted in the most recent rendered frame, after culling.
    pub triangles: u32,
    /// CPU time spent in the renderer in the most recent frame.
    pub render_cpu_time: Duration,
    /// Latest completed GPU sample; unavailable timestamps are never reported as zero.
    pub gpu_passes: Option<Vec<crate::render::PassTiming>>,
    /// Resources and execution order of the most recently recorded render graph.
    pub render_graph: String,
}

impl Resource for FrameDiagnostics {}

impl FrameDiagnostics {
    pub fn record_frame(&mut self, frame_time: Duration, fixed_ticks: u32) {
        if self.samples.len() == FRAME_SAMPLES {
            self.samples.pop_front();
        }
        self.samples.push_back(frame_time);
        self.frames += 1;
        self.fixed_ticks += u64::from(fixed_ticks);
    }

    /// Mean frame time over the rolling window.
    pub fn average_frame_time(&self) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        self.samples.iter().sum::<Duration>() / self.samples.len() as u32
    }

    /// Worst frame time over the rolling window.
    pub fn max_frame_time(&self) -> Duration {
        self.samples.iter().copied().max().unwrap_or_default()
    }

    pub fn fps(&self) -> f64 {
        let average = self.average_frame_time().as_secs_f64();
        if average > 0.0 { 1.0 / average } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_fields_and_quotes_when_needed() {
        let width = 1280;
        let name = "Intel Iris Xe";
        let record = Record {
            level: Level::Info,
            target: "renderer",
            message: "adapter selected",
            fields: &[("width", &width), ("name", &name), ("empty", &"")],
        };
        let line = format_record(Duration::from_millis(1500), &record);
        assert_eq!(
            line,
            "[    1.500s INFO  renderer] adapter selected width=1280 name=\"Intel Iris Xe\" empty=\"\""
        );
    }

    #[test]
    fn level_ordering_filters_verbose_levels() {
        assert!(Level::Error < Level::Warn);
        assert!(Level::Debug < Level::Trace);
    }

    #[test]
    fn frame_diagnostics_rolling_window() {
        let mut diagnostics = FrameDiagnostics::default();
        assert_eq!(diagnostics.fps(), 0.0);
        for _ in 0..FRAME_SAMPLES {
            diagnostics.record_frame(Duration::from_millis(40), 2);
        }
        for _ in 0..FRAME_SAMPLES {
            diagnostics.record_frame(Duration::from_millis(10), 1);
        }
        assert_eq!(diagnostics.average_frame_time(), Duration::from_millis(10));
        assert_eq!(diagnostics.max_frame_time(), Duration::from_millis(10));
        assert!((diagnostics.fps() - 100.0).abs() < 1e-9);
        assert_eq!(diagnostics.frames, 2 * FRAME_SAMPLES as u64);
        assert_eq!(diagnostics.fixed_ticks, 3 * FRAME_SAMPLES as u64);
    }
}
