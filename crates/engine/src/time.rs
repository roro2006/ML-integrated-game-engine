//! Frame clock and fixed-timestep accumulator.
//!
//! Both are driven by explicitly supplied instants/durations so they can be
//! tested headlessly and replayed deterministically.

use std::fmt;
use std::time::{Duration, Instant};

use crate::ecs::Resource;

/// Measures wall-clock time between frames, clamped to a maximum delta so a
/// stall (debugger, window drag, load) cannot trigger a long catch-up.
#[derive(Debug, Clone)]
pub struct FrameClock {
    last: Option<Instant>,
    max_delta: Duration,
}

impl FrameClock {
    pub fn new(max_delta: Duration) -> Self {
        Self {
            last: None,
            max_delta,
        }
    }

    /// Records a frame boundary at `now`. The first call returns zero.
    pub fn tick(&mut self, now: Instant) -> Duration {
        let delta = self
            .last
            .map_or(Duration::ZERO, |last| now.saturating_duration_since(last));
        self.last = Some(now);
        delta.min(self.max_delta)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimeError {
    ZeroTickRate,
    ZeroMaxSteps,
}

impl fmt::Display for TimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroTickRate => f.write_str("fixed tick rate must be greater than zero"),
            Self::ZeroMaxSteps => f.write_str("max fixed steps per frame must be greater than zero"),
        }
    }
}

impl std::error::Error for TimeError {}

/// Fixed-timestep accumulator.
///
/// Frame time is accumulated and converted into whole simulation steps. At
/// most `max_steps_per_frame` steps run per frame; whole steps beyond that are
/// dropped (and counted) instead of spiralling. Sub-step remainder is kept.
#[derive(Debug, Clone)]
pub struct FixedTimestep {
    step: Duration,
    max_steps_per_frame: u32,
    accumulator: Duration,
    dropped: Duration,
}

impl FixedTimestep {
    pub fn new(step: Duration, max_steps_per_frame: u32) -> Result<Self, TimeError> {
        if step.is_zero() {
            return Err(TimeError::ZeroTickRate);
        }
        if max_steps_per_frame == 0 {
            return Err(TimeError::ZeroMaxSteps);
        }
        Ok(Self {
            step,
            max_steps_per_frame,
            accumulator: Duration::ZERO,
            dropped: Duration::ZERO,
        })
    }

    pub fn from_hz(hz: u32, max_steps_per_frame: u32) -> Result<Self, TimeError> {
        if hz == 0 {
            return Err(TimeError::ZeroTickRate);
        }
        Self::new(
            Duration::from_nanos(1_000_000_000 / u64::from(hz)),
            max_steps_per_frame,
        )
    }

    /// Adds frame time and returns how many fixed steps to run now.
    pub fn advance(&mut self, frame_delta: Duration) -> u32 {
        self.accumulator += frame_delta;
        let mut steps = 0;
        while self.accumulator >= self.step && steps < self.max_steps_per_frame {
            self.accumulator -= self.step;
            steps += 1;
        }
        if self.accumulator >= self.step {
            let step_nanos = self.step.as_nanos();
            let remainder = Duration::from_nanos((self.accumulator.as_nanos() % step_nanos) as u64);
            self.dropped += self.accumulator - remainder;
            self.accumulator = remainder;
        }
        steps
    }

    pub fn step(&self) -> Duration {
        self.step
    }

    /// Fraction of a step accumulated but not yet simulated, for interpolation.
    pub fn alpha(&self) -> f32 {
        (self.accumulator.as_secs_f64() / self.step.as_secs_f64()) as f32
    }

    /// Total simulation time discarded because a frame exceeded the step cap.
    pub fn dropped(&self) -> Duration {
        self.dropped
    }
}

/// Variable-rate frame timing, updated once per frame before frame systems run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FrameTime {
    /// Clamped duration of the current frame.
    pub delta: Duration,
    /// Sum of clamped frame deltas.
    pub elapsed: Duration,
    /// Index of the current frame, starting at 0.
    pub frame: u64,
}

impl Resource for FrameTime {}

/// Fixed-step timing, updated before each fixed tick.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FixedTime {
    /// Simulated duration of one tick. Use this, never frame delta, in fixed systems.
    pub step: Duration,
    /// Index of the current (or most recent) tick, starting at 0.
    pub tick: u64,
    /// Interpolation fraction toward the next tick; valid in frame systems.
    pub alpha: f32,
}

impl Resource for FixedTime {}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: Duration = Duration::from_millis(1);

    #[test]
    fn frame_clock_first_tick_is_zero_and_clamps() {
        let start = Instant::now();
        let mut clock = FrameClock::new(100 * MS);
        assert_eq!(clock.tick(start), Duration::ZERO);
        assert_eq!(clock.tick(start + 16 * MS), 16 * MS);
        assert_eq!(clock.tick(start + 1016 * MS), 100 * MS);
        assert_eq!(
            clock.tick(start + 1000 * MS),
            Duration::ZERO,
            "time going backwards saturates"
        );
    }

    #[test]
    fn rejects_invalid_configuration() {
        assert_eq!(FixedTimestep::from_hz(0, 5).unwrap_err(), TimeError::ZeroTickRate);
        assert_eq!(FixedTimestep::new(MS, 0).unwrap_err(), TimeError::ZeroMaxSteps);
    }

    #[test]
    fn accumulates_partial_steps() {
        let mut timestep = FixedTimestep::new(10 * MS, 5).unwrap();
        assert_eq!(timestep.advance(4 * MS), 0);
        assert_eq!(timestep.advance(4 * MS), 0);
        assert!((timestep.alpha() - 0.8).abs() < 1e-6);
        assert_eq!(timestep.advance(4 * MS), 1);
        assert_eq!(timestep.advance(25 * MS), 2);
        assert!((timestep.alpha() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn caps_steps_and_counts_dropped_time() {
        let mut timestep = FixedTimestep::new(10 * MS, 3).unwrap();
        assert_eq!(timestep.advance(75 * MS), 3);
        assert_eq!(timestep.dropped(), 40 * MS);
        assert!((timestep.alpha() - 0.5).abs() < 1e-6);
        assert_eq!(timestep.advance(5 * MS), 1);
    }

    #[test]
    fn step_count_is_independent_of_frame_slicing() {
        let mut coarse = FixedTimestep::from_hz(60, 1000).unwrap();
        let mut fine = FixedTimestep::from_hz(60, 1000).unwrap();
        let coarse_steps: u32 = (0..10).map(|_| coarse.advance(100 * MS)).sum();
        let fine_steps: u32 = (0..1000).map(|_| fine.advance(MS)).sum();
        assert_eq!(coarse_steps, fine_steps);
        assert_eq!(coarse_steps, 60);
    }
}
