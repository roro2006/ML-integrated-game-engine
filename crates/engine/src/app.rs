//! Application lifecycle: world, schedules, and the frame/tick boundary.

use std::fmt;
use std::time::Duration;

use crate::diagnostics::FrameDiagnostics;
use crate::ecs::{Commands, Event, EventCadence, Resource, Schedule, SystemMetricsView, World};
use crate::error::EngineError;
use crate::input::{FixedInput, FrameInput, PlatformEvent};
use crate::render::{ClearColor, Color, HeadlessRenderAssets, RenderAssets, RendererConfig};
use crate::time::{FixedTime, FixedTimestep, FrameTime};

#[derive(Debug, Clone)]
pub struct AppConfig {
    /// Lowercase kebab-case identifier used for config/save/log directories.
    pub app_id: String,
    pub title: String,
    pub window_width: u32,
    pub window_height: u32,
    pub fixed_hz: u32,
    /// Upper bound on simulation steps per frame; excess time is dropped.
    pub max_fixed_steps_per_frame: u32,
    /// Frame deltas longer than this are clamped (debugger pauses, window drags).
    pub max_frame_delta: Duration,
    pub renderer: RendererConfig,
    /// How often the runner logs a frame-time summary.
    pub diagnostics_interval: Duration,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            app_id: "engine-app".to_owned(),
            title: "Engine".to_owned(),
            window_width: 1280,
            window_height: 720,
            fixed_hz: 60,
            max_fixed_steps_per_frame: 5,
            max_frame_delta: Duration::from_millis(250),
            renderer: RendererConfig::default(),
            diagnostics_interval: Duration::from_secs(5),
        }
    }
}

/// Stages of the fixed-timestep schedule, run in this order every tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedStage {
    PreUpdate,
    Update,
    PostUpdate,
}

impl FixedStage {
    pub(crate) const ALL: [Self; 3] = [Self::PreUpdate, Self::Update, Self::PostUpdate];

    pub const fn label(self) -> &'static str {
        match self {
            Self::PreUpdate => "fixed-pre-update",
            Self::Update => "fixed-update",
            Self::PostUpdate => "fixed-post-update",
        }
    }
}

/// Stages of the variable-rate schedule, run once per frame after fixed ticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameStage {
    PreUpdate,
    Update,
    PostUpdate,
}

impl FrameStage {
    pub(crate) const ALL: [Self; 3] = [Self::PreUpdate, Self::Update, Self::PostUpdate];

    pub const fn label(self) -> &'static str {
        match self {
            Self::PreUpdate => "pre-update",
            Self::Update => "update",
            Self::PostUpdate => "post-update",
        }
    }
}

/// Lifecycle control available to systems.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AppControl {
    exit_requested: bool,
}

impl Resource for AppControl {}

impl AppControl {
    /// Requests a clean shutdown after the current frame.
    pub fn request_exit(&mut self) {
        self.exit_requested = true;
    }

    pub fn exit_requested(self) -> bool {
        self.exit_requested
    }
}

/// What one [`App::update`] did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameReport {
    pub frame: u64,
    /// Frame delta after clamping.
    pub delta: Duration,
    pub fixed_steps: u32,
}

/// The engine application: a world, two schedules, and the timing that drives them.
///
/// [`update`](Self::update) is independent of any window, so games and tests
/// can drive the whole simulation headlessly.
/// Work that runs once, after the renderer exists: upload meshes, register
/// materials, spawn the scene.
///
/// Held in a newtype with a hand-written `Debug` because boxed closures have
/// none, and `App` stays `Debug` for the inspector tooling.
type StartupHook = Box<dyn FnOnce(&mut World, &mut dyn RenderAssets)>;

#[derive(Default)]
struct StartupHooks(Vec<StartupHook>);

impl fmt::Debug for StartupHooks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StartupHooks")
            .field("pending", &self.0.len())
            .finish()
    }
}

#[derive(Debug)]
pub struct App {
    config: AppConfig,
    world: World,
    fixed_schedule: Schedule,
    frame_schedule: Schedule,
    timestep: FixedTimestep,
    frames: u64,
    ticks: u64,
    elapsed: Duration,
    startup_hooks: StartupHooks,
}

impl App {
    pub fn new(config: AppConfig) -> Result<Self, EngineError> {
        platform::paths::validate_app_id(&config.app_id)?;
        let timestep = FixedTimestep::from_hz(config.fixed_hz, config.max_fixed_steps_per_frame)?;

        let mut fixed_schedule = Schedule::new("fixed");
        for stage in FixedStage::ALL {
            fixed_schedule.add_stage(stage.label())?;
        }
        let mut frame_schedule = Schedule::new("frame");
        for stage in FrameStage::ALL {
            frame_schedule.add_stage(stage.label())?;
        }

        let mut world = World::new();
        world.insert_resource(FrameTime::default());
        world.insert_resource(FixedTime {
            step: timestep.step(),
            tick: 0,
            alpha: 0.0,
        });
        world.insert_resource(FrameInput::default());
        world.insert_resource(FixedInput::default());
        world.insert_resource(AppControl::default());
        world.insert_resource(ClearColor::default());
        world.insert_resource(FrameDiagnostics::default());

        Ok(Self {
            config,
            world,
            fixed_schedule,
            frame_schedule,
            timestep,
            frames: 0,
            ticks: 0,
            elapsed: Duration::ZERO,
            startup_hooks: StartupHooks::default(),
        })
    }

    pub fn config(&self) -> &AppConfig {
        &self.config
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// Registers scene setup that runs once, as soon as asset registration is
    /// possible: in a windowed run that is just after the renderer exists, and
    /// in a headless run it is immediately, against fabricated handles.
    ///
    /// Meshes cannot be uploaded at `App::new` time because there is no GPU
    /// yet, so scene building is deferred rather than split across two places.
    pub fn add_startup_hook(
        &mut self,
        hook: impl FnOnce(&mut World, &mut dyn RenderAssets) + 'static,
    ) -> &mut Self {
        self.startup_hooks.0.push(Box::new(hook));
        self
    }

    /// Runs and consumes every registered startup hook. Called by the runner.
    pub fn run_startup_hooks(&mut self, assets: &mut dyn RenderAssets) {
        for hook in std::mem::take(&mut self.startup_hooks.0) {
            hook(&mut self.world, assets);
        }
    }

    /// Whether any startup hook is still pending.
    pub fn has_pending_startup_hooks(&self) -> bool {
        !self.startup_hooks.0.is_empty()
    }

    /// Adds a system to the fixed-timestep schedule.
    pub fn add_fixed_system(
        &mut self,
        stage: FixedStage,
        name: &'static str,
        system: impl FnMut(&mut World, &mut Commands) + 'static,
    ) -> Result<&mut Self, EngineError> {
        self.fixed_schedule.add_system(stage.label(), name, system)?;
        Ok(self)
    }

    /// Adds a system to the variable-rate frame schedule.
    pub fn add_frame_system(
        &mut self,
        stage: FrameStage,
        name: &'static str,
        system: impl FnMut(&mut World, &mut Commands) + 'static,
    ) -> Result<&mut Self, EngineError> {
        self.frame_schedule.add_system(stage.label(), name, system)?;
        Ok(self)
    }

    /// Registers an event channel; see [`EventCadence`] for lifetime rules.
    pub fn add_event<E: Event>(&mut self, cadence: EventCadence) -> &mut Self {
        self.world.add_event::<E>(cadence);
        self
    }

    /// Feeds a normalized platform event into both input views.
    pub fn handle_platform_event(&mut self, event: &PlatformEvent) {
        if matches!(event, PlatformEvent::CloseRequested) {
            self.request_exit();
        }
        if let Some(input) = self.world.resource_mut::<FrameInput>() {
            input.apply(event);
        }
        if let Some(input) = self.world.resource_mut::<FixedInput>() {
            input.apply(event);
        }
    }

    pub fn request_exit(&mut self) {
        if let Some(control) = self.world.resource_mut::<AppControl>() {
            control.request_exit();
        }
    }

    pub fn exit_requested(&self) -> bool {
        self.world
            .resource::<AppControl>()
            .is_some_and(|control| control.exit_requested())
    }

    pub fn clear_color(&self) -> Color {
        self.world.resource::<ClearColor>().copied().unwrap_or_default().0
    }

    /// Advances the application by one frame: fixed ticks first, then frame systems.
    ///
    /// Fixed-tick input edges are cleared once a tick has consumed them, so a
    /// press during a frame that runs no ticks is still observed later.
    pub fn update(&mut self, frame_delta: Duration) -> FrameReport {
        let delta = frame_delta.min(self.config.max_frame_delta);
        let frame = self.frames;
        self.frames += 1;
        self.elapsed += delta;
        if let Some(time) = self.world.resource_mut::<FrameTime>() {
            time.delta = delta;
            time.elapsed = self.elapsed;
            time.frame = frame;
        }

        let steps = self.timestep.advance(delta);
        for _ in 0..steps {
            if let Some(time) = self.world.resource_mut::<FixedTime>() {
                time.tick = self.ticks;
            }
            self.fixed_schedule.run(&mut self.world);
            self.world.update_events(EventCadence::FixedTick);
            if let Some(input) = self.world.resource_mut::<FixedInput>() {
                input.clear_edges();
            }
            self.ticks += 1;
        }

        if let Some(time) = self.world.resource_mut::<FixedTime>() {
            time.alpha = self.timestep.alpha();
        }
        self.frame_schedule.run(&mut self.world);
        self.world.update_events(EventCadence::Frame);
        if let Some(input) = self.world.resource_mut::<FrameInput>() {
            input.clear_edges();
        }
        if let Some(diagnostics) = self.world.resource_mut::<FrameDiagnostics>() {
            diagnostics.record_frame(frame_delta, steps);
        }

        FrameReport {
            frame,
            delta,
            fixed_steps: steps,
        }
    }

    /// Runs `frames` updates with a constant delta. For headless smoke tests.
    pub fn run_headless(&mut self, frames: u64, frame_delta: Duration) -> FrameReport {
        // Headless runs build the same scene as a windowed one, against
        // fabricated handles, so extraction and culling are exercised rather
        // than skipped over an empty world.
        if self.has_pending_startup_hooks() {
            let mut assets = HeadlessRenderAssets::default();
            self.run_startup_hooks(&mut assets);
        }
        let mut report = FrameReport {
            frame: 0,
            delta: Duration::ZERO,
            fixed_steps: 0,
        };
        for _ in 0..frames {
            report = self.update(frame_delta);
            if self.exit_requested() {
                break;
            }
        }
        report
    }

    pub fn fixed_metrics(&self) -> impl Iterator<Item = SystemMetricsView<'_>> {
        self.fixed_schedule.metrics()
    }

    pub fn frame_metrics(&self) -> impl Iterator<Item = SystemMetricsView<'_>> {
        self.frame_schedule.metrics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{ButtonState, KeyCode};

    #[derive(Default)]
    struct Counters {
        fixed: u32,
        frame: u32,
        taps: u32,
    }
    impl Resource for Counters {}

    fn test_app(fixed_hz: u32) -> App {
        let mut app = App::new(AppConfig {
            app_id: "test-app".to_owned(),
            fixed_hz,
            ..AppConfig::default()
        })
        .unwrap();
        app.world_mut().insert_resource(Counters::default());
        app.add_fixed_system(FixedStage::Update, "count-ticks", |world, _| {
            if let Some(counters) = world.resource_mut::<Counters>() {
                counters.fixed += 1;
            }
        })
        .unwrap();
        app.add_frame_system(FrameStage::Update, "count-frames", |world, _| {
            if let Some(counters) = world.resource_mut::<Counters>() {
                counters.frame += 1;
            }
        })
        .unwrap();
        app
    }

    fn counters(app: &App) -> (u32, u32, u32) {
        let counters = app.world().resource::<Counters>().unwrap();
        (counters.fixed, counters.frame, counters.taps)
    }

    #[test]
    fn rejects_invalid_configuration() {
        assert!(matches!(
            App::new(AppConfig {
                app_id: "Invalid App".to_owned(),
                ..AppConfig::default()
            }),
            Err(EngineError::Paths(_))
        ));
        assert!(matches!(
            App::new(AppConfig {
                fixed_hz: 0,
                ..AppConfig::default()
            }),
            Err(EngineError::Time(_))
        ));
    }

    #[test]
    fn fixed_systems_run_at_the_configured_rate() {
        let mut app = test_app(60);
        // Exactly one step per frame: the 60 Hz step is 16_666_666ns.
        app.run_headless(60, Duration::from_nanos(16_666_666));
        let (fixed, frame, _) = counters(&app);
        assert_eq!(frame, 60);
        assert_eq!(fixed, 60, "one tick per frame when delta equals the step");

        let report = app.update(Duration::ZERO);
        assert_eq!(report.frame, 60);
        assert_eq!(report.fixed_steps, 0, "a zero-length frame runs no ticks");
    }

    #[test]
    fn long_frames_are_clamped_and_capped() {
        let mut app = test_app(60);
        let report = app.update(Duration::from_secs(10));
        assert_eq!(report.delta, app.config().max_frame_delta);
        assert_eq!(report.fixed_steps, app.config().max_fixed_steps_per_frame);
    }

    #[test]
    fn fixed_input_edges_survive_frames_without_ticks() {
        // 10 Hz: a 100ms step, so short frames run no tick at all.
        let mut app = test_app(10);
        app.add_fixed_system(FixedStage::Update, "count-taps", |world, _| {
            let tapped = world
                .resource::<FixedInput>()
                .is_some_and(|input| input.keys.just_pressed(KeyCode::Space));
            if tapped && let Some(counters) = world.resource_mut::<Counters>() {
                counters.taps += 1;
            }
        })
        .unwrap();

        app.handle_platform_event(&PlatformEvent::Key {
            key: KeyCode::Space,
            state: ButtonState::Pressed,
            repeat: false,
        });
        app.handle_platform_event(&PlatformEvent::Key {
            key: KeyCode::Space,
            state: ButtonState::Released,
            repeat: false,
        });

        // Frames that run no tick must not discard the tap.
        for _ in 0..3 {
            let report = app.update(Duration::from_millis(10));
            assert_eq!(report.fixed_steps, 0);
        }
        assert_eq!(counters(&app).2, 0);
        assert!(
            !app.world()
                .resource::<FrameInput>()
                .unwrap()
                .keys
                .just_pressed(KeyCode::Space),
            "frame edges clear every frame"
        );

        let report = app.update(Duration::from_millis(100));
        assert_eq!(report.fixed_steps, 1);
        assert_eq!(counters(&app).2, 1, "the tick must observe the tap exactly once");
        let report = app.update(Duration::from_millis(100));
        assert_eq!(report.fixed_steps, 1);
        assert_eq!(counters(&app).2, 1, "the edge is consumed by the first tick only");
    }

    #[test]
    fn close_request_and_system_exit_stop_the_loop() {
        let mut app = test_app(60);
        assert!(!app.exit_requested());
        app.handle_platform_event(&PlatformEvent::CloseRequested);
        assert!(app.exit_requested());

        let mut app = test_app(60);
        app.add_frame_system(FrameStage::PostUpdate, "quit-on-escape", |world, _| {
            let pressed = world
                .resource::<FrameInput>()
                .is_some_and(|input| input.keys.just_pressed(KeyCode::Escape));
            if pressed && let Some(control) = world.resource_mut::<AppControl>() {
                control.request_exit();
            }
        })
        .unwrap();
        app.handle_platform_event(&PlatformEvent::Key {
            key: KeyCode::Escape,
            state: ButtonState::Pressed,
            repeat: false,
        });
        let report = app.run_headless(100, Duration::from_millis(16));
        assert!(app.exit_requested());
        assert_eq!(report.frame, 0, "run_headless stops at the requesting frame");
    }
}
