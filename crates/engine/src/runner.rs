//! Windowed runner: owns the platform loop, the renderer, and diagnostics.

use std::time::{Duration, Instant};

use platform::paths::{EnginePaths, PathEnvironment, asset_root_override_var};
use platform::{LoopAction, PlatformHandler, PlatformWindow, WindowConfig};
use renderer::{
    FrameOutcome, MaterialDesc, MaterialHandle, MeshData, MeshHandle, RenderError, RenderFrame, Renderer,
};

use crate::app::{App, FrameReport};
use crate::debug_draw::DebugDraw;
use crate::diagnostics::{self, FrameDiagnostics, LogConfig};
use crate::error::EngineError;
use crate::input::PlatformEvent;
use crate::render::{ActiveAdapter, RenderAssets, RenderExtractor};
use crate::time::FrameClock;

/// Resolves paths, installs logging, opens a window, and runs until exit.
pub fn run(app: App) -> Result<(), EngineError> {
    let config = app.config().clone();
    let environment = PathEnvironment::from_process(&config.app_id);
    let paths = EnginePaths::resolve(&config.app_id, &environment)?;
    paths.create_writable_dirs()?;

    if let Err(error) = diagnostics::init(LogConfig {
        file: Some(paths.logs.join("latest.log")),
        ..LogConfig::default()
    }) {
        crate::log_warn!("diagnostics", "file logging unavailable", error = error);
    }

    crate::log_info!(
        "engine",
        "starting",
        app = config.app_id,
        fixed_hz = config.fixed_hz,
        max_fixed_steps = config.max_fixed_steps_per_frame,
    );
    crate::log_info!(
        "paths",
        "user directories resolved",
        config = paths.config.display(),
        saves = paths.saves.display(),
        cache = paths.cache.display(),
        logs = paths.logs.display(),
    );
    match &paths.assets {
        Some(root) => crate::log_info!("paths", "asset root resolved", path = root.display()),
        None => crate::log_warn!(
            "paths",
            "no asset root found; content loading will be unavailable",
            override_variable = asset_root_override_var(&config.app_id),
        ),
    }

    let mut runner = Runner {
        app,
        renderer: None,
        clock: FrameClock::new(config.max_frame_delta),
        report_interval: config.diagnostics_interval,
        last_report: None,
        error: None,
        frame: RenderFrame::default(),
        last_extract: (0, 0),
    };
    runner.app.world_mut().insert_resource(paths);

    let window_config = WindowConfig {
        title: config.title.clone(),
        width: config.window_width,
        height: config.window_height,
        resizable: true,
    };
    let loop_result = platform::run(window_config, &mut runner);

    let diagnostics = runner.app.world().resource::<FrameDiagnostics>();
    crate::log_info!(
        "engine",
        "stopped",
        frames = diagnostics.map_or(0, |d| d.frames),
        fixed_ticks = diagnostics.map_or(0, |d| d.fixed_ticks),
    );

    // A renderer failure recorded during the loop is more specific than the
    // loop's own result, so it is reported first.
    match runner.error.take() {
        Some(error) => Err(error),
        None => loop_result.map_err(EngineError::from),
    }
}

struct Runner {
    app: App,
    renderer: Option<Renderer>,
    clock: FrameClock,
    report_interval: Duration,
    last_report: Option<Instant>,
    error: Option<EngineError>,
    /// Reused across frames so steady-state extraction allocates nothing.
    frame: RenderFrame,
    /// Most recent (drawn, culled) counts, for the periodic report.
    last_extract: (u32, u32),
}

/// [`RenderAssets`] backed by the live renderer, handed to startup hooks once
/// the GPU exists.
struct GpuRenderAssets<'a> {
    renderer: &'a mut Renderer,
}

impl RenderAssets for GpuRenderAssets<'_> {
    fn upload_mesh(&mut self, mesh: &MeshData) -> Result<MeshHandle, RenderError> {
        self.renderer.upload_mesh(mesh)
    }

    fn create_material(&mut self, desc: &MaterialDesc) -> MaterialHandle {
        self.renderer.create_material(desc)
    }
}

impl Runner {
    fn report_diagnostics(&mut self, report: FrameReport) {
        let now = Instant::now();
        let due = self
            .last_report
            .is_none_or(|last| now.duration_since(last) >= self.report_interval);
        if !due {
            return;
        }
        self.last_report = Some(now);
        let Some(diagnostics) = self.app.world().resource::<FrameDiagnostics>() else {
            return;
        };
        crate::log_info!(
            "diagnostics",
            "frame timing",
            fps = format_args!("{:.1}", diagnostics.fps()),
            avg_ms = format_args!("{:.2}", diagnostics.average_frame_time().as_secs_f64() * 1000.0),
            max_ms = format_args!("{:.2}", diagnostics.max_frame_time().as_secs_f64() * 1000.0),
            frames = diagnostics.frames,
            fixed_ticks = diagnostics.fixed_ticks,
            last_frame_steps = report.fixed_steps,
            draw_calls = diagnostics.draw_calls,
            triangles = diagnostics.triangles,
            drawn = self.last_extract.0,
            culled = self.last_extract.1,
        );

        // A hot-reloaded shader that failed to compile keeps the previous
        // pipeline, so it is only visible if something surfaces it.
        if let Some(renderer) = &self.renderer {
            if let Some(error) = renderer.shader_error() {
                crate::log_warn!(
                    "renderer",
                    "using last good shader after a reload failure",
                    error = error
                );
            }
            let skipped = renderer.skipped_draws();
            if skipped > 0 {
                crate::log_warn!("renderer", "draws skipped for unknown handles", count = skipped);
            }
        }
    }
}

impl PlatformHandler for Runner {
    fn window_created(&mut self, window: &PlatformWindow) -> LoopAction {
        let (width, height) = window.inner_size();
        match Renderer::new(window.clone(), width, height, self.app.config().renderer) {
            Ok(mut renderer) => {
                let info = renderer.adapter_info().clone();
                crate::log_info!(
                    "renderer",
                    "adapter selected",
                    name = info.name,
                    backend = info.backend,
                    kind = info.device_kind,
                    driver = info.driver,
                    width = width,
                    height = height,
                );
                self.app.world_mut().insert_resource(ActiveAdapter(info));

                // Shader hot reload is a development convenience; a missing
                // asset root must not stop the game from starting.
                let shader_dir = self
                    .app
                    .world()
                    .resource::<EnginePaths>()
                    .and_then(|paths| paths.assets.as_ref().map(|root| root.join("shaders")));
                match shader_dir {
                    Some(dir) => {
                        if let Err(error) = renderer.enable_shader_hot_reload(dir) {
                            crate::log_warn!("renderer", "shader hot reload unavailable", error = error);
                        }
                    }
                    None => crate::log_debug!("renderer", "no asset root; shaders stay embedded"),
                }

                // Scene setup needs the GPU, so it runs here rather than at App::new.
                let mut assets = GpuRenderAssets {
                    renderer: &mut renderer,
                };
                self.app.run_startup_hooks(&mut assets);

                // What the game needs to turn a cursor into a point in the
                // world; kept current on resize.
                let (width, height) = renderer.surface_size();
                self.app
                    .world_mut()
                    .insert_resource(crate::camera::Viewport { width, height });
                self.renderer = Some(renderer);
                LoopAction::Continue
            }
            Err(error) => {
                crate::log_error!("renderer", "initialisation failed", error = error);
                self.error = Some(error.into());
                LoopAction::Exit
            }
        }
    }

    fn event(&mut self, _window: &PlatformWindow, event: PlatformEvent) -> LoopAction {
        if let PlatformEvent::Resized { width, height } = event
            && let Some(renderer) = &mut self.renderer
        {
            renderer.resize(width, height);
            let (width, height) = renderer.surface_size();
            self.app
                .world_mut()
                .insert_resource(crate::camera::Viewport { width, height });
        }
        self.app.handle_platform_event(&event);
        self.loop_action()
    }

    fn frame(&mut self, _window: &PlatformWindow) -> LoopAction {
        let delta = self.clock.tick(Instant::now());
        let report = self.app.update(delta);

        if let Some(renderer) = &mut self.renderer {
            let (width, height) = renderer.surface_size();
            let aspect = if height == 0 {
                0.0
            } else {
                width as f32 / height as f32
            };

            // Extraction reads the world and fills the reusable frame: culling
            // and ordering are engine work, so the backend receives a finished
            // list (docs/plans/m3-renderer.md).
            let scratch = &mut self.frame;
            let world = self.app.world_mut();
            if !world.has_resource::<RenderExtractor>() {
                world.insert_resource(RenderExtractor::new());
            }
            if let Some(stats) = world.resource_scope::<RenderExtractor, _>(|world, extractor| {
                extractor.extract(world, aspect, scratch)
            }) {
                self.last_extract = (stats.drawn, stats.culled);
            }

            // Debug lines are accumulated by frame systems and handed over
            // here, after extraction, so extraction stays read-only over the
            // world. `drain_into` swaps buffers, so neither side reallocates.
            if let Some(debug) = self.app.world_mut().resource_mut::<DebugDraw>() {
                debug.drain_into(&mut self.frame);
            }

            match renderer.render(&self.frame) {
                Ok(FrameOutcome::Presented(stats)) => {
                    if let Some(diagnostics) = self.app.world_mut().resource_mut::<FrameDiagnostics>() {
                        diagnostics.draw_calls = stats.draw_calls;
                        diagnostics.triangles = stats.triangles;
                        diagnostics.render_cpu_time = stats.cpu_time;
                    }
                }
                Ok(FrameOutcome::Skipped(reason)) => {
                    crate::log_trace!("renderer", "frame skipped", reason = format_args!("{reason:?}"));
                }
                Err(error) => {
                    crate::log_error!("renderer", "frame failed", error = error);
                    self.error = Some(error.into());
                    return LoopAction::Exit;
                }
            }
        }

        self.report_diagnostics(report);
        self.loop_action()
    }

    fn exiting(&mut self) {
        crate::log_info!("engine", "event loop exiting");
    }
}

impl Runner {
    fn loop_action(&self) -> LoopAction {
        if self.app.exit_requested() {
            LoopAction::Exit
        } else {
            LoopAction::Continue
        }
    }
}
