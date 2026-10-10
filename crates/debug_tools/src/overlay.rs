//! The performance overlay.
//!
//! Turns [`FrameDiagnostics`] and [`WorldMetrics`] into labelled lines with a
//! verdict against a budget. Text, not pixels: drawing is the renderer's job,
//! and keeping this side pure means the overlay's *content* — including whether
//! something is over budget — is testable headless in CI, which is where a
//! regression in it would otherwise go unnoticed until someone looked at a
//! screen.
//!
//! # Budgets come from the device profile, not from here
//!
//! [`Budget`] takes its numbers from the caller. This module deliberately ships
//! no default draw-call or triangle limits: those belong to the profile being
//! validated, and a plausible-looking constant here would quietly become the
//! number everyone trusts.

use std::time::Duration;

use engine::diagnostics::FrameDiagnostics;
use engine::ecs::WorldMetrics;

/// Sixty frames per second, as a frame time.
pub const SIXTY_FPS: Duration = Duration::from_nanos(16_666_667);

/// What the overlay measures against.
///
/// `draw_calls` and `triangles` default to unbounded, so an unconfigured
/// overlay reports the numbers without pretending to judge them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    pub frame_time: Duration,
    pub draw_calls: u32,
    pub triangles: u32,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            frame_time: SIXTY_FPS,
            draw_calls: u32::MAX,
            triangles: u32::MAX,
        }
    }
}

impl Budget {
    /// A budget constraining frame time only.
    pub const fn frame_time(frame_time: Duration) -> Self {
        Self {
            frame_time,
            draw_calls: u32::MAX,
            triangles: u32::MAX,
        }
    }

    /// Adds render-submission limits from the device profile being validated.
    pub const fn with_submission(mut self, draw_calls: u32, triangles: u32) -> Self {
        self.draw_calls = draw_calls;
        self.triangles = triangles;
        self
    }
}

/// Whether a measurement is acceptable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Within budget, or not budgeted at all.
    Ok,
    /// Over budget. The overlay's whole purpose is making this obvious.
    Over,
}

impl Status {
    /// A marker for text output, so an over-budget row is visible in a log or a
    /// CI failure and not only in colour on a screen.
    pub const fn marker(self) -> &'static str {
        match self {
            Self::Ok => " ",
            Self::Over => "!",
        }
    }
}

/// One row of the overlay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub label: &'static str,
    pub value: String,
    pub status: Status,
}

impl Row {
    fn ok(label: &'static str, value: impl Into<String>) -> Self {
        Self {
            label,
            value: value.into(),
            status: Status::Ok,
        }
    }

    fn judged(label: &'static str, value: impl Into<String>, over: bool) -> Self {
        Self {
            label,
            value: value.into(),
            status: if over { Status::Over } else { Status::Ok },
        }
    }
}

fn millis(duration: Duration) -> String {
    format!("{:.2} ms", duration.as_secs_f64() * 1000.0)
}

/// The overlay's rows, in a fixed order so the display never reshuffles.
///
/// `world` is optional because the overlay is useful before a world exists — at
/// a menu, or while content is loading.
pub fn rows(diagnostics: &FrameDiagnostics, world: Option<&WorldMetrics>, budget: &Budget) -> Vec<Row> {
    let average = diagnostics.average_frame_time();
    let worst = diagnostics.max_frame_time();

    let mut rows = vec![
        Row::ok("fps", format!("{:.1}", diagnostics.fps())),
        Row::judged("frame avg", millis(average), average > budget.frame_time),
        // The worst frame in the window is judged too: a smooth average with a
        // 40 ms spike in it is a stutter the player feels and the mean hides.
        Row::judged("frame max", millis(worst), worst > budget.frame_time),
        Row::ok("render cpu", millis(diagnostics.render_cpu_time)),
        Row::ok(
            "gpu passes",
            diagnostics.gpu_passes.as_ref().map_or_else(
                || "unavailable".to_owned(),
                |passes| {
                    passes
                        .iter()
                        .map(|pass| {
                            format!(
                                "{}: {}",
                                pass.name,
                                pass.gpu_ms
                                    .map_or_else(|| "unavailable".to_owned(), |ms| format!("{ms:.3} ms"))
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                },
            ),
        ),
        Row::ok(
            "render graph",
            if diagnostics.render_graph.is_empty() {
                "unavailable".to_owned()
            } else {
                diagnostics.render_graph.clone()
            },
        ),
        Row::judged(
            "draw calls",
            diagnostics.draw_calls.to_string(),
            diagnostics.draw_calls > budget.draw_calls,
        ),
        Row::judged(
            "triangles",
            diagnostics.triangles.to_string(),
            diagnostics.triangles > budget.triangles,
        ),
        Row::ok("frames", diagnostics.frames.to_string()),
        Row::ok("fixed ticks", diagnostics.fixed_ticks.to_string()),
    ];

    if let Some(metrics) = world {
        rows.push(Row::ok("entities", metrics.entities.to_string()));
        rows.push(Row::ok("resources", metrics.resources.to_string()));
        rows.push(Row::ok("component types", metrics.components.len().to_string()));
    }

    rows
}

/// The rows as aligned text.
pub fn lines(rows: &[Row]) -> Vec<String> {
    let width = rows.iter().map(|row| row.label.len()).max().unwrap_or(0);
    rows.iter()
        .map(|row| {
            format!(
                "{} {:width$}  {}",
                row.status.marker(),
                row.label,
                row.value,
                width = width
            )
        })
        .collect()
}

/// Whether anything is over budget.
///
/// Lets a headless performance test fail on the same rule the overlay displays,
/// rather than on a second threshold that could drift away from it.
pub fn over_budget(rows: &[Row]) -> bool {
    rows.iter().any(|row| row.status == Status::Over)
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::ecs::{ComponentMetrics, WorldMetrics};

    fn diagnostics(frames: &[u64]) -> FrameDiagnostics {
        let mut diagnostics = FrameDiagnostics::default();
        for millis in frames {
            diagnostics.record_frame(Duration::from_millis(*millis), 1);
        }
        diagnostics
    }

    fn metrics() -> WorldMetrics {
        WorldMetrics {
            entities: 42,
            components: vec![
                ComponentMetrics {
                    name: "Transform",
                    count: 42,
                },
                ComponentMetrics {
                    name: "EnemyBrain",
                    count: 7,
                },
            ],
            resources: 3,
        }
    }

    fn row<'a>(rows: &'a [Row], label: &str) -> &'a Row {
        rows.iter()
            .find(|row| row.label == label)
            .expect("the overlay always shows this row")
    }

    #[test]
    fn a_comfortable_frame_is_within_budget() {
        let rows = rows(&diagnostics(&[10, 11, 10]), None, &Budget::default());
        assert!(!over_budget(&rows));
        assert_eq!(row(&rows, "frame avg").status, Status::Ok);
    }

    #[test]
    fn a_slow_average_is_flagged() {
        let rows = rows(&diagnostics(&[30, 32, 31]), None, &Budget::default());
        assert_eq!(row(&rows, "frame avg").status, Status::Over);
        assert!(over_budget(&rows));
    }

    /// The reason `frame max` exists at all.
    #[test]
    fn a_single_spike_is_flagged_even_when_the_average_is_fine() {
        let mut frames = vec![8; 40];
        frames.push(45);
        let rows = rows(&diagnostics(&frames), None, &Budget::default());

        assert_eq!(
            row(&rows, "frame avg").status,
            Status::Ok,
            "the mean absorbs one bad frame"
        );
        assert_eq!(
            row(&rows, "frame max").status,
            Status::Over,
            "but the player still felt the stutter"
        );
        assert!(over_budget(&rows));
    }

    #[test]
    fn submission_counts_are_reported_but_unjudged_by_default() {
        let mut stats = diagnostics(&[10]);
        stats.draw_calls = 5_000;
        stats.triangles = 2_000_000;

        let rows = rows(&stats, None, &Budget::default());
        assert_eq!(row(&rows, "draw calls").value, "5000");
        assert_eq!(
            row(&rows, "draw calls").status,
            Status::Ok,
            "an unconfigured overlay reports without pretending to judge"
        );
    }

    #[test]
    fn a_profile_budget_judges_submission_counts() {
        let mut stats = diagnostics(&[10]);
        stats.draw_calls = 800;
        stats.triangles = 90_000;

        let budget = Budget::frame_time(SIXTY_FPS).with_submission(600, 300_000);
        let rows = rows(&stats, None, &budget);
        assert_eq!(row(&rows, "draw calls").status, Status::Over);
        assert_eq!(
            row(&rows, "triangles").status,
            Status::Ok,
            "each limit is judged on its own"
        );
    }

    #[test]
    fn world_rows_appear_only_with_a_world() {
        let stats = diagnostics(&[10]);
        let without = rows(&stats, None, &Budget::default());
        assert!(!without.iter().any(|row| row.label == "entities"));

        let with = rows(&stats, Some(&metrics()), &Budget::default());
        assert_eq!(row(&with, "entities").value, "42");
        assert_eq!(row(&with, "component types").value, "2");
    }

    #[test]
    fn row_order_is_fixed() {
        let stats = diagnostics(&[10]);
        let labels: Vec<&str> = rows(&stats, Some(&metrics()), &Budget::default())
            .iter()
            .map(|row| row.label)
            .collect();
        assert_eq!(
            labels,
            vec![
                "fps",
                "frame avg",
                "frame max",
                "render cpu",
                "gpu passes",
                "render graph",
                "draw calls",
                "triangles",
                "frames",
                "fixed ticks",
                "entities",
                "resources",
                "component types",
            ],
            "an overlay whose rows move is unreadable at a glance"
        );
    }

    #[test]
    fn over_budget_rows_are_visible_in_plain_text() {
        let rows = rows(&diagnostics(&[40]), None, &Budget::default());
        let text = lines(&rows);
        assert!(
            text.iter().any(|line| line.starts_with('!')),
            "a log or CI failure must show the problem without colour: {text:?}"
        );
        assert!(text.iter().all(|line| line.contains("  ")), "labels align");
    }

    #[test]
    fn an_overlay_with_no_frames_yet_does_not_divide_by_zero() {
        let rows = rows(&FrameDiagnostics::default(), None, &Budget::default());
        assert_eq!(row(&rows, "fps").value, "0.0");
        assert!(!over_budget(&rows));
    }

    #[test]
    fn gpu_measurements_and_graph_are_visible_without_inventing_missing_times() {
        let mut stats = FrameDiagnostics::default();
        let unavailable = rows(&stats, None, &Budget::default());
        assert_eq!(row(&unavailable, "gpu passes").value, "unavailable");
        stats.gpu_passes = Some(vec![
            engine::render::PassTiming {
                name: "lit",
                gpu_ms: Some(1.25),
            },
            engine::render::PassTiming {
                name: "tonemap",
                gpu_ms: None,
            },
        ]);
        stats.render_graph = "0: lit -> hdr\n1: tonemap -> surface".to_owned();
        let measured = rows(&stats, None, &Budget::default());
        assert_eq!(
            row(&measured, "gpu passes").value,
            "lit: 1.250 ms, tonemap: unavailable"
        );
        assert_eq!(row(&measured, "render graph").value, stats.render_graph);
    }
}
