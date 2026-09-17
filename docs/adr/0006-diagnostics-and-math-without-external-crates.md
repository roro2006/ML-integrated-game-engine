# ADR-0006: Engine-owned diagnostics, and math deferred

- Status: Accepted; the math portion is superseded by ADR-0008 (2026-09-15)
- Date: 2026-09-15

## Context

`PROJECT_CONTEXT.md` §5.5 lists `tracing` as an acceptable logging dependency
and a math crate as an option. Both were proposed for approval on 2026-09-15
and declined for now, along with `serde_json`; `serde` and `ron` were approved
for content and saves (M4). The engine still needs structured diagnostics from
M1 onward, and will need math from M3.

## Decision

**Diagnostics** are engine-owned (`engine::diagnostics`): five levels, a global
level filter, a stderr sink, and an optional per-run file sink under the
resolved logs directory. Records carry a static message plus `key = value`
fields (`log_info!("renderer", "adapter selected", name = ..., backend = ...)`)
so logs stay greppable and parseable. Formatting is a pure function and is unit
tested; sink write failures are ignored so logging can never take the engine
down. `FrameDiagnostics` keeps a 120-frame rolling window for the performance
overlay and periodic summaries.

**Math** is deferred: M1 needs none. (Superseded: ADR-0008 adopts `glam` and
fixes the coordinate conventions.) When M3 requires vectors, matrices, and
quaternions, that is its own ADR — either a small internal `engine::math` with
explicit coordinate and handedness conventions, or `glam` with the same
conventions documented and the type boundary defined. Domain state must not
depend on the choice.

## Consequences

- No dependency, and the log format is exactly what the debug overlay and
  future replay tooling need.
- No spans, no per-thread context, no subscriber ecosystem. If those become
  necessary the facade is small enough to re-point at `tracing` without
  touching call sites.
- Deferring math avoids picking conventions before the renderer and camera
  decisions (§16.1, §16.4, product decision 1) are settled.

## Alternatives considered

- **`tracing`.** Richer (spans, structured subscribers, flamegraph tooling) and
  the eventual likely choice if diagnostics grow; not approved at this time.
- **`println!` debugging.** Rejected by project conventions and unusable for a
  performance overlay.
