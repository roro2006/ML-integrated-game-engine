# ADR-0001: Engine-from-scratch scope and dependency posture

- Status: Accepted
- Date: 2026-09-15

## Context

`PROJECT_CONTEXT.md` (§3.1, §5.5) records the decision to build and own the
engine rather than adopt Bevy or another framework. Implementation needs a
concrete, checkable rule for what "from scratch" means when a dependency is
proposed, otherwise the boundary erodes one convenient crate at a time.

## Decision

The project owns every engine-level behaviour: application/frame lifecycle,
platform abstraction, ECS and simulation architecture, scheduling and timing,
renderer facade and render lifecycle, asset/content registries, save/replay and
content-revision boundaries, and the AI gateway's queue, validation, fallback,
and activation behaviour.

The project does not reimplement OS windowing, GPU drivers, codecs, audio
device access, serialization, or HTTP. A third-party crate for one of those is
acceptable when all of the following hold:

1. It solves a plumbing problem, not an engine-architecture problem.
2. Exactly one crate in the workspace depends on it.
3. Its types do not appear in APIs above that crate, except neutral interop
   traits that cannot be avoided (currently `raw-window-handle`, which the
   renderer needs to create a surface for a platform window).
4. Its version is pinned exactly and the rationale is recorded in an ADR.

Bevy and Bevy ecosystem crates remain reference material only and must not
appear in the dependency graph. CI enforces this mechanically.

Approved dependencies so far: `winit` (ADR-0004), `wgpu` (ADR-0003), and
`serde` + `ron` for content and saves when M4 begins. `tracing`, `glam`, and
`serde_json` were considered and declined for now (ADR-0006).

## Consequences

- Milestones stay ordered platform → ECS → renderer → domain/content → slice;
  engine scope is driven by what the vertical slice needs.
- Replacing a backend is a single-crate change, which keeps the SteamOS and
  low-spec targets negotiable.
- More code to write and test than adopting a framework, and no free ecosystem
  of plugins. Accepted deliberately: understanding and owning these systems is
  a project goal.

## Alternatives considered

- **Use Bevy.** Fastest path to a prototype; rejected by the project's purpose,
  since it would hide the ECS, scheduling, asset, and render architecture the
  project exists to learn and own.
- **Zero dependencies.** Writing Vulkan loaders, Wayland/Win32 windowing, and
  codecs would consume the entire budget without teaching what the project is
  about.
