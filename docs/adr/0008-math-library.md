# ADR-0008: Math library — glam

- Status: Accepted
- Date: 2026-09-15
- Supersedes: the math portion of ADR-0006

## Context

ADR-0006 deferred the math decision to the M3 renderer work. Asked again with
the performance question framed explicitly, the answer was "whichever leads to
the best, most optimal performance."

## Decision

Use `glam` for vector, matrix, and quaternion math, pinned exactly when it is
first added (at the start of M3, not before it is used).

Engine-owned conventions, which do **not** come from glam and must be stated
once here:

- Right-handed coordinate system, **Y up**, **−Z forward**.
- World units are metres.
- Counter-clockwise winding is front-facing.
- Depth range is `0..1` (wgpu/DirectX convention), never OpenGL's `-1..1`.
  In glam 0.33 the older `Mat4::perspective_rh`, `orthographic_rh`, and
  `look_to_rh` helpers are deprecated: projections come from
  `glam::camera::rh::proj::directx` and views from `glam::camera::rh::view`.
- **Trap worth stating explicitly**: glam also ships a `vulkan` projection
  family that is `0..1` but **Y-down**. Targeting Vulkan-capable hardware is
  *not* a reason to reach for it. wgpu's clip space is Y-up, so the `directx`
  family (which glam documents with a `webgpu` alias) is the correct one.
  Choosing `vulkan` produces a scene that renders correctly and is upside
  down — a failure that looks like an art bug, not a math bug.
- `engine::math` owns the `Transform` type (translation, rotation, scale) and
  re-exports the glam types the engine exposes. Game code uses
  `engine::math`, not `glam` directly.
- `game_core` and the domain crates must **not** depend on glam. Canonical
  state stores plain numeric fields; conversion happens in `game_runtime`
  during projection.

## Consequences

- No quaternion, matrix-order, or handedness bugs to re-derive, and glam's
  SIMD paths are the fastest option available.
- Honest caveat: at vertical-slice entity counts the performance difference
  against a competent scalar implementation would almost certainly not be
  measurable. The decision is defensible mainly for correctness and time saved;
  it is recorded that way rather than as a performance win that was measured.
- One more third-party crate in the graph. It stays behind `engine::math`, so
  replacing it later is a single-module change, and it never reaches canonical
  state or save files.
- Writing the math was an opportunity for hands-on learning that this
  choice trades away — a real cost for a project whose purpose is learning.

## Alternatives considered

- **Internal `engine::math`** (~500 lines): maximum ownership, zero
  dependencies, best fit for the from-scratch goal, and educational. Rejected
  here in favour of proven correctness and speed.
