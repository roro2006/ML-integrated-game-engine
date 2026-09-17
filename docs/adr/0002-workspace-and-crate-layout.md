# ADR-0002: Workspace and crate layout

- Status: Accepted
- Date: 2026-09-15

## Context

`PROJECT_CONTEXT.md` §8 proposes a crate layout and notes the exact split should
be confirmed by an ADR. Layering must be enforced by something stronger than
convention, and the canonical domain must stay free of runtime and renderer
types so it can be saved, replayed, validated, and later replicated.

## Decision

One Cargo workspace, `crates/<name>`, with dependencies flowing strictly
downward:

```text
sample_game -> game_runtime -> engine -> platform | renderer
                  |   |  |
                  |   |  +-> content_runtime -> game_core
                  |   +----> narrative       -> game_core
                  +--------> procedural_gen  -> game_core
debug_tools -> engine, game_core
ai_gateway  -> game_core
```

- `platform` owns windowing, normalized input, and path services (contains `winit`).
- `renderer` owns the GPU backend (contains `wgpu`).
- `engine` owns the ECS, schedules, time, input state, diagnostics, and the
  renderer facade. It is the only crate that depends on both backends.
- `game_core` and the crates above it are engine-free: no ECS, renderer, or
  platform types. `game_runtime` is the single place where canonical domain
  state meets the runtime ECS.
- Cargo rejects dependency cycles, so the direction above is mechanically
  enforced by the manifests themselves.

Crates that later milestones fill in exist now as documented stubs so the
layering is visible and violations fail at the manifest level rather than after
a refactor.

Workspace-wide settings live in the root manifest: pinned dependency versions,
`unsafe_code = "forbid"`, and warnings for `unwrap`/`expect`/`dbg!`/`print` in
non-test code.

## Consequences

- Swapping a backend or making the domain headless-testable stays cheap.
- Empty crates add a little noise to the tree and to build output.
- The engine depends on `renderer` directly rather than through a trait object.
  A backend trait would be premature with one backend; if a second appears, the
  facade in `engine::render` is where it goes.

## Alternatives considered

- **One crate.** Simplest to start, but nothing would prevent the domain from
  reaching for renderer or ECS types, which is the mistake this project most
  wants to avoid.
- **`renderer` depends on `platform`.** Rejected: it would couple the GPU
  backend to the windowing backend. They meet only through
  `raw-window-handle` traits, which both already use.
