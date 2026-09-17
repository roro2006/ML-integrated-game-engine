# ADR-0003: Renderer GPU API boundary — wgpu

- Status: Accepted
- Date: 2026-09-15

## Context

Open decision §16.1 in `PROJECT_CONTEXT.md`: a portable graphics abstraction
versus direct Vulkan bindings. Targets are native 64-bit Windows and Linux on
Vulkan-capable integrated graphics, with SteamOS/Steam Deck-class hardware as
the low-spec validation target. The renderer must stay small, debuggable, and
portable; high-fidelity rendering is an explicit non-goal.

## Decision

Use `wgpu` (pinned `=30.0.1`) inside `crates/renderer`, behind engine-owned
types: `Color`, `RendererConfig`, `PresentMode`, `GpuPreference`,
`BackendPreference`, `AdapterInfo`, `DeviceKind`, `FrameStats`, `FrameOutcome`,
and `RenderError`. No `wgpu` type appears in `engine` or above.

Supporting choices:

- Backends default to wgpu's platform set (Vulkan on Linux; Vulkan/DX12 on
  Windows); `BackendPreference::Vulkan` forces Vulkan to reproduce SteamOS
  behaviour while testing on Windows.
- Device limits are requested at `downlevel_defaults()`, raised only for
  resolution-derived limits. Exceeding the low-spec budget then fails loudly at
  device creation instead of silently on a user's machine.
- Present mode defaults to Fifo (vsync), which every backend supports;
  `LowLatency` prefers Mailbox, then Immediate, then Fifo.
- The surface format prefers an sRGB format when the adapter offers one.
- wgpu's async adapter/device requests are driven by a ~30-line thread-parking
  `block_on`, so no async runtime enters the dependency graph.
- Surface acquisition states are handled explicitly: `Lost`/`Outdated`
  reconfigure and skip the frame, `Suboptimal` presents and then reconfigures,
  `Timeout` and `Occluded` skip, and a zero-sized surface (minimized window)
  skips. Only a validation failure is reported as an error.

## Consequences

- One backend covers Windows and Linux, and shaders are written once in WGSL.
- wgpu's validation layer catches API misuse early, at some CPU cost per call;
  release builds can disable validation features later if profiling asks for it.
- wgpu releases break API regularly; the pin plus a single containing crate
  keeps upgrades to one file.
- Vulkan-specific extensions and precise driver-level control are not
  available. If profiling on Steam Deck ever demands them, ADR-0001's boundary
  means a Vulkan backend would replace `crates/renderer` only.

## Alternatives considered

- **`ash` (raw Vulkan).** Maximum control and the deepest learning, but
  swapchain, memory, and synchronization management would delay M3
  substantially and require `unsafe` throughout, which the workspace currently
  forbids. Reconsider only if wgpu becomes the bottleneck.
- **Defer the decision.** Rejected: M1's exit criteria require clearing and
  presenting a window.
