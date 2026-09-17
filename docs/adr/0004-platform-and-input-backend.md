# ADR-0004: Platform and input backend — winit

- Status: Accepted
- Date: 2026-09-15

## Context

Open decision §16.9: window/input crate and controller support timeline. The
engine needs windows, keyboard, and mouse on native Windows and Linux (X11 and
Wayland) now, and gamepads before Steam Deck UX validation. The candidates were
`winit` and SDL3 bindings.

## Decision

Use `winit` (pinned `=0.30.13`) inside `crates/platform`. Gamepad support will
be added later with a focused crate (`gilrs` is the expected choice) behind the
same normalized input types, before Steam Deck validation.

The platform layer exposes engine-owned vocabulary only: `KeyCode` (physical
key positions, so WASD works on AZERTY), `MouseButton`, `ButtonState`, and
`PlatformEvent`. Unmapped keys are dropped at the boundary. `PlatformWindow`
exposes size, scale factor, redraw requests, and title, plus the
`raw-window-handle` traits the renderer needs.

Loop shape: winit owns the event loop and calls a `PlatformHandler`
(`window_created`, `event`, `frame`). Control flow is `Wait`, and each frame
requests the next redraw, so presentation paces the loop; while minimized no
redraw is requested and the loop idles on events instead of spinning.

## Consequences

- `cargo build` is all that is needed on both platforms: no C toolchain, no
  bundled DLL, no system library.
- The engine's frame loop lives inside winit callbacks rather than owning
  `loop { }` directly. `App::update` is independent of the window, so headless
  runs and tests are unaffected.
- Audio and gamepads need their own crates later, rather than arriving with the
  windowing library.
- winit's API changes significantly between releases; the pin plus one
  containing module limits the blast radius. winit 0.31 is in beta and reworks
  the window API; upgrade deliberately, not automatically.

## Alternatives considered

- **SDL3 bindings.** Best-in-class gamepad and Steam Deck support, an
  engine-owned poll loop, and audio in the same library. Rejected for now: it
  is a broad framework rather than narrow plumbing (exactly the creep ADR-0001
  guards against), and it adds a C dependency plus packaging work on Windows.
  Reconsider if controller handling or Steam Input integration becomes painful.
- **Direct Win32/Wayland/X11.** Full ownership, but weeks of platform plumbing
  for no project-specific insight.
