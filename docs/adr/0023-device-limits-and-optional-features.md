# ADR-0023: Device limits and optional GPU features

- Status: Accepted (the recommended default at the M17 phase 1 plan gate, 2026-10-09)
- Date: 2026-10-09
- Amends: ADR-0003 (device creation behind the wgpu boundary)

## Context

`Renderer::new` requests wgpu's downlevel default limits, raised only for resolution-derived limits, and no features. That keeps the Low tier honest on integrated graphics, but the High tier (ADR-0019) needs more: block-compressed textures (B3), texture binding arrays for a bindless material path (B3, B4), more storage buffers per stage for clustered lights and GPU culling (B5, C5), and more bind groups for material and environment data (B4, C2). Requesting any of these unconditionally makes device creation fail on hardware that lacks them, which would make the Low tier unbootable.

## Decision

1. **Downlevel is the floor.** Device creation always starts from `Limits::downlevel_defaults()`. A renderer that runs only on the floor is a working renderer.
2. **Optional features are requested only when the adapter reports them.** The set is `TEXTURE_COMPRESSION_BC` and `TEXTURE_BINDING_ARRAY` today. `required_features` is the intersection of that set with `adapter.features()`, so asking can never fail device creation. What was granted is reported to the engine as `AdapterInfo::features` (`GpuFeatures`).
3. **Named limits are raised from the adapter when a work item needs them**, one by one, each with its own fallback, rather than requesting the adapter's full limits. Resolution-derived limits are raised today; others join when B5, C2 or C5 needs them, and this ADR is amended with each.
4. **Every feature use has a path without it.** Uncompressed RGBA8 stands in for BC data, and one bind group per material stands in for binding arrays. Quality tiers choose between paths; they never assume a feature.

## Consequences

- Low-tier hardware boots with exactly the device it did before.
- Code that uses an optional feature must check `GpuFeatures` and keep its fallback tested.
- Per-tier device limits (one device per tier) were considered and rejected: a tier is a settings choice that can change at runtime, while a device is created once.
