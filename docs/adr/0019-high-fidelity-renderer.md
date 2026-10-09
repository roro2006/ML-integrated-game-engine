# ADR-0019: High-fidelity renderer: render graph, PBR, shadows, global illumination, and quality tiers

- Status: Accepted
- Date: 2026-10-09
- Supersedes: PROJECT_CONTEXT §2 non-goal "High-fidelity rendering"; §5.4 renderer non-goals; the §5.2 deferral of "advanced PBR pipeline, ray tracing, complex post-processing" and the §4 "simple forward rendering" target.

## Context

The renderer draws one pass with one pipeline (`lit.wgsl`), one directional light, a flat ambient term, and untextured flat-colour materials. ADR-0017 added per-vertex colour and a flash overlay on that base. PROJECT_CONTEXT §5.4 excludes full PBR, deferred rendering, render graphs, volumetrics, screen-space effects and expensive global illumination. The owner approved the master plan on 2026-10-09, which sets a modern PBR target on a Radeon RX 7900 XT class High tier, with a Low tier that still runs on integrated graphics.

The reference game was studied only at the level of configuration files and asset statistics (`docs/reference/observed-settings.md`, `docs/reference/observed-asset-budgets.md`). Those records fix the envelope this ADR works within. The reference runs classic LOD chains with no Nanite, cascaded shadow maps with virtual shadow maps configured but disabled, software Lumen-style GI with a global distance field and no hardware ray tracing, TSR as its upscaler, and volumetric fog on every tier.

## Decision

1. **One GPU API.** wgpu stays the only GPU API, and it stays behind the renderer's engine-owned facade (ADR-0003). `unsafe_code = "forbid"` stays workspace-wide.

2. **Render graph.** Passes declare the resources they read and write. The graph orders them, rejects cycles with an error rather than a panic, and pools transient targets. It emits no manual barriers; wgpu tracks synchronisation within a queue.

   > **Amended 2026-10-09 (B1, as built).** The graph does not reorder passes: declaration order is execution order, and the graph validates it instead. A pass that reads a transient no earlier pass wrote is an error naming that pass, which is the mistake a sorting graph would report as a cycle. With one wgpu queue and no async compute, reordering would buy nothing and make a frame harder to read. Transients are pooled by exact descriptor and retired after four unused frames; aliasing by lifetime is recorded but not done.

3. **HDR and PBR.** The lit pass writes an HDR target, and tonemapping runs in a final pass. Materials are PBR metallic-roughness with base colour, normal, packed ORM and emissive textures. A small set of master shaders covers the families in `observed-asset-budgets.md` §7, and each material is an instance: a master plus a parameter block, with feature flags for permutations. ADR-0017's flash overlay is kept as a material parameter, so hit cues still work on textured bodies. The lighting path (clustered forward+ or deferred) is chosen by the B2 prototype benchmark and recorded in a later ADR. This ADR does not choose it.

4. **Shadows.** Cascaded shadow maps with caching, 2 cascades on Low, 4 on High and 5 on Epic. Virtual shadow maps are not in scope. The reference configures them and then disables them, so they are not a proven need.

5. **Global illumination.** Software GI with no hardware ray tracing. The target is a global distance field built in compute for the far field, plus a screen-space probe gather for the near field. A DDGI-style probe grid is the fallback if the prototype gate fails. Per-mesh distance-field tracing is excluded. GI is off on the Low tier. This is a deliberate deviation from the reference, which runs GI on every tier; the owner confirmed it on 2026-10-09.

6. **Sky and fog.** Physically based sky, aerial perspective and height fog. Volumetric fog is on every tier, including Low. Volumetric clouds are the first thing cut if the frame budget fails.

7. **Anti-aliasing and upscaling.** TAA first, then an in-house temporal upscaler. Vendor upscalers (FSR, DLSS) are allowed only through a future isolated `renderer_ffi` crate, which needs its own ADR. No other crate may contain `unsafe`.

8. **Textures.** Base colour is BC7 sRGB, at most 2048 on High and 1024 on Low. Normals are BC5 linear at half the base width, with a floor of 512. ORM is BC1 linear, packed. Emissive is BC1 sRGB at 1024. Virtual textures are deferred and remain a design note only.

9. **Geometry.** Classic LOD chains with no Nanite. Four levels by default, five for architecture, large landscape and rocks, one for small natural pieces. The reduction ratio is 0.5, or 0.7 for base-building. Switch sizes follow `observed-asset-budgets.md` §3. Meshlets and GPU-driven culling are optional under B5.

10. **Quality tiers.** Four tiers: Low, Medium, High and Epic. The name "Epic" follows the master plan and the reference, which supersedes "Ultra" in renderer workstream E5. High and Epic target an RX 7900 XT class GPU at 1440p native or 4K upscaled, 60 fps. Medium targets a mid-range discrete GPU at 1080p, 60 fps. Low targets integrated graphics and Steam Deck class hardware at 720p to 900p upscaled, 30 to 40 fps, with no GI. Tiers are data. A tier cannot enable a feature the adapter lacks; the profile is reduced and the reduction is reported.

## Consequences

- The renderer's single-pass, one-light, flat-ambient contract is gone. Every pass is declared to the graph, and the golden-image harness (G3) becomes the regression gate for each change.
- The vertex layout grows. At least tangents are needed for normal maps. Whether the changes are batched into one layout change is an open question in renderer workstream §6, and this ADR does not settle it.
- Device limits are an open question as well. The downlevel floor with named limits raised from the adapter is recommended in renderer workstream §2, and it is for the renderer workstream to settle.
- Low-tier scope is real. GI and volumetric clouds are absent there, and fog and shadows stay. Low is tested on integrated hardware through G4.
- Asset budgets become enforced rules in the cooker (B6), with warn and error thresholds from `observed-asset-budgets.md` §2c.
- Shader sources stay snake_case and are validated with naga in CI (G4).

## Alternatives considered

- **Keep the ADR-0017 forward renderer and add only textures.** Lower risk, but it cannot reach the look the owner approved, and the later PBR work would still need a graph.
- **Hardware ray tracing for GI.** The reference does not use it, and wgpu has no stable ray tracing. Rejected.
- **Nanite-style virtualised geometry.** The reference ships without it, and no source mesh in the budget set has Nanite data. Classic LODs cover the target.
- **A vendor upscaler as the default.** Faster to ship, but it puts vendor FFI and `unsafe` into the core renderer. It stays optional behind its own ADR.

## Relation to other ADRs

- Amends **ADR-0017** §4 and the "out of scope for M7" list: bloom and textures are now decided here, and the vertex layout grows beyond the 11 floats it has today.
- Amends **ADR-0011**: the baked mesh format must carry tangents and vertex colour, and the cooker enforces the budgets above.
- Preserves **ADR-0003** (wgpu boundary) and **ADR-0010** (no new network path is introduced by rendering).
- The renderer's `10-renderer-workstream.md` still lists VSM as a second step under C1. This ADR supersedes that item, and the workstream needs updating to match.
