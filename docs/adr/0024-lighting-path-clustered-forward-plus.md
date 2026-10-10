# ADR-0024: Lighting path is clustered forward+ after a depth prepass

- Status: Accepted (owner, at the B2 benchmark gate, 2026-10-09)
- Date: 2026-10-09
- Amends: ADR-0019 decision 3 (the lighting path it left to B2)

## Context

ADR-0019 left the choice between clustered forward+ and deferred shading to the B2 prototype benchmark. The renderer workstream made deferred the default, to stand only if the measurements confirmed it, and asked for numbers from the 7900 XT class High tier and from an integrated GPU.

The benchmark (`crates/renderer/src/b2_bench.rs`, test-only) draws one fixed scene generated in code: 8 overlapping full-screen layers submitted back to front, at 2560x1440. Both prototypes use the production BRDF (`brdf.wgsl`), the same light set and the same GPU light culling into a 16x9x16 cluster grid. Each prototype runs after the same position-only depth prepass. The deferred G-buffer has three targets: RGBA16F emission, RGB10A2 normal and metallic, RGBA8 albedo and roughness. Every pass is timed with GPU timestamps (B7), with 30 warm-up frames and 200 measured frames per case. A GPU test (`every_path_draws_the_same_lit_image`) holds every path to the same image. The clustered result matches the brute-force light loop, and the test was mutation-checked by shrinking the culling radius. A forward pass without the prepass is measured as the control.

## Measurements

Median GPU milliseconds for the whole path (first pass begin to last pass end). The two hardware rows are the adapters the runs reported, both on Vulkan with AMD driver 26.8.1.

Large triangles (2,048 triangles):

| Lights | Hardware | Forward+, no prepass | Clustered forward+ | Deferred |
| --- | --- | --- | --- | --- |
| 64 | RX 7900 XT (discrete) | 0.294 | **0.089** | 0.155 |
| 256 | RX 7900 XT (discrete) | 0.700 | **0.132** | 0.214 |
| 1024 | RX 7900 XT (discrete) | 2.261 | **0.280** | 0.408 |
| 64 | Radeon Graphics (integrated) | 16.1 | **3.51** | 5.60 |
| 256 | Radeon Graphics (integrated) | 41.3 | **5.24** | 8.29 |
| 1024 | Radeon Graphics (integrated) | 141.5 | **12.2** | 18.1 |

Dense geometry (1,638,400 triangles, about 22 pixels each, 256 lights), the case where 2x2-quad waste penalises forward shading:

| Hardware | Forward+, no prepass | Clustered forward+ | Deferred |
| --- | --- | --- | --- |
| RX 7900 XT (discrete) | 1.09 | **1.14** | 1.21 |
| Radeon Graphics (integrated) | 81.6 | **26–33** | 40–45 |

Per-pass medians at 256 lights with large triangles, RX 7900 XT: prepass 0.030, cull 0.022, forward 0.076; deferred G-buffer fill 0.059, lighting 0.096. On the integrated GPU the G-buffer fill alone (1.43 ms) costs as much as the depth prepass (1.46 ms), and the deferred lighting pass costs more than the forward pass because it reads three targets and depth for every pixel.

The integrated dense runs vary between runs (the depth prepass alternates between about 12.5 and 20 ms, likely clock or memory state). The forward+ lead held in every run.

## Decision

1. **Clustered forward+ is the lighting path**, not the planned deferred default. It was faster on both hardware classes, at every light count, and with both large and small triangles. Its margin is largest on the integrated GPU, where bandwidth is scarce and where the Low tier has to run.
2. **The depth prepass stays and runs first in every frame.** Without it, overdraw multiplies the shading cost 3 to 12 times on these scenes. The forward pass then tests depth with `LessEqual` and does not write it.
3. **Light culling is a compute pass over a cluster grid**, as prototyped. Each cluster list reserves `light_count + 1` words, so no light can overflow. The grid, the depth slicing (exponential, per C5) and a per-cluster cap are set when C5 brings point and spot lights; the CPU reference in `clustered.rs` keeps the culling test conservative.
4. **Screen-space consumers get a thin G-buffer from the prepass, not a full one.** C4 (GTAO) and E2 (TAA) need depth, normals and velocity. The prepass gains normal and velocity targets when those items start, rather than paying for albedo and material targets every frame now.
5. **Deferred never reaches production code.** Its prototype shaders (`gbuffer.wgsl`, `deferred_lighting.wgsl`) are compiled only by the test-only benchmark, which stays so this decision can be measured again with the same scene. No renderer path or setting selects them.
6. **Integration, as built.** A `depth-prepass` graph pass runs before `lit` and writes the depth of every draw except cut-out (alpha-mask) materials. The lit and PBR pipelines test `LessEqual` and keep depth writes on, so cut-outs still occlude what is behind them. All three vertex stages compute the clip position with the same expression and mark it `@invariant`. The golden reference scene is unchanged, so the single-pass path needs no switch to remain selectable.

## Consequences

- Material shaders stay single-pass forward (`pbr_lit.wgsl`), as B4 already wrote them. B4's material permutations need no G-buffer encoding, and transparent and special materials use the same lighting loop.
- Shading cost per pixel scales with the lights in its cluster. C5's per-cluster cap and the overflow count in `FrameStats` become the guard against light-dense scenes.
- The benchmark scene uses cheap materials (no texture fetches) and an orthographic frustum. A material-heavy scene shifts cost toward deferred's single lighting pass. If profiling with production content (D-series assets) reverses the result, this ADR is revisited with the same harness.
- The prepass doubles vertex work. In the dense case on the 7900 XT, the no-prepass forward path was 0.06 ms faster, so engine-side culling (B5) and Hi-Z from the prepass depth matter for geometry-heavy frames.

## Alternatives considered

- **Deferred, as planned.** It was slower in every measured case, by 30 to 60 percent on the integrated GPU, and it costs three render targets of memory at full resolution.
- **Forward without a prepass.** It was the fastest only on dense geometry on the discrete GPU, by 5 percent, and it was 3 to 12 times slower everywhere else.

## Amendment 1: the normal target for GTAO (C4), 2026-10-10

Decision 4, as built for ambient occlusion:

- **The prepass writes normals only while ambient occlusion is on.** The target is `Rgb10a2Unorm` at full resolution, holding the world-space geometric normal as `n * 0.5 + 0.5`. World space rather than view space, because the renderer receives a view-projection and no separate view matrix, and GTAO reconstructs positions from depth through the inverse view-projection anyway. Its slice plane is built from the view vector and the screen direction lifted to the pixel's depth, so it needs no view space. With occlusion off the prepass is the depth-only pipeline of decision 6, unchanged in cost and output.
- **Cut-outs are drawn into the prepass while ambient occlusion is on**, by a masked pipeline that runs their alpha test and binds their material group. Without that, occlusion would not see foliage: depth would show the ground behind a leaf, and the ground's occlusion would land on the leaf. The forward pass still runs the same alpha test and writes depth as before, so the two passes agree on every hole.
- **Measured** at 1920x1080 on the RX 7900 XT, in a corner scene with 64 boxes and 64 cut-outs: the prepass takes 0.014 ms depth-only and 0.030 ms with normals and cut-outs; the GTAO trace takes 0.16 ms and its denoise 0.02 ms, against a 1 ms budget. The masked pipeline's cost is part of the 0.016 ms difference, so it stays.
- A velocity target joins this one when E2 (TAA) starts.
