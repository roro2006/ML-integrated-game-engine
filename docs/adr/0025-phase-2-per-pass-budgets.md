# ADR-0025: Per-pass GPU budgets for phase 2 lighting

- Status: Proposed (awaiting the owner, at the M18 exit gate)
- Date: 2026-10-10
- Relates to: ADR-0019 (frame budget), ADR-0024 (lighting path), renderer workstream B7

## Context

Phase 2 (M18) added cascaded shadows, the physically based sky with aerial perspective and height fog, image-based light, clustered point and spot lights, ground-truth ambient occlusion and contact shadows. The workstream asks B7's per-pass timing to set each pass's share of the frame before later phases spend the rest.

The measurement is the day/night showcase (`crates/renderer/src/day_night_showcase.rs`, test-only). It is a small yard with crates, a polished metal sphere and a cut-out plant, with every phase 2 pass on and auto-exposure and bloom from phase 1. It is rendered at 2560x1440 on the RX 7900 XT (High tier) at noon, and at night with 64 point lights. Wall time is taken over 240 frames. Each pass's GPU time is its least over 60 further frames, because other work on the GPU only ever stretches a timestamp; the totals are of those least times.

| Pass | Noon, ms | Night, ms |
|---|---|---|
| shadow (4 cascades, 2048) | 0.032 | 0.031 |
| sky-view | 0.019 | 0.017 |
| light-clusters | — | 0.022 |
| depth-prepass (with normals and cut-outs) | 0.027 | 0.027 |
| ao + ao-denoise | 0.225 | 0.222 |
| contact-shadows | 0.133 | 0.127 |
| lit | 0.207 | 0.362 |
| sky | 0.014 | 0.013 |
| exposure (histogram + average) | 0.050 | 0.047 |
| bloom (11 passes) | 0.113 | 0.115 |
| tonemap | 0.056 | 0.059 |
| **All passes (GPU)** | **0.877** | **1.042** |
| Wall time per frame | 1.018 | 1.200 |

The environment bake does not appear, because it is throttled and re-runs only when the sky changes.

## Decision

Passes split into two kinds.

- **Screen-space passes.** Their cost follows the pixel count, not the scene, so the showcase measures them fairly. Each gets a budget at 1440p on the High tier of at least twice its measured cost, rounded up to a round figure; the cheapest passes get more, because timestamp noise is a larger share of a small number. The showcase test enforces these budgets on the reference GPU:

  | Pass | Budget, ms |
  |---|---|
  | ao + ao-denoise | 0.50 |
  | contact-shadows | 0.30 |
  | sky-view + sky | 0.10 |
  | exposure | 0.10 |
  | bloom | 0.25 |
  | tonemap | 0.15 |
  | light-clusters, with 64 lights | 0.10 |

- **Geometry passes** (shadow, depth-prepass, lit). Their cost follows the scene, and a yard of six objects says little about a town. Their budgets are proposed here and confirmed by the first content-scale scene, the B5 foliage test or E5's tier pass, whichever comes first: shadow 2.0 ms, depth-prepass 1.0 ms, lit 3.0 ms, at 1440p on the High tier. The showcase test does not enforce them.

Together these come to 7.5 ms, under half of the 16.7 ms frame at 60 fps. The rest is left for later phases: GI (C3), TAA and upscaling (E2), volumetrics (M23) and transparency (D4).

## Consequences

- A change that pushes a screen-space pass over its budget fails `the_day_night_showcase_holds_60_fps_at_1440p` on the reference GPU (an adapter named "7900 XT"; an XTX or GRE only prints). Other GPUs only print their timings, and a budget whose passes are missing from the timings fails rather than reading 0 ms.
- The Low and Medium tiers need their own numbers, taken on an integrated GPU when E5 defines the tiers. These are High-tier budgets only.
- The geometry budgets stay proposals until a content-scale scene measures them.

## Alternatives considered

- **One whole-frame budget only.** Simpler, but a single regression could hide inside the large headroom for months.
- **Budgets at measured cost plus 10 %.** Too tight to survive driver updates and timestamp noise. These passes would fail on noise, not on regressions.
