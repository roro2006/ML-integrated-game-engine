# Architecture Decision Records

One file per decision that is expensive to reverse. Status is `Proposed`,
`Accepted`, `Superseded by ADR-NNNN`, or `Rejected`. Never rewrite history in an
accepted ADR: supersede it with a new one.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-engine-from-scratch-scope.md) | Engine-from-scratch scope and dependency posture | Accepted |
| [0002](0002-workspace-and-crate-layout.md) | Workspace and crate layout | Accepted |
| [0003](0003-renderer-gpu-api-boundary.md) | Renderer GPU API boundary: wgpu | Accepted |
| [0004](0004-platform-and-input-backend.md) | Platform and input backend: winit | Accepted |
| [0005](0005-ecs-storage-and-scheduling.md) | ECS storage and scheduling | Accepted |
| [0006](0006-diagnostics-and-math-without-external-crates.md) | Engine-owned diagnostics and math | Accepted (math superseded by ADR-0008) |
| [0007](0007-vertical-slice-game-design.md) | Vertical-slice game design decisions | Accepted |
| [0008](0008-math-library.md) | Math library: glam | Accepted |
| [0009](0009-collision-and-navigation.md) | Collision and navigation | Accepted |
| [0010](0010-ai-content-policy.md) | AI content policy: visibility, permanence, boundaries | Accepted |
| [0011](0011-asset-pipeline-and-shaders.md) | Asset pipeline and shader workflow | Accepted |
| [0012](0012-audio-backend.md) | Audio backend: cpal with an engine-owned mixer | Accepted |
| [0013](0013-save-envelope-and-event-log.md) | Save envelope, migration policy, event log format | Accepted |
| [0014](0014-content-hashing.md) | Content hashing: BLAKE3 | Accepted |
| [0015](0015-class-derived-pools.md) | Class-derived pools carried in `ClassChosen` | Accepted |
| [0016](0016-mechanics-first-milestone-order.md) | Mechanics-first milestone order | Accepted |
| [0017](0017-presentation-runtime.md) | Presentation runtime: recipe models, rigid-part animation, VFX, skillshots | Accepted |
| [0018](0018-builds-talents-and-ability-loadout.md) | Builds: one class, five-slot loadout, talents, item empowerment | Accepted |
| [0019](0019-high-fidelity-renderer.md) | High-fidelity renderer: render graph, PBR, shadows, global illumination, quality tiers | Accepted (supersedes PROJECT_CONTEXT §2 and §5.4 rendering non-goals) |
| [0020](0020-free-camera-and-3d-interaction.md) | Free third-person camera and 3D interaction | Accepted (supersedes ADR-0007 camera and mouse aim; amends ADR-0009) |
| [0021](0021-open-world-building-and-generated-sites.md) | Open-world cells, snap-piece building, generated sites in saves | Accepted (supersedes PROJECT_CONTEXT §2 world and building non-goals; amends ADR-0007 and ADR-0010) |
| [0022](0022-skill-based-progression.md) | Skill-based progression | Accepted (amends ADR-0015 and ADR-0018) |
| [0023](0023-device-limits-and-optional-features.md) | Device limits and optional GPU features | Accepted (amends ADR-0003) |

## Approved dependencies

| Crate | Purpose | Contained in | ADR |
|---|---|---|---|
| `winit` | Window, event loop, keyboard/mouse | `crates/platform` | 0004 |
| `wgpu` | GPU surface, device, rendering | `crates/renderer` | 0003 |
| `serde` + `ron` | Content files and save envelopes (M4) | `game_core`, `content_runtime` | 0001 |
| `glam` | Vector/matrix/quaternion math (M3) | `engine::math` | 0008 |
| `cpal` | Audio device access (M5) | audio module | 0012 |
| `blake3` | Content revision hashing | `content_runtime` | 0014 |

Declined for now: `tracing`, `serde_json`. An audio decoder crate and a gamepad
crate (`gilrs`) need their own approval when that work starts.
