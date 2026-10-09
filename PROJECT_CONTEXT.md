# AI-Native RPG Engine — Project Context and Source of Truth

> **Purpose:** This document is the authoritative project brief for humans and coding assistants. Keep it at the repository root as `PROJECT_CONTEXT.md`. Update it deliberately when product or architectural decisions change.
>
> **Instruction to AI agents:** Read this file before proposing architecture, changing project structure, adding dependencies, or implementing gameplay or engine systems. Treat explicit decisions here as constraints. If a requested task conflicts with this document, identify the conflict and ask before making a broad architectural change.
>
> **Engine-from-scratch decision:** The engine-from-scratch decision (originally `ENGINE_FROM_SCRATCH_ADDENDUM.md`, accepted 2026-09-15) is incorporated into this document. The project builds and owns its own Rust engine. Bevy is a reference implementation only and must not be a runtime, rendering, ECS, asset, UI, scheduling, scene, serialization, or game-framework dependency.

## 1. Project summary

Build a from-scratch Rust game engine and data-driven runtime for persistent worlds where trusted game systems, human authors, and AI-assisted tooling can create and modify content safely. The engine must support AI-assisted generation of quests, narrative branches, maps, encounters, enemies, missions, and visual variants while preserving mechanical validity, performance, determinism where needed, save compatibility, and player agency.

The first validation game is a free-form, third-person survival and crafting sandbox vertical slice, single-player first, inspired by the high-level gameplay loop of games such as Dimraeth: exploration, combat, quests, gathering, crafting, settlement progression, factions, and persistent world consequences. Building, gathering, skill-based progression, crafting, and combat form one system, with combat one part among several. The AI-content pipeline (generated dungeons, sites, and quests) runs on top of the open world. The visual target is high-fidelity PBR rendering, with a Low tier for integrated graphics.

This is not initially a general replacement for mature engines. It is a focused, extensible engine architecture that proves safe AI-native content creation in a playable single-player RPG. Engine scope is driven by what the vertical slice needs, not by feature parity with Bevy, Unity, Godot, or Unreal.

## 2. Product goals

### Primary goals

1. Build the game engine from scratch in Rust, owning engine-level architecture and behavior.
2. Create a playable single-player RPG vertical slice that validates the engine.
3. Support native 64-bit Windows and native 64-bit Linux as first-class desktop targets; treat SteamOS/Steam Deck-class hardware as a low-spec Linux performance and UX validation target.
4. Make core gameplay data-driven: quests, enemies, abilities, encounters, items, dialogue, maps, biomes, and visuals must be definable through typed content files.
5. Support AI as a constrained content planner and authoring assistant, never as an unrestricted runtime code executor.
6. Make AI-generated content valid through compilation, validation, simulation/dry-runs where practical, approval/activation rules, versioning, and rollback.
7. Run well on the target hardware: the High and Epic tiers target a Radeon RX 7900 XT class GPU at 1440p native or 4K upscaled, 60 fps; the Low tier keeps integrated-GPU and Steam Deck class support at 720p to 900p upscaled, 30 to 40 fps.
8. Remain fully playable without internet access or any local model installed.
9. Prefer free, local, privacy-preserving tools and assets where reasonable.
10. Preserve a clean future path to server-authoritative multiplayer after single-player is proven.
11. Support text-first authoring, Git-friendly review, automated validation, and powerful in-game/debug tooling.

### Explicit non-goals for the initial vertical slice

- Multiplayer or online networking implementation.
- Open-world streaming beyond the 1 km² cell-streamed world (ADR-0021).
- Fully arbitrary AI-authored native Rust code.
- Runtime execution of model-generated scripts with unrestricted capabilities.
- Mandatory cloud services or mandatory AI inference.
- Runtime image, 3D mesh, music, voice, or video generation.
- General-purpose editor, full scene editor, or generic reflection UI before text-first content workflows are stable.
- General-purpose visual scripting or an arbitrary scripting runtime.
- Freeform voxel building or terrain deformation (snap-piece building in claimed areas is in scope, per ADR-0021).
- Broad mod publishing ecosystem before the content format stabilizes.
- Hardware ray tracing, virtual shadow maps, or Nanite-style virtualised geometry (ADR-0019).
- Using Bevy or another game engine/framework as the runtime substrate.

## 3. Design pillars

### 3.1 The project owns its engine

"From scratch" means the project owns engine-level architecture and behavior, including:

- Application/frame lifecycle and platform abstraction.
- Window/input/event integration.
- ECS or alternative data-oriented simulation architecture.
- Entity lifecycle, component storage, queries, commands, resources, and event channels.
- System scheduling, execution model, time, fixed timestep, and deterministic simulation controls.
- Asset/content loading and registries.
- Renderer facade, render lifecycle, game-facing mesh/material/camera APIs, culling, debug rendering, and performance instrumentation.
- Scene/world projection and runtime entity management.
- UI/debug tooling integration.
- Audio integration boundary.
- Physics/collision/navigation integration boundaries.
- Save/load, domain events, event log, replay, content revisions, and rollback boundaries.
- AI provider gateway, generation queue, validation, fallback, activation, and provenance behavior.

"From scratch" does **not** mean reimplementing operating-system APIs, GPU drivers, OS windowing, image codecs, audio device drivers, serialization libraries, or HTTP. Focused third-party libraries for those plumbing layers are allowed when justified and kept behind engine-owned interfaces (see Section 5.4).

**Rationale.** The project exists to learn and prove engine architecture, particularly:

- Data-driven runtime content systems.
- AI-safe content composition and validation.
- Canonical world state versus runtime ECS projection.
- Deterministic event-sourced persistence and later multiplayer readiness.
- Low-spec desktop and SteamOS-class performance discipline.
- Explicit control over ECS storage, scheduling, asset loading, serialization, and content activation.

Using Bevy as the core runtime would accelerate a game prototype but would hide or constrain several subsystems this project intends to understand and own. Building from scratch makes the scope larger, so implementation must be staged and driven by a small vertical slice rather than broad engine ambition.

### 3.2 Trusted mechanics, generative composition

The Rust engine kernel owns all foundational rules and safety-critical behavior:

- Simulation rules.
- Damage, cooldowns, costs, resources, and inventory semantics.
- Physics/collision and navigation.
- Quest-state transitions.
- Serialization, save/load, event replay, validation, and rollback.
- Asset loading and resource budgets.
- AI permissions, rate limits, and tool boundaries.

AI and data content can compose **allowlisted, tested primitives**. They must not directly write arbitrary Rust, access the filesystem, run shell commands, mutate arbitrary ECS state, or bypass validation.

### 3.3 Canonical state is independent from the runtime ECS

The engine's runtime ECS world is a live projection, not the sole source of truth.

- Domain state is serializable, versioned, stable-ID-based, and suitable for save/load/replay.
- Runtime ECS entities hold transient transforms, rendering, animation, navigation, physics, and other runtime state.
- Domain events are the authority for durable changes.
- Systems project accepted domain changes into the runtime ECS at controlled points.
- Raw runtime entity handles/IDs must never be persisted in canonical state, save files, or AI-generated content.

### 3.4 AI is optional, asynchronous, and bounded

AI never blocks input, rendering, fixed-timestep simulation, combat, saving, or load completion.

- Local inference is preferred, but optional.
- Cloud inference is optional and must be opt-in.
- Deterministic template-based generation is always available as fallback.
- AI generation happens in a separate process/service through a narrow, engine-owned HTTP provider abstraction.
- AI outputs are schema-constrained proposals, not executable instructions.
- Generation is queued, cached, cancellable, and activated only at safe points.

### 3.5 Text-first, data-driven development

Human-authored runtime content should be typed, serializable, reviewable, and diff-friendly.

- Preferred authored format: RON.
- Preferred model interchange format: JSON constrained by explicit schemas.
- Rust types use `serde` serialization and schema generation/validation tooling such as `schemars` where appropriate.
- Content is compiled and validated into an internal runtime registry.
- Content files are version-controlled and separated from save-specific generated content.

### 3.6 Low quality tier and handheld delivery

The Low tier keeps the game running on integrated graphics and Steam Deck class hardware on native Windows, Linux, and SteamOS. It is a scalability tier, not the design target (ADR-0019).

- Low targets 720p to 900p upscaled at 30 to 40 fps, with no global illumination, 2 shadow cascades, and volumetric fog kept.
- Design the game to run well with AI disabled.
- Test the Low tier on integrated hardware through the CI plan (G4).
- Apply asset budgets (docs/reference/observed-asset-budgets.md, enforced by the asset cooker per tier) and draw-call, texture-memory, and particle budgets on Low. Keep materials shared.
- Profile early on native Windows, native Linux, and handheld-like performance profiles.
- Treat memory and bandwidth as shared/limited resources when local inference is active.

## 4. Technology decisions

| Category | Decision |
|---|---|
| Language | Rust |
| Runtime/game engine | Custom Rust engine built from scratch; Bevy is a reference implementation only and not a runtime dependency |
| Third-party libraries | Narrow plumbing only (windowing, GPU bindings, codecs, audio device, serialization, HTTP, logging, hashing), hidden behind engine-owned interfaces |
| Primary rendering target | A high-fidelity PBR 3D renderer on wgpu (ADR-0003, ADR-0019), with a render graph, HDR, PBR metallic-roughness materials, cascaded shadow maps, software global illumination with no hardware ray tracing, and Low, Medium, High and Epic quality tiers |
| Platforms | Native 64-bit Windows and native 64-bit Linux as first-class targets; SteamOS/Steam Deck-class hardware as low-spec Linux validation target; WSL is a development convenience only |
| Initial mode | Single-player only |
| Future multiplayer posture | Design domain state, events, content revisions, and mutations to become server-authoritative later; defer actual networking implementation |
| Content authoring | RON files, supported by JSON import/interchange for AI outputs |
| Canonical state | Engine-owned serializable domain state plus event log and snapshots |
| AI connection | Out-of-process local HTTP service behind an engine-owned provider abstraction; OpenAI-compatible API boundary |
| Reference local backend | `llama.cpp` server with quantized GGUF models; Vulkan/CPU depending on hardware |
| Convenience local backend | Ollama adapter |
| Cloud model use | Optional, opt-in provider adapter only |
| No-AI fallback | Deterministic templates, seeded generators, constraint solvers, and authored content |
| Asset strategy | Original PBR assets authored to the budgets in `docs/reference/observed-asset-budgets.md` (LOD chains, BC7/BC5/ORM texture sets), seeded procedural props, carefully tracked CC0 assets; nothing taken from the reference game (section 11, ADR-0019) |
| Build/dependency policy | Favor small, focused, maintained Rust crates; no dependency may become the de facto engine core; pin versions and preserve `Cargo.lock` |

### Relationship to Bevy

Bevy is a reference, not a dependency.

Permitted use:

- Study Bevy source code, documentation, examples, release notes, architecture, tests, and benchmarks.
- Compare ECS API ergonomics, scheduling approaches, asset hot-reload behavior, renderer architecture, scene/reflection systems, and plugin organization.
- Recreate high-level concepts only after independently understanding tradeoffs.
- Cite or link to Bevy documentation/source in project notes where useful.
- Create isolated experiments outside the engine runtime to learn APIs or compare behavior.

Not permitted without an explicit project decision:

- Adding `bevy`, `bevy_*`, or any Bevy ecosystem crate as a Cargo dependency.
- Wrapping Bevy behind a project facade while relying on it internally, or using it as a hidden ECS, renderer, asset, UI, scheduler, scene, or runtime layer.
- Copying source code without first checking license obligations and recording provenance.
- Treating Bevy entity IDs, schedules, scenes, reflection types, or asset handles as part of engine state or public APIs.

## 5. Engine architecture

### 5.1 Layers

Use explicit layers. Lower layers must not depend on higher layers.

```text
+----------------------------------------------------------+
| sample_game                                              |
| Gameplay rules, content packs, Cinderwake vertical slice |
+----------------------------------------------------------+
| game_runtime                                             |
| Domain-to-runtime projection, gameplay systems, UI glue  |
+----------------------------------------------------------+
| engine                                                   |
| ECS, schedule, time, events, assets, renderer facade,    |
| input, windowing facade, debug/profiling interfaces      |
+----------------------------------------------------------+
| platform / renderer / audio / input backends             |
| and focused third-party libraries                        |
| (e.g. winit, wgpu/Vulkan bindings, image/audio decoders) |
+----------------------------------------------------------+
```

The game-domain core (`game_core` and content/narrative crates) remains independent from live engine runtime and renderer types:

```text
Canonical domain state
  - Stable typed IDs
  - World facts, quests, factions, NPC facts, inventory, progression,
    world flags, region state, generated-content revisions
  - Serializable snapshots and replayable domain events
  - AI context source, validation source, content-compilation input,
    and later multiplayer replication source

Runtime ECS world
  - Transforms, rendering state, animation state, physics contacts
  - Transient pathfinding/navigation caches, particles
  - Runtime entity handles
  - Projection of canonical state, accepted domain events,
    and current simulation state
```

Raw runtime entity handles are never persisted in canonical saves or referenced by AI-generated content.

### 5.2 Scope control: build only what the vertical slice needs

Initial engine capabilities:

- Native desktop application loop on Windows and Linux.
- Keyboard/mouse input; controller support may follow after the core loop is stable (required before Steam Deck UX validation).
- A high-fidelity PBR 3D renderer with Low, Medium, High and Epic quality tiers (ADR-0019).
- Free third-person camera (ADR-0020), transforms, meshes, PBR materials, directional light with sky-based ambient lighting, and simple particle/VFX support.
- ECS world with entities, sparse component storage, queries, commands/deferred mutation, resources, events, and staged systems.
- Fixed-timestep gameplay simulation plus variable-rate rendering.
- Basic collision and simple navigation/pathfinding integration.
- RON content loading and strongly typed registries.
- Canonical domain state, events, saves, replay, and generated-content revisions.
- Debug overlay, logging, profiling counters, and headless validation/testing where possible.
- Explicit engine path services for config, cache, logs, saves, and assets.
- Skeletal animation: glTF skeletons and clips, GPU skinning, and a data-driven animation state machine (ADR-0020).

Explicitly defer:

- General-purpose editor.
- Full scene editor and generic reflection UI.
- General-purpose visual scripting.
- Arbitrary scripting runtime.
- Generic plugin/reflection systems before the vertical slice requires them.
- Ray tracing, virtual shadow maps, and Nanite-style virtualised geometry (ADR-0019).
- Platforms beyond native Windows and Linux (e.g., macOS, consoles, web, mobile).
- Networking implementation.
- Runtime generative media pipelines.

### 5.3 ECS requirements

The initial ECS should be deliberately small and comprehensible. It must serve the vertical slice, not aspire to universal ECS feature parity.

Initial requirements:

- Generational entity IDs to detect stale handles.
- Component storage suitable for sparse entity/component composition.
- Typed component insertion, removal, lookup, and mutable access.
- Queries over one or more component sets.
- Resources/singletons for shared simulation data.
- Event queues/channels with clear frame/tick lifetime.
- Deferred structural changes through a command buffer (no unsafe mutation during queries).
- Explicit system stages/schedule ordering.
- Fixed-step and variable-step stage separation.
- Debug-visible entity/component/system metrics.
- Deterministic iteration policy where game correctness requires it.

Initial non-requirements:

- Runtime reflection for all components.
- Parallel scheduling from day one.
- Arbitrary dynamic query expressions.
- Full change detection semantics.
- Archetype optimization before profiling proves need.
- ECS serialization as the save format.

Implementation path:

1. Start with generational entity IDs and sparse-set component pools.
2. Implement basic typed storage and single-component queries.
3. Add multi-component queries and command buffering.
4. Add resources/events/stages.
5. Add profiling and deterministic ordering.
6. Only then evaluate archetype storage, parallel query execution, and advanced scheduling based on measured bottlenecks.

The canonical world state remains the durable model; the ECS is optimized for live simulation.

### 5.4 Renderer requirements

The renderer targets high fidelity on the High and Epic tiers and runs the Low tier on integrated graphics, optimized for clarity and debuggability as well as visual quality (ADR-0019).

MVP renderer capabilities:

- wgpu, behind the renderer facade (ADR-0003, ADR-0019), on native Windows and Linux.
- Window/swapchain management.
- Mesh buffers for LOD-chain static meshes (4 to 5 levels, per the asset budgets) and simple dynamic geometry, with tangents and vertex colour (ADR-0019, ADR-0011).
- Transform hierarchy or flat transform representation.
- Free third-person perspective camera (ADR-0020).
- PBR metallic-roughness materials, with unlit debug materials.
- Cascaded shadows for a directional light, with sky-based ambient lighting (ADR-0019).
- Clustered local lights.
- Simple transparent materials and particles.
- Frustum culling.
- Texture loading with block-compressed formats and mips (ADR-0019 texture rules).
- Debug primitives: lines, bounding boxes, nav/path visualization, text/overlay hooks.
- GPU and CPU timing counters where backend support permits.

Renderer non-goals (ADR-0019):

- Hardware ray tracing.
- Virtual shadow maps.
- Nanite-style virtualised geometry; classic LOD chains are used instead.
- Shader graph editor.
- Virtual textures (design note only).

Rendering budget principles:

- Lighting path (clustered forward+ or deferred) is chosen by prototype benchmark (ADR-0019).
- Batch/instance static props and LOD meshes where practical.
- Share materials aggressively.
- Limit unique textures/materials and alpha blending.
- Constrain dynamic light counts.
- Treat draw-call count, overdraw, texture memory, and shared-memory bandwidth as first-class budgets.
- Test frequently on integrated graphics and SteamOS-class systems.

### 5.5 Dependency posture

Evaluate every engine-relevant dependency under these rules:

1. Prefer well-maintained, small, focused Rust crates.
2. Put third-party library types behind engine-owned interfaces at module boundaries; do not leak them through high-level engine or game APIs where practical.
3. Avoid dependencies that become the de facto engine core.
4. Pin versions and document rationale in an ADR when a dependency shapes architecture.
5. Prefer dependencies that work well on native Windows, native Linux, Vulkan, and SteamOS.
6. Keep optional subsystems feature-gated where practical.

Before adding an engine-relevant dependency, state why it is needed, whether it is narrow plumbing or an engine/framework dependency, which engine-owned abstraction contains it, and whether an existing dependency already solves the problem. Request approval before adding any dependency that could constrain engine design.

Potential categories to evaluate (not mandatory selections):

| Need | Acceptable type of dependency | Engine responsibility |
|---|---|---|
| Window/events/input | Windowing crate such as `winit` | Normalize engine input events and lifecycle |
| GPU/rendering API | `wgpu` or direct Vulkan bindings | Own render/frame lifecycle and material/mesh/camera abstractions |
| Math | Focused math crate or small internal types | Own transform conventions and serialization boundary |
| Image decoding | Image/texture decoding crate | Own asset IDs, loading states, texture resources, atlasing policy |
| Audio device/backend | Focused audio crate | Own game audio events, mixers/categories, asset references |
| Serialization | `serde`, RON, JSON | Own schema versions, migrations, save envelopes, validation |
| HTTP (AI gateway) | Focused HTTP client | Own provider abstraction, timeouts, cancellation, fallback |
| Async | Tokio/smol/async channels if justified | Own job policy, thread ownership, cancellation, game-loop boundaries |
| Logging | `tracing` ecosystem | Own diagnostics taxonomy and debug UI integration |
| Hashing | Focused cryptographic hash crate | Own content-hash and provenance semantics |
| Physics/navigation | Narrow physics/pathfinding libraries or internal MVP implementations | Own domain-facing collision/nav abstractions and determinism policy |

### 5.6 Platform rules

- Build and test native Windows and native Linux binaries from the renderer milestone onward.
- WSL is for productive shell/headless work only. It is not release, renderer, input, audio, packaging, performance, SteamOS, or local-AI validation.
- Use `Path`/`PathBuf`; never concatenate filesystem paths manually.
- Treat filenames as case-sensitive. Use lowercase kebab-case for content filenames and IDs.
- Shader sources are excluded and use `snake_case.wgsl`. The kebab-case rule covers human-authored content addressed by stable ID; shaders are code referenced from Rust by constant and `include_str!` (decided 2026-09-16, see ADR-0011).
- Do not rely on the executable's working directory for asset paths. Resolve config, cache, logs, saves, and assets through explicit engine path services.
- Sort directory traversal results when deterministic content compilation matters.

## 6. Reference game: vertical slice

### Working premise

A player-character is a Wayfinder helping rebuild a frontier settlement after a magical collapse. The settlement, nearby wilderness, factions, ruined locations, and NPC relationships change in response to player choices.

### Slice map and cast

- Hub settlement: `Cinderwake Settlement`.
- Open-world cells around the settlement, streamed around the player (ADR-0021).
- One generated dungeon instance, kept as an instance (ADR-0021).
- Generated open-world sites placed into cells by seeded procedural code (ADR-0021).
- NPCs: a settlement leader, a researcher/quest giver, and a rival-faction representative.
- Factions: settlers and an old-forest order.
- Initial crisis: magical corruption is disrupting a water source.

### Slice gameplay

- Skill-based progression: use-based skills, with classes as starting presets. The slice ships one fully playable preset and a second defined in data (ADR-0022, amending ADR-0015 and ADR-0018).
- Free third-person movement and camera, with interaction by camera-centre ray (ADR-0020).
- One melee attack and one defensive action or dodge, with 3D hit shapes (ADR-0020).
- Combat as one system among gathering, crafting, building and survival.
- Two active abilities.
- Three regular enemies and one elite/boss variation.
- Health, stamina or equivalent action resource, damage, loot, inventory, and simple equipment.
- Five gatherable resources and three craftable recipes.
- Snap-piece building in a claimed area (ADR-0021), and one small settlement-upgrade loop.
- Dialogue choices, faction reputation, and persistent world flags.

### Slice AI proof

A meaningful player choice must lead to one of several valid quest branches. A branch should produce:

- A 4–6 objective quest graph composed from approved objectives.
- A reachable destination: a generated dungeon instance, or a generated open-world site placed into a cell (ADR-0021).
- A valid encounter plan within a difficulty budget.
- Faction-aware dialogue using canonical world facts.
- A reward package.
- A persistent world-state delta.
- A visual recipe using only approved asset/material/VFX libraries.

ADR-0010 safety rules apply unchanged to generated sites and quests.

Example branch directions:

- Help settlers secure waterworks and defend workers.
- Help a forest order investigate the corruption and negotiate access.
- Pursue a hidden relic for a higher-risk personal reward.

## 7. Core gameplay data model

### Stable identifiers

Use newtypes or strongly typed IDs. Avoid untyped strings in gameplay APIs where an ID type is appropriate.

```rust
pub struct QuestId(pub String);
pub struct NpcId(pub String);
pub struct FactionId(pub String);
pub struct RegionId(pub String);
pub struct EncounterId(pub String);
pub struct ItemId(pub String);
pub struct AbilityId(pub String);
pub struct PrefabId(pub String);
pub struct ContentHash(pub [u8; 32]);
```

Exact internal representation may evolve, but IDs must be stable across save/load and content revisions.

### Canonical world state

```rust
pub struct WorldState {
    pub world_seed: u64,
    pub content_revision: ContentRevisionId,
    pub clock: GameClock,
    pub player: PlayerState,
    pub regions: BTreeMap<RegionId, RegionState>,
    pub npcs: BTreeMap<NpcId, NpcState>,
    pub factions: BTreeMap<FactionId, FactionState>,
    pub quests: BTreeMap<QuestId, QuestState>,
    pub world_flags: BTreeSet<WorldFlag>,
}
```

This is illustrative, not a frozen API. Preserve the principles: explicit ownership, stable IDs, serializability, versionability, no runtime entity IDs.

### Domain events

Use events for durable state mutation.

```rust
pub enum DomainEvent {
    QuestAccepted { quest_id: QuestId },
    QuestObjectiveCompleted { quest_id: QuestId, objective_id: ObjectiveId },
    QuestCompleted { quest_id: QuestId, resolution: QuestResolutionId },
    FactionReputationChanged { faction_id: FactionId, delta: i16 },
    RegionStateChanged { region_id: RegionId, change: RegionChange },
    ItemGranted { item_id: ItemId, quantity: u32 },
    ContentRevisionAccepted { revision: ContentRevisionId, hash: ContentHash },
}
```

All domain events must be serializable, versioned/migratable, replayable, and understandable without a live runtime ECS world.

### Approved quest objectives

Initial quest objectives should be finite and explicit:

```rust
pub enum Objective {
    Reach { location: LocationId },
    TalkTo { npc: NpcId },
    Defeat { encounter: EncounterId },
    Collect { item: ItemId, quantity: u16 },
    Escort { npc: NpcId, destination: LocationId },
    Investigate { clue: ClueId },
    Decide { choice: ChoiceId },
}
```

### Approved ability/effect primitives

Game content may compose registered triggers, conditions, targeting rules, costs, cooldowns, and effects. The initial effect set may include:

```rust
pub enum Effect {
    DealDamage { amount: Fixed, damage_type: DamageType },
    ApplyStatus { status: StatusId, duration_ticks: u32 },
    SpawnEntity { prefab: PrefabId, count: u8 },
    ModifyResource { resource: ResourceType, amount: Fixed },
    Teleport { max_distance: Fixed },
    EmitEvent { event: GameEvent },
}
```

Do not add arbitrary code execution as an ability mechanism. Extend the trusted primitive set deliberately, with tests.

### Skills and XP (illustrative)

Skill progression uses integer ledgers and event-driven updates.

```rust
pub struct SkillId(pub String);            // content id from skills.ron
pub struct SkillState { pub xp: u32, pub level: u16 }
pub struct Progression {
    pub skills: BTreeMap<SkillId, SkillState>,
}
pub enum SkillEvent {
    SkillExperienceGained { skill: SkillId, amount: u32 },
    SkillLeveled { skill: SkillId, level: u16 },
}
```

Note: XP is an integer ledger. Every XP change is a `SkillExperienceGained` event, and there are no floats on any skill path. A level is recorded when it is reached and is never recomputed on load. Derived values such as gather speed are computed from the level and not stored. This is illustrative, not a frozen API.

### World cells (illustrative)

World state is stored as deltas relative to a deterministic baseline.

```rust
pub struct CellId { pub x: i32, pub y: i32 }   // grid coordinates
pub struct CellDelta {
    pub nodes: BTreeMap<NodeId, NodeState>,
    pub structures: BTreeMap<StructureId, StructureState>,
    pub sites: BTreeMap<SiteId, SiteRecord>,
}
pub struct WorldState {
    // ... existing fields ...
    pub cells: BTreeMap<CellId, CellDelta>,
}
```

Note: the seeded baseline of a cell is generated as a pure function of (world seed, cell id, content revision) and is never stored. Saved state is only the delta. A cell with no key in `cells` is untouched. Save size grows with the number of deltas, not with explored area. This is illustrative, not a frozen API.

### Building pieces (illustrative)

Building structures use fixed-point coordinates and explicit socket definitions.

```rust
pub struct PieceId(pub String);            // content id from pieces.ron
pub struct StructureId(pub u32);           // unique within its cell; producer takes the next value from the cell's counter, reducer checks it is free
pub enum SocketKind { WallEdge, FloorEdge, RoofEdge, FoundationTop }   // closed set, defined in Rust
pub struct Socket { pub kind: SocketKind, pub position_milli: [i32; 3], pub yaw_milli: i32 }
pub struct StructureState {
    pub piece: PieceId,
    pub position_milli: [i32; 3],
    pub yaw_milli: i32,
    pub owner: ClaimId,
    pub hit_points: u32,
}
```

Note: the support graph is a derived cache rebuilt from state on load and updated incrementally on place and remove. It is never saved. Collapse order is sorted by `StructureId`. This is illustrative, not a frozen API.

### Generated sites (illustrative)

Generated sites store provenance and player state. Their position comes from the seed.

```rust
pub struct SiteId(pub String);             // "site-" prefix, minted by code, never by a model
pub struct SiteProvenance {
    pub model: String,
    pub prompt_hash: ContentHash,
    pub seed: u64,
    pub content_hash: ContentHash,
}
pub struct SiteRecord {
    pub id: SiteId,
    pub overlay: Option<SiteOverlay>,      // names and encounter choices; None means template defaults
    pub provenance: Option<SiteProvenance>,
    pub state: SiteState,                  // state changes made by the player
}
```

Note: a site's position is not saved. Placement code recomputes it from the seed. A site that fails the fit checks is dropped and never recorded. A saved record holds the overlay, provenance and state changes. An overlay that fails validation is replaced by the template default. Provenance is never shown to the player. This is illustrative, not a frozen API.

## 8. Repository and content structure

Recommended repository organization (exact crate split to be confirmed by ADR; dependency direction must follow Section 5.1):

```text
.
├── PROJECT_CONTEXT.md
├── Cargo.toml
├── Cargo.lock
├── crates/
│   ├── platform/           # window/event loop/input backend integration, path services
│   ├── renderer/           # GPU backend implementation behind the engine renderer facade
│   ├── engine/             # ECS, schedule, time, events, assets, renderer/input/window facades, diagnostics
│   ├── game_core/          # IDs, schema types, domain state, events, rules (no engine/renderer types)
│   ├── content_runtime/    # manifests, content compiler, registry, validators
│   ├── procedural_gen/     # map grammar, encounters, seeded generation
│   ├── narrative/          # quest runtime, story state, template director
│   ├── ai_gateway/         # local/cloud provider adapters, schemas, queues/cache
│   ├── game_runtime/       # domain-to-ECS projection, gameplay systems, UI glue
│   ├── debug_tools/        # debug UI, inspection, replay, content candidate tools
│   ├── world_stream/       # (planned, D5) CPU cell loader; crate name assumed in renderer workstream
│   ├── asset_cooker/       # (planned, B6) glTF to cooked mesh and texture pipeline
│   ├── asset_format/       # (planned, B6) shared cooked-asset format; crate name assumed
│   └── sample_game/        # Cinderwake vertical slice
├── assets/
│   ├── content/
│   │   ├── manifests/
│   │   ├── abilities/
│   │   ├── biomes/
│   │   ├── dialogue/
│   │   ├── encounters/
│   │   ├── enemies/
│   │   ├── items/
│   │   ├── prefabs/
│   │   ├── quests/
│   │   ├── recipes/
│   │   ├── visual_recipes/
│   │   ├── skills.ron                # (planned, F5) skill definitions and unlock thresholds
│   │   ├── xp-rules.ron              # (planned, F5) action XP and daily caps
│   │   ├── pieces.ron                # (planned, F4) building piece definitions and sockets
│   │   ├── claims.ron                # (planned, F4) claim decay and upkeep
│   │   ├── stability.ron             # (planned, F4) support depth and cantilever limits, or folded into pieces.ron
│   │   ├── resource-nodes.ron        # (planned, F6) gatherable node definitions and yield tables
│   │   ├── survival.ron              # (planned, F6) survival stat tuning
│   │   ├── world.ron                 # (planned, F3) cell size, streaming radii, baseline seed salt
│   │   ├── site-templates.ron        # (planned, F9) generated site templates, bounded by the loader
│   │   └── generation.ron            # (planned, F9) site density, budgets and spawn-safety radius
│   ├── models/
│   ├── materials/
│   ├── textures/
│   ├── audio/
│   └── shaders/
├── saves/                  # ignored by Git except documented examples/fixtures
├── tests/
│   ├── fixtures/
│   ├── content_validation/
│   └── simulation/
└── docs/
    ├── adr/
    ├── ai_contracts/
    ├── content_authoring/
    └── plans/
```

Save-world generated content belongs under the save directory, not the shipped authored asset directory. Save-specific generated content must never silently overwrite shipped/authored content.

```text
saves/
  world_<world-id>/
    state.ron
    event_log.ndjson
    generated_content/
      candidates/
      accepted/
      rejected/
      assets/
```

## 9. AI content pipeline

### The AI's role

AI proposes semantic content and constrained configurations. It can propose:

- Quest graph parameters and narrative beats.
- Dialogue consistent with known NPC/faction/world facts and style guides.
- Map semantics: biome, room roles, progression constraints, landmarks, and encounter themes.
- Enemy variants assembled from approved archetypes, behavior templates, abilities, and visual tags.
- Encounter composition within explicit budgets.
- Reward flavor and allowlisted reward selections.
- Visual recipes referencing known assets, tags, materials, and VFX.

AI cannot:

- Execute commands or generate executable code that runs in the game.
- Write arbitrary Rust or native code into a running game.
- Directly mutate runtime ECS entities.
- Directly modify save files or write arbitrary files.
- Load arbitrary local files or access environment secrets.
- Bypass content validation.
- Introduce unbounded loops, unknown assets/effects, or unvalidated content references without a defined, reviewed creation path.

### Candidate lifecycle

```text
World facts + player context
        ↓
Template/director selects generation need
        ↓
AI proposes schema-constrained JSON or deterministic fallback builds candidate
        ↓
Parse and schema validate
        ↓
Resolve references and compile to internal content representation
        ↓
Validate invariants, budgets, compatibility, and progression
        ↓
Optional bounded repair attempt
        ↓
Dry-run maps/encounters/quest graph where applicable
        ↓
Queue for preview or safe-point activation
        ↓
Atomically commit an accepted, versioned WorldDelta (domain event + content revision)
        ↓
Project domain change into the runtime ECS
        ↓
Persist event, seed, content revision, provenance, and content hash
```

### Required validations

At minimum validate:

- Schema conformance and content version.
- Stable ID syntax and uniqueness.
- Every asset/content reference resolves.
- Quest graph has valid start/completion paths and no impossible prerequisites.
- All map objectives are reachable from a legal spawn/entrance.
- Spawn positions are safe and navigable.
- Enemy abilities and behavior templates are compatible.
- Resource, entity, particle, draw, memory, and simulation budgets are respected.
- Rewards are legal, non-duplicative where required, and progression-safe.
- Candidate mutations do not violate immutable story facts or player choices.
- Generated text satisfies configured content-safety rules.

### Generation failure policy

- Reject malformed or invalid candidates.
- Optionally send compact machine-readable validation errors to the model for a bounded number of repair attempts.
- Never loop indefinitely.
- On timeout, unavailable local model, memory pressure/OOM risk, invalid output, or repeated repair failure: use a deterministic template/content fallback.
- Never block a critical quest, save operation, load completion, combat state, or player movement on generation.

## 10. Local AI policy

### Runtime model strategy

- Reference backend: `llama.cpp` server exposing an OpenAI-compatible local HTTP endpoint.
- Convenience backend: Ollama adapter.
- The engine owns a provider abstraction, not a provider-specific API surface.
- Inference runs out of process, with at most one generation job by default.
- Models are external/configurable; do not bundle or vendor model weights into the repository or base game.
- Use compact quantized models and structured outputs for local operation.
- Keep prompts concise and pass only relevant canonical state.
- Cache completed semantic outputs by context/model/prompt/schema/seed hashes.

### Resource policy

Local AI is an enhancement. The baseline game must target a minimum device profile with AI disabled or templates-only.

Do not start AI inference:

- In the rendering loop.
- In fixed-step simulation.
- During combat/input-sensitive sections by default.
- When frame-time targets are missed.
- When free RAM is below the configured threshold.
- When the player selects a battery/thermal-constrained profile.

Prefer inference:

- At title screen or session setup.
- During rest/inn/campfire/hub interactions.
- While preparing an instanced region.
- During loading screens.
- Between gameplay sessions.

Default configuration example:

```toml
[ai]
enabled = true
mode = "local_preferred" # disabled | templates_only | local_only | cloud_only | local_preferred
max_generation_jobs = 1
max_queue_depth = 3
max_context_tokens = 4096
max_output_tokens = 1200
request_timeout_seconds = 30
max_repair_attempts = 1
allow_generation_during_gameplay = false
min_free_memory_mb = 4096
cache_enabled = true
cloud_fallback_enabled = false
```

### Device profiles

| Profile | Design expectation |
|---|---|
| Minimum | 8 GB RAM and a Vulkan-capable integrated GPU. Game is playable using authored/templates-only content; no local model requirement. |
| Recommended local AI | 16 GB RAM and a reasonably modern Vulkan-capable iGPU/APU. Small quantized model runs at safe points only. |
| Enhanced | 32 GB RAM or a discrete GPU. Larger model and more background generation are allowed. |

## 11. Visual and asset rules

### MVP art direction

- High-fidelity PBR (ADR-0019). The exact art style (for example stylised versus realistic) is an open owner decision.
- Physically based metallic-roughness materials.
- HDR lighting with tonemapping.
- Physically based sky, aerial perspective, and height fog.
- Volumetric fog on every tier.
- Readable silhouettes.
- Compact palette per biome and faction.
- Limited real-time lights.
- Simple particle effects.
- Classic LOD chains, no Nanite.
- Default 4 levels (LOD0 to LOD3).
- 5 levels for architecture, large landscape, and rocks.
- 1 level for small natural pieces.
- Reduction ratio 0.5 per level, or 0.7 for base-building.
- Triangle targets and warn/error limits are in docs/reference/observed-asset-budgets.md section 2c.
- Textures mostly 1024 to 2048.
- Max 2048 on High, 1024 on Low.
- Base colour is BC7 sRGB.
- Normals are BC5 linear at half the base-colour width, with a floor of 512.
- ORM (occlusion, roughness, metallic) is packed into one BC1 linear texture.
- Emissive is a separate BC1 sRGB texture, max 1024.
- Virtual textures are a design note only, not in scope.
- Small set of master materials: environment metallic-roughness, character, foliage small and tree, tree trunk, impostor, VFX opaque/translucent/additive, two-sided cloth.
- Each material is an instance: a master plus a parameter block.
- Variants are feature flags, not new masters.
- Cascaded shadows: 2 cascades on Low, 4 on High, 5 on Epic.
- Global illumination is off on Low.
- Visual variation comes from palettes, silhouettes, materials, decals, animation timing, and procedural placement.

### Asset sources

1. Original assets only. The reference game is studied for configuration and statistics only; no assets, pixels, or geometry from it are used (clean-room policy, master plan section 3).
2. Generate procedural props and environmental variation from deterministic seeds where appropriate.
3. Use explicitly licensed CC0/free assets only when provenance is tracked.
4. Keep model, texture, and audio source and license metadata in an asset manifest.
5. Do not depend on runtime generative-art workflows for the MVP.

## 12. Performance rules

### Target frame rates

| Tier | Hardware class | Resolution and upscaling | Frame target | Notes |
| High | Radeon RX 7900 XT class | 1440p native or 4K upscaled | 60 fps, GPU frame time <= 16.6 ms | Full feature set, including GI |
| Epic | Radeon RX 7900 XT class | 1440p native or 4K upscaled | 60 fps | Highest settings tier; same hardware target as High |
| Medium | Mid-range discrete GPU | 1080p | 60 fps | Reduced settings |
| Low | Integrated graphics and Steam Deck class | 720p to 900p, upscaled | 30 to 40 fps | No GI; volumetric fog and shadows kept (2 cascades) |

- Tiers are data. A tier cannot enable a feature the adapter lacks; the profile is reduced and the reduction is reported.
- AI must not be needed to meet any gameplay or rendering target.

### Performance constraints

- Establish profiling early on native Windows and native Linux builds.
- Test on integrated graphics before adding visual complexity.
- Separate `Graphics Quality` from `AI Quality` in settings.
- Keep local inference in a separate process and use low scheduling priority where possible.
- Cap simultaneous AI jobs at one by default.
- Favor pre-generation and caching over live generation.
- Do not memory-map or retain large models indefinitely on low-memory systems if this causes pressure.
- Use deterministic seeds for maps, encounters, and variation.
- Do not optimize ECS storage, parallel scheduling, rendering architecture, or asset streaming before profiling demonstrates need.

### Tracked metrics

Do not impose premature hard performance gates, but track:

- CPU and GPU frame time on designated low-end hardware.
- Memory use with AI disabled (the required baseline).
- Memory/frame-time effect of optional local inference.
- Entity count, draw calls, visible triangles, texture memory, dynamic-light count, and particle count.

## 13. Milestone plan

Build in this order. Do not skip directly to AI generation, networking, a general editor, or high-fidelity rendering.

### M0 — Design, reference study, and workspace

Deliverables:

- ADRs for engine-from-scratch scope, renderer dependency choice, ECS storage strategy, and platform/input strategy.
- Study Bevy and other Rust-engine implementations as references.
- Measurable MVP renderer and ECS capability requirements.
- Rust workspace and crate boundaries matching Section 8.
- Root developer tooling: `rustfmt`, `clippy`, test commands, and CI for Linux and Windows.

Exit criteria:

- ADRs accepted; workspace builds; `cargo fmt --check`, `cargo clippy`, and `cargo test` run in CI.
- No elaborate plugin, editor, or renderer architecture committed prematurely.

### M1 — Platform and frame loop

Deliverables:

- Window creation and event loop behind the engine platform facade.
- Input normalization.
- Clock, frame timing, fixed-timestep accumulator.
- Engine path services for config, cache, logs, saves, and assets.
- Logging, error reporting, debug build configuration.
- Clear/swap a renderer target.

Exit criteria:

- A native executable opens a window, handles input, runs stable update/render timing, and reports diagnostics.

### M2 — Minimal ECS

Deliverables:

- Generational entities.
- Sparse component pools.
- Resources, events, simple and multi-component queries.
- Commands/deferred structural mutation.
- Ordered systems and fixed/variable stages.
- Headless ECS tests.

Exit criteria:

- A headless test world can spawn, update, query, despawn, and reject stale handles deterministically.

### M3 — Minimal low-poly renderer

Deliverables:

- Mesh upload/draw.
- Camera and transforms.
- Basic material/shader.
- Simple directional/ambient lighting.
- Renderer-facing ECS components and extraction/projection boundary.
- Debug draw and basic debug overlay.
- Frame timing and draw-call counters.

Exit criteria:

- The engine renders a controllable low-poly scene at the intended frame target on integrated graphics, validated on native Windows and native Linux.

### M4 — Domain and content foundation

Deliverables:

- Stable game-domain IDs.
- Canonical world state and domain event reducers.
- Versioned RON save envelope and save/load.
- Seed management and replay-friendly event log.
- RON content manifests/registries and a headless content-validation command.
- Domain-to-ECS projection.
- Mock AI provider interface with no actual model dependency.

Exit criteria:

- A saved world can be loaded, projected into runtime entities, changed through domain events, and saved/replayed without runtime entity identity leaking into persistence.
- Save/load round trip restores equivalent canonical state.
- Headless content validation runs in CI.

### M5 — Hand-authored playable slice

Deliverables:

- Character controller, camera, collision, interaction, health, action resource, combat, enemy behavior, loot, inventory.
- Hub, wilderness lane, and dungeon using low-poly placeholders.
- Dialogue, branching quest, faction state, world flags.
- Minimal gathering, crafting, and settlement upgrade loop.
- Three enemy types and one elite.
- Primitive/low-poly asset kit and asset registry.
- Debug inspectors and performance overlays.

Exit criteria:

- The slice's loop is closed and playable offline, end to end: talk, accept a quest, fight, gather, craft, upgrade, travel, die, respawn, save, and resume.

> **Re-scoped 2026-09-18 (ADR-0016).** M5 originally exited on "a 20–30 minute
> hand-authored quest loop". Auditing the slice showed combat is one hardcoded
> swing and the authored ability content is never executed, so the content pass
> moved to M10, after the mechanics milestones that make it worth authoring
> against. See `docs/plans/m5-vertical-slice.md`.

### M6 — Combat and the ability runtime

Deliverables:

- An executor for the `Effect` allowlist: `DealDamage`, `ApplyStatus`, `SpawnEntity`, `ModifyResource`, `Teleport`, `EmitEvent`, each bounded and tested.
- Ability costs, cooldowns in fixed ticks, targeting resolution, and wind-up/recovery commitment.
- Status effects as an allowlisted modifier set with durations in ticks.
- Typed damage and resistances on the runtime side; canonical `Pool` stays `u32`.
- Equipment that modifies combat, through the existing `ItemEquipped` event.
- Player and enemy ability loadouts driven by `ClassDef.abilities` and `EnemyDef.abilities`.

Exit criteria:

- The player can use two active abilities from content (§6), and `ember-cleave`, `warden-guard`, and `call-husks` all do what they say.
- An ability authored purely in RON changes gameplay with no Rust change.
- Every `Effect` variant has an executor and a test proving it bounded.
- No save-format change: `format_version` stays 1.

Plan: `docs/plans/m6-combat-ability-runtime.md`.

### M7 — Presentation runtime

> **Inserted 2026-09-22 (ADR-0017).** Playtesting M6 showed that a runtime nobody can see cannot be judged. This milestone makes combat perceivable and makes presentation content-driven, ahead of the enemy variety that depends on it. Former M7–M14 are now M8–M15.

Deliverables:

- Per-vertex colour, and a flash overlay so cue materials still read on vertex-coloured bodies.
- A flat-shaded low-poly shape library, and multi-part `Model` rendering composed at extraction with no ECS hierarchy.
- Model, body-plan, and palette recipes as content, with every slice character modelled.
- Animation clips as data, timed as fractions of the gameplay windows.
- Composed, opaque, capped visual effects, and ability presentation links.
- `Effect::Projectile`: skillshots bounded in speed, range, radius, pierce, live count, and nesting.

Exit criteria:

- A new enemy with its own model, animations, and a skillshot ability is a content file and nothing else.
- Presentation never moves simulation, for bodies, parts, and effects alike.
- Every projectile bound has a test that fails loudly.
- No save-format change: `format_version` stays 1.

Plan: `docs/plans/m7-presentation-runtime.md`.

### M8 — Enemies, encounters, and combat variety

Deliverables:

- Per-enemy tuning from content — aggro, preferred range, move speed, attack timing — with `BehaviorTemplate` supplying defaults rather than values.
- Enemy ability selection by range and cooldown, deterministic in id order.
- Visible telegraphs, poise, and bounded hitstun.
- Elite mechanics built from abilities rather than a larger health pool.
- `threat_budget` validated against what encounter slots actually cost.

Exit criteria:

- A new enemy that plays differently is a content file and nothing else.
- The elite fights differently from the three regulars.
- The three authored encounters each demand something different of the player.

Plan: `docs/plans/m8-enemies-encounters.md`.

### M9 — Data-driven content runtime

Deliverables:

- Typed definitions for abilities, enemies, encounters, items, loot, recipes, quest graphs, dialogue conditions, biomes, and visual recipes.
- Content compiler/registry.
- Runtime-safe content loading/reloading for development.
- Validators and regression fixtures.

Exit criteria:

- A new quest, enemy variation, and encounter can be created through content files without changing Rust game logic.

### M10 — Content pass: the playable slice

Deliverables:

- The hub, wilderness lane, and dungeon authored to depth against a runtime where content drives behaviour.
- Quest, dialogue, faction, and world-flag content sufficient for a full session.
- Tuning passes on combat, encounters, economy, and pacing.

Exit criteria:

- A 20–30 minute hand-authored quest loop works offline on the minimum device profile.

### M11 — Procedural regions

Deliverables:

- Biome grammar, room graph generation, seeded assembly, landmark/objective/spawn placement.
- Navigation/reachability validation.
- Encounter budgeting and placement validation.
- Generated dungeon instancing from a `MapSpec`.

Exit criteria:

- The engine can deterministically generate a navigable dungeon that contains reachable quest objectives and valid encounters.

### M12 — Deterministic content director

Deliverables:

- World hooks, quest templates, constraint solver, faction-aware content selection, candidate queue, acceptance/rollback tools.
- No LLM dependency.

Exit criteria:

- The game creates varied, persistent, mechanically valid quests using templates only.

### M13 — Optional local AI integration

Deliverables:

- Out-of-process AI gateway.
- `llama.cpp` and optional Ollama adapters behind an engine-owned provider interface.
- Structured JSON request/response contracts.
- Candidate parser, compiler, validation/repair loop, cache, timeouts, and template fallback.
- Local-AI settings and resource policy.
- Debug UI for prompts, outputs, diffs, validator failures, seeds, and provenance.

Exit criteria:

- With local AI enabled, the game can propose an accepted quest variation. With AI unavailable, the exact flow reliably uses deterministic fallback content, and the game remains fully functional.

### M14 — AI-generated quest arc demo

Deliverables:

- Background story-arc planning.
- Generated contextual dialogue.
- AI-proposed map semantics, encounter composition, reward flavor, enemy variation, and visual recipes.
- Automated batch validation/simulation harness.

Exit criteria:

- A player choice produces a coherent, persistent, mechanically valid questline with no softlock and no dependency on cloud services.

### M15 — Future multiplayer readiness review

No networking implementation yet. Verify:

- Domain events are deterministic/serializable and versioned.
- State reducers are explicit.
- Content packs and generated revisions have hashes/manifests.
- AI changes are atomic transactions suitable for server authority.
- No gameplay persistence depends on local-only ECS identity.

### M16 — Decisions and reference docs

Deliverables: ADRs 0019 to 0022 approved, this document updated for them, and reference docs A1 and A3 done. Exit criteria: the owner signs off the scope and the conflicts are resolved.

### M17 — Renderer foundation

Deliverables: render graph, HDR, textures, and PBR materials (renderer workstream B1 to B4, E1). Exit criteria: a textured PBR scene with tonemapping runs at 60 fps on an RX 7900 XT class GPU, and the existing slice still runs.

### M18 — Lighting and sky

Deliverables: cascaded shadows, image-based lighting, procedural sky, ambient occlusion, and clustered lights (C1, C2, C4, C5, D3 sky). Exit criteria: a day and night scene with stable shadows, and golden-image tests in CI.

### M19 — Free camera and 3D controller

Deliverables: free third-person camera, 3D character controller, and part of the skeletal animation work (F1, F2, F8 partial). Exit criteria: the player traverses 3D terrain, and the old 2.5D mode can be removed.

### M20 — World

Deliverables: terrain, foliage, water, cell streaming, and the instancing that supports them (D1, D2, D4, D5, F3, B5). Exit criteria: a 1 km² streamed world with instanced foliage, within the frame-time budget.

### M21 — Systems

Deliverables: building, skills, gathering and crafting, and survival (F4 to F6). Exit criteria: a player can build a base, craft gear, and have both persist across save, load, and replay.

### M22 — Combat and AI

Deliverables: 3D combat, skeletal animation, and navmesh pathing (F7, F8). Exit criteria: three enemy types can be fought in the open world.

### M23 — Global illumination and polish

Deliverables: dynamic GI, volumetrics, TAA and an upscaler, particles, and quality tiers (C3, D3, E2 to E5). Exit criteria: the visual target is met in a capture comparison.

### M24 — Integration

Deliverables: AI-content adaptation to the open world, multiplayer readiness, and save migration (F9, F10, G5). Exit criteria: the new vertical slice is playable end to end.

## 14. Engineering conventions

### General

- Prefer simple, explicit, deterministic, testable designs over abstraction for hypothetical future needs.
- Prefer straightforward, readable systems to clever generalized abstractions.
- Keep public engine-facing APIs small, typed, and intentional.
- Make state ownership, lifecycle, and mutation boundaries clear.
- Add an Architecture Decision Record (ADR) for decisions that are expensive to reverse, including dependency and architecture choices.
- Do not introduce a dependency without documenting why existing crates/stdlib are insufficient (see Section 5.5).
- Pin dependency versions and keep a lockfile.
- Favor deterministic, testable pure functions for domain reducers, validators, procedural generation, and content compilation.
- Keep engine runtime, renderer, and third-party backend types out of core domain types wherever practical.
- Use generational runtime entity IDs and validate stale-handle behavior.
- Use command buffers/deferred structural mutation in the ECS rather than mutation during queries.
- Separate fixed-timestep simulation from variable-rate rendering.
- Avoid `unwrap()`/`expect()` in recoverable runtime paths; propagate or handle errors with useful context.
- Use structured logging and diagnostics rather than ad hoc `println!` debugging.

### Testing

Add tests alongside major systems, before expanding scope:

- ECS: entity generation/reuse/stale-handle behavior, component insertion/removal/despawn cleanup, query correctness, deferred command ordering, event lifecycle, deterministic update/schedule order.
- Renderer: startup and device capability checks on native Windows and Linux, shader/pipeline error reporting, simple render smoke tests where feasible, frame timing and draw-call counters.
- Domain: unit tests for state reducers, ability rules, and validators; save/load/replay equivalence; deterministic replay using event log and seed.
- Content: fixture tests for RON/JSON content parsing, schema-version migrations, and ID/reference resolution.
- Generation: property/fuzz tests for map and quest generation where useful; quest graph reachability/completion validation; seed reproducibility; encounter budgets and spawn safety.
- AI pipeline: candidate parsing, rejection, bounded repair, timeout/unavailable-provider fallback, atomic commit, and rollback.
- Integration: headless tests for quest completion/reward/rollback flows; native Windows/Linux smoke checks for runtime/platform changes.
- Performance benchmarks/profiling checks only after a stable functional path exists.

### Save compatibility

- Include explicit format/schema versions.
- Support migrations for shipped/save-worthy formats.
- Preserve immutable accepted content revisions with saves.
- Never silently discard player world state on a parsing or version error.

### Error behavior

- Prefer graceful degradation and fallback over crashing.
- Display clear developer diagnostics in debug builds.
- Keep player-facing errors short and actionable.
- AI/provider failures must never make a save unusable.

## 15. Development workflow

### Required project-context workflow

Before nontrivial work, the agent should:

1. Read `PROJECT_CONTEXT.md`.
2. Read any relevant ADRs in `docs/adr/`.
3. Inspect the relevant crate/module, tests, and content fixtures before proposing architecture.
4. State the requested change, constraints, assumptions, files likely affected, risks, and verification plan.
5. For any proposed dependency, state whether it is a narrow building block or an engine-level framework dependency.
6. Identify how the change preserves low-spec native Windows/Linux/SteamOS goals and single-player-first scope.
7. For broad or high-risk changes, propose tests and a minimal implementation plan and wait for approval before editing.
8. Implement the smallest coherent change.
9. Run formatting, targeted tests, the full relevant test suite, and content validation.
10. Summarize changed files, design choices, validation performed, limitations, and follow-up work.

### Required restrictions for agents

An agent must not, without explicit approval:

- Add Bevy or Bevy ecosystem crates as dependencies.
- Add another game engine or game framework as a hidden engine substrate.
- Add a renderer/ECS framework that defeats the from-scratch objective.
- Introduce a generic plugin/reflection/editor/scripting system before the vertical slice requires it.
- Add cloud API keys, tokens, or credentials to source control.
- Add analytics/telemetry that sends data externally, or mandatory accounts/external services.
- Add a cloud inference dependency as a requirement for gameplay.
- Add arbitrary code execution to AI or content pathways.
- Add mandatory local-model dependencies or AI model weights to source control.
- Persist runtime entity IDs as canonical identifiers.
- Change save formats without migration/version handling.
- Replace the content format wholesale.
- Add networking implementation before the single-player readiness review.
- Add heavyweight rendering, model-generation, or asset-generation dependencies to the MVP.
- Make broad directory/crate restructuring unrelated to the requested task.

### Preferred task format

Use task requests with this structure:

```text
Goal:
Scope:
Non-goals:
Acceptance criteria:
Relevant files/systems:
Constraints from PROJECT_CONTEXT.md:
Verification commands:
```

### Definition of done

A change is complete when:

- It meets stated acceptance criteria.
- It respects this project context and relevant ADRs.
- It includes focused tests or a documented reason tests are not applicable.
- It passes `cargo fmt --check`.
- It passes targeted `cargo test` commands.
- It passes content validation when content/schema behavior is changed.
- It avoids unrelated refactors and generated-file churn.
- It updates docs/ADRs when behavior or architecture changes materially.

## 16. Open decisions

### Product and design decisions

These must be resolved before permanently committing to related systems:

All fifteen were resolved on 2026-09-15.

1. ~~Exact camera/perspective.~~ Superseded 2026-10-09: free third-person orbit camera (ADR-0020). The earlier fixed-angle 2.5D choice was ADR-0007.
2. ~~Exact combat feel.~~ Souls-lite and deliberate: stamina-gated, committed attacks, dodge i-frames; all timing windows expressed in fixed ticks (ADR-0007). Amended 2026-10-09: 3D hit shapes and camera-centre aim (ADR-0020).
3. ~~World topology.~~ Handcrafted hub and wilderness lane plus generated dungeon instances (ADR-0007). Amended 2026-10-09: the open world is cell-streamed, generated sites are placed into cells, and dungeons stay instanced (ADR-0021).
4. ~~Core-loop priority ranking.~~ Narrative/quests and combat led (ADR-0007). Amended 2026-10-09: a free-form survival and crafting sandbox, with combat as one system among building, gathering, crafting, and survival (master plan section 1, ADR-0020).
5. ~~Player progression.~~ Superseded 2026-10-09: skill-based progression, with classes as starting presets (ADR-0022, amending ADR-0015 and ADR-0018).
6. ~~Death/failure model.~~ Respawn at the settlement, keep progress; no corpse run (ADR-0007). Survival failure (zero warmth or food) is still open (gameplay workstream D-h).
7. ~~Settlement building scope.~~ Superseded 2026-10-09 for claimed areas: snap-piece building, with no freeform voxel building or terrain deformation (ADR-0021, amending ADR-0007).
8. ~~Player-facing AI visibility.~~ Invisible; provenance recorded in saves but never surfaced (ADR-0010).
9. ~~Whether AI may create permanent named NPCs, factions, locations, items, and lore.~~ Only inside generated instances, in an instance-scoped ID namespace; promotion to canonical status requires human authoring (ADR-0010). Amended 2026-10-09 for persistent open-world sites: their records are saved, while the seed still decides where they are (ADR-0021, owner decision D-a).
10. ~~Whether player free-form natural-language actions are a near-term goal.~~ No; player input stays bounded (ADR-0010).
11. ~~Generated content reroll/edit/review controls for players.~~ Developer-only review tooling; no player-facing controls (ADR-0010).
12. ~~Cloud fallback consent, privacy policy, and optional telemetry policy.~~ Cloud adapter opt-in and disabled by default, never required; no telemetry ever (ADR-0010).
13. ~~Project intent.~~ Portfolio piece: favour a tight, finished vertical slice over engine breadth.
14. ~~License choice for code and content.~~ Code dual-licensed `MIT OR Apache-2.0`; assets licensed separately with tracked provenance.
15. ~~Team size, expected development time, and desired demo duration.~~ Solo and part-time: milestones stay small, independently valuable, and resumable after gaps. Demo target remains a 20–30 minute offline slice, delivered in **M10** rather than M5 (amended 2026-09-18, ADR-0016: mechanics precede the content pass).

Owner decisions of 2026-10-09 (master plan section 9): single-player first, with networking out of scope; the High tier targets an RX 7900 XT class GPU, with a Low fallback tier; the AI-content pipeline stays on top of the open world under the ADR-0010 safety rules.

### Engine technical decisions

Resolve through ADRs and small spikes, not long speculative implementation:

1. ~~Rendering API boundary: portable abstraction such as `wgpu` versus direct Vulkan bindings.~~ Resolved 2026-09-15: `wgpu`, see ADR-0003.
2. ~~ECS storage: sparse sets only initially versus early archetype support.~~ Resolved 2026-09-15: sparse sets only, see ADR-0005.
3. ~~ECS execution: single-threaded deterministic schedule first versus staged parallelism.~~ Resolved 2026-09-15: single-threaded deterministic schedule, see ADR-0005.
4. ~~Math crate/internal math strategy and coordinate conventions.~~ Resolved 2026-09-15: `glam` behind `engine::math`; right-handed, Y-up, −Z forward, metres, `0..1` depth, see ADR-0008.
5. ~~Physics/collision: minimal internal 2D/3D collision versus narrow external backend.~~ Resolved 2026-09-15: minimal internal collision, see ADR-0009. Amended by ADR-0020: collision now covers terrain heightfields and vertical motion.
6. ~~Navigation: grid/A* for MVP versus navmesh pathfinding.~~ Resolved 2026-09-15: navmesh with A* plus funnelling; validators run on the navmesh, see ADR-0009. 3D navigation is still open (gameplay workstream F8); ADR-0020 does not decide it.
7. ~~Audio backend.~~ Resolved 2026-09-15: `cpal` with an engine-owned mixer; decoder crate still to be approved, see ADR-0012.
8. ~~Asset import approach and shader workflow.~~ Resolved 2026-09-15: glTF source baked to an engine format offline; WGSL hot-reloaded in dev, embedded in release, see ADR-0011. Amended by ADR-0019 (baked meshes carry tangents and vertex colour) and ADR-0020 (skeletal clips baked through the asset cooker).
9. ~~Window/input crate and controller support timeline.~~ Resolved 2026-09-15: `winit` now, gamepads via a focused crate before Steam Deck validation, see ADR-0004.

(Visual perspective is tracked as product decision 1 above.)

Open after ADR-0019 to ADR-0022:

- Lighting path, clustered forward+ or deferred, chosen by the B2 prototype benchmark (ADR-0019).
- Global illumination design, chosen by the C3 prototype gate, with a DDGI-style probe grid as the fallback (ADR-0019).
- Batching of vertex layout changes into one step, and the device limits to request (renderer workstream sections 2 and 6).
- Gameplay decisions: D-a and D-c are decided 2026-10-09; the other D items in docs/plans/20-gameplay-workstream.md section 5 are open.

## 17. Immediate next tasks

1. Install the Rust toolchain on the development machine.
2. Start Phase 1 (M17, renderer foundation): render graph (B1), HDR pipeline and tonemapping (E1), texture system (B3), and PBR materials (B4), following `docs/plans/10-renderer-workstream.md`.
3. Capture reference frames with Frame Generation off, and collect RGP timings per pass on the RX 7900 XT, following `docs/reference/capture-plan.md`.
4. Run a gameplay design pass for F4 (building) and F6 (gathering, crafting, survival), following section 3 of `docs/plans/20-gameplay-workstream.md`.

## 18. Documentation map

- Full project source of truth: `PROJECT_CONTEXT.md` (this document, including the engine-from-scratch decision)
- Architecture decisions: `docs/adr/` — establish ADRs before selecting windowing, rendering, or physics dependencies
- AI schemas, prompts, and contracts: `docs/ai_contracts/`
- Content authoring documentation: `docs/content_authoring/`
- Milestone/feature plans: `docs/plans/`
- Test fixtures: `tests/fixtures/`
- Master plan for high-fidelity rendering and free-form gameplay: `docs/plans/00-master-plan-fidelity-and-freeform.md`
- Renderer workstream: `docs/plans/10-renderer-workstream.md`
- Gameplay workstream: `docs/plans/20-gameplay-workstream.md`
- Test and CI plan: `docs/plans/30-test-and-ci-plan.md`
- Reference studies (prose specifications from public sources and owned-copy observation; no extracted code, shaders, or assets are committed): `docs/reference/*.md`, currently `ue5-feature-survey.md`, `capture-plan.md`, `gameplay-survey.md`, `observed-asset-budgets.md`, and `observed-settings.md`

## 19. Change log

| Date | Decision/change | Notes |
|---|---|---|
| 2026-10-09 | High-fidelity renderer and free-form gameplay: ADRs 0019 to 0022 accepted; master plan approved; PROJECT_CONTEXT updated | ADR-0019: high-fidelity PBR renderer on wgpu with render graph, HDR, cascaded shadows, software GI, and Low/Medium/High/Epic tiers (amends ADR-0017 and ADR-0011). ADR-0020: free third-person camera, camera-ray interaction, 3D hit shapes, skeletal animation (supersedes the ADR-0007 camera and mouse-aim; amends ADR-0009). ADR-0021: cell streaming, snap-piece building in claimed areas, generated open-world sites saved in cells (amends ADR-0007 and ADR-0010 for open-world sites only). ADR-0022: skill-based progression with classes as presets (amends ADR-0015 and ADR-0018). Single-player first is unchanged. Milestones M16 to M24 added. |
| 2026-09-15 | Initial source-of-truth document created | Single-player first; local-first optional AI; Bevy/Rust; Linux/SteamOS and low-end integrated graphics; text-first ECS-driven workflow; low-poly MVP assets |
| 2026-09-15 | Engine-from-scratch decision accepted | Bevy is a reference implementation only; no Bevy runtime dependency |
| 2026-09-15 | M0 workspace, tooling, and ADRs 0001–0006 created | Crate layout per Section 8; `rustfmt`/`clippy`/CI for Linux and Windows; `unsafe_code` forbidden workspace-wide |
| 2026-09-15 | Dependencies approved: `wgpu`, `winit`, `serde` + `ron` | `tracing`, `glam`, and `serde_json` declined for now; diagnostics are engine-owned and math is deferred (ADR-0006) |
| 2026-09-15 | All Section 16 open decisions resolved | Product decisions 1–15 and engine decisions 4–8; recorded in ADR-0007 through ADR-0012; `glam` and `cpal` approved, `gilrs` and an audio decoder still pending |
| 2026-09-22 | M8 waves 2 and 3: poise, the ashlord, and encounters in waves | Built together. Poise (`game_runtime::combat::Poise`) makes a trade cost something: damage taken, read from the drop in health so swings, abilities and bolts all count, drains it, and at zero the body staggers for 24 ticks, losing a blow it was still winding up (never a live one) and unable to move or act. Poise then refills and cannot break again for 45 ticks, so a body is staggered at most 24 of every 69 ticks however fast it is hit; a test hits one every tick for ten seconds to prove there is no stunlock. Enemy poise is tuning with template defaults (melee 20, ranged 12, charger 40, support 30), and the player has 30, so one clean swing staggers a husk while a bark warden holds. Enemies no longer swing at air: a fallback swing starts only if it can reach, which ended the spitter's and the ashlord's swings from six and seven metres. The ashlord gained `ash-slam`, a telegraphed area blow for anyone who closes in, dropped from 240 to 160 health, and stands on 120 poise, so it is hard because of what it does rather than how long it takes to kill. Enemy placements gained a `wave`: wave 0 stands on arrival, each later wave comes a second after the one before it falls, announced by the encounter's optional `arrival` effect where each enemy will stand, and a fight clears only after its last wave. The waterworks ambush now comes in two waves. The loader now refuses an encounter whose slots cost more threat than its budget, whose placements do not match its slots, or whose waves skip a number; `waterworks-ambush` had declared 4000 against 4200, proof the field was inert, and now declares 4500. Eight existing tests assumed all four ambushers stand at arrival and were updated to their intent rather than their old numbers. Every rule was mutation-checked. Content validates at 67 definitions. 726 tests |
| 2026-09-22 | M8 wave 1: enemy tuning moves into content | `EnemyDef` gains an optional `tuning` block - aggro acquire and release, preferred range, move speed in metres per second, swing wind-up, active and recovery ticks, reach and strike radius - with every field optional, falling back to the behaviour template's value bit for bit, so existing content plays exactly as before; a test holds that for all four templates. The template still decides how an enemy thinks, and content sets only numbers (ADR-0010). Validation refuses out-of-range values, and a release closer than acquire (new `ReleaseInsideAcquire` error), which would make an enemy flicker at the edge of aggro; `EnemyBrain::tuned` clamps everything anyway, and keeps release at least acquire for input that bypassed validation. Enemies now choose among their abilities by a rule: the first in their authored list that is off cooldown, affordable and in range, otherwise a swing, with each ability's range read from the ability itself - a projectile's range, the targeting reach, or anywhere for self-casts. A deliberate deviation from the plan: authored order rather than id order, so the list is the enemy's priority while remaining data, not a procedure. The resin spitter now leads with `resin-spit`, a slow telegraphed skillshot doing corruption damage, with a `resin-glob` model, a drip trail and a splash. The exit criterion is met by a test that parses `ash-lurker` purely from RON; in a running slice it notices the player at its own 12 m, closes at its own 6 m/s, stops at its own 3 m, and winds up over its own 40 ticks. Every rule was mutation-checked. Two brittle tests surfaced and were fixed: a stamina test read the first stamina pool in the world, which became the spitter's once it had an ability, and a test pinned the exact number of shipped effects, which now checks named effects. Content validates at 65 definitions. 712 tests |
| 2026-09-22 | Mouse aim and cast modes | Swings and abilities now aim at the cursor, projected onto the ground each fixed tick as input (`sample_game::aim`); the engine gained a `Viewport` resource, published by the runner on start and resize, plus `Camera::ray_through` and `Camera::ground_point`, tested by projecting ground points to pixels and back. Facing separates walking from acting: the body faces where it walks until it acts, an action turns to the aim the tick it starts and keeps turning through the wind-up only, and locks the tick the blow is live or the effects land, so a bolt leaves toward where the cursor is at release, while walking never turns a committed body; this is recorded as an amendment to ADR-0007, the project owner having chosen tracking through the wind-up over locking aim at the start. Left-click and Space swing, and ability keys cast on release by default - holding a key shows the ability's indicator and right-click cancels - with the indicator drawn as the ability's own telegraph shape in a white aiming colour no threat uses: a projectile's lane cut short at walls, an area's circle, a single-target ability's reach, or a small ring for self-casts, and the indicator and a reticle under the cursor appear only while an ability is being aimed. F2 cycles League-style cast modes - hold-and-release (default), normal cast (press to aim, left-click to cast), and quick cast (casts on the press) - with the cast rules held in one pure function carrying unit tests for every mode. A mutation check showed that a left-click confirming a refused cast, for example on cooldown, could fall through to a swing, and the test was extended to cover it; tests that tap ability keys now press, tick, release and tick, because the release is what casts. 699 tests |
| 2026-09-22 | M7 closed | Every M7 exit criterion is met and the native playtest passed; the milestone delivered recipe models, rigid-part animation clips timed by gameplay, composed effects with ground telegraphs, bounded `Effect::Projectile` skillshots, and a test authoring a new enemy with its own model, animation and skillshot purely in RON. The plan is marked closed. |
| 2026-09-22 | M7 wave 5: skillshots | A new gameplay primitive, `Effect::Projectile { speed, range, radius, pierce, on_hit }`, is the first extension of the effect allowlist since M6. A projectile is simulation: it flies in the fixed step by a swept test, so no speed lets it skip a body or a wall between ticks, walls stop it, and its `on_hit` runs through the same executor as any ability, so armour, statuses and a dodge's invulnerable window answer it exactly as they answer a swing. It is bounded in every dimension content can set, each clamped by the executor as well as validated: speed 2-30 m/s, range up to 20 m, radius 5 cm-1 m beyond a body's own, pierce up to 4, up to 4 on-hit effects, and 32 in flight per location. `on_hit` holds only effects that act on what was struck - DealDamage, ApplyStatus, ModifyResource, EmitEvent - while Projectile, SpawnEntity and Teleport are refused so one cast can never seed another, and lowering health inside it is refused even on a self-targeted ability, because what a projectile strikes is never its caster. The bound tests were written as their own module, `game_runtime::projectile`, and each was mutation-checked; two first survived mutation and were tightened, the tunnelling test having used a body too wide to slip between ticks and a nested-effect test masked by saturated counters. Aim is the caster's facing. Presentation adds `presentation.projectile`, naming a recipe model on a new one-slot `bolt` body plan flown nose first, and `presentation.trail`, which replays an effect along the flight so trail is composition rather than a new primitive; the telegraph is a lane along the aim, as wide as it strikes and cut short where a wall will stop it. Shipped is `ember-bolt` on F, the wayfinder's third ability and the first content to use the third key, with its `ember-bolt` model, `ember-trail`, `ember-bolt-burst` and a `bolt-throw` clip; three loadout tests were updated for a three-ability loadout and the determinism run now throws a bolt. M7's first exit criterion is met by a test that parses a new enemy, its model, its cast clip, its projectile look and its skillshot purely from RON, and watches it wind up, throw and hit. That test found a real bug: an enemy could start a sword swing on the tick after it began a cast, because the brain's fallback read already casting as cannot cast, so swing, and the ashlord had been summoning and swinging at once since M6; a cast now commits the brain exactly as a swing does. Content validates at 61 definitions. 684 tests |
| 2026-09-22 | M7 wave 4: effects, ability presentation, and ground telegraphs | A new `vfx` content family, `VfxDef`, composes effects from three primitives: burst, chunky shards thrown up and out, tumbling, falling under optional gravity and shrinking away; pulse, a flat ring spreading along the ground; and beam, a bar from where the effect came from to where it plays. Each is painted from a palette swatch, bounded to 4 parts, 32 shards per burst and 180 ticks, with ranges on size, speed, gravity, radius and width. `AbilityDef` gains an optional `presentation` with four fields: `clip`, the ability's own Windows clip played instead of the body's cast clip; `cast`, an effect on the caster when the effects land; `impact`, an effect on everyone the outcomes actually changed - damaged, afflicted or healed, not everyone in reach, so a summon next to the player does not look like it hit them; and `telegraph`, a switch. Telegraph shapes are derived from gameplay, never authored, so a telegraph cannot disagree with the blow it warns of, and only area abilities may ask for one, enforced as validation error `TelegraphWithoutArea`, a deliberate deviation from the plan recorded as an ADR-0017 amendment. Enemy and player swings now get ground marks that replace the interim DebugDraw rings: the outline is the hit radius, the fill grows through the wind-up, a threat is yellow turning red when live, and the player's own swings are pale blue so they never read as a threat. Runtime `sample_game::vfx` keeps gameplay writing only requests; each effect is one entity with a `Model`, one part per piece, so no renderer change was needed, and pieces glow through the material flash overlay rather than transparency. Caps drop whole effects and count them without ever stalling - 384 live pieces, 64 queued requests - and telegraph marks are pooled; trail moved to wave 5. Shipped effects are `ember-lash`, a beam with embers and a pulse on Ember Cleave's target, `guard-shimmer` and `husk-summoning`, with clips `ember-cleave-cast` and `guard-brace`; Call Husks deliberately has no telegraph, since a summon harms no ground. A test strips every effect from content and shows an identical fight - canonical state, pools and positions - and telegraph, particle cap, per-ability clip, effect expiry and reference checks were each mutation-checked. Content validates at 55 definitions. 660 tests |
| 2026-09-22 | M7 wave 3: animation clips | Clips become content: keyframed turns, offsets and scales per slot, so one clip serves every model on a body plan that shares its slot names, and a track for a slot the body lacks is skipped. Gameplay owns time, and each clip says which clock it runs on: `Windows` clips are keyed across a committed action's wind-up, active and recovery windows, so retiming an attack in content retimes its animation and no clip can make combat faster or slower; `Span` runs across one action such as a dodge; `Loop` repeats every so many ticks; and `Stride` completes one cycle per so many metres travelled, so feet keep pace with the ground. Body plans name default clips per state and a model may override any of them — the husks and the ashlord shamble where the wayfinder walks, and keep the plan's swing. The loader refuses a clip used for a state whose clock it does not run on, since a walk keyed in attack windows would sit frozen, and a track naming a slot no body plan has, since that is a typo rather than reuse. At runtime each modelled body picks its state in priority order — dodge, swing, cast, walk, idle — samples that state's clip for every slot, and crossfades over six ticks on a change; walking is judged per simulation tick rather than per frame, because a frame on which no tick ran moved nothing and would flicker the legs between walk and idle. The body-level lean, roll and flinch from the feedback pass still play on top of the limbs. Six clips ship: idle, walk, an overhead-chop swing, a two-handed cast, a tucked dodge, and the husk shamble. The ownership rule widens to parts: animation writes part poses and never a body's translation, so no clip can move a hitbox. Four guarantees were mutation-checked — movement detection, the windows clock following the attack, animation never moving a body, and timing mismatches being refused; one check first reported a false pass because a regex was given to cargo's substring test filter and ran nothing, and was re-run correctly. Content validates at 50 definitions. 640 tests |
| 2026-09-22 | OSRS-style shapes: tapers, domes, and stretch | After the first native look at the recipe models, the kit gained what the Old School RuneScape silhouette is made of, recorded as an amendment to ADR-0017. `Taper` is a box whose top differs from its bottom — the trapezoid torso wide at the chest and narrow at the waist, the limb thinning to the wrist, the flared boot — and becomes a ridge or a spike when its top is zero. `Dome` is a faceted half-sphere for pauldrons, helmets and hair caps. `stretch` is a bounded per-axis scale applied in a part's own frame before it is turned, which makes a sphere an oval head and a dome a shoulder plate. All stay convex, so orientation is still computed rather than hand-wound; coincident corners are now merged when a face is built, so a spike emits no zero-area triangle and no NaN normal. Triangle counts for both shapes are computed in content and held equal to what the engine builds, including the degenerate spike and ridge and the largest dome the clamps allow. One of the new tests caught a wrong bound written for it — a triangle ceiling of 264 where the true maximum is 286 — and the bound was replaced by the formula. The whole cast was re-sculpted in that language: flared skirts and tabards, a cloak with a jagged hem, domed pauldrons and helms, bone spines on the husks, a bone-spiked crown on the ashlord, a peaked hood and beard on Thal. Budgets are unchanged and every model still stands on the ground. 629 tests |
| 2026-09-22 | M7 wave 2: models as content | Three new content families make models data. A body plan names joints with pivots measured from the ground, so a recipe does not depend on the size of the collision body it is worn over; a palette names sRGB swatches as integers, so it hashes and compares exactly; and a model hangs parts on those joints — each a shape from the approved set, offset and turned relative to its slot's pivot, and painted by facing from swatch names. Nothing is vertex data, which is what lets validation bound a model and an AI propose one: budgets of 24 parts and 1500 triangles, twelve slots and 32 swatches, and ranges on every dimension, offset, turn and scale are enforced intrinsically, while the loader resolves body plan, palette, slot and swatch references, all covered by the content hash. The triangle budget is computed from shape parameters in `game_core`, where no renderer exists, and a test in the game crate holds that count equal to what the engine actually builds. At startup every model is built once, the parts of each slot merged into one mesh resting on its joint, so a limb turns about its pivot rather than its middle, a sword carried in the right arm swings with it, and a character costs about six draws. `EnemyDef` and `ClassDef` gain an optional `model`; NPCs take theirs on the placement because no NPC content family exists yet, a stopgap to move when one does. The cast is modelled on one humanoid plan and one 25-colour palette — wayfinder, husk, resin-spitter, bark-warden, a 1.45-scale ashlord, Sela and Thal — while the second class stays block-out as ADR-0007 intends. Two things a cube had never needed: bodies now turn to face their facing, and cues moved to flash-overlay materials so a hit still reads on painted vertices. The fallback to the block-out shape is permanent, including for a named model that failed to build. `validate-content` had been omitting classes from its summary — the very failure its own comment warns of — and now lists every family. The content hash changes with the new families, so older saves load with a mismatch warning. Mutation-checked: standing height, scaled pivots, facing, and slot resolution. 628 tests |
| 2026-09-22 | Negative health confined to the caster; numbering notes | The decision left open when M6 closed: `ModifyResource` lowering health is allowed only with `SelfOnly` targeting, so an ability may cost its caster blood while damage to anyone else keeps one door, `DealDamage`, which is typed and answered by armour and statuses. Validation rejects any other targeting, and the executor independently lets a negative health change land only on the caster, because it never assumes the validator ran; healing others and draining stamina stay allowed. Both halves were mutation-checked. Separately, ADR-0009, ADR-0010 and the M4 and M5 plans each gained a numbering note naming what their milestone numbers now are, since two renumberings left some referring to milestones by numbers that now mean something else; the decisions themselves were not edited. 611 tests |
| 2026-09-22 | M7 wave 1: renderer groundwork | Everything wave 2 needed, and nothing visible. `MeshData` gains per-vertex linear colour, eleven floats per vertex instead of eight, with an empty list meaning white so every existing mesh renders exactly as before; materials become a tint over vertex colour and gain a flash overlay mixed in after lighting, without which a white hit-flash material multiplied into a painted body would have vanished without trace. The per-draw uniform grows from 144 to 160 bytes, and the GPU-gated tests compile the shipped shader against the new layouts on a real adapter. `engine::shapes` is the modelling kit — box, wedge, a frustum covering cylinder, prism and cone, and a faceted icosphere — flat-shaded, painted by facing, with each face's outward orientation computed from the centroid rather than hand-wound, since hand-written winding is how invisible inside-out meshes happen. `Model` is a multi-part component drawn as entity times rest times pose, one draw per part, culled per part, with a part index in the sort key so identical parts still order deterministically; there is no ECS hierarchy. One test was circular — it compared draws against the same function extraction uses, so a wrong multiplication order would have passed — and now checks the posed part's position independently. The milestone renumber itself was done by a delegated agent and verified line by line. 609 tests |
| 2026-09-22 | Presentation runtime inserted as M7 (ADR-0017) | Playtesting the feedback passes led to a larger, deliberate ask: a modular presentation layer an AI can use to build new enemies and abilities, with visible skillshots, effects, and low-poly models in a RuneScape-like style. ADR-0017 inserts it as M7, ahead of enemies — enemy variety nobody can see cannot be tuned — so former M7–M14 become M8–M15. Models are recipes of parametric flat-shaded shapes attached to the named slots of a body plan and generated at load, which an AI can compose without ever emitting vertex data; ADR-0011's glTF path stays for hand-made meshes later. Animation is rigid-part, with no skinning and no ECS hierarchy, and clips are timed as fractions of the gameplay windows so ADR-0007's windows stay the only timing authority. `MeshData` gains per-vertex colour, which also forces a flash overlay on materials: a white hurt material multiplied into a vertex-coloured body would otherwise make the hit flash vanish without a trace. Effects are opaque and capped, and skillshots are a new `Effect::Projectile` bounded in speed, range, radius, pierce, live count, and nesting. Every new content field is optional, so `CONTENT_SCHEMA_VERSION` stays 1. No code changed. 594 tests |
| 2026-09-22 | Combat feedback: cues, gauges, and motion | The first native playtest after M6 had no way to see that an attack was coming, had landed, or had been dodged — every window combat runs on was invisible. Two passes, both on existing engine seams and neither touching the renderer. Cues: a body swaps to a shared material while winding up, while its hit volume is live, when struck, and inside a roll's invulnerable window, in that priority, with `BaseMaterial` recording what to swap back to and damage noticed by comparing pools each tick rather than by hooking every source. Gauges: health, stamina, and per-ability cooldown bars, plus hit-volume, guard, and cast rings, drawn with `DebugDraw` always rather than behind F1, as an interim HUD until text rendering exists. Motion: a swing coils and lunges, a cast swells and pops, a dodge rolls, a hit squashes, and a dying enemy leaves a visual-only copy that slumps and sinks while the enemy itself still despawns the tick it dies, so no objective waits on an animation. The rule that makes animation safe was established here and is now permanent: presentation writes rotation and scale, simulation owns translation, so a lunge cannot move a hitbox — a test calls the posing code directly and asserts that no translation changed, and it was mutation-checked along with the lean direction and the remains. `Dodge` gained a read-only `direction()`. 594 tests |
| 2026-09-22 | M6 closed: melee ignored armour and statuses | Closing M6 against its exit criteria found two that held only at unit level. Enemy melee swings built the player's `Target` with neither resistances nor the damage-taken multiplier, and every enemy in the slice deals its damage by swinging — so armour could never reduce a real hit, and `status-guarded` never halved one. The wave 3 row's claim that the guard halves what the player takes "in a running game" was therefore false: its test asserted the multiplier's value, not the damage taken. Both swing paths now build their target through one helper that attaches resistances and statuses exactly as ability outcomes already did, and `step_combat` takes the fixed-step tick. Four tests close criteria that had been argued rather than shown: a guarded player takes half of a real enemy swing; armour with half physical resistance halves one, using an armour piece parsed from RON inside the test and never shipped; an ability parsed from RON inside the test, with its own cost, timing, duration, and cooldown, plays when bound to the third key; and two runs of one ability sequence end in identical state. Both damage tests were mutation-checked against the old code. Open and awaiting a decision: `ModifyResource` with a negative health amount reaches `apply_damage` directly — a second door for damage that skips resistances; the recommended fix allows negative health only on `SelfOnly` targeting, in both validator and executor. 582 tests |
| 2026-09-22 | M6 wave 4: equipping, and the key that binds | The wave that makes M6's equipment criterion true, and the one where the gap turned out to be at both ends. `selection` gains `Offer::Equipment`, `Pick::Equip`, and `slot_for` — the first mapping in the project from an `ItemCategory` onto an `EquipmentSlot`, written as a small total function rather than a `slot:` field on `ItemDef`, because a content-authored slot would let a quest item be put in a weapon hand. Weapon to Weapon, Armor to Armor, and nothing else: `Offhand` and `Trinket` stay unreachable rather than faked, since no category maps onto them and no item is authored for them, and a mapping invented for slots nothing can fill is the unreachable content ADR-0016 objects to. `take` finally raises `ItemEquipped`, which has existed, been validated, and been reduced since M4 without a single caller ever emitting it. The reducer shaped the design more than expected: it validates the item id and fills the slot, and neither requires the item to be held nor consumes it, so adding a held-check there would be a domain-event semantics change that the project workflow puts behind approval — the check lives in `take` instead, beside affordability, as `SelectionError::NotHeld`, with a test that a row drawn before a craft spent the last of something is refused afterwards. The larger find was that equipment was unwired at *both* ends: `resolve_loadout` reached only `resistances_of`, so armour would have worked the moment anything equipped it, while `start_attack` built its `Attack` from the `PLAYER_ATTACK_DAMAGE` constant and never consulted the loadout at all — equipping a weapon would have filled a slot that nothing read. Wave 2 proved the rule at unit level; wave 4 is where it reaches a swing, which is why the test asserts on the live `Attack` rather than on a husk's health: a health delta is also a test of reach, of resistances, and of whether anything was standing there, which is three ways to go green for the wrong reason. Ability keys become `ABILITY_KEYS`, a table indexed by loadout position rather than a match chain on Q and R, so a third or fourth authored ability binds with no Rust change. Equipment opens on `I` rather than at a station: it is the only offer with no fixture in the world, and authoring an armoury to stand in would be inventing a location to justify a menu — it refuses while anything else is open, so it cannot discard a conversation cursor mid-sentence. Nine tests, 569 to 578: six on the selection model and three in a running game, including the swing going from 22 to 26 and from Physical to Fire. One gap left open honestly: the key table is pinned at slots 1 and 2 and past the end of the loadout, but nothing proves a third ability binds to F, because no class authors three and adding one to `classes.ron` purely to satisfy the suite would be content written for tests rather than for the game. No schema and no save change: `format_version` and `CONTENT_SCHEMA_VERSION` both stay 1, no `DomainEvent` variant was added, and the content hash is unchanged at 15123bfb because no authored file was touched. 578 tests |
| 2026-09-18 | M6 wave 3: abilities reach the game | The wiring wave, and the first point at which any of M6 is visible to a player. `slice-abilities` registers between `slice-select` and `slice-player-combat`; `spawn_player` grants the class's authored loadout, so the wayfinder's `["ember-cleave", "warden-guard"]` come from `classes.ron` and are bound to Q and R; and enemies reach for an authored ability before falling back to a swing, so the ashlord can finally call husks — the elite's one distinguishing field stops being dead content. Two design calls are load-bearing. A cast is deliberately **not** an `Attack`: `step_combat` resolves every live `Attack` against whatever is in reach, so letting `ember-cleave` borrow one to carry its commitment window would have dealt its damage twice, once as a melee hit and once through the executor — `Casting` carries the same three windows and nothing else. And the tick that statuses expire against is `FixedTime::tick`, never `state.clock.tick`: nothing in the slice ever emits `ClockAdvanced`, so the canonical clock sits at 0 for an entire session, and measuring against it would have made every status permanent in the running game while all of wave 1's unit tests still passed — a bound defeated by its own wiring. `spawn_enemy_at` was extracted as the single place an enemy is constructed, so a summoned husk is the same creature as a placed one. ADR-0007's tie-break is preserved by having the ability system yield when a dodge is pressed that tick, rather than by relying on registration order: running first, a cast would have started and then made `start_dodge` refuse it as committed, silently inverting which action wins. Six headless tests, because the wave was written code-first and briefly stood at a green suite that proved none of it — the same shape ADR-0016 condemns: the loadout comes from content, a cast costs stamina and damages a husk, a dodge beats a cast on the same tick while a committed cast cannot be rolled out of, `status-guarded` halves what the player takes *in a running game* and lapses on schedule, a cooling ability is refused and spends nothing, and `call-husks` summons its authored pair inside the per-location cap. The ashlord test activates the ability directly rather than walking into the fight and waiting: driving the brain there depends on aggro tuning and on surviving a three-enemy elite encounter, and a death would respawn to the settlement and rebuild the location, so the enemy count afterwards would describe the wrong room. Consequently the executor-and-spawn path is pinned but the brain's *choice* to cast mid-fight is not. Still open at the end of the wave: nothing emits `ItemEquipped`, so equipment resolves correctly and is never equipped; ability keys are hardcoded to loadout indices 0 and 1, so a fourth authored ability needs a Rust change to bind; and `ABILITY_REACH` is one global constant because `AbilityDef` authors no range. No save-format change: `format_version` and `CONTENT_SCHEMA_VERSION` both stay 1, and content validates at 35 definitions. 569 tests |
| 2026-09-18 | M6 wave 2: typed damage, status effects, and equipment | `DamageType` finally leaves `game_core`. `combat` gains `Damage`, floored at construction so no path through the runtime can express healing by dealing a negative hit, and `Resistances`, a fixed `[f32; 4]` indexed by an explicit match rather than a cast, so reordering the enum cannot silently permute everyone's armour. `resolve_hit` applies the status multiplier first and resistances second, which is load-bearing rather than arbitrary: resistances clamp at `MAX_RESISTANCE` — 0.75, never immunity, because ADR-0007's combat rests on trades being real — so applying them last means no stack of buffs can push mitigation past that ceiling. The plan's one-line summary said `apply_damage` would learn a `DamageType`; it did not, deliberately, because `Health` is a dumb pool and giving it resistance lookup would make the pool own combat rules, so the typing lives in `resolve_hit` and `Resistances` instead. Statuses moved to `game_runtime::status`, where the plan assigns them, and gained the thing that makes them mean anything: since there is no `StatusDef` content family, the `StatusId`-to-behaviour mapping has to be a Rust allowlist, which sharpens ADR-0010's claim more than expected — a proposed `ApplyStatus(status: "status-godmode")` parses, validates, applies, and does **nothing**, because safety here is the absence of a lookup rather than the presence of a check, and there is a test for exactly that. Bounded three ways: duration by an absolute expiry tick, count by `MAX_ACTIVE_STATUSES` so distinct ids cannot accumulate, and magnitude by clamping the *product* rather than each factor — eight stacked halvings would otherwise reach 0.004 and be indistinguishable from immunity. Reapplying refreshes rather than stacks, and never shortens what is already held. `ItemDef` gains `#[serde(default)] combat: Option<CombatStats>` — damage, an optional retyping, and per-kind resistances in milli-units — so all eight existing items parse untouched and `CONTENT_SCHEMA_VERSION` stays 1; `resolve_loadout` takes a lookup closure rather than a `ContentRegistry`, so `combat` acquires no dependency on `content_runtime` and the rule stays testable without building a registry. Only the weapon slot may retype attacks, with a test asserting armour's `damage_type` is ignored, since armour that turned your sword into a torch would be a surprise. `ember-glaive` is the one authored item given stats. Two gaps left open deliberately: nothing emits `ItemEquipped` yet, so equipment changing a live fight is proven at unit level and not in the game, and no armour item was authored, because none is obtainable in the slice and unreachable content is the exact failure ADR-0016 criticises. No save-format change: `format_version` and `CONTENT_SCHEMA_VERSION` both stay 1. Content validates at 35 definitions. 563 tests |
| 2026-09-18 | M6 wave 1: the effect executor | `game_runtime::ability` makes the `Effect` allowlist executable for the first time since M4 defined it. ADR-0016's point was that inert primitives make ADR-0010's safety claim untestable by construction: nothing executed the allowlist, so nothing proved the allowlist bounded anything. One bounded function per variant, returning outcomes the caller applies — the same shape as `resolve_hit`, so a pure rule stays exhaustively testable without building a scene and the ECS queries stay in one place. Four bounds, each with a test written to fail loudly: `SpawnEntity` is capped per location at 24 as well as by its own `count`, and the budget accumulates across an ability's effect list so one request cannot be split in two to beat it; `Teleport` is clamped to `max_distance` and resolved by sweeping `CollisionWorld::move_capsule` rather than testing the destination, which is the whole difference between a blink and a wall-clip, and means a teleport obeys exactly the same geometry as walking rather than a second implementation that can drift; `ApplyStatus` is clamped to 3600 ticks and carries an absolute expiry tick; and area targeting is capped at 16, so an ability's worst case does not scale with how crowded a room happens to be. The status bound forced a deliberate crossing of the wave boundary: a duration nothing enforces is not a bound, it is a permanent effect with a number attached, so the minimal expiry mechanism moved forward from wave 2 — `apply`/`expire`/`is_active` against an absolute tick, where expiry is a comparison against the clock rather than a side effect of ticking, so a caller that forgets to tick cannot leave a status in force forever. Wave 2 still owns stacking policy and what a status actually modifies. Cooldowns are per entity per ability in a `BTreeMap`-keyed component rather than a global table, so two husks cool down independently; target ties break on the caller's candidate order, because distance alone stops being a total order the moment two things stand equally far away and an unstable target set would make replay diverge. Nothing is wired yet: wave 1 is contracts and skeletons as the plan specifies, so `ember-cleave`'s `Fire` now survives into an outcome but no gameplay caller exists until wave 3, and `status-guarded` has a bounded lifetime while still meaning nothing. No schema change and no new `DomainEvent` variants: `format_version` and `CONTENT_SCHEMA_VERSION` both stay 1. 544 tests |
| 2026-09-18 | Milestone plan reordered mechanics-first (ADR-0016) | Auditing M5 against §6 before starting the old M6 found combat far thinner than the plan assumed, and thinner than 450 passing tests suggest. `AbilityDef` and the whole `Effect` allowlist are parsed, validated, reference-resolved and hashed — and **never executed**: no `Effect::*` match arm exists outside `game_core` and `content_runtime`. `StatusId`, `DamageType` and `ResourceType` never leave `game_core`, so damage is an untyped `f32` and `ember-cleave`'s `Fire` is decorative; there is no cooldown code in the runtime at all. `ItemDef` carries no stats and nothing reads `PlayerState::equipment`, so `ItemEquipped` is never emitted and equipping would change nothing. The player has exactly one swing, hardcoded at `slice.rs:180`–`196`, against §6's stated requirement of **two active abilities** — which is why all three authored abilities and both classes' `abilities: []` are dead content. Enemies are the same shape: `slice.rs:836` builds `EnemyBrain::new(def.behavior)` and overrides only `damage` and `max_health`, taking aggro radii, preferred range, move speed and every attack timing from a hardcoded four-arm match at `enemy.rs:99`–`108` — so "three enemy types and one elite" is one behaviour tree with four number sets, and the elite's only distinguishing ability can never fire. ADR-0007 had already called this ordering ("should be built before enemy variety, not after") and was not followed. Building the old M6 on top would have satisfied its exit criterion only in the narrow sense that a new enemy *definition* parses; any variation with a real ability, status or damage type would still need Rust. So M5 is re-scoped to what it actually proved (the loop, closed and playable offline), M6 becomes the ability runtime, M7 becomes enemies and encounters, the former M6 becomes M8, the content pass becomes M9 — deliberately *after* M8, so the 20–30 minutes is authored against a runtime where content genuinely drives behaviour — and the former M7–M11 shift to M10–M14. §16 decision 15 amended accordingly: the demo target is unchanged in substance and lands in M9. No code changed and no schema moved: `CONTENT_SCHEMA_VERSION` stays 1, `format_version` stays 1, and M6 adds no `DomainEvent` variants, since M5 already settled that combat runs in the runtime and only its outcomes become events. The accepted cost is a demo two milestones later; the alternative was authoring the content twice. Also fixed: `docs/adr/README.md` had never listed ADR-0015. 450 tests |
| 2026-09-17 | First native Windows run: a portability bug, and two fixes that make validation measurable | The first time the suite was ever run on native Windows, five `platform::paths` tests failed: `xdg_var` filtered XDG values through `Path::is_absolute`, which answers for the platform the code was *compiled* for rather than for the `HostOs` value being passed in. In a Windows build `/home/u` reads as relative, so resolving Linux paths from a Windows host discarded every XDG value and then reported `HOME` unset. `resolve_with` takes `HostOs` as a value precisely so it can answer for either host from either host, and `is_absolute` silently broke that contract; XDG is a POSIX spec, so a leading `/` is now the test. The bug is invisible on Linux by construction — `/home/u` satisfies both checks — so **no Linux test discriminates between the old and new code, and the fix could only be confirmed on Windows — where it was, the same day: 11 passed, including all five**. Also corrected: this project has no `.git`, so `.github/workflows/ci.yml` has never run on any platform; the claim that CI validates Linux and Windows is aspirational, and CI was not masking this bug so much as never existing. Alongside it, two fixes so a native run produces numbers rather than impressions: `--gpu low-power|high-performance` and `--backend auto|vulkan` now reach `RendererConfig` through a `Launch { mode, renderer }` returned by `parse_args` (the shipped default stays `HighPerformance`, since changing it would alter what a player with a discrete card gets, and validating the minimum profile is an explicit request), refused rather than ignored on headless modes because a headless run creates no device; and `overlay-log` finally calls `overlay::rows`/`lines`/`over_budget`, which had been built and unit-tested since M3 with no caller at all — over-budget logs as a warning, and the budget is frame time only, because inventing draw-call limits here would become the numbers everyone trusts. The windowed slice had registered no overlay of any kind. 520 tests |
| 2026-09-17 | M5 wave 4: the load path | `game_core::save` had both halves since M4 and the slice had only ever wired up the writing one — `write_save` produced a `world.ron` that nothing read back — so M4's exit criterion was true of `game_core` and not of the game. `Slice::resume` is `Slice::new`'s sibling, not a variant: a new world is seeded and bootstrapped, a resumed one adopts what was recorded. It loads from the snapshot rather than replaying the log, because ADR-0013's replay contract is a property asserted by test, not a load mechanism; making the log load-bearing would be a different and worse design. No save-format change (`format_version` stays 1, no migration). `ClassChosen` is deliberately not re-applied, since rebuilding the character from today's class content is the exact failure ADR-0015 prevents. What refuses versus what degrades is decided by repairability: a vanished location has a safe recovery (wake in the settlement), a deleted item, class, or quest does not — so the reference check lives in `content_runtime`, which holds the registries, as `validate_canonical`'s own docs instruct. A hash mismatch warns rather than refuses, or saves would die with every content edit during authoring. `--resume` is explicit because CI's `--slice 240` determinism run writes a save, and auto-resume would make the second run differ from the first; it lives on the slice variants of `Mode` so "resume the M3 scene" is unrepresentable. The event log truncates on a new game so it describes exactly the snapshot's history. Found and fixed a latent bug: `enter` emits `PlayerMoved` only on a *change*, so a world that never used a door recorded no location at all and its save could not say where to put the player back — ADR-0015's shape again, the runtime knowing something durable the record did not. That broke one test informatively: it had asserted `player.location == None` as a proxy for "no move logged", and was re-pointed at the event log itself, with a non-empty check so it cannot pass vacuously. The escort is *not* re-derived on load: an open `Escort` objective and an escort in progress are not the same fact, and re-deriving would have an NPC appear unasked. 519 tests |
| 2026-09-17 | M5 wave 4: combat exit as a safe point | The fourth safe point for health. Combat is the one durable thing that happens entirely in the runtime, and its result was written only on a transition, on death, or before a save — so a player who fought, won, and stood still kept a stale record until they used a door, died, or quit. "In combat" is defined as enemy *awareness* rather than the player's own swinging: a swing at empty air is not a fight, and `EnemyBrain`'s aggro is already hysteretic, so the edge does not chatter. The *edge* is what fires — syncing every tick out of combat rewrites an unchanged value, and syncing every tick in combat is the running commentary `sync_health` exists to avoid — so one unserialized bool on `Slice` notices the transition. Both endings (the last aware enemy dies, or the player leaves) record the same thing, and the call is a no-op after a respawn because `respawn_if_dead` has already synced. The test was mutation-checked: with the call disabled it fails, which is the only way to know an edge-detection test is not vacuous. Also found: the slice has **no load path** — `save::load` is complete and uncalled, so nothing survives a save yet and "the escort does not survive a save" is a consequence of that rather than its own bug. 512 tests |
| 2026-09-17 | M5 wave 4: choosing | `game_runtime::selection` replaces the three "take the first thing on offer" stand-ins — `talk_to`'s option 0, the forge's first affordable recipe, the board's first affordable blueprint — with one offered-rows model driven by digit keys. It is a model, not a widget: nothing draws, so all of selection is testable headless, and it is the code a real menu will call. Rows are recomputed rather than stored, because a stored list is a second source of truth the world can invalidate between the tick a menu is drawn and the tick a key lands; a `Pick` names a thing (`RecipeDef::id` is an `ItemId` — there is no `RecipeId`) rather than a row number, so a list that changed shape cannot redirect a choice; and `take` re-runs the rule that offered the row, so a stale row is refused rather than executed — a third *check* of one rule, with `interaction` still owning affordability and the reducer still the authority. `E` is modal because `slice-save-on-exit` owns Escape; stations stay open so two things can be made without a round trip. Splitting open-from-choose reintroduced the wave 4 commit-ordering bug by a different door: `Talked` was raised on opening, a tick before the conversation's own `QuestAccepted`, so `advance_quests` drained it against a still-inactive quest and the player had to talk to Sela twice — it now fires when the conversation ends. Capability gained: a player can take a line other than the first, so Thal's authored "Then I'll go around." is sayable at last. 510 tests |
| 2026-09-17 | M5 wave 4: the old forest's picket | `bark-warden` and the `old-forest-picket` encounter were authored and placed nowhere; two wardens now hold the approach to the crypt, so the whole enemy roster is reachable. Chargers rather than husks, so the road to the crypt is a different fight rather than more of the ambush. The lane now holds two independent encounters, which immediately exposed that the test helper cleared *every* tagged enemy in a location — harmless with one fight per place, wrong with two. It now takes an encounter id, and assertions that tallied "enemies in the lane" count the ambush's four specifically. 493 tests |
| 2026-09-17 | M5 wave 4: escorting | `still-waters`' second branch is playable, so no authored path through the main quest is unreachable. The escort lives on `Slice` as runtime state rather than on an entity, because a transition despawns every NPC and rebuilds from content — a follower marked only on an entity would vanish through the first door, Thal being authored in the lane and not the hub; `travel_to` re-spawns whoever is in tow. Starting it is inferred from the open `Escort` objective rather than a new dialogue effect, since the quest already states who and where. Following is straight-line with collision: the route is open ground, and `EnemyBrain`'s motion keys off combat states. The destination moved to the settlement, where terms are actually made. Not serialized, so a save mid-escort reloads with nobody following. 493 tests |
| 2026-09-17 | M5 wave 4: a second quest | `ember-and-ash` gives the crypt a reason to exist — the `ashlord-stand` encounter was authored, placed, and referenced by no quest, so the one elite was a fight nobody was sent to. Linear by design, since `still-waters` already covers branching. Quests cannot chain in the schema (`QuestDef` has no prerequisite field), so the gate is a dialogue option conditioned on `QuestCompleted`; authored option order is load-bearing, because the first-available stand-in would skip a gated offer placed below an ungated one. The loader now resolves `Reach` and `Escort` destinations against the location registry — a check that immediately found `still-waters` escorting Thal to `old-forest-verge`, a location that never existed and had gone unnoticed because the branch is unreachable. 491 tests |
| 2026-09-17 | M5 wave 4: the slice becomes visible | Until now nothing the slice spawned had a `MeshRenderer` and `attach_slice` was reachable only headlessly, so 486 tests described a game nobody had seen. `SliceVisuals` uploads blocked-out primitives once at startup — the only place a `RenderAssets` handle exists, and not in `populate`, which re-runs per transition and would leak a mesh per visit — and every spawn kind is now dressed, with the elite distinguished by `variant_of` and the ground tagged `Scenery` so transitions replace it. `build_slice_app` runs the slice without M3's scene, which also stops "something was drawn" tests passing on M3's prop ring alone; the slice gained its own camera system, since `build_app`'s tracks a different `Player` type entirely. `--slice` now opens a window and saves on Escape (`engine::run` consumes the app); `--slice N` stays headless for CI. Hub 6 entities, lane 13, crypt 7, all dressed. Not verified visually: WSL is not a renderer target. 488 tests |
| 2026-09-17 | M5 wave 4: the settlement loop closes | The upgrade board now builds the first affordable blueprint, the same stand-in the forge and `talk_to` use. A test walks the whole loop: the glaive's materials buy nothing, the board raises the forge, and the same materials at the same forge then make the glaive — gather, build, unlock, craft. A build signals nothing, because a blueprint's effect is the world flag it grants, which recipes and dialogue already gate on. What remains a placeholder is selection rather than capability: three places take the first thing on offer, and one list UI retires all of them. 486 tests |
| 2026-09-17 | M5 wave 4: stations and crafting | `SpawnRole::Station` gained an authored `StationKind` (`Doorway`, `Crafting`, `Upgrades`) with no serde default, replacing "a door is a station a link happens to start at"; the loader now requires a link's `from` to be a `Doorway`, the stronger form of the check that caught both missing exits. A forge was added to the hub and crafts the first affordable recipe in id order — the stand-in `talk_to` already uses for dialogue — raising a `Collected` signal so crafted items count toward objectives. The upgrade board is declared but inert, since building a blueprint without a UI would mean choosing one for the player; a test pins that. Consequence: only `clearwater-draught` is reachable, as both other recipes need blueprints nothing can build. 484 tests |
| 2026-09-17 | M5 wave 4: classes and health at safe points | Implementing health synchronisation exposed that a fresh world recorded the player as `0/0` in both pools — `PlayerState` derives `Default`, `Pool` has none, and nothing had noticed because the runtime invented `Health::full(100.0)` instead of reading the record. ADR-0015 gives pools an authored origin: classes become a content family (eleventh), and `ClassChosen` carries the resolved pools so `apply` stays a pure function of canonical state and a replayed log reproduces the character that existed even if the class was later retuned. `spawn_player` now derives from canonical state, and `sync_health` reconciles at safe points — transition, death (before and after restoring), and the exit save — measuring against what the tick will commit rather than what is already committed, so nested safe points cannot double-count. Combat exit is not yet a safe point. 480 tests |
| 2026-09-17 | M5 wave 4: death and respawn | ADR-0007's death rule implemented: the dead wake up in the settlement — found by `LocationKind::Settlement` rather than a constant — with health and stamina restored and nothing lost. Respawn reuses the extracted `travel_to`, so the fight the player died in is standing again when they return, falling out of rebuilding a location from content rather than being special-cased. Death emits no event of its own; the transition's `PlayerMoved` records where they woke up, and re-entering the location you are already in logs nothing. Canonical health is still not written at safe points, so a save taken mid-fight records full health. 472 tests |
| 2026-09-17 | M5 wave 4: the quest runs end to end | `still-waters` now completes down the combat branch, driven only by interactions and combat, recording its `waters-restored` resolution and paying its rewards. Writing that test found three defects narrower tests had missed: neither the lane nor the crypt had a usable exit, because both return links named a `PlayerStart` rather than a station (the loader now rejects that); accepting a quest discarded the interaction that accepted it, because `advance_quests` ran before `commit` and drained the tick's signals against a still-inactive quest (now commit → advance → commit); and a dead player keeps acting, since ADR-0007's respawn is unimplemented. `reap_dead` was extracted from `step_combat` and now scans health rather than the player's hit results, so an enemy killed by any future damage source is still looted and counted. 468 tests |
| 2026-09-17 | M5 wave 4: clues and choices | The last two objective kinds nothing could satisfy. `SpawnRole::Clue` makes examining spatial; `SpawnRole::Choice` and `DialogueEffect::MakeChoice` let a decision be settled in the world or in conversation, whichever comes first. Neither adds a domain event — the quest objective each completes is already the durable record. `Conversation::choose` now returns a `Turn` carrying events and settled choices separately, because events go through the reducer while choices are signals a quest consumes that tick. Arbiter Thal, who had a conversation and an escort objective but stood in no location, now stands in the lane. `Escort` remains unimplemented, so the quest has two authored branches and one playable. 467 tests |
| 2026-09-17 | M5 wave 4: travel and combat | Two `#[serde(default)]` schema fields closed the gap between content that described fights and a runtime that could resolve them: `EnemyDef.loot_table` (a property of the enemy) and `SpawnRole::Enemy.encounter` (a property of the placement), both resolved by the loader like every other cross-family reference. Location transitions rebuild the far side from content and land the player on the door's `arrive_at`, recording `PlayerMoved` and a one-time region discovery. Combat is bound to input: committed attacks and dodges gated on stamina, i-frames, enemies that face and swing back, loot rolled from the world seed on death, and an encounter cleared when its last member falls. The lane now contains `waterworks-ambush` as authored. 461 tests |
| 2026-09-17 | M5 wave 4 (part): the slice runs | Authored content moved from `tests/fixtures/content` to `assets/content` (content hash unchanged, confirming ADR-0014 hashes meaning rather than files); `sample_game` gained a `--slice` mode that loads content through `EnginePaths`, projects `LocationDef` spawns into entities with real transforms, and runs movement, navmesh-pathing enemies, interaction, dialogue, and quest advancement in the fixed step, committing domain events once per tick and writing a save on exit. Eleven headless integration tests cover population, placement, aggro, harvesting, quest acceptance through dialogue, determinism, and that no runtime handle reaches canonical state. Location transitions and input-driven combat remain. 455 tests |
| 2026-09-17 | M5 wave 3: quest, dialogue, and interaction runtimes plus debug tooling | `narrative` and `game_runtime::interaction` implemented as pure functions over content and canonical state, returning domain events the caller applies: quest graphs with computed open objectives, delta-counted progress, and recorded resolutions; conversations with allowlisted conditions, filter-stable option indices, and effects that become events; gathering, crafting, blueprint upgrades gated on their granted world flag, and loot rolled from a seed mixing the world seed, the in-world tick, and the table id. Nothing here mutates the world, so every outcome is replayable from the log alone. 424 tests |
| 2026-09-16 | M5 wave 2: movement, combat, enemy behaviour | Swept capsule collision with sliding (a 40 m step cannot tunnel a 1 m wall), stamina and dodge i-frames, a tick-counted attack state machine where one swing connects at most once, and enemy behaviour templates with aggro hysteresis, and navmesh navigation (A* over polygon neighbours with index tie-breaks, funnel smoothing, and path following with arrival tolerance and re-path hysteresis) wired into enemy steering. 378 tests |
| 2026-09-16 | M5 wave 1: spatial and narrative content | `LocationDef` with authored nav polygons and spawn points, `DialogueDef` with allowlisted conditions and effects, `LootTableDef` with seeded rolls, `SettlementUpgradeDef`; five new id types; `content.rs` split into a directory module; loader extended to ten families with cross-family resolution and hashing; Cinderwake fixtures validate at 32 definitions. 318 tests |
| 2026-09-16 | Content hashing decided: BLAKE3 | Chosen for speed, the deciding factor for a hash computed on every load; hashes the compiled registry in sorted-id order rather than raw file bytes, so formatting changes do not invalidate saves. `blake3` pinned and confined to `content_runtime`; see ADR-0014 |
| 2026-09-16 | M4 domain and content foundation implemented | Typed ids with instance scoping, content schemas and Cinderwake fixtures, canonical world state, domain events with pure reducers, versioned save envelope with migration chain, line-delimited event log, content loader with a `validate-content` binary wired into CI, domain-to-ECS projection, and a mock AI provider. 290 tests |
| 2026-09-16 | Shader assets excluded from the kebab-case filename rule | Shaders are code referenced from Rust by constant and `include_str!`, not content addressed by stable id; recorded in Section 5.6 and ADR-0011 |
| 2026-09-16 | M3 renderer implemented; native validation still outstanding | Mesh upload, materials, pipeline, depth, fixed-angle camera, ECS extraction with frustum culling, debug lines, shader hot reload. Built as six parallel tasks against fixed contracts; 175 tests. Exit criteria still require native Windows and Linux GPU runs |
| 2026-09-15 | M1 and M2 implemented | Window/input/paths/timing/diagnostics and a minimal ECS with headless determinism tests; renderer clears and presents |
| 2026-09-15 | Engine-from-scratch addendum merged into this document | Custom engine replaces Bevy throughout; `bevy_adapter` replaced by `platform`/`renderer`/`engine`/`game_runtime`; added engine architecture, ECS, renderer, and dependency sections; milestones re-sequenced to platform → ECS → renderer → domain/content → slice; native Windows and Linux are first-class targets |
