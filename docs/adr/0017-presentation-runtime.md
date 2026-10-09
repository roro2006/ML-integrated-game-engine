# ADR-0017: Presentation runtime — recipe models, rigid-part animation, VFX, and skillshots

- Status: Accepted; §3 rigid-parts-only rule superseded by ADR-0020 (skeletal animation in scope)
- Date: 2026-09-22

## Context

M6 finished the ability runtime and then, in playtesting, showed that a runtime
nobody can see cannot be judged. The first native session after M6 closed had no
way to tell that an attack was coming, had landed, or had been dodged. Two
feedback passes followed — material-swap cues plus `DebugDraw` gauges, then
procedural poses on transform rotation and scale — and made combat legible on
blocked-out cubes.

The next ask is larger and deliberate: a **modular** presentation layer that an
AI can use to create new enemies and abilities, with visible skillshots,
effects, and models in a low-poly, RuneScape-like style.

Four existing commitments bound the answer:

- `PROJECT_CONTEXT.md` §11 already sets the art direction — low-poly, readable
  silhouettes, variation from "palettes, silhouettes, materials, decals,
  animation timing" rather than dense geometry.
- §4 defers "complex skeletal animation unless the MVP demands it" and "runtime
  generative media pipelines".
- ADR-0010 and §7 already let AI propose "visual recipes referencing known
  assets, tags, materials, and VFX" — composition from approved libraries,
  never code, never unvalidated references.
- ADR-0011 already names the hand-made model path: Blender → glTF → an offline
  bake to an engine format, with the shipped runtime loading only baked meshes.
  It also notes that code-generated primitives are trusted in a way imported
  meshes are not.

The renderer today draws one mesh and one flat-colour material per entity.
`MeshData` interleaves eight floats per vertex (position, normal, uv); there is
no vertex colour, no ECS hierarchy, no transparency, and no particle path.

## Decision

### 1. A new M7, "Presentation runtime", ahead of enemies

The milestone is inserted before "Enemies, encounters, and combat variety".
Former M7–M14 become M8–M15. The reasoning is the lesson of the feedback passes:
enemy variety that cannot be seen cannot be tuned, and M8's exit criterion ("a
new enemy that plays differently is a content file") is only checkable by a
player once enemies look different too. This work also carries a different risk
— renderer and asset format rather than gameplay — and is safer isolated.

This does not reverse ADR-0016. Mechanics came first and are done; this is the
layer that makes them perceivable, before more mechanics are added on top.

### 2. Models are recipes of parametric shapes (`ModelDef`)

A model is a list of **parts**. Each part is a shape from a fixed library — box,
wedge, prism, cylinder, cone, low-poly sphere — with dimensions, an offset and
rotation, and a palette colour per face group, attached to a named **slot** from
a body plan (e.g. humanoid: `torso`, `head`, `arm_l`, `arm_r`, `leg_l`, `leg_r`,
`weapon`). Meshes are generated from the recipe when content loads, flat-shaded,
deterministically.

> **Amended 2026-09-22.** After the first native look at the recipe models,
> two shapes and one part property were added to reach the OSRS silhouette
> language — trapezoid torsos, limbs that taper, flared boots, domed pauldrons
> and helmets: **`Taper`**, a box whose top differs from its bottom (a zero top
> is a ridge or a spike), **`Dome`**, a faceted half-sphere, and **`stretch`**,
> a bounded per-axis scale applied before a part is turned. All stay convex, so
> computed orientation still holds; coincident corners are merged so a spike
> emits no zero-area triangle. Budgets are unchanged.

This is the primary model source for M7. It is AI-composable by construction:
an AI picks shapes, sizes, colours and slots, and never emits vertex data. It
needs no new dependency. ADR-0011's glTF path is unchanged and remains the route
for hand-made meshes later, as an additional part source referenced by id.

### 3. Rigid-part animation, no skinning

Parts move as rigid pieces. There is no skinning, no bone weights and no
skeletal retargeting, which keeps inside the §4 deferral. There is also no ECS
hierarchy: a `Model` component holds its parts, and render extraction composes
each part's matrix as `entity × part rest × part pose`, emitting one draw per
part. Parent/child entities are not needed for this and are not introduced.

### 4. Per-vertex colour, and cues as a flash overlay

`MeshData` gains a per-vertex linear RGB colour; the interleaved vertex grows
from 8 to 11 floats. A face group's palette colour lives in the mesh, which is
what the RuneScape look actually is and costs fewer draws than a material per
colour. Every existing mesh defaults to white, so the current slice renders
unchanged.

Materials become a **tint** multiplied with vertex colour. That breaks the
existing cue mechanism on its first real model: a white "hurt" material
multiplied into a vertex-coloured body changes nothing, so the hit flash would
vanish silently. Materials therefore also carry a **flash overlay** — a colour
and an amount the lit result is mixed toward — and cue materials use it. The
material-swap mechanism itself is kept.

This amends ADR-0011's mesh format. The baked glTF format, when it lands, must
carry vertex colour too.

### 5. Animation clips are data, and gameplay drives them

A `ClipDef` holds keyframed tracks per slot — rotation, offset, scale. Clips
for committed actions are timed as **fractions of the gameplay windows** (wind-up,
active, recovery), not in their own seconds, so a clip cannot change how long an
attack takes; ADR-0007's windows remain the only timing authority. Looping clips
(idle, walk) run on their own period. A model's animation set maps states —
idle, walk, wind-up, strike, recovery, cast, dodge, hurt, death — to clips; an
ability may name its own. Clips address slots by name, so one clip serves every
model with those slots, and a missing slot is skipped rather than an error.

The ownership rule established by the procedural-pose pass becomes permanent:
**simulation owns an entity's translation; presentation owns its rotation,
scale, and every part transform.** Nothing a clip does can move a hitbox, and a
test holds that line for bodies, parts and effects alike.

### 6. Effects are composed from a small allowlist (`VfxDef`)

An effect composes primitives: burst (chunky low-poly shards), trail, ground
telegraph (circle, cone, line), pulse, and beam. Particles are **opaque** —
which suits the style and needs no transparency pipeline — and presentation-only:
nothing in the simulation reads them. Each effect has a particle cap and a
lifetime cap, with a global cap over all live effects, so an effect's worst case
is bounded the way ADR-0010 requires of everything content can express.

> **Amended 2026-09-22 (wave 4).** Three choices made while building this:
> - **Telegraphs are derived, not authored.** `AbilityDef.presentation.telegraph`
>   is a switch; the shape comes from the ability's targeting and the runtime's
>   reach, and a melee swing's from its live `Attack`. A telegraph that content
>   could shape could disagree with the blow it warns of. Only area targeting
>   may ask for one, since a single target is chosen when the ability lands.
> - **The wave 4 set is burst, pulse and beam.** Trail needs something that
>   moves, and moves to wave 5 with the projectile.
> - **Effects are models.** Each effect is one entity with a `Model`, one part
>   per piece, so effects reuse the model draw path and need no renderer
>   change. Pieces glow through the flash overlay (§4) rather than
>   transparency. Effect entities are not pooled — profiling first — but
>   telegraph marks are, since they appear with every swing.

### 7. Skillshots are a new bounded `Effect::Projectile`

A projectile is gameplay, not presentation: it lives in the fixed step, moves by
swept collision (the same approach that stops `Teleport` clipping walls), and
applies `on_hit` effects through the existing executor. It extends §7's trusted
primitive set, "deliberately, with tests", and is bounded in every dimension
content can set:

| Bound | Why |
|---|---|
| Speed cap | A projectile faster than a tick's sweep can resolve is a hitscan in disguise |
| Range cap, as a lifetime | Nothing lives forever |
| Radius cap | An enormous projectile is an area effect that ignores `MAX_TARGETS` |
| Pierce cap | Bounds how many targets one cast can reach |
| Live cap per location | Bounds the worst case of rapid casting |
| `on_hit` depth of one | `on_hit` may not contain `Projectile` or `SpawnEntity`, so a projectile cannot seed another |

> **Amended 2026-09-22 (wave 5).** As built:
> - **The bounds, as numbers:** speed 2–30 m/s, range up to 20 m, radius
>   5 cm–1 m (beyond a body's own), pierce up to 4, `on_hit` up to 4 effects,
>   32 in flight per location. Each is clamped by the executor as well as
>   validated, and each has a test that fails when its clamp is removed.
> - **`on_hit` holds only what acts on the struck:** `DealDamage`,
>   `ApplyStatus`, `ModifyResource` and `EmitEvent`. `Teleport` is excluded
>   alongside `Projectile` and `SpawnEntity`, since a distant hit has no
>   business moving the caster; lowering health inside it is refused even on
>   a self-targeted ability, because what a projectile strikes is never its
>   caster.
> - **Aim is the caster's facing** (or an aimed point, when one exists); an
>   ability's targeting names whom its *other* effects touch. A projectile
>   respects a dodge's invulnerable window exactly as a swing does.
> - **Its look is a model and a trail.** `presentation.projectile` names a
>   recipe model (on a one-slot `bolt` body plan), flown nose first;
>   `presentation.trail` names an effect replayed along the flight. Trail is
>   therefore composition, not a new effect primitive.
> - **Its telegraph is a lane** along the aim, as wide as it strikes and cut
>   short where a wall will stop it.
> - The simulation lives in `game_runtime::projectile`, beside rather than
>   inside `ability`.

> **Amended 2026-09-22 (before M8 closes).** Three additions:
> - **Clips curve.** Keys are joined by a monotone cubic (Fritsch–Carlson),
>   per component, on the keys' real times: motion carries through a key
>   instead of changing speed on it, a held pose stays held, and nothing
>   overshoots a key. Linear interpolation read as a string of poses.
> - **Two more effect primitives:** `Slash` — an arc around where an effect
>   came from, sweeping in order across a blow's path — and `Crack` — fissures
>   racing out along the ground. Enemies and classes gain an optional
>   `presentation` with a `swing` effect (played the tick a blow goes live)
>   and a `hit` effect (where it connects), so a plain swing can look like
>   one; the ashlord's slam splits the ground and throws rock.
> - **Text and a menu.** `engine::pixel_text` is an engine-owned 5×7 pixel
>   font drawn as meshes — no font dependency. Names over heads are one mesh
>   per name, built at startup; the on-screen menu for conversations and
>   stations is one pre-built mesh per glyph, placed per character, on a
>   panel a metre in front of the camera and square to its view, so screen
>   layout and click testing share one function. There is still no 2D pass
>   and no runtime mesh upload; if menus grow, one of those is the next
>   decision.

### 8. Links from gameplay content, all additive

`EnemyDef` gains an optional `model`; classes gain one for the player;
`AbilityDef` gains an optional `presentation` — cast clip, cast effect,
projectile look, impact effect, telegraph shape. Every new field is
`#[serde(default)]`, absent means the current fallback, and
`CONTENT_SCHEMA_VERSION` stays 1. No save-format change: presentation state is
runtime-only, and a projectile in flight is not saved, like an attack mid-swing.

### 9. What this means for AI

Every family here is schema-constrained RON, reference-validated, budgeted, and
content-hashed. When M13 (formerly M12) adds AI proposals, they target these
schemas and nothing new is required. The "approved libraries" of ADR-0010 are,
concretely: the shape set, the body plans, the palettes, the clip library, and
the VFX primitives. AI never supplies vertex data, shader code, or executable
behaviour.

## Consequences

- Five new or extended content families: models, body plans, palettes, clips,
  effects — plus `Effect::Projectile` and optional presentation fields.
- A renderer vertex-format change and a new per-material flash overlay; the
  shader and pipeline change together, with tests on the interleaved layout.
- Draw count grows with part count. A ten-part enemy is ten draws; the budget
  work in §11 applies, and part-count caps are part of model validation.
- Hand-made art is not blocked, only deferred: glTF parts slot into the same
  `Model` component once the bake step exists.
- Transparency, bloom, textures and skinning are out of scope for M7 and would
  each need their own decision.

## Alternatives considered

- **One flat material per part, no vertex colour.** No renderer change, but a
  part cannot carry two colours — no painted face, no trim, no coloured blade —
  and draw calls grow with colour count. Rejected on the look it cannot reach.
- **Recipes plus glTF import now.** More artistic freedom, but a new parsing
  dependency and the bake step are a second project, and AI cannot author those
  meshes. Deferred, not rejected.
- **Skeletal skinning.** The deferral in §4 stands; rigid parts reach the
  intended style at a fraction of the cost.
- **Fold into the existing M7, or run after M8.** Folding doubles M7 and mixes
  gameplay tuning with renderer risk; deferring means two more milestones of
  playtesting cubes. Rejected in favour of a dedicated milestone.

## Relation to other ADRs

- Amends **ADR-0011**: `MeshData` carries vertex colour; recipes are a second,
  code-generated model source beside the glTF bake.
- Extends **ADR-0010**'s allowlist with the visual libraries above and
  `Effect::Projectile`, each bounded.
- Follows **ADR-0016**: milestones shift once more, M7–M14 becoming M8–M15.
- Preserves **ADR-0007**: gameplay windows remain the only source of timing.
