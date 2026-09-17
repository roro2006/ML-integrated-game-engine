# ADR-0009: Collision and navigation

- Status: Accepted
- Date: 2026-09-15

> **Numbering note (2026-09-22).** Milestone numbers below predate two renumberings
> (ADR-0016, ADR-0017). The "M7" this ADR refers to — procedural regions, whose
> reachability and spawn-safety validation runs on the navmesh — is now **M11**.
> The current list is `PROJECT_CONTEXT.md` §13. The decision itself is unchanged.

## Context

Open decisions §16.5 and §16.6. The slice needs character movement that cannot
pass through walls, melee and ability hit detection, and enemies that path
around obstacles in both an authored hub and generated dungeons. Navigation
also underpins M7's reachability and spawn-safety validation.

## Decision

**Collision: minimal and internal.** A capsule for characters, swept against
static level geometry (triangles and AABBs), resolved in the fixed-step
simulation. Abilities use sphere, capsule, and cone overlap queries. No rigid
bodies, joints, stacking, or continuous dynamics.

**Navigation: navmesh.** Enemies and NPCs path over a navmesh rather than a
grid:

- The navmesh is generated from level geometry — authored for the hub, and
  produced deterministically from the seed when a dungeon instance is created.
- Pathfinding is A\* over navmesh polygons, followed by funnel/string-pulling
  to produce direct paths.
- Agent radius and step height are baked into the mesh at generation time.
- The navmesh is transient runtime/derived data: it is rebuilt from the seed
  and geometry, never persisted in canonical state.

M7 validators run **on the navmesh**: every quest objective, entrance, and
spawn point must lie on a connected navmesh region reachable from the legal
entrance, and spawn positions must be far enough from mesh edges.

## Consequences

- Navmesh generation is the largest single piece of work implied by these
  decisions, and it must be deterministic, or generated dungeons stop being
  reproducible from a seed. It needs its own tests before M7 depends on it.
- Instance creation gains an ordering constraint: geometry → navmesh →
  reachability validation → spawn placement. A dungeon cannot be declared valid
  before its navmesh exists.
- Navigation quality on organic, non-grid wilderness is much better than a grid
  would give, and agents move naturally without grid-stair artefacts.
- Internal collision keeps the fixed-step simulation fully deterministic and
  debuggable, with no external physics determinism caveats.
- Anything needing real physics later (ragdolls, thrown objects, destructible
  props) is out of scope and would need a new decision.

## Alternatives considered

- **Grid + A\***: far less machinery, trivially deterministic, and it would
  have made reachability validation nearly free — but it constrains level
  geometry to a grid and looks it, especially outdoors.
- **External physics (Rapier/Parry)**: more capability than the slice needs,
  a large dependency, and determinism requires care.
- **Recast-style external navmesh generator**: proven and would save
  significant work; revisit if in-house navmesh generation proves to be a time
  sink, since it is contained behind the navigation interface either way.
