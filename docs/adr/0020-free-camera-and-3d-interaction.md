# ADR-0020: Free third-person camera and 3D interaction

- Status: Accepted
- Date: 2026-10-09
- Supersedes: ADR-0007's fixed-angle 2.5D camera decision, and the 2026-09-22 mouse-aim amendment to ADR-0007's combat decision (cursor projected onto the ground, and the 2D planar hit test).
- Amends: ADR-0009 (collision scope: terrain and vertical motion).

## Context

ADR-0007 chose a fixed-angle 2.5D camera. Its stated reasons were that models and animations then read from one angle, that a solo schedule could afford that, and that culling was cheap. The owner approved the master plan on 2026-10-09, which moves the game to a free-form third-person survival and crafting sandbox with building, gathering, skill progression, and combat as one system among several. The master plan (§1 and workstream F1, F2, F7) asks for a free orbit camera, camera-relative movement, look-at interaction, and 3D hit volumes.

Current code aims by projecting the cursor onto the ground (`crates/sample_game/src/aim.rs`). Hit tests are a ground-plane circle with a vertical span (`crates/game_runtime/src/combat.rs`). Collision is a capsule against static boxes and the y = 0 plane. None of this works once the camera can orbit. The fixed-angle reason also falls away, because all-angle asset quality is now budgeted (ADR-0019) rather than avoided.

## Decision

1. **Free third-person orbit camera.** Yaw is unbounded. Pitch is clamped to a maximum and a minimum. Distance has a minimum and a maximum. A signed shoulder offset moves the focus to either side. First-person is a second configuration of the same camera type, and it is optional and later.

2. **Camera collision.** The camera sweeps from the focus point toward the desired eye using the same swept-capsule test as movement. When it is blocked, the eye moves to the contact point minus a skin width. Smoothing is unchanged. The camera is presentation: it is not canonical state and it does not enter the log.

3. **Camera-relative movement and aim.** Movement input is relative to camera yaw. The aim point is the camera-centre ray, hitting the nearest collider or the ground plane when nothing is hit. The body faces its walking direction until it acts. An action turns the body toward the aim point when it starts and keeps turning through wind-up only. It locks when the blow goes live or the effect lands. This is ADR-0007's 2026-09-22 facing rule, with the camera ray in place of the cursor projection.

4. **Cast state machine kept.** Abilities keep ADR-0007's cast modes: press to aim, release to cast, right-click cancels, and the cycle key switches normal and quick cast. The keys are the five ability keys Q R F C V from ADR-0018. The key table moves to an input map, so bindings are settings and not content.

5. **Interaction by ray.** Each fixed tick, the camera-centre ray selects the nearest interactable within reach. Selection is by stable target id, with ties broken by id order, so it is deterministic. Context verbs such as talk, use station and open door return domain events and check affordability before raising them.

6. **3D hit shapes.** Abilities hit with a sphere, capsule, cone (melee arc) or ray, tested against the target capsule. The planar `hit_connects` is retired once every ability uses a shape. A live hit volume never steers, and a committed attack cannot be cancelled or retimed. ADR-0007's windows, stamina gating and dodge invulnerability are unchanged, and they stay in fixed ticks.

7. **3D character controller.** A kinematic capsule runs in the fixed step, with gravity, jump, step height, slope limit and ground detection against static geometry, including terrain heightfields. Climb and swim are volume-triggered states. The limits come from content, bounded by the loader.

## Consequences

- ADR-0009's collision scope grows. It was static triangles and AABBs with a ground plane. It now includes terrain heightfields and vertical motion. The rule that still holds is no rigid bodies, no stacking, and no continuous dynamics. Anything beyond that needs a new decision.
- ADR-0009's navmesh is still 2D per location. Extending pathing to 3D is a separate decision under workstream F8, and this ADR does not make it.
- Skeletal animation is in scope (owner decision, 2026-10-09). It supersedes ADR-0017 §3 (rigid parts, no skinning) and the PROJECT_CONTEXT §5.2 deferral. Scope: glTF skeleton and clip import through the asset cooker (B6), GPU skinning (vertex or compute), a data-driven animation state machine with blend spaces, root motion opt-in per clip, and animation events driving gameplay timing (attack windows, gather hits). IK (foot placement, look-at) follows after the base system. Bone budgets come from `docs/reference/observed-asset-budgets.md` (player about 43 bones, NPC median 106, max observed 336), so the skinning path supports at least 512 bones per mesh. Gameplay workstream F8 owns the work; ADR-0017's clip and presentation boundary rules stay in force for everything else.
- Every asset must now read from every angle. The budget work in ADR-0019 carries that cost.
- The cast-mode tests in `aim.rs` survive. The ground-projection path and its tests are retired.
- Camera collision needs a property test: over random camera targets, the eye is never inside a static box.

## Alternatives considered

- **Keep the fixed-angle camera and add 3D hits.** It preserves the single-angle benefit, but it fails the owner's game-shape decision and would be rebuilt later anyway.
- **First-person only.** It is simpler to animate, but it is the wrong genre fit for a building and crafting sandbox.
- **Free camera with cursor aim.** It keeps the mouse projection, but the cursor is not a usable aim point once the camera can look up or orbit. Rejected in favour of the camera-centre ray.
- **Rigid-body physics for the controller.** It is more capable and would bring the determinism caveats ADR-0009 avoided. Rejected.
