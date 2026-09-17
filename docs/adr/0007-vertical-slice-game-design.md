# ADR-0007: Vertical-slice game design decisions

- Status: Accepted
- Date: 2026-09-15

## Context

`PROJECT_CONTEXT.md` §16 listed fifteen open product decisions. Several of them
(camera, combat feel, topology, progression, death, settlement scope) directly
determine engine work in M3–M6, so they were resolved together.

## Decision

- **Camera**: fixed-angle 2.5D. A perspective camera at a fixed pitch and yaw
  follows the player; the player cannot free-rotate it.
- **Combat**: Souls-lite and deliberate. Stamina-gated, committed attacks with
  recovery windows and dodge invincibility frames.

  > **Amended 2026-09-22: mouse aim.** Swings and abilities aim at the cursor,
  > projected onto the ground (`sample_game::aim`). The body faces where it
  > walks until it acts; an action turns it to the aim the tick it starts and
  > keeps turning through the **wind-up only**, locking the tick the blow is
  > live or the effects land. Commitment is kept where it matters — once
  > begun, an action cannot be cancelled or retimed, a live hit volume never
  > steers, and walking never turns a committed body — while the wind-up
  > becomes the window in which a skillshot is aimed. An ability key casts on
  > **release**: held, it shows the ability's indicator and a reticle under
  > the cursor, which appear only while aiming; right-click cancels. F2 cycles
  > to normal cast (press to aim, left-click to cast) or quick cast (casts on
  > the press). Left-click and Space both swing. Enemies are unchanged. Chosen
  > by the project owner over locking aim at the start.

- **World topology**: handcrafted hub (Cinderwake) and wilderness lane, with
  generated dungeon instances.
- **Core-loop priority**: narrative/quests and combat lead. Exploration,
  gathering, crafting, and settlement upgrades support them.
- **Progression**: distinct classes, scoped deliberately. Classes are
  first-class in the content schema from M6, but the vertical slice ships **one**
  fully playable and tuned class, with a second defined in data and left
  unpolished. This keeps the class architecture honest without tripling the
  animation, balance, and tuning bill on a demo.
- **Death/failure**: respawn at the settlement and keep progress. No corpse run,
  no dropped-currency recovery.
- **Settlement**: functional blueprint upgrades chosen from authored slots. No
  placement or building tools.

## Consequences

- All combat timing windows (wind-up, active, recovery, i-frames) are expressed
  in **fixed ticks**, never in frame time or seconds, so the feel is identical
  regardless of frame rate. This is why the fixed-timestep separation in M1
  matters more for this project than for a forgiving action game.
- Souls-lite combat needs an explicit attack state machine with cancel rules;
  it is the most timing-sensitive system in the slice and should be built
  before enemy variety, not after.
- A fixed camera angle means models and animations only have to read from one
  angle, which suits low-poly assets, a solo part-time schedule (ADR-0012's
  context), and cheap culling.
- Respawn-with-progress means no state rollback machinery for death; the
  rollback work in the AI pipeline stays isolated to content revisions.
- Blueprint upgrades keep settlement state as world flags and counters, not
  spatial data, so saves stay small and migration stays simple.

## Alternatives considered

- **One flexible archetype** (as originally written in §6): less content work
  and a faster slice, but it would defer the class-schema design until after
  the content format stabilized, which is the more expensive order.
- **Free-rotating third-person camera**: more immersive, but it demands camera
  collision, all-angle asset quality, and more animation work.
- **Souls-like corpse recovery**: thematically consistent with the combat
  choice, but it adds enemy-reset and drop-persistence rules for little payoff
  in a 20–30 minute demo.
