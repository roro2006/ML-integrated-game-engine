# ADR-0015: Character classes are content, and `ClassChosen` carries the pools

- Status: accepted; amended by ADR-0022 (classes become loadout presets)
- Date: 2026-09-17
- Supersedes: none

## Context

The player's health and stamina had no authored source. `PlayerState` derives
`Default` and `Pool` has no `Default` impl, so a fresh world recorded the player
as `0/0` in both pools. Nothing noticed, because nothing read canonical health:
the runtime invented `Health::full(100.0)` when it spawned the player.

That became urgent when implementing health synchronisation at safe points
(M5). Writing the runtime value into canonical state would have clamped it to a
`max` of zero — the reducer clamps to the pool — leaving the player recorded as
dead the first time they walked through a door, and respawning forever.

So the pools need an authored origin. ADR-0007 already chose distinct classes
over one flexible archetype, and `ClassId` was always a content id rather than
an enum precisely so that adding a class is an authoring change.

## Decision

**Classes are a content family.** `ClassDef` carries `base_health` and
`base_stamina` as `Fixed` milli-units, alongside the abilities the class starts
with. It validates that neither pool is zero or negative — a class nobody could
survive as is an authoring mistake, not a difficulty option.

**`ClassChosen` carries the resolved pools**, not just the class id:

```rust
ClassChosen { class: ClassId, health: u32, stamina: u32 }
```

The reducer sets `player.health` and `player.stamina` from the event. It does
*not* look the class up.

## Why the event carries the pools

Three reasons, in order of weight:

1. **`apply` stays a pure function of canonical state.** Reducers take
   `&mut WorldState` and nothing else. Giving them a content registry to consult
   would make replay depend on which content happened to be loaded, which is the
   one property the event-sourced design cannot afford to lose.
2. **A replayed log reproduces the pools the character actually had.** Retuning
   `wayfinder` from 120 to 140 health must not silently rewrite an existing
   save's history. This is the same rule an attack already follows when it
   copies its timing at the start of a swing, so that retuning an ability cannot
   alter a swing already in flight.
3. **`game_core` keeps no dependency on `content_runtime`.** The domain crate
   describes what is true, not where content lives.

The cost is a slightly fatter event. That is the right trade: the alternative
buys a smaller log by making replay conditional on content revision.

## Consequences

- A new `classes` family in the loader, hashed and reference-checked like the
  rest; its `abilities` resolve against the ability registry.
- `Slice` chooses a class when it builds a world, so canonical pools are real
  before anything reads them.
- Health synchronisation at safe points becomes meaningful, which was the
  change that exposed all of this.
- `validate_canonical` *should* treat a zero `max` as malformed, since no
  legitimate path now produces one — but it cannot yet say so.
  `WorldValidationError` is shaped entirely around ids (`site`, `id`, `source`),
  so a pool problem does not fit without turning it into an enum. Left as a
  follow-up rather than forced in: the guard is worth having, and worth having
  in a shape that can also express the next non-id invariant.
- M6 inherits the family rather than inventing it: adding per-class abilities,
  starting equipment, or progression curves is an additive schema change.

## Alternatives considered

- **Hand-written `PlayerState::default()` with 100/100.** Simplest, and fixes
  the immediate bug, but it puts a balance number in the domain crate and makes
  every world identical regardless of class — which contradicts ADR-0007.
- **The slice seeds the pools from its own constants.** Keeps `game_core`
  neutral, but any other consumer of `WorldState::new` still gets `0/0`, so the
  bug waits for the next caller instead of being fixed.
- **The reducer looks up the class.** Rejected for reason 1 above.
