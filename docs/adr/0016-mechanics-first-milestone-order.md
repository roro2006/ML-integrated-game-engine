# ADR-0016: Mechanics-first milestone order

- Status: Accepted
- Date: 2026-09-18

## Context

M5 closed the vertical slice's loop: a player can talk, accept a quest, fight,
gather, craft, build an upgrade, travel, die, respawn, save, and resume. 450
tests pass. On the roadmap as written, the next milestone is M6 "Data-driven
content runtime", and the one after that is procedural generation.

Auditing the slice against `PROJECT_CONTEXT.md` §6 before starting M6 showed
that combat is considerably thinner than the milestone plan assumes, and thinner
than the test count suggests:

| Capability | Schema | Runtime |
|---|---|---|
| `AbilityDef` and the `Effect` allowlist | defined, validated, reference-resolved, hashed | **never executed**; no `Effect::*` match arm exists outside `game_core` and `content_runtime` |
| `StatusId` / status effects | defined | never leaves `game_core` |
| `cooldown_ticks` | authored per ability | no cooldown code in the runtime |
| `DamageType` | defined; `ember-cleave` deals `Fire` | never leaves `game_core`; `Health::apply_damage` takes a bare `f32` |
| Equipment | `ItemEquipped`/`ItemUnequipped` events, `EquipmentSlot`, canonical slots | `ItemDef` carries no stats, nothing in the runtime reads equipment, and the events are never emitted |
| Player moveset | — | exactly one swing, hardcoded at `sample_game/src/slice.rs:180`–`196` |

§6 requires "two active abilities". Zero exist. The three authored abilities —
`ember-cleave`, `warden-guard`, `call-husks` — are dead content, and both
classes in `classes.ron` declare `abilities: []`.

Enemies are similarly thinner than the roster implies. `slice.rs:836` builds a
brain with `EnemyBrain::new(def.behavior)` and then overrides only `damage` and
`max_health` from content. Aggro radii, preferred range, move speed, and every
attack timing come from a hardcoded four-arm match in `enemy.rs:99`–`108`. So
"three enemy types and one elite" is one behaviour tree with four sets of
numbers, and the elite's distinguishing ability can never fire.

ADR-0007 anticipated this exact ordering risk and was not followed:

> Souls-lite combat needs an explicit attack state machine with cancel rules; it
> is the most timing-sensitive system in the slice and should be built before
> enemy variety, not after.

Building the data-driven content runtime (old M6) on top of this would satisfy
its exit criterion — "a new quest, enemy variation, and encounter can be created
through content files without changing Rust game logic" — only in the narrow
sense that a new *enemy definition* parses. Any enemy variation with an actual
ability, status, or damage type would still require Rust. The milestone was
scoped as schema work when the real gap is the runtime that executes a schema
that already exists.

## Decision

**Reorder the milestone plan so combat mechanics precede world and quest
content.** Concretely:

- **M5 is re-scoped and complete.** Its exit criterion becomes "the slice's loop
  is closed and playable offline, end to end" — which is already true and is
  what M5 actually proved. The "20–30 minute hand-authored quest loop" moves out
  of M5.
- **M6 becomes "Combat and the ability runtime"** (new): an executor for the
  `Effect` allowlist, with costs, cooldowns, targeting, status effects, typed
  damage, and equipment that modifies combat.
- **M7 becomes "Enemies, encounters, and combat variety"** (new): per-enemy
  tuning driven by content rather than a match arm, telegraphs, elite mechanics,
  and encounters that play differently from one another.
- **M8 is the former M6**, "Data-driven content runtime".
- **M9 is the content pass** that authors the 20–30 minutes, deliberately placed
  *after* M8 so it is authored against a runtime where content genuinely drives
  behaviour.
- **M10–M14 are the former M7–M11**, shifted by three.

**Amend §16 product decision 15.** It reads "Demo target remains the 20–30
minute slice in M5." The demo target is unchanged in substance — it is still a
20–30 minute offline slice — but it is delivered in M9 rather than M5.

## Consequences

- **The ability runtime is the milestone, not a feature of it.** Roughly 200
  lines of schema in `game_core::content::abilities` and the `Effect` enum become
  live code paths. The loader already resolves `ClassDef.abilities` and
  `EnemyDef.abilities` (`content_runtime/src/lib.rs:670`–`681`), rejects empty
  effect lists, and checks non-negative costs, so reference integrity is free and
  M6 is purely executor work.
- **No save-format change and no migration.** `DomainEvent` is finite and holds
  no ability or status variants, which is correct under M5's established rule:
  combat runs in the runtime and only its *outcomes* become events. Statuses,
  cooldowns, and in-flight ability state are runtime state, rebuilt on load like
  attack phase and current stamina. `format_version` stays 1.
- **Typed damage lands on the runtime side.** Canonical `Pool` is `u32` and stays
  that way; runtime `Health`/`Stamina` are `f32`, and resistances belong there.
  `Fixed` (milli-units) remains the content representation. This keeps the change
  out of ADR-0013's expensive-to-reverse territory entirely.
- **ADR-0010's safety claim becomes testable.** That ADR argues the worst a
  hallucinating model can do is propose an invalid instance that validation
  rejects. Today the `Effect` primitives are inert, so the claim is untested by
  construction — nothing executes the allowlist, so nothing proves the allowlist
  bounds anything. An executor with per-primitive tests is what makes the
  AI-safety story real, and it must exist before M12's AI integration rather
  than alongside it.
- **The demo slips by two milestones.** This is the real cost and it is accepted
  deliberately: a 20–30 minute slice built on one hardcoded swing would be 20–30
  minutes of walking between conversations, and the content authored against it
  would need reworking once abilities landed. Authoring twice is more expensive
  than reordering once.
- **Enemy variety gets cheaper, not more expensive.** Once behaviour tuning comes
  from `EnemyDef` rather than a match arm, a new enemy is a content file. That is
  the old M6 exit criterion, now actually reachable.
- **Procedural generation and the director are unaffected in substance.** They
  shift by three numbers and inherit a richer set of primitives to compose,
  which is the order those milestones assumed anyway.
- **Milestone references written before this ADR mean the old numbering.**
  Accepted ADRs are never rewritten (`docs/adr/README.md`), so ADR-0007's "engine
  work in M3–M6" and "first-class in the content schema from M6", ADR-0009's
  "M7's reachability and spawn-safety validation", and ADR-0010's "AI arrives in
  M9" all refer to the pre-2026-09-18 sequence — respectively the old M6
  (data-driven content runtime, now M8), the old M7 (procedural regions, now
  M10), and the old M9 (local AI, now M12). This ADR is the mapping. Source
  doc-comments and the M4/M5 plan documents carry the same stale references and
  *are* correctable, since they are not history; that correction is tracked as
  follow-up work rather than folded into the reordering itself.

## Alternatives considered

- **Insert a mechanics milestone and renumber nothing else** (M5.5). Less churn
  in the document, but it leaves the old M6 exit criterion stating something
  false about the codebase, and it hides the fact that the content pass also
  moved. Renumbering is a one-time documentation cost that keeps the plan
  honest.
- **Deepen the melee core first** — poise, hitstun, stagger, weapon movesets,
  directional blocking — before building the ability runtime. This would make
  the moment-to-moment fight good soonest, and needs no new schema. Rejected as
  the *first* step because §6's "two active abilities" would stay unmet, the
  authored ability content would stay dead, and ADR-0010's safety boundary would
  stay untested. Combat feel work is folded into M7, on top of a runtime worth
  extending.
- **Do both before any content** (melee depth then abilities, as one milestone).
  Highest confidence in the end result, roughly double the milestone size, and
  the demo slips furthest. Split into M6 and M7 instead so each has an
  independently verifiable exit.
- **Proceed with the old order and accept shallow combat.** Defensible if the
  project's goal were the AI pipeline alone, but §16 decision 13 sets the intent
  as a portfolio piece favouring "a tight, finished vertical slice over engine
  breadth", and a tight slice with one attack is not tight, it is short.
