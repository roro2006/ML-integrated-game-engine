# ADR-0018: Builds — one class, a five-slot loadout, talents, and item empowerment

- Status: Accepted
- Date: 2026-09-23 (accepted the same day)
- Amends: ADR-0007 (progression), ADR-0013 (save format 1 → 2)

## Context

A class today is a fixed ability list bound to Q R F C, pools, and (since M9)
its own swing, stance, and roll. Nothing is chosen after character creation;
equipment adds flat damage and resistances; experience is granted but no
level is ever reached.

The design direction (2026-09-23), after surveying Grim Dawn, Path of Exile,
Guild Wars 2, Last Epoch, Diablo III/IV, and Elden Ring:

- **One class per character** — no Grim Dawn dual mastery.
- **Guild Wars 2's mesh of class and weapon**, extended: items can **grant**
  new abilities or **modify/empower** abilities the player already has.
- **Five ability slots, all open from the start.**
- **Skills are chosen**, not found as loot.
- **Respec is free.**

Every piece must stay inside ADR-0010: content composes bounded, allowlisted
primitives; nothing executes code; a model can later propose an item or a
talent only by choosing from these allowlists.

## Decision

### 1. Three sources of abilities, one loadout of five

The player's **available** abilities are the union of:

1. **Class abilities unlocked by talents** — the class's pool, opened by
   spending talent points (§3). Choosing a talent is choosing a skill.
2. **Item-granted abilities** — any equipped item may grant abilities while
   worn (`ItemDef.grants`). A weapon's own moves are simply grants.

The player slots **up to five** of the available abilities into the
**loadout**, keys **Q R F C V**. All five slots exist from the first minute;
what limits a new character is how many abilities are available, not slots.
Unequipping an item empties any slot holding an ability it granted.

The **basic attack** (left click / Space) is not a slot: it is the equipped
weapon's swing (§4), or the class's own swing unarmed.

*Considered:* Guild Wars 2's fixed split (weapon owns slots 1–5, class owns
6–0). Rejected for the slice: with five slots it would leave two or three
class choices, and the brief asks for maximum customisation within the cap.
Any ability in any slot keeps weapon grants meaningful — they compete for a
slot on their merits.

### 2. Tags, derived rather than authored

Every ability has **tags** from a closed set, derived from its effects so
they cannot be mis-authored: `Melee` (no projectile, targets in reach),
`Projectile`, `Area`, `Fire`/`Corruption`/`Physical` (from damage types),
`Status` (applies one), `Summon`, `Movement` (teleport), plus two authored
tags content may add: `Guard` and `Channel`. Tags are what empowerment
targets, so a fire talent strengthens a fire ability from any source —
class or item — without anyone writing the pairing.

### 3. Talents: one tree per class

`talent-trees.ron` gives each class one tree of **nodes** in **tiers**. A node
is one of:

- **Unlock** — adds a class ability to the available pool.
- **Empower** — one or more modifiers (§5) on an ability or a tag.
- **Attribute** — a bounded bump to a class number (health pool, stamina
  pool, poise, swing damage), reusing `ClassCombat`'s bounds.

Tiers gate depth: a tier opens once enough points are spent in the tiers
below it (Grim Dawn's mastery bar, simplified). Some nodes are **exclusive
groups** — take one of two or three — which is where a skill's behaviour is
chosen rather than only its numbers (Grim Dawn's transmuters, Last Epoch's
branches).

**Points** come from levels: a class states points at level 1 and per level.
Levelling needs a curve, which today does not exist; this ADR adds
`progression.ron` (experience per level, level cap) so granted experience
finally produces `LeveledUp`.

**Respec is free:** outside combat, the player may reset the tree and every
point returns. The loadout drops any ability no longer available.

### 4. Weapons as items with a swing and grants

`ItemDef.weapon` (new, optional, weapons only) states the basic attack —
damage, reach, radius, windows, cost — in `ClassCombat`'s shape and bounds,
and `grants` lists its abilities. The class still owns stance (poise) and
roll; the weapon owns the swing. Unarmed, the class's swing applies, which is
today's behaviour bit for bit.

### 5. Empowerment: bounded modifiers on an ability, applied at cast

An `AbilityModifier` is a closed enum of numeric changes, each bounded by the
loader and clamped again at runtime:

- damage ±%, cooldown −%, cost −%, wind-up −%, range/reach +, area radius +,
  projectile speed +, pierce +, status duration +%;
- **adds a status on hit** — naming a defined status, with a bounded duration.

A modifier targets **one ability by id** or **every ability with a tag**. It
comes from talents (§3) and from items (`ItemDef.empowers`, while equipped).

At cast time the runtime folds the player's modifiers into the ability's
definition — `empower(def, modifiers) -> AbilityDef`, a pure function — and
the executor runs the result unchanged. The executor's existing bounds
(`MAX_TARGETS`, projectile caps, status caps) still hold, because the folded
definition is clamped to the same limits the loader enforces. Nothing about
the executor learns about builds.

### 6. Canonical state and the save (format 1 → 2)

`Progression` gains:

- `talents: BTreeSet<TalentId>` — nodes taken.
- `loadout: [Option<AbilityId>; 5]` — what is slotted.

New domain events: `TalentTaken`, `TalentsReset`, `AbilitySlotted`,
`AbilityUnslotted`. The reducer checks only canonical ids and shapes; point
costs, tier gates, and availability are checked by `game_runtime` against
content before an event is raised, as affordability is today.

`SAVE_FORMAT_VERSION` becomes 2. The v1→v2 migration adds empty talents and a
loadout filled from the class's current ability list in order, so an
existing character keeps exactly the abilities on the keys it had. It ships
with a checked-in v1 fixture and a test, per ADR-0013.

Content schema stays at version 1: every content addition is optional.

### 7. UI

- **K — Abilities:** the five slots and every available ability with its
  source (class or item) and tags; pick a slot, pick an ability.
- **P — Talents:** the tree by tier, points to spend, each node's effect in
  words, a reset row.
- The HUD's ability list shows the five slots with their keys.
- Item descriptions in the inventory name what an item grants or empowers.

Both panels are the existing menu panel; no new UI technology.

### 8. What content and a model may do

Everything above is data over closed sets: talent nodes, modifiers, grants,
tags, weapon swings. A proposed item or talent can only choose from them, and
the loader bounds every number. There is no expression, script, or
conditional in any of it.

### 9. Where it sits

M9 finishes first (wave 3 reload, wave 4 exit test): tuning trees and items
is exactly the content edit loop reload shortens. Builds are then **M10's
first half**, before the content pass — plan in `docs/plans/m10-builds.md`.

## Consequences

- A save-format bump, with a migration — the first since ADR-0013.
- Four new content families or fields (`talent-trees`, `progression`,
  `ItemDef.weapon`/`grants`/`empowers`, ability tags) and the validation
  that comes with them.
- Class swing numbers (M9) become the unarmed fallback; weapons take over the
  swing.
- Two new panels. The menu-panel UI is list-based; a tree read as a list of
  tiers is adequate for the slice and is the thing most likely to want a
  real layout later.
- Deferred, with room left in the data model: a second mastery, Grim
  Dawn-style skill-bound triggers, per-skill trees beyond talents that target
  one ability, and skills dropped as loot.

## Resolved in review

1. **The fifth key is V.** Right-click keeps cancelling an aimed skill.
2. **No cap on abilities known.** Talents and items may make any number of
   abilities available; five are slotted at once because there are five
   keys, and all five slots are open from the start.
