# ADR-0022: Skill-based progression

- Status: Accepted
- Date: 2026-10-09
- Amends: ADR-0015 (classes are starting presets and no longer the progression source; their pools and `ClassChosen` shape stay). Amends ADR-0018 (abilities become available through skill thresholds as well as talents and items; talent trees attach to skills rather than to a class; one class per character, the five-slot loadout, and the event names all stay).

## Context

ADR-0015 made classes the source of health and stamina pools and of the abilities a character can have. ADR-0018 then made a class carry one talent tree, unlocked by points earned from character level. The owner approved the master plan on 2026-10-09. Its workstream F5 replaces class-only progression with use-based skills, and it says classes become starting loadouts. The reference game uses per-skill assets for construction and farming, so the direction has a precedent in the target game's configuration (`docs/reference/observed-settings.md` §4).

The code already has pieces of this. `Progression` carries an experience pool, a level, a talent set and a loadout. A `level-curves` family exists and is reused. The `ExperienceGained` and `LeveledUp` events are in old logs and must replay forever (ADR-0013).

## Decision

1. **Skills are the progression.** Each skill is a content entry in `skills.ron`, with a level curve from `level-curves` and unlock thresholds for abilities, recipes and tiers. Skills gain integer XP from domain events: gathering, crafting, combat and construction. `xp-rules.ron` maps each action to an XP amount, with a daily cap in ticks and a diminishing-returns table. Nothing in a skill path uses floating point.

2. **State.** `Progression` holds `skills: BTreeMap<SkillId, SkillState>`, where `SkillState` is `{ xp, level }`. A level is recorded at the moment it is reached and is never recomputed on load, so replay reproduces it on the same tick. Derived values, such as gather speed, craft quality and unlock checks, are computed from the level and are not stored.

3. **Abilities and talents.** A character can use an ability if it is unlocked by a skill threshold or a talent, or granted by an equipped item (ADR-0018 §1, extended). Each skill may have one talent tree, organised in tiers and built from ADR-0018's node kinds. Talent points come from skill levels, and the amount per level is content. Respec remains free outside combat. The gameplay workstream's D-c is decided: the owner confirmed keeping talents, attached to skills, on 2026-10-09.

4. **Classes are starting presets.** A class keeps `base_health` and `base_stamina`, and `ClassChosen` still carries the resolved pools (ADR-0015). It gains an optional preset: starting skill XP, starting items and a starting loadout. Choosing a class sets the starting state. It no longer decides which abilities the character can ever use.

5. **Events.** Skill XP and level events are added, and the talent events from ADR-0018 §6 keep their names: `TalentTaken`, `TalentsReset`, `AbilitySlotted` and `AbilityUnslotted`. Reducers check canonical ids and shapes only. Affordability, tier gates and availability are checked in `game_runtime` before an event is raised. `ExperienceGained` and `LeveledUp` remain in the event enum, with their reducers, for old logs.

6. **Migration.** The save bump follows D-f (one bump per phase), and this ADR does not fix its number. The migration adds an empty `skills` map. It sets the class's primary skill's XP from the old `experience` value and recomputes that skill's level once. The loadout and talents carry over unchanged. A checked-in fixture of an existing save with a level above 1 is required. The migration must reproduce the same abilities on the same keys, the same pools and the same level.

7. **Single-player first, co-op ready.** There is no networking. Skill and talent rules use stable ids, and reducers and runtime checks are authority-side, so a client cannot grant XP or a talent by sending an event. This keeps the F10 readiness rules without adding networking.

## Consequences

- ADR-0015's class-derived pool is no longer the only way a character gets abilities. The `ClassDef` shape changes only by an optional preset, which is additive under schema version 1.
- The ADR-0018 talent-tree design is kept and moved onto skills. Its deferred items stay deferred: skill-bound triggers, skills dropped as loot, and per-skill trees beyond talents that target one ability.
- Every skill XP source must emit a domain event. A gathering or crafting action that does not emit one earns nothing, which is intended.
- The daily cap is measured in ticks from the game clock, so the grind test is deterministic.
- Menu panels for skills and talents use the existing menu model. This does not add a UI technology.
- Open decisions that touch skills are left to their own items: D-d (instances for tools and gear), D-h (survival failure) and D-j (hotbar layout).

## Alternatives considered

- **Keep class-only progression.** It is what ADR-0015 and ADR-0018 provide. Rejected because the owner's game-shape decision asks for use-based skills.
- **Skills earned from loot.** Deferred by ADR-0018, and not needed for the first skill set.
- **One talent tree per class (ADR-0018 as written).** It would tie talents to a class that is now only a preset. Rejected in favour of trees on skills.
- **Mastery trees in the Grim Dawn style, with a second mastery.** More choice, and more balance work. Deferred, with room left in the data model.
