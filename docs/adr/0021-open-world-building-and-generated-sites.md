# ADR-0021: Open-world cells, snap-piece building, and generated sites in saves

- Status: Accepted
- Date: 2026-10-09
- Supersedes, in part: PROJECT_CONTEXT §2 non-goals "Open-world streaming at large scale" and "Freeform voxel building or terrain deformation". Only the scope set out below is superseded. Terrain deformation stays excluded.
- Amends: ADR-0007, in its "no spatial data in saves" consequence and its "no placement or building tools" settlement rule, for claimed areas only. Amends ADR-0010, Permanence rule 2 and the rule against persistent generated locations, for open-world sites only (see Decision 4).

## Context

The world is location-based today. `PlayerState::location` names an authored location, the saved world has no spatial state, and collision is a ground plane. ADR-0007 kept settlement state as flags and counters, so that saves stay small and free of spatial data. The owner approved the master plan on 2026-10-09. It asks for a streamed open world with building and generated content (workstreams F3, F4, F9).

Owner decision D-a, recorded in the gameplay workstream §5 as decided on 2026-10-09: generated open-world sites are stored in the save. Their overlays are pinned there with provenance. Site existence and position stay deterministic from the seed. Instanced dungeons keep ADR-0010's discard rule. The owner also approved keeping the AI-content pipeline on top of the open world, so generated quests and locations stay. ADR-0010's safety rules apply to all of it.

## Decision

1. **Cell world streaming.** The world is partitioned into cells on a grid. Each cell has a baseline generated as a pure function of the world seed, the cell id and the content revision, plus saved deltas for whatever the player changed. Cells load and unload around the player with hysteresis, so the boundary does not thrash. Baselines are generated off-thread and return plain data. Entities are created only at fixed-step safe points. The hub and dungeons remain authored locations.

2. **Snap-piece building.** Construction pieces are data in `pieces.ron`. Socket kinds form a closed Rust enum with a compatibility table. A support graph checks stability, and collapse order is sorted by structure id, so it is deterministic. Placement is checked by one pure function, which the ghost preview and the reducer both call. Building is limited to claimed areas. Ownership of a claim that crosses a cell boundary is one record. There is no freeform voxel building and no terrain deformation.

3. **Placement of generated sites.** Sites are placed into cells by seeded procedural code. Their ids are minted by code, in the `site-` form, and never by a model. A site that fails the fit checks (terrain, clearance from structures, reachability from the cell boundary) is dropped, not moved, so placement remains a pure function. An optional AI overlay may choose names and encounter composition from closed sets, and its names are instance-scoped `inst:` ids. A rejected overlay falls back to the template's deterministic defaults.

4. **Owner decision D-a.** For persistent open-world sites, the save holds the site record: its overlay, its provenance (model, prompt hash, seed, content hash) and its state changes. The seed still determines where a site is. A site is discarded only by an explicit owner or content action. Generated site records live in cell deltas and never in the authored location registry. This is the exception to ADR-0010's permanence rules and discard rule. Instanced dungeons keep the discard rule in full.

5. **ADR-0010 safety rules hold.** Schema conformance, id syntax and uniqueness, reference resolution, budgets, progression safety and immutable-fact protection all apply, as does the allowlist of rewards. Instance-scoped names are still not canonical ids (rule 1). Generated content writes nothing to the NPC roster, faction list or location registry (rule 3). Nothing is promoted automatically (rule 4). Only the open-world discard rule and the permanence clause change, as Decision 4 describes.

6. **Saves.** The save format moves to the next version as one bump per phase, under D-f. Migrations follow ADR-0013 and ship with checked-in fixtures. The migration form is a separate decision (D-l), and this ADR does not settle it.

7. **Adversarial overlays.** Every overlay is checked against a fixture set. The set covers an out-of-allowlist enemy, a canonical id, a reversed navmesh polygon, an unreachable site, an instance-scoped id leaking into canonical state, and a reward that is not an authored item. Each one must be rejected with the deterministic fallback used.

## Consequences

- Save size grows with the number of deltas and site records, not with explored area. A test asserts this.
- ADR-0010's permanence guarantee is weaker for open-world sites than for dungeons. The trade is deliberate, and the consequence is that the validators carry more of the safety story.
- Invisibility still holds (ADR-0010). Provenance is recorded and never surfaced, and a UI audit test checks for generated labels.
- Decay, upkeep and offline progress are not decided here. They are open decisions D-b, D-e and D-k in the gameplay workstream. Cell-boundary and load-time mismatch policy also need to be settled before F3 is built.
- Stability checks must stay fast on large bases. The budget for an incremental update is set at the first benchmark, not here.
- The gameplay workstream's ADR map notes that ADR-0021 covers three areas. If the owner prefers smaller records, splitting it needs numbers beyond 0022.

## Alternatives considered

- **Keep the single-location world and author every site.** Simplest, and it fails the owner's open-world and content decisions.
- **Freeform voxel building.** Reachable with the renderer work, but the support and save costs are far higher and the stability rules are much harder to test. Rejected.
- **Regenerate sites from the seed without saving them.** Keeps saves small, but any later change to the generator would silently change the player's world, and player-visible overlays could not be pinned. D-a rejects this.
- **Let the model place sites.** Rejected by ADR-0010's rule that a model may not choose canonical ids and by the owner's choice of procedural placement.
