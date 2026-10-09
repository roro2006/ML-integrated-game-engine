# ADR-0010: AI content policy — visibility, permanence, and boundaries

- Status: Accepted; permanence rule amended for open-world sites by ADR-0021 (safety rules unchanged)
- Date: 2026-09-15

> **Numbering note (2026-09-22).** Milestone numbers below predate two renumberings
> (ADR-0016, ADR-0017). The "M9" this ADR refers to — optional local AI — is now
> **M13**. The current list is `PROJECT_CONTEXT.md` §13. The policy itself is
> unchanged.

## Context

Open decisions §16.8–§16.12. These determine what the AI pipeline is allowed to
produce, what validators must enforce, and whether a network path exists in a
shipped build. They are the decisions most expensive to reverse, because saves
and content revisions created under one policy have to remain loadable.

## Decision

**Visibility: invisible.** Generated content is presented exactly like authored
content. No AI labels, no "generated" badges. Provenance is still recorded in
saves (model, prompt hash, seed, content hash, schema version) for debugging,
replay, and review — it is simply never surfaced to the player.

**Permanence: instance-scoped names only.** AI may mint new named entities
(mini-bosses, room names, local lore) **only inside a generated instance**. It
may never create permanent NPCs, factions, locations, items, or lore in the
persistent world. Concretely, validators enforce:

1. Generated names live in an instance-scoped ID namespace tied to the
   instance's content revision; they are not valid canonical IDs.
2. Instance-scoped content is discarded with its instance. Nothing outside that
   instance may hold a reference to an instance-scoped ID.
3. Instance-scoped content may affect the persistent world only through
   allowlisted domain events (reputation deltas, world flags, item grants from
   the authored item set) — never by writing to the NPC roster, faction list,
   or location registry.
4. Promotion of an instance-scoped entity to canonical status requires a human
   authoring step; the pipeline has no automatic path.
5. All existing validations still apply: schema conformance, ID syntax and
   uniqueness, reference resolution, budgets, progression safety, and
   immutable-fact protection.

**Free-form natural language: not a near-term goal.** Player input stays
bounded (controls, menus, authored dialogue choices). No model sits between
player intent and the simulation.

**Review controls: developer-only.** The debug tooling shows candidates,
prompts, responses, diffs, validator failures, seeds, provenance, and offers
rollback. Players get no reroll, accept, or reject controls — that would
contradict the invisibility decision.

**Cloud and telemetry: opt-in cloud, never telemetry.** A cloud provider
adapter may ship behind the engine-owned provider interface, disabled by
default, never required for any gameplay path. No analytics, crash reporting,
or usage data is transmitted, ever. Diagnostics stay on the local disk.

## Consequences

- The safety story stays simple and testable: the worst a compromised or
  hallucinating model can do is produce an invalid instance, which validation
  rejects and a deterministic template replaces.
- Instance-scoped naming needs to exist in the ID model from the start of M4,
  even though AI arrives in M9. Retrofitting a namespace into an ID scheme
  after saves exist is exactly the kind of migration worth avoiding.
- Invisible AI means content quality must clear the same bar as authored
  content, with no "it was generated" excuse available to the player.
- An opt-in cloud adapter means the shipped binary contains network code, so
  the "works fully offline" guarantee must be covered by tests, not by the
  absence of the capability.

## Alternatives considered

- **Fully canonical AI authorship** (new permanent NPCs and factions): the most
  interesting emergent worlds, and the hardest to keep consistent, validated,
  and rollback-safe. The instance-scoped middle ground preserves most of the
  creative payoff at a fraction of the risk.
- **Explicit player-facing AI with reroll**: more honest with players and a
  useful feedback channel, but it invites content to be judged as AI output and
  contradicts invisibility.
- **No cloud path at all**: the strongest privacy promise; rejected to keep
  higher-quality models available to users who choose them.
