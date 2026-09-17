# ADR-0013: Save envelope, migration policy, and event log format

- Status: Accepted
- Date: 2026-09-16

## Context

M4 introduces the first durable persistence. Three project rules bind this
decision hard: saves carry explicit format versions and support migrations,
player state is never silently discarded on a parse or version error, and
replay from a seed must reproduce state exactly. `serde` and `ron` are approved
dependencies; `serde_json` is not.

Save formats are close to irreversible. Once a player has a world on disk, every
later version must either load it or migrate it, so the shape and the failure
policy need fixing before the first file is written — not after.

## Decision

**Envelope.** Every save is a versioned envelope wrapping canonical state:

- `format_version: u32` — the envelope's own version, read and checked *before*
  any state is parsed.
- the `WorldState` (which carries its own `schema_version`),
- engine version, content revision id, and content hash,
- the world seed,
- provenance, including AI provenance for accepted generated content.

**Format.** RON, pretty-printed. Saves are diffable and inspectable, which
matters for a text-first project and for debugging replay divergence.

**Version policy.**

- *Older* than current: run migrations in sequence to the current version.
- *Newer* than current: refuse with a clear error naming both versions, and
  leave the file untouched. Never downgrade, never partially load.
- Unknown fields: reject (`deny_unknown_fields`) rather than ignore. Silently
  dropping a field is precisely how player state disappears; the friction of a
  version bump is the intended cost.

**Migrations** are pure functions from version N to N+1 operating on a
permissive intermediate representation (`ron::Value`), not on typed structs.
Typed structs describe only the *current* shape, so a migration written against
them stops compiling the moment the types move on — which is how migration
chains rot. Every migration ships with a checked-in fixture of the old shape and
a test asserting the migrated result.

**Atomic writes.** Serialize to a temporary file in the destination directory,
flush and sync it, then rename over the target. A crash or power loss mid-save
must leave the previous save intact. `std::fs::rename` replaces an existing file
on both Windows and Unix; syncing the containing directory is Unix-only and is
done where available.

**Event log.** Line-delimited RON — one accepted domain event per line,
append-only, stored in the save world directory. This deviates from
PROJECT_CONTEXT §8, which names NDJSON: `serde_json` is not an approved
dependency, and line-delimited RON provides the properties replay actually
needs — cheap appends, tailing, and a torn final line that can be detected and
discarded without losing the events before it.

**Replay contract.** State at any point equals the seeded initial state folded
over its accepted events. This is asserted by test, not assumed: a world is
saved, reloaded, replayed from its log, and compared for equality.

**No runtime identity.** ECS entity handles never appear in any of this. That
is structural rather than merely disciplined, because `game_core` does not
depend on the ECS at all (ADR-0002).

## Consequences

- Migration work starts immediately rather than being deferred to the first
  breaking change, when there would be worlds on disk and no mechanism.
- `deny_unknown_fields` makes every added field a versioned change. That is
  friction on purpose.
- The event log is not readable by JSON tooling. If `serde_json` is later
  approved for AI interchange, the log format is worth revisiting — the
  envelope and the log are independent choices.
- Pretty RON costs file size against binary formats. For single-player worlds
  of this scale that is a good trade; revisit only if profiling says otherwise.

## Alternatives considered

- **NDJSON** as §8 originally specified: better ecosystem tooling, but requires
  an unapproved dependency for no capability replay actually needs.
- **A binary format**: smaller and faster to parse, but opaque to inspection
  and diffing, which fights the project's text-first premise.
- **Serializing the ECS world** as the save: explicitly rejected in §5.3, and it
  would persist runtime identity — the exact thing the domain/runtime split
  exists to prevent.
- **Ignoring unknown fields** for forward compatibility: tempting, and the
  reason so many games lose player state on downgrade. Rejected.
