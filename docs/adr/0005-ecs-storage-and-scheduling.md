# ADR-0005: ECS storage and scheduling

- Status: Accepted
- Date: 2026-09-15

## Context

Open decisions §16.2 and §16.3: sparse sets versus early archetypes, and
single-threaded determinism versus staged parallelism. `PROJECT_CONTEXT.md`
§5.3 already prescribes a small, comprehensible ECS that serves the vertical
slice, with archetypes and parallelism deferred until profiling demands them.

## Decision

- **Entities** are generational handles (`index`, `generation`). Freed slots
  are reused FIFO; a slot whose generation would overflow is retired instead of
  wrapping, so a stale handle can never alias a live entity.
- **Storage** is one sparse set per component type: a sparse index table plus
  dense entity and value vectors, with swap-remove on removal. Component types
  are opt-in via an explicit `Component` marker trait.
- **Queries** are explicit methods (`query`, `query_mut`, `query2`,
  `query2_mut`, `for_each2_mut`) rather than a generic tuple-query system.
  Paired queries iterate the first component's dense order and look up the
  second, and two mutable storages are obtained with
  `slice::get_disjoint_mut`, so no `unsafe` is needed.
- **Determinism**: iteration order is a pure function of the history of
  structural changes, never of hash order or addresses. Component storages are
  kept in a `Vec` indexed by registration order for the same reason.
- **Structural change** is deferred through `Commands`, applied by the schedule
  immediately after each system, strictly in recorded order. Commands targeting
  stale entities are skipped and counted, never panicking. Pending entities
  carry a buffer and batch id so a handle cannot be misused across batches.
- **Resources** are typed singletons; `resource_scope` temporarily removes one
  so a system can use it alongside mutable world access.
- **Events** are double-buffered per type with an explicit cadence (`Frame` or
  `FixedTick`). An event is readable in the update it was sent in and the next
  one of that cadence. Readers own cursors and count missed events.
- **Scheduling** is single-threaded, with two schedules (fixed and frame), each
  an ordered list of named stages and systems, and per-system metrics.

Explicitly deferred: archetypes, parallel execution, change detection, runtime
reflection, dynamic query expressions, and ECS-as-save-format.

## Consequences

- Every behaviour above is testable headlessly and reproducible from a seed,
  which the M2 determinism test exercises over a 200-tick simulation.
- Sparse sets favour random access and cheap add/remove over the cache locality
  archetypes give during iteration. With vertical-slice entity counts this is
  not expected to matter; profiling decides, not taste.
- Explicit query methods mean adding a three-component mutable query is a new
  method rather than free generic composition. Accepted while the set is small.
- Single-threaded execution keeps ordering trivially deterministic and removes
  a large class of bugs; parallelism, if ever needed, must preserve determinism.

## Alternatives considered

- **Archetype storage now.** Better iteration locality, but a far larger
  implementation and a worse fit for sparse, rapidly changing composition; it
  also optimizes before any profile exists.
- **Generic tuple queries with runtime borrow checks.** Nicer ergonomics, but
  needs `unsafe` or `RefCell` everywhere and turns aliasing mistakes into
  runtime panics instead of compile errors.
