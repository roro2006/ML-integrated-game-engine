//! Deterministic procedural generation.
//!
//! Seeded room-graph and biome assembly, landmark/objective/spawn placement,
//! reachability and spawn-safety validation, and encounter budgeting. All
//! generation is a pure function of a seed and a specification so results are
//! reproducible and replayable.
//!
//! Milestone M11.
