//! Canonical game-domain state.
//!
//! This crate is deliberately independent of the engine, renderer, and any
//! runtime ECS type: it holds stable typed IDs, serializable world state,
//! domain events and reducers, and the rules that validate them. It is the
//! input to saves, replay, content validation, AI context, and (later)
//! server-authoritative multiplayer.
//!
//! Runtime entity handles must never appear here.
//!
//! # Determinism
//!
//! Replaying an event log from a seed must reproduce state exactly, so
//! canonical state uses ordered containers (`BTreeMap`/`BTreeSet`) throughout
//! and never a hash-ordered one. That also keeps serialized output stable, so
//! saves diff cleanly and content hashes mean something.
//!
//! See `docs/plans/m4-domain-content.md`.

pub mod content;
pub mod events;
pub mod ids;
pub mod log;
pub mod save;
pub mod world;
