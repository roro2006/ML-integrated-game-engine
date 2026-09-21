//! Optional AI gateway: the engine-owned boundary around content generation.
//!
//! AI proposes data. It never executes code, runs commands, reads or writes
//! files, touches the environment, mutates ECS or save state, or bypasses
//! validation (ADR-0010; PROJECT_CONTEXT §9). This crate is where that rule is
//! made structural rather than merely stated: a [`GenerationRequest`] carries
//! typed facts, a seed, and budgets — no prompt string, no path, no handle —
//! and a [`GenerationResponse`] carries text and [`Provenance`], nothing that
//! can act.
//!
//! # Where this sits in the candidate lifecycle
//!
//! ```text
//! world context
//!   -> provider            <- this crate ends here
//!   -> raw candidate       <- this crate ends here
//!   -> parse and schema validation
//!   -> reference resolution and content compilation
//!   -> invariant, budget, and progression validation
//!   -> optional bounded repair attempt
//!   -> dry run where applicable
//!   -> safe-point activation
//!   -> atomic domain event and content revision commit
//!   -> runtime ECS projection
//!   -> save provenance, hash, seed, and replay data
//! ```
//!
//! Everything after "raw candidate" belongs to other crates, which own the
//! content model. This crate obtains bounded bytes and records where they came
//! from; it does not judge them.
//!
//! # Every failure ends in a deterministic template
//!
//! Generation is an enhancement, never a dependency: the game must be fully
//! playable with this crate inert, on minimum-spec hardware, offline, with AI
//! disabled (PROJECT_CONTEXT §10). So every failure here is a *typed outcome*
//! rather than an exception — see [`ProviderError`] — and the caller's answer
//! to all of them is to build the content from a deterministic template
//! instead. That fallback lives in the crates that own the content, not here.
//! Nothing in this crate may block input, rendering, fixed-step simulation,
//! combat, saving, or load completion.
//!
//! # Milestone status
//!
//! M4 provides the interface, a deterministic [`MockProvider`], and the
//! failure modes — no model, no network, no dependencies beyond `serde`. Real
//! llama.cpp and Ollama adapters arrive in M13 and will be out-of-process HTTP
//! implementations of [`ContentProvider`]; the trait is shaped so that they
//! can be, and so that downstream consumers written against the mock today do
//! not change when they land.

mod hash;
mod mock;
mod provider;
mod request;

pub use mock::{MockBehavior, MockProvider};
pub use provider::{ContentProvider, GenerationResponse, Provenance, ProviderError};
pub use request::{
    Budget, ContextError, GenerationKind, GenerationRequest, InstanceScope, PromptHash, WorldContext,
    WorldFact,
};
