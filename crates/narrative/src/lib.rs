//! Quest runtime, dialogue, and the deterministic content director.
//!
//! Quest graph evaluation, objective completion paths, dialogue conditions,
//! and template-driven content selection that works with no AI available.
//!
//! Milestones M5 (authored quests) and M12 (template director).
//!
//! # Everything here is a pure function over content and canonical state
//!
//! Both runtimes take `&ContentDef` plus `&WorldState` and return
//! [`DomainEvent`](game_core::events::DomainEvent)s. Neither mutates the world,
//! and neither knows the ECS exists. Three things follow, all of them load-
//! bearing:
//!
//! - **Every narrative outcome is replayable**, because the only way a quest or
//!   a conversation can change anything durable is through the event log.
//! - **Both are testable without a game**, from synthetic state — no frame loop,
//!   no renderer, no content directory.
//! - **The AI pipeline inherits the same boundary for free** (ADR-0010): a
//!   proposed quest or conversation can only compose primitives these runtimes
//!   already implement, so the worst a bad proposal can do is name something
//!   that does not exist, which validation rejects before it is ever run.
//!
//! This crate deliberately does not depend on `content_runtime`. Callers look
//! definitions up and pass them in, which is what keeps the runtimes drivable
//! from a test fixture rather than only from a loaded content set.

pub mod dialogue;
pub mod quest;

pub use dialogue::{
    Choice, Conversation, DialogueError, Turn, available, evaluate, is_available, is_open, open_conversations,
};
pub use quest::{QuestError, Signal, accept, advance, open_objectives, reward_events};
