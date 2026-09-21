//! What the engine asks a provider for.
//!
//! A request is data: a category, a bounded set of canonical facts, a schema
//! version, a seed, and the budgets the call must respect. There is
//! deliberately no free-form prompt field and no path field. The prompt a
//! backend eventually sends is *derived* from these typed parts by the
//! adapter, so no caller can smuggle arbitrary text — or a file path, or an
//! environment variable — into the model's context by assembling a string
//! (ADR-0010; PROJECT_CONTEXT §9 "AI cannot ... load arbitrary local files or
//! access environment secrets").

use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::hash::{hash_field, hash_init, hash_u64};

/// The categories of proposal the AI is permitted to make.
///
/// This is an allowlist, not an open string, because "what may AI author" is a
/// policy decision rather than a caller's choice. Adding a category is a
/// deliberate edit here, reviewed against PROJECT_CONTEXT §9, instead of
/// something a downstream crate can do by inventing a new label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum GenerationKind {
    /// Quest graph parameters and narrative beats.
    Quest,
    /// Dialogue consistent with known NPC, faction, and world facts.
    Dialogue,
    /// Biome, room roles, landmarks, constraints, and encounter themes.
    MapSemantics,
    /// An enemy variant composed from approved archetypes and abilities.
    EnemyVariant,
    /// Encounter composition within explicit budgets.
    EncounterComposition,
    /// Reward flavor over an allowlisted reward set.
    RewardFlavor,
    /// A visual recipe referencing known assets, tags, materials, and VFX.
    VisualRecipe,
}

impl GenerationKind {
    /// Stable kebab-case wire name. Stable because it is hashed into
    /// provenance that saves must keep meaningful across releases, so these
    /// strings may not be renamed casually.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Quest => "quest",
            Self::Dialogue => "dialogue",
            Self::MapSemantics => "map-semantics",
            Self::EnemyVariant => "enemy-variant",
            Self::EncounterComposition => "encounter-composition",
            Self::RewardFlavor => "reward-flavor",
            Self::VisualRecipe => "visual-recipe",
        }
    }
}

impl fmt::Display for GenerationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The instance whose namespace a candidate may mint names in.
///
/// ADR-0010 permits AI to name things only *inside a generated instance*:
/// never a permanent NPC, faction, location, item, or piece of lore. Carrying
/// the scope on the request makes that rule visible at the boundary — there is
/// no way to ask for generation that is not attached to an instance, so
/// "canonical authorship" is not a request a caller can express.
///
/// # Seam
///
/// This is an owned string today. When `game_core`'s ID model lands, this
/// becomes a newtype over the typed instance ID and validation of the
/// `inst:<instance-id>/<name>` shape moves there; the field's meaning and its
/// position in the prompt hash do not change. Validation is deliberately not
/// duplicated here in the meantime, so there is one owner of ID syntax.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct InstanceScope(String);

impl InstanceScope {
    pub fn new(instance: impl Into<String>) -> Self {
        Self(instance.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for InstanceScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One canonical fact, as a subject/predicate/value triple.
///
/// Facts are triples rather than prose so that context is assembled from
/// things the world actually knows. A caller must decide, per fact, what it is
/// asserting; it cannot hand over a paragraph it composed elsewhere.
///
/// # Seam
///
/// `subject` becomes a typed `game_core` ID once that crate's ID model is
/// stable. It is a plain string here only to keep this crate compilable and
/// testable ahead of it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct WorldFact {
    subject: String,
    predicate: String,
    value: String,
}

impl WorldFact {
    pub fn new(subject: impl Into<String>, predicate: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            value: value.into(),
        }
    }

    pub fn subject(&self) -> &str {
        &self.subject
    }

    pub fn predicate(&self) -> &str {
        &self.predicate
    }

    pub fn value(&self) -> &str {
        &self.value
    }

    /// Total bytes of the triple, used to enforce the per-fact bound.
    fn byte_len(&self) -> usize {
        self.subject
            .len()
            .saturating_add(self.predicate.len())
            .saturating_add(self.value.len())
    }
}

/// A fact rejected by the context bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextError {
    /// The context already holds [`WorldContext::MAX_FACTS`] facts.
    TooManyFacts { limit: usize },
    /// A single fact exceeds [`WorldContext::MAX_FACT_BYTES`].
    FactTooLong { limit: usize, actual: usize },
}

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyFacts { limit } => {
                write!(f, "world context is limited to {limit} facts")
            }
            Self::FactTooLong { limit, actual } => {
                write!(f, "fact of {actual} bytes exceeds the {limit} byte limit")
            }
        }
    }
}

impl std::error::Error for ContextError {}

/// The canonical facts a request is allowed to reason from.
///
/// Bounded on purpose, in two directions: a cap on facts and a cap on the size
/// of each one. The caps keep prompts short (PROJECT_CONTEXT §10: "Keep
/// prompts concise and pass only relevant canonical state"), but their real
/// job is structural — an unbounded context is a free-form prompt wearing a
/// different type, and a single 100 KB "fact" would be exactly the blob this
/// design exists to prevent.
///
/// Order is significant: it is part of the prompt hash, because two orderings
/// are two different prompts and must not share a cache entry or a provenance
/// record.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct WorldContext {
    facts: Vec<WorldFact>,
}

impl WorldContext {
    /// Maximum number of facts in one context.
    pub const MAX_FACTS: usize = 64;
    /// Maximum bytes in a single fact triple.
    pub const MAX_FACT_BYTES: usize = 512;

    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a fact, rejecting it if it would break a bound.
    pub fn push(&mut self, fact: WorldFact) -> Result<(), ContextError> {
        if self.facts.len() >= Self::MAX_FACTS {
            return Err(ContextError::TooManyFacts {
                limit: Self::MAX_FACTS,
            });
        }
        let actual = fact.byte_len();
        if actual > Self::MAX_FACT_BYTES {
            return Err(ContextError::FactTooLong {
                limit: Self::MAX_FACT_BYTES,
                actual,
            });
        }
        self.facts.push(fact);
        Ok(())
    }

    /// Builder form of [`WorldContext::push`].
    pub fn with_fact(mut self, fact: WorldFact) -> Result<Self, ContextError> {
        self.push(fact)?;
        Ok(self)
    }

    pub fn facts(&self) -> &[WorldFact] {
        &self.facts
    }

    pub fn len(&self) -> usize {
        self.facts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }
}

/// The limits a single generation call must respect.
///
/// Both limits exist so that a failure is a *typed outcome* rather than a
/// stall: a provider that never answers costs at most `timeout`, and a
/// provider that answers forever costs at most `max_output_bytes`. Neither one
/// is advisory — the caller's fallback path is what runs when either is hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    max_output_bytes: usize,
    timeout: Duration,
}

impl Budget {
    /// Default output cap, sized from PROJECT_CONTEXT §10's
    /// `max_output_tokens = 1200` with generous headroom for bytes per token.
    pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 8192;
    /// Default deadline, matching §10's `request_timeout_seconds = 30`.
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

    pub const fn new(max_output_bytes: usize, timeout: Duration) -> Self {
        Self {
            max_output_bytes,
            timeout,
        }
    }

    pub const fn max_output_bytes(self) -> usize {
        self.max_output_bytes
    }

    /// The wall-clock deadline for the whole call, including connect and read
    /// time for an out-of-process provider.
    pub const fn timeout(self) -> Duration {
        self.timeout
    }
}

impl Default for Budget {
    fn default() -> Self {
        Self::new(Self::DEFAULT_MAX_OUTPUT_BYTES, Self::DEFAULT_TIMEOUT)
    }
}

/// A stable hash of the non-seed parts of a request.
///
/// Recorded in provenance so a saved candidate can be traced back to the exact
/// context that produced it, and usable as the prompt component of the cache
/// key described in PROJECT_CONTEXT §10 (context/model/prompt/schema/seed).
/// The seed is excluded because it is recorded separately: two requests that
/// differ only by seed share a prompt and must be visibly related.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PromptHash(u64);

impl PromptHash {
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for PromptHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

/// One bounded, self-contained generation request.
///
/// Everything a provider needs is here, and nothing else is: no callback, no
/// handle to engine state, no path, no writer. That is what lets the same
/// request be hashed for a cache, logged for a replay, and handed to a worker
/// thread or an out-of-process adapter without any of them needing to reach
/// back into the running game.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationRequest {
    kind: GenerationKind,
    scope: InstanceScope,
    context: WorldContext,
    schema_version: u32,
    seed: u64,
    budget: Budget,
}

impl GenerationRequest {
    /// Creates a request with an empty context and the default budget.
    pub fn new(kind: GenerationKind, scope: InstanceScope, schema_version: u32, seed: u64) -> Self {
        Self {
            kind,
            scope,
            context: WorldContext::new(),
            schema_version,
            seed,
            budget: Budget::default(),
        }
    }

    #[must_use]
    pub fn with_context(mut self, context: WorldContext) -> Self {
        self.context = context;
        self
    }

    #[must_use]
    pub fn with_budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    pub fn kind(&self) -> GenerationKind {
        self.kind
    }

    pub fn scope(&self) -> &InstanceScope {
        &self.scope
    }

    pub fn context(&self) -> &WorldContext {
        &self.context
    }

    /// Version of the response schema the caller will validate against
    /// downstream. Sent to the provider so an adapter can select the right
    /// structured-output grammar.
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn budget(&self) -> Budget {
        self.budget
    }

    /// Hashes kind, scope, schema version, and context, in that order.
    ///
    /// Excludes the seed (recorded separately in provenance) and the budget (a
    /// local resource policy, not part of what was asked). Every field is
    /// length-prefixed, so no rearrangement of the same bytes across fields
    /// can collide.
    pub fn prompt_hash(&self) -> PromptHash {
        let mut state = hash_init();
        state = hash_field(state, self.kind.as_str().as_bytes());
        state = hash_field(state, self.scope.as_str().as_bytes());
        state = hash_u64(state, u64::from(self.schema_version));
        state = hash_u64(state, self.context.len() as u64);
        for fact in self.context.facts() {
            state = hash_field(state, fact.subject().as_bytes());
            state = hash_field(state, fact.predicate().as_bytes());
            state = hash_field(state, fact.value().as_bytes());
        }
        PromptHash(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context_of(pairs: &[(&str, &str, &str)]) -> WorldContext {
        let mut context = WorldContext::new();
        for (subject, predicate, value) in pairs {
            let pushed = context.push(WorldFact::new(*subject, *predicate, *value));
            assert!(pushed.is_ok());
        }
        context
    }

    fn request() -> GenerationRequest {
        GenerationRequest::new(GenerationKind::Quest, InstanceScope::new("crypt-01"), 1, 7)
            .with_context(context_of(&[("npc-aldric", "faction", "wardens")]))
    }

    #[test]
    fn prompt_hash_is_stable_for_the_same_request() {
        assert_eq!(request().prompt_hash(), request().prompt_hash());
    }

    #[test]
    fn prompt_hash_ignores_seed_and_budget() {
        let base = request();
        let reseeded = request().with_budget(Budget::new(16, Duration::from_millis(1)));
        let other_seed =
            GenerationRequest::new(GenerationKind::Quest, InstanceScope::new("crypt-01"), 1, 999)
                .with_context(context_of(&[("npc-aldric", "faction", "wardens")]));
        assert_eq!(base.prompt_hash(), reseeded.prompt_hash());
        assert_eq!(base.prompt_hash(), other_seed.prompt_hash());
    }

    #[test]
    fn prompt_hash_distinguishes_kind_scope_schema_and_context() {
        let base = request().prompt_hash();
        let other_kind =
            GenerationRequest::new(GenerationKind::Dialogue, InstanceScope::new("crypt-01"), 1, 7)
                .with_context(context_of(&[("npc-aldric", "faction", "wardens")]));
        let other_scope = GenerationRequest::new(GenerationKind::Quest, InstanceScope::new("crypt-02"), 1, 7)
            .with_context(context_of(&[("npc-aldric", "faction", "wardens")]));
        let other_schema =
            GenerationRequest::new(GenerationKind::Quest, InstanceScope::new("crypt-01"), 2, 7)
                .with_context(context_of(&[("npc-aldric", "faction", "wardens")]));
        let other_facts = request().with_context(context_of(&[("npc-aldric", "faction", "ashen")]));
        assert_ne!(base, other_kind.prompt_hash());
        assert_ne!(base, other_scope.prompt_hash());
        assert_ne!(base, other_schema.prompt_hash());
        assert_ne!(base, other_facts.prompt_hash());
    }

    #[test]
    fn prompt_hash_depends_on_fact_order() {
        let forward = request().with_context(context_of(&[("a", "p", "1"), ("b", "p", "2")]));
        let reversed = request().with_context(context_of(&[("b", "p", "2"), ("a", "p", "1")]));
        assert_ne!(forward.prompt_hash(), reversed.prompt_hash());
    }

    #[test]
    fn context_rejects_too_many_facts() {
        let mut context = WorldContext::new();
        for index in 0..WorldContext::MAX_FACTS {
            let pushed = context.push(WorldFact::new(index.to_string(), "p", "v"));
            assert!(pushed.is_ok());
        }
        assert_eq!(
            context.push(WorldFact::new("one", "too", "many")),
            Err(ContextError::TooManyFacts {
                limit: WorldContext::MAX_FACTS
            })
        );
        assert_eq!(context.len(), WorldContext::MAX_FACTS);
    }

    #[test]
    fn context_rejects_an_oversized_fact() {
        let mut context = WorldContext::new();
        let huge = "x".repeat(WorldContext::MAX_FACT_BYTES + 1);
        let actual = huge.len() + "p".len() + "s".len();
        assert_eq!(
            context.push(WorldFact::new("s", "p", huge)),
            Err(ContextError::FactTooLong {
                limit: WorldContext::MAX_FACT_BYTES,
                actual,
            })
        );
        assert!(context.is_empty(), "a rejected fact must not be stored");
    }

    #[test]
    fn generation_kind_wire_names_are_unique_and_kebab_case() {
        let kinds = [
            GenerationKind::Quest,
            GenerationKind::Dialogue,
            GenerationKind::MapSemantics,
            GenerationKind::EnemyVariant,
            GenerationKind::EncounterComposition,
            GenerationKind::RewardFlavor,
            GenerationKind::VisualRecipe,
        ];
        let mut names: Vec<&str> = kinds.iter().map(|kind| kind.as_str()).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count);
        for kind in kinds {
            let name = kind.as_str();
            assert!(
                name.bytes().all(|byte| byte.is_ascii_lowercase() || byte == b'-'),
                "{name} must be lowercase kebab-case"
            );
            assert_eq!(kind.to_string(), name);
        }
    }

    #[test]
    fn prompt_hash_displays_as_padded_hex() {
        assert_eq!(PromptHash(0xabc).to_string(), "0000000000000abc");
        assert_eq!(PromptHash(0xabc).as_u64(), 0xabc);
    }
}
