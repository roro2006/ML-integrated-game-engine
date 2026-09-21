//! The provider boundary: one call, one bounded answer, or one typed failure.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::request::{Budget, GenerationRequest, PromptHash};

/// Why a generation call did not produce a usable candidate.
///
/// Every variant is a normal outcome, not an exception. Generation is an
/// enhancement (PROJECT_CONTEXT §10), so the caller's response to all of these
/// is the same in kind — fall back to a deterministic template and carry on —
/// and the variants exist to inform *how*, not *whether*, to recover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderError {
    /// No provider is configured, or the configured one cannot be reached.
    ///
    /// Actionable as: use templates, and stop asking for the rest of the
    /// session (or until configuration changes). Retrying a missing model at
    /// each safe point only wastes time.
    Unavailable { detail: String },
    /// The budget's deadline elapsed before an answer arrived.
    ///
    /// Actionable as: use templates now; a later attempt at a quieter safe
    /// point, or with a smaller context, may still succeed. The partial work
    /// is discarded — there is no such thing as a partly accepted candidate.
    Timeout { budget: std::time::Duration },
    /// The response exceeded the request's output budget.
    ///
    /// Actionable as: use templates, and treat the request as too open-ended.
    /// The oversized bytes are dropped rather than truncated, because a
    /// truncated candidate is a *malformed* candidate that happens to parse
    /// sometimes, which is worse than no candidate at all.
    OutputTooLarge { limit: usize, actual: usize },
    /// The provider answered, but not with usable text — a truncated stream,
    /// a non-UTF-8 body, or a transport-level framing error.
    ///
    /// Actionable as: the one case where a bounded repair attempt is worth
    /// making (PROJECT_CONTEXT §9), then templates. See
    /// [`ProviderError::permits_repair`].
    Malformed { detail: String },
    /// A failure specific to one backend that the shared taxonomy does not
    /// describe — an HTTP status, a model-load error, a quota refusal.
    ///
    /// Actionable as: templates, plus a diagnostic worth logging locally.
    /// Kept deliberately opaque so backend-specific vocabulary does not leak
    /// into the engine's control flow.
    Provider { message: String },
}

impl ProviderError {
    /// Whether re-asking the model with validation feedback could plausibly
    /// help.
    ///
    /// Only malformed output qualifies. A repair round trip spends the same
    /// budget again, so it is worth it only when the provider demonstrably
    /// responded and merely responded badly; an unavailable, timed-out, or
    /// over-budget provider will fail the same way twice.
    pub fn permits_repair(&self) -> bool {
        matches!(self, Self::Malformed { .. })
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable { detail } => write!(f, "no AI provider available: {detail}"),
            Self::Timeout { budget } => {
                write!(f, "generation exceeded its {budget:?} budget")
            }
            Self::OutputTooLarge { limit, actual } => {
                write!(f, "response of {actual} bytes exceeds the {limit} byte budget")
            }
            Self::Malformed { detail } => write!(f, "provider returned unusable output: {detail}"),
            Self::Provider { message } => write!(f, "provider error: {message}"),
        }
    }
}

impl std::error::Error for ProviderError {}

/// Where a candidate came from.
///
/// ADR-0010 makes generated content invisible to players but fully traceable
/// to developers: saves record model, prompt hash, seed, and schema version so
/// a candidate can be explained, diffed, replayed, and rolled back long after
/// the fact. It is recorded for every candidate, including ones that are later
/// rejected, because the rejected ones are the interesting ones.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    model: String,
    prompt_hash: PromptHash,
    seed: u64,
    schema_version: u32,
}

impl Provenance {
    /// Builds provenance from the request it describes.
    ///
    /// Taking the request rather than loose fields is the point: the hash,
    /// seed, and schema version cannot drift from what was actually asked,
    /// because a provider never gets to state them independently.
    pub fn for_request(model: impl Into<String>, request: &GenerationRequest) -> Self {
        Self {
            model: model.into(),
            prompt_hash: request.prompt_hash(),
            seed: request.seed(),
            schema_version: request.schema_version(),
        }
    }

    /// Identity of the model that answered, e.g. a local GGUF file's name and
    /// quantization. Never a path — see [`ContentProvider`].
    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn prompt_hash(&self) -> PromptHash {
        self.prompt_hash
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }
}

/// A raw candidate and the record of where it came from.
///
/// # This crate does not parse candidates
///
/// `candidate` is text, and it stays text. Parsing, schema validation,
/// reference resolution, compilation, budget and progression checks, dry runs,
/// and activation all happen downstream, in crates that own the content model
/// (PROJECT_CONTEXT §9). That boundary is worth defending: if the gateway
/// parsed candidates, it would need to know the content schemas, and the
/// single place that decides whether content is admissible would become two
/// places that can disagree. The gateway's entire job is to obtain bytes
/// within a budget and say honestly where they came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationResponse {
    candidate: String,
    provenance: Provenance,
}

impl GenerationResponse {
    /// Builds a response, enforcing the output budget.
    ///
    /// This is the only constructor, so an over-budget response cannot exist:
    /// the check is not something an adapter author can forget, and there is
    /// no truncating variant to reach for. An oversized answer is a
    /// [`ProviderError::OutputTooLarge`] and the bytes are dropped.
    pub fn new(
        candidate: impl Into<String>,
        provenance: Provenance,
        budget: Budget,
    ) -> Result<Self, ProviderError> {
        let candidate = candidate.into();
        let actual = candidate.len();
        let limit = budget.max_output_bytes();
        if actual > limit {
            return Err(ProviderError::OutputTooLarge { limit, actual });
        }
        Ok(Self {
            candidate,
            provenance,
        })
    }

    /// The unparsed candidate text.
    pub fn candidate(&self) -> &str {
        &self.candidate
    }

    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// Consumes the response, yielding the candidate text and its provenance
    /// so a downstream validator can take ownership of both without cloning.
    pub fn into_parts(self) -> (String, Provenance) {
        (self.candidate, self.provenance)
    }
}

/// An engine-owned source of content candidates.
///
/// One method, taking data and returning data. Implementations may not execute
/// code, run commands, read or write files, or read environment secrets: the
/// request carries no path and no handle precisely so that a conforming
/// implementation has nothing to open (ADR-0010; PROJECT_CONTEXT §9).
///
/// # Why synchronous
///
/// Generation runs on a dedicated worker, never on the frame or fixed-step
/// thread (PROJECT_CONTEXT §10), so this call is allowed to be a blocking call
/// — the thread it blocks is the one whose whole purpose is to wait. A
/// synchronous signature keeps the engine free of an async runtime dependency
/// and keeps the trait implementable by an adapter that is itself a blocking
/// HTTP request.
///
/// # Why this shape suits an out-of-process HTTP provider
///
/// - **Bounded.** [`Budget::timeout`] maps directly onto connect/read timeouts
///   and [`Budget::max_output_bytes`] onto a capped response reader, so an
///   adapter can honor both without cooperation from the caller.
/// - **Cancellable by deadline.** The only cancellation mechanism is the
///   budget, which every implementation already has. The caller never needs a
///   cancel handle, so it can never leak one, and abandoning a request is
///   always safe.
/// - **No shared mutable global state.** `&self` means a connection pool or
///   session handle must be declared as interior state rather than assumed,
///   and `Send + Sync` lets one provider be shared by, or moved to, the
///   generation worker. Two calls with the same request must not influence
///   each other.
///
/// # Determinism
///
/// Providers are not required to be deterministic — a real model is not — but
/// they must be *reproducible in provenance*: the recorded prompt hash, seed,
/// and schema version must describe the request that was actually sent.
pub trait ContentProvider: fmt::Debug + Send + Sync {
    /// Produces one candidate for `request`, or a typed reason it could not.
    ///
    /// Implementations must return within the request's budget and must never
    /// panic; a caller treats every error as "use the deterministic template".
    fn generate(&self, request: &GenerationRequest) -> Result<GenerationResponse, ProviderError>;
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::request::{GenerationKind, InstanceScope};

    fn request() -> GenerationRequest {
        GenerationRequest::new(GenerationKind::Quest, InstanceScope::new("crypt-01"), 3, 11)
    }

    #[test]
    fn provenance_mirrors_the_request() {
        let request = request();
        let provenance = Provenance::for_request("mock", &request);
        assert_eq!(provenance.model(), "mock");
        assert_eq!(provenance.prompt_hash(), request.prompt_hash());
        assert_eq!(provenance.seed(), 11);
        assert_eq!(provenance.schema_version(), 3);
    }

    #[test]
    fn response_rejects_output_over_budget_without_truncating() {
        let request = request().with_budget(Budget::new(8, Duration::from_secs(1)));
        let provenance = Provenance::for_request("mock", &request);
        let result = GenerationResponse::new("0123456789", provenance, request.budget());
        assert_eq!(
            result,
            Err(ProviderError::OutputTooLarge { limit: 8, actual: 10 })
        );
    }

    #[test]
    fn response_accepts_output_exactly_at_the_limit() {
        let request = request().with_budget(Budget::new(4, Duration::from_secs(1)));
        let provenance = Provenance::for_request("mock", &request);
        match GenerationResponse::new("abcd", provenance, request.budget()) {
            Ok(response) => assert_eq!(response.candidate(), "abcd"),
            Err(error) => panic!("boundary value must be accepted, got {error}"),
        }
    }

    #[test]
    fn only_malformed_output_permits_repair() {
        assert!(
            ProviderError::Malformed {
                detail: "truncated".into()
            }
            .permits_repair()
        );
        for error in [
            ProviderError::Unavailable {
                detail: "not configured".into(),
            },
            ProviderError::Timeout {
                budget: Duration::from_secs(1),
            },
            ProviderError::OutputTooLarge { limit: 1, actual: 2 },
            ProviderError::Provider {
                message: "http 503".into(),
            },
        ] {
            assert!(!error.permits_repair(), "{error} must not permit repair");
        }
    }

    #[test]
    fn errors_describe_themselves() {
        let error = ProviderError::OutputTooLarge { limit: 8, actual: 10 };
        let text = error.to_string();
        assert!(text.contains('8') && text.contains("10"), "{text}");
        let as_std_error: &dyn std::error::Error = &error;
        assert_eq!(as_std_error.to_string(), text);
    }

    #[test]
    fn into_parts_preserves_candidate_and_provenance() {
        let request = request();
        let provenance = Provenance::for_request("mock", &request);
        match GenerationResponse::new("candidate", provenance.clone(), request.budget()) {
            Ok(response) => {
                let (candidate, recorded) = response.into_parts();
                assert_eq!(candidate, "candidate");
                assert_eq!(recorded, provenance);
            }
            Err(error) => panic!("in-budget response must build, got {error}"),
        }
    }
}
