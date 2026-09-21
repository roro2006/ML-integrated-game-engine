//! A deterministic, scriptable provider that needs no model.
//!
//! Its purpose is to let every downstream consumer — the generation queue,
//! validators, the content compiler, activation, save provenance — be written
//! and tested now, years before a GGUF file exists, and to make the *failure*
//! paths as easy to test as the success path. Fallback code that has never run
//! is fallback code that does not work, and the fallback path is the one the
//! minimum-spec player will actually be on.

use std::fmt::Write as _;

use crate::hash::{SplitMix64, hash_u64};
use crate::provider::{ContentProvider, GenerationResponse, Provenance, ProviderError};
use crate::request::GenerationRequest;

/// What the mock does when asked to generate.
///
/// One variant per outcome a caller must handle, so a test can name the
/// situation it is exercising instead of contriving it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MockBehavior {
    /// Return a deterministic candidate derived from the request.
    #[default]
    Succeed,
    /// Report the deadline as elapsed — without sleeping. Simulated because a
    /// test suite that really waited out a 30 second budget would be a test
    /// suite nobody runs, and because sleeping would make the mock's behavior
    /// depend on the clock.
    Timeout,
    /// Report that no provider is configured or reachable.
    Unavailable,
    /// Report output that arrived but is unusable at the transport level.
    Malformed,
    /// Attempt a response one byte over the request's output budget, which the
    /// budget check then rejects.
    Oversized,
}

/// A provider with no model, no network, and no I/O.
///
/// Determinism is the contract: the same request always produces byte-
/// identical output, because the candidate is a pure function of the request's
/// prompt hash and seed. Nothing else is consulted — no clock, no environment,
/// no filesystem, no global state — so a replay from a seed reproduces exactly
/// what the first run saw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockProvider {
    model: String,
    behavior: MockBehavior,
}

impl MockProvider {
    /// Model identity recorded in provenance, so a save written against the
    /// mock is never mistaken for one written against a real model.
    pub const MODEL: &'static str = "mock-deterministic-1";

    /// Largest synthetic response the mock will allocate, so a caller that
    /// sets an enormous output budget cannot make [`MockBehavior::Oversized`]
    /// exhaust memory. Above this the mock reports the same error without
    /// building the bytes.
    const MAX_SYNTHETIC_BYTES: usize = 1 << 20;

    pub fn new() -> Self {
        Self {
            model: Self::MODEL.to_string(),
            behavior: MockBehavior::Succeed,
        }
    }

    /// Scripts the outcome of every subsequent call.
    #[must_use]
    pub fn with_behavior(mut self, behavior: MockBehavior) -> Self {
        self.behavior = behavior;
        self
    }

    /// Overrides the recorded model identity, for tests that assert
    /// provenance distinguishes one model from another.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn behavior(&self) -> MockBehavior {
        self.behavior
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// Builds the deterministic candidate text for a request.
    ///
    /// Shaped like the JSON a structured-output backend would return, so
    /// downstream parsers have something realistic to chew on. It deliberately
    /// never echoes caller-supplied fact text: quoting someone else's strings
    /// into JSON is how a mock starts emitting invalid output for reasons that
    /// have nothing to do with the test at hand.
    fn candidate(request: &GenerationRequest) -> String {
        const ADJECTIVES: [&str; 8] = [
            "ashen", "hollow", "gilded", "sunken", "quiet", "riven", "pale", "brackish",
        ];
        const NOUNS: [&str; 8] = [
            "vigil",
            "reliquary",
            "causeway",
            "cistern",
            "bastion",
            "hearth",
            "hollow",
            "verge",
        ];
        const BEATS: [&str; 6] = ["arrive", "discover", "bargain", "descend", "confront", "withdraw"];

        let mut rng = SplitMix64::new(hash_u64(request.prompt_hash().as_u64(), request.seed()));
        let title = format!(
            "{}-{}",
            ADJECTIVES[rng.index(ADJECTIVES.len())],
            NOUNS[rng.index(NOUNS.len())]
        );
        let beat_count = 2 + rng.index(3);
        let mut beats = String::new();
        for index in 0..beat_count {
            if index > 0 {
                beats.push(',');
            }
            // Writing into a String is infallible; the result is discarded
            // rather than unwrapped to keep the mock panic-free by
            // construction.
            let _ = write!(beats, "\"{}\"", BEATS[rng.index(BEATS.len())]);
        }

        let mut candidate = String::new();
        let _ = write!(
            candidate,
            concat!(
                "{{\"kind\":\"{}\",\"schema_version\":{},\"scope\":\"{}\",",
                "\"title\":\"{}\",\"beats\":[{}],\"facts_used\":{}}}"
            ),
            request.kind().as_str(),
            request.schema_version(),
            request.scope().as_str(),
            title,
            beats,
            request.context().len()
        );
        candidate
    }
}

impl Default for MockProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl ContentProvider for MockProvider {
    fn generate(&self, request: &GenerationRequest) -> Result<GenerationResponse, ProviderError> {
        let provenance = Provenance::for_request(self.model.clone(), request);
        let budget = request.budget();
        match self.behavior {
            // A tiny budget can still reject this candidate; that is the
            // budget doing its job, not a mock failure.
            MockBehavior::Succeed => GenerationResponse::new(Self::candidate(request), provenance, budget),
            MockBehavior::Timeout => Err(ProviderError::Timeout {
                budget: budget.timeout(),
            }),
            MockBehavior::Unavailable => Err(ProviderError::Unavailable {
                detail: "mock provider scripted as unavailable".to_string(),
            }),
            MockBehavior::Malformed => Err(ProviderError::Malformed {
                detail: "mock provider scripted to return unusable output".to_string(),
            }),
            MockBehavior::Oversized => {
                let limit = budget.max_output_bytes();
                match limit.checked_add(1) {
                    Some(len) if len <= Self::MAX_SYNTHETIC_BYTES => {
                        GenerationResponse::new("x".repeat(len), provenance, budget)
                    }
                    _ => Err(ProviderError::OutputTooLarge {
                        limit,
                        actual: limit.saturating_add(1),
                    }),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;
    use crate::request::{Budget, GenerationKind, InstanceScope, WorldContext, WorldFact};

    fn context_of(pairs: &[(&str, &str, &str)]) -> WorldContext {
        let mut context = WorldContext::new();
        for (subject, predicate, value) in pairs {
            let pushed = context.push(WorldFact::new(*subject, *predicate, *value));
            assert!(pushed.is_ok());
        }
        context
    }

    fn request(seed: u64) -> GenerationRequest {
        GenerationRequest::new(GenerationKind::Quest, InstanceScope::new("crypt-01"), 1, seed)
            .with_context(context_of(&[("npc-aldric", "faction", "wardens")]))
    }

    fn candidate_of(provider: &MockProvider, request: &GenerationRequest) -> String {
        match provider.generate(request) {
            Ok(response) => response.candidate().to_string(),
            Err(error) => panic!("expected success, got {error}"),
        }
    }

    #[test]
    fn same_request_yields_byte_identical_output() {
        let provider = MockProvider::new();
        let request = request(7);
        let first = candidate_of(&provider, &request);
        let second = candidate_of(&provider, &request);
        assert_eq!(first, second);
        // And across provider instances, since the mock holds no state.
        assert_eq!(first, candidate_of(&MockProvider::new(), &request));
    }

    #[test]
    fn a_different_seed_yields_different_output() {
        let provider = MockProvider::new();
        assert_ne!(
            candidate_of(&provider, &request(7)),
            candidate_of(&provider, &request(8))
        );
    }

    #[test]
    fn context_content_changes_the_output() {
        let provider = MockProvider::new();
        let base = request(7);
        let changed = request(7).with_context(context_of(&[("npc-aldric", "faction", "ashen")]));
        assert_ne!(candidate_of(&provider, &base), candidate_of(&provider, &changed));
    }

    #[test]
    fn context_order_changes_the_output() {
        let provider = MockProvider::new();
        let forward = request(7).with_context(context_of(&[("a", "p", "1"), ("b", "p", "2")]));
        let reversed = request(7).with_context(context_of(&[("b", "p", "2"), ("a", "p", "1")]));
        assert_ne!(
            candidate_of(&provider, &forward),
            candidate_of(&provider, &reversed)
        );
    }

    #[test]
    fn kind_and_scope_reach_the_candidate() {
        let provider = MockProvider::new();
        let request =
            GenerationRequest::new(GenerationKind::EnemyVariant, InstanceScope::new("marsh-04"), 2, 1);
        let candidate = candidate_of(&provider, &request);
        assert!(candidate.contains("enemy-variant"), "{candidate}");
        assert!(candidate.contains("marsh-04"), "{candidate}");
        assert!(candidate.contains("\"schema_version\":2"), "{candidate}");
    }

    #[test]
    fn scripted_timeout_reports_the_budget_without_sleeping() {
        let provider = MockProvider::new().with_behavior(MockBehavior::Timeout);
        let budget = Budget::new(4096, Duration::from_secs(3600));
        let request = request(7).with_budget(budget);
        let started = Instant::now();
        let result = provider.generate(&request);
        let elapsed = started.elapsed();
        assert_eq!(
            result,
            Err(ProviderError::Timeout {
                budget: Duration::from_secs(3600)
            })
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "timeout must be simulated, not waited out (took {elapsed:?})"
        );
    }

    #[test]
    fn scripted_unavailable_reports_unavailable() {
        let provider = MockProvider::new().with_behavior(MockBehavior::Unavailable);
        match provider.generate(&request(7)) {
            Err(error @ ProviderError::Unavailable { .. }) => assert!(!error.permits_repair()),
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }

    #[test]
    fn scripted_malformed_reports_malformed_and_permits_one_repair() {
        let provider = MockProvider::new().with_behavior(MockBehavior::Malformed);
        match provider.generate(&request(7)) {
            Err(error @ ProviderError::Malformed { .. }) => assert!(error.permits_repair()),
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn scripted_oversized_is_rejected_rather_than_truncated() {
        let provider = MockProvider::new().with_behavior(MockBehavior::Oversized);
        let request = request(7).with_budget(Budget::new(128, Duration::from_secs(1)));
        assert_eq!(
            provider.generate(&request),
            Err(ProviderError::OutputTooLarge {
                limit: 128,
                actual: 129
            })
        );
    }

    #[test]
    fn scripted_oversized_allocates_nothing_for_an_absurd_budget() {
        let provider = MockProvider::new().with_behavior(MockBehavior::Oversized);
        let request = request(7).with_budget(Budget::new(usize::MAX, Duration::from_secs(1)));
        assert_eq!(
            provider.generate(&request),
            Err(ProviderError::OutputTooLarge {
                limit: usize::MAX,
                actual: usize::MAX
            })
        );
    }

    #[test]
    fn provenance_is_populated_and_tracks_the_request() {
        let provider = MockProvider::new().with_model("mock-alt");
        let first = request(7);
        let second = request(8);
        match (provider.generate(&first), provider.generate(&second)) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a.provenance().model(), "mock-alt");
                assert_eq!(a.provenance().seed(), 7);
                assert_eq!(a.provenance().schema_version(), 1);
                assert_eq!(a.provenance().prompt_hash(), first.prompt_hash());
                // Same prompt, different seed: hashes match, outputs do not.
                assert_eq!(a.provenance().prompt_hash(), b.provenance().prompt_hash());
                assert_ne!(a.candidate(), b.candidate());
            }
            other => panic!("expected two successes, got {other:?}"),
        }
    }

    #[test]
    fn provenance_prompt_hash_differs_for_different_requests() {
        let provider = MockProvider::new();
        let quest = request(7);
        let dialogue = GenerationRequest::new(GenerationKind::Dialogue, InstanceScope::new("crypt-01"), 1, 7)
            .with_context(context_of(&[("npc-aldric", "faction", "wardens")]));
        match (provider.generate(&quest), provider.generate(&dialogue)) {
            (Ok(a), Ok(b)) => assert_ne!(a.provenance().prompt_hash(), b.provenance().prompt_hash()),
            other => panic!("expected two successes, got {other:?}"),
        }
    }

    #[test]
    fn calling_a_provider_requires_no_filesystem_path() {
        // There is no direct assertion that a function performs no I/O, so the
        // guarantee is structural instead: the whole call is expressible from
        // literals. No `Path`, `PathBuf`, file handle, URL, or environment
        // lookup appears in the request, the trait, or the response, so a
        // conforming provider has nothing to open. The mock additionally reads
        // nothing but its argument, which keeps it a pure function.
        let provider: &dyn ContentProvider = &MockProvider::new();
        let request = GenerationRequest::new(
            GenerationKind::VisualRecipe,
            InstanceScope::new("crypt-01"),
            1,
            42,
        )
        .with_context(context_of(&[("region-fen", "biome", "marsh")]));
        assert!(provider.generate(&request).is_ok());
    }

    #[test]
    fn mock_is_shareable_across_threads() {
        // The trait's Send + Sync bound is what lets one provider live on the
        // generation worker; assert the mock actually satisfies it.
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockProvider>();
    }

    #[test]
    fn every_behavior_returns_without_panicking() {
        let behaviors = [
            MockBehavior::Succeed,
            MockBehavior::Timeout,
            MockBehavior::Unavailable,
            MockBehavior::Malformed,
            MockBehavior::Oversized,
        ];
        for behavior in behaviors {
            let provider = MockProvider::new().with_behavior(behavior);
            let result = provider.generate(&request(3));
            assert_eq!(result.is_ok(), behavior == MockBehavior::Succeed);
        }
        assert_eq!(MockBehavior::default(), MockBehavior::Succeed);
        assert_eq!(MockProvider::default().behavior(), MockBehavior::Succeed);
        assert_eq!(MockProvider::default().model(), MockProvider::MODEL);
    }
}
