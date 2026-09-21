//! Small deterministic hash and PRNG primitives.
//!
//! These exist so the crate can hash a request and drive the mock's output
//! without a dependency. A general-purpose hashing crate would be a heavier
//! commitment than the job needs, and `std`'s `DefaultHasher` is explicitly
//! not stable across releases — a prompt hash that changes when the toolchain
//! changes would invalidate every save's provenance and every cache entry.
//! FNV-1a and SplitMix64 are both fully specified by these few lines, so the
//! values they produce are reproducible forever and across platforms.

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Starting state for a hash chain.
pub(crate) const fn hash_init() -> u64 {
    FNV_OFFSET_BASIS
}

/// Folds `bytes` into the running hash.
pub(crate) fn hash_bytes(state: u64, bytes: &[u8]) -> u64 {
    let mut state = state;
    for byte in bytes {
        state ^= u64::from(*byte);
        state = state.wrapping_mul(FNV_PRIME);
    }
    state
}

/// Folds a length-prefixed field into the running hash.
///
/// The length prefix is what makes the hash *structural* rather than a hash of
/// concatenated text. Without it, the fact pair `("npc", "aldric")` and the
/// fact pair `("npca", "ldric")` would hash identically, and two different
/// world contexts would silently share a cache entry and a provenance record.
pub(crate) fn hash_field(state: u64, bytes: &[u8]) -> u64 {
    let state = hash_u64(state, bytes.len() as u64);
    hash_bytes(state, bytes)
}

/// Folds a `u64` into the running hash in fixed little-endian byte order, so
/// the result does not depend on the host's endianness.
pub(crate) fn hash_u64(state: u64, value: u64) -> u64 {
    hash_bytes(state, &value.to_le_bytes())
}

/// SplitMix64: a deterministic sequence from a 64-bit state.
///
/// Used only to shape mock output. It is not a cryptographic generator and
/// nothing security-relevant may depend on it.
#[derive(Debug, Clone)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Picks an index below `len`, saturating at zero for an empty range so
    /// callers can never construct an out-of-bounds index.
    pub(crate) fn index(&mut self, len: usize) -> usize {
        if len == 0 {
            return 0;
        }
        (self.next_u64() % len as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashing_is_order_sensitive() {
        let a = hash_field(hash_field(hash_init(), b"alpha"), b"beta");
        let b = hash_field(hash_field(hash_init(), b"beta"), b"alpha");
        assert_ne!(a, b);
    }

    #[test]
    fn length_prefix_separates_field_boundaries() {
        let a = hash_field(hash_field(hash_init(), b"npc"), b"aldric");
        let b = hash_field(hash_field(hash_init(), b"npca"), b"ldric");
        assert_ne!(a, b, "field boundaries must be part of the hash");
    }

    #[test]
    fn hashing_is_reproducible() {
        let once = hash_u64(hash_field(hash_init(), b"quest"), 7);
        let twice = hash_u64(hash_field(hash_init(), b"quest"), 7);
        assert_eq!(once, twice);
    }

    #[test]
    fn splitmix_is_reproducible_and_varies_by_seed() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(1);
        let mut c = SplitMix64::new(2);
        let a_seq: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let b_seq: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        let c_seq: Vec<u64> = (0..8).map(|_| c.next_u64()).collect();
        assert_eq!(a_seq, b_seq);
        assert_ne!(a_seq, c_seq);
    }

    #[test]
    fn index_of_empty_range_is_zero() {
        let mut rng = SplitMix64::new(42);
        assert_eq!(rng.index(0), 0);
    }
}
