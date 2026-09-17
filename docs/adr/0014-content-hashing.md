# ADR-0014: Content hashing — BLAKE3

- Status: Accepted
- Date: 2026-09-16

## Context

`game_core::ids::ContentHash` is a 32-byte value that identifies a compiled
content revision. It appears in the save envelope (ADR-0013) and in AI
provenance (ADR-0010), and it answers one question: *is the content this world
was validated against the same content we have now?*

`game_core` only carries and validates the value, so nothing had to choose an
algorithm until the content pipeline began computing real hashes. Both BLAKE3
and SHA-256 produce 32 bytes, so the type does not change either way.

## Decision

Use `blake3` in `crates/content_runtime`, the only crate that computes hashes.

The hash covers the **compiled registry**, not the raw files: definitions are
serialized in sorted-id order, family by family, and hashed as one stream. That
makes the hash a property of the content's *meaning* rather than of its
formatting — reindenting a RON file or splitting one family across two files
does not change it, while renaming an id or altering a value does.

Rationale for BLAKE3 over the alternatives:

- **Speed** is the deciding factor and the reason it was chosen. BLAKE3 is
  SIMD-accelerated and internally parallel, several times faster than SHA-256
  on the same hardware. Content hashing runs on every load and every content
  revision, including on low-spec targets.
- **Collision resistance still matters.** A content hash decides whether a save
  is revalidated, so a non-cryptographic hash (FNV, xxHash) is the wrong tool
  even though it would be faster still.
- Pure Rust, no C toolchain, and it builds cleanly for both first-class targets.

## Consequences

- One more third-party crate, contained in `content_runtime` behind
  `ContentHash`. `game_core` stays algorithm-agnostic, so a future change is a
  one-crate edit.
- **Stored hashes are algorithm-specific and carry no algorithm tag.** Changing
  algorithm later would make every stored hash mismatch. That is survivable by
  design — a mismatch triggers revalidation of the save against current
  content, not data loss — but it is a real migration cost and the reason not
  to change casually.
- BLAKE3 pulls in SIMD backends and adds some compile time. Acceptable for a
  crate that is not in the hot render path.
- Determinism is a property of *our* serialization, not of the hash: the
  sorted-order rule above is what makes two builds agree, and it is tested.

## Alternatives considered

- **SHA-256 (`sha2`)**: the conservative, ubiquitous choice, and interoperable
  with external tooling. Rejected because nothing here needs to interoperate,
  and it is materially slower for no benefit this project can use.
- **A non-cryptographic hash (xxHash, FNV)**: fastest, and already used
  internally for the AI gateway's prompt hashes where collisions are harmless.
  Rejected here because a content hash is a correctness signal.
- **Hashing raw file bytes** instead of the compiled registry: simpler, but it
  would make a whitespace edit look like a content change and invalidate saves
  for no reason.
