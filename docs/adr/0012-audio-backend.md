# ADR-0012: Audio backend — cpal with an engine-owned mixer

- Status: Accepted
- Date: 2026-09-15

## Context

Open decision §16.7. Audio is not needed until the playable slice (M5), but the
choice determines who owns mixing, buses, and latency policy, so it was settled
with the other backend decisions.

## Decision

Use `cpal` for audio device access only. The engine owns everything above the
device:

- Game-facing audio **events** (play sound for a domain/gameplay event), not
  direct device calls.
- A mixer with categories/buses: music, SFX, UI, and ambience, each separately
  attenuated, with a master volume.
- Voice management: a fixed voice budget, oldest/quietest voice stealing, and
  per-sound limits so a busy fight cannot exhaust the mixer.
- Asset references by stable ID through the content registry, like every other
  asset.

Audio decoding needs a focused decoder crate (Symphonia, or smaller
format-specific crates). That is a separate dependency decision to be made and
approved when audio work actually starts, following ADR-0001's rules.

`cpal` is added when audio work begins, not before.

## Consequences

- Mixing, resampling, and voice management are our code. This is genuinely the
  most real-time-sensitive code in the project: the audio callback runs on its
  own thread with a hard deadline and must never allocate, lock, or block. It
  needs a lock-free command queue from the game thread.
- In exchange, latency, bus behaviour, and budget enforcement are fully under
  project control, consistent with how windowing and rendering are handled.
- Audio is deferred until M5, so none of this blocks current milestones.

## Alternatives considered

- **kira**: a game audio engine with mixing, tweens, and sound instances built
  in. It would save the hardest part of this work, and it is the pragmatic
  choice if the mixer becomes a time sink — but it owns engine-level behaviour
  the project has consistently kept in-house.
- **rodio**: quickest path to a sound playing, with the least control over
  buses, latency, and voice limits.
