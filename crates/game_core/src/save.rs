//! Versioned save envelope.
//!
//! Wraps canonical state with the format version, content revision and hash,
//! world seed, and provenance needed to validate and migrate it. Specified by
//! ADR-0013; this module implements that decision.
//!
//! # Why the envelope is parsed in two steps
//!
//! [`peek_version`] reads only `format_version` before anything else is
//! touched. A save written by a *newer* build must be refused with a clear
//! message and the file left exactly as it was — parsing it into today's types
//! first would either fail with a confusing error or, worse, succeed against a
//! shape that has since changed meaning.
//!
//! # Why migrations work on `ron::Value`
//!
//! Typed structs describe only the *current* shape. A migration written
//! against them stops compiling the moment those types move on, which is how
//! migration chains rot: the usual fix is to delete the old migration, and the
//! old saves with it. Migrating an untyped value keeps every historical shape
//! expressible forever.
//!
//! # Why unknown fields are rejected
//!
//! Silently ignoring a field is how player state disappears: a build that does
//! not understand `settlement_upgrades` would load the save, drop them, and
//! write them back out as gone. Rejecting forces a version bump and a
//! migration, which is friction on purpose.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::ids::{ContentHash, ContentRevisionId};
use crate::world::WorldState;

/// Version of the envelope itself.
///
/// Distinct from [`crate::world::WORLD_SCHEMA_VERSION`]: the envelope can gain
/// a field without the state changing, and vice versa.
pub const SAVE_FORMAT_VERSION: u32 = 2;

/// Where a candidate piece of content came from, carried with the save.
///
/// Recorded even though players never see it (ADR-0010 keeps AI invisible):
/// reproducing a bug in generated content needs the model, seed, and hashes
/// that produced it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    /// Model identity, empty when the content was authored or templated.
    pub model: String,
    /// Hash of the prompt that produced the candidate, if any.
    pub prompt_hash: Option<String>,
    /// Seed the candidate was generated from.
    pub seed: Option<u64>,
    /// Schema version the candidate was validated against.
    pub schema_version: Option<u32>,
}

/// A save file: canonical state plus everything needed to validate it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SaveEnvelope {
    /// Envelope version. Checked before anything else is parsed.
    pub format_version: u32,
    /// Engine version that wrote the file, for bug reports rather than logic.
    pub engine_version: String,
    /// Compiled content revision this state was validated against.
    pub content_revision: ContentRevisionId,
    /// Hash of that compiled content, so a mismatch is detectable.
    pub content_hash: ContentHash,
    /// Provenance of generated content accepted into this world.
    pub provenance: Vec<Provenance>,
    /// The world itself.
    pub world: WorldState,
}

/// Just enough of an envelope to read its version without committing to a shape.
#[derive(Debug, Deserialize)]
struct VersionProbe {
    format_version: u32,
}

#[derive(Debug)]
pub enum SaveError {
    /// The file could not be read or written.
    Io { path: PathBuf, source: io::Error },
    /// The file is not valid RON, or does not match the expected shape.
    Parse { path: PathBuf, message: String },
    /// Serializing the envelope failed.
    Serialize(String),
    /// The save was written by a newer build. The file is left untouched.
    FutureVersion { found: u32, supported: u32 },
    /// No migration path exists from this version.
    NoMigration { from: u32, to: u32 },
    /// A migration produced something that no longer parses.
    MigrationFailed { from: u32, message: String },
    /// The loaded world contains an instance-scoped id (ADR-0010).
    NotCanonical(String),
}

impl fmt::Display for SaveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "cannot access {}: {source}", path.display()),
            Self::Parse { path, message } => write!(f, "cannot parse {}: {message}", path.display()),
            Self::Serialize(message) => write!(f, "cannot serialize save: {message}"),
            Self::FutureVersion { found, supported } => write!(
                f,
                "save format version {found} is newer than this build supports ({supported}); \
                 the file has not been modified"
            ),
            Self::NoMigration { from, to } => {
                write!(f, "no migration from save format version {from} to {to}")
            }
            Self::MigrationFailed { from, message } => {
                write!(f, "migration from version {from} failed: {message}")
            }
            Self::NotCanonical(message) => {
                write!(f, "save contains an instance-scoped id: {message}")
            }
        }
    }
}

impl std::error::Error for SaveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl SaveEnvelope {
    /// Wraps a world for saving at the current format version.
    pub fn new(world: WorldState, engine_version: impl Into<String>, content_hash: ContentHash) -> Self {
        Self {
            format_version: SAVE_FORMAT_VERSION,
            engine_version: engine_version.into(),
            content_revision: world.content_revision.clone(),
            content_hash,
            provenance: Vec::new(),
            world,
        }
    }

    /// Renders the envelope as pretty RON.
    pub fn to_ron(&self) -> Result<String, SaveError> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
            .map_err(|error| SaveError::Serialize(error.to_string()))
    }
}

/// Reads only the envelope's version.
///
/// Deliberately tolerant of every other field, so a future save can be
/// *identified* even though it cannot be loaded.
pub fn peek_version(text: &str) -> Result<u32, SaveError> {
    ron::from_str::<VersionProbe>(text)
        .map(|probe| probe.format_version)
        .map_err(|error| SaveError::Parse {
            path: PathBuf::from("<memory>"),
            message: error.to_string(),
        })
}

/// Parses an envelope from RON, migrating older versions forward.
///
/// Returns [`SaveError::FutureVersion`] for anything newer than this build,
/// without attempting to interpret it.
pub fn from_ron(text: &str) -> Result<SaveEnvelope, SaveError> {
    let version = peek_version(text)?;
    if version > SAVE_FORMAT_VERSION {
        return Err(SaveError::FutureVersion {
            found: version,
            supported: SAVE_FORMAT_VERSION,
        });
    }

    let parse_error = |error: ron::error::SpannedError| SaveError::Parse {
        path: PathBuf::from("<memory>"),
        message: error.to_string(),
    };
    let envelope: SaveEnvelope = if version == SAVE_FORMAT_VERSION {
        ron::from_str(text).map_err(parse_error)?
    } else {
        let migrated = migrate_to_current(text, version)?;
        ron::from_str(&migrated).map_err(|error| SaveError::MigrationFailed {
            from: version,
            message: error.to_string(),
        })?
    };

    // A save is canonical state by definition; an instance-scoped id in one
    // means generated content leaked (ADR-0010). Refuse rather than load it.
    envelope
        .world
        .validate_canonical()
        .map_err(|error| SaveError::NotCanonical(error.to_string()))?;
    Ok(envelope)
}

/// Applies each migration in turn from `from` up to the current version.
fn migrate_to_current(text: &str, from: u32) -> Result<String, SaveError> {
    let mut text = text.to_owned();
    for version in from..SAVE_FORMAT_VERSION {
        let migration = migration_for(version).ok_or(SaveError::NoMigration {
            from: version,
            to: version + 1,
        })?;
        text = migration(&text).map_err(|message| SaveError::MigrationFailed {
            from: version,
            message,
        })?;
    }
    Ok(text)
}

/// One step of the migration chain: version N's text to version N+1's.
///
/// Text rather than `ron::Value`, which ADR-0013 first chose: `ron::Value`
/// forgets struct and enum-variant names (`status: Active` reads back as a
/// unit), so a save round-tripped through it no longer parses. Text edited
/// through [`ron_text`], which walks RON's structure, keeps every byte a
/// migration does not mean to change — and still never touches the typed
/// structs, which is what keeps an old migration compiling (ADR-0018).
type Migration = fn(&str) -> Result<String, String>;

/// The migration that upgrades `from` to `from + 1`, if one exists.
fn migration_for(from: u32) -> Option<Migration> {
    // Each version adds exactly one arm here, with a fixture of the old shape
    // and a test that migrating it preserves state. Nothing may ever be
    // removed from this chain: a save at that version may still exist on
    // someone's disk.
    match from {
        1 => Some(v1_to_v2),
        _ => None,
    }
}

/// Version 1 → 2 (ADR-0018): the player's progression gains `talents`, none
/// taken, and `loadout`, `None` — which means "the class's own abilities in
/// order", so a character plays exactly as it did. Content is not needed,
/// and must not be: a migration is a pure function of the file.
fn v1_to_v2(text: &str) -> Result<String, String> {
    use ron_text::{field, inside, root};
    let envelope = inside(text, root(text).ok_or("a save is a struct")?).ok_or("a save is a struct")?;
    let version = field(text, envelope.clone(), "format_version").ok_or("no format_version")?;
    let world = field(text, envelope, "world").ok_or("no world")?;
    let world = inside(text, world).ok_or("the world is a struct")?;
    let schema = field(text, world.clone(), "schema_version").ok_or("no world schema_version")?;
    let player =
        inside(text, field(text, world, "player").ok_or("no player")?).ok_or("the player is a struct")?;
    let progression = inside(text, field(text, player, "progression").ok_or("no progression")?)
        .ok_or("the progression is a struct")?;
    for name in ["talents", "loadout"] {
        if field(text, progression.clone(), name).is_some() {
            return Err(format!("a version 1 progression has no {name}"));
        }
    }

    // The two new fields go at the end of the progression, after a comma if
    // the last field lacks one.
    let body = text[progression.clone()].trim_end();
    let added = if body.is_empty() || body.ends_with(',') {
        "talents: [], loadout: None,"
    } else {
        ", talents: [], loadout: None,"
    };
    let at = progression.start + body.len();

    // Applied back to front, so each edit's offsets are still true.
    let mut edits = [(version, "2"), (schema, "2"), (at..at, added)];
    edits.sort_by(|a, b| b.0.start.cmp(&a.0.start));
    let mut out = text.to_owned();
    for (range, replacement) in edits {
        out.replace_range(range, replacement);
    }
    Ok(out)
}

/// Just enough of RON's grammar to find a field's value by name without
/// parsing it into types: strings, characters, and comments are skipped, and
/// brackets are balanced.
mod ron_text {
    use std::ops::Range;

    /// The first value in `text`: the whole document.
    pub(super) fn root(text: &str) -> Option<Range<usize>> {
        let start = skip_blank(text, 0);
        let end = value_end(text, start)?;
        (end > start).then_some(start..end)
    }

    /// Inside a struct value's parentheses, which may follow a struct name.
    pub(super) fn inside(text: &str, value: Range<usize>) -> Option<Range<usize>> {
        let open = value.start + text[value.clone()].find('(')?;
        let close = value.start + text[value].rfind(')')?;
        (close > open).then_some(open + 1..close)
    }

    /// The value of the field `name` directly within a struct's body.
    pub(super) fn field(text: &str, body: Range<usize>, name: &str) -> Option<Range<usize>> {
        let bytes = text.as_bytes();
        let mut at = body.start;
        loop {
            at = skip_blank(text, at);
            if at >= body.end {
                return None;
            }
            let key_start = at;
            while at < body.end && (bytes[at].is_ascii_alphanumeric() || bytes[at] == b'_') {
                at += 1;
            }
            let key = &text[key_start..at];
            at = skip_blank(text, at);
            if key.is_empty() || bytes.get(at) != Some(&b':') {
                return None;
            }
            let value_start = skip_blank(text, at + 1);
            let value_stop = value_end(text, value_start)?;
            if key == name {
                return Some(value_start..value_stop);
            }
            at = skip_blank(text, value_stop);
            if bytes.get(at) == Some(&b',') {
                at += 1;
            }
        }
    }

    /// Where the value starting at `start` ends: at a comma or closing bracket
    /// at its own depth, or the end of the text.
    fn value_end(text: &str, start: usize) -> Option<usize> {
        let bytes = text.as_bytes();
        let mut depth = 0usize;
        let mut at = start;
        let mut last = start;
        while at < bytes.len() {
            match bytes[at] {
                b'"' => at = skip_string(text, at)?,
                b'\'' => at = skip_char(text, at)?,
                b'/' if bytes.get(at + 1) == Some(&b'/') || bytes.get(at + 1) == Some(&b'*') => {
                    at = skip_blank(text, at);
                    continue;
                }
                b'(' | b'[' | b'{' => {
                    depth += 1;
                    at += 1;
                }
                b')' | b']' | b'}' => {
                    if depth == 0 {
                        return Some(last);
                    }
                    depth -= 1;
                    at += 1;
                    if depth == 0 {
                        return Some(at);
                    }
                }
                b',' if depth == 0 => return Some(last),
                byte if byte.is_ascii_whitespace() => at += 1,
                _ => at += 1,
            }
            if !bytes[at - 1].is_ascii_whitespace() {
                last = at;
            }
        }
        (depth == 0).then_some(last)
    }

    fn skip_string(text: &str, start: usize) -> Option<usize> {
        let bytes = text.as_bytes();
        let mut at = start + 1;
        while at < bytes.len() {
            match bytes[at] {
                b'\\' => at += 2,
                b'"' => return Some(at + 1),
                _ => at += 1,
            }
        }
        None
    }

    fn skip_char(text: &str, start: usize) -> Option<usize> {
        let bytes = text.as_bytes();
        let mut at = start + 1;
        while at < bytes.len() {
            match bytes[at] {
                b'\\' => at += 2,
                b'\'' => return Some(at + 1),
                _ => at += 1,
            }
        }
        None
    }

    /// Past whitespace and comments.
    fn skip_blank(text: &str, start: usize) -> usize {
        let bytes = text.as_bytes();
        let mut at = start;
        loop {
            while at < bytes.len() && bytes[at].is_ascii_whitespace() {
                at += 1;
            }
            if text[at..].starts_with("//") {
                at = text[at..].find('\n').map_or(bytes.len(), |line| at + line);
            } else if text[at..].starts_with("/*") {
                at = text[at + 2..]
                    .find("*/")
                    .map_or(bytes.len(), |close| at + 2 + close + 2);
            } else {
                return at;
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn fields_are_found_by_name_past_strings_and_comments() {
            let text = r#"( a: "x), b: (", // b: fake
                b: Some((c: [1, (2)], d: 'x')), /* d: 9 */ d: Done(why: "ok"), )"#;
            let body = inside(text, root(text).expect("a root")).expect("a body");
            assert_eq!(&text[field(text, body.clone(), "a").expect("a")], "\"x), b: (\"");
            let b = field(text, body.clone(), "b").expect("b");
            assert_eq!(&text[b.clone()], "Some((c: [1, (2)], d: 'x'))");
            assert_eq!(
                &text[field(text, body.clone(), "d").expect("d")],
                "Done(why: \"ok\")"
            );
            assert_eq!(field(text, body, "e"), None);
        }
    }
}

/// Reads and parses a save file.
pub fn load(path: &Path) -> Result<SaveEnvelope, SaveError> {
    let text = fs::read_to_string(path).map_err(|source| SaveError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    from_ron(&text).map_err(|error| match error {
        // Re-attach the real path to errors raised against in-memory text.
        SaveError::Parse { message, .. } => SaveError::Parse {
            path: path.to_path_buf(),
            message,
        },
        other => other,
    })
}

/// Writes a save atomically.
///
/// Serializes to a temporary file beside the target, flushes and syncs it, then
/// renames over the destination. A crash or power loss mid-write therefore
/// leaves the previous save intact rather than a half-written file: `rename` is
/// atomic on both Unix and Windows.
pub fn save(path: &Path, envelope: &SaveEnvelope) -> Result<(), SaveError> {
    use std::io::Write as _;

    let text = envelope.to_ron()?;
    let temporary = path.with_extension("ron.tmp");

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|source| SaveError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }

    let mut file = fs::File::create(&temporary).map_err(|source| SaveError::Io {
        path: temporary.clone(),
        source,
    })?;
    file.write_all(text.as_bytes())
        .and_then(|()| file.sync_all())
        .map_err(|source| SaveError::Io {
            path: temporary.clone(),
            source,
        })?;
    drop(file);

    fs::rename(&temporary, path).map_err(|source| SaveError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{DomainEvent, apply};
    use crate::ids::{InstanceId, ItemId, NpcId, QuestId, WorldFlag};

    fn hash() -> ContentHash {
        ContentHash::from_bytes([7; 32])
    }

    fn revision() -> ContentRevisionId {
        ContentRevisionId::new("rev-0001").expect("valid id")
    }

    fn populated() -> WorldState {
        let mut world = WorldState::new(0xC1DE_2026, revision());
        apply(
            &mut world,
            &DomainEvent::QuestAccepted {
                quest: QuestId::new("still-waters").expect("valid id"),
            },
        )
        .expect("applies");
        apply(
            &mut world,
            &DomainEvent::ItemGranted {
                item: ItemId::new("ember-ore").expect("valid id"),
                quantity: 4,
            },
        )
        .expect("applies");
        apply(
            &mut world,
            &DomainEvent::WorldFlagSet {
                flag: WorldFlag::new("prologue-complete").expect("valid id"),
            },
        )
        .expect("applies");
        world
    }

    fn temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "cinderwake-save-{}-{}-{name}.ron",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_nanos())
                .unwrap_or_default()
        ));
        path
    }

    /// ADR-0013's rule for every migration: a checked-in save of the old
    /// shape, and a test that it arrives with its state intact (ADR-0018).
    #[test]
    fn a_version_1_save_migrates_with_its_state_intact() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/saves/v1-populated.ron");
        let text = fs::read_to_string(&path).expect("the v1 fixture");
        assert_eq!(peek_version(&text).expect("a version"), 1);

        let migrated = from_ron(&text).expect("a v1 save loads");
        assert_eq!(migrated.format_version, SAVE_FORMAT_VERSION);
        assert_eq!(migrated.world.schema_version, crate::world::WORLD_SCHEMA_VERSION);
        let progression = &migrated.world.player.progression;
        assert!(progression.talents.is_empty(), "nothing taken");
        assert_eq!(progression.loadout, None, "the class's own abilities, in order");

        // Everything else is exactly what the v1 file said.
        let expected = populated();
        assert_eq!(migrated.world.player.inventory, expected.player.inventory);
        assert_eq!(migrated.world.quests, expected.quests);
        assert_eq!(migrated.world.world_flags, expected.world_flags);
        assert_eq!(migrated.world.world_seed, expected.world_seed);
    }

    /// The v1→v2 edit on compact text: a comma added where the last field
    /// lacked one, versions bumped, and nothing else touched — and a file that
    /// already has the new fields is refused rather than given them twice.
    #[test]
    fn v1_to_v2_edits_only_what_it_means_to() {
        let v1 = "(format_version:1,note:\"progression: (\",world:(schema_version:1,player:(progression:(class:None,level:0,experience:0))))";
        let v2 = v1_to_v2(v1).expect("migrates");
        assert_eq!(
            v2,
            "(format_version:2,note:\"progression: (\",world:(schema_version:2,player:(progression:(class:None,level:0,experience:0, talents: [], loadout: None,))))"
        );
        assert!(v1_to_v2(&v2).is_err(), "already has talents");
    }

    #[test]
    fn round_trips_through_ron() {
        let envelope = SaveEnvelope::new(populated(), "0.1.0", hash());
        let text = envelope.to_ron().expect("serializes");
        let parsed = from_ron(&text).expect("parses");
        assert_eq!(parsed, envelope);
    }

    #[test]
    fn round_trips_through_a_file_atomically() {
        let path = temp_path("round-trip");
        let envelope = SaveEnvelope::new(populated(), "0.1.0", hash());
        save(&path, &envelope).expect("writes");
        assert!(
            !path.with_extension("ron.tmp").exists(),
            "the temporary file is renamed away, not left behind"
        );

        let loaded = load(&path).expect("loads");
        assert_eq!(loaded, envelope);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn overwriting_preserves_the_previous_save_until_the_rename() {
        let path = temp_path("overwrite");
        let first = SaveEnvelope::new(populated(), "0.1.0", hash());
        save(&path, &first).expect("writes");

        let mut changed = populated();
        apply(&mut changed, &DomainEvent::ClockAdvanced { ticks: 500 }).expect("applies");
        let second = SaveEnvelope::new(changed, "0.1.0", hash());
        save(&path, &second).expect("overwrites");

        let loaded = load(&path).expect("loads");
        assert_eq!(loaded.world.clock.tick, 500);
        assert_ne!(loaded, first);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_newer_save_is_refused_without_being_touched() {
        let mut envelope = SaveEnvelope::new(populated(), "0.1.0", hash());
        envelope.format_version = SAVE_FORMAT_VERSION + 5;
        let text = envelope.to_ron().expect("serializes");

        match from_ron(&text) {
            Err(SaveError::FutureVersion { found, supported }) => {
                assert_eq!(found, SAVE_FORMAT_VERSION + 5);
                assert_eq!(supported, SAVE_FORMAT_VERSION);
            }
            other => panic!("expected a FutureVersion refusal, got {other:?}"),
        }
    }

    #[test]
    fn peek_version_reads_a_shape_it_cannot_otherwise_understand() {
        let text = r#"(format_version: 99, something_from_the_future: "unknown")"#;
        assert_eq!(peek_version(text).expect("reads the version"), 99);
        assert!(
            matches!(from_ron(text), Err(SaveError::FutureVersion { .. })),
            "identifying a future save must not mean loading it"
        );
    }

    #[test]
    fn unknown_fields_are_rejected_rather_than_dropped() {
        let envelope = SaveEnvelope::new(populated(), "0.1.0", hash());
        let text = envelope.to_ron().expect("serializes");
        let tampered = text.replacen("format_version:", "unexpected_field: 1,\n    format_version:", 1);
        assert!(
            matches!(from_ron(&tampered), Err(SaveError::Parse { .. })),
            "an unrecognised field must fail the load, not vanish silently"
        );
    }

    #[test]
    fn a_save_holding_a_scoped_id_is_refused() {
        let mut world = populated();
        let instance = InstanceId::new("crypt-of-ash").expect("valid id");
        // Bypass the reducers deliberately: this is the corruption the check exists for.
        world
            .world_flags
            .insert(WorldFlag::scoped(&instance, "leaked").expect("valid scoped id"));
        let envelope = SaveEnvelope::new(world, "0.1.0", hash());
        let text = envelope.to_ron().expect("serializes");
        assert!(matches!(from_ron(&text), Err(SaveError::NotCanonical(_))));
    }

    #[test]
    fn missing_files_report_the_path() {
        let path = temp_path("absent");
        match load(&path) {
            Err(SaveError::Io { path: reported, .. }) => assert_eq!(reported, path),
            other => panic!("expected an Io error naming the path, got {other:?}"),
        }
    }

    /// The exit criterion for M4: a world that has been saved, loaded, and then
    /// rebuilt by replaying its events from the seed must be the same world.
    #[test]
    fn save_load_and_replay_agree() {
        let log = vec![
            DomainEvent::ClockAdvanced { ticks: 1_200 },
            DomainEvent::QuestAccepted {
                quest: QuestId::new("still-waters").expect("valid id"),
            },
            DomainEvent::ItemGranted {
                item: ItemId::new("ember-ore").expect("valid id"),
                quantity: 4,
            },
            DomainEvent::NpcLearnedClue {
                npc: NpcId::new("warden-sela").expect("valid id"),
                clue: crate::ids::ClueId::new("clue-fouled-intake").expect("valid id"),
            },
        ];

        let mut live = WorldState::new(0xC1DE_2026, revision());
        live.npcs.insert(
            NpcId::new("warden-sela").expect("valid id"),
            crate::world::NpcState::default(),
        );
        for event in &log {
            apply(&mut live, event).expect("applies");
        }

        let envelope = SaveEnvelope::new(live.clone(), "0.1.0", hash());
        let loaded = from_ron(&envelope.to_ron().expect("serializes")).expect("parses");
        assert_eq!(loaded.world, live, "save/load preserves state exactly");

        let mut replayed = WorldState::new(0xC1DE_2026, revision());
        replayed.npcs.insert(
            NpcId::new("warden-sela").expect("valid id"),
            crate::world::NpcState::default(),
        );
        for event in &log {
            apply(&mut replayed, event).expect("applies");
        }
        assert_eq!(replayed, loaded.world, "replaying the log reproduces the save");
    }
}
