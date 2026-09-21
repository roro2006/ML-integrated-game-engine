//! Domain event log and replay.
//!
//! An append-only, line-delimited record of accepted events. Replaying it from
//! the world seed must reproduce saved state exactly; that equivalence is the
//! test that keeps the reducers honest (ADR-0013).
//!
//! # Why line-delimited RON
//!
//! One event per line, rather than one RON document containing a list. Appends
//! are a single write with no rewriting of what came before, the file can be
//! tailed while the game runs, and — most importantly — a process killed
//! mid-append leaves a torn *final* line rather than a corrupt document. Every
//! event before it is still readable history, so [`read_events`] reports the
//! torn tail instead of discarding the session.
//!
//! PROJECT_CONTEXT §8 names NDJSON for this; RON is used because `serde_json`
//! is not an approved dependency and nothing here needs JSON tooling.

use std::fmt;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use crate::events::{DomainError, DomainEvent, apply};
use crate::world::WorldState;

#[derive(Debug)]
pub enum LogError {
    Io {
        path: PathBuf,
        source: io::Error,
    },
    /// A line could not be parsed. Carries the line number, because "the log is
    /// corrupt" is useless and "line 4,210 is corrupt" is actionable.
    Parse {
        line: usize,
        message: String,
    },
    Serialize(String),
    /// An event did not apply during replay, meaning the log and the reducers
    /// disagree — a bug, not a user error.
    Replay {
        index: usize,
        source: DomainError,
    },
}

impl fmt::Display for LogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "cannot access {}: {source}", path.display()),
            Self::Parse { line, message } => write!(f, "event log line {line} is invalid: {message}"),
            Self::Serialize(message) => write!(f, "cannot serialize event: {message}"),
            Self::Replay { index, source } => {
                write!(f, "event {index} could not be replayed: {source}")
            }
        }
    }
}

impl std::error::Error for LogError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Replay { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// What reading a log produced.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct LogContents {
    /// Every complete event, in order.
    pub events: Vec<DomainEvent>,
    /// Set when the final line was incomplete — a write interrupted by a crash.
    /// The events before it are still valid; the caller decides whether to warn.
    pub torn_tail: bool,
}

/// Serializes one event to a single line, with no interior newlines.
///
/// Compact rather than pretty on purpose: the line *is* the record separator,
/// so a pretty-printed event would break the format.
pub fn encode_event(event: &DomainEvent) -> Result<String, LogError> {
    let line = ron::to_string(event).map_err(|error| LogError::Serialize(error.to_string()))?;
    debug_assert!(!line.contains('\n'), "compact RON must not contain newlines");
    Ok(line)
}

/// Appends events to the log, creating it if absent.
///
/// Each event is written as one line terminated by `\n`, then the file is
/// synced, so an accepted event that the player saw is on disk before the game
/// proceeds.
pub fn append_events(path: &Path, events: &[DomainEvent]) -> Result<(), LogError> {
    if events.is_empty() {
        return Ok(());
    }
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|source| LogError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }

    let mut buffer = String::new();
    for event in events {
        buffer.push_str(&encode_event(event)?);
        buffer.push('\n');
    }

    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|source| LogError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    file.write_all(buffer.as_bytes())
        .and_then(|()| file.sync_all())
        .map_err(|source| LogError::Io {
            path: path.to_path_buf(),
            source,
        })
}

/// Reads every event from a log file.
///
/// A missing file is an empty log, not an error: a world that has done nothing
/// yet has nothing to record.
pub fn read_events(path: &Path) -> Result<LogContents, LogError> {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(LogContents::default()),
        Err(source) => {
            return Err(LogError::Io {
                path: path.to_path_buf(),
                source,
            });
        }
    };

    let mut contents = LogContents::default();
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut number = 0;

    loop {
        line.clear();
        let read = reader.read_line(&mut line).map_err(|source| LogError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if read == 0 {
            break;
        }
        number += 1;

        // A line without its terminator is the tail of an interrupted write.
        // Everything before it remains valid history.
        if !line.ends_with('\n') {
            contents.torn_tail = true;
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let event = ron::from_str::<DomainEvent>(trimmed).map_err(|error| LogError::Parse {
            line: number,
            message: error.to_string(),
        })?;
        contents.events.push(event);
    }

    Ok(contents)
}

/// Folds a log over a state, reporting which event failed if one does.
pub fn replay(state: &mut WorldState, events: &[DomainEvent]) -> Result<(), LogError> {
    for (index, event) in events.iter().enumerate() {
        apply(state, event).map_err(|source| LogError::Replay { index, source })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::{ContentRevisionId, ItemId, QuestId, WorldFlag};

    fn revision() -> ContentRevisionId {
        ContentRevisionId::new("rev-0001").expect("valid id")
    }

    fn sample_log() -> Vec<DomainEvent> {
        vec![
            DomainEvent::ClockAdvanced { ticks: 60 },
            DomainEvent::QuestAccepted {
                quest: QuestId::new("still-waters").expect("valid id"),
            },
            DomainEvent::ItemGranted {
                item: ItemId::new("ember-ore").expect("valid id"),
                quantity: 2,
            },
            DomainEvent::WorldFlagSet {
                flag: WorldFlag::new("prologue-complete").expect("valid id"),
            },
        ]
    }

    fn temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "cinderwake-log-{}-{}-{name}.ron",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_nanos())
                .unwrap_or_default()
        ));
        path
    }

    #[test]
    fn events_encode_to_exactly_one_line() {
        for event in sample_log() {
            let line = encode_event(&event).expect("encodes");
            assert!(!line.contains('\n'), "an event must not span lines: {line}");
        }
    }

    #[test]
    fn append_then_read_round_trips() {
        let path = temp_path("round-trip");
        let events = sample_log();
        append_events(&path, &events).expect("appends");

        let contents = read_events(&path).expect("reads");
        assert_eq!(contents.events, events);
        assert!(!contents.torn_tail);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn appending_twice_preserves_order() {
        let path = temp_path("append-twice");
        let events = sample_log();
        append_events(&path, &events[..2]).expect("appends");
        append_events(&path, &events[2..]).expect("appends again");

        assert_eq!(read_events(&path).expect("reads").events, events);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_missing_log_is_empty_rather_than_an_error() {
        let contents = read_events(&temp_path("absent")).expect("reads");
        assert!(contents.events.is_empty());
        assert!(!contents.torn_tail);
    }

    #[test]
    fn a_torn_final_line_keeps_the_events_before_it() {
        let path = temp_path("torn");
        append_events(&path, &sample_log()).expect("appends");

        // Simulate a process killed mid-append.
        let mut text = fs::read_to_string(&path).expect("reads");
        text.push_str("ClockAdvanced(ticks: 4");
        fs::write(&path, text).expect("writes");

        let contents = read_events(&path).expect("reads");
        assert_eq!(contents.events, sample_log(), "complete events survive");
        assert!(contents.torn_tail, "and the torn tail is reported, not hidden");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_corrupt_complete_line_names_its_line_number() {
        let path = temp_path("corrupt");
        append_events(&path, &sample_log()).expect("appends");
        let mut text = fs::read_to_string(&path).expect("reads");
        text.push_str("this is not an event\n");
        fs::write(&path, text).expect("writes");

        match read_events(&path) {
            Err(LogError::Parse { line, .. }) => assert_eq!(line, 5),
            other => panic!("expected a Parse error naming line 5, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    /// M4's replay exit criterion, at the log level: the state rebuilt from a
    /// log on disk equals the state built by applying the same events live.
    #[test]
    fn replaying_a_log_from_disk_reproduces_live_state() {
        let path = temp_path("replay");
        let events = sample_log();
        append_events(&path, &events).expect("appends");

        let mut live = WorldState::new(7, revision());
        replay(&mut live, &events).expect("replays");

        let from_disk = read_events(&path).expect("reads");
        let mut rebuilt = WorldState::new(7, revision());
        replay(&mut rebuilt, &from_disk.events).expect("replays");

        assert_eq!(rebuilt, live);
        assert_eq!(
            ron::to_string(&rebuilt).expect("serializes"),
            ron::to_string(&live).expect("serializes"),
            "and serializes identically, which is what content hashing relies on"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_failing_event_names_its_index() {
        let mut state = WorldState::new(1, revision());
        let events = vec![
            DomainEvent::ClockAdvanced { ticks: 10 },
            // Not accepted, so completing its objective must fail.
            DomainEvent::QuestObjectiveCompleted {
                quest: QuestId::new("never-started").expect("valid id"),
                objective: crate::ids::ObjectiveId::new("meet-sela").expect("valid id"),
            },
        ];
        match replay(&mut state, &events) {
            Err(LogError::Replay { index, .. }) => assert_eq!(index, 1),
            other => panic!("expected a Replay error at index 1, got {other:?}"),
        }
    }
}
