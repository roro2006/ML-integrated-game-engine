//! Stable typed identifiers.
//!
//! IDs are the vocabulary every other layer shares: content files, saves,
//! domain events, AI proposals, and later multiplayer replication all address
//! things by ID rather than by position or runtime handle. Because an ID
//! written into a save today must still resolve years later, the syntax is
//! narrow and validated at construction, and every kind of thing gets its own
//! type so the compiler catches a [`QuestId`] passed where an [`NpcId`] belongs.
//!
//! # Syntax
//!
//! A canonical id matches `[a-z0-9]([a-z0-9-]*[a-z0-9])?` and is at most
//! [`MAX_ID_LEN`] bytes. Lowercase-only keeps ids stable across case-sensitive
//! and case-insensitive filesystems, since ids name content files; the absence
//! of `.`, `/`, `\`, and `:` means a hostile or hallucinated id can never be
//! path-traversal shaped once it has passed the type boundary.
//!
//! This is deliberately *looser* than [`platform::paths::validate_app_id`],
//! which additionally forbids `--`. That rule exists because the app id names a
//! single directory whose readability matters; content ids such as
//! `ember-warden--elite` are merely ugly, not dangerous, and rejecting them
//! would make an authoring mistake fail late and mysteriously. The two
//! validators are intentionally separate: do not "unify" them, and note that
//! `game_core` may not depend on `platform` in any case (ADR-0002).
//!
//! [`platform::paths::validate_app_id`]: https://docs.rs/platform
//!
//! # Instance scoping
//!
//! ADR-0010 lets AI mint new names only inside a generated instance, never in
//! the persistent world. That distinction lives here, in the ID model, rather
//! than in the AI pipeline, because an ID namespace retrofitted after saves
//! exist is a migration worth avoiding — hence this exists in M4 though AI
//! arrives in M13.
//!
//! An id is therefore either **canonical** (authored content, permanent world
//! facts) or **instance-scoped**, written `inst:<instance-id>/<name>` with both
//! parts individually valid. Canonical registries reject scoped ids via
//! [`Id::require_canonical`]; instance content may reference canonical ids, but
//! nothing canonical may hold a scoped one. Promotion to canonical status is a
//! human authoring step with no automatic path.
//!
//! # Serde
//!
//! Every id serializes as the plain string in both forms, so content files and
//! saves stay readable and diffable. Deserialization validates: an invalid id
//! in a save or content file is a parse error, never a silently accepted value.

use std::fmt;
use std::str::FromStr;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Maximum length, in bytes, of one id segment.
///
/// Applies to each part of a scoped id separately, so the serialized form of a
/// scoped id can reach `5 + 64 + 1 + 64` bytes.
pub const MAX_ID_LEN: usize = 64;

/// Prefix marking an instance-scoped id. `inst` is not a reserved canonical id:
/// only `inst:` (with the colon, which no canonical id may contain) is special.
const SCOPE_PREFIX: &str = "inst:";

/// Separates the instance from the name inside a scoped id.
const SCOPE_SEPARATOR: char = '/';

/// Why a string is not a valid id.
///
/// Variants carry enough detail to point an author at the offending character,
/// because these errors surface in content validation and save loading where
/// the human fixing them is not the one who wrote the parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdError {
    /// The id, or one segment of a scoped id, was empty.
    Empty,
    /// The segment exceeded [`MAX_ID_LEN`] bytes.
    TooLong {
        /// Length of the offending segment, in bytes.
        len: usize,
    },
    /// A character outside `[a-z0-9-]` appeared.
    InvalidCharacter {
        /// Byte offset into the text passed to the constructor.
        position: usize,
        /// The offending character.
        character: char,
    },
    /// The segment started with a character that may not begin an id.
    LeadingCharacter {
        /// The offending character.
        character: char,
    },
    /// The segment ended with a character that may not end an id.
    TrailingCharacter {
        /// The offending character.
        character: char,
    },
    /// An `inst:` prefix was present but the `inst:<instance>/<name>` shape was not.
    MalformedScope,
    /// The instance part of a scoped id is not itself a valid canonical id.
    InvalidScope(Box<IdError>),
    /// A canonical id was required — typically by a canonical registry — but an
    /// instance-scoped id was supplied. See ADR-0010.
    UnexpectedScope,
    /// An instance-scoped id was required but a canonical id was supplied.
    MissingScope,
}

impl fmt::Display for IdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("id is empty"),
            Self::TooLong { len } => {
                write!(f, "id is {len} bytes, the limit is {MAX_ID_LEN}")
            }
            Self::InvalidCharacter { position, character } => write!(
                f,
                "invalid character {character:?} at byte {position}: ids allow only [a-z0-9-]"
            ),
            Self::LeadingCharacter { character } => {
                write!(f, "id must not start with {character:?}")
            }
            Self::TrailingCharacter { character } => {
                write!(f, "id must not end with {character:?}")
            }
            Self::MalformedScope => {
                write!(
                    f,
                    "instance-scoped ids have the form `{SCOPE_PREFIX}<instance>/<name>`"
                )
            }
            Self::InvalidScope(source) => write!(f, "invalid instance scope: {source}"),
            Self::UnexpectedScope => f.write_str("expected a canonical id, found an instance-scoped one"),
            Self::MissingScope => f.write_str("expected an instance-scoped id, found a canonical one"),
        }
    }
}

impl std::error::Error for IdError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidScope(source) => Some(source),
            _ => None,
        }
    }
}

/// Validates one segment, reporting character positions relative to the start
/// of the whole text so messages point into what the caller actually passed.
fn validate_segment(segment: &str, offset: usize) -> Result<(), IdError> {
    if segment.is_empty() {
        return Err(IdError::Empty);
    }
    if segment.len() > MAX_ID_LEN {
        return Err(IdError::TooLong { len: segment.len() });
    }
    for (index, character) in segment.char_indices() {
        if !(character.is_ascii_lowercase() || character.is_ascii_digit() || character == '-') {
            return Err(IdError::InvalidCharacter {
                position: offset + index,
                character,
            });
        }
    }
    // Only `-` can reach here, but the rule is expressed as "boundaries are
    // alphanumeric" so it still holds if the character set ever widens.
    let mut characters = segment.chars();
    let first = characters.next().unwrap_or_default();
    let last = characters.next_back().unwrap_or(first);
    if !first.is_ascii_alphanumeric() {
        return Err(IdError::LeadingCharacter { character: first });
    }
    if !last.is_ascii_alphanumeric() {
        return Err(IdError::TrailingCharacter { character: last });
    }
    Ok(())
}

/// A validated identifier, canonical or instance-scoped.
///
/// This is the shared core; game code uses the typed newtypes below rather than
/// `Id` directly, so that a quest id cannot stand in for an npc id.
///
/// # Ordering
///
/// `Ord` is derived over `(scope, name)`, so canonical ids sort before every
/// scoped id and scoped ids group by instance. The order is total and depends
/// only on the id's contents — canonical state keys `BTreeMap`s by these, and
/// replay determinism requires that iteration order never depend on insertion
/// order, hashing, or memory addresses.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Id {
    /// `None` for canonical ids. Boxed so `Id` and `InstanceId` can refer to
    /// one another without an infinitely sized type; nesting cannot occur
    /// because an `InstanceId` is itself always canonical.
    scope: Option<Box<InstanceId>>,
    name: String,
}

impl Id {
    /// Parses either form: a canonical id, or `inst:<instance>/<name>`.
    pub fn new(text: &str) -> Result<Self, IdError> {
        let Some(rest) = text.strip_prefix(SCOPE_PREFIX) else {
            validate_segment(text, 0)?;
            return Ok(Self {
                scope: None,
                name: text.to_owned(),
            });
        };
        let (instance, name) = rest.split_once(SCOPE_SEPARATOR).ok_or(IdError::MalformedScope)?;
        let instance_offset = SCOPE_PREFIX.len();
        validate_segment(instance, instance_offset)
            .map_err(|source| IdError::InvalidScope(Box::new(source)))?;
        validate_segment(name, instance_offset + instance.len() + 1)?;
        Ok(Self {
            scope: Some(Box::new(InstanceId(Self {
                scope: None,
                name: instance.to_owned(),
            }))),
            name: name.to_owned(),
        })
    }

    /// Builds an id owned by `instance`. `name` must be a bare canonical name;
    /// it is not re-parsed for a scope, so scopes cannot nest.
    pub fn scoped(instance: &InstanceId, name: &str) -> Result<Self, IdError> {
        validate_segment(name, 0)?;
        Ok(Self {
            scope: Some(Box::new(instance.clone())),
            name: name.to_owned(),
        })
    }

    /// The name without its scope, if any.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The owning instance, or `None` for a canonical id.
    pub fn scope(&self) -> Option<&InstanceId> {
        self.scope.as_deref()
    }

    /// Whether this id may be stored in the persistent world (ADR-0010).
    pub fn is_canonical(&self) -> bool {
        self.scope.is_none()
    }

    /// Guard for canonical registries: rejects instance-scoped ids so a
    /// generated name can never leak into the persistent world.
    pub fn require_canonical(&self) -> Result<&Self, IdError> {
        if self.is_canonical() {
            Ok(self)
        } else {
            Err(IdError::UnexpectedScope)
        }
    }

    /// Guard for instance content: rejects canonical ids and yields the owning
    /// instance, so a validator can check the reference stays inside it.
    pub fn require_scoped(&self) -> Result<&InstanceId, IdError> {
        self.scope().ok_or(IdError::MissingScope)
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.scope {
            Some(instance) => {
                write!(
                    f,
                    "{SCOPE_PREFIX}{}{SCOPE_SEPARATOR}{}",
                    instance.name(),
                    self.name
                )
            }
            None => f.write_str(&self.name),
        }
    }
}

impl FromStr for Id {
    type Err = IdError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::new(text)
    }
}

/// Deserializes a string through a fallible parser, so an invalid id in a save
/// or content file fails the parse instead of becoming a value.
struct StrVisitor<T, E> {
    expecting: &'static str,
    parse: fn(&str) -> Result<T, E>,
}

impl<'de, T, E: fmt::Display> Visitor<'de> for StrVisitor<T, E> {
    type Value = T;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.expecting)
    }

    fn visit_str<D: de::Error>(self, value: &str) -> Result<Self::Value, D> {
        (self.parse)(value).map_err(de::Error::custom)
    }
}

impl Serialize for Id {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Id {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_str(StrVisitor {
            expecting: "a canonical id or `inst:<instance>/<name>`",
            parse: Id::new,
        })
    }
}

/// Shared string behaviour for every id newtype: `Display`, `FromStr`, serde as
/// a bare string, and conversion down to the untyped [`Id`].
macro_rules! id_string_repr {
    ($name:ident, $expecting:expr) => {
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, f)
            }
        }

        impl FromStr for $name {
            type Err = IdError;

            fn from_str(text: &str) -> Result<Self, Self::Err> {
                Self::new(text)
            }
        }

        impl From<$name> for Id {
            fn from(id: $name) -> Self {
                id.0
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.collect_str(self)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                deserializer.deserialize_str(StrVisitor {
                    expecting: $expecting,
                    parse: Self::new,
                })
            }
        }
    };
}

/// Declares id newtypes that accept both canonical and instance-scoped forms.
///
/// Each is a distinct type wrapping [`Id`], so the compiler rejects a
/// [`QuestId`] used where an [`NpcId`] belongs, and all derive the total,
/// content-only ordering that `BTreeMap`-keyed canonical state depends on.
macro_rules! scopable_ids {
    ($( $(#[$attr:meta])* $name:ident ),+ $(,)?) => {$(
        $(#[$attr])*
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(Id);

        impl $name {
            /// Parses a canonical id or `inst:<instance>/<name>`.
            pub fn new(text: &str) -> Result<Self, IdError> {
                Id::new(text).map(Self)
            }

            /// Builds an id owned by `instance` (ADR-0010).
            pub fn scoped(instance: &InstanceId, name: &str) -> Result<Self, IdError> {
                Id::scoped(instance, name).map(Self)
            }

            /// The underlying untyped id.
            pub fn as_id(&self) -> &Id {
                &self.0
            }

            /// The name without its scope, if any.
            pub fn name(&self) -> &str {
                self.0.name()
            }

            /// The owning instance, or `None` for a canonical id.
            pub fn scope(&self) -> Option<&InstanceId> {
                self.0.scope()
            }

            /// Whether this id may be stored in the persistent world.
            pub fn is_canonical(&self) -> bool {
                self.0.is_canonical()
            }

            /// Guard for canonical registries; see [`Id::require_canonical`].
            pub fn require_canonical(&self) -> Result<&Self, IdError> {
                self.0.require_canonical()?;
                Ok(self)
            }

            /// Guard for instance content; see [`Id::require_scoped`].
            pub fn require_scoped(&self) -> Result<&InstanceId, IdError> {
                self.0.require_scoped()
            }
        }

        id_string_repr!($name, "a canonical id or `inst:<instance>/<name>`");
    )+};
}

scopable_ids! {
    /// Names a quest definition and its state.
    QuestId,
    /// Names an NPC. Canonical NPCs are authored; AI may name one only inside
    /// a generated instance (ADR-0010).
    NpcId,
    /// Names a faction. Always canonical in practice: factions are permanent
    /// world structure that instance content may reference but never create.
    FactionId,
    /// Names a region of the world map.
    RegionId,
    /// Names a location within a region.
    LocationId,
    /// Names an encounter definition or a generated encounter instance.
    EncounterId,
    /// Names an item definition. Instance rewards must resolve to canonical
    /// items, so item grants stay inside the authored item set.
    ItemId,
    /// Names an ability definition.
    AbilityId,
    /// Names a spawnable prefab composed from approved archetypes.
    PrefabId,
    /// Names one objective within a quest.
    ObjectiveId,
    /// Names one dialogue or quest choice.
    ChoiceId,
    /// Names a piece of discoverable knowledge.
    ClueId,
    /// Names one accepted revision of compiled content; saves record the
    /// revision they were written against alongside its [`ContentHash`].
    ContentRevisionId,
    /// Names a character class.
    ///
    /// ADR-0007 chose distinct classes over one flexible archetype, and classes
    /// are data-driven from M9, so this is a content id rather than an enum:
    /// adding a class must be an authoring change, not a code change.
    ClassId,
    /// Names one node of a class's talent tree (ADR-0018).
    TalentId,
    /// Names a levelling curve: experience per level (ADR-0018).
    LevelCurveId,
    /// Names a body plan: the joints a model's parts hang from (ADR-0017).
    BodyPlanId,
    /// Names one joint within a body plan. Animation clips address slots by
    /// this name, so one clip serves every model on the same plan.
    SlotId,
    /// Names a model recipe.
    ModelId,
    /// Names a palette models are painted from.
    PaletteId,
    /// Names one colour within a palette.
    SwatchId,
    /// Names an animation clip.
    ClipId,
    /// Names a visual effect composed from the effect allowlist (ADR-0017 §6).
    ///
    /// "Vfx" rather than "effect" because [`crate::content::Effect`] is the
    /// gameplay allowlist: an ability's effects change the world, and its
    /// visual effects change only what a player sees.
    VfxId,
    /// Names one way a quest ended.
    ///
    /// A quest that can end several ways records *which* ending happened,
    /// because later content gates on the choice, not merely on completion.
    QuestResolutionId,
    /// Names one durable boolean world fact.
    ///
    /// A typed id rather than a free string so the set of flags is an authored,
    /// validatable vocabulary — a typo in a flag name is otherwise a silently
    /// never-satisfied quest condition — and so flags order deterministically
    /// inside a `BTreeSet`.
    WorldFlag,
    /// Names a placement point inside a location: where the player starts, where
    /// an NPC stands, where a gatherable sits.
    ///
    /// Authored rather than derived, so a quest can say "spawn at the waterworks
    /// gate" and keep meaning it after the level is re-laid-out.
    SpawnId,
    /// Names one conversation.
    DialogueId,
    /// Names one node within a conversation.
    DialogueNodeId,
    /// Names a weighted drop table.
    LootTableId,
    /// Names a settlement blueprint upgrade (ADR-0007).
    UpgradeId,
}

/// Names one generated instance and the ID namespace it owns.
///
/// Written by hand rather than generated because an instance id is the
/// namespace: it is always canonical, and permitting `inst:a/b` here would
/// allow scopes to nest, which nothing in ADR-0010 wants and which would make
/// "discarded with its instance" ambiguous.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InstanceId(Id);

impl InstanceId {
    /// Parses a canonical id; an instance-scoped string is rejected.
    pub fn new(text: &str) -> Result<Self, IdError> {
        let id = Id::new(text)?;
        id.require_canonical()?;
        Ok(Self(id))
    }

    /// The instance name.
    pub fn name(&self) -> &str {
        self.0.name()
    }

    /// The underlying untyped id.
    pub fn as_id(&self) -> &Id {
        &self.0
    }
}

id_string_repr!(InstanceId, "a canonical instance id");

/// Number of bytes in a [`ContentHash`].
pub const CONTENT_HASH_BYTES: usize = 32;

/// Digest identifying one content revision, recorded in save provenance so a
/// save can tell whether the content it was written against still matches.
///
/// `game_core` carries and validates the value but never computes it; hashing
/// belongs to the content pipeline. The canonical text form is lowercase hex,
/// and parsing rejects uppercase so the same bytes always produce the same
/// bytes on disk — provenance that diffs cleanly is the whole point.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContentHash([u8; CONTENT_HASH_BYTES]);

/// Why a string is not a valid [`ContentHash`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentHashError {
    /// The text was not exactly `2 * CONTENT_HASH_BYTES` characters.
    InvalidLength {
        /// Length of the supplied text, in bytes.
        len: usize,
    },
    /// A byte outside `[0-9a-f]` appeared.
    InvalidDigit {
        /// Byte offset into the supplied text.
        position: usize,
        /// The offending byte.
        byte: u8,
    },
}

impl fmt::Display for ContentHashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { len } => write!(
                f,
                "content hash is {len} characters, expected {}",
                CONTENT_HASH_BYTES * 2
            ),
            Self::InvalidDigit { position, byte } => write!(
                f,
                "invalid hex digit 0x{byte:02x} at position {position}: expected lowercase [0-9a-f]"
            ),
        }
    }
}

impl std::error::Error for ContentHashError {}

fn hex_digit(byte: u8, position: usize) -> Result<u8, ContentHashError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err(ContentHashError::InvalidDigit { position, byte }),
    }
}

impl ContentHash {
    /// Wraps digest bytes produced elsewhere.
    pub const fn from_bytes(bytes: [u8; CONTENT_HASH_BYTES]) -> Self {
        Self(bytes)
    }

    /// The digest bytes.
    pub const fn as_bytes(&self) -> &[u8; CONTENT_HASH_BYTES] {
        &self.0
    }

    /// Parses the lowercase-hex text form.
    pub fn parse_hex(text: &str) -> Result<Self, ContentHashError> {
        if text.len() != CONTENT_HASH_BYTES * 2 {
            return Err(ContentHashError::InvalidLength { len: text.len() });
        }
        let mut bytes = [0u8; CONTENT_HASH_BYTES];
        for (index, (byte, pair)) in bytes.iter_mut().zip(text.as_bytes().chunks_exact(2)).enumerate() {
            *byte = (hex_digit(pair[0], index * 2)? << 4) | hex_digit(pair[1], index * 2 + 1)?;
        }
        Ok(Self(bytes))
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContentHash({self})")
    }
}

impl FromStr for ContentHash {
    type Err = ContentHashError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::parse_hex(text)
    }
}

impl Serialize for ContentHash {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for ContentHash {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_str(StrVisitor {
            expecting: "a 64-character lowercase hex content hash",
            parse: ContentHash::parse_hex,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn instance() -> InstanceId {
        InstanceId::new("crypt-of-ash").unwrap()
    }

    #[test]
    fn accepts_ordinary_canonical_ids() {
        let max_length = "a".repeat(MAX_ID_LEN);
        for text in [
            "a",
            "0",
            "quest",
            "ember-warden",
            "npc-42",
            "inst",
            // `--` is legal here although `platform::paths::validate_app_id`
            // forbids it; see the module docs before "fixing" this.
            "ember-warden--elite",
            max_length.as_str(),
        ] {
            let id = Id::new(text).unwrap_or_else(|error| panic!("{text:?}: {error}"));
            assert!(id.is_canonical(), "{text:?}");
            assert_eq!(id.name(), text);
            assert_eq!(id.to_string(), text);
            assert_eq!(id.scope(), None);
        }
    }

    #[test]
    fn rejects_malformed_ids() {
        let too_long = "a".repeat(MAX_ID_LEN + 1);
        let long_scope = format!("inst:{}/name", "a".repeat(MAX_ID_LEN + 1));
        let cases: &[(&str, IdError)] = &[
            ("", IdError::Empty),
            (too_long.as_str(), IdError::TooLong { len: MAX_ID_LEN + 1 }),
            (
                "Quest",
                IdError::InvalidCharacter {
                    position: 0,
                    character: 'Q',
                },
            ),
            (
                "my quest",
                IdError::InvalidCharacter {
                    position: 2,
                    character: ' ',
                },
            ),
            (
                "my_quest",
                IdError::InvalidCharacter {
                    position: 2,
                    character: '_',
                },
            ),
            ("-quest", IdError::LeadingCharacter { character: '-' }),
            ("quest-", IdError::TrailingCharacter { character: '-' }),
            (
                "..",
                IdError::InvalidCharacter {
                    position: 0,
                    character: '.',
                },
            ),
            (
                "../../etc/passwd",
                IdError::InvalidCharacter {
                    position: 0,
                    character: '.',
                },
            ),
            (
                "a/b",
                IdError::InvalidCharacter {
                    position: 1,
                    character: '/',
                },
            ),
            (
                "a\\b",
                IdError::InvalidCharacter {
                    position: 1,
                    character: '\\',
                },
            ),
            (
                "quêst",
                IdError::InvalidCharacter {
                    position: 2,
                    character: 'ê',
                },
            ),
            (
                "quest:extra",
                IdError::InvalidCharacter {
                    position: 5,
                    character: ':',
                },
            ),
            (
                "notinst:a/b",
                IdError::InvalidCharacter {
                    position: 7,
                    character: ':',
                },
            ),
            ("inst:", IdError::MalformedScope),
            ("inst:crypt-of-ash", IdError::MalformedScope),
            ("inst:/name", IdError::InvalidScope(Box::new(IdError::Empty))),
            ("inst:crypt/", IdError::Empty),
            (
                "inst:crypt/name/extra",
                IdError::InvalidCharacter {
                    position: 15,
                    character: '/',
                },
            ),
            (
                "inst:Crypt/name",
                IdError::InvalidScope(Box::new(IdError::InvalidCharacter {
                    position: 5,
                    character: 'C',
                })),
            ),
            (
                "inst:inst:a/b",
                IdError::InvalidScope(Box::new(IdError::InvalidCharacter {
                    position: 9,
                    character: ':',
                })),
            ),
            (
                long_scope.as_str(),
                IdError::InvalidScope(Box::new(IdError::TooLong { len: MAX_ID_LEN + 1 })),
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(Id::new(text).unwrap_err(), *expected, "{text:?}");
            assert_eq!(text.parse::<QuestId>().unwrap_err(), *expected, "{text:?}");
        }
    }

    #[test]
    fn error_messages_name_the_problem() {
        let error = Id::new("../etc").unwrap_err();
        let message = error.to_string();
        assert!(message.contains("'.'"), "{message}");
        assert!(message.contains("byte 0"), "{message}");

        let scope_error = Id::new("inst:Crypt/name").unwrap_err();
        assert!(std::error::Error::source(&scope_error).is_some());
        assert!(scope_error.to_string().contains("instance scope"));
    }

    /// Every newtype must serialize as a bare quoted string, not a struct.
    macro_rules! assert_ron_round_trip {
        ($($ty:ty),+ $(,)?) => {$({
            let id = <$ty>::new("ember-warden").unwrap();
            let text = ron::to_string(&id).unwrap();
            assert_eq!(text, "\"ember-warden\"", "{}", stringify!($ty));
            assert_eq!(ron::from_str::<$ty>(&text).unwrap(), id, "{}", stringify!($ty));
        })+};
    }

    macro_rules! assert_scoped_ron_round_trip {
        ($($ty:ty),+ $(,)?) => {$({
            let id = <$ty>::scoped(&instance(), "ember-warden").unwrap();
            let text = ron::to_string(&id).unwrap();
            assert_eq!(text, "\"inst:crypt-of-ash/ember-warden\"", "{}", stringify!($ty));
            assert_eq!(ron::from_str::<$ty>(&text).unwrap(), id, "{}", stringify!($ty));
        })+};
    }

    #[test]
    fn every_newtype_round_trips_through_ron_as_a_string() {
        assert_ron_round_trip!(
            QuestId,
            NpcId,
            FactionId,
            RegionId,
            LocationId,
            EncounterId,
            ItemId,
            AbilityId,
            PrefabId,
            ObjectiveId,
            ChoiceId,
            ClueId,
            ContentRevisionId,
            InstanceId,
            Id,
        );
        assert_scoped_ron_round_trip!(
            QuestId,
            NpcId,
            FactionId,
            RegionId,
            LocationId,
            EncounterId,
            ItemId,
            AbilityId,
            PrefabId,
            ObjectiveId,
            ChoiceId,
            ClueId,
            ContentRevisionId,
        );
    }

    #[test]
    fn deserializing_an_invalid_id_fails() {
        for text in [
            "\"\"",
            "\"Ember Warden\"",
            "\"../../etc/passwd\"",
            "\"ember_warden\"",
            "\"-ember\"",
            "\"inst:\"",
            "\"inst:crypt-of-ash\"",
        ] {
            assert!(ron::from_str::<QuestId>(text).is_err(), "{text}");
        }
        // An instance id names the namespace, so it can never be scoped itself.
        assert!(ron::from_str::<InstanceId>("\"inst:crypt-of-ash/nested\"").is_err());
        // Nothing accepts a struct-shaped representation.
        assert!(ron::from_str::<QuestId>("QuestId(\"ember-warden\")").is_err());
    }

    #[test]
    fn scoped_ids_round_trip_through_text() {
        let text = "inst:crypt-of-ash/ember-warden";
        let id = QuestId::new(text).unwrap();
        assert_eq!(id.to_string(), text);
        assert_eq!(id.name(), "ember-warden");
        assert_eq!(id.scope(), Some(&instance()));
        assert!(!id.is_canonical());
        assert_eq!(QuestId::scoped(&instance(), "ember-warden").unwrap(), id);
        assert_eq!(text.parse::<QuestId>().unwrap(), id);
    }

    #[test]
    fn scoped_constructor_validates_the_name() {
        assert_eq!(
            QuestId::scoped(&instance(), "Ember Warden").unwrap_err(),
            IdError::InvalidCharacter {
                position: 0,
                character: 'E'
            }
        );
        // Scopes cannot nest: the name is a bare name, never re-parsed.
        assert_eq!(
            QuestId::scoped(&instance(), "inst:other/name").unwrap_err(),
            IdError::InvalidCharacter {
                position: 4,
                character: ':'
            }
        );
    }

    #[test]
    fn canonical_and_scoped_ids_are_rejected_where_the_other_is_required() {
        let canonical = NpcId::new("ember-warden").unwrap();
        let scoped = NpcId::scoped(&instance(), "ember-warden").unwrap();

        // A canonical registry refuses a generated name (ADR-0010).
        assert_eq!(scoped.require_canonical().unwrap_err(), IdError::UnexpectedScope);
        assert_eq!(canonical.require_canonical().unwrap(), &canonical);

        // Instance content must not claim a canonical name as its own.
        assert_eq!(canonical.require_scoped().unwrap_err(), IdError::MissingScope);
        assert_eq!(scoped.require_scoped().unwrap(), &instance());

        // Same name, different namespace: never equal, so no accidental aliasing.
        assert_ne!(canonical, scoped);
        assert_eq!(canonical.name(), scoped.name());
    }

    #[test]
    fn instance_ids_are_always_canonical() {
        assert_eq!(
            InstanceId::new("inst:crypt-of-ash/nested").unwrap_err(),
            IdError::UnexpectedScope
        );
        assert!(instance().as_id().is_canonical());
        assert_eq!(instance().name(), "crypt-of-ash");
    }

    fn ordering_sample() -> Vec<QuestId> {
        [
            "zephyr-hunt",
            "inst:crypt-of-ash/warden",
            "amber-road",
            "inst:ashfall/warden",
            "ember-oath",
            "inst:ashfall/acolyte",
        ]
        .into_iter()
        .map(|text| QuestId::new(text).unwrap())
        .collect()
    }

    #[test]
    fn ordering_is_total_and_deterministic() {
        let mut forward = ordering_sample();
        let mut reversed: Vec<QuestId> = ordering_sample().into_iter().rev().collect();
        // A different starting permutation, to catch an unstable comparison.
        let mut rotated = ordering_sample();
        rotated.rotate_left(3);

        forward.sort();
        reversed.sort();
        rotated.sort();
        assert_eq!(forward, reversed);
        assert_eq!(forward, rotated);

        let order: Vec<String> = forward.iter().map(QuestId::to_string).collect();
        assert_eq!(
            order,
            [
                // Canonical ids sort before every scoped id, then scoped ids
                // group by instance.
                "amber-road",
                "ember-oath",
                "zephyr-hunt",
                "inst:ashfall/acolyte",
                "inst:ashfall/warden",
                "inst:crypt-of-ash/warden",
            ]
        );

        // Sorting an already sorted list changes nothing.
        let stable = forward.clone();
        forward.sort();
        assert_eq!(forward, stable);
    }

    #[test]
    fn btree_map_iteration_order_is_independent_of_insertion_order() {
        let sample = ordering_sample();
        let forward: BTreeMap<QuestId, usize> = sample.iter().cloned().zip(0..).collect();
        let backward: BTreeMap<QuestId, usize> = sample.iter().rev().cloned().zip(0..).collect();
        let forward_keys: Vec<&QuestId> = forward.keys().collect();
        let backward_keys: Vec<&QuestId> = backward.keys().collect();
        assert_eq!(forward_keys, backward_keys);

        // And it survives a serialization round trip, which is what replay needs.
        let text = ron::to_string(&forward_keys).unwrap();
        assert_eq!(ron::from_str::<Vec<QuestId>>(&text).unwrap().len(), sample.len());
    }

    #[test]
    fn content_hash_round_trips_through_hex() {
        let mut bytes = [0u8; CONTENT_HASH_BYTES];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = (index * 7) as u8;
        }
        let hash = ContentHash::from_bytes(bytes);
        let text = hash.to_string();
        assert_eq!(text.len(), CONTENT_HASH_BYTES * 2);
        assert_eq!(&text[..6], "00070e");
        assert_eq!(ContentHash::parse_hex(&text).unwrap(), hash);
        assert_eq!(text.parse::<ContentHash>().unwrap(), hash);
        assert_eq!(hash.as_bytes(), &bytes);
        assert_eq!(format!("{hash:?}"), format!("ContentHash({text})"));

        let ron_text = ron::to_string(&hash).unwrap();
        assert_eq!(ron_text, format!("\"{text}\""));
        assert_eq!(ron::from_str::<ContentHash>(&ron_text).unwrap(), hash);

        let zero = ContentHash::from_bytes([0; CONTENT_HASH_BYTES]);
        assert_eq!(zero.to_string(), "0".repeat(CONTENT_HASH_BYTES * 2));
    }

    #[test]
    fn content_hash_rejects_bad_hex() {
        let valid = "a".repeat(CONTENT_HASH_BYTES * 2);
        assert_eq!(
            ContentHash::parse_hex(&valid[..63]).unwrap_err(),
            ContentHashError::InvalidLength { len: 63 }
        );
        assert_eq!(
            ContentHash::parse_hex(&format!("{valid}a")).unwrap_err(),
            ContentHashError::InvalidLength { len: 65 }
        );
        assert_eq!(
            ContentHash::parse_hex("").unwrap_err(),
            ContentHashError::InvalidLength { len: 0 }
        );

        let non_hex = format!("gg{}", &valid[2..]);
        assert_eq!(
            ContentHash::parse_hex(&non_hex).unwrap_err(),
            ContentHashError::InvalidDigit {
                position: 0,
                byte: b'g'
            }
        );
        // Uppercase is rejected so the text form stays canonical and diffable.
        let upper = format!("{}A{}", &valid[..4], &valid[5..]);
        assert_eq!(
            ContentHash::parse_hex(&upper).unwrap_err(),
            ContentHashError::InvalidDigit {
                position: 4,
                byte: b'A'
            }
        );
        assert!(ron::from_str::<ContentHash>("\"not-a-hash\"").is_err());
    }
}
