//! Shared content primitives: the schema version, fixed-point numerics,
//! and the small id-like newtypes the definitions are built from.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::ids::AbilityId;

/// Version of the content schemas in this module.
///
/// Every definition carries it so the loader can reject or migrate a file
/// written against an older shape, rather than misreading it.
pub const CONTENT_SCHEMA_VERSION: u32 = 1;

/// Scale of [`Fixed`]: values are stored in thousandths.
pub const FIXED_SCALE: i64 = 1_000;

/// A fixed-point number in milli-units.
///
/// Stored and serialized as an integer count of thousandths, so `1.5` damage is
/// `1500`. Exact, portable, and replay-stable — see the module docs for why
/// this is not an `f32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Fixed(i64);

impl Fixed {
    pub const ZERO: Self = Self(0);

    /// Builds a value from a raw count of thousandths.
    pub const fn from_milli(milli: i64) -> Self {
        Self(milli)
    }

    /// Builds a value from whole units.
    pub const fn from_whole(whole: i64) -> Self {
        Self(whole * FIXED_SCALE)
    }

    /// The raw count of thousandths.
    pub const fn as_milli(self) -> i64 {
        self.0
    }

    pub const fn is_negative(self) -> bool {
        self.0 < 0
    }

    /// Saturating addition; content values never wrap.
    pub const fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }
}

impl fmt::Display for Fixed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let whole = self.0 / FIXED_SCALE;
        let fraction = (self.0 % FIXED_SCALE).abs();
        if fraction == 0 {
            write!(f, "{whole}")
        } else {
            write!(f, "{whole}.{fraction:03}")
        }
    }
}

/// Kinds of damage an effect can deal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum DamageType {
    Physical,
    Fire,
    Frost,
    Corruption,
}

/// Resources an effect can modify.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum ResourceType {
    Health,
    Stamina,
}

/// Names a status effect an ability can apply.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StatusId(pub AbilityId);

impl std::fmt::Display for StatusId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Names a gameplay event an effect can emit.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GameEventId(pub AbilityId);
