//! Canonical world state.
//!
//! The authoritative record of durable game facts: player progression, quests,
//! factions, NPC facts, region state, and world flags. Serializable,
//! versioned, and independent of the runtime ECS, which is a projection of
//! this rather than the source of truth.
//!
//! # What belongs here
//!
//! A fact belongs in canonical state when it cannot be recomputed from content
//! plus the rest of this structure. Transforms, velocities, animation state,
//! navigation caches, particle state, and every runtime handle are derived or
//! transient, so they live in the ECS and are rebuilt on load. Conversely, "the
//! player completed this objective" survives a content revision that changes
//! how many kills the objective requires, so it is recorded rather than
//! inferred.
//!
//! # Determinism
//!
//! Every container here is ordered (`BTreeMap`/`BTreeSet`). Replay folds an
//! event log over a seeded initial state and must reproduce this structure
//! exactly, and saves must serialize byte-stably so content hashes and diffs
//! mean something; hash iteration order would break both. For the same reason
//! no field is a float: comparison and replay equality must not depend on
//! cross-platform floating-point behaviour, so bounded integers are used even
//! where a designer would think in percentages.
//!
//! # Mutation
//!
//! This module deliberately exposes no mutators. Every durable change is a
//! domain event applied by a pure reducer (`crate::events`), so that the log,
//! replay, and save state cannot disagree about how a fact came to be. Fields
//! are public because reducers need to write them; callers outside the
//! reducers should read only.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::ids::{
    AbilityId, ClassId, ClueId, ContentRevisionId, FactionId, Id, IdError, ItemId, LocationId, NpcId,
    ObjectiveId, QuestId, QuestResolutionId, RegionId, TalentId, WorldFlag,
};

/// Version of the [`WorldState`] shape itself.
///
/// Distinct from the save envelope's own version (ADR-0013): the envelope may
/// change without the state changing, and vice versa. Bump this whenever a
/// field is added, removed, or reinterpreted, and ship a migration with it —
/// `deny_unknown_fields` means an unversioned addition makes older builds
/// refuse the save rather than silently drop the field.
pub const WORLD_SCHEMA_VERSION: u32 = 2;

/// Simulation ticks in one in-world day.
///
/// The clock counts fixed simulation ticks (ADR-0007 expresses all timing in
/// ticks, never in seconds, so behaviour is frame-rate independent). At the
/// engine's default 60 Hz fixed step this makes an in-world day 20 real
/// minutes. `game_core` cannot depend on the engine to read that rate, so the
/// relationship is an assumption stated here rather than a computed value: if
/// the fixed step ever changes, day length changes with it, and this constant
/// is what to revisit.
pub const TICKS_PER_DAY: u64 = 72_000;

/// Coarse phase of the in-world day, derived from the clock.
///
/// Derived rather than stored: storing it would let it disagree with the tick
/// count it came from, and replay would then have two sources of truth.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TimeOfDay {
    /// First quarter of the day, which begins at dawn.
    Dawn,
    /// Second quarter.
    Day,
    /// Third quarter.
    Dusk,
    /// Final quarter.
    Night,
}

/// In-world time, as a count of fixed simulation ticks.
///
/// Wall-clock time is deliberately absent: a save that recorded when it was
/// written would make replay depend on the machine that ran it, and in-world
/// scheduling (shop hours, patrols, timed quests) needs in-world time anyway.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameClock {
    /// Ticks elapsed since the world began. Monotonic; only a reducer advances it.
    pub tick: u64,
}

impl GameClock {
    /// A clock at `tick`.
    pub const fn from_tick(tick: u64) -> Self {
        Self { tick }
    }

    /// Whole in-world days elapsed, counting from day zero.
    pub const fn day(self) -> u64 {
        self.tick / TICKS_PER_DAY
    }

    /// Ticks elapsed within the current day.
    pub const fn tick_of_day(self) -> u64 {
        self.tick % TICKS_PER_DAY
    }

    /// The phase of the current day.
    pub const fn time_of_day(self) -> TimeOfDay {
        match self.tick_of_day() * 4 / TICKS_PER_DAY {
            0 => TimeOfDay::Dawn,
            1 => TimeOfDay::Day,
            2 => TimeOfDay::Dusk,
            _ => TimeOfDay::Night,
        }
    }
}

/// A depletable resource with a permanent ceiling.
///
/// `max` is durable because permanent upgrades must survive a content revision
/// that retunes the class table it originally came from; recomputing it from
/// content would silently rewrite a player's progression when the numbers
/// change. `current` is durable because ADR-0007's respawn rule keeps progress,
/// so health carries across death and across a save.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pool {
    /// Present value, never above `max` when built through the constructors.
    pub current: u32,
    /// Ceiling, raised only by durable progression.
    pub max: u32,
}

impl Pool {
    /// A full pool.
    pub const fn full(max: u32) -> Self {
        Self { current: max, max }
    }

    /// A partially filled pool, clamping `current` into range.
    pub const fn new(current: u32, max: u32) -> Self {
        Self {
            current: if current > max { max } else { current },
            max,
        }
    }

    /// Whether the pool is empty.
    pub const fn is_depleted(self) -> bool {
        self.current == 0
    }

    /// Whether the pool is at its ceiling.
    pub const fn is_full(self) -> bool {
        self.current >= self.max
    }
}

/// A bounded standing on a symmetric integer scale.
///
/// Integer, not float: reputation is compared, serialized, replayed, and summed
/// from deltas, and float drift across platforms would make two replays of the
/// same log disagree. Bounded so that an unbounded chain of deltas cannot
/// overflow or turn a "hated" faction friendly by wrapping. The range is a
/// serialization property rather than game balance — what counts as friendly is
/// content's business, not this module's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Reputation(i16);

impl Reputation {
    /// Most hostile representable standing.
    pub const MIN: Self = Self(-1000);
    /// Most favourable representable standing.
    pub const MAX: Self = Self(1000);
    /// The starting standing for a faction the player has not affected.
    pub const NEUTRAL: Self = Self(0);

    /// Clamps `value` into `MIN..=MAX`.
    pub const fn new(value: i16) -> Self {
        if value < Self::MIN.0 {
            Self::MIN
        } else if value > Self::MAX.0 {
            Self::MAX
        } else {
            Self(value)
        }
    }

    /// The underlying value.
    pub const fn get(self) -> i16 {
        self.0
    }

    /// Applies a delta, saturating at the bounds.
    pub const fn saturating_add(self, delta: i16) -> Self {
        Self::new(self.0.saturating_add(delta))
    }
}

impl fmt::Display for Reputation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl Serialize for Reputation {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_i16(self.0)
    }
}

impl<'de> Deserialize<'de> for Reputation {
    /// Clamps rather than rejects an out-of-range value. A save whose
    /// reputation sits outside the bounds is either hand-edited or was written
    /// under different bounds; clamping keeps the sign and lets the world load,
    /// whereas refusing would make a cosmetic out-of-range value cost the
    /// player their save. This is a repair, not a dropped field — the
    /// `deny_unknown_fields` rule is about losing data, and no data is lost.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        i16::deserialize(deserializer).map(Self::new)
    }
}

/// Where a piece of equipment is worn.
///
/// An enum rather than a content id: the slot set is engine-and-animation
/// shaped, so adding one is a code change by nature. Keying equipment by slot
/// makes "two items in one slot" unrepresentable instead of merely invalid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EquipmentSlot {
    /// Primary weapon.
    Weapon,
    /// Shield or secondary implement.
    Offhand,
    /// Body armour.
    Armor,
    /// Accessory.
    Trinket,
}

impl fmt::Display for EquipmentSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Weapon => "weapon",
            Self::Offhand => "offhand",
            Self::Armor => "armor",
            Self::Trinket => "trinket",
        })
    }
}

/// Durable character progression.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Progression {
    /// The chosen class, or `None` before character creation.
    ///
    /// Optional because a world exists before a class is picked, and inventing
    /// a placeholder class id would put a name in canonical state that no
    /// content file defines.
    pub class: Option<ClassId>,
    /// Current level. Durable: derived from experience only under the *current*
    /// levelling curve, and a retuned curve must not demote a player.
    pub level: u16,
    /// Accumulated experience.
    pub experience: u32,
    /// Talent nodes taken (ADR-0018). Spent points are derived from these,
    /// never stored, so the two cannot disagree.
    pub talents: BTreeSet<TalentId>,
    /// What is slotted on the five ability keys (ADR-0018). `None` until the
    /// player first arranges it, which means the class's own abilities in
    /// order — so a character who never opens the panel, and every save from
    /// before loadouts existed, plays exactly as before.
    pub loadout: Option<[Option<AbilityId>; LOADOUT_SLOTS]>,
}

/// How many abilities are slotted at once: one per key, Q R F C V (ADR-0018).
pub const LOADOUT_SLOTS: usize = 5;

/// Everything durable about the player.
///
/// Currency is deliberately not a field: it is an item in `inventory`, so
/// grants, costs, and validation all go through one allowlisted path instead of
/// two.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlayerState {
    /// Hit points.
    pub health: Pool,
    /// The action resource gating ADR-0007's committed attacks and dodges.
    pub stamina: Pool,
    /// Where the player is, as an authored location id — never a transform.
    ///
    /// The exact spawn point within the location comes from content, so this
    /// stays a stable id that survives a level being re-laid-out. `None` before
    /// the player has been placed.
    pub location: Option<LocationId>,
    /// Item quantities. Absent key means none held; a zero quantity should be
    /// removed rather than stored, so that equality and serialization have one
    /// representation of "none".
    pub inventory: BTreeMap<ItemId, u32>,
    /// Worn equipment by slot.
    pub equipment: BTreeMap<EquipmentSlot, ItemId>,
    /// Class, level, and experience.
    pub progression: Progression,
}

impl PlayerState {
    /// Quantity of `item` held, zero when absent.
    pub fn item_count(&self, item: &ItemId) -> u32 {
        self.inventory.get(item).copied().unwrap_or(0)
    }

    /// The item worn in `slot`, if any.
    pub fn equipped(&self, slot: EquipmentSlot) -> Option<&ItemId> {
        self.equipment.get(&slot)
    }
}

/// How far a quest has got.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum QuestStatus {
    /// Never accepted. Equivalent to having no entry in [`WorldState::quests`]
    /// at all, which is why [`WorldState::quest_status`] returns it for an
    /// unknown quest instead of an `Option`.
    #[default]
    NotStarted,
    /// Accepted and in progress.
    Active,
    /// Finished, recording which ending occurred.
    Completed {
        /// The ending that occurred.
        resolution: QuestResolutionId,
    },
    /// Ended unsuccessfully and cannot be resumed.
    Failed,
}

impl QuestStatus {
    /// Whether the quest has finished, either way.
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed { .. } | Self::Failed)
    }
}

/// Progress against one quest objective.
///
/// `completed` is recorded rather than derived from `count`, because the
/// required count lives in content: a revision that raises "collect 3" to
/// "collect 5" must not un-complete an objective the player already finished.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectiveProgress {
    /// Countable progress, for objectives that accumulate.
    pub count: u32,
    /// Whether the objective has been satisfied.
    pub completed: bool,
}

/// Durable state of one quest.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuestState {
    /// Overall status.
    pub status: QuestStatus,
    /// Per-objective progress. Objectives absent here have not been started.
    pub objectives: BTreeMap<ObjectiveId, ObjectiveProgress>,
}

impl QuestState {
    /// Progress on `objective`, defaulting to untouched when absent.
    pub fn objective(&self, objective: &ObjectiveId) -> ObjectiveProgress {
        self.objectives.get(objective).copied().unwrap_or_default()
    }
}

/// Durable state of one faction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactionState {
    /// Standing with the player.
    pub reputation: Reputation,
}

/// Durable facts about one NPC.
///
/// Positions, animation, and combat state are runtime concerns rebuilt from
/// content on load; only what must survive a session is here.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NpcState {
    /// Whether the NPC is still alive. Durable and not derivable: quests gate
    /// on it and no other field records it.
    pub alive: bool,
    /// Personal standing toward the player, on the same bounded scale as
    /// faction reputation but tracked separately, since an NPC can like the
    /// player while their faction does not.
    pub disposition: Reputation,
    /// Where the NPC currently resides, as an authored location id.
    pub location: Option<LocationId>,
    /// Knowledge the NPC has acquired, which gates dialogue.
    pub known_clues: BTreeSet<ClueId>,
}

impl Default for NpcState {
    /// Alive, neutral, unplaced, and ignorant — the state of an NPC the world
    /// has just learned about. `derive(Default)` would make them dead.
    fn default() -> Self {
        Self {
            alive: true,
            disposition: Reputation::NEUTRAL,
            location: None,
            known_clues: BTreeSet::new(),
        }
    }
}

/// Durable state of one region.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionState {
    /// Whether the player has ever visited.
    pub discovered: bool,
    /// Whether the region's set-piece threat has been resolved.
    pub cleared: bool,
    /// Region-scoped facts, drawn from the same authored flag vocabulary as
    /// [`WorldState::world_flags`]; where a flag is stored is what makes it
    /// region-scoped.
    pub flags: BTreeSet<WorldFlag>,
}

/// The authoritative record of durable game facts.
///
/// Saves persist this, domain events mutate it, replay reconstructs it, AI
/// reads it as context, and a future server would replicate it. It contains no
/// runtime identity: `game_core` does not depend on the ECS at all (ADR-0002),
/// so that is structural rather than a rule to remember.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldState {
    /// Shape version of this structure; see [`WORLD_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// The seed every deterministic generator derives from. Durable because
    /// replay and procedural generation must reproduce the same world.
    pub world_seed: u64,
    /// The compiled content revision this state was last validated against.
    pub content_revision: ContentRevisionId,
    /// In-world time.
    pub clock: GameClock,
    /// The player.
    pub player: PlayerState,
    /// Per-region state, for regions the world has recorded anything about.
    pub regions: BTreeMap<RegionId, RegionState>,
    /// Per-NPC durable facts.
    pub npcs: BTreeMap<NpcId, NpcState>,
    /// Per-faction standing.
    pub factions: BTreeMap<FactionId, FactionState>,
    /// Per-quest progress.
    pub quests: BTreeMap<QuestId, QuestState>,
    /// Global durable facts.
    pub world_flags: BTreeSet<WorldFlag>,
}

impl WorldState {
    /// A valid, empty world at tick zero.
    ///
    /// The maps start empty rather than pre-populated from content: absence
    /// means "nothing has happened to this yet", which keeps a new save small
    /// and keeps the accessors' defaults (below) the single definition of an
    /// untouched entity's state.
    pub fn new(world_seed: u64, content_revision: ContentRevisionId) -> Self {
        Self {
            schema_version: WORLD_SCHEMA_VERSION,
            world_seed,
            content_revision,
            clock: GameClock::default(),
            player: PlayerState::default(),
            regions: BTreeMap::new(),
            npcs: BTreeMap::new(),
            factions: BTreeMap::new(),
            quests: BTreeMap::new(),
            world_flags: BTreeSet::new(),
        }
    }

    /// State of `quest`, or [`QuestStatus::NotStarted`] if the world has never
    /// heard of it.
    pub fn quest_status(&self, quest: &QuestId) -> QuestStatus {
        self.quests
            .get(quest)
            .map_or(QuestStatus::NotStarted, |state| state.status.clone())
    }

    /// Progress on one objective, defaulting to untouched.
    pub fn objective_progress(&self, quest: &QuestId, objective: &ObjectiveId) -> ObjectiveProgress {
        self.quests
            .get(quest)
            .map_or_else(ObjectiveProgress::default, |state| state.objective(objective))
    }

    /// Standing with `faction`, defaulting to [`Reputation::NEUTRAL`].
    pub fn reputation(&self, faction: &FactionId) -> Reputation {
        self.factions
            .get(faction)
            .map_or(Reputation::NEUTRAL, |state| state.reputation)
    }

    /// Whether a global flag is set.
    pub fn has_world_flag(&self, flag: &WorldFlag) -> bool {
        self.world_flags.contains(flag)
    }

    /// Whether a flag is set on a specific region.
    pub fn has_region_flag(&self, region: &RegionId, flag: &WorldFlag) -> bool {
        self.regions
            .get(region)
            .is_some_and(|state| state.flags.contains(flag))
    }

    /// Recorded state of one quest, if any.
    pub fn quest(&self, quest: &QuestId) -> Option<&QuestState> {
        self.quests.get(quest)
    }

    /// Recorded state of one region, if any.
    pub fn region(&self, region: &RegionId) -> Option<&RegionState> {
        self.regions.get(region)
    }

    /// Recorded facts about one NPC, if any.
    pub fn npc(&self, npc: &NpcId) -> Option<&NpcState> {
        self.npcs.get(npc)
    }

    /// Recorded state of one faction, if any.
    pub fn faction(&self, faction: &FactionId) -> Option<&FactionState> {
        self.factions.get(faction)
    }

    /// Rejects any instance-scoped id anywhere in canonical state.
    ///
    /// ADR-0010's central rule is that generated names never leak into the
    /// persistent world, and serde cannot enforce it per field: a save
    /// containing `"inst:x/y"` in a [`QuestId`] deserializes happily. Registries
    /// call `require_canonical()` at their own edges, but a rule that depends on
    /// every future caller remembering a guard is a rule that will be broken
    /// silently. This walk is the backstop, run at the save/load boundary: it
    /// visits every id in the structure, so a missed guard becomes a loud error
    /// naming the id and where it was found rather than a permanent NPC that an
    /// AI invented.
    ///
    /// It validates the id namespace only. Reference resolution (does this
    /// quest exist in content?) belongs to `content_runtime`, which has the
    /// registries; this module has no way to know.
    pub fn validate_canonical(&self) -> Result<(), WorldValidationError> {
        check(self.content_revision.as_id(), || IdSite::ContentRevision)?;
        self.validate_player()?;
        self.validate_quests()?;
        self.validate_npcs()?;

        for faction in self.factions.keys() {
            check(faction.as_id(), || IdSite::Faction)?;
        }
        for (region_id, region) in &self.regions {
            check(region_id.as_id(), || IdSite::Region)?;
            for flag in &region.flags {
                check(flag.as_id(), || IdSite::RegionFlag {
                    region: region_id.clone(),
                })?;
            }
        }
        for flag in &self.world_flags {
            check(flag.as_id(), || IdSite::WorldFlag)?;
        }
        Ok(())
    }

    fn validate_player(&self) -> Result<(), WorldValidationError> {
        if let Some(location) = &self.player.location {
            check(location.as_id(), || IdSite::PlayerLocation)?;
        }
        if let Some(class) = &self.player.progression.class {
            check(class.as_id(), || IdSite::PlayerClass)?;
        }
        for talent in &self.player.progression.talents {
            check(talent.as_id(), || IdSite::PlayerTalent)?;
        }
        for ability in self.player.progression.loadout.iter().flatten().flatten() {
            check(ability.as_id(), || IdSite::LoadoutAbility)?;
        }
        for item in self.player.inventory.keys() {
            check(item.as_id(), || IdSite::InventoryItem)?;
        }
        for (slot, item) in &self.player.equipment {
            check(item.as_id(), || IdSite::EquippedItem { slot: *slot })?;
        }
        Ok(())
    }

    fn validate_quests(&self) -> Result<(), WorldValidationError> {
        for (quest_id, quest) in &self.quests {
            check(quest_id.as_id(), || IdSite::Quest)?;
            if let QuestStatus::Completed { resolution } = &quest.status {
                check(resolution.as_id(), || IdSite::QuestResolution {
                    quest: quest_id.clone(),
                })?;
            }
            for objective in quest.objectives.keys() {
                check(objective.as_id(), || IdSite::QuestObjective {
                    quest: quest_id.clone(),
                })?;
            }
        }
        Ok(())
    }

    fn validate_npcs(&self) -> Result<(), WorldValidationError> {
        for (npc_id, npc) in &self.npcs {
            check(npc_id.as_id(), || IdSite::Npc)?;
            if let Some(location) = &npc.location {
                check(location.as_id(), || IdSite::NpcLocation { npc: npc_id.clone() })?;
            }
            for clue in &npc.known_clues {
                check(clue.as_id(), || IdSite::NpcKnownClue { npc: npc_id.clone() })?;
            }
        }
        Ok(())
    }
}

/// Checks one id, building the (allocating) site description only on failure.
fn check(id: &Id, site: impl FnOnce() -> IdSite) -> Result<(), WorldValidationError> {
    match id.require_canonical() {
        Ok(_) => Ok(()),
        Err(source) => Err(WorldValidationError {
            site: site(),
            id: id.clone(),
            source,
        }),
    }
}

/// Where in [`WorldState`] an offending id was found.
///
/// Structured rather than a preformatted string so a caller can group or filter
/// failures; the owning id is carried where the position alone would not
/// identify the entry, because "an objective id was scoped" is not actionable
/// without knowing which quest it was on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IdSite {
    /// The world's content revision.
    ContentRevision,
    /// The player's location.
    PlayerLocation,
    /// The player's class.
    PlayerClass,
    /// A talent the player has taken (ADR-0018).
    PlayerTalent,
    /// An ability slotted in the player's loadout (ADR-0018).
    LoadoutAbility,
    /// A key in the player's inventory.
    InventoryItem,
    /// An item worn by the player.
    EquippedItem {
        /// The slot holding it.
        slot: EquipmentSlot,
    },
    /// A key in the quest map.
    Quest,
    /// The resolution recorded for a completed quest.
    QuestResolution {
        /// The quest that completed.
        quest: QuestId,
    },
    /// An objective key within a quest.
    QuestObjective {
        /// The quest owning the objective.
        quest: QuestId,
    },
    /// A key in the NPC map.
    Npc,
    /// An NPC's location.
    NpcLocation {
        /// The NPC.
        npc: NpcId,
    },
    /// A clue known to an NPC.
    NpcKnownClue {
        /// The NPC.
        npc: NpcId,
    },
    /// A key in the faction map.
    Faction,
    /// A key in the region map.
    Region,
    /// A flag set on a region.
    RegionFlag {
        /// The region.
        region: RegionId,
    },
    /// A global world flag.
    WorldFlag,
}

impl fmt::Display for IdSite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContentRevision => f.write_str("the content revision"),
            Self::PlayerLocation => f.write_str("the player's location"),
            Self::PlayerClass => f.write_str("the player's class"),
            Self::PlayerTalent => f.write_str("a talent the player has taken"),
            Self::LoadoutAbility => f.write_str("an ability in the player's loadout"),
            Self::InventoryItem => f.write_str("the player's inventory"),
            Self::EquippedItem { slot } => write!(f, "the player's {slot} equipment slot"),
            Self::Quest => f.write_str("the quest map"),
            Self::QuestResolution { quest } => write!(f, "the resolution of quest `{quest}`"),
            Self::QuestObjective { quest } => write!(f, "an objective of quest `{quest}`"),
            Self::Npc => f.write_str("the npc map"),
            Self::NpcLocation { npc } => write!(f, "the location of npc `{npc}`"),
            Self::NpcKnownClue { npc } => write!(f, "the clues known to npc `{npc}`"),
            Self::Faction => f.write_str("the faction map"),
            Self::Region => f.write_str("the region map"),
            Self::RegionFlag { region } => write!(f, "the flags of region `{region}`"),
            Self::WorldFlag => f.write_str("the world flags"),
        }
    }
}

/// An id that canonical state may not contain.
///
/// Names both the id and its position, because this error's whole job is to
/// turn "something leaked" into a bug report someone can act on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorldValidationError {
    /// Where the id was found.
    pub site: IdSite,
    /// The offending id, in its serialized form.
    pub id: Id,
    /// Why it was rejected.
    pub source: IdError,
}

impl fmt::Display for WorldValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} contains the id `{}`, which canonical world state may not hold: {}",
            self.site, self.id, self.source
        )
    }
}

impl std::error::Error for WorldValidationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Only the tests build scoped ids, to prove `validate_canonical` rejects
    // them. The library itself never constructs one.
    use crate::ids::InstanceId;

    fn revision() -> ContentRevisionId {
        ContentRevisionId::new("rev-0007").unwrap()
    }

    fn instance() -> InstanceId {
        InstanceId::new("crypt-of-ash").unwrap()
    }

    fn quest(text: &str) -> QuestId {
        QuestId::new(text).unwrap()
    }

    fn flag(text: &str) -> WorldFlag {
        WorldFlag::new(text).unwrap()
    }

    /// A world with something in every container, so serialization,
    /// round-tripping, and the validation walk all have work to do.
    fn populated() -> WorldState {
        let mut world = WorldState::new(0xfeed_face_dead_beef, revision());
        world.clock = GameClock::from_tick(TICKS_PER_DAY * 3 + 17);

        world.player.health = Pool::new(42, 120);
        world.player.stamina = Pool::full(80);
        world.player.location = Some(LocationId::new("cinderwake-square").unwrap());
        world.player.progression = Progression {
            class: Some(ClassId::new("warden").unwrap()),
            level: 4,
            experience: 1250,
            ..Progression::default()
        };
        world
            .player
            .inventory
            .insert(ItemId::new("ash-cinder").unwrap(), 3);
        world
            .player
            .inventory
            .insert(ItemId::new("bronze-mark").unwrap(), 250);
        world
            .player
            .equipment
            .insert(EquipmentSlot::Weapon, ItemId::new("ember-glaive").unwrap());
        world
            .player
            .equipment
            .insert(EquipmentSlot::Armor, ItemId::new("scorched-mail").unwrap());

        let mut objectives = BTreeMap::new();
        objectives.insert(
            ObjectiveId::new("light-the-beacon").unwrap(),
            ObjectiveProgress {
                count: 1,
                completed: true,
            },
        );
        objectives.insert(
            ObjectiveId::new("gather-cinders").unwrap(),
            ObjectiveProgress {
                count: 2,
                completed: false,
            },
        );
        world.quests.insert(
            quest("ember-oath"),
            QuestState {
                status: QuestStatus::Active,
                objectives,
            },
        );
        world.quests.insert(
            quest("amber-road"),
            QuestState {
                status: QuestStatus::Completed {
                    resolution: QuestResolutionId::new("spared-the-warden").unwrap(),
                },
                objectives: BTreeMap::new(),
            },
        );

        world.npcs.insert(
            NpcId::new("ember-warden").unwrap(),
            NpcState {
                alive: true,
                disposition: Reputation::new(180),
                location: Some(LocationId::new("cinderwake-forge").unwrap()),
                known_clues: [ClueId::new("the-sealed-door").unwrap()].into_iter().collect(),
            },
        );
        world.npcs.insert(
            NpcId::new("ash-acolyte").unwrap(),
            NpcState {
                alive: false,
                ..NpcState::default()
            },
        );

        world.factions.insert(
            FactionId::new("cinder-court").unwrap(),
            FactionState {
                reputation: Reputation::new(-320),
            },
        );

        world.regions.insert(
            RegionId::new("cinderwake").unwrap(),
            RegionState {
                discovered: true,
                cleared: false,
                flags: [flag("bridge-repaired")].into_iter().collect(),
            },
        );
        world
            .regions
            .insert(RegionId::new("ashfall-waste").unwrap(), RegionState::default());

        world.world_flags.insert(flag("prologue-complete"));
        world.world_flags.insert(flag("forge-relit"));
        world
    }

    #[test]
    fn new_world_is_empty_valid_and_round_trips() {
        let world = WorldState::new(7, revision());
        assert_eq!(world.schema_version, WORLD_SCHEMA_VERSION);
        assert_eq!(world.clock.tick, 0);
        assert!(world.quests.is_empty());
        assert!(world.player.progression.class.is_none());
        assert_eq!(world.validate_canonical(), Ok(()));

        let text = ron::to_string(&world).unwrap();
        assert_eq!(ron::from_str::<WorldState>(&text).unwrap(), world);
    }

    #[test]
    fn serialization_is_byte_stable() {
        let world = populated();
        let first = ron::to_string(&world).unwrap();
        let second = ron::to_string(&world).unwrap();
        assert_eq!(first, second);
        // And a clone, which has different allocations, serializes identically.
        assert_eq!(ron::to_string(&world.clone()).unwrap(), first);
    }

    /// The property replay depends on: the same logical state must produce the
    /// same bytes regardless of the order events happened to build it in. This
    /// holds because every container is ordered, and this test is what keeps a
    /// future `HashMap` from quietly breaking it.
    #[test]
    fn insertion_order_does_not_change_serialization() {
        let forward = populated();

        let mut reversed = WorldState::new(forward.world_seed, revision());
        reversed.clock = forward.clock;
        reversed.player = forward.player.clone();
        reversed.player.inventory = forward
            .player
            .inventory
            .iter()
            .rev()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        reversed.player.equipment = forward
            .player
            .equipment
            .iter()
            .rev()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        reversed.quests = forward
            .quests
            .iter()
            .rev()
            .map(|(id, state)| {
                let objectives = state
                    .objectives
                    .iter()
                    .rev()
                    .map(|(k, v)| (k.clone(), *v))
                    .collect();
                (
                    id.clone(),
                    QuestState {
                        status: state.status.clone(),
                        objectives,
                    },
                )
            })
            .collect();
        reversed.npcs = forward
            .npcs
            .iter()
            .rev()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        reversed.factions = forward
            .factions
            .iter()
            .rev()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        reversed.regions = forward
            .regions
            .iter()
            .rev()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        reversed.world_flags = forward.world_flags.iter().rev().cloned().collect();

        assert_eq!(reversed, forward);
        assert_eq!(
            ron::to_string(&reversed).unwrap(),
            ron::to_string(&forward).unwrap()
        );
    }

    #[test]
    fn populated_world_round_trips_through_ron() {
        let world = populated();
        let text = ron::to_string(&world).unwrap();
        let parsed: WorldState = ron::from_str(&text).unwrap();
        assert_eq!(parsed, world);
        // Round-tripping again reproduces the same bytes, so a save rewritten
        // without changes diffs empty.
        assert_eq!(ron::to_string(&parsed).unwrap(), text);
    }

    /// Ids must appear as bare strings, not struct-shaped, so saves stay
    /// readable and so the serialized form never depends on the newtype's name.
    #[test]
    fn ids_serialize_as_plain_strings() {
        let text = ron::to_string(&populated()).unwrap();
        assert!(text.contains("\"ember-oath\""), "{text}");
        assert!(text.contains("\"prologue-complete\""), "{text}");
        assert!(!text.contains("WorldFlag("), "{text}");
        assert!(!text.contains("ClassId("), "{text}");
    }

    /// ADR-0013: an unknown field is rejected, never dropped. A dropped field is
    /// how player state disappears between versions.
    #[test]
    fn unknown_fields_are_rejected() {
        let cases: &[(&str, &str)] = &[
            ("GameClock", "(tick: 5, bogus: 1)"),
            ("Pool", "(current: 1, max: 2, bogus: 1)"),
            (
                "Progression",
                "(class: None, level: 1, experience: 0, talents: [], loadout: None, bogus: 1)",
            ),
            ("ObjectiveProgress", "(count: 0, completed: false, bogus: 1)"),
            (
                "RegionState",
                "(discovered: true, cleared: false, flags: [], bogus: 1)",
            ),
            ("FactionState", "(reputation: 0, bogus: 1)"),
        ];
        for (name, text) in cases {
            let accepted = text.replace(", bogus: 1", "");
            match *name {
                "GameClock" => {
                    assert!(ron::from_str::<GameClock>(&accepted).is_ok(), "{name}");
                    assert!(ron::from_str::<GameClock>(text).is_err(), "{name}");
                }
                "Pool" => {
                    assert!(ron::from_str::<Pool>(&accepted).is_ok(), "{name}");
                    assert!(ron::from_str::<Pool>(text).is_err(), "{name}");
                }
                "Progression" => {
                    assert!(ron::from_str::<Progression>(&accepted).is_ok(), "{name}");
                    assert!(ron::from_str::<Progression>(text).is_err(), "{name}");
                }
                "ObjectiveProgress" => {
                    assert!(ron::from_str::<ObjectiveProgress>(&accepted).is_ok(), "{name}");
                    assert!(ron::from_str::<ObjectiveProgress>(text).is_err(), "{name}");
                }
                "RegionState" => {
                    assert!(ron::from_str::<RegionState>(&accepted).is_ok(), "{name}");
                    assert!(ron::from_str::<RegionState>(text).is_err(), "{name}");
                }
                _ => {
                    assert!(ron::from_str::<FactionState>(&accepted).is_ok(), "{name}");
                    assert!(ron::from_str::<FactionState>(text).is_err(), "{name}");
                }
            }
        }
    }

    #[test]
    fn unknown_field_in_world_state_is_rejected() {
        let world = WorldState::new(1, revision());
        let text = ron::to_string(&world).unwrap();
        assert!(ron::from_str::<WorldState>(&text).is_ok());

        let tampered = text.replace("schema_version:", "bogus_field:1,schema_version:");
        assert!(ron::from_str::<WorldState>(&tampered).is_err(), "{tampered}");

        // A missing field is an error too, rather than a silent default: a save
        // that lost a field needs a migration, not a guess.
        let truncated = text.replace(&format!("world_seed:{},", world.world_seed), "");
        assert!(ron::from_str::<WorldState>(&truncated).is_err(), "{truncated}");
    }

    #[test]
    fn nested_quest_state_round_trips_and_rejects_unknown_fields() {
        let state = QuestState {
            status: QuestStatus::Completed {
                resolution: QuestResolutionId::new("spared-the-warden").unwrap(),
            },
            objectives: BTreeMap::new(),
        };
        let text = ron::to_string(&state).unwrap();
        assert_eq!(ron::from_str::<QuestState>(&text).unwrap(), state);
        assert!(ron::from_str::<QuestState>(&text.replace("status:", "bogus:1,status:")).is_err());
    }

    #[test]
    fn validate_canonical_accepts_a_canonical_world() {
        assert_eq!(populated().validate_canonical(), Ok(()));
    }

    /// Every distinct place an id can hide, each planted with an
    /// instance-scoped id that ADR-0010 forbids in the persistent world. If a
    /// field holding an id is added without extending the walk, the matching
    /// case here is the thing that was never written — which is the limit of
    /// what a test can do about it, and the reason the walk is written as an
    /// explicit enumeration rather than something clever.
    #[test]
    fn validate_canonical_rejects_a_scoped_id_in_every_location() {
        let scoped_text = "inst:crypt-of-ash/leaked";
        type Plant = (&'static str, fn(&mut WorldState));
        let plants: &[Plant] = &[
            ("the content revision", |world| {
                world.content_revision = ContentRevisionId::scoped(&instance(), "leaked").unwrap();
            }),
            ("the player's location", |world| {
                world.player.location = Some(LocationId::scoped(&instance(), "leaked").unwrap());
            }),
            ("the player's class", |world| {
                world.player.progression.class = Some(ClassId::scoped(&instance(), "leaked").unwrap());
            }),
            ("the player's inventory", |world| {
                world
                    .player
                    .inventory
                    .insert(ItemId::scoped(&instance(), "leaked").unwrap(), 1);
            }),
            ("the player's weapon equipment slot", |world| {
                world.player.equipment.insert(
                    EquipmentSlot::Weapon,
                    ItemId::scoped(&instance(), "leaked").unwrap(),
                );
            }),
            ("the quest map", |world| {
                world.quests.insert(
                    QuestId::scoped(&instance(), "leaked").unwrap(),
                    QuestState::default(),
                );
            }),
            ("the resolution of quest `amber-road`", |world| {
                let state = world.quests.get_mut(&quest("amber-road")).unwrap();
                state.status = QuestStatus::Completed {
                    resolution: QuestResolutionId::scoped(&instance(), "leaked").unwrap(),
                };
            }),
            ("an objective of quest `ember-oath`", |world| {
                let state = world.quests.get_mut(&quest("ember-oath")).unwrap();
                state.objectives.insert(
                    ObjectiveId::scoped(&instance(), "leaked").unwrap(),
                    ObjectiveProgress::default(),
                );
            }),
            ("the npc map", |world| {
                world
                    .npcs
                    .insert(NpcId::scoped(&instance(), "leaked").unwrap(), NpcState::default());
            }),
            ("the location of npc `ember-warden`", |world| {
                let npc = world.npcs.get_mut(&NpcId::new("ember-warden").unwrap()).unwrap();
                npc.location = Some(LocationId::scoped(&instance(), "leaked").unwrap());
            }),
            ("the clues known to npc `ember-warden`", |world| {
                let npc = world.npcs.get_mut(&NpcId::new("ember-warden").unwrap()).unwrap();
                npc.known_clues
                    .insert(ClueId::scoped(&instance(), "leaked").unwrap());
            }),
            ("the faction map", |world| {
                world.factions.insert(
                    FactionId::scoped(&instance(), "leaked").unwrap(),
                    FactionState::default(),
                );
            }),
            ("the region map", |world| {
                world.regions.insert(
                    RegionId::scoped(&instance(), "leaked").unwrap(),
                    RegionState::default(),
                );
            }),
            ("the flags of region `cinderwake`", |world| {
                let region = world
                    .regions
                    .get_mut(&RegionId::new("cinderwake").unwrap())
                    .unwrap();
                region
                    .flags
                    .insert(WorldFlag::scoped(&instance(), "leaked").unwrap());
            }),
            ("the world flags", |world| {
                world
                    .world_flags
                    .insert(WorldFlag::scoped(&instance(), "leaked").unwrap());
            }),
        ];

        for (expected_site, plant) in plants {
            let mut world = populated();
            plant(&mut world);
            let error = world
                .validate_canonical()
                .expect_err(&format!("expected a rejection for {expected_site}"));
            let message = error.to_string();
            assert!(message.contains(expected_site), "{message}");
            assert!(message.contains(scoped_text), "{message}");
            assert_eq!(error.id.to_string(), scoped_text);
            assert_eq!(error.source, IdError::UnexpectedScope);
            assert!(std::error::Error::source(&error).is_some());
        }
    }

    /// A scoped id in a save parses fine — which is exactly why the walk exists
    /// rather than relying on serde.
    #[test]
    fn serde_alone_does_not_reject_scoped_ids() {
        let mut world = populated();
        world
            .world_flags
            .insert(WorldFlag::scoped(&instance(), "leaked").unwrap());
        let text = ron::to_string(&world).unwrap();
        let parsed: WorldState = ron::from_str(&text).unwrap();
        assert_eq!(parsed, world);
        assert!(parsed.validate_canonical().is_err());
    }

    #[test]
    fn reputation_is_clamped_at_construction_and_on_deltas() {
        assert_eq!(Reputation::new(5_000), Reputation::MAX);
        assert_eq!(Reputation::new(-5_000), Reputation::MIN);
        assert_eq!(Reputation::new(0), Reputation::NEUTRAL);
        assert_eq!(Reputation::default(), Reputation::NEUTRAL);

        assert_eq!(Reputation::MAX.saturating_add(1), Reputation::MAX);
        assert_eq!(Reputation::MIN.saturating_add(-1), Reputation::MIN);
        // Saturating at the i16 bound, then clamped: no wrap-around in either step.
        assert_eq!(Reputation::MAX.saturating_add(i16::MAX), Reputation::MAX);
        assert_eq!(Reputation::MIN.saturating_add(i16::MIN), Reputation::MIN);
        assert_eq!(Reputation::NEUTRAL.saturating_add(-40).get(), -40);
    }

    #[test]
    fn reputation_serializes_as_a_bare_integer_and_clamps_on_load() {
        let value = Reputation::new(-320);
        assert_eq!(ron::to_string(&value).unwrap(), "-320");
        assert_eq!(ron::from_str::<Reputation>("-320").unwrap(), value);
        // Out of range in a hand-edited save: repaired, not refused, so the
        // world still loads.
        assert_eq!(ron::from_str::<Reputation>("30000").unwrap(), Reputation::MAX);
        assert_eq!(ron::from_str::<Reputation>("-30000").unwrap(), Reputation::MIN);
        // Beyond i16 is still an error; the clamp is not a licence to accept junk.
        assert!(ron::from_str::<Reputation>("99999999").is_err());
    }

    #[test]
    fn pool_clamps_current_to_max() {
        assert_eq!(
            Pool::new(200, 120),
            Pool {
                current: 120,
                max: 120
            }
        );
        assert_eq!(Pool::full(80), Pool { current: 80, max: 80 });
        assert!(Pool::new(0, 10).is_depleted());
        assert!(!Pool::new(1, 10).is_depleted());
        assert!(Pool::full(10).is_full());
        assert!(!Pool::new(9, 10).is_full());
        assert_eq!(Pool::default(), Pool { current: 0, max: 0 });
    }

    #[test]
    fn clock_derives_day_and_phase_without_storing_them() {
        let clock = GameClock::from_tick(TICKS_PER_DAY * 3 + 17);
        assert_eq!(clock.day(), 3);
        assert_eq!(clock.tick_of_day(), 17);
        assert_eq!(clock.time_of_day(), TimeOfDay::Dawn);

        assert_eq!(GameClock::default().time_of_day(), TimeOfDay::Dawn);
        assert_eq!(
            GameClock::from_tick(TICKS_PER_DAY / 4).time_of_day(),
            TimeOfDay::Day
        );
        assert_eq!(
            GameClock::from_tick(TICKS_PER_DAY / 2).time_of_day(),
            TimeOfDay::Dusk
        );
        assert_eq!(
            GameClock::from_tick(TICKS_PER_DAY * 3 / 4).time_of_day(),
            TimeOfDay::Night
        );
        assert_eq!(
            GameClock::from_tick(TICKS_PER_DAY - 1).time_of_day(),
            TimeOfDay::Night
        );
        // The phase depends only on position within the day, not on which day.
        assert_eq!(
            GameClock::from_tick(TICKS_PER_DAY * 99 + TICKS_PER_DAY / 2).time_of_day(),
            TimeOfDay::Dusk
        );
    }

    #[test]
    fn accessors_report_untouched_defaults_for_absent_entries() {
        let world = populated();
        let unknown_quest = quest("never-offered");
        let objective = ObjectiveId::new("gather-cinders").unwrap();

        assert_eq!(world.quest_status(&unknown_quest), QuestStatus::NotStarted);
        assert_eq!(world.quest(&unknown_quest), None);
        assert_eq!(
            world.objective_progress(&unknown_quest, &objective),
            ObjectiveProgress::default()
        );
        assert_eq!(
            world.reputation(&FactionId::new("unmet-order").unwrap()),
            Reputation::NEUTRAL
        );
        assert_eq!(world.npc(&NpcId::new("nobody").unwrap()), None);
        assert!(!world.has_world_flag(&flag("never-set")));
        assert!(!world.has_region_flag(&RegionId::new("nowhere").unwrap(), &flag("bridge-repaired")));
    }

    #[test]
    fn accessors_report_recorded_state() {
        let world = populated();
        assert_eq!(world.quest_status(&quest("ember-oath")), QuestStatus::Active);
        assert!(world.quest_status(&quest("amber-road")).is_terminal());
        assert!(!world.quest_status(&quest("ember-oath")).is_terminal());
        assert_eq!(
            world.objective_progress(&quest("ember-oath"), &ObjectiveId::new("gather-cinders").unwrap()),
            ObjectiveProgress {
                count: 2,
                completed: false
            }
        );
        assert_eq!(
            world.reputation(&FactionId::new("cinder-court").unwrap()).get(),
            -320
        );
        assert!(world.has_world_flag(&flag("forge-relit")));
        assert!(world.has_region_flag(&RegionId::new("cinderwake").unwrap(), &flag("bridge-repaired")));
        assert!(!world.has_region_flag(&RegionId::new("ashfall-waste").unwrap(), &flag("bridge-repaired")));
        assert!(
            world
                .region(&RegionId::new("cinderwake").unwrap())
                .unwrap()
                .discovered
        );

        assert_eq!(world.player.item_count(&ItemId::new("ash-cinder").unwrap()), 3);
        assert_eq!(world.player.item_count(&ItemId::new("nothing-held").unwrap()), 0);
        assert_eq!(
            world.player.equipped(EquipmentSlot::Weapon),
            Some(&ItemId::new("ember-glaive").unwrap())
        );
        assert_eq!(world.player.equipped(EquipmentSlot::Trinket), None);
    }

    #[test]
    fn npcs_default_to_alive() {
        let npc = NpcState::default();
        assert!(npc.alive);
        assert_eq!(npc.disposition, Reputation::NEUTRAL);
        assert!(npc.location.is_none());
        assert!(npc.known_clues.is_empty());
    }

    #[test]
    fn world_ids_behave_like_the_ids_module_newtypes() {
        let canonical = WorldFlag::new("forge-relit").unwrap();
        assert!(canonical.is_canonical());
        assert_eq!(canonical.name(), "forge-relit");
        assert_eq!(canonical.to_string(), "forge-relit");
        assert_eq!(canonical.require_canonical().unwrap(), &canonical);
        assert_eq!("forge-relit".parse::<WorldFlag>().unwrap(), canonical);
        assert_eq!(Id::from(canonical.clone()), *canonical.as_id());

        let scoped = WorldFlag::scoped(&instance(), "forge-relit").unwrap();
        assert!(!scoped.is_canonical());
        assert_eq!(scoped.to_string(), "inst:crypt-of-ash/forge-relit");
        assert_eq!(scoped.require_canonical().unwrap_err(), IdError::UnexpectedScope);
        assert_ne!(scoped, canonical);

        // Validation happens at construction and at parse time, not later.
        assert!(ClassId::new("Warden").is_err());
        assert!(QuestResolutionId::new("").is_err());
        assert!(ron::from_str::<ClassId>("\"../../etc/passwd\"").is_err());
        assert_eq!(ron::to_string(&canonical).unwrap(), "\"forge-relit\"");
    }
}
