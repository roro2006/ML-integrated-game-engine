//! The Cinderwake slice: authored content, projected and played.
//!
//! This is where M1–M4 finally meet. Content is loaded from `assets/content`,
//! canonical state is projected into entities that have *real transforms* for
//! the first time, and the fixed step runs movement, combat, and enemy
//! behaviour over them while quests and dialogue turn what happens into domain
//! events.
//!
//! # The one rule everything here follows
//!
//! Runtime state is rebuilt; domain events are recorded. Positions, health in
//! combat, aggro, and paths live in the ECS and are thrown away on load.
//! Quest progress, pickups, crafting, upgrades, and deaths quests gate on
//! become [`DomainEvent`]s. [`Slice::pending`] is the queue between the two:
//! systems push events, and [`Slice::commit`] folds them into canonical state
//! and appends them to the log.
//!
//! Nothing in the fixed step writes `WorldState` directly. That is what keeps
//! the slice replayable, and it is the property the whole milestone exists to
//! demonstrate.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use content_runtime::ContentRegistry;
use engine::debug_draw::DebugDraw;
use engine::ecs::{Component, Entity, Resource, World};
use engine::math::{Quat, Transform, Vec2, Vec3, Vec4};
use engine::primitives;
use engine::render::{
    MaterialDesc, MaterialHandle, MeshData, MeshHandle, MeshRenderer, Model, ModelPart, RenderAssets,
    RenderBounds, Visibility,
};
use engine::shapes::{self, FaceColors, Shape};
use game_core::content::{
    Fixed, LocationDef, LocationKind, Point, ResourceType, ShapeDef, SpawnRole, StationKind,
};
use game_core::events::{DomainError, DomainEvent, RegionChange, apply};
use game_core::ids::{
    AbilityId, ChoiceId, ClassId, ClueId, EncounterId, ItemId, LocationId, ModelId, NpcId, PrefabId, SlotId,
    SpawnId,
};
use game_core::log;
use game_core::save::{self, SaveEnvelope};
use game_core::world::WorldState;
use game_runtime::ability::{self, Aim, Candidate, Cast, Cooldowns, Outcome};
use game_runtime::combat::Poise;
use game_runtime::combat::{
    Attack, AttackPhase, AttackTiming, Attacker, Damage, Health, Resistances, Target, resolve_hit,
    resolve_loadout,
};
use game_runtime::enemy::{EnemyBrain, EnemyState, PathFollower};
use game_runtime::interaction::{loot, loot_seed};
use game_runtime::movement::{Capsule, CollisionWorld, Dodge, Stamina};
use game_runtime::navigation::NavMesh;
use game_runtime::projectile::{self, End, Launch, Projectile};
use game_runtime::selection::{self, Offer};
use game_runtime::status::Statuses;
use narrative::quest::Signal;

/// Marks the player's entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Player;

impl Component for Player {}

/// Marks geometry a location owns that is not a gameplay entity — the ground.
///
/// Tagged so [`clear_location`] can take it away again: a transition rebuilds
/// the whole location, and a floor left behind would stack one per visit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Scenery;

impl Component for Scenery {}

/// Marks an enemy and names the prefab it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Enemy {
    pub prefab: PrefabId,
    /// The spawn point that produced it, so a cleared encounter can be
    /// attributed to a place rather than to a runtime handle.
    pub spawn: SpawnId,
    /// The authored fight this placement belongs to, if any.
    ///
    /// Carried on the entity because by the time it dies, the placement it came
    /// from is the only way to tell whether a `Defeat` objective just finished.
    pub encounter: Option<EncounterId>,
}

impl Component for Enemy {}

/// Marks a talkable NPC.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Npc(pub NpcId);

impl Component for Npc {}

/// Marks the NPC currently walking with the player.
///
/// A marker rather than a scan by id: the follow system runs every tick, and
/// "which entity is this `NpcId`" is a lookup the ECS is better at than a
/// linear search through every NPC in the location.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Escorting;

impl Component for Escorting {}

/// How close an escorted NPC tries to stay.
///
/// Larger than [`INTERACT_RANGE`] so that following someone does not
/// permanently offer you a conversation with them.
pub(crate) const ESCORT_FOLLOW_DISTANCE: f32 = 2.8;
/// Metres per tick. A shade slower than the player, so they trail rather than
/// crowd, but fast enough not to be lost.
pub(crate) const ESCORT_SPEED: f32 = 0.068;

/// A harvestable node and what it yields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Gatherable {
    pub item: ItemId,
    pub spawn: SpawnId,
    /// Harvested nodes stay in the world as scenery; the flag is what stops a
    /// player from farming one bush forever.
    pub taken: bool,
}

impl Component for Gatherable {}

/// Something examinable that teaches a piece of knowledge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Clue {
    pub clue: ClueId,
    pub spawn: SpawnId,
    /// What examining it shows, if content wrote anything.
    pub text: Option<String>,
    /// Examined clues stay in the world as scenery. Re-examining one would let
    /// a counted objective be satisfied over and over by the same prop.
    pub examined: bool,
}

impl Component for Clue {}

/// A point at which the player settles an authored decision.
///
/// Named `ChoicePoint` rather than `Choice` because `narrative::Choice` is a
/// line of dialogue on offer, and having two `Choice` types in one file is how
/// a wrong import becomes a confusing compile error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChoicePoint {
    pub choice: ChoiceId,
    pub spawn: SpawnId,
    /// What the confirm panel asks, if content wrote it.
    pub prompt: Option<String>,
    /// A decision is made once. Letting it be re-made would mean the branch a
    /// quest recorded could disagree with the last thing the player did.
    pub settled: bool,
}

impl Component for ChoicePoint {}

/// A fixture the player can use: an upgrade board, a doorway to elsewhere.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Station {
    pub spawn: SpawnId,
    /// What this fixture is for, as content declares it.
    pub kind: StationKind,
    /// Where this leads, if it is a doorway.
    pub leads_to: Option<LocationId>,
}

impl Component for Station {}

/// Which direction an entity is facing, as a unit vector on the ground plane.
///
/// Separate from [`Transform`] because combat needs a facing even when nothing
/// is moving, and a rotation quaternion is an awkward thing to ask "which way
/// is this swinging" of.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Facing(pub Vec3);

impl Component for Facing {}

impl Default for Facing {
    fn default() -> Self {
        Self(Vec3::new(0.0, 0.0, -1.0))
    }
}

/// How far the player can reach to talk, harvest, or use a station.
pub(crate) const INTERACT_RANGE: f32 = 2.5;

/// Metres per fixed tick at a walk.
pub(crate) const PLAYER_SPEED: f32 = 0.075;

/// The player's swing, in ticks (ADR-0007: committed attacks).
///
/// Faster than any enemy's so that trading blows favours the player slightly,
/// and short enough that a missed swing is a punish window rather than a
/// sentence.
pub(crate) const PLAYER_ATTACK_TIMING: AttackTiming = AttackTiming {
    windup: 10,
    active: 5,
    recovery: 18,
};
pub(crate) const PLAYER_ATTACK_DAMAGE: f32 = 22.0;
pub(crate) const PLAYER_ATTACK_REACH: f32 = 1.4;
pub(crate) const PLAYER_ATTACK_RADIUS: f32 = 0.9;
/// Stamina a swing and a dodge cost. Both matter: an action resource nobody
/// runs out of is not gating anything.
pub(crate) const PLAYER_ATTACK_COST: f32 = 25.0;
/// The player's stance (M8): a husk's 8-point swings take four to break it, a
/// warden's 11 three, so trading blows with a crowd is a way to be staggered.
pub(crate) const PLAYER_POISE: f32 = 30.0;
pub(crate) const PLAYER_DODGE_COST: f32 = 20.0;
pub(crate) const DODGE_DURATION: u32 = game_core::content::DODGE_TICKS;
/// I-frames cover the first two-thirds of the roll, so a well-timed dodge beats
/// a swing but a late one still eats it.
pub(crate) const DODGE_IFRAMES: u32 = 12;
pub(crate) const DODGE_SPEED: f32 = 0.22;

/// How the player's class fights (M9): its swing, stance, and roll, resolved
/// once from content when the player is placed, with every number content
/// leaves out taken from the constants above — so a class that says nothing
/// plays exactly as the wayfinder always has.
///
/// Runtime only, and re-derived from the class id every time the player is
/// placed: retuning a class retunes everyone playing it, which is what a
/// tuning pass wants. Pools are different — they are canonical, copied when
/// the class is chosen — because health is a fact about a character.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PlayerCombat {
    pub timing: AttackTiming,
    pub damage: f32,
    pub reach: f32,
    pub radius: f32,
    pub cost: f32,
    pub poise: f32,
    pub dodge_cost: f32,
    pub dodge_iframes: u32,
}

impl Component for PlayerCombat {}

impl Default for PlayerCombat {
    fn default() -> Self {
        Self {
            timing: PLAYER_ATTACK_TIMING,
            damage: PLAYER_ATTACK_DAMAGE,
            reach: PLAYER_ATTACK_REACH,
            radius: PLAYER_ATTACK_RADIUS,
            cost: PLAYER_ATTACK_COST,
            poise: PLAYER_POISE,
            dodge_cost: PLAYER_DODGE_COST,
            dodge_iframes: DODGE_IFRAMES,
        }
    }
}

impl PlayerCombat {
    /// A class's numbers over the defaults.
    pub(crate) fn of(combat: &game_core::content::ClassCombat) -> Self {
        let base = Self::default();
        let whole = |value: Option<Fixed>, default: f32| value.map_or(default, metres);
        Self {
            timing: AttackTiming {
                windup: combat.windup_ticks.unwrap_or(base.timing.windup),
                active: combat.active_ticks.unwrap_or(base.timing.active),
                recovery: combat.recovery_ticks.unwrap_or(base.timing.recovery),
            },
            damage: whole(combat.damage, base.damage),
            reach: whole(combat.reach, base.reach),
            radius: whole(combat.radius, base.radius),
            cost: whole(combat.cost, base.cost),
            poise: whole(combat.poise, base.poise),
            dodge_cost: whole(combat.dodge_cost, base.dodge_cost),
            dodge_iframes: combat
                .dodge_iframes
                .unwrap_or(base.dodge_iframes)
                .min(DODGE_DURATION),
        }
    }
}

fn metres(value: Fixed) -> f32 {
    value.as_milli() as f32 / 1000.0
}

/// A content position in engine metres.
pub(crate) fn to_world(point: Point) -> Vec3 {
    Vec3::new(metres(point.x), metres(point.y), metres(point.z))
}

/// Why the slice could not start.
#[derive(Debug)]
pub(crate) enum SliceError {
    /// Content is missing or invalid. Loading a game with broken content is
    /// worse than refusing to: the player would find out at the first dangling
    /// reference, mid-quest.
    Content(Box<content_runtime::ContentReport>),
    /// Content loaded, but the location the slice starts in is not in it.
    UnknownLocation(LocationId),
    /// Content defines no class for the player to start as, so there are no
    /// pools to give them (ADR-0015).
    UnknownClass(String),
    /// An event could not be applied to canonical state.
    Domain(DomainError),
    /// The slice's own revision id is malformed, which is a build-time mistake
    /// in [`SLICE_REVISION`] rather than anything the player did.
    Revision(game_core::ids::IdError),
    /// The save could not be read, does not parse, or was written by a newer
    /// build. Never silently discarded: ADR-0013's whole point is that player
    /// state disappearing quietly is the failure worth engineering against.
    Save(save::SaveError),
}

/// The content revision this build of the slice authors against.
pub(crate) const SLICE_REVISION: &str = "cinderwake-slice";

/// The save file within a save directory.
///
/// A constant rather than a literal at each site: two spellings of this name
/// would not fail, they would quietly write one file and read another.
pub(crate) const SAVE_FILE: &str = "world.ron";

/// The append-only domain event log beside it (ADR-0013).
pub(crate) const EVENT_LOG: &str = "events.log";

/// The class the slice starts you as.
///
/// ADR-0007 ships one tuned class and one defined-but-unpolished. No milestone
/// yet schedules choosing at character creation, so the tuned one is chosen
/// for you.
pub(crate) const SLICE_CLASS: &str = "wayfinder";

impl std::fmt::Display for SliceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Content(report) => write!(f, "content is invalid: {report}"),
            Self::UnknownLocation(location) => write!(f, "no such location: {location}"),
            Self::UnknownClass(class) => write!(f, "no such class: {class}"),
            Self::Domain(error) => write!(f, "cannot apply event: {error}"),
            Self::Revision(error) => write!(f, "slice revision id is invalid: {error}"),
            Self::Save(error) => write!(f, "cannot load the save: {error}"),
        }
    }
}

impl std::error::Error for SliceError {}

/// Everything the slice needs that is not an entity.
///
/// A world resource, so systems reach it the same way they reach anything else.
pub(crate) struct Slice {
    /// Compiled, cross-checked content.
    pub content: ContentRegistry,
    /// Canonical domain state: the authority on durable facts.
    pub state: WorldState,
    /// Where the player currently is.
    pub location: LocationId,
    /// The navmesh of the current location, rebuilt on every transition.
    pub navmesh: NavMesh,
    /// Events produced this tick, awaiting [`commit`](Self::commit).
    pub pending: Vec<DomainEvent>,
    /// Where saves and the event log are written.
    pub save_dir: Option<PathBuf>,
    /// Gameplay signals raised this tick, which quests consume.
    pub signals: Vec<Signal>,
    /// Who is currently following the player, and where they are being taken.
    ///
    /// Runtime state, like aggro: a transition despawns every NPC and rebuilds
    /// the location from content, so "who is in tow" cannot live on an entity
    /// that is about to stop existing. [`travel_to`] re-spawns the follower on
    /// the far side.
    ///
    /// Not serialized, so a save taken mid-escort reloads with nobody
    /// following. The *objective* is durable, so this could be re-derived on
    /// load; until it is, reloading means walking back to fetch them.
    pub escorting: Option<(NpcId, LocationId)>,
    /// What the player is currently being asked to choose from, if anything.
    ///
    /// Runtime state, like [`escorting`](Self::escorting), and not serialized:
    /// a save taken with a menu open reloads with it closed, because everything
    /// a menu can do becomes a domain event the moment it is done. Persisting a
    /// half-made choice would be persisting something that never happened.
    pub offer: Option<Offer>,
    /// Whether a fight was in progress as of the last tick.
    ///
    /// Runtime state, and not serialized: it exists only to notice the *edge*
    /// where a fight ends, which is a safe point for health
    /// ([`sync_on_combat_exit`]). A save taken mid-fight reloads out of combat,
    /// which is correct — the fight is not durable, only its cost is.
    pub in_combat: bool,
    /// What to tell the player that has not reached the screen yet (M9): text
    /// derived from events [`commit`](Self::commit) applied, and the odd
    /// refusal ("has nothing to say"). An outbox — the HUD empties it each
    /// tick — so the slice never has to know how long a notice lasts.
    pub notices: Vec<String>,
    /// A consumable the player picked from the inventory to use (M9). Using
    /// takes a moment in the world, which the menu cannot start itself, so
    /// the pick waits here for `slice-use`.
    pub use_requested: Option<ItemId>,
    /// Whether the player picked a class this tick (M9), so the body that
    /// carries the old one's pools, loadout, and stance must be rebuilt once
    /// the choice commits.
    pub class_changed: bool,
    /// Where content was loaded from, so a development run can reload it
    /// (M9). `None` for a slice built from content already in memory.
    pub content_root: Option<PathBuf>,
    /// F5 was pressed in a development run and the reload waits for a safe
    /// point: no fight, no panel open.
    pub reload_requested: bool,
}

impl Resource for Slice {}

impl Slice {
    /// Loads content and builds a fresh world.
    pub(crate) fn new(content_root: &Path, start: &LocationId, seed: u64) -> Result<Self, SliceError> {
        let content =
            content_runtime::load(content_root).map_err(|report| SliceError::Content(Box::new(report)))?;
        if !content.locations.contains_key(start) {
            return Err(SliceError::UnknownLocation(start.clone()));
        }

        // The world records which content it was built against, so a save can
        // be checked against the content it is replayed on (ADR-0013).
        let revision =
            game_core::ids::ContentRevisionId::new(SLICE_REVISION).map_err(SliceError::Revision)?;
        let mut state = WorldState::new(seed, revision);

        // The class is what gives the player pools at all: `PlayerState`
        // derives `Default`, so an unclassed world records health and stamina
        // of 0/0 (ADR-0015). Applied as an event rather than assigned, so the
        // character a save describes is one a replay can reproduce.
        let class = content
            .classes
            .get(&ClassId::new(SLICE_CLASS).map_err(SliceError::Revision)?)
            .ok_or_else(|| SliceError::UnknownClass(SLICE_CLASS.to_owned()))?;
        apply(
            &mut state,
            &DomainEvent::ClassChosen {
                class: class.id.clone(),
                health: class.health_pool(),
                stamina: class.stamina_pool(),
            },
        )
        .map_err(SliceError::Domain)?;
        // A new character is level 1 (ADR-0018); experience raises it.
        apply(&mut state, &DomainEvent::LeveledUp { level: 1 }).map_err(SliceError::Domain)?;

        // Where the player starts is durable, and used not to be recorded at
        // all: `enter` emits `PlayerMoved` only when the location *changes*, so
        // a world that had never used a door carried `player.location: None`
        // and a save that could not say where to put anyone back. The same
        // argument as ADR-0015's pools — the runtime knew something durable
        // that the record did not.
        apply(
            &mut state,
            &DomainEvent::PlayerMoved {
                location: start.clone(),
            },
        )
        .map_err(SliceError::Domain)?;

        // Every NPC the slice's dialogue can address must exist in canonical
        // state, or talking to them would fail at the reducer.
        for dialogue in content.dialogues.values() {
            state.npcs.entry(dialogue.speaker.clone()).or_default();
        }

        let navmesh = content
            .locations
            .get(start)
            .map(NavMesh::from_location)
            .unwrap_or_default();

        Ok(Self {
            content,
            state,
            location: start.clone(),
            navmesh,
            pending: Vec::new(),
            save_dir: None,
            signals: Vec::new(),
            escorting: None,
            offer: None,
            notices: Vec::new(),
            use_requested: None,
            class_changed: false,
            content_root: Some(content_root.to_path_buf()),
            reload_requested: false,
            in_combat: false,
        })
    }

    /// Restores a world from the save in `save_dir`.
    ///
    /// The sibling of [`new`](Self::new), and deliberately not a variant of it:
    /// the two disagree about almost everything that matters. A new world is
    /// seeded and bootstrapped; a resumed one adopts what was recorded and must
    /// not re-bootstrap anything.
    ///
    /// # Loaded from the snapshot, not replayed from the log
    ///
    /// ADR-0013's replay contract — state equals the seeded initial state
    /// folded over its accepted events — is a property *asserted by test*, not
    /// the mechanism for loading. The envelope is the authority; the log exists
    /// to verify it and to debug divergence. Reconstructing state by replay
    /// would be slower, would fail differently, and would quietly make the log
    /// load-bearing.
    pub(crate) fn resume(content_root: &Path, save_dir: &Path) -> Result<Self, SliceError> {
        let content =
            content_runtime::load(content_root).map_err(|report| SliceError::Content(Box::new(report)))?;

        let path = save_dir.join(SAVE_FILE);
        let envelope = save::load(&path).map_err(SliceError::Save)?;

        // A hash mismatch is a warning rather than a refusal. Content changes
        // constantly while the slice is being authored, and refusing would make
        // saves useless within a day; what must not be silent is a world naming
        // content that no longer exists, which the reference check below is for.
        match content.content_hash() {
            Ok(hash) if hash != envelope.content_hash => engine::log_warn!(
                "slice",
                "save was written against different content; loading anyway",
                revision = envelope.content_revision
            ),
            Ok(_) => {}
            Err(error) => {
                engine::log_warn!(
                    "slice",
                    "cannot hash content to compare with the save",
                    error = error
                )
            }
        }

        let report = content_runtime::verify_world_references(&content, &envelope.world, &path);
        if !report.is_empty() {
            return Err(SliceError::Content(Box::new(report)));
        }

        let mut state = envelope.world;

        // No `ClassChosen` here, deliberately. The saved world already carries
        // the pools it was made with, and ADR-0015's point is that a save
        // reproduces the character who actually existed even if the class was
        // retuned since. Re-applying would overwrite the record with today's
        // content — the precise failure the ADR exists to prevent.

        // Idempotent, and not merely defensive: content may have gained a
        // dialogue speaker since the save was written, and an NPC missing from
        // canonical state fails at the reducer the moment you talk to them.
        for dialogue in content.dialogues.values() {
            state.npcs.entry(dialogue.speaker.clone()).or_default();
        }

        // Where the player was, if that location still exists. Falling back to
        // the settlement degrades rather than refusing to boot: a save pointing
        // at a location content dropped is recoverable, and stranding the
        // player in the void is not.
        let location = state
            .player
            .location
            .clone()
            .filter(|id| content.locations.contains_key(id))
            .or_else(|| settlement(&content))
            .ok_or_else(|| {
                SliceError::UnknownLocation(state.player.location.clone().unwrap_or_else(|| {
                    LocationId::new("unknown").unwrap_or_else(|_| unreachable!("literal id is valid"))
                }))
            })?;

        let navmesh = content
            .locations
            .get(&location)
            .map(NavMesh::from_location)
            .unwrap_or_default();

        Ok(Self {
            content,
            state,
            location,
            navmesh,
            pending: Vec::new(),
            save_dir: Some(save_dir.to_path_buf()),
            signals: Vec::new(),
            // Nobody follows a freshly loaded world. An open `Escort` objective
            // is *not* the same fact as an escort in progress — the objective
            // opens when the branch does, before the player has asked anyone to
            // come along — so re-deriving from it would have an NPC appear at
            // your side having never been asked. The objective stays open, so
            // nothing is lost; re-asking costs a walk.
            escorting: None,
            offer: None,
            notices: Vec::new(),
            use_requested: None,
            class_changed: false,
            content_root: Some(content_root.to_path_buf()),
            reload_requested: false,
            in_combat: false,
        })
    }

    /// The location the player is in.
    pub(crate) fn current_location(&self) -> Option<&LocationDef> {
        self.content.locations.get(&self.location)
    }

    /// Queues an event for this tick.
    ///
    /// Queued rather than applied immediately so that a tick's outcomes are
    /// committed together, in order, and land in the log as one group.
    pub(crate) fn raise(&mut self, event: DomainEvent) {
        self.pending.push(event);
    }

    /// Queues a gameplay signal for quests to consume.
    pub(crate) fn signal(&mut self, signal: Signal) {
        self.signals.push(signal);
    }

    /// Folds queued events into canonical state and appends them to the log.
    ///
    /// Returns the events that were applied. An event that the reducer rejects
    /// stops the batch: the log up to that point is still history, and a
    /// rejected event is a bug worth surfacing rather than skipping.
    pub(crate) fn commit(&mut self) -> Result<Vec<DomainEvent>, SliceError> {
        let events = std::mem::take(&mut self.pending);
        for event in &events {
            apply(&mut self.state, event).map_err(SliceError::Domain)?;
            // Experience that crosses a threshold is a level (ADR-0018),
            // recorded as its own event so a retuned curve never demotes.
            if matches!(event, DomainEvent::ExperienceGained { .. })
                && let Some(level) = self.level_reached()
            {
                self.pending.push(DomainEvent::LeveledUp { level });
            }
            // Worded after applying, so a count reads as it now stands.
            if let Some(notice) = crate::hud::notice_for(event, &self.content, &self.state) {
                self.notices.push(notice);
            }
        }

        if let Some(dir) = &self.save_dir
            && !events.is_empty()
            && let Err(error) = log::append_events(&dir.join(EVENT_LOG), &events)
        {
            // A log that cannot be written must not take the game down: the
            // player keeps playing, and the failure is visible in diagnostics.
            engine::log_error!("slice", "cannot append to the event log", error = error);
        }
        Ok(events)
    }

    /// The level the player's experience has reached on their class's curve,
    /// if it is higher than the level recorded.
    pub(crate) fn level_reached(&self) -> Option<u16> {
        let progression = &self.state.player.progression;
        let curve = progression
            .class
            .as_ref()
            .and_then(|class| self.content.classes.get(class))
            .and_then(|class| class.level_curve.as_ref())
            .and_then(|curve| self.content.level_curves.get(curve))?;
        let level = curve.level_at(progression.experience);
        (level > progression.level).then_some(level)
    }

    /// Opens what `npc` has to say, if anything.
    ///
    /// Of the NPC's conversations whose conditions hold (M9): one opens
    /// directly, several open a list of topics to pick from, and none opens
    /// nothing. Opening is all this does. The player picks a line with a digit
    /// key and [`take_offer`](Self::take_offer) turns that into events — the
    /// runtime no longer speaks for the player, which is the point, but it
    /// also means a caller must not assume a conversation has *finished*
    /// merely because it started.
    ///
    /// Returns whether anything opened.
    pub(crate) fn open_conversation(&mut self, npc: &NpcId) -> bool {
        let open = narrative::open_conversations(self.content.dialogues.values(), npc, &self.state);
        self.offer = match open.as_slice() {
            [] => return false,
            [only] => Some(Offer::Talking(narrative::Conversation::open(only))),
            _ => Some(Offer::Topics(npc.clone())),
        };
        self.announce_offer();
        true
    }

    /// Where a station leads, if it is a doorway.
    pub(crate) fn link_from(&self, spawn: &SpawnId) -> Option<(LocationId, SpawnId)> {
        let location = self.current_location()?;
        location
            .links
            .iter()
            .find(|link| &link.from == spawn)
            .map(|link| (link.to.clone(), link.arrive_at.clone()))
    }

    /// Moves the player to another location.
    ///
    /// Arriving somewhere is durable — a save must remember where you were — so
    /// it is recorded as events rather than only as a changed field.
    pub(crate) fn enter(&mut self, destination: &LocationId) {
        let Some(def) = self.content.locations.get(destination) else {
            return;
        };
        self.navmesh = NavMesh::from_location(def);
        let region = def.region.clone();
        let arrived_somewhere_new = &self.location != destination;
        self.location = destination.clone();

        // Only when the location actually changed. Dying in the settlement
        // re-enters the settlement, and logging a move that moved nobody would
        // put noise in the record of where the player has been.
        if arrived_somewhere_new {
            self.pending.push(DomainEvent::PlayerMoved {
                location: destination.clone(),
            });
        }
        // Discovery happens once. Logging it on every visit would bury the
        // moment it actually mattered under repetitions of itself.
        if !self.state.region(&region).is_some_and(|state| state.discovered) {
            self.pending.push(DomainEvent::RegionStateChanged {
                region,
                change: RegionChange::Discovered,
            });
        }
        self.signals.push(Signal::Reached {
            location: destination.clone(),
        });
    }

    /// Opens the forge's list of recipes.
    ///
    /// Which recipes are on it is `selection::rows`' business, and whether one
    /// can be afforded stays `interaction::craft`'s rule. This only says which
    /// list is open.
    pub(crate) fn open_forge(&mut self) {
        self.offer = Some(Offer::Forge);
        self.announce_offer();
    }

    /// Opens the settlement board's list of blueprints.
    pub(crate) fn open_board(&mut self) {
        self.offer = Some(Offer::Board);
        self.announce_offer();
    }

    /// Opens what the player is carrying that can be worn or wielded.
    ///
    /// Unlike the two stations this has no fixture in the world: it opens on a
    /// key, anywhere. Equipment is not a place you walk to, and authoring an
    /// armoury to stand in would be inventing a location to justify a menu.
    ///
    /// Refuses while something else is open, so pressing it mid-conversation
    /// cannot silently discard a dialogue cursor — walking away is the only way
    /// to leave a conversation, and it should stay that way.
    pub(crate) fn open_equipment(&mut self) -> bool {
        if self.offer.is_some() {
            return false;
        }
        self.offer = Some(Offer::Equipment);
        self.announce_offer();
        true
    }

    /// Opens something to read (M9): an examined clue's text. Refuses while
    /// anything else is open, like every panel that is not a conversation.
    pub(crate) fn open_note(&mut self, title: String, text: String) -> bool {
        self.open_panel(Offer::Note { title, text })
    }

    /// Asks the player to confirm a decision (M9).
    pub(crate) fn open_decision(&mut self, choice: ChoiceId, prompt: String) -> bool {
        self.open_panel(Offer::Decision {
            choice,
            prompt,
            settled: false,
        })
    }

    /// Toggles a read-only panel — the journal or the controls (M9): opens it
    /// when nothing is open, closes it when it is the one open, and leaves
    /// anything else alone.
    pub(crate) fn toggle_panel(&mut self, panel: Offer) -> bool {
        if self.offer.as_ref() == Some(&panel) {
            return self.close_offer();
        }
        self.open_panel(panel)
    }

    fn open_panel(&mut self, panel: Offer) -> bool {
        if self.offer.is_some() {
            return false;
        }
        self.offer = Some(panel);
        self.announce_offer();
        true
    }

    /// Logs what is on offer, which is the whole of the interface for now.
    ///
    /// There is no text rendering yet — two shaders, both geometry — so the
    /// offered lines go to diagnostics rather than to the screen.
    /// `selection::lines` is the same function a drawn menu will use, so what a
    /// player eventually reads and what a log records cannot drift apart.
    fn announce_offer(&self) {
        let Some(offer) = &self.offer else {
            return;
        };
        let rows = selection::rows(offer, &self.content, &self.state);
        engine::log_info!(
            "slice",
            "offering",
            prompt = selection::prompt(offer, &self.content).unwrap_or_default(),
            options = selection::lines(&rows).join(" "),
        );
    }

    /// Takes the nth offered row, counting from 1.
    ///
    /// Returns whether anything was taken. No offer open, a digit past the end
    /// of the list, and a row that has gone stale are all simply "nothing
    /// happened": pressing a key that means nothing must not cost the player
    /// anything.
    pub(crate) fn take_offer(&mut self, digit: u32) -> bool {
        let Some(offer) = self.offer.as_mut() else {
            return false;
        };
        let Some(pick) = selection::pick_at(offer, &self.content, &self.state, digit) else {
            return false;
        };
        let outcome = match selection::take(offer, &self.content, &self.state, &pick) {
            Ok(outcome) => outcome,
            Err(error) => {
                engine::log_warn!("slice", "a choice was refused", error = error);
                return false;
            }
        };
        let closed = selection::is_over(offer);

        if outcome
            .events
            .iter()
            .any(|event| matches!(event, DomainEvent::ClassChosen { .. }))
        {
            self.class_changed = true;
        }
        self.pending.extend(outcome.events);
        self.signals.extend(
            outcome
                .choices
                .into_iter()
                .map(|choice| Signal::Decided { choice }),
        );
        // Crafting feeds a `Collect` objective like any other way of acquiring
        // something.
        if let Some(item) = outcome.crafted {
            self.signals.push(Signal::Collected { item, quantity: 1 });
        }
        if let Some(item) = outcome.used {
            self.use_requested = Some(item);
        }

        if closed {
            self.end_offer();
        } else {
            self.announce_offer();
        }
        true
    }

    /// The NPC an open conversation is with, if the offer is one.
    fn speaking_with(&self) -> Option<NpcId> {
        let Some(Offer::Talking(conversation)) = &self.offer else {
            return None;
        };
        self.content
            .dialogues
            .get(conversation.dialogue())
            .map(|def| def.speaker.clone())
    }

    /// Closes the offer, raising the signal that speaking to someone produces.
    ///
    /// `Talked` is raised *here*, when a conversation ends, rather than when it
    /// opens. Opening put the signal a whole tick ahead of the `QuestAccepted`
    /// that the conversation itself produces, so `advance_quests` drained it
    /// against a quest still `NotStarted` and the player had to speak to the
    /// same NPC twice to be credited once — the wave 4 ordering bug, returning
    /// by a different door now that opening and choosing are separate ticks.
    ///
    /// It is also the honest moment: "the player spoke with X" becomes true
    /// when the exchange is over, however it ended.
    fn end_offer(&mut self) {
        if let Some(npc) = self.speaking_with() {
            self.signals.push(Signal::Talked { npc });
        }
        self.offer = None;
    }

    /// Closes whatever is open, reporting whether anything was.
    ///
    /// Walking away is always allowed — a node whose every line is gated off,
    /// or a forge with nothing affordable, must not be a trap.
    pub(crate) fn close_offer(&mut self) -> bool {
        if self.offer.is_none() {
            return false;
        }
        self.end_offer();
        true
    }

    /// Starts an escort if a quest is currently asking for this NPC to be led
    /// somewhere.
    ///
    /// Inferred from the open objective rather than from a dialogue effect:
    /// there is no `StartEscort`, and adding one would put the same fact in two
    /// places — the quest already says who is to be taken where.
    ///
    /// Returns whether an escort began.
    pub(crate) fn begin_escort(&mut self, npc: &NpcId) -> bool {
        let wanted = self.content.quests.values().find_map(|quest| {
            narrative::quest::open_objectives(quest, &self.state)
                .iter()
                .find_map(|open| match &quest.nodes.get(open)?.objective {
                    game_core::content::Objective::Escort {
                        npc: subject,
                        destination,
                    } if subject == npc => Some(destination.clone()),
                    _ => None,
                })
        });

        let Some(destination) = wanted else {
            return false;
        };
        self.escorting = Some((npc.clone(), destination));
        true
    }

    /// Advances quests with everything that happened this tick.
    ///
    /// Signals are drained whether or not any quest cares, so a signal cannot
    /// be counted twice by a quest accepted later in the same run.
    pub(crate) fn advance_quests(&mut self) {
        let signals = std::mem::take(&mut self.signals);
        if signals.is_empty() {
            return;
        }

        // Quests are visited in id order so two runs of the same tick produce
        // the same events in the same sequence.
        for quest in self.content.quests.values() {
            for signal in &signals {
                let events = narrative::quest::advance(quest, &self.state, signal);
                self.pending.extend(events);
            }
        }
    }

    /// Writes a save alongside the event log.
    pub(crate) fn write_save(&self) -> Result<(), SliceError> {
        let Some(dir) = &self.save_dir else {
            return Ok(());
        };
        // No hash, no save. A save stamped with a placeholder hash would claim
        // provenance it does not have, and the point of recording one is being
        // able to tell later whether the content still matches.
        let hash = match self.content.content_hash() {
            Ok(hash) => hash,
            Err(error) => {
                engine::log_error!("slice", "cannot hash content; save skipped", error = error);
                return Ok(());
            }
        };
        let envelope = SaveEnvelope::new(self.state.clone(), env!("CARGO_PKG_VERSION"), hash);
        if let Err(error) = save::save(&dir.join(SAVE_FILE), &envelope) {
            engine::log_error!("slice", "cannot write save", error = error);
        }
        Ok(())
    }
}

/// Spawns everything an authored location describes.
///
/// Returns the player's start position. Called on entry to a location and again
/// after a transition — the runtime is rebuilt from content and canonical state
/// every time, never carried over.
pub(crate) fn populate(world: &mut World, slice: &Slice, arrive_at: Option<&SpawnId>) -> Vec3 {
    let Some(location) = slice.current_location() else {
        return Vec3::ZERO;
    };
    world.insert_resource(Waves::of(location));

    let mut start = Vec3::ZERO;
    let capsule = Capsule::default();
    let visuals = world.resource::<SliceVisuals>().copied();

    // A floor, so the scene reads as a place rather than objects in void. Part
    // of the location's entities, so a transition replaces it like everything
    // else.
    if let Some(visuals) = visuals {
        let ground = world.spawn();
        let _ = world.insert(ground, Transform::IDENTITY);
        let _ = world.insert(ground, Scenery);
        dress(world, ground, &visuals, visuals.ground);
    }

    for spawn in &location.spawns {
        let at = to_world(spawn.placement.at);
        // Arriving through a door lands on that door's `arrive_at` spawn, not on
        // the location's player start — otherwise walking back into the hub
        // would deposit you at the front gate every time.
        if arrive_at.is_some_and(|id| id == &spawn.id) {
            start = Vec3::new(at.x, capsule.half_extent(), at.z);
        }
        match &spawn.role {
            SpawnRole::PlayerStart if arrive_at.is_none() => {
                start = Vec3::new(at.x, capsule.half_extent(), at.z);
            }
            SpawnRole::PlayerStart => {}
            SpawnRole::Npc { npc, model } => {
                let entity = world.spawn();
                let _ = world.insert(entity, Transform::from_xyz(at.x, capsule.half_extent(), at.z));
                let _ = world.insert(entity, Npc(npc.clone()));
                if let Some(visuals) = visuals {
                    dress_as(world, entity, &visuals, model.as_ref(), visuals.npc);
                }
            }
            // Only the first wave is there on arrival; later waves wait in
            // `Waves` until the ones before them are down (M8).
            SpawnRole::Enemy {
                prefab,
                encounter,
                wave: 0,
            } => {
                let stand = Vec3::new(at.x, capsule.half_extent(), at.z);
                spawn_enemy_at(world, slice, prefab, spawn.id.clone(), encounter.clone(), stand);
            }
            SpawnRole::Enemy { .. } => {}
            SpawnRole::Gatherable { item } => {
                let entity = world.spawn();
                let _ = world.insert(entity, Transform::from_xyz(at.x, at.y, at.z));
                let _ = world.insert(
                    entity,
                    Gatherable {
                        item: item.clone(),
                        spawn: spawn.id.clone(),
                        taken: false,
                    },
                );
                if let Some(visuals) = visuals {
                    dress(world, entity, &visuals, visuals.gatherable);
                }
            }
            SpawnRole::Clue { clue, text } => {
                let entity = world.spawn();
                let _ = world.insert(entity, Transform::from_xyz(at.x, at.y, at.z));
                let _ = world.insert(
                    entity,
                    Clue {
                        clue: clue.clone(),
                        spawn: spawn.id.clone(),
                        text: text.clone(),
                        examined: false,
                    },
                );
                if let Some(visuals) = visuals {
                    dress(world, entity, &visuals, visuals.clue);
                }
            }
            SpawnRole::Choice { choice, prompt } => {
                let entity = world.spawn();
                let _ = world.insert(entity, Transform::from_xyz(at.x, at.y, at.z));
                let _ = world.insert(
                    entity,
                    ChoicePoint {
                        choice: choice.clone(),
                        spawn: spawn.id.clone(),
                        prompt: prompt.clone(),
                        settled: false,
                    },
                );
                if let Some(visuals) = visuals {
                    dress(world, entity, &visuals, visuals.choice);
                }
            }
            SpawnRole::Station { kind } => {
                let leads_to = location
                    .links
                    .iter()
                    .find(|link| link.from == spawn.id)
                    .map(|link| link.to.clone());
                let entity = world.spawn();
                let _ = world.insert(entity, Transform::from_xyz(at.x, at.y, at.z));
                let _ = world.insert(
                    entity,
                    Station {
                        spawn: spawn.id.clone(),
                        kind: *kind,
                        leads_to,
                    },
                );
                if let Some(visuals) = visuals {
                    let look = match kind {
                        StationKind::Doorway => visuals.doorway,
                        StationKind::Crafting => visuals.crafting,
                        StationKind::Upgrades => visuals.upgrades,
                    };
                    dress(world, entity, &visuals, look);
                }
            }
        }
    }

    start
}

/// Builds the static collision for a location.
///
/// The authored spaces are open ground, and their navmeshes already bound where
/// anything may walk — pathing keeps enemies inside it, and the player is
/// clamped to it separately. So this is the ground plane and nothing else.
/// Prop collision arrives when props do; inventing walls here that no authored
/// geometry corresponds to would make the collision world disagree with what is
/// on screen.
pub(crate) fn collision_for(_location: &LocationDef) -> CollisionWorld {
    CollisionWorld::new(0.0)
}

/// The meshes and materials the slice draws itself with.
///
/// Built once, at startup, because that is the only place a `RenderAssets`
/// handle exists — and because [`populate`] runs again on every location
/// change, so uploading there would leak a fresh mesh and material per
/// transition for as long as the session lasted.
///
/// Blocked-out on purpose: coloured primitives, one per kind of thing, so the
/// slice is *legible* rather than pretty. Real models are M7's (ADR-0017), and
/// these stay as the fallback for content that names none; being able to see
/// where the husks are is today's.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SliceVisuals {
    pub ground: (MeshHandle, MaterialHandle),
    pub player: (MeshHandle, MaterialHandle),
    pub enemy: (MeshHandle, MaterialHandle),
    pub elite: (MeshHandle, MaterialHandle),
    pub npc: (MeshHandle, MaterialHandle),
    pub doorway: (MeshHandle, MaterialHandle),
    pub crafting: (MeshHandle, MaterialHandle),
    pub upgrades: (MeshHandle, MaterialHandle),
    pub gatherable: (MeshHandle, MaterialHandle),
    pub clue: (MeshHandle, MaterialHandle),
    pub choice: (MeshHandle, MaterialHandle),
    /// What a body borrows while something is happening to it.
    pub cues: Cues,
    /// The material every modelled part draws with: a white tint, so the
    /// colour comes entirely from the recipe's painted vertices.
    pub model_base: MaterialHandle,
    /// Kept so bounds can be attached without re-deriving them per spawn.
    bounds: SliceBounds,
}

impl Resource for SliceVisuals {}

/// The materials a body borrows while something is happening to it.
///
/// Created once beside the base palette, for the same reason it is: `populate`
/// reruns on every location change. Swapping a handle is the whole mechanism —
/// no shader, no per-entity material, nothing the renderer has to know about —
/// which is what keeps combat feedback a gameplay change rather than a
/// renderer one.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Cues {
    /// Winding up: committed, and about to act. The dodge cue.
    pub windup: MaterialHandle,
    /// The hit volume is live this tick.
    pub strike: MaterialHandle,
    /// Just took damage.
    pub hurt: MaterialHandle,
    /// Mid-roll, inside the invulnerable window.
    pub evading: MaterialHandle,
}

/// Bounds for each mesh, derived once alongside the upload.
#[derive(Debug, Clone, Copy)]
struct SliceBounds {
    ground: Option<RenderBounds>,
    body: Option<RenderBounds>,
    elite: Option<RenderBounds>,
    post: Option<RenderBounds>,
    marker: Option<RenderBounds>,
}

impl SliceVisuals {
    /// Uploads the slice's placeholder geometry.
    ///
    /// Colours are linear RGBA — the surface is sRGB and the GPU converts on
    /// write — and deliberately far apart in hue, because the whole point is
    /// telling a husk from a warden at a glance.
    pub(crate) fn build(assets: &mut dyn RenderAssets) -> Option<Self> {
        let ground_mesh = primitives::plane(48.0, 8);
        let body_mesh = primitives::cube(0.8);
        let elite_mesh = primitives::cube(1.3);
        let post_mesh = primitives::cylinder(0.4, 1.8, 10);
        let marker_mesh = primitives::cone(0.45, 1.0, 8);

        // A failed upload means a prop nobody can see, which is worth reporting
        // rather than silently drawing an empty world.
        fn upload(assets: &mut dyn RenderAssets, mesh: &MeshData, name: &'static str) -> Option<MeshHandle> {
            match assets.upload_mesh(mesh) {
                Ok(handle) => Some(handle),
                Err(error) => {
                    engine::log_error!("slice", "mesh upload failed", mesh = name, error = error);
                    None
                }
            }
        }

        let ground = upload(assets, &ground_mesh, "ground")?;
        let body = upload(assets, &body_mesh, "body")?;
        let elite_shape = upload(assets, &elite_mesh, "elite")?;
        let post = upload(assets, &post_mesh, "post")?;
        let marker = upload(assets, &marker_mesh, "marker")?;

        // Cues and models first: the tint closure below borrows `assets` for
        // the rest of the function.
        let model_base = assets.create_material(&MaterialDesc::default());
        // Cues are flash overlays over a white tint rather than flat colours
        // (ADR-0017): a flat-coloured cue multiplied into a painted body would
        // change nothing, and the flash would vanish on the first real model.
        let mut cue = |r: f32, g: f32, b: f32, amount: f32| {
            assets.create_material(&MaterialDesc {
                flash: [r, g, b, amount],
                ..MaterialDesc::default()
            })
        };
        // Chosen against the palette below: yellow and hot red read as
        // "coming" and "now" against every body colour, white is the one thing
        // nothing is painted, and the roll's pale cyan is far from both the
        // player's orange and the NPC's blue.
        let cues = Cues {
            windup: cue(0.95, 0.80, 0.10, 0.55),
            strike: cue(1.00, 0.05, 0.02, 0.70),
            hurt: cue(1.00, 1.00, 1.00, 0.85),
            evading: cue(0.45, 0.85, 0.95, 0.50),
        };

        let mut material = |r: f32, g: f32, b: f32| {
            assets.create_material(&MaterialDesc {
                base_color: [r, g, b, 1.0],
                ..MaterialDesc::default()
            })
        };

        Some(Self {
            ground: (ground, material(0.14, 0.17, 0.13)),
            player: (body, material(0.85, 0.45, 0.16)),
            enemy: (body, material(0.55, 0.13, 0.15)),
            elite: (elite_shape, material(0.72, 0.10, 0.30)),
            npc: (body, material(0.20, 0.42, 0.68)),
            doorway: (post, material(0.62, 0.60, 0.55)),
            crafting: (post, material(0.70, 0.38, 0.12)),
            upgrades: (post, material(0.40, 0.46, 0.30)),
            gatherable: (marker, material(0.30, 0.62, 0.34)),
            clue: (marker, material(0.78, 0.74, 0.44)),
            choice: (marker, material(0.56, 0.36, 0.70)),
            cues,
            model_base,
            bounds: SliceBounds {
                ground: RenderBounds::from_mesh_data(&ground_mesh),
                body: RenderBounds::from_mesh_data(&body_mesh),
                elite: RenderBounds::from_mesh_data(&elite_mesh),
                post: RenderBounds::from_mesh_data(&post_mesh),
                marker: RenderBounds::from_mesh_data(&marker_mesh),
            },
        })
    }

    fn bounds_for(&self, mesh: MeshHandle) -> Option<RenderBounds> {
        match mesh {
            m if m == self.ground.0 => self.bounds.ground,
            m if m == self.elite.0 => self.bounds.elite,
            m if m == self.doorway.0 => self.bounds.post,
            m if m == self.gatherable.0 => self.bounds.marker,
            _ => self.bounds.body,
        }
    }
}

/// Attaches the render components for one look.
///
/// Bounds are optional to the extractor — `passes_culling` takes an
/// `Option<&RenderBounds>` — so a missing one draws rather than disappears. They
/// are attached anyway, because culling is exactly what M3 exists to exercise.
/// Dresses `entity` in its model when content names one, or in `fallback`.
///
/// The fallback is permanent rather than transitional: content without a
/// model still plays, drawn as the block-out shape it always was. A model that
/// is named but was not built — its mesh failed to upload — falls back too,
/// with a warning, because a missing look must never take a body out of play.
pub(crate) fn dress_as(
    world: &mut World,
    entity: Entity,
    visuals: &SliceVisuals,
    model: Option<&ModelId>,
    fallback: (MeshHandle, MaterialHandle),
) {
    let built = model.and_then(|id| {
        let built = world
            .resource::<ModelLibrary>()
            .and_then(|library| library.get(id))
            .cloned();
        if built.is_none() {
            engine::log_warn!(
                "slice",
                "a named model was not built; drawing the block-out",
                model = id
            );
        }
        built
    });
    let Some(built) = built else {
        dress(world, entity, visuals, fallback);
        return;
    };
    let _ = world.insert(entity, Model { parts: built.parts });
    if let Some(id) = model {
        let _ = world.insert(entity, Wears(id.clone()));
    }
    let _ = world.insert(entity, BaseMaterial(visuals.model_base));
    let _ = world.insert(entity, Visibility::default());
    if let Some(bounds) = built.bounds {
        let _ = world.insert(entity, bounds);
    }
}

/// The model any placement dresses `npc` in, first by location id.
///
/// Needed where no placement is at hand — an escort walking through a door
/// into a location that does not author them. Deterministic because locations
/// iterate in id order.
fn npc_model<'a>(content: &'a ContentRegistry, npc: &NpcId) -> Option<&'a ModelId> {
    content.locations.values().find_map(|location| {
        location.spawns.iter().find_map(|spawn| match &spawn.role {
            SpawnRole::Npc { npc: placed, model } if placed == npc => model.as_ref(),
            _ => None,
        })
    })
}

/// A model built for drawing: one part per occupied slot.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BuiltModel {
    pub parts: Vec<ModelPart>,
    /// Which slot each part is, in the same order — what an animation clip
    /// will address (M7 wave 3).
    pub slots: Vec<SlotId>,
    /// The whole model at rest, in the entity's space: where gauges sit and
    /// what the debug overlay boxes.
    pub bounds: Option<RenderBounds>,
    /// The clips it plays: its own where it names them, its body plan's
    /// otherwise.
    pub animations: game_core::content::AnimationSet,
}

/// Which model a body wears, so the animator can find its slots and clips.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Wears(pub ModelId);

impl Component for Wears {}

/// Every model content defines, built once at startup (ADR-0017).
///
/// Built where `SliceVisuals` is, and for the same reason: `populate` reruns
/// on every location change, and uploading there would leak meshes.
#[derive(Debug, Default)]
pub(crate) struct ModelLibrary {
    models: BTreeMap<ModelId, BuiltModel>,
}

impl Resource for ModelLibrary {}

impl ModelLibrary {
    pub(crate) fn get(&self, id: &ModelId) -> Option<&BuiltModel> {
        self.models.get(id)
    }

    /// Builds every model in `content`.
    ///
    /// `stand_height` is how far a body's origin sits above the ground: model
    /// pivots are measured from the ground, so a recipe does not depend on the
    /// size of the collision body it happens to be worn over.
    pub(crate) fn build(
        content: &ContentRegistry,
        assets: &mut dyn RenderAssets,
        material: MaterialHandle,
        stand_height: f32,
    ) -> Self {
        let mut models = BTreeMap::new();
        for def in content.models.values() {
            // Validated at load, so a miss here means content changed under
            // us; skip the model rather than guess at it.
            let (Some(plan), Some(palette)) = (
                content.body_plans.get(&def.body_plan),
                content.palettes.get(&def.palette),
            ) else {
                continue;
            };
            let scale = metres(def.scale);
            let paint = |swatch: &game_core::ids::SwatchId| {
                palette
                    .swatches
                    .get(swatch)
                    .map_or(renderer_white(), |&rgb| linear(rgb))
            };

            let mut built = BuiltModel {
                parts: Vec::new(),
                slots: Vec::new(),
                bounds: None,
                animations: def.animations.or(&plan.animations),
            };
            let mut whole: Option<engine::math::Aabb> = None;
            for slot in &plan.slots {
                let meshes: Vec<MeshData> = def
                    .parts
                    .iter()
                    .filter(|part| part.slot == slot.name)
                    .map(|part| {
                        let side = paint(&part.color);
                        let colors = FaceColors {
                            top: part.top.as_ref().map_or(side, &paint),
                            side,
                            bottom: part.bottom.as_ref().map_or(side, &paint),
                        };
                        // Stretch applies in the shape's own frame, before
                        // the turn, so a flattened dome stays flat along its
                        // own axis however it is tilted onto a shoulder.
                        let placement = Transform {
                            translation: to_world(part.at) * scale,
                            rotation: turn(part.turn),
                            scale: to_world(part.stretch) * scale,
                        };
                        shapes::placed(shapes::mesh(shape_of(part.shape), colors), &placement)
                    })
                    .collect();
                if meshes.is_empty() {
                    continue;
                }
                let mesh = shapes::merged(meshes);
                let bounds = RenderBounds::from_mesh_data(&mesh);
                let Ok(handle) = assets.upload_mesh(&mesh) else {
                    engine::log_warn!(
                        "slice",
                        "a model slot failed to upload",
                        model = def.id,
                        slot = slot.name
                    );
                    continue;
                };
                let rest = Transform::from_translation(to_world(slot.pivot) * scale - Vec3::Y * stand_height);
                if let Some(bounds) = bounds {
                    let placed = bounds.world_bounds(&rest.to_matrix());
                    whole = Some(match whole {
                        Some(so_far) => engine::math::Aabb {
                            min: so_far.min.min(placed.min),
                            max: so_far.max.max(placed.max),
                        },
                        None => placed,
                    });
                }
                built.parts.push(ModelPart {
                    mesh: handle,
                    material,
                    rest,
                    pose: Transform::IDENTITY,
                    bounds,
                });
                built.slots.push(slot.name.clone());
            }
            built.bounds = whole.map(RenderBounds);
            if !built.parts.is_empty() {
                models.insert(def.id.clone(), built);
            }
        }
        Self { models }
    }
}

fn renderer_white() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

/// An sRGB swatch as linear RGB, which is what the shader works in.
pub(crate) fn linear((r, g, b): (u8, u8, u8)) -> [f32; 3] {
    let channel = |value: u8| {
        let c = f32::from(value) / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    [channel(r), channel(g), channel(b)]
}

/// Degrees about X, then Y, then Z — each about the fixed axes, so a turn
/// written as one rotation reads as one rotation.
pub(crate) fn turn(degrees: Point) -> engine::math::Quat {
    let radians = |value: Fixed| metres(value).to_radians();
    engine::math::Quat::from_rotation_z(radians(degrees.z))
        * engine::math::Quat::from_rotation_y(radians(degrees.y))
        * engine::math::Quat::from_rotation_x(radians(degrees.x))
}

/// A content shape as the engine's. The content family names cylinders and
/// cones separately because that is how an author thinks of them; the engine
/// builds both as frustums.
pub(crate) fn shape_of(shape: ShapeDef) -> Shape {
    let m = metres;
    match shape {
        ShapeDef::Box { x, y, z } => Shape::Box {
            size: Vec3::new(m(x), m(y), m(z)),
        },
        ShapeDef::Wedge { x, y, z } => Shape::Wedge {
            size: Vec3::new(m(x), m(y), m(z)),
        },
        ShapeDef::Cylinder {
            radius,
            height,
            sides,
        } => Shape::Frustum {
            bottom_radius: m(radius),
            top_radius: m(radius),
            height: m(height),
            sides: u32::from(sides),
        },
        ShapeDef::Cone {
            radius,
            height,
            sides,
        } => Shape::Frustum {
            bottom_radius: m(radius),
            top_radius: 0.0,
            height: m(height),
            sides: u32::from(sides),
        },
        ShapeDef::Frustum {
            bottom,
            top,
            height,
            sides,
        } => Shape::Frustum {
            bottom_radius: m(bottom),
            top_radius: m(top),
            height: m(height),
            sides: u32::from(sides),
        },
        ShapeDef::Sphere { radius, detail } => Shape::Sphere {
            radius: m(radius),
            detail: u32::from(detail),
        },
        ShapeDef::Taper {
            x,
            z,
            top_x,
            top_z,
            height,
        } => Shape::Taper {
            bottom: Vec2::new(m(x), m(z)),
            top: Vec2::new(m(top_x), m(top_z)),
            height: m(height),
        },
        ShapeDef::Dome {
            radius,
            segments,
            rings,
        } => Shape::Dome {
            radius: m(radius),
            segments: u32::from(segments),
            rings: u32::from(rings),
        },
    }
}

fn dress(world: &mut World, entity: Entity, visuals: &SliceVisuals, look: (MeshHandle, MaterialHandle)) {
    let (mesh, material) = look;
    let _ = world.insert(entity, MeshRenderer::new(mesh, material));
    let _ = world.insert(entity, BaseMaterial(material));
    let _ = world.insert(entity, Visibility::default());
    if let Some(bounds) = visuals.bounds_for(mesh) {
        let _ = world.insert(entity, bounds);
    }
}

/// Rebuilds the world for `destination` and puts the player down in it.
///
/// Every way of arriving somewhere goes through here — walking through a door,
/// and dying. The runtime is derived from content and canonical state, so
/// everything the old location placed is despawned first rather than carried
/// across.
///
/// `arrive_at` names the spawn to land on; `None` uses the location's player
/// start, which is what a respawn wants.
pub(crate) fn travel_to(
    world: &mut World,
    slice: &mut Slice,
    destination: &LocationId,
    arrive_at: Option<&SpawnId>,
) {
    // A safe point: whatever the fight cost belongs to the place it happened,
    // so it is recorded before the move is.
    sync_health(world, slice);

    clear_location(world);
    slice.enter(destination);

    let collision = slice
        .current_location()
        .map_or_else(|| CollisionWorld::new(0.0), collision_for);
    world.insert_resource(collision);

    let arrival = populate(world, slice, arrive_at);
    move_player_to(world, arrival);

    // Whoever was in tow comes through the door too. `populate` rebuilt the
    // location from content, so unless the destination happens to author them
    // there, they need putting back.
    if let Some((npc, destination)) = slice.escorting.clone() {
        let already_here = world.query::<Npc>().any(|(_, present)| present.0 == npc);
        if !already_here {
            let entity = world.spawn();
            let beside = arrival + Vec3::new(1.2, 0.0, 0.0);
            let _ = world.insert(entity, Transform::from_xyz(beside.x, beside.y, beside.z));
            let _ = world.insert(entity, Npc(npc.clone()));
            let _ = world.insert(entity, Escorting);
            if let Some(visuals) = world.resource::<SliceVisuals>().copied() {
                let model = npc_model(&slice.content, &npc).cloned();
                dress_as(world, entity, &visuals, model.as_ref(), visuals.npc);
            }
        }

        if destination == slice.location {
            slice.signals.push(Signal::Escorted {
                npc,
                destination: destination.clone(),
            });
            slice.escorting = None;
        }
    }
}

/// Walks the escorted NPC after the player.
///
/// Straight-line with collision rather than navmesh pathing: the escort route
/// is open ground, and driving a friendly NPC with [`EnemyBrain`] — whose
/// motion keys off combat states like `Approach` and `Attack` — would be
/// borrowing the wrong abstraction to save a dozen lines.
pub(crate) fn follow_leader(world: &mut World, collision: &CollisionWorld) {
    let Some(player) = player_entity(world) else {
        return;
    };
    let Some(target) = world.get::<Transform>(player).map(|t| t.translation) else {
        return;
    };

    let followers: Vec<(Entity, Vec3)> = world
        .query2::<Transform, Escorting>()
        .map(|(entity, transform, _)| (entity, transform.translation))
        .collect();

    for (entity, position) in followers {
        let offset = target - position;
        let distance = offset.length();
        // Close enough is close enough: a follower that insists on an exact
        // distance jitters against the player's own movement every tick.
        if distance <= ESCORT_FOLLOW_DISTANCE {
            continue;
        }
        let Some(direction) = offset.try_normalize() else {
            continue;
        };

        let motion = Vec3::new(direction.x, 0.0, direction.z) * ESCORT_SPEED;
        let result = collision.move_capsule(Capsule::default(), position, motion);
        if let Some(transform) = world.get_mut::<Transform>(entity) {
            transform.translation = result.position;
        }
    }
}

/// Sends the player back to the settlement if they have died.
///
/// ADR-0007 keeps progress: there is no corpse to run back to and no state to
/// roll back, so dying costs the walk back rather than the run. Returning also
/// rebuilds wherever the player was, which is why the fight they died in is
/// standing again when they return — that falls out of [`travel_to`] rather
/// than being special-cased.
///
/// Emits no death event of its own. Where the player woke up is recorded by the
/// `PlayerMoved` that the transition already produces, and nothing yet gates on
/// having died.
///
/// Returns whether a respawn happened.
pub(crate) fn respawn_if_dead(world: &mut World, slice: &mut Slice) -> bool {
    let Some(player) = player_entity(world) else {
        return false;
    };
    if !world.get::<Health>(player).is_some_and(Health::is_dead) {
        return false;
    }

    // Recorded before anything is restored, so the log contains the moment
    // health reached zero. Syncing only afterwards would show a character who
    // went from full to full and moved house for no stated reason.
    sync_health(world, slice);

    let home = settlement(&slice.content);

    match home {
        Some(home) => travel_to(world, slice, &home, None),
        // Degrade rather than freeze: stand the player up where they fell
        // instead of leaving a corpse that can never act again.
        None => engine::log_error!("slice", "no settlement authored to respawn into"),
    }

    if let Some(health) = world.get_mut::<Health>(player) {
        *health = Health::full(health.max);
    }
    if let Some(stamina) = world.get_mut::<Stamina>(player) {
        *stamina = Stamina::new(stamina.max, stamina.regen_per_tick, stamina.recovery_delay);
    }
    let _ = world.remove::<Attack>(player);
    let _ = world.remove::<Dodge>(player);

    // And again, so waking up whole is recorded rather than inferred.
    sync_health(world, slice);
    true
}

/// Despawns everything a location placed, leaving the player standing.
///
/// A location change rebuilds the runtime from content and canonical state.
/// Carrying entities across would mean the crypt still contained the lane's
/// husks — and, worse, that what is on screen depended on where you had been.
pub(crate) fn clear_location(world: &mut World) -> usize {
    let mut doomed: Vec<Entity> = Vec::new();
    doomed.extend(world.query::<Npc>().map(|(entity, _)| entity));
    doomed.extend(world.query::<Enemy>().map(|(entity, _)| entity));
    doomed.extend(world.query::<Gatherable>().map(|(entity, _)| entity));
    doomed.extend(world.query::<Station>().map(|(entity, _)| entity));
    doomed.extend(world.query::<Clue>().map(|(entity, _)| entity));
    doomed.extend(world.query::<ChoicePoint>().map(|(entity, _)| entity));
    doomed.extend(world.query::<Scenery>().map(|(entity, _)| entity));
    doomed.extend(world.query::<Remains>().map(|(entity, _)| entity));
    // Effects and telegraphs belong to the fight they were drawn for, and an
    // ember still falling after the door would be falling in the wrong place.
    doomed.extend(world.query::<crate::vfx::Playing>().map(|(entity, _)| entity));
    // A bolt in the air is lost at a door, as a swing is: not carried across,
    // and not saved.
    doomed.extend(world.query::<Projectile>().map(|(entity, _)| entity));
    doomed.extend(
        world
            .query::<crate::vfx::TelegraphMark>()
            .map(|(entity, _)| entity),
    );
    doomed.extend(world.query::<crate::vfx::LaneMark>().map(|(entity, _)| entity));
    doomed.extend(world.query::<crate::vfx::Reticle>().map(|(entity, _)| entity));
    doomed.extend(
        world
            .query::<crate::nameplate::Nameplate>()
            .map(|(entity, _)| entity),
    );

    let mut removed = 0;
    for entity in doomed {
        if world.despawn(entity) {
            removed += 1;
        }
    }
    removed
}

/// The player's entity, if one has been spawned.
pub(crate) fn player_entity(world: &World) -> Option<Entity> {
    world
        .query2::<Transform, Player>()
        .next()
        .map(|(entity, _, _)| entity)
}

/// Puts the player down at `position`, ending any swing or roll.
///
/// Walking through a door mid-attack would otherwise arrive still swinging at
/// something a room away.
pub(crate) fn move_player_to(world: &mut World, position: Vec3) {
    let Some(player) = player_entity(world) else {
        return;
    };
    if let Some(transform) = world.get_mut::<Transform>(player) {
        transform.translation = position;
    }
    let _ = world.remove::<Attack>(player);
    let _ = world.remove::<Dodge>(player);
}

/// How far an ability reaches, in metres.
///
/// One number for every ability because `AbilityDef` authors no range: adding a
/// `range` field is a schema change, and M6 was scoped as executor work. Until
/// then this is the honest default rather than a per-ability guess.
pub(crate) const ABILITY_REACH: f32 = 6.0;

/// Where a summoned entity is placed relative to its summoner.
const SUMMON_SPACING: f32 = 1.2;

/// Abilities an actor may use, in authored order.
///
/// Granted from content at spawn — the class's list for the player, the
/// `EnemyDef`'s for an enemy — so *which* abilities exist is an authoring
/// decision and never a Rust one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Abilities(pub Vec<AbilityId>);

impl Component for Abilities {}

/// An ability in flight.
///
/// Deliberately **not** an [`Attack`]. A cast and a swing are different things:
/// [`step_combat`] resolves every live `Attack` against whatever is in reach, so
/// letting `ember-cleave` borrow an `Attack` to carry its commitment window
/// would deal its damage twice — once as a melee hit, once through the effect
/// executor. This carries the same three windows and nothing else.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Casting {
    pub ability: AbilityId,
    pub(crate) timing: AttackTiming,
    pub(crate) elapsed: u32,
    fired: bool,
}

impl Component for Casting {}

impl Casting {
    fn new(ability: AbilityId, timing: AttackTiming) -> Self {
        Self {
            ability,
            timing,
            elapsed: 0,
            fired: false,
        }
    }

    /// Whether this is the tick the effects land on.
    ///
    /// Once, at the end of wind-up. The executor is not idempotent — firing it
    /// on every tick of the active window would have `call-husks` summon a
    /// fresh pair each tick until the location cap stopped it.
    fn should_fire(&self) -> bool {
        !self.fired && self.elapsed >= self.timing.windup
    }

    fn is_finished(&self) -> bool {
        self.elapsed >= self.timing.total()
    }

    fn tick(&mut self) {
        if !self.is_finished() {
            self.elapsed += 1;
        }
    }

    /// Where the cast is in its windows, in the same terms as a swing, so a
    /// wind-up reads the same whether it ends in a blow or a spell.
    pub(crate) fn phase(&self) -> AttackPhase {
        let timing = self.timing;
        if self.elapsed < timing.windup {
            AttackPhase::Windup
        } else if self.elapsed < timing.windup + timing.active {
            AttackPhase::Active
        } else if self.elapsed < timing.total() {
            AttackPhase::Recovery
        } else {
            AttackPhase::Finished
        }
    }
}

/// Builds one enemy from content at `at`, and returns it.
///
/// The single place an enemy is constructed. [`populate`] uses it for authored
/// placements and [`apply_outcomes`] for `Effect::SpawnEntity`, so a summoned
/// husk is the same creature as a placed one — two constructors would be the
/// bug, in the way two definitions of "which location is home" would be.
pub(crate) fn spawn_enemy_at(
    world: &mut World,
    slice: &Slice,
    prefab: &PrefabId,
    spawn: SpawnId,
    encounter: Option<EncounterId>,
    at: Vec3,
) -> Option<Entity> {
    // Content validation resolves prefab references, so a miss is unreachable
    // with valid content — and skipping beats panicking if it ever is not.
    let def = slice.content.enemies.get(prefab)?;
    // The template's numbers, overridden by the enemy's own tuning (M8);
    // content speaks in seconds, the brain in ticks.
    let step = world
        .resource::<engine::time::FixedTime>()
        .map_or(1.0 / 60.0, |time| time.step.as_secs_f32());
    let mut brain = EnemyBrain::tuned(def.behavior, &def.tuning, step);
    brain.damage = metres(def.damage);

    let entity = world.spawn();
    let _ = world.insert(entity, Transform::from_xyz(at.x, at.y, at.z));
    let _ = world.insert(entity, brain);
    let _ = world.insert(entity, PathFollower::default());
    let _ = world.insert(entity, Health::full(metres(def.max_health)));
    let _ = world.insert(entity, Poise::new(brain.poise));
    let _ = world.insert(entity, Facing::default());
    // Enemies carry their authored loadout and their own cooldowns, so two
    // husks cool down independently (M6 wave 1).
    if !def.abilities.is_empty() {
        let _ = world.insert(entity, Abilities(def.abilities.clone()));
        let _ = world.insert(entity, Cooldowns::new());
        let _ = world.insert(entity, Stamina::default());
    }
    let elite = def.variant_of.is_some();
    let _ = world.insert(
        entity,
        Enemy {
            prefab: prefab.clone(),
            spawn,
            encounter,
        },
    );
    if let Some(visuals) = world.resource::<SliceVisuals>().copied() {
        // `variant_of` is what makes the ashlord an elite, so it is also what
        // makes it look like one.
        let look = if elite { visuals.elite } else { visuals.enemy };
        let model = def.model.clone();
        dress_as(world, entity, &visuals, model.as_ref(), look);
    }
    Some(entity)
}

/// Which of `enemy`'s abilities it uses now, if any (M8): the first in its
/// authored list that is off cooldown, affordable, and in range of a player
/// `distance` away. `None` means it swings instead.
pub(crate) fn enemy_ability(world: &World, slice: &Slice, enemy: Entity, distance: f32) -> Option<usize> {
    let known = world.get::<Abilities>(enemy)?;
    let defs: Vec<_> = known.0.iter().map(|id| slice.content.abilities.get(id)).collect();
    let cooldowns = world.get::<Cooldowns>(enemy).cloned().unwrap_or_default();
    let stamina = world.get::<Stamina>(enemy).map_or(0.0, |stamina| stamina.current);
    game_runtime::enemy::choose_ability(&defs, &cooldowns, stamina, distance, ABILITY_REACH)
}

/// A consumable being used (M9): a short committed moment, like a wind-up,
/// in which the player cannot swing, roll, or cast. The item is spent, and
/// its effects land, only when the moment is over — a stagger that cancels
/// it costs the moment, not the item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Using {
    pub item: ItemId,
    pub done_at: u64,
}

impl Component for Using {}

/// The first consumable the player carries that does something when used,
/// in id order: what X uses without opening the inventory.
pub(crate) fn quick_use(slice: &Slice) -> Option<ItemId> {
    slice
        .state
        .player
        .inventory
        .iter()
        .filter(|(_, held)| **held > 0)
        .find(|(id, _)| {
            slice
                .content
                .items
                .get(*id)
                .is_some_and(|def| def.on_use.is_some())
        })
        .map(|(id, _)| id.clone())
}

/// Starts using `item`, if the player holds one, it does something when used,
/// and the player is free to act.
pub(crate) fn start_use(world: &mut World, slice: &Slice, item: &ItemId, tick: u64) -> bool {
    let Some(player) = player_entity(world) else {
        return false;
    };
    if world.get::<Attack>(player).is_some()
        || world.get::<Dodge>(player).is_some()
        || world.get::<Casting>(player).is_some()
        || world.get::<Using>(player).is_some()
        || staggered(world, player)
    {
        return false;
    }
    if slice.state.player.inventory.get(item).copied().unwrap_or(0) == 0 {
        return false;
    }
    let Some(used) = slice.content.items.get(item).and_then(|def| def.on_use.as_ref()) else {
        return false;
    };
    let _ = world.insert(
        player,
        Using {
            item: item.clone(),
            done_at: tick + u64::from(used.use_ticks),
        },
    );
    true
}

/// Finishes a use whose moment is over: spends the item, then applies what
/// it does to the player, through the same bounded executor abilities use.
pub(crate) fn step_using(world: &mut World, slice: &mut Slice, tick: u64) {
    let Some(player) = player_entity(world) else {
        return;
    };
    let Some(using) = world.get::<Using>(player).cloned() else {
        return;
    };
    // However the stance broke, the moment is lost and the item kept.
    if staggered(world, player) {
        let _ = world.remove::<Using>(player);
        return;
    }
    if tick < using.done_at {
        return;
    }
    let _ = world.remove::<Using>(player);
    // Spent since it began — a craft, a hand-in — so there is nothing to use.
    if slice
        .state
        .player
        .inventory
        .get(&using.item)
        .copied()
        .unwrap_or(0)
        == 0
    {
        return;
    }
    let Some(used) = slice
        .content
        .items
        .get(&using.item)
        .and_then(|def| def.on_use.clone())
    else {
        return;
    };
    slice.raise(DomainEvent::ItemConsumed {
        item: using.item.clone(),
        quantity: 1,
    });

    let origin = world
        .get::<Transform>(player)
        .map_or(Vec3::ZERO, |transform| transform.translation);
    let facing = world.get::<Facing>(player).map_or(Vec3::NEG_Z, |facing| facing.0);
    let collision = world.resource::<CollisionWorld>().cloned().unwrap_or_default();
    let targets = [player];
    let cast = Cast {
        actor: player,
        origin,
        facing,
        capsule: Capsule::default(),
        targets: &targets,
        point: None,
        collision: &collision,
        live_spawns: 0,
        live_projectiles: 0,
        tick,
    };
    // Filtered again here, not only at load: the executor never assumes the
    // validator ran.
    let outcomes: Vec<Outcome> = used
        .effects
        .iter()
        .filter(|effect| effect.allowed_on_use())
        .flat_map(|effect| ability::execute_effect(effect, cast, 0))
        .collect();
    apply_outcomes(world, slice, outcomes, 1.0, tick);
}

/// Starts `actor`'s `index`th ability, if it knows one and can pay for it.
///
/// Returns whether the cast began, so a caller can tell a refused input from one
/// that did nothing — the same contract [`start_attack`] follows.
pub(crate) fn start_ability(world: &mut World, slice: &Slice, actor: Entity, index: usize) -> bool {
    // ADR-0007: a committed action cannot be interrupted by another one, and
    // a staggered body cannot act at all.
    if world.get::<Attack>(actor).is_some()
        || world.get::<Dodge>(actor).is_some()
        || world.get::<Casting>(actor).is_some()
        || world.get::<Using>(actor).is_some()
        || staggered(world, actor)
    {
        return false;
    }
    let Some(id) = world
        .get::<Abilities>(actor)
        .and_then(|known| known.0.get(index).cloned())
    else {
        return false;
    };
    let Some(def) = slice.content.abilities.get(&id).cloned() else {
        return false;
    };
    let cooldowns = world.get::<Cooldowns>(actor).cloned().unwrap_or_default();
    let Some(stamina) = world.get_mut::<Stamina>(actor) else {
        return false;
    };
    // `activate` spends the cost only when every gate passes, so a refusal
    // leaves the actor exactly as it was.
    let Ok(activation) = ability::activate(&def, &cooldowns, stamina, false) else {
        return false;
    };

    let mut cooldowns = cooldowns;
    cooldowns.start(activation.ability.clone(), activation.cooldown_ticks);
    let _ = world.insert(actor, cooldowns);
    let _ = world.insert(actor, Casting::new(id, activation.timing));
    true
}

/// One tick of cooldowns, statuses, and every cast in flight.
///
/// `tick` is the engine's fixed-step index, never `state.clock.tick`: nothing
/// advances the canonical clock during play, so statuses measured against it
/// would never expire.
pub(crate) fn step_abilities(world: &mut World, slice: &mut Slice, tick: u64) {
    let cooling: Vec<Entity> = world.query::<Cooldowns>().map(|(entity, _)| entity).collect();
    for entity in cooling {
        if let Some(cooldowns) = world.get_mut::<Cooldowns>(entity) {
            cooldowns.tick();
        }
    }

    step_using(world, slice, tick);

    let afflicted: Vec<Entity> = world.query::<Statuses>().map(|(entity, _)| entity).collect();
    for entity in afflicted {
        if let Some(statuses) = world.get_mut::<Statuses>(entity) {
            statuses.expire(tick);
        }
        // Damage over time (M9): each tick's share, answered by the bearer's
        // resistances like any other blow of its kind.
        let burning = world
            .get::<Statuses>(entity)
            .map(|statuses| statuses.damage_over_time(tick))
            .unwrap_or_default();
        for (per_tick, kind) in burning {
            let landed = resistances_of(world, slice, entity).mitigate(Damage::new(per_tick, kind));
            if let Some(health) = world.get_mut::<Health>(entity) {
                health.apply_damage(landed);
            }
        }
    }

    let casting: Vec<(Entity, Casting)> = world
        .query::<Casting>()
        .map(|(entity, cast)| (entity, cast.clone()))
        .collect();
    for (actor, cast) in casting {
        if cast.should_fire() {
            fire(world, slice, actor, &cast.ability, tick);
            if let Some(active) = world.get_mut::<Casting>(actor) {
                active.fired = true;
            }
        }
        let finished = match world.get_mut::<Casting>(actor) {
            Some(active) => {
                active.tick();
                active.is_finished()
            }
            None => false,
        };
        if finished {
            let _ = world.remove::<Casting>(actor);
        }
    }
}

/// Resolves one ability's targets and executes its effects.
fn fire(world: &mut World, slice: &mut Slice, actor: Entity, id: &AbilityId, tick: u64) {
    let Some(def) = slice.content.abilities.get(id).cloned() else {
        return;
    };
    let origin = world
        .get::<Transform>(actor)
        .map_or(Vec3::ZERO, |transform| transform.translation);
    let facing = world.get::<Facing>(actor).map_or(Vec3::NEG_Z, |facing| facing.0);
    let caster_is_player = world.get::<Player>(actor).is_some();

    // Hostility is relative to the caster, which is what lets one rule serve a
    // player ability and an enemy one.
    let mut candidates: Vec<Candidate> = world
        .query2::<Transform, Enemy>()
        .map(|(entity, transform, _)| Candidate {
            entity,
            position: transform.translation,
            hostile: caster_is_player,
        })
        .collect();
    if let Some((player, position)) = world
        .query2::<Transform, Player>()
        .next()
        .map(|(entity, transform, _)| (entity, transform.translation))
    {
        candidates.push(Candidate {
            entity: player,
            position,
            hostile: !caster_is_player,
        });
    }

    let targets = ability::resolve_targets(
        def.targeting,
        Aim {
            actor,
            origin,
            facing,
            reach: ABILITY_REACH,
            point: None,
            candidates: &candidates,
        },
    );

    let live_spawns = world.query::<Enemy>().count();
    let live_projectiles = world.query::<Projectile>().count();
    let collision = world.resource::<CollisionWorld>().cloned().unwrap_or_default();
    let outcomes = ability::execute(
        &def,
        Cast {
            actor,
            origin,
            facing,
            capsule: Capsule::default(),
            targets: &targets,
            point: None,
            collision: &collision,
            live_spawns,
            live_projectiles,
            tick,
        },
    );

    // How it looks, queued for presentation (ADR-0017 §8). Queued after the
    // outcomes are decided and read by nothing that decides them: a request is
    // a note for a later frame, never an input to this one.
    let look = &def.presentation;
    if let Some(cast) = &look.cast {
        crate::vfx::request(world, cast, origin, origin);
    }
    if let Some(impact) = &look.impact {
        for target in touched(&outcomes) {
            if let Some(at) = world
                .get::<Transform>(target)
                .map(|transform| transform.translation)
            {
                crate::vfx::request(world, impact, origin, at);
            }
        }
    }
    // What the caster's statuses do to its damage, fixed now: a bolt carries
    // the strength it was thrown with, not whatever its thrower has later.
    let dealt = dealt_by(world, actor, tick);
    for outcome in &outcomes {
        if let Outcome::Launch(launch) = outcome {
            launch_projectile(world, &def.id, launch.clone(), caster_is_player, dealt, tick);
        }
    }
    apply_outcomes(world, slice, outcomes, dealt, tick);
}

/// The multiplier `entity`'s statuses put on damage it deals (M9).
fn dealt_by(world: &World, entity: Entity, tick: u64) -> f32 {
    world
        .get::<Statuses>(entity)
        .map_or(1.0, |statuses| statuses.damage_dealt_multiplier(tick))
}

/// Which ability a projectile belongs to, and whose side it is on.
///
/// Its side is fixed at launch rather than read from the owner as it flies:
/// an enemy's bolt still in the air when its caster dies is still aimed at
/// the player.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Flying {
    pub ability: AbilityId,
    pub from_player: bool,
    pub launched: u64,
    /// The damage multiplier its caster's statuses gave it at launch (M9).
    pub dealt: f32,
}

impl Component for Flying {}

/// Puts one launched projectile in the world. Simulation only: what it looks
/// like is presentation's to add, on the next frame.
fn launch_projectile(
    world: &mut World,
    ability: &AbilityId,
    launch: Launch,
    from_player: bool,
    dealt: f32,
    tick: u64,
) {
    let entity = world.spawn();
    let _ = world.insert(entity, Transform::from_translation(launch.from));
    let _ = world.insert(entity, Projectile::new(launch));
    let _ = world.insert(
        entity,
        Flying {
            ability: ability.clone(),
            from_player,
            launched: tick,
            dealt,
        },
    );
}

/// One tick of every projectile in flight (ADR-0017 §7).
///
/// Each flies by a swept test, strikes whoever of the other side it passes —
/// but not a body mid-roll, whose invulnerable window a projectile respects
/// exactly as a swing does — and applies its `on_hit` through the same door
/// as every other ability outcome, so armour and statuses answer it too.
pub(crate) fn step_projectiles(world: &mut World, slice: &mut Slice, tick: u64, seconds: f32) {
    let flying: Vec<Entity> = world.query::<Projectile>().map(|(entity, _)| entity).collect();
    if flying.is_empty() {
        return;
    }
    let collision = world.resource::<CollisionWorld>().cloned().unwrap_or_default();
    let body = Capsule::default().radius;
    for entity in flying {
        let (Some(mut projectile), Some(flight_info)) = (
            world.get::<Projectile>(entity).cloned(),
            world.get::<Flying>(entity).cloned(),
        ) else {
            continue;
        };
        let can_be_struck = |world: &World, target: Entity| {
            world.get::<Health>(target).is_some()
                && !world.get::<Dodge>(target).is_some_and(Dodge::is_invulnerable)
        };
        let mut candidates: Vec<Candidate> = world
            .query2::<Transform, Enemy>()
            .filter(|(target, _, _)| can_be_struck(world, *target))
            .map(|(target, transform, _)| Candidate {
                entity: target,
                position: transform.translation,
                hostile: flight_info.from_player,
            })
            .collect();
        if let Some((player, position)) = world
            .query2::<Transform, Player>()
            .next()
            .map(|(target, transform, _)| (target, transform.translation))
            && can_be_struck(world, player)
        {
            candidates.push(Candidate {
                entity: player,
                position,
                hostile: !flight_info.from_player,
            });
        }

        let flight = projectile::fly(&mut projectile, &candidates, body, &collision, seconds);
        let impact = slice
            .content
            .abilities
            .get(&flight_info.ability)
            .and_then(|def| def.presentation.impact.clone());
        for (target, at) in &flight.struck {
            let outcomes = projectile::strike(&projectile, *target, *at, &collision, tick);
            if let Some(impact) = &impact {
                crate::vfx::request(world, impact, *at, *at);
            }
            apply_outcomes(world, slice, outcomes, flight_info.dealt, tick);
        }
        match flight.end {
            Some(end) => {
                // A bolt that meets a wall bursts against it; one that simply
                // runs out of range fades without a show.
                if let (End::Wall(at), Some(impact)) = (end, &impact) {
                    crate::vfx::request(world, impact, at, at);
                }
                let _ = world.despawn(entity);
            }
            None => {
                if let Some(transform) = world.get_mut::<Transform>(entity) {
                    transform.translation = projectile.position;
                }
                let _ = world.insert(entity, projectile);
            }
        }
    }
}

/// Everyone an ability's outcomes change, once each, in outcome order.
///
/// What an impact effect plays on: not everyone in reach, only those the
/// ability actually did something to.
pub(crate) fn touched(outcomes: &[Outcome]) -> Vec<Entity> {
    let mut found: Vec<Entity> = Vec::new();
    for outcome in outcomes {
        let target = match outcome {
            Outcome::Damage { target, .. }
            | Outcome::Status { target, .. }
            | Outcome::Resource { target, .. } => *target,
            Outcome::Spawn { .. } | Outcome::Teleport { .. } | Outcome::Event { .. } | Outcome::Launch(_) => {
                continue;
            }
        };
        if !found.contains(&target) {
            found.push(target);
        }
    }
    found
}

/// Per-kind reduction protecting `entity`.
///
/// Equipment is the player's alone for now: nothing grants an enemy a loadout,
/// and inventing one here would be balance invented in the runtime rather than
/// authored in content.
fn resistances_of(world: &World, slice: &Slice, entity: Entity) -> Resistances {
    if world.get::<Player>(entity).is_some() {
        resolve_loadout(&slice.state.player, |id| slice.content.items.get(id)).resistances
    } else {
        Resistances::none()
    }
}

/// The receiving end of a melee hit: where it stands, whether it is mid-roll,
/// and everything standing between it and the blow.
///
/// Both swing paths build their target here, so armour and statuses answer a
/// sword exactly as they answer an ability. Before this existed only
/// `apply_outcomes` consulted either — and every enemy in the slice deals its
/// damage by swinging, so neither armour nor `status-guarded` changed anything
/// a player could actually be hit by.
fn protected(
    world: &World,
    slice: &Slice,
    entity: Entity,
    position: Vec3,
    invulnerable: bool,
    tick: u64,
) -> Target {
    let taken = world
        .get::<Statuses>(entity)
        .map_or(1.0, |statuses| statuses.damage_taken_multiplier(tick));
    Target::new(entity, Capsule::default(), position, invulnerable)
        .with_resistances(resistances_of(world, slice, entity))
        .with_damage_taken(taken)
}

/// Performs what the executor decided, and nothing it did not.
///
/// The executor bounded every one of these before they got here: counts are
/// capped, durations clamped, teleports collision-resolved. This is the half
/// that touches the ECS, kept separate so the rules stay testable without a
/// world (M6 wave 1).
/// `dealt` scales every blow, from the caster's statuses (M9).
fn apply_outcomes(world: &mut World, slice: &mut Slice, outcomes: Vec<Outcome>, dealt: f32, tick: u64) {
    for outcome in outcomes {
        match outcome {
            Outcome::Damage {
                target,
                amount,
                damage_type,
            } => {
                let taken = world
                    .get::<Statuses>(target)
                    .map_or(1.0, |statuses| statuses.damage_taken_multiplier(tick));
                let resistances = resistances_of(world, slice, target);
                let landed = resistances.mitigate(Damage::new(amount * dealt * taken, damage_type));
                if let Some(health) = world.get_mut::<Health>(target) {
                    health.apply_damage(landed);
                }
            }
            Outcome::Status {
                target,
                status,
                expires_at,
            } => {
                // What the status does is content (M9). The loader refuses an
                // `ApplyStatus` naming an undefined status, so a miss here is
                // content that bypassed it — and it is refused, not guessed at.
                let Some(def) = slice.content.statuses.get(&status) else {
                    engine::log_warn!("slice", "an undefined status was applied", status = status);
                    continue;
                };
                let modifiers = game_runtime::status::modifiers_of(def);
                let look = def.vfx.clone();
                if world.get::<Statuses>(target).is_none() {
                    let _ = world.insert(target, Statuses::new());
                }
                let applied = world
                    .get_mut::<Statuses>(target)
                    .is_some_and(|statuses| statuses.apply(status, expires_at, &modifiers));
                if applied
                    && let Some(look) = look
                    && let Some(at) = world
                        .get::<Transform>(target)
                        .map(|transform| transform.translation)
                {
                    crate::vfx::request(world, &look, at, at);
                }
            }
            Outcome::Spawn { prefab, count, at } => {
                for index in 0..count {
                    // Fanned out rather than stacked, so summons do not all
                    // arrive inside one another.
                    let offset = Vec3::new(f32::from(index + 1) * SUMMON_SPACING, 0.0, 0.0);
                    let spawn = match SpawnId::new("summoned") {
                        Ok(id) => id,
                        Err(_) => continue,
                    };
                    // No encounter: a summon belongs to no authored fight, so
                    // killing it cannot complete someone else's `Defeat`.
                    spawn_enemy_at(world, slice, &prefab, spawn, None, at + offset);
                }
            }
            Outcome::Resource {
                target,
                resource,
                amount,
            } => match resource {
                ResourceType::Health => {
                    if let Some(health) = world.get_mut::<Health>(target) {
                        if amount >= 0.0 {
                            health.heal(amount);
                        } else {
                            health.apply_damage(-amount);
                        }
                    }
                }
                ResourceType::Stamina => {
                    if let Some(stamina) = world.get_mut::<Stamina>(target) {
                        if amount >= 0.0 {
                            stamina.current = (stamina.current + amount).min(stamina.max);
                        } else {
                            stamina.try_spend(-amount);
                        }
                    }
                }
            },
            Outcome::Teleport { actor, to } => {
                if let Some(transform) = world.get_mut::<Transform>(actor) {
                    transform.translation = to;
                }
            }
            // Put in flight by `fire`, which knows the ability it belongs to;
            // nothing else produces one, and a strike's `on_hit` cannot.
            Outcome::Launch(_) => {}
            Outcome::Event { event } => {
                // Logged, not recorded. `Signal` is a quest vocabulary and
                // `DomainEvent` is finite; M6 adds to neither, because an entry
                // per cast would make the log unreadable for exactly the reason
                // one per sword swing would.
                engine::log_info!("slice", "ability event", event = event.0.to_string());
            }
        }
    }
}

/// Starts a player swing, if stamina allows and nothing else is in progress.
///
/// Returns whether the swing started, so a caller can tell a refused input from
/// one that did nothing.
pub(crate) fn start_attack(world: &mut World, slice: &Slice) -> bool {
    let Some(player) = player_entity(world) else {
        return false;
    };
    if world.get::<Attack>(player).is_some()
        || world.get::<Dodge>(player).is_some()
        || world.get::<Casting>(player).is_some()
        || world.get::<Using>(player).is_some()
        || staggered(world, player)
    {
        return false;
    }
    let combat = world.get::<PlayerCombat>(player).copied().unwrap_or_default();
    let Some(stamina) = world.get_mut::<Stamina>(player) else {
        return false;
    };
    if !stamina.try_spend(combat.cost) {
        return false;
    }
    // What is wielded is read at the moment of the swing, not cached on the
    // player: equipment changes between swings, and a cached copy would be a
    // second source of truth that the equip menu could invalidate. This is the
    // other half of the equip path — without it `ItemEquipped` would fill a
    // slot that nothing ever consults, which is the state wave 2 left behind.
    let loadout = resolve_loadout(&slice.state.player, |id| slice.content.items.get(id));
    let mut attack = Attack::new(
        combat.timing,
        combat.damage + loadout.bonus_damage,
        combat.reach,
        combat.radius,
    );
    // Only a weapon may retype a swing; `resolve_loadout` already enforces
    // that, so an unarmed player keeps the authored default.
    if let Some(kind) = loadout.damage_type {
        attack = attack.with_damage_type(kind);
    }
    let _ = world.insert(player, attack);
    true
}

/// Starts a dodge in `direction`, or along the player's facing if none is held.
///
/// A committed attack cannot be dodge-cancelled — that commitment is the whole
/// of ADR-0007's combat, and letting a roll erase it would make every swing free.
pub(crate) fn start_dodge(world: &mut World, direction: Vec2) -> bool {
    let Some(player) = player_entity(world) else {
        return false;
    };
    // A broken stance cannot roll out of its own stagger.
    if world.get::<Dodge>(player).is_some() || staggered(world, player) {
        return false;
    }
    if world.get::<Attack>(player).is_some_and(Attack::is_committed) {
        return false;
    }
    // A cast is committed exactly like a swing, so it cannot be rolled out of;
    // so is using an item (M9).
    if world.get::<Casting>(player).is_some() || world.get::<Using>(player).is_some() {
        return false;
    }

    let direction = if direction == Vec2::ZERO {
        let facing = world.get::<Facing>(player).map_or(Vec3::NEG_Z, |facing| facing.0);
        Vec2::new(facing.x, facing.z)
    } else {
        direction
    };

    let combat = world.get::<PlayerCombat>(player).copied().unwrap_or_default();
    let Some(stamina) = world.get_mut::<Stamina>(player) else {
        return false;
    };
    if !stamina.try_spend(combat.dodge_cost) {
        return false;
    }
    let _ = world.remove::<Attack>(player);
    let _ = world.insert(
        player,
        Dodge::new(direction, DODGE_DURATION, combat.dodge_iframes, DODGE_SPEED),
    );
    true
}

/// How long after a wave falls before the next arrives, in ticks: a second at
/// 60 Hz, long enough for the arrival effect to be seen where each one will
/// stand.
pub(crate) const WAVE_DELAY_TICKS: u64 = 60;

/// Where each encounter in this location stands in its waves (M8).
///
/// Rebuilt by [`populate`] on every arrival, like everything else a location
/// places: waves are how a fight unfolds, not a fact about the world, so
/// nothing here is saved and walking back in starts the fight over.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Waves {
    encounters: BTreeMap<EncounterId, WaveState>,
}

impl Resource for Waves {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WaveState {
    /// The wave that has not arrived yet.
    next: u8,
    /// The last wave this encounter has.
    last: u8,
    /// When `next` arrives, once the wave before it has fallen.
    due: Option<u64>,
}

impl Waves {
    /// Every encounter in `location` with more than one wave, waiting on its
    /// second.
    fn of(location: &LocationDef) -> Self {
        let mut encounters: BTreeMap<EncounterId, WaveState> = BTreeMap::new();
        for spawn in &location.spawns {
            if let SpawnRole::Enemy {
                encounter: Some(encounter),
                wave,
                ..
            } = &spawn.role
                && *wave > 0
            {
                let state = encounters.entry(encounter.clone()).or_insert(WaveState {
                    next: 1,
                    last: 0,
                    due: None,
                });
                state.last = state.last.max(*wave);
            }
        }
        Self { encounters }
    }

    /// Whether `encounter` has a wave still to come.
    pub(crate) fn pending(&self, encounter: &EncounterId) -> bool {
        self.encounters
            .get(encounter)
            .is_some_and(|state| state.next <= state.last)
    }
}

/// Sends in every wave that is due.
///
/// Fixed step, because a wave arriving is simulation: where enemies stand and
/// when is part of the fight, not of how it looks.
pub(crate) fn step_waves(world: &mut World, slice: &Slice, tick: u64) {
    let due: Vec<(EncounterId, u8)> = world
        .resource::<Waves>()
        .map(|waves| {
            waves
                .encounters
                .iter()
                .filter(|(_, state)| state.due.is_some_and(|at| at <= tick))
                .map(|(encounter, state)| (encounter.clone(), state.next))
                .collect()
        })
        .unwrap_or_default();
    let Some(location) = slice.current_location() else {
        return;
    };
    let capsule = Capsule::default();
    for (encounter, wave) in due {
        for spawn in &location.spawns {
            if let SpawnRole::Enemy {
                prefab,
                encounter: Some(placed),
                wave: arrives,
            } = &spawn.role
                && *placed == encounter
                && *arrives == wave
            {
                let at = to_world(spawn.placement.at);
                let stand = Vec3::new(at.x, capsule.half_extent(), at.z);
                spawn_enemy_at(
                    world,
                    slice,
                    prefab,
                    spawn.id.clone(),
                    Some(encounter.clone()),
                    stand,
                );
            }
        }
        if let Some(state) = world
            .resource_mut::<Waves>()
            .and_then(|waves| waves.encounters.get_mut(&encounter))
        {
            state.next += 1;
            state.due = None;
        }
    }
}

/// Schedules `encounter`'s next wave, and marks where each of it will stand.
fn call_next_wave(world: &mut World, slice: &Slice, encounter: &EncounterId, tick: u64) {
    let Some(next) = world
        .resource_mut::<Waves>()
        .and_then(|waves| waves.encounters.get_mut(encounter))
        .filter(|state| state.due.is_none())
        .map(|state| {
            state.due = Some(tick + WAVE_DELAY_TICKS);
            state.next
        })
    else {
        return;
    };
    let arrival = slice
        .content
        .encounters
        .get(encounter)
        .and_then(|def| def.arrival.clone());
    let (Some(arrival), Some(location)) = (arrival, slice.current_location()) else {
        return;
    };
    for spawn in &location.spawns {
        if let SpawnRole::Enemy {
            encounter: Some(placed),
            wave,
            ..
        } = &spawn.role
            && placed == encounter
            && *wave == next
        {
            let at = to_world(spawn.placement.at);
            crate::vfx::request(world, &arrival, at, at);
        }
    }
}

/// Loots, despawns, and counts every enemy whose health has run out.
///
/// Separate from [`step_combat`], and driven by *health* rather than by what a
/// particular swing returned. The earlier version collected the dead from the
/// player's hit results, which meant an enemy killed by anything else — an
/// ability, a hazard, a scripted death — would keep standing, drop nothing, and
/// never count toward its encounter. Nothing else deals damage yet, so that was
/// latent rather than broken; reaping on `is_dead` makes it correct for whatever
/// arrives later.
pub(crate) fn reap_dead(world: &mut World, slice: &mut Slice) {
    let dead: Vec<Entity> = world
        .query2::<Health, Enemy>()
        .filter(|(_, health, _)| health.is_dead())
        .map(|(entity, _, _)| entity)
        .collect();

    for entity in dead {
        let Some(enemy) = world.get::<Enemy>(entity).cloned() else {
            continue;
        };

        // Experience for the kill (ADR-0018): levels bring talent points.
        let experience = slice
            .content
            .enemies
            .get(&enemy.prefab)
            .map_or(0, |def| def.experience);
        if experience > 0 {
            slice
                .pending
                .push(DomainEvent::ExperienceGained { amount: experience });
        }

        let drops: Vec<DomainEvent> = slice
            .content
            .enemies
            .get(&enemy.prefab)
            .and_then(|def| def.loot_table.clone())
            .and_then(|table| {
                let seed = loot_seed(&slice.state, &table);
                slice
                    .content
                    .loot_tables
                    .get(&table)
                    .map(|table| loot(table, seed))
            })
            .unwrap_or_default();
        slice.pending.extend(drops);

        leave_remains(world, entity);
        world.despawn(entity);

        // Asked after the despawn, so the enemy that just died is not counted
        // among the survivors of its own encounter.
        //
        // A fight with a wave still to come is not over when its standing
        // enemies are: the next wave is called instead, and the fight clears
        // only when its last wave falls (M8).
        if let Some(encounter) = enemy.encounter.clone() {
            let remaining = world
                .query::<Enemy>()
                .filter(|(_, other)| other.encounter.as_ref() == Some(&encounter))
                .count();
            if remaining == 0 {
                let pending = world
                    .resource::<Waves>()
                    .is_some_and(|waves| waves.pending(&encounter));
                if pending {
                    let tick = world
                        .resource::<engine::time::FixedTime>()
                        .map_or(0, |time| time.tick);
                    call_next_wave(world, slice, &encounter, tick);
                } else {
                    slice.signals.push(Signal::EncounterCleared { encounter });
                }
            }
        }
    }
}

/// How `entity`'s plain swings look: the player's class's, or an enemy's own.
fn melee_look(world: &World, slice: &Slice, entity: Entity) -> game_core::content::MeleePresentation {
    if world.get::<Player>(entity).is_some() {
        return slice
            .state
            .player
            .progression
            .class
            .as_ref()
            .and_then(|class| slice.content.classes.get(class))
            .map(|class| class.presentation.clone())
            .unwrap_or_default();
    }
    world
        .get::<Enemy>(entity)
        .and_then(|enemy| slice.content.enemies.get(&enemy.prefab))
        .map(|def| def.presentation.clone())
        .unwrap_or_default()
}

/// One tick of combat: attacks advance, hits resolve, the dead are cleared.
///
/// Runs in the fixed step, because every window it reads is counted in ticks.
pub(crate) fn step_combat(world: &mut World, slice: &mut Slice, tick: u64) {
    let player = player_entity(world);

    let regenerating: Vec<Entity> = world.query::<Stamina>().map(|(entity, _)| entity).collect();
    for entity in regenerating {
        // A status can slow or hasten recovery (M9).
        let scale = world
            .get::<Statuses>(entity)
            .map_or(1.0, |statuses| statuses.stamina_regen_multiplier(tick));
        if let Some(stamina) = world.get_mut::<Stamina>(entity) {
            stamina.tick_scaled(scale);
        }
    }

    let swinging: Vec<Entity> = world.query::<Attack>().map(|(entity, _)| entity).collect();
    for entity in &swinging {
        if let Some(attack) = world.get_mut::<Attack>(*entity) {
            attack.tick();
        }
    }

    // A swing that went live this tick shows its path: the swinger's slash,
    // from where it stands toward where the blow lands.
    for entity in &swinging {
        let Some(attack) = world.get::<Attack>(*entity).copied() else {
            continue;
        };
        if attack.elapsed() != attack.timing.windup || attack.phase() != AttackPhase::Active {
            continue;
        }
        let Some(swing) = melee_look(world, slice, *entity).swing else {
            continue;
        };
        let from = world
            .get::<Transform>(*entity)
            .map_or(Vec3::ZERO, |transform| transform.translation);
        let facing = world
            .get::<Facing>(*entity)
            .map_or(Vec3::NEG_Z, |facing| facing.0);
        crate::vfx::request(world, &swing, from, attack.hit_centre(from, facing));
    }

    // The player's swing. One connecting hit per attack: `mark_spent` is what
    // stops a five-tick active window dealing its damage five times.
    if let Some(player) = player
        && let Some(mut swing) = world.get::<Attack>(player).copied()
    {
        // The copy that lands carries the swinger's statuses (M9); the stored
        // attack is only ever marked spent.
        swing.damage *= dealt_by(world, player, tick);
        let origin = world
            .get::<Transform>(player)
            .map_or(Vec3::ZERO, |transform| transform.translation);
        let facing = world.get::<Facing>(player).map_or(Vec3::NEG_Z, |facing| facing.0);
        let targets: Vec<(Entity, Vec3)> = world
            .query2::<Transform, Enemy>()
            .map(|(entity, transform, _)| (entity, transform.translation))
            .collect();

        for (entity, position) in targets {
            let target = protected(world, slice, entity, position, false, tick);
            let Some(health) = world.get_mut::<Health>(entity) else {
                continue;
            };
            let outcome = resolve_hit(
                Attacker {
                    attack: &swing,
                    position: origin,
                    facing,
                },
                target,
                health,
            );
            if outcome.is_ok() {
                if let Some(attack) = world.get_mut::<Attack>(player) {
                    attack.mark_spent();
                }
                if let Some(hit) = melee_look(world, slice, player).hit {
                    crate::vfx::request(world, &hit, origin, position);
                }
                break;
            }
        }
    }

    // Enemy swings, against a player who may be mid-roll.
    if let Some(player) = player {
        let player_position = world
            .get::<Transform>(player)
            .map_or(Vec3::ZERO, |transform| transform.translation);
        let invulnerable = world.get::<Dodge>(player).is_some_and(Dodge::is_invulnerable);

        let swings: Vec<(Entity, Attack, Vec3, Vec3)> = world
            .query2::<Transform, Enemy>()
            .filter_map(|(entity, transform, _)| {
                let mut attack = world.get::<Attack>(entity).copied()?;
                attack.damage *= dealt_by(world, entity, tick);
                let facing = world.get::<Facing>(entity).map_or(Vec3::NEG_Z, |facing| facing.0);
                Some((entity, attack, transform.translation, facing))
            })
            .collect();

        for (entity, attack, position, facing) in swings {
            let target = protected(world, slice, player, player_position, invulnerable, tick);
            let Some(health) = world.get_mut::<Health>(player) else {
                break;
            };
            let outcome = resolve_hit(
                Attacker {
                    attack: &attack,
                    position,
                    facing,
                },
                target,
                health,
            );
            if outcome.is_ok() {
                if let Some(attack) = world.get_mut::<Attack>(entity) {
                    attack.mark_spent();
                }
                if let Some(hit) = melee_look(world, slice, entity).hit {
                    crate::vfx::request(world, &hit, position, player_position);
                }
            }
        }
    }

    reap_dead(world, slice);

    // Finished attacks and rolls stop existing, and an enemy that finished
    // swinging is told so, which is what starts its recovery.
    let spent: Vec<Entity> = world
        .query::<Attack>()
        .filter(|(_, attack)| attack.is_finished())
        .map(|(entity, _)| entity)
        .collect();
    for entity in spent {
        let _ = world.remove::<Attack>(entity);
        if let Some(brain) = world.get_mut::<EnemyBrain>(entity) {
            brain.finish_attack();
        }
    }

    let rolled: Vec<Entity> = world
        .query::<Dodge>()
        .filter(|(_, dodge)| dodge.is_finished())
        .map(|(entity, _)| entity)
        .collect();
    for entity in rolled {
        let _ = world.remove::<Dodge>(entity);
    }
}

/// How many ticks a full stamina bar takes to refill, and how long spending it
/// delays that refill.
const STAMINA_REFILL_TICKS: f32 = 120.0;
const STAMINA_RECOVERY_DELAY: u32 = 30;

/// Spawns the player at `start`, with the pools canonical state records.
///
/// Derived, never invented. The runtime used to open with `Health::full(100.0)`
/// while canonical state said `0/0`, so the durable record and the thing the
/// player actually had were two unrelated numbers (ADR-0015).
/// Why a reload of content in a development run was refused (M9).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReloadRefused {
    /// No content root is known, so there is nothing to reload from.
    NoRoot,
    /// The new content does not load; each problem, as the loader reports it.
    Invalid(Vec<String>),
    /// The new content loads but would orphan the running world: the save it
    /// becomes would name content that is gone.
    Orphans(Vec<String>),
}

impl ReloadRefused {
    pub(crate) fn problems(&self) -> &[String] {
        match self {
            Self::NoRoot => &[],
            Self::Invalid(problems) | Self::Orphans(problems) => problems,
        }
    }
}

/// Reloads content from disk and swaps it in, or refuses and changes nothing
/// (M9 wave 3). The caller chooses the moment: a safe point with no fight
/// and no panel open.
///
/// On success the current location is rebuilt from the new content the way
/// a door rebuilds one, and the player's body from its class, where it
/// stands. Returns the new content's definition count, and whether its
/// models, effects, or names changed — those are uploaded once at startup,
/// so they wait for a restart.
pub(crate) fn reload_content(world: &mut World, slice: &mut Slice) -> Result<(usize, bool), ReloadRefused> {
    let root = slice.content_root.clone().ok_or(ReloadRefused::NoRoot)?;
    // Paths relative to the content folder, so a notice names the file.
    let prefix = format!("{}{}", root.display(), std::path::MAIN_SEPARATOR);
    let fresh = content_runtime::load(&root).map_err(|report| {
        ReloadRefused::Invalid(
            report
                .problems
                .iter()
                .map(|problem| problem.to_string().replace(&prefix, ""))
                .collect(),
        )
    })?;

    // Worded for a one-line notice: what in the world would lose what.
    let mut orphans: Vec<String> = content_runtime::verify_world_references(&fresh, &slice.state, &root)
        .problems
        .iter()
        .map(|problem| match problem {
            content_runtime::Problem::Dangling { from, to, .. } => format!("{from}: {to} is gone"),
            other => other.to_string(),
        })
        .collect();
    if !fresh.locations.contains_key(&slice.location) {
        orphans.push(format!("the location you stand in, {}, is gone", slice.location));
    }
    if !orphans.is_empty() {
        return Err(ReloadRefused::Orphans(orphans));
    }

    let old = &slice.content;
    let presentation_changed = old.models != fresh.models
        || old.palettes != fresh.palettes
        || old.body_plans != fresh.body_plans
        || old.clips != fresh.clips
        || old.vfx != fresh.vfx
        || old
            .statuses
            .values()
            .map(|def| (&def.name, &def.color))
            .ne(fresh.statuses.values().map(|def| (&def.name, &def.color)))
        || old
            .enemies
            .values()
            .map(|def| &def.name)
            .ne(fresh.enemies.values().map(|def| &def.name));
    let count = fresh.len();

    let at = player_entity(world)
        .and_then(|player| world.get::<Transform>(player))
        .map(|transform| transform.translation);
    slice.content = fresh;
    let here = slice.location.clone();
    travel_to(world, slice, &here, None);
    respawn_player(world, slice);
    if let Some(at) = at {
        move_player_to(world, at);
    }
    Ok((count, presentation_changed))
}

/// Replaces the player's body with one built from canonical state, where it
/// stands (M9): what picking a class needs, since pools, loadout, stance, and
/// model all come from the class.
pub(crate) fn respawn_player(world: &mut World, slice: &Slice) -> Option<Entity> {
    let old = player_entity(world)?;
    let at = world.get::<Transform>(old)?.translation;
    world.despawn(old);
    Some(spawn_player(world, slice, at))
}

pub(crate) fn spawn_player(world: &mut World, slice: &Slice, start: Vec3) -> Entity {
    let pools = &slice.state.player;
    let entity = world.spawn();
    let _ = world.insert(entity, Transform::from_xyz(start.x, start.y, start.z));
    let _ = world.insert(entity, Player);
    let _ = world.insert(entity, Facing::default());
    let _ = world.insert(
        entity,
        Health {
            current: pools.health.current as f32,
            max: pools.health.max as f32,
        },
    );
    let stamina = pools.stamina.max as f32;
    let _ = world.insert(
        entity,
        Stamina::new(stamina, stamina / STAMINA_REFILL_TICKS, STAMINA_RECOVERY_DELAY),
    );
    // The class's authored loadout. Which abilities the player has is a content
    // decision, the way ADR-0007's classes are data rather than an enum; a class
    // declaring none simply gets none, and the runtime invents nothing.
    let known: Vec<AbilityId> = pools
        .progression
        .class
        .as_ref()
        .and_then(|class| slice.content.classes.get(class))
        .map(|class| class.abilities.clone())
        .unwrap_or_default();
    if !known.is_empty() {
        let _ = world.insert(entity, Abilities(known));
    }
    let _ = world.insert(entity, Cooldowns::new());
    let _ = world.insert(entity, Statuses::new());
    let combat = pools
        .progression
        .class
        .as_ref()
        .and_then(|class| slice.content.classes.get(class))
        .map_or_else(PlayerCombat::default, |class| PlayerCombat::of(&class.combat));
    let _ = world.insert(entity, combat);
    let _ = world.insert(entity, Poise::new(combat.poise));
    if let Some(visuals) = world.resource::<SliceVisuals>().copied() {
        let model = pools
            .progression
            .class
            .as_ref()
            .and_then(|class| slice.content.classes.get(class))
            .and_then(|class| class.model.clone());
        dress_as(world, entity, &visuals, model.as_ref(), visuals.player);
    }
    entity
}

/// Which authored location is home.
///
/// Content's business rather than a constant here: the one marked `Settlement`
/// is where the dead wake up, and where a save whose location no longer exists
/// falls back to. `values()` is sorted by id, so a world with two settlements
/// still picks the same one twice. Expressed once and called from both places —
/// two implementations of "which location is home" would be the bug.
pub(crate) fn settlement(content: &ContentRegistry) -> Option<LocationId> {
    content
        .locations
        .values()
        .find(|location| location.kind == LocationKind::Settlement)
        .map(|location| location.id.clone())
}

/// Whether any enemy is currently aware of the player.
///
/// Awareness rather than the player's own swinging: a swing at empty air is not
/// a fight, and an enemy that has noticed you is one whether or not either side
/// has landed anything yet. `EnemyBrain` already answers this — aggro is
/// hysteretic, so a player hovering at the edge of a radius does not flicker in
/// and out of combat every tick.
fn anyone_hostile(world: &World) -> bool {
    world
        .query::<EnemyBrain>()
        .any(|(_, brain)| brain.state() != EnemyState::Idle)
}

/// Reconciles health when a fight ends.
///
/// Combat is the one durable thing that happens entirely in the runtime, and
/// its result used to be written only at a transition, a death, or a save. A
/// player who fought, won, and then stood still kept an out-of-date record
/// until one of those happened.
///
/// The end of a fight is the fourth safe point, and the cheapest to notice: the
/// tick on which the last enemy aware of the player stops being aware — because
/// it died, or because they left. Both endings want the same thing recorded.
///
/// The *edge* is what matters, not the state. Syncing on every tick out of
/// combat would be a per-tick write of something that has not changed, and
/// syncing on every tick *in* combat is exactly the running commentary
/// [`sync_health`] exists to avoid.
///
/// Returns whether a fight ended on this tick.
pub(crate) fn sync_on_combat_exit(world: &World, slice: &mut Slice) -> bool {
    let fighting = anyone_hostile(world);
    let left_combat = slice.in_combat && !fighting;
    slice.in_combat = fighting;

    if left_combat {
        sync_health(world, slice);
    }
    left_combat
}

/// Writes the runtime's health into canonical state, if they disagree.
///
/// Combat runs in the runtime simulation and only its outcomes become events: a
/// `PlayerHealthChanged` per hit would make the log unreadable and the save
/// churn every tick. So health is reconciled at *safe points* — entering a
/// location, dying, and saving — and the log carries a handful of coarse deltas
/// instead of a running commentary on a fight.
///
/// Returns whether anything was queued.
pub(crate) fn sync_health(world: &World, slice: &mut Slice) -> bool {
    let Some(player) = player_entity(world) else {
        return false;
    };
    let Some(health) = world.get::<Health>(player) else {
        return false;
    };

    // Canonical pools are whole points and the runtime works in floats.
    // Rounded rather than truncated, so surviving on 0.4 health is recorded as
    // surviving rather than as having died.
    let runtime = i64::from(health.current.max(0.0).round() as u32);

    // Measured against what canonical state *will* say once this tick commits,
    // not against what it says now. Safe points nest — `respawn_if_dead` syncs,
    // then calls `travel_to`, which syncs again — and comparing against
    // committed state each time made the second call re-queue a delta the first
    // had already queued. Dying then recorded -120 twice and clamped to zero.
    let pending: i64 = slice
        .pending
        .iter()
        .filter_map(|event| match event {
            DomainEvent::PlayerHealthChanged { delta } => Some(i64::from(*delta)),
            _ => None,
        })
        .sum();
    let recorded = (i64::from(slice.state.player.health.current) + pending)
        .clamp(0, i64::from(slice.state.player.health.max));

    if runtime == recorded {
        return false;
    }

    let delta = runtime - recorded;
    slice.raise(DomainEvent::PlayerHealthChanged {
        delta: delta.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
    });
    true
}

/// The nearest entity of a kind within [`INTERACT_RANGE`], if any.
fn nearest<C: Component>(world: &World, from: Vec3) -> Option<(engine::ecs::Entity, f32)> {
    let mut best: Option<(engine::ecs::Entity, f32)> = None;
    for (entity, transform, _) in world.query2::<Transform, C>() {
        let distance = (transform.translation - from).length();
        if distance <= INTERACT_RANGE && best.is_none_or(|(_, closest)| distance < closest) {
            best = Some((entity, distance));
        }
    }
    best
}

/// What the player's interact key found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Interaction {
    Talk(NpcId),
    Examine {
        entity: engine::ecs::Entity,
        clue: ClueId,
    },
    Decide {
        entity: engine::ecs::Entity,
        choice: ChoiceId,
    },
    Harvest {
        entity: engine::ecs::Entity,
        item: ItemId,
    },
    /// A forge or bench within reach.
    Craft,
    /// The settlement's upgrade board within reach.
    Upgrade,
    /// A doorway. Where it leads is resolved from the location's links.
    Use(SpawnId),
    Nothing,
}

/// Retires every choice marker whose choice was settled this tick (M9),
/// however it was settled — at the marker, or in a conversation. A decision is
/// made once; a marker left standing after it would offer to make it again.
pub(crate) fn retire_settled_points(world: &mut World) {
    let settled: Vec<ChoiceId> = world.resource::<Slice>().map_or_else(Vec::new, |slice| {
        slice
            .signals
            .iter()
            .filter_map(|signal| match signal {
                Signal::Decided { choice } => Some(choice.clone()),
                _ => None,
            })
            .collect()
    });
    if settled.is_empty() {
        return;
    }
    let points: Vec<Entity> = world
        .query::<ChoicePoint>()
        .filter(|(_, point)| settled.contains(&point.choice))
        .map(|(entity, _)| entity)
        .collect();
    for entity in points {
        if let Some(point) = world.get_mut::<ChoicePoint>(entity) {
            point.settled = true;
        }
    }
}

/// What the player would interact with from `from`.
///
/// Resolved in a fixed order — people, then things the story turns on, then
/// resources, then fixtures — so standing between a merchant and a bush does
/// not pick a different one each tick.
///
/// Clues and decisions outrank resources deliberately: a quest step the player
/// is looking for should win over a bush they can pick any time.
pub(crate) fn interaction_at(world: &World, from: Vec3) -> Interaction {
    if let Some((entity, _)) = nearest::<Npc>(world, from)
        && let Some(npc) = world.get::<Npc>(entity)
    {
        return Interaction::Talk(npc.0.clone());
    }
    if let Some((entity, _)) = nearest::<Clue>(world, from)
        && let Some(clue) = world.get::<Clue>(entity)
        && !clue.examined
    {
        return Interaction::Examine {
            entity,
            clue: clue.clue.clone(),
        };
    }
    if let Some((entity, _)) = nearest::<ChoicePoint>(world, from)
        && let Some(point) = world.get::<ChoicePoint>(entity)
        && !point.settled
    {
        return Interaction::Decide {
            entity,
            choice: point.choice.clone(),
        };
    }
    if let Some((entity, _)) = nearest::<Gatherable>(world, from)
        && let Some(node) = world.get::<Gatherable>(entity)
        && !node.taken
    {
        return Interaction::Harvest {
            entity,
            item: node.item.clone(),
        };
    }
    if let Some((entity, _)) = nearest::<Station>(world, from)
        && let Some(station) = world.get::<Station>(entity)
    {
        // What a fixture does is authored now, rather than inferred from
        // whether a link happens to start at it.
        return match station.kind {
            StationKind::Crafting => Interaction::Craft,
            StationKind::Upgrades => Interaction::Upgrade,
            StationKind::Doorway => Interaction::Use(station.spawn.clone()),
        };
    }
    Interaction::Nothing
}

// ---------------------------------------------------------------------------
// Combat feedback
//
// Presentation only: nothing below changes what happens in a fight, only
// whether a player can see it happen. Every window combat runs on is counted
// in ticks, and a wind-up nobody can see is not a telegraph.
//
// Two channels, both existing engine seams. A body's colour says what it is
// doing (material swaps), and `DebugDraw` lines carry the numbers — pools,
// cooldowns, where a blow will land. The lines are an interim HUD until text
// and UI rendering exist; they are drawn always, not behind F1, because a
// health bar is not a debugging aid.
// ---------------------------------------------------------------------------

/// How long a body flashes after taking damage, in fixed ticks.
///
/// About 130 ms: long enough to register at a glance, short enough that two
/// quick hits read as two flashes rather than one smear.
pub(crate) const HURT_FLASH_TICKS: u64 = 8;

/// The material a body draws with when nothing is happening to it.
///
/// Recorded by [`dress`], so a cue can always be undone: a cue swaps the
/// renderer's material, and this is what it swaps back to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BaseMaterial(pub MaterialHandle);

impl Component for BaseMaterial {}

/// Health as of the last fixed tick, and until when the body flashes.
///
/// Damage is noticed by comparing pools rather than by hooking each place that
/// deals it: swings, ability effects, M7's projectiles, and whatever comes after
/// all end in a lower number, and one comparison cannot forget a path the way
/// three hooks could.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Wounds {
    last: f32,
    until: u64,
}

impl Component for Wounds {}

/// What a body is visibly doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cue {
    Idle,
    Windup,
    Strike,
    Evading,
    Hurt,
}

/// The one cue a body shows, when several are true at once.
///
/// A body has one colour, so overlapping states need an order, and the order is
/// "what would you most regret missing". Being hit beats everything, because it
/// answers "did that land?". The roll's invulnerability beats an attack,
/// because a player mid-dodge needs to know whether the i-frames are up. A live
/// hit volume beats its own wind-up.
pub(crate) fn cue(hurt: bool, evading: bool, phase: Option<AttackPhase>) -> Cue {
    if hurt {
        Cue::Hurt
    } else if evading {
        Cue::Evading
    } else {
        match phase {
            Some(AttackPhase::Active) => Cue::Strike,
            Some(AttackPhase::Windup) => Cue::Windup,
            _ => Cue::Idle,
        }
    }
}

/// Notices damage taken this tick, for [`show_cues`] to flash.
///
/// Runs in the fixed step after everything that can deal damage, so the flash
/// is measured in the same ticks as the hit that caused it.
pub(crate) fn note_wounds(world: &mut World, tick: u64) {
    let pools: Vec<(Entity, f32)> = world
        .query::<Health>()
        .map(|(entity, health)| (entity, health.current))
        .collect();
    for (entity, current) in pools {
        if let Some(wounds) = world.get_mut::<Wounds>(entity) {
            let taken = wounds.last - current;
            if current < wounds.last {
                wounds.until = tick + HURT_FLASH_TICKS;
            }
            wounds.last = current;
            // Poise reads the same drop in health the flash does, so a swing,
            // an ability and a bolt all break a stance the same way (M8).
            let broke = world
                .get_mut::<Poise>(entity)
                .is_some_and(|poise| poise.take(taken, tick));
            if broke {
                stagger(world, entity);
            }
            continue;
        }
        let _ = world.insert(
            entity,
            Wounds {
                last: current,
                until: 0,
            },
        );
    }
}

/// Whether `entity`'s stance is broken this tick.
pub(crate) fn staggered(world: &World, entity: Entity) -> bool {
    let tick = world
        .resource::<engine::time::FixedTime>()
        .map_or(0, |time| time.tick);
    world
        .get::<Poise>(entity)
        .is_some_and(|poise| poise.is_staggered(tick))
}

/// A broken stance: whatever `entity` was still winding up is lost.
///
/// Only a wind-up. A blow already live lands — a trade is a trade — and one
/// in recovery has nothing left to cancel. This is the whole point of poise:
/// swinging into someone's blade costs you your swing.
fn stagger(world: &mut World, entity: Entity) {
    if world
        .get::<Attack>(entity)
        .is_some_and(|attack| attack.phase() == AttackPhase::Windup)
    {
        let _ = world.remove::<Attack>(entity);
    }
    if world
        .get::<Casting>(entity)
        .is_some_and(|casting| casting.phase() == AttackPhase::Windup)
    {
        let _ = world.remove::<Casting>(entity);
    }
    // An item being used is lost with the stance — but not spent, since it
    // is only spent on finishing (M9).
    let _ = world.remove::<Using>(entity);
}

/// One tick of every stance recovering.
pub(crate) fn recover_poise(world: &mut World) {
    let bodies: Vec<Entity> = world.query::<Poise>().map(|(entity, _)| entity).collect();
    for entity in bodies {
        if let Some(poise) = world.get_mut::<Poise>(entity) {
            poise.tick();
        }
    }
}

/// Recolours every body to the cue it should show this frame.
pub(crate) fn show_cues(world: &mut World, tick: u64) {
    let Some(visuals) = world.resource::<SliceVisuals>().copied() else {
        return;
    };
    let bodies: Vec<(Entity, MaterialHandle)> = world
        .query::<BaseMaterial>()
        .map(|(entity, base)| (entity, base.0))
        .collect();
    for (entity, base) in bodies {
        let hurt = world
            .get::<Wounds>(entity)
            .is_some_and(|wounds| tick < wounds.until);
        let evading = world.get::<Dodge>(entity).is_some_and(Dodge::is_invulnerable);
        let phase = world
            .get::<Attack>(entity)
            .map(Attack::phase)
            .or_else(|| world.get::<Casting>(entity).map(Casting::phase));
        let material = match cue(hurt, evading, phase) {
            Cue::Idle => base,
            Cue::Windup => visuals.cues.windup,
            Cue::Strike => visuals.cues.strike,
            Cue::Evading => visuals.cues.evading,
            Cue::Hurt => visuals.cues.hurt,
        };
        paint(world, entity, material);
    }
}

/// Sets what a body draws with: every part of a model, or its one mesh.
fn paint(world: &mut World, entity: Entity, material: MaterialHandle) {
    if let Some(model) = world.get_mut::<Model>(entity) {
        for part in &mut model.parts {
            part.material = material;
        }
    } else if let Some(renderer) = world.get_mut::<MeshRenderer>(entity) {
        renderer.material = material;
    }
}

const BAR_WIDTH: f32 = 1.0;
/// Gap between the top of a body and its first bar.
const BAR_LIFT: f32 = 0.3;
/// A debug line is one pixel wide, so a bar is several stacked.
const BAR_ROWS: [f32; 3] = [0.0, 0.025, 0.05];
/// Vertical distance between one bar and the next above it.
const BAR_STEP: f32 = 0.12;
const PIP_WIDTH: f32 = 0.28;
const PIP_GAP: f32 = 0.08;
const CAST_BURST_RADIUS: f32 = 1.1;
const GUARD_RING_RADIUS: f32 = 0.75;

const BAR_EMPTY: Vec4 = Vec4::new(0.12, 0.12, 0.14, 1.0);
const ENEMY_HEALTH: Vec4 = Vec4::new(0.90, 0.18, 0.18, 1.0);
const PLAYER_HEALTH: Vec4 = Vec4::new(0.30, 0.85, 0.30, 1.0);
const PLAYER_STAMINA: Vec4 = Vec4::new(0.95, 0.80, 0.20, 1.0);
const COOLDOWN_READY: Vec4 = Vec4::new(0.30, 0.80, 1.00, 1.0);
const COOLDOWN_WAITING: Vec4 = Vec4::new(0.40, 0.40, 0.46, 1.0);
const CAST_RING: Vec4 = Vec4::new(1.00, 0.95, 0.70, 1.0);
const GUARD_RING: Vec4 = Vec4::new(0.25, 0.45, 1.00, 1.0);

/// A share of a pool, safe to draw: never NaN, never outside `0..=1`.
pub(crate) fn fraction(current: f32, max: f32) -> f32 {
    if max > 0.0 && current.is_finite() {
        (current / max).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// A horizontal bar centred on `centre`, `filled` of it in `color`.
///
/// The filled and empty parts are drawn as separate segments rather than one
/// laid over the other: two lines at the same depth would z-fight.
fn bar(debug: &mut DebugDraw, centre: Vec3, width: f32, filled: f32, color: Vec4) {
    let left = centre - Vec3::X * (width * 0.5);
    let split = left + Vec3::X * (width * filled.clamp(0.0, 1.0));
    let right = left + Vec3::X * width;
    for lift in BAR_ROWS {
        let up = Vec3::Y * lift;
        if filled > 0.0 {
            debug.line(left + up, split + up, color);
        }
        if filled < 1.0 {
            debug.line(split + up, right + up, BAR_EMPTY);
        }
    }
}

/// A flat circle of `radius` around `centre`.
fn ring(debug: &mut DebugDraw, centre: Vec3, radius: f32, color: Vec4) {
    const SEGMENTS: u16 = 20;
    let point = |index: u16| {
        let angle = std::f32::consts::TAU * f32::from(index) / f32::from(SEGMENTS);
        centre + Vec3::new(angle.cos() * radius, 0.0, angle.sin() * radius)
    };
    for index in 0..SEGMENTS {
        debug.line(point(index), point(index + 1), color);
    }
}

/// Just above the top of a body, where its bars sit.
pub(crate) fn crown(world: &World, entity: Entity, transform: &Transform) -> Vec3 {
    let top = world
        .get::<RenderBounds>(entity)
        .map_or(transform.translation.y + 0.5, |bounds| {
            bounds.world_bounds(&transform.to_matrix()).max.y
        });
    Vec3::new(transform.translation.x, top + BAR_LIFT, transform.translation.z)
}

/// Draws the numbers a fight runs on: pools, cooldowns, and where blows land.
pub(crate) fn draw_gauges(world: &World, debug: &mut DebugDraw, content: &ContentRegistry, tick: u64) {
    for (entity, transform, _) in world.query2::<Transform, Enemy>() {
        if let Some(health) = world.get::<Health>(entity) {
            let at = crown(world, entity, transform);
            bar(
                debug,
                at,
                BAR_WIDTH,
                fraction(health.current, health.max),
                ENEMY_HEALTH,
            );
        }
    }

    // Where a swing will land is drawn on the ground now, by
    // `vfx::show_telegraphs` (M7 wave 4), which replaced the rings drawn here.

    // A burst once a cast's effects have landed, for as long as its recovery
    // lasts: "the ability went off", for abilities whose effect is not itself
    // visible. The fallback for an ability with no effect of its own to play
    // on its caster; one that has one plays that instead.
    for (_, transform, casting) in world.query2::<Transform, Casting>() {
        let shown = content
            .abilities
            .get(&casting.ability)
            .is_some_and(|def| def.presentation.cast.is_some());
        if casting.fired && !casting.is_finished() && !shown {
            ring(debug, transform.translation, CAST_BURST_RADIUS, CAST_RING);
        }
    }

    let Some((player, transform, _)) = world.query2::<Transform, Player>().next() else {
        return;
    };
    let top = crown(world, player, transform);
    if let Some(health) = world.get::<Health>(player) {
        bar(
            debug,
            top + Vec3::Y * (BAR_STEP * 2.0),
            BAR_WIDTH,
            fraction(health.current, health.max),
            PLAYER_HEALTH,
        );
    }
    if let Some(stamina) = world.get::<Stamina>(player) {
        bar(
            debug,
            top + Vec3::Y * BAR_STEP,
            BAR_WIDTH,
            fraction(stamina.current, stamina.max),
            PLAYER_STAMINA,
        );
    }

    // One pip per loadout slot, in key order: full and bright when ready,
    // filling back up while it cools.
    if let (Some(abilities), Some(cooldowns)) =
        (world.get::<Abilities>(player), world.get::<Cooldowns>(player))
    {
        let count = abilities.0.len() as f32;
        let span = count * PIP_WIDTH + (count - 1.0).max(0.0) * PIP_GAP;
        let first = top.x - span * 0.5 + PIP_WIDTH * 0.5;
        for (slot, id) in abilities.0.iter().enumerate() {
            let total = content.abilities.get(id).map_or(0, |def| def.cooldown_ticks);
            let remaining = cooldowns.remaining_for(id);
            let filled = if total == 0 {
                1.0
            } else {
                1.0 - fraction(remaining as f32, total as f32)
            };
            let color = if remaining == 0 {
                COOLDOWN_READY
            } else {
                COOLDOWN_WAITING
            };
            let x = first + slot as f32 * (PIP_WIDTH + PIP_GAP);
            bar(debug, Vec3::new(x, top.y, top.z), PIP_WIDTH, filled, color);
        }
    }

    // Protected: any status that lowers damage taken, not one status by name,
    // so a second defensive status shows up with no change here.
    if world
        .get::<Statuses>(player)
        .is_some_and(|statuses| statuses.damage_taken_multiplier(tick) < 1.0)
    {
        ring(debug, transform.translation, GUARD_RING_RADIUS, GUARD_RING);
    }
}

// ---------------------------------------------------------------------------
// Motion
//
// Procedural animation for blocked-out bodies: a swing coils and then lunges,
// a cast swells and releases, a dodge rolls, a hit makes the body flinch, and
// a death slumps rather than blinking out.
//
// The rule that makes this safe: **presentation owns rotation and scale;
// simulation owns translation.** Movement, collision and every hit volume read
// `translation`, and nothing in the simulation reads or writes the other two —
// so a body can lean a long way into a blow without its hitbox moving at all.
// A test holds that line.
// ---------------------------------------------------------------------------

/// How long a fallen enemy takes to go, in fixed ticks: half a second.
pub(crate) const REMAINS_TICKS: u64 = 30;

/// Lean back while coiling, in radians (about 17 degrees).
const COIL_LEAN: f32 = -0.30;
/// Lean into the blow (about 29 degrees).
const STRIKE_LEAN: f32 = 0.50;
const COIL_SCALE: Vec3 = Vec3::new(1.10, 0.86, 1.10);
const STRIKE_SCALE: Vec3 = Vec3::new(0.94, 1.10, 0.94);
/// How far a cast swells while it channels, and how far it pops on release.
const CHANNEL_SCALE: f32 = 1.18;
const RELEASE_SCALE: f32 = 1.30;
const ROLL_SQUASH: Vec3 = Vec3::new(1.05, 0.80, 1.05);
/// The squash of a body that has just been hit, at full strength.
const FLINCH_SCALE: Vec3 = Vec3::new(1.20, 0.78, 1.20);
/// How far remains sink into the ground as they go.
const REMAINS_SINK: f32 = 0.4;

/// What a body is doing, and how far through it: the input to [`pose`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Motion {
    Still,
    Swing { phase: AttackPhase, progress: f32 },
    Cast { phase: AttackPhase, progress: f32 },
    Roll { progress: f32 },
}

/// A body's posture: a lean toward its facing (radians, negative is back), a
/// roll about the axis of a dodge, and a scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Pose {
    pub lean: f32,
    pub roll: f32,
    pub scale: Vec3,
}

impl Pose {
    pub(crate) const REST: Pose = Pose {
        lean: 0.0,
        roll: 0.0,
        scale: Vec3::ONE,
    };
}

/// Ease in and out, clamped to `0..=1`.
fn smooth(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn mix(from: f32, to: f32, t: f32) -> f32 {
    from + (to - from) * t
}

/// Which window a timed action is in, and how far through that window.
pub(crate) fn phase_progress(timing: AttackTiming, elapsed: u32) -> (AttackPhase, f32) {
    let through = |start: u32, length: u32| {
        if length == 0 {
            1.0
        } else {
            elapsed.saturating_sub(start) as f32 / length as f32
        }
    };
    if elapsed < timing.windup {
        (AttackPhase::Windup, through(0, timing.windup))
    } else if elapsed < timing.windup + timing.active {
        (AttackPhase::Active, through(timing.windup, timing.active))
    } else if elapsed < timing.total() {
        (
            AttackPhase::Recovery,
            through(timing.windup + timing.active, timing.recovery),
        )
    } else {
        (AttackPhase::Finished, 1.0)
    }
}

/// The posture for a motion, with `flinch` (`0..=1`) of a hit reaction on top.
///
/// Pure, so every pose can be tested without a scene. The shapes follow the
/// windows combat already runs on, which is the point: a wind-up you can *see*
/// coil is a wind-up you can learn to dodge.
pub(crate) fn pose(motion: Motion, flinch: f32) -> Pose {
    let mut posed = match motion {
        Motion::Still => Pose::REST,
        Motion::Swing { phase, progress } => {
            let eased = smooth(progress);
            match phase {
                AttackPhase::Windup => Pose {
                    lean: COIL_LEAN * eased,
                    roll: 0.0,
                    scale: Vec3::ONE.lerp(COIL_SCALE, eased),
                },
                // Snapped rather than eased across the window: the blow lands
                // in the first ticks of a short active window, so it reads as
                // a strike rather than a slow shove.
                AttackPhase::Active => {
                    let snap = smooth(progress * 3.0);
                    Pose {
                        lean: mix(COIL_LEAN, STRIKE_LEAN, snap),
                        roll: 0.0,
                        scale: COIL_SCALE.lerp(STRIKE_SCALE, snap),
                    }
                }
                AttackPhase::Recovery => Pose {
                    lean: mix(STRIKE_LEAN, 0.0, eased),
                    roll: 0.0,
                    scale: STRIKE_SCALE.lerp(Vec3::ONE, eased),
                },
                AttackPhase::Finished => Pose::REST,
            }
        }
        Motion::Cast { phase, progress } => {
            let eased = smooth(progress);
            let swell = match phase {
                AttackPhase::Windup => mix(1.0, CHANNEL_SCALE, eased),
                AttackPhase::Active => RELEASE_SCALE,
                AttackPhase::Recovery => mix(RELEASE_SCALE, 1.0, eased),
                AttackPhase::Finished => 1.0,
            };
            Pose {
                lean: if phase == AttackPhase::Windup {
                    COIL_LEAN * 0.5 * eased
                } else {
                    0.0
                },
                roll: 0.0,
                scale: Vec3::splat(swell),
            }
        }
        Motion::Roll { progress } => {
            // Squashed most in the middle of the roll, upright at both ends.
            let squash = (std::f32::consts::PI * progress.clamp(0.0, 1.0)).sin();
            Pose {
                lean: 0.0,
                roll: std::f32::consts::TAU * smooth(progress),
                scale: Vec3::ONE.lerp(ROLL_SQUASH, squash),
            }
        }
    };
    let flinch = flinch.clamp(0.0, 1.0);
    if flinch > 0.0 {
        posed.scale *= Vec3::ONE.lerp(FLINCH_SCALE, flinch);
    }
    posed
}

/// Turns a pose into a rotation: face `facing`, lean toward it, roll toward
/// `rolling`.
fn orient(posed: Pose, facing: Vec3, rolling: Vec3) -> Quat {
    // Rotating `Y` about `Y x d` by a positive angle tips the top toward `d`,
    // so a positive lean is forward and a positive roll goes the way of the
    // dodge.
    let about = |direction: Vec3, angle: f32| {
        let axis = Vec3::Y
            .cross(Vec3::new(direction.x, 0.0, direction.z))
            .normalize_or_zero();
        if axis == Vec3::ZERO || angle == 0.0 {
            Quat::IDENTITY
        } else {
            Quat::from_axis_angle(axis, angle)
        }
    };
    // Turned to face where it is facing first — a model has a front, where a
    // cube did not — then leaned and rolled about world axes derived from
    // that same facing. `-Z` is forward, so a facing of `-Z` is no turn.
    let flat = Vec3::new(facing.x, 0.0, facing.z);
    let yaw = if flat.length_squared() < 1e-6 {
        Quat::IDENTITY
    } else {
        Quat::from_rotation_y((-flat.x).atan2(-flat.z))
    };
    about(rolling, posed.roll) * about(facing, posed.lean) * yaw
}

/// Poses every body that fights, for this frame.
///
/// Writes `rotation` and `scale` only — see the section notes for why that is
/// the whole of the safety argument.
pub(crate) fn pose_bodies(world: &mut World, tick: u64) {
    let bodies: Vec<Entity> = world.query::<Health>().map(|(entity, _)| entity).collect();
    for entity in bodies {
        let motion = if let Some(dodge) = world.get::<Dodge>(entity) {
            Motion::Roll {
                progress: dodge.elapsed() as f32 / dodge.duration.max(1) as f32,
            }
        } else if let Some(attack) = world.get::<Attack>(entity) {
            let (phase, progress) = phase_progress(attack.timing, attack.elapsed());
            Motion::Swing { phase, progress }
        } else if let Some(casting) = world.get::<Casting>(entity) {
            let (phase, progress) = phase_progress(casting.timing, casting.elapsed);
            Motion::Cast { phase, progress }
        } else {
            Motion::Still
        };
        // A broken stance holds the full flinch for as long as it lasts, so a
        // stagger reads as more than a hit.
        let reeling = world
            .get::<Poise>(entity)
            .is_some_and(|poise| poise.is_staggered(tick));
        let flinch = if reeling {
            1.0
        } else {
            world.get::<Wounds>(entity).map_or(0.0, |wounds| {
                if tick < wounds.until {
                    (wounds.until - tick) as f32 / HURT_FLASH_TICKS as f32
                } else {
                    0.0
                }
            })
        };
        let facing = world.get::<Facing>(entity).map_or(Vec3::NEG_Z, |facing| facing.0);
        let rolling = world.get::<Dodge>(entity).map_or(Vec3::ZERO, |dodge| {
            let direction = dodge.direction();
            Vec3::new(direction.x, 0.0, direction.y)
        });

        let posed = pose(motion, flinch);
        if let Some(transform) = world.get_mut::<Transform>(entity) {
            transform.rotation = orient(posed, facing, rolling);
            transform.scale = posed.scale;
        }
    }
}

/// What is left of an enemy for a moment after it dies.
///
/// Visual only: no `Enemy`, no `Health`, nothing a fight or a quest can see.
/// The enemy itself still despawns the tick it dies, exactly as before — an
/// objective counting survivors must not wait on an animation — and this
/// stands where it fell just long enough for "did I kill it?" to have an answer
/// you can see.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Remains {
    /// The tick it was first drawn, set on first sight: `reap_dead` runs inside
    /// the fixed step and does not carry the clock.
    started: Option<u64>,
    rest: Vec3,
    base: MaterialHandle,
}

impl Component for Remains {}

/// Leaves a copy of `entity`'s body where it stands, for [`age_remains`].
fn leave_remains(world: &mut World, entity: Entity) {
    let Some(transform) = world.get::<Transform>(entity).copied() else {
        return;
    };
    let model = world.get::<Model>(entity).cloned();
    let renderer = world.get::<MeshRenderer>(entity).copied();
    let Some(drawn) = model
        .as_ref()
        .and_then(|model| model.parts.first().map(|part| part.material))
        .or(renderer.map(|renderer| renderer.material))
    else {
        return;
    };
    let base = world.get::<BaseMaterial>(entity).map_or(drawn, |base| base.0);
    let bounds = world.get::<RenderBounds>(entity).copied();

    let remains = world.spawn();
    let _ = world.insert(remains, transform);
    if let Some(mut model) = model {
        // Slumped from rest rather than frozen mid-swing: the pose is
        // animation, and a body that has stopped acting has no animation.
        for part in &mut model.parts {
            part.material = base;
            part.pose = Transform::IDENTITY;
        }
        let _ = world.insert(remains, model);
    } else if let Some(renderer) = renderer {
        let _ = world.insert(remains, MeshRenderer::new(renderer.mesh, base));
    }
    let _ = world.insert(remains, Visibility::default());
    if let Some(bounds) = bounds {
        let _ = world.insert(remains, bounds);
    }
    let _ = world.insert(
        remains,
        Remains {
            started: None,
            rest: transform.translation,
            base,
        },
    );
}

/// Flashes, slumps and sinks each set of remains, then removes it.
///
/// Remains may move, because they are not simulated: nothing collides with
/// them and nothing can hit them.
pub(crate) fn age_remains(world: &mut World, tick: u64) {
    let hurt = world.resource::<SliceVisuals>().map(|visuals| visuals.cues.hurt);
    let all: Vec<Entity> = world.query::<Remains>().map(|(entity, _)| entity).collect();
    for entity in all {
        let Some(remains) = world.get_mut::<Remains>(entity) else {
            continue;
        };
        let started = *remains.started.get_or_insert(tick);
        let (rest, base) = (remains.rest, remains.base);
        let age = tick.saturating_sub(started);
        if age >= REMAINS_TICKS {
            world.despawn(entity);
            continue;
        }

        let fall = smooth(age as f32 / REMAINS_TICKS as f32);
        if let Some(transform) = world.get_mut::<Transform>(entity) {
            transform.translation = rest - Vec3::Y * (REMAINS_SINK * fall);
            transform.scale = Vec3::new(1.0 + 0.35 * fall, (1.0 - fall).max(0.02), 1.0 + 0.35 * fall);
        }
        // The killing blow's flash, which the enemy itself never gets to show:
        // it is despawned the same tick its health reaches zero.
        let material = match hurt {
            Some(hurt) if age < HURT_FLASH_TICKS => hurt,
            _ => base,
        };
        paint(world, entity, material);
    }
}
