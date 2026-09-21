//! Content manifests, compilation, registries, and validation.
//!
//! Loads human-authored RON from the asset root, compiles it into typed
//! registries, and reports every problem it finds. `game_core::content` owns
//! the *shapes* and their intrinsic invariants; this crate owns everything that
//! needs to have seen more than one file — uniqueness, cross-references, and
//! the final say on whether a content set is loadable.
//!
//! # Determinism
//!
//! Directory traversal is sorted before anything is parsed. Content ids feed
//! content hashes and save provenance, so "which file won" must never depend on
//! the order a filesystem happened to return entries in.
//!
//! # Every problem, not the first one
//!
//! Loading collects a full [`ContentReport`] rather than returning at the first
//! failure. An author fixing content wants the list; failing fast turns one
//! pass into ten.
//!
//! # Layout
//!
//! Each family is either a directory of RON files (`items/*.ron`) or a single
//! file (`items.ron`), so a small family stays in one file and a large one can
//! be split without a format change. Every file holds a list of definitions.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde::de::DeserializeOwned;

mod completable;
pub use completable::Unfinishable;

use game_core::content::{
    AbilityDef, BodyPlanDef, ClassDef, ClipDef, ContentError, DialogueDef, DialogueEffect, Effect,
    EncounterDef, EnemyDef, ItemDef, LevelCurveDef, LocationDef, LootTableDef, ModelDef, Objective,
    PaletteDef, QuestDef, RecipeDef, RegionDef, SettlementUpgradeDef, SpawnRole, StationKind, StatusDef,
    StatusId, VfxDef,
};
use game_core::ids::{
    AbilityId, BodyPlanId, ClassId, ClipId, ContentHash, DialogueId, EncounterId, ItemId, LocationId,
    LootTableId, ModelId, PaletteId, PrefabId, QuestId, UpgradeId, VfxId,
};

/// The families this loader knows about, in load order.
pub const FAMILIES: [&str; 19] = [
    "items",
    "recipes",
    "abilities",
    "classes",
    "enemies",
    "encounters",
    "quests",
    "locations",
    "dialogues",
    "loot-tables",
    "upgrades",
    "body-plans",
    "palettes",
    "models",
    "clips",
    "vfx",
    "statuses",
    "regions",
    "level-curves",
];

/// Compiled, cross-checked content.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ContentRegistry {
    pub items: BTreeMap<ItemId, ItemDef>,
    pub recipes: BTreeMap<ItemId, RecipeDef>,
    pub abilities: BTreeMap<AbilityId, AbilityDef>,
    pub classes: BTreeMap<ClassId, ClassDef>,
    pub enemies: BTreeMap<PrefabId, EnemyDef>,
    pub encounters: BTreeMap<EncounterId, EncounterDef>,
    pub quests: BTreeMap<QuestId, QuestDef>,
    pub locations: BTreeMap<LocationId, LocationDef>,
    pub dialogues: BTreeMap<DialogueId, DialogueDef>,
    pub loot_tables: BTreeMap<LootTableId, LootTableDef>,
    pub upgrades: BTreeMap<UpgradeId, SettlementUpgradeDef>,
    /// The joints models hang their parts from (ADR-0017).
    pub body_plans: BTreeMap<BodyPlanId, BodyPlanDef>,
    /// The colours models are painted from.
    pub palettes: BTreeMap<PaletteId, PaletteDef>,
    /// Low-poly model recipes.
    pub models: BTreeMap<ModelId, ModelDef>,
    /// Animation clips (ADR-0017 §5).
    pub clips: BTreeMap<ClipId, ClipDef>,
    /// Visual effects (ADR-0017 §6).
    pub vfx: BTreeMap<VfxId, VfxDef>,
    /// What each status means (M9): its name, colour, and modifiers.
    pub statuses: BTreeMap<StatusId, StatusDef>,
    /// The named stretches of world locations belong to (M9).
    pub regions: BTreeMap<game_core::ids::RegionId, RegionDef>,
    /// How much experience each level takes (ADR-0018).
    pub level_curves: BTreeMap<game_core::ids::LevelCurveId, LevelCurveDef>,
}

impl ContentRegistry {
    /// Total definitions across every family.
    pub fn len(&self) -> usize {
        self.items.len()
            + self.recipes.len()
            + self.abilities.len()
            + self.classes.len()
            + self.enemies.len()
            + self.encounters.len()
            + self.quests.len()
            + self.locations.len()
            + self.dialogues.len()
            + self.loot_tables.len()
            + self.upgrades.len()
            + self.body_plans.len()
            + self.palettes.len()
            + self.models.len()
            + self.clips.len()
            + self.vfx.len()
            + self.statuses.len()
            + self.regions.len()
            + self.level_curves.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Identifies this compiled content set (ADR-0014).
    ///
    /// Hashes the *compiled registry* rather than raw file bytes, in sorted-id
    /// order family by family, so the value is a property of the content's
    /// meaning: reindenting a file or splitting a family across two files
    /// leaves it unchanged, while renaming an id or altering a value does not.
    ///
    /// Determinism comes from our own ordering rules, not from BLAKE3 —
    /// `BTreeMap` iteration is sorted, and each entry is length-prefixed so no
    /// two different content sets can produce the same byte stream.
    pub fn content_hash(&self) -> Result<ContentHash, HashError> {
        let mut hasher = blake3::Hasher::new();
        hash_family(&mut hasher, "items", &self.items)?;
        hash_family(&mut hasher, "recipes", &self.recipes)?;
        hash_family(&mut hasher, "abilities", &self.abilities)?;
        hash_family(&mut hasher, "classes", &self.classes)?;
        hash_family(&mut hasher, "enemies", &self.enemies)?;
        hash_family(&mut hasher, "encounters", &self.encounters)?;
        hash_family(&mut hasher, "quests", &self.quests)?;
        hash_family(&mut hasher, "locations", &self.locations)?;
        hash_family(&mut hasher, "dialogues", &self.dialogues)?;
        hash_family(&mut hasher, "loot-tables", &self.loot_tables)?;
        hash_family(&mut hasher, "upgrades", &self.upgrades)?;
        hash_family(&mut hasher, "body-plans", &self.body_plans)?;
        hash_family(&mut hasher, "palettes", &self.palettes)?;
        hash_family(&mut hasher, "models", &self.models)?;
        hash_family(&mut hasher, "clips", &self.clips)?;
        hash_family(&mut hasher, "vfx", &self.vfx)?;
        hash_family(&mut hasher, "statuses", &self.statuses)?;
        hash_family(&mut hasher, "regions", &self.regions)?;
        hash_family(&mut hasher, "level-curves", &self.level_curves)?;
        Ok(ContentHash::from_bytes(*hasher.finalize().as_bytes()))
    }
}

/// A definition could not be serialized for hashing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashError {
    pub family: &'static str,
    pub id: String,
    pub message: String,
}

impl fmt::Display for HashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cannot hash {} {}: {}", self.family, self.id, self.message)
    }
}

impl std::error::Error for HashError {}

/// Feeds one family into the hasher.
///
/// The family name and entry count are mixed in as domain separators, and every
/// id and body is length-prefixed. Without that, two different content sets
/// could concatenate to the same bytes — ids `ab` + `c` and `a` + `bc` being
/// the simplest example.
fn hash_family<K: fmt::Display, V: Serialize>(
    hasher: &mut blake3::Hasher,
    family: &'static str,
    definitions: &BTreeMap<K, V>,
) -> Result<(), HashError> {
    hasher.update(family.as_bytes());
    hasher.update(&(definitions.len() as u64).to_le_bytes());
    for (id, definition) in definitions {
        let id = id.to_string();
        let body = ron::to_string(definition).map_err(|error| HashError {
            family,
            id: id.clone(),
            message: error.to_string(),
        })?;
        hasher.update(&(id.len() as u64).to_le_bytes());
        hasher.update(id.as_bytes());
        hasher.update(&(body.len() as u64).to_le_bytes());
        hasher.update(body.as_bytes());
    }
    Ok(())
}

/// One thing wrong with a content set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Problem {
    /// A file or directory could not be read.
    Unreadable { path: PathBuf, message: String },
    /// A file is not valid RON, or does not match its family's shape.
    Parse { path: PathBuf, message: String },
    /// A definition failed its own intrinsic validation.
    Invalid {
        path: PathBuf,
        id: String,
        source: ContentError,
    },
    /// Two definitions in the same family share an id.
    DuplicateId {
        family: &'static str,
        id: String,
        first: PathBuf,
        second: PathBuf,
    },
    /// An instance-scoped id appeared in authored content (ADR-0010): only
    /// generated instance content may carry one, and this is not that.
    ScopedId {
        path: PathBuf,
        family: &'static str,
        id: String,
    },
    /// A reference points at something that does not exist. Names both ends,
    /// because "missing item" without the referrer is a scavenger hunt.
    Dangling {
        path: PathBuf,
        from: String,
        to_family: &'static str,
        to: String,
    },
    /// A quest that loads but that no player could finish (M9): an objective
    /// whose NPC, clue, choice, fight, item, or place cannot be reached from
    /// where the quest is given. `objective` is `None` when the quest itself
    /// can never start.
    Unfinishable {
        quest: QuestId,
        objective: Option<game_core::ids::ObjectiveId>,
        why: Unfinishable,
    },
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unreadable { path, message } => {
                write!(f, "{}: cannot read: {message}", path.display())
            }
            Self::Parse { path, message } => write!(f, "{}: {message}", path.display()),
            Self::Invalid { path, id, source } => {
                write!(f, "{}: {id}: {source}", path.display())
            }
            Self::DuplicateId {
                family,
                id,
                first,
                second,
            } => write!(
                f,
                "duplicate {family} id {id}: defined in {} and {}",
                first.display(),
                second.display()
            ),
            Self::ScopedId { path, family, id } => write!(
                f,
                "{}: {family} id {id} is instance-scoped; authored content must be canonical",
                path.display()
            ),
            Self::Dangling {
                path,
                from,
                to_family,
                to,
            } => write!(
                f,
                "{}: {from} references {to_family} {to}, which does not exist",
                path.display()
            ),
            Self::Unfinishable {
                quest,
                objective: Some(objective),
                why,
            } => write!(
                f,
                "quest {quest}: objective {objective} cannot be completed: {why}"
            ),
            Self::Unfinishable {
                quest,
                objective: None,
                why,
            } => write!(f, "quest {quest} cannot be started: {why}"),
        }
    }
}

/// Everything wrong with a content set.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ContentReport {
    pub problems: Vec<Problem>,
}

impl ContentReport {
    pub fn is_empty(&self) -> bool {
        self.problems.is_empty()
    }

    pub fn len(&self) -> usize {
        self.problems.len()
    }
}

/// Checks that every content id a saved world names still exists.
///
/// A save carries a content hash, so a mismatch is *detectable* — but during
/// authoring content changes constantly, and refusing to load on a mismatch
/// would make every save dead within a day. What must not pass silently is a
/// world naming content that no longer exists: a deleted quest, a renamed item.
/// Those become [`Problem::Dangling`], reported exactly as the loader reports a
/// dangling reference in authored content, with the save as the referring path.
///
/// This lives here rather than in `game_core` because
/// `WorldState::validate_canonical` says so in as many words: it validates the
/// id *namespace* only, and reference resolution belongs to the layer holding
/// the registries.
///
/// # Locations are deliberately not checked
///
/// A save naming a location content no longer defines is *recoverable*: the
/// player wakes up in the settlement, which is what `Slice::resume` does with
/// it. A deleted item, class, or quest has no such recovery, so those are
/// refused. Reporting a vanished location here would turn a survivable content
/// edit into a dead save — the opposite of what refusing is for. What is worth
/// refusing is what cannot be repaired.
///
/// # What this cannot check
///
/// Only ids with a registry behind them can be resolved. Faction ids, world and
/// region flags, clue ids, and NPC ids have no content family — they are
/// vocabularies authored inline across locations and dialogue — so a save may
/// name one that nothing defines and this will not notice. Saying so plainly
/// beats implying a completeness the registries cannot support.
pub fn verify_world_references(
    content: &ContentRegistry,
    world: &game_core::world::WorldState,
    save: &Path,
) -> ContentReport {
    let mut problems = Vec::new();
    {
        // Every map walked here is a `BTreeMap`, so problems come out in id
        // order: the same broken save reports the same list on every machine.
        let mut dangling = |from: &str, to_family: &'static str, to: String| {
            problems.push(Problem::Dangling {
                path: save.to_path_buf(),
                from: from.to_owned(),
                to_family,
                to,
            });
        };

        if let Some(class) = &world.player.progression.class
            && !content.classes.contains_key(class)
        {
            dangling("the player's class", "classes", class.to_string());
        }
        for item in world.player.inventory.keys() {
            if !content.items.contains_key(item) {
                dangling("the player's inventory", "items", item.to_string());
            }
        }
        for item in world.player.equipment.values() {
            if !content.items.contains_key(item) {
                dangling("the player's equipment", "items", item.to_string());
            }
        }
        for (quest, progress) in &world.quests {
            let Some(def) = content.quests.get(quest) else {
                dangling("recorded quest progress", "quests", quest.to_string());
                continue;
            };
            // A step the save records progress on must still be a step (M9):
            // a quest whose current node vanished could never advance again.
            for objective in progress.objectives.keys() {
                if !def.nodes.contains_key(objective) {
                    dangling(
                        "recorded quest progress",
                        "quest objectives",
                        format!("{quest}#{objective}"),
                    );
                }
            }
        }
    }
    ContentReport { problems }
}

impl fmt::Display for ContentReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for problem in &self.problems {
            writeln!(f, "{problem}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ContentReport {}

/// Loads and cross-checks every family under `root`.
///
/// Returns the registry only when the content set is wholly valid; otherwise
/// every problem found, so one pass fixes everything.
pub fn load(root: &Path) -> Result<ContentRegistry, ContentReport> {
    let mut report = ContentReport::default();
    let mut registry = ContentRegistry::default();

    // A missing root must be loud. Treating it as an empty content set would
    // let the game boot with no content at all and no explanation, which is a
    // far worse failure than refusing to start.
    if !root.is_dir() {
        report.problems.push(Problem::Unreadable {
            path: root.to_path_buf(),
            message: "content root does not exist or is not a directory".to_owned(),
        });
        return Err(report);
    }

    let items = read_family::<ItemDef>(root, "items", &mut report);
    let recipes = read_family::<RecipeDef>(root, "recipes", &mut report);
    let abilities = read_family::<AbilityDef>(root, "abilities", &mut report);
    let classes = read_family::<ClassDef>(root, "classes", &mut report);
    let enemies = read_family::<EnemyDef>(root, "enemies", &mut report);
    let encounters = read_family::<EncounterDef>(root, "encounters", &mut report);
    let quests = read_family::<QuestDef>(root, "quests", &mut report);
    let locations = read_family::<LocationDef>(root, "locations", &mut report);
    let dialogues = read_family::<DialogueDef>(root, "dialogues", &mut report);
    let loot_tables = read_family::<LootTableDef>(root, "loot-tables", &mut report);
    let upgrades = read_family::<SettlementUpgradeDef>(root, "upgrades", &mut report);
    let body_plans = read_family::<BodyPlanDef>(root, "body-plans", &mut report);
    let palettes = read_family::<PaletteDef>(root, "palettes", &mut report);
    let models = read_family::<ModelDef>(root, "models", &mut report);
    let clips = read_family::<ClipDef>(root, "clips", &mut report);
    let vfx = read_family::<VfxDef>(root, "vfx", &mut report);
    let statuses = read_family::<StatusDef>(root, "statuses", &mut report);
    let regions = read_family::<RegionDef>(root, "regions", &mut report);
    let level_curves = read_family::<LevelCurveDef>(root, "level-curves", &mut report);

    for (path, def) in items {
        validate_intrinsic(&path, def.id.to_string(), def.validate(), &mut report);
        require_canonical(&path, "items", def.id.is_canonical(), &def.id, &mut report);
        insert_unique(
            &mut registry.items,
            def.id.clone(),
            def,
            &path,
            "items",
            &mut report,
        );
    }
    for (path, def) in recipes {
        validate_intrinsic(&path, def.id.to_string(), def.validate(), &mut report);
        require_canonical(&path, "recipes", def.id.is_canonical(), &def.id, &mut report);
        insert_unique(
            &mut registry.recipes,
            def.id.clone(),
            def,
            &path,
            "recipes",
            &mut report,
        );
    }
    for (path, def) in abilities {
        validate_intrinsic(&path, def.id.to_string(), def.validate(), &mut report);
        require_canonical(&path, "abilities", def.id.is_canonical(), &def.id, &mut report);
        insert_unique(
            &mut registry.abilities,
            def.id.clone(),
            def,
            &path,
            "abilities",
            &mut report,
        );
    }
    for (path, def) in classes {
        validate_intrinsic(&path, def.id.to_string(), def.validate(), &mut report);
        require_canonical(&path, "classes", def.id.is_canonical(), &def.id, &mut report);
        insert_unique(
            &mut registry.classes,
            def.id.clone(),
            def,
            &path,
            "classes",
            &mut report,
        );
    }

    for (path, def) in enemies {
        validate_intrinsic(&path, def.id.to_string(), def.validate(), &mut report);
        require_canonical(&path, "enemies", def.id.is_canonical(), &def.id, &mut report);
        insert_unique(
            &mut registry.enemies,
            def.id.clone(),
            def,
            &path,
            "enemies",
            &mut report,
        );
    }
    for (path, def) in encounters {
        validate_intrinsic(&path, def.id.to_string(), def.validate(), &mut report);
        require_canonical(&path, "encounters", def.id.is_canonical(), &def.id, &mut report);
        insert_unique(
            &mut registry.encounters,
            def.id.clone(),
            def,
            &path,
            "encounters",
            &mut report,
        );
    }
    for (path, def) in quests {
        validate_intrinsic(&path, def.id.to_string(), def.validate(), &mut report);
        require_canonical(&path, "quests", def.id.is_canonical(), &def.id, &mut report);
        insert_unique(
            &mut registry.quests,
            def.id.clone(),
            def,
            &path,
            "quests",
            &mut report,
        );
    }

    for (path, def) in locations {
        validate_intrinsic(&path, def.id.to_string(), def.validate(), &mut report);
        require_canonical(&path, "locations", def.id.is_canonical(), &def.id, &mut report);
        insert_unique(
            &mut registry.locations,
            def.id.clone(),
            def,
            &path,
            "locations",
            &mut report,
        );
    }
    for (path, def) in dialogues {
        validate_intrinsic(&path, def.id.to_string(), def.validate(), &mut report);
        require_canonical(&path, "dialogues", def.id.is_canonical(), &def.id, &mut report);
        insert_unique(
            &mut registry.dialogues,
            def.id.clone(),
            def,
            &path,
            "dialogues",
            &mut report,
        );
    }
    for (path, def) in loot_tables {
        validate_intrinsic(&path, def.id.to_string(), def.validate(), &mut report);
        require_canonical(&path, "loot-tables", def.id.is_canonical(), &def.id, &mut report);
        insert_unique(
            &mut registry.loot_tables,
            def.id.clone(),
            def,
            &path,
            "loot-tables",
            &mut report,
        );
    }
    for (path, def) in upgrades {
        validate_intrinsic(&path, def.id.to_string(), def.validate(), &mut report);
        require_canonical(&path, "upgrades", def.id.is_canonical(), &def.id, &mut report);
        insert_unique(
            &mut registry.upgrades,
            def.id.clone(),
            def,
            &path,
            "upgrades",
            &mut report,
        );
    }

    for (path, def) in body_plans {
        validate_intrinsic(&path, def.id.to_string(), def.validate(), &mut report);
        require_canonical(&path, "body-plans", def.id.is_canonical(), &def.id, &mut report);
        insert_unique(
            &mut registry.body_plans,
            def.id.clone(),
            def,
            &path,
            "body-plans",
            &mut report,
        );
    }
    for (path, def) in palettes {
        validate_intrinsic(&path, def.id.to_string(), def.validate(), &mut report);
        require_canonical(&path, "palettes", def.id.is_canonical(), &def.id, &mut report);
        insert_unique(
            &mut registry.palettes,
            def.id.clone(),
            def,
            &path,
            "palettes",
            &mut report,
        );
    }
    for (path, def) in models {
        validate_intrinsic(&path, def.id.to_string(), def.validate(), &mut report);
        require_canonical(&path, "models", def.id.is_canonical(), &def.id, &mut report);
        insert_unique(
            &mut registry.models,
            def.id.clone(),
            def,
            &path,
            "models",
            &mut report,
        );
    }
    for (path, def) in clips {
        validate_intrinsic(&path, def.id.to_string(), def.validate(), &mut report);
        require_canonical(&path, "clips", def.id.is_canonical(), &def.id, &mut report);
        insert_unique(
            &mut registry.clips,
            def.id.clone(),
            def,
            &path,
            "clips",
            &mut report,
        );
    }
    for (path, def) in vfx {
        validate_intrinsic(&path, def.id.to_string(), def.validate(), &mut report);
        require_canonical(&path, "vfx", def.id.is_canonical(), &def.id, &mut report);
        insert_unique(&mut registry.vfx, def.id.clone(), def, &path, "vfx", &mut report);
    }
    for (path, def) in statuses {
        validate_intrinsic(&path, def.id.0.to_string(), def.validate(), &mut report);
        require_canonical(&path, "statuses", def.id.0.is_canonical(), &def.id.0, &mut report);
        insert_unique(
            &mut registry.statuses,
            def.id.clone(),
            def,
            &path,
            "statuses",
            &mut report,
        );
    }

    for (path, def) in regions {
        validate_intrinsic(&path, def.id.to_string(), def.validate(), &mut report);
        require_canonical(&path, "regions", def.id.is_canonical(), &def.id, &mut report);
        insert_unique(
            &mut registry.regions,
            def.id.clone(),
            def,
            &path,
            "regions",
            &mut report,
        );
    }

    for (path, def) in level_curves {
        validate_intrinsic(&path, def.id.to_string(), def.validate(), &mut report);
        require_canonical(&path, "level-curves", def.id.is_canonical(), &def.id, &mut report);
        insert_unique(
            &mut registry.level_curves,
            def.id.clone(),
            def,
            &path,
            "level-curves",
            &mut report,
        );
    }

    resolve_references(&registry, &mut report);
    completable::check(&registry, &mut report);

    if report.is_empty() {
        Ok(registry)
    } else {
        Err(report)
    }
}

/// Files belonging to `family`, sorted, from either layout.
fn family_files(root: &Path, family: &str, report: &mut ContentReport) -> Vec<PathBuf> {
    let directory = root.join(family);
    if directory.is_dir() {
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) => {
                report.problems.push(Problem::Unreadable {
                    path: directory,
                    message: error.to_string(),
                });
                return Vec::new();
            }
        };
        let mut files: Vec<PathBuf> = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|extension| extension == "ron"))
            .collect();
        // Sorted so compilation is deterministic regardless of the filesystem.
        files.sort();
        return files;
    }

    let single = root.join(format!("{family}.ron"));
    if single.is_file() {
        vec![single]
    } else {
        Vec::new()
    }
}

/// Parses every file of one family, recording problems rather than stopping.
fn read_family<T: DeserializeOwned>(
    root: &Path,
    family: &'static str,
    report: &mut ContentReport,
) -> Vec<(PathBuf, T)> {
    let mut loaded = Vec::new();
    for path in family_files(root, family, report) {
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) => {
                report.problems.push(Problem::Unreadable {
                    path,
                    message: error.to_string(),
                });
                continue;
            }
        };
        match ron::from_str::<Vec<T>>(&text) {
            Ok(defs) => loaded.extend(defs.into_iter().map(|def| (path.clone(), def))),
            Err(error) => report.problems.push(Problem::Parse {
                path,
                message: error.to_string(),
            }),
        }
    }
    loaded
}

fn validate_intrinsic(
    path: &Path,
    id: String,
    outcome: Result<(), ContentError>,
    report: &mut ContentReport,
) {
    if let Err(source) = outcome {
        report.problems.push(Problem::Invalid {
            path: path.to_path_buf(),
            id,
            source,
        });
    }
}

fn require_canonical(
    path: &Path,
    family: &'static str,
    canonical: bool,
    id: &impl fmt::Display,
    report: &mut ContentReport,
) {
    if !canonical {
        report.problems.push(Problem::ScopedId {
            path: path.to_path_buf(),
            family,
            id: id.to_string(),
        });
    }
}

fn insert_unique<K: Ord + fmt::Display + Clone, V>(
    registry: &mut BTreeMap<K, V>,
    key: K,
    value: V,
    path: &Path,
    family: &'static str,
    report: &mut ContentReport,
) {
    if registry.contains_key(&key) {
        report.problems.push(Problem::DuplicateId {
            family,
            id: key.to_string(),
            // The first definition's file is not tracked per entry; naming the
            // duplicate's file and the id is enough to find both by grep, and
            // avoids carrying provenance for every definition.
            first: PathBuf::from("<previously loaded>"),
            second: path.to_path_buf(),
        });
        return;
    }
    registry.insert(key, value);
}

/// Checks every cross-family reference the schemas can express.
fn resolve_references(registry: &ContentRegistry, report: &mut ContentReport) {
    let dangling = |report: &mut ContentReport, from: String, to_family: &'static str, to: String| {
        report.problems.push(Problem::Dangling {
            path: PathBuf::from("<registry>"),
            from,
            to_family,
            to,
        });
    };

    // A model names its body plan and palette, and every part names a slot on
    // that plan and swatches in that palette. Slots and swatches are local
    // names rather than families, so they report as `<plan>/slots` and
    // `<palette>/swatches` to say where the name was looked for.
    for model in registry.models.values() {
        let plan = registry.body_plans.get(&model.body_plan);
        if plan.is_none() {
            dangling(
                report,
                model.id.to_string(),
                "body-plans",
                model.body_plan.to_string(),
            );
        }
        let palette = registry.palettes.get(&model.palette);
        if palette.is_none() {
            dangling(
                report,
                model.id.to_string(),
                "palettes",
                model.palette.to_string(),
            );
        }
        for part in &model.parts {
            if let Some(plan) = plan
                && !plan.slots.iter().any(|slot| slot.name == part.slot)
            {
                dangling(
                    report,
                    model.id.to_string(),
                    "slots",
                    format!("{}/{}", plan.id, part.slot),
                );
            }
            if let Some(palette) = palette {
                for swatch in [Some(&part.color), part.top.as_ref(), part.bottom.as_ref()]
                    .into_iter()
                    .flatten()
                {
                    if !palette.swatches.contains_key(swatch) {
                        dangling(
                            report,
                            model.id.to_string(),
                            "swatches",
                            format!("{}/{}", palette.id, swatch),
                        );
                    }
                }
            }
        }
    }

    // Animation sets name clips, and each clip must suit the state it plays
    // in. A clip's tracks name slots that must exist on *some* body plan: not
    // necessarily the one using it, since a clip is shared across plans and a
    // missing slot is skipped — but a slot no plan has is a typo.
    let known_slots: std::collections::BTreeSet<_> = registry
        .body_plans
        .values()
        .flat_map(|plan| plan.slots.iter().map(|slot| &slot.name))
        .collect();
    let check_set = |report: &mut ContentReport, owner: String, set: &game_core::content::AnimationSet| {
        for (state, clip) in set.uses() {
            match registry.clips.get(clip) {
                None => dangling(report, owner.clone(), "clips", clip.to_string()),
                Some(def) if !game_core::content::timing_suits(state, def.timing) => {
                    report.problems.push(Problem::Invalid {
                        path: PathBuf::from("<registry>"),
                        id: owner.clone(),
                        source: ContentError::ClipTimingMismatch {
                            clip: clip.clone(),
                            state,
                        },
                    });
                }
                Some(_) => {}
            }
        }
    };
    for plan in registry.body_plans.values() {
        check_set(report, plan.id.to_string(), &plan.animations);
    }
    for model in registry.models.values() {
        check_set(report, model.id.to_string(), &model.animations);
    }
    for clip in registry.clips.values() {
        for track in &clip.tracks {
            if !known_slots.contains(&track.slot) {
                dangling(report, clip.id.to_string(), "slots", track.slot.to_string());
            }
        }
    }

    for recipe in registry.recipes.values() {
        for input in recipe.inputs.keys() {
            if !registry.items.contains_key(input) {
                dangling(report, recipe.id.to_string(), "items", input.to_string());
            }
        }
        if !registry.items.contains_key(&recipe.output) {
            dangling(report, recipe.id.to_string(), "items", recipe.output.to_string());
        }
        if let Some(upgrade) = &recipe.requires_upgrade
            && !registry.upgrades.contains_key(upgrade)
        {
            dangling(report, recipe.id.to_string(), "upgrades", upgrade.to_string());
        }
    }

    // An effect names its palette, and every part a swatch in it.
    for vfx in registry.vfx.values() {
        let Some(palette) = registry.palettes.get(&vfx.palette) else {
            dangling(report, vfx.id.to_string(), "palettes", vfx.palette.to_string());
            continue;
        };
        for part in &vfx.parts {
            for swatch in part.swatches() {
                if !palette.swatches.contains_key(swatch) {
                    dangling(
                        report,
                        vfx.id.to_string(),
                        "swatches",
                        format!("{}/{}", palette.id, swatch),
                    );
                }
            }
        }
    }

    // A status is a colour and a meaning (M9): its chip's swatch must be in
    // its palette, and an effect it plays must exist.
    for status in registry.statuses.values() {
        let owner = status.id.0.to_string();
        match registry.palettes.get(&status.palette) {
            None => dangling(report, owner.clone(), "palettes", status.palette.to_string()),
            Some(palette) if !palette.swatches.contains_key(&status.color) => dangling(
                report,
                owner.clone(),
                "swatches",
                format!("{}/{}", palette.id, status.color),
            ),
            Some(_) => {}
        }
        if let Some(vfx) = &status.vfx
            && !registry.vfx.contains_key(vfx)
        {
            dangling(report, owner, "vfx", vfx.to_string());
        }
    }

    for ability in registry.abilities.values() {
        // Every effect, including those a projectile carries to its target.
        let mut effects: Vec<&Effect> = ability.effects.iter().collect();
        let mut index = 0;
        while let Some(effect) = effects.get(index).copied() {
            if let Effect::Projectile { on_hit, .. } = effect {
                effects.extend(on_hit);
            }
            index += 1;
        }
        for effect in effects {
            match effect {
                Effect::SpawnEntity { prefab, .. } if !registry.enemies.contains_key(prefab) => {
                    dangling(report, ability.id.to_string(), "enemies", prefab.to_string());
                }
                // Until M9 an unknown status applied and meant nothing. Now it
                // does not load.
                Effect::ApplyStatus { status, .. } if !registry.statuses.contains_key(status) => {
                    dangling(report, ability.id.to_string(), "statuses", status.0.to_string());
                }
                _ => {}
            }
        }
        // An ability's own clip stands in for its caster's `cast` clip, so it
        // must run on the cast's clock.
        let look = &ability.presentation;
        if let Some(clip) = &look.clip {
            match registry.clips.get(clip) {
                None => dangling(report, ability.id.to_string(), "clips", clip.to_string()),
                Some(def) if !game_core::content::timing_suits("cast", def.timing) => {
                    report.problems.push(Problem::Invalid {
                        path: PathBuf::from("<registry>"),
                        id: ability.id.to_string(),
                        source: ContentError::ClipTimingMismatch {
                            clip: clip.clone(),
                            state: "cast",
                        },
                    });
                }
                Some(_) => {}
            }
        }
        for vfx in [&look.cast, &look.impact, &look.trail].into_iter().flatten() {
            if !registry.vfx.contains_key(vfx) {
                dangling(report, ability.id.to_string(), "vfx", vfx.to_string());
            }
        }
        if let Some(model) = &look.projectile
            && !registry.models.contains_key(model)
        {
            dangling(report, ability.id.to_string(), "models", model.to_string());
        }
    }

    // What a consumable does when used names statuses too (M9).
    for item in registry.items.values() {
        for effect in item.on_use.iter().flat_map(|used| &used.effects) {
            if let Effect::ApplyStatus { status, .. } = effect
                && !registry.statuses.contains_key(status)
            {
                dangling(report, item.id.to_string(), "statuses", status.0.to_string());
            }
        }
    }

    for class in registry.classes.values() {
        if let Some(curve) = &class.level_curve
            && !registry.level_curves.contains_key(curve)
        {
            dangling(report, class.id.to_string(), "level-curves", curve.to_string());
        }
        for ability in &class.abilities {
            if !registry.abilities.contains_key(ability) {
                dangling(report, class.id.to_string(), "abilities", ability.to_string());
            }
        }
        if let Some(model) = &class.model
            && !registry.models.contains_key(model)
        {
            dangling(report, class.id.to_string(), "models", model.to_string());
        }
        for vfx in [&class.presentation.swing, &class.presentation.hit]
            .into_iter()
            .flatten()
        {
            if !registry.vfx.contains_key(vfx) {
                dangling(report, class.id.to_string(), "vfx", vfx.to_string());
            }
        }
    }

    for enemy in registry.enemies.values() {
        for ability in &enemy.abilities {
            if !registry.abilities.contains_key(ability) {
                dangling(report, enemy.id.to_string(), "abilities", ability.to_string());
            }
        }
        if let Some(parent) = &enemy.variant_of
            && !registry.enemies.contains_key(parent)
        {
            dangling(report, enemy.id.to_string(), "enemies", parent.to_string());
        }
        // A misspelled loot table would otherwise mean an enemy that silently
        // drops nothing — the kind of content bug nobody notices until a player
        // asks why the elite gave them no reward.
        if let Some(table) = &enemy.loot_table
            && !registry.loot_tables.contains_key(table)
        {
            dangling(report, enemy.id.to_string(), "loot-tables", table.to_string());
        }
        // A misspelled model would otherwise be a silent fallback to the
        // block-out cube: the enemy still plays, and looks wrong for no reason
        // anyone can see.
        if let Some(model) = &enemy.model
            && !registry.models.contains_key(model)
        {
            dangling(report, enemy.id.to_string(), "models", model.to_string());
        }
        for vfx in [&enemy.presentation.swing, &enemy.presentation.hit]
            .into_iter()
            .flatten()
        {
            if !registry.vfx.contains_key(vfx) {
                dangling(report, enemy.id.to_string(), "vfx", vfx.to_string());
            }
        }
    }

    // What every encounter's placements actually contain, across every
    // location: per enemy, how many; and which waves they arrive in.
    let mut placed: BTreeMap<&EncounterId, BTreeMap<&PrefabId, u32>> = BTreeMap::new();
    let mut waves: BTreeMap<&EncounterId, std::collections::BTreeSet<u8>> = BTreeMap::new();
    for location in registry.locations.values() {
        for spawn in &location.spawns {
            if let SpawnRole::Enemy {
                prefab,
                encounter: Some(encounter),
                wave,
            } = &spawn.role
            {
                *placed.entry(encounter).or_default().entry(prefab).or_default() += 1;
                waves.entry(encounter).or_default().insert(*wave);
            }
        }
    }
    let invalid = |report: &mut ContentReport, id: String, source: ContentError| {
        report.problems.push(Problem::Invalid {
            path: PathBuf::from("<registry>"),
            id,
            source,
        });
    };

    for encounter in registry.encounters.values() {
        for slot in &encounter.slots {
            if !registry.enemies.contains_key(&slot.enemy) {
                dangling(
                    report,
                    encounter.id.to_string(),
                    "enemies",
                    slot.enemy.to_string(),
                );
            }
        }
        if let Some(arrival) = &encounter.arrival
            && !registry.vfx.contains_key(arrival)
        {
            dangling(report, encounter.id.to_string(), "vfx", arrival.to_string());
        }

        // The budget binds (M8): what the slots cost, at each enemy's own
        // threat, must fit what the encounter declares. M11 generates against
        // this number, so it has to mean the sum it claims to bound.
        let total: i64 = encounter
            .slots
            .iter()
            .filter_map(|slot| {
                registry
                    .enemies
                    .get(&slot.enemy)
                    .map(|enemy| enemy.threat.as_milli() * i64::from(slot.count))
            })
            .sum();
        if total > encounter.threat_budget.as_milli() {
            invalid(
                report,
                encounter.id.to_string(),
                ContentError::OverThreatBudget {
                    encounter: encounter.id.clone(),
                    total: game_core::content::Fixed::from_milli(total),
                    budget: encounter.threat_budget,
                },
            );
        }

        // And the slots describe the placements: an encounter whose slots
        // say three husks while four stand in the world has a budget for a
        // fight that is not the one the player meets.
        let mut listed: BTreeMap<&PrefabId, u32> = BTreeMap::new();
        for slot in &encounter.slots {
            *listed.entry(&slot.enemy).or_default() += u32::from(slot.count);
        }
        let empty = BTreeMap::new();
        let here = placed.get(&encounter.id).unwrap_or(&empty);
        let mut enemies: std::collections::BTreeSet<&PrefabId> = listed.keys().copied().collect();
        enemies.extend(here.keys().copied());
        for enemy in enemies {
            let (slots, count) = (
                listed.get(enemy).copied().unwrap_or(0),
                here.get(enemy).copied().unwrap_or(0),
            );
            if slots != count {
                invalid(
                    report,
                    encounter.id.to_string(),
                    ContentError::PlacementsDisagree {
                        encounter: encounter.id.clone(),
                        enemy: enemy.clone(),
                        slots,
                        placed: count,
                    },
                );
            }
        }

        // Waves run 0, 1, 2 … with no gap: a wave after a gap would wait for
        // an empty wave to be cleared, and never come.
        if let Some(seen) = waves.get(&encounter.id) {
            for (expected, wave) in seen.iter().enumerate() {
                let expected = u8::try_from(expected).unwrap_or(u8::MAX);
                if *wave != expected || *wave >= game_core::content::MAX_WAVES {
                    invalid(
                        report,
                        encounter.id.to_string(),
                        ContentError::WaveGap {
                            encounter: encounter.id.clone(),
                            wave: *wave,
                        },
                    );
                    break;
                }
            }
        }
    }

    for quest in registry.quests.values() {
        for node in quest.nodes.values() {
            match &node.objective {
                Objective::Defeat { encounter } => {
                    if !registry.encounters.contains_key(encounter) {
                        dangling(report, quest.id.to_string(), "encounters", encounter.to_string());
                    }
                }
                Objective::Collect { item, .. } => {
                    if !registry.items.contains_key(item) {
                        dangling(report, quest.id.to_string(), "items", item.to_string());
                    }
                }
                // Locations became a content family in M5, so a quest that
                // sends the player somewhere can finally be checked. Without
                // this a typo is an objective that waits forever for an
                // arrival that can never happen.
                Objective::Reach { location } => {
                    if !registry.locations.contains_key(location) {
                        dangling(report, quest.id.to_string(), "locations", location.to_string());
                    }
                }
                Objective::Escort { destination, .. } => {
                    if !registry.locations.contains_key(destination) {
                        dangling(report, quest.id.to_string(), "locations", destination.to_string());
                    }
                }
                // NPCs, clues, and choices are still world-state and narrative
                // ids rather than content families.
                Objective::TalkTo { .. } | Objective::Investigate { .. } | Objective::Decide { .. } => {}
            }
        }
        for item in quest.rewards.items.keys() {
            if !registry.items.contains_key(item) {
                dangling(report, quest.id.to_string(), "items", item.to_string());
            }
        }
    }

    for location in registry.locations.values() {
        // Every location belongs to a region content defines (M9).
        if !registry.regions.contains_key(&location.region) {
            dangling(
                report,
                location.id.to_string(),
                "regions",
                location.region.to_string(),
            );
        }
        for spawn in &location.spawns {
            match &spawn.role {
                SpawnRole::Enemy {
                    prefab, encounter, ..
                } => {
                    if !registry.enemies.contains_key(prefab) {
                        dangling(report, location.id.to_string(), "enemies", prefab.to_string());
                    }
                    // A placement tagged with an encounter that does not exist
                    // is worse than an untagged one: a `Defeat` objective would
                    // wait forever for a fight nothing belongs to.
                    if let Some(encounter) = encounter
                        && !registry.encounters.contains_key(encounter)
                    {
                        dangling(
                            report,
                            location.id.to_string(),
                            "encounters",
                            encounter.to_string(),
                        );
                    }
                }
                SpawnRole::Gatherable { item } => {
                    if !registry.items.contains_key(item) {
                        dangling(report, location.id.to_string(), "items", item.to_string());
                    }
                }
                // NPCs themselves are world-state ids rather than a content
                // family, so there is nothing to resolve them against yet —
                // but the model a placement dresses one in is content.
                SpawnRole::Npc {
                    model: Some(model), ..
                } if !registry.models.contains_key(model) => {
                    dangling(report, location.id.to_string(), "models", model.to_string());
                }
                // Clues and choices are narrative ids rather than content
                // families, so there is no registry to resolve them against.
                SpawnRole::PlayerStart
                | SpawnRole::Npc { .. }
                | SpawnRole::Station { .. }
                | SpawnRole::Clue { .. }
                | SpawnRole::Choice { .. } => {}
            }
        }
        for link in &location.links {
            // A door has to *be* something. The spawn a link leads from must
            // exist here and be a station, because that is what becomes an
            // entity the player can use; a link pointing at a `PlayerStart`
            // leaves a location with no way out and nothing to say so.
            match location.spawn(&link.from) {
                Some(spawn)
                    if matches!(
                        spawn.role,
                        SpawnRole::Station {
                            kind: StationKind::Doorway
                        }
                    ) => {}
                _ => dangling(report, location.id.to_string(), "stations", link.from.to_string()),
            }
            match registry.locations.get(&link.to) {
                // A door must open onto somewhere you can actually stand.
                Some(destination) => {
                    if destination.spawn(&link.arrive_at).is_none() {
                        dangling(
                            report,
                            location.id.to_string(),
                            "locations",
                            format!("{}#{}", link.to, link.arrive_at),
                        );
                    }
                }
                None => dangling(report, location.id.to_string(), "locations", link.to.to_string()),
            }
        }
    }

    for dialogue in registry.dialogues.values() {
        for node in dialogue.nodes.values() {
            for option in &node.options {
                for effect in &option.effects {
                    match effect {
                        DialogueEffect::StartQuest(quest)
                        | DialogueEffect::CompleteObjective { quest, .. } => {
                            if !registry.quests.contains_key(quest) {
                                dangling(report, dialogue.id.to_string(), "quests", quest.to_string());
                            }
                        }
                        DialogueEffect::GrantItem { item, .. } | DialogueEffect::TakeItem { item, .. } => {
                            if !registry.items.contains_key(item) {
                                dangling(report, dialogue.id.to_string(), "items", item.to_string());
                            }
                        }
                        // Flags, factions, and choices are world-state and
                        // narrative ids, not content families.
                        DialogueEffect::SetWorldFlag(_)
                        | DialogueEffect::AdjustReputation { .. }
                        | DialogueEffect::MakeChoice(_) => {}
                    }
                }
            }
        }
    }

    // One NPC may own several conversations (M9). When more than one is open
    // the player picks by topic, so a speaker with several must title each.
    let mut by_speaker: BTreeMap<&game_core::ids::NpcId, Vec<&DialogueDef>> = BTreeMap::new();
    for dialogue in registry.dialogues.values() {
        by_speaker.entry(&dialogue.speaker).or_default().push(dialogue);
    }
    for dialogues in by_speaker.values().filter(|dialogues| dialogues.len() > 1) {
        for dialogue in dialogues.iter().filter(|dialogue| dialogue.topic.is_none()) {
            invalid(
                report,
                dialogue.id.to_string(),
                ContentError::UntitledConversation {
                    speaker: dialogue.speaker.clone(),
                    dialogue: dialogue.id.clone(),
                },
            );
        }
    }

    for table in registry.loot_tables.values() {
        for entry in &table.entries {
            if !registry.items.contains_key(&entry.item) {
                dangling(report, table.id.to_string(), "items", entry.item.to_string());
            }
        }
    }

    for upgrade in registry.upgrades.values() {
        for item in upgrade.costs.keys() {
            if !registry.items.contains_key(item) {
                dangling(report, upgrade.id.to_string(), "items", item.to_string());
            }
        }
        for required in &upgrade.requires {
            if !registry.upgrades.contains_key(required) {
                dangling(report, upgrade.id.to_string(), "upgrades", required.to_string());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/content")
    }

    fn invalid_fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/content-invalid")
    }

    #[test]
    fn shipped_content_loads_and_cross_checks() {
        let registry = match load(&fixtures()) {
            Ok(registry) => registry,
            Err(report) => panic!("shipped fixtures should be valid, got:\n{report}"),
        };
        assert!(registry.items.len() >= 5, "five gatherables plus crafted goods");
        assert_eq!(registry.recipes.len(), 3);
        assert_eq!(registry.enemies.len(), 4);
        assert_eq!(registry.quests.len(), 2, "the waterworks and the crypt");
        assert!(!registry.is_empty());

        // The M5 families. Asserted explicitly because a loader that silently
        // read none of them would otherwise pass every test above.
        assert_eq!(registry.locations.len(), 3, "hub, lane, and dungeon");
        assert_eq!(registry.dialogues.len(), 2);
        assert_eq!(registry.loot_tables.len(), 3);
        assert_eq!(registry.upgrades.len(), 2);
        assert_eq!(registry.classes.len(), 2, "one tuned class and one defined");
        assert_eq!(
            registry.len(),
            registry.items.len()
                + registry.recipes.len()
                + registry.abilities.len()
                + registry.classes.len()
                + registry.enemies.len()
                + registry.encounters.len()
                + registry.quests.len()
                + registry.locations.len()
                + registry.dialogues.len()
                + registry.loot_tables.len()
                + registry.upgrades.len()
                + registry.body_plans.len()
                + registry.palettes.len()
                + registry.models.len()
                + registry.clips.len()
                + registry.vfx.len()
                + registry.statuses.len()
                + registry.regions.len()
                + registry.level_curves.len(),
            "the reported total must account for every family"
        );
        assert_eq!(
            registry.statuses.len(),
            4,
            "guarded, mired, smouldering, refreshed (M9)"
        );
    }

    /// A save whose recorded quest step no longer exists is refused (M9):
    /// the quest could never advance past it.
    #[test]
    fn a_vanished_quest_step_is_an_orphan() {
        let content = load(&fixtures()).expect("shipped content loads");
        let mut world = game_core::world::WorldState::new(
            1,
            game_core::ids::ContentRevisionId::new("rev").expect("valid id"),
        );
        let quest = QuestId::new("still-waters").expect("valid id");
        let mut progress = game_core::world::QuestState::default();
        progress.objectives.insert(
            game_core::ids::ObjectiveId::new("meet-sela").expect("valid id"),
            Default::default(),
        );
        world.quests.insert(quest.clone(), progress.clone());
        assert!(verify_world_references(&content, &world, Path::new("save")).is_empty());

        progress.objectives.insert(
            game_core::ids::ObjectiveId::new("a-step-since-deleted").expect("valid id"),
            Default::default(),
        );
        world.quests.insert(quest, progress);
        let report = verify_world_references(&content, &world, Path::new("save"));
        assert!(
            report.to_string().contains("still-waters#a-step-since-deleted"),
            "{report}"
        );
    }

    /// Every location's region is defined (M9).
    #[test]
    fn a_location_in_an_undefined_region_does_not_load() {
        let shipped = load(&fixtures()).expect("shipped content loads");
        assert_eq!(shipped.regions.len(), 2, "cinderwake and the ashfall waste");
        let mut lost = shipped.clone();
        lost.regions
            .remove(&game_core::ids::RegionId::new("ashfall-waste").expect("valid id"));
        let mut report = ContentReport::default();
        resolve_references(&lost, &mut report);
        let text = report.to_string();
        assert!(
            text.contains("ashfall-lane references regions ashfall-waste")
                && text.contains("crypt-of-ash references regions ashfall-waste"),
            "{text}"
        );
    }

    /// Every status an ability applies is defined, and a status names a real
    /// swatch and effect (M9). Before, an unknown status loaded and did nothing.
    #[test]
    fn statuses_resolve_and_an_undefined_one_does_not_load() {
        let shipped = load(&fixtures()).expect("shipped content loads");
        let reported = |registry: &ContentRegistry| {
            let mut report = ContentReport::default();
            resolve_references(registry, &mut report);
            report.to_string()
        };
        assert_eq!(reported(&shipped), "");

        // Applied directly by an ability, and carried by a projectile.
        let mut undefined = shipped.clone();
        undefined
            .statuses
            .remove(&StatusId(AbilityId::new("status-guarded").expect("valid id")));
        undefined
            .statuses
            .remove(&StatusId(AbilityId::new("status-smouldering").expect("valid id")));
        let text = reported(&undefined);
        assert!(
            text.contains("warden-guard references statuses status-guarded")
                && text.contains("ember-bolt references statuses status-smouldering"),
            "{text}"
        );

        // And named by what a consumable does when used.
        let mut undrinkable = shipped.clone();
        undrinkable
            .statuses
            .remove(&StatusId(AbilityId::new("status-refreshed").expect("valid id")));
        let text = reported(&undrinkable);
        assert!(
            text.contains("clearwater-draught references statuses status-refreshed"),
            "{text}"
        );

        let mut unpainted = shipped.clone();
        if let Some(status) = unpainted.statuses.values_mut().next() {
            status.color = game_core::ids::SwatchId::new("no-such-colour").expect("valid id");
            status.vfx = Some(VfxId::new("no-such-effect").expect("valid id"));
        }
        let text = reported(&unpainted);
        assert!(
            text.contains("no-such-colour") && text.contains("no-such-effect"),
            "{text}"
        );
    }

    /// An encounter's budget binds, its slots describe its placements, and its
    /// waves run without a gap (M8).
    #[test]
    fn encounters_are_honest_about_what_they_hold() {
        let shipped = match load(&fixtures()) {
            Ok(registry) => registry,
            Err(report) => panic!("shipped fixtures should be valid, got:\n{report}"),
        };
        let ambush = EncounterId::new("waterworks-ambush").expect("valid id");
        let reported = |registry: &ContentRegistry| {
            let mut report = ContentReport::default();
            resolve_references(registry, &mut report);
            report
        };
        let complains = |registry: &ContentRegistry, check: &dyn Fn(&ContentError) -> bool| {
            reported(registry)
                .problems
                .iter()
                .any(|problem| matches!(problem, Problem::Invalid { source, .. } if check(source)))
        };
        assert!(reported(&shipped).is_empty(), "the shipped encounters are honest");

        // The waterworks slots cost 4200: a 4000 budget, as first authored, is
        // refused now that the number binds.
        let mut tight = shipped.clone();
        if let Some(encounter) = tight.encounters.get_mut(&ambush) {
            encounter.threat_budget = game_core::content::Fixed::from_whole(4);
        }
        assert!(complains(&tight, &|error| matches!(
            error,
            ContentError::OverThreatBudget { total, .. } if total.as_milli() == 4_200
        )));

        // A slot the placements do not match.
        let mut miscounted = shipped.clone();
        if let Some(encounter) = miscounted.encounters.get_mut(&ambush) {
            encounter.slots[0].count = 2;
        }
        assert!(complains(&miscounted, &|error| matches!(
            error,
            ContentError::PlacementsDisagree {
                slots: 2,
                placed: 3,
                ..
            }
        )));

        // A wave after a gap could never arrive.
        let mut gapped = shipped.clone();
        let lane = LocationId::new("ashfall-lane").expect("valid id");
        if let Some(location) = gapped.locations.get_mut(&lane) {
            for spawn in &mut location.spawns {
                if let SpawnRole::Enemy { wave, .. } = &mut spawn.role
                    && *wave == 1
                {
                    *wave = 2;
                }
            }
        }
        assert!(complains(&gapped, &|error| matches!(
            error,
            ContentError::WaveGap { wave: 2, .. }
        )));
    }

    /// An effect's palette and swatches, and an ability's clip and effects,
    /// all resolve — and a wrong one is named rather than drawn as nothing.
    #[test]
    fn effects_and_ability_presentation_resolve() {
        let shipped = match load(&fixtures()) {
            Ok(registry) => registry,
            Err(report) => panic!("shipped fixtures should be valid, got:\n{report}"),
        };
        // By name rather than by count: the point is that the loader read the
        // family, and a count breaks on every effect authored after it.
        for name in [
            "ember-lash",
            "guard-shimmer",
            "husk-summoning",
            "ember-trail",
            "resin-splash",
        ] {
            assert!(
                shipped.vfx.contains_key(&VfxId::new(name).expect("valid id")),
                "{name} is loaded"
            );
        }
        let cleave = AbilityId::new("ember-cleave").expect("valid id");
        let reported = |registry: &ContentRegistry| {
            let mut report = ContentReport::default();
            resolve_references(registry, &mut report);
            report
        };

        let mut unknown = shipped.clone();
        if let Some(ability) = unknown.abilities.get_mut(&cleave) {
            ability.presentation.impact = Some(VfxId::new("ember-lsah").expect("valid id"));
        }
        let report = reported(&unknown);
        assert!(
            report.problems.iter().any(|problem| matches!(
                problem,
                Problem::Dangling { to_family: "vfx", to, .. } if to == "ember-lsah"
            )),
            "a misspelt effect is named:\n{report}"
        );

        // A bolt's look names a model and a trail; both must exist.
        let bolt = AbilityId::new("ember-bolt").expect("valid id");
        let mut lost = shipped.clone();
        if let Some(ability) = lost.abilities.get_mut(&bolt) {
            ability.presentation.projectile = Some(ModelId::new("ember-blot").expect("valid id"));
            ability.presentation.trail = Some(VfxId::new("ember-trial").expect("valid id"));
        }
        let report = reported(&lost);
        for (family, name) in [("models", "ember-blot"), ("vfx", "ember-trial")] {
            assert!(
                report.problems.iter().any(|problem| matches!(
                    problem,
                    Problem::Dangling { to_family, to, .. } if *to_family == family && to == name
                )),
                "a misspelt {family} reference is named:\n{report}"
            );
        }

        let mut unpainted = shipped.clone();
        if let Some(effect) = unpainted.vfx.values_mut().next() {
            effect.parts[0] = game_core::content::VfxPart::Pulse {
                color: game_core::ids::SwatchId::new("no-such-colour").expect("valid id"),
                radius: game_core::content::Fixed::from_whole(1),
                ticks: 10,
            };
        }
        let report = reported(&unpainted);
        assert!(
            report.to_string().contains("no-such-colour"),
            "a swatch the palette lacks is named:\n{report}"
        );

        // A spire's glow is a swatch too, and resolved like its colour.
        let mut unlit = shipped.clone();
        if let Some(quake) = unlit.vfx.get_mut(&VfxId::new("ash-quake").expect("valid id"))
            && let Some(game_core::content::VfxPart::Spire { glow, .. }) = quake
                .parts
                .iter_mut()
                .find(|part| matches!(part, game_core::content::VfxPart::Spire { .. }))
        {
            *glow = Some(game_core::ids::SwatchId::new("no-such-glow").expect("valid id"));
        }
        let report = reported(&unlit);
        assert!(
            report.to_string().contains("no-such-glow"),
            "a spire's glow the palette lacks is named:\n{report}"
        );

        // An idle loop never runs on a cast's clock, so as an ability's clip
        // it would sit frozen through the cast.
        let mut frozen = shipped.clone();
        if let Some(ability) = frozen.abilities.get_mut(&cleave) {
            ability.presentation.clip = Some(ClipId::new("humanoid-idle").expect("valid id"));
        }
        let report = reported(&frozen);
        assert!(
            report.problems.iter().any(|problem| matches!(
                problem,
                Problem::Invalid {
                    source: ContentError::ClipTimingMismatch { state: "cast", .. },
                    ..
                }
            )),
            "a looping clip is refused as a cast:\n{report}"
        );
    }

    /// A clip used for a state whose clock it does not run on would sit
    /// frozen; a track for a slot no body has is a typo. Both are caught.
    #[test]
    fn animation_sets_and_clip_slots_resolve() {
        let plan: BodyPlanDef = ron::from_str(
            r#"(schema_version: 1, id: "humanoid", name: "Humanoid",
                slots: [(name: "leg-l", pivot: (x: 0, y: 900, z: 0))],
                animations: (walk: Some("walk")))"#,
        )
        .expect("parses");
        let walk: ClipDef = ron::from_str(
            r#"(schema_version: 1, id: "walk", name: "Walk", timing: Stride(metres: 1500),
                tracks: [(slot: "leg-l", keys: [(at: 0), (at: 500, turn: (x: 20000, y: 0, z: 0))])])"#,
        )
        .expect("parses");
        let mut registry = ContentRegistry::default();
        registry.body_plans.insert(plan.id.clone(), plan.clone());
        registry.clips.insert(walk.id.clone(), walk.clone());

        let mut clean = ContentReport::default();
        resolve_references(&registry, &mut clean);
        assert!(clean.is_empty(), "a matching clip resolves:\n{clean}");

        let mut frozen = registry.clone();
        let mut windows = walk.clone();
        windows.timing = game_core::content::ClipTiming::Windows;
        frozen.clips.insert(windows.id.clone(), windows);
        let mut report = ContentReport::default();
        resolve_references(&frozen, &mut report);
        assert!(
            report.problems.iter().any(|problem| matches!(
                problem,
                Problem::Invalid {
                    source: ContentError::ClipTimingMismatch { state: "walk", .. },
                    ..
                }
            )),
            "a walk keyed in attack windows is refused:\n{report}"
        );

        let mut typo = registry.clone();
        let mut misspelt = walk.clone();
        misspelt.tracks[0].slot = game_core::ids::SlotId::new("lef-l").expect("valid id");
        typo.clips.insert(misspelt.id.clone(), misspelt);
        let mut report = ContentReport::default();
        resolve_references(&typo, &mut report);
        assert!(
            report.to_string().contains("lef-l"),
            "a slot no plan has is named:\n{report}"
        );

        let mut missing = registry.clone();
        missing.clips.clear();
        let mut report = ContentReport::default();
        resolve_references(&missing, &mut report);
        assert!(
            report.to_string().contains("walk"),
            "a set naming no clip is dangling:\n{report}"
        );
    }

    /// Every way a model can point at something that is not there, caught
    /// before the game starts rather than as a silent cube in the middle of a
    /// fight.
    #[test]
    fn model_references_resolve_or_are_reported() {
        let plan: BodyPlanDef = ron::from_str(
            r#"(schema_version: 1, id: "humanoid", name: "Humanoid", slots: [(name: "torso", pivot: (x: 0, y: 900, z: 0))])"#,
        )
        .expect("parses");
        let palette: PaletteDef = ron::from_str(
            r#"(schema_version: 1, id: "ash", name: "Ash", swatches: { "grey": (90, 90, 90) })"#,
        )
        .expect("parses");
        let model: ModelDef = ron::from_str(
            r#"(schema_version: 1, id: "husk", name: "Husk", body_plan: "humanoid", palette: "ash",
                parts: [(slot: "torso", shape: Box(x: 400, y: 600, z: 300), color: "grey")])"#,
        )
        .expect("parses");

        let mut registry = ContentRegistry::default();
        registry.body_plans.insert(plan.id.clone(), plan);
        registry.palettes.insert(palette.id.clone(), palette);
        registry.models.insert(model.id.clone(), model.clone());

        let mut clean = ContentReport::default();
        resolve_references(&registry, &mut clean);
        assert!(clean.is_empty(), "a complete model resolves:\n{clean}");

        let broken = |change: &dyn Fn(&mut ModelDef)| {
            let mut registry = registry.clone();
            let mut model = model.clone();
            change(&mut model);
            registry.models.insert(model.id.clone(), model);
            let mut report = ContentReport::default();
            resolve_references(&registry, &mut report);
            report.to_string()
        };
        let body = |text: &str| game_core::ids::BodyPlanId::new(text).expect("valid id");
        let slot = |text: &str| game_core::ids::SlotId::new(text).expect("valid id");
        let swatch = |text: &str| game_core::ids::SwatchId::new(text).expect("valid id");

        assert!(broken(&|m| m.body_plan = body("quadruped")).contains("quadruped"));
        assert!(broken(&|m| m.palette = PaletteId::new("bright").expect("valid id")).contains("bright"));
        assert!(broken(&|m| m.parts[0].slot = slot("tail")).contains("humanoid/tail"));
        assert!(broken(&|m| m.parts[0].color = swatch("gold")).contains("ash/gold"));
        assert!(
            broken(&|m| m.parts[0].top = Some(swatch("gold"))).contains("ash/gold"),
            "an override colour is checked like the main one"
        );
    }

    #[test]
    fn the_same_content_always_hashes_the_same() {
        let first = load(&fixtures()).expect("valid").content_hash().expect("hashes");
        let second = load(&fixtures()).expect("valid").content_hash().expect("hashes");
        assert_eq!(
            first, second,
            "a content hash must not depend on load order or timing"
        );
        assert_ne!(
            first,
            ContentRegistry::default().content_hash().expect("hashes"),
            "and must distinguish real content from none"
        );
    }

    #[test]
    fn changing_a_value_changes_the_hash() {
        let mut registry = load(&fixtures()).expect("valid");
        let before = registry.content_hash().expect("hashes");

        let key = registry.items.keys().next().expect("fixtures have items").clone();
        let item = registry.items.get_mut(&key).expect("present");
        item.stack_limit += 1;

        assert_ne!(
            before,
            registry.content_hash().expect("hashes"),
            "an altered definition must produce a different revision"
        );
    }

    /// Length-prefixing is what stops two different content sets from
    /// concatenating into identical bytes.
    #[test]
    fn ids_cannot_be_confused_by_concatenation() {
        use game_core::content::ItemCategory;

        let build = |first: &str, second: &str| {
            let mut registry = ContentRegistry::default();
            for id in [first, second] {
                let id = ItemId::new(id).expect("valid id");
                registry.items.insert(
                    id.clone(),
                    ItemDef {
                        schema_version: game_core::content::CONTENT_SCHEMA_VERSION,
                        id,
                        name: String::new(),
                        category: ItemCategory::Material,
                        stack_limit: 1,
                        combat: None,
                        on_use: None,
                    },
                );
            }
            registry.content_hash().expect("hashes")
        };

        assert_ne!(build("ab", "c"), build("a", "bc"));
    }

    #[test]
    fn loading_is_deterministic() {
        let first = load(&fixtures()).expect("valid");
        let second = load(&fixtures()).expect("valid");
        assert_eq!(first, second);
    }

    #[test]
    fn dangling_references_name_both_ends() {
        let report = load(&invalid_fixtures()).expect_err("the broken fixture must fail");
        let rendered = report.to_string();
        assert!(
            report
                .problems
                .iter()
                .any(|problem| matches!(problem, Problem::Dangling { .. })),
            "expected a dangling reference, got:\n{rendered}"
        );
        assert!(
            rendered.contains("ghost-quest") && rendered.contains("no-such-encounter"),
            "the message must name the referrer and the missing target:\n{rendered}"
        );
    }

    #[test]
    fn a_speaker_with_several_conversations_must_title_each() {
        let shipped = load(&fixtures()).expect("shipped content loads");
        let untitled = |registry: &ContentRegistry| {
            let mut report = ContentReport::default();
            resolve_references(registry, &mut report);
            let mut named: Vec<String> = report
                .problems
                .iter()
                .filter_map(|problem| match problem {
                    Problem::Invalid {
                        source: ContentError::UntitledConversation { dialogue, .. },
                        ..
                    } => Some(dialogue.to_string()),
                    _ => None,
                })
                .collect();
            named.sort();
            named
        };
        assert_eq!(
            untitled(&shipped),
            Vec::<String>::new(),
            "one each needs no topic"
        );

        let mut two = shipped.clone();
        let greeting = two
            .dialogues
            .get(&DialogueId::new("sela-greeting").expect("valid id"))
            .cloned()
            .expect("shipped");
        let mut second = greeting.clone();
        second.id = DialogueId::new("sela-second").expect("valid id");
        second.topic = Some("Something else".to_owned());
        two.dialogues.insert(second.id.clone(), second);
        assert_eq!(
            untitled(&two),
            vec!["sela-greeting"],
            "only the untitled one is named"
        );

        if let Some(greeting) = two.dialogues.get_mut(&greeting.id) {
            greeting.topic = Some("The intake".to_owned());
        }
        assert_eq!(untitled(&two), Vec::<String>::new());
    }

    #[test]
    fn a_missing_root_is_reported_rather_than_panicking() {
        let report = load(Path::new("/nonexistent/content/root")).expect_err("must fail");
        assert!(
            report
                .problems
                .iter()
                .any(|problem| matches!(problem, Problem::Unreadable { .. })),
            "a missing content root must be reported, not silently treated as empty:\n{report}"
        );
    }

    #[test]
    fn every_family_name_is_kebab_case() {
        for family in FAMILIES {
            assert!(
                family
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte == b'-'),
                "{family} should be lowercase kebab-case"
            );
        }
    }
}
