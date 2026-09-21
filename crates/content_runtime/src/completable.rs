//! Whether every quest can actually be finished (M9 wave 1).
//!
//! Reference resolution proves a quest names things that exist. That is not
//! the same as a quest a player can finish: a `TalkTo` whose NPC stands
//! nowhere, a clue nobody placed, a choice nothing offers, or a destination no
//! door leads to all resolve — or name vocabularies with no registry at all —
//! and still leave the player stuck forever. Content that will later come from
//! a model (ADR-0010) must fail *here*, at load, rather than in someone's
//! save.
//!
//! # A static proof, not a playthrough
//!
//! Every check is a lookup over the registry; nothing simulates. Where a proof
//! would need to reason about conditions — a dialogue line gated on a flag
//! that something may or may not set — the check accepts rather than guesses.
//! A refused quest is always genuinely unfinishable; a quest that passes may
//! still hide a condition nothing satisfies, which is a narrower hole than the
//! one this closes.
//!
//! # Where a quest is played from
//!
//! A quest is started by a conversation, and conversations happen where their
//! speaker stands. Everything a quest asks for must be reachable, through
//! doors, from somewhere it is given — otherwise the objective exists but the
//! way to it does not.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use game_core::content::{DialogueEffect, Objective, SpawnRole};
use game_core::ids::{ChoiceId, ClueId, EncounterId, ItemId, LocationId, NpcId, ObjectiveId, QuestId};

use crate::{ContentRegistry, ContentReport, Problem};

/// Why a quest objective can never be completed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unfinishable {
    /// No conversation starts the quest, or none whose speaker stands anywhere.
    NeverStarted,
    /// The NPC the objective needs is placed in no location.
    NobodyThere(NpcId),
    /// The NPC stands somewhere but has no conversation to open.
    NothingToSay(NpcId),
    /// The clue is placed in no location.
    ClueNotPlaced(ClueId),
    /// No marker settles the choice, and no placed speaker offers it.
    ChoiceNotOffered(ChoiceId),
    /// No enemy placement belongs to the encounter.
    EncounterNotPlaced(EncounterId),
    /// Nothing produces the item: no gatherable, drop, recipe, reward, or gift.
    NoSource(ItemId),
    /// The place the objective happens has no door path from where the quest
    /// is given.
    Unreachable(LocationId),
}

impl fmt::Display for Unfinishable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NeverStarted => f.write_str("no placed speaker's conversation starts it"),
            Self::NobodyThere(npc) => write!(f, "npc {npc} stands in no location"),
            Self::NothingToSay(npc) => write!(f, "npc {npc} has no conversation"),
            Self::ClueNotPlaced(clue) => write!(f, "clue {clue} is placed in no location"),
            Self::ChoiceNotOffered(choice) => {
                write!(f, "choice {choice} is offered by no marker and no placed speaker")
            }
            Self::EncounterNotPlaced(encounter) => {
                write!(f, "encounter {encounter} has no enemy placed for it")
            }
            Self::NoSource(item) => write!(
                f,
                "item {item} comes from no gatherable, drop, recipe, reward, or conversation"
            ),
            Self::Unreachable(location) => write!(
                f,
                "location {location} has no door path from where the quest is given"
            ),
        }
    }
}

/// Where everything a quest might ask for can be found.
#[derive(Default)]
struct Atlas {
    npcs: BTreeMap<NpcId, BTreeSet<LocationId>>,
    clues: BTreeMap<ClueId, BTreeSet<LocationId>>,
    choice_markers: BTreeMap<ChoiceId, BTreeSet<LocationId>>,
    encounters: BTreeMap<EncounterId, BTreeSet<LocationId>>,
    /// Items found lying in, or dropped by enemies in, each location.
    found: BTreeMap<ItemId, BTreeSet<LocationId>>,
    /// Items produced without a place of their own: recipes, rewards, gifts.
    produced: BTreeSet<ItemId>,
    speakers: BTreeSet<NpcId>,
    /// Choices a conversation can settle, by who speaks it.
    choices_spoken: BTreeMap<ChoiceId, BTreeSet<NpcId>>,
    /// Quests a conversation can start, by who speaks it.
    starters: BTreeMap<QuestId, BTreeSet<NpcId>>,
    doors: BTreeMap<LocationId, BTreeSet<LocationId>>,
}

fn place<K: Ord>(map: &mut BTreeMap<K, BTreeSet<LocationId>>, key: K, location: &LocationId) {
    map.entry(key).or_default().insert(location.clone());
}

impl Atlas {
    fn of(registry: &ContentRegistry) -> Self {
        let mut atlas = Self::default();
        for location in registry.locations.values() {
            let here = &location.id;
            for spawn in &location.spawns {
                match &spawn.role {
                    SpawnRole::Npc { npc, .. } => place(&mut atlas.npcs, npc.clone(), here),
                    SpawnRole::Clue { clue, .. } => place(&mut atlas.clues, clue.clone(), here),
                    SpawnRole::Choice { choice, .. } => {
                        place(&mut atlas.choice_markers, choice.clone(), here)
                    }
                    SpawnRole::Gatherable { item } => place(&mut atlas.found, item.clone(), here),
                    SpawnRole::Enemy {
                        prefab, encounter, ..
                    } => {
                        if let Some(encounter) = encounter {
                            place(&mut atlas.encounters, encounter.clone(), here);
                        }
                        let table = registry
                            .enemies
                            .get(prefab)
                            .and_then(|enemy| enemy.loot_table.as_ref())
                            .and_then(|table| registry.loot_tables.get(table));
                        for entry in table.iter().flat_map(|table| &table.entries) {
                            place(&mut atlas.found, entry.item.clone(), here);
                        }
                    }
                    SpawnRole::PlayerStart | SpawnRole::Station { .. } => {}
                }
            }
            for link in &location.links {
                atlas
                    .doors
                    .entry(here.clone())
                    .or_default()
                    .insert(link.to.clone());
            }
        }
        for recipe in registry.recipes.values() {
            atlas.produced.insert(recipe.output.clone());
        }
        for quest in registry.quests.values() {
            atlas.produced.extend(quest.rewards.items.keys().cloned());
        }
        for dialogue in registry.dialogues.values() {
            atlas.speakers.insert(dialogue.speaker.clone());
            let effects = dialogue
                .nodes
                .values()
                .flat_map(|node| &node.options)
                .flat_map(|option| &option.effects);
            for effect in effects {
                match effect {
                    DialogueEffect::StartQuest(quest) => {
                        atlas
                            .starters
                            .entry(quest.clone())
                            .or_default()
                            .insert(dialogue.speaker.clone());
                    }
                    DialogueEffect::MakeChoice(choice) => {
                        atlas
                            .choices_spoken
                            .entry(choice.clone())
                            .or_default()
                            .insert(dialogue.speaker.clone());
                    }
                    DialogueEffect::GrantItem { item, .. } => {
                        atlas.produced.insert(item.clone());
                    }
                    DialogueEffect::SetWorldFlag(_)
                    | DialogueEffect::AdjustReputation { .. }
                    | DialogueEffect::CompleteObjective { .. }
                    | DialogueEffect::TakeItem { .. } => {}
                }
            }
        }
        atlas
    }

    /// Every location reachable through doors from any of `from`, including
    /// `from` itself.
    fn reachable(&self, from: impl IntoIterator<Item = LocationId>) -> BTreeSet<LocationId> {
        let mut seen = BTreeSet::new();
        let mut frontier: Vec<LocationId> = from.into_iter().collect();
        while let Some(location) = frontier.pop() {
            if seen.contains(&location) {
                continue;
            }
            frontier.extend(self.doors.get(&location).into_iter().flatten().cloned());
            seen.insert(location);
        }
        seen
    }

    fn placed<'a>(&'a self, npc: &NpcId) -> impl Iterator<Item = &'a LocationId> + 'a {
        self.npcs.get(npc).into_iter().flatten()
    }
}

/// Why `places` cannot serve the objective: nowhere at all, or nowhere `reach`
/// covers. The first unreachable place stands for the rest, since one reason
/// per objective is what an author needs to act on.
fn unplaced(
    places: Option<&BTreeSet<LocationId>>,
    reach: &BTreeSet<LocationId>,
    nowhere: impl FnOnce() -> Unfinishable,
) -> Option<Unfinishable> {
    match places {
        None => Some(nowhere()),
        Some(places) if places.iter().any(|place| reach.contains(place)) => None,
        Some(places) => places
            .iter()
            .next()
            .map(|place| Unfinishable::Unreachable(place.clone())),
    }
}

/// Reports every objective of every quest that no player could complete.
pub(crate) fn check(registry: &ContentRegistry, report: &mut ContentReport) {
    let atlas = Atlas::of(registry);
    for quest in registry.quests.values() {
        let mut refuse = |objective: Option<&ObjectiveId>, why: Unfinishable| {
            report.problems.push(Problem::Unfinishable {
                quest: quest.id.clone(),
                objective: objective.cloned(),
                why,
            });
        };

        // Where the quest is given: wherever a speaker who starts it stands.
        let givers: Vec<LocationId> = atlas
            .starters
            .get(&quest.id)
            .into_iter()
            .flatten()
            .flat_map(|npc| atlas.placed(npc).cloned())
            .collect();
        if givers.is_empty() {
            refuse(None, Unfinishable::NeverStarted);
            continue;
        }
        let reach = atlas.reachable(givers);

        for (id, node) in &quest.nodes {
            let problem = match &node.objective {
                Objective::TalkTo { npc } => unplaced(atlas.npcs.get(npc), &reach, || {
                    Unfinishable::NobodyThere(npc.clone())
                })
                .or_else(|| (!atlas.speakers.contains(npc)).then(|| Unfinishable::NothingToSay(npc.clone()))),
                Objective::Escort { npc, destination } => unplaced(atlas.npcs.get(npc), &reach, || {
                    Unfinishable::NobodyThere(npc.clone())
                })
                .or_else(|| {
                    (!reach.contains(destination)).then(|| Unfinishable::Unreachable(destination.clone()))
                }),
                Objective::Reach { location } => {
                    (!reach.contains(location)).then(|| Unfinishable::Unreachable(location.clone()))
                }
                Objective::Investigate { clue } => unplaced(atlas.clues.get(clue), &reach, || {
                    Unfinishable::ClueNotPlaced(clue.clone())
                }),
                Objective::Decide { choice } => {
                    let marked = atlas
                        .choice_markers
                        .get(choice)
                        .is_some_and(|places| places.iter().any(|place| reach.contains(place)));
                    let spoken = atlas
                        .choices_spoken
                        .get(choice)
                        .into_iter()
                        .flatten()
                        .any(|npc| atlas.placed(npc).any(|place| reach.contains(place)));
                    (!marked && !spoken).then(|| Unfinishable::ChoiceNotOffered(choice.clone()))
                }
                Objective::Defeat { encounter } => unplaced(atlas.encounters.get(encounter), &reach, || {
                    Unfinishable::EncounterNotPlaced(encounter.clone())
                }),
                Objective::Collect { item, .. } => {
                    let found = atlas
                        .found
                        .get(item)
                        .is_some_and(|places| places.iter().any(|place| reach.contains(place)));
                    (!found && !atlas.produced.contains(item)).then(|| Unfinishable::NoSource(item.clone()))
                }
            };
            if let Some(why) = problem {
                refuse(Some(id), why);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use game_core::content::{Placement, Point, QuestDef, SpawnPoint};
    use game_core::ids::SpawnId;

    use super::*;

    fn shipped() -> ContentRegistry {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/content");
        crate::load(&root).unwrap_or_else(|report| panic!("shipped content must load:\n{report}"))
    }

    fn id<T>(made: Result<T, impl fmt::Debug>) -> T {
        made.expect("valid id")
    }

    fn quest<'a>(registry: &'a mut ContentRegistry, name: &str) -> &'a mut QuestDef {
        registry
            .quests
            .get_mut(&id(QuestId::new(name)))
            .expect("shipped quest")
    }

    /// Replaces `node`'s objective in `quest`.
    fn ask(registry: &mut ContentRegistry, quest_name: &str, node: &str, objective: Objective) {
        quest(registry, quest_name)
            .nodes
            .get_mut(&id(ObjectiveId::new(node)))
            .expect("shipped node")
            .objective = objective;
    }

    fn refusals(registry: &ContentRegistry) -> Vec<(QuestId, Option<ObjectiveId>, Unfinishable)> {
        let mut report = ContentReport::default();
        check(registry, &mut report);
        report
            .problems
            .into_iter()
            .filter_map(|problem| match problem {
                Problem::Unfinishable {
                    quest,
                    objective,
                    why,
                } => Some((quest, objective, why)),
                _ => None,
            })
            .collect()
    }

    /// The single refusal `registry` earns, which must be for `node`.
    fn refused(registry: &ContentRegistry, node: &str) -> Unfinishable {
        let found = refusals(registry);
        match found.as_slice() {
            [(_, Some(objective), why)] if objective.name() == node => why.clone(),
            other => panic!("expected one refusal, for {node}; got {other:?}"),
        }
    }

    #[test]
    fn every_shipped_quest_can_be_finished() {
        assert_eq!(refusals(&shipped()), Vec::new());
    }

    #[test]
    fn a_quest_no_one_gives_is_refused() {
        let mut content = shipped();
        for dialogue in content.dialogues.values_mut() {
            for option in dialogue.nodes.values_mut().flat_map(|node| &mut node.options) {
                option
                    .effects
                    .retain(|effect| !matches!(effect, DialogueEffect::StartQuest(quest) if quest.name() == "still-waters"));
            }
        }
        let found = refusals(&content);
        assert_eq!(
            found,
            vec![(id(QuestId::new("still-waters")), None, Unfinishable::NeverStarted)]
        );
    }

    #[test]
    fn a_quest_given_by_someone_who_stands_nowhere_is_refused() {
        let mut content = shipped();
        for location in content.locations.values_mut() {
            location.spawns.retain(
                |spawn| !matches!(&spawn.role, SpawnRole::Npc { npc, .. } if npc.name() == "warden-sela"),
            );
        }
        let found = refusals(&content);
        assert!(
            found.contains(&(
                id(QuestId::new("ember-and-ash")),
                None,
                Unfinishable::NeverStarted
            )),
            "{found:?}"
        );
    }

    #[test]
    fn talking_to_someone_placed_nowhere_is_refused() {
        let mut content = shipped();
        let ghost = id(NpcId::new("nobody-at-all"));
        ask(
            &mut content,
            "ember-and-ash",
            "carry-word-back",
            Objective::TalkTo { npc: ghost.clone() },
        );
        assert_eq!(
            refused(&content, "carry-word-back"),
            Unfinishable::NobodyThere(ghost)
        );
    }

    #[test]
    fn talking_to_someone_with_nothing_to_say_is_refused() {
        let mut content = shipped();
        let mute = id(NpcId::new("mute-miller"));
        let square = content
            .locations
            .get_mut(&id(LocationId::new("cinderwake-square")))
            .expect("the square");
        square.spawns.push(SpawnPoint {
            id: id(SpawnId::new("miller-post")),
            placement: Placement {
                at: Point::default(),
                yaw_milli_turns: 0,
            },
            role: SpawnRole::Npc {
                npc: mute.clone(),
                model: None,
            },
        });
        ask(
            &mut content,
            "ember-and-ash",
            "carry-word-back",
            Objective::TalkTo { npc: mute.clone() },
        );
        assert_eq!(
            refused(&content, "carry-word-back"),
            Unfinishable::NothingToSay(mute)
        );
    }

    #[test]
    fn a_clue_placed_nowhere_is_refused() {
        let mut content = shipped();
        let clue = id(ClueId::new("clue-nowhere"));
        ask(
            &mut content,
            "still-waters",
            "survey-waterworks",
            Objective::Investigate { clue: clue.clone() },
        );
        assert_eq!(
            refused(&content, "survey-waterworks"),
            Unfinishable::ClueNotPlaced(clue)
        );
    }

    #[test]
    fn a_choice_is_offered_by_a_marker_or_a_speaker_and_refused_by_neither() {
        // The shipped choice is offered both ways; take the marker away and
        // Thal still offers it, so the quest stands.
        let mut content = shipped();
        for location in content.locations.values_mut() {
            location
                .spawns
                .retain(|spawn| !matches!(spawn.role, SpawnRole::Choice { .. }));
        }
        assert_eq!(refusals(&content), Vec::new(), "a speaker alone offers it");

        // And with Thal gone too, nothing does.
        for location in content.locations.values_mut() {
            location.spawns.retain(
                |spawn| !matches!(&spawn.role, SpawnRole::Npc { npc, .. } if npc.name() == "arbiter-thal"),
            );
        }
        let found = refusals(&content);
        let choice = id(ChoiceId::new("choice-water-rights"));
        assert!(
            found
                .iter()
                .any(|(_, _, why)| *why == Unfinishable::ChoiceNotOffered(choice.clone())),
            "{found:?}"
        );
    }

    #[test]
    fn a_fight_placed_nowhere_is_refused() {
        let mut content = shipped();
        let encounter = id(EncounterId::new("unplaced-fight"));
        ask(
            &mut content,
            "ember-and-ash",
            "end-the-ashlord",
            Objective::Defeat {
                encounter: encounter.clone(),
            },
        );
        assert_eq!(
            refused(&content, "end-the-ashlord"),
            Unfinishable::EncounterNotPlaced(encounter)
        );
    }

    #[test]
    fn an_item_is_collectable_from_a_drop_and_refused_with_no_source() {
        // Ash cinder only drops from husks.
        let cinder = id(ItemId::new("ash-cinder"));
        let mut content = shipped();
        ask(
            &mut content,
            "ember-and-ash",
            "descend",
            Objective::Collect {
                item: cinder.clone(),
                quantity: 3,
            },
        );
        assert_eq!(refusals(&content), Vec::new(), "husks drop it");

        for enemy in content.enemies.values_mut() {
            enemy.loot_table = None;
        }
        assert_eq!(refused(&content, "descend"), Unfinishable::NoSource(cinder));
    }

    #[test]
    fn a_place_no_door_leads_to_is_refused() {
        let mut content = shipped();
        let crypt = id(LocationId::new("crypt-of-ash"));
        for location in content.locations.values_mut() {
            location.links.retain(|link| link.to != crypt);
        }
        let found = refusals(&content);
        let nodes: Vec<(&str, &Unfinishable)> = found
            .iter()
            .filter_map(|(_, objective, why)| objective.as_ref().map(|objective| (objective.name(), why)))
            .collect();
        assert_eq!(
            nodes,
            vec![
                ("descend", &Unfinishable::Unreachable(crypt.clone())),
                ("end-the-ashlord", &Unfinishable::Unreachable(crypt.clone())),
            ],
            "the crypt and the fight in it, and nothing else"
        );
    }

    #[test]
    fn a_refusal_names_the_quest_the_objective_and_the_reason() {
        let problem = Problem::Unfinishable {
            quest: id(QuestId::new("still-waters")),
            objective: Some(id(ObjectiveId::new("survey-waterworks"))),
            why: Unfinishable::ClueNotPlaced(id(ClueId::new("clue-nowhere"))),
        };
        let text = problem.to_string();
        for part in ["still-waters", "survey-waterworks", "clue-nowhere"] {
            assert!(text.contains(part), "{text}");
        }
    }
}

/// The committed fixtures under `tests/fixtures/unfinishable/`: a small valid
/// world and one copy per refusal, each broken in exactly one place. These are
/// the files later generators (M12, M13) are judged against, so each must fail
/// for its own reason and no other.
#[cfg(test)]
mod fixtures {
    use std::path::{Path, PathBuf};

    use super::*;

    fn root(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/unfinishable")
            .join(name)
    }

    /// What a fixture was refused for: each objective by name, and why.
    type Refusals = Vec<(Option<String>, Unfinishable)>;

    fn refusals(name: &str) -> Refusals {
        let report = crate::load(&root(name)).expect_err("the fixture must be refused");
        report
            .problems
            .iter()
            .map(|problem| match problem {
                Problem::Unfinishable { objective, why, .. } => {
                    (objective.as_ref().map(|id| id.name().to_owned()), why.clone())
                }
                other => panic!("{name}: only completability should fail, got: {other}"),
            })
            .collect()
    }

    fn one(objective: &str, why: Unfinishable) -> Refusals {
        vec![(Some(objective.to_owned()), why)]
    }

    fn id<T>(made: Result<T, impl fmt::Debug>) -> T {
        made.expect("valid id")
    }

    #[test]
    fn the_baseline_loads() {
        if let Err(report) = crate::load(&root("finishable")) {
            panic!("the baseline fixture must be valid:\n{report}");
        }
    }

    #[test]
    fn each_broken_fixture_fails_for_its_own_reason() {
        let cellar = || Unfinishable::Unreachable(id(LocationId::new("cellar")));
        let cases: Vec<(&str, Refusals)> = vec![
            ("never-started", vec![(None, Unfinishable::NeverStarted)]),
            (
                "nobody-there",
                one(
                    "report",
                    Unfinishable::NobodyThere(id(NpcId::new("absent-steward"))),
                ),
            ),
            (
                "nothing-to-say",
                one(
                    "report",
                    Unfinishable::NothingToSay(id(NpcId::new("mute-miller"))),
                ),
            ),
            (
                "clue-not-placed",
                one(
                    "read-ledger",
                    Unfinishable::ClueNotPlaced(id(ClueId::new("clue-lost-page"))),
                ),
            ),
            (
                "choice-not-offered",
                one(
                    "pull-lever",
                    Unfinishable::ChoiceNotOffered(id(ChoiceId::new("choice-lever"))),
                ),
            ),
            (
                "no-source",
                one(
                    "gather-moss",
                    Unfinishable::NoSource(id(ItemId::new("cellar-moss"))),
                ),
            ),
            (
                "unreachable",
                vec![
                    (
                        Some("gather-moss".to_owned()),
                        Unfinishable::NoSource(id(ItemId::new("cellar-moss"))),
                    ),
                    (Some("go-down".to_owned()), cellar()),
                ],
            ),
        ];
        for (name, expected) in cases {
            assert_eq!(refusals(name), expected, "fixture {name}");
        }
    }
}
