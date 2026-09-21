//! Driving an authored quest graph from gameplay.
//!
//! A [`QuestDef`] is a graph of objectives; canonical state records which of
//! them are done. This module is the pure function between the two: given what
//! just happened in the world, which domain events should be logged?
//!
//! # Progress is never inferred
//!
//! Nothing here reads the runtime ECS or counts entities. Gameplay reports a
//! [`Signal`] — an enemy group fell, an NPC was spoken to, reeds were picked up
//! — and this module decides what that means for the quest. Everything it
//! decides comes back as [`DomainEvent`]s for the caller to apply, so quest
//! progress is replayable from the log alone. That is the same contract the
//! dialogue runtime keeps, and for the same reason.
//!
//! # Which objectives are open
//!
//! Open objectives are computed from the graph and canonical state rather than
//! stored. The start node is open; any node listed in `next` by a completed node
//! is open; completed nodes are not. Storing an "active node" pointer instead
//! would be a second source of truth that a content revision could desynchronise
//! from the recorded progress — and the save would be the thing that broke.

use std::collections::BTreeSet;

use game_core::content::{Objective, QuestDef, RewardPackage};
use game_core::events::DomainEvent;
use game_core::ids::{ChoiceId, ClueId, EncounterId, ItemId, LocationId, NpcId, ObjectiveId, QuestId};
use game_core::world::{QuestStatus, WorldState};

/// Something that happened in the world which a quest might care about.
///
/// A closed set matching [`Objective`]: gameplay cannot report a fact no
/// objective could express, which keeps the runtime and the schema in step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Signal {
    /// The player arrived at an authored location.
    Reached { location: LocationId },
    /// The player spoke with an NPC.
    Talked { npc: NpcId },
    /// Every enemy of an encounter is down.
    EncounterCleared { encounter: EncounterId },
    /// The player gained countable quest items.
    Collected { item: ItemId, quantity: u32 },
    /// An escorted NPC arrived.
    Escorted { npc: NpcId, destination: LocationId },
    /// A clue was examined.
    Investigated { clue: ClueId },
    /// The player settled an authored choice.
    Decided { choice: ChoiceId },
}

/// Why a quest could not be accepted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuestError {
    /// It is already in progress.
    AlreadyAccepted(QuestId),
    /// It already ended; its outcome is immutable.
    AlreadyFinished(QuestId),
    /// The definition's entry point is not among its nodes. Content validation
    /// catches this, but a runtime that indexed blindly would panic on it.
    MissingStartNode { quest: QuestId, objective: ObjectiveId },
}

impl std::fmt::Display for QuestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyAccepted(quest) => write!(f, "quest {quest} is already active"),
            Self::AlreadyFinished(quest) => write!(f, "quest {quest} has already ended"),
            Self::MissingStartNode { quest, objective } => {
                write!(f, "quest {quest} starts at unknown objective {objective}")
            }
        }
    }
}

impl std::error::Error for QuestError {}

/// Accepts a quest, returning the event that records it.
pub fn accept(def: &QuestDef, state: &WorldState) -> Result<Vec<DomainEvent>, QuestError> {
    let status = state.quest_status(&def.id);
    if status == QuestStatus::Active {
        return Err(QuestError::AlreadyAccepted(def.id.clone()));
    }
    if status.is_terminal() {
        return Err(QuestError::AlreadyFinished(def.id.clone()));
    }
    if !def.nodes.contains_key(&def.start) {
        return Err(QuestError::MissingStartNode {
            quest: def.id.clone(),
            objective: def.start.clone(),
        });
    }
    Ok(vec![DomainEvent::QuestAccepted {
        quest: def.id.clone(),
    }])
}

/// The objectives currently awaiting the player, in id order.
///
/// Empty unless the quest is active — a quest that has ended has nothing open,
/// and one never accepted has not started.
pub fn open_objectives(def: &QuestDef, state: &WorldState) -> Vec<ObjectiveId> {
    let Some(quest) = state.quest(&def.id) else {
        return Vec::new();
    };
    if quest.status != QuestStatus::Active {
        return Vec::new();
    }

    let mut unlocked: BTreeSet<&ObjectiveId> = BTreeSet::new();
    if let Some((start, _)) = def.nodes.get_key_value(&def.start) {
        unlocked.insert(start);
    }
    for (id, node) in &def.nodes {
        if quest.objective(id).completed {
            for next in &node.next {
                if let Some((key, _)) = def.nodes.get_key_value(next) {
                    unlocked.insert(key);
                }
            }
        }
    }

    unlocked
        .into_iter()
        .filter(|id| !quest.objective(id).completed)
        .cloned()
        .collect()
}

/// How a signal bears on an objective.
enum Bearing {
    /// The objective is satisfied outright.
    Satisfied,
    /// Countable progress was made.
    Progress(u32),
}

fn bearing(objective: &Objective, signal: &Signal) -> Option<Bearing> {
    match (objective, signal) {
        (Objective::Reach { location }, Signal::Reached { location: at }) if location == at => {
            Some(Bearing::Satisfied)
        }
        (Objective::TalkTo { npc }, Signal::Talked { npc: with }) if npc == with => Some(Bearing::Satisfied),
        (
            Objective::Defeat { encounter },
            Signal::EncounterCleared {
                encounter: cleared, ..
            },
        ) if encounter == cleared => Some(Bearing::Satisfied),
        (
            Objective::Collect { item, .. },
            Signal::Collected {
                item: gained,
                quantity,
            },
        ) if item == gained => Some(Bearing::Progress(*quantity)),
        (
            Objective::Escort { npc, destination },
            Signal::Escorted {
                npc: escorted,
                destination: arrived,
            },
        ) if npc == escorted && destination == arrived => Some(Bearing::Satisfied),
        (Objective::Investigate { clue }, Signal::Investigated { clue: found }) if clue == found => {
            Some(Bearing::Satisfied)
        }
        (Objective::Decide { choice }, Signal::Decided { choice: made }) if choice == made => {
            Some(Bearing::Satisfied)
        }
        _ => None,
    }
}

/// What a quest's rewards become when it completes.
///
/// No-op rewards are skipped: the reducer rejects a zero quantity outright, and
/// a zero reputation delta would put a line in the log that changed nothing.
pub fn reward_events(quest: &QuestDef) -> Vec<DomainEvent> {
    let RewardPackage {
        items,
        experience,
        reputation,
    } = &quest.rewards;

    let mut events = Vec::new();
    for (item, quantity) in items {
        if *quantity > 0 {
            events.push(DomainEvent::ItemGranted {
                item: item.clone(),
                quantity: *quantity,
            });
        }
    }
    if *experience > 0 {
        events.push(DomainEvent::ExperienceGained { amount: *experience });
    }
    for (faction, delta) in reputation {
        if *delta != 0 {
            events.push(DomainEvent::FactionReputationChanged {
                faction: faction.clone(),
                delta: *delta,
            });
        }
    }
    events
}

/// Applies a gameplay signal to a quest, returning the events it causes.
///
/// Returns an empty vector when the signal is irrelevant, which is the common
/// case: most things that happen do not concern most quests.
///
/// The events are ordered so they apply cleanly in sequence — objective
/// progress, then completion, then the quest's ending and its rewards. Counted
/// objectives emit an *advance* carrying the delta, and additionally a
/// *completion* when the running total reaches what content requires.
pub fn advance(def: &QuestDef, state: &WorldState, signal: &Signal) -> Vec<DomainEvent> {
    let Some(quest) = state.quest(&def.id) else {
        return Vec::new();
    };

    let mut events = Vec::new();
    let mut satisfied: Vec<ObjectiveId> = Vec::new();

    for id in open_objectives(def, state) {
        let Some(node) = def.nodes.get(&id) else {
            continue;
        };
        let Some(bearing) = bearing(&node.objective, signal) else {
            continue;
        };

        match bearing {
            Bearing::Satisfied => {
                events.push(DomainEvent::QuestObjectiveCompleted {
                    quest: def.id.clone(),
                    objective: id.clone(),
                });
                satisfied.push(id);
            }
            Bearing::Progress(delta) => {
                if delta == 0 {
                    continue;
                }
                events.push(DomainEvent::QuestObjectiveAdvanced {
                    quest: def.id.clone(),
                    objective: id.clone(),
                    count: delta,
                });

                let required = match node.objective {
                    Objective::Collect { quantity, .. } => u32::from(quantity),
                    _ => 0,
                };
                if quest.objective(&id).count.saturating_add(delta) >= required {
                    events.push(DomainEvent::QuestObjectiveCompleted {
                        quest: def.id.clone(),
                        objective: id.clone(),
                    });
                    satisfied.push(id);
                }
            }
        }
    }

    // A terminal node ends the quest. If one signal somehow satisfies two
    // endings at once the first in id order wins, so the outcome is at least
    // deterministic; content validation is what keeps that from arising.
    for id in satisfied {
        let Some(resolution) = def.nodes.get(&id).and_then(|node| node.resolution.as_ref()) else {
            continue;
        };
        events.push(DomainEvent::QuestCompleted {
            quest: def.id.clone(),
            resolution: resolution.clone(),
        });
        events.extend(reward_events(def));
        break;
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::content::{CONTENT_SCHEMA_VERSION, QuestNode};
    use game_core::events::{apply, apply_all};
    use game_core::ids::{ContentRevisionId, FactionId, QuestResolutionId};
    use std::collections::BTreeMap;

    fn objective_id(name: &str) -> ObjectiveId {
        ObjectiveId::new(name).expect("valid id")
    }

    fn npc(name: &str) -> NpcId {
        NpcId::new(name).expect("valid id")
    }

    fn item(name: &str) -> ItemId {
        ItemId::new(name).expect("valid id")
    }

    fn node(objective: Objective, next: &[&str], resolution: Option<&str>) -> QuestNode {
        QuestNode {
            objective,
            next: next.iter().map(|id| objective_id(id)).collect(),
            resolution: resolution.map(|id| QuestResolutionId::new(id).expect("valid id")),
            text: None,
        }
    }

    /// The slice's opening quest: hear the problem, gather reeds, report back.
    fn still_waters() -> QuestDef {
        QuestDef {
            schema_version: CONTENT_SCHEMA_VERSION,
            id: QuestId::new("still-waters").expect("valid id"),
            title: "Still Waters".to_owned(),
            start: objective_id("hear-sela"),
            nodes: BTreeMap::from([
                (
                    objective_id("hear-sela"),
                    node(
                        Objective::TalkTo {
                            npc: npc("warden-sela"),
                        },
                        &["gather-reeds"],
                        None,
                    ),
                ),
                (
                    objective_id("gather-reeds"),
                    node(
                        Objective::Collect {
                            item: item("still-reed"),
                            quantity: 3,
                        },
                        &["report-back"],
                        None,
                    ),
                ),
                (
                    objective_id("report-back"),
                    node(
                        Objective::TalkTo {
                            npc: npc("warden-sela"),
                        },
                        &[],
                        Some("intake-cleared"),
                    ),
                ),
            ]),
            rewards: RewardPackage {
                items: BTreeMap::from([(item("ember-ore"), 4)]),
                experience: 120,
                reputation: BTreeMap::from([(FactionId::new("cinderwake-settlers").expect("valid id"), 10)]),
            },
        }
    }

    fn world() -> WorldState {
        WorldState::new(7, ContentRevisionId::new("rev-0001").expect("valid id"))
    }

    fn started() -> (QuestDef, WorldState) {
        let def = still_waters();
        let mut state = world();
        let events = accept(&def, &state).expect("a fresh quest can be accepted");
        apply_all(&mut state, events.iter()).expect("applies");
        (def, state)
    }

    #[test]
    fn accepting_records_the_quest() {
        let def = still_waters();
        let state = world();
        assert_eq!(
            accept(&def, &state).expect("accepted"),
            vec![DomainEvent::QuestAccepted {
                quest: def.id.clone()
            }]
        );
    }

    #[test]
    fn a_quest_cannot_be_accepted_twice() {
        let (def, state) = started();
        assert_eq!(
            accept(&def, &state),
            Err(QuestError::AlreadyAccepted(def.id.clone()))
        );
    }

    #[test]
    fn a_finished_quest_cannot_be_restarted() {
        let (def, mut state) = started();
        apply(
            &mut state,
            &DomainEvent::QuestCompleted {
                quest: def.id.clone(),
                resolution: QuestResolutionId::new("intake-cleared").expect("valid id"),
            },
        )
        .expect("applies");

        assert_eq!(
            accept(&def, &state),
            Err(QuestError::AlreadyFinished(def.id.clone())),
            "an ending is immutable; re-accepting would rewrite history"
        );
    }

    #[test]
    fn a_quest_whose_entry_point_is_missing_is_refused_rather_than_panicking() {
        let mut def = still_waters();
        def.start = objective_id("nowhere");
        let state = world();
        assert_eq!(
            accept(&def, &state),
            Err(QuestError::MissingStartNode {
                quest: def.id.clone(),
                objective: objective_id("nowhere")
            })
        );
    }

    #[test]
    fn only_the_start_is_open_at_first() {
        let (def, state) = started();
        assert_eq!(open_objectives(&def, &state), vec![objective_id("hear-sela")]);
    }

    #[test]
    fn an_unaccepted_quest_has_nothing_open_and_ignores_signals() {
        let def = still_waters();
        let state = world();
        assert!(open_objectives(&def, &state).is_empty());
        assert!(
            advance(
                &def,
                &state,
                &Signal::Talked {
                    npc: npc("warden-sela")
                }
            )
            .is_empty(),
            "a quest nobody accepted must not progress"
        );
    }

    #[test]
    fn completing_a_node_opens_the_next_one() {
        let (def, mut state) = started();
        let events = advance(
            &def,
            &state,
            &Signal::Talked {
                npc: npc("warden-sela"),
            },
        );
        assert_eq!(
            events,
            vec![DomainEvent::QuestObjectiveCompleted {
                quest: def.id.clone(),
                objective: objective_id("hear-sela"),
            }]
        );

        apply_all(&mut state, events.iter()).expect("applies");
        assert_eq!(
            open_objectives(&def, &state),
            vec![objective_id("gather-reeds")],
            "the graph advances, and the finished node closes"
        );
    }

    #[test]
    fn an_irrelevant_signal_produces_nothing() {
        let (def, state) = started();
        assert!(
            advance(
                &def,
                &state,
                &Signal::Reached {
                    location: LocationId::new("cinderwake-square").expect("valid id")
                }
            )
            .is_empty()
        );
        assert!(
            advance(
                &def,
                &state,
                &Signal::Talked {
                    npc: npc("smith-orin")
                }
            )
            .is_empty(),
            "the right kind of signal about the wrong subject must not count"
        );
    }

    /// The reducer treats `count` as a delta, so the runtime must send deltas
    /// and do the threshold arithmetic itself. Sending a running total here
    /// would double-count on the second pickup.
    #[test]
    fn counted_progress_accumulates_and_completes_at_the_threshold() {
        let (def, mut state) = started();
        let talk = advance(
            &def,
            &state,
            &Signal::Talked {
                npc: npc("warden-sela"),
            },
        );
        apply_all(&mut state, talk.iter()).expect("applies");

        let first = advance(
            &def,
            &state,
            &Signal::Collected {
                item: item("still-reed"),
                quantity: 2,
            },
        );
        assert_eq!(
            first,
            vec![DomainEvent::QuestObjectiveAdvanced {
                quest: def.id.clone(),
                objective: objective_id("gather-reeds"),
                count: 2,
            }],
            "two of three is progress, not completion"
        );
        apply_all(&mut state, first.iter()).expect("applies");
        assert_eq!(
            state
                .objective_progress(&def.id, &objective_id("gather-reeds"))
                .count,
            2
        );

        let second = advance(
            &def,
            &state,
            &Signal::Collected {
                item: item("still-reed"),
                quantity: 1,
            },
        );
        assert_eq!(
            second,
            vec![
                DomainEvent::QuestObjectiveAdvanced {
                    quest: def.id.clone(),
                    objective: objective_id("gather-reeds"),
                    count: 1,
                },
                DomainEvent::QuestObjectiveCompleted {
                    quest: def.id.clone(),
                    objective: objective_id("gather-reeds"),
                },
            ],
            "the third reed both advances and finishes the objective"
        );
        apply_all(&mut state, second.iter()).expect("applies");
        assert_eq!(open_objectives(&def, &state), vec![objective_id("report-back")]);
    }

    #[test]
    fn overshooting_the_threshold_still_completes_once() {
        let (def, mut state) = started();
        let talk = advance(
            &def,
            &state,
            &Signal::Talked {
                npc: npc("warden-sela"),
            },
        );
        apply_all(&mut state, talk.iter()).expect("applies");

        let events = advance(
            &def,
            &state,
            &Signal::Collected {
                item: item("still-reed"),
                quantity: 9,
            },
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, DomainEvent::QuestObjectiveCompleted { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn a_collected_signal_for_the_wrong_item_does_not_count() {
        let (def, mut state) = started();
        let talk = advance(
            &def,
            &state,
            &Signal::Talked {
                npc: npc("warden-sela"),
            },
        );
        apply_all(&mut state, talk.iter()).expect("applies");

        assert!(
            advance(
                &def,
                &state,
                &Signal::Collected {
                    item: item("ash-cinder"),
                    quantity: 5
                }
            )
            .is_empty()
        );
    }

    #[test]
    fn a_zero_quantity_pickup_logs_nothing() {
        let (def, mut state) = started();
        let talk = advance(
            &def,
            &state,
            &Signal::Talked {
                npc: npc("warden-sela"),
            },
        );
        apply_all(&mut state, talk.iter()).expect("applies");

        assert!(
            advance(
                &def,
                &state,
                &Signal::Collected {
                    item: item("still-reed"),
                    quantity: 0
                }
            )
            .is_empty(),
            "the reducer rejects zero progress outright"
        );
    }

    /// The whole quest, driven only by signals and the reducer.
    #[test]
    fn the_quest_can_be_played_start_to_finish() {
        let (def, mut state) = started();
        let signals = [
            Signal::Talked {
                npc: npc("warden-sela"),
            },
            Signal::Collected {
                item: item("still-reed"),
                quantity: 3,
            },
            Signal::Talked {
                npc: npc("warden-sela"),
            },
        ];

        let mut log = Vec::new();
        for signal in &signals {
            let events = advance(&def, &state, signal);
            apply_all(&mut state, events.iter()).expect("every emitted event applies");
            log.extend(events);
        }

        assert_eq!(
            state.quest_status(&def.id),
            QuestStatus::Completed {
                resolution: QuestResolutionId::new("intake-cleared").expect("valid id")
            },
            "the ending is recorded, not merely 'done'"
        );
        assert!(open_objectives(&def, &state).is_empty());

        // And the rewards actually arrived.
        assert_eq!(state.player.item_count(&item("ember-ore")), 4);
        assert_eq!(state.player.progression.experience, 120);
        assert_eq!(
            state
                .reputation(&FactionId::new("cinderwake-settlers").expect("valid id"))
                .get(),
            10
        );
        assert!(log.contains(&DomainEvent::QuestCompleted {
            quest: def.id.clone(),
            resolution: QuestResolutionId::new("intake-cleared").expect("valid id"),
        }));
    }

    #[test]
    fn a_finished_quest_ignores_further_signals() {
        let (def, mut state) = started();
        for signal in [
            Signal::Talked {
                npc: npc("warden-sela"),
            },
            Signal::Collected {
                item: item("still-reed"),
                quantity: 3,
            },
            Signal::Talked {
                npc: npc("warden-sela"),
            },
        ] {
            let events = advance(&def, &state, &signal);
            apply_all(&mut state, events.iter()).expect("applies");
        }

        assert!(
            advance(
                &def,
                &state,
                &Signal::Talked {
                    npc: npc("warden-sela")
                }
            )
            .is_empty(),
            "a completed quest must not hand out its rewards twice"
        );
    }

    #[test]
    fn rewards_skip_no_op_entries() {
        let mut def = still_waters();
        def.rewards = RewardPackage {
            items: BTreeMap::from([(item("ember-ore"), 0)]),
            experience: 0,
            reputation: BTreeMap::from([(FactionId::new("cinderwake-settlers").expect("valid id"), 0)]),
        };
        assert!(
            reward_events(&def).is_empty(),
            "a reward of nothing must not appear in the log"
        );
    }

    #[test]
    fn advancing_is_deterministic() {
        let (def, mut state) = started();
        let talk = advance(
            &def,
            &state,
            &Signal::Talked {
                npc: npc("warden-sela"),
            },
        );
        apply_all(&mut state, talk.iter()).expect("applies");

        let signal = Signal::Collected {
            item: item("still-reed"),
            quantity: 3,
        };
        assert_eq!(advance(&def, &state, &signal), advance(&def, &state, &signal));
    }

    #[test]
    fn advancing_leaves_canonical_state_untouched() {
        let (def, state) = started();
        let before = state.clone();
        let _ = advance(
            &def,
            &state,
            &Signal::Talked {
                npc: npc("warden-sela"),
            },
        );
        assert_eq!(
            state, before,
            "the quest runtime proposes events; only the reducer writes"
        );
    }
}
