//! Running an authored conversation.
//!
//! A conversation is a cursor over a [`DialogueDef`] graph plus two pure
//! questions: which options may be shown right now, and what taking one means.
//!
//! # Nothing here mutates the world
//!
//! Choosing an option returns [`DomainEvent`]s. It does not touch
//! [`WorldState`], because every durable consequence of a conversation — a flag,
//! a quest, an item, a reputation shift — has to travel through the event log to
//! be replayable. A dialogue runtime that wrote state directly would be the one
//! place in the game where progress could not be replayed, and it would not be
//! obvious until a save failed to reproduce.
//!
//! # Conditions are evaluated, never executed
//!
//! [`Condition`] is a closed enum answerable from canonical state alone
//! (ADR-0010). There is no expression to parse and no predicate to call, which
//! is what will later make model-proposed dialogue safe to run unreviewed: the
//! worst a bad proposal can express is a reference to something that does not
//! exist, and content validation already rejects that.

use game_core::content::{Condition, DialogueDef, DialogueEffect, DialogueNode, DialogueOption};
use game_core::events::DomainEvent;
use game_core::ids::{ChoiceId, DialogueId, DialogueNodeId, NpcId};
use game_core::world::{QuestStatus, WorldState};

/// Why a conversation could not take a step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DialogueError {
    /// The conversation has already closed.
    Ended(DialogueId),
    /// The option index is not one of the options currently on offer. Either it
    /// is out of range, or its conditions do not hold — a caller must not be
    /// able to take a line the player could not see.
    UnavailableOption { index: usize },
    /// The cursor names a node the definition does not contain.
    UnknownNode(DialogueNodeId),
}

impl std::fmt::Display for DialogueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ended(dialogue) => write!(f, "conversation {dialogue} has already ended"),
            Self::UnavailableOption { index } => {
                write!(f, "option {index} is not currently on offer")
            }
            Self::UnknownNode(node) => write!(f, "no such dialogue node: {node}"),
        }
    }
}

impl std::error::Error for DialogueError {}

/// One line the player may currently say.
///
/// Carries the index the caller passes back to [`Conversation::choose`], so a UI
/// never has to reconstruct which option a row came from — the filtered list and
/// the original indices cannot drift apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Choice<'a> {
    /// Index into the node's full option list.
    pub index: usize,
    pub text: &'a str,
}

/// What taking an option produced.
///
/// Two outputs rather than one because they travel differently: `events` are
/// durable and go through the reducer, while `choices` are gameplay signals a
/// quest consumes this tick. Returning only events would make a settled choice
/// invisible to the quest runtime; inventing a `ChoiceMade` event to carry it
/// would record the same fact twice.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Turn {
    /// Durable consequences, for the caller to apply.
    pub events: Vec<DomainEvent>,
    /// Authored decisions this option settled, in the order the content lists
    /// them.
    pub choices: Vec<ChoiceId>,
}

/// Answers one condition from canonical state.
pub fn evaluate(condition: &Condition, state: &WorldState) -> bool {
    match condition {
        Condition::WorldFlagSet(flag) => state.has_world_flag(flag),
        Condition::WorldFlagUnset(flag) => !state.has_world_flag(flag),
        Condition::QuestNotStarted(quest) => state.quest_status(quest) == QuestStatus::NotStarted,
        Condition::QuestActive(quest) => state.quest_status(quest) == QuestStatus::Active,
        Condition::QuestCompleted(quest) => {
            matches!(state.quest_status(quest), QuestStatus::Completed { .. })
        }
        Condition::ReputationAtLeast { faction, value } => state.reputation(faction).get() >= *value,
        Condition::HasItem { item, quantity } => state.player.item_count(item) >= *quantity,
    }
}

/// Whether every condition on an option holds. An option with no conditions
/// always shows.
pub fn is_available(option: &DialogueOption, state: &WorldState) -> bool {
    option
        .conditions
        .iter()
        .all(|condition| evaluate(condition, state))
}

/// Whether a whole conversation is open: every condition on it holds (M9).
pub fn is_open(def: &DialogueDef, state: &WorldState) -> bool {
    def.conditions.iter().all(|condition| evaluate(condition, state))
}

/// The conversations `npc` would hold right now, in id order.
///
/// Id order because the registry is keyed by id, so the same state always
/// offers the same list in the same sequence — which is what lets a replayed
/// digit pick the same topic.
pub fn open_conversations<'a>(
    dialogues: impl IntoIterator<Item = &'a DialogueDef>,
    npc: &NpcId,
    state: &WorldState,
) -> Vec<&'a DialogueDef> {
    dialogues
        .into_iter()
        .filter(|def| &def.speaker == npc && is_open(def, state))
        .collect()
}

/// The options a node currently offers, with their original indices.
pub fn available<'a>(node: &'a DialogueNode, state: &WorldState) -> Vec<Choice<'a>> {
    node.options
        .iter()
        .enumerate()
        .filter(|(_, option)| is_available(option, state))
        .map(|(index, option)| Choice {
            index,
            text: &option.text,
        })
        .collect()
}

/// The event one effect becomes.
///
/// Returns `None` for an effect that would be a no-op: a zero-quantity grant or
/// a zero reputation delta. The reducer rejects zero quantities outright
/// (`DomainError::ZeroQuantity`), and a zero delta would put a line in the log
/// that changed nothing — both make a replay harder to read and to trust.
pub fn effect_event(effect: &DialogueEffect) -> Option<DomainEvent> {
    match effect {
        DialogueEffect::SetWorldFlag(flag) => Some(DomainEvent::WorldFlagSet { flag: flag.clone() }),
        DialogueEffect::AdjustReputation { faction, delta } => {
            (*delta != 0).then(|| DomainEvent::FactionReputationChanged {
                faction: faction.clone(),
                delta: *delta,
            })
        }
        DialogueEffect::StartQuest(quest) => Some(DomainEvent::QuestAccepted { quest: quest.clone() }),
        DialogueEffect::CompleteObjective { quest, objective } => {
            Some(DomainEvent::QuestObjectiveCompleted {
                quest: quest.clone(),
                objective: objective.clone(),
            })
        }
        DialogueEffect::GrantItem { item, quantity } => (*quantity > 0).then(|| DomainEvent::ItemGranted {
            item: item.clone(),
            quantity: *quantity,
        }),
        DialogueEffect::TakeItem { item, quantity } => (*quantity > 0).then(|| DomainEvent::ItemConsumed {
            item: item.clone(),
            quantity: *quantity,
        }),
        // Settling a choice is not itself a durable fact: the quest objective it
        // completes is what gets recorded. See [`effect_choice`].
        DialogueEffect::MakeChoice(_) => None,
    }
}

/// The decision an effect settles, if it settles one.
pub fn effect_choice(effect: &DialogueEffect) -> Option<&ChoiceId> {
    match effect {
        DialogueEffect::MakeChoice(choice) => Some(choice),
        _ => None,
    }
}

/// A conversation in progress.
///
/// Runtime state, deliberately not serializable: a save taken mid-conversation
/// resumes with the conversation closed, and everything the player already
/// agreed to is in the event log. Persisting a cursor would make a dialogue
/// graph edit able to strand an old save inside a node that no longer exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Conversation {
    dialogue: DialogueId,
    current: Option<DialogueNodeId>,
}

impl Conversation {
    /// Starts `def` at its authored entry point.
    pub fn open(def: &DialogueDef) -> Self {
        Self {
            dialogue: def.id.clone(),
            current: def.nodes.contains_key(&def.start).then(|| def.start.clone()),
        }
    }

    /// Which conversation this is.
    pub fn dialogue(&self) -> &DialogueId {
        &self.dialogue
    }

    /// The node the player is looking at, or `None` once the conversation closed.
    pub fn current(&self) -> Option<&DialogueNodeId> {
        self.current.as_ref()
    }

    pub fn is_over(&self) -> bool {
        self.current.is_none()
    }

    /// Closes the conversation. Walking away is always allowed.
    pub fn end(&mut self) {
        self.current = None;
    }

    /// The node being spoken, if the conversation is still open.
    pub fn node<'a>(&self, def: &'a DialogueDef) -> Option<&'a DialogueNode> {
        def.nodes.get(self.current.as_ref()?)
    }

    /// What the player may say right now.
    ///
    /// An empty list means the only remaining action is to walk away — which is
    /// why [`end`](Self::end) never fails. A node whose every option is gated
    /// off is a dead end in *this* world state, not a soft lock.
    pub fn options<'a>(&self, def: &'a DialogueDef, state: &WorldState) -> Vec<Choice<'a>> {
        self.node(def)
            .map(|node| available(node, state))
            .unwrap_or_default()
    }

    /// Takes an option, returning the events it causes.
    ///
    /// The caller applies them (`game_core::events::apply_all`); this advances
    /// the cursor. If the option leads to a node the definition lacks, the
    /// conversation closes rather than dangling — content validation checks node
    /// references, so reaching that is a content bug, and ending gracefully
    /// beats trapping the player in a broken conversation.
    pub fn choose(
        &mut self,
        def: &DialogueDef,
        state: &WorldState,
        index: usize,
    ) -> Result<Turn, DialogueError> {
        let current = self
            .current
            .clone()
            .ok_or_else(|| DialogueError::Ended(self.dialogue.clone()))?;
        let node = def
            .nodes
            .get(&current)
            .ok_or(DialogueError::UnknownNode(current))?;

        let option = node
            .options
            .get(index)
            .filter(|option| is_available(option, state))
            .ok_or(DialogueError::UnavailableOption { index })?;

        let turn = Turn {
            events: option.effects.iter().filter_map(effect_event).collect(),
            choices: option.effects.iter().filter_map(effect_choice).cloned().collect(),
        };
        self.current = option
            .next
            .as_ref()
            .filter(|next| def.nodes.contains_key(*next))
            .cloned();
        Ok(turn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::content::CONTENT_SCHEMA_VERSION;
    use game_core::events::apply_all;
    use game_core::ids::{ContentRevisionId, FactionId, ItemId, NpcId, QuestId, WorldFlag};
    use game_core::world::WorldState;
    use std::collections::BTreeMap;

    fn node_id(name: &str) -> DialogueNodeId {
        DialogueNodeId::new(name).expect("valid id")
    }

    fn quest_id(name: &str) -> QuestId {
        QuestId::new(name).expect("valid id")
    }

    fn item_id(name: &str) -> ItemId {
        ItemId::new(name).expect("valid id")
    }

    fn flag(name: &str) -> WorldFlag {
        WorldFlag::new(name).expect("valid id")
    }

    fn faction(name: &str) -> FactionId {
        FactionId::new(name).expect("valid id")
    }

    fn option(
        text: &str,
        conditions: Vec<Condition>,
        effects: Vec<DialogueEffect>,
        next: Option<&str>,
    ) -> DialogueOption {
        DialogueOption {
            text: text.to_owned(),
            conditions,
            effects,
            next: next.map(node_id),
        }
    }

    fn node(text: &str, options: Vec<DialogueOption>) -> DialogueNode {
        DialogueNode {
            text: text.to_owned(),
            options,
        }
    }

    fn def(start: &str, nodes: Vec<(&str, DialogueNode)>) -> DialogueDef {
        DialogueDef {
            schema_version: CONTENT_SCHEMA_VERSION,
            id: DialogueId::new("sela-intake").expect("valid id"),
            speaker: NpcId::new("warden-sela").expect("valid id"),
            conditions: Vec::new(),
            topic: None,
            start: node_id(start),
            nodes: nodes.into_iter().map(|(id, node)| (node_id(id), node)).collect(),
        }
    }

    fn world() -> WorldState {
        WorldState::new(7, ContentRevisionId::new("rev-0001").expect("valid id"))
    }

    /// The quest-giving conversation the slice actually needs: an offer gated on
    /// the quest not having been taken, and a hand-in gated on holding the goods.
    fn intake() -> DialogueDef {
        def(
            "greeting",
            vec![
                (
                    "greeting",
                    node(
                        "The intake is fouled.",
                        vec![
                            option(
                                "I will clear it.",
                                vec![Condition::QuestNotStarted(quest_id("still-waters"))],
                                vec![DialogueEffect::StartQuest(quest_id("still-waters"))],
                                Some("thanks"),
                            ),
                            option(
                                "Here are the reeds.",
                                vec![Condition::HasItem {
                                    item: item_id("still-reed"),
                                    quantity: 2,
                                }],
                                vec![
                                    DialogueEffect::TakeItem {
                                        item: item_id("still-reed"),
                                        quantity: 2,
                                    },
                                    DialogueEffect::AdjustReputation {
                                        faction: faction("cinderwake-settlers"),
                                        delta: 5,
                                    },
                                    DialogueEffect::SetWorldFlag(flag("intake-cleared")),
                                ],
                                None,
                            ),
                            option("Not now.", Vec::new(), Vec::new(), None),
                        ],
                    ),
                ),
                ("thanks", node("Fortune go with you.", Vec::new())),
            ],
        )
    }

    #[test]
    fn a_conversation_starts_at_its_authored_entry_point() {
        let def = intake();
        let talk = Conversation::open(&def);
        assert_eq!(talk.current(), Some(&node_id("greeting")));
        assert!(!talk.is_over());
        assert_eq!(talk.dialogue(), &def.id);
    }

    #[test]
    fn a_definition_whose_start_is_missing_opens_already_closed() {
        let broken = def("nowhere", vec![("greeting", node("...", Vec::new()))]);
        let talk = Conversation::open(&broken);
        assert!(
            talk.is_over(),
            "a dangling entry point must not panic on first use"
        );
    }

    #[test]
    fn options_are_gated_on_world_state() {
        let def = intake();
        let state = world();
        let talk = Conversation::open(&def);

        let offered = talk.options(&def, &state);
        let texts: Vec<&str> = offered.iter().map(|choice| choice.text).collect();
        assert_eq!(
            texts,
            vec!["I will clear it.", "Not now."],
            "the hand-in line must stay hidden until the reeds are held"
        );
        assert_eq!(
            offered.iter().map(|choice| choice.index).collect::<Vec<_>>(),
            vec![0, 2],
            "indices must stay those of the full option list, not the filtered one"
        );
    }

    #[test]
    fn holding_the_goods_reveals_the_hand_in() {
        let def = intake();
        let mut state = world();
        state.player.inventory.insert(item_id("still-reed"), 2);

        let talk = Conversation::open(&def);
        let texts: Vec<&str> = talk
            .options(&def, &state)
            .iter()
            .map(|choice| choice.text)
            .collect();
        assert!(texts.contains(&"Here are the reeds."));
    }

    #[test]
    fn taking_an_option_returns_its_effects_as_events() {
        let def = intake();
        let mut state = world();
        state.player.inventory.insert(item_id("still-reed"), 2);
        let mut talk = Conversation::open(&def);

        let turn = talk.choose(&def, &state, 1).expect("the option is on offer");
        assert_eq!(
            turn.events,
            vec![
                DomainEvent::ItemConsumed {
                    item: item_id("still-reed"),
                    quantity: 2
                },
                DomainEvent::FactionReputationChanged {
                    faction: faction("cinderwake-settlers"),
                    delta: 5
                },
                DomainEvent::WorldFlagSet {
                    flag: flag("intake-cleared")
                },
            ],
            "effects become events in authored order"
        );
        assert!(talk.is_over(), "an option with no next closes the conversation");
    }

    /// The contract that matters: dialogue proposes, the reducer disposes.
    #[test]
    fn the_events_a_conversation_returns_actually_apply() {
        let def = intake();
        let mut state = world();
        state.player.inventory.insert(item_id("still-reed"), 2);
        let mut talk = Conversation::open(&def);

        let turn = talk.choose(&def, &state, 1).expect("the option is on offer");
        apply_all(&mut state, turn.events.iter()).expect("every event is valid against the world");

        assert_eq!(state.player.item_count(&item_id("still-reed")), 0);
        assert_eq!(state.reputation(&faction("cinderwake-settlers")).get(), 5);
        assert!(state.has_world_flag(&flag("intake-cleared")));
    }

    #[test]
    fn choosing_a_hidden_option_is_refused() {
        let def = intake();
        let state = world();
        let mut talk = Conversation::open(&def);

        assert_eq!(
            talk.choose(&def, &state, 1),
            Err(DialogueError::UnavailableOption { index: 1 }),
            "a line the player cannot see must not be reachable through the api"
        );
        assert_eq!(
            talk.choose(&def, &state, 9),
            Err(DialogueError::UnavailableOption { index: 9 })
        );
        assert!(!talk.is_over(), "a refused choice leaves the cursor alone");
    }

    #[test]
    fn a_closed_conversation_refuses_further_choices() {
        let def = intake();
        let state = world();
        let mut talk = Conversation::open(&def);
        talk.end();

        assert_eq!(
            talk.choose(&def, &state, 0),
            Err(DialogueError::Ended(def.id.clone()))
        );
    }

    #[test]
    fn a_node_with_no_available_options_can_still_be_left() {
        let def = intake();
        let mut state = world();
        // Quest already taken and no reeds held: only "Not now." remains.
        apply_all(
            &mut state,
            [DomainEvent::QuestAccepted {
                quest: quest_id("still-waters"),
            }]
            .iter(),
        )
        .expect("applies");

        let mut talk = Conversation::open(&def);
        assert_eq!(
            talk.options(&def, &state)
                .iter()
                .map(|choice| choice.text)
                .collect::<Vec<_>>(),
            vec!["Not now."],
            "both substantive lines are gated off in this world state"
        );

        let turn = talk.choose(&def, &state, 2).expect("walking away always works");
        assert!(turn.events.is_empty());
        assert!(turn.choices.is_empty());
        assert!(talk.is_over());
    }

    #[test]
    fn following_an_option_moves_to_its_next_node() {
        let def = intake();
        let state = world();
        let mut talk = Conversation::open(&def);

        let turn = talk.choose(&def, &state, 0).expect("the offer is available");
        assert_eq!(
            turn.events,
            vec![DomainEvent::QuestAccepted {
                quest: quest_id("still-waters")
            }]
        );
        assert_eq!(talk.current(), Some(&node_id("thanks")));
        assert!(
            talk.options(&def, &state).is_empty(),
            "the closing node has nothing left to say"
        );
    }

    #[test]
    fn an_option_leading_nowhere_closes_rather_than_dangles() {
        let broken = def(
            "greeting",
            vec![(
                "greeting",
                node(
                    "...",
                    vec![option("Onward", Vec::new(), Vec::new(), Some("missing"))],
                ),
            )],
        );
        let state = world();
        let mut talk = Conversation::open(&broken);
        talk.choose(&broken, &state, 0).expect("the option is available");
        assert!(talk.is_over());
    }

    #[test]
    fn every_condition_reads_canonical_state() {
        let mut state = world();
        state.player.inventory.insert(item_id("ember-ore"), 3);
        apply_all(
            &mut state,
            [
                // Accepted through the reducer: `QuestState::default()` is
                // `NotStarted`, so inserting one would leave the quest inactive
                // and quietly test the opposite of what this asserts.
                DomainEvent::QuestAccepted {
                    quest: quest_id("still-waters"),
                },
                DomainEvent::WorldFlagSet {
                    flag: flag("forge-relit"),
                },
                DomainEvent::FactionReputationChanged {
                    faction: faction("cinderwake-settlers"),
                    delta: 20,
                },
            ]
            .iter(),
        )
        .expect("applies");

        assert!(evaluate(&Condition::WorldFlagSet(flag("forge-relit")), &state));
        assert!(!evaluate(&Condition::WorldFlagUnset(flag("forge-relit")), &state));
        assert!(evaluate(&Condition::WorldFlagUnset(flag("never-set")), &state));
        assert!(evaluate(
            &Condition::QuestActive(quest_id("still-waters")),
            &state
        ));
        assert!(evaluate(
            &Condition::QuestNotStarted(quest_id("untouched")),
            &state
        ));
        assert!(!evaluate(
            &Condition::QuestCompleted(quest_id("still-waters")),
            &state
        ));
        assert!(evaluate(
            &Condition::ReputationAtLeast {
                faction: faction("cinderwake-settlers"),
                value: 20
            },
            &state
        ));
        assert!(!evaluate(
            &Condition::ReputationAtLeast {
                faction: faction("cinderwake-settlers"),
                value: 21
            },
            &state
        ));
        assert!(evaluate(
            &Condition::HasItem {
                item: item_id("ember-ore"),
                quantity: 3
            },
            &state
        ));
        assert!(!evaluate(
            &Condition::HasItem {
                item: item_id("ember-ore"),
                quantity: 4
            },
            &state
        ));
    }

    #[test]
    fn no_op_effects_produce_no_events() {
        assert_eq!(
            effect_event(&DialogueEffect::AdjustReputation {
                faction: faction("cinderwake-settlers"),
                delta: 0
            }),
            None,
            "a zero delta would log a line that changed nothing"
        );
        assert_eq!(
            effect_event(&DialogueEffect::GrantItem {
                item: item_id("ash-cinder"),
                quantity: 0
            }),
            None,
            "the reducer rejects zero quantities outright"
        );
        assert_eq!(
            effect_event(&DialogueEffect::TakeItem {
                item: item_id("ash-cinder"),
                quantity: 0
            }),
            None
        );
    }

    /// A settled choice travels as a choice, not as an event.
    #[test]
    fn making_a_choice_is_reported_without_inventing_an_event() {
        let choice = ChoiceId::new("choice-water-rights").expect("valid id");
        let def = def(
            "terms",
            vec![(
                "terms",
                node(
                    "Whose claim stands?",
                    vec![option(
                        "The settlers'.",
                        Vec::new(),
                        vec![
                            DialogueEffect::MakeChoice(choice.clone()),
                            DialogueEffect::SetWorldFlag(flag("sided-with-settlers")),
                        ],
                        None,
                    )],
                ),
            )],
        );

        let state = world();
        let mut talk = Conversation::open(&def);
        let turn = talk.choose(&def, &state, 0).expect("the option is available");

        assert_eq!(turn.choices, vec![choice], "the decision is reported");
        assert_eq!(
            turn.events,
            vec![DomainEvent::WorldFlagSet {
                flag: flag("sided-with-settlers")
            }],
            "and it adds no event of its own: the quest objective it completes \
             is what records it durably"
        );
    }

    #[test]
    fn a_conversation_is_deterministic() {
        let def = intake();
        let mut state = world();
        state.player.inventory.insert(item_id("still-reed"), 2);

        let run = || {
            let mut talk = Conversation::open(&def);
            let turn = talk.choose(&def, &state, 1).expect("available");
            (turn, talk)
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn dialogue_returns_events_rather_than_touching_the_world() {
        let def = intake();
        let mut state = world();
        state.player.inventory.insert(item_id("still-reed"), 2);
        let before = state.clone();

        let mut talk = Conversation::open(&def);
        let _ = talk.choose(&def, &state, 1).expect("available");
        assert_eq!(
            BTreeMap::from_iter(state.player.inventory.clone()),
            BTreeMap::from_iter(before.player.inventory),
            "choosing must not mutate canonical state; only applying events does"
        );
    }
}
