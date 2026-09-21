//! Conversations, their conditions, and their consequences.
//!
//! Dialogue is where content most wants to become a scripting language, and
//! where it most must not. A line may be *gated* on world state and may *cause*
//! allowlisted effects, but it cannot express arbitrary logic — no expressions,
//! no embedded code, no escape hatch. That restraint is what will later let a
//! model propose dialogue safely (ADR-0010): the worst a bad proposal can do is
//! reference something that does not exist, which validation catches.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::ids::{
    ChoiceId, DialogueId, DialogueNodeId, FactionId, ItemId, NpcId, ObjectiveId, QuestId, WorldFlag,
};

/// A test against canonical world state.
///
/// Every variant is answerable from `WorldState` alone, which is what keeps
/// dialogue deterministic and replayable.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Condition {
    WorldFlagSet(WorldFlag),
    WorldFlagUnset(WorldFlag),
    QuestNotStarted(QuestId),
    QuestActive(QuestId),
    QuestCompleted(QuestId),
    /// Standing with a faction is at least `value`, on `Reputation`'s scale.
    ReputationAtLeast {
        faction: FactionId,
        value: i16,
    },
    HasItem {
        item: ItemId,
        quantity: u32,
    },
}

/// What choosing an option does.
///
/// These become domain events when the option is taken; the runtime does not
/// apply them directly, so everything dialogue changes is logged and replayable
/// like any other durable change.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum DialogueEffect {
    SetWorldFlag(WorldFlag),
    AdjustReputation {
        faction: FactionId,
        delta: i16,
    },
    StartQuest(QuestId),
    CompleteObjective {
        quest: QuestId,
        objective: ObjectiveId,
    },
    GrantItem {
        item: ItemId,
        quantity: u32,
    },
    TakeItem {
        item: ItemId,
        quantity: u32,
    },
    /// Settles an authored decision.
    ///
    /// Deliberately *not* a domain event of its own: a settled choice completes
    /// a quest's `Decide` objective, and that completion is already recorded
    /// durably. A second record of the same fact is a second thing that can
    /// disagree with the first.
    MakeChoice(ChoiceId),
}

/// One thing the player can say.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DialogueOption {
    pub text: String,
    /// All must hold for the option to be offered. An empty list always shows.
    pub conditions: Vec<Condition>,
    pub effects: Vec<DialogueEffect>,
    /// Where the conversation goes next; `None` ends it.
    pub next: Option<DialogueNodeId>,
}

/// One thing the NPC says, and the replies available.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DialogueNode {
    pub text: String,
    pub options: Vec<DialogueOption>,
}

impl DialogueNode {
    /// Whether this node can end the conversation.
    ///
    /// A node with no options at all is terminal: the NPC says their piece and
    /// the conversation closes.
    pub fn can_end(&self) -> bool {
        self.options.is_empty() || self.options.iter().any(|option| option.next.is_none())
    }
}

/// A whole conversation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DialogueDef {
    pub schema_version: u32,
    pub id: DialogueId,
    /// Who is speaking. One NPC may own several conversations.
    pub speaker: NpcId,
    pub start: DialogueNodeId,
    pub nodes: BTreeMap<DialogueNodeId, DialogueNode>,
    /// All must hold for this conversation to be open at all (M9). An empty
    /// list is always open. Gating the whole conversation, rather than every
    /// line in it, is what lets one NPC hold a quest's opening and its
    /// hand-in as two conversations instead of one tangle.
    #[serde(default)]
    pub conditions: Vec<Condition>,
    /// What the player picks to open this conversation when its speaker has
    /// more than one open at once. Required whenever a speaker owns several —
    /// the loader refuses two untitled conversations for one NPC, because
    /// the player would have no way to tell them apart.
    #[serde(default)]
    pub topic: Option<String>,
}

impl DialogueDef {
    /// Node ids reachable from the start, following every option.
    ///
    /// Conditions are ignored here on purpose: reachability is about the graph,
    /// and a node reachable only under some world state is still authored
    /// content that must exist.
    pub fn reachable(&self) -> BTreeMap<&DialogueNodeId, ()> {
        let mut seen = BTreeMap::new();
        let mut frontier = vec![&self.start];
        while let Some(current) = frontier.pop() {
            let Some((key, node)) = self.nodes.get_key_value(current) else {
                continue;
            };
            if seen.insert(key, ()).is_some() {
                continue;
            }
            for option in &node.options {
                if let Some(next) = &option.next {
                    frontier.push(next);
                }
            }
        }
        seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node_id(name: &str) -> DialogueNodeId {
        DialogueNodeId::new(name).expect("valid id")
    }

    fn say(text: &str, options: Vec<DialogueOption>) -> DialogueNode {
        DialogueNode {
            text: text.to_owned(),
            options,
        }
    }

    fn reply(text: &str, next: Option<&str>) -> DialogueOption {
        DialogueOption {
            text: text.to_owned(),
            conditions: Vec::new(),
            effects: Vec::new(),
            next: next.map(node_id),
        }
    }

    fn conversation(nodes: Vec<(&str, DialogueNode)>) -> DialogueDef {
        DialogueDef {
            schema_version: super::super::CONTENT_SCHEMA_VERSION,
            id: DialogueId::new("sela-greeting").expect("valid id"),
            speaker: NpcId::new("warden-sela").expect("valid id"),
            conditions: Vec::new(),
            topic: None,
            start: node_id("greeting"),
            nodes: nodes.into_iter().map(|(id, node)| (node_id(id), node)).collect(),
        }
    }

    #[test]
    fn reachability_follows_every_option() {
        let dialogue = conversation(vec![
            (
                "greeting",
                say(
                    "The intake is fouled.",
                    vec![reply("Tell me more", Some("detail")), reply("Later", None)],
                ),
            ),
            (
                "detail",
                say("Corruption, upstream.", vec![reply("I will look", None)]),
            ),
            ("orphan", say("Nobody reaches this line.", vec![])),
        ]);

        let reachable = dialogue.reachable();
        assert_eq!(reachable.len(), 2, "the orphaned node is not reachable");
        assert!(!reachable.contains_key(&node_id("orphan")));
    }

    #[test]
    fn a_conversation_must_be_escapable() {
        let ending = say("Farewell.", vec![reply("Goodbye", None)]);
        assert!(ending.can_end());

        let silent = say("...", vec![]);
        assert!(silent.can_end(), "a node with no replies closes the conversation");

        let trap = say("Stay.", vec![reply("Wait", Some("greeting"))]);
        assert!(!trap.can_end(), "a node whose every option loops cannot end");
    }

    #[test]
    fn conditions_and_effects_round_trip() {
        let option = DialogueOption {
            text: "Here are the reeds you asked for.".to_owned(),
            conditions: vec![
                Condition::QuestActive(QuestId::new("still-waters").expect("valid id")),
                Condition::HasItem {
                    item: ItemId::new("still-reed").expect("valid id"),
                    quantity: 2,
                },
            ],
            effects: vec![
                DialogueEffect::TakeItem {
                    item: ItemId::new("still-reed").expect("valid id"),
                    quantity: 2,
                },
                DialogueEffect::AdjustReputation {
                    faction: FactionId::new("cinderwake-settlers").expect("valid id"),
                    delta: 5,
                },
            ],
            next: None,
        };

        let text = ron::to_string(&option).expect("serializes");
        assert_eq!(ron::from_str::<DialogueOption>(&text).expect("parses"), option);
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let text = r#"(text: "Hello", conditions: [], effects: [], next: None, mood: "warm")"#;
        assert!(
            ron::from_str::<DialogueOption>(text).is_err(),
            "a stray field must fail the parse rather than be ignored"
        );
    }
}
