//! Quest graphs, objectives, and reward packages.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::ids::{
    ChoiceId, ClueId, EncounterId, FactionId, ItemId, LocationId, NpcId, ObjectiveId, QuestId,
    QuestResolutionId,
};

/// The allowlisted objectives a quest node may require (PROJECT_CONTEXT §7).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Objective {
    Reach { location: LocationId },
    TalkTo { npc: NpcId },
    Defeat { encounter: EncounterId },
    Collect { item: ItemId, quantity: u16 },
    Escort { npc: NpcId, destination: LocationId },
    Investigate { clue: ClueId },
    Decide { choice: ChoiceId },
}

/// What the player receives on completion. Allowlisted: rewards name authored
/// items, never arbitrary effects.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RewardPackage {
    pub items: BTreeMap<ItemId, u32>,
    pub experience: u32,
    /// Reputation deltas by faction, as bounded integers so replay stays exact.
    pub reputation: BTreeMap<FactionId, i16>,
}

/// One step of a quest graph.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuestNode {
    pub objective: Objective,
    /// Nodes reachable once this one completes. Empty means this node ends the
    /// quest, which is only legal when `resolution` is set.
    pub next: Vec<ObjectiveId>,
    /// Set on terminal nodes: which ending this is. Later content gates on the
    /// specific resolution, not merely on completion.
    pub resolution: Option<QuestResolutionId>,
    /// What the objective tracker says for this step (M9). Without it the
    /// tracker words the objective from its type and the names it refers to
    /// ("Talk to Warden Sela"), which is right far more often than not.
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuestDef {
    pub schema_version: u32,
    pub id: QuestId,
    pub title: String,
    /// Where the quest begins; must exist in `nodes`.
    pub start: ObjectiveId,
    pub nodes: BTreeMap<ObjectiveId, QuestNode>,
    pub rewards: RewardPackage,
}
