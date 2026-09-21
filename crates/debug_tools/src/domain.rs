//! Inspecting canonical domain state.
//!
//! A readable view of what the game believes is durably true: progression,
//! quests, flags, factions, and NPCs. This is the tool for the question that
//! comes up constantly while building content — *did that actually get
//! recorded, or does it only look right on screen?*
//!
//! Every listing is ordered, because canonical state is `BTreeMap`-backed and
//! two inspections of the same world must read identically. A view that
//! reshuffled between frames would be useless for spotting what changed, which
//! is the only reason to look at it.

use game_core::world::{QuestStatus, WorldState};

/// A heading and its lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub title: &'static str,
    pub lines: Vec<String>,
}

impl Section {
    fn new(title: &'static str, lines: Vec<String>) -> Self {
        Self { title, lines }
    }
}

fn describe(status: &QuestStatus) -> String {
    match status {
        QuestStatus::NotStarted => "not started".to_owned(),
        QuestStatus::Active => "active".to_owned(),
        QuestStatus::Completed { resolution } => format!("completed ({resolution})"),
        QuestStatus::Failed => "failed".to_owned(),
    }
}

/// Who the player is and what they are carrying.
pub fn player(state: &WorldState) -> Section {
    let progression = &state.player.progression;
    let mut lines = vec![
        format!(
            "class {}",
            progression
                .class
                .as_ref()
                .map_or_else(|| "unchosen".to_owned(), ToString::to_string)
        ),
        format!("level {} ({} xp)", progression.level, progression.experience),
        format!(
            "health {}/{}",
            state.player.health.current, state.player.health.max
        ),
        format!(
            "stamina {}/{}",
            state.player.stamina.current, state.player.stamina.max
        ),
        format!(
            "at {}",
            state
                .player
                .location
                .as_ref()
                .map_or_else(|| "nowhere".to_owned(), ToString::to_string)
        ),
    ];

    for (item, quantity) in &state.player.inventory {
        lines.push(format!("carries {quantity} x {item}"));
    }
    for (slot, item) in &state.player.equipment {
        lines.push(format!("{slot}: {item}"));
    }

    Section::new("player", lines)
}

/// Quests and their per-objective progress.
pub fn quests(state: &WorldState) -> Section {
    let mut lines = Vec::new();
    for (quest, quest_state) in &state.quests {
        lines.push(format!("{quest}: {}", describe(&quest_state.status)));
        for (objective, progress) in &quest_state.objectives {
            let mark = if progress.completed { "done" } else { "open" };
            lines.push(format!("  {objective}: {mark} ({})", progress.count));
        }
    }
    Section::new("quests", lines)
}

/// Global flags.
pub fn flags(state: &WorldState) -> Section {
    Section::new(
        "world flags",
        state.world_flags.iter().map(ToString::to_string).collect(),
    )
}

/// Standing with each faction.
pub fn factions(state: &WorldState) -> Section {
    Section::new(
        "factions",
        state
            .factions
            .iter()
            .map(|(faction, faction_state)| format!("{faction}: {}", faction_state.reputation.get()))
            .collect(),
    )
}

/// NPCs, whether they are alive, and what they know.
pub fn npcs(state: &WorldState) -> Section {
    Section::new(
        "npcs",
        state
            .npcs
            .iter()
            .map(|(npc, npc_state)| {
                let liveness = if npc_state.alive { "alive" } else { "dead" };
                format!(
                    "{npc}: {liveness}, disposition {}, {} clues",
                    npc_state.disposition.get(),
                    npc_state.known_clues.len()
                )
            })
            .collect(),
    )
}

/// Regions, their discovery state, and their flags.
pub fn regions(state: &WorldState) -> Section {
    let mut lines = Vec::new();
    for (region, region_state) in &state.regions {
        let mut notes = Vec::new();
        if region_state.discovered {
            notes.push("discovered");
        }
        if region_state.cleared {
            notes.push("cleared");
        }
        if notes.is_empty() {
            notes.push("untouched");
        }
        lines.push(format!("{region}: {}", notes.join(", ")));
        for flag in &region_state.flags {
            lines.push(format!("  {flag}"));
        }
    }
    Section::new("regions", lines)
}

/// What identifies this world: its seed and the content revision it runs on.
///
/// Worth showing first, because a bug report without these two values cannot be
/// reproduced (ADR-0013 provenance).
pub fn provenance(state: &WorldState) -> Section {
    Section::new(
        "world",
        vec![
            format!("seed {}", state.world_seed),
            format!("content {}", state.content_revision),
            format!("schema v{}", state.schema_version),
            format!("tick {}", state.clock.tick),
        ],
    )
}

/// Every section, in a fixed order.
pub fn inspect(state: &WorldState) -> Vec<Section> {
    vec![
        provenance(state),
        player(state),
        quests(state),
        flags(state),
        factions(state),
        npcs(state),
        regions(state),
    ]
}

/// Sections as flat text, skipping empty ones.
pub fn lines(sections: &[Section]) -> Vec<String> {
    let mut out = Vec::new();
    for section in sections {
        if section.lines.is_empty() {
            continue;
        }
        out.push(format!("[{}]", section.title));
        out.extend(section.lines.iter().map(|line| format!("  {line}")));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::events::{DomainEvent, RegionChange, apply_all};
    use game_core::ids::{
        ContentRevisionId, FactionId, ItemId, NpcId, ObjectiveId, QuestId, QuestResolutionId, RegionId,
        WorldFlag,
    };
    use game_core::world::NpcState;

    fn world() -> WorldState {
        WorldState::new(0xC1DE, ContentRevisionId::new("rev-0001").expect("valid id"))
    }

    fn quest(name: &str) -> QuestId {
        QuestId::new(name).expect("valid id")
    }

    fn played() -> WorldState {
        let mut state = world();
        state
            .npcs
            .insert(NpcId::new("warden-sela").expect("valid id"), NpcState::default());
        state
            .npcs
            .insert(NpcId::new("arbiter-thal").expect("valid id"), NpcState::default());

        apply_all(
            &mut state,
            [
                DomainEvent::QuestAccepted {
                    quest: quest("still-waters"),
                },
                DomainEvent::QuestObjectiveAdvanced {
                    quest: quest("still-waters"),
                    objective: ObjectiveId::new("gather-reeds").expect("valid id"),
                    count: 2,
                },
                DomainEvent::ItemGranted {
                    item: ItemId::new("still-reed").expect("valid id"),
                    quantity: 2,
                },
                DomainEvent::WorldFlagSet {
                    flag: WorldFlag::new("intake-cleared").expect("valid id"),
                },
                DomainEvent::FactionReputationChanged {
                    faction: FactionId::new("cinderwake-settlers").expect("valid id"),
                    delta: 15,
                },
                DomainEvent::NpcDied {
                    npc: NpcId::new("arbiter-thal").expect("valid id"),
                },
                DomainEvent::RegionStateChanged {
                    region: RegionId::new("cinderwake").expect("valid id"),
                    change: RegionChange::Discovered,
                },
                DomainEvent::ExperienceGained { amount: 120 },
            ]
            .iter(),
        )
        .expect("applies");
        state
    }

    fn section<'a>(sections: &'a [Section], title: &str) -> &'a Section {
        sections
            .iter()
            .find(|section| section.title == title)
            .expect("the inspector always emits this section")
    }

    #[test]
    fn provenance_identifies_the_world() {
        let state = world();
        let lines = provenance(&state).lines;
        assert!(lines.iter().any(|line| line.contains("49630")), "{lines:?}");
        assert!(lines.iter().any(|line| line.contains("rev-0001")));
    }

    #[test]
    fn the_player_section_shows_progression_and_inventory() {
        let sections = inspect(&played());
        let lines = &section(&sections, "player").lines;
        assert!(lines.iter().any(|line| line.contains("120 xp")));
        assert!(lines.iter().any(|line| line == "carries 2 x still-reed"));
        assert!(lines.iter().any(|line| line.starts_with("class unchosen")));
    }

    #[test]
    fn quests_show_status_and_objective_progress() {
        let sections = inspect(&played());
        let lines = &section(&sections, "quests").lines;
        assert_eq!(lines[0], "still-waters: active");
        assert_eq!(lines[1], "  gather-reeds: open (2)");
    }

    #[test]
    fn a_finished_quest_names_its_ending() {
        let mut state = played();
        apply_all(
            &mut state,
            [DomainEvent::QuestCompleted {
                quest: quest("still-waters"),
                resolution: QuestResolutionId::new("intake-cleared").expect("valid id"),
            }]
            .iter(),
        )
        .expect("applies");

        assert_eq!(
            quests(&state).lines[0],
            "still-waters: completed (intake-cleared)",
            "which ending occurred is the part worth inspecting"
        );
    }

    #[test]
    fn npcs_report_liveness() {
        let sections = inspect(&played());
        let lines = &section(&sections, "npcs").lines;
        assert!(lines.iter().any(|line| line.starts_with("arbiter-thal: dead")));
        assert!(lines.iter().any(|line| line.starts_with("warden-sela: alive")));
    }

    #[test]
    fn flags_and_factions_are_listed() {
        let sections = inspect(&played());
        assert_eq!(section(&sections, "world flags").lines, vec!["intake-cleared"]);
        assert_eq!(
            section(&sections, "factions").lines,
            vec!["cinderwake-settlers: 15"]
        );
    }

    #[test]
    fn regions_show_what_has_happened_to_them() {
        let sections = inspect(&played());
        assert_eq!(
            section(&sections, "regions").lines,
            vec!["cinderwake: discovered"]
        );
    }

    #[test]
    fn inspecting_is_deterministic() {
        let state = played();
        assert_eq!(inspect(&state), inspect(&state));
        assert_eq!(lines(&inspect(&state)), lines(&inspect(&state)));
    }

    /// A reloaded save must inspect identically, or the view cannot be trusted
    /// to say whether something was actually recorded.
    #[test]
    fn a_reloaded_world_inspects_identically() {
        let state = played();
        let text = ron::to_string(&state).expect("serializes");
        let reloaded: WorldState = ron::from_str(&text).expect("parses");
        assert_eq!(inspect(&state), inspect(&reloaded));
    }

    #[test]
    fn empty_sections_are_skipped_in_text() {
        let text = lines(&inspect(&world()));
        assert!(
            !text.iter().any(|line| line == "[quests]"),
            "a fresh world has no quests, and an empty heading is noise: {text:?}"
        );
        assert!(text.iter().any(|line| line == "[world]"));
    }

    #[test]
    fn the_inspector_does_not_touch_the_world() {
        let state = played();
        let before = ron::to_string(&state).expect("serializes");
        let _ = inspect(&state);
        assert_eq!(
            before,
            ron::to_string(&state).expect("serializes"),
            "a debug view that mutated what it inspected would be worse than none"
        );
    }
}
