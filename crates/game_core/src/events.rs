//! Domain events and reducers.
//!
//! The only path by which canonical state changes. A [`DomainEvent`] is a
//! record of something that happened; [`apply`] is the pure function that folds
//! one into a [`WorldState`]. Nothing else may mutate durable state.
//!
//! # Why events rather than setters
//!
//! Saving, replaying, rolling back an AI content revision, and (later)
//! replicating to a server all need the same thing: an ordered, serializable
//! record of what changed. Setters scattered through gameplay code cannot give
//! that. With reducers, state at any point equals the seeded initial state
//! folded over its accepted events — which is a property tests can assert, and
//! do (ADR-0013).
//!
//! # Rules every reducer obeys
//!
//! - **Pure and deterministic.** No clocks, no randomness, no I/O. The same
//!   event applied to the same state always produces the same state.
//! - **Atomic.** Preconditions are checked *before* any field is written, so a
//!   rejected event leaves the world exactly as it was. A half-applied event
//!   would desynchronise the log from the state it is supposed to explain.
//! - **Canonical only.** An instance-scoped id (ADR-0010) is rejected rather
//!   than written, because AI-generated names must never reach the persistent
//!   world. [`WorldState::validate_canonical`] is the backstop; this is the
//!   gate.
//! - **Total.** Every failure is a typed [`DomainError`], never a panic.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::ids::{
    AbilityId, ClassId, ClueId, ContentHash, ContentRevisionId, FactionId, IdError, ItemId, LocationId,
    NpcId, ObjectiveId, QuestId, QuestResolutionId, RegionId, WorldFlag,
};
use crate::world::{EquipmentSlot, Pool, QuestState, QuestStatus, WorldState};

/// What changed about a region.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum RegionChange {
    /// The player has seen it for the first time.
    Discovered,
    /// Its set-piece threat is resolved.
    Cleared,
    /// A region-scoped fact became true.
    FlagSet(WorldFlag),
}

/// Something that happened and must survive the session.
///
/// This set is deliberately finite: an event is a promise to keep loading and
/// replaying that shape forever, so adding one is a versioned decision, not a
/// convenience.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum DomainEvent {
    /// In-world time advanced. Separate from wall-clock time, which is never stored.
    ClockAdvanced { ticks: u64 },
    /// The player picked a class (ADR-0007), with the pools it granted.
    ///
    /// The pools travel in the event rather than being looked up, so `apply`
    /// stays a pure function of canonical state and a replayed log reproduces
    /// the character that actually existed even if the class was retuned
    /// afterwards (ADR-0015).
    ClassChosen {
        class: ClassId,
        health: u32,
        stamina: u32,
    },
    /// The player arrived somewhere, as an authored location rather than a transform.
    PlayerMoved { location: LocationId },
    /// The player's health changed by a signed amount, clamped to the pool.
    PlayerHealthChanged { delta: i32 },
    /// Experience was awarded.
    ExperienceGained { amount: u32 },
    /// The player reached a new level. Recorded rather than derived, so retuning
    /// the levelling curve cannot demote someone.
    LeveledUp { level: u16 },
    /// The player arranged their five ability keys (ADR-0018). The whole
    /// loadout travels in one event, so replay never has to reconstruct a
    /// swap from two halves; whether each ability is available to the player
    /// is checked against content before this is raised.
    LoadoutSet {
        slots: [Option<AbilityId>; crate::world::LOADOUT_SLOTS],
    },
    /// Items entered the inventory.
    ItemGranted { item: ItemId, quantity: u32 },
    /// Items left the inventory. Fails rather than clamping if too few are held.
    ItemConsumed { item: ItemId, quantity: u32 },
    /// An item was placed in an equipment slot.
    ItemEquipped { slot: EquipmentSlot, item: ItemId },
    /// A slot was emptied.
    ItemUnequipped { slot: EquipmentSlot },
    /// A quest was accepted.
    QuestAccepted { quest: QuestId },
    /// One objective of an active quest was satisfied.
    QuestObjectiveCompleted { quest: QuestId, objective: ObjectiveId },
    /// Countable progress was made against an objective.
    QuestObjectiveAdvanced {
        quest: QuestId,
        objective: ObjectiveId,
        count: u32,
    },
    /// A quest ended successfully, recording which ending occurred.
    QuestCompleted {
        quest: QuestId,
        resolution: QuestResolutionId,
    },
    /// A quest ended unsuccessfully and cannot be resumed.
    QuestFailed { quest: QuestId },
    /// Standing with a faction moved.
    FactionReputationChanged { faction: FactionId, delta: i16 },
    /// An NPC's personal standing toward the player moved.
    NpcDispositionChanged { npc: NpcId, delta: i16 },
    /// An NPC died. Durable: quests gate on it.
    NpcDied { npc: NpcId },
    /// An NPC learned something the player can later ask about.
    NpcLearnedClue { npc: NpcId, clue: ClueId },
    /// A region changed.
    RegionStateChanged { region: RegionId, change: RegionChange },
    /// A global fact became true.
    WorldFlagSet { flag: WorldFlag },
    /// A compiled content revision was accepted, with the hash it was validated
    /// against (ADR-0013 provenance).
    ContentRevisionAccepted {
        revision: ContentRevisionId,
        hash: ContentHash,
    },
}

/// Why an event could not be applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DomainError {
    /// An instance-scoped id was offered where canonical state is written.
    ScopedId { site: &'static str, source: IdError },
    /// The quest has not been accepted.
    QuestNotActive(QuestId),
    /// The quest already ended; its outcome is immutable.
    QuestAlreadyFinished(QuestId),
    /// Fewer items are held than the event consumes.
    InsufficientItems { item: ItemId, held: u32, requested: u32 },
    /// No such NPC is recorded.
    UnknownNpc(NpcId),
    /// The NPC is already dead.
    NpcAlreadyDead(NpcId),
    /// The in-world clock would wrap.
    ClockOverflow,
    /// A quantity of zero would make the event a no-op that still appears in
    /// the log, which makes replay harder to read and to trust.
    ZeroQuantity { what: &'static str },
}

impl fmt::Display for DomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ScopedId { site, source } => {
                write!(f, "instance-scoped id rejected at {site}: {source}")
            }
            Self::QuestNotActive(quest) => write!(f, "quest {quest} is not active"),
            Self::QuestAlreadyFinished(quest) => write!(f, "quest {quest} has already ended"),
            Self::InsufficientItems {
                item,
                held,
                requested,
            } => write!(f, "cannot consume {requested} of {item}: only {held} held"),
            Self::UnknownNpc(npc) => write!(f, "no such npc: {npc}"),
            Self::NpcAlreadyDead(npc) => write!(f, "npc {npc} is already dead"),
            Self::ClockOverflow => f.write_str("in-world clock would overflow"),
            Self::ZeroQuantity { what } => write!(f, "{what} must be greater than zero"),
        }
    }
}

impl std::error::Error for DomainError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ScopedId { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Rejects an instance-scoped id before it can enter canonical state.
macro_rules! canonical {
    ($id:expr, $site:literal) => {
        $id.require_canonical()
            .map_err(|source| DomainError::ScopedId { site: $site, source })?
    };
}

/// Folds one event into the world.
///
/// On `Err` the world is untouched: every precondition is checked before the
/// first write.
pub fn apply(state: &mut WorldState, event: &DomainEvent) -> Result<(), DomainError> {
    match event {
        DomainEvent::ClockAdvanced { ticks } => {
            let next = state
                .clock
                .tick
                .checked_add(*ticks)
                .ok_or(DomainError::ClockOverflow)?;
            state.clock.tick = next;
        }

        DomainEvent::ClassChosen {
            class,
            health,
            stamina,
        } => {
            canonical!(class, "class");
            if *health == 0 || *stamina == 0 {
                return Err(DomainError::ZeroQuantity { what: "class pools" });
            }
            state.player.progression.class = Some(class.clone());
            state.player.health = Pool::full(*health);
            state.player.stamina = Pool::full(*stamina);
        }

        DomainEvent::PlayerMoved { location } => {
            canonical!(location, "player location");
            state.player.location = Some(location.clone());
        }

        DomainEvent::PlayerHealthChanged { delta } => {
            let health = &mut state.player.health;
            let current = i64::from(health.current) + i64::from(*delta);
            let clamped = current.clamp(0, i64::from(health.max));
            // The cast is safe: `clamped` is within `0..=max`, and `max` is u32.
            health.current = clamped as u32;
        }

        DomainEvent::ExperienceGained { amount } => {
            if *amount == 0 {
                return Err(DomainError::ZeroQuantity { what: "experience" });
            }
            state.player.progression.experience = state.player.progression.experience.saturating_add(*amount);
        }

        DomainEvent::LoadoutSet { slots } => {
            for ability in slots.iter().flatten() {
                canonical!(ability, "loadout ability");
            }
            state.player.progression.loadout = Some(slots.clone());
        }

        DomainEvent::LeveledUp { level } => {
            state.player.progression.level = *level;
        }

        DomainEvent::ItemGranted { item, quantity } => {
            canonical!(item, "granted item");
            if *quantity == 0 {
                return Err(DomainError::ZeroQuantity {
                    what: "item quantity",
                });
            }
            let held = state.player.inventory.entry(item.clone()).or_insert(0);
            *held = held.saturating_add(*quantity);
        }

        DomainEvent::ItemConsumed { item, quantity } => {
            if *quantity == 0 {
                return Err(DomainError::ZeroQuantity {
                    what: "item quantity",
                });
            }
            let held = state.player.item_count(item);
            if held < *quantity {
                return Err(DomainError::InsufficientItems {
                    item: item.clone(),
                    held,
                    requested: *quantity,
                });
            }
            // Zero quantities are removed rather than stored, so "none held"
            // has exactly one representation and equality stays meaningful.
            if held == *quantity {
                state.player.inventory.remove(item);
            } else {
                state.player.inventory.insert(item.clone(), held - quantity);
            }
        }

        DomainEvent::ItemEquipped { slot, item } => {
            canonical!(item, "equipped item");
            state.player.equipment.insert(*slot, item.clone());
        }

        DomainEvent::ItemUnequipped { slot } => {
            state.player.equipment.remove(slot);
        }

        DomainEvent::QuestAccepted { quest } => {
            canonical!(quest, "quest");
            if let Some(existing) = state.quests.get(quest)
                && existing.status.is_terminal()
            {
                return Err(DomainError::QuestAlreadyFinished(quest.clone()));
            }
            state.quests.entry(quest.clone()).or_default().status = QuestStatus::Active;
        }

        DomainEvent::QuestObjectiveCompleted { quest, objective } => {
            canonical!(objective, "objective");
            let state_for_quest = active_quest_mut(state, quest)?;
            let progress = state_for_quest.objectives.entry(objective.clone()).or_default();
            progress.completed = true;
        }

        DomainEvent::QuestObjectiveAdvanced {
            quest,
            objective,
            count,
        } => {
            canonical!(objective, "objective");
            if *count == 0 {
                return Err(DomainError::ZeroQuantity {
                    what: "objective progress",
                });
            }
            let state_for_quest = active_quest_mut(state, quest)?;
            let progress = state_for_quest.objectives.entry(objective.clone()).or_default();
            progress.count = progress.count.saturating_add(*count);
        }

        DomainEvent::QuestCompleted { quest, resolution } => {
            canonical!(resolution, "quest resolution");
            let state_for_quest = active_quest_mut(state, quest)?;
            state_for_quest.status = QuestStatus::Completed {
                resolution: resolution.clone(),
            };
        }

        DomainEvent::QuestFailed { quest } => {
            let state_for_quest = active_quest_mut(state, quest)?;
            state_for_quest.status = QuestStatus::Failed;
        }

        DomainEvent::FactionReputationChanged { faction, delta } => {
            canonical!(faction, "faction");
            let entry = state.factions.entry(faction.clone()).or_default();
            entry.reputation = entry.reputation.saturating_add(*delta);
        }

        DomainEvent::NpcDispositionChanged { npc, delta } => {
            let npc_state = state
                .npcs
                .get_mut(npc)
                .ok_or_else(|| DomainError::UnknownNpc(npc.clone()))?;
            npc_state.disposition = npc_state.disposition.saturating_add(*delta);
        }

        DomainEvent::NpcDied { npc } => {
            let npc_state = state
                .npcs
                .get_mut(npc)
                .ok_or_else(|| DomainError::UnknownNpc(npc.clone()))?;
            if !npc_state.alive {
                return Err(DomainError::NpcAlreadyDead(npc.clone()));
            }
            npc_state.alive = false;
        }

        DomainEvent::NpcLearnedClue { npc, clue } => {
            canonical!(clue, "clue");
            let npc_state = state
                .npcs
                .get_mut(npc)
                .ok_or_else(|| DomainError::UnknownNpc(npc.clone()))?;
            npc_state.known_clues.insert(clue.clone());
        }

        DomainEvent::RegionStateChanged { region, change } => {
            canonical!(region, "region");
            if let RegionChange::FlagSet(flag) = change {
                canonical!(flag, "region flag");
            }
            let region_state = state.regions.entry(region.clone()).or_default();
            match change {
                RegionChange::Discovered => region_state.discovered = true,
                RegionChange::Cleared => region_state.cleared = true,
                RegionChange::FlagSet(flag) => {
                    region_state.flags.insert(flag.clone());
                }
            }
        }

        DomainEvent::WorldFlagSet { flag } => {
            canonical!(flag, "world flag");
            state.world_flags.insert(flag.clone());
        }

        DomainEvent::ContentRevisionAccepted { revision, hash: _ } => {
            canonical!(revision, "content revision");
            state.content_revision = revision.clone();
        }
    }
    Ok(())
}

/// Folds a sequence of events, stopping at the first failure.
///
/// The world is left with every event before the failure applied, which is what
/// replay wants: the log up to the bad entry is still history.
pub fn apply_all<'a>(
    state: &mut WorldState,
    events: impl IntoIterator<Item = &'a DomainEvent>,
) -> Result<(), DomainError> {
    for event in events {
        apply(state, event)?;
    }
    Ok(())
}

/// Borrows a quest that is accepted and unfinished.
fn active_quest_mut<'a>(
    state: &'a mut WorldState,
    quest: &QuestId,
) -> Result<&'a mut QuestState, DomainError> {
    let entry = state
        .quests
        .get_mut(quest)
        .ok_or_else(|| DomainError::QuestNotActive(quest.clone()))?;
    match entry.status {
        QuestStatus::Active => Ok(entry),
        QuestStatus::NotStarted => Err(DomainError::QuestNotActive(quest.clone())),
        QuestStatus::Completed { .. } | QuestStatus::Failed => {
            Err(DomainError::QuestAlreadyFinished(quest.clone()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::InstanceId;
    use crate::world::{NpcState, Pool, Reputation};

    fn revision() -> ContentRevisionId {
        ContentRevisionId::new("rev-0001").expect("valid id")
    }

    fn world() -> WorldState {
        let mut state = WorldState::new(42, revision());
        state.player.health = Pool::full(100);
        state
            .npcs
            .insert(NpcId::new("warden-sela").expect("valid id"), NpcState::default());
        state
    }

    fn quest() -> QuestId {
        QuestId::new("still-waters").expect("valid id")
    }

    fn objective() -> ObjectiveId {
        ObjectiveId::new("meet-sela").expect("valid id")
    }

    fn instance() -> InstanceId {
        InstanceId::new("crypt-of-ash").expect("valid id")
    }

    #[test]
    fn quest_lifecycle() {
        let mut state = world();
        let accept = DomainEvent::QuestAccepted { quest: quest() };
        apply(&mut state, &accept).expect("accepting a quest succeeds");
        assert_eq!(state.quest_status(&quest()), QuestStatus::Active);

        apply(
            &mut state,
            &DomainEvent::QuestObjectiveCompleted {
                quest: quest(),
                objective: objective(),
            },
        )
        .expect("completing an objective succeeds");
        assert!(state.objective_progress(&quest(), &objective()).completed);

        let resolution = QuestResolutionId::new("waters-restored").expect("valid id");
        apply(
            &mut state,
            &DomainEvent::QuestCompleted {
                quest: quest(),
                resolution: resolution.clone(),
            },
        )
        .expect("completing a quest succeeds");
        assert_eq!(
            state.quest_status(&quest()),
            QuestStatus::Completed { resolution }
        );

        // A finished quest's outcome is immutable.
        assert_eq!(
            apply(&mut state, &accept),
            Err(DomainError::QuestAlreadyFinished(quest()))
        );
    }

    #[test]
    fn objectives_require_an_active_quest() {
        let mut state = world();
        assert_eq!(
            apply(
                &mut state,
                &DomainEvent::QuestObjectiveCompleted {
                    quest: quest(),
                    objective: objective(),
                }
            ),
            Err(DomainError::QuestNotActive(quest()))
        );
        assert!(state.quests.is_empty(), "a rejected event writes nothing");
    }

    #[test]
    fn inventory_grants_and_consumes_exactly() {
        let mut state = world();
        let item = ItemId::new("ember-ore").expect("valid id");
        apply(
            &mut state,
            &DomainEvent::ItemGranted {
                item: item.clone(),
                quantity: 5,
            },
        )
        .expect("granting succeeds");
        assert_eq!(state.player.item_count(&item), 5);

        assert_eq!(
            apply(
                &mut state,
                &DomainEvent::ItemConsumed {
                    item: item.clone(),
                    quantity: 9,
                }
            ),
            Err(DomainError::InsufficientItems {
                item: item.clone(),
                held: 5,
                requested: 9,
            })
        );
        assert_eq!(
            state.player.item_count(&item),
            5,
            "a failed consume takes nothing"
        );

        apply(
            &mut state,
            &DomainEvent::ItemConsumed {
                item: item.clone(),
                quantity: 5,
            },
        )
        .expect("consuming all succeeds");
        assert!(
            !state.player.inventory.contains_key(&item),
            "an emptied stack is removed, not stored as zero"
        );
    }

    #[test]
    fn health_clamps_to_the_pool() {
        let mut state = world();
        apply(&mut state, &DomainEvent::PlayerHealthChanged { delta: -250 }).expect("applies");
        assert_eq!(state.player.health.current, 0, "health floors at zero");

        apply(&mut state, &DomainEvent::PlayerHealthChanged { delta: 10_000 }).expect("applies");
        assert_eq!(
            state.player.health.current, state.player.health.max,
            "and caps at max"
        );
    }

    #[test]
    fn reputation_saturates_rather_than_wrapping() {
        let mut state = world();
        let faction = FactionId::new("cinderwake-settlers").expect("valid id");
        for _ in 0..10 {
            apply(
                &mut state,
                &DomainEvent::FactionReputationChanged {
                    faction: faction.clone(),
                    delta: i16::MAX,
                },
            )
            .expect("applies");
        }
        assert_eq!(state.reputation(&faction), Reputation::MAX);
    }

    #[test]
    fn npc_events_require_a_known_npc() {
        let mut state = world();
        let stranger = NpcId::new("nobody-at-all").expect("valid id");
        assert_eq!(
            apply(
                &mut state,
                &DomainEvent::NpcDied {
                    npc: stranger.clone()
                }
            ),
            Err(DomainError::UnknownNpc(stranger))
        );

        let sela = NpcId::new("warden-sela").expect("valid id");
        apply(&mut state, &DomainEvent::NpcDied { npc: sela.clone() }).expect("applies");
        assert_eq!(
            apply(&mut state, &DomainEvent::NpcDied { npc: sela.clone() }),
            Err(DomainError::NpcAlreadyDead(sela))
        );
    }

    #[test]
    fn scoped_ids_are_rejected_before_they_reach_canonical_state() {
        let mut state = world();
        let scoped_flag = WorldFlag::scoped(&instance(), "leaked").expect("valid scoped id");
        let error = apply(&mut state, &DomainEvent::WorldFlagSet { flag: scoped_flag });
        assert!(matches!(error, Err(DomainError::ScopedId { .. })));
        assert!(state.world_flags.is_empty(), "nothing was written");

        let scoped_item = ItemId::scoped(&instance(), "loot").expect("valid scoped id");
        assert!(matches!(
            apply(
                &mut state,
                &DomainEvent::ItemGranted {
                    item: scoped_item,
                    quantity: 1
                }
            ),
            Err(DomainError::ScopedId { .. })
        ));
        assert!(state.player.inventory.is_empty());
        state
            .validate_canonical()
            .expect("state stayed canonical throughout");
    }

    #[test]
    fn clock_advances_and_refuses_to_wrap() {
        let mut state = world();
        apply(&mut state, &DomainEvent::ClockAdvanced { ticks: 120 }).expect("applies");
        assert_eq!(state.clock.tick, 120);
        assert_eq!(
            apply(&mut state, &DomainEvent::ClockAdvanced { ticks: u64::MAX }),
            Err(DomainError::ClockOverflow)
        );
        assert_eq!(state.clock.tick, 120, "a rejected event leaves the clock alone");
    }

    /// The property the whole event-sourced design rests on: folding the same
    /// log over the same start state twice yields the same world.
    #[test]
    fn replay_is_deterministic() {
        let log = vec![
            DomainEvent::ClockAdvanced { ticks: 900 },
            DomainEvent::ClassChosen {
                class: ClassId::new("wayfinder").expect("valid id"),
                health: 120,
                stamina: 100,
            },
            DomainEvent::QuestAccepted { quest: quest() },
            DomainEvent::QuestObjectiveAdvanced {
                quest: quest(),
                objective: objective(),
                count: 2,
            },
            DomainEvent::ItemGranted {
                item: ItemId::new("ember-ore").expect("valid id"),
                quantity: 3,
            },
            DomainEvent::FactionReputationChanged {
                faction: FactionId::new("cinderwake-settlers").expect("valid id"),
                delta: 12,
            },
            DomainEvent::RegionStateChanged {
                region: RegionId::new("cinderwake").expect("valid id"),
                change: RegionChange::Discovered,
            },
            DomainEvent::WorldFlagSet {
                flag: WorldFlag::new("prologue-complete").expect("valid id"),
            },
        ];

        let mut first = world();
        apply_all(&mut first, &log).expect("log applies");
        let mut second = world();
        apply_all(&mut second, &log).expect("log applies");
        assert_eq!(first, second);

        // And the serialized form matches too, which is what makes save
        // comparison and content hashing meaningful.
        assert_eq!(
            ron::to_string(&first).expect("serializes"),
            ron::to_string(&second).expect("serializes")
        );
    }

    #[test]
    fn events_round_trip_through_ron() {
        let event = DomainEvent::QuestCompleted {
            quest: quest(),
            resolution: QuestResolutionId::new("waters-restored").expect("valid id"),
        };
        let text = ron::to_string(&event).expect("serializes");
        assert_eq!(ron::from_str::<DomainEvent>(&text).expect("parses"), event);
    }

    #[test]
    fn zero_quantities_are_rejected_as_no_ops() {
        let mut state = world();
        assert!(matches!(
            apply(
                &mut state,
                &DomainEvent::ItemGranted {
                    item: ItemId::new("ember-ore").expect("valid id"),
                    quantity: 0
                }
            ),
            Err(DomainError::ZeroQuantity { .. })
        ));
        assert!(matches!(
            apply(&mut state, &DomainEvent::ExperienceGained { amount: 0 }),
            Err(DomainError::ZeroQuantity { .. })
        ));
    }

    #[test]
    fn equipment_slots_hold_one_item() {
        let mut state = world();
        let glaive = ItemId::new("ember-glaive").expect("valid id");
        apply(
            &mut state,
            &DomainEvent::ItemEquipped {
                slot: EquipmentSlot::Weapon,
                item: glaive.clone(),
            },
        )
        .expect("applies");
        assert_eq!(state.player.equipped(EquipmentSlot::Weapon), Some(&glaive));

        apply(
            &mut state,
            &DomainEvent::ItemUnequipped {
                slot: EquipmentSlot::Weapon,
            },
        )
        .expect("applies");
        assert_eq!(state.player.equipped(EquipmentSlot::Weapon), None);
    }
}
