//! Builds (ADR-0018): which abilities the player can use, and which five are
//! on the keys.
//!
//! Pure functions of content and canonical state. The loadout is canonical —
//! `Progression::loadout`, changed only by `DomainEvent::LoadoutSet` — and
//! everything here either reads it or proposes the event that changes it,
//! having checked the change first, the way `interaction` checks a craft.
//!
//! "Available" grows over M10: the class's abilities now; talents (wave 2)
//! and items (wave 4) add to it. Whatever the source, an ability that stops
//! being available drops off the keys rather than staying usable.

use content_runtime::ContentRegistry;
use game_core::events::DomainEvent;
use game_core::ids::AbilityId;
use game_core::world::{LOADOUT_SLOTS, WorldState};

/// Five keys' worth of abilities, some perhaps empty.
pub type Loadout = [Option<AbilityId>; LOADOUT_SLOTS];

/// Why a loadout change was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BuildError {
    /// There is no such slot.
    NoSuchSlot(usize),
    /// The player cannot use that ability.
    Unavailable(AbilityId),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSuchSlot(slot) => write!(f, "there is no ability slot {slot}"),
            Self::Unavailable(ability) => write!(f, "{ability} is not available"),
        }
    }
}

impl std::error::Error for BuildError {}

/// Every ability the player may slot, in a stable order: the class's own,
/// in authored order. There is no cap on how many (ADR-0018).
pub fn available(content: &ContentRegistry, state: &WorldState) -> Vec<AbilityId> {
    let mut found: Vec<AbilityId> = Vec::new();
    let class = state
        .player
        .progression
        .class
        .as_ref()
        .and_then(|class| content.classes.get(class));
    for ability in class.iter().flat_map(|class| &class.abilities) {
        if content.abilities.contains_key(ability) && !found.contains(ability) {
            found.push(ability.clone());
        }
    }
    found
}

/// What is on the keys: the recorded loadout, or — until the player first
/// arranges one — the available abilities in order. Anything no longer
/// available reads as an empty slot.
pub fn loadout(content: &ContentRegistry, state: &WorldState) -> Loadout {
    let usable = available(content, state);
    let mut slots: Loadout = Default::default();
    match &state.player.progression.loadout {
        Some(recorded) => {
            for (slot, ability) in slots.iter_mut().zip(recorded) {
                *slot = ability.clone().filter(|ability| usable.contains(ability));
            }
        }
        None => {
            for (slot, ability) in slots.iter_mut().zip(usable) {
                *slot = Some(ability);
            }
        }
    }
    slots
}

/// The event that puts `ability` on key `slot`, or clears it with `None`.
///
/// An ability already on another key moves: the two keys swap, so one
/// ability is never on two keys and nothing slotted is lost by a move.
pub fn assign(
    content: &ContentRegistry,
    state: &WorldState,
    slot: usize,
    ability: Option<AbilityId>,
) -> Result<DomainEvent, BuildError> {
    if slot >= LOADOUT_SLOTS {
        return Err(BuildError::NoSuchSlot(slot));
    }
    if let Some(ability) = &ability
        && !available(content, state).contains(ability)
    {
        return Err(BuildError::Unavailable(ability.clone()));
    }
    let mut slots = loadout(content, state);
    if let Some(ability) = &ability
        && let Some(elsewhere) = slots.iter().position(|held| held.as_ref() == Some(ability))
    {
        slots[elsewhere] = slots[slot].take();
    }
    slots[slot] = ability;
    Ok(DomainEvent::LoadoutSet { slots })
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::content::ClassDef;
    use game_core::events::apply;
    use game_core::ids::{ClassId, ContentRevisionId};

    fn id(name: &str) -> AbilityId {
        AbilityId::new(name).expect("valid id")
    }

    fn content() -> ContentRegistry {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/content");
        content_runtime::load(&root).expect("shipped content loads")
    }

    fn wayfinder(content: &ContentRegistry) -> WorldState {
        let mut state = WorldState::new(1, ContentRevisionId::new("rev").expect("valid id"));
        let class: &ClassDef = &content.classes[&ClassId::new("wayfinder").expect("valid id")];
        apply(
            &mut state,
            &DomainEvent::ClassChosen {
                class: class.id.clone(),
                health: class.health_pool(),
                stamina: class.stamina_pool(),
            },
        )
        .expect("applies");
        state
    }

    #[test]
    fn a_character_who_never_arranged_keys_has_the_class_abilities_in_order() {
        let content = content();
        let state = wayfinder(&content);
        assert_eq!(
            loadout(&content, &state),
            [
                Some(id("ember-cleave")),
                Some(id("warden-guard")),
                Some(id("ember-bolt")),
                None,
                None
            ]
        );
    }

    #[test]
    fn slotting_moves_rather_than_duplicates() {
        let content = content();
        let mut state = wayfinder(&content);
        let event = assign(&content, &state, 4, Some(id("ember-cleave"))).expect("allowed");
        apply(&mut state, &event).expect("applies");
        assert_eq!(
            loadout(&content, &state),
            [
                None,
                Some(id("warden-guard")),
                Some(id("ember-bolt")),
                None,
                Some(id("ember-cleave"))
            ],
            "Q's cleave moved to V, leaving Q empty"
        );

        let event = assign(&content, &state, 1, Some(id("ember-cleave"))).expect("allowed");
        apply(&mut state, &event).expect("applies");
        assert_eq!(
            loadout(&content, &state),
            [
                None,
                Some(id("ember-cleave")),
                Some(id("ember-bolt")),
                None,
                Some(id("warden-guard"))
            ],
            "moving onto an occupied key swaps the two"
        );

        let event = assign(&content, &state, 2, None).expect("clearing is always allowed");
        apply(&mut state, &event).expect("applies");
        assert_eq!(loadout(&content, &state)[2], None);
    }

    #[test]
    fn only_available_abilities_may_be_slotted_or_stay_slotted() {
        let content = content();
        let mut state = wayfinder(&content);
        assert_eq!(
            assign(&content, &state, 0, Some(id("ash-slam"))),
            Err(BuildError::Unavailable(id("ash-slam"))),
            "the ashlord's, not the wayfinder's"
        );
        assert_eq!(
            assign(&content, &state, LOADOUT_SLOTS, None),
            Err(BuildError::NoSuchSlot(LOADOUT_SLOTS))
        );

        // A recorded loadout naming something no longer available reads empty.
        apply(
            &mut state,
            &DomainEvent::LoadoutSet {
                slots: [Some(id("ash-slam")), Some(id("ember-bolt")), None, None, None],
            },
        )
        .expect("the reducer checks ids, not availability");
        assert_eq!(loadout(&content, &state)[0], None);
        assert_eq!(loadout(&content, &state)[1], Some(id("ember-bolt")));
    }
}
