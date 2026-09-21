//! Gathering, crafting, settlement upgrades, and loot.
//!
//! The verbs that turn exploration into progression. Like the narrative
//! runtimes, everything here is a pure function returning [`DomainEvent`]s for
//! the caller to apply — inventory and world flags are durable, so they change
//! only through the log.
//!
//! # One rule, two enforcement points
//!
//! These functions check affordability *before* producing events, so a UI can
//! grey out a recipe the player cannot afford. The reducer checks again when the
//! events apply, and returns `DomainError::InsufficientItems` if it disagrees.
//! That is deliberate: the check is expressed once, here, as "what does this
//! cost"; the reducer's check is the authority. Two independent implementations
//! of the *rule* would be the bug — two checks of the same rule is just defence.
//!
//! # Loot is rolled from the log, not from a clock
//!
//! [`loot_seed`] mixes the world seed, the in-world tick, and the loot table's
//! id. All three are recoverable from canonical state and the event log, so
//! replaying a kill drops exactly the same items. Seeding from wall-clock time,
//! a thread-local RNG, or a runtime entity id would each make replay — and
//! therefore save verification — quietly meaningless.

use std::collections::BTreeMap;

use content_runtime::ContentRegistry;
use game_core::content::{LootTableDef, RecipeDef, SettlementUpgradeDef};
use game_core::events::DomainEvent;
use game_core::ids::{ItemId, LootTableId, UpgradeId};
use game_core::world::WorldState;

/// Why an interaction could not be performed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InteractionError {
    /// Too few of an input are held. Reports the first shortfall in id order, so
    /// the message is the same on every machine.
    MissingInputs { item: ItemId, held: u32, needed: u32 },
    /// A recipe or blueprint names an upgrade that content does not define.
    UnknownUpgrade(UpgradeId),
    /// A required blueprint has not been built yet.
    UpgradeNotBuilt(UpgradeId),
    /// The blueprint is already standing.
    AlreadyBuilt(UpgradeId),
}

impl std::fmt::Display for InteractionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingInputs { item, held, needed } => {
                write!(f, "needs {needed} of {item}, holds {held}")
            }
            Self::UnknownUpgrade(upgrade) => write!(f, "no such upgrade: {upgrade}"),
            Self::UpgradeNotBuilt(upgrade) => write!(f, "upgrade {upgrade} has not been built"),
            Self::AlreadyBuilt(upgrade) => write!(f, "upgrade {upgrade} is already built"),
        }
    }
}

impl std::error::Error for InteractionError {}

/// Whether the player holds everything in `costs`.
fn afford(costs: &BTreeMap<ItemId, u32>, state: &WorldState) -> Result<(), InteractionError> {
    for (item, needed) in costs {
        let held = state.player.item_count(item);
        if held < *needed {
            return Err(InteractionError::MissingInputs {
                item: item.clone(),
                held,
                needed: *needed,
            });
        }
    }
    Ok(())
}

/// The consumption events for a cost table, in id order.
fn spend(costs: &BTreeMap<ItemId, u32>) -> Vec<DomainEvent> {
    costs
        .iter()
        .filter(|(_, quantity)| **quantity > 0)
        .map(|(item, quantity)| DomainEvent::ItemConsumed {
            item: item.clone(),
            quantity: *quantity,
        })
        .collect()
}

/// Whether the upgrade that `required` names has been built.
///
/// An upgrade is "built" when its `grants_flag` is set — there is deliberately
/// no separate list of built upgrades that could disagree with the flags the
/// rest of the game gates on.
fn built(
    required: &UpgradeId,
    content: &ContentRegistry,
    state: &WorldState,
) -> Result<bool, InteractionError> {
    let upgrade = content
        .upgrades
        .get(required)
        .ok_or_else(|| InteractionError::UnknownUpgrade(required.clone()))?;
    Ok(state.has_world_flag(&upgrade.grants_flag))
}

/// Harvesting a gatherable node.
///
/// Returns no event for a zero yield: the reducer rejects a zero quantity, and a
/// pickup of nothing should not appear in the log.
pub fn gather(item: &ItemId, quantity: u32) -> Vec<DomainEvent> {
    if quantity == 0 {
        return Vec::new();
    }
    vec![DomainEvent::ItemGranted {
        item: item.clone(),
        quantity,
    }]
}

/// Crafting a recipe: inputs consumed, output granted.
pub fn craft(
    recipe: &RecipeDef,
    content: &ContentRegistry,
    state: &WorldState,
) -> Result<Vec<DomainEvent>, InteractionError> {
    if let Some(required) = &recipe.requires_upgrade
        && !built(required, content, state)?
    {
        return Err(InteractionError::UpgradeNotBuilt(required.clone()));
    }
    afford(&recipe.inputs, state)?;

    let mut events = spend(&recipe.inputs);
    if recipe.output_quantity > 0 {
        events.push(DomainEvent::ItemGranted {
            item: recipe.output.clone(),
            quantity: recipe.output_quantity,
        });
    }
    Ok(events)
}

/// Building a settlement blueprint: costs consumed, world flag set.
///
/// The flag is the whole point (ADR-0007). Services, dialogue, and recipes gate
/// on it, so building an upgrade and the world noticing are the same event.
pub fn build_upgrade(
    upgrade: &SettlementUpgradeDef,
    content: &ContentRegistry,
    state: &WorldState,
) -> Result<Vec<DomainEvent>, InteractionError> {
    if state.has_world_flag(&upgrade.grants_flag) {
        return Err(InteractionError::AlreadyBuilt(upgrade.id.clone()));
    }
    for required in &upgrade.requires {
        if !built(required, content, state)? {
            return Err(InteractionError::UpgradeNotBuilt(required.clone()));
        }
    }
    afford(&upgrade.costs, state)?;

    let mut events = spend(&upgrade.costs);
    events.push(DomainEvent::WorldFlagSet {
        flag: upgrade.grants_flag.clone(),
    });
    Ok(events)
}

/// A reproducible seed for one loot roll.
///
/// Mixes the world seed, the in-world tick, and the table id. The tick makes two
/// kills from the same table at different moments differ; the table id makes two
/// tables at the same moment differ; the world seed makes two saves differ. All
/// three replay exactly.
pub fn loot_seed(state: &WorldState, table: &LootTableId) -> u64 {
    // FNV-1a over the id, then a SplitMix64 finalizer over the mixture. Neither
    // needs to be strong — only reproducible on every platform.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in table.to_string().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }

    let mut mixed = state
        .world_seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(state.clock.tick)
        ^ hash;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

/// Rolls a loot table into grant events, in item-id order.
pub fn loot(table: &LootTableDef, seed: u64) -> Vec<DomainEvent> {
    table
        .roll(seed)
        .into_iter()
        .filter(|(_, quantity)| *quantity > 0)
        .map(|(item, quantity)| DomainEvent::ItemGranted { item, quantity })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::content::{CONTENT_SCHEMA_VERSION, LootEntry};
    use game_core::events::apply_all;
    use game_core::ids::{ContentRevisionId, WorldFlag};

    fn item(name: &str) -> ItemId {
        ItemId::new(name).expect("valid id")
    }

    fn upgrade_id(name: &str) -> UpgradeId {
        UpgradeId::new(name).expect("valid id")
    }

    fn flag(name: &str) -> WorldFlag {
        WorldFlag::new(name).expect("valid id")
    }

    fn world() -> WorldState {
        WorldState::new(0xC1DE, ContentRevisionId::new("rev-0001").expect("valid id"))
    }

    fn forge() -> SettlementUpgradeDef {
        SettlementUpgradeDef {
            schema_version: CONTENT_SCHEMA_VERSION,
            id: upgrade_id("upgrade-forge"),
            name: "Rebuild the Forge".to_owned(),
            costs: BTreeMap::from([(item("ember-ore"), 6), (item("old-forest-bark"), 2)]),
            requires: Vec::new(),
            grants_flag: flag("forge-relit"),
        }
    }

    fn anvil() -> SettlementUpgradeDef {
        SettlementUpgradeDef {
            schema_version: CONTENT_SCHEMA_VERSION,
            id: upgrade_id("upgrade-anvil"),
            name: "Set the Anvil".to_owned(),
            costs: BTreeMap::from([(item("ember-ore"), 4)]),
            requires: vec![upgrade_id("upgrade-forge")],
            grants_flag: flag("anvil-set"),
        }
    }

    fn blade() -> RecipeDef {
        RecipeDef {
            schema_version: CONTENT_SCHEMA_VERSION,
            id: item("recipe-ember-blade"),
            name: "Ember Blade".to_owned(),
            inputs: BTreeMap::from([(item("ember-ore"), 3), (item("still-reed"), 1)]),
            output: item("ember-blade"),
            output_quantity: 1,
            requires_upgrade: Some(upgrade_id("upgrade-forge")),
        }
    }

    fn registry() -> ContentRegistry {
        let mut content = ContentRegistry::default();
        content.upgrades.insert(upgrade_id("upgrade-forge"), forge());
        content.upgrades.insert(upgrade_id("upgrade-anvil"), anvil());
        content
    }

    fn stocked() -> WorldState {
        let mut state = world();
        state.player.inventory.insert(item("ember-ore"), 10);
        state.player.inventory.insert(item("old-forest-bark"), 4);
        state.player.inventory.insert(item("still-reed"), 2);
        state
    }

    #[test]
    fn gathering_grants_what_was_picked_up() {
        assert_eq!(
            gather(&item("still-reed"), 2),
            vec![DomainEvent::ItemGranted {
                item: item("still-reed"),
                quantity: 2
            }]
        );
        assert!(
            gather(&item("still-reed"), 0).is_empty(),
            "a pickup of nothing must not reach the log"
        );
    }

    #[test]
    fn crafting_requires_its_blueprint() {
        let state = stocked();
        assert_eq!(
            craft(&blade(), &registry(), &state),
            Err(InteractionError::UpgradeNotBuilt(upgrade_id("upgrade-forge"))),
            "materials alone are not enough; the forge has to be standing"
        );
    }

    #[test]
    fn crafting_consumes_inputs_and_grants_the_output() {
        let mut state = stocked();
        state.world_flags.insert(flag("forge-relit"));

        let events = craft(&blade(), &registry(), &state).expect("affordable and unlocked");
        assert_eq!(
            events,
            vec![
                DomainEvent::ItemConsumed {
                    item: item("ember-ore"),
                    quantity: 3
                },
                DomainEvent::ItemConsumed {
                    item: item("still-reed"),
                    quantity: 1
                },
                DomainEvent::ItemGranted {
                    item: item("ember-blade"),
                    quantity: 1
                },
            ],
            "inputs are spent in id order, then the output arrives"
        );

        apply_all(&mut state, events.iter()).expect("the reducer agrees");
        assert_eq!(state.player.item_count(&item("ember-ore")), 7);
        assert_eq!(state.player.item_count(&item("ember-blade")), 1);
    }

    #[test]
    fn crafting_reports_the_first_shortfall() {
        let mut state = world();
        state.world_flags.insert(flag("forge-relit"));
        state.player.inventory.insert(item("ember-ore"), 1);

        assert_eq!(
            craft(&blade(), &registry(), &state),
            Err(InteractionError::MissingInputs {
                item: item("ember-ore"),
                held: 1,
                needed: 3
            })
        );
    }

    #[test]
    fn a_recipe_naming_an_unknown_blueprint_is_refused() {
        let mut recipe = blade();
        recipe.requires_upgrade = Some(upgrade_id("upgrade-phantom"));
        assert_eq!(
            craft(&recipe, &registry(), &stocked()),
            Err(InteractionError::UnknownUpgrade(upgrade_id("upgrade-phantom")))
        );
    }

    #[test]
    fn building_spends_the_cost_and_sets_the_flag() {
        let mut state = stocked();
        let events = build_upgrade(&forge(), &registry(), &state).expect("affordable");
        assert_eq!(
            events.last(),
            Some(&DomainEvent::WorldFlagSet {
                flag: flag("forge-relit")
            }),
            "the flag is how the rest of the game notices"
        );

        apply_all(&mut state, events.iter()).expect("the reducer agrees");
        assert_eq!(state.player.item_count(&item("ember-ore")), 4);
        assert!(state.has_world_flag(&flag("forge-relit")));
    }

    #[test]
    fn building_twice_is_refused() {
        let mut state = stocked();
        state.world_flags.insert(flag("forge-relit"));
        assert_eq!(
            build_upgrade(&forge(), &registry(), &state),
            Err(InteractionError::AlreadyBuilt(upgrade_id("upgrade-forge")))
        );
    }

    #[test]
    fn a_blueprint_waits_for_its_prerequisite() {
        let mut state = stocked();
        assert_eq!(
            build_upgrade(&anvil(), &registry(), &state),
            Err(InteractionError::UpgradeNotBuilt(upgrade_id("upgrade-forge")))
        );

        state.world_flags.insert(flag("forge-relit"));
        assert!(build_upgrade(&anvil(), &registry(), &state).is_ok());
    }

    #[test]
    fn building_reports_a_shortfall_rather_than_half_building() {
        let state = world();
        assert!(matches!(
            build_upgrade(&forge(), &registry(), &state),
            Err(InteractionError::MissingInputs { .. })
        ));
    }

    /// Recipes unlock as the settlement grows: the same call that failed before
    /// the forge went up succeeds afterwards, with no other change.
    #[test]
    fn the_upgrade_loop_closes() {
        let mut state = stocked();
        let content = registry();

        let build = build_upgrade(&forge(), &content, &state).expect("affordable");
        apply_all(&mut state, build.iter()).expect("applies");

        let craft = craft(&blade(), &content, &state).expect("now unlocked");
        apply_all(&mut state, craft.iter()).expect("applies");
        assert_eq!(state.player.item_count(&item("ember-blade")), 1);
    }

    fn husk_drops() -> LootTableDef {
        LootTableDef {
            schema_version: CONTENT_SCHEMA_VERSION,
            id: LootTableId::new("husk-drops").expect("valid id"),
            rolls: 2,
            entries: vec![
                LootEntry {
                    item: item("ash-cinder"),
                    quantity: 1,
                    weight: 70,
                },
                LootEntry {
                    item: item("corrupt-resin"),
                    quantity: 1,
                    weight: 30,
                },
            ],
        }
    }

    #[test]
    fn the_same_seed_drops_the_same_loot() {
        let table = husk_drops();
        assert_eq!(loot(&table, 99), loot(&table, 99));
    }

    #[test]
    fn a_loot_seed_is_reproducible_from_canonical_state() {
        let state = world();
        let table = LootTableId::new("husk-drops").expect("valid id");
        assert_eq!(
            loot_seed(&state, &table),
            loot_seed(&state, &table),
            "replay depends on this being a pure function of logged facts"
        );
    }

    #[test]
    fn loot_seeds_separate_tables_moments_and_saves() {
        let state = world();
        let husk = LootTableId::new("husk-drops").expect("valid id");
        let elite = LootTableId::new("elite-drops").expect("valid id");
        assert_ne!(
            loot_seed(&state, &husk),
            loot_seed(&state, &elite),
            "two tables opened at the same moment must not roll alike"
        );

        let mut later = state.clone();
        later.clock.tick = 512;
        assert_ne!(
            loot_seed(&state, &husk),
            loot_seed(&later, &husk),
            "two kills from the same table must not roll alike"
        );

        let mut other_save = state.clone();
        other_save.world_seed = state.world_seed.wrapping_add(1);
        assert_ne!(
            loot_seed(&state, &husk),
            loot_seed(&other_save, &husk),
            "two saves must not share a drop sequence"
        );
    }

    #[test]
    fn rolled_loot_applies_to_the_world() {
        let mut state = world();
        let table = husk_drops();
        let events = loot(&table, loot_seed(&state, &table.id));
        assert!(!events.is_empty(), "a weighted table should drop something");
        apply_all(&mut state, events.iter()).expect("every grant is valid");
    }
}
