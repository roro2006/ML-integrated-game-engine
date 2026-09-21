//! Loot tables and settlement upgrades.
//!
//! Both are about what the player gets and what it costs, and both must be
//! deterministic: loot is rolled from the world seed rather than from a random
//! source, so replaying an event log drops exactly the same items. A drop that
//! depended on wall-clock time or thread scheduling would make replay — and
//! therefore save verification — meaningless.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::ids::{ItemId, LootTableId, UpgradeId, WorldFlag};

/// One possible drop, with its weight relative to the table's other entries.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LootEntry {
    pub item: ItemId,
    pub quantity: u32,
    /// Relative weight. Zero means the entry can never be rolled, which
    /// validation treats as an authoring mistake rather than a silent no-op.
    pub weight: u32,
}

/// A weighted table rolled a fixed number of times.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LootTableDef {
    pub schema_version: u32,
    pub id: LootTableId,
    /// How many times the table is rolled per drop.
    pub rolls: u8,
    pub entries: Vec<LootEntry>,
}

impl LootTableDef {
    /// Total weight across entries.
    pub fn total_weight(&self) -> u64 {
        self.entries.iter().map(|entry| u64::from(entry.weight)).sum()
    }

    /// Rolls the table deterministically.
    ///
    /// The same seed always yields the same result, so a logged kill replays to
    /// the same loot. Callers derive the seed from the world seed plus
    /// something stable about the event — never from a clock.
    pub fn roll(&self, seed: u64) -> BTreeMap<ItemId, u32> {
        let mut dropped: BTreeMap<ItemId, u32> = BTreeMap::new();
        let total = self.total_weight();
        if total == 0 {
            return dropped;
        }

        let mut state = seed;
        for _ in 0..self.rolls {
            let pick = next_random(&mut state) % total;
            let mut cursor = 0;
            for entry in &self.entries {
                cursor += u64::from(entry.weight);
                if pick < cursor {
                    *dropped.entry(entry.item.clone()).or_insert(0) += entry.quantity;
                    break;
                }
            }
        }
        dropped
    }
}

/// SplitMix64: a few lines, no dependency, and good enough for loot.
///
/// Deliberately *not* a cryptographic generator and deliberately not the
/// platform's RNG — reproducibility is the requirement here, not unpredictability.
fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One settlement blueprint (ADR-0007: functional upgrades, no placement tools).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SettlementUpgradeDef {
    pub schema_version: u32,
    pub id: UpgradeId,
    pub name: String,
    /// What it costs to build, by item.
    pub costs: BTreeMap<ItemId, u32>,
    /// Blueprints that must already be built.
    pub requires: Vec<UpgradeId>,
    /// The world flag set once it is built, which is how the rest of the game
    /// notices: services, dialogue, and recipes gate on the flag, not on a
    /// separate "upgrades built" list that could disagree with it.
    pub grants_flag: WorldFlag,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(name: &str) -> ItemId {
        ItemId::new(name).expect("valid id")
    }

    fn table(rolls: u8, entries: &[(&str, u32, u32)]) -> LootTableDef {
        LootTableDef {
            schema_version: super::super::CONTENT_SCHEMA_VERSION,
            id: LootTableId::new("husk-drops").expect("valid id"),
            rolls,
            entries: entries
                .iter()
                .map(|(name, quantity, weight)| LootEntry {
                    item: item(name),
                    quantity: *quantity,
                    weight: *weight,
                })
                .collect(),
        }
    }

    #[test]
    fn the_same_seed_always_drops_the_same_loot() {
        let table = table(
            3,
            &[
                ("ash-cinder", 1, 70),
                ("corrupt-resin", 1, 25),
                ("ember-ore", 2, 5),
            ],
        );
        let first = table.roll(0xC1DE_2026);
        let second = table.roll(0xC1DE_2026);
        assert_eq!(first, second, "replay depends on this being exactly reproducible");
    }

    #[test]
    fn different_seeds_generally_differ() {
        let table = table(4, &[("ash-cinder", 1, 50), ("corrupt-resin", 1, 50)]);
        let outcomes: Vec<_> = (0..16).map(|seed| table.roll(seed)).collect();
        assert!(
            outcomes.windows(2).any(|pair| pair[0] != pair[1]),
            "a table with two equal-weight entries should not roll identically for every seed"
        );
    }

    #[test]
    fn weights_are_respected_over_many_rolls() {
        let table = table(1, &[("common", 1, 99), ("rare", 1, 1)]);
        let commons = (0..400u64)
            .filter(|seed| table.roll(*seed).contains_key(&item("common")))
            .count();
        assert!(
            commons > 340,
            "a 99:1 table should overwhelmingly drop the common item, got {commons}/400"
        );
    }

    #[test]
    fn an_empty_or_weightless_table_drops_nothing() {
        assert!(table(3, &[]).roll(1).is_empty());
        assert!(
            table(3, &[("ash-cinder", 1, 0)]).roll(1).is_empty(),
            "zero total weight must not divide by zero or drop anything"
        );
    }

    #[test]
    fn rolls_accumulate_quantities() {
        let table = table(5, &[("ash-cinder", 2, 1)]);
        let dropped = table.roll(42);
        assert_eq!(
            dropped.get(&item("ash-cinder")),
            Some(&10),
            "five rolls of a single-entry table yield five stacks of two"
        );
    }

    #[test]
    fn upgrades_round_trip() {
        let upgrade = SettlementUpgradeDef {
            schema_version: super::super::CONTENT_SCHEMA_VERSION,
            id: UpgradeId::new("upgrade-forge").expect("valid id"),
            name: "Rebuild the Forge".to_owned(),
            costs: BTreeMap::from([(item("ember-ore"), 12), (item("old-forest-bark"), 4)]),
            requires: Vec::new(),
            grants_flag: WorldFlag::new("forge-relit").expect("valid id"),
        };
        let text = ron::to_string(&upgrade).expect("serializes");
        assert_eq!(
            ron::from_str::<SettlementUpgradeDef>(&text).expect("parses"),
            upgrade
        );
    }
}
