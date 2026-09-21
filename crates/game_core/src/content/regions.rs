//! Regions: the named stretches of the world locations belong to (M9).
//!
//! Small on purpose. A region is a name the player reads and a biome from a
//! closed set; the grammar that turns a biome into generated rooms is M11's.
//! What M9 needs is that `LocationDef::region` names something real, so a
//! typo is a load error rather than a region nobody can discover.

use serde::{Deserialize, Serialize};

use crate::ids::RegionId;

/// What kind of land a region is. A closed set: procedural generation (M11)
/// and any later model proposal choose from these, never invent one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Biome {
    Settlement,
    Wasteland,
    Forest,
    Marsh,
    Highland,
    Underground,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionDef {
    pub schema_version: u32,
    pub id: RegionId,
    /// What the player reads on arriving.
    pub name: String,
    pub biome: Biome,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_region_parses_and_refuses_an_invented_biome() {
        let def: RegionDef = ron::from_str(
            r#"(schema_version: 1, id: "ashfall-waste", name: "Ashfall Waste", biome: Wasteland)"#,
        )
        .expect("parses");
        assert_eq!(def.biome, Biome::Wasteland);
        assert!(
            ron::from_str::<RegionDef>(r#"(schema_version: 1, id: "x", name: "X", biome: Volcano)"#).is_err(),
            "the biome set is closed"
        );
    }
}
