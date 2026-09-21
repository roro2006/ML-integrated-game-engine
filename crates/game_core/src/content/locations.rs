//! Spatial content: where things are, and where they may walk.
//!
//! M4 deliberately had no location family, which is why projection could give
//! entities identity but not position. This is that missing piece: authored
//! spawn points, prop placements, links between locations, and the navigation
//! polygons enemies path over.
//!
//! # Authored now, generated later
//!
//! These navmeshes are hand-authored. M11 generates them from procedural
//! geometry, but the *shape* is fixed here so the pathfinding written against
//! it in M5 does not change when generation arrives.
//!
//! # Fixed-point coordinates
//!
//! Positions are [`Fixed`] metres, not floats. A spawn point feeds simulation:
//! two machines replaying the same log must place the same enemy in the same
//! place, and float rounding across platforms would quietly break that.

use serde::{Deserialize, Serialize};

use crate::ids::{
    ChoiceId, ClueId, EncounterId, ItemId, LocationId, ModelId, NpcId, PrefabId, RegionId, SpawnId,
};

use super::Fixed;

/// A position in world space, in fixed-point metres.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Point {
    pub x: Fixed,
    pub y: Fixed,
    pub z: Fixed,
}

impl Point {
    pub const fn new(x: Fixed, y: Fixed, z: Fixed) -> Self {
        Self { x, y, z }
    }
}

/// A position plus a facing.
///
/// Yaw is stored in thousandths of a turn rather than radians: integers replay
/// identically everywhere, and a turn is a friendlier unit to author than a
/// float multiple of pi.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Placement {
    pub at: Point,
    pub yaw_milli_turns: i32,
}

/// What kind of space a location is, which drives how it is populated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum LocationKind {
    /// The hub: safe, populated, has the settlement upgrade slots.
    Settlement,
    /// Open ground between places, lightly populated.
    Wilderness,
    /// An instanced interior with a set-piece threat.
    Dungeon,
}

/// What a fixture is for.
///
/// Authored rather than inferred. A doorway used to be "a station that happens
/// to have a link starting at it", which meant content could not distinguish a
/// forge from a door and the loader could not check that a door was one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum StationKind {
    /// A way into another location. *Where* it leads is a [`LocationLink`];
    /// this only says that it leads somewhere.
    Doorway,
    /// A forge or bench: turns held materials into other items.
    Crafting,
    /// The settlement's upgrade board (ADR-0007 blueprints).
    Upgrades,
}

/// What a spawn point is for.
///
/// An allowlist, like every other content primitive: a spawn cannot describe
/// behaviour, only which authored thing belongs there.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum SpawnRole {
    /// Where the player appears when entering, and where they respawn
    /// (ADR-0007 keeps progress on death).
    PlayerStart,
    Npc {
        npc: NpcId,
        /// What this NPC looks like (ADR-0017).
        ///
        /// On the placement because there is no NPC content family yet —
        /// an NPC is only ever named, never defined. When one exists, this
        /// moves there; until then an NPC placed twice must be given its model
        /// twice.
        #[serde(default)]
        model: Option<ModelId>,
    },
    Enemy {
        prefab: PrefabId,
        /// Which authored fight this placement belongs to.
        ///
        /// A property of the placement, not of the enemy: the same prefab can
        /// stand alone on a road or be one of four in an ambush. A quest's
        /// `Defeat` objective completes when every placement carrying its
        /// encounter is down, so without this there is no way to tell which
        /// kills counted. Optional, because a wandering enemy belongs to no
        /// fight.
        #[serde(default)]
        encounter: Option<EncounterId>,
        /// Which wave of its encounter it arrives in (M8). Wave 0 is there
        /// when the player arrives; each later wave comes once every enemy of
        /// the waves before it is down. Only meaningful with an encounter.
        #[serde(default)]
        wave: u8,
    },
    /// A harvestable node yielding an item.
    Gatherable { item: ItemId },
    /// A fixture the player can use. Deliberately has no default: a station
    /// whose purpose went unstated would be an inert prop nobody noticed.
    Station { kind: StationKind },
    /// Something examinable that teaches the player a piece of knowledge.
    ///
    /// Investigating is inherently spatial — you walk up to a thing and look at
    /// it — which is why a clue is a placement rather than a dialogue effect.
    Clue {
        clue: ClueId,
        /// What the player reads on examining it (M9). Without it examining
        /// still counts; there is just nothing to read.
        #[serde(default)]
        text: Option<String>,
    },
    /// A point at which the player settles an authored decision.
    ///
    /// The companion to [`DialogueEffect::MakeChoice`](super::DialogueEffect):
    /// most choices are made in conversation, but some have no obvious speaker
    /// — a valve to turn, a banner to plant — and this is those.
    Choice {
        choice: ChoiceId,
        /// What the confirm panel asks before the choice is settled (M9), so
        /// a decision is never taken by walking into it.
        #[serde(default)]
        prompt: Option<String>,
    },
}

/// One authored placement within a location.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpawnPoint {
    pub id: SpawnId,
    pub placement: Placement,
    pub role: SpawnRole,
}

/// Static scenery. Props have no behaviour; they are geometry and collision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropPlacement {
    pub prefab: PrefabId,
    pub placement: Placement,
}

/// A way out of this location and into another.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocationLink {
    /// The spawn point in *this* location that acts as the doorway.
    pub from: SpawnId,
    pub to: LocationId,
    /// Where the player appears on the other side.
    pub arrive_at: SpawnId,
}

/// One convex navigation polygon.
///
/// Convex and counter-clockwise seen from above, so pathfinding can treat any
/// point inside as reachable from any other without further checks. Neighbours
/// index into the location's polygon list.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NavPolygon {
    pub vertices: Vec<Point>,
    pub neighbours: Vec<u32>,
}

impl NavPolygon {
    /// Twice the signed area on the XZ plane.
    ///
    /// Positive means counter-clockwise viewed from +Y. Doubling avoids a
    /// division and keeps the value exact in integer milli-units.
    pub fn signed_double_area(&self) -> i64 {
        let mut total: i64 = 0;
        for index in 0..self.vertices.len() {
            let current = self.vertices[index];
            let next = self.vertices[(index + 1) % self.vertices.len()];
            total += current.x.as_milli() * next.z.as_milli() - next.x.as_milli() * current.z.as_milli();
        }
        total
    }
}

/// A place in the world, with everything authored about it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocationDef {
    pub schema_version: u32,
    pub id: LocationId,
    pub name: String,
    /// The region this belongs to, so region-scoped world flags apply.
    pub region: RegionId,
    pub kind: LocationKind,
    pub spawns: Vec<SpawnPoint>,
    pub props: Vec<PropPlacement>,
    pub navmesh: Vec<NavPolygon>,
    pub links: Vec<LocationLink>,
}

impl LocationDef {
    /// The spawn point with `id`, if this location has one.
    pub fn spawn(&self, id: &SpawnId) -> Option<&SpawnPoint> {
        self.spawns.iter().find(|spawn| &spawn.id == id)
    }

    /// Where the player starts here, if anywhere.
    pub fn player_start(&self) -> Option<&SpawnPoint> {
        self.spawns
            .iter()
            .find(|spawn| matches!(spawn.role, SpawnRole::PlayerStart))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: i64, z: i64) -> Point {
        Point::new(Fixed::from_whole(x), Fixed::ZERO, Fixed::from_whole(z))
    }

    fn square(counter_clockwise: bool) -> NavPolygon {
        let mut vertices = vec![point(0, 0), point(4, 0), point(4, 4), point(0, 4)];
        if !counter_clockwise {
            vertices.reverse();
        }
        NavPolygon {
            vertices,
            neighbours: Vec::new(),
        }
    }

    #[test]
    fn winding_is_detectable() {
        assert!(
            square(true).signed_double_area() > 0,
            "counter-clockwise polygons must read positive"
        );
        assert!(
            square(false).signed_double_area() < 0,
            "and an inside-out polygon must be distinguishable, not silently walkable"
        );
    }

    #[test]
    fn degenerate_polygons_have_no_area() {
        let line = NavPolygon {
            vertices: vec![point(0, 0), point(4, 0), point(8, 0)],
            neighbours: Vec::new(),
        };
        assert_eq!(line.signed_double_area(), 0);
    }

    #[test]
    fn placements_round_trip_exactly() {
        let placement = Placement {
            at: Point::new(
                Fixed::from_milli(1_500),
                Fixed::from_milli(-250),
                Fixed::from_whole(7),
            ),
            yaw_milli_turns: 250,
        };
        let text = ron::to_string(&placement).expect("serializes");
        assert_eq!(
            ron::from_str::<Placement>(&text).expect("parses"),
            placement,
            "fixed-point coordinates must survive a round trip bit for bit"
        );
    }

    #[test]
    fn spawn_lookup_finds_roles() {
        let location = LocationDef {
            schema_version: super::super::CONTENT_SCHEMA_VERSION,
            id: LocationId::new("cinderwake-square").expect("valid id"),
            name: "Cinderwake Square".to_owned(),
            region: RegionId::new("cinderwake").expect("valid id"),
            kind: LocationKind::Settlement,
            spawns: vec![
                SpawnPoint {
                    id: SpawnId::new("player-start").expect("valid id"),
                    placement: Placement::default(),
                    role: SpawnRole::PlayerStart,
                },
                SpawnPoint {
                    id: SpawnId::new("sela-post").expect("valid id"),
                    placement: Placement::default(),
                    role: SpawnRole::Npc {
                        npc: NpcId::new("warden-sela").expect("valid id"),
                        model: None,
                    },
                },
            ],
            props: Vec::new(),
            navmesh: vec![square(true)],
            links: Vec::new(),
        };

        assert!(location.player_start().is_some());
        assert!(
            location
                .spawn(&SpawnId::new("sela-post").expect("valid id"))
                .is_some()
        );
        assert!(
            location
                .spawn(&SpawnId::new("nowhere").expect("valid id"))
                .is_none()
        );
    }
}
