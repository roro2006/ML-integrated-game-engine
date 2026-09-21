//! Intrinsic validation: the problems a definition has on its own,
//! findable without having seen any other file.

use std::collections::BTreeSet;
use std::fmt;

use crate::ids::{
    AbilityId, BodyPlanId, ClipId, DialogueId, DialogueNodeId, EncounterId, ItemId, LocationId, LootTableId,
    NpcId, ObjectiveId, PrefabId, QuestId, SlotId, SpawnId, UpgradeId,
};

use super::*;

/// Why a definition is not internally consistent.
///
/// These are *intrinsic* problems, findable without seeing any other file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContentError {
    SchemaVersion {
        found: u32,
        expected: u32,
    },
    StackLimitZero(ItemId),
    NegativeValue {
        what: &'static str,
        value: Fixed,
    },
    EmptyEffects(AbilityId),
    /// A `ModifyResource` lowers health on someone other than the caster.
    ///
    /// Damage to others has one door, `DealDamage`, which is typed and goes
    /// through resistances and statuses; a negative health change would be a
    /// second, untyped door that armour could never answer. Lowering the
    /// caster's *own* health is allowed — a spell paid for in blood.
    NegativeHealthOnOthers(AbilityId),
    /// An ability asks for a ground telegraph but has no area to mark: a
    /// single target is chosen when the ability lands, so marking ground in
    /// advance would warn of the wrong place.
    TelegraphWithoutArea(AbilityId),
    /// An encounter's slots cost more threat than its budget allows.
    OverThreatBudget {
        encounter: EncounterId,
        total: Fixed,
        budget: Fixed,
    },
    /// An encounter says one number of an enemy; its placements another. The
    /// placements are what a fight is actually made of, so the two must agree
    /// or the encounter's slots and budget describe a different fight.
    PlacementsDisagree {
        encounter: EncounterId,
        enemy: PrefabId,
        slots: u32,
        placed: u32,
    },
    /// An encounter's waves skip a number, so the waves after the gap could
    /// never arrive — or it has more waves than allowed.
    WaveGap {
        encounter: EncounterId,
        wave: u8,
    },
    /// An item that is not a consumable says what using it does.
    UseOnNonConsumable(ItemId),
    /// An item's use holds an effect that does not act on the user — damage,
    /// a summon, a projectile — which would make a potion a weapon.
    NotAllowedOnUse {
        item: ItemId,
        effect: &'static str,
    },
    /// A speaker owns more than one conversation and this one has no topic,
    /// so a player offered several could not tell which is which.
    UntitledConversation {
        speaker: NpcId,
        dialogue: DialogueId,
    },
    /// A placement names a wave but belongs to no encounter, so nothing could
    /// ever send it in.
    WaveWithoutEncounter {
        location: LocationId,
        spawn: SpawnId,
    },
    /// An enemy forgets the player closer than it notices them, which would
    /// have it flicker between idle and chasing at the edge.
    ReleaseInsideAcquire(PrefabId),
    /// A projectile applies nothing when it strikes.
    EmptyOnHit(AbilityId),
    /// A projectile's `on_hit` holds an effect that does not act on what was
    /// struck — another projectile, a summon, a teleport — which could seed a
    /// chain no bound covers.
    NotAllowedOnHit {
        ability: AbilityId,
        effect: &'static str,
    },
    /// An ability names a projectile model or trail but launches nothing.
    ProjectileLookWithoutProjectile(AbilityId),
    /// A visual definition that must have entries has none.
    EmptyVisual {
        owner: String,
        what: &'static str,
    },
    /// A visual definition exceeds one of its budgets (ADR-0017).
    OverBudget {
        owner: String,
        what: &'static str,
        count: usize,
        limit: usize,
    },
    /// A number lies outside the range its field allows.
    OutOfRange {
        what: &'static str,
        value: Fixed,
    },
    /// A body plan names the same slot twice.
    DuplicateSlot {
        plan: BodyPlanId,
        slot: SlotId,
    },
    /// A clip has two tracks for one slot, so which one plays is ambiguous.
    DuplicateTrack {
        clip: ClipId,
        slot: SlotId,
    },
    /// A track's keys are not in time order.
    KeysOutOfOrder {
        clip: ClipId,
        slot: SlotId,
    },
    /// A clip is used for a state whose clock its timing does not run on —
    /// a walk keyed in attack windows would never advance.
    ClipTimingMismatch {
        clip: ClipId,
        state: &'static str,
    },
    EmptyEncounter(EncounterId),
    ZeroCount {
        what: &'static str,
    },
    MissingStartNode {
        quest: QuestId,
        node: ObjectiveId,
    },
    DanglingNode {
        quest: QuestId,
        from: ObjectiveId,
        to: ObjectiveId,
    },
    UnreachableNode {
        quest: QuestId,
        node: ObjectiveId,
    },
    NoCompletion(QuestId),
    /// Two spawn points in one location share an id, so a link or quest naming
    /// it cannot say which is meant.
    DuplicateSpawn {
        location: LocationId,
        spawn: SpawnId,
    },
    /// A navigation polygon is wound clockwise or has no area. An inside-out
    /// polygon is the navmesh equivalent of an invisible mesh: pathfinding
    /// silently refuses to walk it.
    NavPolygonWinding {
        location: LocationId,
        polygon: usize,
    },
    /// A polygon names a neighbour that is not in the mesh.
    NavNeighbourOutOfRange {
        location: LocationId,
        polygon: usize,
        neighbour: u32,
    },
    /// A link's doorway names a spawn point this location does not have.
    UnknownLinkSpawn {
        location: LocationId,
        spawn: SpawnId,
    },
    /// A conversation starts at a node it does not contain.
    MissingDialogueStart {
        dialogue: DialogueId,
        node: DialogueNodeId,
    },
    /// An option leads somewhere that does not exist.
    DanglingDialogueNode {
        dialogue: DialogueId,
        from: DialogueNodeId,
        to: DialogueNodeId,
    },
    /// A node is authored but unreachable from the start.
    UnreachableDialogueNode {
        dialogue: DialogueId,
        node: DialogueNodeId,
    },
    /// Every reachable path loops, so the player could never leave the
    /// conversation.
    InescapableDialogue(DialogueId),
    /// A loot table can never drop anything, which is almost always a mistake
    /// rather than an intentionally empty table.
    LootTableCannotDrop(LootTableId),
    /// An upgrade requires itself, directly.
    SelfRequiringUpgrade(UpgradeId),
    TerminalWithoutResolution {
        quest: QuestId,
        node: ObjectiveId,
    },
}

impl fmt::Display for ContentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SchemaVersion { found, expected } => {
                write!(f, "content schema version {found}, expected {expected}")
            }
            Self::StackLimitZero(item) => write!(f, "item {item} has a stack limit of zero"),
            Self::NegativeValue { what, value } => write!(f, "{what} must not be negative, got {value}"),
            Self::EmptyEffects(ability) => write!(f, "ability {ability} has no effects"),
            Self::EmptyVisual { owner, what } => write!(f, "{owner} has no {what}"),
            Self::OverBudget {
                owner,
                what,
                count,
                limit,
            } => write!(f, "{owner} has {count} {what}, over the limit of {limit}"),
            Self::OutOfRange { what, value } => write!(f, "{what} is out of range: {value}"),
            Self::DuplicateSlot { plan, slot } => {
                write!(f, "body plan {plan} names slot {slot} twice")
            }
            Self::DuplicateTrack { clip, slot } => write!(f, "clip {clip} has two tracks for slot {slot}"),
            Self::KeysOutOfOrder { clip, slot } => {
                write!(f, "clip {clip}'s keys for slot {slot} are not in time order")
            }
            Self::ClipTimingMismatch { clip, state } => write!(
                f,
                "clip {clip} is used for {state}, but its timing does not run on that state's clock"
            ),
            Self::NegativeHealthOnOthers(ability) => write!(
                f,
                "ability {ability} lowers health through ModifyResource on targets other than \
                 its caster; damage to others must use DealDamage"
            ),
            Self::TelegraphWithoutArea(ability) => write!(
                f,
                "ability {ability} asks for a telegraph, but only area targeting or a projectile \
                 has ground to mark"
            ),
            Self::OverThreatBudget {
                encounter,
                total,
                budget,
            } => write!(
                f,
                "encounter {encounter}'s slots cost {total} threat, over its budget of {budget}"
            ),
            Self::PlacementsDisagree {
                encounter,
                enemy,
                slots,
                placed,
            } => write!(
                f,
                "encounter {encounter} lists {slots} {enemy}, but {placed} are placed"
            ),
            Self::WaveGap { encounter, wave } => write!(
                f,
                "encounter {encounter}: wave {wave} can never arrive; waves run from 0 without \
                 gaps, at most {}",
                enemies::MAX_WAVES
            ),
            Self::UseOnNonConsumable(item) => {
                write!(f, "item {item} says what using it does but is not a consumable")
            }
            Self::NotAllowedOnUse { item, effect } => {
                write!(f, "item {item}: {effect} cannot run when an item is used")
            }
            Self::UntitledConversation { speaker, dialogue } => write!(
                f,
                "dialogue {dialogue}: {speaker} has several conversations, so each needs a topic"
            ),
            Self::WaveWithoutEncounter { location, spawn } => write!(
                f,
                "location {location}: spawn {spawn} names a wave but belongs to no encounter"
            ),
            Self::ReleaseInsideAcquire(enemy) => write!(
                f,
                "enemy {enemy} forgets the player closer than it notices them; release must be at \
                 least acquire"
            ),
            Self::EmptyOnHit(ability) => {
                write!(f, "ability {ability} launches a projectile that does nothing")
            }
            Self::NotAllowedOnHit { ability, effect } => write!(
                f,
                "ability {ability}: a projectile may not {effect} on hit; on_hit holds only \
                 DealDamage, ApplyStatus, ModifyResource and EmitEvent"
            ),
            Self::ProjectileLookWithoutProjectile(ability) => write!(
                f,
                "ability {ability} names a projectile look but launches no projectile"
            ),
            Self::EmptyEncounter(encounter) => write!(f, "encounter {encounter} has no slots"),
            Self::ZeroCount { what } => write!(f, "{what} must be at least one"),
            Self::MissingStartNode { quest, node } => {
                write!(f, "quest {quest} starts at {node}, which is not a node")
            }
            Self::DanglingNode { quest, from, to } => {
                write!(
                    f,
                    "quest {quest}: node {from} leads to {to}, which does not exist"
                )
            }
            Self::UnreachableNode { quest, node } => {
                write!(f, "quest {quest}: node {node} is unreachable from the start")
            }
            Self::DuplicateSpawn { location, spawn } => {
                write!(f, "location {location}: spawn id {spawn} is used twice")
            }
            Self::NavPolygonWinding { location, polygon } => write!(
                f,
                "location {location}: nav polygon {polygon} is clockwise or degenerate; \
                 vertices must wind counter-clockwise seen from above"
            ),
            Self::NavNeighbourOutOfRange {
                location,
                polygon,
                neighbour,
            } => write!(
                f,
                "location {location}: nav polygon {polygon} names neighbour {neighbour}, which is not in the mesh"
            ),
            Self::UnknownLinkSpawn { location, spawn } => {
                write!(
                    f,
                    "location {location}: link uses spawn {spawn}, which does not exist here"
                )
            }
            Self::MissingDialogueStart { dialogue, node } => {
                write!(f, "dialogue {dialogue} starts at {node}, which is not a node")
            }
            Self::DanglingDialogueNode { dialogue, from, to } => write!(
                f,
                "dialogue {dialogue}: node {from} leads to {to}, which does not exist"
            ),
            Self::UnreachableDialogueNode { dialogue, node } => {
                write!(
                    f,
                    "dialogue {dialogue}: node {node} is unreachable from the start"
                )
            }
            Self::InescapableDialogue(dialogue) => {
                write!(
                    f,
                    "dialogue {dialogue} has no reachable way to end the conversation"
                )
            }
            Self::LootTableCannotDrop(table) => {
                write!(f, "loot table {table} can never drop anything")
            }
            Self::SelfRequiringUpgrade(upgrade) => {
                write!(f, "upgrade {upgrade} requires itself")
            }
            Self::NoCompletion(quest) => {
                write!(f, "quest {quest} has no reachable node that completes it")
            }
            Self::TerminalWithoutResolution { quest, node } => {
                write!(
                    f,
                    "quest {quest}: node {node} ends the quest but names no resolution"
                )
            }
        }
    }
}

impl std::error::Error for ContentError {}

fn check_version(found: u32) -> Result<(), ContentError> {
    if found == CONTENT_SCHEMA_VERSION {
        Ok(())
    } else {
        Err(ContentError::SchemaVersion {
            found,
            expected: CONTENT_SCHEMA_VERSION,
        })
    }
}

impl ClassDef {
    /// Intrinsic checks: a class nobody could survive as is an authoring
    /// mistake, not a difficulty option.
    pub fn validate(&self) -> Result<(), ContentError> {
        check_version(self.schema_version)?;
        for (what, value) in [
            ("base_health", self.base_health),
            ("base_stamina", self.base_stamina),
        ] {
            check_non_negative(what, value)?;
            if value.as_milli() == 0 {
                return Err(ContentError::ZeroCount { what });
            }
        }
        self.combat.validate()
    }
}

impl ClassCombat {
    /// Every number set lies in its range. Fields left out are the slice's
    /// defaults, valid by construction.
    pub fn validate(&self) -> Result<(), ContentError> {
        let bounded = |what, value: Option<Fixed>, min_milli: i64, max_whole: i64| match value {
            Some(value) => check_range(what, value, min_milli, max_whole * FIXED_SCALE),
            None => Ok(()),
        };
        bounded(
            "class swing damage",
            self.damage,
            1_000,
            classes::MAX_CLASS_DAMAGE,
        )?;
        bounded("class swing reach", self.reach, 500, enemies::MAX_REACH_METRES)?;
        bounded(
            "class swing radius",
            self.radius,
            100,
            enemies::MAX_STRIKE_RADIUS_METRES,
        )?;
        bounded("class swing cost", self.cost, 0, classes::MAX_CLASS_COST)?;
        bounded("class poise", self.poise, 1_000, enemies::MAX_POISE)?;
        bounded("class roll cost", self.dodge_cost, 0, classes::MAX_CLASS_COST)?;
        for (what, ticks, least, most) in [
            ("class wind-up", self.windup_ticks, 1, enemies::MAX_WINDOW_TICKS),
            (
                "class active window",
                self.active_ticks,
                1,
                enemies::MAX_WINDOW_TICKS,
            ),
            (
                "class recovery",
                self.recovery_ticks,
                0,
                enemies::MAX_WINDOW_TICKS,
            ),
            ("class roll i-frames", self.dodge_iframes, 0, classes::DODGE_TICKS),
        ] {
            if let Some(ticks) = ticks
                && !(least..=most).contains(&ticks)
            {
                return Err(ContentError::OutOfRange {
                    what,
                    value: Fixed::from_whole(i64::from(ticks)),
                });
            }
        }
        Ok(())
    }
}

fn check_non_negative(what: &'static str, value: Fixed) -> Result<(), ContentError> {
    if value.is_negative() {
        Err(ContentError::NegativeValue { what, value })
    } else {
        Ok(())
    }
}

impl ItemDef {
    pub fn validate(&self) -> Result<(), ContentError> {
        check_version(self.schema_version)?;
        if self.stack_limit == 0 {
            return Err(ContentError::StackLimitZero(self.id.clone()));
        }
        if let Some(combat) = &self.combat {
            check_non_negative("item damage", combat.damage)?;
            // Resistances are checked for sign only. The *ceiling* is a runtime
            // concern: `game_runtime::combat::MAX_RESISTANCE` clamps the total
            // so that no combination of equipment reaches immunity, and a
            // validator duplicating that number here would be a second source
            // of truth that drifts the first time it is retuned.
            for fraction in combat.resistances.values() {
                check_non_negative("item resistance", *fraction)?;
            }
        }
        if let Some(used) = &self.on_use {
            if self.category != ItemCategory::Consumable {
                return Err(ContentError::UseOnNonConsumable(self.id.clone()));
            }
            if used.effects.is_empty() {
                return Err(ContentError::ZeroCount { what: "use effects" });
            }
            check_budget(&self.id, "use effects", used.effects.len(), MAX_USE_EFFECTS)?;
            if !(1..=MAX_USE_TICKS).contains(&used.use_ticks) {
                return Err(ContentError::OutOfRange {
                    what: "use time",
                    value: Fixed::from_whole(i64::from(used.use_ticks)),
                });
            }
            for effect in &used.effects {
                if !effect.allowed_on_use() {
                    return Err(ContentError::NotAllowedOnUse {
                        item: self.id.clone(),
                        effect: effect.kind(),
                    });
                }
            }
        }
        Ok(())
    }
}

impl RecipeDef {
    pub fn validate(&self) -> Result<(), ContentError> {
        check_version(self.schema_version)?;
        if self.output_quantity == 0 {
            return Err(ContentError::ZeroCount {
                what: "recipe output quantity",
            });
        }
        if self.inputs.values().any(|quantity| *quantity == 0) {
            return Err(ContentError::ZeroCount {
                what: "recipe input quantity",
            });
        }
        Ok(())
    }
}

impl AbilityDef {
    pub fn validate(&self) -> Result<(), ContentError> {
        check_version(self.schema_version)?;
        check_non_negative("ability cost", self.cost)?;
        if self.effects.is_empty() {
            return Err(ContentError::EmptyEffects(self.id.clone()));
        }
        for effect in &self.effects {
            self.check_effect(effect, self.targeting == Targeting::SelfOnly)?;
            if let Effect::Projectile {
                speed,
                range,
                radius,
                pierce,
                on_hit,
            } = effect
            {
                check_range(
                    "projectile speed",
                    *speed,
                    MIN_PROJECTILE_SPEED_METRES * FIXED_SCALE,
                    MAX_PROJECTILE_SPEED_METRES * FIXED_SCALE,
                )?;
                check_range(
                    "projectile range",
                    *range,
                    1,
                    MAX_PROJECTILE_RANGE_METRES * FIXED_SCALE,
                )?;
                check_range(
                    "projectile radius",
                    *radius,
                    MIN_PROJECTILE_RADIUS_MILLI,
                    MAX_PROJECTILE_RADIUS_MILLI,
                )?;
                check_budget(
                    &self.id,
                    "pierced targets",
                    usize::from(*pierce),
                    usize::from(MAX_PIERCE),
                )?;
                if on_hit.is_empty() {
                    return Err(ContentError::EmptyOnHit(self.id.clone()));
                }
                check_budget(&self.id, "on-hit effects", on_hit.len(), MAX_ON_HIT_EFFECTS)?;
                for inner in on_hit {
                    if !inner.allowed_on_hit() {
                        return Err(ContentError::NotAllowedOnHit {
                            ability: self.id.clone(),
                            effect: inner.kind(),
                        });
                    }
                    // What a projectile strikes is never its caster.
                    self.check_effect(inner, false)?;
                }
            }
        }
        let launches = self.projectile().is_some();
        if self.presentation.telegraph
            && !launches
            && !matches!(self.targeting, Targeting::AreaAroundSelf | Targeting::AreaAtPoint)
        {
            return Err(ContentError::TelegraphWithoutArea(self.id.clone()));
        }
        if !launches && (self.presentation.projectile.is_some() || self.presentation.trail.is_some()) {
            return Err(ContentError::ProjectileLookWithoutProjectile(self.id.clone()));
        }
        Ok(())
    }

    /// The rules one effect obeys wherever it sits. `on_caster` is whether it
    /// lands on the caster alone, the one place lowering health is a cost
    /// rather than an attack.
    fn check_effect(&self, effect: &Effect, on_caster: bool) -> Result<(), ContentError> {
        if let Effect::DealDamage { amount, .. } = effect {
            check_non_negative("damage", *amount)?;
        }
        if let Effect::ModifyResource {
            resource: ResourceType::Health,
            amount,
        } = effect
            && amount.is_negative()
            && !on_caster
        {
            return Err(ContentError::NegativeHealthOnOthers(self.id.clone()));
        }
        if let Effect::SpawnEntity { count, .. } = effect
            && *count == 0
        {
            return Err(ContentError::ZeroCount { what: "spawn count" });
        }
        Ok(())
    }
}

/// Checks a count against its budget.
fn check_budget(
    owner: &dyn fmt::Display,
    what: &'static str,
    count: usize,
    limit: usize,
) -> Result<(), ContentError> {
    if count > limit {
        Err(ContentError::OverBudget {
            owner: owner.to_string(),
            what,
            count,
            limit,
        })
    } else {
        Ok(())
    }
}

/// Checks a value lies within `min..=max`, both in milli-units.
fn check_range(what: &'static str, value: Fixed, min: i64, max: i64) -> Result<(), ContentError> {
    if (min..=max).contains(&value.as_milli()) {
        Ok(())
    } else {
        Err(ContentError::OutOfRange { what, value })
    }
}

fn check_point(what: &'static str, point: Point, limit: i64) -> Result<(), ContentError> {
    for axis in [point.x, point.y, point.z] {
        check_range(what, axis, -limit * FIXED_SCALE, limit * FIXED_SCALE)?;
    }
    Ok(())
}

/// A shape's dimensions are positive and bounded; its tessellation is in range.
///
/// Rejected here rather than clamped, even though the engine clamps too: a
/// clamp is the runtime's safety net for input that got past validation, and
/// silently building something other than what was written is not a result an
/// author should get.
fn check_shape(shape: ShapeDef) -> Result<(), ContentError> {
    let longest = models::MAX_EXTENT_METRES * FIXED_SCALE;
    let length = |what, value| check_range(what, value, 1, longest);
    let sides = |sides: u8| {
        if (MIN_SIDES..=MAX_SIDES).contains(&sides) {
            Ok(())
        } else {
            Err(ContentError::OutOfRange {
                what: "shape sides",
                value: Fixed::from_whole(i64::from(sides)),
            })
        }
    };
    match shape {
        ShapeDef::Box { x, y, z } | ShapeDef::Wedge { x, y, z } => {
            length("shape width", x)?;
            length("shape height", y)?;
            length("shape depth", z)
        }
        ShapeDef::Cylinder {
            radius,
            height,
            sides: n,
        }
        | ShapeDef::Cone {
            radius,
            height,
            sides: n,
        } => {
            length("shape radius", radius)?;
            length("shape height", height)?;
            sides(n)
        }
        ShapeDef::Frustum {
            bottom,
            top,
            height,
            sides: n,
        } => {
            length("frustum bottom radius", bottom)?;
            // Zero is allowed: that is a cone.
            check_range("frustum top radius", top, 0, longest)?;
            length("shape height", height)?;
            sides(n)
        }
        ShapeDef::Taper {
            x,
            z,
            top_x,
            top_z,
            height,
        } => {
            length("taper width", x)?;
            length("taper depth", z)?;
            // Zero is allowed on the top: a ridge, or a spike.
            check_range("taper top width", top_x, 0, longest)?;
            check_range("taper top depth", top_z, 0, longest)?;
            length("shape height", height)
        }
        ShapeDef::Dome {
            radius,
            segments,
            rings,
        } => {
            length("shape radius", radius)?;
            sides(segments)?;
            if !(1..=models::MAX_DOME_RINGS).contains(&rings) {
                return Err(ContentError::OutOfRange {
                    what: "dome rings",
                    value: Fixed::from_whole(i64::from(rings)),
                });
            }
            Ok(())
        }
        ShapeDef::Sphere { radius, detail } => {
            length("shape radius", radius)?;
            if detail > MAX_SPHERE_DETAIL {
                return Err(ContentError::OutOfRange {
                    what: "sphere detail",
                    value: Fixed::from_whole(i64::from(detail)),
                });
            }
            Ok(())
        }
    }
}

impl BodyPlanDef {
    pub fn validate(&self) -> Result<(), ContentError> {
        check_version(self.schema_version)?;
        if self.slots.is_empty() {
            return Err(ContentError::EmptyVisual {
                owner: self.id.to_string(),
                what: "slots",
            });
        }
        check_budget(&self.id, "slots", self.slots.len(), MAX_SLOTS)?;
        let mut seen = BTreeSet::new();
        for slot in &self.slots {
            if !seen.insert(&slot.name) {
                return Err(ContentError::DuplicateSlot {
                    plan: self.id.clone(),
                    slot: slot.name.clone(),
                });
            }
            check_point("slot pivot", slot.pivot, models::MAX_OFFSET_METRES)?;
        }
        Ok(())
    }
}

impl PaletteDef {
    pub fn validate(&self) -> Result<(), ContentError> {
        check_version(self.schema_version)?;
        if self.swatches.is_empty() {
            return Err(ContentError::EmptyVisual {
                owner: self.id.to_string(),
                what: "swatches",
            });
        }
        check_budget(&self.id, "swatches", self.swatches.len(), MAX_SWATCHES)
    }
}

impl ModelDef {
    /// Budgets and ranges. Whether each slot and swatch exists needs the body
    /// plan and palette, so that is reference resolution's job.
    pub fn validate(&self) -> Result<(), ContentError> {
        check_version(self.schema_version)?;
        if self.parts.is_empty() {
            return Err(ContentError::EmptyVisual {
                owner: self.id.to_string(),
                what: "parts",
            });
        }
        check_budget(&self.id, "parts", self.parts.len(), MAX_PARTS)?;
        check_budget(&self.id, "triangles", self.triangles(), MAX_TRIANGLES)?;
        check_range(
            "model scale",
            self.scale,
            models::MIN_SCALE_MILLI,
            models::MAX_SCALE_MILLI,
        )?;
        for part in &self.parts {
            check_shape(part.shape)?;
            check_point("part offset", part.at, models::MAX_OFFSET_METRES)?;
            check_point("part turn", part.turn, models::MAX_TURN_DEGREES)?;
            for axis in [part.stretch.x, part.stretch.y, part.stretch.z] {
                check_range(
                    "part stretch",
                    axis,
                    models::MIN_STRETCH_MILLI,
                    models::MAX_STRETCH_MILLI,
                )?;
            }
        }
        Ok(())
    }
}

impl ClipDef {
    /// Budgets, ranges, and key order. Whether a track's slot exists, and
    /// whether the clip suits the state it is used for, need other families.
    pub fn validate(&self) -> Result<(), ContentError> {
        check_version(self.schema_version)?;
        match self.timing {
            ClipTiming::Loop { ticks } => {
                if !(1..=MAX_LOOP_TICKS).contains(&ticks) {
                    return Err(ContentError::OutOfRange {
                        what: "loop length",
                        value: Fixed::from_whole(i64::from(ticks)),
                    });
                }
            }
            ClipTiming::Stride { metres } => {
                check_range("stride length", metres, MIN_STRIDE_MILLI, MAX_STRIDE_MILLI)?;
            }
            ClipTiming::Windows | ClipTiming::Span => {}
        }
        if self.tracks.is_empty() {
            return Err(ContentError::EmptyVisual {
                owner: self.id.to_string(),
                what: "tracks",
            });
        }
        check_budget(&self.id, "tracks", self.tracks.len(), MAX_TRACKS)?;

        let mut seen = BTreeSet::new();
        for track in &self.tracks {
            if !seen.insert(&track.slot) {
                return Err(ContentError::DuplicateTrack {
                    clip: self.id.clone(),
                    slot: track.slot.clone(),
                });
            }
            if track.keys.is_empty() {
                return Err(ContentError::EmptyVisual {
                    owner: format!("{}/{}", self.id, track.slot),
                    what: "keys",
                });
            }
            check_budget(&self.id, "keys in one track", track.keys.len(), MAX_KEYS)?;
            for key in &track.keys {
                check_range("key time", key.at, 0, self.timing.end())?;
                check_point("key turn", key.turn, models::MAX_TURN_DEGREES)?;
                check_point("key offset", key.offset, MAX_KEY_OFFSET_METRES)?;
                for axis in [key.scale.x, key.scale.y, key.scale.z] {
                    check_range(
                        "key scale",
                        axis,
                        models::MIN_STRETCH_MILLI,
                        models::MAX_STRETCH_MILLI,
                    )?;
                }
            }
            if track.keys.windows(2).any(|pair| pair[1].at < pair[0].at) {
                return Err(ContentError::KeysOutOfOrder {
                    clip: self.id.clone(),
                    slot: track.slot.clone(),
                });
            }
        }
        Ok(())
    }
}

impl LevelCurveDef {
    /// Each threshold more than the last, the first more than nothing, and
    /// no more levels than the cap.
    pub fn validate(&self) -> Result<(), ContentError> {
        check_version(self.schema_version)?;
        if self.thresholds.is_empty() {
            return Err(ContentError::ZeroCount { what: "levels" });
        }
        check_budget(&self.id, "levels", self.thresholds.len() + 1, MAX_LEVEL)?;
        let mut previous = 0;
        for needed in &self.thresholds {
            if *needed <= previous {
                return Err(ContentError::OutOfRange {
                    what: "level threshold",
                    value: Fixed::from_whole(i64::from(*needed)),
                });
            }
            previous = *needed;
        }
        Ok(())
    }
}

impl RegionDef {
    pub fn validate(&self) -> Result<(), ContentError> {
        check_version(self.schema_version)
    }
}

impl StatusDef {
    /// Modifier count and bounds. Whether the palette, swatch, and effect
    /// exist is reference resolution's job.
    pub fn validate(&self) -> Result<(), ContentError> {
        check_version(self.schema_version)?;
        if self.modifiers.is_empty() {
            return Err(ContentError::ZeroCount {
                what: "status modifiers",
            });
        }
        check_budget(
            &self.id.0,
            "modifiers",
            self.modifiers.len(),
            MAX_STATUS_MODIFIERS,
        )?;
        for modifier in &self.modifiers {
            match modifier {
                StatusModifier::DamageTaken(scale)
                | StatusModifier::DamageDealt(scale)
                | StatusModifier::StaminaRegen(scale) => check_range(
                    "status multiplier",
                    *scale,
                    MIN_STATUS_MULTIPLIER_MILLI,
                    MAX_STATUS_MULTIPLIER_MILLI,
                )?,
                StatusModifier::DamageOverTime { per_second, .. } => {
                    check_range("status damage per second", *per_second, 1, MAX_STATUS_DPS_MILLI)?;
                }
            }
        }
        Ok(())
    }
}

impl VfxDef {
    /// Budgets and ranges. Whether the palette and its swatches exist is
    /// reference resolution's job.
    pub fn validate(&self) -> Result<(), ContentError> {
        check_version(self.schema_version)?;
        if self.parts.is_empty() {
            return Err(ContentError::EmptyVisual {
                owner: self.id.to_string(),
                what: "parts",
            });
        }
        check_budget(&self.id, "parts", self.parts.len(), MAX_VFX_PARTS)?;
        for part in &self.parts {
            if !(1..=MAX_VFX_TICKS).contains(&part.ticks()) {
                return Err(ContentError::OutOfRange {
                    what: "effect length",
                    value: Fixed::from_whole(i64::from(part.ticks())),
                });
            }
            match part {
                VfxPart::Burst {
                    count,
                    size,
                    speed,
                    gravity,
                    ..
                } => {
                    if *count == 0 {
                        return Err(ContentError::ZeroCount { what: "burst shards" });
                    }
                    check_budget(
                        &self.id,
                        "shards in one burst",
                        usize::from(*count),
                        usize::from(MAX_BURST_SHARDS),
                    )?;
                    check_range("shard size", *size, vfx::MIN_SHARD_MILLI, vfx::MAX_SHARD_MILLI)?;
                    check_range(
                        "shard speed",
                        *speed,
                        0,
                        vfx::MAX_SHARD_SPEED_METRES * FIXED_SCALE,
                    )?;
                    check_range(
                        "shard gravity",
                        *gravity,
                        0,
                        vfx::MAX_GRAVITY_METRES * FIXED_SCALE,
                    )?;
                }
                VfxPart::Pulse { radius, .. } => {
                    check_range(
                        "pulse radius",
                        *radius,
                        vfx::MIN_PULSE_MILLI,
                        vfx::MAX_PULSE_MILLI,
                    )?;
                }
                VfxPart::Beam { width, .. } => {
                    check_range("beam width", *width, vfx::MIN_BEAM_MILLI, vfx::MAX_BEAM_MILLI)?;
                }
                VfxPart::Slash {
                    radius,
                    arc,
                    width,
                    tilt,
                    ..
                } => {
                    check_range(
                        "slash radius",
                        *radius,
                        vfx::MIN_SLASH_RADIUS_MILLI,
                        vfx::MAX_SLASH_RADIUS_MILLI,
                    )?;
                    check_range(
                        "slash arc",
                        *arc,
                        vfx::MIN_SLASH_ARC_DEGREES * FIXED_SCALE,
                        vfx::MAX_SLASH_ARC_DEGREES * FIXED_SCALE,
                    )?;
                    check_range("slash width", *width, vfx::MIN_BEAM_MILLI, vfx::MAX_BEAM_MILLI)?;
                    check_range("slash tilt", *tilt, -90 * FIXED_SCALE, 90 * FIXED_SCALE)?;
                }
                VfxPart::Crack {
                    count, length, width, ..
                } => {
                    if *count == 0 {
                        return Err(ContentError::ZeroCount { what: "cracks" });
                    }
                    check_budget(
                        &self.id,
                        "fissures in one crack",
                        usize::from(*count),
                        usize::from(vfx::MAX_CRACKS),
                    )?;
                    check_range(
                        "crack length",
                        *length,
                        vfx::MIN_CRACK_MILLI,
                        vfx::MAX_CRACK_MILLI,
                    )?;
                    check_range("crack width", *width, vfx::MIN_BEAM_MILLI, vfx::MAX_BEAM_MILLI)?;
                }
                VfxPart::Spire {
                    count,
                    height,
                    width,
                    spread,
                    ..
                } => {
                    if *count == 0 {
                        return Err(ContentError::ZeroCount { what: "spires" });
                    }
                    check_budget(
                        &self.id,
                        "spires in one eruption",
                        usize::from(*count),
                        usize::from(vfx::MAX_SPIRES),
                    )?;
                    check_range(
                        "spire height",
                        *height,
                        vfx::MIN_SPIRE_HEIGHT_MILLI,
                        vfx::MAX_SPIRE_HEIGHT_MILLI,
                    )?;
                    check_range(
                        "spire width",
                        *width,
                        vfx::MIN_SPIRE_WIDTH_MILLI,
                        vfx::MAX_SPIRE_WIDTH_MILLI,
                    )?;
                    check_range(
                        "spire spread",
                        *spread,
                        vfx::MIN_SPIRE_SPREAD_MILLI,
                        vfx::MAX_SPIRE_SPREAD_MILLI,
                    )?;
                }
            }
        }
        Ok(())
    }
}

impl EnemyDef {
    pub fn validate(&self) -> Result<(), ContentError> {
        check_version(self.schema_version)?;
        check_non_negative("enemy health", self.max_health)?;
        check_non_negative("enemy damage", self.damage)?;
        check_non_negative("enemy threat", self.threat)?;
        self.tuning.validate(&self.id)
    }
}

impl EnemyTuning {
    /// Every number set lies in its range, and release is never inside
    /// acquire. Fields left out are the template's, and valid by
    /// construction.
    pub fn validate(&self, enemy: &PrefabId) -> Result<(), ContentError> {
        let metres = |what, value: Option<Fixed>, min_milli: i64, max_metres: i64| match value {
            Some(value) => check_range(what, value, min_milli, max_metres * FIXED_SCALE),
            None => Ok(()),
        };
        metres("aggro acquire", self.acquire, 500, enemies::MAX_AGGRO_METRES)?;
        metres("aggro release", self.release, 500, enemies::MAX_AGGRO_METRES)?;
        if let (Some(acquire), Some(release)) = (self.acquire, self.release)
            && release < acquire
        {
            return Err(ContentError::ReleaseInsideAcquire(enemy.clone()));
        }
        metres(
            "preferred range",
            self.preferred_range,
            500,
            enemies::MAX_PREFERRED_RANGE_METRES,
        )?;
        metres(
            "move speed",
            self.move_speed,
            100,
            enemies::MAX_ENEMY_SPEED_METRES,
        )?;
        metres("swing reach", self.reach, 500, enemies::MAX_REACH_METRES)?;
        metres(
            "strike radius",
            self.strike_radius,
            100,
            enemies::MAX_STRIKE_RADIUS_METRES,
        )?;
        metres("poise", self.poise, 1_000, enemies::MAX_POISE)?;
        for (what, ticks, least) in [
            ("wind-up", self.windup_ticks, 1),
            ("active window", self.active_ticks, 1),
            ("recovery", self.recovery_ticks, 0),
        ] {
            if let Some(ticks) = ticks
                && !(least..=enemies::MAX_WINDOW_TICKS).contains(&ticks)
            {
                return Err(ContentError::OutOfRange {
                    what,
                    value: Fixed::from_whole(i64::from(ticks)),
                });
            }
        }
        Ok(())
    }
}

impl EncounterDef {
    pub fn validate(&self) -> Result<(), ContentError> {
        check_version(self.schema_version)?;
        check_non_negative("threat budget", self.threat_budget)?;
        if self.slots.is_empty() {
            return Err(ContentError::EmptyEncounter(self.id.clone()));
        }
        if self.slots.iter().any(|slot| slot.count == 0) {
            return Err(ContentError::ZeroCount {
                what: "encounter slot count",
            });
        }
        Ok(())
    }
}

impl LocationDef {
    /// Checks the space is navigable and internally consistent.
    ///
    /// The winding check matters more than it looks: pathfinding treats a
    /// counter-clockwise polygon as walkable floor, so a reversed one becomes
    /// a hole that enemies refuse to cross, with nothing on screen to explain
    /// why. It is the same class of bug as an inside-out mesh, and equally
    /// invisible without a test.
    pub fn validate(&self) -> Result<(), ContentError> {
        check_version(self.schema_version)?;

        let mut seen = BTreeSet::new();
        for spawn in &self.spawns {
            if !seen.insert(&spawn.id) {
                return Err(ContentError::DuplicateSpawn {
                    location: self.id.clone(),
                    spawn: spawn.id.clone(),
                });
            }
            if let SpawnRole::Enemy {
                encounter: None,
                wave,
                ..
            } = &spawn.role
                && *wave > 0
            {
                return Err(ContentError::WaveWithoutEncounter {
                    location: self.id.clone(),
                    spawn: spawn.id.clone(),
                });
            }
        }

        for (index, polygon) in self.navmesh.iter().enumerate() {
            if polygon.vertices.len() < 3 || polygon.signed_double_area() <= 0 {
                return Err(ContentError::NavPolygonWinding {
                    location: self.id.clone(),
                    polygon: index,
                });
            }
            for neighbour in &polygon.neighbours {
                if *neighbour as usize >= self.navmesh.len() {
                    return Err(ContentError::NavNeighbourOutOfRange {
                        location: self.id.clone(),
                        polygon: index,
                        neighbour: *neighbour,
                    });
                }
            }
        }

        for link in &self.links {
            if self.spawn(&link.from).is_none() {
                return Err(ContentError::UnknownLinkSpawn {
                    location: self.id.clone(),
                    spawn: link.from.clone(),
                });
            }
        }
        Ok(())
    }
}

impl DialogueDef {
    /// Checks the conversation is playable and escapable.
    ///
    /// A conversation the player cannot leave is a soft-lock, so "some reachable
    /// node can end it" is checked rather than assumed.
    pub fn validate(&self) -> Result<(), ContentError> {
        check_version(self.schema_version)?;
        if !self.nodes.contains_key(&self.start) {
            return Err(ContentError::MissingDialogueStart {
                dialogue: self.id.clone(),
                node: self.start.clone(),
            });
        }

        for (from, node) in &self.nodes {
            for option in &node.options {
                if let Some(to) = &option.next
                    && !self.nodes.contains_key(to)
                {
                    return Err(ContentError::DanglingDialogueNode {
                        dialogue: self.id.clone(),
                        from: from.clone(),
                        to: to.clone(),
                    });
                }
            }
        }

        let reachable = self.reachable();
        if let Some(orphan) = self.nodes.keys().find(|node| !reachable.contains_key(node)) {
            return Err(ContentError::UnreachableDialogueNode {
                dialogue: self.id.clone(),
                node: orphan.clone(),
            });
        }
        if !reachable
            .keys()
            .filter_map(|node| self.nodes.get(*node))
            .any(DialogueNode::can_end)
        {
            return Err(ContentError::InescapableDialogue(self.id.clone()));
        }
        Ok(())
    }
}

impl LootTableDef {
    pub fn validate(&self) -> Result<(), ContentError> {
        check_version(self.schema_version)?;
        if self.rolls == 0 {
            return Err(ContentError::ZeroCount {
                what: "loot table rolls",
            });
        }
        // An entry with zero quantity would "drop" nothing while appearing to
        // drop something, which reads as a bug at the till rather than here.
        if self.entries.iter().any(|entry| entry.quantity == 0) {
            return Err(ContentError::ZeroCount {
                what: "loot entry quantity",
            });
        }
        if self.total_weight() == 0 {
            return Err(ContentError::LootTableCannotDrop(self.id.clone()));
        }
        Ok(())
    }
}

impl SettlementUpgradeDef {
    pub fn validate(&self) -> Result<(), ContentError> {
        check_version(self.schema_version)?;
        if self.costs.values().any(|quantity| *quantity == 0) {
            return Err(ContentError::ZeroCount {
                what: "upgrade cost quantity",
            });
        }
        if self.requires.contains(&self.id) {
            return Err(ContentError::SelfRequiringUpgrade(self.id.clone()));
        }
        Ok(())
    }
}

impl QuestDef {
    /// Checks the graph is playable: the start exists, every edge lands
    /// somewhere, every node is reachable, and at least one reachable node ends
    /// the quest.
    ///
    /// An unreachable node is reported rather than ignored because it is almost
    /// always an authoring mistake, and because M11 and the AI pipeline will
    /// generate these graphs — a silently orphaned branch is a quest that
    /// cannot be finished.
    pub fn validate(&self) -> Result<(), ContentError> {
        check_version(self.schema_version)?;
        if !self.nodes.contains_key(&self.start) {
            return Err(ContentError::MissingStartNode {
                quest: self.id.clone(),
                node: self.start.clone(),
            });
        }

        for (from, node) in &self.nodes {
            for to in &node.next {
                if !self.nodes.contains_key(to) {
                    return Err(ContentError::DanglingNode {
                        quest: self.id.clone(),
                        from: from.clone(),
                        to: to.clone(),
                    });
                }
            }
            if node.next.is_empty() && node.resolution.is_none() {
                return Err(ContentError::TerminalWithoutResolution {
                    quest: self.id.clone(),
                    node: from.clone(),
                });
            }
        }

        let mut reached = BTreeSet::new();
        let mut frontier = vec![self.start.clone()];
        let mut completes = false;
        while let Some(current) = frontier.pop() {
            if !reached.insert(current.clone()) {
                continue;
            }
            let Some(node) = self.nodes.get(&current) else {
                continue;
            };
            if node.resolution.is_some() {
                completes = true;
            }
            frontier.extend(node.next.iter().cloned());
        }

        if let Some(orphan) = self.nodes.keys().find(|node| !reached.contains(*node)) {
            return Err(ContentError::UnreachableNode {
                quest: self.id.clone(),
                node: orphan.clone(),
            });
        }
        if !completes {
            return Err(ContentError::NoCompletion(self.id.clone()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod new_family_tests {
    use super::*;
    use crate::ids::{FactionId, NpcId, PrefabId, RegionId, WorldFlag};

    fn point(x: i64, z: i64) -> Point {
        Point::new(Fixed::from_whole(x), Fixed::ZERO, Fixed::from_whole(z))
    }

    /// A counter-clockwise unit square, optionally reversed.
    fn square(counter_clockwise: bool, neighbours: Vec<u32>) -> NavPolygon {
        let mut vertices = vec![point(0, 0), point(4, 0), point(4, 4), point(0, 4)];
        if !counter_clockwise {
            vertices.reverse();
        }
        NavPolygon { vertices, neighbours }
    }

    fn spawn(id: &str, role: SpawnRole) -> SpawnPoint {
        SpawnPoint {
            id: SpawnId::new(id).expect("valid id"),
            placement: Placement::default(),
            role,
        }
    }

    fn location(spawns: Vec<SpawnPoint>, navmesh: Vec<NavPolygon>, links: Vec<LocationLink>) -> LocationDef {
        LocationDef {
            schema_version: CONTENT_SCHEMA_VERSION,
            id: LocationId::new("cinderwake-square").expect("valid id"),
            name: "Cinderwake Square".to_owned(),
            region: RegionId::new("cinderwake").expect("valid id"),
            kind: LocationKind::Settlement,
            spawns,
            props: Vec::new(),
            navmesh,
            links,
        }
    }

    #[test]
    fn a_valid_location_passes() {
        let valid = location(
            vec![spawn("player-start", SpawnRole::PlayerStart)],
            vec![square(true, vec![])],
            Vec::new(),
        );
        assert_eq!(valid.validate(), Ok(()));
    }

    #[test]
    fn duplicate_spawn_ids_are_rejected() {
        let duplicated = location(
            vec![
                spawn("post", SpawnRole::PlayerStart),
                spawn(
                    "post",
                    SpawnRole::Npc {
                        npc: NpcId::new("warden-sela").expect("valid id"),
                        model: None,
                    },
                ),
            ],
            Vec::new(),
            Vec::new(),
        );
        assert!(matches!(
            duplicated.validate(),
            Err(ContentError::DuplicateSpawn { .. })
        ));
    }

    /// The check that matters most here: a reversed polygon is walkable-looking
    /// floor that pathfinding silently refuses to cross.
    #[test]
    fn a_clockwise_or_degenerate_polygon_is_rejected() {
        let reversed = location(Vec::new(), vec![square(false, vec![])], Vec::new());
        assert!(matches!(
            reversed.validate(),
            Err(ContentError::NavPolygonWinding { polygon: 0, .. })
        ));

        let sliver = location(
            Vec::new(),
            vec![NavPolygon {
                vertices: vec![point(0, 0), point(4, 0), point(8, 0)],
                neighbours: Vec::new(),
            }],
            Vec::new(),
        );
        assert!(matches!(
            sliver.validate(),
            Err(ContentError::NavPolygonWinding { .. })
        ));

        let stub = location(
            Vec::new(),
            vec![NavPolygon {
                vertices: vec![point(0, 0), point(1, 0)],
                neighbours: Vec::new(),
            }],
            Vec::new(),
        );
        assert!(
            matches!(stub.validate(), Err(ContentError::NavPolygonWinding { .. })),
            "a polygon with fewer than three vertices is not a polygon"
        );
    }

    #[test]
    fn nav_neighbours_must_exist() {
        let dangling = location(Vec::new(), vec![square(true, vec![7])], Vec::new());
        assert!(matches!(
            dangling.validate(),
            Err(ContentError::NavNeighbourOutOfRange { neighbour: 7, .. })
        ));

        let paired = location(
            Vec::new(),
            vec![square(true, vec![1]), square(true, vec![0])],
            Vec::new(),
        );
        assert_eq!(paired.validate(), Ok(()), "in-range neighbours are fine");
    }

    #[test]
    fn a_link_must_start_from_a_real_spawn() {
        let broken = location(
            vec![spawn("player-start", SpawnRole::PlayerStart)],
            Vec::new(),
            vec![LocationLink {
                from: SpawnId::new("no-such-gate").expect("valid id"),
                to: LocationId::new("ashfall-lane").expect("valid id"),
                arrive_at: SpawnId::new("lane-entrance").expect("valid id"),
            }],
        );
        assert!(matches!(
            broken.validate(),
            Err(ContentError::UnknownLinkSpawn { .. })
        ));
    }

    fn node_id(name: &str) -> DialogueNodeId {
        DialogueNodeId::new(name).expect("valid id")
    }

    fn reply(next: Option<&str>) -> DialogueOption {
        DialogueOption {
            text: "...".to_owned(),
            conditions: Vec::new(),
            effects: Vec::new(),
            next: next.map(node_id),
        }
    }

    fn dialogue(start: &str, nodes: Vec<(&str, Vec<DialogueOption>)>) -> DialogueDef {
        DialogueDef {
            schema_version: CONTENT_SCHEMA_VERSION,
            id: DialogueId::new("sela-greeting").expect("valid id"),
            speaker: NpcId::new("warden-sela").expect("valid id"),
            conditions: Vec::new(),
            topic: None,
            start: node_id(start),
            nodes: nodes
                .into_iter()
                .map(|(id, options)| {
                    (
                        node_id(id),
                        DialogueNode {
                            text: "...".to_owned(),
                            options,
                        },
                    )
                })
                .collect(),
        }
    }

    #[test]
    fn a_playable_conversation_passes() {
        let good = dialogue(
            "greeting",
            vec![
                ("greeting", vec![reply(Some("detail")), reply(None)]),
                ("detail", vec![reply(None)]),
            ],
        );
        assert_eq!(good.validate(), Ok(()));
    }

    #[test]
    fn dialogue_graphs_must_hold_together() {
        let missing_start = dialogue("nowhere", vec![("greeting", vec![reply(None)])]);
        assert!(matches!(
            missing_start.validate(),
            Err(ContentError::MissingDialogueStart { .. })
        ));

        let dangling = dialogue("greeting", vec![("greeting", vec![reply(Some("ghost"))])]);
        assert!(matches!(
            dangling.validate(),
            Err(ContentError::DanglingDialogueNode { .. })
        ));

        let orphaned = dialogue(
            "greeting",
            vec![("greeting", vec![reply(None)]), ("forgotten", vec![reply(None)])],
        );
        assert!(matches!(
            orphaned.validate(),
            Err(ContentError::UnreachableDialogueNode { .. })
        ));
    }

    /// A conversation with no exit is a soft-lock, so it is a content error
    /// rather than something to discover while playing.
    #[test]
    fn an_inescapable_conversation_is_rejected() {
        let trap = dialogue(
            "greeting",
            vec![
                ("greeting", vec![reply(Some("loop"))]),
                ("loop", vec![reply(Some("greeting"))]),
            ],
        );
        assert!(matches!(
            trap.validate(),
            Err(ContentError::InescapableDialogue(_))
        ));
    }

    fn loot(rolls: u8, entries: Vec<(&str, u32, u32)>) -> LootTableDef {
        LootTableDef {
            schema_version: CONTENT_SCHEMA_VERSION,
            id: LootTableId::new("husk-drops").expect("valid id"),
            rolls,
            entries: entries
                .into_iter()
                .map(|(item, quantity, weight)| LootEntry {
                    item: ItemId::new(item).expect("valid id"),
                    quantity,
                    weight,
                })
                .collect(),
        }
    }

    #[test]
    fn loot_tables_must_be_able_to_drop_something() {
        assert_eq!(loot(2, vec![("ash-cinder", 1, 10)]).validate(), Ok(()));

        assert!(matches!(
            loot(0, vec![("ash-cinder", 1, 10)]).validate(),
            Err(ContentError::ZeroCount { .. })
        ));
        assert!(matches!(
            loot(1, vec![("ash-cinder", 0, 10)]).validate(),
            Err(ContentError::ZeroCount { .. })
        ));
        assert!(matches!(
            loot(1, vec![("ash-cinder", 1, 0)]).validate(),
            Err(ContentError::LootTableCannotDrop(_))
        ));
    }

    fn upgrade(id: &str, requires: Vec<&str>, costs: Vec<(&str, u32)>) -> SettlementUpgradeDef {
        SettlementUpgradeDef {
            schema_version: CONTENT_SCHEMA_VERSION,
            id: UpgradeId::new(id).expect("valid id"),
            name: "Rebuild".to_owned(),
            costs: costs
                .into_iter()
                .map(|(item, quantity)| (ItemId::new(item).expect("valid id"), quantity))
                .collect(),
            requires: requires
                .into_iter()
                .map(|id| UpgradeId::new(id).expect("valid id"))
                .collect(),
            grants_flag: WorldFlag::new("forge-relit").expect("valid id"),
        }
    }

    #[test]
    fn upgrades_reject_impossible_prerequisites_and_costs() {
        assert_eq!(
            upgrade("upgrade-forge", vec![], vec![("ember-ore", 12)]).validate(),
            Ok(())
        );
        assert!(matches!(
            upgrade("upgrade-forge", vec!["upgrade-forge"], vec![("ember-ore", 1)]).validate(),
            Err(ContentError::SelfRequiringUpgrade(_))
        ));
        assert!(matches!(
            upgrade("upgrade-forge", vec![], vec![("ember-ore", 0)]).validate(),
            Err(ContentError::ZeroCount { .. })
        ));
    }

    #[test]
    fn every_new_family_checks_its_schema_version() {
        let mut location = location(Vec::new(), Vec::new(), Vec::new());
        location.schema_version = CONTENT_SCHEMA_VERSION + 1;
        assert!(matches!(
            location.validate(),
            Err(ContentError::SchemaVersion { .. })
        ));

        let mut table = loot(1, vec![("ash-cinder", 1, 1)]);
        table.schema_version = CONTENT_SCHEMA_VERSION + 1;
        assert!(matches!(
            table.validate(),
            Err(ContentError::SchemaVersion { .. })
        ));

        // Faction ids are unused here beyond proving the import is exercised by
        // a real construction rather than left dangling.
        assert!(FactionId::new("cinderwake-settlers").is_ok());
        assert!(PrefabId::new("corrupted-husk").is_ok());
    }
}
