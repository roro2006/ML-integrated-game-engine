//! Choosing one thing from what is currently on offer.
//!
//! Three places in the slice ask the player to pick from a list: a
//! conversation's lines, a forge's recipes, and the settlement board's
//! blueprints. Each of them used to take the *first* thing on offer, because
//! there was no way to ask. This module is the asking — the model of it, with
//! no pixels: offered rows out, a chosen row in.
//!
//! It is deliberately not a widget. Nothing here draws, owns a window, or
//! knows the ECS exists, which is what lets the whole of selection be tested
//! headless. When text rendering lands, a menu draws [`rows`] and calls
//! [`take`]; until then the caller logs [`lines`] and reads digit keys. The
//! code a menu will exercise is the code that runs today.
//!
//! # Rows are recomputed, never stored
//!
//! [`rows`] is a pure function of content and canonical state, and an [`Offer`]
//! holds no list. A stored list would be a second source of truth: the world
//! can change between the tick a menu was drawn and the tick a key was pressed
//! — an enemy's hit spends nothing, but a crafted item does — and the stored
//! copy would then offer something that is no longer true. This is the same
//! rule the quest runtime follows when it computes open objectives rather than
//! keeping an "active node" pointer.
//!
//! # A pick names a thing, not a position
//!
//! [`Pick`] carries an id, never a row number. A digit is presentation: it is
//! resolved against a freshly computed list at the moment of the press, and
//! from there on the choice travels as "the ember glaive", not as "row 2". So
//! a list that grew or shrank in between cannot silently redirect a choice onto
//! a different recipe. Dialogue is the exception that proves it — a
//! [`Pick::Say`] carries an index because `narrative::Choice` already defines
//! that index against the node's *full* option list, so it does not move when
//! filtering hides a gated line.
//!
//! # Taking re-checks, and never trusts the caller
//!
//! [`take`] re-runs the same rule that decided whether to offer the row at all.
//! A recipe that was affordable when the list was drawn and is not when the key
//! is pressed is refused, not executed. That mirrors `Conversation::choose`,
//! which re-evaluates an option's conditions rather than trusting an index —
//! otherwise the one place a player could act on a stale offer would be a UI
//! bug. Affordability is still `interaction`'s rule and the reducer is still
//! the authority; this is a third *check* of one rule, not a second
//! implementation of it.

use content_runtime::ContentRegistry;
use game_core::content::ItemCategory;
use game_core::events::DomainEvent;
use game_core::ids::{ChoiceId, ClassId, DialogueId, ItemId, NpcId, UpgradeId};
use game_core::world::{EquipmentSlot, WorldState};
use narrative::{Conversation, DialogueError};

use crate::interaction::{self, InteractionError};

/// What the player currently has open.
///
/// Runtime state. A conversation cursor is deliberately not serializable
/// (`narrative::dialogue`), and the same reasoning covers the other two: a save
/// taken standing at a forge reloads standing at a forge, with the menu closed
/// and nothing lost, because everything a menu can do travels as a domain
/// event the moment it is done.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Offer {
    /// An NPC with more than one conversation open, waiting for the player to
    /// pick a topic (M9). Picking one becomes [`Offer::Talking`].
    Topics(NpcId),
    /// A conversation in progress.
    Talking(Conversation),
    /// A crafting station's recipes.
    Forge,
    /// The settlement board's blueprints.
    Board,
    /// Something written, to read and put down: an examined clue's text
    /// (M9). Offers nothing to pick.
    Note { title: String, text: String },
    /// A choice marker asking to be confirmed (M9), so a decision is never
    /// taken by walking into it. `settled` flips when the player confirms,
    /// which is what closes it.
    Decision {
        choice: ChoiceId,
        prompt: String,
        settled: bool,
    },
    /// The quest journal (M9). Read-only; what it says is drawn from state by
    /// whoever shows it.
    Journal,
    /// Who to play as, offered once on a fresh start when content defines
    /// more than one class (M9). `chosen` flips on a pick, which closes it.
    ClassChoice { chosen: bool },
    /// The list of controls (M9). Read-only.
    Controls,
    /// What the player is carrying that can be worn or wielded.
    ///
    /// The only offer with no fixture in the world: a forge and a board are
    /// places you stand at, and equipment is not. That is a deliberate break in
    /// the pattern rather than an oversight — inventing an armoury to stand in
    /// would be authoring a location to justify a menu.
    Equipment,
}

/// One thing the player could choose, named in a way that cannot drift.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pick {
    /// A conversation to open, by id, from an NPC's topics.
    Topic(DialogueId),
    /// A dialogue option, by its index into the node's full option list.
    Say { option: usize },
    /// A recipe, by its own id. Recipes are keyed by id in the registry, and
    /// `RecipeDef::id` is an [`ItemId`] because a recipe is named for what it
    /// makes.
    Craft(ItemId),
    /// A settlement blueprint, by id.
    Build(UpgradeId),
    /// Settle the decision on offer.
    Decide(ChoiceId),
    /// Play as this class (M9).
    Class(ClassId),
    /// Use a consumable (M9). Using takes a committed moment in the world, so
    /// the pick only asks for it; see [`Outcome::used`].
    Use(ItemId),
    /// An item to wear or wield, by id.
    ///
    /// The slot is deliberately *not* carried. It is derived from the item's
    /// category when the pick is taken, so a pick cannot name a slot the item
    /// does not belong in — the same reason a craft carries a recipe id rather
    /// than a row number.
    Equip(ItemId),
}

/// Which slot an item of this category occupies, if any.
///
/// The first place in the project that maps a content category onto an
/// equipment slot, and deliberately a small total function rather than a field
/// on [`ItemDef`]. Two thirds of the categories equip nothing, and a `slot:`
/// field would let content put a quest item in a weapon hand.
///
/// [`EquipmentSlot::Offhand`] and [`EquipmentSlot::Trinket`] are unreachable:
/// no category maps onto them and no item is authored for them. That is left
/// honest rather than papered over — a mapping invented here for slots nothing
/// can fill would be exactly the unreachable content ADR-0016 objects to.
///
/// [`ItemDef`]: game_core::content::ItemDef
pub fn slot_for(category: ItemCategory) -> Option<EquipmentSlot> {
    match category {
        ItemCategory::Weapon => Some(EquipmentSlot::Weapon),
        ItemCategory::Armor => Some(EquipmentSlot::Armor),
        ItemCategory::Material | ItemCategory::Consumable | ItemCategory::QuestItem => None,
    }
}

/// One offered row: what to show, and what choosing it means.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub label: String,
    pub pick: Pick,
}

/// Why a selection could not be taken.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectionError {
    /// The conversation refused the line — it is gated off, out of range, or
    /// the conversation has already closed.
    Dialogue(DialogueError),
    /// The cost rule refused it: too few inputs, a blueprint not yet standing,
    /// or one already built.
    Interaction(InteractionError),
    /// The pick does not belong to the offer that is open — a craft pressed at
    /// the upgrade board, say. Refused rather than coerced, because the two
    /// lists are different lists.
    WrongOffer,
    /// The conversation's definition is no longer in content.
    UnknownDialogue,
    /// The topic is not one the NPC on offer holds right now: someone else's,
    /// or gated shut since the list was drawn.
    ClosedTopic(DialogueId),
    /// The pick names a recipe content does not define.
    UnknownRecipe(ItemId),
    /// The pick names a blueprint content does not define.
    UnknownUpgrade(UpgradeId),
    /// The pick names an item content does not define.
    UnknownItem(ItemId),
    /// The item exists but belongs in no slot — a material, a draught, a seal.
    NotEquippable(ItemId),
    /// The item does nothing when used.
    NotUsable(ItemId),
    /// The pick names a class content does not define.
    UnknownClass(ClassId),
    /// The item is not in the player's inventory any more. The equipment
    /// reducer does not check this, so the check has to live here: see the
    /// note on the [`Pick::Equip`] arm of [`take`].
    NotHeld(ItemId),
}

impl std::fmt::Display for SelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dialogue(error) => write!(f, "{error}"),
            Self::Interaction(error) => write!(f, "{error}"),
            Self::WrongOffer => write!(f, "that choice belongs to a different offer"),
            Self::UnknownDialogue => write!(f, "the conversation is no longer defined"),
            Self::ClosedTopic(dialogue) => write!(f, "{dialogue} is not open to talk about"),
            Self::UnknownRecipe(recipe) => write!(f, "no such recipe: {recipe}"),
            Self::UnknownUpgrade(upgrade) => write!(f, "no such upgrade: {upgrade}"),
            Self::UnknownItem(item) => write!(f, "no such item: {item}"),
            Self::NotEquippable(item) => write!(f, "{item} cannot be equipped"),
            Self::NotUsable(item) => write!(f, "{item} does nothing when used"),
            Self::UnknownClass(class) => write!(f, "no such class: {class}"),
            Self::NotHeld(item) => write!(f, "{item} is not carried"),
        }
    }
}

impl std::error::Error for SelectionError {}

/// What taking a row produced.
///
/// Events and choices travel separately for the reason `narrative::Turn`
/// already establishes: events go through the reducer, while a settled choice
/// is a signal a quest consumes this tick. `crafted` is the third, for the same
/// kind of reason — a crafted item feeds a `Collect` objective, and the caller
/// raises that signal.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    /// Durable consequences, for the caller to apply.
    pub events: Vec<DomainEvent>,
    /// Authored decisions this choice settled.
    pub choices: Vec<ChoiceId>,
    /// What a craft produced, if it produced anything.
    pub crafted: Option<ItemId>,
    /// A consumable the player asked to use (M9). Nothing is spent here:
    /// using takes time, and the caller starts it in the world.
    pub used: Option<ItemId>,
}

/// The rows currently on offer, in a deterministic order.
///
/// Recipes and blueprints come out in id order, because the registry is keyed
/// by id — two runs of the same tick offer the same list in the same sequence,
/// which is what lets a replayed log press the same digit and get the same
/// thing. Dialogue keeps its authored order, because the order lines are
/// written in is the order they are meant to be read in.
///
/// An empty list is a legible answer: a forge with nothing affordable, or a
/// node whose every line is gated off. Neither is an error, and walking away
/// is always available.
pub fn rows(offer: &Offer, content: &ContentRegistry, state: &WorldState) -> Vec<Row> {
    match offer {
        // Untitled conversations cannot be here: the loader refuses a speaker
        // with several conversations unless each has a topic. The id is a
        // fallback for content that bypassed the loader, never a label anyone
        // authored.
        Offer::Topics(npc) => narrative::open_conversations(content.dialogues.values(), npc, state)
            .into_iter()
            .map(|def| Row {
                label: def.topic.clone().unwrap_or_else(|| def.id.to_string()),
                pick: Pick::Topic(def.id.clone()),
            })
            .collect(),
        Offer::Talking(conversation) => {
            let Some(def) = content.dialogues.get(conversation.dialogue()) else {
                return Vec::new();
            };
            conversation
                .options(def, state)
                .into_iter()
                .map(|choice| Row {
                    label: choice.text.to_owned(),
                    pick: Pick::Say { option: choice.index },
                })
                .collect()
        }
        Offer::Decision {
            choice,
            settled: false,
            ..
        } => vec![Row {
            label: "Do it.".to_owned(),
            pick: Pick::Decide(choice.clone()),
        }],
        // Every class, in id order, with the pools it starts with.
        Offer::ClassChoice { chosen: false } => content
            .classes
            .values()
            .map(|class| Row {
                label: format!(
                    "{} - {} health, {} stamina",
                    class.name,
                    class.health_pool(),
                    class.stamina_pool()
                ),
                pick: Pick::Class(class.id.clone()),
            })
            .collect(),
        Offer::ClassChoice { chosen: true }
        | Offer::Decision { .. }
        | Offer::Note { .. }
        | Offer::Journal
        | Offer::Controls => Vec::new(),
        Offer::Forge => content
            .recipes
            .values()
            .filter(|recipe| interaction::craft(recipe, content, state).is_ok())
            .map(|recipe| Row {
                label: recipe.name.clone(),
                pick: Pick::Craft(recipe.id.clone()),
            })
            .collect(),
        Offer::Board => content
            .upgrades
            .values()
            .filter(|upgrade| interaction::build_upgrade(upgrade, content, state).is_ok())
            .map(|upgrade| Row {
                label: upgrade.name.clone(),
                pick: Pick::Build(upgrade.id.clone()),
            })
            .collect(),
        // Driven by the inventory rather than by the item registry, because the
        // offer is "what you are carrying", not "what exists". A `BTreeMap` is
        // already in id order, so the list is deterministic for free.
        // Wearables to equip, then consumables to use (M9), each in id order.
        Offer::Equipment => {
            let held = || state.player.inventory.iter().filter(|(_, held)| **held > 0);
            let wear = held().filter_map(|(id, _)| {
                let def = content.items.get(id)?;
                slot_for(def.category)?;
                Some(Row {
                    label: def.name.clone(),
                    pick: Pick::Equip(id.clone()),
                })
            });
            let usable = held().filter_map(|(id, count)| {
                let def = content.items.get(id)?;
                def.on_use.as_ref()?;
                Some(Row {
                    label: format!("Use {} ({count})", def.name),
                    pick: Pick::Use(id.clone()),
                })
            });
            wear.chain(usable).collect()
        }
    }
}

/// The caption above the rows: what is being said, or what is being used.
///
/// A conversation's prompt is authored content — the line the NPC is speaking.
/// The stations' are bare labels rather than invented dialogue, because putting
/// a written-sounding sentence in Rust would make content out of source.
pub fn prompt(offer: &Offer, content: &ContentRegistry) -> Option<String> {
    match offer {
        Offer::Topics(_) => Some("topics".to_owned()),
        Offer::Note { text, .. } => Some(text.clone()),
        Offer::Decision { prompt, .. } => Some(prompt.clone()),
        Offer::Journal => Some("journal".to_owned()),
        Offer::ClassChoice { .. } => Some("class".to_owned()),
        Offer::Controls => Some("controls".to_owned()),
        Offer::Talking(conversation) => content
            .dialogues
            .get(conversation.dialogue())
            .and_then(|def| conversation.node(def))
            .map(|node| node.text.clone()),
        Offer::Forge => Some("forge".to_owned()),
        Offer::Board => Some("upgrade board".to_owned()),
        Offer::Equipment => Some("equipment".to_owned()),
    }
}

/// The offered rows as numbered text, for a log or an overlay.
///
/// Numbered from 1 because that is the key the player presses. The same
/// function serves a terminal, a log file, and — when text rendering exists —
/// the screen, which matters because the machine where something looks wrong is
/// usually the one without a debugger attached.
pub fn lines(rows: &[Row]) -> Vec<String> {
    rows.iter()
        .enumerate()
        .map(|(index, row)| format!("{}) {}", index + 1, row.label))
        .collect()
}

/// What the nth offered row means, counting from 1.
///
/// Recomputes the list rather than consulting a remembered one, so a digit
/// pressed a tick after the menu was drawn resolves against what is true now.
/// Returns `None` for 0 and for any digit past the end of the offer — pressing
/// 5 at a forge with two recipes does nothing, rather than wrapping around onto
/// something the player did not mean.
pub fn pick_at(offer: &Offer, content: &ContentRegistry, state: &WorldState, digit: u32) -> Option<Pick> {
    let index = usize::try_from(digit.checked_sub(1)?).ok()?;
    rows(offer, content, state).get(index).map(|row| row.pick.clone())
}

/// Takes a choice, returning what it produced.
///
/// Re-checks the rule that offered the row in the first place: see the module
/// docs. Advances the conversation cursor on a dialogue pick; the stations have
/// no cursor and stay open, so a player can make two things without walking
/// away and back.
pub fn take(
    offer: &mut Offer,
    content: &ContentRegistry,
    state: &WorldState,
    pick: &Pick,
) -> Result<Outcome, SelectionError> {
    match pick {
        // Opening a conversation changes nothing durable: what it says and
        // does happens when the player speaks in it.
        Pick::Topic(dialogue) => {
            let Offer::Topics(npc) = offer else {
                return Err(SelectionError::WrongOffer);
            };
            let def = content
                .dialogues
                .get(dialogue)
                .ok_or(SelectionError::UnknownDialogue)?;
            if &def.speaker != npc || !narrative::is_open(def, state) {
                return Err(SelectionError::ClosedTopic(dialogue.clone()));
            }
            *offer = Offer::Talking(Conversation::open(def));
            Ok(Outcome::default())
        }
        // Choosing re-grants the class's pools in full: this is only offered
        // before anything has happened to the character.
        Pick::Class(class) => {
            let Offer::ClassChoice {
                chosen: chosen @ false,
            } = offer
            else {
                return Err(SelectionError::WrongOffer);
            };
            let def = content
                .classes
                .get(class)
                .ok_or_else(|| SelectionError::UnknownClass(class.clone()))?;
            *chosen = true;
            Ok(Outcome {
                events: vec![DomainEvent::ClassChosen {
                    class: class.clone(),
                    health: def.health_pool(),
                    stamina: def.stamina_pool(),
                }],
                ..Outcome::default()
            })
        }
        Pick::Use(item) => {
            if !matches!(offer, Offer::Equipment) {
                return Err(SelectionError::WrongOffer);
            }
            let def = content
                .items
                .get(item)
                .ok_or_else(|| SelectionError::UnknownItem(item.clone()))?;
            if def.on_use.is_none() {
                return Err(SelectionError::NotUsable(item.clone()));
            }
            if state.player.inventory.get(item).copied().unwrap_or(0) == 0 {
                return Err(SelectionError::NotHeld(item.clone()));
            }
            Ok(Outcome {
                used: Some(item.clone()),
                ..Outcome::default()
            })
        }
        Pick::Decide(choice) => {
            let Offer::Decision {
                choice: offered,
                settled,
                ..
            } = offer
            else {
                return Err(SelectionError::WrongOffer);
            };
            if offered != choice || *settled {
                return Err(SelectionError::WrongOffer);
            }
            *settled = true;
            Ok(Outcome {
                events: Vec::new(),
                choices: vec![choice.clone()],
                crafted: None,
                used: None,
            })
        }
        Pick::Say { option } => {
            let Offer::Talking(conversation) = offer else {
                return Err(SelectionError::WrongOffer);
            };
            let def = content
                .dialogues
                .get(conversation.dialogue())
                .ok_or(SelectionError::UnknownDialogue)?;
            let turn = conversation
                .choose(def, state, *option)
                .map_err(SelectionError::Dialogue)?;
            Ok(Outcome {
                events: turn.events,
                choices: turn.choices,
                crafted: None,
                used: None,
            })
        }
        Pick::Craft(recipe) => {
            if !matches!(offer, Offer::Forge) {
                return Err(SelectionError::WrongOffer);
            }
            let def = content
                .recipes
                .get(recipe)
                .ok_or_else(|| SelectionError::UnknownRecipe(recipe.clone()))?;
            let events = interaction::craft(def, content, state).map_err(SelectionError::Interaction)?;
            Ok(Outcome {
                events,
                choices: Vec::new(),
                // Only what actually granted something counts as crafted: a
                // recipe with a zero output produces no grant event, and
                // signalling `Collected` for it would advance an objective on
                // an item the player never received.
                crafted: (def.output_quantity > 0).then(|| def.output.clone()),
                used: None,
            })
        }
        Pick::Build(upgrade) => {
            if !matches!(offer, Offer::Board) {
                return Err(SelectionError::WrongOffer);
            }
            let def = content
                .upgrades
                .get(upgrade)
                .ok_or_else(|| SelectionError::UnknownUpgrade(upgrade.clone()))?;
            let events =
                interaction::build_upgrade(def, content, state).map_err(SelectionError::Interaction)?;
            Ok(Outcome {
                events,
                choices: Vec::new(),
                crafted: None,
                used: None,
            })
        }
        Pick::Equip(item) => {
            if !matches!(offer, Offer::Equipment) {
                return Err(SelectionError::WrongOffer);
            }
            let def = content
                .items
                .get(item)
                .ok_or_else(|| SelectionError::UnknownItem(item.clone()))?;
            let slot = slot_for(def.category).ok_or_else(|| SelectionError::NotEquippable(item.clone()))?;
            // The one check the reducer does not make. `ItemEquipped` validates
            // that the id is canonical and then fills the slot — it neither
            // requires the item to be held nor consumes it — so "you must
            // actually have it" is this module's rule to enforce, exactly as
            // affordability is `interaction`'s. Without it a row drawn before a
            // craft spent the last ore would still equip afterwards.
            if state.player.inventory.get(item).copied().unwrap_or(0) == 0 {
                return Err(SelectionError::NotHeld(item.clone()));
            }
            Ok(Outcome {
                events: vec![DomainEvent::ItemEquipped {
                    slot,
                    item: item.clone(),
                }],
                choices: Vec::new(),
                crafted: None,
                used: None,
            })
        }
    }
}

/// Whether the offer has closed itself.
///
/// Only a conversation can: it runs out of nodes. A station is left by walking
/// away, so it never ends on its own — a forge with nothing affordable is an
/// empty list, not a closed menu, and the difference is what the player sees.
pub fn is_over(offer: &Offer) -> bool {
    match offer {
        Offer::Talking(conversation) => conversation.is_over(),
        // Equipment closes like a station rather than like a conversation:
        // changing one thing is rarely changing only one thing, and a list that
        // shut itself after a single pick would make swapping a weapon and
        // armour two trips.
        // A decision closes once it is made; there is nothing left to do at it.
        Offer::Decision { settled, .. } => *settled,
        Offer::ClassChoice { chosen } => *chosen,
        Offer::Topics(_)
        | Offer::Note { .. }
        | Offer::Journal
        | Offer::Controls
        | Offer::Forge
        | Offer::Board
        | Offer::Equipment => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::content::{
        CONTENT_SCHEMA_VERSION, Condition, DialogueDef, DialogueEffect, DialogueNode, DialogueOption,
        ItemDef, RecipeDef, SettlementUpgradeDef,
    };
    use game_core::ids::{ContentRevisionId, DialogueId, DialogueNodeId, NpcId, QuestId, WorldFlag};
    use std::collections::BTreeMap;

    fn item(name: &str) -> ItemId {
        ItemId::new(name).expect("valid id")
    }

    fn upgrade_id(name: &str) -> UpgradeId {
        UpgradeId::new(name).expect("valid id")
    }

    fn flag(name: &str) -> WorldFlag {
        WorldFlag::new(name).expect("valid id")
    }

    fn node_id(name: &str) -> DialogueNodeId {
        DialogueNodeId::new(name).expect("valid id")
    }

    fn world() -> WorldState {
        WorldState::new(7, ContentRevisionId::new("rev-0001").expect("valid id"))
    }

    fn recipe(name: &str, inputs: &[(&str, u32)], output: &str) -> RecipeDef {
        RecipeDef {
            schema_version: CONTENT_SCHEMA_VERSION,
            id: item(name),
            name: name.to_owned(),
            inputs: inputs
                .iter()
                .map(|(id, quantity)| (item(id), *quantity))
                .collect(),
            output: item(output),
            output_quantity: 1,
            requires_upgrade: None,
        }
    }

    fn blueprint(name: &str, costs: &[(&str, u32)], grants: &str) -> SettlementUpgradeDef {
        SettlementUpgradeDef {
            schema_version: CONTENT_SCHEMA_VERSION,
            id: upgrade_id(name),
            name: name.to_owned(),
            costs: costs.iter().map(|(id, quantity)| (item(id), *quantity)).collect(),
            requires: Vec::new(),
            grants_flag: flag(grants),
        }
    }

    /// A forge offering two recipes, one of them gated on a blueprint.
    fn forge_content() -> ContentRegistry {
        let mut content = ContentRegistry::default();
        let draught = recipe("clearwater-draught", &[("still-reed", 2)], "clearwater-draught");
        let mut glaive = recipe("ember-glaive", &[("ember-ore", 3)], "ember-glaive");
        glaive.requires_upgrade = Some(upgrade_id("upgrade-forge"));

        content.recipes.insert(draught.id.clone(), draught);
        content.recipes.insert(glaive.id.clone(), glaive);

        let forge = blueprint("upgrade-forge", &[("ember-ore", 6)], "forge-relit");
        content.upgrades.insert(forge.id.clone(), forge);
        content
    }

    fn option(
        text: &str,
        conditions: Vec<Condition>,
        effects: Vec<DialogueEffect>,
        next: Option<&str>,
    ) -> DialogueOption {
        DialogueOption {
            text: text.to_owned(),
            conditions,
            effects,
            next: next.map(node_id),
        }
    }

    /// A greeting whose middle line is gated, so filtering has something to do.
    fn talk_content() -> ContentRegistry {
        let mut nodes = BTreeMap::new();
        nodes.insert(
            node_id("greeting"),
            DialogueNode {
                text: "The intake is fouled.".to_owned(),
                options: vec![
                    option(
                        "I will clear it.",
                        Vec::new(),
                        vec![DialogueEffect::StartQuest(
                            QuestId::new("still-waters").expect("valid id"),
                        )],
                        Some("thanks"),
                    ),
                    option(
                        "Here are the reeds.",
                        vec![Condition::HasItem {
                            item: item("still-reed"),
                            quantity: 2,
                        }],
                        vec![DialogueEffect::SetWorldFlag(flag("intake-cleared"))],
                        None,
                    ),
                    option("Not now.", Vec::new(), Vec::new(), None),
                ],
            },
        );
        nodes.insert(
            node_id("thanks"),
            DialogueNode {
                text: "Fortune go with you.".to_owned(),
                options: Vec::new(),
            },
        );

        let def = DialogueDef {
            schema_version: CONTENT_SCHEMA_VERSION,
            id: DialogueId::new("sela-intake").expect("valid id"),
            speaker: NpcId::new("warden-sela").expect("valid id"),
            conditions: Vec::new(),
            topic: None,
            start: node_id("greeting"),
            nodes,
        };

        let mut content = ContentRegistry::default();
        content.dialogues.insert(def.id.clone(), def);
        content
    }

    fn talking(content: &ContentRegistry) -> Offer {
        let def = content
            .dialogues
            .get(&DialogueId::new("sela-intake").expect("valid id"))
            .expect("the fixture defines it");
        Offer::Talking(Conversation::open(def))
    }

    /// Sela with two conversations: the intake, always open, and a second
    /// open only once the intake is cleared. Both titled, as the loader
    /// demands of a speaker with several.
    fn topic_content() -> ContentRegistry {
        let mut content = talk_content();
        let sela = NpcId::new("warden-sela").expect("valid id");
        let mut intake = content
            .dialogues
            .values()
            .next()
            .cloned()
            .expect("the talk fixture has one");
        intake.topic = Some("The intake".to_owned());
        let mut after = intake.clone();
        after.id = DialogueId::new("sela-after").expect("valid id");
        after.topic = Some("What now?".to_owned());
        after.conditions = vec![Condition::WorldFlagSet(flag("intake-cleared"))];
        let mut stranger = intake.clone();
        stranger.id = DialogueId::new("thal-terms").expect("valid id");
        stranger.speaker = NpcId::new("arbiter-thal").expect("valid id");
        assert_eq!(intake.speaker, sela);
        content.dialogues = [intake, after, stranger]
            .into_iter()
            .map(|def| (def.id.clone(), def))
            .collect();
        content
    }

    #[test]
    fn topics_offer_only_what_the_speaker_holds_open() {
        let content = topic_content();
        let sela = Offer::Topics(NpcId::new("warden-sela").expect("valid id"));
        let mut state = world();
        let labels = |state: &WorldState| {
            rows(&sela, &content, state)
                .into_iter()
                .map(|row| row.label)
                .collect::<Vec<_>>()
        };
        assert_eq!(labels(&state), vec!["The intake"], "gated shut, and not Thal's");

        state.world_flags.insert(flag("intake-cleared"));
        assert_eq!(
            labels(&state),
            vec!["What now?", "The intake"],
            "in id order: sela-after, then sela-intake"
        );
        assert!(!is_over(&sela), "a topic list waits to be picked from");
    }

    #[test]
    fn picking_a_topic_opens_that_conversation_and_nothing_else() {
        let content = topic_content();
        let mut state = world();
        state.world_flags.insert(flag("intake-cleared"));
        let mut offer = Offer::Topics(NpcId::new("warden-sela").expect("valid id"));

        let pick = pick_at(&offer, &content, &state, 1).expect("two topics");
        let outcome = take(&mut offer, &content, &state, &pick).expect("an open topic");
        assert_eq!(outcome, Outcome::default(), "opening changes nothing durable");
        match &offer {
            Offer::Talking(conversation) => assert_eq!(conversation.dialogue().name(), "sela-after"),
            other => panic!("expected a conversation, got {other:?}"),
        }
    }

    #[test]
    fn a_topic_that_is_shut_or_someone_elses_is_refused() {
        let content = topic_content();
        let state = world();
        let sela = || Offer::Topics(NpcId::new("warden-sela").expect("valid id"));

        for dialogue in ["sela-after", "thal-terms"] {
            let mut offer = sela();
            let pick = Pick::Topic(DialogueId::new(dialogue).expect("valid id"));
            assert_eq!(
                take(&mut offer, &content, &state, &pick),
                Err(SelectionError::ClosedTopic(
                    DialogueId::new(dialogue).expect("valid id")
                )),
            );
            assert_eq!(offer, sela(), "a refused topic leaves the list open");
        }
    }

    #[test]
    fn only_what_the_player_can_actually_do_is_offered() {
        let content = forge_content();
        let mut state = world();
        state.player.inventory.insert(item("still-reed"), 2);
        state.player.inventory.insert(item("ember-ore"), 3);

        let offered = rows(&Offer::Forge, &content, &state);
        assert_eq!(
            offered.iter().map(|row| &row.label).collect::<Vec<_>>(),
            vec!["clearwater-draught"],
            "the glaive is affordable but its blueprint is not standing"
        );
    }

    #[test]
    fn an_empty_offer_is_a_list_not_an_error() {
        let content = forge_content();
        let state = world();
        assert!(rows(&Offer::Forge, &content, &state).is_empty());
        assert!(
            !is_over(&Offer::Forge),
            "nothing to make is not the same as the menu closing"
        );
    }

    #[test]
    fn an_already_built_blueprint_stops_being_offered() {
        let content = forge_content();
        let mut state = world();
        state.player.inventory.insert(item("ember-ore"), 6);
        assert_eq!(rows(&Offer::Board, &content, &state).len(), 1);

        state.world_flags.insert(flag("forge-relit"));
        assert!(
            rows(&Offer::Board, &content, &state).is_empty(),
            "a blueprint already standing is not on offer"
        );
    }

    /// The rule the whole module exists for: the list is a view, not a promise.
    #[test]
    fn a_row_that_went_stale_is_refused_rather_than_taken() {
        let content = forge_content();
        let mut state = world();
        state.player.inventory.insert(item("still-reed"), 2);

        let pick = pick_at(&Offer::Forge, &content, &state, 1).expect("the draught is offered");

        // The world moves on between the menu being drawn and the key landing.
        state.player.inventory.insert(item("still-reed"), 0);

        let mut offer = Offer::Forge;
        assert_eq!(
            take(&mut offer, &content, &state, &pick),
            Err(SelectionError::Interaction(InteractionError::MissingInputs {
                item: item("still-reed"),
                held: 0,
                needed: 2,
            })),
            "taking re-checks the rule that offered the row"
        );
    }

    /// A pick survives the list changing shape underneath it.
    #[test]
    fn a_pick_names_a_recipe_not_a_row() {
        let mut content = forge_content();
        let mut state = world();
        state.player.inventory.insert(item("still-reed"), 2);

        let pick = pick_at(&Offer::Forge, &content, &state, 1).expect("the draught is offered");

        // A recipe that sorts before the draught appears and is affordable, so
        // row 1 is now something else entirely.
        let ash = recipe("ash-cinder", &[("still-reed", 1)], "ash-cinder");
        content.recipes.insert(ash.id.clone(), ash);
        assert_eq!(
            rows(&Offer::Forge, &content, &state)[0].label,
            "ash-cinder",
            "the fixture only tests what it claims if row 1 really moved"
        );

        let mut offer = Offer::Forge;
        let outcome = take(&mut offer, &content, &state, &pick).expect("still valid");
        assert_eq!(
            outcome.crafted,
            Some(item("clearwater-draught")),
            "the pick made what it named, not whatever is now first"
        );
    }

    #[test]
    fn a_digit_past_the_end_of_the_offer_picks_nothing() {
        let content = forge_content();
        let mut state = world();
        state.player.inventory.insert(item("still-reed"), 2);

        assert!(
            pick_at(&Offer::Forge, &content, &state, 0).is_none(),
            "digits count from 1"
        );
        assert!(pick_at(&Offer::Forge, &content, &state, 2).is_none());
        assert!(pick_at(&Offer::Forge, &content, &state, 9).is_none());
    }

    #[test]
    fn crafting_reports_its_output_so_a_collect_objective_notices() {
        let content = forge_content();
        let mut state = world();
        state.player.inventory.insert(item("still-reed"), 2);

        let mut offer = Offer::Forge;
        let pick = Pick::Craft(item("clearwater-draught"));
        let outcome = take(&mut offer, &content, &state, &pick).expect("affordable");

        assert_eq!(outcome.crafted, Some(item("clearwater-draught")));
        assert!(
            outcome.events.iter().any(|event| matches!(
                event,
                DomainEvent::ItemGranted { item: granted, .. } if granted == &item("clearwater-draught")
            )),
            "and the grant is what makes it durable"
        );
    }

    #[test]
    fn a_zero_output_recipe_reports_nothing_crafted() {
        let mut content = forge_content();
        let mut state = world();
        state.player.inventory.insert(item("still-reed"), 2);

        let mut empty = recipe("clearwater-draught", &[("still-reed", 2)], "clearwater-draught");
        empty.output_quantity = 0;
        content.recipes.insert(empty.id.clone(), empty);

        let mut offer = Offer::Forge;
        let outcome = take(
            &mut offer,
            &content,
            &state,
            &Pick::Craft(item("clearwater-draught")),
        )
        .expect("the inputs are held");
        assert_eq!(
            outcome.crafted, None,
            "signalling a collect for an item that was never granted would \
             advance an objective the player did not satisfy"
        );
    }

    #[test]
    fn a_pick_meant_for_another_offer_is_refused() {
        let content = forge_content();
        let state = world();

        let mut board = Offer::Board;
        assert_eq!(
            take(
                &mut board,
                &content,
                &state,
                &Pick::Craft(item("clearwater-draught"))
            ),
            Err(SelectionError::WrongOffer)
        );

        let mut forge = Offer::Forge;
        assert_eq!(
            take(
                &mut forge,
                &content,
                &state,
                &Pick::Build(upgrade_id("upgrade-forge"))
            ),
            Err(SelectionError::WrongOffer)
        );
        assert_eq!(
            take(&mut forge, &content, &state, &Pick::Say { option: 0 }),
            Err(SelectionError::WrongOffer)
        );
    }

    #[test]
    fn an_unknown_recipe_is_named_in_the_refusal() {
        let content = forge_content();
        let state = world();
        let mut offer = Offer::Forge;
        assert_eq!(
            take(&mut offer, &content, &state, &Pick::Craft(item("nothing-real"))),
            Err(SelectionError::UnknownRecipe(item("nothing-real")))
        );
    }

    /// Dialogue rows keep indices into the *full* option list, so hiding a line
    /// cannot map a press onto the wrong one.
    #[test]
    fn dialogue_picks_carry_full_list_indices() {
        let content = talk_content();
        let state = world();
        let offer = talking(&content);

        let offered = rows(&offer, &content, &state);
        assert_eq!(
            offered
                .iter()
                .map(|row| (row.label.as_str(), row.pick.clone()))
                .collect::<Vec<_>>(),
            vec![
                ("I will clear it.", Pick::Say { option: 0 }),
                ("Not now.", Pick::Say { option: 2 }),
            ],
            "the gated middle line is hidden, and the last line keeps index 2"
        );

        assert_eq!(
            pick_at(&offer, &content, &state, 2),
            Some(Pick::Say { option: 2 }),
            "pressing 2 takes the second offered row, not option 1"
        );
    }

    #[test]
    fn a_hidden_dialogue_line_cannot_be_taken_through_the_api() {
        let content = talk_content();
        let state = world();
        let mut offer = talking(&content);

        assert_eq!(
            take(&mut offer, &content, &state, &Pick::Say { option: 1 }),
            Err(SelectionError::Dialogue(DialogueError::UnavailableOption {
                index: 1
            })),
            "the reeds line is gated off, so no press may reach it"
        );
    }

    #[test]
    fn taking_a_line_advances_the_conversation_and_can_end_it() {
        let content = talk_content();
        let state = world();
        let mut offer = talking(&content);

        let outcome =
            take(&mut offer, &content, &state, &Pick::Say { option: 0 }).expect("the offer is available");
        assert_eq!(outcome.events.len(), 1, "accepting the quest is one event");
        assert!(!is_over(&offer), "it leads to the closing node");
        assert!(
            rows(&offer, &content, &state).is_empty(),
            "which has nothing left to say"
        );

        // Walking away from a node with no lines is how it ends.
        let Offer::Talking(conversation) = &mut offer else {
            unreachable!("still a conversation");
        };
        conversation.end();
        assert!(is_over(&offer));
    }

    #[test]
    fn offered_rows_are_numbered_from_the_key_that_takes_them() {
        let content = talk_content();
        let state = world();
        let offer = talking(&content);

        assert_eq!(
            lines(&rows(&offer, &content, &state)),
            vec!["1) I will clear it.".to_owned(), "2) Not now.".to_owned()],
        );
        assert_eq!(
            prompt(&offer, &content).as_deref(),
            Some("The intake is fouled."),
            "the caption is the line being spoken"
        );
    }

    /// Two runs of the same tick must offer the same list in the same order, or
    /// a replayed log pressing the same digit would get something else.
    #[test]
    fn an_offer_is_deterministic() {
        let content = forge_content();
        let mut state = world();
        state.player.inventory.insert(item("still-reed"), 2);
        state.player.inventory.insert(item("ember-ore"), 6);

        assert_eq!(
            rows(&Offer::Board, &content, &state),
            rows(&Offer::Board, &content, &state)
        );
        assert_eq!(
            rows(&Offer::Forge, &content, &state),
            rows(&Offer::Forge, &content, &state)
        );
    }

    /// One of each category, so the mapping has something to refuse.
    fn kit() -> ContentRegistry {
        let mut content = ContentRegistry::default();
        for (id, name, category) in [
            ("ember-glaive", "Ember Glaive", ItemCategory::Weapon),
            ("scorched-mail", "Scorched Mail", ItemCategory::Armor),
            ("ember-ore", "Ember Ore", ItemCategory::Material),
            (
                "clearwater-draught",
                "Clearwater Draught",
                ItemCategory::Consumable,
            ),
            ("warden-seal", "Warden's Seal", ItemCategory::QuestItem),
        ] {
            let def = ItemDef {
                schema_version: CONTENT_SCHEMA_VERSION,
                id: item(id),
                name: name.to_owned(),
                category,
                stack_limit: 1,
                combat: None,
                on_use: None,
            };
            content.items.insert(def.id.clone(), def);
        }
        content
    }

    /// The categories that equip nothing must not appear, or the list becomes
    /// "everything you are carrying" and the slot mapping means nothing.
    #[test]
    fn only_items_that_belong_in_a_slot_are_offered() {
        let content = kit();
        let mut state = world();
        for id in [
            "ember-glaive",
            "scorched-mail",
            "ember-ore",
            "clearwater-draught",
            "warden-seal",
        ] {
            state.player.inventory.insert(item(id), 1);
        }

        assert_eq!(
            rows(&Offer::Equipment, &content, &state)
                .iter()
                .map(|row| row.label.as_str())
                .collect::<Vec<_>>(),
            vec!["Ember Glaive", "Scorched Mail"],
            "a material, a draught and a quest item equip nothing"
        );
    }

    #[test]
    fn categories_map_onto_the_slots_they_name() {
        assert_eq!(slot_for(ItemCategory::Weapon), Some(EquipmentSlot::Weapon));
        assert_eq!(slot_for(ItemCategory::Armor), Some(EquipmentSlot::Armor));
        assert_eq!(slot_for(ItemCategory::Material), None);
        assert_eq!(slot_for(ItemCategory::Consumable), None);
        assert_eq!(slot_for(ItemCategory::QuestItem), None);
    }

    /// The point of the whole wave: something finally emits `ItemEquipped`.
    #[test]
    fn equipping_emits_the_event_that_already_existed() {
        let content = kit();
        let mut state = world();
        state.player.inventory.insert(item("ember-glaive"), 1);

        let mut offer = Offer::Equipment;
        let pick = pick_at(&Offer::Equipment, &content, &state, 1).expect("the glaive is offered");
        assert_eq!(pick, Pick::Equip(item("ember-glaive")));

        let outcome = take(&mut offer, &content, &state, &pick).expect("it is carried");
        assert_eq!(
            outcome.events,
            vec![DomainEvent::ItemEquipped {
                slot: EquipmentSlot::Weapon,
                item: item("ember-glaive"),
            }],
            "the slot is derived from the category, not carried by the pick"
        );
        assert!(!is_over(&offer), "equipping one thing does not close the list");
    }

    /// The reducer fills the slot after checking only that the id is canonical,
    /// so if this module did not check, nothing would.
    #[test]
    fn an_item_no_longer_carried_cannot_be_equipped() {
        let content = kit();
        let mut state = world();
        state.player.inventory.insert(item("ember-glaive"), 1);

        let pick = pick_at(&Offer::Equipment, &content, &state, 1).expect("offered");

        // Spent, dropped, or handed over between the row and the key.
        state.player.inventory.insert(item("ember-glaive"), 0);

        let mut offer = Offer::Equipment;
        assert_eq!(
            take(&mut offer, &content, &state, &pick),
            Err(SelectionError::NotHeld(item("ember-glaive"))),
            "taking re-checks that the item is still held"
        );
        assert!(
            rows(&Offer::Equipment, &content, &state).is_empty(),
            "and a zero count is not on offer in the first place"
        );
    }

    #[test]
    fn an_unequippable_item_is_refused_even_when_named_directly() {
        let content = kit();
        let mut state = world();
        state.player.inventory.insert(item("warden-seal"), 1);

        let mut offer = Offer::Equipment;
        assert_eq!(
            take(&mut offer, &content, &state, &Pick::Equip(item("warden-seal"))),
            Err(SelectionError::NotEquippable(item("warden-seal"))),
            "the list hides it, and the API refuses it"
        );
        assert_eq!(
            take(&mut offer, &content, &state, &Pick::Equip(item("nothing-real"))),
            Err(SelectionError::UnknownItem(item("nothing-real")))
        );
    }

    #[test]
    fn an_equip_pressed_at_a_forge_is_refused() {
        let content = kit();
        let mut state = world();
        state.player.inventory.insert(item("ember-glaive"), 1);

        let mut forge = Offer::Forge;
        assert_eq!(
            take(&mut forge, &content, &state, &Pick::Equip(item("ember-glaive"))),
            Err(SelectionError::WrongOffer)
        );

        let mut equipment = Offer::Equipment;
        assert_eq!(
            take(
                &mut equipment,
                &content,
                &state,
                &Pick::Craft(item("clearwater-draught"))
            ),
            Err(SelectionError::WrongOffer)
        );
    }

    /// Selection proposes; the reducer disposes. Nothing here writes state.
    #[test]
    fn taking_a_row_returns_events_rather_than_touching_the_world() {
        let content = forge_content();
        let mut state = world();
        state.player.inventory.insert(item("still-reed"), 2);
        let before = state.clone();

        let mut offer = Offer::Forge;
        let _ = take(
            &mut offer,
            &content,
            &state,
            &Pick::Craft(item("clearwater-draught")),
        )
        .expect("affordable");

        assert_eq!(
            state.player.inventory, before.player.inventory,
            "only applying the events may change the inventory"
        );
    }
}
