//! What the player needs to know, on screen (M9 wave 1).
//!
//! Four blocks of text, drawn the way the menu panel is — a model a metre in
//! front of the eye, glyph by glyph (`menu`):
//!
//! - **The tracker**, top left: each active quest and its open objectives,
//!   with counts where an objective accumulates.
//! - **Notices**, top centre: what just happened — a quest taken, a step
//!   done, an item received — fading after a few seconds.
//! - **The prompt**, bottom centre: what E would do here.
//! - **The ability list**, bottom left: each key, its ability, and whether it
//!   is ready.
//!
//! # Reads, never writes
//!
//! Every line is recomputed from canonical state, committed events, and the
//! same rule the E key uses, the way the menu's rows are. Nothing in play
//! consults the HUD, so it cannot become a second, disagreeing source of
//! truth. Notices are the one thing with a memory, and all they remember is
//! text already derived from events the reducer accepted: a notice cannot
//! announce something that did not happen.

use std::collections::VecDeque;

use content_runtime::ContentRegistry;
use engine::camera::Camera;
use engine::ecs::{Component, Resource, World};
use engine::math::{Quat, Transform, Vec3};
use engine::pixel_text::{ADVANCE, GLYPH_HEIGHT};
use engine::render::{MaterialDesc, MaterialHandle, Model, ModelPart, RenderAssets, Visibility};
use game_core::content::StatusDef;
use game_core::content::StatusId;
use game_core::content::{Objective, QuestNode};
use game_core::events::{DomainEvent, RegionChange};
use game_core::ids::{ObjectiveId, QuestId};
use game_core::world::{ObjectiveProgress, QuestStatus, WorldState};
use game_runtime::ability::Cooldowns;
use game_runtime::status::Statuses;
use std::collections::BTreeMap;

use crate::menu::{self, MenuLook};
use crate::slice::{self, Abilities, Interaction, Slice};

/// How long a notice stays up, in fixed ticks: four seconds.
pub(crate) const NOTICE_TICKS: u64 = 240;
/// Most notices shown at once; older ones give way.
pub(crate) const MAX_NOTICES: usize = 3;
/// Most quests the tracker follows at once. The journal has the rest.
pub(crate) const MAX_TRACKED: usize = 3;
/// The keys abilities are bound to, in loadout order, as the player reads
/// them. Kept beside `aim::ABILITY_KEYS`, which is what they name.
pub(crate) const ABILITY_KEY_NAMES: [&str; 4] = ["Q", "R", "F", "C"];

/// Every control, one per line, for the controls panel.
pub(crate) const CONTROLS: &str = "WASD - move\n\
    Left click or Space - attack\n\
    Shift - dodge\n\
    Q R F C - abilities, cast on release\n\
    F2 - change how abilities aim\n\
    E - talk, examine, gather, use; E again closes a panel\n\
    1-9 or click - choose in a panel\n\
    I - inventory: wear and use\n\
    X - use a consumable\n\
    J - journal\n\
    H - this list\n\
    Esc - save and quit\n\
    F5 - reload content (with --dev)";

/// A kebab-case id read as words: `fouled-intake` is "Fouled Intake".
pub(crate) fn words(id: &str) -> String {
    id.split('-')
        .filter(|word| !word.is_empty())
        .map(|word| {
            let mut letters = word.chars();
            letters.next().map_or_else(String::new, |first| {
                first.to_uppercase().chain(letters).collect::<String>()
            })
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// A clue or choice id read as words, without the `clue-`/`choice-` prefix
/// its vocabulary names every id with.
pub(crate) fn subject(id: &str) -> String {
    words(
        id.strip_prefix("clue-")
            .or_else(|| id.strip_prefix("choice-"))
            .unwrap_or(id),
    )
}

/// What the tracker says for one objective: the node's own `text` if content
/// wrote one, else worded from the objective and the names it refers to.
pub(crate) fn objective_text(
    node: &QuestNode,
    progress: ObjectiveProgress,
    content: &ContentRegistry,
) -> String {
    if let Some(text) = &node.text {
        return text.clone();
    }
    let location = |id: &game_core::ids::LocationId| {
        content
            .locations
            .get(id)
            .map_or_else(|| words(id.name()), |def| def.name.clone())
    };
    match &node.objective {
        Objective::TalkTo { npc } => format!("Talk to {}", words(npc.name())),
        Objective::Reach { location: to } => format!("Go to {}", location(to)),
        Objective::Defeat { encounter } => format!(
            "Defeat {}",
            content
                .encounters
                .get(encounter)
                .map_or_else(|| words(encounter.name()), |def| def.name.clone())
        ),
        Objective::Collect { item, quantity } => format!(
            "Collect {} {}/{}",
            content
                .items
                .get(item)
                .map_or_else(|| words(item.name()), |def| def.name.clone()),
            progress.count.min(u32::from(*quantity)),
            quantity
        ),
        Objective::Escort { npc, destination } => {
            format!("Escort {} to {}", words(npc.name()), location(destination))
        }
        Objective::Investigate { clue } => format!("Examine {}", subject(clue.name())),
        Objective::Decide { choice } => format!("Decide: {}", subject(choice.name())),
    }
}

/// What a tracker or journal line is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Track {
    Quest,
    Objective,
}

/// Every active quest's title and open objectives, in quest id order, capped
/// at [`MAX_TRACKED`] quests.
pub(crate) fn tracker(content: &ContentRegistry, state: &WorldState) -> Vec<(String, Track)> {
    let mut lines = Vec::new();
    let active = state
        .quests
        .iter()
        .filter(|(_, quest)| quest.status == QuestStatus::Active)
        .filter_map(|(id, quest)| content.quests.get(id).map(|def| (def, quest)))
        .take(MAX_TRACKED);
    for (def, quest) in active {
        lines.push((def.title.clone(), Track::Quest));
        for objective in narrative::open_objectives(def, state) {
            if let Some(node) = def.nodes.get(&objective) {
                lines.push((
                    format!("- {}", objective_text(node, quest.objective(&objective), content)),
                    Track::Objective,
                ));
            }
        }
    }
    lines
}

/// The journal: active quests with their open steps, then finished ones with
/// how they ended.
pub(crate) fn journal(content: &ContentRegistry, state: &WorldState) -> String {
    let title = |id: &QuestId| {
        content
            .quests
            .get(id)
            .map_or_else(|| words(id.name()), |def| def.title.clone())
    };
    let mut lines = Vec::new();
    for (id, quest) in &state.quests {
        if quest.status != QuestStatus::Active {
            continue;
        }
        lines.push(title(id));
        if let Some(def) = content.quests.get(id) {
            for objective in narrative::open_objectives(def, state) {
                if let Some(node) = def.nodes.get(&objective) {
                    lines.push(format!(
                        "  - {}",
                        objective_text(node, quest.objective(&objective), content)
                    ));
                }
            }
        }
    }
    for (id, quest) in &state.quests {
        match &quest.status {
            QuestStatus::Completed { resolution } => {
                lines.push(format!("{} - done: {}", title(id), words(resolution.name())));
            }
            QuestStatus::Failed => lines.push(format!("{} - failed", title(id))),
            QuestStatus::Active | QuestStatus::NotStarted => {}
        }
    }
    if lines.is_empty() {
        "No quests yet. Talk to people.".to_owned()
    } else {
        lines.join("\n")
    }
}

/// What an accepted event tells the player, if it is worth saying.
pub(crate) fn notice_for(
    event: &DomainEvent,
    content: &ContentRegistry,
    state: &WorldState,
) -> Option<String> {
    let title = |quest: &QuestId| {
        content
            .quests
            .get(quest)
            .map_or_else(|| words(quest.name()), |def| def.title.clone())
    };
    let step = |quest: &QuestId, objective: &ObjectiveId| {
        let def = content.quests.get(quest)?;
        let node = def.nodes.get(objective)?;
        let progress = state
            .quest(quest)
            .map(|state| state.objective(objective))
            .unwrap_or_default();
        Some(objective_text(node, progress, content))
    };
    match event {
        DomainEvent::QuestAccepted { quest } => Some(format!("New quest: {}", title(quest))),
        DomainEvent::QuestObjectiveCompleted { quest, objective } => {
            step(quest, objective).map(|text| format!("Done: {text}"))
        }
        DomainEvent::QuestObjectiveAdvanced { quest, objective, .. } => step(quest, objective),
        DomainEvent::QuestCompleted { quest, resolution } => Some(format!(
            "Quest complete: {} - {}",
            title(quest),
            words(resolution.name())
        )),
        DomainEvent::QuestFailed { quest } => Some(format!("Quest failed: {}", title(quest))),
        DomainEvent::ItemGranted { item, quantity } => Some(format!(
            "+{quantity} {}",
            content
                .items
                .get(item)
                .map_or_else(|| words(item.name()), |def| def.name.clone())
        )),
        // Level 1 is where everyone starts, not news.
        DomainEvent::LeveledUp { level } if *level > 1 => Some(format!("Level {level}!")),
        DomainEvent::RegionStateChanged {
            region,
            change: RegionChange::Discovered,
        } => Some(format!("Discovered: {}", region_name(region, content))),
        // Where you are, and in which region (M9).
        DomainEvent::PlayerMoved { location } => content
            .locations
            .get(location)
            .map(|def| format!("{} - {}", def.name, region_name(&def.region, content))),
        _ => None,
    }
}

/// A region's name as content gives it.
fn region_name(region: &game_core::ids::RegionId, content: &ContentRegistry) -> String {
    content
        .regions
        .get(region)
        .map_or_else(|| words(region.name()), |def| def.name.clone())
}

/// The notices on screen, with the tick each was raised on.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Notices {
    pub shown: VecDeque<(String, u64)>,
}

impl Resource for Notices {}

impl Notices {
    /// Adds a notice raised on `tick`, dropping the oldest past the cap.
    pub(crate) fn raise(&mut self, text: String, tick: u64) {
        self.shown.push_back((text, tick));
        while self.shown.len() > MAX_NOTICES {
            self.shown.pop_front();
        }
    }

    /// The notices still up on `tick`, oldest first.
    pub(crate) fn live(&self, tick: u64) -> Vec<&str> {
        self.shown
            .iter()
            .filter(|(_, raised)| tick < raised.saturating_add(NOTICE_TICKS))
            .map(|(text, _)| text.as_str())
            .collect()
    }
}

/// Moves whatever the slice has announced since last tick onto the screen.
pub(crate) fn collect_notices(world: &mut World, tick: u64) {
    let fresh = world
        .resource_mut::<Slice>()
        .map(|slice| std::mem::take(&mut slice.notices))
        .unwrap_or_default();
    if fresh.is_empty() {
        return;
    }
    if world.resource::<Notices>().is_none() {
        world.insert_resource(Notices::default());
    }
    if let Some(notices) = world.resource_mut::<Notices>() {
        for text in fresh {
            notices.raise(text, tick);
        }
    }
}

/// What pressing E would do here, as the prompt says it.
pub(crate) fn prompt_for(interaction: &Interaction, slice: &Slice) -> Option<String> {
    let text = match interaction {
        Interaction::Talk(npc) => format!("Talk to {}", words(npc.name())),
        Interaction::Examine { clue, .. } => format!("Examine {}", subject(clue.name())),
        Interaction::Decide { choice, .. } => format!("Decide: {}", subject(choice.name())),
        Interaction::Harvest { item, .. } => format!(
            "Gather {}",
            slice
                .content
                .items
                .get(item)
                .map_or_else(|| words(item.name()), |def| def.name.clone())
        ),
        Interaction::Craft => "Use the forge".to_owned(),
        Interaction::Upgrade => "Use the upgrade board".to_owned(),
        Interaction::Use(spawn) => {
            let (to, _) = slice.link_from(spawn)?;
            let name = slice
                .content
                .locations
                .get(&to)
                .map_or_else(|| words(to.name()), |def| def.name.clone());
            format!("Enter {name}")
        }
        Interaction::Nothing => return None,
    };
    Some(format!("E - {text}"))
}

/// A status's chip colour: its swatch, from its palette, as linear-ish floats.
/// White when content cannot say, which the loader makes unreachable.
pub(crate) fn status_rgb(def: &StatusDef, content: &ContentRegistry) -> [f32; 3] {
    content
        .palettes
        .get(&def.palette)
        .and_then(|palette| palette.swatches.get(&def.color))
        .map_or([1.0, 1.0, 1.0], |&(r, g, b)| {
            [f32::from(r) / 255.0, f32::from(g) / 255.0, f32::from(b) / 255.0]
        })
}

/// What holds on the player right now: each status's name and the whole
/// seconds it has left, rounded up so the last second reads "1s", not "0s".
pub(crate) fn status_lines(world: &World, content: &ContentRegistry, tick: u64) -> Vec<(String, StatusId)> {
    let Some(statuses) = slice::player_entity(world).and_then(|player| world.get::<Statuses>(player)) else {
        return Vec::new();
    };
    statuses
        .iter()
        .filter(|held| held.expires_at > tick)
        .map(|held| {
            let name = content
                .statuses
                .get(&held.status)
                .map_or_else(|| words(held.status.0.name()), |def| def.name.clone());
            let seconds = (held.expires_at - tick).div_ceil(60);
            (format!("{name} {seconds}s"), held.status.clone())
        })
        .collect()
}

/// Each loadout slot's key and ability name, and whether it is ready.
pub(crate) fn ability_lines(world: &World, content: &ContentRegistry) -> Vec<(String, bool)> {
    let Some(player) = slice::player_entity(world) else {
        return Vec::new();
    };
    let Some(abilities) = world.get::<Abilities>(player) else {
        return Vec::new();
    };
    let cooldowns = world.get::<Cooldowns>(player);
    abilities
        .0
        .iter()
        .zip(ABILITY_KEY_NAMES)
        .map(|(id, key)| {
            let name = content
                .abilities
                .get(id)
                .map_or_else(|| words(id.name()), |def| def.name.clone());
            let ready = cooldowns.is_none_or(|cooldowns| cooldowns.remaining_for(id) == 0);
            (format!("{key}  {name}"), ready)
        })
        .collect()
}

/// The HUD's colours, built once.
#[derive(Debug, Clone)]
pub(crate) struct HudLook {
    backing: MaterialHandle,
    quest: MaterialHandle,
    text: MaterialHandle,
    notice: MaterialHandle,
    prompt: MaterialHandle,
    ready: MaterialHandle,
    waiting: MaterialHandle,
    /// Each status's chip colour (M9).
    statuses: BTreeMap<StatusId, MaterialHandle>,
}

impl Resource for HudLook {}

impl HudLook {
    pub(crate) fn build(assets: &mut dyn RenderAssets, content: Option<&ContentRegistry>) -> Self {
        let mut flat = |[r, g, b]: [f32; 3]| {
            assets.create_material(&MaterialDesc {
                base_color: [r, g, b, 1.0],
                flash: [r, g, b, 0.95],
                ..MaterialDesc::default()
            })
        };
        let statuses = content
            .map(|content| {
                content
                    .statuses
                    .values()
                    .map(|def| (def.id.clone(), flat(status_rgb(def, content))))
                    .collect()
            })
            .unwrap_or_default();
        Self {
            statuses,
            backing: flat([0.05, 0.04, 0.03]),
            quest: flat([1.0, 0.78, 0.25]),
            text: flat([0.92, 0.88, 0.80]),
            notice: flat([1.0, 1.0, 1.0]),
            prompt: flat([0.55, 0.92, 1.0]),
            ready: flat([0.45, 0.72, 1.0]),
            waiting: flat([0.45, 0.45, 0.50]),
        }
    }

    fn material(&self, colour: Colour) -> MaterialHandle {
        match colour {
            Colour::Quest => self.quest,
            Colour::Text => self.text,
            Colour::Notice => self.notice,
            Colour::Prompt => self.prompt,
            Colour::Ready => self.ready,
            Colour::Waiting => self.waiting,
            Colour::Material(material) => material,
        }
    }
}

/// What the HUD says this frame, kept so tests can read it without pixels.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Hud {
    pub tracker: Vec<String>,
    pub notices: Vec<String>,
    pub prompt: Option<String>,
    pub abilities: Vec<String>,
    /// The player's statuses, as their chips say them (M9).
    pub statuses: Vec<String>,
}

impl Component for Hud {}

/// Where a block of lines sits on screen.
#[derive(Debug, Clone, Copy)]
enum Anchor {
    TopLeft,
    TopCentre,
    BottomCentre,
    BottomLeft,
}

#[derive(Debug, Clone, Copy)]
enum Colour {
    Quest,
    Text,
    Notice,
    Prompt,
    Ready,
    Waiting,
    /// A colour of its own: a status chip's.
    Material(MaterialHandle),
}

/// A block of coloured lines, and where it goes.
type Block = (Anchor, Vec<(String, Colour)>);

/// The gap between a block and the screen's edge, as a share of its height.
const EDGE: f32 = 0.03;
/// How far the prompt sits above the bottom edge, clear of the ability list.
const PROMPT_LIFT: f32 = 0.12;
/// The border inside a block, in text pixels.
const PADDING_PIXELS: f32 = 3.0;

/// What the HUD shows this frame: the words, and the four blocks to draw.
fn contents(world: &World, tick: u64, look: &HudLook) -> (Hud, Vec<Block>) {
    let mut hud = Hud::default();
    let mut blocks = Vec::new();
    let Some(slice) = world.resource::<Slice>() else {
        return (hud, blocks);
    };

    let tracked = tracker(&slice.content, &slice.state);
    hud.tracker = tracked.iter().map(|(text, _)| text.clone()).collect();
    blocks.push((
        Anchor::TopLeft,
        tracked
            .into_iter()
            .map(|(text, kind)| {
                let colour = match kind {
                    Track::Quest => Colour::Quest,
                    Track::Objective => Colour::Text,
                };
                (text, colour)
            })
            .collect(),
    ));

    hud.notices = world
        .resource::<Notices>()
        .map(|notices| notices.live(tick).into_iter().map(str::to_owned).collect())
        .unwrap_or_default();
    blocks.push((
        Anchor::TopCentre,
        hud.notices
            .iter()
            .map(|text| (text.clone(), Colour::Notice))
            .collect(),
    ));

    // The prompt and the ability list share the bottom of the screen with the
    // menu panel, so they give way while a panel is open.
    if slice.offer.is_none() {
        hud.prompt = world
            .query2::<Transform, slice::Player>()
            .next()
            .map(|(_, transform, _)| transform.translation)
            .and_then(|from| prompt_for(&slice::interaction_at(world, from), slice));
        blocks.push((
            Anchor::BottomCentre,
            hud.prompt
                .iter()
                .map(|text| (text.clone(), Colour::Prompt))
                .collect(),
        ));

        // Statuses above the abilities, in their own colours.
        let statuses = status_lines(world, &slice.content, tick);
        hud.statuses = statuses.iter().map(|(text, _)| text.clone()).collect();
        let mut lines: Vec<(String, Colour)> = statuses
            .into_iter()
            .map(|(text, id)| {
                let colour = look
                    .statuses
                    .get(&id)
                    .map_or(Colour::Text, |material| Colour::Material(*material));
                (text, colour)
            })
            .collect();
        let mut abilities = ability_lines(world, &slice.content);
        // What X would use, and how many are left (M9).
        if let Some(item) = slice::quick_use(slice) {
            let name = slice
                .content
                .items
                .get(&item)
                .map_or_else(|| words(item.name()), |def| def.name.clone());
            let count = slice.state.player.inventory.get(&item).copied().unwrap_or(0);
            let busy =
                slice::player_entity(world).is_some_and(|player| world.get::<slice::Using>(player).is_some());
            abilities.push((format!("X  {name} x{count}"), !busy));
        }
        hud.abilities = abilities.iter().map(|(text, _)| text.clone()).collect();
        lines.extend(
            abilities
                .into_iter()
                .map(|(text, ready)| (text, if ready { Colour::Ready } else { Colour::Waiting })),
        );
        blocks.push((Anchor::BottomLeft, lines));
    }
    (hud, blocks)
}

/// Draws the HUD for this frame.
pub(crate) fn show_hud(world: &mut World, tick: u64) {
    let (Some(glyphs), Some(look)) = (
        world.resource::<MenuLook>().cloned(),
        world.resource::<HudLook>().cloned(),
    ) else {
        return;
    };
    let (hud, blocks) = contents(world, tick, &look);
    let existing = world.query::<Hud>().map(|(entity, _)| entity).next();

    let aspect = menu::screen_aspect(world);
    let camera = world.resource::<Camera>().copied().unwrap_or_default();
    // A hair behind the menu, so an open panel is always drawn over the HUD.
    let distance = menu::PANEL_DISTANCE + 0.01;
    let tall = 2.0 * distance * (camera.fov_y_radians * 0.5).tan();
    let to_plane = |x: f32, y: f32| Vec3::new((x - 0.5) * tall * aspect, (0.5 - y) * tall, 0.0);
    let pixel = menu::LINE_SCREEN / menu::LINE_PIXELS;
    let pixel_wide = pixel / aspect;

    let mut parts = Vec::new();
    for (anchor, lines) in blocks {
        if lines.is_empty() {
            continue;
        }
        let columns = lines
            .iter()
            .map(|(text, _)| text.chars().count())
            .max()
            .unwrap_or(0);
        let width = (columns as f32 * ADVANCE as f32 + 2.0 * PADDING_PIXELS) * pixel_wide;
        let height = (lines.len() as f32 * menu::LINE_PIXELS + 2.0 * PADDING_PIXELS) * pixel;
        let (left, top) = match anchor {
            Anchor::TopLeft => (EDGE / aspect, EDGE),
            Anchor::TopCentre => (0.5 - width * 0.5, EDGE),
            Anchor::BottomCentre => (0.5 - width * 0.5, 1.0 - EDGE - PROMPT_LIFT - height),
            Anchor::BottomLeft => (EDGE / aspect, 1.0 - EDGE - height),
        };

        let mut backing = ModelPart::new(glyphs.quad(), look.backing, Transform::IDENTITY);
        backing.pose = Transform {
            translation: to_plane(left, top + height),
            rotation: Quat::IDENTITY,
            scale: Vec3::new(width * tall * aspect, height * tall, 1.0),
        };
        parts.push(backing);

        for (row, (text, colour)) in lines.iter().enumerate() {
            let baseline =
                top + (PADDING_PIXELS + row as f32 * menu::LINE_PIXELS + GLYPH_HEIGHT as f32) * pixel;
            let material = look.material(*colour);
            for (column, character) in text.chars().enumerate() {
                let Some(mesh) = glyphs.glyph(character.to_ascii_uppercase()) else {
                    continue;
                };
                let x = left + (PADDING_PIXELS + (column * ADVANCE) as f32) * pixel_wide;
                let mut glyph = ModelPart::new(mesh, material, Transform::IDENTITY);
                glyph.pose = Transform {
                    translation: to_plane(x, baseline) + Vec3::Z * menu::LAYER,
                    rotation: Quat::IDENTITY,
                    scale: Vec3::new(pixel * tall, pixel * tall, 1.0),
                };
                parts.push(glyph);
            }
        }
    }

    let entity = existing.unwrap_or_else(|| world.spawn());
    let _ = world.insert(entity, hud);
    let _ = world.insert(entity, Model { parts });
    let _ = world.insert(
        entity,
        Transform {
            translation: camera.eye + camera.forward() * distance,
            rotation: crate::nameplate::facing_camera(&camera),
            scale: Vec3::ONE,
        },
    );
    let _ = world.insert(entity, Visibility::VISIBLE);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_read_as_words() {
        assert_eq!(words("fouled-intake"), "Fouled Intake");
        assert_eq!(subject("clue-fouled-intake"), "Fouled Intake");
        assert_eq!(subject("choice-water-rights"), "Water Rights");
        assert_eq!(subject("plain"), "Plain");
    }

    #[test]
    fn notices_fade_and_the_oldest_give_way() {
        let mut notices = Notices::default();
        notices.raise("one".to_owned(), 0);
        notices.raise("two".to_owned(), 10);
        assert_eq!(notices.live(5), vec!["one", "two"]);
        assert_eq!(notices.live(NOTICE_TICKS), vec!["two"], "the first has faded");
        assert!(notices.live(NOTICE_TICKS + 10).is_empty());

        for (index, text) in ["a", "b", "c", "d"].into_iter().enumerate() {
            notices.raise(text.to_owned(), 100 + index as u64);
        }
        assert_eq!(
            notices.live(104),
            vec!["b", "c", "d"],
            "at most three, newest kept"
        );
    }

    #[test]
    fn the_controls_name_every_bound_key() {
        for key in [
            "WASD", "Space", "Shift", "Q R F C", "F2", "E -", "1-9", "I -", "X -", "J -", "H -", "Esc",
        ] {
            assert!(CONTROLS.contains(key), "{key} is missing from the controls");
        }
        assert_eq!(ABILITY_KEY_NAMES.len(), crate::aim::ABILITY_KEYS.len());
    }
}
