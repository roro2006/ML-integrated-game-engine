//! The menu on screen: what someone is saying, and what you can say back.
//!
//! Whatever is on offer — a conversation, the forge, the upgrade board, the
//! equipment list — is drawn as a panel along the bottom of the screen: a
//! title, the prompt word-wrapped, and the numbered options, with a hint
//! line under them. The option under the cursor is highlighted; clicking it
//! or pressing its digit chooses it. What the panel shows is exactly what
//! `selection` offers — the rows are recomputed, never stored — so what a
//! player reads and what a pick means cannot drift apart.
//!
//! # A panel in the world, drawn like a screen
//!
//! There is no 2D pass. The panel is a model a metre in front of the camera,
//! square to its view, sized from the camera's field of view so that it
//! covers a fixed share of the screen whatever the window. On a plane square
//! to the view, screen position maps linearly onto the plane, so one layout —
//! in shares of the screen — serves both drawing and the click test, and a
//! click lands on exactly the row it looks like it lands on.
//!
//! Text is one pre-built mesh per glyph (`engine::pixel_text::glyph_mesh`),
//! placed per character, because what a menu says is decided at run time and
//! meshes are only uploaded at startup. That is a draw per character while a
//! menu is open, and none when it is not.

use std::collections::BTreeMap;

use engine::camera::{Camera, Viewport};
use engine::ecs::{Component, Resource, World};
use engine::math::{Quat, Transform, Vec3};
use engine::pixel_text::{self, ADVANCE, GLYPH_HEIGHT};
use engine::render::{
    MaterialDesc, MaterialHandle, MeshData, MeshHandle, Model, ModelPart, RenderAssets, Visibility,
};
use game_runtime::selection::{self, Offer};

use crate::slice::Slice;

/// One line of text's height, as a share of the screen's.
pub(crate) const LINE_SCREEN: f32 = 0.034;
/// One line's height in text pixels: a glyph and the gap under it.
pub(crate) const LINE_PIXELS: f32 = 10.0;
/// The border inside the panel, in text pixels.
const PADDING_PIXELS: f32 = 6.0;
/// Most characters on one line before it wraps.
pub(crate) const WRAP_COLUMNS: usize = 56;
/// The gap under the panel, as a share of the screen's height.
const MARGIN_SCREEN: f32 = 0.04;
/// How far in front of the eye the panel stands, in metres: well inside the
/// camera's distance from anything in the scene.
pub(crate) const PANEL_DISTANCE: f32 = 1.0;
/// The aspect assumed with no window to measure, as in a headless run.
pub(crate) const DEFAULT_ASPECT: f32 = 16.0 / 9.0;
/// Depth between the panel's layers, toward the eye, in metres.
pub(crate) const LAYER: f32 = 0.001;

/// What a line of the panel is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Style {
    Title,
    Prompt,
    /// Part of the option at this index, counting from 0.
    Option(usize),
    Hint,
    Blank,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Line {
    pub text: String,
    pub style: Style,
}

/// The panel's text, laid out in lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Layout {
    pub lines: Vec<Line>,
    /// The longest line, in characters.
    pub columns: usize,
}

fn blank() -> Line {
    Line {
        text: String::new(),
        style: Style::Blank,
    }
}

/// Lays out a title, a prompt and numbered options.
pub(crate) fn layout(title: &str, prompt: &str, options: &[String]) -> Layout {
    let mut lines = vec![
        Line {
            text: title.to_owned(),
            style: Style::Title,
        },
        blank(),
    ];
    // A line break in the prompt is kept — the journal and the controls list
    // are lists — and each line wraps on its own.
    lines.extend(
        prompt
            .split('\n')
            .flat_map(|paragraph| pixel_text::wrap(paragraph, WRAP_COLUMNS))
            .map(|text| Line {
                text,
                style: Style::Prompt,
            }),
    );
    lines.push(blank());
    for (index, option) in options.iter().enumerate() {
        // Continuation lines are indented under the text, past the number.
        for (part, text) in pixel_text::wrap(option, WRAP_COLUMNS - 3).into_iter().enumerate() {
            lines.push(Line {
                text: if part == 0 { text } else { format!("   {text}") },
                style: Style::Option(index),
            });
        }
    }
    lines.push(blank());
    lines.push(Line {
        // E is what closes a panel (`slice-interact`). This used to say "walk
        // away", which nothing implements.
        text: if options.is_empty() {
            "Press E to close.".to_owned()
        } else {
            "Press a number or click to choose. E to close.".to_owned()
        },
        style: Style::Hint,
    });
    let columns = lines
        .iter()
        .map(|line| line.text.chars().count())
        .max()
        .unwrap_or(0);
    Layout { lines, columns }
}

/// Where the panel is on screen, in shares of the screen — `x` from the left,
/// `y` from the top, as the platform reports the cursor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Panel {
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
    /// One text pixel, as a share of the screen's height.
    pub pixel: f32,
    pub aspect: f32,
}

impl Panel {
    /// One text pixel, as a share of the screen's width.
    fn pixel_wide(&self) -> f32 {
        self.pixel / self.aspect
    }

    /// The top and bottom of line `index`, as shares of the screen's height.
    pub(crate) fn line_span(&self, index: usize) -> (f32, f32) {
        let top = self.top + (PADDING_PIXELS + index as f32 * LINE_PIXELS) * self.pixel;
        (top, top + LINE_PIXELS * self.pixel)
    }

    /// Whether a point, in shares of the screen, is on the panel.
    pub(crate) fn contains(&self, (x, y): (f32, f32)) -> bool {
        x >= self.left && x <= self.left + self.width && y >= self.top && y <= self.top + self.height
    }
}

/// Places `layout` on a screen of `aspect`: centred, along the bottom.
pub(crate) fn panel(layout: &Layout, aspect: f32) -> Panel {
    let aspect = if aspect.is_finite() && aspect > 0.0 {
        aspect
    } else {
        DEFAULT_ASPECT
    };
    let pixel = LINE_SCREEN / LINE_PIXELS;
    let width = (layout.columns as f32 * ADVANCE as f32 + 2.0 * PADDING_PIXELS) * pixel / aspect;
    let height = (layout.lines.len() as f32 * LINE_PIXELS + 2.0 * PADDING_PIXELS) * pixel;
    Panel {
        left: 0.5 - width * 0.5,
        top: 1.0 - MARGIN_SCREEN - height,
        width,
        height,
        pixel,
        aspect,
    }
}

/// The option under a cursor at `(x, y)`, in shares of the screen.
pub(crate) fn option_at(layout: &Layout, panel: &Panel, (x, y): (f32, f32)) -> Option<usize> {
    if x < panel.left || x > panel.left + panel.width {
        return None;
    }
    layout.lines.iter().enumerate().find_map(|(index, line)| {
        let (top, bottom) = panel.line_span(index);
        match line.style {
            Style::Option(option) if y >= top && y < bottom => Some(option),
            _ => None,
        }
    })
}

/// What the open offer says: a title, a prompt, and its numbered options.
pub(crate) fn contents(slice: &Slice) -> Option<(String, String, Vec<String>)> {
    let offer = slice.offer.as_ref()?;
    let rows = selection::rows(offer, &slice.content, &slice.state);
    let options = selection::lines(&rows);
    let (title, prompt) = match offer {
        Offer::Topics(npc) => (crate::nameplate::person_name(npc), "What about?".to_owned()),
        Offer::Talking(conversation) => {
            let speaker = slice
                .content
                .dialogues
                .get(conversation.dialogue())
                .map_or_else(String::new, |def| crate::nameplate::person_name(&def.speaker));
            (
                speaker,
                selection::prompt(offer, &slice.content).unwrap_or_default(),
            )
        }
        Offer::Note { title, text } => (title.clone(), text.clone()),
        Offer::Decision { choice, prompt, .. } => (
            format!("Decide: {}", crate::hud::subject(choice.name())),
            prompt.clone(),
        ),
        Offer::Journal => (
            "Journal".to_owned(),
            crate::hud::journal(&slice.content, &slice.state),
        ),
        Offer::Controls => ("Controls".to_owned(), crate::hud::CONTROLS.to_owned()),
        Offer::ClassChoice { .. } => (
            "Who are you?".to_owned(),
            format!(
                "Choose a class. E keeps the {}.",
                slice
                    .state
                    .player
                    .progression
                    .class
                    .as_ref()
                    .and_then(|class| slice.content.classes.get(class))
                    .map_or("default", |class| class.name.as_str())
            ),
        ),
        Offer::Forge => ("Forge".to_owned(), "What will you make?".to_owned()),
        Offer::Board => ("Upgrade Board".to_owned(), "What will you build?".to_owned()),
        Offer::Equipment => ("Inventory".to_owned(), "What will you wear or use?".to_owned()),
    };
    Some((title, prompt, options))
}

/// A cursor in pixels, as shares of the screen.
pub(crate) fn cursor_share(viewport: Viewport, cursor: (f64, f64)) -> Option<(f32, f32)> {
    (viewport.width > 0 && viewport.height > 0).then(|| {
        (
            cursor.0 as f32 / viewport.width as f32,
            cursor.1 as f32 / viewport.height as f32,
        )
    })
}

/// The glyphs and panel pieces every menu is drawn with, built once.
#[derive(Debug, Clone)]
pub(crate) struct MenuLook {
    glyphs: BTreeMap<char, MeshHandle>,
    quad: MeshHandle,
    panel: MaterialHandle,
    highlight: MaterialHandle,
    title: MaterialHandle,
    prompt: MaterialHandle,
    option: MaterialHandle,
    hovered: MaterialHandle,
    hint: MaterialHandle,
}

impl Resource for MenuLook {}

/// A one-unit square with its corner at the origin, facing +Z.
fn unit_quad() -> MeshData {
    MeshData {
        positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        normals: vec![[0.0, 0.0, 1.0]; 4],
        uvs: vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        colors: vec![[1.0, 1.0, 1.0]; 4],
        indices: vec![0, 1, 2, 0, 2, 3],
        ..MeshData::default()
    }
}

impl MenuLook {
    pub(crate) fn build(assets: &mut dyn RenderAssets) -> Option<Self> {
        let mut glyphs = BTreeMap::new();
        for character in pixel_text::glyphs() {
            let mesh = pixel_text::glyph_mesh(character, [1.0, 1.0, 1.0])?;
            let handle = assets.upload_mesh(&mesh).ok()?;
            glyphs.insert(character, handle);
        }
        let quad = assets.upload_mesh(&unit_quad()).ok()?;
        // Flat colours: a menu should read the same whatever the light.
        let mut flat = |[r, g, b]: [f32; 3]| {
            assets.create_material(&MaterialDesc {
                base_color: [r, g, b, 1.0],
                flash: [r, g, b, 0.95],
                ..MaterialDesc::default()
            })
        };
        // After the old chatbox: a dark panel, the speaker in gold, pale
        // text, and options in blue that turn white under the cursor.
        Some(Self {
            glyphs,
            quad,
            panel: flat([0.07, 0.055, 0.04]),
            highlight: flat([0.22, 0.17, 0.10]),
            title: flat([1.0, 0.78, 0.25]),
            prompt: flat([0.92, 0.88, 0.80]),
            option: flat([0.45, 0.72, 1.0]),
            hovered: flat([1.0, 1.0, 1.0]),
            hint: flat([0.55, 0.52, 0.48]),
        })
    }

    /// The mesh for one character, if the font has it.
    pub(crate) fn glyph(&self, character: char) -> Option<MeshHandle> {
        self.glyphs.get(&character).copied()
    }

    /// A one-unit square with its corner at the origin, facing the eye.
    pub(crate) fn quad(&self) -> MeshHandle {
        self.quad
    }

    fn material(&self, style: Style, hovered: Option<usize>) -> MaterialHandle {
        match style {
            Style::Title => self.title,
            Style::Prompt | Style::Blank => self.prompt,
            Style::Option(index) if hovered == Some(index) => self.hovered,
            Style::Option(_) => self.option,
            Style::Hint => self.hint,
        }
    }
}

/// The on-screen menu, and what it currently says — kept so tests and
/// diagnostics can read the menu without reading pixels.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Menu {
    pub lines: Vec<String>,
    pub hovered: Option<usize>,
}

impl Component for Menu {}

/// The screen's aspect, or the headless default.
pub(crate) fn screen_aspect(world: &World) -> f32 {
    world
        .resource::<Viewport>()
        .and_then(|viewport| viewport.aspect())
        .unwrap_or(DEFAULT_ASPECT)
}

/// The option under the cursor this frame, if a menu is open.
fn hovered(world: &World, layout: &Layout, panel: &Panel) -> Option<usize> {
    let cursor = world.resource::<engine::input::FrameInput>()?.cursor_position?;
    let share = cursor_share(*world.resource::<Viewport>()?, cursor)?;
    option_at(layout, panel, share)
}

/// Takes the option a left-click landed on this tick, if a menu is open.
/// Returns whether the click was spent on the menu: anywhere on an open
/// panel is the menu's, whether or not it hits an option, so a click there
/// is never a swing.
pub(crate) fn click(world: &mut World) -> bool {
    let share = {
        let (Some(input), Some(viewport)) = (
            world.resource::<engine::input::FixedInput>(),
            world.resource::<Viewport>(),
        ) else {
            return false;
        };
        if !input.mouse_buttons.just_pressed(engine::input::MouseButton::Left) {
            return false;
        }
        let Some(share) = input
            .cursor_position
            .and_then(|cursor| cursor_share(*viewport, cursor))
        else {
            return false;
        };
        share
    };
    let aspect = screen_aspect(world);
    let Some(slice) = world.resource_mut::<Slice>() else {
        return false;
    };
    let Some((title, prompt, options)) = contents(slice) else {
        return false;
    };
    let layout = layout(&title, &prompt, &options);
    let panel = panel(&layout, aspect);
    if let Some(option) = option_at(&layout, &panel, share) {
        slice.take_offer(option as u32 + 1);
    }
    panel.contains(share)
}

/// Draws the open menu, or hides it.
pub(crate) fn show_menu(world: &mut World) {
    let Some(look) = world.resource::<MenuLook>().cloned() else {
        return;
    };
    let contents = world.resource::<Slice>().and_then(contents);
    let existing = world.query::<Menu>().map(|(entity, _)| entity).next();

    let Some((title, prompt, options)) = contents else {
        if let Some(menu) = existing {
            let _ = world.insert(menu, Visibility::HIDDEN);
        }
        return;
    };
    let aspect = screen_aspect(world);
    let layout = layout(&title, &prompt, &options);
    let panel = panel(&layout, aspect);
    let hovered = hovered(world, &layout, &panel);
    let camera = world.resource::<Camera>().copied().unwrap_or_default();

    // The screen, a metre in front of the eye: how tall it is there, and
    // where a point on it, in shares of the screen, falls on that plane.
    let tall = 2.0 * PANEL_DISTANCE * (camera.fov_y_radians * 0.5).tan();
    let to_plane = |x: f32, y: f32| Vec3::new((x - 0.5) * tall * aspect, (0.5 - y) * tall, 0.0);
    let pixel = panel.pixel * tall;
    let wide = |share: f32| share * tall * aspect;

    let mut parts = Vec::new();
    // The panel itself, from its bottom-left corner.
    let mut background = ModelPart::new(look.quad, look.panel, Transform::IDENTITY);
    background.pose = Transform {
        translation: to_plane(panel.left, panel.top + panel.height),
        rotation: Quat::IDENTITY,
        scale: Vec3::new(wide(panel.width), panel.height * tall, 1.0),
    };
    parts.push(background);

    for (index, line) in layout.lines.iter().enumerate() {
        let (top, bottom) = panel.line_span(index);
        if matches!(line.style, Style::Option(option) if hovered == Some(option)) {
            let mut bar = ModelPart::new(look.quad, look.highlight, Transform::IDENTITY);
            bar.pose = Transform {
                translation: to_plane(panel.left, bottom) + Vec3::Z * LAYER,
                rotation: Quat::IDENTITY,
                scale: Vec3::new(wide(panel.width), (bottom - top) * tall, 1.0),
            };
            parts.push(bar);
        }
        let material = look.material(line.style, hovered);
        let baseline = top + GLYPH_HEIGHT as f32 * panel.pixel;
        for (column, character) in line.text.chars().enumerate() {
            let Some(&mesh) = look.glyphs.get(&character.to_ascii_uppercase()) else {
                continue;
            };
            let x = panel.left + (PADDING_PIXELS + (column * ADVANCE) as f32) * panel.pixel_wide();
            let mut glyph = ModelPart::new(mesh, material, Transform::IDENTITY);
            glyph.pose = Transform {
                translation: to_plane(x, baseline) + Vec3::Z * (2.0 * LAYER),
                rotation: Quat::IDENTITY,
                scale: Vec3::new(pixel, pixel, 1.0),
            };
            parts.push(glyph);
        }
    }

    let menu = existing.unwrap_or_else(|| world.spawn());
    let _ = world.insert(
        menu,
        Menu {
            lines: layout.lines.iter().map(|line| line.text.clone()).collect(),
            hovered,
        },
    );
    let _ = world.insert(menu, Model { parts });
    let _ = world.insert(
        menu,
        Transform {
            translation: camera.eye + camera.forward() * PANEL_DISTANCE,
            rotation: crate::nameplate::facing_camera(&camera),
            scale: Vec3::ONE,
        },
    );
    let _ = world.insert(menu, Visibility::VISIBLE);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sela() -> Layout {
        layout(
            "Warden Sela",
            "The intake's fouled again. Third time since the collapse, and nobody will say why.",
            &["1) I'll look into it.".to_owned(), "2) Not now.".to_owned()],
        )
    }

    #[test]
    fn the_layout_reads_title_prompt_options_hint() {
        let menu = sela();
        let styles: Vec<Style> = menu.lines.iter().map(|line| line.style).collect();
        assert_eq!(styles.first(), Some(&Style::Title));
        assert_eq!(styles.last(), Some(&Style::Hint));
        assert!(styles.contains(&Style::Option(0)) && styles.contains(&Style::Option(1)));
        let prompt: Vec<&str> = menu
            .lines
            .iter()
            .filter(|line| line.style == Style::Prompt)
            .map(|line| line.text.as_str())
            .collect();
        assert!(prompt.len() >= 2, "a long line wraps: {prompt:?}");
        assert!(prompt.iter().all(|line| line.chars().count() <= WRAP_COLUMNS));
        assert_eq!(
            prompt.join(" "),
            "The intake's fouled again. Third time since the collapse, and nobody will say why."
        );
    }

    #[test]
    fn the_panel_sits_centred_along_the_bottom() {
        let menu = sela();
        let placed = panel(&menu, DEFAULT_ASPECT);
        assert!((placed.left + placed.width * 0.5 - 0.5).abs() < 1e-5, "centred");
        assert!(
            (placed.top + placed.height - (1.0 - MARGIN_SCREEN)).abs() < 1e-5,
            "along the bottom"
        );
        assert!(placed.width < 1.0 && placed.height < 0.6, "on screen: {placed:?}");
    }

    #[test]
    fn a_cursor_over_an_option_line_picks_that_option() {
        let menu = sela();
        let placed = panel(&menu, DEFAULT_ASPECT);
        let x = placed.left + placed.width * 0.5;
        for (index, line) in menu.lines.iter().enumerate() {
            let (top, bottom) = placed.line_span(index);
            let hit = option_at(&menu, &placed, (x, (top + bottom) * 0.5));
            match line.style {
                Style::Option(option) => assert_eq!(hit, Some(option), "line {index}: {}", line.text),
                _ => assert_eq!(hit, None, "line {index} is not an option: {}", line.text),
            }
        }
        let second = menu
            .lines
            .iter()
            .position(|line| line.style == Style::Option(1))
            .expect("a second option");
        let (top, bottom) = placed.line_span(second);
        assert_eq!(
            option_at(&menu, &placed, (placed.left - 0.01, (top + bottom) * 0.5)),
            None,
            "beside the panel"
        );
    }

    #[test]
    fn an_empty_list_says_how_to_leave() {
        let menu = layout("Warden Sela", "Go well.", &[]);
        assert_eq!(
            menu.lines.last().map(|line| line.text.as_str()),
            Some("Press E to close.")
        );
        assert!(
            !menu
                .lines
                .iter()
                .any(|line| matches!(line.style, Style::Option(_)))
        );
    }
}
