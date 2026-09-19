//! Engine input state built from normalized platform events.
//!
//! Two views are kept because frames and fixed ticks do not align:
//! [`FrameInput`] edges (just pressed/released) last one rendered frame, while
//! [`FixedInput`] edges last until the next fixed tick has consumed them, so a
//! quick tap is never lost when a frame runs zero ticks.

use std::collections::BTreeSet;
use std::ops::{Deref, DerefMut};

use crate::ecs::Resource;
pub use platform::input::{ButtonState, KeyCode, MouseButton, PlatformEvent};

/// Pressed state plus press/release edges for a set of buttons.
#[derive(Debug, Clone)]
pub struct ButtonInput<T: Copy + Ord> {
    pressed: BTreeSet<T>,
    just_pressed: BTreeSet<T>,
    just_released: BTreeSet<T>,
}

impl<T: Copy + Ord> Default for ButtonInput<T> {
    fn default() -> Self {
        Self {
            pressed: BTreeSet::new(),
            just_pressed: BTreeSet::new(),
            just_released: BTreeSet::new(),
        }
    }
}

impl<T: Copy + Ord> ButtonInput<T> {
    /// Marks `button` pressed. Repeats while already held do not create a new edge.
    pub fn press(&mut self, button: T) {
        if self.pressed.insert(button) {
            self.just_pressed.insert(button);
        }
    }

    pub fn release(&mut self, button: T) {
        if self.pressed.remove(&button) {
            self.just_released.insert(button);
        }
    }

    /// Releases everything, e.g. when the window loses focus.
    pub fn release_all(&mut self) {
        let held: Vec<T> = self.pressed.iter().copied().collect();
        for button in held {
            self.release(button);
        }
    }

    pub fn pressed(&self, button: T) -> bool {
        self.pressed.contains(&button)
    }

    pub fn just_pressed(&self, button: T) -> bool {
        self.just_pressed.contains(&button)
    }

    pub fn just_released(&self, button: T) -> bool {
        self.just_released.contains(&button)
    }

    pub fn any_pressed(&self) -> bool {
        !self.pressed.is_empty()
    }

    pub fn clear_edges(&mut self) {
        self.just_pressed.clear();
        self.just_released.clear();
    }
}

/// Keyboard and mouse state accumulated from platform events.
#[derive(Debug, Clone, Default)]
pub struct InputState {
    pub keys: ButtonInput<KeyCode>,
    pub mouse_buttons: ButtonInput<MouseButton>,
    /// Cursor position in physical window pixels, if inside the window.
    pub cursor_position: Option<(f64, f64)>,
    /// Accumulated raw mouse motion since edges were last cleared.
    pub mouse_motion: (f64, f64),
    /// Accumulated scroll in lines since edges were last cleared.
    pub scroll: (f32, f32),
    pub focused: bool,
}

impl InputState {
    pub fn apply(&mut self, event: &PlatformEvent) {
        match *event {
            PlatformEvent::Key { key, state, .. } => match state {
                ButtonState::Pressed => self.keys.press(key),
                ButtonState::Released => self.keys.release(key),
            },
            PlatformEvent::MouseButton { button, state } => match state {
                ButtonState::Pressed => self.mouse_buttons.press(button),
                ButtonState::Released => self.mouse_buttons.release(button),
            },
            PlatformEvent::CursorMoved { x, y } => self.cursor_position = Some((x, y)),
            PlatformEvent::CursorLeft => self.cursor_position = None,
            PlatformEvent::MouseMotion { dx, dy } => {
                self.mouse_motion.0 += dx;
                self.mouse_motion.1 += dy;
            }
            PlatformEvent::Scroll { dx, dy } => {
                self.scroll.0 += dx;
                self.scroll.1 += dy;
            }
            PlatformEvent::Focused(focused) => {
                self.focused = focused;
                if !focused {
                    self.keys.release_all();
                    self.mouse_buttons.release_all();
                }
            }
            PlatformEvent::Resized { .. }
            | PlatformEvent::ScaleFactorChanged { .. }
            | PlatformEvent::CloseRequested => {}
        }
    }

    /// Clears edges and accumulated deltas while keeping held state.
    pub fn clear_edges(&mut self) {
        self.keys.clear_edges();
        self.mouse_buttons.clear_edges();
        self.mouse_motion = (0.0, 0.0);
        self.scroll = (0.0, 0.0);
    }
}

/// Input view for frame (variable-rate) systems.
#[derive(Debug, Clone, Default)]
pub struct FrameInput(pub InputState);

impl Resource for FrameInput {}

impl Deref for FrameInput {
    type Target = InputState;
    fn deref(&self) -> &InputState {
        &self.0
    }
}

impl DerefMut for FrameInput {
    fn deref_mut(&mut self) -> &mut InputState {
        &mut self.0
    }
}

/// Input view for fixed-step systems.
#[derive(Debug, Clone, Default)]
pub struct FixedInput(pub InputState);

impl Resource for FixedInput {}

impl Deref for FixedInput {
    type Target = InputState;
    fn deref(&self) -> &InputState {
        &self.0
    }
}

impl DerefMut for FixedInput {
    fn deref_mut(&mut self) -> &mut InputState {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(key: KeyCode, state: ButtonState) -> PlatformEvent {
        PlatformEvent::Key {
            key,
            state,
            repeat: false,
        }
    }

    #[test]
    fn press_release_edges() {
        let mut input = InputState::default();
        input.apply(&key(KeyCode::W, ButtonState::Pressed));
        assert!(input.keys.pressed(KeyCode::W));
        assert!(input.keys.just_pressed(KeyCode::W));

        input.clear_edges();
        input.apply(&key(KeyCode::W, ButtonState::Pressed));
        assert!(input.keys.pressed(KeyCode::W));
        assert!(
            !input.keys.just_pressed(KeyCode::W),
            "held repeat must not re-edge"
        );

        input.apply(&key(KeyCode::W, ButtonState::Released));
        assert!(!input.keys.pressed(KeyCode::W));
        assert!(input.keys.just_released(KeyCode::W));
    }

    #[test]
    fn tap_within_one_frame_keeps_both_edges() {
        let mut input = InputState::default();
        input.apply(&key(KeyCode::Space, ButtonState::Pressed));
        input.apply(&key(KeyCode::Space, ButtonState::Released));
        assert!(input.keys.just_pressed(KeyCode::Space));
        assert!(input.keys.just_released(KeyCode::Space));
        assert!(!input.keys.pressed(KeyCode::Space));
    }

    #[test]
    fn focus_loss_releases_everything() {
        let mut input = InputState::default();
        input.apply(&key(KeyCode::A, ButtonState::Pressed));
        input.apply(&PlatformEvent::MouseButton {
            button: MouseButton::Left,
            state: ButtonState::Pressed,
        });
        input.clear_edges();
        input.apply(&PlatformEvent::Focused(false));
        assert!(!input.keys.any_pressed());
        assert!(!input.mouse_buttons.any_pressed());
        assert!(input.keys.just_released(KeyCode::A));
        assert!(input.mouse_buttons.just_released(MouseButton::Left));
    }

    #[test]
    fn deltas_accumulate_until_cleared() {
        let mut input = InputState::default();
        input.apply(&PlatformEvent::MouseMotion { dx: 1.0, dy: 2.0 });
        input.apply(&PlatformEvent::MouseMotion { dx: 3.0, dy: -1.0 });
        input.apply(&PlatformEvent::Scroll { dx: 0.0, dy: 1.0 });
        assert_eq!(input.mouse_motion, (4.0, 1.0));
        assert_eq!(input.scroll, (0.0, 1.0));
        input.clear_edges();
        assert_eq!(input.mouse_motion, (0.0, 0.0));
        assert_eq!(input.scroll, (0.0, 0.0));
    }
}
