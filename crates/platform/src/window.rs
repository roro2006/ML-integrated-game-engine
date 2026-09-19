//! winit-backed window and event loop.
//!
//! The event loop calls back into a [`PlatformHandler`]: `window_created` once,
//! `event` for each normalized event, and `frame` once per redraw. Redraws are
//! requested continuously while the window is visible, so presentation (and
//! therefore vsync) paces the loop; when the window is minimized the loop waits
//! for events instead of spinning.

use std::fmt;
use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode as WinitKeyCode, PhysicalKey};
use winit::raw_window_handle::{DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, WindowHandle};
use winit::window::{Window, WindowId};

use crate::input::{ButtonState, KeyCode, MouseButton, PlatformEvent};

/// Pixels per scroll line, used to normalize pixel-precise scrolling (touchpads).
const PIXELS_PER_LINE: f64 = 20.0;

#[derive(Debug, Clone)]
pub struct WindowConfig {
    pub title: String,
    /// Requested size in logical pixels.
    pub width: u32,
    pub height: u32,
    pub resizable: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "Engine".to_owned(),
            width: 1280,
            height: 720,
            resizable: true,
        }
    }
}

/// Whether the event loop should keep running after a callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopAction {
    Continue,
    Exit,
}

#[derive(Debug)]
pub enum PlatformError {
    EventLoop(String),
    WindowCreation(String),
}

impl fmt::Display for PlatformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EventLoop(message) => write!(f, "event loop failure: {message}"),
            Self::WindowCreation(message) => write!(f, "cannot create window: {message}"),
        }
    }
}

impl std::error::Error for PlatformError {}

/// Callbacks the platform delivers to the engine.
pub trait PlatformHandler {
    /// Called once, after the window exists and before the first frame.
    fn window_created(&mut self, window: &PlatformWindow) -> LoopAction;
    fn event(&mut self, window: &PlatformWindow, event: PlatformEvent) -> LoopAction;
    /// Called once per redraw; this is where the engine updates and renders.
    fn frame(&mut self, window: &PlatformWindow) -> LoopAction;
    fn exiting(&mut self) {}
}

/// A platform window handle. Cheap to clone; clones refer to the same window.
#[derive(Clone)]
pub struct PlatformWindow {
    window: Arc<Window>,
}

impl PlatformWindow {
    /// Drawable size in physical pixels. Zero in either axis means minimized.
    pub fn inner_size(&self) -> (u32, u32) {
        let size = self.window.inner_size();
        (size.width, size.height)
    }

    pub fn scale_factor(&self) -> f64 {
        self.window.scale_factor()
    }

    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }

    pub fn set_title(&self, title: &str) {
        self.window.set_title(title);
    }
}

impl fmt::Debug for PlatformWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (width, height) = self.inner_size();
        f.debug_struct("PlatformWindow")
            .field("width", &width)
            .field("height", &height)
            .finish()
    }
}

impl HasWindowHandle for PlatformWindow {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        self.window.window_handle()
    }
}

impl HasDisplayHandle for PlatformWindow {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        self.window.display_handle()
    }
}

/// Opens a window and runs the event loop until the handler or the user exits.
pub fn run<H: PlatformHandler>(config: WindowConfig, handler: &mut H) -> Result<(), PlatformError> {
    let event_loop = EventLoop::new().map_err(|error| PlatformError::EventLoop(error.to_string()))?;
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = WinitApp {
        config,
        handler,
        window: None,
        minimized: false,
        error: None,
    };
    event_loop
        .run_app(&mut app)
        .map_err(|error| PlatformError::EventLoop(error.to_string()))?;
    match app.error.take() {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

struct WinitApp<'handler, H: PlatformHandler> {
    config: WindowConfig,
    handler: &'handler mut H,
    window: Option<PlatformWindow>,
    minimized: bool,
    error: Option<PlatformError>,
}

impl<H: PlatformHandler> WinitApp<'_, H> {
    fn deliver(&mut self, window: &PlatformWindow, event: PlatformEvent) -> LoopAction {
        self.handler.event(window, event)
    }
}

impl<H: PlatformHandler> ApplicationHandler for WinitApp<'_, H> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title(self.config.title.clone())
            .with_inner_size(LogicalSize::new(self.config.width, self.config.height))
            .with_resizable(self.config.resizable);
        match event_loop.create_window(attributes) {
            Ok(window) => {
                let window = PlatformWindow {
                    window: Arc::new(window),
                };
                let (width, height) = window.inner_size();
                self.minimized = width == 0 || height == 0;
                if self.handler.window_created(&window) == LoopAction::Exit {
                    event_loop.exit();
                }
                window.request_redraw();
                self.window = Some(window);
            }
            Err(error) => {
                self.error = Some(PlatformError::WindowCreation(error.to_string()));
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(window) = self.window.clone() else {
            return;
        };
        let action = match event {
            WindowEvent::RedrawRequested => {
                let action = self.handler.frame(&window);
                if action == LoopAction::Continue && !self.minimized {
                    window.request_redraw();
                }
                action
            }
            WindowEvent::CloseRequested => self.deliver(&window, PlatformEvent::CloseRequested),
            WindowEvent::Resized(size) => {
                let was_minimized = self.minimized;
                self.minimized = size.width == 0 || size.height == 0;
                let action = self.deliver(
                    &window,
                    PlatformEvent::Resized {
                        width: size.width,
                        height: size.height,
                    },
                );
                if was_minimized && !self.minimized {
                    window.request_redraw();
                }
                action
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.deliver(&window, PlatformEvent::ScaleFactorChanged { scale: scale_factor })
            }
            WindowEvent::Focused(focused) => self.deliver(&window, PlatformEvent::Focused(focused)),
            WindowEvent::KeyboardInput {
                event,
                is_synthetic: false,
                ..
            } => match translate_key(event.physical_key) {
                Some(key) => self.deliver(
                    &window,
                    PlatformEvent::Key {
                        key,
                        state: translate_state(event.state),
                        repeat: event.repeat,
                    },
                ),
                None => LoopAction::Continue,
            },
            WindowEvent::MouseInput { state, button, .. } => self.deliver(
                &window,
                PlatformEvent::MouseButton {
                    button: translate_mouse_button(button),
                    state: translate_state(state),
                },
            ),
            WindowEvent::CursorMoved { position, .. } => self.deliver(
                &window,
                PlatformEvent::CursorMoved {
                    x: position.x,
                    y: position.y,
                },
            ),
            WindowEvent::CursorLeft { .. } => self.deliver(&window, PlatformEvent::CursorLeft),
            WindowEvent::MouseWheel { delta, .. } => {
                let (dx, dy) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (x, y),
                    MouseScrollDelta::PixelDelta(position) => (
                        (position.x / PIXELS_PER_LINE) as f32,
                        (position.y / PIXELS_PER_LINE) as f32,
                    ),
                };
                self.deliver(&window, PlatformEvent::Scroll { dx, dy })
            }
            _ => LoopAction::Continue,
        };
        if action == LoopAction::Exit {
            event_loop.exit();
        }
    }

    fn device_event(&mut self, event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        let DeviceEvent::MouseMotion { delta: (dx, dy) } = event else {
            return;
        };
        let Some(window) = self.window.clone() else {
            return;
        };
        if self.deliver(&window, PlatformEvent::MouseMotion { dx, dy }) == LoopAction::Exit {
            event_loop.exit();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.handler.exiting();
    }
}

fn translate_state(state: ElementState) -> ButtonState {
    match state {
        ElementState::Pressed => ButtonState::Pressed,
        ElementState::Released => ButtonState::Released,
    }
}

fn translate_mouse_button(button: winit::event::MouseButton) -> MouseButton {
    match button {
        winit::event::MouseButton::Left => MouseButton::Left,
        winit::event::MouseButton::Right => MouseButton::Right,
        winit::event::MouseButton::Middle => MouseButton::Middle,
        winit::event::MouseButton::Back => MouseButton::Back,
        winit::event::MouseButton::Forward => MouseButton::Forward,
        winit::event::MouseButton::Other(code) => MouseButton::Other(code),
    }
}

/// Maps physical key positions to engine key codes. Unmapped keys are ignored.
fn translate_key(key: PhysicalKey) -> Option<KeyCode> {
    let PhysicalKey::Code(code) = key else {
        return None;
    };
    let mapped = match code {
        WinitKeyCode::KeyA => KeyCode::A,
        WinitKeyCode::KeyB => KeyCode::B,
        WinitKeyCode::KeyC => KeyCode::C,
        WinitKeyCode::KeyD => KeyCode::D,
        WinitKeyCode::KeyE => KeyCode::E,
        WinitKeyCode::KeyF => KeyCode::F,
        WinitKeyCode::KeyG => KeyCode::G,
        WinitKeyCode::KeyH => KeyCode::H,
        WinitKeyCode::KeyI => KeyCode::I,
        WinitKeyCode::KeyJ => KeyCode::J,
        WinitKeyCode::KeyK => KeyCode::K,
        WinitKeyCode::KeyL => KeyCode::L,
        WinitKeyCode::KeyM => KeyCode::M,
        WinitKeyCode::KeyN => KeyCode::N,
        WinitKeyCode::KeyO => KeyCode::O,
        WinitKeyCode::KeyP => KeyCode::P,
        WinitKeyCode::KeyQ => KeyCode::Q,
        WinitKeyCode::KeyR => KeyCode::R,
        WinitKeyCode::KeyS => KeyCode::S,
        WinitKeyCode::KeyT => KeyCode::T,
        WinitKeyCode::KeyU => KeyCode::U,
        WinitKeyCode::KeyV => KeyCode::V,
        WinitKeyCode::KeyW => KeyCode::W,
        WinitKeyCode::KeyX => KeyCode::X,
        WinitKeyCode::KeyY => KeyCode::Y,
        WinitKeyCode::KeyZ => KeyCode::Z,
        WinitKeyCode::Digit0 => KeyCode::Digit0,
        WinitKeyCode::Digit1 => KeyCode::Digit1,
        WinitKeyCode::Digit2 => KeyCode::Digit2,
        WinitKeyCode::Digit3 => KeyCode::Digit3,
        WinitKeyCode::Digit4 => KeyCode::Digit4,
        WinitKeyCode::Digit5 => KeyCode::Digit5,
        WinitKeyCode::Digit6 => KeyCode::Digit6,
        WinitKeyCode::Digit7 => KeyCode::Digit7,
        WinitKeyCode::Digit8 => KeyCode::Digit8,
        WinitKeyCode::Digit9 => KeyCode::Digit9,
        WinitKeyCode::F1 => KeyCode::F1,
        WinitKeyCode::F2 => KeyCode::F2,
        WinitKeyCode::F3 => KeyCode::F3,
        WinitKeyCode::F4 => KeyCode::F4,
        WinitKeyCode::F5 => KeyCode::F5,
        WinitKeyCode::F6 => KeyCode::F6,
        WinitKeyCode::F7 => KeyCode::F7,
        WinitKeyCode::F8 => KeyCode::F8,
        WinitKeyCode::F9 => KeyCode::F9,
        WinitKeyCode::F10 => KeyCode::F10,
        WinitKeyCode::F11 => KeyCode::F11,
        WinitKeyCode::F12 => KeyCode::F12,
        WinitKeyCode::Escape => KeyCode::Escape,
        WinitKeyCode::Enter | WinitKeyCode::NumpadEnter => KeyCode::Enter,
        WinitKeyCode::Space => KeyCode::Space,
        WinitKeyCode::Tab => KeyCode::Tab,
        WinitKeyCode::Backspace => KeyCode::Backspace,
        WinitKeyCode::Delete => KeyCode::Delete,
        WinitKeyCode::Insert => KeyCode::Insert,
        WinitKeyCode::Home => KeyCode::Home,
        WinitKeyCode::End => KeyCode::End,
        WinitKeyCode::PageUp => KeyCode::PageUp,
        WinitKeyCode::PageDown => KeyCode::PageDown,
        WinitKeyCode::ArrowUp => KeyCode::ArrowUp,
        WinitKeyCode::ArrowDown => KeyCode::ArrowDown,
        WinitKeyCode::ArrowLeft => KeyCode::ArrowLeft,
        WinitKeyCode::ArrowRight => KeyCode::ArrowRight,
        WinitKeyCode::ShiftLeft => KeyCode::ShiftLeft,
        WinitKeyCode::ShiftRight => KeyCode::ShiftRight,
        WinitKeyCode::ControlLeft => KeyCode::ControlLeft,
        WinitKeyCode::ControlRight => KeyCode::ControlRight,
        WinitKeyCode::AltLeft => KeyCode::AltLeft,
        WinitKeyCode::AltRight => KeyCode::AltRight,
        WinitKeyCode::CapsLock => KeyCode::CapsLock,
        WinitKeyCode::Backquote => KeyCode::Backquote,
        WinitKeyCode::Minus => KeyCode::Minus,
        WinitKeyCode::Equal => KeyCode::Equal,
        WinitKeyCode::BracketLeft => KeyCode::BracketLeft,
        WinitKeyCode::BracketRight => KeyCode::BracketRight,
        WinitKeyCode::Backslash => KeyCode::Backslash,
        WinitKeyCode::Semicolon => KeyCode::Semicolon,
        WinitKeyCode::Quote => KeyCode::Quote,
        WinitKeyCode::Comma => KeyCode::Comma,
        WinitKeyCode::Period => KeyCode::Period,
        WinitKeyCode::Slash => KeyCode::Slash,
        _ => return None,
    };
    Some(mapped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_physical_keys_by_position() {
        assert_eq!(
            translate_key(PhysicalKey::Code(WinitKeyCode::KeyW)),
            Some(KeyCode::W)
        );
        assert_eq!(
            translate_key(PhysicalKey::Code(WinitKeyCode::NumpadEnter)),
            Some(KeyCode::Enter)
        );
        assert_eq!(translate_key(PhysicalKey::Code(WinitKeyCode::F24)), None);
        assert_eq!(
            translate_key(PhysicalKey::Unidentified(
                winit::keyboard::NativeKeyCode::Unidentified
            )),
            None
        );
    }

    #[test]
    fn maps_mouse_buttons_and_states() {
        assert_eq!(
            translate_mouse_button(winit::event::MouseButton::Other(9)),
            MouseButton::Other(9)
        );
        assert_eq!(translate_state(ElementState::Pressed), ButtonState::Pressed);
    }
}
