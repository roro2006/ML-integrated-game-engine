//! Normalized, backend-independent input and window events.
//!
//! These types are the only input vocabulary the engine sees; windowing-crate
//! types are translated at the backend boundary.

/// Physical key location (layout-independent), named after the US QWERTY layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum KeyCode {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    Digit0,
    Digit1,
    Digit2,
    Digit3,
    Digit4,
    Digit5,
    Digit6,
    Digit7,
    Digit8,
    Digit9,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    Escape,
    Enter,
    Space,
    Tab,
    Backspace,
    Delete,
    Insert,
    Home,
    End,
    PageUp,
    PageDown,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    ShiftLeft,
    ShiftRight,
    ControlLeft,
    ControlRight,
    AltLeft,
    AltRight,
    CapsLock,
    Backquote,
    Minus,
    Equal,
    BracketLeft,
    BracketRight,
    Backslash,
    Semicolon,
    Quote,
    Comma,
    Period,
    Slash,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Back,
    Forward,
    Other(u16),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ButtonState {
    Pressed,
    Released,
}

/// A normalized platform event delivered to the engine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlatformEvent {
    /// New drawable size in physical pixels. May be zero while minimized.
    Resized {
        width: u32,
        height: u32,
    },
    ScaleFactorChanged {
        scale: f64,
    },
    Focused(bool),
    CloseRequested,
    Key {
        key: KeyCode,
        state: ButtonState,
        repeat: bool,
    },
    MouseButton {
        button: MouseButton,
        state: ButtonState,
    },
    /// Cursor position in physical pixels relative to the window's top-left.
    CursorMoved {
        x: f64,
        y: f64,
    },
    CursorLeft,
    /// Raw, unaccelerated mouse motion (for camera control).
    MouseMotion {
        dx: f64,
        dy: f64,
    },
    /// Scroll amount in lines (pixel deltas are converted).
    Scroll {
        dx: f32,
        dy: f32,
    },
}
