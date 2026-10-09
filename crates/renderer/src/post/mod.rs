//! Post-processing: the passes between the lit HDR scene and the display.
//!
//! Tonemapping today; exposure adaptation, bloom and grading join it in the
//! second half of E1.

pub(crate) mod exposure;
pub(crate) mod tonemap;
