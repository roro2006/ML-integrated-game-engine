//! Post-processing: the passes between the lit HDR scene and the display.
//!
//! In frame order: automatic exposure measures the scene, bloom blurs its
//! brightest light, and the tonemap pass applies exposure, adds bloom, maps
//! through the curve and grades the result onto the display.

pub(crate) mod bloom;
pub(crate) mod exposure;
pub(crate) mod tonemap;
