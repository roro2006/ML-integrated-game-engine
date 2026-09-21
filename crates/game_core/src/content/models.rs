//! Low-poly models as recipes (ADR-0017).
//!
//! Three families work together. A **body plan** names the joints a body has —
//! a shoulder, a hip, a neck — and where each pivots. A **palette** names the
//! colours a model may be painted with. A **model** is a list of parts, each a
//! shape from a small fixed set, hung from one of its body plan's slots and
//! painted from its palette.
//!
//! Everything here is a choice among approved things plus numbers inside
//! bounds: a shape kind, its dimensions, an offset, a turn, and swatch names.
//! Nothing is vertex data. That is what makes a model something an AI can
//! propose and validation can bound (ADR-0010), and it is why the mesh is built
//! from the recipe when content loads rather than stored.
//!
//! # Coordinates
//!
//! Pivots are measured from the **ground under the body's centre**: `y = 0` is
//! where the feet stand, whatever the collision body's size. A part's `at` is
//! measured from its slot's pivot, and its `turn` is in degrees about X, then
//! Y, then Z. All lengths are fixed-point metres; `-Z` is forward.
//!
//! A part moves with its slot. Everything a slot carries is merged into one
//! mesh and rotated about the slot's pivot, which is how an arm swings from the
//! shoulder and a sword held in that arm swings with it.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::ids::{BodyPlanId, ModelId, PaletteId, SlotId, SwatchId};

use super::{AnimationSet, Fixed, Point};

/// Most parts one model may have.
pub const MAX_PARTS: usize = 24;
/// Most triangles one model may have, summed over its parts.
pub const MAX_TRIANGLES: usize = 1_500;
/// Most slots one body plan may have. Each slot is a draw, so this is also a
/// ceiling on what one character costs the renderer.
pub const MAX_SLOTS: usize = 12;
/// Most swatches one palette may have.
pub const MAX_SWATCHES: usize = 32;
/// Fewest and most sides a round shape may have.
pub const MIN_SIDES: u8 = 3;
pub const MAX_SIDES: u8 = 24;
/// Most subdivisions a sphere may have: 0 is 20 faces, 1 is 80.
pub const MAX_SPHERE_DETAIL: u8 = 1;
/// Most rings of facets a dome may have between crown and rim.
pub const MAX_DOME_RINGS: u8 = 6;
/// A part's per-axis stretch lies in `1/10 ..= 3`.
pub const MIN_STRETCH_MILLI: i64 = 100;
pub const MAX_STRETCH_MILLI: i64 = 3_000;
/// Longest any one shape dimension may be, in metres.
pub const MAX_EXTENT_METRES: i64 = 8;
/// Furthest a pivot may sit from the body's origin, or a part from its pivot.
pub const MAX_OFFSET_METRES: i64 = 4;
/// Largest rotation a part may be turned by about any axis, in degrees.
pub const MAX_TURN_DEGREES: i64 = 360;
/// A model's uniform size multiplier lies in `1/4 ..= 4`.
pub const MIN_SCALE_MILLI: i64 = 250;
pub const MAX_SCALE_MILLI: i64 = 4_000;

/// A joint that parts hang from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlotDef {
    pub name: SlotId,
    /// Where the joint sits, measured from the ground under the body's centre.
    pub pivot: Point,
}

/// The joints a kind of body has.
///
/// Shared between models, so one animation clip serves every model built on
/// the same plan: a clip addresses slots by name, never parts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodyPlanDef {
    pub schema_version: u32,
    pub id: BodyPlanId,
    pub name: String,
    pub slots: Vec<SlotDef>,
    /// The clips every model on this plan plays by default (ADR-0017 §5).
    #[serde(default)]
    pub animations: AnimationSet,
}

/// A set of named colours.
///
/// Colours are sRGB, 0–255 per channel, the way an artist picks them; the
/// runtime converts to linear once. Integers rather than floats so a palette
/// hashes and compares exactly, like every other content value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PaletteDef {
    pub schema_version: u32,
    pub id: PaletteId,
    pub name: String,
    pub swatches: BTreeMap<SwatchId, (u8, u8, u8)>,
}

/// One shape from the approved set. Every variant is convex and low-poly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum ShapeDef {
    /// Full extents along each axis.
    Box { x: Fixed, y: Fixed, z: Fixed },
    /// A ramp, full height at the back (+Z) falling to nothing at the front.
    Wedge { x: Fixed, y: Fixed, z: Fixed },
    /// An upright n-sided column.
    Cylinder { radius: Fixed, height: Fixed, sides: u8 },
    /// An upright n-sided cone, point up.
    Cone { radius: Fixed, height: Fixed, sides: u8 },
    /// An upright tapered column; a zero `top` is a cone.
    Frustum {
        bottom: Fixed,
        top: Fixed,
        height: Fixed,
        sides: u8,
    },
    /// A faceted ball.
    Sphere { radius: Fixed, detail: u8 },
    /// A box whose top differs from its bottom: `x` and `z` are the bottom's
    /// full extents, `top_x` and `top_z` the top's. A torso wide at the chest
    /// and narrow at the waist; a limb thinning to the wrist. A zero top on
    /// one axis is a ridge, on both a spike.
    Taper {
        x: Fixed,
        z: Fixed,
        top_x: Fixed,
        top_z: Fixed,
        height: Fixed,
    },
    /// A faceted half-sphere, flat side down: pauldrons, helmets, knee plates,
    /// the crown of a head. One ring is a cone.
    Dome { radius: Fixed, segments: u8, rings: u8 },
}

impl ShapeDef {
    /// Triangles this shape will be built with.
    ///
    /// Computed from the parameters rather than by building the mesh, so a
    /// budget can be checked where no renderer exists. A test in the game
    /// crate holds this equal to what the engine actually builds.
    pub fn triangles(&self) -> usize {
        let sides = |sides: u8| usize::from(sides.clamp(MIN_SIDES, MAX_SIDES));
        // Two n-gon caps fan into n-2 triangles each; n quads up the side.
        let column = |n: usize| 2 * (n - 2) + 2 * n;
        // One cap, and n triangles up to the point.
        let point = |n: usize| (n - 2) + n;
        match *self {
            Self::Box { .. } => 12,
            Self::Wedge { .. } => 8,
            Self::Cylinder { sides: n, .. } => column(sides(n)),
            Self::Cone { sides: n, .. } => point(sides(n)),
            Self::Frustum { top, sides: n, .. } => {
                if top.as_milli() <= 0 {
                    point(sides(n))
                } else {
                    column(sides(n))
                }
            }
            Self::Sphere { detail, .. } => 20 * 4_usize.pow(u32::from(detail.min(MAX_SPHERE_DETAIL))),
            Self::Taper { top_x, top_z, .. } => {
                // The bottom is two triangles; a side whose top edge has
                // shrunk to nothing is one triangle rather than two; and the
                // top exists only while it has area.
                let edge = |width: Fixed| if width.as_milli() > 0 { 2 } else { 1 };
                let top = if top_x.as_milli() > 0 && top_z.as_milli() > 0 {
                    2
                } else {
                    0
                };
                2 + top + 2 * edge(top_x) + 2 * edge(top_z)
            }
            Self::Dome { segments, rings, .. } => {
                let n = sides(segments);
                let rings = usize::from(rings.clamp(1, MAX_DOME_RINGS));
                n + 2 * n * (rings - 1) + (n - 2)
            }
        }
    }
}

fn unit_scale() -> Fixed {
    Fixed::from_whole(1)
}

fn unit_stretch() -> Point {
    let one = Fixed::from_whole(1);
    Point::new(one, one, one)
}

/// One piece of a model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartDef {
    /// The joint this part hangs from; it moves when that joint does.
    pub slot: SlotId,
    pub shape: ShapeDef,
    /// Where the shape's centre sits, measured from the slot's pivot.
    #[serde(default)]
    pub at: Point,
    /// Rotation in degrees about X, then Y, then Z.
    #[serde(default)]
    pub turn: Point,
    /// Per-axis scale of the shape before it is turned, in milli-units:
    /// `(x: 1000, y: 700, z: 1000)` flattens a dome into a shoulder plate,
    /// and a stretched sphere is an oval head. Absent means unstretched.
    #[serde(default = "unit_stretch")]
    pub stretch: Point,
    /// The colour of the sides, and of every face unless overridden below.
    pub color: SwatchId,
    /// Faces within about 45 degrees of straight up, if not `color`.
    #[serde(default)]
    pub top: Option<SwatchId>,
    /// Faces within about 45 degrees of straight down, if not `color`.
    #[serde(default)]
    pub bottom: Option<SwatchId>,
}

/// A model: parts on a body plan, painted from a palette.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDef {
    pub schema_version: u32,
    pub id: ModelId,
    pub name: String,
    pub body_plan: BodyPlanId,
    pub palette: PaletteId,
    /// Uniform size, pivots and parts alike: `1000` is as authored. How an
    /// elite is the same body plan as the husks around it, only larger.
    #[serde(default = "unit_scale")]
    pub scale: Fixed,
    pub parts: Vec<PartDef>,
    /// Clips this model plays instead of its body plan's, state by state. A
    /// husk that shambles where a wayfinder walks overrides `walk` and keeps
    /// the rest.
    #[serde(default)]
    pub animations: AnimationSet,
}

impl ModelDef {
    /// Triangles the whole model will be built with.
    pub fn triangles(&self) -> usize {
        self.parts.iter().map(|part| part.shape.triangles()).sum()
    }
}
