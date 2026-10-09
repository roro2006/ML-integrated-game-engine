//! Frame submission: the boundary between the engine's extraction step and the
//! GPU backend.
//!
//! Matrices cross this boundary as plain `[[f32; 4]; 4]` rather than math-library
//! types, which is what keeps `glam` confined to `engine::math` (ADR-0008).
//!
//! Culling and draw ordering happen in the engine, not here. A [`RenderFrame`]
//! is a finished list: the backend draws exactly what it is given, in the order
//! given. That keeps the GPU layer dumb and leaves culling testable headlessly.

use crate::Color;
use crate::debug_lines::DebugVertex;
use crate::sampler::SamplerDesc;

/// Handle to a mesh uploaded to the GPU.
///
/// Runtime-only, like an ECS entity handle: never persist one in canonical
/// state, a save file, or generated content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MeshHandle(u32);

/// Handle to a registered material.
///
/// Runtime-only; see [`MeshHandle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MaterialHandle(u32);

/// Handle to an uploaded texture.
///
/// Runtime-only; see [`MeshHandle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TextureHandle(u32);

impl TextureHandle {
    /// Builds a handle from a raw index. See [`MeshHandle::from_index`].
    pub const fn from_index(index: u32) -> Self {
        Self(index)
    }

    /// The raw index, for diagnostics and tests.
    pub const fn index(self) -> u32 {
        self.0
    }
}

impl MeshHandle {
    /// Builds a handle from a raw index.
    ///
    /// A handle is a plain index, validated at draw time rather than at
    /// construction: `Renderer::render` skips a draw whose handle resolves to
    /// nothing and counts it, so a fabricated or stale handle costs that one
    /// draw instead of the frame or the process.
    ///
    /// Public because the engine's extraction and culling logic has to be
    /// testable without a GPU, and `Renderer::upload_mesh` — the only other way
    /// to get a handle — needs a device.
    pub const fn from_index(index: u32) -> Self {
        Self(index)
    }

    pub const fn index(self) -> u32 {
        self.0
    }
}

impl MaterialHandle {
    /// Builds a handle from a raw index. See [`MeshHandle::from_index`].
    pub const fn from_index(index: u32) -> Self {
        Self(index)
    }

    pub const fn index(self) -> u32 {
        self.0
    }
}

/// Which shader lights a material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Shading {
    /// The low-poly look: Lambert diffuse and flat ambient, no textures
    /// (`lit.wgsl`). The Low tier and the existing slice use it.
    #[default]
    Flat,
    /// Physically based metallic-roughness with textures (`pbr_lit.wgsl`).
    Pbr,
}

/// How a material treats alpha.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AlphaMode {
    #[default]
    Opaque,
    /// Cut-out: fragments with alpha below the cutoff are discarded, for
    /// foliage and fences. Blending waits for transparency sorting (D4).
    Mask { cutoff: f32 },
}

/// A material's textures. Each slot is optional; an empty slot behaves as a
/// texture of the slot's neutral value, so the factors alone decide.
///
/// The slot, not the texture, decides how texels are read: base colour and
/// emissive are sampled as sRGB, normal and ORM as linear, whatever colour
/// space the texture was uploaded with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MaterialTextures {
    pub base_color: Option<TextureHandle>,
    /// Tangent-space normal map, +Y up (OpenGL convention, as glTF).
    pub normal: Option<TextureHandle>,
    /// Packed occlusion (R), roughness (G), metallic (B), as glTF.
    pub orm: Option<TextureHandle>,
    pub emissive: Option<TextureHandle>,
}

/// What a surface looks like.
///
/// Build one with `MaterialDesc { base_color: .., ..MaterialDesc::default() }`:
/// fields keep being added, and the defaults reproduce the flat look.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialDesc {
    /// Linear RGBA: scene light, which the tonemap pass maps to the display,
    /// so these are linear values, not sRGB ones.
    ///
    /// A *tint*, multiplied by each vertex's colour (ADR-0017). On a mesh with
    /// no vertex colours that is the whole colour, exactly as before.
    pub base_color: [f32; 4],
    /// A colour the lit result is mixed toward: rgb, with `a` as the amount
    /// from 0 (none) to 1 (solid).
    ///
    /// What lets a cue — a hit flash, a wind-up — read on a vertex-coloured
    /// body. A white tint multiplied into painted vertices changes nothing, so
    /// a flash expressed as a tint would vanish without a trace.
    pub flash: [f32; 4],
    /// Flat by default. Every field below is read only by [`Shading::Pbr`].
    pub shading: Shading,
    pub textures: MaterialTextures,
    pub sampler: SamplerDesc,
    /// 0 for dielectrics, 1 for metals; multiplied by the ORM texture's blue.
    pub metallic: f32,
    /// Perceptual roughness, 0 (mirror) to 1; multiplied by the ORM green.
    /// Clamped to 0.045 in the shader, below which GGX highlights alias.
    pub roughness: f32,
    /// Scales the normal map's tangent-space tilt; 0 flattens it.
    pub normal_scale: f32,
    /// How much the ORM texture's red darkens ambient light, 0 to 1.
    pub occlusion_strength: f32,
    /// Linear emitted light, multiplied by the emissive texture.
    pub emissive: [f32; 3],
    pub alpha: AlphaMode,
}

impl Default for MaterialDesc {
    fn default() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            flash: [0.0, 0.0, 0.0, 0.0],
            shading: Shading::Flat,
            textures: MaterialTextures::default(),
            sampler: SamplerDesc::default(),
            metallic: 0.0,
            roughness: 0.5,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            emissive: [0.0, 0.0, 0.0],
            alpha: AlphaMode::Opaque,
        }
    }
}

/// The single directional light the forward pass supports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalLight {
    /// Direction **to** the light, normalized. Use [`with_direction`](Self::with_direction)
    /// if the source value might not be.
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
}

impl DirectionalLight {
    /// Sets the direction, normalizing it. A zero-length direction keeps the
    /// previous one rather than producing NaNs in the shader.
    #[must_use]
    pub fn with_direction(mut self, direction: [f32; 3]) -> Self {
        let length_squared =
            direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];
        if length_squared > 1e-12 {
            let inverse = 1.0 / length_squared.sqrt();
            self.direction = [
                direction[0] * inverse,
                direction[1] * inverse,
                direction[2] * inverse,
            ];
        }
        self
    }
}

impl Default for DirectionalLight {
    /// A high afternoon key light, angled so low-poly facets read clearly.
    fn default() -> Self {
        Self {
            direction: [0.0, 1.0, 0.0],
            color: [1.0, 0.98, 0.92],
            intensity: 1.0,
        }
        .with_direction([0.35, 0.8, 0.45])
    }
}

/// Flat ambient fill, so faces pointing away from the key light stay readable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmbientLight {
    pub color: [f32; 3],
    pub intensity: f32,
}

impl Default for AmbientLight {
    fn default() -> Self {
        Self {
            color: [0.55, 0.62, 0.78],
            intensity: 0.25,
        }
    }
}

/// The curve that maps scene light onto the display.
///
/// ACES is the default because the reference captures come from a UE5 game,
/// whose filmic tonemapper is ACES-derived (owner decision, 2026-10-09). AgX
/// desaturates bright, saturated light more gracefully and is kept selectable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Tonemapper {
    /// Stephen Hill's fit of the ACES reference and output transforms.
    #[default]
    AcesFitted,
    /// AgX with Benjamin Wrensch's polynomial contrast curve.
    Agx,
}

/// How the HDR scene becomes a display image. Engine data, read by the
/// tonemap pass each frame.
///
/// Build one with `PostSettings { exposure_ev: 1.0, ..PostSettings::default() }`
/// so that bloom and grading fields can join it without breaking callers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PostSettings {
    /// Exposure in stops: +1 doubles the light reaching the tonemapper. A
    /// non-finite value means [`DEFAULT_EXPOSURE_EV`], and the value is
    /// clamped to ±16.
    pub exposure_ev: f32,
    pub tonemapper: Tonemapper,
    /// Colour grading applied to the display image. Neutral by default.
    pub grade: Grade,
}

impl Default for PostSettings {
    fn default() -> Self {
        Self {
            exposure_ev: DEFAULT_EXPOSURE_EV,
            tonemapper: Tonemapper::default(),
            grade: Grade::default(),
        }
    }
}

/// Lift, gamma and gain, per channel, applied to display values after the
/// tonemapper: `gain * (c + lift * (1 - c))`, then raised to `1 / gamma`.
///
/// Lift raises the blacks, gain scales the whites, and gamma bends the mid
/// tones (above 1 brightens them). Three controls rather than a lookup table,
/// until grading needs more than three.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grade {
    pub lift: [f32; 3],
    pub gamma: [f32; 3],
    pub gain: [f32; 3],
}

impl Default for Grade {
    /// The identity: changes nothing.
    fn default() -> Self {
        Self {
            lift: [0.0; 3],
            gamma: [1.0; 3],
            gain: [1.0; 3],
        }
    }
}

/// The exposure a frame gets unless the engine sets one.
///
/// The slice's lighting was authored for a display that showed linear values
/// clamped at 1, with no tonemapper. ACES darkens mid-tones (18 percent grey
/// lands at about 10.6 percent at 0 EV). 0.55 EV, solved from the ACES fit,
/// puts 18 percent grey back within a code value of where it was, so authored
/// scenes keep their overall brightness while highlights roll off instead of
/// clipping. A test in `post::tonemap` holds that.
pub const DEFAULT_EXPOSURE_EV: f32 = 0.55;

/// One mesh drawn once, with its transform already resolved to world space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawCommand {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub model: [[f32; 4]; 4],
    /// Inverse-transpose of the model matrix's 3x3 linear part, widened to 4x4
    /// for uniform alignment. See `docs/plans/m3-renderer.md`.
    pub normal_matrix: [[f32; 4]; 4],
}

/// Everything the backend needs to render one frame.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderFrame {
    pub view_proj: [[f32; 4]; 4],
    /// Last frame's `view_proj`, for motion vectors (E2). Equal to
    /// `view_proj` on a first frame or after a cut.
    pub prev_view_proj: [[f32; 4]; 4],
    /// The eye in world space, for view-dependent shading (specular, fog).
    pub camera_position: [f32; 3],
    /// The background, as linear scene light: it is exposed and tonemapped
    /// with everything else, so (1, 1, 1) is a bright sky, not display white.
    /// [`Renderer::render_clear`](crate::Renderer::render_clear) takes a
    /// display colour instead.
    pub clear_color: Color,
    pub light: DirectionalLight,
    pub ambient: AmbientLight,
    /// Already culled and ordered by the engine.
    pub draws: Vec<DrawCommand>,
    /// Debug line vertices in **world space**, consumed in pairs as a
    /// `LineList` after the opaque geometry.
    ///
    /// Separate from `draws` because they share nothing: a debug line has no
    /// mesh, no material, and no model matrix, and the pass that draws them
    /// tests depth without writing it. An empty list means no debug pass is
    /// recorded at all, so a frame that draws none costs nothing.
    pub debug_lines: Vec<DebugVertex>,
    /// Exposure and tonemapping for this frame.
    pub post: PostSettings,
}

const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

impl RenderFrame {
    pub fn new(view_proj: [[f32; 4]; 4], clear_color: Color) -> Self {
        Self {
            view_proj,
            prev_view_proj: view_proj,
            camera_position: [0.0, 0.0, 0.0],
            clear_color,
            light: DirectionalLight::default(),
            ambient: AmbientLight::default(),
            draws: Vec::new(),
            debug_lines: Vec::new(),
            post: PostSettings::default(),
        }
    }

    pub fn push(&mut self, draw: DrawCommand) {
        self.draws.push(draw);
    }

    /// Clears the draw list while keeping allocated capacity, so per-frame
    /// extraction does not reallocate every frame.
    ///
    /// Leaves `debug_lines` alone: debug drawing accumulates in the engine and
    /// is moved in after extraction, so the two lists have different lifetimes
    /// and clearing them together would drop one of them at the wrong moment.
    pub fn clear_draws(&mut self) {
        self.draws.clear();
    }

    /// Appends one world-space line segment.
    ///
    /// The engine's `DebugDraw` resource is the usual producer; this exists for
    /// callers already holding plain arrays, and for tests.
    pub fn push_debug_line(&mut self, start: [f32; 3], end: [f32; 3], color: [f32; 4]) {
        self.debug_lines.push(DebugVertex::new(start, color));
        self.debug_lines.push(DebugVertex::new(end, color));
    }

    /// Clears the debug line list, keeping its allocated capacity.
    pub fn clear_debug_lines(&mut self) {
        self.debug_lines.clear();
    }
}

impl Default for RenderFrame {
    fn default() -> Self {
        Self::new(IDENTITY, Color::BLACK)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_light_direction_is_normalized() {
        let light = DirectionalLight::default();
        let length = light
            .direction
            .iter()
            .map(|component| component * component)
            .sum::<f32>()
            .sqrt();
        assert!((length - 1.0).abs() < 1e-6, "direction was {length}");
    }

    #[test]
    fn with_direction_normalizes_and_ignores_zero() {
        let light = DirectionalLight::default().with_direction([0.0, 5.0, 0.0]);
        assert_eq!(light.direction, [0.0, 1.0, 0.0]);

        let unchanged = light.with_direction([0.0, 0.0, 0.0]);
        assert_eq!(unchanged.direction, [0.0, 1.0, 0.0], "zero must not produce NaN");
    }

    #[test]
    fn frame_draw_list_round_trips() {
        let mut frame = RenderFrame::default();
        assert!(frame.draws.is_empty());
        frame.push(DrawCommand {
            mesh: MeshHandle::from_index(0),
            material: MaterialHandle::from_index(0),
            model: IDENTITY,
            normal_matrix: IDENTITY,
        });
        assert_eq!(frame.draws.len(), 1);
        frame.clear_draws();
        assert!(frame.draws.is_empty());
    }

    #[test]
    fn debug_lines_are_pairs_of_vertices_and_survive_clear_draws() {
        let mut frame = RenderFrame::default();
        assert!(frame.debug_lines.is_empty(), "a new frame draws no debug lines");

        frame.push_debug_line([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(frame.debug_lines.len(), 2, "one line is two vertices");
        assert_eq!(frame.debug_lines[1].position, [1.0, 2.0, 3.0]);
        assert_eq!(
            frame.debug_lines[0].color, frame.debug_lines[1].color,
            "both ends carry the line's colour"
        );

        // The two lists are cleared on different schedules.
        frame.clear_draws();
        assert_eq!(frame.debug_lines.len(), 2);
        frame.clear_debug_lines();
        assert!(frame.debug_lines.is_empty());
    }
}
