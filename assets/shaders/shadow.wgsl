// shadow.wgsl — sampling the directional light's cascaded shadows (C1).
// Included by lit.wgsl and pbr_lit.wgsl, never compiled alone.
//
// Bindings, group 0 next to Globals (crates/renderer/src/pipeline.rs):
//
//   @group(0) @binding(1)  ShadowData  uniform, one per frame
//   @group(0) @binding(2)  the atlas   texture_depth_2d, cascades side by side
//   @group(0) @binding(3)  a comparison sampler (LessEqual, linear: 2x2 PCF per tap)
//
// A fragment uses the first cascade whose square contains it. Selecting by
// projection rather than by view depth needs no camera data and never picks
// a cascade the fragment falls outside of. Near the edge of a cascade the
// result blends into the next one, so the change in resolution has no seam.

struct ShadowData {
    view_proj: array<mat4x4<f32>, 5>,  // world to light clip, per cascade
    far: array<vec4<f32>, 2>,          // cascade i at [i / 4][i % 4]; view depth where it ends
    texel_world: array<vec4<f32>, 2>,  // world size of one texel, per cascade
    atlas: vec4<f32>,                  // cascade count, resolution, 1 / width, 1 / height
}

@group(0) @binding(1) var<uniform> shadow_data: ShadowData;
@group(0) @binding(2) var shadow_atlas: texture_depth_2d;
@group(0) @binding(3) var shadow_sampler: sampler_comparison;

// PCF taps on a Vogel disk, and its radius in texels.
const SHADOW_TAPS: u32 = 12u;
const SHADOW_FILTER_TEXELS: f32 = 1.5;
// The fragment moves this many texels along its normal before the lookup,
// which removes acne without the peter-panning a large depth bias causes.
const SHADOW_NORMAL_OFFSET_TEXELS: f32 = 1.5;
// The outer part of a cascade, as a fraction of its half-width, over which
// it blends into the next cascade.
const SHADOW_BLEND_BAND: f32 = 0.1;
const GOLDEN_ANGLE: f32 = 2.39996323;

fn shadow_texel_world(index: u32) -> f32 {
    return shadow_data.texel_world[index / 4u][index % 4u];
}

// Light clip position of `world` in cascade `index`, pushed off the surface.
fn shadow_clip(index: u32, world: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let offset = normal * shadow_texel_world(index) * SHADOW_NORMAL_OFFSET_TEXELS;
    let clip = shadow_data.view_proj[index] * vec4<f32>(world + offset, 1.0);
    return clip.xyz / clip.w;
}

fn shadow_inside(clip: vec3<f32>) -> bool {
    return all(abs(clip.xy) <= vec2<f32>(1.0)) && clip.z >= 0.0 && clip.z <= 1.0;
}

// Fraction of light reaching `clip` in cascade `index`: 1 lit, 0 shadowed.
fn shadow_filtered(index: u32, clip: vec3<f32>) -> f32 {
    let resolution = shadow_data.atlas.y;
    let texel = shadow_data.atlas.zw;
    // Light clip y is up, texture v is down.
    let local = vec2<f32>(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5) * resolution;
    // Taps stay inside this cascade's square, half a texel from its edge, so
    // filtering never reads the neighbouring cascade.
    let low = vec2<f32>(f32(index) * resolution + 0.5, 0.5);
    let high = vec2<f32>(f32(index + 1u) * resolution - 0.5, resolution - 0.5);
    let centre = vec2<f32>(f32(index) * resolution, 0.0) + local;
    var lit = 0.0;
    for (var tap = 0u; tap < SHADOW_TAPS; tap++) {
        let radius = sqrt((f32(tap) + 0.5) / f32(SHADOW_TAPS)) * SHADOW_FILTER_TEXELS;
        let angle = f32(tap) * GOLDEN_ANGLE;
        let position = clamp(centre + radius * vec2<f32>(cos(angle), sin(angle)), low, high);
        lit += textureSampleCompareLevel(shadow_atlas, shadow_sampler, position * texel, clip.z);
    }
    return lit / f32(SHADOW_TAPS);
}

// The directional light's visibility at a surface point. `normal` is the
// geometric normal, unit length, facing the viewer's side of the surface.
fn directional_shadow(world: vec3<f32>, normal: vec3<f32>) -> f32 {
    let count = u32(shadow_data.atlas.x);
    for (var index = 0u; index < count; index++) {
        let clip = shadow_clip(index, world, normal);
        if !shadow_inside(clip) {
            continue;
        }
        let visibility = shadow_filtered(index, clip);
        let edge = 1.0 - max(abs(clip.x), abs(clip.y));
        if edge >= SHADOW_BLEND_BAND || index + 1u >= count {
            return visibility;
        }
        let next = shadow_clip(index + 1u, world, normal);
        if !shadow_inside(next) {
            return visibility;
        }
        return mix(shadow_filtered(index + 1u, next), visibility, edge / SHADOW_BLEND_BAND);
    }
    // Beyond the last cascade: unshadowed.
    return 1.0;
}
