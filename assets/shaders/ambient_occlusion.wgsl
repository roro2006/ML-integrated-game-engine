// ambient_occlusion.wgsl — reading GTAO's result in the forward pass (C4).
// Included by lit.wgsl and pbr_lit.wgsl, never compiled alone.
//
// Bindings, group 0 (crates/renderer/src/pipeline.rs):
//
//   @group(0) @binding(13)  occlusion  texture_2d<f32>, half resolution:
//                           r = visibility, g = distance from the camera
//
// The texture is half resolution, so each fragment blends the four nearest
// texels bilinearly, weighting down any whose distance says it lies on
// another surface: occlusion does not bleed across a silhouette. With
// occlusion off the renderer clears the texture to visibility 1 at distance
// 0; no texel then matches a surface, and the plain blend returns 1.

@group(0) @binding(13) var occlusion_texture: texture_2d<f32>;

// As gtao.wgsl's denoise: a few percent of distance separates surfaces.
const OCCLUSION_DEPTH_TOLERANCE: f32 = 0.05;

// Visibility of ambient light at the fragment at `pixel` (its framebuffer
// position) and `world_position`, seen from `eye`: 1 open, 0 fully hidden.
fn screen_ambient_occlusion(pixel: vec2<f32>, world_position: vec3<f32>, eye: vec3<f32>) -> f32 {
    let limit = vec2<i32>(textureDimensions(occlusion_texture)) - 1;
    let distance = length(eye - world_position);
    // Texel i was traced at full-resolution pixel centre 2i + 0.5.
    let at = (pixel - 0.5) * 0.5;
    let base = vec2<i32>(floor(at));
    let blend = at - floor(at);
    var matched = 0.0;
    var matched_weight = 0.0;
    var plain = 0.0;
    for (var corner = 0; corner < 4; corner++) {
        let offset = vec2<i32>(corner & 1, corner >> 1u);
        let texel = textureLoad(occlusion_texture, clamp(base + offset, vec2<i32>(0), limit), 0);
        let weights = select(1.0 - blend, blend, offset == vec2<i32>(1));
        let bilinear = weights.x * weights.y;
        let difference = (texel.g - distance) / (distance * OCCLUSION_DEPTH_TOLERANCE);
        let w = bilinear * exp(-difference * difference);
        matched += texel.r * w;
        matched_weight += w;
        plain += texel.r * bilinear;
    }
    if matched_weight < 1e-4 {
        return plain;
    }
    return matched / matched_weight;
}
