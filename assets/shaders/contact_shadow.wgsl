// contact_shadow.wgsl — reading the contact shadows in the forward pass (C4).
// Included by lit.wgsl and pbr_lit.wgsl, never compiled alone.
//
// Bindings, group 0 (crates/renderer/src/pipeline.rs):
//
//   @group(0) @binding(14)  contact shadows  texture_2d<f32>, full resolution:
//                           r = 1 lit, 0 in contact shadow
//
// Traced per pixel by contact_shadow_trace.wgsl, so a fragment reads its own
// texel. With contact shadows off the renderer clears the texture to 1.
//
// The march sees only what the depth prepass drew. A fragment the prepass
// skips, such as a blended one, reads the value traced for the opaque
// surface behind it, as it does for ambient occlusion.

@group(0) @binding(14) var contact_shadow_texture: texture_2d<f32>;

// The directional light's visibility at the fragment at `pixel` (its
// framebuffer position) by contact shadows alone: 1 lit, 0 shadowed.
fn contact_shadow(pixel: vec2<f32>) -> f32 {
    return textureLoad(contact_shadow_texture, vec2<i32>(pixel), 0).r;
}
