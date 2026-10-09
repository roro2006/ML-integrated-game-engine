// tonemap.wgsl — maps the HDR scene onto the display.
//
// A fullscreen triangle reads the Rgba16Float scene target texel for texel
// (the two are the same size, so no sampler), scales it by the exposure,
// applies the chosen tonemapper, and writes the swapchain image.
//
// Bind group layout (the pipeline must match this exactly):
//
//   @group(0) @binding(0)  hdr   texture_2d<f32>, the lit scene, linear
//   @group(0) @binding(1)  Post  uniform, one per frame
//
// COLOUR SPACE: this pass is where scene-referred linear light becomes a
// display value. On an sRGB swapchain the GPU applies the sRGB transfer
// function when it writes, so this shader returns LINEAR display values, as
// the lit pass used to. When the only swapchain format available is not sRGB,
// the pipeline sets `encode_srgb` and the encode happens here instead. Either
// way it happens exactly once.

#include "colour.wgsl"

struct Post {
    exposure: f32,   // linear scale, 2^EV
    curve: u32,      // 0 = ACES fitted, 1 = AgX; see `Tonemapper` in frame.rs
    _padding: vec2<f32>,
}

@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var<uniform> post: Post;

override encode_srgb: bool = false;

// One triangle covering the screen: (-1, -1), (3, -1), (-1, 3).
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let scene = textureLoad(hdr, vec2<i32>(position.xy), 0).rgb * post.exposure;
    var display: vec3<f32>;
    if post.curve == 1u {
        display = agx(scene);
    } else {
        display = aces_fitted(scene);
    }
    if encode_srgb {
        display = linear_to_srgb(display);
    }
    return vec4<f32>(display, 1.0);
}
