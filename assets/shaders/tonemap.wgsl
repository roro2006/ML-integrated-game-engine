// tonemap.wgsl — maps the HDR scene onto the display.
//
// A fullscreen triangle reads the Rgba16Float scene target texel for texel
// (the two are the same size, so no sampler), scales it by the exposure,
// applies the chosen tonemapper, and writes the swapchain image.
//
// Bind group layout (the pipeline must match this exactly):
//
//   @group(0) @binding(0)  hdr   texture_2d<f32>, the lit scene, linear
//   @group(0) @binding(1)  Post      uniform, one per frame
//   @group(0) @binding(2)  Exposure  uniform, the adapted luminance left by the
//                                    auto-exposure passes (unused when off)
//   @group(0) @binding(3)  bloom     texture_2d<f32>, the bloom chain's first
//                                    level, or the scene itself when bloom is
//                                    off (multiplied by an intensity of 0)
//   @group(0) @binding(4)  sampler   linear, clamp to edge
//
// COLOUR SPACE: this pass is where scene-referred linear light becomes a
// display value. On an sRGB swapchain the GPU applies the sRGB transfer
// function when it writes, so this shader returns LINEAR display values, as
// the lit pass used to. When the only swapchain format available is not sRGB,
// the pipeline sets `encode_srgb` and the encode happens here instead. Either
// way it happens exactly once.

#include "colour.wgsl"

struct Post {
    exposure: f32,         // linear scale, 2^EV
    curve: u32,            // 0 = ACES fitted, 1 = AgX; see `Tonemapper` in frame.rs
    bloom_intensity: f32,  // 0 = no bloom
    auto_exposure: u32,    // 1 = scale by the adapted exposure as well
    lift: vec4<f32>,       // rgb used; see `Grade` in frame.rs
    inverse_gamma: vec4<f32>,
    gain: vec4<f32>,
}

struct Exposure {
    luminance: f32,      // adapted average scene luminance
    initialised: u32,    // 0 until a frame has been measured
    _padding: vec2<u32>,
}

// Auto-exposure maps the adapted average onto this mid grey.
const EXPOSURE_KEY: f32 = 0.18;

@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var<uniform> post: Post;
@group(0) @binding(2) var<uniform> exposure: Exposure;
@group(0) @binding(3) var bloom_texture: texture_2d<f32>;
@group(0) @binding(4) var bloom_sampler: sampler;

override encode_srgb: bool = false;

// One triangle covering the screen: (-1, -1), (3, -1), (-1, 3).
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    // Bloom is scene light, so it joins before exposure and the curve.
    // Bloom is sampled only when it is on; the switch is uniform, so the
    // sample stays in uniform control flow.
    var bloom = vec3<f32>(0.0);
    if post.bloom_intensity > 0.0 {
        let uv = position.xy / vec2<f32>(textureDimensions(hdr));
        bloom = textureSample(bloom_texture, bloom_sampler, uv).rgb * post.bloom_intensity;
    }
    let stored = finite_scene(textureLoad(hdr, vec2<i32>(position.xy), 0).rgb + bloom);
    var scale = post.exposure;
    if post.auto_exposure == 1u && exposure.initialised != 0u {
        scale *= EXPOSURE_KEY / max(exposure.luminance, 1e-6);
    }
    let scene = stored * scale;
    var display: vec3<f32>;
    if post.curve == 1u {
        display = agx(scene);
    } else {
        display = aces_fitted(scene);
    }
    // Graded in the encoded space; decoded again when the swapchain encodes.
    let graded = grade(linear_to_srgb(display), post.lift.rgb, post.inverse_gamma.rgb, post.gain.rgb);
    if encode_srgb {
        display = graded;
    } else {
        display = srgb_to_linear(graded);
    }
    return vec4<f32>(display, 1.0);
}
