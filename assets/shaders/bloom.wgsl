// bloom.wgsl — the dual-filter bloom chain: downsample, then upsample.
//
// `downsample` halves its source with five bilinear taps (a centre weighted
// four and four corners a texel out), which is wide and cheap. The first
// level, read from the HDR scene, runs with `PREFILTER` on: each tap keeps
// only the light above the threshold, with a soft knee so the cut does not
// show as a hard edge. `upsample` reads the next smaller level with a 3x3
// tent and is drawn with additive blending onto the level above, so the
// chain accumulates from the smallest level back up to the first. The
// tonemap pass then adds the first level to the scene.
//
// Bind group layout (both entry points):
//
//   @group(0) @binding(0)  source   texture_2d<f32>, the level read
//   @group(0) @binding(1)  sampler  linear, clamp to edge
//   @group(0) @binding(2)  Params   uniform, per frame
//
// The prefilter has a CPU mirror in crates/renderer/src/post/bloom.rs.

#include "colour.wgsl"

override PREFILTER: bool = false;

struct Params {
    threshold: f32,   // scene light where bloom starts
    knee: f32,        // width of the soft transition below the threshold
    _padding: vec2<f32>,
}

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linear_clamp: sampler;
@group(0) @binding(2) var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var output: VertexOutput;
    output.position = vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
    output.uv = vec2<f32>(corner.x, 1.0 - corner.y);
    return output;
}

// The part of a colour bright enough to bloom: none below threshold - knee,
// all of it above threshold + knee, a quadratic blend between. Judged on the
// brightest channel so saturated lights bloom in their own colour.
fn bright_part(colour: vec3<f32>) -> vec3<f32> {
    let safe = finite_scene(colour);
    let brightness = max(safe.r, max(safe.g, safe.b));
    let knee = max(params.knee, 1e-5);
    var soft = clamp(brightness - params.threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee);
    let weight = max(soft, brightness - params.threshold) / max(brightness, 1e-5);
    return safe * weight;
}

fn tap(uv: vec2<f32>) -> vec3<f32> {
    let colour = textureSample(source, linear_clamp, uv).rgb;
    if PREFILTER {
        return bright_part(colour);
    }
    return colour;
}

@fragment
fn downsample(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(source));
    let centre = tap(input.uv);
    let corners = tap(input.uv + texel * vec2<f32>(-1.0, -1.0))
        + tap(input.uv + texel * vec2<f32>(1.0, -1.0))
        + tap(input.uv + texel * vec2<f32>(-1.0, 1.0))
        + tap(input.uv + texel * vec2<f32>(1.0, 1.0));
    return vec4<f32>((centre * 4.0 + corners) / 8.0, 1.0);
}

@fragment
fn upsample(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(source));
    var sum = textureSample(source, linear_clamp, input.uv).rgb * 4.0;
    sum += textureSample(source, linear_clamp, input.uv + texel * vec2<f32>(-1.0, 0.0)).rgb * 2.0;
    sum += textureSample(source, linear_clamp, input.uv + texel * vec2<f32>(1.0, 0.0)).rgb * 2.0;
    sum += textureSample(source, linear_clamp, input.uv + texel * vec2<f32>(0.0, -1.0)).rgb * 2.0;
    sum += textureSample(source, linear_clamp, input.uv + texel * vec2<f32>(0.0, 1.0)).rgb * 2.0;
    sum += textureSample(source, linear_clamp, input.uv + texel * vec2<f32>(-1.0, -1.0)).rgb;
    sum += textureSample(source, linear_clamp, input.uv + texel * vec2<f32>(1.0, -1.0)).rgb;
    sum += textureSample(source, linear_clamp, input.uv + texel * vec2<f32>(-1.0, 1.0)).rgb;
    sum += textureSample(source, linear_clamp, input.uv + texel * vec2<f32>(1.0, 1.0)).rgb;
    return vec4<f32>(sum / 16.0, 1.0);
}
