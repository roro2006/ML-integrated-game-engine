// mip_downsample.wgsl — draws one mip level from the level above it.
//
// A fullscreen triangle over the target level. Each target pixel centre maps
// to the corner shared by four source texels, so one linear-filtered sample
// is the average of those four: an exact 2x2 box filter for even sizes.
//
// Bind group layout (the pipeline must match this exactly):
//
//   @group(0) @binding(0)  source   texture_2d<f32>, one mip level
//   @group(0) @binding(1)  sampler  linear, clamp to edge
//
// COLOUR SPACE: for an sRGB texture, both the source view and the target are
// sRGB, so the sample is decoded to linear, averaged in linear light, and
// encoded again on write. Averaging encoded values would darken every mip.

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linear_clamp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// One triangle covering the target: (-1, -1), (3, -1), (-1, 3), with texture
// coordinates running top to bottom.
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var output: VertexOutput;
    output.position = vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
    output.uv = vec2<f32>(corner.x, 1.0 - corner.y);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(source, linear_clamp, input.uv);
}
