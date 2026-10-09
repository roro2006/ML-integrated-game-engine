// debug_line.wgsl — world-space debug lines: bounds, axes, nav links, rays.
//
// Bind group layout (the pipeline must match this exactly):
//
//   @group(0) @binding(0)  Globals  uniform, one per frame
//
// There is no per-draw bind group. Debug vertices arrive already in world
// space, so there is no model matrix to apply and no material to look up — the
// colour rides on the vertex. That is what lets the whole debug pass be one
// draw call over one vertex buffer, however many separate shapes it contains.
//
// Vertex buffer: interleaved, 7 floats (28 bytes) per vertex, matching
// `DebugVertex` in crates/renderer/src/debug_lines.rs:
//
//   @location(0) position  Float32x3   offset 0    world space
//   @location(1) color     Float32x4   offset 12   linear RGBA
//
// COLOUR SPACE: lines draw after tonemapping, straight onto the swapchain, so
// their colour is a LINEAR display value: not exposed, not tonemapped. On an
// sRGB swapchain the GPU encodes on write; on a non-sRGB one the pipeline sets
// `encode_srgb` and the encode happens here, exactly as in tonemap.wgsl, so
// lines and scene always agree.

#include "colour.wgsl"

override encode_srgb: bool = false;

// Declared in full even though only `view_proj` is read. The bind group layout
// this pipeline shares with the lit pass declares a `min_binding_size` of 112
// bytes (the whole `Globals` block), and a shader-side struct smaller than that
// is a pipeline validation error rather than a silent truncation. Sharing the
// layout is deliberate: one per-frame uniform buffer, written once, serves both
// passes.
struct Globals {
    view_proj: mat4x4<f32>,
    light_direction: vec4<f32>,  // unused here; lines are unlit by design
    light_color: vec4<f32>,      // unused here
    ambient_color: vec4<f32>,    // unused here
    camera_position: vec4<f32>,  // unused here
    inv_view_proj: mat4x4<f32>,  // unused here
    prev_view_proj: mat4x4<f32>, // unused here
}

@group(0) @binding(0) var<uniform> globals: Globals;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    // No model matrix: the position is already in world space.
    output.clip_position = globals.view_proj * vec4<f32>(vertex.position, 1.0);
    output.color = vertex.color;
    return output;
}

@fragment
fn fs_main(fragment: VertexOutput) -> @location(0) vec4<f32> {
    // Unlit: a debug line is an annotation, not a surface. Shading it would
    // make its colour depend on the scene's lighting, which is exactly the
    // thing the annotation is usually being used to diagnose.
    //
    // See the COLOUR SPACE note at the top of this file.
    if encode_srgb {
        return vec4<f32>(linear_to_srgb(fragment.color.rgb), fragment.color.a);
    }
    return fragment.color;
}
