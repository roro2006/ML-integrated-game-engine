// depth_prepass.wgsl — depth only, before the forward pass (ADR-0024).
//
// Writes the nearest depth of every opaque draw so the forward pass shades
// each visible pixel once: it tests `LessEqual` against this depth. The clip
// position must be bit-identical to lit.wgsl and pbr_lit.wgsl, so all three
// compute it with the same expression and mark it `@invariant`. Cut-out
// materials are not drawn here; they write their own depth in the forward
// pass, since depth here cannot see their holes.
//
// Bind group layout, the lit pipeline layout:
//
//   @group(0) @binding(0)  Globals   uniform, one per frame
//   @group(1) @binding(0)  DrawData  uniform, one per draw, dynamic offset
//
// Vertex buffer: the shared 64-byte layout; only the position is read.

struct Globals {
    view_proj: mat4x4<f32>,
    light_direction: vec4<f32>,  // xyz = normalized direction to the light, w unused
    light_color: vec4<f32>,      // rgb = linear colour, w = intensity
    ambient_color: vec4<f32>,    // rgb = linear colour, w = intensity
    camera_position: vec4<f32>,  // xyz = world-space eye, w unused
    inv_view_proj: mat4x4<f32>,  // clip to world, for depth reconstruction
    prev_view_proj: mat4x4<f32>, // last frame's view_proj, for motion vectors
}

// Per-draw constants. `normal_matrix` is the inverse-transpose of `model`,
// supplied as a mat4x4 rather than a mat3x3 because a mat3x3 uniform member
// pads each column to 16 bytes anyway — same size, simpler upload code.
//
// Size is 160 bytes; the pipeline pads the per-draw stride up to the device's
// `min_uniform_buffer_offset_alignment` (256 on most drivers) for the dynamic
// offset. That padding is the pipeline's concern, not the shader's.
struct DrawData {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    base_color: vec4<f32>,       // rgb = linear tint over vertex colour, w = alpha
    flash: vec4<f32>,            // rgb = overlay colour, w = amount 0..1 (ADR-0017)
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> draw: DrawData;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) @invariant vec4<f32> {
    let world_position = draw.model * vec4<f32>(position, 1.0);
    return globals.view_proj * world_position;
}
