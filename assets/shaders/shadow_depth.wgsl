// shadow_depth.wgsl — one cascade of the directional light's shadow map (C1).
//
// Position-only and with no fragment stage. The shadow pass draws each
// cascade into its own viewport of the atlas, selecting that cascade's
// light view-projection with a dynamic offset into group 0.
//
// Bind group layout:
//
//   @group(0) @binding(0)  Cascade   uniform, one per cascade, dynamic offset
//   @group(1) @binding(0)  DrawData  uniform, one per draw, dynamic offset
//
// Vertex buffer: the shared 64-byte layout; only the position is read.

struct Cascade {
    view_proj: mat4x4<f32>, // world to light clip space, depth 0..1
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

@group(0) @binding(0) var<uniform> cascade: Cascade;
@group(1) @binding(0) var<uniform> draw: DrawData;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return cascade.view_proj * (draw.model * vec4<f32>(position, 1.0));
}
