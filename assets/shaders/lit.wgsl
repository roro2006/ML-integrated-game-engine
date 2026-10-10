// lit.wgsl — the basic lit material for low-poly geometry.
//
// One directional light plus a flat ambient term, no textures. Deliberately
// cheap: the target is Vulkan-capable integrated graphics, so there is no
// per-pixel normalization of anything that can be normalized on the CPU, no
// specular term, and no branching.
//
// Bind group layout (the pipeline must match this exactly):
//
//   @group(0) @binding(0)  Globals   uniform, one per frame
//   @group(1) @binding(0)  DrawData  uniform, one per draw, dynamic offset
//
// Vertex buffer: interleaved, 16 floats (64 bytes) per vertex, matching
// `MeshData::interleaved` in crates/renderer/src/mesh_data.rs:
//
//   @location(0) position  Float32x3   offset 0
//   @location(1) normal    Float32x3   offset 12
//   @location(2) uv        Float32x2   offset 24
//   @location(3) colour    Float32x3   offset 32   (linear RGB; white if unset)
//   @location(4) tangent   Float32x4   offset 44   (not read here; for normal maps)
//   @location(5) wind      Float32     offset 60   (not read here; for foliage)
//
// COLOUR SPACE: this shader writes the Rgba16Float HDR scene target, not the
// swapchain. Everything below — light colours, ambient, base colour, and the
// value returned by `fs_main` — is LINEAR scene light, and may exceed 1. The
// tonemap pass (tonemap.wgsl) applies exposure and the tonemapping curve, and
// the sRGB encode happens once, there or in the swapchain format. Do NOT clamp,
// tonemap or gamma-correct here; any of those would be applied twice.

// Per-frame constants. `light_direction.xyz` points TO the light (i.e. it is
// the negated direction the light travels), which is what the Lambert dot
// product wants without a sign flip.
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

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
}

// `uv` is interpolated but unused until textured materials land. It stays in
// the varyings so the vertex layout, this struct, and the textured variant
// that replaces it do not have to change in lockstep; one extra vec2 of
// interpolation is noise next to the draw-call cost at this poly count.
struct VertexOutput {
    // Invariant: must match depth_prepass.wgsl bit for bit (ADR-0024).
    @builtin(position) @invariant clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    // Flat within a face, because a low-poly face's vertices all carry the
    // same colour; interpolated anyway, so nothing breaks if they do not.
    @location(2) color: vec3<f32>,
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let world_position = draw.model * vec4<f32>(vertex.position, 1.0);
    output.clip_position = globals.view_proj * world_position;
    // w = 0: a normal is a direction, so the model translation must not apply.
    output.world_normal = (draw.normal_matrix * vec4<f32>(vertex.normal, 0.0)).xyz;
    output.uv = vertex.uv;
    output.color = vertex.color;
    return output;
}

@fragment
fn fs_main(fragment: VertexOutput) -> @location(0) vec4<f32> {
    // Renormalize: interpolating unit normals across a triangle shortens them,
    // and the normal matrix rescales them under non-uniform scale.
    let normal = normalize(fragment.world_normal);
    let to_light = normalize(globals.light_direction.xyz);

    let lambert = max(dot(normal, to_light), 0.0);
    let direct = globals.light_color.rgb * globals.light_color.w * lambert;
    let ambient = globals.ambient_color.rgb * globals.ambient_color.w;

    // The tint multiplies the painted vertex colour; white vertices make this
    // exactly the old flat colour.
    let albedo = draw.base_color.rgb * fragment.color;
    let lit = albedo * (direct + ambient);

    // The flash overlay is mixed in after lighting, so a hit flash reads the
    // same on the lit and the shadowed side of a body. Amount 0 is a no-op.
    let flashed = mix(lit, draw.flash.rgb, clamp(draw.flash.w, 0.0, 1.0));

    // Linear scene light out — see the COLOUR SPACE note at the top.
    return vec4<f32>(flashed, draw.base_color.w);
}
