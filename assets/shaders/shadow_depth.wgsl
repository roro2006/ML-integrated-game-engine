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
// The masked entry points (`vs_masked`, `fs_masked`) draw cut-out materials,
// discarding where the base colour's alpha is below the cutoff exactly as
// pbr_lit.wgsl does, so a leaf's shadow has the leaf's holes. They add the
// material group, laid out as pbr_lit.wgsl declares it:
//
//   @group(2) @binding(0)  MaterialParams
//   @group(2) @binding(1)  base colour texture
//   @group(2) @binding(5)  sampler
//
// Vertex buffer: the shared 64-byte layout; the position, and for the masked
// entry points the uv.

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

struct MaterialParams {
    emissive: vec4<f32>,
    factors: vec4<f32>,          // metallic, roughness, alpha cutoff, occlusion strength
}

@group(0) @binding(0) var<uniform> cascade: Cascade;
@group(1) @binding(0) var<uniform> draw: DrawData;
@group(2) @binding(0) var<uniform> material: MaterialParams;
@group(2) @binding(1) var base_color_texture: texture_2d<f32>;
@group(2) @binding(5) var material_sampler: sampler;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return cascade.view_proj * (draw.model * vec4<f32>(position, 1.0));
}

struct MaskedVertex {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_masked(@location(0) position: vec3<f32>, @location(2) uv: vec2<f32>) -> MaskedVertex {
    return MaskedVertex(cascade.view_proj * (draw.model * vec4<f32>(position, 1.0)), uv);
}

@fragment
fn fs_masked(vertex: MaskedVertex) {
    let alpha = draw.base_color.w * textureSample(base_color_texture, material_sampler, vertex.uv).a;
    if alpha < material.factors.z {
        discard;
    }
}
