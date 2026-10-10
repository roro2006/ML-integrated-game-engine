// depth_prepass.wgsl — depth only, before the forward pass (ADR-0024).
//
// Writes the nearest depth of every opaque draw so the forward pass shades
// each visible pixel once: it tests `LessEqual` against this depth. The clip
// position must be bit-identical to lit.wgsl and pbr_lit.wgsl, so all three
// compute it with the same expression and mark it `@invariant`.
//
// `vs_main` alone, with no fragment stage, is the depth-only prepass. Cut-out
// materials are not drawn by it; they write their own depth in the forward
// pass, since depth alone cannot see their holes.
//
// With ambient occlusion on (C4), the prepass also writes the thin G-buffer
// of ADR-0024 decision 4: the world-space geometric normal, encoded as
// `n * 0.5 + 0.5` into an Rgb10a2 target. `fs_normals` draws opaque
// materials; `fs_masked_normals` draws cut-outs with their alpha test, so a
// leaf's normal exists and its holes show what is behind them.
//
// Bind group layout, the lit pipeline layout, plus for the masked entry
// point the material group as pbr_lit.wgsl declares it:
//
//   @group(0) @binding(0)  Globals   uniform, one per frame
//   @group(1) @binding(0)  DrawData  uniform, one per draw, dynamic offset
//   @group(2) @binding(0)  MaterialParams
//   @group(2) @binding(1)  base colour texture
//   @group(2) @binding(5)  sampler
//
// Vertex buffer: the shared 64-byte layout; the position, and for the normal
// entry points the normal and uv.

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

struct MaterialParams {
    emissive: vec4<f32>,
    factors: vec4<f32>,          // metallic, roughness, alpha cutoff, occlusion strength
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> draw: DrawData;
@group(2) @binding(0) var<uniform> material: MaterialParams;
@group(2) @binding(1) var base_color_texture: texture_2d<f32>;
@group(2) @binding(5) var material_sampler: sampler;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) @invariant vec4<f32> {
    let world_position = draw.model * vec4<f32>(position, 1.0);
    return globals.view_proj * world_position;
}

struct NormalVertex {
    @builtin(position) @invariant clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_normals(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> NormalVertex {
    let world_position = draw.model * vec4<f32>(position, 1.0);
    var output: NormalVertex;
    output.clip_position = globals.view_proj * world_position;
    output.world_normal = (draw.normal_matrix * vec4<f32>(normal, 0.0)).xyz;
    output.uv = uv;
    return output;
}

// The geometric normal on the side the viewer sees: cut-outs draw back faces.
fn encode_normal(world_normal: vec3<f32>, front_facing: bool) -> vec4<f32> {
    let normal = normalize(world_normal) * select(-1.0, 1.0, front_facing);
    return vec4<f32>(normal * 0.5 + 0.5, 1.0);
}

@fragment
fn fs_normals(vertex: NormalVertex, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    return encode_normal(vertex.world_normal, front_facing);
}

@fragment
fn fs_masked_normals(vertex: NormalVertex, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    // The same test as pbr_lit.wgsl, so the two passes agree on every hole.
    let alpha = draw.base_color.w * textureSample(base_color_texture, material_sampler, vertex.uv).a;
    if alpha < material.factors.z {
        discard;
    }
    return encode_normal(vertex.world_normal, front_facing);
}
