// pbr_lit.wgsl — physically based metallic-roughness material, forward.
//
// One directional light plus flat ambient, as lit.wgsl, but with GGX
// specular, textures and normal maps. Written forward-first; if the B2
// benchmark picks deferred, the BRDF in brdf.wgsl moves to the lighting pass
// unchanged.
//
// Bind group layout (the pipeline must match this exactly):
//
//   @group(0) @binding(0)  Globals         uniform, one per frame
//   @group(1) @binding(0)  DrawData        uniform, one per draw, dynamic offset
//   @group(2) @binding(0)  MaterialParams  uniform, one per material
//   @group(2) @binding(1)  base colour     texture_2d<f32>, sRGB view
//   @group(2) @binding(2)  normal          texture_2d<f32>, linear view
//   @group(2) @binding(3)  ORM             texture_2d<f32>, linear view
//   @group(2) @binding(4)  emissive        texture_2d<f32>, sRGB view
//   @group(2) @binding(5)  sampler
//
// Vertex buffer: the 64-byte layout documented in lit.wgsl; this shader also
// reads the tangent (location 4).
//
// Permutations are `override` constants, set per pipeline from
// `PermutationKey` (crates/renderer/src/permutation.rs).
//
// COLOUR SPACE: as lit.wgsl, the output is linear scene light into the HDR
// target. Texture views decode sRGB slots on sampling.

#include "brdf.wgsl"

override HAS_NORMAL_MAP: bool = false;
override HAS_EMISSIVE: bool = false;
override ALPHA_MASK: bool = false;

struct Globals {
    view_proj: mat4x4<f32>,
    light_direction: vec4<f32>,  // xyz = normalized direction to the light, w unused
    light_color: vec4<f32>,      // rgb = linear colour, w = intensity
    ambient_color: vec4<f32>,    // rgb = linear colour, w = intensity
    camera_position: vec4<f32>,  // xyz = world-space eye, w unused
    inv_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
}

struct DrawData {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    base_color: vec4<f32>,       // rgb = linear tint over vertex colour, w = alpha
    flash: vec4<f32>,            // rgb = overlay colour, w = amount 0..1 (ADR-0017)
}

struct MaterialParams {
    emissive: vec4<f32>,         // rgb = linear emitted light, w = normal scale
    factors: vec4<f32>,          // metallic, roughness, alpha cutoff, occlusion strength
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> draw: DrawData;
@group(2) @binding(0) var<uniform> material: MaterialParams;
@group(2) @binding(1) var base_color_texture: texture_2d<f32>;
@group(2) @binding(2) var normal_texture: texture_2d<f32>;
@group(2) @binding(3) var orm_texture: texture_2d<f32>;
@group(2) @binding(4) var emissive_texture: texture_2d<f32>;
@group(2) @binding(5) var material_sampler: sampler;

// The directional light's cascaded shadows (C1).
#include "shadow.wgsl"

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
    @location(4) tangent: vec4<f32>,
}

struct VertexOutput {
    // Invariant: must match depth_prepass.wgsl bit for bit (ADR-0024).
    @builtin(position) @invariant clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
    // xyz = world tangent, w = handedness.
    @location(4) world_tangent: vec4<f32>,
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let world_position = draw.model * vec4<f32>(vertex.position, 1.0);
    output.clip_position = globals.view_proj * world_position;
    output.world_position = world_position.xyz;
    output.world_normal = (draw.normal_matrix * vec4<f32>(vertex.normal, 0.0)).xyz;
    // A tangent lies in the surface, so it transforms with the model matrix
    // itself rather than the normal matrix.
    output.world_tangent = vec4<f32>((draw.model * vec4<f32>(vertex.tangent.xyz, 0.0)).xyz, vertex.tangent.w);
    output.uv = vertex.uv;
    output.color = vertex.color;
    return output;
}

// The shading normal. `facing` is +1 for a front face and -1 for a back face,
// which only cut-out materials draw, so a leaf seen from behind is lit on the
// side the viewer sees.
fn surface_normal(fragment: VertexOutput, encoded_tilt: vec3<f32>, facing: f32) -> vec3<f32> {
    let geometric = normalize(fragment.world_normal) * facing;
    if !HAS_NORMAL_MAP {
        return geometric;
    }
    // Gram-Schmidt again after interpolation, then the bitangent from the
    // handedness the mesh carries: up the image, where +Y-up maps point.
    let tangent = normalize(fragment.world_tangent.xyz - geometric * dot(geometric, fragment.world_tangent.xyz));
    let bitangent = cross(geometric, tangent) * fragment.world_tangent.w;
    var tilt = encoded_tilt * 2.0 - 1.0;
    tilt = vec3<f32>(tilt.xy * material.emissive.w, tilt.z);
    // Renormalised: mip averaging shortens normals.
    return normalize(tangent * tilt.x + bitangent * tilt.y + geometric * tilt.z);
}

// The largest value written to the half-float target, kept below its 65504
// limit: a tight GGX highlight can exceed it, and the infinity it would store
// spreads through every later filter, bloom first.
const MAX_SCENE_LIGHT: f32 = 60000.0;

@fragment
fn fs_main(fragment: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    // Every texture is sampled before the alpha test, so derivatives, and with
    // them mip selection, stay defined across the whole quad even on targets
    // where `discard` ends the invocation rather than demoting it.
    let base = textureSample(base_color_texture, material_sampler, fragment.uv);
    let orm = textureSample(orm_texture, material_sampler, fragment.uv).rgb;
    var encoded_tilt = vec3<f32>(0.5, 0.5, 1.0);
    if HAS_NORMAL_MAP {
        encoded_tilt = textureSample(normal_texture, material_sampler, fragment.uv).xyz;
    }
    var emitted = vec3<f32>(0.0);
    if HAS_EMISSIVE {
        emitted = material.emissive.rgb * textureSample(emissive_texture, material_sampler, fragment.uv).rgb;
    }

    let albedo = draw.base_color.rgb * fragment.color * base.rgb;
    let alpha = draw.base_color.w * base.a;
    if ALPHA_MASK && alpha < material.factors.z {
        discard;
    }

    let occlusion = mix(1.0, orm.r, material.factors.w);
    let roughness = clamp(material.factors.y * orm.g, 0.045, 1.0);
    let metallic = clamp(material.factors.x * orm.b, 0.0, 1.0);

    let normal = surface_normal(fragment, encoded_tilt, select(-1.0, 1.0, front_facing));
    let to_view = normalize(globals.camera_position.xyz - fragment.world_position);
    let to_light = normalize(globals.light_direction.xyz);
    let irradiance = globals.light_color.rgb * globals.light_color.w * PI;
    // The geometric normal, facing the viewer's side, places the shadow
    // lookup; a normal map tilts shading, not where the surface is.
    let geometric = normalize(fragment.world_normal) * select(-1.0, 1.0, front_facing);
    let shadow = directional_shadow(fragment.world_position, geometric);
    var lit = brdf_direct(normal, to_view, to_light, albedo, metallic, roughness, irradiance * shadow);
    lit += ambient_flat(albedo, metallic, globals.ambient_color.rgb * globals.ambient_color.w, occlusion);
    lit += emitted;

    // The flash overlay after lighting, exactly as lit.wgsl (ADR-0017).
    let flashed = mix(lit, draw.flash.rgb, clamp(draw.flash.w, 0.0, 1.0));
    return vec4<f32>(min(flashed, vec3<f32>(MAX_SCENE_LIGHT)), alpha);
}
