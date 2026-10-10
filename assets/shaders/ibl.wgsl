// ibl.wgsl — image-based ambient light for PBR surfaces (C2). Included by
// pbr_lit.wgsl, never compiled alone.
//
// Bindings, group 0 (crates/renderer/src/pipeline.rs):
//
//   @group(0) @binding(5)  linear sampler, shared with aerial.wgsl
//   @group(0) @binding(7)  the prefiltered environment cube: mip m is roughness m / (mips - 1)
//   @group(0) @binding(8)  Environment uniform: on/off, mips, nine SH irradiance coefficients
//   @group(0) @binding(9)  the split-sum BRDF LUT: (A, B) by n.v and roughness

#include "brdf.wgsl"
#include "aerial.wgsl"

struct Environment {
    header: vec4<f32>,          // x = 1 when image-based light is on, y = prefiltered mip count
    sh: array<vec4<f32>, 9>,    // radiance projected on band-2 harmonics, rgb
}

@group(0) @binding(7) var environment_cube: texture_cube<f32>;
@group(0) @binding(8) var<uniform> environment: Environment;
@group(0) @binding(9) var brdf_lut: texture_2d<f32>;

// Irradiance arriving at a surface facing `n`: the radiance's harmonics
// convolved with the clamped cosine (Ramamoorthi and Hanrahan, 2001).
fn sh_irradiance(n: vec3<f32>) -> vec3<f32> {
    let band0 = PI;
    let band1 = 2.0 * PI / 3.0;
    let band2 = PI / 4.0;
    let sh = environment.sh;
    var irradiance = sh[0].rgb * 0.282095 * band0;
    irradiance += (sh[1].rgb * n.y + sh[2].rgb * n.z + sh[3].rgb * n.x) * 0.488603 * band1;
    irradiance += (sh[4].rgb * 1.092548 * n.x * n.y
        + sh[5].rgb * 1.092548 * n.y * n.z
        + sh[6].rgb * 0.315392 * (3.0 * n.z * n.z - 1.0)
        + sh[7].rgb * 1.092548 * n.x * n.z
        + sh[8].rgb * 0.546274 * (n.x * n.x - n.y * n.y)) * band2;
    return max(irradiance, vec3<f32>(0.0));
}

// Ambient light from the environment: diffuse from the irradiance, and
// specular from the prefiltered cube through the split-sum BRDF table.
fn image_based_light(
    normal: vec3<f32>,
    to_view: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    occlusion: f32,
) -> vec3<f32> {
    let n_dot_v = clamp(dot(normal, to_view), 1e-3, 1.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let scale_bias = textureSampleLevel(brdf_lut, aerial_sampler, vec2<f32>(n_dot_v, roughness), 0.0).rg;
    let specular_reflectance = f0 * scale_bias.x + scale_bias.y;
    let diffuse = sh_irradiance(normal) * albedo * (1.0 - metallic) / PI;
    let reflected = reflect(-to_view, normal);
    let level = roughness * (environment.header.y - 1.0);
    let specular = textureSampleLevel(environment_cube, aerial_sampler, reflected, level).rgb * specular_reflectance;
    return (diffuse * (vec3<f32>(1.0) - specular_reflectance) + specular) * occlusion;
}
