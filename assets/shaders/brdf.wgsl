// brdf.wgsl — the metallic-roughness BRDF. Included, never compiled alone.
//
// Shared so that the forward PBR pass and a later deferred lighting pass (B2)
// light surfaces identically. A CPU mirror in crates/renderer/src/pbr.rs is
// what the GPU tests hold this to. Change both together.
//
// Conventions: every vector is unit length and in the same space; `n_dot_x`
// values are clamped by the caller; roughness is perceptual (alpha =
// roughness squared).

const PI: f32 = 3.14159265358979;

// GGX / Trowbridge-Reitz normal distribution.
fn distribution_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let alpha2 = alpha * alpha;
    let denominator = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * denominator * denominator);
}

// Height-correlated Smith visibility: G / (4 n.l n.v) in one term.
fn visibility_smith_ggx(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let alpha2 = alpha * alpha;
    let view = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - alpha2) + alpha2);
    let light = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - alpha2) + alpha2);
    return 0.5 / max(view + light, 1e-5);
}

fn fresnel_schlick(v_dot_h: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - v_dot_h, 5.0);
}

// Reflected radiance from one light of `irradiance` (light colour times
// intensity times pi, so intensity 1 lights a white Lambert surface to 1).
fn brdf_direct(
    normal: vec3<f32>,
    to_view: vec3<f32>,
    to_light: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    irradiance: vec3<f32>,
) -> vec3<f32> {
    let n_dot_l = max(dot(normal, to_light), 0.0);
    if n_dot_l <= 0.0 {
        return vec3<f32>(0.0);
    }
    let halfway = normalize(to_view + to_light);
    let n_dot_v = max(dot(normal, to_view), 1e-4);
    let n_dot_h = max(dot(normal, halfway), 0.0);
    let v_dot_h = max(dot(to_view, halfway), 0.0);
    let alpha = roughness * roughness;

    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let fresnel = fresnel_schlick(v_dot_h, f0);
    let specular = distribution_ggx(n_dot_h, alpha) * visibility_smith_ggx(n_dot_v, n_dot_l, alpha) * fresnel;
    let diffuse = (vec3<f32>(1.0) - fresnel) * (1.0 - metallic) * albedo / PI;
    return (diffuse + specular) * irradiance * n_dot_l;
}

// Flat ambient light, until image-based lighting replaces it (C2): diffuse
// from the albedo that is not metal, plus a specular share at F0 so metals do
// not go black where the key light does not reach.
fn ambient_flat(albedo: vec3<f32>, metallic: f32, ambient: vec3<f32>, occlusion: f32) -> vec3<f32> {
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    return (albedo * (1.0 - metallic) + f0) * ambient * occlusion;
}
