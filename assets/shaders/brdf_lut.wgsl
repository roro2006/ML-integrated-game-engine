// brdf_lut.wgsl — the split-sum BRDF integration LUT for image-based
// lighting (C2), baked once at startup.
//
// For a view angle (x: n.v) and perceptual roughness (y), the texel holds
// (A, B) with the specular reflectance of a uniform environment equal to
// F0 * A + B. It integrates the very BRDF the direct light uses
// (brdf.wgsl: GGX, height-correlated Smith, Schlick), so image-based and
// direct light agree on how rough metal looks. Karis, "Real Shading in
// Unreal Engine 4" (2013), with the visibility term swapped for ours.
//
//   @group(0) @binding(0)  the LUT  storage, rgba16float: (A, B, 0, 1)

#include "brdf.wgsl"

@group(0) @binding(0) var lut_out: texture_storage_2d<rgba16float, write>;

// Must match BRDF_LUT_SAMPLES in crates/renderer/src/ibl.rs.
const SAMPLES: u32 = 1024u;

// The `index`th of `count` low-discrepancy points in the unit square.
fn hammersley(index: u32, count: u32) -> vec2<f32> {
    return vec2<f32>(f32(index) / f32(count), f32(reverseBits(index)) * 2.3283064365386963e-10);
}

// A half vector, around +z, distributed as GGX with `alpha`.
fn importance_sample_ggx(xi: vec2<f32>, alpha: f32) -> vec3<f32> {
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    return vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

@compute @workgroup_size(8, 8)
fn brdf_lut_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(lut_out);
    if any(id.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let n_dot_v = uv.x;
    let alpha = uv.y * uv.y;
    let view = vec3<f32>(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);
    var scale = 0.0;
    var bias = 0.0;
    for (var index = 0u; index < SAMPLES; index++) {
        let half_vector = importance_sample_ggx(hammersley(index, SAMPLES), alpha);
        let light = reflect(-view, half_vector);
        let n_dot_l = light.z;
        if n_dot_l <= 0.0 {
            continue;
        }
        let n_dot_h = max(half_vector.z, 0.0);
        let v_dot_h = max(dot(view, half_vector), 0.0);
        // BRDF times n.l over the GGX half-vector pdf: D cancels, leaving
        // the visibility and the change of variables from h to l.
        let weight = 4.0 * visibility_smith_ggx(n_dot_v, n_dot_l, alpha) * n_dot_l * v_dot_h / max(n_dot_h, 1e-5);
        let fresnel = pow(1.0 - v_dot_h, 5.0);
        scale += (1.0 - fresnel) * weight;
        bias += fresnel * weight;
    }
    textureStore(lut_out, id.xy, vec4<f32>(scale / f32(SAMPLES), bias / f32(SAMPLES), 0.0, 1.0));
}
