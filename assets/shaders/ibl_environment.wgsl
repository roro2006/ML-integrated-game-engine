// ibl_environment.wgsl — the environment for image-based lighting (C2).
//
// Three compute entry points, run in order when the sky changes:
//
//   cube_main       the sky's radiance in every direction, into a cube
//   prefilter_main  that cube blurred by GGX for one roughness per mip
//   sh_main         its irradiance as nine spherical-harmonic coefficients
//
// Bindings, one group; each entry point uses only its own:
//
//   @group(0) @binding(0)  Atmosphere           uniform
//   @group(0) @binding(1)  EnvironmentParams    uniform
//   @group(0) @binding(2)  sky-view LUT         sampled
//   @group(0) @binding(3)  linear sampler
//   @group(0) @binding(4)  radiance out         storage, 2D array of six faces
//   @group(0) @binding(5)  radiance cube        sampled cube
//   @group(0) @binding(6)  prefiltered out      storage, one mip, six faces
//   @group(0) @binding(7)  SH out               storage buffer, nine vec4
//   @group(0) @binding(8)  radiance faces       sampled 2D array, for loads
//
// Cube faces follow the WebGPU order and orientation: +X, -X, +Y, -Y, +Z, -Z.

#include "atmosphere.wgsl"

struct EnvironmentParams {
    sun: vec4<f32>,               // xyz = unit direction to the sun
    sun_illuminance: vec4<f32>,   // rgb
    camera: vec4<f32>,            // x = camera radius from the planet centre, km
    uniform_radiance: vec4<f32>,  // rgb; w = 1 replaces the sky with it (tests)
}

@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var<uniform> params: EnvironmentParams;
@group(0) @binding(2) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(3) var lut_sampler: sampler;
@group(0) @binding(4) var radiance_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(5) var radiance_cube: texture_cube<f32>;
@group(0) @binding(6) var prefiltered_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(7) var<storage, read_write> sh_out: array<vec4<f32>, 9>;
@group(0) @binding(8) var radiance_faces: texture_2d_array<f32>;

const PI: f32 = 3.14159265358979;
// Must match ENVIRONMENT_SIZE and PREFILTER_MIPS in crates/renderer/src/ibl.rs.
const BASE_SIZE: f32 = 64.0;
const PREFILTER_MIPS: f32 = 7.0;
const PREFILTER_SAMPLES: u32 = 64u;
const SH_THREADS: u32 = 64u;

// The direction through texel `xy` of cube face `face`, `size` texels wide.
fn face_direction(face: u32, xy: vec2<u32>, size: f32) -> vec3<f32> {
    let u = (f32(xy.x) + 0.5) / size * 2.0 - 1.0;
    let v = (f32(xy.y) + 0.5) / size * 2.0 - 1.0;
    var direction: vec3<f32>;
    switch face {
        case 0u: { direction = vec3<f32>(1.0, -v, -u); }
        case 1u: { direction = vec3<f32>(-1.0, -v, u); }
        case 2u: { direction = vec3<f32>(u, 1.0, v); }
        case 3u: { direction = vec3<f32>(u, -1.0, -v); }
        case 4u: { direction = vec3<f32>(u, -v, 1.0); }
        default: { direction = vec3<f32>(-u, -v, -1.0); }
    }
    return normalize(direction);
}

// The sky's radiance toward `direction`, as the sky pass draws it but
// without the sun's disc: the directional light already carries the sun.
fn sky_radiance(direction: vec3<f32>) -> vec3<f32> {
    if params.uniform_radiance.w > 0.5 {
        return params.uniform_radiance.rgb;
    }
    let flat_view = vec2<f32>(direction.x, direction.z);
    let flat_sun = vec2<f32>(params.sun.x, params.sun.z);
    var cos_azimuth = 1.0;
    if dot(flat_view, flat_view) > 1e-8 && dot(flat_sun, flat_sun) > 1e-8 {
        cos_azimuth = dot(normalize(flat_view), normalize(flat_sun));
    }
    let uv = sky_view_uv(atmosphere, params.camera.x, direction.y, cos_azimuth);
    return textureSampleLevel(sky_view_lut, lut_sampler, uv, 0.0).rgb * params.sun_illuminance.rgb;
}

@compute @workgroup_size(8, 8, 1)
fn cube_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(radiance_out);
    if any(id.xy >= size) || id.z >= 6u {
        return;
    }
    let direction = face_direction(id.z, id.xy, f32(size.x));
    textureStore(radiance_out, id.xy, id.z, vec4<f32>(sky_radiance(direction), 1.0));
}

fn hammersley(index: u32, count: u32) -> vec2<f32> {
    return vec2<f32>(f32(index) / f32(count), f32(reverseBits(index)) * 2.3283064365386963e-10);
}

@compute @workgroup_size(8, 8, 1)
fn prefilter_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(prefiltered_out);
    if any(id.xy >= size) || id.z >= 6u {
        return;
    }
    // Each mip is one roughness, from 0 at the base to 1 at the last mip.
    let mip = log2(BASE_SIZE / f32(size.x));
    let roughness = mip / (PREFILTER_MIPS - 1.0);
    let normal = face_direction(id.z, id.xy, f32(size.x));
    if roughness < 1e-3 {
        textureStore(prefiltered_out, id.xy, id.z, textureSampleLevel(radiance_cube, lut_sampler, normal, 0.0));
        return;
    }
    // Karis's approximation: view = normal, and samples weighted by n.l.
    let alpha = roughness * roughness;
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if abs(normal.y) > 0.999 {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    var total = vec3<f32>(0.0);
    var weight = 0.0;
    for (var index = 0u; index < PREFILTER_SAMPLES; index++) {
        let xi = hammersley(index, PREFILTER_SAMPLES);
        let phi = 2.0 * PI * xi.x;
        let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
        let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
        let half_vector = tangent * (sin_theta * cos(phi)) + bitangent * (sin_theta * sin(phi)) + normal * cos_theta;
        let light = reflect(-normal, half_vector);
        let n_dot_l = dot(normal, light);
        if n_dot_l > 0.0 {
            total += textureSampleLevel(radiance_cube, lut_sampler, light, 0.0).rgb * n_dot_l;
            weight += n_dot_l;
        }
    }
    textureStore(prefiltered_out, id.xy, id.z, vec4<f32>(total / max(weight, 1e-6), 1.0));
}

// Real spherical harmonics to band 2, unnormalised constants included.
fn sh_basis(d: vec3<f32>) -> array<f32, 9> {
    return array<f32, 9>(
        0.282095,
        0.488603 * d.y,
        0.488603 * d.z,
        0.488603 * d.x,
        1.092548 * d.x * d.y,
        1.092548 * d.y * d.z,
        0.315392 * (3.0 * d.z * d.z - 1.0),
        1.092548 * d.x * d.z,
        0.546274 * (d.x * d.x - d.y * d.y),
    );
}

var<workgroup> partial: array<vec4<f32>, 576>;

@compute @workgroup_size(64)
fn sh_main(@builtin(local_invocation_index) thread: u32) {
    let size = textureDimensions(radiance_faces).x;
    let texels = size * size * 6u;
    var sums: array<vec3<f32>, 9>;
    for (var i = 0u; i < 9u; i++) {
        sums[i] = vec3<f32>(0.0);
    }
    for (var texel = thread; texel < texels; texel += SH_THREADS) {
        let face = texel / (size * size);
        let xy = vec2<u32>(texel % size, (texel / size) % size);
        let direction = face_direction(face, xy, f32(size));
        // The texel's solid angle on the unit cube.
        let u = (f32(xy.x) + 0.5) / f32(size) * 2.0 - 1.0;
        let v = (f32(xy.y) + 0.5) / f32(size) * 2.0 - 1.0;
        let solid_angle = 4.0 / (f32(size * size) * pow(1.0 + u * u + v * v, 1.5));
        let radiance = textureLoad(radiance_faces, xy, face, 0).rgb * solid_angle;
        let basis = sh_basis(direction);
        for (var i = 0u; i < 9u; i++) {
            sums[i] += radiance * basis[i];
        }
    }
    for (var i = 0u; i < 9u; i++) {
        partial[thread * 9u + i] = vec4<f32>(sums[i], 0.0);
    }
    workgroupBarrier();
    if thread < 9u {
        var total = vec3<f32>(0.0);
        for (var other = 0u; other < SH_THREADS; other++) {
            total += partial[other * 9u + thread].rgb;
        }
        sh_out[thread] = vec4<f32>(total, 0.0);
    }
}
