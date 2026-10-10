// atmosphere_luts.wgsl — bakes the transmittance and multiple-scattering
// LUTs (D3) in compute. Run once per change of the atmosphere parameters,
// transmittance first: the multiple-scattering pass reads it.
//
// Bindings, one group; each entry point uses only its own:
//
//   @group(0) @binding(0)  Atmosphere           uniform
//   @group(0) @binding(1)  transmittance out    storage, rgba16float
//   @group(0) @binding(2)  transmittance LUT    sampled, for multiple scattering
//   @group(0) @binding(3)  linear sampler
//   @group(0) @binding(4)  multi-scattering out storage, rgba16float

#include "atmosphere.wgsl"

@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var transmittance_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(3) var lut_sampler: sampler;
@group(0) @binding(4) var multi_scattering_out: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265358979;
const TRANSMITTANCE_STEPS: u32 = 40u;
// Must match MS_DIRECTIONS_PER_SIDE and MS_STEPS in atmosphere/reference.rs.
const MS_DIRECTIONS_PER_SIDE: u32 = 8u;
const MS_STEPS: u32 = 20u;

@compute @workgroup_size(8, 8)
fn transmittance_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(transmittance_out);
    if any(id.xy >= size) {
        return;
    }
    let r_mu = transmittance_r_mu(atmosphere, (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size));
    let r = r_mu.x;
    let mu = r_mu.y;
    let dt = distance_to_top(atmosphere, r, mu) / f32(TRANSMITTANCE_STEPS);
    var depth = vec3<f32>(0.0);
    for (var step = 0u; step < TRANSMITTANCE_STEPS; step++) {
        let t = (f32(step) + 0.5) * dt;
        let height = sqrt(r * r + t * t + 2.0 * r * mu * t);
        depth += atmosphere_medium(atmosphere, height - atmosphere.radii.x).extinction * dt;
    }
    textureStore(transmittance_out, id.xy, vec4<f32>(exp(-depth), 1.0));
}

// Transmittance toward the sun from radius `r`, zero in the planet's shadow.
fn sun_transmittance(r: f32, mu: f32) -> vec3<f32> {
    let bottom = atmosphere.radii.x;
    let horizon = -sqrt(max(1.0 - (bottom / r) * (bottom / r), 0.0));
    if mu < horizon {
        return vec3<f32>(0.0);
    }
    return textureSampleLevel(transmittance_lut, lut_sampler, transmittance_uv(atmosphere, r, mu), 0.0).rgb;
}

@compute @workgroup_size(8, 8)
fn multi_scattering_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(multi_scattering_out);
    if any(id.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let bottom = atmosphere.radii.x;
    let r = bottom + uv.y * (atmosphere.radii.y - bottom);
    let mu_sun = uv.x * 2.0 - 1.0;
    let sun = vec3<f32>(sqrt(max(1.0 - mu_sun * mu_sun, 0.0)), mu_sun, 0.0);
    let isotropic = 1.0 / (4.0 * PI);

    var luminance = vec3<f32>(0.0);
    var transfer = vec3<f32>(0.0);
    for (var i = 0u; i < MS_DIRECTIONS_PER_SIDE; i++) {
        for (var j = 0u; j < MS_DIRECTIONS_PER_SIDE; j++) {
            let cos_theta = 1.0 - 2.0 * (f32(i) + 0.5) / f32(MS_DIRECTIONS_PER_SIDE);
            let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
            let phi = 2.0 * PI * (f32(j) + 0.5) / f32(MS_DIRECTIONS_PER_SIDE);
            let direction = vec3<f32>(sin_theta * cos(phi), cos_theta, sin_theta * sin(phi));
            let ground = distance_to_ground(atmosphere, r, direction.y);
            var ray_length = distance_to_top(atmosphere, r, direction.y);
            if ground >= 0.0 {
                ray_length = ground;
            }
            let dt = ray_length / f32(MS_STEPS);
            var throughput = vec3<f32>(1.0);
            for (var step = 0u; step < MS_STEPS; step++) {
                let t = (f32(step) + 0.5) * dt;
                let point = vec3<f32>(0.0, r, 0.0) + direction * t;
                let height = length(point);
                let local = atmosphere_medium(atmosphere, height - bottom);
                let sunlight = sun_transmittance(height, dot(point, sun) / height);
                let step_transmittance = exp(-local.extinction * dt);
                // Energy-conserving: the scattered light integrated over the
                // step. Extinction is never zero inside this atmosphere.
                let fraction = local.scattering * (1.0 - step_transmittance) / max(local.extinction, vec3<f32>(1e-12));
                luminance += throughput * sunlight * isotropic * fraction;
                transfer += throughput * fraction;
                throughput *= step_transmittance;
            }
            if ground >= 0.0 {
                let point = vec3<f32>(0.0, r, 0.0) + direction * ground;
                let mu_sun_ground = dot(point, sun) / length(point);
                let sunlight = sun_transmittance(bottom, mu_sun_ground);
                luminance += throughput * sunlight * max(mu_sun_ground, 0.0) * atmosphere.ground_albedo.rgb / PI;
            }
        }
    }
    // Sphere averages: solid angle 4 pi / N per direction times the isotropic
    // phase 1 / (4 pi) is 1 / N. Then all orders: L2 / (1 - f_ms).
    let count = f32(MS_DIRECTIONS_PER_SIDE * MS_DIRECTIONS_PER_SIDE);
    let psi = (luminance / count) / (vec3<f32>(1.0) - transfer / count);
    textureStore(multi_scattering_out, id.xy, vec4<f32>(psi, 1.0));
}
