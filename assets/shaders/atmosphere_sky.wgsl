// atmosphere_sky.wgsl — the sky (D3): a per-frame sky-view LUT, then a
// fullscreen pass that draws it behind all geometry, with the sun disc.
//
// The sky-view LUT holds sky luminance for unit sun illuminance around the
// camera, by view zenith (v, with the horizon in the middle and texels
// packed toward it) and azimuth relative to the sun (u). Hillaire 2020 §5.3.
//
// Bindings, one group; each entry point uses only its own:
//
//   @group(0) @binding(0)  Atmosphere            uniform
//   @group(0) @binding(1)  SkyFrame              uniform, per frame
//   @group(0) @binding(2)  transmittance LUT     sampled
//   @group(0) @binding(3)  multi-scattering LUT  sampled
//   @group(0) @binding(4)  linear sampler
//   @group(0) @binding(5)  sky-view out          storage, rgba16float
//   @group(0) @binding(6)  sky-view LUT          sampled, for the sky pass

#include "atmosphere.wgsl"

struct SkyFrame {
    inv_view_proj: mat4x4<f32>,  // clip to world, for view rays
    sun: vec4<f32>,              // xyz = unit direction to the sun, w = cos of its angular radius
    sun_illuminance: vec4<f32>,  // rgb = illuminance at the top of the atmosphere
    camera: vec4<f32>,           // x = camera radius from the planet centre, km
}

@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var<uniform> sky: SkyFrame;
@group(0) @binding(2) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(3) var multi_scattering_lut: texture_2d<f32>;
@group(0) @binding(4) var lut_sampler: sampler;
@group(0) @binding(5) var sky_view_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var sky_view_lut: texture_2d<f32>;

const PI: f32 = 3.14159265358979;
// Must match SKY_STEPS in atmosphere/sky.rs.
const SKY_STEPS: u32 = 30u;

fn rayleigh_phase(cos_angle: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + cos_angle * cos_angle);
}

// Cornette-Shanks: Henyey-Greenstein with the Rayleigh-like correction.
fn mie_phase(cos_angle: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denominator = pow(max(1.0 + g2 - 2.0 * g * cos_angle, 1e-6), 1.5);
    return 3.0 / (8.0 * PI) * (1.0 - g2) * (1.0 + cos_angle * cos_angle) / ((2.0 + g2) * denominator);
}

fn transmittance_toward(r: f32, mu: f32) -> vec3<f32> {
    let bottom = atmosphere.radii.x;
    let horizon = -sqrt(max(1.0 - (bottom / r) * (bottom / r), 0.0));
    if mu < horizon {
        return vec3<f32>(0.0);
    }
    return textureSampleLevel(transmittance_lut, lut_sampler, transmittance_uv(atmosphere, r, mu), 0.0).rgb;
}

fn multi_scattering(r: f32, mu_sun: f32) -> vec3<f32> {
    let bottom = atmosphere.radii.x;
    let uv = vec2<f32>(mu_sun * 0.5 + 0.5, (r - bottom) / (atmosphere.radii.y - bottom));
    return textureSampleLevel(multi_scattering_lut, lut_sampler, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
}

// The view zenith angle where the planet's limb is seen from radius `r`.
fn horizon_zenith(r: f32) -> vec2<f32> {
    let bottom = atmosphere.radii.x;
    let beta = acos(clamp(sqrt(max(r * r - bottom * bottom, 0.0)) / r, -1.0, 1.0));
    return vec2<f32>(PI - beta, beta);
}

// Sky luminance per unit sun illuminance along `view` from radius `r`, in a
// frame whose up is +y. Single scattering plus the multiple-scattering LUT.
fn integrate_sky(r: f32, view: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    let bottom = atmosphere.radii.x;
    let ground = distance_to_ground(atmosphere, r, view.y);
    var ray_length = distance_to_top(atmosphere, r, view.y);
    if ground >= 0.0 {
        ray_length = ground;
    }
    let cos_angle = dot(view, sun);
    let phase_rayleigh = rayleigh_phase(cos_angle);
    let phase_mie = mie_phase(cos_angle, atmosphere.mie.w);
    let dt = ray_length / f32(SKY_STEPS);
    var luminance = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    for (var step = 0u; step < SKY_STEPS; step++) {
        let t = (f32(step) + 0.5) * dt;
        let point = vec3<f32>(0.0, r, 0.0) + view * t;
        let height = length(point);
        let mu_sun = dot(point, sun) / height;
        let local = atmosphere_medium(atmosphere, height - bottom);
        let sunlight = transmittance_toward(height, mu_sun);
        let single = sunlight * (local.rayleigh * phase_rayleigh + vec3<f32>(local.mie * phase_mie));
        let multiple = local.scattering * multi_scattering(height, mu_sun);
        let step_transmittance = exp(-local.extinction * dt);
        let fraction = (vec3<f32>(1.0) - step_transmittance) / max(local.extinction, vec3<f32>(1e-12));
        luminance += throughput * (single + multiple) * fraction;
        throughput *= step_transmittance;
    }
    return luminance;
}

// LUT texel to view zenith cosine and the cosine of the azimuth from the sun.
fn sky_view_angles(uv: vec2<f32>, r: f32) -> vec2<f32> {
    let horizon = horizon_zenith(r);
    var zenith: f32;
    if uv.y < 0.5 {
        let coord = 1.0 - 2.0 * uv.y;
        zenith = horizon.x * (1.0 - coord * coord);
    } else {
        let coord = uv.y * 2.0 - 1.0;
        zenith = horizon.x + horizon.y * coord * coord;
    }
    let azimuth_coord = uv.x * uv.x;
    return vec2<f32>(cos(zenith), -(azimuth_coord * 2.0 - 1.0));
}

fn sky_view_uv(r: f32, cos_zenith: f32, cos_azimuth: f32) -> vec2<f32> {
    let horizon = horizon_zenith(r);
    let zenith = acos(clamp(cos_zenith, -1.0, 1.0));
    var v: f32;
    if zenith < horizon.x {
        v = (1.0 - sqrt(max(1.0 - zenith / horizon.x, 0.0))) * 0.5;
    } else {
        v = sqrt(clamp((zenith - horizon.x) / horizon.y, 0.0, 1.0)) * 0.5 + 0.5;
    }
    let u = sqrt(clamp(-cos_azimuth * 0.5 + 0.5, 0.0, 1.0));
    return vec2<f32>(u, v);
}

@compute @workgroup_size(8, 8)
fn sky_view_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(sky_view_out);
    if any(id.xy >= size) {
        return;
    }
    let r = sky.camera.x;
    let angles = sky_view_angles((vec2<f32>(id.xy) + 0.5) / vec2<f32>(size), r);
    let sin_zenith = sqrt(max(1.0 - angles.x * angles.x, 0.0));
    let sin_azimuth = sqrt(max(1.0 - angles.y * angles.y, 0.0));
    let view = vec3<f32>(sin_zenith * angles.y, angles.x, sin_zenith * sin_azimuth);
    let mu_sun = clamp(sky.sun.y, -1.0, 1.0);
    let sun = vec3<f32>(sqrt(max(1.0 - mu_sun * mu_sun, 0.0)), mu_sun, 0.0);
    textureStore(sky_view_out, id.xy, vec4<f32>(integrate_sky(r, view, sun), 1.0));
}

struct SkyVertex {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

// A fullscreen triangle on the far plane: depth 1, drawn with `LessEqual`
// against the scene's depth, so it fills only where nothing was drawn.
@vertex
fn sky_vs(@builtin(vertex_index) index: u32) -> SkyVertex {
    let xy = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u)) * 2.0 - 1.0;
    return SkyVertex(vec4<f32>(xy, 1.0, 1.0), xy);
}

@fragment
fn sky_fs(vertex: SkyVertex) -> @location(0) vec4<f32> {
    let near = sky.inv_view_proj * vec4<f32>(vertex.ndc, 0.0, 1.0);
    let far = sky.inv_view_proj * vec4<f32>(vertex.ndc, 1.0, 1.0);
    let view = normalize(far.xyz / far.w - near.xyz / near.w);
    let sun = sky.sun.xyz;
    let r = sky.camera.x;

    // Azimuth from the sun, measured in the horizontal plane.
    let flat_view = vec2<f32>(view.x, view.z);
    let flat_sun = vec2<f32>(sun.x, sun.z);
    var cos_azimuth = 1.0;
    if dot(flat_view, flat_view) > 1e-8 && dot(flat_sun, flat_sun) > 1e-8 {
        cos_azimuth = dot(normalize(flat_view), normalize(flat_sun));
    }
    let uv = sky_view_uv(r, view.y, cos_azimuth);
    var luminance = textureSampleLevel(sky_view_lut, lut_sampler, uv, 0.0).rgb * sky.sun_illuminance.rgb;

    // The sun's disc: its radiance is the illuminance spread over its solid
    // angle, dimmed by the air in front of it, and hidden by the ground.
    if dot(view, sun) > sky.sun.w && distance_to_ground(atmosphere, r, view.y) < 0.0 {
        let solid_angle = 2.0 * PI * (1.0 - sky.sun.w);
        luminance += transmittance_toward(r, view.y) * sky.sun_illuminance.rgb / solid_angle;
    }
    return vec4<f32>(min(luminance, vec3<f32>(60000.0)), 1.0);
}
