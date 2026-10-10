// aerial.wgsl — the air between the camera and a lit surface (D3): aerial
// perspective from the atmosphere's froxel LUT, then analytic height fog.
// Included by lit.wgsl and pbr_lit.wgsl, never compiled alone.
//
// Bindings, group 0 next to Globals and the shadows (crates/renderer/src/pipeline.rs):
//
//   @group(0) @binding(4)  aerial perspective  texture_3d, rgb = in-scattered light
//                                              per unit sun illuminance, a = transmittance
//   @group(0) @binding(5)  linear sampler
//   @group(0) @binding(6)  FogData             uniform, one per frame

struct FogData {
    aerial: vec4<f32>,           // x = 1 when aerial perspective is on, y = its range in km
    sun_illuminance: vec4<f32>,  // rgb, the sky's sun
    height_fog: vec4<f32>,       // density per metre at the base height, falloff per metre, base height, max opacity
    fog_color: vec4<f32>,        // rgb = the fog's light, w = 1 when height fog is on
}

@group(0) @binding(4) var aerial_lut: texture_3d<f32>;
@group(0) @binding(5) var aerial_sampler: sampler;
@group(0) @binding(6) var<uniform> fog: FogData;

const METRES_PER_KM: f32 = 1000.0;

// Optical depth of exponential height fog along the segment from `start` to
// `end`, in closed form: the density integral of exp(-falloff * height).
fn height_fog_depth(start: vec3<f32>, end: vec3<f32>) -> f32 {
    let density = fog.height_fog.x;
    let falloff = fog.height_fog.y;
    let base = fog.height_fog.z;
    let length = distance(start, end);
    let rise = (end.y - start.y) * falloff;
    var shape = 1.0;
    if abs(rise) > 1e-4 {
        shape = (1.0 - exp(-rise)) / rise;
    }
    return density * exp(-falloff * (start.y - base)) * length * shape;
}

// The light reaching the camera from a surface lit to `color` at `world`.
fn apply_atmosphere(color: vec3<f32>, world: vec3<f32>, camera: vec3<f32>, view_proj: mat4x4<f32>) -> vec3<f32> {
    var result = color;
    if fog.aerial.x > 0.5 {
        let clip = view_proj * vec4<f32>(world, 1.0);
        let ndc = clip.xy / clip.w;
        let slice = distance(world, camera) / METRES_PER_KM / fog.aerial.y;
        let uvw = vec3<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, slice);
        let air = textureSampleLevel(aerial_lut, aerial_sampler, uvw, 0.0);
        // The first slice's centre is half a slice out; closer than that,
        // fade to no air at all so a surface at the camera is untouched.
        let slices = f32(textureDimensions(aerial_lut).z);
        let weight = clamp(slice * slices * 2.0, 0.0, 1.0);
        let lit = result * air.a + air.rgb * fog.sun_illuminance.rgb;
        result = mix(result, lit, weight);
    }
    if fog.fog_color.w > 0.5 {
        let amount = min(1.0 - exp(-height_fog_depth(camera, world)), fog.height_fog.w);
        result = mix(result, fog.fog_color.rgb, amount);
    }
    return result;
}
