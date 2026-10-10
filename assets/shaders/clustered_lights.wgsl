// clustered_lights.wgsl — point and spot lights by cluster (C5). Included
// by light_cluster_build.wgsl, which fills the lists, and pbr_lit.wgsl,
// which shades with them. Never compiled alone.
//
// Bindings, group 0 (crates/renderer/src/pipeline.rs):
//
//   @group(0) @binding(10)  ClusterParams  uniform, per frame
//   @group(0) @binding(11)  lights         storage, read: GpuLight per light
//   @group(0) @binding(12)  light grid     storage, read: per cluster a count, then its light indices

#include "brdf.wgsl"

struct ClusterParams {
    inv_view_proj: mat4x4<f32>,
    eye_near: vec4<f32>,     // xyz = the eye, w = view depth where the clusters start
    forward_far: vec4<f32>,  // xyz = unit view direction, w = view depth where they end
    grid: vec4<u32>,         // xyz = clusters across, down and in depth
    counts: vec4<u32>,       // x = lights this frame, y = most lights a cluster holds
    screen: vec4<f32>,       // xy = the target's size in pixels
}

struct GpuLight {
    position_range: vec4<f32>,   // xyz = world position, w = range
    color_intensity: vec4<f32>,  // rgb, intensity
    direction_spot: vec4<f32>,   // xyz = spot direction, w = 1 for a spot
    cone: vec4<f32>,             // x = cos inner angle, y = cos outer angle, z = sin outer angle
}

@group(0) @binding(10) var<uniform> clusters: ClusterParams;
@group(0) @binding(11) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(12) var<storage, read> light_grid: array<u32>;

// Where depth slice `slice` begins: exponential between near and far.
fn slice_depth(slice: u32) -> f32 {
    let near = clusters.eye_near.w;
    let far = clusters.forward_far.w;
    return near * pow(far / near, f32(slice) / f32(clusters.grid.z));
}

// The cluster holding a fragment at `pixel` (framebuffer coordinates, y
// down) and `world`.
fn cluster_of(pixel: vec2<f32>, world: vec3<f32>) -> u32 {
    let grid = clusters.grid;
    let across = clamp(u32(pixel.x / clusters.screen.x * f32(grid.x)), 0u, grid.x - 1u);
    // Clusters count up the screen, as NDC y does.
    let up = clamp(u32((1.0 - pixel.y / clusters.screen.y) * f32(grid.y)), 0u, grid.y - 1u);
    let near = clusters.eye_near.w;
    let far = clusters.forward_far.w;
    let depth = max(dot(world - clusters.eye_near.xyz, clusters.forward_far.xyz), near);
    let slice = clamp(u32(log(depth / near) / log(far / near) * f32(grid.z)), 0u, grid.z - 1u);
    return across + grid.x * (up + grid.y * slice);
}

// Closer than this, in metres, a light is treated as this far away, so its
// inverse square stays finite on a surface it touches.
const MIN_LIGHT_DISTANCE: f32 = 0.1;

// How much of a light reaches `world`, by distance and cone: inverse square
// windowed to reach zero at the range (Karis 2013). Exactly 1 at 1 m, less
// the window, so `intensity` means what it does for the key light there.
fn light_falloff(light: GpuLight, to_light: vec3<f32>, distance: f32) -> f32 {
    let ratio = distance / light.position_range.w;
    let window = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
    let clamped = max(distance, MIN_LIGHT_DISTANCE);
    var falloff = window * window / (clamped * clamped);
    if light.direction_spot.w > 0.5 {
        let along = dot(-to_light, light.direction_spot.xyz);
        falloff *= smoothstep(light.cone.y, light.cone.x, along);
    }
    return falloff;
}

// Light reflected toward the viewer from every local light in the cluster.
fn local_lights(
    pixel: vec2<f32>,
    world: vec3<f32>,
    normal: vec3<f32>,
    to_view: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    if clusters.counts.x == 0u {
        return vec3<f32>(0.0);
    }
    let base = cluster_of(pixel, world) * (clusters.counts.y + 1u);
    let count = light_grid[base];
    var total = vec3<f32>(0.0);
    for (var slot = 0u; slot < count; slot++) {
        let light = lights[light_grid[base + 1u + slot]];
        let offset = light.position_range.xyz - world;
        let distance = length(offset);
        // At the light itself there is no direction to light from.
        if distance >= light.position_range.w || distance < 1e-4 {
            continue;
        }
        let to_light = offset / distance;
        let irradiance = light.color_intensity.rgb * light.color_intensity.w * PI * light_falloff(light, to_light, distance);
        total += brdf_direct(normal, to_view, to_light, albedo, metallic, roughness, irradiance);
    }
    return total;
}
