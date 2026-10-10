// gtao.wgsl — ground-truth ambient occlusion at half resolution (C4).
//
// Clean-room from Jimenez, Wu, Pesce and Jarabo, "Practical Real-Time
// Strategies for Accurate Indirect Occlusion" (2016). Each half-resolution
// pixel takes the full-resolution pixel at twice its coordinates, cuts the
// hemisphere around the view direction into a few slices, finds the highest
// occluder on either side of each slice by marching the depth buffer, and
// integrates cosine-weighted visibility between the two horizons
// analytically. Occluders farther than the radius fade out, so a thin object
// far in front of a surface leaves no halo on it.
//
// Positions come from depth through the inverse view-projection, and normals
// from the prepass's world-space target, so everything is in world space:
// no view matrix is needed, and the slice plane is built from the view
// vector and the screen direction lifted onto the pixel's depth.
//
// Two entry points, each with its own bind group:
//
//   gtao_main     traces into a half-resolution target
//     @group(0) @binding(0)  AoParams
//     @group(0) @binding(1)  depth           texture_depth_2d, full resolution
//     @group(0) @binding(2)  normals         texture_2d<f32>, n * 0.5 + 0.5
//     @group(0) @binding(3)  raw occlusion   storage, rgba16float
//   denoise_main  a depth-aware spatial blur of that target
//     @group(0) @binding(0)  AoParams
//     @group(0) @binding(4)  raw occlusion   texture_2d<f32>
//     @group(0) @binding(5)  occlusion       storage, rgba16float
//
// Both targets hold r = visibility (1 open, 0 fully hidden) and g = the
// traced point's distance from the camera, which the denoise and the lit
// shaders' upsample (ambient_occlusion.wgsl) weigh neighbours by.

struct AoParams {
    inv_view_proj: mat4x4<f32>, // clip to world
    eye: vec4<f32>,             // xyz = camera, w = radius in metres
    size: vec4<f32>,            // full width, full height, 1 / width, 1 / height
    settings: vec4<f32>,        // strength, largest radius in pixels, unused, unused
}

@group(0) @binding(0) var<uniform> params: AoParams;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var normal_texture: texture_2d<f32>;
@group(0) @binding(3) var raw_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var raw_input: texture_2d<f32>;
@group(0) @binding(5) var denoised_output: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265;
const HALF_PI: f32 = 1.57079633;
// Slices per pixel and steps per side of each slice. The spatial noise
// rotates the slices over a 4 x 4 block, which the denoise then averages, so
// a pixel's result effectively draws on sixteen times as many slices.
const SLICES: u32 = 2u;
const STEPS: u32 = 6u;
// Occluders fade out over the last part of the radius rather than vanish.
const FALLOFF_FRACTION: f32 = 0.4;
// A distance written where there is only sky; no surface is this far, and
// it fits a half float.
const SKY_DISTANCE: f32 = 60000.0;
// Neighbours whose distance differs by more than about this fraction are
// different surfaces, and the denoise ignores them.
const DEPTH_TOLERANCE: f32 = 0.05;
const DENOISE_RADIUS: i32 = 2;

// A 4 x 4 ordered-dither matrix, sixteen evenly spread values.
const BAYER: array<u32, 16> = array<u32, 16>(0u, 8u, 2u, 10u, 12u, 4u, 14u, 6u, 3u, 11u, 1u, 9u, 15u, 7u, 13u, 5u);

fn world_at(pixel: vec2<f32>, depth: f32) -> vec3<f32> {
    let uv = pixel * params.size.zw;
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = params.inv_view_proj * clip;
    return world.xyz / world.w;
}

// Cosine-weighted visibility of one side of a slice, from the normal's angle
// `n` to the horizon `h`, both measured from the view vector (the paper's
// equation 10, halved per side). Mirrored by `ao::side_visibility`.
fn side_visibility(h: f32, n: f32) -> f32 {
    return 0.25 * (-cos(2.0 * h - n) + cos(n) + 2.0 * h * sin(n));
}

// What a slice integrates to with nothing in the way: both horizons at the
// tangent. Not 1 unless the normal faces the viewer; each slice's result is
// divided by it, so an open surface is exactly 1 at any angle. Mirrored by
// `ao::open_slice`.
fn open_slice(n: f32) -> f32 {
    return cos(n) + n * sin(n);
}

@compute @workgroup_size(8, 8, 1)
fn gtao_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let half_size = textureDimensions(raw_output);
    if id.x >= half_size.x || id.y >= half_size.y {
        return;
    }
    let full_size = vec2<i32>(textureDimensions(depth_texture));
    let full_pixel = min(vec2<i32>(id.xy) * 2, full_size - 1);
    let depth = textureLoad(depth_texture, full_pixel, 0);
    if depth >= 1.0 {
        textureStore(raw_output, id.xy, vec4<f32>(1.0, SKY_DISTANCE, 0.0, 1.0));
        return;
    }

    let centre = vec2<f32>(full_pixel) + 0.5;
    let position = world_at(centre, depth);
    let to_eye = params.eye.xyz - position;
    let distance = length(to_eye);
    let view = to_eye / distance;
    let normal = normalize(textureLoad(normal_texture, full_pixel, 0).xyz * 2.0 - 1.0);
    let radius = params.eye.w;
    // Points at one depth-buffer value share a view depth, so the next
    // pixel over at the same depth is exactly one pixel's width away.
    let pixel_world = length(world_at(centre + vec2<f32>(1.0, 0.0), depth) - position);
    let radius_pixels = min(radius / max(pixel_world, 1e-6), params.settings.y);
    if radius_pixels < 1.0 {
        // The radius is under a pixel here: nothing to resolve on screen.
        textureStore(raw_output, id.xy, vec4<f32>(1.0, distance, 0.0, 1.0));
        return;
    }

    let pattern = (id.x & 3u) + 4u * (id.y & 3u);
    let rotation = (f32(BAYER[pattern]) + 0.5) / 16.0;
    let jitter = (f32(BAYER[(pattern + 5u) & 15u]) + 0.5) / 16.0;
    let falloff_start = radius * (1.0 - FALLOFF_FRACTION);

    var visibility = 0.0;
    var weight = 0.0;
    for (var slice = 0u; slice < SLICES; slice++) {
        let angle = (f32(slice) + rotation) * PI / f32(SLICES);
        let direction = vec2<f32>(cos(angle), sin(angle));
        // The slice plane holds the view vector and this screen direction.
        let across = world_at(centre + direction, depth) - position;
        let slice_direction = normalize(across - view * dot(across, view));
        let axis = cross(slice_direction, view);
        let projected = normal - axis * dot(normal, axis);
        let projected_length = length(projected);
        if projected_length < 1e-4 {
            continue;
        }
        let cos_n = clamp(dot(projected, view) / projected_length, -1.0, 1.0);
        // Clamped to the visible half: a smooth-shaded normal near a
        // silhouette can lean away from the viewer, and past pi / 2 the
        // tangent seeds below leave acos's range.
        let n = clamp(sign(dot(projected, slice_direction)) * acos(cos_n), -HALF_PI, HALF_PI);
        // Each side's horizon starts at the surface's own tangent: index 0
        // along +direction, index 1 along -direction.
        var tangent_cos = array<f32, 2>(cos(n + HALF_PI), cos(n - HALF_PI));
        var horizon_cos = tangent_cos;
        for (var step_index = 0u; step_index < STEPS; step_index++) {
            let t = (f32(step_index) + jitter) / f32(STEPS);
            // Squared: more samples near the pixel, where occlusion is strongest.
            let offset = mix(1.0, radius_pixels, t * t);
            for (var side = 0u; side < 2u; side++) {
                let side_sign = select(-1.0, 1.0, side == 0u);
                let coord = vec2<i32>(floor(centre + direction * offset * side_sign));
                if any(coord < vec2<i32>(0)) || any(coord >= full_size) {
                    continue;
                }
                let sample_depth = textureLoad(depth_texture, coord, 0);
                let delta = world_at(vec2<f32>(coord) + 0.5, sample_depth) - position;
                let delta_length = length(delta);
                if delta_length < 1e-5 {
                    continue;
                }
                let fade = clamp((delta_length - falloff_start) / (radius - falloff_start), 0.0, 1.0);
                let sample_cos = mix(dot(delta, view) / delta_length, tangent_cos[side], fade);
                horizon_cos[side] = max(horizon_cos[side], sample_cos);
            }
        }
        let h_positive = n + min(acos(horizon_cos[0]) - n, HALF_PI);
        let h_negative = n + max(-acos(horizon_cos[1]) - n, -HALF_PI);
        let slice_visibility = side_visibility(h_positive, n) + side_visibility(h_negative, n);
        // Weighted by the projected normal's length, as the paper's outer
        // integral over slices weighs them.
        visibility += projected_length * slice_visibility;
        weight += projected_length * open_slice(n);
    }
    var open = 1.0;
    if weight > 0.0 {
        open = clamp(visibility / weight, 0.0, 1.0);
    }
    let strength = params.settings.x;
    textureStore(raw_output, id.xy, vec4<f32>(mix(1.0, open, strength), distance, 0.0, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn denoise_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(denoised_output);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let centre = textureLoad(raw_input, vec2<i32>(id.xy), 0);
    if centre.g >= SKY_DISTANCE * 0.5 {
        textureStore(denoised_output, id.xy, centre);
        return;
    }
    let limit = vec2<i32>(size) - 1;
    let tolerance = centre.g * DEPTH_TOLERANCE;
    var sum = 0.0;
    var weight = 0.0;
    for (var y = -DENOISE_RADIUS; y <= DENOISE_RADIUS; y++) {
        for (var x = -DENOISE_RADIUS; x <= DENOISE_RADIUS; x++) {
            let coord = clamp(vec2<i32>(id.xy) + vec2<i32>(x, y), vec2<i32>(0), limit);
            let neighbour = textureLoad(raw_input, coord, 0);
            let difference = (neighbour.g - centre.g) / tolerance;
            let w = exp(-difference * difference);
            sum += neighbour.r * w;
            weight += w;
        }
    }
    // The centre always weighs 1, so the weight is never zero.
    textureStore(denoised_output, id.xy, vec4<f32>(sum / weight, centre.g, 0.0, 1.0));
}
