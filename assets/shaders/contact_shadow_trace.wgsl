// contact_shadow_trace.wgsl — screen-space contact shadows (C4).
//
// Each full-resolution pixel marches a short way from its surface toward the
// directional light, projecting every step onto the depth buffer. If a step
// lands behind the surface drawn there, by less than that surface's assumed
// thickness, something stands between the pixel and the light: it is in
// contact shadow. The lit shaders take the darker of this and the cascades,
// so it only adds the small and near shadows the cascades are too coarse
// to hold.
//
// The march starts a pixel's width off the surface along its normal (from
// the prepass's normal target). A step then clears its own surface by more
// than the half-pixel error of reading depth at the nearest pixel centre,
// so a plane never shadows itself, even seen at a grazing angle.
//
//   @group(0) @binding(0)  ContactParams
//   @group(0) @binding(1)  depth     texture_depth_2d
//   @group(0) @binding(2)  normals   texture_2d<f32>, n * 0.5 + 0.5
//   @group(0) @binding(3)  result    storage, rgba8unorm: r = 1 lit, 0 shadowed

struct ContactParams {
    view_proj: mat4x4<f32>,     // world to clip
    inv_view_proj: mat4x4<f32>, // clip to world
    light: vec4<f32>,           // xyz = unit direction to the light, w = march length in metres
    thickness: vec4<f32>,       // x = assumed surface thickness in metres
    size: vec4<f32>,            // width, height, 1 / width, 1 / height
}

@group(0) @binding(0) var<uniform> params: ContactParams;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var normal_texture: texture_2d<f32>;
@group(0) @binding(3) var result: texture_storage_2d<rgba8unorm, write>;

// Steps along the march. Evenly spaced and the same for every pixel: with no
// temporal filter yet (E2) a jittered start would show as noise.
const STEPS: u32 = 16u;
// How far off the surface the march starts, in pixel widths there.
const START_OFFSET_PIXELS: f32 = 1.0;

fn clip_to_world(ndc: vec3<f32>) -> vec4<f32> {
    return params.inv_view_proj * vec4<f32>(ndc, 1.0);
}

fn ndc_at(pixel: vec2<f32>, depth: f32) -> vec3<f32> {
    let uv = pixel * params.size.zw;
    return vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth);
}

fn world_at(pixel: vec2<f32>, depth: f32) -> vec3<f32> {
    let world = clip_to_world(ndc_at(pixel, depth));
    return world.xyz / world.w;
}

// The view depth (clip w) of the surface drawn at `pixel`. Unprojecting a
// clip-space point divided by its w gives the world point divided by that w,
// so the result's w is its reciprocal.
fn view_depth_at(pixel: vec2<f32>, depth: f32) -> f32 {
    return 1.0 / clip_to_world(ndc_at(pixel, depth)).w;
}

@compute @workgroup_size(8, 8, 1)
fn contact_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<i32>(textureDimensions(result));
    if i32(id.x) >= size.x || i32(id.y) >= size.y {
        return;
    }
    let pixel = vec2<i32>(id.xy);
    let depth = textureLoad(depth_texture, pixel, 0);
    if depth >= 1.0 {
        textureStore(result, pixel, vec4<f32>(1.0)); // Sky.
        return;
    }
    let encoded = textureLoad(normal_texture, pixel, 0).xyz * 2.0 - 1.0;
    let to_light = params.light.xyz;
    // A surface facing away from the light, which its own shading already
    // leaves unlit, or no usable normal to start the march from.
    if dot(encoded, encoded) < 1e-6 || dot(encoded, to_light) <= 0.0 {
        textureStore(result, pixel, vec4<f32>(1.0));
        return;
    }
    let normal = normalize(encoded);

    let centre = vec2<f32>(pixel) + 0.5;
    let position = world_at(centre, depth);
    let pixel_world = length(world_at(centre + vec2<f32>(1.0, 0.0), depth) - position);
    let start = position + normal * pixel_world * START_OFFSET_PIXELS;
    let march = params.light.w;
    let thickness = params.thickness.x;

    var lit = 1.0;
    for (var index = 1u; index <= STEPS; index++) {
        let point = start + to_light * (march * f32(index) / f32(STEPS));
        let clip = params.view_proj * vec4<f32>(point, 1.0);
        if clip.w <= 0.0 {
            break; // Behind the camera.
        }
        let ndc = clip.xy / clip.w;
        if any(abs(ndc) >= vec2<f32>(1.0)) {
            break; // Off screen: nothing more to see.
        }
        let coord = vec2<i32>(vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * params.size.xy);
        let scene_depth = textureLoad(depth_texture, coord, 0);
        if scene_depth >= 1.0 {
            continue; // Sky casts nothing.
        }
        let behind = clip.w - view_depth_at(vec2<f32>(coord) + 0.5, scene_depth);
        if behind > 0.0 && behind < thickness {
            lit = 0.0;
            break;
        }
    }
    textureStore(result, pixel, vec4<f32>(lit, lit, lit, 1.0));
}
