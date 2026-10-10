// light_cluster_build.wgsl — clustered light lists (C5), each frame with
// local lights: first every cluster's bounds, then the lights touching it.
//
// The view frustum is cut into a grid of clusters: tiles across the screen,
// and slices in view depth spaced exponentially, so near clusters stay
// small. Each cluster's bounds are the world-space box around its eight
// corners, which is conservative: a light that touches the box may miss the
// cluster, never the reverse.
//
// Bindings, one group; each entry point uses only its own:
//
//   @group(0) @binding(10) ClusterParams   uniform (from clustered_lights.wgsl)
//   @group(0) @binding(11) lights          storage, read (from clustered_lights.wgsl)
//   @group(0) @binding(2)  bounds          storage, read-write: two vec4 per cluster
//   @group(0) @binding(3)  grid            storage, read-write: per cluster a count, then indices
//   @group(0) @binding(4)  overflow        storage, read-write: lights dropped this frame

#include "clustered_lights.wgsl"

@group(0) @binding(2) var<storage, read_write> bounds: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> grid_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> overflow: atomic<u32>;

// The world position at screen `ndc` and view depth `depth` along the
// camera's forward axis, via the corner rays of the frustum.
fn frustum_point(ndc: vec2<f32>, depth: f32) -> vec3<f32> {
    let near_point = clusters.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far_point = clusters.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let near_world = near_point.xyz / near_point.w;
    let far_world = far_point.xyz / far_point.w;
    let forward = clusters.forward_far.xyz;
    let near_depth = dot(near_world - clusters.eye_near.xyz, forward);
    let far_depth = dot(far_world - clusters.eye_near.xyz, forward);
    let t = (depth - near_depth) / (far_depth - near_depth);
    return mix(near_world, far_world, t);
}

@compute @workgroup_size(64)
fn bounds_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let grid = clusters.grid.xyz;
    let count = grid.x * grid.y * grid.z;
    if id.x >= count {
        return;
    }
    let cell = vec3<u32>(id.x % grid.x, (id.x / grid.x) % grid.y, id.x / (grid.x * grid.y));
    let low_ndc = vec2<f32>(cell.xy) / vec2<f32>(grid.xy) * 2.0 - 1.0;
    let high_ndc = vec2<f32>(cell.xy + 1u) / vec2<f32>(grid.xy) * 2.0 - 1.0;
    let near_depth = slice_depth(cell.z);
    let far_depth = slice_depth(cell.z + 1u);
    var low = vec3<f32>(1e30);
    var high = vec3<f32>(-1e30);
    for (var corner = 0u; corner < 8u; corner++) {
        let ndc = vec2<f32>(
            select(low_ndc.x, high_ndc.x, (corner & 1u) != 0u),
            select(low_ndc.y, high_ndc.y, (corner & 2u) != 0u),
        );
        let depth = select(near_depth, far_depth, (corner & 4u) != 0u);
        let point = frustum_point(ndc, depth);
        low = min(low, point);
        high = max(high, point);
    }
    bounds[id.x * 2u] = vec4<f32>(low, 0.0);
    bounds[id.x * 2u + 1u] = vec4<f32>(high, 0.0);
}

// Whether a spot's cone certainly misses the sphere at `centre` (Wronski,
// "Cull that cone", 2017). Cones wider than a hemisphere are never culled by
// angle. Mirror of `cone_misses_sphere` in light_clusters.rs.
fn cone_misses_sphere(light: GpuLight, centre: vec3<f32>, radius: f32) -> bool {
    let cos_outer = light.cone.y;
    if cos_outer <= 0.0 {
        return false;
    }
    let offset = centre - light.position_range.xyz;
    let along = dot(offset, light.direction_spot.xyz);
    let across = sqrt(max(dot(offset, offset) - along * along, 0.0));
    let closest = cos_outer * across - along * light.cone.z;
    return closest > radius || along > radius + light.position_range.w || along < -radius;
}

@compute @workgroup_size(64)
fn cull_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let grid = clusters.grid.xyz;
    let count = grid.x * grid.y * grid.z;
    if id.x >= count {
        return;
    }
    let low = bounds[id.x * 2u].xyz;
    let high = bounds[id.x * 2u + 1u].xyz;
    // The box's bounding sphere, for the spot cone test.
    let box_centre = (low + high) * 0.5;
    let box_radius = length(high - low) * 0.5;
    let cap = clusters.counts.y;
    let base = id.x * (cap + 1u);
    var found = 0u;
    for (var index = 0u; index < clusters.counts.x; index++) {
        let light = lights[index];
        let centre = light.position_range.xyz;
        let range = light.position_range.w;
        let outside = max(low - centre, vec3<f32>(0.0)) + max(centre - high, vec3<f32>(0.0));
        if dot(outside, outside) > range * range {
            continue;
        }
        if light.direction_spot.w > 0.5 && cone_misses_sphere(light, box_centre, box_radius) {
            continue;
        }
        if found < cap {
            grid_out[base + 1u + found] = index;
            found++;
        } else {
            atomicAdd(&overflow, 1u);
        }
    }
    grid_out[base] = found;
}
