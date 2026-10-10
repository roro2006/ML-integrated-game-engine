// One invocation per 3D cluster. Fixed orthographic frustum [-1,1]^2 x [0,1].
// Lists reserve light_count + 1 words per cluster, so no light can overflow.
struct Params { extent: vec2<u32>, light_count: u32, unused: u32, grid: vec4<u32> }
struct Light { position_radius: vec4<f32>, color_intensity: vec4<f32> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> lights: array<Light>;
@group(0) @binding(2) var<storage, read_write> lists: array<u32>;
@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let total = params.grid.x * params.grid.y * params.grid.z;
    if id.x >= total { return; }
    let cell = vec3<u32>(id.x % params.grid.x, (id.x / params.grid.x) % params.grid.y, id.x / (params.grid.x * params.grid.y));
    let size = vec3<f32>(2.0, 2.0, 1.0) / vec3<f32>(params.grid.xyz);
    let low = vec3<f32>(-1.0, -1.0, 0.0) + vec3<f32>(cell) * size;
    let high = low + size;
    let base = id.x * (params.light_count + 1u);
    var count = 0u;
    for (var i = 0u; i < params.light_count; i++) {
        let light = lights[i].position_radius;
        let delta = light.xyz - clamp(light.xyz, low, high);
        if dot(delta, delta) <= light.w * light.w {
            lists[base + 1u + count] = i;
            count++;
        }
    }
    lists[base] = count;
}
