// The harness inserts brdf.wgsl before this common source. Both prototypes
// use the production PBR BRDF and the exact same bounded light attenuation.
struct Params { extent: vec2<u32>, light_count: u32, unused: u32, grid: vec4<u32> }
struct Light { position_radius: vec4<f32>, color_intensity: vec4<f32> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> lights: array<Light>;
@group(0) @binding(2) var<storage, read> lists: array<u32>;
fn cluster_base(world: vec3<f32>) -> u32 {
    let unit = clamp(world * vec3<f32>(0.5,0.5,1.0) + vec3<f32>(0.5,0.5,0.0), vec3<f32>(0.0), vec3<f32>(0.999999));
    let cell = vec3<u32>(unit * vec3<f32>(params.grid.xyz));
    return (cell.x + params.grid.x * (cell.y + params.grid.y * cell.z)) * (params.light_count + 1u);
}
fn shade(world: vec3<f32>, normal: vec3<f32>, albedo: vec3<f32>, metallic: f32, roughness: f32, clustered: bool) -> vec3<f32> {
    let base = cluster_base(world);
    var count = params.light_count;
    if clustered { count = lists[base]; }
    let view = normalize(vec3<f32>(0.0,0.0,-2.0) - world);
    var color = ambient_flat(albedo, metallic, vec3<f32>(0.025), 1.0);
    for (var j = 0u; j < count; j++) {
        var index = j;
        if clustered { index = lists[base + j + 1u]; }
        let light = lights[index];
        let delta = light.position_radius.xyz - world;
        let distance = length(delta);
        if distance > 0.00001 && distance < light.position_radius.w {
            let fade = pow(1.0 - distance / light.position_radius.w, 2.0);
            color += brdf_direct(normal, view, delta / distance, albedo, metallic, roughness, light.color_intensity.rgb * light.color_intensity.w * fade);
        }
    }
    return min(color, vec3<f32>(60000.0));
}
