@group(1) @binding(0) var emission: texture_2d<f32>;
@group(1) @binding(1) var normal_metallic: texture_2d<f32>;
@group(1) @binding(2) var albedo_roughness: texture_2d<f32>;
@group(1) @binding(3) var depth: texture_depth_2d;
@vertex fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let xy = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs_main(@builtin(position) pixel: vec4<f32>) -> @location(0) vec4<f32> {
    let at = vec2<i32>(pixel.xy);
    let z = textureLoad(depth, at, 0);
    if z >= 1.0 { return vec4<f32>(0.0,0.0,0.0,1.0); }
    let uv = pixel.xy / vec2<f32>(params.extent);
    let world = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z);
    let packed = textureLoad(normal_metallic, at, 0);
    let material = textureLoad(albedo_roughness, at, 0);
    let normal = normalize(packed.xyz * 2.0 - 1.0);
    return vec4<f32>(textureLoad(emission, at, 0).rgb + shade(world, normal, material.rgb, packed.a, material.a, true), 1.0);
}
