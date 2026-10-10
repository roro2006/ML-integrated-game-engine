// Three MRTs: linear HDR emission, packed normal/metallic, albedo/roughness.
// AO is 1 in this asset-free scene. A separate base-colour/ORM target and
// velocity remain explicit follow-up measurements for production materials.
struct Vertex { @builtin(position) clip: vec4<f32>, @location(0) color: vec3<f32> }
struct GBuffer { @location(0) emission: vec4<f32>, @location(1) normal_metallic: vec4<f32>, @location(2) albedo_roughness: vec4<f32> }
@vertex fn vs_main(@location(0) position: vec3<f32>, @location(3) color: vec3<f32>) -> Vertex {
    return Vertex(vec4<f32>(position, 1.0), color);
}
@fragment fn fs_main(vertex: Vertex) -> GBuffer {
    return GBuffer(vec4<f32>(0.0), vec4<f32>(0.5,0.5,0.0,0.0), vec4<f32>(vertex.color,128.0/255.0));
}
