override CLUSTERED: bool = true;
struct Vertex { @builtin(position) clip: vec4<f32>, @location(0) world: vec3<f32>, @location(1) color: vec3<f32> }
@vertex fn vs_main(@location(0) position: vec3<f32>, @location(3) color: vec3<f32>) -> Vertex {
    return Vertex(vec4<f32>(position, 1.0), position, color);
}
@fragment fn fs_main(vertex: Vertex) -> @location(0) vec4<f32> {
    return vec4<f32>(shade(vertex.world, vec3<f32>(0.0,0.0,-1.0), vertex.color, 0.0, 128.0/255.0, CLUSTERED), 1.0);
}
