// B2 fixed orthographic benchmark. Position-only stage, the same 64-byte
// mesh buffer as both shading prototypes; no fragment stage.
@vertex fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(position, 1.0);
}
