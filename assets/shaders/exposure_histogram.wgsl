// exposure_histogram.wgsl — automatic exposure, in two compute entry points.
//
// `build` sorts every HDR pixel into one of 256 bins by log2 luminance;
// `average` turns the histogram into a mean luminance, adapts the running
// value towards it, and clears the histogram for the next frame. The adapted
// luminance never leaves the GPU: the renderer copies it into the tonemap
// pass's uniform within the same submission.
//
// Bin 0 holds black pixels (below the measured range or with no light) and
// is left out of the average, so a dark sky does not drag the exposure up.
// Bins 1 to 255 split the measured range evenly in stops.
//
// Bind group layout (both entry points):
//
//   @group(0) @binding(0)  hdr        texture_2d<f32>, the lit scene
//   @group(0) @binding(1)  histogram  storage, 256 x u32
//   @group(0) @binding(2)  state      storage, the adapted luminance
//   @group(0) @binding(3)  params     uniform, per frame
//
// CPU mirrors of the bin mapping and the average are in
// crates/renderer/src/post/exposure.rs.

const BINS: u32 = 256u;
const MEASURED_BINS: f32 = 255.0;

struct Params {
    min_log2: f32,          // log2 luminance of the bottom of bin 1
    range_log2: f32,        // stops covered by bins 1 to 255
    brighten_blend: f32,    // share of the gap closed this frame, 0..1
    darken_blend: f32,
}

struct State {
    luminance: f32,         // adapted average scene luminance
    initialised: u32,       // 0 until the first frame has been measured
    _padding: vec2<u32>,
}

@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>, 256>;
@group(0) @binding(2) var<storage, read_write> state: State;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> local_bins: array<atomic<u32>, 256>;
var<workgroup> weighted: array<f32, 256>;
var<workgroup> counted: array<f32, 256>;

fn luminance_bin(colour: vec3<f32>) -> u32 {
    let luminance = dot(colour, vec3<f32>(0.2126, 0.7152, 0.0722));
    if !(luminance > 0.0) {
        return 0u;
    }
    let position = (log2(luminance) - params.min_log2) / params.range_log2;
    if position < 0.0 {
        return 0u;
    }
    return 1u + u32(clamp(position, 0.0, 1.0) * (MEASURED_BINS - 1.0));
}

@compute @workgroup_size(16, 16)
fn build(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    atomicStore(&local_bins[local], 0u);
    workgroupBarrier();

    let size = textureDimensions(hdr);
    if id.x < size.x && id.y < size.y {
        let colour = textureLoad(hdr, vec2<i32>(id.xy), 0).rgb;
        atomicAdd(&local_bins[luminance_bin(colour)], 1u);
    }
    workgroupBarrier();

    // One global atomic per bin per workgroup instead of one per pixel.
    atomicAdd(&histogram[local], atomicLoad(&local_bins[local]));
}

@compute @workgroup_size(256)
fn average(@builtin(local_invocation_index) local: u32) {
    let count = atomicLoad(&histogram[local]);
    atomicStore(&histogram[local], 0u);
    // Bin 0 is black and does not count.
    let measured = select(f32(count), 0.0, local == 0u);
    weighted[local] = measured * f32(local);
    counted[local] = measured;
    workgroupBarrier();

    for (var stride = BINS / 2u; stride > 0u; stride = stride / 2u) {
        if local < stride {
            weighted[local] += weighted[local + stride];
            counted[local] += counted[local + stride];
        }
        workgroupBarrier();
    }

    if local == 0u {
        if counted[0] > 0.0 {
            let mean_bin = weighted[0] / counted[0];
            let position = (mean_bin - 1.0) / (MEASURED_BINS - 1.0);
            let measured_luminance = exp2(position * params.range_log2 + params.min_log2);
            var adapted = measured_luminance;
            if state.initialised != 0u {
                let previous = state.luminance;
                let blend = select(params.darken_blend, params.brighten_blend, measured_luminance > previous);
                adapted = previous + (measured_luminance - previous) * blend;
            }
            state.luminance = adapted;
            state.initialised = 1u;
        }
    }
}
