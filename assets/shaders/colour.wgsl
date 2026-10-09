// colour.wgsl — shared colour maths. Included with `#include "colour.wgsl"`,
// never compiled on its own.
//
// Every function here has a CPU mirror in crates/renderer/src/post/tonemap.rs,
// and a GPU test holds the two equal. Change both together.

// The largest finite half-float: what an Rgba16Float target can hold.
const HALF_MAX: f32 = 65504.0;

// Scene light the curves can take: NaN becomes 0, and negatives and anything
// beyond the half-float range are clamped. A lit value past 65504 is stored as
// infinity, and a curve would turn that into NaN and then black or garbage on
// exactly the brightest pixels.
fn finite_scene(colour: vec3<f32>) -> vec3<f32> {
    let not_nan = select(vec3<f32>(0.0), colour, colour == colour);
    return clamp(not_nan, vec3<f32>(0.0), vec3<f32>(HALF_MAX));
}

// ACES, as fitted by Stephen Hill (BakingLab): the sRGB-to-ACES input matrix,
// a rational fit of the reference rendering and output transforms, and the
// output matrix back to linear sRGB. Columns, so `M * v` matches the original
// row-major `mul(M, v)`.
const ACES_INPUT = mat3x3<f32>(
    vec3<f32>(0.59719, 0.07600, 0.02840),
    vec3<f32>(0.35458, 0.90834, 0.13383),
    vec3<f32>(0.04823, 0.01566, 0.83777),
);
const ACES_OUTPUT = mat3x3<f32>(
    vec3<f32>(1.60475, -0.10208, -0.00327),
    vec3<f32>(-0.53108, 1.10813, -0.07276),
    vec3<f32>(-0.07367, -0.00605, 1.07602),
);

fn aces_rrt_odt_fit(v: vec3<f32>) -> vec3<f32> {
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

// Scene-referred linear in, display-referred linear in 0..1 out.
fn aces_fitted(colour: vec3<f32>) -> vec3<f32> {
    let fitted = aces_rrt_odt_fit(ACES_INPUT * colour);
    return clamp(ACES_OUTPUT * fitted, vec3<f32>(0.0), vec3<f32>(1.0));
}

// AgX, after Troy Sobotka's design, with the polynomial sigmoid fit published
// by Benjamin Wrensch: an inset matrix, a log2 encoding over a fixed range of
// stops, the contrast curve, and the outset matrix back.
const AGX_INSET = mat3x3<f32>(
    vec3<f32>(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
    vec3<f32>(0.0784335999999992, 0.878468636469772, 0.0784336),
    vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104),
);
const AGX_OUTSET = mat3x3<f32>(
    vec3<f32>(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
    vec3<f32>(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
    vec3<f32>(-0.0990297440797205, -0.0989611768448433, 1.15107367264116),
);
const AGX_MIN_EV: f32 = -12.47393;
const AGX_MAX_EV: f32 = 4.026069;

fn agx_contrast(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

// Scene-referred linear in, display-referred linear in 0..1 out.
fn agx(colour: vec3<f32>) -> vec3<f32> {
    let inset = AGX_INSET * max(colour, vec3<f32>(1e-10));
    let stops = clamp(log2(inset), vec3<f32>(AGX_MIN_EV), vec3<f32>(AGX_MAX_EV));
    let encoded = (stops - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV);
    let outset = AGX_OUTSET * agx_contrast(encoded);
    return clamp(pow(max(outset, vec3<f32>(0.0)), vec3<f32>(2.2)), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Lift, gamma and gain on sRGB-encoded display values in 0..1, the space
// grading controls are defined in elsewhere: lift raises the blacks, gain
// scales the whites, and the inverse gamma bends the mid tones.
fn grade(display: vec3<f32>, lift: vec3<f32>, inverse_gamma: vec3<f32>, gain: vec3<f32>) -> vec3<f32> {
    let lifted = gain * (display + lift * (vec3<f32>(1.0) - display));
    return clamp(pow(max(lifted, vec3<f32>(0.0)), inverse_gamma), vec3<f32>(0.0), vec3<f32>(1.0));
}

// The inverse of `linear_to_srgb`.
fn srgb_to_linear(colour: vec3<f32>) -> vec3<f32> {
    let low = colour / 12.92;
    let high = pow((colour + 0.055) / 1.055, vec3<f32>(2.4));
    return select(high, low, colour <= vec3<f32>(0.04045));
}

// The sRGB transfer function, for targets the GPU does not encode itself.
fn linear_to_srgb(colour: vec3<f32>) -> vec3<f32> {
    let low = colour * 12.92;
    let high = 1.055 * pow(colour, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, colour <= vec3<f32>(0.0031308));
}
