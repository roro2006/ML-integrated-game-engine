// atmosphere.wgsl — the atmosphere's medium and LUT parameterisations (D3).
// Included, never compiled alone. Every function takes the atmosphere as an
// argument, so the snippet validates on its own and an includer binds the
// uniform wherever its layout puts it. Mirrors crates/renderer/src/atmosphere,
// whose CPU references the GPU tests hold these to; change both together.
//
// Hillaire, "A Scalable and Production Ready Sky and Atmosphere Rendering
// Technique" (2020), with the transmittance parameterisation of Bruneton
// (2017). Distances in kilometres, coefficients per kilometre.

struct Atmosphere {
    rayleigh: vec4<f32>,          // scattering rgb, scale height
    mie: vec4<f32>,               // scattering, absorption, scale height, g
    ozone_absorption: vec4<f32>,  // rgb, unused
    ozone: vec4<f32>,             // tent centre, tent width, unused, unused
    ground_albedo: vec4<f32>,     // rgb, unused
    radii: vec4<f32>,             // bottom, top, unused, unused
}

struct Medium {
    rayleigh: vec3<f32>,
    mie: f32,
    scattering: vec3<f32>,
    extinction: vec3<f32>,
}

fn atmosphere_medium(atmosphere: Atmosphere, altitude: f32) -> Medium {
    let rayleigh_density = exp(-altitude / atmosphere.rayleigh.w);
    let mie_density = exp(-altitude / atmosphere.mie.z);
    let ozone_density = max(0.0, 1.0 - abs(altitude - atmosphere.ozone.x) / (atmosphere.ozone.y * 0.5));
    var medium: Medium;
    medium.rayleigh = atmosphere.rayleigh.rgb * rayleigh_density;
    medium.mie = atmosphere.mie.x * mie_density;
    medium.scattering = medium.rayleigh + vec3<f32>(medium.mie);
    medium.extinction = medium.scattering + vec3<f32>(atmosphere.mie.y * mie_density)
        + atmosphere.ozone_absorption.rgb * ozone_density;
    return medium;
}

fn distance_to_top(atmosphere: Atmosphere, r: f32, mu: f32) -> f32 {
    let top = atmosphere.radii.y;
    let discriminant = r * r * (mu * mu - 1.0) + top * top;
    return max(0.0, -r * mu + sqrt(max(discriminant, 0.0)));
}

// Distance to the ground, or a negative value when the ray misses it.
fn distance_to_ground(atmosphere: Atmosphere, r: f32, mu: f32) -> f32 {
    let bottom = atmosphere.radii.x;
    let discriminant = r * r * (mu * mu - 1.0) + bottom * bottom;
    if mu >= 0.0 || discriminant < 0.0 {
        return -1.0;
    }
    return max(0.0, -r * mu - sqrt(discriminant));
}

fn transmittance_r_mu(atmosphere: Atmosphere, uv: vec2<f32>) -> vec2<f32> {
    let bottom = atmosphere.radii.x;
    let top = atmosphere.radii.y;
    let horizon = sqrt(top * top - bottom * bottom);
    let rho = horizon * uv.y;
    let r = sqrt(rho * rho + bottom * bottom);
    let d_min = top - r;
    let d_max = rho + horizon;
    let d = d_min + uv.x * (d_max - d_min);
    var mu = 1.0;
    if d > 0.0 {
        mu = clamp((horizon * horizon - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
    }
    return vec2<f32>(r, mu);
}

fn transmittance_uv(atmosphere: Atmosphere, r: f32, mu: f32) -> vec2<f32> {
    let bottom = atmosphere.radii.x;
    let top = atmosphere.radii.y;
    let horizon = sqrt(top * top - bottom * bottom);
    let rho = sqrt(max(r * r - bottom * bottom, 0.0));
    let d = distance_to_top(atmosphere, r, mu);
    let d_min = top - r;
    let d_max = rho + horizon;
    return vec2<f32>((d - d_min) / (d_max - d_min), rho / horizon);
}
