//! CPU references for the atmosphere LUTs, in `f64`. Test-only.
//!
//! Transmittance is integrated far more finely than the GPU does, so it is
//! the ground truth. Multiple scattering follows the GPU algorithm step for
//! step (same directions, same steps), but with exact transmittance where the
//! GPU reads its LUT, so a disagreement is a shader bug or LUT error, not a
//! difference in method.

use super::{AtmosphereParams, distance_to_ground, distance_to_top, medium};

type Rgb = [f64; 3];

/// Transmittance from radius `r` along zenith cosine `mu` to the top of the
/// atmosphere, by midpoint integration of the optical depth.
pub(crate) fn transmittance(params: &AtmosphereParams, r: f64, mu: f64, steps: u32) -> Rgb {
    let length = distance_to_top(params, r, mu);
    let dt = length / f64::from(steps);
    let mut depth = [0.0; 3];
    for step in 0..steps {
        let t = (f64::from(step) + 0.5) * dt;
        let height = (r * r + t * t + 2.0 * r * mu * t).sqrt();
        let extinction = medium(params, height - f64::from(params.bottom_radius)).extinction;
        for channel in 0..3 {
            depth[channel] += extinction[channel] * dt;
        }
    }
    depth.map(|value| (-value).exp())
}

/// Directions per side of the sphere grid the multiple-scattering LUT
/// integrates over (64 in all), and ray-march steps per direction. The GPU
/// uses the same.
pub(crate) const MS_DIRECTIONS_PER_SIDE: u32 = 8;
pub(crate) const MS_STEPS: u32 = 20;

/// Fine transmittance toward the sun, standing in for the GPU's LUT lookup.
/// Zero when the planet is in the way.
fn sun_transmittance(params: &AtmosphereParams, r: f64, mu: f64) -> Rgb {
    let bottom = f64::from(params.bottom_radius);
    let horizon = -(1.0 - (bottom / r).powi(2)).max(0.0).sqrt();
    if mu < horizon {
        return [0.0; 3];
    }
    transmittance(params, r, mu, 512)
}

fn length3(point: [f64; 3]) -> f64 {
    (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt()
}

/// The multiple-scattering transfer Psi at radius `r` with the sun at zenith
/// cosine `mu_sun`: second-order luminance for unit sun illuminance, summed
/// over all orders by the geometric series 1 / (1 - f_ms).
pub(crate) fn multiple_scattering(params: &AtmosphereParams, r: f64, mu_sun: f64) -> Rgb {
    let bottom = f64::from(params.bottom_radius);
    let isotropic = 1.0 / (4.0 * std::f64::consts::PI);
    let sun = [(1.0 - mu_sun * mu_sun).max(0.0).sqrt(), mu_sun, 0.0];
    let side = MS_DIRECTIONS_PER_SIDE;
    let mut luminance = [0.0; 3];
    let mut transfer = [0.0; 3];
    for i in 0..side {
        for j in 0..side {
            let cos_theta = 1.0 - 2.0 * (f64::from(i) + 0.5) / f64::from(side);
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
            let phi = 2.0 * std::f64::consts::PI * (f64::from(j) + 0.5) / f64::from(side);
            let direction = [sin_theta * phi.cos(), cos_theta, sin_theta * phi.sin()];
            let ground = distance_to_ground(params, r, direction[1]);
            let length = ground.unwrap_or_else(|| distance_to_top(params, r, direction[1]));
            let dt = length / f64::from(MS_STEPS);
            let mut throughput = [1.0; 3];
            for step in 0..MS_STEPS {
                let t = (f64::from(step) + 0.5) * dt;
                let point = [direction[0] * t, r + direction[1] * t, direction[2] * t];
                let height = length3(point);
                let mu_sun_here = (point[0] * sun[0] + point[1] * sun[1] + point[2] * sun[2]) / height;
                let local = medium(params, height - bottom);
                let sunlight = sun_transmittance(params, height, mu_sun_here);
                for channel in 0..3 {
                    let scattering = local.rayleigh_scattering[channel] + local.mie_scattering;
                    let extinction = local.extinction[channel];
                    let step_transmittance = (-extinction * dt).exp();
                    // Energy-conserving: the scattered light integrated over
                    // the step, not its midpoint value times dt.
                    let fraction = if extinction > 0.0 {
                        scattering * (1.0 - step_transmittance) / extinction
                    } else {
                        scattering * dt
                    };
                    luminance[channel] += throughput[channel] * sunlight[channel] * isotropic * fraction;
                    transfer[channel] += throughput[channel] * fraction;
                    throughput[channel] *= step_transmittance;
                }
            }
            if let Some(distance) = ground {
                let point = [
                    direction[0] * distance,
                    r + direction[1] * distance,
                    direction[2] * distance,
                ];
                let mu_sun_ground =
                    (point[0] * sun[0] + point[1] * sun[1] + point[2] * sun[2]) / length3(point);
                let sunlight = sun_transmittance(params, bottom, mu_sun_ground);
                for channel in 0..3 {
                    luminance[channel] += throughput[channel]
                        * sunlight[channel]
                        * mu_sun_ground.max(0.0)
                        * f64::from(params.ground_albedo[channel])
                        / std::f64::consts::PI;
                }
            }
        }
    }
    // Averages over the sphere: solid angle 4 pi / N per direction times the
    // isotropic phase 1 / (4 pi) leaves 1 / N.
    let count = f64::from(side * side);
    std::array::from_fn(|channel| (luminance[channel] / count) / (1.0 - transfer[channel] / count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looking_straight_up_from_the_ground_matches_the_vertical_optical_depth() {
        // Rayleigh's vertical optical depth is beta * scale height, plus Mie
        // and ozone; for blue: 33.1e-3 * 8 = 0.265.
        let params = AtmosphereParams::default();
        let blue = transmittance(&params, 6360.0, 1.0, 4096)[2];
        let mie: f64 = (3.996e-3 + 4.40e-3) * 1.2;
        let ozone = 0.085e-3 * 15.0;
        let expected = (-(33.1e-3 * 8.0 + mie + ozone)).exp();
        assert!((blue - expected).abs() < 0.01, "{blue} against {expected}");
    }

    #[test]
    fn transmittance_falls_toward_the_horizon_and_rises_with_height() {
        let params = AtmosphereParams::default();
        let up = transmittance(&params, 6360.0, 1.0, 1024)[1];
        let slant = transmittance(&params, 6360.0, 0.2, 1024)[1];
        let horizon = transmittance(&params, 6360.0, 0.0, 1024)[1];
        let high = transmittance(&params, 6380.0, 0.2, 1024)[1];
        assert!(up > slant && slant > horizon && high > slant);
    }

    #[test]
    fn multiple_scattering_is_brighter_in_daylight_and_bluer_low_down() {
        let params = AtmosphereParams::default();
        let noon = multiple_scattering(&params, 6361.0, 1.0);
        let night = multiple_scattering(&params, 6361.0, -0.5);
        assert!(noon.iter().all(|&value| value > 0.0 && value.is_finite()));
        assert!(noon[2] > night[2] * 10.0, "noon {noon:?} against night {night:?}");
        assert!(noon[2] > noon[0], "blue scatters more than red: {noon:?}");
    }
}
