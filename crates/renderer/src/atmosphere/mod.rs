//! Physically based atmosphere (D3): parameters, the lookup-table
//! parameterisations, and CPU references for the tables the GPU bakes.
//!
//! Implemented from Hillaire, "A Scalable and Production Ready Sky and
//! Atmosphere Rendering Technique" (EGSR 2020), with the transmittance
//! parameterisation from Bruneton, "Precomputed Atmospheric Scattering"
//! (2008, revised 2017). Clean-room from the papers. Distances are in
//! kilometres and coefficients per kilometre, as in the papers, so a
//! planet's radius and an atmosphere's thickness sit in one float's range.

pub(crate) mod luts;
#[cfg(test)]
mod reference;
pub(crate) mod sky;

/// Transmittance LUT size: view zenith (x) by height (y).
pub(crate) const TRANSMITTANCE_SIZE: (u32, u32) = (256, 64);
/// Multiple-scattering LUT size: sun zenith (x) by height (y).
pub(crate) const MULTI_SCATTERING_SIZE: (u32, u32) = (32, 32);

/// The atmosphere's composition. `Default` is Earth, from the paper.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct AtmosphereParams {
    pub(crate) bottom_radius: f32,
    pub(crate) top_radius: f32,
    pub(crate) rayleigh_scattering: [f32; 3],
    pub(crate) rayleigh_scale_height: f32,
    pub(crate) mie_scattering: f32,
    pub(crate) mie_absorption: f32,
    pub(crate) mie_scale_height: f32,
    /// Cornette-Shanks asymmetry.
    pub(crate) mie_g: f32,
    pub(crate) ozone_absorption: [f32; 3],
    /// Ozone sits in a tent profile around this height, this wide.
    pub(crate) ozone_centre: f32,
    pub(crate) ozone_width: f32,
    pub(crate) ground_albedo: [f32; 3],
}

impl Default for AtmosphereParams {
    fn default() -> Self {
        Self {
            bottom_radius: 6360.0,
            top_radius: 6460.0,
            rayleigh_scattering: [5.802e-3, 13.558e-3, 33.1e-3],
            rayleigh_scale_height: 8.0,
            mie_scattering: 3.996e-3,
            mie_absorption: 4.40e-3,
            mie_scale_height: 1.2,
            mie_g: 0.8,
            ozone_absorption: [0.650e-3, 1.881e-3, 0.085e-3],
            ozone_centre: 25.0,
            ozone_width: 30.0,
            ground_albedo: [0.3, 0.3, 0.3],
        }
    }
}

/// The uniform block `atmosphere.wgsl` declares, 96 bytes: six `vec4`s.
pub(crate) const PARAMS_SIZE: usize = 96;

impl AtmosphereParams {
    pub(crate) fn bytes(&self) -> [u8; PARAMS_SIZE] {
        let words: [f32; PARAMS_SIZE / 4] = [
            self.rayleigh_scattering[0],
            self.rayleigh_scattering[1],
            self.rayleigh_scattering[2],
            self.rayleigh_scale_height,
            self.mie_scattering,
            self.mie_absorption,
            self.mie_scale_height,
            self.mie_g,
            self.ozone_absorption[0],
            self.ozone_absorption[1],
            self.ozone_absorption[2],
            0.0,
            self.ozone_centre,
            self.ozone_width,
            0.0,
            0.0,
            self.ground_albedo[0],
            self.ground_albedo[1],
            self.ground_albedo[2],
            0.0,
            self.bottom_radius,
            self.top_radius,
            0.0,
            0.0,
        ];
        let mut out = [0_u8; PARAMS_SIZE];
        for (chunk, word) in out.chunks_exact_mut(4).zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        out
    }
}

// The CPU mirrors below exist for the GPU tests' references.
#[cfg(test)]
/// Maps a transmittance LUT coordinate in `0..1` to the height (radius from
/// the planet's centre) and view zenith cosine it stands for.
///
/// The mapping spends texels where transmittance changes fastest: near the
/// horizon, and low in the atmosphere.
pub(crate) fn transmittance_uv_to_r_mu(params: &AtmosphereParams, u: f64, v: f64) -> (f64, f64) {
    let (bottom, top) = (f64::from(params.bottom_radius), f64::from(params.top_radius));
    let horizon = (top * top - bottom * bottom).sqrt();
    let rho = horizon * v;
    let r = (rho * rho + bottom * bottom).sqrt();
    let d_min = top - r;
    let d_max = rho + horizon;
    let d = d_min + u * (d_max - d_min);
    let mu = if d == 0.0 {
        1.0
    } else {
        ((horizon * horizon - rho * rho - d * d) / (2.0 * r * d)).clamp(-1.0, 1.0)
    };
    (r, mu)
}

#[cfg(test)]
/// The inverse of [`transmittance_uv_to_r_mu`].
pub(crate) fn transmittance_r_mu_to_uv(params: &AtmosphereParams, r: f64, mu: f64) -> (f64, f64) {
    let (bottom, top) = (f64::from(params.bottom_radius), f64::from(params.top_radius));
    let horizon = (top * top - bottom * bottom).sqrt();
    let rho = (r * r - bottom * bottom).max(0.0).sqrt();
    let d = distance_to_top(params, r, mu);
    let d_min = top - r;
    let d_max = rho + horizon;
    ((d - d_min) / (d_max - d_min), rho / horizon)
}

/// Distance from radius `r` along a ray with zenith cosine `mu` to the top
/// of the atmosphere.
pub(crate) fn distance_to_top(params: &AtmosphereParams, r: f64, mu: f64) -> f64 {
    let top = f64::from(params.top_radius);
    let discriminant = r * r * (mu * mu - 1.0) + top * top;
    (-r * mu + discriminant.max(0.0).sqrt()).max(0.0)
}

#[cfg(test)]
/// Distance to the ground, or `None` when the ray misses it.
pub(crate) fn distance_to_ground(params: &AtmosphereParams, r: f64, mu: f64) -> Option<f64> {
    let bottom = f64::from(params.bottom_radius);
    let discriminant = r * r * (mu * mu - 1.0) + bottom * bottom;
    (mu < 0.0 && discriminant >= 0.0).then(|| (-r * mu - discriminant.sqrt()).max(0.0))
}

/// Scattering and extinction at `altitude` above the ground, per kilometre.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Medium {
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "scattering feeds only the CPU references")
    )]
    pub(crate) rayleigh_scattering: [f64; 3],
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "scattering feeds only the CPU references")
    )]
    pub(crate) mie_scattering: f64,
    pub(crate) extinction: [f64; 3],
}

pub(crate) fn medium(params: &AtmosphereParams, altitude: f64) -> Medium {
    let rayleigh_density = (-altitude / f64::from(params.rayleigh_scale_height)).exp();
    let mie_density = (-altitude / f64::from(params.mie_scale_height)).exp();
    let ozone_density = (1.0
        - (altitude - f64::from(params.ozone_centre)).abs() / (f64::from(params.ozone_width) / 2.0))
        .max(0.0);
    let rayleigh: [f64; 3] = params
        .rayleigh_scattering
        .map(|value| f64::from(value) * rayleigh_density);
    let mie_scattering = f64::from(params.mie_scattering) * mie_density;
    let mie_extinction = mie_scattering + f64::from(params.mie_absorption) * mie_density;
    let extinction = std::array::from_fn(|channel| {
        rayleigh[channel] + mie_extinction + f64::from(params.ozone_absorption[channel]) * ozone_density
    });
    Medium {
        rayleigh_scattering: rayleigh,
        mie_scattering,
        extinction,
    }
}

/// Steps for [`sunlight_transmittance`]: enough for a fraction of a percent
/// at the horizon, and cheap enough for once a frame on the CPU.
const SUNLIGHT_STEPS: u32 = 64;

/// The fraction of sunlight that reaches a point `altitude_metres` above the
/// ground from `to_sun`, per RGB channel, in the default (Earth) atmosphere.
/// Zero when the sun is below the planet's horizon.
pub fn sunlight_transmittance(altitude_metres: f32, to_sun: [f32; 3]) -> [f32; 3] {
    let params = AtmosphereParams::default();
    let Some(to_sun) = glam::Vec3::from(to_sun).try_normalize() else {
        return [0.0; 3];
    };
    let bottom = f64::from(params.bottom_radius);
    let r = bottom + f64::from((altitude_metres / 1000.0).max(0.001));
    let mu = f64::from(to_sun.y);
    let horizon = -(1.0 - (bottom / r).powi(2)).max(0.0).sqrt();
    if mu < horizon {
        return [0.0; 3];
    }
    let length = distance_to_top(&params, r, mu);
    let dt = length / f64::from(SUNLIGHT_STEPS);
    let mut depth = [0.0_f64; 3];
    for step in 0..SUNLIGHT_STEPS {
        let t = (f64::from(step) + 0.5) * dt;
        let height = (r * r + t * t + 2.0 * r * mu * t).sqrt();
        let extinction = medium(&params, height - bottom).extinction;
        for channel in 0..3 {
            depth[channel] += extinction[channel] * dt;
        }
    }
    depth.map(|value| (-value).exp() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_transmittance_mapping_round_trips() {
        let params = AtmosphereParams::default();
        for &(u, v) in &[(0.1, 0.2), (0.5, 0.5), (0.9, 0.05), (0.3, 0.95)] {
            let (r, mu) = transmittance_uv_to_r_mu(&params, u, v);
            let (u2, v2) = transmittance_r_mu_to_uv(&params, r, mu);
            assert!(
                (u - u2).abs() < 1e-9 && (v - v2).abs() < 1e-9,
                "({u}, {v}) -> ({u2}, {v2})"
            );
        }
    }

    #[test]
    fn the_mapping_covers_ground_to_top_and_the_horizon() {
        let params = AtmosphereParams::default();
        let (r, mu) = transmittance_uv_to_r_mu(&params, 0.0, 0.0);
        assert!(
            (r - 6360.0).abs() < 1e-9 && (mu - 1.0).abs() < 1e-9,
            "u 0 at the ground looks straight up"
        );
        let (r, mu) = transmittance_uv_to_r_mu(&params, 1.0, 0.0);
        assert!(
            (r - 6360.0).abs() < 1e-9 && mu.abs() < 1e-9,
            "u 1 at the ground is the horizon"
        );
        let (r, _) = transmittance_uv_to_r_mu(&params, 0.5, 1.0);
        assert!((r - 6460.0).abs() < 1e-6, "v 1 is the top");
    }

    #[test]
    fn ozone_sits_in_a_tent_around_its_centre() {
        let pure_ozone = AtmosphereParams {
            rayleigh_scattering: [0.0; 3],
            mie_scattering: 0.0,
            mie_absorption: 0.0,
            ..AtmosphereParams::default()
        };
        let at = |altitude| medium(&pure_ozone, altitude).extinction[1];
        assert!((at(25.0) - 1.881e-3).abs() < 1e-9);
        assert!((at(17.5) - at(32.5)).abs() < 1e-12, "symmetric");
        assert_eq!(at(9.0), 0.0);
        assert_eq!(at(41.0), 0.0);
    }

    #[test]
    fn air_thins_with_height() {
        let params = AtmosphereParams::default();
        assert!(medium(&params, 0.0).extinction[2] > medium(&params, 8.0).extinction[2]);
    }

    #[test]
    fn sweeping_down_to_the_horizon_moves_smoothly_across_the_table() {
        // At several heights, the view zenith runs from straight up to just
        // above the planet's horizon; u must rise steadily with no jumps.
        let params = AtmosphereParams::default();
        let bottom = f64::from(params.bottom_radius);
        for r in [6360.001, 6365.0, 6400.0, 6459.0] {
            let horizon = -(1.0 - (bottom / r).powi(2)).max(0.0).sqrt();
            let steps = 2000;
            let mut previous = transmittance_r_mu_to_uv(&params, r, 1.0);
            for step in 1..=steps {
                let mu = 1.0 + (horizon + 1e-6 - 1.0) * f64::from(step) / f64::from(steps);
                let uv = transmittance_r_mu_to_uv(&params, r, mu);
                assert!(uv.0 >= previous.0 - 1e-12, "u falls at r {r}, mu {mu}");
                assert!(uv.0 - previous.0 < 0.01, "u jumps at r {r}, mu {mu}");
                assert!((uv.1 - previous.1).abs() < 1e-12, "v holds the height");
                assert!((0.0..=1.0).contains(&uv.0) && (0.0..=1.0).contains(&uv.1));
                previous = uv;
            }
            assert!(
                previous.0 > 0.99,
                "the horizon is the far edge of the table at r {r}"
            );
        }
    }

    #[test]
    fn sunlight_reddens_toward_the_horizon_and_vanishes_below_it() {
        let noon = sunlight_transmittance(0.0, [0.0, 1.0, 0.0]);
        let low = sunlight_transmittance(0.0, [1.0, 0.03, 0.0]);
        assert!(noon.iter().all(|&value| value > 0.6 && value < 1.0), "{noon:?}");
        assert!(low[0] > low[2] * 3.0, "a low sun is red: {low:?}");
        assert_eq!(sunlight_transmittance(0.0, [1.0, -0.2, 0.0]), [0.0; 3]);
        let fine = reference::transmittance(&AtmosphereParams::default(), 6360.001, 0.5, 4096);
        let coarse = sunlight_transmittance(1.0, [0.866, 0.5, 0.0]);
        for channel in 0..3 {
            assert!((f64::from(coarse[channel]) - fine[channel]).abs() / fine[channel] < 0.005);
        }
    }

    #[test]
    fn rays_find_the_top_and_the_ground() {
        let params = AtmosphereParams::default();
        assert!((distance_to_top(&params, 6360.0, 1.0) - 100.0).abs() < 1e-9);
        assert_eq!(distance_to_ground(&params, 6361.0, 1.0), None);
        let down = distance_to_ground(&params, 6361.0, -1.0).expect("straight down hits");
        assert!((down - 1.0).abs() < 1e-9);
    }

    #[test]
    fn the_params_block_puts_radii_where_the_shader_reads_them() {
        let bytes = AtmosphereParams::default().bytes();
        let word = |index: usize| {
            let mut four = [0_u8; 4];
            four.copy_from_slice(&bytes[index * 4..index * 4 + 4]);
            f32::from_le_bytes(four)
        };
        assert_eq!(word(20), 6360.0);
        assert_eq!(word(21), 6460.0);
        assert_eq!(word(7), 0.8, "mie g");
        assert_eq!(word(12), 25.0, "ozone centre");
    }
}
