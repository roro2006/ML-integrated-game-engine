//! Time of day: where the sun and moon are, and the light and sky they give
//! (D3). The renderer never owns time; the engine computes the celestial
//! lighting here and writes it into each frame.
//!
//! World axes: +x is east, +y is up, and -z is north, so the noon sun of the
//! northern hemisphere stands toward +z.

use std::time::Duration;

use glam::Vec3;
use renderer::{DirectionalLight, Sky, sunlight_transmittance};

use crate::ecs::Resource;

/// The key light's intensity under a high sun.
const SUN_INTENSITY: f32 = 1.0;
/// The moon's intensity: dim, but enough to read shapes at night.
const MOON_INTENSITY: f32 = 0.06;
/// Moonlight is cool.
const MOON_COLOR: [f32; 3] = [0.62, 0.72, 1.0];
/// Ambient at night as a fraction of the day's.
const NIGHT_AMBIENT: f32 = 0.15;
/// The sun's elevation (sine) over which the key light hands over between
/// sun and moon, through zero intensity, so the switch never pops.
const TWILIGHT_BAND: f32 = 0.1;
const HOURS_PER_DAY: f32 = 24.0;

/// The clock and the place on the planet that set the sun's path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeOfDay {
    /// Local solar time, `0..24`. 12 is solar noon.
    pub hours: f32,
    /// Game hours per real second; 0 holds the time still.
    pub hours_per_second: f32,
    /// Degrees north of the equator.
    pub latitude_degrees: f32,
    /// The sun's declination, degrees: 0 at the equinoxes, about +23.4 at the
    /// northern summer solstice.
    pub declination_degrees: f32,
    /// The drawn sun disc's angular radius, radians. Twice the real sun reads
    /// better at game scale.
    pub sun_disc_radius: f32,
}

impl Default for TimeOfDay {
    /// Mid-afternoon in early summer at mid-latitude, with time held still.
    fn default() -> Self {
        Self {
            hours: 15.0,
            hours_per_second: 0.0,
            latitude_degrees: 45.0,
            declination_degrees: 10.0,
            sun_disc_radius: 0.0093,
        }
    }
}

impl Resource for TimeOfDay {}

/// The lighting the sky gives at one moment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Celestial {
    /// The key light: the sun by day, the moon by night.
    pub key: DirectionalLight,
    /// The sky and its sun, which can sit below the horizon at twilight.
    pub sky: Sky,
    /// 0 at night, 1 in daylight, smooth across twilight; scales ambient.
    pub daylight: f32,
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl TimeOfDay {
    /// Moves the clock on by `elapsed` real time, wrapping at midnight.
    pub fn advance(&mut self, elapsed: Duration) {
        self.hours = (self.hours + self.hours_per_second * elapsed.as_secs_f32()).rem_euclid(HOURS_PER_DAY);
    }

    /// Unit direction to the sun.
    pub fn sun_direction(&self) -> Vec3 {
        let hour_angle = (self.hours - 12.0) / HOURS_PER_DAY * std::f32::consts::TAU;
        let (sin_latitude, cos_latitude) = self.latitude_degrees.to_radians().sin_cos();
        let (sin_declination, cos_declination) = self.declination_degrees.to_radians().sin_cos();
        let (sin_hour, cos_hour) = hour_angle.sin_cos();
        let up = sin_latitude * sin_declination + cos_latitude * cos_declination * cos_hour;
        let east = -cos_declination * sin_hour;
        let north = cos_latitude * sin_declination - sin_latitude * cos_declination * cos_hour;
        Vec3::new(east, up, -north).normalize_or(Vec3::Y)
    }

    /// The key light, sky and daylight for an observer `altitude_metres` up.
    pub fn celestial(&self, altitude_metres: f32) -> Celestial {
        let sun = self.sun_direction();
        let daylight = smoothstep(-TWILIGHT_BAND, TWILIGHT_BAND, sun.y);
        let key = if daylight >= 0.5 {
            DirectionalLight {
                color: sunlight_transmittance(altitude_metres, sun.to_array()),
                intensity: SUN_INTENSITY * (daylight - 0.5) * 2.0,
                ..DirectionalLight::default()
            }
            .with_direction(sun.to_array())
        } else {
            DirectionalLight {
                color: MOON_COLOR,
                intensity: MOON_INTENSITY * (0.5 - daylight) * 2.0,
                ..DirectionalLight::default()
            }
            .with_direction((-sun).to_array())
        };
        Celestial {
            key,
            sky: Sky {
                sun_direction: sun.to_array(),
                sun_illuminance: [SUN_INTENSITY * std::f32::consts::PI; 3],
                sun_angular_radius: self.sun_disc_radius,
            },
            daylight,
        }
    }

    /// Ambient intensity scale for the current daylight.
    pub fn ambient_scale(daylight: f32) -> f32 {
        NIGHT_AMBIENT + (1.0 - NIGHT_AMBIENT) * daylight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(hours: f32) -> TimeOfDay {
        TimeOfDay {
            hours,
            ..TimeOfDay::default()
        }
    }

    #[test]
    fn the_noon_sun_stands_due_south_at_its_highest() {
        let noon = at(12.0).sun_direction();
        assert!(noon.x.abs() < 1e-5, "due south: {noon}");
        assert!(noon.z > 0.0, "south is +z: {noon}");
        for hours in [9.0, 11.5, 12.5, 15.0] {
            assert!(at(hours).sun_direction().y < noon.y);
        }
        // Elevation at noon is 90 - latitude + declination = 55 degrees.
        assert!((noon.y.asin().to_degrees() - 55.0).abs() < 1e-3);
    }

    #[test]
    fn at_the_equinox_the_sun_rises_east_at_six_and_sets_west_at_eighteen() {
        let equinox = |hours| TimeOfDay {
            hours,
            declination_degrees: 0.0,
            ..TimeOfDay::default()
        };
        let rise = equinox(6.0).sun_direction();
        let set = equinox(18.0).sun_direction();
        assert!(rise.y.abs() < 1e-5 && rise.x > 0.99, "{rise}");
        assert!(set.y.abs() < 1e-5 && set.x < -0.99, "{set}");
        assert!(equinox(0.0).sun_direction().y < 0.0, "midnight is dark");
    }

    #[test]
    fn the_clock_advances_and_wraps_at_midnight() {
        let mut time = TimeOfDay {
            hours: 23.5,
            hours_per_second: 0.5,
            ..TimeOfDay::default()
        };
        time.advance(Duration::from_secs(2));
        assert!((time.hours - 0.5).abs() < 1e-5);
        let mut held = TimeOfDay::default();
        held.advance(Duration::from_secs(100));
        assert_eq!(held.hours, TimeOfDay::default().hours);
    }

    #[test]
    fn a_whole_day_changes_the_light_smoothly_and_never_lights_from_below() {
        let minutes = 24 * 60;
        let mut previous = at(0.0).celestial(2.0);
        for minute in 1..=minutes {
            let current = at(minute as f32 / 60.0).celestial(2.0);
            let previous_light = previous.key.intensity;
            assert!(
                (current.key.intensity - previous_light).abs() < 0.05,
                "the key light pops at minute {minute}: {previous_light} to {}",
                current.key.intensity
            );
            assert!((current.daylight - previous.daylight).abs() < 0.05);
            if current.key.intensity > 1e-3 {
                assert!(
                    current.key.direction[1] > 0.0,
                    "lit from below at minute {minute}"
                );
            }
            previous = current;
        }
    }

    #[test]
    fn the_sun_lights_the_day_and_the_moon_the_night() {
        let day = at(12.0).celestial(2.0);
        let night = at(0.0).celestial(2.0);
        assert!(day.key.intensity > 0.9 && day.daylight == 1.0);
        assert!(night.daylight == 0.0 && night.key.intensity > 0.0 && night.key.intensity < 0.1);
        assert!(night.key.color[2] > night.key.color[0], "moonlight is cool");
        assert!(
            day.key.color[0] > 0.5,
            "the noon sun is near white: {:?}",
            day.key.color
        );
        assert_eq!(TimeOfDay::ambient_scale(1.0), 1.0);
        assert_eq!(TimeOfDay::ambient_scale(0.0), NIGHT_AMBIENT);
    }
}
