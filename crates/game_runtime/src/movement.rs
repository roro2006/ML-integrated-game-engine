//! Character movement and static collision.
//!
//! The player and every enemy move as an upright capsule swept against static
//! world geometry, resolved inside the fixed step. Nothing here reads frame
//! time: ADR-0007's committed attacks and dodge windows only mean anything if
//! movement advances at exactly the same rate on every machine.
//!
//! # Why swept rather than teleport-and-push
//!
//! Moving a body to its destination and then pushing it out of whatever it
//! overlaps is simpler, and it lets a fast body pass through a thin wall
//! between two ticks. Sweeping finds the first blocking contact along the
//! motion instead, so a dodge — the fastest thing in the game — cannot tunnel.
//!
//! # Scope
//!
//! Static geometry only: axis-aligned boxes and the ground plane. Bodies do not
//! collide with each other, which the slice does not need and which would make
//! enemy crowding a tuning problem before combat even works.

use engine::ecs::{Component, Resource};
use engine::math::{Vec2, Vec3};

/// How far a body is nudged out of a surface it is resting against.
///
/// Without a small separation, a body resting exactly on a face re-collides
/// every tick and jitters.
const SKIN: f32 = 0.001;

/// Maximum sweep iterations per tick.
///
/// Each iteration resolves one contact and slides along it. Three is enough for
/// an inside corner (two walls plus the floor); the cap stops a body wedged in
/// degenerate geometry from spinning forever.
const MAX_SLIDES: u8 = 3;

/// An upright capsule: the shape every character collides with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Capsule {
    /// Distance from the centre line to the surface.
    pub radius: f32,
    /// Distance between the centres of the two end caps.
    pub height: f32,
}

impl Capsule {
    pub const fn new(radius: f32, height: f32) -> Self {
        Self { radius, height }
    }

    /// Half the total vertical extent, cap tips included.
    pub fn half_extent(self) -> f32 {
        self.height / 2.0 + self.radius
    }
}

impl Default for Capsule {
    /// Roughly human: 1.8 m tall, 0.35 m across.
    fn default() -> Self {
        Self::new(0.35, 1.1)
    }
}

/// A static axis-aligned box of world geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticBox {
    pub min: Vec3,
    pub max: Vec3,
}

impl StaticBox {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self {
            min: min.min(max),
            max: min.max(max),
        }
    }

    /// The box grown by a capsule's radius and half-height.
    ///
    /// Sweeping a capsule against a box is equivalent to sweeping a point
    /// against the box expanded by the capsule's extents. That approximation
    /// squares off the capsule's rounded corners, which for blocking geometry
    /// is imperceptible and far cheaper than the exact test.
    fn expanded_by(&self, capsule: Capsule) -> Self {
        let padding = Vec3::new(capsule.radius, capsule.half_extent(), capsule.radius);
        Self {
            min: self.min - padding,
            max: self.max + padding,
        }
    }

    fn contains(&self, point: Vec3) -> bool {
        point.cmpge(self.min).all() && point.cmple(self.max).all()
    }
}

/// The static collision world for one location.
#[derive(Debug, Default, Clone)]
pub struct CollisionWorld {
    boxes: Vec<StaticBox>,
    /// Height of the flat ground. Bodies never fall below it.
    ground_y: f32,
}

impl Resource for CollisionWorld {}

impl CollisionWorld {
    pub fn new(ground_y: f32) -> Self {
        Self {
            boxes: Vec::new(),
            ground_y,
        }
    }

    pub fn push_box(&mut self, geometry: StaticBox) {
        self.boxes.push(geometry);
    }

    pub fn len(&self) -> usize {
        self.boxes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.boxes.is_empty()
    }

    /// Sweeps `capsule` from `from` along `motion`, sliding along whatever it
    /// hits, and returns where it ends up.
    ///
    /// Sliding rather than stopping is what makes walls feel walkable: a body
    /// pressed into a wall at an angle keeps the component of its motion that
    /// runs along the surface.
    pub fn move_capsule(&self, capsule: Capsule, from: Vec3, motion: Vec3) -> MoveResult {
        let mut position = from;
        let mut remaining = motion;
        let mut result = MoveResult {
            position: from,
            hit_wall: false,
            grounded: false,
        };

        for _ in 0..MAX_SLIDES {
            if remaining.length_squared() <= f32::EPSILON {
                break;
            }
            match self.first_contact(capsule, position, remaining) {
                None => {
                    position += remaining;
                    remaining = Vec3::ZERO;
                }
                Some(contact) => {
                    result.hit_wall = true;
                    // Advance to just before the surface, then redirect what is
                    // left along it.
                    position += remaining * contact.time;
                    let leftover = remaining * (1.0 - contact.time);
                    remaining = leftover - contact.normal * leftover.dot(contact.normal);
                    position += contact.normal * SKIN;
                }
            }
        }

        let floor = self.ground_y + capsule.half_extent();
        if position.y <= floor {
            position.y = floor;
            result.grounded = true;
        }
        result.position = position;
        result
    }

    /// The nearest blocking contact along `motion`, if any.
    fn first_contact(&self, capsule: Capsule, from: Vec3, motion: Vec3) -> Option<Contact> {
        let mut nearest: Option<Contact> = None;
        for geometry in &self.boxes {
            let Some(contact) = sweep_point_into_box(from, motion, &geometry.expanded_by(capsule)) else {
                continue;
            };
            if nearest.is_none_or(|current| contact.time < current.time) {
                nearest = Some(contact);
            }
        }
        nearest
    }

    /// How far along `motion` a capsule gets before it first touches static
    /// geometry, as a fraction in `0..=1`, or `None` if nothing is in the way.
    ///
    /// No sliding, unlike [`Self::move_capsule`]: what a projectile needs is
    /// where it *stops*. The same contact test, so a projectile and a body
    /// agree on where every wall is.
    pub fn sweep(&self, capsule: Capsule, from: Vec3, motion: Vec3) -> Option<f32> {
        if motion.length_squared() <= f32::EPSILON {
            return None;
        }
        self.first_contact(capsule, from, motion)
            .map(|contact| contact.time)
    }

    /// Whether a capsule at `position` overlaps any static geometry.
    pub fn overlaps(&self, capsule: Capsule, position: Vec3) -> bool {
        self.boxes
            .iter()
            .any(|geometry| geometry.expanded_by(capsule).contains(position))
    }
}

/// Where a sweep was stopped.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Contact {
    /// Fraction of the motion travelled before contact, in `0..=1`.
    time: f32,
    /// Surface normal pointing away from the geometry.
    normal: Vec3,
}

/// What a move ended up doing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveResult {
    pub position: Vec3,
    /// Whether anything blocked the motion, which callers use to stop a charge
    /// or cancel a dodge.
    pub hit_wall: bool,
    pub grounded: bool,
}

/// Slab-method sweep of a point into an axis-aligned box.
///
/// Returns the entry time and the face normal, or `None` when the ray misses,
/// starts inside, or only touches within `0..=1`.
fn sweep_point_into_box(from: Vec3, motion: Vec3, target: &StaticBox) -> Option<Contact> {
    // Already inside: report an immediate contact pushing out the shallowest
    // face, so a body that somehow starts embedded escapes rather than sticking.
    if target.contains(from) {
        return Some(Contact {
            time: 0.0,
            normal: shallowest_exit(from, target),
        });
    }

    let mut entry = f32::NEG_INFINITY;
    let mut exit = f32::INFINITY;
    let mut entry_axis = 0;

    for axis in 0..3 {
        let start = from[axis];
        let delta = motion[axis];
        let (low, high) = (target.min[axis], target.max[axis]);

        if delta.abs() <= f32::EPSILON {
            // Parallel to this slab: a miss here is a miss overall.
            if start < low || start > high {
                return None;
            }
            continue;
        }

        let mut near = (low - start) / delta;
        let mut far = (high - start) / delta;
        if near > far {
            std::mem::swap(&mut near, &mut far);
        }
        if near > entry {
            entry = near;
            entry_axis = axis;
        }
        exit = exit.min(far);
        if entry > exit {
            return None;
        }
    }

    if !(0.0..=1.0).contains(&entry) || entry > exit {
        return None;
    }

    let mut normal = Vec3::ZERO;
    normal[entry_axis] = if motion[entry_axis] > 0.0 { -1.0 } else { 1.0 };
    Some(Contact { time: entry, normal })
}

/// The face normal of the nearest wall to a point already inside a box.
fn shallowest_exit(point: Vec3, target: &StaticBox) -> Vec3 {
    let mut best_axis = 0;
    let mut best_depth = f32::INFINITY;
    let mut best_sign = 1.0;

    for axis in 0..3 {
        let to_min = point[axis] - target.min[axis];
        let to_max = target.max[axis] - point[axis];
        let (depth, sign) = if to_min < to_max {
            (to_min, -1.0)
        } else {
            (to_max, 1.0)
        };
        if depth < best_depth {
            best_depth = depth;
            best_axis = axis;
            best_sign = sign;
        }
    }

    let mut normal = Vec3::ZERO;
    normal[best_axis] = best_sign;
    normal
}

/// The action resource gating ADR-0007's committed attacks and dodges.
///
/// Regeneration is expressed per tick rather than per second so that, like
/// every combat window, it is frame-rate independent by construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stamina {
    pub current: f32,
    pub max: f32,
    pub regen_per_tick: f32,
    /// Ticks remaining before regeneration resumes after a spend.
    pub recovery_delay: u32,
    delay_remaining: u32,
}

impl Component for Stamina {}

impl Stamina {
    pub fn new(max: f32, regen_per_tick: f32, recovery_delay: u32) -> Self {
        Self {
            current: max,
            max,
            regen_per_tick,
            recovery_delay,
            delay_remaining: 0,
        }
    }

    /// Spends `cost` if it is available, and pauses regeneration.
    ///
    /// Returns whether the spend happened: a dodge the player cannot afford
    /// must not half-happen.
    pub fn try_spend(&mut self, cost: f32) -> bool {
        if cost > self.current {
            return false;
        }
        self.current -= cost;
        self.delay_remaining = self.recovery_delay;
        true
    }

    /// Advances one fixed tick.
    pub fn tick(&mut self) {
        self.tick_scaled(1.0);
    }

    /// Advances one fixed tick with regeneration scaled by `scale` — what a
    /// status that slows or hastens recovery does (M9). A scale that is not
    /// a finite, non-negative number regenerates nothing rather than
    /// draining or overflowing the pool.
    pub fn tick_scaled(&mut self, scale: f32) {
        if self.delay_remaining > 0 {
            self.delay_remaining -= 1;
            return;
        }
        let scale = if scale.is_finite() { scale.max(0.0) } else { 0.0 };
        self.current = (self.current + self.regen_per_tick * scale).min(self.max);
    }

    pub fn is_regenerating(&self) -> bool {
        self.delay_remaining == 0 && self.current < self.max
    }
}

impl Default for Stamina {
    /// 100 stamina, refilling in about two seconds at 60 Hz after a half-second
    /// pause.
    fn default() -> Self {
        Self::new(100.0, 100.0 / 120.0, 30)
    }
}

/// A dodge in progress: brief, fast, and briefly invulnerable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dodge {
    /// Ticks the dodge lasts in total.
    pub duration: u32,
    /// Ticks from the start during which damage is ignored.
    pub invulnerable_for: u32,
    pub speed: f32,
    direction: Vec2,
    elapsed: u32,
}

impl Component for Dodge {}

impl Dodge {
    pub fn new(direction: Vec2, duration: u32, invulnerable_for: u32, speed: f32) -> Self {
        Self {
            duration,
            invulnerable_for: invulnerable_for.min(duration),
            speed,
            direction: direction.normalize_or_zero(),
            elapsed: 0,
        }
    }

    /// Whether damage should be ignored this tick (ADR-0007's i-frames).
    pub fn is_invulnerable(&self) -> bool {
        self.elapsed < self.invulnerable_for
    }

    pub fn is_finished(&self) -> bool {
        self.elapsed >= self.duration
    }

    /// Motion for this tick, then advances the dodge.
    pub fn advance(&mut self) -> Vec3 {
        if self.is_finished() {
            return Vec3::ZERO;
        }
        self.elapsed += 1;
        Vec3::new(self.direction.x * self.speed, 0.0, self.direction.y * self.speed)
    }

    pub fn elapsed(&self) -> u32 {
        self.elapsed
    }

    /// Which way the roll goes, on the ground plane: normalised, or zero.
    ///
    /// Read-only, for presentation — a roll animation needs its axis, and the
    /// direction is fixed at construction so nothing may change it mid-dodge.
    pub fn direction(&self) -> Vec2 {
        self.direction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wall() -> CollisionWorld {
        let mut world = CollisionWorld::new(0.0);
        // A wall spanning x, sitting at z = 5.
        world.push_box(StaticBox::new(
            Vec3::new(-10.0, 0.0, 5.0),
            Vec3::new(10.0, 3.0, 6.0),
        ));
        world
    }

    fn body() -> Capsule {
        Capsule::new(0.5, 1.0)
    }

    fn at(x: f32, z: f32) -> Vec3 {
        Vec3::new(x, body().half_extent(), z)
    }

    #[test]
    fn unobstructed_motion_is_unchanged() {
        let world = CollisionWorld::new(0.0);
        let result = world.move_capsule(body(), at(0.0, 0.0), Vec3::new(1.0, 0.0, 2.0));
        assert!((result.position.x - 1.0).abs() < 1e-4);
        assert!((result.position.z - 2.0).abs() < 1e-4);
        assert!(!result.hit_wall);
    }

    #[test]
    fn a_wall_stops_motion_into_it() {
        let world = wall();
        let result = world.move_capsule(body(), at(0.0, 0.0), Vec3::new(0.0, 0.0, 10.0));
        assert!(result.hit_wall);
        assert!(
            result.position.z < 4.6,
            "the body should stop short of the wall face, got z={}",
            result.position.z
        );
    }

    /// The reason sweeping exists: a dodge covers several metres in a few
    /// ticks, and must not pass through a one-metre wall.
    #[test]
    fn fast_motion_cannot_tunnel_through_a_wall() {
        let world = wall();
        let result = world.move_capsule(body(), at(0.0, 0.0), Vec3::new(0.0, 0.0, 40.0));
        assert!(result.hit_wall);
        assert!(
            result.position.z < 5.0,
            "a 40 m step must not end up past the wall, got z={}",
            result.position.z
        );
    }

    #[test]
    fn motion_slides_along_a_surface() {
        let world = wall();
        // Diagonally into the wall: the z component is blocked, the x survives.
        let result = world.move_capsule(body(), at(0.0, 0.0), Vec3::new(3.0, 0.0, 10.0));
        assert!(result.hit_wall);
        assert!(
            result.position.x > 0.5,
            "sliding should preserve motion along the wall, got x={}",
            result.position.x
        );
        assert!(result.position.z < 5.0);
    }

    #[test]
    fn bodies_rest_on_the_ground_rather_than_sinking() {
        let world = CollisionWorld::new(0.0);
        let result = world.move_capsule(body(), at(0.0, 0.0), Vec3::new(0.0, -5.0, 0.0));
        assert!(result.grounded);
        assert!((result.position.y - body().half_extent()).abs() < 1e-4);
    }

    #[test]
    fn a_body_starting_inside_geometry_is_pushed_out() {
        let world = wall();
        // Straddling the wall face, moving further in.
        let inside = Vec3::new(0.0, body().half_extent(), 5.5);
        let result = world.move_capsule(body(), inside, Vec3::new(0.0, 0.0, 0.5));
        assert!(
            result.hit_wall,
            "an embedded body must report contact, not slide free"
        );
        assert!(result.position.is_finite(), "and must not produce NaN");
    }

    #[test]
    fn overlap_detection_matches_the_sweep() {
        let world = wall();
        assert!(world.overlaps(body(), Vec3::new(0.0, 1.0, 5.5)));
        assert!(!world.overlaps(body(), Vec3::new(0.0, 1.0, 0.0)));
    }

    #[test]
    fn stamina_spends_atomically_and_regenerates_after_a_delay() {
        let mut stamina = Stamina::new(100.0, 1.0, 3);
        assert!(stamina.try_spend(40.0));
        assert!((stamina.current - 60.0).abs() < 1e-6);

        assert!(
            !stamina.try_spend(1000.0),
            "an unaffordable spend must not half-happen"
        );
        assert!((stamina.current - 60.0).abs() < 1e-6);

        // The delay holds regeneration off for exactly three ticks.
        for _ in 0..3 {
            stamina.tick();
            assert!((stamina.current - 60.0).abs() < 1e-6);
        }
        stamina.tick();
        assert!(stamina.current > 60.0);
    }

    #[test]
    fn stamina_never_exceeds_its_maximum() {
        let mut stamina = Stamina::new(10.0, 4.0, 0);
        stamina.try_spend(5.0);
        for _ in 0..10 {
            stamina.tick();
        }
        assert!((stamina.current - 10.0).abs() < 1e-6);
        assert!(!stamina.is_regenerating(), "a full pool is not regenerating");
    }

    #[test]
    fn a_dodge_is_invulnerable_early_and_ends_on_schedule() {
        let mut dodge = Dodge::new(Vec2::new(0.0, 1.0), 12, 8, 0.4);
        assert!(dodge.is_invulnerable());

        for tick in 0..12 {
            assert!(!dodge.is_finished(), "the dodge ended early at tick {tick}");
            let motion = dodge.advance();
            assert!((motion.z - 0.4).abs() < 1e-6, "dodges move at a fixed speed");
        }

        assert!(dodge.is_finished());
        assert!(!dodge.is_invulnerable(), "i-frames end before the dodge does");
        assert_eq!(dodge.advance(), Vec3::ZERO, "a finished dodge stops moving");
    }

    #[test]
    fn i_frames_cannot_outlast_the_dodge() {
        let dodge = Dodge::new(Vec2::new(1.0, 0.0), 10, 99, 0.5);
        assert_eq!(
            dodge.invulnerable_for, 10,
            "invulnerability is clamped to the duration"
        );
    }

    #[test]
    fn a_dodge_with_no_direction_does_not_produce_nan() {
        let mut dodge = Dodge::new(Vec2::ZERO, 8, 4, 0.5);
        assert_eq!(dodge.advance(), Vec3::ZERO);
    }

    /// Movement must be identical wherever it runs, or replay and the fixed
    /// step are both lies.
    #[test]
    fn stepping_is_deterministic() {
        let world = wall();
        let run = || {
            let mut position = at(0.0, 0.0);
            for tick in 0..60 {
                let motion = Vec3::new(0.05, 0.0, 0.12 + (tick % 3) as f32 * 0.01);
                position = world.move_capsule(body(), position, motion).position;
            }
            position
        };
        assert_eq!(run(), run());
    }
}
