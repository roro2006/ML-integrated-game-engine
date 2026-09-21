//! Enemy behaviour: noticing the player, getting to them, and swinging.
//!
//! Behaviour is chosen from the implemented templates in `EnemyDef`, never
//! authored as logic. Content picks `Melee` or `Charger`; it cannot describe
//! *how* either one thinks. That is the same allowlist principle as abilities
//! and effects, and it is what will later make an AI-proposed enemy variant
//! safe to run (ADR-0010): the worst a proposal can do is choose a template and
//! some numbers.
//!
//! # Hysteresis, not a single radius
//!
//! Aggro uses a smaller acquire radius than its release radius. With one
//! threshold, a player standing exactly at the edge makes the enemy flicker
//! between idle and chasing every tick — which looks broken and, worse, keeps
//! restarting the approach.
//!
//! # Pathing
//!
//! Enemies steer along a path of navmesh polygon centres produced by
//! [`crate::navigation`]. Within a polygon they move straight at the next
//! waypoint; the navmesh's convexity is what makes that safe.

use engine::ecs::Component;
use engine::math::{Vec2, Vec3};
use game_core::content::{AbilityDef, BehaviorTemplate, Effect, EnemyTuning, Fixed, Targeting};

use crate::ability::{Cooldowns, units};
use crate::combat::{Attack, AttackTiming};
use crate::movement::{Capsule, CollisionWorld, MoveResult};
use crate::navigation::NavMesh;

/// A vector flattened to the ground plane.
fn ground(vector: Vec3) -> Vec2 {
    Vec2::new(vector.x, vector.z)
}

/// Distances at which an enemy notices and forgets the player.
///
/// `release` is deliberately larger than `acquire`; see the module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AggroRange {
    pub acquire: f32,
    pub release: f32,
}

impl AggroRange {
    /// Builds a range, forcing `release` to be at least `acquire` so a
    /// mistuned pair cannot reintroduce the flicker this type exists to stop.
    pub fn new(acquire: f32, release: f32) -> Self {
        Self {
            acquire,
            release: release.max(acquire),
        }
    }
}

impl Default for AggroRange {
    fn default() -> Self {
        Self::new(9.0, 14.0)
    }
}

/// A swing's hit radius when content does not set one.
const DEFAULT_STRIKE_RADIUS: f32 = 0.8;

/// How far a swing reaches when content does not say: where the enemy likes
/// to stand, but never so close that a swing lands inside its own body.
fn default_reach(preferred_range: f32) -> f32 {
    preferred_range.max(1.2)
}

/// What an enemy is doing right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnemyState {
    /// Unaware, holding its post.
    Idle,
    /// Closing the distance.
    Approach,
    /// In range and swinging.
    Attack,
    /// Backing off after a swing, so melee does not become a grinding shove.
    Reposition,
}

/// How an enemy behaves, and its current state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnemyBrain {
    pub template: BehaviorTemplate,
    pub aggro: AggroRange,
    /// How close it wants to be before attacking.
    pub preferred_range: f32,
    pub move_speed: f32,
    pub timing: AttackTiming,
    pub damage: f32,
    pub reach: f32,
    /// How wide a swing's hit is.
    pub strike_radius: f32,
    /// How much damage its stance absorbs before it staggers.
    pub poise: f32,
    state: EnemyState,
    /// Ticks left in [`EnemyState::Reposition`].
    recover_ticks: u32,
}

impl Component for EnemyBrain {}

impl EnemyBrain {
    /// Builds a brain with the ranges and pacing each template implies.
    ///
    /// The numbers are starting points for tuning, not balance: what matters
    /// here is that each template is *distinguishable* in how it closes.
    pub fn new(template: BehaviorTemplate) -> Self {
        // Poise against the player's 22-point swing: a husk or a spitter
        // breaks on one clean hit, a charger or a support takes two.
        let (preferred_range, move_speed, timing, poise) = match template {
            // Gets in your face and stays committed.
            BehaviorTemplate::Melee => (1.6, 0.055, AttackTiming::new(16, 6, 26), 20.0),
            // Keeps its distance and punishes approach.
            BehaviorTemplate::Ranged => (7.5, 0.040, AttackTiming::new(26, 4, 30), 12.0),
            // Rushes, hits hard, and is slow to recover.
            BehaviorTemplate::Charger => (2.2, 0.085, AttackTiming::new(22, 8, 40), 40.0),
            // Hangs back; its threat is what it enables, not what it swings.
            BehaviorTemplate::Support => (6.0, 0.045, AttackTiming::new(30, 4, 34), 30.0),
        };
        Self {
            template,
            aggro: AggroRange::default(),
            preferred_range,
            move_speed,
            timing,
            damage: 8.0,
            reach: default_reach(preferred_range),
            strike_radius: DEFAULT_STRIKE_RADIUS,
            poise,
            state: EnemyState::Idle,
            recover_ticks: 0,
        }
    }

    /// Builds a brain from content: the template's numbers, overridden by
    /// whatever `tuning` sets (M8).
    ///
    /// `step_seconds` is one fixed tick, because content speaks in metres per
    /// second and the brain moves in metres per tick. A number content leaves
    /// out is the template's exactly — not converted there and back — so an
    /// enemy with no tuning plays bit for bit as it did before tuning existed.
    ///
    /// Clamped here as well as validated, because the brain does not assume
    /// the validator ran; and whatever acquire and release say,
    /// [`AggroRange::new`] keeps release at least acquire.
    pub fn tuned(template: BehaviorTemplate, tuning: &EnemyTuning, step_seconds: f32) -> Self {
        use game_core::content::{
            MAX_AGGRO_METRES, MAX_ENEMY_SPEED_METRES, MAX_POISE, MAX_PREFERRED_RANGE_METRES,
            MAX_REACH_METRES, MAX_STRIKE_RADIUS_METRES, MAX_WINDOW_TICKS,
        };
        let metres =
            |value: Fixed, low: f32, high: i64| (value.as_milli() as f32 / 1000.0).clamp(low, high as f32);
        let mut brain = Self::new(template);
        brain.aggro = AggroRange::new(
            tuning
                .acquire
                .map_or(brain.aggro.acquire, |value| metres(value, 0.5, MAX_AGGRO_METRES)),
            tuning
                .release
                .map_or(brain.aggro.release, |value| metres(value, 0.5, MAX_AGGRO_METRES)),
        );
        if let Some(range) = tuning.preferred_range {
            brain.preferred_range = metres(range, 0.5, MAX_PREFERRED_RANGE_METRES);
        }
        if let Some(speed) = tuning.move_speed {
            brain.move_speed = metres(speed, 0.1, MAX_ENEMY_SPEED_METRES) * step_seconds.max(0.0);
        }
        let window = |value: Option<u32>, default: u32, least: u32| {
            value.map_or(default, |ticks| ticks.clamp(least, MAX_WINDOW_TICKS))
        };
        brain.timing = AttackTiming::new(
            window(tuning.windup_ticks, brain.timing.windup, 1),
            window(tuning.active_ticks, brain.timing.active, 1),
            window(tuning.recovery_ticks, brain.timing.recovery, 0),
        );
        // Reach follows a tuned preferred range unless it is tuned itself,
        // as it follows the template's: an enemy that wants to stand further
        // off swings from further off.
        brain.reach = tuning.reach.map_or_else(
            || default_reach(brain.preferred_range),
            |reach| metres(reach, 0.5, MAX_REACH_METRES),
        );
        if let Some(radius) = tuning.strike_radius {
            brain.strike_radius = metres(radius, 0.1, MAX_STRIKE_RADIUS_METRES);
        }
        if let Some(poise) = tuning.poise {
            brain.poise = metres(poise, 1.0, MAX_POISE);
        }
        brain
    }

    pub fn state(&self) -> EnemyState {
        self.state
    }

    /// Whether this enemy wants to start a swing this tick.
    pub fn wants_to_attack(&self) -> bool {
        self.state == EnemyState::Attack
    }

    /// The attack this enemy makes, built from its own tuning.
    pub fn build_attack(&self) -> Attack {
        Attack::new(self.timing, self.damage, self.reach, self.strike_radius)
    }

    /// Advances one fixed tick of decision-making.
    ///
    /// Pure: it reads distances and returns intent. Actually moving and
    /// swinging is the caller's job, which keeps this exhaustively testable
    /// without a world.
    pub fn think(&mut self, distance_to_player: f32, mid_attack: bool) -> EnemyState {
        if mid_attack {
            // Committed (ADR-0007): an enemy mid-swing does not re-decide.
            return self.state;
        }
        if self.recover_ticks > 0 {
            self.recover_ticks -= 1;
            self.state = EnemyState::Reposition;
            return self.state;
        }

        // The hysteresis, stated directly: an unaware enemy needs the player
        // inside the tighter `acquire` radius, while one already engaged only
        // breaks off past the wider `release` radius. Burying this in match-arm
        // order made an enemy that acquired at point-blank spend a tick walking
        // before it could swing.
        let engaged = self.state != EnemyState::Idle;
        let interested = if engaged {
            distance_to_player <= self.aggro.release
        } else {
            distance_to_player <= self.aggro.acquire
        };

        self.state = if !interested {
            EnemyState::Idle
        } else if distance_to_player <= self.preferred_range {
            // Swinging immediately is fair here: the attack's own wind-up is
            // the telegraph, so no extra reaction delay is needed.
            EnemyState::Attack
        } else {
            EnemyState::Approach
        };
        self.state
    }

    /// Called once a swing finishes, so the enemy steps off before the next.
    pub fn finish_attack(&mut self) {
        self.recover_ticks = self.timing.recovery / 2;
        self.state = EnemyState::Reposition;
    }

    /// Motion for this tick, given where the enemy wants to go.
    ///
    /// Returns zero when it should hold position, which is what keeps a ranged
    /// enemy from walking into melee out of sheer enthusiasm.
    pub fn desired_motion(&self, from: Vec3, toward: Vec3) -> Vec3 {
        let offset = ground(toward) - ground(from);
        let distance = offset.length();
        let direction = offset.normalize_or_zero();

        let step = match self.state {
            EnemyState::Approach => self.move_speed,
            // Ranged and support back away if the player closes inside their
            // preferred distance.
            EnemyState::Attack | EnemyState::Reposition
                if distance < self.preferred_range * 0.6
                    && matches!(
                        self.template,
                        BehaviorTemplate::Ranged | BehaviorTemplate::Support
                    ) =>
            {
                -self.move_speed
            }
            _ => 0.0,
        };

        Vec3::new(direction.x * step, 0.0, direction.y * step)
    }
}

/// Moves an enemy one tick toward `waypoint`, respecting static collision.
pub fn step_enemy(
    brain: &EnemyBrain,
    capsule: Capsule,
    from: Vec3,
    waypoint: Vec3,
    world: &CollisionWorld,
) -> MoveResult {
    world.move_capsule(capsule, from, brain.desired_motion(from, waypoint))
}

/// How close an enemy must get to a waypoint before moving to the next.
///
/// Larger than it looks necessary: a body that has to land exactly on a
/// waypoint orbits it forever, because each tick overshoots and turns back.
const WAYPOINT_REACHED: f32 = 0.35;

/// How far a target must move before the path to it is recomputed.
///
/// Re-pathing every tick would be both wasteful and jittery — a path that
/// changes constantly makes an enemy wobble. Re-pathing never would have it
/// walk confidently to where the player used to be.
const REPATH_DISTANCE: f32 = 2.0;

/// An enemy's progress along a navmesh path.
///
/// Held per enemy. Runtime state: rebuilt on load like everything else that
/// derives from where things currently are.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct PathFollower {
    waypoints: Vec<Vec3>,
    next: usize,
    /// Where the target was when this path was computed.
    planned_for: Option<Vec3>,
}

impl Component for PathFollower {}

impl PathFollower {
    /// The waypoint being walked toward, if any.
    pub fn current_waypoint(&self) -> Option<Vec3> {
        self.waypoints.get(self.next).copied()
    }

    pub fn is_finished(&self) -> bool {
        self.next >= self.waypoints.len()
    }

    pub fn waypoints(&self) -> &[Vec3] {
        &self.waypoints
    }

    /// Whether the path should be recomputed for a target now at `target`.
    pub fn needs_repath(&self, target: Vec3) -> bool {
        match self.planned_for {
            None => true,
            Some(planned) => self.is_finished() || ground(planned).distance(ground(target)) > REPATH_DISTANCE,
        }
    }

    /// Replaces the path.
    pub fn set_path(&mut self, waypoints: Vec<Vec3>, target: Vec3) {
        self.waypoints = waypoints;
        // The first waypoint is where the body already stands, so skip it.
        self.next = usize::from(self.waypoints.len() > 1);
        self.planned_for = Some(target);
    }

    pub fn clear(&mut self) {
        self.waypoints.clear();
        self.next = 0;
        self.planned_for = None;
    }

    /// Advances past any waypoints `position` has reached, and returns the one
    /// to steer at now.
    pub fn advance(&mut self, position: Vec3) -> Option<Vec3> {
        while let Some(waypoint) = self.current_waypoint() {
            if ground(position).distance(ground(waypoint)) <= WAYPOINT_REACHED {
                self.next += 1;
            } else {
                return Some(waypoint);
            }
        }
        None
    }
}

/// Steers an enemy along its path toward `target`, re-pathing when needed.
///
/// Falls back to steering straight at the target when the navmesh cannot
/// produce a path: an enemy that stands still because pathing failed looks far
/// more broken than one that walks into a wall.
pub fn follow_path(
    brain: &EnemyBrain,
    follower: &mut PathFollower,
    capsule: Capsule,
    from: Vec3,
    target: Vec3,
    navmesh: &NavMesh,
    world: &CollisionWorld,
) -> MoveResult {
    if follower.needs_repath(target) {
        match navmesh.find_path(from, target) {
            Some(path) => follower.set_path(path, target),
            None => follower.clear(),
        }
    }

    let waypoint = follower.advance(from).unwrap_or(target);
    step_enemy(brain, capsule, from, waypoint, world)
}

/// How near its target an ability must be used from, for choosing one:
/// `None` when it lands on the caster and works from anywhere.
///
/// Read from the ability, never authored separately, so the range an enemy
/// chooses by is the range the ability actually works at: a projectile's own
/// range, and `reach` — the runtime's targeting reach — for everything aimed
/// at someone or somewhere.
pub fn usable_within(def: &AbilityDef, reach: f32) -> Option<f32> {
    if let Some(Effect::Projectile { range, .. }) = def.projectile() {
        return Some(units(*range).clamp(0.0, game_core::content::MAX_PROJECTILE_RANGE_METRES as f32));
    }
    match def.targeting {
        Targeting::SelfOnly => None,
        Targeting::SingleEnemy
        | Targeting::SingleAlly
        | Targeting::AreaAroundSelf
        | Targeting::AreaAtPoint => Some(reach),
    }
}

/// Which of its abilities an enemy uses now, if any: the first, in authored
/// order, that is off cooldown, affordable, and in range of the player.
///
/// A rule, not a script (M8). Content decides what an enemy knows and in what
/// order it reaches for it — its list is its priority — but not how it
/// decides; extending this rule is a code change with tests. `None` means
/// swing instead.
pub fn choose_ability(
    known: &[Option<&AbilityDef>],
    cooldowns: &Cooldowns,
    stamina: f32,
    distance: f32,
    reach: f32,
) -> Option<usize> {
    known.iter().position(|def| {
        def.is_some_and(|def| {
            cooldowns.is_ready(&def.id)
                && stamina >= units(def.cost)
                && usable_within(def, reach).is_none_or(|within| distance <= within)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn melee() -> EnemyBrain {
        EnemyBrain::new(BehaviorTemplate::Melee)
    }

    #[test]
    fn templates_differ_in_how_they_close() {
        let melee = EnemyBrain::new(BehaviorTemplate::Melee);
        let ranged = EnemyBrain::new(BehaviorTemplate::Ranged);
        let charger = EnemyBrain::new(BehaviorTemplate::Charger);

        assert!(
            ranged.preferred_range > melee.preferred_range,
            "a ranged enemy should want distance a melee one does not"
        );
        assert!(
            charger.move_speed > melee.move_speed,
            "a charger should close faster than a plain melee enemy"
        );
        assert!(
            charger.timing.recovery > melee.timing.recovery,
            "and pay for it with a longer punish window"
        );
    }

    #[test]
    fn an_unaware_enemy_holds_its_post() {
        let mut brain = melee();
        assert_eq!(brain.think(50.0, false), EnemyState::Idle);
        assert_eq!(
            brain.desired_motion(Vec3::ZERO, Vec3::new(50.0, 0.0, 0.0)),
            Vec3::ZERO
        );
    }

    #[test]
    fn approach_then_attack_as_the_player_closes() {
        let mut brain = melee();
        assert_eq!(brain.think(8.0, false), EnemyState::Approach);
        assert_eq!(brain.think(1.0, false), EnemyState::Attack);
        assert!(brain.wants_to_attack());
    }

    /// The flicker this design exists to prevent: at the acquire edge, a single
    /// threshold would toggle state every tick.
    #[test]
    fn aggro_hysteresis_prevents_flicker() {
        let mut brain = melee();
        let edge = brain.aggro.acquire;

        assert_eq!(brain.think(edge - 0.01, false), EnemyState::Approach);
        // Just outside acquire, but inside release: it keeps chasing.
        assert_eq!(brain.think(edge + 0.5, false), EnemyState::Approach);
        assert_eq!(brain.think(edge + 1.0, false), EnemyState::Approach);
        // Only past the wider radius does it give up.
        assert_eq!(brain.think(brain.aggro.release + 0.1, false), EnemyState::Idle);
    }

    #[test]
    fn a_mistuned_range_cannot_reintroduce_flicker() {
        let range = AggroRange::new(10.0, 4.0);
        assert!(
            range.release >= range.acquire,
            "release must never be tighter than acquire"
        );
    }

    #[test]
    fn an_enemy_mid_swing_does_not_rethink() {
        let mut brain = melee();
        brain.think(1.0, false);
        assert_eq!(brain.state(), EnemyState::Attack);

        // Player sprints away mid-swing; the enemy stays committed.
        assert_eq!(brain.think(40.0, true), EnemyState::Attack);
    }

    #[test]
    fn finishing_a_swing_forces_a_step_off() {
        let mut brain = melee();
        brain.think(1.0, false);
        brain.finish_attack();
        assert_eq!(brain.state(), EnemyState::Reposition);

        // It stays in reposition for a while rather than immediately re-swinging.
        let mut repositioning = 0;
        for _ in 0..brain.timing.recovery {
            if brain.think(1.0, false) == EnemyState::Reposition {
                repositioning += 1;
            } else {
                break;
            }
        }
        assert!(repositioning > 0, "an enemy should not chain swings with no gap");
    }

    #[test]
    fn ranged_enemies_back_away_when_crowded() {
        let mut brain = EnemyBrain::new(BehaviorTemplate::Ranged);
        brain.think(1.0, false);
        let motion = brain.desired_motion(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        assert!(
            motion.x < 0.0,
            "a ranged enemy crowded in melee should retreat, got {motion:?}"
        );
    }

    #[test]
    fn melee_enemies_do_not_retreat() {
        let mut brain = melee();
        brain.think(1.0, false);
        let motion = brain.desired_motion(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(motion, Vec3::ZERO, "melee holds its ground while attacking");
    }

    #[test]
    fn approach_motion_points_at_the_target() {
        let mut brain = melee();
        brain.think(8.0, false);
        let motion = brain.desired_motion(Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0));
        assert!(motion.z < 0.0, "should move along -Z toward the target");
        assert!((motion.length() - brain.move_speed).abs() < 1e-5);
    }

    #[test]
    fn coincident_positions_do_not_produce_nan() {
        let mut brain = melee();
        brain.think(0.0, false);
        let motion = brain.desired_motion(Vec3::ZERO, Vec3::ZERO);
        assert!(motion.is_finite());
    }

    #[test]
    fn enemies_cannot_walk_through_walls() {
        let mut world = CollisionWorld::new(0.0);
        world.push_box(crate::movement::StaticBox::new(
            Vec3::new(-5.0, 0.0, 2.0),
            Vec3::new(5.0, 3.0, 3.0),
        ));

        let capsule = Capsule::new(0.4, 1.0);
        let mut brain = EnemyBrain::new(BehaviorTemplate::Charger);
        brain.think(8.0, false);

        let mut position = Vec3::new(0.0, capsule.half_extent(), 0.0);
        let target = Vec3::new(0.0, capsule.half_extent(), 10.0);
        for _ in 0..200 {
            position = step_enemy(&brain, capsule, position, target, &world).position;
        }

        assert!(
            position.z < 2.0,
            "a charger should be stopped by the wall, ended at z={}",
            position.z
        );
    }

    /// Enemy decisions feed the fixed step, so two identical runs must agree.
    #[test]
    fn thinking_is_deterministic() {
        let run = || {
            let mut brain = EnemyBrain::new(BehaviorTemplate::Charger);
            let mut states = Vec::new();
            for tick in 0..120 {
                let distance = 12.0 - (tick as f32 * 0.1);
                states.push(brain.think(distance.max(0.5), false));
            }
            states
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn the_attack_an_enemy_builds_matches_its_tuning() {
        let brain = EnemyBrain::new(BehaviorTemplate::Charger);
        let attack = brain.build_attack();
        assert_eq!(attack.timing, brain.timing);
        assert!((attack.damage - brain.damage).abs() < 1e-6);
    }

    // ---- tuning from content (M8 wave 1) ----------------------------------

    const TICK: f32 = 1.0 / 60.0;

    fn metres(value: f32) -> Option<Fixed> {
        Some(Fixed::from_milli((value * 1000.0).round() as i64))
    }

    #[test]
    fn an_untuned_enemy_plays_exactly_as_its_template() {
        for template in [
            BehaviorTemplate::Melee,
            BehaviorTemplate::Ranged,
            BehaviorTemplate::Charger,
            BehaviorTemplate::Support,
        ] {
            assert_eq!(
                EnemyBrain::tuned(template, &EnemyTuning::default(), TICK),
                EnemyBrain::new(template),
                "{template:?}: saying nothing changes nothing, bit for bit"
            );
        }
    }

    #[test]
    fn content_sets_range_pacing_timing_and_swing() {
        let tuning = EnemyTuning {
            acquire: metres(11.0),
            release: metres(16.0),
            preferred_range: metres(3.5),
            move_speed: metres(6.0),
            windup_ticks: Some(40),
            active_ticks: Some(3),
            recovery_ticks: Some(12),
            reach: metres(2.5),
            strike_radius: metres(1.4),
            poise: metres(75.0),
        };
        let brain = EnemyBrain::tuned(BehaviorTemplate::Melee, &tuning, TICK);
        assert!((brain.poise - 75.0).abs() < 1e-6);
        assert_eq!(brain.aggro, AggroRange::new(11.0, 16.0));
        assert!((brain.preferred_range - 3.5).abs() < 1e-6);
        assert!((brain.move_speed - 0.1).abs() < 1e-6, "6 m/s is 0.1 m a tick");
        assert_eq!(brain.timing, AttackTiming::new(40, 3, 12));
        let attack = brain.build_attack();
        assert!((attack.reach - 2.5).abs() < 1e-6);
        assert!(
            (attack.radius - 1.4).abs() < 1e-6,
            "the strike radius reaches the swing"
        );
        assert_eq!(attack.timing, AttackTiming::new(40, 3, 12));
        assert_eq!(
            brain.template,
            BehaviorTemplate::Melee,
            "the template still decides how it thinks"
        );
    }

    #[test]
    fn a_tuned_preferred_range_carries_the_reach_with_it() {
        let tuning = EnemyTuning {
            preferred_range: metres(4.0),
            ..EnemyTuning::default()
        };
        let brain = EnemyBrain::tuned(BehaviorTemplate::Melee, &tuning, TICK);
        assert!(
            (brain.reach - 4.0).abs() < 1e-6,
            "reach follows where it likes to stand"
        );
    }

    /// Validation refuses these; the brain holds the line anyway, because a
    /// generated candidate reaches it by a different path.
    #[test]
    fn hostile_tuning_is_clamped_and_keeps_its_hysteresis() {
        let hostile = EnemyTuning {
            acquire: metres(20.0),
            release: metres(5.0),
            preferred_range: metres(900.0),
            move_speed: metres(-4.0),
            windup_ticks: Some(0),
            active_ticks: Some(0),
            recovery_ticks: Some(100_000),
            reach: metres(1_000.0),
            strike_radius: metres(50.0),
            poise: metres(-10.0),
        };
        let brain = EnemyBrain::tuned(BehaviorTemplate::Charger, &hostile, TICK);
        assert!(
            brain.aggro.release >= brain.aggro.acquire,
            "release is never inside acquire"
        );
        assert!(brain.preferred_range <= game_core::content::MAX_PREFERRED_RANGE_METRES as f32);
        assert!(
            brain.move_speed > 0.0,
            "a negative speed does not make it walk backwards"
        );
        assert!(brain.timing.windup >= 1, "a swing always has a wind-up to read");
        assert!(brain.timing.active >= 1);
        assert!(brain.timing.recovery <= game_core::content::MAX_WINDOW_TICKS);
        assert!(brain.reach <= game_core::content::MAX_REACH_METRES as f32);
        assert!(brain.strike_radius <= game_core::content::MAX_STRIKE_RADIUS_METRES as f32);
        assert!(
            brain.poise >= 1.0,
            "a stance of nothing would stagger on every scratch"
        );
    }

    mod ability_choice {
        use super::*;
        use game_core::content::{AbilityPresentation, DamageType};
        use game_core::ids::AbilityId;

        fn ability(name: &str, targeting: Targeting, cost: i64, effect: Effect) -> AbilityDef {
            AbilityDef {
                schema_version: 1,
                id: AbilityId::new(name).expect("valid id"),
                name: name.to_owned(),
                targeting,
                cost: Fixed::from_whole(cost),
                cooldown_ticks: 60,
                windup_ticks: 10,
                recovery_ticks: 10,
                effects: vec![effect],
                presentation: AbilityPresentation::default(),
            }
        }

        fn fire() -> Effect {
            Effect::DealDamage {
                amount: Fixed::from_whole(5),
                damage_type: DamageType::Fire,
            }
        }

        fn spit(range: i64) -> AbilityDef {
            ability(
                "spit",
                Targeting::SelfOnly,
                0,
                Effect::Projectile {
                    speed: Fixed::from_whole(12),
                    range: Fixed::from_whole(range),
                    radius: Fixed::from_milli(300),
                    pierce: 0,
                    on_hit: vec![fire()],
                },
            )
        }

        #[test]
        fn range_is_read_from_the_ability() {
            assert_eq!(
                usable_within(&spit(12), 6.0),
                Some(12.0),
                "a projectile's own range"
            );
            let slam = ability("slam", Targeting::AreaAroundSelf, 0, fire());
            assert_eq!(
                usable_within(&slam, 6.0),
                Some(6.0),
                "an area: the targeting reach"
            );
            let brace = ability("brace", Targeting::SelfOnly, 0, fire());
            assert_eq!(usable_within(&brace, 6.0), None, "on the caster: anywhere");
        }

        #[test]
        fn the_first_ready_affordable_ability_in_range_is_chosen() {
            let spit = spit(12);
            let slam = ability("slam", Targeting::AreaAroundSelf, 0, fire());
            let known = [Some(&slam), Some(&spit)];
            let cooldowns = Cooldowns::new();

            assert_eq!(
                choose_ability(&known, &cooldowns, 100.0, 4.0, 6.0),
                Some(0),
                "both in range: the first"
            );
            assert_eq!(
                choose_ability(&known, &cooldowns, 100.0, 9.0, 6.0),
                Some(1),
                "only the spit reaches 9 m"
            );
            assert_eq!(
                choose_ability(&known, &cooldowns, 100.0, 15.0, 6.0),
                None,
                "nothing reaches: swing"
            );

            let mut cooling = Cooldowns::new();
            cooling.start(slam.id.clone(), 30);
            assert_eq!(
                choose_ability(&known, &cooling, 100.0, 4.0, 6.0),
                Some(1),
                "the first cooling down, so the next"
            );
        }

        #[test]
        fn an_ability_it_cannot_afford_is_passed_over() {
            let costly = ability("costly", Targeting::AreaAroundSelf, 50, fire());
            let cheap = ability("cheap", Targeting::AreaAroundSelf, 5, fire());
            let known = [Some(&costly), Some(&cheap)];
            assert_eq!(choose_ability(&known, &Cooldowns::new(), 20.0, 2.0, 6.0), Some(1));
            assert_eq!(choose_ability(&known, &Cooldowns::new(), 1.0, 2.0, 6.0), None);
        }

        #[test]
        fn an_unknown_ability_is_skipped_not_fatal() {
            let slam = ability("slam", Targeting::AreaAroundSelf, 0, fire());
            let known = [None, Some(&slam)];
            assert_eq!(
                choose_ability(&known, &Cooldowns::new(), 100.0, 2.0, 6.0),
                Some(1)
            );
        }
    }

    mod path_following {
        use super::*;
        use game_core::content::{
            CONTENT_SCHEMA_VERSION, Fixed, LocationDef, LocationKind, NavPolygon, Point,
        };
        use game_core::ids::{LocationId, RegionId};

        fn at(x: f32, z: f32) -> Vec3 {
            Vec3::new(x, 0.0, z)
        }

        fn corner(x: i64, z: i64) -> Point {
            Point::new(Fixed::from_whole(x), Fixed::ZERO, Fixed::from_whole(z))
        }

        /// Two rooms joined along z = 10.
        fn two_rooms() -> LocationDef {
            LocationDef {
                schema_version: CONTENT_SCHEMA_VERSION,
                id: LocationId::new("two-rooms").expect("valid id"),
                name: "Two Rooms".to_owned(),
                region: RegionId::new("cinderwake").expect("valid id"),
                kind: LocationKind::Dungeon,
                spawns: Vec::new(),
                props: Vec::new(),
                navmesh: vec![
                    NavPolygon {
                        vertices: vec![corner(-5, 0), corner(5, 0), corner(5, 10), corner(-5, 10)],
                        neighbours: vec![1],
                    },
                    NavPolygon {
                        vertices: vec![corner(-5, 10), corner(5, 10), corner(5, 20), corner(-5, 20)],
                        neighbours: vec![0],
                    },
                ],
                links: Vec::new(),
            }
        }

        fn approaching() -> EnemyBrain {
            let mut brain = EnemyBrain::new(BehaviorTemplate::Melee);
            brain.think(8.0, false);
            assert_eq!(brain.state(), EnemyState::Approach);
            brain
        }

        #[test]
        fn a_new_path_skips_the_step_underfoot() {
            let mut follower = PathFollower::default();
            follower.set_path(vec![at(0.0, 0.0), at(0.0, 10.0), at(0.0, 18.0)], at(0.0, 18.0));
            assert_eq!(
                follower.current_waypoint(),
                Some(at(0.0, 10.0)),
                "the first waypoint is where the body already stands"
            );
        }

        #[test]
        fn a_single_waypoint_path_is_walked_rather_than_skipped() {
            let mut follower = PathFollower::default();
            follower.set_path(vec![at(3.0, 3.0)], at(3.0, 3.0));
            assert_eq!(follower.current_waypoint(), Some(at(3.0, 3.0)));
        }

        #[test]
        fn waypoints_are_reached_with_tolerance() {
            let mut follower = PathFollower::default();
            follower.set_path(vec![at(0.0, 0.0), at(0.0, 5.0), at(0.0, 9.0)], at(0.0, 9.0));

            assert_eq!(follower.advance(at(0.0, 0.0)), Some(at(0.0, 5.0)));
            // Close enough counts: demanding an exact landing makes a body
            // orbit a waypoint forever.
            assert_eq!(follower.advance(at(0.0, 4.8)), Some(at(0.0, 9.0)));
            assert_eq!(follower.advance(at(0.0, 9.1)), None, "the path is walked out");
            assert!(follower.is_finished());
        }

        #[test]
        fn repathing_waits_until_the_target_has_actually_moved() {
            let mut follower = PathFollower::default();
            assert!(
                follower.needs_repath(at(0.0, 10.0)),
                "a follower with no path needs one"
            );

            follower.set_path(vec![at(0.0, 0.0), at(0.0, 10.0)], at(0.0, 10.0));
            assert!(
                !follower.needs_repath(at(0.5, 10.2)),
                "a target that shuffled must not trigger a recompute every tick"
            );
            assert!(
                follower.needs_repath(at(0.0, 16.0)),
                "a target that genuinely moved must"
            );
        }

        #[test]
        fn a_walked_out_path_asks_to_be_replaced() {
            let mut follower = PathFollower::default();
            follower.set_path(vec![at(0.0, 0.0), at(0.0, 1.0)], at(0.0, 1.0));
            follower.advance(at(0.0, 1.0));
            assert!(follower.is_finished());
            assert!(follower.needs_repath(at(0.0, 1.0)));
        }

        #[test]
        fn clearing_forgets_everything() {
            let mut follower = PathFollower::default();
            follower.set_path(vec![at(0.0, 0.0), at(0.0, 5.0)], at(0.0, 5.0));
            follower.clear();
            assert!(follower.waypoints().is_empty());
            assert_eq!(follower.current_waypoint(), None);
            assert!(follower.needs_repath(at(0.0, 5.0)));
        }

        #[test]
        fn following_a_path_routes_through_the_portal() {
            let navmesh = NavMesh::from_location(&two_rooms());
            let world = CollisionWorld::new(0.0);
            let capsule = Capsule::new(0.4, 1.0);
            let brain = approaching();
            let mut follower = PathFollower::default();

            let from = at(-3.0, 2.0);
            follow_path(
                &brain,
                &mut follower,
                capsule,
                from,
                at(3.0, 18.0),
                &navmesh,
                &world,
            );

            assert!(!follower.waypoints().is_empty(), "a path should have been found");
            let steering_at = follower.current_waypoint().expect("a waypoint to steer at");
            assert!(
                steering_at.z <= 10.5,
                "the enemy should aim at the portal first, not the far room: {steering_at:?}"
            );
        }

        #[test]
        fn an_enemy_walks_the_whole_path_to_the_other_room() {
            let navmesh = NavMesh::from_location(&two_rooms());
            let world = CollisionWorld::new(0.0);
            let capsule = Capsule::new(0.4, 1.0);
            let brain = approaching();
            let mut follower = PathFollower::default();

            let target = at(0.0, 17.0);
            let mut position = Vec3::new(0.0, capsule.half_extent(), 2.0);
            for _ in 0..1_200 {
                position =
                    follow_path(&brain, &mut follower, capsule, position, target, &navmesh, &world).position;
            }

            assert!(
                position.z > 10.0,
                "the enemy should have crossed into the far room, ended at z={}",
                position.z
            );
        }

        /// Standing still because pathing failed looks far more broken than
        /// walking into a wall, so the fallback steers straight at the target.
        #[test]
        fn an_unpathable_target_falls_back_to_steering_directly() {
            let empty = NavMesh::default();
            let world = CollisionWorld::new(0.0);
            let capsule = Capsule::new(0.4, 1.0);
            let brain = approaching();
            let mut follower = PathFollower::default();

            let from = Vec3::new(0.0, capsule.half_extent(), 0.0);
            let result = follow_path(
                &brain,
                &mut follower,
                capsule,
                from,
                at(0.0, 10.0),
                &empty,
                &world,
            );

            assert!(
                result.position.z > from.z,
                "the enemy should still advance toward its target, got {:?}",
                result.position
            );
            assert!(follower.waypoints().is_empty(), "and hold no stale path");
        }

        #[test]
        fn path_following_is_deterministic() {
            let navmesh = NavMesh::from_location(&two_rooms());
            let world = CollisionWorld::new(0.0);
            let capsule = Capsule::new(0.4, 1.0);

            let run = || {
                let brain = approaching();
                let mut follower = PathFollower::default();
                let mut position = Vec3::new(-2.0, capsule.half_extent(), 1.0);
                for _ in 0..300 {
                    position = follow_path(
                        &brain,
                        &mut follower,
                        capsule,
                        position,
                        at(2.0, 18.0),
                        &navmesh,
                        &world,
                    )
                    .position;
                }
                position
            };
            assert_eq!(run(), run());
        }
    }
}
