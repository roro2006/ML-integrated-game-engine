//! Ordered, single-threaded system execution.

use std::fmt;
use std::time::{Duration, Instant};

use super::command::Commands;
use super::world::World;

type BoxedSystem = Box<dyn FnMut(&mut World, &mut Commands)>;

/// Timing and command statistics for one system.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SystemMetrics {
    pub runs: u64,
    /// Duration of the most recent run, including command application.
    pub last: Duration,
    pub max: Duration,
    pub total: Duration,
    pub skipped_commands: u64,
}

/// Borrowed view of a system's metrics.
#[derive(Debug, Clone, Copy)]
pub struct SystemMetricsView<'a> {
    pub stage: &'static str,
    pub system: &'static str,
    pub metrics: &'a SystemMetrics,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleError {
    DuplicateStage(&'static str),
    UnknownStage(&'static str),
    DuplicateSystem(&'static str),
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateStage(stage) => write!(f, "stage already exists: {stage}"),
            Self::UnknownStage(stage) => write!(f, "unknown stage: {stage}"),
            Self::DuplicateSystem(system) => write!(f, "system name already used: {system}"),
        }
    }
}

impl std::error::Error for ScheduleError {}

struct SystemEntry {
    name: &'static str,
    run: BoxedSystem,
    metrics: SystemMetrics,
}

struct Stage {
    name: &'static str,
    systems: Vec<SystemEntry>,
}

/// An ordered list of stages, each an ordered list of systems.
///
/// Systems run strictly in stage order, then insertion order. Commands a
/// system records are applied immediately after that system returns, so the
/// next system observes them.
pub struct Schedule {
    name: &'static str,
    stages: Vec<Stage>,
    commands: Commands,
}

impl fmt::Debug for Schedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stages: Vec<(&str, usize)> = self
            .stages
            .iter()
            .map(|stage| (stage.name, stage.systems.len()))
            .collect();
        f.debug_struct("Schedule")
            .field("name", &self.name)
            .field("stages", &stages)
            .finish()
    }
}

impl Schedule {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            stages: Vec::new(),
            commands: Commands::new(),
        }
    }

    pub fn with_stages(name: &'static str, stages: &[&'static str]) -> Result<Self, ScheduleError> {
        let mut schedule = Self::new(name);
        for stage in stages {
            schedule.add_stage(stage)?;
        }
        Ok(schedule)
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Appends a stage after all existing stages.
    pub fn add_stage(&mut self, name: &'static str) -> Result<(), ScheduleError> {
        if self.stages.iter().any(|stage| stage.name == name) {
            return Err(ScheduleError::DuplicateStage(name));
        }
        self.stages.push(Stage {
            name,
            systems: Vec::new(),
        });
        Ok(())
    }

    /// Appends a system to the end of `stage`. System names must be unique
    /// within the schedule.
    pub fn add_system(
        &mut self,
        stage: &'static str,
        name: &'static str,
        system: impl FnMut(&mut World, &mut Commands) + 'static,
    ) -> Result<(), ScheduleError> {
        if self
            .stages
            .iter()
            .flat_map(|stage| &stage.systems)
            .any(|entry| entry.name == name)
        {
            return Err(ScheduleError::DuplicateSystem(name));
        }
        let stage = self
            .stages
            .iter_mut()
            .find(|candidate| candidate.name == stage)
            .ok_or(ScheduleError::UnknownStage(stage))?;
        stage.systems.push(SystemEntry {
            name,
            run: Box::new(system),
            metrics: SystemMetrics::default(),
        });
        Ok(())
    }

    pub fn run(&mut self, world: &mut World) {
        for stage in &mut self.stages {
            for system in &mut stage.systems {
                let started = Instant::now();
                (system.run)(world, &mut self.commands);
                let report = self.commands.apply(world);
                let elapsed = started.elapsed();

                let metrics = &mut system.metrics;
                metrics.runs += 1;
                metrics.last = elapsed;
                metrics.max = metrics.max.max(elapsed);
                metrics.total += elapsed;
                metrics.skipped_commands += report.skipped as u64;
            }
        }
    }

    /// Metrics for every system in execution order.
    pub fn metrics(&self) -> impl Iterator<Item = SystemMetricsView<'_>> {
        self.stages.iter().flat_map(|stage| {
            stage.systems.iter().map(|system| SystemMetricsView {
                stage: stage.name,
                system: system.name,
                metrics: &system.metrics,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::super::storage::Component;
    use super::super::world::Resource;
    use super::*;

    #[derive(Default)]
    struct Log(Vec<&'static str>);
    impl Resource for Log {}

    struct Marker;
    impl Component for Marker {}

    fn logger(name: &'static str) -> impl FnMut(&mut World, &mut Commands) {
        move |world, _| world.resource_mut::<Log>().unwrap().0.push(name)
    }

    #[test]
    fn systems_run_in_stage_then_insertion_order() {
        let mut schedule = Schedule::with_stages("test", &["first", "second"]).unwrap();
        schedule.add_system("second", "c", logger("c")).unwrap();
        schedule.add_system("first", "a", logger("a")).unwrap();
        schedule.add_system("second", "d", logger("d")).unwrap();
        schedule.add_system("first", "b", logger("b")).unwrap();

        let mut world = World::new();
        world.insert_resource(Log::default());
        schedule.run(&mut world);
        schedule.run(&mut world);
        assert_eq!(
            world.resource::<Log>().unwrap().0,
            vec!["a", "b", "c", "d", "a", "b", "c", "d"]
        );
        let names: Vec<_> = schedule
            .metrics()
            .map(|m| (m.stage, m.system, m.metrics.runs))
            .collect();
        assert_eq!(
            names,
            vec![
                ("first", "a", 2),
                ("first", "b", 2),
                ("second", "c", 2),
                ("second", "d", 2)
            ]
        );
    }

    #[test]
    fn commands_are_visible_to_the_next_system() {
        let mut schedule = Schedule::with_stages("test", &["update"]).unwrap();
        schedule
            .add_system("update", "spawner", |_, commands| {
                let e = commands.spawn();
                commands.insert(e, Marker);
            })
            .unwrap();
        let seen = Rc::new(RefCell::new(Vec::new()));
        let seen_in_system = Rc::clone(&seen);
        schedule
            .add_system("update", "counter", move |world, _| {
                seen_in_system.borrow_mut().push(world.query::<Marker>().count());
            })
            .unwrap();

        let mut world = World::new();
        schedule.run(&mut world);
        schedule.run(&mut world);
        assert_eq!(*seen.borrow(), vec![1, 2]);
    }

    #[test]
    fn rejects_duplicates_and_unknown_stages() {
        let mut schedule = Schedule::with_stages("test", &["update"]).unwrap();
        assert_eq!(
            schedule.add_stage("update"),
            Err(ScheduleError::DuplicateStage("update"))
        );
        assert_eq!(
            schedule.add_system("missing", "a", |_, _| {}),
            Err(ScheduleError::UnknownStage("missing"))
        );
        schedule.add_system("update", "a", |_, _| {}).unwrap();
        assert_eq!(
            schedule.add_system("update", "a", |_, _| {}),
            Err(ScheduleError::DuplicateSystem("a"))
        );
    }

    #[test]
    fn skipped_commands_are_counted() {
        let mut world = World::new();
        let dead = world.spawn();
        world.despawn(dead);
        let mut schedule = Schedule::with_stages("test", &["update"]).unwrap();
        schedule
            .add_system("update", "stale", move |_, commands| {
                commands.insert(dead, Marker)
            })
            .unwrap();
        schedule.run(&mut world);
        let metrics = schedule.metrics().next().unwrap().metrics.clone();
        assert_eq!(metrics.skipped_commands, 1);
    }
}
