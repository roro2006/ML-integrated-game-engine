//! Levelling curves (ADR-0018): how much experience each level takes.
//!
//! A class names its curve. Levels are recorded as they are reached
//! (`LeveledUp`), never recomputed from experience on load, so retuning a
//! curve cannot demote anyone; it only changes when the next level comes.

use serde::{Deserialize, Serialize};

use crate::ids::LevelCurveId;

/// Highest level a curve may describe.
pub const MAX_LEVEL: usize = 50;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LevelCurveDef {
    pub schema_version: u32,
    pub id: LevelCurveId,
    /// Total experience to reach level 2, 3, 4, … in order, each more than the
    /// last. Level 1 needs none; the last entry is the level cap.
    pub thresholds: Vec<u32>,
}

impl LevelCurveDef {
    /// The level `experience` reaches on this curve: 1, plus every threshold
    /// passed.
    pub fn level_at(&self, experience: u32) -> u16 {
        let passed = self
            .thresholds
            .iter()
            .take_while(|needed| experience >= **needed)
            .count();
        u16::try_from(1 + passed).unwrap_or(u16::MAX)
    }

    /// The level cap.
    pub fn cap(&self) -> u16 {
        u16::try_from(1 + self.thresholds.len()).unwrap_or(u16::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(thresholds: Vec<u32>) -> LevelCurveDef {
        LevelCurveDef {
            schema_version: 1,
            id: LevelCurveId::new("standard").expect("valid id"),
            thresholds,
        }
    }

    #[test]
    fn levels_come_at_each_threshold_and_stop_at_the_cap() {
        let standard = curve(vec![100, 250, 450]);
        assert_eq!(standard.level_at(0), 1);
        assert_eq!(standard.level_at(99), 1);
        assert_eq!(standard.level_at(100), 2);
        assert_eq!(standard.level_at(449), 3);
        assert_eq!(standard.level_at(450), 4);
        assert_eq!(standard.level_at(u32::MAX), 4, "the cap");
        assert_eq!(standard.cap(), 4);
    }

    #[test]
    fn a_curve_must_climb() {
        assert!(curve(vec![100, 250, 450]).validate().is_ok());
        assert!(curve(Vec::new()).validate().is_err(), "a curve with no levels");
        assert!(curve(vec![100, 100]).validate().is_err(), "a level for nothing");
        assert!(curve(vec![0]).validate().is_err(), "level 2 for free");
        assert!(curve(vec![1; MAX_LEVEL]).validate().is_err(), "past the cap");
    }
}
