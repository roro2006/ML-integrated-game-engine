//! Schema round-trips, intrinsic validation, and the shipped fixtures.

use super::*;
use crate::ids::{AbilityId, EncounterId, ItemId, NpcId, ObjectiveId, PrefabId, QuestId, QuestResolutionId};

fn objective(name: &str) -> ObjectiveId {
    ObjectiveId::new(name).expect("valid id")
}

fn quest_with(nodes: Vec<(&str, QuestNode)>, start: &str) -> QuestDef {
    QuestDef {
        schema_version: CONTENT_SCHEMA_VERSION,
        id: QuestId::new("still-waters").expect("valid id"),
        title: "Still Waters".to_owned(),
        start: objective(start),
        nodes: nodes
            .into_iter()
            .map(|(id, node)| (objective(id), node))
            .collect(),
        rewards: RewardPackage::default(),
    }
}

fn talk_node(next: &[&str], resolution: Option<&str>) -> QuestNode {
    QuestNode {
        objective: Objective::TalkTo {
            npc: NpcId::new("warden-sela").expect("valid id"),
        },
        next: next.iter().map(|id| objective(id)).collect(),
        resolution: resolution.map(|id| QuestResolutionId::new(id).expect("valid id")),
        text: None,
    }
}

#[test]
fn fixed_point_is_exact_and_round_trips() {
    assert_eq!(Fixed::from_whole(2).as_milli(), 2_000);
    assert_eq!(Fixed::from_milli(1_500).to_string(), "1.500");
    assert_eq!(Fixed::from_whole(3).to_string(), "3");

    let value = Fixed::from_milli(1_234);
    let text = ron::to_string(&value).expect("serializes");
    assert_eq!(text, "1234", "fixed point serializes as a bare integer");
    assert_eq!(ron::from_str::<Fixed>(&text).expect("parses"), value);
}

#[test]
fn ability_round_trips_and_validates() {
    let ability = AbilityDef {
        schema_version: CONTENT_SCHEMA_VERSION,
        id: AbilityId::new("ember-cleave").expect("valid id"),
        name: "Ember Cleave".to_owned(),
        targeting: Targeting::SingleEnemy,
        cost: Fixed::from_whole(20),
        cooldown_ticks: 90,
        windup_ticks: 12,
        recovery_ticks: 24,
        effects: vec![Effect::DealDamage {
            amount: Fixed::from_whole(18),
            damage_type: DamageType::Fire,
        }],
        presentation: Default::default(),
    };
    assert_eq!(ability.validate(), Ok(()));

    let text = ron::to_string(&ability).expect("serializes");
    let parsed: AbilityDef = ron::from_str(&text).expect("parses");
    assert_eq!(parsed, ability);
}

/// Damage to others has one door. A negative health change is allowed only on
/// the caster, where it is a cost rather than an attack.
#[test]
fn negative_health_is_a_cost_on_the_caster_and_nothing_else() {
    let with = |targeting, resource, milli| AbilityDef {
        schema_version: CONTENT_SCHEMA_VERSION,
        id: AbilityId::new("blood-price").expect("valid id"),
        name: "Blood Price".to_owned(),
        targeting,
        cost: Fixed::ZERO,
        cooldown_ticks: 60,
        windup_ticks: 6,
        recovery_ticks: 6,
        effects: vec![Effect::ModifyResource {
            resource,
            amount: Fixed::from_milli(milli),
        }],
        presentation: Default::default(),
    };

    assert_eq!(
        with(Targeting::SelfOnly, ResourceType::Health, -10_000).validate(),
        Ok(()),
        "paying in your own blood is a cost"
    );
    for targeting in [
        Targeting::SingleEnemy,
        Targeting::SingleAlly,
        Targeting::AreaAroundSelf,
        Targeting::AreaAtPoint,
    ] {
        assert_eq!(
            with(targeting, ResourceType::Health, -10_000).validate(),
            Err(ContentError::NegativeHealthOnOthers(
                AbilityId::new("blood-price").expect("valid id")
            )),
            "{targeting:?}: an untyped second door for damage"
        );
    }
    assert_eq!(
        with(Targeting::SingleEnemy, ResourceType::Health, 10_000).validate(),
        Ok(()),
        "healing someone else is not damage"
    );
    assert_eq!(
        with(Targeting::SingleEnemy, ResourceType::Stamina, -10_000).validate(),
        Ok(()),
        "draining stamina is not damage either"
    );
}

// ---- models (ADR-0017) ----------------------------------------------------

const HUMANOID: &str = r#"(
    schema_version: 1,
    id: "humanoid",
    name: "Humanoid",
    slots: [
        (name: "torso", pivot: (x: 0, y: 900, z: 0)),
        (name: "arm-r", pivot: (x: 320, y: 1350, z: 0)),
    ],
)"#;

const WAYFINDER: &str = r#"(
    schema_version: 1,
    id: "wayfinder",
    name: "Wayfinder",
    body_plan: "humanoid",
    palette: "cinderwake",
    parts: [
        (slot: "torso", shape: Box(x: 500, y: 600, z: 300), at: (x: 0, y: 200, z: 0), color: "tunic", top: Some("leather")),
        (slot: "arm-r", shape: Cylinder(radius: 90, height: 600, sides: 6), at: (x: 0, y: -300, z: 0), color: "skin"),
        (slot: "arm-r", shape: Wedge(x: 60, y: 700, z: 120), at: (x: 0, y: -600, z: -300), turn: (x: 90000, y: 0, z: 0), color: "steel"),
    ],
)"#;

fn model() -> ModelDef {
    ron::from_str(WAYFINDER).expect("the model parses as an author writes it")
}

#[test]
fn models_parse_as_authored_and_validate() {
    let plan: BodyPlanDef = ron::from_str(HUMANOID).expect("the body plan parses");
    assert_eq!(plan.validate(), Ok(()));

    let wayfinder = model();
    assert_eq!(wayfinder.validate(), Ok(()));
    assert_eq!(
        wayfinder.scale,
        Fixed::from_whole(1),
        "an absent scale means as authored"
    );
    assert_eq!(
        wayfinder.parts[1].top, None,
        "an absent top colour means the side colour"
    );

    let text = ron::to_string(&wayfinder).expect("serializes");
    assert_eq!(ron::from_str::<ModelDef>(&text).expect("parses"), wayfinder);
}

#[test]
fn shape_triangle_counts_follow_their_formulas() {
    let f = Fixed::from_milli;
    assert_eq!(
        ShapeDef::Box {
            x: f(1),
            y: f(1),
            z: f(1)
        }
        .triangles(),
        12
    );
    assert_eq!(
        ShapeDef::Wedge {
            x: f(1),
            y: f(1),
            z: f(1)
        }
        .triangles(),
        8
    );
    assert_eq!(
        ShapeDef::Cylinder {
            radius: f(1),
            height: f(1),
            sides: 8
        }
        .triangles(),
        28
    );
    assert_eq!(
        ShapeDef::Cone {
            radius: f(1),
            height: f(1),
            sides: 8
        }
        .triangles(),
        14
    );
    assert_eq!(
        ShapeDef::Frustum {
            bottom: f(1),
            top: f(0),
            height: f(1),
            sides: 8
        }
        .triangles(),
        14,
        "a frustum with no top is a cone"
    );
    assert_eq!(
        ShapeDef::Frustum {
            bottom: f(1),
            top: f(1),
            height: f(1),
            sides: 8
        }
        .triangles(),
        28
    );
    assert_eq!(
        ShapeDef::Sphere {
            radius: f(1),
            detail: 0
        }
        .triangles(),
        20
    );
    assert_eq!(
        ShapeDef::Sphere {
            radius: f(1),
            detail: 1
        }
        .triangles(),
        80
    );
    assert_eq!(model().triangles(), 12 + 20 + 8);
}

/// Each budget and range must refuse, not merely exist.
#[test]
fn models_that_break_a_bound_are_refused() {
    let with = |change: &dyn Fn(&mut ModelDef)| {
        let mut model = model();
        change(&mut model);
        model.validate()
    };
    let over = |error: Result<(), ContentError>, expected: &str| match error {
        Err(ContentError::OverBudget { what, .. }) => assert_eq!(what, expected),
        other => panic!("expected an over-budget {expected}, got {other:?}"),
    };
    let out = |error: Result<(), ContentError>, expected: &str| match error {
        Err(ContentError::OutOfRange { what, .. }) => assert_eq!(what, expected),
        other => panic!("expected {expected} out of range, got {other:?}"),
    };
    let f = Fixed::from_milli;

    assert!(matches!(
        with(&|m| m.parts.clear()),
        Err(ContentError::EmptyVisual { what: "parts", .. })
    ));
    over(
        with(&|m| {
            let part = m.parts[0].clone();
            m.parts = vec![part; MAX_PARTS + 1];
        }),
        "parts",
    );
    over(
        with(&|m| {
            // Nineteen 80-face spheres fit the part budget and not the
            // triangle one.
            m.parts[0].shape = ShapeDef::Sphere {
                radius: f(100),
                detail: 1,
            };
            let sphere = m.parts[0].clone();
            m.parts = vec![sphere; 19];
        }),
        "triangles",
    );
    out(
        with(&|m| {
            m.parts[0].shape = ShapeDef::Box {
                x: Fixed::ZERO,
                y: f(1),
                z: f(1),
            }
        }),
        "shape width",
    );
    out(
        with(&|m| {
            m.parts[1].shape = ShapeDef::Cylinder {
                radius: f(90),
                height: f(600),
                sides: 2,
            }
        }),
        "shape sides",
    );
    out(
        with(&|m| {
            m.parts[1].shape = ShapeDef::Cylinder {
                radius: f(90),
                height: f(600),
                sides: MAX_SIDES + 1,
            }
        }),
        "shape sides",
    );
    out(
        with(&|m| {
            m.parts[0].shape = ShapeDef::Sphere {
                radius: f(100),
                detail: MAX_SPHERE_DETAIL + 1,
            }
        }),
        "sphere detail",
    );
    out(
        with(&|m| {
            m.parts[0].shape = ShapeDef::Box {
                x: Fixed::from_whole(MAX_EXTENT_METRES + 1),
                y: f(1),
                z: f(1),
            }
        }),
        "shape width",
    );
    out(
        with(&|m| m.parts[0].at.x = Fixed::from_whole(MAX_OFFSET_METRES + 1)),
        "part offset",
    );
    out(
        with(&|m| m.parts[0].turn.y = Fixed::from_whole(MAX_TURN_DEGREES + 1)),
        "part turn",
    );
    out(
        with(&|m| m.scale = Fixed::from_milli(MIN_SCALE_MILLI - 1)),
        "model scale",
    );
    out(
        with(&|m| m.scale = Fixed::from_milli(MAX_SCALE_MILLI + 1)),
        "model scale",
    );

    assert_eq!(
        with(&|m| m.parts[0].shape = ShapeDef::Frustum {
            bottom: f(200),
            top: Fixed::ZERO,
            height: f(300),
            sides: 6
        }),
        Ok(()),
        "a zero-topped frustum is a cone, not a mistake"
    );
}

/// The OSRS-style additions: tapers, domes and stretch, each bounded.
#[test]
fn tapers_domes_and_stretch_validate_and_are_bounded() {
    let f = Fixed::from_milli;
    let with = |change: &dyn Fn(&mut ModelDef)| {
        let mut model = model();
        change(&mut model);
        model.validate()
    };
    assert_eq!(
        model().parts[0].stretch,
        Point::new(f(1000), f(1000), f(1000)),
        "absent means unstretched"
    );

    assert_eq!(
        with(&|m| m.parts[0].shape = ShapeDef::Taper {
            x: f(380),
            z: f(240),
            top_x: f(500),
            top_z: f(300),
            height: f(560)
        }),
        Ok(())
    );
    assert_eq!(
        with(&|m| m.parts[0].shape = ShapeDef::Taper {
            x: f(80),
            z: f(80),
            top_x: Fixed::ZERO,
            top_z: Fixed::ZERO,
            height: f(300)
        }),
        Ok(()),
        "a spike is a taper with no top"
    );
    assert!(
        matches!(
            with(&|m| m.parts[0].shape = ShapeDef::Taper {
                x: Fixed::ZERO,
                z: f(80),
                top_x: f(80),
                top_z: f(80),
                height: f(300)
            }),
            Err(ContentError::OutOfRange {
                what: "taper width",
                ..
            })
        ),
        "the bottom must have area"
    );

    assert_eq!(
        with(&|m| m.parts[0].shape = ShapeDef::Dome {
            radius: f(150),
            segments: 6,
            rings: 2
        }),
        Ok(())
    );
    assert!(matches!(
        with(&|m| m.parts[0].shape = ShapeDef::Dome {
            radius: f(150),
            segments: 6,
            rings: 0
        }),
        Err(ContentError::OutOfRange {
            what: "dome rings",
            ..
        })
    ));
    assert!(matches!(
        with(&|m| m.parts[0].shape = ShapeDef::Dome {
            radius: f(150),
            segments: 6,
            rings: MAX_DOME_RINGS + 1
        }),
        Err(ContentError::OutOfRange {
            what: "dome rings",
            ..
        })
    ));

    assert_eq!(
        with(&|m| m.parts[0].stretch = Point::new(f(1000), f(600), f(1200))),
        Ok(())
    );
    for bad in [
        f(MIN_STRETCH_MILLI - 1),
        f(MAX_STRETCH_MILLI + 1),
        Fixed::ZERO,
        f(-1000),
    ] {
        assert!(
            matches!(
                with(&|m| m.parts[0].stretch.y = bad),
                Err(ContentError::OutOfRange {
                    what: "part stretch",
                    ..
                })
            ),
            "{bad:?}: a stretch cannot flatten a shape to nothing or blow it up"
        );
    }

    let taper = |top_x, top_z| ShapeDef::Taper {
        x: f(300),
        z: f(300),
        top_x: f(top_x),
        top_z: f(top_z),
        height: f(300),
    };
    assert_eq!(taper(200, 200).triangles(), 12);
    assert_eq!(taper(0, 200).triangles(), 8);
    assert_eq!(taper(0, 0).triangles(), 6);
    assert_eq!(
        ShapeDef::Dome {
            radius: f(1),
            segments: 8,
            rings: 3
        }
        .triangles(),
        8 + 32 + 6
    );
}

const SWING: &str = r#"(
    schema_version: 1,
    id: "humanoid-swing",
    name: "Swing",
    timing: Windows,
    tracks: [
        (slot: "arm-r", keys: [(at: 0, turn: (x: 0, y: 0, z: 0)), (at: 900, turn: (x: 150000, y: 0, z: 0)), (at: 3000)]),
    ],
)"#;

#[test]
fn clips_parse_as_authored_and_their_rules_hold() {
    let swing: ClipDef = ron::from_str(SWING).expect("a clip parses as an author writes it");
    assert_eq!(swing.validate(), Ok(()));
    assert_eq!(
        swing.tracks[0].keys[2].scale,
        Point::new(Fixed::from_whole(1), Fixed::from_whole(1), Fixed::from_whole(1)),
        "an absent scale is unscaled"
    );

    let with = |change: &dyn Fn(&mut ClipDef)| {
        let mut clip = swing.clone();
        change(&mut clip);
        clip.validate()
    };
    assert!(
        matches!(
            with(&|c| c.tracks[0].keys.swap(1, 2)),
            Err(ContentError::KeysOutOfOrder { .. })
        ),
        "keys must run forward in time"
    );
    assert!(
        matches!(
            with(&|c| c.tracks.push(c.tracks[0].clone())),
            Err(ContentError::DuplicateTrack { .. })
        ),
        "two tracks for one slot is ambiguous"
    );
    assert!(
        matches!(
            with(&|c| c.tracks[0].keys[2].at = Fixed::from_milli(WINDOWS_END_MILLI + 1)),
            Err(ContentError::OutOfRange { what: "key time", .. })
        ),
        "a key past the end of the windows never plays"
    );
    assert!(
        matches!(
            with(&|c| {
                c.timing = ClipTiming::Span;
                c.tracks[0].keys[2].at = Fixed::from_milli(CYCLE_END_MILLI + 1);
            }),
            Err(ContentError::OutOfRange { what: "key time", .. })
        ),
        "a span runs 0 to 1000"
    );
    assert!(matches!(
        with(&|c| c.timing = ClipTiming::Loop { ticks: 0 }),
        Err(ContentError::OutOfRange {
            what: "loop length",
            ..
        })
    ));
    assert!(matches!(
        with(&|c| c.timing = ClipTiming::Stride {
            metres: Fixed::from_milli(MIN_STRIDE_MILLI - 1)
        }),
        Err(ContentError::OutOfRange {
            what: "stride length",
            ..
        })
    ));
    assert!(matches!(
        with(&|c| c.tracks[0].keys[1].offset.y = Fixed::from_whole(MAX_KEY_OFFSET_METRES + 1)),
        Err(ContentError::OutOfRange {
            what: "key offset",
            ..
        })
    ));
    assert!(matches!(
        with(&|c| c.tracks.clear()),
        Err(ContentError::EmptyVisual { what: "tracks", .. })
    ));
}

#[test]
fn a_model_fills_its_animation_gaps_from_its_body_plan() {
    let clip = |name: &str| Some(game_core_clip(name));
    let plan = AnimationSet {
        idle: clip("humanoid-idle"),
        walk: clip("humanoid-walk"),
        ..AnimationSet::default()
    };
    let husk = AnimationSet {
        walk: clip("husk-shamble"),
        ..AnimationSet::default()
    };
    let resolved = husk.or(&plan);
    assert_eq!(resolved.walk, clip("husk-shamble"), "the model's own choice wins");
    assert_eq!(
        resolved.idle,
        clip("humanoid-idle"),
        "and the plan fills the rest"
    );
    assert_eq!(resolved.swing, None, "a state nobody names stays empty");

    assert!(timing_suits(
        "walk",
        ClipTiming::Stride {
            metres: Fixed::from_whole(1)
        }
    ));
    assert!(
        !timing_suits("walk", ClipTiming::Windows),
        "a walk on attack windows would never advance"
    );
    assert!(timing_suits("swing", ClipTiming::Windows));
    assert!(!timing_suits("idle", ClipTiming::Span));
}

fn game_core_clip(name: &str) -> crate::ids::ClipId {
    crate::ids::ClipId::new(name).expect("valid id")
}

#[test]
fn body_plans_and_palettes_are_checked() {
    let mut plan: BodyPlanDef = ron::from_str(HUMANOID).expect("parses");
    plan.slots.push(plan.slots[0].clone());
    assert!(
        matches!(plan.validate(), Err(ContentError::DuplicateSlot { .. })),
        "a slot named twice"
    );

    let mut plan: BodyPlanDef = ron::from_str(HUMANOID).expect("parses");
    plan.slots.clear();
    assert!(matches!(
        plan.validate(),
        Err(ContentError::EmptyVisual { what: "slots", .. })
    ));

    let palette: PaletteDef = ron::from_str(
        r#"(schema_version: 1, id: "cinderwake", name: "Cinderwake", swatches: { "skin": (224, 172, 140), "tunic": (120, 60, 40) })"#,
    )
    .expect("a palette parses as an author writes it");
    assert_eq!(palette.validate(), Ok(()));

    let mut empty = palette.clone();
    empty.swatches.clear();
    assert!(matches!(
        empty.validate(),
        Err(ContentError::EmptyVisual { what: "swatches", .. })
    ));
}

#[test]
fn an_unknown_model_field_is_rejected() {
    let text = WAYFINDER.replace("name: \"Wayfinder\",", "name: \"Wayfinder\", glow: true,");
    assert!(
        ron::from_str::<ModelDef>(&text).is_err(),
        "a field nobody defined must not pass silently"
    );
}

#[test]
fn unknown_fields_are_rejected() {
    let text = r#"(schema_version: 1, id: "iron-ore", name: "Iron Ore",
             category: material, stack_limit: 99, colour: "grey")"#;
    let parsed = ron::from_str::<ItemDef>(text);
    assert!(parsed.is_err(), "a misspelled or stray field must not be ignored");
}

#[test]
fn intrinsic_validation_rejects_nonsense() {
    let item = ItemDef {
        schema_version: CONTENT_SCHEMA_VERSION,
        id: ItemId::new("iron-ore").expect("valid id"),
        name: "Iron Ore".to_owned(),
        category: ItemCategory::Material,
        stack_limit: 0,
        combat: None,
        on_use: None,
    };
    assert!(matches!(item.validate(), Err(ContentError::StackLimitZero(_))));

    let encounter = EncounterDef {
        schema_version: CONTENT_SCHEMA_VERSION,
        id: EncounterId::new("waterworks-ambush").expect("valid id"),
        name: "Waterworks Ambush".to_owned(),
        threat_budget: Fixed::from_milli(-1),
        slots: vec![EncounterSlot {
            enemy: PrefabId::new("corrupted-husk").expect("valid id"),
            count: 2,
        }],
        arrival: None,
    };
    assert!(matches!(
        encounter.validate(),
        Err(ContentError::NegativeValue { .. })
    ));
}

/// Only a consumable may be used, only for what acts on the user, and only
/// for a bounded moment (M9).
#[test]
fn an_item_use_is_bounded_and_self_directed() {
    let heal = Effect::ModifyResource {
        resource: ResourceType::Health,
        amount: Fixed::from_whole(30),
    };
    let draught = |category, effects: Vec<Effect>, use_ticks| ItemDef {
        schema_version: CONTENT_SCHEMA_VERSION,
        id: ItemId::new("clearwater-draught").expect("valid id"),
        name: "Clearwater Draught".to_owned(),
        category,
        stack_limit: 10,
        combat: None,
        on_use: Some(ItemUse { effects, use_ticks }),
    };
    assert_eq!(
        draught(ItemCategory::Consumable, vec![heal.clone()], 30).validate(),
        Ok(())
    );
    assert!(matches!(
        draught(ItemCategory::Material, vec![heal.clone()], 30).validate(),
        Err(ContentError::UseOnNonConsumable(_))
    ));
    assert!(
        draught(ItemCategory::Consumable, Vec::new(), 30)
            .validate()
            .is_err()
    );
    assert!(
        draught(ItemCategory::Consumable, vec![heal.clone()], 0)
            .validate()
            .is_err()
    );
    assert!(
        draught(ItemCategory::Consumable, vec![heal.clone()], MAX_USE_TICKS + 1)
            .validate()
            .is_err()
    );
    assert!(
        draught(
            ItemCategory::Consumable,
            vec![heal.clone(); MAX_USE_EFFECTS + 1],
            30
        )
        .validate()
        .is_err()
    );
    for weapon in [
        Effect::DealDamage {
            amount: Fixed::from_whole(50),
            damage_type: DamageType::Fire,
        },
        Effect::SpawnEntity {
            prefab: PrefabId::new("corrupted-husk").expect("valid id"),
            count: 1,
        },
        Effect::Teleport {
            max_distance: Fixed::from_whole(5),
        },
    ] {
        assert!(
            matches!(
                draught(ItemCategory::Consumable, vec![weapon.clone()], 30).validate(),
                Err(ContentError::NotAllowedOnUse { .. })
            ),
            "{weapon:?} must not run from an item"
        );
    }
}

#[test]
fn wrong_schema_version_is_rejected() {
    let mut item = ItemDef {
        schema_version: CONTENT_SCHEMA_VERSION + 1,
        id: ItemId::new("iron-ore").expect("valid id"),
        name: "Iron Ore".to_owned(),
        category: ItemCategory::Material,
        stack_limit: 10,
        combat: None,
        on_use: None,
    };
    assert!(matches!(item.validate(), Err(ContentError::SchemaVersion { .. })));
    item.schema_version = CONTENT_SCHEMA_VERSION;
    assert_eq!(item.validate(), Ok(()));
}

#[test]
fn quest_graph_must_be_playable() {
    let good = quest_with(
        vec![
            ("meet-sela", talk_node(&["secure-works", "seek-order"], None)),
            ("secure-works", talk_node(&[], Some("settlers-favoured"))),
            ("seek-order", talk_node(&[], Some("order-favoured"))),
        ],
        "meet-sela",
    );
    assert_eq!(
        good.validate(),
        Ok(()),
        "a branching quest with two endings is valid"
    );

    let mut missing_start = good.clone();
    missing_start.start = objective("nowhere");
    assert!(matches!(
        missing_start.validate(),
        Err(ContentError::MissingStartNode { .. })
    ));

    let dangling = quest_with(vec![("meet-sela", talk_node(&["ghost-node"], None))], "meet-sela");
    assert!(matches!(
        dangling.validate(),
        Err(ContentError::DanglingNode { .. })
    ));

    let orphaned = quest_with(
        vec![
            ("meet-sela", talk_node(&[], Some("settlers-favoured"))),
            ("forgotten", talk_node(&[], Some("order-favoured"))),
        ],
        "meet-sela",
    );
    assert!(matches!(
        orphaned.validate(),
        Err(ContentError::UnreachableNode { .. })
    ));

    let endless = quest_with(vec![("meet-sela", talk_node(&["meet-sela"], None))], "meet-sela");
    assert!(matches!(endless.validate(), Err(ContentError::NoCompletion(_))));

    let terminal = quest_with(vec![("meet-sela", talk_node(&[], None))], "meet-sela");
    assert!(matches!(
        terminal.validate(),
        Err(ContentError::TerminalWithoutResolution { .. })
    ));
}

#[test]
fn shipped_fixtures_parse_and_validate() {
    let items: Vec<ItemDef> =
        ron::from_str(include_str!("../../../../assets/content/items.ron")).expect("items.ron parses");
    assert!(items.len() >= 5, "the slice promises five gatherable resources");
    for item in &items {
        item.validate().expect("fixture item is valid");
    }

    let recipes: Vec<RecipeDef> =
        ron::from_str(include_str!("../../../../assets/content/recipes.ron")).expect("recipes.ron parses");
    assert_eq!(recipes.len(), 3, "the slice promises three craftable recipes");
    for recipe in &recipes {
        recipe.validate().expect("fixture recipe is valid");
    }

    let abilities: Vec<AbilityDef> = ron::from_str(include_str!("../../../../assets/content/abilities.ron"))
        .expect("abilities.ron parses");
    for ability in &abilities {
        ability.validate().expect("fixture ability is valid");
    }

    let enemies: Vec<EnemyDef> =
        ron::from_str(include_str!("../../../../assets/content/enemies.ron")).expect("enemies.ron parses");
    assert_eq!(enemies.len(), 4, "three regular enemies and one elite");
    assert!(
        enemies.iter().any(|enemy| enemy.variant_of.is_some()),
        "the elite should be a variant rather than a duplicated definition"
    );
    for enemy in &enemies {
        enemy.validate().expect("fixture enemy is valid");
    }

    let encounters: Vec<EncounterDef> =
        ron::from_str(include_str!("../../../../assets/content/encounters.ron"))
            .expect("encounters.ron parses");
    for encounter in &encounters {
        encounter.validate().expect("fixture encounter is valid");
    }

    let quests: Vec<QuestDef> =
        ron::from_str(include_str!("../../../../assets/content/quests.ron")).expect("quests.ron parses");
    for quest in &quests {
        quest.validate().expect("fixture quest graph is playable");
    }
    assert!(
        quests
            .iter()
            .any(|quest| quest.nodes.values().any(|node| node.next.len() > 1)),
        "the slice quest must actually branch"
    );
}

#[test]
fn effects_reject_zero_spawn_counts() {
    let ability = AbilityDef {
        schema_version: CONTENT_SCHEMA_VERSION,
        id: AbilityId::new("call-husks").expect("valid id"),
        name: "Call Husks".to_owned(),
        targeting: Targeting::AreaAroundSelf,
        cost: Fixed::from_whole(30),
        cooldown_ticks: 300,
        windup_ticks: 30,
        recovery_ticks: 30,
        effects: vec![Effect::SpawnEntity {
            prefab: PrefabId::new("corrupted-husk").expect("valid id"),
            count: 0,
        }],
        presentation: Default::default(),
    };
    assert!(matches!(ability.validate(), Err(ContentError::ZeroCount { .. })));
}

// ---- effects (M7 wave 4) ----------------------------------------------------

fn vfx(parts: Vec<VfxPart>) -> VfxDef {
    VfxDef {
        schema_version: CONTENT_SCHEMA_VERSION,
        id: crate::ids::VfxId::new("ember-lash").expect("valid id"),
        name: "Ember lash".to_owned(),
        palette: crate::ids::PaletteId::new("cinderwake").expect("valid id"),
        parts,
    }
}

fn swatch() -> crate::ids::SwatchId {
    crate::ids::SwatchId::new("ember").expect("valid id")
}

fn shards(count: u8) -> VfxPart {
    VfxPart::Burst {
        color: swatch(),
        count,
        size: Fixed::from_milli(120),
        speed: Fixed::from_milli(4_500),
        gravity: Fixed::from_milli(12_000),
        ticks: 28,
    }
}

#[test]
fn effects_parse_as_authored_and_validate() {
    let text = r#"(
        schema_version: 1,
        id: "ember-lash",
        name: "Ember lash",
        palette: "cinderwake",
        parts: [
            Beam(color: "ember", width: 160, ticks: 12),
            Burst(color: "ember", count: 16, size: 120, speed: 4500, ticks: 28),
            Pulse(color: "eye", radius: 1400, ticks: 16),
        ],
    )"#;
    let parsed: VfxDef = ron::from_str(text).expect("parses");
    assert_eq!(parsed.validate(), Ok(()));
    assert_eq!(parsed.particles(), 1 + 16 + 1);
    assert_eq!(parsed.ticks(), 28, "an effect lasts as long as its longest part");
    // Gravity is optional: absent, embers drift.
    assert!(matches!(parsed.parts[1], VfxPart::Burst { gravity, .. } if gravity == Fixed::ZERO));
}

#[test]
fn effects_that_break_a_bound_are_refused() {
    assert!(matches!(
        vfx(Vec::new()).validate(),
        Err(ContentError::EmptyVisual { what: "parts", .. })
    ));
    assert!(matches!(
        vfx(vec![shards(1); MAX_VFX_PARTS + 1]).validate(),
        Err(ContentError::OverBudget { what: "parts", .. })
    ));
    assert!(matches!(
        vfx(vec![shards(MAX_BURST_SHARDS + 1)]).validate(),
        Err(ContentError::OverBudget { .. })
    ));
    assert_eq!(
        vfx(vec![shards(0)]).validate(),
        Err(ContentError::ZeroCount { what: "burst shards" })
    );
    assert_eq!(
        vfx(vec![shards(MAX_BURST_SHARDS)]).validate(),
        Ok(()),
        "the cap itself is allowed"
    );

    let long = VfxPart::Pulse {
        color: swatch(),
        radius: Fixed::from_whole(2),
        ticks: MAX_VFX_TICKS + 1,
    };
    assert!(matches!(
        vfx(vec![long]).validate(),
        Err(ContentError::OutOfRange {
            what: "effect length",
            ..
        })
    ));
    let wide = VfxPart::Pulse {
        color: swatch(),
        radius: Fixed::from_milli(MAX_PULSE_MILLI + 1),
        ticks: 10,
    };
    assert!(matches!(
        vfx(vec![wide]).validate(),
        Err(ContentError::OutOfRange {
            what: "pulse radius",
            ..
        })
    ));
    let fat = VfxPart::Beam {
        color: swatch(),
        width: Fixed::from_milli(MAX_BEAM_MILLI + 1),
        ticks: 10,
    };
    assert!(matches!(
        vfx(vec![fat]).validate(),
        Err(ContentError::OutOfRange {
            what: "beam width",
            ..
        })
    ));
    let fast = VfxPart::Burst {
        color: swatch(),
        count: 4,
        size: Fixed::from_milli(100),
        speed: Fixed::from_whole(MAX_SHARD_SPEED_METRES + 1),
        gravity: Fixed::ZERO,
        ticks: 10,
    };
    assert!(matches!(
        vfx(vec![fast]).validate(),
        Err(ContentError::OutOfRange {
            what: "shard speed",
            ..
        })
    ));
}

/// A telegraph's shape is read from targeting, so only an area has ground to
/// mark: a single target is chosen when the ability lands.
#[test]
fn only_an_area_ability_may_ask_for_a_telegraph() {
    let with = |targeting| AbilityDef {
        schema_version: CONTENT_SCHEMA_VERSION,
        id: AbilityId::new("ember-ring").expect("valid id"),
        name: "Ember Ring".to_owned(),
        targeting,
        cost: Fixed::ZERO,
        cooldown_ticks: 60,
        windup_ticks: 20,
        recovery_ticks: 10,
        effects: vec![Effect::DealDamage {
            amount: Fixed::from_whole(5),
            damage_type: DamageType::Fire,
        }],
        presentation: AbilityPresentation {
            telegraph: true,
            ..AbilityPresentation::default()
        },
    };
    for targeting in [Targeting::AreaAroundSelf, Targeting::AreaAtPoint] {
        assert_eq!(
            with(targeting).validate(),
            Ok(()),
            "{targeting:?} has ground to mark"
        );
    }
    for targeting in [Targeting::SelfOnly, Targeting::SingleEnemy, Targeting::SingleAlly] {
        assert_eq!(
            with(targeting).validate(),
            Err(ContentError::TelegraphWithoutArea(
                AbilityId::new("ember-ring").expect("valid id")
            )),
            "{targeting:?} has no ground to mark in advance"
        );
    }
}

#[test]
fn an_ability_without_presentation_still_parses() {
    let text = r#"(
        schema_version: 1,
        id: "ember-cleave",
        name: "Ember Cleave",
        targeting: SingleEnemy,
        cost: 20000,
        cooldown_ticks: 90,
        windup_ticks: 14,
        recovery_ticks: 26,
        effects: [DealDamage(amount: 18000, damage_type: Fire)],
    )"#;
    let parsed: AbilityDef = ron::from_str(text).expect("presentation is optional");
    assert_eq!(parsed.presentation, AbilityPresentation::default());
}

// ---- skillshots (M7 wave 5) ---------------------------------------------------

fn bolt(effect: Effect) -> AbilityDef {
    AbilityDef {
        schema_version: CONTENT_SCHEMA_VERSION,
        id: AbilityId::new("ember-bolt").expect("valid id"),
        name: "Ember Bolt".to_owned(),
        targeting: Targeting::SelfOnly,
        cost: Fixed::from_whole(15),
        cooldown_ticks: 75,
        windup_ticks: 12,
        recovery_ticks: 14,
        effects: vec![effect],
        presentation: AbilityPresentation::default(),
    }
}

fn fire(amount: i64) -> Effect {
    Effect::DealDamage {
        amount: Fixed::from_whole(amount),
        damage_type: DamageType::Fire,
    }
}

fn projectile(speed: i64, range: i64, radius: i64, pierce: u8, on_hit: Vec<Effect>) -> Effect {
    Effect::Projectile {
        speed: Fixed::from_milli(speed),
        range: Fixed::from_milli(range),
        radius: Fixed::from_milli(radius),
        pierce,
        on_hit,
    }
}

#[test]
fn a_projectile_parses_as_authored_and_validates() {
    let text = r#"(
        schema_version: 1,
        id: "ember-bolt",
        name: "Ember Bolt",
        targeting: SelfOnly,
        cost: 15000,
        cooldown_ticks: 75,
        windup_ticks: 12,
        recovery_ticks: 14,
        effects: [
            Projectile(speed: 16000, range: 14000, radius: 250, on_hit: [DealDamage(amount: 14000, damage_type: Fire)]),
        ],
        presentation: (telegraph: true),
    )"#;
    let parsed: AbilityDef = ron::from_str(text).expect("parses");
    assert_eq!(parsed.validate(), Ok(()));
    assert!(
        matches!(parsed.projectile(), Some(Effect::Projectile { pierce: 0, .. })),
        "pierce is optional and stops at the first target"
    );
}

#[test]
fn a_projectile_out_of_bounds_is_refused() {
    let hit = || vec![fire(10)];
    let what = |effect| match bolt(effect).validate() {
        Err(ContentError::OutOfRange { what, .. }) => what,
        other => panic!("expected an out-of-range error, got {other:?}"),
    };
    let top_speed = MAX_PROJECTILE_SPEED_METRES * 1000;
    assert_eq!(
        what(projectile(top_speed + 1, 10_000, 250, 0, hit())),
        "projectile speed"
    );
    assert_eq!(
        what(projectile(
            MIN_PROJECTILE_SPEED_METRES * 1000 - 1,
            10_000,
            250,
            0,
            hit()
        )),
        "projectile speed"
    );
    assert_eq!(
        what(projectile(
            16_000,
            MAX_PROJECTILE_RANGE_METRES * 1000 + 1,
            250,
            0,
            hit()
        )),
        "projectile range"
    );
    assert_eq!(what(projectile(16_000, 0, 250, 0, hit())), "projectile range");
    assert_eq!(
        what(projectile(
            16_000,
            10_000,
            MAX_PROJECTILE_RADIUS_MILLI + 1,
            0,
            hit()
        )),
        "projectile radius"
    );
    assert!(matches!(
        bolt(projectile(16_000, 10_000, 250, MAX_PIERCE + 1, hit())).validate(),
        Err(ContentError::OverBudget {
            what: "pierced targets",
            ..
        })
    ));
    assert_eq!(
        bolt(projectile(top_speed, 10_000, 250, MAX_PIERCE, hit())).validate(),
        Ok(()),
        "every cap itself is allowed"
    );
    assert!(matches!(
        bolt(projectile(
            16_000,
            10_000,
            250,
            0,
            vec![fire(1); MAX_ON_HIT_EFFECTS + 1]
        ))
        .validate(),
        Err(ContentError::OverBudget {
            what: "on-hit effects",
            ..
        })
    ));
    assert_eq!(
        bolt(projectile(16_000, 10_000, 250, 0, Vec::new())).validate(),
        Err(ContentError::EmptyOnHit(
            AbilityId::new("ember-bolt").expect("valid id")
        ))
    );
}

/// `on_hit` is one level deep: nothing in it may launch, summon, or move the
/// caster.
#[test]
fn on_hit_holds_only_what_acts_on_the_struck() {
    let id = AbilityId::new("ember-bolt").expect("valid id");
    let nested = [
        projectile(16_000, 10_000, 250, 0, vec![fire(1)]),
        Effect::SpawnEntity {
            prefab: PrefabId::new("corrupted-husk").expect("valid id"),
            count: 1,
        },
        Effect::Teleport {
            max_distance: Fixed::from_whole(3),
        },
    ];
    for inner in nested {
        let kind = inner.kind();
        assert_eq!(
            bolt(projectile(16_000, 10_000, 250, 0, vec![fire(5), inner])).validate(),
            Err(ContentError::NotAllowedOnHit {
                ability: id.clone(),
                effect: kind,
            }),
            "{kind} may not run on a hit"
        );
    }
    // What a projectile strikes is never its caster, so the blood-price
    // exception does not reach inside it, even on a self-targeted ability.
    let drain = Effect::ModifyResource {
        resource: ResourceType::Health,
        amount: Fixed::from_whole(-5),
    };
    assert_eq!(
        bolt(projectile(16_000, 10_000, 250, 0, vec![drain])).validate(),
        Err(ContentError::NegativeHealthOnOthers(id.clone()))
    );
    // And the rules for damage still hold inside it.
    assert!(matches!(
        bolt(projectile(16_000, 10_000, 250, 0, vec![fire(-3)])).validate(),
        Err(ContentError::NegativeValue { what: "damage", .. })
    ));
}

#[test]
fn a_projectile_may_be_telegraphed_and_its_look_needs_one() {
    let id = AbilityId::new("ember-bolt").expect("valid id");
    let mut lane = bolt(projectile(16_000, 10_000, 250, 0, vec![fire(10)]));
    lane.presentation.telegraph = true;
    assert_eq!(
        lane.validate(),
        Ok(()),
        "a projectile has a lane to mark, whatever its targeting"
    );

    let mut hollow = bolt(fire(10));
    hollow.presentation.projectile = Some(crate::ids::ModelId::new("ember-bolt").expect("valid id"));
    assert_eq!(
        hollow.validate(),
        Err(ContentError::ProjectileLookWithoutProjectile(id.clone()))
    );
    let mut trailing = bolt(fire(10));
    trailing.presentation.trail = Some(crate::ids::VfxId::new("ember-trail").expect("valid id"));
    assert_eq!(
        trailing.validate(),
        Err(ContentError::ProjectileLookWithoutProjectile(id))
    );
}

// ---- enemy tuning (M8 wave 1) -------------------------------------------------

fn lurker(tuning: &str) -> EnemyDef {
    ron::from_str(&format!(
        r#"(
            schema_version: 1,
            id: "ash-lurker",
            name: "Ash Lurker",
            behavior: Melee,
            max_health: 50000,
            damage: 6000,
            abilities: [],
            visual_tags: [],
            threat: 1100,
            variant_of: None,
            tuning: {tuning},
        )"#
    ))
    .expect("parses")
}

#[test]
fn tuning_is_optional_field_by_field() {
    let silent: EnemyDef = ron::from_str(
        r#"(
            schema_version: 1,
            id: "corrupted-husk",
            name: "Corrupted Husk",
            behavior: Melee,
            max_health: 60000,
            damage: 8000,
            abilities: [],
            visual_tags: [],
            threat: 1000,
            variant_of: None,
        )"#,
    )
    .expect("an enemy with no tuning still parses");
    assert_eq!(silent.tuning, EnemyTuning::default());
    assert_eq!(silent.validate(), Ok(()));

    let partial = lurker("(move_speed: Some(6000))");
    assert_eq!(partial.validate(), Ok(()));
    assert_eq!(partial.tuning.move_speed, Some(Fixed::from_whole(6)));
    assert_eq!(
        partial.tuning.acquire, None,
        "what it leaves out stays the template's"
    );
}

#[test]
fn release_inside_acquire_is_refused() {
    assert_eq!(
        lurker("(acquire: Some(12000), release: Some(8000))").validate(),
        Err(ContentError::ReleaseInsideAcquire(
            PrefabId::new("ash-lurker").expect("valid id")
        ))
    );
    assert_eq!(
        lurker("(acquire: Some(12000), release: Some(12000))").validate(),
        Ok(()),
        "equal is allowed: no gap, but no inversion either"
    );
}

#[test]
fn tuning_out_of_range_is_refused() {
    let what = |tuning: &str| match lurker(tuning).validate() {
        Err(ContentError::OutOfRange { what, .. }) => what,
        other => panic!("{tuning}: expected out of range, got {other:?}"),
    };
    assert_eq!(what("(acquire: Some(31000))"), "aggro acquire");
    assert_eq!(what("(preferred_range: Some(21000))"), "preferred range");
    assert_eq!(what("(move_speed: Some(13000))"), "move speed");
    assert_eq!(what("(move_speed: Some(0))"), "move speed");
    assert_eq!(what("(reach: Some(11000))"), "swing reach");
    assert_eq!(what("(strike_radius: Some(4000))"), "strike radius");
    assert_eq!(what("(windup_ticks: Some(0))"), "wind-up");
    assert_eq!(what("(active_ticks: Some(241))"), "active window");
    assert_eq!(
        lurker("(recovery_ticks: Some(0))").validate(),
        Ok(()),
        "no recovery is a choice, not an error"
    );
}

#[test]
fn a_wave_needs_an_encounter_to_send_it() {
    let location: LocationDef = ron::from_str(
        r#"(
            schema_version: 1,
            id: "drift-yard",
            name: "Drift Yard",
            region: "cinderwake",
            kind: Wilderness,
            spawns: [
                (
                    id: "straggler",
                    placement: (at: (x: 0, y: 0, z: 0), yaw_milli_turns: 0),
                    role: Enemy(prefab: "corrupted-husk", wave: 1),
                ),
            ],
            props: [],
            navmesh: [],
            links: [],
        )"#,
    )
    .expect("parses");
    assert!(matches!(
        location.validate(),
        Err(ContentError::WaveWithoutEncounter { .. })
    ));
}

#[test]
fn a_spire_eruption_is_bounded() {
    let spires = |count: u8, height: i64, width: i64, spread: i64| {
        vfx(vec![VfxPart::Spire {
            color: swatch(),
            glow: Some(swatch()),
            count,
            height: Fixed::from_milli(height),
            width: Fixed::from_milli(width),
            spread: Fixed::from_milli(spread),
            ticks: 100,
        }])
    };
    assert_eq!(spires(9, 2_800, 1_000, 4_500).validate(), Ok(()));
    assert_eq!(spires(9, 2_800, 1_000, 4_500).particles(), 9, "a piece a spire");
    assert_eq!(
        spires(0, 2_800, 1_000, 4_500).validate(),
        Err(ContentError::ZeroCount { what: "spires" })
    );
    assert!(matches!(
        spires(MAX_SPIRES + 1, 2_800, 1_000, 4_500).validate(),
        Err(ContentError::OverBudget { .. })
    ));
    let what = |part: VfxDef| match part.validate() {
        Err(ContentError::OutOfRange { what, .. }) => what,
        other => panic!("expected out of range, got {other:?}"),
    };
    assert_eq!(
        what(spires(9, MAX_SPIRE_HEIGHT_MILLI + 1, 1_000, 4_500)),
        "spire height"
    );
    assert_eq!(
        what(spires(9, 2_800, MAX_SPIRE_WIDTH_MILLI + 1, 4_500)),
        "spire width"
    );
    assert_eq!(
        what(spires(9, 2_800, 1_000, MAX_SPIRE_SPREAD_MILLI + 1)),
        "spire spread"
    );

    let glowing = spires(3, 2_000, 800, 3_000);
    let painted: Vec<_> = glowing.parts[0].swatches().collect();
    assert_eq!(
        painted.len(),
        2,
        "its rock and its glow are both swatches to resolve"
    );
}
