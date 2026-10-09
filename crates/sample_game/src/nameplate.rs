//! Names over heads: who everyone is, at a glance.
//!
//! Every named body near the player — an enemy by its content name, an NPC by
//! its id read as a name — carries a line of blocky text above its health
//! bar, turned to face the camera. The text is a mesh from
//! [`engine::pixel_text`], built once per name at startup, so a nameplate is a
//! model like any other: two draws, the words and a black shadow a pixel
//! behind them so they read against any ground.
//!
//! Colour says what a name belongs to, the way the rest of the slice's cues
//! do: enemies in yellow, an elite in orange, and the people of the
//! settlement in a pale cyan nothing hostile wears.
//!
//! Presentation only. Nothing reads a nameplate, and plates are pooled and
//! reassigned every frame rather than owned by the bodies they label, so a
//! body's death or a door leaves nothing dangling.

use std::collections::BTreeMap;

use content_runtime::ContentRegistry;
use engine::camera::Camera;
use engine::ecs::{Component, Entity, Resource, World};
use engine::math::{Mat3, Quat, Transform, Vec3};
use engine::pixel_text;
use engine::render::{MaterialDesc, MaterialHandle, MeshHandle, Model, ModelPart, RenderAssets, Visibility};
use game_core::content::StatusId;
use game_core::ids::NpcId;
use game_runtime::status::Statuses;

use crate::slice::{self, Enemy, Npc, Player};

/// Edge of one text pixel, in metres: a name is about as tall as a hand.
pub(crate) const PIXEL: f32 = 0.03;

/// Furthest a name is shown from the player, in metres: the fight around you,
/// not the whole location.
pub(crate) const NAME_RANGE: f32 = 14.0;

/// Height of a name's baseline above the top of its health bar.
const ABOVE_BAR: f32 = 0.16;

/// How a name is coloured, by what it belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Kind {
    Enemy,
    Elite,
    Person,
}

/// Every name content can put over a head, built once, and the materials they
/// are drawn in.
#[derive(Debug)]
pub(crate) struct NameLibrary {
    meshes: BTreeMap<String, MeshHandle>,
    enemy: MaterialHandle,
    elite: MaterialHandle,
    person: MaterialHandle,
    shadow: MaterialHandle,
    /// Each status's name as a mesh, in its chip colour (M9).
    statuses: BTreeMap<StatusId, (MeshHandle, MaterialHandle)>,
}

impl Resource for NameLibrary {}

/// An NPC's id read as a name: `warden-sela` is "Warden Sela".
pub(crate) fn person_name(npc: &NpcId) -> String {
    npc.name()
        .split('-')
        .map(|word| {
            let mut letters = word.chars();
            letters.next().map_or_else(String::new, |first| {
                first.to_uppercase().chain(letters).collect::<String>()
            })
        })
        .collect::<Vec<_>>()
        .join(" ")
}

impl NameLibrary {
    /// Builds a mesh for every enemy name and every placed NPC in `content`.
    pub(crate) fn build(content: &ContentRegistry, assets: &mut dyn RenderAssets) -> Self {
        let mut names: Vec<String> = content.enemies.values().map(|def| def.name.clone()).collect();
        for location in content.locations.values() {
            for spawn in &location.spawns {
                if let game_core::content::SpawnRole::Npc { npc, .. } = &spawn.role {
                    names.push(person_name(npc));
                }
            }
        }
        let mut meshes = BTreeMap::new();
        for name in names {
            if meshes.contains_key(&name) {
                continue;
            }
            let mesh = pixel_text::text_mesh(&name, PIXEL, [1.0, 1.0, 1.0]);
            if mesh.bounds().is_none() {
                continue;
            }
            match assets.upload_mesh(&mesh) {
                Ok(handle) => {
                    meshes.insert(name, handle);
                }
                Err(error) => {
                    engine::log_warn!("names", "a name failed to upload", name = name, error = error);
                }
            }
        }
        // Each status's name, for the lines under an afflicted enemy's name.
        let status_meshes: Vec<(StatusId, MeshHandle, [f32; 3])> = content
            .statuses
            .values()
            .filter_map(|def| {
                let mesh = pixel_text::text_mesh(&def.name, PIXEL, [1.0, 1.0, 1.0]);
                mesh.bounds()?;
                let handle = assets.upload_mesh(&mesh).ok()?;
                Some((def.id.clone(), handle, crate::hud::status_rgb(def, content)))
            })
            .collect();
        // Lit colours, pushed almost all the way to flat so a name reads the
        // same on the sunny and the shaded side of a scene.
        let mut flat = |[r, g, b]: [f32; 3]| {
            assets.create_material(&MaterialDesc {
                base_color: [r, g, b, 1.0],
                flash: [r, g, b, 0.9],
                ..MaterialDesc::default()
            })
        };
        let statuses = status_meshes
            .into_iter()
            .map(|(id, mesh, rgb)| (id, (mesh, flat(rgb))))
            .collect();
        Self {
            statuses,
            meshes,
            enemy: flat([1.0, 0.92, 0.25]),
            elite: flat([1.0, 0.55, 0.12]),
            person: flat([0.55, 0.92, 1.0]),
            shadow: flat([0.0, 0.0, 0.0]),
        }
    }

    pub(crate) fn get(&self, name: &str) -> Option<MeshHandle> {
        self.meshes.get(name).copied()
    }

    fn material(&self, kind: Kind) -> MaterialHandle {
        match kind {
            Kind::Enemy => self.enemy,
            Kind::Elite => self.elite,
            Kind::Person => self.person,
        }
    }
}

/// A pooled nameplate. Which body it labels changes from frame to frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Nameplate;

impl Component for Nameplate {}

/// Every body that should wear a name this frame: its name, its kind, where
/// the name sits, and the statuses on it.
pub(crate) fn named(world: &World, content: &ContentRegistry) -> Vec<(String, Kind, Vec3, Vec<StatusId>)> {
    let Some(player) = world
        .query2::<Transform, Player>()
        .next()
        .map(|(_, transform, _)| transform.translation)
    else {
        return Vec::new();
    };
    let near = |at: Vec3| Vec3::new(at.x - player.x, 0.0, at.z - player.z).length() <= NAME_RANGE;
    let mut found = Vec::new();
    for (entity, transform, enemy) in world.query2::<Transform, Enemy>() {
        if !near(transform.translation) {
            continue;
        }
        let Some(def) = content.enemies.get(&enemy.prefab) else {
            continue;
        };
        let kind = if def.variant_of.is_some() {
            Kind::Elite
        } else {
            Kind::Enemy
        };
        let tick = world
            .resource::<engine::time::FixedTime>()
            .map_or(0, |time| time.tick);
        let afflicted = world.get::<Statuses>(entity).map_or_else(Vec::new, |statuses| {
            statuses
                .iter()
                .filter(|held| held.expires_at > tick)
                .map(|held| held.status.clone())
                .collect()
        });
        found.push((def.name.clone(), kind, above(world, entity, transform), afflicted));
    }
    for (entity, transform, npc) in world.query2::<Transform, Npc>() {
        if near(transform.translation) {
            found.push((
                person_name(&npc.0),
                Kind::Person,
                above(world, entity, transform),
                Vec::new(),
            ));
        }
    }
    found
}

/// Where a body's name sits: over its health bar.
fn above(world: &World, entity: Entity, transform: &Transform) -> Vec3 {
    slice::crown(world, entity, transform) + Vec3::Y * ABOVE_BAR
}

/// Turned so its +Z faces back along the camera's view: text that reads flat
/// on to the screen wherever it stands.
pub(crate) fn facing_camera(camera: &Camera) -> Quat {
    let forward = camera.forward();
    let right = forward.cross(Vec3::Y).try_normalize().unwrap_or(Vec3::X);
    let up = right.cross(forward);
    Quat::from_mat3(&Mat3::from_cols(right, up, -forward))
}

/// Puts a name over every named body near the player, and hides the rest.
pub(crate) fn show_names(world: &mut World, content: &ContentRegistry) {
    let named = named(world, content);
    let Some((shadow, looks, chips)) = world.resource::<NameLibrary>().map(|library| {
        (
            library.shadow,
            named
                .iter()
                .map(|(name, kind, _, _)| library.get(name).map(|mesh| (mesh, library.material(*kind))))
                .collect::<Vec<_>>(),
            named
                .iter()
                .map(|(_, _, _, statuses)| {
                    statuses
                        .iter()
                        .filter_map(|status| library.statuses.get(status).copied())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        )
    }) else {
        return;
    };
    let rotation = facing_camera(&world.resource::<Camera>().copied().unwrap_or_default());

    let mut plates: Vec<Entity> = world.query::<Nameplate>().map(|(entity, _)| entity).collect();
    plates.sort();
    let mut used = 0;
    for (((_, _, at, _), look), chips) in named.iter().zip(looks).zip(chips) {
        let Some((mesh, material)) = look else {
            continue;
        };
        let plate = if let Some(&plate) = plates.get(used) {
            plate
        } else {
            let plate = world.spawn();
            let _ = world.insert(plate, Nameplate);
            plates.push(plate);
            plate
        };
        used += 1;
        // The shadow a pixel down and to the right, and a hair behind.
        let mut behind = ModelPart::new(mesh, shadow, Transform::IDENTITY);
        behind.pose = Transform::from_translation(Vec3::new(PIXEL * 0.6, -PIXEL * 0.6, -0.01));
        let mut parts = vec![behind, ModelPart::new(mesh, material, Transform::IDENTITY)];
        // Statuses, a line each, under the name (M9).
        for (row, (chip, colour)) in chips.into_iter().enumerate() {
            let mut line = ModelPart::new(chip, colour, Transform::IDENTITY);
            line.pose = Transform::from_translation(Vec3::new(0.0, -PIXEL * 9.0 * (row as f32 + 1.0), 0.0));
            parts.push(line);
        }
        let _ = world.insert(plate, Model { parts });
        let _ = world.insert(
            plate,
            Transform {
                translation: *at,
                rotation,
                scale: Vec3::ONE,
            },
        );
        let _ = world.insert(plate, Visibility::VISIBLE);
    }
    for &spare in plates.iter().skip(used) {
        let _ = world.insert(spare, Visibility::HIDDEN);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_npc_id_reads_as_a_name() {
        assert_eq!(
            person_name(&NpcId::new("warden-sela").expect("valid id")),
            "Warden Sela"
        );
        assert_eq!(
            person_name(&NpcId::new("arbiter-thal").expect("valid id")),
            "Arbiter Thal"
        );
    }

    #[test]
    fn a_plate_faces_back_along_the_view() {
        let camera = Camera::default();
        let rotation = facing_camera(&camera);
        let front = rotation * Vec3::Z;
        assert!(
            (front + camera.forward()).length() < 1e-4,
            "its face points at the eye"
        );
        let right = rotation * Vec3::X;
        assert!(right.y.abs() < 1e-4, "and it stays level: {right:?}");
    }
}
