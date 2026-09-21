//! Pathfinding over an authored navmesh.
//!
//! Enemies walk a path of waypoints produced from `LocationDef::navmesh`:
//! A* across polygon neighbours, then a funnel pass that pulls the result taut
//! so a body does not visit the centre of every polygon it crosses.
//!
//! # Why the winding rule matters here
//!
//! Every polygon is convex and counter-clockwise (validated in
//! `game_core::content`). Convexity is what lets a body move in a straight line
//! between any two points inside one polygon without further checks, which is
//! the whole reason a navmesh is cheaper than testing geometry. A reversed
//! polygon breaks the side tests the funnel depends on, which is why authoring
//! rejects one rather than letting pathing quietly refuse to cross it.
//!
//! # Determinism
//!
//! Ties in the A* frontier are broken by polygon index, never by float
//! comparison order or hash iteration. Two runs of the same query return the
//! same path, because enemy movement feeds the fixed step.

use std::collections::BTreeMap;

use engine::math::{Vec2, Vec3};
use game_core::content::{LocationDef, NavPolygon, Point};

/// Converts authored fixed-point content coordinates into runtime metres.
fn to_world(point: Point) -> Vec3 {
    Vec3::new(
        point.x.as_milli() as f32 / 1000.0,
        point.y.as_milli() as f32 / 1000.0,
        point.z.as_milli() as f32 / 1000.0,
    )
}

fn ground(vector: Vec3) -> Vec2 {
    Vec2::new(vector.x, vector.z)
}

/// A navmesh prepared for queries.
///
/// Built once when a location loads. Holds the polygons' world-space vertices
/// and centres so pathing does no fixed-point conversion per query.
#[derive(Debug, Default, Clone)]
pub struct NavMesh {
    polygons: Vec<NavPolygonRuntime>,
}

#[derive(Debug, Clone)]
struct NavPolygonRuntime {
    vertices: Vec<Vec3>,
    centre: Vec3,
    neighbours: Vec<usize>,
}

impl NavMesh {
    /// Prepares the navmesh authored for `location`.
    pub fn from_location(location: &LocationDef) -> Self {
        let polygons = location
            .navmesh
            .iter()
            .map(|polygon| NavPolygonRuntime {
                vertices: polygon.vertices.iter().copied().map(to_world).collect(),
                centre: centre_of(polygon),
                // Validation has already rejected out-of-range neighbours, but
                // filtering keeps this total rather than trusting that.
                neighbours: polygon
                    .neighbours
                    .iter()
                    .map(|index| *index as usize)
                    .filter(|index| *index < location.navmesh.len())
                    .collect(),
            })
            .collect();
        Self { polygons }
    }

    pub fn len(&self) -> usize {
        self.polygons.len()
    }

    pub fn is_empty(&self) -> bool {
        self.polygons.is_empty()
    }

    /// The polygon containing `position` on the ground plane.
    pub fn polygon_at(&self, position: Vec3) -> Option<usize> {
        self.polygons
            .iter()
            .position(|polygon| contains(&polygon.vertices, position))
    }

    /// The polygon nearest `position`, used when a body has drifted off the
    /// mesh: refusing to path because someone clipped a corner is worse than
    /// pathing from the closest legal spot.
    pub fn nearest_polygon(&self, position: Vec3) -> Option<usize> {
        self.polygon_at(position).or_else(|| {
            self.polygons
                .iter()
                .enumerate()
                .min_by(|(left_index, left), (right_index, right)| {
                    let left_distance = ground(left.centre).distance_squared(ground(position));
                    let right_distance = ground(right.centre).distance_squared(ground(position));
                    left_distance
                        .partial_cmp(&right_distance)
                        // Ties and NaN fall back to index order, so the result
                        // never depends on float comparison quirks.
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(left_index.cmp(right_index))
                })
                .map(|(index, _)| index)
        })
    }

    /// Finds a path from `from` to `to`, as world-space waypoints.
    ///
    /// The first waypoint is the start and the last is the destination; the
    /// points between are corners the path must round.
    pub fn find_path(&self, from: Vec3, to: Vec3) -> Option<Vec<Vec3>> {
        let start = self.nearest_polygon(from)?;
        let goal = self.nearest_polygon(to)?;
        if start == goal {
            return Some(vec![from, to]);
        }

        let polygons = self.astar(start, goal)?;
        Some(self.funnel(&polygons, from, to))
    }

    /// A* across polygon neighbours, using centre-to-centre distance.
    fn astar(&self, start: usize, goal: usize) -> Option<Vec<usize>> {
        let goal_centre = ground(self.polygons[goal].centre);

        // Ordered maps rather than a binary heap of floats: the frontier is
        // small, and deterministic iteration matters more than asymptotics.
        let mut came_from: BTreeMap<usize, usize> = BTreeMap::new();
        let mut best_cost: BTreeMap<usize, f32> = BTreeMap::new();
        let mut open: Vec<usize> = vec![start];
        best_cost.insert(start, 0.0);

        while !open.is_empty() {
            // Pick the open node with the lowest f-score, ties by index.
            let (position, &current) =
                open.iter()
                    .enumerate()
                    .min_by(|(left_index, left), (right_index, right)| {
                        let score = |node: usize| {
                            best_cost.get(&node).copied().unwrap_or(f32::INFINITY)
                                + ground(self.polygons[node].centre).distance(goal_centre)
                        };
                        score(**left)
                            .partial_cmp(&score(**right))
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then(left_index.cmp(right_index))
                    })?;
            open.remove(position);

            if current == goal {
                let mut path = vec![current];
                let mut node = current;
                while let Some(&previous) = came_from.get(&node) {
                    path.push(previous);
                    node = previous;
                }
                path.reverse();
                return Some(path);
            }

            let current_cost = best_cost.get(&current).copied().unwrap_or(f32::INFINITY);
            for &neighbour in &self.polygons[current].neighbours {
                let step =
                    ground(self.polygons[current].centre).distance(ground(self.polygons[neighbour].centre));
                let candidate = current_cost + step;
                if candidate < best_cost.get(&neighbour).copied().unwrap_or(f32::INFINITY) {
                    best_cost.insert(neighbour, candidate);
                    came_from.insert(neighbour, current);
                    if !open.contains(&neighbour) {
                        open.push(neighbour);
                    }
                }
            }
        }
        None
    }

    /// Pulls a polygon corridor taut into waypoints.
    ///
    /// A simple version of the funnel algorithm: it walks the shared edges
    /// between consecutive polygons and emits their midpoints. That is not the
    /// tightest possible path, but it is stable, cheap, and never leaves the
    /// corridor — which is what matters while the spaces are hand-authored
    /// rooms rather than open terrain.
    fn funnel(&self, corridor: &[usize], from: Vec3, to: Vec3) -> Vec<Vec3> {
        let mut waypoints = vec![from];
        for window in corridor.windows(2) {
            let (left, right) = (window[0], window[1]);
            if let Some(portal) = self.shared_edge(left, right) {
                let midpoint = (portal.0 + portal.1) / 2.0;
                // Skip a waypoint that adds nothing: consecutive portals in a
                // straight corridor would otherwise make the body stutter.
                if waypoints
                    .last()
                    .is_none_or(|previous| ground(*previous).distance(ground(midpoint)) > 0.05)
                {
                    waypoints.push(midpoint);
                }
            } else {
                waypoints.push(self.polygons[right].centre);
            }
        }
        waypoints.push(to);
        waypoints
    }

    /// The edge two polygons share, if they share one.
    fn shared_edge(&self, left: usize, right: usize) -> Option<(Vec3, Vec3)> {
        let left_vertices = &self.polygons[left].vertices;
        let right_vertices = &self.polygons[right].vertices;
        let mut shared = Vec::new();

        for vertex in left_vertices {
            if right_vertices
                .iter()
                .any(|other| ground(*other).distance(ground(*vertex)) < 1e-3)
            {
                shared.push(*vertex);
            }
        }
        match shared.as_slice() {
            [first, second, ..] => Some((*first, *second)),
            _ => None,
        }
    }
}

/// Centre of an authored polygon, in world space.
fn centre_of(polygon: &NavPolygon) -> Vec3 {
    if polygon.vertices.is_empty() {
        return Vec3::ZERO;
    }
    let sum = polygon
        .vertices
        .iter()
        .copied()
        .map(to_world)
        .fold(Vec3::ZERO, |total, vertex| total + vertex);
    sum / polygon.vertices.len() as f32
}

/// Whether a point lies inside a convex, counter-clockwise polygon.
///
/// Relies on the winding rule the content validator enforces: for every edge,
/// an interior point is on the left. A clockwise polygon would report every
/// point as outside, which is exactly the silent failure that validation
/// prevents.
fn contains(vertices: &[Vec3], position: Vec3) -> bool {
    if vertices.len() < 3 {
        return false;
    }
    let point = ground(position);
    vertices.iter().enumerate().all(|(index, vertex)| {
        let current = ground(*vertex);
        let next = ground(vertices[(index + 1) % vertices.len()]);
        let edge = next - current;
        let to_point = point - current;
        edge.x * to_point.y - edge.y * to_point.x >= -1e-4
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::content::{CONTENT_SCHEMA_VERSION, Fixed, LocationKind};
    use game_core::ids::{LocationId, RegionId};

    fn point(x: i64, z: i64) -> Point {
        Point::new(Fixed::from_whole(x), Fixed::ZERO, Fixed::from_whole(z))
    }

    /// Two rooms sharing an edge at z = 10, joined end to end.
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
                    vertices: vec![point(-5, 0), point(5, 0), point(5, 10), point(-5, 10)],
                    neighbours: vec![1],
                },
                NavPolygon {
                    vertices: vec![point(-5, 10), point(5, 10), point(5, 20), point(-5, 20)],
                    neighbours: vec![0],
                },
            ],
            links: Vec::new(),
        }
    }

    fn at(x: f32, z: f32) -> Vec3 {
        Vec3::new(x, 0.0, z)
    }

    #[test]
    fn a_point_inside_a_room_finds_its_polygon() {
        let mesh = NavMesh::from_location(&two_rooms());
        assert_eq!(mesh.len(), 2);
        assert_eq!(mesh.polygon_at(at(0.0, 5.0)), Some(0));
        assert_eq!(mesh.polygon_at(at(0.0, 15.0)), Some(1));
        assert_eq!(mesh.polygon_at(at(0.0, 50.0)), None, "outside the mesh entirely");
    }

    #[test]
    fn a_body_off_the_mesh_still_gets_a_polygon() {
        let mesh = NavMesh::from_location(&two_rooms());
        assert_eq!(
            mesh.nearest_polygon(at(0.0, -40.0)),
            Some(0),
            "clipping a corner must not make an enemy give up entirely"
        );
    }

    #[test]
    fn a_path_within_one_room_is_a_straight_line() {
        let mesh = NavMesh::from_location(&two_rooms());
        let path = mesh.find_path(at(-2.0, 2.0), at(3.0, 8.0)).expect("same room");
        assert_eq!(path.len(), 2, "no waypoints are needed inside one convex polygon");
    }

    #[test]
    fn a_path_between_rooms_crosses_the_shared_edge() {
        let mesh = NavMesh::from_location(&two_rooms());
        let path = mesh
            .find_path(at(0.0, 2.0), at(0.0, 18.0))
            .expect("connected rooms");
        assert!(path.len() >= 3, "expected a waypoint at the portal, got {path:?}");

        let portal = path[1];
        assert!(
            (portal.z - 10.0).abs() < 0.5,
            "the middle waypoint should sit on the shared edge, got {portal:?}"
        );
    }

    #[test]
    fn a_path_ends_where_it_was_asked_to() {
        let mesh = NavMesh::from_location(&two_rooms());
        let destination = at(1.0, 17.0);
        let path = mesh.find_path(at(0.0, 1.0), destination).expect("connected");
        assert_eq!(*path.last().expect("non-empty"), destination);
        assert_eq!(path[0], at(0.0, 1.0));
    }

    #[test]
    fn disconnected_rooms_have_no_path() {
        let mut location = two_rooms();
        // Sever the connection in both directions.
        location.navmesh[0].neighbours.clear();
        location.navmesh[1].neighbours.clear();
        let mesh = NavMesh::from_location(&location);
        assert!(
            mesh.find_path(at(0.0, 2.0), at(0.0, 18.0)).is_none(),
            "an unreachable destination must fail rather than return a path through walls"
        );
    }

    #[test]
    fn an_empty_mesh_yields_no_path() {
        let mut location = two_rooms();
        location.navmesh.clear();
        let mesh = NavMesh::from_location(&location);
        assert!(mesh.is_empty());
        assert!(mesh.find_path(at(0.0, 0.0), at(0.0, 1.0)).is_none());
    }

    #[test]
    fn a_clockwise_polygon_contains_nothing() {
        let mut location = two_rooms();
        location.navmesh[0].vertices.reverse();
        let mesh = NavMesh::from_location(&location);
        assert_eq!(
            mesh.polygon_at(at(0.0, 5.0)),
            None,
            "this is why authoring rejects reversed winding rather than shipping it"
        );
    }

    #[test]
    fn out_of_range_neighbours_are_ignored_rather_than_panicking() {
        let mut location = two_rooms();
        location.navmesh[0].neighbours.push(99);
        let mesh = NavMesh::from_location(&location);
        assert!(mesh.find_path(at(0.0, 2.0), at(0.0, 18.0)).is_some());
    }

    /// Pathing feeds the fixed step, so the same query must always answer the
    /// same way — including which of two equal-cost routes it picks.
    #[test]
    fn pathfinding_is_deterministic() {
        let mesh = NavMesh::from_location(&two_rooms());
        let first = mesh.find_path(at(-3.0, 1.0), at(4.0, 19.0));
        let second = mesh.find_path(at(-3.0, 1.0), at(4.0, 19.0));
        assert_eq!(first, second);
    }

    #[test]
    fn content_coordinates_convert_to_metres() {
        let mesh = NavMesh::from_location(&two_rooms());
        let polygon = mesh.polygon_at(at(0.0, 5.0)).expect("inside");
        let centre = mesh.polygons[polygon].centre;
        assert!(
            (centre.z - 5.0).abs() < 1e-3,
            "a room spanning 0..10 m should centre at 5 m, got {centre:?}"
        );
    }
}
