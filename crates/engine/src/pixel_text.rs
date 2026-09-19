//! Blocky text built as meshes: the engine's first way to put words on screen.
//!
//! There is no font renderer, texture atlas or glyph cache here, and no font
//! dependency. A small pixel font — five by seven, capitals, digits and a
//! little punctuation — is data in this file, and a string becomes one mesh
//! of flat squares, one quad per horizontal run of lit pixels. That suits a low-poly
//! game whose text should look like it belongs in the same world, and it
//! reuses the mesh path whole: a line of text is one draw.
//!
//! Lower-case letters are drawn as capitals. A character the font does not
//! know is drawn as a gap, never an error, so any string yields a mesh.
//!
//! # Conventions
//!
//! Text lies in the XY plane and faces +Z, centred on the origin horizontally
//! with its baseline at `y = 0`. It is flat and one-sided — made to be turned
//! toward the camera, as a nameplate is — which keeps a whole name to a few
//! hundred triangles: two per run, where a box per pixel would be a dozen.

use renderer::MeshData;

/// Pixel columns in one glyph.
pub const GLYPH_WIDTH: usize = 5;
/// Pixel rows in one glyph.
pub const GLYPH_HEIGHT: usize = 7;
/// Blank columns between glyphs.
const SPACING: usize = 1;
/// Columns from one glyph's left edge to the next's.
pub const ADVANCE: usize = GLYPH_WIDTH + SPACING;

/// A glyph's rows, top first; `#` is a lit pixel.
fn rows(character: char) -> Option<[&'static str; GLYPH_HEIGHT]> {
    Some(match character.to_ascii_uppercase() {
        'A' => [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
        'B' => ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."],
        'C' => [".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."],
        'D' => ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."],
        'E' => ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
        'F' => ["#####", "#....", "#....", "####.", "#....", "#....", "#...."],
        'G' => [".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".####"],
        'H' => ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
        'I' => [".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."],
        'J' => ["..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."],
        'K' => ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
        'L' => ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
        'M' => ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"],
        'N' => ["#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"],
        'O' => [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
        'P' => ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
        'Q' => [".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"],
        'R' => ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
        'S' => [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
        'T' => ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
        'U' => ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
        'V' => ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."],
        'W' => ["#...#", "#...#", "#...#", "#.#.#", "#.#.#", "#.#.#", ".#.#."],
        'X' => ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"],
        'Y' => ["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."],
        'Z' => ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"],
        '0' => [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
        '1' => ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
        '2' => [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
        '3' => ["####.", "....#", "....#", ".###.", "....#", "....#", "####."],
        '4' => ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
        '5' => ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
        '6' => [".###.", "#....", "#....", "####.", "#...#", "#...#", ".###."],
        '7' => ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
        '8' => [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
        '9' => [".###.", "#...#", "#...#", ".####", "....#", "....#", ".###."],
        '\'' => ["..#..", "..#..", ".#...", ".....", ".....", ".....", "....."],
        '-' => [".....", ".....", ".....", ".###.", ".....", ".....", "....."],
        '.' => [".....", ".....", ".....", ".....", ".....", ".##..", ".##.."],
        ',' => [".....", ".....", ".....", ".....", ".##..", "..#..", ".#..."],
        '!' => ["..#..", "..#..", "..#..", "..#..", "..#..", ".....", "..#.."],
        '?' => [".###.", "#...#", "....#", "...#.", "..#..", ".....", "..#.."],
        _ => return None,
    })
}

/// Whether the font can draw `character` (spaces aside).
pub fn has_glyph(character: char) -> bool {
    rows(character).is_some()
}

/// How wide `text` is, in pixels.
pub fn width_in_pixels(text: &str) -> usize {
    let count = text.chars().count();
    (count * (GLYPH_WIDTH + SPACING)).saturating_sub(SPACING)
}

/// Every character the font draws, in a fixed order: what a glyph library
/// builds a mesh for.
pub fn glyphs() -> impl Iterator<Item = char> {
    ('A'..='Z').chain('0'..='9').chain("'-.,!?".chars())
}

/// One glyph as a mesh of one-unit pixels, its bottom-left corner at the
/// origin, facing +Z. Scaled and placed per character, a set of these draws
/// any string at run time — where [`text_mesh`] needs the string up front.
pub fn glyph_mesh(character: char, color: [f32; 3]) -> Option<MeshData> {
    has_glyph(character).then(|| {
        let mesh = text_mesh(&character.to_string(), 1.0, color);
        // `text_mesh` centres; a glyph is placed by its corner.
        let shift = width_in_pixels(&character.to_string()) as f32 * 0.5;
        let mut mesh = mesh;
        for position in &mut mesh.positions {
            position[0] += shift;
        }
        mesh
    })
}

/// `text` broken into lines of at most `columns` characters, at spaces where
/// it can be and mid-word only where a single word is longer than a line.
pub fn wrap(text: &str, columns: usize) -> Vec<String> {
    let columns = columns.max(1);
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        let mut word = word.to_owned();
        loop {
            let needed = if line.is_empty() {
                word.chars().count()
            } else {
                line.chars().count() + 1 + word.chars().count()
            };
            if needed <= columns {
                if !line.is_empty() {
                    line.push(' ');
                }
                line.push_str(&word);
                break;
            }
            if !line.is_empty() {
                lines.push(std::mem::take(&mut line));
                continue;
            }
            // One word longer than a whole line: break it.
            let head: String = word.chars().take(columns).collect();
            word = word.chars().skip(columns).collect();
            lines.push(head);
            if word.is_empty() {
                break;
            }
        }
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

/// Every horizontal run of lit pixels in `text`, as `(column, row, length)`
/// with row 0 at the top.
fn runs(text: &str) -> Vec<(usize, usize, usize)> {
    let mut found = Vec::new();
    for (index, character) in text.chars().enumerate() {
        let Some(glyph) = rows(character) else {
            continue;
        };
        let left = index * (GLYPH_WIDTH + SPACING);
        for (row, line) in glyph.iter().enumerate() {
            let mut start = None;
            for (column, lit) in line
                .chars()
                .map(|c| c == '#')
                .chain(std::iter::once(false))
                .enumerate()
            {
                match (lit, start) {
                    (true, None) => start = Some(column),
                    (false, Some(from)) => {
                        found.push((left + from, row, column - from));
                        start = None;
                    }
                    _ => {}
                }
            }
        }
    }
    found
}

/// `text` as a mesh of pixel blocks `pixel` metres on a side, painted
/// `color`, centred on the origin horizontally with its baseline at `y = 0`.
///
/// Returns an empty mesh for text with nothing to draw.
pub fn text_mesh(text: &str, pixel: f32, color: [f32; 3]) -> MeshData {
    let pixel = pixel.max(1e-4);
    let half_width = width_in_pixels(text) as f32 * pixel * 0.5;
    let mut mesh = MeshData::default();
    for (column, row, length) in runs(text) {
        let left = column as f32 * pixel - half_width;
        let right = left + length as f32 * pixel;
        let bottom = (GLYPH_HEIGHT - 1 - row) as f32 * pixel;
        let top = bottom + pixel;
        let base = mesh.positions.len() as u32;
        // Counter-clockwise seen from +Z, the side it faces: the pipeline's
        // front face.
        mesh.positions.extend([
            [left, bottom, 0.0],
            [right, bottom, 0.0],
            [right, top, 0.0],
            [left, top, 0.0],
        ]);
        mesh.normals.extend([[0.0, 0.0, 1.0]; 4]);
        mesh.uvs.extend([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]);
        mesh.colors.extend([color; 4]);
        mesh.indices
            .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_glyph_is_five_by_seven() {
        for character in ('A'..='Z').chain('0'..='9').chain("'-.,!?".chars()) {
            let glyph = rows(character).unwrap_or_else(|| panic!("{character} has a glyph"));
            for line in glyph {
                assert_eq!(line.len(), GLYPH_WIDTH, "{character}: {line}");
                assert!(line.chars().all(|c| c == '#' || c == '.'), "{character}: {line}");
            }
            assert!(
                glyph.iter().any(|line| line.contains('#')),
                "{character} draws something"
            );
        }
    }

    #[test]
    fn lower_case_draws_as_capitals_and_unknowns_as_gaps() {
        assert_eq!(runs("sela"), runs("SELA"));
        assert!(has_glyph('q'));
        assert!(!has_glyph('~'));
        assert!(
            runs("~~").is_empty(),
            "an unknown character is a gap, not an error"
        );
    }

    #[test]
    fn text_is_centred_on_its_baseline() {
        let pixel = 0.1;
        let mesh = text_mesh("HUSK", pixel, [1.0, 1.0, 1.0]);
        let bounds = mesh.bounds().expect("something to draw");
        let width = width_in_pixels("HUSK") as f32 * pixel;
        assert!(
            (bounds.min[0] + width * 0.5).abs() < 1e-4,
            "left edge: {bounds:?}"
        );
        assert!(
            (bounds.max[0] - width * 0.5).abs() < 1e-4,
            "right edge: {bounds:?}"
        );
        assert!(bounds.min[1].abs() < 1e-4, "baseline at zero: {bounds:?}");
        assert!(
            (bounds.max[1] - GLYPH_HEIGHT as f32 * pixel).abs() < 1e-4,
            "a glyph's height tall: {bounds:?}"
        );
        assert!(mesh.validate().is_ok());
    }

    #[test]
    fn runs_merge_so_a_line_of_text_stays_cheap() {
        // 'T' is a five-pixel bar over a one-pixel stem: its top row is one
        // quad, not five.
        let top: Vec<_> = runs("T").into_iter().filter(|(_, row, _)| *row == 0).collect();
        assert_eq!(top, vec![(0, 0, 5)]);
        let mesh = text_mesh("CORRUPTED HUSK", 0.03, [1.0, 1.0, 1.0]);
        assert_eq!(
            mesh.triangle_count(),
            runs("CORRUPTED HUSK").len() * 2,
            "two per run"
        );
        assert!(
            mesh.triangle_count() < 400,
            "a whole name for a few hundred triangles: {}",
            mesh.triangle_count()
        );
    }

    /// One-sided, so it must face the way it claims: with back faces culled,
    /// a quad wound the wrong way would simply not be there.
    #[test]
    fn every_quad_faces_plus_z() {
        let mesh = text_mesh("SELA", 0.1, [1.0, 1.0, 1.0]);
        for triangle in mesh.indices.chunks(3) {
            let [a, b, c] = [0, 1, 2].map(|corner| {
                let [x, y, z] = mesh.positions[triangle[corner] as usize];
                crate::math::Vec3::new(x, y, z)
            });
            let facing = (b - a).cross(c - a);
            assert!(facing.z > 0.0, "a triangle faces away from +Z: {facing:?}");
        }
    }

    #[test]
    fn a_glyph_sits_on_its_corner() {
        let a = glyph_mesh('a', [1.0, 1.0, 1.0]).expect("a glyph");
        let bounds = a.bounds().expect("drawn");
        assert!(
            bounds.min[0].abs() < 1e-5 && bounds.min[1].abs() < 1e-5,
            "{bounds:?}"
        );
        assert!((bounds.max[0] - GLYPH_WIDTH as f32).abs() < 1e-5);
        assert!((bounds.max[1] - GLYPH_HEIGHT as f32).abs() < 1e-5);
        assert!(
            glyph_mesh(' ', [1.0, 1.0, 1.0]).is_none(),
            "a space draws nothing"
        );
        assert_eq!(glyphs().count(), 42);
        assert!(glyphs().all(has_glyph));
    }

    #[test]
    fn text_wraps_at_spaces_and_breaks_only_what_cannot_fit() {
        assert_eq!(
            wrap("The intake's fouled again. Third time since the collapse.", 20),
            vec!["The intake's fouled", "again. Third time", "since the collapse."]
        );
        assert_eq!(wrap("", 10), Vec::<String>::new());
        assert_eq!(wrap("a  b", 10), vec!["a b"], "runs of spaces collapse");
        assert_eq!(
            wrap("Unbelievably", 5),
            vec!["Unbel", "ievab", "ly"],
            "a word longer than a line is broken rather than overflowing"
        );
        for line in wrap("Walk the drowned road to the old works and see what fouls it", 12) {
            assert!(line.chars().count() <= 12, "{line}");
        }
    }

    #[test]
    fn nothing_to_draw_is_an_empty_mesh() {
        assert!(text_mesh("", 0.1, [1.0, 1.0, 1.0]).bounds().is_none());
        assert!(text_mesh("   ", 0.1, [1.0, 1.0, 1.0]).bounds().is_none());
    }
}
