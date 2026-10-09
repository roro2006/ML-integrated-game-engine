//! Golden-image fixtures for the GPU tests.
//!
//! A stand-in until the G3 harness exists (`docs/plans/11-m17-phase1-task-list.md`,
//! slice 0). References live in `tests/fixtures/golden/` as raw RGBA8 behind a
//! 16-byte header, so no image codec is needed:
//!
//! | Bytes  | Field                       |
//! |--------|-----------------------------|
//! | 0..8   | magic `GOLDRGBA`            |
//! | 8..12  | width, little-endian `u32`  |
//! | 12..16 | height, little-endian `u32` |
//! | 16..   | pixels, as in [`Image`]     |
//!
//! References are captured on one machine's driver, so tolerance 0 holds on
//! that machine and not across vendors. Run a test with `UPDATE_GOLDEN=1` to
//! write its reference instead of comparing against it.

use std::fmt;
use std::path::PathBuf;

use crate::image::Image;

const MAGIC: &[u8; 8] = b"GOLDRGBA";
const HEADER_LEN: usize = 16;

pub(crate) fn encode(image: &Image) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(HEADER_LEN + image.rgba8.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&image.width.to_le_bytes());
    bytes.extend_from_slice(&image.height.to_le_bytes());
    bytes.extend_from_slice(&image.rgba8);
    bytes
}

pub(crate) fn decode(bytes: &[u8]) -> Result<Image, String> {
    let header = bytes.get(..HEADER_LEN).ok_or("shorter than the header")?;
    if &header[..8] != MAGIC {
        return Err("not a golden image (bad magic)".to_owned());
    }
    let width = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
    let height = u32::from_le_bytes([header[12], header[13], header[14], header[15]]);
    let rgba8 = bytes[HEADER_LEN..].to_vec();
    let expected = width as usize * height as usize * 4;
    if rgba8.len() != expected {
        return Err(format!(
            "{width}x{height} needs {expected} pixel bytes, found {}",
            rgba8.len()
        ));
    }
    Ok(Image { width, height, rgba8 })
}

/// How an image differs from its reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Mismatch {
    Size {
        expected: (u32, u32),
        actual: (u32, u32),
    },
    Pixels {
        /// Pixels with any channel further than the tolerance from the reference.
        differing: usize,
        max_delta: u8,
        /// The first pixel with the largest difference.
        worst: (u32, u32),
    },
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Size { expected, actual } => write!(
                f,
                "size differs: reference {}x{}, rendered {}x{}",
                expected.0, expected.1, actual.0, actual.1
            ),
            Self::Pixels {
                differing,
                max_delta,
                worst,
            } => write!(
                f,
                "{differing} pixels differ, by up to {max_delta} (first worst at {}, {})",
                worst.0, worst.1
            ),
        }
    }
}

/// Compares two images channel by channel, allowing `tolerance` per channel.
pub(crate) fn compare(expected: &Image, actual: &Image, tolerance: u8) -> Result<(), Mismatch> {
    if (expected.width, expected.height) != (actual.width, actual.height)
        || expected.rgba8.len() != actual.rgba8.len()
    {
        return Err(Mismatch::Size {
            expected: (expected.width, expected.height),
            actual: (actual.width, actual.height),
        });
    }
    let mut differing = 0;
    let mut max_delta = 0;
    let mut worst = (0, 0);
    let pixels = expected.rgba8.chunks_exact(4).zip(actual.rgba8.chunks_exact(4));
    for (index, (reference, rendered)) in pixels.enumerate() {
        let delta = reference
            .iter()
            .zip(rendered)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        if delta > tolerance {
            differing += 1;
        }
        if delta > max_delta {
            max_delta = delta;
            let index = index as u32;
            worst = (index % expected.width, index / expected.width);
        }
    }
    if differing == 0 {
        Ok(())
    } else {
        Err(Mismatch::Pixels {
            differing,
            max_delta,
            worst,
        })
    }
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/golden")
        .join(format!("{name}.rgba"))
}

/// Checks `image` against the reference called `name`, or writes the reference
/// when `UPDATE_GOLDEN` is set.
pub(crate) fn check(name: &str, image: &Image, tolerance: u8) -> Result<(), String> {
    let path = fixture_path(name);
    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
        }
        return std::fs::write(&path, encode(image)).map_err(|error| format!("{}: {error}", path.display()));
    }
    let bytes = std::fs::read(&path).map_err(|error| {
        format!(
            "{}: {error}; run with UPDATE_GOLDEN=1 to capture it",
            path.display()
        )
    })?;
    let reference = decode(&bytes).map_err(|error| format!("{}: {error}", path.display()))?;
    compare(&reference, image, tolerance).map_err(|mismatch| format!("{name}: {mismatch}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: u32, height: u32, fill: u8) -> Image {
        Image {
            width,
            height,
            rgba8: vec![fill; (width * height * 4) as usize],
        }
    }

    #[test]
    fn an_image_survives_encoding() {
        let mut original = image(3, 2, 7);
        original.rgba8[5] = 200;
        assert_eq!(decode(&encode(&original)), Ok(original));
    }

    #[test]
    fn decoding_rejects_a_wrong_magic_and_a_short_body() {
        let mut bytes = encode(&image(2, 2, 0));
        assert!(decode(&bytes[..HEADER_LEN - 1]).is_err(), "truncated header");
        bytes.pop();
        assert!(decode(&bytes).is_err(), "one pixel byte missing");
        bytes[0] = b'X';
        assert!(decode(&bytes).is_err(), "bad magic");
    }

    #[test]
    fn identical_images_match_at_tolerance_zero() {
        assert_eq!(compare(&image(4, 4, 9), &image(4, 4, 9), 0), Ok(()));
    }

    #[test]
    fn a_difference_within_tolerance_passes_and_beyond_it_is_located() {
        let reference = image(4, 4, 100);
        let mut rendered = reference.clone();
        // Pixel (1, 0) off by 2; pixel (2, 3) off by 5 in its blue channel.
        rendered.rgba8[4] = 102;
        rendered.rgba8[(3 * 4 + 2) * 4 + 2] = 95;

        assert_eq!(compare(&reference, &rendered, 5), Ok(()));
        assert_eq!(
            compare(&reference, &rendered, 1),
            Err(Mismatch::Pixels {
                differing: 2,
                max_delta: 5,
                worst: (2, 3),
            })
        );
        assert_eq!(
            compare(&reference, &rendered, 2),
            Err(Mismatch::Pixels {
                differing: 1,
                max_delta: 5,
                worst: (2, 3),
            }),
            "a pixel exactly at the tolerance passes"
        );
    }

    #[test]
    fn a_short_image_never_matches_its_reference() {
        let reference = image(4, 4, 0);
        let mut short = reference.clone();
        short.rgba8.truncate(8);
        assert!(
            compare(&reference, &short, 255).is_err(),
            "a matching prefix is not a match"
        );
    }

    #[test]
    fn a_size_change_is_reported_before_any_pixel() {
        assert_eq!(
            compare(&image(4, 4, 0), &image(4, 2, 0), 255),
            Err(Mismatch::Size {
                expected: (4, 4),
                actual: (4, 2),
            })
        );
    }
}
