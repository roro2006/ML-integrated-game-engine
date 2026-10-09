//! Frames read back from the GPU.
//!
//! A buffer copied out of a texture has every row padded to
//! `COPY_BYTES_PER_ROW_ALIGNMENT` bytes. The padding is a GPU detail, so it is
//! stripped here and never reaches an [`Image`].

/// A rendered frame read back to the CPU.
///
/// Tightly packed RGBA8, rows top to bottom, display-encoded: the same sRGB
/// bytes a window would have shown, in RGBA order whatever the swapchain uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

impl Image {
    /// The pixel at `(x, y)`, or `None` outside the image.
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let start = (y as usize * self.width as usize + x as usize) * BYTES_PER_PIXEL;
        let texel = self.rgba8.get(start..start + BYTES_PER_PIXEL)?;
        Some([texel[0], texel[1], texel[2], texel[3]])
    }
}

pub(crate) const BYTES_PER_PIXEL: usize = 4;

/// Bytes per row in a readback buffer for an RGBA8 image `width` pixels wide.
pub(crate) fn padded_bytes_per_row(width: u32) -> u32 {
    let unpadded = width * BYTES_PER_PIXEL as u32;
    unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
}

/// Copies the pixel bytes out of a padded readback buffer, dropping each row's
/// padding. Rows the buffer is too short to hold are left out rather than
/// read past its end.
pub(crate) fn unpad_rows(padded: &[u8], width: u32, height: u32) -> Vec<u8> {
    let row = width as usize * BYTES_PER_PIXEL;
    let stride = padded_bytes_per_row(width) as usize;
    let mut out = Vec::with_capacity(row * height as usize);
    for y in 0..height as usize {
        let Some(bytes) = padded.get(y * stride..y * stride + row) else {
            break;
        };
        out.extend_from_slice(bytes);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_padded_to_the_copy_alignment() {
        assert_eq!(padded_bytes_per_row(64), 256, "exactly one alignment unit");
        assert_eq!(padded_bytes_per_row(65), 512, "one pixel over rounds up");
        assert_eq!(padded_bytes_per_row(1), 256);
        assert_eq!(padded_bytes_per_row(128), 512);
    }

    #[test]
    fn unpadding_keeps_each_row_and_drops_the_tail() {
        // Two rows of a 3-pixel image: 12 bytes of pixels, 244 of padding.
        let mut padded = vec![0xEE; 512];
        for (i, byte) in padded[..12].iter_mut().enumerate() {
            *byte = i as u8;
        }
        for (i, byte) in padded[256..268].iter_mut().enumerate() {
            *byte = 100 + i as u8;
        }

        let rows = unpad_rows(&padded, 3, 2);

        assert_eq!(rows.len(), 24);
        assert_eq!(&rows[..12], &(0..12).collect::<Vec<u8>>()[..]);
        assert_eq!(&rows[12..], &(100..112).collect::<Vec<u8>>()[..]);
        assert!(!rows.contains(&0xEE), "no padding byte survives");
    }

    #[test]
    fn a_short_buffer_is_not_read_past_its_end() {
        // The second row would run from byte 256 to 268.
        assert_eq!(unpad_rows(&[1; 260], 3, 2).len(), 12, "only the first row fits");
        assert_eq!(unpad_rows(&[1; 268], 3, 2).len(), 24, "both rows fit exactly");
    }

    #[test]
    fn pixels_are_addressed_row_major_from_the_top() {
        let image = Image {
            width: 2,
            height: 2,
            rgba8: (0..16).collect(),
        };
        assert_eq!(image.pixel(0, 0), Some([0, 1, 2, 3]));
        assert_eq!(image.pixel(1, 0), Some([4, 5, 6, 7]));
        assert_eq!(image.pixel(0, 1), Some([8, 9, 10, 11]));
        assert_eq!(image.pixel(2, 0), None);
        assert_eq!(image.pixel(0, 2), None);
    }
}
