//! Uniform byte layout and the arithmetic around per-draw dynamic offsets.
//!
//! Everything here is deliberately free of `wgpu` types. Uniform padding is the
//! classic source of silent rendering corruption — a member at the wrong offset
//! produces a picture that is merely *wrong*, never an error — so the layout
//! maths lives in pure functions that can be asserted on a machine with no GPU.
//!
//! `bytemuck` is not a dependency, so the structs are written out byte by byte
//! rather than transmuted. At two uniform blocks this is a few lines of
//! `to_le_bytes`, and it keeps the crate's dependency list at exactly `wgpu`.
//! Little-endian is hard-coded because every platform this engine targets
//! (x86-64 Windows, x86-64 Linux, SteamOS) is little-endian, and WGSL uniform
//! buffers are defined in terms of the host's byte order.
//!
//! Both blocks follow WGSL's uniform address space rules (the std140-compatible
//! subset): `mat4x4<f32>` is four 16-byte columns, `vec4<f32>` is 16 bytes, and
//! every member starts on a 16-byte boundary, so no implicit padding is ever
//! inserted between the members below.

use crate::frame::{DrawCommand, RenderFrame};
use crate::material::BaseColor;
use crate::registry::{PlannedDraw, Registry};

/// `Globals`: `view_proj` (64), light direction, light colour and ambient
/// (three `vec4<f32>`, 48), camera position (16), `inv_view_proj` (64) and
/// `prev_view_proj` (64).
pub(crate) const GLOBALS_SIZE: usize = 256;

/// `DrawData`: `model` (64) + `normal_matrix` (64) + `base_color` (16) +
/// `flash` (16).
///
/// The shader's comment quotes this number; if it ever changes, `lit.wgsl`
/// changes with it. (It was 144 until ADR-0017 added `flash`; the M3 plan
/// records the original.)
pub(crate) const DRAW_DATA_SIZE: usize = 160;

/// Per-draw stride, in bytes, for a device whose uniform offset alignment is
/// `alignment`.
///
/// `DrawData` is 160 bytes but a dynamic offset must be a multiple of the
/// device's `min_uniform_buffer_offset_alignment`, so consecutive draws sit
/// further apart than the struct is wide and the gap is left as padding. The
/// alignment is read from `device.limits()` rather than assumed to be 256:
/// 256 is the common value and the WebGPU default, but the limit exists
/// precisely because it is not universal, and a hard-coded 256 would be a
/// validation error on a device that reports more.
///
/// `alignment` of zero is treated as 1. A real device never reports zero, but
/// this function must be total: returning a zero stride would stack every draw
/// on top of the first one.
pub(crate) fn draw_stride(alignment: u32) -> u32 {
    let unit = alignment.max(1);
    // Real devices report a power of two (32, 64, 128, 256...), so the result
    // is also a multiple of 4 and therefore a legal `write_buffer` offset.
    (DRAW_DATA_SIZE as u32).div_ceil(unit) * unit
}

/// Capacity, in draws, that the per-draw uniform buffer should grow to in order
/// to hold `needed` draws.
///
/// Doubling rather than fitting exactly: draw counts wobble frame to frame, and
/// reallocating a GPU buffer also means rebuilding the bind group that points at
/// it. Growth is one-way — the buffer is never shrunk — because a frame that
/// once needed the space will very likely need it again.
pub(crate) fn grown_capacity(current: u32, needed: u32) -> u32 {
    if needed <= current {
        return current;
    }
    let mut capacity = current.max(1);
    while capacity < needed {
        match capacity.checked_mul(2) {
            Some(doubled) => capacity = doubled,
            // Unreachable with any real draw list; clamping beats overflowing.
            None => return needed,
        }
    }
    capacity
}

fn write_f32(out: &mut [u8], offset: usize, value: f32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_vec4(out: &mut [u8], offset: usize, value: [f32; 4]) {
    for (component, value) in value.into_iter().enumerate() {
        write_f32(out, offset + component * 4, value);
    }
}

/// Writes a column-major `mat4x4<f32>`.
///
/// `[[f32; 4]; 4]` crossing the frame boundary is already column-major — it is
/// what `glam`'s `to_cols_array_2d` produces — and WGSL matrices are
/// column-major too, so the columns are copied straight through.
fn write_mat4(out: &mut [u8], offset: usize, value: &[[f32; 4]; 4]) {
    for (column, values) in value.iter().enumerate() {
        write_vec4(out, offset + column * 16, *values);
    }
}

type Mat4 = [[f32; 4]; 4];

const IDENTITY: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// The inverse of a column-major 4x4 matrix, or `None` if it is singular.
///
/// Cofactor expansion in `f64`, then narrowed: a view-projection with a far
/// plane hundreds of metres out loses enough precision in `f32` to show in
/// reconstructed positions.
pub(crate) fn inverse(m: &Mat4) -> Option<Mat4> {
    // a[row][column], from the column-major input.
    let a: [[f64; 4]; 4] = std::array::from_fn(|row| std::array::from_fn(|column| f64::from(m[column][row])));
    // The three indices other than `skip`, without allocating: this runs every
    // frame.
    let others = |skip: usize| -> [usize; 3] {
        let mut out = [0; 3];
        for (slot, index) in (0..4).filter(|&index| index != skip).enumerate() {
            out[slot] = index;
        }
        out
    };
    let minor = |skip_row: usize, skip_column: usize| -> f64 {
        let rows = others(skip_row);
        let columns = others(skip_column);
        let e = |r: usize, c: usize| a[rows[r]][columns[c]];
        e(0, 0) * (e(1, 1) * e(2, 2) - e(1, 2) * e(2, 1)) - e(0, 1) * (e(1, 0) * e(2, 2) - e(1, 2) * e(2, 0))
            + e(0, 2) * (e(1, 0) * e(2, 1) - e(1, 1) * e(2, 0))
    };
    let cofactor = |row: usize, column: usize| {
        let sign = if (row + column).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        sign * minor(row, column)
    };
    let determinant: f64 = (0..4).map(|column| a[0][column] * cofactor(0, column)).sum();
    if !determinant.is_finite() || determinant.abs() < 1e-12 {
        return None;
    }
    // inverse[row][column] = cofactor(column, row) / det; stored column-major.
    Some(std::array::from_fn(|column| {
        std::array::from_fn(|row| (cofactor(column, row) / determinant) as f32)
    }))
}

/// Builds the `Globals` block for one frame.
///
/// A singular `view_proj` has no inverse; identity is written instead, so a
/// shader reconstructing positions gets nonsense for that frame rather than
/// infinities.
pub(crate) fn globals_bytes(frame: &RenderFrame) -> [u8; GLOBALS_SIZE] {
    let light = &frame.light;
    let ambient = &frame.ambient;
    let mut out = [0_u8; GLOBALS_SIZE];
    write_mat4(&mut out, 0, &frame.view_proj);
    // `w` of the direction is unused by the shader; the light's intensity rides
    // in `light_color.w` and the ambient intensity in `ambient_color.w`.
    write_vec4(
        &mut out,
        64,
        [light.direction[0], light.direction[1], light.direction[2], 0.0],
    );
    write_vec4(
        &mut out,
        80,
        [light.color[0], light.color[1], light.color[2], light.intensity],
    );
    write_vec4(
        &mut out,
        96,
        [
            ambient.color[0],
            ambient.color[1],
            ambient.color[2],
            ambient.intensity,
        ],
    );
    let [x, y, z] = frame.camera_position;
    write_vec4(&mut out, 112, [x, y, z, 1.0]);
    write_mat4(&mut out, 128, &inverse(&frame.view_proj).unwrap_or(IDENTITY));
    write_mat4(&mut out, 192, &frame.prev_view_proj);
    out
}

/// Builds the `DrawData` block for one draw.
pub(crate) fn draw_data_bytes(
    model: &[[f32; 4]; 4],
    normal_matrix: &[[f32; 4]; 4],
    base_color: [f32; 4],
    flash: [f32; 4],
) -> [u8; DRAW_DATA_SIZE] {
    let mut out = [0_u8; DRAW_DATA_SIZE];
    write_mat4(&mut out, 0, model);
    write_mat4(&mut out, 64, normal_matrix);
    write_vec4(&mut out, 128, base_color);
    write_vec4(&mut out, 144, flash);
    out
}

/// Packs every planned draw into one staging buffer, each at its own dynamic
/// offset.
///
/// One `Vec<u8>` uploaded with a single `write_buffer` call, rather than one
/// call per draw: the padding between draws has to be part of the upload
/// anyway, and staging it CPU-side keeps the queue traffic to a single
/// transfer per frame. The buffer is passed in so the caller can reuse its
/// allocation across frames.
pub(crate) fn pack_draw_uniforms<M: BaseColor>(
    out: &mut Vec<u8>,
    stride: usize,
    draws: &[DrawCommand],
    plan: &[PlannedDraw],
    materials: &Registry<crate::MaterialHandle, M>,
) {
    out.clear();
    out.resize(plan.len() * stride, 0);
    for planned in plan {
        // `plan_draws` only emits entries whose handles resolved, so neither
        // lookup can fail here; skipping beats indexing past the end.
        let Some(draw) = draws.get(planned.draw) else {
            continue;
        };
        let Some(material) = materials.get(draw.material) else {
            continue;
        };
        let bytes = draw_data_bytes(
            &draw.model,
            &draw.normal_matrix,
            material.base_color(),
            material.flash(),
        );
        let start = planned.slot * stride;
        out[start..start + DRAW_DATA_SIZE].copy_from_slice(&bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{AmbientLight, DirectionalLight, MaterialHandle, MeshHandle};
    use crate::material::GpuMaterial;

    const IDENTITY: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    fn read_f32(bytes: &[u8], offset: usize) -> f32 {
        let mut word = [0_u8; 4];
        word.copy_from_slice(&bytes[offset..offset + 4]);
        f32::from_le_bytes(word)
    }

    #[test]
    fn blocks_have_the_sizes_the_shader_declares() {
        assert_eq!(DRAW_DATA_SIZE, 160, "lit.wgsl documents DrawData as 160 bytes");
        assert_eq!(
            GLOBALS_SIZE, 256,
            "lit.wgsl and debug_line.wgsl declare the same Globals"
        );
    }

    /// The uniform address space requires each member to start on a 16-byte
    /// boundary and the block size to be a multiple of 16. If either fails, the
    /// GPU reads members from the wrong place and the image is subtly wrong
    /// rather than obviously broken.
    #[test]
    fn blocks_are_std140_compatible() {
        for size in [GLOBALS_SIZE, DRAW_DATA_SIZE] {
            assert_eq!(size % 16, 0, "block size {size} is not a multiple of 16");
        }
        // Globals: view_proj @0, light_direction @64, light_color @80, ambient @96,
        // camera_position @112, inv_view_proj @128, prev_view_proj @192.
        // DrawData: model @0, normal_matrix @64, base_color @128, flash @144.
        for offset in [0, 64, 80, 96, 112, 128, 144, 192] {
            assert_eq!(offset % 16, 0, "member offset {offset} is not 16-byte aligned");
        }
    }

    #[test]
    fn globals_members_land_at_their_declared_offsets() {
        let light = DirectionalLight {
            direction: [0.0, 1.0, 0.0],
            color: [0.25, 0.5, 0.75],
            intensity: 2.0,
            shadow: crate::ShadowSettings::default(),
        };
        let ambient = AmbientLight {
            color: [0.1, 0.2, 0.3],
            intensity: 0.5,
        };
        let mut frame = RenderFrame::new(IDENTITY, crate::Color::BLACK);
        frame.light = light;
        frame.ambient = ambient;
        frame.camera_position = [4.0, 5.0, 6.0];
        frame.prev_view_proj[3] = [7.0, 8.0, 9.0, 1.0];
        let bytes = globals_bytes(&frame);

        // Column-major identity: 1.0 at 0, 20, 40, 60.
        for column in 0..4 {
            assert_eq!(read_f32(&bytes, column * 16 + column * 4), 1.0);
        }
        assert_eq!(read_f32(&bytes, 64 + 4), 1.0, "light_direction.y");
        assert_eq!(read_f32(&bytes, 80), 0.25, "light_color.r");
        assert_eq!(read_f32(&bytes, 80 + 12), 2.0, "light intensity in light_color.w");
        assert_eq!(read_f32(&bytes, 96), 0.1, "ambient_color.r");
        assert_eq!(
            read_f32(&bytes, 96 + 12),
            0.5,
            "ambient intensity in ambient_color.w"
        );
        assert_eq!(read_f32(&bytes, 112), 4.0, "camera_position.x");
        assert_eq!(read_f32(&bytes, 112 + 8), 6.0, "camera_position.z");
        assert_eq!(read_f32(&bytes, 128), 1.0, "the inverse of identity is identity");
        assert_eq!(read_f32(&bytes, 192 + 48), 7.0, "prev_view_proj translation x");
    }

    fn multiply(a: &Mat4, b: &Mat4) -> Mat4 {
        std::array::from_fn(|column| {
            std::array::from_fn(|row| (0..4).map(|k| a[k][row] * b[column][k]).sum())
        })
    }

    #[test]
    fn a_matrix_times_its_inverse_is_identity() {
        // A perspective projection times a look-at view, like a real frame.
        let view_proj: Mat4 = [
            [1.2, 0.1, -0.4, -0.39],
            [0.0, 1.6, 0.3, 0.3],
            [-0.5, 0.2, -0.9, -0.88],
            [0.3, -1.1, 6.2, 6.4],
        ];
        let inverse = inverse(&view_proj).expect("invertible");
        let product = multiply(&view_proj, &inverse);
        for (column, values) in product.iter().enumerate() {
            for (row, value) in values.iter().enumerate() {
                let expected = if row == column { 1.0 } else { 0.0 };
                assert!((value - expected).abs() < 1e-4, "[{column}][{row}] = {value}");
            }
        }
    }

    #[test]
    fn a_singular_matrix_has_no_inverse_and_identity_is_written() {
        let flat: Mat4 = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0; 4],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_eq!(inverse(&flat), None);
        let bytes = globals_bytes(&RenderFrame::new(flat, crate::Color::BLACK));
        assert_eq!(read_f32(&bytes, 128 + 40), 1.0, "identity stands in");
    }

    #[test]
    fn draw_data_members_land_at_their_declared_offsets() {
        let mut model = IDENTITY;
        model[3] = [1.0, 2.0, 3.0, 1.0];
        let bytes = draw_data_bytes(&model, &IDENTITY, [0.9, 0.8, 0.7, 0.6], [0.1, 0.2, 0.3, 0.4]);

        // Translation is the fourth column of a column-major matrix: bytes 48..64.
        assert_eq!(read_f32(&bytes, 48), 1.0);
        assert_eq!(read_f32(&bytes, 52), 2.0);
        assert_eq!(read_f32(&bytes, 56), 3.0);
        assert_eq!(read_f32(&bytes, 64), 1.0, "normal_matrix starts at 64");
        assert_eq!(read_f32(&bytes, 128), 0.9, "base_color.r");
        assert_eq!(read_f32(&bytes, 140), 0.6, "base_color.a");
        assert_eq!(read_f32(&bytes, 144), 0.1, "flash.r");
        assert_eq!(read_f32(&bytes, 156), 0.4, "flash amount in flash.a");
    }

    #[test]
    fn stride_pads_up_to_the_devices_alignment() {
        assert_eq!(
            draw_stride(256),
            256,
            "the common case: one 160-byte block per 256"
        );
        assert_eq!(draw_stride(64), 192);
        assert_eq!(draw_stride(32), 160);
        assert_eq!(draw_stride(128), 256);
        // A device that needs more than the struct is wide still gets one block
        // per slot, not a truncated one.
        assert_eq!(draw_stride(512), 512);
    }

    #[test]
    fn stride_is_never_zero_and_never_smaller_than_the_block() {
        for alignment in [0, 1, 4, 16, 256] {
            let stride = draw_stride(alignment);
            assert!(
                stride as usize >= DRAW_DATA_SIZE,
                "stride {stride} truncates DrawData"
            );
            if alignment > 0 {
                assert_eq!(stride % alignment, 0, "stride {stride} breaks the dynamic offset");
            }
        }
    }

    #[test]
    fn capacity_doubles_and_never_shrinks() {
        assert_eq!(grown_capacity(64, 10), 64, "a smaller frame keeps the buffer");
        assert_eq!(grown_capacity(64, 64), 64);
        assert_eq!(grown_capacity(64, 65), 128);
        assert_eq!(grown_capacity(64, 600), 1024);
        assert_eq!(grown_capacity(0, 3), 4);
        assert_eq!(grown_capacity(0, 0), 0, "no draws needs no buffer growth");
    }

    #[test]
    fn packed_draws_sit_at_their_dynamic_offsets() {
        let mut materials = Registry::default();
        let red = materials.insert(GpuMaterial {
            base_color: [1.0, 0.0, 0.0, 1.0],
            flash: [0.0; 4],
            pbr: None,
        });
        let blue = materials.insert(GpuMaterial {
            base_color: [0.0, 0.0, 1.0, 1.0],
            flash: [0.0; 4],
            pbr: None,
        });

        let draws = vec![
            DrawCommand {
                mesh: MeshHandle::from_index(0),
                material: red,
                model: IDENTITY,
                normal_matrix: IDENTITY,
            },
            DrawCommand {
                mesh: MeshHandle::from_index(0),
                material: blue,
                model: IDENTITY,
                normal_matrix: IDENTITY,
            },
        ];
        let plan = vec![
            PlannedDraw {
                draw: 0,
                mesh: 0,
                slot: 0,
            },
            PlannedDraw {
                draw: 1,
                mesh: 0,
                slot: 1,
            },
        ];

        let stride = draw_stride(256) as usize;
        let mut packed = Vec::new();
        pack_draw_uniforms(&mut packed, stride, &draws, &plan, &materials);

        assert_eq!(packed.len(), 2 * stride);
        assert_eq!(read_f32(&packed, 128), 1.0, "first draw is red");
        assert_eq!(read_f32(&packed, stride + 128), 0.0, "second draw is not red");
        assert_eq!(read_f32(&packed, stride + 136), 1.0, "second draw is blue");
        // The gap between blocks is padding, and must be left alone.
        assert!(
            packed[DRAW_DATA_SIZE..stride].iter().all(|byte| *byte == 0),
            "padding between draws should stay zeroed"
        );
    }

    #[test]
    fn packing_reuses_the_callers_allocation() {
        let materials: Registry<MaterialHandle, GpuMaterial> = Registry::default();
        let mut packed = vec![0xAB; 4096];
        pack_draw_uniforms(&mut packed, 256, &[], &[], &materials);
        assert!(packed.is_empty(), "an empty frame packs to nothing");
        assert!(packed.capacity() >= 4096, "the allocation should survive");
    }
}
