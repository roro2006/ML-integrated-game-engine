//! Textures: engine-facing descriptions, upload, and mip generation.
//!
//! Colour space is part of the texture, not a choice the sampler or shader
//! makes later. An sRGB texture is stored in an `*Srgb` format, so the GPU
//! decodes it to linear on every sample and encodes on every write; a linear
//! texture (normals, packed occlusion-roughness-metallic) is stored as plain
//! `Unorm`. The material descriptor picks the colour space per slot (B4), so
//! a caller cannot sample albedo as linear by accident.
//!
//! Upload goes through `queue.write_texture`, which stages the data itself, so
//! no staging buffer is kept here. Asynchronous streaming is D5's problem.
//!
//! Mips are generated at upload by a chain of render-pass blits, each level
//! drawn from the one above with a linear-filtered sample at the texel corner
//! shared by four source texels. For even sizes that is an exact 2x2 box
//! filter; an odd edge loses a sliver, which is acceptable for v1 and goes
//! away when the asset cooker (B6) ships its own mips. A blit rather than a
//! compute pass because sRGB formats cannot be storage textures, and the
//! sRGB views are what make the average happen in linear light. Generation
//! runs once per texture in its own submission, not in the frame graph,
//! because it is upload work rather than per-frame work.
//!
//! Block-compressed upload (BC7, BC5) arrives with the cooker: until it
//! exists nothing produces BC data to upload.

use std::fmt;

use crate::block_on::block_on;
use crate::shaders::MIP_DOWNSAMPLE;

/// Whether a texture holds colour to be seen or data to be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    /// Colour, as an artist painted it: base colour, emissive. Decoded to
    /// linear when sampled.
    Srgb,
    /// Data: normal maps, packed occlusion-roughness-metallic. Sampled as is.
    Linear,
}

/// Whether the renderer builds the mip chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mips {
    /// One level only, for UI images and other textures seen at one size.
    Single,
    /// Generate the full chain from the uploaded level.
    Generate,
}

/// A texture to upload: RGBA8 texels, rows top to bottom, tightly packed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    pub color_space: ColorSpace,
    pub mips: Mips,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextureError {
    ZeroSize,
    /// Larger than the device's biggest 2D texture.
    TooLarge {
        max: u32,
    },
    /// The texel data is not `width * height * 4` bytes.
    DataLength {
        expected: usize,
        actual: usize,
    },
    /// The device refused the texture, usually for want of memory.
    Device(String),
}

impl fmt::Display for TextureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSize => f.write_str("a texture needs a non-zero width and height"),
            Self::TooLarge { max } => write!(f, "larger than this device's {max}-texel limit"),
            Self::DataLength { expected, actual } => {
                write!(f, "expected {expected} bytes of RGBA8 texels, got {actual}")
            }
            Self::Device(message) => write!(f, "the device refused the texture: {message}"),
        }
    }
}

impl std::error::Error for TextureError {}

pub(crate) const BYTES_PER_TEXEL: usize = 4;

/// The levels in a full mip chain: `floor(log2(max(width, height))) + 1`.
pub(crate) fn mip_count(width: u32, height: u32) -> u32 {
    let largest = width.max(height).max(1);
    u32::BITS - largest.leading_zeros()
}

/// The size of mip `level`: each level halves, rounding down, never below 1.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by readback tests until B6 cooks mips")
)]
pub(crate) fn mip_extent(width: u32, height: u32, level: u32) -> (u32, u32) {
    let shrink = |size: u32| size.checked_shr(level).unwrap_or(0).max(1);
    (shrink(width), shrink(height))
}

/// The same texels read in the other colour space.
fn other_space(format: wgpu::TextureFormat) -> wgpu::TextureFormat {
    if format.is_srgb() {
        format.remove_srgb_suffix()
    } else {
        format.add_srgb_suffix()
    }
}

pub(crate) fn format_for(color_space: ColorSpace) -> wgpu::TextureFormat {
    match color_space {
        ColorSpace::Srgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        ColorSpace::Linear => wgpu::TextureFormat::Rgba8Unorm,
    }
}

/// Checks a texture against its data and the device's size limit.
pub(crate) fn validate(desc: &TextureDesc, texels: &[u8], max_dimension: u32) -> Result<(), TextureError> {
    if desc.width == 0 || desc.height == 0 {
        return Err(TextureError::ZeroSize);
    }
    if desc.width > max_dimension || desc.height > max_dimension {
        return Err(TextureError::TooLarge { max: max_dimension });
    }
    let expected = desc.width as usize * desc.height as usize * BYTES_PER_TEXEL;
    if texels.len() != expected {
        return Err(TextureError::DataLength {
            expected,
            actual: texels.len(),
        });
    }
    Ok(())
}

/// What an uploaded texture turned out to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureInfo {
    pub width: u32,
    pub height: u32,
    /// 1 unless mips were generated.
    pub mip_levels: u32,
    pub color_space: ColorSpace,
}

/// An uploaded texture.
#[derive(Debug)]
pub(crate) struct GpuTexture {
    /// Viewable as sRGB or linear: a material slot picks (`pbr::slot_view`).
    pub(crate) texture: wgpu::Texture,
    pub(crate) color_space: ColorSpace,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) mip_levels: u32,
}

/// Creates the texture, uploads level 0, and generates the rest of the chain.
///
/// Inside error scopes: a texture within the size limit can still be more
/// than the device has memory for, and wgpu's default handler would abort the
/// process rather than let the caller refuse one bad asset.
pub(crate) fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mips: &MipGenerator,
    desc: &TextureDesc,
    texels: &[u8],
) -> Result<GpuTexture, TextureError> {
    let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let uploaded = upload_unscoped(device, queue, mips, desc, texels);
    let validation_error = block_on(validation.pop());
    let memory_error = block_on(out_of_memory.pop());
    match validation_error.or(memory_error) {
        Some(error) => Err(TextureError::Device(error.to_string())),
        None => Ok(uploaded),
    }
}

fn upload_unscoped(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mips: &MipGenerator,
    desc: &TextureDesc,
    texels: &[u8],
) -> GpuTexture {
    let format = format_for(desc.color_space);
    let mip_levels = match desc.mips {
        Mips::Single => 1,
        Mips::Generate => mip_count(desc.width, desc.height),
    };
    let size = wgpu::Extent3d {
        width: desc.width,
        height: desc.height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("texture"),
        size,
        mip_level_count: mip_levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: usage(mip_levels),
        // Either colour space can be viewed, so the material slot, not the
        // upload, decides how texels are read.
        view_formats: &[other_space(format)],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        texels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(desc.width * BYTES_PER_TEXEL as u32),
            rows_per_image: Some(desc.height),
        },
        size,
    );
    if mip_levels > 1 {
        mips.generate(device, queue, &texture, format, mip_levels);
    }
    GpuTexture {
        texture,
        color_space: desc.color_space,
        width: desc.width,
        height: desc.height,
        mip_levels,
    }
}

/// What a texture is used for. Render-attachment use is asked for only when
/// mips are generated into it, since some drivers store attachments less
/// compactly; copying out is for the readback tests only.
fn usage(mip_levels: u32) -> wgpu::TextureUsages {
    let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
    if mip_levels > 1 {
        usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }
    if cfg!(test) {
        usage |= wgpu::TextureUsages::COPY_SRC;
    }
    usage
}

/// Builds mip chains: one downsample pipeline per texture format.
#[derive(Debug)]
pub(crate) struct MipGenerator {
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    srgb: wgpu::RenderPipeline,
    linear: wgpu::RenderPipeline,
}

impl GpuTexture {
    pub(crate) fn info(&self) -> TextureInfo {
        TextureInfo {
            width: self.width,
            height: self.height,
            mip_levels: self.mip_levels,
            color_space: self.color_space,
        }
    }
}

impl MipGenerator {
    pub(crate) fn new(device: &wgpu::Device, source: &str) -> Result<Self, String> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mip-downsample-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mip-downsample-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mip-downsample-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let srgb =
            create_downsample_pipeline(device, &pipeline_layout, source, format_for(ColorSpace::Srgb))?;
        let linear =
            create_downsample_pipeline(device, &pipeline_layout, source, format_for(ColorSpace::Linear))?;
        Ok(Self {
            layout,
            sampler,
            srgb,
            linear,
        })
    }

    fn generate(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        format: wgpu::TextureFormat,
        mip_levels: u32,
    ) {
        let pipeline = if format.is_srgb() {
            &self.srgb
        } else {
            &self.linear
        };
        let level_view = |level: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("mip-level"),
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mip-generation"),
        });
        for level in 1..mip_levels {
            let source = level_view(level - 1);
            let target = level_view(level);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mip-downsample-bind-group"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mip-downsample-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
}

fn create_downsample_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    source: &str,
    format: wgpu::TextureFormat,
) -> Result<wgpu::RenderPipeline, String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(MIP_DOWNSAMPLE),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let targets = [Some(wgpu::ColorTargetState {
        format,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    })];
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mip-downsample-pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &targets,
        }),
        multiview_mask: None,
        cache: None,
    });
    match block_on(scope.pop()) {
        Some(error) => Err(error.to_string()),
        None => Ok(pipeline),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Image, RenderError, Renderer, RendererConfig, TextureHandle};

    fn headless() -> Renderer {
        Renderer::headless(16, 16, RendererConfig::default())
            .expect("these tests are ignored unless a GPU adapter exists")
    }

    impl Renderer {
        /// One mip level of an uploaded texture, as stored.
        fn read_texture_level(&self, handle: TextureHandle, level: u32) -> Result<Image, RenderError> {
            let uploaded = self
                .textures
                .get(handle)
                .ok_or_else(|| RenderError::Readback("unknown texture".to_owned()))?;
            let (width, height) = mip_extent(uploaded.width, uploaded.height, level);
            self.read_texture(&uploaded.texture, level, width, height)
        }
    }

    /// 2x2: black, white / white, black.
    fn checker() -> Vec<u8> {
        [0u8, 255, 255, 0]
            .iter()
            .flat_map(|&value| [value, value, value, 255])
            .collect()
    }

    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn an_upload_reads_back_exactly() {
        let mut renderer = headless();
        // 3x2: not a power of two, so row handling cannot hide behind padding.
        let texels: Vec<u8> = (0..24).map(|i| (i * 10) as u8).collect();
        let handle = renderer
            .create_texture(
                &TextureDesc {
                    width: 3,
                    height: 2,
                    color_space: ColorSpace::Linear,
                    mips: Mips::Single,
                },
                &texels,
            )
            .expect("valid");
        assert_eq!(renderer.texture_info(handle).map(|info| info.mip_levels), Some(1));
        let level0 = renderer.read_texture_level(handle, 0).expect("reads back");
        assert_eq!(level0.rgba8, texels);
    }

    /// The colour-space acceptance check for B3: a black and white checker
    /// averages to half the *light*, which an sRGB texture stores as code 188,
    /// while a linear texture averages the stored numbers to 128. A mip chain
    /// built from encoded values would make every distant sRGB texture too
    /// dark.
    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn srgb_mips_average_light_and_linear_mips_average_values() {
        let mut renderer = headless();
        for (color_space, expected) in [(ColorSpace::Srgb, 188u8), (ColorSpace::Linear, 128)] {
            let handle = renderer
                .create_texture(
                    &TextureDesc {
                        width: 2,
                        height: 2,
                        color_space,
                        mips: Mips::Generate,
                    },
                    &checker(),
                )
                .expect("valid");
            assert_eq!(renderer.texture_info(handle).map(|info| info.mip_levels), Some(2));
            let level1 = renderer.read_texture_level(handle, 1).expect("reads back");
            let [r, g, b, a] = level1.pixel(0, 0).expect("1x1");
            for channel in [r, g, b] {
                assert!(
                    channel.abs_diff(expected) <= 1,
                    "{color_space:?}: got {channel}, expected {expected}"
                );
            }
            assert_eq!(a, 255);
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn a_non_square_texture_gets_its_whole_chain() {
        let mut renderer = headless();
        let texels = vec![200; 8 * 2 * 4];
        let handle = renderer
            .create_texture(
                &TextureDesc {
                    width: 8,
                    height: 2,
                    color_space: ColorSpace::Srgb,
                    mips: Mips::Generate,
                },
                &texels,
            )
            .expect("valid");
        assert_eq!(renderer.texture_info(handle).map(|info| info.mip_levels), Some(4));
        let last = renderer.read_texture_level(handle, 3).expect("reads back");
        assert_eq!((last.width, last.height), (1, 1));
        let [r, ..] = last.pixel(0, 0).expect("1x1");
        assert!(
            r.abs_diff(200) <= 1,
            "a flat texture stays flat down the chain, got {r}"
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter and driver"]
    fn a_bad_upload_is_refused_without_reaching_the_gpu() {
        let mut renderer = headless();
        let result = renderer.create_texture(
            &TextureDesc {
                width: 2,
                height: 2,
                color_space: ColorSpace::Srgb,
                mips: Mips::Generate,
            },
            &[0; 3],
        );
        assert!(matches!(
            result,
            Err(RenderError::TextureUpload(TextureError::DataLength {
                expected: 16,
                actual: 3
            }))
        ));
    }

    fn desc(width: u32, height: u32) -> TextureDesc {
        TextureDesc {
            width,
            height,
            color_space: ColorSpace::Srgb,
            mips: Mips::Generate,
        }
    }

    #[test]
    fn a_full_chain_has_floor_log2_of_the_larger_side_plus_one_levels() {
        assert_eq!(mip_count(1, 1), 1);
        assert_eq!(mip_count(2, 2), 2);
        assert_eq!(mip_count(256, 256), 9);
        assert_eq!(mip_count(255, 1), 8);
        assert_eq!(mip_count(300, 20), 9, "the larger side decides");
        assert_eq!(mip_count(1, 4096), 13);
    }

    #[test]
    fn each_level_halves_and_rounds_down_but_never_below_one() {
        assert_eq!(mip_extent(300, 20, 0), (300, 20));
        assert_eq!(mip_extent(300, 20, 1), (150, 10));
        assert_eq!(mip_extent(300, 20, 5), (9, 1));
        assert_eq!(mip_extent(300, 20, 8), (1, 1), "the last level is 1x1");
        assert_eq!(
            mip_extent(300, 20, 40),
            (1, 1),
            "a shift past the width is not a panic"
        );
    }

    #[test]
    fn only_a_mipped_texture_can_be_rendered_into() {
        assert!(!usage(1).contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
        assert!(usage(4).contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
        assert!(usage(1).contains(wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST));
    }

    #[test]
    fn colour_space_decides_the_format() {
        assert_eq!(format_for(ColorSpace::Srgb), wgpu::TextureFormat::Rgba8UnormSrgb);
        assert_eq!(format_for(ColorSpace::Linear), wgpu::TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn uploads_are_validated_before_reaching_the_gpu() {
        assert_eq!(validate(&desc(2, 2), &[0; 16], 4096), Ok(()));
        assert_eq!(validate(&desc(0, 2), &[], 4096), Err(TextureError::ZeroSize));
        assert_eq!(
            validate(&desc(8192, 1), &[0; 8192 * 4], 4096),
            Err(TextureError::TooLarge { max: 4096 })
        );
        assert_eq!(
            validate(&desc(2, 2), &[0; 15], 4096),
            Err(TextureError::DataLength {
                expected: 16,
                actual: 15,
            })
        );
    }
}
