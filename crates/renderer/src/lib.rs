//! Rendering backend.
//!
//! wgpu types stay inside this crate: surface configuration, adapter
//! selection, and frame submission are expressed through the engine-owned
//! types below. See `docs/adr/0003-renderer-gpu-api-boundary.md`.
//!
//! Scope as of M3: acquire and present a surface, upload meshes, register
//! materials, and draw a submitted [`RenderFrame`] with one directional light
//! and a flat ambient term.
//!
//! The backend is deliberately dumb. Culling and draw ordering happen in the
//! engine, so a frame arrives as a finished list and this crate draws exactly
//! what it is given, in the order given. Textures and transparency sorting are
//! later work.
//!
//! A frame is drawn in two passes that share one render pass and one depth
//! buffer: opaque geometry, then world-space debug lines that test depth
//! without writing it (see [`debug_lines`]).
//!
//! Frames go to a window's swapchain, or, for a renderer built with
//! [`Renderer::headless`], to an owned texture that can be read back as an
//! [`Image`]. Both draw through the same code, which is what makes a readback a
//! fair stand-in for what a window shows.

mod block_on;
mod debug_lines;
mod frame;
#[cfg(test)]
mod golden;
mod gpu_timing;
pub use gpu_timing::PassTiming;
mod graph;
mod image;
mod material;
mod mesh;
mod mesh_data;
mod passes;
mod pbr;
mod permutation;
mod pipeline;
mod post;
mod registry;
mod sampler;
mod shaders;
mod tangents;
#[cfg(test)]
mod test_scene;
mod texture;
mod uniforms;

pub use debug_lines::DebugVertex;
pub use frame::{
    AlphaMode, AmbientLight, AutoExposure, Bloom, DEFAULT_EXPOSURE_EV, DirectionalLight, DrawCommand, Grade,
    MaterialDesc, MaterialHandle, MaterialTextures, MeshHandle, PostSettings, RenderFrame, Shading,
    TextureHandle, Tonemapper,
};
pub use image::Image;
pub use mesh::MeshInfo;
pub use mesh_data::{Aabb, MeshData, MeshDataError, VERTEX_FLOATS, WHITE};
pub use sampler::{Filter, SamplerDesc, Wrap};
pub use shaders::{DEBUG_LINE, LIT, ShaderError, ShaderLibrary};
pub use texture::{ColorSpace, Mips, TextureDesc, TextureError, TextureInfo};

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use graph::resources::TexturePool;
use material::GpuMaterial;
use mesh::GpuMesh;
use post::tonemap;
use registry::Registry;

use wgpu::rwh::{HasDisplayHandle, HasWindowHandle};

use block_on::block_on;

/// Linear clear/render colour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

impl Color {
    pub const BLACK: Self = Self::rgb(0.0, 0.0, 0.0);

    pub const fn rgb(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b, a: 1.0 }
    }
}

/// Presentation pacing preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentMode {
    /// Wait for vsync. Always supported; the default for battery and thermals.
    Vsync,
    /// Prefer the lowest-latency mode the surface supports.
    LowLatency,
}

/// Which GPU to prefer when a system has more than one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPreference {
    /// Prefer integrated GPUs; useful for validating the low-spec target.
    LowPower,
    HighPerformance,
}

/// Graphics backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendPreference {
    /// Let wgpu pick the platform default (Vulkan on Linux, DX12/Vulkan on Windows).
    Auto,
    /// Force Vulkan, e.g. to match SteamOS behaviour while testing on Windows.
    Vulkan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RendererConfig {
    pub present_mode: PresentMode,
    pub gpu_preference: GpuPreference,
    pub backend: BackendPreference,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            present_mode: PresentMode::Vsync,
            gpu_preference: GpuPreference::HighPerformance,
            backend: BackendPreference::Auto,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    IntegratedGpu,
    DiscreteGpu,
    VirtualGpu,
    Cpu,
    Other,
}

impl fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::IntegratedGpu => "integrated-gpu",
            Self::DiscreteGpu => "discrete-gpu",
            Self::VirtualGpu => "virtual-gpu",
            Self::Cpu => "cpu",
            Self::Other => "other",
        })
    }
}

/// Selected GPU, reported for diagnostics and performance profiles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterInfo {
    pub name: String,
    pub backend: String,
    pub device_kind: DeviceKind,
    pub driver: String,
    /// Optional GPU features this device was created with.
    pub features: GpuFeatures,
}

/// Optional features, requested only when the adapter has them; see
/// `docs/adr/0023-device-limits-and-optional-features.md`. Everything that
/// uses one has a path that works without it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GpuFeatures {
    /// BC1 to BC7 block-compressed textures: desktop GPUs have them, some
    /// integrated and mobile ones do not.
    pub bc_compression: bool,
}

/// The optional features the renderer asks for when the adapter offers them.
///
/// Only what something uses or reports: a feature can change which driver
/// paths are taken, so texture binding arrays wait for the bindless path.
const OPTIONAL_FEATURES: wgpu::Features = wgpu::Features::TEXTURE_COMPRESSION_BC.union(gpu_timing::FEATURES);

impl GpuFeatures {
    fn from_wgpu(features: wgpu::Features) -> Self {
        Self {
            bc_compression: features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC),
        }
    }
}

#[derive(Debug)]
pub enum RenderError {
    CreateSurface(String),
    NoAdapter(String),
    RequestDevice(String),
    UnsupportedSurface(String),
    /// Unrecoverable surface failure (device loss, out of memory).
    Surface(String),
    /// The mesh failed its own invariants, so nothing was uploaded.
    MeshUpload(MeshDataError),
    /// The texture failed validation, so nothing was uploaded.
    TextureUpload(TextureError),
    /// WGSL source could not be resolved.
    Shader(ShaderError),
    /// WGSL compiled into a pipeline that the GPU rejected. Carries the
    /// validation message, which names the line for a syntax error.
    ShaderCompile {
        shader: String,
        message: String,
    },
    /// The frame's render graph was declared out of order or a pass failed to
    /// record. Always a renderer bug, never bad content.
    Graph(String),
    /// A frame could not be read back: the renderer draws to a window, the
    /// frame was skipped, or mapping the copy failed.
    Readback(String),
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CreateSurface(message) => write!(f, "cannot create surface: {message}"),
            Self::NoAdapter(message) => write!(f, "no suitable GPU adapter: {message}"),
            Self::RequestDevice(message) => write!(f, "cannot create device: {message}"),
            Self::UnsupportedSurface(message) => write!(f, "unsupported surface: {message}"),
            Self::Surface(message) => write!(f, "surface failure: {message}"),
            Self::MeshUpload(error) => write!(f, "cannot upload mesh: {error}"),
            Self::TextureUpload(error) => write!(f, "cannot upload texture: {error}"),
            Self::Shader(error) => write!(f, "{error}"),
            Self::ShaderCompile { shader, message } => {
                write!(f, "cannot build a pipeline for {shader}: {message}")
            }
            Self::Graph(message) => write!(f, "render graph error: {message}"),
            Self::Readback(message) => write!(f, "cannot read the frame back: {message}"),
        }
    }
}

impl std::error::Error for RenderError {}

/// Per-frame counters for the performance overlay.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameStats {
    /// Delayed GPU readings; absent before readback or without timestamp support.
    pub gpu_passes: Option<Vec<PassTiming>>,
    /// Frame associated with the GPU readings, which lag the CPU counters.
    pub gpu_frame: Option<u64>,
    pub draw_calls: u32,
    /// Triangles submitted this frame, after engine-side culling.
    pub triangles: u32,
    /// CPU time spent recording and submitting the frame.
    pub cpu_time: Duration,
    /// Debug line segments drawn this frame.
    ///
    /// Counted separately rather than folded into `draw_calls` and `triangles`:
    /// those two describe the scene, and an overlay that inflated them would
    /// make the performance numbers lie about the thing being profiled. The
    /// whole debug pass is one draw call whatever this number is.
    pub debug_lines: u32,
}

/// Why a frame produced no presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// The window is minimized or has a zero-sized surface.
    ZeroSizedSurface,
    /// The swapchain timed out; the next frame will retry.
    SurfaceTimeout,
    /// The window is occluded (minimized or fully covered by another window).
    SurfaceOccluded,
    /// The swapchain was lost or outdated and has been reconfigured.
    SurfaceReconfigured,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FrameOutcome {
    Presented(FrameStats),
    Skipped(SkipReason),
}

/// Draws the per-draw uniform buffer starts out able to hold.
///
/// Sized so a first-pass blocked-out scene never reallocates on its opening
/// frame; the buffer grows from here and is never shrunk.
const INITIAL_DRAW_CAPACITY: u32 = 64;

/// The colour format of a headless renderer's target.
///
/// sRGB, like the swapchain formats `Renderer::new` prefers, so a readback holds
/// the same display-encoded bytes a window would show.
const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Where frames go.
#[derive(Debug)]
enum Output {
    /// A window's swapchain. `config` holds the current size.
    Window {
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
    },
    /// An owned texture that frames can be read back from. `None` while the
    /// size is zero, since a zero-extent texture is a validation error.
    Offscreen { texture: Option<wgpu::Texture> },
}

/// Staged pipelines are installed only after their whole shader family compiles.
enum BuiltShader {
    Lit(wgpu::RenderPipeline),
    Debug(wgpu::RenderPipeline),
    Tonemap(wgpu::RenderPipeline),
    Pbr(BTreeMap<permutation::PermutationKey, wgpu::RenderPipeline>),
    Bloom(post::bloom::BloomResources),
    Exposure(post::exposure::ExposureResources),
    Mips(texture::MipGenerator),
}

/// Owns the GPU device, queue, and output (a window's swapchain or an
/// offscreen texture).
#[derive(Debug)]
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    output: Output,
    color_format: wgpu::TextureFormat,
    /// The output's size; zero while a window is minimized.
    size: (u32, u32),
    adapter_info: AdapterInfo,

    shaders: ShaderLibrary,
    pipeline_layout: wgpu::PipelineLayout,
    draw_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    /// The last hot-reload failure, kept so the engine can show it on the debug
    /// overlay. The renderer does not log: that is the engine layer's job.
    shader_error: Option<String>,
    /// The failure behind each stale pipeline, by shader name. `shader_error`
    /// is these joined, so fixing one shader never hides another still broken.
    shader_failures: BTreeMap<&'static str, String>,

    /// Textures the render graph allocates per frame, the depth buffer among
    /// them.
    transients: TexturePool<wgpu::Texture>,
    gpu_timer: Option<gpu_timing::GpuTimer>,
    graph_dump: String,

    meshes: Registry<MeshHandle, GpuMesh>,
    materials: Registry<MaterialHandle, GpuMaterial>,

    globals_buffer: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    draw_buffer: wgpu::Buffer,
    draw_bind_group: wgpu::BindGroup,
    /// Bytes between consecutive `DrawData` blocks; see `uniforms::draw_stride`.
    draw_stride: u32,
    draw_capacity: u32,
    /// Reused between frames so per-draw uniform packing does not allocate.
    draw_staging: Vec<u8>,
    skipped_draws: u32,

    /// The debug line pass. It binds group 0 only, so it needs its own pipeline
    /// layout rather than sharing the lit pass's two-group one.
    debug_pipeline_layout: wgpu::PipelineLayout,
    debug_pipeline: wgpu::RenderPipeline,
    debug_vertex_buffer: wgpu::Buffer,
    debug_capacity: u32,
    /// Reused between frames, like `draw_staging`.
    debug_staging: Vec<u8>,

    /// The tonemap pass: HDR scene to swapchain. Its bind group points at a
    /// pooled texture, so it is built per frame from this layout.
    tonemap_layout: wgpu::BindGroupLayout,
    tonemap_pipeline_layout: wgpu::PipelineLayout,
    tonemap_pipeline: wgpu::RenderPipeline,
    post_buffer: wgpu::Buffer,

    textures: Registry<TextureHandle, texture::GpuTexture>,
    mips: texture::MipGenerator,

    /// The PBR material: group 2's layout, a pipeline per permutation, all
    /// built at startup, and the textures empty slots fall back to.
    material_layout: wgpu::BindGroupLayout,
    pbr_pipeline_layout: wgpu::PipelineLayout,
    pbr_pipelines: BTreeMap<permutation::PermutationKey, wgpu::RenderPipeline>,
    default_textures: pbr::DefaultTextures,
    samplers: sampler::SamplerCache<wgpu::Sampler>,

    /// Automatic exposure's buffers and pipelines, and when the last frame
    /// was drawn, which sets how far exposure adapts.
    exposure: post::exposure::ExposureResources,
    last_frame: Option<Instant>,
    bloom: post::bloom::BloomResources,
}

impl Renderer {
    /// Creates a renderer for `window`, which must outlive the renderer.
    pub fn new<W>(window: W, width: u32, height: u32, config: RendererConfig) -> Result<Self, RenderError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        let instance = create_instance(config);
        let surface = instance
            .create_surface(window)
            .map_err(|error| RenderError::CreateSurface(error.to_string()))?;
        let adapter = request_adapter(&instance, config, Some(&surface))?;
        let (device, queue) = request_device(&adapter)?;

        let capabilities = surface.get_capabilities(&adapter);
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(|format| format.is_srgb())
            .or_else(|| capabilities.formats.first().copied())
            .ok_or_else(|| {
                RenderError::UnsupportedSurface("adapter reports no surface formats".to_owned())
            })?;
        let present_mode = select_present_mode(config.present_mode, &capabilities.present_modes);
        let alpha_mode = capabilities
            .alpha_modes
            .first()
            .copied()
            .unwrap_or(wgpu::CompositeAlphaMode::Auto);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            // Auto is supported for every format the adapter reports.
            color_space: wgpu::SurfaceColorSpace::Auto,
            width,
            height,
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: Vec::new(),
        };
        let output = Output::Window {
            surface,
            config: surface_config,
        };
        let info = adapter_info(&adapter, &device);
        Self::assemble(device, queue, info, output, format, (width, height))
    }

    /// Creates a renderer with no window, drawing into an offscreen texture of
    /// `width` by `height` that [`render_to_image`](Self::render_to_image) reads
    /// back.
    ///
    /// For golden-image tests and tools. Device limits and every pipeline are
    /// the same as for a window. Adapter selection follows the same preference
    /// but has no surface to match, so on a machine with two GPUs it may pick a
    /// different one than a windowed run; check [`adapter_info`](Self::adapter_info).
    ///
    /// A size larger than the device's largest 2D texture is an error rather
    /// than a validation failure later.
    pub fn headless(width: u32, height: u32, config: RendererConfig) -> Result<Self, RenderError> {
        Self::headless_with_format(width, height, config, OFFSCREEN_FORMAT)
    }

    /// A headless renderer with a chosen target format. Tests use a non-sRGB
    /// one to exercise the shader-side encode a window falls back to when its
    /// surface offers no sRGB format.
    pub(crate) fn headless_with_format(
        width: u32,
        height: u32,
        config: RendererConfig,
        format: wgpu::TextureFormat,
    ) -> Result<Self, RenderError> {
        let instance = create_instance(config);
        let adapter = request_adapter(&instance, config, None)?;
        let (device, queue) = request_device(&adapter)?;
        let max = device.limits().max_texture_dimension_2d;
        if width > max || height > max {
            return Err(RenderError::UnsupportedSurface(format!(
                "{width}x{height} exceeds the largest texture this device supports ({max})"
            )));
        }
        let output = Output::Offscreen { texture: None };
        let info = adapter_info(&adapter, &device);
        Self::assemble(device, queue, info, output, format, (width, height))
    }

    /// Builds the pipelines and per-frame resources shared by every output.
    fn assemble(
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_info: AdapterInfo,
        output: Output,
        color_format: wgpu::TextureFormat,
        size: (u32, u32),
    ) -> Result<Self, RenderError> {
        // Shaders start embedded. Hot reload is opt-in through
        // `enable_shader_hot_reload`, so a renderer is always constructible from
        // the binary alone and a missing asset directory cannot stop startup.
        let mut shaders = ShaderLibrary::new(None);
        let globals_layout = pipeline::globals_layout(&device);
        let draw_layout = pipeline::draw_layout(&device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lit-pipeline-layout"),
            bind_group_layouts: &[Some(&globals_layout), Some(&draw_layout)],
            immediate_size: 0,
        });
        let pipeline = {
            let source = shaders
                .source(shaders::LIT)
                .map_err(RenderError::Shader)?
                .into_owned();
            pipeline::create_pipeline(&device, &pipeline_layout, &source, passes::HDR_FORMAT).map_err(
                |message| RenderError::ShaderCompile {
                    shader: shaders::LIT.to_owned(),
                    message,
                },
            )?
        };

        // The debug pass reuses the *same* globals layout and, at draw time, the
        // same globals buffer: it needs the view-projection and nothing else, so
        // a second per-frame uniform would be a second copy of data already
        // uploaded. Only the pipeline layout differs, because there is no
        // per-draw group.
        let debug_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("debug-line-pipeline-layout"),
            bind_group_layouts: &[Some(&globals_layout)],
            immediate_size: 0,
        });
        let debug_pipeline = {
            let source = shaders
                .source(shaders::DEBUG_LINE)
                .map_err(RenderError::Shader)?
                .into_owned();
            debug_lines::create_pipeline(&device, &debug_pipeline_layout, &source, color_format).map_err(
                |message| RenderError::ShaderCompile {
                    shader: shaders::DEBUG_LINE.to_owned(),
                    message,
                },
            )?
        };
        let debug_vertex_buffer =
            debug_lines::create_vertex_buffer(&device, debug_lines::INITIAL_VERTEX_CAPACITY);

        let material_layout = pbr::material_layout(&device);
        let pbr_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pbr-pipeline-layout"),
            bind_group_layouts: &[Some(&globals_layout), Some(&draw_layout), Some(&material_layout)],
            immediate_size: 0,
        });
        let pbr_pipelines = build_pbr_pipelines(&device, &mut shaders, &pbr_pipeline_layout)?;

        let tonemap_layout = tonemap::bind_group_layout(&device);
        let tonemap_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tonemap-pipeline-layout"),
            bind_group_layouts: &[Some(&tonemap_layout)],
            immediate_size: 0,
        });
        let tonemap_pipeline =
            build_tonemap_pipeline(&device, &mut shaders, &tonemap_pipeline_layout, color_format)?;
        let post_buffer = tonemap::create_post_buffer(&device);
        let bloom = {
            let expanded = shaders.expanded(shaders::BLOOM).map_err(RenderError::Shader)?;
            post::bloom::BloomResources::new(&device, &expanded.text).map_err(|message| {
                RenderError::ShaderCompile {
                    shader: shaders::BLOOM.to_owned(),
                    message: expanded.annotate(&message),
                }
            })?
        };
        let exposure = {
            let source = shaders
                .source(shaders::EXPOSURE_HISTOGRAM)
                .map_err(RenderError::Shader)?
                .into_owned();
            post::exposure::ExposureResources::new(&device, &source).map_err(|message| {
                RenderError::ShaderCompile {
                    shader: shaders::EXPOSURE_HISTOGRAM.to_owned(),
                    message,
                }
            })?
        };
        let mips = {
            let source = shaders
                .source(shaders::MIP_DOWNSAMPLE)
                .map_err(RenderError::Shader)?
                .into_owned();
            texture::MipGenerator::new(&device, &source).map_err(|message| RenderError::ShaderCompile {
                shader: shaders::MIP_DOWNSAMPLE.to_owned(),
                message,
            })?
        };

        // Read from the device rather than assumed: 256 is the common value for
        // `min_uniform_buffer_offset_alignment`, but it is a limit precisely
        // because hardware disagrees, and a wrong stride is a validation error
        // on every single draw.
        let draw_stride = uniforms::draw_stride(device.limits().min_uniform_buffer_offset_alignment);
        let (globals_buffer, globals_bind_group) =
            pipeline::create_globals_resources(&device, &globals_layout);
        let (draw_buffer, draw_bind_group) =
            pipeline::create_draw_resources(&device, &draw_layout, draw_stride, INITIAL_DRAW_CAPACITY);

        let default_textures = {
            let one = |texel: [u8; 4], color_space| {
                let desc = TextureDesc {
                    width: 1,
                    height: 1,
                    color_space,
                    mips: Mips::Single,
                };
                texture::upload(&device, &queue, &mips, &desc, &texel).map_err(RenderError::TextureUpload)
            };
            pbr::DefaultTextures {
                white: one(pbr::WHITE_TEXEL, ColorSpace::Srgb)?,
                flat_normal: one(pbr::FLAT_NORMAL_TEXEL, ColorSpace::Linear)?,
            }
        };

        let gpu_timer = gpu_timing::GpuTimer::new(&device, &queue);
        let mut renderer = Self {
            device,
            queue,
            output,
            color_format,
            size,
            adapter_info,
            shaders,
            pipeline_layout,
            draw_layout,
            pipeline,
            shader_error: None,
            shader_failures: BTreeMap::new(),
            transients: TexturePool::default(),
            gpu_timer,
            graph_dump: String::new(),
            meshes: Registry::default(),
            materials: Registry::default(),
            globals_buffer,
            globals_bind_group,
            draw_buffer,
            draw_bind_group,
            draw_stride,
            draw_capacity: INITIAL_DRAW_CAPACITY,
            draw_staging: Vec::new(),
            skipped_draws: 0,
            debug_pipeline_layout,
            debug_pipeline,
            debug_vertex_buffer,
            debug_capacity: debug_lines::INITIAL_VERTEX_CAPACITY,
            debug_staging: Vec::new(),
            tonemap_layout,
            tonemap_pipeline_layout,
            tonemap_pipeline,
            post_buffer,
            textures: Registry::default(),
            mips,
            material_layout,
            pbr_pipeline_layout,
            pbr_pipelines,
            default_textures,
            samplers: sampler::SamplerCache::default(),
            exposure,
            last_frame: None,
            bloom,
        };
        renderer.configure_output();
        Ok(renderer)
    }

    /// Configures the swapchain, or rebuilds the offscreen texture, at the
    /// current size. A zero size configures nothing.
    fn configure_output(&mut self) {
        let (width, height) = self.size;
        match &mut self.output {
            Output::Window { config, .. } => {
                config.width = width;
                config.height = height;
                self.configure_surface();
            }
            Output::Offscreen { texture } => {
                // An oversized resize leaves no texture, so frames are skipped
                // and a readback reports it, instead of failing validation.
                let max = self.device.limits().max_texture_dimension_2d;
                *texture = pipeline::depth_size(width, height)
                    .filter(|&(width, height)| width <= max && height <= max)
                    .map(|(width, height)| {
                        create_offscreen_texture(&self.device, self.color_format, width, height)
                    });
            }
        }
    }

    /// Reconfigures the swapchain after it was lost or went stale. Does nothing
    /// for an offscreen output, which cannot go stale.
    fn configure_surface(&self) {
        if let Output::Window { surface, config } = &self.output
            && config.width > 0
            && config.height > 0
        {
            surface.configure(&self.device, config);
        }
    }

    /// Resizes the output. A zero size is recorded but not configured.
    pub fn resize(&mut self, width: u32, height: u32) {
        if (width, height) == self.size {
            return;
        }
        self.size = (width, height);
        self.configure_output();
        // Every transient is sized to the output, so none survives a resize.
        // The pool would retire them after a frame anyway; clearing returns
        // the memory now, which matters while a window is being dragged.
        self.transients.clear();
    }

    pub fn surface_size(&self) -> (u32, u32) {
        self.size
    }

    pub fn adapter_info(&self) -> &AdapterInfo {
        &self.adapter_info
    }

    /// Uploads a mesh and returns a handle to it.
    ///
    /// The mesh is validated first: an out-of-range index or a short attribute
    /// array is a content bug, and rejecting it here produces a message naming
    /// the problem instead of a driver-level fault or silently wrong geometry.
    pub fn upload_mesh(&mut self, data: &MeshData) -> Result<MeshHandle, RenderError> {
        data.validate().map_err(RenderError::MeshUpload)?;
        // `validate` guarantees at least one vertex, so bounds always exist.
        let bounds = data
            .bounds()
            .ok_or(RenderError::MeshUpload(MeshDataError::Empty))?;

        let vertices = mesh::float_bytes(&data.interleaved());
        let indices = mesh::index_bytes(&data.indices);
        let vertex_buffer = self.upload_buffer("mesh-vertices", &vertices, wgpu::BufferUsages::VERTEX);
        let index_buffer = self.upload_buffer("mesh-indices", &indices, wgpu::BufferUsages::INDEX);

        // Four billion indices is not a reachable mesh; saturating keeps this
        // total without an unwrap.
        let index_count = u32::try_from(data.indices.len()).unwrap_or(u32::MAX);
        Ok(self.meshes.insert(GpuMesh {
            vertex_buffer,
            index_buffer,
            index_count,
            triangle_count: index_count / 3,
            bounds,
        }))
    }

    /// Uploads RGBA8 texels and returns a handle to the texture.
    ///
    /// The texels are validated against the description first, and mips are
    /// generated here when asked for, so the texture is complete before any
    /// frame can sample it.
    pub fn create_texture(
        &mut self,
        desc: &TextureDesc,
        texels: &[u8],
    ) -> Result<TextureHandle, RenderError> {
        let max = self.device.limits().max_texture_dimension_2d;
        texture::validate(desc, texels, max).map_err(RenderError::TextureUpload)?;
        let uploaded = texture::upload(&self.device, &self.queue, &self.mips, desc, texels)
            .map_err(RenderError::TextureUpload)?;
        Ok(self.textures.insert(uploaded))
    }

    /// Size, mip count and colour space of an uploaded texture, or `None` for
    /// an unknown handle.
    pub fn texture_info(&self, handle: TextureHandle) -> Option<TextureInfo> {
        self.textures.get(handle).map(texture::GpuTexture::info)
    }

    /// Registers a material and returns a handle to it.
    ///
    /// A PBR material's bind group is built here, once. A texture handle that
    /// resolves to nothing falls back to the slot's neutral texture, so a
    /// missing asset shows as an untextured surface rather than failing.
    pub fn create_material(&mut self, desc: &MaterialDesc) -> MaterialHandle {
        let mut material = GpuMaterial::from_desc(desc);
        if desc.shading == Shading::Pbr {
            material.pbr = Some(self.pbr_binding(desc));
        }
        self.materials.insert(material)
    }

    fn pbr_binding(&mut self, desc: &MaterialDesc) -> material::PbrBinding {
        use pbr::SlotSpace;

        let params = self.upload_buffer(
            "material-params",
            &pbr::material_params_bytes(desc),
            wgpu::BufferUsages::UNIFORM,
        );
        let slot = |handle: Option<TextureHandle>, fallback: &texture::GpuTexture, space: SlotSpace| {
            let uploaded = handle
                .and_then(|handle| self.textures.get(handle))
                .unwrap_or(fallback);
            pbr::slot_view(uploaded, space)
        };
        let defaults = &self.default_textures;
        let base_color = slot(desc.textures.base_color, &defaults.white, SlotSpace::Srgb);
        let normal = slot(desc.textures.normal, &defaults.flat_normal, SlotSpace::Linear);
        let orm = slot(desc.textures.orm, &defaults.white, SlotSpace::Linear);
        let emissive = slot(desc.textures.emissive, &defaults.white, SlotSpace::Srgb);
        let device = &self.device;
        let sampler = self.samplers.get_or_create(desc.sampler, |normalised| {
            device.create_sampler(&normalised.to_wgpu())
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material-bind-group"),
            layout: &self.material_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&base_color),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&normal),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&orm),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&emissive),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        material::PbrBinding {
            key: permutation::PermutationKey::for_material(desc),
            bind_group,
        }
    }

    /// PBR pipelines compiled, one per permutation.
    pub fn pbr_pipeline_count(&self) -> usize {
        self.pbr_pipelines.len()
    }

    /// Counts and bounds for an uploaded mesh, or `None` for an unknown handle.
    ///
    /// Bounds live here because upload is where they are known and the engine's
    /// frustum culling is where they are needed.
    pub fn mesh_info(&self, handle: MeshHandle) -> Option<MeshInfo> {
        self.meshes.get(handle).map(|mesh| MeshInfo {
            index_count: mesh.index_count,
            triangle_count: mesh.triangle_count,
            bounds: mesh.bounds,
        })
    }

    /// Switches shader loading to `shader_dir` so edits to WGSL take effect
    /// without a restart, and rebuilds the pipeline from it immediately.
    ///
    /// A no-op in release builds, where [`ShaderLibrary`] ignores the directory
    /// and the sources are embedded (ADR-0011). An error leaves the current
    /// pipeline in place.
    pub fn enable_shader_hot_reload(&mut self, shader_dir: PathBuf) -> Result<(), RenderError> {
        if !cfg!(debug_assertions) {
            return Ok(());
        }
        let previous = std::mem::replace(&mut self.shaders, ShaderLibrary::new(Some(shader_dir)));
        let mut staged = Vec::new();
        for &(name, family) in shaders::PIPELINES {
            match self.build_shader_family(family) {
                Ok(pipelines) => staged.push(pipelines),
                Err(error) => {
                    self.shaders = previous;
                    self.shader_failures.insert(name, error.to_string());
                    self.refresh_shader_error();
                    return Err(error);
                }
            }
        }
        for pipelines in staged {
            self.install_shader_family(pipelines);
        }
        self.shader_failures.clear();
        self.shader_error = None;
        Ok(())
    }

    /// The last shader hot-reload failure, if the current pipeline is stale.
    ///
    /// Returned rather than logged so the engine layer owns diagnostics output.
    pub fn shader_error(&self) -> Option<&str> {
        self.shader_error.as_deref()
    }

    /// Failures keyed by the shader whose previous pipelines remain installed.
    pub fn shader_errors(&self) -> &BTreeMap<&'static str, String> {
        &self.shader_failures
    }

    /// Draws skipped in the last frame because their mesh or material handle
    /// resolved to nothing.
    pub fn skipped_draws(&self) -> u32 {
        self.skipped_draws
    }

    fn upload_buffer(&self, label: &str, contents: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: contents.len() as u64,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, contents);
        buffer
    }

    fn build_lit_pipeline(&mut self) -> Result<wgpu::RenderPipeline, RenderError> {
        let source = self
            .shaders
            .source(shaders::LIT)
            .map_err(RenderError::Shader)?
            .into_owned();
        pipeline::create_pipeline(&self.device, &self.pipeline_layout, &source, passes::HDR_FORMAT).map_err(
            |message| RenderError::ShaderCompile {
                shader: shaders::LIT.to_owned(),
                message,
            },
        )
    }

    fn build_debug_pipeline(&mut self) -> Result<wgpu::RenderPipeline, RenderError> {
        let source = self
            .shaders
            .source(shaders::DEBUG_LINE)
            .map_err(RenderError::Shader)?
            .into_owned();
        debug_lines::create_pipeline(
            &self.device,
            &self.debug_pipeline_layout,
            &source,
            self.color_format,
        )
        .map_err(|message| RenderError::ShaderCompile {
            shader: shaders::DEBUG_LINE.to_owned(),
            message,
        })
    }

    /// Rebuilds a pipeline when its shader has been edited on disk.
    ///
    /// A failed rebuild keeps the pipeline that already works and records the
    /// message: a typo saved mid-session should cost the next redraw of the
    /// shader, not the session. Each shader rebuilds only its own pipeline, so
    /// an edit to one cannot disturb the other.
    fn reload_shaders_if_changed(&mut self) {
        let changed = self.shaders.changed();
        for (name, family) in shaders::affected_pipelines(&changed) {
            match self.build_shader_family(family) {
                Ok(pipelines) => {
                    self.install_shader_family(pipelines);
                    self.shader_failures.remove(name);
                }
                Err(error) => {
                    self.shader_failures.insert(name, error.to_string());
                }
            }
        }
        self.refresh_shader_error();
    }

    fn build_shader_family(&mut self, family: shaders::PipelineFamily) -> Result<BuiltShader, RenderError> {
        use shaders::PipelineFamily;
        Ok(match family {
            PipelineFamily::Lit => BuiltShader::Lit(self.build_lit_pipeline()?),
            PipelineFamily::Debug => BuiltShader::Debug(self.build_debug_pipeline()?),
            PipelineFamily::Tonemap => BuiltShader::Tonemap(build_tonemap_pipeline(
                &self.device,
                &mut self.shaders,
                &self.tonemap_pipeline_layout,
                self.color_format,
            )?),
            PipelineFamily::Pbr => BuiltShader::Pbr(build_pbr_pipelines(
                &self.device,
                &mut self.shaders,
                &self.pbr_pipeline_layout,
            )?),
            PipelineFamily::Bloom => {
                let source = self
                    .shaders
                    .expanded(shaders::BLOOM)
                    .map_err(RenderError::Shader)?;
                BuiltShader::Bloom(
                    post::bloom::BloomResources::new(&self.device, &source.text).map_err(|message| {
                        RenderError::ShaderCompile {
                            shader: shaders::BLOOM.into(),
                            message: source.annotate(&message),
                        }
                    })?,
                )
            }
            PipelineFamily::Exposure => {
                let source = self
                    .shaders
                    .expanded(shaders::EXPOSURE_HISTOGRAM)
                    .map_err(RenderError::Shader)?;
                BuiltShader::Exposure(
                    post::exposure::ExposureResources::new(&self.device, &source.text).map_err(
                        |message| RenderError::ShaderCompile {
                            shader: shaders::EXPOSURE_HISTOGRAM.into(),
                            message: source.annotate(&message),
                        },
                    )?,
                )
            }
            PipelineFamily::Mips => {
                let source = self
                    .shaders
                    .expanded(shaders::MIP_DOWNSAMPLE)
                    .map_err(RenderError::Shader)?;
                BuiltShader::Mips(texture::MipGenerator::new(&self.device, &source.text).map_err(
                    |message| RenderError::ShaderCompile {
                        shader: shaders::MIP_DOWNSAMPLE.into(),
                        message: source.annotate(&message),
                    },
                )?)
            }
        })
    }

    fn install_shader_family(&mut self, built: BuiltShader) {
        match built {
            BuiltShader::Lit(pipeline) => self.pipeline = pipeline,
            BuiltShader::Debug(pipeline) => self.debug_pipeline = pipeline,
            BuiltShader::Tonemap(pipeline) => self.tonemap_pipeline = pipeline,
            BuiltShader::Pbr(pipelines) => self.pbr_pipelines = pipelines,
            BuiltShader::Bloom(resources) => self.bloom = resources,
            BuiltShader::Exposure(resources) => {
                // Preserve adapted luminance and the uniform the tonemap reads.
                self.exposure.layout = resources.layout;
                self.exposure.build = resources.build;
                self.exposure.average = resources.average;
            }
            BuiltShader::Mips(generator) => self.mips = generator,
        }
    }

    fn refresh_shader_error(&mut self) {
        self.shader_error = (!self.shader_failures.is_empty()).then(|| {
            self.shader_failures
                .values()
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        });
    }

    /// Grows the per-draw uniform buffer to hold at least `needed` draws.
    ///
    /// The bind group is rebuilt with the buffer, since it points at it.
    fn ensure_draw_capacity(&mut self, needed: u32) {
        let capacity = uniforms::grown_capacity(self.draw_capacity, needed);
        if capacity == self.draw_capacity {
            return;
        }
        let (buffer, bind_group) =
            pipeline::create_draw_resources(&self.device, &self.draw_layout, self.draw_stride, capacity);
        self.draw_buffer = buffer;
        self.draw_bind_group = bind_group;
        self.draw_capacity = capacity;
    }

    /// Grows the debug vertex buffer to hold at least `needed` vertices.
    ///
    /// Same one-way doubling as the per-draw uniform buffer, but with no bind
    /// group to rebuild: a vertex buffer is bound directly by the render pass.
    fn ensure_debug_capacity(&mut self, needed: u32) {
        let capacity = uniforms::grown_capacity(self.debug_capacity, needed);
        if capacity == self.debug_capacity {
            return;
        }
        self.debug_vertex_buffer = debug_lines::create_vertex_buffer(&self.device, capacity);
        self.debug_capacity = capacity;
    }

    /// The texture this frame draws into: the swapchain's next image, or the
    /// offscreen texture.
    fn acquire(&self) -> Result<Acquired, RenderError> {
        let surface = match &self.output {
            Output::Window { surface, .. } => surface,
            Output::Offscreen { texture } => {
                return Ok(match texture {
                    Some(texture) => Acquired::Ready {
                        target: Target::Offscreen(texture.clone()),
                        suboptimal: false,
                    },
                    None => Acquired::Skipped(SkipReason::ZeroSizedSurface),
                });
            }
        };
        Ok(match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => Acquired::Ready {
                target: Target::Surface(texture),
                suboptimal: false,
            },
            // Presentable, but the swapchain no longer matches the surface:
            // present this frame, then reconfigure for the next one.
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => Acquired::Ready {
                target: Target::Surface(texture),
                suboptimal: true,
            },
            wgpu::CurrentSurfaceTexture::Timeout => Acquired::Skipped(SkipReason::SurfaceTimeout),
            wgpu::CurrentSurfaceTexture::Occluded => Acquired::Skipped(SkipReason::SurfaceOccluded),
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.configure_surface();
                Acquired::Skipped(SkipReason::SurfaceReconfigured)
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(RenderError::Surface(
                    "validation error while acquiring the surface texture".to_owned(),
                ));
            }
        })
    }

    /// GPU time per graph pass from the latest completed readback, which lags
    /// the CPU by a few frames. `None` without timestamp support or before the
    /// first readback.
    pub fn gpu_timings(&self) -> Option<&[PassTiming]> {
        self.gpu_timer.as_ref().and_then(gpu_timing::GpuTimer::latest)
    }

    /// Pass order, resource dependencies and lifetimes from the last graph.
    pub fn graph_dump(&self) -> &str {
        &self.graph_dump
    }

    /// Renders one submitted frame.
    ///
    /// This is the entry point the engine calls. The frame arrives already
    /// culled and ordered; the backend writes the uniforms and records the
    /// draws in exactly the order given. Where timestamps are supported, the
    /// frame's per-pass GPU times are queued for a later, non-blocking read.
    pub fn render(&mut self, frame: &RenderFrame) -> Result<FrameOutcome, RenderError> {
        if self.size.0 == 0 || self.size.1 == 0 {
            return Ok(FrameOutcome::Skipped(SkipReason::ZeroSizedSurface));
        }
        let started = Instant::now();
        self.reload_shaders_if_changed();

        let plan = registry::plan_draws(&frame.draws, &self.meshes, &self.materials);
        self.skipped_draws = plan.skipped;
        let draw_count = u32::try_from(plan.draws.len()).unwrap_or(u32::MAX);
        self.ensure_draw_capacity(draw_count);

        let stride = self.draw_stride as usize;
        if !plan.draws.is_empty() {
            // Disjoint field borrows: the staging buffer is filled from the
            // material registry without either going through `self`.
            uniforms::pack_draw_uniforms(
                &mut self.draw_staging,
                stride,
                &frame.draws,
                &plan.draws,
                &self.materials,
            );
        }

        // `None` means this frame draws no debug lines, and the pass is skipped
        // entirely: no growth, no upload, no pipeline switch. Debug drawing that
        // is switched off has to cost nothing.
        let debug_vertices = debug_lines::draw_vertex_count(&frame.debug_lines);
        if let Some(count) = debug_vertices {
            self.ensure_debug_capacity(count);
            // Packed before the surface is acquired, like the draw uniforms:
            // allocation belongs outside the acquire/present window.
            let vertices = &frame.debug_lines[..count as usize];
            debug_lines::pack_vertices(&mut self.debug_staging, vertices);
        }

        let (target, suboptimal) = match self.acquire()? {
            Acquired::Ready { target, suboptimal } => (target, suboptimal),
            Acquired::Skipped(reason) => return Ok(FrameOutcome::Skipped(reason)),
        };

        self.queue
            .write_buffer(&self.globals_buffer, 0, &uniforms::globals_bytes(frame));
        self.queue
            .write_buffer(&self.post_buffer, 0, &tonemap::post_bytes(&frame.post));
        let now = Instant::now();
        let frame_seconds = self
            .last_frame
            .map_or(0.0, |last| now.duration_since(last).as_secs_f32());
        self.last_frame = Some(now);
        if post::bloom::enabled(&frame.post.bloom) {
            self.queue.write_buffer(
                &self.bloom.params,
                0,
                &post::bloom::params_bytes(&frame.post.bloom),
            );
        }
        if let Some(auto) = frame.post.auto_exposure {
            self.queue.write_buffer(
                &self.exposure.params,
                0,
                &post::exposure::params_bytes(&auto, frame_seconds),
            );
        }
        if !plan.draws.is_empty() {
            self.queue.write_buffer(&self.draw_buffer, 0, &self.draw_staging);
        }
        if debug_vertices.is_some() {
            self.queue
                .write_buffer(&self.debug_vertex_buffer, 0, &self.debug_staging);
        }

        // The pool is moved out while the graph borrows `self` for its passes,
        // and put back whatever happens.
        let mut transients = std::mem::take(&mut self.transients);
        let mut timer = self.gpu_timer.take();
        if let Some(timer) = &mut timer {
            timer.begin_frame(&self.device);
        }
        let commands = self.record_frame(
            &mut transients,
            target.view(),
            frame,
            &plan,
            debug_vertices,
            &mut timer,
        );
        self.transients = transients;
        if commands.is_err()
            && let Some(timer) = &mut timer
        {
            timer.cancel();
        }
        self.gpu_timer = timer;
        let (commands, dump) = commands?;
        self.queue.submit(std::iter::once(commands));
        self.graph_dump = dump;
        if let Some(timer) = &mut self.gpu_timer {
            timer.submitted();
        }
        self.finish(target, suboptimal);

        Ok(FrameOutcome::Presented(FrameStats {
            gpu_passes: self.gpu_timings().map(<[PassTiming]>::to_vec),
            gpu_frame: self
                .gpu_timer
                .as_ref()
                .and_then(gpu_timing::GpuTimer::latest_frame),
            draw_calls: draw_count,
            triangles: plan.triangles,
            cpu_time: started.elapsed(),
            // Vertices are consumed in pairs, so segments are half of them.
            debug_lines: debug_vertices.unwrap_or(0) / 2,
        }))
    }

    /// Clears the surface to `color` and presents it.
    ///
    /// Kept alongside [`render`](Self::render) for callers that have no scene to
    /// draw yet, and for the startup path before any content is loaded.
    ///
    /// `color` is a linear *display* colour written straight to the
    /// swapchain, with no exposure or tonemapping, unlike
    /// [`RenderFrame::clear_color`], which is scene light.
    pub fn render_clear(&mut self, color: Color) -> Result<FrameOutcome, RenderError> {
        if self.size.0 == 0 || self.size.1 == 0 {
            return Ok(FrameOutcome::Skipped(SkipReason::ZeroSizedSurface));
        }
        let started = Instant::now();
        let (target, suboptimal) = match self.acquire()? {
            Acquired::Ready { target, suboptimal } => (target, suboptimal),
            Acquired::Skipped(reason) => return Ok(FrameOutcome::Skipped(reason)),
        };

        let view = target.view();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame-encoder"),
            });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: color.r,
                            g: color.g,
                            b: color.b,
                            a: color.a,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.finish(target, suboptimal);

        Ok(FrameOutcome::Presented(FrameStats {
            gpu_passes: None,
            gpu_frame: None,
            draw_calls: 0,
            triangles: 0,
            cpu_time: started.elapsed(),
            debug_lines: 0,
        }))
    }

    /// Presents a swapchain image and reconfigures a stale swapchain. An
    /// offscreen frame needs neither: its texture simply holds the result.
    fn finish(&self, target: Target, suboptimal: bool) {
        if let Target::Surface(texture) = target {
            self.queue.present(texture);
        }
        if suboptimal {
            self.configure_surface();
        }
    }

    /// Renders `frame` and reads the result back.
    ///
    /// Only a renderer built with [`headless`](Self::headless) has a frame to
    /// read: a swapchain image is gone once presented. Blocks until the GPU has
    /// finished, so this is for tests and tools, never the frame loop.
    pub fn render_to_image(&mut self, frame: &RenderFrame) -> Result<Image, RenderError> {
        if !matches!(self.output, Output::Offscreen { .. }) {
            return Err(RenderError::Readback(
                "only a headless renderer can read its frames back".to_owned(),
            ));
        }
        match self.render(frame)? {
            FrameOutcome::Presented(_) => self.read_back(),
            FrameOutcome::Skipped(reason) => Err(RenderError::Readback(format!(
                "the frame was skipped: {reason:?}"
            ))),
        }
    }

    /// Copies the offscreen texture into a buffer and maps it.
    fn read_back(&self) -> Result<Image, RenderError> {
        let Output::Offscreen {
            texture: Some(texture),
        } = &self.output
        else {
            return Err(RenderError::Readback("there is no offscreen texture".to_owned()));
        };
        let (width, height) = self.size;
        self.read_texture(texture, 0, width, height)
    }

    /// Reads one mip level of an RGBA8 texture back to the CPU, blocking until
    /// the GPU is done. The bytes are as stored: sRGB-encoded for an sRGB
    /// texture.
    fn read_texture(
        &self,
        texture: &wgpu::Texture,
        mip_level: u32,
        width: u32,
        height: u32,
    ) -> Result<Image, RenderError> {
        let bytes_per_row = image::padded_bytes_per_row(width);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback-encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let (sender, receiver) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            // The receiver outlives the poll below, so a failed send cannot
            // happen; if it somehow did, `recv` reports it.
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| RenderError::Readback(error.to_string()))?;
        receiver
            .recv()
            .map_err(|error| RenderError::Readback(error.to_string()))?
            .map_err(|error| RenderError::Readback(error.to_string()))?;

        let rgba8 = {
            let mapped = buffer
                .get_mapped_range(..)
                .map_err(|error| RenderError::Readback(error.to_string()))?;
            image::unpad_rows(&mapped, width, height)
        };
        buffer.unmap();
        // A short buffer would otherwise become a short image, which a golden
        // comparison of the common prefix could pass.
        let expected = width as usize * height as usize * image::BYTES_PER_PIXEL;
        if rgba8.len() != expected {
            return Err(RenderError::Readback(format!(
                "read {} bytes where {width}x{height} needs {expected}",
                rgba8.len()
            )));
        }
        Ok(Image { width, height, rgba8 })
    }
}

/// The outcome of asking the output for a texture to draw into.
#[derive(Debug)]
enum Acquired {
    Ready {
        target: Target,
        /// The swapchain presented but no longer matches the surface, so it is
        /// reconfigured after this frame rather than before it.
        suboptimal: bool,
    },
    Skipped(SkipReason),
}

/// The texture one frame draws into.
#[derive(Debug)]
enum Target {
    Surface(wgpu::SurfaceTexture),
    Offscreen(wgpu::Texture),
}

impl Target {
    fn view(&self) -> wgpu::TextureView {
        let texture = match self {
            Self::Surface(surface) => &surface.texture,
            Self::Offscreen(texture) => texture,
        };
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }
}

/// Compiles every PBR permutation. Any failure fails the whole set, so a
/// material can never find its permutation missing.
fn build_pbr_pipelines(
    device: &wgpu::Device,
    shaders: &mut ShaderLibrary,
    layout: &wgpu::PipelineLayout,
) -> Result<BTreeMap<permutation::PermutationKey, wgpu::RenderPipeline>, RenderError> {
    let expanded = shaders.expanded(shaders::PBR_LIT).map_err(RenderError::Shader)?;
    permutation::PermutationKey::all()
        .map(|key| {
            pbr::create_pipeline(device, layout, &expanded.text, key)
                .map(|pipeline| (key, pipeline))
                .map_err(|message| RenderError::ShaderCompile {
                    shader: format!("{} {key:?}", shaders::PBR_LIT),
                    message: expanded.annotate(&message),
                })
        })
        .collect()
}

/// Builds the tonemap pipeline for a `format` swapchain. A compile error names
/// the file and line it came from, which may be in an included snippet.
fn build_tonemap_pipeline(
    device: &wgpu::Device,
    shaders: &mut ShaderLibrary,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
) -> Result<wgpu::RenderPipeline, RenderError> {
    let expanded = shaders.expanded(shaders::TONEMAP).map_err(RenderError::Shader)?;
    tonemap::create_pipeline(device, layout, &expanded.text, format).map_err(|message| {
        RenderError::ShaderCompile {
            shader: shaders::TONEMAP.to_owned(),
            message: expanded.annotate(&message),
        }
    })
}

fn create_instance(config: RendererConfig) -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: match config.backend {
            BackendPreference::Auto => wgpu::Backends::PRIMARY,
            BackendPreference::Vulkan => wgpu::Backends::VULKAN,
        },
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    })
}

fn request_adapter(
    instance: &wgpu::Instance,
    config: RendererConfig,
    surface: Option<&wgpu::Surface<'_>>,
) -> Result<wgpu::Adapter, RenderError> {
    block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: match config.gpu_preference {
            GpuPreference::LowPower => wgpu::PowerPreference::LowPower,
            GpuPreference::HighPerformance => wgpu::PowerPreference::HighPerformance,
        },
        compatible_surface: surface,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .map_err(|error| RenderError::NoAdapter(error.to_string()))
}

fn request_device(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue), RenderError> {
    // Downlevel limits keep the engine honest about the low-spec target;
    // they are raised to what the adapter reports for resolution-derived
    // limits only. Optional features are asked for only when the adapter has
    // them, so requesting one can never fail device creation (ADR-0023).
    let required_limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
    block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("engine-device"),
        required_features: adapter.features() & OPTIONAL_FEATURES,
        required_limits,
        ..Default::default()
    }))
    .map_err(|error| RenderError::RequestDevice(error.to_string()))
}

/// What was selected, with the features the device was actually created
/// with, read back rather than recomputed so the report cannot drift.
fn adapter_info(adapter: &wgpu::Adapter, device: &wgpu::Device) -> AdapterInfo {
    let info = adapter.get_info();
    AdapterInfo {
        name: info.name.clone(),
        backend: info.backend.to_string(),
        device_kind: match info.device_type {
            wgpu::DeviceType::IntegratedGpu => DeviceKind::IntegratedGpu,
            wgpu::DeviceType::DiscreteGpu => DeviceKind::DiscreteGpu,
            wgpu::DeviceType::VirtualGpu => DeviceKind::VirtualGpu,
            wgpu::DeviceType::Cpu => DeviceKind::Cpu,
            wgpu::DeviceType::Other => DeviceKind::Other,
        },
        driver: format!("{} {}", info.driver, info.driver_info).trim().to_owned(),
        features: GpuFeatures::from_wgpu(device.features()),
    }
}

/// `format` is the renderer's colour format, so the attachment always matches
/// the pipelines built for it.
fn create_offscreen_texture(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen-color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn select_present_mode(preference: PresentMode, supported: &[wgpu::PresentMode]) -> wgpu::PresentMode {
    match preference {
        // Fifo is guaranteed to be supported by every wgpu backend.
        PresentMode::Vsync => wgpu::PresentMode::Fifo,
        PresentMode::LowLatency => [wgpu::PresentMode::Mailbox, wgpu::PresentMode::Immediate]
            .into_iter()
            .find(|mode| supported.contains(mode))
            .unwrap_or(wgpu::PresentMode::Fifo),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(debug_assertions)]
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn hot_reload_keeps_failures_separate_and_recovers_every_family() {
        let mut renderer = Renderer::headless(8, 8, RendererConfig::default()).expect("adapter");
        let dir = std::env::temp_dir().join(format!(
            "renderer-reload-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let assets = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/shaders");
        for file in std::fs::read_dir(assets).unwrap() {
            let file = file.unwrap();
            if file
                .path()
                .extension()
                .is_some_and(|extension| extension == "wgsl")
            {
                std::fs::copy(file.path(), dir.join(file.file_name())).unwrap();
            }
        }
        renderer.enable_shader_hot_reload(dir.clone()).unwrap();
        let mut embedded = ShaderLibrary::new(None);
        // Each family fails without disturbing other errors, then repairs independently.
        for &(name, _) in shaders::PIPELINES {
            std::fs::write(dir.join(name), "this is invalid WGSL").unwrap();
        }
        renderer.reload_shaders_if_changed();
        assert_eq!(renderer.shader_errors().len(), shaders::PIPELINES.len());
        for (index, &(name, _)) in shaders::PIPELINES.iter().enumerate() {
            std::fs::write(dir.join(name), embedded.source(name).unwrap().as_bytes()).unwrap();
            renderer.reload_shaders_if_changed();
            assert!(!renderer.shader_errors().contains_key(name));
            assert_eq!(
                renderer.shader_errors().len(),
                shaders::PIPELINES.len() - index - 1
            );
        }
        assert!(renderer.shader_error().is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn vsync_always_picks_fifo() {
        let supported = [wgpu::PresentMode::Fifo, wgpu::PresentMode::Mailbox];
        assert_eq!(
            select_present_mode(PresentMode::Vsync, &supported),
            wgpu::PresentMode::Fifo
        );
    }

    #[test]
    fn low_latency_prefers_mailbox_then_immediate_then_fifo() {
        assert_eq!(
            select_present_mode(
                PresentMode::LowLatency,
                &[wgpu::PresentMode::Fifo, wgpu::PresentMode::Mailbox]
            ),
            wgpu::PresentMode::Mailbox
        );
        assert_eq!(
            select_present_mode(
                PresentMode::LowLatency,
                &[wgpu::PresentMode::Fifo, wgpu::PresentMode::Immediate]
            ),
            wgpu::PresentMode::Immediate
        );
        assert_eq!(
            select_present_mode(PresentMode::LowLatency, &[wgpu::PresentMode::Fifo]),
            wgpu::PresentMode::Fifo
        );
    }
}
