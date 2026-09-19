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

mod block_on;
mod debug_lines;
mod frame;
mod material;
mod mesh;
mod mesh_data;
mod pipeline;
mod registry;
mod shaders;
mod uniforms;

pub use debug_lines::DebugVertex;
pub use frame::{
    AmbientLight, DirectionalLight, DrawCommand, MaterialDesc, MaterialHandle, MeshHandle, RenderFrame,
};
pub use mesh::MeshInfo;
pub use mesh_data::{Aabb, MeshData, MeshDataError, VERTEX_FLOATS, WHITE};
pub use shaders::{DEBUG_LINE, LIT, ShaderError, ShaderLibrary};

use std::fmt;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use material::GpuMaterial;
use mesh::GpuMesh;
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
    /// WGSL source could not be resolved.
    Shader(ShaderError),
    /// WGSL compiled into a pipeline that the GPU rejected. Carries the
    /// validation message, which names the line for a syntax error.
    ShaderCompile {
        shader: String,
        message: String,
    },
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
            Self::Shader(error) => write!(f, "{error}"),
            Self::ShaderCompile { shader, message } => {
                write!(f, "cannot build a pipeline for {shader}: {message}")
            }
        }
    }
}

impl std::error::Error for RenderError {}

/// Per-frame counters for the performance overlay.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameStats {
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameOutcome {
    Presented(FrameStats),
    Skipped(SkipReason),
}

/// Draws the per-draw uniform buffer starts out able to hold.
///
/// Sized so a first-pass blocked-out scene never reallocates on its opening
/// frame; the buffer grows from here and is never shrunk.
const INITIAL_DRAW_CAPACITY: u32 = 64;

/// Owns the GPU device, queue, and swapchain for one window.
#[derive(Debug)]
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    adapter_info: AdapterInfo,

    shaders: ShaderLibrary,
    pipeline_layout: wgpu::PipelineLayout,
    draw_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    /// The last hot-reload failure, kept so the engine can show it on the debug
    /// overlay. The renderer does not log: that is the engine layer's job.
    shader_error: Option<String>,

    /// `None` only while the surface is zero-sized.
    depth_view: Option<wgpu::TextureView>,
    depth_size: (u32, u32),

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
}

impl Renderer {
    /// Creates a renderer for `window`, which must outlive the renderer.
    pub fn new<W>(window: W, width: u32, height: u32, config: RendererConfig) -> Result<Self, RenderError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: match config.backend {
                BackendPreference::Auto => wgpu::Backends::PRIMARY,
                BackendPreference::Vulkan => wgpu::Backends::VULKAN,
            },
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance
            .create_surface(window)
            .map_err(|error| RenderError::CreateSurface(error.to_string()))?;

        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: match config.gpu_preference {
                GpuPreference::LowPower => wgpu::PowerPreference::LowPower,
                GpuPreference::HighPerformance => wgpu::PowerPreference::HighPerformance,
            },
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .map_err(|error| RenderError::NoAdapter(error.to_string()))?;

        let info = adapter.get_info();
        let adapter_info = AdapterInfo {
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
        };

        // Downlevel limits keep the engine honest about the low-spec target;
        // they are raised to what the adapter reports for resolution-derived
        // limits only.
        let required_limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("engine-device"),
            required_features: wgpu::Features::empty(),
            required_limits,
            ..Default::default()
        }))
        .map_err(|error| RenderError::RequestDevice(error.to_string()))?;

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
            pipeline::create_pipeline(&device, &pipeline_layout, &source, surface_config.format).map_err(
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
            debug_lines::create_pipeline(&device, &debug_pipeline_layout, &source, surface_config.format)
                .map_err(|message| RenderError::ShaderCompile {
                    shader: shaders::DEBUG_LINE.to_owned(),
                    message,
                })?
        };
        let debug_vertex_buffer =
            debug_lines::create_vertex_buffer(&device, debug_lines::INITIAL_VERTEX_CAPACITY);

        // Read from the device rather than assumed: 256 is the common value for
        // `min_uniform_buffer_offset_alignment`, but it is a limit precisely
        // because hardware disagrees, and a wrong stride is a validation error
        // on every single draw.
        let draw_stride = uniforms::draw_stride(device.limits().min_uniform_buffer_offset_alignment);
        let (globals_buffer, globals_bind_group) =
            pipeline::create_globals_resources(&device, &globals_layout);
        let (draw_buffer, draw_bind_group) =
            pipeline::create_draw_resources(&device, &draw_layout, draw_stride, INITIAL_DRAW_CAPACITY);

        let mut renderer = Self {
            surface,
            device,
            queue,
            surface_config,
            adapter_info,
            shaders,
            pipeline_layout,
            draw_layout,
            pipeline,
            shader_error: None,
            depth_view: None,
            depth_size: (0, 0),
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
        };
        renderer.configure_surface();
        renderer.ensure_depth_view();
        Ok(renderer)
    }

    fn configure_surface(&self) {
        if self.surface_config.width > 0 && self.surface_config.height > 0 {
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    /// Resizes the swapchain. A zero-sized surface is recorded but not configured.
    pub fn resize(&mut self, width: u32, height: u32) {
        if (width, height) == (self.surface_config.width, self.surface_config.height) {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.configure_surface();
        // The depth buffer must match the colour attachment exactly, so it is
        // rebuilt here rather than on the next frame. A zero-sized surface
        // leaves it absent instead of attempting a zero-extent texture.
        self.ensure_depth_view();
    }

    /// Rebuilds the depth buffer if it no longer matches the surface.
    fn ensure_depth_view(&mut self) {
        let size = (self.surface_config.width, self.surface_config.height);
        if self.depth_size == size && self.depth_view.is_some() {
            return;
        }
        self.depth_view = pipeline::create_depth_view(&self.device, size.0, size.1);
        self.depth_size = size;
    }

    pub fn surface_size(&self) -> (u32, u32) {
        (self.surface_config.width, self.surface_config.height)
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

    /// Registers a material and returns a handle to it.
    pub fn create_material(&mut self, desc: &MaterialDesc) -> MaterialHandle {
        self.materials.insert(GpuMaterial::from_desc(desc))
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
        self.shaders = ShaderLibrary::new(Some(shader_dir));
        // Both pipelines are rebuilt before either is installed, so a directory
        // holding one good shader and one broken one leaves the renderer on the
        // pair that already works instead of a mismatched half.
        let pipeline = self.build_lit_pipeline()?;
        let debug_pipeline = self.build_debug_pipeline()?;
        self.pipeline = pipeline;
        self.debug_pipeline = debug_pipeline;
        self.shader_error = None;
        Ok(())
    }

    /// The last shader hot-reload failure, if the current pipeline is stale.
    ///
    /// Returned rather than logged so the engine layer owns diagnostics output.
    pub fn shader_error(&self) -> Option<&str> {
        self.shader_error.as_deref()
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
        pipeline::create_pipeline(
            &self.device,
            &self.pipeline_layout,
            &source,
            self.surface_config.format,
        )
        .map_err(|message| RenderError::ShaderCompile {
            shader: shaders::LIT.to_owned(),
            message,
        })
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
            self.surface_config.format,
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
        if pipeline::lit_shader_changed(&changed) {
            match self.build_lit_pipeline() {
                Ok(pipeline) => {
                    self.pipeline = pipeline;
                    self.shader_error = None;
                }
                Err(error) => self.shader_error = Some(error.to_string()),
            }
        }
        if debug_lines::debug_shader_changed(&changed) {
            match self.build_debug_pipeline() {
                Ok(pipeline) => {
                    self.debug_pipeline = pipeline;
                    self.shader_error = None;
                }
                Err(error) => self.shader_error = Some(error.to_string()),
            }
        }
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

    /// Asks the swapchain for the texture this frame draws into.
    fn acquire_surface(&self) -> Result<Acquired, RenderError> {
        Ok(match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => Acquired::Ready {
                texture,
                suboptimal: false,
            },
            // Presentable, but the swapchain no longer matches the surface:
            // present this frame, then reconfigure for the next one.
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => Acquired::Ready {
                texture,
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

    /// Renders one submitted frame.
    ///
    /// This is the entry point the engine calls. The frame arrives already
    /// culled and ordered; the backend writes the uniforms and records the
    /// draws in exactly the order given.
    pub fn render(&mut self, frame: &RenderFrame) -> Result<FrameOutcome, RenderError> {
        if self.surface_config.width == 0 || self.surface_config.height == 0 {
            return Ok(FrameOutcome::Skipped(SkipReason::ZeroSizedSurface));
        }
        let started = Instant::now();
        self.reload_shaders_if_changed();
        self.ensure_depth_view();

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

        let (surface_texture, suboptimal) = match self.acquire_surface()? {
            Acquired::Ready { texture, suboptimal } => (texture, suboptimal),
            Acquired::Skipped(reason) => return Ok(FrameOutcome::Skipped(reason)),
        };

        // The pipeline declares a depth attachment, so a frame without one
        // cannot be recorded. Only a zero-sized surface produces that, and it
        // returned above; skipping keeps the invariant without a panic.
        let Some(depth_view) = self.depth_view.as_ref() else {
            return Ok(FrameOutcome::Skipped(SkipReason::ZeroSizedSurface));
        };

        self.queue.write_buffer(
            &self.globals_buffer,
            0,
            &uniforms::globals_bytes(&frame.view_proj, &frame.light, &frame.ambient),
        );
        if !plan.draws.is_empty() {
            self.queue.write_buffer(&self.draw_buffer, 0, &self.draw_staging);
        }
        if debug_vertices.is_some() {
            self.queue
                .write_buffer(&self.debug_vertex_buffer, 0, &self.debug_staging);
        }

        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame-encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("lit-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: frame.clear_color.r,
                            g: frame.clear_color.g,
                            b: frame.clear_color.b,
                            a: frame.clear_color.a,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        // 1.0 is the far plane for the `0..1` depth range
                        // (ADR-0008), and `Less` keeps the nearest fragment.
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            if !plan.draws.is_empty() {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.globals_bind_group, &[]);
                for planned in &plan.draws {
                    let Some(mesh) = self.meshes.get_by_index(planned.mesh) else {
                        continue;
                    };
                    let Ok(offset) = u32::try_from(planned.slot * stride) else {
                        continue;
                    };
                    pass.set_bind_group(1, &self.draw_bind_group, &[offset]);
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }

            // Debug lines last, inside the same pass and against the same depth
            // buffer. The pipeline tests depth, so the scene occludes the lines,
            // and does not write it, so the lines never occlude the scene. One
            // draw call covers every line in the frame.
            if let Some(count) = debug_vertices {
                pass.set_pipeline(&self.debug_pipeline);
                // Re-bound even though the lit pass bound the same buffer:
                // switching to a pipeline with a different layout invalidates
                // the bind groups already set.
                pass.set_bind_group(0, &self.globals_bind_group, &[]);
                pass.set_vertex_buffer(0, self.debug_vertex_buffer.slice(..));
                pass.draw(0..count, 0..1);
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(surface_texture);
        if suboptimal {
            self.configure_surface();
        }

        Ok(FrameOutcome::Presented(FrameStats {
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
    pub fn render_clear(&mut self, color: Color) -> Result<FrameOutcome, RenderError> {
        if self.surface_config.width == 0 || self.surface_config.height == 0 {
            return Ok(FrameOutcome::Skipped(SkipReason::ZeroSizedSurface));
        }
        let started = Instant::now();
        let (surface_texture, suboptimal) = match self.acquire_surface()? {
            Acquired::Ready { texture, suboptimal } => (texture, suboptimal),
            Acquired::Skipped(reason) => return Ok(FrameOutcome::Skipped(reason)),
        };

        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
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
        self.queue.present(surface_texture);
        if suboptimal {
            self.configure_surface();
        }

        Ok(FrameOutcome::Presented(FrameStats {
            draw_calls: 0,
            triangles: 0,
            cpu_time: started.elapsed(),
            debug_lines: 0,
        }))
    }
}

/// The outcome of asking the swapchain for a texture to draw into.
#[derive(Debug)]
enum Acquired {
    Ready {
        texture: wgpu::SurfaceTexture,
        /// The swapchain presented but no longer matches the surface, so it is
        /// reconfigured after this frame rather than before it.
        suboptimal: bool,
    },
    Skipped(SkipReason),
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
