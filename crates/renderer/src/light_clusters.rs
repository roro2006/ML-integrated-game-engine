//! Clustered point and spot lights (C5): the lights' upload, the per-frame
//! cluster cull, and the overflow count the engine sees in `FrameStats`.
//!
//! The view frustum is cut into [`GRID`] clusters: tiles across the screen,
//! and slices in view depth spaced exponentially between the camera's near
//! and far planes, so near clusters stay small. A compute pass bounds every
//! cluster with a world-space box and lists the lights touching it, up to a
//! per-cluster cap; `pbr_lit.wgsl` then shades a fragment with its cluster's
//! list only. Flat materials ignore local lights (M18 Gate 1, decision 3).
//!
//! The functions marked as mirrors here are the CPU copies of
//! `clustered_lights.wgsl` and `light_cluster_build.wgsl` that the tests hold
//! the shaders to. Change both together.
//!
//! Storage buffers per stage (ADR-0023, amended for C5): the cull entry point
//! binds four (lights, bounds, lists, overflow), exactly the downlevel floor;
//! the PBR fragment stage binds two (lights, lists). Neither needs a raised
//! limit, which a test checks against the shaders themselves.

use glam::Vec3;

use crate::block_on::block_on;
use crate::frame::{LocalLight, RenderFrame};
use crate::gpu_timing::{LATENCY_FRAMES, ReadbackRing};
use crate::shaders::LIGHT_CLUSTER_BUILD;
use crate::shadow::ViewDepths;

/// Clusters across, down and in depth. The shaders read it from
/// `ClusterParams`.
pub(crate) const GRID: [u32; 3] = [16, 9, 24];
const CLUSTER_COUNT: u32 = GRID[0] * GRID[1] * GRID[2];
/// Bytes of one `GpuLight`.
const GPU_LIGHT_SIZE: usize = 64;
/// Bytes of `ClusterParams`.
pub(crate) const CLUSTER_PARAMS_SIZE: usize = 144;
/// Lights uploaded per frame; any beyond are ignored. Bounds the cull's
/// per-cluster loop, which visits every light.
pub(crate) const MAX_LOCAL_LIGHTS: usize = 16_384;
/// The largest per-cluster cap honoured; larger requests are clamped.
pub(crate) const MAX_CLUSTER_CAP: u32 = 256;
const WORKGROUP: u32 = 64;
/// A spot's inner cone is kept this far inside its outer one in cosine, so
/// the shader's `smoothstep` never sees equal edges.
const CONE_EDGE: f32 = 1e-4;

/// Where depth slice `slice` begins, in view depth: exponential from `near`
/// (slice 0) to `far` (slice `GRID[2]`). Mirror of `slice_depth`.
#[cfg(test)]
pub(crate) fn slice_depth(slice: u32, near: f32, far: f32) -> f32 {
    near * (far / near).powf(slice as f32 / GRID[2] as f32)
}

/// The depth slice holding view depth `depth`, clamped to the grid. Mirror
/// of the slice term in `cluster_of`.
#[cfg(test)]
pub(crate) fn slice_of(depth: f32, near: f32, far: f32) -> u32 {
    let depth = depth.max(near);
    // A float-to-int `as` saturates, as WGSL's `u32()` does for in-range values.
    let slice = ((depth / near).ln() / (far / near).ln() * GRID[2] as f32) as u32;
    slice.min(GRID[2] - 1)
}

/// Whether a light's sphere reaches the box from `low` to `high`. Mirror of
/// the sphere test in `cull_main`.
#[cfg(test)]
pub(crate) fn sphere_touches_box(centre: Vec3, radius: f32, low: Vec3, high: Vec3) -> bool {
    let outside = (low - centre).max(Vec3::ZERO) + (centre - high).max(Vec3::ZERO);
    outside.length_squared() <= radius * radius
}

/// A spot's cone, as the cull tests it.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CullCone {
    pub(crate) apex: Vec3,
    pub(crate) axis: Vec3,
    pub(crate) range: f32,
    pub(crate) cos_outer: f32,
    pub(crate) sin_outer: f32,
}

/// Whether the cone certainly misses the sphere at `centre` (Wronski, "Cull
/// that cone", 2017). Valid for cones narrower than a hemisphere; wider ones
/// are never culled by angle. Mirror of `cone_misses_sphere`.
#[cfg(test)]
pub(crate) fn cone_misses_sphere(cone: &CullCone, centre: Vec3, radius: f32) -> bool {
    if cone.cos_outer <= 0.0 {
        return false;
    }
    let offset = centre - cone.apex;
    let along = offset.dot(cone.axis);
    let across = (offset.length_squared() - along * along).max(0.0).sqrt();
    let closest = cone.cos_outer * across - along * cone.sin_outer;
    closest > radius || along > radius + cone.range || along < -radius
}

/// A light as the GPU stores it, or `None` for one that can light nothing:
/// a non-finite value, no range, no intensity, or a spot with no direction.
fn gpu_light(light: &LocalLight) -> Option<[f32; 16]> {
    let finite = light.position.iter().chain(&light.color).all(|v| v.is_finite())
        && light.range.is_finite()
        && light.intensity.is_finite();
    if !finite || light.range <= 0.0 || light.intensity <= 0.0 {
        return None;
    }
    let [x, y, z] = light.position;
    let [r, g, b] = light.color;
    let mut words = [
        x,
        y,
        z,
        light.range,
        r,
        g,
        b,
        light.intensity,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    if let Some(spot) = light.spot {
        let axis = Vec3::from(spot.direction).try_normalize()?;
        if !(spot.inner_angle.is_finite() && spot.outer_angle.is_finite()) {
            return None;
        }
        let outer = spot.outer_angle.clamp(1e-3, std::f32::consts::PI);
        let inner = spot.inner_angle.clamp(0.0, outer);
        // Both edges stay at or below 1, so the axis gets full intensity.
        let cos_outer = outer.cos().min(1.0 - CONE_EDGE);
        words[8..12].copy_from_slice(&[axis.x, axis.y, axis.z, 1.0]);
        words[12..15].copy_from_slice(&[inner.cos().max(cos_outer + CONE_EDGE), cos_outer, outer.sin()]);
    }
    Some(words)
}

/// Packs the valid lights, at most [`MAX_LOCAL_LIGHTS`], into `out` as
/// `GpuLight`s and returns how many there are.
pub(crate) fn pack_lights(lights: &[LocalLight], out: &mut Vec<u8>) -> u32 {
    out.clear();
    let mut count = 0;
    for words in lights.iter().filter_map(gpu_light).take(MAX_LOCAL_LIGHTS) {
        for word in words {
            out.extend_from_slice(&word.to_le_bytes());
        }
        count += 1;
    }
    count
}

/// `ClusterParams` for one frame.
pub(crate) fn cluster_params_bytes(
    inverse_view_proj: &[[f32; 4]; 4],
    depths: &ViewDepths,
    lights: u32,
    cap: u32,
    size: (u32, u32),
) -> [u8; CLUSTER_PARAMS_SIZE] {
    let mut words = [0_u32; CLUSTER_PARAMS_SIZE / 4];
    for (column, values) in inverse_view_proj.iter().enumerate() {
        for (row, value) in values.iter().enumerate() {
            words[column * 4 + row] = value.to_bits();
        }
    }
    let float = |values: [f32; 4]| values.map(f32::to_bits);
    words[16..20].copy_from_slice(&float(depths.eye.extend(depths.near).to_array()));
    words[20..24].copy_from_slice(&float(depths.forward.extend(depths.far).to_array()));
    words[24..28].copy_from_slice(&[GRID[0], GRID[1], GRID[2], 0]);
    words[28..32].copy_from_slice(&[lights, cap, 0, 0]);
    words[32..36].copy_from_slice(&float([size.0 as f32, size.1 as f32, 0.0, 0.0]));
    let mut out = [0_u8; CLUSTER_PARAMS_SIZE];
    for (chunk, word) in out.chunks_exact_mut(4).zip(words) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    out
}

/// Words in the light lists: per cluster a count, then up to `cap` indices.
fn grid_words(cap: u32) -> u64 {
    u64::from(CLUSTER_COUNT) * u64::from(cap + 1)
}

fn storage_buffer(device: &wgpu::Device, label: &str, size: u64, extra: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | extra,
        mapped_at_creation: false,
    })
}

/// The cull's bind group over, in order, the cluster parameters, lights,
/// bounds, lists and overflow counter.
fn cull_group(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    buffers: [&wgpu::Buffer; 5],
) -> wgpu::BindGroup {
    const BINDINGS: [u32; 5] = [10, 11, 2, 3, 4];
    let entries: Vec<_> = BINDINGS
        .into_iter()
        .zip(buffers)
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("light-cull"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

type MapResult = std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>;

/// Overflow readbacks in flight; each completes a few frames after its own.
#[derive(Debug)]
struct OverflowReadback {
    buffers: Vec<wgpu::Buffer>,
    /// Each busy slot holds its frame number and, once submitted, its
    /// mapping's result channel.
    ring: ReadbackRing<(u64, Option<MapResult>)>,
    /// The slot this frame copies into, between recording and submission.
    active: Option<usize>,
    /// Frames prepared so far, to order readbacks that finish together.
    frame: u64,
    /// The newest reading and the frame it came from.
    latest: Option<(u64, u32)>,
}

impl OverflowReadback {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            buffers: (0..LATENCY_FRAMES)
                .map(|_| {
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("light-overflow-readback"),
                        size: 4,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                })
                .collect(),
            ring: ReadbackRing::new(LATENCY_FRAMES),
            active: None,
            frame: 0,
            latest: None,
        }
    }

    /// Reads every readback whose mapping has finished. Never blocks.
    fn collect(&mut self) {
        for slot in 0..self.buffers.len() {
            let Some((_, Some(ready))) = self.ring.get(slot) else {
                continue;
            };
            let result = match ready.try_recv() {
                Ok(result) => result.is_ok(),
                Err(std::sync::mpsc::TryRecvError::Empty) => continue,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => false,
            };
            let Some((frame, _)) = self.ring.complete(slot) else {
                continue;
            };
            let newer = self.latest.is_none_or(|(latest, _)| frame > latest);
            if result
                && newer
                && let Ok(bytes) = self.buffers[slot].slice(..).get_mapped_range()
            {
                let value = bytes
                    .get(..4)
                    .and_then(|word| word.try_into().ok())
                    .map(u32::from_le_bytes);
                self.latest = value.map(|value| (frame, value));
            }
            self.buffers[slot].unmap();
        }
    }
}

/// The light buffers, the cull pipelines and the overflow readback.
#[derive(Debug)]
pub(crate) struct LightClusters {
    pub(crate) params: wgpu::Buffer,
    pub(crate) lights: wgpu::Buffer,
    pub(crate) grid: wgpu::Buffer,
    light_capacity: u32,
    cap: u32,
    bounds: wgpu::Buffer,
    overflow: wgpu::Buffer,
    bounds_pipeline: wgpu::ComputePipeline,
    cull_pipeline: wgpu::ComputePipeline,
    bounds_group: wgpu::BindGroup,
    cull_group: wgpu::BindGroup,
    staging: Vec<u8>,
    /// Lights culled this frame; zero records no cull pass.
    count: u32,
    readback: OverflowReadback,
}

impl LightClusters {
    /// Compiles `source` (`light_cluster_build.wgsl`, includes expanded) and
    /// allocates room for one light at the default cap.
    pub(crate) fn new(device: &wgpu::Device, source: &str) -> Result<Self, String> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(LIGHT_CLUSTER_BUILD),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let bounds_pipeline = pipeline("bounds_main");
        let cull_pipeline = pipeline("cull_main");
        if let Some(error) = block_on(scope.pop()) {
            return Err(error.to_string());
        }
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster-params"),
            size: CLUSTER_PARAMS_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cap = RenderFrame::DEFAULT_CLUSTER_LIGHT_CAP;
        let lights = storage_buffer(
            device,
            "local-lights",
            GPU_LIGHT_SIZE as u64,
            wgpu::BufferUsages::COPY_DST,
        );
        let grid = storage_buffer(
            device,
            "light-grid",
            grid_words(cap) * 4,
            wgpu::BufferUsages::empty(),
        );
        let bounds = storage_buffer(
            device,
            "cluster-bounds",
            u64::from(CLUSTER_COUNT) * 32,
            wgpu::BufferUsages::empty(),
        );
        let overflow = storage_buffer(
            device,
            "light-overflow",
            4,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let bounds_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cluster-bounds"),
            layout: &bounds_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bounds.as_entire_binding(),
                },
            ],
        });
        let cull_group = cull_group(
            device,
            &cull_pipeline,
            [&params, &lights, &bounds, &grid, &overflow],
        );
        Ok(Self {
            params,
            lights,
            grid,
            light_capacity: 1,
            cap,
            bounds,
            overflow,
            bounds_pipeline,
            cull_pipeline,
            bounds_group,
            cull_group,
            staging: Vec::new(),
            count: 0,
            readback: OverflowReadback::new(device),
        })
    }

    /// Uploads this frame's lights and cluster parameters, growing the
    /// buffers first if needed. Call before the surface is acquired. Returns
    /// whether a buffer group 0 binds was recreated, so group 0 must be too.
    ///
    /// A frame without a perspective camera has no depth slices to cluster,
    /// so it turns local lights off.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &RenderFrame,
        size: (u32, u32),
    ) -> bool {
        // A slot still active belongs to a frame that was never submitted.
        self.cancel();
        // Mapping callbacks run only when the device is polled, and the GPU
        // timer, the other poller, is optional.
        let _ = device.poll(wgpu::PollType::Poll);
        self.readback.collect();
        self.readback.frame += 1;
        let depths = crate::shadow::view_depths(&frame.view_proj);
        self.count = match depths {
            Some(_) => pack_lights(&frame.lights, &mut self.staging),
            None => 0,
        };
        if self.count == 0 {
            // No lights, nothing dropped; this outranks any reading in flight.
            self.readback.latest = Some((self.readback.frame, 0));
        }
        let cap = frame.cluster_light_cap.clamp(1, MAX_CLUSTER_CAP);
        let capacity = crate::uniforms::grown_capacity(self.light_capacity, self.count);
        let recreated = capacity != self.light_capacity || cap != self.cap;
        if capacity != self.light_capacity {
            let bytes = u64::from(capacity) * GPU_LIGHT_SIZE as u64;
            self.lights = storage_buffer(device, "local-lights", bytes, wgpu::BufferUsages::COPY_DST);
            self.light_capacity = capacity;
        }
        if cap != self.cap {
            self.grid = storage_buffer(
                device,
                "light-grid",
                grid_words(cap) * 4,
                wgpu::BufferUsages::empty(),
            );
            self.cap = cap;
        }
        if recreated {
            self.cull_group = cull_group(
                device,
                &self.cull_pipeline,
                [
                    &self.params,
                    &self.lights,
                    &self.bounds,
                    &self.grid,
                    &self.overflow,
                ],
            );
        }
        let depths = depths.unwrap_or(ViewDepths {
            eye: Vec3::ZERO,
            forward: Vec3::Z,
            near: 1.0,
            far: 2.0,
        });
        let inverse = crate::uniforms::inverse(&frame.view_proj).unwrap_or(crate::uniforms::IDENTITY);
        queue.write_buffer(
            &self.params,
            0,
            &cluster_params_bytes(&inverse, &depths, self.count, self.cap, size),
        );
        if self.count > 0 {
            queue.write_buffer(&self.lights, 0, &self.staging);
            // Reserved now, as recording borrows the clusters immutably. A
            // full ring leaves this frame's overflow unread.
            self.readback.active = self.readback.ring.acquire((self.readback.frame, None));
        }
        recreated
    }

    /// Whether this frame culls lights, so records the cluster pass.
    pub(crate) fn active(&self) -> bool {
        self.count > 0
    }

    /// Bounds every cluster, lists its lights, and copies the overflow count
    /// for a later readback.
    pub(crate) fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.overflow, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("light-clusters"),
                timestamp_writes: None,
            });
            let groups = CLUSTER_COUNT.div_ceil(WORKGROUP);
            pass.set_pipeline(&self.bounds_pipeline);
            pass.set_bind_group(0, &self.bounds_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
            pass.set_pipeline(&self.cull_pipeline);
            pass.set_bind_group(0, &self.cull_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        if let Some(slot) = self.readback.active {
            encoder.copy_buffer_to_buffer(&self.overflow, 0, &self.readback.buffers[slot], 0, 4);
        }
    }

    /// Starts mapping this frame's overflow readback, once it is submitted.
    pub(crate) fn submitted(&mut self) {
        let Some(slot) = self.readback.active.take() else {
            return;
        };
        let (sender, receiver) = std::sync::mpsc::channel();
        if let Some((_, pending)) = self.readback.ring.get_mut(slot) {
            *pending = Some(receiver);
        }
        self.readback.buffers[slot]
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
    }

    /// Frees the readback slot of a frame that was prepared but never
    /// submitted.
    pub(crate) fn cancel(&mut self) {
        if let Some(slot) = self.readback.active.take() {
            self.readback.ring.complete(slot);
        }
    }

    /// Light-cluster pairs dropped for want of room, from the latest
    /// readback, which lags the frame a little. `None` before the first.
    pub(crate) fn latest_overflow(&self) -> Option<u32> {
        self.readback.latest.map(|(_, value)| value)
    }

    /// Waits for every readback in flight and reads it. Tests only.
    #[cfg(test)]
    pub(crate) fn collect_blocking(&mut self, device: &wgpu::Device) {
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the device polls");
        self.readback.collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::SpotCone;

    /// A small deterministic generator, so failures reproduce.
    struct Random(u64);

    impl Random {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1_u64 << 24) as f32
        }

        fn range(&mut self, low: f32, high: f32) -> f32 {
            low + (high - low) * self.next()
        }

        fn vec(&mut self, low: f32, high: f32) -> Vec3 {
            Vec3::new(
                self.range(low, high),
                self.range(low, high),
                self.range(low, high),
            )
        }

        fn unit(&mut self) -> Vec3 {
            loop {
                if let Some(unit) = self.vec(-1.0, 1.0).try_normalize() {
                    return unit;
                }
            }
        }
    }

    #[test]
    fn depth_slices_run_from_near_to_far_monotonic_and_contiguous() {
        let (near, far) = (0.1, 500.0);
        assert!((slice_depth(0, near, far) - near).abs() < 1e-6);
        assert!((slice_depth(GRID[2], near, far) - far).abs() < far * 1e-5);
        for slice in 0..GRID[2] {
            let (start, end) = (slice_depth(slice, near, far), slice_depth(slice + 1, near, far));
            assert!(end > start, "slice {slice} has positive depth");
            assert_eq!(
                slice_of(start * 1.0001, near, far),
                slice,
                "just past the start of {slice}"
            );
            assert_eq!(
                slice_of(end * 0.9999, near, far),
                slice,
                "just before the end of {slice}"
            );
        }
        let mut previous = 0;
        let mut seen = vec![false; GRID[2] as usize];
        for step in 0..=10_000 {
            let depth = near * (far / near).powf(step as f32 / 10_000.0);
            let slice = slice_of(depth, near, far);
            assert!(
                slice >= previous && slice <= previous + 1,
                "no slice is skipped at {depth}"
            );
            seen[slice as usize] = true;
            previous = slice;
        }
        assert!(seen.iter().all(|&hit| hit), "every slice holds some depth");
        assert_eq!(slice_of(0.0, near, far), 0, "in front of the near plane");
        assert_eq!(
            slice_of(far * 10.0, near, far),
            GRID[2] - 1,
            "beyond the far plane"
        );
    }

    /// The closest point of the box to `centre`, found by sampling the box
    /// on a grid that includes its faces.
    fn brute_force_distance(centre: Vec3, low: Vec3, high: Vec3) -> f32 {
        const STEPS: u32 = 10;
        let mut best = f32::MAX;
        for i in 0..=STEPS {
            for j in 0..=STEPS {
                for k in 0..=STEPS {
                    let t = Vec3::new(i as f32, j as f32, k as f32) / STEPS as f32;
                    best = best.min(centre.distance(low + (high - low) * t));
                }
            }
        }
        best
    }

    #[test]
    fn the_sphere_box_test_agrees_with_brute_force_on_random_samples() {
        let mut random = Random(0x5eed_1234_abcd_ef01);
        for sample in 0..10_000 {
            let low = random.vec(-4.0, 4.0);
            let high = low + random.vec(0.05, 3.0);
            let centre = random.vec(-8.0, 8.0);
            let radius = random.range(0.01, 4.0);
            let closest = brute_force_distance(centre, low, high);
            let touches = sphere_touches_box(centre, radius, low, high);
            if closest <= radius {
                assert!(
                    touches,
                    "sample {sample}: a sampled box point lies inside the sphere"
                );
            }
            // The sampling grid can miss the true closest point by half a cell.
            let slack = (high - low).length() / 20.0;
            if touches {
                assert!(
                    closest <= radius + slack,
                    "sample {sample}: touches only when it is close"
                );
            }
        }
    }

    #[test]
    fn the_cone_test_never_culls_a_sphere_holding_a_lit_point() {
        let mut random = Random(0x0c0e_5eed_7777_1111);
        for sample in 0..10_000 {
            let outer = random.range(0.05, 1.5);
            let cone = CullCone {
                apex: random.vec(-5.0, 5.0),
                axis: random.unit(),
                range: random.range(0.5, 20.0),
                cos_outer: outer.cos(),
                sin_outer: outer.sin(),
            };
            // A point the light reaches: inside its range and its cone.
            let direction = loop {
                let candidate = random.unit();
                if candidate.dot(cone.axis) >= cone.cos_outer {
                    break candidate;
                }
            };
            let lit = cone.apex + direction * random.range(0.0, cone.range);
            let radius = random.range(0.01, 3.0);
            let centre = lit + random.unit() * random.range(0.0, radius);
            assert!(
                !cone_misses_sphere(&cone, centre, radius),
                "sample {sample}: a sphere holding a lit point was culled"
            );
            let behind = cone.apex - cone.axis * (radius + random.range(0.01, 5.0));
            assert!(
                cone_misses_sphere(&cone, behind, radius),
                "sample {sample}: a sphere wholly behind the apex is culled"
            );
            let beyond = cone.apex + cone.axis * (cone.range + radius + random.range(0.01, 5.0));
            assert!(
                cone_misses_sphere(&cone, beyond, radius),
                "sample {sample}: a sphere beyond the range is culled"
            );
        }
    }

    #[test]
    fn a_cone_wider_than_a_hemisphere_is_never_culled_by_angle() {
        let cone = CullCone {
            apex: Vec3::ZERO,
            axis: Vec3::Y,
            range: 5.0,
            cos_outer: 2.0_f32.cos(),
            sin_outer: 2.0_f32.sin(),
        };
        assert!(!cone_misses_sphere(&cone, Vec3::new(0.0, -1.0, 0.0), 0.1));
    }

    fn floats(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
            .collect()
    }

    #[test]
    fn lights_pack_as_the_shader_reads_them_and_bad_ones_are_dropped() {
        let point = LocalLight {
            position: [1.0, 2.0, 3.0],
            range: 4.0,
            color: [0.5, 0.25, 1.0],
            intensity: 2.0,
            spot: None,
        };
        let spot = LocalLight {
            spot: Some(SpotCone {
                direction: [0.0, -2.0, 0.0],
                inner_angle: 0.5,
                outer_angle: 0.5,
            }),
            ..point
        };
        let broken = [
            LocalLight { range: 0.0, ..point },
            LocalLight {
                intensity: 0.0,
                ..point
            },
            LocalLight {
                position: [f32::NAN, 0.0, 0.0],
                ..point
            },
            LocalLight {
                spot: Some(SpotCone {
                    direction: [0.0; 3],
                    inner_angle: 0.1,
                    outer_angle: 0.2,
                }),
                ..point
            },
        ];
        let mut lights = vec![point, spot];
        lights.extend(broken);
        let mut out = Vec::new();
        assert_eq!(pack_lights(&lights, &mut out), 2);
        assert_eq!(out.len(), 2 * GPU_LIGHT_SIZE);

        let packed = floats(&out);
        assert_eq!(&packed[0..8], &[1.0, 2.0, 3.0, 4.0, 0.5, 0.25, 1.0, 2.0]);
        assert_eq!(packed[11], 0.0, "a point light is not a spot");
        let spot_words = &packed[16..32];
        assert_eq!(
            &spot_words[8..12],
            &[0.0, -1.0, 0.0, 1.0],
            "unit direction, spot flag"
        );
        let (cos_inner, cos_outer) = (spot_words[12], spot_words[13]);
        assert!(
            cos_inner > cos_outer,
            "equal angles still give distinct smoothstep edges"
        );
        assert!((cos_outer - 0.5_f32.cos()).abs() < 1e-6);
        assert!((spot_words[14] - 0.5_f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn a_needle_thin_spot_still_reaches_full_intensity_on_its_axis() {
        let needle = LocalLight {
            spot: Some(SpotCone {
                direction: [0.0, 0.0, 1.0],
                inner_angle: 0.0,
                outer_angle: 0.0,
            }),
            ..LocalLight::default()
        };
        let mut out = Vec::new();
        assert_eq!(pack_lights(&[needle], &mut out), 1);
        let packed = floats(&out);
        let (cos_inner, cos_outer) = (packed[12], packed[13]);
        assert!(
            cos_inner <= 1.0,
            "the axis, at cosine 1, is inside the inner edge"
        );
        assert!(cos_inner > cos_outer);
    }

    #[test]
    fn at_most_the_light_limit_is_uploaded() {
        let lights = vec![LocalLight::default(); MAX_LOCAL_LIGHTS + 5];
        let mut out = Vec::new();
        assert_eq!(pack_lights(&lights, &mut out) as usize, MAX_LOCAL_LIGHTS);
    }

    #[test]
    fn cluster_params_follow_the_shader_layout() {
        let depths = ViewDepths {
            eye: Vec3::new(1.0, 2.0, 3.0),
            forward: Vec3::Z,
            near: 0.1,
            far: 100.0,
        };
        let mut inverse = crate::uniforms::IDENTITY;
        inverse[3][0] = 7.0;
        let bytes = cluster_params_bytes(&inverse, &depths, 12, 64, (1920, 1080));
        let values = floats(&bytes);
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four bytes")))
            .collect();
        assert_eq!(values[12], 7.0, "column-major: column 3, row 0");
        assert_eq!(&values[16..24], &[1.0, 2.0, 3.0, 0.1, 0.0, 0.0, 1.0, 100.0]);
        assert_eq!(&words[24..27], &GRID);
        assert_eq!(&words[28..30], &[12, 64]);
        assert_eq!(&values[32..34], &[1920.0, 1080.0]);
    }

    #[test]
    fn the_largest_light_lists_fit_the_downlevel_binding_size() {
        assert_eq!(CLUSTER_COUNT, 3456);
        let bytes = grid_words(MAX_CLUSTER_CAP) * 4;
        assert_eq!(bytes, 3456 * 257 * 4, "about 3.5 MB at the largest cap");
        let limit = wgpu::Limits::downlevel_defaults().max_storage_buffer_binding_size;
        assert!(bytes <= limit);
        assert!(MAX_LOCAL_LIGHTS as u64 * GPU_LIGHT_SIZE as u64 <= limit);
    }

    /// Storage buffers each entry point of `shader` uses, by name, counted
    /// from the shader itself so a new binding cannot slip past the ADR-0023
    /// floor unnoticed.
    fn storage_buffers(shader: &str) -> Vec<(String, usize)> {
        use wgpu::naga::{AddressSpace, valid};
        let mut library = crate::shaders::ShaderLibrary::new(None);
        let source = library.expanded(shader).expect("embedded includes resolve");
        let module = wgpu::naga::front::wgsl::parse_str(&source.text).expect("parses");
        let info = valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
            .validate(&module)
            .expect("validates");
        module
            .entry_points
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                let usage = info.get_entry_point(index);
                let count = module
                    .global_variables
                    .iter()
                    .filter(|(handle, global)| {
                        matches!(global.space, AddressSpace::Storage { .. }) && !usage[*handle].is_empty()
                    })
                    .count();
                (entry.name.clone(), count)
            })
            .collect()
    }

    #[test]
    fn every_stage_stays_within_the_downlevel_storage_buffer_limit() {
        let floor = wgpu::Limits::downlevel_defaults().max_storage_buffers_per_shader_stage as usize;
        let named = |entry: &str, count: usize| (entry.to_owned(), count);
        assert_eq!(
            storage_buffers(LIGHT_CLUSTER_BUILD),
            vec![named("bounds_main", 1), named("cull_main", 4)],
            "the cull binds lights, bounds, lists and the overflow counter"
        );
        assert_eq!(
            storage_buffers(crate::shaders::PBR_LIT),
            vec![named("vs_main", 0), named("fs_main", 2)],
            "PBR fragments read lights and lists"
        );
        assert_eq!(floor, 4, "the floor these counts are held to");
    }
}
