//! GPU pass timing with bounded, asynchronous readback.
//!
//! Timestamps are written at the start and end of each graph pass, resolved
//! into a buffer, and read back [`LATENCY_FRAMES`] frames later, so measuring
//! never stalls the CPU. A frame that finds every readback slot busy is simply
//! not timed. A timing is `None`, never zero, when its value is unknown.

/// GPU time spent in one render-graph pass.
///
/// `pub` rather than `pub(crate)` because the overlay's public stats will carry
/// it; until that re-export exists the module is private and this lint is
/// expected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassTiming {
    /// The graph name of the pass, so the overlay can label the number.
    pub name: &'static str,
    /// Measured GPU time in milliseconds.
    pub gpu_ms: Option<f32>,
}

/// Passes that can be timed in one frame.
///
/// Fixed so the query set and resolve buffer are sized once at startup. A
/// graph with more passes than this leaves the extra passes untimed rather
/// than growing a buffer on the frame path.
pub(crate) const MAX_TIMED_PASSES: u32 = 32;

/// Frames between recording a timestamp and reading it back.
///
/// The GPU is usually a few frames behind the CPU, so reading a frame's
/// timestamps this many frames later finds them already resolved without
/// waiting on the device.
pub(crate) const LATENCY_FRAMES: usize = 3;

/// Nanoseconds in a millisecond. Timestamp periods are in nanoseconds, and
/// the overlay shows milliseconds.
const NANOS_PER_MILLI: f64 = 1_000_000.0;

/// The two query slots that bracket `pass`, begin first.
///
/// Each pass owns an adjacent pair so one resolve covers every pass with no
/// per-pass bookkeeping. Returns `None` past [`MAX_TIMED_PASSES`], which
/// leaves that pass untimed instead of indexing past the query set.
pub(crate) fn query_indices(pass: u32) -> Option<(u32, u32)> {
    if pass >= MAX_TIMED_PASSES {
        return None;
    }
    Some((pass * 2, pass * 2 + 1))
}

/// Converts flat begin/end tick pairs into milliseconds, one entry per pair.
///
/// An odd trailing tick belongs to no pair and is ignored. A pair is `None`
/// when its begin is zero, which means the timestamp was never written, when
/// the end does not come after the begin, which is a wrapped or stale
/// reading, or when the period is not a positive finite number. Reporting
/// those as zero would make a broken measurement look like a free pass.
pub(crate) fn pass_millis(ticks: &[u64], period_ns: f32) -> Vec<Option<f32>> {
    let period_is_valid = period_ns.is_finite() && period_ns > 0.0;
    ticks
        .chunks_exact(2)
        .map(|pair| {
            let &[begin, end] = pair else {
                return None;
            };
            if !period_is_valid || begin == 0 {
                return None;
            }
            let elapsed = end.checked_sub(begin)?;
            if elapsed == 0 {
                return None;
            }
            Some((elapsed as f64 * period_ns as f64 / NANOS_PER_MILLI) as f32)
        })
        .collect()
}

/// Pairs each pass name with its measured time, retaining unknown readings.
///
/// Names are matched to pairs by position, so the graph must report passes
/// in the order it recorded them. Invalid readings remain `None`.
pub(crate) fn timings(names: &[&'static str], ticks: &[u64], period_ns: f32) -> Vec<PassTiming> {
    names
        .iter()
        .zip(pass_millis(ticks, period_ns))
        .map(|(&name, gpu_ms)| PassTiming { name, gpu_ms })
        .collect()
}

/// Slots for readbacks still in flight, each holding the label of its frame.
///
/// A slot is reserved when its resolve is recorded and freed when its
/// readback is read. The ring is bounded so a stalled readback cannot make
/// the timing buffers grow without limit.
#[derive(Debug)]
pub(crate) struct ReadbackRing<T> {
    slots: Vec<Option<T>>,
    /// Where the next search starts. Advancing it spreads acquisitions around
    /// the ring instead of always reusing slot 0 first.
    next: usize,
}

impl<T> ReadbackRing<T> {
    /// A ring with `len` free slots. Zero is allowed and never hands out a slot.
    pub(crate) fn new(len: usize) -> Self {
        Self {
            slots: std::iter::repeat_with(|| None).take(len).collect(),
            next: 0,
        }
    }

    /// Reserves a free slot for `label`, or returns `None` when all are busy.
    ///
    /// A busy ring means the frame is not timed, which is the intended
    /// back-pressure: the frame still renders. On `None` the ring is left
    /// unchanged, and `label` is dropped.
    pub(crate) fn acquire(&mut self, label: T) -> Option<usize> {
        let len = self.slots.len();
        let free = (0..len)
            .map(|offset| (self.next + offset) % len)
            .find(|&index| self.slots.get(index).is_some_and(Option::is_none))?;
        let slot = self.slots.get_mut(free)?;
        *slot = Some(label);
        self.next = (free + 1) % len;
        Some(free)
    }

    /// Frees `slot` and returns its label. An out-of-range or already free
    /// slot gives `None`, so a bad index cannot corrupt the ring.
    pub(crate) fn complete(&mut self, slot: usize) -> Option<T> {
        self.slots.get_mut(slot).and_then(Option::take)
    }

    /// The label in a busy slot; `None` for a free or out-of-range one.
    pub(crate) fn get(&self, slot: usize) -> Option<&T> {
        self.slots.get(slot).and_then(Option::as_ref)
    }

    /// The label in a busy slot, to update in place.
    pub(crate) fn get_mut(&mut self, slot: usize) -> Option<&mut T> {
        self.slots.get_mut(slot).and_then(Option::as_mut)
    }

    /// How many slots hold a readback that has not completed.
    #[cfg(test)]
    pub(crate) fn in_flight(&self) -> usize {
        self.slots.iter().filter(|slot| slot.is_some()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn query_indices_pairs_each_pass_with_its_two_slots() {
        assert_eq!(query_indices(0), Some((0, 1)));
        assert_eq!(query_indices(5), Some((10, 11)));
        assert_eq!(query_indices(31), Some((62, 63)));
        assert_eq!(query_indices(32), None);
    }

    #[test]
    fn a_begin_end_pair_converts_to_milliseconds() {
        // 1000 ticks at 1 ns per tick is 1 microsecond, i.e. 0.001 ms.
        let result = pass_millis(&[100, 1100], 1.0);
        assert_eq!(result.len(), 1);
        let millis = result[0].expect("a valid pair has a time");
        assert!((millis - 0.001).abs() < EPSILON);
    }

    #[test]
    fn a_zero_length_pass_is_unknown_not_zero() {
        let result = pass_millis(&[100, 1100, 5000, 5000], 1.0);
        assert_eq!(result.len(), 2);
        assert!(result[0].is_some());
        assert_eq!(result[1], None);
    }

    #[test]
    fn a_wrapped_reading_is_unknown() {
        assert_eq!(pass_millis(&[5000, 100], 1.0), vec![None]);
    }

    #[test]
    fn an_unwritten_begin_timestamp_is_unknown() {
        assert_eq!(pass_millis(&[0, 100], 1.0), vec![None]);
    }

    #[test]
    fn an_odd_trailing_tick_is_ignored() {
        assert_eq!(pass_millis(&[1, 2, 3], 1.0).len(), 1);
    }

    #[test]
    fn a_non_positive_or_non_finite_period_makes_every_pass_unknown() {
        let ticks = [100, 1100];
        assert_eq!(pass_millis(&ticks, f32::NAN), vec![None]);
        assert_eq!(pass_millis(&ticks, 0.0), vec![None]);
        assert_eq!(pass_millis(&ticks, -1.0), vec![None]);
    }

    #[test]
    fn no_ticks_gives_no_passes() {
        assert!(pass_millis(&[], 1.0).is_empty());
    }

    #[test]
    fn timings_keep_only_the_passes_with_a_known_time() {
        let names = ["lit", "tonemap"];
        let result = timings(&names, &[0, 10, 100, 300], 1000.0);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].gpu_ms, None);
        assert_eq!(result[1].name, "tonemap");
        assert!((result[1].gpu_ms.expect("valid") - 0.2).abs() < EPSILON);
    }

    #[test]
    fn a_ring_hands_out_free_slots_in_order_then_runs_out() {
        let mut ring = ReadbackRing::new(3);
        assert_eq!(ring.acquire("a"), Some(0));
        assert_eq!(ring.acquire("b"), Some(1));
        assert_eq!(ring.acquire("c"), Some(2));
        assert_eq!(ring.acquire("d"), None, "a full ring leaves the frame untimed");
        assert_eq!(ring.in_flight(), 3);
    }

    #[test]
    fn a_completed_slot_is_the_next_one_reused() {
        let mut ring = ReadbackRing::new(3);
        ring.acquire("a");
        ring.acquire("b");
        ring.acquire("c");
        assert_eq!(ring.complete(1), Some("b"));
        assert_eq!(ring.acquire("e"), Some(1));
        assert_eq!(ring.complete(7), None, "an out-of-range slot is ignored");
    }

    #[test]
    fn a_ring_sized_to_the_latency_holds_one_readback_per_frame() {
        let mut ring = ReadbackRing::new(LATENCY_FRAMES);
        for frame in 0..LATENCY_FRAMES {
            assert!(ring.acquire(frame).is_some(), "frame {frame} gets a slot");
        }
        assert_eq!(ring.acquire(LATENCY_FRAMES), None);
        assert_eq!(ring.in_flight(), LATENCY_FRAMES);
    }

    #[test]
    fn a_zero_length_ring_never_hands_out_a_slot() {
        let mut ring = ReadbackRing::<&str>::new(0);
        assert_eq!(ring.acquire("x"), None);
        assert_eq!(ring.in_flight(), 0);
    }
}

/// Optional features used by the graph profiler.
pub(crate) const FEATURES: wgpu::Features =
    wgpu::Features::TIMESTAMP_QUERY.union(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);

#[derive(Debug)]
struct Pending {
    frame: u64,
    names: Vec<&'static str>,
    ready: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

/// Query resources shared by frames on one queue. Readback slots are never reused
/// until their mapping callback has completed and their buffer is unmapped.
#[derive(Debug)]
pub(crate) struct GpuTimer {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readbacks: Vec<wgpu::Buffer>,
    ring: ReadbackRing<Pending>,
    active: Option<usize>,
    frame: u64,
    period: f32,
    latest: Option<(u64, Vec<PassTiming>)>,
}

impl GpuTimer {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(FEATURES) {
            return None;
        }
        let size = u64::from(MAX_TIMED_PASSES) * 2 * 8;
        Some(Self {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("graph-timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: MAX_TIMED_PASSES * 2,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp-resolve"),
                size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readbacks: (0..LATENCY_FRAMES)
                .map(|_| {
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("timestamp-readback"),
                        size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                })
                .collect(),
            ring: ReadbackRing::new(LATENCY_FRAMES),
            active: None,
            frame: 0,
            period: queue.get_timestamp_period(),
            latest: None,
        })
    }

    pub(crate) fn begin_frame(&mut self, device: &wgpu::Device) {
        self.frame += 1;
        let _ = device.poll(wgpu::PollType::Poll);
        for slot in 0..self.readbacks.len() {
            let Some(pending) = &self.ring.slots[slot] else {
                continue;
            };
            if self.frame.saturating_sub(pending.frame) < LATENCY_FRAMES as u64 {
                continue;
            }
            let Some(ready) = &pending.ready else {
                continue;
            };
            let result = match ready.try_recv() {
                Ok(result) => result,
                Err(std::sync::mpsc::TryRecvError::Empty) => continue,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.readbacks[slot].unmap();
                    self.ring.complete(slot);
                    continue;
                }
            };
            let Some(pending) = self.ring.complete(slot) else {
                continue;
            };
            if result.is_ok() {
                let Ok(bytes) = self.readbacks[slot]
                    .slice(..(pending.names.len() * 16) as u64)
                    .get_mapped_range()
                else {
                    self.readbacks[slot].unmap();
                    continue;
                };
                let ticks: Vec<u64> = bytes
                    .chunks_exact(8)
                    .filter_map(|tick| tick.try_into().ok().map(u64::from_ne_bytes))
                    .collect();
                let values = timings(&pending.names, &ticks, self.period);
                if self
                    .latest
                    .as_ref()
                    .is_none_or(|(frame, _)| pending.frame > *frame)
                {
                    self.latest = Some((pending.frame, values));
                }
                drop(bytes);
            }
            self.readbacks[slot].unmap();
        }
    }

    pub(crate) fn start(&mut self, mut names: Vec<&'static str>) -> Option<wgpu::QuerySet> {
        names.truncate(MAX_TIMED_PASSES as usize);
        if names.is_empty() {
            return None;
        }
        self.active = self.ring.acquire(Pending {
            frame: self.frame,
            names,
            ready: None,
        });
        self.active.map(|_| self.queries.clone())
    }

    pub(crate) fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let Some(slot) = self.active else {
            return;
        };
        let Some(pending) = self.ring.slots.get(slot).and_then(Option::as_ref) else {
            return;
        };
        // `start` truncates to MAX_TIMED_PASSES, so this cannot overflow.
        let count = pending.names.len() as u32 * 2;
        encoder.resolve_query_set(&self.queries, 0..count, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readbacks[slot], 0, u64::from(count) * 8);
    }

    pub(crate) fn submitted(&mut self) {
        let Some(slot) = self.active.take() else {
            return;
        };
        let Some(pending) = self.ring.slots.get_mut(slot).and_then(Option::as_mut) else {
            return;
        };
        let (sender, receiver) = std::sync::mpsc::channel();
        pending.ready = Some(receiver);
        self.readbacks[slot]
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
    }

    pub(crate) fn cancel(&mut self) {
        if let Some(slot) = self.active.take() {
            self.ring.complete(slot);
        }
    }

    pub(crate) fn latest(&self) -> Option<&[PassTiming]> {
        self.latest.as_ref().map(|(_, values)| values.as_slice())
    }

    pub(crate) fn latest_frame(&self) -> Option<u64> {
        self.latest.as_ref().map(|(frame, _)| *frame)
    }
}
