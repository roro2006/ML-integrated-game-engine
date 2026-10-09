//! The transient texture pool.
//!
//! Each frame's transients are taken from textures kept since the last frame
//! whenever a descriptor matches exactly, so a steady frame creates no
//! textures at all. Each transient holds its texture for the whole frame:
//! memory aliasing by lifetime is not done in v1, though the graph records
//! the lifetimes it would need.
//!
//! A texture no frame has asked for in [`RETIRE_AFTER_FRAMES`] frames is
//! dropped. Waiting a few frames rather than one keeps a target that a setting
//! toggles on and off from being destroyed and recreated on every toggle, and
//! still retires the old size after a resize without a separate invalidation
//! step: the new size is a new descriptor, so nothing asks for the old one.
//! `clear` exists for when the renderer wants the memory back at once.
//!
//! The pool is generic over the texture type so that reuse is tested without
//! a device; the renderer instantiates it with `wgpu::Texture`.

use super::TransientDesc;

/// Frames a texture may go unasked for before the pool drops it.
pub(crate) const RETIRE_AFTER_FRAMES: u32 = 4;

#[derive(Debug)]
struct Entry<T> {
    desc: TransientDesc,
    texture: T,
    in_use: bool,
    /// Consecutive frames, before this one, in which nothing asked for it.
    unused_frames: u32,
}

/// Index of a texture handed out this frame. Valid until the next
/// [`TexturePool::begin_frame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PoolSlot(usize);

#[derive(Debug)]
pub(crate) struct TexturePool<T> {
    entries: Vec<Entry<T>>,
    /// Textures created over the pool's life, for tests and the overlay.
    created: u64,
}

impl<T> Default for TexturePool<T> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            created: 0,
        }
    }
}

impl<T> TexturePool<T> {
    /// Drops what has gone unused for [`RETIRE_AFTER_FRAMES`] frames, then
    /// frees everything for this frame.
    pub(crate) fn begin_frame(&mut self) {
        for entry in &mut self.entries {
            entry.unused_frames = if entry.in_use { 0 } else { entry.unused_frames + 1 };
            entry.in_use = false;
        }
        self.entries
            .retain(|entry| entry.unused_frames < RETIRE_AFTER_FRAMES);
    }

    /// A free texture matching `desc`, created with `create` if none is free.
    ///
    /// Two transients with the same descriptor in one frame get two textures.
    pub(crate) fn acquire(
        &mut self,
        desc: TransientDesc,
        create: impl FnOnce(&TransientDesc) -> T,
    ) -> PoolSlot {
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| !entry.in_use && entry.desc == desc)
        {
            self.entries[index].in_use = true;
            return PoolSlot(index);
        }
        self.entries.push(Entry {
            desc,
            texture: create(&desc),
            in_use: true,
            unused_frames: 0,
        });
        self.created += 1;
        PoolSlot(self.entries.len() - 1)
    }

    pub(crate) fn get(&self, slot: PoolSlot) -> Option<&T> {
        self.entries.get(slot.0).map(|entry| &entry.texture)
    }

    /// Releases every texture now.
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub(crate) fn created(&self) -> u64 {
        self.created
    }
}

/// Creates the GPU texture for a transient, labelled with its graph name so
/// captures and validation messages say which target it is.
pub(crate) fn create_texture(device: &wgpu::Device, label: &str, desc: &TransientDesc) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: desc.width,
            height: desc.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: desc.format,
        usage: desc.usage,
        view_formats: &[],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(width: u32, height: u32) -> TransientDesc {
        TransientDesc {
            format: wgpu::TextureFormat::Rgba16Float,
            width,
            height,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        }
    }

    /// Stands in for a texture: records the descriptor it was made from.
    fn make(desc: &TransientDesc) -> TransientDesc {
        *desc
    }

    #[test]
    fn a_steady_frame_reuses_every_texture() {
        let mut pool = TexturePool::default();
        for _ in 0..3 {
            pool.begin_frame();
            pool.acquire(desc(64, 64), make);
            pool.acquire(
                TransientDesc {
                    format: wgpu::TextureFormat::Depth32Float,
                    ..desc(64, 64)
                },
                make,
            );
        }
        assert_eq!(pool.created(), 2, "created on the first frame only");
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn a_differing_descriptor_is_a_miss() {
        let mut pool = TexturePool::default();
        pool.begin_frame();
        let colour = pool.acquire(desc(64, 64), make);
        let storage = pool.acquire(
            TransientDesc {
                usage: wgpu::TextureUsages::STORAGE_BINDING,
                ..desc(64, 64)
            },
            make,
        );
        assert_ne!(colour, storage);
        assert_eq!(pool.created(), 2, "usage is part of the key");
        assert_eq!(pool.get(colour), Some(&desc(64, 64)));
    }

    #[test]
    fn identical_descriptors_in_one_frame_get_separate_textures() {
        let mut pool = TexturePool::default();
        pool.begin_frame();
        let a = pool.acquire(desc(32, 32), make);
        let b = pool.acquire(desc(32, 32), make);
        assert_ne!(a, b, "two live transients never share a texture in v1");
        assert_eq!(pool.created(), 2);
    }

    #[test]
    fn an_old_size_is_retired_after_going_unused_for_a_few_frames() {
        let mut pool = TexturePool::default();
        pool.begin_frame();
        pool.acquire(desc(1280, 720), make);

        // Frames 1..=RETIRE_AFTER_FRAMES ask only for the new size.
        for frame in 1..=RETIRE_AFTER_FRAMES {
            pool.begin_frame();
            pool.acquire(desc(1920, 1080), make);
            assert_eq!(pool.len(), 2, "the old size is kept on frame {frame}");
        }
        pool.begin_frame();
        assert_eq!(pool.len(), 1, "then it is gone");
        pool.acquire(desc(1920, 1080), make);
        assert_eq!(pool.created(), 2, "and the new size was only ever created once");
    }

    #[test]
    fn a_target_toggled_off_briefly_is_not_recreated() {
        let mut pool = TexturePool::default();
        pool.begin_frame();
        pool.acquire(desc(64, 64), make);
        // Off for fewer frames than the retirement threshold, then back on.
        for _ in 1..RETIRE_AFTER_FRAMES {
            pool.begin_frame();
        }
        pool.begin_frame();
        pool.acquire(desc(64, 64), make);
        assert_eq!(pool.created(), 1);
    }

    #[test]
    fn clear_releases_everything_at_once() {
        let mut pool = TexturePool::default();
        pool.begin_frame();
        pool.acquire(desc(8, 8), make);
        pool.clear();
        assert_eq!(pool.len(), 0);
        pool.begin_frame();
        pool.acquire(desc(8, 8), make);
        assert_eq!(pool.created(), 2, "a cleared pool creates again");
    }
}
