//! The transient texture pool.
//!
//! Each frame's transients are taken from textures kept since the last frame
//! whenever a descriptor matches exactly, so a steady frame creates no
//! textures at all. Each transient holds its texture for the whole frame:
//! memory aliasing by lifetime is not done in v1, though the graph records
//! the lifetimes it would need.
//!
//! A texture no frame asked for is dropped at the start of the next frame.
//! That is what makes a resize safe without a separate invalidation step: the
//! new size is a new descriptor, nothing asks for the old one, and it goes.
//! `clear` exists for when the renderer wants the memory back at once.
//!
//! The pool is generic over the texture type so that reuse is tested without
//! a device; the renderer instantiates it with `wgpu::Texture`.

use super::TextureDesc;

#[derive(Debug)]
struct Entry<T> {
    desc: TextureDesc,
    texture: T,
    in_use: bool,
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
    /// Drops what the last frame did not use, then frees everything for this
    /// frame.
    pub(crate) fn begin_frame(&mut self) {
        self.entries.retain(|entry| entry.in_use);
        for entry in &mut self.entries {
            entry.in_use = false;
        }
    }

    /// A free texture matching `desc`, created with `create` if none is free.
    ///
    /// Two transients with the same descriptor in one frame get two textures.
    pub(crate) fn acquire(&mut self, desc: TextureDesc, create: impl FnOnce(&TextureDesc) -> T) -> PoolSlot {
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

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn created(&self) -> u64 {
        self.created
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(width: u32, height: u32) -> TextureDesc {
        TextureDesc {
            format: wgpu::TextureFormat::Rgba16Float,
            width,
            height,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        }
    }

    /// Stands in for a texture: records the descriptor it was made from.
    fn make(desc: &TextureDesc) -> TextureDesc {
        *desc
    }

    #[test]
    fn a_steady_frame_reuses_every_texture() {
        let mut pool = TexturePool::default();
        for _ in 0..3 {
            pool.begin_frame();
            pool.acquire(desc(64, 64), make);
            pool.acquire(
                TextureDesc {
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
            TextureDesc {
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
    fn a_resize_drops_the_old_size_after_one_frame() {
        let mut pool = TexturePool::default();
        pool.begin_frame();
        pool.acquire(desc(1280, 720), make);

        pool.begin_frame();
        pool.acquire(desc(1920, 1080), make);
        assert_eq!(pool.created(), 2, "the new size is a new texture");
        assert_eq!(
            pool.len(),
            2,
            "the old one survives until nothing used it for a frame"
        );

        pool.begin_frame();
        assert_eq!(pool.len(), 1, "the old size is gone");
        pool.acquire(desc(1920, 1080), make);
        assert_eq!(pool.created(), 2, "and the new one is reused");
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
