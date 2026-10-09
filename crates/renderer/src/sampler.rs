//! Samplers, created once per distinct description and shared.
//!
//! Descriptions are normalised before they are looked up, so two that the GPU
//! would treat identically share one sampler: anisotropy is clamped to 1..=16,
//! and forced to 1 for nearest filtering, where wgpu forbids it anyway.

use std::collections::HashMap;

/// How texture coordinates outside 0..1 behave.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Wrap {
    #[default]
    Repeat,
    Clamp,
    Mirror,
}

/// How texels are interpolated, within a level and between levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Filter {
    /// Hard texel edges, for pixel-art and lookup textures.
    Nearest,
    #[default]
    Linear,
}

/// How a material samples one of its textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerDesc {
    pub wrap: Wrap,
    pub filter: Filter,
    /// Anisotropic filtering, 1 (off) to 16. Ignored for nearest filtering.
    pub anisotropy: u8,
}

impl Default for SamplerDesc {
    fn default() -> Self {
        Self {
            wrap: Wrap::Repeat,
            filter: Filter::Linear,
            anisotropy: 1,
        }
    }
}

const MAX_ANISOTROPY: u8 = 16;

impl SamplerDesc {
    /// The description the GPU will actually honour.
    pub(crate) fn normalised(self) -> Self {
        let anisotropy = match self.filter {
            Filter::Nearest => 1,
            Filter::Linear => self.anisotropy.clamp(1, MAX_ANISOTROPY),
        };
        Self { anisotropy, ..self }
    }

    pub(crate) fn to_wgpu(self) -> wgpu::SamplerDescriptor<'static> {
        let address_mode = match self.wrap {
            Wrap::Repeat => wgpu::AddressMode::Repeat,
            Wrap::Clamp => wgpu::AddressMode::ClampToEdge,
            Wrap::Mirror => wgpu::AddressMode::MirrorRepeat,
        };
        let (filter, mipmap_filter) = match self.filter {
            Filter::Nearest => (wgpu::FilterMode::Nearest, wgpu::MipmapFilterMode::Nearest),
            Filter::Linear => (wgpu::FilterMode::Linear, wgpu::MipmapFilterMode::Linear),
        };
        wgpu::SamplerDescriptor {
            label: Some("material-sampler"),
            address_mode_u: address_mode,
            address_mode_v: address_mode,
            address_mode_w: address_mode,
            mag_filter: filter,
            min_filter: filter,
            mipmap_filter,
            anisotropy_clamp: u16::from(self.normalised().anisotropy),
            ..Default::default()
        }
    }
}

/// One sampler per normalised description. Generic over the sampler so the
/// sharing is tested without a device.
#[derive(Debug)]
pub(crate) struct SamplerCache<S> {
    samplers: HashMap<SamplerDesc, S>,
}

impl<S> Default for SamplerCache<S> {
    fn default() -> Self {
        Self {
            samplers: HashMap::new(),
        }
    }
}

impl<S> SamplerCache<S> {
    pub(crate) fn get_or_create(&mut self, desc: SamplerDesc, create: impl FnOnce(SamplerDesc) -> S) -> &S {
        let key = desc.normalised();
        self.samplers.entry(key).or_insert_with(|| create(key))
    }

    pub(crate) fn len(&self) -> usize {
        self.samplers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_descriptions_share_one_sampler() {
        let mut cache = SamplerCache::default();
        let mut created = 0;
        for _ in 0..3 {
            cache.get_or_create(SamplerDesc::default(), |_| {
                created += 1;
                created
            });
        }
        assert_eq!(created, 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn descriptions_the_gpu_treats_alike_share_one_sampler() {
        let mut cache = SamplerCache::default();
        let nearest = |anisotropy| SamplerDesc {
            filter: Filter::Nearest,
            anisotropy,
            ..SamplerDesc::default()
        };
        let first = *cache.get_or_create(nearest(1), |_| 1);
        let second = *cache.get_or_create(nearest(8), |_| 2);
        assert_eq!(first, second, "anisotropy means nothing to nearest filtering");

        let a = *cache.get_or_create(
            SamplerDesc {
                anisotropy: 0,
                ..SamplerDesc::default()
            },
            |_| 3,
        );
        let b = *cache.get_or_create(SamplerDesc::default(), |_| 4);
        assert_eq!(a, b, "anisotropy 0 is read as 1");
    }

    #[test]
    fn different_wraps_or_filters_get_their_own_sampler() {
        let mut cache = SamplerCache::default();
        cache.get_or_create(SamplerDesc::default(), |_| ());
        cache.get_or_create(
            SamplerDesc {
                wrap: Wrap::Clamp,
                ..SamplerDesc::default()
            },
            |_| (),
        );
        cache.get_or_create(
            SamplerDesc {
                anisotropy: 16,
                ..SamplerDesc::default()
            },
            |_| (),
        );
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn anisotropy_is_clamped_to_what_hardware_offers() {
        let wide = SamplerDesc {
            anisotropy: 200,
            ..SamplerDesc::default()
        };
        assert_eq!(wide.normalised().anisotropy, 16);
        assert_eq!(wide.to_wgpu().anisotropy_clamp, 16);
        let nearest = SamplerDesc {
            filter: Filter::Nearest,
            anisotropy: 16,
            ..SamplerDesc::default()
        };
        assert_eq!(nearest.to_wgpu().anisotropy_clamp, 1);
        assert_eq!(nearest.to_wgpu().mipmap_filter, wgpu::MipmapFilterMode::Nearest);
    }
}
