//! Shader permutations for the PBR material.
//!
//! A permutation is a set of features a material switches on, mapped to WGSL
//! `override` constants through the pipeline's compilation options. The
//! compiler folds the branches away, so a material without a normal map pays
//! nothing for the normal-map code, and there is no preprocessor and no new
//! dependency. Every permutation is small enough to build all of them at
//! startup, so a material never waits on a compile and a broken shader is
//! caught before the first frame.

use crate::frame::{AlphaMode, MaterialDesc};

/// The features one PBR pipeline is compiled for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct PermutationKey {
    pub(crate) normal_map: bool,
    pub(crate) emissive: bool,
    pub(crate) alpha_mask: bool,
}

impl PermutationKey {
    /// Every permutation, in a fixed order.
    pub(crate) fn all() -> impl Iterator<Item = Self> {
        (0..8u8).map(|bits| Self {
            normal_map: bits & 1 != 0,
            emissive: bits & 2 != 0,
            alpha_mask: bits & 4 != 0,
        })
    }

    /// The permutation a material needs. Emission is on when it can be seen:
    /// an emissive texture with a factor of zero emits nothing.
    pub(crate) fn for_material(desc: &MaterialDesc) -> Self {
        Self {
            normal_map: desc.textures.normal.is_some(),
            emissive: desc.emissive.iter().any(|&channel| channel > 0.0),
            alpha_mask: matches!(desc.alpha, AlphaMode::Mask { .. }),
        }
    }

    /// The `override` constants for `pbr_lit.wgsl`, as WGSL booleans (0 or 1).
    pub(crate) fn constants(self) -> [(&'static str, f64); 3] {
        let flag = |on: bool| if on { 1.0 } else { 0.0 };
        [
            ("HAS_NORMAL_MAP", flag(self.normal_map)),
            ("HAS_EMISSIVE", flag(self.emissive)),
            ("ALPHA_MASK", flag(self.alpha_mask)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{MaterialTextures, TextureHandle};

    #[test]
    fn there_are_eight_distinct_permutations() {
        let mut keys: Vec<_> = PermutationKey::all().collect();
        assert_eq!(keys.len(), 8);
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), 8);
    }

    #[test]
    fn a_key_maps_to_the_constants_the_shader_declares() {
        let key = PermutationKey {
            normal_map: true,
            emissive: false,
            alpha_mask: true,
        };
        assert_eq!(
            key.constants(),
            [
                ("HAS_NORMAL_MAP", 1.0),
                ("HAS_EMISSIVE", 0.0),
                ("ALPHA_MASK", 1.0)
            ]
        );
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/pbr_lit.wgsl"
        ));
        for (name, _) in key.constants() {
            assert!(
                source.contains(&format!("override {name}: bool")),
                "{name} is not declared"
            );
        }
    }

    #[test]
    fn a_material_asks_only_for_what_it_uses() {
        let plain = PermutationKey::for_material(&MaterialDesc::default());
        assert_eq!(
            plain,
            PermutationKey {
                normal_map: false,
                emissive: false,
                alpha_mask: false,
            }
        );
        let detailed = PermutationKey::for_material(&MaterialDesc {
            textures: MaterialTextures {
                normal: Some(TextureHandle::from_index(0)),
                ..MaterialTextures::default()
            },
            emissive: [0.0, 0.5, 0.0],
            alpha: AlphaMode::Mask { cutoff: 0.5 },
            ..MaterialDesc::default()
        });
        assert_eq!(
            detailed,
            PermutationKey {
                normal_map: true,
                emissive: true,
                alpha_mask: true,
            }
        );
    }

    #[test]
    fn an_emissive_texture_without_a_factor_emits_nothing() {
        let key = PermutationKey::for_material(&MaterialDesc {
            textures: MaterialTextures {
                emissive: Some(TextureHandle::from_index(0)),
                ..MaterialTextures::default()
            },
            ..MaterialDesc::default()
        });
        assert!(!key.emissive);
    }
}
