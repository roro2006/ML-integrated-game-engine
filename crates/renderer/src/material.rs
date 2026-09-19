//! Registered materials.
//!
//! A material is currently just a linear base colour: the lit shader has no
//! textures, so there is nothing to bind per material and no GPU resource to
//! own. Materials still go through a registry and a handle, because the handle
//! is the thing the rest of the engine holds, and keeping that boundary in
//! place now means textures can arrive later without changing any caller.

/// The colours a material contributes to its draw's `DrawData` block.
///
/// A trait so the uniform packer can be tested against a stand-in material.
pub(crate) trait BaseColor {
    fn base_color(&self) -> [f32; 4];
    fn flash(&self) -> [f32; 4];
}

/// A registered material.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GpuMaterial {
    /// Linear RGBA tint, copied from the `MaterialDesc` it was created from.
    pub(crate) base_color: [f32; 4],
    /// The flash overlay, likewise copied.
    pub(crate) flash: [f32; 4],
}

impl GpuMaterial {
    pub(crate) fn from_desc(desc: &crate::frame::MaterialDesc) -> Self {
        Self {
            base_color: desc.base_color,
            flash: desc.flash,
        }
    }
}

impl BaseColor for GpuMaterial {
    fn base_color(&self) -> [f32; 4] {
        self.base_color
    }

    fn flash(&self) -> [f32; 4] {
        self.flash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{MaterialDesc, MaterialHandle};
    use crate::registry::Registry;

    #[test]
    fn materials_keep_their_base_colour_through_the_registry() {
        let mut materials: Registry<MaterialHandle, GpuMaterial> = Registry::default();
        let desc = MaterialDesc {
            base_color: [0.2, 0.4, 0.6, 1.0],
            flash: [1.0, 1.0, 1.0, 0.5],
        };
        let handle = materials.insert(GpuMaterial::from_desc(&desc));

        assert_eq!(
            materials.get(handle).map(GpuMaterial::base_color),
            Some([0.2, 0.4, 0.6, 1.0])
        );
        assert_eq!(
            materials.get(handle).map(GpuMaterial::flash),
            Some([1.0, 1.0, 1.0, 0.5])
        );
    }

    #[test]
    fn the_default_material_is_opaque_white() {
        let material = GpuMaterial::from_desc(&MaterialDesc::default());
        assert_eq!(material.base_color(), [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(
            material.flash()[3],
            0.0,
            "and flashes nothing, so every existing material draws as before"
        );
    }
}
