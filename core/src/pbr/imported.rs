//! Direct conversion of the existing mesh-reader appearance into the canonical material schema.
//!
//! Texture bindings deliberately refer back to `Appearance.images`. Renderer loads retain this
//! canonical conversion and use `Appearance::canonical_shader_words` to resolve those IDs, transform
//! raw canonical UVs once, and pack the decoded RGBA pixels into the renderer's read-only allocation.
//! Canonical consumers use `FaceAppearance.canonical_uv`: those role-resolved source UVs have no
//! affine transform baked in, and the converted `TextureSampling` owns that transform and wrapping.
//! The older `base_uv`/`normal_uv`/`finish_uv` fields are exclusively pre-transformed legacy data.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::{
    ChannelSelector, Color, CoverageMode, Feature, Input, MaterialId, MaterialSpec, Projection,
    TextureColorSpace, TextureId, TextureInput, TextureResource, TextureSampling, TwoSidedPolicy,
    UvTransform, Vector3, WrapMode,
};
use crate::mesh::read::{Appearance, ImportedAlphaMode, ImportedMaterial, MaterialMap};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanonicalImport {
    /// Caller-owned stable asset identity. Material and texture IDs are derived from it.
    pub source_id: String,
    pub materials: Vec<ConvertedImportedMaterial>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConvertedImportedMaterial {
    pub source_index: usize,
    pub spec: MaterialSpec,
    /// Stable canonical-to-source image bindings, resolved by the renderer's canonical projection.
    pub texture_bindings: Vec<ImportedTextureBinding>,
    pub diagnostics: Vec<ImportDiagnostic>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportedTextureBinding {
    pub texture: TextureId,
    /// Index into the original `Appearance.images` vector.
    pub imported_image: usize,
    pub role: ImportedTextureRole,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportedTextureRole {
    BaseColor,
    Normal,
    Height,
    MetallicRoughness,
}

impl ImportedTextureRole {
    fn id_component(self) -> &'static str {
        match self {
            Self::BaseColor => "base-color",
            Self::Normal => "normal",
            Self::Height => "height",
            Self::MetallicRoughness => "metallic-roughness",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportDiagnosticKind {
    UnsupportedExtension,
    UnsupportedFeature,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportDiagnostic {
    pub kind: ImportDiagnosticKind,
    pub extension: Option<String>,
    pub feature: String,
    pub path: String,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportConversionError {
    pub material_index: Option<usize>,
    pub path: String,
    pub message: String,
}

impl ImportConversionError {
    fn new(
        material_index: Option<usize>,
        path: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            material_index,
            path: path.into(),
            message: message.into(),
        }
    }

    fn at_material(index: usize, path: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(Some(index), path, message)
    }
}

impl fmt::Display for ImportConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(index) = self.material_index {
            write!(f, "imported material {index}, ")?;
        }
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl std::error::Error for ImportConversionError {}

/// Convert every material slot. Slot order and caller-supplied asset identity make the generated
/// material IDs stable without depending on a preset position or a transient texture table.
pub fn convert_imported_appearance(
    source_id: &str,
    appearance: &Appearance,
) -> Result<CanonicalImport, ImportConversionError> {
    validate_source_id(source_id, None)?;
    let materials = appearance
        .materials
        .iter()
        .enumerate()
        .map(|(index, material)| {
            convert_imported_material(source_id, index, material, appearance.images.len())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CanonicalImport {
        source_id: source_id.into(),
        materials,
    })
}

/// Convert one existing imported material. Unsupported source features are diagnostics, not partial
/// implementations: their values are not applied to the core metallic-roughness defaults.
pub fn convert_imported_material(
    source_id: &str,
    source_index: usize,
    imported: &ImportedMaterial,
    imported_image_count: usize,
) -> Result<ConvertedImportedMaterial, ImportConversionError> {
    validate_source_id(source_id, Some(source_index))?;
    let id = MaterialId::from(format!("import:{source_id}/material:{source_index}"));
    let mut spec = MaterialSpec {
        id,
        name: imported.name.clone(),
        preset: None,
        // glTF/OBJ metallic-roughness imports are opaque surface materials unless coverage says
        // otherwise. Do not inherit the canonical default material's active clear-glass lobe.
        transvolume: Feature::disabled(),
        ..MaterialSpec::default()
    };
    let base_color = Color([
        f64::from(imported.base[0]),
        f64::from(imported.base[1]),
        f64::from(imported.base[2]),
    ]);
    let alpha = f64::from(imported.base[3]);
    spec.base.values.color = base_color.into();
    spec.base.values.diffuse_weight = 1.0.into();
    spec.base.values.metalness = f64::from(imported.metallic).into();
    spec.specular.values.roughness = f64::from(imported.roughness).into();
    spec.coverage.values.opacity = alpha.into();
    spec.coverage.values.alpha_cutoff = f64::from(imported.alpha_cutoff).into();
    spec.coverage.values.mode = match imported.alpha_mode {
        ImportedAlphaMode::Opaque | ImportedAlphaMode::Mask => CoverageMode::Deterministic,
        ImportedAlphaMode::Blend => CoverageMode::Stochastic,
    };
    spec.coverage.values.front_back = if imported.double_sided {
        TwoSidedPolicy::Both
    } else {
        TwoSidedPolicy::Front
    };
    // Front/back policy lives in this group, so even opaque single-sided materials need it enabled.
    spec.coverage.enabled = true;

    let mut texture_bindings = Vec::new();
    if let Some(map) = imported.colour {
        let texture = add_texture(
            &mut spec,
            &mut texture_bindings,
            source_index,
            imported_image_count,
            ImportedTextureRole::BaseColor,
            map,
            TextureColorSpace::Srgb,
        )?;
        // glTF base-colour RGB is sRGB-encoded while its factor and alpha are linear multipliers.
        spec.base.values.color = Input::Texture(TextureInput {
            texture: texture.clone(),
            fallback: base_color,
            channels: ChannelSelector::Rgb,
            mapping: Some(super::Affine {
                scale: base_color,
                offset: Color::BLACK,
            }),
            remap: None,
        });
        // Alpha is sampled without sRGB decoding, including when an opaque material keeps this
        // source value dormant in its disabled coverage group.
        spec.coverage.values.opacity = Input::Texture(TextureInput {
            texture,
            fallback: alpha,
            channels: ChannelSelector::Alpha,
            mapping: Some(super::Affine {
                scale: alpha,
                offset: 0.0,
            }),
            remap: None,
        });
    }

    // glTF OPAQUE ignores source alpha even though coverage is active for its sidedness policy.
    if imported.alpha_mode == ImportedAlphaMode::Opaque {
        spec.coverage.values.opacity = 1.0.into();
    }

    if let Some(map) = imported.normal {
        spec.geometry.enabled = true;
        if imported.height {
            let texture = add_texture(
                &mut spec,
                &mut texture_bindings,
                source_index,
                imported_image_count,
                ImportedTextureRole::Height,
                map,
                TextureColorSpace::NonColor,
            )?;
            // OBJ -bm is a height-to-bump strength, not a tangent-space normal scale.
            spec.geometry.values.bump.height = Input::Texture(TextureInput {
                texture,
                fallback: 0.0,
                channels: ChannelSelector::Red,
                mapping: None,
                remap: None,
            });
            spec.geometry.values.bump.strength = f64::from(imported.normal_scale).into();
            spec.geometry.values.bump.scale = 1.0.into();
        } else {
            let texture = add_texture(
                &mut spec,
                &mut texture_bindings,
                source_index,
                imported_image_count,
                ImportedTextureRole::Normal,
                map,
                TextureColorSpace::NonColor,
            )?;
            let strength = f64::from(imported.normal_scale);
            // Convert [0,1] tangent normals to [-1,1] and apply glTF strength to X/Y only.
            spec.geometry.values.base_normal = Input::Texture(TextureInput {
                texture,
                fallback: Vector3([0.0, 0.0, 1.0]),
                channels: ChannelSelector::Vector3,
                mapping: Some(super::Affine {
                    scale: Vector3([2.0 * strength, 2.0 * strength, 2.0]),
                    offset: Vector3([-strength, -strength, -1.0]),
                }),
                remap: None,
            });
        }
    }

    if let Some(map) = imported.metallic_roughness {
        let texture = add_texture(
            &mut spec,
            &mut texture_bindings,
            source_index,
            imported_image_count,
            ImportedTextureRole::MetallicRoughness,
            map,
            TextureColorSpace::NonColor,
        )?;
        // One linear packed resource feeds both inputs: G = roughness, B = metalness.
        spec.specular.values.roughness = Input::Texture(TextureInput {
            texture: texture.clone(),
            fallback: f64::from(imported.roughness),
            channels: ChannelSelector::Green,
            mapping: Some(super::Affine {
                scale: f64::from(imported.roughness),
                offset: 0.0,
            }),
            remap: None,
        });
        spec.base.values.metalness = Input::Texture(TextureInput {
            texture,
            fallback: f64::from(imported.metallic),
            channels: ChannelSelector::Blue,
            mapping: Some(super::Affine {
                scale: f64::from(imported.metallic),
                offset: 0.0,
            }),
            remap: None,
        });
    }

    let diagnostics = imported
        .compatibility
        .iter()
        .map(|unsupported| {
            spec.compatibility
                .notices
                .push(format!("{}: {}", unsupported.path, unsupported.reason));
            ImportDiagnostic {
                kind: if unsupported.extension.is_some() {
                    ImportDiagnosticKind::UnsupportedExtension
                } else {
                    ImportDiagnosticKind::UnsupportedFeature
                },
                extension: unsupported.extension.clone(),
                feature: unsupported.feature.clone(),
                path: unsupported.path.clone(),
                message: unsupported.reason.clone(),
            }
        })
        .collect();
    spec.validate().map_err(|error| {
        ImportConversionError::at_material(source_index, error.path, error.kind.to_string())
    })?;
    Ok(ConvertedImportedMaterial {
        source_index,
        spec,
        texture_bindings,
        diagnostics,
    })
}

fn validate_source_id(
    source_id: &str,
    material_index: Option<usize>,
) -> Result<(), ImportConversionError> {
    MaterialId::from(format!("import:{source_id}/material:0"))
        .validate("source_id")
        .map_err(|error| {
            ImportConversionError::new(material_index, error.path, error.kind.to_string())
        })
}

#[allow(clippy::too_many_arguments)]
fn add_texture(
    spec: &mut MaterialSpec,
    bindings: &mut Vec<ImportedTextureBinding>,
    material_index: usize,
    image_count: usize,
    role: ImportedTextureRole,
    map: MaterialMap,
    color_space: TextureColorSpace,
) -> Result<TextureId, ImportConversionError> {
    if map.image >= image_count {
        return Err(ImportConversionError::at_material(
            material_index,
            format!("{}.image", role.id_component()),
            format!(
                "imported image {} does not exist (there are {image_count})",
                map.image
            ),
        ));
    }
    let id = TextureId::from(format!("{}/texture:{}", spec.id, role.id_component()));
    let transform = decode_uv_transform(map.uv_transform).map_err(|message| {
        ImportConversionError::at_material(
            material_index,
            format!("{}.uv_transform", role.id_component()),
            message,
        )
    })?;
    let wrap = |value: u32, axis: &str| match value {
        10497 => Ok(WrapMode::Repeat),
        33071 => Ok(WrapMode::Clamp),
        33648 => Ok(WrapMode::Mirror),
        other => Err(ImportConversionError::at_material(
            material_index,
            format!("{}.wrap.{axis}", role.id_component()),
            format!("glTF sampler value {other} is not a supported wrap mode"),
        )),
    };
    spec.textures.insert(
        id.clone(),
        TextureResource {
            uri: format!("imported-image:{}", map.image),
            color_space,
            sampling: TextureSampling {
                uv_set: format!("uv{}", map.texcoord),
                projection: Projection::Uv,
                transform,
                wrap_u: wrap(map.wrap[0], "u")?,
                wrap_v: wrap(map.wrap[1], "v")?,
                filter: super::FilterMode::LinearMip,
            },
        },
    );
    bindings.push(ImportedTextureBinding {
        texture: id.clone(),
        imported_image: map.image,
        role,
    });
    Ok(id)
}

/// The reader stores the row-major 2x3 result of KHR_texture_transform. The canonical decomposition
/// represents exactly offset + rotation * non-uniform scale; reject shear or other affine matrices
/// instead of silently dropping a term.
fn decode_uv_transform(matrix: [f32; 6]) -> Result<UvTransform, String> {
    let m = matrix.map(f64::from);
    if !m.iter().all(|value| value.is_finite()) {
        return Err("UV transform must be finite".into());
    }
    let scale_x = m[0].hypot(m[3]);
    let (rotation_radians, scale_y) = if scale_x > 0.0 {
        let rotation = m[3].atan2(m[0]);
        let (sin, cos) = rotation.sin_cos();
        (rotation, -sin * m[1] + cos * m[4])
    } else {
        let scale_y = m[1].hypot(m[4]);
        let rotation = (-m[1]).atan2(m[4]);
        (rotation, scale_y)
    };
    let result = UvTransform {
        offset: [m[2], m[5]],
        scale: [scale_x, scale_y],
        rotation_radians,
    };
    let (sin, cos) = result.rotation_radians.sin_cos();
    let reconstructed = [
        cos * result.scale[0],
        -sin * result.scale[1],
        result.offset[0],
        sin * result.scale[0],
        cos * result.scale[1],
        result.offset[1],
    ];
    for (original, decoded) in m.into_iter().zip(reconstructed) {
        if (original - decoded).abs() > 1e-6 * original.abs().max(1.0) {
            return Err(
                "affine UV shear is not representable by offset, rotation, and scale".into(),
            );
        }
    }
    Ok(result)
}
