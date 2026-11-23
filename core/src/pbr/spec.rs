use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::pbr::field::{FieldGraph, FieldNode};
use crate::pbr::value::*;
use crate::pbr::MATERIAL_SCHEMA_VERSION;

fn version() -> u32 {
    MATERIAL_SCHEMA_VERSION
}
fn default_material_id() -> MaterialId {
    MaterialId::from("material:default")
}
fn default_medium_id() -> MediumId {
    MediumId::from("medium:default")
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MaterialSpec {
    pub schema_version: u32,
    pub id: MaterialId,
    pub name: String,
    pub preset: Option<PresetId>,
    pub base: Feature<BaseProperties>,
    pub specular: Feature<SpecularProperties>,
    pub coat: Feature<CoatProperties>,
    pub sheen: Feature<SheenProperties>,
    pub film: Feature<FilmProperties>,
    pub transvolume: Feature<TransmissionVolumeProperties>,
    pub subsurface: Feature<SubsurfaceProperties>,
    pub emission: Feature<EmissionProperties>,
    pub coverage: Feature<CoverageProperties>,
    pub geometry: Feature<GeometryProperties>,
    pub wear: Feature<WearProperties>,
    pub special_optics: SpecialOpticsProperties,
    pub fields: FieldGraph,
    pub textures: BTreeMap<TextureId, TextureResource>,
    pub compatibility: CompatibilityMetadata,
}

impl Default for MaterialSpec {
    fn default() -> Self {
        Self {
            schema_version: version(),
            id: default_material_id(),
            name: "Clear glass".into(),
            preset: None,
            base: Feature::enabled(BaseProperties::default()),
            specular: Feature::enabled(SpecularProperties::default()),
            coat: Feature::default(),
            sheen: Feature::default(),
            film: Feature::default(),
            transvolume: Feature::enabled(TransmissionVolumeProperties::default()),
            subsurface: Feature::default(),
            emission: Feature::default(),
            coverage: Feature::default(),
            geometry: Feature::default(),
            wear: Feature::default(),
            special_optics: SpecialOpticsProperties::default(),
            fields: FieldGraph::default(),
            textures: BTreeMap::new(),
            compatibility: CompatibilityMetadata::default(),
        }
    }
}

impl MaterialSpec {
    pub fn validate(&self) -> Result<(), SchemaError> {
        if self.schema_version != MATERIAL_SCHEMA_VERSION {
            return Err(SchemaError::new(
                "schema_version",
                ErrorKind::UnsupportedVersion {
                    found: self.schema_version,
                    supported: MATERIAL_SCHEMA_VERSION,
                },
            ));
        }
        self.id.validate("id")?;
        if let Some(preset) = &self.preset {
            preset.validate("preset")?;
        }
        self.fields.validate()?;
        for (id, texture) in &self.textures {
            id.validate(&format!("textures.{id}"))?;
            texture.validate(&format!("textures.{id}"))?;
        }
        for (id, node) in &self.fields.nodes {
            if let FieldNode::Texture { texture, .. } = node {
                if !self.textures.contains_key(texture) {
                    return Err(SchemaError::new(
                        format!("fields.{id}.texture"),
                        ErrorKind::UnknownTexture(texture.clone()),
                    ));
                }
            }
        }
        let cx = ValidationContext {
            fields: &self.fields,
            textures: &self.textures,
        };
        self.base.validate(&cx, "base")?;
        self.specular.validate(&cx, "specular")?;
        self.coat.validate(&cx, "coat")?;
        self.sheen.validate(&cx, "sheen")?;
        self.film.validate(&cx, "film")?;
        self.transvolume.validate(&cx, "transvolume")?;
        self.subsurface.validate(&cx, "subsurface")?;
        self.emission.validate(&cx, "emission")?;
        self.coverage.validate(&cx, "coverage")?;
        self.geometry.validate(&cx, "geometry")?;
        self.wear.validate(&cx, "wear")?;
        self.special_optics.validate(&cx, "special_optics")?;
        self.compatibility.validate()?;

        scalar_range(&self.base.values.weight, 0.0, 1.0, "base.values.weight")?;
        scalar_range(
            &self.base.values.diffuse_weight,
            0.0,
            1.0,
            "base.values.diffuse_weight",
        )?;
        scalar_range(
            &self.base.values.metalness,
            0.0,
            1.0,
            "base.values.metalness",
        )?;
        scalar_range(
            &self.base.values.diffuse_roughness,
            0.0,
            1.0,
            "base.values.diffuse_roughness",
        )?;
        self.base
            .values
            .conductor
            .validate(&cx, "base.values.conductor")?;
        scalar_range(
            &self.specular.values.weight,
            0.0,
            1.0,
            "specular.values.weight",
        )?;
        scalar_positive(&self.specular.values.ior, "specular.values.ior")?;
        scalar_range(
            &self.specular.values.roughness,
            0.0,
            1.0,
            "specular.values.roughness",
        )?;
        scalar_range(
            &self.specular.values.anisotropy,
            -1.0,
            1.0,
            "specular.values.anisotropy",
        )?;
        scalar_range(&self.coat.values.weight, 0.0, 1.0, "coat.values.weight")?;
        scalar_positive(&self.coat.values.ior, "coat.values.ior")?;
        scalar_positive(
            &self.coat.values.attenuation_distance,
            "coat.values.attenuation_distance",
        )?;
        scalar_range(
            &self.coat.values.roughness,
            0.0,
            1.0,
            "coat.values.roughness",
        )?;
        scalar_range(&self.film.values.weight, 0.0, 1.0, "film.values.weight")?;
        scalar_nonnegative(&self.film.values.thickness_nm, "film.values.thickness_nm")?;
        scalar_positive(&self.film.values.ior, "film.values.ior")?;
        scalar_range(
            &self.transvolume.values.weight,
            0.0,
            1.0,
            "transvolume.values.weight",
        )?;
        scalar_positive(&self.transvolume.values.ior, "transvolume.values.ior")?;
        scalar_positive(
            &self.transvolume.values.absorption_reference_distance,
            "transvolume.values.absorption_reference_distance",
        )?;
        scalar_nonnegative(
            &self.transvolume.values.absorption_scale,
            "transvolume.values.absorption_scale",
        )?;
        scalar_range(
            &self.subsurface.values.weight,
            0.0,
            1.0,
            "subsurface.values.weight",
        )?;
        scalar_positive(
            &self.subsurface.values.mean_free_path,
            "subsurface.values.mean_free_path",
        )?;
        scalar_positive(
            &self.subsurface.values.radius_scale,
            "subsurface.values.radius_scale",
        )?;
        scalar_range(
            &self.subsurface.values.phase_anisotropy,
            -0.95,
            0.95,
            "subsurface.values.phase_anisotropy",
        )?;
        scalar_nonnegative(&self.emission.values.luminance, "emission.values.luminance")?;
        scalar_range(
            &self.coverage.values.opacity,
            0.0,
            1.0,
            "coverage.values.opacity",
        )?;
        scalar_nonnegative(
            &self.coverage.values.alpha_cutoff,
            "coverage.values.alpha_cutoff",
        )?;

        if self.transvolume.enabled
            && matches!(self.transvolume.values.wall, WallMode::ThinWalled { .. })
            && self.subsurface.enabled
            && self
                .subsurface
                .values
                .weight
                .as_constant()
                .is_some_and(|weight| *weight > 0.0)
        {
            return Err(SchemaError::new("subsurface", ErrorKind::Incompatible(
                "thin-walled transmission and volumetric subsurface need an explicit composition mode".into(),
            )));
        }
        let walk = self.subsurface.enabled
            && self.subsurface.values.estimator == ScatteringEstimator::RandomWalk;
        let biref = self.special_optics.birefringence.enabled;
        let grin = self.special_optics.grin.enabled;
        if (walk as u8 + biref as u8 + grin as u8) > 1 {
            return Err(SchemaError::new("special_optics", ErrorKind::Incompatible(
                "random walk, birefringence, and GRIN currently require separate transport variants".into(),
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BaseProperties {
    /// Overall substrate weight, not an additional diffuse lobe. Diffuse, conductor, SSS, and
    /// transmission requests are combined by the future common energy partition, never radiance-summed.
    pub weight: ScalarInput,
    pub color: ColorInput,
    /// Dielectric diffuse request within that partition. Zero for clear glass; one does not imply
    /// an extra contribution on top of transmission. Conductors are selected by `metalness` instead.
    pub diffuse_weight: ScalarInput,
    /// Continuous final-response mixture; categorical conversion never quantises an override.
    pub metalness: ScalarInput,
    pub diffuse_roughness: ScalarInput,
    pub conductor: ConductorModel,
    pub ceramic: Feature<CeramicProperties>,
}

impl Default for BaseProperties {
    fn default() -> Self {
        Self {
            weight: 1.0.into(),
            color: Color::WHITE.into(),
            diffuse_weight: 0.0.into(),
            metalness: 0.0.into(),
            diffuse_roughness: 0.0.into(),
            conductor: ConductorModel::default(),
            ceramic: Feature::default(),
        }
    }
}

impl Validate for BaseProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.weight.validate(cx, &format!("{path}.weight"))?;
        self.diffuse_weight
            .validate(cx, &format!("{path}.diffuse_weight"))?;
        self.color.validate(cx, &format!("{path}.color"))?;
        self.metalness.validate(cx, &format!("{path}.metalness"))?;
        self.diffuse_roughness
            .validate(cx, &format!("{path}.diffuse_roughness"))?;
        self.ceramic.validate(cx, &format!("{path}.ceramic"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "model", content = "values")]
pub enum ConductorModel {
    Ideal {
        tint: ColorInput,
    },
    /// A stable key into Glassforms' measured n,k tables, never an array position.
    Measured {
        id: ConductorId,
    },
    CustomSpectral {
        samples: Vec<ComplexIndexSample>,
    },
    F0F82 {
        f0: ColorInput,
        f82: ColorInput,
    },
}

impl Default for ConductorModel {
    fn default() -> Self {
        Self::Ideal {
            tint: Color::WHITE.into(),
        }
    }
}

impl ConductorModel {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        match self {
            Self::Ideal { tint } => tint.validate(cx, &format!("{path}.tint")),
            Self::Measured { id } => id.validate(&format!("{path}.id")),
            Self::CustomSpectral { samples } => {
                if samples.len() < 2 {
                    return Err(SchemaError::new(
                        path,
                        ErrorKind::InvalidFieldGraph(
                            "custom spectral conductor needs at least two samples".into(),
                        ),
                    ));
                }
                let mut previous = None;
                for (i, sample) in samples.iter().enumerate() {
                    let p = format!("{path}.samples[{i}]");
                    finite(sample.wavelength_nm, &format!("{p}.wavelength_nm"))?;
                    finite(sample.n, &format!("{p}.n"))?;
                    finite(sample.k, &format!("{p}.k"))?;
                    if sample.wavelength_nm <= 0.0 || sample.n < 0.0 || sample.k < 0.0 {
                        return Err(SchemaError::new(
                            p,
                            ErrorKind::InvalidFieldGraph(
                                "wavelength must be positive and n,k non-negative".into(),
                            ),
                        ));
                    }
                    if previous.is_some_and(|nm| sample.wavelength_nm <= nm) {
                        return Err(SchemaError::new(
                            p,
                            ErrorKind::InvalidFieldGraph(
                                "wavelengths must strictly increase".into(),
                            ),
                        ));
                    }
                    previous = Some(sample.wavelength_nm);
                }
                Ok(())
            }
            Self::F0F82 { f0, f82 } => {
                f0.validate(cx, &format!("{path}.f0"))?;
                f82.validate(cx, &format!("{path}.f82"))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComplexIndexSample {
    pub wavelength_nm: f64,
    pub n: f64,
    pub k: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CeramicProperties {
    pub transmitted_fraction: ScalarInput,
}
impl Default for CeramicProperties {
    fn default() -> Self {
        Self {
            transmitted_fraction: 0.15.into(),
        }
    }
}
impl Validate for CeramicProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.transmitted_fraction
            .validate(cx, &format!("{path}.transmitted_fraction"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpecularProperties {
    pub weight: ScalarInput,
    pub color: ColorInput,
    pub ior: ScalarInput,
    /// Perceptual roughness. Legacy conversion records the old alpha = roughness² convention.
    pub roughness: ScalarInput,
    pub anisotropy: ScalarInput,
    pub rotation_radians: ScalarInput,
    pub tangent: Option<Vector3Input>,
}

impl Default for SpecularProperties {
    fn default() -> Self {
        Self {
            weight: 1.0.into(),
            color: Color::WHITE.into(),
            ior: 1.5.into(),
            roughness: 0.0.into(),
            anisotropy: 0.0.into(),
            rotation_radians: 0.0.into(),
            tangent: None,
        }
    }
}

impl Validate for SpecularProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.weight.validate(cx, &format!("{path}.weight"))?;
        self.color.validate(cx, &format!("{path}.color"))?;
        self.ior.validate(cx, &format!("{path}.ior"))?;
        self.roughness.validate(cx, &format!("{path}.roughness"))?;
        self.anisotropy
            .validate(cx, &format!("{path}.anisotropy"))?;
        self.rotation_radians
            .validate(cx, &format!("{path}.rotation_radians"))?;
        if let Some(tangent) = &self.tangent {
            tangent.validate(cx, &format!("{path}.tangent"))?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CoatProperties {
    pub weight: ScalarInput,
    pub absorption_color: ColorInput,
    pub attenuation_distance: ScalarInput,
    pub ior: ScalarInput,
    pub roughness: ScalarInput,
    pub anisotropy: ScalarInput,
    pub rotation_radians: ScalarInput,
    pub internal_returns: bool,
}

impl Default for CoatProperties {
    fn default() -> Self {
        Self {
            weight: 0.0.into(),
            absorption_color: Color::WHITE.into(),
            attenuation_distance: 1.0.into(),
            ior: 1.5.into(),
            roughness: 0.0.into(),
            anisotropy: 0.0.into(),
            rotation_radians: 0.0.into(),
            internal_returns: true,
        }
    }
}

impl Validate for CoatProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.weight.validate(cx, &format!("{path}.weight"))?;
        self.absorption_color
            .validate(cx, &format!("{path}.absorption_color"))?;
        self.attenuation_distance
            .validate(cx, &format!("{path}.attenuation_distance"))?;
        self.ior.validate(cx, &format!("{path}.ior"))?;
        self.roughness.validate(cx, &format!("{path}.roughness"))?;
        self.anisotropy
            .validate(cx, &format!("{path}.anisotropy"))?;
        self.rotation_radians
            .validate(cx, &format!("{path}.rotation_radians"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SheenProperties {
    pub weight: ScalarInput,
    pub color: ColorInput,
    pub roughness: ScalarInput,
    pub model: SheenModel,
    pub thin_transmission: ThinTransmission,
}

impl Default for SheenProperties {
    fn default() -> Self {
        Self {
            weight: 0.0.into(),
            color: Color::WHITE.into(),
            roughness: 0.3.into(),
            model: SheenModel::Charlie,
            thin_transmission: ThinTransmission::default(),
        }
    }
}

impl Validate for SheenProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.weight.validate(cx, &format!("{path}.weight"))?;
        self.color.validate(cx, &format!("{path}.color"))?;
        self.roughness.validate(cx, &format!("{path}.roughness"))?;
        self.thin_transmission
            .validate(cx, &format!("{path}.thin_transmission"))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SheenModel {
    #[default]
    Charlie,
    Ashikhmin,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ThinTransmission {
    pub weight: ScalarInput,
    pub color: ColorInput,
    pub roughness: ScalarInput,
    pub two_sided: TwoSidedPolicy,
}

impl Default for ThinTransmission {
    fn default() -> Self {
        Self {
            weight: 0.0.into(),
            color: Color::WHITE.into(),
            roughness: 0.0.into(),
            two_sided: TwoSidedPolicy::Both,
        }
    }
}

impl ThinTransmission {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.weight.validate(cx, &format!("{path}.weight"))?;
        self.color.validate(cx, &format!("{path}.color"))?;
        self.roughness.validate(cx, &format!("{path}.roughness"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FilmProperties {
    pub weight: ScalarInput,
    pub thickness_nm: ScalarInput,
    pub ior: ScalarInput,
    pub thickness_variation: ScalarInput,
    pub pattern_scale: ScalarInput,
    pub flow: ScalarInput,
    pub drainage: ScalarInput,
    pub ambient_weight: ScalarInput,
    pub placement: FilmPlacement,
    pub filtering: FilmFiltering,
    pub flakes: Feature<FlakeProperties>,
}

impl Default for FilmProperties {
    fn default() -> Self {
        Self {
            weight: 0.0.into(),
            thickness_nm: 0.0.into(),
            ior: 1.33.into(),
            thickness_variation: 0.0.into(),
            pattern_scale: 1.0.into(),
            flow: 0.0.into(),
            drainage: 0.0.into(),
            ambient_weight: 0.5.into(),
            placement: FilmPlacement::OverCoat,
            filtering: FilmFiltering::RgbIntegrated,
            flakes: Feature::default(),
        }
    }
}

impl Validate for FilmProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.weight.validate(cx, &format!("{path}.weight"))?;
        self.thickness_nm
            .validate(cx, &format!("{path}.thickness_nm"))?;
        self.ior.validate(cx, &format!("{path}.ior"))?;
        self.thickness_variation
            .validate(cx, &format!("{path}.thickness_variation"))?;
        self.pattern_scale
            .validate(cx, &format!("{path}.pattern_scale"))?;
        self.flow.validate(cx, &format!("{path}.flow"))?;
        self.drainage.validate(cx, &format!("{path}.drainage"))?;
        self.ambient_weight
            .validate(cx, &format!("{path}.ambient_weight"))?;
        self.flakes.validate(cx, &format!("{path}.flakes"))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilmPlacement {
    OverSubstrate,
    #[default]
    OverCoat,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilmFiltering {
    #[default]
    RgbIntegrated,
    SpectralOnly,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FlakeProperties {
    pub density: ScalarInput,
    pub size: ScalarInput,
    pub color: ColorInput,
    pub roughness: ScalarInput,
    pub orientation_spread: ScalarInput,
    pub coverage: ScalarInput,
}

impl Default for FlakeProperties {
    fn default() -> Self {
        Self {
            density: 0.0.into(),
            size: 0.01.into(),
            color: Color::WHITE.into(),
            roughness: 0.2.into(),
            orientation_spread: 0.5.into(),
            coverage: 0.0.into(),
        }
    }
}

impl Validate for FlakeProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.density.validate(cx, &format!("{path}.density"))?;
        self.size.validate(cx, &format!("{path}.size"))?;
        self.color.validate(cx, &format!("{path}.color"))?;
        self.roughness.validate(cx, &format!("{path}.roughness"))?;
        self.orientation_spread
            .validate(cx, &format!("{path}.orientation_spread"))?;
        self.coverage.validate(cx, &format!("{path}.coverage"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TransmissionVolumeProperties {
    pub weight: ScalarInput,
    pub color: ColorInput,
    pub absorption_coefficients: ColorInput,
    pub absorption_scale: ScalarInput,
    pub absorption_reference_distance: ScalarInput,
    pub measured_spectrum: Option<String>,
    pub ior: ScalarInput,
    pub dispersion: DispersionModel,
    pub rough_estimator: RoughTransmissionEstimator,
    pub wall: WallMode,
    pub medium: MediumDescriptor,
}

impl Default for TransmissionVolumeProperties {
    fn default() -> Self {
        Self {
            weight: 1.0.into(),
            color: Color::WHITE.into(),
            absorption_coefficients: Color::BLACK.into(),
            absorption_scale: 1.0.into(),
            absorption_reference_distance: 1.0.into(),
            measured_spectrum: None,
            ior: 1.5.into(),
            dispersion: DispersionModel::Cauchy {
                amount: 0.08.into(),
            },
            rough_estimator: RoughTransmissionEstimator::Fast,
            wall: WallMode::ClosedVolume,
            medium: MediumDescriptor::default(),
        }
    }
}

impl Validate for TransmissionVolumeProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.weight.validate(cx, &format!("{path}.weight"))?;
        self.color.validate(cx, &format!("{path}.color"))?;
        self.absorption_coefficients
            .validate(cx, &format!("{path}.absorption_coefficients"))?;
        self.absorption_scale
            .validate(cx, &format!("{path}.absorption_scale"))?;
        self.absorption_reference_distance
            .validate(cx, &format!("{path}.absorption_reference_distance"))?;
        self.ior.validate(cx, &format!("{path}.ior"))?;
        if let DispersionModel::Cauchy { amount }
        | DispersionModel::SpectralAbbe {
            cauchy_amount: amount,
            ..
        } = &self.dispersion
        {
            amount.validate(cx, &format!("{path}.dispersion.amount"))?;
        }
        if let DispersionModel::SpectralAbbe { abbe_number, .. } = &self.dispersion {
            abbe_number.validate(cx, &format!("{path}.dispersion.abbe_number"))?;
            scalar_positive(abbe_number, &format!("{path}.dispersion.abbe_number"))?;
        }
        if let RoughTransmissionEstimator::Hybrid { sampled_share } = self.rough_estimator {
            range(
                sampled_share,
                0.0,
                1.0,
                &format!("{path}.rough_estimator.sampled_share"),
            )?;
        }
        if let WallMode::ThinWalled { thickness } = &self.wall {
            thickness.validate(cx, &format!("{path}.wall.thickness"))?;
            scalar_positive(thickness, &format!("{path}.wall.thickness"))?;
        }
        self.medium.id.validate(&format!("{path}.medium.id"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "model", content = "values")]
pub enum DispersionModel {
    None,
    Cauchy {
        amount: ScalarInput,
    },
    SpectralAbbe {
        cauchy_amount: ScalarInput,
        abbe_number: ScalarInput,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode", content = "share")]
pub enum RoughTransmissionEstimator {
    #[default]
    Fast,
    Sampled,
    Hybrid {
        sampled_share: f64,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode", content = "values")]
pub enum WallMode {
    ClosedVolume,
    ThinWalled { thickness: ScalarInput },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MediumDescriptor {
    pub id: MediumId,
    pub interface_priority: i32,
}

impl Default for MediumDescriptor {
    fn default() -> Self {
        Self {
            id: default_medium_id(),
            interface_priority: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SubsurfaceProperties {
    pub weight: ScalarInput,
    pub target_color: ColorInput,
    pub scatter_color: ColorInput,
    pub radii: ColorInput,
    pub radius_scale: ScalarInput,
    pub mean_free_path: ScalarInput,
    pub absorption_coefficients: ColorInput,
    pub phase_anisotropy: ScalarInput,
    pub estimator: ScatteringEstimator,
    pub pigment: LegacyPigmentResponse,
}

impl Default for SubsurfaceProperties {
    fn default() -> Self {
        Self {
            weight: 0.0.into(),
            target_color: Color::WHITE.into(),
            scatter_color: Color::WHITE.into(),
            radii: Color([1.0, 0.5, 0.25]).into(),
            radius_scale: 1.0.into(),
            mean_free_path: 0.2.into(),
            absorption_coefficients: Color::BLACK.into(),
            phase_anisotropy: 0.0.into(),
            estimator: ScatteringEstimator::FastVolume,
            pigment: LegacyPigmentResponse::default(),
        }
    }
}

impl Validate for SubsurfaceProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.weight.validate(cx, &format!("{path}.weight"))?;
        self.target_color
            .validate(cx, &format!("{path}.target_color"))?;
        self.scatter_color
            .validate(cx, &format!("{path}.scatter_color"))?;
        self.radii.validate(cx, &format!("{path}.radii"))?;
        self.radius_scale
            .validate(cx, &format!("{path}.radius_scale"))?;
        self.mean_free_path
            .validate(cx, &format!("{path}.mean_free_path"))?;
        self.absorption_coefficients
            .validate(cx, &format!("{path}.absorption_coefficients"))?;
        self.phase_anisotropy
            .validate(cx, &format!("{path}.phase_anisotropy"))?;
        self.pigment.validate(cx, &format!("{path}.pigment"))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScatteringEstimator {
    #[default]
    FastVolume,
    RandomWalk,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LegacyPigmentResponse {
    pub spectral_spread: ScalarInput,
    pub near_surface_hue: ScalarInput,
    pub unscattered_hue: ScalarInput,
    pub opaque_exit: ScalarInput,
}

impl Default for LegacyPigmentResponse {
    fn default() -> Self {
        Self {
            spectral_spread: 0.0.into(),
            near_surface_hue: 0.0.into(),
            unscattered_hue: 0.0.into(),
            opaque_exit: 0.0.into(),
        }
    }
}

impl LegacyPigmentResponse {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.spectral_spread
            .validate(cx, &format!("{path}.spectral_spread"))?;
        self.near_surface_hue
            .validate(cx, &format!("{path}.near_surface_hue"))?;
        self.unscattered_hue
            .validate(cx, &format!("{path}.unscattered_hue"))?;
        self.opaque_exit
            .validate(cx, &format!("{path}.opaque_exit"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EmissionProperties {
    pub color: ColorInput,
    pub luminance: ScalarInput,
    pub blackbody: Option<BlackbodyEmission>,
    pub volume: VolumeEmission,
}

impl Default for EmissionProperties {
    fn default() -> Self {
        Self {
            color: Color([1.0, 0.32, 0.08]).into(),
            luminance: 0.0.into(),
            blackbody: None,
            volume: VolumeEmission::default(),
        }
    }
}

impl Validate for EmissionProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.color.validate(cx, &format!("{path}.color"))?;
        self.luminance.validate(cx, &format!("{path}.luminance"))?;
        if let Some(blackbody) = &self.blackbody {
            blackbody.validate(cx, &format!("{path}.blackbody"))?;
        }
        self.volume.validate(cx, &format!("{path}.volume"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlackbodyEmission {
    pub temperature_kelvin: ScalarInput,
    pub strength: ScalarInput,
}

impl BlackbodyEmission {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.temperature_kelvin
            .validate(cx, &format!("{path}.temperature_kelvin"))?;
        scalar_positive(
            &self.temperature_kelvin,
            &format!("{path}.temperature_kelvin"),
        )?;
        self.strength.validate(cx, &format!("{path}.strength"))?;
        scalar_nonnegative(&self.strength, &format!("{path}.strength"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VolumeEmission {
    pub strength: ScalarInput,
    pub core_concentration: ScalarInput,
    pub distance_fitted: bool,
}

impl Default for VolumeEmission {
    fn default() -> Self {
        Self {
            strength: 0.0.into(),
            core_concentration: 0.5.into(),
            distance_fitted: false,
        }
    }
}

impl VolumeEmission {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.strength.validate(cx, &format!("{path}.strength"))?;
        self.core_concentration
            .validate(cx, &format!("{path}.core_concentration"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CoverageProperties {
    pub opacity: ScalarInput,
    pub alpha_cutoff: ScalarInput,
    pub mode: CoverageMode,
    pub front_back: TwoSidedPolicy,
    pub transparent_shadows: bool,
}

impl Default for CoverageProperties {
    fn default() -> Self {
        Self {
            opacity: 1.0.into(),
            alpha_cutoff: 0.5.into(),
            mode: CoverageMode::Stochastic,
            front_back: TwoSidedPolicy::Both,
            transparent_shadows: true,
        }
    }
}

impl Validate for CoverageProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.opacity.validate(cx, &format!("{path}.opacity"))?;
        self.alpha_cutoff
            .validate(cx, &format!("{path}.alpha_cutoff"))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageMode {
    Deterministic,
    #[default]
    Stochastic,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TwoSidedPolicy {
    Front,
    Back,
    #[default]
    Both,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GeometryProperties {
    pub base_normal: Vector3Input,
    pub coat_normal: Vector3Input,
    pub bump: HeightBump,
    pub displacement: Displacement,
}

impl Default for GeometryProperties {
    fn default() -> Self {
        Self {
            base_normal: Vector3([0.0, 0.0, 1.0]).into(),
            coat_normal: Vector3([0.0, 0.0, 1.0]).into(),
            bump: HeightBump::default(),
            displacement: Displacement::default(),
        }
    }
}

impl Validate for GeometryProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.base_normal
            .validate(cx, &format!("{path}.base_normal"))?;
        self.coat_normal
            .validate(cx, &format!("{path}.coat_normal"))?;
        self.bump.validate(cx, &format!("{path}.bump"))?;
        self.displacement
            .validate(cx, &format!("{path}.displacement"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HeightBump {
    pub height: ScalarInput,
    pub strength: ScalarInput,
    pub scale: ScalarInput,
}
impl Default for HeightBump {
    fn default() -> Self {
        Self {
            height: 0.0.into(),
            strength: 0.0.into(),
            scale: 1.0.into(),
        }
    }
}
impl HeightBump {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.height.validate(cx, &format!("{path}.height"))?;
        self.strength.validate(cx, &format!("{path}.strength"))?;
        self.scale.validate(cx, &format!("{path}.scale"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Displacement {
    pub height: ScalarInput,
    pub midpoint: ScalarInput,
    /// Storage only: the geometry/SDF integrator must perform and price the rebuild.
    pub rebuild_required: bool,
}
impl Default for Displacement {
    fn default() -> Self {
        Self {
            height: 0.0.into(),
            midpoint: 0.5.into(),
            rebuild_required: false,
        }
    }
}
impl Displacement {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.height.validate(cx, &format!("{path}.height"))?;
        self.midpoint.validate(cx, &format!("{path}.midpoint"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WearProperties {
    pub age: ScalarInput,
    pub edge_wear: ScalarInput,
    pub cavity: ScalarInput,
    pub cavity_color: ColorInput,
    pub scratches: ScalarInput,
    pub swirls: ScalarInput,
    pub orange_peel: ScalarInput,
    pub smudges: ScalarInput,
    pub dust: ScalarInput,
    pub brushing: ScalarInput,
    pub brushing_angle_radians: ScalarInput,
    pub mottle: ScalarInput,
}

impl Default for WearProperties {
    fn default() -> Self {
        Self {
            age: 1.0.into(),
            edge_wear: 0.0.into(),
            cavity: 0.0.into(),
            cavity_color: Color([0.35, 0.27, 0.2]).into(),
            scratches: 0.0.into(),
            swirls: 0.0.into(),
            orange_peel: 0.0.into(),
            smudges: 0.0.into(),
            dust: 0.0.into(),
            brushing: 0.0.into(),
            brushing_angle_radians: 0.0.into(),
            mottle: 0.0.into(),
        }
    }
}

impl Validate for WearProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        macro_rules! inputs { ($($field:ident),+ $(,)?) => {$( self.$field.validate(cx, &format!("{path}.{}", stringify!($field)))?; )+}; }
        inputs!(
            age,
            edge_wear,
            cavity,
            cavity_color,
            scratches,
            swirls,
            orange_peel,
            smudges,
            dust,
            brushing,
            brushing_angle_radians,
            mottle
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpecialOpticsProperties {
    pub grating: Feature<GratingProperties>,
    pub birefringence: Feature<BirefringenceProperties>,
    pub grin: Feature<GrinProperties>,
    pub bubbles: Feature<BubbleProperties>,
    pub gradient: Feature<GradientProperties>,
    pub cracks: Feature<CrackProperties>,
    pub photonic: Feature<PhotonicProperties>,
    pub silk: Feature<SilkProperties>,
    pub pressure: Feature<PressureProperties>,
}

impl SpecialOpticsProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.grating.validate(cx, &format!("{path}.grating"))?;
        self.birefringence
            .validate(cx, &format!("{path}.birefringence"))?;
        self.grin.validate(cx, &format!("{path}.grin"))?;
        self.bubbles.validate(cx, &format!("{path}.bubbles"))?;
        self.gradient.validate(cx, &format!("{path}.gradient"))?;
        self.cracks.validate(cx, &format!("{path}.cracks"))?;
        self.photonic.validate(cx, &format!("{path}.photonic"))?;
        self.silk.validate(cx, &format!("{path}.silk"))?;
        self.pressure.validate(cx, &format!("{path}.pressure"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GratingProperties {
    pub strength: ScalarInput,
    pub pitch_nm: ScalarInput,
    pub angle_radians: ScalarInput,
    pub radial: ScalarInput,
    pub orders: GratingOrders,
}
impl Default for GratingProperties {
    fn default() -> Self {
        Self {
            strength: 0.0.into(),
            pitch_nm: 1600.0.into(),
            angle_radians: 0.0.into(),
            radial: 0.0.into(),
            orders: GratingOrders::SymmetricFirst,
        }
    }
}
impl Validate for GratingProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.strength.validate(cx, &format!("{path}.strength"))?;
        self.pitch_nm.validate(cx, &format!("{path}.pitch_nm"))?;
        self.angle_radians
            .validate(cx, &format!("{path}.angle_radians"))?;
        self.radial.validate(cx, &format!("{path}.radial"))
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GratingOrders {
    #[default]
    SymmetricFirst,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BirefringenceProperties {
    pub delta_ior: ScalarInput,
    pub azimuth_radians: ScalarInput,
    pub elevation_radians: ScalarInput,
}
impl Default for BirefringenceProperties {
    fn default() -> Self {
        Self {
            delta_ior: 0.0.into(),
            azimuth_radians: 0.0.into(),
            elevation_radians: std::f64::consts::FRAC_PI_2.into(),
        }
    }
}
impl Validate for BirefringenceProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.delta_ior.validate(cx, &format!("{path}.delta_ior"))?;
        self.azimuth_radians
            .validate(cx, &format!("{path}.azimuth_radians"))?;
        self.elevation_radians
            .validate(cx, &format!("{path}.elevation_radians"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GrinProperties {
    pub strength: ScalarInput,
    pub depth: ScalarInput,
}
impl Default for GrinProperties {
    fn default() -> Self {
        Self {
            strength: 0.0.into(),
            depth: 0.02.into(),
        }
    }
}
impl Validate for GrinProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.strength.validate(cx, &format!("{path}.strength"))?;
        self.depth.validate(cx, &format!("{path}.depth"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BubbleProperties {
    pub amount: ScalarInput,
    pub size: ScalarInput,
    pub evenness: ScalarInput,
    pub shape: ScalarInput,
    pub seed: ScalarInput,
}
impl Default for BubbleProperties {
    fn default() -> Self {
        Self {
            amount: 0.0.into(),
            size: 0.03.into(),
            evenness: 0.5.into(),
            shape: 0.0.into(),
            seed: 1.0.into(),
        }
    }
}
impl Validate for BubbleProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.amount.validate(cx, &format!("{path}.amount"))?;
        self.size.validate(cx, &format!("{path}.size"))?;
        self.evenness.validate(cx, &format!("{path}.evenness"))?;
        self.shape.validate(cx, &format!("{path}.shape"))?;
        self.seed.validate(cx, &format!("{path}.seed"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GradientProperties {
    pub amount: ScalarInput,
    pub mode: GradientMode,
    pub target_tint: ColorInput,
    pub azimuth_radians: ScalarInput,
    pub elevation_radians: ScalarInput,
    pub offset: ScalarInput,
    pub softness: ScalarInput,
    pub bands: ScalarInput,
    pub wave: ScalarInput,
}
impl Default for GradientProperties {
    fn default() -> Self {
        Self {
            amount: 0.0.into(),
            mode: GradientMode::Axial,
            target_tint: Color([0.55, 0.85, 1.0]).into(),
            azimuth_radians: 0.0.into(),
            elevation_radians: std::f64::consts::FRAC_PI_2.into(),
            offset: 0.0.into(),
            softness: 0.6.into(),
            bands: 6.0.into(),
            wave: 0.3.into(),
        }
    }
}
impl Validate for GradientProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.amount.validate(cx, &format!("{path}.amount"))?;
        self.target_tint
            .validate(cx, &format!("{path}.target_tint"))?;
        self.azimuth_radians
            .validate(cx, &format!("{path}.azimuth_radians"))?;
        self.elevation_radians
            .validate(cx, &format!("{path}.elevation_radians"))?;
        self.offset.validate(cx, &format!("{path}.offset"))?;
        self.softness.validate(cx, &format!("{path}.softness"))?;
        self.bands.validate(cx, &format!("{path}.bands"))?;
        self.wave.validate(cx, &format!("{path}.wave"))
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GradientMode {
    #[default]
    Axial,
    Radial,
    Bands,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CrackProperties {
    pub amount: ScalarInput,
    pub fractures: ScalarInput,
    pub size: ScalarInput,
    pub depth: ScalarInput,
    pub openness: ScalarInput,
    pub growth: ScalarInput,
    pub curvature: ScalarInput,
    pub roughness: ScalarInput,
    pub seed: ScalarInput,
}
impl Default for CrackProperties {
    fn default() -> Self {
        Self {
            amount: 0.0.into(),
            fractures: 0.0.into(),
            size: 0.25.into(),
            depth: 1.0.into(),
            openness: 0.8.into(),
            growth: 1.0.into(),
            curvature: 0.5.into(),
            roughness: 0.0.into(),
            seed: 1.0.into(),
        }
    }
}
impl Validate for CrackProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        macro_rules! inputs { ($($field:ident),+ $(,)?) => {$( self.$field.validate(cx, &format!("{path}.{}", stringify!($field)))?; )+}; }
        inputs!(amount, fractures, size, depth, openness, growth, curvature, roughness, seed);
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PhotonicProperties {
    pub amount: ScalarInput,
    pub size: ScalarInput,
    pub pattern: PhotonicPattern,
    pub longest_wavelength: ScalarInput,
    pub softness: ScalarInput,
    pub roll: ScalarInput,
    pub seed: ScalarInput,
    pub platelet_color: ColorInput,
}
impl Default for PhotonicProperties {
    fn default() -> Self {
        Self {
            amount: 0.0.into(),
            size: 0.12.into(),
            pattern: PhotonicPattern::Blocks,
            longest_wavelength: 0.8.into(),
            softness: 0.3.into(),
            roll: 0.0.into(),
            seed: 1.0.into(),
            platelet_color: Color([1.0, 0.85, 0.6]).into(),
        }
    }
}
impl Validate for PhotonicProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.amount.validate(cx, &format!("{path}.amount"))?;
        self.size.validate(cx, &format!("{path}.size"))?;
        self.longest_wavelength
            .validate(cx, &format!("{path}.longest_wavelength"))?;
        self.softness.validate(cx, &format!("{path}.softness"))?;
        self.roll.validate(cx, &format!("{path}.roll"))?;
        self.seed.validate(cx, &format!("{path}.seed"))?;
        self.platelet_color
            .validate(cx, &format!("{path}.platelet_color"))
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhotonicPattern {
    #[default]
    Blocks,
    Pinfire,
    BroadFlash,
    Platelets,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SilkProperties {
    pub amount: ScalarInput,
    pub azimuth_radians: ScalarInput,
    pub elevation_radians: ScalarInput,
    pub softness: ScalarInput,
    pub wave: ScalarInput,
}
impl Default for SilkProperties {
    fn default() -> Self {
        Self {
            amount: 0.0.into(),
            azimuth_radians: 0.0.into(),
            elevation_radians: 0.0.into(),
            softness: 0.3.into(),
            wave: 0.0.into(),
        }
    }
}
impl Validate for SilkProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.amount.validate(cx, &format!("{path}.amount"))?;
        self.azimuth_radians
            .validate(cx, &format!("{path}.azimuth_radians"))?;
        self.elevation_radians
            .validate(cx, &format!("{path}.elevation_radians"))?;
        self.softness.validate(cx, &format!("{path}.softness"))?;
        self.wave.validate(cx, &format!("{path}.wave"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PressureProperties {
    pub amount: ScalarInput,
    pub color_bias: ScalarInput,
}
impl Default for PressureProperties {
    fn default() -> Self {
        Self {
            amount: 0.0.into(),
            color_bias: 0.5.into(),
        }
    }
}
impl Validate for PressureProperties {
    fn validate(&self, cx: &ValidationContext<'_>, path: &str) -> Result<(), SchemaError> {
        self.amount.validate(cx, &format!("{path}.amount"))?;
        self.color_bias.validate(cx, &format!("{path}.color_bias"))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CompatibilityMetadata {
    pub legacy: Option<LegacyCompatibility>,
    pub notices: Vec<String>,
}

impl CompatibilityMetadata {
    fn validate(&self) -> Result<(), SchemaError> {
        if let Some(legacy) = &self.legacy {
            finite(
                legacy.categorical_metal,
                "compatibility.legacy.categorical_metal",
            )?;
            finite(
                legacy.rough_sample_share,
                "compatibility.legacy.rough_sample_share",
            )?;
            if let Some(index) = legacy.preset_index {
                finite(index, "compatibility.legacy.preset_index")?;
                if index < 0.0 || index.fract() != 0.0 {
                    return Err(SchemaError::new(
                        "compatibility.legacy.preset_index",
                        ErrorKind::InvalidLegacySetting {
                            name: "mat".into(),
                            reason: "preset index must be a non-negative integer".into(),
                        },
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LegacyCompatibility {
    pub preset_index: Option<f64>,
    pub categorical_metal: f64,
    pub roughness: LegacyRoughnessConvention,
    pub coat_roughness: LegacyCoatRoughnessConvention,
    pub rough_sample_share: f64,
    pub fit_distances_to_shape: bool,
    pub sharp_environment: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LegacyRoughnessConvention {
    #[default]
    PerceptualSquaredGgx,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LegacyCoatRoughnessConvention {
    SubstrateRoughness,
    #[default]
    GlazeHazeIsGgxAlpha,
}

/// A complete body owns a `MaterialSpec`; a construction node owns only a patch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "ownership", content = "material")]
pub enum MaterialRecipe {
    Full(MaterialSpec),
    Sparse(MaterialPatch),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialOwnership {
    FullBody,
    SparseNode,
}

impl MaterialRecipe {
    pub fn ownership(&self) -> MaterialOwnership {
        match self {
            Self::Full(_) => MaterialOwnership::FullBody,
            Self::Sparse(_) => MaterialOwnership::SparseNode,
        }
    }

    /// Resolve one effective spec. Full recipes never read `inherited`; sparse recipes require its stable ID.
    pub fn resolve(&self, inherited: Option<&MaterialSpec>) -> Result<MaterialSpec, SchemaError> {
        let resolved = match self {
            Self::Full(full) => full.clone(),
            Self::Sparse(patch) => {
                let base = inherited.ok_or_else(|| {
                    SchemaError::new("base", ErrorKind::MissingBase(patch.base.clone()))
                })?;
                patch.apply(base)?
            }
        };
        resolved.validate()?;
        Ok(resolved)
    }

    pub fn validate(&self) -> Result<(), SchemaError> {
        match self {
            Self::Full(full) => full.validate(),
            Self::Sparse(patch) => patch.validate(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MaterialPatch {
    pub schema_version: u32,
    pub base: MaterialId,
    /// Values not listed here are inherited, including disabled groups' dormant payloads.
    pub properties: BTreeMap<PropertyPath, PropertyValue>,
    pub features: BTreeMap<FeaturePath, bool>,
}

impl Default for MaterialPatch {
    fn default() -> Self {
        Self {
            schema_version: version(),
            base: default_material_id(),
            properties: BTreeMap::new(),
            features: BTreeMap::new(),
        }
    }
}

impl MaterialPatch {
    pub fn new(base: MaterialId) -> Self {
        Self {
            base,
            ..Self::default()
        }
    }
    pub fn is_empty(&self) -> bool {
        self.properties.is_empty() && self.features.is_empty()
    }

    pub fn validate(&self) -> Result<(), SchemaError> {
        if self.schema_version != MATERIAL_SCHEMA_VERSION {
            return Err(SchemaError::new(
                "schema_version",
                ErrorKind::UnsupportedVersion {
                    found: self.schema_version,
                    supported: MATERIAL_SCHEMA_VERSION,
                },
            ));
        }
        self.base.validate("base")?;
        Ok(())
    }

    pub fn apply(&self, inherited: &MaterialSpec) -> Result<MaterialSpec, SchemaError> {
        self.validate()?;
        if inherited.id != self.base {
            return Err(SchemaError::new(
                "base",
                ErrorKind::BaseIdentity {
                    expected: self.base.clone(),
                    actual: inherited.id.clone(),
                },
            ));
        }
        let mut result = inherited.clone();
        for (path, value) in &self.properties {
            set_property(&mut result, path.as_str(), value)?;
        }
        for (path, enabled) in &self.features {
            set_feature(&mut result, path.as_str(), *enabled)?;
        }
        Ok(result)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PropertyPath(pub String);
impl PropertyPath {
    pub fn new(path: impl Into<String>) -> Self {
        Self(path.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl From<&str> for PropertyPath {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}
impl fmt::Display for PropertyPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FeaturePath(pub String);
impl FeaturePath {
    pub fn new(path: impl Into<String>) -> Self {
        Self(path.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl From<&str> for FeaturePath {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "value")]
pub enum PropertyValue {
    Scalar(ScalarInput),
    Color(ColorInput),
    Vector3(Vector3Input),
    Conductor(ConductorModel),
    Spectrum(Option<String>),
    Wall(WallMode),
    RoughTransmissionEstimator(RoughTransmissionEstimator),
    ScatteringEstimator(ScatteringEstimator),
    GradientMode(GradientMode),
    PhotonicPattern(PhotonicPattern),
}

impl PropertyValue {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Scalar(_) => "scalar",
            Self::Color(_) => "color",
            Self::Vector3(_) => "vector3",
            Self::Conductor(_) => "conductor",
            Self::Spectrum(_) => "spectrum",
            Self::Wall(_) => "wall",
            Self::RoughTransmissionEstimator(_) => "rough transmission estimator",
            Self::ScatteringEstimator(_) => "scattering estimator",
            Self::GradientMode(_) => "gradient mode",
            Self::PhotonicPattern(_) => "photonic pattern",
        }
    }
}

fn set_scalar(
    input: &mut ScalarInput,
    value: &PropertyValue,
    path: &str,
) -> Result<(), SchemaError> {
    let PropertyValue::Scalar(value) = value else {
        return Err(SchemaError::new(
            path,
            ErrorKind::PropertyType {
                expected: "scalar",
                actual: value.type_name(),
            },
        ));
    };
    *input = value.clone();
    Ok(())
}
fn set_color(input: &mut ColorInput, value: &PropertyValue, path: &str) -> Result<(), SchemaError> {
    let PropertyValue::Color(value) = value else {
        return Err(SchemaError::new(
            path,
            ErrorKind::PropertyType {
                expected: "color",
                actual: value.type_name(),
            },
        ));
    };
    *input = value.clone();
    Ok(())
}
fn set_vector3(
    input: &mut Vector3Input,
    value: &PropertyValue,
    path: &str,
) -> Result<(), SchemaError> {
    let PropertyValue::Vector3(value) = value else {
        return Err(SchemaError::new(
            path,
            ErrorKind::PropertyType {
                expected: "vector3",
                actual: value.type_name(),
            },
        ));
    };
    *input = value.clone();
    Ok(())
}
fn set_color_channel(
    input: &mut ColorInput,
    channel: usize,
    value: &PropertyValue,
    path: &str,
) -> Result<(), SchemaError> {
    let PropertyValue::Scalar(value) = value else {
        return Err(SchemaError::new(
            path,
            ErrorKind::PropertyType {
                expected: "scalar",
                actual: value.type_name(),
            },
        ));
    };
    let Some(scalar) = value.as_constant() else {
        return Err(SchemaError::new(
            path,
            ErrorKind::Incompatible("a colour-channel patch must use a constant scalar".into()),
        ));
    };
    let ColorInput::Constant(mut color) = input.clone() else {
        return Err(SchemaError::new(path, ErrorKind::Incompatible("a channel cannot replace part of a field/texture/animated colour; patch the whole colour".into())));
    };
    color.0[channel] = *scalar;
    *input = ColorInput::Constant(color);
    Ok(())
}

fn set_property(
    spec: &mut MaterialSpec,
    path: &str,
    value: &PropertyValue,
) -> Result<(), SchemaError> {
    macro_rules! scalar {
        ($name:literal, $target:expr) => {
            if path == $name {
                return set_scalar(&mut $target, value, path);
            }
        };
    }
    macro_rules! color {
        ($name:literal, $target:expr) => {
            if path == $name {
                return set_color(&mut $target, value, path);
            }
        };
    }
    scalar!("base.weight", spec.base.values.weight);
    scalar!("base.diffuse_weight", spec.base.values.diffuse_weight);
    color!("base.color", spec.base.values.color);
    scalar!(
        "base.ceramic.transmitted_fraction",
        spec.base.values.ceramic.values.transmitted_fraction
    );
    if path == "base.color.r" {
        return set_color_channel(&mut spec.base.values.color, 0, value, path);
    }
    if path == "base.color.g" {
        return set_color_channel(&mut spec.base.values.color, 1, value, path);
    }
    if path == "base.color.b" {
        return set_color_channel(&mut spec.base.values.color, 2, value, path);
    }
    scalar!("base.metalness", spec.base.values.metalness);
    scalar!("base.diffuse_roughness", spec.base.values.diffuse_roughness);
    if path == "base.conductor" {
        let PropertyValue::Conductor(conductor) = value else {
            return Err(SchemaError::new(
                path,
                ErrorKind::PropertyType {
                    expected: "conductor",
                    actual: value.type_name(),
                },
            ));
        };
        spec.base.values.conductor = conductor.clone();
        return Ok(());
    }
    scalar!("specular.weight", spec.specular.values.weight);
    color!("specular.color", spec.specular.values.color);
    scalar!("specular.ior", spec.specular.values.ior);
    scalar!("specular.roughness", spec.specular.values.roughness);
    scalar!("specular.anisotropy", spec.specular.values.anisotropy);
    scalar!(
        "specular.rotation_radians",
        spec.specular.values.rotation_radians
    );
    if path == "specular.tangent" {
        let tangent = spec
            .specular
            .values
            .tangent
            .get_or_insert_with(|| Vector3([1.0, 0.0, 0.0]).into());
        return set_vector3(tangent, value, path);
    }
    scalar!("coat.weight", spec.coat.values.weight);
    color!("coat.absorption_color", spec.coat.values.absorption_color);
    scalar!(
        "coat.attenuation_distance",
        spec.coat.values.attenuation_distance
    );
    scalar!("coat.ior", spec.coat.values.ior);
    scalar!("coat.roughness", spec.coat.values.roughness);
    scalar!("coat.anisotropy", spec.coat.values.anisotropy);
    scalar!("coat.rotation_radians", spec.coat.values.rotation_radians);
    scalar!("sheen.weight", spec.sheen.values.weight);
    color!("sheen.color", spec.sheen.values.color);
    scalar!("sheen.roughness", spec.sheen.values.roughness);
    scalar!("film.weight", spec.film.values.weight);
    scalar!("film.thickness_nm", spec.film.values.thickness_nm);
    scalar!("film.ior", spec.film.values.ior);
    scalar!(
        "film.thickness_variation",
        spec.film.values.thickness_variation
    );
    scalar!("film.pattern_scale", spec.film.values.pattern_scale);
    scalar!("film.flow", spec.film.values.flow);
    scalar!("film.drainage", spec.film.values.drainage);
    scalar!("film.ambient_weight", spec.film.values.ambient_weight);
    scalar!("transvolume.weight", spec.transvolume.values.weight);
    color!("transvolume.color", spec.transvolume.values.color);
    color!(
        "transvolume.absorption_coefficients",
        spec.transvolume.values.absorption_coefficients
    );
    if let Some(channel) = path
        .strip_prefix("transvolume.absorption_coefficients.")
        .and_then(channel_index)
    {
        return set_color_channel(
            &mut spec.transvolume.values.absorption_coefficients,
            channel,
            value,
            path,
        );
    }
    if let Some(channel) = path
        .strip_prefix("transvolume.color.")
        .and_then(channel_index)
    {
        return set_color_channel(&mut spec.transvolume.values.color, channel, value, path);
    }
    scalar!(
        "transvolume.absorption_scale",
        spec.transvolume.values.absorption_scale
    );
    scalar!(
        "transvolume.absorption_reference_distance",
        spec.transvolume.values.absorption_reference_distance
    );
    scalar!("transvolume.ior", spec.transvolume.values.ior);
    if path == "transvolume.dispersion.cauchy_amount" {
        let PropertyValue::Scalar(amount) = value else {
            return Err(SchemaError::new(
                path,
                ErrorKind::PropertyType {
                    expected: "scalar",
                    actual: value.type_name(),
                },
            ));
        };
        spec.transvolume.values.dispersion = DispersionModel::Cauchy {
            amount: amount.clone(),
        };
        return Ok(());
    }
    if path == "transvolume.rough_estimator" {
        let PropertyValue::RoughTransmissionEstimator(estimator) = value else {
            return Err(SchemaError::new(
                path,
                ErrorKind::PropertyType {
                    expected: "rough transmission estimator",
                    actual: value.type_name(),
                },
            ));
        };
        spec.transvolume.values.rough_estimator = *estimator;
        if let Some(legacy) = &mut spec.compatibility.legacy {
            legacy.rough_sample_share = match estimator {
                RoughTransmissionEstimator::Fast => 0.0,
                RoughTransmissionEstimator::Sampled => 1.0,
                RoughTransmissionEstimator::Hybrid { sampled_share } => *sampled_share,
            };
        }
        return Ok(());
    }
    if path == "transvolume.wall" {
        let PropertyValue::Wall(wall) = value else {
            return Err(SchemaError::new(
                path,
                ErrorKind::PropertyType {
                    expected: "wall",
                    actual: value.type_name(),
                },
            ));
        };
        spec.transvolume.values.wall = wall.clone();
        return Ok(());
    }
    if path == "transvolume.measured_spectrum" {
        let PropertyValue::Spectrum(spectrum) = value else {
            return Err(SchemaError::new(
                path,
                ErrorKind::PropertyType {
                    expected: "spectrum",
                    actual: value.type_name(),
                },
            ));
        };
        spec.transvolume.values.measured_spectrum = spectrum.clone();
        return Ok(());
    }
    scalar!("subsurface.weight", spec.subsurface.values.weight);
    color!(
        "subsurface.target_color",
        spec.subsurface.values.target_color
    );
    if let Some(channel) = path
        .strip_prefix("subsurface.target_color.")
        .and_then(channel_index)
    {
        return set_color_channel(
            &mut spec.subsurface.values.target_color,
            channel,
            value,
            path,
        );
    }
    color!(
        "subsurface.scatter_color",
        spec.subsurface.values.scatter_color
    );
    if let Some(channel) = path
        .strip_prefix("subsurface.scatter_color.")
        .and_then(channel_index)
    {
        return set_color_channel(
            &mut spec.subsurface.values.scatter_color,
            channel,
            value,
            path,
        );
    }
    color!("subsurface.radii", spec.subsurface.values.radii);
    scalar!(
        "subsurface.radius_scale",
        spec.subsurface.values.radius_scale
    );
    scalar!(
        "subsurface.mean_free_path",
        spec.subsurface.values.mean_free_path
    );
    color!(
        "subsurface.absorption_coefficients",
        spec.subsurface.values.absorption_coefficients
    );
    scalar!(
        "subsurface.phase_anisotropy",
        spec.subsurface.values.phase_anisotropy
    );
    if path == "subsurface.estimator" {
        let PropertyValue::ScatteringEstimator(estimator) = value else {
            return Err(SchemaError::new(
                path,
                ErrorKind::PropertyType {
                    expected: "scattering estimator",
                    actual: value.type_name(),
                },
            ));
        };
        spec.subsurface.values.estimator = *estimator;
        return Ok(());
    }
    scalar!(
        "subsurface.pigment.spectral_spread",
        spec.subsurface.values.pigment.spectral_spread
    );
    scalar!(
        "subsurface.pigment.near_surface_hue",
        spec.subsurface.values.pigment.near_surface_hue
    );
    scalar!(
        "subsurface.pigment.unscattered_hue",
        spec.subsurface.values.pigment.unscattered_hue
    );
    scalar!(
        "subsurface.pigment.opaque_exit",
        spec.subsurface.values.pigment.opaque_exit
    );
    color!("emission.color", spec.emission.values.color);
    if let Some(channel) = path.strip_prefix("emission.color.").and_then(channel_index) {
        return set_color_channel(&mut spec.emission.values.color, channel, value, path);
    }
    scalar!("emission.luminance", spec.emission.values.luminance);
    scalar!(
        "emission.volume.strength",
        spec.emission.values.volume.strength
    );
    scalar!(
        "emission.volume.core_concentration",
        spec.emission.values.volume.core_concentration
    );
    scalar!("coverage.opacity", spec.coverage.values.opacity);
    scalar!("coverage.alpha_cutoff", spec.coverage.values.alpha_cutoff);
    scalar!("geometry.bump.height", spec.geometry.values.bump.height);
    scalar!("geometry.bump.strength", spec.geometry.values.bump.strength);
    scalar!("geometry.bump.scale", spec.geometry.values.bump.scale);
    scalar!(
        "geometry.displacement.height",
        spec.geometry.values.displacement.height
    );
    scalar!(
        "geometry.displacement.midpoint",
        spec.geometry.values.displacement.midpoint
    );
    if path == "geometry.base_normal" {
        return set_vector3(&mut spec.geometry.values.base_normal, value, path);
    }
    if path == "geometry.coat_normal" {
        return set_vector3(&mut spec.geometry.values.coat_normal, value, path);
    }
    scalar!("wear.age", spec.wear.values.age);
    scalar!("wear.edge_wear", spec.wear.values.edge_wear);
    scalar!("wear.cavity", spec.wear.values.cavity);
    color!("wear.cavity_color", spec.wear.values.cavity_color);
    if let Some(channel) = path
        .strip_prefix("wear.cavity_color.")
        .and_then(channel_index)
    {
        return set_color_channel(&mut spec.wear.values.cavity_color, channel, value, path);
    }
    scalar!("wear.scratches", spec.wear.values.scratches);
    scalar!("wear.swirls", spec.wear.values.swirls);
    scalar!("wear.orange_peel", spec.wear.values.orange_peel);
    scalar!("wear.smudges", spec.wear.values.smudges);
    scalar!("wear.dust", spec.wear.values.dust);
    scalar!("wear.brushing", spec.wear.values.brushing);
    scalar!(
        "wear.brushing_angle_radians",
        spec.wear.values.brushing_angle_radians
    );
    scalar!("wear.mottle", spec.wear.values.mottle);
    scalar!(
        "special_optics.grating.strength",
        spec.special_optics.grating.values.strength
    );
    scalar!(
        "special_optics.grating.pitch_nm",
        spec.special_optics.grating.values.pitch_nm
    );
    scalar!(
        "special_optics.grating.angle_radians",
        spec.special_optics.grating.values.angle_radians
    );
    scalar!(
        "special_optics.grating.radial",
        spec.special_optics.grating.values.radial
    );
    scalar!(
        "special_optics.birefringence.delta_ior",
        spec.special_optics.birefringence.values.delta_ior
    );
    scalar!(
        "special_optics.birefringence.azimuth_radians",
        spec.special_optics.birefringence.values.azimuth_radians
    );
    scalar!(
        "special_optics.birefringence.elevation_radians",
        spec.special_optics.birefringence.values.elevation_radians
    );
    scalar!(
        "special_optics.grin.strength",
        spec.special_optics.grin.values.strength
    );
    scalar!(
        "special_optics.grin.depth",
        spec.special_optics.grin.values.depth
    );
    scalar!(
        "special_optics.bubbles.amount",
        spec.special_optics.bubbles.values.amount
    );
    scalar!(
        "special_optics.bubbles.size",
        spec.special_optics.bubbles.values.size
    );
    scalar!(
        "special_optics.bubbles.evenness",
        spec.special_optics.bubbles.values.evenness
    );
    scalar!(
        "special_optics.bubbles.shape",
        spec.special_optics.bubbles.values.shape
    );
    scalar!(
        "special_optics.bubbles.seed",
        spec.special_optics.bubbles.values.seed
    );
    scalar!(
        "special_optics.gradient.amount",
        spec.special_optics.gradient.values.amount
    );
    color!(
        "special_optics.gradient.target_tint",
        spec.special_optics.gradient.values.target_tint
    );
    if let Some(channel) = path
        .strip_prefix("special_optics.gradient.target_tint.")
        .and_then(channel_index)
    {
        return set_color_channel(
            &mut spec.special_optics.gradient.values.target_tint,
            channel,
            value,
            path,
        );
    }
    scalar!(
        "special_optics.gradient.azimuth_radians",
        spec.special_optics.gradient.values.azimuth_radians
    );
    scalar!(
        "special_optics.gradient.elevation_radians",
        spec.special_optics.gradient.values.elevation_radians
    );
    scalar!(
        "special_optics.gradient.offset",
        spec.special_optics.gradient.values.offset
    );
    scalar!(
        "special_optics.gradient.softness",
        spec.special_optics.gradient.values.softness
    );
    scalar!(
        "special_optics.gradient.bands",
        spec.special_optics.gradient.values.bands
    );
    scalar!(
        "special_optics.gradient.wave",
        spec.special_optics.gradient.values.wave
    );
    if path == "special_optics.gradient.mode" {
        let PropertyValue::GradientMode(mode) = value else {
            return Err(SchemaError::new(
                path,
                ErrorKind::PropertyType {
                    expected: "gradient mode",
                    actual: value.type_name(),
                },
            ));
        };
        spec.special_optics.gradient.values.mode = *mode;
        return Ok(());
    }
    scalar!(
        "special_optics.cracks.amount",
        spec.special_optics.cracks.values.amount
    );
    scalar!(
        "special_optics.cracks.fractures",
        spec.special_optics.cracks.values.fractures
    );
    scalar!(
        "special_optics.cracks.size",
        spec.special_optics.cracks.values.size
    );
    scalar!(
        "special_optics.cracks.depth",
        spec.special_optics.cracks.values.depth
    );
    scalar!(
        "special_optics.cracks.openness",
        spec.special_optics.cracks.values.openness
    );
    scalar!(
        "special_optics.cracks.growth",
        spec.special_optics.cracks.values.growth
    );
    scalar!(
        "special_optics.cracks.curvature",
        spec.special_optics.cracks.values.curvature
    );
    scalar!(
        "special_optics.cracks.roughness",
        spec.special_optics.cracks.values.roughness
    );
    scalar!(
        "special_optics.cracks.seed",
        spec.special_optics.cracks.values.seed
    );
    scalar!(
        "special_optics.photonic.amount",
        spec.special_optics.photonic.values.amount
    );
    scalar!(
        "special_optics.photonic.size",
        spec.special_optics.photonic.values.size
    );
    scalar!(
        "special_optics.photonic.longest_wavelength",
        spec.special_optics.photonic.values.longest_wavelength
    );
    scalar!(
        "special_optics.photonic.softness",
        spec.special_optics.photonic.values.softness
    );
    if path == "special_optics.photonic.pattern" {
        let PropertyValue::PhotonicPattern(pattern) = value else {
            return Err(SchemaError::new(
                path,
                ErrorKind::PropertyType {
                    expected: "photonic pattern",
                    actual: value.type_name(),
                },
            ));
        };
        spec.special_optics.photonic.values.pattern = *pattern;
        return Ok(());
    }
    scalar!(
        "special_optics.photonic.roll",
        spec.special_optics.photonic.values.roll
    );
    scalar!(
        "special_optics.photonic.seed",
        spec.special_optics.photonic.values.seed
    );
    color!(
        "special_optics.photonic.platelet_color",
        spec.special_optics.photonic.values.platelet_color
    );
    if let Some(channel) = path
        .strip_prefix("special_optics.photonic.platelet_color.")
        .and_then(channel_index)
    {
        return set_color_channel(
            &mut spec.special_optics.photonic.values.platelet_color,
            channel,
            value,
            path,
        );
    }
    scalar!(
        "special_optics.silk.amount",
        spec.special_optics.silk.values.amount
    );
    scalar!(
        "special_optics.silk.azimuth_radians",
        spec.special_optics.silk.values.azimuth_radians
    );
    scalar!(
        "special_optics.silk.elevation_radians",
        spec.special_optics.silk.values.elevation_radians
    );
    scalar!(
        "special_optics.silk.softness",
        spec.special_optics.silk.values.softness
    );
    scalar!(
        "special_optics.silk.wave",
        spec.special_optics.silk.values.wave
    );
    scalar!(
        "special_optics.pressure.amount",
        spec.special_optics.pressure.values.amount
    );
    scalar!(
        "special_optics.pressure.color_bias",
        spec.special_optics.pressure.values.color_bias
    );
    if path == "compatibility.legacy.rough_sample_share" {
        let value = constant_scalar(value, path)?;
        spec.compatibility
            .legacy
            .get_or_insert_with(empty_legacy)
            .rough_sample_share = value;
        return Ok(());
    }
    if path == "compatibility.legacy.sharp_environment" {
        let value = constant_scalar(value, path)?;
        spec.compatibility
            .legacy
            .get_or_insert_with(empty_legacy)
            .sharp_environment = value > 0.5;
        return Ok(());
    }
    if path == "compatibility.legacy.fit_distances_to_shape" {
        let value = constant_scalar(value, path)?;
        spec.compatibility
            .legacy
            .get_or_insert_with(empty_legacy)
            .fit_distances_to_shape = value > 0.5;
        spec.emission.values.volume.distance_fitted = value > 0.5;
        return Ok(());
    }
    Err(SchemaError::new(
        path,
        ErrorKind::UnknownProperty(path.into()),
    ))
}

fn constant_scalar(value: &PropertyValue, path: &str) -> Result<f64, SchemaError> {
    let PropertyValue::Scalar(input) = value else {
        return Err(SchemaError::new(
            path,
            ErrorKind::PropertyType {
                expected: "scalar",
                actual: value.type_name(),
            },
        ));
    };
    input.as_constant().copied().ok_or_else(|| {
        SchemaError::new(
            path,
            ErrorKind::Incompatible("compatibility values must be constant".into()),
        )
    })
}

fn empty_legacy() -> LegacyCompatibility {
    LegacyCompatibility {
        preset_index: None,
        categorical_metal: 0.0,
        roughness: LegacyRoughnessConvention::PerceptualSquaredGgx,
        coat_roughness: LegacyCoatRoughnessConvention::GlazeHazeIsGgxAlpha,
        rough_sample_share: 0.0,
        fit_distances_to_shape: false,
        sharp_environment: true,
    }
}

fn channel_index(name: &str) -> Option<usize> {
    match name {
        "r" => Some(0),
        "g" => Some(1),
        "b" => Some(2),
        _ => None,
    }
}

fn set_feature(spec: &mut MaterialSpec, path: &str, enabled: bool) -> Result<(), SchemaError> {
    let target = match path {
        "base" => &mut spec.base.enabled,
        "base.ceramic" => &mut spec.base.values.ceramic.enabled,
        "specular" => &mut spec.specular.enabled,
        "coat" => &mut spec.coat.enabled,
        "sheen" => &mut spec.sheen.enabled,
        "film" => &mut spec.film.enabled,
        "film.flakes" => &mut spec.film.values.flakes.enabled,
        "transvolume" => &mut spec.transvolume.enabled,
        "subsurface" => &mut spec.subsurface.enabled,
        "emission" => &mut spec.emission.enabled,
        "coverage" => &mut spec.coverage.enabled,
        "geometry" => &mut spec.geometry.enabled,
        "wear" => &mut spec.wear.enabled,
        "special_optics.grating" => &mut spec.special_optics.grating.enabled,
        "special_optics.birefringence" => &mut spec.special_optics.birefringence.enabled,
        "special_optics.grin" => &mut spec.special_optics.grin.enabled,
        "special_optics.bubbles" => &mut spec.special_optics.bubbles.enabled,
        "special_optics.gradient" => &mut spec.special_optics.gradient.enabled,
        "special_optics.cracks" => &mut spec.special_optics.cracks.enabled,
        "special_optics.photonic" => &mut spec.special_optics.photonic.enabled,
        "special_optics.silk" => &mut spec.special_optics.silk.enabled,
        "special_optics.pressure" => &mut spec.special_optics.pressure.enabled,
        _ => {
            return Err(SchemaError::new(
                path,
                ErrorKind::UnknownProperty(path.into()),
            ))
        }
    };
    *target = enabled;
    Ok(())
}

fn scalar_range(input: &ScalarInput, min: f64, max: f64, path: &str) -> Result<(), SchemaError> {
    let mut result = Ok(());
    input.for_each_authored(|value| {
        if result.is_ok() {
            result = range(*value, min, max, path);
        }
    });
    result
}
fn scalar_positive(input: &ScalarInput, path: &str) -> Result<(), SchemaError> {
    let mut result = Ok(());
    input.for_each_authored(|value| {
        if result.is_ok() && *value <= 0.0 {
            result = Err(SchemaError::new(
                path,
                ErrorKind::OutOfRange {
                    min: f64::MIN_POSITIVE,
                    max: f64::MAX,
                    value: *value,
                },
            ));
        }
    });
    result
}
fn scalar_nonnegative(input: &ScalarInput, path: &str) -> Result<(), SchemaError> {
    scalar_range(input, 0.0, f64::MAX, path)
}

/// Insert or replace a canonical scalar in a sparse patch.
pub fn scalar_override(patch: &mut MaterialPatch, path: &str, value: ScalarInput) {
    patch
        .properties
        .insert(PropertyPath::from(path), PropertyValue::Scalar(value));
}
/// Insert or replace a canonical feature switch without touching its values.
pub fn feature_override(patch: &mut MaterialPatch, path: &str, enabled: bool) {
    patch.features.insert(FeaturePath::from(path), enabled);
}

pub(crate) fn conductor_override(patch: &mut MaterialPatch, value: ConductorModel) {
    patch.properties.insert(
        PropertyPath::from("base.conductor"),
        PropertyValue::Conductor(value),
    );
}
