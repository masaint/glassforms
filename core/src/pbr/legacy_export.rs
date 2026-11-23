//! Incremental-adoption bridge back to the current scalar renderer payload.
//!
//! This is not a canonical persistence export: legacy `Settings` cannot retain dormant
//! canonical-only groups or non-constant inputs. Such cases fail explicitly.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::material::{self, NAMED};
use crate::pbr::spec::*;
use crate::pbr::value::*;
use crate::settings::Settings;

#[derive(Clone, Debug, PartialEq)]
pub struct LegacyExportError {
    pub path: String,
    pub kind: LegacyExportErrorKind,
}

impl LegacyExportError {
    fn new(path: impl Into<String>, kind: LegacyExportErrorKind) -> Self {
        Self {
            path: path.into(),
            kind,
        }
    }
}

impl fmt::Display for LegacyExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.kind)
    }
}
impl std::error::Error for LegacyExportError {}

#[derive(Clone, Debug, PartialEq)]
pub enum LegacyExportErrorKind {
    Schema(SchemaError),
    NonConstantInput,
    FractionalMetalness {
        value: f64,
    },
    UnsupportedFeature {
        reason: String,
    },
    ConflictingValues {
        reason: String,
    },
    UnknownMeasuredConductor(ConductorId),
    UnknownMeasuredSpectrum(String),
    InvalidPresetIndex(f64),
    ActiveVariantMismatch {
        feature: &'static str,
        canonical: bool,
        legacy: bool,
    },
    PatchSchema(SchemaError),
}

impl fmt::Display for LegacyExportErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Schema(error) => write!(f, "{error}"),
            Self::PatchSchema(error) => write!(f, "{error}"),
            Self::NonConstantInput => f.write_str("legacy Settings accept only constant inputs"),
            Self::FractionalMetalness { value } => write!(
                f,
                "fractional metalness {value} has no categorical legacy representation"
            ),
            Self::UnsupportedFeature { reason } => {
                write!(f, "current legacy renderer cannot represent {reason}")
            }
            Self::ConflictingValues { reason } => f.write_str(reason),
            Self::UnknownMeasuredConductor(id) => write!(f, "unknown measured conductor `{id}`"),
            Self::UnknownMeasuredSpectrum(id) => write!(f, "unknown measured spectrum `{id}`"),
            Self::InvalidPresetIndex(index) => write!(f, "invalid legacy preset index {index}"),
            Self::ActiveVariantMismatch {
                feature,
                canonical,
                legacy,
            } => write!(
                f,
                "{feature} enabled is {canonical} canonically but {legacy} in legacy Variant::of"
            ),
        }
    }
}

pub type LegacyExportResult<T> = Result<T, LegacyExportError>;

fn error(path: &str, reason: impl Into<String>) -> LegacyExportError {
    LegacyExportError::new(
        path,
        LegacyExportErrorKind::UnsupportedFeature {
            reason: reason.into(),
        },
    )
}

fn scalar(input: &ScalarInput, path: &str) -> LegacyExportResult<f64> {
    input
        .as_constant()
        .copied()
        .ok_or_else(|| LegacyExportError::new(path, LegacyExportErrorKind::NonConstantInput))
}

fn color(input: &ColorInput, path: &str) -> LegacyExportResult<Color> {
    input
        .as_constant()
        .copied()
        .ok_or_else(|| LegacyExportError::new(path, LegacyExportErrorKind::NonConstantInput))
}

fn conflict(path: &str, reason: impl Into<String>) -> LegacyExportError {
    LegacyExportError::new(
        path,
        LegacyExportErrorKind::ConflictingValues {
            reason: reason.into(),
        },
    )
}

impl MaterialSpec {
    /// Convert to the current renderer's complete scalar payload. Canonical-only features and
    /// non-constant inputs are rejected; fractional metalness is never categorically quantised.
    pub fn to_legacy_settings(&self) -> LegacyExportResult<Settings> {
        self.to_legacy_settings_impl(true)
    }

    /// Export the scalar values before applying legacy activity/exclusivity rules. Legacy-originated
    /// edits use this to let `Variant::of` reconcile a candidate without first rejecting the very
    /// combination (for example GRIN plus birefringence) whose priority it needs to determine.
    fn to_legacy_settings_impl(&self, strict: bool) -> LegacyExportResult<Settings> {
        if strict {
            self.validate().map_err(|error| {
                LegacyExportError::new(error.path.clone(), LegacyExportErrorKind::Schema(error))
            })?;
        }
        if !self.base.enabled {
            return Err(error("base.enabled", "a disabled base group"));
        }
        if !self.specular.enabled {
            return Err(error(
                "specular.enabled",
                "a disabled dielectric specular group",
            ));
        }
        for (enabled, path, reason) in [
            (
                self.sheen.enabled,
                "sheen.enabled",
                "sheen or thin-surface transmission",
            ),
            (
                self.coverage.enabled,
                "coverage.enabled",
                "coverage controls",
            ),
            (
                self.geometry.enabled,
                "geometry.enabled",
                "canonical normal, bump, or displacement controls",
            ),
            (
                self.film.values.flakes.enabled,
                "film.flakes.enabled",
                "flakes",
            ),
        ] {
            if enabled {
                return Err(error(path, reason));
            }
        }

        let metalness = scalar(&self.base.values.metalness, "base.metalness")?;
        if metalness != 0.0 && metalness != 1.0 {
            return Err(LegacyExportError::new(
                "base.metalness",
                LegacyExportErrorKind::FractionalMetalness { value: metalness },
            ));
        }
        if scalar(&self.base.values.weight, "base.weight")? != 1.0 {
            return Err(error("base.weight", "a substrate weight other than 1"));
        }
        let body = color(&self.base.values.color, "base.color")?;
        let target = color(
            &self.subsurface.values.target_color,
            "subsurface.target_color",
        )?;
        let scatter = color(
            &self.subsurface.values.scatter_color,
            "subsurface.scatter_color",
        )?;
        if target != body || scatter != body {
            return Err(conflict(
                "subsurface",
                "legacy bodyR/G/B must equal both subsurface target and scatter colours",
            ));
        }
        let ceramic = self.base.values.ceramic.enabled;
        let diffuse = scalar(&self.base.values.diffuse_weight, "base.diffuse_weight")?;
        if strict && diffuse != 0.0 && (!ceramic || diffuse != 1.0) {
            return Err(error(
                "base.diffuse_weight",
                "diffuse outside an enabled legacy ceramic surface",
            ));
        }
        let mut p = Settings::default();
        p.mat = match self
            .compatibility
            .legacy
            .as_ref()
            .and_then(|legacy| legacy.preset_index)
        {
            Some(index)
                if index >= 0.0 && index.fract() == 0.0 && NAMED.get(index as usize).is_some() =>
            {
                index
            }
            Some(index) => {
                return Err(LegacyExportError::new(
                    "compatibility.legacy.preset_index",
                    LegacyExportErrorKind::InvalidPresetIndex(index),
                ))
            }
            None => 0.0,
        };
        p.metal = if metalness == 0.0 {
            0.0
        } else {
            match &self.base.values.conductor {
                ConductorModel::Ideal { tint } => {
                    if color(tint, "base.conductor.tint")? != body {
                        return Err(conflict(
                            "base.conductor.tint",
                            "legacy ideal-conductor tint must equal bodyR/G/B",
                        ));
                    }
                    1.0
                }
                ConductorModel::Measured { id } => {
                    let table_id = id.as_str().strip_prefix("measured:").unwrap_or(id.as_str());
                    material::METALS
                        .iter()
                        .position(|metal| metal.id == table_id)
                        .map(|index| index as f64 + 2.0)
                        .ok_or_else(|| {
                            LegacyExportError::new(
                                "base.conductor.id",
                                LegacyExportErrorKind::UnknownMeasuredConductor(id.clone()),
                            )
                        })?
                }
                ConductorModel::CustomSpectral { .. } => {
                    return Err(error("base.conductor", "custom spectral conductors"))
                }
                ConductorModel::F0F82 { .. } => {
                    return Err(error("base.conductor", "F0/F82 conductors"))
                }
            }
        };
        if metalness == 0.0 && !self.transvolume.enabled {
            return Err(error(
                "transvolume.enabled",
                "a non-transmitting dielectric outside legacy ceramic/milky transport",
            ));
        }

        let specular = &self.specular.values;
        if scalar(&specular.weight, "specular.weight")? != 1.0
            || color(&specular.color, "specular.color")? != Color::WHITE
        {
            return Err(error(
                "specular",
                "non-unit dielectric specular weight or coloured specular",
            ));
        }
        if scalar(&specular.anisotropy, "specular.anisotropy")? != 0.0
            || scalar(&specular.rotation_radians, "specular.rotation_radians")? != 0.0
            || specular.tangent.is_some()
        {
            return Err(error(
                "specular",
                "canonical anisotropy, rotation, or tangent inputs",
            ));
        }
        let spec_ior = scalar(&specular.ior, "specular.ior")?;
        let volume = &self.transvolume.values;
        let volume_ior = scalar(&volume.ior, "transvolume.ior")?;
        if spec_ior != volume_ior {
            return Err(conflict(
                "ior",
                "legacy uses one IOR for specular and transmission",
            ));
        }
        p.ior = spec_ior;
        p.rough = scalar(&specular.roughness, "specular.roughness")?;

        let absorption = color(
            &volume.absorption_coefficients,
            "transvolume.absorption_coefficients",
        )?;
        let tint = color(&volume.color, "transvolume.color")?;
        for axis in 0..3 {
            if tint.0[axis] != (-absorption.0[axis]).exp() {
                return Err(conflict(
                    "transvolume.color",
                    "legacy tint must be exp(-absorption) on every channel",
                ));
            }
        }
        if scalar(
            &volume.absorption_reference_distance,
            "transvolume.absorption_reference_distance",
        )? != 1.0
        {
            return Err(error(
                "transvolume.absorption_reference_distance",
                "a non-unit absorption reference distance",
            ));
        }
        p.absorb = absorption.0;
        p.density = scalar(&volume.absorption_scale, "transvolume.absorption_scale")?;
        p.spectrum = match &volume.measured_spectrum {
            Some(id) => material::SPECTRA
                .iter()
                .position(|spectrum| spectrum.id == id)
                .map(|index| index as f64 + 1.0)
                .ok_or_else(|| {
                    LegacyExportError::new(
                        "transvolume.measured_spectrum",
                        LegacyExportErrorKind::UnknownMeasuredSpectrum(id.clone()),
                    )
                })?,
            None => 0.0,
        };
        p.disp = match &volume.dispersion {
            DispersionModel::None => 0.0,
            DispersionModel::Cauchy { amount } => scalar(amount, "transvolume.dispersion.amount")?,
            DispersionModel::SpectralAbbe { .. } => {
                return Err(error("transvolume.dispersion", "spectral/Abbe dispersion"))
            }
        };
        p.rough_ref = match volume.rough_estimator {
            RoughTransmissionEstimator::Fast => 0.0,
            RoughTransmissionEstimator::Sampled => 1.0,
            RoughTransmissionEstimator::Hybrid { sampled_share } => sampled_share,
        };
        match &volume.wall {
            WallMode::ClosedVolume => p.hollow = 0.0,
            WallMode::ThinWalled { thickness } => {
                if scalar(thickness, "transvolume.wall.thickness")? != material::WALL {
                    return Err(error(
                        "transvolume.wall.thickness",
                        "a thin-wall thickness other than legacy WALL",
                    ));
                }
                p.hollow = 1.0;
            }
        }
        let expected_transmission = if ceramic {
            scalar(
                &self.base.values.ceramic.values.transmitted_fraction,
                "base.ceramic.transmitted_fraction",
            )?
        } else if self.subsurface.enabled {
            1.0 - scalar(&self.subsurface.values.weight, "subsurface.weight")?.clamp(0.0, 1.0)
        } else {
            1.0
        };
        if strict && scalar(&volume.weight, "transvolume.weight")? != expected_transmission {
            return Err(error(
                "transvolume.weight",
                "a transmission weight outside the legacy SSS/ceramic partition",
            ));
        }

        let coat = &self.coat.values;
        if self.coat.enabled
            && (color(&coat.absorption_color, "coat.absorption_color")? != Color::WHITE
                || scalar(&coat.attenuation_distance, "coat.attenuation_distance")? != 1.0
                || scalar(&coat.anisotropy, "coat.anisotropy")? != 0.0
                || scalar(&coat.rotation_radians, "coat.rotation_radians")? != 0.0
                || !coat.internal_returns)
        {
            return Err(error(
                "coat",
                "coat absorption, anisotropy, rotation, or disabled internal returns",
            ));
        }
        p.coat = scalar(&coat.weight, "coat.weight")?;
        p.coat_ior = scalar(&coat.ior, "coat.ior")?;
        p.glaze_haze = scalar(&coat.roughness, "coat.roughness")?.powi(2);

        let film = &self.film.values;
        if self.film.enabled
            && (film.placement != FilmPlacement::OverCoat
                || film.filtering != FilmFiltering::RgbIntegrated)
        {
            return Err(error("film", "non-legacy film placement or filtering"));
        }
        let film_weight = scalar(&film.weight, "film.weight")?;
        if self.film.enabled && film_weight != 1.0 {
            return Err(error(
                "film.weight",
                "a film weight other than the legacy zero/one coverage",
            ));
        }
        p.film = scalar(&film.thickness_nm, "film.thickness_nm")?;
        p.film_ior = scalar(&film.ior, "film.ior")?;
        p.film_var = scalar(&film.thickness_variation, "film.thickness_variation")?;
        p.film_scale = scalar(&film.pattern_scale, "film.pattern_scale")?;
        p.film_flow = scalar(&film.flow, "film.flow")?;
        p.film_drain = scalar(&film.drainage, "film.drainage")?;
        p.film_sky = scalar(&film.ambient_weight, "film.ambient_weight")?;

        let sss = &self.subsurface.values;
        if self.subsurface.enabled
            && (color(&sss.radii, "subsurface.radii")? != Color([1.0, 0.5, 0.25])
                || scalar(&sss.radius_scale, "subsurface.radius_scale")? != 1.0
                || color(
                    &sss.absorption_coefficients,
                    "subsurface.absorption_coefficients",
                )? != Color::BLACK)
        {
            return Err(error(
                "subsurface",
                "canonical radii, radius scale, or absorption coefficients",
            ));
        }
        p.sss = scalar(&sss.weight, "subsurface.weight")?;
        p.sss_depth = scalar(&sss.mean_free_path, "subsurface.mean_free_path")?;
        p.sss_walk = if sss.estimator == ScatteringEstimator::RandomWalk {
            1.0
        } else {
            0.0
        };
        p.sss_spread = scalar(
            &sss.pigment.spectral_spread,
            "subsurface.pigment.spectral_spread",
        )?;
        p.sss_hue = scalar(
            &sss.pigment.near_surface_hue,
            "subsurface.pigment.near_surface_hue",
        )?;
        p.sss_through = scalar(
            &sss.pigment.unscattered_hue,
            "subsurface.pigment.unscattered_hue",
        )?;
        p.sss_opaque = scalar(
            &sss.pigment.opaque_exit,
            "subsurface.pigment.opaque_exit",
        )?;
        p.sss_forward = scalar(&sss.phase_anisotropy, "subsurface.phase_anisotropy")?;
        p.ceramic = if ceramic { 1.0 } else { 0.0 };
        p.ceramic_through = scalar(
            &self.base.values.ceramic.values.transmitted_fraction,
            "base.ceramic.transmitted_fraction",
        )?;
        p.body_r = body.0[0];
        p.body_g = body.0[1];
        p.body_b = body.0[2];

        let emission = &self.emission.values;
        if self.emission.enabled
            && (emission.blackbody.is_some()
                || scalar(&emission.luminance, "emission.luminance")? != 0.0)
        {
            return Err(error("emission", "surface luminance or blackbody emission"));
        }
        let glow = color(&emission.color, "emission.color")?;
        p.glow = scalar(&emission.volume.strength, "emission.volume.strength")?;
        p.glow_r = glow.0[0];
        p.glow_g = glow.0[1];
        p.glow_b = glow.0[2];
        p.glow_core = scalar(
            &emission.volume.core_concentration,
            "emission.volume.core_concentration",
        )?;

        let wear = &self.wear.values;
        p.age = scalar(&wear.age, "wear.age")?;
        p.edge_wear = scalar(&wear.edge_wear, "wear.edge_wear")?;
        p.cavity = scalar(&wear.cavity, "wear.cavity")?;
        let cavity = color(&wear.cavity_color, "wear.cavity_color")?;
        p.cavity_r = cavity.0[0];
        p.cavity_g = cavity.0[1];
        p.cavity_b = cavity.0[2];
        p.scratches = scalar(&wear.scratches, "wear.scratches")?;
        p.swirls = scalar(&wear.swirls, "wear.swirls")?;
        p.peel = scalar(&wear.orange_peel, "wear.orange_peel")?;
        p.smudges = scalar(&wear.smudges, "wear.smudges")?;
        p.dust = scalar(&wear.dust, "wear.dust")?;
        p.brush = scalar(&wear.brushing, "wear.brushing")?;
        p.brush_angle =
            scalar(&wear.brushing_angle_radians, "wear.brushing_angle_radians")?.to_degrees();
        p.mottle = scalar(&wear.mottle, "wear.mottle")?;

        let special = &self.special_optics;
        p.grating = scalar(
            &special.grating.values.strength,
            "special_optics.grating.strength",
        )?;
        p.grating_pitch = scalar(
            &special.grating.values.pitch_nm,
            "special_optics.grating.pitch_nm",
        )?;
        p.grating_angle = scalar(
            &special.grating.values.angle_radians,
            "special_optics.grating.angle_radians",
        )?
        .to_degrees();
        p.grating_radial = scalar(
            &special.grating.values.radial,
            "special_optics.grating.radial",
        )?;
        p.biref = scalar(
            &special.birefringence.values.delta_ior,
            "special_optics.birefringence.delta_ior",
        )?;
        p.biref_az = scalar(
            &special.birefringence.values.azimuth_radians,
            "special_optics.birefringence.azimuth_radians",
        )?
        .to_degrees();
        p.biref_el = scalar(
            &special.birefringence.values.elevation_radians,
            "special_optics.birefringence.elevation_radians",
        )?
        .to_degrees();
        p.grin = scalar(
            &special.grin.values.strength,
            "special_optics.grin.strength",
        )?;
        p.grin_depth = scalar(&special.grin.values.depth, "special_optics.grin.depth")?;
        p.bubbles = scalar(
            &special.bubbles.values.amount,
            "special_optics.bubbles.amount",
        )?;
        p.bubble_size = scalar(&special.bubbles.values.size, "special_optics.bubbles.size")?;
        p.bubble_even = scalar(
            &special.bubbles.values.evenness,
            "special_optics.bubbles.evenness",
        )?;
        p.bubble_shape = scalar(
            &special.bubbles.values.shape,
            "special_optics.bubbles.shape",
        )?;
        p.bubble_seed = scalar(&special.bubbles.values.seed, "special_optics.bubbles.seed")?;
        let gradient = &special.gradient.values;
        p.grad = scalar(&gradient.amount, "special_optics.gradient.amount")?;
        p.grad_mode = match gradient.mode {
            GradientMode::Axial => 0.0,
            GradientMode::Radial => 1.0,
            GradientMode::Bands => 2.0,
        };
        let target = color(&gradient.target_tint, "special_optics.gradient.target_tint")?;
        p.grad_r = target.0[0];
        p.grad_g = target.0[1];
        p.grad_b = target.0[2];
        p.grad_az = scalar(
            &gradient.azimuth_radians,
            "special_optics.gradient.azimuth_radians",
        )?
        .to_degrees();
        p.grad_el = scalar(
            &gradient.elevation_radians,
            "special_optics.gradient.elevation_radians",
        )?
        .to_degrees();
        p.grad_offset = scalar(&gradient.offset, "special_optics.gradient.offset")?;
        p.grad_soft = scalar(&gradient.softness, "special_optics.gradient.softness")?;
        p.grad_bands = scalar(&gradient.bands, "special_optics.gradient.bands")?;
        p.grad_wave = scalar(&gradient.wave, "special_optics.gradient.wave")?;
        let cracks = &special.cracks.values;
        p.cracks = scalar(&cracks.amount, "special_optics.cracks.amount")?;
        p.fractures = scalar(&cracks.fractures, "special_optics.cracks.fractures")?;
        p.crack_size = scalar(&cracks.size, "special_optics.cracks.size")?;
        p.crack_depth = scalar(&cracks.depth, "special_optics.cracks.depth")?;
        p.crack_open = scalar(&cracks.openness, "special_optics.cracks.openness")?;
        p.crack_grow = scalar(&cracks.growth, "special_optics.cracks.growth")?;
        p.crack_curve = scalar(&cracks.curvature, "special_optics.cracks.curvature")?;
        p.crack_rough = scalar(&cracks.roughness, "special_optics.cracks.roughness")?;
        p.crack_seed = scalar(&cracks.seed, "special_optics.cracks.seed")?;
        let photonic = &special.photonic.values;
        p.play = scalar(&photonic.amount, "special_optics.photonic.amount")?;
        p.play_size = scalar(&photonic.size, "special_optics.photonic.size")?;
        p.play_pattern = match photonic.pattern {
            PhotonicPattern::Blocks => 0.0,
            PhotonicPattern::Pinfire => 1.0,
            PhotonicPattern::BroadFlash => 2.0,
            PhotonicPattern::Platelets => 3.0,
        };
        p.play_colour = scalar(
            &photonic.longest_wavelength,
            "special_optics.photonic.longest_wavelength",
        )?;
        p.play_soft = scalar(&photonic.softness, "special_optics.photonic.softness")?;
        p.play_roll = scalar(&photonic.roll, "special_optics.photonic.roll")?;
        p.play_seed = scalar(&photonic.seed, "special_optics.photonic.seed")?;
        let platelets = color(
            &photonic.platelet_color,
            "special_optics.photonic.platelet_color",
        )?;
        p.glint_r = platelets.0[0];
        p.glint_g = platelets.0[1];
        p.glint_b = platelets.0[2];
        let silk = &special.silk.values;
        p.silk = scalar(&silk.amount, "special_optics.silk.amount")?;
        p.silk_az =
            scalar(&silk.azimuth_radians, "special_optics.silk.azimuth_radians")?.to_degrees();
        p.silk_el = scalar(
            &silk.elevation_radians,
            "special_optics.silk.elevation_radians",
        )?
        .to_degrees();
        p.silk_soft = scalar(&silk.softness, "special_optics.silk.softness")?;
        p.silk_wave = scalar(&silk.wave, "special_optics.silk.wave")?;
        p.piezo = scalar(
            &special.pressure.values.amount,
            "special_optics.pressure.amount",
        )?;
        p.piezo_colour = scalar(
            &special.pressure.values.color_bias,
            "special_optics.pressure.color_bias",
        )?;

        let legacy_fit = self
            .compatibility
            .legacy
            .as_ref()
            .is_some_and(|legacy| legacy.fit_distances_to_shape);
        if emission.volume.distance_fitted != legacy_fit {
            return Err(conflict(
                "emission.volume.distance_fitted",
                "distance fitting disagrees with compatibility.legacy.fit_distances_to_shape",
            ));
        }
        if self.coat.enabled
            && self.compatibility.legacy.as_ref().is_some_and(|legacy| {
                legacy.coat_roughness != LegacyCoatRoughnessConvention::GlazeHazeIsGgxAlpha
            })
        {
            return Err(error(
                "coat.roughness",
                "a non-legacy coat roughness convention",
            ));
        }
        if let Some(legacy) = &self.compatibility.legacy {
            p.fit = if legacy.fit_distances_to_shape {
                1.0
            } else {
                0.0
            };
            p.sharp = if legacy.sharp_environment { 1.0 } else { 0.0 };
        }
        if strict {
            disable_inactive_legacy_features(self, &mut p);
            check_active_variant(self, &p, metalness)?;
            let mut held = p.clone();
            held.hold_hard();
            if let Some(name) = material::body_names()
                .find(|name| p.get(name) != held.get(name))
            {
                return Err(error(
                    name,
                    format!("legacy {name}={} is outside its hard limit {}", p.get(name).unwrap(), held.get(name).unwrap()),
                ));
            }
            if p.absorb != held.absorb {
                return Err(error(
                    "transvolume.absorption_coefficients",
                    format!("legacy absorb={:?} is outside its hard limit {:?}", p.absorb, held.absorb),
                ));
            }
        }
        Ok(p)
    }

    /// Reconcile switches and derived partitions after an edit made through legacy controls. Authored
    /// values in losing groups remain dormant in the canonical owner; only their effective flags and
    /// the legacy transmission/diffuse partition are changed.
    pub(crate) fn reconcile_legacy_activity(
        &mut self,
        base: &MaterialSpec,
        edited: Option<&BTreeMap<String, f64>>,
    ) -> LegacyExportResult<()> {
        let mut p = self.to_legacy_settings_impl(false)?;
        const WEAR_AMOUNTS: [&str; 10] = [
            "age", "edgeWear", "cavity", "scratches", "swirls", "peel", "smudges", "dust",
            "brush", "mottle",
        ];
        let wear_edited = edited.is_some_and(|values| WEAR_AMOUNTS.iter().any(|name| values.contains_key(*name)));
        let before = base.to_legacy_settings_impl(false)?;
        if wear_edited {
            // Legacy wear has several amount controls but only one canonical group switch. Start from
            // the previously effective amounts, not dormant canonical payloads, then overlay exactly
            // the edited controls. This supports both enabling and disabling the group without
            // discarding any authored canonical values.
            let mut active_before = before.clone();
            disable_inactive_legacy_features(base, &mut active_before);
            if let Some(values) = edited {
                for name in WEAR_AMOUNTS {
                    let value = values.get(name).copied().or_else(|| active_before.get(name)).unwrap_or(0.0);
                    p.set(name, value);
                }
            }
        }
        let previous = material::Variant::of(&before);
        let candidate = material::Variant::of(&p);
        // A non-zero dormant amount is not by itself consent to reactivate an explicitly disabled
        // group during an unrelated edit. It can, however, become effective when the legacy condition
        // that previously suppressed it (thin wall, metal, random walk, or birefringence) is removed.
        // Keep the authored amounts themselves; only the temporary activity projection is masked.
        if !self.coat.enabled && candidate.coat && previous.coat {
            p.coat = 0.0;
        }
        if !self.film.enabled && candidate.film && previous.film {
            p.film = 0.0;
        }
        if !self.subsurface.enabled && candidate.sss && previous.sss {
            p.sss = 0.0;
        }
        if !self.emission.enabled && candidate.glow && previous.glow {
            p.glow = 0.0;
        }
        if !self.special_optics.grating.enabled && candidate.grating && previous.grating {
            p.grating = 0.0;
        }
        if !self.special_optics.bubbles.enabled && candidate.bubbles && previous.bubbles {
            p.bubbles = 0.0;
        }
        if !self.special_optics.birefringence.enabled && candidate.biref && previous.biref {
            p.biref = 0.0;
        }
        if !self.special_optics.grin.enabled && candidate.grin && previous.grin {
            p.grin = 0.0;
        }
        if !self.special_optics.gradient.enabled && candidate.inner.grad && previous.inner.grad {
            p.grad = 0.0;
        }
        if !self.special_optics.cracks.enabled && candidate.inner.crack && previous.inner.crack {
            p.cracks = 0.0;
            p.fractures = 0.0;
        }
        if !self.special_optics.photonic.enabled && candidate.inner.play && previous.inner.play {
            p.play = 0.0;
        }
        if !self.special_optics.silk.enabled && candidate.inner.silk && previous.inner.silk {
            p.silk = 0.0;
        }
        if !self.special_optics.pressure.enabled && candidate.inner.piezo && previous.inner.piezo {
            p.piezo = 0.0;
        }
        if !self.base.values.ceramic.enabled && candidate.ceramic && previous.ceramic {
            p.ceramic = 0.0;
        }
        if !wear_edited && !self.wear.enabled && candidate.wear && previous.wear {
            p.edge_wear = 0.0;
            p.cavity = 0.0;
            p.scratches = 0.0;
            p.swirls = 0.0;
            p.peel = 0.0;
            p.smudges = 0.0;
            p.dust = 0.0;
            p.brush = 0.0;
            p.mottle = 0.0;
        }
        let variant = material::Variant::of(&p);
        // Ceramic is a substrate selected by active SSS, not an independent transport mode. Keep its
        // authored selection while SSS is off so re-enabling scattering restores the same material.
        if variant.sss {
            self.base.values.ceramic.enabled = variant.ceramic;
        }
        self.base.values.diffuse_weight = if variant.ceramic { 1.0 } else { 0.0 }.into();
        self.transvolume.enabled = !variant.metal;
        self.transvolume.values.weight = if variant.ceramic {
            p.ceramic_through
        } else if variant.sss {
            1.0 - p.sss.clamp(0.0, 1.0)
        } else {
            1.0
        }
        .into();
        self.subsurface.enabled = variant.sss;
        self.coat.enabled = variant.coat;
        self.film.enabled = variant.film;
        self.emission.enabled = variant.glow;
        self.wear.enabled = variant.wear;
        self.special_optics.grating.enabled = variant.grating;
        self.special_optics.birefringence.enabled = variant.biref;
        self.special_optics.grin.enabled = variant.grin;
        self.special_optics.bubbles.enabled = variant.bubbles;
        self.special_optics.gradient.enabled = variant.inner.grad;
        self.special_optics.cracks.enabled = variant.inner.crack;
        self.special_optics.photonic.enabled = variant.inner.play;
        self.special_optics.silk.enabled = variant.inner.silk;
        self.special_optics.pressure.enabled = variant.inner.piezo;
        Ok(())
    }
}

fn disable_inactive_legacy_features(spec: &MaterialSpec, p: &mut Settings) {
    // A legacy amount is both value and implicit switch. Preserve suppressed authored amounts
    // (notably walk > biref > GRIN), but zero an amount when Variant would otherwise enable a
    // canonically disabled feature. The canonical source remains the lossless persistence owner.
    let initial = material::Variant::of(p);
    if !spec.film.enabled && initial.film {
        p.film = 0.0;
    }
    if !spec.special_optics.grating.enabled && initial.grating {
        p.grating = 0.0;
    }
    if !spec.coat.enabled && initial.coat {
        p.coat = 0.0;
    }
    if !spec.subsurface.enabled && initial.sss {
        p.sss = 0.0;
    }
    if !spec.emission.enabled && initial.glow {
        p.glow = 0.0;
    }
    if !spec.special_optics.bubbles.enabled && initial.bubbles {
        p.bubbles = 0.0;
    }
    if !spec.special_optics.birefringence.enabled && initial.biref {
        p.biref = 0.0;
    }
    if !spec.special_optics.grin.enabled && initial.grin {
        p.grin = 0.0;
    }
    if !spec.special_optics.gradient.enabled && initial.inner.grad {
        p.grad = 0.0;
    }
    if !spec.special_optics.cracks.enabled && initial.inner.crack {
        p.cracks = 0.0;
        p.fractures = 0.0;
    }
    if !spec.special_optics.photonic.enabled && initial.inner.play {
        p.play = 0.0;
    }
    if !spec.special_optics.silk.enabled && initial.inner.silk {
        p.silk = 0.0;
    }
    if !spec.special_optics.pressure.enabled && initial.inner.piezo {
        p.piezo = 0.0;
    }
    if !spec.base.values.ceramic.enabled && initial.ceramic {
        p.ceramic = 0.0;
    }
    if !spec.wear.enabled && initial.wear {
        p.edge_wear = 0.0;
        p.cavity = 0.0;
        p.scratches = 0.0;
        p.swirls = 0.0;
        p.peel = 0.0;
        p.smudges = 0.0;
        p.dust = 0.0;
        p.brush = 0.0;
        p.mottle = 0.0;
    }
}

fn check_active_variant(
    spec: &MaterialSpec,
    p: &Settings,
    metalness: f64,
) -> LegacyExportResult<()> {
    let actual = material::Variant::of(p);
    let sss = &spec.subsurface.values;
    let special = &spec.special_optics;
    let checks = [
        ("metal", metalness == 1.0, actual.metal),
        ("film", spec.film.enabled, actual.film),
        ("grating", special.grating.enabled, actual.grating),
        ("coat", spec.coat.enabled, actual.coat),
        ("subsurface", spec.subsurface.enabled, actual.sss),
        (
            "random walk",
            spec.subsurface.enabled && sss.estimator == ScatteringEstimator::RandomWalk,
            actual.walk,
        ),
        ("emission", spec.emission.enabled, actual.glow),
        ("GRIN", special.grin.enabled, actual.grin),
        ("birefringence", special.birefringence.enabled, actual.biref),
        ("bubbles", special.bubbles.enabled, actual.bubbles),
        (
            "thin wall",
            matches!(spec.transvolume.values.wall, WallMode::ThinWalled { .. }),
            actual.hollow,
        ),
        ("wear", spec.wear.enabled, actual.wear),
        // Keep the ceramic substrate selected while its parent SSS group is disabled. Ceramic cannot
        // be active without scattering, so that dormant selection must not reject the strict export.
        (
            "ceramic",
            spec.base.values.ceramic.enabled && spec.subsurface.enabled,
            actual.ceramic,
        ),
        ("gradient", special.gradient.enabled, actual.inner.grad),
        ("cracks", special.cracks.enabled, actual.inner.crack),
        (
            "photonic domains",
            special.photonic.enabled,
            actual.inner.play,
        ),
        ("silk", special.silk.enabled, actual.inner.silk),
        (
            "pressure emission",
            special.pressure.enabled,
            actual.inner.piezo,
        ),
    ];
    for (feature, canonical, legacy) in checks {
        if canonical != legacy {
            return Err(LegacyExportError::new(
                format!("{feature}.enabled"),
                LegacyExportErrorKind::ActiveVariantMismatch {
                    feature,
                    canonical,
                    legacy,
                },
            ));
        }
    }
    Ok(())
}

impl MaterialPatch {
    /// Export only keys affected by this sparse patch. `base` supplies inherited values and is
    /// checked by stable ID. Conflicting multi-field legacy aliases are rejected by full export.
    pub fn to_legacy_map(&self, base: &MaterialSpec) -> LegacyExportResult<BTreeMap<String, f64>> {
        let resolved = self.apply(base).map_err(|error| {
            LegacyExportError::new(
                error.path.clone(),
                LegacyExportErrorKind::PatchSchema(error),
            )
        })?;
        let settings = resolved.to_legacy_settings()?;
        let inherited = base.to_legacy_settings()?;
        let mut keys = BTreeSet::new();
        for path in self.properties.keys() {
            // Derived from sss/ceramicThrough/metal during legacy conversion, never an independent key.
            if !matches!(path.as_str(), "transvolume.weight" | "base.diffuse_weight" | "base.conductor") {
                legacy_keys_for_property(path.as_str(), &mut keys)?;
            }
        }
        for path in self.features.keys() {
            if feature_enabled(&resolved, path.as_str())? != feature_enabled(base, path.as_str())? {
                legacy_keys_for_feature(path.as_str(), &mut keys)?;
            }
        }
        if self
            .properties
            .contains_key(&PropertyPath::from("base.conductor"))
        {
            if settings.metal != inherited.metal {
                keys.insert("metal");
            }
            for key in ["bodyR", "bodyG", "bodyB"] {
                if settings.get(key) != inherited.get(key) {
                    keys.insert(key);
                }
            }
        }
        keys.into_iter()
            .map(|key| {
                settings
                    .get(key)
                    .map(|value| (key.to_string(), value))
                    .ok_or_else(|| {
                        error("patch", format!("internal legacy key {key} is unavailable"))
                    })
            })
            .collect()
    }
}

fn legacy_keys_for_property(
    path: &str,
    keys: &mut BTreeSet<&'static str>,
) -> LegacyExportResult<()> {
    macro_rules! map {
        ($($canonical:literal => $legacy:literal),+ $(,)?) => { match path { $($canonical => { keys.insert($legacy); return Ok(()); })+ _ => {} } };
    }
    map!(
        "base.metalness" => "metal", "base.ceramic.transmitted_fraction" => "ceramicThrough",
        "specular.ior" => "ior", "specular.roughness" => "rough", "coat.weight" => "coat", "coat.ior" => "coatIor",
        "coat.roughness" => "glazeHaze", "film.weight" => "film", "film.thickness_nm" => "film", "film.ior" => "filmIor",
        "film.thickness_variation" => "filmVar", "film.pattern_scale" => "filmScale", "film.flow" => "filmFlow",
        "film.drainage" => "filmDrain", "film.ambient_weight" => "filmSky",
        "transvolume.absorption_scale" => "density", "transvolume.measured_spectrum" => "spectrum", "transvolume.ior" => "ior",
        "transvolume.dispersion.cauchy_amount" => "disp", "transvolume.rough_estimator" => "roughRef",
        "subsurface.weight" => "sss", "subsurface.mean_free_path" => "sssDepth",
        "subsurface.estimator" => "sssWalk", "subsurface.phase_anisotropy" => "sssForward",
        "subsurface.pigment.spectral_spread" => "sssSpread", "subsurface.pigment.near_surface_hue" => "sssHue",
        "subsurface.pigment.unscattered_hue" => "sssThrough", "subsurface.pigment.opaque_exit" => "sssOpaque",
        "emission.volume.strength" => "glow",
        "emission.volume.core_concentration" => "glowCore", "wear.age" => "age", "wear.edge_wear" => "edgeWear",
        "wear.cavity" => "cavity", "wear.scratches" => "scratches", "wear.swirls" => "swirls",
        "wear.orange_peel" => "peel", "wear.smudges" => "smudges", "wear.dust" => "dust", "wear.brushing" => "brush",
        "wear.brushing_angle_radians" => "brushAngle", "wear.mottle" => "mottle",
        "special_optics.grating.strength" => "grating", "special_optics.grating.pitch_nm" => "gratingPitch",
        "special_optics.grating.angle_radians" => "gratingAngle", "special_optics.grating.radial" => "gratingRadial",
        "special_optics.birefringence.delta_ior" => "biref", "special_optics.birefringence.azimuth_radians" => "birefAz",
        "special_optics.birefringence.elevation_radians" => "birefEl", "special_optics.grin.strength" => "grin",
        "special_optics.grin.depth" => "grinDepth", "special_optics.bubbles.amount" => "bubbles",
        "special_optics.bubbles.size" => "bubbleSize", "special_optics.bubbles.evenness" => "bubbleEven",
        "special_optics.bubbles.shape" => "bubbleShape", "special_optics.bubbles.seed" => "bubbleSeed",
        "special_optics.gradient.amount" => "grad", "special_optics.gradient.mode" => "gradMode",
        "special_optics.gradient.azimuth_radians" => "gradAz", "special_optics.gradient.elevation_radians" => "gradEl",
        "special_optics.gradient.offset" => "gradOffset", "special_optics.gradient.softness" => "gradSoft",
        "special_optics.gradient.bands" => "gradBands", "special_optics.gradient.wave" => "gradWave",
        "special_optics.cracks.amount" => "cracks", "special_optics.cracks.fractures" => "fractures",
        "special_optics.cracks.size" => "crackSize", "special_optics.cracks.depth" => "crackDepth",
        "special_optics.cracks.openness" => "crackOpen", "special_optics.cracks.growth" => "crackGrow",
        "special_optics.cracks.curvature" => "crackCurve", "special_optics.cracks.roughness" => "crackRough",
        "special_optics.cracks.seed" => "crackSeed", "special_optics.photonic.amount" => "play",
        "special_optics.photonic.size" => "playSize", "special_optics.photonic.pattern" => "playPattern",
        "special_optics.photonic.longest_wavelength" => "playColour", "special_optics.photonic.softness" => "playSoft",
        "special_optics.photonic.roll" => "playRoll", "special_optics.photonic.seed" => "playSeed",
        "special_optics.silk.amount" => "silk", "special_optics.silk.azimuth_radians" => "silkAz",
        "special_optics.silk.elevation_radians" => "silkEl", "special_optics.silk.softness" => "silkSoft",
        "special_optics.silk.wave" => "silkWave", "special_optics.pressure.amount" => "piezo",
        "special_optics.pressure.color_bias" => "piezoColour", "compatibility.legacy.rough_sample_share" => "roughRef",
        "compatibility.legacy.sharp_environment" => "sharp", "compatibility.legacy.fit_distances_to_shape" => "fit"
    );
    if path == "transvolume.wall" {
        keys.insert("hollow");
        return Ok(());
    }
    for (prefix, rgb) in [
        ("base.color", "body"),
        ("subsurface.target_color", "body"),
        ("subsurface.scatter_color", "body"),
        ("emission.color", "glow"),
        ("wear.cavity_color", "cavity"),
        ("special_optics.gradient.target_tint", "grad"),
        ("special_optics.photonic.platelet_color", "glint"),
        ("transvolume.color", "tint"),
        ("transvolume.absorption_coefficients", "tint"),
    ] {
        if path == prefix {
            for channel in 0..3 {
                keys.insert(legacy_color_key(rgb, channel));
            }
            return Ok(());
        }
        for (suffix, channel) in [(".r", 0), (".g", 1), (".b", 2)] {
            if path == format!("{prefix}{suffix}") {
                keys.insert(legacy_color_key(rgb, channel));
                return Ok(());
            }
        }
    }
    Err(error(
        path,
        "this canonical property in a sparse legacy map",
    ))
}

fn legacy_color_key(rgb: &str, channel: usize) -> &'static str {
    const GROUPS: &[(&str, [&str; 3])] = &[
        ("body", ["bodyR", "bodyG", "bodyB"]),
        ("glow", ["glowR", "glowG", "glowB"]),
        ("cavity", ["cavityR", "cavityG", "cavityB"]),
        ("grad", ["gradR", "gradG", "gradB"]),
        ("glint", ["glintR", "glintG", "glintB"]),
        ("tint", ["tintR", "tintG", "tintB"]),
    ];
    GROUPS
        .iter()
        .find(|(name, _)| *name == rgb)
        .expect("internal colour group")
        .1[channel]
}

fn feature_enabled(spec: &MaterialSpec, path: &str) -> LegacyExportResult<bool> {
    let special = &spec.special_optics;
    Ok(match path {
        "base" => spec.base.enabled,
        "base.ceramic" => spec.base.values.ceramic.enabled,
        "specular" => spec.specular.enabled,
        "coat" => spec.coat.enabled,
        "sheen" => spec.sheen.enabled,
        "film" => spec.film.enabled,
        "film.flakes" => spec.film.values.flakes.enabled,
        "transvolume" => spec.transvolume.enabled,
        "subsurface" => spec.subsurface.enabled,
        "emission" => spec.emission.enabled,
        "coverage" => spec.coverage.enabled,
        "geometry" => spec.geometry.enabled,
        "wear" => spec.wear.enabled,
        "special_optics.grating" => special.grating.enabled,
        "special_optics.birefringence" => special.birefringence.enabled,
        "special_optics.grin" => special.grin.enabled,
        "special_optics.bubbles" => special.bubbles.enabled,
        "special_optics.gradient" => special.gradient.enabled,
        "special_optics.cracks" => special.cracks.enabled,
        "special_optics.photonic" => special.photonic.enabled,
        "special_optics.silk" => special.silk.enabled,
        "special_optics.pressure" => special.pressure.enabled,
        _ => return Err(error(path, "unknown feature switch in a sparse legacy map")),
    })
}

fn legacy_keys_for_feature(
    path: &str,
    keys: &mut BTreeSet<&'static str>,
) -> LegacyExportResult<()> {
    let feature_keys: &[&str] = match path {
        "coat" => &["coat"],
        "film" => &["film"],
        "transvolume" => &["metal"],
        "subsurface" => &["sss"],
        "emission" => &["glow"],
        "base.ceramic" => &["ceramic"],
        "special_optics.grating" => &["grating"],
        "special_optics.birefringence" => &["biref"],
        "special_optics.grin" => &["grin"],
        "special_optics.bubbles" => &["bubbles"],
        "special_optics.gradient" => &["grad"],
        "special_optics.cracks" => &["cracks", "fractures"],
        "special_optics.photonic" => &["play"],
        "special_optics.silk" => &["silk"],
        "special_optics.pressure" => &["piezo"],
        "wear" => &[
            "age",
            "edgeWear",
            "cavity",
            "scratches",
            "swirls",
            "peel",
            "smudges",
            "dust",
            "brush",
            "mottle",
        ],
        "film.flakes" | "sheen" | "coverage" | "geometry" => &[],
        "base" | "specular" => return Err(error(path, "this feature switch in legacy Settings")),
        _ => return Err(error(path, "unknown feature switch in a sparse legacy map")),
    };
    keys.extend(feature_keys.iter().copied());
    Ok(())
}
