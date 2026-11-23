use std::collections::BTreeMap;

use crate::material::{self, NAMED};
use crate::pbr::spec::*;
use crate::pbr::value::*;
use crate::pbr::MATERIAL_SCHEMA_VERSION;
use crate::settings::Settings;

impl MaterialSpec {
    /// Convert a complete legacy scene/body payload. The caller supplies the new stable material ID.
    /// Numeric preset identity is retained only as compatibility metadata; new identity uses `NAMED.id`.
    pub fn from_legacy(settings: &Settings, id: MaterialId) -> Self {
        Self::convert_legacy(settings, id, None)
    }

    /// Convert legacy values while supplying a continuous metalness from an import or new UI.
    /// Unlike `from_legacy`, this never quantises a fractional metalness into the old selector.
    pub fn from_legacy_with_metalness(
        settings: &Settings,
        id: MaterialId,
        metalness: f64,
    ) -> Result<Self, SchemaError> {
        range(metalness, 0.0, 1.0, "metalness")?;
        Ok(Self::convert_legacy(settings, id, Some(metalness)))
    }

    fn convert_legacy(p: &Settings, id: MaterialId, continuous_metalness: Option<f64>) -> Self {
        let mut spec = MaterialSpec {
            id,
            ..MaterialSpec::default()
        };
        let preset_index = (p.mat >= 0.0 && p.mat.fract() == 0.0)
            .then_some(p.mat)
            .filter(|index| NAMED.get(*index as usize).is_some());
        if let Some(index) = preset_index {
            let named = &NAMED[index as usize];
            spec.name = named.name.into();
            spec.preset = Some(PresetId::from(named.id));
        } else {
            spec.name = "Legacy custom material".into();
        }

        let variant = material::Variant::of(p);
        let metal_active = variant.metal;
        let hollow = variant.hollow;
        let sss_active = variant.sss;
        let ceramic_active = variant.ceramic;
        let body = Color([p.body_r, p.body_g, p.body_b]);

        spec.base.values.color = body.into();
        spec.base.values.diffuse_weight = if ceramic_active { 1.0 } else { 0.0 }.into();
        spec.base.values.metalness = continuous_metalness
            .unwrap_or(if metal_active { 1.0 } else { 0.0 })
            .into();
        spec.base.values.conductor = legacy_conductor(p, body);
        spec.base.values.ceramic = Feature {
            enabled: ceramic_active,
            values: CeramicProperties {
                transmitted_fraction: p.ceramic_through.into(),
            },
        };
        spec.specular.values.ior = p.ior.into();
        spec.specular.values.roughness = p.rough.into();

        spec.coat = Feature {
            enabled: variant.coat,
            values: CoatProperties {
                weight: p.coat.into(),
                ior: p.coat_ior.into(),
                // Legacy ceramic haze is already GGX alpha; canonical roughness is perceptual.
                roughness: if ceramic_active {
                    p.glaze_haze.clamp(0.0, 0.6).sqrt()
                } else {
                    0.0
                }
                .into(),
                ..CoatProperties::default()
            },
        };
        spec.film = Feature {
            enabled: variant.film,
            values: FilmProperties {
                weight: if p.film > 0.0 { 1.0 } else { 0.0 }.into(),
                thickness_nm: p.film.into(),
                ior: p.film_ior.into(),
                thickness_variation: p.film_var.into(),
                pattern_scale: p.film_scale.into(),
                flow: p.film_flow.into(),
                drainage: p.film_drain.into(),
                ambient_weight: p.film_sky.into(),
                ..FilmProperties::default()
            },
        };
        spec.transvolume = Feature {
            enabled: !metal_active,
            values: TransmissionVolumeProperties {
                weight: if ceramic_active {
                    p.ceramic_through
                } else if sss_active {
                    1.0 - p.sss.clamp(0.0, 1.0)
                } else {
                    1.0
                }
                .into(),
                color: Color(p.absorb.map(|a| (-a).exp())).into(),
                absorption_coefficients: Color(p.absorb).into(),
                absorption_scale: p.density.into(),
                measured_spectrum: material::spectrum_of(p).map(|spectrum| spectrum.id.into()),
                ior: p.ior.into(),
                dispersion: DispersionModel::Cauchy {
                    amount: p.disp.into(),
                },
                rough_estimator: if p.rough_ref >= 1.0 {
                    RoughTransmissionEstimator::Sampled
                } else if p.rough_ref > 0.0 {
                    RoughTransmissionEstimator::Hybrid {
                        sampled_share: p.rough_ref,
                    }
                } else {
                    RoughTransmissionEstimator::Fast
                },
                wall: if hollow {
                    WallMode::ThinWalled {
                        thickness: material::WALL.into(),
                    }
                } else {
                    WallMode::ClosedVolume
                },
                medium: MediumDescriptor {
                    id: crate::pbr::legacy_medium_id(&spec.id),
                    interface_priority: 0,
                },
                ..TransmissionVolumeProperties::default()
            },
        };
        spec.subsurface = Feature {
            enabled: sss_active,
            values: SubsurfaceProperties {
                weight: p.sss.into(),
                target_color: body.into(),
                scatter_color: body.into(),
                mean_free_path: p.sss_depth.into(),
                phase_anisotropy: p.sss_forward.into(),
                estimator: if p.sss_walk > 0.5 {
                    ScatteringEstimator::RandomWalk
                } else {
                    ScatteringEstimator::FastVolume
                },
                pigment: LegacyPigmentResponse {
                    spectral_spread: p.sss_spread.into(),
                    near_surface_hue: p.sss_hue.into(),
                    unscattered_hue: p.sss_through.into(),
                    opaque_exit: p.sss_opaque.into(),
                },
                ..SubsurfaceProperties::default()
            },
        };
        spec.emission = Feature {
            enabled: variant.glow,
            values: EmissionProperties {
                color: Color([p.glow_r, p.glow_g, p.glow_b]).into(),
                volume: VolumeEmission {
                    strength: p.glow.into(),
                    core_concentration: p.glow_core.into(),
                    distance_fitted: p.fit > 0.5,
                },
                ..EmissionProperties::default()
            },
        };
        spec.wear = Feature {
            enabled: variant.wear,
            values: WearProperties {
                age: p.age.into(),
                edge_wear: p.edge_wear.into(),
                cavity: p.cavity.into(),
                cavity_color: Color([p.cavity_r, p.cavity_g, p.cavity_b]).into(),
                scratches: p.scratches.into(),
                swirls: p.swirls.into(),
                orange_peel: p.peel.into(),
                smudges: p.smudges.into(),
                dust: p.dust.into(),
                brushing: p.brush.into(),
                brushing_angle_radians: p.brush_angle.to_radians().into(),
                mottle: p.mottle.into(),
            },
        };
        spec.special_optics = SpecialOpticsProperties {
            grating: Feature {
                enabled: variant.grating,
                values: GratingProperties {
                    strength: p.grating.into(),
                    pitch_nm: p.grating_pitch.into(),
                    angle_radians: p.grating_angle.to_radians().into(),
                    radial: p.grating_radial.into(),
                    ..GratingProperties::default()
                },
            },
            birefringence: Feature {
                enabled: variant.biref,
                values: BirefringenceProperties {
                    delta_ior: p.biref.into(),
                    azimuth_radians: p.biref_az.to_radians().into(),
                    elevation_radians: p.biref_el.to_radians().into(),
                },
            },
            grin: Feature {
                enabled: variant.grin,
                values: GrinProperties {
                    strength: p.grin.into(),
                    depth: p.grin_depth.into(),
                },
            },
            bubbles: Feature {
                enabled: variant.bubbles,
                values: BubbleProperties {
                    amount: p.bubbles.into(),
                    size: p.bubble_size.into(),
                    evenness: p.bubble_even.into(),
                    shape: p.bubble_shape.into(),
                    seed: p.bubble_seed.into(),
                },
            },
            gradient: Feature {
                enabled: variant.inner.grad,
                values: GradientProperties {
                    amount: p.grad.into(),
                    mode: match p.grad_mode.round() as i64 {
                        1 => GradientMode::Radial,
                        2 => GradientMode::Bands,
                        _ => GradientMode::Axial,
                    },
                    target_tint: Color([p.grad_r, p.grad_g, p.grad_b]).into(),
                    azimuth_radians: p.grad_az.to_radians().into(),
                    elevation_radians: p.grad_el.to_radians().into(),
                    offset: p.grad_offset.into(),
                    softness: p.grad_soft.into(),
                    bands: p.grad_bands.into(),
                    wave: p.grad_wave.into(),
                },
            },
            cracks: Feature {
                enabled: variant.inner.crack,
                values: CrackProperties {
                    amount: p.cracks.into(),
                    fractures: p.fractures.into(),
                    size: p.crack_size.into(),
                    depth: p.crack_depth.into(),
                    openness: p.crack_open.into(),
                    growth: p.crack_grow.into(),
                    curvature: p.crack_curve.into(),
                    roughness: p.crack_rough.into(),
                    seed: p.crack_seed.into(),
                },
            },
            photonic: Feature {
                enabled: variant.inner.play,
                values: PhotonicProperties {
                    amount: p.play.into(),
                    size: p.play_size.into(),
                    pattern: match p.play_pattern.round() as i64 {
                        1 => PhotonicPattern::Pinfire,
                        2 => PhotonicPattern::BroadFlash,
                        3 => PhotonicPattern::Platelets,
                        _ => PhotonicPattern::Blocks,
                    },
                    longest_wavelength: p.play_colour.into(),
                    softness: p.play_soft.into(),
                    roll: p.play_roll.into(),
                    seed: p.play_seed.into(),
                    platelet_color: Color([p.glint_r, p.glint_g, p.glint_b]).into(),
                },
            },
            silk: Feature {
                enabled: variant.inner.silk,
                values: SilkProperties {
                    amount: p.silk.into(),
                    azimuth_radians: p.silk_az.to_radians().into(),
                    elevation_radians: p.silk_el.to_radians().into(),
                    softness: p.silk_soft.into(),
                    wave: p.silk_wave.into(),
                },
            },
            pressure: Feature {
                enabled: variant.inner.piezo,
                values: PressureProperties {
                    amount: p.piezo.into(),
                    color_bias: p.piezo_colour.into(),
                },
            },
        };
        // Preserve old load/render behaviour without discarding the losing authored controls.
        // In particular Variant::of implements walk > birefringence > GRIN. Newly authored canonical
        // combinations still fail validation; this notice records why a migrated value is dormant.
        for (name, authored, effective) in [
            ("grating", p.grating > 0.0, variant.grating),
            ("birefringence", p.biref != 0.0, variant.biref),
            ("GRIN", p.grin != 0.0, variant.grin),
            ("bubbles", p.bubbles > 0.0, variant.bubbles),
            ("gradient", p.grad > 0.0, variant.inner.grad),
            (
                "cracks",
                p.cracks > 0.0 || p.fractures > 0.0,
                variant.inner.crack,
            ),
            ("photonic domains", p.play > 0.0, variant.inner.play),
            ("silk", p.silk > 0.0, variant.inner.silk),
            ("pressure emission", p.piezo > 0.0, variant.inner.piezo),
        ] {
            if authored && !effective {
                spec.compatibility.notices.push(format!(
                    "legacy Variant::of disabled {name}; authored values retained as dormant compatibility data"
                ));
            }
        }
        spec.compatibility.legacy = Some(LegacyCompatibility {
            preset_index: Some(p.mat),
            categorical_metal: p.metal,
            roughness: LegacyRoughnessConvention::PerceptualSquaredGgx,
            coat_roughness: LegacyCoatRoughnessConvention::GlazeHazeIsGgxAlpha,
            rough_sample_share: p.rough_ref,
            fit_distances_to_shape: p.fit > 0.5,
            sharp_environment: p.sharp > 0.5,
        });
        spec
    }
}

fn legacy_conductor(p: &Settings, body: Color) -> ConductorModel {
    match material::metal_of(p) {
        Some(metal) if p.metal.round() >= 2.0 => ConductorModel::Measured {
            id: ConductorId::from(format!("measured:{}", metal.id)),
        },
        _ => ConductorModel::Ideal { tint: body.into() },
    }
}

impl MaterialPatch {
    /// Convert the stored RGB absorption control directly, without a lossy derived-tint round trip.
    pub(crate) fn from_legacy_absorption(base: &MaterialSpec, absorb: [f64; 3]) -> Self {
        let mut patch = MaterialPatch::new(base.id.clone());
        absorption_override(&mut patch, absorb);
        patch
    }

    /// Convert sparse construction/body map entries into canonical field-level overrides.
    /// `base` supplies inherited values for colour channels and categorical conductor changes.
    /// Unsupported keys fail explicitly; no entry is silently dropped.
    pub fn from_legacy_map(
        base: &MaterialSpec,
        values: &BTreeMap<String, f64>,
    ) -> Result<Self, SchemaError> {
        if values.contains_key("mat") {
            return Err(invalid("mat", "a positional preset is only valid on a construction override"));
        }
        Self::from_legacy_construction_map(base, values)
    }

    /// Convert a sparse construction override, including the positional `mat` recipe used by old presets.
    pub fn from_legacy_construction_map(
        base: &MaterialSpec,
        values: &BTreeMap<String, f64>,
    ) -> Result<Self, SchemaError> {
        // Older construction presets stored `mat` beside a complete recipe. Rebase that recipe on the
        // effective owner, apply any later sparse edits over it, and retain only actual differences.
        let migrated;
        let absorption;
        let values = if let Some(index) = values.get("mat") {
            if index.fract() != 0.0 || *index < 0.0 {
                return Err(invalid("mat", "expected a named material index"));
            }
            let named = NAMED.get(*index as usize).ok_or_else(|| invalid("mat", "unknown named material index"))?;
            let inherited = base.to_legacy_settings().map_err(|error| invalid("mat", error.to_string()))?;
            let mut selected = inherited.clone();
            named.apply(&mut selected);
            for (name, value) in values.iter().filter(|(name, _)| name.as_str() != "mat") {
                selected.set(name, *value);
            }
            absorption = (selected.absorb != inherited.absorb).then_some(selected.absorb);
            migrated = material::body_names()
                .filter(|name| *name != "mat")
                .filter(|name| absorption.is_none() || !Settings::DERIVED.contains(name))
                .filter_map(|name| {
                    let value = selected.get(name)?;
                    (inherited.get(name) != Some(value)).then(|| (name.to_string(), value))
                })
                .chain(values.iter().filter(|(name, _)| {
                    name.as_str() != "mat" && !material::body_names().any(|known| known == *name)
                }).map(|(name, value)| (name.clone(), *value)))
                .collect();
            &migrated
        } else {
            absorption = None;
            values
        };
        let mut patch = MaterialPatch::new(base.id.clone());
        for (name, value) in values {
            legacy_patch_value(&mut patch, base, name, *value)?;
        }
        if let Some(absorb) = absorption {
            absorption_override(&mut patch, absorb);
        }
        // An ideal conductor uses the legacy body colour; measured conductors retain their table identity.
        patch.reconciled_legacy_activity(base, Some(values))
            .map_err(|error| invalid("activity", error))
    }

    /// Re-resolve a legacy-originated patch against a possibly changed owner. Only activity fields
    /// that differ from the base (or were already explicit) are added, so ordinary scalar patches keep
    /// inheriting unrelated group state.
    pub(crate) fn reconciled_legacy_activity(
        self,
        base: &MaterialSpec,
        edited: Option<&BTreeMap<String, f64>>,
    ) -> Result<Self, String> {
        self.reconciled_legacy_activity_from(base, base, edited)
    }

    /// Reconcile against the previously effective patch, which may contain dormant values absent from
    /// its inherited owner. Whole replacement and rebasing use the owner itself as that previous state.
    pub(crate) fn reconciled_legacy_activity_from(
        mut self,
        base: &MaterialSpec,
        previous: &MaterialSpec,
        edited: Option<&BTreeMap<String, f64>>,
    ) -> Result<Self, String> {
        self.base = base.id.clone();
        // Rebasing can turn an inherited measured conductor into an ideal one. A body-colour patch
        // must then rederive the linked ideal tint, just as it did when the patch was first converted.
        let body_changed = self.properties.keys().any(|path| {
            path.as_str() == "base.color" || path.as_str().starts_with("base.color.")
        });
        if body_changed {
            let effective = self.apply(base).map_err(|error| error.to_string())?;
            if matches!(effective.base.values.conductor, ConductorModel::Ideal { .. }) {
                conductor_override(
                    &mut self,
                    ConductorModel::Ideal {
                        tint: effective.base.values.color,
                    },
                );
            }
        }
        let mut effective = self.apply(base).map_err(|error| error.to_string())?;
        effective.reconcile_legacy_activity(previous, edited).map_err(|error| error.to_string())?;
        let feature_values = [
            ("base.ceramic", effective.base.values.ceramic.enabled, base.base.values.ceramic.enabled),
            ("transvolume", effective.transvolume.enabled, base.transvolume.enabled),
            ("subsurface", effective.subsurface.enabled, base.subsurface.enabled),
            ("coat", effective.coat.enabled, base.coat.enabled),
            ("film", effective.film.enabled, base.film.enabled),
            ("emission", effective.emission.enabled, base.emission.enabled),
            ("wear", effective.wear.enabled, base.wear.enabled),
            ("special_optics.grating", effective.special_optics.grating.enabled, base.special_optics.grating.enabled),
            ("special_optics.birefringence", effective.special_optics.birefringence.enabled, base.special_optics.birefringence.enabled),
            ("special_optics.grin", effective.special_optics.grin.enabled, base.special_optics.grin.enabled),
            ("special_optics.bubbles", effective.special_optics.bubbles.enabled, base.special_optics.bubbles.enabled),
            ("special_optics.gradient", effective.special_optics.gradient.enabled, base.special_optics.gradient.enabled),
            ("special_optics.cracks", effective.special_optics.cracks.enabled, base.special_optics.cracks.enabled),
            ("special_optics.photonic", effective.special_optics.photonic.enabled, base.special_optics.photonic.enabled),
            ("special_optics.silk", effective.special_optics.silk.enabled, base.special_optics.silk.enabled),
            ("special_optics.pressure", effective.special_optics.pressure.enabled, base.special_optics.pressure.enabled),
        ];
        for (path, value, inherited) in feature_values {
            if value != inherited || self.features.contains_key(&FeaturePath::from(path)) {
                feature(&mut self, path, value);
            }
        }
        for (path, value, inherited) in [
            ("base.diffuse_weight", &effective.base.values.diffuse_weight, &base.base.values.diffuse_weight),
            ("transvolume.weight", &effective.transvolume.values.weight, &base.transvolume.values.weight),
        ] {
            if value != inherited || self.properties.contains_key(&PropertyPath::from(path)) {
                scalar_override(&mut self, path, value.clone());
            }
        }
        Ok(self)
    }
}

fn invalid(name: &str, reason: impl Into<String>) -> SchemaError {
    SchemaError::new(
        format!("legacy.{name}"),
        ErrorKind::InvalidLegacySetting {
            name: name.into(),
            reason: reason.into(),
        },
    )
}

fn absorption_override(patch: &mut MaterialPatch, absorb: [f64; 3]) {
    for axis in 0..3 {
        scalar(patch, &format!("transvolume.absorption_coefficients.{}", channel_name(axis)), absorb[axis]);
        scalar(patch, &format!("transvolume.color.{}", channel_name(axis)), (-absorb[axis]).exp());
    }
}

fn scalar(patch: &mut MaterialPatch, path: &str, value: f64) {
    scalar_override(patch, path, value.into());
}
fn feature(patch: &mut MaterialPatch, path: &str, enabled: bool) {
    feature_override(patch, path, enabled);
}
fn channel_name(axis: usize) -> &'static str {
    ["r", "g", "b"][axis]
}

fn legacy_patch_value(
    patch: &mut MaterialPatch,
    base: &MaterialSpec,
    name: &str,
    value: f64,
) -> Result<(), SchemaError> {
    if !value.is_finite() {
        return Err(invalid(name, "value must be finite"));
    }
    macro_rules! simple {
        ($($legacy:literal => $path:literal),+ $(,)?) => { match name { $($legacy => { scalar(patch, $path, value); return Ok(()); })+ _ => {} } };
    }
    simple!(
        "density" => "transvolume.absorption_scale",
        "rough" => "specular.roughness",
        "filmIor" => "film.ior", "filmVar" => "film.thickness_variation", "filmScale" => "film.pattern_scale",
        "filmFlow" => "film.flow", "filmDrain" => "film.drainage", "filmSky" => "film.ambient_weight",
        "coatIor" => "coat.ior", "sssDepth" => "subsurface.mean_free_path", "sssSpread" => "subsurface.pigment.spectral_spread",
        "sssHue" => "subsurface.pigment.near_surface_hue", "sssThrough" => "subsurface.pigment.unscattered_hue",
        "sssOpaque" => "subsurface.pigment.opaque_exit",
        "sssForward" => "subsurface.phase_anisotropy", "ceramicThrough" => "base.ceramic.transmitted_fraction",
        "age" => "wear.age", "edgeWear" => "wear.edge_wear", "cavity" => "wear.cavity", "scratches" => "wear.scratches",
        "swirls" => "wear.swirls", "peel" => "wear.orange_peel", "smudges" => "wear.smudges", "dust" => "wear.dust",
        "brush" => "wear.brushing", "mottle" => "wear.mottle", "bubbleSize" => "special_optics.bubbles.size",
        "bubbleEven" => "special_optics.bubbles.evenness", "bubbleShape" => "special_optics.bubbles.shape", "bubbleSeed" => "special_optics.bubbles.seed",
        "glowCore" => "emission.volume.core_concentration", "grinDepth" => "special_optics.grin.depth",
        "gratingPitch" => "special_optics.grating.pitch_nm", "gratingRadial" => "special_optics.grating.radial",
        "gradOffset" => "special_optics.gradient.offset", "gradSoft" => "special_optics.gradient.softness",
        "gradBands" => "special_optics.gradient.bands", "gradWave" => "special_optics.gradient.wave",
        "crackSize" => "special_optics.cracks.size", "crackDepth" => "special_optics.cracks.depth", "crackOpen" => "special_optics.cracks.openness",
        "crackGrow" => "special_optics.cracks.growth", "crackCurve" => "special_optics.cracks.curvature", "crackRough" => "special_optics.cracks.roughness",
        "crackSeed" => "special_optics.cracks.seed", "playSize" => "special_optics.photonic.size", "playColour" => "special_optics.photonic.longest_wavelength",
        "playSoft" => "special_optics.photonic.softness", "playRoll" => "special_optics.photonic.roll", "playSeed" => "special_optics.photonic.seed",
        "silkSoft" => "special_optics.silk.softness", "silkWave" => "special_optics.silk.wave", "piezoColour" => "special_optics.pressure.color_bias"
    );
    match name {
        "ior" => {
            scalar(patch, "specular.ior", value);
            scalar(patch, "transvolume.ior", value);
        }
        "disp" => scalar(patch, "transvolume.dispersion.cauchy_amount", value),
        "coat" => {
            scalar(patch, "coat.weight", value);
            feature(
                patch,
                "coat",
                value > 0.0 && !matches!(base.transvolume.values.wall, WallMode::ThinWalled { .. }),
            );
        }
        "film" => {
            scalar(patch, "film.thickness_nm", value);
            scalar(patch, "film.weight", if value > 0.0 { 1.0 } else { 0.0 });
            feature(patch, "film", value > 0.0);
        }
        "metal" => {
            let mut legacy = Settings::default();
            legacy.metal = value;
            let body = match patch.properties.get(&PropertyPath::from("base.color")) {
                Some(PropertyValue::Color(Input::Constant(color))) => *color,
                _ => base
                    .base
                    .values
                    .color
                    .as_constant()
                    .copied()
                    .unwrap_or(Color::WHITE),
            };
            let conductor = match material::metal_of(&legacy) {
                Some(metal) if value.round() >= 2.0 => ConductorModel::Measured {
                    id: ConductorId::from(format!("measured:{}", metal.id)),
                },
                _ => ConductorModel::Ideal { tint: body.into() },
            };
            scalar(
                patch,
                "base.metalness",
                if value >= 0.5 { 1.0 } else { 0.0 },
            );
            conductor_override(patch, conductor);
            feature(patch, "transvolume", value < 0.5);
            if value >= 0.5 {
                feature(patch, "subsurface", false);
            }
        }
        "glazeHaze" => scalar(patch, "coat.roughness", value.clamp(0.0, 0.6).sqrt()),
        "brushAngle" => scalar(patch, "wear.brushing_angle_radians", value.to_radians()),
        "gratingAngle" => scalar(
            patch,
            "special_optics.grating.angle_radians",
            value.to_radians(),
        ),
        "birefAz" => scalar(
            patch,
            "special_optics.birefringence.azimuth_radians",
            value.to_radians(),
        ),
        "birefEl" => scalar(
            patch,
            "special_optics.birefringence.elevation_radians",
            value.to_radians(),
        ),
        "gradAz" => scalar(
            patch,
            "special_optics.gradient.azimuth_radians",
            value.to_radians(),
        ),
        "gradEl" => scalar(
            patch,
            "special_optics.gradient.elevation_radians",
            value.to_radians(),
        ),
        "silkAz" => scalar(
            patch,
            "special_optics.silk.azimuth_radians",
            value.to_radians(),
        ),
        "silkEl" => scalar(
            patch,
            "special_optics.silk.elevation_radians",
            value.to_radians(),
        ),
        "roughRef" => {
            scalar(patch, "compatibility.legacy.rough_sample_share", value);
            patch.properties.insert(
                PropertyPath::from("transvolume.rough_estimator"),
                PropertyValue::RoughTransmissionEstimator(if value >= 1.0 {
                    RoughTransmissionEstimator::Sampled
                } else if value > 0.0 {
                    RoughTransmissionEstimator::Hybrid {
                        sampled_share: value,
                    }
                } else {
                    RoughTransmissionEstimator::Fast
                }),
            );
        }
        "sharp" => scalar(patch, "compatibility.legacy.sharp_environment", value),
        "fit" => scalar(patch, "compatibility.legacy.fit_distances_to_shape", value),
        "spectrum" => {
            let mut legacy = Settings::default();
            legacy.spectrum = value;
            let spectrum = material::spectrum_of(&legacy).map(|s| s.id.into());
            if value >= 0.5 && spectrum.is_none() {
                return Err(invalid(name, "unknown measured spectrum index"));
            }
            patch.properties.insert(
                PropertyPath::from("transvolume.measured_spectrum"),
                PropertyValue::Spectrum(spectrum),
            );
        }
        "hollow" => {
            patch.properties.insert(
                PropertyPath::from("transvolume.wall"),
                PropertyValue::Wall(if value > 0.5 {
                    WallMode::ThinWalled {
                        thickness: material::WALL.into(),
                    }
                } else {
                    WallMode::ClosedVolume
                }),
            );
            if value > 0.5 {
                feature(patch, "subsurface", false);
            }
        }
        "sssWalk" => {
            patch.properties.insert(
                PropertyPath::from("subsurface.estimator"),
                PropertyValue::ScatteringEstimator(if value > 0.5 {
                    ScatteringEstimator::RandomWalk
                } else {
                    ScatteringEstimator::FastVolume
                }),
            );
        }
        "ceramic" => feature(patch, "base.ceramic", value > 0.5),
        "gradMode" => {
            let mode = match value.round() as i64 {
                0 => GradientMode::Axial,
                1 => GradientMode::Radial,
                2 => GradientMode::Bands,
                _ => return Err(invalid(name, "expected 0, 1, or 2")),
            };
            patch.properties.insert(
                PropertyPath::from("special_optics.gradient.mode"),
                PropertyValue::GradientMode(mode),
            );
        }
        "playPattern" => {
            let pattern = match value.round() as i64 {
                0 => PhotonicPattern::Blocks,
                1 => PhotonicPattern::Pinfire,
                2 => PhotonicPattern::BroadFlash,
                3 => PhotonicPattern::Platelets,
                _ => return Err(invalid(name, "expected 0 through 3")),
            };
            patch.properties.insert(
                PropertyPath::from("special_optics.photonic.pattern"),
                PropertyValue::PhotonicPattern(pattern),
            );
        }
        "mat" => {
            return Err(invalid(
                name,
                "select the named full base material before applying a sparse patch",
            ))
        }
        "bodyR" | "bodyG" | "bodyB" => {
            let axis = ["bodyR", "bodyG", "bodyB"]
                .iter()
                .position(|key| *key == name)
                .unwrap();
            for group in [
                "base.color",
                "subsurface.target_color",
                "subsurface.scatter_color",
            ] {
                scalar(patch, &format!("{group}.{}", channel_name(axis)), value);
            }
        }
        "tintR" | "tintG" | "tintB" => {
            if !(0.0..=1.0).contains(&value) {
                return Err(invalid(name, "tint must be in 0..=1"));
            }
            let axis = ["tintR", "tintG", "tintB"]
                .iter()
                .position(|key| *key == name)
                .unwrap();
            let tint = value.clamp(1e-4, 1.0);
            let absorption = -tint.ln();
            scalar(
                patch,
                &format!("transvolume.color.{}", channel_name(axis)),
                (-absorption).exp(),
            );
            scalar(
                patch,
                &format!("transvolume.absorption_coefficients.{}", channel_name(axis)),
                absorption,
            );
        }
        "glowR" | "glowG" | "glowB" => {
            let axis = ["glowR", "glowG", "glowB"]
                .iter()
                .position(|key| *key == name)
                .unwrap();
            scalar(
                patch,
                &format!("emission.color.{}", channel_name(axis)),
                value,
            );
        }
        "cavityR" | "cavityG" | "cavityB" => {
            let axis = ["cavityR", "cavityG", "cavityB"]
                .iter()
                .position(|key| *key == name)
                .unwrap();
            scalar(
                patch,
                &format!("wear.cavity_color.{}", channel_name(axis)),
                value,
            );
        }
        "gradR" | "gradG" | "gradB" => {
            let axis = ["gradR", "gradG", "gradB"]
                .iter()
                .position(|key| *key == name)
                .unwrap();
            scalar(
                patch,
                &format!("special_optics.gradient.target_tint.{}", channel_name(axis)),
                value,
            );
        }
        "glintR" | "glintG" | "glintB" => {
            let axis = ["glintR", "glintG", "glintB"]
                .iter()
                .position(|key| *key == name)
                .unwrap();
            scalar(
                patch,
                &format!(
                    "special_optics.photonic.platelet_color.{}",
                    channel_name(axis)
                ),
                value,
            );
        }
        "sss" => {
            scalar(patch, "subsurface.weight", value);
            feature(patch, "subsurface", value > 0.0);
            scalar(patch, "transvolume.weight", 1.0 - value.clamp(0.0, 1.0));
        }
        "glow" => {
            scalar(patch, "emission.volume.strength", value);
            feature(patch, "emission", value > 0.0);
        }
        "grating" => {
            scalar(patch, "special_optics.grating.strength", value);
            feature(patch, "special_optics.grating", value > 0.0);
        }
        "biref" => {
            scalar(patch, "special_optics.birefringence.delta_ior", value);
            feature(patch, "special_optics.birefringence", value != 0.0);
        }
        "grin" => {
            scalar(patch, "special_optics.grin.strength", value);
            feature(patch, "special_optics.grin", value != 0.0);
        }
        "bubbles" => {
            scalar(patch, "special_optics.bubbles.amount", value);
            feature(patch, "special_optics.bubbles", value > 0.0);
        }
        "grad" => {
            scalar(patch, "special_optics.gradient.amount", value);
            feature(patch, "special_optics.gradient", value > 0.0);
        }
        "cracks" | "fractures" => {
            scalar(
                patch,
                if name == "cracks" {
                    "special_optics.cracks.amount"
                } else {
                    "special_optics.cracks.fractures"
                },
                value,
            );
            let other = if name == "cracks" {
                base.special_optics.cracks.values.fractures.as_constant()
            } else {
                base.special_optics.cracks.values.amount.as_constant()
            };
            feature(
                patch,
                "special_optics.cracks",
                value > 0.0 || other.is_some_and(|other| *other > 0.0),
            );
        }
        "play" => {
            scalar(patch, "special_optics.photonic.amount", value);
            feature(patch, "special_optics.photonic", value > 0.0);
        }
        "silk" => {
            scalar(patch, "special_optics.silk.amount", value);
            feature(patch, "special_optics.silk", value > 0.0);
        }
        "piezo" => {
            scalar(patch, "special_optics.pressure.amount", value);
            feature(patch, "special_optics.pressure", value > 0.0);
        }
        _ => return Err(invalid(name, "this key is not a legacy material override")),
    }
    Ok(())
}

/// Convenience for a full owned body recipe with the current schema version.
pub fn full_body_from_legacy(settings: &Settings, id: MaterialId) -> MaterialRecipe {
    MaterialRecipe::Full(MaterialSpec::from_legacy(settings, id))
}

/// Check both version and finite/ranged values at integration boundaries.
pub fn validate_legacy_conversion(spec: &MaterialSpec) -> Result<(), SchemaError> {
    if spec.schema_version != MATERIAL_SCHEMA_VERSION {
        return Err(SchemaError::new(
            "schema_version",
            ErrorKind::UnsupportedVersion {
                found: spec.schema_version,
                supported: MATERIAL_SCHEMA_VERSION,
            },
        ));
    }
    spec.validate()
}
