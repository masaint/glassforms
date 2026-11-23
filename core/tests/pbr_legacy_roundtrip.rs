//! Legacy renderer bridge tests: compare active variants and shader payloads, not JSON equality.
use gf_core::{material, pbr, settings};

use std::collections::{BTreeMap, BTreeSet};

use material::NAMED;
use pbr::*;
use settings::Settings;

fn id(value: &str) -> MaterialId {
    MaterialId::from(value)
}

fn named_settings(index: usize) -> Settings {
    let mut settings = Settings::default();
    NAMED[index].apply(&mut settings);
    settings.mat = index as f64;
    settings
}

fn assert_same_active_payload(original: &Settings, exported: &Settings) {
    let before = material::Variant::of(original);
    let after = material::Variant::of(exported);
    assert_eq!(
        after, before,
        "active variant changed\nbefore: {original:#?}\nafter: {exported:#?}"
    );
    assert_eq!(
        material::uniforms(exported, 0.73),
        material::uniforms(original, 0.73)
    );
    if before.inner.any() {
        let tint = original.absorb.map(|value| value * original.density);
        assert_eq!(
            gf_core::inner::uniforms(exported, 0.73, 0.002, tint),
            gf_core::inner::uniforms(original, 0.73, 0.002, tint),
        );
    }
}

#[test]
fn every_named_recipe_exports_same_active_variant_and_payload() {
    for (index, named) in NAMED.iter().enumerate() {
        let original = named_settings(index);
        let spec = MaterialSpec::from_legacy(&original, id(&format!("material:{}", named.id)));
        let exported = spec
            .to_legacy_settings()
            .unwrap_or_else(|error| panic!("{}: {error}", named.id));
        assert_eq!(
            exported.mat, index as f64,
            "{} lost stable preset identity",
            named.id
        );
        assert_same_active_payload(&original, &exported);
    }
}

#[test]
fn representative_custom_recipe_exports_same_active_payload() {
    let mut original = Settings::default();
    original.absorb = [0.12, 0.24, 0.36];
    for (key, value) in [
        ("ior", 1.61),
        ("disp", 0.023),
        ("density", 1.7),
        ("spectrum", 3.0),
        ("rough", 0.31),
        ("roughRef", 0.27),
        ("fit", 1.0),
        ("sharp", 0.0),
        ("coat", 0.72),
        ("coatIor", 1.57),
        ("film", 280.0),
        ("filmIor", 1.42),
        ("filmVar", 0.22),
        ("filmScale", 1.3),
        ("filmFlow", 0.4),
        ("filmDrain", 0.18),
        ("bodyR", 0.32),
        ("bodyG", 0.48),
        ("bodyB", 0.59),
        ("sss", 0.65),
        ("sssDepth", 0.23),
        ("sssSpread", 0.31),
        ("sssHue", 0.42),
        ("sssForward", 0.36),
        ("sssThrough", 0.28),
        ("sssOpaque", 0.62),
        ("ceramic", 1.0),
        ("ceramicThrough", 0.21),
        ("glazeHaze", 0.16),
        ("age", 0.8),
        ("edgeWear", 0.2),
        ("cavity", 0.3),
        ("cavityR", 0.2),
        ("scratches", 0.14),
        ("swirls", 0.15),
        ("peel", 0.16),
        ("smudges", 0.17),
        ("dust", 0.18),
        ("brush", 0.4),
        ("brushAngle", 23.0),
        ("mottle", 0.21),
        ("grating", 0.33),
        ("gratingPitch", 1450.0),
        ("gratingAngle", 31.0),
        ("gratingRadial", 0.4),
        ("bubbles", 0.2),
        ("bubbleSize", 0.04),
        ("bubbleEven", 0.6),
        ("bubbleShape", 0.2),
        ("glow", 0.35),
        ("glowR", 0.8),
        ("glowG", 0.4),
        ("glowB", 0.2),
        ("glowCore", 0.6),
        ("grad", 0.4),
        ("gradMode", 2.0),
        ("gradR", 0.4),
        ("gradG", 0.7),
        ("gradB", 0.9),
        ("gradAz", 12.0),
        ("gradEl", 45.0),
        ("gradOffset", 0.2),
        ("gradSoft", 0.5),
        ("gradBands", 7.0),
        ("gradWave", 0.25),
        ("cracks", 0.25),
        ("fractures", 0.12),
        ("crackSize", 0.3),
        ("crackDepth", 0.8),
        ("crackOpen", 0.7),
        ("crackGrow", 0.9),
        ("crackCurve", 0.4),
        ("crackRough", 0.2),
        ("play", 0.3),
        ("playSize", 0.15),
        ("playPattern", 3.0),
        ("playColour", 0.7),
        ("playSoft", 0.25),
        ("playRoll", 0.2),
        ("glintR", 0.9),
        ("glintG", 0.8),
        ("silk", 0.25),
        ("silkAz", 22.0),
        ("silkEl", 33.0),
        ("silkSoft", 0.4),
        ("silkWave", 0.2),
        ("piezo", 0.2),
        ("piezoColour", 0.6),
    ] {
        assert!(original.set(key, value), "{key}");
    }
    let spec = MaterialSpec::from_legacy(&original, id("material:custom"));
    let exported = spec.to_legacy_settings().unwrap();
    assert_same_active_payload(&original, &exported);

    let mut walk = Settings::default();
    walk.sss = 1.0;
    walk.sss_walk = 1.0;
    walk.biref = 0.2;
    walk.biref_az = 32.0;
    walk.grin = 0.3;
    walk.grin_depth = 0.04;
    let spec = MaterialSpec::from_legacy(&walk, id("material:walk"));
    let exported = spec.to_legacy_settings().unwrap();
    assert!(!spec.special_optics.birefringence.enabled && !spec.special_optics.grin.enabled);
    assert_eq!(
        exported.biref, 0.2,
        "dormant authored birefringence was discarded"
    );
    assert_eq!(exported.grin, 0.3, "dormant authored GRIN was discarded");
    assert_same_active_payload(&walk, &exported);
}

#[test]
fn sparse_export_emits_only_overridden_keys() {
    let base = MaterialSpec::from_legacy(
        &named_settings(material::named_index("crown").unwrap()),
        id("material:base"),
    );
    let original = BTreeMap::from([
        ("rough".to_string(), 0.35),
        ("roughRef".to_string(), 0.25),
        ("coat".to_string(), 0.8),
        ("coatIor".to_string(), 1.57),
        ("tintG".to_string(), 0.5),
        ("bodyR".to_string(), 0.2),
    ]);
    let patch = MaterialPatch::from_legacy_map(&base, &original).unwrap();
    let exported = patch.to_legacy_map(&base).unwrap();
    assert_eq!(
        exported.keys().collect::<BTreeSet<_>>(),
        original.keys().collect::<BTreeSet<_>>()
    );
    for (key, value) in original {
        assert!(
            (exported[&key] - value).abs() < 1e-12,
            "{key}: {} != {value}",
            exported[&key]
        );
    }

    let patch =
        MaterialPatch::from_legacy_map(&base, &BTreeMap::from([("rough".into(), 0.4)])).unwrap();
    let exported = patch.to_legacy_map(&base).unwrap();
    assert_eq!(exported, BTreeMap::from([("rough".into(), 0.4)]));
}

#[test]
fn sparse_feature_disable_emits_only_the_legacy_activator() {
    let mut settings = Settings::default();
    settings.coat = 0.8;
    let base = MaterialSpec::from_legacy(&settings, id("material:coated"));
    let mut patch = MaterialPatch::new(base.id.clone());
    feature_override(&mut patch, "coat", false);
    let exported = patch.to_legacy_map(&base).unwrap();
    assert_eq!(exported, BTreeMap::from([("coat".into(), 0.0)]));

    let resolved = patch.apply(&base).unwrap();
    let mut effective = base.to_legacy_settings().unwrap();
    for (key, value) in exported {
        assert!(effective.set(&key, value));
    }
    assert_same_active_payload(&effective, &resolved.to_legacy_settings().unwrap());
}

#[test]
fn every_sparse_legacy_key_exports_without_extra_ownership() {
    let base = MaterialSpec::default();
    let base_settings = base.to_legacy_settings().unwrap();
    for (name, value) in material::RESET.iter().chain(gf_core::inner::RESET) {
        let original = BTreeMap::from([((*name).to_string(), *value)]);
        let patch = MaterialPatch::from_legacy_map(&base, &original)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let exported = patch
            .to_legacy_map(&base)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert!(
            exported.keys().all(|key| key == name),
            "{name} emitted unrelated keys: {exported:?}"
        );
        let resolved = patch.apply(&base).unwrap();
        let mut through_map = base_settings.clone();
        for (key, value) in exported {
            assert!(through_map.set(&key, value));
        }
        assert_same_active_payload(&through_map, &resolved.to_legacy_settings().unwrap());
    }
}

#[test]
fn fractional_metalness_is_a_specific_export_error() {
    let spec = MaterialSpec::from_legacy_with_metalness(
        &Settings::default(),
        id("material:fractional"),
        0.37,
    )
    .unwrap();
    assert!(matches!(
        spec.to_legacy_settings(),
        Err(LegacyExportError { kind: LegacyExportErrorKind::FractionalMetalness { value }, .. }) if value == 0.37
    ));

    let base = MaterialSpec::default();
    let mut patch = MaterialPatch::new(base.id.clone());
    scalar_override(&mut patch, "base.metalness", 0.25.into());
    assert!(matches!(
        patch.to_legacy_map(&base),
        Err(LegacyExportError { kind: LegacyExportErrorKind::FractionalMetalness { value }, .. }) if value == 0.25
    ));
}

#[test]
fn nonconstant_and_canonical_only_features_are_not_silently_dropped() {
    let mut spec = MaterialSpec::default();
    spec.specular.values.roughness = Input::Animated(AnimatedInput {
        id: TrackId::from("track:rough"),
        fallback: 0.2,
        interpolation: Interpolation::Linear,
        keys: vec![AnimationKey {
            time_seconds: 0.0,
            value: 0.2,
        }],
    });
    assert!(matches!(
        spec.to_legacy_settings(),
        Err(LegacyExportError {
            kind: LegacyExportErrorKind::NonConstantInput,
            ..
        })
    ));

    let mut spec = MaterialSpec::default();
    spec.sheen.enabled = true;
    assert!(matches!(
        spec.to_legacy_settings(),
        Err(LegacyExportError {
            kind: LegacyExportErrorKind::UnsupportedFeature { .. },
            ..
        })
    ));

    let base = MaterialSpec::default();
    let mut patch = MaterialPatch::new(base.id.clone());
    feature_override(&mut patch, "coverage", true);
    assert!(matches!(
        patch.to_legacy_map(&base),
        Err(LegacyExportError {
            kind: LegacyExportErrorKind::UnsupportedFeature { .. },
            ..
        })
    ));
}

#[test]
fn conflicting_aliases_and_unknown_optical_models_are_errors() {
    let mut spec = MaterialSpec::default();
    spec.specular.values.ior = 1.7.into();
    assert!(matches!(
        spec.to_legacy_settings(),
        Err(LegacyExportError {
            kind: LegacyExportErrorKind::ConflictingValues { .. },
            ..
        })
    ));

    let mut fitting = MaterialSpec::default();
    fitting.emission.values.volume.distance_fitted = true;
    assert!(matches!(
        fitting.to_legacy_settings(),
        Err(LegacyExportError {
            kind: LegacyExportErrorKind::ConflictingValues { .. },
            ..
        })
    ));

    let mut spec = MaterialSpec::default();
    spec.base.values.metalness = 1.0.into();
    spec.transvolume.enabled = false;
    spec.base.values.conductor = ConductorModel::Measured {
        id: ConductorId::from("measured:unobtainium"),
    };
    assert!(matches!(
        spec.to_legacy_settings(),
        Err(LegacyExportError {
            kind: LegacyExportErrorKind::UnknownMeasuredConductor(_),
            ..
        })
    ));

    spec.base.values.conductor = ConductorModel::CustomSpectral {
        samples: vec![
            ComplexIndexSample {
                wavelength_nm: 450.0,
                n: 0.2,
                k: 2.0,
            },
            ComplexIndexSample {
                wavelength_nm: 650.0,
                n: 0.3,
                k: 2.5,
            },
        ],
    };
    assert!(matches!(
        spec.to_legacy_settings(),
        Err(LegacyExportError {
            kind: LegacyExportErrorKind::UnsupportedFeature { .. },
            ..
        })
    ));
}
