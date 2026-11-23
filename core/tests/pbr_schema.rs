//! CPU-only contract tests for the canonical schema exported as `gf_core::pbr`.
use gf_core::{material, pbr, settings};

use std::collections::BTreeMap;

use material::{NAMED, RESET};
use pbr::*;
use settings::Settings;

fn id(value: &str) -> MaterialId {
    MaterialId::from(value)
}
fn constant(input: &ScalarInput) -> f64 {
    *input.as_constant().unwrap()
}
fn color(input: &ColorInput) -> Color {
    *input.as_constant().unwrap()
}

fn named_settings(name: &str) -> Settings {
    let index = material::named_index(name).unwrap();
    let mut settings = Settings::default();
    NAMED[index].apply(&mut settings);
    settings.mat = index as f64;
    settings
}

fn scalar_field_graph() -> FieldGraph {
    FieldGraph {
        nodes: BTreeMap::from([
            (FieldId::from("uv"), FieldNode::Builtin(BuiltinField::Uv)),
            (
                FieldId::from("noise"),
                FieldNode::Noise {
                    kind: NoiseKind::Ridged,
                    position: FieldId::from("uv"),
                    scale: 4.0,
                    seed: 17,
                },
            ),
        ]),
    }
}

#[test]
fn defaults_are_versioned_clear_glass_with_dormant_groups() {
    let spec = MaterialSpec::default();
    assert_eq!(spec.schema_version, MATERIAL_SCHEMA_VERSION);
    assert_eq!(spec.id.as_str(), "material:default");
    assert!(spec.base.enabled && spec.specular.enabled && spec.transvolume.enabled);
    for enabled in [
        spec.coat.enabled,
        spec.sheen.enabled,
        spec.film.enabled,
        spec.subsurface.enabled,
        spec.emission.enabled,
        spec.coverage.enabled,
        spec.geometry.enabled,
        spec.wear.enabled,
    ] {
        assert!(!enabled);
    }
    assert_eq!(constant(&spec.base.values.metalness), 0.0);
    assert_eq!(constant(&spec.base.values.weight), 1.0);
    assert_eq!(constant(&spec.base.values.diffuse_weight), 0.0);
    assert_eq!(constant(&spec.transvolume.values.weight), 1.0);
    assert_eq!(
        color(&spec.transvolume.values.absorption_coefficients),
        Color::BLACK
    );
    assert!(spec.fields.nodes.is_empty() && spec.textures.is_empty());
    spec.validate().unwrap();
}

#[test]
fn compiled_media_use_sorted_shared_ids_and_signed_priorities() {
    let medium = |id: &str, priority| MediumDescriptor {
        id: MediumId::from(id),
        interface_priority: priority,
    };
    let scene = medium("medium:m", 6);
    let z = medium("medium:z", 19);
    let a = medium("medium:a", -23);
    let compiled = CompiledMedia::from_descriptors(&scene, &[z.clone(), a.clone(), z.clone()]);
    assert_eq!(compiled.scene, CompiledMedium { identity: 2, interface_priority: 6 });
    assert_eq!(compiled.bodies[0], CompiledMedium { identity: 3, interface_priority: 19 });
    assert_eq!(compiled.bodies[1], CompiledMedium { identity: 1, interface_priority: -23 });
    assert_eq!(compiled.bodies[2].identity, 3);
    assert_eq!(compiled.bodies[2].interface_priority, 19);

    let reordered = CompiledMedia::from_descriptors(&scene, &[a, z.clone(), z]);
    assert_eq!(reordered.scene, compiled.scene);
    assert_eq!(reordered.bodies[0].identity, 1);
    assert_eq!(reordered.bodies[1].identity, 3);
    assert_eq!(reordered.bodies[2].identity, 3);

    let legacy = MaterialSpec::from_legacy(&Settings::default(), id("material:body:2"));
    assert_eq!(legacy.transvolume.values.medium.id.as_str(), "medium:material:body:2");
    assert_eq!(legacy.transvolume.values.medium.interface_priority, 0);
    assert_eq!(legacy_medium_id(&legacy.id), legacy.transvolume.values.medium.id);
}

#[test]
fn medium_ownership_uses_priority_then_body_index_without_using_identity() {
    let medium = |identity, interface_priority| CompiledMedium { identity, interface_priority };
    let media = [medium(7, 10), medium(7, 0), medium(9, -4), medium(9, 10)];

    // The lower body index wins when its signed priority is higher, even with a shared MediumId.
    assert_eq!(containing_body([true, true, false, false], &media), Some(0));
    assert!(visible_body(0, Some(1), &media));
    assert!(!visible_body(1, Some(0), &media));
    assert_eq!(interface_owner(Some(1), Some(0), &media), Some(0));
    assert_eq!(interface_owner(Some(0), Some(1), &media), Some(0));

    // Entering body 0 and leaving it back into still-containing body 1 are dual transitions.
    assert_eq!(exit_destination([true, true, false, false], 1, &media), Some(0));
    assert_eq!(exit_destination([true, true, false, false], 0, &media), Some(1));

    // Equal priorities retain the highest-index rule, including an all-zero negative control.
    assert_eq!(containing_body([true, false, false, true], &media), Some(3));
    let zero = [medium(1, 0), medium(2, 0), medium(3, 0), medium(4, 0)];
    assert_eq!(containing_body([true, true, true, true], &zero), Some(3));
    assert!(visible_body(1, Some(0), &zero));
    assert!(!visible_body(0, Some(1), &zero));
    assert_eq!(exit_destination([true, true, false, false], 1, &zero), Some(0));
    assert_eq!(interface_owner(Some(0), Some(1), &zero), Some(1));

    // A negative authored priority still beats air; air is not a zero-priority body.
    assert_eq!(containing_body([false, false, true, false], &media), Some(2));
    assert!(visible_body(2, None, &media));
    assert_eq!(interface_owner(None, Some(2), &media), Some(2));
    assert_eq!(interface_owner(Some(2), None, &media), Some(2));
    assert_eq!(containing_body([false; 4], &media), None);
}

#[test]
fn disabling_preserves_the_complete_dormant_payload() {
    let mut spec = MaterialSpec::default();
    spec.coat.values.weight = 0.73.into();
    spec.coat.values.roughness = 0.28.into();
    spec.coat.values.absorption_color = Color([0.8, 0.9, 1.0]).into();
    spec.coat.enabled = true;
    let authored = spec.coat.values.clone();
    spec.coat.enabled = false;
    spec.validate().unwrap();
    assert_eq!(spec.coat.values, authored);
    let text = serde_json::to_string(&spec).unwrap();
    let mut reopened: MaterialSpec = serde_json::from_str(&text).unwrap();
    assert!(!reopened.coat.enabled);
    assert_eq!(reopened.coat.values, authored);
    reopened.coat.enabled = true;
    assert_eq!(reopened.coat.values, authored);
}

#[test]
fn constant_animated_field_and_texture_inputs_are_typed() {
    let mut spec = MaterialSpec::default();
    spec.fields = scalar_field_graph();
    spec.base.values.metalness = Input::Field(FieldInput {
        field: FieldId::from("noise"),
        fallback: 0.25,
        mapping: Some(Affine {
            scale: 0.5,
            offset: 0.1,
        }),
        remap: Some(Remap {
            from: [0.0, 1.0],
            to: [0.1, 0.8],
            clamp: true,
        }),
    });
    spec.specular.values.roughness = Input::Animated(AnimatedInput {
        id: TrackId::from("track:roughness"),
        fallback: 0.2,
        interpolation: Interpolation::Linear,
        keys: vec![
            AnimationKey {
                time_seconds: 0.0,
                value: 0.2,
            },
            AnimationKey {
                time_seconds: 2.0,
                value: 0.7,
            },
        ],
    });
    spec.textures.insert(
        TextureId::from("texture:base"),
        TextureResource {
            uri: "textures/base.png".into(),
            color_space: TextureColorSpace::Srgb,
            sampling: TextureSampling {
                projection: Projection::Triplanar,
                ..TextureSampling::default()
            },
        },
    );
    spec.base.values.color = Input::Texture(TextureInput {
        texture: TextureId::from("texture:base"),
        fallback: Color([0.2, 0.3, 0.4]),
        channels: ChannelSelector::Rgb,
        mapping: None,
        remap: None,
    });
    spec.validate().unwrap();
    assert_eq!(spec.base.values.metalness.value_type(), ValueType::Scalar);
    assert_eq!(spec.base.values.color.value_type(), ValueType::Color);
    let json = serde_json::to_value(&spec).unwrap();
    assert_eq!(json["base"]["values"]["metalness"]["source"], "field");
    assert_eq!(json["base"]["values"]["color"]["source"], "texture");
    assert_eq!(
        json["specular"]["values"]["roughness"]["source"],
        "animated"
    );
    let decoded: MaterialSpec = serde_json::from_value(json).unwrap();
    assert_eq!(decoded, spec);
}

#[test]
fn continuous_metalness_is_not_quantised() {
    let settings = Settings::default();
    for metalness in [0.0, 0.01, 0.37, 0.5, 0.99, 1.0] {
        let spec =
            MaterialSpec::from_legacy_with_metalness(&settings, id("material:import"), metalness)
                .unwrap();
        assert_eq!(constant(&spec.base.values.metalness), metalness);
        let text = serde_json::to_string(&spec).unwrap();
        let reopened: MaterialSpec = serde_json::from_str(&text).unwrap();
        assert_eq!(constant(&reopened.base.values.metalness), metalness);
        reopened.validate().unwrap();
    }
    assert!(MaterialSpec::from_legacy_with_metalness(&settings, id("bad"), 1.1).is_err());
    assert!(MaterialSpec::from_legacy_with_metalness(&settings, id("bad"), f64::NAN).is_err());
}

#[test]
fn categorical_legacy_metal_maps_to_ideal_or_measured_conductor() {
    let dielectric = MaterialSpec::from_legacy(&Settings::default(), id("material:glass"));
    assert_eq!(constant(&dielectric.base.values.metalness), 0.0);
    assert!(dielectric.transvolume.enabled);

    let mirror = MaterialSpec::from_legacy(&named_settings("mirror"), id("material:mirror"));
    assert_eq!(constant(&mirror.base.values.metalness), 1.0);
    assert!(!mirror.transvolume.enabled);
    assert!(matches!(
        mirror.base.values.conductor,
        ConductorModel::Ideal { .. }
    ));

    let gold = MaterialSpec::from_legacy(&named_settings("gold"), id("material:gold-copy"));
    assert_eq!(constant(&gold.base.values.metalness), 1.0);
    match gold.base.values.conductor {
        ConductorModel::Measured { id: conductor } => {
            assert_eq!(conductor.as_str(), "measured:gold")
        }
        ref other => panic!("gold lost its measured identity: {other:?}"),
    }
    assert_eq!(gold.preset.as_ref().unwrap().as_str(), "gold");
}

#[test]
fn every_named_material_converts_deterministically_and_finitely() {
    for (index, named) in NAMED.iter().enumerate() {
        let mut settings = Settings::default();
        named.apply(&mut settings);
        settings.mat = index as f64;
        let a = MaterialSpec::from_legacy(&settings, id(&format!("material:{}", named.id)));
        let b = MaterialSpec::from_legacy(&settings, id(&format!("material:{}", named.id)));
        assert_eq!(a, b, "{} conversion is not deterministic", named.id);
        validate_legacy_conversion(&a).unwrap_or_else(|error| panic!("{}: {error}", named.id));
        assert_eq!(a.preset.as_ref().unwrap().as_str(), named.id);
        let text = serde_json::to_string(&a).unwrap();
        let reopened: MaterialSpec = serde_json::from_str(&text).unwrap();
        assert_eq!(reopened, a, "{} did not round-trip", named.id);
    }
}

#[test]
fn legacy_tint_roughness_coat_and_haze_have_explicit_conventions() {
    let mut settings = Settings::default();
    settings.absorb = [0.1, 0.5, 1.2];
    settings.density = 2.5;
    settings.rough = 0.4;
    settings.coat = 0.75;
    settings.coat_ior = 1.62;
    let spec = MaterialSpec::from_legacy(&settings, id("material:legacy"));
    assert_eq!(
        color(&spec.transvolume.values.absorption_coefficients),
        Color([0.1, 0.5, 1.2])
    );
    assert_eq!(
        color(&spec.transvolume.values.color),
        Color([
            0.1f64.exp().recip(),
            0.5f64.exp().recip(),
            1.2f64.exp().recip()
        ])
    );
    assert_eq!(constant(&spec.transvolume.values.absorption_scale), 2.5);
    assert_eq!(constant(&spec.specular.values.roughness), 0.4);
    assert_eq!(constant(&spec.coat.values.weight), 0.75);
    assert_eq!(constant(&spec.coat.values.ior), 1.62);
    assert_eq!(constant(&spec.coat.values.roughness), 0.0);
    assert_eq!(
        spec.compatibility.legacy.as_ref().unwrap().roughness,
        LegacyRoughnessConvention::PerceptualSquaredGgx
    );

    let mut ceramic = named_settings("porcelain");
    ceramic.glaze_haze = 0.16;
    let spec = MaterialSpec::from_legacy(&ceramic, id("material:ceramic"));
    assert!((constant(&spec.coat.values.roughness) - 0.4).abs() < 1e-12);
    assert_eq!(
        spec.compatibility.legacy.unwrap().coat_roughness,
        LegacyCoatRoughnessConvention::GlazeHazeIsGgxAlpha
    );
}

#[test]
fn legacy_walk_birefringence_grin_priority_retains_dormant_values() {
    let mut legacy = Settings::default();
    legacy.sss = 1.0;
    legacy.sss_walk = 1.0;
    legacy.biref = 0.2;
    legacy.grin = 0.3;
    let spec = MaterialSpec::from_legacy(&legacy, id("material:walk-priority"));
    spec.validate().unwrap();
    assert!(spec.subsurface.enabled);
    assert_eq!(
        spec.subsurface.values.estimator,
        ScatteringEstimator::RandomWalk
    );
    assert!(!spec.special_optics.birefringence.enabled);
    assert!(!spec.special_optics.grin.enabled);
    assert_eq!(
        constant(&spec.special_optics.birefringence.values.delta_ior),
        0.2
    );
    assert_eq!(constant(&spec.special_optics.grin.values.strength), 0.3);
    assert!(spec
        .compatibility
        .notices
        .iter()
        .any(|notice| notice.contains("birefringence")));
    assert!(spec
        .compatibility
        .notices
        .iter()
        .any(|notice| notice.contains("GRIN")));
    let text = serde_json::to_string(&spec).unwrap();
    let reopened: MaterialSpec = serde_json::from_str(&text).unwrap();
    assert_eq!(reopened, spec);

    legacy.sss_walk = 0.0;
    let spec = MaterialSpec::from_legacy(&legacy, id("material:birefringence-priority"));
    spec.validate().unwrap();
    assert!(spec.special_optics.birefringence.enabled);
    assert!(!spec.special_optics.grin.enabled);
    assert_eq!(constant(&spec.special_optics.grin.values.strength), 0.3);
    assert!(spec
        .compatibility
        .notices
        .iter()
        .any(|notice| notice.contains("GRIN")));

    legacy.biref = 0.0;
    let spec = MaterialSpec::from_legacy(&legacy, id("material:grin"));
    spec.validate().unwrap();
    assert!(!spec.special_optics.birefringence.enabled);
    assert!(spec.special_optics.grin.enabled);
    assert!(spec.compatibility.notices.is_empty());
}

#[test]
fn full_body_and_sparse_node_have_distinct_ownership() {
    let mut base = MaterialSpec::default();
    base.id = id("material:base");
    base.specular.values.roughness = 0.15.into();
    base.coat.values.weight = 0.9.into();
    let full = full_body_from_legacy(&named_settings("gold"), id("material:owned"));
    assert_eq!(full.ownership(), MaterialOwnership::FullBody);
    let owned = full.resolve(Some(&base)).unwrap();
    assert_eq!(owned.id, id("material:owned"));
    assert_eq!(constant(&owned.base.values.metalness), 1.0);

    let mut patch = MaterialPatch::new(base.id.clone());
    scalar_override(&mut patch, "specular.roughness", 0.6.into());
    let sparse = MaterialRecipe::Sparse(patch);
    sparse.validate().unwrap();
    assert_eq!(sparse.ownership(), MaterialOwnership::SparseNode);
    let resolved = sparse.resolve(Some(&base)).unwrap();
    assert_eq!(resolved.id, base.id);
    assert_eq!(constant(&resolved.specular.values.roughness), 0.6);
    assert_eq!(constant(&resolved.coat.values.weight), 0.9);
    assert_eq!(
        constant(&base.specular.values.roughness),
        0.15,
        "resolution mutated the base"
    );
    assert!(sparse.resolve(None).is_err());
    let wrong = MaterialSpec {
        id: id("material:other"),
        ..base.clone()
    };
    assert!(matches!(
        sparse.resolve(Some(&wrong)),
        Err(SchemaError {
            kind: ErrorKind::BaseIdentity { .. },
            ..
        })
    ));
}

#[test]
fn sparse_legacy_map_is_canonical_and_does_not_own_unmentioned_values() {
    let base = MaterialSpec::from_legacy(&named_settings("crown"), id("material:base"));
    let values = BTreeMap::from([
        ("rough".to_string(), 0.35),
        ("coat".to_string(), 0.8),
        ("coatIor".to_string(), 1.57),
        ("tintG".to_string(), 0.5),
        ("bodyR".to_string(), 0.2),
    ]);
    let patch = MaterialPatch::from_legacy_map(&base, &values).unwrap();
    assert!(!patch
        .properties
        .contains_key(&PropertyPath::from("film.thickness_nm")));
    let resolved = patch.apply(&base).unwrap();
    resolved.validate().unwrap();
    assert_eq!(constant(&resolved.specular.values.roughness), 0.35);
    assert!(resolved.coat.enabled);
    assert_eq!(constant(&resolved.coat.values.ior), 1.57);
    assert!(
        (color(&resolved.transvolume.values.absorption_coefficients).0[1] - std::f64::consts::LN_2)
            .abs()
            < 1e-12
    );
    assert_eq!(color(&resolved.transvolume.values.color).0[1], 0.5);
    assert_eq!(color(&resolved.base.values.color).0[0], 0.2);
    assert_eq!(resolved.film, base.film);
    assert_eq!(resolved.emission, base.emission);
}

#[test]
fn every_existing_sparse_material_key_has_an_explicit_conversion() {
    let base = MaterialSpec::default();
    for (name, value) in RESET.iter().chain(gf_core::inner::RESET) {
        let values = BTreeMap::from([((*name).to_string(), *value)]);
        let patch = MaterialPatch::from_legacy_map(&base, &values)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let resolved = patch
            .apply(&base)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        resolved
            .validate()
            .unwrap_or_else(|error| panic!("{name}: {error}"));
    }
    let empty = MaterialPatch::from_legacy_map(&base, &BTreeMap::new()).unwrap();
    assert!(empty.is_empty());
    let unsupported = BTreeMap::from([("notASetting".to_string(), 1.0)]);
    assert!(MaterialPatch::from_legacy_map(&base, &unsupported).is_err());
    let positional = BTreeMap::from([("mat".to_string(), 2.0)]);
    assert!(MaterialPatch::from_legacy_map(&base, &positional).is_err());
}

#[test]
fn categorical_sparse_metal_switch_restores_dielectric_activity() {
    let base = MaterialSpec::from_legacy(&named_settings("gold"), id("material:gold"));
    let patch =
        MaterialPatch::from_legacy_map(&base, &BTreeMap::from([("metal".into(), 0.0)])).unwrap();
    let dielectric = patch.apply(&base).unwrap();
    assert!(dielectric.transvolume.enabled);
    assert_eq!(constant(&dielectric.base.values.metalness), 0.0);
    assert!(matches!(
        dielectric.base.values.conductor,
        ConductorModel::Ideal { .. }
    ));

    let mut milky = named_settings("porcelain");
    milky.ceramic = 0.0;
    let base = MaterialSpec::from_legacy(&milky, id("material:milky"));
    let owned =
        MaterialPatch::from_legacy_map(&base, &BTreeMap::from([("metal".into(), 1.0)])).unwrap();
    let metal = owned.apply(&base).unwrap();
    assert!(!metal.subsurface.enabled && !metal.transvolume.enabled);
    let restored = MaterialPatch::from_legacy_map(&metal, &BTreeMap::from([("metal".into(), 0.0)]))
        .unwrap()
        .apply(&metal)
        .unwrap();
    assert!(restored.subsurface.enabled && restored.transvolume.enabled);
}

#[test]
fn sparse_property_type_errors_are_reported() {
    let base = MaterialSpec::default();
    let mut patch = MaterialPatch::new(base.id.clone());
    patch.properties.insert(
        PropertyPath::from("specular.roughness"),
        PropertyValue::Color(Color::WHITE.into()),
    );
    assert!(matches!(
        patch.apply(&base),
        Err(SchemaError {
            kind: ErrorKind::PropertyType {
                expected: "scalar",
                ..
            },
            ..
        })
    ));
    let mut patch = MaterialPatch::new(base.id.clone());
    patch.properties.insert(
        PropertyPath::from("not.a.property"),
        PropertyValue::Scalar(1.0.into()),
    );
    assert!(matches!(
        patch.apply(&base),
        Err(SchemaError {
            kind: ErrorKind::UnknownProperty(_),
            ..
        })
    ));
}

#[test]
fn unknown_fields_and_type_mismatches_are_rejected() {
    let mut spec = MaterialSpec::default();
    spec.base.values.metalness = Input::Field(FieldInput {
        field: FieldId::from("missing"),
        fallback: 0.0,
        mapping: None,
        remap: None,
    });
    assert!(matches!(
        spec.validate(),
        Err(SchemaError {
            kind: ErrorKind::UnknownField(_),
            ..
        })
    ));

    spec.fields.nodes.insert(
        FieldId::from("tint"),
        FieldNode::Constant(FieldValue::Color([1.0, 0.0, 0.0])),
    );
    spec.base.values.metalness = Input::Field(FieldInput {
        field: FieldId::from("tint"),
        fallback: 0.0,
        mapping: None,
        remap: None,
    });
    assert!(matches!(
        spec.validate(),
        Err(SchemaError {
            kind: ErrorKind::FieldType {
                expected: ValueType::Scalar,
                actual: ValueType::Color
            },
            ..
        })
    ));
}

#[test]
fn field_cycles_are_rejected_without_recursing_forever() {
    let graph = FieldGraph {
        nodes: BTreeMap::from([
            (
                FieldId::from("a"),
                FieldNode::Math {
                    op: MathOp::Add,
                    inputs: vec![FieldId::from("b"), FieldId::from("b")],
                },
            ),
            (
                FieldId::from("b"),
                FieldNode::Math {
                    op: MathOp::Add,
                    inputs: vec![FieldId::from("a"), FieldId::from("a")],
                },
            ),
        ]),
    };
    assert!(matches!(
        graph.validate(),
        Err(SchemaError {
            kind: ErrorKind::FieldCycle(_),
            ..
        })
    ));
}

#[test]
fn invalid_math_arity_and_operand_types_are_rejected() {
    let arity = FieldGraph {
        nodes: BTreeMap::from([(
            FieldId::from("bad"),
            FieldNode::Math {
                op: MathOp::Add,
                inputs: vec![],
            },
        )]),
    };
    assert!(arity.validate().is_err());
    let types = FieldGraph {
        nodes: BTreeMap::from([
            (FieldId::from("uv"), FieldNode::Builtin(BuiltinField::Uv)),
            (
                FieldId::from("bad"),
                FieldNode::Math {
                    op: MathOp::Cross,
                    inputs: vec![FieldId::from("uv"), FieldId::from("uv")],
                },
            ),
        ]),
    };
    assert!(types.validate().is_err());
}

#[test]
fn non_finite_values_fail_even_while_their_group_is_disabled() {
    let mut spec = MaterialSpec::default();
    spec.coat.enabled = false;
    spec.coat.values.roughness = f64::NAN.into();
    assert!(matches!(
        spec.validate(),
        Err(SchemaError {
            kind: ErrorKind::NonFinite,
            ..
        })
    ));

    let mut spec = MaterialSpec::default();
    spec.fields.nodes.insert(
        FieldId::from("bad"),
        FieldNode::Constant(FieldValue::Scalar(f64::INFINITY)),
    );
    assert!(matches!(
        spec.validate(),
        Err(SchemaError {
            kind: ErrorKind::NonFinite,
            ..
        })
    ));

    let mut spec = MaterialSpec::default();
    spec.transvolume.values.rough_estimator = RoughTransmissionEstimator::Hybrid {
        sampled_share: f64::NAN,
    };
    assert!(matches!(
        spec.validate(),
        Err(SchemaError {
            kind: ErrorKind::NonFinite,
            ..
        })
    ));
    spec.transvolume.values.rough_estimator = RoughTransmissionEstimator::Fast;
    spec.transvolume.values.dispersion = DispersionModel::SpectralAbbe {
        cauchy_amount: 0.02.into(),
        abbe_number: f64::INFINITY.into(),
    };
    assert!(matches!(
        spec.validate(),
        Err(SchemaError {
            kind: ErrorKind::NonFinite,
            ..
        })
    ));
}

#[test]
fn zero_uv_scale_and_unbounded_alpha_cutoff_validate_but_malformed_values_fail() {
    let mut spec = MaterialSpec::default();
    spec.coverage.values.alpha_cutoff = 2.0.into();
    spec.textures.insert(
        TextureId::from("texture:zero-scale"),
        TextureResource {
            uri: "base.png".into(),
            color_space: TextureColorSpace::Srgb,
            sampling: TextureSampling {
                transform: UvTransform {
                    scale: [0.0, 1.0],
                    ..UvTransform::default()
                },
                ..TextureSampling::default()
            },
        },
    );
    spec.validate().unwrap();

    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        spec.textures.get_mut(&TextureId::from("texture:zero-scale")).unwrap()
            .sampling.transform.scale[0] = value;
        assert!(matches!(
            spec.validate(),
            Err(SchemaError { kind: ErrorKind::NonFinite, .. })
        ));
    }
    spec.textures.get_mut(&TextureId::from("texture:zero-scale")).unwrap()
        .sampling.transform.scale[0] = 0.0;

    spec.coverage.values.alpha_cutoff = (-0.01).into();
    assert!(matches!(
        spec.validate(),
        Err(SchemaError { kind: ErrorKind::OutOfRange { .. }, .. })
    ));
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        spec.coverage.values.alpha_cutoff = value.into();
        assert!(matches!(
            spec.validate(),
            Err(SchemaError { kind: ErrorKind::NonFinite, .. })
        ));
    }
}

#[test]
fn animation_and_texture_descriptors_are_validated() {
    let mut spec = MaterialSpec::default();
    spec.specular.values.roughness = Input::Animated(AnimatedInput {
        id: TrackId::from("track:bad"),
        fallback: 0.0,
        interpolation: Interpolation::Linear,
        keys: vec![
            AnimationKey {
                time_seconds: 1.0,
                value: 0.2,
            },
            AnimationKey {
                time_seconds: 1.0,
                value: 0.3,
            },
        ],
    });
    assert!(matches!(
        spec.validate(),
        Err(SchemaError {
            kind: ErrorKind::InvalidAnimation(_),
            ..
        })
    ));

    let mut spec = MaterialSpec::default();
    spec.base.values.color = Input::Texture(TextureInput {
        texture: TextureId::from("texture:missing"),
        fallback: Color::WHITE,
        channels: ChannelSelector::Rgb,
        mapping: None,
        remap: None,
    });
    assert!(matches!(
        spec.validate(),
        Err(SchemaError {
            kind: ErrorKind::UnknownTexture(_),
            ..
        })
    ));

    spec.textures.insert(
        TextureId::from("texture:missing"),
        TextureResource {
            uri: "base.png".into(),
            color_space: TextureColorSpace::Srgb,
            sampling: TextureSampling::default(),
        },
    );
    spec.validate().unwrap();

    spec.base.values.color = Input::Texture(TextureInput {
        texture: TextureId::from("texture:missing"),
        fallback: Color::WHITE,
        channels: ChannelSelector::Red,
        mapping: None,
        remap: None,
    });
    assert!(
        spec.validate().is_err(),
        "a scalar channel cannot satisfy a colour input"
    );

    spec.base.values.color = Color::WHITE.into();
    spec.fields.nodes.insert(
        FieldId::from("unbound"),
        FieldNode::Texture {
            texture: TextureId::from("texture:absent"),
            output: ValueType::Scalar,
        },
    );
    assert!(matches!(
        spec.validate(),
        Err(SchemaError {
            kind: ErrorKind::UnknownTexture(_),
            ..
        })
    ));
}

#[test]
fn serialization_is_stable_and_ordered() {
    let mut spec = MaterialSpec::default();
    spec.fields = scalar_field_graph();
    for name in ["texture:z", "texture:a", "texture:m"] {
        spec.textures.insert(
            TextureId::from(name),
            TextureResource {
                uri: format!("{name}.png"),
                color_space: TextureColorSpace::NonColor,
                sampling: TextureSampling::default(),
            },
        );
    }
    let first = serde_json::to_string_pretty(&spec).unwrap();
    let second = serde_json::to_string_pretty(&spec).unwrap();
    assert_eq!(first, second);
    let a = first.find("texture:a").unwrap();
    let m = first.find("texture:m").unwrap();
    let z = first.find("texture:z").unwrap();
    assert!(
        a < m && m < z,
        "BTreeMap resource order leaked into serialization"
    );
    assert!(first.contains("\"schema_version\": 1"));
    let reopened: MaterialSpec = serde_json::from_str(&first).unwrap();
    assert_eq!(serde_json::to_string_pretty(&reopened).unwrap(), first);
}

#[test]
fn newer_versions_and_invalid_compatibility_combinations_fail_safely() {
    let mut future = MaterialSpec::default();
    future.schema_version = MATERIAL_SCHEMA_VERSION + 1;
    assert!(matches!(
        future.validate(),
        Err(SchemaError {
            kind: ErrorKind::UnsupportedVersion { .. },
            ..
        })
    ));

    let mut incompatible = MaterialSpec::default();
    incompatible.transvolume.values.wall = WallMode::ThinWalled {
        thickness: material::WALL.into(),
    };
    incompatible.subsurface.enabled = true;
    incompatible.subsurface.values.weight = 0.5.into();
    assert!(matches!(
        incompatible.validate(),
        Err(SchemaError {
            kind: ErrorKind::Incompatible(_),
            ..
        })
    ));

    let mut incompatible = MaterialSpec::default();
    incompatible.special_optics.grin.enabled = true;
    incompatible.special_optics.birefringence.enabled = true;
    assert!(matches!(
        incompatible.validate(),
        Err(SchemaError {
            kind: ErrorKind::Incompatible(_),
            ..
        })
    ));
}
