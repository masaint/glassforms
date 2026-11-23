use gf_core::construction::Source;
use gf_core::csg::Part;
use gf_core::pbr::*;
use gf_core::setup::{CANONICAL_VERSION, VERSION};
use gf_core::state::State;
use std::collections::BTreeMap;

fn state(args: &[(&str, &str)]) -> State {
    State::with_args(
        &args.iter().map(|(key, value)| (key.to_string(), value.to_string())).collect::<Vec<_>>(),
        true,
    )
}

fn assembly(extra: &[(&str, &str)]) -> State {
    let mut args = vec![
        ("scene", "knots"),
        ("preset", "trefoil"),
        ("layer", "algebraic:heart"),
        ("layer", "algebraic:tanglecube"),
        ("J12.op", "apart"),
        ("J13.op", "apart"),
        ("J23.op", "apart"),
    ];
    args.extend_from_slice(extra);
    state(&args)
}

#[test]
fn ordinary_setup_version_one_is_byte_and_shape_stable() {
    let source = assembly(&[("B2.material", "gold"), ("rough", "0.19")]);
    let text = source.setup_json();
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(json["version"], VERSION);
    assert!(json.get("canonical").is_none());
    assert!(!text.contains("material_v2"));

    let mut reopened = State::new();
    reopened.apply_setup_json(&text).unwrap();
    assert!(reopened.canonical.is_none());
    assert_eq!(reopened.setup().version, VERSION);
    assert_eq!(reopened.setup_json(), text);
}

#[test]
fn canonical_scene_round_trips_dormant_values_without_changing_other_settings() {
    let mut source = State::new();
    source.p.rough = 0.23;
    source.p.fov = 47.0;
    source.enable_canonical_materials().unwrap();
    {
        let scene = &mut source.canonical.as_mut().unwrap().scene;
        scene.coat.values.weight = 0.73.into();
        scene.coat.values.roughness = 0.28.into();
        scene.sheen.values.color = Color([0.2, 0.4, 0.6]).into();
        assert!(!scene.coat.enabled && !scene.sheen.enabled);
    }
    // Re-project the explicitly edited canonical owner through the strict renderer bridge.
    source.enable_canonical_materials().unwrap();
    let saved = source.setup();
    assert_eq!(saved.version, CANONICAL_VERSION);
    let text = serde_json::to_string_pretty(&saved).unwrap();

    let mut reopened = State::new();
    reopened.apply_setup_json(&text).unwrap();
    assert_eq!(reopened.setup(), saved);
    assert_eq!(reopened.setup_json(), text);
    assert_eq!(reopened.p.fov, 47.0);
    let scene = &reopened.canonical.as_ref().unwrap().scene;
    assert_eq!(scene.coat.values.weight.as_constant(), Some(&0.73));
    assert_eq!(scene.coat.values.roughness.as_constant(), Some(&0.28));
    assert_eq!(scene.sheen.values.color.as_constant(), Some(&Color([0.2, 0.4, 0.6])));
}

#[test]
fn canonical_bodies_are_full_owners_and_absent_bodies_inherit() {
    let mut source = assembly(&[("B2.material", "gold"), ("B2.rough", "0.31")]);
    source.enable_canonical_materials().unwrap();
    assert_eq!(source.setup().canonical.as_ref().unwrap().bodies.keys().collect::<Vec<_>>(), ["2"]);
    let saved = source.setup();

    let mut reopened = State::new();
    reopened.apply_setup(&saved).unwrap();
    assert!(reopened.body_own(1));
    assert!(!reopened.body_own(2));
    assert_eq!(reopened.body_settings(2), reopened.body_settings(0));
    assert_eq!(reopened.setup(), saved);
    let owner = reopened.canonical.as_ref().unwrap();
    let gold = owner.bodies[1].as_deref().unwrap();
    assert_eq!(gold.base.values.metalness.as_constant(), Some(&1.0));
    assert_eq!(gold.specular.values.roughness.as_constant(), Some(&0.31));
    assert!(owner.bodies[2].is_none());
}

#[test]
fn effective_medium_descriptors_inherit_and_survive_save_reopen() {
    let mut source = assembly(&[("B2.material", "gold")]);
    source.enable_canonical_materials().unwrap();
    {
        let owner = source.canonical.as_mut().unwrap();
        owner.scene.transvolume.values.medium = MediumDescriptor {
            id: MediumId::from("medium:scene.custom"),
            interface_priority: -7,
        };
        owner.bodies[1].as_deref_mut().unwrap().transvolume.values.medium = MediumDescriptor {
            id: MediumId::from("medium:shared"),
            interface_priority: 12,
        };
    }
    let before = source.medium_descriptors();
    assert_eq!(before.len(), source.bodies().len());
    assert_eq!(before[0].id.as_str(), "medium:scene.custom");
    assert_eq!(before[1].id.as_str(), "medium:shared");
    assert_eq!(before[2], before[0]);
    assert_eq!(source.compiled_media().bodies[0].identity, 1);
    assert_eq!(source.compiled_media().bodies[1].identity, 2);
    assert_eq!(source.compiled_media().bodies[1].interface_priority, 12);

    let saved = source.setup();
    let text = serde_json::to_string_pretty(&saved).unwrap();
    let mut reopened = State::new();
    reopened.apply_setup_json(&text).unwrap();
    assert_eq!(reopened.medium_descriptors(), before);
    assert_eq!(reopened.compiled_media(), source.compiled_media());

    let owner = reopened.canonical.as_ref().unwrap();
    let gold = owner.bodies[1].as_deref().unwrap();
    let patch = MaterialPatch {
        base: gold.id.clone(),
        ..MaterialPatch::default()
    };
    assert_eq!(patch.apply(gold).unwrap().transvolume.values.medium, gold.transvolume.values.medium);
}

#[test]
fn medium_priority_overlap_fixtures_round_trip() {
    for (name, first, second) in [("zero", 0, 0), ("priority", 10, 0), ("tie", 10, 10)] {
        let mut source = state(&[
            ("scene", "solids"), ("preset", "superellipsoid"),
            ("p.p", "2"), ("p.q", "2"), ("p.tall", "1.15"), ("p.deep", "0.72"),
            ("material", "crown"), ("ior", "1.50"), ("disp", "0"), ("nb", "8"),
            ("layer", "solids:superellipsoid"),
            ("L2.p.p", "2"), ("L2.p.q", "2"), ("L2.p.tall", "0.80"), ("L2.p.deep", "1.05"),
            ("L2.size", "0.90"), ("L2.rx", "-52"), ("L2.ry", "28"), ("L2.rz", "57"),
            ("L2.x", "0.18"), ("L2.y", "-0.08"), ("J12.op", "apart"),
            ("B2.material", "crown"), ("B2.ior", "1.33"), ("B2.disp", "0"),
            ("rx", "23"), ("ry", "-31"), ("rz", "11"), ("scale", "1"),
            ("spin", "0"), ("grain", "0"), ("res", "192"),
        ]);
        assert_eq!(source.bodies().len(), 2);
        source.enable_canonical_materials().unwrap();
        let owner = source.canonical.as_mut().unwrap();
        owner.scene.transvolume.values.medium = MediumDescriptor {
            id: MediumId::from("medium:overlap:ellipsoid-a"), interface_priority: first,
        };
        owner.bodies[1].as_deref_mut().unwrap().transvolume.values.medium = MediumDescriptor {
            id: MediumId::from("medium:overlap:ellipsoid-b"), interface_priority: second,
        };
        let media = source.compiled_media();
        assert_ne!(media.bodies[0].identity, media.bodies[1].identity);
        assert_eq!(media.bodies[0].interface_priority, first);
        assert_eq!(media.bodies[1].interface_priority, second);
        assert_eq!(source.body_settings(0).ior, 1.50);
        assert_eq!(source.body_settings(1).ior, 1.33);
        let text = source.setup_json();
        let saved: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(saved["values"]["p"], 2.0);
        assert_eq!(saved["values"]["q"], 2.0);
        assert_eq!(saved["values"]["tall"], 1.15);
        assert_eq!(saved["values"]["deep"], 0.72);
        assert_eq!(saved["layers"]["more"][0]["values"]["p"], 2.0);
        assert_eq!(saved["layers"]["more"][0]["values"]["q"], 2.0);
        assert_eq!(saved["layers"]["more"][0]["values"]["tall"], 0.80);
        assert_eq!(saved["layers"]["more"][0]["values"]["deep"], 1.05);
        assert_eq!(saved["layers"]["stack"]["layers"][1]["place"]["rot"], serde_json::json!([-52.0, 28.0, 57.0]));
        let mut reopened = State::new();
        reopened.apply_setup_json(&text).unwrap();
        assert_eq!(reopened.compiled_media(), media);
        assert_eq!(reopened.setup_json(), text);
        if let Ok(folder) = std::env::var("GLASSFORMS_MEDIUM_PRIORITY_FIXTURES") {
            std::fs::create_dir_all(&folder).unwrap();
            std::fs::write(std::path::Path::new(&folder).join(format!("overlap-{name}.json")), text).unwrap();
        }
    }
}

#[test]
fn legacy_state_projection_uses_from_legacy_scene_and_body_mediums() {
    let source = assembly(&[("B2.material", "gold")]);
    let scene = MaterialSpec::from_legacy(&source.p, MaterialId::from("material:scene"));
    let body = MaterialSpec::from_legacy(source.body_settings(1), MaterialId::from("material:body:2"));
    let actual = source.medium_descriptors();
    assert_eq!(actual[0], scene.transvolume.values.medium);
    assert_eq!(actual[1], body.transvolume.values.medium);
    assert_eq!(actual[2], scene.transvolume.values.medium);
}

#[test]
fn new_legacy_body_owners_receive_the_shared_from_legacy_medium_id() {
    let mut source = assembly(&[]);
    source.enable_canonical_materials().unwrap();
    source.set_body_own(1, true);
    let descriptors = source.medium_descriptors();
    assert_eq!(descriptors[1].id.as_str(), "medium:material:body:2");
    assert_eq!(descriptors[1].interface_priority, 0);
    let projected = MaterialSpec::from_legacy(source.body_settings(1), MaterialId::from("material:body:2"));
    assert_eq!(descriptors[1], projected.transvolume.values.medium);
}

#[test]
fn canonical_legacy_mirrors_are_projected_on_the_first_save() {
    let mut source = assembly(&[("B2.material", "moulded"), ("B2.glazeHaze", "0.3")]);
    source.enable_canonical_materials().unwrap();
    let saved = source.setup();
    let exported = source.canonical.as_ref().unwrap().bodies[1].as_ref().unwrap().to_legacy_settings().unwrap();
    assert_eq!(saved.bodies["2"]["glazeHaze"], exported.glaze_haze);
    assert_eq!(saved.settings.rough, source.canonical.as_ref().unwrap().scene.to_legacy_settings().unwrap().rough);

    let text = serde_json::to_string_pretty(&saved).unwrap();
    let mut reopened = State::new();
    reopened.apply_setup_json(&text).unwrap();
    assert_eq!(reopened.setup(), saved);
    assert_eq!(reopened.setup_json(), text);
}

#[test]
fn sparse_construction_patches_round_trip_on_nested_and_secondary_layers() {
    let mut source = assembly(&[("B2.material", "gold")]);
    let mut nested = Vec::new();
    let root = source.layer_csg_mut(0).unwrap().add_construction_part(&[], Part::default()).unwrap();
    let sibling = source.layer_csg_mut(0).unwrap().add_construction_part(&[], Part::default()).unwrap();
    let group = source.layer_csg_mut(0).unwrap().edit_construction().group(&[], &[root[0], sibling[0]], "Paint").unwrap();
    source.layer_csg_mut(0).unwrap().edit_construction().node_mut(&group).unwrap().material = Some(BTreeMap::from([("rough".into(), 0.17)]));
    let mut child = group.clone();
    child.push(1);
    source.layer_csg_mut(0).unwrap().edit_construction().node_mut(&child).unwrap().material = Some(BTreeMap::from([("coatIor".into(), 1.57)]));
    nested.push(group);
    nested.push(child);
    let secondary = source.layer_csg_mut(1).unwrap().add_construction_part(&[], Part::default()).unwrap();
    source.layer_csg_mut(1).unwrap().edit_construction().node_mut(&secondary).unwrap().material = Some(BTreeMap::from([("rough".into(), 0.39)]));

    source.enable_canonical_materials().unwrap();
    let saved = source.setup();
    assert_eq!(saved.version, CANONICAL_VERSION);
    let text = serde_json::to_string_pretty(&saved).unwrap();
    assert_eq!(text.matches("material_v2").count(), 3);
    let mut reopened = State::new();
    reopened.apply_setup_json(&text).unwrap();
    assert_eq!(reopened.setup(), saved);

    let owner = reopened.canonical.as_ref().unwrap();
    for path in nested {
        let node = reopened.layer_csg(0).construction.as_ref().unwrap().node(&path).unwrap();
        let patch = node.material_v2.as_ref().unwrap();
        let resolved = patch.apply(&owner.scene).unwrap();
        assert_eq!(resolved.id, owner.scene.id);
        assert_eq!(resolved.specular.values.ior, owner.scene.specular.values.ior);
        if path.len() > 1 {
            assert_eq!(resolved.specular.values.roughness, owner.scene.specular.values.roughness,
                "construction children inherit the body, not their parent group's paint");
        }
    }
    let node = reopened.layer_csg(1).construction.as_ref().unwrap().node(&secondary).unwrap();
    assert!(matches!(node.source, Source::Part(_)));
    let patch = node.material_v2.as_ref().unwrap();
    assert_eq!(patch.properties.len(), 1);
    assert_eq!(node.material.as_ref().unwrap().keys().collect::<Vec<_>>(), ["rough"]);
    let gold = owner.bodies[1].as_deref().unwrap();
    let resolved = patch.apply(gold).unwrap();
    assert_eq!(resolved.specular.values.roughness.as_constant(), Some(&0.39));
    assert_eq!(resolved.base.values.metalness, gold.base.values.metalness);

    let mut file = serde_json::to_value(&saved).unwrap();
    file.pointer_mut("/construction/roots/0").unwrap().as_object_mut().unwrap()
        .insert("material".into(), serde_json::json!({"rough": 0.99}));
    let mut ignores_mirror = State::new();
    ignores_mirror.apply_setup_json(&file.to_string()).unwrap();
    let unpainted = &ignores_mirror.layer_csg(0).construction.as_ref().unwrap().roots[0];
    assert!(unpainted.material.is_none() && unpainted.material_v2.is_none());
}

#[test]
fn a_canonical_construction_patch_implies_a_complete_setup_owner() {
    let mut source = State::new();
    let path = source.layer_csg_mut(0).unwrap().add_construction_part(&[], Part::default()).unwrap();
    let base = MaterialSpec::from_legacy(&source.p, MaterialId::from("material:scene"));
    let patch = MaterialPatch::from_legacy_map(&base, &BTreeMap::from([("rough".into(), 0.37)])).unwrap();
    let node = source.layer_csg_mut(0).unwrap().edit_construction().node_mut(&path).unwrap();
    node.material = Some(BTreeMap::from([("rough".into(), 0.37)]));
    node.material_v2 = Some(patch);

    let saved = source.setup();
    assert_eq!(saved.version, CANONICAL_VERSION);
    assert!(saved.canonical.is_some());
    let text = serde_json::to_string_pretty(&saved).unwrap();
    let mut reopened = State::new();
    reopened.apply_setup_json(&text).unwrap();
    let after = reopened.setup();
    assert_eq!(after.canonical, saved.canonical);
    assert_eq!(after.construction, saved.construction);
    let after_json = serde_json::to_value(&after).unwrap();
    for (key, value) in serde_json::to_value(&saved).unwrap().as_object().unwrap() {
        assert!(after_json.get(key) == Some(value), "setup field {key} differs");
    }
    assert_eq!(after, saved);
}

#[test]
fn the_undo_fast_path_restores_canonical_only_values() {
    let mut source = State::new();
    source.enable_canonical_materials().unwrap();
    let saved = source.undo_snapshot();
    source.canonical.as_mut().unwrap().scene.coat.values.weight = 0.81.into();
    assert_ne!(source.setup().canonical, saved.canonical);
    source.restore_snapshot(&saved);
    assert_eq!(source.canonical.as_ref().unwrap().scene.coat.values.weight.as_constant(), Some(&0.0));
    assert_eq!(source.setup(), saved);
}

#[test]
fn existing_material_controls_update_canonical_owners_and_sparse_patches() {
    let mut source = assembly(&[("B2.material", "gold")]);
    source.enable_canonical_materials().unwrap();
    source.canonical.as_mut().unwrap().scene.coat.values.weight = 0.67.into();
    assert!(source.body_set(0, "rough", 0.46));
    assert!(source.body_set(1, "coatIor", 1.61));
    let path = source.layer_csg_mut(0).unwrap().add_construction_part(&[], Part::default()).unwrap();
    source.set_construction_material(0, &path, Some(BTreeMap::from([("rough".into(), 0.52)]))).unwrap();

    let owner = source.canonical.as_ref().unwrap();
    assert_eq!(owner.scene.specular.values.roughness.as_constant(), Some(&0.46));
    assert!(!owner.scene.coat.enabled, "an unrelated edit must not activate a dormant group");
    assert_eq!(source.p.coat, 0.0);
    assert_eq!(owner.scene.coat.values.weight.as_constant(), Some(&0.67), "unrelated dormant values survive");
    assert_eq!(owner.bodies[1].as_deref().unwrap().coat.values.ior.as_constant(), Some(&1.61));
    let saved = source.setup();
    let text = serde_json::to_string_pretty(&saved).unwrap();
    let mut reopened = State::new();
    reopened.apply_setup_json(&text).unwrap();
    assert_eq!(reopened.setup(), saved);
    assert_eq!(reopened.p.rough, 0.46);
    assert_eq!(reopened.body_settings(1).coat_ior, 1.61);
    let node = reopened.layer_csg(0).construction.as_ref().unwrap().node(&path).unwrap();
    assert_eq!(node.material.as_ref().unwrap()["rough"], 0.52);
    assert_eq!(node.material_v2.as_ref().unwrap().properties.len(), 1);

    reopened.set_construction_material(0, &path, None).unwrap();
    let node = reopened.layer_csg(0).construction.as_ref().unwrap().node(&path).unwrap();
    assert!(node.material.is_none() && node.material_v2.is_none());
    reopened.set_body_own(1, false);
    assert!(reopened.canonical.as_ref().unwrap().bodies[1].is_none());
}

#[test]
fn removing_the_first_layer_promotes_its_canonical_body_material() {
    let mut source = assembly(&[("B2.material", "gold")]);
    source.enable_canonical_materials().unwrap();
    let gold = source.canonical.as_ref().unwrap().bodies[1].clone();
    assert!(source.remove_layer(0));
    let owner = source.canonical.as_ref().unwrap();
    assert_eq!(owner.scene, *gold.unwrap());
    assert!(owner.bodies[1].is_none());
    assert_eq!(owner.scene.base.values.metalness.as_constant(), Some(&1.0));

    let saved = source.setup();
    let mut reopened = State::new();
    reopened.apply_setup(&saved).unwrap();
    assert_eq!(reopened.setup(), saved);
}

#[test]
fn pre_grating_recipe_migration_does_not_opt_into_version_two() {
    let source = assembly(&[("material", "gold"), ("B2.material", "bisque")]);
    let original = source.setup();
    let mut file = serde_json::to_value(&original).unwrap();
    for (path, index) in [("/settings", 59), ("/bodies/2", 103)] {
        let values = file.pointer_mut(path).unwrap().as_object_mut().unwrap();
        values.retain(|key, _| !key.starts_with("grating"));
        values.insert("mat".into(), index.into());
    }
    let old_text = file.to_string();
    let mut reopened = State::new();
    reopened.apply_setup_json(&old_text).unwrap();
    assert_eq!(reopened.p.mat, source.p.mat);
    assert_eq!(reopened.body_settings(1).mat, source.body_settings(1).mat);
    assert!(reopened.canonical.is_none());
    assert_eq!(reopened.setup(), original);
    assert_eq!(reopened.setup().version, VERSION);
}

#[test]
fn invalid_or_nonconstant_canonical_application_is_atomic() {
    let mut authored = State::new();
    authored.enable_canonical_materials().unwrap();
    let valid = authored.setup();
    let mut target = assembly(&[("B2.material", "gold")]);
    target.p.rough = 0.44;
    let before = target.setup();

    let mut fractional = valid.clone();
    fractional.canonical.as_mut().unwrap().scene.base.values.metalness = 0.37.into();
    let error = target.apply_setup(&fractional).unwrap_err();
    assert!(error.contains("fractional metalness"), "{error}");
    assert_eq!(target.setup(), before);

    let mut unsupported = valid.clone();
    unsupported.canonical.as_mut().unwrap().scene.sheen.enabled = true;
    let error = target.apply_setup(&unsupported).unwrap_err();
    assert!(error.contains("current legacy renderer cannot represent"), "{error}");
    assert_eq!(target.setup(), before);

    let mut nonconstant = valid;
    nonconstant.canonical.as_mut().unwrap().scene.specular.values.roughness = Input::Animated(AnimatedInput {
        id: TrackId::from("track:roughness"),
        fallback: 0.2,
        interpolation: Interpolation::Linear,
        keys: vec![AnimationKey { time_seconds: 0.0, value: 0.2 }],
    });
    // Nonconstant topology is losslessly serialisable even though this renderer cannot apply it.
    let text = serde_json::to_string(&nonconstant).unwrap();
    let decoded: gf_core::setup::Setup = serde_json::from_str(&text).unwrap();
    assert_eq!(decoded, nonconstant);
    let error = target.apply_setup(&decoded).unwrap_err();
    assert!(error.contains("only constant inputs"), "{error}");
    assert_eq!(target.setup(), before);
}

#[test]
fn loading_version_one_after_version_two_cannot_edit_the_previous_owner() {
    let legacy = assembly(&[("B2.material", "moulded"), ("B2.glazeHaze", "0.3")]).setup();
    assert_eq!(legacy.version, VERSION);
    assert!(legacy.bodies["2"].contains_key("mat"));
    let mut target = assembly(&[("B2.material", "gold")]);
    target.enable_canonical_materials().unwrap();
    target.apply_setup(&legacy).unwrap();
    assert!(target.canonical.is_none());
    assert_eq!(target.body_settings(1).glaze_haze, 0.3);
    assert_eq!(target.setup(), legacy);
}

#[test]
fn construction_named_recipes_become_sparse_values_without_preset_identity() {
    let mut source = assembly(&[]);
    source.enable_canonical_materials().unwrap();
    let path = source.layer_csg_mut(0).unwrap().add_construction_part(&[], Part::default()).unwrap();
    for (index, named) in gf_core::material::NAMED.iter().enumerate() {
        let inherited = source.p.clone();
        let mut selected = inherited.clone();
        named.apply(&mut selected);
        selected.mat = index as f64;
        let values: BTreeMap<_, _> = gf_core::material::body_names().filter(|name| *name != "mat").filter_map(|name| {
            let value = selected.get(name)?;
            (inherited.get(name) != Some(value)).then(|| (name.to_string(), value))
        }).collect();
        assert!(!values.contains_key("mat"));
        source.set_construction_material(0, &path, Some(values)).unwrap();
        let node = source.layer_csg(0).construction.as_ref().unwrap().node(&path).unwrap();
        let patch = node.material_v2.as_ref().unwrap();
        let base = &source.canonical.as_ref().unwrap().scene;
        patch.apply(base).unwrap().to_legacy_settings().unwrap();
        assert!(!node.material.as_ref().unwrap().contains_key("mat"));
    }
}

#[test]
fn changing_body_ownership_rebases_dependent_construction_patches() {
    let mut source = assembly(&[("B2.material", "gold")]);
    source.enable_canonical_materials().unwrap();
    let path = source.layer_csg_mut(1).unwrap().add_construction_part(&[], Part::default()).unwrap();
    source.set_construction_material(1, &path, Some(BTreeMap::from([("rough".into(), 0.37)]))).unwrap();
    assert_eq!(source.layer_csg(1).construction.as_ref().unwrap().node(&path).unwrap().material_v2.as_ref().unwrap().base.as_str(), "material:body:2");

    source.set_body_own(1, false);
    let node = source.layer_csg(1).construction.as_ref().unwrap().node(&path).unwrap();
    let patch = node.material_v2.as_ref().unwrap();
    assert_eq!(patch.base.as_str(), "material:scene");
    assert_eq!(patch.apply(&source.canonical.as_ref().unwrap().scene).unwrap().specular.values.roughness.as_constant(), Some(&0.37));
    assert_eq!(node.material.as_ref().unwrap()["rough"], 0.37);

    source.set_body_own(1, true);
    let owner = source.canonical.as_ref().unwrap();
    let node = source.layer_csg(1).construction.as_ref().unwrap().node(&path).unwrap();
    let patch = node.material_v2.as_ref().unwrap();
    assert_eq!(patch.base.as_str(), "material:body:2");
    patch.apply(owner.bodies[1].as_deref().unwrap()).unwrap().to_legacy_settings().unwrap();
    let text = source.setup_json();
    let mut reopened = State::new();
    reopened.apply_setup_json(&text).unwrap();
    assert_eq!(reopened.setup_json(), text);
}

#[test]
fn rebasing_a_body_colour_patch_relinks_an_inherited_ideal_conductor() {
    let mut source = assembly(&[("material", "mirror"), ("B2.material", "gold")]);
    let path = source.layer_csg_mut(1).unwrap().add_construction_part(&[], Part::default()).unwrap();
    source.enable_canonical_materials().unwrap();
    source.set_construction_material(1, &path, Some(BTreeMap::from([("bodyR".into(), 0.41)]))).unwrap();
    source.set_body_own(1, false);
    assert!(!source.body_own(1), "a representable ownership change must not be rejected");
    let owner = source.canonical.as_ref().unwrap();
    let node = source.layer_csg(1).construction.as_ref().unwrap().node(&path).unwrap();
    let patch = node.material_v2.as_ref().unwrap();
    assert_eq!(patch.base, MaterialId::from("material:scene"));
    let resolved = patch.apply(&owner.scene).unwrap();
    let exported = resolved.to_legacy_settings().unwrap();
    assert_eq!(exported.body_r, 0.41);
    let ConductorModel::Ideal { tint } = &resolved.base.values.conductor else {
        panic!("the scene's ideal conductor must be inherited")
    };
    assert_eq!(tint.as_constant(), Some(&Color([0.41, 1.0, 1.0])));
    let saved = source.setup();
    let mut reopened = State::new();
    reopened.apply_setup(&saved).unwrap();
    assert_eq!(reopened.setup(), saved);
}

#[test]
fn changing_boolean_topology_rebases_construction_patches_before_saving() {
    let mut source = assembly(&[("B2.material", "gold")]);
    source.enable_canonical_materials().unwrap();
    let path = source.layer_csg_mut(1).unwrap().add_construction_part(&[], Part::default()).unwrap();
    source.set_construction_material(1, &path, Some(BTreeMap::from([("rough".into(), 0.37)]))).unwrap();
    assert!(source.apply_arg("J12.op", "join"));
    let node = source.layer_csg(1).construction.as_ref().unwrap().node(&path).unwrap();
    assert_eq!(node.material_v2.as_ref().unwrap().base, MaterialId::from("material:scene"));
    let saved = source.setup();
    let mut reopened = State::new();
    reopened.apply_setup(&saved).unwrap();
    assert_eq!(reopened.setup(), saved);
    assert!(reopened.apply_arg("J12.op", "apart"));
    let node = reopened.layer_csg(1).construction.as_ref().unwrap().node(&path).unwrap();
    assert_eq!(node.material_v2.as_ref().unwrap().base, MaterialId::from("material:body:2"));
    let saved = reopened.setup();
    let mut again = State::new();
    again.apply_setup(&saved).unwrap();
    assert_eq!(again.setup(), saved);
}

#[test]
fn sparse_construction_edits_preserve_untouched_dormant_values() {
    let mut source = State::new();
    source.enable_canonical_materials().unwrap();
    let path = source.layer_csg_mut(0).unwrap().add_construction_part(&[], Part::default()).unwrap();
    let mut dormant = MaterialPatch::new(MaterialId::from("material:scene"));
    dormant.properties.insert(PropertyPath::from("coat.weight"), PropertyValue::Scalar(0.73.into()));
    dormant.features.insert(FeaturePath::from("coat"), false);
    let mirror = dormant.to_legacy_map(&source.canonical.as_ref().unwrap().scene).unwrap();
    assert_eq!(mirror["coat"], 0.0);
    let node = source.layer_csg_mut(0).unwrap().construction.as_mut().unwrap().node_mut(&path).unwrap();
    node.material = Some(mirror);
    node.material_v2 = Some(dormant);
    source.update_construction_material(0, &path, BTreeMap::from([("rough".into(), 0.42)])).unwrap();
    let node = source.layer_csg(0).construction.as_ref().unwrap().node(&path).unwrap();
    let patch = node.material_v2.as_ref().unwrap();
    assert_eq!(patch.features.get(&FeaturePath::from("coat")), Some(&false));
    assert!(!patch.apply(&source.canonical.as_ref().unwrap().scene).unwrap().coat.enabled);
    assert_eq!(patch.properties.get(&PropertyPath::from("coat.weight")).and_then(|value| match value {
        PropertyValue::Scalar(value) => value.as_constant().copied(),
        _ => None,
    }), Some(0.73));
    assert_eq!(node.material.as_ref().unwrap()["rough"], 0.42);
    let saved = source.setup();
    let mut reopened = State::new();
    reopened.apply_setup(&saved).unwrap();
    assert_eq!(reopened.setup(), saved);
}

#[test]
fn canonical_absorption_overrides_a_stale_compatibility_mirror() {
    let mut source = State::new();
    source.enable_canonical_materials().unwrap();
    {
        let volume = &mut source.canonical.as_mut().unwrap().scene.transvolume.values;
        volume.absorption_coefficients = Color([0.1, 0.2, 0.3]).into();
        volume.color = Color([(-0.1f64).exp(), (-0.2f64).exp(), (-0.3f64).exp()]).into();
    }
    let mut saved = source.setup();
    saved.settings.absorb = [0.9, 0.8, 0.7];
    let mut reopened = State::new();
    reopened.apply_setup(&saved).unwrap();
    assert_eq!(reopened.p.absorb, [0.1, 0.2, 0.3]);
    let normalised = reopened.setup();
    assert_eq!(normalised.settings.absorb, [0.1, 0.2, 0.3]);
    let mut again = State::new();
    again.apply_setup(&normalised).unwrap();
    assert_eq!(again.setup(), normalised);
}

#[test]
fn scripted_material_edits_update_the_canonical_scene() {
    let mut source = State::new();
    source.enable_canonical_materials().unwrap();
    assert!(source.apply_arg("rough", "0.37"));
    assert!(source.apply_arg("absorb", "10,14,1000"));
    let scene = &source.canonical.as_ref().unwrap().scene;
    assert_eq!(scene.specular.values.roughness.as_constant(), Some(&0.37));
    assert_eq!(scene.transvolume.values.absorption_coefficients.as_constant(), Some(&Color([10.0, 14.0, 1000.0])));
    assert_eq!(source.p.rough, 0.37);
    assert_eq!(source.p.absorb, scene.transvolume.values.absorption_coefficients.as_constant().unwrap().0);
    let saved = source.setup();
    let mut reopened = State::new();
    reopened.apply_setup(&saved).unwrap();
    assert_eq!(reopened.setup(), saved);
}

#[test]
fn absorption_outside_legacy_hard_limits_is_rejected_atomically() {
    let mut source = State::new();
    source.enable_canonical_materials().unwrap();
    let before_edit = source.setup();
    assert!(!source.apply_arg("absorb", "-1,0,0"));
    assert_eq!(source.setup(), before_edit);
    assert!(!source.apply_arg("absorb", "1001,0,0"));
    assert_eq!(source.setup(), before_edit);
    let mut saved = source.setup();
    let volume = &mut saved.canonical.as_mut().unwrap().scene.transvolume.values;
    volume.absorption_coefficients = Color([1001.0, 0.0, 0.0]).into();
    volume.color = Color([0.0, 1.0, 1.0]).into();
    let before = source.setup();
    assert!(source.apply_setup(&saved).is_err());
    assert_eq!(source.setup(), before);
}

#[test]
fn legacy_construction_preset_identity_is_migrated_to_sparse_values() {
    let mut source = State::new();
    let path = source.layer_csg_mut(0).unwrap().add_construction_part(&[], Part::default()).unwrap();
    let gold = gf_core::material::NAMED.iter().position(|named| named.id == "gold").unwrap();
    source.layer_csg_mut(0).unwrap().edit_construction().node_mut(&path).unwrap().material = Some(BTreeMap::from([
        ("mat".into(), gold as f64),
        ("rough".into(), 0.33),
    ]));
    source.enable_canonical_materials().unwrap();
    let node = source.layer_csg(0).construction.as_ref().unwrap().node(&path).unwrap();
    assert!(!node.material.as_ref().unwrap().contains_key("mat"));
    let resolved = node.material_v2.as_ref().unwrap().apply(&source.canonical.as_ref().unwrap().scene).unwrap();
    let exported = resolved.to_legacy_settings().unwrap();
    assert_eq!(exported.metal, 4.0);
    assert_eq!(exported.rough, 0.33);
    let saved = source.setup();
    let mut reopened = State::new();
    reopened.apply_setup(&saved).unwrap();
    assert_eq!(reopened.setup(), saved);
}

#[test]
fn legacy_obsidian_recipe_preserves_absorption_above_the_tint_slider_limit() {
    let mut source = State::new();
    let path = source.layer_csg_mut(0).unwrap().add_construction_part(&[], Part::default()).unwrap();
    let obsidian = gf_core::material::NAMED.iter().position(|named| named.id == "obsidian").unwrap();
    source.layer_csg_mut(0).unwrap().edit_construction().node_mut(&path).unwrap().material = Some(BTreeMap::from([("mat".into(), obsidian as f64)]));
    source.enable_canonical_materials().unwrap();
    let node = source.layer_csg(0).construction.as_ref().unwrap().node(&path).unwrap();
    let resolved = node.material_v2.as_ref().unwrap().apply(&source.canonical.as_ref().unwrap().scene).unwrap();
    let volume = &resolved.transvolume.values;
    assert_eq!(volume.absorption_coefficients.as_constant(), Some(&Color([14.0, 15.0, 16.0])));
    assert_eq!(volume.color.as_constant(), Some(&Color([(-14.0f64).exp(), (-15.0f64).exp(), (-16.0f64).exp()])));
    let exported = resolved.to_legacy_settings().unwrap();
    assert_eq!(exported.absorb, [14.0, 15.0, 16.0]);
    let saved = source.setup();
    let mut reopened = State::new();
    reopened.apply_setup(&saved).unwrap();
    assert_eq!(reopened.setup(), saved);
}

#[test]
fn live_obsidian_construction_preset_uses_direct_recipe_absorption() {
    let mut source = State::new();
    source.enable_canonical_materials().unwrap();
    let path = source.layer_csg_mut(0).unwrap().add_construction_part(&[], Part::default()).unwrap();
    let obsidian = gf_core::material::NAMED.iter().position(|named| named.id == "obsidian").unwrap();
    source.set_construction_material(0, &path, Some(BTreeMap::from([("mat".into(), obsidian as f64)]))).unwrap();
    let node = source.layer_csg(0).construction.as_ref().unwrap().node(&path).unwrap();
    assert!(!node.material.as_ref().unwrap().contains_key("mat"));
    let resolved = node.material_v2.as_ref().unwrap().apply(&source.canonical.as_ref().unwrap().scene).unwrap();
    assert_eq!(resolved.transvolume.values.absorption_coefficients.as_constant(), Some(&Color([14.0, 15.0, 16.0])));
}

#[test]
fn explicit_wear_edits_use_effective_amounts_not_dormant_payloads() {
    let mut source = State::new();
    source.enable_canonical_materials().unwrap();
    source.canonical.as_mut().unwrap().scene.wear.values.mottle = 0.5.into();
    assert!(source.body_set(0, "brush", 0.0));
    assert!(!source.canonical.as_ref().unwrap().scene.wear.enabled, "a zero edit does not activate dormant wear");
    assert!(source.body_set(0, "brush", 0.7));
    let scene = &source.canonical.as_ref().unwrap().scene;
    assert!(scene.wear.enabled);
    assert_eq!(scene.wear.values.mottle.as_constant(), Some(&0.5));
    assert_eq!(source.p.brush, 0.7);
    assert_eq!(source.p.mottle, 0.5, "enabling the shared wear group activates its complete payload");
    assert!(source.body_set(0, "brush", 0.0));
    assert!(source.canonical.as_ref().unwrap().scene.wear.enabled, "the now-effective mottle remains");
    assert!(source.body_set(0, "mottle", 0.0));
    let scene = &source.canonical.as_ref().unwrap().scene;
    assert!(!scene.wear.enabled, "turning off the last effective amount disables wear");
    assert_eq!(scene.wear.values.mottle.as_constant(), Some(&0.0));
    assert_eq!(source.p.mottle, 0.0);
    let saved = source.setup();
    let mut reopened = State::new();
    reopened.apply_setup(&saved).unwrap();
    assert_eq!(reopened.setup(), saved);
}

#[test]
fn categorical_edits_reactivate_dormant_values_and_refresh_legacy_projection() {
    let mut source = State::new();
    source.p.hollow = 1.0;
    source.enable_canonical_materials().unwrap();
    source.canonical.as_mut().unwrap().scene.coat.values.weight = 0.73.into();
    assert!(source.body_set(0, "hollow", 0.0));
    let owner = &source.canonical.as_ref().unwrap().scene;
    assert!(owner.coat.enabled);
    assert_eq!(owner.coat.values.weight.as_constant(), Some(&0.73));
    assert_eq!(source.p.coat, 0.73, "the renderer projection changes in the same edit");
    assert_eq!(source.setup().settings.coat, 0.73);
    let saved = source.setup();
    let mut reopened = State::new();
    reopened.apply_setup(&saved).unwrap();
    assert_eq!(reopened.setup(), saved);
}

#[test]
fn material_edits_use_stored_clamps_and_ignore_nan() {
    let mut source = assembly(&[]);
    source.enable_canonical_materials().unwrap();
    assert!(source.body_set(0, "rough", 2.0));
    assert_eq!(source.p.rough, 1.0);
    assert_eq!(source.canonical.as_ref().unwrap().scene.specular.values.roughness.as_constant(), Some(&1.0));
    let before = source.setup();
    assert!(source.body_set(0, "rough", f64::NAN));
    assert!(source.body_set(1, "rough", f64::NAN));
    assert!(!source.body_own(1));
    assert_eq!(source.setup(), before);
}

#[test]
fn scene_and_body_exports_enforce_the_same_legacy_hard_limits() {
    let scene = MaterialSpec::from_legacy(&State::new().p, MaterialId::from("material:scene"));
    let mut body = scene.clone();
    body.id = MaterialId::from("material:body:2");
    for spec in [scene, body] {
        let mut spec = spec;
        spec.specular.values.ior = 5.0.into();
        spec.transvolume.values.ior = 5.0.into();
        let error = spec.to_legacy_settings().unwrap_err().to_string();
        assert!(error.contains("outside its hard limit"), "{error}");
    }
}

#[test]
fn duplicate_textual_body_aliases_are_rejected_atomically() {
    let mut source = assembly(&[("B2.material", "gold")]);
    source.enable_canonical_materials().unwrap();
    let mut saved = source.setup();
    let body = saved.canonical.as_ref().unwrap().bodies["2"].clone();
    saved.canonical.as_mut().unwrap().bodies.insert("02".into(), body);
    let before = source.setup();
    let error = source.apply_setup(&saved).unwrap_err();
    assert!(error.contains("more than once"), "{error}");
    assert_eq!(source.setup(), before);
}

#[test]
fn legacy_optical_priority_reconciles_canonical_switches_and_retains_values() {
    let mut source = State::new();
    source.p.grin = 0.2;
    source.enable_canonical_materials().unwrap();
    assert!(source.body_set(0, "biref", 0.1));
    let scene = &source.canonical.as_ref().unwrap().scene;
    assert!(scene.special_optics.birefringence.enabled);
    assert!(!scene.special_optics.grin.enabled);
    assert_eq!(scene.special_optics.grin.values.strength.as_constant(), Some(&0.2));
    assert_eq!(source.p.biref, 0.1);
    assert_eq!(source.p.grin, 0.2, "the losing authored legacy value also remains dormant");
    let saved = source.setup();
    let mut reopened = State::new();
    reopened.apply_setup(&saved).unwrap();
    assert_eq!(reopened.setup(), saved);
}

#[test]
fn owner_edits_revalidate_and_reproject_dependent_optical_patches() {
    let mut source = State::new();
    source.enable_canonical_materials().unwrap();
    let path = source.layer_csg_mut(0).unwrap().add_construction_part(&[], Part::default()).unwrap();
    source.set_construction_material(0, &path, Some(BTreeMap::from([("biref".into(), 0.1)]))).unwrap();
    assert!(source.body_set(0, "grin", 0.2));
    let scene = &source.canonical.as_ref().unwrap().scene;
    let node = source.layer_csg(0).construction.as_ref().unwrap().node(&path).unwrap();
    let patch = node.material_v2.as_ref().unwrap();
    let resolved = patch.apply(scene).unwrap();
    assert!(resolved.special_optics.birefringence.enabled);
    assert!(!resolved.special_optics.grin.enabled);
    assert_eq!(node.material.as_ref().unwrap()["grin"], 0.2);
    resolved.to_legacy_settings().unwrap();
    let saved = source.setup();
    let mut reopened = State::new();
    reopened.apply_setup(&saved).unwrap();
    assert_eq!(reopened.setup(), saved);
}

#[test]
fn family_defaults_replace_the_canonical_scene_projection() {
    let mut source = State::new();
    source.enable_canonical_materials().unwrap();
    let family = (0..source.families.len()).find(|index| *index != source.fam).unwrap();
    source.set_scene(family);
    let scene = &source.canonical.as_ref().unwrap().scene;
    assert_eq!(scene.specular.values.ior.as_constant(), Some(&source.p.ior));
    assert_eq!(scene.transvolume.values.absorption_coefficients.as_constant().unwrap().0, source.p.absorb);
    match &scene.transvolume.values.dispersion {
        DispersionModel::Cauchy { amount } => assert_eq!(amount.as_constant(), Some(&source.p.disp)),
        other => panic!("unexpected dispersion model {other:?}"),
    }
    assert_eq!(scene.compatibility.legacy.as_ref().unwrap().preset_index, Some(source.p.mat));
    let saved = source.setup();
    let mut reopened = State::new();
    reopened.apply_setup(&saved).unwrap();
    assert_eq!(reopened.setup(), saved);
}

#[test]
fn full_body_patch_edits_opt_in_transactionally_and_round_trip() {
    let mut source = State::new();
    assert!(source.canonical.is_none());
    let inherited = source.body_material_spec(0);
    assert_eq!(inherited, MaterialSpec::from_legacy(&source.p, MaterialId::from("material:scene")));
    let mut patch = MaterialPatch::new(inherited.id.clone());
    scalar_override(&mut patch, "specular.roughness", 0.37.into());
    patch.properties.insert("emission.color".into(), PropertyValue::Color(Color([0.2, 0.6, 0.9]).into()));
    scalar_override(&mut patch, "emission.volume.strength", 2.0.into());
    feature_override(&mut patch, "emission", true);
    source.apply_body_material_patch(0, patch).unwrap();
    assert!(source.canonical.is_some());
    let edited = source.body_material_spec(0);
    assert_eq!(edited.specular.values.roughness.as_constant(), Some(&0.37));
    assert_eq!(edited.emission.values.color.as_constant(), Some(&Color([0.2, 0.6, 0.9])));
    assert!(edited.emission.enabled);
    assert_eq!(source.p.rough, edited.to_legacy_settings().unwrap().rough);

    let saved = source.setup();
    let text = serde_json::to_string_pretty(&saved).unwrap();
    let mut reopened = State::new();
    reopened.apply_setup_json(&text).unwrap();
    assert_eq!(reopened.body_material_spec(0), edited);
    assert_eq!(reopened.setup(), saved);
    assert_eq!(reopened.setup_json(), text);
}

#[test]
fn first_patch_on_an_inherited_body_creates_only_that_owner() {
    let mut source = assembly(&[]);
    assert!(!source.body_own(1) && !source.body_own(2));
    assert_eq!(source.body_material_spec(1), source.body_material_spec(0));
    assert_eq!(source.body_material_spec(2), source.body_material_spec(0));
    let scene = source.body_material_spec(0);
    let mut patch = MaterialPatch::new(scene.id.clone());
    scalar_override(&mut patch, "specular.roughness", 0.43.into());
    source.apply_body_material_patch(1, patch).unwrap();
    assert!(source.body_own(1));
    assert!(!source.body_own(2));
    assert_eq!(source.body_material_spec(0), scene);
    assert_eq!(source.body_material_spec(2), scene);
    let body = source.body_material_spec(1);
    assert_eq!(body.id, MaterialId::from("material:body:2"));
    assert_eq!(body.specular.values.roughness.as_constant(), Some(&0.43));

    let saved = source.setup();
    let mut reopened = State::new();
    reopened.apply_setup(&saved).unwrap();
    assert_eq!(reopened.body_material_spec(1), body);
    assert_eq!(reopened.body_material_spec(2), scene);
    assert_eq!(reopened.setup(), saved);
}

#[test]
fn rejected_full_body_patches_leave_legacy_and_canonical_state_unchanged() {
    for opted_in in [false, true] {
        let mut source = assembly(&[]);
        if opted_in { source.enable_canonical_materials().unwrap(); }
        let before = source.setup();
        let spec = source.body_material_spec(1);
        let mut unsupported = MaterialPatch::new(spec.id.clone());
        feature_override(&mut unsupported, "sheen", true);
        let error = source.apply_body_material_patch(1, unsupported).unwrap_err();
        assert!(error.contains("legacy bridge"), "{error}");
        assert_eq!(source.setup(), before);
        assert!(!source.body_own(1));
        assert_eq!(source.canonical.is_some(), opted_in);

        let mut wrong_base = MaterialPatch::new(MaterialId::from("material:wrong"));
        scalar_override(&mut wrong_base, "specular.roughness", 0.2.into());
        assert!(source.apply_body_material_patch(1, wrong_base).is_err());
        assert_eq!(source.setup(), before);
        let empty = MaterialPatch::new(source.body_material_spec(0).id.clone());
        assert!(source.apply_body_material_patch(99, empty).is_err());
        assert_eq!(source.setup(), before);
    }
}
