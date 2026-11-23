//! A material for each body: what the state keeps, how it is named on the command line, what follows a
//! layer when layers change places, what a saved setup holds, and the numbers the tracer is given.

use gf_core::material::{self, MB_FLOATS, MB_HEAD, MU_FLOATS};
use gf_core::state::State;

fn scene(extra: &[(&str, &str)]) -> State {
    let mut args: Vec<(String, String)> = [("scene", "knots"), ("preset", "trefoil"), ("layer", "algebraic:heart"), ("J12.op", "apart")]
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    args.extend(extra.iter().map(|(k, v)| (k.to_string(), v.to_string())));
    State::with_args(&args, true)
}

#[test]
fn a_scene_of_one_material_has_no_bodies_materials() {
    let s = scene(&[]);
    assert_eq!(s.bodies().len(), 2);
    assert!(s.body_materials().is_none(), "no body has a material of its own");
    assert!(!s.body_own(1));
    // one body: nothing to tell apart, whatever a layer has been given
    let mut one = State::with_args(&[("scene".into(), "knots".into()), ("layer".into(), "algebraic:heart".into()), ("B2.material".into(), "gold".into())], true);
    assert_eq!(one.bodies().len(), 1);
    assert!(one.body_materials().is_none(), "joined layers are one body, made of the scene's material");
    // set apart, the second layer's body has the material it was given
    assert!(one.apply_arg("J12.op", "apart"));
    assert!(one.body_materials().is_some());
}

#[test]
fn a_body_is_given_a_material_by_name() {
    let s = scene(&[("B2.material", "gold")]);
    let m = s.body_materials().expect("the second body has its own");
    assert_eq!(m.len(), 2);
    assert_eq!(material::body_kind(m[0]), 0.0, "the first body is the scene's glass");
    assert_eq!(material::body_kind(m[1]), 1.0, "the second is a metal");
    assert_eq!(m[1].mat as usize, material::named_index("gold").unwrap());
    // the scene's own settings are untouched
    assert_eq!(s.p.metal, 0.0);
    assert_eq!(s.p.mat, 0.0);
}

#[test]
fn older_scene_and_body_recipes_reopen_with_their_authored_settings() {
    let source = scene(&[("material", "gold"), ("rough", "0.19"), ("B2.material", "bisque"), ("B2.bodyR", "0.7")]);
    let original = source.setup();
    let mut file = serde_json::to_value(&original).unwrap();
    for (path, index) in [("/settings", 59), ("/bodies/2", 103)] {
        let values = file.pointer_mut(path).unwrap().as_object_mut().unwrap();
        values.retain(|k, _| !k.starts_with("grating"));
        values.insert("mat".into(), index.into());
    }
    let mut loaded = scene(&[]);
    loaded.apply_setup_json(&file.to_string()).unwrap();
    assert_eq!(loaded.p, source.p);
    assert_eq!(loaded.body_settings(1).mat, source.body_settings(1).mat);
    assert_eq!(loaded.body_settings(1).body_r, 0.7);
    loaded.apply_setup_json(&loaded.setup_json()).unwrap();
    assert_eq!(loaded.setup(), original);

    // The intervening writer included the grating controls, even when the disc
    // was unused. Its aluminium and optical-disc positions are already current.
    let current = scene(&[("material", "aluminium"), ("B2.material", "discgrating")]);
    loaded.apply_setup_json(&current.setup_json()).unwrap();
    assert_eq!(loaded.setup(), current.setup());
}

#[test]
fn settings_given_to_a_body_make_its_material_its_own() {
    let mut s = scene(&[("rough", "0.1")]);
    assert!(s.apply_arg("B2.rough", "0.4"));
    let m = s.body_materials().expect("own now");
    assert_eq!(m[0].rough, 0.1);
    assert_eq!(m[1].rough, 0.4);
    assert_eq!(m[1].ior, m[0].ior, "it started as the scene's material");
    // the first body's are the scene's
    assert!(s.apply_arg("B1.rough", "0.2"));
    assert_eq!(s.p.rough, 0.2);
    // not part of a material: refused
    assert!(!s.apply_arg("B2.fov", "30"));
    assert!(!s.apply_arg("B2.nonsense", "1"));
    // and taken away again
    assert!(s.apply_arg("B2.own", "0"));
    assert!(s.body_materials().is_none());
    assert!(!s.apply_arg("B1.own", "1"), "the first body is always the scene's material");
}

#[test]
fn a_bodys_sliders_are_knobs() {
    let mut s = scene(&[("B2.material", "jade")]);
    let k = s.knob_list().into_iter().find(|k| k.key == "B2.sss").expect("the body's milkiness is a knob");
    assert_eq!(s.get(&k.r), s.body_settings(1).sss);
    s.set(&k.r, 0.25);
    assert_eq!(s.body_settings(1).sss, 0.25);
    assert_ne!(s.p.sss, 0.25);
    assert!(s.knob_list().iter().any(|k| k.key == "B2.ior"));
    // no material of its own: no knobs, and a leftover knob does nothing
    s.set_body_own(1, false);
    assert!(!s.knob_list().iter().any(|k| k.key.starts_with("B2.")));
    s.set(&k.r, 0.9);
    assert!(s.body_materials().is_none());
}

#[test]
fn a_setup_keeps_each_bodys_material() {
    let s = scene(&[("B2.material", "gold"), ("B2.rough", "0.3")]);
    let text = s.setup_json();
    assert!(text.contains("\"bodies\""));
    let mut t = State::new();
    t.apply_setup_json(&text).expect("the setup reads back");
    let (a, b) = (s.body_materials().unwrap(), t.body_materials().expect("the body's material came back"));
    assert_eq!(a.len(), b.len());
    for name in material::body_names() {
        assert_eq!(a[1].get(name), b[1].get(name), "{name}");
    }
    // a scene of one material saves nothing about bodies, and a setup without them clears them
    let plain = scene(&[]);
    let text = plain.setup_json();
    assert!(!text.contains("\"bodies\""));
    t.apply_setup_json(&text).unwrap();
    assert!(t.body_materials().is_none());
}

#[test]
fn a_material_goes_with_its_layer() {
    // three layers, each a body; the third is gold
    let mut s = scene(&[("layer", "algebraic:tanglecube"), ("J13.op", "apart"), ("J23.op", "apart"), ("B3.material", "gold"), ("track", "B3.rough:0:0.5")]);
    assert_eq!(s.bodies().len(), 3);
    assert_eq!(material::body_kind(s.body_materials().unwrap()[2]), 1.0);
    assert!(s.tracks.contains_key("B3.rough"));
    // the second and third change places: the gold is now the second's
    s.swap_layers(1, 2);
    let m = s.body_materials().unwrap();
    assert_eq!(material::body_kind(m[1]), 1.0);
    assert_eq!(material::body_kind(m[2]), 0.0);
    assert!(s.tracks.contains_key("B2.rough") && !s.tracks.contains_key("B3.rough"));
    // the first is taken away: the gold layer is first, and its material is the scene's
    assert!(s.remove_layer(0));
    assert_eq!(s.p.metal, s.body_settings(0).metal);
    assert!(s.p.metal >= 0.5, "the layer that became first brought its material with it");
    assert!(s.body_materials().is_none(), "the other body never had its own");
}

#[test]
fn the_tracer_is_given_each_bodys_numbers() {
    let s = scene(&[("B2.material", "gold"), ("density", "2")]);
    let m = s.body_materials().unwrap();
    let u = material::bodies_uniforms(&m, &[]);
    assert_eq!(u.len(), MB_FLOATS);
    assert_eq!(MB_HEAD, 48);
    // kind, index and dispersion of each body
    assert_eq!((u[0], u[1]), (0.0, 1.0));
    assert_eq!(u[32].to_bits(), 2, "the legacy scene medium keeps its deterministic token");
    assert_eq!(u[36].to_bits(), 1, "the legacy body medium uses the same from_legacy ID convention");
    assert_eq!(u[33].to_bits() as i32, 0);
    assert_eq!(u[4], s.p.ior as f32);
    assert_eq!(u[8], s.p.disp as f32);
    // what the first body's inside absorbs: its tint by its depth; a metal's inside absorbs nothing
    for c in 0..3 {
        assert_eq!(u[16 + c], (s.p.absorb[c] * 2.0) as f32);
        assert_eq!(u[20 + c], 0.0);
    }
    // then each body's material as a single material's numbers are laid out
    assert_eq!(&u[MB_HEAD..MB_HEAD + MU_FLOATS], &material::uniforms(m[0], 1.0)[..]);
    assert_eq!(&u[MB_HEAD + MU_FLOATS..MB_HEAD + 2 * MU_FLOATS], &material::uniforms(m[1], 1.0)[..]);
    // the tracer for several bodies is its own variant, whatever the materials are
    let parts = material::bodies_parts(&material::Variant::of_bodies(&m), "", "const COUNT: i32 = {{COUNT}};", false);
    assert!(parts.decl.contains("COUNT: i32 = 2;") && parts.shade.starts_with("shadeB") && parts.min_layers == 2);
}

#[test]
fn body_medium_rows_are_raw_signed_metadata_not_optical_floats() {
    let s = scene(&[("B2.material", "gold")]);
    let m = s.body_materials().unwrap();
    let media = [
        gf_core::pbr::CompiledMedium { identity: 9, interface_priority: -17 },
        gf_core::pbr::CompiledMedium { identity: 9, interface_priority: 31 },
    ];
    let optical = material::bodies_uniforms(&m, &[]);
    let u = material::bodies_uniforms_with_media(&m, &[], &media);
    assert_eq!(u[32].to_bits(), 9);
    assert_eq!(u[33].to_bits() as i32, -17);
    assert_eq!(u[36].to_bits(), 9);
    assert_eq!(u[37].to_bits() as i32, 31);
    assert_eq!((u[34].to_bits(), u[35].to_bits()), (0, 0));
    for k in 0..2 {
        assert_eq!(u[4 + k], optical[4 + k]);
        assert_eq!(u[8 + k], optical[8 + k]);
        assert_eq!(&u[MB_HEAD + k * MU_FLOATS..MB_HEAD + (k + 1) * MU_FLOATS],
                   &optical[MB_HEAD + k * MU_FLOATS..MB_HEAD + (k + 1) * MU_FLOATS]);
    }
}

#[test]
fn every_part_of_a_material_is_drawn_on_a_body_among_several() {
    // each named material that has colour by wavelength, a graded index, double refraction, bubbles or
    // something inside it keeps that part as the heart of a knot
    for (name, bit) in [("didymium", 1u32), ("gradient", 2), ("calcite", 4), ("crackedquartz", 8), ("whiteopal", 8), ("tigerseye", 8), ("agate", 8), ("piezoquartz", 24)] {
        let s = scene(&[("B2.material", name)]);
        let m = s.body_materials().unwrap();
        assert!(material::body_lacks(m[1]).is_empty(), "{name}");
        assert_eq!(material::body_bits(m[1]) & bit, bit, "{name}");
        assert_eq!(material::bodies_uniforms(&m, &[])[13], material::body_bits(m[1]) as f32, "{name}");
        let v = material::Variant::of_bodies(&m);
        assert_eq!((v.spectrum, v.grin, v.biref, v.inner.any()), (bit & 1 != 0, bit & 2 != 0, bit & 4 != 0, bit & 8 != 0), "{name}");
    }
    let s = scene(&[("B2.material", "foam")]);
    assert!(material::body_variant(s.body_materials().unwrap()[1]).bubbles);
    // A body can request true scattering alongside the other material effects.
    let s = scene(&[("B2.material", "whiteopal"), ("B2.sssWalk", "1")]);
    let m = s.body_materials().unwrap();
    assert!(material::body_lacks(m[1]).is_empty());
    assert!(material::Variant::of_bodies(&m).walk);
    assert_ne!(material::body_bits(m[1]) & 32, 0);
    assert!(material::body_variant(m[1]).inner.play && !material::body_variant(m[1]).walk);
    // each body is told its own numbers of what lies inside it
    let u = material::bodies_inner(&m, &[], 0.001);
    let n = gf_core::inner::IU_FLOATS;
    assert_eq!(u.len(), material::MAX_BODIES * n);
    assert_ne!(&u[..n], &u[n..2 * n]);
}

#[test]
fn a_scene_that_glows_under_pressure_is_drawn_body_by_body() {
    // so that it glows where its bodies press on each other, though no body has a material of its own
    let s = scene(&[("material", "piezoquartz")]);
    assert!(s.body_materials().is_some_and(|m| m.len() == 2));
    assert!(scene(&[("material", "quartz")]).body_materials().is_none());
}

#[test]
fn opaque_finishes_remain_independent_of_the_glass_in_a_saved_scene() {
    assert_eq!(material::named_index("bisque"), Some(104));
    for (name, index) in [("modelclay", 105), ("moulded", 106), ("satinpaint", 107)] {
        let s = scene(&[("B2.material", name)]);
        let mut restored = scene(&[]);
        restored.apply_setup_json(&s.setup_json()).unwrap();
        let bodies = restored.body_materials().unwrap();
        assert_eq!(bodies[1].mat, index as f64, "{name}");
        assert_eq!(material::NAMED[index].id, name);
        assert_eq!(material::body_kind(bodies[0]), 0.0);
        assert_eq!(material::body_kind(bodies[1]), 4.0);
        assert_eq!(bodies[1].ceramic_through, 0.0);
        let variant = material::Variant::of_bodies(&bodies);
        assert!(variant.paint_bodies && !variant.paint_only);
    }
}

#[test]
fn a_conductor_keeps_its_sampled_surface_when_glass_is_added() {
    let state = scene(&[("material", "crown"), ("B2.material", "copper"), ("B2.rough", ".65"), ("B2.roughRef", "1")]);
    let materials = state.body_materials().unwrap();
    let variant = material::Variant::of_bodies(&materials);
    assert!(variant.sampled_metal);
    assert!(!variant.rough_ref);
    assert_eq!(material::body_bits(materials[0]) & 128, 0);
    assert_eq!(material::body_bits(materials[1]) & 192, 128);
}

#[test]
fn panel_shadow_specialization_tracks_the_only_transmitting_material() {
    let glass = scene(&[("material", "crown")]).p;
    let iron = scene(&[("material", "iron")]).p;
    let clay = scene(&[("material", "modelclay")]).p;
    let medium = |bodies: &[&gf_core::settings::Settings]| material::Variant::of_bodies(bodies).paint_shadow_medium;
    assert_eq!(medium(&[&glass, &iron, &clay]), Some(0));
    assert_eq!(medium(&[&clay, &glass, &iron]), Some(1));
    assert_eq!(medium(&[&iron, &clay, &glass]), Some(2));
    assert_eq!(medium(&[&glass, &iron]), None);
    assert_eq!(medium(&[&iron, &clay]), None);
    assert_eq!(medium(&[&glass, &clay, &glass]), None);
}
