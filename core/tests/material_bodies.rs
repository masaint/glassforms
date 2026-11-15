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
    assert_eq!(MB_HEAD, 32);
    // kind, index and dispersion of each body
    assert_eq!((u[0], u[1]), (0.0, 1.0));
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
