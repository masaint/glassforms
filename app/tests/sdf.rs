//! The tracer that reads a shape's formula (gf_core::sdf, app/src/sdf.rs): built for every formula of
//! every family it parses and validates, in the ways of drawing it is woven into. Needs no GPU.

use glassforms::render::{gem_weave, tracer_source, tracer_source_lens, LightFrom, March, Variant};

fn check(name: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name} does not parse:\n{}", e.emit_to_string(src)));
    let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
    v.validate(&module).unwrap_or_else(|e| panic!("{name} is not valid:\n{}", e.emit_to_string(src)));
}

#[test]
fn every_formula_weaves_into_the_tracer() {
    let m = March { step_scale: 0.9, steps_out: 220, steps_in: 56, layers: 2 };
    let plain = tracer_source(&m, Variant::default(), false);
    for f in gf_core::presets::families() {
        for p in &f.presets {
            let Some(decl) = gf_core::sdf::tracer_decl(&f, p) else { continue };
            let src = glassforms::sdf::weave(&plain, &decl);
            assert_ne!(src, plain, "{}/{}: nothing was woven in", f.id, p.id);
            assert!(src.contains("{ v = sdfDist(p); }") && src.contains("sdfNormal(toObj(p))"), "{}/{}", f.id, p.id);
            check(&format!("{}/{}", f.id, p.id), &src);
            if f.id == "gems" {
                let g = glassforms::sdf::weave(&gem_weave(&plain), &decl);
                assert!(g.contains("*mOut = gemWorld(nk);"), "{}: the facet normal is not left exact", p.id);
                check(&format!("{}/{} (stone)", f.id, p.id), &g);
            }
        }
    }
}

/// The formula goes into other ways of drawing too: lit by a picture with it behind, clay, a few
/// materials, the counting tracer and the tracer for any lens.
#[test]
fn the_formula_weaves_into_other_tracers() {
    let fams = gf_core::presets::families();
    let alg = fams.iter().find(|f| f.id == "algebraic").unwrap();
    let decl = gf_core::sdf::tracer_decl(alg, &alg.presets[0]).unwrap();
    let m = March { step_scale: 0.9, steps_out: 220, steps_in: 56, layers: 2 };
    let mut vs = vec![
        Variant { light: LightFrom::Picture, backdrop: true, ..Default::default() },
        Variant { clay: true, ..Default::default() },
    ];
    for mat in gf_core::material::Variant::samples().into_iter().filter(|v| v.bodies == 0 && !v.bubbles).take(12) {
        vs.push(Variant { mat, ..Default::default() });
    }
    for v in vs {
        assert!(glassforms::sdf::fits(&v));
        for profile in [false, true] {
            let src = glassforms::sdf::weave(&tracer_source(&m, v, profile), &decl);
            assert!(src.contains("{ v = sdfDist(p); }"), "{v:?}: not woven");
            check(&format!("{v:?}"), &src);
        }
        let src = glassforms::sdf::weave(&tracer_source_lens(&m, v, false), &decl);
        assert!(src.contains("{ v = sdfDist(p); }"), "{v:?} (lens): not woven");
        check(&format!("{v:?} (lens)"), &src);
    }
}

#[test]
fn layers_and_bubbles_read_the_volume() {
    let bodies = Variant { mat: gf_core::material::Variant { bodies: 2, ..Default::default() }, ..Default::default() };
    let bubbles = Variant { mat: gf_core::material::Variant { bubbles: true, ..Default::default() }, ..Default::default() };
    let grown = Variant { grown: true, ..Default::default() };
    for v in [bodies, bubbles, grown] { assert!(!glassforms::sdf::fits(&v), "{v:?}"); }
}

#[test]
fn setups_keep_the_way_a_shape_is_traced() {
    let mut st = gf_core::state::State::new();
    let alg = st.families.iter().position(|f| f.id == "algebraic").unwrap();
    st.set_scene(alg);
    assert!(st.apply_arg("trace", "sdf"));
    let s = st.setup();
    let text = serde_json::to_string(&s).unwrap();
    assert!(text.contains("\"trace\":\"sdf\""), "{text}");
    let mut other = gf_core::state::State::new();
    other.apply_setup(&serde_json::from_str(&text).unwrap()).unwrap();
    assert_eq!(other.shape().trace, gf_core::sdf::Trace::Sdf);
    // a shape traced from its volume leaves the word out, so its setup is the file it always was
    assert!(st.apply_arg("trace", "volume"));
    assert!(!serde_json::to_string(&st.setup()).unwrap().contains("trace"));
}
