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
            // the march is the volume's own; hits settle in loops; work done once from the sliders is done once a pixel
            assert!(!src.contains("sdfDist(p); }
  {{PROFILE") && src.contains("for (var sdfS = 0u;"), "{}/{}", f.id, p.id);
            if f.id == "gems" { assert!(!src.contains("if (!sdf_gem_ready)") && src.contains("  sdf_gem_setup();
"), "{}: the stone's setup is not hoisted", p.id); }
            assert!(src.contains("fn mapN(p: vec3f) -> f32 { return sdfDist(p); }") && src.contains("sdfNormal(toObj(p))"), "{}/{}", f.id, p.id);
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
            assert!(src.contains("fn mapN(p: vec3f) -> f32 { return sdfDist(p); }"), "{v:?}: not woven");
            check(&format!("{v:?}"), &src);
        }
        let src = glassforms::sdf::weave(&tracer_source_lens(&m, v, false), &decl);
        assert!(src.contains("fn mapN(p: vec3f) -> f32 { return sdfDist(p); }"), "{v:?} (lens): not woven");
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
    // (the word is "csg" now; "sdf", from before the tree, still reads)
    assert!(text.contains("\"trace\":\"csg\""), "{text}");
    let mut other = gf_core::state::State::new();
    other.apply_setup(&serde_json::from_str(&text).unwrap()).unwrap();
    assert_eq!(other.shape().trace, gf_core::sdf::Trace::Csg);
    // a shape traced from its volume leaves the word out, so its setup is the file it always was
    assert!(st.apply_arg("trace", "volume"));
    assert!(!serde_json::to_string(&st.setup()).unwrap().contains("trace"));
}

// ---- the tree of formula parts (gf_core::csg)

fn formulas(fams: &[gf_core::presets::Family]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (fi, f) in fams.iter().enumerate() {
        for (pi, p) in f.presets.iter().enumerate() {
            if matches!(gf_core::sdf::support(f, p), gf_core::sdf::Support::Formula { .. }) { out.push((fi, pi)); }
        }
    }
    out
}

/// Every formula as a part of a tree: in the tracer, in the pass that measures the tree into a volume, and
/// in the one that measures the parts' boxes.
#[test]
fn every_formula_is_a_part() {
    let fams = gf_core::presets::families();
    let m = March { step_scale: 0.9, steps_out: 220, steps_in: 56, layers: 2 };
    let plain = tracer_source(&m, Variant::default(), false);
    let all = formulas(&fams);
    assert_eq!(all.len(), 77);
    for &k in &all {
        let name = format!("{}/{}", fams[k.0].id, fams[k.0].presets[k.1].id);
        let src = glassforms::sdf::weave(&plain, &gf_core::csg::wgsl(&fams, &[k], gf_core::sdf::BINDING, false));
        assert!(src.contains("let c = csgDist(p);") && src.contains("csgNormal(po)") && src.contains("for (var sdfS = 0u;"), "{name}");
        check(&format!("{name} (tree)"), &src);
        check(&format!("{name} (merge)"), &glassforms::csg::merge_source(&gf_core::csg::wgsl(&fams, &[k], 0, true)));
        check(&format!("{name} (extents)"), &glassforms::csg::extent_source(&gf_core::csg::wgsl(&fams, &[k], 0, false)));
    }
}

/// One tree holding many kinds at once (their names must not meet), in other ways of drawing too.
#[test]
fn many_kinds_in_one_tree() {
    let fams = gf_core::presets::families();
    let kinds: Vec<(usize, usize)> = formulas(&fams).into_iter().step_by(3).collect();
    let decl = gf_core::csg::wgsl(&fams, &kinds, gf_core::sdf::BINDING, false);
    let m = March { step_scale: 0.9, steps_out: 220, steps_in: 56, layers: 2 };
    check("many kinds", &glassforms::sdf::weave(&tracer_source(&m, Variant::default(), false), &decl));
    check("many kinds (merge)", &glassforms::csg::merge_source(&gf_core::csg::wgsl(&fams, &kinds, 0, true)));
    let mut vs = vec![Variant { light: LightFrom::Picture, backdrop: true, ..Default::default() }, Variant { clay: true, ..Default::default() }];
    for mat in gf_core::material::Variant::samples().into_iter().filter(|v| v.bodies == 0 && !v.bubbles).take(8) {
        vs.push(Variant { mat, ..Default::default() });
    }
    for v in vs {
        check(&format!("tree {v:?}"), &glassforms::sdf::weave(&tracer_source(&m, v, false), &decl));
        check(&format!("tree {v:?} (lens)"), &glassforms::sdf::weave(&tracer_source_lens(&m, v, false), &decl));
    }
}

#[test]
fn setups_keep_the_tree() {
    let mut st = gf_core::state::State::new();
    let alg = st.families.iter().position(|f| f.id == "algebraic").unwrap();
    st.set_scene(alg);
    assert!(st.apply_arg("trace", "csg"));
    assert!(st.apply_arg("csg", "studio/sphere@0.5,0,0@0,0,0@0.4@subtract@0.05; gems/brilliant@-0.5,0,0@0,0,0@0.3@union@0@1"));
    st.csg.place.pos = [0.1, 0.2, 0.0];
    st.csg.parts[0].smooth_rel = true;
    let (tree, joined) = st.csg_tree();
    assert_eq!(tree.len(), 3);
    assert!(!joined);
    let text = serde_json::to_string(&st.setup()).unwrap();
    assert!(text.contains("\"subtract\"") && text.contains("csg_place") && text.contains("smooth_rel"), "{text}");
    let mut other = gf_core::state::State::new();
    other.apply_setup(&serde_json::from_str(&text).unwrap()).unwrap();
    assert_eq!(other.csg.parts, st.csg.parts);
    assert_eq!(other.csg.place, st.csg.place);
    // a volume object with parts: they are joined to it
    let att = st.families.iter().position(|f| f.id == "attractors").unwrap();
    st.set_scene(att);
    assert!(st.apply_arg("csgGrid", "5:studio/capsule"));
    let (tree, joined) = st.csg_tree();
    assert_eq!(tree.len(), 5);
    assert!(joined);
    // nothing of the tree in a setup without one
    let mut plain = gf_core::state::State::new();
    plain.set_scene(alg);
    assert!(!serde_json::to_string(&plain.setup()).unwrap().contains("csg"));
}
