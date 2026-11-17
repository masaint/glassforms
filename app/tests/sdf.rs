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
        assert!(src.contains("let c = csgDist(p);") && src.contains("csgNormal(po, near)") && src.contains("for (var sdfS = 0u;"), "{name}");
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

/// Undo puts a moved part back (the parts are in the setup an undo entry is), and leaves the part selected.
#[test]
fn undo_puts_a_moved_part_back() {
    let mut st = gf_core::state::State::new();
    let studio = st.families.iter().position(|f| f.id == "studio").unwrap();
    st.set_scene(studio);
    assert!(st.apply_arg("trace", "csg"));
    assert!(st.apply_arg("csg", "studio/sphere@0.3,0,0@0,0,0@0.5"));
    st.csg.sel = Some(0);
    let mut h = gf_core::history::History::default();
    h.settle(&mut st);
    // a drag of the part (as the handles write it), then the quiet moment after it
    st.csg.parts[0].place.pos = [0.6, 0.2, -0.1];
    h.settle(&mut st);
    // and a move of the whole shape
    st.csg.place.pos = [0.0, 0.3, 0.0];
    h.settle(&mut st);
    assert!(h.undo(&mut st));
    assert_eq!(st.csg.place.pos, [0.0; 3]);
    assert_eq!(st.csg.parts[0].place.pos, [0.6, 0.2, -0.1]);
    assert!(h.undo(&mut st));
    assert_eq!(st.csg.parts[0].place.pos, [0.3, 0.0, 0.0]);
    assert_eq!(st.csg.sel, Some(0));
    assert!(h.redo(&mut st));
    assert_eq!(st.csg.parts[0].place.pos, [0.6, 0.2, -0.1]);
}

// ---- every shape as a part: grid parts (shapes with no formula, each read from a volume of its own)

/// A tree holding grid parts beside formula parts: in the tracer (in other ways of drawing too), in the
/// pass that measures it into a volume and in the one that measures the boxes; read as a tree of plain
/// unions and not.
#[test]
fn grid_parts_weave_into_the_tree() {
    let fams = gf_core::presets::families();
    let mut kinds: Vec<(usize, usize)> = formulas(&fams).into_iter().step_by(19).collect();
    kinds.push(gf_core::csg::GRID_KIND);
    let m = March { step_scale: 0.9, steps_out: 220, steps_in: 56, layers: 2 };
    for plain in [false, true] {
        let decl = gf_core::csg::wgsl_tree(&fams, &kinds, gf_core::sdf::BINDING, false, plain);
        assert!(decl.contains("fn csgGridAt(") && decl.contains(&format!("@binding({})", gf_core::csg::GRID_BINDING_TRACER)), "plain {plain}");
        let src = glassforms::sdf::weave(&tracer_source(&m, Variant::default(), false), &decl);
        assert!(src.contains("csgNormal(po, near)"));
        check(&format!("grid tree, plain {plain}"), &src);
        check(&format!("grid tree (merge), plain {plain}"), &glassforms::csg::merge_source(&gf_core::csg::wgsl_tree(&fams, &kinds, 0, true, plain)));
        check(&format!("grid tree (extents), plain {plain}"), &glassforms::csg::extent_source(&gf_core::csg::wgsl_tree(&fams, &kinds, 0, false, plain)));
        for v in [Variant { light: LightFrom::Picture, backdrop: true, ..Default::default() }, Variant { clay: true, ..Default::default() }] {
            check(&format!("grid tree {v:?}, plain {plain}"), &glassforms::sdf::weave(&tracer_source(&m, v, false), &decl));
            check(&format!("grid tree {v:?} (lens), plain {plain}"), &glassforms::sdf::weave(&tracer_source_lens(&m, v, false), &decl));
        }
    }
    // a tree of formulas alone holds no grid reading
    assert!(!gf_core::csg::wgsl(&fams, &kinds[..kinds.len() - 1], gf_core::sdf::BINDING, false).contains("csgGridAt"));
}

/// The shape up with no formula, traced as "csg", is the tree's first part as a grid part reading its own
/// volume; parts with no formula are grid parts of their own, each given a volume in turn, up to the cap.
#[test]
fn every_shape_is_a_part_of_the_tree() {
    use gf_core::csg::GridFrom;
    let mut st = gf_core::state::State::new();
    let att = st.families.iter().position(|f| f.id == "attractors").unwrap();
    st.set_scene(att);
    assert!(st.apply_arg("trace", "csg"));
    assert!(st.apply_arg("csg", "studio/sphere@0.3,0,0@0,0,0@0.5@subtract@0.05; knots/trefoil@0,0,0@0,0,0@0.8@intersect@0@0@256; models/teapot@0,0.4,0@0,0,0@0.5"));
    let (tree, joined) = st.csg_tree();
    assert!(!joined);
    assert_eq!(tree.len(), 4);
    assert_eq!(tree[0].grid.as_ref().map(|g| (g.slot, g.from)), Some((0, GridFrom::Up)));
    assert!(tree[1].grid.is_none());
    assert_eq!(tree[2].grid.as_ref().map(|g| (g.slot, g.res, g.from)), Some((1, 256, GridFrom::Part(1))));
    assert_eq!(tree[3].grid.as_ref().map(|g| (g.slot, g.from)), Some((2, GridFrom::Part(2))));
    assert!(!gf_core::csg::plain(&tree));
    // the memory the parts' own volumes take
    let (bytes, own) = st.csg_grid_bytes();
    assert_eq!(own, 2);
    assert_eq!(bytes, gf_core::csg::grid_bytes(256) + gf_core::csg::grid_bytes(gf_core::csg::GRID_RES));
    // a formula reshaped by the surface studio is a grid part too (its studio-shaped volume)
    let studio = st.families.iter().position(|f| f.id == "studio").unwrap();
    st.set_scene(studio);
    assert!(st.apply_arg("trace", "csg"));
    assert!(st.csg_tree().0[0].grid.is_none());
    // past the cap: at most MAX_GRIDS grid parts, the rest left out
    let many: Vec<String> = (0..10).map(|i| format!("knots/trefoil@{},0,0@0,0,0@0.2", -1.0 + 0.2 * i as f64)).collect();
    assert!(st.apply_arg("csg", &many.join(";")));
    let (tree, _, left) = st.csg_tree_counted();
    assert_eq!(tree.iter().filter(|p| p.grid.is_some()).count(), gf_core::csg::MAX_GRIDS);
    assert_eq!(left, 10 - gf_core::csg::MAX_GRIDS);
    // and past the memory: eight of the finest are more than the budget allows
    let fine: Vec<String> = (0..8).map(|i| format!("knots/trefoil@{},0,0@0,0,0@0.2@union@0@0@768", -1.0 + 0.2 * i as f64)).collect();
    assert!(st.apply_arg("csg", &fine.join(";")));
    let (tree, _, left) = st.csg_tree_counted();
    let used: u64 = tree.iter().filter_map(|p| p.grid.as_ref()).map(|g| gf_core::csg::grid_bytes(g.res)).sum();
    assert!(left > 0 && used <= gf_core::csg::GRID_BUDGET, "{left} {used}");
}

/// A part's job: a curve part is a curve built as the shape up would be, a model part a model, and the
/// shape up is left as it was.
#[test]
fn a_grid_part_is_built_as_its_shape_would_be() {
    let mut st = gf_core::state::State::new();
    let studio = st.families.iter().position(|f| f.id == "studio").unwrap();
    st.set_scene(studio);
    let before = serde_json::to_string(&st.setup()).unwrap();
    let knot = gf_core::csg::Part { family: "knots".into(), preset: "trefoil".into(), thick: Some(1.5), ..Default::default() };
    let job = st.part_job(&knot, 256).unwrap();
    assert!(matches!(job.shape, gf_core::state::ShapeJob::Curve { .. }) && job.res == 256);
    let teapot = gf_core::csg::Part { family: "models".into(), preset: "teapot".into(), ..Default::default() };
    assert!(matches!(st.part_job(&teapot, 384).unwrap().shape, gf_core::state::ShapeJob::Mesh(_)));
    assert_eq!(serde_json::to_string(&st.setup()).unwrap(), before);
    // what a grid part is made from: its tube's thickness is, its placement is not
    let mut moved = knot.clone();
    moved.place.pos = [0.4, 0.0, 0.0];
    assert_eq!(moved.grid_sig(), knot.grid_sig());
    let thicker = gf_core::csg::Part { thick: Some(2.0), ..knot.clone() };
    assert_ne!(thicker.grid_sig(), knot.grid_sig());
}

/// Setups keep grid parts (their shape, sliders, voxels, tube and placement); setups from before read as
/// they did.
#[test]
fn setups_keep_grid_parts() {
    let mut st = gf_core::state::State::new();
    let att = st.families.iter().position(|f| f.id == "attractors").unwrap();
    st.set_scene(att);
    assert!(st.apply_arg("trace", "csg"));
    assert!(st.apply_arg("csg", "knots/trefoil@0.2,0,0@0,30,0@0.6@subtract@0.04@0@320; models/teapot@0,0.4,0@0,0,0@0.5"));
    st.csg.parts[0].thick = Some(1.4);
    st.csg.parts[0].values.insert("p".into(), 3.0);
    let text = serde_json::to_string(&st.setup()).unwrap();
    assert!(text.contains("\"res\":320") && text.contains("\"thick\":1.4") && text.contains("\"trace\":\"csg\""), "{text}");
    let mut other = gf_core::state::State::new();
    other.apply_setup(&serde_json::from_str(&text).unwrap()).unwrap();
    assert_eq!(other.csg.parts, st.csg.parts);
    assert_eq!(other.shape().trace, gf_core::sdf::Trace::Csg);
    // a formula part leaves the grid's members out, so a setup of formula parts is the file it was
    let mut plain = gf_core::state::State::new();
    let alg = plain.families.iter().position(|f| f.id == "algebraic").unwrap();
    plain.set_scene(alg);
    assert!(plain.apply_arg("csg", "studio/sphere@0.5,0,0"));
    let t = serde_json::to_string(&plain.setup().csg).unwrap();
    assert!(["\"res\"", "thick", "melt", "len", "model"].iter().all(|w| !t.contains(w)), "{t}");
    // and an old setup's part (no such members) reads with the defaults
    let old: gf_core::csg::Part = serde_json::from_str(r#"{"family":"studio","preset":"sphere","values":{},"puff":0.0,"place":{"pos":[0,0,0],"rot":[0,0,0],"size":1.0},"op":"union","smooth":0.0,"level":0}"#).unwrap();
    assert_eq!(old, gf_core::csg::Part::default());
}

/// Undo puts a grid part back: moved, turned, its detail and its tube changed.
#[test]
fn undo_puts_a_grid_part_back() {
    let mut st = gf_core::state::State::new();
    let att = st.families.iter().position(|f| f.id == "attractors").unwrap();
    st.set_scene(att);
    assert!(st.apply_arg("trace", "csg"));
    assert!(st.apply_arg("csg", "knots/trefoil@0.3,0,0@0,0,0@0.5@subtract@0.05"));
    st.csg.sel = Some(0);
    let mut h = gf_core::history::History::default();
    h.settle(&mut st);
    st.csg.parts[0].place.pos = [0.5, 0.1, 0.0];
    st.csg.parts[0].place.rot = [0.0, 45.0, 0.0];
    h.settle(&mut st);
    st.csg.parts[0].res = 512;
    st.csg.parts[0].thick = Some(2.0);
    h.settle(&mut st);
    assert!(h.undo(&mut st));
    assert_eq!((st.csg.parts[0].res, st.csg.parts[0].thick), (0, None));
    assert_eq!(st.csg.parts[0].place.pos, [0.5, 0.1, 0.0]);
    assert!(h.undo(&mut st));
    assert_eq!(st.csg.parts[0].place.pos, [0.3, 0.0, 0.0]);
    assert_eq!(st.csg.parts[0].place.rot, [0.0; 3]);
    assert_eq!(st.csg.sel, Some(0));
    assert!(h.redo(&mut st) && h.redo(&mut st));
    assert_eq!((st.csg.parts[0].res, st.csg.parts[0].thick), (512, Some(2.0)));
}