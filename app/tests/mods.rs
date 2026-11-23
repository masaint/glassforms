//! Modifiers (gf_core::mods): the exact stack woven into a tree's tracer and merge pass, the baked pass,
//! setups, undo and objects among others. Needs no GPU.

use gf_core::mods::{Axes, Kind, Modifier, Mods};
use glassforms::render::{tracer_source, tracer_source_lens, LightFrom, March, Variant};

/// Every kind traced exactly.
const ALL: [u32; 11] = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12];

fn check(name: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name} does not parse:\n{}", e.emit_to_string(src)));
    let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
    v.validate(&module).unwrap_or_else(|e| panic!("{name} is not valid:\n{}", e.emit_to_string(src)));
}

/// A tree's text with its object's stack in it: in the tracer (other ways of drawing too) and the merge
/// pass, plain unions or not, with grid parts or without; and the baked pass alone.
#[test]
fn the_stack_weaves_into_a_tree() {
    let fams = gf_core::presets::families();
    let mut kinds: Vec<(usize, usize)> = Vec::new();
    for (fi, f) in fams.iter().enumerate() {
        for (pi, p) in f.presets.iter().enumerate() {
            if gf_core::csg::formula_part(f, p) { kinds.push((fi, pi)); }
        }
    }
    let mut kinds: Vec<(usize, usize)> = kinds.into_iter().step_by(23).collect();
    let m = March { step_scale: 0.9, steps_out: 220, steps_in: 56, layers: 2 };
    for grid in [false, true] {
        if grid { kinds.push(gf_core::csg::GRID_KIND); }
        for plain in [false, true] {
            let decl = gf_core::mods::wgsl_exact(&gf_core::csg::wgsl_tree(&fams, &kinds, gf_core::sdf::BINDING, false, plain), &ALL);
            assert!(decl.contains("fn csgDistW0(") && decl.contains("fn modBack("));
            // The distance texture drives the march; only the hit calls the exact gradient.
            let woven = glassforms::sdf::weave(&tracer_source(&m, Variant::default(), false), &decl);
            assert!(woven.contains("t += warpStep(ro + rd * t, rd);") && woven.contains("return warpNormal(p);"));
            assert!(!woven.contains("v = csgDistN(p);"));
            check(&format!("stack, grid {grid}, plain {plain}"), &glassforms::sdf::weave(&tracer_source(&m, Variant::default(), false), &decl));
            check(&format!("stack (merge), grid {grid}, plain {plain}"), &glassforms::csg::merge_source(&gf_core::mods::wgsl_exact(&gf_core::csg::wgsl_tree(&fams, &kinds, 0, true, plain), &ALL)));
            for v in [Variant { light: LightFrom::Picture, backdrop: true, ..Default::default() }, Variant { clay: true, ..Default::default() }] {
                check(&format!("stack {v:?}"), &glassforms::sdf::weave(&tracer_source(&m, v, false), &decl));
                check(&format!("stack {v:?} (lens)"), &glassforms::sdf::weave(&tracer_source_lens(&m, v, false), &decl));
            }
        }
    }
    let direct = gf_core::csg::direct_union(gf_core::csg::wgsl_tree(&fams, &kinds, gf_core::sdf::BINDING, false, true), &[0, 1]);
    for codes in [vec![1], vec![2, 1], vec![12]] {
        let variant = gf_core::mods::wgsl_exact(&direct, &codes);
        check("direct motif and its gradient", &glassforms::sdf::weave(&tracer_source(&m, Variant::default(), false), &variant));
    }
    let tree = gf_core::csg::wgsl_tree(&fams, &kinds, gf_core::sdf::BINDING, false, true);
    let direct = gf_core::csg::direct_union(tree, &[0]);
    let warp = gf_core::mods::wgsl_exact(&direct, &[1]);
    check("a directly evaluated warped part", &glassforms::sdf::weave(&tracer_source(&m, Variant::default(), false), &warp));
    let studio = fams.iter().position(|f| f.id == "studio").unwrap();
    let capsule = fams[studio].presets.iter().position(|p| p.id == "capsule").unwrap();
    let tree = gf_core::csg::wgsl_tree(&fams, &[(studio, capsule)], gf_core::sdf::BINDING, false, true);
    let direct = gf_core::csg::direct_union(tree, &[0]);
    let turns = gf_core::mods::wgsl_exact(&direct, &[2, 1]);
    let turns = gf_core::mods::specialize_turns(turns, &[1, 15, 10]);
    assert!(turns.contains("let sample = csgSingleSample(y);"));
    check("compound capsule through its live axes", &glassforms::sdf::weave(&tracer_source(&m, Variant::default(), false), &turns));
    check("baked pass", &gf_core::mods::wgsl_baked());
    // (one kind alone, and none with a repeat's room: the text without what it does not need)
    let tree = gf_core::csg::wgsl_tree(&fams, &kinds, gf_core::sdf::BINDING, false, false);
    for codes in [vec![1u32], vec![9], vec![7, 8], vec![12], vec![11, 12]] {
        check(&format!("stack of {codes:?}"), &glassforms::sdf::weave(&tracer_source(&m, Variant::default(), false), &gf_core::mods::wgsl_exact(&tree, &codes)));
    }
}

/// A stack with every kind, a lab among them, in the numbers both passes are handed (the sizes they read).
#[test]
fn the_numbers_fit_their_blocks() {
    let mut m = Mods::default();
    for k in [Kind::Twist, Kind::Swirl, Kind::Bend, Kind::Taper, Kind::Ripple, Kind::Mirror, Kind::Shell, Kind::Offset] { m.list.push(Modifier::of(k)); }
    let shape = [-0.4, -0.8, -0.4, 0.4, 0.8, 0.4];
    assert!(m.exact(&shape));
    let n = m.exact_numbers(&shape, &Default::default());
    assert_eq!(n.len(), gf_core::csg::PART_FLOATS * (gf_core::mods::TABLE_ENTRIES - gf_core::mods::TABLE_AT));
    assert_eq!(n[0], 8.0);
    let b = m.bake(&gf_core::mods::whole_box()).unwrap();
    for s in &b.stages {
        if let gf_core::mods::Stage::Warp(run) = s {
            let u = gf_core::mods::baked_numbers(run, 256, gf_core::HB, &Default::default());
            assert!(u.len() * 4 <= gf_core::mods::BAKED_STRIDE);
            assert_eq!(u[2] as usize, run.len());
        }
    }
}

#[test]
fn flat_cone_profiles_keep_a_finite_tangent() {
    for (top, expected) in [(0.75, [0., 1.]), (0.25, [-1., 0.]), (1.0, [1., 0.])] {
        let mut stage = Modifier::of(Kind::Wrap);
        stage.a.v = [0.75, 0.0, top];
        let mut mods = Mods::default();
        mods.list.push(stage);
        let packed = mods.exact_numbers(&gf_core::mods::whole_box(), &Default::default());
        let start = gf_core::csg::PART_FLOATS;
        assert!(packed.iter().all(|v| v.is_finite()));
        assert_eq!(&packed[start + 30..start + 32], &expected);
    }
}

fn studio(st: &gf_core::state::State) -> usize {
    st.families.iter().position(|f| f.id == "studio").unwrap()
}

/// A stack is kept in setups (left out while empty), comes back with every number, and undo puts a change
/// to it back; an object among others keeps its own.
#[test]
fn setups_and_undo_keep_the_stack() {
    let mut st = gf_core::state::State::new();
    let s = studio(&st);
    st.set_scene(s);
    assert!(!serde_json::to_string(&st.setup()).unwrap().contains("\"mods\""));
    let mut tw = Modifier::of(Kind::Twist);
    tw.a = Axes::new([30.0, 200.0, -10.0], false);
    let mut rp = Modifier::of(Kind::Repeat);
    rp.on = false;
    st.shape_mut().mods.list = vec![tw.clone(), rp];
    st.sculpt_mut().round = 0.02;
    let text = serde_json::to_string(&st.setup()).unwrap();
    assert!(text.contains("\"mods\"") && text.contains("surface_lab") && !text.contains("\"sculpt\""), "{text}");
    let mut other = gf_core::state::State::new();
    other.apply_setup(&serde_json::from_str(&text).unwrap()).unwrap();
    assert_eq!(other.shape().mods, st.shape().mods);
    assert_eq!(other.sculpt().round, 0.02);
    // undo
    let mut h = gf_core::history::History::default();
    h.settle(&mut st);
    st.shape_mut().mods.list[0].a.set(1, 90.0);
    h.settle(&mut st);
    st.shape_mut().mods.list.swap(0, 1);
    h.settle(&mut st);
    assert!(h.undo(&mut st));
    assert_eq!(st.shape().mods.list[0].kind, Kind::Twist);
    assert_eq!(st.shape().mods.list[0].a.get()[1], 90.0);
    assert!(h.undo(&mut st));
    assert_eq!(st.shape().mods.list[0].a, tw.a);
    assert!(h.redo(&mut st));
    assert_eq!(st.shape().mods.list[0].a.get()[1], 90.0);
    // a second object keeps its own stack, apart from the first's
    let (f, p) = (s, 0);
    st.create_object(f, p, true).unwrap();
    st.shape_mut().mods.list = vec![Modifier::of(Kind::Mirror)];
    let text = serde_json::to_string(&st.setup()).unwrap();
    let mut other = gf_core::state::State::new();
    other.apply_setup(&serde_json::from_str(&text).unwrap()).unwrap();
    assert_eq!(other.layer_mods(1).unwrap().list, vec![Modifier::of(Kind::Mirror)]);
    assert_eq!(other.layer_mods(0).unwrap().list.len(), 3);
}

/// Where a stack is applied: a volume object's to its own volume; a csg object's to its tree (none of it to
/// the shape's own volume), exactly unless a kind has no bound.
#[test]
fn objects_apply_their_stacks_where_they_should() {
    let mut st = gf_core::state::State::new();
    let s = studio(&st);
    st.set_scene(s);
    st.shape_mut().mods.list = vec![Modifier::of(Kind::Twist)];
    assert!(st.layer_bake(0).is_some());
    assert!(st.tree_mods(0).is_none());
    assert!(st.apply_arg("trace", "csg"));
    assert!(st.layer_bake(0).is_none());
    let m = st.tree_mods(0).unwrap();
    let shape = [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5];
    assert!(m.exact(&shape));
    st.shape_mut().mods.list.push(Modifier::of(Kind::Noise));
    assert!(!st.tree_mods(0).unwrap().exact(&shape));
    // (switched off, it does nothing at all: the object costs nothing)
    for e in st.shape_mut().mods.list.iter_mut() { e.on = false; }
    assert!(st.tree_mods(0).is_none());
    st.apply_arg("trace", "volume");
    assert!(st.layer_bake(0).is_none());
}

/// Strokes (gf_core::csg::Strokes): put into a tree's text they read and validate (with the stack too), and a
/// part's and the shape up's are kept in setups.
#[test]
fn strokes_weave_and_are_kept() {
    let fams = gf_core::presets::families();
    let kinds: Vec<(usize, usize)> = vec![(fams.iter().position(|f| f.id == "studio").unwrap(), 0), gf_core::csg::GRID_KIND];
    let m = March { step_scale: 0.9, steps_out: 220, steps_in: 56, layers: 2 };
    for plain in [false, true] {
        let tree = gf_core::csg::wgsl_tree_s(&fams, &kinds, gf_core::sdf::BINDING, false, plain, true);
        assert!(tree.contains("fn csgPartAt0(") && tree.contains("csgP[i].s1"));
        check(&format!("strokes, plain {plain}"), &glassforms::sdf::weave(&tracer_source(&m, Variant::default(), false), &tree));
        check(&format!("strokes and stack, plain {plain}"), &glassforms::sdf::weave(&tracer_source(&m, Variant::default(), false), &gf_core::mods::wgsl_exact(&tree, &ALL)));
        check(&format!("strokes (merge), plain {plain}"), &glassforms::csg::merge_source(&gf_core::csg::wgsl_tree_s(&fams, &kinds, 0, true, plain, true)));
        check(&format!("strokes (extents), plain {plain}"), &glassforms::csg::extent_source(&gf_core::csg::wgsl_tree_s(&fams, &kinds, 0, false, plain, true)));
    }
    let mut st = gf_core::state::State::new();
    let s = studio(&st);
    st.set_scene(s);
    assert!(st.apply_arg("trace", "csg"));
    assert!(st.apply_arg("csg", "studio/cylinder@0.5,0,0@0,0,0@0.4"));
    st.csg.parts[0].strokes = gf_core::csg::Strokes { star: 0.3, points: 7, mirror: [true, false, false], ..Default::default() };
    st.shape_mut().strokes = gf_core::csg::Strokes { shell: 0.05, hole: 0.2, ..Default::default() };
    let text = serde_json::to_string(&st.setup()).unwrap();
    let mut other = gf_core::state::State::new();
    other.apply_setup(&serde_json::from_str(&text).unwrap()).unwrap();
    assert_eq!(other.csg.parts[0].strokes, st.csg.parts[0].strokes);
    assert_eq!(other.shape().strokes, st.shape().strokes);
    let (tree, _) = other.csg_tree();
    assert!(gf_core::csg::any_strokes(&tree));
    // (none: nothing written)
    let mut plain = gf_core::state::State::new();
    plain.set_scene(s);
    assert!(!serde_json::to_string(&plain.setup()).unwrap().contains("strokes"));
    // a star and a cone are bounded; nothing else stretches
    assert_eq!(gf_core::csg::Strokes { shell: 0.1, hole: 0.3, round: 0.1, mirror: [true; 3], ..Default::default() }.lipschitz(), 1.0);
    assert!(gf_core::csg::Strokes { star: 0.5, cone: 0.5, ..Default::default() }.lipschitz() > 1.0);
}

/// Validate the studio specializations, including a beveled bar and perforated hub.
#[test]
fn profile_motifs_validate() {
    let fams = gf_core::presets::families();
    let studio = fams.iter().position(|f| f.id == "studio").unwrap();
    let kinds: Vec<_> = ["bar", "washer", "rectframe", "cylinder", "sphere", "wedge"].iter().map(|id| {
        (studio, fams[studio].presets.iter().position(|p| p.id == *id).unwrap())
    }).collect();
    let parts: Vec<_> = kinds.iter().map(|&(f, p)| {
        let preset = &fams[f].presets[p];
        let v = preset.params.iter().map(|p| p.value).collect::<Vec<_>>();
        gf_core::csg::Part::of(&fams[f],preset,&v,0.,Default::default()).resolve(&fams).unwrap()
    }).collect();
    let mut flags: Vec<_> = parts.iter().map(|p| gf_core::csg::studio_flags(p,&fams)).collect();
    flags[3] |= 0x4000;
    flags[5] |= 0x0800;
    let march = March { step_scale: 0.9, steps_out: 220, steps_in: 56, layers: 2 };
    for code in [8, 11, 12] {
        let base = gf_core::csg::wgsl_tree_s(&fams,&kinds,gf_core::sdf::BINDING,false,true,true);
        let direct = gf_core::csg::direct_union(base,&[0,1,2,3,4,5]);
        let tree = gf_core::csg::studio_motif(direct,&[0,1,2,3,4,5],&flags);
        assert!(tree.contains("fn motifSlope4("));
        let mut numbers=vec![0.;2*gf_core::csg::PART_FLOATS]; numbers[2]=1.; numbers[gf_core::csg::PART_FLOATS+1]=1.;
        let decl=gf_core::mods::repeat_overlap(gf_core::mods::wgsl_exact(&tree,&[code]),&[code],&numbers);
        let decl=gf_core::mods::specialize_profile(decl,Some([1,1,1,0,0]));
        check("analytic studio profile", &glassforms::sdf::weave(&tracer_source(&march,Variant::default(),false),&decl));
    }

    // The angular shortcut relies on a single centered, reflected wedge. With
    // another part in the union, even an unprepared one, its bounds cannot be used.
    for extra in [false,true] {
        let mut only = vec![kinds[5]];
        let mut route = vec![0];
        let mut layout = vec![flags[5] | 0x1000];
        if extra {
            only.push(kinds[0]);
            route.push(1);
            layout.push(0);
        }
        let base = gf_core::csg::wgsl_tree_s(&fams,&only,gf_core::sdf::BINDING,false,true,true);
        let tree = gf_core::csg::studio_motif(gf_core::csg::direct_union(base,&route),&route,&layout);
        let mut numbers = vec![0.;2*gf_core::csg::PART_FLOATS];
        numbers[2]=1.;
        let decl=gf_core::mods::repeat_overlap(gf_core::mods::wgsl_exact(&tree,&[8]),&[8],&numbers);
        let decl=gf_core::mods::specialize_profile(decl,Some([1,1,1,1,4]));
        assert_eq!(decl.contains("let near=m.q3.w/ratio"),!extra);
        check("bounded wedge sectors", &glassforms::sdf::weave(&tracer_source(&march,Variant::default(),false),&decl));
    }
}

#[test]
fn the_prepared_cone_uses_free_uniform_slots() {
    use glassforms::csg::TreeMods;
    let fams = gf_core::presets::families();
    let f = fams.iter().position(|f| f.id == "studio").unwrap();
    let mut parts = Vec::new();
    for (name, rot, size) in [("rectframe", [0.,0.,45.], 0.6), ("washer", [90.,0.,0.], 0.2)] {
        let preset = fams[f].presets.iter().find(|p| p.id == name).unwrap();
        let values = preset.params.iter().map(|p| p.value).collect::<Vec<_>>();
        let mut part = gf_core::csg::Part::of(&fams[f], preset, &values, 0., Default::default()).resolve(&fams).unwrap();
        part.place.rot = rot;
        part.place.size = size;
        parts.push(part);
    }
    let kinds = gf_core::csg::kinds(&parts);
    let object = Default::default();
    let mut cone = Modifier::of(Kind::Wrap);
    cone.mode = 1;
    cone.a.v = [1.15, 1.7, 0.42];
    cone.size = 0.17;
    let mut stack = Mods::default();
    stack.list.push(cone);
    let plan = TreeMods::plan(&fams, Some(&stack), &parts, &[], false, &object);
    assert_ne!(plan.tracer_key("test"), "test");
    let base = gf_core::csg::wgsl_tree_s(&fams,&kinds,gf_core::sdf::BINDING,false,true,false);
    let src = plan.tracer_text(base);
    let march = March { step_scale: 0.9, steps_out: 220, steps_in: 56, layers: 2 };
    let shader = glassforms::sdf::weave(&tracer_source(&march, Variant::default(), false), &src);
    assert!(shader.contains("var m = warpUnitNormal(warpSlope);"));
    assert!(shader.contains("csgU.parts[8 + j]"));
    assert!(shader.contains("let delta = profileFinish(ro + rd * depth, rd, found);"));
    assert!(shader.contains("let radialSlope = tangent.x*g.y + tangent.y*g.z;"));
    check("uniform cone and settled surface normals", &shader);
    let mut numbers = gf_core::csg::uniforms(&fams, &parts, &kinds, 128, false, &[], &object);
    let original = numbers.clone();
    plan.tracer_uniforms(&mut numbers);
    let stride = gf_core::csg::PART_FLOATS;
    let start = 8 + 8 * stride;
    let end = 8 + 58 * stride;
    assert_eq!(&original[..start], &numbers[..start]);
    assert_eq!(&original[end..], &numbers[end..]);
    assert!(numbers[start..end].iter().all(|v| v.is_finite()));
    let i = 8 + (gf_core::csg::MAX_PARTS - 2) * stride;
    assert!((numbers[i+16] - 0.2).abs() < 1e-6);
    for (i, part) in parts.iter().enumerate() {
        let placed = 8 + stride * i;
        let prepared = 8 + stride * (gf_core::csg::MAX_PARTS - 1 - i);
        for p in [[-0.25,0.33,0.11,1.0], [0.52,-0.08,0.96,1.0]] {
            for (axis, bias) in [0.0,0.5*numbers[1]/numbers[2],-0.5*numbers[1]/numbers[2]].into_iter().enumerate() {
                let projection = |base| (0..4).map(|k| numbers[base+axis*4+k]*p[k]).sum::<f32>();
                let expected = (projection(placed)+bias) * part.place.size as f32;
                assert!((projection(prepared)-expected).abs()<1e-6);
            }
        }
    }
}
