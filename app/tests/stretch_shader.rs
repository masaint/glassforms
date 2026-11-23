//! Parts sized along their own axes (gf_core::stretch) woven into the tracer, the merge pass and the extent
//! pass: each parses and validates, in a construction and in a tree of parts of before, with parts taken
//! into their own sizes (a block), made an ellipsoid and an oval, and left stretched (an arch: its normal
//! read along its own axes). A tree with none is woven as before. Needs no GPU.

use gf_core::csg::{Op, Part};
use gf_core::layers::Placement;
use gf_core::state::State;
use glassforms::render::{tracer_source, March, Variant};

fn check(name: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name} does not parse:\n{}", e.emit_to_string(src)));
    let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
    v.validate(&module).unwrap_or_else(|e| panic!("{name} is not valid:\n{}", e.emit_to_string(src)));
}

fn part(preset: &str, x: f64, op: Op, stretch: Option<[f64; 3]>) -> Part {
    Part { family: "studio".into(), preset: preset.into(), place: Placement { pos: [x, 0.1, 0.0], rot: [10.0, 20.0, 0.0], size: 0.4 }, op, smooth: 0.05, stretch, ..Part::default() }
}

/// The tracer, the merge pass and the extent pass for the state's tree, as the engine puts them together.
fn texts(state: &State) -> [String; 3] {
    let (parts, _) = state.csg_tree();
    let fams = &state.families;
    let kinds = gf_core::csg::kinds(&parts);
    let (strokes, stretch) = (gf_core::csg::any_strokes(&parts), gf_core::csg::any_stretch(&parts));
    let program = state.construction_program(0, &parts);
    let plain = program.is_none() && gf_core::csg::plain(&parts);
    let b = gf_core::sdf::BINDING;
    let tree = gf_core::construction_eval::shader(program.as_ref(), gf_core::csg::wgsl_tree_x(fams, &kinds, b, false, plain, strokes, stretch), b, false);
    let march = March { step_scale: 1.0, steps_out: 120, steps_in: 70, layers: 0 };
    [
        glassforms::sdf::weave_tree(&tracer_source(&march, Variant::default(), false), &tree),
        glassforms::csg::merge_source(&gf_core::construction_eval::shader(program.as_ref(), gf_core::csg::wgsl_tree_x(fams, &kinds, 0, true, plain, strokes, stretch), 0, true)),
        glassforms::csg::extent_source(&gf_core::csg::wgsl_tree_x(fams, &kinds, 0, false, plain, strokes, stretch)),
    ]
}

#[test]
fn parts_sized_along_their_axes_weave_into_valid_passes() {
    let settings = [("scene", "studio"), ("preset", "block"), ("trace", "csg")].map(|(k, v)| (k.to_string(), v.to_string()));
    for (construction, plain) in [(false, true), (false, false), (true, false)] {
        let mut state = State::with_args(&settings, true);
        let add = |state: &mut State, p: Part| {
            if construction { state.csg.add_construction_part(&[], p).unwrap(); } else { state.csg.parts.push(p); }
        };
        let cut = if plain { Op::Union } else { Op::Subtract };
        add(&mut state, part("arch", 0.3, Op::Union, None));
        add(&mut state, part("block", -0.3, Op::Union, None));
        add(&mut state, part("sphere", 0.0, Op::Union, None));
        add(&mut state, part("cylinder", 0.0, cut, None));
        let before = texts(&state);
        const ROWS: &str = "P.inv1.xyz * (ne / dot(P.inv1.xyz, P.inv1.xyz))";
        assert!(before.iter().all(|t| !t.contains(ROWS)), "a tree with no stretch is woven as before");
        let [w, x, y, z] = [[1.0, 2.0, 0.6], [2.0, 0.5, 1.0], [1.0, 1.8, 0.7], [1.6, 1.0, 0.6]];
        for (p, s) in state.csg.parts.iter_mut().zip([w, x, y, z]) { p.stretch = Some(s); }
        let (parts, _) = state.csg_tree();
        let ids: Vec<&str> = parts.iter().map(|p| state.families[p.fam].presets[p.preset].id).collect();
        assert_eq!(ids, ["block", "arch", "block", "ellipsoid", "oval"]);
        // only the arch is left stretched; the others take it into their own sizes
        assert_eq!(parts.iter().map(|p| p.stretched()).collect::<Vec<_>>(), [false, true, false, false, false]);
        let after = texts(&state);
        for (name, t) in ["tracer", "merge pass", "extent pass"].iter().zip(&after) {
            assert_eq!(t.matches(ROWS).count(), 1, "{name}");
            check(&format!("{name}, construction {construction}, plain {plain}"), t);
        }
    }
}
