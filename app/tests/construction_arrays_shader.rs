//! Listed arrays (an arc, a path, steps, a blend; `gf_core::array`) in a construction's stacks, nested in one
//! stack and through a group, woven into the tracer: it parses and validates, each listed stage brings its
//! anchors once, and a tree whose arrays are plain weaves none of it. Needs no GPU.

use gf_core::csg::Part;
use gf_core::mods::{Axes, Kind, Modifier};
use gf_core::state::State;
use glassforms::render::{tracer_source, March, Variant};

fn check(name: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name} does not parse:\n{}", e.emit_to_string(src)));
    let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
    v.validate(&module).unwrap_or_else(|e| panic!("{name} is not valid:\n{}", e.emit_to_string(src)));
}

fn woven(state: &State) -> String {
    let (parts, _) = state.csg_tree();
    let binding = gf_core::sdf::BINDING;
    let tree = gf_core::csg::wgsl_tree_s(&state.families, &gf_core::csg::kinds(&parts), binding, false, false, false);
    let tree = state.construction_program(0, &parts).unwrap().shader(tree, binding, false);
    let march = March { step_scale: 1.0, steps_out: 120, steps_in: 70, layers: 0 };
    glassforms::sdf::weave_tree(&tracer_source(&march, Variant::default(), false), &tree)
}

fn count(text: &str, anchor: &str) -> usize {
    text.matches(anchor).count()
}

#[test]
fn listed_arrays_nested_in_a_stack_and_a_group_weave_into_a_valid_tracer() {
    let settings = [("scene", "studio"), ("preset", "block"), ("trace", "csg")].map(|(k, v)| (k.to_string(), v.to_string()));
    let mut state = State::with_args(&settings, true);
    let a = state.csg.add_construction_part(&[], Part::default()).unwrap();
    let b = state.csg.add_construction_part(&[], Part::default()).unwrap();
    let mut grid = Modifier::of(Kind::Repeat);
    grid.a = Axes::new([3.0, 2.0, 1.0], false);
    // (copies that leave room: a stack with a fold that touches is baked)
    grid.b = Axes::new([4.0; 3], true);
    let mut ring = Modifier::of(Kind::Repeat);
    ring.mode = 1;
    ring.a = Axes::new([7.0, 1.0, 1.0], false);
    ring.detail = 0.5;
    // plain arrays: no listed text at all
    state.csg.edit_construction().node_mut(&a).unwrap().mods.list = vec![grid.clone(), ring.clone()];
    let plain = woven(&state);
    for (name, anchor) in gf_core::array::ANCHORS {
        assert_eq!(count(&plain, anchor), 0, "{name} in a tree of plain arrays");
    }
    check("plain arrays", &plain);
    // a grid then an arc in one stack (a ring of grids), and the pair grouped under copies along a path
    let mut arc = ring.clone();
    arc.array.span = 200.0;
    arc.array.start = -20.0;
    arc.array.step_turn = 12.0;
    state.csg.edit_construction().node_mut(&a).unwrap().mods.list = vec![grid, arc];
    let tree = state.csg.edit_construction();
    let group = tree.group(&[], &[a[0], b[0]], "Pair").unwrap();
    let mut along = Modifier::of(Kind::Repeat);
    along.mode = 2;
    along.a = Axes::new([5.0, 1.0, 1.0], false);
    along.array.blend = 0.05;
    along.array.step_grow = 0.05;
    tree.node_mut(&group).unwrap().mods.list = vec![along];
    let listed = woven(&state);
    for (name, anchor) in gf_core::array::ANCHORS {
        assert_eq!(count(&listed, anchor), 2, "{name}: once per listed stage");
    }
    check("listed arrays nested", &listed);
}

/// Setups for timing listed arrays at 4K (`--setup`): run with `GF_ARRAY_SETUPS=<folder>` and `--ignored`.
#[test]
#[ignore]
fn write_array_timing_setups() {
    let Ok(dir) = std::env::var("GF_ARRAY_SETUPS") else { return };
    let settings = [("scene", "studio"), ("preset", "block"), ("trace", "csg"), ("p.width", "0.5"), ("p.height", "0.5"), ("p.depth", "0.5")].map(|(k, v)| (k.to_string(), v.to_string()));
    let mut ring = Modifier::of(Kind::Repeat);
    ring.mode = 1;
    ring.a = Axes::new([16.0, 1.0, 1.0], false);
    ring.detail = 1.1;
    let mut arc = ring.clone();
    arc.a = Axes::new([64.0, 1.0, 1.0], false);
    arc.array.span = 300.0;
    arc.array.step_turn = 6.0;
    arc.array.step_move = [0.0, 0.004, 0.0];
    arc.array.blend = 0.03;
    let mut grid = Modifier::of(Kind::Repeat);
    grid.a = Axes::new([3.0, 3.0, 1.0], false);
    let mut listed_ring = ring.clone();
    listed_ring.array.blend = 0.02;
    let mut path = ring.clone();
    path.mode = 2;
    path.a = Axes::new([24.0, 1.0, 1.0], false);
    path.array.step_grow = 0.02;
    for (name, stages) in [("ring16_folded", vec![ring]), ("arc64_listed", vec![arc]), ("grid_ring16_nested", vec![grid, listed_ring]), ("path24_listed", vec![path])] {
        let mut state = State::with_args(&settings, true);
        state.csg.edit_construction().node_mut(&[0]).unwrap().mods.list = stages;
        std::fs::write(format!("{dir}/{name}.json"), state.setup_json()).unwrap();
    }
}