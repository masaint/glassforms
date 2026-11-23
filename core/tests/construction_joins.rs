//! Construction joins by seam profile (gf_core::join): what a node's record and the tree's text hold.
use gf_core::construction::GroupMode;
use gf_core::construction_eval::Program;
use gf_core::csg::{Op, Part, FORMULA_AT, PART_FLOATS};
use gf_core::join::{Band, Profile};
use gf_core::layers::Placement;
use gf_core::state::State;

/// A tree of three parts: a sphere, a second joined by a blend, a third cut out of them.
fn three() -> (State, Vec<Vec<usize>>) {
    let mut state = State::new();
    let mut paths = Vec::new();
    for (x, op, blend) in [(0.0, Op::Union, 0.0), (0.35, Op::Union, 0.12), (-0.2, Op::Subtract, 0.06)] {
        let part = Part { place: Placement { pos: [x, 0.1, 0.0], rot: [0.0; 3], size: 0.5 }, ..Part::default() };
        let path = state.csg.add_construction_part(&[], part).unwrap();
        let node = state.csg.edit_construction().node_mut(&path).unwrap();
        node.op = op;
        node.blend = blend;
        paths.push(path);
    }
    (state, paths)
}

fn program(state: &State) -> Program {
    state.construction_program(0, &state.csg_tree_layer(0).0).unwrap()
}

/// The node records the text joins by `cnJoinP`, as offsets of their k (profile, band, depth, width) into
/// the program's numbers.
fn profiled_records(text: &str) -> Vec<usize> {
    let first = FORMULA_AT * 4 / PART_FLOATS;
    text.match_indices("=cnJoinP(").map(|(at, _)| {
        let rest = &text[at..];
        let open = rest.find("].k)").expect("reads the record's k");
        let start = rest[..open].rfind('[').unwrap() + 1;
        let entry: usize = rest[start..open].parse().unwrap();
        (entry - first) * PART_FLOATS + 20
    }).collect()
}

#[test]
fn a_tree_of_fillets_keeps_its_text_and_numbers() {
    let (state, _) = three();
    let p = program(&state);
    let text = p.shader(String::new(), 0, false);
    assert!(!text.contains("cnJoinP") && !text.contains("fn csgPhi("), "a fillet tree reads the profile text");
    for (name, anchor) in gf_core::join::ANCHORS {
        assert!(!text.contains(anchor), "anchor {name} in a tree that does not use it");
    }
    // the default profile changes no number: every node's k is as it was (zero)
    let first = FORMULA_AT * 4 / PART_FLOATS;
    for (at, _) in text.match_indices("=cnJoin(d,cn") {
        let rest = &text[at..];
        let open = rest.find("].inv0)").unwrap();
        let start = rest[..open].rfind('[').unwrap() + 1;
        let entry: usize = rest[start..open].parse().unwrap();
        let k = (entry - first) * PART_FLOATS + 20;
        assert_eq!(&p.numbers[k..k + 4], &[0.0; 4]);
    }
}

#[test]
fn a_profiled_node_joins_by_its_profile_and_others_do_not() {
    let mut keys = Vec::new();
    for profile in Profile::ALL.into_iter().filter(|p| !p.is_default()) {
        let (mut state, paths) = three();
        let plain = program(&state);
        state.csg.edit_construction().node_mut(&paths[1]).unwrap().profile = profile;
        let p = program(&state);
        let text = p.shader(String::new(), 0, false);
        assert_eq!(text.matches("fn csgPhi(").count(), 1, "{profile:?}: the profile text once");
        for (name, anchor) in gf_core::join::ANCHORS {
            assert_eq!(text.matches(anchor).count(), 1, "{profile:?}: anchor {name}");
        }
        let records = profiled_records(&text);
        assert_eq!(records.len(), 1, "{profile:?}: only the node with a profile joins by it");
        assert_eq!(p.numbers[records[0]], profile.code());
        // the profile is a number: the rest of the record is unchanged
        let mut back = p.numbers.clone();
        back[records[0]] = 0.0;
        assert_eq!(back, plain.numbers);
        // a profiled tree is a shader of its own; every non-default profile is the same shader
        assert_ne!(p.key(""), plain.key(""));
        keys.push(p.key(""));
    }
    keys.dedup();
    assert_eq!(keys.len(), 1, "changing a profile only uploads a number");
}

#[test]
fn a_band_records_its_depth_and_width_and_shows_its_operation() {
    let (mut state, paths) = three();
    let node = state.csg.edit_construction().node_mut(&paths[2]).unwrap();
    node.band = Band::Groove;
    node.op = Op::Intersect;
    node.depth = 0.1;
    node.width = 0.03;
    let p = program(&state);
    let text = p.shader(String::new(), 0, false);
    let k = profiled_records(&text);
    assert_eq!(k.len(), 1);
    // a groove cuts: it records the subtraction whatever op the node held
    assert_eq!(&p.numbers[k[0] - 20..k[0] - 18], &[1.0, 0.06]);
    assert_eq!(&p.numbers[k[0]..k[0] + 4], &[0.0, 2.0, 0.1, 0.03]);
    // a ridge adds, and has no width; relative amounts are a share of the node's size (0.5)
    let node = state.csg.edit_construction().node_mut(&paths[2]).unwrap();
    node.band = Band::Ridge;
    node.blend_relative = true;
    let p = program(&state);
    let k = profiled_records(&p.shader(String::new(), 0, false));
    assert_eq!(&p.numbers[k[0] - 20..k[0] - 18], &[0.0, 0.03]);
    assert_eq!(&p.numbers[k[0]..k[0] + 4], &[0.0, 3.0, 0.05, 0.0]);
}

/// Tree text and numbers of a state's program, for comparing two trees.
fn reads(state: &State) -> (String, Vec<f32>) {
    let p = program(state);
    (p.shader(String::new(), 0, false), p.numbers)
}

/// Four parts, the middle two grouped under a turned, scaled group whose join differs from both children's.
fn grouped(mode: GroupMode) -> (State, Vec<usize>) {
    let (mut state, mut paths) = three();
    let extra = Part { place: Placement { pos: [0.1, -0.3, 0.2], rot: [0.0; 3], size: 0.4 }, ..Part::default() };
    paths.push(state.csg.add_construction_part(&[], extra).unwrap());
    let tree = state.csg.edit_construction();
    let group = tree.group(&[], &[paths[1][0], paths[2][0]], "Pair").unwrap();
    let g = tree.node_mut(&group).unwrap();
    g.mode = mode;
    g.place = Placement { pos: [0.05, 0.1, -0.1], rot: [10.0, 20.0, 30.0], size: 0.8 };
    g.op = Op::Intersect;
    g.blend = 0.05;
    g.profile = Profile::Chamfer;
    let first = tree.node_mut(&[group[0], 0]).unwrap();
    first.blend = 0.1;
    first.profile = Profile::Smooth;
    let second = tree.node_mut(&[group[0], 1]).unwrap();
    second.band = Band::Pocket;
    second.depth = 0.08;
    (state, group)
}

#[test]
fn passthrough_reads_as_its_children_ungrouped() {
    let (mut state, group) = grouped(GroupMode::Passthrough);
    let inlined = reads(&state);
    assert!(state.csg.edit_construction().ungroup(&group));
    assert_eq!(inlined, reads(&state), "a passthrough group is its children, each by its own join");
}

#[test]
fn overridden_reads_as_its_children_each_joined_by_the_group() {
    let (mut state, group) = grouped(GroupMode::Overridden);
    let inlined = reads(&state);
    let tree = state.csg.edit_construction();
    let join = tree.node(&group).unwrap().join();
    assert!(tree.ungroup(&group));
    for i in 0..2 { assert_eq!(tree.node(&[group[0] + i]).unwrap().join(), join, "ungrouping keeps the override"); }
    assert_eq!(inlined, reads(&state), "an overridden group is its children, each by the group's join");
}

#[test]
fn an_isolated_group_and_one_with_a_material_of_its_own_join_as_one() {
    let (state, _) = grouped(GroupMode::Isolated);
    let isolated = reads(&state);
    let (mut state, group) = grouped(GroupMode::Passthrough);
    assert_ne!(isolated, reads(&state));
    state.csg.edit_construction().node_mut(&group).unwrap().material = Some(Default::default());
    let (text, _) = reads(&state);
    assert!(text.contains("/*cache"), "a group with a material stays one shape: {text}");
}

#[test]
fn grouping_and_ungrouping_carry_the_profile() {
    let (mut state, paths) = three();
    let tree = state.csg.edit_construction();
    tree.node_mut(&paths[1]).unwrap().profile = Profile::Smooth;
    let group = tree.group(&[], &[paths[1][0], paths[2][0]], "Pair").unwrap();
    assert_eq!(tree.node(&group).unwrap().profile, Profile::Smooth);
    assert!(tree.ungroup(&group));
    assert_eq!(tree.node(&paths[1]).unwrap().profile, Profile::Smooth);
}

#[test]
fn a_setup_keeps_the_profile_and_leaves_it_out_when_unused() {
    let (mut state, paths) = three();
    let saved = state.setup_json();
    assert!(!saved.contains("\"profile\""), "a default profile is not written");
    state.csg.edit_construction().node_mut(&paths[1]).unwrap().profile = Profile::Smooth;
    let saved = state.setup_json();
    assert!(saved.contains("\"profile\"") && saved.contains("\"smooth\""));
    let mut restored = State::new();
    restored.apply_setup_json(&saved).unwrap();
    assert_eq!(restored.csg.construction_tree(), state.csg.construction_tree());
}
