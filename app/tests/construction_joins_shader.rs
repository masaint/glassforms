//! A construction joined by seam profiles, bands and group modes woven into the tracer: it parses and
//! validates, and a tree that uses none of them is woven exactly as before. Needs no GPU.

use gf_core::construction::GroupMode;
use gf_core::csg::{Op, Part};
use gf_core::join::{Band, Profile};
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

#[test]
fn joins_by_profile_band_and_mode_weave_into_a_valid_tracer() {
    let settings = [("scene", "studio"), ("preset", "block"), ("trace", "csg")].map(|(k, v)| (k.to_string(), v.to_string()));
    let mut state = State::with_args(&settings, true);
    let a = state.csg.add_construction_part(&[], Part { op: Op::Union, smooth: 0.1, ..Part::default() }).unwrap();
    let b = state.csg.add_construction_part(&[], Part { op: Op::Subtract, ..Part::default() }).unwrap();
    let c = state.csg.add_construction_part(&[], Part::default()).unwrap();
    let plain = woven(&state);
    assert!(!plain.contains("cnJoinP"), "a tree without profiles weaves no profile text");
    check("plain construction", &plain);
    let tree = state.csg.edit_construction();
    tree.node_mut(&a).unwrap().profile = Profile::Round;
    let cut = tree.node_mut(&b).unwrap();
    cut.band = Band::Groove;
    cut.depth = 0.1;
    cut.width = 0.02;
    cut.profile = Profile::Chamfer;
    let group = tree.group(&[], &[b[0], c[0]], "Pair").unwrap();
    tree.node_mut(&group).unwrap().mode = GroupMode::Overridden;
    let joined = woven(&state);
    for (name, anchor) in gf_core::join::ANCHORS {
        assert_eq!(joined.matches(anchor).count(), 1, "anchor {name}");
    }
    check("construction joined by profile, band and override", &joined);
}
