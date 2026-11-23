//! Setups for stills and timings of the construction joins (not a check): with GF_JOIN_SETUPS set to a
//! folder, `cargo test -p gf-core --test join_stills -- --ignored` writes one setup per case there, to be
//! rendered with `setup=<file>`.
use gf_core::construction::GroupMode;
use gf_core::csg::{Op, Part};
use gf_core::join::{Band, Profile};
use gf_core::layers::Placement;
use gf_core::state::State;

fn part(preset: &str, pos: [f64; 3], size: f64) -> Part {
    Part { family: "studio".into(), preset: preset.into(), place: Placement { pos, rot: [0.0; 3], size }, ..Part::default() }
}

/// A block with a ball joined on its top edge and a cylinder cut through a corner, every join by `profile`.
fn seams(profile: Profile) -> State {
    let args = [("scene", "studio"), ("preset", "block"), ("trace", "csg")].map(|(k, v)| (k.to_string(), v.to_string()));
    let mut s = State::with_args(&args, true);
    let ball = s.csg.add_construction_part(&[], part("sphere", [0.35, 0.45, 0.0], 0.45)).unwrap();
    let cut = s.csg.add_construction_part(&[], part("cylinder", [-0.45, 0.0, 0.45], 0.4)).unwrap();
    let tree = s.csg.edit_construction();
    let n = tree.node_mut(&ball).unwrap();
    n.blend = 0.15;
    n.profile = profile;
    let n = tree.node_mut(&cut).unwrap();
    n.op = Op::Subtract;
    n.blend = 0.1;
    n.profile = profile;
    s
}

/// The block's top face with four balls on it: pocket, groove, ridge, bead.
fn bands() -> State {
    let args = [("scene", "studio"), ("preset", "block"), ("trace", "csg")].map(|(k, v)| (k.to_string(), v.to_string()));
    let mut s = State::with_args(&args, true);
    for (i, band) in [Band::Pocket, Band::Groove, Band::Ridge, Band::Bead].into_iter().enumerate() {
        let x = if i % 2 == 0 { -0.3 } else { 0.3 };
        let z = if i < 2 { -0.3 } else { 0.3 };
        let p = s.csg.add_construction_part(&[], part("sphere", [x, 0.5, z], 0.35)).unwrap();
        let n = s.csg.edit_construction().node_mut(&p).unwrap();
        n.band = band;
        n.op = band.op().unwrap();
        n.depth = 0.08;
        n.width = 0.025;
        n.blend = 0.02;
        n.profile = Profile::Round;
    }
    s
}

/// Two cylinders grouped, the group overriding their joins with a chamfered cut.
fn overridden() -> State {
    let mut s = seams(Profile::Fillet);
    let a = s.csg.add_construction_part(&[], part("cylinder", [0.3, 0.0, -0.4], 0.3)).unwrap();
    let b = s.csg.add_construction_part(&[], part("cylinder", [-0.1, 0.3, -0.4], 0.3)).unwrap();
    let tree = s.csg.edit_construction();
    let g = tree.group(&[], &[a[0], b[0]], "Cutters").unwrap();
    let n = tree.node_mut(&g).unwrap();
    n.mode = GroupMode::Overridden;
    n.op = Op::Subtract;
    n.blend = 0.08;
    n.profile = Profile::Chamfer;
    s
}

#[test]
#[ignore]
fn write_join_setups() {
    let Ok(dir) = std::env::var("GF_JOIN_SETUPS") else { return };
    std::fs::create_dir_all(&dir).unwrap();
    let mut cases: Vec<(String, State)> = Profile::ALL.into_iter().map(|p| (format!("seam_{}", p.id()), seams(p))).collect();
    cases.push(("bands".into(), bands()));
    cases.push(("overridden".into(), overridden()));
    for (name, s) in cases {
        std::fs::write(std::path::Path::new(&dir).join(format!("{name}.json")), s.setup_json()).unwrap();
    }
}
