//! Setups for stills and timings of parts sized along their own axes (not a check): with GF_STRETCH_SETUPS
//! set to a folder, `cargo test -p gf-core --test stretch_stills -- --ignored` writes one setup per case
//! there, to be rendered with `--setup <file>`.
use gf_core::csg::{Op, Part, Strokes};
use gf_core::join::{Band, Profile};
use gf_core::layers::Placement;
use gf_core::state::State;
use std::collections::BTreeMap;

fn part(preset: &str, pos: [f64; 3], rot: [f64; 3], size: f64, stretch: [f64; 3]) -> Part {
    Part { family: "studio".into(), preset: preset.into(), place: Placement { pos, rot, size }, stretch: Some(stretch), ..Part::default() }
}

/// The shape up made a speck of a ball inside the subject, so that the subject is all that shows.
fn state(extra: &[(&str, &str)]) -> State {
    let mut args: Vec<(String, String)> = [("scene", "studio"), ("preset", "sphere"), ("trace", "csg"), ("p.radius", "0.02"), ("dist", "4.2"), ("rot", "24,-32,0")]
        .iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    args.extend(extra.iter().map(|(k, v)| (k.to_string(), v.to_string())));
    let mut s = State::with_args(&args, true);
    s.csg.edit_construction();
    s
}

fn values(v: &[(&str, f64)]) -> BTreeMap<String, f64> {
    v.iter().map(|(k, x)| (k.to_string(), *x)).collect()
}

/// A block stretched 2.2 : 0.6 : 1.1, its flats and rounds kept true (taken into its own sizes).
fn stretched_box(extra: &[(&str, &str)]) -> State {
    let mut s = state(extra);
    let mut b = part("block", [0.0; 3], [0.0, 18.0, 0.0], 0.75, [2.2, 0.6, 1.1]);
    b.values = values(&[("width", 1.0), ("height", 1.0), ("depth", 1.0), ("bevel", 0.04), ("round", 0.025)]);
    s.csg.add_construction_part(&[], b).unwrap();
    s
}

/// A whole ball stretched 1 : 0.55 : 1.6: the ellipsoid, read exactly.
fn stretched_ball(extra: &[(&str, &str)]) -> State {
    let mut s = state(extra);
    let mut b = part("sphere", [0.0; 3], [0.0, 25.0, 0.0], 0.85, [1.0, 0.55, 1.6]);
    b.values = values(&[("radius", 0.8)]);
    s.csg.add_construction_part(&[], b).unwrap();
    s
}

/// The ellipsoid hollowed by a shell and cut open by a block: its wall is even.
fn shelled_ball(extra: &[(&str, &str)]) -> State {
    let mut s = stretched_ball(extra);
    s.csg.parts[0].strokes = Strokes { shell: 0.08, ..Strokes::default() };
    let mut cut = Part { family: "studio".into(), preset: "block".into(), place: Placement { pos: [0.0, 0.0, 0.62], rot: [0.0; 3], size: 1.0 }, op: Op::Subtract, ..Part::default() };
    cut.values = values(&[("width", 2.0), ("height", 1.6), ("depth", 1.2), ("bevel", 0.0), ("round", 0.0)]);
    s.csg.add_construction_part(&[], cut).unwrap();
    s
}

/// A block with an arch stretched across it (it has no sizes of its own along its axes: read by its
/// estimate near the surface) joined by a seam, and a capsule stretched unevenly across its axis (its length
/// taken in, the rest left stretched) cut from it by a seam.
fn stretched_in_joins(extra: &[(&str, &str)]) -> State {
    let mut s = state(extra);
    let mut base = part("block", [0.0, -0.05, 0.0], [0.0; 3], 0.9, [1.0; 3]);
    base.stretch = None;
    base.values = values(&[("width", 1.4), ("height", 0.6), ("depth", 1.0), ("bevel", 0.0), ("round", 0.03)]);
    s.csg.add_construction_part(&[], base).unwrap();
    let mut arch = part("arch", [0.0, 0.2, 0.0], [0.0, 90.0, 0.0], 0.55, [1.0, 1.5, 2.6]);
    arch.smooth = 0.08;
    s.csg.add_construction_part(&[], arch).unwrap();
    let mut slot = part("capsule", [0.0, 0.12, 0.38], [0.0, 0.0, 0.0], 0.5, [1.8, 0.45, 1.0]);
    slot.op = Op::Subtract;
    slot.smooth = 0.05;
    s.csg.add_construction_part(&[], slot).unwrap();
    s
}

/// The joins of `stretched_in_joins` given seam profiles: the arch chamfered on, and a groove run along
/// the capsule's cut.
fn profiled_joins(extra: &[(&str, &str)]) -> State {
    let mut s = stretched_in_joins(extra);
    let tree = s.csg.edit_construction();
    tree.node_mut(&[1]).unwrap().profile = Profile::Chamfer;
    let slot = tree.node_mut(&[2]).unwrap();
    slot.band = Band::Groove;
    slot.depth = 0.05;
    slot.width = 0.03;
    s
}

/// A thick ring stretched unevenly across its axis (its height taken in, the rest left stretched: not
/// convex, read by its estimate near the surface).
fn stretched_ring(extra: &[(&str, &str)]) -> State {
    let mut s = state(extra);
    let mut r = part("washer", [0.0; 3], [0.0, 20.0, 0.0], 0.8, [2.0, 1.3, 0.75]);
    r.values = values(&[("outer", 1.0), ("hole", 0.55), ("height", 0.3), ("bevel", 0.03), ("round", 0.02)]);
    s.csg.add_construction_part(&[], r).unwrap();
    s
}

/// A ball drawn out along two of its axes by the elongate stroke (a rounded slab, exact), its seam with a
/// stretched block cut into it.
fn elongated(extra: &[(&str, &str)]) -> State {
    let mut s = state(extra);
    let mut b = part("sphere", [0.0; 3], [0.0, 15.0, 0.0], 0.8, [1.0; 3]);
    b.stretch = None;
    b.values = values(&[("radius", 0.35)]);
    b.strokes = Strokes { elongate: [0.55, 0.0, 0.3], ..Strokes::default() };
    s.csg.add_construction_part(&[], b).unwrap();
    let mut cut = part("block", [0.25, 0.3, 0.0], [0.0, 15.0, 0.0], 0.5, [1.6, 1.0, 0.5]);
    cut.values = values(&[("width", 0.5), ("height", 0.4), ("depth", 0.5), ("bevel", 0.0), ("round", 0.04)]);
    cut.op = Op::Subtract;
    cut.smooth = 0.04;
    s.csg.add_construction_part(&[], cut).unwrap();
    s
}

/// The ellipsoid of `stretched_ball` left a stretched ball (a flat base of a millionth: not taken into an
/// ellipsoid), read by the estimate: the same picture, by the other reading.
fn ball_by_estimate(extra: &[(&str, &str)]) -> State {
    let mut s = stretched_ball(extra);
    s.csg.parts[0].values.insert("cut".into(), 1e-6);
    s
}

/// The shape of `elongated` made without the stroke: a block whose round is half its height (for timing).
fn elongated_as_block(extra: &[(&str, &str)]) -> State {
    let mut s = elongated(extra);
    let b = &mut s.csg.parts[0];
    b.preset = "block".into();
    b.strokes = Strokes::default();
    b.values = values(&[("width", 1.8), ("height", 0.7), ("depth", 1.3), ("bevel", 0.0), ("round", 0.35)]);
    s
}

/// A case with every stretch taken away (the pictures of before).
fn unstretched(mut s: State) -> State {
    for p in &mut s.csg.parts { p.stretch = None; }
    s
}

#[test]
#[ignore]
fn write_stretch_setups() {
    let Ok(dir) = std::env::var("GF_STRETCH_SETUPS") else { return };
    std::fs::create_dir_all(&dir).unwrap();
    let clay: &[(&str, &str)] = &[("clay", "1")];
    let cases: Vec<(&str, State)> = vec![
        ("box_glass", stretched_box(&[])), ("box_clay", stretched_box(clay)),
        ("ball_glass", stretched_ball(&[])), ("ball_clay", stretched_ball(clay)),
        ("shell_glass", shelled_ball(&[])), ("shell_clay", shelled_ball(clay)),
        ("joins_glass", stretched_in_joins(&[])), ("joins_clay", stretched_in_joins(clay)),
        ("ring_glass", stretched_ring(&[])), ("ring_clay", stretched_ring(clay)),
        ("joins_plain", unstretched(stretched_in_joins(&[]))), ("shell_round_clay", unstretched(shelled_ball(clay))),
        ("elongate_glass", elongated(&[])), ("elongate_clay", elongated(clay)), ("elongate_ref", elongated_as_block(&[])), ("elongate_ref_clay", elongated_as_block(clay)),
        ("strokes_ref", { let mut s = elongated_as_block(&[]); s.csg.parts[0].strokes.mirror = [true, false, false]; s }),
        ("ball_est_glass", ball_by_estimate(&[])), ("ball_est_clay", ball_by_estimate(clay)),
        ("profiled_glass", profiled_joins(&[])), ("profiled_clay", profiled_joins(clay)), ("profiled_plain", unstretched(profiled_joins(&[]))),
        ("box_plain", unstretched(stretched_box(&[]))), ("ball_plain", unstretched(stretched_ball(&[]))), ("ring_plain", unstretched(stretched_ring(&[]))),
    ];
    for (name, s) in cases {
        std::fs::write(std::path::Path::new(&dir).join(format!("{name}.json")), s.setup_json()).unwrap();
    }
}
