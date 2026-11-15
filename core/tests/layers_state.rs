//! Layers in the program's state: adding, selecting, removing and reordering them, their knobs, keys and
//! tracks, what has to be built again when something changes, arguments by name, and setups.

use gf_core::layers::{mask_of, Op, Opt, Profile};
use gf_core::state::{Detail, KnobRef, State};

fn args(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

fn fam(s: &State, id: &str) -> usize {
    s.families.iter().position(|f| f.id == id).unwrap()
}
fn preset(s: &State, f: &str, p: &str) -> usize {
    s.families[fam(s, f)].presets.iter().position(|x| x.id == p).unwrap()
}

/// Thomas, a gyroid and a trefoil.
fn three() -> State {
    let mut s = State::new();
    let (f, p) = (fam(&s, "surfaces"), preset(&s, "surfaces", "gyroid"));
    assert_eq!(s.add_layer_of(f, p), Some(1));
    let (f, p) = (fam(&s, "knots"), preset(&s, "knots", "trefoil"));
    assert_eq!(s.add_layer_of(f, p), Some(2));
    s
}

#[test]
fn one_layer_is_the_program_as_it_was() {
    let s = State::new();
    assert!(s.plain());
    assert_eq!((s.layer_count(), s.stack.sel), (1, 0));
    // the first layer's keys are what a single shape's always were
    let keys: Vec<String> = s.shape_knobs().iter().map(|k| k.key.clone()).collect();
    assert!(keys.contains(&"attractors/thomas/b".to_string()) && keys.contains(&"attractors/thomas/@travel".to_string()), "{keys:?}");
    assert!(s.tracks.contains_key("attractors/thomas/@travel"));
    // nothing about layers is written into the setup of a single plain shape
    assert!(!s.setup_json().contains("layers"));
    // its march is its family's own
    let f = s.family();
    assert_eq!(s.march(), (f.step_scale, f.steps_out, f.steps_in, f.layers));
}

#[test]
fn layers_are_added_selected_and_removed() {
    let mut s = three();
    assert!(!s.plain());
    assert_eq!((s.layer_count(), s.stack.sel), (3, 2));
    // what "the shape that is up" means is the selected layer's
    assert_eq!((s.family().id, s.preset().id), ("knots", "trefoil"));
    assert_eq!((s.layer_family(0).id, s.layer_preset(0).id), ("attractors", "thomas"));
    assert_eq!((s.layer_family(1).id, s.layer_preset(1).id), ("surfaces", "gyroid"));
    s.select_layer(1);
    assert_eq!((s.family().id, s.preset().id), ("surfaces", "gyroid"));
    // the picker acts on the selected layer and leaves the others alone
    let p = preset(&s, "surfaces", "schwarzp");
    s.choose(p);
    assert_eq!(s.layer_preset(1).id, "schwarzp");
    assert_eq!((s.layer_preset(0).id, s.layer_preset(2).id), ("thomas", "trefoil"));
    // changing another layer's family does not touch the scene's settings or view: those are the first layer's
    let (dist, r) = (s.p.dist, s.r);
    let a = fam(&s, "algebraic");
    s.set_scene(a);
    assert_eq!((s.layer_family(1).id, s.p.dist, s.r), ("algebraic", dist, r));
    assert_eq!(s.look_family().id, "attractors");
    // each layer keeps its shape in every family, as a single shape always did
    let su = fam(&s, "surfaces");
    s.set_scene(su);
    assert_eq!(s.preset().id, "schwarzp");
    // names for the window
    assert_eq!((s.layer_title(0), s.combo_title(0b011), s.combo_title(0b111)), ("Thomas".to_string(), "Thomas + Schwarz P".to_string(), "Thomas + Schwarz P + Trefoil".to_string()));
    assert_eq!((s.op_title(0b011, Op::BCutsA), s.op_title(0b011, Op::Join)), ("Schwarz P cuts Thomas".to_string(), "join".to_string()));
    assert_eq!((s.combo_summary(0b011), s.combo_summary(0b111)), ("join · 0.12".to_string(), "fill +0.00".to_string()));
    // the three are joined, so they are one body; the selected one kept apart from the others is its own
    assert_eq!(s.bodies(), [vec![0, 1, 2]]);
    s.set_pair_op(0b011, Op::Apart);
    s.set_pair_op(0b110, Op::Apart);
    assert_eq!((s.bodies(), s.body_of(0), s.body_of(1), s.body_of(2)), (vec![vec![0, 2], vec![1]], 0, 1, 0));
    assert_eq!((s.one_body(0b101), s.one_body(0b011), s.one_body(0b111)), (true, false, false));
    s.set_pair_op(0b011, Op::Join);
    s.set_pair_op(0b110, Op::Join);
    // a fourth, and no fifth
    assert_eq!(s.add_layer(), Some(3));
    assert_eq!(s.add_layer(), None);
    assert_eq!(s.stack.combos().len(), 11);
    // two layers with the same shape are told apart by their numbers
    let (f, p) = (fam(&s, "attractors"), preset(&s, "attractors", "thomas"));
    s.set_scene(f);
    s.choose(p);
    assert_eq!((s.layer_title(0), s.layer_title(3)), ("Thomas 1".to_string(), "Thomas 4".to_string()));
    // taking the second away: the later ones move up, and the selection stays on the layer it was on
    assert!(s.remove_layer(1));
    assert_eq!((s.layer_count(), s.stack.sel), (3, 2));
    assert_eq!((s.layer_preset(0).id, s.layer_preset(1).id, s.layer_preset(2).id), ("thomas", "trefoil", "thomas"));
    // the last layer cannot be taken away
    assert!(s.remove_layer(0) && s.remove_layer(0));
    assert!(!s.remove_layer(0));
    assert_eq!((s.layer_count(), s.family().id, s.preset().id), (1, "attractors", "thomas"));
}

#[test]
fn every_layer_placement_and_combination_has_its_knobs() {
    let mut s = three();
    // shape knobs: the first layer's keys as ever, the others' under their layer
    assert!(s.layer_knobs(0).iter().all(|k| k.key.starts_with("attractors/thomas/")));
    assert!(s.layer_knobs(1).iter().all(|k| k.key.starts_with("L2/surfaces/gyroid/")));
    assert!(s.layer_knobs(2).iter().all(|k| k.key.starts_with("L3/knots/trefoil/")));
    // a knob reaches its own layer whichever is selected
    let thick = s.layer_knobs(2).into_iter().find(|k| k.id == "@thick").unwrap();
    s.select_layer(0);
    s.user_set(&thick, 1.7, false);
    assert_eq!(s.layer_shape(2).unwrap().thick, 1.7);
    assert_eq!((s.layer_shape(0).unwrap().thick, s.get(&thick.r)), (1.0, 1.7));
    // on their own, the shape's references mean the selected layer's
    assert_eq!(s.get(&KnobRef::Thick), 1.0);
    s.select_layer(2);
    assert_eq!(s.get(&KnobRef::Thick), 1.7);
    // placements
    let keys: Vec<String> = s.place_knobs(1).iter().map(|k| k.key.clone()).collect();
    assert_eq!(keys, ["L2.x", "L2.y", "L2.z", "L2.rx", "L2.ry", "L2.rz", "L2.size"]);
    let k = &s.place_knobs(1)[6];
    s.user_set(k, 0.5, false);
    assert_eq!((s.stack.layers[1].place.size, s.get(&k.r), s.want), (0.5, 0.5, Some(Detail::Full)));
    // a pair's sliders follow its operation
    let ids = |s: &State, m: u8| s.combo_knobs(m).iter().map(|k| k.key.clone()).collect::<Vec<_>>();
    assert_eq!(ids(&s, 0b011), ["J12.radius"]);
    s.set_pair_op(0b011, Op::BCutsA);
    assert_eq!(ids(&s, 0b011), ["J12.radius", "J12.gap"]);
    s.set_pair_op(0b011, Op::Morph);
    assert_eq!(ids(&s, 0b011), ["J12.amount"]);
    s.set_pair_op(0b011, Op::Apart);
    assert!(ids(&s, 0b011).is_empty());
    // another pair is untouched by all that, and so is this pair's radius, kept for when it is joined again
    assert_eq!((s.stack.pair(0b101).op, s.stack.pair(0b011).radius), (Op::Join, 0.12));
    assert_eq!(ids(&s, 0b111), ["J123.fill", "J123.reach"]);
    // every one of them is in the list that tracks are looked up in
    let all: Vec<String> = s.knob_list().iter().map(|k| k.key.clone()).collect();
    for key in ["P.ior", "attractors/thomas/b", "L2/surfaces/gyroid/@puff", "L3/knots/trefoil/@thick", "L1.x", "L3.size", "J13.radius", "J23.radius", "J123.fill"] {
        assert!(all.iter().any(|k| k == key), "{key} missing from {all:?}");
    }
    assert!(!all.iter().any(|k| k == "J12.radius"));
    let mut unique = all.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), all.len());
}

#[test]
fn placements_and_options_animate_and_loop() {
    let mut s = three();
    s.p.spin = 0.0;
    s.tracks.clear();
    let radius = s.combo_knob(0b011, Opt::Radius);
    let turn = s.place_knobs(2)[4].clone();
    s.toggle_wave(&radius);
    s.toggle_wave(&turn);
    assert!(s.tracks["J12.radius"].on && s.tracks["L3.ry"].on);
    assert!(s.shape_moving());
    // what moves is named with its layer or its combination
    let plan = s.loop_plan();
    let labels: Vec<&str> = plan.items.iter().map(|i| i.label.as_str()).collect();
    assert_eq!(labels, ["Trefoil: turn y °", "Thomas + Gyroid: seam radius"]);
    // playing moves both, and asks for the volume to be made again
    s.set_playing(true);
    s.want = None;
    let (r0, t0) = (s.stack.pair(0b011).radius, s.stack.layers[2].place.rot[1]);
    s.tick(1.0);
    assert!(s.stack.pair(0b011).radius != r0 && s.stack.layers[2].place.rot[1] != t0);
    assert_eq!(s.want, Some(Detail::Anim));
    // after one whole loop everything is where it was
    s.p.lock = 1.0;
    let plan = s.loop_plan();
    s.clock = 0.7;
    s.apply_time();
    let at = (s.stack.pair(0b011).radius, s.stack.layers[2].place.rot[1]);
    s.clock = 0.7 + plan.l;
    s.apply_time();
    assert!((s.stack.pair(0b011).radius - at.0).abs() < 1e-9 && (s.stack.layers[2].place.rot[1] - at.1).abs() < 1e-7);
    // a video frame that only moves those still has to make the volume again
    s.set_playing(false);
    let keep = s.video_begin(0);
    assert!(s.video_frame(&keep, 3.0, 100));
    s.video_end(keep);
}

#[test]
fn only_what_changed_is_built_again() {
    let mut s = three();
    let keys = |s: &State| (0..3).map(|i| s.layer_key(i)).collect::<Vec<_>>();
    let before = keys(&s);
    // moving a layer, changing how layers combine, hiding one: no layer's own volume changes
    s.stack.set_place(1, 0, 0.3);
    s.set_pair_op(0b011, Op::Overlap);
    s.stack.set_opt(0b011, Opt::Radius, 0.3);
    s.stack.set_opt(0b111, Opt::Fill, 0.5);
    s.set_layer_show(2, false);
    s.select_layer(0);
    assert_eq!(keys(&s), before);
    // one layer's own slider: that layer's only
    let puff = s.layer_knobs(1).into_iter().find(|k| k.id == "@puff").unwrap();
    s.user_set(&puff, 0.02, false);
    let after = keys(&s);
    assert_eq!((&after[0], &after[2]), (&before[0], &before[2]));
    assert_ne!(after[1], before[1]);
    // the clock moves a layer only if it has travelling beads
    s.clock = 3.0;
    assert_eq!(keys(&s), after);
    s.layer_shape_mut(2).unwrap().beads = 0.3;
    let with_beads = keys(&s);
    s.clock = 3.5;
    assert_ne!(keys(&s)[2], with_beads[2]);
    assert_eq!(keys(&s)[0], with_beads[0]);
    // each layer's job is its own shape's
    use gf_core::state::ShapeJob;
    let jobs: Vec<bool> = (0..3).map(|i| matches!(s.layer_job(i, Detail::Quick, Some(64)).unwrap().shape, ShapeJob::Curve { .. })).collect();
    assert_eq!(jobs, [true, false, true]);
    assert_eq!(s.stack.sel, 0);
    // the combined volume is as fine as the finest layer
    s.layer_shape_mut(1).unwrap().res = 512;
    s.layer_shape_mut(2).unwrap().res = 384;
    s.layer_shape_mut(0).unwrap().res = 256;
    assert_eq!((s.layer_res(0, Detail::Full), s.layer_res(1, Detail::Full), s.scene_res(Detail::Full), s.scene_res(Detail::Anim)), (256, 512, 512, 256));
}

#[test]
fn the_march_is_shortened_only_where_the_result_can_be_steeper() {
    let mut s = three();
    // every pair the same seam: nothing steeper than the layers, so the families' own most careful stride
    assert_eq!(s.march(), (0.9, 220, 56, 2));
    // one pair's radius different: one layer now meets a seam shared between two radii
    s.stack.set_opt(0b011, Opt::Radius, 0.4);
    let bound = s.stack.combine().slope_bound();
    assert!(bound > 1.0 && bound < 1.2, "{bound}");
    let (step, out, inn, _) = s.march();
    assert!(step < 0.9 && step * bound <= 0.9 + 1e-9, "{step} for a bound of {bound}");
    assert!(out > 220 && inn > 56);
    // a junction's fill adds its own
    s.stack.set_opt(0b111, Opt::Fill, 1.0);
    assert!(s.stack.combine().slope_bound() > bound + 0.25);
}

#[test]
fn arguments_by_name() {
    let s = State::with_args(
        &args(&[
            ("scene", "knots"), ("preset", "trefoil"), ("thick", "1.2"),
            ("layer", "surfaces:gyroid"), ("L2.size", "0.6"), ("L2.x", "0.2"), ("L2.rz", "30"), ("L2.puff", "0.01"), ("L2.p.level", "0.3"),
            ("layer", "algebraic:tanglecube"), ("L3.show", "0"), ("L3.preset", "heart"),
            ("J12.radius", "0.2"), ("J12.profile", "round"), ("J13.op", "3cuts1"), ("J13.gap", "0.05"), ("J23.op", "morph"), ("J23.amount", "0.25"),
            ("J123.fill", "0.5"), ("J123.reach", "0.3"), ("track", "J12.radius:0.05:0.3:6"), ("select", "2"),
        ]),
        true,
    );
    assert_eq!((s.layer_count(), s.stack.sel), (3, 1));
    assert_eq!((s.layer_preset(0).id, s.layer_preset(1).id, s.layer_preset(2).id), ("trefoil", "gyroid", "heart"));
    // arguments without a layer are the first layer's, as they always were
    assert_eq!((s.layer_shape(0).unwrap().thick, s.layer_shape(1).unwrap().thick), (1.2, 1.0));
    let l2 = s.stack.layers[1];
    assert_eq!((l2.place.size, l2.place.pos, l2.place.rot), (0.6, [0.2, 0.0, 0.0], [0.0, 0.0, 30.0]));
    assert_eq!(s.layer_shape(1).unwrap().puff, 0.01);
    let level = s.layer_preset(1).param_index("level").unwrap();
    assert_eq!(s.layer_shape(1).unwrap().vals[level], 0.3);
    assert!(!s.stack.layers[2].show);
    let p = s.stack.pair(0b011);
    assert_eq!((p.op, p.radius, p.profile), (Op::Join, 0.2, Profile::Round));
    let p = s.stack.pair(0b101);
    assert_eq!((p.op, p.gap), (Op::BCutsA, 0.05));
    assert_eq!((s.stack.pair(0b110).op, s.stack.pair(0b110).amount), (Op::Morph, 0.25));
    assert_eq!((s.stack.junction(0b111).fill, s.stack.junction(0b111).reach), (0.5, 0.3));
    let t = &s.tracks["J12.radius"];
    assert!(t.on && (t.from, t.to, t.period) == (0.05, 0.3, 6.0));
    // what is not there is refused
    let mut s = s;
    for (k, v) in [("L4.x", "1"), ("J14.radius", "0.1"), ("J12.op", "3cuts1"), ("J12.fill", "1"), ("J123.radius", "0.1"), ("layer", "nosuch:thing"), ("L2.x", "abc"), ("select", "7"), ("track", "J12.nothing:0:1")] {
        assert!(!s.apply_arg(k, v), "{k}={v}");
    }
    // the names of the operations
    for (v, op) in [("join", Op::Join), ("1cuts2", Op::ACutsB), ("2cuts1", Op::BCutsA), ("a_cuts_b", Op::ACutsB), ("overlap", Op::Overlap), ("morph", Op::Morph), ("apart", Op::Apart)] {
        assert!(s.apply_arg("J12.op", v));
        assert_eq!(s.stack.pair(0b011).op, op);
    }
}

#[test]
fn a_setup_holds_every_layer_and_every_combination() {
    let mut s = three();
    s.stack.layers[1].place.size = 0.55;
    s.stack.layers[1].place.rot = [10.0, 20.0, 30.0];
    s.stack.layers[2].place.pos = [0.2, -0.1, 0.0];
    s.set_layer_show(2, false);
    s.set_pair_op(0b011, Op::BCutsA);
    s.stack.set_opt(0b011, Opt::Gap, 0.03);
    s.set_pair_profile(0b101, Profile::Bevel);
    s.stack.set_opt(0b110, Opt::Radius, 0.31);
    s.stack.set_opt(0b111, Opt::Fill, -0.4);
    s.layer_shape_mut(1).unwrap().puff = 0.015;
    let k = s.combo_knob(0b110, Opt::Radius);
    s.toggle_wave(&k);
    s.select_layer(1);
    let text = s.setup_json();
    let mut t = State::new();
    t.apply_setup_json(&text).unwrap();
    assert_eq!(t.setup(), s.setup());
    assert_eq!((t.layer_count(), t.stack.sel), (3, 1));
    assert_eq!(t.stack, s.stack);
    assert_eq!((t.layer_preset(0).id, t.layer_preset(1).id, t.layer_preset(2).id), ("thomas", "gyroid", "trefoil"));
    assert_eq!(t.layer_shape(1).unwrap().puff, 0.015);
    assert!(t.tracks["J23.radius"].on);
    for i in 0..3 {
        assert_eq!(t.layer_key(i), s.layer_key(i));
    }
    // loading it over a state that already has layers gives the same again, not more layers
    t.apply_setup_json(&text).unwrap();
    assert_eq!((t.setup(), t.layer_count()), (s.setup(), 3));
    // a setup naming a shape that does not exist is refused and changes nothing
    let before = t.setup();
    assert!(t.apply_setup_json(&text.replace("\"gyroid\"", "\"nosuchshape\"")).is_err());
    assert_eq!(t.setup(), before);
    // a setup saved before there were layers loads as one layer
    let old = State::with_args(&args(&[("scene", "surfaces"), ("preset", "schwarzp"), ("puff", "0.02")]), false).setup_json();
    assert!(!old.contains("layers"));
    t.apply_setup_json(&old).unwrap();
    assert!(t.plain());
    assert_eq!((t.layer_count(), t.family().id, t.preset().id, t.shape().puff), (1, "surfaces", "schwarzp", 0.02));
    // one layer that has been moved is not plain, and its setup says where it is
    t.stack.layers[0].place.pos[0] = 0.25;
    let moved = t.setup_json();
    let mut u = State::new();
    u.apply_setup_json(&moved).unwrap();
    assert_eq!((u.layer_count(), u.stack.layers[0].place.pos[0], u.plain()), (1, 0.25, false));
}

#[test]
fn layers_can_change_places_without_changing_the_scene() {
    let mut s = three();
    s.set_pair_op(0b011, Op::ACutsB);
    s.stack.set_opt(0b011, Opt::Gap, 0.04);
    s.set_pair_op(0b110, Op::Morph);
    s.stack.set_opt(0b110, Opt::Amount, 0.2);
    s.stack.set_opt(0b111, Opt::Fill, 0.7);
    s.stack.layers[2].place.size = 0.7;
    s.tracks.clear();
    for k in [s.combo_knob(0b110, Opt::Amount), s.place_knobs(2)[6].clone(), s.layer_knobs(1).into_iter().find(|k| k.id == "@puff").unwrap(), s.layer_knobs(0).into_iter().find(|k| k.id == "b").unwrap()] {
        s.toggle_wave(&k);
    }
    let (amount_from, amount_to) = (s.tracks["J23.amount"].from, s.tracks["J23.amount"].to);
    let d = [0.03, -0.02, 0.05, 0.0];
    let before = s.stack.combine().eval(&d);
    let keys: Vec<String> = (0..3).map(|i| s.layer_key(i)).collect();
    // the gyroid (second) and the trefoil (third) change places
    s.select_layer(2);
    s.swap_layers(1, 2);
    assert_eq!((s.layer_preset(1).id, s.layer_preset(2).id, s.stack.sel), ("trefoil", "gyroid", 1));
    assert_eq!((s.layer_key(1), s.layer_key(2)), (keys[2].clone(), keys[1].clone()));
    // the same solid: the distances go with their layers
    let after = s.stack.combine().eval(&[d[0], d[2], d[1], 0.0]);
    assert!((before - after).abs() < 1e-12, "{before} against {after}");
    // the cut still has Thomas cutting the gyroid, now its third layer
    assert_eq!((s.stack.pair(mask_of(&[0, 2])).op, s.stack.pair(mask_of(&[0, 2])).gap), (Op::ACutsB, 0.04));
    // the morph counts from the earlier layer, which is now the other one
    assert!((s.stack.pair(0b110).amount - 0.8).abs() < 1e-12);
    // tracks have gone with their layers
    let have: Vec<&String> = s.tracks.keys().collect();
    assert_eq!(have, ["J23.amount", "L2.size", "L3/surfaces/gyroid/@puff", "attractors/thomas/b"]);
    assert!((s.tracks["J23.amount"].from - (1.0 - amount_from)).abs() < 1e-12 && (s.tracks["J23.amount"].to - (1.0 - amount_to)).abs() < 1e-12);
    // taking the first layer away: the second becomes the first, and its keys the first layer's
    assert!(s.remove_layer(0));
    let have: Vec<&String> = s.tracks.keys().collect();
    assert_eq!(have, ["J12.amount", "L1.size", "L2/surfaces/gyroid/@puff"]);
    assert_eq!((s.look_family().id, s.layer_count()), ("knots", 2));
}

#[test]
fn a_family_can_have_its_own_detail_in_motion() {
    // cut stones are built at 384 voxels a side while they animate; every other family at 256, as ever
    let s = State::new();
    for f in &s.families {
        assert_eq!(State::family_anim_res(f), if matches!(f.id, "gems" | "studio") { 384 } else { 256 }, "{}", f.id);
    }
    assert_eq!((s.shape().anim_res, s.layer_res(0, Detail::Anim)), (256, 256));
    let g = State::with_args(&args(&[("scene", "gems"), ("preset", "brilliant")]), true);
    assert_eq!((g.shape().anim_res, g.layer_res(0, Detail::Anim), g.layer_res(0, Detail::Full)), (384, 384, 768));
    // the entry is the family's, not one of the engine's settings: nothing else about the scene differs
    assert_eq!(g.p.get("animRes"), None);
    // an explicit argument wins, and so does a setup
    let h = State::with_args(&args(&[("scene", "gems"), ("animRes", "256")]), true);
    assert_eq!(h.shape().anim_res, 256);
    let mut k = State::new();
    k.apply_setup_json(&h.setup_json()).unwrap();
    assert_eq!((k.family().id, k.shape().anim_res), ("gems", 256));
    k.apply_setup_json(&g.setup_json()).unwrap();
    assert_eq!(k.shape().anim_res, 384);
    // a stone added to a scene of another family keeps its own detail in motion, and the combined volume is as fine
    let mut s = State::new();
    let (f, p) = (fam(&s, "gems"), preset(&s, "gems", "brilliant"));
    s.add_layer_of(f, p);
    assert_eq!((s.layer_res(0, Detail::Anim), s.layer_res(1, Detail::Anim), s.scene_res(Detail::Anim)), (256, 384, 384));
    // unless the first layer's has been chosen by hand: then the new layer takes that
    let mut s = State::new();
    s.set_anim_detail(128);
    s.add_layer_of(f, p);
    assert_eq!((s.layer_res(1, Detail::Anim), s.scene_res(Detail::Anim)), (128, 128));
}

#[test]
fn glass_of_another_layer_is_followed_even_among_single_stones() {
    // a cut stone is one convex piece: a ray that has left it meets no more of it, and its family says so
    let one = State::with_args(&args(&[("scene", "gems"), ("preset", "brilliant")]), true);
    assert_eq!(one.march().3, 0);
    // one stone moved is still one stone
    let moved = State::with_args(&args(&[("scene", "gems"), ("L1.x", "0.3")]), true);
    assert_eq!((moved.plain(), moved.march().3), (false, 0));
    // two stones: a ray leaving one may meet the other
    let two = State::with_args(&args(&[("scene", "gems"), ("layer", "gems:princess"), ("J12.op", "apart"), ("L2.z", "-0.6")]), true);
    assert_eq!(two.march().3, 1);
    // three or more: two further pieces
    let three = State::with_args(&args(&[("scene", "gems"), ("layer", "gems:princess"), ("layer", "gems:emerald")]), true);
    assert_eq!(three.march().3, 2);
    // families that already follow further pieces keep their number
    assert_eq!(State::with_args(&args(&[("layer", "surfaces:gyroid")]), true).march().3, 2);
}
