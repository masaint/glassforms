//! Every size that depends on the number of layers follows `MAX_LAYERS` / `COMBINE_WIDTH`, and a saved
//! scene of up to four layers loads as it always did.

use gf_core::layers::{mask_of, members, parse_tag, tag, Combine, Layer, Mask, Op, Stack, StackSetup, COMBINE_WIDTH, MAX_LAYERS, MAX_SCENE_LAYERS};

#[test]
fn derived_sizes_follow_the_constants() {
    assert!(COMBINE_WIDTH <= MAX_LAYERS);
    assert_eq!(COMBINE_WIDTH, gf_core::material::MAX_BODIES);
    assert_eq!(COMBINE_WIDTH, gf_core::cast::MAX_BODIES);
    assert!(MAX_LAYERS <= MAX_SCENE_LAYERS && MAX_SCENE_LAYERS <= Mask::BITS as usize);
    let c: Combine = Stack::new().combine();
    assert_eq!(c.inv.len(), COMBINE_WIDTH);
    assert_eq!(c.morph.len(), COMBINE_WIDTH);
    assert_eq!(c.morph[0].len(), COMBINE_WIDTH);
    assert_eq!(c.body.len(), COMBINE_WIDTH);
}

#[test]
fn stack_stops_at_max_layers_and_keeps_bodies_narrow() {
    let mut s = Stack::new();
    for i in 1..MAX_LAYERS {
        assert_eq!(s.add(), Some(i));
        assert!(s.bodies().iter().all(|b| b.count_ones() as usize <= COMBINE_WIDTH));
    }
    assert_eq!(s.len(), MAX_LAYERS);
    assert_eq!(s.add(), None);
    // the first four join as they always did; the next four join each other, apart from the first
    let w = COMBINE_WIDTH;
    assert_eq!(s.bodies(), vec![mask_of(&(0..w).collect::<Vec<_>>()), mask_of(&(w..MAX_LAYERS).collect::<Vec<_>>())]);
    assert_eq!(s.combos().len(), MAX_LAYERS * (MAX_LAYERS - 1) / 2 + 2 * (4 + 1));
    let plan = s.pass_plan(&s);
    assert_eq!(plan.len(), 2);
    assert_eq!(plan[0].chan, [0, u32::MAX, u32::MAX, u32::MAX]);
    assert_eq!(plan[1].chan, [1, u32::MAX, u32::MAX, u32::MAX]);
}

#[test]
fn bodies_past_the_fourth_share_the_first_channel() {
    let mut s = stack_of(MAX_LAYERS);
    for a in 0..MAX_LAYERS { for b in a + 1..MAX_LAYERS { s.pair_mut(mask_of(&[a, b])).unwrap().op = Op::Apart; } }
    let plan = s.pass_plan(&s);
    assert_eq!(plan.iter().map(|p| p.layers.clone()).collect::<Vec<_>>(), vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
    assert_eq!(plan[0].chan, [0, 1, 2, 3]);
    assert_eq!(plan[1].chan, [0, 0, 0, 0]);
    assert_eq!(s.body_channels(), COMBINE_WIDTH);
    // every pass on the whole stack's stage
    s.layers[7].place.pos = [3.0, 0.0, 0.0];
    let plan = s.pass_plan(&s);
    assert!(plan.iter().all(|p| p.combine.box_k == s.box_now()) && s.box_now() > 1.0);
}

#[test]
fn five_to_eight_layers_combine_as_their_groups() {
    for n in 5..=MAX_LAYERS {
        let mut s = stack_of(n);
        // a body of the first three joined (one with a cut), the rest apart in pairs where they can be
        for a in 0..n {
            for b in a + 1..n {
                let p = s.pair_mut(mask_of(&[a, b])).unwrap();
                p.op = if b < 3 { if a == 0 && b == 2 { Op::ACutsB } else { Op::Join } } else if a >= 3 && b == a + 1 && a % 2 == 1 { Op::Join } else { Op::Apart };
                p.radius = 0.07;
            }
        }
        assert!(s.bodies().iter().all(|b| b.count_ones() as usize <= COMBINE_WIDTH));
        let d: Vec<f64> = (0..n).map(|i| ((i * 37 + n) % 11) as f64 * 0.02 - 0.09).collect();
        // the nearest of each body, each combined as a stack of its own layers
        let mut want = f64::MAX;
        for b in s.bodies() {
            let m = members(b);
            let own = s.pass_stack(&m);
            let mut v = [0.0; COMBINE_WIDTH];
            for (k, &i) in m.iter().enumerate() { v[k] = d[i]; }
            want = want.min(own.combine().eval(&v));
        }
        assert!((s.eval_passes(&d) - want).abs() < 1e-12, "{n} layers: {} vs {want}", s.eval_passes(&d));
    }
}

#[test]
fn setups_of_one_to_eight_layers_round_trip() {
    for n in 1..=MAX_LAYERS {
        let mut s = Stack::new();
        for _ in 1..n { s.add(); }
        for i in 0..n { s.layers[i].place.pos = [0.1 * i as f64, 0.0, 0.0]; }
        if n >= 3 { s.pair_mut(mask_of(&[0, 2])).unwrap().radius = 0.3; }
        let setup = s.to_setup();
        let json = serde_json::to_string(&setup).unwrap();
        let back = Stack::from_setup(&serde_json::from_str(&json).unwrap());
        assert_eq!(back.len(), n);
        assert_eq!(back.to_setup(), setup, "{n} layers");
        assert_eq!(back.bodies(), s.bodies());
        // one digit a layer in every tag, as always
        assert!(setup.pairs.keys().chain(setup.junctions.keys()).all(|t| !t.contains(',')));
    }
}

#[test]
fn tags_cover_every_layer_and_no_more() {
    let all: Vec<usize> = (0..MAX_LAYERS).collect();
    let m = mask_of(&all);
    assert_eq!(members(m), all);
    assert_eq!(parse_tag(&tag(m)), Some(m));
    assert_eq!(parse_tag(&format!("1,{}", MAX_SCENE_LAYERS + 1)), None);
    // up to nine layers a tag is one digit a layer, as always; past that, numbers with commas
    assert_eq!(tag(mask_of(&[0, 1, 2, 3])), "1234");
    assert_eq!(tag(mask_of(&[0, 8])), "19");
    assert_eq!(tag(mask_of(&[2, 11])), "3,12");
    for m in [mask_of(&[2, 11]), mask_of(&[0, 9, 15]), mask_of(&[14, 15])] {
        assert_eq!(parse_tag(&tag(m)), Some(m));
    }
    assert_eq!(parse_tag("21"), None);
    assert_eq!(parse_tag("12,3"), None);
}

#[test]
fn four_layer_setup_round_trips() {
    let json = r#"{"layers":[{"show":true},{"show":true,"place":{"pos":[0.5,0,0],"rot":[0,0,0],"size":0.5}},
        {"show":false},{"show":true}],"selected":2,"pairs":{"12":{"op":"apart"},"34":{"radius":0.2}},"junctions":{"123":{"fill":0.5}}}"#;
    let setup: StackSetup = serde_json::from_str(json).expect("an old four-layer setup parses");
    let s = Stack::from_setup(&setup);
    assert_eq!(s.len(), 4);
    assert_eq!(s.sel, 2);
    assert_eq!(s.to_setup(), Stack::from_setup(&s.to_setup()).to_setup());
}

fn stack_of(n: usize) -> Stack {
    let mut s = Stack::new();
    s.layers = vec![Layer::default(); n];
    s
}

#[test]
fn up_to_the_width_there_is_one_pass_of_every_layer() {
    for n in 1..=COMBINE_WIDTH {
        let mut s = stack_of(n);
        if n >= 2 { s.pair_mut(mask_of(&[0, 1])).unwrap().op = Op::Apart; }
        assert_eq!(s.passes(), vec![(0..n).collect::<Vec<_>>()]);
        let p = s.pass_stack(&(0..n).collect::<Vec<_>>());
        assert_eq!(p.combine(), s.combine(), "a single pass is the stack itself");
        assert_eq!(s.combos().len(), (1 << n) - n - 1);
        let d: Vec<f64> = (0..n).map(|i| 0.1 * i as f64 - 0.05).collect();
        let mut v = [0.0; COMBINE_WIDTH];
        v[..n].copy_from_slice(&d);
        assert_eq!(s.eval_passes(&d), s.combine().eval(&v));
    }
}

/// Sixteen layers: four bodies of four joined layers each. The scene is the nearest of the bodies, each
/// combined exactly as a stack of its own four layers is.
#[test]
fn sixteen_layers_combine_in_passes_of_the_width() {
    let n = MAX_SCENE_LAYERS;
    let mut s = stack_of(n);
    for a in 0..n {
        for b in a + 1..n {
            let same = a / COMBINE_WIDTH == b / COMBINE_WIDTH;
            let p = s.pair_mut(mask_of(&[a, b])).unwrap();
            p.op = if same { Op::Join } else { Op::Apart };
            p.radius = 0.1 + 0.01 * (a + b) as f64;
        }
    }
    let passes = s.passes();
    assert_eq!(passes.len(), n / COMBINE_WIDTH);
    assert!(passes.iter().all(|p| p.len() == COMBINE_WIDTH));
    let d: Vec<f64> = (0..n).map(|i| ((i * 7919) % 23) as f64 * 0.01 - 0.08).collect();
    let mut want = f64::MAX;
    for (g, p) in passes.iter().enumerate() {
        assert_eq!(*p, (g * COMBINE_WIDTH..(g + 1) * COMBINE_WIDTH).collect::<Vec<_>>());
        let mut own = stack_of(COMBINE_WIDTH);
        for a in 0..COMBINE_WIDTH {
            for b in a + 1..COMBINE_WIDTH {
                *own.pair_mut(mask_of(&[a, b])).unwrap() = s.pair(mask_of(&[p[a], p[b]]));
            }
        }
        let mut v = [0.0; COMBINE_WIDTH];
        for (k, &i) in p.iter().enumerate() { v[k] = d[i]; }
        want = want.min(own.combine().eval(&v));
    }
    assert_eq!(s.eval_passes(&d), want);
    // pairs everywhere, triples and quads only within a body: 120 + 4 * (4 + 1)
    assert_eq!(s.combos().len(), n * (n - 1) / 2 + 4 * 5);
}

#[test]
fn no_body_grows_past_the_width() {
    let mut s = stack_of(8);
    for a in 0..8 { for b in a + 1..8 { s.pair_mut(mask_of(&[a, b])).unwrap().op = Op::Apart; } }
    for b in 1..COMBINE_WIDTH { assert!(s.op_allowed(mask_of(&[0, b]), Op::Join)); s.pair_mut(mask_of(&[0, b])).unwrap().op = Op::Join; }
    assert!(!s.op_allowed(mask_of(&[0, 5]), Op::Join), "a fifth layer may not join a body of four");
    assert!(!s.op_allowed(mask_of(&[3, 7]), Op::Morph));
    assert!(s.op_allowed(mask_of(&[4, 5]), Op::Join), "another body may form beside it");
    assert!(s.op_allowed(mask_of(&[0, 5]), Op::Apart));
    // a hand-written setup that joins everything is narrowed when read
    let mut wide = stack_of(8);
    wide.keep_bodies_narrow();
    assert!(wide.bodies().iter().all(|b| b.count_ones() as usize <= COMBINE_WIDTH));
}

#[test]
fn keys_name_layers_past_nine() {
    use gf_core::state::State;
    assert_eq!(State::layer_prefix(11), "L12/");
}

#[test]
fn detail_is_as_built_past_the_width_and_fits_the_budget() {
    use gf_core::budget::TOTAL_MAX as VOLUME_BUDGET;
    use gf_core::state::{Detail, State};
    // up to the width: every layer as asked, as always
    let mut s = State::new();
    for _ in 1..COMBINE_WIDTH { s.add_layer(); }
    for i in 0..COMBINE_WIDTH { s.stack.layers[i].place.size = 0.3; }
    let plan = s.detail_plan(Detail::Full);
    assert!(!plan.lowered);
    assert!((0..COMBINE_WIDTH).all(|i| plan.layers[i] == s.requested_res(i, Detail::Full)));
    assert_eq!(plan.scene, s.requested_res(0, Detail::Full));
    // past it: every layer as it is built, at the detail asked (not by its placed size: many_objects.md 3b)
    let mut s = State::new();
    for _ in 1..MAX_LAYERS { s.add_layer(); }
    for i in 0..MAX_LAYERS { s.stack.layers[i].place.size = if i == 0 { 1.0 } else { 0.3 }; }
    let asked = s.requested_res(1, Detail::Full);
    let plan = s.detail_plan(Detail::Full);
    assert!(plan.layers.iter().all(|&r| r == asked), "{plan:?}");
    assert_eq!(plan.scene, asked);
    assert_eq!(s.layer_build_res(3, Detail::Full), asked);
    assert!(plan.bytes <= VOLUME_BUDGET && !plan.lowered);
    // each object with a tree of parts beside its shape: its tree's volume is counted too, and at the
    // finest detail the budget holds every volume (own and tree) within it, with room left
    let part = gf_core::csg::Part::default();
    for i in 0..MAX_LAYERS {
        s.select_layer(i);
        s.csg.parts.push(part.clone());
    }
    s.set_detail(1024);
    let plan = s.detail_plan(Detail::Full);
    let vox = |r: u32| 2 * u64::from(r).pow(3);
    assert!(plan.lowered && plan.layers.iter().all(|&r| r < 1024), "{plan:?}");
    assert_eq!(plan.bytes, plan.layers.iter().map(|&r| 2 * vox(r)).sum::<u64>() + vox(plan.scene), "own and tree volumes, and the combined one");
    assert!(plan.bytes as f64 <= VOLUME_BUDGET as f64 * (1.0 - gf_core::budget::HEADROOM), "{plan:?}");
    assert!(s.memory_note(Detail::Full).is_some_and(|n| n.contains("trees")));
    for i in 0..MAX_LAYERS {
        s.select_layer(i);
        s.csg.parts.clear();
    }
    // eight layers at full size and the finest detail would need far more: built coarser, within it
    for i in 0..MAX_LAYERS { s.stack.layers[i].place.size = 1.0; }
    s.set_detail(gf_core::state::RES_MAX);
    let plan = s.detail_plan(Detail::Full);
    assert!(plan.lowered && plan.bytes <= VOLUME_BUDGET, "{plan:?}");
    assert!(s.memory_note(Detail::Full).is_some());
    assert_eq!(s.layer_res(3, Detail::Full), plan.layers[3]);
    assert_eq!(s.scene_res(Detail::Full), plan.scene);
}

#[test]
fn combine_of_more_layers_than_the_width_is_its_first_pass_width() {
    let mut s = stack_of(MAX_LAYERS);
    for a in 0..MAX_LAYERS { for b in a + 1..MAX_LAYERS { s.pair_mut(mask_of(&[a, b])).unwrap().op = Op::Apart; } }
    let c = s.combine();
    assert_eq!((c.n, c.bodies), (COMBINE_WIDTH, COMBINE_WIDTH));
    assert_eq!(c.body, [0, 1, 2, 3]);
}

/// Eight objects kept apart (eight bodies): everything that walks the bodies stays within the four that
/// can have a material of their own.
#[test]
fn eight_bodies_walk_only_four_materials() {
    use gf_core::state::State;
    let mut s = State::new();
    for _ in 1..MAX_LAYERS { s.add_layer(); }
    for a in 0..MAX_LAYERS { for b in a + 1..MAX_LAYERS { s.set_pair_op(mask_of(&[a, b]), Op::Apart); } }
    assert_eq!(s.bodies().len(), MAX_LAYERS);
    assert!(s.body_set(1, "ior", 1.7));
    assert!(!s.body_set(5, "ior", 1.7));
    assert_eq!(s.medium_descriptors().len(), MAX_LAYERS);
    assert!(s.body_materials().is_some_and(|m| m.len() == COMBINE_WIDTH));
    assert!(s.apply_body_material_patch(5, Default::default()).is_err());
    let _ = s.memory_scene();
    let _ = s.detail_plan(gf_core::state::Detail::Full);
}
