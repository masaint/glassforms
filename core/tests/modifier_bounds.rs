//! Whatever a modifier is set to (large turns, large sizes, numbers that are not numbers), the volumes the
//! scene asks for stay within their caps: a bound blown up by a modifier must never turn into a gigantic
//! allocation.
use gf_core::{
    groups::Item,
    layers::{BOX_MAX, BOX_RES_FULL},
    mods::{Axes, Kind, Modifier},
    state::{valid_res, Detail, State, RES_MAX},
};

const KINDS: [Kind; 13] = [Kind::Mirror, Kind::Repeat, Kind::Revolve, Kind::Wrap, Kind::Shell, Kind::Offset, Kind::Smooth, Kind::Twist, Kind::Swirl, Kind::Bend, Kind::Taper, Kind::Noise, Kind::Ripple];

fn extremes(kind: Kind) -> Vec<Modifier> {
    let mut out = Vec::new();
    for v in [720.0, -720.0, 1e6, -1e6, 1e-12, f64::NAN, f64::INFINITY] {
        let mut m = Modifier::of(kind);
        m.a = Axes::new([v, v, v], false);
        m.b = Axes::new([v, v, v], false);
        m.size = v;
        m.amount = if v.is_finite() { v.clamp(-10.0, 10.0) } else { v };
        m.centre = [v.min(1e6).max(-1e6); 3];
        out.push(m);
    }
    out
}

fn check(st: &State, what: &str) {
    let k = st.stack.box_k();
    assert!(k.is_finite() && (1.0..=BOX_MAX).contains(&k), "{what}: box {k}");
    for mode in [Detail::Full, Detail::Quick, Detail::Anim, Detail::Hq] {
        let res = st.scene_res(mode);
        assert!(res <= RES_MAX && res >= 64, "{what}: scene res {res} ({mode:?})");
        // a grown stage is held to its cap; one that has not grown is held to the layers' own detail
        if k > 1.0 { assert!(f64::from(res) <= BOX_RES_FULL.max(f64::from(st.layer_res(0, mode))), "{what}: grown scene res {res}"); }
    }
    let plan = st.stack.world_plan();
    for field in &plan.fields {
        let (b, scale) = field.domain(gf_core::mods::whole_box());
        // the domain may be large, but what is made from it is not: the field's side is held below RES_MAX
        let side = valid_res(768.0 * if scale.is_finite() { scale } else { f64::MAX });
        assert!(side <= RES_MAX, "{what}: field side {side} from {b:?} x{scale}");
    }
}

#[test]
fn object_modifiers_at_extreme_values_keep_every_volume_within_its_cap() {
    for kind in KINDS {
        for m in extremes(kind) {
            let mut st = State::new();
            st.selected_modifiers_mut().list.push(m.clone());
            check(&st, &format!("object {kind:?} {:?}", m.a.get()));
            assert!(st.layer_res(0, Detail::Full) <= gf_core::sculpt::RES_MAX);
        }
    }
}

#[test]
fn world_modifiers_at_extreme_values_keep_every_volume_within_its_cap() {
    for kind in KINDS {
        for m in extremes(kind) {
            let mut st = State::new();
            st.create_world_object(st.fam, st.shape().preset, false, None).unwrap();
            st.stack.world_modifiers_mut(Item::Object(0)).unwrap().list.push(m.clone());
            check(&st, &format!("world {kind:?} {:?}", m.a.get()));
        }
    }
}

#[test]
fn detail_1536_with_several_objects_is_held_within_the_memory_budget_and_says_so() {
    use gf_core::budget::{SINGLE_MAX, TOTAL_MAX};
    let mut st = State::new();
    for _ in 0..2 { st.create_world_object(st.fam, st.shape().preset, false, None).unwrap(); }
    st.set_detail(1536);
    let res = st.scene_res(Detail::Full);
    let (single, total) = st.memory_scene().bytes(res);
    assert!(res < 1536 && single <= SINGLE_MAX && total <= TOTAL_MAX, "{res}: {single} {total}");
    assert!((0..3).all(|i| st.layer_res(i, Detail::Full) <= res));
    assert!(st.memory_note(Detail::Full).is_some_and(|n| n.contains("1536")));
    // one plain object at 1536 fits as it is, and nothing is said
    let mut one = State::new();
    one.set_detail(1536);
    assert_eq!(one.scene_res(Detail::Full), 1536);
    assert_eq!(one.memory_note(Detail::Full), None);
}

#[test]
fn a_twisted_object_at_1024_is_held_under_the_total_budget() {
    let mut st = State::new();
    st.set_detail(1024);
    st.selected_modifiers_mut().list.push(Modifier::of(Kind::Twist));
    let res = st.layer_res(0, Detail::Full);
    assert!(res < 1024, "{res}");
    assert!(st.memory_scene().bytes(res).1 <= gf_core::budget::TOTAL_MAX);
}

#[test]
fn a_world_twist_on_an_object_does_not_grow_the_stage_or_project_a_field() {
    // the owner's steps: in the world view a twist (or swirl, bend) on an object, cascade on. A turn about
    // the shape's centre keeps it within the ball it was fitted in; taken as the box round that ball, and
    // then that box's corners, it grew the stage to 3.25 and projected a field of 5.8 GB at 896³.
    use gf_core::world_mods::Scope;
    for kind in [Kind::Twist, Kind::Swirl, Kind::Bend] {
        for cascade in [false, true] {
            let args = [("scene", "attractors"), ("preset", "thomas")].map(|(k, v)| (k.to_string(), v.to_string()));
            let mut st = State::with_args(&args, true);
            let mut m = Modifier::of(kind);
            m.scope = Scope::SubObjects;
            m.cascade = cascade;
            st.stack.world_modifiers_mut(Item::Object(0)).unwrap().list.push(m);
            // (as the engine does before it works out the stage's size)
            st.fit_world_sources();
            assert_eq!(st.stack.box_k(), 1.0, "{kind:?} cascade {cascade}: need {}", st.stack.box_need());
            assert_eq!(st.memory_scene().world_fields, 0, "{kind:?} cascade {cascade}");
            assert_eq!(st.scene_res(Detail::Full), st.layer_res(0, Detail::Full));
        }
    }
}
