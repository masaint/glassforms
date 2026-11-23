//! Undo and redo (gf_core::history): a change between two quiet moments is one entry; undo puts back the
//! state before it exactly, redo puts the change back; what moves by itself makes no entry; the camera
//! slots are kept in a setup.

use gf_core::history::History;
use gf_core::state::State;

fn args(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

#[test]
fn a_gesture_is_one_entry_and_undo_puts_it_back() {
    let mut s = State::with_args(&[], true);
    let mut h = History::default();
    h.settle(&mut s);
    let before = s.setup();
    // a drag: many values, one quiet moment at its end
    let k = s.knob_list().into_iter().find(|k| k.id == "ior").expect("a knob");
    for i in 0..50 { s.user_set(&k, 1.5 + i as f64 * 0.004, true); }
    assert!(h.settle(&mut s));
    assert_eq!(h.can_undo(), 1);
    let after = s.setup();
    assert!(h.undo(&mut s));
    assert_eq!(s.setup(), before);
    assert!(h.redo(&mut s));
    assert_eq!(s.setup(), after);
    // a new change after an undo forgets what could be redone
    h.undo(&mut s);
    s.p.exposure = 2.0;
    h.settle(&mut s);
    assert_eq!(h.can_redo(), 0);
}

#[test]
fn a_family_and_a_studio_are_undone() {
    let mut s = State::with_args(&[], true);
    let mut h = History::default();
    h.settle(&mut s);
    let (fam, preset) = (s.family().id, s.preset().id);
    s.set_scene(3);
    h.settle(&mut s);
    assert_ne!(s.family().id, fam);
    assert!(s.apply_arg("studio", "dusk"));
    h.settle(&mut s);
    assert!(s.p.backdrop > 0.5);
    h.undo(&mut s);
    assert!(s.p.backdrop < 0.5);
    h.undo(&mut s);
    assert_eq!((s.family().id, s.preset().id), (fam, preset));
}

#[test]
fn what_moves_by_itself_makes_no_entry() {
    let mut s = State::with_args(&args(&[("spin", "0.4")]), false);
    let mut h = History::default();
    h.settle(&mut s);
    s.set_playing(true);
    for _ in 0..30 { s.tick(0.05); }
    assert!(!h.settle(&mut s), "playing and spinning made an entry");
    // a change while it plays is one, and undoing it leaves the clock where it is
    s.p.exposure = 1.7;
    assert!(h.settle(&mut s));
    let clock = s.clock;
    h.undo(&mut s);
    assert_eq!((s.p.exposure, s.clock, s.playing), (1.0, clock, true));
}

#[test]
fn the_camera_slots_are_kept_in_a_setup() {
    let mut s = State::with_args(&[], true);
    s.p.dist = 9.0;
    s.p.azim = 30.0;
    s.store_view(3);
    s.p.dist = 4.0;
    s.p.azim = 0.0;
    assert!(s.recall_view(3));
    assert_eq!((s.p.dist, s.p.azim), (9.0, 30.0));
    assert!(!s.recall_view(4));
    let text = s.setup_json();
    let mut t = State::with_args(&[], true);
    t.apply_setup_json(&text).expect("read back");
    assert!(t.view_stored(3) && !t.view_stored(0));
    t.p.dist = 5.0;
    t.recall_view(3);
    assert_eq!(t.p.dist, 9.0);
    // storing a slot is not an entry, and an undo leaves the slots alone
    let mut h = History::default();
    h.settle(&mut t);
    t.store_view(0);
    assert!(!h.settle(&mut t));
}

#[test]
fn the_history_is_kept_beside_a_setup() {
    let mut s = State::with_args(&[], true);
    let mut h = History::default();
    h.settle(&mut s);
    let first = s.setup();
    s.p.exposure = 1.6;
    h.settle(&mut s);
    let second = s.setup();
    s.p.dist = 9.0;
    h.settle(&mut s);
    let (text, history) = (s.setup_json(), h.to_json());
    // opened again elsewhere, undo walks back through what was done before it was saved
    let mut t = State::with_args(&[], true);
    let mut g = History::default();
    g.settle(&mut t);
    t.apply_setup_json(&text).expect("a setup");
    g.adopt_json(&history, &mut t).expect("a history");
    assert_eq!(g.can_undo(), 2);
    g.undo(&mut t);
    assert_eq!(t.setup(), second);
    g.undo(&mut t);
    assert_eq!(t.setup(), first);
    assert!(g.adopt_json("not a history", &mut t).is_err());
}

fn two_objects() -> State {
    let mut s = State::with_args(&args(&[("layer", "add")]), true);
    s.select_world_item(gf_core::groups::Item::Object(0), false);
    s
}

#[test]
fn a_selection_is_an_entry_and_undo_puts_the_one_before_back() {
    use gf_core::groups::Item;
    let mut s = two_objects();
    let mut h = History::default();
    h.settle(&mut s);
    assert_eq!(s.world_selection(), [Item::Object(0)]);
    s.select_world_item(Item::Object(1), false);
    assert!(h.settle(&mut s), "a click that selects is an entry");
    // nothing selected or hovered since: no entry
    assert!(!h.settle(&mut s));
    assert!(h.undo(&mut s));
    assert_eq!((s.world_selection(), s.stack.sel), (vec![Item::Object(0)], 0));
    assert!(h.redo(&mut s));
    assert_eq!((s.world_selection(), s.stack.sel), (vec![Item::Object(1)], 1));
    // emptied on purpose, and put back
    s.clear_world_selection();
    assert!(h.settle(&mut s));
    h.undo(&mut s);
    assert_eq!(s.world_selection(), [Item::Object(1)]);
}

#[test]
fn a_drag_select_is_one_entry() {
    use gf_core::groups::Item;
    let mut s = two_objects();
    let mut h = History::default();
    h.settle(&mut s);
    // the selection grows and shrinks as a box is dragged across; one quiet moment at the end
    for i in 0..20 {
        s.select_world_item(Item::Object(i % 2), false);
        if i % 3 == 0 { s.select_world_item(Item::Object((i + 1) % 2), true); }
    }
    assert!(h.settle(&mut s));
    assert_eq!(h.can_undo(), 1);
    h.undo(&mut s);
    assert_eq!(s.world_selection(), [Item::Object(0)]);
}

#[test]
fn entering_an_object_is_no_entry_and_its_parts_selection_is_undone() {
    use gf_core::groups::Item;
    let mut s = two_objects();
    let mut h = History::default();
    h.settle(&mut s);
    assert!(s.enter_world_item(Item::Object(0)));
    assert!(!h.settle(&mut s), "navigation made an entry");
    let outer = s.csg.edit_construction().group(&[], &[0], "Housing").unwrap();
    h.settle(&mut s);
    let marked = s.edit_scope.marked().to_vec();
    s.select_construction_node(&outer, false);
    assert_ne!(s.edit_scope.marked(), marked.as_slice());
    assert!(h.settle(&mut s));
    h.undo(&mut s);
    assert_eq!(s.edit_scope.marked(), marked.as_slice());
    assert_eq!(s.edit_scope.object(), Some(0), "undo left the object");
    h.redo(&mut s);
    assert_eq!(s.edit_scope.marked(), [outer]);
}
