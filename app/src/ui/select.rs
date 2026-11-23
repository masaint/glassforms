//! Choosing several things at once at a world level: by a box dragged over the picture, and by turning the
//! selection over (invert), keeping only what is shown (visible only), or turning it over among what is
//! shown (invert within visible).
//!
//! The rules are plain functions over the items of the level (objects and groups), so they can be tested
//! without a window and shared by keys, scripts and any later menu. Whether a change of selection is an
//! entry of the history is the history's business, not decided here.

use eframe::egui::{Pos2, Rect};
use gf_core::groups::Item;
use gf_core::state::State;

/// Everything not chosen, in the level's order.
pub fn invert(all: &[Item], chosen: &[Item]) -> Vec<Item> {
    all.iter().copied().filter(|i| !chosen.contains(i)).collect()
}

/// What is chosen and shown.
pub fn visible_only(chosen: &[Item], shown: impl Fn(Item) -> bool) -> Vec<Item> {
    chosen.iter().copied().filter(|i| shown(*i)).collect()
}

/// Everything shown that is not chosen.
pub fn invert_within_visible(all: &[Item], chosen: &[Item], shown: impl Fn(Item) -> bool) -> Vec<Item> {
    all.iter().copied().filter(|i| shown(*i) && !chosen.contains(i)).collect()
}

/// What a box over the picture takes: every item whose box on the screen it meets. `add`: added to what
/// is chosen (shift held) rather than taking its place.
pub fn in_box(boxes: &[(Item, Rect)], r: Rect, chosen: &[Item], add: bool) -> Vec<Item> {
    let mut out: Vec<Item> = if add { chosen.to_vec() } else { Vec::new() };
    for (item, b) in boxes {
        if b.intersects(r) && !out.contains(item) { out.push(*item); }
    }
    out
}

/// The items of the world level being edited, in order, with whether each is shown (a group is shown
/// while any object in it is).
pub fn level(st: &State) -> Vec<(Item, bool)> {
    let mut out: Vec<(Item, bool)> = Vec::new();
    for i in 0..st.object_count() {
        let Some(item) = st.world_pick(i) else { continue };
        let shown = st.stack.layers.get(i).is_none_or(|l| l.show);
        match out.iter_mut().find(|(it, _)| *it == item) {
            Some(row) => row.1 |= shown,
            None => out.push((item, shown)),
        }
    }
    out
}

/// Make the world selection exactly these items.
pub fn set(st: &mut State, items: &[Item]) {
    st.clear_world_selection();
    for item in items { st.select_world_item(*item, true); }
}

/// The four commands, at the world level only (inside an object they do nothing). Returns a few words
/// for the line under the tools.
pub fn command(st: &mut State, which: &str) -> Option<String> {
    if st.edit_scope.object().is_some() { return None; }
    let rows = level(st);
    let all: Vec<Item> = rows.iter().map(|r| r.0).collect();
    let shown = |i: Item| rows.iter().any(|r| r.0 == i && r.1);
    let chosen = st.world_selection();
    let next = match which {
        "invert" => invert(&all, &chosen),
        "visible" => visible_only(&chosen, shown),
        "invert-visible" => invert_within_visible(&all, &chosen, shown),
        _ => return None,
    };
    set(st, &next);
    Some(format!("{} chosen", next.len()))
}

/// The box between two points of a drag.
pub fn drag_box(a: Pos2, b: Pos2) -> Rect {
    Rect::from_two_pos(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::pos2;

    const A: Item = Item::Object(0);
    const B: Item = Item::Object(1);
    const C: Item = Item::Object(2);
    const G: Item = Item::Group(7);

    #[test]
    fn the_selection_turns_over() {
        assert_eq!(invert(&[A, B, C, G], &[B]), [A, C, G]);
        assert_eq!(invert(&[A, B], &[A, B]), []);
        let shown = |i: Item| i != C;
        assert_eq!(visible_only(&[A, C], shown), [A]);
        assert_eq!(invert_within_visible(&[A, B, C, G], &[A], shown), [B, G]);
        // inverting twice gives back what was chosen (in the level's order)
        assert_eq!(invert(&[A, B, C], &invert(&[A, B, C], &[C, A])), [A, C]);
    }

    #[test]
    fn a_box_takes_what_it_meets() {
        let boxes = [(A, Rect::from_min_max(pos2(0.0, 0.0), pos2(10.0, 10.0))), (B, Rect::from_min_max(pos2(50.0, 50.0), pos2(60.0, 60.0)))];
        let r = drag_box(pos2(15.0, 15.0), pos2(5.0, 5.0));
        assert_eq!(in_box(&boxes, r, &[C], false), [A]);
        assert_eq!(in_box(&boxes, r, &[C, A], true), [C, A], "added, and not twice");
        assert_eq!(in_box(&boxes, drag_box(pos2(20.0, 20.0), pos2(30.0, 30.0)), &[], false), []);
    }

    #[test]
    fn the_commands_act_on_the_world_level() {
        let mut st = State::new();
        let rows = level(&st);
        assert!(!rows.is_empty());
        let before = st.world_selection();
        assert!(command(&mut st, "invert").is_some());
        let after = st.world_selection();
        assert!(after.iter().all(|i| !before.contains(i)));
        assert!(command(&mut st, "nonsense").is_none());
    }
}
