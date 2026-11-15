//! Several shapes in one space. At the head of the right panel: the layers, one row each (its number, its
//! name, an eye to hide it, a cross to take it away), with rows that are combined into one body sharing
//! an outline, the body's first row carrying a dot of what the body is made of; and under them the joins, one row for every pair, triple and the four together, each
//! opening to its own options. Further down the panel, among the selected layer's own controls: where
//! that layer is placed.
//!
//! With one layer there is one row and a "+", and nothing else differs from a single shape.

use eframe::egui::{self, Align2, Color32, CursorIcon, Id, Pos2, Rect, Sense, Stroke, StrokeKind, Ui};
use gf_core::layers::{mask_of, members, tag, JunctionOpts, Op, PairOpts, Profile, MAX_LAYERS};
use gf_core::state::State;

use super::theme::*;
use super::widgets::*;
use super::{knob, material, Panels};

/// Height of a layer's row, the gap between rows of different bodies, and the width kept at the end of
/// the rows for the "+".
const ROW: f32 = 24.0;
const BODY_GAP: f32 = 5.0;
const PLUS: f32 = 24.0;

/// An eye, open or shut: whether a layer shows.
fn eye(ui: &Ui, rect: Rect, open: bool, ink: Color32) {
    let (c, p) = (rect.center(), ui.painter());
    let lid = |sign: f32| -> Vec<Pos2> { (0..=10).map(|i| { let u = i as f32 / 10.0; Pos2::new(c.x - 5.5 + 11.0 * u, c.y - sign * 3.4 * (u * std::f32::consts::PI).sin()) }).collect() };
    p.add(egui::Shape::line(lid(1.0), Stroke::new(1.2, ink)));
    if open {
        p.add(egui::Shape::line(lid(-1.0), Stroke::new(1.2, ink)));
        p.circle_filled(c, 1.7, ink);
    } else {
        // shut: the upper lid only, turned down, with lashes
        for dx in [-3.0, 0.0, 3.0] { p.line_segment([Pos2::new(c.x + dx, c.y + 1.0), Pos2::new(c.x + dx * 1.25, c.y + 3.2)], Stroke::new(1.0, ink)); }
    }
}

/// What a body is, in the words of a hover note: which layers it is made of, or that it stands alone.
fn body_words(st: &State, body: usize, bodies: &[Vec<usize>]) -> String {
    let names: Vec<String> = bodies[body].iter().map(|i| st.layer_title(*i)).collect();
    if names.len() > 1 {
        format!("body {}: {}, combined", body + 1, names.join(" + "))
    } else {
        format!("body {}: {}, stands alone: where it coincides with the others nothing is cut", body + 1, names.join(""))
    }
}

/// The layers: a row for each, and a "+" while there is room for another. A press on a row selects its
/// layer (everything below then shows and changes that layer's shape); dragging a row onto another
/// exchanges the two.
pub fn strip(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let n = st.layer_count();
    let bodies = st.bodies();
    let body: Vec<usize> = (0..n).map(|i| st.body_of(i)).collect();
    let width = ui.available_width();
    // where each row goes: rows of one body touch, rows of different bodies stand a little apart
    let mut tops = Vec::with_capacity(n);
    let mut y = 0.0;
    for i in 0..n {
        if i > 0 { y += if body[i] == body[i - 1] { 0.0 } else { BODY_GAP }; }
        tops.push(y);
        y += ROW;
    }
    let space = row(ui, y);
    let place = |i: usize| Rect::from_min_size(Pos2::new(space.left(), space.top() + tops[i]), egui::vec2(width - PLUS - 6.0, ROW));

    // what the pointer asks for, carried out once every row is drawn
    let (mut select, mut toggle, mut remove, mut exchange, mut lab) = (None, None, None, None, None);
    let held = Id::unique("layer row held");
    let dragging: Option<usize> = ui.data(|d| d.get_temp(held));
    let pointer = ui.input(|i| i.pointer.interact_pos());
    for i in 0..n {
        let r = place(i);
        px.note(format!("layer:{}", i + 1), r);
        let (chosen, shown) = (i == st.stack.sel, st.stack.layers[i].show);
        let resp = ui.interact(r, Id::unique(("layer row", i)), Sense::click_and_drag());
        let live = ui.is_enabled();
        let hot = live && resp.hovered();
        let aimed = dragging.is_some_and(|d| d != i) && pointer.is_some_and(|p| r.contains(p));
        // the row: brighter when it is the selected one; lined when another is about to be dropped on it
        let fill = if chosen { white(0.14) } else if hot { white(0.08) } else { white(0.04) };
        ui.painter().rect_filled(r.shrink(1.0), 4, fill);
        if aimed { ui.painter().rect_stroke(r.shrink(1.0), 4, Stroke::new(1.0, ACCENT), StrokeKind::Inside); }
        if chosen { ui.painter().rect_filled(Rect::from_min_max(Pos2::new(r.left() + 1.0, r.top() + 5.0), Pos2::new(r.left() + 3.0, r.bottom() - 5.0)), 1, ACCENT); }
        let ink = if !shown { HINT } else if chosen { BRIGHT } else { BUTTON };
        text_in(ui, Rect::from_min_max(Pos2::new(r.left() + 9.0, r.top()), Pos2::new(r.left() + 22.0, r.bottom())), Align2::LEFT_CENTER, &format!("{}", i + 1), mono(10.0), if chosen { ACCENT } else { DIM });
        let cross = Rect::from_center_size(Pos2::new(r.right() - 12.0, r.center().y), egui::vec2(16.0, 16.0));
        let lid = Rect::from_center_size(Pos2::new(cross.left() - 12.0, r.center().y), egui::vec2(18.0, 18.0));
        // (every name starts clear of the dot that a body's first row carries: see below)
        text_in(ui, Rect::from_min_max(Pos2::new(r.left() + 37.0, r.top()), Pos2::new(lid.left() - 4.0, r.bottom())), Align2::LEFT_CENTER, &st.layer_title(i), mono(SIZE), ink);
        if live {
            if resp.clicked() { select = Some(i); }
            if resp.drag_started() { ui.data_mut(|d| d.insert_temp(held, i)); }
            if resp.dragged() && n > 1 { ui.ctx().set_cursor_icon(CursorIcon::Grabbing); }
            if resp.drag_stopped() {
                // let go over another row: the two change places
                let onto = pointer.and_then(|p| (0..n).find(|j| place(*j).contains(p)));
                if let Some(j) = onto.filter(|j| *j != i) { exchange = Some((i, j)); }
                ui.data_mut(|d| d.remove::<usize>(held));
            }
        }
        let mut words = format!("layer {}: {} ({})", i + 1, st.layer_title(i), st.layer_family(i).name);
        if n > 1 { words += &format!("\n{}", body_words(st, body[i], &bodies)); }
        if i == 0 { words += "\nthe first layer's family sets the lights and the starting view"; }
        if n > 1 { words += "\ndrag it onto another layer to exchange the two"; }
        resp.on_hover_text(words);

        // what the body is made of: a dot of its material's colour on the body's first row. A press on it
        // shows that body's material in the materials lab
        if bodies[body[i]].first() == Some(&i) {
            let spot = Rect::from_center_size(Pos2::new(r.left() + 28.0, r.center().y), egui::vec2(14.0, 14.0));
            px.note(format!("layer:{}:material", i + 1), spot);
            let made = ui.interact(spot, Id::unique(("layer material", i)), Sense::click());
            if made.hovered() { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); }
            dot(ui, spot.center(), material::body_colour(st, i), false);
            let of = if i == 0 || st.body_own(i) { material::body_material(st, i) } else { "the first body's material" };
            if made.on_hover_text(format!("made of {of} · press to show its material")).clicked() { lab = Some(i); }
        }

        // the eye: show the layer, or keep it only as something that cuts and masks the others
        px.note(format!("layer:{}:eye", i + 1), lid);
        let seen = ui.interact(lid, Id::unique(("layer eye", i)), Sense::click());
        if seen.hovered() { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); }
        eye(ui, lid, shown, if seen.hovered() { Color32::WHITE } else if shown { BUTTON } else { HINT });
        if seen.on_hover_text(if shown { "hide this layer (it still cuts and masks the others)" } else { "show this layer" }).clicked() { toggle = Some(i); }

        // the cross, on the selected layer when it is not the only one
        if chosen && n > 1 {
            px.note(format!("layer:{}:x", i + 1), cross);
            let gone = ui.interact(cross, Id::unique(("layer cross", i)), Sense::click());
            let hot = gone.hovered();
            if hot { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); }
            let (c, d, ink) = (cross.center(), 3.2, if hot { Color32::WHITE } else { DIM });
            for s in [1.0, -1.0] { ui.painter().line_segment([c + egui::vec2(-d, -d * s), c + egui::vec2(d, d * s)], Stroke::new(1.3, ink)); }
            if gone.on_hover_text("take this layer away, with its joins").clicked() { remove = Some(i); }
        }
    }

    // an outline round the rows of each body; a body whose rows are not neighbours has its parts linked
    let line = Stroke::new(1.0, white(0.17));
    for members in &bodies {
        let mut parts: Vec<Rect> = Vec::new();
        for &i in members {
            match parts.last_mut() {
                Some(last) if (place(i).top() - last.bottom()).abs() < 0.5 => last.max.y = place(i).bottom(),
                _ => parts.push(place(i)),
            }
        }
        for part in &parts { ui.painter().rect_stroke(*part, 5, line, StrokeKind::Inside); }
        for pair in parts.windows(2) {
            let x = space.left() - 5.0;
            let (a, b) = (pair[0].center().y, pair[1].center().y);
            ui.painter().add(egui::Shape::line(vec![Pos2::new(space.left(), a), Pos2::new(x, a), Pos2::new(x, b), Pos2::new(space.left(), b)], line));
        }
    }

    // the "+", at the end of the first row
    let plus = Rect::from_min_size(Pos2::new(space.right() - PLUS, space.top()), egui::vec2(PLUS, ROW));
    if n < MAX_LAYERS {
        px.note("layer:add", plus);
        let words = if n == 1 { "add a layer: a second shape in the same space, joined to this one" } else { "add another layer (four at most)" };
        if button(ui, plus.shrink2(egui::vec2(0.0, 1.0)), Id::unique("layer add"), "+", false, mono(13.0)).on_hover_text(words).clicked() { add(st, px); }
    }

    if let Some(i) = select { st.select_layer(i); }
    // (the material section shows the material of the selected layer's body: it is opened to it)
    if let Some(i) = lab {
        st.select_layer(i);
        if px.store.is_closed("material") { px.store.fold("material"); }
    }
    if let Some(i) = toggle { st.set_layer_show(i, !st.stack.layers[i].show); }
    if let Some((a, b)) = exchange {
        st.swap_layers(a, b);
        px.combo_open = None;
    }
    if let Some(i) = remove {
        st.remove_layer(i);
        px.combo_open = None;
    }
    gap(ui, 4.0);
}

/// Add a layer as the "+" does, and select it. Its join with the layer it was added beside is opened, so
/// that how the two are combined is the first thing seen.
pub fn add(st: &mut State, px: &mut Panels) {
    let was = st.stack.sel;
    if let Some(i) = st.add_layer() {
        px.combo_open = Some(mask_of(&[was, i]));
        if px.store.is_closed("joins") { px.store.fold("joins"); }
    }
}

/// The little arrow of something that opens: pointing down when it is open.
fn arrow(ui: &Ui, c: Pos2, open: bool, ink: Color32) {
    let tip = if open {
        vec![c + egui::vec2(-3.0, -1.5), c + egui::vec2(3.0, -1.5), c + egui::vec2(0.0, 2.2)]
    } else {
        vec![c + egui::vec2(-1.3, -3.0), c + egui::vec2(2.3, 0.0), c + egui::vec2(-1.3, 3.0)]
    };
    ui.painter().add(egui::Shape::convex_polygon(tip, ink, Stroke::NONE));
}

/// A combination's options in the fewest words, for the end of its row: "join · 0.12", "2 cuts 1 · 0.05"
/// (the layers by their numbers: the row has no room for their names twice), "morph · 30%", "apart",
/// "fill +0.50".
fn brief(st: &State, mask: u8) -> String {
    if mask.count_ones() != 2 { return format!("fill {:+.2}", st.stack.junction(mask).fill); }
    let (p, m) = (st.stack.pair(mask), members(mask));
    match p.op {
        Op::Apart => "apart".to_owned(),
        Op::Morph => format!("morph · {:.0}%", 100.0 * p.amount),
        Op::ACutsB => format!("{} cuts {} · {:.2}", m[0] + 1, m[1] + 1, p.radius),
        Op::BCutsA => format!("{} cuts {} · {:.2}", m[1] + 1, m[0] + 1, p.radius),
        op => format!("{} · {:.2}", op.id(), p.radius),
    }
}

/// A row that opens: a combination's layers by number and by name, and at its end its options in a few
/// words. Returns true if it was pressed.
fn combo_title(ui: &mut Ui, id: Id, st: &State, mask: u8, open: bool, dim: bool) -> (bool, Rect) {
    let r = row(ui, 22.0);
    let resp = ui.interact(r, id, Sense::click());
    let hot = resp.hovered();
    if hot { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); }
    let ink = if dim { HINT } else if hot || open { BRIGHT } else { BUTTON };
    arrow(ui, Pos2::new(r.left() + 5.5, r.center().y), open, ink);
    let numbers = members(mask).iter().map(|i| (i + 1).to_string()).collect::<Vec<_>>().join("+");
    let numbers_w = text_width(ui, &numbers, &mono(10.0));
    text_in(ui, Rect::from_min_max(Pos2::new(r.left() + 14.0, r.top()), Pos2::new(r.left() + 14.0 + numbers_w, r.bottom())), Align2::LEFT_CENTER, &numbers, mono(10.0), if dim { HINT } else { ACCENT_MINI });
    // the options at the right; the names take what room is left, cut short if need be
    let says = brief(st, mask);
    let says_w = text_width(ui, &says, &mono(10.0)) + 8.0;
    let title = st.combo_title(mask);
    text_in(ui, Rect::from_min_max(Pos2::new(r.left() + 20.0 + numbers_w, r.top()), Pos2::new(r.right() - says_w, r.bottom())), Align2::LEFT_CENTER, &title, mono(SIZE), ink);
    text_in(ui, Rect::from_min_max(Pos2::new(r.right() - says_w, r.top()), r.max), Align2::RIGHT_CENTER, &says, mono(10.0), HINT);
    (resp.on_hover_text(format!("{title}: {}", st.combo_summary(mask))).clicked(), r)
}

/// A heading that opens and shuts the rows under it: its words, and how many rows there are.
fn meets_title(ui: &mut Ui, id: Id, text: &str, count: usize, open: bool) -> (bool, Rect) {
    gap(ui, 2.0);
    let r = row(ui, 18.0);
    let resp = ui.interact(r, id, Sense::click());
    let hot = resp.hovered();
    if hot { ui.ctx().set_cursor_icon(CursorIcon::PointingHand); }
    let ink = if hot { BRIGHT } else { FAINT };
    arrow(ui, Pos2::new(r.left() + 5.5, r.center().y), open, ink);
    text_in(ui, Rect::from_min_max(Pos2::new(r.left() + 14.0, r.top()), r.max), Align2::LEFT_CENTER, text, mono(10.0), ink);
    text_in(ui, r, Align2::RIGHT_CENTER, &format!("{count}"), mono(10.0), HINT);
    (resp.clicked(), r)
}

/// A small heading over some of the joins' rows.
fn heading(ui: &mut Ui, text: &str) {
    gap(ui, 3.0);
    let r = row(ui, 14.0);
    text_in(ui, r, Align2::LEFT_CENTER, text, mono(10.0), FAINT);
}

/// A line of plain words among a combination's controls.
fn words(ui: &mut Ui, text: &str) {
    gap(ui, 2.0);
    paragraph(ui, text, sans(SMALL), FAINT, 1.4, 0.0);
    gap(ui, 3.0);
}

/// What an operation does, for its hover note.
fn op_words(st: &State, mask: u8, op: Op) -> String {
    let m = members(mask);
    let (a, b) = (st.layer_title(m[0]), st.layer_title(m[1]));
    match op {
        Op::Join => "one solid: the two flow into each other across a seam".to_owned(),
        Op::ACutsB => format!("{a} is taken out of {b}"),
        Op::BCutsA => format!("{b} is taken out of {a}"),
        Op::Overlap => "only what is inside both of them is left".to_owned(),
        Op::Morph => format!("one shape between the two: all {a} at 0, all {b} at 1"),
        Op::Apart => "not combined: both are there, each a body of its own".to_owned(),
    }
}

/// The controls of a combination that is open, drawn in from the row above with a bar down their left.
fn combo_body(ui: &mut Ui, st: &mut State, px: &mut Panels, mask: u8) {
    let (top, left) = (ui.cursor().top(), ui.cursor().left());
    let inner = Rect::from_min_max(Pos2::new(left + 10.0, top), Pos2::new(ui.max_rect().right(), top + 2000.0));
    ui.scope_builder(egui::UiBuilder::new().max_rect(inner), |ui| {
        ui.set_width(inner.width());
        gap(ui, 2.0);
        let key = tag(mask);
        if mask.count_ones() == 2 {
            let now = st.stack.pair(mask);
            let m = members(mask);
            // what the two do to each other: on one line the four that treat both alike, then the two cuts,
            // which are named with the layers' own names
            for (line, ops) in [&[Op::Join, Op::Overlap, Op::Morph, Op::Apart][..], &[Op::ACutsB, Op::BCutsA][..]].into_iter().enumerate() {
                let names: Vec<String> = ops.iter().map(|op| st.op_title(mask, *op)).collect();
                let items: Vec<Item> = ops.iter().zip(&names).map(|(op, name)| Item::new(name, *op == now.op).tip(op_words(st, mask, *op))).collect();
                let run = filled_run(ui, Id::unique(("join op", mask, line)), &items, egui::vec2(7.0, 2.0), &mono(SIZE));
                drop(items);
                for (op, place) in ops.iter().zip(&run.places) { px.note(format!("join:{key}:{}", op.id()), *place); }
                if let Some(i) = run.clicked { st.set_pair_op(mask, ops[i]); }
                gap(ui, 4.0);
            }

            let now = st.stack.pair(mask);
            let (a, b) = (st.layer_title(m[0]), st.layer_title(m[1]));
            let shows = |i: usize| st.stack.layers[i].show;
            match now.op {
                Op::Apart if st.one_body(mask) => {
                    let b0 = st.body_of(m[0]);
                    words(ui, &format!("Apart from each other, but both part of body {} through another layer: they meet in a hard edge.", b0 + 1));
                }
                Op::Apart => words(ui, "Not combined: where these two coincide each keeps its own shape."),
                // a join of something hidden joins nothing; a cutter that shows fills its own cut
                Op::Join | Op::Overlap | Op::Morph if !shows(m[0]) || !shows(m[1]) => {
                    words(ui, &format!("{} is hidden: there is nothing of it to {}.", if shows(m[0]) { &b } else { &a }, now.op.id()));
                }
                Op::ACutsB if shows(m[0]) && now.gap <= 0.0 => words(ui, &format!("{a} shows and fills its own cut: hide it, or give the cut a clearance.")),
                Op::BCutsA if shows(m[1]) && now.gap <= 0.0 => words(ui, &format!("{b} shows and fills its own cut: hide it, or give the cut a clearance.")),
                _ => {}
            }
            // the sliders that mean something for this operation, and no others
            for k in st.combo_knobs(mask) { knob::knob_row(ui, st, &k, px); }
            if now.op.has_profile() {
                let items: Vec<Item> = Profile::ALL
                    .iter()
                    .map(|p| {
                        Item::new(p.id(), *p == now.profile).tip(match p {
                            Profile::Smooth => "the surfaces run into the seam without a line: nothing shows in a reflection",
                            Profile::Round => "close to a fillet of constant radius; its two edges show faintly",
                            Profile::Bevel => "a flat facet with two hard edges",
                        })
                    })
                    .collect();
                // (a short name and three short words: one line, whatever the width of the names' column)
                let run = button_run(ui, Id::unique(("join profile", mask)), 46.0, &items, egui::vec2(7.0, 2.0), &mono(SIZE), 24.0);
                text_in(ui, Rect::from_min_size(run.rect.min, egui::vec2(42.0, run.rect.height())), Align2::LEFT_CENTER, "seam", mono(SIZE), DIM);
                for (p, place) in Profile::ALL.iter().zip(&run.places) { px.note(format!("join:{key}:{}", p.id()), *place); }
                if let Some(i) = run.clicked { st.set_pair_profile(mask, Profile::ALL[i]); }
            }
            reset(ui, st, px, mask, st.stack.pair(mask) == PairOpts::default());
        } else {
            let whole = st.one_body(mask);
            ui.scope(|ui| {
                if !whole { ui.disable(); }
                for k in st.combo_knobs(mask) { knob::knob_row(ui, st, &k, px); }
            });
            words(ui, if whole { "Glass added or taken away where all of these meet." } else { "These are not all one body: there is nowhere for them all to meet." });
            reset(ui, st, px, mask, st.stack.junction(mask) == JunctionOpts::default());
        }
    });
    side_bar(ui, left + 4.0, top + 1.0, ui.cursor().top() - 2.0, 0.4);
    gap(ui, 3.0);
}

/// The small button that puts a combination's options back as they start; greyed while they are so.
fn reset(ui: &mut Ui, st: &mut State, px: &mut Panels, mask: u8, as_new: bool) {
    let font = mono(SIZE);
    let size = button_size(ui, "reset", &font, egui::vec2(8.0, 2.0));
    let r = row(ui, size.y + 2.0);
    let place = Rect::from_min_size(Pos2::new(r.right() - size.x, r.center().y - size.y / 2.0), size);
    px.note(format!("join:{}:reset", tag(mask)), place);
    ui.scope(|ui| {
        if as_new { ui.disable(); }
        if button(ui, place, Id::unique(("join reset", mask)), "reset", false, font).on_hover_text("this combination's options back as they start").clicked() { st.reset_combo(mask); }
    });
}

/// The joins: how the layers are combined. One row for every pair (those that are combined, then those
/// kept apart), every three and the four together; a row opens to that combination's own options, one
/// at a time.
pub fn joins(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let n = st.layer_count();
    if n < 2 {
        px.combo_open = None;
        return;
    }
    let combos = st.stack.combos();
    let pairs: Vec<u8> = combos.iter().copied().filter(|m| m.count_ones() == 2).collect();
    let (joined, apart): (Vec<u8>, Vec<u8>) = pairs.iter().partition(|m| st.stack.pair(**m).op != Op::Apart);
    let says = if pairs.len() == 1 { st.combo_summary(pairs[0]) } else { format!("{} combined · {} apart", joined.len(), apart.len()) };
    if !px.group(ui, "joins", "joins", &says) { return; }
    if px.combo_open.is_some_and(|m| !combos.contains(&m)) { px.combo_open = None; }

    let line = |ui: &mut Ui, st: &mut State, px: &mut Panels, mask: u8| {
        let open = px.combo_open == Some(mask);
        let dim = mask.count_ones() > 2 && !st.one_body(mask);
        let (pressed, at) = combo_title(ui, Id::unique(("join row", mask)), st, mask, open, dim);
        px.note(format!("join:{}", tag(mask)), at);
        if pressed { px.combo_open = if open { None } else { Some(mask) }; }
        if px.combo_open == Some(mask) { combo_body(ui, st, px, mask); }
    };
    // (the two kinds of pair get their headings only when there are both, or there is more than one pair)
    let named = pairs.len() > 1;
    if named && !joined.is_empty() { heading(ui, "combined"); }
    for mask in &joined { line(ui, st, px, *mask); }
    if !apart.is_empty() {
        if named { heading(ui, "kept apart") }
        for mask in &apart { line(ui, st, px, *mask); }
    }
    // threes and the four: their rows are kept under a heading that opens, since there can be five of them
    for (which, size) in [3u32, 4].into_iter().enumerate() {
        let many: Vec<u8> = combos.iter().copied().filter(|m| m.count_ones() == size).collect();
        if many.is_empty() { continue; }
        // (the heading is open if it was opened, or if the row that is open is one of its own)
        let open = px.meets_open[which] || px.combo_open.is_some_and(|m| many.contains(&m));
        let (pressed, at) = meets_title(ui, Id::unique(("joins of", size)), if size == 3 { "where three meet" } else { "where all four meet" }, many.len(), open);
        px.note(format!("joins:{size}"), at);
        if pressed {
            px.meets_open[which] = !open;
            if open && px.combo_open.is_some_and(|m| many.contains(&m)) { px.combo_open = None; }
        }
        if px.meets_open[which] || px.combo_open.is_some_and(|m| many.contains(&m)) {
            for mask in many { line(ui, st, px, mask); }
        }
    }
    gap(ui, 4.0);
}

/// Where the selected layer is placed among the others: moved, turned, made larger or smaller. Offered
/// once there is more than one layer (a single shape is moved about with the camera).
pub fn placement(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let i = st.stack.sel;
    let as_built = st.stack.layers[i].place.is_identity();
    // (a sub-group of the shape: detail, folded at first)
    px.sub(ui, "placement", "placement", if as_built { "as it is built" } else { "moved" }, |ui, px| placement_rows(ui, st, px, i, as_built));
}

/// The rows of the placement: the layer's place, turn and size (the turn of the whole view is the
/// camera's, and called "view turn"), and the reset.
fn placement_rows(ui: &mut Ui, st: &mut State, px: &mut Panels, i: usize, as_built: bool) {
    for k in st.place_knobs(i) { knob::knob_row(ui, st, &k, px); }
    let font = mono(SIZE);
    let size = button_size(ui, "reset", &font, egui::vec2(8.0, 2.0));
    let r = row(ui, size.y + 4.0);
    let place = Rect::from_min_size(Pos2::new(r.right() - size.x, r.center().y - size.y / 2.0), size);
    px.note("placement:reset", place);
    ui.scope(|ui| {
        // nothing to put back while the layer is where it is built
        if as_built { ui.disable(); }
        if button(ui, place, Id::unique("placement reset"), "reset", false, font.clone()).on_hover_text("this layer back where it is built: centred, unturned, full size").clicked() { st.reset_placement(i); }
    });
    // the handles over the picture, for moving, turning and sizing this layer by hand
    let word = if px.handles.on { "hide handles" } else { "handles" };
    let hsize = button_size(ui, word, &font, egui::vec2(8.0, 2.0));
    let hplace = Rect::from_min_size(Pos2::new(place.left() - 8.0 - hsize.x, place.top()), hsize);
    px.note("placement:handles", hplace);
    if button(ui, hplace, Id::unique("placement handles"), word, px.handles.on, font).on_hover_text("arrows, squares and rings over the picture to drag this layer about (G). T: move, turn or size; Ctrl: in steps; Shift: finely; Esc: put them away. While they show, a click on a shape selects its layer").clicked() { px.handles.toggle(); }
}
