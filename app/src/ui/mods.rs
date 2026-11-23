//! The selected object's modifiers (gf_core::mods), on the right panel just above the animation: the stack
//! one row an entry, first applied at the top (each switched off and on, dragged to another place, taken away), the
//! selected entry's controls under its row, and "+ add modifier", which shows the kinds, utility and creative.
//! The surface lab's controls are its own window (ui/sculpt.rs), opened from its row.

use eframe::egui::{self, Align2, Id, Ui};
use gf_core::mods::{Group, Kind, Modifier, KINDS, MAX_MODS};
use gf_core::state::State;

use super::theme::*;
use super::widgets::*;
use super::{knob, Panels};

/// How far the selected entry's controls stand in under its row.
const INDENT: f32 = 10.0;

/// The stack changed: built again quickly while a slider is dragged, fully when it is let go.
fn after(st: &mut State, m: &Moved, set: impl FnOnce(&mut Modifier, f64), i: usize) {
    if let Some(v) = m.value {
        if let Some(e) = st.selected_modifiers_mut().list.get_mut(i) { set(e, v); }
        st.sculpt_changed(true);
    }
    if m.released { st.sculpt_changed(false); }
}

fn changed(st: &mut State) {
    st.sculpt_changed(false);
}

/// An entry's row title.
fn title(m: &Modifier) -> String {
    m.kind.name().to_owned()
}

/// The section: one call from the right panel (just above animate).
pub fn show(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    show_in(ui, st, px, false);
}

/// The stack; `headed`: under a heading of the caller's (the right panel's fold), with no title row.
pub fn show_in(ui: &mut Ui, st: &mut State, px: &mut Panels, headed: bool) {
    let n = st.selected_modifiers().list.len();
    if !headed {
        let r = row(ui, 14.0);
        text_in(ui, r, Align2::LEFT_CENTER, "modifiers", mono(10.0), pal().faint);
        if n > 1 { text_in(ui, r, Align2::RIGHT_CENTER, "applied from bottom to top", mono(10.0), pal().hint); }
    }
    gap(ui, 3.0);
    if let Some(s) = px.mods_sel { if s >= n { px.mods_sel = n.checked_sub(1); } }
    list(ui, st, px);
    // how the stack was applied, when that is not what one would expect (a csg object traced from its baked grid)
    if n > 0 {
        if let Some(note) = &px.mods_note {
            if st.is_csg_object(st.stack.sel) && st.layer_count() == 1 {
                paragraph(ui, note, mono(SMALL), pal().hint, LINE, 0.0);
            }
        }
    }
    add(ui, st, px);
    gap(ui, 4.0);
}

/// A row's height, and the room between rows.
const ROW: f32 = 22.0;
const SPACE: f32 = 3.0;
/// How quickly rows slide to where they are going (per second, of the way left).
const SLIDE: f32 = 22.0;

/// The rows. Each is picked up anywhere but its two buttons (the grip at its left says so) and dragged; while
/// it is carried the rows between where it was and where it would land slide aside to open its place, a line
/// marks that place, and it lands there when let go (Esc while carrying puts everything back). A press
/// without a drag opens or closes the row's controls.
fn list(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    // (a carry is the press that started it: with no button held it is over)
    if !ui.input(|i| i.pointer.any_down() || i.pointer.any_released()) { px.mods_drag = None; }
    let n = st.selected_modifiers().list.len();
    if n == 0 { return; }
    let pitch = ROW + SPACE;
    px.mods_slide.resize(n, 0.0);
    let (dt, pointer) = ui.input(|i| (i.stable_dt.min(0.05), i.pointer.latest_pos()));
    if px.mods_drag.is_some() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        px.mods_drag = None;
    }
    let drag = px.mods_drag.filter(|d| d.0 < n);
    let top = ui.cursor().top();
    // where the row carried would land: the slot nearest its top (rows are evenly spaced while one is carried:
    // the open controls fold away when it is picked up)
    let target = match (drag, pointer) {
        (Some((_, grab)), Some(p)) => Some(((p.y - grab - top) / pitch).round().clamp(0.0, (n - 1) as f32) as usize),
        _ => None,
    };
    // every other row eases toward its place: aside, between where the carried one was and where it would land
    let k = 1.0 - (-dt * SLIDE).exp();
    let mut moving = drag.is_some();
    for i in 0..n {
        let want = match (drag, target) {
            (Some((f, _)), Some(t)) if i != f => {
                if f < t && i > f && i <= t { -pitch } else if t < f && i >= t && i < f { pitch } else { 0.0 }
            }
            _ => 0.0,
        };
        let s = &mut px.mods_slide[i];
        *s = if (want - *s).abs() < 0.25 { want } else { *s + (want - *s) * k };
        if *s != want { moving = true; }
    }
    if moving { ui.ctx().request_repaint(); }

    let mut picked: Option<(usize, f32)> = None;
    let mut dropped = false;
    for slot in 0..n {
        let i = n - 1 - slot;
        if st.selected_modifiers().list.len() != n { break; }
        let m = st.selected_modifiers().list[i].clone();
        let chosen = px.mods_sel == Some(i) && drag.is_none();
        let rest = row(ui, ROW);
        gap(ui, SPACE);
        px.note(format!("mods:{}", i + 1), rest);
        let carried = drag.is_some_and(|d| d.0 == slot);
        let at = rest.translate(egui::vec2(0.0, px.mods_slide[slot]));
        let id = Id::unique(("mods row", i));
        let resp = ui.interact(rest, id, egui::Sense::click_and_drag());
        // its two buttons, at the right (only while nothing is carried)
        let off_r = egui::Rect::from_min_max(egui::pos2(at.right() - 92.0, at.top() + 2.0), egui::pos2(at.right() - 58.0, at.bottom() - 2.0));
        let rm_r = egui::Rect::from_min_max(egui::pos2(at.right() - 54.0, at.top() + 2.0), egui::pos2(at.right(), at.bottom() - 2.0));
        if !carried {
            paint_row(ui.painter(), at, &m, chosen, resp.hovered() && drag.is_none());
        }
        if drag.is_none() {
            px.note(format!("mods:{}:on", i + 1), off_r);
            px.note(format!("mods:{}:remove", i + 1), rm_r);
            let on = button(ui, off_r, id.with("on"), if m.on { "on" } else { "off" }, m.on, mono(10.5)).on_hover_text("switch it off and on (it keeps its place and its numbers)");
            let rm = button(ui, rm_r, id.with("rm"), "remove", false, mono(10.5)).on_hover_text("take it away");
            if on.clicked() {
                st.selected_modifiers_mut().list[i].on = !m.on;
                changed(st);
            } else if rm.clicked() {
                st.selected_modifiers_mut().list.remove(i);
                px.draw_cut.list_moved(st, i, None);
                if m.kind == Kind::SurfaceLab { px.store.sculpt_open = false; }
                px.mods_sel = None;
                px.mods_slide.clear();
                changed(st);
                break;
            } else if resp.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            }
        }
        if resp.drag_started() {
            if let Some(p) = pointer { picked = Some((slot, p.y - rest.top())); }
        }
        if resp.drag_stopped() && carried { dropped = true; }
        if resp.clicked() && drag.is_none() {
            px.mods_sel = if px.mods_sel == Some(i) { None } else { Some(i) };
        }
        if chosen {
            let (ctop, left) = (ui.cursor().top(), ui.cursor().left());
            let inner = egui::Rect::from_min_max(egui::pos2(left + INDENT, ctop), egui::pos2(ui.max_rect().right(), ctop + 8000.0));
            let names = label_width(ui);
            ui.scope_builder(egui::UiBuilder::new().max_rect(inner), |ui| {
                ui.set_width(inner.width());
                set_label_width(ui, names - INDENT);
                gap(ui, 2.0);
                controls(ui, st, px, i);
            });
            set_label_width(ui, names);
            side_bar(ui, left + 3.0, ctop + 1.0, ui.cursor().top() - 2.0, 0.3);
            gap(ui, 4.0);
            px.mods_ctrl_h = ui.cursor().top() - ctop;
        }
    }
    let width = ui.max_rect().width();
    let left = ui.max_rect().left();

    // the row carried, over everything, and the line where it would land
    if let (Some((f, grab)), Some(t), Some(p)) = (drag, target, pointer) {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
        let y = top + t as f32 * pitch - SPACE * 0.5 - 0.5;
        ui.painter().line_segment([egui::pos2(left, y), egui::pos2(left + width, y)], egui::Stroke::new(2.0, pal().accent));
        ui.painter().circle_filled(egui::pos2(left + 2.0, y), 3.0, pal().accent);
        if let Some(m) = st.selected_modifiers().list.get(n - 1 - f) {
            let lifted = egui::Rect::from_min_size(egui::pos2(left, p.y - grab), egui::vec2(width, ROW));
            let painter = ui.ctx().layer_painter(egui::LayerId::new(egui::Order::Tooltip, Id::unique("mods carried")));
            painter.rect_filled(lifted.expand(1.0).translate(egui::vec2(0.0, 3.0)), 4.0, egui::Color32::from_black_alpha(110));
            painter.rect_filled(lifted, 4.0, pal().menu_fill);
            painter.rect_stroke(lifted, 4.0, egui::Stroke::new(1.0, fade(pal().accent, 0.7)), egui::StrokeKind::Inside);
            paint_row(&painter, lifted, m, true, false);
        }
        if dropped {
            // it lands where the line was: the rows that slid aside are in their places already, and it eases
            // the last of the way from where it was let go
            let list = &mut st.selected_modifiers_mut().list;
            let e = list.remove(n - 1 - f);
            list.insert(n - 1 - t, e);
            px.draw_cut.list_moved(st, n - 1 - f, Some(n - 1 - t));
            px.mods_slide = vec![0.0; n];
            px.mods_slide[t] = (p.y - grab) - (top + t as f32 * pitch);
            px.mods_drag = None;
            if f != t { changed(st); }
            ui.ctx().request_repaint();
        }
    } else if dropped {
        px.mods_drag = None;
    }
    if let Some((i, grab)) = picked {
        // picked up: its controls fold away, and the rows under them ease up into the room they leave
        if let Some(s) = px.mods_sel.take() {
            for k in n - s..n { px.mods_slide[k] += px.mods_ctrl_h; }
        }
        px.mods_drag = Some((i, grab));
        ui.ctx().request_repaint();
    }
}

/// One row: its grip, its name (faint while off), its frame; the buttons are drawn over it by the list.
fn paint_row(painter: &egui::Painter, r: egui::Rect, m: &Modifier, chosen: bool, hovered: bool) {
    if chosen || hovered {
        painter.rect_filled(r, 3.0, fade(pal().accent, if chosen { 0.13 } else { 0.06 }));
    }
    for (dx, dy) in [(0.0, -4.0), (0.0, 0.0), (0.0, 4.0), (4.0, -4.0), (4.0, 0.0), (4.0, 4.0)] {
        painter.circle_filled(egui::pos2(r.left() + 6.0 + dx, r.center().y + dy), 1.1, if hovered || chosen { pal().faint } else { pal().hint });
    }
    let ink = if !m.on { pal().hint } else if chosen { pal().bright } else { pal().text };
    painter.text(egui::pos2(r.left() + 18.0, r.center().y), Align2::LEFT_CENTER, title(m), mono(SIZE), ink);
}

/// "+ add modifier", and under it while open the kinds: utility, then creative.
fn add(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let n = st.selected_modifiers().list.len();
    if n >= MAX_MODS { return; }
    let (resp, at) = wide_button(ui, Id::unique("mods add"), "+ add modifier", px.mods_adding);
    px.note("mods:add", at);
    if resp.on_hover_text("change the whole object: utility (mirror, shell, offset, smooth), pattern (repeat, revolve, wrap) or creative (surface lab, twist, swirl, bend, taper, noise, ripple)").clicked() {
        px.mods_adding = !px.mods_adding;
    }
    if !px.mods_adding { return; }
    gap(ui, 2.0);
    let has_lab = st.selected_modifiers().lab_at().is_some();
    for (group, word) in [(Group::Utility, "utility"), (Group::Pattern, "pattern"), (Group::Creative, "creative")] {
        let kinds: Vec<(Kind, &str, &str)> = KINDS.iter().filter(|k| k.0 != Kind::DrawCut || (st.world_modifier_target.is_none() && st.selected_group.is_none())).filter(|k| st.world_modifier_target.is_none() || super::world_modifiers::supported(k.0)).filter(|k| k.2 == group && !(k.0 == Kind::SurfaceLab && has_lab)).map(|k| (k.0, k.1, k.3)).collect();
        let items: Vec<Item> = kinds.iter().map(|(_, name, tip)| Item::new(name, false).tip(*tip)).collect();
        let run = named_run(ui, Id::unique(("mods kinds", word)), word, &items);
        for ((_, name, _), place) in kinds.iter().zip(&run.places) { px.note(format!("mods:add:{name}"), *place); }
        if let Some(k) = run.clicked {
            let kind = kinds[k].0;
            st.selected_modifiers_mut().list.push(Modifier::of(kind));
            px.mods_sel = Some(n);
            px.mods_adding = false;
            // (the lab opens its window at once: its layers are added there)
            if kind == Kind::SurfaceLab { px.store.sculpt_open = true; }
            if kind == Kind::DrawCut { px.draw_cut.arm(st,n); }
            changed(st);
        }
    }
}

/// The controls of entry i.
fn controls(ui: &mut Ui, st: &mut State, px: &mut Panels, i: usize) {
    let Some(m) = st.selected_modifiers().list.get(i).cloned() else { return };
    let key = |what: &str| format!("mods:{}:{what}", i + 1);
    if st.world_modifier_target.is_some() { super::world_modifiers::scope_controls(ui, st, i, &m); }
    if m.kind == Kind::DrawCut { super::draw_cut::controls(ui,st,px,i); }
    if m.kind == Kind::SurfaceLab {
        let s = st.sculpt();
        let live = s.live().count();
        let text = if s.layers.is_empty() { "open surface lab…".to_owned() } else { format!("open surface lab… ({live} layer{})", if live == 1 { "" } else { "s" }) };
        let (resp, at) = wide_button(ui, Id::unique(("mods lab", i)), &text, px.store.sculpt_open);
        px.note(key("open"), at);
        if resp.on_hover_text("its layers: cells, foam, noise, veins, dimples, lattices, a picture of your own").clicked() {
            px.store.sculpt_open = !px.store.sculpt_open;
        }
        gap(ui, 2.0);
        return;
    }
    if m.kind.amounted() {
        let v = knob::value_row(ui, px, &key("amount"), "amount", m.amount, 0.0, 1.0, 0.01);
        after(st, &v, |e, x| e.amount = x, i);
    }
    // a mirror's axes are switched on and off
    if m.kind == Kind::Mirror {
        let a = m.a.get();
        let items: Vec<Item> = ["x", "y", "z"].iter().enumerate().map(|(k, w)| Item::new(w, a[k] > 0.5).tip("reflected across the plane at right angles to this axis")).collect();
        let run = named_run(ui, Id::unique(("mods mirror", i)), "across", &items);
        for (w, place) in ["x", "y", "z"].iter().zip(&run.places) { px.note(key(&format!("mirror:{w}")), *place); }
        if let Some(k) = run.clicked {
            if let Some(e) = st.selected_modifiers_mut().list.get_mut(i) {
                e.a.set_link(false);
                let on = e.a.v[k] > 0.5;
                e.a.v[k] = if on { 0.0 } else { 1.0 };
            }
            changed(st);
        }
    }
    // a ring counts its copies with one number, and so does a path
    if m.kind == Kind::Repeat && m.mode == 1 {
        let mv = knob::value_row(ui, px, &key("ring copies"), "copies", m.a.get()[0], 2.0, gf_core::array::RING_MAX, 1.0);
        after(st, &mv, |e, x| { e.a.set_link(false); e.a.v[0] = x; }, i);
    }
    if m.kind == Kind::Repeat && m.mode == 2 {
        let mv = knob::value_row(ui, px, &key("path copies"), "copies", m.a.get()[0], 1.0, gf_core::array::LIST_MAX as f64, 1.0);
        after(st, &mv, |e, x| { e.a.set_link(false); e.a.v[0] = x; }, i);
    }
    // its per-axis rows: linked (one number for the three) or per axis
    for (r, spec) in m.kind.rows().iter().enumerate() {
        let Some(spec) = spec else { continue };
        if m.kind == Kind::Mirror || (m.kind == Kind::Repeat && m.mode >= 1) { continue; }
        let axes = if r == 0 { m.a } else { m.b };
        // (a wrap's form is three different measures: nothing to link)
        let linkable = m.kind != Kind::Wrap;
        let items = [
            Item::new("linked", axes.link).tip("one number drives all three axes"),
            Item::new("per axis", !axes.link).tip("each axis its own number"),
        ];
        let run = if linkable { named_run(ui, Id::unique(("mods link", i, r)), spec.label, &items) } else { Run { clicked: None, rect: egui::Rect::NOTHING, places: Vec::new() } };
        for (w, place) in ["linked", "per axis"].iter().zip(&run.places) { px.note(key(&format!("{}:{w}", spec.label)), *place); }
        if let Some(k) = run.clicked {
            if let Some(e) = st.selected_modifiers_mut().list.get_mut(i) {
                let a = if r == 0 { &mut e.a } else { &mut e.b };
                a.set_link(k == 0);
            }
            changed(st);
        }
        let v = axes.get();
        let shown: &[usize] = if axes.link { &[0] } else { &[0, 1, 2] };
        for &ax in shown {
            let label = if axes.link { "x y z" } else { spec.axes[ax] };
            let mv = knob::value_row(ui, px, &key(&format!("{}:{ax}", spec.label)), label, v[ax], spec.lo, if m.kind == Kind::Repeat && r == 0 && st.world_modifier_target.is_some() { 64.0 } else { spec.hi }, spec.step);
            after(st, &mv, |e, x| if r == 0 { e.a.set(ax, x) } else { e.b.set(ax, x) }, i);
        }
        if r == 0 && !axes.link {
            // (a row's tip, once: what its numbers mean)
            paragraph(ui, short(spec.tip), mono(9.5), pal().hint, LINE, label_width(ui) + COL_GAP);
        }
    }
    // its own single sliders
    for (j, s) in m.kind.scalars().iter().enumerate() {
        let Some(s) = s else { continue };
        // (a ring's radius only in a ring; the row offset only in a grid)
        if m.kind == Kind::Repeat && (if m.mode == 2 { j > 0 } else { (j == 1) != (m.mode == 1) }) { continue; }
        let now = [m.size, m.detail, if m.kind == Kind::Repeat { m.stagger } else { m.lift }][j];
        let mv = knob::value_row(ui, px, &key(s.label), s.label, now, s.lo, s.hi, s.step);
        after(st, &mv, |e, x| match j { 0 => e.size = x, 1 => e.detail = x, _ => if e.kind == Kind::Repeat { e.stagger = x } else { e.lift = x } }, i);
    }
    // its mode
    let modes = m.kind.modes();
    // (copies along a path are a construction node's: `gf_core::array`)
    let modes = if m.kind == Kind::Repeat && !node_stack(st) { &modes[..2] } else { modes };
    if !modes.is_empty() {
        let items: Vec<Item> = modes.iter().enumerate().map(|(k, (w, tip))| Item::new(w, m.mode as usize == k).tip(*tip)).collect();
        if let Some(k) = named_run(ui, Id::unique(("mods mode", i)), "mode", &items).clicked {
            if let Some(e) = st.selected_modifiers_mut().list.get_mut(i) {
                e.mode = k as u32;
                // (a ring counts its copies on the first number)
                if e.kind == Kind::Repeat && k == 1 && e.a.get()[0] < 2.0 { e.a.set(0, 6.0); }
            }
            changed(st);
        }
    }
    if m.kind == Kind::Repeat {
        let items = [Item::new("fitted", m.fit).tip("the copies scaled down together to stand where the shape stood"), Item::new("as they are", !m.fit).tip("each copy the shape's own size")];
        if let Some(k) = named_run(ui, Id::unique(("mods fit", i)), "size", &items).clicked {
            if let Some(e) = st.selected_modifiers_mut().list.get_mut(i) { e.fit = k == 0; }
            changed(st);
        }
    }
    if m.kind == Kind::Wrap {
        let items = [Item::new("on its face", !m.flip).tip("the motif lies on the form as it lies in its own x and y"), Item::new("standing", m.flip).tip("the motif stands on the form on its own y: a ring or a star about y faces out")];
        if let Some(k) = named_run(ui, Id::unique(("mods wrap face", i)), "motif", &items).clicked {
            if let Some(e) = st.shape_mut().mods.list.get_mut(i) { e.flip = k == 1; }
            changed(st);
        }
    }
    if m.kind == Kind::Repeat && node_stack(st) {
        array_controls(ui, st, px, i, &m);
    }
    if m.kind == Kind::Repeat && m.mode == 1 {
        let items = [Item::new("as they are", !m.flip).tip("every slice the same"), Item::new("mirrored", m.flip).tip("each slice mirrored about its middle, so that neighbours meet as reflections")];
        if let Some(k) = named_run(ui, Id::unique(("mods flip", i)), "slices", &items).clicked {
            if let Some(e) = st.shape_mut().mods.list.get_mut(i) { e.flip = k == 1; }
            changed(st);
        }
    }
    if m.kind == Kind::Noise {
        let mv = knob::value_row(ui, px, &key("seed"), "seed", m.seed, 1.0, 99.0, 1.0);
        after(st, &mv, |e, x| e.seed = x, i);
    }
    // where it acts along its axis
    if m.kind.ranged() {
        let mv = knob::value_row(ui, px, &key("from"), "from", m.from, -1.5, 1.5, 0.01);
        after(st, &mv, |e, x| e.from = x, i);
        let mv = knob::value_row(ui, px, &key("to"), "to", m.to, -1.5, 1.5, 0.01);
        after(st, &mv, |e, x| e.to = x, i);
    }
    // its frame: a centre and an axis
    if m.kind.framed() {
        let open = px.mods_frame;
        let (pressed, at) = fold_heading(ui, Id::unique(("mods frame", i)), "centre and axis", !open, &frame_says(&m));
        px.note(key("frame"), at);
        if pressed { px.mods_frame = !open; }
        if open {
            super::modifier_pivot::controls(ui, st, px, i, &m);
            let mv = knob::value_row(ui, px, &key("turn"), "turn", m.turn, -180.0, 180.0, 1.0);
            after(st, &mv, |e, x| e.turn = x, i);
            let mv = knob::value_row(ui, px, &key("tilt"), "tilt", m.tilt, -90.0, 90.0, 1.0);
            after(st, &mv, |e, x| e.tilt = x, i);
        }
    }
    gap(ui, 2.0);
}

/// The first clause of a tip, short enough for the line under a row.
fn short(tip: &str) -> &str {
    tip.split([':', ',', '(']).next().unwrap_or(tip).trim()
}

/// What the folded frame says: where its centre is and where its axis points, when not the object's own.
fn frame_says(m: &Modifier) -> String {
    let moved = m.centre != [0.0; 3];
    let turned = m.tilt != 90.0;
    match (moved, turned) {
        (false, false) => "the object's".into(),
        (true, false) => format!("at {:.2}, {:.2}, {:.2}", m.centre[0], m.centre[1], m.centre[2]),
        (false, true) => format!("axis turned {:.0}°, tilted {:.0}°", m.turn, m.tilt),
        (true, true) => "moved and turned".into(),
    }
}

/// The selected stack is a construction node's (where an array may be listed: `gf_core::array`).
fn node_stack(st: &State) -> bool {
    st.edit_scope.object().is_some() && !st.edit_scope.construction().is_empty()
}

/// An array's arc, path, steps per copy and blend: plain numbers of the entry (`gf_core::array::ArrayOpts`).
fn array_controls(ui: &mut Ui, st: &mut State, px: &mut Panels, i: usize, m: &Modifier) {
    let key = |what: &str| format!("mods:{}:{what}", i + 1);
    let o = &m.array;
    if m.mode == 1 {
        // (the whole ring is 360: copies evenly round; less, copies from the start to the end, both ends kept)
        let span = if o.span == 0.0 { 360.0 } else { o.span.abs().min(360.0) };
        let mv = knob::value_row(ui, px, &key("span"), "span", span, 1.0, 360.0, 1.0);
        after(st, &mv, |e, x| e.array.span = if x >= 360.0 { 0.0 } else { x }, i);
        let mv = knob::value_row(ui, px, &key("start"), "start", o.start, -180.0, 180.0, 1.0);
        after(st, &mv, |e, x| e.array.start = x, i);
    }
    if m.mode == 2 {
        // the path followed: the shape of one of this object's path parts, or a half turn
        let paths: Vec<String> = st.csg.parts.iter().filter(|p| p.family == gf_core::presets::paths::FAMILY && !p.model.is_empty()).map(|p| p.model.clone()).collect();
        let names: Vec<String> = (1..=paths.len()).map(|k| format!("path {k}")).collect();
        let mut items = vec![Item::new("half turn", o.path.is_empty()).tip("a half turn about the axis")];
        for (text, name) in paths.iter().zip(&names) {
            items.push(Item::new(name, o.path == *text).tip("this path part's shape (not where it stands), from its start to its end"));
        }
        if let Some(k) = named_run(ui, Id::unique(("mods array path", i)), "path", &items).clicked {
            if let Some(e) = st.selected_modifiers_mut().list.get_mut(i) { e.array.path = if k == 0 { String::new() } else { paths[k - 1].clone() }; }
            changed(st);
        }
    }
    if m.mode >= 1 {
        let items = [Item::new("turn with it", !o.keep_heading).tip("each copy turned to face the way the ring or path goes"), Item::new("keep heading", o.keep_heading).tip("each copy keeps the shape's own heading")];
        if let Some(k) = named_run(ui, Id::unique(("mods array heading", i)), "copies", &items).clicked {
            if let Some(e) = st.selected_modifiers_mut().list.get_mut(i) { e.array.keep_heading = k == 1; }
            changed(st);
        }
    }
    // steps: each copy one more than the copy before
    for (ax, w) in ["x", "y", "z"].iter().enumerate() {
        let label = format!("move step {w}");
        let mv = knob::value_row(ui, px, &key(&label), &label, o.step_move[ax], -0.5, 0.5, 0.001);
        after(st, &mv, |e, x| e.array.step_move[ax] = x, i);
    }
    let mv = knob::value_row(ui, px, &key("turn step"), "turn step", o.step_turn, -180.0, 180.0, 1.0);
    after(st, &mv, |e, x| e.array.step_turn = x, i);
    let mv = knob::value_row(ui, px, &key("grow step"), "grow step", o.step_grow, -0.5, 0.5, 0.005);
    after(st, &mv, |e, x| e.array.step_grow = x, i);
    let mv = knob::value_row(ui, px, &key("blend"), "blend", o.blend, 0.0, 0.3, 0.001);
    after(st, &mv, |e, x| e.array.blend = x, i);
    if m.mode == 0 && (o.stepped() || o.blend > 0.0) && !m.listed() {
        paragraph(ui, &format!("steps and blend reach {} copies at most", gf_core::array::LIST_MAX), mono(9.5), pal().hint, LINE, label_width(ui) + COL_GAP);
    }
}