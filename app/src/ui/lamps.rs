//! The light room's lamps: lights that stand in the scene at a place (a ball, a tube, a panel), beside
//! the lights at infinity above them. None until one is added. Each lamp's numbers are ordinary sliders
//! (they animate and are kept in a setup); the handles over the picture move, turn and size the chosen
//! one (G).

use eframe::egui::{self, Align2, Id, Pos2, Rect, Ui};
use gf_core::lamps::{LampKind, MAX_LAMPS};
use gf_core::state::State;

use super::theme::*;
use super::widgets::*;
use super::{knob, Panels};

pub fn lamps(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let n = st.lamps.len();
    let says = if n == 0 { "none".to_string() } else { format!("{n}") };
    if !px.window_group(ui, "room-lamps", "lamps", &says) { return; }
    px.lamp = px.lamp.min(n.saturating_sub(1));

    // one chip for every lamp
    if n > 0 {
        let (size, space) = (24.0f32, 4.0f32);
        let r = row(ui, size);
        let mut choose = None;
        for i in 0..n {
            let place = Rect::from_min_size(Pos2::new(r.left() + i as f32 * (size + space), r.top()), egui::vec2(size, size));
            px.note(format!("room/lamp:{}", i + 1), place);
            let l = &st.lamps[i];
            let top = l.rgb[0].max(l.rgb[1]).max(l.rgb[2]).max(1e-6);
            let ink = shown_colour(l.rgb.map(|c| c / top));
            let tip = format!("lamp {}: a {} at {:.2}, {:.2}, {:.2}", i + 1, l.kind.id(), l.pos[0], l.pos[1], l.pos[2]);
            if chip(ui, place, Id::unique(("lamp chip", i)), &format!("{}", i + 1), ink, i == px.lamp).on_hover_text(tip).clicked() { choose = Some(i); }
        }
        if let Some(i) = choose {
            px.lamp = i;
            if px.handles.lamp.is_some() { px.handles.lamp = Some(i); }
        }
        gap(ui, 4.0);
    } else {
        let r = row(ui, 16.0);
        text_in(ui, r, Align2::LEFT_CENTER, "lights standing in the scene, at a place", mono(SMALL), pal().faint);
    }

    // more of them, fewer of them
    let items = [
        egui_item("ball", "a glowing ball, in view up and to the left in front of the shape"),
        egui_item("tube", "a glowing tube, lying across, in view up and to the left"),
        egui_item("panel", "a glowing rectangle facing the shape: a soft box, in view up and to the left"),
    ];
    let (pad, font) = (egui::vec2(8.0, 2.0), mono(SIZE));
    let run = named_run(ui, Id::unique("lamps more"), "add", &items);
    for (place, name) in run.places.iter().zip(["room/lamp-ball", "room/lamp-tube", "room/lamp-panel"]) { px.note(name, *place); }
    let size = button_size(ui, "delete", &font, pad);
    let place = Rect::from_min_size(Pos2::new(run.rect.right() - size.x, run.rect.center().y - size.y / 2.0), size);
    px.note("room/lamp-delete", place);
    let mut delete = false;
    ui.scope(|ui| {
        if n == 0 { ui.disable(); }
        delete = button(ui, place, Id::unique("lamp delete"), "delete", false, font.clone()).on_hover_text("take the chosen lamp away").clicked();
    });
    match run.clicked {
        Some(k) if n < MAX_LAMPS => {
            let kind = [LampKind::Ball, LampKind::Tube, LampKind::Panel][k];
            if let Some(i) = st.add_lamp(kind) {
                // where it can be seen, chosen, with its handles on it
                if let Some(pos) = in_view(st, px) {
                    st.lamps[i].pos = pos;
                    if kind == LampKind::Panel { st.lamps[i].rot = gf_core::lamps::Lamp::facing_middle(pos); }
                    st.lamps_rev += 1;
                }
                px.lamp = i;
                px.handles.pick(Some(i));
                if !px.handles.on { px.handles.set_tool(super::handles::Tool::All); }
            }
        }
        _ => {}
    }
    if delete && n > 0 {
        st.remove_lamp(px.lamp);
        px.lamp = px.lamp.saturating_sub(1);
        px.handles.lamp = None;
    }

    // the chosen lamp
    let i = px.lamp;
    if i >= st.lamps.len() { return; }
    gap(ui, 4.0);
    let l = st.lamps[i];
    let run = named_run(ui, Id::unique(("lamp kind", i)), "kind", &[
        Item::new("ball", l.kind == LampKind::Ball).tip("a glowing ball: size is its radius"),
        Item::new("tube", l.kind == LampKind::Tube).tip("a glowing tube along its own x: size is its radius, length half its length"),
        Item::new("panel", l.kind == LampKind::Panel).tip("a glowing rectangle shining from its front: size and length are its half width and half height"),
    ]);
    if let Some(k) = run.clicked {
        st.lamps[i].kind = [LampKind::Ball, LampKind::Tube, LampKind::Panel][k];
        st.lamps_rev += 1;
    }
    let run = named_run(ui, Id::unique(("lamp seen", i)), "seen", &[
        Item::new("hidden", !l.seen).tip("the camera does not see the lamp itself; the glass still mirrors it"),
        Item::new("shown", l.seen).tip("the camera sees the lamp as a glowing shape"),
    ]);
    if let Some(k) = run.clicked {
        st.lamps[i].seen = k == 1;
        st.lamps_rev += 1;
    }
    let knobs = st.lamp_knobs_of(i);
    let row_of = |c: usize| {
        let k = &knobs[c];
        let label = match (c, l.kind) {
            (6, LampKind::Ball) => "radius",
            (6, LampKind::Tube) => "radius",
            (7, LampKind::Tube) => "half length",
            (6, LampKind::Panel) => "half width",
            (7, LampKind::Panel) => "half height",
            _ => k.label.strip_prefix("lamp ").unwrap_or(&k.label),
        };
        gf_core::state::Knob { label: label.into(), ..k.clone() }
    };
    // its colour, set as a light's is: hue and saturation (its power is how bright); or, to animate it, as
    // its red, green and blue sliders (which is also how it shows while any of the three travels)
    let travelling = (9..=11).any(|c| st.tracks.get(&knobs[c].key).is_some_and(|t| t.on));
    let parts_id = Id::unique(("lamp colour in parts", i));
    let parts = travelling || ui.data(|d| d.get_temp::<bool>(parts_id)).unwrap_or(false);
    if parts {
        for c in 9..=11 { knob::knob_row(ui, st, &row_of(c), px); }
    } else {
        let rgb = st.lamps[i].rgb;
        let top = rgb[0].max(rgb[1]).max(rgb[2]);
        if let Some(new) = knob::hue_rows(ui, px, &format!("lamp{}", i + 1), rgb) {
            st.lamps[i].rgb = new.map(|c| (c * top.max(1e-3)).clamp(0.0, 1.0));
            st.lamps_rev += 1;
        }
    }
    let run = named_run(ui, Id::unique(("lamp colour set by", i)), "colour by", &[
        Item::new("hue", !parts).tip("its hue and how far it is from white, as the lights' colours are set"),
        Item::new("red green blue", parts).tip("the three parts as sliders of their own, each of which can be set travelling"),
    ]);
    if let (Some(k), false) = (run.clicked, travelling) { ui.data_mut(|d| d.insert_temp(parts_id, k == 1)); }
    knob::knob_row(ui, st, &row_of(8), px);
    knob::knob_row(ui, st, &row_of(12), px);
    // its size (a ball has no length), then where it stands and how it is turned, folded as a layer's
    // placement is
    for c in [6, 7] {
        if c == 7 && l.kind == LampKind::Ball { continue; }
        knob::knob_row(ui, st, &row_of(c), px);
    }
    let says = format!("{:.2}, {:.2}, {:.2}", l.pos[0], l.pos[1], l.pos[2]);
    px.sub(ui, "room-lamp-place", "placement", &says, |ui, px| {
        for c in 0..6 { knob::knob_row(ui, st, &row_of(c), px); }
    });
    // the handles over the picture, on this lamp
    let on_lamp = px.handles.shown_on(st, Some(i));
    let word = if on_lamp { "hide handles" } else { "handles" };
    let size = button_size(ui, word, &font, pad);
    let r = row(ui, size.y + 4.0);
    let place = Rect::from_min_size(Pos2::new(r.right() - size.x, r.center().y - size.y / 2.0), size);
    px.note("room/lamp-handles", place);
    if button(ui, place, Id::unique("lamp handles"), word, on_lamp, font).on_hover_text("arrows, squares and rings over the picture to drag this lamp about (G); T: all at once, move, turn or size").clicked() {
        if on_lamp { px.handles.toggle(); } else { px.handles.show_on(Some(i)); }
    }
}

fn egui_item(word: &'static str, tip: &'static str) -> Item<'static> {
    Item::new(word, false).tip(tip)
}

/// A place for a new lamp where it can be seen: up and to the left in the free part of the picture (between
/// the panels), most of the way from the lens to the middle of the scene, in the lights' frame.
fn in_view(st: &State, px: &Panels) -> Option<[f64; 3]> {
    let full = px.full;
    if full.height() < 2.0 { return None; }
    let lo = px.left_at.map_or(full.left(), |r| r.right());
    // (clear of the light room, from which it is added)
    let lo = match px.store.room_at { Some(p) if px.store.room_open => lo.max(p[0] + FLOAT_W + 24.0), _ => lo };
    let hi = px.right_at.map_or(full.right(), |r| r.left());
    let at = Pos2::new(lo + 0.25 * (hi - lo).max(0.0), full.top() + 0.3 * full.height());
    let h = full.height() as f64 / 2.0;
    let (o, d) = st.camera().ray([(at.x - full.center().x) as f64 / h, -(at.y - full.center().y) as f64 / h])?;
    let far = (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt().max(0.5);
    let t = 0.8 * far;
    let c = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
    Some(gf_core::math::apply(&gf_core::math::transpose(&st.light_m()), c))
}
