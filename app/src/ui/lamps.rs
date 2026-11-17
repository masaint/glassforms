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
            let word = match l.kind { LampKind::Ball => "o", LampKind::Tube => "|", LampKind::Panel => "□" };
            let tip = format!("lamp {}: a {} at {:.2}, {:.2}, {:.2}", i + 1, l.kind.id(), l.pos[0], l.pos[1], l.pos[2]);
            if chip(ui, place, Id::unique(("lamp chip", i)), word, ink, i == px.lamp).on_hover_text(tip).clicked() { choose = Some(i); }
        }
        if let Some(i) = choose {
            px.lamp = i;
            if px.handles.lamp.is_some() { px.handles.lamp = Some(i); }
        }
        gap(ui, 4.0);
    } else {
        let r = row(ui, 16.0);
        text_in(ui, r, Align2::LEFT_CENTER, "lights standing in the scene, at a place", mono(SMALL), FAINT);
    }

    // more of them, fewer of them
    let items = [
        egui_item("ball", "a glowing ball, up and to the right in front of the shape"),
        egui_item("tube", "a glowing tube, lying across"),
        egui_item("panel", "a glowing rectangle facing the shape: a soft box"),
        egui_item("delete", "take the chosen lamp away"),
    ];
    let (pad, font) = (egui::vec2(8.0, 2.0), mono(SIZE));
    let run = button_run(ui, Id::unique("lamps more or fewer"), 0.0, &items, pad, &font, 22.0);
    for (place, name) in run.places.iter().zip(["room/lamp-ball", "room/lamp-tube", "room/lamp-panel", "room/lamp-delete"]) { px.note(name, *place); }
    match run.clicked {
        Some(k) if k < 3 && n < MAX_LAMPS => {
            let kind = [LampKind::Ball, LampKind::Tube, LampKind::Panel][k];
            if let Some(i) = st.add_lamp(kind) {
                px.lamp = i;
                px.handles.lamp = Some(i);
            }
        }
        Some(3) if n > 0 => {
            st.remove_lamp(px.lamp);
            px.lamp = px.lamp.saturating_sub(1);
            px.handles.lamp = None;
        }
        _ => {}
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
    for (c, k) in knobs.iter().enumerate() {
        // a ball has no length
        if c == 7 && l.kind == LampKind::Ball { continue; }
        let label = match (c, l.kind) {
            (6, LampKind::Ball) => "radius",
            (6, LampKind::Tube) => "radius",
            (7, LampKind::Tube) => "half length",
            (6, LampKind::Panel) => "half width",
            (7, LampKind::Panel) => "half height",
            _ => k.label.strip_prefix("lamp ").unwrap_or(&k.label),
        };
        let k = gf_core::state::Knob { label: label.into(), ..k.clone() };
        knob::knob_row(ui, st, &k, px);
    }
    // the handles over the picture, on this lamp
    let on_lamp = px.handles.on && px.handles.lamp == Some(i);
    let word = if on_lamp { "hide handles" } else { "handles" };
    let size = button_size(ui, word, &font, pad);
    let r = row(ui, size.y + 4.0);
    let place = Rect::from_min_size(Pos2::new(r.right() - size.x, r.center().y - size.y / 2.0), size);
    px.note("room/lamp-handles", place);
    if button(ui, place, Id::unique("lamp handles"), word, on_lamp, font).on_hover_text("arrows, squares and rings over the picture to drag this lamp about (G); T: all at once, move, turn or size").clicked() {
        if on_lamp { px.handles.toggle(); } else {
            px.handles.lamp = Some(i);
            if !px.handles.on { px.handles.toggle(); }
        }
    }
}

fn egui_item(word: &'static str, tip: &'static str) -> Item<'static> {
    Item::new(word, false).tip(tip)
}
