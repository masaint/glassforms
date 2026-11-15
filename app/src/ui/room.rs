//! The light room: a window floating over the picture with everything about the lighting in it. It is
//! opened from the light group of the left panel (or with L), dragged wherever it is wanted, and
//! remembered: whether it is open, and where.
//!
//! From the top: the lighting as it stands, as a small picture of everything round the shape, with a mark
//! on it for every light (press one to choose it, drag it to move the light); the studio (studios set up in
//! one go, a backdrop, a floor); the lights themselves, one at a time, and the light that comes from
//! everywhere; an environment picture to light the shape instead or as well; the shadows and caustics the
//! lights cast; and the turning of the whole lot.

use eframe::egui::{self, Align2, Color32, CursorIcon, Id, Pos2, Rect, Sense, Stroke, Ui};
use gf_core::anim::js_round;
use gf_core::presets::Light;
use gf_core::state::{State, LIGHT_SOFT};

use super::theme::*;
use super::widgets::*;
use super::{gizmo, knob, studio, Panels};

/// The most lights the room lets a family have.
pub const MOST_LIGHTS: usize = 16;
/// How near a mark on the picture of the lighting a press must be to take hold of its light, in points.
const REACH: f32 = 13.0;

/// Show the light room if it is open. `full`: the inside of the program's window.
pub fn show(ctx: &egui::Context, st: &mut State, px: &mut Panels, full: Rect) {
    if !px.store.room_open { return; }
    // it first appears beside the left panel, where it covers neither panel
    let first = px.store.room_at.map_or(Pos2::new(EDGE + LEFT_W + 36.0, EDGE), |p| Pos2::new(p[0], p[1]));
    let max_h = (full.height() - 2.0 * EDGE - 44.0).max(80.0);
    let out = floating(ctx, "light room", "light room", FLOAT_W, first, max_h, |ui| {
        if px.rendering { ui.disable(); }
        set_label_width(ui, COL_LABEL);
        px.place = "room";
        px.light = px.light.min(st.rig().lights.len().saturating_sub(1));
        preview(ui, st, px);
        studio::surroundings(ui, st, px);
        studio::floor(ui, st, px);
        lights(ui, st, px);
        studio::casting(ui, st, px);
        turning(ui, st, px);
        gap(ui, 2.0);
    });
    if out.closed { px.store.room_open = false; }
    px.store.room_at = Some([out.rect.min.x.round(), out.rect.min.y.round()]);
    px.note("room", out.rect);
    px.note("room:bar", out.bar);
    px.note("room:close", out.cross);
}

/// Are the family's lights lit? Always, unless an environment picture lights the shape and they were not
/// asked to stay on with it.
pub fn lights_lit(st: &State) -> bool {
    st.hdri.is_none() || st.p.hdri_lights > 0.5
}

/// A light's colour at full brightness, as the screen shows it.
fn light_ink(l: &Light) -> Color32 {
    let top = l.rgb[0].max(l.rgb[1]).max(l.rgb[2]);
    if top <= 0.0 { Color32::from_gray(70) } else { shown_colour(l.rgb.map(|c| c / top)) }
}

/// The lighting as it stands: a picture of everything round the shape, laid flat (straight ahead in the
/// middle, behind the camera at the two ends, straight up along the top), and on it a mark for every
/// light. A press on a mark chooses its light; a drag moves the light with the pointer.
fn preview(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    gap(ui, 3.0);
    let r = row(ui, (ui.available_width() / 2.0).round());
    px.note("room/preview", r);
    let p = ui.painter().clone();
    p.rect_filled(r, 4, Color32::from_gray(6));
    match px.lighting.preview {
        Some(id) => {
            p.add(egui::epaint::RectShape::filled(r, 4, Color32::WHITE).with_texture(id, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0))));
        }
        // the engine draws it once it has been asked for: it will be there in a frame or two
        None => ui.ctx().request_repaint(),
    }
    p.rect_stroke(r, 4, Stroke::new(1.0, white(0.14)), egui::StrokeKind::Inside);

    let lit = lights_lit(st);
    let at = |st: &State, l: &Light| {
        let (u, v) = st.light_on_preview(l.az, l.el);
        Pos2::new(r.left() + u as f32 * r.width(), r.top() + v as f32 * r.height())
    };
    let resp = ui.interact(r, Id::unique("lighting preview"), Sense::click_and_drag());
    let live = ui.is_enabled() && lit;
    let held = Id::unique("light held on the preview");
    let nearest = |st: &State, from: Pos2| -> Option<usize> {
        let near = st.rig().lights.iter().enumerate().map(|(i, l)| (i, at(st, l).distance(from))).filter(|(_, d)| *d <= REACH);
        near.min_by(|a, b| a.1.total_cmp(&b.1)).map(|(i, _)| i)
    };
    if live {
        // a press takes hold of the nearest mark within reach
        if resp.drag_started() || resp.clicked() {
            let from = ui.input(|i| i.pointer.press_origin()).or(resp.interact_pointer_pos());
            let found = from.and_then(|from| nearest(st, from));
            if let Some(i) = found { px.light = i; }
            if resp.drag_started() { ui.data_mut(|d| d.insert_temp(held, found)); }
        }
        let grabbed = ui.data(|d| d.get_temp::<Option<usize>>(held)).flatten().filter(|i| *i < st.rig().lights.len());
        if let (true, Some(i), Some(to)) = (resp.dragged(), grabbed, resp.interact_pointer_pos()) {
            // the light goes where the pointer is, to the nearest half degree
            let (u, v) = (((to.x - r.left()) / r.width()).clamp(0.0, 1.0) as f64, ((to.y - r.top()) / r.height()).clamp(0.003, 0.997) as f64);
            let (az, el) = st.light_at_preview(u, v);
            let half = |x: f64| (x * 2.0).round() / 2.0;
            let l = &mut st.rig_mut().lights[i];
            (l.az, l.el) = (half(az), half(el));
            ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
        } else if resp.hovered() && resp.hover_pos().is_some_and(|h| nearest(st, h).is_some()) {
            ui.ctx().set_cursor_icon(CursorIcon::Grab);
        }
        if resp.drag_stopped() { ui.data_mut(|d| d.remove::<Option<usize>>(held)); }
    }

    // the marks: a ring in the light's colour with its number; the chosen one stands out
    if lit {
        let clip = p.with_clip_rect(r);
        for (i, l) in st.rig().lights.iter().enumerate() {
            let (c, chosen) = (at(st, l), i == px.light);
            px.note(format!("room/mark:{}", i + 1), Rect::from_center_size(c, egui::vec2(12.0, 12.0)));
            let radius = if chosen { 6.5 } else { 5.0 };
            clip.circle_stroke(c, radius + 1.2, Stroke::new(1.5, Color32::from_black_alpha(170)));
            clip.circle_stroke(c, radius, Stroke::new(if chosen { 2.0 } else { 1.4 }, if chosen { Color32::WHITE } else { light_ink(l) }));
            if chosen { clip.circle_filled(c, 2.6, light_ink(l)); }
            let label = Rect::from_min_size(Pos2::new(c.x + radius + 2.0, c.y - radius - 8.0), egui::vec2(16.0, 10.0));
            for d in [egui::vec2(1.0, 1.0), egui::vec2(-1.0, 0.0)] {
                clip.text(label.left_center() + d, Align2::LEFT_CENTER, format!("{}", i + 1), mono(9.0), Color32::from_black_alpha(190));
            }
            clip.text(label.left_center(), Align2::LEFT_CENTER, format!("{}", i + 1), mono(9.0), if chosen { Color32::WHITE } else { BUTTON });
        }
    }
    gap(ui, 1.0);
}

/// The family's lights: which one is being worked on, adding, copying and taking away; the chosen one's
/// colour, place, size and softness; the light from everywhere; all of them brighter or dimmer together;
/// and back to the lights the family came with.
fn lights(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let (n, own, lit) = (st.rig().lights.len(), st.rig_is_default(), lights_lit(st));
    let says = format!("{n} · {}{}", if own { "the family's own" } else { "changed" }, if lit { "" } else { " · not lit" });
    if !px.window_group(ui, "room-lights", "lights", &says) { return; }
    if st.hdri.is_some() {
        // with an environment picture the lights are an extra
        let run = named_run(ui, Id::unique("lights with picture"), "with picture", &[
            Item::new("off", !lit).tip("the picture alone lights the shape"),
            Item::new("lit too", lit).tip("the family's lights stay lit together with the picture"),
        ]);
        px.note("room/lit", run.rect);
        for (place, name) in run.places.iter().zip(["room/lit:off", "room/lit:on"]) { px.note(name, *place); }
        if let Some(i) = run.clicked { st.p.hdri_lights = i as f64; }
        if !lights_lit(st) { return; }
    }

    // one chip for every light
    let width = ui.available_width();
    let (size, space) = (24.0f32, 4.0f32);
    let per_line = (((width + space) / (size + space)).floor() as usize).max(1);
    let lines = n.div_ceil(per_line).max(1);
    let r = row(ui, lines as f32 * (size + space) - space);
    let mut choose = None;
    for i in 0..n {
        let place = Rect::from_min_size(Pos2::new(r.left() + (i % per_line) as f32 * (size + space), r.top() + (i / per_line) as f32 * (size + space)), egui::vec2(size, size));
        px.note(format!("room/light:{}", i + 1), place);
        let l = &st.rig().lights[i];
        let tip = format!("light {}: at {:.0}°, {:.0}° up", i + 1, l.az, l.el);
        if chip(ui, place, Id::unique(("light chip", i)), &format!("{}", i + 1), light_ink(l), i == px.light).on_hover_text(tip).clicked() { choose = Some(i); }
    }
    if n == 0 { text_in(ui, r, Align2::LEFT_CENTER, "no lights", mono(SIZE), FAINT); }
    if let Some(i) = choose { px.light = i; }
    gap(ui, 4.0);

    // more of them, fewer of them; and at the end of the line, back to the lights the family came with
    let items = [
        Item::new("add", false).tip("a new white light, straight ahead and a little up"),
        Item::new("copy", false).tip("another like the chosen one, a little to its right"),
        Item::new("delete", false).tip("take the chosen light away"),
    ];
    let (pad, font) = (egui::vec2(8.0, 2.0), mono(SIZE));
    let run = button_run(ui, Id::unique("lights more or fewer"), 0.0, &items, pad, &font, 22.0);
    for (place, name) in run.places.iter().zip(["room/add", "room/copy", "room/delete"]) { px.note(name, *place); }
    let size = button_size(ui, "reset", &font, pad);
    let place = Rect::from_min_size(Pos2::new(run.rect.right() - size.x, run.rect.center().y - size.y / 2.0), size);
    px.note("room/reset", place);
    let mut reset = false;
    ui.scope(|ui| {
        // nothing to put back while the lights are the family's own
        if own { ui.disable(); }
        reset = button(ui, place, Id::unique("lights reset"), "reset", false, font.clone()).on_hover_text("back to the lights this family came with, and its light from everywhere").clicked();
    });
    match run.clicked {
        Some(0) if n < MOST_LIGHTS => {
            st.rig_mut().lights.push(Light { az: 0.0, el: 25.0, w: 10.0, h: 10.0, rgb: [3.0, 3.0, 3.0], soft: LIGHT_SOFT });
            px.light = n;
        }
        Some(1) if n > 0 && n < MOST_LIGHTS => {
            let mut twin = st.rig().lights[px.light].clone();
            twin.az = (twin.az + 20.0 + 180.0).rem_euclid(360.0) - 180.0;
            st.rig_mut().lights.insert(px.light + 1, twin);
            px.light += 1;
        }
        Some(2) if n > 0 => {
            st.rig_mut().lights.remove(px.light);
            px.light = px.light.saturating_sub(1);
        }
        _ => {}
    }
    if reset {
        st.reset_rig();
        px.light = 0;
    }
    gap(ui, 4.0);

    // the chosen light
    let i = px.light;
    if let Some(l) = st.rig().lights.get(i).cloned() {
        let key = format!("light{}", i + 1);
        let mut new = l.clone();
        if let Some(rgb) = knob::colour_rows(ui, px, &key, l.rgb, 12.0, 0.05) { new.rgb = rgb; }
        // (an azimuth is shown between -180 and 180, wherever the family wrote it)
        let az = (l.az + 180.0).rem_euclid(360.0) - 180.0;
        if let Some(v) = knob::value_row(ui, px, &format!("{key}.az"), "azimuth °", az, -180.0, 180.0, 0.5).value { new.az = v; }
        if let Some(v) = knob::value_row(ui, px, &format!("{key}.el"), "elevation °", l.el, -90.0, 90.0, 0.5).value { new.el = v; }
        if let Some(v) = knob::value_row(ui, px, &format!("{key}.w"), "width °", l.w, 1.0, 90.0, 0.5).value { new.w = v; }
        if let Some(v) = knob::value_row(ui, px, &format!("{key}.h"), "height °", l.h, 1.0, 90.0, 0.5).value { new.h = v; }
        if let Some(v) = knob::value_row(ui, px, &format!("{key}.soft"), "softness", l.soft, 0.0, 1.0, 0.01).value { new.soft = v; }
        if new != l { st.rig_mut().lights[i] = new; }
    }

    // the light that comes from everywhere, however the lights stand
    gap(ui, 6.0);
    let r = row(ui, 14.0);
    text_in(ui, r, Align2::LEFT_CENTER, "from everywhere", mono(SMALL), FAINT);
    if let Some(rgb) = knob::colour_rows(ui, px, "ambient", st.rig().ambient, 0.5, 0.001) { st.rig_mut().ambient = rgb; }

    // all of it brighter or dimmer together
    gap(ui, 6.0);
    knob::knob_row(ui, st, &State::engine_knob("lightGain"), px);
}

/// Turning the whole lighting, lights and picture alike: the ball, the three angles, circling by itself,
/// and whether it stays with the room or goes round with the shape.
fn turning(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let turned = st.p.lyaw != 0.0 || st.p.lpitch != 0.0 || st.p.lroll != 0.0 || st.lphi != 0.0;
    let says = if st.p.lspin > 0.0 { "circling" } else if turned { "turned" } else { "as it started" };
    if !px.window_group(ui, "room-turn", "turn the lighting", says) { return; }
    let r = row(ui, gizmo::SIZE);
    let ball = Rect::from_min_size(r.min, egui::vec2(gizmo::SIZE, gizmo::SIZE));
    px.note("ball", ball);
    gizmo::light_ball(ui, st, ball, lights_lit(st).then_some(px.light));
    // beside the ball: what the lighting is fixed to, and how to read the ball
    let beside = Rect::from_min_max(Pos2::new(ball.right() + 12.0, r.top()), r.max);
    let fixed = js_round(st.p.lights) as usize;
    text_in(ui, Rect::from_min_size(beside.min, egui::vec2(beside.width(), 16.0)), Align2::LEFT_CENTER, "fixed to", mono(SIZE), DIM);
    let (pad, font) = (egui::vec2(7.0, 2.0), mono(SIZE));
    let mut x = beside.left();
    for (i, (name, tip)) in [("the room", "the shape turns in front of lighting that stays where it is"), ("the shape", "the lighting goes round with the shape")].into_iter().enumerate() {
        let size = button_size(ui, name, &font, pad);
        let place = Rect::from_min_size(Pos2::new(x, beside.top() + 19.0), size);
        if button(ui, place, Id::unique(("fixed to", i)), name, i == fixed, font.clone()).on_hover_text(tip).clicked() { st.p.lights = i as f64; }
        x += size.x + 4.0;
    }
    let size = button_size(ui, "turn back", &font, pad);
    let place = Rect::from_min_size(Pos2::new(beside.left(), beside.bottom() - size.y - 4.0), size);
    px.note("room/turn back", place);
    ui.scope(|ui| {
        // nothing to put back while it is as it started
        if !turned { ui.disable(); }
        if button(ui, place, Id::unique("lights back"), "turn back", false, font.clone()).on_hover_text("the lighting back the way it started").clicked() { st.reset_lights(); }
    });
    gap(ui, 6.0);
    let knobs = State::engine_knobs();
    for name in ["lyaw", "lpitch", "lroll", "lspin"] {
        if let Some(k) = knobs.iter().find(|k| k.id == name) { knob::knob_row(ui, st, k, px); }
    }
}
