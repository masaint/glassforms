//! The surface studio: a window floating over the picture in which the selected layer's surface is
//! reshaped by a stack of procedural layers (gf_core::sculpt says what they are; app/src/sculpt.rs does it).
//!
//! It is opened from the right panel, under the shape's own sliders. From the top: the studio on or
//! off and what it last cost; the layers, one open at a time (each with its kind, how
//! it changes the surface, how it blends with the layers before it, its mask, its opacity and sliders,
//! and buttons to switch it off, move it, copy it or take it away); a menu that adds a layer; and the
//! cleanups of the whole stack.

use eframe::egui::{self, Align2, Id, Pos2, Rect, Ui};
use gf_core::sculpt::{Kind, Layer, Mask, Sculpt, BLENDS, KINDS, MASKS, MAX_LAYERS, MODES};
use gf_core::state::State;

use super::theme::*;
use super::widgets::*;
use super::{knob, Panels, Request};

const W: f32 = LAB_W;

/// The button on the right panel that opens the studio, with a word on what is in it.
pub fn button(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let s = st.sculpt();
    let n = s.live().count();
    let text = if s.layers.is_empty() { "surface studio…".to_owned() } else if !s.on { "surface studio… (off)".to_owned() } else { format!("surface studio… ({n} layer{})", if n == 1 { "" } else { "s" }) };
    let (resp, at) = wide_button(ui, Id::unique("surface studio"), &text, px.store.sculpt_open);
    px.note("sculpt button", at);
    if resp.on_hover_text("reshape the surface of this layer's solid: cells, foam, noise, veins, dimples, lattices, a picture of your own, layered over one another").clicked() {
        px.store.sculpt_open = !px.store.sculpt_open;
    }
}

/// Show the studio if it is open. `full`: the inside of the program's window.
pub fn show(ctx: &egui::Context, st: &mut State, px: &mut Panels, full: Rect) {
    if !px.store.sculpt_open { return; }
    // it first appears to the left of the right panel
    let first = px.store.sculpt_at.map_or(Pos2::new(full.right() - EDGE - RIGHT_W - 56.0 - W, EDGE), |p| Pos2::new(p[0], p[1]));
    let max_h = (full.height() - 2.0 * EDGE - 44.0).max(80.0);
    let title = if st.layer_count() > 1 { format!("surface studio · {}", st.layer_title(st.stack.sel)) } else { "surface studio".to_owned() };
    let out = floating(ctx, "surface studio", &title, W, first, max_h, |ui| {
        if px.rendering { ui.disable(); }
        set_label_width(ui, COL_LABEL);
        px.place = "sculpt";
        head(ui, st, px);
        layers(ui, st, px);
        cleanup(ui, st, px);
        gap(ui, 2.0);
    });
    if out.closed { px.store.sculpt_open = false; }
    px.store.sculpt_at = Some([out.rect.min.x.round(), out.rect.min.y.round()]);
    px.note("sculpt", out.rect);
    px.note("sculpt:close", out.cross);
}

/// What the user did to a slider, made: `set` puts the value in, and the shape is built again (quickly
/// while the slider is dragged, fully when it is let go).
fn after(st: &mut State, m: &Moved, set: impl FnOnce(&mut Sculpt, f64)) {
    if let Some(v) = m.value {
        set(st.sculpt_mut(), v);
        st.sculpt_changed(true);
    }
    if m.released { st.sculpt_changed(false); }
}

fn changed(st: &mut State) {
    st.sculpt_changed(false);
}

/// The studio on or off, and what it cost the last time it ran.
fn head(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    gap(ui, 4.0);
    let on = st.sculpt().on;
    let items = [Item::new("on", on).tip("the studio reshapes the surface"), Item::new("off", !on).tip("the shape as it is, the studio's layers kept")];
    let run = named_run(ui, Id::unique("sculpt on"), "studio", &items);
    if let Some(i) = run.clicked {
        st.sculpt_mut().on = i == 0;
        changed(st);
    }
    let says = match px.sculpted {
        Some((res, ms)) if !st.sculpt().is_empty() => format!("last made in {ms:.0} ms, {res} voxels a side"),
        _ if !st.sculpt().is_empty() => "being made…".to_owned(),
        _ => "add a layer with the menu below".to_owned(),
    };
    let r = row(ui, 14.0);
    text_in(ui, r, Align2::LEFT_CENTER, &says, mono(10.0), HINT);
    if let Some(e) = &px.sculpt_error {
        note(ui, e);
    }
}

/// A layer's line when it is folded: what it is and how much of it acts.
fn layer_says(l: &Layer) -> String {
    let mode = MODES.iter().find(|m| m.0 == l.mode).map_or("", |m| m.1);
    if !l.on { return "off".into(); }
    if l.kind == Kind::Warp { return format!("{:.0}%", l.opacity * 100.0); }
    format!("{mode} · {:.0}%", l.opacity * 100.0)
}

fn layers(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let n = st.sculpt().layers.len();
    section(ui, "layers", if n == 0 { "none yet" } else { "first applied at the top" });
    if let Some(open) = px.sculpt_layer { if open >= n { px.sculpt_layer = n.checked_sub(1); } }
    for i in 0..n {
        // (a layer's own buttons may take it away, move it or copy it: the list drawn this frame stops
        // there, and the next frame draws it as it now is)
        if st.sculpt().layers.len() != n { break; }
        let Some(l) = st.sculpt().layers.get(i).cloned() else { break };
        let open = px.sculpt_layer == Some(i);
        let title = format!("{}  {}", i + 1, l.kind.name());
        let (pressed, at) = fold_heading(ui, Id::unique(("sculpt layer", i)), &title, !open, &layer_says(&l));
        px.note(format!("sculpt:layer:{}", i + 1), at);
        if pressed { px.sculpt_layer = if open { None } else { Some(i) }; }
        if open {
            let (top, left) = (ui.cursor().top(), ui.cursor().left());
            let inner = Rect::from_min_max(egui::pos2(left + 10.0, top), egui::pos2(ui.max_rect().right(), top + 8000.0));
            let names = label_width(ui);
            ui.scope_builder(egui::UiBuilder::new().max_rect(inner), |ui| {
                ui.set_width(inner.width());
                set_label_width(ui, names - 10.0);
                layer_rows(ui, st, px, i);
            });
            set_label_width(ui, names);
            side_bar(ui, left + 3.0, top + 1.0, ui.cursor().top() - 2.0, 0.3);
            gap(ui, 2.0);
        }
    }
    // a layer of any kind, added at the end (and opened)
    let n = st.sculpt().layers.len();
    if n < MAX_LAYERS {
        gap(ui, 4.0);
        let mut names: Vec<String> = vec!["add a layer…".to_owned()];
        names.extend(KINDS.iter().map(|k| format!("{} ({})", k.1, k.2)));
        let r = row(ui, 22.0);
        if let Some(i) = select(ui, r, Id::unique("sculpt add"), &names, 0, mono(SIZE)) {
            if i > 0 {
                st.sculpt_mut().layers.push(Layer::of(KINDS[i - 1].0));
                st.sculpt_mut().on = true;
                px.sculpt_layer = Some(n);
                changed(st);
            }
        }
        px.note("sculpt:add", r);
    }
}

/// The rows of one layer: its buttons, kind, what it does and how it blends, its sliders, its mask.
fn layer_rows(ui: &mut Ui, st: &mut State, px: &mut Panels, i: usize) {
    let n = st.sculpt().layers.len();
    let Some(l) = st.sculpt().layers.get(i).cloned() else { return };
    // on, up, down, copy, remove
    gap(ui, 2.0);
    let items = [
        Item::new(if l.on { "on" } else { "off" }, l.on).tip("switch the layer off and on (it stays where it is)"),
        Item::new("up", false).tip("apply it earlier: before the layer above it"),
        Item::new("down", false).tip("apply it later: after the layer below it"),
        Item::new("copy", false).tip("a copy of it, just below it"),
        Item::new("remove", false).tip("take the layer away"),
    ];
    let run = button_run(ui, Id::unique(("sculpt acts", i)), 0.0, &items, egui::vec2(6.0, 2.0), &mono(10.5), 20.0);
    for (w, place) in ["on", "up", "down", "copy", "remove"].iter().zip(&run.places) { px.note(format!("sculpt:{}:{w}", i + 1), *place); }
    if let Some(a) = run.clicked {
        let s = st.sculpt_mut();
        match a {
            0 => s.layers[i].on = !s.layers[i].on,
            1 if i > 0 => { s.layers.swap(i, i - 1); px.sculpt_layer = Some(i - 1); }
            2 if i + 1 < n => { s.layers.swap(i, i + 1); px.sculpt_layer = Some(i + 1); }
            3 if n < MAX_LAYERS => { let c = s.layers[i].clone(); s.layers.insert(i + 1, c); px.sculpt_layer = Some(i + 1); }
            4 => { s.layers.remove(i); px.sculpt_layer = None; }
            _ => {}
        }
        changed(st);
        return;
    }

    // its kind
    let kinds: Vec<String> = KINDS.iter().map(|k| k.1.to_owned()).collect();
    if let Some(k) = pick(ui, Id::unique(("sculpt kind", i)), "kind", &kinds, l.kind.index()) {
        st.sculpt_mut().layers[i].change_kind(KINDS[k].0);
        changed(st);
        return;
    }
    let r = row(ui, 13.0);
    text_in(ui, beside_label(ui, r, 13.0), Align2::LEFT_CENTER, KINDS[l.kind.index()].2, mono(9.5), HINT);

    if l.kind == Kind::Image { picture_row(ui, st, px, i, &l); }

    let key = |what: &str| format!("sculpt:{}:{what}", i + 1);
    // how much of it acts
    let m = knob::value_row(ui, px, &key("opacity"), "opacity", l.opacity, 0.0, 1.0, 0.01);
    after(st, &m, |s, v| s.layers[i].opacity = v);

    if l.kind != Kind::Warp {
        // raise, carve or both, and how it blends with what is before it
        let items: Vec<Item> = MODES.iter().map(|(m, w, tip)| Item::new(w, *m == l.mode).tip(*tip)).collect();
        if let Some(k) = named_run(ui, Id::unique(("sculpt mode", i)), "does", &items).clicked {
            st.sculpt_mut().layers[i].mode = MODES[k].0;
            changed(st);
        }
        let items: Vec<Item> = BLENDS.iter().map(|(b, w, tip)| Item::new(w, *b == l.blend).tip(*tip)).collect();
        if let Some(k) = named_run(ui, Id::unique(("sculpt blend", i)), "blend", &items).clicked {
            st.sculpt_mut().layers[i].blend = BLENDS[k].0;
            changed(st);
        }
    }

    // its size and how far it reaches
    let size_label = if l.kind == Kind::Image { "picture size" } else { "size" };
    let m = knob::value_row(ui, px, &key("scale"), size_label, l.scale, 0.02, if l.kind == Kind::Image { 3.0 } else { 1.0 }, 0.005);
    after(st, &m, |s, v| s.layers[i].scale = v);
    let depth_label = match (l.kind, l.kind.solid(), l.mode) {
        (Kind::Warp, _, _) => "amount",
        (_, true, gf_core::sculpt::Mode::Both) => "skin",
        (_, true, _) => "reach",
        _ => "depth",
    };
    let m = knob::value_row(ui, px, &key("depth"), depth_label, l.depth, 0.0, if l.kind.solid() || l.kind == Kind::Warp { 0.4 } else { 0.1 }, 0.001);
    after(st, &m, |s, v| s.layers[i].depth = v);
    // its own two sliders
    for (j, e) in l.kind.extras().iter().enumerate() {
        let Some(e) = e else { continue };
        let now = if j == 0 { l.a } else { l.b };
        let m = knob::value_row(ui, px, &key(if j == 0 { "a" } else { "b" }), e.label, now, e.lo, e.hi, e.step);
        after(st, &m, |s, v| if j == 0 { s.layers[i].a = v } else { s.layers[i].b = v });
    }
    if l.kind != Kind::Warp {
        let m = knob::value_row(ui, px, &key("soft"), "softness", l.soft, 0.001, 0.08, 0.001);
        after(st, &m, |s, v| s.layers[i].soft = v);
    }
    let m = knob::value_row(ui, px, &key("seed"), "seed", l.seed, 1.0, 99.0, 1.0);
    after(st, &m, |s, v| s.layers[i].seed = v);
    let m = knob::value_row(ui, px, &key("stretch"), "stretch", l.stretch, 0.25, 6.0, 0.05);
    after(st, &m, |s, v| s.layers[i].stretch = v);
    let m = knob::value_row(ui, px, &key("yaw"), "turn", l.yaw, -180.0, 180.0, 1.0);
    after(st, &m, |s, v| s.layers[i].yaw = v);
    let m = knob::value_row(ui, px, &key("pitch"), "tilt", l.pitch, -90.0, 90.0, 1.0);
    after(st, &m, |s, v| s.layers[i].pitch = v);
    let m = knob::value_row(ui, px, &key("evolve"), "evolve", l.evolve, 0.0, 4.0, 0.05);
    after(st, &m, |s, v| s.layers[i].evolve = v);

    // its mask
    let masks: Vec<String> = MASKS.iter().map(|m| m.1.to_owned()).collect();
    let now = MASKS.iter().position(|m| m.0 == l.mask).unwrap_or(0);
    if let Some(k) = pick(ui, Id::unique(("sculpt mask", i)), "mask", &masks, now) {
        st.sculpt_mut().layers[i].mask = MASKS[k].0;
        changed(st);
    }
    let r = row(ui, 13.0);
    text_in(ui, beside_label(ui, r, 13.0), Align2::LEFT_CENTER, MASKS[now].2, mono(9.5), HINT);
    if matches!(l.mask, Mask::Height | Mask::Radial) {
        let (lo, hi) = if l.mask == Mask::Height { (-1.5, 1.5) } else { (0.0, 1.5) };
        let m = knob::value_row(ui, px, &key("mask at"), "from", l.mask_at, lo, hi, 0.01);
        after(st, &m, |s, v| s.layers[i].mask_at = v);
        let m = knob::value_row(ui, px, &key("mask width"), "edge", l.mask_width, 0.01, 1.0, 0.01);
        after(st, &m, |s, v| s.layers[i].mask_width = v);
    }
    if l.mask != Mask::None {
        let items = [Item::new("as it is", !l.mask_invert), Item::new("turned round", l.mask_invert)];
        if let Some(k) = named_run(ui, Id::unique(("sculpt invert", i)), "mask is", &items).clicked {
            st.sculpt_mut().layers[i].mask_invert = k == 1;
            changed(st);
        }
    }
    gap(ui, 2.0);
}

/// An image layer's picture: its file, and a button to choose one.
fn picture_row(ui: &mut Ui, st: &mut State, px: &mut Panels, i: usize, l: &Layer) {
    let r = row(ui, 22.0);
    label(ui, r, "picture");
    let slot = beside_label(ui, r, 22.0);
    let b = Rect::from_min_max(Pos2::new(slot.right() - 62.0, slot.top()), slot.max);
    let words = Rect::from_min_max(slot.min, Pos2::new(b.left() - 6.0, slot.bottom()));
    let name = if l.image.is_empty() { "none chosen (PNG)".to_owned() } else { tail(ui, &l.image, &mono(10.0), words.width()) };
    text_in(ui, words, Align2::LEFT_CENTER, &name, mono(10.0), if l.image.is_empty() { HINT } else { BRIGHT });
    if super::widgets::button(ui, b, Id::unique(("sculpt picture", i)), "choose…", false, mono(10.5)).on_hover_text("a PNG picture: its light parts raise or carve the surface").clicked() {
        px.requests.push(Request::LoadSculptPicture(i));
    }
    px.note(format!("sculpt:{}:picture", i + 1), b);
    let _ = st;
}

/// The cleanups of the whole stack.
fn cleanup(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    section(ui, "cleanup", "the whole stack");
    let s = st.sculpt().clone();
    let m = knob::value_row(ui, px, "sculpt:round", "round edges", s.round, 0.0, 0.05, 0.001);
    after(st, &m, |s, v| s.round = v);
    let m = knob::value_row(ui, px, "sculpt:wall", "thinnest wall", s.wall, 0.0, 0.1, 0.001);
    after(st, &m, |s, v| s.wall = v);
    let m = knob::value_row(ui, px, "sculpt:crumbs", "drop crumbs", s.crumbs, 0.0, 0.3, 0.005);
    after(st, &m, |s, v| s.crumbs = v);
    let items = [Item::new("on", s.prefilter).tip("detail finer than about two pixels at this view is left out, so it cannot turn into speckle; made again when the zoom or the picture size changes by about 40%"), Item::new("off", !s.prefilter).tip("every detail is made, however small it is on the screen")];
    if let Some(k) = named_run(ui, Id::unique("sculpt prefilter"), "pixel filter", &items).clicked {
        st.sculpt_mut().prefilter = k == 0;
        changed(st);
    }
    note(ui, "round edges: every edge rounded to this radius. thinnest wall: anything thinner goes. drop crumbs: pieces left loose, smaller than this across, are taken away (the largest piece always stays).");
}
