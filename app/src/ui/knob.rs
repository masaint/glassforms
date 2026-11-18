//! One slider's row: its name, the slider, its value, and (if it can move smoothly) a wave button. With
//! the wave lit the knob travels by itself while the animation plays, and the slider turns into a range:
//! two handles for the stops it travels between, the stretch between them, and a mark at the present
//! value. Under it then: the two stops as numbers, how long the trip there and back takes, and how it eases.

use eframe::egui::{self, Align2, Id, Pos2, Rect, Sense, Ui};
use gf_core::anim::{EASES, EASE_AT};
use gf_core::state::{Knob, KnobRef, State};

use super::theme::*;
use super::widgets::*;
use super::{digits_of, Panels};

/// The four columns of a slider's row.
struct Cols {
    label: Rect,
    slot: Rect,
    value: Rect,
    wave: Rect,
}
fn cols(ui: &Ui, r: Rect) -> Cols {
    let y = r.center().y;
    let wave = Rect::from_min_size(Pos2::new(r.right() - COL_WAVE, y - 8.5), egui::vec2(COL_WAVE, 17.0));
    let value = Rect::from_min_max(Pos2::new(wave.left() - COL_GAP - COL_VALUE, r.top()), Pos2::new(wave.left() - COL_GAP, r.bottom()));
    let label = Rect::from_min_max(r.min, Pos2::new(r.left() + label_width(ui), r.bottom()));
    let slot = Rect::from_min_max(Pos2::new(label.right() + COL_GAP, y - 6.0), Pos2::new(value.left() - COL_GAP, y + 6.0));
    Cols { label, slot, value, wave }
}

/// Write a name in its column; one too long for it even set smaller is cut short and shown whole on hover.
/// `tip`: what else hovering over the name says (also while the control is greyed).
pub fn name(ui: &mut Ui, rect: Rect, id: Id, text: &str, size: f32, tip: &str) {
    let cut = name_in(ui, rect, text, size);
    let says = match (cut, tip.is_empty()) {
        (false, true) => return,
        (true, true) => text.to_owned(),
        (true, false) => format!("{text}\n{tip}"),
        (false, false) => tip.to_owned(),
    };
    ui.interact(rect, id, Sense::hover()).on_hover_text(says.as_str()).on_disabled_hover_text(says.as_str());
}

/// Move the stops of a knob's track. The value is kept between the new stops, and the trip carries on
/// from wherever the value now is.
pub fn set_stops(st: &mut State, k: &Knob, from: f64, to: f64) {
    let hard = gf_core::limits::knob(k);
    let (from, to) = (gf_core::limits::hold(from, hard), gf_core::limits::hold(to, hard));
    let Some(tr) = st.tracks.get_mut(&k.key) else { return };
    tr.from = from;
    tr.to = to;
    let now = st.get(&k.r);
    let kept = now.clamp(from.min(to), from.max(to));
    if kept != now {
        st.set(&k.r, kept);
        st.after_user(k, true);
    }
    st.rephase(k);
}

/// Set how long a knob's trip there and back takes.
pub fn set_period(st: &mut State, k: &Knob, seconds: f64) {
    if let Some(tr) = st.tracks.get_mut(&k.key) {
        tr.period = seconds;
        st.rephase(k);
    }
}

/// Set how one knob eases: `ease` -1 = as all the others, else an index into the easing types;
/// `at`: start, finish or both.
pub fn set_ease(st: &mut State, k: &Knob, ease: i32, at: u8) {
    if let Some(tr) = st.tracks.get_mut(&k.key) {
        tr.ease = ease;
        tr.at = at;
        st.rephase(k);
    }
}

/// The value at the end of a slider's row (`value_text`): what a drag or typing made it, held to `hard`.
#[allow(clippy::too_many_arguments)]
fn value_end(ui: &mut Ui, rect: Rect, id: Id, out: &mut Moved, value: f64, lo: f64, hi: f64, step: f64, hard: (f64, f64), size: f32) {
    let shown = out.value.unwrap_or(value);
    let digits = if out.finer { digits_of(step) + 2 } else { digits_for(shown, digits_of(step)) };
    if let Some(v) = value_text(ui, rect, id.with("value"), shown, digits, digits_of(step), lo, hi, size) {
        out.value = Some(gf_core::limits::hold(v, (hard.0.min(lo), hard.1.max(hi))));
        out.released = true;
    }
}

/// A plain slider's row: a name, a slider, the value (click it to type one). Returns what the user did.
pub fn plain_row(ui: &mut Ui, id: Id, label: &str, value: f64, lo: f64, hi: f64, step: f64) -> Moved {
    let r = row(ui, ROW_H);
    let c = cols(ui, r);
    name(ui, c.label, id.with("name"), label, SIZE, "");
    let hard = gf_core::limits::widen(lo, hi);
    let mut out = slider_to(ui, c.slot, id, value, lo, hi, step, pal().accent, hard);
    value_end(ui, c.value, id, &mut out, value, lo, hi, step, hard, SIZE);
    out
}

/// The row of a value that is not one of the knobs (a light's angle, say): a name, a slider, the value.
/// `key` is what a test knows the slider by. Returns what the user did to it.
pub fn value_row(ui: &mut Ui, px: &mut Panels, key: &str, label: &str, value: f64, lo: f64, hi: f64, step: f64) -> Moved {
    let r = row(ui, ROW_H);
    let c = cols(ui, r);
    let id = Id::unique(("value", px.place, key));
    px.note(format!("{}slot:{key}", px.whose()), c.slot);
    name(ui, c.label, id.with("name"), label, SIZE, "");
    let hard = gf_core::limits::widen(lo, hi);
    let mut out = slider_to(ui, c.slot, id, value, lo, hi, step, pal().accent, hard);
    value_end(ui, c.value, id, &mut out, value, lo, hi, step, hard, SIZE);
    out
}

/// The row of a hue: its slider is the hues themselves, and at its end is the colour that the hue and
/// the strength make.
pub fn hue_row(ui: &mut Ui, px: &mut Panels, key: &str, label: &str, hue: f64, strength: f64) -> Moved {
    let r = row(ui, ROW_H);
    let c = cols(ui, r);
    let id = Id::unique(("value", px.place, key));
    px.note(format!("{}slot:{key}", px.whose()), c.slot);
    name(ui, c.label, id.with("name"), label, SIZE, "");
    let out = hue_slider(ui, c.slot, id, hue, strength);
    let shown = out.value.unwrap_or(hue);
    text_in(ui, c.value, Align2::RIGHT_CENTER, &format!("{shown:.0}"), mono(SIZE), pal().bright);
    let swatch = Rect::from_center_size(c.wave.center(), egui::vec2(12.0, 12.0));
    ui.painter().rect(swatch, 3, shown_colour(hue_rgb(shown, strength)), egui::Stroke::new(1.0, white(0.25)), egui::StrokeKind::Outside);
    out
}

/// The knob of a setting, by the setting's name: the engine's own if it has one (which can then be
/// animated), otherwise a plain one with this label and range. None if there is no such setting.
pub fn setting_knob(st: &State, name: &'static str, label: &str, lo: f64, hi: f64, step: f64) -> Option<Knob> {
    st.p.get(name)?;
    let own = State::engine_knobs().into_iter().find(|k| k.id == name);
    Some(own.unwrap_or_else(|| Knob { key: format!("P.{name}"), id: name.into(), label: label.into(), lo, hi, step, shape: false, wave: false, r: KnobRef::P(name) }))
}

/// What a knob's row may have beyond the knob itself.
#[derive(Clone, Copy, Default)]
pub struct Extra<'a> {
    /// what hovering over its name says
    pub tip: &'a str,
    /// the mark of something costly beside its name
    pub costly: bool,
    /// it counts in whole numbers: a button at each end of the slider steps it by one
    pub stepped: bool,
    /// the value in words, in place of the number (a focal length in millimetres for a slider that zooms
    /// by doublings, an f-stop for one that moves by thirds of a stop)
    pub shown: &'a str,
}

/// The row of one knob, and under it the rows of its track when it is animated.
pub fn knob_row(ui: &mut Ui, st: &mut State, k: &Knob, px: &mut Panels) {
    knob_row_with(ui, st, k, px, Extra::default());
}

/// How wide the buttons are that step a whole number by one.
const STEP_W: f32 = 15.0;

/// The same, with what else the row may have.
pub fn knob_row_with(ui: &mut Ui, st: &mut State, k: &Knob, px: &mut Panels, extra: Extra) {
    let id = Id::unique(("knob", px.place, &k.key));
    let digits = digits_of(k.step);
    let r = row(ui, ROW_H);
    let mut c = cols(ui, r);
    if !extra.shown.is_empty() {
        // words may want more room than a number: the slot gives way to them
        let w = text_width(ui, extra.shown, &mono(SIZE)).ceil();
        if w > c.value.width() {
            c.value.min.x = c.value.right() - w;
            c.slot.max.x = (c.value.left() - COL_GAP).max(c.slot.left() + 30.0);
        }
    }
    if extra.costly {
        // the mark stands behind the name, which keeps clear of it
        c.label.max.x -= MARK_ROOM;
        mark_after(ui, c.label, &k.label);
    }
    name(ui, c.label, id.with("name"), &k.label, SIZE, extra.tip);
    if extra.stepped {
        // one less, the slider, one more
        let (less, more) = (Rect::from_min_size(c.slot.min - egui::vec2(0.0, 2.0), egui::vec2(STEP_W, 16.0)), Rect::from_min_size(Pos2::new(c.slot.right() - STEP_W, c.slot.top() - 2.0), egui::vec2(STEP_W, 16.0)));
        c.slot = Rect::from_min_max(Pos2::new(less.right() + 5.0, c.slot.top()), Pos2::new(more.left() - 5.0, c.slot.bottom()));
        px.note(format!("{}less:{}", px.whose(), k.key), less);
        px.note(format!("{}more:{}", px.whose(), k.key), more);
        let now = st.get(&k.r);
        for (place, by, what) in [(less, -1.0, "−"), (more, 1.0, "+")] {
            if button(ui, place, id.with(what), what, false, mono(SIZE)).clicked() {
                st.user_set(k, (now + by).clamp(k.lo, k.hi), false);
            }
        }
    }
    px.note(format!("{}slot:{}", px.whose(), k.key), c.slot);
    px.note(format!("{}wave:{}", px.whose(), k.key), c.wave);
    let value = st.get(&k.r);
    let hard = gf_core::limits::knob(k);
    let track = st.tracks.get(&k.key).filter(|t| t.on).cloned();
    let mut finer = false;
    match &track {
        Some(tr) => {
            let out = range(ui, c.slot, id.with("range"), tr.from, tr.to, value, k.lo, k.hi, k.step, digits, hard);
            if let Some((from, to)) = out.moved { set_stops(st, k, from, to); }
            if out.released { st.after_user(k, false); }
        }
        None => {
            let out = slider_to(ui, c.slot, id.with("slider"), value, k.lo, k.hi, k.step, pal().accent, hard);
            finer = out.finer;
            if let Some(v) = out.value { st.user_set(k, v, true); }
            if out.released { st.after_user(k, false); }
        }
    }
    if extra.shown.is_empty() {
        // (typed: kept at once, as a slider let go is; an animated knob's value is where its trip has
        // got to, so typing moves it there and its trip carries on from it)
        let now = st.get(&k.r);
        let shown_digits = if finer { digits + 2 } else { digits_for(now, digits) };
        if let Some(v) = value_text(ui, c.value, id.with("value"), now, shown_digits, digits, k.lo, k.hi, SIZE) {
            st.user_set(k, v, false);
            if track.is_some() { st.rephase(k); }
        }
    } else {
        // (a value in words, a focal length for a slider that zooms by doublings: typed as the slider's own
        // number is not what it shows, so it is not typed into; past the ends it still shows)
        let past = value < k.lo - 1e-12 || value > k.hi + 1e-12;
        text_in(ui, c.value, Align2::RIGHT_CENTER, extra.shown, mono(SIZE), if past { pal().past } else { pal().bright });
    }
    if k.wave {
        let lit = track.is_some();
        let resp = wave_button(ui, c.wave, id.with("wave"), lit).on_hover_text("let this slider travel by itself while the animation plays");
        if resp.clicked() { st.toggle_wave(k); }
    }
    if st.tracks.get(&k.key).is_some_and(|t| t.on) {
        track_rows(ui, st, k, id, digits);
    }
}

/// How far the rows under an animated knob stand in from the panel's edge. Their names take the rest of
/// the width a knob's own name has, so that their sliders line up with the one above.
const TRACK_INSET: f32 = 14.0;

/// Under an animated knob: the stops it travels between, how long the trip takes, and how it eases.
fn track_rows(ui: &mut Ui, st: &mut State, k: &Knob, id: Id, digits: usize) {
    let Some(tr) = st.tracks.get(&k.key).cloned() else { return };
    gap(ui, 1.0);
    let top = ui.cursor().top();
    let left = ui.cursor().left();
    let names = label_width(ui);
    let part = |r: Rect| -> (Rect, Rect) {
        let label = Rect::from_min_max(Pos2::new(r.left() + TRACK_INSET, r.top()), Pos2::new(r.left() + names, r.bottom()));
        (label, Rect::from_min_max(Pos2::new(label.right() + COL_GAP, r.top()), r.max))
    };

    // the two stops, in the order the trip takes them
    let (label, rest) = part(row(ui, 16.0));
    text_in(ui, label, Align2::LEFT_CENTER, "between", mono(SMALL), pal().dim);
    let stops = format!("{:.digits$} and {:.digits$}", tr.from, tr.to);
    text_in(ui, Rect::from_min_max(rest.min, Pos2::new(rest.right() - COL_WAVE - COL_GAP, rest.bottom())), Align2::RIGHT_CENTER, &stops, mono(SMALL), pal().button);

    let r = row(ui, 17.0);
    let c = cols(ui, r);
    text_in(ui, part(Rect::from_min_max(Pos2::new(left, c.label.top()), c.label.max)).0, Align2::LEFT_CENTER, "every s", mono(SMALL), pal().dim);
    // (a trip of no time at all would divide by nothing; one of an hour is as slow as anything wants)
    let hard = (0.1, 3600.0);
    let mut out = slider_to(ui, c.slot, id.with("every"), tr.period, 1.0, 120.0, 0.5, pal().accent_mini, hard);
    let shown = out.value.unwrap_or(tr.period);
    if let Some(v) = value_text(ui, c.value, id.with("every value"), shown, digits_for(shown, 2), 2, 1.0, 120.0, SMALL) { out.value = Some(gf_core::limits::hold(v, hard)); }
    if let Some(v) = out.value { set_period(st, k, gf_core::limits::hold(v, hard)); }

    let (label, rest) = part(row(ui, 22.0));
    text_in(ui, label, Align2::LEFT_CENTER, "ease", mono(SMALL), pal().dim);
    let w = (rest.width() - COL_GAP) / 2.0;
    let place = |i: f32| Rect::from_min_size(Pos2::new(rest.left() + i * (w + COL_GAP), rest.center().y - 9.5), egui::vec2(w, 19.0));
    let kinds: Vec<String> = std::iter::once("as all".to_owned()).chain(EASES.iter().map(|e| e.to_string())).collect();
    if let Some(i) = select(ui, place(0.0), id.with("ease"), &kinds, (tr.ease + 1).max(0) as usize, mono(SMALL)) {
        set_ease(st, k, i as i32 - 1, tr.at);
    }
    // where the easing acts only means something once this knob has an easing of its own
    if tr.ease >= 0 {
        let ends: Vec<String> = EASE_AT.iter().map(|e| format!("at {e}")).collect();
        if let Some(i) = select(ui, place(1.0), id.with("at"), &ends, tr.at as usize, mono(SMALL)) {
            set_ease(st, k, tr.ease, i as u8);
        }
    }
    side_bar(ui, left + 4.0, top, ui.cursor().top(), 0.4);
    gap(ui, 5.0);
}

/// What a colour's sliders remember when the colour itself cannot say: a white has no hue, and a light
/// turned right down has no colour at all, yet their sliders should stay where they were put.
#[derive(Clone, Copy)]
struct Tint {
    hue: f64,
    strength: f64,
}

/// Three rows for a colour that is also a brightness (a light's red, green and blue): its hue, how far
/// it is from white, and how bright it is. `key` names the three for tests and for what is remembered.
/// Returns the colour as changed, if it was.
pub fn colour_rows(ui: &mut Ui, px: &mut Panels, key: &str, rgb: [f64; 3], most: f64, step: f64) -> Option<[f64; 3]> {
    let memory = Id::unique(("tint", key));
    let (hue, strength, bright) = rgb_hue(rgb);
    let kept = ui.data(|d| d.get_temp::<Tint>(memory)).unwrap_or(Tint { hue: 40.0, strength: 0.0 });
    let mut now = Tint { hue: hue.unwrap_or(kept.hue), strength: strength.unwrap_or(kept.strength) };
    let mut level = bright;
    let mut changed = false;
    if let Some(v) = hue_row(ui, px, &format!("{key}.hue"), "hue °", now.hue, now.strength).value {
        now.hue = v;
        changed = true;
    }
    if let Some(v) = value_row(ui, px, &format!("{key}.strength"), "saturation", now.strength, 0.0, 1.0, 0.01).value {
        now.strength = v;
        changed = true;
    }
    if let Some(v) = value_row(ui, px, &format!("{key}.bright"), "brightness", level, 0.0, most.max(level), step).value {
        level = v;
        changed = true;
    }
    ui.data_mut(|d| d.insert_temp(memory, now));
    changed.then(|| hue_rgb(now.hue, now.strength).map(|c| c * level))
}

/// A colour as its hue and how far it is from white, without its brightness (which something else says,
/// as a lamp's power does). The colour comes back at full brightness; None if neither was moved.
pub fn hue_rows(ui: &mut Ui, px: &mut Panels, key: &str, rgb: [f64; 3]) -> Option<[f64; 3]> {
    let memory = Id::unique(("tint", key));
    let (hue, strength, _) = rgb_hue(rgb);
    let kept = ui.data(|d| d.get_temp::<Tint>(memory)).unwrap_or(Tint { hue: 40.0, strength: 0.0 });
    let mut now = Tint { hue: hue.unwrap_or(kept.hue), strength: strength.unwrap_or(kept.strength) };
    let mut changed = false;
    if let Some(v) = hue_row(ui, px, &format!("{key}.hue"), "hue °", now.hue, now.strength).value {
        now.hue = v;
        changed = true;
    }
    if let Some(v) = value_row(ui, px, &format!("{key}.strength"), "saturation", now.strength, 0.0, 1.0, 0.01).value {
        now.strength = v;
        changed = true;
    }
    ui.data_mut(|d| d.insert_temp(memory, now));
    changed.then(|| hue_rgb(now.hue, now.strength))
}
