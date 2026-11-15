//! How the shape moves, at the foot of the right panel under the sliders that do the moving: the play
//! button and what it plays, the time slider, the speed of everything, how travelling sliders ease, the
//! spin, and the loop it all fits into.

use eframe::egui::{self, Align2, Id, Pos2, Rect, Ui};
use gf_core::anim::{js_round, EASES};
use gf_core::state::State;

use super::theme::*;
use super::widgets::*;
use super::{knob, Panels};

/// Put the animation clock at `seconds` into the loop it is in, as dragging the time slider does. Paused,
/// everything animated is shown as it is at that moment; playing, the animation carries on from there.
pub fn scrub(st: &mut State, seconds: f64) {
    let l = st.loop_plan().l;
    // stay in the loop the clock has reached, so that nothing playing freely (not tied to the loop) jumps
    st.clock = (st.clock / l).floor() * l + seconds.clamp(0.0, l);
    if !st.playing {
        let waiting = st.want.take();
        st.apply_clock(false);
        // with no slider of the shape travelling, the shape has not changed: nothing to build
        if !st.shape_moving() { st.want = waiting; }
    }
}

/// The time slider has been let go: a shape that was rebuilt quickly while it moved is built at full
/// detail, as it is at that moment of the animation.
pub fn scrub_done(st: &mut State) {
    if !st.playing && st.shape_moving() { st.apply_clock(true); }
}

/// What the loop comes to: its length, and how many times each moving thing goes round in it.
pub fn plan_text(st: &State) -> String {
    let p = st.loop_plan();
    let mut lines = vec![format!("one loop: {:.2} s", p.l)];
    for it in &p.items {
        let times = if it.turn { if it.n == 1 { " turn" } else { " turns" } } else { "×" };
        let nudged = if (it.now - it.was).abs() / it.was > 0.005 { format!(", every {:.1} → {:.1} s", it.was, it.now) } else { String::new() };
        lines.push(format!("{}: {}{times}{nudged}", it.label, it.n));
    }
    lines.join("\n")
}

/// The sliders that travel while the animation plays, by name; empty if none does.
pub fn moving_names(st: &State) -> String {
    // (with several layers a slider is named with its layer, or with the layers it joins)
    let mut names: Vec<String> = st.active().iter().map(|k| st.knob_title(k)).collect();
    if st.beads().is_some() { names.push("beads".into()); }
    names.join(", ")
}

/// What moves, in the words beside the play button.
pub fn moves_text(st: &State) -> String {
    let names = moving_names(st);
    if names.is_empty() { "light a ∿ beside a slider".into() } else { names }
}

pub fn show(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    // folded, the title still says whether it is playing, and what
    let names = moving_names(st);
    let says = match (st.playing, names.is_empty()) {
        (true, false) => format!("playing {names}"),
        (true, true) => "playing".to_owned(),
        (false, _) => names,
    };
    if !px.group(ui, "animate", "animate", &says) { return; }

    // the play button, and beside it what it sets moving
    let caption = if st.playing { "❚❚  pause" } else { "▶  animate" };
    let size = button_size(ui, caption, &mono(SIZE), egui::vec2(12.0, 4.0));
    let words = wrapped(ui, &moves_text(st), mono(SMALL), DIM, ui.available_width() - size.x - 9.0, 1.3);
    let r = row(ui, size.y.max(words.size().y));
    let place = Rect::from_min_size(Pos2::new(r.left(), r.center().y - size.y / 2.0), size);
    px.note("play", place);
    if button(ui, place, Id::unique("play"), caption, st.playing, mono(SIZE)).on_hover_text("key P").clicked() {
        st.set_playing(!st.playing);
    }
    ui.painter().galley(Pos2::new(r.left() + size.x + 9.0, r.center().y - words.size().y / 2.0), words, DIM);
    gap(ui, 5.0);

    // where in the loop the animation is: it follows the clock while playing, and can be dragged
    let length = st.loop_plan().l;
    let moved = knob::value_row(ui, px, "time", "time s", st.clock.rem_euclid(length), 0.0, length, 0.01);
    if let Some(v) = moved.value { scrub(st, v); }
    if moved.released { scrub_done(st); }
    knob::knob_row(ui, st, &State::engine_knob("speed"), px);

    // how the travelling sliders ease, unless one has an easing of its own
    gap(ui, 4.0);
    let eases: Vec<String> = EASES.iter().map(|e| e.to_string()).collect();
    if let Some(i) = pick(ui, Id::unique("ease all"), "ease all", &eases, st.p.easing().0) {
        st.p.ease = i as f64;
        st.rephase_all();
    }
    if let Some(i) = choice(ui, Id::unique("eased at"), "eased at", &["start", "finish", "both"], st.p.easing().1 as usize) {
        st.p.ease_at = i as f64;
        st.rephase_all();
    }

    // the shape turning by itself, whether or not the animation plays
    gap(ui, 4.0);
    knob::knob_row(ui, st, &State::engine_knob("spin"), px);
    if let Some(i) = choice(ui, Id::unique("spin as"), "spin as", &["tumble", "turntable"], js_round(st.p.spin_mode) as usize) {
        // the other way of spinning carries on from the orientation the object has now
        let r = st.r;
        st.p.spin_mode = i as f64;
        st.set_r(&r);
    }

    // the loop everything fits into: its length (a number, or nothing to have it worked out), whether
    // what plays here is held to it as a video will be, and what it comes to
    gap(ui, 4.0);
    let r = row(ui, 24.0);
    label(ui, r, "loop s");
    let field = beside_label(ui, r, 22.0);
    let id = Id::unique("loop length");
    px.note("loop", field);
    if !ui.memory(|m| m.has_focus(id)) {
        px.loop_text = if st.p.loop_len > 0.0 { format!("{}", st.p.loop_len) } else { String::new() };
    }
    let edit = egui::TextEdit::singleline(&mut px.loop_text).id(id).hint_text("auto").font(mono(SIZE)).text_color(BRIGHT).margin(egui::Margin::symmetric(5, 2)).align(Align2::LEFT_CENTER).desired_width(field.width());
    if ui.put(field, edit).changed() {
        let v = px.loop_text.trim().parse::<f64>().ok().filter(|v| v.is_finite() && *v > 0.0);
        st.p.loop_len = v.unwrap_or(0.0);
    }
    if let Some(i) = choice(ui, Id::unique("play it"), "play it", &["freely", "as looped"], js_round(st.p.lock) as usize) {
        st.p.lock = i as f64;
    }
    gap(ui, 3.0);
    paragraph(ui, &plan_text(st), mono(SMALL), BUTTON, LINE, 0.0);
    gap(ui, 1.0);
}
