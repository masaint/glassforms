//! The right panel: the shape and how it moves. It is made of parts, one under the other. At its head the
//! layers and how they are joined (see `layers`; one layer is one shape, as ever). Then the parts of the
//! selected layer's shape: the families, the family's shapes, the mathematics of the one chosen, its own
//! sliders, where it is placed. At the foot, what is not any one shape's: the animation (see `animate`).
//! (How finely the shape is built is with the rest of the quality, in the left panel's output.)

use eframe::egui::{self, Align2, Id, Pos2, Rect, Sense, Ui};
use gf_core::presets::Kind;
use gf_core::state::{Knob, State, RES_MAX};

use super::theme::*;
use super::widgets::*;
use super::{animate, knob, layers, Panels, Request};

/// Voxels per side offered for the still shape, and for the shape while it animates.
pub const DETAIL_STILL: [u32; 5] = [384, 512, 768, 1024, RES_MAX];
pub const DETAIL_MOVING: [u32; 5] = [192, 256, 320, 384, 512];
/// How far the equations stand in from the bar at their left.
const INSET: f32 = 9.0;
/// Where the detail buttons begin, leaving room for their name.
const DETAIL_INDENT: f32 = 68.0;
/// The fewest lines of buttons the list of shapes keeps when the panel is short of room (one layer; with
/// several, one line).
const LIST_LEAST: f32 = 3.0;

/// A whole number with commas between its thousands.
pub fn with_commas(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 { out.push(','); }
        out.push(c);
    }
    out
}

/// What the last build of the selected layer's shape came to: how much was built (how long it took is in
/// the line at the foot of the window). `joined`: the size of the volume the layers were combined into.
pub fn built_text(st: &State, joined: Option<(u32, f64)>) -> String {
    let m = &st.shape().meta;
    let mut s = String::new();
    if m.segments > 0 { s += &format!("{} segments · ", with_commas(m.segments)); }
    s += &format!("{}³ voxels", m.res);
    if let Some((res, _)) = joined { s += &format!("\nall layers together: {res}³ voxels"); }
    s
}

/// A number of bytes of video memory, in words.
fn memory(bytes: u64) -> String {
    let mb = bytes as f64 / 1048576.0;
    if mb >= 1000.0 { format!("{:.1} GB", mb / 1024.0) } else { format!("{mb:.0} MB") }
}

pub fn show(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    if px.rendering { ui.disable(); }
    set_label_width(ui, COL_LABEL_WIDE);
    let top = ui.cursor().top();

    // the layers (which is selected decides what everything below shows), then the selected layer's
    // shape and where it is placed, then how the layers are joined
    layers::strip(ui, st, px);
    families(ui, st, px);
    let list = shapes(ui, st, px);
    maths(ui, st, px);
    sliders(ui, st, px);
    layers::placement(ui, st, px);
    layers::joins(ui, st, px);
    // everything that moves
    animate::show(ui, st, px);

    // what everything but the list of shapes came to: next frame the list is given the height that is left
    let rest = ui.cursor().top() - top - list;
    if (rest - px.right_rest).abs() > 0.5 {
        px.right_rest = rest;
        ui.ctx().request_repaint();
    }
}

/// The families: a button each. However many there are, they fill whole lines. (The buttons are as tall
/// as those of the groups of materials across the window, a little lower than a shape's: with eight
/// families there are four lines of them.)
fn families(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let items: Vec<Item> = st.families.iter().enumerate().map(|(i, f)| Item::new(f.name, i == st.fam).tip(format!("key {}", i + 1))).collect();
    let run = filled_run(ui, Id::unique("families"), &items, egui::vec2(7.0, 2.0), &mono(SIZE));
    drop(items);
    for (f, place) in st.families.iter().zip(&run.places) { px.note(format!("family:{}", f.id), *place); }
    if let Some(i) = run.clicked { st.set_scene(i); }
    gap(ui, 8.0);
}

/// The shapes of the family that is up: a run of buttons that takes what height the rest of the panel
/// leaves, and scrolls. Returns the height the list was shown at.
fn shapes(ui: &mut Ui, st: &mut State, px: &mut Panels) -> f32 {
    // how many there are, and how to step through them
    let r = row(ui, 13.0);
    let n = st.family().presets.len();
    text_in(ui, r, Align2::LEFT_CENTER, &format!("{n} shapes"), mono(10.0), HINT);
    text_in(ui, r, Align2::RIGHT_CENTER, "← → to step", mono(10.0), HINT);
    gap(ui, 5.0);

    let chosen = st.shape().preset;
    // a shape newly chosen (with the arrow keys, perhaps) is brought into view
    let moved = px.lit_shape.replace((st.fam, chosen)) != Some((st.fam, chosen));
    let (width, pad, font) = (ui.available_width(), egui::vec2(8.0, 3.0), mono(SIZE));
    // whole lines of buttons only: as many as the height left over holds
    let line = button_size(ui, "x", &font, pad).y + RUN_SPACE;
    // (with several layers the joins and the animation need the room: the list may shrink to the one line
    // that holds the chosen shape, and scrolls within its box)
    let least = if st.layer_count() > 1 { 1.0 } else { LIST_LEAST };
    let lines = ((px.room_h - px.right_rest - 1.0 + RUN_SPACE) / line).floor().max(least);
    let list = egui::ScrollArea::vertical().id_salt("shapes").max_height(lines * line - RUN_SPACE).min_scrolled_height(line - RUN_SPACE).auto_shrink([false, true]).show(ui, |ui| {
        ui.set_width(width);
        let items: Vec<Item> = st.family().presets.iter().enumerate().map(|(i, p)| Item::new(p.name, i == chosen).reveal(moved && i == chosen)).collect();
        button_run(ui, Id::unique("shapes"), 0.0, &items, pad, &font, 0.0)
    });
    // (the list is as tall as its buttons when they need less than they were allowed)
    let shown = Rect::from_min_size(list.inner_rect.min, egui::vec2(list.inner_rect.width(), list.inner_rect.height().min(list.content_size.y)));
    px.note("shapes", shown);
    for (p, place) in st.family().presets.iter().zip(&list.inner.places) {
        if shown.contains_rect(*place) { px.note(format!("shape:{}", p.id), *place); }
    }
    if let Some(i) = list.inner.clicked {
        st.choose(i);
        // the shape that stands for a file of the user's, chosen with no file yet: that asks for one
        if st.needs_file(st.stack.sel) { px.requests.push(Request::LoadModel); }
    }
    // a model is a file: any file can be loaded in place of the ones that come with the program
    if st.family().kind == Kind::Mesh {
        let (resp, at) = wide_button(ui, Id::unique("load a model"), "load a file…", false);
        px.note("model load", at);
        if resp.on_hover_text("any OBJ, STL, PLY, glTF or GLB file. Or drop one on the window: it takes this layer's place, or with shift held becomes one more layer").clicked() {
            px.requests.push(Request::LoadModel);
        }
    }
    shown.height()
}

/// What there is to say of a model, where a shape's mathematics go: its file and what it is made of, and
/// what was found on examining it (the answer to "was my file all right?"). It stays for as long as the
/// model is the shape. While a file is being read it says so instead; and if the last file given could
/// not be used, it says why.
fn model_lines(ui: &mut Ui, st: &State, px: &mut Panels) {
    let m = px.model.clone();
    let top = ui.cursor().top();
    if let Some(name) = &m.reading {
        paragraph(ui, &format!("reading {name}…"), mono(SMALL), BRIGHT, LINE, INSET);
    } else if st.needs_file(st.stack.sel) {
        paragraph(ui, "no file yet: load one, or drop one on the window", mono(SMALL), DIM, LINE, INSET);
    } else if m.known {
        paragraph(ui, &m.file, mono(SMALL), BRIGHT, LINE, INSET);
        gap(ui, 3.0);
        paragraph(ui, &m.line, sans(SMALL), TEXT, 1.4, INSET);
    }
    if !m.said.is_empty() {
        gap(ui, 3.0);
        paragraph(ui, &m.said, sans(SMALL), DIM, 1.4, INSET);
    }
    if let Some(why) = &m.error {
        gap(ui, 3.0);
        paragraph(ui, why, mono(SMALL), ERROR_TEXT, LINE, INSET);
    }
    let at = Rect::from_min_max(Pos2::new(ui.cursor().left(), top), Pos2::new(ui.cursor().left() + ui.available_width(), ui.cursor().top()));
    px.note("model lines", at);
    if !m.tip.is_empty() && m.reading.is_none() { ui.interact(at, Id::unique("model lines"), Sense::hover()).on_hover_text(m.tip.as_str()); }
}

/// The mathematics of the shape chosen, a word about it, and what the curve turned out to be (or, of a
/// model, what its file turned out to be).
fn maths(ui: &mut Ui, st: &State, px: &mut Panels) {
    let (eq, note, info) = (&st.preset().eq, st.preset().note, &st.shape().meta.info);
    let model = st.is_model();
    if !eq.is_empty() || !note.is_empty() || model {
        gap(ui, 8.0);
        let (left, top) = (ui.cursor().left(), ui.cursor().top());
        gap(ui, 6.0);
        for line in eq {
            // a line a little too long is set a little smaller rather than broken in two
            let room = ui.available_width() - INSET;
            // (and one much too long is broken, at the smallest of those sizes)
            let size = [11.5, 11.0, 10.5, 10.0].into_iter().find(|s| text_width(ui, line, &mono(*s)) <= room).unwrap_or(10.0);
            // whatever its size, a line stands as tall as the others
            paragraph(ui, line, mono(size), BRIGHT, 11.5 * LINE / size, INSET);
        }
        // the word about the shape: shown while it is the only one (with several layers the room goes to
        // the joins; hovering over the equations still says it)
        if !note.is_empty() && st.layer_count() < 2 {
            gap(ui, 4.0);
            paragraph(ui, note, sans(SMALL), FAINT, 1.4, INSET);
        }
        if !info.is_empty() {
            gap(ui, 4.0);
            paragraph(ui, info, mono(SMALL), DIM, LINE, INSET);
        }
        if model {
            gap(ui, 5.0);
            model_lines(ui, st, px);
        }
        gap(ui, 6.0);
        side_bar(ui, left, top, ui.cursor().top(), 0.45);
        if !note.is_empty() && st.layer_count() > 1 {
            let at = Rect::from_min_max(Pos2::new(left, top), Pos2::new(left + ui.available_width(), ui.cursor().top()));
            ui.interact(at, Id::unique("maths note"), Sense::hover()).on_hover_text(note);
        }
    }
    gap(ui, 8.0);
}

/// A knob that is a choice between a few named things (a model drawn as it was found, as a solid or as
/// a shell): a run of buttons, the lit one being the knob's value. Beside its name if they fit there,
/// otherwise across the panel.
fn choice_knob(ui: &mut Ui, st: &mut State, px: &mut Panels, k: &Knob, mut words: Vec<String>) {
    let now = (st.get(&k.r) - k.lo).round().max(0.0) as usize;
    // what "as found" comes to is said with it, once the model has been examined
    if st.is_model() && k.id == "mode" && px.model.known {
        if let Some(w) = words.first_mut() { *w = format!("{w} ({})", if px.model.found_shell { "shell" } else { "solid" }); }
    }
    let items: Vec<Item> = words.iter().enumerate().map(|(i, w)| Item::new(w, i == now)).collect();
    let (font, pad) = (mono(SIZE), egui::vec2(7.0, 2.0));
    let needs: f32 = items.iter().map(|it| button_size(ui, it.text, &font, pad).x).sum::<f32>() + RUN_SPACE * (items.len() as f32 - 1.0);
    let id = Id::unique(("choice knob", px.place, &k.key));
    let run = if needs <= ui.available_width() - label_width(ui) - COL_GAP {
        named_run(ui, id, &k.label, &items)
    } else {
        // (with no room for the name beside them, the buttons say on hover what they are a choice of)
        let items: Vec<Item> = items.into_iter().map(|it| it.tip(k.label.as_str())).collect();
        gap(ui, 2.0);
        let run = filled_run(ui, id, &items, pad, &font);
        gap(ui, 3.0);
        run
    };
    for (i, place) in run.places.iter().enumerate() { px.note(format!("choice:{}:{i}", k.id), *place); }
    if let Some(i) = run.clicked { st.user_set(k, k.lo + i as f64, false); }
}

/// One slider of a model. How far its detail is melted is not known until its file has been read (the
/// row is empty for that moment), and can be put back to what the model's own facets call for; the
/// thickness of its wall counts only while it is drawn as a shell.
fn model_slider(ui: &mut Ui, st: &mut State, px: &mut Panels, k: &Knob) {
    match k.id.as_str() {
        "smooth" => {
            if st.get(&k.r) < 0.0 {
                ui.scope(|ui| {
                    ui.disable();
                    let r = row(ui, ROW_H);
                    label(ui, r, &k.label);
                });
            } else {
                knob::knob_row(ui, st, k, px);
            }
            let r = row(ui, 19.0);
            let (text, font) = ("as the model", mono(SMALL));
            let w = button_size(ui, text, &font, egui::vec2(6.0, 0.0)).x;
            let at = Rect::from_min_size(Pos2::new(r.right() - COL_WAVE - COL_GAP - w, r.top()), egui::vec2(w, 17.0));
            px.note("model smooth own", at);
            let (own, facet) = (px.model.own_smooth, px.model.facet);
            let same = !px.model.known || (st.get(&k.r) - own).abs() < 0.5 * k.step;
            ui.scope(|ui| {
                if same { ui.disable(); }
                let tip = format!("back to what the model itself calls for: its facets are {facet:.3} wide, which comes to {own:.4}");
                if button(ui, at, Id::unique("smooth as the model"), text, false, font).on_hover_text(tip).clicked() {
                    // (below nothing means "not set": the engine puts the model's own amount there)
                    st.user_set(k, -1.0, false);
                }
            });
        }
        "wall" => {
            let shell = px.model.shell;
            ui.scope(|ui| {
                if !shell { ui.disable(); }
                knob::knob_row_with(ui, st, k, px, knob::Extra { tip: "the wall of a shell: it counts while the model is drawn as a shell", ..Default::default() });
            });
        }
        _ => knob::knob_row(ui, st, k, px),
    }
}

/// The shape's own sliders. (Those that are choices between named things come first, as buttons.)
fn sliders(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let knobs = st.shape_knobs();
    let model = st.is_model();
    for k in &knobs {
        if let Some(words) = st.knob_choices(k) { choice_knob(ui, st, px, k, words); }
    }
    for k in &knobs {
        if st.knob_choices(k).is_some() { continue; }
        if model { model_slider(ui, st, px, k) } else { knob::knob_row(ui, st, k, px) }
    }
    if st.family().kind == Kind::Curve {
        let pace = st.shape().bead_period;
        if let Some(v) = knob::plain_row(ui, Id::unique("bead pace"), "bead pace s", pace, 0.3, 10.0, 0.1).value {
            st.shape_mut().bead_period = v;
        }
    }
}

/// How finely the shape is built: when still, and while it animates (in the output group's quality).
pub fn detail_rows(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let (still, moving) = (st.shape().res, st.shape().anim_res);
    let r = row(ui, 14.0);
    text_in(ui, r, Align2::LEFT_CENTER, "shape detail, voxels a side", mono(10.0), FAINT);
    let name = |ui: &Ui, r: Rect, text: &str| {
        text_in(ui, Rect::from_min_size(r.min, egui::vec2(DETAIL_INDENT, r.height())), Align2::LEFT_CENTER, text, mono(SIZE), DIM);
    };
    let names: Vec<String> = DETAIL_STILL.iter().map(|r| r.to_string()).collect();
    let items: Vec<Item> = DETAIL_STILL
        .iter()
        .zip(&names)
        .map(|(&r, n)| Item::new(n, r == still).tip(format!("{r}³ voxels: {} of video memory for {}", memory(st.volumes_bytes_at(r)), if st.layer_count() > 1 { "all the layers and their joins" } else { "the shape" })))
        .collect();
    let run = button_run(ui, Id::unique("detail"), DETAIL_INDENT, &items, egui::vec2(5.0, 2.0), &mono(SIZE), 24.0);
    for (r, place) in DETAIL_STILL.iter().zip(&run.places) { px.note(format!("detail:{r}"), *place); }
    name(ui, run.rect, "still");
    // (every layer is built this finely: the volume they are combined into is as fine as the finest)
    if let Some(i) = run.clicked { st.set_detail(DETAIL_STILL[i]); }
    let names: Vec<String> = DETAIL_MOVING.iter().map(|r| r.to_string()).collect();
    let items: Vec<Item> = DETAIL_MOVING
        .iter()
        .zip(&names)
        .map(|(&r, n)| Item::new(n, r == moving).tip(format!("{r}³ voxels, rebuilt every frame while the shape animates: finer is slower")))
        .collect();
    let run = button_run(ui, Id::unique("detail moving"), DETAIL_INDENT, &items, egui::vec2(5.0, 2.0), &mono(SIZE), 24.0);
    for (r, place) in DETAIL_MOVING.iter().zip(&run.places) { px.note(format!("motion:{r}"), *place); }
    name(ui, run.rect, "in motion");
    if let Some(i) = run.clicked { st.set_anim_detail(DETAIL_MOVING[i]); }
    gap(ui, 3.0);
    paragraph(ui, &built_text(st, px.joined), mono(SMALL), DIM, LINE, 0.0);
    gap(ui, 1.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thousands_are_set_off_with_commas() {
        assert_eq!(with_commas(0), "0");
        assert_eq!(with_commas(999), "999");
        assert_eq!(with_commas(1000), "1,000");
        assert_eq!(with_commas(40700), "40,700");
        assert_eq!(with_commas(1234567), "1,234,567");
    }

    #[test]
    fn the_detail_buttons_are_sizes_that_can_be_built() {
        for r in DETAIL_STILL.iter().chain(&DETAIL_MOVING) {
            assert_eq!(gf_core::state::valid_res(*r as f64), *r);
        }
    }
}
