//! The tree of formula parts (gf_core::csg), in the right panel under the shape's sliders: how the shape
//! up is traced (from its volume, or as the tree's first part), the parts one row each (a part a level
//! deeper than the row above it is in a group), and the selected part's own controls: its operation and
//! seam, its group level, its shape and sliders, where it sits.

use eframe::egui::{self, Align2, Id, Ui};
use gf_core::csg::{Op, Part, MAX_LEVEL, MAX_PARTS};
use gf_core::layers::Placement;
use gf_core::sdf::{support, Support, Trace};
use gf_core::state::{Detail, State};

use super::theme::*;
use super::widgets::*;
use super::{knob, Panels};

/// How far a row stands in for each group level.
const INDENT: f32 = 12.0;

/// What a change asks for: a quick build while a slider is dragged, a careful one when it is let go.
fn moved(st: &mut State, m: &Moved) -> Option<f64> {
    if m.value.is_some() { st.want = Some(Detail::Anim); }
    if m.released { st.want = Some(Detail::Full); }
    m.value
}

/// The shape up's way of being traced, then the parts.
pub fn show(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    trace_row(ui, st, px);
    parts(ui, st, px);
}

/// How the shape up is traced: from its baked volume, or as the first part of the tree of formulas. A
/// shape with no formula says why it has only the volume; one in the tree says when the volume is read
/// after all.
fn trace_row(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let now = st.shape().trace;
    match support(st.family(), st.preset()) {
        Support::VolumeOnly(why) => {
            // the same control as for every shape, so it is found in the same place: csg is there but greyed
            // (it cannot be chosen), and why is said under it in full, not cut off
            let items = [
                Item::new("volume", true).tip("the shape baked into a grid of distances, which the tracer reads: as fine as its voxels"),
                Item::new("csg", false).tip(format!("not for this shape: {why}")),
            ];
            let run = ui.add_enabled_ui(false, |ui| named_run(ui, Id::unique("trace"), "traced from", &items)).inner;
            for (w, place) in ["volume", "csg"].iter().zip(&run.places) { px.note(format!("trace:{w}"), *place); }
            gap(ui, 2.0);
            paragraph(ui, &format!("csg is only for shapes with a formula: {why}"), mono(SMALL), FAINT, LINE, 9.0);
        }
        Support::Formula { .. } => {
            let items = [
                Item::new("volume", now == Trace::Volume).tip("the shape baked into a grid of distances, which the tracer reads: as fine as its voxels"),
                Item::new("csg", now == Trace::Csg).tip("the shape's formula, worked out by the tracer: exact edges at any zoom, and the first part of the tree below"),
            ];
            let run = named_run(ui, Id::unique("trace"), "traced from", &items);
            for (w, place) in ["volume", "csg"].iter().zip(&run.places) { px.note(format!("trace:{w}"), *place); }
            if let Some(i) = run.clicked { st.apply_arg("trace", if i == 0 { "volume" } else { "csg" }); }
            if now == Trace::Csg {
                let why = if st.layer_count() > 1 { Some("there are several layers") }
                    else if st.layer_sculpt(0).is_some_and(|s| !s.is_empty()) { Some("the surface studio is on") }
                    else if gf_core::material::bubbles(&st.p).is_some() { Some("the material has bubbles") }
                    else { None };
                if let Some(why) = why {
                    gap(ui, 2.0);
                    paragraph(ui, &format!("reads the volume while {why}"), mono(SMALL), FAINT, LINE, 0.0);
                }
            }
        }
    }
    gap(ui, 4.0);
}

/// A part's name in its row.
fn title(st: &State, p: &Part) -> String {
    let shape = st.families.iter().find(|f| f.id == p.family).and_then(|f| f.presets.iter().find(|q| q.id == p.preset)).map_or(p.preset.as_str(), |q| q.name);
    let op = match p.op { Op::Union => "+", Op::Subtract => "−", Op::Intersect => "∩" };
    format!("{op} {}", shape.to_lowercase())
}

fn parts(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let n = st.csg.parts.len();
    if n == 0 && st.layer_count() > 1 { return; }
    let r = row(ui, 14.0);
    text_in(ui, r, Align2::LEFT_CENTER, "csg parts", mono(10.0), FAINT);
    if st.layer_count() > 1 {
        text_in(ui, r, Align2::RIGHT_CENTER, "off while there are several layers", mono(10.0), HINT);
    } else if st.shape().trace == Trace::Volume && n > 0 {
        text_in(ui, r, Align2::RIGHT_CENTER, "joined to the shape up", mono(10.0), HINT);
    }
    gap(ui, 3.0);
    if n > 0 {
        // how the parts are traced: from their formulas, or (faster with many) from the volume they are measured into
        let coarse = st.csg.coarse;
        let items = [
            Item::new("exact", !coarse).tip("the parts' formulas, worked out by the tracer: sharp at any zoom, slower the more parts there are"),
            Item::new("volume", coarse).tip("the parts as the volume holds them: as fine as its voxels, about as fast with fifty parts as with one"),
        ];
        let run = named_run(ui, Id::unique("csg coarse"), "parts traced", &items);
        for (w, place) in ["exact", "volume"].iter().zip(&run.places) { px.note(format!("csg traced:{w}"), *place); }
        if let Some(i) = run.clicked { st.csg.coarse = i == 1; st.want = Some(Detail::Full); }
        gap(ui, 3.0);
    }
    // The whole shape first: the object that places the tree in the scene. Selected, the handles move all of
    // it; a part selected below is moved within it. (Two levels of editing: what the shape is made of, and
    // where the shape stands.)
    if n > 0 {
        let chosen = st.csg.whole;
        let items = [Item::new("whole shape", chosen).tip("the tree as one object: moved, turned and sized with its parts")];
        let run = button_run(ui, Id::unique("csg whole"), 0.0, &items, egui::vec2(8.0, 3.0), &mono(SIZE), 0.0);
        if let Some(r) = run.places.first() { px.note("csg whole", *r); }
        if run.clicked.is_some() {
            st.csg.whole = !chosen;
            if !chosen { st.csg.sel = None; px.handles.pick(None); }
        }
        if st.csg.whole { gap(ui, 3.0); placement_rows(ui, st, None); gap(ui, 6.0); }
    }
    // a row for each part: pressing one selects it (pressing it again closes it)
    for i in 0..n {
        let level = st.csg.parts[i].level as f32;
        let text = title(st, &st.csg.parts[i]);
        let chosen = st.csg.sel == Some(i);
        let items = [Item::new(&text, chosen)];
        let run = button_run(ui, Id::unique(("csg part", i)), level * INDENT, &items, egui::vec2(8.0, 3.0), &mono(SIZE), 0.0);
        if let Some(r) = run.places.first() { px.note(format!("csg:{}", i + 1), *r); }
        if run.clicked.is_some() {
            st.csg.sel = if chosen { None } else { Some(i) };
            if !chosen { st.csg.whole = false; }
            // (a part selected is what the handles over the picture move)
            if !chosen { px.handles.pick(None); }
        }
        if chosen { gap(ui, 3.0); selected(ui, st, px, i); gap(ui, 6.0); }
    }
    if n < MAX_PARTS {
        let (resp, at) = wide_button(ui, Id::unique("csg add"), "add part", false);
        px.note("csg add", at);
        if resp.on_hover_text("a part of the shape up's kind (a ball when the shape up has no formula), beside the others").clicked() {
            add(st, px);
        }
    }
    gap(ui, 4.0);
}

/// A new part: of the shape up's kind with its sliders, when it is a formula, or a ball; a little off the
/// middle, and selected.
fn add(st: &mut State, px: &mut Panels) {
    let n = st.csg.parts.len();
    let place = Placement { pos: [0.35 * ((n % 3) as f64 - 1.0), 0.3 * ((n / 3 % 3) as f64 - 1.0), 0.0], rot: [0.0; 3], size: 0.5 };
    let part = match support(st.family(), st.preset()) {
        Support::Formula { .. } => Part::of(st.family(), st.preset(), &st.shape().vals, st.shape().puff, place),
        Support::VolumeOnly(_) => Part { place, ..Default::default() },
    };
    st.csg.parts.push(part);
    st.csg.sel = Some(n);
    px.handles.pick(None);
    st.want = Some(Detail::Full);
}

/// The selected part's controls.
fn selected(ui: &mut Ui, st: &mut State, px: &mut Panels, i: usize) {
    // how it joins what is above it
    let op = st.csg.parts[i].op;
    let items: Vec<Item> = Op::ALL.iter().map(|o| Item::new(o.id(), *o == op)).collect();
    let run = named_run(ui, Id::unique(("csg op", i)), "joins by", &items);
    for (o, place) in Op::ALL.iter().zip(&run.places) { px.note(format!("csg op:{}", o.id()), *place); }
    if let Some(k) = run.clicked { st.csg.parts[i].op = Op::ALL[k]; st.want = Some(Detail::Full); }
    // the seam's radius: in the scene's units, or as a share of the part's size (which keeps a seam in
    // proportion when the part is sized)
    let (smooth, rel) = (st.csg.parts[i].smooth, st.csg.parts[i].smooth_rel);
    let label = if rel { "seam, share of size" } else { "seam radius" };
    if let Some(v) = moved(st, &knob::plain_row(ui, Id::unique(("csg smooth", i)), label, smooth, 0.0, if rel { 0.5 } else { 0.3 }, 0.002)) { st.csg.parts[i].smooth = v; }
    let items = [Item::new("absolute", !rel).tip("the seam's radius in the scene's units"), Item::new("relative", rel).tip("the seam's radius as a share of the part's size")];
    let run = named_run(ui, Id::unique(("csg seam units", i)), "seam measured", &items);
    if let Some(k) = run.clicked {
        let part = &mut st.csg.parts[i];
        let size = part.place.size.abs().max(1e-3);
        // (the seam keeps its size on the screen: the number is turned to the other kind)
        if (k == 1) != part.smooth_rel { part.smooth = if k == 1 { part.smooth / size } else { part.smooth * size }; }
        part.smooth_rel = k == 1;
        st.want = Some(Detail::Full);
    }
    // its group: a part a level deeper than the one above it opens a group, joined to what is above it
    // by the group's first operation
    let level = st.csg.parts[i].level;
    let words: Vec<String> = (0..=MAX_LEVEL).map(|l| l.to_string()).collect();
    let items: Vec<Item> = words.iter().enumerate().map(|(l, w)| Item::new(w, l as u8 == level).tip("how deep in groups: a part deeper than the one above it starts a group")).collect();
    let run = named_run(ui, Id::unique(("csg level", i)), "group level", &items);
    if let Some(l) = run.clicked { st.csg.parts[i].level = l as u8; st.want = Some(Detail::Full); }

    // its shape: any shape with a formula
    let now = title(st, &st.csg.parts[i]);
    let mut pick = None;
    egui::ComboBox::from_id_salt(("csg shape", i)).width(ui.available_width() - 8.0).selected_text(now.trim_start_matches(['+', '−', '∩', ' '])).show_ui(ui, |ui| {
        for (fi, f) in st.families.iter().enumerate() {
            for (pi, p) in f.presets.iter().enumerate() {
                if !matches!(support(f, p), Support::Formula { .. }) { continue; }
                let on = st.csg.parts[i].family == f.id && st.csg.parts[i].preset == p.id;
                if ui.selectable_label(on, format!("{} · {}", f.name.to_lowercase(), p.name.to_lowercase())).clicked() { pick = Some((fi, pi)); }
            }
        }
    });
    if let Some((fi, pi)) = pick {
        let part = &mut st.csg.parts[i];
        part.family = st.families[fi].id.to_string();
        part.preset = st.families[fi].presets[pi].id.to_string();
        part.values.clear();
        st.want = Some(Detail::Full);
    }
    // its sliders
    let defs: Vec<(String, String, f64, f64, f64, f64)> = st.families.iter().find(|f| f.id == st.csg.parts[i].family)
        .and_then(|f| f.presets.iter().find(|p| p.id == st.csg.parts[i].preset))
        .map(|p| p.params.iter().map(|d| (d.id.to_string(), d.label.to_string(), d.value, d.lo, d.hi, d.step)).collect())
        .unwrap_or_default();
    for (id, label, value, lo, hi, step) in defs {
        let v = st.csg.parts[i].values.get(&id).copied().unwrap_or(value);
        if let Some(v) = moved(st, &knob::plain_row(ui, Id::unique(("csg value", i, id.as_str())), &label, v, lo, hi, step)) { st.csg.parts[i].values.insert(id, v); }
    }
    let puff = st.csg.parts[i].puff;
    if let Some(v) = moved(st, &knob::plain_row(ui, Id::unique(("csg puff", i)), "inflate", puff, -0.06, 0.06, 0.001)) { st.csg.parts[i].puff = v; }
    // where it sits, within the whole shape
    gap(ui, 3.0);
    placement_rows(ui, st, Some(i));
    let (resp, at) = wide_button(ui, Id::unique(("csg remove", i)), "remove part", false);
    px.note(format!("csg remove:{}", i + 1), at);
    if resp.clicked() {
        st.csg.parts.remove(i);
        st.csg.sel = None;
        st.want = Some(Detail::Full);
    }
}

/// Where a part sits within the whole shape (`Some(i)`), or where the whole shape stands (`None`): x, y, z,
/// three turns and a size (uniform: a distance in a part's frame times its size is the scene's; a shape
/// stretched one way only would have no true distance, and the tracer would have to step short).
fn placement_rows(ui: &mut Ui, st: &mut State, part: Option<usize>) {
    let place = match part { Some(i) => st.csg.parts[i].place, None => st.csg.place };
    let rows: [(&str, f64, f64, f64, f64); 7] = [
        ("x", place.pos[0], -1.4, 1.4, 0.005), ("y", place.pos[1], -1.4, 1.4, 0.005), ("z", place.pos[2], -1.4, 1.4, 0.005),
        ("turn x", place.rot[0], -180.0, 180.0, 0.5), ("turn y", place.rot[1], -180.0, 180.0, 0.5), ("turn z", place.rot[2], -180.0, 180.0, 0.5),
        ("size", place.size, 0.05, 2.0, 0.005),
    ];
    for (c, (label, v, lo, hi, step)) in rows.into_iter().enumerate() {
        let id = Id::unique(("csg place", part.map_or(usize::MAX, |i| i), c));
        if let Some(v) = moved(st, &knob::plain_row(ui, id, label, v, lo, hi, step)) {
            let p = match part { Some(i) => &mut st.csg.parts[i].place, None => &mut st.csg.place };
            match c { 0..=2 => p.pos[c] = v, 3..=5 => p.rot[c - 3] = v, _ => p.size = v }
        }
    }
}
