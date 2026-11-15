//! The camera, in a photographer's terms, as one thing: its everyday part in the picture section of the
//! left panel, and the whole of it in the camera window (key K). The window, from the top:
//! the lens (named lenses, the focal length with the field it takes in, the kind of lens, the sensor);
//! the focus (everything sharp, or the lens opened: the subject's real size, the f-stop, where it is
//! focused, how deep a range is held sharp, what that comes to, how the blur is made); where the camera
//! stands (distance, height, the picture slid across, roll); the exposure with the tone curve and, folded,
//! the grade; and folded at the foot, the lens's character (distortion, fringing, fall-off, vignette, the
//! shape of what is out of focus, tilt, bloom and grain).
//!
//! Everything here is a setting by name; the numbers a photographer reads (field of view, the distances
//! sharp, magnification) come from `gf_core::camera`, so this file only lays them out. The view (where the
//! camera stands and how the shape is turned in front of it) is one part, with one reset.

use eframe::egui::{self, Align2, Id, Pos2, Rect, Ui};
use gf_core::anim::js_round;
use gf_core::camera::{self, focal_text, length_text, stop_text, Lens, Projection, LENSES, SENSORS, SUBJECTS};
use gf_core::stage::TONES;
use gf_core::state::{Knob, State};

use super::theme::*;
use super::widgets::*;
use super::{knob, Panels};


/// What the group says beside its title while it is folded: the lens, its stop and where it is focused.
pub fn summary(st: &State, aspect: f64) -> String {
    let r = st.camera().readout(aspect);
    match r.fstop {
        Some(n) => format!("{} · f/{} · {}", focal_text(r.focal), stop_text(n), length_text(r.focus_mm)),
        None => format!("{} · everything sharp", focal_text(r.focal)),
    }
}

/// The same, shorter, for the picture section's title: "52 mm · all sharp", "85 mm · f/4.0".
pub fn brief(st: &State, aspect: f64) -> String {
    let r = st.camera().readout(aspect);
    match r.fstop {
        Some(n) => format!("{} · f/{}", focal_text(r.focal), stop_text(n)),
        None => format!("{} · all sharp", focal_text(r.focal)),
    }
}

/// A line of small words at the right of the panel, under the row they are about.
fn aside(ui: &mut Ui, text: &str, ink: egui::Color32) {
    let r = row(ui, 14.0);
    let room = Rect::from_min_max(Pos2::new(r.left() + label_width(ui) + COL_GAP, r.top()), Pos2::new(r.right() - COL_WAVE - COL_GAP, r.bottom()));
    text_in(ui, room, Align2::RIGHT_CENTER, text, mono(10.0), ink);
}

/// A named lens as its button says it: "50", "100 macro", "fisheye 15".
fn lens_word(l: &Lens) -> String {
    if l.proj.fisheye() {
        format!("fisheye {:.0}", l.focal)
    } else if l.name.contains("macro") {
        format!("{:.0} macro", l.focal)
    } else {
        format!("{:.0}", l.focal)
    }
}

/// One of the engine's knobs by name.
fn engine(name: &str) -> Knob {
    State::engine_knob(name)
}

/// One of the engine's knobs under a shorter name (the column for names is narrow inside a folded part).
fn called(name: &str, label: &str) -> Knob {
    Knob { label: label.into(), ..engine(name) }
}

/// A knob's row with its value said in words.
fn worded(ui: &mut Ui, st: &mut State, px: &mut Panels, name: &str, shown: &str, tip: &str) {
    knob::knob_row_with(ui, st, &engine(name), px, knob::Extra { shown, tip, ..Default::default() });
}

/// The camera's everyday part, in the picture section of the left panel: a named lens, the focal length,
/// all sharp or the lens open (with its f-stop), and the exposure; and the way into the camera window,
/// which has the whole of it.
pub fn everyday(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let c = st.camera();
    let r = c.readout(px.aspect);
    let words: Vec<String> = LENSES.iter().map(lens_word).collect();
    let items: Vec<Item> = LENSES
        .iter()
        .zip(&words)
        .map(|(l, w)| Item::new(w, (r.focal - l.focal).abs() < 0.5 && c.proj == l.proj).tip(format!("{}: the camera moves so that the subject keeps its size", l.name)))
        .collect();
    let run = filled_run(ui, Id::unique(("lenses", px.place)), &items, egui::vec2(5.0, 2.0), &mono(SIZE));
    for (i, place) in run.places.iter().enumerate() { px.note(format!("{}lens:{i}", px.whose()), *place); }
    if let Some(i) = run.clicked { st.lens_preset(i); }
    gap(ui, 3.0);
    let r = st.camera().readout(px.aspect);
    ui.scope(|ui| {
        if st.camera().proj == Projection::Panorama { ui.disable(); }
        worded(ui, st, px, "zoom", &focal_text(r.focal), "the lens's focal length: shift-wheel on the picture changes it and keeps the subject's size");
    });
    dof_switch(ui, st, px);
    if let Some(n) = st.camera().readout(px.aspect).fstop {
        worded(ui, st, px, "av", &format!("f/{}", stop_text(n)), "the f-stop: smaller numbers open the lens wider and blur more; a slider step is a third of a stop. Click the glass to focus there");
    }
    let ev = gf_core::camera::get(&st.p, "ev").unwrap_or(0.0);
    let k = Knob { label: "exposure".into(), ..engine("ev") };
    knob::knob_row_with(ui, st, &k, px, knob::Extra { shown: &format!("{ev:+.2} EV"), tip: "brighter or darker, in stops", ..Default::default() });
    let (resp, at) = wide_button(ui, Id::unique("camera window"), "camera…", px.store.camera_open);
    px.note("camera button", at);
    if resp.on_hover_text("the whole camera: lens, focus, the view, exposure, tone, grade, the lens's character · key K").clicked() {
        px.store.camera_open = !px.store.camera_open;
    }
}

/// Where the camera window first appears: beside the left panel, set down and across from any other
/// window already open there, so that every title shows.
fn first_place(px: &Panels) -> Pos2 {
    let others = usize::from(px.store.room_open) + usize::from(px.store.lab_open);
    Pos2::new(EDGE + LEFT_W + 36.0 + 30.0 * others as f32, EDGE + 34.0 * others as f32)
}

/// The camera window, if it is open: the whole camera, a group for each task (lens, focus, view,
/// exposure, each of which folds) and the grade and the lens's character folded under them. `full`: the
/// inside of the program's window.
pub fn window(ctx: &egui::Context, st: &mut State, px: &mut Panels, full: Rect) {
    if !px.store.camera_open { return; }
    let first = px.store.camera_at.map_or_else(|| first_place(px), |p| Pos2::new(p[0], p[1]));
    let max_h = (full.height() - 2.0 * EDGE - 44.0).max(80.0);
    let out = floating(ctx, "camera", "camera", FLOAT_W, first, max_h, |ui| {
        if px.rendering { ui.disable(); }
        set_label_width(ui, COL_LABEL_WIDE);
        px.place = "cam";
        let r = st.camera().readout(px.aspect);
        let lens_says = format!("{} · {}", focal_text(r.focal), st.camera().proj.name());
        if px.window_group(ui, "cam-lens", "lens", &lens_says) { lens(ui, st, px); }
        let focus_says = match r.fstop {
            Some(n) => format!("f/{} · at {}", stop_text(n), length_text(r.focus_mm)),
            None => "all sharp".to_owned(),
        };
        if px.window_group(ui, "cam-focus", "focus", &focus_says) { focus(ui, st, px); }
        if px.window_group(ui, "cam-view", "view", &length_text(r.distance_mm)) { framing(ui, st, px); }
        let ev = gf_core::camera::get(&st.p, "ev").unwrap_or(0.0);
        if px.window_group(ui, "cam-exposure", "exposure", &format!("{ev:+.2} EV")) { exposure(ui, st, px); }
        character(ui, st, px);
        gap(ui, 2.0);
    });
    if out.closed { px.store.camera_open = false; }
    px.store.camera_at = Some([out.rect.min.x.round(), out.rect.min.y.round()]);
    px.note("camera", out.rect);
    px.note("camera:bar", out.bar);
    px.note("camera:close", out.cross);
}

/// All sharp, or the lens open (at the f-stop it was last open at, f/4 the first time).
fn dof_switch(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let r = st.camera().readout(px.aspect);
    if let Some(n) = r.fstop { px.last_fstop = n; }
    let open = r.fstop.is_some();
    let run = named_run(ui, Id::unique(("depth of field", px.place)), "focus", &[
        Item::new("all sharp", !open).tip("no depth of field: everything is sharp"),
        Item::new("lens open", open).tip(format!("depth of field: the lens opened at an f-stop (f/{} to begin with)", stop_text(px.last_fstop))),
    ]);
    for (place, name) in run.places.iter().zip(["dof:off", "dof:on"]) { px.note(format!("{}{name}", px.whose()), *place); }
    match run.clicked {
        Some(0) => st.set_depth_of_field(None),
        Some(_) if !open => st.set_depth_of_field(Some(px.last_fstop)),
        _ => {}
    }
}

/// The lens: a row of named lenses, the focal length with the field it takes in, the kind of lens and the sensor.
fn lens(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let c = st.camera();
    let r = c.readout(px.aspect);
    let words: Vec<String> = LENSES.iter().map(lens_word).collect();
    let items: Vec<Item> = LENSES
        .iter()
        .zip(&words)
        .map(|(l, w)| Item::new(w, (r.focal - l.focal).abs() < 0.5 && c.proj == l.proj).tip(format!("{}: the camera moves so that the subject keeps its size", l.name)))
        .collect();
    let run = filled_run(ui, Id::unique(("lenses", px.place)), &items, egui::vec2(5.0, 2.0), &mono(SIZE));
    for (i, place) in run.places.iter().enumerate() { px.note(format!("{}lens:{i}", px.whose()), *place); }
    if let Some(i) = run.clicked { st.lens_preset(i); }
    gap(ui, 3.0);

    // the focal length, on a slider that zooms evenly; greyed for a panorama, which takes in everything
    let r = st.camera().readout(px.aspect);
    let panorama = st.camera().proj == Projection::Panorama;
    ui.scope(|ui| {
        if panorama { ui.disable(); }
        worded(ui, st, px, "zoom", &focal_text(r.focal), "the lens's focal length: shift-wheel on the picture changes it and keeps the subject's size");
    });
    let field = match r.view_mm {
        Some(h) => format!("shows {} top to bottom", length_text(h)),
        None => format!("takes in {:.0}° × {:.0}°", r.field_v, r.field_h),
    };
    aside(ui, &field, DIM);

    // the kind of lens, and the sensor (the picture is as tall as the sensor)
    let kinds: Vec<String> = Projection::ALL.iter().map(|p| p.name().to_owned()).collect();
    if let Some(i) = pick(ui, Id::unique(("lens kind", px.place)), "kind", &kinds, st.camera().proj.index()) {
        st.camera_arg("proj", &i.to_string());
    }
    let sensor = st.camera().readout(px.aspect).sensor;
    let mut sensors: Vec<String> = SENSORS.iter().map(|s| s.0.to_owned()).collect();
    let now = SENSORS.iter().position(|s| (s.2 - sensor).abs() < 0.05).unwrap_or_else(|| {
        sensors.push(format!("{sensor:.1} mm high"));
        sensors.len() - 1
    });
    if let Some(i) = pick(ui, Id::unique(("sensor", px.place)), "sensor", &sensors, now) {
        if let Some(s) = SENSORS.get(i) { camera::set(&mut st.p, "sensor", s.2); }
    }
}

/// The focus: everything sharp, or a lens opened at an f-stop with what follows from it.
fn focus(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    dof_switch(ui, st, px);
    let r = st.camera().readout(px.aspect);
    let Some(n) = r.fstop else {
        note(ui, "Open the lens, then click the glass to focus there.");
        return;
    };

    // how big the subject is in the real world: small things blur fast
    worded(ui, st, px, "subjectLog", &length_text(st.p.subject), "How big the shape is in the real world. Small things blur fast: this decides how deep the focus is at a given f-stop.");
    let subject = st.p.subject;
    let words: Vec<&str> = SUBJECTS.iter().map(|s| s.0.split(" · ").next().unwrap_or(s.0)).collect();
    let items: Vec<Item> = SUBJECTS.iter().zip(&words).map(|(s, w)| Item::new(w, (s.1 - subject).abs() <= 0.01 * s.1).tip(s.0)).collect();
    let run = filled_run(ui, Id::unique(("subjects", px.place)), &items, egui::vec2(4.0, 1.0), &mono(10.0));
    for (i, place) in run.places.iter().enumerate() { px.note(format!("{}subject:{i}", px.whose()), *place); }
    if let Some(i) = run.clicked { camera::set(&mut st.p, "subject", SUBJECTS[i].1); }
    gap(ui, 3.0);

    worded(ui, st, px, "av", &format!("f/{}", stop_text(n)), "the f-stop: smaller numbers open the lens wider and blur more; a slider step is a third of a stop");
    let r = st.camera().readout(px.aspect);
    worded(ui, st, px, "focus", &format!("at {}", length_text(r.focus_mm)), "where the lens is focused (click the glass to focus there)");
    let r = st.camera().readout(px.aspect);
    let band = if st.p.ffar - st.p.fnear <= 0.0 { "one plane".to_owned() } else { length_text(r.band_mm) };
    worded(ui, st, px, "band", &band, "A focus stack: everything in this range is sharp, as if several exposures were merged. 0 is a single plane, as a real lens has.");

    // what that comes to, in a photographer's words; and anything no real lens would do, said plainly
    let r = st.camera().readout(px.aspect);
    let to = if r.sharp_to_mm.is_finite() { length_text(r.sharp_to_mm) } else { "∞".to_owned() };
    paragraph(ui, &format!("sharp from {} to {} · {:.2}× life size", length_text(r.sharp_from_mm), to, r.magnification), mono(10.0), DIM, LINE, 0.0);
    for line in &r.notes { paragraph(ui, line, sans(SMALL), COSTLY, 1.35, 0.0); }
    let font = mono(SIZE);
    let size = button_size(ui, "one plane", &font, egui::vec2(7.0, 2.0));
    let row_r = row(ui, size.y + 2.0);
    let at = Rect::from_min_size(Pos2::new(row_r.right() - COL_WAVE - COL_GAP - size.x, row_r.center().y - size.y / 2.0), size);
    px.note(format!("{}band:one plane", px.whose()), at);
    ui.scope(|ui| {
        if st.p.ffar - st.p.fnear <= 0.0 { ui.disable(); }
        if button(ui, at, Id::unique(("true lens", px.place)), "one plane", false, font.clone()).on_hover_text("a single plane in focus, as a real lens has").clicked() {
            camera::set(&mut st.p, "band", 0.0);
        }
    });

    // how the blur is made: laid over the sharp picture afterwards, or every ray through the lens
    let exact = st.p.dof < 0.5;
    let run = named_run(ui, Id::unique(("focus done", px.place)), "blur", &[
        Item::new("clean", !exact).tip("the picture is traced sharp and blurred afterwards by how far the glass is from the band in focus"),
        Item::new("exact", exact).tip("every ray is traced through the lens, as a camera's would go"),
    ]);
    for (place, name) in run.places.iter().zip(["focus:clean", "focus:exact"]) { px.note(format!("{}{name}", px.whose()), *place); }
    if let Some(i) = run.clicked { st.p.dof = 1.0 - i as f64; }
    // (two lines of words either way, so that nothing below jumps when the choice is made)
    let exact = st.p.dof < 0.5;
    note(ui, if exact {
        "The true lens: right also for what shows through the glass, but grainy until it settles. Best for stills."
    } else {
        "Clean at once. What shows through the glass blurs with the glass in front of it."
    });
    ui.scope(|ui| {
        // rays through the lens are only counted when rays go through the lens
        if !exact { ui.disable(); }
        knob::knob_row(ui, st, &engine("lensN"), px);
    });
    if let Some(i) = choice(ui, Id::unique(("planes", px.place)), "planes", &["hide", "show"], js_round(st.p.planes) as usize) {
        st.p.planes = i as f64;
    }
    note(ui, "Click the glass to focus there.");
}

/// The view: where the camera stands (how far off, how high, looking down; the picture slid across the
/// lens; its roll) and how the shape is turned in front of it, with the named views, and one reset for
/// all of it.
fn framing(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let r = st.camera().readout(px.aspect);
    worded(ui, st, px, "dist", &length_text(r.distance_mm), "from the lens to the middle of the shape");
    let elev = Knob { label: "height °".into(), ..engine("elev") };
    knob::knob_row_with(ui, st, &elev, px, knob::Extra { tip: "the camera's height: raised on an arc and looking down at the shape, the room, its lights and the shape tilting together", ..Default::default() });
    let tip = "the picture slid across the lens's image, in half-heights of the picture: with a long lens this is how a close-up is framed";
    for name in ["shiftX", "shiftY"] { knob::knob_row_with(ui, st, &engine(name), px, knob::Extra { tip, ..Default::default() }); }
    knob::knob_row(ui, st, &engine("roll"), px);
    // how the shape is turned in front of the camera (dragging the picture turns it too)
    for kn in State::orient_knobs() {
        let label = format!("view {}", kn.label);
        knob::knob_row(ui, st, &Knob { label, ..kn }, px);
    }
    gap(ui, 3.0);
    let views = [("front", 0.0, 0.0), ("back", 0.0, 180.0), ("left", 0.0, 90.0), ("right", 0.0, -90.0), ("top", 90.0, 0.0), ("under", -90.0, 0.0)];
    let items: Vec<Item> = views.iter().map(|v| Item::new(v.0, false).tip(format!("the shape seen from the {}", v.0))).collect();
    let run = filled_run(ui, Id::unique(("views", px.place)), &items, egui::vec2(5.0, 2.0), &mono(SIZE));
    for (v, place) in views.iter().zip(&run.places) { px.note(format!("{}view:{}", px.whose(), v.0), *place); }
    if let Some(i) = run.clicked { st.set_r(&gf_core::math::euler(views[i].1, views[i].2, 0.0)); }
    let (resp, at) = wide_button(ui, Id::unique(("reset view", px.place)), "reset view", false);
    px.note(format!("{}reset view", px.whose()), at);
    if resp.on_hover_text("the camera and the shape's turn back the way this shape first appears · key R, or a double click").clicked() {
        st.reset_view();
    }
}

/// Is the grade as it starts (warmth, tint, contrast, saturation)?
fn graded(st: &State) -> bool {
    st.p.get("wbTemp") != Some(0.0) || st.p.get("wbTint") != Some(0.0) || st.p.get("contrast") != Some(1.0) || st.p.get("saturation") != Some(1.0)
}

/// The exposure, bloom, the tone curve, and folded under them the grade.
fn exposure(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let ev = gf_core::camera::get(&st.p, "ev").unwrap_or(0.0);
    let k = Knob { label: "exposure".into(), ..engine("ev") };
    knob::knob_row_with(ui, st, &k, px, knob::Extra { shown: &format!("{ev:+.2} EV"), tip: "brighter or darker, in stops", ..Default::default() });
    knob::knob_row(ui, st, &engine("bloom"), px);
    let tone = (js_round(st.p.get("tone").unwrap_or(0.0)).max(0.0) as usize).min(TONES.len() - 1);
    // the tone curve, from a menu (each says what it is for when hovered in the light of its name)
    let names: Vec<String> = TONES.iter().map(|t| t.0.to_owned()).collect();
    let at = ui.cursor().min;
    if let Some(i) = pick(ui, Id::unique(("tone", px.place)), "tone", &names, tone) { st.p.set("tone", i as f64); }
    let r = Rect::from_min_max(at, Pos2::new(at.x + ui.available_width(), ui.cursor().top()));
    px.note(format!("{}tone", px.whose()), r);
    ui.interact(Rect::from_min_size(r.min, egui::vec2(label_width(ui), r.height())), Id::unique(("tone name", px.place)), egui::Sense::hover()).on_hover_text(TONES[tone].1);

    let says = if graded(st) { "changed" } else { "as shot" };
    px.sub(ui, "camera-grade", "grade", says, |ui, px| {
        {
            for name in ["wbTemp", "wbTint", "contrast", "saturation"] { knob::knob_row(ui, st, &engine(name), px); }
            let font = mono(SIZE);
            let size = button_size(ui, "reset", &font, egui::vec2(8.0, 2.0));
            let r = row(ui, size.y + 2.0);
            let at = Rect::from_min_size(Pos2::new(r.right() - size.x, r.center().y - size.y / 2.0), size);
            px.note(format!("{}grade:reset", px.whose()), at);
            let changed = graded(st);
            ui.scope(|ui| {
                if !changed { ui.disable(); }
                if button(ui, at, Id::unique(("grade reset", px.place)), "reset", false, font).on_hover_text("warmth, tint, contrast and saturation as the picture was shot").clicked() {
                    for (name, v) in [("wbTemp", 0.0), ("wbTint", 0.0), ("contrast", 1.0), ("saturation", 1.0)] { st.p.set(name, v); }
                }
            });
        }
    });
}

/// What of the lens's character is on, in a few words.
fn character_says(st: &State) -> String {
    let p = &st.p;
    let v = |n: &str| p.get(n).unwrap_or(0.0);
    let mut on: Vec<&str> = Vec::new();
    if v("distort") != 0.0 { on.push(if v("distort") > 0.0 { "barrel" } else { "pincushion" }); }
    if v("fringe") > 0.0 { on.push("fringing"); }
    if v("falloff") > 0.0 { on.push("fall-off"); }
    if v("blades") > 2.5 { on.push("blades"); }
    if v("squeeze") > 1.0 { on.push("anamorphic"); }
    if v("tiltX") != 0.0 || v("tiltY") != 0.0 { on.push("tilt"); }
    if on.is_empty() { "plain".to_owned() } else { on.join(" · ") }
}

/// The lens's character, folded: what a real lens does to the picture besides focusing it.
fn character(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let says = character_says(st);
    let proj = st.camera().proj;
    px.sub(ui, "camera-character", "lens character", &says, |ui, px| {
        ui.scope(|ui| {
            if proj == Projection::Panorama { ui.disable(); }
            knob::knob_row_with(ui, st, &engine("distort"), px, knob::Extra { tip: "straight lines bowed: left of the middle inward (pincushion), right outward (barrel)", ..Default::default() });
        });
        knob::knob_row(ui, st, &called("fringe", "fringing"), px);
        ui.scope(|ui| {
            // (these lenses' own law has no darkening towards the corners)
            if matches!(proj, Projection::Orthographic | Projection::Panorama | Projection::Equisolid) { ui.disable(); }
            knob::knob_row_with(ui, st, &engine("falloff"), px, knob::Extra { tip: "how much of its own darkening of the corners the lens keeps", ..Default::default() });
        });
        knob::knob_row(ui, st, &engine("vignette"), px);
        // the shape of what is out of focus
        let shapes: Vec<(u32, String)> = std::iter::once((0, "round".to_owned())).chain((3..=8).map(|n| (n, format!("{n} blades")))).collect();
        if let Some(n) = pick_value(ui, "lens shape", "lens shape", &shapes, &(js_round(st.p.blades) as u32)) {
            st.p.blades = n as f64;
        }
        ui.scope(|ui| {
            // a round opening looks the same however it is turned
            if st.p.blades < 2.5 { ui.disable(); }
            knob::knob_row(ui, st, &called("bokehRot", "blade turn"), px);
        });
        knob::knob_row_with(ui, st, &called("bokeh", "highlights"), px, knob::Extra { tip: "out-of-focus highlights: left a soft core, right a bright rim", ..Default::default() });
        knob::knob_row(ui, st, &engine("catseye"), px);
        knob::knob_row_with(ui, st, &called("squeeze", "anamorphic"), px, knob::Extra { tip: "an anamorphic lens: out-of-focus highlights this many times taller than wide", ..Default::default() });
        ui.scope(|ui| {
            // a plane of focus can only lean where there is a plane in front of the lens
            if proj.radial() { ui.disable(); }
            knob::knob_row_with(ui, st, &called("tiltX", "tilt °"), px, knob::Extra { tip: "the plane of focus leant back, about the picture's horizontal", ..Default::default() });
            knob::knob_row_with(ui, st, &called("tiltY", "swing °"), px, knob::Extra { tip: "the plane of focus swung, about the picture's vertical", ..Default::default() });
        });
        knob::knob_row(ui, st, &engine("grain"), px);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_name_the_group_uses_is_a_knob() {
        for name in ["zoom", "subjectLog", "av", "focus", "band", "lensN", "dist", "elev", "shiftX", "shiftY", "roll", "ev", "bloom", "wbTemp", "wbTint", "contrast", "saturation",
                     "distort", "fringe", "falloff", "vignette", "bokehRot", "bokeh", "catseye", "squeeze", "tiltX", "tiltY", "grain"] {
            assert!(State::engine_knobs().iter().any(|k| k.id == name), "{name} is not a knob");
        }
        let p = gf_core::settings::Settings::default();
        for name in ["tone", "blades", "planes", "dof", "subject", "proj"] { assert!(p.get(name).is_some(), "{name} is not a setting"); }
    }

    #[test]
    fn the_lenses_have_short_names() {
        let words: Vec<String> = LENSES.iter().map(lens_word).collect();
        assert_eq!(words, ["24", "35", "50", "85", "100 macro", "200", "fisheye 15", "fisheye 8"]);
    }
}
