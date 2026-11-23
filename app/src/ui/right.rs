//! The right panel: the shape and how it moves. It is made of parts, one under the other. At its head the
//! layers and how they are joined (see `layers`; one layer is one shape, as ever). Then the parts of the
//! selected layer's shape: the families, the family's shapes, the mathematics of the one chosen, its own
//! sliders, where it is placed. At the foot, what is not any one shape's: the animation (see `animate`).
//! Volume detail belongs to the selected object.

use eframe::egui::{self, Align2, Id, Pos2, Rect, Sense, Ui};
use gf_core::presets::Kind;
use gf_core::state::{Knob, State, RES_MAX};

use super::theme::*;
use super::widgets::*;
use super::{animate, knob, layers, Panels};

/// Voxels per side offered for the still shape, and for the shape while it animates.
pub const DETAIL_STILL: [u32; 5] = [384, 512, 768, 1024, RES_MAX];
pub const DETAIL_MOVING: [u32; 5] = [192, 256, 320, 384, 512];
/// How far the equations stand in from the bar at their left.
const INSET: f32 = 9.0;
/// Where the detail buttons begin, leaving room for their name.
const DETAIL_INDENT: f32 = 68.0;

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
    let width = ui.available_width();

    // Keep the tabs, objects and shape picker visible above the scrolling controls.
    let tools = row(ui, 20.0);
    let open = button(ui, tools, Id::unique("open layer hierarchy"), "layers", px.palette.open, mono(SMALL));
    px.note("outline:toggle", tools);
    if open.on_hover_text("layers and groups (O)").clicked() { px.palette.open = !px.palette.open; }
    super::outline::breadcrumbs(ui, st, px);
    let world = st.edit_scope.object().is_none();
    let creating = world && tabs(ui, st, px);
    if world && (st.edit_scope.world.is_some() || st.stack.hierarchy.as_ref().is_some_and(|h| h.0.iter().any(|n| matches!(n, gf_core::groups::Node::Group(_))))) {
        super::world::contents(ui, st, px);
    } else if world && st.object_count() > 0 { layers::strip(ui, st, px); }
    let group_name = st.selected_group.and_then(|id| st.stack.hierarchy.as_ref()?.group(id)).map(|g| g.name.clone());
    if let Some(name) = &group_name {
        let header = row(ui, 28.0);
        text_in(ui, header, Align2::LEFT_CENTER, name, mono(SIZE), pal().bright);
    } else if creating {
        super::picker::add_button(ui, st, px);
    } else if world && matches!(st.world_selection().last(), Some(gf_core::groups::Item::Object(_))) {
        // one object: its shape card as inside it (a press changes its shape), and the way inside for its parts
        if one_object(st) {
            super::picker::shape_card(ui, st, px);
            gap(ui, 4.0);
        }
        let (edit, at) = wide_button(ui, Id::unique("scope edit"), &format!("inside {}…", st.layer_title(st.stack.sel).to_lowercase()), false);
        px.note("scope:edit", at);
        if edit.on_hover_text("go inside the object: its csg parts, modifiers of its own, its construction").clicked() { st.enter_world_item(gf_core::groups::Item::Object(st.stack.sel)); }
    } else if !world && st.object_count() > 0 {
        super::picker::shape_card(ui, st, px);
    }
    let head = ui.cursor().top() - top;

    // The selected object's own controls, which scroll when they need more room than is left: its shape's
    // mathematics and sliders, how finely it is built (a volume object), its tree (a csg object), where it is
    // placed, how the objects are joined, the surface studio.
    let room = (px.room_h - head - px.anim_h - 4.0).max(40.0);
    // (as tall as the room there is, however little is in it: the animation stays at the foot)
    let mut area = egui::ScrollArea::vertical().id_salt("object controls").max_height(room).min_scrolled_height(room).auto_shrink([false, false]);
    if let Some(y) = px.right_scroll.take() { area = area.vertical_scroll_offset(y.clamp(0.0, px.controls_max)); }
    let controls = area.show(ui, |ui| {
        ui.set_width(width);
        px.clip = Some(ui.clip_rect());
        if group_name.is_some() {
            super::world::transform(ui, st, px);
            if let Some(id) = st.selected_group { super::world_modifiers::show(ui, st, px, gf_core::groups::Item::Group(id)); }
            return;
        }
        if creating {
            let what = if px.tab.unwrap_or_else(|| st.object_count() > 0 && st.is_csg_object(st.stack.sel)) { "a csg object: a tree whose first part is this shape, its formula traced exactly, or a grid of it" } else { "a volume object: the shape baked into a grid of distances" };
            paragraph(ui, &format!("a shape picked there makes {what}"), mono(SMALL), pal().faint, LINE, INSET);
            gap(ui, 6.0);
            return;
        }
        if world {
            if !st.world_selection().is_empty() {
                // one object chosen: its shape's formula, sliders and detail straight away, then where it is
                // placed (going inside is for its parts and construction)
                if one_object(st) { shape_folds(ui, st, px, false); }
                super::world::transform(ui, st, px);
                super::world_modifiers::show(ui, st, px, gf_core::groups::Item::Object(st.stack.sel));
            }
            return;
        }
        shape_folds(ui, st, px, true);
        // A volume can supply the base of a construction just as an analytic shape can.
        let n = st.csg.parts.len();
        let says = [format!("{n} part{}", if n == 1 { "" } else { "s" }), super::csg::hint(st).to_owned()].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" · ");
        if px.window_group(ui, "object-parts", "csg parts", &says) { super::csg::show_headed(ui, st, px); }
        let m = st.selected_modifiers().list.len();
        let says = match m { 0 => "none".to_owned(), 1 => "1".to_owned(), _ => format!("{m} · applied from bottom to top") };
        if px.window_group(ui, "object-modifiers", "modifiers", &says) { super::mods::show_in(ui, st, px, true); }
    });
    px.clip = None;
    let controls_h = controls.content_size.y;
    px.controls_max = (controls_h - controls.inner_rect.height()).max(0.0);

    // everything that moves, at the foot, wherever the controls above have scrolled to
    let a0 = ui.cursor().top();
    animate::show(ui, st, px);
    let anim_h = ui.cursor().top() - a0;

    // Remeasure when the pinned animation changes height.
    let rest = head + controls_h + anim_h + 4.0;
    if (rest - px.right_rest).abs() > 0.5 || (anim_h - px.anim_h).abs() > 0.5 {
        px.right_rest = rest;
        px.anim_h = anim_h;
        ui.ctx().request_repaint();
    }
}

/// One object is chosen at the world level (the one the panel's controls are for).
fn one_object(st: &State) -> bool {
    let sel = st.world_selection();
    sel.len() == 1 && sel.last() == Some(&gf_core::groups::Item::Object(st.stack.sel))
}

/// The object's shape and how finely it is built, each folding with a word of what it is set to. `inside`:
/// within the object (a path's builder shows there too).
fn shape_folds(ui: &mut Ui, st: &mut State, px: &mut Panels, inside: bool) {
    let says = format!("{} · {}", st.preset().name.to_lowercase(), st.family().name.to_lowercase());
    if px.window_group(ui, "object-shape", "shape", &says) {
        maths(ui, st, px);
        words(ui, st, px);
        sliders(ui, st, px);
        // (a path: the builder, under the profile's sliders)
        if inside { super::paths::panel(ui, st, px); }
    }
    let sel = st.stack.sel;
    // (how finely the object is built is the volume object's; a csg object's when its first part is a grid)
    if !st.is_csg_object(sel) || st.csg_tree_layer(sel).0.first().is_some_and(|p| p.grid.is_some()) {
        let says = format!("voxels a side · {} · {} moving", st.shape().res, st.shape().anim_res);
        if px.window_group(ui, "object-detail", "detail", &says) { detail_rows(ui, st, px); }
    }
}

/// The two tabs at the head of the panel: volume objects and csg objects. The tab shown is the selected
/// object's kind, put back whenever the selection (or what kind it is) changes; the other tab, chosen, lets a
/// shape be picked to make a new object of that kind. True while picking a shape makes a new object (the
/// other tab is up, or there is no object at all).
fn tabs(ui: &mut Ui, st: &mut State, px: &mut Panels) -> bool {
    let count = st.object_count();
    let kind = count > 0 && st.is_csg_object(st.stack.sel);
    let seen = (st.stack.sel, count, kind);
    if px.tab_seen != seen {
        px.tab_seen = seen;
        px.tab = None;
        px.adding = false;
        px.browse_fam = st.fam;
    }
    let csg = px.tab.unwrap_or(kind);
    let items = [
        Item::new("volume", !csg).tip("volume objects: an independent shape traced from a sampled distance field"),
        Item::new("csg", csg).tip("csg objects: a tree of parts joined by union, subtract or intersect with smooth seams; each part a shape's formula, traced exactly, or a grid of a shape with none"),
    ];
    let run = filled_run(ui, Id::unique("object tabs"), &items, egui::vec2(8.0, 3.0), &mono(SIZE));
    for (w, place) in ["volume", "csg"].iter().zip(&run.places) { px.note(format!("tab:{w}"), *place); }
    // A bar under the tab that is up, which slides to the other when the tab changes (by a click, or when an
    // object is converted or another selected): the change is seen to happen, not only found to have.
    if let [a, b] = run.places[..] {
        let t = ui.ctx().animate_value_with_time(Id::unique("tab slide"), if csg { 1.0 } else { 0.0 }, 0.22);
        let x0 = egui::lerp(a.left()..=b.left(), t) + 10.0;
        let x1 = egui::lerp(a.right()..=b.right(), t) - 10.0;
        let y = a.bottom() + 2.0;
        ui.painter().line_segment([Pos2::new(x0, y), Pos2::new(x1, y)], egui::Stroke::new(2.0, white(0.85)));
    }
    if let Some(i) = run.clicked {
        st.selected_group = None;
        let want = i == 1;
        // (the selected object's own tab, pressed again: a shape picked now adds another object of its kind;
        // pressed once more, back to the object)
        if want == kind && count > 0 && px.tab.is_none() { px.adding = !px.adding; } else { px.adding = false; }
        px.tab = (want != kind || count == 0).then_some(want);
        px.browse_fam = st.fam;
    }
    gap(ui, 6.0);
    count == 0 || px.adding || px.tab.is_some_and(|t| t != kind)
}

/// What there is to say of a model, where a shape's mathematics go: its file and what it is made of, and
/// what was found on examining it (the answer to "was my file all right?"). It stays for as long as the
/// model is the shape. While a file is being read it says so instead; and if the last file given could
/// not be used, it says why.
fn model_lines(ui: &mut Ui, st: &State, px: &mut Panels) {
    let m = px.model.clone();
    let top = ui.cursor().top();
    if let Some(name) = &m.reading {
        paragraph(ui, &format!("reading {name}…"), mono(SMALL), pal().bright, LINE, INSET);
    } else if st.needs_file(st.stack.sel) {
        paragraph(ui, "no file yet: load one, or drop one on the window", mono(SMALL), pal().dim, LINE, INSET);
    } else if m.known {
        paragraph(ui, &m.file, mono(SMALL), pal().bright, LINE, INSET);
        gap(ui, 3.0);
        paragraph(ui, &m.line, sans(SMALL), pal().text, 1.4, INSET);
    }
    if !m.said.is_empty() {
        gap(ui, 3.0);
        paragraph(ui, &m.said, sans(SMALL), pal().dim, 1.4, INSET);
    }
    if let Some(why) = &m.error {
        gap(ui, 3.0);
        paragraph(ui, why, mono(SMALL), pal().error_text, LINE, INSET);
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
            paragraph(ui, line, mono(size), pal().bright, 11.5 * LINE / size, INSET);
        }
        // the word about the shape: shown while it is the only one (with several layers the room goes to
        // the joins; hovering over the equations still says it)
        if !note.is_empty() && st.layer_count() < 2 {
            gap(ui, 4.0);
            paragraph(ui, note, sans(SMALL), pal().faint, 1.4, INSET);
        }
        if !info.is_empty() {
            gap(ui, 4.0);
            paragraph(ui, info, mono(SMALL), pal().dim, LINE, INSET);
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

/// The words of the 3D text, typed in a box: kept as its shape's `model`, the letters rebuilt as they change.
fn words(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    use gf_core::presets::text3d;
    if st.family().id != text3d::FAMILY { return; }
    let mut text = st.shape().model.clone();
    let edit = egui::TextEdit::singleline(&mut text).hint_text(text3d::DEFAULT).font(mono(SIZE)).desired_width(ui.available_width() - 2.0 * INSET);
    let r = ui.horizontal(|ui| {
        ui.add_space(INSET);
        ui.add(edit)
    }).inner;
    px.note("text words", r.rect);
    let r = r.on_hover_text("the words to make solid; a | starts another line");
    if r.changed() {
        st.shape_mut().model = text;
        st.request(gf_core::build::BuildRequest::edit());
    }
    gap(ui, 6.0);
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
    if st.family().kind == Kind::Curve && st.family().id != gf_core::presets::paths::FAMILY {
        let pace = st.shape().bead_period;
        if let Some(v) = knob::plain_row(ui, Id::unique("bead pace"), "bead pace s", pace, 0.3, 10.0, 0.1).value {
            st.shape_mut().bead_period = v;
        }
    }
}

/// What the scene's volumes hold in video memory at full detail, as a thin bar against the budget, and
/// a word when every volume has been built coarser to stay within it.
fn memory_meter(ui: &mut Ui, st: &State) {
    let plan = st.detail_plan(gf_core::state::Detail::Full);
    let budget = gf_core::budget::TOTAL_MAX;
    let r = row(ui, 12.0);
    let words = format!("video memory {} of {}", memory(plan.bytes), memory(budget));
    text_in(ui, r, Align2::LEFT_CENTER, &words, mono(SMALL), pal().faint);
    let bar = Rect::from_min_max(Pos2::new(r.right() - 60.0, r.center().y - 1.5), Pos2::new(r.right(), r.center().y + 1.5));
    let share = (plan.bytes as f32 / budget as f32).clamp(0.0, 1.0);
    ui.painter().rect_filled(bar, 1.0, pal().button);
    ui.painter().rect_filled(Rect::from_min_size(bar.min, egui::vec2(bar.width() * share, bar.height())), 1.0, if plan.lowered { pal().costly } else { pal().dim });
    let note = st.memory_note(gf_core::state::Detail::Full);
    ui.allocate_rect(r, Sense::hover()).on_hover_text(format!(
        "every object's volume, the scene's and the bodies': at most {} together. {}", memory(budget),
        note.clone().unwrap_or_else(|| "Within it, everything is built at the detail asked for".to_string())));
    if plan.lowered {
        paragraph(ui, &format!("detail held to {}³ to fit {} of video memory", plan.scene, memory(budget)), mono(SMALL), pal().costly, LINE, 0.0);
    }
}

/// How finely the object is built: when still, and while it animates (the volume object's own controls).
/// (The heading over them, "detail", is the fold's: see `object_controls`.)
pub fn detail_rows(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let (still, moving) = (st.shape().res, st.shape().anim_res);
    let name = |ui: &Ui, r: Rect, text: &str| {
        text_in(ui, Rect::from_min_size(r.min, egui::vec2(DETAIL_INDENT, r.height())), Align2::LEFT_CENTER, text, mono(SIZE), pal().dim);
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
    paragraph(ui, &built_text(st, px.joined), mono(SMALL), pal().dim, LINE, 0.0);
    gap(ui, 1.0);
    memory_meter(ui, st);
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
