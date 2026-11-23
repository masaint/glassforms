//! A csg object's tree of parts (gf_core::csg), in the right panel under its shape's sliders: how its parts'
//! edges are drawn, its parts one row each (the first, the object's own shape; a part a level deeper than
//! the row above it is in a group), and the selected part's own controls: its operation and seam, its group
//! level, its shape and sliders, where it sits. Then "add part" and "convert to volume".

use eframe::egui::{self, Align2, Id, Ui};
use gf_core::csg::{Op, Part, MAX_LEVEL, MAX_PARTS};
use gf_core::layers::Placement;
use gf_core::sdf::{support, Support, Trace};
use gf_core::build::BuildRequest;
use gf_core::state::State;

use super::theme::*;
use super::widgets::*;
use super::{knob, Panels};

/// How far a note stands in from the panel's edge (as right.rs's).
const INSET: f32 = 9.0;
/// How far a row stands in for each group level.
const INDENT: f32 = 12.0;

/// What a change asks for: a quick build while a slider is dragged, a careful one when it is let go.
fn moved(st: &mut State, m: &Moved) -> Option<f64> {
    if m.value.is_some() { st.request(BuildRequest::handle(true)); }
    if m.released { st.request(BuildRequest::handle(false)); }
    m.value
}

/// The parts (the way the shape up is traced is the row above them, right.rs `trace_row`).
pub fn show(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    parts(ui, st, px, false);
}
/// The same under a heading of the caller's that says `hint` (the right panel's fold), with no title row.
pub fn show_headed(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    parts(ui, st, px, true);
}

/// How the parts are traced, when that is worth a word beside their title.
pub fn hint(st: &State) -> &'static str {
    if st.csg.baked {
        "baked into this volume object"
    } else if st.layer_count() > 1 {
        "traced from its volume among others"
    } else if st.shape().trace == Trace::Volume && !st.csg.parts.is_empty() {
        "joined to the shape up"
    } else {
        ""
    }
}

/// A part's name in its row.
fn title(st: &State, p: &Part) -> String {
    let shape = st.families.iter().find(|f| f.id == p.family).and_then(|f| f.presets.iter().find(|q| q.id == p.preset)).map_or(p.preset.as_str(), |q| q.name);
    let op = match p.op { Op::Union => "+", Op::Subtract => "−", Op::Intersect => "∩" };
    format!("{op} {}", shape.to_lowercase())
}

fn parts(ui: &mut Ui, st: &mut State, px: &mut Panels, headed: bool) {
    let n = st.csg.parts.len();
    if !headed {
        let r = row(ui, 14.0);
        text_in(ui, r, Align2::LEFT_CENTER, "csg parts", mono(10.0), pal().faint);
        text_in(ui, r, Align2::RIGHT_CENTER, hint(st), mono(10.0), pal().hint);
    }
    gap(ui, 3.0);
    // The object's own setting for its tree, beside its parts: their edges drawn from the parts' formulas, or
    // from the volume the tree is measured into (faster with many parts). (Among other objects a tree is always
    // drawn from its volume: nothing to choose.)
    if n > 0 && st.layer_count() == 1 && !st.csg.baked {
        let coarse = st.csg.coarse;
        let items = [
            Item::new("exact", !coarse).tip("each part's edges from its formula, worked out by the tracer: sharp at any zoom, slower the more parts there are"),
            Item::new("voxel", coarse).tip("each part's edges as the tree's volume holds them: as fine as its voxels, about as fast with fifty parts as with one"),
        ];
        let run = named_run(ui, Id::unique("csg coarse"), "part edges", &items);
        for (w, place) in ["exact", "voxel"].iter().zip(&run.places) { px.note(format!("csg edges:{w}"), *place); }
        if let Some(i) = run.clicked { st.csg.coarse = i == 1; st.request(BuildRequest::edit()); }
        gap(ui, 3.0);
    }
    // The shape's own strokes (it is the tree's first part): cheap, exact changes to it alone
    if st.is_csg_object(st.stack.sel) {
        let open = px.strokes_open;
        let s = st.shape().strokes;
        let says = if s.is_none() { "none" } else { "in use" };
        let (pressed, at) = fold_heading(ui, Id::unique("csg up strokes"), "strokes of the shape", !open, says);
        px.note("csg strokes", at);
        if pressed { px.strokes_open = !open; }
        if open {
            let own = has_own_round(st, st.family().id, st.preset().id);
            let got = strokes_rows(ui, Id::unique("csg up strokes rows"), s, own);
            strokes_done(st, got, |st, s| st.shape_mut().strokes = s);
            gap(ui, 4.0);
        }
    }
    // The whole shape first: the object that places the tree in the scene. Selected, the handles move all of
    // it; a part selected below is moved within it. (Two levels of editing: what the shape is made of, and
    // where the shape stands.)
    if st.edit_scope.construction_level().is_empty() && (n > 0 || st.edit_scope.object().is_some()) {
        let chosen = st.csg.whole;
        let items = [Item::new("whole shape", chosen).tip("the tree as one object: moved, turned and sized with its parts")];
        let run = button_run(ui, Id::unique("csg whole"), 0.0, &items, egui::vec2(8.0, 3.0), &mono(SIZE), 0.0);
        if let Some(r) = run.places.first() { px.note("csg whole", *r); }
        if run.clicked.is_some() {
            st.csg.whole = !chosen;
            if !chosen { st.enter_construction_level(&[]); px.handles.pick(None); }
        }
        if st.csg.whole { gap(ui, 3.0); placement_rows(ui, st, super::handles::Target::Whole); gap(ui, 6.0); }
    }
    if st.edit_scope.object().is_some() {
        super::construction::panel(ui, st, px);
    } else {
    // The tree's first part: the object's own shape, whose sliders are at the top of these controls (pressing
    // its row brings them into view).
    let first = first_number(st);
    if first == 2 {
        let name = format!("1  {}", st.preset().name.to_lowercase());
        let items = [Item::new(&name, false).tip("the object's own shape, the tree's first part: its sliders are at the top")];
        let run = button_run(ui, Id::unique("csg first part"), 0.0, &items, egui::vec2(8.0, 3.0), &mono(SIZE), 0.0);
        if let Some(r) = run.places.first() { px.note("csg:first", *r); }
        if run.clicked.is_some() { px.right_scroll = Some(0.0); }
    }
    // a row for each part: pressing one selects it (pressing it again closes it)
    for i in 0..n {
        let level = st.csg.parts[i].level as f32;
        let text = format!("{}  {}", i + first, title(st, &st.csg.parts[i]));
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
    if n + 1 < MAX_PARTS {
        let (resp, at) = wide_button(ui, Id::unique("csg add"), "add part", false);
        px.note("csg add", at);
        if resp.on_hover_text("a new part where it shows: a ball (a block beside a ball), on the right of the shape as it is seen now, selected, with its handles").clicked() {
            add(st, px);
        }
    }
    }
    // a csg object made a volume object: its tree baked into a volume, traced from that (the tab then shows it
    // as the volume object it is)
    if st.is_csg_object(st.stack.sel) {
        let (resp, at) = wide_button(ui, Id::unique("csg convert"), "convert to volume", false);
        px.note("csg convert", at);
        if resp.on_hover_text("bake this tree into a volume object: traced from its volume, with its placement and material").clicked() {
            st.convert_to_volume();
            px.handles.pick(None);
        }
    }
    // the grid parts' video memory, and the cap on them
    if n > 0 {
        let (bytes, own) = st.csg_grid_bytes();
        let (_, _, left) = st.csg_tree_counted();
        if own > 0 || left > 0 {
            gap(ui, 3.0);
            let mut say = format!("grid parts: {own}, {} of video memory", mib(bytes));
            if left > 0 {
                say += &format!(". {left} left out: at most {} grid parts and {} at once", gf_core::csg::MAX_GRIDS, mib(gf_core::csg::GRID_BUDGET));
            }
            paragraph(ui, &say, mono(SMALL), if left > 0 { pal().hint } else { pal().faint }, LINE, INSET);
        }
    }
    gap(ui, 4.0);
}

/// Bytes as the panel says them.
fn mib(bytes: u64) -> String {
    if bytes >= 1 << 30 { format!("{:.1} gb", bytes as f64 / (1u64 << 30) as f64) } else { format!("{} mb", (bytes as f64 / (1u64 << 20) as f64).round()) }
}

/// The number the tree's parts are counted from in its list (and on the picture): 2 when the object's own
/// shape is the tree's first part (a csg object, or a volume object made from one), 1 for parts beside a
/// volume object (setups of before).
pub fn first_number(st: &State) -> usize {
    if st.is_csg_object(st.stack.sel) || st.csg.baked { 2 } else { 1 }
}

/// A new part, made where it shows (an exact copy inside the shape would change nothing on the picture): a
/// ball, or a block when the object's own shape is a ball, at a little under half its size, standing out on
/// the right of the shape as it is seen now (each further one a step lower or higher). It is selected, with
/// its handles shown.
pub(super) fn add(st: &mut State, px: &mut Panels) {
    let n = st.csg.parts.len();
    let ball = st.family().id == "studio" && st.preset().id == "sphere";
    let (family, preset) = if ball { ("studio", "block") } else { ("studio", "sphere") };
    // Offset from the active level along the screen axes. The stored placement
    // is local to that level, even inside a rotated World object or a nested group.
    let v = st.view_r();
    let (right, up) = ([v[0], v[1], v[2]], [v[3], v[4], v[5]]);
    let lift = 0.35 * ((n % 3) as f64 - 1.0) * if n % 2 == 0 { 1.0 } else { -1.0 };
    let frame = st.csg.construction.as_ref()
        .and_then(|t| t.node(&st.edit_scope.construction_level()))
        .map(|group| st.construction_frame(st.stack.sel, group))
        .unwrap_or_else(|| st.modifier_frame(st.stack.sel));
    let origin = frame.to_world(frame.origin);
    let at = std::array::from_fn(|k| origin[k] + frame.placement.size * (0.7 * right[k] + lift * up[k]));
    let pos = frame.to_local(at);
    let part = Part { family: family.into(), preset: preset.into(), place: Placement { pos, rot: [0.0; 3], size: 0.4 }, smooth: 0.03, ..Default::default() };
    if st.edit_scope.object().is_some() {
        let parent = st.edit_scope.construction_level().to_vec();
        let Some(path) = st.csg.add_construction_part(&parent, part) else { return; };
        st.select_construction_node(&path, false);
    } else { st.csg.parts.push(part); }
    st.csg.sel = Some(n);
    st.csg.whole = false;
    px.handles.pick(None);
    // (its handles shown at once: the tool that moves it, unless one with handles is up already)
    if matches!(px.handles.tool(), super::handles::Tool::Camera | super::handles::Tool::Select) { px.handles.set_tool(super::handles::Tool::Move); }
    st.request(BuildRequest::edit());
}

/// The selected part's controls.
fn selected(ui: &mut Ui, st: &mut State, px: &mut Panels, i: usize) {
    // how it joins what is above it
    let op = st.csg.parts[i].op;
    // (on one row of its own, the width of the panel: union, subtract, intersect)
    let items: Vec<Item> = Op::ALL.iter().map(|o| Item::new(o.id(), *o == op).tip("how this part joins what is above it in the list")).collect();
    let run = filled_run(ui, Id::unique(("csg op", i)), &items, egui::vec2(8.0, 3.0), &mono(SIZE));
    gap(ui, 3.0);
    for (o, place) in Op::ALL.iter().zip(&run.places) { px.note(format!("csg op:{}", o.id()), *place); }
    if let Some(k) = run.clicked { st.csg.parts[i].op = Op::ALL[k]; st.request(BuildRequest::edit()); }
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
        st.request(BuildRequest::edit());
    }
    // its group: a part a level deeper than the one above it opens a group, joined to what is above it
    // by the group's first operation
    let level = st.csg.parts[i].level;
    let words: Vec<String> = (0..=MAX_LEVEL).map(|l| l.to_string()).collect();
    let items: Vec<Item> = words.iter().enumerate().map(|(l, w)| Item::new(w, l as u8 == level).tip("how deep in groups: a part deeper than the one above it starts a group")).collect();
    let run = named_run(ui, Id::unique(("csg level", i)), "group level", &items);
    if let Some(l) = run.clicked { st.csg.parts[i].level = l as u8; st.request(BuildRequest::edit()); }

    primitive(ui, st, i);
    let (resp, at) = wide_button(ui, Id::unique(("csg remove", i)), "remove part", false);
    px.note(format!("csg remove:{}", i + 1), at);
    if resp.clicked() {
        st.csg.parts.remove(i);
        st.csg.sel = None;
        st.request(BuildRequest::edit());
    }
}

pub(super) fn primitive(ui: &mut Ui, st: &mut State, i: usize) {
    // its shape: any shape of any family (one with no formula is a grid part: baked into a volume of its
    // own); "your file" when the shape up is a model read from a file, which the part then reads too
    let now = title(st, &st.csg.parts[i]);
    let up_model = st.is_model().then(|| st.shape().model.clone()).filter(|m| !m.trim().is_empty());
    let mut pick = None;
    egui::ComboBox::from_id_salt(("csg shape", i)).width(ui.available_width() - 8.0).selected_text(now.trim_start_matches(['+', '−', '∩', ' '])).show_ui(ui, |ui| {
        for (fi, f) in st.families.iter().enumerate() {
            for (pi, p) in f.presets.iter().enumerate() {
                let file = matches!(p.shape, gf_core::presets::Shape::Mesh { file } if file.is_empty());
                if file && up_model.is_none() { continue; }
                let on = st.csg.parts[i].family == f.id && st.csg.parts[i].preset == p.id;
                let grid = if matches!(support(f, p), Support::Formula { .. }) { "" } else { " (grid)" };
                if ui.selectable_label(on, format!("{} · {}{grid}", f.name.to_lowercase(), p.name.to_lowercase())).clicked() { pick = Some((fi, pi, file)); }
            }
        }
    });
    if let Some((fi, pi, file)) = pick {
        // (a path part takes the path up now, if the shape up is a path)
        let paths = gf_core::presets::paths::FAMILY;
        let up_path = (st.families[fi].id == paths && st.family().id == paths).then(|| st.shape().model.clone());
        let part = &mut st.csg.parts[i];
        part.family = st.families[fi].id.to_string();
        part.preset = st.families[fi].presets[pi].id.to_string();
        part.values.clear();
        part.model = if file { up_model.unwrap_or_default() } else { up_path.unwrap_or_default() };
        st.request(BuildRequest::edit());
    }
    // a grid part: its volume's voxels a side (and memory), and a tube's thickness, melt and length
    let grid = st.families.iter().find(|f| f.id == st.csg.parts[i].family)
        .and_then(|f| f.presets.iter().find(|p| p.id == st.csg.parts[i].preset).map(|p| (f, p)))
        .filter(|(f, p)| !matches!(support(f, p), Support::Formula { .. }))
        .map(|(f, p)| (f.kind == gf_core::presets::Kind::Curve, f.melt, p.fixed_length));
    {
        if let Some((curve, melt0, fixed)) = grid {
            let res = st.csg.parts[i].grid_res();
            let label = format!("detail, {}", mib(gf_core::csg::grid_bytes(res)));
            if let Some(v) = moved(st, &knob::plain_row(ui, Id::unique(("csg res", i)), &label, res as f64, gf_core::csg::GRID_RES_MIN as f64, gf_core::csg::GRID_RES_MAX as f64, 32.0)) {
                st.csg.parts[i].res = ((v / 32.0).round() * 32.0) as u32;
            }
            if curve {
                let part = &st.csg.parts[i];
                let rows = [("thickness", part.thick.unwrap_or(1.0), 0.4, 3.0, 0.02), ("melt", part.melt.unwrap_or(melt0), 0.0, 0.03, 0.0005), ("length", part.len.unwrap_or(1.0), 0.1, 3.0, 0.02)];
                for (k, (label, v, lo, hi, step)) in rows.into_iter().enumerate() {
                    if k == 2 && fixed { continue; }
                    if let Some(v) = moved(st, &knob::plain_row(ui, Id::unique(("csg tube", i, k)), label, v, lo, hi, step)) {
                        let part = &mut st.csg.parts[i];
                        match k { 0 => part.thick = Some(v), 1 => part.melt = Some(v), _ => part.len = Some(v) }
                    }
                }
            }
            paragraph(ui, "a grid part: baked into a volume of its own, as fine as its voxels; the surface studio works on the shape up only", mono(SMALL), pal().faint, LINE, INSET);
        }
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
    // its strokes
    let r = row(ui, 14.0);
    text_in(ui, r, Align2::LEFT_CENTER, "strokes", mono(10.0), pal().faint);
    let own = has_own_round(st, &st.csg.parts[i].family.clone(), &st.csg.parts[i].preset.clone());
    let got = strokes_rows(ui, Id::unique(("csg strokes", i)), st.csg.parts[i].strokes, own);
    strokes_done(st, got, |st, s| st.csg.parts[i].strokes = s);
    // where it sits, within the whole shape
    gap(ui, 3.0);
    if let Some(id) = st.csg.part_id(i) { placement_rows(ui, st, super::handles::Target::Part(id)); }
}

/// Where a part sits within the whole shape (`Some(i)`), or where the whole shape stands (`None`): x, y, z,
/// three turns and a size; a part's size may also differ along its own axes (`stretch_rows`).
pub(super) fn placement_rows(ui: &mut Ui, st: &mut State, target: super::handles::Target) {
    use super::handles::Target;
    let Some(handle) = target.construction(st) else { return };
    let mut place = handle.parent_pose(handle.pose);
    let key = match target {
        Target::Part(id) => id as usize,
        Target::Whole => usize::MAX,
        Target::ConstructionNode => usize::MAX - 1,
        _ => return,
    };
    let rows: [(&str, f64, f64, f64, f64); 7] = [
        ("x", place.pos[0], -1.4, 1.4, 0.005), ("y", place.pos[1], -1.4, 1.4, 0.005), ("z", place.pos[2], -1.4, 1.4, 0.005),
        ("turn x", place.rot[0], -180.0, 180.0, 0.5), ("turn y", place.rot[1], -180.0, 180.0, 0.5), ("turn z", place.rot[2], -180.0, 180.0, 0.5),
        ("size", place.size, 0.05, 2.0, 0.005),
    ];
    for (c, (label, v, lo, hi, step)) in rows.into_iter().enumerate() {
        let id = Id::unique(("csg place", key, c));
        if let Some(v) = moved(st, &knob::plain_row(ui, id, label, v, lo, hi, step)) {
            let p = match target {
                Target::Part(id) => {
                    let Some(i) = st.csg.construction_part(id) else { return };
                    &mut st.csg.parts[i].place
                }
                Target::Whole => {
                    if !st.has_tree(st.stack.sel) { st.csg.edit_construction(); }
                    &mut st.csg.place
                }
                Target::ConstructionNode => {
                    let Some(node) = st.csg.edit_construction().node_mut(&st.edit_scope.construction()) else { return };
                    &mut node.place
                }
                _ => return,
            };
            match c { 0..=2 => place.pos[c] = v, 3..=5 => place.rot[c - 3] = v, _ => place.size = v }
            *p = handle.geometry_placement(place);
        }
    }
    if let Target::Part(id) = target {
        if let Some(i) = st.csg.construction_part(id) { stretch_rows(ui, st, i, place.size); }
    }
}

/// A part's size along its own axes (gf_core::stretch): alike, or per axis as three sizes (its size times
/// its stretch along each). A shape with a size of its own along an axis takes it there and stays exact
/// (a block's width, a disc's radius; a whole sphere becomes an ellipsoid, a disc or cylinder an oval);
/// anything else is stretched.
fn stretch_rows(ui: &mut Ui, st: &mut State, i: usize, size: f64) {
    let Some(per) = st.csg.parts.get(i).map(|p| p.stretch.is_some()) else { return };
    let items = [
        Item::new("alike", !per).tip("one size along all three of its own axes"),
        Item::new("per axis", per).tip("a size along each of its own axes: a shape with a size of its own there takes it and stays exact; anything else is stretched"),
    ];
    let run = named_run(ui, Id::unique(("csg stretch", i)), "size along", &items);
    if let Some(k) = run.clicked {
        let part = &mut st.csg.parts[i];
        part.stretch = (k == 1).then(|| part.stretch.unwrap_or([1.0; 3]));
        st.request(BuildRequest::edit());
    }
    let Some(s) = st.csg.parts[i].stretch else { return };
    let size = size.abs().max(1e-3);
    for (k, label) in ["size x", "size y", "size z"].into_iter().enumerate() {
        if let Some(v) = moved(st, &knob::plain_row(ui, Id::unique(("csg stretch row", i, k)), label, size * s[k], 0.01, 4.0, 0.005)) {
            if let Some(s) = &mut st.csg.parts[i].stretch { s[k] = (v / size).max(1e-3); }
        }
    }
}

/// A shape's strokes (gf_core::csg::Strokes): cheap, exact changes to one part alone, before it is joined. Its
/// own sliders already hold a bevel and a round where the shape has edges of its own; `own_round`: it has, so
/// the round here is left to them. Returns the strokes changed, if they were, and whether a slider is still
/// moving.
fn strokes_rows(ui: &mut Ui, id: Id, s: gf_core::csg::Strokes, own_round: bool) -> Option<(gf_core::csg::Strokes, bool)> {
    let mut out = s;
    let mut moving = None;
    let mut row = |ui: &mut Ui, k: usize, label: &str, v: f64, lo: f64, hi: f64, step: f64| -> Option<f64> {
        let m = knob::plain_row(ui, id.with(("stroke", k)), label, v, lo, hi, step);
        if m.value.is_some() { moving = Some(true); }
        if m.released { moving = moving.or(Some(false)); }
        m.value
    };
    if let Some(v) = row(ui, 0, "shell", s.shell, 0.0, 0.4, 0.001) { out.shell = v; }
    if let Some(v) = row(ui, 1, "hole", s.hole, 0.0, 1.0, 0.002) { out.hole = v; }
    if !own_round { if let Some(v) = row(ui, 2, "round", s.round, 0.0, 0.4, 0.002) { out.round = v; } }
    if let Some(v) = row(ui, 3, "cone", s.cone, 0.0, 0.95, 0.005) { out.cone = v; }
    if let Some(v) = row(ui, 4, "star", s.star, 0.0, 0.8, 0.005) { out.star = v; }
    if s.star > 0.0 {
        if let Some(v) = row(ui, 5, "points", s.points as f64, 2.0, 24.0, 1.0) { out.points = v.round() as u32; }
    }
    // (drawn out from its middle planes along each of its own axes: a ball becomes a capsule, a rounded box)
    for (k, label) in ["elongate x", "elongate y", "elongate z"].into_iter().enumerate() {
        if let Some(v) = row(ui, 6 + k, label, s.elongate[k], 0.0, 1.0, 0.002) { out.elongate[k] = v; }
    }
    let items: Vec<Item> = ["x", "y", "z"].iter().enumerate().map(|(k, w)| Item::new(w, s.mirror[k]).tip("reflected across its own middle plane at right angles to this axis")).collect();
    let run = named_run(ui, id.with("mirror"), "mirror", &items);
    let mut clicked = false;
    if let Some(k) = run.clicked { out.mirror[k] = !s.mirror[k]; clicked = true; }
    if clicked { return Some((out, false)); }
    moving.map(|m| (out, m))
}

/// What came of the strokes' rows: put in, and the shape built again (quickly while a slider moves).
fn strokes_done(st: &mut State, got: Option<(gf_core::csg::Strokes, bool)>, put: impl FnOnce(&mut State, gf_core::csg::Strokes)) {
    if let Some((s, moving)) = got {
        put(st, s);
        st.request(BuildRequest::handle(moving));
    }
}

/// Does this shape have a round of its own among its sliders?
fn has_own_round(st: &State, family: &str, preset: &str) -> bool {
    st.families.iter().find(|f| f.id == family).and_then(|f| f.presets.iter().find(|p| p.id == preset)).is_some_and(|p| p.params.iter().any(|d| d.id == "round"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adding_a_part_in_a_transformed_group_uses_its_origin_and_the_view_axes() {
        let mut state = State::new();
        state.enter_world_item(gf_core::groups::Item::Object(0));
        state.stack.layers[0].place = Placement { pos: [0.4, 0.2, 0.1], rot: [21., 53., 18.], size: 1.2 };
        state.csg.place.rot[0] = 30.;
        let group = state.csg.edit_construction().group(&[], &[0], "Shell").unwrap();
        let node = state.csg.edit_construction().node_mut(&group).unwrap();
        node.place = Placement { pos: [-0.3, 0.08, 0.4], rot: [-17., 26., 91.], size: 0.6 };
        node.local_origin = Some([0.32, -0.1, 0.21]);
        assert!(state.enter_construction_level(&group));
        let tree = state.csg.construction_tree();
        let frame = state.construction_frame(0, tree.node(&group).unwrap());
        let origin = frame.to_world(frame.origin);
        let view = state.view_r();
        add(&mut state, &mut Panels::new(Default::default()));
        let pos = state.construction_handle(Some(0)).unwrap().in_scene().0;
        let delta = std::array::from_fn(|i| (pos[i] - origin[i]) / frame.placement.size);
        let screen = gf_core::math::apply(&view, delta);
        for (actual, expected) in screen.into_iter().zip([0.7, -0.35, 0.]) {
            assert!((actual - expected).abs() < 1e-8);
        }
        assert_eq!(state.edit_scope.construction(), [0, 1]);
    }
}
