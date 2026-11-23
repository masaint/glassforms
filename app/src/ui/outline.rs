//! Floating scene hierarchy. Drag a row to its insertion line or onto a group.
use super::{theme::*, widgets::*, Panels};
use eframe::egui::{self, Align2, Id, Pos2, Rect, Sense, Stroke, StrokeKind, Ui};
use gf_core::{groups::{DropAt, GroupOp, Item, Node}, sdf::Trace, state::State};
use std::collections::BTreeSet;

#[derive(Default)]
pub struct Palette {
    pub open: bool,
    pub drag: Option<Item>,
    pub add_to: Option<u32>,
    closed: BTreeSet<u32>,
    error: Option<String>,
}

impl Palette {
    pub fn report(&mut self, error: String) { self.error = Some(error); }
}

fn tag(item: Item) -> String {
    match item { Item::Object(i) => format!("object:{}", i + 1), Item::Group(id) => format!("group:{id}") }
}

fn rows(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let outline = st.outline();
    let (pointer, held) = ui.input(|i| (i.pointer.interact_pos(), i.pointer.any_down() || i.pointer.any_released()));
    // (a carry is the press that started it: with no button held it is over)
    if !held { px.palette.drag = None; }
    let mut target = None;
    let mut hidden_below = None;
    for (depth, node) in outline.rows() {
        if hidden_below.is_some_and(|d| depth > d) { continue; }
        hidden_below = None;
        let key = node.item();
        let active = st.world_selection().contains(&key);
        let (label, detail) = match node {
            Node::Object(i) => (st.layer_title(*i), if st.is_csg_object(*i) { "CSG" } else { "volume" }),
            Node::Group(g) => {
                if px.palette.closed.contains(&g.id) { hidden_below = Some(depth); }
                (g.name.clone(), match g.kind { None => "group", Some(Trace::Csg) => "CSG group", Some(Trace::Volume) => "volume group" })
            }
        };
        let rect = row(ui, 25.0);
        let left = (rect.left() + 12.0 * depth as f32).min(rect.right() - 90.0);
        let disclosure = Rect::from_min_size(Pos2::new(left, rect.top()), egui::vec2(18.0, rect.height()));
        let body = Rect::from_min_max(Pos2::new(disclosure.right(), rect.top()), rect.max);
        let response = ui.interact(body, Id::unique(("outline row", key)), Sense::click_and_drag());
        px.note(format!("outline:{}", tag(key)), body);
        if active || response.hovered() {
            ui.painter().rect_filled(rect, 3, white(if active { 0.15 } else { 0.07 }));
        }
        if let Node::Group(g) = node {
            let open = !px.palette.closed.contains(&g.id);
            if button(ui, disclosure, Id::unique(("outline fold", g.id)), if open { "−" } else { "+" }, false, mono(11.0)).clicked() {
                if open { px.palette.closed.insert(g.id); } else { px.palette.closed.remove(&g.id); }
            }
        }
        if response.double_clicked() {
            st.enter_world_item(key);
            px.palette.error = None;
        } else if response.clicked() {
            st.select_world_item(key, ui.input(|i| i.modifiers.shift || i.modifiers.command));
            px.palette.error = None;
        }
        if response.drag_started() { px.palette.drag = Some(key); }
        let mut offset = 0.0;
        if let (Some(from), Some(at)) = (px.palette.drag, pointer) {
            if from != key && rect.contains(at) {
                let into = matches!(key, Item::Group(_)) && (at.y - rect.center().y).abs() < rect.height() * 0.28;
                if into {
                    if let Item::Group(id) = key { target = Some(DropAt::Inside(id)); }
                    ui.painter().rect_stroke(rect, 3, Stroke::new(1.5, pal().accent), StrokeKind::Inside);
                } else {
                    let before = at.y < rect.center().y;
                    target = Some(if before { DropAt::Before(key) } else { DropAt::After(key) });
                    let y = if before { rect.top() } else { rect.bottom() };
                    ui.painter().hline(left..=rect.right(), y, Stroke::new(2.0, pal().accent));
                    offset = if before { 4.0 } else { -4.0 };
                }
            }
        }
        let offset = ui.ctx().animate_value_with_time(Id::unique(("outline gap", key)), offset, 0.12);
        let label_area = Rect::from_min_max(body.min, Pos2::new((body.right() - 80.0).max(body.left()), body.bottom())).translate(egui::vec2(0.0, offset));
        ui.painter().with_clip_rect(label_area).text(label_area.left_center(), Align2::LEFT_CENTER, label, mono(SIZE), pal().bright);
        text_in(ui, body, Align2::RIGHT_CENTER, detail, mono(SMALL), pal().dim);
    }
    if px.palette.drag.is_some() {
        let root = row(ui, 23.0);
        text_in(ui, root, Align2::CENTER_CENTER, "move to scene root", mono(SMALL), pal().dim);
        if pointer.is_some_and(|p| root.contains(p)) {
            target = Some(DropAt::Root);
            ui.painter().rect_stroke(root, 3, Stroke::new(1.5, pal().accent), StrokeKind::Inside);
        }
        if ui.input(|i| i.pointer.any_released()) {
            if let (Some(source), Some(dest)) = (px.palette.drag.take(), target) {
                px.palette.error = st.move_item(source, dest).err();
            }
        }
    }
}

/// The scene location is shared by the viewport and both hierarchy views.
pub fn breadcrumbs(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let tree = st.outline();
    let mut groups = st.edit_scope.world.and_then(|id| {
        let mut path = tree.ancestors(Item::Group(id))?;
        path.push(id);
        Some(path)
    }).unwrap_or_default();
    ui.horizontal_wrapped(|ui| {
        let root = ui.selectable_label(st.edit_scope.object().is_none() && groups.is_empty(), "World");
        px.note("scope:world", root.rect);
        if root.clicked() { st.world_level(None); }
        for id in groups.drain(..) {
            if let Some(g) = tree.group(id) {
                ui.label("›");
                let b = ui.selectable_label(st.edit_scope.world == Some(id) && st.edit_scope.object().is_none(), &g.name);
                px.note(format!("scope:group:{id}"), b.rect);
                if b.clicked() { st.world_level(Some(id)); }
            }
        }
        if let Some(i) = st.edit_scope.object().filter(|i| *i < st.object_count()) {
            ui.label("›");
            let obj = ui.selectable_label(true, st.layer_title(i));
            px.note("scope:object", obj.rect);
            if obj.clicked() { st.enter_construction_level(&[]); }
            let construction = st.csg.construction_tree();
            for depth in 1..=st.edit_scope.construction_level().len() {
                let path = st.edit_scope.construction_level()[..depth].to_vec();
                let Some(node) = construction.node(&path) else { break; };
                ui.label(">");
                let crumb = ui.small_button(&node.name);
                px.note(format!("scope:construction:{path:?}"), crumb.rect);
                if crumb.clicked() {
                    st.enter_construction_level(&path);
                    break;
                }
            }
            let leave = ui.small_button("exit object");
            px.note("scope:exit", leave.rect);
            if leave.clicked() { st.leave_object(); }
        }
    });
}

pub fn show(ctx: &egui::Context, st: &mut State, px: &mut Panels, full: Rect) {
    // a row carried lands only where the rows are drawn: a frame without them ends the carry
    if !px.palette.open || st.edit_scope.object().is_some() { px.palette.drag = None; }
    if !px.palette.open { return; }
    let first = Pos2::new((full.center().x - 180.0).max(EDGE), (full.top() + 80.0).min(full.center().y));
    let window = floating(ctx, "layer hierarchy", "layers", 342.0, first, (full.height() - 70.0).max(90.0), |ui| {
        if px.rendering { ui.disable(); }
        if st.selected_group.is_some_and(|id| st.stack.hierarchy.as_ref().and_then(|h| h.kind_for_group(id)).is_none()) { st.selected_group = None; }
        breadcrumbs(ui, st, px);
        if st.edit_scope.object().is_some() {
            super::construction::hierarchy(ui, st, px, true);
            return;
        }
        rows(ui, st, px);
        let group_selection = st.world_selection().len() > 1;
        let buttons = ui.horizontal(|ui| (ui.add_enabled(st.object_count() < gf_core::layers::MAX_LAYERS, egui::Button::new("+ object")), ui.button(if group_selection { "Group selection" } else { "+ group" }), ui.button("scene root"))).inner;
        px.note("outline:add-object", buttons.0.rect);
        px.note("outline:add-group", buttons.1.rect);
        px.note("outline:root", buttons.2.rect);
        if buttons.2.clicked() { st.world_level(None); }
        if buttons.1.clicked() {
            if group_selection {
                px.palette.error = st.group_world_selection().err();
            } else if let Some(id) = st.new_world_group(st.selected_group.or(st.edit_scope.world)) {
                st.select_world_item(Item::Group(id), false);
                px.palette.error = None;
            }
        }
        if buttons.0.clicked() {
            let target = st.selected_group.or(st.edit_scope.world);
            super::picker::open_add(st, px);
            px.palette.add_to = target;
            px.picker.anchor = Some(buttons.0.rect);
            if let Some(kind) = target.and_then(|id| st.outline().kind_for_group(id)).flatten() {
                px.picker.open = Some(super::picker::Mode::Add { csg: kind == Trace::Csg });
            }
        }
        if let Some(id) = st.selected_group {
            let group = st.stack.hierarchy.as_mut().and_then(|h| h.group_mut(id)).unwrap();
            ui.text_edit_singleline(&mut group.name);
            let kind = group.kind;
            if !group.world {
                let ops = [("joins", GroupOp::Joins), ("union", GroupOp::Union), ("subtract", GroupOp::Subtract), ("intersect", GroupOp::Intersect)];
                let items: Vec<_> = ops.iter().map(|(name, op)| super::widgets::Item::new(name, *op == group.operation)).collect();
                let run = named_run(ui, Id::unique("group operation"), "parts", &items);
                for ((name, _), r) in ops.iter().zip(&run.places) { px.note(format!("outline:{name}"), *r); }
                if let Some(i) = run.clicked {
                    group.operation = ops[i].1;
                    st.request(gf_core::build::BuildRequest::edit());
                    st.cut = true;
                }
            }
            let kinds = [None, Some(Trace::Csg), Some(Trace::Volume)];
            let choices = [super::widgets::Item::new("any", kind.is_none()), super::widgets::Item::new("CSG", kind == kinds[1]), super::widgets::Item::new("volume", kind == kinds[2])];
            let run = named_run(ui, Id::unique("group type lock"), "type lock", &choices);
            for (i, r) in run.places.iter().enumerate() { px.note(format!("outline:lock:{i}"), *r); }
            if let Some(i) = run.clicked { px.palette.error = st.set_group_kind(id, kinds[i]).err(); }
            if ui.button("ungroup").clicked() {
                let parent = st.outline().ancestors(Item::Group(id)).and_then(|p| p.last().copied());
                if let Some(h) = st.stack.hierarchy.as_mut() { h.ungroup(id); }
                let level = if st.edit_scope.world == Some(id) { parent } else { st.edit_scope.world };
                st.world_level(level);
                st.request(gf_core::build::BuildRequest::edit());
            }
        }
        if let Some(error) = &px.palette.error { ui.colored_label(pal().error_text, error); }
    });
    px.note("outline", window.rect);
    px.note("outline:close", window.cross);
    if window.closed { px.palette.open = false; px.palette.drag = None; }
}
