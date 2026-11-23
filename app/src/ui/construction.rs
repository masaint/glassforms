use super::Panels;
use eframe::egui::{self, Ui};
use gf_core::{
    construction::{CutMaterial, GroupMode, Node, Source},
    csg::{Op, MAX_PARTS},
    join::{Band, Profile},
    state::State,
};

fn rows(ui: &mut Ui, st: &mut State, px: &mut Panels, nodes: &[Node], parent: &[usize], outline: bool) {
    for (i, node) in nodes.iter().enumerate().rev() {
        let mut path = parent.to_vec();
        path.push(i);
        let group = matches!(node.source, Source::Group(_));
        let op = match node.op { Op::Union => "+", Op::Subtract => "−", Op::Intersect => "∩" };
        let label = format!("{op}  {}{}", if group { "▸ " } else { "" }, node.name);
        let response = ui.horizontal(|ui| {
            if outline { ui.add_space(10.0 * parent.len() as f32); }
            ui.selectable_label(st.edit_scope.marked().contains(&path) || st.edit_scope.construction() == path, label)
        }).inner;
        px.note(format!("construction:{path:?}"), response.rect);
        if response.clicked() {
            st.select_construction_node(&path, ui.input(|i| i.modifiers.shift || i.modifiers.command));
            px.handles.pick(None);
        }
        if response.double_clicked() {
            if group { st.enter_construction_level(&path); }
            else { st.csg.edit_construction(); if let Some(o) = &mut st.edit_scope.inside { o.isolated = Some(path.clone()); } st.refresh_scope(); }
        }
        if outline {
            if let Source::Group(children) = &node.source { rows(ui, st, px, children, &path, true); }
        }
    }
}

pub(super) fn hierarchy(ui: &mut Ui, st: &mut State, px: &mut Panels, outline: bool) {
    let tree = st.csg.construction_tree();
    let level = if outline { vec![] } else { st.edit_scope.construction_level().to_vec() };
    if let Some(children) = tree.children(&level) { rows(ui, st, px, children, &level, outline); }
    let path = st.edit_scope.construction().to_vec();
    ui.horizontal_wrapped(|ui| {
        let b = ui.add_enabled(st.csg.parts.len() + 1 < MAX_PARTS, egui::Button::new("+ part"));
        px.note("csg add", b.rect);
        if b.clicked() { super::csg::add(st, px); if let Some(o) = &mut st.edit_scope.inside { o.marked = vec![o.construction.clone()]; } }
        let parent = path.split_last().map(|(_, p)| p).unwrap_or(&[]);
        let selection: Vec<_> = st.edit_scope.marked().iter().filter_map(|p| p.split_last()).filter(|(_, p)| *p == parent).map(|(&i,_)| i).collect();
        let group = ui.add_enabled(!selection.is_empty() && selection.len() == st.edit_scope.marked().len(), egui::Button::new("Group"));
        px.note("construction:group", group.rect);
        if group.clicked() {
            let name = format!("Group {}", parent.len() + 1);
            if let Some(p) = st.csg.edit_construction().group(parent, &selection, name) { st.select_construction_node(&p, false); st.refresh_scope(); }
        }
        if matches!(tree.node(&path).map(|n| &n.source), Some(Source::Group(_))) {
            if st.edit_scope.construction_level() != path {
                let enter = ui.button("Enter group");
                px.note("construction:enter", enter.rect);
                if enter.clicked() { st.enter_construction_level(&path); }
            }
            let enabled = tree.node(&path).is_some_and(Node::can_ungroup);
            let ungroup = ui.add_enabled(enabled, egui::Button::new("Ungroup"))
                .on_disabled_hover_text("Remove the group's own modifiers and material before ungrouping.");
            px.note("construction:ungroup", ungroup.rect);
            if ungroup.clicked() && st.csg.edit_construction().ungroup(&path) {
                st.enter_construction_level(parent);
            }
        }
    });
    if path.is_empty() { return; }
    ui.horizontal_wrapped(|ui| {
        let isolated = st.edit_scope.isolated() == Some(path.as_slice());
        let b = ui.selectable_label(isolated, "Isolate");
        px.note("construction:isolate", b.rect);
        if b.clicked() {
            st.csg.edit_construction();
            if let Some(o) = &mut st.edit_scope.inside { o.isolated = (!isolated).then_some(path.clone()); }
            st.refresh_scope();
        }
        if ui.checkbox(&mut st.edit_scope.preview_parents, "Preview parent modifiers").changed() {
            st.csg.edit_construction(); st.refresh_scope();
        }
    });
}

pub(super) fn panel(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    hierarchy(ui, st, px, false);
    let path = st.edit_scope.construction().to_vec();
    let tree = st.csg.construction_tree();
    let Some(node) = tree.node(&path) else { return; };
    let mut next = node.clone();
    ui.text_edit_singleline(&mut next.name);
    ui.horizontal_wrapped(|ui| {
        for op in Op::ALL {
            let label = match op { Op::Union => "Add", Op::Subtract => "Subtract", Op::Intersect => "Intersect" };
            let b = ui.selectable_label(next.band.is_none() && next.op == op, label);
            px.note(format!("construction:{}", op.id()), b.rect);
            if b.clicked() { next.op = op; next.band = Band::None; }
        }
        // a boolean held within a band of what is below (gf_core::join::band)
        for band in Band::ALL.into_iter().filter(|b| !b.is_none()) {
            let (label, tip) = match band {
                Band::Pocket => ("Pocket", "Cut this out of what is below, no deeper than the depth"),
                Band::Groove => ("Groove", "Cut a channel along this layer's surface, the width either side of it, no deeper than the depth"),
                Band::Ridge => ("Ridge", "Add this to what is below, no higher than the depth above its surface"),
                _ => ("Bead", "Raise a bead along this layer's surface, the width either side of it, no higher than the depth"),
            };
            let b = ui.selectable_label(next.band == band, label).on_hover_text(tip);
            px.note(format!("construction:{}", band.id()), b.rect);
            if b.clicked() {
                next.band = band;
                next.op = band.op().unwrap_or(next.op);
                if next.depth <= 0.0 { next.depth = 0.05; }
                if band.along() && next.width <= 0.0 { next.width = 0.02; }
            }
        }
    });
    if !next.band.is_none() {
        ui.horizontal(|ui| {
            ui.label("Depth");
            let depth = ui.add(egui::DragValue::new(&mut next.depth).speed(0.001).range(0.0..=1.0));
            px.note("construction:depth", depth.rect);
            if next.band.along() {
                ui.label("Width");
                let width = ui.add(egui::DragValue::new(&mut next.width).speed(0.001).range(0.0..=0.5));
                px.note("construction:width", width.rect);
            }
        });
    }
    ui.horizontal(|ui| {
        ui.label("Blend");
        let blend = ui.add(egui::DragValue::new(&mut next.blend).speed(0.002).range(0.0..=2.0));
        px.note("construction:blend", blend.rect);
        let relative = ui.checkbox(&mut next.blend_relative, "Relative");
        px.note("construction:relative", relative.rect);
    });
    // the seam's cross-section over the blend (gf_core::join): numbers in the node's record, no new shape
    ui.horizontal_wrapped(|ui| {
        ui.label("Seam");
        for profile in Profile::ALL {
            let label = match profile { Profile::Fillet => "Fillet", Profile::Smooth => "Smooth", Profile::Round => "Round", Profile::Chamfer => "Chamfer" };
            let b = ui.selectable_value(&mut next.profile, profile, label);
            px.note(format!("construction:seam:{}", profile.id()), b.rect);
        }
    });
    // how a group's children join (gf_core::construction::GroupMode)
    if matches!(next.source, Source::Group(_)) {
        let free = !next.mods.any_live() && next.material.is_none();
        ui.horizontal_wrapped(|ui| {
            ui.label("Children");
            for mode in GroupMode::ALL {
                let (label, tip) = match mode {
                    GroupMode::Isolated => ("Isolated", "Joined to each other, then the group to what is below it as one"),
                    GroupMode::Passthrough => ("Pass through", "Each joined straight into what is below the group, by its own join"),
                    GroupMode::Overridden => ("Overridden", "Each joined straight into what is below the group, by the group's join"),
                };
                let enabled = free || mode == GroupMode::Isolated;
                let b = ui.add_enabled_ui(enabled, |ui| ui.selectable_label(next.mode == mode, label)).inner
                    .on_hover_text(tip)
                    .on_disabled_hover_text("A group with its own modifiers or material joins its children first.");
                px.note(format!("construction:mode:{mode:?}"), b.rect);
                if b.clicked() { next.mode = mode; }
            }
        });
    }
    if next.op == Op::Subtract {
        ui.horizontal(|ui| {
            ui.label("Cut material");
            for (value, label) in [(CutMaterial::Subtractor, "Subtractor"), (CutMaterial::Recipient, "Recipient")] {
                let option = ui.selectable_value(&mut next.cut_material, value, label);
                px.note(format!("construction:cut:{label}"), option.rect);
            }
        });
    }
    let label = if node.material.is_some() { "edit layer material…" } else { "assign layer material…" };
    let edit = ui.button(label).on_hover_text("open this construction layer in the materials lab");
    px.note("construction:material-lab", edit.rect);
    if edit.clicked() { super::material::open_construction(st, px); }
    if &next != node {
        if let Some(slot) = st.csg.edit_construction().node_mut(&path) { *slot = next; st.refresh_scope(); }
    }
    let (&index, parent) = path.split_last().unwrap();
    let count = tree.children(parent).map_or(0, |n| n.len());
    ui.horizontal_wrapped(|ui| {
        for (label, dest) in [("Raise", index.checked_add(1).filter(|n| *n < count)), ("Lower", index.checked_sub(1))] {
            let change_order = ui.add_enabled(dest.is_some(), egui::Button::new(label));
            px.note(format!("construction:{label}"), change_order.rect);
            if change_order.clicked() {
                if let Some(children) = st.csg.edit_construction().children_mut(parent) {
                    let other = dest.unwrap();
                    children.swap(index, other);
                    let mut p = parent.to_vec(); p.push(other); st.select_construction_node(&p, false); st.refresh_scope();
                }
            }
        }
        if ui.button("Remove layer").clicked() && st.csg.remove_construction_node(&path) {
            st.enter_construction_level(parent);
        }
    });
    if path != st.edit_scope.construction() { return; }
    if !st.edit_scope.construction().is_empty() {
        let edit = ui.selectable_label(px.handles.construction_origin, "edit layer origin");
        px.note("construction:edit-origin", edit.rect);
        if edit.clicked() {
            let on = !px.handles.construction_origin;
            px.handles.set_tool(super::handles::Tool::Move);
            px.handles.pick(None);
            px.handles.construction_origin = on;
        }
        if px.handles.construction_origin {
            let frame = st.construction_frame(st.stack.sel, node);
            let mut world = frame.to_world(frame.origin);
            ui.label("Origin in world coordinates");
            let mut changed = false;
            ui.horizontal(|ui| {
                for (axis, value) in ["X", "Y", "Z"].into_iter().zip(&mut world) {
                    let field = ui.add(egui::DragValue::new(value).speed(0.005).prefix(format!("{axis} ")));
                    px.note(format!("construction:origin:{axis}"), field.rect);
                    changed |= field.changed();
                }
            });
            if changed { st.set_construction_origin(world); }
        }
    }
    if let Some(i) = st.csg.sel { super::csg::primitive(ui, st, i); }
    else if !st.edit_scope.construction().is_empty() {
        ui.collapsing("Layer transform", |ui| super::csg::placement_rows(ui, st, super::handles::Target::ConstructionNode));
    }
}
