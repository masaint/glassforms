use super::Panels;
use eframe::egui::{self, Ui};
use gf_core::{groups::Node, state::State, world_transform::Reference};

const REFERENCES: [(Reference, &str); 4] = [
    (Reference::Min, "minimum"), (Reference::Centre, "centre"),
    (Reference::Max, "maximum"), (Reference::Origin, "origin"),
];

pub(super) fn contents(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let tree = st.outline();
    let nodes = st.edit_scope.world.and_then(|id| tree.group(id)).map(|g| &g.children).unwrap_or(&tree.0);
    let selected = st.world_selection();
    for node in nodes {
        let item = node.item();
        let text = match node {
            Node::Group(g) => format!("▸  {}", g.name),
            Node::Object(i) => st.layer_title(*i),
        };
        let b = ui.selectable_label(selected.contains(&item), text);
        px.note(format!("world:{item:?}"), b.rect);
        if b.double_clicked() { st.enter_world_item(item); }
        else if b.clicked() { st.select_world_item(item, ui.input(|i| i.modifiers.shift || i.modifiers.command)); px.handles.pick(None); }
    }
    if ui.small_button("+ object").clicked() {
        super::picker::open_add(st, px);
        px.palette.add_to = st.edit_scope.world;
    }
}

pub(super) fn transform(ui: &mut Ui, st: &mut State, px: &mut Panels) {
    let selected = st.world_selection();
    let Some(&item) = selected.last() else { return; };
    let Some(mut frame) = st.world_placement(item) else { return; };
    if let gf_core::groups::Item::Object(i) = item {
        let mut name = st.stack.names.get(&i).cloned().unwrap_or_default();
        ui.horizontal(|ui| {
            ui.label("Name");
            if ui.add(egui::TextEdit::singleline(&mut name).hint_text(st.layer_title(i))).changed() {
                if name.is_empty() { st.stack.names.remove(&i); } else { st.stack.names.insert(i, name); }
            }
        });
    }
    ui.horizontal(|ui| {
        let b = ui.selectable_label(px.handles.move_origin, "Edit origin");
        px.note("world:edit-origin", b.rect);
        if b.clicked() {
            px.handles.move_origin = !px.handles.move_origin;
            px.handles.set_tool(super::handles::Tool::Move);
            px.handles.pivot = None;
            px.handles.pick(None);
        }
        ui.checkbox(&mut px.handles.snap, "Snap");
    });
    if px.handles.snap && !px.handles.move_origin {
        egui::ComboBox::from_label("Snap reference")
            .selected_text(REFERENCES.iter().find(|(r, _)| Some(*r) == px.handles.snap_reference).map_or("automatic", |(_, n)| *n))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut px.handles.snap_reference, None, "automatic");
                for (r, label) in REFERENCES { ui.selectable_value(&mut px.handles.snap_reference, Some(r), label); }
            });
    }
    let moving_origin = px.handles.move_origin;
    let mut changed = false;
    let mut triplet = |ui: &mut Ui, name: &str, values: &mut [f64;3], speed| {
        ui.label(name);
        ui.horizontal(|ui| {
            for (axis, v) in ["X", "Y", "Z"].iter().zip(values) {
                let b = ui.add(egui::DragValue::new(v).speed(speed).prefix(format!("{axis} ")));
                px.note(format!("world:{name}:{axis}"), b.rect);
                changed |= b.changed();
            }
        });
    };
    triplet(ui, if moving_origin { "Origin" } else { "Position" }, &mut frame.pos, 0.005);
    if !moving_origin {
        triplet(ui, "Rotation", &mut frame.rot, 0.25);
        ui.horizontal(|ui| {
            ui.label("Scale");
            let field = ui.add(egui::DragValue::new(&mut frame.size).range(0.01..=100.0).speed(0.005));
            px.note("world:Scale", field.rect);
            changed |= field.changed();
        });
    }
    if changed {
        if px.handles.move_origin { st.set_world_origin(item, frame.pos); }
        else { st.transform_world_items(&selected, item, frame); }
    }
    if selected.len() == 1 {
        if let gf_core::groups::Item::Object(i) = item {
            let animation = ui.collapsing("placement animation", |ui| {
                for knob in st.place_knobs(i) { super::knob::knob_row(ui, st, &knob, px); }
            });
            px.note("world:placement-animation", animation.header_response.rect);
        }
        return;
    }
    egui::ComboBox::from_label("Reference")
        .selected_text(REFERENCES.iter().find(|(r,_)| *r == px.handles.align_reference).unwrap().1)
        .show_ui(ui, |ui| { for (r, name) in REFERENCES { ui.selectable_value(&mut px.handles.align_reference, r, name); } });
    for (label, distribute) in [("Align", false), ("Distribute", true)] {
        ui.horizontal(|ui| {
            ui.label(label);
            for (i, axis) in ["X","Y","Z"].iter().enumerate() {
                let b = ui.add_enabled(!distribute || selected.len() >= 3, egui::Button::new(*axis));
                px.note(format!("world:{label}:{axis}"), b.rect);
                if b.clicked() {
                    if distribute { st.world_distribute(&selected, i, px.handles.align_reference); }
                    else { st.world_align(&selected, i, px.handles.align_reference); }
                }
            }
        });
    }
}
