//! Reference selection and precise coordinates for the existing move gizmo.
use super::{knob, widgets::{Item, named_run, wide_button}, Panels};
use eframe::egui::{Id, Ui};
use gf_core::{modifier_pivot::{Pivot, PivotOwner}, mods::Modifier, state::State};

pub(super) fn controls(ui: &mut Ui, st: &mut State, px: &mut Panels, i: usize, m: &Modifier) {
    let layer = st.stack.sel;
    let group = st.selected_group;
    let owner = st.construction_pivot_owner().unwrap_or_else(|| st.world_modifier_target.map_or(PivotOwner::Object(layer), PivotOwner::World));
    let measured = owner.measured(st, i);
    let frame = owner.frame(st);
    let items = [Item::new("local", m.pivot == Pivot::Local),
        Item::new("mass", matches!(m.pivot, Pivot::CentreOfMass | Pivot::PinnedMass)),
        Item::new("world", m.pivot == Pivot::World)];
    if let Some(k) = named_run(ui, Id::unique(("pivot reference", i)), "pivot", &items).clicked {
        let local = m.pivot_position(frame, measured);
        let e = &mut st.selected_modifiers_mut().list[i];
        match k {
            0 => { e.pivot = Pivot::Local; e.centre = [0.; 3]; }
            1 => { e.pivot = Pivot::CentreOfMass; }
            _ => { e.pivot = Pivot::World; e.centre = frame.to_world(local); }
        }
        px.handles.pivot = None;
        st.sculpt_changed(false);
        return;
    }
    if matches!(m.pivot, Pivot::CentreOfMass | Pivot::PinnedMass) {
        let following = m.pivot == Pivot::CentreOfMass;
        let label = if following { "unlink from mass" } else { "follow mass" };
        let r = ui.add_enabled_ui(!following || measured.is_some(), |ui| wide_button(ui, Id::unique(("mass link", i)), label, false).0);
        if r.inner.clicked() {
            let e = &mut st.selected_modifiers_mut().list[i];
            if following { e.unlink_mass(measured.unwrap()); } else { e.pivot = Pivot::CentreOfMass; }
            st.sculpt_changed(false);
        }
    }
    if m.pivot == Pivot::CentreOfMass { return; }
    for (k, name) in ["centre x", "centre y", "centre z"].iter().enumerate() {
        let v = knob::value_row(ui, px, &format!("mods:{}:{name}", i + 1), name, m.centre[k], -100.0, 100.0, 0.001);
        if let Some(x) = v.value { st.selected_modifiers_mut().list[i].centre[k] = x; st.sculpt_changed(true); }
        if v.released { st.sculpt_changed(false); }
    }
    if wide_button(ui, Id::unique(("pivot recenter", i)), "recenter", false).0.clicked() {
        let e = &mut st.selected_modifiers_mut().list[i];
        match e.pivot {
            Pivot::PinnedMass => e.pivot = Pivot::CentreOfMass,
            Pivot::World => e.centre = frame.to_world(frame.origin),
            _ => e.centre = [0.; 3],
        }
        st.sculpt_changed(false);
    }
    if group.is_none() || st.world_modifier_target.is_some() {
        let on = px.handles.pivot == Some((owner, i));
        if wide_button(ui, Id::unique(("pivot handles", i)), "move pivot in viewport", on).0.clicked() {
            px.handles.set_tool(super::handles::Tool::Move);
            px.handles.pivot = (!on).then_some((owner, i));
        }
    }
}
