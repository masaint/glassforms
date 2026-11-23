//! The World panel supplies the selected authored item. Its stack edits the
//! World arrangement or the incoming geometry of its members.
use super::{mods, widgets::{Item, named_run}, Panels};
use eframe::egui::{Id, Ui};
use gf_core::{groups, mods::{Kind, Modifier}, state::State, world_mods::Scope};

pub fn show(ui: &mut Ui, st: &mut State, px: &mut Panels, target: groups::Item) {
    let present = match target {
        groups::Item::Object(i) => i < st.object_count(),
        groups::Item::Group(g) => st.stack.hierarchy.as_ref().and_then(|h| h.group(g)).is_some(),
    };
    if !present { return; }
    st.world_modifier_target = Some(target);
    mods::show(ui, st, px);
    st.world_modifier_target = None;
}

pub(super) fn supported(kind: Kind) -> bool {
    matches!(kind, Kind::Repeat | Kind::Twist | Kind::Swirl | Kind::Bend | Kind::Taper | Kind::Mirror)
}
pub(super) fn scope_controls(ui: &mut Ui, st: &mut State, i: usize, m: &Modifier) {
    let items = [Item::new("world", m.scope == Scope::World).tip("move whole objects"),
                 Item::new("sub-objects", m.scope == Scope::SubObjects).tip("change the geometry of each object")];
    if let Some(k) = named_run(ui, Id::unique(("world scope", i)), "scope", &items).clicked {
        st.selected_modifiers_mut().list[i].scope = if k == 0 { Scope::World } else { Scope::SubObjects };
        st.sculpt_changed(false);
    }
    let items = [Item::new("off", !m.cascade).tip("treat each child group as one element"),
                 Item::new("on", m.cascade).tip("apply to individual members of child groups")];
    if let Some(k) = named_run(ui, Id::unique(("world cascade", i)), "cascade", &items).clicked {
        st.selected_modifiers_mut().list[i].cascade = k == 1;
        st.sculpt_changed(false);
    }
}
