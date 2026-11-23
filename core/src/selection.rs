//! What is selected, as an undo entry keeps it beside the setup (`history.rs`).
//!
//! Selection is undone with the rest: undo after a click or a drag-select puts back what was selected
//! before it. Where the editor is (the world level, the object being edited, the construction level
//! shown) is navigation, and stays transient as `edit_scope.rs` says: a selection is put back only at
//! the level it was made at, and moving between levels makes no entry of its own.
//!
//! Indices (the layer, the parts, construction paths) are those of the setup taken at the same moment,
//! and an entry puts back its setup first, so they always refer to what they referred to then.

use crate::groups::Item;
use crate::state::State;

/// Where a selection was made: at a world level (the group shown, none at the root), or inside an object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    World(Option<u32>),
    Object(usize),
}

/// What is selected.
#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    pub level: Level,
    /// the selected layer: the one the panels act on (`stack.sel`)
    pub layer: usize,
    /// world level: the group whose settings are shown, the items chosen, and whether the choice was
    /// emptied on purpose
    pub group: Option<u32>,
    pub items: Vec<Item>,
    pub cleared: bool,
    /// inside an object: the construction level, the node in view and the nodes marked
    pub construction_level: Vec<usize>,
    pub construction: Vec<usize>,
    pub marked: Vec<Vec<usize>>,
    /// the selected layer's parts panel: the part chosen, or the whole
    pub part: Option<usize>,
    pub whole: bool,
}

impl State {
    /// Where the editor is now, as far as selection goes.
    pub fn selection_level(&self) -> Level {
        match self.edit_scope.object() {
            Some(i) => Level::Object(i),
            None => Level::World(self.edit_scope.world),
        }
    }

    /// What is selected now.
    pub fn selection_snapshot(&self) -> Selection {
        let s = &self.edit_scope;
        Selection {
            level: self.selection_level(),
            layer: self.stack.sel,
            group: self.selected_group,
            items: s.selection.clone(),
            cleared: s.selection_cleared,
            construction_level: s.construction_level().to_vec(),
            construction: s.construction().to_vec(),
            marked: s.marked().to_vec(),
            part: self.csg.sel,
            whole: self.csg.whole,
        }
    }

    /// Put back a selection taken earlier, if the editor is at the level it was made at; anything that
    /// is no longer there is left out. True if the selection changed.
    pub fn restore_selection(&mut self, s: &Selection) -> bool {
        if s.level != self.selection_level() { return false; }
        let before = self.selection_snapshot();
        match s.level {
            Level::World(_) => {
                if s.layer < self.layer_count() && s.layer != self.stack.sel { self.select_layer(s.layer); }
                let tree = self.outline();
                self.edit_scope.selection = s.items.iter().copied().filter(|&item| tree.ancestors(item).is_some()).collect();
                self.edit_scope.selection_cleared = s.cleared;
                self.selected_group = s.group.filter(|&id| tree.group(id).is_some());
            }
            Level::Object(_) => {
                let tree = self.csg.construction_tree();
                let Some(o) = self.edit_scope.inside.as_mut() else { return false };
                o.construction_level = s.construction_level.clone();
                while !o.construction_level.is_empty() && tree.children(&o.construction_level).is_none() { o.construction_level.pop(); }
                o.construction = s.construction.clone();
                while !o.construction.is_empty() && tree.node(&o.construction).is_none() { o.construction.pop(); }
                o.marked = s.marked.iter().filter(|p| tree.node(p).is_some()).cloned().collect();
                if o.isolated.is_some() { o.isolated = (!o.construction.is_empty()).then(|| o.construction.clone()); }
            }
        }
        if s.layer == self.stack.sel {
            self.csg.sel = s.part.filter(|&i| i < self.csg.parts.len());
            self.csg.whole = s.whole;
        }
        let now = self.selection_snapshot();
        // (inside an object what is shown follows the selection: the preview is made again)
        if now != before && matches!(s.level, Level::Object(_)) { self.refresh_scope(); }
        now != before
    }
}
