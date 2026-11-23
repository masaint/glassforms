//! Editor navigation is transient. A setup records its authored objects and groups.
use crate::build::BuildRequest;
use crate::{
    groups::{Hierarchy, Item, Node},
    layers::{mask_of, Op},
    state::State,
};

/// Where the editor is. Construction-level navigation exists only inside an
/// object, so it lives in [`ObjectScope`]: leaving or losing the object drops it.
#[derive(Clone, Debug)]
pub struct EditScope {
    pub world: Option<u32>,
    pub selection: Vec<Item>,
    pub selection_cleared: bool,
    /// The object being edited, with its construction navigation.
    pub inside: Option<ObjectScope>,
    pub preview_parents: bool,
}

/// Editing one object (its layer index), and where in its construction tree.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ObjectScope {
    pub index: usize,
    /// The world level to go back to on leaving.
    pub return_world: Option<u32>,
    /// The construction group whose immediate children are being edited.
    pub construction_level: Vec<usize>,
    /// Selected layer, or the current group when its own settings are in view.
    pub construction: Vec<usize>,
    pub isolated: Option<Vec<usize>>,
    pub marked: Vec<Vec<usize>>,
}

impl ObjectScope {
    pub fn new(index: usize) -> Self { Self { index, ..Default::default() } }
}

impl Default for EditScope {
    fn default() -> Self {
        Self { world: None, selection: Vec::new(), selection_cleared: false, inside: None, preview_parents: true }
    }
}

impl EditScope {
    /// The object being edited.
    pub fn object(&self) -> Option<usize> { self.inside.as_ref().map(|o| o.index) }
    pub fn construction_level(&self) -> &[usize] { self.inside.as_ref().map_or(&[], |o| &o.construction_level) }
    pub fn construction(&self) -> &[usize] { self.inside.as_ref().map_or(&[], |o| &o.construction) }
    pub fn isolated(&self) -> Option<&[usize]> { self.inside.as_ref().and_then(|o| o.isolated.as_deref()) }
    pub fn marked(&self) -> &[Vec<usize>] { self.inside.as_ref().map_or(&[], |o| &o.marked) }
    /// Stop editing the object (it was deleted or is no longer valid): back to the
    /// world level it was entered from, if that group still exists.
    pub fn drop_object(&mut self, group_exists: impl Fn(u32) -> bool) {
        if let Some(o) = self.inside.take() { self.world = o.return_world.filter(|&id| group_exists(id)); }
    }
}

impl Hierarchy {
    /// Enclosing group IDs in order from the scene root.
    pub fn ancestors(&self, item: Item) -> Option<Vec<u32>> {
        let mut path = Vec::new();
        for (depth, node) in self.rows() {
            path.truncate(depth);
            if node.item() == item { return Some(path); }
            if let Node::Group(g) = node { path.push(g.id); }
        }
        None
    }
}

impl State {
    pub fn world_selection(&self) -> Vec<Item> {
        if self.edit_scope.selection_cleared || !self.edit_scope.selection.is_empty() { return self.edit_scope.selection.clone(); }
        self.selected_group.map(Item::Group).or_else(|| self.world_pick(self.stack.sel)).into_iter().collect()
    }

    pub fn select_world_item(&mut self, item: Item, toggle: bool) {
        if !toggle { self.edit_scope.selection.clear(); }
        if toggle && self.edit_scope.selection.is_empty() {
            self.edit_scope.selection = self.world_selection();
        }
        if let Some(i) = self.edit_scope.selection.iter().position(|v| *v == item) {
            if toggle { self.edit_scope.selection.remove(i); }
        } else { self.edit_scope.selection.push(item); }
        self.edit_scope.selection_cleared = self.edit_scope.selection.is_empty();
        match self.edit_scope.selection.last().copied() {
            Some(Item::Group(id)) => self.selected_group = Some(id),
            Some(Item::Object(i)) => self.select_layer(i),
            None => self.selected_group = None,
        }
    }

    pub fn clear_world_selection(&mut self) {
        self.edit_scope.selection.clear();
        self.edit_scope.selection_cleared = true;
        self.selected_group = None;
    }

    pub fn new_world_group(&mut self, parent: Option<u32>) -> Option<u32> {
        let id = self.new_group(parent)?;
        self.stack.hierarchy.as_mut()?.group_mut(id)?.world = true;
        Some(id)
    }

    /// Hit-testing returns a leaf. At a world level the selectable unit is its
    /// immediately enclosing child of that level, even deep inside an assembly.
    pub fn world_pick(&self, layer: usize) -> Option<Item> {
        let path = self.outline().ancestors(Item::Object(layer))?;
        let depth = match self.edit_scope.world {
            Some(id) => path.iter().position(|g| *g == id)? + 1,
            None => 0,
        };
        Some(path.get(depth).copied().map(Item::Group).unwrap_or(Item::Object(layer)))
    }

    pub fn enter_world_item(&mut self, item: Item) -> bool {
        let Some(path) = self.outline().ancestors(item) else { return false };
        match item {
            Item::Group(id) => self.world_level(Some(id)),
            Item::Object(i) => {
                self.edit_scope.selection.clear();
                self.select_layer(i);
                let return_world = match &self.edit_scope.inside { Some(o) => o.return_world, None => self.edit_scope.world };
                self.edit_scope.world = path.last().copied();
                self.edit_scope.inside = Some(ObjectScope { return_world, ..ObjectScope::new(i) });
                self.csg.sel = None;
                self.csg.whole = true;
                self.selected_group = None;
                self.refresh_scope();
                true
            }
        }
    }

    pub fn world_level(&mut self, group: Option<u32>) -> bool {
        if group.is_some_and(|id| self.outline().group(id).is_none()) { return false; }
        self.edit_scope.selection.clear();
        self.edit_scope.selection_cleared = false;
        self.edit_scope.world = group;
        self.edit_scope.inside = None;
        self.selected_group = None;
        let outline = self.outline();
        let children = group.and_then(|id| outline.group(id)).map_or(outline.0.as_slice(), |g| g.children.as_slice());
        if let Some(item) = self.world_pick(self.stack.sel).or_else(|| children.first().map(Node::item)) {
            self.select_world_item(item, false);
        }
        self.refresh_scope();
        true
    }

    pub fn enter_construction_level(&mut self, path: &[usize]) -> bool {
        if self.csg.construction_tree().children(path).is_none() { return false; }
        let Some(scope) = self.edit_scope.inside.as_mut() else { return false };
        scope.construction_level = path.to_vec();
        scope.construction = path.to_vec();
        scope.marked.clear();
        scope.isolated = None;
        self.csg.sel = None;
        self.csg.whole = path.is_empty();
        self.refresh_scope();
        true
    }

    pub fn select_construction_node(&mut self, path: &[usize], toggle: bool) -> bool {
        use crate::construction::Source;
        let Some((_, parent)) = path.split_last() else { return false };
        let tree = self.csg.construction_tree();
        if tree.node(path).is_none() { return false; }
        let preview_parents = self.edit_scope.preview_parents;
        let Some(scope) = self.edit_scope.inside.as_mut() else { return false };
        if !toggle || scope.construction_level != parent { scope.marked.clear(); }
        if let Some(index) = scope.marked.iter().position(|p| p == path) {
            scope.marked.remove(index);
        } else { scope.marked.push(path.to_vec()); }
        scope.construction_level = parent.to_vec();
        scope.construction = scope.marked.last().cloned().unwrap_or_else(|| parent.to_vec());
        self.csg.sel = tree.node(&scope.construction).and_then(|n| match n.source {
            Source::Part(id) => self.csg.construction_part(id), _ => None,
        });
        self.csg.whole = scope.construction.is_empty();
        if scope.isolated.is_some() { scope.isolated = (!scope.construction.is_empty()).then(|| scope.construction.clone()); }
        if scope.isolated.is_some() || !preview_parents || scope.marked.is_empty() { self.refresh_scope(); }
        true
    }

    /// Inside an object: no construction layer or part selected, at the level being edited (the
    /// level's own settings in view, as on entering it).
    pub fn clear_construction_selection(&mut self) {
        let Some(scope) = self.edit_scope.inside.as_mut() else { return };
        let changed = !scope.marked.is_empty() || scope.construction != scope.construction_level;
        scope.marked.clear();
        scope.construction = scope.construction_level.clone();
        if scope.isolated.is_some() { scope.isolated = (!scope.construction.is_empty()).then(|| scope.construction.clone()); }
        self.csg.sel = None;
        self.csg.whole = scope.construction.is_empty();
        if changed { self.refresh_scope(); }
    }

    pub fn leave_object(&mut self) {
        let Some((i, return_world)) = self.edit_scope.inside.as_ref().map(|o| (o.index, o.return_world)) else { return };
        // (a group entered from can be gone by now: then the root)
        if !self.world_level(return_world) { self.world_level(None); }
        if let Some(item) = self.world_pick(i) { self.select_world_item(item, false); }
    }

    pub fn scope_up(&mut self) {
        if let Some((_, parent)) = self.edit_scope.construction_level().split_last() {
            let (left, parent) = (self.edit_scope.construction_level().to_vec(), parent.to_vec());
            self.enter_construction_level(&parent);
            self.select_construction_node(&left, false);
        } else if self.edit_scope.inside.is_some() {
            self.leave_object();
        } else if let Some(id) = self.edit_scope.world {
            let parent = self.outline().ancestors(Item::Group(id)).and_then(|p| p.last().copied());
            self.world_level(parent);
            self.select_world_item(Item::Group(id), false);
        }
    }

    pub fn refresh_scope(&mut self) {
        self.request(BuildRequest::edit());
        self.cut = true;
    }

    /// The new object is a separate material body in the scene. Older setups
    /// that explicitly authored layer joins retain their recorded operations.
    pub fn create_world_object(&mut self, family: usize, preset: usize, csg: bool, parent: Option<u32>) -> Result<usize, String> {
        if let Some(id) = parent {
            let lock = self.outline().kind_for_group(id).ok_or("scene group no longer exists")?;
            let kind = if csg { crate::sdf::Trace::Csg } else { crate::sdf::Trace::Volume };
            if lock.is_some_and(|expected| expected != kind) { return Err("object conflicts with the group type lock".into()); }
        }
        let new = self.create_object(family, preset, csg).ok_or("could not add another scene object")?;
        for i in 0..self.object_count() {
            if i != new {
                let pair = self.stack.pair_mut(mask_of(&[i, new])).expect("two existing scene objects");
                pair.op = Op::Apart;
                pair.radius = 0.0;
            }
        }
        if let Some(id) = parent { self.move_item(Item::Object(new), crate::groups::DropAt::Inside(id))?; }
        let selected = self.stack.sel;
        self.material_owners_changed();
        self.select_world_item(Item::Object(selected), false);
        Ok(selected)
    }
}
