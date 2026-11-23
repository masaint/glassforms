//! The ordered object outline. A group owns its children and its modifier stack.
//! Object references follow the layer remapping used by animation tracks and body materials.

use crate::{mods::Mods, sdf::Trace};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Item {
    Object(usize),
    Group(u32),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Node {
    Object(usize),
    Group(Group),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Group {
    pub id: u32,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub world: bool,
    #[serde(default, skip_serializing_if = "crate::layers::Placement::is_identity")]
    pub frame: crate::layers::Placement,
    /// The group's reference point as a member of an enclosing arrangement,
    /// expressed in its editable frame.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arrangement_anchor: Option<[f64; 3]>,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<Trace>,
    #[serde(default, skip_serializing_if = "GroupOp::uses_joins")]
    pub operation: GroupOp,
    #[serde(default, skip_serializing_if = "Mods::is_empty")]
    pub mods: Mods,
    #[serde(default, skip_serializing_if = "Mods::is_empty")]
    pub world_mods: Mods,
    pub children: Vec<Node>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GroupOp {
    #[default]
    Joins,
    Union,
    Subtract,
    Intersect,
}

impl GroupOp {
    fn uses_joins(&self) -> bool { *self == Self::Joins }
}

impl Node {
    pub fn item(&self) -> Item {
        match self { Self::Object(i) => Item::Object(*i), Self::Group(g) => Item::Group(g.id) }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum DropAt {
    Before(Item),
    After(Item),
    Inside(u32),
    Root,
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Hierarchy(pub Vec<Node>);

fn group_mut(nodes: &mut [Node], id: u32) -> Option<&mut Group> {
    for node in nodes {
        if let Node::Group(g) = node {
            if g.id == id { return Some(g); }
            if let Some(found) = group_mut(&mut g.children, id) { return Some(found); }
        }
    }
    None
}

fn siblings(nodes: &mut Vec<Node>, item: Item) -> Option<(&mut Vec<Node>, usize)> {
    if let Some(i) = nodes.iter().position(|n| n.item() == item) { return Some((nodes, i)); }
    for n in nodes {
        if let Node::Group(g) = n {
            if let Some(found) = siblings(&mut g.children, item) { return Some(found); }
        }
    }
    None
}

impl Hierarchy {
    pub fn flat(count: usize) -> Self { Self((0..count).map(Node::Object).collect()) }

    pub fn changes_geometry(&self) -> bool {
        fn active(nodes: &[Node]) -> bool {
            nodes.iter().any(|n| match n {
                Node::Group(g) => !g.children.is_empty() && (g.operation != GroupOp::Joins || g.mods.any_live() || active(&g.children)),
                Node::Object(_) => false,
            })
        }
        active(&self.0)
    }

    pub fn group(&self, id: u32) -> Option<&Group> {
        fn find(nodes: &[Node], id: u32) -> Option<&Group> {
            for n in nodes {
                if let Node::Group(g) = n {
                    if g.id == id { return Some(g); }
                    if let Some(child) = find(&g.children, id) { return Some(child); }
                }
            }
            None
        }
        find(&self.0, id)
    }

    /// Depth-first rows in drawing and evaluation order.
    pub fn rows(&self) -> Vec<(usize, &Node)> {
        fn walk<'a>(out: &mut Vec<(usize, &'a Node)>, nodes: &'a [Node], depth: usize) {
            for n in nodes {
                out.push((depth, n));
                if let Node::Group(g) = n { walk(out, &g.children, depth + 1); }
            }
        }
        let mut out = Vec::new();
        walk(&mut out, &self.0, 0);
        out
    }

    pub fn group_mut(&mut self, id: u32) -> Option<&mut Group> { group_mut(&mut self.0, id) }

    /// Include locks on every enclosing group when choosing the type of a new part.
    pub fn kind_for_group(&self, id: u32) -> Option<Option<Trace>> {
        let mut locks = Vec::new();
        for (depth, node) in self.rows() {
            locks.truncate(depth);
            if let Node::Group(group) = node {
                let kind = group.kind.or(locks.last().copied().flatten());
                if group.id == id { return Some(kind); }
                locks.push(kind);
            }
        }
        None
    }

    /// Modifier order for an individual leaf is innermost group first.
    fn enclosing_modifiers(&self, layer: usize) -> Vec<&Mods> {
        let mut chain = Vec::new();
        for (depth, node) in self.rows() {
            chain.truncate(depth);
            match node {
                Node::Group(g) => chain.push(&g.mods),
                Node::Object(i) if *i == layer => { chain.reverse(); return chain; }
                _ => {}
            }
        }
        Vec::new()
    }

    pub fn create_group(&mut self, parent: Option<u32>) -> Option<u32> {
        let id = self.rows().into_iter().filter_map(|(_, n)| match n {
            Node::Group(g) => Some(g.id), _ => None,
        }).max().unwrap_or(0).checked_add(1)?;
        let children = match parent { Some(p) => &mut self.group_mut(p)?.children, None => &mut self.0 };
        children.push(Node::Group(Group { id, world: false, frame: Default::default(), arrangement_anchor: None, name: format!("Group {id}"), kind: None, operation: GroupOp::Joins, mods: Mods::default(), world_mods: Mods::default(), children: Vec::new() }));
        Some(id)
    }

    pub fn ungroup(&mut self, id: u32) -> bool {
        let Some((nodes, at)) = siblings(&mut self.0, Item::Group(id)) else { return false };
        let Node::Group(group) = nodes.remove(at) else { unreachable!() };
        nodes.splice(at..at, group.children);
        true
    }

    /// Stage the entire move so an invalid target, descendant drop or type conflict
    /// leaves both the original parent and the sibling order intact.
    pub fn move_to(&mut self, item: Item, dest: DropAt, kinds: &[Trace]) -> Result<(), String> {
        let mut next = self.clone();
        let (from, at) = siblings(&mut next.0, item).ok_or("item no longer exists")?;
        let node = from.remove(at);
        match dest {
            DropAt::Root => next.0.push(node),
            DropAt::Inside(id) => next.group_mut(id).ok_or("cannot move a group into itself or a descendant")?.children.push(node),
            DropAt::Before(target) | DropAt::After(target) => {
                let (to, at) = siblings(&mut next.0, target).ok_or("cannot drop relative to this item")?;
                to.insert(at + usize::from(matches!(dest, DropAt::After(_))), node);
            }
        }
        next.validate(kinds)?;
        *self = next;
        Ok(())
    }

    pub fn set_kind(&mut self, id: u32, kind: Option<Trace>, kinds: &[Trace]) -> Result<(), String> {
        let group = self.group_mut(id).ok_or("group no longer exists")?;
        let old = group.kind;
        group.kind = kind;
        if let Err(error) = self.validate(kinds) {
            self.group_mut(id).unwrap().kind = old;
            return Err(error);
        }
        Ok(())
    }

    pub fn validate(&self, kinds: &[Trace]) -> Result<(), String> {
        let mut ids = BTreeSet::new();
        let mut found = vec![false; kinds.len()];
        let mut locks = Vec::new();
        for (depth, node) in self.rows() {
            locks.truncate(depth);
            let parent = locks.last().copied().flatten();
            match node {
                Node::Object(i) => {
                    let seen = found.get_mut(*i).ok_or("group refers to a missing object")?;
                    if *seen { return Err("an object occurs twice in the layer hierarchy".into()); }
                    if parent.is_some_and(|k| k != kinds[*i]) { return Err("object conflicts with the group type lock".into()); }
                    *seen = true;
                }
                Node::Group(g) => {
                    if !ids.insert(g.id) { return Err("duplicate layer group ID".into()); }
                    if parent.is_some() && g.kind.is_some() && parent != g.kind { return Err("nested groups have conflicting type locks".into()); }
                    locks.push(g.kind.or(parent));
                }
            }
        }
        if found.contains(&false) { return Err("an object is missing from the layer hierarchy".into()); }
        Ok(())
    }

    pub(crate) fn remap(&mut self, to: &[Option<usize>]) {
        fn walk(nodes: &mut Vec<Node>, to: &[Option<usize>]) {
            nodes.retain_mut(|n| match n {
                Node::Object(i) => match to.get(*i).copied().flatten() {
                    Some(j) => { *i = j; true }, None => false,
                },
                Node::Group(g) => { walk(&mut g.children, to); true },
            });
        }
        walk(&mut self.0, to);
    }
}

impl crate::state::State {
    /// The exact tree tracer can evaluate a single object's enclosing groups in
    /// one stack when their coordinate frames coincide and all entries fit.
    pub fn single_group_tree_mods(&self) -> Option<Mods> {
        if self.edit_scope.object().is_some() || self.object_count() != 1 || !self.stack.plain() || !self.is_csg_object(0) || !self.csg.place.is_identity() { return None; }
        let h = self.stack.hierarchy.as_ref()?;
        if !h.changes_geometry() { return None; }
        let mut combined = self.layer_mods(0)?.clone();
        for outer in h.enclosing_modifiers(0) { combined.list.extend(outer.live().cloned()); }
        if combined.live().count() > crate::mods::MAX_MODS || !combined.exact(&crate::mods::whole_box()) { return None; }
        Some(combined)
    }
    pub fn selected_modifiers(&self) -> &Mods {
        if self.edit_scope.object().is_some() && !self.edit_scope.construction().is_empty() {
            static EMPTY: std::sync::LazyLock<Mods> = std::sync::LazyLock::new(Mods::default);
            return self.csg.construction.as_ref().and_then(|t| t.node(&self.edit_scope.construction()))
                .map_or(&EMPTY, |node| &node.mods);
        }
        if let Some(item) = self.world_modifier_target {
            if !matches!(item, Item::Object(i) if i >= self.object_count()) {
                static EMPTY: Mods = Mods { list: Vec::new() };
                return self.stack.world_modifiers(item).unwrap_or(&EMPTY);
            }
        }
        if let Some(g) = self.selected_group.and_then(|id| self.stack.hierarchy.as_ref()?.group(id)) {
            return &g.mods;
        }
        &self.shape().mods
    }

    pub fn selected_modifiers_mut(&mut self) -> &mut Mods {
        if self.edit_scope.object().is_some() && !self.edit_scope.construction().is_empty()
            && self.csg.construction_tree().node(&self.edit_scope.construction()).is_some() {
            return &mut self.csg.edit_construction().node_mut(&self.edit_scope.construction()).unwrap().mods;
        }
        if let Some(item) = self.world_modifier_target {
            let valid = match item {
                Item::Object(i) => i < self.object_count(),
                Item::Group(id) => self.stack.hierarchy.as_ref().is_some_and(|h| h.group(id).is_some()),
            };
            if valid {
                return self.stack.world_modifiers_mut(item).expect("validated World modifier owner");
            }
            self.world_modifier_target = None;
        }
        if let Some(g) = self.selected_group.and_then(|id| self.stack.hierarchy.as_mut()?.group_mut(id)) {
            return &mut g.mods;
        }
        &mut self.shapes[self.fam].mods
    }
    pub fn outline(&self) -> Hierarchy {
        self.stack.hierarchy.clone().unwrap_or_else(|| Hierarchy::flat(self.object_count()))
    }

    fn object_kinds(&self) -> Vec<Trace> {
        (0..self.object_count()).map(|i| self.layer_shape(i).unwrap().trace).collect()
    }

    pub fn permits_object_kind(&self, i: usize, kind: Trace) -> bool {
        let Some(h) = &self.stack.hierarchy else { return true };
        let mut kinds = self.object_kinds();
        let Some(k) = kinds.get_mut(i) else { return false };
        *k = kind;
        h.validate(&kinds).is_ok()
    }

    pub fn new_group(&mut self, parent: Option<u32>) -> Option<u32> {
        let mut outline = self.outline();
        let id = outline.create_group(parent)?;
        self.stack.hierarchy = Some(outline);
        Some(id)
    }

    pub fn set_group_kind(&mut self, id: u32, kind: Option<Trace>) -> Result<(), String> {
        let kinds = self.object_kinds();
        self.stack.hierarchy.as_mut().ok_or("group no longer exists")?.set_kind(id, kind, &kinds)
    }

    pub fn move_item(&mut self, item: Item, target: DropAt) -> Result<(), String> {
        let mut outline = self.outline();
        outline.move_to(item, target, &self.object_kinds())?;
        self.apply_outline_order(outline);
        Ok(())
    }

    /// Group the selected branches at their closest common World level. Build
    /// the new outline before renumbering layers so every selected item keeps
    /// its identity, even when an earlier move changes depth-first order.
    pub fn group_world_selection(&mut self) -> Result<u32, String> {
        let selection = self.world_selection();
        if self.edit_scope.object().is_some() || selection.len() < 2 {
            return Err("Select two or more World items to group".into());
        }
        let mut outline = self.outline();
        let mut paths = selection.iter().map(|item| outline.ancestors(*item)
            .ok_or_else(|| "A selected item no longer exists".to_string()));
        let mut common = paths.next().unwrap()?;
        for path in paths {
            let path = path?;
            common.truncate(common.iter().zip(path.iter()).take_while(|(a, b)| a == b).count());
        }
        let parent = common.last().copied();
        let mut members = Vec::new();
        let mut selected_depth = None;
        for (depth, node) in outline.rows() {
            if selected_depth.is_some_and(|d| depth > d) { continue; }
            selected_depth = None;
            if selection.contains(&node.item()) {
                members.push(node.item());
                selected_depth = Some(depth);
            }
        }
        let first = members[0];
        let path = outline.ancestors(first).unwrap();
        let insert_before = path.get(common.len()).copied().map(Item::Group).unwrap_or(first);
        let id = outline.create_group(parent).ok_or("Could not create the group")?;
        let kinds = self.object_kinds();
        outline.move_to(Item::Group(id), DropAt::Before(insert_before), &kinds)?;
        for item in members {
            outline.move_to(item, DropAt::Inside(id), &kinds)?;
        }
        // A newly grouped selection turns about its own bounds. Later changes
        // and modifier-generated copies leave this authored frame in place.
        let bounds = self.placed_bounds(Item::Group(id), &outline, None);
        let group = outline.group_mut(id).unwrap();
        group.world = true;
        if let Some(b) = bounds {
            group.frame.pos = std::array::from_fn(|i| 0.5 * (b[i] + b[i + 3]));
        }
        self.apply_outline_order(outline);
        self.world_level(parent);
        self.select_world_item(Item::Group(id), false);
        Ok(id)
    }

    fn apply_outline_order(&mut self, outline: Hierarchy) {
        let order: Vec<usize> = outline.rows().iter().filter_map(|(_, n)| match n {
            Node::Object(i) => Some(*i), _ => None,
        }).collect();
        // A hierarchy edit can move a different shape family into slot zero.
        // Keep the studio that lit the scene while moving its authored objects.
        if let Some(&first) = order.first() {
            self.preserve_rig_for_first_family(self.layer_fam(first));
        }
        self.stack.hierarchy = Some(outline);
        // Move placements, geometry, animation tracks and materials together into
        // depth-first order. swap_layers updates the outline's object indices too.
        let mut current: Vec<usize> = (0..order.len()).collect();
        for (i, object) in order.iter().enumerate() {
            let j = current.iter().position(|v| v == object).unwrap();
            self.swap_layers(i, j);
            current.swap(i, j);
        }
        self.request(crate::build::BuildRequest::edit());
        self.cut = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::State;

    #[test]
    fn nesting_reordering_and_locks_round_trip() {
        let mut s = State::with_args(&[], true);
        let first_title = s.layer_title(0);
        let studio = s.families.iter().position(|f| f.id == "studio").unwrap();
        s.create_object(studio, 0, true).unwrap();
        let a = s.new_group(None).unwrap();
        let b = s.new_group(Some(a)).unwrap();
        s.set_group_kind(a, Some(Trace::Csg)).unwrap();
        s.move_item(Item::Object(1), DropAt::Inside(b)).unwrap();
        let saved = s.setup();
        assert!(s.move_item(Item::Group(a), DropAt::Inside(b)).is_err());
        assert!(s.move_item(Item::Object(0), DropAt::Inside(b)).is_err());
        assert!(s.set_group_kind(b, Some(Trace::Volume)).is_err());
        assert_eq!(s.setup(), saved);
        s.move_item(Item::Group(a), DropAt::Before(Item::Object(0))).unwrap();
        assert_eq!(s.layer_title(1), first_title);
        assert!(s.is_csg_object(0));
        s.select_layer(0);
        assert!(!s.convert_to_volume());
        assert!(!s.apply_arg("trace", "volume"));
        let saved = s.setup_json();
        let mut reopened = State::with_args(&[], true);
        reopened.apply_setup_json(&saved).unwrap();
        assert_eq!(reopened.outline(), s.outline());
        assert_eq!(reopened.setup_json(), saved);
    }

    #[test]
    fn deleting_the_last_part_preserves_the_empty_group() {
        let mut s = State::with_args(&[], true);
        let a = s.new_group(None).unwrap();
        s.move_item(Item::Object(0), DropAt::Inside(a)).unwrap();
        s.delete_object(0);
        let saved = s.setup_json();
        s.apply_setup_json(&saved).unwrap();
        assert_eq!(s.object_count(), 0);
        assert_eq!(s.outline().rows().len(), 1);
        s.create_object(s.fam, 0, false).unwrap();
        assert!(s.outline().validate(&[Trace::Volume]).is_ok());
    }
}
