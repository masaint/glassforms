//! An object's ordered, editable construction. The first child is the lowest layer.
use crate::csg::{Csg, Op, Part};
use crate::mods::Mods;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

/// Material settings use the same names and units as the scene's body materials.
pub type Material = BTreeMap<String, f64>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CutMaterial {
    #[default]
    Subtractor,
    Recipient,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    #[default]
    Base,
    /// Stable identity of a primitive in this object's CSG part table.
    Part(u64),
    Group(Vec<Node>),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Node {
    pub name: String,
    pub source: Source,
    /// Transform this layer in its parent's construction space.
    #[serde(skip_serializing_if = "crate::layers::Placement::is_identity")]
    pub place: crate::layers::Placement,
    /// The origin in this layer's geometry coordinates. Older setups may inherit
    /// the object's modifier reference until the artist assigns one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_origin: Option<[f64; 3]>,
    pub op: Op,
    pub blend: f64,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub blend_relative: bool,
    /// the seam's cross-section over the blend (gf_core::join)
    #[serde(skip_serializing_if = "crate::join::Profile::is_default")]
    pub profile: crate::join::Profile,
    /// joined as a boolean held within a band of what is below (gf_core::join::band): its depth, and its
    /// width either side of this node's surface (groove, bead); in the scene's units, or with `blend_relative`
    /// as a share of the node's size
    #[serde(skip_serializing_if = "crate::join::Band::is_none")]
    pub band: crate::join::Band,
    #[serde(skip_serializing_if = "is_zero")]
    pub depth: f64,
    #[serde(skip_serializing_if = "is_zero")]
    pub width: f64,
    /// how a group's children join (a group's own; ignored on a part)
    #[serde(skip_serializing_if = "GroupMode::is_isolated")]
    pub mode: GroupMode,
    pub mods: Mods,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub material: Option<Material>,
    /// Sparse canonical overrides over the body's effective material. The legacy map above is the
    /// current renderer's projection; this typed patch is the lossless owner when it is present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material_v2: Option<crate::pbr::MaterialPatch>,
    pub cut_material: CutMaterial,
}

fn is_zero(v: &f64) -> bool { *v == 0.0 }

/// How a group's children join.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum GroupMode {
    /// joined to each other, then the group joined to what is below it as one (as groups always were)
    #[default]
    Isolated,
    /// joined straight into what is below the group, each by its own join, as if not grouped
    Passthrough,
    /// joined straight into what is below the group, each by the group's join (operation, blend, seam profile,
    /// band, cut material) instead of its own
    Overridden,
}
impl GroupMode {
    pub const ALL: [GroupMode; 3] = [GroupMode::Isolated, GroupMode::Passthrough, GroupMode::Overridden];
    pub fn is_isolated(&self) -> bool { *self == GroupMode::Isolated }
}

impl Node {
    pub fn can_ungroup(&self) -> bool {
        matches!(self.source, Source::Group(_)) && self.mods.list.is_empty() && self.material.is_none() && self.material_v2.is_none()
    }
    /// A group whose children join straight into what is below it: its mode says so, and nothing of its
    /// own (a live modifier, a material) needs its result as one shape.
    pub fn inlines(&self) -> bool {
        matches!(self.source, Source::Group(_)) && !self.mode.is_isolated() && !self.mods.any_live() && self.material.is_none()
    }
    /// Its join, given to another node (a group's override, a group's join handed to its first child).
    pub fn take_join(&mut self) -> Join {
        Join {
            op: std::mem::take(&mut self.op),
            blend: std::mem::take(&mut self.blend),
            blend_relative: std::mem::take(&mut self.blend_relative),
            profile: std::mem::take(&mut self.profile),
            band: std::mem::take(&mut self.band),
            depth: std::mem::take(&mut self.depth),
            width: std::mem::take(&mut self.width),
            cut_material: std::mem::take(&mut self.cut_material),
        }
    }
    pub fn join(&self) -> Join {
        Join {
            op: self.op, blend: self.blend, blend_relative: self.blend_relative, profile: self.profile,
            band: self.band, depth: self.depth, width: self.width, cut_material: self.cut_material,
        }
    }
    pub fn set_join(&mut self, j: Join) {
        (self.op, self.blend, self.blend_relative, self.profile) = (j.op, j.blend, j.blend_relative, j.profile);
        (self.band, self.depth, self.width, self.cut_material) = (j.band, j.depth, j.width, j.cut_material);
    }
}

/// How a node joins what is below it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Join {
    pub op: Op,
    pub blend: f64,
    pub blend_relative: bool,
    pub profile: crate::join::Profile,
    pub band: crate::join::Band,
    pub depth: f64,
    pub width: f64,
    pub cut_material: CutMaterial,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Construction {
    pub roots: Vec<Node>,
}

impl Construction {
    pub fn has_material(&self) -> bool {
        fn painted(n: &Node) -> bool {
            n.material.is_some() || n.material_v2.is_some() || matches!(&n.source,Source::Group(c) if c.iter().any(painted))
        }
        self.roots.iter().any(painted)
    }

    pub fn children(&self, path: &[usize]) -> Option<&[Node]> {
        let mut nodes = self.roots.as_slice();
        for &i in path {
            let Source::Group(children) = &nodes.get(i)?.source else {
                return None;
            };
            nodes = children;
        }
        Some(nodes)
    }

    pub fn children_mut(&mut self, path: &[usize]) -> Option<&mut Vec<Node>> {
        let mut nodes = &mut self.roots;
        for &i in path {
            let Source::Group(children) = &mut nodes.get_mut(i)?.source else {
                return None;
            };
            nodes = children;
        }
        Some(nodes)
    }

    pub fn node(&self, path: &[usize]) -> Option<&Node> {
        let (last, parent) = path.split_last()?;
        self.children(parent)?.get(*last)
    }

    pub fn node_mut(&mut self, path: &[usize]) -> Option<&mut Node> {
        let (last, parent) = path.split_last()?;
        self.children_mut(parent)?.get_mut(*last)
    }

    /// Preserve the selected children's internal order and put the group at the lowest selection.
    pub fn group(
        &mut self,
        parent: &[usize],
        indices: &[usize],
        name: impl Into<String>,
    ) -> Option<Vec<usize>> {
        let children = self.children_mut(parent)?;
        let mut selected = indices.to_vec();
        selected.sort_unstable();
        selected.dedup();
        let first = *selected.first()?;
        if *selected.last()? >= children.len() {
            return None;
        }
        let mut members: Vec<Node> = selected
            .into_iter()
            .rev()
            .map(|i| children.remove(i))
            .collect();
        members.reverse();
        let mut group = Node {
            name: name.into(),
            local_origin: Some([0.; 3]),
            ..Node::default()
        };
        // The lowest layer's parent operation now belongs to the group: the inside
        // starts from that layer's solid, including when the whole group cuts its parent.
        group.set_join(members[0].take_join());
        group.source = Source::Group(members);
        children.insert(first, group);
        let mut path = parent.to_vec();
        path.push(first);
        Some(path)
    }

    /// A modifier or material on a group requires the group boundary to stay.
    pub fn ungroup(&mut self, path: &[usize]) -> bool {
        let Some((&i, parent)) = path.split_last() else {
            return false;
        };
        let Some(nodes) = self.children_mut(parent) else {
            return false;
        };
        let Some(group) = nodes.get(i) else {
            return false;
        };
        if !group.can_ungroup() {
            return false;
        }
        let group = nodes.remove(i);
        let (join, mode) = (group.join(), group.mode);
        let Source::Group(mut contents) = group.source else {
            unreachable!()
        };
        // (a group that joins its children straight in already looks ungrouped: keep that look)
        match mode {
            GroupMode::Isolated => if let Some(first) = contents.first_mut() { first.set_join(join); },
            GroupMode::Passthrough => {}
            GroupMode::Overridden => for child in &mut contents { child.set_join(join); },
        }
        for child in &mut contents {
            child.place = crate::construction_pivot::combine(group.place, child.place);
        }
        nodes.splice(i..i, contents);
        true
    }

    fn from_parts(parts: &[Part], ids: &[u64]) -> Self {
        let mut roots = vec![Node {
            name: "Base".into(),
            ..Node::default()
        }];
        let mut path = Vec::new();
        for (part, &id) in parts.iter().zip(ids) {
            let next = usize::from(part.level.min(crate::csg::MAX_LEVEL)).min(path.len() + 1);
            path.truncate(next);
            let mut children = &mut roots;
            for &i in &path {
                let node: &mut Node = &mut children[i];
                let Source::Group(c) = &mut node.source else {
                    unreachable!()
                };
                children = c;
            }
            let mut node = Node {
                name: part.preset.clone(),
                source: Source::Part(id),
                op: part.op,
                blend: part.smooth,
                blend_relative: part.smooth_rel,
                ..Node::default()
            };
            if next > path.len() {
                let index = children.len();
                let group = Node {
                    name: "Group".into(),
                    op: std::mem::take(&mut node.op),
                    blend: std::mem::take(&mut node.blend),
                    blend_relative: std::mem::take(&mut node.blend_relative),
                    source: Source::Group(vec![node]),
                    ..Node::default()
                };
                children.push(group);
                path.push(index);
            } else {
                children.push(node);
            }
        }
        Self { roots }
    }
}

/// Preserve assigned identities and assign unique IDs to imported or newly appended primitives.
fn part_ids(parts: &[Part]) -> Vec<u64> {
    let mut used: HashSet<u64> = parts
        .iter()
        .map(|p| p.construction_id)
        .filter(|id| *id > 0)
        .collect();
    let mut seen = HashSet::new();
    let mut next = 1;
    parts
        .iter()
        .map(|p| {
            if p.construction_id > 0 && seen.insert(p.construction_id) {
                return p.construction_id;
            }
            while used.contains(&next) {
                next += 1;
            }
            used.insert(next);
            next += 1;
            next - 1
        })
        .collect()
}

impl Csg {
    /// Reading an old setup does not change it. Its legacy part nesting appears as editable groups.
    pub fn construction_tree(&self) -> Construction {
        self.construction
            .clone()
            .unwrap_or_else(|| Construction::from_parts(&self.parts, &part_ids(&self.parts)))
    }

    pub fn edit_construction(&mut self) -> &mut Construction {
        let ids = part_ids(&self.parts);
        for (p, &id) in self.parts.iter_mut().zip(&ids) {
            p.construction_id = id;
        }
        self.construction
            .get_or_insert_with(|| Construction::from_parts(&self.parts, &ids))
    }

    pub fn construction_part(&self, id: u64) -> Option<usize> {
        part_ids(&self.parts).iter().position(|&n| n == id)
    }

    /// The stable identity of the part at this place in the list: what a gesture or command holds on to
    /// instead of the place (a part not yet given one is named as `part_ids` would name it).
    pub fn part_id(&self, index: usize) -> Option<u64> {
        part_ids(&self.parts).get(index).copied()
    }

    /// Insert a new CSG primitive directly into the selected group.
    pub fn add_construction_part(
        &mut self,
        parent: &[usize],
        mut part: Part,
    ) -> Option<Vec<usize>> {
        if self.parts.len() + 1 >= crate::csg::MAX_PARTS
            || self.construction_tree().children(parent).is_none()
        {
            return None;
        }
        self.edit_construction();
        part.construction_id = 0;
        self.parts.push(part);
        let ids = part_ids(&self.parts);
        let id = *ids.last()?;
        let p = self.parts.last_mut()?;
        p.construction_id = id;
        let node = Node {
            name: p.preset.clone(),
            source: Source::Part(id),
            op: p.op,
            blend: p.smooth,
            blend_relative: p.smooth_rel,
            ..Node::default()
        };
        let children = self.construction.as_mut()?.children_mut(parent)?;
        let mut path = parent.to_vec();
        path.push(children.len());
        children.push(node);
        Some(path)
    }
}

impl Csg {
    /// Remove the editable node and the primitive definitions it owns.
    pub fn remove_construction_node(&mut self, path: &[usize]) -> bool {
        let Some((&index, parent)) = path.split_last() else {
            return false;
        };
        if self.construction_tree().node(path).is_none() {
            return false;
        }
        let tree = self.edit_construction();
        let Some(nodes) = tree.children_mut(parent) else {
            return false;
        };
        let mut removed = Vec::new();
        fn collect(node: Node, ids: &mut Vec<u64>) {
            match node.source {
                Source::Part(id) => ids.push(id),
                Source::Group(children) => {
                    for child in children {
                        collect(child, ids);
                    }
                }
                Source::Base => {}
            }
        }
        collect(nodes.remove(index), &mut removed);
        self.parts.retain(|p| !removed.contains(&p.construction_id));
        self.sel = None;
        true
    }
}
