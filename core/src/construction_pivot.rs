//! Construction modifier frames. Primitive axes travel with their placement,
//! and a group's axes are the object's local axes.
use crate::construction::{Construction, Node, Source};
use crate::csg::Csg;
use crate::layers::Placement;
use crate::modifier_pivot::{PivotFrame, PivotOwner};
use crate::state::State;
use std::hash::{DefaultHasher, Hash, Hasher};

impl Node {
    /// Identifies a handle's owner across moves within the tree. An edit to a
    /// group's membership invalidates its old handle.
    pub fn pivot_key(&self) -> u64 {
        let mut h = DefaultHasher::new();
        std::mem::discriminant(&self.source).hash(&mut h);
        match &self.source {
            Source::Base => {}
            Source::Part(id) => id.hash(&mut h),
            Source::Group(children) => {
                let mut keys = children.iter().map(Node::pivot_key).collect::<Vec<_>>();
                keys.sort_unstable();
                keys.hash(&mut h);
            }
        }
        h.finish()
    }

    pub fn placement(&self, csg: &Csg) -> Placement {
        match self.source {
            Source::Part(id) => csg
                .construction_part(id)
                .and_then(|i| csg.parts.get(i))
                .map(|p| combine(self.place, p.place))
                .unwrap_or(self.place),
            _ => self.place,
        }
    }
}

impl Construction {
    /// The ordered transforms down to this layer, excluding the primitive's own parameters.
    pub fn placement_at(&self, path: &[usize]) -> Option<Placement> {
        let mut place = Placement::default();
        let mut nodes = self.roots.as_slice();
        for (depth, &i) in path.iter().enumerate() {
            let node = nodes.get(i)?;
            place = combine(place, node.place);
            if depth + 1 < path.len() {
                let Source::Group(children) = &node.source else { return None };
                nodes = children;
            }
        }
        Some(place)
    }

    pub fn place_parts(&self, csg: &Csg, parts: &mut [crate::csg::Resolved]) {
        use crate::csg::GridFrom;
        fn visit(nodes: &[Node], parent: Placement, csg: &Csg, parts: &mut [crate::csg::Resolved]) {
            for node in nodes {
                let frame = combine(parent, node.place);
                let source = match &node.source {
                    Source::Group(children) => { visit(children, frame, csg, parts); continue; }
                    _ if frame.is_identity() => continue,
                    Source::Base => Some(GridFrom::Up),
                    Source::Part(id) => csg.construction_part(*id).map(GridFrom::Part),
                };
                if let Some(p) = parts.iter_mut().find(|p| source == Some(p.source)) {
                    p.place = combine(frame, p.place);
                }
            }
        }
        visit(&self.roots, Placement::default(), csg, parts);
    }

    pub fn find_path(&self, matches: impl Fn(&Node) -> bool) -> Option<Vec<usize>> {
        fn visit(nodes: &[Node], matches: &impl Fn(&Node) -> bool, path: &mut Vec<usize>) -> bool {
            for (i, node) in nodes.iter().enumerate() {
                path.push(i);
                if matches(node) {
                    return true;
                }
                if let Source::Group(children) = &node.source {
                    if visit(children, matches, path) {
                        return true;
                    }
                }
                path.pop();
            }
            false
        }
        let mut path = Vec::new();
        visit(&self.roots, &matches, &mut path).then_some(path)
    }

    pub fn pivot_path(&self, key: u64) -> Option<Vec<usize>> {
        self.find_path(|node| node.pivot_key() == key)
    }
}

pub fn combine(outer: Placement, child: Placement) -> Placement {
    if child.is_identity() {
        return outer;
    }
    if outer.is_identity() {
        return child;
    }
    let turn = |r: [f64; 3]| crate::math::euler(r[0], r[1], r[2]);
    let rot = crate::math::euler_of(&crate::math::mul(&turn(outer.rot), &turn(child.rot)));
    Placement {
        pos: PivotFrame {
            placement: outer,
            ..Default::default()
        }
        .to_world(child.pos),
        rot,
        size: outer.size * child.size,
    }
}

impl State {
    pub fn construction_frame(&self, layer: usize, node: &Node) -> PivotFrame {
        let parent = self.modifier_frame(layer);
        let csg = self.layer_csg(layer);
        let ancestors = csg.construction.as_ref().and_then(|tree| {
            let path = tree.pivot_path(node.pivot_key())?;
            tree.placement_at(path.split_last()?.1)
        }).unwrap_or_default();
        let child = combine(ancestors, node.placement(csg));
        PivotFrame {
            placement: combine(parent.placement, child),
            origin: node.local_origin.unwrap_or(if matches!(node.source, Source::Part(_)) {
                [0.; 3]
            } else {
                parent.origin
            }),
        }
    }

    /// Move the selected origin in world coordinates while keeping the incoming
    /// geometry and the positions of its locally referenced operations fixed.
    pub fn set_construction_origin(&mut self, world: [f64; 3]) -> bool {
        if !world.into_iter().all(f64::is_finite) { return false; }
        let Some(layer) = self.edit_scope.object() else { return false };
        let path = self.edit_scope.construction().to_vec();
        let tree = self.csg.construction_tree();
        let Some(node) = tree.node(&path) else { return false };
        let frame = self.construction_frame(layer, node);
        let local = frame.to_local(world);
        let node = self.csg.edit_construction().node_mut(&path).expect("selected construction layer");
        node.mods.rebase_origin(std::array::from_fn(|k| frame.origin[k] - local[k]), false);
        node.local_origin = Some(local);
        self.refresh_scope();
        true
    }

    pub fn construction_pivot_owner(&self) -> Option<PivotOwner> {
        let layer = self.edit_scope.object()?;
        let node = self
            .layer_csg(layer)
            .construction
            .as_ref()?
            .node(&self.edit_scope.construction())?;
        Some(PivotOwner::Construction(layer, node.pivot_key()))
    }

    pub fn node_with_pivot(&self, layer: usize, key: u64) -> Option<&Node> {
        let tree = self.layer_csg(layer).construction.as_ref()?;
        tree.node(&tree.pivot_path(key)?)
    }
}
