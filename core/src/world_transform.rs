//! Scene elements store world-space transforms. A reparent is an outline edit, while a
//! transform applies the same similarity transform to every selected subtree exactly once.
use crate::{
    groups::Item,
    layers::Placement,
    math::{apply, euler, euler_of, mul, transpose},
    modifier_pivot::PivotFrame,
    state::State,
};

fn map_placement(from: Placement, to: Placement, mut value: Placement) -> Placement {
    if from.rot == to.rot && from.size == to.size {
        value.pos = std::array::from_fn(|i| value.pos[i] + (to.pos[i] - from.pos[i]));
        return value;
    }
    let rotation = |p: Placement| euler(p.rot[0], p.rot[1], p.rot[2]);
    let delta = mul(&rotation(to), &transpose(&rotation(from)));
    let offset = apply(&delta, std::array::from_fn(|i| value.pos[i] - from.pos[i]));
    let ratio = to.size / from.size;
    value.pos = std::array::from_fn(|i| to.pos[i] + ratio * offset[i]);
    value.rot = euler_of(&mul(&delta, &rotation(value)));
    value.size *= ratio;
    value
}

impl State {
    /// The frame used by the gizmo includes the artist's fixed origin.
    pub fn world_placement(&self, item: Item) -> Option<Placement> {
        match item {
            Item::Group(id) => Some(self.stack.hierarchy.as_ref()?.group(id)?.frame),
            Item::Object(i) if i < self.object_count() => {
                let layer = &self.stack.layers[i];
                let pos = PivotFrame { placement: layer.place, origin: layer.local_origin }.to_world(layer.local_origin);
                Some(Placement { pos, ..layer.place })
            }
            _ => None,
        }
    }

    pub fn set_world_origin(&mut self, item: Item, world: [f64; 3]) -> bool {
        if world.iter().any(|v| !v.is_finite()) { return false; }
        let Some(old) = self.world_placement(item) else { return false };
        if old.pos == world { return true; }
        match item {
            Item::Group(id) => {
                let Some(g) = self.stack.hierarchy.as_mut().and_then(|h| h.group_mut(id)) else { return false };
                let frame = PivotFrame { placement: g.frame, ..Default::default() };
                let offset = frame.to_local(world).map(|v| -v);
                let anchor = g.arrangement_anchor.unwrap_or_default();
                g.arrangement_anchor = Some(std::array::from_fn(|i| anchor[i] + offset[i]));
                g.world_mods.rebase_origin(offset, true);
                g.frame.pos = world;
            }
            Item::Object(i) if i < self.object_count() => {
                let shape_frame = self.modifier_frame(i);
                let layer = &mut self.stack.layers[i];
                let frame = PivotFrame { placement: layer.place, origin: layer.local_origin };
                let anchor = layer.arrangement_anchor.unwrap_or(layer.local_origin);
                layer.local_origin = frame.to_local(world);
                layer.arrangement_anchor = (anchor != layer.local_origin).then_some(anchor);
                if let Some(mods) = self.stack.world_mods.get_mut(&i) {
                    mods.rebase_origin(std::array::from_fn(|k| frame.origin[k] - layer.local_origin[k]), false);
                }
                let new_origin = self.modifier_frame(i).origin;
                let offset = std::array::from_fn(|k| shape_frame.origin[k] - new_origin[k]);
                if let Some(shape) = self.layer_shape_mut(i) {
                    shape.mods.rebase_origin(offset, false);
                }
                if let Some(tree) = self.layer_csg_mut(i).and_then(|c| c.construction.as_mut()) {
                    rebase_inherited_origins(&mut tree.roots, offset);
                }
            }
            _ => return false,
        }
        self.refresh_scope();
        true
    }

    pub fn set_world_placement(&mut self, item: Item, frame: Placement) -> bool {
        self.transform_world_items(&[item], item, frame)
    }

    pub fn transform_world_items(&mut self, items: &[Item], anchor: Item, frame: Placement) -> bool {
        let Some(from) = self.world_placement(anchor) else { return false };
        if !items.contains(&anchor) || from.size.abs() < 1e-12 || !frame.size.is_finite() || frame.size <= 0.0
            || !frame.pos.into_iter().chain(frame.rot).all(f64::is_finite) { return false; }
        if frame == from { return true; }
        let h = self.outline();
        for (_, node) in h.rows() {
            let key = node.item();
            let affected = items.contains(&key) || h.ancestors(key).is_some_and(|ancestors| {
                ancestors.into_iter().any(|id| items.contains(&Item::Group(id)))
            });
            if !affected { continue; }
            match key {
                Item::Group(id) => {
                    let g = self.stack.hierarchy.as_mut().unwrap().group_mut(id).unwrap();
                    g.frame = if anchor == key { frame } else { map_placement(from, frame, g.frame) };
                }
                Item::Object(i) => {
                    let old = self.stack.layers[i].place;
                    let mut next = map_placement(from, frame, old);
                    if !self.tracks.is_empty() {
                        for (c, slider) in crate::layers::PLACE.iter().enumerate() {
                            if self.tracks.contains_key(&format!("L{}.{}", i + 1, slider.id)) {
                                match c {
                                    0..=2 => next.pos[c] = old.pos[c],
                                    3..=5 => next.rot[c - 3] = old.rot[c - 3],
                                    _ => next.size = old.size,
                                }
                            }
                        }
                    }
                    self.stack.layers[i].place = next;
                }
            }
        }
        self.refresh_scope();
        true
    }
}

fn rebase_inherited_origins(nodes: &mut [crate::construction::Node], offset: [f64; 3]) {
    use crate::construction::Source;
    for node in nodes {
        if node.local_origin.is_none() && !matches!(node.source, Source::Part(_)) {
            node.mods.rebase_origin(offset, false);
        }
        if let Source::Group(children) = &mut node.source {
            rebase_inherited_origins(children, offset);
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Reference {
    Min,
    #[default]
    Centre,
    Max,
    Origin,
}

/// Movement allowed by the active translation handle, expressed in World coordinates.
#[derive(Clone, Copy)]
pub enum DragConstraint {
    Axis([f64; 3]),
    Plane { normal: [f64; 3] },
}

impl DragConstraint {
    fn projection(self) -> [[f64; 3]; 3] {
        let direction = match self { Self::Axis(d) | Self::Plane { normal: d } => d };
        let length_sq = direction.iter().map(|x| x * x).sum::<f64>();
        if length_sq < 1e-20 { return [[0.; 3]; 3]; }
        std::array::from_fn(|i| std::array::from_fn(|j| {
            let along = direction[i] * direction[j] / length_sq;
            match self {
                Self::Axis(_) => along,
                Self::Plane { .. } => f64::from(i == j) - along,
            }
        }))
    }
}

impl State {
    pub fn world_name(&self, item: Item) -> String {
        match item {
            Item::Object(i) => self.layer_title(i),
            Item::Group(id) => self.stack.hierarchy.as_ref().and_then(|h| h.group(id))
                .map(|g| g.name.clone()).unwrap_or_default(),
        }
    }

    pub fn world_bounds(&self, item: Item) -> Option<crate::mods::Box3> {
        let tree = self.outline();
        self.stack.check_world_table(u64::MAX).ok()?;
        let plan = self.stack.has_world_modifiers().then(|| self.stack.world_plan());
        self.placed_bounds(item, &tree, plan.as_ref())
    }

    pub(crate) fn placed_bounds(&self, item: Item, tree: &crate::groups::Hierarchy, plan: Option<&crate::world_mods::WorldPlan>) -> Option<crate::mods::Box3> {
        let mut out = [f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for i in 0..self.object_count() {
            let layer = &self.stack.layers[i];
            let leaf = Item::Object(i);
            if leaf != item {
                let Item::Group(id) = item else { continue };
                if !tree.ancestors(leaf)?.contains(&id) { continue; }
            }
            let local = self.layer_shape(i)?.meta.bounds.unwrap_or([-crate::FIT, -crate::FIT, -crate::FIT, crate::FIT, crate::FIT, crate::FIT]);
            let placements = plan.map_or(std::slice::from_ref(&layer.place), |p| p.placements[i].as_slice());
            for (ordinal,&placement) in placements.iter().enumerate() {
                let local=plan.and_then(|p|p.occurrence_bounds(&self.stack.world_field_bounds,i,ordinal)).unwrap_or(local);
                if local[0]>local[3] {continue;}
                let transform = PivotFrame { placement, ..Default::default() };
                for mask in 0..8 {
                    let point = transform.to_world(std::array::from_fn(|k| local[if mask & (1 << k) == 0 { k } else { k+3 }]));
                    for k in 0..3 {
                        out[k] = out[k].min(point[k]);
                        out[k+3] = out[k+3].max(point[k]);
                    }
                }
            }
        }
        (out[0] <= out[3]).then_some(out)
    }

    pub fn world_framing(&self, item: Item) -> Option<([f64; 3], f64)> {
        let Some(b) = self.world_bounds(item) else { return self.world_placement(item).map(|p| (p.pos, 0.1)); };
        let centre = std::array::from_fn(|i| (b[i] + b[i+3]) * 0.5);
        let radius = (0..3).map(|i| (b[i] - centre[i]).powi(2)).sum::<f64>().sqrt();
        Some((centre, radius.max(0.01)))
    }

    fn world_reference(&self, item: Item, axis: usize, reference: Reference) -> Option<f64> {
        if axis >= 3 { return None; }
        if reference == Reference::Origin { return Some(self.world_placement(item)?.pos[axis]); }
        let b = self.world_bounds(item)?;
        Some(match reference {
            Reference::Min => b[axis],
            Reference::Max => b[axis+3],
            _ => (b[axis] + b[axis+3]) * 0.5,
        })
    }

    /// Selected descendants already move with their selected ancestor.
    fn independent_items(&self, items: &[Item]) -> Vec<Item> {
        let h = self.outline();
        items.iter().copied().filter(|item| h.ancestors(*item).is_some_and(|a| {
            !a.iter().any(|&id| items.contains(&Item::Group(id)))
        })).collect()
    }

    /// Align to the last independently selected element.
    pub fn world_align(&mut self, items: &[Item], axis: usize, reference: Reference) {
        let roots = self.independent_items(items);
        let Some(at) = roots.last().and_then(|v| self.world_reference(*v, axis, reference)) else { return; };
        for item in roots {
            if let (Some(here), Some(mut p)) = (self.world_reference(item, axis, reference), self.world_placement(item)) {
                p.pos[axis] += at - here;
                self.set_world_placement(item, p);
            }
        }
    }

    pub fn world_distribute(&mut self, items: &[Item], axis: usize, reference: Reference) {
        let mut rows: Vec<_> = self.independent_items(items).into_iter().filter_map(|item| Some((item, self.world_reference(item, axis, reference)?))).collect();
        rows.sort_by(|a, b| a.1.total_cmp(&b.1));
        let n = rows.len();
        if n < 3 { return; }
        let first = rows[0].1;
        let gap = (rows[n-1].1 - first) / (n-1) as f64;
        for (i, (item, at)) in rows.iter().enumerate().take(n-1).skip(1) {
            if let Some(mut p) = self.world_placement(*item) {
                p.pos[axis] += first + gap * i as f64 - at;
                self.set_world_placement(*item, p);
            }
        }
    }

    pub fn world_snap(&self, item: Item, proposed: [f64;3], constraint: DragConstraint, reference: Option<Reference>, tolerance: f64) -> [f64;3] {
        let Some(origin) = self.world_placement(item) else { return proposed; };
        let h = self.outline();
        let selected = self.world_selection();
        let peers = match self.edit_scope.world { Some(id) => h.group(id).map(|g| g.children.as_slice()).unwrap_or(&[]), None => &h.0 };
        if self.stack.check_world_table(u64::MAX).is_err() { return proposed; }
        let plan = self.stack.has_world_modifiers().then(|| self.stack.world_plan());
        let references = |owner| {
            let placement = self.world_placement(owner)?;
            let bounds = self.placed_bounds(owner, &h, plan.as_ref()).unwrap_or_else(|| std::array::from_fn(|k| placement.pos[k % 3]));
            Some([[bounds[0], bounds[1], bounds[2]],
                  std::array::from_fn(|k| (bounds[k] + bounds[k+3]) * 0.5),
                  [bounds[3], bounds[4], bounds[5]], placement.pos])
        };
        let all = [Reference::Min, Reference::Centre, Reference::Max, Reference::Origin];
        let Some(sources) = references(item) else { return proposed };
        let targets: Vec<_> = peers.iter().map(|n| n.item()).filter(|p| *p != item && !selected.contains(p)).filter_map(references).collect();
        let mut snapped = proposed;
        let mut projection = constraint.projection();
        let mut remaining_sq = tolerance.max(0.0).powi(2);
        // A line can meet one guide, and a plane can meet two. Rank candidates by the
        // distance the handle must travel along its constraint to reach the guide.
        for _ in 0..2 {
            let mut winner = None;
            let mut best_sq = remaining_sq;
            for axis in 0..3 {
                let weight = projection[axis][axis];
                if weight < 1e-12 { continue; }
                for target in targets.iter().flatten() {
                    for (kind, source) in all.iter().zip(&sources) {
                        if reference.is_some_and(|r| r != *kind) { continue; }
                        let offset = target[axis] - (source[axis] + snapped[axis] - origin.pos[axis]);
                        let distance_sq = offset * offset / weight;
                        if distance_sq < best_sq {
                            best_sq = distance_sq;
                            winner = Some((axis, offset));
                        }
                    }
                }
            }
            let Some((axis, offset)) = winner else { break };
            let guide = projection[axis];
            let weight = guide[axis];
            for i in 0..3 {
                snapped[i] += guide[i] * offset / weight;
                // Any second correction must preserve the guide already reached.
                for j in 0..3 { projection[i][j] -= guide[i] * guide[j] / weight; }
            }
            remaining_sq -= best_sq;
        }
        snapped
    }
}
