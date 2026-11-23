//! Spatial references of modifier operations. The saved stack holds the artist's
//! reference and coordinates. A geometry build receives a separately resolved
//! stack, so evaluating a stage never rewrites its own reference.
use crate::layers::Placement;
use crate::mods::{Modifier, Mods};
use serde::{Deserialize, Serialize};

pub type Point = [f64; 3];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pivot {
    #[default]
    Local,
    CentreOfMass,
    PinnedMass,
    World,
}

impl Pivot {
    pub fn is_local(&self) -> bool { *self == Self::Local }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PivotFrame {
    /// Transform from the geometry's existing coordinates to true world space.
    pub placement: Placement,
    /// The editable fixed origin expressed in the geometry's coordinates.
    pub origin: Point,
}

impl PivotFrame {
    pub fn to_local(&self, world: Point) -> Point {
        let r = self.placement.inverse();
        std::array::from_fn(|i| {
            let row = &r[4 * i..4 * i + 4];
            row[0] * world[0] + row[1] * world[1] + row[2] * world[2] + row[3]
        })
    }
    pub fn to_world(&self, local: Point) -> Point {
        let p = &self.placement;
        let rot = crate::math::euler(p.rot[0], p.rot[1], p.rot[2]);
        let v = crate::math::apply(&rot, local);
        std::array::from_fn(|i| p.pos[i] + p.size * v[i])
    }
}

impl Modifier {
    /// Detach without moving the pivot. This coordinate travels with the shape.
    pub fn unlink_mass(&mut self, measured: Point) {
        self.pivot = Pivot::PinnedMass;
        self.centre = measured;
    }

    pub fn pivot_position(&self, frame: PivotFrame, incoming: Option<Point>) -> Point {
        match self.pivot {
            Pivot::Local => std::array::from_fn(|i| frame.origin[i] + self.centre[i]),
            Pivot::World => frame.to_local(self.centre),
            Pivot::CentreOfMass => incoming.unwrap_or(frame.origin),
            Pivot::PinnedMass => self.centre,
        }
    }
}

impl Mods {
    /// Keep existing operations in place when an artist moves an origin.
    /// Pinned points also need an offset if the coordinate frame itself moved.
    pub(crate) fn rebase_origin(&mut self, offset: Point, frame_moved: bool) {
        for stage in &mut self.list {
            if stage.pivot.is_local() || (frame_moved && stage.pivot == Pivot::PinnedMass) {
                stage.centre = std::array::from_fn(|i| stage.centre[i] + offset[i]);
            }
        }
    }

    /// Resolve the saved stack in evaluation order. A mass measurement sees only
    /// earlier live stages, with their pivots already resolved in geometry space.
    pub fn resolve_pivots<E>(&self, frame: PivotFrame, mut incoming: impl FnMut(usize, &Mods) -> Result<Option<Point>, E>) -> Result<Self, E> {
        let mut working = self.clone();
        let mut previous = Mods::default();
        for (i, stage) in working.list.iter_mut().enumerate() {
            if !stage.live() { continue; }
            stage.localize_cut(frame);
            let mass = if stage.pivot == Pivot::CentreOfMass { incoming(i, &previous)? } else { None };
            stage.centre = stage.pivot_position(frame, mass);
            stage.pivot = Pivot::Local;
            previous.list.push(stage.clone());
        }
        Ok(working)
    }
}

/// Integrate a sampled solid of uniform density. Fractional coverage near the
/// zero level reduces grid bias. The result is independent of material/lighting.
pub fn centre_of_mass(side: usize, lo: Point, hi: Point, distances: impl IntoIterator<Item = f64>) -> Option<Point> {
    mass_properties(side, lo, hi, distances).map(|v| v.1)
}

pub fn mass_properties(side: usize, lo: Point, hi: Point, distances: impl IntoIterator<Item = f64>) -> Option<(f64, Point)> {
    if side == 0 { return None; }
    let cells = side.checked_pow(3)?;
    let n = side as f64;
    let widths: Point = std::array::from_fn(|i| (hi[i] - lo[i]) / n);
    let edge = widths.into_iter().fold(0.0, f64::max);
    if edge <= 0.0 { return None; }
    let mut weight = 0.0;
    let mut moments = [0.0; 3];
    let mut count = 0;
    for (idx, d) in distances.into_iter().enumerate() {
        if idx >= cells || !d.is_finite() { return None; }
        count += 1;
        let w = (0.5 - d / edge).clamp(0.0, 1.0);
        let which = [idx % side, (idx / side) % side, idx / side / side];
        weight += w;
        for k in 0..3 { moments[k] += w * (which[k] as f64 + 0.5); }
    }
    if weight == 0.0 || count != cells { return None; }
    Some((weight * widths.iter().product::<f64>(), std::array::from_fn(|i| lo[i] + widths[i] * moments[i] / weight)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mods::Kind;
    #[test]
    fn signed_occupancy_weighs_the_solid() {
        // One fat cube and an arm extending its bounding box to the right.
        let side = 16;
        let d = (0..side * side * side).map(|i| {
            let (x, y, z) = (i % side, i / side % side, i / side / side);
            if x < 8 || (y < 2 && z < 2) { -2. } else { 2. }
        });
        let v = centre_of_mass(side, [0.; 3], [16.; 3], d).unwrap();
        assert!((v[0] - (2048. * 4. + 32. * 12.) / 2080.).abs() < 1e-12);
        assert!(v[0] < 5.0);
        assert_eq!(centre_of_mass(2, [0.; 3], [1.; 3], [1.; 8]), None);
    }
    #[test]
    fn each_mass_sees_earlier_geometry_before_its_own_stage() {
        let mut turn = Modifier::of(Kind::Twist);
        turn.pivot = Pivot::CentreOfMass;
        let mut off = turn.clone(); off.on = false;
        let source = Mods { list: vec![turn.clone(), off, turn] };
        let mut seen = vec![];
        let out = source.resolve_pivots(PivotFrame::default(), |i, prefix| {
            seen.push(i);
            Ok::<_, ()>(Some([prefix.list.len() as f64, 0., 0.]))
        }).unwrap();
        assert_eq!(seen, [0, 2]);
        assert_eq!(out.list[2].centre[0], 1.);
        assert_eq!(source.list[2].pivot, Pivot::CentreOfMass);
        let mut stage = source.list[2].clone(); stage.unlink_mass([1., 2., 3.]);
        assert_eq!(stage.pivot_position(PivotFrame::default(), Some([3.; 3])), [1., 2., 3.]);
    }
    #[test]
    fn world_points_use_the_actual_object_transform() {
        let f = PivotFrame { placement: Placement { pos: [2., -1., 0.5], rot: [10., 45., 70.], size: 1.7 }, origin: [0.1, 0., -0.2] };
        let mut m = Modifier::of(Kind::Twist);
        m.centre = [0.4, 0.3, -0.7]; let l = m.pivot_position(f, None);
        m.pivot = Pivot::World; m.centre = f.to_world(l);
        let p = m.pivot_position(f, None);
        for i in 0..3 { assert!((p[i] - l[i]).abs() < 1e-12); }
        let old: Modifier = serde_json::from_str(r#"{"kind":"twist","centre":[0.4,0.3,-0.7]}"#).unwrap();
        assert_eq!(old.pivot, Pivot::Local);
    }
}
impl crate::state::State {
    /// Fixed scene and construction transforms of a layer, in evaluation order.
    pub fn modifier_frame(&self, layer: usize) -> PivotFrame {
        let layer_state = self.stack.layers.get(layer).copied().unwrap_or_default();
        let scene = layer_state.place;
        let child = self.layer_csg(layer).place;
        let outer = PivotFrame { placement: scene, origin: layer_state.local_origin };
        if child.is_identity() { return outer; }
        let matrix = |r: Point| crate::math::euler(r[0], r[1], r[2]);
        let rot = crate::math::euler_of(&crate::math::mul(&matrix(scene.rot), &matrix(child.rot)));
        let origin = PivotFrame { placement: child, ..Default::default() }.to_local(outer.origin);
        PivotFrame { placement: Placement { pos: outer.to_world(child.pos), rot, size: scene.size * child.size }, origin }
    }
}

/// A viewport pivot keeps its owner even after the corresponding panel has closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PivotOwner {
    Object(usize),
    Construction(usize,u64),
    World(crate::groups::Item),
}

impl PivotOwner {
    pub fn frame(self, state: &crate::state::State) -> PivotFrame {
        match self {
            Self::Object(layer) => state.modifier_frame(layer),
            Self::World(item) => state.stack.world_frame(item),
            Self::Construction(layer,key) => state.node_with_pivot(layer,key).map(|n|state.construction_frame(layer,n)).unwrap_or_default(),
        }
    }

    pub fn measured(self, state: &crate::state::State, stage: usize) -> Option<Point> {
        match self {
            Self::Object(layer) => state.modifier_pivots.get(&(crate::groups::Item::Object(layer), stage)).copied(),
            Self::Construction(layer,key) => state.construction_pivots.get(&(layer,key,stage)).copied(),
            Self::World(item) => state.stack.world_pivots.get(&(item, stage)).map(|p| self.frame(state).to_local(*p)),
        }
    }

    /// The modifier stack this owner's pivots belong to.
    fn stack(self, state: &crate::state::State) -> Option<&crate::mods::Mods> {
        match self {
            Self::Object(layer) => state.layer_mods(layer),
            Self::World(item) => state.stack.world_modifiers(item),
            Self::Construction(layer,key) => Some(&state.node_with_pivot(layer,key)?.mods),
        }
    }

    /// How many modifiers the owner's stack holds (none: the owner is gone).
    pub fn stack_len(self, state: &crate::state::State) -> Option<usize> {
        self.stack(state).map(|m| m.list.len())
    }

    pub fn position(self, state: &crate::state::State, stage: usize) -> Option<Point> {
        let m = self.stack(state)?.list.get(stage)?;
        if !m.kind.framed() && !matches!(self, Self::World(_)) { return None; }
        let frame = self.frame(state);
        Some(frame.to_world(m.pivot_position(frame, self.measured(state, stage))))
    }

    pub fn set_position(self, state: &mut crate::state::State, stage: usize, world: Point) {
        let frame = self.frame(state);
        let local = frame.to_local(world);
        let mods = match self {
            Self::Object(layer) => state.layer_shape_mut(layer).map(|shape| &mut shape.mods),
            Self::World(item) => state.stack.world_modifiers_mut(item),
            Self::Construction(layer,key) => state.layer_csg_mut(layer).and_then(|c|c.construction.as_mut()).and_then(|tree| {
                let path=tree.pivot_path(key)?;
                tree.node_mut(&path).map(|n|&mut n.mods)
            }),
        };
        let Some(m) = mods.and_then(|stack| stack.list.get_mut(stage)) else { return };
        m.centre = match m.pivot {
            Pivot::World => world,
            Pivot::PinnedMass => local,
            Pivot::Local => std::array::from_fn(|k| local[k] - frame.origin[k]),
            Pivot::CentreOfMass => return,
        };
    }

    pub fn selected(self, state: &crate::state::State) -> bool {
        match self {
            Self::Object(layer) => layer == state.stack.sel && state.selected_group.is_none() && state.construction_pivot_owner().is_none(),
            Self::Construction(_,_) => state.construction_pivot_owner()==Some(self),
            Self::World(crate::groups::Item::Object(layer)) => state.edit_scope.object().is_none() && layer == state.stack.sel && state.selected_group.is_none(),
            Self::World(crate::groups::Item::Group(id)) => state.edit_scope.object().is_none() && state.selected_group == Some(id),
        }
    }
}

impl crate::state::State {
    pub fn modifier_world_pivot(&self, layer: usize, stage: usize) -> Option<Point> {
        PivotOwner::Object(layer).position(self, stage)
    }
    pub fn set_modifier_world_pivot(&mut self, layer: usize, stage: usize, world: Point) {
        PivotOwner::Object(layer).set_position(self, stage, world);
    }
}
