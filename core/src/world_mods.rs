//! A World modifier acts on occurrences of authored objects. Content is still
//! owned by the source layer. Moving or deleting an arrangement never rewrites
//! the source placement or creates extra rows in the hierarchy.
use crate::groups::{Item, Node};
use crate::layers::{Placement, Stack};
use crate::math::{apply, axis_angle, euler, euler_of, mul, transpose};
use crate::modifier_pivot::{PivotFrame, Point};
use crate::mods::{Kind, Modifier, Mods};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    #[default]
    World,
    SubObjects,
}
impl Scope {
    pub fn is_world(&self) -> bool {
        *self == Self::World
    }
}

/// A derived field belongs to its authored object; copies with the same
/// incoming history and mapped modifier share the same field.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldField {
    pub source: usize,
    pub mods: Mods,
}

#[derive(Clone, Default, Debug)]
pub struct WorldPlan {
    pub pivots: std::collections::HashMap<(Item, usize), Point>,
    pub placements: Vec<Vec<Placement>>,
    /// Incoming object-local deformations, kept per source independently.
    pub geometry: Vec<Mods>,
    /// Field IDs follow exactly the same source and occurrence order as placements.
    pub field_ids: Vec<Vec<usize>>,
    pub fields: Vec<WorldField>,
    /// How far each source is known to reach from its origin (`Stack::world_source_reach`).
    pub source_reach: Vec<Option<f64>>,
}

/// Every shape is fitted within this distance of its own origin (times its size): `layers::Stack::box_need`.
pub const FIT_REACH: f64 = 1.4;

/// How far from the origin a box's farthest corner lies.
pub fn corner_reach(b: &crate::mods::Box3) -> f64 {
    let m = |i: usize| b[i].abs().max(b[i + 3].abs());
    (m(0) * m(0) + m(1) * m(1) + m(2) * m(2)).sqrt()
}

impl WorldField {
    /// Source-local bounds after the last stage, and the scale of the cube needed
    /// to retain the solid between stages. An unfitted repeat can grow and a
    /// later modifier can bring its result back inside the original volume.
    pub fn domain(&self, source: crate::mods::Box3) -> (crate::mods::Box3, f64) {
        let (bounds, scale, _) = self.domain_in(source, corner_reach(&source));
        (bounds, scale)
    }

    /// The same, for a source known to lie within `reach` of its origin, and how far from the origin the
    /// result can reach. A box alone cannot hold a ball: a twist, swirl or bend turns points about their
    /// centre, keeping their distance to it, so their bound is a ball, and the box round that ball has
    /// corners √3 times as far out. Every stage's ball is carried alongside its box and the box is cut to
    /// it, so a twisted shape is not taken to reach three times as far as it does.
    pub fn domain_in(&self, source: crate::mods::Box3, reach: f64) -> (crate::mods::Box3, f64, f64) {
        let mut bounds = source;
        let mut reach = reach.min(corner_reach(&source));
        let mut extent = crate::HB;
        for stage in self.mods.live() {
            bounds = if stage.kind == Kind::Taper { stage.tapered_bounds(&bounds) } else { stage.bound(&bounds) };
            let c = stage.centre;
            let off = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
            reach = match stage.kind {
                // (distance to the centre kept: within reach + |c| of it, so within reach + 2|c| of the origin)
                Kind::Twist | Kind::Swirl | Kind::Bend => reach + 2.0 * off,
                Kind::Ripple => reach + 3f64.sqrt() * stage.a.get().iter().fold(0.0f64, |m, v| m.max(v.abs())) * stage.amount,
                Kind::Noise => reach + 3f64.sqrt() * 2.0 * stage.a.get().iter().fold(0.0f64, |m, v| m.max(v.abs())) * stage.amount,
                _ => corner_reach(&bounds),
            }.min(corner_reach(&bounds));
            for i in 0..3 {
                bounds[i] = bounds[i].max(-reach);
                bounds[i + 3] = bounds[i + 3].min(reach);
            }
            extent = bounds.iter().map(|v| v.abs()).fold(extent, f64::max);
        }
        (bounds, extent / crate::HB, reach)
    }
}

pub type FieldBounds = std::collections::HashMap<usize, (WorldField, crate::mods::Box3)>;

impl WorldPlan {
    /// The measured field in source coordinates, valid across a later rigid
    /// move while the mapped modifier stack stays identical.
    pub fn occurrence_bounds(
        &self,
        known: &FieldBounds,
        source: usize,
        ordinal: usize,
    ) -> Option<crate::mods::Box3> {
        let id = *self.field_ids.get(source)?.get(ordinal)?;
        let (field, bounds) = known.get(&id)?;
        (self.fields.get(id)? == field).then_some(*bounds)
    }

    pub fn projects_source(&self, source: usize, bounds: crate::mods::Box3) -> bool {
        self.has_distinct_fields(source) || self.field_ids.get(source)
            .and_then(|ids| ids.first())
            .is_some_and(|&id| self.field_domain(id, bounds).1 > 1.0 + 1e-8)
    }

    /// Field `id`'s domain (`WorldField::domain_in`) for its source with these bounds, and as far as the
    /// source is known to reach.
    pub fn field_domain(&self, id: usize, bounds: crate::mods::Box3) -> (crate::mods::Box3, f64, f64) {
        let field = &self.fields[id];
        field.domain_in(bounds, self.source_reach.get(field.source).copied().flatten().unwrap_or(f64::MAX))
    }

    pub fn has_distinct_fields(&self, source: usize) -> bool {
        self.field_ids.get(source).is_some_and(|ids| {
            ids.first()
                .is_some_and(|first| ids.iter().any(|id| id != first))
        })
    }
}

#[derive(Clone)]
struct Unit {
    origin: Point,
    members: Vec<(usize, Placement)>,
}

fn add(a: Point, b: Point) -> Point {
    std::array::from_fn(|i| a[i] + b[i])
}
fn sub(a: Point, b: Point) -> Point {
    std::array::from_fn(|i| a[i] - b[i])
}
fn scaled(p: Point, s: f64) -> Point {
    p.map(|v| v * s)
}
fn rotation(p: &Placement) -> crate::math::Mat3 {
    euler(p.rot[0], p.rot[1], p.rot[2])
}

impl Modifier {
    /// Express an inherited operation in the incoming geometry's coordinates.
    /// Angles stay angles while lengths and rates follow the relative scale.
    pub fn in_geometry_frame(
        &self,
        owner: Placement,
        geometry: Placement,
        world_pivot: Point,
    ) -> Self {
        let mut local = self.clone();
        let inner = PivotFrame {
            placement: geometry,
            origin: [0.; 3],
        };
        local.centre = inner.to_local(world_pivot);
        local.pivot = crate::modifier_pivot::Pivot::PinnedMass;
        let rotation = mul(&transpose(&rotation(&geometry)), &rotation(&owner));
        let sign = if owner.size * geometry.size < 0. {
            -1.
        } else {
            1.
        };
        local.basis = Some(
            self.frame()
                .map(|axis| scaled(apply(&rotation, axis), sign)),
        );
        let length = owner.size.abs().max(1e-9) / geometry.size.abs().max(1e-9);
        match self.kind {
            Kind::Twist | Kind::Taper | Kind::Bend => {
                local.a.v = local.a.v.map(|x| x / length);
                local.from *= length;
                local.to *= length;
            }
            Kind::Swirl => local.size *= length,
            Kind::Repeat => {
                local.b.v = local.b.v.map(|x| x * length);
                local.detail *= length;
            }
            _ => {}
        }
        local
    }
}

fn world_copies(stage: &Modifier) -> u64 {
    if stage.kind != Kind::Repeat || stage.scope != Scope::World {
        return 1;
    }
    let dimensions = stage.a.get().map(|v| v.round().clamp(1., 64.) as u64);
    if stage.mode == 1 {
        dimensions[0] * if stage.flip { 2 } else { 1 }
    } else {
        dimensions.iter().product()
    }
}

fn transforms(m: &Modifier, anchor: Point, owner: PivotFrame) -> Vec<Placement> {
    let frame = m
        .frame()
        .map(|axis| apply(&rotation(&owner.placement), axis));
    let length = owner.placement.size.abs().max(1e-9);
    let local = |v: Point| frame.map(|axis| axis.iter().zip(v).map(|(a, b)| a * b).sum::<f64>());
    let scene = |v: Point| std::array::from_fn(|i| (0..3).map(|k| v[k] * frame[k][i]).sum());
    let pivot = m.centre;
    let v = scaled(local(sub(anchor, pivot)), 1. / length);
    let a = m.a.get();
    let transform = |rot: crate::math::Mat3, offset: Point, scale: f64| Placement {
        pos: add(sub(pivot, scaled(apply(&rot, pivot), scale)), offset),
        rot: euler_of(&rot),
        size: scale,
    };
    let id = euler(0., 0., 0.);
    if m.kind == Kind::Repeat {
        let size = m.fit_scale();
        if m.mode == 1 {
            let count = a[0].round().clamp(1., 64.) as usize;
            let mut copies = Vec::with_capacity(world_copies(m) as usize);
            let axis = frame[1];
            let normal = frame[2];
            // A negative uniform scale followed by a half-turn about the
            // plane normal is precisely reflection in that plane.
            let reflection = axis_angle(normal[0], normal[1], normal[2], std::f64::consts::PI);
            for j in 0..count {
                let angle = std::f64::consts::TAU * j as f64 / count as f64;
                let around = axis_angle(axis[0], axis[1], axis[2], angle);
                let offset = scaled(apply(&around, frame[0]), m.detail * size * length);
                copies.push(transform(around, offset, size));
                if m.flip {
                    copies.push(transform(mul(&around, &reflection), offset, -size));
                }
            }
            return copies;
        }
        let counts = a.map(|x| x.round().clamp(1., 64.) as usize);
        let steps = m.b.get().map(|x| x.max(0.05));
        let span = (0..3)
            .map(|k| (counts[k] - 1) as f64 * steps[k])
            .fold(0., f64::max);
        let width = 2. * crate::mods::H;
        let size = if m.fit { width / (width + span) } else { 1. };
        let mut out = Vec::with_capacity(counts.iter().product());
        for z in 0..counts[2] {
            for y in 0..counts[1] {
                for x in 0..counts[0] {
                    let at = [x, y, z];
                    let mut offset = std::array::from_fn(|i| {
                        (at[i] as f64 - 0.5 * (counts[i] - 1) as f64) * steps[i] * size
                    });
                    if y % 2 == 1 {
                        offset[0] += m.stagger * steps[0] * size;
                    }
                    out.push(transform(id, scaled(scene(offset), length), size));
                }
            }
        }
        return out;
    }
    let mut r = id;
    let mut moved = v;
    for i in (0..3).rev() {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let amount = a[i] * m.amount;
        let angle = match m.kind {
            Kind::Twist => {
                -amount.to_radians() * moved[i].clamp(m.from.min(m.to), m.from.max(m.to))
                    / (2. * crate::mods::H)
            }
            Kind::Swirl => {
                -amount.to_radians()
                    * (-(moved[j] * moved[j] + moved[k] * moved[k]) / m.size.max(0.01).powi(2))
                        .exp()
            }
            Kind::Bend => {
                -amount.to_radians() * moved[if i == 1 { 0 } else { 1 }] / (2. * crate::mods::H)
            }
            Kind::Taper => {
                let s = (1. + amount * moved[i] / crate::mods::H).max(0.05);
                moved[j] *= s;
                moved[k] *= s;
                0.
            }
            Kind::Mirror => {
                if a[i] > 0.5 {
                    moved[i] = -moved[i];
                }
                0.
            }
            _ => 0.,
        };
        if angle != 0. {
            let axis_i = if m.kind == Kind::Bend {
                [2, 2, 0][i]
            } else {
                i
            };
            let angle = if m.kind == Kind::Bend && i != 0 {
                -angle
            } else {
                angle
            };
            let mut unit = [0.; 3];
            unit[axis_i] = 1.;
            moved = apply(&axis_angle(unit[0], unit[1], unit[2], angle), moved);
            let axis = frame[axis_i];
            r = mul(&axis_angle(axis[0], axis[1], axis[2], angle), &r);
        }
    }
    vec![Placement {
        pos: sub(add(pivot, scaled(scene(moved), length)), apply(&r, anchor)),
        rot: euler_of(&r),
        size: 1.,
    }]
}

fn composed(parent: &Placement, child: &Placement) -> Placement {
    let r = rotation(parent);
    Placement {
        pos: add(parent.pos, scaled(apply(&r, child.pos), parent.size)),
        rot: euler_of(&mul(&r, &rotation(child))),
        size: parent.size * child.size,
    }
}

#[derive(Clone,Copy,PartialEq,Eq,Hash,Debug)]
pub struct MassQuery(Item,usize,u64);
impl MassQuery {
    pub fn new(owner:Item,stage:usize,members:&[(usize,Placement)])->Self {
        use std::hash::{Hash,Hasher};
        let mut hash=std::collections::hash_map::DefaultHasher::new();
        for &(field,p) in members {
            field.hash(&mut hash);
            for x in p.pos.into_iter().chain(p.rot).chain([p.size]) {x.to_bits().hash(&mut hash);}
        }
        Self(owner,stage,hash.finish())
    }
}

pub type WorldMass<'a> = dyn FnMut(Item, usize, &[(usize, Placement)], &[WorldField]) -> Result<Option<Point>, String>
    + 'a;

impl Stack {
    pub fn world_modifiers(&self, item: Item) -> Option<&Mods> {
        match item {
            Item::Object(i) => self.world_mods.get(&i),
            Item::Group(id) => Some(&self.hierarchy.as_ref()?.group(id)?.world_mods),
        }
    }
    pub fn world_modifiers_mut(&mut self, item: Item) -> Option<&mut Mods> {
        match item {
            Item::Object(i) if i < self.layers.len() => Some(self.world_mods.entry(i).or_default()),
            Item::Group(id) => Some(&mut self.hierarchy.as_mut()?.group_mut(id)?.world_mods),
            _ => None,
        }
    }
    pub fn has_world_modifiers(&self) -> bool {
        self.world_mods.values().any(Mods::any_live)
            || self.hierarchy.as_ref().is_some_and(|h| {
                h.rows()
                    .iter()
                    .any(|(_, n)| matches!(n,Node::Group(g) if g.world_mods.any_live()))
            })
    }
    /// Count generated leaves before expanding any arrays. GPU jump indices are
    /// exact f32 integers and the binding must fit this device's storage limit.
    pub fn check_world_table(&self, byte_limit: u64) -> Result<(), String> {
        fn count(stack: &Stack, node: &Node) -> u64 {
            let mut leaves = match node {
                Node::Object(_) => 1,
                Node::Group(g) => g
                    .children
                    .iter()
                    .fold(0u64, |sum, n| sum.saturating_add(count(stack, n))),
            };
            if let Some(mods) = stack.world_modifiers(node.item()) {
                for m in mods.live() {
                    leaves = leaves.saturating_mul(world_copies(m));
                }
            }
            leaves
        }
        let roots = self
            .hierarchy
            .clone()
            .unwrap_or_else(|| crate::groups::Hierarchy::flat(self.layers.len()));
        let leaves = roots
            .0
            .iter()
            .fold(0u64, |sum, n| sum.saturating_add(count(self, n)));
        let nodes = leaves
            .saturating_mul(2)
            .saturating_sub(self.layers.len() as u64);
        let slots = nodes.saturating_mul(5).saturating_add(self.layers.len().max(4) as u64);
        if slots > 1 << 24 || slots.saturating_mul(16) > byte_limit {
            return Err(format!(
                "World array of {leaves} occurrences exceeds the instance table capacity"
            ));
        }
        Ok(())
    }
    pub fn world_plan(&self) -> WorldPlan {
        // The renderer surfaces the error at build time. View bounds and input
        // still work on the original definitions while an oversized edit is open.
        self.resolve_world(&mut |item, i, members, _| {
            Ok(self.world_mass_points.get(&MassQuery::new(item,i,members)).or_else(||self.world_pivots.get(&(item,i))).copied())
        })
            .unwrap_or_else(|_| WorldPlan {
                placements: self.layers.iter().map(|l| vec![l.place]).collect(),
                geometry: vec![Mods::default(); self.layers.len()],
                ..Default::default()
            })
    }
    pub fn world_frame(&self, item: Item) -> PivotFrame {
        match item {
            Item::Object(i) => self
                .layers
                .get(i)
                .map(|p| PivotFrame {
                    placement: p.place,
                    origin: p.local_origin,
                })
                .unwrap_or_default(),
            Item::Group(id) => PivotFrame {
                placement: self
                    .hierarchy
                    .as_ref()
                    .and_then(|h| h.group(id))
                    .map(|g| g.frame)
                    .unwrap_or_default(),
                origin: [0.; 3],
            },
        }
    }
    pub fn resolve_world(&self, measure: &mut WorldMass<'_>) -> Result<WorldPlan, String> {
        self.check_world_table(u64::MAX)?;
        let mut result = WorldPlan {
            source_reach: self.world_source_reach[..self.layers.len().min(crate::layers::MAX_LAYERS)].to_vec(),
            pivots: Default::default(),
            placements: vec![Vec::new(); self.layers.len()],
            geometry: vec![Mods::default(); self.layers.len()],
            field_ids: vec![Vec::new(); self.layers.len()],
            fields: (0..self.layers.len())
                .map(|source| WorldField {
                    source,
                    mods: Mods::default(),
                })
                .collect(),
        };
        let roots = self
            .hierarchy
            .clone()
            .unwrap_or_else(|| crate::groups::Hierarchy::flat(self.layers.len()));
        for root in &roots.0 {
            let group = self.evaluate_world(root, &mut result, measure)?;
            for (field, place) in group.members {
                let source = result.fields[field].source;
                result.placements[source].push(place);
                result.field_ids[source].push(field);
            }
        }
        for source in 0..self.layers.len() {
            if !result.has_distinct_fields(source) {
                if let Some(&field) = result.field_ids[source].first() {
                    result.geometry[source] = result.fields[field].mods.clone();
                }
            }
        }
        Ok(result)
    }
    fn evaluate_world(
        &self,
        node: &Node,
        result: &mut WorldPlan,
        measure: &mut WorldMass<'_>,
    ) -> Result<Unit, String> {
        let owner = node.item();
        let frame = self.world_frame(owner);
        let anchor = match node {
            Node::Object(i) => self.layers.get(*i).and_then(|l| l.arrangement_anchor),
            Node::Group(g) => g.arrangement_anchor,
        };
        // A parent evaluates bends and similar arrangements at this member
        // point. Dragging the gizmo alone must leave that sample in place.
        let arrangement_origin = frame.to_world(anchor.unwrap_or(frame.origin));
        let mut units = match node {
            Node::Object(i) => self
                .layers
                .get(*i)
                .map(|l| {
                    vec![Unit {
                        origin: arrangement_origin,
                        members: vec![(*i, l.place)],
                    }]
                })
                .unwrap_or_default(),
            Node::Group(g) => g
                .children
                .iter()
                .map(|n| self.evaluate_world(n, result, measure))
                .collect::<Result<_, _>>()?,
        };
        if let Some(mods) = self.world_modifiers(owner) {
            for (stage_index, stage) in mods.list.iter().enumerate().filter(|(_, s)| s.live()) {
                let mut operation = stage.clone();
                operation.centre = match stage.pivot {
                    crate::modifier_pivot::Pivot::CentreOfMass => {
                        let members: Vec<_> = units
                            .iter()
                            .flat_map(|u| u.members.iter().copied())
                            .collect();
                        measure(owner, stage_index, &members, &result.fields)?.unwrap_or(arrangement_origin)
                    }
                    crate::modifier_pivot::Pivot::World => stage.centre,
                    _ => frame.to_world(stage.pivot_position(frame, None)),
                };
                result.pivots.insert((owner, stage_index), operation.centre);
                if operation.scope == Scope::SubObjects {
                    // Resolve every reference on the incoming fields before
                    // publishing any output from this stage.
                    let mut mapped = Vec::new();
                    for unit in &units {
                        let group_mass = if !stage.cascade
                            && stage.pivot == crate::modifier_pivot::Pivot::CentreOfMass
                        {
                            measure(owner, stage_index, &unit.members, &result.fields)?
                        } else {
                            None
                        };
                        for &(field, placement) in &unit.members {
                            let source = result.fields[field].source;
                            let member_frame = PivotFrame {
                                placement,
                                origin: [0.; 3],
                            };
                            let anchor = if stage.cascade {
                                let layer = &self.layers[source];
                                member_frame.to_world(layer.arrangement_anchor.unwrap_or(layer.local_origin))
                            } else {
                                unit.origin
                            };
                            let centre = match stage.pivot {
                                crate::modifier_pivot::Pivot::Local => {
                                    add(anchor, sub(operation.centre, arrangement_origin))
                                }
                                crate::modifier_pivot::Pivot::CentreOfMass if stage.cascade => {
                                    measure(
                                        owner,
                                        stage_index,
                                        &[(field, placement)],
                                        &result.fields,
                                    )?
                                    .unwrap_or(anchor)
                                }
                                crate::modifier_pivot::Pivot::CentreOfMass => {
                                    group_mass.unwrap_or(anchor)
                                }
                                _ => operation.centre,
                            };
                            mapped.push((
                                field,
                                stage.in_geometry_frame(frame.placement, placement, centre),
                            ));
                        }
                    }
                    let mut shared = std::collections::HashMap::new();
                    for (member, (incoming, stage)) in units
                        .iter_mut()
                        .flat_map(|u| u.members.iter_mut())
                        .zip(mapped)
                    {
                        let key = (
                            incoming,
                            serde_json::to_string(&stage).map_err(|e| e.to_string())?,
                        );
                        let field = *shared.entry(key).or_insert_with(|| {
                            let mut derived = result.fields[incoming].clone();
                            derived.mods.list.push(stage);
                            let field = result.fields.len();
                            result.fields.push(derived);
                            field
                        });
                        member.0 = field;
                    }
                    continue;
                }
                let mut next = Vec::new();
                for unit in &units {
                    if operation.cascade && unit.members.len() > 1 {
                        // Slot k from each leaf remains part of the same generated
                        // subgroup. A later non-cascading stage can still act on it as one.
                        let mut copies: Vec<Unit> = Vec::new();
                        for (source, p) in &unit.members {
                            for (k, map) in
                                transforms(&operation, p.pos, frame).into_iter().enumerate()
                            {
                                if copies.len() <= k {
                                    copies.push(Unit {
                                        origin: unit.origin,
                                        members: Vec::new(),
                                    });
                                }
                                copies[k].members.push((*source, composed(&map, p)));
                            }
                        }
                        next.extend(copies);
                        continue;
                    }
                    for map in transforms(&operation, unit.origin, frame) {
                        let members = unit
                            .members
                            .iter()
                            .map(|(i, p)| (*i, composed(&map, p)))
                            .collect();
                        next.push(Unit {
                            origin: add(
                                map.pos,
                                scaled(apply(&rotation(&map), unit.origin), map.size),
                            ),
                            members,
                        });
                    }
                }
                units = next;
            }
        }
        Ok(Unit {
            origin: arrangement_origin,
            members: units.into_iter().flat_map(|u| u.members).collect(),
        })
    }
}
/// A stackless BVH per source. Entries contain instance transforms and box bounds
/// only. All leaf queries read the same source texture.
pub fn occurrence_table(layers: &[Vec<Placement>]) -> Vec<[f32; 4]> {
    fn extent(p: &Placement) -> Point {
        let r = rotation(p);
        std::array::from_fn(|i| {
            crate::HB * p.size.abs() * r[i * 3..i * 3 + 3].iter().map(|x| x.abs()).sum::<f64>()
        })
    }
    fn append(out: &mut Vec<[f32; 4]>, list: &mut [Placement]) {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in list.iter() {
            let e = extent(p);
            for k in 0..3 {
                lo[k] = lo[k].min(p.pos[k] - e[k]);
                hi[k] = hi[k].max(p.pos[k] + e[k]);
            }
        }
        let at = out.len();
        out.extend([[0.; 4]; 5]);
        out[at] = [lo[0] as f32, lo[1] as f32, lo[2] as f32, 0.];
        out[at + 1] = [hi[0] as f32, hi[1] as f32, hi[2] as f32, 0.];
        if list.len() == 1 {
            let p = list[0];
            let inv = p.inverse();
            out[at + 1][3] = p.size.abs() as f32;
            for k in 0..3 {
                out[at + 2 + k] = std::array::from_fn(|j| inv[k * 4 + j] as f32);
            }
        } else {
            let axis = (0..3)
                .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
                .unwrap();
            let mid = list.len() / 2;
            list.select_nth_unstable_by(mid, |a, b| a.pos[axis].total_cmp(&b.pos[axis]));
            let (left, right) = list.split_at_mut(mid);
            append(out, left);
            append(out, right);
        }
        out[at][3] = out.len() as f32;
    }
    // (one header entry per scene layer, at least the four a single pass reads; the shader's node offsets are absolute)
    let mut out = vec![[0.; 4]; layers.len().max(4)];
    for (i, layer) in layers.iter().enumerate() {
        if layer.is_empty() {
            continue;
        }
        let start = out.len();
        append(&mut out, &mut layer.clone());
        out[i] = [start as f32, out.len() as f32, 0., 0.];
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{groups::Hierarchy, mods::Axes};
    #[test]
    fn occurrence_table_has_a_header_entry_for_every_layer() {
        let p = Placement::default();
        let layers: Vec<Vec<Placement>> = (0..8).map(|_| vec![p, p]).collect();
        let t = occurrence_table(&layers);
        for (g, h) in t[..8].iter().enumerate() {
            assert!(h[0] >= 8.0 && h[1] > h[0], "layer {g}: {h:?}");
        }
        let small = occurrence_table(&layers[..2]);
        assert_eq!(small.len() >= 4, true);
        assert!(small[0][0] >= 4.0 && small[2] == [0.; 4]);
    }
    #[test]
    fn five_hundred_instances_survive_save_without_copying_definitions() {
        let mut stack = Stack::new();
        stack.layers[0].place.pos = [0.2, 0.1, 0.];
        let mut array = Modifier::of(Kind::Repeat);
        array.a = Axes::new([5., 10., 10.], false);
        stack.world_modifiers_mut(Item::Object(0)).unwrap().list = vec![array];
        assert_eq!(stack.world_plan().placements[0].len(), 500);
        assert_eq!(stack.layers.len(), 1);
        let json = serde_json::to_string(&stack.to_setup()).unwrap();
        let loaded = Stack::from_setup(&serde_json::from_str(&json).unwrap());
        assert_eq!(
            loaded.world_plan().placements,
            stack.world_plan().placements
        );
        stack.world_mods.clear();
        assert_eq!(
            stack.world_plan().placements[0],
            vec![stack.layers[0].place]
        );
        assert_eq!(stack.layers[0].place.pos, [0.2, 0.1, 0.]);
    }
    #[test]
    fn a_rotated_owner_carries_the_array_and_oversized_nested_arrays_report_an_error() {
        let mut stack = Stack::new();
        stack.layers[0].place.rot = [0., 0., 90.];
        stack.layers[0].place.size = 0.5;
        let mut array = Modifier::of(Kind::Repeat);
        array.fit = false;
        array.a = Axes::new([3., 1., 1.], false);
        array.b = Axes::new([1.; 3], false);
        let source = Item::Object(0);
        stack
            .world_modifiers_mut(source)
            .unwrap()
            .list
            .push(array.clone());
        let p = stack.world_plan().placements[0][2].pos;
        assert!(
            p[0].abs() < 1e-8 && (p[1].abs() - 0.5).abs() < 1e-8,
            "{p:?}"
        );
        array.a = Axes::new([64.; 3], false);
        stack.world_modifiers_mut(source).unwrap().list = vec![array.clone(), array];
        assert!(stack.check_world_table(1 << 30).is_err());
        assert_eq!(
            stack.world_plan().placements[0],
            vec![stack.layers[0].place]
        );
    }

    #[test]
    fn world_mirrors_create_rigid_occurrences_while_sub_object_repeats_share_geometry() {
        let mut stack = Stack::new();
        let source = Item::Object(0);
        let mut m = Modifier::of(Kind::Repeat);
        m.fit = false;
        m.mode = 1;
        m.flip = true;
        m.a = Axes::new([4., 1., 1.], false);
        stack.world_modifiers_mut(source).unwrap().list = vec![m];
        let plan = stack.world_plan();
        assert_eq!(plan.placements[0].len(), 8);
        for pair in plan.placements[0].chunks_exact(2) {
            assert!(pair[0].size > 0. && pair[1].size < 0.);
            assert_eq!(pair[0].pos, pair[1].pos);
        }
        let bytes = occurrence_table(&plan.placements).len() as u64 * 16;
        assert!(stack.check_world_table(bytes).is_ok());
        assert!(stack.check_world_table(bytes - 1).is_err());
        let m = &mut stack.world_modifiers_mut(source).unwrap().list[0];
        m.scope = Scope::SubObjects;
        let plan = stack.world_plan();
        assert_eq!(plan.placements[0], vec![stack.layers[0].place]);
        assert!(plan.geometry[0].list[0].flip);
        assert!(stack.check_world_table(144).is_ok());
        let m = &mut stack.world_modifiers_mut(source).unwrap().list[0];
        m.scope = Scope::World;
        m.mode = 0;
        m.a = Axes::new([1., 2., 1.], false);
        m.b = Axes::new([0.6, 0.4, 1.], false);
        m.stagger = 0.5;
        let plan = stack.world_plan();
        assert!((plan.placements[0][1].pos[0] - plan.placements[0][0].pos[0] - 0.3).abs() < 1e-9);
    }

    #[test]
    fn scope_and_cascade_are_independent_for_nested_scene_objects() {
        let mut s = Stack::new();
        s.add();
        s.layers[1].place.pos = [0., 0.6, 0.4];
        let inner = crate::groups::Group {
            id: 2,
            arrangement_anchor: None,
            name: "inner".into(),
            world: true,
            frame: Placement::default(),
            kind: None,
            operation: Default::default(),
            mods: Default::default(),
            world_mods: Default::default(),
            children: vec![Node::Object(0), Node::Object(1)],
        };
        let mut bend = Modifier::of(Kind::Bend);
        bend.a = Axes::new([120., 0., 0.], false);
        let outer = crate::groups::Group {
            id: 1,
            arrangement_anchor: None,
            name: "outer".into(),
            world: true,
            frame: Placement::default(),
            kind: None,
            operation: Default::default(),
            mods: Default::default(),
            world_mods: Mods { list: vec![bend] },
            children: vec![Node::Group(inner)],
        };
        s.hierarchy = Some(Hierarchy(vec![Node::Group(outer)]));
        let p = s.world_plan();
        assert_eq!(p.placements[1][0].pos, s.layers[1].place.pos);
        assert!(p.geometry[1].is_empty());
        s.world_modifiers_mut(Item::Group(1)).unwrap().list[0].cascade = true;
        let p = s.world_plan();
        assert_ne!(p.placements[1][0].pos, s.layers[1].place.pos);
        s.world_modifiers_mut(Item::Group(1)).unwrap().list[0].scope = Scope::SubObjects;
        let p = s.world_plan();
        assert_eq!(p.placements[1][0].pos, s.layers[1].place.pos);
        assert_eq!(p.geometry[1].list.len(), 1);
        assert_eq!(p.geometry[0].list.len(), 1);
        let leaf_pivot = p.geometry[1].list[0].centre;
        assert!(leaf_pivot.iter().all(|v| v.abs() < 1e-10));
        s.world_modifiers_mut(Item::Group(1)).unwrap().list[0].cascade = false;
        let group_pivot = s.world_plan().geometry[1].list[0].centre;
        assert_eq!(group_pivot, [0., -0.6, -0.4]);
    }
}
