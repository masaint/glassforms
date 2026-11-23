//! Resolve geometry-following pivots from the solid immediately before each
//! modifier. Measurement runs in an isolated geometry builder, kept between
//! edits. Its input excludes the stage being measured and every later stage.
use crate::{engine::Engine, gpu::R};
use gf_core::groups::Item;
use gf_core::layers::{LayerShape, Placement};
use gf_core::modifier_pivot::{mass_properties, PivotFrame, Point};
use gf_core::mods::Mods;
use gf_core::state::{Detail, State};
use std::collections::HashMap;

#[derive(Default)]
pub(crate) struct PivotWork {
    builder: Option<Box<Engine>>,
    bounds: HashMap<usize, (LayerShape, u64)>,
    /// Only stages that currently follow the mass or were just measured.
    cached: HashMap<MassKey, (LayerShape, Mods, u64, Option<(f64, Point)>)>,
    used: std::collections::HashSet<MassKey>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MassKey {
    Construction(usize,u64,usize),
    Object(usize, usize),
    World(Item, usize, usize),
}

impl Engine {
    pub(crate) fn start_pivot_measurements(&mut self) { self.pivot_work.used.clear(); }
    pub(crate) fn install_modifier_pivots(&mut self) -> R<Vec<(usize, Mods)>> {

        let isolated = self.state.edit_scope.object().is_some();
        let inherited = if !isolated && self.state.stack.has_world_modifiers() {
            let limit = self.gpu.device.limits().max_storage_buffer_binding_size;
            self.state.stack.check_world_table(u64::from(limit))?;
            let stack = self.state.stack.clone();
            let mut points=HashMap::new();
            let plan = stack.resolve_world(&mut |owner, i, members, geometry| {
                let mass=self.world_mass(owner,i,members,geometry)?;
                if let Some(p)=mass {points.insert(gf_core::world_mods::MassQuery::new(owner,i,members),p);}
                Ok(mass)
            })?;
            self.state.stack.world_mass_points=points;
            self.state.stack.world_pivots = plan.pivots.clone();
            let mut geometry=plan.geometry.clone();
            // Determine the field domain from the authored source before installing
            // any inherited modifiers. Reuse intrinsic measurements across rigid edits.
            self.state.stack.world_source_bounds = Default::default();
            if plan.fields.iter().any(|f| f.mods.any_live()) {
                self.measure_world_bounds()?;
                for (source, shared) in geometry.iter_mut().enumerate() {
                    let key = self.source_bounds_key(source);
                    let bounds = self.pivot_work.bounds.get(&source)
                        .filter(|kept| **kept == key)
                        .and_then(|_| self.state.layer_shape(source)?.meta.bounds)
                        .map(|b| padded_bounds(b,128)).unwrap_or_else(gf_core::mods::whole_box);
                    self.state.stack.world_source_bounds[source] = Some(bounds);
                    if plan.projects_source(source, bounds) { *shared = Mods::default(); }
                }
            }
            self.layers.world_fields.plan=Some(plan);
            Some(geometry)
        } else {
            self.layers.world_fields.plan=None;
            if !isolated {
                self.state.stack.world_pivots.clear();
                self.state.stack.world_mass_points.clear();
                self.state.stack.world_field_bounds.clear();
                self.state.stack.world_source_bounds = Default::default();
            }
            None
        };
        let mut plans = Vec::new();
        self.state.modifier_pivots.clear();
        for layer in 0..self.state.layer_count() {
            let Some(stack) = self.state.layer_mods(layer) else { continue };
            let geometry = inherited.as_ref().and_then(|all| all.get(layer)).filter(|s| s.any_live());
            let frame = self.state.modifier_frame(layer);
            if geometry.is_none() && !stack.list.iter().any(|m| m.live() && (!m.pivot.is_local() || m.has_world_cut() || frame.origin != [0.; 3])) { continue; }
            let mut authored = stack.clone();
            if let Some(geometry) = geometry {
                let inner=self.state.layer_csg(layer).place;
                authored.list.extend(geometry.live().map(|m|m.in_geometry_frame(Placement::default(),inner,m.centre)));
            }
            let base = self.state.layer_shape_setup(layer);
            let clock = self.state.clock;
            let working = authored.resolve_pivots(frame, |i, prefix| {
                let mut shape = base.clone();
                shape.shape.mods = Some(prefix.clone());
                self.measure_incoming(MassKey::Object(layer, i), shape, clock)
                    .map(|mass| mass.map(|(_, c)| c))
            })?;
            for (i, stage) in working.list.iter().enumerate() {
                if stage.live() {
                    self.state
                        .modifier_pivots
                        .insert((Item::Object(layer), i), stage.centre);
                }
            }
            plans.push((layer, working));
        }
        self.pivot_work
            .cached
            .retain(|k, _| self.pivot_work.used.contains(k));
        for (layer, stack) in &mut plans {
            let shape = self
                .state
                .layer_shape_mut(*layer)
                .expect("layer selected above");
            std::mem::swap(stack, &mut shape.mods);
        }
        Ok(plans)
    }

    fn world_mass(
        &mut self,
        owner: Item,
        stage: usize,
        members: &[(usize, Placement)],
        geometry: &[gf_core::world_mods::WorldField],
    ) -> R<Option<Point>> {
        let clock = self.state.clock;
        let mut measured = HashMap::new();
        let mut total = 0.;
        let mut moments = [0.; 3];
        for &(field, placed) in members {
            let source=geometry[field].source;
            if !measured.contains_key(&field) {
                let mut shape = self.state.layer_shape_setup(source);
                let all = shape.shape.mods.clone().unwrap_or_default();
                let resolved =
                    all.resolve_pivots(self.state.modifier_frame(source), |i, prefix| {
                        let mut incoming = shape.clone();
                        incoming.shape.mods = Some(prefix.clone());
                        self.measure_incoming(MassKey::Object(source, i), incoming, clock)
                            .map(|v| v.map(|(_, c)| c))
                    })?;
                shape.shape.mods = Some(resolved);
                let key = MassKey::World(owner, stage, field);
                let inherited = &geometry[field].mods;
                let raw = if inherited.any_live() {
                    self.measure_mass(key, shape, clock, inherited)?
                } else {
                    // An undeformed definition can be weighed before placement,
                    // even if its transform carries it outside the sampling cube.
                    let frame = PivotFrame { placement: shape.csg_place, origin: [0.; 3] };
                    self.measure_incoming(key, shape, clock)?.map(|(mass, centre)| (
                        mass * frame.placement.size.abs().powi(3), frame.to_world(centre)
                    ))
                };
                measured.insert(field, raw);
            }
            if let Some((w, c)) = measured[&field] {
                let world = PivotFrame {
                    placement: placed,
                    origin: [0.; 3],
                }
                .to_world(c);
                let weight = w * placed.size.abs().powi(3);
                total += weight;
                for k in 0..3 {
                    moments[k] += weight * world[k];
                }
            }
        }
        Ok((total > 0.).then(|| moments.map(|v| v / total)))
    }

    pub(crate) fn measure_incoming(
        &mut self,
        key: MassKey,
        mut shape: LayerShape,
        clock: f64,
    ) -> R<Option<(f64, Point)>> {
        shape.csg_place = Placement::default();
        self.measure_mass(key, shape, clock, &Mods::default())
    }

    fn measure_mass(&mut self, key: MassKey, mut shape: LayerShape, clock: f64, inherited: &Mods) -> R<Option<(f64, Point)>> {
        self.pivot_work.used.insert(key);
        let time_key = shape_time(&shape, clock);
        shape.shape.res = 128;
        shape.shape.anim_res = 128;
        if let Some((kept, prior, time, centre)) = self.pivot_work.cached.get(&key) {
            if kept == &shape && prior == inherited && *time == time_key {
                return Ok(*centre);
            }
        }
        let base = self.build_sample_shape(&shape, clock)?;
        let mut scale = 1.0;
        let volume = if inherited.any_live() {
            let (side, planes) = self.read_sample_shape(&base)?;
            let bounds = padded_bounds(sample_bounds(side, &planes), side);
            let field = gf_core::world_mods::WorldField { source: 0, mods: inherited.clone() };
            scale = field.domain(bounds).1;
            let bake = inherited.bake(&bounds).expect("live World deformation");
            let res = gf_core::state::valid_res(f64::from(base.res) * scale).min(256);
            let out = crate::bake::Volume::new(&self.gpu, res, self.gpu.storage16);
            let sculpt = &mut self.pivot_work.builder.as_mut().unwrap().sculptor;
            let slot = crate::sculpt::PIVOT_SLOT;
            sculpt.keep(slot, &base, "incoming World mass")?;
            sculpt.run_world_field(slot, &out, "incoming World mass", bake, false, scale, base.res, None)?;
            out
        } else { base };
        let (res, samples) = self.read_sample_shape(&volume)?;
        let field = samples.iter().flat_map(|plane| plane.chunks_exact(2))
            .map(|p| half::f16::from_le_bytes([p[0], p[1]]).to_f64() * scale);
        let half_width = gf_core::HB * scale;
        let centre = mass_properties(res, [-half_width; 3], [half_width; 3], field);
        self.pivot_work.cached.insert(key, (shape, inherited.clone(), time_key, centre));
        Ok(centre)
    }

    /// Keep an intrinsic box with each editable shape. Translate, rotate and reparent
    /// use this cached box and never trigger a second geometry readback.
    pub(crate) fn measure_world_bounds(&mut self) -> R<()> {
        if self.state.playing { return Ok(()); }
        for layer in 0..self.state.object_count() {
            let key = self.source_bounds_key(layer);
            if self.pivot_work.bounds.get(&layer).is_some_and(|known| known == &key)
                && self.state.layer_shape(layer).is_some_and(|s| s.meta.bounds.is_some()) { continue; }
            let volume = self.build_sample_shape(&key.0, self.state.clock)?;
            let (side, samples) = self.read_sample_shape(&volume)?;
            let bounds = sample_bounds(side, &samples);
            if let Some(st) = self.state.layer_shape_mut(layer) { st.meta.bounds = Some(bounds); }
            self.pivot_work.bounds.insert(layer, key);
        }
        Ok(())
    }

    fn source_bounds_key(&self, layer: usize) -> (LayerShape, u64) {
        let mut shape = self.state.layer_shape_setup(layer);
        shape.shape.res = 128;
        shape.shape.anim_res = 128;
        let time = shape_time(&shape, self.state.clock);
        (shape, time)
    }

    fn build_sample_shape(&mut self, shape: &LayerShape, clock: f64) -> R<crate::bake::Volume> {
        let builder = self.pivot_work.builder.get_or_insert_with(|| Box::new(Engine::new(self.gpu.clone(), State::new(), true)));
        let mut setup = builder.state.setup();
        setup.family = shape.family.clone();
        setup.preset = shape.preset.clone();
        setup.values = shape.values.clone();
        setup.shape = shape.shape.clone();
        setup.csg = shape.csg.clone();
        setup.construction = shape.construction.clone();
        setup.csg_place = shape.csg_place;
        setup.csg_coarse = true;
        setup.csg_baked = shape.csg_baked;
        setup.clock = clock;
        builder.state.apply_setup(&setup)?;
        builder.build_geometry(Detail::Full)?;
        builder.scene_volume().cloned().ok_or_else(|| "no incoming modifier geometry".into())
    }

    fn read_sample_shape(&self, volume: &crate::bake::Volume) -> R<(usize, Vec<Vec<u8>>)> {
        let res = volume.res;
        let planes: Vec<_> = (0..res).collect();
        let read = self.gpu.read_later(&volume.tex, &planes, res, res, 2);
        self.gpu.wait();
        let samples = read.take(&self.gpu).ok_or("could not read the incoming modifier geometry")?;
        Ok((res as usize, samples))
    }
}

fn sample_bounds(side: usize, samples: &[Vec<u8>]) -> gf_core::mods::Box3 {
    let mut bounds = [f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    let cell = 2.0 * gf_core::HB / side as f64;
    for (i, pair) in samples.iter().flat_map(|plane| plane.chunks_exact(2)).enumerate() {
        let d = half::f16::from_le_bytes([pair[0], pair[1]]).to_f64();
        if d > cell * 0.5 { continue; }
        let index = [i % side, (i / side) % side, i / (side * side)];
        for k in 0..3 {
            let p = -gf_core::HB + index[k] as f64 * cell;
            bounds[k] = bounds[k].min(p);
            bounds[k+3] = bounds[k+3].max(p + cell);
        }
    }
    bounds
}

fn padded_bounds(bounds: gf_core::mods::Box3, side: usize) -> gf_core::mods::Box3 {
    if !bounds.iter().all(|v| v.is_finite()) || bounds[0] > bounds[3] {
        return gf_core::mods::whole_box();
    }
    let cell = 2.0 * gf_core::HB / side as f64;
    std::array::from_fn(|k| (bounds[k] + if k < 3 { -cell } else { cell }).clamp(-gf_core::HB, gf_core::HB))
}

fn shape_time(shape: &LayerShape, clock: f64) -> u64 {
    let evolving = shape.shape.beads > 0.0
        || shape.shape.mods.as_ref().and_then(Mods::lab_live)
            .is_some_and(|s| s.layers.iter().any(|l| l.period().is_some()))
        || shape.construction.as_ref().is_some_and(|t| construction_evolves(&t.roots));
    if evolving { clock.to_bits() } else { 0 }
}

fn construction_evolves(nodes: &[gf_core::construction::Node]) -> bool {
    nodes.iter().any(|node| node.mods.lab_live().is_some_and(|s|s.layers.iter().any(|l|l.period().is_some()))
        || matches!(&node.source,gf_core::construction::Source::Group(children) if construction_evolves(children)))
}
