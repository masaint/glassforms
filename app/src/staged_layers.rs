//! Settle a layered scene over several frames without publishing a partly written field: after a handle
//! is let go, after a pause, the full build goes in pieces behind the coarser picture already shown.
//! Each frame does one piece: a layer's shape (a formula field in bands of slices, any other shape
//! whole), an object's tree measured into its volume, or a band of the combined volume. The renderer
//! keeps its previous scene and bodies' volumes until every slice is ready; then the ordinary full build
//! runs, finds everything made, and points the tracer at it. Any other build asked for meanwhile
//! cancels the pieces (the layers already finished are kept).

use super::{exact_to, Built, Combine, Detail, Engine, ShapeJob, Volume, W};
use crate::gpu::R;
use gf_core::layers::{Pass, WideUse, MAX_LAYERS};
use gf_core::presets::Kind;
use gf_core::state::Job;

/// How a layer is built again: a formula field slice band by slice band, or any other shape whole.
enum Shape {
    Field { family: usize, preset: usize, uniforms: [f32; 12], table: Vec<f32> },
    Whole,
}

struct Field {
    key: String,
    resolution: u32,
    shape: Shape,
    rebuild: bool,
    /// the object has a tree: measured into its own volume once its shape is built (or at once, when
    /// only the tree's volume is out of date)
    tree: bool,
}

/// The combining step as the full build will make it: one pass of up to COMBINE_WIDTH layers, or passes.
enum Combining {
    One(Combine),
    Passes(Vec<Pass>),
}

pub(super) struct FullRefinement {
    pub(super) visible_scene: Volume,
    pub(super) visible_body: Option<Volume>,
    pub(super) visible_channels: u32,
    fields: Vec<Field>,
    combination: Combining,
    resolution: u32,
    pub(super) layer: usize,
    /// the tree of layer `layer` is measured next (its shape is done)
    tree_next: bool,
    /// the job of layer `layer`, worked out and waiting to be built
    job: Option<Job>,
    pub(super) next_slice: u32,
}

impl FullRefinement {
    /// The combined volume is being written (every layer is done).
    pub(super) fn combining(&self) -> bool {
        self.layer == self.fields.len()
    }
}

/// Slices of a formula field built in one frame, and of the combined volume.
const FIELD_SLICES: u32 = 384;
/// (about 8 ms of the 8-wide step at 1024 a side, 4 to 6 ms of one pass)
const COMBINE_SLICES: u32 = 192;

impl Engine {
    fn can_refine_fields(&self) -> bool {
        let count = self.state.layer_count();
        self.gpu.storage16
            && !self.blocking()
            && !self.state.rendering
            && !self.state.playing
            && self.layers.combined
            && !self.layers.force
            && !self.layers.in_place
            && count > 1
            && count <= MAX_LAYERS
            && self.state.edit_scope.object().is_none()
            && self.state.edit_scope.isolated().is_none()
            && !self.state.stack.has_world_modifiers()
            && !self
                .state
                .stack
                .hierarchy
                .as_ref()
                .is_some_and(|h| h.changes_geometry())
            && self.bubbles_wanted().is_none()
            && self.body_bubbles_wanted().is_none()
            && (0..count).all(|i| self.state.layer_bake(i).is_none())
    }

    fn start_layer_refinement(&mut self) -> Option<FullRefinement> {
        // (GF_SETTLE_WHOLE=1: the settled build made at once, to compare the two)
        if self.state.wanted() != Some(Detail::Full) || !self.can_refine_fields() || std::env::var_os("GF_SETTLE_WHOLE").is_some() {
            return None;
        }
        let resolution = self.state.scene_res(Detail::Full);
        let visible_scene = self.scene_volume()?.clone();
        let count = self.state.layer_count();
        let mut fields = Vec::with_capacity(count);
        for i in 0..count {
            let key = self.state.layer_key(i);
            // (as the ordinary build decides it: `bake_layers`; a tube's job is not made here, it is
            // the costly part of building one)
            let resolution = Detail::Full.res(self.state.layer_shape(i)?, self.state.anim_cap);
            let rebuild = !self.layers.built[i]
                .as_ref()
                .is_some_and(|old| old.exact() && old.res == resolution && old.key == key);
            let shape = match self.state.layer_family(i).kind {
                Kind::Field if rebuild && !self.own_volume_unread(i) => {
                    let mut job = self.state.layer_job(i, Detail::Full, None).ok()?;
                    job.prepare_formula(&self.state.families).ok()?;
                    let ShapeJob::Field { uniforms, table } = job.shape else { return None };
                    Shape::Field { family: job.family, preset: job.preset, uniforms, table }
                }
                Kind::Field | Kind::Curve => Shape::Whole,
                // (a model is read and built on its own schedule)
                Kind::Mesh => return None,
            };
            fields.push(Field { key, resolution, shape, rebuild, tree: self.state.has_tree(i) });
        }
        // Only where a coarser picture stands in: layers to build again, or a combined volume coarser
        // than the full one (an edit made at full detail goes straight through the ordinary build).
        if resolution <= 384 || (visible_scene.res >= resolution && !fields.iter().any(|f| f.rebuild)) {
            return None;
        }
        let combination = if count > W {
            Combining::Passes(self.state.stack.pass_plan(&self.state.stack))
        } else {
            Combining::One(self.state.stack.combine())
        };
        // A tube whose seam reaches past its own volume's exact distance needs its wide volume, which the
        // ordinary build makes: such a scene is not staged (a tube built again is checked once built).
        for (i, field) in fields.iter().enumerate() {
            let Some((_, to)) = self.layers.built[i].as_ref().filter(|_| !field.rebuild).and_then(|b| b.exact_to) else { continue };
            if self.needs_wide(&combination, i, field, to) { return None; }
        }
        let (visible_body, visible_channels) = match self.bodies() {
            Some(b) => (Some(b.volume.clone()), b.channels),
            None => (None, 0),
        };
        Some(FullRefinement {
            visible_scene,
            visible_body,
            visible_channels,
            fields,
            combination,
            resolution,
            layer: 0,
            tree_next: false,
            job: None,
            next_slice: 0,
        })
    }

    pub(crate) fn has_layer_refinement(&self) -> bool {
        self.layers.pending.is_some()
    }

    /// A normal rebuild or an export supersedes outstanding slices. Restore the last complete pair
    /// before handing off, including when the new request subsequently fails.
    pub(crate) fn cancel_layer_refinement(&mut self) -> bool {
        let Some(stage) = self.layers.pending.take() else {
            return false;
        };
        if stage.combining() && stage.next_slice > 0 {
            let combiner = &mut self.layers.combiner;
            match combiner.vols.iter().position(|v| v.res == stage.visible_scene.res) {
                Some(i) => combiner.cur = i,
                None => {
                    combiner.vols.push(stage.visible_scene);
                    combiner.cur = combiner.vols.len() - 1;
                }
            }
            combiner.bodies_spare = combiner.bodies.take().map(|v| (v, combiner.bodies_channels));
            combiner.bodies = stage.visible_body;
            combiner.bodies_channels = stage.visible_channels;
        }
        self.layers.last = None;
        self.layers.last_multi = None;
        // (a field cut short holds slices of two shapes; a layer finished whole is good as it is)
        if let Some(field) = stage.fields.get(stage.layer) {
            if field.rebuild && matches!(field.shape, Shape::Field { .. }) && stage.next_slice > 0 {
                self.layers.built[stage.layer] = None;
            }
        }
        true
    }

    /// Return true while the live frame can keep the last coherent pose on screen.
    pub(crate) fn refine_layers_live(&mut self) -> bool {
        if self.has_layer_refinement() && (self.state.build_wanted() || !self.can_refine_fields()) {
            self.cancel_layer_refinement();
            self.state.request_if_idle(gf_core::build::BuildRequest::refinement());
            return false;
        }
        let Some(mut stage) = self
            .layers
            .pending
            .take()
            .or_else(|| self.start_layer_refinement())
        else {
            return false;
        };
        self.state.drop_request();
        match self.next_layer_slices(&mut stage) {
            Ok(false) => {
                self.layers.pending = Some(stage);
                self.trim_spares();
                true
            }
            Ok(true) => {
                self.state.request(gf_core::build::BuildRequest::refinement());
                false
            }
            Err(_) => {
                self.layers.pending = Some(stage);
                self.cancel_layer_refinement();
                self.state.request(gf_core::build::BuildRequest::refinement());
                false
            }
        }
    }

    /// Does layer i (a tube exact out to `to`) need its wide volume in this combination?
    fn needs_wide(&self, combination: &Combining, i: usize, field: &Field, to: f64) -> bool {
        if field.tree || self.layers.no_wide { return false; }
        let reach = match combination {
            Combining::One(c) => c.reach(i),
            Combining::Passes(plan) => {
                let Some((p, k)) = plan.iter().enumerate().find_map(|(p, pass)| pass.layers.iter().position(|&g| g == i).map(|k| (p, k))) else { return true };
                plan[p].combine.reach(k)
            }
        };
        reach > to
    }

    /// Mark layer i as built at full detail (its volume is complete).
    fn layer_done(&mut self, i: usize, field: &Field, melt: Option<f64>) {
        self.layers.built[i] = Some(Built {
            key: field.key.clone(),
            res: field.resolution,
            mode: Detail::Full,
            exact_to: melt.map(|m| exact_to(field.resolution, m)),
            wide: None,
        });
        self.layers.builds[i] += 1;
        if let Some(shape) = self.state.layer_shape_mut(i) {
            shape.meta.res = field.resolution;
        }
    }

    /// One piece of the settled build. True when every piece is done.
    fn next_layer_slices(&mut self, stage: &mut FullRefinement) -> R<bool> {
        while stage.layer < stage.fields.len() && !stage.tree_next && !stage.fields[stage.layer].rebuild {
            // (a layer whose shape stands may still have a tree measured at another size)
            let field = &stage.fields[stage.layer];
            if field.tree && self.csg.layer_vols.get(stage.layer).and_then(|v| v.as_ref()).is_none_or(|v| v.1.res != field.resolution) {
                stage.tree_next = true;
                break;
            }
            stage.layer += 1;
        }
        if stage.tree_next {
            // the object's tree measured into its own volume (`csg_layer_volume` keeps it for the full build)
            let i = stage.layer;
            self.csg_layer_volume(i, true, Detail::Full)?;
            stage.tree_next = false;
            stage.layer += 1;
            return Ok(false);
        }
        if let Some(field) = stage.fields.get(stage.layer) {
            let i = stage.layer;
            self.baker.set_layer(i);
            match &field.shape {
                Shape::Field { family, preset, uniforms, table } => {
                    let family = &self.state.families[*family];
                    let preset = &family.presets[*preset];
                    let end = (stage.next_slice + FIELD_SLICES).min(field.resolution);
                    self.baker.field_slices(&format!("{}/{}", family.id, preset.id), preset, uniforms, table, field.resolution, stage.next_slice..end)?;
                    stage.next_slice = end;
                    if end < field.resolution {
                        self.baker.set_layer(0);
                        return Ok(false);
                    }
                    stage.next_slice = 0;
                    let field = &stage.fields[i];
                    self.layer_done(i, field, None);
                }
                Shape::Whole => {
                    // (the shape worked out in one frame, built the next: a tube's points cost several
                    // milliseconds of CPU, its volume tens of the GPU's)
                    let Some(job) = stage.job.take() else {
                        let mut job = self.state.layer_job(i, Detail::Full, None)?;
                        job.prepare_formula(&self.state.families)?;
                        let res = self.state.layer_build_res(i, Detail::Full);
                        self.share_formula_table(i, Detail::Full, &job, res);
                        stage.job = Some(job);
                        self.baker.set_layer(0);
                        return Ok(false);
                    };
                    let melt = match &job.shape {
                        // (its tree stands for it: no volume of its own is made)
                        _ if self.own_volume_unread(i) => {
                            self.baker.clear_layer(i);
                            None
                        }
                        ShapeJob::Curve { curve, melt } => {
                            self.baker.curve(curve, job.res, *melt)?;
                            Some(*melt)
                        }
                        ShapeJob::Field { uniforms, table } => {
                            let family = &self.state.families[job.family];
                            let preset = &family.presets[job.preset];
                            self.baker.field(&format!("{}/{}", family.id, preset.id), preset, uniforms, table, job.res)?;
                            None
                        }
                        ShapeJob::Mesh(_) => return Err("a model is not built in pieces".into()),
                    };
                    self.layer_done(i, field, melt);
                    // (a tube whose seam now needs its wide volume: the ordinary build makes it)
                    if let Some(m) = melt {
                        if self.needs_wide(&stage.combination, i, field, exact_to(field.resolution, m).1) {
                            return Err("a tube needs its wide volume".into());
                        }
                    }
                }
            }
            self.baker.set_layer(0);
            if stage.fields[i].tree {
                stage.tree_next = true;
            } else {
                stage.layer += 1;
            }
            return Ok(false);
        }
        // every layer done: the combined volume, a band of slices a frame
        let n = stage.fields.len();
        let volumes: Vec<Volume> = (0..n)
            .map(|i| {
                let tree = if stage.fields[i].tree { self.csg.layer_vols.get(i).and_then(|v| v.as_ref()).map(|v| &v.1) } else { None };
                tree.or_else(|| self.baker.layer_volume(i))
                    .cloned()
                    .ok_or_else(|| format!("layer {} has no completed volume", i + 1))
            })
            .collect::<R<_>>()?;
        let refs: Vec<&Volume> = volumes.iter().collect();
        let end = (stage.next_slice + COMBINE_SLICES).min(stage.resolution);
        let wides: Vec<Option<(&Volume, WideUse)>> = vec![None; n];
        match &stage.combination {
            Combining::One(c) => self.layers.combiner.run_slices(&refs, &wides, c, stage.resolution, stage.next_slice..end)?,
            Combining::Passes(plan) => {
                let channels = if self.state.stack.body_channels() == 2 { 2 } else { 4 };
                self.layers.combiner.run_passes_slices(&refs, &wides, plan, &[], channels, stage.resolution, stage.next_slice..end)?
            }
        }
        stage.next_slice = end;
        if end < stage.resolution {
            return Ok(false);
        }

        // Let the normal full bake install these volumes and configure the tracer, optics, cuts and
        // floor in their usual order. Its cache check recognizes this completed combination.
        let sizes: Vec<u32> = volumes.iter().map(|v| v.res).collect();
        let uses: Vec<Option<WideUse>> = vec![None; n.max(W)];
        match std::mem::replace(&mut stage.combination, Combining::Passes(Vec::new())) {
            Combining::One(c) => {
                self.layers.last = Some((c, stage.resolution, sizes, uses));
                self.layers.last_multi = None;
            }
            Combining::Passes(plan) => {
                self.layers.last = None;
                self.layers.last_multi = Some((plan, stage.resolution, sizes, uses));
            }
        }
        self.layers.group_key = None;
        self.layers.combines += 1;
        Ok(true)
    }
}

#[cfg(test)]
#[path = "staged_layers_test.rs"]
mod tests;
