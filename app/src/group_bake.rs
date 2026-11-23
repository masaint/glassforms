//! Evaluate modifier stacks after the children of a group have been combined.
//! Nested groups hand their modified distance field to their parent. Separate
//! material bodies keep independent distance fields throughout.
use crate::{bake::Volume, combine::Combiner, gpu::Gpu, sculpt::Sculptor};
use gf_core::{
    groups::{GroupOp, Node},
    layers::{Combine, Op, PairOpts, Placement, Stack, WideUse, MAX_LAYERS},
    mods,
    state::State,
};

struct Input {
    original: usize,
    volume: Volume,
    wide: Option<(Volume, WideUse)>,
    placement: Placement,
    shown: bool,
}

fn subset(stack: &Stack, inputs: &[Input], unit: f64) -> Stack {
    let mut part = stack.clone();
    part.hierarchy = None;
    // World occurrences have already been collected into their source fields.
    part.world_mods.clear();
    let mut to = vec![None; stack.len()];
    for (i, input) in inputs.iter().enumerate() { to[input.original] = Some(i); }
    part.remap(&to);
    for (layer, input) in part.layers.iter_mut().zip(inputs) {
        layer.show = input.shown;
        layer.place = input.placement;
        layer.place.pos = layer.place.pos.map(|x| x / unit);
        layer.place.size /= unit;
    }
    if unit != 1.0 {
        for mask in part.combos() {
            if mask.count_ones() == 2 {
                let pair = part.pair_mut(mask).unwrap();
                pair.radius /= unit;
                pair.gap /= unit;
            } else if let Some(j) = part.junction_mut(mask) {
                j.reach /= unit;
            }
        }
    }
    part
}

fn combine(combiner: &mut Combiner, inputs: &[Input], params: &Combine, res: u32) -> Result<(), String> {
    let volumes: Vec<_> = inputs.iter().map(|i| &i.volume).collect();
    let wides: Vec<_> = inputs.iter().map(|i| i.wide.as_ref().map(|(v,w)| (v, *w))).collect();
    combiner.run(&volumes, &wides, params, res)
}

struct Groups<'a> {
    state: &'a State,
    sources: &'a [&'a Volume],
    wides: &'a [Option<(&'a Volume, WideUse)>],
    res: u32,
    // (groups with modifiers combine at most COMBINE_WIDTH objects: `bake_layers`)
    body: [usize; gf_core::layers::COMBINE_WIDTH],
    scene_scale: f64,
    projected: [bool;MAX_LAYERS],
    moving: bool,
    temp: &'a mut Combiner,
    sculptor: &'a mut Sculptor,
}

impl Groups<'_> {
    fn fields(&mut self, nodes: &[Node]) -> Result<Vec<Input>, String> {
        let mut fields = Vec::new();
        for node in nodes {
            match node {
                Node::Object(i) => {
                    let layer = self.state.stack.layers[*i];
                    fields.push(Input {
                        original: *i, volume: self.sources[*i].clone(),
                        wide: self.wides[*i].map(|(v,w)| (v.clone(), w)),
                        placement: if self.projected[*i] {Placement {size:self.scene_scale,..Default::default()}} else {layer.place}, shown: layer.show,
                    });
                }
                Node::Group(group) => {
                    let children = self.fields(&group.children)?;
                    let bake = group.mods.bake(&mods::whole_box());
                    if bake.is_none() && group.operation == GroupOp::Joins {
                        fields.extend(children);
                        continue;
                    }
                    let phases: Vec<_> = bake.as_ref().and_then(|b| b.lab()).into_iter().flat_map(|s| &s.layers).map(|l|
                        l.period().map_or(0.0, |p| std::f64::consts::TAU * self.state.cycles(p))
                    ).collect();
                    let mut bodies: [Vec<Input>; MAX_LAYERS] = Default::default();
                    for input in children {
                        let body = if group.operation == GroupOp::Joins { self.body[input.original] } else { 0 };
                        bodies[body].push(input);
                    }
                    for (body, inputs) in bodies.iter().enumerate().filter(|(_, v)| !v.is_empty()) {
                        let shown = inputs.iter().any(|i| i.shown);
                        let mut local = subset(&self.state.stack, inputs, self.scene_scale);
                        local.box_held = Some(1.0);
                        // A hidden group can still cut a visible sibling.
                        if !shown { for l in &mut local.layers { l.show = true; } }
                        if group.operation != GroupOp::Joins {
                            for mask in local.combos() {
                                if let Some(pair) = local.pair_mut(mask) {
                                    let op = match group.operation {
                                        GroupOp::Subtract => Op::BCutsA,
                                        GroupOp::Intersect => Op::Overlap,
                                        _ => Op::Join,
                                    };
                                    *pair = PairOpts { op, radius: 0.0, ..Default::default() };
                                } else if let Some(j) = local.junction_mut(mask) { j.fill = 0.0; }
                            }
                            if group.operation == GroupOp::Subtract {
                                for cutter in local.layers.iter_mut().skip(1) { cutter.show = false; }
                            }
                        }
                        combine(self.temp, inputs, &local.combine(), self.res)?;
                        let volume = self.temp.take_output();
                        let place = Placement { size: self.scene_scale, ..Default::default() };
                        if let Some(bake) = &bake {
                            let key = format!("group-{}-{body}|{}|{}", group.id, self.res, self.moving);
                            let slot = crate::sculpt::GROUP_SLOT + body;
                            self.sculptor.keep(slot, &volume, &key)?;
                            self.sculptor.run(slot, &volume, &key, bake, &phases, false, self.moving, &place)?;
                        }
                        fields.push(Input { original: inputs[0].original, volume, wide: None, placement: place, shown });
                    }
                }
            }
        }
        Ok(fields)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    state: &State, gpu: &Gpu, sources: &[&Volume], wides: &[Option<(&Volume, WideUse)>],
    projected: &[bool;MAX_LAYERS], params: &Combine, res: u32, moving: bool, sculptor: &mut Sculptor,
    scratch: &mut Option<Combiner>, output: &mut Combiner,
) -> Result<(), String> {
    let hierarchy = state.stack.hierarchy.as_ref().ok_or("missing group hierarchy")?;
    let temp = scratch.get_or_insert_with(|| Combiner::new(gpu));
    temp.bodies_wanted = false;
    let mut groups = Groups { state, sources, wides, res, body: params.body, scene_scale: params.box_k, projected:*projected, moving, temp, sculptor };
    let inputs = groups.fields(&hierarchy.0)?;
    let projected = subset(&state.stack, &inputs, 1.0);
    let mut root = projected.combine();
    root.box_k = params.box_k;
    root.bodies = params.bodies;
    for (i, input) in inputs.iter().enumerate() { root.body[i] = params.body[input.original]; }
    combine(output, &inputs, &root, res)
}

/// The same ordered Boolean as the group bake, evaluated from its leaf formulas
/// at a ray hit. The combined volume is still the guide for the long march.
struct GroupFormula {
    parts: Vec<gf_core::csg::Resolved>,
    layers: Vec<usize>,
    expr: String,
}

impl GroupFormula {
    fn read(state: &State) -> Option<Self> {
        let outline = state.stack.hierarchy.as_ref()?;
        if state.edit_scope.object().is_some() || state.stack.has_world_modifiers() || !outline.changes_geometry() || state.stack.bodies().len() != 1 { return None; }
        let mut parts = Vec::new();
        let mut layers = Vec::new();
        fn visit(node: &Node, state: &State, parts: &mut Vec<gf_core::csg::Resolved>, layers: &mut Vec<usize>) -> Option<String> {
            match node {
                Node::Object(i) => {
                    let layer = &state.stack.layers[*i];
                    let csg = state.layer_csg(*i);
                    if !layer.show || !state.is_csg_object(*i) || csg.coarse || !csg.place.is_identity() || state.tree_mods(*i).is_some() { return None; }
                    if state.layer_csg(*i).construction.is_some() { return None; }
                    let (mut object, joined, omitted) = state.csg_tree_layer(*i);
                    if joined || omitted != 0 || object.len() != 1 { return None; }
                    let mut part = object.pop()?;
                    if part.grid.is_some() || !part.place.is_identity() { return None; }
                    part.place = layer.place;
                    let slot = parts.len();
                    parts.push(part);
                    layers.push(*i);
                    Some(format!("csgPartAt({slot}, csgLocal({slot}, p), false) * csgP[{slot}].m.x"))
                }
                Node::Group(group) => {
                    if group.mods.any_live() { return None; }
                    let mut children = group.children.iter();
                    let mut value = visit(children.next()?, state, parts, layers)?;
                    for child in children {
                        let rhs = visit(child, state, parts, layers)?;
                        value = match group.operation {
                            GroupOp::Union => format!("min({value}, {rhs})"),
                            GroupOp::Subtract => format!("max({value}, -({rhs}))"),
                            GroupOp::Intersect => format!("max({value}, {rhs})"),
                            GroupOp::Joins => return None,
                        };
                    }
                    if group.operation != GroupOp::Joins {
                        // The group is assembled in a cube whose half-width is 1.5,
                        // then placed in the scene at the current stage scale.
                        value = format!("groupClip({value}, p, {} * HB / 1.5)", gf_core::layers::STAGE_SOFT);
                    }
                    Some(value)
                }
            }
        }
        let [root] = outline.0.as_slice() else { return None };
        let body = visit(root, state, &mut parts, &mut layers)?;
        let expr = format!("groupClip({body}, p, {})", gf_core::layers::STAGE_SOFT);
        Some(Self { parts, layers, expr })
    }

    fn shader(&self, tree: &mut String) {
        let start = tree.find("fn csgDistW(").expect("CSG distance");
        let end = tree[start..].find("fn csgDist(p:").expect("CSG surface") + start;
        let code = format!("fn groupClip(d: f32, p: vec3f, soft: f32) -> f32 {{
  let stage = length(p) - {} * HB / 1.5;
  let t = max(1.0 - abs(d - stage) / soft, 0.0);
  return max(d, stage) + soft * t * t * t / 6.0;
}}
fn csgDistW(p: vec3f, only: i32) -> vec2f {{ return vec2f({}, 0.0); }}
", gf_core::layers::STAGE_R, self.expr);
        tree.replace_range(start..end, &code);
    }
}

impl crate::engine::Engine {
    pub(crate) fn set_group_formula(&mut self, res: u32, mode: gf_core::state::Detail) -> Result<bool, String> {
        use gf_core::csg;
        let Some(mut group) = GroupFormula::read(&self.state) else { return Ok(false) };
        self.csg_formula_tables(|i| group.layers[i], &mut group.parts, mode, None, &[])?;
        let fams = &self.state.families;
        let parts = &group.parts;
        let kinds = csg::kinds(parts);
        let strokes = csg::any_strokes(parts);
        let key = format!("{}|group:{}", csg::key_tree_s(fams, &kinds, true, strokes), group.expr);
        let decl = (self.renderer.sdf_key() != Some(key.as_str())).then(|| {
            let mut tree = csg::wgsl_tree_s(fams, &kinds, gf_core::sdf::BINDING, false, true, strokes);
            group.shader(&mut tree);
            tree
        });
        let numbers = csg::uniforms(fams, parts, &kinds, res, false, &[], &Placement::default());
        self.csg.set_table(&numbers);
        self.csg.set_paths(&csg::path_data(fams, parts).0);
        self.renderer.set_grids(&[], &self.csg.lists, &self.csg.table);
        self.renderer.set_sdf(Some((&key, decl, &numbers)));
        Ok(true)
    }
}
