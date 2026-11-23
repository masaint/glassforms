//! Compile authored construction groups to callable distance fields. The same result is
//! read by the guide bake and the exact tracer. Each group evaluates before its parent.
use crate::construction::{Construction, CutMaterial, GroupMode, Join, Node, Source};
use crate::csg::{Csg, GridFrom, Op, Resolved, PART_FLOATS};
#[path = "construction_pattern.rs"]
mod pattern;
use crate::weave;
use std::fmt::Write;
use std::hash::{DefaultHasher, Hash, Hasher};

const EMPTY: &str = "vec3f(1e6, 0.0, -1.0)";
const FIRST: usize = crate::csg::FORMULA_AT * 4 / PART_FLOATS;

/// Optical properties and the frame in which this construction layer was authored.
#[derive(Clone)]
pub struct FaceMaterial {
    pub settings: crate::settings::Settings,
    pub object_to_face: [f64; 12],
}

/// The topology is a shader key; changing a blend or material only uploads numbers.
#[derive(Default)]
pub struct Program {
    source: String,
    pub numbers: Vec<f32>,
    codes: Vec<u32>,
    root: usize,
    pub has_materials: bool,
    pub materials: Vec<crate::material::Variant>,
    /// The complete material for each authored construction face in this object.
    pub faces: Vec<FaceMaterial>,
    pub boundaries: Vec<Boundary>,
}

/// One subtree, available as a cached result at the next level. Distances near an
/// analytic surface can still be refined from the original functions.
pub struct Boundary {
    pub transport_faces: bool,
    pub result: usize,
    pub wrapper: usize,
    pub placement: crate::layers::Placement,
    pub signature: u64,
    pub bake: Option<crate::mods::Bake>,
    pub slot: Option<usize>,
    pub res: u32,
}
struct Builder<'a> {
    csg: &'a Csg,
    material: &'a crate::settings::Settings,
    parts: &'a [Resolved],
    extents: &'a [crate::csg::Extent],
    isolate: Option<&'a [usize]>,
    active: &'a [usize],
    parents: bool,
    program: Program,
    serial: usize,
}

struct CompiledNode {
    function: usize,
    entry: usize,
    /// joined by `cnJoinP` (its record's k holds the profile, band, depth and width)
    profiled: bool,
    /// Result bounds, expressed in the immediate parent's coordinates.
    bounds: crate::mods::Box3,
}

fn combined_bounds(children: &[CompiledNode]) -> crate::mods::Box3 {
    let Some((first, rest)) = children.split_first() else { return crate::mods::whole_box(); };
    rest.iter().fold(first.bounds, |b, child| std::array::from_fn(|k| {
        if k < 3 { b[k].min(child.bounds[k]) } else { b[k].max(child.bounds[k]) }
    }))
}

fn placed_bounds(b: crate::mods::Box3, local: crate::layers::Placement, blend: f64) -> crate::mods::Box3 {
    let frame = crate::modifier_pivot::PivotFrame { placement: local, ..Default::default() };
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for mask in 0..8 {
        let point = frame.to_world(std::array::from_fn(|k| b[if mask & (1 << k) == 0 { k } else { k + 3 }]));
        for k in 0..3 {
            lo[k] = lo[k].min(point[k]);
            hi[k] = hi[k].max(point[k]);
        }
    }
    std::array::from_fn(|k| if k < 3 { lo[k] - blend } else { hi[k - 3] + blend })
}

impl Program {
    pub fn build(
        csg: &Csg,
        tree: &Construction,
        material: &crate::settings::Settings,
        parts: &[Resolved],
        active: &[usize],
        isolate: Option<&[usize]>,
        parents: bool,
    ) -> Self {
        Self::build_measured(csg, tree, material, parts, &[], active, isolate, parents)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_measured(
        csg: &Csg, tree: &Construction, material: &crate::settings::Settings,
        parts: &[Resolved], extents: &[crate::csg::Extent], active: &[usize],
        isolate: Option<&[usize]>, parents: bool,
    ) -> Self {
        let mut b = Builder {
            csg,
            extents,
            material,
            parts,
            isolate,
            active,
            parents,
            program: Self::default(),
            serial: 0,
        };
        let mut children = Vec::new();
        for (i, node) in tree.roots.iter().enumerate() {
            b.node(node, &mut vec![i], csg.place, true, None, &mut children);
        }
        b.program.root = b.group(&children, isolate.is_some());
        b.program
    }

    pub fn key(&self, base: &str) -> String {
        let mut hash = DefaultHasher::new();
        self.source.hash(&mut hash);
        self.codes.hash(&mut hash);
        self.materials.hash(&mut hash);
        for b in &self.boundaries {
            b.transport_faces.hash(&mut hash);
            b.bake.is_some().hash(&mut hash);
            b.slot.hash(&mut hash);
            b.res.hash(&mut hash);
        }
        format!("{base}|construction:{:x}", hash.finish())
    }

    /// Include all values for a cached evaluated object, even when they share a shader.
    pub fn field_key(&self) -> u64 {
        let mut hash = DefaultHasher::new();
        self.source.hash(&mut hash);
        for b in &self.boundaries {
            b.signature.hash(&mut hash);
        }
        for x in &self.numbers {
            x.to_bits().hash(&mut hash);
        }
        hash.finish()
    }

    pub fn shader(&self, tree: String, binding: u32, walked: bool) -> String {
        let mut src = self.subtree_shader(tree, binding, walked, self.root);
        if self.has_materials && !walked {
            writeln!(
                src,
                "\n// construction surface materials {}",
                serde_json::to_string(&self.materials).unwrap()
            )
            .unwrap();
        }
        src
    }

    pub fn subtree_shader(&self, tree: String, binding: u32, walked: bool, root: usize) -> String {
        let mut text = weave::replacen(&tree, "construction_eval.legacy", "fn csgDistW(", "fn cnLegacy(", 1);
        if !self.codes.is_empty() {
            let lib = crate::mods::lib_for(&self.codes);
            text += &crate::sdf::prefix_names(&lib, &crate::sdf::top_names(&lib), "cn_");
            text+="\nfn cnMod(i: u32) -> cn_ModE { let r=csgP[i]; return cn_ModE(r.inv0,r.inv1,r.inv2,r.b,r.m,r.k,r.v0,r.v1,r.e); }\n";
        }
        text += JOIN;
        // (joins by a seam profile or a band: only when a node uses one, so other trees keep their text)
        if self.source.contains(JOIN_P_CALL) {
            text += crate::join::WGSL;
        }
        let mut src = self.source.clone();
        let slots = self
            .boundaries
            .iter()
            .filter(|b| b.slot.is_some() && (walked || b.bake.is_some()));
        for b in slots {
            if !text.contains("fn csgGridVol(") {
                text += &crate::csg::grid_text(binding, walked);
            }
            let marker = format!("/*cache{}*/", b.wrapper);
            let read = format!(
                "let v=csgGridVol({}, p/(2.0*csgU.info.y)+0.5);",
                b.slot.unwrap()
            );
            let fallback = if b.transport_faces {
                let slot=b.slot.expect("bound construction material atlas");
                let r=b.res;
                let n=r.min(128);
                let area=r*r;
                // The original signed distance occupies the first r slices.
                // Two further half texels encode each unsigned surface-table ID.
                let depth=r+(2*n.pow(3)).div_ceil(area);
                format!("let uv=clamp(p/(2.0*csgU.info.y)+0.5,vec3f(0.5/{r}.0),vec3f(1.0-0.5/{r}.0));
                    let distance=textureSampleLevel(csgG{slot},csgGSmp,uv*vec3f(1.0,1.0,{r}.0/{depth}.0),0.0).r;
                    let cell=vec3u(uv*{n}.0); let i=2u*(cell.x+{n}u*(cell.y+{n}u*cell.z));
                    let coord=vec3u(i%{r}u,(i/{r}u)%{r}u,{r}u+i/{area}u);
                    let lo=textureLoad(csgG{slot},coord,0).r;
                    let hi=textureLoad(csgG{slot},coord+vec3u(1,0,0),0).r;
                    return vec3f(distance,0.0,lo+256.0*hi-1.0);")
            } else if b.bake.is_some() {
                // Voxel-only modifiers must use the edited surface even in the tracer.
                format!("{read} return vec3f(v,0.0,cn{}(p).z);", b.result)
            } else {
                // The guide reads a conservative bound away from the zero set. Its
                // interpolation band and the exact tracer retain the source surface.
                format!("{read} let h=2.0*csgU.info.y/{}.0; if(abs(v)>4.0*h) {{return vec3f(v-sign(v)*1.75*h,0.0,-1.0);}} return cn{}(p);",b.res,b.result)
            };
            src = weave::replace(&src, "construction_eval.cache", &format!("{marker} return cn{}(p);", b.result), &fallback);
        }
        text += &src;
        writeln!(
            text,
            "fn cnSample(p: vec3f) -> vec3f {{ return cn{}(p); }}",
            root
        )
        .unwrap();
        text+="fn csgDistW(p: vec3f, only: i32) -> vec2f { if (only >= 0) { return cnLegacy(p, only); } return cnSample(p).xy; }\n";
        text
    }
}

const JOIN: &str = "
fn cnJoin(a: vec3f, b: vec3f, control: vec4f) -> vec3f {
  let op=i32(control.x);
  if (b.x >= 1e5) { return select(a,b,op == 2); }
  if (a.x >= 1e5) { return select(a,b,op == 0); }
  let answer=csgJoin(a.x,a.y,b.x,b.y,op,control.y);
  let below = b.x < a.x;
  let above = select(b.x,-b.x,op == 1) > a.x;
  let wins = select(below,above,op != 0);
  let material=select(a.z,b.z,wins && !(op == 1 && control.z > 0.5));
  return vec3f(answer,material);
}
";
/// A node joined by its seam profile or band (gf_core::join::WGSL), reading the node's record k =
/// (profile, band, depth, width) beside inv0.
const JOIN_P_CALL: &str = "=cnJoinP(";

/// Does this join need anything the plain `cnJoin` does not hold?
fn joins_by_profile(j: &Join) -> bool {
    !j.profile.is_default() || !j.band.is_none()
}

/// A join as a node's record holds it: inv0 = (operation, blend, 1 if the cut keeps the recipient's
/// material, -), k = (profile, band, depth, width); blend, depth and width in the scene's units (`scale`: the
/// size of the node the join is that node's own, when they are relative to it). Returns how far the join may
/// reach beyond the node's own box (its blend, and a bead's width).
fn record_join(rec: &mut [f32; PART_FLOATS], j: &Join, scale: f64) -> f64 {
    let unit = if j.blend_relative { scale } else { 1. };
    let op = j.band.op().unwrap_or(j.op);
    rec[0] = match op {
        Op::Union => 0.,
        Op::Subtract => 1.,
        Op::Intersect => 2.,
    };
    rec[1] = (j.blend.max(0.) * unit) as f32;
    rec[2] = f32::from(j.cut_material == CutMaterial::Recipient);
    // (k: read by cnJoinP alone; all zero for a plain join, as the record always was)
    rec[20] = j.profile.code();
    if !j.band.is_none() {
        rec[21] = j.band.code();
        rec[22] = (j.depth.max(0.) * unit) as f32;
        rec[23] = if j.band.along() { (j.width.max(0.) * unit) as f32 } else { 0. };
    }
    f64::from(rec[1]) + if j.band == crate::join::Band::Bead { f64::from(rec[23]) } else { 0. }
}
impl<'a> Builder<'a> {
    fn next(&mut self) -> usize {
        let id = self.serial;
        self.serial += 1;
        id
    }

    fn group(&mut self, children: &[CompiledNode], isolated: bool) -> usize {
        let id = self.next();
        let text = &mut self.program.source;
        writeln!(text, "fn cn{id}(p: vec3f) -> vec3f {{ var d={EMPTY};").unwrap();
        for (index, compiled) in children.iter().enumerate() {
            let (child, data) = (compiled.function, compiled.entry);
            if isolated && index == 0 {
                writeln!(text, "  d=cn{child}(p);").unwrap();
            } else if compiled.profiled {
                writeln!(text, "  d{JOIN_P_CALL}d,cn{child}(p),csgP[{data}].inv0,csgP[{data}].k);").unwrap();
            } else {
                writeln!(text, "  d=cnJoin(d,cn{child}(p),csgP[{data}].inv0);").unwrap();
            }
        }
        *text += "  return d; }\n";
        id
    }

    /// The node compiled, joined by its own join or (`over`) a group's, with that group's size. A group whose
    /// children join straight into its parent (`Node::inlines`) puts them in `out` in its place.
    fn node(&mut self, node: &Node, path: &mut Vec<usize>, parent: crate::layers::Placement, surface_visible: bool, over: Option<(Join, f64)>, out: &mut Vec<CompiledNode>) {
        if self
            .isolate
            .is_some_and(|selected| !selected.starts_with(path) && !path.starts_with(selected))
        {
            return;
        }
        if node.inlines() {
            if let Source::Group(children) = &node.source {
                // passthrough: each child by its own join (or an override from above, which this level is
                // under); overridden: each by the group's own, unless one from above overrides that too
                let placed = crate::construction_pivot::combine(parent, node.placement(self.csg));
                let over = match node.mode {
                    GroupMode::Overridden => over.or(Some((node.join(), placed.size.abs()))),
                    _ => over,
                };
                let mut inner = Vec::new();
                for (i, child) in children.iter().enumerate() {
                    path.push(i);
                    self.node(child, path, placed, surface_visible, over, &mut inner);
                    path.pop();
                }
                // (their bounds, from the group's coordinates to its parent's)
                let local = node.placement(self.csg);
                out.extend(inner.into_iter().map(|c| CompiledNode { bounds: placed_bounds(c.bounds, local, 0.), ..c }));
                return;
            }
        }
        let entry = FIRST + self.program.numbers.len() / PART_FLOATS;
        let mut rec = [0.0; PART_FLOATS];
        let index = match &node.source {
            Source::Base => self.parts.iter().position(|p| p.source == GridFrom::Up),
            Source::Part(id) => self.csg.construction_part(*id).and_then(|slot| {
                self.parts
                    .iter()
                    .position(|p| p.source == GridFrom::Part(slot))
            }),
            Source::Group(_) => None,
        };
        let placed = crate::construction_pivot::combine(parent, node.placement(self.csg));
        // its own join, or the overriding group's (in that group's units)
        let (join, join_scale) = over.unwrap_or((node.join(), placed.size.abs()));
        let reach = record_join(&mut rec, &join, join_scale);
        let join_op = join.band.op().unwrap_or(join.op);
        let mut optical = Vec::new();
        // A recipient-owned boolean keeps its left operand's material. Isolating
        // the cutter bypasses that join and reveals the dormant cutter recipe.
        let isolated_operand = self.isolate.is_some_and(|selected| selected.starts_with(path));
        let surface_visible = surface_visible &&
            (isolated_operand || join_op != Op::Subtract || join.cut_material != CutMaterial::Recipient);
        let face_material = node.material.as_ref().filter(|_| surface_visible);
        if let Some(mat) = face_material {
            self.program.has_materials = true;
            rec[3] = 1.;
            // Store physical optical coefficients in the same units as the body's
            // uniform. Negative entries inherit the live, fitted object material.
            let mut own = self.material.clone();
            for (k, v) in mat {
                own.set(k, *v);
            }
            for (axis, (body, tint)) in [("bodyR", "tintR"), ("bodyG", "tintG"), ("bodyB", "tintB")]
                .iter()
                .enumerate()
            {
                rec[4 + axis] = mat
                    .get(*body)
                    .map_or(-1., |_| own.get(body).unwrap() as f32);
                rec[8 + axis] = if mat.contains_key(*tint) || mat.contains_key("density") {
                    (own.absorb[axis] * own.density) as f32
                } else {
                    -1.
                };
            }
            rec[12] = mat.get("ior").map_or(-1., |_| own.ior as f32);
            rec[13] = mat.get("disp").map_or(-1., |_| own.disp as f32);
            // The child has no private diffusion volume. A translucent cut surface
            // uses the native per-path random walk through the final geometry.
            let kind = crate::material::body_kind(&own);
            if kind == 2. {
                own.sss_walk = 1.;
            }
            let variant = crate::material::Variant::of(&own);
            let class = self
                .program
                .materials
                .iter()
                .position(|v| *v == variant)
                .unwrap_or_else(|| {
                    self.program.materials.push(variant);
                    self.program.materials.len() - 1
                });
            rec[14] = (class + 1) as f32;
            rec[15] = kind;
            // Zero is the ordinary object material in the optional World provenance grid.
            let object_to_face = placed.inverse();
            self.program.faces.push(FaceMaterial { settings: own.clone(), object_to_face });
            rec[16] = self.program.faces.len() as f32;
            if !variant.plain() {
                let mut material = crate::material::uniforms(&own, 1.);
                crate::material::set_grating_frame(&mut material, &object_to_face);
                optical.extend_from_slice(&material);
                if variant.inner.any() {
                    let absorbed = own.absorb.map(|a| a * own.density);
                    optical.extend_from_slice(&crate::inner::uniforms(&own, 1., 0.002, absorbed));
                }
                optical.resize(optical.len().div_ceil(PART_FLOATS) * PART_FLOATS, 0.);
            }
        }
        self.program.numbers.extend_from_slice(&rec);
        self.program.numbers.extend_from_slice(&optical);
        let (id, shape) = match &node.source {
            Source::Base | Source::Part(_) => {
                let Some(slot) = index else { return };
                let id = self.next();
                writeln!(self.program.source,"fn cn{id}(p: vec3f) -> vec3f {{ let d=csgPartAt({slot},csgLocal({slot},p),false)*csgP[{slot}].m.x; return vec3f(d,{}.0,-1.0); }}",slot+1).unwrap();
                let mut local = self.parts[slot].clone();
                local.place = crate::layers::Placement::default();
                local.smooth = 0.;
                let measured = self.extents.get(slot).map_or(&[][..], std::slice::from_ref);
                (id, crate::csg::tree_box(&[local], measured))
            }
            Source::Group(children) => {
                let mut compiled = Vec::new();
                for (i, child) in children.iter().enumerate() {
                    path.push(i);
                    self.node(child, path, placed, surface_visible && face_material.is_none(), None, &mut compiled);
                    path.pop();
                }
                let isolated = self
                    .isolate
                    .is_some_and(|s| s.starts_with(path) && s.len() > path.len());
                (self.group(&compiled, isolated), combined_bounds(&compiled))
            }
        };
        let has_parent = self.active.starts_with(path) && self.active.len() > path.len();
        let mut current = id;
        let mut fallback = None;
        let mut result_bounds = shape;
        if node.mods.any_live() && (!has_parent || self.parents) {
            result_bounds = node.mods.live().fold(shape, |b, stage| stage.bound(&b));
            if !node.mods.exact(&shape) {
                fallback = node.mods.bake(&shape);
            } else {
                let codes = node.mods.node_codes();
                let count = codes.len();
                for &code in &codes {
                    if !self.program.codes.contains(&code) {
                        self.program.codes.push(code);
                    }
                }
                let (nums, array_rows) = node.mods.node_numbers(&shape,&placed);
                let at = FIRST + self.program.numbers.len() / PART_FLOATS;
                self.program
                    .numbers
                    .extend_from_slice(&nums[..(count + 1) * PART_FLOATS]);
                // (a listed array's copies: its rows after the stack, `crate::array`)
                let listed = codes.contains(&crate::array::LISTED);
                let mut starts = Vec::new();
                for r in array_rows.iter().filter(|_| listed) {
                    starts.push(if r.is_empty() { 0 } else { FIRST + self.program.numbers.len() / PART_FLOATS });
                    self.program.numbers.extend_from_slice(r);
                }
                let wid = self.next();
                let text = &mut self.program.source;
                if codes == [8] && nums[2] != 0. {
                    let rows = FIRST + self.program.numbers.len() / PART_FLOATS;
                    self.program.numbers.extend(crate::mods::profile_rows(&nums));
                    pattern::overlapping_repeat(text, id, wid, at, rows);
                } else if codes.contains(&12) || listed {
                    pattern::wrap_stack(text, id, wid, at, &codes, &starts);
                } else {
                    writeln!(text,"fn cn{wid}(p: vec3f) -> vec3f {{ let H=cnMod({at}u); let h=vec4f(p,1.0); var x=vec3f(dot(H.q1,h),dot(H.q2,h),dot(H.q3,h));").unwrap();
                    for j in (0..count).rev() {
                        writeln!(text,"  let m{j}=cnMod({}u); var w{j}=1e9; x=cn_modWorld(m{j},cn_modBack(m{j},cn_modLocal(m{j},x),&w{j}));",at+j+1).unwrap();
                    }
                    writeln!(text,"  let q=vec4f(x,1.0); var v=cn{id}(vec3f(dot(H.q4,q),dot(H.q5,q),dot(H.q6,q))); v.x/=H.q0.y;").unwrap();
                    for j in 0..count {
                        writeln!(text, "  v.x=cn_modOn(m{j},v.x,w{j});").unwrap();
                    }
                    *text += "  return vec3f(v.x*H.q0.y,0.0,v.z); }\n";
                }
                current = wid;
            }
        }
        if face_material.is_some() {
            let id = self.next();
            writeln!(self.program.source,"fn cn{id}(p: vec3f) -> vec3f {{ let a=cn{current}(p); return vec3f(a.xy,{entry}.0); }}").unwrap();
            current = id;
        }
        if matches!(node.source, Source::Group(_)) || fallback.is_some() {
            let group = self.next();
            writeln!(
                self.program.source,
                "fn cn{group}(p: vec3f) -> vec3f {{ /*cache{group}*/ return cn{current}(p); }}"
            )
            .unwrap();
            let mut hash = DefaultHasher::new();
            serde_json::to_string(node)
                .unwrap_or_default()
                .hash(&mut hash);
            let mut ids = Vec::new();
            fn collect(n: &Node, out: &mut Vec<Source>) {
                match &n.source {
                    Source::Group(c) => {
                        for n in c {
                            collect(n, out)
                        }
                    }
                    s => out.push(s.clone()),
                }
            }
            collect(node, &mut ids);
            for source in ids {
                let from = match source {
                    Source::Base => Some(GridFrom::Up),
                    Source::Part(id) => self.csg.construction_part(id).map(GridFrom::Part),
                    Source::Group(_) => None,
                };
                if let Some(part) = self.parts.iter().find(|p| Some(p.source) == from) {
                    part.sig().hash(&mut hash);
                    part.formula_key.hash(&mut hash);
                    serde_json::to_string(&part.place)
                        .unwrap_or_default()
                        .hash(&mut hash);
                    for value in &part.vals {
                        value.to_bits().hash(&mut hash);
                    }
                }
            }
            serde_json::to_string(&placed)
                .unwrap_or_default()
                .hash(&mut hash);
            // A preview can alter this field only when it isolates a descendant
            // or hides modifiers above a descendant selection. Navigation in a
            // sibling and selection with all modifiers visible reuse this field.
            let descendant = |selection: &'a [usize]| {
                selection.strip_prefix(path.as_slice()).filter(|suffix| !suffix.is_empty())
            };
            self.isolate.and_then(descendant).hash(&mut hash);
            (!self.parents).then_some(self.active).and_then(descendant).hash(&mut hash);
            fn has_face(n:&Node)->bool {
                n.material.is_some() || matches!(&n.source, Source::Group(children) if children.iter().any(has_face))
            }
            let transport_faces=surface_visible && fallback.is_some() && face_material.is_none() &&
                matches!(&node.source,Source::Group(children) if children.iter().any(has_face));
            self.program.boundaries.push(Boundary {
                transport_faces,
                placement: placed,
                result: current,
                wrapper: group,
                signature: hash.finish(),
                bake: fallback,
                slot: None,
                res: 0,
            });
            current = group;
        }
        let blend = reach / parent.size.abs().max(1e-9);
        out.push(CompiledNode { function: current, entry, profiled: joins_by_profile(&join), bounds: placed_bounds(result_bounds, node.placement(self.csg), blend) });
    }
}

impl crate::state::State {
    /// Ancestor stacks above the edited node are optional in a local preview.
    pub fn show_construction_parents(&self, layer: usize) -> bool {
        let scope = &self.edit_scope;
        scope.object() != Some(layer)
            || scope.preview_parents
            || (scope.construction().is_empty() && scope.isolated().is_none())
    }

    pub fn construction_program(&self, layer: usize, parts: &[Resolved]) -> Option<Program> {
        self.construction_program_measured(layer, parts, &[])
    }

    pub fn construction_program_measured(
        &self, layer: usize, parts: &[Resolved], extents: &[crate::csg::Extent],
    ) -> Option<Program> {
        let csg = self.layer_csg(layer);
        let tree = csg.construction.as_ref()?;
        let scope = &self.edit_scope;
        let editing = scope.object() == Some(layer);
        Some(Program::build_measured(
            csg,
            tree,
            self.body_settings(layer),
            parts,
            extents,
            if editing { scope.construction() } else { &[] },
            if editing {
                scope.isolated()
            } else {
                None
            },
            !editing || scope.preview_parents,
        ))
    }
}

pub fn shader(program: Option<&Program>, tree: String, binding: u32, walked: bool) -> String {
    match program {
        Some(p) => p.shader(tree, binding, walked),
        None => tree,
    }
}
