//! Constructive solid geometry on shapes' formulas: any number of parts, each a formula shape (studio
//! primitives, solids, surfaces, gems, forms, algebraic surfaces) with a placement of its own, combined in
//! order by booleans (union, subtract, intersect, each smooth by a radius of its own), and grouped:
//! a part a level deeper than the one before it opens a group, which is combined into what is above it,
//! as one part, by the operation of its first part.
//!
//! The tracer evaluates the whole tree itself (app/src/sdf.rs), near the surface; the volume baked
//! beside it holds the same tree measured coarsely (each part walked to its surface as the bake walks a
//! shape), so that rays skip empty space by one lookup and every pass that reads the volume (shadows,
//! the surface pass, the floor) sees the parts too.
//!
//! Each part has a sphere that holds it. A part whose sphere is further from a point than the result so
//! far (plus its seam's radius) cannot change the result there and is not evaluated; a point outside a
//! part's sphere is told the distance to the sphere instead of the part's own (never more than it). So a
//! step costs a few spheres' distances plus the parts actually near it.
//!
//! The shape up (the first layer) is the first part when it is traced as "csg"; otherwise it stays a
//! volume object, and the parts' result is joined to it as a union.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::field::{member_name, HELPERS, MAX_PARAMS};
use crate::layers::Placement;
use crate::presets::{Family, Preset, Shape};
use crate::sdf::{prefix_names, rename, support, top_names, Support, FINER, NORMAL_E, SLOPE_E, STEP_MAX};

/// The most parts a tree can have.
pub const MAX_PARTS: usize = 64;
/// The deepest a group may be nested.
pub const MAX_LEVEL: u8 = 3;
/// Floats of the uniform block: a head of four, then 36 a part.
pub const PART_FLOATS: usize = 36;
pub const UNIFORM_FLOATS: usize = 8 + PART_FLOATS * MAX_PARTS;
/// The volume is cut into this many cells a side, and each cell lists the parts that can matter in it
/// (`numbers`): a step evaluates those alone. With them, how near to the cell the nearest part left out of
/// it may come: the distance a step is told is held within that.
pub const CELLS: usize = 16;
/// A part is listed in every cell within this of its box (and its seam's radius); the distance a step is
/// told is held within it, or within how far the parts left out are, so that no part left out is ever
/// stepped past.
pub const CELL_PAD: f64 = 0.1;
/// How far a part may reach from its middle, in its own units: every shape is fitted within the tracer's
/// ball of 1.47 (a little more is allowed for its inflate slider and a wall about it). Its box is this
/// cube until it has been measured (`Extent`).
pub const PART_REACH: f64 = 1.56;
/// The box a part was measured to fill, in its own units: lowest corner, highest corner.
pub type Extent = [f32; 6];
/// Cells a side of the grid a part is measured on (app/src/csg.rs), and how a cell is judged to hold
/// some of the part: its distance at the cell's middle is under half the cell's diagonal and this.
pub const EXTENT_CELLS: u32 = 32;
pub const EXTENT_SLACK: f32 = 0.02;

/// How a part joins what is above it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Op {
    #[default]
    Union,
    Subtract,
    Intersect,
}
impl Op {
    pub const ALL: [Op; 3] = [Op::Union, Op::Subtract, Op::Intersect];
    pub fn id(self) -> &'static str {
        match self { Op::Union => "union", Op::Subtract => "subtract", Op::Intersect => "intersect" }
    }
    pub fn from_id(s: &str) -> Option<Op> {
        Op::ALL.into_iter().find(|o| o.id() == s)
    }
    fn code(self) -> f32 {
        match self { Op::Union => 0.0, Op::Subtract => 1.0, Op::Intersect => 2.0 }
    }
}

/// One part of the tree, as a setup keeps it: which shape, its sliders by name (a missing one is the
/// slider's starting value), how much it is inflated, where it sits, how it joins what is above it.
/// A shape with no formula (drawn from curves, or a model file) is a grid part: it is baked into a volume
/// of its own, `res` voxels a side, and the tracer reads that (its tube's thickness, melt and length, and a
/// model's file, are kept with it; each is left out of a setup while unused).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Part {
    pub family: String,
    pub preset: String,
    pub values: BTreeMap<String, f64>,
    pub puff: f64,
    pub place: Placement,
    pub op: Op,
    /// the radius over which the seam is rounded (0: a hard edge), in the scene's units, or (`smooth_rel`)
    /// as a share of the part's size
    pub smooth: f64,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub smooth_rel: bool,
    /// how deep in groups: a part one level deeper than the one before it opens a group
    pub level: u8,
    /// a grid part's voxels a side (0: `GRID_RES`)
    #[serde(skip_serializing_if = "is_zero")]
    pub res: u32,
    /// a grid part's tube: thickness, melt, length (None: the family's own)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thick: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub melt: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub len: Option<f64>,
    /// the model file of a part that is "your file" (the models' `file` shape)
    #[serde(skip_serializing_if = "String::is_empty")]
    pub model: String,
}
fn is_zero(v: &u32) -> bool { *v == 0 }
impl Default for Part {
    fn default() -> Self {
        Self { family: "studio".into(), preset: "sphere".into(), values: BTreeMap::new(), puff: 0.0, place: Placement::default(), op: Op::Union, smooth: 0.0, smooth_rel: false, level: 0, res: 0, thick: None, melt: None, len: None, model: String::new() }
    }
}

/// Voxels a side of a grid part's volume, unless the part says otherwise; and the least and most it may say.
pub const GRID_RES: u32 = 384;
pub const GRID_RES_MIN: u32 = 128;
pub const GRID_RES_MAX: u32 = 768;
/// The most grid parts a tree holds at once (each is a volume, and a texture the tracer reads), and the
/// most video memory their own volumes may take together. Grid parts past either are left out of the tree
/// (the panel says so). The shape up as a grid part reads its own volume: it takes no memory of its own.
pub const MAX_GRIDS: usize = 8;
pub const GRID_BUDGET: u64 = 2 << 30;
/// The kind every grid part is (in `kinds`): one reading of a volume serves them all.
pub const GRID_KIND: (usize, usize) = (usize::MAX, 0);
/// Where a tracer finds the grid parts' volumes (eight textures and a sampler), and where the passes that
/// measure the tree do.
pub const GRID_BINDING_TRACER: u32 = 50;
pub const GRID_BINDING_PASS: u32 = 10;

/// Bytes of a grid part's volume of `res` voxels a side (16-bit floats).
pub fn grid_bytes(res: u32) -> u64 {
    (res as u64).pow(3) * 2
}

/// A grid part's volume: which of the tracer's eight it is, its voxels a side, where it comes from, and
/// what it was made from (the same text, the same volume).
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub slot: usize,
    pub res: u32,
    pub from: GridFrom,
    pub sig: String,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GridFrom {
    /// the shape up's own volume (as built, and as the surface studio has shaped it)
    Up,
    /// baked for part i of `Csg::parts`
    Part(usize),
}

/// A part with its shape found: indices into the families, its sliders in the preset's order.
#[derive(Clone, Debug, PartialEq)]
pub struct Resolved {
    pub fam: usize,
    pub preset: usize,
    pub vals: Vec<f64>,
    pub puff: f64,
    pub place: Placement,
    pub op: Op,
    pub smooth: f64,
    pub smooth_rel: bool,
    pub level: u8,
    /// read from a volume of its own, not from a formula
    pub grid: Option<Grid>,
}

impl Resolved {
    /// The kind of part it is (`kinds`).
    pub fn kind(&self) -> (usize, usize) {
        if self.grid.is_some() { GRID_KIND } else { (self.fam, self.preset) }
    }
    /// What it is (its shape, sliders and inflation): the box measured for it is kept under this.
    pub fn sig(&self) -> String {
        match &self.grid {
            Some(g) => format!("grid|{}|{}|{}", g.sig, g.res, self.puff),
            None => format!("{}/{}|{:?}|{}", self.fam, self.preset, self.vals, self.puff),
        }
    }
}

impl Part {
    /// The part of this shape, at this place.
    pub fn of(fam: &Family, p: &Preset, vals: &[f64], puff: f64, place: Placement) -> Self {
        Self {
            family: fam.id.into(), preset: p.id.into(),
            values: p.params.iter().zip(vals).map(|(d, v)| (d.id.to_string(), *v)).collect(),
            puff, place, ..Default::default()
        }
    }
    /// Its shape found among the families; None if there is no such shape. A shape with no formula is a
    /// grid part (its slot is given by the tree).
    pub fn resolve(&self, fams: &[Family]) -> Option<Resolved> {
        let fam = fams.iter().position(|f| f.id == self.family)?;
        let preset = fams[fam].presets.iter().position(|p| p.id == self.preset)?;
        let p = &fams[fam].presets[preset];
        let vals = p.params.iter().map(|d| self.values.get(d.id).copied().unwrap_or(d.value)).collect();
        let grid = (!matches!(support(&fams[fam], p), Support::Formula { .. })).then(|| Grid { slot: 0, res: self.grid_res(), from: GridFrom::Part(0), sig: self.grid_sig() });
        Some(Resolved { fam, preset, vals, puff: self.puff, place: self.place, op: self.op, smooth: self.smooth.max(0.0), smooth_rel: self.smooth_rel, level: self.level.min(MAX_LEVEL), grid })
    }
    /// A grid part's voxels a side.
    pub fn grid_res(&self) -> u32 {
        let r = if self.res == 0 { GRID_RES } else { self.res.clamp(GRID_RES_MIN, GRID_RES_MAX) };
        r / 8 * 8
    }
    /// Everything a grid part's volume is made from, as text (its placement and inflation are not: the
    /// tracer applies them).
    pub fn grid_sig(&self) -> String {
        let bits = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{:x}", v.to_bits()));
        let vals: Vec<String> = self.values.iter().map(|(k, v)| format!("{k}={:x}", v.to_bits())).collect();
        format!("{}/{}|{}|{}|{}|{}|{}", self.family, self.preset, vals.join(","), bits(self.thick), bits(self.melt), bits(self.len), self.model)
    }
}

/// The tree: its shape (the parts, in order: what the form is) and the object that places it (where the
/// form stands in the scene). The two are kept apart, so that one day several objects can place one
/// shape; today there is one object. `sel`: which part the panel has selected, or (`whole`) the object
/// itself (neither is kept).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Csg {
    pub parts: Vec<Part>,
    /// where the whole shape stands: every part, and the shape up when it is the first part, move with it
    pub place: Placement,
    pub sel: Option<usize>,
    /// the handles move the whole shape, not the selected part
    pub whole: bool,
    /// the parts are traced from the volume they are measured into, not from their formulas: several
    /// times faster with many parts, as fine as the volume's voxels
    pub coarse: bool,
    /// the tree belongs to a volume object made from a csg object ("convert to volume"): it is measured into
    /// a volume and traced from that, its first part the object's shape
    pub baked: bool,
}

/// The parts as text for a command line: `family/preset@x,y,z@rx,ry,rz@size@op@smooth@level@res` joined by
/// `;` (everything after the shape may be left off). For checks and benchmarks.
pub fn parse_arg(text: &str) -> Option<Vec<Part>> {
    let mut out = Vec::new();
    for item in text.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let mut f = item.split('@');
        let (family, preset) = f.next()?.split_once('/')?;
        let mut part = Part { family: family.into(), preset: preset.into(), ..Default::default() };
        let three = |s: &str| -> Option<[f64; 3]> {
            let v: Vec<f64> = s.split(',').map(|x| x.trim().parse().ok()).collect::<Option<_>>()?;
            (v.len() == 3).then(|| [v[0], v[1], v[2]])
        };
        if let Some(s) = f.next() { part.place.pos = three(s)?; }
        if let Some(s) = f.next() { part.place.rot = three(s)?; }
        if let Some(s) = f.next() { part.place.size = s.parse().ok()?; }
        if let Some(s) = f.next() { part.op = Op::from_id(s)?; }
        if let Some(s) = f.next() { part.smooth = s.parse().ok()?; }
        if let Some(s) = f.next() { part.level = s.parse().ok()?; }
        // (a grid part's voxels a side)
        if let Some(s) = f.next() { part.res = s.parse().ok()?; }
        out.push(part);
    }
    Some(out)
}

/// The distinct shapes a list of parts uses, in order of first use: each is a kind of part the tracer
/// is built with, and what names the tracer.
pub fn kinds(parts: &[Resolved]) -> Vec<(usize, usize)> {
    let mut k: Vec<(usize, usize)> = Vec::new();
    for p in parts {
        if !k.contains(&p.kind()) { k.push(p.kind()); }
    }
    k
}

/// The name a tracer for these kinds is kept under.
pub fn key(fams: &[Family], kinds: &[(usize, usize)]) -> String {
    let names: Vec<String> = kinds.iter().map(|&(f, p)| if (f, p) == GRID_KIND { "grid".to_string() } else { format!("{}/{}", fams[f].id, fams[f].presets[p].id) }).collect();
    format!("csg:{}", names.join(","))
}

/// One kind's text: the preset's own, its names made its own (`k<i>_`), its sliders read from a private
/// copy that is filled from the part being evaluated (`k<i>_use`), its distance (`k<i>_dist`: as the
/// tracer reads it) and its walk to the surface (`k<i>_walk`: as the bake measures it).
fn kind_text(i: usize, fam: &Family, p: &Preset, walked: bool) -> String {
    let Shape::Field { wgsl, clip, shell } = &p.shape else { unreachable!("only formulas are parts") };
    let exact = matches!(support(fam, p), Support::Formula { exact: true });
    let pre = format!("k{i}_");
    let mut members: Vec<String> = p.params.iter().take(MAX_PARAMS).map(|d| member_name(d.id)).collect();
    for j in members.len()..MAX_PARAMS { members.push(format!("_p{j}")); }
    let adapt = |t: &str| {
        let t = rename(t, "pr.", "PRV.");
        let t = rename(&t, "u.hb", "csgU.info.y");
        rename(&t, "u.res", "csgU.info.z")
    };
    let own = adapt(wgsl);
    let names = top_names(&own);
    // the private values the preset keeps between evaluations (worked out once, as for a voxel): they are
    // put back to their first values whenever another part of this kind is evaluated
    let mut resets = String::new();
    for line in own.lines() {
        let Some(rest) = line.strip_prefix("var<private>") else { continue };
        let Some((decl, init)) = rest.split_once('=') else { continue };
        let name: String = decl.trim().chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
        resets += &format!("  {name} = {};\n", init.trim().trim_end_matches(';'));
    }
    let fill: Vec<String> = (0..MAX_PARAMS).map(|j| format!("csgU.parts[i].v{}.{}", j / 4, ['x', 'y', 'z', 'w'][j % 4])).collect();
    let estimate = if exact {
        "  var sd = raw(p);\n".to_string()
    } else {
        format!("  var f = 0.0;
  var g = vec3f(0.0);
  for (var t = 0u; t < 4u + u32(csgU.info2.y); t++) {{
    let o = csgTap(t);
    let v = raw(p + {SLOPE_E} * o);
    f += v;
    g += o * v;
  }}
  g /= {e4};
  var sd = clamp(0.25 * f * inverseSqrt(max(dot(g, g), 1e-12)), -{STEP_MAX}, {STEP_MAX});
", e4 = 4.0 * SLOPE_E)
    };
    let walk = if exact {
        "  var sd = raw(p);\n".to_string()
    } else {
        "  var x = p;
  var v0 = 0.0;
  for (var n = 0; n < 8; n++) {
    let a = raw(x + 0.0007*kk.xyy);
    let b = raw(x + 0.0007*kk.yyx);
    let c = raw(x + 0.0007*kk.yxy);
    let d = raw(x + 0.0007*kk.xxx);
    let fv = 0.25 * (a + b + c + d);
    let g = (kk.xyy*a + kk.yyx*b + kk.yxy*c + kk.xxx*d) / 0.0028;
    if (n == 0) { v0 = fv; }
    var st = g * (fv / max(dot(g, g), 1e-12));
    let l = length(st);
    if (l > 0.12) { st *= 0.12 / l; }
    x -= st;
    if (l < 2e-5) { break; }
  }
  var dist = length(x - p);
  dist *= mix(1.0, 0.7, smoothstep(0.06, 0.2, dist));
  var sd = select(dist, -dist, v0 < 0.0);
".to_string()
    };
    let tail = "  sd -= puff;
  let w = shell(p);
  if (w > 0.0) { sd = abs(sd) - w; }
  return clamp(smax(sd, clip(p), 0.03), -1.0, 1.0);
}
";
    // (the sliders' block is declared apart, after the names are made the kind's own: a slider may share
    // its name with a function of the preset's, as the spiral band's `pitch` does)
    let pr_struct = format!("struct {pre}Pr {{ {} }}\n", members.iter().map(|m| format!("{m}: f32")).collect::<Vec<_>>().join(", "));
    let mut t = format!("\n// ---- kind {i}: {name} ({fam})\nvar<private> PRV: Pr;\nvar<private> cur: i32 = -1;\n{own}\n", name = p.name, fam = fam.id);
    t += &format!("fn shell(p: vec3f) -> f32 {{ return {}; }}\nfn clip(p: vec3f) -> f32 {{ return {}; }}\n", adapt(shell), adapt(clip));
    t += &format!("// the sliders of part i\nfn use_(i: i32) {{\n  if (cur == i) {{ return; }}\n  cur = i;\n  PRV = Pr({});\n{resets}}}\n", fill.join(", "));
    // (as the tracer reads it, or as the bake measures it)
    t += &format!("fn dist(p: vec3f, puff: f32) -> f32 {{\n  const kk = vec2f(1.0, -1.0);\n{}{tail}", if walked { &walk } else { &estimate });
    let mut all = names;
    for n in ["Pr", "PRV", "cur", "shell", "clip", "use_", "dist"] { all.push(n.to_string()); }
    // the helpers every preset may use are shared (`csg_smin`...)
    let t = prefix_names(&t, &all, &pre);
    let helpers = top_names(HELPERS);
    pr_struct + &prefix_names(&t, &helpers, "csg_")
}

/// Where the grid parts' volumes are bound beside the numbers' block at `binding` (the tracer's, or a
/// pass's).
pub fn grid_binding(binding: u32) -> u32 {
    if binding == crate::sdf::BINDING { GRID_BINDING_TRACER } else { GRID_BINDING_PASS }
}

/// The reading of the grid parts: each part's volume (one of eight: `k.w` says which, `e.w` its voxels a
/// side) spans the bake's box, [-HB, HB] in the part's own frame. Read (the tracer) by the same smooth
/// cubic reading the volume tracer's `mapN` uses, eight trilinear lookups (in a loop whose length the
/// compiler cannot see, so that it goes into the shader once); walked (the passes that measure the tree)
/// by one trilinear lookup. Beyond the box, the way to the box (the part lies within it), or the reading
/// at the box less that way, whichever is more: never more than the distance.
fn grid_text(binding: u32, walked: bool) -> String {
    let b = grid_binding(binding);
    let mut t = String::from("\n// ---- grid parts: shapes with no formula, each baked into a volume of its own\n");
    for s in 0..MAX_GRIDS { t += &format!("@group(0) @binding({}) var csgG{s}: texture_3d<f32>;\n", b + s as u32); }
    t += &format!("@group(0) @binding({}) var csgGSmp: sampler;\n", b + MAX_GRIDS as u32);
    let cases: String = (0..MAX_GRIDS).map(|s| format!("    case {s}: {{ return textureSampleLevel(csgG{s}, csgGSmp, t, 0.0).r; }}\n")).collect();
    t += &format!("fn csgGridVol(s: i32, t: vec3f) -> f32 {{
  switch (s) {{
{cases}    default: {{ return 1.0; }}
  }}
}}
");
    // (walked, or not `smooth`: one trilinear lookup)
    let read = if walked {
        "  let v = csgGridVol(s, c / (2.0 * hb) + 0.5);\n".to_string()
    } else {
        "  var v = 1.0;
  if (!fine) {
    v = csgGridVol(s, c / (2.0 * hb) + 0.5);
  } else {
  let res = csgU.parts[i].e.w;
  let cc = (c / (2.0 * hb) + 0.5) * res - 0.5;
  let fi = floor(cc);
  let f = cc - fi;
  let f2 = f * f;
  let f3 = f2 * f;
  let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
  let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
  let w3 = f3 / 6.0;
  let w2 = 1.0 - w0 - w1 - w3;
  let g0 = w0 + w1;
  let g1 = w2 + w3;
  let a = (fi - 0.5 + w1 / g0) / res;
  let bb = (fi + 1.5 + w3 / g1) / res;
  // (which volume is chosen once, and its eight lookups read it straight)
  switch (s) {
".to_string() + &(0..MAX_GRIDS).map(|s| format!("    case {s}: {{ v = csgCubic{s}(a, bb, g0, g1); }}\n")).collect::<String>() + "    default: {}
  }
  }
"
    };
    if !walked {
        // the smooth reading of each volume: eight trilinear lookups placed between its voxels, in a loop the
        // compiler cannot see the length of
        for s in 0..MAX_GRIDS {
            t += &format!("fn csgCubic{s}(a: vec3f, b: vec3f, g0: vec3f, g1: vec3f) -> f32 {{
  var v = 0.0;
  for (var n = 0u; n < 8u + u32(csgU.info2.y); n++) {{
    let hi = vec3<bool>((n & 1u) != 0u, (n & 2u) != 0u, (n & 4u) != 0u);
    let w = select(g0, g1, hi);
    v += w.x * w.y * w.z * textureSampleLevel(csgG{s}, csgGSmp, select(a, b, hi), 0.0).r;
  }}
  return v;
}}
");
        }
    }
    t += &format!("// part i's volume at q (its own frame), in its own units: by the smooth reading (`fine`), or by one
// trilinear lookup
fn csgGridAt(i: i32, q: vec3f, fine: bool) -> f32 {{
  let s = i32(csgU.parts[i].k.w);
  let hb = csgU.info.y;
  let c = clamp(q, vec3f(-hb), vec3f(hb));
  let out = length(q - c);
{read}  return select(v, max(out, v - out), out > 0.0);
}}
");
    t
}

/// The text a tracer (and the bake's pass) holds for a tree of these kinds: the numbers' block at
/// `binding`, every kind's text, `csgPartAt(i, q)`, `csgDist(p)` and `csgNormal(p)`: as the tracer reads
/// them, or (`walked`) as the bake measures the tree, each part walked to its surface.
pub fn wgsl(fams: &[Family], kinds: &[(usize, usize)], binding: u32, walked: bool) -> String {
    wgsl_tree(fams, kinds, binding, walked, false)
}

/// Is every part a plain union, at the top level (`plain`)? Such a tree is read by a leaner text: no groups
/// to keep, no other operation to join by (fewer registers the whole tracer must hold), its cells' lists
/// nearest first.
pub fn plain(parts: &[Resolved]) -> bool {
    parts.iter().take(MAX_PARTS).enumerate().all(|(i, p)| i == 0 || (p.op == Op::Union && p.level == 0))
}

/// The name a tracer for these kinds is kept under, read as a tree of plain unions or not.
pub fn key_tree(fams: &[Family], kinds: &[(usize, usize)], plain: bool) -> String {
    let k = key(fams, kinds);
    if plain { k + "|plain" } else { k }
}

/// The same text, for a tree of plain unions when `plain` (see `plain`), which reads only such a tree.
pub fn wgsl_tree(fams: &[Family], kinds: &[(usize, usize)], binding: u32, walked: bool, plain: bool) -> String {
    let mut t = format!("
// ---- constructive solid geometry on formulas (gf_core::csg)
// a part: the way back from the scene to its own frame (three rows), the box that holds it (b: its middle,
// e: its half sizes, in its own frame), m = (size, inflate, operation, seam radius), k = (kind, groups closed before it, 1 if it opens one, -), its sliders
struct CsgPart {{ inv0: vec4f, inv1: vec4f, inv2: vec4f, b: vec4f, m: vec4f, k: vec4f, v0: vec4f, v1: vec4f, e: vec4f }}
// info: (how many parts, half-width of the volume, the formula's voxels a side, 1 = joined to a volume object)
// info2: (the pad, always 0 (see csgTap), the widest seam, 1 = every part a plain union: the lists are nearest first)
struct CsgU {{ info: vec4f, info2: vec4f, parts: array<CsgPart, {MAX_PARTS}> }}
@group(0) @binding({binding}) var<uniform> csgU: CsgU;
// The cells' lists (`lists`): for each cell four words (where its list starts, how long it is, how near the
// parts it leaves out come, -), then the lists: a part's number in the low half of each word, in the high
// half how near to the cell it may come (in 4096ths, never more than it does).
@group(0) @binding({lists}) var<storage, read> csgL: array<u32>;
// The parts again, read from here (a storage block: pixels side by side often read different parts, which a
// uniform block serves one after another)
@group(0) @binding({table}) var<storage, read> csgP: array<CsgPart>;
const SDF_NE: f32 = {NORMAL_E};
const CSG_CELLS: f32 = {CELLS}.0;
// the corners of a tetrahedron, the four directions a slope is read in (info2.y is always 0: the loops that
// read them have a length the compiler cannot see, and the formula is put into the shader once)
fn csgTap(i: u32) -> vec3f {{
  return vec3f(select(-1.0, 1.0, i == 0u || i == 3u), select(-1.0, 1.0, i >= 2u), select(-1.0, 1.0, i == 1u || i == 3u));
}}
{helpers}
", lists = grid_binding(binding) + MAX_GRIDS as u32 + 1, table = grid_binding(binding) + MAX_GRIDS as u32 + 2, helpers = prefix_names(HELPERS.trim_end(), &top_names(HELPERS), "csg_"));
    for (i, &(f, p)) in kinds.iter().enumerate() {
        if (f, p) == GRID_KIND { t += &grid_text(binding, walked); } else { t += &kind_text(i, &fams[f], &fams[f].presets[p], walked); }
    }
    // one evaluation of part i at q (its own frame), as read or as walked
    let case: String = kinds.iter().enumerate().map(|(i, &k)| if k == GRID_KIND {
        format!("    case {i}: {{ d = clamp(csgGridAt(i, q, fine) - P.m.y, -1.0, 1.0); }}\n")
    } else {
        format!("    case {i}: {{ k{i}_use_(i); d = k{i}_dist(q, P.m.y); }}\n")
    }).collect();
    t += &format!("
fn csgLocal(i: i32, p: vec3f) -> vec3f {{
  let h = vec4f(p, 1.0);
  return vec3f(dot(csgU.parts[i].inv0, h), dot(csgU.parts[i].inv1, h), dot(csgU.parts[i].inv2, h));
}}
// part i's distance at q, a point of its own frame, in its own units
fn csgPartAt(i: i32, q: vec3f, fine: bool) -> f32 {{
  let P = csgU.parts[i];
  var d = 1.0;
  switch (i32(P.k.x)) {{
{case}    default: {{}}
  }}
  return d;
}}
// how far q (its own frame) is from part i's box, in its own units: never more than from the part
fn csgBox(i: i32, q: vec3f) -> f32 {{
  let dq = abs(q - csgU.parts[i].b.xyz) - csgU.parts[i].e.xyz;
  return length(max(dq, vec3f(0.0))) + min(max(dq.x, max(dq.y, dq.z)), 0.0);
}}
// d joined by part distance di: 0 union, 1 subtract, 2 intersect, the seam rounded over k (0: hard)
// With which part decides the result there (`wd`, `wi`: the part's number + 1, negative when the surface
// is that part's turned inside out by a subtraction; 0 where a seam blends two or more).
fn csgJoin(d: f32, wd: f32, di: f32, wi: f32, op: i32, k: f32) -> vec2f {{
  let e = select(di, -di, op == 1);
  let we = select(wi, -wi, op == 1);
  let hard = op != 0;
  let h = max(k - abs(d - e), 0.0) / max(k, 1e-9);
  let bl = h * h * k * 0.25;
  let r = select(min(d, e) - bl, max(d, e) + bl, hard);
  let below = e < d;
  let above = e > d;
  var w = select(wd, we, select(below, above, hard));
  if (h > 0.0) {{ w = 0.0; }}
  return vec2f(r, w);
}}
// the tree's distance at p. Groups: a part a level deeper than the one before opens a group (the result so
// far is put aside, with the group's operation and seam: three named places, no list); a shallower part
// closes groups, each joined to what was put aside. k.y: how many groups close before part i; k.z: 1 if
// it opens one. Only the parts listed in p's cell are visited (a part is left out of a cell only when it
// is too far to matter there, and never when it opens or closes a group or intersects); of those, one whose
// box is too far to change the result is not evaluated (the box's distance stands in, which the join
// then leaves without effect: never a box's distance where a seam could blend it in). The result is held
// within how near the parts left out of the cell come. A tree of plain unions lists each cell's parts
// nearest first, each with how near to the cell it may come: once that is further than the result so far
// and the widest seam, so is every part after it, and the rest are passed by.
// Every join is the one call in the inner loop or the one at the end of a part, so that the text stays
// small wherever it is put in.
// `only` >= 0: that part alone, as if it were the whole tree (its normal is read so: one text serves both).
fn csgDistW(p: vec3f, only: i32) -> vec2f {{
  let solo = only >= 0;
  let cell = vec3u(clamp(floor((p / (2.0 * csgU.info.y) + 0.5) * CSG_CELLS), vec3f(0.0), vec3f(CSG_CELLS - 1.0)));
  let ci = 4u * (cell.x + {cells_i}u * (cell.y + {cells_i}u * cell.z));
  let start = csgL[ci];
  let count = select(csgL[ci + 1u], 1u, solo);
  let far = select(bitcast<f32>(csgL[ci + 2u]), 1.0, solo);
  let sorted = csgU.info2.w > 0.5 && !solo;
  var d = vec2f(4.0, 0.0);
  var lv = 0;
  var s0 = vec4f(0.0);
  var s1 = vec4f(0.0);
  var s2 = vec4f(0.0);
  for (var n = 0u; n <= count; n++) {{
    // the next part listed (or, past the last, none: every group still open is closed)
    var i = -1;
    var closes = lv;
    if (n < count) {{
      let en = select(csgL[start + n], u32(only), solo);
      if (sorted && f32(en >> 16u) * (1.0 / 4096.0) > d.x + csgU.info2.z) {{ break; }}
      i = i32(en & 0xffffu);
      closes = select(i32(csgU.parts[i].k.y), 0, solo);
    }}
    for (var c = 0; c < closes; c++) {{
      d = csgJoin(s0.x, s0.w, d.x, d.y, i32(s0.y), s0.z);
      s0 = s1;
      s1 = s2;
    }}
    lv -= closes;
    if (i < 0) {{ break; }}
    let m = csgU.parts[i].m;
    var first = solo;
    if (!solo && csgU.parts[i].k.z > 0.5) {{ s2 = s1; s1 = s0; s0 = vec4f(d.x, m.z, m.w, d.y); lv += 1; first = true; }}
    let op = i32(m.z);
    let q = csgLocal(i, p);
    let b = csgBox(i, q) * m.x;
    var di = b;
    if (first || op == 2 || (op == 0 && b <= d.x + m.w) || (op == 1 && b <= m.w - d.x)) {{ di = csgPartAt(i, q, only != -1) * m.x; }}
    let mine = vec2f(di, f32(i + 1));
    d = select(csgJoin(d.x, d.y, di, f32(i + 1), op, m.w), mine, first);
  }}
  return vec2f(clamp(d.x, -far, far), d.y);
}}
fn csgDist(p: vec3f) -> f32 {{ return csgDistW(p, -1).x; }}
// The outward normal of the tree at p, where the tree reads r = csgDistW(p, -1). Where one part decides the
// surface (no seam blends there), it is that part's own: that part alone read at the corners of a
// tetrahedron in its own frame (each corner taken back to the scene by its turn and size), turned into the
// scene's (four readings of one part instead of four of the whole tree). Elsewhere, the tree's. The first
// reading (the tree at p: which part decides there) and the four go through the one reading in the loop, so
// that the tree's text is put in here once. Beside a volume object: vec3f(0) where the tree reads `near` or
// more at p (the object's surface is the nearer there).
fn csgNormal(p: vec3f, near: f32) -> vec3f {{
  var i = -1;
  var w = 0.0;
  var dpm = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
  var g = vec3f(0.0);
  for (var t = 0u; t < 5u + u32(csgU.info2.y); t++) {{
    let o = select(csgTap(t - 1u), vec3f(0.0), t == 0u);
    let v = csgDistW(p + dpm * o, select(i, -2, t > 0u && i < 0));
    if (t > 0u) {{
      g += o * v.x;
      continue;
    }}
    if (v.x >= near) {{ return vec3f(0.0); }}
    w = v.y;
    i = i32(abs(w)) - 1;
{grid_winner}
    dpm = mat3x3f(SDF_NE, 0.0, 0.0, 0.0, SDF_NE, 0.0, 0.0, 0.0, SDF_NE);
    if (i >= 0) {{
{ne}      // (the rows of the way back are the turn's columns over the size: their length is 1 / size)
      let P = csgU.parts[i];
      let s2 = ne / dot(P.inv0.xyz, P.inv0.xyz);
      dpm = mat3x3f(P.inv0.xyz * s2, P.inv1.xyz * s2, P.inv2.xyz * s2);
    }}
  }}
  // (no slope at all, where the reading is flat: none here; the caller reads the volume's)
  if (dot(g, g) < 1e-30) {{ return vec3f(0.0); }}
  if (i >= 0) {{
    // (the part's frame to the scene's: the transpose of the way back, whose rows are inv0..2)
    let P = csgU.parts[i];
    return sign(w) * normalize(g.x * P.inv0.xyz + g.y * P.inv1.xyz + g.z * P.inv2.xyz);
  }}
  return normalize(g);
}}
", cells_i = CELLS, grid_winner = if kinds.contains(&GRID_KIND) {
        // (a grid part alone deciding the surface: the volume the tree is measured into holds its distances
        // there, read from its grid; its normal is that volume's, as the volume tracer reads one)
        "    if (i >= 0 && csgU.parts[i].e.w > 0.0) {{ return vec3f(0.0); }}\n".replace("{{", "{").replace("}}", "}")
    } else {
        String::new()
    }, ne = if kinds.contains(&GRID_KIND) {
        // (a grid part's slope is read over one of its own voxels, as the volume tracer reads a volume's)
        "      let ne = select(SDF_NE, 2.0 * csgU.info.y / max(csgU.parts[i].e.w, 1.0), csgU.parts[i].e.w > 0.0);\n"
    } else {
        "      let ne = SDF_NE;\n"
    });
    if plain {
        // A tree of plain unions: its cells' lists are nearest first, and nothing is put aside for groups.
        let (a, b) = (t.find("fn csgDistW(").expect("written above"), t.find("fn csgDist(p: vec3f)").expect("written above"));
        t.replace_range(a..b, &format!("fn csgDistW(p: vec3f, only: i32) -> vec2f {{
  let solo = only >= 0;
  let cell = vec3u(clamp(floor((p / (2.0 * csgU.info.y) + 0.5) * CSG_CELLS), vec3f(0.0), vec3f(CSG_CELLS - 1.0)));
  let ci = 4u * (cell.x + {c}u * (cell.y + {c}u * cell.z));
  let start = csgL[ci];
  let count = select(csgL[ci + 1u], 1u, solo);
  let far = select(bitcast<f32>(csgL[ci + 2u]), 1.0, solo);
  var d = vec2f(4.0, 0.0);
  for (var n = 0u; n < count; n++) {{
    let en = select(csgL[start + n], u32(only), solo);
    // (nearest first: once one may come no nearer than the result so far and the widest seam, nor may the rest)
    if (!solo && f32(en >> 16u) * (1.0 / 4096.0) > d.x + csgU.info2.z) {{ break; }}
    let i = i32(en & 0xffffu);
    let m = csgU.parts[i].m;
    let q = csgLocal(i, p);
    if (solo || csgBox(i, q) * m.x <= d.x + m.w) {{
      let di = csgPartAt(i, q, only != -1) * m.x;
      d = select(csgJoin(d.x, d.y, di, f32(i + 1), 0, m.w), vec2f(di, f32(i + 1)), solo);
    }}
  }}
  return vec2f(clamp(d.x, -far, far), d.y);
}}
", c = CELLS));
    }
    // (every reading of a part is from the storage table)
    t.replace("csgU.parts[", "csgP[")
}

/// The numbers of the tree (the block `wgsl` declares). `res`: the volume's voxels a side; `joined`: the
/// parts are joined to a volume object (the shape up is not one of them).
/// `extents`: each part's measured box (missing ones: the cube it may reach to). `object`: where the whole
/// shape stands (every part's placement is taken within it).
pub fn uniforms(fams: &[Family], parts: &[Resolved], kinds: &[(usize, usize)], res: u32, joined: bool, extents: &[Extent], object: &Placement) -> Vec<f32> {
    numbers_with(fams, parts, kinds, res, joined, extents, object, false).0
}

/// What the cells' lists are made from (where the parts stand, their boxes, how they join): the same, the
/// same lists (so that they need not be worked out again while only a slider moves).
pub fn lists_key(parts: &[Resolved], extents: &[Extent], object: &Placement) -> Vec<u64> {
    let mut k: Vec<u64> = Vec::with_capacity(parts.len() * 20 + 8);
    let mut put = |v: f64| k.push(v.to_bits());
    for v in object.pos.iter().chain(&object.rot).chain([&object.size]) { put(*v); }
    for (i, p) in parts.iter().take(MAX_PARTS).enumerate() {
        for v in p.place.pos.iter().chain(&p.place.rot).chain([&p.place.size, &p.puff, &p.smooth]) { put(*v); }
        put(p.op.code() as f64 + 4.0 * p.level as f64 + if p.smooth_rel { 64.0 } else { 0.0 });
        for v in extents.get(i).copied().unwrap_or([0.0; 6]) { put(v as f64); }
    }
    k
}

/// The tree's numbers and its cells' lists (see `numbers_with`).
pub fn numbers(fams: &[Family], parts: &[Resolved], kinds: &[(usize, usize)], res: u32, joined: bool, extents: &[Extent], object: &Placement) -> (Vec<f32>, Vec<u32>) {
    numbers_with(fams, parts, kinds, res, joined, extents, object, true)
}

/// The tree's numbers (`uniforms`) and its cells' lists (the storage block `wgsl` declares as `csgL`).
///
/// Each cell lists the parts that can matter in it: those that may come within the pad of it (judged twice,
/// by the part's box turned into the scene, and by the cell taken into the part's own frame, where a long
/// part turned askew is held tightly), and every part that opens or closes a group or intersects. With each,
/// how near to the cell it may come (the larger of the two judgements: never more than it does). A tree of
/// plain unions lists them nearest first (the tracer then stops at the first too far to matter); any other in
/// order, as the tree joins them. And for the cell, how near the parts it leaves out come, less the widest
/// seam: a step is told no more than that.
#[allow(clippy::too_many_arguments)]
pub fn numbers_with(fams: &[Family], parts: &[Resolved], kinds: &[(usize, usize)], res: u32, joined: bool, extents: &[Extent], object: &Placement, want_lists: bool) -> (Vec<f32>, Vec<u32>) {
    let mut u = vec![0.0f32; UNIFORM_FLOATS];
    let n = parts.len().min(MAX_PARTS);
    u[0] = n as f32;
    u[1] = crate::HB as f32;
    u[2] = res as f32 * FINER;
    u[3] = if joined { 1.0 } else { 0.0 };
    // each part as it stands in the scene: its own placement within the object's
    let placed: Vec<Placed> = parts.iter().take(n).map(|p| Placed::of(object, &p.place)).collect();
    let seam = |i: usize| -> f64 { let p = &parts[i]; if p.smooth_rel { p.smooth * placed[i].size } else { p.smooth } };
    let most_k = (0..n).map(seam).fold(0.0f64, f64::max);
    let pad = CELL_PAD + most_k;
    u[4] = pad as f32;
    u[6] = most_k as f32;
    let plain = parts.iter().take(n).enumerate().all(|(i, p)| i == 0 || (p.op == Op::Union && p.level == 0));
    u[7] = if plain { 1.0 } else { 0.0 };
    let cell = 2.0 * crate::HB / CELLS as f64;
    let cells = CELLS * CELLS * CELLS;
    // each part's box: in its own frame, and turned into the scene (the box of its turned box)
    struct PartBox { e: [f64; 6], lo: [f64; 3], hi: [f64; 3], structural: bool }
    let mut boxes: Vec<PartBox> = Vec::with_capacity(n);
    let mut prev = 0u8;
    for (i, p) in parts.iter().take(n).enumerate() {
        let r = PART_REACH + p.puff.abs();
        let e = extents.get(i).copied().filter(|e| e[0] <= e[3]).map_or([-r, -r, -r, r, r, r], |e| e.map(|x| x as f64));
        let level = if i == 0 { 0 } else { p.level.min(prev + 1) };
        let structural = level != prev || (i > 0 && p.op == Op::Intersect);
        prev = level;
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for c in 0..8 {
            let corner = [e[if c & 1 == 0 { 0 } else { 3 }], e[if c & 2 == 0 { 1 } else { 4 }], e[if c & 4 == 0 { 2 } else { 5 }]];
            let w = placed[i].apply(corner);
            for k in 0..3 {
                lo[k] = lo[k].min(w[k]);
                hi[k] = hi[k].max(w[k]);
            }
        }
        boxes.push(PartBox { e, lo, hi, structural });
    }
    let gap3 = |alo: &[f64; 3], ahi: &[f64; 3], blo: &[f64], bhi: &[f64]| -> f64 {
        let mut s = 0.0;
        for k in 0..3 {
            let g = (blo[k] - ahi[k]).max(alo[k] - bhi[k]).max(0.0);
            s += g * g;
        }
        s.sqrt()
    };
    let mut heads = vec![0u32; 4 * cells];
    let mut entries: Vec<u32> = Vec::with_capacity(cells * 4);
    let half = 0.5 * cell;
    let mut list: Vec<(f64, usize)> = Vec::with_capacity(n);
    for ci in 0..if want_lists { cells } else { 0 } {
        let (x, y, z) = (ci % CELLS, (ci / CELLS) % CELLS, ci / (CELLS * CELLS));
        let clo = [x as f64 * cell - crate::HB, y as f64 * cell - crate::HB, z as f64 * cell - crate::HB];
        let chi = [clo[0] + cell, clo[1] + cell, clo[2] + cell];
        let mid = [clo[0] + half, clo[1] + half, clo[2] + half];
        list.clear();
        let mut far = 1.0f64;
        for (i, b) in boxes.iter().enumerate() {
            let mut near = gap3(&clo, &chi, &b.lo, &b.hi);
            if near <= pad {
                // the cell in the part's own frame: its middle taken there, and the box that holds it turned
                let inv = &placed[i].inverse;
                let mut lm = [0.0; 3];
                let mut lh = [0.0; 3];
                for r in 0..3 {
                    lm[r] = inv[r * 4] * mid[0] + inv[r * 4 + 1] * mid[1] + inv[r * 4 + 2] * mid[2] + inv[r * 4 + 3];
                    lh[r] = half * (inv[r * 4].abs() + inv[r * 4 + 1].abs() + inv[r * 4 + 2].abs());
                }
                let llo = [lm[0] - lh[0], lm[1] - lh[1], lm[2] - lh[2]];
                let lhi = [lm[0] + lh[0], lm[1] + lh[1], lm[2] + lh[2]];
                near = near.max(gap3(&llo, &lhi, &b.e[0..3], &b.e[3..6]) * placed[i].size);
            }
            if b.structural || near <= pad {
                list.push((near, i));
            } else {
                far = far.min(near - most_k);
            }
        }
        if plain { list.sort_by(|a, b| a.0.total_cmp(&b.0)); }
        heads[4 * ci] = (4 * cells + entries.len()) as u32;
        heads[4 * ci + 1] = list.len() as u32;
        heads[4 * ci + 2] = (far.max(CELL_PAD) as f32).to_bits();
        // (how near, in 4096ths, rounded down: never more than it is)
        entries.extend(list.iter().map(|&(near, i)| (((near * 4096.0).floor().clamp(0.0, 65535.0) as u32) << 16) | i as u32));
    }
    heads.extend(entries);
    let lists = heads;
    let mut prev = 0u8;
    for (i, p) in parts.iter().take(n).enumerate() {
        let o = 8 + i * PART_FLOATS;
        for (j, v) in placed[i].inverse.iter().enumerate() { u[o + j] = *v as f32; }
        let size = placed[i].size;
        let r = (PART_REACH + p.puff.abs()) as f32;
        let e = extents.get(i).copied().filter(|e| e[0] <= e[3]).unwrap_or([-r, -r, -r, r, r, r]);
        for c in 0..3 {
            u[o + 12 + c] = 0.5 * (e[c] + e[3 + c]);
            u[o + 32 + c] = 0.5 * (e[3 + c] - e[c]);
        }
        u[o + 16] = size.abs() as f32;
        u[o + 17] = p.puff as f32;
        // (the first part has nothing above it to join: it is what the others join)
        u[o + 18] = if i == 0 { Op::Union.code() } else { p.op.code() };
        u[o + 19] = seam(i) as f32;
        u[o + 20] = kinds.iter().position(|&k| k == p.kind()).unwrap_or(0) as f32;
        // a grid part: which volume, and its voxels a side (0 for a formula: its normal is read finely)
        if let Some(g) = &p.grid {
            u[o + 23] = g.slot as f32;
            u[o + 35] = g.res as f32;
        }
        // groups: how many close before this part, whether it opens one (levels deeper by one at most)
        let level = if i == 0 { 0 } else { p.level.min(prev + 1) };
        u[o + 21] = prev.saturating_sub(level) as f32;
        u[o + 22] = if level > prev { 1.0 } else { 0.0 };
        prev = level;
        let pre = &fams[p.fam].presets[p.preset];
        for (j, v) in p.vals.iter().take(MAX_PARAMS.min(pre.params.len())).enumerate() { u[o + 24 + j] = *v as f32; }
    }
    (u, lists)
}

/// A part as it stands in the scene: within the object, at its own placement.
struct Placed {
    /// the scene to the part's own frame, three rows of four
    inverse: [f64; 12],
    turn: crate::math::Mat3,
    pos: [f64; 3],
    /// the part's size times the object's (uniform scale only: a distance in the part's frame times this
    /// is the scene's)
    size: f64,
}
impl Placed {
    fn of(object: &Placement, part: &Placement) -> Self {
        let so = if object.size != 0.0 { object.size.abs() } else { 1.0 };
        let sp = if part.size != 0.0 { part.size.abs() } else { 1.0 };
        let ro = crate::math::euler(object.rot[0], object.rot[1], object.rot[2]);
        let rp = crate::math::euler(part.rot[0], part.rot[1], part.rot[2]);
        let turn = crate::math::mul(&ro, &rp);
        let pp = crate::math::apply(&ro, part.pos);
        let pos = [object.pos[0] + so * pp[0], object.pos[1] + so * pp[1], object.pos[2] + so * pp[2]];
        let size = so * sp;
        // q = turn^T (p - pos) / size
        let mut inverse = [0.0; 12];
        for r in 0..3 {
            let mut t = 0.0;
            for c in 0..3 {
                inverse[r * 4 + c] = turn[c * 3 + r] / size;
                t += turn[c * 3 + r] * pos[c];
            }
            inverse[r * 4 + 3] = -t / size;
        }
        Self { inverse, turn, pos, size }
    }
    /// a point of the part's own frame, in the scene
    fn apply(&self, q: [f64; 3]) -> [f64; 3] {
        let w = crate::math::apply(&self.turn, q);
        [self.pos[0] + self.size * w[0], self.pos[1] + self.size * w[1], self.pos[2] + self.size * w[2]]
    }
}

/// A row of `n` parts for timing: the shape given, small enough that they sit side by side in the
/// volume (a cube of them, joined by unions with a little seam).
pub fn grid_of(n: usize, family: &str, preset: &str) -> Vec<Part> {
    let side = (n as f64).cbrt().ceil().max(1.0) as usize;
    let size = 1.9 / (side as f64 * PART_REACH);
    let step = if side > 1 { 2.0 / (side as f64 - 1.0) } else { 0.0 };
    (0..n).map(|i| {
        let (x, y, z) = (i % side, (i / side) % side, i / (side * side));
        let c = |v: usize| if side > 1 { -1.0 + v as f64 * step } else { 0.0 };
        Part {
            family: family.into(), preset: preset.into(),
            place: Placement { pos: [c(x) * 0.62, c(y) * 0.62, c(z) * 0.62], rot: [17.0 * i as f64, 29.0 * i as f64, 0.0], size },
            smooth: 0.02, ..Default::default()
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_read_from_text() {
        let p = parse_arg("studio/sphere@0.5,0,0@0,45,0@0.5@subtract@0.05@1; gems/brilliant").unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].place.pos, [0.5, 0.0, 0.0]);
        assert_eq!(p[0].op, Op::Subtract);
        assert_eq!(p[0].level, 1);
        assert_eq!(p[1].preset, "brilliant");
        assert!(parse_arg("nothing").is_none());
    }

    #[test]
    fn every_shape_is_a_part() {
        let fams = crate::presets::families();
        // a formula is read from its formula; anything else (curves, models) from a volume of its own
        let thomas = Part { family: "attractors".into(), preset: "thomas".into(), ..Default::default() }.resolve(&fams).unwrap();
        assert!(thomas.grid.is_some() && thomas.kind() == GRID_KIND);
        assert!(Part::default().resolve(&fams).unwrap().grid.is_none());
        for f in &fams {
            for p in &f.presets {
                assert!(Part { family: f.id.into(), preset: p.id.into(), ..Default::default() }.resolve(&fams).is_some(), "{}/{}", f.id, p.id);
            }
        }
        // a grid part's voxels a side: its own, within the least and most
        assert_eq!(Part { res: 50, ..Default::default() }.grid_res(), GRID_RES_MIN);
        assert_eq!(Part { res: 4000, ..Default::default() }.grid_res(), GRID_RES_MAX);
        assert_eq!(Part::default().grid_res(), GRID_RES);
    }

    #[test]
    fn plain_unions_are_read_nearest_first() {
        let fams = crate::presets::families();
        let parts: Vec<Resolved> = grid_of(20, "studio", "capsule").iter().map(|p| p.resolve(&fams).unwrap()).collect();
        assert!(plain(&parts));
        let kinds = kinds(&parts);
        let (_, lists) = numbers(&fams, &parts, &kinds, 768, false, &[], &Placement::default());
        let cells = CELLS * CELLS * CELLS;
        let mut listed = 0;
        for c in 0..cells {
            let (start, count) = (lists[4 * c] as usize, lists[4 * c + 1] as usize);
            let near: Vec<u32> = lists[start..start + count].iter().map(|e| e >> 16).collect();
            assert!(near.windows(2).all(|w| w[0] <= w[1]), "cell {c} is not nearest first");
            // (and every part left out is further than the pad)
            assert!(f32::from_bits(lists[4 * c + 2]) >= CELL_PAD as f32 - 1e-6);
            listed += count;
        }
        // (a cell lists only the parts near it: far fewer than every part in every cell)
        assert!(listed < cells * parts.len() / 4, "{listed}");
        // anything but unions keeps the tree's order
        let mut mixed = parts.clone();
        mixed[3].op = Op::Subtract;
        assert!(!plain(&mixed));
    }
}

impl crate::state::State {
    /// The tree as it is traced now: its parts with their shapes found (the shape up first, when it is
    /// traced as "csg" and is a formula the surface studio has not reshaped), and whether they are joined
    /// to a volume object (the shape up, when it is not one of them). Empty while there are several layers.
    pub fn csg_tree(&self) -> (Vec<Resolved>, bool) {
        let (out, joined, _) = self.csg_tree_counted();
        (out, joined)
    }

    // ---- objects: volume objects and csg objects ------------------------------------------------------
    /// The tree of the object at this place in the list (the selected one's is `csg`).
    pub fn layer_csg(&self, layer: usize) -> &Csg {
        static NONE: std::sync::OnceLock<Csg> = std::sync::OnceLock::new();
        if layer == self.stack.sel { return &self.csg; }
        self.parked.get(layer).and_then(|p| p.as_ref()).map_or_else(|| NONE.get_or_init(Csg::default), |p| &p.csg)
    }
    /// The same, to change.
    pub fn layer_csg_mut(&mut self, layer: usize) -> Option<&mut Csg> {
        if layer == self.stack.sel { return Some(&mut self.csg); }
        self.parked.get_mut(layer).and_then(|p| p.as_mut()).map(|p| &mut p.csg)
    }
    /// Is the object at this place a csg object (its kind, fixed when it was made)?
    pub fn is_csg_object(&self, layer: usize) -> bool {
        self.layer_shape(layer).is_some_and(|st| st.trace == crate::sdf::Trace::Csg)
    }
    /// Is the object at this place read as a tree (a csg object; a volume object made from one; or, from setups
    /// of before, a volume object with parts beside it)?
    pub fn has_tree(&self, layer: usize) -> bool {
        let c = self.layer_csg(layer);
        self.is_csg_object(layer) || c.baked || !c.parts.is_empty()
    }
    /// Are the selected tree's parts traced from the volume they are measured into (asked for, or a volume
    /// object made from a tree)?
    pub fn csg_coarse(&self) -> bool {
        self.csg.coarse || self.csg.baked
    }
    /// How many objects there are (none: the scene shows the backdrop and the floor alone).
    pub fn object_count(&self) -> usize {
        if self.empty { 0 } else { self.layer_count() }
    }
    /// Make an object of this shape, of this kind (a csg object: a tree whose first part is the shape, exact
    /// if it has a formula and a grid part otherwise; else a volume object), and select it. With no objects it
    /// is the first; otherwise it is added (None when there are as many as there may be).
    pub fn create_object(&mut self, fam: usize, preset: usize, csg: bool) -> Option<usize> {
        if fam >= self.families.len() || preset >= self.families[fam].presets.len() { return None; }
        let kind = if csg { crate::sdf::Trace::Csg } else { crate::sdf::Trace::Volume };
        let i = if self.empty {
            // (the scene's one layer, hidden while there was nothing, becomes the object)
            self.empty = false;
            self.stack.layers[0] = crate::layers::Layer::default();
            self.set_scene(fam);
            self.choose(preset);
            0
        } else {
            self.add_layer_of(fam, preset)?
        };
        self.shape_mut().trace = kind;
        self.csg = Csg::default();
        self.cut = true;
        self.want = Some(crate::state::Detail::Full);
        Some(i)
    }
    /// Take the object at this place away. The last one leaves the scene empty (its layer stays, hidden,
    /// for the next object made to take up).
    pub fn delete_object(&mut self, layer: usize) -> bool {
        if self.empty || layer >= self.layer_count() { return false; }
        if self.layer_count() > 1 { return self.remove_layer(layer); }
        self.empty = true;
        self.stack.layers[0].show = false;
        self.csg = Csg::default();
        self.shape_mut().trace = crate::sdf::Trace::Volume;
        self.cut = true;
        self.want = Some(crate::state::Detail::Full);
        true
    }
    /// The selected csg object made a volume object: its tree is measured into a volume, as "parts traced:
    /// volume" does, and traced from that; it keeps its placement and its material. False if it is not a csg
    /// object.
    pub fn convert_to_volume(&mut self) -> bool {
        if self.empty || !self.is_csg_object(self.stack.sel) { return false; }
        self.shape_mut().trace = crate::sdf::Trace::Volume;
        // (a lone shape, moved nowhere, is simply its own volume; anything more is the tree baked)
        self.csg.baked = !self.csg.parts.is_empty() || !self.csg.place.is_identity();
        self.csg.sel = None;
        self.csg.whole = false;
        self.cut = true;
        self.want = Some(crate::state::Detail::Full);
        true
    }

    /// The same, and how many grid parts were left out of it (past `MAX_GRIDS`, or past `GRID_BUDGET`).
    /// The shape up is a part when it is traced as "csg": from its formula; or, when it has none (curves, a
    /// model) or the surface studio has reshaped it, as a grid part reading its own volume (the studio's
    /// work included). Grid parts are given their volumes in order.
    pub fn csg_tree_counted(&self) -> (Vec<Resolved>, bool, usize) {
        if !self.plain() { return (Vec::new(), false, 0); }
        self.csg_tree_layer(0)
    }

    /// The tree of the object at this place in the list, wherever it stands among others (what
    /// `csg_tree_counted` gives for one object alone). Its shape is its first part when it is a csg object,
    /// or a volume object made from one (`Csg::baked`).
    pub fn csg_tree_layer(&self, layer: usize) -> (Vec<Resolved>, bool, usize) {
        let mut out = Vec::new();
        let csg = self.layer_csg(layer);
        let up = self.layer_shape(layer).filter(|st| st.trace == crate::sdf::Trace::Csg || csg.baked);
        if let Some(st) = up {
            let sculpt = self.layer_sculpt(layer).filter(|s| !s.is_empty());
            let formula = matches!(support(self.layer_family(layer), self.layer_preset(layer)), Support::Formula { .. }) && sculpt.is_none();
            let grid = (!formula).then(|| Grid {
                slot: 0, res: st.res, from: GridFrom::Up,
                sig: format!("up{layer}|{}|{}", self.layer_key(layer), sculpt.map(|s| serde_json::to_string(s).unwrap_or_default()).unwrap_or_default()),
            });
            out.push(Resolved { fam: self.layer_fam(layer), preset: st.preset, vals: st.vals.clone(), puff: if grid.is_some() { 0.0 } else { st.puff }, place: Placement::default(), op: Op::Union, smooth: 0.0, smooth_rel: false, level: 0, grid });
        }
        for (i, p) in csg.parts.iter().enumerate() {
            if let Some(mut r) = p.resolve(&self.families) {
                if let Some(g) = &mut r.grid { g.from = GridFrom::Part(i); }
                out.push(r);
            }
        }
        out.truncate(MAX_PARTS);
        // each grid part its volume, in order, while there are volumes and memory for them
        let (mut slot, mut bytes, mut left) = (0usize, 0u64, 0usize);
        out.retain_mut(|p| {
            let Some(g) = &mut p.grid else { return true };
            let cost = if g.from == GridFrom::Up { 0 } else { grid_bytes(g.res) };
            if slot >= MAX_GRIDS || bytes + cost > GRID_BUDGET {
                left += 1;
                return false;
            }
            g.slot = slot;
            slot += 1;
            bytes += cost;
            true
        });
        let joined = up.is_none();
        (out, joined, left)
    }

    /// The video memory the tree's grid parts take for volumes of their own, and how many there are.
    pub fn csg_grid_bytes(&self) -> (u64, usize) {
        let (tree, _, _) = self.csg_tree_layer(self.stack.sel);
        let own: Vec<&Grid> = tree.iter().filter_map(|p| p.grid.as_ref()).filter(|g| g.from != GridFrom::Up).collect();
        (own.iter().map(|g| grid_bytes(g.res)).sum(), own.len())
    }

    /// What the GPU has to build for a grid part's shape, `res` voxels a side: the job its family would
    /// give for it were it the shape up (its sliders, its tube's, its model file), worked out on a shape
    /// of its own so that the shape up is left as it was.
    pub fn part_job(&mut self, part: &Part, res: u32) -> Result<crate::state::Job, String> {
        let fam = self.families.iter().position(|f| f.id == part.family).ok_or_else(|| format!("no family called \"{}\"", part.family))?;
        let preset = self.families[fam].presets.iter().position(|p| p.id == part.preset).ok_or_else(|| format!("no shape called \"{}\"", part.preset))?;
        let mut st = crate::state::State::fresh_shapes(std::slice::from_ref(&self.families[fam]), &[]).pop().expect("one family, one shape");
        let p = &self.families[fam].presets[preset];
        st.preset = preset;
        st.vals = p.params.iter().map(|d| part.values.get(d.id).copied().unwrap_or(d.value)).collect();
        if let Some(v) = part.thick { st.thick = crate::limits::hold(v, crate::limits::THICK); }
        if let Some(v) = part.melt { st.melt = crate::limits::hold(v, crate::limits::MELT); }
        if let Some(v) = part.len { st.len = crate::limits::hold(v, crate::limits::LEN); }
        st.model = part.model.clone();
        st.res = res;
        st.anim_res = res;
        let was_fam = self.fam;
        std::mem::swap(&mut self.shapes[fam], &mut st);
        self.fam = fam;
        let job = self.shape_job(crate::state::Detail::Full, Some(res));
        self.fam = was_fam;
        std::mem::swap(&mut self.shapes[fam], &mut st);
        job
    }
}

impl Csg {
    /// How the handles see part i: the frame they move it in (the object's turn: a drag along the frame's
    /// axes moves the part along the object's own), its middle in that frame (so that turned by the frame
    /// it lands where the part stands), its own turn and its size.
    pub fn part_handle(&self, i: usize) -> Option<(crate::math::Mat3, [f64; 3], [f64; 3], f64)> {
        let p = self.parts.get(i)?.place;
        let o = self.place;
        let so = if o.size != 0.0 { o.size.abs() } else { 1.0 };
        let ro = crate::math::euler(o.rot[0], o.rot[1], o.rot[2]);
        let back = crate::math::apply(&crate::math::transpose(&ro), o.pos);
        Some((ro, [back[0] + so * p.pos[0], back[1] + so * p.pos[1], back[2] + so * p.pos[2]], p.rot, p.size))
    }
    /// Put part i where the handles say (`pos` in the frame `part_handle` gave).
    pub fn set_part_handle(&mut self, i: usize, pos: [f64; 3], rot: [f64; 3], size: f64) {
        let o = self.place;
        let so = if o.size != 0.0 { o.size.abs() } else { 1.0 };
        let ro = crate::math::euler(o.rot[0], o.rot[1], o.rot[2]);
        let back = crate::math::apply(&crate::math::transpose(&ro), o.pos);
        let Some(p) = self.parts.get_mut(i) else { return };
        for c in 0..3 { p.place.pos[c] = ((pos[c] - back[c]) / so).clamp(-1.4, 1.4); }
        p.place.rot = rot;
        p.place.size = size.clamp(0.05, 2.0);
    }
    /// Where part i's middle stands in the scene, and its size there.
    pub fn part_in_scene(&self, i: usize) -> Option<([f64; 3], f64)> {
        let placed = Placed::of(&self.place, &self.parts.get(i)?.place);
        Some((placed.pos, placed.size))
    }
}

#[cfg(test)]
mod handle_tests {
    use super::*;

    #[test]
    fn a_part_moved_by_the_handles_lands_where_they_say() {
        let mut c = Csg { parts: vec![Part::default()], place: Placement { pos: [0.2, -0.1, 0.3], rot: [10.0, 40.0, -20.0], size: 0.8 }, ..Default::default() };
        c.parts[0].place.pos = [0.3, 0.1, -0.2];
        let (ro, pos, rot, size) = c.part_handle(0).unwrap();
        // turned by the frame, the handle's point is the part's middle in the scene
        let w = crate::math::apply(&ro, pos);
        let (scene, _) = c.part_in_scene(0).unwrap();
        for k in 0..3 { assert!((w[k] - scene[k]).abs() < 1e-9, "{w:?} {scene:?}"); }
        // and writing back what was read changes nothing
        let before = c.parts[0].place;
        c.set_part_handle(0, pos, rot, size);
        for k in 0..3 { assert!((c.parts[0].place.pos[k] - before.pos[k]).abs() < 1e-9); }
    }
}
