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
pub const UNIFORM_FLOATS: usize = 8 + PART_FLOATS * MAX_PARTS + CELLS * CELLS * CELLS * 2;
/// The volume is cut into this many cells a side, and each cell lists the parts (as two words of bits)
/// that can matter in it: a step evaluates those alone.
pub const CELLS: usize = 8;
/// A part is listed in every cell within this of its box (and its seam's radius); the distance a step is
/// told is held within it either way, so that no part left out is ever stepped past.
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
}
impl Default for Part {
    fn default() -> Self {
        Self { family: "studio".into(), preset: "sphere".into(), values: BTreeMap::new(), puff: 0.0, place: Placement::default(), op: Op::Union, smooth: 0.0, smooth_rel: false, level: 0 }
    }
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
    /// Its shape found among the families; None if there is no such shape, or it has no formula.
    pub fn resolve(&self, fams: &[Family]) -> Option<Resolved> {
        let fam = fams.iter().position(|f| f.id == self.family)?;
        let preset = fams[fam].presets.iter().position(|p| p.id == self.preset)?;
        let p = &fams[fam].presets[preset];
        if !matches!(support(&fams[fam], p), Support::Formula { .. }) { return None; }
        let vals = p.params.iter().map(|d| self.values.get(d.id).copied().unwrap_or(d.value)).collect();
        Some(Resolved { fam, preset, vals, puff: self.puff, place: self.place, op: self.op, smooth: self.smooth.max(0.0), smooth_rel: self.smooth_rel, level: self.level.min(MAX_LEVEL) })
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
}

/// The parts as text for a command line: `family/preset@x,y,z@rx,ry,rz@size@op@smooth@level` joined by
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
        out.push(part);
    }
    Some(out)
}

/// The distinct shapes a list of parts uses, in order of first use: each is a kind of part the tracer
/// is built with, and what names the tracer.
pub fn kinds(parts: &[Resolved]) -> Vec<(usize, usize)> {
    let mut k: Vec<(usize, usize)> = Vec::new();
    for p in parts {
        if !k.contains(&(p.fam, p.preset)) { k.push((p.fam, p.preset)); }
    }
    k
}

/// The name a tracer for these kinds is kept under.
pub fn key(fams: &[Family], kinds: &[(usize, usize)]) -> String {
    let names: Vec<String> = kinds.iter().map(|&(f, p)| format!("{}/{}", fams[f].id, fams[f].presets[p].id)).collect();
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

/// The text a tracer (and the bake's pass) holds for a tree of these kinds: the numbers' block at
/// `binding`, every kind's text, `csgPartAt(i, q)`, `csgDist(p)` and `csgNormal(p)`: as the tracer reads
/// them, or (`walked`) as the bake measures the tree, each part walked to its surface.
pub fn wgsl(fams: &[Family], kinds: &[(usize, usize)], binding: u32, walked: bool) -> String {
    let mut t = format!("
// ---- constructive solid geometry on formulas (gf_core::csg)
// a part: the way back from the scene to its own frame (three rows), the box that holds it (b: its middle,
// e: its half sizes, in its own frame), m = (size, inflate, operation, seam radius), k = (kind, groups closed before it, 1 if it opens one, -), its sliders
struct CsgPart {{ inv0: vec4f, inv1: vec4f, inv2: vec4f, b: vec4f, m: vec4f, k: vec4f, v0: vec4f, v1: vec4f, e: vec4f }}
// info: (how many parts, half-width of the volume, the formula's voxels a side, 1 = joined to a volume object)
// info2: (the distance a step is held within, always 0 (see csgTap), -, -); cells: two words of bits a cell, two cells a vec4
struct CsgU {{ info: vec4f, info2: vec4f, parts: array<CsgPart, {MAX_PARTS}>, cells: array<vec4u, {cells}> }}
@group(0) @binding({binding}) var<uniform> csgU: CsgU;
const SDF_NE: f32 = {NORMAL_E};
const CSG_CELLS: f32 = {CELLS}.0;
// the corners of a tetrahedron, the four directions a slope is read in (info2.y is always 0: the loops that
// read them have a length the compiler cannot see, and the formula is put into the shader once)
fn csgTap(i: u32) -> vec3f {{
  return vec3f(select(-1.0, 1.0, i == 0u || i == 3u), select(-1.0, 1.0, i >= 2u), select(-1.0, 1.0, i == 1u || i == 3u));
}}
{helpers}
", cells = CELLS * CELLS * CELLS / 2, helpers = prefix_names(HELPERS.trim_end(), &top_names(HELPERS), "csg_"));
    for (i, &(f, p)) in kinds.iter().enumerate() { t += &kind_text(i, &fams[f], &fams[f].presets[p], walked); }
    // one evaluation of part i at q (its own frame), as read or as walked
    let case: String = kinds.iter().enumerate().map(|(i, _)| format!("    case {i}: {{ k{i}_use_(i); d = k{i}_dist(q, P.m.y); }}\n")).collect();
    t += &format!("
fn csgLocal(i: i32, p: vec3f) -> vec3f {{
  let h = vec4f(p, 1.0);
  return vec3f(dot(csgU.parts[i].inv0, h), dot(csgU.parts[i].inv1, h), dot(csgU.parts[i].inv2, h));
}}
// part i's distance at q, a point of its own frame, in its own units
fn csgPartAt(i: i32, q: vec3f) -> f32 {{
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
// then leaves without effect: never a box's distance where a seam could blend it in). The result is held within the distance beyond which parts were left out.
// Every join is the one call in the inner loop or the one at the end of a part, so that the text stays
// small wherever it is put in.
fn csgDistW(p: vec3f) -> vec2f {{
  let cell = vec3i(clamp(floor((p / (2.0 * csgU.info.y) + 0.5) * CSG_CELLS), vec3f(0.0), vec3f(CSG_CELLS - 1.0)));
  let ci = cell.x + {cells_i} * (cell.y + {cells_i} * cell.z);
  let pair = csgU.cells[ci / 2];
  let mask = select(pair.xy, pair.zw, (ci & 1) == 1);
  var d = vec2f(4.0, 0.0);
  var lv = 0;
  var s0 = vec4f(0.0);
  var s1 = vec4f(0.0);
  var s2 = vec4f(0.0);
  for (var w = 0; w < 3; w++) {{
    var bits = 0u;
    if (w == 0) {{ bits = mask.x; }} else if (w == 1) {{ bits = mask.y; }}
    loop {{
      // the next part listed (or, past the last word, none: every group still open is closed)
      var i = -1;
      var closes = lv;
      if (bits != 0u) {{
        i = i32(firstTrailingBit(bits)) + 32 * w;
        bits &= bits - 1u;
        closes = i32(csgU.parts[i].k.y);
      }} else if (w < 2) {{
        break;
      }}
      for (var c = 0; c < closes; c++) {{
        d = csgJoin(s0.x, s0.w, d.x, d.y, i32(s0.y), s0.z);
        s0 = s1;
        s1 = s2;
      }}
      lv -= closes;
      if (i < 0) {{ break; }}
      let m = csgU.parts[i].m;
      var first = false;
      if (csgU.parts[i].k.z > 0.5) {{ s2 = s1; s1 = s0; s0 = vec4f(d.x, m.z, m.w, d.y); lv += 1; first = true; }}
      let op = i32(m.z);
      let q = csgLocal(i, p);
      let b = csgBox(i, q) * m.x;
      var di = b;
      if (first || op == 2 || (op == 0 && b <= d.x + m.w) || (op == 1 && b <= m.w - d.x)) {{ di = csgPartAt(i, q) * m.x; }}
      let mine = vec2f(di, f32(i + 1));
      d = select(csgJoin(d.x, d.y, di, f32(i + 1), op, m.w), mine, first);
    }}
  }}
  let pad = csgU.info2.x;
  return vec2f(clamp(d.x, -pad, pad), d.y);
}}
fn csgDist(p: vec3f) -> f32 {{ return csgDistW(p).x; }}
// The outward normal of the tree at p. Where one part decides the surface (no seam blends there), it is
// that part's own: its distance read at the corners of a tetrahedron in its own frame, turned into the
// scene's (four readings of one part instead of four of the whole tree). Elsewhere, the tree's.
fn csgNormal(p: vec3f) -> vec3f {{
  let r = csgDistW(p);
  let i = i32(abs(r.y)) - 1;
  var g = vec3f(0.0);
  if (i >= 0) {{
    let q = csgLocal(i, p);
    for (var t = 0u; t < 4u + u32(csgU.info2.y); t++) {{
      let o = csgTap(t);
      g += o * csgPartAt(i, q + SDF_NE * o);
    }}
    // (the part's frame to the scene's: the transpose of the way back, whose rows are inv0..2)
    let P = csgU.parts[i];
    return sign(r.y) * normalize(g.x * P.inv0.xyz + g.y * P.inv1.xyz + g.z * P.inv2.xyz);
  }}
  for (var t = 0u; t < 4u + u32(csgU.info2.y); t++) {{
    let o = csgTap(t);
    g += o * csgDist(p + SDF_NE * o);
  }}
  return normalize(g);
}}
", cells_i = CELLS);
    t
}

/// The numbers of the tree (the block `wgsl` declares). `res`: the volume's voxels a side; `joined`: the
/// parts are joined to a volume object (the shape up is not one of them).
/// `extents`: each part's measured box (missing ones: the cube it may reach to). `object`: where the whole
/// shape stands (every part's placement is taken within it).
pub fn uniforms(fams: &[Family], parts: &[Resolved], kinds: &[(usize, usize)], res: u32, joined: bool, extents: &[Extent], object: &Placement) -> Vec<f32> {
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
    // which parts each cell lists: those whose box (turned and placed, in the scene) comes within the pad of
    // the cell, and every part that opens or closes a group or intersects
    let cell = 2.0 * crate::HB / CELLS as f64;
    let mut masks = vec![0u64; CELLS * CELLS * CELLS];
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
                let x = w[k];
                lo[k] = lo[k].min(x - pad);
                hi[k] = hi[k].max(x + pad);
            }
        }
        let span = |k: usize| -> (usize, usize) {
            let a = ((lo[k] + crate::HB) / cell).floor().clamp(0.0, CELLS as f64 - 1.0) as usize;
            let b = ((hi[k] + crate::HB) / cell).floor().clamp(0.0, CELLS as f64 - 1.0) as usize;
            (a, b)
        };
        let ((x0, x1), (y0, y1), (z0, z1)) = (span(0), span(1), span(2));
        for (ci, m) in masks.iter_mut().enumerate() {
            let (x, y, z) = (ci % CELLS, (ci / CELLS) % CELLS, ci / (CELLS * CELLS));
            if structural || (x0..=x1).contains(&x) && (y0..=y1).contains(&y) && (z0..=z1).contains(&z) { *m |= 1u64 << i; }
        }
    }
    let base = 8 + PART_FLOATS * MAX_PARTS;
    for (ci, m) in masks.iter().enumerate() {
        u[base + 2 * ci] = f32::from_bits(*m as u32);
        u[base + 2 * ci + 1] = f32::from_bits((*m >> 32) as u32);
    }
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
        u[o + 20] = kinds.iter().position(|&k| k == (p.fam, p.preset)).unwrap_or(0) as f32;
        // groups: how many close before this part, whether it opens one (levels deeper by one at most)
        let level = if i == 0 { 0 } else { p.level.min(prev + 1) };
        u[o + 21] = prev.saturating_sub(level) as f32;
        u[o + 22] = if level > prev { 1.0 } else { 0.0 };
        prev = level;
        let pre = &fams[p.fam].presets[p.preset];
        for (j, v) in p.vals.iter().take(MAX_PARAMS.min(pre.params.len())).enumerate() { u[o + 24 + j] = *v as f32; }
    }
    u
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
    fn only_formulas_are_parts() {
        let fams = crate::presets::families();
        assert!(Part { family: "attractors".into(), preset: "thomas".into(), ..Default::default() }.resolve(&fams).is_none());
        assert!(Part::default().resolve(&fams).is_some());
    }
}

impl crate::state::State {
    /// The tree as it is traced now: its parts with their shapes found (the shape up first, when it is
    /// traced as "csg" and is a formula the surface studio has not reshaped), and whether they are joined
    /// to a volume object (the shape up, when it is not one of them). Empty while there are several layers.
    pub fn csg_tree(&self) -> (Vec<Resolved>, bool) {
        if !self.plain() { return (Vec::new(), false); }
        let mut out = Vec::new();
        let up = self.layer_shape(0).filter(|st| {
            st.trace == crate::sdf::Trace::Csg
                && !self.layer_sculpt(0).is_some_and(|s| !s.is_empty())
                && matches!(support(self.layer_family(0), self.layer_preset(0)), Support::Formula { .. })
        });
        if let Some(st) = up {
            out.push(Resolved { fam: self.layer_fam(0), preset: st.preset, vals: st.vals.clone(), puff: st.puff, place: Placement::default(), op: Op::Union, smooth: 0.0, smooth_rel: false, level: 0 });
        }
        out.extend(self.csg.parts.iter().filter_map(|p| p.resolve(&self.families)));
        out.truncate(MAX_PARTS);
        let joined = up.is_none();
        (out, joined)
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
