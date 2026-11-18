//! Tracing a shape's own formula ("sdf") instead of the volume baked from it ("volume").
//!
//! A shape given by an equation (`Shape::Field`) is baked into a grid of distances, and the tracer reads
//! the grid. The grid is what limits how sharp the glass can be: an edge or a thin wall is as fine as a
//! voxel and no finer, however close the camera comes. Traced directly, the tracer still marches through
//! the volume (a sure distance everywhere, one lookup a step: formulas that jump or flatten far from their
//! surface made sheets of false surface in mid-air when the march trusted them), but settles every hit
//! onto the formula's own zero and takes every normal from the formula: the surface is the formula's own,
//! at any zoom, for a few readings of it a hit.
//!
//! The formula is not a distance. Most presets' `raw` is only negative inside and positive outside, so the
//! tracer divides it by the length of its slope (four evaluations about the point, the same four the bake's
//! walk takes): near the surface that is the distance to it; far from it, where the slope misleads, the step
//! is held to `STEP_MAX`. A preset whose `raw` is a true distance (the studio's shapes) or never more than
//! one (a cut stone's facets) is read as it is, one evaluation a step.
//!
//! Shapes made from curves (attractors, knots, trees) and models read from files have no formula to
//! evaluate per step (a point would have to be measured against every piece of every strand or every
//! triangle): they are traced from their volume only. See `docs/sdf_mode.md`.
//!
//! The grid is still baked in either mode: everything besides the tracer (shadows and caustics, the
//! surface pass, the floor's height, the light inside a milky body, the measured thickness) reads it.

use crate::field::{member_name, HELPERS, MAX_PARAMS};
use crate::presets::{Family, Preset, Shape};

/// How a shape is traced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Trace {
    /// from the distance volume baked from it (as always)
    #[default]
    Volume,
    /// from its formula: alone, or as the first part of a tree of formulas (gf_core::csg). Setups and
    /// command lines from before the tree say "sdf"
    #[serde(rename = "csg", alias = "sdf")]
    Csg,
}

impl Trace {
    /// The word a setup and a command line use.
    pub fn name(self) -> &'static str {
        match self { Trace::Volume => "volume", Trace::Csg => "csg" }
    }
    pub fn is_volume(&self) -> bool { *self == Trace::Volume }
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() { "volume" => Some(Trace::Volume), "csg" | "sdf" => Some(Trace::Csg), _ => None }
    }
}

/// Whether a preset can be traced from its formula.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Support {
    /// It can. `exact`: its `raw` is a distance (or never more than one), read as it is.
    Formula { exact: bool },
    /// It cannot, for this reason (lower case, plain words: shown in the panel).
    VolumeOnly(&'static str),
}

/// Families whose `raw` is a true distance, or never more than one: the studio's shapes measure their
/// distance exactly, inside and out; a cut stone's is the largest of its facets' distances, exact
/// inside and short of the truth outside.
const EXACT: &[&str] = &["studio", "gems"];
/// Single presets whose `raw` is a true distance: the kaleidoscope's polyhedra and frames (the distance to
/// the corner their faces make, exactly) and the stellations (the distance to a wedge of face planes).
const EXACT_PRESETS: &[&str] = &[
    "solids/tetrahedron", "solids/cube", "solids/octahedron", "solids/dodecahedron", "solids/icosahedron",
    "solids/cuboctahedron", "solids/football", "solids/frame",
    "solids/stellaoctangula", "solids/stardodecahedron", "solids/staricosahedron",
    // fractals: the Menger sponge (smooth unions and cuts of exact distances, never more than the distance) and
    // the Apollonian packing (every ball near the point found in the packing's tree and measured exactly)
    "fractals/menger", "fractals/apollonian",
    // topology: walls about surfaces whose `raw` is the distance to the nearest point of the surface found by a
    // search (Dini's and the Calabi-Yau slice to first order: read as distances, as their bake reads them)
    "topology/kleinbottle", "topology/enneper", "topology/dini", "topology/kuen", "topology/breather", "topology/calabiyau", "topology/costa",
    // the polyhedra: the three planes a point lies farthest beyond, measured to exactly (exact inside, never
    // more than the distance outside); a compound is the smooth union of such members
    "polyhedra/squarepyramid", "polyhedra/pentpyramid", "polyhedra/tricupola", "polyhedra/pentcupola", "polyhedra/rotunda",
    "polyhedra/gyrobicupola", "polyhedra/gyrobipyramid", "polyhedra/bidiminished", "polyhedra/augdodeca",
    "polyhedra/fivecubes", "polyhedra/fivetetra", "polyhedra/tentetra", "polyhedra/fiveocta", "polyhedra/cubeocta", "polyhedra/dodecaicosa",
    "polyhedra/rhombicdodeca", "polyhedra/triacontahedron", "polyhedra/truncoct", "polyhedra/enneacontahedron",
    "polyhedra/truncicosidodeca", "polyhedra/polarzono",
];

/// Can this family's preset be traced from its formula?
pub fn support(fam: &Family, p: &Preset) -> Support {
    match &p.shape {
        Shape::Field { .. } => Support::Formula { exact: EXACT.contains(&fam.id) || EXACT_PRESETS.contains(&format!("{}/{}", fam.id, p.id).as_str()) },
        Shape::Flow(_) | Shape::Curve(_) => Support::VolumeOnly("drawn from curves: no formula to trace"),
        Shape::Mesh { .. } => Support::VolumeOnly("a model file: no formula to trace"),
    }
}

/// The binding of the formula's numbers in the tracer (read only by tracers built for a formula).
pub const BINDING: u32 = 35;
/// The longest step the slope's estimate is trusted for, in the volume's units (the volume is 3 across).
/// The bake's walk to the surface trusts its gradient no further than 0.12 a step; a ray marching
/// through the glass and out of it takes this, scaled by the family's step scale.
pub const STEP_MAX: f32 = 0.1;
/// Some shapes round every edge to a voxel or two of the volume being built (`fine()` in solids.rs: the
/// bake's walk cannot measure a sharp edge). Traced from the formula they are told of a voxel this many
/// times finer, so that their edges are as sharp as the formula allows, not as the grid does.
pub const FINER: f32 = 4.0;
/// Within how many voxels of the surface the tracer marches by the formula; further out by the volume,
/// which is a sure distance there (app/src/sdf.rs). None: the march is the volume's own throughout, and the
/// formula only settles each hit onto its own surface and gives the normal there. Marching by it within 1.5
/// voxels, as at first, cost up to six times the volume's time (the cube, the seashell) and showed nothing
/// that settling does not (docs/sdf_mode.md).
pub const BAND: f32 = 0.0;
/// How far apart the four evaluations that give the slope are (as in the bake's walk).
pub const SLOPE_E: f32 = 0.0007;
/// How far apart the readings that give a normal are: a tenth of a 768-voxel grid's spacing.
pub const NORMAL_E: f32 = 0.0004;

/// `text` with every `from` that starts a name (not preceded by a letter, digit, `_` or `.`) made `to`.
pub(crate) fn rename(text: &str, from: &str, to: &str) -> String {
    let mut out = String::with_capacity(text.len() + 64);
    let mut rest = text;
    while let Some(i) = rest.find(from) {
        let prev = if i > 0 { rest[..i].chars().last() } else { out.chars().last() };
        let starts = prev.is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_' || c == '.'));
        out.push_str(&rest[..i]);
        out.push_str(if starts { to } else { from });
        rest = &rest[i + from.len()..];
    }
    out.push_str(rest);
    out
}

/// The preset's text as the tracer holds it: its sliders read from the tracer's block for the formula
/// (`sdfU`), and so are the bake's sizes, which some shapes read (`u.hb`, `u.res`): the tracer's own `u`
/// is another block.
fn adapt(text: &str) -> String {
    let t = rename(text, "pr.", "sdfU.pr.");
    let t = rename(&t, "u.hb", "sdfU.hb");
    rename(&t, "u.res", "sdfU.res")
}

/// The names a text declares at its top level (`fn`, `const`, `struct`, `alias`, `var<...>` at the start
/// of a line).
pub(crate) fn top_names(text: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in text.lines() {
        let rest = ["fn ", "const ", "struct ", "alias ", "override "].iter().find_map(|k| line.strip_prefix(k))
            .or_else(|| line.strip_prefix("var<").and_then(|r| r.split_once('>')).map(|(_, r)| r.trim_start()));
        let Some(rest) = rest else { continue };
        let name: String = rest.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
        if !name.is_empty() && !names.contains(&name) { names.push(name); }
    }
    names
}

/// `text` with every name in `names` (whole names only, and not after a `.`, where a name is a member)
/// given `prefix`: the preset's own names must not meet the tracer's (a seashell has a `beyond`, and so
/// has the tracer).
pub(crate) fn prefix_names(text: &str, names: &[String], prefix: &str) -> String {
    let mut out = String::with_capacity(text.len() + 256);
    let mut word = String::new();
    let mut prev = ' ';
    let flush = |out: &mut String, word: &mut String, prev: char| {
        if !word.is_empty() {
            if prev != '.' && !word.starts_with(|c: char| c.is_ascii_digit()) && names.contains(word) { out.push_str(prefix); }
            out.push_str(word);
            word.clear();
        }
    };
    let mut before = ' ';
    for c in text.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            if word.is_empty() { before = prev; }
            word.push(c);
        } else {
            flush(&mut out, &mut word, before);
            out.push(c);
        }
        prev = c;
    }
    flush(&mut out, &mut word, before);
    out
}

/// What goes into the tracer to trace this preset from its formula: the formula's numbers (the same
/// block of 48 bytes the bake reads, `field::uniforms`), the preset's own text and `sdfDist(p)`, the
/// distance to the surface from a point of the object's frame, made as the bake makes the volume's
/// values: the preset's puff, its shell, its clip, the same smooth cut to the clip. None for a preset
/// that is not a formula.
pub fn tracer_decl(fam: &Family, p: &Preset) -> Option<String> {
    let Support::Formula { exact } = support(fam, p) else { return None };
    let Shape::Field { wgsl, clip, shell } = &p.shape else { return None };
    let mut members: Vec<String> = p.params.iter().take(MAX_PARAMS).map(|d| member_name(d.id)).collect();
    for i in members.len()..MAX_PARAMS { members.push(format!("_p{i}")); }
    let pr: Vec<String> = members.iter().map(|m| format!("{m}: f32")).collect();
    let estimate = if exact {
        "  var sd = raw(p);\n".to_string()
    } else {
        format!("  const e: f32 = {SLOPE_E};
  // (the four readings in a loop the compiler is not told the length of: the preset's text is then put
  // in once, not four times, wherever the tracer settles a hit; a larger shader runs slower everywhere)
  var f = 0.0;
  var g = vec3f(0.0);
  for (var i = 0u; i < SDF_TAPS; i++) {{
    let o = sdfTap(i);
    let v = raw(p + e * o);
    f += v;
    g += o * v;
  }}
  g /= 4.0 * e;
  // the value over the slope: the distance near the surface; held to a step that can be trusted
  var sd = clamp(0.25 * f * inverseSqrt(max(dot(g, g), 1e-12)), -SDF_STEP_MAX, SDF_STEP_MAX);
")
    };
    let head = format!(
        "
// ---- {name}, traced from its formula (gf_core::sdf) ----
struct SdfPr {{ {pr} }}
struct SdfU {{ hb: f32, res: f32, puff: f32, pad: f32, pr: SdfPr }}
@group(0) @binding({BINDING}) var<uniform> sdfU: SdfU;
const SDF_STEP_MAX: f32 = {STEP_MAX};
const SDF_NE: f32 = {NORMAL_E};
const SDF_BAND: f32 = {BAND:.1};
// how many readings a slope takes: four, though the compiler cannot see it (`pad` is always 0)
var<private> SDF_TAPS: u32;
// the corners of a tetrahedron, the four directions a slope is read in
fn sdfTap(i: u32) -> vec3f {{
  return vec3f(select(-1.0, 1.0, i == 0u || i == 3u), select(-1.0, 1.0, i >= 2u), select(-1.0, 1.0, i == 1u || i == 3u));
}}
",
        name = p.name, pr = pr.join(", "),
    );
    let own = format!("{helpers}\n{body}\n", helpers = HELPERS.trim_end(), body = adapt(wgsl).trim_end());
    let names = top_names(&own);
    let all = format!(
        "{own}fn sdfShell(p: vec3f) -> f32 {{ return {shell}; }}
fn sdfClip(p: vec3f) -> f32 {{ return {clip}; }}
// the distance to the surface from p, in the object's frame, as the bake would store it
fn sdfDist(p: vec3f) -> f32 {{
  SDF_TAPS = 4u + u32(sdfU.pad);
{estimate}  sd -= sdfU.puff;
  let w = sdfShell(p);
  if (w > 0.0) {{ sd = abs(sd) - w; }}
  return clamp(smax(sd, sdfClip(p), 0.03), -1.0, 1.0);
}}
// The outward normal at p, in the object's frame: the slope of the formula, four readings (a wall about
// the surface faces away from it on either side: the slope, turned to the side p is on; how its thickness
// changes along the surface is too slow to tilt it). Where the clip comes within the 0.03 it is blended
// over, its own slope is read at the same four points and the two are blended as the cut blends them
// (the share each has in a smooth maximum). The distance itself is never read here: it would put the
// whole formula into the shader once more.
fn sdfNormal(p: vec3f) -> vec3f {{
  SDF_TAPS = 4u + u32(sdfU.pad);
  let clipNear = sdfClip(p) >= -0.035;
  var f = 0.0;
  var g = vec3f(0.0);
  var cf = 0.0;
  var cg = vec3f(0.0);
  for (var i = 0u; i < SDF_TAPS; i++) {{
    let o = sdfTap(i);
    let v = raw(p + SDF_NE * o);
    f += v;
    g += o * v;
    if (clipNear) {{
      let c = sdfClip(p + SDF_NE * o);
      cf += c;
      cg += o * c;
    }}
  }}
  var n = normalize(g);
  // the distance at p, as sdfDist would have it (the readings' mean over the slope)
  var sd = f * SDF_NE * inverseSqrt(max(dot(g, g), 1e-30)) - sdfU.puff;
  let w = sdfShell(p);
  if (w > 0.0) {{
    n *= select(1.0, -1.0, sd < 0.0);
    sd = abs(sd) - w;
  }}
  if (clipNear) {{
    let share = clamp(0.5 + 0.5 * (sd - 0.25 * cf) / 0.03, 0.0, 1.0);
    n = normalize(share * n + (1.0 - share) * normalize(cg));
  }}
  return n;
}}
",
        shell = adapt(shell), clip = adapt(clip),
    );
    // (the preset's own names, and the helpers', made its own: `sdf_raw`, `sdf_smax`...)
    Some(head + &prefix_names(&all, &names, "sdf_"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rename_only_names() {
        assert_eq!(rename("pr.a + spr.b + x.pr.c + (pr.d)", "pr.", "S.pr."), "S.pr.a + spr.b + x.pr.c + (S.pr.d)");
        assert_eq!(rename("u.hb / u.res + mu.res", "u.res", "V"), "u.hb / V + mu.res");
    }

    #[test]
    fn every_formula_has_text_and_every_curve_none() {
        for f in crate::presets::families() {
            for p in &f.presets {
                let d = tracer_decl(&f, p);
                assert_eq!(d.is_some(), matches!(p.shape, Shape::Field { .. }), "{}/{}", f.id, p.id);
                if let Some(d) = d {
                    assert!(d.contains("fn sdfDist(") && !d.contains(" pr.") && !d.contains("(pr."), "{}/{}", f.id, p.id);
                }
            }
        }
    }
}
