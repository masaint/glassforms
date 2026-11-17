//! A tracer that reads the shape's formula instead of its volume (gf_core::sdf): `map` and `mapN`, the
//! two readings every part of the tracer marches and settles by, become the formula's distance near the surface, and the
//! normals are the formula's own. Woven in only while a shape is traced so, so the
//! tracer of every other shape keeps its text.

use crate::render::Variant;

/// The plain reading of the volume in `map` (trace.wgsl).
const MAP_READ: &str = "  let v = textureSampleLevel(volTex, volSmp, p / (2.0 * HB) + 0.5, 0.0).r;\n";
/// What `map` reads instead: the volume, far from the surface, where it is a sure distance whatever the
/// formula's slope does (some formulas jump or flatten far from their surface, where the slope's
/// estimate is no distance at all: a prism's caps grew sheets of false surface in mid-air); within
/// `SDF_BAND` voxels of the surface, the formula. The volume only says how far a ray may skip; where
/// the ray stops, and which way the surface faces there, is the formula's.
const MAP_HYBRID: &str = "  var v = textureSampleLevel(volTex, volSmp, p / (2.0 * HB) + 0.5, 0.0).r;
  if (abs(v) < SDF_BAND * (2.0 * HB / u.vres)) { v = sdfDist(p); }
";
/// The head of the smooth reading.
const MAPN_HEAD: &str = "fn mapN(p: vec3f) -> f32 {\n";
/// How `nrm` reads the field: four smooth readings a voxel apart.
const NRM_STEP: &str = "  let e = vec2f(1.0, -1.0) * (2.0 * HB / u.vres);
  return normalize(e.xyy * mapNW(p + e.xyy) + e.yyx * mapNW(p + e.yyx) + e.yxy * mapNW(p + e.yxy) + e.xxx * mapNW(p + e.xxx));
";
/// What it reads instead: the formula's normal (`sdfNormal`, in the object's frame), turned into the world's.
const NRM_SDF: &str = "  let n = sdfNormal(toObj(p));
  return normalize(n.x * vec3f(u.obj0.x, u.obj1.x, u.obj2.x) + n.y * vec3f(u.obj0.y, u.obj1.y, u.obj2.y) + n.z * vec3f(u.obj0.z, u.obj1.z, u.obj2.z));
";
/// A cut stone's exact facet normal is given the volume's wander, to look as it always has (gem.wgsl);
/// traced from the formula it is left exact.
const GEM_UNEVEN: &str = "*mOut = gemUneven(gemWorld(nk), q);";

/// Can a tracer of this kind read the formula? A shape alone (no layers: their scene is put together
/// from several volumes, a stage grown for them included), without the material's bubbles, which are cut
/// into the volume.
pub fn fits(v: &Variant) -> bool {
    v.mat.bodies == 0 && !v.grown && !v.mat.bubbles
}

/// The tracer's text reading the formula `decl` (gf_core::sdf::tracer_decl). Text in which the readings
/// are not found as expected is given back as it was.
pub fn weave(src: &str, decl: &str) -> String {
    if decl.contains("fn csgDist(") { return weave_tree(src, decl); }
    if src.matches(MAP_READ).count() != 1 || src.matches(MAPN_HEAD).count() != 1 || src.matches(NRM_STEP).count() != 1 { return src.to_string(); }
    let band = gf_core::sdf::BAND;
    // (with no band the march is the volume's own: the formula only settles hits and gives normals)
    let src = if band > 0.0 { src.replace(MAP_READ, MAP_HYBRID) } else { src.to_string() };
    let src = settle_loops(&src);
    // What the preset works out once from its sliders (a cut stone's planes, a prism's sides), which it
    // does at its first evaluation behind a flag: done once at the start of the pixel instead, so that
    // that work is not put into the shader again at every place the formula is read.
    let (hoisted, setups) = hoist_setups(decl);
    let entry = src.find("\nfn fs(").and_then(|i| src[i..].find("{\n").map(|j| i + j + 2));
    let (src, decl) = match entry {
        Some(at) if !setups.is_empty() => (format!("{}{setups}{}", &src[..at], &src[at..]), hoisted),
        _ => (src, decl.to_string()),
    };
    let src = src
        // (the volume's smooth reading is kept under another name, unused)
        .replace(MAPN_HEAD, &format!("{decl}\nfn mapN(p: vec3f) -> f32 {{ return sdfDist(p); }}\nfn mapNVolume(p: vec3f) -> f32 {{\n"))
        .replace(NRM_STEP, NRM_SDF);
    src.replace(GEM_UNEVEN, "*mOut = gemWorld(nk);")
}

/// Every pair of settling steps (`t += mapNW(...)` twice, as the tracer settles a hit onto the surface)
/// made one step in a loop of two whose length the compiler cannot see (`sdfU.pad` is always 0): the
/// formula is then put in once at each place a hit is settled, not twice. A larger shader runs slower
/// everywhere, marching through the volume included (the seashell: 10.3 ms with the formula written out at
/// every step, 7.3 with its readings in loops).
fn settle_loops(src: &str) -> String {
    let lines: Vec<&str> = src.split_inclusive('\n').collect();
    let mut out = String::with_capacity(src.len() + 256);
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i];
        let t = l.trim_start();
        // (the code of a line, without its comment)
        let code = |x: &str| x.split("//").next().unwrap_or("").trim().to_string();
        let c = code(t);
        let settle = c.starts_with("t += mapNW(") || c.starts_with("t -= mapNW(") || c.starts_with("s -= mapNW(") || c.starts_with("s += mapNW(");
        if settle && i + 1 < lines.len() && code(lines[i + 1]) == c {
            let indent = &l[..l.len() - t.len()];
            out.push_str(&format!("{indent}for (var sdfS = 0u; sdfS < 2u + u32(sdfU.pad); sdfS++) {{ {c} }}\n"));
            i += 2;
        } else {
            out.push_str(l);
            i += 1;
        }
    }
    out
}

/// The formula's text with every first-evaluation setup (`if (!sdf_gem_ready) { sdf_gem_setup(); }`) taken
/// out, and the calls that do those setups, as lines to put at the start of the pixel.
fn hoist_setups(decl: &str) -> (String, String) {
    let mut out = String::with_capacity(decl.len());
    let mut calls = String::new();
    for line in decl.split_inclusive('\n') {
        let t = line.trim();
        let call = t.strip_prefix("if (!sdf_").and_then(|r| r.split_once(") { ")).and_then(|(_, r)| r.strip_suffix("(); }"));
        match call {
            Some(f) if f.starts_with("sdf_") && !f.contains(' ') => calls += &format!("  {f}();\n"),
            _ => out.push_str(line),
        }
    }
    (out, calls)
}

/// The tracer's text reading a tree of formula parts (gf_core::csg::wgsl), or several parts beside a volume
/// object. As for one shape, the march is the volume's own (here the volume the tree is measured into) and
/// each hit is settled onto the tree's surface and takes its normal. Where the volume also holds a volume
/// object (an attractor beside the parts: `csgU.info.w` = 1), the tree is taken only where it is the nearer,
/// by more than `SDF_NEAR` voxels; elsewhere the volume object's own smooth reading and normal stand.
pub fn weave_tree(src: &str, decl: &str) -> String {
    if src.matches(MAPN_HEAD).count() != 1 || src.matches(NRM_STEP).count() != 1 { return src.to_string(); }
    let mapn = "fn mapN(p: vec3f) -> f32 {
  let c = csgDist(p);
  if (csgU.info.w < 0.5) { return c; }
  let vn = mapNVolume(p);
  return select(vn, c, c < vn + SDF_NEAR * (2.0 * HB / u.vres));
}
const SDF_NEAR: f32 = 1.5;
";
    // (the tree is read at p and at the four taps of its normal in one loop: put into the shader once
    // wherever a normal is taken. Where the volume's own normal stands instead (beside a volume object whose
    // surface is the nearer there, or where a grid part alone decides the surface: the volume holds its
    // distances there), four smooth readings of the volume a voxel apart, also in a loop.)
    let nrm = "  let po = toObj(p);
  var near = 1e9;
  if (csgU.info.w > 0.5) { near = mapNVolume(po) + SDF_NEAR * (2.0 * HB / u.vres); }
  var n = csgNormal(po, near);
  if (all(n == vec3f(0.0))) {
    let e = 2.0 * HB / u.vres;
    for (var t = 0u; t < 4u + u32(csgU.info2.y); t++) {
      let o = csgTap(t);
      n += o * mapNVolume(po + e * o);
    }
  }
  // (a reading flat all about p, which only a march that gave up inside the glass meets, where the tree's
  // volume is held to how near the parts left out of a cell come: a normal, not none, so that no sample of
  // the picture is spoilt)
  if (dot(n, n) < 1e-30) { n = vec3f(0.0, 1.0, 0.0); }
  return normalize(n.x * vec3f(u.obj0.x, u.obj1.x, u.obj2.x) + n.y * vec3f(u.obj0.y, u.obj1.y, u.obj2.y) + n.z * vec3f(u.obj0.z, u.obj1.z, u.obj2.z));
";
    let src = settle_loops(src).replace("u32(sdfU.pad)", "u32(csgU.info2.y)");
    // (measuring only: GLASSFORMS_CSG_EXP=volnormal / volsettle take the tree out of the normal or the settling)
    let exp = std::env::var("GLASSFORMS_CSG_EXP").unwrap_or_default();
    let mapn = if exp.contains("volsettle") { "fn mapN(p: vec3f) -> f32 { return mapNVolume(p); }\nconst SDF_NEAR: f32 = 1.5;\n" } else { mapn };
    let nrm = if exp.contains("volnormal") { "  let po = toObj(p);
  var n = vec3f(0.0);
  let e = 2.0 * HB / u.vres;
  for (var t = 0u; t < 4u + u32(csgU.info2.y); t++) { let o = csgTap(t); n += o * mapNVolume(po + e * o); }
  return normalize(n.x * vec3f(u.obj0.x, u.obj1.x, u.obj2.x) + n.y * vec3f(u.obj0.y, u.obj1.y, u.obj2.y) + n.z * vec3f(u.obj0.z, u.obj1.z, u.obj2.z));
" } else { nrm };
    src.replace(MAPN_HEAD, &format!("{decl}\n{mapn}fn mapNVolume(p: vec3f) -> f32 {{\n"))
        .replace(NRM_STEP, nrm)
        .replace(GEM_UNEVEN, "*mOut = gemWorld(nk);")
}
