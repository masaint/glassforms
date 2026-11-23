//! A tracer that reads the shape's formula instead of its volume (gf_core::sdf): `map` and `mapN`, the
//! two readings every part of the tracer marches and settles by, become the formula's distance near the surface, and the
//! normals are the formula's own. Woven in only while a shape is traced so, so the
//! tracer of every other shape keeps its text.

use crate::render::Variant;
use crate::weave::Woven;

/// Measuring only. GF_TRACER_DUMP=<folder> writes every tracer's text there (`tracer_<hash>.wgsl`);
/// GF_TRACER_EDIT=<file> applies the edits the file lists to every tracer before it is compiled, each as
/// `<<<<\n old text\n====\n new text\n>>>>` (an old text not found is reported, not an error), so that
/// one build can time what each part of a tracer costs.
pub fn measure_edits(src: &str) -> Option<String> {
    let dump = std::env::var_os("GF_TRACER_DUMP");
    let edit = std::env::var_os("GF_TRACER_EDIT");
    if dump.is_none() && edit.is_none() { return None; }
    let mut out = src.to_string();
    if let Some(path) = edit {
        let text = std::fs::read_to_string(&path).unwrap_or_default().replace("\r\n", "\n");
        for block in text.split("<<<<\n").skip(1) {
            let Some((old, rest)) = block.split_once("\n====\n") else { continue };
            let new = rest.split_once("\n>>>>").map_or(rest, |(n, _)| n);
            let n = out.matches(old).count();
            eprintln!("[edit] {n} x {:?}", old.lines().next().unwrap_or(""));
            out = out.replace(old, new);
        }
    }
    if let Some(dir) = dump {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        out.hash(&mut h);
        let path = std::path::Path::new(&dir).join(format!("tracer_{:016x}.wgsl", h.finish()));
        let _ = std::fs::write(&path, &out);
        eprintln!("[dump] {}", path.display());
    }
    Some(out)
}

/// The plain reading of the volume in `map` (trace.wgsl).
const MAP_READ: &str = "  let v = textureSampleLevel(volTex, volSmp, p / (2.0 * HB) + 0.5, 0.0).r;\n";
const SAMPLE_HEAD: &str = "fn sampleAt(fc: vec2f, xi: f32, W: i32, lens: vec2f) -> vec4f {\n";
const WAVE_LOOP: &str = "  for (var i = 0; i < MAX_NW; i++) {\n";
const HERO_WAVE_LOOP: &str = "  for (var h = 0; h < MAX_NW; h++) {\n";
const MAT_WAVE_LOOP: &str = "  for (var j = 0; j < MAX_NW * MODES; j++) {\n";
const MAT_MONO_LOOP: &str = "  for (var j = 0; j < MODES; j++) {\n";
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
/// How `nrm` reads the field: the analytic derivative of the cubic B-spline (`nrmVolume`, 24 reads).
const NRM_STEP: &str = "  return nrmVolume(p);\n";
/// What it reads instead: the formula's normal (`sdfNormal`, in the object's frame), turned into the world's.
const NRM_SDF: &str = "  let n = sdfNormal(toObj(p));
  return normalize(n.x * vec3f(u.obj0.x, u.obj1.x, u.obj2.x) + n.y * vec3f(u.obj0.y, u.obj1.y, u.obj2.y) + n.z * vec3f(u.obj0.z, u.obj1.z, u.obj2.z));
";
/// `trace.wgsl` already computes the analytic cubic B-spline derivative (`nrmVolume`) and counts its
/// three eight-read planes when profiling.
pub(crate) fn conductor_volume_normal(src: &str, _profile: bool) -> String {
    src.to_string()
}

/// How many surfaces along a ray a formula with a table is read at (see `weave`): the first face, the
/// face the light leaves by, and the next it meets.
pub const EXACT_HITS: u32 = 2;
/// Swept tubes keep analytic normals for the first surfaces a path meets (each wavelength anew); later
/// ones are the volume's. 3, not 6: the same look in close-ups (the deeper reflections' speckle is the
/// volume trace's own) for Thomas 17.5 against 23.3 ms at 4K (docs/project-notes/plans/csg_perf.md).
pub const TUBE_EXACT_HITS: u32 = 3;
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
    if decl == GUARD_DECL { return guard_normals(src); }
    if decl.contains("fn csgDist(") { return weave_tree(src, decl); }
    if crate::weave::found(src, "sdf.guard_map_read", MAP_READ) != 1 || crate::weave::found(src, "sdf.guard_mapn_head", MAPN_HEAD) != 1 || crate::weave::found(src, "sdf.guard_nrm_step", NRM_STEP) != 1 { return src.to_string(); }
    let band = gf_core::sdf::BAND;
    // (with no band the march is the volume's own: the formula only settles hits and gives normals)
    let src = if band > 0.0 { src.woven("sdf.map_read", MAP_READ, MAP_HYBRID) } else { src.to_string() };
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
    // A formula with a table (a shape of many pieces: a snowflake, crystals, foam) is read only at the first
    // `EXACT_HITS` surfaces a ray meets; past them the light has been bent and blurred by as many faces, and
    // the volume's own reading and normal stand. Each reading of such a formula measures a cell's worth of
    // pieces, and glass full of faces is met many times over by every ray.
    if decl.contains("sdfTbl[") {
        let is_tubes = decl.contains("fn sdf_tube_eval(");
        if is_tubes {
            let out = src
                .woven("sdf.mapn_head", MAPN_HEAD, &format!("{decl}\nconst SDF_EXACT_HITS: u32 = {TUBE_EXACT_HITS}u;\nvar<private> SDF_HITS: u32 = 0u;\nfn mapN(p: vec3f) -> f32 {{ if (SDF_HITS >= SDF_EXACT_HITS) {{ return mapNVolume(p); }} return sdfDist(p); }}\nfn mapNVolume(p: vec3f) -> f32 {{\n"))
                .woven("sdf.nrm_step", NRM_STEP, &format!("  SDF_HITS += 1u;\n  if (SDF_HITS > SDF_EXACT_HITS) {{\n{NRM_STEP}  }}\n{NRM_SDF}"))
                .woven("sdf.sample_head", SAMPLE_HEAD, &format!("{SAMPLE_HEAD}  SDF_HITS = 0u;\n"))
                .woven("sdf.wave_loop", WAVE_LOOP, &format!("{WAVE_LOOP}    SDF_HITS = 1u;\n"))
                .woven("sdf.hero_wave_loop", HERO_WAVE_LOOP, &format!("{HERO_WAVE_LOOP}    SDF_HITS = 1u;\n"))
                .woven("sdf.mat_wave_loop", MAT_WAVE_LOOP, &format!("{MAT_WAVE_LOOP}    SDF_HITS = 1u;\n"))
                .woven("sdf.mat_mono_loop", MAT_MONO_LOOP, &format!("{MAT_MONO_LOOP}    SDF_HITS = 1u;\n"))
                .woven("sdf.gem_uneven", GEM_UNEVEN, "*mOut = gemWorld(nk);");
            let out = tube_primary(&out);
            return if decl.contains(TUBE_SHELL) && decl.contains(TUBE_CLIP) { tube_interior(&tube_newton(&out)) } else { out };
        }
        let out = src
            .woven("sdf.mapn_head", MAPN_HEAD, &format!("{decl}\nconst SDF_EXACT_HITS: u32 = {EXACT_HITS}u;\nvar<private> SDF_HITS: u32 = 0u;\nfn mapN(p: vec3f) -> f32 {{ if (SDF_HITS >= SDF_EXACT_HITS) {{ return mapNVolume(p); }} return sdfDist(p); }}\nfn mapNVolume(p: vec3f) -> f32 {{\n"))
            .woven("sdf.nrm_step", NRM_STEP, &format!("  SDF_HITS += 1u;\n  if (SDF_HITS > SDF_EXACT_HITS) {{\n{NRM_STEP}  }}\n{NRM_SDF}"))
            .woven("sdf.sample_head", SAMPLE_HEAD, &format!("{SAMPLE_HEAD}  SDF_HITS = 0u;\n"))
            .woven("sdf.gem_uneven", GEM_UNEVEN, "*mOut = gemWorld(nk);");
        return out;
    }
    // A formula too dear for every surface inside (gf_core::sdf::FIRST_ONLY_MARK): the first surface along
    // the eye's ray is settled onto it and takes its normal; every other is the volume's, as when traced from it.
    if decl.contains(gf_core::sdf::FIRST_ONLY_DECL) {
        if let Some(src) = first_only(&src) {
            return src
                .woven("sdf.mapn_head", MAPN_HEAD, &format!("{decl}\n{FIRST_DECL}{MAPN_HEAD}"))
                .woven("sdf.nrm_step", NRM_STEP, &format!("{FIRST_NRM}{NRM_STEP}"))
                .woven("sdf.gem_uneven", GEM_UNEVEN, "*mOut = gemWorld(nk);");
        }
    }
    let src = src
        // (the volume's smooth reading is kept under another name, unused)
        .woven("sdf.mapn_head", MAPN_HEAD, &format!("{decl}\nfn mapN(p: vec3f) -> f32 {{ return sdfDist(p); }}\nfn mapNVolume(p: vec3f) -> f32 {{\n"))
        .woven("sdf.nrm_step", NRM_STEP, NRM_SDF);
    src.woven("sdf.gem_uneven", GEM_UNEVEN, "*mOut = gemWorld(nk);")
}

/// A tube formula with no shell and no clip: its distance is the reading less the puff, and the reading
/// comes with its slope (`rawGrad`, one evaluation for both).
const TUBE_SHELL: &str = "fn sdfShell(p: vec3f) -> f32 { return 0.0; }";
const TUBE_CLIP: &str = "fn sdfClip(p: vec3f) -> f32 { return -1e4; }";
/// The tubes' readings as `tube_newton` makes them: each exact reading keeps its point and slope; a
/// settling step is the reading over the slope along the ray (Newton's step on the distance along the
/// ray: it lands on the surface in one step where the sphere's step creeps up on it at a slant), and a
/// normal asked for within `SDF_REUSE` of the last reading is that reading's own slope rather than a
/// third reading. Where the slope does not point against the ray (or lies within `SDF_NEWTON_C` of
/// across it) the step is the plain one.
const TUBE_NEWTON_MAPN: &str = "fn mapN(p: vec3f) -> f32 { if (SDF_HITS >= SDF_EXACT_HITS) { return mapNVolume(p); } return sdfDist(p); }";
const TUBE_NEWTON_DECL: &str = "const SDF_NEWTON_C: f32 = 0.25;
const SDF_REUSE: f32 = 1e-4;
var<private> sdfLastP: vec3f = vec3f(1e9);
var<private> sdfLastG: vec3f;
fn mapN(p: vec3f) -> f32 {
  if (SDF_HITS >= SDF_EXACT_HITS) { sdfLastP = vec3f(1e9); return mapNVolume(p); }
  let g = sdf_rawGrad(p);
  sdfLastP = p;
  sdfLastG = g.xyz;
  return clamp(g.w - sdfU.puff, -1.0, 1.0);
}
// how much longer than the reading a settling step along `ray` is (`sense`: 1 where the step is added,
// -1 where it is taken away)
fn sdfAlong(ray: vec3f, sense: f32) -> f32 {
  if (sdfLastP.x > 1e8) { return 1.0; }
  let n = sdfLastG;
  let gw = n.x * vec3f(u.obj0.x, u.obj1.x, u.obj2.x) + n.y * vec3f(u.obj0.y, u.obj1.y, u.obj2.y) + n.z * vec3f(u.obj0.z, u.obj1.z, u.obj2.z);
  let s = dot(gw, ray) * sense;
  return select(1.0, 1.0 / max(-s, SDF_NEWTON_C), s < 0.0);
}";
const TUBE_NEWTON_NRM: &str = "  let n = sdfNormal(toObj(p));\n";
const TUBE_NEWTON_NRM_REUSE: &str = "  let po = toObj(p);
  let dp = po - sdfLastP;
  var n = sdfLastG;
  if (dot(dp, dp) > SDF_REUSE * SDF_REUSE) { n = sdfNormal(toObj(p)); }
";

/// Newton settling and the normal from the settling reading, for a tube formula (see `TUBE_NEWTON_DECL`).
/// Each change stands alone: one whose text is not found is left out.
fn tube_newton(src: &str) -> String {
    let mut out = src.replace(TUBE_NEWTON_MAPN, TUBE_NEWTON_DECL);
    if out == src { return out; }
    if out.matches(TUBE_NEWTON_NRM).count() == 1 { out = out.replace(TUBE_NEWTON_NRM, TUBE_NEWTON_NRM_REUSE); }
    // every settling loop (`settle_loops`): `let sdfD = mapNW(o + d * s); s += sdfD;` (or `-=`)
    let mut done = String::with_capacity(out.len() + 1024);
    let mut rest = out.as_str();
    const HEAD: &str = "let sdfD = mapNW(";
    while let Some(i) = rest.find(HEAD) {
        let at = i + HEAD.len();
        done.push_str(&rest[..at]);
        rest = &rest[at..];
        let Some(close) = rest.find(");") else { continue };
        let arg = &rest[..close];
        let after = &rest[close + 2..];
        // (the ray's direction and its length's name, from `o + d * s`, and which way the step is taken)
        let parts = arg.split_once(" + ").and_then(|(_, r)| r.split_once(" * "));
        let sense = parts.and_then(|(_, s)| {
            if after.starts_with(&format!(" {s} += sdfD;")) { Some(1.0) } else if after.starts_with(&format!(" {s} -= sdfD;")) { Some(-1.0) } else { None }
        });
        match (parts, sense) {
            (Some((d, _)), Some(k)) => { done.push_str(arg); done.push_str(&format!(") * sdfAlong({d}, {k:.1});")); rest = after; }
            _ => {}
        }
    }
    done.push_str(rest);
    done.replace("        sdfLast = mapNW(ro + rd * t);\n        t += sdfLast;\n", "        sdfLast = mapNW(ro + rd * t);\n        t += sdfLast * sdfAlong(rd, 1.0);\n")
}

/// Inside the glass, and beyond it, a tube hit is settled by the volume; only the first surface along
/// the eye's ray (its settling, its rescue of grazing misses) reads the formula, and every normal of the
/// first `SDF_EXACT_HITS` stays the formula's. The interior settling readings were most of the cost (Thomas,
/// 4K: 71 to 23 ms a frame) and showed nothing a 100% close-up could tell from settling exactly (against
/// a reference settled to 1e-7 at every surface, pixels over 32/255: 1.04% settled exactly, 2.14% by the
/// volume, 6.8% for the volume trace; docs/project-notes/plans/csg_perf.md). GF_TUBE_INTERIOR=exact keeps
/// the exact settling everywhere, for comparing.
fn tube_interior(src: &str) -> String {
    if std::env::var("GF_TUBE_INTERIOR").is_ok_and(|v| v == "exact") { return src.to_string(); }
    const MAPN: &str = "fn mapN(p: vec3f) -> f32 {
  if (SDF_HITS >= SDF_EXACT_HITS) { sdfLastP = vec3f(1e9); return mapNVolume(p); }";
    if src.matches(MAPN).count() != 1 { return src.to_string(); }
    let out = src.replace(MAPN, "fn mapN(p: vec3f) -> f32 { sdfLastP = vec3f(1e9); return mapNVolume(p); }
// the formula, where the first surface is settled
fn mapNWX(p: vec3f) -> f32 { return mapNX(toObj(p)); }
fn mapNX(p: vec3f) -> f32 {
  if (SDF_HITS >= SDF_EXACT_HITS) { sdfLastP = vec3f(1e9); return mapNVolume(p); }");
    // (in every `primary`: its settling and its rescue)
    let mut done = String::with_capacity(out.len() + 256);
    let mut rest = out.as_str();
    while let Some(i) = rest.find("
fn primary(") {
        done.push_str(&rest[..i]);
        let len = rest[i + 1..].find("
fn ").map_or(rest.len() - i, |j| j + 1);
        done.push_str(&rest[i..i + len].replace("mapNW(", "mapNWX("));
        rest = &rest[i + len..];
    }
    done.push_str(rest);
    done
}

/// Keeps the primary ray march purely in `volTex` (`mapW`) while eliminating swept-tube false hits
/// and grazing silhouette voxel bites: exact `mapNW` confirms hits (advancing past guide-volume
/// bridges instead of stalling) and rescues miss rays whose closest `volTex` approach grazed within
/// 0.75 voxel of a strand.
fn tube_primary(src: &str) -> String {
    const HEAD: &str = "\nfn primary(";
    const LOOP: &str = "  for (var i = 0; i < STEPS_OUT; i++) {\n";
    const READ_D: &str = "    let d = mapW(ro + rd * t);\n";
    const SETTLE: &str = "      for (var sdfS = 0u; sdfS < 2u + u32(sdfU.pad); sdfS++) { let sdfD = mapNW(ro + rd * t); t += sdfD; if (abs(sdfD) < 0.00015) { break; } }\n      return t;\n";
    const MISS: &str = "\n  return -1.0;\n}";
    if !src.contains(HEAD) { return src.to_string(); }
    let mut out = String::with_capacity(src.len() + 512);
    let mut rest = src;
    while let Some(i) = rest.find(HEAD) {
        out.push_str(&rest[..i]);
        let len = rest[i + 1..].find("\nfn ").map_or(rest.len() - i, |j| j + 1);
        let body = &rest[i..i + len];
        if body.matches(LOOP).count() == 1
            && body.matches(READ_D).count() == 1
            && body.matches(SETTLE).count() == 1
            && body.rfind(MISS).is_some()
        {
            let transformed = body
                .woven("sdf.loop", LOOP, &format!("  var sdfBestD = 1e9;\n  var sdfBestT = -1.0;\n{LOOP}"))
                .woven("sdf.read_d", READ_D, &format!("{READ_D}    if (d < sdfBestD) {{ sdfBestD = d; sdfBestT = t; }}\n"))
                .replace(
                    SETTLE,
                    "      let tHit0 = t;\n      var sdfLast = 0.0;\n      for (var sdfS = 0u; sdfS < 2u + u32(sdfU.pad); sdfS++) {\n        sdfLast = mapNW(ro + rd * t);\n        t += sdfLast;\n        if (abs(sdfLast) < 0.00015) { break; }\n      }\n      if (sdfLast < 0.0008) { return t; }\n      t = max(t, tHit0 + 0.001);\n      if (t > tFar) { break; }\n      continue;\n",
                );
            if let Some(miss_at) = transformed.rfind(MISS) {
                out.push_str(&transformed[..miss_at]);
                out.push_str(
                    "\n  let sdfVox = 2.0 * HB / u.vres;\n  if (sdfBestD < 0.75 * sdfVox) {\n    var rt = max(sdfBestT - 0.75 * sdfVox, max(-b - h, 0.0));\n    let rtEnd = min(sdfBestT + 1.5 * sdfVox, tFar);\n    for (var ri = 0; ri < 6; ri++) {\n      let rd0 = mapNW(ro + rd * rt);\n      if (rd0 < 0.0005) {\n        rt = max(rt + rd0, 0.0);\n        if (abs(rd0) >= 0.00015) { rt = max(rt + mapNW(ro + rd * rt), 0.0); }\n        return rt;\n      }\n      rt += rd0;\n      if (rt > rtEnd) { break; }\n    }\n  }\n  return -1.0;\n}",
                );
                out.push_str(&transformed[miss_at + MISS.len()..]);
                rest = &rest[i + len..];
                continue;
            }
        }
        out.push_str(body);
        rest = &rest[i + len..];
    }
    out.push_str(rest);
    out
}

/// What names, and what is given as the formula of, a tracer that reads a tree measured into a volume (a tree's
/// parts traced from its volume, a volume object made from a tree, a scene of several objects with a tree in
/// it): the volume's own march and normals, the normal guarded (`guard_normals`).
pub const GUARD_KEY: &str = "merged";
pub const GUARD_DECL: &str = "// a tree measured into a volume: normals guarded";

/// The volume tracer with its normal guarded against a reading flat all about a point (`nrmVolume` already
/// guards `dot(local, local) < 1e-30`; kept distinct so merged-tree tracer keys remain distinguishable).
pub fn guard_normals(src: &str) -> String {
    if crate::weave::found(src, "sdf.guard_nrm_step", NRM_STEP) != 1 { return src.to_string(); }
    src.woven("sdf.nrm_step", NRM_STEP, "  return nrmVolume(p); // guarded\n")
}

/// What a formula read at the first surface only adds to the tracer: its own settling reading and its normal,
/// which only `primary` uses (so that none of the formula's text is put into the loops inside the glass,
/// where it would make the whole shader dearer even where it is never read), and the normal kept for the
/// first `nrm` of the sample.
const FIRST_DECL: &str = "fn mapNWF(p: vec3f) -> f32 { return sdfDist(toObj(p)); }
fn nrmF(p: vec3f) -> vec3f {
  let n = sdfNormal(toObj(p));
  return normalize(n.x * vec3f(u.obj0.x, u.obj1.x, u.obj2.x) + n.y * vec3f(u.obj0.y, u.obj1.y, u.obj2.y) + n.z * vec3f(u.obj0.z, u.obj1.z, u.obj2.z));
}
// the first surface's normal, from the formula (kept where `primary` finds the surface), and whether it is
// still to be used
var<private> sdfN: vec3f;
var<private> sdfFirst: bool = false;
";
/// The first `nrm` after a surface is found is the formula's normal, kept by `primary`; every later one is
/// the volume's.
const FIRST_NRM: &str = "  if (sdfFirst) { sdfFirst = false; return sdfN; }
";
/// The tracer with the first surface traced from the formula: in every `primary` (of whichever tracer) the
/// hit is settled onto the formula and its normal kept. None if the text is not as expected.
fn first_only(src: &str) -> Option<String> {
    const HEAD: &str = "\nfn primary(";
    const HIT: &str = "      return t;\n";
    if !src.contains(HEAD) { return None; }
    let mut out = String::with_capacity(src.len() + 256);
    let mut rest = src;
    while let Some(i) = rest.find(HEAD) {
        out.push_str(&rest[..i]);
        let len = rest[i + 1..].find("\nfn ").map_or(rest.len() - i, |j| j + 1);
        let body = &rest[i..i + len];
        if body.matches(HIT).count() != 1 || !body.contains("mapNW(") { return None; }
        out.push_str(&body.replace("mapNW(", "mapNWF(").woven("sdf.hit", HIT, "      sdfN = nrmF(ro + rd * t);\n      sdfFirst = true;\n      return t;\n"));
        rest = &rest[i + len..];
    }
    out.push_str(rest);
    Some(out)
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
            if let Some((lhs, op, rhs)) = c.strip_suffix(';').and_then(|s| {
                s.split_once(" += ")
                    .map(|(l, r)| (l, "+=", r))
                    .or_else(|| s.split_once(" -= ").map(|(l, r)| (l, "-=", r)))
            }) {
                out.push_str(&format!("{indent}for (var sdfS = 0u; sdfS < 2u + u32(sdfU.pad); sdfS++) {{ let sdfD = {rhs}; {lhs} {op} sdfD; if (abs(sdfD) < 0.00015) {{ break; }} }}\n"));
            } else {
                out.push_str(&format!("{indent}for (var sdfS = 0u; sdfS < 2u + u32(sdfU.pad); sdfS++) {{ {c} }}\n"));
            }
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

/// Replace the distance corrections at each ray hit by a Newton step using the
/// derivative along that ray. A warp may shrink its field by a large bound.
fn warp_settles(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut lines = src.lines().peekable();
    while let Some(line) = lines.next() {
        let code = line.split("//").next().unwrap_or(line).trim();
        let replacement = match code {
            "t += mapNW(ro + rd * t);" | "t -= mapNW(ro + rd * t);" => "t += warpStep(ro + rd * t, rd);",
            "t -= mapNW(p + d * t);" => "t += warpStep(p + d * t, d);",
            "s -= mapNW(pos + dir * s);" => "s += warpStep(pos + dir * s, dir);",
            _ => { out.push_str(line); out.push('\n'); continue; }
        };
        if lines.peek().is_some_and(|next| next.trim() == code) { lines.next(); }
        out.push_str(&line[..line.len() - line.trim_start().len()]);
        out.push_str(replacement);
        out.push('\n');
    }
    out
}

const WARP_HIT: &str = r#"
var<private> warpPoint: vec3f;
var<private> warpSlope: vec3f;
fn warpStep(p: vec3f, ray: vec3f) -> f32 {
  let sample = csgWarpGradient(toObj(p));
  let slope = sample.xyz;
  let dir = toObj(ray);
  let deriv = dot(slope, dir);
  // Remain within the few voxels around the candidate hit, also at a silhouette.
  let band = 2.0 * (2.0 * HB / u.vres);
  let least = max(0.04 * length(slope), 1e-20);
  let denom = select(-1.0, 1.0, deriv >= 0.0) * max(abs(deriv), least);
  let step = clamp(-sample.w / denom, -band, band);
  // Cache the normal at the sample point. The corrected point may be elsewhere.
  warpPoint = p;
  warpSlope = slope;
  return step;
}
fn warpNormal(p: vec3f) -> vec3f {
  var n = warpSlope;
  let delta = p - warpPoint;
  if (dot(delta, delta) > SDF_NE * SDF_NE || dot(n, n) < 1e-24) {
    n = csgWarpGradient(toObj(p)).xyz;
  }
  return warpUnitNormal(n);
}
fn warpUnitNormal(slope: vec3f) -> vec3f {
  var n = slope;
  if (dot(n, n) < 1e-24) { n = vec3f(0.0, 1.0, 0.0); }
  return normalize(n.x * vec3f(u.obj0.x, u.obj1.x, u.obj2.x) + n.y * vec3f(u.obj0.y, u.obj1.y, u.obj2.y) + n.z * vec3f(u.obj0.z, u.obj1.z, u.obj2.z));
}
"#;

/// Settle the hit and read its final normal in the same rolled loop. A full
/// repeated motif may span several parts and sectors; sharing the shader body
/// keeps both gradient evaluations resident in the instruction cache.
fn profile_hit() -> String {
    let start=WARP_HIT.find("  let sample =").unwrap();
    let end=WARP_HIT.find("  return step;").unwrap();
    let mut shader=WARP_HIT.to_owned();
    shader.replace_range(start..end, r#"  let dir = toObj(ray);
  let band = 2.0 * (2.0 * HB / u.vres);
  var step = 0.0;
  for (var i=0u; i<2u+u32(csgU.info2.y); i++) {
    let at=p+ray*step;
    let sample=csgWarpGradient(toObj(p)+dir*step);
    warpPoint=at;
    warpSlope=sample.xyz;
    if (i>0u) { break; }
    let deriv=dot(sample.xyz,dir);
    let least=max(0.04*length(sample.xyz),1e-20);
    let denom=select(-1.0,1.0,deriv>=0.0)*max(abs(deriv),least);
    step=clamp(-sample.w/denom,-band,band);
    if (abs(step)<=SDF_NE && dot(sample.xyz,sample.xyz)>=1e-24) { break; }
  }
"#);
    shader
}

/// A prepared profile computes its final slope while settling the optical hit.
fn settled_profile_normals(mut src: String) -> String {
    let march = src.split_once("fn marchIn(").and_then(|(_, rest)| rest.split_once("\n}\n"))
        .map_or("", |(body, _)| body);
    let inside_settles = march.matches("return ").count() == 1
        && march.ends_with("t += warpStep(ro + rd * t, rd);\n  return t;");
    let outside = "t += warpStep(ro + rd * t, rd);\n    let p = ro + rd * t;\n    let n = ";
    src = src.woven("sdf.profile_outside_normal", &format!("{outside}nrm(p);"), &format!("{outside}warpUnitNormal(warpSlope);"));
    if inside_settles {
        for lead in [
            "let s = marchIn(q0, d2);\n    let q = q0 + d2 * s;\n    var m = ",
            "let s = marchIn(pos, dir);\n      let q = pos + dir * s;\n      var m = ",
        ] {
            src = src.woven("sdf.profile_inside_normal", &format!("{lead}nrm(q);"), &format!("{lead}warpUnitNormal(warpSlope);"));
        }
    }
    src
}

/// The standard glass camera ends either at a surface hit or within a pixel
/// of its silhouette. Both need the gradient. Sharing a single evaluation
/// site keeps the complete wrapped motif out of the camera shader twice.
fn finish_profile_camera(src: &mut String, hit: &mut String) {
    if src.matches("shade(").count() != 2
        || !src.contains("col = shade(ro, rd, t, xi, W);")
        || !src.contains("var t = primary(ro, rd, &tg, &g);")
        || !src.contains("env(reflect(rd, nrm(p)))") {
        return;
    }
    let Some(start) = src.find("fn primary(ro:") else { return };
    let Some(length) = src[start..].find("\n}\n") else { return };
    let range = start..start + length + 2;
    let original = &src[range.clone()];
    let exit = "t += warpStep(ro + rd * t, rd);\n      return t;";
    if !original.contains(exit) || !original.contains("let px = 2.0 * u.tan_fov / u.res.y;") {
        return;
    }
    let camera = original.woven("sdf.profile_camera_loop", "  for (var i = 0; i < STEPS_OUT; i++)", "  var found = false;\n  for (var i = 0; i < STEPS_OUT; i++)")
        .woven("sdf.profile_camera_exit", exit, "found = true; break;")
        .woven("sdf.profile_camera_miss", "  return -1.0;\n}", r#"  if (found || *graze < px) {
    let depth = select(*tGraze, t, found);
    let delta = profileFinish(ro + rd * depth, rd, found);
    if (found) { return t + delta; }
  }
  return -1.0;
}"#);
    src.replace_range(range, &camera);
    let shade = "fn shade(ro: vec3f, rd: vec3f, t: f32, xi: f32, W: i32) -> vec3f {\n  let p = ro + rd * t;\n  let n = ";
    *src = src.woven("sdf.profile_shade_normal", &format!("{shade}nrm(p);"), &format!("{shade}warpUnitNormal(warpSlope);"))
        .woven("sdf.profile_shade_env", "env(reflect(rd, nrm(p)))", "env(reflect(rd, warpUnitNormal(warpSlope)))");
    *hit = hit.woven("sdf.profile_warp_step", "fn warpStep(p: vec3f, ray: vec3f) -> f32 {",
        "fn warpStep(p: vec3f, ray: vec3f) -> f32 { return profileFinish(p,ray,true); }\nfn profileFinish(p:vec3f,ray:vec3f,refine:bool)->f32 {")
        .woven("sdf.profile_refine", "if (i>0u) { break; }", "if (i>0u || !refine) { break; }");
}

/// The tracer's text reading a tree of formula parts (gf_core::csg::wgsl), or several parts beside a volume
/// object. As for one shape, the march is the volume's own (here the volume the tree is measured into) and
/// each hit is settled onto the tree's surface and takes its normal. Where the volume also holds a volume
/// object (an attractor beside the parts: `csgU.info.w` = 1), the tree is taken only where it is the nearer,
/// by more than `SDF_NEAR` voxels; elsewhere the volume object's own smooth reading and normal stand.
pub fn weave_tree(src: &str, decl: &str) -> String {
    if crate::weave::found(src, "sdf.guard_mapn_head", MAPN_HEAD) != 1 || crate::weave::found(src, "sdf.guard_nrm_step", NRM_STEP) != 1 { return src.to_string(); }
    let material;
    let src = if decl.contains("// construction surface materials") {
        material = crate::construction_material::weave(src, decl);
        material.as_str()
    } else { src };
    if decl.contains("fn csgWarpGradient(") {
        let mut tracer=warp_settles(src);
        let envelope=if decl.contains("// cone envelope in object coordinates") {
            tracer=tracer.woven("sdf.cone_entry", "  let tFar = -b + h;", "  let tFar = -b + h;\n  t=max(t,coneEntry(ro,rd,tFar)); if(t>tFar){return -1.0;}");
            // Outside rays also cross the empty interior between opposite
            // lattice faces. Their protected shell contains all exact motifs.
            tracer=tracer.woven("sdf.cone_skip", "t += d * STEP_SCALE;\n      if (t > tFar)","if(d>0.05){t+=max(d*STEP_SCALE,coneEntry(ro+rd*t,rd,tFar-t));}else{t+=d*STEP_SCALE;}\n      if (t > tFar)");
            include_str!("shaders/profile_cone.wgsl")
        } else { "" };
        let envelope = if decl.contains("// precomputed cone envelope") {
            envelope.woven("sdf.cone_k", "let k=(m.q5.z-m.q5.x)/m.q5.y;", "let coeff=modAt(0).q8; let k=coeff.x;")
                .woven("sdf.cone_stretch", "let stretch=sqrt(1.0+k*k);", "let stretch=coeff.y;")
                .woven("sdf.cone_mid", "let mid=0.5*(m.q5.x+m.q5.z)+m.q4.w*stretch;", "let mid=coeff.z;")
        } else { envelope.to_owned() };
        let repeated = decl.contains("fn repeatHit(") || decl.contains("// cone envelope in object coordinates");
        let hit=if repeated && decl.contains("fn motifSlope(") {
            tracer = settled_profile_normals(tracer);
            let mut shader = profile_hit();
            if decl.contains("// precomputed cone envelope") {
                finish_profile_camera(&mut tracer, &mut shader);
            }
            shader
        } else { WARP_HIT.to_owned() };
        return tracer
            .woven("sdf.mapn_head", MAPN_HEAD, &format!("{decl}\n{envelope}\n{hit}\n{MAPN_HEAD}"))
            .woven("sdf.nrm_step", NRM_STEP, "  return warpNormal(p);\n")
            .woven("sdf.gem_uneven", GEM_UNEVEN, "*mOut = gemWorld(nk);");
    }
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
    let out = src.woven("sdf.mapn_head", MAPN_HEAD, &format!("{decl}\n{mapn}fn mapNVolume(p: vec3f) -> f32 {{\n"))
        .woven("sdf.nrm_step", NRM_STEP, nrm)
        .woven("sdf.gem_uneven", GEM_UNEVEN, "*mOut = gemWorld(nk);");
    tree_interior(&out)
}

/// How many normals of a sample are the tree's (the first surface's among them); later ones are the
/// volume's (see `tree_interior`).
// (6, not every normal: deeper ones read from the volume show the same speckle in deep internal reflections
// as the volume trace itself, and cost tree_stress 47.8 against 27.7 ms; 64 makes every one exact)
pub const TREE_EXACT_NORMALS: u32 = 6;

/// The tree read as the tubes are (`tube_interior`): the first surface along the eye's ray is settled onto
/// the tree, every later one by the volume, and the first `TREE_EXACT_NORMALS` normals of a sample are the
/// tree's, the first of them from the tree's reading where the hit was settled (the part that decides there,
/// four readings of it alone, not one of the whole tree and four). Interior settling and normals were most
/// of the cost of a tree traced exactly (tree_stress, 4K: 68 to about 22 ms a frame); docs/project-notes/
/// plans/csg_perf.md has the pictures. GF_TREE_INTERIOR=exact keeps the tree at every surface, for
/// comparing.
fn tree_interior(src: &str) -> String {
    if std::env::var("GF_TREE_INTERIOR").is_ok_and(|v| v == "exact") { return src.to_string(); }
    const MAPN: &str = "fn mapN(p: vec3f) -> f32 {\n  let c = csgDist(p);\n";
    const NRM: &str = "  let po = toObj(p);\n  var near = 1e9;\n";
    const LOOP: &str = "  var g = vec3f(0.0);\n  for (var t = 0u; t < 5u + u32(csgU.info2.y); t++) {\n";
    if src.matches(MAPN).count() != 1 || src.matches(NRM).count() != 1 || src.matches(LOOP).count() != 1 || !src.contains(SAMPLE_HEAD) { return src.to_string(); }
    let out = src
        .replace(MAPN, &format!("var<private> csgLastP: vec3f = vec3f(1e9);
var<private> csgLastW: f32 = 0.0;
var<private> csgHits: u32 = 0u;
const CSG_EXACT_NORMALS: u32 = {TREE_EXACT_NORMALS}u;
fn mapN(p: vec3f) -> f32 {{ return mapNVolume(p); }}
// the tree, where the first surface is settled; it keeps where it read and which part decides there
fn mapNWX(p: vec3f) -> f32 {{ return mapNX(toObj(p)); }}
fn mapNX(p: vec3f) -> f32 {{
  let cw = csgDistW(p, -1);
  csgLastP = p;
  csgLastW = cw.y;
  let c = cw.x;
"))
        .replace(NRM, &format!("  if (csgHits >= CSG_EXACT_NORMALS) {{\n  {}  }}\n  csgHits += 1u;\n{NRM}", NRM_STEP.trim_start()))
        .replace(LOOP, "  var g = vec3f(0.0);
  // (at the point the tree was last read, within a fifth of a normal's tap: which part decides is known)
  var t0 = 0u;
  let dl = p - csgLastP;
  if (dot(dl, dl) < 4e-8 && near > 1e8) {
    t0 = 1u;
    w = csgLastW;
    i = i32(abs(w)) - 1;
    dpm = mat3x3f(SDF_NE, 0.0, 0.0, 0.0, SDF_NE, 0.0, 0.0, 0.0, SDF_NE);
    if (i >= 0) {
      let i0 = csgP[i].inv0.xyz;
      let i1 = csgP[i].inv1.xyz;
      let i2 = csgP[i].inv2.xyz;
      dpm = mat3x3f(i0 * (SDF_NE / dot(i0, i0)), i1 * (SDF_NE / dot(i1, i1)), i2 * (SDF_NE / dot(i2, i2)));
    }
  }
  for (var t = t0; t < 5u + u32(csgU.info2.y); t++) {
")
        .replace(SAMPLE_HEAD, &format!("{SAMPLE_HEAD}  csgHits = 0u;\n"));
    primary_reads(&out, "mapNW(", "mapNWX(")
}

/// `src` with `from` made `to` in every `primary` (whichever tracer it is in).
fn primary_reads(src: &str, from: &str, to: &str) -> String {
    let mut done = String::with_capacity(src.len() + 256);
    let mut rest = src;
    while let Some(i) = rest.find("\nfn primary(") {
        done.push_str(&rest[..i]);
        let len = rest[i + 1..].find("\nfn ").map_or(rest.len() - i, |j| j + 1);
        done.push_str(&rest[i..i + len].replace(from, to));
        rest = &rest[i + len..];
    }
    done.push_str(rest);
    done
}
