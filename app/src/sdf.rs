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
    if src.matches(MAP_READ).count() != 1 || src.matches(MAPN_HEAD).count() != 1 || src.matches(NRM_STEP).count() != 1 { return src.to_string(); }
    let src = src
        .replace(MAP_READ, MAP_HYBRID)
        // (the volume's smooth reading is kept under another name, unused)
        .replace(MAPN_HEAD, &format!("{decl}\nfn mapN(p: vec3f) -> f32 {{ return sdfDist(p); }}\nfn mapNVolume(p: vec3f) -> f32 {{\n"))
        .replace(NRM_STEP, NRM_SDF);
    src.replace(GEM_UNEVEN, "*mOut = gemWorld(nk);")
}
