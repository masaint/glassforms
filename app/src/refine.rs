//! Selective refinement of thin parts (parts engine): the tracer's reading of the fine bricks.
//!
//! A thin part (a rigging stay, a rail) is thinner than the scene volume's voxels. The parts engine bakes a small brick of
//! finer voxels (`parts_volume`) for each cell of its grid that holds one; the tracer reads the brick where there is one.
//! What it reads: a page (the cell grid, one number a cell: its brick's slot + 1, 0 for none) and an atlas (the bricks side
//! by side in one 3D texture, each with a one-texel apron so that the cubic reading does not leave its brick).
//!
//! Where it is read: `mapN` (the settled surface) and `nrmVolume` (the normal) only, never `map`: the march's reading is the
//! hottest in the tracer, and a new read there cost 12 ms a frame at 4K (lab). A tracer built without the refinement is
//! the tracer as it always was: none of this text is in it.

pub const PAGE_BINDING: u32 = 70;
pub const ATLAS_BINDING: u32 = 71;
pub const UNIFORM_BINDING: u32 = 72;

/// What the tracer's text declares: the page, the atlas and the numbers (the cell grid's low corner and cell side; the gate, the
/// brick's texels a side, bricks along x and y; the fine voxel and the cells each way; the atlas' texels each way).
const DECL: &str = "
@group(0) @binding(70) var fpPage: texture_3d<f32>;
@group(0) @binding(71) var fpAtlas: texture_3d<f32>;
struct FR { a: vec4f, b: vec4f, c: vec4f, d: vec4f }
@group(0) @binding(72) var<uniform> fr: FR;
// the atlas coordinate of p (the object's own frame) in its cell's brick (w = 1), or w = 0 where the cell has none
fn fineCoord(p: vec3f) -> vec4f {
  let cf = floor((p - fr.a.xyz) / fr.a.w);
  if (any(cf < vec3f(0.0)) || any(cf >= fr.c.yzw)) { return vec4f(0.0); }
  let k = textureLoad(fpPage, vec3i(cf), 0).r - 1.0;
  if (k < 0.0) { return vec4f(0.0); }
  let org = vec3f(k % fr.b.z, floor(k / fr.b.z) % fr.b.w, floor(k / (fr.b.z * fr.b.w))) * fr.b.y;
  let u = (p - (fr.a.xyz + cf * fr.a.w)) / fr.c.x + 1.0;
  return vec4f((org + u) / fr.d.xyz, 1.0);
}
fn fineAt(p: vec3f) -> vec2f {
  let c = fineCoord(p);
  if (c.w < 0.5) { return vec2f(0.0); }
  return vec2f(textureSampleLevel(fpAtlas, volSmp, c.xyz, 0.0).r, 1.0);
}
";

/// The tracer's text with the refinement read in (given back as it was where its anchors are not found).
pub fn weave(src: &str) -> String {
    let anchor = "@group(0) @binding(4) var volSmp: sampler;";
    let map_n = "fn mapN(p: vec3f) -> f32 {\n";
    let nrm = "fn nrmVolume(p: vec3f) -> vec3f {\n";
    if src.matches(anchor).count() != 1 || src.matches(map_n).count() != 1 || src.matches(nrm).count() != 1 { return src.to_string(); }
    let src = src.replacen(anchor, &format!("{anchor}{DECL}"), 1);
    let src = src.replacen(map_n, &format!("{map_n}  let f0 = fineAt(p);\n  if (f0.y > 0.5) {{ return f0.x; }}\n"), 1);
    // (the normal from the brick: half a fine voxel either side, inside its apron: one page read, six atlas reads)
    let fine_normal = "  let po = toObj(p);
  let fc = fineCoord(po);
  if (fc.w > 0.5) {
    let e = 0.5 / fr.d.xyz;
    let g = vec3f(
      textureSampleLevel(fpAtlas, volSmp, fc.xyz + vec3f(e.x, 0.0, 0.0), 0.0).r - textureSampleLevel(fpAtlas, volSmp, fc.xyz - vec3f(e.x, 0.0, 0.0), 0.0).r,
      textureSampleLevel(fpAtlas, volSmp, fc.xyz + vec3f(0.0, e.y, 0.0), 0.0).r - textureSampleLevel(fpAtlas, volSmp, fc.xyz - vec3f(0.0, e.y, 0.0), 0.0).r,
      textureSampleLevel(fpAtlas, volSmp, fc.xyz + vec3f(0.0, 0.0, e.z), 0.0).r - textureSampleLevel(fpAtlas, volSmp, fc.xyz - vec3f(0.0, 0.0, e.z), 0.0).r);
    if (dot(g, g) > 1e-30) { return normalize(g.x * vec3f(u.obj0.x, u.obj1.x, u.obj2.x) + g.y * vec3f(u.obj0.y, u.obj1.y, u.obj2.y) + g.z * vec3f(u.obj0.z, u.obj1.z, u.obj2.z)); }
  }
";
    src.replacen(nrm, &format!("{nrm}{fine_normal}"), 1)
}
