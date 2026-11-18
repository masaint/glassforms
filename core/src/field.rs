//! The shader that turns an equation into a distance field. Mirrors FIELD_HEAD, FIELD_MAIN and bakeField
//! in `forms.js`.
//!
//! A field preset gives `raw(p)`: any function that is negative inside the glass and positive outside.
//! Its values are not distances, and dividing by the slope (the usual shortcut) goes wrong wherever the
//! slope is small or uneven, which shows on glass as seams and terraces. So each voxel walks to the
//! surface instead: repeated Newton steps down the gradient until raw = 0, and the distance walked is
//! the distance stored. The preset's `clip` is a true distance that the result is cut to; its `shell`,
//! when above 0, makes a wall of that half-thickness about the surface instead of filling one side.
//!
//! Every preset's module looks the same from outside, so whatever draws with it need not know the preset:
//!
//! ```wgsl
//! struct Pr { /* eight f32: the preset's parameters in order, then _p<i> for the places not used */ }
//! struct U { hb: f32, res: f32, puff: f32, pad: f32, pr: Pr }       // 48 bytes, pr at byte 16
//! @group(0) @binding(0) var<uniform> u: U;                          // read by both stages
//! @vertex   fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) slice: u32) -> VOut
//! @fragment fn fs(in: VOut) -> @location(0) vec4f                   // (distance, 0, 0, 1)
//! ```
//!
//! One slice of the volume is drawn per render pass, as `draw(0..3, z..z+1)` into layer z with a viewport
//! of res x res: the three vertices make one triangle that covers the slice, and the slice's number
//! arrives as the instance index. Texel (i, j) of slice z is the point
//! ((i + 0.5, j + 0.5, z + 0.5) / res * 2 - 1) * hb, with no flip of any axis.
//!
//! `bake_shader_compute` gives the same module with one more entry point, for devices on which a compute
//! shader can write a 16-bit float volume: the whole volume in a single dispatch instead of a pass per slice.
//!
//! ```wgsl
//! @group(0) @binding(1) var outVol: texture_storage_3d<r16float, write>;
//! @compute @workgroup_size(8, 8, 4) fn cs(@builtin(global_invocation_id) id: vec3u)   // dispatch res/8, res/8, res/4
//! ```

use crate::presets::{Preset, Shape};

/// Bytes of the uniform block: hb, res, puff, a spare, then eight parameters, all f32.
pub const UNIFORM_SIZE: usize = 48;
/// The most sliders a field preset can have.
pub const MAX_PARAMS: usize = 8;
/// What a preset that gives no `clip` is cut to: a ball.
pub const CLIP_DEFAULT: &str = "length(p) - 1.38";
/// The `shell` of a preset that fills one side of its surface.
pub const SHELL_DEFAULT: &str = "0.0";

/// Words WGSL keeps for itself (its keywords and its reserved words). None of them can name anything.
const WGSL_RESERVED: &[&str] = &[
    // keywords
    "alias", "break", "case", "const", "const_assert", "continue", "continuing", "default", "diagnostic",
    "discard", "else", "enable", "false", "fn", "for", "if", "let", "loop", "override", "requires", "return",
    "struct", "switch", "true", "var", "while",
    // reserved
    "NULL", "Self", "abstract", "active", "alignas", "alignof", "as", "asm", "asm_fragment", "async",
    "attribute", "auto", "await", "become", "cast", "catch", "class", "co_await", "co_return", "co_yield",
    "coherent", "column_major", "common", "compile", "compile_fragment", "concept", "const_cast", "consteval",
    "constexpr", "constinit", "crate", "debugger", "decltype", "delete", "demote", "demote_to_helper", "do",
    "dynamic_cast", "enum", "explicit", "export", "extends", "extern", "external", "fallthrough", "filter",
    "final", "finally", "friend", "from", "fxgroup", "get", "goto", "groupshared", "highp", "impl",
    "implements", "import", "inline", "instanceof", "interface", "layout", "lowp", "macro", "macro_rules",
    "match", "mediump", "meta", "mod", "module", "move", "mut", "mutable", "namespace", "new", "nil",
    "noexcept", "noinline", "nointerpolation", "non_coherent", "noncoherent", "noperspective", "null",
    "nullptr", "of", "operator", "package", "packoffset", "partition", "pass", "patch", "pixelfragment",
    "precise", "precision", "premerge", "priv", "protected", "pub", "public", "readonly", "ref", "regardless",
    "register", "reinterpret_cast", "require", "resource", "restrict", "self", "set", "shared", "sizeof",
    "smooth", "snorm", "static", "static_assert", "static_cast", "std", "subroutine", "super", "target",
    "template", "this", "thread_local", "throw", "trait", "try", "type", "typedef", "typeid", "typename",
    "typeof", "union", "unless", "unorm", "unsafe", "unsized", "use", "using", "varying", "virtual",
    "volatile", "wgsl", "where", "with", "writeonly", "yield",
];

/// The words WGSL keeps for itself, for anything that needs to steer clear of them.
pub fn wgsl_reserved() -> &'static [&'static str] { WGSL_RESERVED }

/// The name a parameter goes by inside the shader, where a preset reads it as `pr.<name>`.
///
/// That is the parameter's own id whenever WGSL can take it: a letter followed by letters, digits and
/// underscores, and not one of the language's own words. Any other id (a reserved word such as `filter`
/// or `mod`, one that starts with an underscore or a digit, one with other characters in it) keeps its
/// id everywhere else, in saved settings and in links, and inside the shader alone is called `v_<id>`
/// with every character that is not a letter, a digit or an underscore turned into `_`.
pub fn member_name(id: &str) -> String {
    let mut chars = id.chars();
    let plain = chars.next().is_some_and(|c| c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !WGSL_RESERVED.contains(&id);
    if plain { return id.to_string(); }
    let safe: String = id.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect();
    format!("v_{safe}")
}

// The helpers every preset may use, as in FIELD_HEAD.
pub(crate) const HELPERS: &str = "\
fn smin(a: f32, b: f32, k: f32) -> f32 { let h = max(k - abs(a - b), 0.0) / k; return min(a, b) - h*h*k*0.25; }
fn smax(a: f32, b: f32, k: f32) -> f32 { return -smin(-a, -b, k); }
fn sdBall(p: vec3f, r: f32) -> f32 { return length(p) - r; }
fn sdBox(p: vec3f, b: vec3f, r: f32) -> f32 { let q = abs(p) - b + r; return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r; }
";

// One triangle that covers the slice. The slice's depth in the volume, 0..1, goes to every pixel unchanged.
const VERTEX: &str = "\
struct VOut { @builtin(position) pos: vec4f, @location(0) @interpolate(flat) zc: f32 }

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) slice: u32) -> VOut {
  let c = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  var o: VOut;
  o.pos = vec4f(c * 2.0 - 1.0, 0.0, 1.0);
  o.zc = (f32(slice) + 0.5) / u.res;
  return o;
}
";

// The walk to the surface, in three pieces: the preset's shell goes between the first two and its clip
// between the last two. Both are worked out at p, the voxel itself, not at x, where the walk ended.
const WALK: &str = "
// the distance stored for the voxel at pixel `xy` of the slice at depth `zc` (0..1)
fn walk(xy: vec2f, zc: f32) -> vec4f {
  let p = (vec3f(xy / u.res, zc) * 2.0 - 1.0) * u.hb;
  const e: f32 = 0.0007;
  const k = vec2f(1.0, -1.0);
  var x = p;
  var v0 = 0.0;
  var last = 1.0;
  for (var i = 0; i < 8; i++) {
    let a = raw(x + e*k.xyy);
    let b = raw(x + e*k.yyx);
    let c = raw(x + e*k.yxy);
    let d = raw(x + e*k.xxx);
    let fv = 0.25 * (a + b + c + d);
    let g = (k.xyy*a + k.yyx*b + k.yxy*c + k.xxx*d) / (4.0 * e);
    if (i == 0) { v0 = fv; }
    var st = g * (fv / max(dot(g, g), 1e-12));
    last = length(st);
    if (last > 0.12) { st *= 0.12 / last; }                // never leap: far from the surface the gradient misleads
    x -= st;
    if (last < 2e-5) { break; }
  }
  var dist = length(x - p);
  dist *= mix(1.0, 0.7, smoothstep(0.06, 0.2, dist));     // a long walk may not have ended at the nearest point: stay cautious
  var sd = select(dist, -dist, v0 < 0.0) - u.puff;        // puff swells (or shrinks) the whole surface evenly
  let w = (";
const WALK_SHELL_TO_CLIP: &str = ");
  if (w > 0.0) { sd = abs(sd) - w; }
  return vec4f(clamp(smax(sd, (";
const WALK_END: &str = "), 0.03), -1.0, 1.0), 0.0, 0.0, 1.0);
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  pr = u.pr;
  return walk(in.pos.xy, in.zc);
}
";

// The same walk for every voxel of the volume at once, written straight into it.
const COMPUTE: &str = "
@group(0) @binding(1) var outVol: texture_storage_3d<r16float, write>;

@compute @workgroup_size(8, 8, 4)
fn cs(@builtin(global_invocation_id) id: vec3u) {
  pr = u.pr;
  textureStore(outVol, id, walk(vec2f(id.xy) + 0.5, (f32(id.z) + 0.5) / u.res));
}
";

/// The walk for a cut stone (gems.rs, whose `raw` is the largest of the stone's facet distances, and of its
/// rim's): a voxel far from the surface is not walked. Inside, the walk would go straight to the nearest
/// facet, so its distance is the field's own value there; outside, the field's value is never more than
/// the true distance, and such a voxel is only ever stepped across by the tracer, never read for where
/// the surface is or which way it faces (that takes the voxels within three of it). Nine tenths of the
/// volume or more is far, and a stone bakes several times faster: it is rebuilt every frame while it
/// animates.
fn gem_walk() -> String {
    const LOOP: &str = "  for (var i = 0; i < 8; i++) {";
    const FAR: &str = "  // far from the surface (more than 0.1, and more than five voxels): no walk
  let vf = raw(p);
  let far = abs(vf) > max(0.1, 10.0 * u.hb / u.res);
  if (far) { v0 = vf; x = p + vec3f(abs(vf), 0.0, 0.0); }
  for (var i = 0; i < 8 && !far; i++) {";
    assert_eq!(WALK.matches(LOOP).count(), 1, "the walk's loop is not where the stone's shortcut goes");
    WALK.replace(LOOP, FAR)
}

/// A line a preset's text may hold to say that its `raw` is a distance inside and never more than one
/// outside, as a cut stone's is: far voxels are then not walked (`gem_walk`).
pub const BOUND_MARK: &str = "// raw: exact inside, never more than the distance outside";

/// What a preset puts in its text to say that its `raw` is a distance already (exact, or never more than
/// one): see `distance_walk`.
pub const DISTANCE_MARK: &str = "// raw: a distance";

/// The walk for a preset whose `raw` is a distance already: none at all, the value read as it is. This is
/// what a wall about a surface with no sides needs (topology.rs: the Klein bottle, Costa, Enneper...): its
/// `raw` is a distance that is never negative, with a crease along the surface itself, and a voxel whose
/// four readings straddle the crease finds no slope there, takes the longest step the walk allows and
/// stores a distance far too large; on an axis-aligned stretch of surface whole rows of voxels do, and the
/// wall shows slots.
fn distance_walk() -> String {
    const LOOP: &str = "  for (var i = 0; i < 8; i++) {";
    const READ: &str = "  // a distance already: read, not walked
  v0 = raw(p);
  x = p + vec3f(abs(v0), 0.0, 0.0);
  for (var i = 0; i < 0; i++) {";
    assert_eq!(WALK.matches(LOOP).count(), 1, "the walk's loop is not where the reading goes");
    WALK.replace(LOOP, READ)
}

/// `bake_shader` with one more entry point, `cs`, which works out the whole volume in one dispatch and
/// writes it straight into a 16-bit float 3D texture.
pub fn bake_shader_compute(p: &Preset) -> String {
    bake_shader(p) + COMPUTE
}

/// The complete WGSL module that bakes this field preset into one slice of the distance volume.
pub fn bake_shader(p: &Preset) -> String {
    let Shape::Field { wgsl, clip, shell } = &p.shape else { panic!("{}: not an equation, so there is nothing to bake this way", p.id) };
    assert!(p.params.len() <= MAX_PARAMS, "{}: a field preset can have at most {MAX_PARAMS} sliders", p.id);
    let mut members: Vec<String> = p.params.iter().map(|d| member_name(d.id)).collect();
    for (i, m) in members.iter().enumerate() {
        assert!(!members[..i].contains(m), "{}: two sliders would both be called pr.{m} in the shader", p.id);
    }
    for i in members.len()..MAX_PARAMS { members.push(format!("_p{i}")); }
    let pr: Vec<String> = members.iter().map(|m| format!("{m}: f32")).collect();
    format!(
        "// {name}: one slice of its distance volume.\n\
         struct Pr {{ {pr} }}\n\
         struct U {{ hb: f32, res: f32, puff: f32, pad: f32, pr: Pr }}\n\
         @group(0) @binding(0) var<uniform> u: U;\n\
         var<private> pr: Pr;\n\n\
         {HELPERS}\n{wgsl}\n\n{VERTEX}{walk}{shell}{WALK_SHELL_TO_CLIP}{clip}{WALK_END}",
        name = p.name, pr = pr.join(", "),
        walk = if wgsl.contains("fn gem_fold(") || wgsl.contains(BOUND_MARK) { gem_walk() } else if wgsl.contains(DISTANCE_MARK) { distance_walk() } else { WALK.to_string() },
    )
}

/// The uniform block's contents for these values: hb, res, puff, a spare, then the parameters in order.
/// A value that is missing is taken to be the slider's starting value.
pub fn uniforms(p: &Preset, values: &[f64], res: u32, puff: f64) -> [f32; 12] {
    let mut u = [0.0f32; 12];
    u[0] = crate::HB as f32;
    u[1] = res as f32;
    u[2] = puff as f32;
    for (i, d) in p.params.iter().enumerate().take(MAX_PARAMS) {
        u[4 + i] = values.get(i).copied().unwrap_or(d.value) as f32;
    }
    u
}
