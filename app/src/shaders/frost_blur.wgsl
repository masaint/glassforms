// Rough glass: what is seen through it, blurred after tracing.
//
// The tracer follows one path for each ray and softens everything that depends on direction (see
// frost.wgsl), but a single path still lands on one point of whatever lies behind the surface, so the
// outlines of what is seen through rough glass come out sharp. A rough surface spreads the light that
// leaves it over a cone, and what the eye sees through it is everything the cone covers: the picture
// behind, blurred by the cone's width where it lands. This stage does that blur on the traced frame,
// before it is folded into the picture so far: each pixel of glass is averaged with its neighbours over
// the radius the tracer worked out for it (the cone's radius at the first thing it met), and only with
// neighbours on the same piece of surface that face much the same way, so that the glass's own
// outlines and creases stay sharp while what shows through it goes soft. What the surface itself
// reflects is not seen through it: the tracer writes that beside the frame, and it is taken off before
// the blur and put back after. And what is seen through the surface is the light behind it times the
// share the surface lets through, which belongs to the surface: it is the light behind that is blurred,
// so that a surface seen at a glancing angle (which lets little through) stays as dim as it is.
//
// It is done with smaller copies of the frame, each half the size of the one before and that much
// softer (fs_down), from which every pixel reads the two that are nearest the softness it needs (fs_out).
// Nothing in it is random, and nothing is kept from frame to frame.

struct FU {
  res: vec2f,         // size of the traced frame
  px: f32,            // one pixel as an angle: 2 tan(fov / 2) / height
  skirt: f32,         // the share of the light that is spread SKIRT times as far (the facets' skirt)
  levels: f32,        // how many smaller copies there are
  show: f32,          // 1: draw the blur's radius instead of the picture
  allocation: vec2f,      // full allocation containing the traced region
}
struct LU {
  src: vec2f,         // size of the copy this one is made from
  k: f32,             // this copy is 1 / 2^k of the frame
  pad: f32,
}

@group(0) @binding(0) var<uniform> fu: FU;
@group(0) @binding(1) var<uniform> lu: LU;
@group(0) @binding(2) var srcTex: texture_2d<f32>;     // fs_down: the copy one size up (rgb, a = distance to the glass, 0 = none)
@group(0) @binding(6) var srcFace: texture_2d<f32>;    // fs_down: which way its surface faces (x, y of the normal); z: for a copy the blur's radius, for the frame the share of light the surface reflects
@group(0) @binding(8) var srcTop: texture_2d<f32>;     // fs_down, for the first copy: what the surface itself reflects (rgb), to be taken off; the radius (a)
@group(0) @binding(3) var picTex: texture_2d<f32>;     // fs_out: the traced frame
@group(0) @binding(4) var auxTex: texture_2d<f32>;     // fs_out: rgb = what the surface itself reflects; a = the blur's radius, in the shape's own units
@group(0) @binding(9) var faceTex: texture_2d<f32>;    // fs_out: which way the surface faces (x, y of the normal); z = the share of light it reflects
@group(0) @binding(5) var pyrTex: texture_2d<f32>;     // fs_out: all the smaller copies, as levels
@group(0) @binding(7) var pyrFace: texture_2d<f32>;    // fs_out: ... and which way each faces (x, y), with its radius (z)
@group(0) @binding(10) var smp: sampler;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

const SKIRT: f32 = 3.46;
// The steepest a surface is taken to slope away from the eye, as distance per unit across it. Steeper
// than this between neighbours is the edge of a piece, not its slope.
const STEEP: f32 = 4.0;
// How fast a neighbour counts for less as its surface faces another way: at 30 degrees apart it
// counts for a little under half.
const TURN: f32 = 6.0;
// The least share a surface is taken to let through, when the light behind it is worked back from what
// is seen through it (at a glancing angle the share is next to nothing, and so is what is known).
const LEAST: f32 = 0.25;

fn tent(i: i32) -> f32 { return select(1.0, 3.0, i == 0 || i == 1); }
// the whole normal from its x and y (it faces the eye)
fn whole(n: vec2f) -> vec3f { return vec3f(n, sqrt(max(1.0 - dot(n, n), 0.0))); }
fn faceAt(t: vec2i) -> vec2f { return textureLoad(srcFace, t, 0).xy; }

struct Down { @location(0) col: vec4f, @location(1) face: vec4f }

// A copy half the size: each texel is the average of the sixteen around it one size up (weighed 1, 3,
// 3, 1 each way, which makes each copy a bell twice as wide as the last), taking only those that lie on
// the same piece of surface as the nearest of the four in its middle and face much the same way. Which
// lie on it is judged against the slope of that piece, so that a surface tilted away from the eye is
// still one surface.
@fragment
fn fs_down(@builtin(position) pos: vec4f) -> Down {
  let o = vec2i(pos.xy) * 2;
  let hi = vec2i(min(lu.src, ceil(fu.res / exp2(lu.k - 1.0)))) - 1;
  // the nearest of the four in the middle
  var zf = 0.0;
  var at = vec2i(0);
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      let z = textureLoad(srcTex, clamp(o + vec2i(i, j), vec2i(0), hi), 0).a;
      if (z > 0.0 && (zf == 0.0 || z < zf)) { zf = z; at = vec2i(i, j); }
    }
  }
  if (zf == 0.0) { return Down(vec4f(0.0), vec4f(0.0)); }
  // how the distance slopes across the piece, per texel one size up: from the nearest texel to its
  // neighbours either side, whichever side is the gentler (the other may be across an edge)
  let one = zf * fu.px * exp2(lu.k - 1.0);       // a texel one size up, in the shape's units, at that distance
  let c = o + at;
  let nf = whole(faceAt(clamp(c, vec2i(0), hi)));
  var g = vec2f(0.0);
  {
    let a = textureLoad(srcTex, clamp(c + vec2i(1, 0), vec2i(0), hi), 0).a;
    let b = textureLoad(srcTex, clamp(c - vec2i(1, 0), vec2i(0), hi), 0).a;
    let da = select(1e9, a - zf, a > 0.0);
    let db = select(1e9, zf - b, b > 0.0);
    let d = select(db, da, abs(da) < abs(db));
    g.x = select(0.0, d, abs(d) < STEEP * one);
  }
  {
    let a = textureLoad(srcTex, clamp(c + vec2i(0, 1), vec2i(0), hi), 0).a;
    let b = textureLoad(srcTex, clamp(c - vec2i(0, 1), vec2i(0), hi), 0).a;
    let da = select(1e9, a - zf, a > 0.0);
    let db = select(1e9, zf - b, b > 0.0);
    let d = select(db, da, abs(da) < abs(db));
    g.y = select(0.0, d, abs(d) < STEEP * one);
  }
  let tol = one + 0.5 * (abs(g.x) + abs(g.y)) + 0.002;
  var sum = vec4f(0.0);
  var face = vec3f(0.0);
  var rad = 0.0;
  var ws = 0.0;
  for (var j = -1; j < 3; j++) {
    for (var i = -1; i < 3; i++) {
      let t = clamp(o + vec2i(i, j), vec2i(0), hi);
      var s = textureLoad(srcTex, t, 0);
      let fc = textureLoad(srcFace, t, 0);
      var r = fc.z;
      // (the first copy is made from the frame itself: less what the surface reflects)
      if (lu.k < 1.5) {
        let top = textureLoad(srcTop, t, 0);
        s = vec4f(max(s.rgb - top.rgb, vec3f(0.0)) / max(1.0 - fc.z, LEAST), s.a);
        r = top.a;
      }
      let n = whole(fc.xy);
      let off = vec2f(vec2i(i, j) - at);
      let d = (s.a - (zf + dot(g, off))) / tol;
      let w = tent(i) * tent(j) * select(0.0, exp(-d * d - TURN * (1.0 - dot(n, nf))), s.a > 0.0);
      // (each one's distance is carried along the slope to the middle of this texel)
      sum += w * vec4f(s.rgb, s.a - dot(g, vec2f(f32(i) - 0.5, f32(j) - 0.5)));
      face += w * n;
      rad += w * r;
      ws += w;
    }
  }
  return Down(sum / max(ws, 1e-6), vec4f(normalize(face).xy, rad / max(ws, 1e-6), 0.0));
}

// What copy k (1 = half size) shows at the pixel `at`, of the surface that is z away there, slopes by
// g a pixel and faces n: the four texels around it, weighed as usual by how near they are, and by
// whether they lie on that surface and face that way. rgb is not yet divided by the weight (a).
fn copy(at: vec2f, z: f32, g: vec2f, n: vec3f, k: i32) -> vec4f {
  let sc = exp2(f32(k));
  let size = min(vec2i(textureDimensions(pyrTex, k - 1)), vec2i(ceil(fu.res / sc)));
  let c = at / sc - 0.5;
  let i0 = floor(c);
  let f = c - i0;
  let tol = z * fu.px * sc + 0.5 * (abs(g.x) + abs(g.y)) * sc + 0.002;
  var acc = vec4f(0.0);
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      let t = clamp(vec2i(i0) + vec2i(i, j), vec2i(0), size - 1);
      let s = textureLoad(pyrTex, t, k - 1);
      let m = whole(textureLoad(pyrFace, t, k - 1).xy);
      let off = (vec2f(t) + 0.5) * sc - at;
      let d = (s.a - (z + dot(g, off))) / tol;
      let w = select(1.0 - f.x, f.x, i == 1) * select(1.0 - f.y, f.y, j == 1) * select(0.0, exp(-d * d - TURN * (1.0 - dot(m, n))), s.a > 0.0);
      acc += vec4f(s.rgb, 1.0) * w;
    }
  }
  return acc;
}

// The pixel's colour blurred by a bell of standard deviation sd pixels: from the two copies either side
// of that softness (copy k is as soft as 0.65 x 2^k pixels, counting what reading it back adds), or the
// pixel itself where none is needed.
fn blurred(own: vec3f, at: vec2f, z: f32, g: vec2f, n: vec3f, sd: f32) -> vec3f {
  let l = clamp(log2(max(sd, 1e-3) / 0.65), 0.0, fu.levels);
  let k = i32(floor(l));
  let t = l - f32(k);
  var a = vec4f(own, 1.0) * (1.0 - t);
  if (k > 0) { a = copy(at, z, g, n, k) * (1.0 - t); }
  if (t > 0.0 && k < i32(fu.levels)) { a += copy(at, z, g, n, k + 1) * t; } else { a = a / max(1.0 - t, 1e-6); }
  // (where no texel of a copy lies on this surface, the pixel keeps its own colour for that share)
  let w = clamp(a.w, 0.0, 1.0);
  return a.rgb / max(a.w, 1e-6) * w + own * (1.0 - w);
}

@fragment
fn fs_out(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = vec2i(pos.xy);
  let hi = vec2i(fu.res) - 1;
  let own = textureLoad(picTex, p, 0);
  let z = own.a;
  let aux = textureLoad(auxTex, p, 0);
  if (z <= 0.0 || aux.a <= 0.0) { return own; }
  // The radius: the pixel's own, or what the pixels around it on the same surface have between them if
  // that is more (a blur beside this pixel reaches into it: read from the copies whose texels are about
  // as wide as the blur they report).
  var r = aux.a;
  for (var k = 1; k <= min(i32(fu.levels), 5); k++) {
    let dims = vec2f(textureDimensions(pyrFace, k - 1));
    let last = min(dims, ceil(fu.res / exp2(f32(k)))) - 0.5;
    let uv = min(pos.xy / fu.allocation, last / dims);
    let rk = textureSampleLevel(pyrFace, smp, uv, f32(k - 1)).z;
    r = max(r, rk * smoothstep(0.35, 0.7, rk / (z * fu.px * exp2(f32(k)))));
  }
  // what is seen through the surface: all the light, less what the surface itself reflects; and the
  // light behind it, which is that over the share the surface lets through
  let top = min(aux.rgb, own.rgb);
  let face = textureLoad(faceTex, p, 0);
  let lets = max(1.0 - face.z, LEAST);
  let through = (own.rgb - top) / lets;
  // the radius in pixels: the cone's width where it lands, as the eye sees it at the surface's distance
  let sd = r / (z * fu.px);
  if (fu.show > 0.5) { return vec4f(vec3f(sd / 32.0, r * 4.0, 0.0), z); }
  if (sd < 0.15) { return own; }
  // how the surface's distance slopes here, a pixel: the gentler of the two sides each way
  let one = z * fu.px;
  var g = vec2f(0.0);
  {
    let a = textureLoad(picTex, clamp(p + vec2i(1, 0), vec2i(0), hi), 0).a;
    let b = textureLoad(picTex, clamp(p - vec2i(1, 0), vec2i(0), hi), 0).a;
    let da = select(1e9, a - z, a > 0.0);
    let db = select(1e9, z - b, b > 0.0);
    let d = select(db, da, abs(da) < abs(db));
    g.x = select(0.0, d, abs(d) < STEEP * one);
  }
  {
    let a = textureLoad(picTex, clamp(p + vec2i(0, 1), vec2i(0), hi), 0).a;
    let b = textureLoad(picTex, clamp(p - vec2i(0, 1), vec2i(0), hi), 0).a;
    let da = select(1e9, a - z, a > 0.0);
    let db = select(1e9, z - b, b > 0.0);
    let d = select(db, da, abs(da) < abs(db));
    g.y = select(0.0, d, abs(d) < STEEP * one);
  }
  let n = whole(face.xy);
  let at = pos.xy;
  let col = mix(blurred(through, at, z, g, n, sd), blurred(through, at, z, g, n, SKIRT * sd), fu.skirt);
  return vec4f(top + col * lets, z);
}
