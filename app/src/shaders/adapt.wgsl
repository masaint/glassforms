// Adaptive refining (adapt.rs). While a still picture refines, each pixel keeps its own count of samples
// and, beside its mean, a second mean of every other sample only (the odd-numbered ones). How far apart
// the two are, as a step of the screen's brightness, is how far the mean may still be off. (Not the
// samples' variance: the tracer's samples are not independent but placed in well-spread sequences, the
// spot in the pixel and the spectrum both, and their mean settles much faster than the variance says.
// Two means of one sequence settle alike, and their difference measures that.) Pixels whose mean is
// certain are left out of the next pass of the tracer: a pass marks them in a stencil, which the tracer's
// pipeline tests before running for a pixel, so that a left-out pixel costs nothing and the pixels still
// traced are packed together.
//
//   fold:  the still picture's running mean, done here in place of accum.wgsl's still branch, with each
//          pixel's own count: a pixel the tracer left out (its frame still holds the mark the pass was
//          cleared to, alpha < 0) keeps its mean, count and second mean as they were.
//   mask:  writes the stencil: a 2 x 2 block is left out only when it and the ring of pixels around it are
//          all certain (so no left-out pixel borders on one still noisy, and whole quads drop out), and
//          not when it is its turn to be traced anyway (every block is traced once every `recheck`
//          passes, so that a pixel that happened to look certain before a rare bright path turned up is
//          found out). A pass that only fills what is left of a frame's budget is for a share of the
//          blocks, chosen afresh each pass (each pixel keeps its own count, so this is no harm to the mean).

struct AU {
  res: vec2f,
  tol: f32,         // a mean is certain when its uncertainty is below this, as a step of the screen (0..1)
  nmin: f32,        // and it has at least this many samples
  expo: f32,        // the exposure the screen is drawn with
  base: f32,        // >= 0: the counts start afresh, every pixel's from this many; < 0: carry on
  turn: f32,        // passes since the picture started (for whose turn it is to be traced anyway)
  recheck: f32,
  share: f32,       // the share of the blocks this pass is for (1: all; less: a pass that fills what is
                    // left of the frame's budget, for blocks chosen afresh each time)
  pad0: f32,
  pad1: f32,
  pad2: f32,
}

@group(0) @binding(0) var<uniform> a: AU;

// ---------------------------------------------------------------------------------------------- fold
@group(0) @binding(1) var prevTex: texture_2d<f32>;
@group(0) @binding(2) var curTex: texture_2d<f32>;
@group(0) @binding(3) var outTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var<storage, read_write> stats: array<vec4f>;     // the mean of the odd samples, count
@group(0) @binding(5) var<storage, read_write> err: array<f32>;

// the screen's brightness for a light (a soft roll-off like the tone curve's, then sRGB's encoding)
fn screen(x: vec3f) -> vec3f {
  let y = max(x, vec3f(0.0)) / (1.0 + max(x, vec3f(0.0)));
  return select(1.055 * pow(y, vec3f(1.0 / 2.4)) - 0.055, 12.92 * y, y < vec3f(0.0031308));
}

@compute @workgroup_size(8, 8)
fn fold(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2u(a.res);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let c = vec2i(id.xy);
  let i = id.y * size.x + id.x;
  let prev = textureLoad(prevTex, c, 0);
  let cur = textureLoad(curTex, c, 0);
  var s = stats[i];
  // (starting afresh from a picture that stands for some frames already: it is the first odd sample)
  if (a.base >= 0.0) { s = vec4f(prev.rgb, a.base); }
  if (cur.a < 0.0) {
    // left out this pass: as it was
    textureStore(outTex, c, prev);
    if (a.base >= 0.0) { stats[i] = s; }
    return;
  }
  let n = s.w + 1.0;
  var mean = cur.rgb;
  var odd = s.rgb;
  if (s.w > 0.5) { mean = mix(prev.rgb, cur.rgb, 1.0 / n); }
  // this is sample n: the odd ones go into the second mean too, the ((n + 1) / 2)th of them
  let k = floor(n * 0.5 + 0.5);
  if (n - 2.0 * floor(n * 0.5) > 0.5) { odd = select(mix(s.rgb, cur.rgb, 1.0 / k), cur.rgb, k < 1.5); }
  // how far the mean may yet be off, as a step of the screen: the two means' difference, in the colour
  // that differs most
  var e = 1e9;
  if (n >= a.nmin) {
    let d = abs(screen(mean * a.expo) - screen(odd * a.expo));
    e = max(d.x, max(d.y, d.z));
  }
  // (a value that is not a number, which a rare path can bring, never counts as certain)
  if (!(e <= 1e9)) { e = 1e9; }
  err[i] = e;
  stats[i] = vec4f(odd, n);
  textureStore(outTex, c, vec4f(mean, cur.a));
}

// ---------------------------------------------------------------------------------------------- mask
@group(0) @binding(6) var<storage, read> errIn: array<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn mask(@builtin(position) pos: vec4f) {
  let size = vec2i(a.res);
  let q = vec2i(pos.xy) / 2;
  // this block's turn to be traced anyway: every `recheck` passes, the blocks in a scattered order
  var h = u32(q.x) * 0x8da6b343u ^ u32(q.y) * 0xd8163841u;
  h = (h ^ (h >> 15u)) * 0x2c1b3c6du;
  h = h ^ (h >> 12u);
  // (a pass for a share of the blocks: the others are left out of it, a different choice each pass)
  if (a.share < 1.0) {
    var g = h + u32(a.turn) * 0x9e3779b9u;
    g = (g ^ (g >> 16u)) * 0x7feb352du;
    g = g ^ (g >> 15u);
    if (f32(g >> 8u) / 16777216.0 >= a.share) { return; }
  }
  let r = u32(a.recheck);
  if (r > 0u && (h + u32(a.turn)) % r == 0u) { discard; }
  for (var y = -1; y <= 2; y++) {
    for (var x = -1; x <= 2; x++) {
      let p = clamp(q * 2 + vec2i(x, y), vec2i(0), size - 1);
      if (errIn[u32(p.y * size.x + p.x)] > a.tol) { discard; }
    }
  }
  // certain: marked, and left out of the tracer's pass
}
