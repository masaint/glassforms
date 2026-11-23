// The surface studio, first part: the stack of layers evaluated at every voxel of a shape's distance
// volume (see core/src/sculpt.rs for what the layers are and how they combine). It writes a field whose
// zero is the new surface; sculpt_fix.wgsl then makes that a true distance again.
//
// Everything is worked out in float from the pattern's own formula: cell distances are exact (every cell
// that could hold a nearer point is looked at), so no cut wall is terraced by stored, rounded distances.

struct U {
  head: array<vec4f, 4>,
  l: array<vec4f, 64>,     // layer i: l[i * 8 + 0..5]
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var src: texture_3d<f32>;
@group(0) @binding(2) var smp: sampler;
@group(0) @binding(3) var imgs: texture_2d_array<f32>;
@group(0) @binding(4) var ismp: sampler;
@group(0) @binding(5) var dst: texture_storage_3d<r16float, write>;

const TAU: f32 = 6.2831853;

// ---- hashing and noise --------------------------------------------------------------------------

fn pcg3(v0: vec3u) -> vec3u {
  var v = v0 * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v = v ^ (v >> vec3u(16u));
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v;
}
fn hash3(c: vec3i, seed: u32) -> vec3f {
  let h = pcg3(bitcast<vec3u>(c) + vec3u(seed * 747796405u, seed * 2891336453u + 17u, seed * 277803737u + 101u));
  return vec3f(h >> vec3u(8u)) * (1.0 / 16777216.0);
}
fn hash1(c: vec3i, seed: u32) -> f32 {
  return hash3(c, seed ^ 0x9e3779b9u).x;
}

// gradient noise, about -1..1, smooth to the second derivative
fn gnoise(p: vec3f, seed: u32) -> f32 {
  let i = vec3i(floor(p));
  let f = p - floor(p);
  let w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  var n: array<f32, 8>;
  for (var k = 0; k < 8; k++) {
    let o = vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    let g = normalize(hash3(i + o, seed) * 2.0 - 1.0 + vec3f(1e-4));
    n[k] = dot(g, f - vec3f(o));
  }
  let x0 = mix(n[0], n[1], w.x); let x1 = mix(n[2], n[3], w.x);
  let x2 = mix(n[4], n[5], w.x); let x3 = mix(n[6], n[7], w.x);
  return 1.6 * mix(mix(x0, x1, w.y), mix(x2, x3, w.y), w.z);
}

// a turn between octaves, so that their grids do not line up
const ROT = mat3x3f(vec3f(0.00, 0.80, 0.60), vec3f(-0.80, 0.36, -0.48), vec3f(-0.60, -0.48, 0.64));

// The pixel pre-filter: one pixel's width at the shape, in the pattern's own units (0: off). Detail
// smaller than about two pixels is faded out towards its average before it reaches the surface: what
// cannot be seen is not made, so it cannot turn into speckle. What is larger stays exactly as it was.
var<private> fpq: f32 = 0.0;
fn keepSize(size: f32) -> f32 {
  if (fpq <= 0.0) { return 1.0; }
  return smoothstep(1.0, 2.0, size / fpq);
}

// fractal noise: 0 plain, 1 ridged, 2 billows. Plain is about -1..1, the others 0..1.
fn fbm(p0: vec3f, oct: i32, rough: f32, seed: u32, how: i32) -> f32 {
  var p = p0;
  var amp = 1.0;
  var sum = 0.0;
  var norm = 0.0;
  for (var o = 0; o < oct; o++) {
    let n = gnoise(p, seed + u32(o) * 131u);
    var v = n;
    var mean = 0.0;
    if (how == 1) { let r = 1.0 - abs(n); v = r * r; mean = 0.42; }
    if (how == 2) { v = abs(n); mean = 0.4; }
    // (an octave finer than the pre-filter allows is replaced by its average)
    let wavelength = pow(1.0 / 2.03, f32(o));
    v = mix(mean, v, keepSize(wavelength));
    sum += amp * v;
    norm += amp;
    amp *= rough;
    p = ROT * p * 2.03 + vec3f(7.1, -3.3, 1.7);
  }
  return sum / norm;
}

// ---- cells -------------------------------------------------------------------------------------

struct Cells {
  d1: f32,
  d2: f32,
  c1: vec3f,
  c2: vec3f,
  i1: vec3i,
  i2: vec3i,
};

// Holes carved into the skin are kept to points near the surface (so that every hole opens to the outside,
// and none is left as a void inside the glass): the pattern's space back to the scene's, and how near.
var<private> openR: f32 = 0.0;
var<private> openScale: f32 = 1.0;
var<private> openAxis: vec3f = vec3f(0.0, 1.0, 0.0);
var<private> openStretch: f32 = 1.0;
var<private> openSize: f32 = 1.0;

// the point of a cell (none if the cell is left empty: `keep` is the share of cells that have one)
fn cellPoint(c: vec3i, seed: u32, jit: f32, phase: f32, keep: f32) -> vec4f {
  let r = hash3(c, seed);
  var o = 0.5 + (r - 0.5) * (0.8 * jit);
  if (phase != 0.0) {
    // evolving: each point goes round a small loop of its own, and comes back to where it started
    let ph = phase + TAU * r.zxy;
    o += 0.1 * jit * vec3f(sin(ph.x), sin(ph.y), sin(ph.z));
  }
  var present = select(0.0, 1.0, keep >= 1.0 || hash1(c, seed) < keep);
  let q = (vec3f(c) + o) * openSize;
  if (openR > 0.0 && present > 0.0) {
    let w = (q + openAxis * dot(q, openAxis) * (openStretch - 1.0)) * openScale;
    if (abs(srcAt(w)) > openR) { present = 0.0; }
  }
  return vec4f(vec3f(c) + o, present);
}

// how near a point of this cell can possibly be to p (squared)
fn cellBound(p: vec3f, c: vec3i, jit: f32) -> f32 {
  let lo = vec3f(c) + 0.5 - 0.5 * jit;
  let hi = vec3f(c) + 0.5 + 0.5 * jit;
  let d = max(max(lo - p, p - hi), vec3f(0.0));
  return dot(d, d);
}

// the nearest two points, exactly: the 27 cells about p, then any of the 98 beyond that could hold a nearer one
fn cells(p: vec3f, seed: u32, jit: f32, phase: f32, keep: f32) -> Cells {
  let b = vec3i(floor(p));
  var r: Cells;
  r.d1 = 1e9; r.d2 = 1e9;
  for (var round = 0; round < 2; round++) {
    for (var z = -2; z <= 2; z++) {
      for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
          let o = vec3i(x, y, z);
          let inner = all(abs(o) <= vec3i(1));
          if (inner != (round == 0)) { continue; }
          let c = b + o;
          if (round == 1 && cellBound(p, c, jit) >= r.d2) { continue; }
          let q = cellPoint(c, seed, jit, phase, keep);
          if (q.w == 0.0) { continue; }
          let d = dot(q.xyz - p, q.xyz - p);
          if (d < r.d1) {
            r.d2 = r.d1; r.c2 = r.c1; r.i2 = r.i1;
            r.d1 = d; r.c1 = q.xyz; r.i1 = c;
          } else if (d < r.d2) {
            r.d2 = d; r.c2 = q.xyz; r.i2 = c;
          }
        }
      }
    }
  }
  r.d1 = sqrt(r.d1);
  r.d2 = sqrt(r.d2);
  return r;
}

// the exact distance from p to the edge of its cell (the plane half way to every other point)
fn cellEdge(p: vec3f, r: Cells, seed: u32, jit: f32, phase: f32, keep: f32) -> f32 {
  let b = vec3i(floor(p));
  var e = 1e9;
  for (var z = -2; z <= 2; z++) {
    for (var y = -2; y <= 2; y++) {
      for (var x = -2; x <= 2; x++) {
        let c = b + vec3i(x, y, z);
        if (all(c == r.i1)) { continue; }
        // (a plane between p's point and this one is at least half their difference in distance away)
        let lb = sqrt(cellBound(p, c, jit));
        if (0.5 * (lb - r.d1) >= e) { continue; }
        let q = cellPoint(c, seed, jit, phase, keep);
        if (q.w == 0.0) { continue; }
        let n = q.xyz - r.c1;
        let len = length(n);
        if (len < 1e-6) { continue; }
        e = min(e, dot(0.5 * (r.c1 + q.xyz) - p, n / len));
      }
    }
  }
  return e;
}

// hammered: a round dish about the nearest point. The nearest is taken smoothly (a soft minimum of the
// distances to the points about p), so where two dishes meet the rim between them is rounded, not a knife edge.
fn dishes(p: vec3f, seed: u32, jit: f32, rad: f32, phase: f32) -> f32 {
  let b = vec3i(floor(p));
  let K = 10.0;
  let reach = 1.6 * rad;
  var sum = 0.0;
  for (var z = -2; z <= 2; z++) {
    for (var y = -2; y <= 2; y++) {
      for (var x = -2; x <= 2; x++) {
        let c = b + vec3i(x, y, z);
        if (cellBound(p, c, jit) >= reach * reach) { continue; }
        let q = cellPoint(c, seed, jit, phase, 1.0);
        sum += exp(-K * length(q.xyz - p) / rad);
      }
    }
  }
  if (sum <= 0.0) { return 0.0; }
  let d = max(-log(sum) / K, 0.0);
  let v = max(1.0 - d * d, 0.0);
  return v * v;
}

// ---- helpers -----------------------------------------------------------------------------------

fn smin(a: f32, b: f32, k: f32) -> f32 {
  let h = max(k - abs(a - b), 0.0) / k;
  return min(a, b) - h * h * k * 0.25;
}
fn smax(a: f32, b: f32, k: f32) -> f32 {
  return -smin(-a, -b, k);
}
// x where it is below -k, 0 where it is above 0, a smooth bend between (no offset where x is 0)
fn sneg(x: f32, k: f32) -> f32 {
  if (x <= -k) { return x + 0.5 * k; }
  if (x >= 0.0) { return 0.0; }
  return -x * x / (2.0 * k);
}
fn spos(x: f32, k: f32) -> f32 {
  return -sneg(-x, k);
}

fn basis(a: vec3f) -> mat3x3f {
  let t = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(a.x) > 0.8);
  let t1 = normalize(cross(a, t));
  let t2 = cross(a, t1);
  return mat3x3f(t1, t2, a);
}

fn L(i: i32, j: i32) -> vec4f {
  return u.l[i * 8 + j];
}

fn srcAt(p: vec3f) -> f32 {
  let hb = u.head[0].w;
  return textureSampleLevel(src, smp, p / (2.0 * hb) + 0.5, 0.0).r;
}

fn picture(uv: vec2f, img: i32) -> f32 {
  let c = textureSampleLevel(imgs, ismp, uv, img, 0.0).rgb;
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// How thick the body is about p (a point at distance f from the surface, h a voxel): its local radius,
// as far as `most`. Looking in from the surface along the normal, the depth below the surface rises
// with the distance in until the middle is passed: the deepest found is the radius. (Beyond `most`
// all that matters is that it is more.)
fn thickness(p: vec3f, f: f32, most: f32, h: f32) -> f32 {
  let g = vec3f(srcAt(p + vec3f(h, 0.0, 0.0)) - srcAt(p - vec3f(h, 0.0, 0.0)),
                srcAt(p + vec3f(0.0, h, 0.0)) - srcAt(p - vec3f(0.0, h, 0.0)),
                srcAt(p + vec3f(0.0, 0.0, h)) - srcAt(p - vec3f(0.0, 0.0, h)));
  let gl = length(g);
  let n = select(vec3f(0.0, 1.0, 0.0), g / gl, gl > 1e-6);
  let reach = clamp(2.0 * most, 2.0 * h, 0.6);
  var best = 0.0;
  for (var k = 1; k <= 10; k++) {
    let t = f32(k) / 10.0 * reach;
    best = max(best, -srcAt(p - n * (f + t)));
  }
  return best;
}

// ---- the stack ---------------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 4)
fn cs_eval(@builtin(global_invocation_id) gid: vec3u) {
  let res = u32(u.head[0].x);
  if (any(gid >= vec3u(res))) { return; }
  let h = u.head[0].y;
  let hb = u.head[0].w;
  let p0 = ((vec3f(gid) + 0.5) / f32(res) * 2.0 - 1.0) * hb;
  let f0 = srcAt(p0);
  // beyond what the layers can reach the field is only moved by the most they could move it: still a
  // safe distance, and nothing to work out
  let up = u.head[1].x;
  let down = u.head[1].y;
  let margin = u.head[1].z;
  if (f0 > up + margin) {
    textureStore(dst, gid, vec4f(f0 - up, 0.0, 0.0, 0.0));
    return;
  }
  if (f0 < -(down + margin)) {
    textureStore(dst, gid, vec4f(f0 + down, 0.0, 0.0, 0.0));
    return;
  }
  // the surface's direction here, for a picture seen from all sides
  var nrm = vec3f(0.0, 1.0, 0.0);
  if (u.head[2].x > 0.5) {
    let e = h;
    let g = vec3f(srcAt(p0 + vec3f(e, 0.0, 0.0)) - srcAt(p0 - vec3f(e, 0.0, 0.0)),
                  srcAt(p0 + vec3f(0.0, e, 0.0)) - srcAt(p0 - vec3f(0.0, e, 0.0)),
                  srcAt(p0 + vec3f(0.0, 0.0, e)) - srcAt(p0 - vec3f(0.0, 0.0, e)));
    let gl = length(g);
    if (gl > 1e-6) { nrm = g / gl; }
  }

  var f = f0;
  var p = p0;            // where the layers read their patterns: moved by every warp layer
  var prev = 1.0;        // the pattern of the layer above (for a mask that follows it)
  let n = i32(u.head[0].z);
  for (var i = 0; i < n; i++) {
    let k0 = L(i, 0);
    let k1 = L(i, 1);
    let k2 = L(i, 2);
    let k3 = L(i, 3);
    let k4 = L(i, 4);
    let kind = i32(k0.x);
    let mode = i32(k0.y);
    let blend = i32(k0.z);
    let maskKind = i32(k0.w);
    let opacity = k1.x;
    let scale = k1.y;
    let depth = k1.z;
    let soft = k1.w;
    let seed = u32(abs(k2.x) * 7919.0) + u32(i) * 1013u;
    let stretch = k2.y;
    let a = k2.z;
    let b = k2.w;
    let axis = k3.xyz;
    let phase = k3.w;

    // the mask: how much of the layer acts here
    var m = 1.0;
    if (maskKind == 1) { m = prev; }
    if (maskKind == 2) { m = smoothstep(k4.x - k4.y, k4.x + k4.y, p0.y); }
    if (maskKind == 3) { m = smoothstep(k4.x - k4.y, k4.x + k4.y, length(p0)); }
    if (maskKind == 4) { m = smoothstep(k4.x - k4.y, k4.x + k4.y, thickness(p0, f0, k4.x + k4.y, h)); }
    if (maskKind != 0 && k4.z > 0.5) { m = 1.0 - m; }
    // the pre-filter, in this layer's units; and how much of a single-sized pattern can be seen at all
    fpq = u.head[2].y / scale;
    var seen = 1.0;
    if (kind == 0) { seen = keepSize(a); }
    else if (kind == 1 || kind == 11 || kind == 12 || kind == 14) { seen = keepSize(0.5); }
    else if (kind == 7) { seen = keepSize(1.0 / sqrt(max(b, 1.0))); }
    else if (kind == 8 || kind == 10) { seen = keepSize(max(2.0 * b, 0.25)); }
    else if (kind == 9) { seen = keepSize(a); }
    else if (kind == 13) { seen = keepSize(0.5 * a); }
    let alpha = opacity * m * seen;

    // the pattern's own space: in units of its size, drawn out along its direction
    var q = p / scale;
    let along = dot(q, axis);
    q = q - axis * along * (1.0 - 1.0 / stretch);
    // evolving: a small circle through the pattern's space, so that a loop comes back to its start
    let drift = vec3f(cos(phase), sin(phase), 0.0) * select(0.0, 0.7, phase != 0.0);

    if (kind == 6) {
      // warp: everything after it reads its pattern somewhere else
      let oct = i32(clamp(a, 1.0, 5.0));
      let w = vec3f(fbm(q + drift, oct, 0.5, seed, 0), fbm(q + drift + vec3f(31.4, 7.7, -12.9), oct, 0.5, seed + 7u, 0), fbm(q + drift + vec3f(-5.3, 19.1, 23.3), oct, 0.5, seed + 13u, 0));
      p = p + alpha * depth * w;
      prev = 0.5 + 0.5 * w.x;
      continue;
    }

    openR = 0.0;
    openScale = scale;
    openAxis = axis;
    openStretch = stretch;
    openSize = 1.0;
    // (carved holes: only about points that lie near the surface, within half a hole of it)
    if (mode == 1 && (kind == 0 || kind == 2)) { openR = 0.45 * a * scale; }

    // A carving solid kind changes nothing where the field is further out than its shapes reach below
    // zero, or deeper in than it carves: there its cell search is not done at all.
    if (mode == 1 && (kind == 0 || kind == 1 || kind == 2 || kind == 13 || kind == 14)) {
      let hmin = select(0.5, 0.25, kind == 13) * a * scale;
      let reach = select(1e9, depth, depth > 0.0);
      if (f > hmin + 2.0 * soft || f < -(reach + 2.0 * soft)) {
        prev = 0.0;
        continue;
      }
    }

    var solid = false;
    var H = 0.0;       // a solid kind: distance to its shape, negative inside (scene units)
    var r = 0.0;       // a relief: its height, 0..1
    if (kind == 0) {
      let c = cells(q, seed, b, phase, 1.0);
      H = (c.d1 - 0.5 * a) * scale;
      solid = true;
    } else if (kind == 1) {
      let c = cells(q, seed, b, phase, 1.0);
      H = (cellEdge(q, c, seed, b, phase, 1.0) - 0.5 * a) * scale;
      solid = true;
    } else if (kind == 2) {
      // three sizes of pore, the smaller ones fewer as the spread falls
      var best = 1e9;
      var s = 1.0;
      for (var o = 0; o < 3; o++) {
        let keep = select(b, 1.0, o == 0);
        openSize = s;
        if (mode == 1) { openR = 0.45 * a * s * scale; }
        let c = cells(q / s, seed + u32(o) * 977u, 0.9, phase, keep);
        // (a size of pore too small to be seen is left out)
        let k = keepSize(a * s);
        if (k > 0.0) { best = min(best, (c.d1 - 0.5 * a * k) * s * scale); }
        s *= 0.46;
      }
      H = best;
      solid = true;
    } else if (kind == 3) {
      r = 0.5 + 0.5 * clamp(fbm(q + drift, i32(clamp(a, 1.0, 6.0)), b, seed, 0) * 1.4, -1.0, 1.0);
    } else if (kind == 4) {
      r = fbm(q + drift, i32(clamp(a, 1.0, 6.0)), b, seed, 1);
    } else if (kind == 5) {
      r = clamp(fbm(q + drift, i32(clamp(a, 1.0, 6.0)), b, seed, 2) * 1.6, 0.0, 1.0);
    } else if (kind == 7) {
      let v = dot(q, axis) * stretch + a * fbm(q * 0.5 + drift, 4, 0.5, seed, 0);
      r = pow(0.5 + 0.5 * cos(TAU * v), b);
    } else if (kind == 8) {
      let v = dot(q, axis) * stretch + a * fbm(q * 0.5 + drift, 3, 0.5, seed, 0);
      let fl = floor(v);
      let st = fl + smoothstep(0.5 - b, 0.5 + b, v - fl);
      r = st - v + 0.5;
    } else if (kind == 9) {
      r = dishes(q, seed, b, 0.5 * a, phase);
    } else if (kind == 10) {
      let ax = dot(q, axis) * stretch;
      let rad = length(q - axis * dot(q, axis));
      let v = select(ax, rad, a > 0.5);
      r = 0.5 + 0.5 * cos(TAU * v - phase) * exp(-b * abs(v) * 0.3);
    } else if (kind == 11) {
      let c = cells(q, seed, b, phase, 1.0);
      let e = cellEdge(q, c, seed, b, phase, 1.0);
      r = 1.0 - smoothstep(0.0, max(0.5 * a, 1e-3), e);
    } else if (kind == 12) {
      let c = cells(q, seed, 0.9, phase, 1.0);
      let e = cellEdge(q, c, seed, 0.9, phase, 1.0);
      let g1 = normalize(hash3(c.i1, seed + 5u) * 2.0 - 1.0 + 1e-4);
      let g2 = normalize(hash3(c.i2, seed + 5u) * 2.0 - 1.0 + 1e-4);
      let h1 = hash1(c.i1, seed + 9u) + a * dot(q - c.c1, g1);
      let h2 = hash1(c.i2, seed + 9u) + a * dot(q - c.c2, g2);
      let w = 0.5 * (1.0 - smoothstep(0.0, b, e));
      r = clamp(0.5 + 0.5 * mix(h1, h2, w) - 0.25, 0.0, 1.0);
    } else if (kind == 13) {
      let z = q * TAU;
      let g = sin(z.x) * cos(z.y) + sin(z.y) * cos(z.z) + sin(z.z) * cos(z.x);
      let gr = TAU * vec3f(cos(z.x) * cos(z.y) - sin(z.z) * sin(z.x),
                           cos(z.y) * cos(z.z) - sin(z.x) * sin(z.y),
                           cos(z.z) * cos(z.x) - sin(z.y) * sin(z.z));
      H = (abs(g) / max(length(gr), 0.5 * TAU) - 0.25 * a) * scale;
      solid = true;
    } else if (kind == 14) {
      let bs = basis(axis);
      let uv = vec2f(dot(q, bs[0]), dot(q, bs[1]));
      let sz = vec2f(1.0, 1.7320508);
      let ga = uv - sz * floor(uv / sz) - 0.5 * sz;
      let uv2 = uv - 0.5 * sz;
      let gb = uv2 - sz * floor(uv2 / sz) - 0.5 * sz;
      let g = select(gb, ga, dot(ga, ga) < dot(gb, gb));
      let ag = abs(g);
      let edge = 0.5 - max(dot(ag, vec2f(0.5, 0.8660254)), ag.x);
      H = (edge - 0.5 * a) * scale;
      solid = true;
    } else if (kind == 15) {
      let img = i32(k4.w);
      if (img >= 0) {
        let proj = i32(round(a));
        let s = p / scale;
        var v = 0.0;
        if (proj == 0) {
          var w = pow(abs(nrm), vec3f(4.0));
          w = w / max(w.x + w.y + w.z, 1e-12);
          v = w.x * picture(s.zy * vec2f(1.0, -1.0) + 0.5, img) + w.y * picture(s.xz + 0.5, img) + w.z * picture(s.xy * vec2f(1.0, -1.0) + 0.5, img);
        } else if (proj == 1) {
          v = picture(s.xy * vec2f(1.0, -1.0) + 0.5, img);
        } else if (proj == 2) {
          v = picture(s.zy * vec2f(1.0, -1.0) + 0.5, img);
        } else if (proj == 3) {
          v = picture(s.xz + 0.5, img);
        } else {
          v = picture(vec2f(atan2(p.z, p.x) / TAU + 0.5, -s.y + 0.5), img);
        }
        r = clamp((v - 0.5) * b + 0.5, 0.0, 1.0);
      }
    }

    if (kind == 16) {
      // growth: a ridged crust read on the surface itself (the pattern's space flattened along the
      // surface's direction, so each crest stands straight out of it), bent sideways by noise
      let flat = q - nrm * dot(q, nrm) * 0.92;
      let bend = b * vec3f(gnoise(q * 0.5 + drift, seed + 3u), gnoise(q * 0.5 + drift + vec3f(17.0, 3.0, -9.0), seed + 5u), 0.0);
      r = pow(clamp(fbm(flat + bend + drift, 4, 0.55, seed, 1), 0.0, 1.0), a);
    }

    // what the field would be with the layer's change made
    var T = f;
    if (solid) {
      if (mode == 1) {
        let reach = select(1e9, depth, depth > 0.0);
        T = smax(f, -max(H, -f - reach), soft);
      } else if (mode == 0) {
        let reach = select(1e9, depth, depth > 0.0);
        T = smin(f, max(H, f - reach), soft);
      } else {
        T = smax(f, min(H, -(f + max(depth, 0.0))), soft);
      }
      r = 1.0 - smoothstep(-soft, soft, H);
    } else {
      if (mode == 0) { T = f - depth * r; }
      else if (mode == 1) { T = f + depth * r; }
      else { T = f + depth * (1.0 - 2.0 * r); }
    }
    let d = alpha * (T - f);
    if (blend == 0) { f = f + d; }
    else if (blend == 1) { f = f - d; }
    else if (blend == 2) { f = f + sneg(d, soft); }
    else { f = f + spos(d, soft); }
    prev = r;
  }
  textureStore(dst, gid, vec4f(f, 0.0, 0.0, 0.0));
}
