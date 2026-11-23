// Layers into one volume: up to four shapes, each built into its own distance volume, read at the point
// carried back through its placement and joined by the rules of core/src/layers.rs (which holds the same
// arithmetic in plain Rust, and whose comments say why it is what it is).
//
//   1. morph   each layer's distance moved towards the layers it morphs with
//   2. carve   each layer cut by the layers that cut it and masked by those it overlaps with
//   3. join    each body (the layers that booleans hold together): the smooth union of its layers that show
//   4. stage   each body cut to a ball; the scene is the nearest of the bodies
//
// Two volumes can be written: the scene's (one distance), and, when there is more than one body, the
// bodies' (a distance per body, for whatever gives each body its own look): two channels for two bodies,
// four for three or four.
//
// Voxel (i, j, k) of the volume written is the point ((i + 0.5, j + 0.5, k + 0.5) / res * 2 - 1) * hb,
// as in every volume of the program.

struct C {
  rows: array<vec4f, 12>,   // world -> layer i's own space: rows 3i, 3i+1, 3i+2, each dotted with (p, 1)
  info: array<vec4f, 4>,    // per layer: its size, how far from the centre its volume holds values, 1 if it shows, its body
  morph: mat4x4f,           // body i = dot(morph[i], distances)
  term: mat4x4f,            // carve of i by j: +1 = j cuts i, -1 = i keeps only what j contains, 0 = neither
  gap: mat4x4f,             // the clearance of the cut of i by j
  kc: mat4x4f,              // seam radius between two layers' surfaces: where they carve,
  ku: mat4x4f,              // and where they join
  prof: mat4x4f,            // each pair's profile, as its power (3 smooth, 2 round, 1 bevel)
  tfill: vec4f,             // a triple's fill (a distance) and reach, kept under the layer it leaves out
  treach: vec4f,
  q: vec4f,                 // the quad's fill and reach; how many layers; voxels per side of the volume written
  stage: vec4f,             // the stage's radius and the rounding of its edge; hb (the scene's: larger than a
                            // layer's when the layers are placed far out); how many bodies
  wide: array<vec4f, 4>,    // per layer with a wide volume: its own volume is read up to x, the wide one
                            // beyond y, a mix between; z = voxels per side of the wide volume (0 = it has none)
  dispatch_offset: vec4u,   // start of the writable slice range in a staged volume
}

@group(0) @binding(0) var<uniform> c: C;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var vol0: texture_3d<f32>;
@group(0) @binding(3) var vol1: texture_3d<f32>;
@group(0) @binding(4) var vol2: texture_3d<f32>;
@group(0) @binding(5) var vol3: texture_3d<f32>;
// The layers' wide volumes: small volumes in which a tube's distance stays right far from the tube. A
// tube's own volume holds the exact distance only a few voxels out and a safe, stepped lower bound beyond,
// which is all the tracer needs but would show as steps in a seam.
@group(0) @binding(6) var wide0: texture_3d<f32>;
@group(0) @binding(7) var wide1: texture_3d<f32>;
@group(0) @binding(8) var wide2: texture_3d<f32>;
@group(0) @binding(9) var wide3: texture_3d<f32>;

const LAYER_HB: f32 = 1.5;          // a layer's own volume spans [-1.5, 1.5] of its own space
const EMPTY: f32 = 1.0;             // what a body none of whose layers shows reads everywhere
const ABSENT: f32 = 4.0;            // and a body that does not exist (the most a volume stores)
const SHARE_WIDTH: f32 = 2.0;       // see layers.rs
const TINY: f32 = 1e-6;

// the seam radii in use (those for carving, or those for joining), and which way a junction's fill acts
var<private> K: mat4x4f;
var<private> FS: f32;

// what a seam of radius k takes off the smaller of two distances that differ by delta
fn dip(k: f32, n: f32, delta: f32) -> f32 {
  if (k <= 0.0 || delta >= k) { return 0.0; }
  let t = 1.0 - delta / k;
  if (n < 1.5) { return 0.5 * k * t; }
  if (n < 2.5) { return 0.25 * k * t * t; }
  return k * t * t * t / 6.0;
}

// a step from 0 (at -1) to 1 (at +1) with no jump in slope or curvature
fn ramp(t: f32) -> f32 {
  let x = clamp(0.5 * t + 0.5, 0.0, 1.0);
  return x * x * x * (x * (6.0 * x - 15.0) + 10.0);
}

// 1 where the values are equal, 0 where their differences reach `reach` (root mean square over the pairs)
fn bump3(a: f32, b: f32, e: f32, reach: f32) -> f32 {
  if (reach <= 0.0) { return 0.0; }
  let q = (a - b) * (a - b) + (a - e) * (a - e) + (b - e) * (b - e);
  let s = 1.0 - q / (3.0 * reach * reach);
  if (s <= 0.0) { return 0.0; }
  return s * s * s;
}
fn bump4(v: vec4f, reach: f32) -> f32 {
  if (reach <= 0.0) { return 0.0; }
  let a = v.xxxy - v.yzwz;
  let b = v.yz - v.ww;
  let q = dot(a, a) + dot(b, b);
  let s = 1.0 - q / (6.0 * reach * reach);
  if (s <= 0.0) { return 0.0; }
  return s * s * s;
}

// two
fn u2(v: vec4f, a: u32, b: u32) -> f32 {
  var w = v;
  return min(w[a], w[b]) - dip(K[a][b], c.prof[a][b], abs(w[a] - w[b]));
}

// l joined to the join of x and y: its own seam with each, in the share that their seam belongs to each
fn meet(v: vec4f, x: u32, y: u32, l: u32, kap: f32) -> f32 {
  var w = v;
  let u = u2(v, x, y);
  let wx = ramp((w[y] - w[x]) / kap);
  let d = abs(u - w[l]);
  return min(u, w[l]) - (wx * dip(K[l][x], c.prof[l][x], d) + (1.0 - wx) * dip(K[l][y], c.prof[l][y], d));
}

// three: each in turn joined to the join of the other two, the three results averaged
fn u3(v: vec4f, a: u32, b: u32, e: u32) -> f32 {
  var w = v;
  let kap = max(SHARE_WIDTH * max(K[a][b], max(K[a][e], K[b][e])), TINY);
  let t0 = meet(v, a, b, e, kap);
  let t1 = meet(v, a, e, b, kap);
  let t2 = meet(v, b, e, a, kap);
  let out = 6u - a - b - e;
  var extra = 0.0;
  let fill = c.tfill[out];
  if (fill != 0.0) { extra = FS * fill * bump3(w[a], w[b], w[e], c.treach[out]); }
  // (the mean, taken so that three equal numbers give exactly that number)
  return t0 + ((t1 - t0) + (t2 - t0)) / 3.0 - extra;
}

// four: the same, one step up
fn u4(v: vec4f) -> f32 {
  var w = v;
  var kap = TINY;
  for (var a = 0u; a < 4u; a++) {
    for (var b = a + 1u; b < 4u; b++) { kap = max(kap, SHARE_WIDTH * K[a][b]); }
  }
  var terms: array<f32, 4>;
  for (var l = 0u; l < 4u; l++) {
    // the other three, lowest first
    var t = vec3u(select(0u, 1u, l == 0u), select(1u, 2u, l <= 1u), select(2u, 3u, l <= 2u));
    let u = u3(v, t.x, t.y, t.z);
    // how much of that seam belongs to each of the three: most to the nearest
    var sh = vec3f(0.0);
    for (var i = 0u; i < 3u; i++) {
      sh[i] = ramp((w[t[(i + 1u) % 3u]] - w[t[i]]) / kap) * ramp((w[t[(i + 2u) % 3u]] - w[t[i]]) / kap);
    }
    let total = sh.x + sh.y + sh.z;
    let d = abs(u - w[l]);
    var take = 0.0;
    for (var i = 0u; i < 3u; i++) {
      take += sh[i] / total * dip(K[l][t[i]], c.prof[l][t[i]], d);
    }
    terms[l] = min(u, w[l]) - take;
  }
  var extra = 0.0;
  if (c.q.x != 0.0) { extra = FS * c.q.x * bump4(v, c.q.y); }
  return terms[0] + ((terms[1] - terms[0]) + (terms[2] - terms[0]) + (terms[3] - terms[0])) / 4.0 - extra;
}

// the smooth union of the values of `m` layers, `ids` (lowest first)
fn join(v: vec4f, ids: vec4u, m: u32) -> f32 {
  var w = v;
  if (m == 1u) { return w[ids.x]; }
  if (m == 2u) { return u2(v, ids.x, ids.y); }
  if (m == 3u) { return u3(v, ids.x, ids.y, ids.z); }
  return u4(v);
}

// A small volume read smoothly: a cubic B-spline through its samples, as eight lookups placed between
// texels (as the tracer reads the surface). t: the place in the volume, 0..1; res: its voxels per side.
fn cubic(tex: texture_3d<f32>, t: vec3f, res: f32) -> f32 {
  let cc = t * res - 0.5;
  let i = floor(cc);
  let f = cc - i;
  let f2 = f * f;
  let f3 = f2 * f;
  let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
  let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
  let w3 = f3 / 6.0;
  let w2 = 1.0 - w0 - w1 - w3;
  let g0 = w0 + w1;
  let g1 = w2 + w3;
  let a = (i - 0.5 + w1 / g0) / res;
  let b = (i + 1.5 + w3 / g1) / res;
  return g0.z * (g0.y * (g0.x * textureSampleLevel(tex, smp, vec3f(a.x, a.y, a.z), 0.0).r + g1.x * textureSampleLevel(tex, smp, vec3f(b.x, a.y, a.z), 0.0).r)
               + g1.y * (g0.x * textureSampleLevel(tex, smp, vec3f(a.x, b.y, a.z), 0.0).r + g1.x * textureSampleLevel(tex, smp, vec3f(b.x, b.y, a.z), 0.0).r))
       + g1.z * (g0.y * (g0.x * textureSampleLevel(tex, smp, vec3f(a.x, a.y, b.z), 0.0).r + g1.x * textureSampleLevel(tex, smp, vec3f(b.x, a.y, b.z), 0.0).r)
               + g1.y * (g0.x * textureSampleLevel(tex, smp, vec3f(a.x, b.y, b.z), 0.0).r + g1.x * textureSampleLevel(tex, smp, vec3f(b.x, b.y, b.z), 0.0).r));
}

// Layer i's distance at the world point p: its volume read at the point carried back through its
// placement, times its size. Beyond the few voxels in which a tube's own volume is exact, its wide volume
// takes over. Outside its box: the value on the nearest point of the box with the distance to the box
// added on, squared and summed (the shape lies inside the box, so it is at least that far).
@group(0) @binding(10) var<storage, read> occurrences: array<vec4f>;

fn layerLocal(t: texture_3d<f32>, tw: texture_3d<f32>, i: u32, q: vec3f) -> f32 {
  let lim = c.info[i].y;
  let qc = clamp(q, vec3f(-lim), vec3f(lim));
  let at = qc / (2.0 * LAYER_HB) + 0.5;
  var d = textureSampleLevel(t, smp, at, 0.0).r;
  let lres = f32(textureDimensions(t).x);
  if (abs(d) < 18.0 * LAYER_HB / lres) {
    d = cubic(t, at, lres);
  }
  let w = c.wide[i];
  if (w.z > 0.0 && d > w.x) {
    let s = clamp((d - w.x) / (w.y - w.x), 0.0, 1.0);
    d += (cubic(tw, at, w.z) - d) * (s * s * (3.0 - 2.0 * s));
  }
  let o = q - qc;
  let o2 = dot(o, o);
  if (o2 > 0.0) {
    let m = max(d, 0.0);
    d = sqrt(o2 + m * m);
  }
  return d;
}

fn layer(t: texture_3d<f32>, tw: texture_3d<f32>, i: u32, p: vec3f) -> f32 {
  if (c.wide[i].w > 0.0) {
    let range = occurrences[i];
    let h = vec4f(p, 1.0);
    var best = 1e6;
    var node = u32(range.x);
    while (node < u32(range.y)) {
      let lo = occurrences[node]; let hi = occurrences[node + 1u];
      let outside = max(max(lo.xyz - p, p - hi.xyz), vec3f(0.0));
      if (dot(outside, outside) > max(best, 0.0) * max(best, 0.0)) {
        node = u32(lo.w); continue;
      }
      if (hi.w > 0.0) {
        let q = vec3f(dot(occurrences[node+2u],h), dot(occurrences[node+3u],h), dot(occurrences[node+4u],h));
        best = min(best, hi.w * layerLocal(t, tw, i, q));
      }
      node += 5u;
    }
    return best;
  }
  let r0 = c.rows[3u * i];
  let r1 = c.rows[3u * i + 1u];
  let r2 = c.rows[3u * i + 2u];
  let q = vec3f(dot(r0.xyz, p) + r0.w, dot(r1.xyz, p) + r1.w, dot(r2.xyz, p) + r2.w);
  return c.info[i].x * layerLocal(t, tw, i, q);
}

// each body's distance at the world point p (a body that does not exist: ABSENT)
fn combine(p: vec3f) -> vec4f {
  let n = u32(c.q.z);
  var d = vec4f(0.0);
  d.x = layer(vol0, wide0, 0u, p);
  if (n > 1u) { d.y = layer(vol1, wide1, 1u, p); }
  if (n > 2u) { d.z = layer(vol2, wide2, 2u, p); }
  if (n > 3u) { d.w = layer(vol3, wide3, 3u, p); }
  // 1. morph
  var b = vec4f(dot(c.morph[0], d), dot(c.morph[1], d), dot(c.morph[2], d), dot(c.morph[3], d));
  // 2. carve: an intersection is the union of the negated distances, negated
  var cv = vec4f(0.0);
  for (var i = 0u; i < n; i++) {
    if (c.info[i].z < 0.5) { continue; }
    var w = vec4f(0.0);
    var ids = vec4u(0u);
    var k = 0u;
    for (var j = 0u; j < n; j++) {
      if (j == i) {
        w[j] = -b[i];
        ids[k] = j;
        k++;
      } else if (c.term[i][j] != 0.0) {
        w[j] = c.term[i][j] * b[j] - c.gap[i][j];
        ids[k] = j;
        k++;
      }
    }
    var ci = b[i];
    if (k > 1u) {
      K = c.kc;
      FS = -1.0;
      ci = -join(w, ids, k);
    }
    cv[i] = ci;
  }
  // 3. join, body by body, and 4. stage
  var bodies = vec4f(ABSENT);
  let nb = u32(c.stage.w);
  let s = length(p) - c.stage.x;
  K = c.ku;
  FS = 1.0;
  for (var k = 0u; k < nb; k++) {
    var shown = vec4u(0u);
    var m = 0u;
    for (var i = 0u; i < n; i++) {
      if (c.info[i].z > 0.5 && u32(c.info[i].w) == k) {
        shown[m] = i;
        m++;
      }
    }
    var f = EMPTY;
    if (m > 0u) { f = join(cv, shown, m); }
    f = max(f, s) + dip(c.stage.y, 3.0, abs(f - s));
    bodies[k] = clamp(f, -1.0, 4.0);
  }
  return bodies;
}

// the scene's distance: the nearest of the bodies
fn nearest(b: vec4f) -> f32 {
  return min(min(b.x, b.y), min(b.z, b.w));
}

fn point(xy: vec2f, zc: f32) -> vec3f {
  return (vec3f(xy / c.q.w, zc) * 2.0 - 1.0) * c.stage.z;
}

// drawn: one slice per pass, the slice number arriving as the instance number
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) @interpolate(flat) zc: f32,
}
@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) slice: u32) -> VOut {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  var o: VOut;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.zc = (f32(slice) + 0.5) / c.q.w;
  return o;
}
@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  return vec4f(nearest(combine(point(in.pos.xy, in.zc))), 0.0, 0.0, 1.0);
}
// the bodies' volume, drawn the same way
@fragment
fn fs_bodies(in: VOut) -> @location(0) vec4f {
  return combine(point(in.pos.xy, in.zc));
}

//#compute  (everything below is left out where volumes cannot be written by compute shaders)
@group(1) @binding(0) var outVol: texture_storage_3d<r16float, write>;

@group(1) @binding(1) var outBodies: texture_storage_3d<rgba16float, write>;

// computed: the whole volume in one dispatch
@compute @workgroup_size(8, 8, 4)
fn cs(@builtin(global_invocation_id) dispatch_id: vec3u) {
  let id = dispatch_id + c.dispatch_offset.xyz;
  textureStore(outVol, id, vec4f(nearest(combine(point(vec2f(id.xy) + 0.5, (f32(id.z) + 0.5) / c.q.w))), 0.0, 0.0, 1.0));
}
// the scene's volume and the bodies' at once
@compute @workgroup_size(8, 8, 4)
fn cs_bodies(@builtin(global_invocation_id) dispatch_id: vec3u) {
  let id = dispatch_id + c.dispatch_offset.xyz;
  let b = combine(point(vec2f(id.xy) + 0.5, (f32(id.z) + 0.5) / c.q.w));
  textureStore(outVol, id, vec4f(nearest(b), 0.0, 0.0, 1.0));
  textureStore(outBodies, id, b);
}

//#compute2  (and this where a compute shader can write a two-channel 16-bit float volume as well)
@group(1) @binding(2) var outBodies2: texture_storage_3d<rg16float, write>;

// the same for two bodies: the bodies' volume has two channels and is half the size
@compute @workgroup_size(8, 8, 4)
fn cs_bodies2(@builtin(global_invocation_id) dispatch_id: vec3u) {
  let id = dispatch_id + c.dispatch_offset.xyz;
  let b = combine(point(vec2f(id.xy) + 0.5, (f32(id.z) + 0.5) / c.q.w));
  textureStore(outVol, id, vec4f(nearest(b), 0.0, 0.0, 1.0));
  textureStore(outBodies2, id, vec4f(b.x, b.y, 0.0, 1.0));
}
