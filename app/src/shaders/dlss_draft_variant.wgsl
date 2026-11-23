// The guides Ray Reconstruction wants, in ONE pass with seven targets, from the guide pass's four (dlss_draft_guide.wgsl):
// guide G2 (the exit point of the refracted ray: depth, normal, the way out as hit distance), diffuse albedo, roughness and specular
// albedo for glass, and the motion vector (mode 1: the surface point seen through the pixel, where it was on the previous screen;
// mode 3: by correspondence with the previous frame's first hits, for a drag). The lab's guide variants, measured in lab_rr.md.

struct V { a: vec4f, b: vec4f, c: vec4f, d: vec4f, o0: vec4f, o1: vec4f, o2: vec4f, q0: vec4f, q1: vec4f, q2: vec4f, e: vec4f }
@group(0) @binding(0) var<uniform> v: V;
@group(0) @binding(1) var g0: texture_2d<f32>;
@group(0) @binding(2) var g1: texture_2d<f32>;
@group(0) @binding(3) var g2: texture_2d<f32>;
@group(0) @binding(4) var g3: texture_2d<f32>;
@group(0) @binding(5) var g5: texture_2d<f32>;
@group(0) @binding(6) var gp0: texture_2d<f32>;
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
const FP16MAX: f32 = 65504.0;
struct F { n: vec3f, nov: f32 }
fn facing(n: vec3f, p: vec3f) -> F {
  let e = vec3f(0.0, 0.0, v.a.z) - p;
  let vv = e / max(length(e), 1e-9);
  let f = dot(n, vv);
  return F(select(n, -n, f < 0.0), abs(f));
}
struct S { glass: bool, depth: f32, n: vec3f, nov: f32, hit: f32, p0: vec3f, pq: vec3f }
fn surf(c: vec2i) -> S {
  var r: S;
  r.glass = false; r.depth = 10000.0; r.n = vec3f(0.0); r.nov = 0.0; r.hit = FP16MAX; r.p0 = vec3f(0.0); r.pq = vec3f(0.0);
  let a = textureLoad(g0, c, 0);
  if (a.w < 0.5) { return r; }
  let n0 = textureLoad(g1, c, 0).xyz;
  let gq = textureLoad(g2, c, 0);
  let nq = textureLoad(g3, c, 0).xyz;
  let lbey = textureLoad(g5, c, 0).x;
  let dist = v.a.z;
  let p0 = a.xyz;
  let q = gq.xyz;
  let s = gq.w;
  let f0 = facing(n0, p0);
  let f2 = facing(nq, q);
  let d0 = dist - p0.z;
  let d2 = dist - q.z;
  let variant = i32(v.a.x + 0.5);
  r.glass = true; r.p0 = p0; r.pq = q;
  var hitDef = FP16MAX;
  if (variant == 0) { r.depth = d0; r.n = f0.n; r.nov = f0.nov; }
  else if (variant == 1) { r.depth = d0; r.n = vec3f(0.0, 0.0, 1.0); r.nov = 1.0; }
  else if (variant == 2) { r.depth = d2; r.n = f2.n; r.nov = f2.nov; if (lbey > 0.0) { hitDef = lbey; } }
  else { r.depth = 0.5 * (d0 + d2); r.n = normalize(f0.n + f2.n); r.nov = 0.5 * (f0.nov + f2.nov); }
  let hm = i32(v.d.x + 0.5);
  if (hm == 255) { r.hit = hitDef; }
  else if (hm == 1) { r.hit = select(FP16MAX, s, s > 0.0); }
  else if (hm == 2) { r.hit = select(FP16MAX, lbey, lbey > 0.0); }
  else { r.hit = FP16MAX; }
  return r;
}
fn envBrdf(spec: f32, alpha: f32, nov_in: f32) -> f32 {
  let nov = abs(nov_in);
  let X = vec4f(1.0, nov, nov * nov, nov * nov * nov);
  let Y = vec4f(1.0, alpha, alpha * alpha, alpha * alpha * alpha);
  let xyw = vec3f(X.x, X.y, X.w);
  let yw = vec3f(Y.x, Y.y, Y.w);
  let xzw = vec3f(X.x, X.z, X.w);
  let b0 = (0.99044 * X.x - 1.28514 * X.y) * Y.x + (1.29678 * X.x - 0.755907 * X.y) * Y.y;
  let m2 = vec3f(dot(vec3f(1.0, 2.92338, 59.4188), xyw), dot(vec3f(20.3225, -27.0302, 222.592), xyw), dot(vec3f(121.563, 626.13, 316.627), xyw));
  let bias = b0 / dot(m2, yw);
  let s0 = (0.0365463 * X.x + 3.32707 * X.y) * Y.x + (9.0632 * X.x - 9.04756 * X.y) * Y.y;
  let m4 = vec3f(dot(vec3f(1.0, 3.59685, -1.36772), xzw), dot(vec3f(9.04401, -16.3174, 9.22949), xzw), dot(vec3f(5.56589, 19.7886, -20.2123), xzw));
  let scale = s0 / dot(m4, yw);
  return spec * max(0.0, scale) + max(0.0, bias * min(1.0, spec * 50.0));
}
fn proj(p: vec3f) -> vec2f {
  let den = (v.a.z - p.z) * v.a.w;
  let ux = p.x / den;
  let uy = p.y / den;
  let fx = ((ux - v.b.z) * v.b.y + v.b.x) / 2.0;
  let fy = ((uy - v.b.w) * v.b.y + v.b.y) / 2.0;
  return vec2f(fx, v.b.y - fy);
}
fn mvOf(s: S) -> vec4f {
  let mode = i32(v.c.w + 0.5);
  if (!s.glass || mode == 0) { return vec4f(0.0); }
  if (mode == 3) {
    // motion by correspondence: the point's place in the frame before is found by trying two hypotheses (it stayed where it is in the scene, or it was
    // `e.xyz` (the drag of this frame, scene units, along x) behind) and keeping the one the previous frame's own first hits agree with (object-space
    // distance under two pixel footprints); no hypothesis agrees (uncovered): the camera's motion alone.
    let p = s.p0;
    let po = p.x * v.o0.xyz + p.y * v.o1.xyz + p.z * v.o2.xyz;
    let foot = 2.0 * v.a.w * (v.a.z - p.z) / v.b.y;
    var best = 1e9;
    var bx = proj(vec3f(dot(v.q0.xyz, po), dot(v.q1.xyz, po), dot(v.q2.xyz, po)));
    for (var hy = 0; hy < 2; hy++) {
      let qo = select(po, po - v.e.xyz, hy == 1);
      let pp = vec3f(dot(v.q0.xyz, qo), dot(v.q1.xyz, qo), dot(v.q2.xyz, qo));
      let x1 = proj(pp);
      let ix = vec2i(floor(x1));
      let dims = vec2i(textureDimensions(gp0));
      if (ix.x < 0 || ix.y < 0 || ix.x >= dims.x || ix.y >= dims.y) { continue; }
      let pv = textureLoad(gp0, ix, 0);
      if (pv.w < 0.5) { continue; }
      let ppo = pv.x * v.q0.xyz + pv.y * v.q1.xyz + pv.z * v.q2.xyz;
      let err = length(ppo - qo);
      if (err < best && err < 2.0 * foot) { best = err; bx = x1; }
    }
    return vec4f(bx - proj(p), 0.0, 0.0);
  }
  let p = select(s.p0, s.pq, mode == 2);
  let po = p.x * v.o0.xyz + p.y * v.o1.xyz + p.z * v.o2.xyz;
  let pp = vec3f(dot(v.q0.xyz, po), dot(v.q1.xyz, po), dot(v.q2.xyz, po));
  let d = proj(pp) - proj(p);
  return vec4f(d, 0.0, 0.0);
}
struct AllOut {
  @location(0) depth: vec4f, @location(1) normal: vec4f, @location(2) rough: vec4f, @location(3) dalb: vec4f,
  @location(4) salb: vec4f, @location(5) hit: vec4f, @location(6) mv: vec4f,
}
@fragment fn fs_all(@builtin(position) pos: vec4f) -> AllOut {
  let s = surf(vec2i(pos.xy));
  var o: AllOut;
  o.depth = vec4f(s.depth, 0.0, 0.0, 0.0);
  o.normal = vec4f(s.n, 0.0);
  o.rough = vec4f(select(0.0, v.a.y, s.glass), 0.0, 0.0, 0.0);
  o.dalb = vec4f(select(vec3f(0.5), v.c.xyz, s.glass), 1.0);
  o.salb = vec4f(vec3f(select(0.0, envBrdf(0.04, v.a.y * v.a.y, s.nov), s.glass)), 1.0);
  o.hit = vec4f(s.hit, 0.0, 0.0, 0.0);
  o.mv = mvOf(s);
  return o;
}
