// Surface lighting: direct light with shadows on opaque and milky bodies, drawn after the tracer.
//
// The tracer is left as it is: it only ever gets the room's light (or, for a milky body, its light
// volume) and writes its colour and the distance to the first surface along each pixel's ray. This pass
// goes back to that surface from the distance alone (nothing more is carried through the tracer's
// marching, which would slow it), and for each light at infinity and each lamp takes one point of the
// light at random, marches from the surface towards it through the distance field, and adds the light
// that arrives. One sample per light per frame: frames averaged together converge to the soft shadow
// of the whole light.
//
// It is drawn onto the traced frame with blending: colour = what this writes + the frame's colour
// times this alpha. So the frame's own light can be dimmed (occlusion, shadow on a milky body's inner
// light) and the direct light added, while the distance in the frame's alpha is kept.

struct SLight {
  a: vec4f,                 // xyz: towards the light's middle (the lights' own frame)   w: tangent of its half width
  u: vec4f,                 // xyz: the direction across it   w: tangent of its half height
  v: vec4f,                 // xyz: the direction up it   w: the share of its size over which its edge fades
  rgb: vec4f,               // rgb: its radiance, already in the units the material wants
}
struct SU {
  cam: vec4f,               // xy: the frame's size in pixels   zw: the sub-pixel jitter of this frame
  cam2: vec4f,              // xy: the view's shift   z: the camera's distance   w: tan(half vertical fov)
  k0: vec4f,                // x: voxels a side of the volume   y: half the volume's box   z: frame number   w: what to show (for looking)
  k1: vec4f,                // x: rays the tracer sends through each pixel
  obj0: vec4f,
  obj1: vec4f,
  obj2: vec4f,
  env0: vec4f,
  env1: vec4f,
  env2: vec4f,
  base: vec4f,              // rgb: the surface's base colour   w: how many lights at infinity
  k: vec4f,                 // x: weight of the direct diffuse light   y: how far the shadows dim the frame's own light
                            // z: how far the ambient occlusion dims it   w: 1 = lamps stand in the scene
  clay: vec4f,              // x: the light clay gathers on average   y: how far differences from it are drawn out
  lights: array<SLight, 12>,
}
@group(0) @binding(0) var<uniform> s: SU;
@group(0) @binding(1) var frameTex: texture_2d<f32>;
@group(0) @binding(2) var volTex: texture_3d<f32>;
@group(0) @binding(3) var volSmp: sampler;

struct LampU {
  p: vec4f,
  u: vec4f,
  v: vec4f,
  rgb: vec4f,
}
struct LampsU {
  n: vec4f,
  floor: vec4f,
  plane: vec4f,
  l: array<LampU, 8>,
}
@group(0) @binding(4) var<uniform> lamps: LampsU;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn HB() -> f32 { return s.k0.y; }
fn toObj(p: vec3f) -> vec3f { return p.x * s.obj0.xyz + p.y * s.obj1.xyz + p.z * s.obj2.xyz; }
// the lights' own frame -> the camera's (the tracer's toEnv the other way: it is a rotation)
fn fromEnv(e: vec3f) -> vec3f { return vec3f(dot(e, s.env0.xyz), dot(e, s.env1.xyz), dot(e, s.env2.xyz)); }

fn map(p: vec3f) -> f32 { return textureSampleLevel(volTex, volSmp, p / (2.0 * HB()) + 0.5, 0.0).r; }
// the smooth (cubic B-spline) reading of the same data, as the tracer finds surfaces and normals with
fn vol(t: vec3f) -> f32 { return textureSampleLevel(volTex, volSmp, t, 0.0).r; }
fn mapN(p: vec3f) -> f32 {
  let res = s.k0.x;
  let c = (p / (2.0 * HB()) + 0.5) * res - 0.5;
  let i = floor(c);
  let f = c - i;
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
  return g0.z * (g0.y * (g0.x * vol(vec3f(a.x, a.y, a.z)) + g1.x * vol(vec3f(b.x, a.y, a.z)))
               + g1.y * (g0.x * vol(vec3f(a.x, b.y, a.z)) + g1.x * vol(vec3f(b.x, b.y, a.z))))
       + g1.z * (g0.y * (g0.x * vol(vec3f(a.x, a.y, b.z)) + g1.x * vol(vec3f(b.x, a.y, b.z)))
               + g1.y * (g0.x * vol(vec3f(a.x, b.y, b.z)) + g1.x * vol(vec3f(b.x, b.y, b.z))));
}
fn mapW(p: vec3f) -> f32 { return map(toObj(p)); }
fn mapNW(p: vec3f) -> f32 { return mapN(toObj(p)); }
// The normal for lighting: from plain lookups a voxel and a half apart (four lookups where the smooth
// reading takes thirty-two). Light that is gathered is smooth across a surface and does not show the
// small kinks plain lookups have at voxel boundaries, which a mirror would.
fn nrm(p: vec3f) -> vec3f {
  let e = vec2f(1.0, -1.0) * (1.5 * 2.0 * HB() / s.k0.x);
  return normalize(e.xyy * mapW(p + e.xyy) + e.yyx * mapW(p + e.yyx) + e.yxy * mapW(p + e.yxy) + e.xxx * mapW(p + e.xxx));
}

// the tracer's interleaved gradient noise (to aim at the very point its first sample went through)
fn ign(p: vec2f) -> f32 { return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715)))); }

// random numbers: a hash of the pixel, the frame and a counter, so every light and every axis of it
// gets numbers of its own, unrelated to its neighbours'
var<private> seed: u32;
fn pcg(v: u32) -> u32 {
  let st = v * 747796405u + 2891336453u;
  let w = ((st >> ((st >> 28u) + 4u)) ^ st) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rnd() -> f32 {
  seed = pcg(seed);
  return f32(seed >> 8u) / 16777216.0;
}

// A point of the unit square for one light this frame: a random start of the pixel's own for each light
// (each axis its own number), moved on by the R2 sequence (Roberts 2018) frame by frame, so that any run
// of frames covers the light evenly and the average converges far faster than from fresh random points.
fn seq2() -> vec2f {
  let start = vec2f(rnd(), rnd());
  return fract(start + f32(u32(s.k0.z) % 4096u) * vec2f(0.7548777, 0.5698403));
}

// a number 0..1 of the pixel's own moved on by the golden ratio each frame
fn walk(x: f32) -> f32 { return fract(x + f32(u32(s.k0.z) % 4096u) * 0.61803399); }

// How much of the light along rd (a unit vector, camera frame) reaches p, a point of the surface facing
// n, from no further than tMax: a march through the distance field in the object's frame. w is the
// tangent of the cone the march is widened by (the light seen from p is that wide): a body passing at
// distance d, a length t along, hides the light by how d compares with w t (the estimate of I. Quilez,
// followed into the body). The first stride starts at a random point (j) and the strides vary, so that
// the stops do not beat against the volume's voxels; frames average the grain away.
fn shadowTo(p: vec3f, n: vec3f, rd_in: vec3f, w: f32, tMax: f32, j: f32) -> f32 {
  if (s.k0.w > 3.5) { return 1.0; }
  let vox = 2.0 * HB() / s.k0.x;
  let ro = toObj(p + n * (2.0 * vox));
  let rd = toObj(rd_in);
  let bound = 0.98 * HB();
  let b = dot(ro, rd);
  let h = b * b - (dot(ro, ro) - bound * bound);
  if (h <= 0.0) { return 1.0; }
  let tFar = min(-b + sqrt(h), tMax);
  var t = vox * (0.5 + j);
  var r = 1.0;
  var q = j;
  for (var i = 0; i < 48; i++) {
    if (t > tFar) { break; }
    let d = map(ro + rd * t);
    r = min(r, d / (w * t));
    if (r <= -1.0) { break; }
    q = fract(q + 0.61803399);
    t += clamp(abs(d), 0.5 * vox, 0.12) * (0.6 + 0.8 * q);
  }
  let x = clamp(0.5 - 0.5 * r, 0.0, 1.0);
  return 1.0 - x * x * (3.0 - 2.0 * x);
}

// Ambient occlusion: how far the distance field closes in over the surface, from five looks along the
// normal (the same arithmetic the clay's own folds are judged by).
fn occlusion(p: vec3f, n: vec3f) -> f32 {
  var shut = 0.0;
  var weight = 1.0;
  for (var i = 0; i < 5; i++) {
    let h = 0.012 + 0.052 * f32(i);
    shut += weight * max(h - mapW(p + n * h), 0.0) / h;
    weight *= 0.8;
  }
  return clamp(1.0 - 0.42 * shut, 0.0, 1.0);
}

struct Direct {
  lit: vec3f,               // the direct light arriving, shadowed (irradiance)
  open: vec3f,              // the same as if nothing were in the way
}

// The lights at infinity: a soft rectangle each (as core/src/env.rs paints them into the lights' map).
// Each light's light as the surface gathers it with nothing in the way is worked out without chance:
// a 4 x 4 grid over the rectangle, on the plane the map is painted from (x, y in tangents of angle), each
// cell weighed by the rectangle's fading edge and by the solid angle it covers, 1 / (1 + x^2 + y^2)^1.5,
// times the cosine at the surface. Only whether the light is hidden is left to chance: one random point
// of the light a frame and a march towards it. So a surface the light reaches unhindered has no grain at
// all, and only the penumbrae need frames to settle.
const LUM: vec3f = vec3f(0.2126, 0.7152, 0.0722);
fn atInfinity(p: vec3f, n: vec3f) -> Direct {
  var o = Direct(vec3f(0.0), vec3f(0.0));
  let count = i32(s.base.w);
  // the normal in the lights' own frame (the tracer's toEnv)
  let ne = n.x * s.env0.xyz + n.y * s.env1.xyz + n.z * s.env2.xyz;
  var each: array<vec3f, 12>;
  var top = -1;
  var wTop = 0.0;
  var wAll = 0.0;
  for (var i = 0; i < 12; i++) {
    if (i >= count) { break; }
    let L = s.lights[i];
    let tw = L.a.w;
    let th = L.u.w;
    let soft = max(L.v.w, 1e-3);
    let na = dot(ne, L.a.xyz);
    let nu = dot(ne, L.u.xyz);
    let nv = dot(ne, L.v.xyz);
    // (wholly behind the surface: every corner of the rectangle faces away)
    if (na + abs(nu) * tw + abs(nv) * th <= 0.0) { continue; }
    var open = 0.0;
    for (var gy = 0; gy < 4; gy++) {
      let y = (f32(gy) * 0.5 - 0.75) * th;
      let sy = 1.0 - smoothstep(0.0, 1.0, (abs(y) / th - (1.0 - soft)) / soft);
      for (var gx = 0; gx < 4; gx++) {
        let x = (f32(gx) * 0.5 - 0.75) * tw;
        let sx = 1.0 - smoothstep(0.0, 1.0, (abs(x) / tw - (1.0 - soft)) / soft);
        let q = 1.0 + x * x + y * y;
        // (cosine = (na + x nu + y nv) / sqrt(q); solid angle per unit of the plane = 1 / q^1.5)
        open += max(na + x * nu + y * nv, 0.0) * sx * sy / (q * q);
      }
    }
    open *= tw * th * 0.25;
    if (open <= 0.0) { continue; }
    let e = L.rgb.rgb * open;
    o.open += e;
    each[i] = e;
    let w = dot(e, LUM);
    if (w > wTop) { wTop = w; top = i; }
    wAll += w;
  }
  if (top < 0) { return o; }
  // Whether the lights are hidden. A march costs far more than everything above, so there are two a
  // frame however many lights there are: one towards the light that gives this surface the most, and one
  // towards one of the rest, chosen by chance in proportion to what each gives (its answer then stands for
  // all of the rest: right on average, and with no grain at all where none of them is hidden).
  let visTop = lightVis(p, n, top);
  var rest = o.open - each[top];
  let wRest = wAll - wTop;
  var visRest = 1.0;
  if (wRest > 1e-6 * wAll) {
    var pick = walk(rnd()) * wRest;
    var c = -1;
    for (var i = 0; i < 12; i++) {
      if (i >= count) { break; }
      if (i == top) { continue; }
      let w = dot(each[i], LUM);
      if (w <= 0.0) { continue; }
      c = i;
      pick -= w;
      if (pick <= 0.0) { break; }
    }
    if (c >= 0) { visRest = lightVis(p, n, c); }
  }
  o.lit = each[top] * visTop + rest * visRest;
  return o;
}

// Whether light i reaches p: a march towards a point of it taken at random (moved on frame by frame).
fn lightVis(p: vec3f, n: vec3f, i: i32) -> f32 {
  let L = s.lights[i];
  let soft = max(L.v.w, 1e-3);
  let q2 = seq2();
  let j = walk(rnd());
  let x = (2.0 * q2.x - 1.0) * L.a.w * (1.0 - 0.5 * soft);
  let y = (2.0 * q2.y - 1.0) * L.u.w * (1.0 - 0.5 * soft);
  let d = fromEnv(normalize(L.a.xyz + x * L.u.xyz + y * L.v.xyz));
  if (dot(n, d) <= 0.0) { return 1.0; }
  return shadowTo(p, n, d, 0.02, 1e9, j);
}

// The lamps (lamps.wgsl): a random point of each (a ball's or a tube's middle line, a panel's face), its
// light as lampsDiffuse gathers it, and a march towards that point that stops there.
fn fromLamps(p: vec3f, n: vec3f) -> Direct {
  var o = Direct(vec3f(0.0), vec3f(0.0));
  if (s.k.w < 0.5) { return o; }
  let count = i32(lamps.n.x);
  for (var i = 0; i < 8; i++) {
    if (i >= count) { break; }
    let L = lamps.l[i];
    let q2 = seq2();
    let r1 = 2.0 * q2.x - 1.0;
    let r2 = 2.0 * q2.y - 1.0;
    let j = walk(rnd());
    var at = L.p.xyz;
    var size = L.u.w;
    if (L.p.w > 1.5) {
      at += L.u.xyz * (r1 * L.u.w) + L.v.xyz * (r2 * L.v.w);
      size = min(L.u.w, L.v.w);
    } else if (L.p.w > 0.5) {
      at += L.u.xyz * (r1 * L.v.w);
    }
    let d = at - p;
    let d2 = dot(d, d);
    let dist = sqrt(d2);
    let l = d / max(dist, 1e-6);
    let cosn = dot(n, l);
    if (cosn <= 0.0) { continue; }
    var area = 3.14159265 * L.u.w * L.u.w;
    if (L.p.w > 0.5 && L.p.w < 1.5) {
      let along = dot(l, L.u.xyz);
      area += 4.0 * L.u.w * L.v.w * sqrt(max(1.0 - along * along, 0.0));
    } else if (L.p.w > 1.5) {
      area = 4.0 * L.u.w * L.v.w * max(-dot(l, cross(L.u.xyz, L.v.xyz)), 0.0);
    }
    let e = L.rgb.rgb * (area * cosn / max(d2, area + 1e-4));
    o.open += e;
    o.lit += e * shadowTo(p, n, l, max(0.25 * size / max(dist, 1e-3), 0.004), dist - size * 0.5, j);
  }
  return o;
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let keep = vec4f(0.0, 0.0, 0.0, 1.0);
  if (s.k0.w > 2.5 && s.k0.w < 3.5) { return vec4f(vec3f(fract(s.k0.z * 0.37)), 0.0); }
  let src = textureLoad(frameTex, vec2i(pos.xy), 0);
  if (src.a <= 0.0) { return keep; }
  // the ray of the tracer's first sample in this pixel, and the surface it met
  let res = s.cam.xy;
  let fc = vec2f(pos.x, res.y - pos.y);
  let shift = vec2f(ign(fc + 19.19), ign(fc.yx + 47.3));
  let o = fract(shift + s.cam.zw) - 0.5;
  let uv = (2.0 * (fc + o) - res) / res.y + s.cam2.xy;
  let ro = vec3f(0.0, 0.0, s.cam2.z);
  let rd = normalize(vec3f(uv * s.cam2.w, -1.0));
  // (the distance is kept as a 16-bit float: settled back onto the smooth surface)
  var t = src.a;
  t += mapNW(ro + rd * t);
  let d2 = mapNW(ro + rd * t);
  if (abs(d2) > 0.01) { return keep; }
  t += d2;
  let p = ro + rd * t;
  let n = nrm(p);
  // How much of the pixel the surface covers. The tracer sends several rays through a pixel and writes
  // their light averaged, but the distance of only one: at an outline, the others may miss. Each of them
  // is followed a little from the same distance (plain lookups): one that still lies further from the
  // surface than this pixel's own ray by more than half a pixel's width missed it. (Measured against this
  // ray's own lookup: a plain lookup is off by a fraction of a voxel near the surface, far more than a
  // pixel, but by nearly the same amount a pixel away.) The light added is weighed by the share that met it, so that
  // outlines do not glow.
  let S = i32(s.k1.x);
  var cover = 1.0;
  if (S > 1) {
    let pxw = 2.0 * s.cam2.w / res.y * t;
    var hits = 1.0;
    let here = mapW(p);
    for (var k = 1; k < 32; k++) {
      if (k >= S) { break; }
      let ok = fract(vec2f(0.7548777, 0.5698403) * f32(k) + shift + s.cam.zw) - 0.5;
      let uvk = (2.0 * (fc + ok) - res) / res.y + s.cam2.xy;
      let rdk = normalize(vec3f(uvk * s.cam2.w, -1.0));
      // (three steps along that ray from the same distance: one that meets the surface settles onto it)
      var tk = t;
      tk += mapW(ro + rdk * tk) - here;
      tk += mapW(ro + rdk * tk) - here;
      if (mapW(ro + rdk * tk) - here < 0.5 * pxw) { hits += 1.0; }
    }
    cover = hits / f32(S);
  }
  // (the pixel's own numbers stay the same from frame to frame; seq2 and walk() move them on)
  seed = pcg(u32(pos.x) + pcg(u32(pos.y) + 12345u));

  var direct = atInfinity(p, n);
  let lampLight = fromLamps(p, n);
  if (s.k0.w > 4.5) { return vec4f(select(lampLight.lit, lampLight.open, s.k0.w < 5.5) * 0.01, 0.0); }
  // (the lamps' light as matte clay always gathered it, the lights at infinity's as the nine terms do: over pi)
  let lit = direct.lit * 0.31830989 + lampLight.lit;
  let open = direct.open * 0.31830989 + lampLight.open;

  // the frame's own light: dimmed by the folds of the surface, and (a milky body) by the share of the
  // direct light that is hidden here
  var keepK = 1.0;
  if (s.k.z > 0.0) { keepK = mix(1.0, occlusion(p, n), s.k.z); }
  if (s.k.y > 0.0) {
    let lum = vec3f(0.2126, 0.7152, 0.0722);
    let lost = dot(open - lit, lum) / max(dot(open, lum), 1e-5);
    keepK *= 1.0 - s.k.y * clamp(lost, 0.0, 1.0);
  }
  // Clay: its light and dark drawn apart, as the tracer's clay always had them (light above the average
  // counts for more, below it for less), now on the room's light and the lights' direct light together.
  // The frame holds the clay's colour times the room's light: the light is read back from it by that
  // colour. The lamps' light is added afterwards, not drawn apart, as the clay always took it.
  let infLit = direct.lit * 0.31830989;
  var addK = 1.0;
  if (s.clay.x > 0.0) {
    let lum = vec3f(0.2126, 0.7152, 0.0722);
    let gathered = dot(src.rgb / max(cover, 0.1) + s.base.rgb * infLit, lum) / max(dot(s.base.rgb, lum), 1e-4);
    addK = pow(max(gathered / s.clay.x, 1e-4), s.clay.y);
    keepK *= addK;
  }
  // (for looking: the frame's own light alone, or the direct light alone)
  if (s.k0.w > 1.5) { return vec4f(0.0, 0.0, 0.0, keepK); }
  let add = s.base.rgb * (infLit * addK + lampLight.lit) * (s.k.x * cover);
  if (s.k0.w > 0.5) { return vec4f(add, 0.0); }
  return vec4f(add, keepK);
}
