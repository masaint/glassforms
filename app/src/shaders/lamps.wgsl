// Lamps standing in the scene (core/src/lamps.rs, app/src/lamps.rs): put into a tracer only while there
// is a lamp; without one the tracer's text is what it always was. Everything here is in the camera's
// frame, as the tracer has it.
//
// A ray that leaves a surface for good looks the lights up by its direction alone (env), which is right
// for lights at infinity. A lamp is not at infinity: whether the ray meets it depends on where the ray
// starts, which is why a lamp near the shape is seen in a different place on each part of it. So every
// lookup of the lights also tests the ray against each lamp, from where it set out: gRo, set wherever
// the tracer works out a surface's normal (a ray always leaves from a point whose normal was just found).

struct LampU {
  p: vec4f,                 // xyz: where it stands;  w: its kind (0 ball, 1 tube, 2 panel)
  u: vec4f,                 // xyz: its own x;  w: a ball's or a tube's radius, a panel's half width
  v: vec4f,                 // xyz: its own y;  w: a tube's half length, a panel's half height
  rgb: vec4f,               // its radiance;  w: the share of its size over which its edge fades
}
struct LampsU {
  n: vec4f,                 // x: how many lamps;  y: which of them the camera sees (bits);
                            // z: how large a lamp is taken to be for the shadows it casts (castSoft)
  floor: vec4f,             // rgb: what a floor gives back of the light falling on it (its colour)
                            // w: how far from the room's middle it has melted away (0: there is no floor)
  plane: vec4f,             // the floor as the camera sees it: xyz the way up from it, w its height along that
  l: array<LampU, 8>,
}
@group(0) @binding(44) var<uniform> lamps: LampsU;
var<private> gRo: vec3f;

// Lamp i met by the ray from ro along rd (a unit vector) before tMax: how much of the lamp covers the
// ray (0 to 1, less for a widened ray, by as much as the lamp's light is spread over more), and how far
// along. spread: the half-angle (radians) a rough or milky surface widens the ray by.
fn lampHit(i: i32, ro: vec3f, rd: vec3f, tMax: f32, spread: f32) -> vec2f {
  let L = lamps.l[i];
  let soft = max(L.rgb.w, 0.02);
  if (L.p.w < 0.5) {
    // a ball: how near the ray passes its middle
    let oc = L.p.xyz - ro;
    let b = dot(oc, rd);
    if (b <= 0.0 || b > tMax) { return vec2f(0.0); }
    let q = max(dot(oc, oc) - b * b, 0.0);
    let r = L.u.w + spread * b;
    let x = sqrt(q) / r;
    return vec2f(clamp((1.0 - x) / soft, 0.0, 1.0) * (L.u.w * L.u.w) / (r * r), b - sqrt(max(r * r - q, 0.0)));
  }
  if (L.p.w < 1.5) {
    // a tube: the point of its axis nearest the ray (held to its length), then as a ball there
    let ax = L.u.xyz;
    let oc = L.p.xyz - ro;
    let a1 = dot(rd, ax);
    let den = 1.0 - a1 * a1;
    var t = dot(oc, rd);
    if (den > 1e-5) { t = (dot(oc, rd) - a1 * dot(oc, ax)) / den; }
    let h = clamp(dot(ro + rd * t - L.p.xyz, ax), -L.v.w, L.v.w);
    let c = L.p.xyz + ax * h - ro;
    let tc = dot(c, rd);
    if (tc <= 0.0 || tc > tMax) { return vec2f(0.0); }
    let q = max(dot(c, c) - tc * tc, 0.0);
    let r = L.u.w + spread * tc;
    let x = sqrt(q) / r;
    return vec2f(clamp((1.0 - x) / soft, 0.0, 1.0) * L.u.w / r, tc - sqrt(max(r * r - q, 0.0)));
  }
  // a panel: lit on its front only (the way its own z faces)
  let n = cross(L.u.xyz, L.v.xyz);
  let dn = dot(rd, n);
  if (dn >= 0.0) { return vec2f(0.0); }
  let t = dot(L.p.xyz - ro, n) / dn;
  if (t <= 0.0 || t > tMax) { return vec2f(0.0); }
  let x = ro + rd * t - L.p.xyz;
  let grow = spread * t;
  let s = vec2f(abs(dot(x, L.u.xyz)) / (L.u.w + grow), abs(dot(x, L.v.xyz)) / (L.v.w + grow));
  let e = clamp((vec2f(1.0) - s) / soft, vec2f(0.0), vec2f(1.0));
  return vec2f(e.x * e.y * (L.u.w * L.v.w) / ((L.u.w + grow) * (L.v.w + grow)), t);
}

// Every lamp along the ray: their light (rgb), and how much of the ray they cover together (a).
fn lampsOn(ro: vec3f, rd: vec3f, tMax: f32, spread: f32) -> vec4f {
  var c = vec4f(0.0);
  let n = i32(lamps.n.x);
  for (var i = 0; i < 8; i++) {
    if (i >= n) { break; }
    let h = lampHit(i, ro, rd, tMax, spread);
    c += vec4f(lamps.l[i].rgb.rgb, 1.0) * h.x;
  }
  return c;
}

// The lights along rd from where the ray set out: what lies behind a lamp is hidden by it.
fn lampsEnv(far: vec3f, rd: vec3f) -> vec3f {
  let c = lampsOn(gRo, rd, 1e9, 0.0);
  return (far + lampsFloor(rd)) * (1.0 - min(c.a, 1.0)) + c.rgb;
}

// The pools of the lamps' light on a floor, as a ray leaving the glass downwards sees them. The glass sees
// the floor by direction, painted into the lights' map; a lamp's pool has a place, so where the ray meets
// the floor is worked out and the pool there added (unshadowed: the shadows of the bodies on it are seen
// on the floor itself), melting away with the floor far out.
fn lampsFloor(rd: vec3f) -> vec3f {
  if (lamps.floor.w <= 0.0) { return vec3f(0.0); }
  let up = lamps.plane.xyz;
  let dn = dot(rd, up);
  if (dn > -1e-4) { return vec3f(0.0); }
  let t = (lamps.plane.w - dot(gRo, up)) / dn;
  if (t <= 0.0) { return vec3f(0.0); }
  let x = gRo + rd * t;
  let across = length(x - up * dot(x, up));
  let melt = 1.0 - smoothstep(0.35 * lamps.floor.w, lamps.floor.w, across);
  return lamps.floor.rgb * lampsPool(x, up) * (0.31830989 * melt);
}

// The light a matte surface at p facing n gathers from the lamps, unshadowed (lampsDiffuse, without the
// march towards each lamp).
fn lampsPool(p: vec3f, n: vec3f) -> vec3f {
  var e = vec3f(0.0);
  let count = i32(lamps.n.x);
  for (var i = 0; i < 8; i++) {
    if (i >= count) { break; }
    let L = lamps.l[i];
    let d = L.p.xyz - p;
    let d2 = dot(d, d);
    let l = d * inverseSqrt(max(d2, 1e-8));
    let cosn = dot(n, l);
    if (cosn <= 0.0) { continue; }
    let area = 3.14159265 * L.u.w * max(L.u.w, select(0.0, L.v.w, L.p.w > 0.5));
    e += L.rgb.rgb * (area * cosn / max(d2, area + 1e-4));
  }
  return e;
}
// The same for a reading spread by sig (radians): a rough surface's, a milky body's.
fn lampsSoft(far: vec3f, rd: vec3f, sig: f32) -> vec3f {
  let c = lampsOn(gRo, rd, 1e9, sig);
  return far * (1.0 - min(c.a, 1.0)) + c.rgb;
}

// The light a matte surface at p facing n gathers from the lamps (no shadows: what is cast is cast.wgsl's).
// Each lamp is taken as the area it shows towards p: a ball's disc, a tube's outline, a panel's face
// (only from in front of it), its radiance times that area times the cosine at the surface, over the
// distance squared.
fn lampsDiffuse(p: vec3f, n: vec3f) -> vec3f {
  var e = vec3f(0.0);
  let count = i32(lamps.n.x);
  for (var i = 0; i < 8; i++) {
    if (i >= count) { break; }
    let L = lamps.l[i];
    let d = L.p.xyz - p;
    let d2 = dot(d, d);
    let l = d * inverseSqrt(max(d2, 1e-8));
    let cosn = dot(n, l);
    if (cosn <= 0.0) { continue; }
    var area = 3.14159265 * L.u.w * L.u.w;
    if (L.p.w > 0.5 && L.p.w < 1.5) {
      let along = dot(l, L.u.xyz);
      area += 4.0 * L.u.w * L.v.w * sqrt(max(1.0 - along * along, 0.0));
    } else if (L.p.w > 1.5) {
      area = 4.0 * L.u.w * L.v.w * max(-dot(l, cross(L.u.xyz, L.v.xyz)), 0.0);
    }
    // (no nearer than the lamp's own size: a surface touching it is not lit without end)
    e += L.rgb.rgb * (area * cosn / max(d2, area + 1e-4)) * lampShadow(p, n, L.p.xyz, max(L.u.w, select(0.0, L.v.w, L.p.w > 0.5)));
  }
  return e;
}

// What the camera sees of the lamps it sees along its ray (ro, rd), nearer than tMax: their light (rgb)
// and how much of the ray they cover (a); and how far the nearest is.
fn lampsSeen(ro: vec3f, rd: vec3f, tMax: f32) -> vec4f {
  var c = vec4f(0.0);
  var near = 1e9;
  let n = i32(lamps.n.x);
  let shown = u32(lamps.n.y);
  for (var i = 0; i < 8; i++) {
    if (i >= n) { break; }
    if (((shown >> u32(i)) & 1u) == 0u) { continue; }
    let h = lampHit(i, ro, rd, tMax, 0.0);
    if (h.x > 0.0) {
      c += vec4f(lamps.l[i].rgb.rgb, 1.0) * h.x;
      near = min(near, h.y);
    }
  }
  lampNear = near;
  return c;
}
var<private> lampNear: f32;

// How much of a lamp of radius r at `at` the bodies let through to p (a surface facing n). Nothing is in
// the way unless something casts: then this is replaced by a march towards the lamp (lamps.rs: SHADOW).
fn lampShadow(p: vec3f, n: vec3f, at: vec3f, r: f32) -> vec3f { return vec3f(1.0); }
