
// ---- shadows and caustics cast by the bodies ------------------------------------------------------------
// Put into the tracer only while something casts (cast.rs: `weave`); without it the tracer's text is what
// it always was. Everything here is worked out at the point being shaded and nothing is carried across the
// tracer's marching.
//
// A surface that gathers light diffusely (clay, a milky body) asks castAt(p, n) what the other bodies, and
// its own, do to that light: a number per colour to multiply the gathered light by. 1 = nothing in the
// way; below 1 = a body stands between the surface and one of the lights; above 1, and coloured = light
// arrives that a clear body bent and focused onto this place. A floor under the shape asks castFloor(p).

struct CastLightU {
  dir: vec4f,                 // xyz: towards the light, in the object's own frame   w: the tangent of the angle it is taken to cover
  rgb: vec4f,                 // rgb: its light, times the strength of shadows
  t0: vec4f,                  // what a white matte surface gathers from it at unit brightness, as nine terms in
  t1: vec4f,                  // the lights' own frame:  1, y, z, x | xy, yz, 3zz - 1, xz | xx - yy
  t2: vec4f,
}
struct CastU {
  k: vec4f,                   // x: how many lights cast   y: strength of caustics   z: how far off the surface a march starts
                              // w: the shortest stride of a march
  total: array<vec4f, 9>,     // what a white matte surface gathers from the whole lighting, as the same nine terms
  body: array<vec4f, 4>,      // each body: rgb = what a unit length of it takes from light passing through, w = what its surfaces let through
  casts: vec4f,               // for each body: 1 = it casts a shadow, 0 = light passes as if it were not there
  solid: vec4f,               // for each body: 1 = it lets nothing through at all
  far: vec4f,                 // for each body: 0 if it casts, a large distance if not (so that it does not shorten the strides)
  k2: vec4f,                  // x: voxels a side of the volume of arriving light (0 = nothing makes caustics)
                              // y: how far off the surface that light was landed (and is read)
                              // z: how many maps of light on the floor (each twice as wide as the one before)
                              // w: the finest of them that is made (1 while things move)
  room0: vec4f,               // the room (in which a floor lies level) -> what the camera sees, as three rows
  room1: vec4f,
  room2: vec4f,
  floor: vec4f,               // x: the floor's height in the room   y: half the width of its finest map of arriving light
                              // z: 1 = there is a floor   w: 1 = caustics are landed on it
  floorTotal: vec4f,          // rgb: the light an unshadowed floor gathers from the whole lighting
                              // w: the share of it that comes from all round (not from the lights), times the strength of shadows
  floorShare: array<vec4f, 12>,      // rgb: of that, each light's own share (times the strength of shadows)
  lights: array<CastLightU, 12>,
  veil: vec4f,                // for each body: how much of the light from all round it hides from a floor beneath it
  floorAlbedo: vec4f,         // rgb: the floor's albedo (what it gives back of the light bodies focus onto it)
}
@group(1) @binding(0) var<uniform> cu: CastU;
@group(1) @binding(1) var castBodies: texture_3d<f32>;      // the distance to each body, one a channel, at modest detail
@group(1) @binding(2) var castLight: texture_3d<f32>;       // light arriving by way of the bodies (caustics), in the object's frame
@group(1) @binding(3) var castSmp: sampler;
@group(1) @binding(4) var castOnFloor: texture_2d_array<f32>;     // the same on the floor: maps of its plane, in the room's frame

const CAST_STEPS: i32 = 40;
const CAST_STRIDE: f32 = 0.7;         // the longest stride

// How much of a light the bodies let through to the point ro (in the object's frame), per colour.
// rd: towards the light.  w: the tangent of the angle the light covers.
//
// One march towards the light through the distances to the bodies. A light of some size is hidden
// gradually: a body that the ray passes at distance d, a length t along it, covers the light's disc
// (which at that range is w t across) by how d compares with w t, from d = w t (clear of it) through
// d = 0 (half) to d = -w t (all of it). The nearest pass decides (the estimate of I. Quilez, in the form
// that also follows the ray inside the body). Nothing is random, so the shadow has no grain; it is sharp
// where the body touches the surface and widens with distance as a real one does.
// A body that lets some light through (milky, or clear and not casting caustics) is not simply in the
// way: the length of the ray inside it is measured, and the light is dimmed and coloured by that.
// (a light at a place, not at infinity: the march stops where it stands. Set only around a lamp's march.)
var<private> castMax: f32 = 1e9;
fn castShadow(ro: vec3f, rd: vec3f, w: f32) -> vec3f {
  let b = dot(ro, rd);
  let h = b * b - (dot(ro, ro) - BOUND * BOUND);
  if (h <= 0.0) { return vec3f(1.0); }
  let tFar = -b + sqrt(h);
  if (tFar <= 0.0) { return vec3f(1.0); }
  // The first look is taken near enough that the surface the march starts from does not count as cover;
  // from a point outside the scene (a floor's), where the ray enters the ball that holds the bodies.
  var t = max(cu.k.z / max(w, 0.2), -b - sqrt(h));
  var r = vec4f(1.0);
  var s = vec4f(0.0);
  for (var i = 0; i < CAST_STEPS; i++) {
    let d = textureSampleLevel(castBodies, castSmp, (ro + rd * t) / (2.0 * HB) + 0.5, 0.0);
    r = min(r, d / (w * t));
    // wholly behind something that lets nothing through: no need to look further
    if (any(r * cu.solid <= vec4f(-1.0))) { break; }
    // on by as far as the nearest casting body is (inside one: as far as its surface)
    let a = abs(d) + cu.far;
    let dt = clamp(min(min(a.x, a.y), min(a.z, a.w)), cu.k.w, CAST_STRIDE);
    s += dt * step(d, vec4f(0.0));
    t += dt;
    if (t > tFar || t > castMax) { break; }
  }
  let x = clamp(0.5 - 0.5 * r, vec4f(0.0), vec4f(1.0));
  let cover = x * x * (3.0 - 2.0 * x) * cu.casts;
  var v = mix(vec3f(1.0), exp(-cu.body[0].rgb * s.x) * cu.body[0].w, cover.x);
  v *= mix(vec3f(1.0), exp(-cu.body[1].rgb * s.y) * cu.body[1].w, cover.y);
  v *= mix(vec3f(1.0), exp(-cu.body[2].rgb * s.z) * cu.body[2].w, cover.z);
  v *= mix(vec3f(1.0), exp(-cu.body[3].rgb * s.w) * cu.body[3].w, cover.w);
  return v;
}

// The light arriving at q (in the object's frame) on a surface facing n, by way of the bodies: read from
// its volume at four places round q, half a voxel off along the surface. One plain lookup shows the sharp
// edge of a caustic as a small step at every voxel; four spread along the surface smooth the steps away.
// (Along the surface only: a smoother reading in every direction also reaches voxels off the surface,
// which hold no light, and a quarter of the caustic's light went missing.)
fn castArriving(q: vec3f, n: vec3f) -> vec3f {
  let res = cu.k2.x;
  if (res < 2.0) { return vec3f(0.0); }
  // two directions along the surface
  let t1 = normalize(cross(n, select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.x) > 0.7)));
  let t2 = cross(n, t1);
  let h = HB / res;                                 // half a voxel
  let c = q / (2.0 * HB) + 0.5;
  let a = (t1 + t2) * (h / (2.0 * HB));
  let b = (t1 - t2) * (h / (2.0 * HB));
  return 0.25 * (textureSampleLevel(castLight, castSmp, c + a, 0.0).rgb + textureSampleLevel(castLight, castSmp, c - a, 0.0).rgb
               + textureSampleLevel(castLight, castSmp, c + b, 0.0).rgb + textureSampleLevel(castLight, castSmp, c - b, 0.0).rgb);
}

// The diffuse light at p (a point in the camera's frame, as the tracer has it) on a surface facing n, in parts:
//   total    the light a matte surface facing n gathers from the whole lighting (nine terms)
//   lost     of that, each light's own share times how much of the light is hidden from p
//   added    light that arrives at p by way of a clear or reflecting body (looked up in a volume that
//            is made beforehand by following light forward from the lights: cast_light.wgsl)
// A light's share is worked out by the same nine-term arithmetic as the total, so hiding every light
// leaves exactly the light from everywhere, and never less than nothing.
struct CastParts {
  total: vec3f,
  lost: vec3f,
  added: vec3f,
}
fn castParts(p: vec3f, n: vec3f) -> CastParts {
  let e = toEnv(n);
  let w0 = vec4f(1.0, e.y, e.z, e.x);
  let w1 = vec4f(e.x * e.y, e.y * e.z, 3.0 * e.z * e.z - 1.0, e.x * e.z);
  let w2 = e.x * e.x - e.y * e.y;
  let total = max(cu.total[0].rgb + cu.total[1].rgb * e.y + cu.total[2].rgb * e.z + cu.total[3].rgb * e.x
                + cu.total[4].rgb * w1.x + cu.total[5].rgb * w1.y + cu.total[6].rgb * w1.z + cu.total[7].rgb * w1.w
                + cu.total[8].rgb * w2, vec3f(0.0));
  let ro = toObj(p) + toObj(n) * cu.k.z;
  var lost = vec3f(0.0);
  let count = i32(cu.k.x);
  for (var i = 0; i < 12; i++) {
    if (i >= count) { break; }
    let share = dot(cu.lights[i].t0, w0) + dot(cu.lights[i].t1, w1) + cu.lights[i].t2.x * w2;
    if (share <= 0.0) { continue; }
    lost += cu.lights[i].rgb.rgb * share * (vec3f(1.0) - castShadow(ro, cu.lights[i].dir.xyz, cu.lights[i].dir.w));
  }
  let no = toObj(n);
  let added = castArriving(toObj(p) + no * cu.k2.y, no) * cu.k.y;
  return CastParts(total, lost, added);
}

// What the bodies do to the diffuse light gathered at p by a surface facing n: a number per colour to
// multiply it by.
fn castAt(p: vec3f, n: vec3f) -> vec3f {
  let c = castParts(p, n);
  return max(c.total - c.lost + c.added, vec3f(0.0)) / max(c.total, vec3f(1e-5));
}

// How much of the sky above a floor point ro (in the object's frame; up: the floor's up there) the bodies
// that cast hide, 0 to 1, each by its veil (cu.veil): the distance to them asked at a few heights straight up, against each height
// (as a matte surface's own folds are judged). Under a body it is most, falling off softly around it.
fn castUnder(ro: vec3f, up: vec3f) -> f32 {
  var shut = 0.0;
  var w = 1.0;
  var ws = 0.0;
  for (var i = 0; i < 5; i++) {
    let h = 0.08 + 0.25 * f32(i * i) * 0.25 + 0.1 * f32(i);
    let d = textureSampleLevel(castBodies, castSmp, (ro + up * h) / (2.0 * HB) + 0.5, 0.0) + cu.far;
    // each body hides what it hides of that light (clear glass little, a solid body all of it)
    let c = clamp((vec4f(h) - d) / h, vec4f(0.0), vec4f(1.0)) * cu.veil;
    shut += w * max(max(c.x, c.y), max(c.z, c.w));
    ws += w;
    w *= 0.8;
  }
  return clamp(shut / ws, 0.0, 1.0);
}

// What the bodies do to the light on a floor under the shape, at p: a point of the floor in the room's
// own frame (the floor is the plane y = cu.floor.x there). A number per colour to multiply the floor's
// unshadowed colour by, as castAt gives for a body.
// The floor is flat and all its light comes from far away, so every point of it gathers the same light:
// that light, and each light's share of it, are worked out beforehand (cu.floorTotal, cu.floorShare).
// Light that arrives by way of the bodies is read from a map of the floor's plane, which is far finer
// than the volume the bodies read theirs from.
fn castFloor(p: vec3f) -> vec3f {
  if (cu.floor.z < 0.5) { return vec3f(1.0); }
  // the point and the floor's normal as the camera sees them, then in the object's frame
  let pw = vec3f(dot(cu.room0.xyz, p), dot(cu.room1.xyz, p), dot(cu.room2.xyz, p));
  let nw = vec3f(cu.room0.y, cu.room1.y, cu.room2.y);
  let ro = toObj(pw) + toObj(nw) * cu.k.z;
  var lost = vec3f(0.0);
  let count = i32(cu.k.x);
  for (var i = 0; i < 12; i++) {
    if (i >= count) { break; }
    let share = cu.floorShare[i].rgb;
    if (share.x + share.y + share.z <= 0.0) { continue; }
    lost += share * (vec3f(1.0) - castShadow(ro, cu.lights[i].dir.xyz, cu.lights[i].dir.w));
  }
  let total = cu.floorTotal.rgb;
  // the light from all round, hidden where bodies stand over the floor: a soft pool of shade under them
  lost += total * (cu.floorTotal.w * castUnder(ro, toObj(nw)));
  return max(total - lost, vec3f(0.0)) / max(total, vec3f(1e-5));
}

// The light the bodies focus onto the floor at p (room's frame), as the floor gives it back: added to
// the floor's colour, not a multiple of it. (It was a multiple once, castFloor's, against the light the
// floor gathers from the lights at infinity: the light a lamp focuses then vanished whenever those were
// dim or off, and the lamp the only light.)
fn castFloorLight(p: vec3f) -> vec3f {
  if (cu.floor.z < 0.5 || cu.floor.w < 0.5) { return vec3f(0.0); }
  var added = vec3f(0.0);
  // the floor's maps all have the same texels a side, each covering twice the width of the one before: the
  // finest that reaches this point, with a margin (light further out than 0.9 of a map's half width was
  // landed in the next). The maps are moved by a part of a texel for each making that refines a still
  // picture (room1.w, room2.w), so that the average of the frames shows no texels.
  let x = p.xz - vec2f(cu.room1.w, cu.room2.w);
  let lv = min(max(ceil(log2(max(max(abs(x.x), abs(x.y)), 1e-6) / (0.9 * cu.floor.y))), cu.k2.w), cu.k2.z - 1.0);
  let q = x / (2.0 * cu.floor.y * exp2(lv));
  if (cu.floor.w > 0.5 && max(abs(q.x), abs(q.y)) < 0.5) {
    added = textureSampleLevel(castOnFloor, castSmp, q + 0.5, i32(lv), 0.0).rgb * cu.k.y;
  }
  return added * cu.floorAlbedo.rgb;
}

// For checks and for looking (GLASSFORMS_CAST_FLOOR): a plain matte floor drawn where a ray from the camera
// misses the shape, lit as castFloor says. It fades out towards the edge of its map.
fn castFloorSeen(ro: vec3f, rd: vec3f) -> vec3f {
  if (cu.floor.z < 0.5) { return vec3f(0.0); }
  let o = ro.x * cu.room0.xyz + ro.y * cu.room1.xyz + ro.z * cu.room2.xyz;
  let d = rd.x * cu.room0.xyz + rd.y * cu.room1.xyz + rd.z * cu.room2.xyz;
  let tf = (cu.floor.x - o.y) / d.y;
  if (!(tf > 0.0)) { return vec3f(0.0); }
  let p = o + d * tf;
  let reach = cu.floor.y * exp2(max(cu.k2.z, 1.0) - 1.0);
  let edge = 1.0 - smoothstep(0.75 * reach, 0.98 * reach, length(p.xz));
  if (edge <= 0.0) { return vec3f(0.0); }
  return 0.7 * (cu.floorTotal.rgb * castFloor(p) + castFloorLight(p) / max(cu.floorAlbedo.rgb, vec3f(1e-5))) * edge;
}
