// Caustics: light followed forward from one light, through the bodies that bend or reflect it, to where it
// lands on a body that gathers light diffusely.
//
// The light is far away, so its rays arrive parallel: a square grid of them, fixed in the light's frame
// (nothing is random, so nothing flickers), is sent through the scene. Each ray is followed at the two
// ends of the spectrum, since a clear body bends each colour by its own amount.
//
//   cs_follow   one ray of the grid: where each of its wavelengths lands, and how much of it is left
//   cs_land     one cell of the grid (four neighbouring rays): the light that entered through the cell is
//               spread over where its corners landed, and added into a volume of arriving light
// Every light's grid is worked in the same pass (a ray's third number says which light it is of): a pass
// takes as long as its slowest ray however few rays it has, so a pass a light cost as much for a small
// lamp as for the key light.
//
// Why cells and not single rays: the corners of a cell say where the whole patch of light between them
// went. Where a body gathers the light the patch lands small and bright, where it spreads the light the
// patch lands large and dim, and filling the landed patch evenly (with as many points as its size asks
// for) gives the smooth light of a real caustic from a number of rays that would otherwise show as dots.
// The same is done between neighbouring wavelengths, so colours run into each other as in a spectrum
// instead of standing apart as a few coloured copies. Corners that went different ways (one missed, or
// they landed on different bodies) say nothing about the patch between them: that light is landed where
// its own ray went, softened.

const HB: f32 = 1.5;                 // the volumes span [-HB, HB]
const BOUND: f32 = 1.47;             // nothing lies further from the middle than this
const BIG: f32 = 1.0e3;
const EVENTS: i32 = 8;               // surfaces a ray is followed across at most
const FAR_APART: f32 = 0.3;          // corners that landed further apart than this did not land together
const FILL: f32 = 2.0;               // points a side with which a landed patch is filled, at most

struct BodyL {
  a: vec4f,          // x: index of refraction at 550 nm   y: n(400 nm) - n(700 nm)
                     // z: what it does to light that nothing has bent yet: 0 = nothing (as if it were not there),
                     //    1 = bends it, 2 = reflects it, 3 = stops it
                     // w: what it does to light already bent or reflected: the same, and 4 = receives it
  absorb: vec4f,     // rgb: what a unit length of it takes from light passing through
  tint: vec4f,       // rgb: what it reflects, when it reflects
  mask: vec4f,       // which channel of the bodies' volume holds its distance
}
struct LU {
  body: array<BodyL, 4>,
  k: vec4f,          // x: voxels a side of the bodies' volume   y: a stride as a share of the distance read
                     // z: strides at most outside the bodies   w: and inside one
  lev: vec4f,        // x: voxels a side of the finest volume of arriving light   y: how many volumes, each half the size
                     // z: wavelengths followed (1 = no body splits colours)   w: how far off a surface light is landed
  soft: vec4f,       // x: the least level at which light is landed whose neighbours went elsewhere
                     // y: one over the brightest light's brightness (landed light is counted against it)
                     // z: voxels a side of the volume of modest detail
                     // w: the finest of the floor's maps that light is landed in (0, or 1 while things move)
  gone0: vec4f,      // for each body: BIG if light not yet bent passes it as if it were not there
  gone1: vec4f,      // the same for light already bent
  fl0: vec4f,        // a floor under the shape, in the object's frame. xyz: a point of it   w: 1 = there is one
  fl1: vec4f,        // xyz: the way up from it   w: half the width of its map of arriving light
  fl2: vec4f,        // xyz: one way along it   w: texels a side of the finest map
  fl3: vec4f,        // xyz: the other way along it   w: how many maps, each half the size
  bands: array<vec4f, 16>,     // the spectrum in 16 bands. rgb: the colour of each (together they make white)
                               // w: how far its bending lies from that of 400 nm (0) towards that of 700 nm (1)
}
struct One {
  dir: vec4f,        // xyz: towards the light, in the object's frame   w: tangent of the angle the light is taken to cover
  u: vec4f,          // xyz: one way across the grid of rays   w: half the grid's width
  v: vec4f,          // xyz: the other way   w: rays a side
  rgb: vec4f,        // rgb: the light it sends through a unit of area facing it, as the brightness of a white
                     //      surface there   w: where its rays' landings begin in `hits`
  at: vec4f,         // a lamp: xyz where it stands (object's frame), w = 1; its rays spread from there through
                     // the grid, which then lies across the middle of the scene (w = 0: a light at infinity;
                     // w = 2: a lamp among the bodies, its rays going every way)
}
struct Ones {
  list: array<One, 16>,
}
// where one wavelength of one ray landed
struct Hit {
  x: f32, y: f32, z: f32,
  sig: u32,          // the way it went (0 = it did not land): rays that went the same way have the same number
  col: u32,          // how much of it is left, per colour (10 bits each)
  travel: f32,       // how far it flew since it was last bent
}

@group(0) @binding(0) var<uniform> lu: LU;
@group(0) @binding(1) var<uniform> ones: Ones;
@group(0) @binding(2) var fineTex: texture_3d<f32>;       // the distance to each body, at full detail
@group(0) @binding(3) var smp: sampler;
@group(0) @binding(4) var<storage, read_write> hits: array<Hit>;
@group(0) @binding(5) var<storage, read_write> acc: array<atomic<i32>>;      // landed light: whole numbers, three a voxel
@group(0) @binding(6) var coarseTex: texture_3d<f32>;     // the distance to each body at modest detail, one body a channel

// ---- the bodies ------------------------------------------------------------------------------------------

// the distance to each body at p
fn dist(p: vec3f) -> vec4f {
  let s = textureSampleLevel(fineTex, smp, p / (2.0 * HB) + 0.5, 0.0);
  return vec4f(dot(s, lu.body[0].mask), dot(s, lu.body[1].mask), dot(s, lu.body[2].mask), dot(s, lu.body[3].mask));
}
fn vol(t: vec3f, m: vec4f) -> f32 { return dot(textureSampleLevel(fineTex, smp, t, 0.0), m); }
// One body's distance read smoothly (a cubic B-spline through the voxels, as the tracer reads its surface):
// plain interpolation has a kink at every voxel, and a caustic shows every kink as a line.
fn smoothD(p: vec3f, m: vec4f) -> f32 {
  let res = lu.k.x;
  let c = (p / (2.0 * HB) + 0.5) * res - 0.5;
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
  return g0.z * (g0.y * (g0.x * vol(vec3f(a.x, a.y, a.z), m) + g1.x * vol(vec3f(b.x, a.y, a.z), m))
               + g1.y * (g0.x * vol(vec3f(a.x, b.y, a.z), m) + g1.x * vol(vec3f(b.x, b.y, a.z), m)))
       + g1.z * (g0.y * (g0.x * vol(vec3f(a.x, a.y, b.z), m) + g1.x * vol(vec3f(b.x, a.y, b.z), m))
               + g1.y * (g0.x * vol(vec3f(a.x, b.y, b.z), m) + g1.x * vol(vec3f(b.x, b.y, b.z), m)));
}
// the way out of a body at p
fn normalOf(p: vec3f, m: vec4f) -> vec3f {
  let e = vec2f(1.0, -1.0) * (2.0 * HB / lu.k.x);
  return normalize(e.xyy * smoothD(p + e.xyy, m) + e.yyx * smoothD(p + e.yyx, m) + e.yxy * smoothD(p + e.yxy, m) + e.xxx * smoothD(p + e.xxx, m));
}

fn iorOf(j: i32, lam: f32) -> f32 {
  return lu.body[j].a.x + lu.body[j].a.y * (1.0e6 / (lam * lam) - 3.3058) / 4.2092;
}
// unpolarised Fresnel reflectance going from index n1 into n2
fn fresnel(cosi: f32, n1: f32, n2: f32) -> f32 {
  let sint = n1 / n2 * sqrt(max(0.0, 1.0 - cosi * cosi));
  if (sint >= 1.0) { return 1.0; }
  let cost = sqrt(max(0.0, 1.0 - sint * sint));
  let rs = (n1 * cosi - n2 * cost) / (n1 * cosi + n2 * cost);
  let rp = (n2 * cosi - n1 * cost) / (n2 * cosi + n1 * cost);
  return 0.5 * (rs * rs + rp * rp);
}

// What a ray inside body m can meet next: m's own surface from within, and the bodies numbered above it
// (where bodies coincide the place belongs to the higher number, so those below are not there).
fn within(a: vec4f, m: i32) -> vec4f {
  let i = vec4i(0, 1, 2, 3);
  return select(select(a, -a, i == vec4i(m)), vec4f(BIG), i < vec4i(m));
}

// March from ro along rd, inside body m (-1: in the open). How far the next surface is, or -1.0 if the ray
// leaves the scene; `which`: the body whose surface it is. gone: BIG for the bodies that are not there.
fn march(ro: vec3f, rd: vec3f, m: i32, gone: vec4f, which: ptr<function, i32>) -> f32 {
  let b = dot(ro, rd);
  let h = b * b - (dot(ro, ro) - BOUND * BOUND);
  if (h <= 0.0) { return -1.0; }
  let tFar = -b + sqrt(h);
  let near = 0.6 / lu.k.x;                           // a fifth of a voxel
  let wide = 4.5 / lu.soft.z;                        // a voxel and a half of the volume of modest detail
  let steps = select(i32(lu.k.z), i32(lu.k.w), m >= 0);
  var t = 0.0;
  for (var i = 0; i < 512; i++) {
    if (i >= steps) { break; }
    // Far from every surface the strides are taken through the volume of modest detail: it fits in the
    // GPU's fast memory where the full one does not, and most rays never come near a body.
    var c = textureSampleLevel(coarseTex, smp, (ro + rd * t) / (2.0 * HB) + 0.5, 0.0) + gone;
    if (m >= 0) { c = within(c, m); }
    let fc = min(min(c.x, c.y), min(c.z, c.w));
    if (fc > wide) {
      t += fc - 0.5 * wide;
      if (t > tFar) { break; }
      continue;
    }
    var a = dist(ro + rd * t) + gone;
    if (m >= 0) { a = within(a, m); }
    let f = min(min(a.x, a.y), min(a.z, a.w));
    if (f < near) {
      *which = select(select(select(3, 2, a.z == f), 1, a.y == f), 0, a.x == f);
      return t;
    }
    t += f * lu.k.y;
    if (t > tFar) { break; }
  }
  return -1.0;
}

fn pack(w: vec3f) -> u32 {
  let q = vec3u(clamp(w, vec3f(0.0), vec3f(1.0)) * 1023.0 + 0.5);
  return q.x | (q.y << 10u) | (q.z << 20u);
}
fn unpack(c: u32) -> vec3f {
  return vec3f(f32(c & 1023u), f32((c >> 10u) & 1023u), f32((c >> 20u) & 1023u)) / 1023.0;
}

// One wavelength of a ray, from where it first met a body that bends or reflects light: p on that body's
// (j's) surface, arriving along d. Followed from surface to surface until it lands on a body that gathers
// light, is stopped, or leaves.
fn follow(p_in: vec3f, d_in: vec3f, j_in: i32, n_in: vec3f, lam: f32) -> Hit {
  var out = Hit(0.0, 0.0, 0.0, 0u, 0u, 0.0);
  var p = p_in;
  var d = d_in;
  var j = j_in;                 // the body whose surface the ray is at
  var m = -1;                   // the body it is inside (-1: in the open)
  var w = vec3f(1.0);           // how much of it is left
  var sig = 1u;
  var travel = 0.0;
  let off = 3.0 / lu.k.x;       // a voxel: how far past a surface the next stretch starts
  for (var e = 0; e < EVENTS; e++) {
    // the surface: body j's, met from outside (entering j) or, if the ray is inside j, from within
    let leaving = j == m;
    var n = n_in;                                  // the way out of body j (the first is given: every wavelength meets it alike)
    if (e > 0) { n = normalOf(p, lu.body[j].mask); }
    // what lies on the other side
    var next = j;
    if (leaving) {
      // out of m, into the highest body below it that is here, or into the open
      next = -1;
      let dd = dist(p) + lu.gone1;
      for (var k = 0; k < 4; k++) {
        if (k < m && dd[k] < 0.0) { next = k; }
      }
    }
    // from here on n points back to the side the ray came from
    if (dot(d, n) > 0.0) { n = -n; }
    let role = select(0, i32(lu.body[max(next, 0)].a.w), next >= 0);
    if (next >= 0 && role == 4) {
      // landed on a body that gathers light
      let q = p + n * lu.lev.w;
      out = Hit(q.x, q.y, q.z, sig * 16u + u32(next) + 11u, pack(w), travel);
      return out;
    }
    if (next >= 0 && role == 3) { return out; }
    let n1 = select(1.0, iorOf(max(m, 0), lam), m >= 0);
    let cosi = clamp(-dot(d, n), 0.0, 1.0);
    if (next >= 0 && role == 2) {
      // a reflecting body: back the way it came from, coloured
      w *= lu.body[next].tint.rgb;
      d = reflect(d, n);
      p += n * off;
      sig = sig * 16u + u32(next) + 6u;
      travel = 0.0;
    } else {
      // a clear body, or the open: through, unless it is turned back whole
      let n2 = select(1.0, iorOf(max(next, 0), lam), next >= 0);
      let F = fresnel(cosi, n1, n2);
      if (F > 0.999) {
        d = reflect(d, n);
        p += n * off;
        sig = sig * 16u + 10u;
      } else {
        w *= 1.0 - F;
        d = refract(d, n, n1 / n2);
        p -= n * off;
        m = next;
        sig = sig * 16u + u32(next + 2);
        travel = 0.0;
      }
    }
    if (max(w.x, max(w.y, w.z)) < 0.004) { return out; }
    // on to the next surface
    var which = 0;
    let t = march(p, d, m, lu.gone1, &which);
    if (t < 0.0) {
      // It leaves the bodies. If there is a floor below and it is heading down onto it, it lands there:
      // where, in the floor's own two directions (a floor's light is kept as a map of its plane).
      let down = -dot(d, lu.fl1.xyz);
      if (lu.fl0.w > 0.5 && m < 0 && down > 1e-4) {
        let tf = dot(p - lu.fl0.xyz, lu.fl1.xyz) / down;
        let q = p + d * tf - lu.fl0.xyz;
        let at = vec2f(dot(q, lu.fl2.xyz), dot(q, lu.fl3.xyz));
        if (tf > 0.0 && max(abs(at.x), abs(at.y)) < lu.fl1.w) {
          out = Hit(at.x, at.y, 0.0, sig * 16u + 15u, pack(w), travel + tf);
        }
      }
      return out;
    }
    if (m >= 0) { w *= exp(-lu.body[m].absorb.rgb * t); }
    travel += t;
    p += d * t;
    // settled onto the smooth surface (two steps, as the tracer does)
    let sgn = select(1.0, -1.0, which == m);
    p += d * (sgn * smoothD(p, lu.body[which].mask));
    p += d * (sgn * smoothD(p, lu.body[which].mask));
    j = which;
  }
  return out;
}

@compute @workgroup_size(8, 8, 1)
fn cs_follow(@builtin(global_invocation_id) id: vec3u) {
  let one = ones.list[id.z];
  let n = u32(one.v.w);
  if (id.x >= n || id.y >= n) { return; }
  let W = u32(lu.lev.z);
  let base = u32(one.rgb.w) + (id.y * n + id.x) * W;
  let none = Hit(0.0, 0.0, 0.0, 0u, 0u, 0.0);
  for (var k = 0u; k < W; k++) { hits[base + k] = none; }
  // this ray of the grid: parallel to all the others, from the light's side of the scene
  let g = (vec2f(id.xy) + 0.5) / f32(n) * 2.0 - 1.0;
  var d = -one.dir.xyz;
  var o = (g.x * one.u.xyz + g.y * one.v.xyz) * one.u.w + one.dir.xyz * BOUND;
  if (one.at.w > 1.5) {
    // a lamp among the bodies: every way from where it stands, the grid laid over the sphere of
    // directions so that every ray stands for the same solid angle (round its axis, and evenly in height)
    let ph = 3.14159265 * g.x;
    let r = sqrt(max(1.0 - g.y * g.y, 0.0));
    d = r * cos(ph) * one.u.xyz + r * sin(ph) * one.v.xyz + g.y * one.dir.xyz;
    o = one.at.xyz;
  } else if (one.at.w > 0.5) {
    // a lamp: from where it stands, through this point of the grid across the middle
    d = normalize((g.x * one.u.xyz + g.y * one.v.xyz) * one.u.w - one.at.xyz);
    o = one.at.xyz;
  }
  let b = dot(o, d);
  let h = b * b - (dot(o, o) - BOUND * BOUND);
  if (h <= 0.0) { return; }
  let start = o + d * max(-b - sqrt(h), 0.0);
  // to the first body that is there for light: only one that bends or reflects it starts a caustic
  var j = 0;
  let t = march(start, d, -1, lu.gone0, &j);
  if (t < 0.0) { return; }
  let first = i32(lu.body[j].a.z);
  if (first != 1 && first != 2) { return; }
  var p = start + d * t;
  p += d * smoothD(p, lu.body[j].mask);
  p += d * smoothD(p, lu.body[j].mask);
  let n0 = normalOf(p, lu.body[j].mask);
  for (var k = 0u; k < W; k++) {
    let lam = select(550.0, 400.0 + 300.0 * f32(k) / f32(max(W, 2u) - 1u), W > 1u);
    hits[base + k] = follow(p, d, j, n0, lam);
  }
}
