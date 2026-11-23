
// ---- several bodies, each of its own material ---------------------------------------------------------
// Put into the tracer only when the scene is of several bodies and at least one of them has a material
// of its own (render.rs, gf_core::material::bodies_parts). A scene of one material is traced as before.
//
// Beside the scene's volume there is then the bodies' volume: for every voxel the distance to each body
// taken alone. Where bodies coincide the place belongs to the one numbered highest (the later layer), so
// what a point is made of is the highest-numbered body it is inside, and air if it is inside none. A ray
// inside body m goes on until it leaves m or meets a body numbered above m; surfaces of bodies numbered
// below m that lie inside m are not there at all. A surface between two bodies is the higher one's: its
// film, its coat and its roughness are what the light meets there.
//
// This text follows a ray from body to body. What happens at a surface and inside a body is not written
// here a second time: the material's own text (trace_material.wgsl, film.wgsl, frost.wgsl) is put in
// ahead of this one, cut to the parts that any of the bodies uses, with every read of the material's block in it reading the
// numbers of body `cb`. So a film drains and mirrors the film's studio, a rough surface carries its cone
// of light and is blurred after tracing, and a milky body is lit by its own light volume, exactly as for
// a scene of one material. Where the tracer of one material chooses a part when it is built (a metal or
// glass, rough or polished), this one chooses at each surface by the numbers of the body there.
//
// Each body is clear (its own index, dispersion and tint), a metal or mirror, milky, or a thin wall
// around air; and each may have colour by wavelength, a graded index, double refraction, bubbles (cut
// into both volumes beforehand, bubbles.rs) and what lies inside (inner.wgsl: a changing tint, cracks,
// grains, fibres, a glow under pressure, which also glows where the body presses on another).
// Milky bodies choose the light volume or wavelength-dependent random walks independently.
// Bare and clear-coated dielectrics can sample their rough interface, including against another body.

struct MB {
  kind: vec4f,              // for each body: 0 clear, 1 a metal or a mirror, 2 milky, 3 a thin wall around air
  ior: vec4f,               // its index at 550 nm
  disp: vec4f,              // its dispersion (index at 400 nm less index at 700 nm)
  has: vec4f,               // what else it has, as bits (material.rs, body_bits): 1 colour by wavelength,
                            // 2 a graded index, 4 double refraction, 8 something inside it, 16 a glow under pressure
  absorb: array<vec4f, 4>,  // rgb: what its inside absorbs per unit length
  medium: array<vec4i, 4>,  // raw identity, signed interface priority, two reserved zero lanes
  b: array<MU, 4>,
  envelope: vec4f,
}
@group(0) @binding(12) var bodTex: texture_3d<f32>;
@group(0) @binding(13) var<uniform> mb: MB;

// The body whose material the material's own functions read (every read of the material's block in them reads mb.b[cb]): set
// before each of them is called.
var<private> cb: i32;

const COUNT: i32 = {{COUNT}};           // how many bodies there are: only that many channels mean anything
const EVENTS: i32 = 2 * LAYERS + 2;     // surfaces a ray that has left the first body may still cross

// does body k have this part (a bit of mb.has)
fn hasB(k: i32, bit: u32) -> bool {
  if (k < 0) { return false; }
  return (u32(mb.has[k]) & bit) != 0u;
}

//#if HOLLOW
// (whether the body in hand is a thin wall around air: its film then thins to black film)
fn isHollow() -> bool { return wallB(cb); }
//#endif

//#if SSS
// The light inside each milky body, worked out beforehand for that body alone (scatter.rs): the first
// body's at binding 24 (declared with the material's parts), the others' here.
@group(0) @binding(25) var glowTex1: texture_3d<f32>;
@group(0) @binding(26) var glowTex2: texture_3d<f32>;
@group(0) @binding(27) var glowTex3: texture_3d<f32>;
fn glowAt(x: vec3f) -> vec3f {
  if (cb == 0) { return textureSampleLevel(glowTex, volSmp, x, 0.0).rgb; }
  if (cb == 1) { return textureSampleLevel(glowTex1, volSmp, x, 0.0).rgb; }
  if (cb == 2) { return textureSampleLevel(glowTex2, volSmp, x, 0.0).rgb; }
  return textureSampleLevel(glowTex3, volSmp, x, 0.0).rgb;
}
//#endif

// The lights from a spread of directions around rd, of standard deviation sig (radians): the softened
// copies of the lights' map (envL). No spread at all reads the lights exactly as plain glass does.
// While a metal body's reflection is being followed, what it finally shows of the surroundings is the
// environment picture sharp, as a single metal's is (`mirrorSharp`: 1 + the level its first surface asks
// for; 0 = the softened surroundings, as glass sees them).
var<private> mirrorSharp: f32;
fn seenB(rd: vec3f, sig: f32) -> vec3f {
  if (sig <= 0.0) {
    if (mirrorSharp > 0.0) { return mirrorEnv(rd, mirrorSharp - 1.0); }
    return env(rd);
  }
  return envL(rd, sig);
}

// What body k's film adds to the light it reflects along d (film.wgsl): its whole studio where the camera
// meets it, its glow alone for light the film has sent back inside, and for further pieces. Nothing for
// a body with no film.
fn skyB(k: i32, d: vec3f) -> vec3f {
//#if FILM
  if (mb.b[k].film.x <= 0.0) { return vec3f(0.0); }
  cb = k;
  return skyOf(d);
//#else
  return vec3f(0.0);
//#endif
}
fn skyDimB(k: i32, d: vec3f) -> vec3f {
//#if FILM
  if (mb.b[k].film.x <= 0.0) { return vec3f(0.0); }
  cb = k;
  return skyDim(d);
//#else
  return vec3f(0.0);
//#endif
}
fn skyFarB(k: i32, d: vec3f) -> vec3f {
//#if FILM
  if (mb.b[k].film.x <= 0.0) { return vec3f(0.0); }
  cb = k;
  return skyFar(d);
//#else
  return vec3f(0.0);
//#endif
}

// the distance to each body at a point (only the first COUNT of the four mean anything)
fn dist4(p: vec3f) -> vec4f { return textureSampleLevel(bodTex, volSmp, toObj(p) / (2.0 * HB) + 0.5, 0.0); }

// What a place is made of: the highest-numbered body it is inside; -1 for the air.
fn mediumOf(v: vec4f) -> i32 {
  var m = -1;
  for (var k = 0; k < COUNT; k++) { if (v[k] < 0.0) { m = k; } }
  return m;
}
// The body whose surface is nearest (of two as near as each other, the higher).
fn nearestOf(v: vec4f) -> i32 {
  var m = 0;
  var d = v[0];
  for (var k = 1; k < COUNT; k++) { if (v[k] <= d) { d = v[k]; m = k; } }
  return m;
}
// Body k's surface normal at p, from its own distances: for a surface that is not the outside of the
// whole scene (one body's surface inside another), which the scene's volume does not hold.
fn nrmB(p: vec3f, k: i32) -> vec3f {
  let e = vec2f(1.0, -1.0) * (3.0 * HB / u.vres);
  return normalize(e.xyy * dist4(p + e.xyy)[k] + e.yyx * dist4(p + e.yyx)[k] + e.yxy * dist4(p + e.yxy)[k] + e.xxx * dist4(p + e.xxx)[k]);
}

fn max3(v: vec3f) -> f32 { return max(v.x, max(v.y, v.z)); }
// a metal or a mirror: light does not pass through it (a milky body lets light through, scattering it)
fn opaqueB(k: i32) -> bool {
  if (k < 0) { return false; }
  return mb.kind[k] > 0.5 && mb.kind[k] < 1.5;
}
// a thin wall of glass round air
fn wallB(k: i32) -> bool {
  if (k < 0) { return false; }
  return mb.kind[k] > 2.5 && mb.kind[k] < 3.5;
}
// A fired or painted body (a ceramic that lets next to nothing through: lacquer, stoneware, porcelain):
// light stops at its surface. Alone it is lit by the surface pass; among several bodies, and wherever it
// is seen through another, it is lit here (paintB).
fn paintB(k: i32) -> bool {
  if (k < 0) { return false; }
  return mb.kind[k] > 3.5;
}
//#if PAINT_BODIES
// The paint closure is in paint_bodies.wgsl.
//#else
fn paintB_seen(k: i32, p: vec3f, n: vec3f, rd: vec3f, incident: f32) -> vec3f { return vec3f(0.0); }
//#endif
// the index of what a place is made of, at a wavelength (air, and the air inside a thin wall: 1)
fn iorB(k: i32, lam: f32) -> f32 {
  if (k < 0) { return 1.0; }
  if (wallB(k)) { return 1.0; }
  return mb.ior[k] + mb.disp[k] * (1.0e6 / (lam * lam) - 3.3058) / 4.2092;
}
fn mediumB(k: i32, ior: f32) -> PbrMedium {
  if (k < 0 || wallB(k)) { return airMedium(); }
  let metadata = mb.medium[k];
  return PbrMedium(ior, 0.0, bitcast<u32>(metadata.x), metadata.y);
}

// What the surface between a (what the light is in) and b (what lies beyond; -1 is the air) sends back,
// as a colour, of light arriving at cosine c at the point q, where the surface faces n. lam: the
// wavelength whose bending is being followed. The surface is the higher-numbered body's, and so are the
// film and the coat on it.
// A rough conductor returns a directional lobe and a broad multiple-scattering lobe.
var<private> conductorBroadB: vec3f;
//#if LIVE_CONDUCTOR_FINISH
// The pixel's polishing sample is shared by a directly visible conductor and
// one reached after reflection or transmission at another surface.
var<private> finishPixelB: vec2f;

struct ConductorFinishB {
  normal: vec3f,
  tint: vec3f,
  variance: f32,
  marked: bool,
}

fn conductorFinishB(body: i32, p: vec3f, geometric: vec3f, incident: vec3f) -> ConductorFinishB {
  cb = body;
  let wear = wearAt(p, geometric);
  var normal = wear.n;
  var variance = wear.sig * wear.sig;
  if (wear.aniso > 0.0) {
    let spread = 1.41421 * mb.b[cb].surf.x;
    let facet = brushN(wear, p, incident, finishPixelB, 0, spread * 0.3, 1.0);
    normal = facet.xyz;
    variance += facet.w * facet.w;
  }
  return ConductorFinishB(normal, wear.keep * wear.tint, variance, wear.sig > 0.0 || wear.aniso > 0.0);
}
//#endif
fn surfB(a: i32, b: i32, c: f32, q: vec3f, n: vec3f, lam: f32) -> vec3f {
  conductorBroadB = vec3f(0.0);
  let o = max(a, b);
  cb = o;
  var d = 0.0;
//#if FILM
  if (mb.b[o].film.x > 0.0) { d = filmNm(q, n); }
//#endif
  let filmOnCoat = d > 0.0;
  let nc = mb.b[o].surf.z;                  // the coat's index (1 = no coat)
  var na = iorB(a, lam);
  var cc = c;
  var top = vec3f(0.0);
  if (a < 0 && nc > 1.001) {
    // from the air on to a coated body: the coat's own reflection on top (with the film, if there is
    // one, lying on the coat), and the body under it meets the coat instead of the air
    top = vec3f(fresnel(c, 1.0, nc));
//#if FILM
    if (d > 0.0) { top = filmRGB(c, d, 1.0, nc); }
//#endif
    cc = sqrt(max(1.0 - (1.0 - c * c) / (nc * nc), 0.0));
    na = nc;
    d = 0.0;
  }
  var under = vec3f(1.0);
  if (opaqueB(b)) {
//#if METAL
    // a metal or a mirror, under its film if it has one
    cb = b;
    if (a < 0 && nc > 1.001 && mb.b[b].surf.x <= 0.0 && !filmOnCoat) {
      return top + polishedCoatReturn(c, cc, nc, top);
    }
    under = metalRGB(cc, na, d);
//#if ROUGH
    if (mb.b[b].surf.x > 0.0 && d <= 0.0) {
      let energy = conductorEnergy(cc, na, under);
      under = energy.single + energy.multiple;
      conductorBroadB = (1.0 - top) * (1.0 - top) * energy.multiple;
    }
//#endif
    cb = o;
//#endif
  } else {
    let nb = iorB(b, lam);
    if (b < 0 && nc > 1.001) {
      // out through the coat into the air: both of its faces must let the light through
      let ccb = sqrt(max(1.0 - (na * na) * (1.0 - c * c) / (nc * nc), 0.0));
      under = vec3f(1.0 - (1.0 - fresnel(c, na, nc)) * (1.0 - fresnel(ccb, nc, 1.0)));
    } else if (d > 0.0) {
//#if FILM
      under = filmRGB(cc, d, na, nb);
//#endif
    } else if (wallB(o)) {
      // a thin wall of glass: both its faces together
      let F = fresnel(c, 1.0, mb.b[o].sss.w);
      under = vec3f(2.0 * F / (1.0 + F));
    } else {
      under = vec3f(fresnel(cc, na, nb));
    }
  }
  return top + (1.0 - top) * (1.0 - top) * under;
}

// From pos inside body m along dir: on to where m ends, or where a body numbered above it (which owns
// the place ahead of m) begins. Returns how far that is. q: the place. nq: the normal of the surface met
// there (not yet turned to face the ray). b: what lies beyond it (-1: the air).
fn eventB(pos: vec3f, dir: vec3f, m: i32, q: ptr<function, vec3f>, nq: ptr<function, vec3f>, b: ptr<function, i32>) -> f32 {
  let px = 2.0 * u.tan_fov / u.res.y;
  var t = 0.0;
  var distances: vec4f;
  for (var i = 0; i < 2 * STEPS_IN; i++) {
    let at = pos + dir * t;
    distances = dist4(at);
    var d = -distances[m];
    if (COUNT > 1 && m < 1) { d = min(d, distances.y); }
    if (COUNT > 2 && m < 2) { d = min(d, distances.z); }
    if (COUNT > 3 && m < 3) { d = min(d, distances.w); }
    let eps = clamp(0.25 * px * max(u.dist - at.z, 0.1), 0.00008, 0.0008);
    if (d < eps) { break; }
    t += d * STEP_SCALE;
    if (t > 2.5 * BOUND) { break; }
  }
  // Only the final step needs a body id. Strict comparison preserves surface priority at a tie.
  var j = -1;
  var nearest = -distances[m];
  for (var k = m + 1; k < COUNT; k++) {
    if (distances[k] < nearest) { nearest = distances[k]; j = k; }
  }
  let p = pos + dir * t;
  if (j >= 0) {
    *b = j;
    *nq = nrmB(p, j);
    *q = p;
    return t;
  }
  return t + leaveB(p, dir, m, q, nq, b);
}

// Body m ends at p, heading dir: beyond it is a body numbered below m that this place is also inside, or
// the air. Sets what lies beyond, the place and the normal there; returns how far the place was moved
// along dir to settle on the surface.
fn leaveB(p: vec3f, dir: vec3f, m: i32, q: ptr<function, vec3f>, nq: ptr<function, vec3f>, b: ptr<function, i32>) -> f32 {
  var lower = -1;
  if (m > 0) {
    let v = dist4(p + dir * 0.004);
    for (var k = 0; k < m; k++) { if (v[k] < 0.0) { lower = k; } }
  }
  var t = 0.0;
  if (lower < 0) {
    // the outside of the whole scene: settle on the scene's own smooth surface, as plain glass does
    t -= mapNW(p);
    t -= mapNW(p + dir * t);
    *nq = nrm(p + dir * t);
  } else {
    *nq = nrmB(p, m);
  }
  *b = lower;
  *q = p + dir * t;
  return t;
}

//#if GRIN
// Inside body m, whose index rises with depth below its own surface: the path curves towards where the
// index is higher, as travel() follows it in a body of one material, but through the body's own
// distances, and it ends where a body numbered above m begins. From pos heading *dir (left as the path
// heads where it ends); n0: the index at the surface for this wavelength. Returns the length of the
// path; q, nq and b as eventB sets them.
fn travelB(pos: vec3f, dir: ptr<function, vec3f>, m: i32, n0: f32, q: ptr<function, vec3f>, nq: ptr<function, vec3f>, b: ptr<function, i32>) -> f32 {
  cb = m;
  let ex = mb.b[m].exo;
  var p = pos;
  var d = *dir;
  var len = 0.0;
  let vox = 2.0 * HB / u.vres;
  let deep = 5.0 / ex.y;                  // below this the index no longer changes
  let hs = max(0.4 / ex.y, 1.5 * vox);    // the short step
  var j = -1;
  for (var i = 0; i < 96; i++) {
    let v = dist4(p);
    let depth = -v[m];
    // the nearest body numbered above m
    var dh = 1e9;
    var jh = -1;
    for (var k = m + 1; k < COUNT; k++) { if (v[k] < dh) { dh = v[k]; jh = k; } }
    if (dh < 0.0008) { j = jh; break; }
    if (depth < 0.0008 && i > 0) { break; }
    if (depth > deep + hs) {
      let s = min(depth - deep, dh) * STEP_SCALE;
      p += d * s;
      len += s;
    } else {
      // the index rises with depth, and depth rises against the slope of the body's distance
      let dn = -nrmB(p, m) * (ex.x * ex.y * exp(-max(depth, 0.0) * ex.y));
      let s = min(hs, max(min(depth, dh) * STEP_SCALE, 0.25 * vox));
      d = normalize(d * (n0 + grinAt(depth)) + dn * s);
      p += d * s;
      len += s;
    }
    if (len > 2.5 * BOUND) { break; }
  }
  *dir = d;
  if (j >= 0) {
    *b = j;
    *nq = nrmB(p, j);
    *q = p;
    return len;
  }
  return len + leaveB(p, d, m, q, nq, b);
}
//#endif

// From pos inside body m heading *dir: on to the next surface (eventB), along a curved path in a body
// whose index changes with depth.
fn stretchB(pos: vec3f, dir: ptr<function, vec3f>, m: i32, lam: f32, q: ptr<function, vec3f>, nq: ptr<function, vec3f>, b: ptr<function, i32>) -> f32 {
//#if GRIN
  if (hasB(m, 2u)) { return travelB(pos, dir, m, iorB(m, lam), q, nq, b); }
//#endif
  return eventB(pos, *dir, m, q, nq, b);
}

//#if CRACK_DEPTH
// the distance to the surface of the body in hand (cb), at a point in the shape's own frame
fn mapOB(o: vec3f) -> f32 { return textureSampleLevel(bodTex, volSmp, o / (2.0 * HB) + 0.5, 0.0)[cb]; }
//#endif

//#if SPECTRUM
// The band of wavelengths the path in hand stands for (where it begins, how wide), at how many
// wavelengths across it absorption by wavelength is weighed, and where in their steps they lie.
var<private> bandLo: f32;
var<private> bandW: f32;
var<private> bandN: i32;
var<private> bandXi: f32;
// What body m, absorbing by wavelength, leaves by colour of the path's band after a stretch s through
// it: each wavelength of the band absorbed by its own amount and weighed by its share of red, green and
// blue, so that a narrow absorption band takes its own colour out.
fn keepSpecB(m: i32, s: f32) -> vec3f {
  cb = m;
  var num = vec3f(0.0);
  var den = vec3f(0.0);
  for (var j = 0; j < bandN; j++) {
    let l = bandLo + bandW * (f32(j) + bandXi) / f32(bandN);
    let w = max(specW(l), vec3f(0.0));
    num += w * exp(-tabAt(l).z * max(s, 0.0));
    den += w;
  }
  return num / max(den, vec3f(1e-6));
}
// the path in hand stands for wavelengths lo .. lo + w, weighed at n of them
fn bandB(lo: f32, w: f32, n: i32, xi: f32) {
  bandLo = lo;
  bandW = w;
  bandN = n;
  bandXi = xi;
}
//#endif

//#if BIREF
// Double refraction. A path in a doubly refracting body is one of its two rays: ordinary (an index like
// glass's) or extraordinary (an index that depends on its direction to the crystal's axis); every path
// is followed twice, once as each, as for a single such material. pmode: which the path in hand is.
// kw: its wave (its length is the index it meets), read only while it is extraordinary in such a body.
var<private> pmode: i32;
var<private> kw: vec3f;
// is the path extraordinary in body k
fn biB(k: i32) -> bool { return pmode == 1 && hasB(k, 4u); }
// is any body doubly refracting (every path is then followed twice)
fn anyBiref() -> bool {
  for (var k = 0; k < COUNT; k++) { if (hasB(k, 4u)) { return true; } }
  return false;
}
// the wave of a path heading d in m (a body, or -1: the air)
fn waveOf(m: i32, d: vec3f, lam: f32) -> vec3f {
  if (biB(m)) { return kw; }
  return d * iorB(m, lam);
}
// A wave kin crossing a surface into b (a body or the air; b is the body it is in for what the surface
// sends back), N the surface's normal pointing into b: the wave keeps the part of itself that lies along
// the surface. Returns the way the light then travels (zero if no wave fits: total reflection) and sets
// kw to the new wave.
fn crossW(kin: vec3f, N: vec3f, b: i32, lam: f32) -> vec3f {
  let kt = kin - dot(kin, N) * N;
  let nb = iorB(b, lam);
  if (biB(b)) {
    let ax = mb.b[b].axis.xyz;
    let a = normalize(vec3f(dot(u.obj0.xyz, ax), dot(u.obj1.xyz, ax), dot(u.obj2.xyz, ax)));
    let ne = nb + mb.b[b].exo.z;
    kw = waveE(kt, N, a, nb, ne);
    if (dot(kw, kw) < 0.01) { return vec3f(0.0); }
    return rayE(kw, a, nb, ne);
  }
  let h = nb * nb - dot(kt, kt);
  if (h <= 0.0) {
    kw = vec3f(0.0);
    return vec3f(0.0);
  }
  kw = kt + sqrt(h) * N;
  return kw / nb;
}
//#endif

//#if PIEZO
// The glow where a body presses on another (piezoluminescence): light given off inside body m wherever
// another body's surface is near it or lies inside it, most at the meeting itself, by m's own pressure
// glow (its colour, and `piezo`, so that it pulses and loops as that does). The stretch is walked in
// steps that shorten near the other bodies, so that a thin zone of contact is never stepped over.
const PRESS_W: f32 = 0.02;              // how far from the meeting the glow reaches (the shape is about 2 across)
const PRESS_GAIN: f32 = 8.0;            // with PRESS_W: about 0.2 of white on each side of a meeting a path crosses straight, at piezo 1
fn pressB(m: i32, pos: vec3f, dir: vec3f, s: f32) -> vec3f {
  if (!hasB(m, 16u)) { return vec3f(0.0); }
  var t = 0.0;
  var sum = 0.0;
  for (var i = 0; i < 32; i++) {
    if (t >= s) { break; }
    let v = dist4(pos + dir * t);
    var f = 0.0;
    var near = 1e9;
    for (var k = 0; k < COUNT; k++) {
      if (k == m) { continue; }
      let d = v[k];
      // at the other's surface, where the two meet (or, where they overlap, where the other's surface
      // lies inside this one), falling off to either side
      f += exp(-abs(d) / PRESS_W);
      near = min(near, abs(d));
    }
    let h = min(max(0.8 * near, 0.5 * PRESS_W), s - t);
    sum += f * h;
    t += h;
  }
  let pz = ibod[m].piezo;
  return pz.rgb / max(pz.r, 1e-6) * (pz.w * PRESS_GAIN * sum);
}
//#endif

// How much wider (as a variance, radians squared) a rough surface of spread alpha makes light that
// passes through it from index na at cosine c into index nb.
fn widenB(alpha: f32, na: f32, c: f32, nb: f32, od: vec3f, nq: vec3f) -> f32 {
  let co = abs(dot(od, nq));
  let w = 0.70711 * alpha * clamp(abs(na * c - nb * co) / max(nb * co, 0.05), 0.0, 6.0);
  return w * w;
}

//#if BODY_PATH
// A scattered or facet-sampled path continues through the actual body volumes. Walking keeps
// all three channels, and every subsequent dielectric interface or opaque body
// can affect the light returning to the camera.
//#if BODY_FROST
fn walkBodyPath(p0:vec3f,d0:vec3f,m0:i32,lam:f32,sampledStart:bool)->vec3f {
  var sampled = sampledStart;
  var survival = 1.0;
//#if FROST_DIRECT
  var naturalWeight=directFrost.escapeWeight;
//#if FROST_ROOM_EXIT
  var roomSampled=sampledStart && directFrost.enabled;
//#endif
//#endif
//#else
fn walkBodies(p0: vec3f,d0: vec3f,m0:i32,lam:f32)->vec3f {
//#endif
//#if BODY_METAL_SAMPLE
  var metSampledMetal = false;
//#endif
  var spread = 0.0;
  var pos=p0; var dir=d0; var m=m0;
  var keep=vec3f(1.0); var light=vec3f(0.0);
  for(var e=0;e<256;e++){
    if(max3(keep)<0.002){break;}
//#if BODY_FROST
    if (sampled && e >= max(u.nb,8)) {
      let moving = u.env0.w > 0.5;
      let tail = select(0.15, 0.02, moving);
      let continuation = select(0.95, 0.82, moving);
      let next = tail + continuation * (survival - tail);
      if (rnd() * survival > next) {break;}
      keep *= survival / next;
      survival = next;
    }
//#endif
    var q=pos; var n=vec3f(0.0); var b=-1;
    if(m<0){
      let bb=dot(pos,dir);let h=bb*bb-dot(pos,pos)+BOUND*BOUND;
//#if BODY_METAL_SAMPLE
      if((h<=0.0) && metSampledMetal){return light+keep*sampledMetalRoomB(pos,dir,spread);}
//#endif
//#if FROST_DIRECT
      if(h<=0.0){return light+keep*naturalWeight*roomEscapeB(pos,dir,spread);}
//#else
      if(h<=0.0){return light+keep*roomEscapeB(pos,dir,spread);}
//#endif
      let far=-bb+sqrt(h);var t=0.0;var found=false;
      for(var j=0;j<STEPS_OUT;j++){
        let d=mapW(pos+t*dir);
        if(d<0.0008){found=true;break;}
        t+=d*STEP_SCALE;if(t>far){break;}
      }
//#if BODY_METAL_SAMPLE
      if((!found) && metSampledMetal){return light+keep*sampledMetalRoomB(pos,dir,spread);}
//#endif
//#if FROST_DIRECT
      if(!found){return light+keep*naturalWeight*roomEscapeB(pos,dir,spread);}
//#else
      if(!found){return light+keep*roomEscapeB(pos,dir,spread);}
//#endif
      t+=mapNW(pos+dir*t);q=pos+dir*t;
      b=nearestOf(dist4(q)); n=nrm(q);
    }else{
      cb=m;
//#if BODY_WALK
      let walking=hasB(m,32u);
      var fly=1e9;var rates=vec3f(0.0);
      if(walking){
        rates=collisionRates();fly=walkFlight(rates);
        let v=dist4(pos);var safety=-v[m];
        for(var j=m+1;j<COUNT;j++){safety=min(safety,v[j]);}
        if(fly<0.9*safety){
          keep*=exp(-(mb.absorb[m].rgb+mb.b[m].sssA.rgb)*fly)*walkWeight(rates,fly,true);
          pos+=dir*fly;dir=walkDirection(dir);continue;
        }
      }
//#endif
//#if BODY_FROST
      let before = dir;
      var len = 0.0;
      if (sampled) {len=max(stretchB(pos,&dir,m,lam,&q,&n,&b),0.0);}
      else {len=max(eventB(pos,dir,m,&q,&n,&b),0.0);}
//#if INNER
      if (sampled && hasB(m,8u)) {
        let inclusions=inner(pos,before,len,iorB(m,lam),e);
        light+=keep*inclusions.light;
        keep*=inclusions.keep;
//#if CRACK
        if (inclusions.t>=0.0) {
          keep*=exp(-mb.absorb[m].rgb*inclusions.t);
          pos+=before*inclusions.t;
          dir=inclusions.dir;
          continue;
        }
//#endif
      }
//#endif
      if (sampled) {light+=keep*mb.b[m].glow.rgb*len;}
//#else
      let len=max(eventB(pos,dir,m,&q,&n,&b),0.0);
//#endif
//#if BODY_WALK
      if(walking && fly<len){
        keep*=exp(-(mb.absorb[m].rgb+mb.b[m].sssA.rgb)*fly)*walkWeight(rates,fly,true);
        pos+=dir*fly;dir=walkDirection(dir);continue;
      }
      if(walking){
        keep*=exp(-(mb.absorb[m].rgb+mb.b[m].sssA.rgb)*len)*walkWeight(rates,len,false);
      }else
//#endif
//#if SSS
      if(mb.kind[m]==2.0){
        cb=m;
        let glassS = glowAlong(pos,dir,len);
        let glassKeep = keep*exp(-(mb.absorb[m].rgb+collisionRates()+mb.b[m].sssA.rgb)*len)*throughHue(len);
        let sssOp = clamp(mb.b[m].tab[19].w, 0.0, 1.0);
        if (sssOp > 0.0) {
          let tauW = waxTau(len);
          let waxKeep = keep * exp(-vec3f(tauW)) * waxThroughTint(tauW);
          light += keep * mix(glassS, waxVolume(pos, dir, len, dir, 1.0, false), sssOp);
          keep = mix(glassKeep, waxKeep, sssOp);
        } else {
          light += keep * glassS;
          keep = glassKeep;
        }
      }else
//#endif
      {
//#if SPECTRUM
//#if BODY_FROST
        if (sampled && hasB(m,1u)) {keep*=keepSpecB(m,len);}
        else
//#endif
//#endif
        {keep*=exp(-mb.absorb[m].rgb*len);}
      }
    }
//#if FROST_DIRECT
    naturalWeight=1.0;
//#endif
    if(dot(dir,n)>0.0){n=-n;}
    n=faceIncident(n,-dir);
    if(paintB(b)){return light+keep*paintB_seen(b,q,n,dir,iorB(m,lam));}
//#if BODY_FROST
    if (sampledInterfaceB(m,b)) {
//#if BODY_METAL_SAMPLE
      metSampledMetal = false;
//#endif
//#if FROST_DIRECT
//#if FROST_ROOM_EXIT
      directFrost.enabled=!roomSampled && m>=0 && b<0 && spread<=0.0 && mirrorSharp<=0.0;
      roomSampled=roomSampled || directFrost.enabled;
//#else
      directFrost.enabled=e<6 && spread<=0.0 && mirrorSharp<=0.0;
//#endif
//#endif
      let path = sampleInterfaceB(q,dir,n,m,b,lam);
//#if FROST_DIRECT
      light+=keep*directFrost.added;
      naturalWeight=directFrost.escapeWeight;
//#endif
      if (path.w == 0.0) {break;}
      dir = path.xyz;
      let crossed = path.w > 0.0;
      pos = q + select(n,-n,crossed)*0.004;
      if (crossed) {m=b;}
      sampled = true;
      continue;
    }
//#endif
//#if BODY_METAL_SAMPLE
    if (b >= 0 && hasB(b, 128u)) {
      let surface = secondaryMetalB(q, n, dir, b, m, lam);
      light += keep * surface.emitted;
      keep *= surface.weight;
      dir = surface.direction;
      pos = q + n * 0.004;
      metSampledMetal = true;
      continue;
    }
//#endif
    let c=clamp(-dot(dir,n),0.0,1.0);let r=surfB(m,b,c,q,n,lam);
    let dielectricHit=traceDielectricHit(n,mediumB(m,iorB(m,lam)),mediumB(b,iorB(b,lam)));
    let dielectricSplit=pbr_dielectric_split(dielectricHit,-dir);
    var refracted=dielectricSplit.transmitted;
    var metalNormal = n;
    var metalTint = vec3f(1.0);
    var finishVariance = 0.0;
//#if LIVE_CONDUCTOR_FINISH
    if (opaqueB(b)) {
      let finish = conductorFinishB(b, q, n, dir);
      metalNormal = finish.normal;
      metalTint = finish.tint;
      finishVariance = finish.variance;
    }
//#endif
    var reflected = reflect(dir, metalNormal);
//#if BODY_FROST
//#if BIREF
    var reflectedWave=kw;
    if (sampled && (biB(m) || biB(b))) {
      let kin=waveOf(m,dir,lam);
      reflected=crossW(kin,metalNormal,m,lam);
      reflectedWave=kw;
      refracted=crossW(kin,-n,b,lam);
    }
//#endif
//#endif
    let prob=clamp(dot(r,vec3f(0.2126,0.7152,0.0722)),0.0,1.0);
    let interfaceOwner=max(m,b);
    var plainDielectric=!opaqueB(b) && !wallB(interfaceOwner) &&
      mb.b[interfaceOwner].surf.x<=0.0 && mb.b[interfaceOwner].surf.z<=1.001;
//#if FILM
    plainDielectric=plainDielectric && mb.b[interfaceOwner].film.x<=0.0;
//#endif
//#if BODY_FROST
    plainDielectric=plainDielectric && !sampled;
//#endif
//#if BIREF
    plainDielectric=plainDielectric && !biB(m) && !biB(b);
//#endif
    if(plainDielectric){
      if(dielectricSplit.total_internal_reflection!=0u){
        dir=dielectricSplit.reflected;pos=q+0.004*n;
      }else{
        let interfaceSample=pbr_dielectric_sample(
          dielectricHit,-dir,PbrPathRng(vec2f(0.0),rnd()));
        dir=interfaceSample.wi;
        if(interfaceSample.medium_transition==PBR_MEDIUM_ENTER_DESTINATION){
          pos=q-0.004*n;m=b;
        }else{
          pos=q+0.004*n;
        }
      }
      continue;
    }
//#if BODY_FROST
    if (sampled && opaqueB(b)) {
      let alpha=mb.b[b].surf.x;
//#if BIREF
      kw=reflectedWave;
      if (dot(reflected,reflected)<0.5) {break;}
//#endif
      light+=keep*(mb.b[b].glow.rgb*0.3+metalTint*r*skyFarB(b,reflected));
//#if ROUGH
      light+=keep*metalTint*conductorBroadB*envL(n,0.80);
//#endif
      keep*=metalTint*(r-conductorBroadB);
      spread+=2.0*alpha*alpha+finishVariance;
      dir=reflected;pos=q+0.004*n;
      continue;
    }
//#endif
    if(opaqueB(b) || dot(refracted,refracted)<0.5){
//#if BODY_FROST
//#if BIREF
      if (sampled) {kw=reflectedWave;if (dot(reflected,reflected)<0.5) {break;}}
//#endif
//#endif
      keep*=r*metalTint;spread+=finishVariance;dir=reflected;pos=q+0.004*n;
    }else if(rnd()<prob){
//#if BODY_FROST
      if (sampled) {let owner=max(m,b);let alpha=mb.b[owner].surf.x;spread+=2.0*alpha*alpha;}
//#endif
//#if BODY_FROST
//#if BIREF
      if (sampled) {kw=reflectedWave;if (dot(reflected,reflected)<0.5) {break;}}
//#endif
//#endif
      keep*=r/max(prob,1e-6);dir=reflected;pos=q+0.004*n;
    }else{
//#if BODY_FROST
      if (sampled) {
        let owner=max(m,b);
        spread+=widenB(mb.b[owner].surf.x,iorB(m,lam),c,iorB(b,lam),refracted,n);
        if (wallB(owner)) {keep*=mb.b[owner].axis.xyz;}
      }
//#endif
      keep*=(vec3f(1.0)-r)/max(1.0-prob,1e-6);
      dir=refracted;pos=q-0.004*n;m=b;
    }
  }
  return light;
}
//#endif
// Integrate emission and scattering on this segment. Rays surviving extinction
// retain their direction and any spread accumulated at actual rough interfaces.
fn throughB(m: i32, pos: vec3f, dir: vec3f, s: f32, lam: f32, thr: ptr<function, vec3f>) -> vec3f {
//#if SSS
  let incoming = *thr;
//#endif
//#if SPECTRUM
  if (hasB(m, 1u)) {
    *thr *= keepSpecB(m, s);
  } else {
    *thr *= exp(-mb.absorb[m].rgb * s);
  }
//#else
  *thr *= exp(-mb.absorb[m].rgb * s);
//#endif
  var add = *thr * mb.b[m].glow.rgb * s;
//#if PIEZO
  add += *thr * pressB(m, pos, dir, s);
//#endif
//#if SSS
  if (mb.kind[m] > 1.5 && mb.kind[m] < 2.5) {
    cb = m;
//#if BODY_WALK
    if(hasB(m,32u)){
      let rates=collisionRates();let fly=walkFlight(rates);
      if(fly<max(s,0.0)){
        let scatter=incoming*exp(-(mb.absorb[m].rgb+mb.b[m].sssA.rgb)*fly)*walkWeight(rates,fly,true);
        add+=scatter*walkBodies(pos+dir*fly,walkDirection(dir),m,lam);
      }
      cb=m;
      *thr*=exp(-(rates+mb.b[m].sssA.rgb)*max(s,0.0));
      return add;
    }
//#endif
    let glassS = glowAlong(pos, dir, s);
    let glassThr = *thr * exp(-(collisionRates() + mb.b[m].sssA.rgb) * max(s, 0.0)) * throughHue(s);
    let sssOp = clamp(mb.b[m].tab[19].w, 0.0, 1.0);
    if (sssOp > 0.0) {
      let tauW = waxTau(s);
      let waxThr = incoming * (0.62 * exp(-vec3f(tauW))) * waxThroughTint(tauW);
      add += incoming * mix(glassS, waxVolume(pos, dir, s, dir, 1.0, false), sssOp);
      *thr = mix(glassThr, waxThr, sssOp);
    } else {
      add += incoming * glassS;
      *thr = glassThr;
    }
  }
//#endif
  return add;
}

// A ray at ro in m0 (a clear or milky body, or -1: the air), heading rd: what it brings back from the
// bodies it goes on to meet and from the lights. It passes through clear and milky bodies and thin walls
// (adding what their outsides reflect of the lights) and is turned by metals. sg0: how spread it is
// already (a variance), which softens the lights it finally sees.
fn beyondB(ro_in: vec3f, rd_in: vec3f, m0: i32, lam: f32, sg0: f32) -> vec3f {
  var ro = ro_in;
  var rd = rd_in;
  var m = m0;
  var col = vec3f(0.0);
  var thr = vec3f(1.0);
  var sg2 = sg0;
//#if BODY_METAL_SAMPLE
  var metSampledMetal = false;
//#endif
  for (var l = 0; l < EVENTS; l++) {
    var q: vec3f;
    var nq: vec3f;
    var b: i32;
    if (m < 0) {
      // through the air to the next body, if there is one
      let bb = dot(ro, rd);
      let h = bb * bb - (dot(ro, ro) - BOUND * BOUND);
      if (h <= 0.0) {
//#if ROUGH
        frMiss(l);
//#endif
//#if BODY_METAL_SAMPLE
        if (metSampledMetal) { return col + thr * sampledMetalRoomB(ro, rd, sg2); }
//#endif
        return col + thr * roomEscapeB(ro, rd, sg2);
      }
      // A ray outside the stage still probes its origin unless the measured field rules it out.
      var tFar = max(0.0, -bb + sqrt(h));
      if (mb.envelope.x > 0.0 && mb.envelope.x < 1e5) {
        let scale = 1.5 / HB;
        let x = toObj(ro) * scale;
        let direction = toObj(rd) * scale;
        let weight = sqrt(abs(x));
        let above = dot(x*x,weight) - mb.envelope.x;
        let along = 1.25 * dot(x*weight,direction);
        let curvature = 1.25 * dot(weight,direction*direction);
        // Convexity of (x*x)^1.25 gives a quadratic lower bound along the ray.
        // Once that quadratic rises above the measured level, no hit remains.
        if (above > 0.0 && along >= 0.0) { tFar = -1.0; }
        else if (curvature > 0.000001) {
          let disc = along*along - curvature*above;
          if (disc < -0.00001) { tFar = -1.0; }
          else {
            let root = sqrt(max(0.0,disc));
            var end = (-along + root) / curvature;
            if (along >= 0.0 && along + root > 0.0000001) { end = -above/(along+root); }
            tFar = min(tFar,end+0.00001);
          }
        }
      }
      let px = 2.0 * u.tan_fov / u.res.y;
      var t = 0.0;
      let hit = marchBeyond(ro, rd, tFar, px, &t);
      if (!hit) {
//#if ROUGH
        frMiss(l);
//#endif
//#if BODY_METAL_SAMPLE
        if (metSampledMetal) { return col + thr * sampledMetalRoomB(ro, rd, sg2); }
//#endif
        return col + thr * roomEscapeB(ro, rd, sg2);
      }
      t += mapNW(ro + rd * t);
      t += mapNW(ro + rd * t);
//#if ROUGH
      // (the first piece the light leaving a rough body lands on: where the blur after tracing takes its width)
      frHit(l, t * sqrt(sg2), sqrt(sg2));
//#endif
      q = ro + rd * t;
      nq = nrm(q);
      b = mediumOf(dist4(q - nq * 0.004));
      if (b < 0) { b = nearestOf(dist4(q)); }
    } else {
      // through body m to where it ends or another takes over (curving, if its index changes with depth)
      let d0 = rd;
      let s = stretchB(ro, &rd, m, lam, &q, &nq, &b);
//#if INNER
      if (hasB(m, 8u)) {
        // what lies inside this body along the way; a crack that would turn the path ends it there,
        // seeing the lights the way it then heads
        cb = m;
        let inn = inner(ro, d0, s, iorB(m, lam), 1);
        col += thr * inn.light;
        thr *= inn.keep;
//#if CRACK
        if (inn.t >= 0.0) {
          col += throughB(m, ro, d0, inn.t, lam, &thr);
          return col + thr * seenB(inn.dir, sqrt(sg2));
        }
//#endif
      }
//#endif
      col += throughB(m, ro, d0, s, lam, &thr);
    }
    var c = -dot(rd, nq);
    if (c < 0.0) { nq = -nq; c = -c; }      // nq now faces the ray
    nq = faceIncident(nq, -rd);
    if (paintB(b)) { return col + thr * paintB_seen(b, q, nq, rd, iorB(m, lam)); }
//#if BODY_FROST
    if (sampledInterfaceB(m,b)) {return col+thr*secondaryFrostB(q,rd,nq,m,b,lam);}
//#endif
//#if BODY_METAL_SAMPLE
    if (b >= 0 && hasB(b, 128u)) {
      let surface = secondaryMetalB(q, nq, rd, b, m, lam);
      col += thr * surface.emitted;
      thr *= surface.weight;
      if (max3(thr) < 0.002) { return col; }
      rd = surface.direction;
      ro = q + nq * 0.004;
      metSampledMetal = true;
      continue;
    }
//#endif
    let o = max(m, b);
    let R = surfB(m, b, c, q, nq, lam);
    let alpha = mb.b[o].surf.x;
    let mir = reflect(rd, nq);
    if (opaqueB(b)) {
      // a metal turns it, and it goes on through what it was in
//#if LIVE_CONDUCTOR_FINISH
      let finish = conductorFinishB(b, q, nq, rd);
      let metalNormal = finish.normal;
      let metalTint = finish.tint;
      sg2 += finish.variance;
      let reflected = reflect(rd, metalNormal);
      if (finish.marked) { mirrorSharp = 0.0; }
//#else
      let metalNormal = nq;
      let metalTint = vec3f(1.0);
      let reflected = mir;
//#endif
      col += thr * (mb.b[b].glow.rgb * 0.3 + R * metalTint * skyFarB(o, mir));
//#if ROUGH
      if (any(conductorBroadB > vec3f(0.0))) { col += thr * metalTint * conductorBroadB * envL(nq, 0.80); }
//#endif
      thr *= metalTint * (R - conductorBroadB);
      sg2 += 2.0 * alpha * alpha;
      if (max3(thr) < 0.02) { return col; }
//#if BIREF
      rd = crossW(waveOf(m, rd, lam), metalNormal, m, lam);
      if (dot(rd, rd) < 0.5) { return col; }
//#else
      rd = reflected;
//#endif
      ro = q + nq * 0.004;
      continue;
    }
    let na = iorB(m, lam);
    let nb = iorB(b, lam);
//#if BIREF
    let od = crossW(waveOf(m, rd, lam), -nq, b, lam);
//#else
    let od = pbr_dielectric_split(traceDielectricHit(nq, mediumB(m, na), mediumB(b, nb)), -rd).transmitted;
//#endif
    if (dot(od, od) < 0.5) {
      // trapped by total reflection: take the bounce and stop there
      return col + thr * seenB(mir, sqrt(sg2));
    }
    // (only what is sent back into the air is added, as for plain glass: reflections inside are let go)
    if (m < 0) { col += thr * R * (seenB(mir, sqrt(sg2 + 2.0 * alpha * alpha)) + skyFarB(o, mir)); }
    thr *= 1.0 - R;
    if (wallB(o)) { thr *= mb.b[o].axis.xyz; }      // through a thin wall: its tint, once
    sg2 += widenB(alpha, na, c, nb, od, nq);
    rd = od;
    ro = q - nq * 0.006;
    m = b;
  }
  return col + thr * seenB(rd, sqrt(sg2));
}

// The terminal light after an air ray misses every other body.
fn roomEscapeB(origin: vec3f, direction: vec3f, spread: f32) -> vec3f {
  return seenB(direction, sqrt(spread)); // terminal room
}

// Follow a smooth surface reflection through the scene. A conductor can see the lit floor.
fn externalReflectionB(metalTop: bool, k: i32, origin: vec3f, direction: vec3f, spread: f32) -> vec3f {
  return beyondB(origin, direction, -1, 550.0, spread); // metal reflection
}

//#if BODY_METAL_SAMPLE
struct MetalSampleB {
  coat: vec3f,
  emitted: vec3f,
  direction: vec3f,
  weight: vec3f,
}

fn sampleMetalSurfaceB(p: vec3f, n: vec3f, rd: vec3f, k: i32, medium: i32,
                       wavelength: f32, r: vec2f, fc: vec2f) -> MetalSampleB {
  cb = k;
  let cosine = clamp(-dot(rd, n), 0.0, 1.0);
  let coatIor = mb.b[k].surf.z;
  let coated = medium < 0 && coatIor > 1.001;
  let outsideIor = select(iorB(medium, wavelength), coatIor, coated);
  var film = 0.0;
//#if FILM
  if (mb.b[k].film.x > 0.0) { film = filmNm(p, n); }
//#endif
  var coat = vec3f(0.0);
  var emitted = mb.b[k].glow.rgb * 0.3;
  var substrateCosine = cosine;
  if (coated) {
    coat = vec3f(fresnel(cosine, 1.0, coatIor));
//#if FILM
    if (film > 0.0) { coat = filmRGB(cosine, film, 1.0, coatIor); }
    film = 0.0;
//#endif
    substrateCosine = sqrt(max(1.0 - (1.0 - cosine * cosine) / (coatIor * coatIor), 0.0));
  }
  var normal = n;
  var alpha = mb.b[k].surf.x;
  var tint = vec3f(1.0);
//#if WEAR
  let wear = wearAt(p, n);
  normal = wear.n;
  alpha = sqrt(alpha * alpha + wear.sig * wear.sig);
  tint = wear.tint * wear.keep;
  let frame = brushSurface(wear,p,alpha);
//#endif
  let smoothReflectance = metalRGB(substrateCosine, outsideIor, film);
  var single = smoothReflectance;
  let substrate = (1.0 - coat) * tint;
  if (film <= 0.0) {
    var energy = conductorEnergy(substrateCosine, outsideIor, smoothReflectance);
//#if WEAR
    if (wear.aniso > 0.0) {
      energy = brushConductorEnergy(substrateCosine,outsideIor,smoothReflectance,frame.across,-rd,normal);
    }
//#endif
    emitted += substrate * energy.multiple * envL(normal, 0.80);
    single = energy.single;
  }
//#if WEAR
  let sample = frGlossyFrame(p,rd,normal,r,alpha,coated,frame.across,frame.slopes,wear.aniso>0.0);
//#else
  let sample = frGlossySample(p, rd, normal, r, alpha, coated);
//#endif
  if (!coated) { single = metalRGB(sample.facetCosine, outsideIor, film); }
  let weight = substrate * single * sample.weight;
  emitted += weight * skyFarB(k, reflect(rd, n));
  return MetalSampleB(coat, emitted, sample.direction, weight);
}

// On a secondary path, sample one of the coat or substrate connections. The choice
// divides its probability back out and lets the scene walk continue without branching.
fn secondaryMetalB(q: vec3f, n: vec3f, rd: vec3f, body: i32, medium: i32, lam: f32) -> MetalSampleB {
  var surface = sampleMetalSurfaceB(q, n, rd, body, medium, lam, hash2(q + rd), q.xy);
  if (any(surface.coat > vec3f(0.0))) {
    let chance = clamp(dot(surface.coat, vec3f(0.2126, 0.7152, 0.0722)), 0.02, 0.98);
    if (hash2(q - rd).x < chance) {
      surface.weight = surface.coat / chance;
      surface.direction = reflect(rd, n);
    } else {
      surface.weight /= 1.0 - chance;
    }
  }
  return surface;
}

fn sampledMetalRoomB(origin: vec3f, direction: vec3f, spread: f32) -> vec3f {
  if (spread > 0.0) { return envL(direction, sqrt(spread)); }
//#if PICTURE
  let incident = env(direction);
//#else
  let incident = mirrorEnv(direction, 0.0);
//#endif
  return incident; // sampled metal escaping the scene
}

fn sampledMetalB(p: vec3f, n: vec3f, rd: vec3f, k: i32, xi: f32, fc: vec2f) -> vec3f {
  let coated = mb.b[k].surf.z > 1.001;
  let surface = sampleMetalSurfaceB(p, n, rd, k, -1, 550.0, rnd2(select(82, 0, coated), rd, xi), fc);
  let origin = p + 0.006 * n;
  var radiance = surface.emitted;
  if (any(surface.coat > vec3f(0.0))) {
    mirrorSharp = 1.0 + mirrorLod(p, n, clamp(-dot(rd, n), 0.0, 1.0));
    radiance += surface.coat * externalReflectionB(true, k, origin, reflect(rd, n), 0.0);
    cb = k;
  }
  if (any(surface.weight > vec3f(0.0))) {
//#if PICTURE
    mirrorSharp = 0.0;
//#else
    mirrorSharp = 1.0;
//#endif
    radiance += surface.weight * externalReflectionB(true, k, origin, surface.direction, 0.0);
    cb = k;
  }
  mirrorSharp = 0.0;
  return radiance;
}
//#endif

fn paintProbeB(p: vec3f, inDir: vec3f, rd: vec3f, dist: f32, k: i32) -> vec3f {
  let q = p + (inDir - rd) * dist;
  let n0 = nrmB(q, k);
  return nrmB(q - n0 * dist4(q)[k], k);
}

fn paintRimNormalB(p: vec3f, rawN: vec3f, rd: vec3f, k: i32) -> vec3f {
  var n = faceIncident(rawN, -rd);
  let c0 = dot(-rd, rawN);
  let perp = rawN - rd * dot(rawN, rd);
  let pl = length(perp);
  if (c0 < 0.72 && pl > 1e-5) {
    let inDir = -perp / pl;
    let vox = 2.0 * HB / min(u.vres, 768.0);
    let nLoc = paintProbeB(p, inDir, rd, 1.2 * vox, k);
    let onBev = smoothstep(0.008, 0.045, dot(-rd, nLoc) - c0);
    if (onBev > 0.0) {
      var qOut = p - inDir * (2.2 * vox) - rd * (4.0 * vox);
      var minD = 1e9;
      for (var s = 0; s < 8; s++) {
        let d = dist4(qOut)[k];
        minD = min(minD, d);
        qOut += rd * max(d, 0.5 * vox);
      }
      let isSil = smoothstep(0.20 * vox, 0.70 * vox, minD);
      if (isSil > 0.0) {
        let n1 = paintProbeB(p, inDir, rd, 2.6 * vox, k);
        let c1 = dot(-rd, n1);
        if (c1 > c0 + 0.015) {
          let n2 = paintProbeB(p, inDir, rd, 3.8 * vox, k);
          let flat = smoothstep(0.9988, 0.9997, dot(n1, n2));
          let w = isSil * onBev * flat * smoothstep(0.015, 0.06, c1 - c0);
          n = normalize(mix(n, faceIncident(n1, -rd), w));
        }
      }
    }
  }
  let minC = clamp(0.22 + 0.60 * mb.b[k].tab[30].w, 0.22, 0.55);
  let cn = -dot(rd, n);
  if (cn < minC) {
    let tang = n + rd * cn;
    let tl = length(tang);
    if (tl > 1e-5) { n = tang * (sqrt(max(1.0 - minC * minC, 0.0)) / tl) - rd * minC; }
  }
  return n;
}

// What a point of the surface sends towards the eye: shade(), for bodies of several materials.
fn shadeB(ro: vec3f, rd: vec3f, t: f32, xi: f32, W: i32, fc: vec2f) -> vec3f {
//#if LIVE_CONDUCTOR_FINISH
  finishPixelB = fc;
//#endif
//#if BODY_PATH
  rState=pcg(bitcast<u32>(rd.x)+pcg(bitcast<u32>(rd.y)+pcg(u32(u.frame)+pcg(bitcast<u32>(xi)))));
//#endif
  let p = ro + rd * t;
  let rawN = nrm(p);
  var n = faceIncident(rawN, -rd);
  var k = mediumOf(dist4(p - n * 0.004));
  if (k < 0) { k = nearestOf(dist4(p)); }
  cb = k;
  if (paintB(k)) { n = paintRimNormalB(p, rawN, rd, k); }
//#if PAINT_ONLY
  return paintB_seen(k, p, n, rd, 1.0);
//#else
  if (paintB(k)) { return paintB_seen(k, p, n, rd, 1.0); }
  var cosi = clamp(-dot(rd, n), 0.0, 1.0);
//#if SSS
  if (mb.kind[k] > 1.5 && mb.kind[k] < 2.5 && mb.b[k].tab[19].w > 0.0) {
    let c0w = dot(-rd, rawN);
    if (c0w < 0.35) {
      let fnw = nrmB(p - (rawN + rd) * (4.0 * HB / u.vres), k);
      if (dot(-rd, fnw) > c0w) {
        n = normalize(mix(n, faceIncident(fnw, -rd), smoothstep(0.35, 0.03, c0w)));
      }
    }
    cosi = clamp(-dot(rd, n), 0.0, 1.0);
    cosi = mix(cosi, max(cosi, 0.62), clamp(mb.b[k].tab[19].w * sqrt(clamp(mb.b[k].surf.x, 0.0, 1.0)), 0.0, 1.0));
  }
//#endif
//#if BODY_FROST
  if (sampledInterfaceB(-1,k)) {return primaryFrostB(p,rd,n,k,xi,W,fc);}
//#endif
//#if BODY_METAL_SAMPLE
  if (hasB(k, 128u) && mb.b[k].grating.x <= 0.0) {
    return sampledMetalB(p, n, rd, k, xi, fc);
  }
//#endif
  let alpha = mb.b[k].surf.x;
  let sigR = 1.41421 * alpha;
  let mir = reflect(rd, n);
  // what the surface itself mirrors: spread by its roughness (a rough surface as frost.wgsl reads it),
  // or, under a film on a polished surface, the surroundings sharp; and the film's studio
  var envR = seenB(mir, sigR);
//#if ROUGH
  if (alpha > 0.0) { envR = frTop(rd, n, xi, cosi); }
//#endif
  // Under a clear coat the roughness is the body's, below the coat: the coat itself is polished and
  // mirrors its surroundings sharp, as a single coated body's does (shadeM: the coat's reflection is
  // sharp whatever lies under it). Frosted glass under a coat: glossy reflections over a frosted body.
  if (alpha > 0.0 && mb.b[k].surf.z > 1.001) { envR = sharpEnv(mir, p, n, cosi); }
//#if FILM
  if (mb.b[k].film.x > 0.0 && alpha <= 0.0) { envR = mirrorEnv(mir, mirrorLod(p, n, cosi)); }
//#endif
  // a polished milky body mirrors the picture sharp, and a worn one as its wear has it (trace_material.wgsl)
  if (alpha <= 0.0 && mb.kind[k] > 1.5 && mb.kind[k] < 2.5 && mb.b[k].film.x <= 0.0) { envR = sharpEnv(mir, p, n, cosi); }
//#if WEAR
  if (alpha <= 0.0 && mb.kind[k] > 1.5 && mb.kind[k] < 2.5 && mb.b[k].film.x <= 0.0 && mb.b[k].more.z * (mb.b[k].tab[7].w + mb.b[k].tab[8].w + mb.b[k].tab[9].w + mb.b[k].tab[10].w + mb.b[k].tab[11].w + mb.b[k].tab[12].w + mb.b[k].tab[13].w) + mb.b[k].more.w + mb.b[k].sssC.w > 0.0) {
    let wr = wearAt(p, n);
    envR = wearSee(rd, wr, p, cosi) * (wr.keep * wr.tint);
  }
//#endif
//#if GRATING
  envR = gratingReflect(p,n,rd,envR,xi);
//#endif
  envR += skyB(k, mir);
  let metalTop = opaqueB(k);
//#if SCENE_REFLECTION
  let sceneGlass = mb.kind[k] < 0.5 && alpha <= 0.0 && mb.b[k].film.x <= 0.0;
//#else
  let sceneGlass = false;
//#endif
  if (metalTop || sceneGlass) {
    // Leaving a smooth interface, one reflected ray serves the whole spectrum.
    // Metals and dielectrics share the external scene walk, including secondary materials.
  //#if SPECTRUM
    bandB(400.0, 300.0, 12, xi);
  //#endif
  //#if BIREF
    pmode = 0;
  //#endif
    var R = vec3f(0.0);
    var broad = vec3f(0.0);
    var dir = mir;
    var spread = 0.0;
    if (metalTop) {
      R = surfB(-1, k, cosi, p, n, 550.0);
      R -= conductorBroadB;
      spread = sigR * sigR;
    //#if ROUGH
      if (any(conductorBroadB > vec3f(0.0))) { broad = conductorBroadB * envL(n, 0.80); }
    //#endif
      if (mb.b[k].more.y > 0.5 && alpha <= 0.0) { mirrorSharp = 1.0 + mirrorLod(p, n, cosi); }
    //#if LIVE_CONDUCTOR_FINISH
      let finish = conductorFinishB(k, p, n, rd);
      dir = reflect(rd, finish.normal);
      R *= finish.tint;
      broad *= finish.tint;
      spread += finish.variance;
      if (finish.marked) { mirrorSharp = 0.0; }
    //#endif
    }
  //#if ROUGH
    let transmittedRadius = frRad;
  //#endif
  //#if BODY_PATH
    let transmittedSeed = rState;
  //#endif
    let far = externalReflectionB(metalTop, k, p + n * 0.006, dir, spread);
    cb = k;
    mirrorSharp = 0.0;
  //#if GRATING
    let reflected = gratingReflect(p,n,rd,far,xi);
  //#else
    let reflected = far;
  //#endif
    if (metalTop) { return mb.b[k].glow.rgb * 0.3 + broad + R * (reflected + skyB(k, mir)); }
    // The transmitted ray maintains its own blur footprint and random sequence.
  //#if ROUGH
    frRad = transmittedRadius;
  //#endif
  //#if BODY_PATH
    rState = transmittedSeed;
  //#endif
    envR = reflected + skyB(k, mir);
  }

  // a clear or milky body or a thin wall: its own reflection, and each wavelength's way through it
  let alike = mb.disp.x == 0.0 && mb.disp.y == 0.0 && mb.disp.z == 0.0 && mb.disp.w == 0.0;
  var one = vec3f(0.0);
  var acc = vec3f(0.0);
  var wsum = vec3f(0.0);
  var racc = vec3f(0.0);
  var r1 = vec3f(0.0);
//#if BIREF
  // (with a doubly refracting body among them every path is followed twice: as an ordinary ray, then as
  // an extraordinary one; each carries half the light, and the surface's own reflection is counted once)
  var rsum = vec3f(0.0);
  let modes = select(1, 2, anyBiref());
  for (var j = 0; j < 2 * MAX_NW; j++) {
    let i = j / modes;
    pmode = j % modes;
//#else
  for (var i = 0; i < MAX_NW; i++) {
//#endif
    if (i >= W || (alike && i >= 1)) { break; }
    let lam = 400.0 + 300.0 * (f32(i) + xi) / f32(W);
//#if SPECTRUM
    // (absorption by wavelength is weighed across the band this path stands for: the whole spectrum
    // when one path stands for every wavelength)
    if (alike) { bandB(400.0, 300.0, 12, xi); } else { bandB(400.0 + 300.0 * f32(i) / f32(W), 300.0 / f32(W), 4, xi); }
//#endif
    let w = specW(lam);
    let R0 = surfB(-1, k, cosi, p, n, lam);
//#if BIREF
    if (pmode == 0) {
      racc += w * R0;
      rsum += w;
      r1 = R0;
    }
//#else
    racc += w * R0;
    r1 = R0;
//#endif
    let n1 = iorB(k, lam);
//#if BIREF
    var dir = crossW(rd, -n, k, lam);
    if (dot(dir, dir) < 0.5) { continue; }
//#else
    var dir = pbr_dielectric_split(traceDielectricHit(n, airMedium(), mediumB(k, n1)), -rd).transmitted;
//#endif
    var sg2 = widenB(alpha, 1.0, cosi, n1, dir, n);
//#if ROUGH
    if (alpha > 0.0) {
      // in through a rough surface: turned towards straight in as it roughens, and spread by its facets
      cb = k;
      dir = frDir(dir, n);
      sg2 = frVin0(n1);
    }
//#endif
    var pos = p - n * 0.004;
    var m = k;
    var att = 1.0 - R0;
    if (wallB(k)) { att *= mb.b[k].axis.xyz; }
    var L = vec3f(0.0);
//#if SSS
    if (mb.kind[k] > 1.5 && mb.kind[k] < 2.5 && !hasB(k, 32u)) {
      let sssOpK = clamp(mb.b[k].tab[19].w, 0.0, 1.0);
      if (sssOpK > 0.0) {
        cb = k;
        let aoB = waxOcclusion(p, n);
        dir = normalize(mix(dir, normalize(mix(rd, -n, 0.12)), sssOpK));
        L += att * sssOpK * waxNear(n, aoB, 0.25);
        att *= mix(1.0, 0.44, sssOpK);
      }
    }
//#endif
    // Surfaces met inside are of two sorts. Where the ray is sent back (by the air outside, a metal) it
    // has bounced, and bounces are counted as plain glass counts them. Where another clear or milky
    // body begins, the ray goes on into it and that is no bounce: two bodies of the same glass then draw
    // exactly as one piece would.
    var b = 0;
    var filmed = false;                     // a film has sent the light back inside at least once
//#if INNER
    var turns = 0;                          // how often cracks have turned this path (not bounces either)
    for (var e = 0; e < MAX_NB + 2 * COUNT + 6; e++) {
//#else
    for (var e = 0; e < MAX_NB + 2 * COUNT; e++) {
//#endif
      if (b >= u.nb || max3(att) < 0.02) { break; }
      var q: vec3f;
      var nq: vec3f;
      var nx: i32;
      let d0 = dir;
      let s = stretchB(pos, &dir, m, lam, &q, &nq, &nx);
//#if INNER
      if (hasB(m, 8u)) {
        // what lies inside the body along this stretch (inner.wgsl): what it adds is gathered, what it
        // takes is taken, and a crack may turn the path on the way
        cb = m;
        let inn = inner(pos, d0, s, iorB(m, lam), b);
        L += att * inn.light;
        att *= inn.keep;
//#if CRACK
        if (inn.t >= 0.0) {
          L += throughB(m, pos, d0, inn.t, lam, &att);
          // after so many turns the path is taken to leave the way it then heads
          if (turns >= i32(ibod[m].crackC.w)) {
            L += att * crackSeen(inn.dir);
            break;
          }
          turns += 1;
          pos = pos + d0 * inn.t;
          dir = inn.dir;
//#if BIREF
          kw = inn.dir * length(kw);
//#endif
          continue;
        }
//#endif
      }
//#endif
//#if SSS
      if (e == 0 && mb.kind[m] > 1.5 && mb.kind[m] < 2.5 && !hasB(m, 32u)) {
        let sssOp0 = clamp(mb.b[m].tab[19].w, 0.0, 1.0);
        if (sssOp0 > 0.0) {
          cb = m;
          L += att * (sssOp0 * 1.65 * waxNear(n, waxOcclusion(p, n), s));
        }
      }
//#endif
      L += throughB(m, pos, d0, s, lam, &att);
      var c = -dot(dir, nq);
      if (c < 0.0) { nq = -nq; c = -c; }
      nq = faceIncident(nq, -dir);
      if (paintB(nx)) {
        // a painted body met from inside a clear one: seen, and the path ends there
        L += att * paintB_seen(nx, q, nq, dir, iorB(m, lam));
        break;
      }
//#if BODY_FROST
      if (sampledInterfaceB(m,nx)) {
        L += att * secondaryFrostB(q,dir,nq,m,nx,lam);
        break;
      }
//#endif
      let o = max(m, nx);
      var R = surfB(m, nx, c, q, nq, lam);
      let al = mb.b[o].surf.x;
      var metalNormal = nq;
      var metalTint = vec3f(1.0);
      var finishVariance = 0.0;
//#if LIVE_CONDUCTOR_FINISH
      if (opaqueB(nx)) {
        let finish = conductorFinishB(nx, q, nq, dir);
        metalNormal = finish.normal;
        metalTint = finish.tint;
        finishVariance = finish.variance;
      }
//#endif
      let mir = reflect(dir, metalNormal);
//#if BIREF
      let kin = waveOf(m, dir, lam);        // (the wave as it meets the surface: both ways on are worked out from it)
//#endif
      var next = sg2 + 2.0 * al * al + finishVariance;      // the spread of what this surface sends back inside
      if (opaqueB(nx)) {
        // a metal met from inside this body
        L += att * mb.b[nx].glow.rgb * 0.3;
        R *= metalTint;
      } else {
        let na = iorB(m, lam);
        let nb = iorB(nx, lam);
//#if BIREF
        var od = crossW(kin, -nq, nx, lam);
//#else
        var od = pbr_dielectric_split(traceDielectricHit(nq, mediumB(m, na), mediumB(nx, nb)), -dir).transmitted;
//#endif
        var through = 1.0 - R;
        var wide = sg2 + widenB(al, na, c, nb, od, nq);
//#if ROUGH
        if (al > 0.0 && nx < 0 && mb.kind[o] < 2.5) {
          // A rough wall out into the air, met by a cone of light (frost.wgsl): what it sends back is
          // averaged over the cone (total reflection lets part of every cone out); of that, the share
          // that has by now been scattered every way is ended here as the light from all round; what
          // leaves is bent as the middle of what is not trapped, and spread by the facets it leaves by.
          cb = o;
          let fw = frWall(c, sg2, na, 1.0, vec2f(0.0));
          let we = frEnd(sg2, vec2f(0.0));
          L += att * (fw.x * we) * frDeep(s) * frAll(nq);
          through = vec3f(1.0 - fw.x);
          R = vec3f(fw.x * (1.0 - we));
          // (frExit and frVout take the wall's outward normal, the way the light is heading; nq here faces
          // the ray. Given nq, the light that left was sent straight back the way it came: a rough
          // body among several showed nothing of what lay behind or inside it)
          od = frExit(dir, -nq, -nq, fw, na);
          wide = frVout(sg2, fw, dot(od, -nq), na, vec2f(0.0));
          next = frNext(sg2, vec2f(0.0));
        }
//#endif
//#if SSS
        var sssOpM = 0.0;
        if (mb.kind[m] > 1.5 && mb.kind[m] < 2.5 && !hasB(m, 32u)) {
          sssOpM = clamp(mb.b[m].tab[19].w, 0.0, 1.0);
          if (sssOpM > 0.0) {
            od = normalize(mix(od, dir, 0.55 * sssOpM));
            through = mix(through, vec3f(1.0), sssOpM);
            R *= (1.0 - sssOpM);
            if (nx < 0) { wide = max(wide, 0.16 * sssOpM * sssOpM); }
          }
        }
//#endif
        if (dot(od, od) > 0.5) {
          var X = att * through;
          if (wallB(o)) { X *= mb.b[o].axis.xyz; }
          if (nx >= 0) {
            // on into the next body; what this surface sends back is taken as lit by what lies that way
            L += att * R * (seenB(mir, sqrt(sg2 + 2.0 * al * al)) + skyFarB(o, mir));
            att = X;
            sg2 = wide;
            dir = od;
            pos = q - nq * 0.004;
            m = nx;
            continue;
          }
          // out into the air: followed on through whatever it meets
          // (light the film has sent back inside at least once sees the film's sky where it leaves)
          var sky = vec3f(0.0);
          if (filmed) { sky = skyDimB(o, od); }
//#if SSS
          var airOut = vec3f(0.0);
          if (sssOpM < 0.999) { airOut = beyondB(q - nq * 0.006, od, nx, lam, wide); }
          if (sssOpM > 0.0) {
            cb = m;
            airOut = mix(airOut, waxExit(q, -nq, dir), sssOpM);
          }
          L += X * (airOut + sky);
//#else
          L += X * (beyondB(q - nq * 0.006, od, nx, lam, wide) + sky);
//#endif
        }
      }
      if (!opaqueB(nx) && mb.b[o].film.x > 0.0) { filmed = true; }
      att *= R;
      sg2 = next;
//#if BIREF
      // (what is sent back keeps the part of the wave that lies along the surface: an extraordinary wave
      // that has no such reflection ends here)
      dir = crossW(kin, metalNormal, m, lam);
      if (dot(dir, dir) < 0.5) { break; }
//#else
      dir = mir;
//#endif
      pos = q + nq * 0.004;
      b++;
    }
    acc += w * L;
    wsum += w;
//#if BIREF
    one += L / f32(modes);
//#else
    one = L;
//#endif
  }
//#if BIREF
  var top = racc / rsum;
//#else
  var top = racc / wsum;
//#endif
  var res = envR * top + acc / wsum;
  if (alike) {
    top = r1;
    res = envR * r1 + one;
  }
//#if ROUGH
  // what the stage that blurs the inside of rough glass is told about this surface: which way it faces,
  // and what it reflects itself (which is not seen through it, and is not blurred)
  if (frFace.z == 0.0) { frFace = vec3f(n.xy, 1.0); }
  frTopLight += top * envR;
  frTopShare += dot(top, vec3f(0.33333));
//#endif
//#if BODY_WALK
  // The random-walk body's transmitted radiance is available here.
//#endif
  return res;
//#endif
}

// the reflection blended in where a ray grazes the outline
fn grazeB(p: vec3f, rd: vec3f) -> vec3f {
  let n = nrm(p);
  let k = nearestOf(dist4(p));
  cb = k;
  if (paintB(k)) { return paintB_seen(k, p, faceIncident(n, -rd), rd, 1.0); }
  let mir = reflect(rd, n);
  var R = vec3f(1.0);
//#if METAL
  if (mb.kind[k] > 0.5 && mb.kind[k] < 1.5) { R = metalRGB(0.05, 1.0, 0.0); }
//#endif
  let baseG = R * (seenB(mir, 1.41421 * mb.b[k].surf.x) + skyB(k, mir));
//#if SSS
  if (mb.kind[k] > 1.5 && mb.kind[k] < 2.5 && mb.b[k].surf.z <= 1.001) {
    let sssOpG = clamp(mb.b[k].tab[19].w, 0.0, 1.0);
    if (sssOpG > 0.0) {
      let fG = fresnel(0.60, 1.0, mb.ior[k]);
      let waxG = fG * baseG + (1.0 - fG) * waxNear(n, 1.0, 0.08);
      return mix(baseG, waxG, sssOpG * smoothstep(0.0, 0.20, sqrt(clamp(mb.b[k].surf.x, 0.0, 1.0))));
    }
  }
//#endif
  return baseG;
}
