//#if INNER
// ---- what lies inside the body ---------------------------------------------------------------------------
// Put into the material tracer only when the body has something inside it (gf_core::inner::Parts): a tint
// that changes through the body, cracks, the grains of precious opal or the platelets of aventurine, the
// silky fibres of tiger's eye, a glow under pressure.
//
// None of it is in the distance volume. Once the march has said how long a straight stretch inside the
// body is, inner() works out by arithmetic what that stretch meets: where a crack turns it, what is added
// to the light along it, and what is left of the light that goes on. Everything is fixed in the shape's
// own frame and drawn from whole-number hashes of where it is, so it is the same in every frame and from
// every side: nothing in here is random.

struct IU {
  crackA: vec4f,      // x: cells of the cracks' lattice per unit of length   y: the share of cells that hold a crack
                      // z: the largest crack's radius, in cells   w: how far the cracks have grown (0 to 1)
  crackB: vec4f,      // x: the air gap where a crack is widest, nm   y: 0 = flat, straight-edged faces .. 1 = round shells
                      // z: how far below the surface cracks reach (large = everywhere)   w: which arrangement
  crackC: vec4f,      // x: how far a rough face spreads what it sends back, radians   y: 0 = a mirror .. 1 = a white veil
                      // z: the width of a pixel at the shape, in cells   w: turns allowed along one path
  spec: array<vec4f, 8>,      // the spectrum in eight bands: rgb = the band's share of white, w = its wavelength, nm
  playA: vec4f,       // x: grains per unit of length   y: the share of grains that send light back   z: which arrangement
                      // w: 1 = flat mirror platelets instead of opal's grains
  playB: vec4f,       // x: the step between the places looked at along a ray   y: how strongly   z: how far
                      // neighbouring grains share a direction   w: how much more than its share of white a grain sends back
  playC: vec4f,       // x: the longest wavelength the grains send back, nm   y: how softened the lights they send back are, radians
  playD: vec4f,       // rgb: what the body takes from light on its way to a grain and back, per unit of length
  glint: vec4f,       // rgb: the platelets' colour   w: a platelet's radius, in cells
  white: vec4f,       // rgb: one over white light added up a nanometre at a time   w: the width of a grain's band, nm
  gradA: vec4f,       // xyz: the tint's axis, in the shape's own frame   w: 0 along it, 1 outward from the middle, 2 in bands
  gradB: vec4f,       // rgb: the second tint less the first, as absorption per unit of length
                      // w: for a milky body, the length over which the tint of where its light comes from is shown (0: clear)
  gradC: vec4f,       // x: where the change lies   y: how wide it is   z: bands across the shape's radius   w: how wavy the bands are
  silkA: vec4f,       // xyz: the fibres' direction, in the shape's own frame   w: how strongly they scatter, per unit of length
  silkB: vec4f,       // x: how softened the lights they send back are, radians   y: how wavy the fibres are
  piezo: vec4f,       // rgb: light given off under pressure, per unit of length   w: how strongly a growing crack's rim glows
}
@group(0) @binding(32) var<uniform> iu: IU;

// a whole number scrambled into another
fn ihash(v: u32) -> u32 {
  let x = v * 747796405u + 2891336453u;
  let w = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (w >> 22u) ^ w;
}
// three numbers in 0..1 from one scrambled number
fn ten3(h: u32) -> vec3f { return vec3f(f32(h & 1023u), f32((h >> 10u) & 1023u), f32((h >> 20u) & 1023u)) * (1.0 / 1023.0); }
// a scrambled number for a cell of a lattice
fn cellHash(c: vec3i, seed: u32) -> u32 { return ihash(u32(c.x) + ihash(u32(c.y) + ihash(u32(c.z) + seed))); }
// a direction drawn evenly from all directions by two numbers in 0..1
fn sphereDir(r: vec2f) -> vec3f {
  let z = 1.0 - 2.0 * r.x;
  let s = sqrt(max(1.0 - z * z, 0.0));
  return vec3f(s * cos(6.2831853 * r.y), s * sin(6.2831853 * r.y), z);
}

// The lattices of cells are turned against the shape's own axes, so that nothing about them lines up with them.
const LAT0: vec3f = vec3f(0.36, 0.48, 0.80);
const LAT1: vec3f = vec3f(-0.80, 0.60, 0.0);
const LAT2: vec3f = vec3f(-0.48, -0.64, 0.60);
fn toLat(v: vec3f) -> vec3f { return vec3f(dot(LAT0, v), dot(LAT1, v), dot(LAT2, v)); }
fn fromLat(v: vec3f) -> vec3f { return v.x * LAT0 + v.y * LAT1 + v.z * LAT2; }
// the shape's own frame -> what the camera sees (toObj() the other way round)
fn fromObj(v: vec3f) -> vec3f { return vec3f(dot(u.obj0.xyz, v), dot(u.obj1.xyz, v), dot(u.obj2.xyz, v)); }
// two directions across a (Duff et al. 2017)
fn across1(a: vec3f) -> vec3f {
  let sg = select(-1.0, 1.0, a.z >= 0.0);
  let aa = -1.0 / (sg + a.z);
  return vec3f(1.0 + sg * a.x * a.x * aa, sg * a.x * a.y * aa, -sg * a.x);
}
fn across2(a: vec3f) -> vec3f {
  let sg = select(-1.0, 1.0, a.z >= 0.0);
  let aa = -1.0 / (sg + a.z);
  return vec3f(a.x * a.y * aa, sg + a.y * a.y * aa, -a.y);
}

//#if CRACK
// the lights as a crack's face sends them back along d: sharp for a clean break, spread for a rough one
//#if CRACK_ROUGH
fn crackSeen(d: vec3f) -> vec3f { return envL(d, iu.crackC.x); }
//#else
fn crackSeen(d: vec3f) -> vec3f { return env(d); }
//#endif

// What an air gap between two faces of the body sends back, as red, green and blue.
// ci: cosine of the angle the light meets the faces at.  gap: its width, nm.  n: the body's index.
// dgap: how much the width changes across one pixel, nm.
//
// Both polarisations, every reflection back and forth across the gap (the Airy sum), and past the
// critical angle the light that leaks across a gap thinner than its wavelength: so a gap of nothing
// reflects nothing, a thin one shows the colours of a soap film, a wide one is two plain surfaces, and
// past the critical angle a wide one is a perfect mirror. The colours come from eight bands of the
// spectrum. Fringes finer than a band of wavelengths, or finer than the pixel, are averaged away into
// the plain two-surface figure, which is what the eye and the camera do: nothing shimmers.
fn crackR(ci: f32, gap: f32, n: f32, dgap: f32) -> vec3f {
  let c = clamp(ci, 0.02, 1.0);
  let cg2 = 1.0 - n * n * (1.0 - c * c);          // cosine squared of the angle in the gap: negative past the critical angle
  let ps = n * c;                                  // how each polarisation meets the gap
  let pp = c / n;
  let gs = (ps * ps - cg2) * (ps * ps - cg2) / (4.0 * ps * ps);
  let gp = (pp * pp - cg2) * (pp * pp - cg2) / (4.0 * pp * pp);
  if (cg2 <= 0.0) {
    // past the critical angle a gap of a few wavelengths lets nothing through
    if (gap * gap * -cg2 > 1.0e6) { return vec3f(1.0); }
  }
  // the two faces as plain surfaces, one after the other
  var plain = 1.0;
  if (cg2 > 0.0) {
    let cg = sqrt(cg2);
    let rs = (ps - cg) / (ps + cg);
    let rp = (pp - cg) / (pp + cg);
    plain = rs * rs / (1.0 + rs * rs) + rp * rp / (1.0 + rp * rp);
    if (cg * gap > 9000.0 || cg * dgap > 900.0) { return vec3f(plain); }
  }
  var col = vec3f(0.0);
  for (var j = 0; j < 8; j++) {
    let lam = iu.spec[j].w;
    let D = 6.2831853 * gap / lam;
    let y = cg2 * D * D;
    var R: f32;
    if (y > 0.0) {
      let x = sqrt(y);
      var sx = 1.0;
      if (x > 1e-3) { sx = sin(x) / x; }
      let k = D * D * sx * sx;
      let coh = 0.5 * (gs * k / (1.0 + gs * k) + gp * k / (1.0 + gp * k));
      // how much of the fringe is left once it is averaged over the band (37.5 nm wide) and over the pixel
      let xb = x * 37.5 / lam;
      var vis = 1.0;
      if (xb > 1e-3) { vis = sin(xb) / xb; }
      let xp = 12.566371 * sqrt(cg2) * dgap / lam;
      vis *= exp(-xp * xp / 12.0);
      R = mix(plain, coh, vis);
    } else {
      let x = min(sqrt(-y), 12.0);
      var sh = 1.0;
      if (x > 1e-3) { sh = sinh(x) / x; }
      let k = D * D * sh * sh;
      R = 0.5 * (gs * k / (1.0 + gs * k) + gp * k / (1.0 + gp * k));
    }
    col += iu.spec[j].rgb * R;
  }
  return clamp(col, vec3f(0.0), vec3f(1.0));
}

// What the cracks do to one straight stretch of a path inside the body.
struct Crack {
  t: f32,             // how far along the stretch a crack turns the path (negative: none does)
  dir: vec3f,         // and where the path heads from there
  keep: vec3f,        // what is left of the light going on: past every face it crossed, or off the one that turned it
  light: vec3f,       // what the faces along the way add: each one's weaker part, read straight from the lights
}

// The cracks crossed by the stretch from pos along dir, s long, inside a body of index n.
//
// Space is divided into cells (a lattice fixed in the shape's own frame); a cell may hold one crack: a
// facet that is a cap of a sphere (a conchoidal break is shell-shaped; with no curvature it is a flat
// plane), with its own place, tilt, size, outline and moment of starting, all drawn from the cell's
// position. A facet lies wholly inside its cell, so the ray walks the cells it passes through and asks
// each for its one facet: where the ray meets the sphere is a quadratic.
//
// At a crossing the gap reflects part and passes part (crackR). Following both would double the work
// at every crack, and choosing one by chance would make grain. Instead the path goes with the larger
// part (it turns at a mirror, and passes through a face that mostly transmits) and the smaller part is
// read at once from the lights in the direction it would have gone. Light is neither made nor lost.
fn cracks(pos: vec3f, dir: vec3f, s: f32, n: f32) -> Crack {
  var out: Crack;
  out.t = -1.0;
  out.dir = dir;
  out.keep = vec3f(1.0);
  out.light = vec3f(0.0);
  let cells = iu.crackA.x;
  let o = toObj(pos);
  let d = toObj(dir);
  let dl = toLat(d);                               // the ray's direction in the lattice (a unit vector)
  let ud = dl * cells;                             // ... and how far it moves through the lattice per unit of length
  let u0 = toLat(o) * cells + iu.crackB.w * vec3f(17.13, 7.77, 3.31);
  let seed = u32(iu.crackB.w) * 2654435761u + 12345u;
  // walking the cells: for each axis, how far along the ray the next wall is, and how far apart walls are
  let safe = select(ud, vec3f(1e-9), abs(ud) < vec3f(1e-9));
  let rinv = 1.0 / safe;
  let stp = vec3i(sign(safe));
  var c = vec3i(floor(u0));
  var tm = (vec3f(c) + step(vec3f(0.0), safe) - u0) * rinv;
  let dt = abs(rinv);
  var t0 = 2e-3 / cells;                           // (a path that has just left a crack does not meet it again at once)
  for (var i = 0; i < 48; i++) {
    let t1 = min(min(tm.x, tm.y), min(tm.z, s));
    let h = cellHash(c, seed);
    if (f32(h >> 8u) * (1.0 / 16777216.0) < iu.crackA.y) {
      let h2 = ihash(h);
      let h3 = ihash(h2);
      let h4 = ihash(h3);
      let k3 = ten3(h3);                           // tilt (two numbers) and size
      let m3 = ten3(h4);                           // curvature, when it starts, how wide it opens
      let rfull = iu.crackA.z * (0.55 + 0.45 * k3.z);
      // growth: each crack starts at its own moment and spreads from its middle
      let g = clamp((iu.crackA.w - 0.6 * m3.y) / 0.4, 0.0, 1.0);
      let r = rfull * g;
      let w0 = u0 - (vec3f(c) + 0.5 + (ten3(h2) - 0.5) * (1.0 - 2.0 * rfull));
      // does the ray pass within the facet's reach at all?
      let wc = w0 - ud * (dot(w0, ud) / dot(ud, ud));
      if (dot(wc, wc) < r * r) {
        let a = sphereDir(k3.xy);
        // the sphere the facet is a cap of: curvature kap, touching the facet's middle with a as its normal there
        let kap = iu.crackB.y * (0.25 + 0.75 * m3.x) * 0.9 / rfull;
        let A = 0.5 * kap * dot(ud, ud);
        let B = dot(ud, a) + kap * dot(w0, ud);
        let C = dot(w0, a) + 0.5 * kap * dot(w0, w0);
        var ta = 1e9;
        var tb = 1e9;
        if (abs(A) < 1e-7 * abs(B)) {
          ta = -C / B;
        } else {
          let disc = B * B - 4.0 * A * C;
          if (disc > 0.0) {
            let q = -0.5 * (B + select(-1.0, 1.0, B >= 0.0) * sqrt(disc));
            ta = min(q / A, C / q);
            tb = max(q / A, C / q);
          }
        }
        let e1 = across1(a);
        let e2 = across2(a);
        let cut = ten3(ihash(h4));                 // where three straight edges cut the outline
        let ang = ten3(ihash(h4 + 77u));           // and how they are turned
        for (var e = 0; e < 2; e++) {
          let tt = select(ta, tb, e == 1);
          if (tt <= t0 || tt > t1 || out.t >= 0.0) { continue; }
          let wl = w0 + tt * ud;
          let rr = dot(wl, wl);
          if (rr >= r * r) { continue; }
          // how far inside the facet's edge this is, as a share of its radius: the round rim, then three
          // straight edges (a flat break in a crystal ends on straight lines, a shell-shaped one on a curve)
          var edge = 1.0 - sqrt(rr) / r;
          let x2 = vec2f(dot(wl, e1), dot(wl, e2)) / r;
          for (var k = 0; k < 3; k++) {
            let th = 2.0943951 * (f32(k) + 0.8 * ang[k]);
            let reach = mix(0.2 + 0.8 * cut[k], 1.3, iu.crackB.y);
            edge = min(edge, reach - dot(x2, vec2f(cos(th), sin(th))));
          }
//#if CRACK_DEPTH
          // cracks that start at the surface: none deeper than the limit (which deepens as they grow)
          edge = min(edge, (iu.crackB.z * g + map(o + d * tt)) * cells / r);
//#endif
          if (edge <= 0.0) { continue; }
          // the gap: nothing at the crack's edge, widening steeply and then levelling off, as a crack
          // held open is (so colour bands crowd along the edge and broaden inwards)
          let ee = min(edge, 1.0);
          let prof = sqrt(ee * (2.0 - ee));
          let wide = iu.crackB.x * (0.5 + m3.z);
          let nl = normalize(a + kap * wl);
          let ci = abs(dot(dl, nl));
          let dgap = wide * (1.0 - ee) / max(prof, 0.03) * iu.crackC.z / (r * max(ci, 0.3));
          var R = crackR(ci, wide * prof, n, dgap);
          // a rough break is a white veil rather than a mirror: it sends back about half, whatever the angle
          R = mix(R, vec3f(0.45), iu.crackC.y);
          let back = fromObj(fromLat(reflect(dl, nl)));
          let ext = exp(-u.absorb.xyz * tt);
//#if PIEZO
          // under pressure a crack glows at the rim where it is still growing (and a little where it has stopped)
          out.light += out.keep * ext * (iu.piezo.rgb / max(iu.piezo.r, 1e-6)) * (iu.piezo.w * exp(-8.0 * edge) * select(0.3, 1.0, g < 1.0));
//#endif
          if (R.g > 0.5) {
            out.light += out.keep * ext * (1.0 - R) * crackSeen(dir);
            out.keep *= R;
            out.t = tt;
            out.dir = back;
          } else {
            out.light += out.keep * ext * R * crackSeen(back);
            out.keep *= 1.0 - R;
          }
        }
      }
    }
    if (t1 >= s || out.t >= 0.0) { break; }
    t0 = t1;
    if (tm.x <= tm.y && tm.x <= tm.z) {
      c.x += stp.x;
      tm.x += dt.x;
    } else if (tm.y <= tm.z) {
      c.y += stp.y;
      tm.y += dt.y;
    } else {
      c.z += stp.z;
      tm.z += dt.z;
    }
  }
  return out;
}
//#endif

//#if PLAY
// Places looked at along a ray for grains (or platelets).
const PLAY_N: i32 = 6;

// What the grains along the stretch from pos along dir, s long, send back towards the eye.
//
// Opal: the body is made of grains, each a stack of layers of silica spheres in its own direction. A
// grain sends back, as a mirror would, only the one wavelength that its layers' spacing selects for
// the angle the light meets them at (Bragg's law): straight on the longest, more steeply shorter ones. So
// each grain is a patch of one pure colour that changes as the stone turns. The ray is looked at in a
// few places along its first stretch: there the grain is found (a cell of a lattice fixed in the
// shape), its direction drawn from the cell (or, with roll, mostly from a smooth field across the
// shape, so that neighbours flash together and the flash rolls across the stone), and the lights read
// in the mirrored direction, softened.
//
// Platelets (aventurine, goldstone): each cell may hold a small flat mirror disc instead, at its own
// tilt; the ray is tested against the disc exactly, so a sparkle belongs to its platelet and appears
// only where that platelet mirrors a light.
fn play(pos: vec3f, dir: vec3f, s: f32) -> vec3f {
  let G = iu.playA.x;
  let seed = u32(iu.playA.z) * 2246822519u + 777u;
  let off = iu.playA.z * vec3f(3.17, 11.3, 5.71);
  let x0 = toLat(toObj(pos)) * G + off;           // where the ray starts, and how it moves, in the lattice
  let dl = toLat(toObj(dir)) * G;
  var sum = vec3f(0.0);
  var last = vec3i(-999999);
  for (var k = 0; k < PLAY_N; k++) {
    let t = (f32(k) + 0.5) * iu.playB.x;
    if (t > s) { break; }
    let c = vec3i(floor(x0 + dl * t));
    if (all(c == last)) { continue; }
    last = c;
    let h = cellHash(c, seed);
    if (f32(h >> 8u) * (1.0 / 16777216.0) >= iu.playA.y) { continue; }
    let r3 = ten3(ihash(h));
    let r4 = ten3(ihash(h + 91u));
    // the grain's direction, in the shape's own frame: its own, or mostly the smooth field's there
    let o = toObj(pos + dir * t);
    let field = normalize(vec3f(sin(1.7 * o.y + 0.3) + 0.4 * sin(2.9 * o.z), sin(1.9 * o.z + 1.1) + 0.4 * sin(2.3 * o.x), 1.2 + sin(1.3 * o.x + 2.0)));
    var gd = normalize(mix(sphereDir(r3.xy), field, iu.playB.z));
    if (dot(gd, toObj(dir)) > 0.0) { gd = -gd; }
    let gw = fromObj(gd);
    let back = reflect(dir, gw);
//#if GLITTER
    // the platelet: a disc about the cell's middle, across gd; where the ray meets its plane
    let mid = vec3f(c) + 0.5 + (r4 - 0.5) * 0.3;
    let gl = toLat(gd);
    let den = dot(dl, gl);
    if (abs(den) < 1e-6) { continue; }
    let th = dot(mid - x0, gl) / den;
    if (th < 0.0 || th > s) { continue; }
    let off2 = x0 + dl * th - mid;
    let rad = iu.glint.w * (0.6 + 0.4 * r3.z);
    if (dot(off2, off2) > rad * rad) { continue; }
    sum += iu.glint.rgb * envL(back, iu.playC.y) * exp(-iu.playD.rgb * th);
//#else
    // the wavelength this grain sends back at this angle (each grain's spacing differs a little)
    let lam = iu.playC.x * abs(dot(dir, gw)) * (0.9 + 0.1 * r3.z);
    if (lam < 390.0 || lam > 710.0) { continue; }
    let band = min(specW(lam) * iu.white.rgb * (iu.white.w * iu.playB.w), vec3f(1.0));
    sum += max(band, vec3f(0.0)) * envL(back, iu.playC.y) * exp(-iu.playD.rgb * t) * (2.0 / f32(PLAY_N));
//#endif
  }
  return sum * iu.playB.y;
}
//#endif

//#if SILK
// Fibres in their thousands, all lying one way: what they scatter is light turned about the fibres'
// direction, keeping its part along them. Seen from inside along dir, a fibre sends back light from
// every direction on the cone about the fibres that dir itself lies on; where that cone takes in a
// light the stone shows a band of it, across the fibres, which glides as the stone or the light moves
// (chatoyancy: the cat's eye). Wavy fibres (tiger's eye) turn the cone from place to place, so the
// band breaks into stripes that change places as the stone turns.
const SILK_N: i32 = 12;
fn silkLit(p: vec3f, dir: vec3f) -> vec3f {
  let o = toObj(p);
  var a = iu.silkA.xyz;
  let e1 = across1(a);
  let e2 = across2(a);
  a = normalize(a + iu.silkB.y * sin(7.0 * dot(o, e1) + 1.7 * sin(3.0 * dot(o, e2))) * e1);
  let aw = fromObj(a);
  let c = dot(dir, aw);
  let r = sqrt(max(1.0 - c * c, 0.0));
  let w1 = fromObj(across1(a));
  let w2 = fromObj(across2(a));
  var sum = vec3f(0.0);
  for (var k = 0; k < SILK_N; k++) {
    let ph = 6.2831853 * (f32(k) + 0.5) / f32(SILK_N);
    sum += envL(c * aw + r * (cos(ph) * w1 + sin(ph) * w2), iu.silkB.x);
  }
  return sum / f32(SILK_N);
}
//#endif

//#if GRAD
// How far the tint has gone over to the second colour at o (in the shape's own frame), 0 to 1.
fn gradF(o: vec3f) -> f32 {
  let w = iu.gradC.y;
  var x: f32;
  if (iu.gradA.w < 0.5) {
    x = dot(o, iu.gradA.xyz) / 1.3;
  } else if (iu.gradA.w < 1.5) {
    x = length(o) / 1.3;
  } else {
    // bands about the middle, as an agate's are laid down: slightly squashed, and wavy
    let wv = sin(2.3 * o.x + 1.1 * sin(3.1 * o.y)) * sin(1.7 * o.z + 0.5) + 0.5 * sin(4.1 * o.y + 2.0 * o.x);
    let r = length(o * vec3f(1.0, 1.25, 0.85)) / 1.3 + iu.gradC.w * 0.12 * wv;
    x = 0.5 * cos(6.2831853 * r * iu.gradC.z);
  }
  let t = clamp((x - iu.gradC.x + w) / (2.0 * w), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}
// What the change of tint does to light over a stretch from pos along dir, len long (the first tint is
// taken off already, along every stretch): the second tint, by how far it has taken over along the
// way, from five places (Simpson's rule). A milky body shows instead the tint of where its light comes
// from, a little way in, on the first stretch.
fn gradKeep(pos: vec3f, dir: vec3f, len: f32, b: i32) -> vec3f {
  if (iu.gradB.w > 0.0) {
    if (b > 0) { return vec3f(1.0); }
    return exp(-iu.gradB.rgb * (gradF(toObj(pos + dir * min(len, 0.1 * iu.gradB.w))) * iu.gradB.w));
  }
  let h = 0.25 * len;
  let f = gradF(toObj(pos)) + 4.0 * gradF(toObj(pos + dir * h)) + 2.0 * gradF(toObj(pos + dir * (2.0 * h))) + 4.0 * gradF(toObj(pos + dir * (3.0 * h))) + gradF(toObj(pos + dir * len));
  return exp(-iu.gradB.rgb * (len * f / 12.0));
}
//#endif

// What a straight stretch of a path inside the body meets: from pos along dir, s long (as far as the
// surface), in a body of index n; b: how many times the path has been turned already.
struct Inner {
  t: f32,             // how far along a crack turns the path (negative: none does, the stretch runs to the surface)
  dir: vec3f,         // where it heads from there
  keep: vec3f,        // what is left of the light that goes on, beyond what the tint takes
  light: vec3f,       // what is added along the way (as seen from where the stretch begins)
}
fn inner(pos: vec3f, dir: vec3f, s: f32, n: f32, b: i32) -> Inner {
  var o: Inner;
  o.t = -1.0;
  o.dir = dir;
  o.keep = vec3f(1.0);
  o.light = vec3f(0.0);
  var len = max(s, 0.0);
//#if CRACK
  let ck = cracks(pos, dir, len, n);
  o.t = ck.t;
  o.dir = ck.dir;
  o.keep = ck.keep;
  o.light = ck.light;
  if (ck.t >= 0.0) { len = ck.t; }
//#endif
//#if PLAY
  if (b < 2) { o.light += o.keep * play(pos, dir, len); }
//#endif
//#if SILK
  if (b < 2) {
    // what the fibres scatter towards the eye along the stretch, and what is left that goes on
    let k = iu.silkA.w;
    let kt = vec3f(k) + u.absorb.xyz;
    let took = k / kt * (1.0 - exp(-kt * len));
    // (the fibres have the body's own colour: golden in tiger's eye, white where the body is left white)
    o.light += o.keep * took * clamp(mu.body.rgb, vec3f(0.0), vec3f(1.0)) * silkLit(pos + dir * min(0.5 * len, 0.5 / k), dir);
    o.keep *= exp(-k * len);
  }
//#endif
//#if PIEZO
  // the glow under pressure, given off evenly through the body
  o.light += o.keep * iu.piezo.rgb * len * exp(-0.5 * u.absorb.xyz * len);
//#endif
//#if GRAD
  o.keep *= gradKeep(pos, dir, len, b);
//#endif
  return o;
}
//#endif
