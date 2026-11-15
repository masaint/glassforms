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
  playA: vec4f,       // x: grains per unit of length   y: the share of patches (platelets: of cells) that send light back   z: which arrangement
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
  playE: vec4f,       // x: how thick a grain is across the layers, as a share of its width   y: how closely the grains of
                      // one patch share a direction   z: a patch's width, in grains   w: a grain's reflectance in its band
}
//#if BODIES
// (among several bodies each has its own numbers: every `iu.` below reads those of the body in hand)
@group(0) @binding(33) var<uniform> ibod: array<IU, 4>;
//#else
@group(0) @binding(32) var<uniform> iu: IU;
//#endif

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
// a number in 0..1 from a scrambled number
fn unit1(h: u32) -> f32 { return f32(h >> 8u) * (1.0 / 16777216.0); }

//#if CRACK
// Smooth noise over a plane, 0..1, fixed by seed: values at whole-number points, blended between.
fn vnoise2(p: vec2f, seed: u32) -> f32 {
  let i = floor(p);
  let f = p - i;
  let w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let x = u32(i32(i.x));
  let y = u32(i32(i.y));
  let a = unit1(ihash(x + ihash(y + seed)));
  let b = unit1(ihash(x + 1u + ihash(y + seed)));
  let c = unit1(ihash(x + ihash(y + 1u + seed)));
  let d = unit1(ihash(x + 1u + ihash(y + 1u + seed)));
  return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}
// three of them at finer and finer scales, turned against each other: 0..1, mostly near the middle
fn fbm2(p: vec2f, seed: u32) -> f32 {
  let q = vec2f(0.8 * p.x - 0.6 * p.y, 0.6 * p.x + 0.8 * p.y);
  return 0.5 * vnoise2(p, seed) + 0.3 * vnoise2(2.03 * q + 5.1, seed + 1u) + 0.2 * vnoise2(4.07 * p - 3.7, seed + 2u);
}
//#endif

//#if CRACK
// the lights as a crack's face sends them back along d: sharp for a clean break, spread for a rough one
//#if BODIES
fn crackSeen(d: vec3f) -> vec3f {
  if (iu.crackC.y > 0.0) { return envL(d, iu.crackC.x); }
  return env(d);
}
// the light from all round a face (what a rough veil sends back)
fn crackAll(d: vec3f) -> vec3f { return envL(d, 0.9); }
//#else
//#if CRACK_ROUGH
fn crackSeen(d: vec3f) -> vec3f { return envL(d, iu.crackC.x); }
fn crackAll(d: vec3f) -> vec3f { return envL(d, 0.9); }
//#else
fn crackSeen(d: vec3f) -> vec3f { return env(d); }
fn crackAll(d: vec3f) -> vec3f { return env(d); }
//#endif
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
// sheet that is flat or gently bent (shell-shaped in glass; in a crystal nearly flat, as often a saddle
// as a bowl), with its own place, tilt, size, ragged outline, mottled gap, rippled faces and moment of
// starting, all drawn from the cell's position. A sheet lies wholly inside its cell, so the ray walks the
// cells it passes through and asks each for its one sheet: where the ray meets it is a quadratic.
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
    if (unit1(h) < iu.crackA.y) {
      let h2 = ihash(h);
      let h3 = ihash(h2);
      let h4 = ihash(h3);
      let h5 = ihash(h4 + 13u);
      let k3 = ten3(h3);                           // tilt (two numbers) and size
      let m3 = ten3(h4);                           // how curved, when it starts, how wide it opens
      let q3 = ten3(h5);                           // the second curvature, the outline's squash, where it began
      let rfull = iu.crackA.z * (0.55 + 0.45 * k3.z);
      // growth: each crack starts at its own moment and spreads from its middle
      let g = clamp((iu.crackA.w - 0.6 * m3.y) / 0.4, 0.0, 1.0);
      let r = rfull * g;
      let w0 = u0 - (vec3f(c) + 0.5 + (ten3(h2) - 0.5) * (1.0 - 2.0 * rfull));
      // does the ray pass within the facet's reach at all?
      let wc = w0 - ud * (dot(w0, ud) / dot(ud, ud));
      if (dot(wc, wc) < r * r) {
        var a = sphereDir(k3.xy);
//#if CRACK_DEPTH
        // a crack that starts at the surface (from a sudden chill, or a blow) runs into the body across
        // it: its sheet is stood upright to the surface nearest its middle, so from outside it shows as
        // a line and the cracks together as a network
        {
          let co = o - fromLat(w0) / cells;
          let hs = 0.01;
          let k = vec2f(1.0, -1.0);
//#if BODIES
          let gn = k.xyy * mapOB(co + k.xyy * hs) + k.yyx * mapOB(co + k.yyx * hs) + k.yxy * mapOB(co + k.yxy * hs) + k.xxx * mapOB(co + k.xxx * hs);
          let upright = iu.crackB.z < 1e8;
//#else
          let gn = k.xyy * map(co + k.xyy * hs) + k.yyx * map(co + k.yyx * hs) + k.yxy * map(co + k.yxy * hs) + k.xxx * map(co + k.xxx * hs);
          let upright = true;
//#endif
          let gl = toLat(gn / max(length(gn), 1e-9));
          let ap = a - gl * dot(a, gl);
          if (upright && dot(ap, ap) > 1e-4) { a = normalize(ap); }
        }
//#endif
        let e1 = across1(a);
        let e2 = across2(a);
        // The sheet: a gently bent surface through the facet's middle, height = half of k1 x^2 + k2 y^2
        // across it. A break in glass is shell-shaped (both bends one way); one in a crystal is nearly
        // flat, and as often a saddle or a trough as a bowl. Either way far flatter than a lens.
        let curve = iu.crackB.y;
        let kap = (0.12 + 0.5 * curve) * (0.3 + 0.7 * m3.x) / rfull;
        let k1 = kap * select(-1.0, 1.0, q3.x > 0.5 * (1.0 - curve));
        let k2 = kap * mix(2.0 * q3.y - 1.0, 0.6 + 0.4 * q3.y, curve);
        let p1 = dot(w0, e1);
        let p2 = dot(w0, e2);
        let v1 = dot(ud, e1);
        let v2 = dot(ud, e2);
        let A = -0.5 * (k1 * v1 * v1 + k2 * v2 * v2);
        let B = dot(ud, a) - (k1 * p1 * v1 + k2 * p2 * v2);
        let C = dot(w0, a) - 0.5 * (k1 * p1 * p1 + k2 * p2 * p2);
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
        let cut = ten3(ihash(h4));                 // where straight edges cut the outline
        let ang = ten3(ihash(h4 + 77u));           // and how they are turned
        let sd = ihash(h5 + 5u);                   // this crack's own noise
        for (var e = 0; e < 2; e++) {
          let tt = select(ta, tb, e == 1);
          if (tt <= t0 || tt > t1 || out.t >= 0.0) { continue; }
          let wl = w0 + tt * ud;
          let rr = dot(wl, wl);
          if (rr >= r * r) { continue; }
          // where on the sheet this is, as a share of the grown crack's radius (x2), and as a share of
          // the full one (xf: the sheet's markings stay put while it grows)
          let x2 = vec2f(dot(wl, e1), dot(wl, e2)) / r;
          let xf = x2 * g;
          // The outline: a squashed oval, cut by straight edges (a crystal splits along planes, glass
          // does not), then eaten into by a ragged, feathery rim.
          let sq = mix(0.55 + 0.45 * q3.y, 0.22 + 0.3 * q3.y, iu.crackC.y);
          let ca = cos(6.2831853 * ang.z);
          let sa = sin(6.2831853 * ang.z);
          let xe = vec2f(ca * x2.x + sa * x2.y, (ca * x2.y - sa * x2.x) / sq);
          var edge = 1.0 - length(xe);
          for (var k = 0; k < 3; k++) {
            let th = 2.0943951 * (f32(k) + 0.8 * ang[k]);
            let reach = mix(0.25 + 0.6 * cut[k], 1.4, curve);
            edge = min(edge, reach - dot(x2, vec2f(cos(th), sin(th))));
          }
          // (creased noise, so the rim runs in sharp bays and points rather than soft lobes)
          let jag = fbm2(xf * 2.4, sd);
          let crease = 1.0 - abs(2.0 * fbm2(xf * 5.5 + 1.3, sd + 7u) - 1.0);
          let lace = vnoise2(xf * 17.0, sd + 9u);
          edge -= 0.38 * (jag - 0.42) + 0.22 * (crease - 0.6) + 0.1 * (lace - 0.5);
//#if CRACK_DEPTH
          // cracks that start at the surface: none deeper than the limit (which deepens as they grow)
//#if BODIES
          // (the depth below the surface of the body in hand)
          if (iu.crackB.z < 1e8) { edge = min(edge, (iu.crackB.z * g + mapOB(o + d * tt)) * cells / r); }
//#else
          edge = min(edge, (iu.crackB.z * g + map(o + d * tt)) * cells / r);
//#endif
//#endif
          if (edge <= 0.0) { continue; }
          // The gap: nothing at the rim, widening steeply inwards, and nowhere even: the two faces touch
          // in patches (there the crack cannot be seen), and between them the gap swells and thins. So a
          // crack is a mottled sheet, its colours in bands that follow the swells (iris), its mirror
          // broken into patches.
          let ee = min(edge * 2.2, 1.0);
          let prof = sqrt(ee * (2.0 - ee));
          let sw = fbm2(xf * 1.7 + 3.3, sd + 11u);
          // feathers: near the rim the sheet frays into fine streaks running out from where it began
          let og0 = vec2f(cos(6.2831853 * q3.z), sin(6.2831853 * q3.z)) * 1.1;
          let fv = xf - og0;
          let fth = vnoise2(vec2f(atan2(fv.y, fv.x) * 70.0 + 9.0 * jag, length(fv) * 3.0), sd + 17u);
          let fray = mix(smoothstep(0.35, 0.65, fth), 1.0, smoothstep(0.05, 0.25, edge));
          // (a thin gap swells gently, so its colour bands are broad; a wide one is held open in patches)
          let thin = 1.0 - smoothstep(1500.0, 4000.0, iu.crackB.x);
          let open = mix(smoothstep(0.22, 0.6, sw), clamp(0.55 + 1.2 * (sw - 0.5), 0.0, 1.0), thin) * fray * (0.6 + 0.4 * smoothstep(0.15, 0.5, vnoise2(xf * 6.5, sd + 19u)));
          let wide = iu.crackB.x * (0.5 + m3.z);
          let gap = wide * prof * open;
          // the faces' own relief: shallow ripples in rings about where the break began (glass), fine
          // ridges running out from there (hackle), and a gentle waviness everywhere; it tilts the face
          // a little from place to place, so a flash lights it in patches and streaks, never all at once
          let og = vec2f(cos(6.2831853 * q3.z), sin(6.2831853 * q3.z)) * 1.1;
          let dv = xf - og;
          let dist = max(length(dv), 1e-4);
          let rad = dv / dist;
          let rip = sin(dist * 23.0 + 9.0 * jag) * (0.015 + 0.035 * curve);
          let hk = (vnoise2(vec2f(atan2(dv.y, dv.x) * 38.0, dist * 2.0), sd + 3u) - 0.5) * (0.03 + 0.15 * curve) * smoothstep(0.5, 1.6, dist);
          let wv = vec2f(vnoise2(xf * 4.0, sd + 21u), vnoise2(xf * 4.0, sd + 22u)) - 0.5;
          let wf = vec2f(vnoise2(xf * 13.0, sd + 23u), vnoise2(xf * 13.0, sd + 24u)) - 0.5;
          // and terraces: a break steps from one level to the next, each step's face tilted its own way, so
          // that from almost any side some steps of a crack catch a light while the rest stay dark
          let tq = vec2f(ca * xf.x - sa * xf.y, sa * xf.x + ca * xf.y) * 3.2 + 2.4 * vec2f(jag - 0.5, crease - 0.6) + 1.3 * vec2f(sin(2.3 * xf.y), sin(2.9 * xf.x));
          let tc = vec2i(floor(tq));
          let tr = ten3(ihash(u32(tc.x) + ihash(u32(tc.y) + sd))).xy - 0.5;
          let tilt = rip * rad + hk * vec2f(-rad.y, rad.x) + (0.035 + 0.12 * curve) * wv + (0.012 + 0.05 * curve) * wf + 0.3 * (1.0 - 0.7 * iu.crackC.y) * tr;
          let pw = wl - a * dot(wl, a);
          var nl = normalize(a - k1 * dot(pw, e1) * e1 - k2 * dot(pw, e2) * e2 + tilt.x * e1 + tilt.y * e2);
          let ci = abs(dot(dl, nl));
          // how much the gap changes across one pixel: along the rim, and over the swells
          let px = iu.crackC.z / (r * max(ci, 0.3));
          let hstep = 0.03;
          let sx = fbm2((xf + vec2f(hstep, 0.0)) * 1.7 + 3.3, sd + 11u);
          let sy = fbm2((xf + vec2f(0.0, hstep)) * 1.7 + 3.3, sd + 11u);
          let dopen = length(vec2f(sx - sw, sy - sw)) / hstep * mix(2.5, 1.2, thin) * g;
          let dgap = wide * ((1.0 - ee) / max(prof, 0.03) * 2.2 * open + prof * dopen) * px;
          let back = fromObj(fromLat(reflect(dl, nl)));
          let ext = exp(-u.absorb.xyz * tt);
          // a rough break is partly a white veil: what that part sends back is the light from all round,
          // whatever the angle, thicker in streaks and feathers; the rest of the face is a mirror
          var R = crackR(ci, gap, n, dgap);
          var V = vec3f(0.0);
          if (iu.crackC.y > 0.0) {
            let veil = 0.22 + 0.5 * smoothstep(0.25, 0.75, fbm2(vec2f(xf.x * 1.5, xf.y * 6.0) + jag, sd + 31u));
            V = vec3f(iu.crackC.y * veil * min(prof * 1.5, 1.0));
            R *= 1.0 - V;
            out.light += out.keep * ext * V * crackAll(back);
          }
          let T = 1.0 - R - V;
//#if PIEZO
          // under pressure a crack glows at the rim where it is still growing (and a little where it has stopped)
          out.light += out.keep * ext * (iu.piezo.rgb / max(iu.piezo.r, 1e-6)) * (iu.piezo.w * exp(-8.0 * edge) * select(0.3, 1.0, g < 1.0));
//#endif
          if (R.g > T.g) {
            out.light += out.keep * ext * T * crackSeen(dir);
            out.keep *= R;
            out.t = tt;
            out.dir = back;
          } else {
            out.light += out.keep * ext * R * crackSeen(back);
            out.keep *= T;
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

// What the grains along a stretch send back towards the eye, and what they let through.
struct Play {
  light: vec3f,
  keep: vec3f,
}

// Platelets (aventurine, goldstone): each cell may hold a small flat mirror disc, at its own tilt; the
// ray is tested against the disc exactly, so a sparkle belongs to its platelet and appears only where
// that platelet mirrors a light.
fn glitter(pos: vec3f, dir: vec3f, s: f32) -> vec3f {
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
    if (unit1(h) >= iu.playA.y) { continue; }
    let r3 = ten3(ihash(h));
    let r4 = ten3(ihash(h + 91u));
    // the platelet's tilt, in the shape's own frame: its own, or mostly the smooth field's there
    let o = toObj(pos + dir * t);
    let field = normalize(vec3f(sin(1.7 * o.y + 0.3) + 0.4 * sin(2.9 * o.z), sin(1.9 * o.z + 1.1) + 0.4 * sin(2.3 * o.x), 1.2 + sin(1.3 * o.x + 2.0)));
    var gd = normalize(mix(sphereDir(r3.xy), field, iu.playB.z));
    if (dot(gd, toObj(dir)) > 0.0) { gd = -gd; }
    let back = reflect(dir, fromObj(gd));
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
  }
  return sum * iu.playB.y;
}

// Opal. The body is made of grains, each a stack of layers of silica spheres in its own direction. A
// grain sends back, as a mirror would, only the band of wavelengths that its layers' spacing selects for
// the angle the light meets them at (Bragg's law): straight on the longest, more steeply shorter ones;
// the rest it lets through. So each grain is a patch of one pure colour, which lights up when the
// mirrored direction finds a light and goes dark when it does not.
//
// The grains are the cells of an uneven, jittered lattice fixed in the shape (each place belongs to the
// nearest of eight scattered centres, the place pushed about first, so the grains are irregular and
// sharp-edged), flattened across the layers opal is laid down in. Grains come in patches: those whose
// centres fall in one larger cell lean the same way, so they flash together as one ragged block with
// flecks of neighbouring hues; some patches are plain body with no play at all. With roll the patches
// lean mostly with a smooth field across the stone, so the flash rolls over it.
//
// The ray is looked at in a few places along its first stretch, the grains met front to back: each
// sends back its band of the lights and passes the rest on, so the nearest grain's colour comes through
// pure, and the body behind a grain shows less of the colour that grain took.
fn grains(pos: vec3f, dir: vec3f, s: f32) -> Play {
  var out: Play;
  out.light = vec3f(0.0);
  out.keep = vec3f(1.0);
  let G = iu.playA.x;
  let seed = u32(iu.playA.z) * 2246822519u + 777u;
  let off = iu.playA.z * vec3f(3.17, 11.3, 5.71);
  // the lattice: in the shape's frame, squashed along its up axis so grains are thin across the layers
  let sq = vec3f(1.0, 1.0 / iu.playE.x, 1.0);
  let od = toObj(dir);
  let x0 = toLat(toObj(pos) * sq) * G + off;
  let dl = toLat(od * sq) * G;
  let span = iu.playE.z;
  var last = 0xffffffffu;
  for (var k = 0; k < PLAY_N; k++) {
    let t = (f32(k) + 0.5) * iu.playB.x;
    if (t > s) { break; }
    let x = x0 + dl * t;
    // the place pushed about, so that the grains' edges are ragged
    let xw = x + 0.32 * sin(1.7 * x.yzx + 2.1 * sin(1.3 * x.zxy)) + 0.12 * sin(4.3 * x.zxy + 1.7 * x.yzx);
    // the nearest of the eight centres about it
    let b = vec3i(floor(xw - 0.5));
    var best = 1e9;
    var bc = b;
    var bh = 0u;
    for (var i = 0u; i < 8u; i++) {
      let cc = b + vec3i(i32(i & 1u), i32((i >> 1u) & 1u), i32((i >> 2u) & 1u));
      let hh = cellHash(cc, seed);
      let dd = xw - (vec3f(cc) + 0.5 + (ten3(hh) - 0.5) * 0.75);
      let d2 = dot(dd, dd);
      if (d2 < best) {
        best = d2;
        bc = cc;
        bh = hh;
      }
    }
    if (bh == last) { continue; }
    last = bh;
    // its patch: plain body, or a block of grains leaning one way
    let pc = vec3i(floor((vec3f(bc) + 0.5) / span + 0.37 * sin(0.9 * vec3f(bc.yzx))));
    let ph = cellHash(pc, seed ^ 0x9e3779b9u);
    if (unit1(ph) >= iu.playA.y) { continue; }
    let rg = ten3(ihash(bh));
    let rp = ten3(ihash(ph));
    // the grain's direction, in the shape's own frame: the patch's, its own, or the smooth field's
    let o = toObj(pos + dir * t);
    let field = normalize(vec3f(sin(1.7 * o.y + 0.3) + 0.4 * sin(2.9 * o.z), sin(1.9 * o.z + 1.1) + 0.4 * sin(2.3 * o.x), 1.2 + sin(1.3 * o.x + 2.0)));
    let pd = normalize(mix(sphereDir(rp.xy), field, iu.playB.z));
    var gd = normalize(mix(sphereDir(rg.xy), pd, iu.playE.y * (0.75 + 0.25 * rp.z)));
    // The spheres of a grain are stacked as a cubic close packing, which has four sets of layers (one
    // across gd, three leaning 70.5 degrees from it, turned the grain's own way about it): each set
    // sends back its own band, mirrored about its own layers, so a grain has four chances to catch a
    // light. The spacing differs a little from grain to grain.
    let e1 = across1(gd);
    let e2 = across2(gd);
    // (the streaks run their own way in each patch)
    let sa = sphereDir(ten3(ihash(ph + 5u)).xy);
    let rho0 = iu.playE.w * iu.playB.y;
    let vis = exp(-iu.playD.rgb * t);
    var took = vec3f(0.0);
    for (var j = 0; j < 4; j++) {
      var pn = gd;
      if (j > 0) {
        let ph = 6.2831853 * (rp.z + 0.06 * rg.z + f32(j) / 3.0);
        pn = -0.33333333 * gd + 0.94280904 * (cos(ph) * e1 + sin(ph) * e2);
      }
      // stacking faults: inside a grain the layers waver a little in fine streaks along them, so a grain
      // lights up in streaks and points rather than all over, and its hue shifts a little across it
      let sv = dot(xw, sa) * 7.0 + 2.2 * sin(dot(xw, vec3f(-0.2, 0.9, 0.4)) * 3.1 + f32(j));
      let sw2 = sin(sv) * sin(0.37 * sv + 3.1 * xw.z + 1.3 * sin(2.3 * xw.x));
      pn = normalize(pn + 0.08 * sw2 * e1 + 0.06 * sin(1.7 * sv + 2.0 * rg.x + xw.y * 5.0) * sin(0.6 * sv + xw.x) * e2);
      if (dot(pn, od) > 0.0) { pn = -pn; }
      let gw = fromObj(pn);
      let lam = iu.playC.x * abs(dot(dir, gw)) * (0.93 + 0.07 * rg.z);
      if (lam < 425.0 || lam > 715.0) { continue; }
      let rho = clamp(specW(lam) * iu.white.rgb * (iu.white.w * iu.playB.w), vec3f(0.0), vec3f(1.0)) * rho0;
      out.light += out.keep * rho * envL(reflect(dir, gw), iu.playC.y) * vis;
      took += rho;
    }
    out.keep *= 1.0 - min(took, vec3f(1.0)) * vis;
  }
  return out;
}

fn play(pos: vec3f, dir: vec3f, s: f32) -> Play {
  // platelets or opal's grains (among several bodies each body has its own: both are compiled in)
//#if BODIES
  if (iu.playA.w > 0.5) { return Play(glitter(pos, dir, s), vec3f(1.0)); }
  return grains(pos, dir, s);
//#else
//#if GLITTER
  return Play(glitter(pos, dir, s), vec3f(1.0));
//#else
  return grains(pos, dir, s);
//#endif
//#endif
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
//#if BODIES
  // (among several bodies only those that have each part look for it)
  if (iu.crackA.y > 0.0 && iu.crackA.w > 0.0) {
//#else
  {
//#endif
    let ck = cracks(pos, dir, len, n);
    o.t = ck.t;
    o.dir = ck.dir;
    o.keep = ck.keep;
    o.light = ck.light;
    if (ck.t >= 0.0) { len = ck.t; }
  }
//#endif
//#if PLAY
//#if BODIES
  if (b < 2 && iu.playB.y > 0.0) {
//#else
  if (b < 2) {
//#endif
    let pl = play(pos, dir, len);
    o.light += o.keep * pl.light;
    o.keep *= pl.keep;
  }
//#endif
//#if SILK
//#if BODIES
  if (b < 2 && iu.silkA.w > 0.0) {
//#else
  if (b < 2) {
//#endif
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
//#if BODIES
  if (any(iu.gradB.rgb != vec3f(0.0))) { o.keep *= gradKeep(pos, dir, len, b); }
//#else
  o.keep *= gradKeep(pos, dir, len, b);
//#endif
//#endif
  return o;
}
//#endif
