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
                      // z: how far the colour runs on from domain to domain in streaks (0: each domain its own)
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
                      // one domain share a direction   z: a domain's width, in grains   w: the stacking's reflectance in its band
  playF: vec4f,       // x: how ragged the domains' outlines are   y: how strongly stacking faults streak the colour
                      // z: how much light a milky body holds inside, as a share of the light from all round
                      // w: how brightly each domain shows the broad light round about
  fracA: vec4f,       // x: a volume fracture's widest gap, nm   y: the shape's radius   z: which arrangement   w: how many sheets
  fracB: vec4f,       // x: how many of them make a fan   y: how many are great ones
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
//#else
//#if CRACK_ROUGH
fn crackSeen(d: vec3f) -> vec3f { return envL(d, iu.crackC.x); }
//#else
fn crackSeen(d: vec3f) -> vec3f { return env(d); }
//#endif
//#endif
// the light from all round a face (what a veil sends back)
fn crackAll(d: vec3f) -> vec3f {
//#if ENVL
  return envL(d, 0.9);
//#else
  return env(d);
//#endif
}

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
//#if CRACK_DEPTH
  // cracks kept near the surface are a net running in from it
  if (iu.crackB.z < 1e8) { return crackleNet(pos, dir, s, n); }
//#endif
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

// ---- crackle: a network of cracks running in from the surface ----
// Glass plunged hot into water (ice glass, crackle) breaks into a net of small polygons all over its
// surface: each crack a short wall standing upright in the glass, meeting its neighbours, reaching only a
// little way in. Here the surface is divided into cells, each place belonging to the nearest of scattered
// centres measured along the surface only (the distance across the wall is left out), so the walls
// between cells stand upright to it and join into a net. A ray near the surface is looked at in a row of
// places; where the nearest centre changes it has crossed a wall, met exactly (the plane halfway between
// the two centres). Most walls hold a crack, some do not, and each is held open in patches.
struct NetCell {
  h: u32,
  p: vec3f,
}
fn netNearest(x: vec3f, nrm: vec3f, seed: u32) -> NetCell {
  let b = vec3i(floor(x - 0.5));
  var best = 1e9;
  var r: NetCell;
  r.h = 0u;
  r.p = x;
  for (var i = 0; i < 8; i++) {
    let cc = b + vec3i(i & 1, (i >> 1) & 1, (i >> 2) & 1);
    let hh = cellHash(cc, seed);
    let sp = vec3f(cc) + 0.5 + (ten3(hh) - 0.5) * 0.8;
    var dd = x - sp;
    dd -= nrm * dot(dd, nrm);
    let d2 = dot(dd, dd);
    if (d2 < best) {
      best = d2;
      r.h = hh;
      r.p = sp;
    }
  }
  return r;
}

//#if CRACK_DEPTH
// (the net of cracks run in from the surface: only with a crack depth, the one case that reads it, and
// the one case in which the tracer of several bodies has mapOB)
// the surface's outward direction about o (the shape's own frame), from four reads of the distance
fn netNormal(o: vec3f) -> vec3f {
  let hs = 0.01;
  let k = vec2f(1.0, -1.0);
//#if BODIES
  let g = k.xyy * mapOB(o + k.xyy * hs) + k.yyx * mapOB(o + k.yyx * hs) + k.yxy * mapOB(o + k.yxy * hs) + k.xxx * mapOB(o + k.xxx * hs);
//#else
  let g = k.xyy * map(o + k.xyy * hs) + k.yyx * map(o + k.yyx * hs) + k.yxy * map(o + k.yxy * hs) + k.xxx * map(o + k.xxx * hs);
//#endif
  return g / max(length(g), 1e-9);
}
fn netDepth(o: vec3f) -> f32 {
//#if BODIES
  return -mapOB(o);
//#else
  return -map(o);
//#endif
}

const NET_N: i32 = 12;
fn crackleNet(pos: vec3f, dir: vec3f, s: f32, n: f32) -> Crack {
  var out: Crack;
  out.t = -1.0;
  out.dir = dir;
  out.keep = vec3f(1.0);
  out.light = vec3f(0.0);
  let cells = iu.crackA.x;
  let o = toObj(pos);
  let d = toObj(dir);
  let depth = iu.crackB.z;
  let seed = u32(iu.crackB.w) * 2654435761u + 4321u;
  let off = iu.crackB.w * vec3f(17.13, 7.77, 3.31);
  let pxs = iu.crackC.z / cells;
  // the two ends of the stretch lie at the surface: near each the net is looked for
  for (var end = 0; end < 2; end++) {
    let at = select(0.0, s, end == 1);
    let nrm = netNormal(o + d * at);
    let cd = max(abs(dot(d, nrm)), 0.15);
    let reach = min(depth * 1.3 / cd, s * 0.5);
    let t0 = select(0.0, s - reach, end == 1);
    let step = reach / f32(NET_N);
    let nl = toLat(nrm);
    var prev = netNearest(toLat(o + d * t0) * cells + off, nl, seed);
    for (var k = 1; k <= NET_N; k++) {
      let tk = t0 + step * f32(k);
      let cur = netNearest(toLat(o + d * tk) * cells + off, nl, seed);
      if (cur.h == prev.h) { continue; }
      let a = prev;
      prev = cur;
      // most walls hold a crack
      let ph = ihash(min(a.h, cur.h) ^ ihash(max(a.h, cur.h)));
      if (unit1(ph) >= iu.crackA.y) { continue; }
      // the wall: halfway between the two centres, upright to the surface
      var m = cur.p - a.p;
      m -= nl * dot(m, nl);
      m = normalize(m);
      let mid = 0.5 * (a.p + cur.p);
      let x0 = toLat(o) * cells + off;
      let xd = toLat(d) * cells;
      let den = dot(xd, m);
      if (abs(den) < 1e-6) { continue; }
      let th = dot(mid - x0, m) / den;
      if (th <= tk - step - 1e-5 || th > tk + 1e-5 || th <= 2e-3 / cells) { continue; }
      // how deep it reaches: the crack fades out at its depth, ragged
      let hp = o + d * th;
      let dep = netDepth(hp);
      let lp = toLat(hp) * cells + off;
      let ragged = depth * (0.55 + 0.45 * vnoise2(vec2f(dot(lp, cross(m, nl)), dot(lp, nl)) * 2.0, ph));
      let edge = 1.0 - dep / max(ragged, 1e-6);
      if (edge <= 0.0) { continue; }
      // the gap: widest at the surface, held open in patches along the wall
      let along = vec2f(dot(lp, cross(m, nl)), dot(lp, nl));
      let mot = smoothstep(0.25, 0.6, vnoise2(along * 1.7, ph + 1u));
      let prof = sqrt(min(edge * 2.0, 1.0));
      let wide = iu.crackB.x * (0.5 + unit1(ihash(ph + 2u)));
      let gap = wide * prof * mot;
      let mw = fromLat(m);
      let ci = abs(dot(d, mw));
      let dgap = wide * pxs * 4.0 / max(ci, 0.3);
      let R = crackR(ci, gap, n, dgap);
      let back = fromObj(reflect(d, mw));
      let ext = exp(-u.absorb.xyz * th);
      if (faceCross(&out, th, R, vec3f(0.0), crackSeen(back), vec3f(0.0), crackSeen(dir), back, ext)) { return out; }
    }
  }
  return out;
}
//#endif

// ---- what a crossing of a crack's face does to the path ----
// R: what the face mirrors; V: what its veil sends back as light from all round; back: the mirrored
// direction; ext: what the tint has taken on the way there. The path goes with the larger part (it
// turns at a mirror, passes a face that mostly lets light through) and the smaller part is read at once
// from the lights. True if the path turned.
fn faceCross(out: ptr<function, Crack>, tt: f32, R: vec3f, V: vec3f, mirror: vec3f, all: vec3f, ahead: vec3f, back: vec3f, ext: vec3f) -> bool {
  (*out).light += (*out).keep * ext * V * all;
  let T = max(1.0 - R - V, vec3f(0.0));
  if (R.g > T.g) {
    (*out).light += (*out).keep * ext * T * ahead;
    (*out).keep *= R;
    (*out).t = tt;
    (*out).dir = back;
    return true;
  }
  (*out).light += (*out).keep * ext * R * mirror;
  (*out).keep *= T;
  return false;
}

// the lights as a fracture's face mirrors them: a little softened (soft, radians), so a flash is not a single point
fn fracSeen(d: vec3f, soft: f32) -> vec3f {
//#if ENVL
  return envL(d, max(iu.crackC.x, soft));
//#else
  return env(d);
//#endif
}

// ---- volume fractures ----
// A few great fracture surfaces run through the body, many smaller ones, and now and then a fan of them
// spreading from one point: each a sheet, flat or gently bent, with a ragged outline. They come in three
// kinds, as in imperfect quartz and ice:
//   open      an air gap held open in patches: silver where it is wide and the angle is past the
//             critical one, the colours of a thin film where it thins, unseen where the faces touch;
//   healed    closed again, but left as a veil of countless tiny fluid inclusions laid out in
//             fingerprint-like lines: a faint white sheet that sparkles, with open patches here and there;
//   cleavage  a flat split along the crystal's planes, straight-edged and thin: rainbow colours.
// Their places, tilts, sizes, kinds and outlines all come from whole-number hashes, fixed in the shape.
struct Sheet {
  c: vec3f,           // a point on it (its middle, or for a fan the point it spreads from)
  a: vec3f,           // its normal there
  e1: vec3f,          // two directions along it
  e2: vec3f,
  k1: f32,            // how it bends along each
  k2: f32,
  r: f32,             // its reach, in the shape's units
  kind: i32,          // 0 open, 1 healed, 2 cleavage
  fan: bool,          // one of a fan: it spreads from c, inward only
  h: u32,             // its own scrambled number
}

fn sheetOf(i: i32) -> Sheet {
  var sh: Sheet;
  let rs = iu.fracA.y;
  let seed = u32(iu.fracA.z) * 1597334677u + 99u;
  let nfan = i32(iu.fracB.x);
  let nbig = i32(iu.fracB.y);
  let h = ihash(u32(i) * 2654435761u + seed);
  sh.h = h;
  let r1 = ten3(ihash(h + 1u));
  let r2 = ten3(ihash(h + 2u));
  let curve = iu.crackB.y;
  sh.fan = i < nfan;
  if (sh.fan) {
    // a fan: sheets through one line near the surface, splayed about it like the pages of a book
    let fh = ihash(seed + 4242u);
    let pc = sphereDir(ten3(fh).xy);
    sh.c = pc * 0.82 * rs;
    let ax = normalize(across1(pc) * cos(6.2831853 * ten3(fh + 1u).z) + across2(pc) * sin(6.2831853 * ten3(fh + 1u).z));
    let u1 = normalize(cross(ax, pc));
    let th = (f32(i) / max(f32(nfan - 1), 1.0) - 0.5) * 2.3 + 0.25 * (r1.x - 0.5);
    sh.a = cos(th) * u1 + sin(th) * pc;
    sh.e1 = ax;
    sh.e2 = cross(sh.a, ax);
    if (dot(sh.e2, pc) > 0.0) { sh.e2 = -sh.e2; }
    sh.k1 = 0.0;
    sh.k2 = 0.0;
    sh.r = rs * (0.35 + 0.3 * r1.y);
    sh.kind = 0;
    return sh;
  }
  let big = i < nfan + nbig;
  // the middle: anywhere in the body, the great ones more often towards the middle
  let rr = select(0.85, 0.6, big) * pow(r1.z, 0.6) * rs;
  sh.c = sphereDir(r2.xy) * rr;
  sh.a = sphereDir(r1.xy);
  // in ice (rough faces) the sheets stand upright, across the frozen surface, as lake ice cracks
  let upr = smoothstep(0.3, 0.6, iu.crackC.y);
  if (upr > 0.0) { sh.a = normalize(sh.a - vec3f(0.0, sh.a.y * 0.98 * upr, 0.0)); }
  sh.e1 = across1(sh.a);
  sh.e2 = across2(sh.a);
  sh.r = rs * select(0.16 + 0.32 * r2.z, 0.55 + 0.55 * r2.z, big);
  // (ice's sheets run far)
  sh.r *= 1.0 + 0.7 * upr;
  let r3 = ten3(ihash(h + 3u));
  let kap = (0.06 + 0.45 * curve) * (0.3 + 0.7 * r3.x) / sh.r;
  sh.k1 = kap * select(-1.0, 1.0, r3.y > 0.5 * (1.0 - curve));
  sh.k2 = kap * mix(2.0 * r3.z - 1.0, 0.6 + 0.4 * r3.z, curve);
  // the kind: in glass (curve high) no cleavage
  let kk = unit1(ihash(h + 5u));
  sh.kind = select(select(2, 1, kk < 0.72), 0, kk < 0.36);
  // (a crystal's great sheets split along its planes as often as not)
  if (big && unit1(ihash(h + 6u)) < 0.5) { sh.kind = 2; }
  // (ice's sheets are open cracks with rough white faces)
  if (upr > 0.0) { sh.kind = 0; }
  if (sh.kind == 2 && curve > 0.5) { sh.kind = 0; }
  if (sh.kind == 2) {
    sh.k1 *= 0.15;
    sh.k2 *= 0.15;
  }
  return sh;
}

// where the ray from o along d meets a sheet: up to two distances (1e9 for none)
fn sheetRoots(sh: Sheet, o: vec3f, d: vec3f) -> vec2f {
  let w0 = o - sh.c;
  let p1 = dot(w0, sh.e1);
  let p2 = dot(w0, sh.e2);
  let v1 = dot(d, sh.e1);
  let v2 = dot(d, sh.e2);
  let A = -0.5 * (sh.k1 * v1 * v1 + sh.k2 * v2 * v2);
  let B = dot(d, sh.a) - (sh.k1 * p1 * v1 + sh.k2 * p2 * v2);
  let C = dot(w0, sh.a) - 0.5 * (sh.k1 * p1 * p1 + sh.k2 * p2 * p2);
  if (abs(A) < 1e-7 * abs(B)) { return vec2f(-C / B, 1e9); }
  let disc = B * B - 4.0 * A * C;
  if (disc <= 0.0) { return vec2f(1e9); }
  let q = -0.5 * (B + select(-1.0, 1.0, B >= 0.0) * sqrt(disc));
  return vec2f(min(q / A, C / q), max(q / A, C / q));
}

// The fractures crossed by the stretch from pos along dir, s long, in a body of index n: as cracks().
const FRAC_MAX: i32 = 48;
const FRAC_HITS: i32 = 6;
fn fractures(pos: vec3f, dir: vec3f, s: f32, n: f32) -> Crack {
  var out: Crack;
  out.t = -1.0;
  out.dir = dir;
  out.keep = vec3f(1.0);
  out.light = vec3f(0.0);
  let o = toObj(pos);
  let d = toObj(dir);
  let rs = iu.fracA.y;
  let count = min(i32(iu.fracA.w), FRAC_MAX);
  // every crossing along the stretch, then taken nearest first
  var ts: array<f32, 64>;
  var ids: array<i32, 64>;
  var lens: array<f32, 64>;
  var m = 0;
  let t0 = 2e-3 * rs;
  // In ice a sheet is not a bare crack but a zone a little thick, full of fine cracks and bubbles: a
  // slab of scatterers. A ray is in it for the slab's thickness seen face on, and far longer seen edge
  // on, so the sheets show as broad white bands where they are seen along their length.
  let ice = smoothstep(0.3, 0.6, iu.crackC.y);
  let half = 0.016 * rs;
  for (var i = 0; i < count; i++) {
    let sh = sheetOf(i);
    let w0 = o - sh.c;
    let wc = w0 - d * dot(w0, d);
    if (dot(wc, wc) > sh.r * sh.r * 1.3) { continue; }
    if (ice > 0.0 && !sh.fan) {
      // where the ray is within the slab (taken flat at this scale)
      let h0 = dot(w0, sh.a);
      let v = dot(d, sh.a);
      var ta = t0;
      var tb = s;
      if (abs(v) > 1e-6) {
        let p = (-half - h0) / v;
        let q = (half - h0) / v;
        ta = max(min(p, q), t0);
        tb = min(max(p, q), s);
      } else if (abs(h0) >= half) {
        tb = ta;
      }
      if (tb > ta && m < 64) {
        ts[m] = 0.5 * (ta + tb);
        ids[m] = i;
        lens[m] = tb - ta;
        m++;
      }
      continue;
    }
    let rt = sheetRoots(sh, o, d);
    for (var e = 0; e < 2; e++) {
      let tt = rt[e];
      if (tt > t0 && tt <= s && m < 64) {
        ts[m] = tt;
        ids[m] = i;
        lens[m] = 0.0;
        m++;
      }
    }
  }
  let pxs = iu.crackC.z / iu.crackA.x;           // a pixel's width at the shape, in its units
  let gapw = iu.fracA.x;
  for (var hit = 0; hit < FRAC_HITS; hit++) {
    // the nearest crossing not yet looked at
    var best = -1;
    var bt = 1e9;
    for (var j = 0; j < m; j++) {
      if (ts[j] < bt) {
        bt = ts[j];
        best = j;
      }
    }
    if (best < 0) { break; }
    ts[best] = 1e10;
    let slab = lens[best];
    let sh = sheetOf(ids[best]);
    let tt = bt;
    let wl = o + d * tt - sh.c;
    if (dot(wl, wl) >= sh.r * sh.r * 1.3) { continue; }
    let sd = ihash(sh.h + 17u);
    let rq = ten3(ihash(sh.h + 9u));
    // growth: each sheet starts at its own moment and spreads from its middle
    let g = clamp((iu.crackA.w - 0.5 * rq.x) / 0.5, 0.0, 1.0);
    if (g <= 0.0) { continue; }
    let r = sh.r * g;
    // where on the sheet: as a share of its grown reach (x2), and of its full reach (xf)
    let x2 = vec2f(dot(wl, sh.e1), dot(wl, sh.e2)) / r;
    let xf = x2 * g;
    // the outline
    var edge: f32;
    if (sh.fan) {
      // a fan's sheet: from its line inward, widening, in a ragged half round
      edge = min(1.0 - length(vec2f(x2.x * 1.4, x2.y)), x2.y + 0.05);
    } else {
      // (long and narrow in ice)
      let sq = mix(0.45 + 0.5 * rq.y, 0.18 + 0.2 * rq.y, smoothstep(0.3, 0.6, iu.crackC.y));
      let ca = cos(6.2831853 * rq.z);
      let sa = sin(6.2831853 * rq.z);
      edge = 1.0 - length(vec2f(ca * x2.x + sa * x2.y, (ca * x2.y - sa * x2.x) / sq));
      if (sh.kind == 2) {
        // a cleavage ends on straight lines
        let cut = ten3(ihash(sh.h + 11u));
        for (var k = 0; k < 3; k++) {
          let th = 2.0943951 * (f32(k) + 0.7 * cut[k]);
          edge = min(edge, 0.3 + 0.5 * cut[(k + 1) % 3] - dot(x2, vec2f(cos(th), sin(th))));
        }
      }
    }
    let jag = fbm2(xf * 2.2, sd);
    let crease = 1.0 - abs(2.0 * fbm2(xf * 5.0 + 1.3, sd + 7u) - 1.0);
    let lace = vnoise2(xf * 16.0, sd + 9u);
    edge -= 0.42 * (jag - 0.42) + 0.2 * (crease - 0.6) + 0.08 * (lace - 0.5);
    if (edge <= 0.0) { continue; }
    let ee = min(edge * 3.0, 1.0);
    let prof = sqrt(ee * (2.0 - ee));
    // the faces' gap: held open in patches that swell and thin, thin all over in a cleavage, mostly
    // closed in a healed one
    let sw = fbm2(xf * 1.6 + 3.3, sd + 11u);
    var open: f32;
    var wide = gapw * (0.35 + 1.15 * rq.x);
    if (sh.kind == 2) {
      open = clamp(0.55 + 1.3 * (sw - 0.5), 0.0, 1.0);
      wide = min(wide, 1500.0);
    } else {
      open = mix(smoothstep(0.3, 0.62, sw), clamp(0.5 + 1.4 * (sw - 0.5), 0.0, 1.0), 1.0 - smoothstep(1200.0, 3000.0, wide));
      if (sh.kind == 1) { open *= 0.35 * smoothstep(0.55, 0.75, fbm2(xf * 3.1, sd + 13u)); }
    }
    // feathers at the rim: the sheet frays into streaks running out from its middle
    let fth = vnoise2(vec2f(atan2(x2.y, x2.x) * 60.0 + 8.0 * jag, length(xf) * 3.0), sd + 17u);
    open *= mix(smoothstep(0.35, 0.65, fth), 1.0, smoothstep(0.04, 0.2, edge));
    let gap = wide * prof * open;
    // the faces' relief: a gentle waviness, and steps each tilted a little its own way
    let wv = vec2f(vnoise2(xf * 3.5, sd + 21u), vnoise2(xf * 3.5, sd + 22u)) - 0.5;
    let tq = xf * 4.0 + 2.0 * vec2f(jag - 0.5, crease - 0.6);
    let tr = ten3(ihash(u32(i32(floor(tq.x))) + ihash(u32(i32(floor(tq.y))) + sd))).xy - 0.5;
    let tilt = 0.08 * wv + select(0.12, 0.07, sh.kind == 2) * tr;
    let pw = wl - sh.a * dot(wl, sh.a);
    let nl = normalize(sh.a - sh.k1 * dot(pw, sh.e1) * sh.e1 - sh.k2 * dot(pw, sh.e2) * sh.e2 + tilt.x * sh.e1 + tilt.y * sh.e2);
    let ci = abs(dot(d, nl));
    // how much the gap changes across a pixel
    let px = pxs / (r * max(ci, 0.3));
    let hs = 0.03;
    let sx = fbm2((xf + vec2f(hs, 0.0)) * 1.6 + 3.3, sd + 11u);
    let sy = fbm2((xf + vec2f(0.0, hs)) * 1.6 + 3.3, sd + 11u);
    let dopen = length(vec2f(sx - sw, sy - sw)) / hs * select(2.5, 1.3, sh.kind == 2) * g;
    let dgap = wide * ((1.0 - ee) / max(prof, 0.03) * 3.0 * open + prof * dopen) * px;
    var R = crackR(ci, gap, n, dgap);
    // A gap thinner than a micron or so shows its colours in every light of a bright room: the faces
    // are never quite flat, so they gather light from a spread of directions, and next to the plain
    // glass around them the colours read strongly. The softened lights stand for the first, and the
    // colour is drawn out and brightened for the second (looks first: a dark studio has too little
    // light round about for the exact figure to show the bands a photograph does).
    // (not past the critical angle: there the faces mirror nearly everything, whatever the gap)
    let crit = sqrt(max(1.0 - 1.0 / (n * n), 0.0));
    let thin = (1.0 - smoothstep(1200.0, 3000.0, wide)) * open * smoothstep(crit, crit + 0.06, ci);
    if (thin > 0.0) {
      let lum = dot(R, vec3f(0.2126, 0.7152, 0.0722));
      let vivid = max(vec3f(lum) + (R - vec3f(lum)) * 4.0, vec3f(0.0)) * 3.5;
      R = clamp(mix(R, vivid, thin), vec3f(0.0), vec3f(0.95));
    }
    // the veil: fluid inclusions in fingerprint lines (healed sheets, and the rough faces of open
    // ones); fine dots that a pixel cannot tell apart are taken at their average
    var V = vec3f(0.0);
    // a rough face: a white veil in soft streaks and feathers, as in ice
    if (iu.crackC.y > 0.0) {
      let streaky = smoothstep(0.3, 0.7, fbm2(vec2f(xf.x * 1.2, xf.y * 5.0) + 1.5 * jag, sd + 29u));
      // (a thin layer of scatterers: faint seen face on, dense seen edge on)
      let k = iu.crackC.y * prof * (0.35 + 0.65 * streaky);
      if (slab > 0.0) {
        // (how many times the slab's own thickness the ray spends in it: 1 face on. The haze grows with
        // its cube, so a sheet seen face on or aslant is all but clear, as lake ice is from the side, and
        // only where it is seen nearly along its length does it draw a white line; from above, looking
        // down the upright sheets, they are all seen along their length and stand white)
        let x = slab / (2.0 * half);
        V = vec3f(0.92 * (1.0 - exp(-0.004 * k * x * x * x)));
      } else {
        V = vec3f(0.9 * (1.0 - exp(-0.05 * k / max(ci, 0.02))));
      }
    }
    if (sh.kind == 1) {
      let xs = xf * (sh.r / (0.1 * rs));
      let org = vec2f(cos(6.2831853 * rq.y), sin(6.2831853 * rq.y)) * 1.5;
      let ridge = 0.5 + 0.5 * sin(6.2831853 * (10.0 * length(xs - org * (sh.r / (0.1 * rs))) * 0.35 + 4.0 * fbm2(xs * 0.3, sd + 23u)));
      let pu = pxs / (0.1 * rs * max(ci, 0.3));
      let lines = mix(smoothstep(0.5, 0.9, ridge), 0.35, clamp(pu * 10.0 - 0.5, 0.0, 1.0));
      let dots = mix(smoothstep(0.45, 0.75, vnoise2(xs * 40.0, sd + 25u)), 0.35, clamp(pu * 40.0 - 0.5, 0.0, 1.0));
      let patchy = smoothstep(0.25, 0.55, fbm2(xf * 1.3 + 7.1, sd + 27u));
      let broken = smoothstep(0.35, 0.65, vnoise2(xs * 1.7, sd + 31u));
      V = max(V, vec3f(0.5 * prof * patchy * (0.2 + 0.8 * mix(0.35, lines, 0.5 * broken) * dots)));
    }
    R *= 1.0 - V;
    // (ice's rough faces scatter what a clean break would mirror)
    R *= 1.0 - 0.85 * ice;
    // a thin crack past the critical angle is a mirror only where its faces stand apart; it touches in
    // patches and seams, so its flash breaks up instead of lying as one flat sheet of the mirrored light
    let thinGap = 1.0 - smoothstep(1200.0, 3000.0, wide);
    if (thinGap > 0.0) {
      let pat = smoothstep(0.38, 0.62, fbm2(xf * 4.5 + 2.7, sd + 33u)) * (0.6 + 0.4 * smoothstep(0.3, 0.7, vnoise2(xf * 13.0, sd + 35u)));
      R *= mix(1.0, 0.15 + 0.85 * pat, thinGap * (1.0 - smoothstep(0.0, 0.08, ci - sqrt(max(1.0 - 1.0 / (n * n), 0.0)))));
    }
    let back = fromObj(reflect(d, nl));
    let ext = exp(-u.absorb.xyz * tt);
    // (a veil of countless scatterers is lit from the whole room; looks first: it reads white even where
    // the lights round about are dim, as the white sheets in lake ice do)
    let veilLit = max(crackAll(back), vec3f(dot(crackAll(-back), vec3f(0.2126, 0.7152, 0.0722)))) * (1.0 + 2.0 * iu.crackC.y);
    if (faceCross(&out, tt, R, V, fracSeen(back, 0.04 + 0.1 * thin), veilLit, crackSeen(dir), back, ext)) { break; }
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

// Smooth noise through space, 0..1, fixed by seed.
fn vnoise3(p: vec3f, seed: u32) -> f32 {
  let i = floor(p);
  let f = p - i;
  let w = f * f * (3.0 - 2.0 * f);
  let c = vec3i(i);
  var v: array<f32, 8>;
  for (var k = 0; k < 8; k++) {
    let o = vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    v[k] = unit1(cellHash(c + o, seed));
  }
  return mix(mix(mix(v[0], v[1], w.x), mix(v[2], v[3], w.x), w.y), mix(mix(v[4], v[5], w.x), mix(v[6], v[7], w.x), w.y), w.z);
}

// The nearest and second nearest of the scattered centres about a place, one per cell of a lattice
// (each centre within its cell, so the eight cells about the place are enough): the cell's scrambled
// number, and the squared distances to the nearest and the next.
struct Near {
  h: u32,
  d1: f32,
  d2: f32,
}
fn nearest(x: vec3f, seed: u32) -> Near {
  let b = vec3i(floor(x - 0.5));
  var r: Near;
  r.d1 = 1e9;
  r.d2 = 1e9;
  r.h = 0u;
  for (var i = 0; i < 8; i++) {
    let cc = b + vec3i(i & 1, (i >> 1) & 1, (i >> 2) & 1);
    let hh = cellHash(cc, seed);
    let dd = x - (vec3f(cc) + 0.5 + (ten3(hh) - 0.5) * 0.8);
    let d2 = dot(dd, dd);
    if (d2 < r.d1) {
      r.d2 = r.d1;
      r.d1 = d2;
      r.h = hh;
    } else if (d2 < r.d2) {
      r.d2 = d2;
    }
  }
  return r;
}

// Opal. The body is made of silica spheres stacked in layers; a stack sends back, as a mirror would,
// only the band of wavelengths its layers' spacing selects for the angle the light meets them at
// (Bragg's law: straight on the longest, more steeply shorter ones) and lets the rest through. So colour
// lies where a stack's mirrored direction finds a light, and is pure.
//
// What the photographs of opal show, and how it is made here, from large to small:
//  - domains: regions of one stacking, several grains wide, with sharp outlines (straight-edged blocks in
//    harlequin, ragged ones elsewhere) and often a thin seam of plain body between them. Each place
//    belongs to the nearest of a few scattered centres (on a coarse lattice fixed in the shape); some
//    domains have no play at all (potch).
//  - inside a domain the layers bend slowly, so its colour drifts across it (red running into orange and
//    green) and the flash moves a little over it as the stone turns;
//  - grains: the domain is broken into smaller cells, each tilted a little its own way, so it lights up
//    piece by piece rather than all at once;
//  - streaks: stacking faults run through the layers in fine parallel streaks (straw, flame), which break
//    the colour into flecks and lines with plain body between, and tilt the layers a little each.
// All of it is flattened across the layers opal is laid down in (the shape's up axis). The ray is
// looked at in a few places along its first stretch, front to back: each place sends back its share of
// the lights in its band and passes the rest on, so deeper colour shows dimmer through the body above it,
// and the body behind colour shows less of the colour taken.
// The light a grain of opal sends back, from what arrives along its mirror direction. Under the lights
// the colour is the band's times the light's own. Under an environment picture only the light's
// brightness counts: a garden is mostly green, but a photograph of an opal outdoors shows
// its reds as vivid as indoors (the eye and the camera take the scene's colour cast for white), so the
// green round about must not leave only the green domains alight. Looks first.
fn opalLit(c: vec3f) -> vec3f {
//#if PICTURE
  return vec3f(dot(c, vec3f(0.2126, 0.7152, 0.0722)));
//#else
  return c;
//#endif
}

// The colour of a field where the colour runs on, for a place 0 to 1 along the run of colours a black
// opal's fields grade through: blue into green, green into yellow, yellow into orange and red, then back
// to blue. The warm colours take as much of the run as the cool ones, as they do in the stones (evenly
// in wavelength, yellow and orange would be thin lines between green and red).
fn opalRun(h: f32) -> f32 {
  var at = array<f32, 8>(0.0, 0.1, 0.24, 0.4, 0.55, 0.7, 0.88, 1.0);
  var nm = array<f32, 8>(478.0, 502.0, 530.0, 556.0, 574.0, 590.0, 615.0, 478.0);
  for (var i = 1; i < 8; i++) {
    if (h <= at[i]) {
      let f = (h - at[i - 1]) / (at[i] - at[i - 1]);
      return mix(nm[i - 1], nm[i], select(f, step(0.5, f), i == 7));
    }
  }
  return 465.0;
}

fn grains(pos: vec3f, dir: vec3f, s: f32) -> Play {
  var out: Play;
  out.light = vec3f(0.0);
  out.keep = vec3f(1.0);
  let G = iu.playA.x;
  let seed = u32(iu.playA.z) * 2246822519u + 777u;
  let off = iu.playA.z * vec3f(3.17, 11.3, 5.71);
  let sq = vec3f(1.0, 1.0 / iu.playE.x, 1.0);
  let od = toObj(dir);
  let x0 = toLat(toObj(pos) * sq) * G + off;
  let dl = toLat(od * sq) * G;
  let span = iu.playE.z;
  let warp = iu.playF.x;
  let rho0 = iu.playE.w * iu.playB.y;
  // the light inside the body: what it gathers from all round
  var inside = vec3f(0.0);
  if (iu.playF.z > 0.0) { inside = envL(-dir, 1.6); }
  // How bright the light all round is, at the least, for the domains' own glow. Under the lights the
  // way each domain faces is enough: the room is lit from many sides. Under an environment picture
  // most ways see the ground and the walls round about, while the light comes from the sky; an opal
  // photographed outdoors still glows from every domain, so a domain facing the lawn is lit as if by the
  // surroundings as a whole (six ways, at the softest level). Looks first.
  var around = 0.0;
  var picBoost = 1.0;
//#if PICTURE
  {
    picBoost = 2.5;
    let wl = vec3f(0.2126, 0.7152, 0.0722);
    around = (dot(envL(vec3f(1.0, 0.0, 0.0), 1.6), wl) + dot(envL(vec3f(-1.0, 0.0, 0.0), 1.6), wl)
            + dot(envL(vec3f(0.0, 1.0, 0.0), 1.6), wl) + dot(envL(vec3f(0.0, -1.0, 0.0), 1.6), wl)
            + dot(envL(vec3f(0.0, 0.0, 1.0), 1.6), wl) + dot(envL(vec3f(0.0, 0.0, -1.0), 1.6), wl)) / 6.0;
  }
//#endif
  // Where the colour runs on from domain to domain (flame, broad flash: the Lightning Ridge black opals),
  // the stone is laid down in sheets that all streak one way: a direction across the up axis, fixed in the
  // shape, along which the colour drifts slowly and across which it changes faster.
  let flow = iu.playC.z;
  let fa = 6.2831853 * unit1(ihash(seed + 101u));
  let sAlong = vec3f(cos(fa), 0.0, sin(fa));
  let sAcross = vec3f(-sin(fa), 0.0, cos(fa));
  for (var k = 0; k < PLAY_N; k++) {
    let t = (f32(k) + 0.5) * iu.playB.x;
    if (t > s) { break; }
    let x = x0 + dl * t;
    // ragged outlines: the place pushed about before the centres are looked for (little for harlequin)
    // (and, at every pattern, wandering a little more about a grain's width, so that no outline is a
    // straight cut: in photographs patches run into each other in ragged fringes)
    let rag = vec3f(vnoise3(x * 0.8, seed + 11u), vnoise3(x * 0.8 + 17.3, seed + 12u), vnoise3(x * 0.8 + 31.7, seed + 13u)) - 0.5;
    let xw = x + warp * (sin(0.53 * x.yzx + 1.7 * sin(0.41 * x.zxy)) + 0.4 * sin(1.9 * x.zxy + 1.3 * x.yzx));
    let xd = xw + 0.8 * rag;
    // patches of every size: many of the large domains are broken into smaller ones, some smaller still
    var cw = span;
    var dm = nearest(xd / cw, seed ^ 0x51ed27u);
    let split = unit1(ihash(dm.h + 31u));
    if (split < 0.5) {
      cw = span * 0.5;
      dm = nearest(xd / cw, seed ^ 0x2c1b3c6du);
      if (split < 0.15) {
        cw = span * 0.3;
        dm = nearest(xd / cw, seed ^ 0x6a09e667u);
      }
    }
    if (unit1(dm.h) >= iu.playA.y) { continue; }
    // where two domains meet the colour thins out over most of a grain into the dark body between, rather
    // than stopping at a line
    var seam = 0.1 + 0.9 * smoothstep(0.05, 0.7, (sqrt(dm.d2) - sqrt(dm.d1)) * cw);
    // (where the colour runs on, neighbouring domains meet without a dark seam between)
    seam = mix(seam, 0.45 + 0.55 * seam, flow);
    // and within a domain the colour is not even: bright cores, places nearly dark, so the body shows
    // through between and across patches
    var lit = 0.1 + 0.9 * smoothstep(0.15, 0.6, vnoise3(xw * 0.55 + 7.0 * ten3(ihash(dm.h + 3u)), seed + 9u));
    // (and little of the body shows between: most of the stone is colour, bright and less bright)
    // (but in broad patches, with the dark body between: a photographed stone is colour broken by dark, not
    // filled edge to edge)
    let opPatch = smoothstep(0.42, 0.72, vnoise3(xw * 0.2 + 3.1, seed + 23u));
    lit = mix(lit, (0.06 + 0.94 * opPatch) * (0.3 + 0.7 * lit), flow);
    let rp = ten3(ihash(dm.h));
    let rq = ten3(ihash(dm.h + 7u));
    let o = toObj(pos + dir * t);
    let field = normalize(vec3f(sin(1.7 * o.y + 0.3) + 0.4 * sin(2.9 * o.z), sin(1.9 * o.z + 1.1) + 0.4 * sin(2.3 * o.x), 1.2 + sin(1.3 * o.x + 2.0)));
    let pd = normalize(mix(sphereDir(rp.xy), field, iu.playB.z));
    let e1 = across1(pd);
    let e2 = across2(pd);
    // the spacing of the domain's spheres: one domain runs to reds, its neighbour only to greens and
    // blues (lit places of one domain share a hue, since for a given light and eye every lit place
    // meets them at the same angle)
    let spacing = 0.76 + 0.38 * rq.x;
    // the layers bending slowly through the domain
    let ph = 6.2831853 * rq;
    let bend = vec2f(sin(0.45 * xw.x + 0.3 * xw.z + ph.x) + 0.5 * sin(0.8 * xw.y + ph.y), sin(0.4 * xw.z - 0.35 * xw.y + ph.z) + 0.5 * sin(0.7 * xw.x + ph.x));
    // the streaks run one way in the layers (the domain's own)
    var sa = cos(ph.z) * e1 + sin(ph.z) * e2;
    // (where the colour runs on, every domain streaks the stone's own way, as near as its layers allow)
    let sg = sAlong - dot(sAlong, pd) * pd;
    if (flow > 0.0 && dot(sg, sg) > 1e-4) { sa = normalize(mix(sa * sign(dot(sa, sg) + 1e-6), normalize(sg), flow)); }
    let sb = cross(pd, sa);
    let sn = vnoise3(vec3f(dot(xw, sa) * mix(0.35, 1.6, flow), dot(xw, sb) * 4.5, xw.y * 1.5), seed + 3u);
    let fl = vnoise3(vec3f(dot(xw, sa) * mix(1.3, 4.5, flow), dot(xw, sb) * 9.0, xw.y * 3.0), seed + 4u);
    // flakes: small pieces drawn out along the streaks, of a size that differs from domain to domain,
    // with ragged edges, each tilted its own way within the domain's spread, so that a domain glitters
    // piece by piece; a faint seam of body between them
    // (where the colour runs on, finer: a field is made of many small flecks, not a few broad strokes)
    let fz = mix(0.6, 1.6, rq.y) * (1.0 + 0.7 * flow);
    let fq = vec3f(dot(xw, sa) * mix(1.3, 3.0, flow), dot(xw, sb) * 3.8, xw.y * 3.2) * fz + 0.9 * vec3f(sn - 0.5, fl - 0.5, sn - fl);
    let gr = nearest(fq, seed);
    let gh = ten3(ihash(gr.h));
    let spread = 0.22 + 0.7 * (1.0 - iu.playE.y);
    // (and inside each flake the layers waver, so it lights in part, unevenly)
    let fine = vnoise3(fq * 2.7, seed + 5u);
    let gt = (gh.xy - 0.5) * 2.0 * spread + 0.3 * (fine - 0.5);
    let flake = select(0.45, 1.0, gh.z < 0.82) * (0.55 + 0.45 * smoothstep(0.0, 0.06, sqrt(gr.d2) - sqrt(gr.d1))) * (0.5 + 0.5 * fine);
    // (where the colour runs on, the streaks shade it rather than cut it into lines)
    let cut = smoothstep(0.4, 0.6, sn) * (0.55 + 0.45 * smoothstep(0.3, 0.7, fl));
    let streak = mix(1.0, mix(cut, 0.6 + 0.4 * cut, flow), iu.playF.y) * flake;
    let tilt = 0.2 * bend + gt + iu.playF.y * 0.3 * vec2f(sn - 0.5, fl - 0.5);
    let gd = normalize(pd + tilt.x * e1 + tilt.y * e2);
    let vis = exp(-iu.playD.rgb * t);
    // (each place stands for a layer of the stone half a grain thick)
    let share = rho0 * streak * seam * lit;
    // the four sets of a close packing's layers: across gd, and three leaning 70.5 degrees from it.
    // At one place the set that sends back the most wins (the sets lie in different parts of a flake,
    // finer than a pixel is not: one colour shows at a point, not a blend of four)
    var best = vec3f(0.0);
    var bestRho = vec3f(0.0);
    var bestY = -1.0;
    for (var j = 0; j < 4; j++) {
      var pn = gd;
      if (j > 0) {
        let a2 = 6.2831853 * (rp.z + f32(j) / 3.0);
        pn = -0.33333333 * gd + 0.94280904 * (cos(a2) * across1(gd) + sin(a2) * across2(gd));
      }
      if (dot(pn, od) > 0.0) { pn = -pn; }
      let gw = fromObj(pn);
      let lam = iu.playC.x * abs(dot(dir, gw)) * spacing;
      if (lam < 425.0 || lam > 715.0) { continue; }
      let rho = clamp(specW(lam) * iu.white.rgb * (iu.white.w * iu.playB.w), vec3f(0.0), vec3f(1.0)) * share;
      let got = rho * opalLit(envL(reflect(dir, gw), iu.playC.y));
      let y = dot(got, vec3f(0.2126, 0.7152, 0.0722));
      if (y > bestY) {
        bestY = y;
        best = got;
        bestRho = rho;
      }
    }
    // Looks first. In a photograph the play of colour is the brightest thing on an opal under any
    // ordinary light, not only where a lamp is mirrored: a stone's domains are lit from the whole room,
    // and light scattered inside a milky body lights them from within. So besides the mirrored lamps
    // above, each domain glows with its own colour lit by the broad light round about (and, in a milky
    // body, the light inside it), over its own window of angles: as the stone turns its hue runs a
    // little within the window and goes out abruptly beyond it, and another domain comes on.
    let cd = abs(dot(od, pd));
    // (where the colour runs on, the windows are wider: at any turn most of the stone shows colour, and only
    // a few domains are out)
    let lo = mix(0.02 + 0.45 * rq.y, 0.01 + 0.12 * rq.y, flow);
    let hi = lo + mix(0.6 + 0.3 * rp.y, 0.85 + 0.3 * rp.y, flow);
    let gate = smoothstep(lo, lo + 0.025, cd) * (1.0 - smoothstep(hi, hi + 0.025, cd));
    var glow = vec3f(0.0);
    var rhoD = vec3f(0.0);
    if (gate > 0.0 && iu.playF.w > 0.0) {
      // the domain's own colour, all hues alike likely, running a little across its window
      // Where the colour runs on, the hue is not the domain's own but the stone's, laid out in long fields
      // along its streaks that grade from one into the next (red into orange and yellow, green into blue),
      // so that neighbouring domains share it and a field flows on across their outlines; each domain
      // moves it a little, and its flakes a little more.
      let q = vec3f(dot(xw, sAlong) * 0.07, xw.y * 0.3, dot(xw, sAcross) * 0.22);
      let hf = vnoise3(q, seed + 21u) + 0.45 * vnoise3(q * 2.3 + 5.1, seed + 22u);
      let hueRun = opalRun(fract(1.1 * hf + 0.1 * (rp.x - 0.5) + 0.31 * f32(seed & 7u)));
      let lamOwn = 430.0 + 270.0 * fract(rp.x + 0.37 * rq.z);
      let lamD = mix(lamOwn, hueRun, flow) + mix(45.0, 25.0, flow) * ((cd - lo) / (hi - lo) - 0.5) + mix(12.0, 40.0, flow) * (gh.z - 0.5)
        + mix(70.0, 30.0, flow) * (vnoise3(xw * 0.4, seed + 10u) - 0.5);
      rhoD = clamp(specW(lamD) * iu.white.rgb * (iu.white.w * iu.playB.w), vec3f(0.0), vec3f(1.0)) * share * gate;
      // flake by flake it glows more or less
      // (a few flakes much brighter than the rest: the patch glitters rather than glowing as one fill)
      let each = (0.3 + 1.0 * gh.y * gh.y) * (0.45 + 0.55 * fine);
      // (lit by how bright the light round about is, not by its colour: a green garden does not leave only
      // the green domains alight)
      let wl = vec3f(0.2126, 0.7152, 0.0722);
      // (where the colour runs on, the fields are as bright as the flashes of the stones photographed:
      // broad, and lit from much of the room at once)
      glow = (1.0 + 0.7 * flow) * rhoD * each * (iu.playF.w * max(dot(envL(reflect(dir, fromObj(pd)), 0.8), wl), around) * picBoost + iu.playF.z * dot(inside, wl));
      // (where the colour runs on, seen through the stone's own faint haze: less pure than a painted hue)
      glow = mix(glow, vec3f(dot(glow, wl)), 0.12 * flow);
    }
    out.light += out.keep * (best + glow) * vis;
    out.keep *= 1.0 - max(bestRho, rhoD) * vis;
    // (and where a domain glows, less of the milky body behind it shows through: its colour stands out
    // as it does in a photograph, not as a wash over white)
    out.keep *= 1.0 - 0.6 * gate * share * vis;
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
  if ((iu.crackA.y > 0.0 || iu.fracA.w > 0.0) && iu.crackA.w > 0.0) {
//#else
  {
//#endif
    // the volume fractures, then the small cracks as far as the path still runs
    if (iu.fracA.w > 0.0) {
      let fk = fractures(pos, dir, len, n);
      o.t = fk.t;
      o.dir = fk.dir;
      o.keep = fk.keep;
      o.light = fk.light;
      if (fk.t >= 0.0) { len = fk.t; }
    }
    if (iu.crackA.y > 0.0) {
      let ck = cracks(pos, dir, len, n);
      o.light += o.keep * ck.light;
      o.keep *= ck.keep;
      if (ck.t >= 0.0) {
        o.t = ck.t;
        o.dir = ck.dir;
        len = ck.t;
      }
    }
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
