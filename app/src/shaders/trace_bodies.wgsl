
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
// around air. What a single material has and a body among several has not: colour by wavelength (here as
// three colours), true scattering (the light volume is used), double refraction, the graded index, bubbles.

struct MB {
  kind: vec4f,              // for each body: 0 clear, 1 a metal or a mirror, 2 milky, 3 a thin wall around air
  ior: vec4f,               // its index at 550 nm
  disp: vec4f,              // its dispersion (index at 400 nm less index at 700 nm)
  spare: vec4f,
  absorb: array<vec4f, 4>,  // rgb: what its inside absorbs per unit length
  b: array<MU, 4>,
}
@group(0) @binding(12) var bodTex: texture_3d<f32>;
@group(0) @binding(13) var<uniform> mb: MB;

// The body whose material the material's own functions read (every read of the material's block in them reads mb.b[cb]): set
// before each of them is called.
var<private> cb: i32;

const COUNT: i32 = {{COUNT}};           // how many bodies there are: only that many channels mean anything
const EVENTS: i32 = 2 * LAYERS + 2;     // surfaces a ray that has left the first body may still cross

//#if HOLLOW
// (whether the body in hand is a thin wall around air: its film then thins to black film)
fn isHollow() -> bool { return mb.kind[cb] > 2.5; }
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
fn seenB(rd: vec3f, sig: f32) -> vec3f {
  if (sig <= 0.0) { return env(rd); }
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
// the index of what a place is made of, at a wavelength (air, and the air inside a thin wall: 1)
fn iorB(k: i32, lam: f32) -> f32 {
  if (k < 0) { return 1.0; }
  if (mb.kind[k] > 2.5) { return 1.0; }
  return mb.ior[k] + mb.disp[k] * (1.0e6 / (lam * lam) - 3.3058) / 4.2092;
}

// What the surface between a (what the light is in) and b (what lies beyond; -1 is the air) sends back,
// as a colour, of light arriving at cosine c at the point q, where the surface faces n. lam: the
// wavelength whose bending is being followed. The surface is the higher-numbered body's, and so are the
// film and the coat on it.
fn surfB(a: i32, b: i32, c: f32, q: vec3f, n: vec3f, lam: f32) -> vec3f {
  let o = max(a, b);
  cb = o;
  var d = 0.0;
//#if FILM
  if (mb.b[o].film.x > 0.0) { d = filmNm(q, n); }
//#endif
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
    under = metalRGB(cc, na, d);
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
    } else if (mb.kind[o] > 2.5) {
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
  var t = 0.0;
  var j = -1;
  for (var i = 0; i < STEPS_IN; i++) {
    let v = dist4(pos + dir * t);
    var d = -v[m];
    j = -1;
    for (var k = m + 1; k < COUNT; k++) { if (v[k] < d) { d = v[k]; j = k; } }
    if (d < 0.0008) { break; }
    t += d * STEP_SCALE;
    if (t > 2.5 * BOUND) { break; }
  }
  var p = pos + dir * t;
  if (j >= 0) {
    *b = j;
    *nq = nrmB(p, j);
    *q = p;
    return t;
  }
  // m ends here: beyond it is a body numbered below m that this place is also inside, or the air
  let v = dist4(p + dir * 0.004);
  var lower = -1;
  for (var k = 0; k < m; k++) { if (v[k] < 0.0) { lower = k; } }
  if (lower < 0) {
    // the outside of the whole scene: settle on the scene's own smooth surface, as plain glass does
    t -= mapNW(pos + dir * t);
    t -= mapNW(pos + dir * t);
    p = pos + dir * t;
    *nq = nrm(p);
  } else {
    *nq = nrmB(p, m);
  }
  *b = lower;
  *q = p;
  return t;
}

// How much wider (as a variance, radians squared) a rough surface of spread alpha makes light that
// passes through it from index na at cosine c into index nb.
fn widenB(alpha: f32, na: f32, c: f32, nb: f32, od: vec3f, nq: vec3f) -> f32 {
  let co = abs(dot(od, nq));
  let w = 0.70711 * alpha * clamp(abs(na * c - nb * co) / max(nb * co, 0.05), 0.0, 6.0);
  return w * w;
}

// A stretch of length s through body m, from pos along dir: what it adds of its own light (its glow; the
// light inside a milky body, from that body's light volume, as glowAlong gathers it for one material),
// and what it leaves of the light passing through (thr, by colour). sg2: the spread of what is seen
// beyond it, widened by a milky body's haze.
fn throughB(m: i32, pos: vec3f, dir: vec3f, s: f32, thr: ptr<function, vec3f>, sg2: ptr<function, f32>) -> vec3f {
  *thr *= exp(-mb.absorb[m].rgb * s);
  var add = *thr * mb.b[m].glow.rgb * s;
//#if SSS
  if (mb.kind[m] > 1.5 && mb.kind[m] < 2.5) {
    cb = m;
    add += *thr * glowAlong(pos, dir, s);
    *thr *= exp(-(mb.b[m].sss.x + mb.b[m].sssA.rgb) * max(s, 0.0));
    *sg2 += HAZE * mb.b[m].sss.x * max(s, 0.0);
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
        break;
      }
      let tFar = -bb + sqrt(h);
      var t = 0.0;
      var hit = false;
      for (var i = 0; i < STEPS_OUT; i++) {
        let d = mapW(ro + rd * t);
        if (d < 0.0008) { hit = true; break; }
        t += d * STEP_SCALE;
        if (t > tFar) { break; }
      }
      if (!hit) {
//#if ROUGH
        frMiss(l);
//#endif
        break;
      }
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
      // through body m to where it ends or another takes over
      let s = eventB(ro, rd, m, &q, &nq, &b);
      col += throughB(m, ro, rd, s, &thr, &sg2);
    }
    var c = -dot(rd, nq);
    if (c < 0.0) { nq = -nq; c = -c; }      // nq now faces the ray
    let o = max(m, b);
    let R = surfB(m, b, c, q, nq, lam);
    let alpha = mb.b[o].surf.x;
    let mir = reflect(rd, nq);
    if (opaqueB(b)) {
      // a metal turns it, and it goes on through what it was in
      col += thr * (mb.b[b].glow.rgb * 0.3 + R * skyFarB(o, mir));
      thr *= R;
      sg2 += 2.0 * alpha * alpha;
      if (max3(thr) < 0.02) { return col; }
      rd = mir;
      ro = q + nq * 0.004;
      continue;
    }
    let na = iorB(m, lam);
    let nb = iorB(b, lam);
    let od = refract(rd, nq, na / nb);
    if (dot(od, od) < 0.5) {
      // trapped by total reflection: take the bounce and stop there
      return col + thr * seenB(mir, sqrt(sg2));
    }
    // (only what is sent back into the air is added, as for plain glass: reflections inside are let go)
    if (m < 0) { col += thr * R * (seenB(mir, sqrt(sg2 + 2.0 * alpha * alpha)) + skyFarB(o, mir)); }
    thr *= 1.0 - R;
    if (mb.kind[o] > 2.5) { thr *= mb.b[o].axis.xyz; }      // through a thin wall: its tint, once
    sg2 += widenB(alpha, na, c, nb, od, nq);
    rd = od;
    ro = q - nq * 0.006;
    m = b;
  }
  return col + thr * seenB(rd, sqrt(sg2));
}

// What a point of the surface sends towards the eye: shade(), for bodies of several materials.
fn shadeB(ro: vec3f, rd: vec3f, t: f32, xi: f32, W: i32, fc: vec2f) -> vec3f {
  let p = ro + rd * t;
  let n = nrm(p);
  let cosi = clamp(-dot(rd, n), 0.0, 1.0);
  var k = mediumOf(dist4(p - n * 0.004));
  if (k < 0) { k = nearestOf(dist4(p)); }
  cb = k;
  let alpha = mb.b[k].surf.x;
  let sigR = 1.41421 * alpha;
  let mir = reflect(rd, n);
  // what the surface itself mirrors: spread by its roughness (a rough surface as frost.wgsl reads it),
  // or, under a film on a polished surface, the surroundings sharp; and the film's studio
  var envR = seenB(mir, sigR);
//#if ROUGH
  if (alpha > 0.0) { envR = frTop(rd, n, xi, cosi); }
//#endif
//#if FILM
  if (mb.b[k].film.x > 0.0 && alpha <= 0.0) { envR = filmEnv(mir, filmLod(p, n, cosi)); }
//#endif
  envR += skyB(k, mir);
  if (opaqueB(k)) {
    // a metal: what it mirrors is followed on, to the lights or to the other bodies
    let R = surfB(-1, k, cosi, p, n, 550.0);
    return mb.b[k].glow.rgb * 0.3 + R * (beyondB(p + n * 0.006, mir, -1, 550.0, sigR * sigR) + skyB(k, mir));
  }

  // a clear or milky body or a thin wall: its own reflection, and each wavelength's way through it
  let alike = mb.disp.x == 0.0 && mb.disp.y == 0.0 && mb.disp.z == 0.0 && mb.disp.w == 0.0;
  var one = vec3f(0.0);
  var acc = vec3f(0.0);
  var wsum = vec3f(0.0);
  var racc = vec3f(0.0);
  var r1 = vec3f(0.0);
  for (var i = 0; i < MAX_NW; i++) {
    if (i >= W || (alike && i >= 1)) { break; }
    let lam = 400.0 + 300.0 * (f32(i) + xi) / f32(W);
    let w = specW(lam);
    let R0 = surfB(-1, k, cosi, p, n, lam);
    racc += w * R0;
    r1 = R0;
    let n1 = iorB(k, lam);
    var dir = refract(rd, n, 1.0 / n1);
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
    if (mb.kind[k] > 2.5) { att *= mb.b[k].axis.xyz; }
    var L = vec3f(0.0);
    // Surfaces met inside are of two sorts. Where the ray is sent back (by the air outside, a metal) it
    // has bounced, and bounces are counted as plain glass counts them. Where another clear or milky
    // body begins, the ray goes on into it and that is no bounce: two bodies of the same glass then draw
    // exactly as one piece would.
    var b = 0;
    var filmed = false;                     // a film has sent the light back inside at least once
    for (var e = 0; e < MAX_NB + 2 * COUNT; e++) {
      if (b >= u.nb || max3(att) < 0.02) { break; }
      var q: vec3f;
      var nq: vec3f;
      var nx: i32;
      let s = eventB(pos, dir, m, &q, &nq, &nx);
      L += throughB(m, pos, dir, s, &att, &sg2);
      var c = -dot(dir, nq);
      if (c < 0.0) { nq = -nq; c = -c; }
      let o = max(m, nx);
      var R = surfB(m, nx, c, q, nq, lam);
      let al = mb.b[o].surf.x;
      let mir = reflect(dir, nq);
      var next = sg2 + 2.0 * al * al;      // the spread of what this surface sends back inside
      if (opaqueB(nx)) {
        // a metal met from inside this body
        L += att * mb.b[nx].glow.rgb * 0.3;
      } else {
        let na = iorB(m, lam);
        let nb = iorB(nx, lam);
        var od = refract(dir, nq, na / nb);
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
          od = frExit(dir, nq, nq, fw, na);
          wide = frVout(sg2, fw, dot(od, nq), na, vec2f(0.0));
          next = frNext(sg2, vec2f(0.0));
        }
//#endif
        if (dot(od, od) > 0.5) {
          var X = att * through;
          if (mb.kind[o] > 2.5) { X *= mb.b[o].axis.xyz; }
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
          L += X * (beyondB(q - nq * 0.006, od, nx, lam, wide) + sky);
        }
      }
      if (!opaqueB(nx) && mb.b[o].film.x > 0.0) { filmed = true; }
      att *= R;
      sg2 = next;
      dir = mir;
      pos = q + nq * 0.004;
      b++;
    }
    acc += w * L;
    wsum += w;
    one = L;
  }
  var top = racc / wsum;
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
  return res;
}

// the reflection blended in where a ray grazes the outline
fn grazeB(p: vec3f, rd: vec3f) -> vec3f {
  let n = nrm(p);
  let k = nearestOf(dist4(p));
  cb = k;
  let mir = reflect(rd, n);
  var R = vec3f(1.0);
//#if METAL
  if (mb.kind[k] > 0.5 && mb.kind[k] < 1.5) { R = metalRGB(0.05, 1.0, 0.0); }
//#endif
  return R * (seenB(mir, 1.41421 * mb.b[k].surf.x) + skyB(k, mir));
}
