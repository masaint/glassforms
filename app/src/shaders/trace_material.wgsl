
// ---- materials beyond plain glass ---------------------------------------------------------------------
// Put into the tracer only when the material is something other than plain glass (render.rs,
// gf_core::material::tracer_parts); of what follows, only the parts whose material is in use are kept
// (the lines that begin with a comment mark and `#if NAME`, `#else`, `#endif` say which is which). Plain glass is traced by shade() and beyond() above, exactly as before;
// everything else by shadeM() here, which does what shade() does and the rest besides.
//
// The same text serves the tracer of several bodies, each of its own material (trace_bodies.wgsl;
// gf_core::material::bodies_parts). It is then cut with BODIES on and with the parts that any of the
// bodies uses, and every `mu.` below reads the numbers of the body in hand (`cb` there) instead of the
// scene's one material: the optics of a film, a metal, a rough surface and the light inside a milky body
// are the very same functions for one material and for several. What only a scene of one material uses
// (shadeM, beyondM, grazeM and their helpers) is left out of it.

struct MU {
  surf: vec4f,        // x: roughness (spread of the surface's slopes; 0 = polished)   y: (kept for the renderer)
                      // z: the index the body sees above it (1 = air, the coat's with a full coat)
                      // w: the share of the roughness that is done by sampling (1 = all of it)
  film: vec4f,        // x: film thickness, nm   y: its index   z: how much it varies   w: size of that pattern
  film2: vec4f,       // x: where the pattern is in its flow (turns)   y: drain   zw: absorption by wavelength as red, green
  body: vec4f,        // rgb: the body's colour (a metal's tint)   w: 1 = a metal from the table, 0 = a perfect mirror
  sss: vec4f,         // x: scatterings per unit length   y: reach of diffuse light beyond the surface   z: events followed at most
                      // w: the index of a thin wall's own glass
  sssA: vec4f,        // rgb: absorption of the milky body per unit length
  sssK: vec4f,        // rgb: how fast diffuse light dies away with depth
  sssC: vec4f,        // rgb: the colour of a thick piece
  glow: vec4f,        // rgb: light given off per unit length   w: how tightly it keeps to the core
  exo: vec4f,         // x: graded index, the change   y: 1 / its depth   z: double refraction (second index minus first)
                      // w: the film's sky once more (read by skyFar: why it is here twice is said there)
  axis: vec4f,        // xyz: the crystal's axis, in the shape's own frame (for a thin wall: the colour its tint
                      // gives light each time it passes through)   w: absorption by wavelength as blue
  pic: vec4f,         // the environment picture's softer levels: x: how bright   y: the angle one texel of the
                      // sharpest level covers   z: how far it is softened for the glass anyway   w: the softest level
  more: vec4f,        // x: the film's sky (a soft white light only the film reflects)
  tab: array<vec4f, 32>,      // every 10 nm from 400: x, y = the metal's n and k   z = absorption per unit length
  film3: vec4f,       // a draining film (film.rs): x: its thickness at the lowest point of the shape, nm
                      // y: the share it is thinner by at the highest   z: how far round from a summit its sink
                      // reaches nothing   w: 1 / the angle over which it climbs back out of the sink
  film4: vec4f,       // x: how deep the sink is (0 = none, 1 = to nothing)   y: below this many nm a free film is
                      // black film   w: which way is up (1, or -1)
  film5: vec4f,       // x: the film's mean thickness, nm (read only for pieces of the body seen through it)
                      // y: 1 = the family's lights are lit beside the environment picture
}
//#if !BODIES
@group(0) @binding(9) var<uniform> mu: MU;
//#if HOLLOW
// (whether the body in hand is a thin wall around air: here it always is; among several bodies, if its kind says so)
fn isHollow() -> bool { return true; }
//#endif
//#endif

//#if HOLLOW
// A thin wall around air: what the wall reflects of light arriving at cosine c, both its faces together
// (the wall is far thicker than light's wavelength, so the two reflections simply add up, over and over).
fn wallR(c: f32) -> f32 {
  let F = fresnel(c, 1.0, mu.sss.w);
  return 2.0 * F / (1.0 + F);
}
//#endif

// the tables at a wavelength
fn tabAt(lam: f32) -> vec4f {
  let x = clamp((lam - 400.0) * 0.1, 0.0, 29.999);
  let i = i32(x);
  return mix(mu.tab[i], mu.tab[i + 1], x - f32(i));
}

//#if ENVL
// The lights as a rough or milky surface gathers them: from a spread of directions around rd, of
// standard deviation sig (radians). Instead of many samples, one lookup of a copy of the lights' map
// that was blurred that much beforehand (material.rs: env_levels; copy k is blurred by one of its own
// texels, T0 * 2^k; the GPU blends the two copies nearest the spread asked for), which costs nothing
// and has no grain. With an environment picture, its own softer levels are read the same way and added.
@group(0) @binding(10) var softTex: texture_2d<f32>;
@group(0) @binding(11) var softSmp: sampler;
const T0: f32 = 0.024543693;
fn envL(rd: vec3f, sig: f32) -> vec3f {
  let d = toEnv(rd);
  let uv = vec2f(atan2(d.z, d.x) * 0.15915494 + 0.5, acos(clamp(d.y, -1.0, 1.0)) * 0.31830989);
  let lv = clamp(select(log2(max(sig, 1e-6) / T0), sig / (2.0 * T0), sig < 2.0 * T0), 0.0, 7.0);
//#if PICTURE
  // (the picture is already softened by mu.pic.z for the glass: the two spreads add as squares)
  let lp = clamp(log2(sqrt(sig * sig + mu.pic.z * mu.pic.z) / mu.pic.y), 0.0, mu.pic.w);
  return textureSampleLevel(softTex, softSmp, uv, lv).rgb + textureSampleLevel(pictureTex, pictureSmp, d, lp).rgb * mu.pic.x;
//#else
  return textureSampleLevel(softTex, softSmp, uv, lv).rgb;
//#endif
}
// the lights along rd, softened by sig (0 = as they are); a rough surface has its own way (frost.wgsl)
//#if !ROUGH
fn seen(rd: vec3f, sig: f32) -> vec3f { return envL(rd, sig); }
//#endif
//#else
fn seen(rd: vec3f, sig: f32) -> vec3f { return env(rd); }
//#endif

//#if RANDOM
fn pcg(v: u32) -> u32 {
  let x = v * 747796405u + 2891336453u;
  let w = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (w >> 22u) ^ w;
}
// Two numbers in 0..1 for the j-th random choice along the path of the ray that left the camera along
// rd0. Each pixel has its own starting point, and from frame to frame (and from ray to ray within a
// frame) the numbers run along a sequence that fills the square evenly, so that the ten or so frames
// kept while the view moves are ten well-spread samples and add up cleanly instead of boiling.
// (Which pixel it is, is worked back from the ray: that way nothing more has to be carried along.)
fn rnd2(j: i32, rd0: vec3f, xi: f32) -> vec2f {
  let uv = rd0.xy / (-rd0.z * u.tan_fov) - u.shift;
  let px = vec2u(max((uv * u.res.y + u.res) * 0.5, vec2f(0.0)));
  var lensN = 1;
  if (u.lens > 0.0) { lensN = max(u.lens_n, 1); }
  let S = clamp(u.ns * lensN, 1, 32);
  let a = pcg(px.x + pcg(px.y + pcg(u32(j) + 77u)));
  let b = pcg(a);
  return fract(vec2f(f32(a >> 8u), f32(b >> 8u)) / 16777216.0 + vec2f(0.7548777, 0.5698403) * f32((u.frame % 2048) * S + i32(xi * f32(S))));
}
//#endif

//#if ROUGH
// two numbers in 0..1 from a point: different for every ray and every frame, with no order to them
fn hash2(v: vec3f) -> vec2f {
  let a = pcg(bitcast<u32>(v.x) + pcg(bitcast<u32>(v.y) + pcg(bitcast<u32>(v.z) + u32(u.frame))));
  return vec2f(f32(a >> 8u), f32(pcg(a) >> 8u)) / 16777216.0;
}
//#endif
// How far the reflection off the surface is spread by its roughness (standard deviation, radians):
// nothing for a polished surface. (A rough one's is in frost.wgsl, with everything else roughness does.)
//#if !ROUGH
fn spreadR() -> f32 { return 0.0; }
//#endif

//#if FILM
// (Where the film is how thick, filmNm, and the light that only the film mirrors, skyOf and skyFar, are
// in film.wgsl with the rest of what makes a film look like one. Here: its optics.)

// Reflectance of a film of index n2, d nm thick, between a medium of index n1 (where the light is) and
// one of index n3, at one wavelength; every reflection back and forth inside the film is counted.
// c1: cosine of the angle the light arrives at.
fn filmR(c: f32, lam: f32, d: f32, n1: f32, n2: f32, n3: f32) -> f32 {
  let c1 = max(c, 1e-4);                    // never quite edge-on: there the sums divide nothing by nothing
  let s1 = sqrt(max(0.0, 1.0 - c1 * c1));
  if (n1 * s1 >= n3) { return 1.0; }
  if (n1 * s1 >= n2) {
    // the light cannot travel in the film at this angle, only leak through it: what gets through is what
    // the bare surface would let through, dying away with the film's thickness
    return 1.0 - (1.0 - fresnel(c1, n1, n3)) * exp(-12.566371 * d * sqrt(n1 * n1 * s1 * s1 - n2 * n2) / lam);
  }
  let s2 = n1 / n2 * s1;
  let c2 = sqrt(max(0.0, 1.0 - s2 * s2));
  let s3 = n1 / n3 * s1;
  let c3 = sqrt(max(0.0, 1.0 - s3 * s3));
  let cphi = cos(12.566371 * n2 * d * c2 / lam);
  var r12 = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2);
  var r23 = (n2 * c2 - n3 * c3) / (n2 * c2 + n3 * c3);
  let Rs = (r12 * r12 + r23 * r23 + 2.0 * r12 * r23 * cphi) / (1.0 + r12 * r12 * r23 * r23 + 2.0 * r12 * r23 * cphi);
  r12 = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2);
  r23 = (n3 * c2 - n2 * c3) / (n3 * c2 + n2 * c3);
  let Rp = (r12 * r12 + r23 * r23 + 2.0 * r12 * r23 * cphi) / (1.0 + r12 * r12 * r23 * r23 + 2.0 * r12 * r23 * cphi);
  return 0.5 * (Rs + Rp);
}
// The same as a colour, for light that is not being followed wavelength by wavelength.
fn filmRGB(c1: f32, d: f32, n1: f32, n3: f32) -> vec3f {
  var c = vec3f(0.0);
  var ws = vec3f(0.0);
  for (var j = 0; j < 10; j++) {
    let lam = 415.0 + 30.0 * f32(j);
    let w = specW(lam);
    c += w * filmR(c1, lam, d, n1, mu.film.y, n3);
    ws += w;
  }
  return clamp(c / ws, vec3f(0.0), vec3f(1.0));
}
//#endif

// ---- what is shared with the tracer of several bodies -------------------------------------------------
//#if METAL
// Unpolarised reflectance of a metal of index n + ik, seen from a clear medium of index n1 (the exact
// Fresnel equations for an absorbing medium).
fn fresMetal(c: f32, n1: f32, n: f32, k: f32) -> f32 {
  let c2 = c * c;
  let s2 = 1.0 - c2;
  let e = n / n1;
  let ek = k / n1;
  let t0 = e * e - ek * ek - s2;
  let ab = sqrt(t0 * t0 + 4.0 * e * e * ek * ek);
  let t1 = ab + c2;
  let a = sqrt(max(0.5 * (ab + t0), 0.0));
  let t2 = 2.0 * a * c;
  let Rs = (t1 - t2) / (t1 + t2);
  let t3 = c2 * ab + s2 * s2;
  let t4 = t2 * s2;
  return 0.5 * (Rs + Rs * (t3 - t4) / (t3 + t4));
}

//#if FILM
// complex numbers, as (real, imaginary)
fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn cdiv(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / dot(b, b); }
fn csqrt(a: vec2f) -> vec2f {
  let m = length(a);
  return vec2f(sqrt(max(0.5 * (m + a.x), 0.0)), select(1.0, -1.0, a.y < 0.0) * sqrt(max(0.5 * (m - a.x), 0.0)));
}
// A film on a metal: as filmR, with the body under the film absorbing (index n3 = n + ik).
fn filmMetal(c: f32, lam: f32, d: f32, n1: f32, n2: f32, n3: vec2f) -> f32 {
  let c1 = max(c, 1e-4);
  let s1 = sqrt(max(0.0, 1.0 - c1 * c1));
  let s2 = min(n1 / n2 * s1, 1.0);
  let c2 = sqrt(max(0.0, 1.0 - s2 * s2));
  let s3 = cdiv(vec2f(n1 * s1, 0.0), n3);
  let c3 = csqrt(vec2f(1.0, 0.0) - cmul(s3, s3));
  let phi = 12.566371 * n2 * d * c2 / lam;
  let e = vec2f(cos(phi), sin(phi));
  let n3c3 = cmul(n3, c3);
  let n2c2 = vec2f(n2 * c2, 0.0);
  var r12 = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2);
  var x = cmul(cdiv(n2c2 - n3c3, n2c2 + n3c3), e);
  var r = cdiv(vec2f(r12, 0.0) + x, vec2f(1.0, 0.0) + r12 * x);
  let Rs = dot(r, r);
  r12 = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2);
  x = cmul(cdiv(n3 * c2 - n2 * c3, n3 * c2 + n2 * c3), e);
  r = cdiv(vec2f(r12, 0.0) + x, vec2f(1.0, 0.0) + r12 * x);
  return 0.5 * (Rs + dot(r, r));
}
//#endif

// What the metal reflects, as a colour, for light arriving at cosine c through a medium of index n1.
// d: thickness of the film on it there (nm).
fn metalRGB(c: f32, n1: f32, d: f32) -> vec3f {
  var col = vec3f(1.0);       // a perfect mirror: all of it, whatever the angle (and whatever film lies on it)
  if (mu.body.w > 0.5) {
    col = vec3f(0.0);
    var ws = vec3f(0.0);
    for (var j = 0; j < 16; j++) {
      let lam = 400.0 + 20.0 * f32(j);
      let t = mu.tab[2 * j];
      let w = specW(lam);
//#if FILM
      col += w * filmMetal(c, lam, d, n1, mu.film.y, t.xy);
//#else
      col += w * fresMetal(c, n1, t.x, t.y);
//#endif
      ws += w;
    }
    col = clamp(col / ws, vec3f(0.0), vec3f(1.0));
  }
  // the body's colour tints the reflection, less and less towards grazing (where every surface is a mirror)
  return col * mix(mu.body.rgb, vec3f(1.0), pow(1.0 - c, 5.0));
}

//#endif

//#if SSS
// Light that has passed through a milky body without being turned right round has still been nudged
// many times, a little each time: what is seen through it is softened, by a spread whose square grows
// with the number of scatterings on the way (this many square radians for each).
const HAZE: f32 = 2.0;
//#if !WALK
// The light inside the body: how much there is at each point, from every direction together, by colour.
// It is worked out beforehand into a small volume (scatter.rs, scatter.wgsl): the lights' own light as
// far as it gets in, and the light scattered on from there, which spreads through the body and leaves
// by its surface. Nothing in it is random and it does not depend on the view.
@group(0) @binding(24) var glowTex: texture_3d<f32>;
//#if !BODIES
fn glowAt(x: vec3f) -> vec3f { return textureSampleLevel(glowTex, volSmp, x, 0.0).rgb; }
//#endif
// Points at which a stretch of path looks the light up.
const GLOW_N: i32 = 6;
// What a stretch of path inside the body gathers of that light: from pos along dir for a length s, at
// each point the light there, the share of it that is scattered into the path per unit length (evenly
// into all directions: one part in 4 pi), and what survives the way back to where the stretch began.
// The points are placed so that each stands for an equal share of what gets back (of the colour that
// gets back best; the others are weighed down by what more they lose): near the start for a dense body,
// evenly along the stretch for a nearly clear one. Leaving the body squeezes light from a narrow cone
// inside into the whole half-space outside, which thins it by the square of the index.
fn glowAlong(pos: vec3f, dir: vec3f, s_in: f32) -> vec3f {
  let s = max(s_in, 0.0);                   // (settling onto the surface can leave a length a hair below nothing)
  let k = mu.sss.x + min(mu.sssA.r, min(mu.sssA.g, mu.sssA.b));
  let all = 1.0 - exp(-k * s);
  var sum = vec3f(0.0);
  for (var j = 0; j < GLOW_N; j++) {
    let t = -log(1.0 - (f32(j) + 0.5) / f32(GLOW_N) * all) / k;
    let x = toObj(pos + dir * t) / (2.0 * HB) + 0.5;
    sum += glowAt(x) * exp(-(mu.sssA.rgb + mu.sss.x - k) * t);
  }
  return sum * (all / (f32(GLOW_N) * k) * mu.sss.x * 0.0795775 / (u.ior * u.ior));
}
//#endif
//#endif

//#if CLAY
// ---- clay under a glaze --------------------------------------------------------------------------------
// The matte clay as it is, with what lies on top of it: a clear coat's reflection of the lighting, or a
// thin film's colours (on bare clay the film is taken to lie on something that bends light as fired clay
// does, index 1.5).
fn clayM(p: vec3f, rd: vec3f) -> vec3f {
  let n = nrm(p);
  let c = clamp(-dot(rd, n), 0.0, 1.0);
//#if FILM
//#if COAT
  let F = filmRGB(c, filmNm(p, n), 1.0, mu.surf.z);
//#else
  let F = filmRGB(c, filmNm(p, n), 1.0, 1.5);
//#endif
//#else
  let F = vec3f(fresnel(c, 1.0, mu.surf.z));
//#endif
//#if FILM
//#if ROUGH
  return F * (seen(reflect(rd, n), spreadR()) + skyOf(reflect(rd, n))) + (1.0 - F) * clay(p, rd);
//#else
  return F * (filmEnv(reflect(rd, n), filmLod(p, n, c)) + skyOf(reflect(rd, n))) + (1.0 - F) * clay(p, rd);
//#endif
//#else
  return F * seen(reflect(rd, n), spreadR()) + (1.0 - F) * clay(p, rd);
//#endif
}
//#else
// ---- a mirror or a metal -------------------------------------------------------------------------------
//#if METAL
//#if !BODIES
// The first surface a ray outside the body meets: true if there is one, and how far along.
fn outHit(ro: vec3f, rd: vec3f, tHit: ptr<function, f32>) -> bool {
  let b = dot(ro, rd);
  let h = b * b - (dot(ro, ro) - BOUND * BOUND);
  if (h <= 0.0) { return false; }
  let tFar = -b + sqrt(h);
  var t = 0.0;
  for (var i = 0; i < STEPS_OUT; i++) {
    let d = mapW(ro + rd * t);
    if (d < 0.0008) {
      t += mapNW(ro + rd * t);
      *tHit = t;
      return true;
    }
    t += d * STEP_SCALE;
    if (t > tFar) { break; }
  }
  return false;
}

// An opaque reflecting body: the reflection is followed from surface to surface (the shape mirrored in
// itself) for up to `nb` bounces, each tinted by the metal's own reflectance at that angle.
fn shadeM(ro: vec3f, rd_in: vec3f, t: f32, xi: f32, W: i32, fc: vec2f) -> vec3f {
  var p = ro + rd_in * t;
  var rd = rd_in;
  var col = vec3f(0.0);
  var thr = vec3f(1.0);
  let nx = mu.surf.z;                       // the index of what lies on the metal: air, or the clear coat
//#if ROUGH
  var fv = 0.0;                             // how far the light followed is spread by now
  var fp = 0.0;                             // and how wide its cone is where it has landed
//#endif
  for (var b = 0; b < MAX_NB; b++) {
//#if ROUGH
    // (the surface as the cone of light meets it: the normal of the patch it lands on; what of the cone
    // the patch does not cover passes it by and sees the lights)
    let n = frNrm(p, fp);
    if (b > 0) {
      let cv = frCover(p, rd, n, fp);
      col += thr * (1.0 - cv) * frMetalLit(rd, rd, fv, b);
      thr *= cv;
    }
//#else
    let n = nrm(p);
//#endif
    let cosi = clamp(-dot(rd, n), 0.0, 1.0);
    let mir = reflect(rd, n);
//#if GLOW
    col += thr * mu.glow.rgb * 0.25;        // a glowing metal simply shines of itself
//#endif
//#if FILM
    let fd = filmNm(p, n);
//#else
    let fd = 0.0;
//#endif
//#if COAT
    // The clear coat: its own sharp reflection on top, and the metal underneath seen through it (at
    // the shallower angle the coat bends the light to, and against the coat instead of air).
    let cc = sqrt(max(1.0 - (1.0 - cosi * cosi) / (nx * nx), 0.0));
//#if FILM
    let Fc = filmRGB(cosi, fd, 1.0, nx);
    let R = metalRGB(cc, nx, 0.0);
//#else
    let Fc = vec3f(fresnel(cosi, 1.0, nx));
    let R = metalRGB(cc, nx, fd);
//#endif
    col += thr * Fc * env(mir);
    thr *= 1.0 - Fc;
//#else
    let R = metalRGB(cosi, 1.0, fd);
//#endif
    // where the reflection goes: straight for a polished surface, off one facet of a rough one
    var dir = mir;
//#if ROUGH
    dir = frMetalDir(rd, n, b, rd_in, xi);
    // The lobe this surface reflects: spread by what the light was spread by already, by the facets,
    // and by the bend of the patch the cone landed on. Its middle is followed to the next surface while
    // it is narrow (share fo); the rest of it sees the lights, less what the shape itself hides of them.
    fv = frMetalV(fv, frBend(p, n, fp, rd));
    let fo = frFollow(fv);
    let lit0 = frMetalLit(dir, frMid(rd, n), fv, b);
    let lit = mix(frMetalHid(p, n, frMid(rd, n), fv, lit0, R), lit0, fo);
//#else
//#if FILM
    let lit = filmEnv(mir, filmLod(p, n, cosi));
//#else
    let lit = env(mir);
//#endif
//#endif
    var t2 = 0.0;
    let o = p + n * 0.004;
//#if ROUGH
    if (b + 1 >= u.nb || fo <= 0.0 || !outHit(o, dir, &t2)) {
//#else
    if (b + 1 >= u.nb || !outHit(o, dir, &t2)) {
//#endif
      // it leaves the shape (or the bounces are used up): the lights
//#if FILM
      col += thr * R * (lit + skyOf(mir));
//#else
      col += thr * R * lit;
//#endif
      break;
    }
//#if ROUGH
    // the part of a wide lobe that is not followed to the next surface
    col += thr * R * lit * (1.0 - fo);
    thr *= fo;
    fp = t2 * sqrt(fv);
//#endif
    thr *= R;
    if (max(thr.x, max(thr.y, thr.z)) < 0.02) { break; }
    p = o + dir * t2;
    rd = dir;
  }
  return col;
}

fn grazeM(p: vec3f, rd: vec3f) -> vec3f {
  return seen(reflect(rd, nrm(p)), spreadR());
}
//#endif
//#else
//#if !BODIES

// ---- a clear or milky body -----------------------------------------------------------------------------
//
// How this is laid out matters as much as what it works out. The tracer's time goes on reads of the
// volume, and how many of those the GPU keeps in flight at once depends on how many numbers a pixel has
// to carry while it waits for them. Past a certain count the frame takes 1.4 times as long for the
// very same work (measured: four more numbers kept across the marching of plain glass do it). So what
// is carried from one stretch of a path to the next is kept to the least there can be: the light gathered
// goes straight into one running total, weighed on the spot, instead of being kept per wavelength.

//#if FINE
// Film colours and absorption by wavelength are weighed at several wavelengths for each path traced,
// four to a group. A path that stands for one band of the spectrum weighs one group spread through
// its band; a path that stands for the whole spectrum (no dispersion) weighs three groups spread
// through all of it. All that is kept per wavelength is how much of it is still travelling.
fn lams(i: i32, g: i32, xi: f32, W: i32) -> vec4f {
//#if MONO
  return 400.0 + 300.0 * (f32(4 * g) + vec4f(0.0, 1.0, 2.0, 3.0) + xi) / 12.0;
//#else
  return 400.0 + 300.0 * (f32(i) + (vec4f(0.0, 1.0, 2.0, 3.0) + xi) * 0.25) / f32(W);
//#endif
}
// what a group adds to red, green and blue, given how much there is of each of its wavelengths
fn tint4(l: vec4f, a: vec4f) -> vec3f { return specW(l.x) * a.x + specW(l.y) * a.y + specW(l.z) * a.z + specW(l.w) * a.w; }
fn max4(v: vec4f) -> f32 { return max(max(v.x, v.y), max(v.z, v.w)); }
//#if FILM
fn filmR4(c: f32, l: vec4f, d: f32, n1: f32, n3: f32) -> vec4f {
  return vec4f(filmR(c, l.x, d, n1, mu.film.y, n3), filmR(c, l.y, d, n1, mu.film.y, n3), filmR(c, l.z, d, n1, mu.film.y, n3), filmR(c, l.w, d, n1, mu.film.y, n3));
}
//#endif
//#if SPECTRUM
fn absorb4(l: vec4f) -> vec4f { return vec4f(tabAt(l.x).z, tabAt(l.y).z, tabAt(l.z).z, tabAt(l.w).z); }
//#endif
// what everything still travelling adds to red, green and blue
fn weigh(i: i32, xi: f32, W: i32, t0: vec4f, t1: vec4f, t2: vec4f) -> vec3f {
//#if MONO
  return tint4(lams(i, 0, xi, W), t0) + tint4(lams(i, 1, xi, W), t1) + tint4(lams(i, 2, xi, W), t2);
//#else
  return tint4(lams(i, 0, xi, W), t0);
//#endif
}
//#endif

//#if GRIN
// The index at depth d below the surface, less the index at the surface: it changes by mu.exo.x over a
// depth of about 1 / mu.exo.y and then stays.
fn grinAt(depth: f32) -> f32 { return mu.exo.x * (1.0 - exp(-max(depth, 0.0) * mu.exo.y)); }

// Inside a body whose index changes with depth a ray curves towards where the index is higher. Follow
// it from pos heading dir (both are moved on to where it reaches the surface again); returns the length
// of the path. n0: the index at the surface for this wavelength.
// Near the surface, where the index is changing, the ray is moved in short steps and turned at each by
// the slope of the index (which points along the slope of the distance field); deeper, where the index
// has settled, it is marched in the usual long safe steps.
fn travel(pos: ptr<function, vec3f>, dir: ptr<function, vec3f>, n0: f32) -> f32 {
  var p = *pos;
  var d = *dir;
  var len = 0.0;
  let vox = 2.0 * HB / u.vres;
  let deep = 5.0 / mu.exo.y;                  // below this the index no longer changes
  let hs = max(0.4 / mu.exo.y, 1.5 * vox);    // the short step
  for (var i = 0; i < 96; i++) {
    let depth = -mapW(p);
    if (depth < 0.0008 && i > 0) { break; }
    if (depth > deep + hs) {
      let s = (depth - deep) * STEP_SCALE;
      p += d * s;
      len += s;
    } else {
      // the slope of the distance field, from four lookups about the point
      let e = vec2f(1.0, -1.0) * vox;
      let g = normalize(e.xyy * mapW(p + e.xyy) + e.yyx * mapW(p + e.yyx) + e.yxy * mapW(p + e.yxy) + e.xxx * mapW(p + e.xxx));
      // the index rises with depth, and depth rises against g
      let dn = -g * (mu.exo.x * mu.exo.y * exp(-max(depth, 0.0) * mu.exo.y));
      let s = min(hs, max(depth * STEP_SCALE, 0.25 * vox));
      d = normalize(d * (n0 + grinAt(depth)) + dn * s);
      p += d * s;
      len += s;
    }
    if (len > 2.5 * BOUND) { break; }
  }
  // settle onto the smooth field's surface, along the way the ray is now heading
  var t = 0.0;
  t -= mapNW(p + d * t);
  t -= mapNW(p + d * t);
  *pos = p + d * t;
  *dir = d;
  return len + t;
}
//#endif

//#if BIREF
// Double refraction. In a crystal with one axis (calcite, quartz) a ray splits in two: an ordinary one
// that behaves as in glass, with index no, and an extraordinary one whose index depends on its direction
// relative to the axis, between no (along the axis) and nE (across it). For the extraordinary ray the
// waves and the light itself travel in slightly different directions, so both are kept: k is the wave
// (its length is the index it meets), and the ray is what is marched.

// the direction the light of an extraordinary wave k travels
fn rayE(k: vec3f, a: vec3f, no: f32, nE: f32) -> vec3f {
  return normalize(k / (nE * nE) + (1.0 / (no * no) - 1.0 / (nE * nE)) * dot(k, a) * a);
}
// A wave crossing a surface keeps the part of k that lies along the surface. kt: that part. N: the
// surface's normal, pointing the way the new wave is to go. a: the axis. Returns the new wave, or zero
// if there is none (total reflection).
fn waveE(kt: vec3f, N: vec3f, a: vec3f, no: f32, nE: f32) -> vec3f {
  let ie = 1.0 / (nE * nE);
  let dl = 1.0 / (no * no) - ie;
  let na = dot(N, a);
  let ka = dot(kt, a);
  let A = ie + dl * na * na;
  let B = 2.0 * dl * ka * na;
  let Cc = dot(kt, kt) * ie + dl * ka * ka - 1.0;
  let disc = B * B - 4.0 * A * Cc;
  if (disc <= 0.0) { return vec3f(0.0); }
  // of the two waves that fit, the one whose light travels towards N
  let k1 = kt + (-B + sqrt(disc)) / (2.0 * A) * N;
  if (dot(rayE(k1, a, no, nE), N) > 0.0) { return k1; }
  return kt + (-B - sqrt(disc)) / (2.0 * A) * N;
}
// the crystal's axis as the camera sees it
fn axisW() -> vec3f {
  return normalize(vec3f(dot(u.obj0.xyz, mu.axis.xyz), dot(u.obj1.xyz, mu.axis.xyz), dot(u.obj2.xyz, mu.axis.xyz)));
}
//#endif

//#if WALK
// plain random numbers, as many as wanted
var<private> rState: u32;
fn rnd() -> f32 {
  rState = pcg(rState);
  return f32(rState >> 8u) / 16777216.0;
}
// A direction drawn evenly from all directions.
fn anywhere() -> vec3f {
  let z = 1.0 - 2.0 * rnd();
  let r = sqrt(max(1.0 - z * z, 0.0));
  let ph = 6.2831853 * rnd();
  return vec3f(r * cos(ph), r * sin(ph), z);
}
// March inside the body from ro along rd for at most tMax: how far it got, and whether that was the
// surface (then it is settled onto the smooth field, as marchIn does).
fn marchTo(ro: vec3f, rd: vec3f, tMax: f32, hit: ptr<function, bool>) -> f32 {
  var t = 0.0;
  *hit = true;
  for (var i = 0; i < STEPS_IN; i++) {
    let d = -mapW(ro + rd * t);
    if (d < 0.0008) { break; }
    t += d * STEP_SCALE;
    if (t >= tMax) { *hit = false; return tMax; }
    if (t > 2.5 * BOUND) { break; }
  }
  t -= mapNW(ro + rd * t);
  t -= mapNW(ro + rd * t);
  return t;
}
// The first surface a ray outside the body meets: true if there is one, and how far along.
fn airHit(ro: vec3f, rd: vec3f, tHit: ptr<function, f32>) -> bool {
  let b = dot(ro, rd);
  let h = b * b - (dot(ro, ro) - BOUND * BOUND);
  if (h <= 0.0) { return false; }
  let tFar = -b + sqrt(h);
  var t = 0.0;
  for (var i = 0; i < STEPS_OUT; i++) {
    let d = mapW(ro + rd * t);
    if (d < 0.0008) {
      t += mapNW(ro + rd * t);
      *tHit = t;
      return true;
    }
    t += d * STEP_SCALE;
    if (t > tFar) { break; }
  }
  return false;
}
// A ray that has just been scattered inside the body, at p, into direction d: followed until it reaches
// the lights. Inside, it flies a random distance and is turned to a random direction, again and again
// (the distance field says at once when a flight is sure to stay inside); the body's absorption is taken
// off along the whole way. At the surface it leaves or is sent back by chance, as the surface reflects;
// outside, it may meet the body again and go in again the same way: so one part of the body shades
// another and lights it, as it does. Returns the light that arrives back along it.
// (Whether the ray is inside is asked of the distance field at every step, not remembered: where the
// surface is too sharply curved for the two ways the field is read to agree, a remembered answer goes
// wrong and stays wrong.)
fn wander(p: vec3f, d: vec3f) -> vec3f {
  var pos = p;
  var dir = d;
  var att = vec3f(1.0);
  var lit = vec3f(0.0);
  for (var e = 0; e < 256; e++) {
    let depth = -mapW(pos);
    if (depth > 0.0) {
      let fly = -log(max(1.0 - rnd(), 1e-6)) / mu.sss.x;
      var got = fly;
      var hit = false;
      if (fly >= depth * 0.9) { got = max(marchTo(pos, dir, fly, &hit), 0.0); }
      pos += dir * got;
//#if GLOW
      lit += att * mu.glow.rgb * got;
//#endif
      att *= exp(-(u.absorb.xyz + mu.sssA.rgb) * got);
      if (max(att.x, max(att.y, att.z)) < 0.002) { break; }      // the body has swallowed it
      if (!hit) {
        dir = anywhere();
        continue;
      }
      let m = nrm(pos);
      let ci = dot(dir, m);
      if (ci <= 0.0) {                      // at the surface, but heading in: carry on
        pos += dir * 0.004;
        continue;
      }
      if (rnd() < fresnel(ci, u.ior, 1.0)) {
        dir = reflect(dir, m);
        pos -= m * 0.004;
      } else {
        dir = refract(dir, -m, u.ior);
        pos += m * 0.006;
      }
    } else {
      var t = 0.0;
      if (!airHit(pos, dir, &t)) { return lit + att * seen(dir, 0.1); }
      pos += dir * max(t, 0.0);
      let n = nrm(pos);
      let ci = -dot(dir, n);
      if (ci <= 0.0) {                      // touching the surface on its way out: carry on
        pos += dir * 0.004;
        continue;
      }
      if (rnd() < fresnel(ci, 1.0, u.ior)) {
        dir = reflect(dir, n);
        pos += n * 0.004;
      } else {
        dir = refract(dir, n, 1.0 / u.ior);
        pos -= n * 0.004;
      }
    }
  }
  return lit;
}
//#endif

// The index a path is traced with: that of the i-th of W wavelengths (the middle of the spectrum when one
// path stands for all), and, where one wavelength is weighed for the path, what it adds to red, green and blue.
fn iorOf(i: i32, xi: f32, W: i32) -> f32 {
//#if MONO
  return iorAt(550.0);
//#else
  return iorAt(400.0 + 300.0 * (f32(i) + xi) / f32(W));
//#endif
}
//#if !FINE
fn wOf(i: i32, xi: f32, W: i32) -> vec3f {
//#if MONO
  return vec3f(1.0);
//#else
  return specW(400.0 + 300.0 * (f32(i) + xi) / f32(W));
//#endif
}
// What the top of the surface reflects of light arriving from outside at cosine c: the coat if there is
// one, else the body (as plain glass has it: at the index of the middle of the spectrum).
fn top1(c: f32) -> f32 {
//#if COAT
  return fresnel(c, 1.0, mu.surf.z);
//#else
//#if HOLLOW
  return wallR(c);
//#else
  return fresnel(c, 1.0, u.ior);
//#endif
//#endif
}
//#else
// The same for a group of wavelengths: with a film on top, each has its own.
fn top4(c: f32, l: vec4f, d: f32) -> vec4f {
//#if COAT
//#if FILM
  return filmR4(c, l, d, 1.0, mu.surf.z);
//#else
  return vec4f(fresnel(c, 1.0, mu.surf.z));
//#endif
//#else
//#if FILM
  return filmR4(c, l, d, 1.0, u.ior);
//#else
//#if HOLLOW
  return vec4f(wallR(c));
//#else
  return vec4f(fresnel(c, 1.0, u.ior));
//#endif
//#endif
//#endif
}
//#endif
//#if COAT
// Under a coat: what the body reflects of light that got into the coat (c: the cosine it arrived at outside).
fn under(c: f32, ior: f32) -> f32 {
  return fresnel(sqrt(max(1.0 - (1.0 - c * c) / (mu.surf.z * mu.surf.z), 0.0)), mu.surf.z, ior);
}
//#endif

//#if !BUBBLE
// (A film around air, a soap bubble, is traced by shadeM, beyondB and grazeM in film.wgsl instead of
// what follows.)
// A ray has just left the body at ro, heading rd: what it brings back from further pieces of the body
// and from the lights. This is beyond() with the material's surface (film, coat, roughness) and inside
// (milkiness, glow, absorption by wavelength taken as three colours). sig: how spread the light leaving
// through the surface behind it is (a rough surface, a milky body), which softens the lights it sees.
// With a film, sky: 1 for light that the film has sent back inside at least once, which therefore sees
// the film's sky where it leaves (it is added here, at the start, because here the direction is in hand:
// added by the caller afterwards it would be one more thing carried through all the marching below).
//#if FILM
fn beyondM(ro_in: vec3f, rd_in: vec3f, ior: f32, sig_in: f32, sky: f32) -> vec3f {
  var ro = ro_in;
  var rd = rd_in;
  var col = sky * skyDim(rd_in);
//#else
fn beyondM(ro_in: vec3f, rd_in: vec3f, ior: f32, sig_in: f32) -> vec3f {
  var ro = ro_in;
  var rd = rd_in;
  var col = vec3f(0.0);
//#endif
  var thr = vec3f(1.0);
  var sig = sig_in;
  for (var l = 0; l < LAYERS; l++) {
    let b = dot(ro, rd);
    let h = b * b - (dot(ro, ro) - BOUND * BOUND);
    if (h <= 0.0) { break; }
    let tFar = -b + sqrt(h);
    var t = 0.0;
    var hit = false;
    for (var i = 0; i < STEPS_OUT; i++) {
      let d = mapW(ro + rd * t);
      if (d < 0.0008) { hit = true; break; }
      t += d * STEP_SCALE;
      if (t > tFar) { break; }
    }
//#if ROUGH
    if (!hit) { frMiss(l); break; }
//#else
    if (!hit) { break; }
//#endif
    t += mapNW(ro + rd * t);
    let p = ro + rd * t;
//#if ROUGH
    // The cone of light lands on a patch of this piece: the normal is the patch's, the patch's own bend
    // spreads what it reflects and lets through, and the piece is taken to cover as much of the cone as
    // it does (the rest of the cone passes it by and sees the lights).
    let fp = t * sig;
    frHit(l, fp, sig);
    let n = frNrm(p, fp);
    let ne = frFarIn(n, rd, p);
//#if COAT
    let ci = clamp(-dot(rd, n), 0.0, 1.0);
//#else
    let ci = frCos(-dot(rd, ne));
//#endif
    {
      let cv = frCover(p, rd, n, fp);
      col += thr * (1.0 - cv) * seen(rd, sig);
      thr *= cv;
    }
//#else
    let n = nrm(p);
    let ci = clamp(-dot(rd, n), 0.0, 1.0);
//#endif
    // what this surface reflects, and what it lets in
//#if COAT
    {
      // the coat's own reflection, on top
//#if FILM
      let Ft = filmRGB(ci, filmFar(p), 1.0, mu.surf.z);
//#else
      let Ft = fresnel(ci, 1.0, mu.surf.z);
//#endif
      col += thr * Ft * seen(reflect(rd, n), sig);
      thr *= 1.0 - Ft;
    }
    let F = fresnel(sqrt(max(1.0 - (1.0 - ci * ci) / (mu.surf.z * mu.surf.z), 0.0)), mu.surf.z, ior);
//#else
//#if FILM
    let F = filmRGB(ci, filmFar(p), 1.0, ior);
//#else
//#if HOLLOW
    let F = wallR(ci);
//#else
    let F = fresnel(ci, 1.0, ior);
//#endif
//#endif
//#endif
//#if ROUGH
//#if FILM
//#if ENVL
    col += thr * F * (frFar(rd, n, p, sig, fp) + skyFar(reflect(rd, n)));
//#else
    // (seen through the body: the picture at a middling level, without measuring how the surface bends)
    col += thr * F * (filmEnv(reflect(rd, n), 3.0) + skyFar(reflect(rd, n)));
//#endif
//#else
    col += thr * F * frFar(rd, n, p, sig, fp);
//#endif
//#else
//#if FILM
    col += thr * F * (seen(reflect(rd, n), sig) + skyFar(reflect(rd, n)));
//#else
    col += thr * F * seen(reflect(rd, n), sig);
//#endif
    let ne = n;
//#endif
//#if ROUGH
    {
      // A wide cone is not followed through the piece: the piece scatters it every way, and what comes
      // out of it this side is the light from all round.
      let wd = frWide(sig);
      col += thr * (1.0 - F) * wd * frAll(rd);
      thr *= 1.0 - wd;
      if (wd > 0.999) { return col; }
    }
//#endif
    var d2 = refract(rd, ne, 1.0 / ior);
    var q0 = p - n * 0.004;
//#if GRIN
    let s = travel(&q0, &d2, ior);
    let q = q0;
//#else
    let s = marchIn(q0, d2);
    let q = q0 + d2 * s;
//#endif
//#if INNER
    {
      // what lies inside this further piece of the body; a crack that would turn the path ends it there
      let inn = inner(q0, d2, s, ior, 1);
      col += thr * (1.0 - F) * inn.light;
      if (inn.t >= 0.0) {
        return col + thr * (1.0 - F) * inn.keep * exp(-u.absorb.xyz * inn.t) * seen(inn.dir, sig);
      }
      thr *= inn.keep;
    }
//#endif
    thr *= (1.0 - F) * exp(-u.absorb.xyz * s);
//#if SPECTRUM
    thr *= exp(-vec3f(mu.film2.z, mu.film2.w, mu.axis.w) * max(s, 0.0));
//#endif
//#if GLOW
    col += thr * mu.glow.rgb * s;
//#endif
//#if SSS
    {
      // the milky body: part of the light is scattered on the way through, and that part is replaced by
      // the diffuse light inside (taken as the lights gathered by the face it came in by)
      let ext = exp(-(mu.sss.x + mu.sssA.rgb) * max(s, 0.0));
//#if WALK
      {
        // the part of the ray scattered on this stretch: one point of the stretch stands for it, drawn
        // as rays are scattered along it, and from there the ray is followed as it wanders
        let all = 1.0 - exp(-mu.sss.x * max(s, 0.0));
        let at = -log(1.0 - rnd() * all) / mu.sss.x;
        col += thr * all * exp(-mu.sssA.rgb * at) * wander(q - d2 * (s - at), anywhere());
      }
//#else
      col += thr * glowAlong(q - d2 * s, d2, s);
//#endif
      thr *= ext;
      sig = sqrt(sig * sig + HAZE * mu.sss.x * s);
    }
//#endif
    var m = nrm(q);
    var co = dot(d2, m);
    if (co < 0.0) { m = -m; co = -co; }
    // leaving again
//#if COAT
    let F2 = 1.0 - (1.0 - fresnel(co, ior, mu.surf.z)) * (1.0 - fresnel(sqrt(max(1.0 - (ior * ior) * (1.0 - co * co) / (mu.surf.z * mu.surf.z), 0.0)), mu.surf.z, 1.0));
//#else
//#if HOLLOW
//#if FILM
    let F2 = filmRGB(co, filmNm(q, m), 1.0, 1.0);
//#else
    let F2 = wallR(co);
//#endif
    thr *= mu.axis.xyz * mu.axis.xyz;       // (through the wall twice: in and out)
//#else
    let F2 = fresnel(co, ior, 1.0);
//#endif
//#endif
//#if ROUGH
//#if HOLLOW
    thr *= 1.0 - F2;
    rd = refract(d2, -m, ior);
//#else
//#if COAT
    if (F2 > 0.999) {                       // trapped by total reflection under the coat: take the bounce and stop there
      return col + thr * seen(reflect(d2, m), max(sig, spreadR()));
    }
    thr *= 1.0 - F2;
    rd = refract(d2, -m, ior);
//#else
    {
      // The wall this piece is left by, met by a cone of light: the part of it beyond the critical angle
      // is sent back inside, and is taken to leave at the bounce, softened; the rest goes on, bent and spread.
      let mf = frFarFacet(m, d2, q);
      let vi = sig * sig / (ior * ior) + frVin0(ior);
      let w = frWall(dot(d2, mf), vi, ior, 1.0, vec2f(0.0));
      if (w.x > 1.0 - frLeak()) { return col + thr * seen(reflect(d2, mf), max(sig, spreadR())); }
      col += thr * w.z * seen(reflect(d2, mf), max(sig, spreadR()));
      thr *= 1.0 - w.x;
      rd = frExit(d2, m, mf, w, ior);
      sig = sqrt(max(sig * sig, frVout(vi, w, dot(rd, m), ior, vec2f(0.0))));
    }
//#endif
//#endif
//#else
//#if !HOLLOW
    if (F2 > 0.999) {                       // trapped by total reflection: take the bounce and stop there
      return col + thr * seen(reflect(d2, m), sig);
    }
//#endif
    thr *= 1.0 - F2;
    rd = refract(d2, -m, ior);
//#endif
    ro = q + m * 0.006;
  }
  return col + thr * seen(rd, sig);
}

// What shade() does, for every material with a clear or milky body.
fn shadeM(ro: vec3f, rd: vec3f, t: f32, xi: f32, W: i32, fc: vec2f) -> vec3f {
  let p = ro + rd * t;
  let n = nrm(p);
//#if ROUGH
//#if COAT
  let ne = n;                               // (under a coat the top of the surface is the coat, and smooth)
  let cosi = clamp(-dot(rd, n), 0.0, 1.0);
//#else
  let ne = frIn(n, rd, xi);                 // the facet light gets in through (the surface's own normal, unless by chance)
  let cosi = frCos(-dot(rd, ne));
//#endif
//#else
  let cosi = clamp(-dot(rd, n), 0.0, 1.0);
//#endif
//#if WALK
  rState = pcg(bitcast<u32>(rd.x) + pcg(bitcast<u32>(rd.y) + pcg(u32(u.frame) + pcg(bitcast<u32>(xi)))));
//#endif

  // ---- the surface: what it reflects, and through which facet light gets in
//#if ROUGH
  let envB = frTop(rd, n, xi, cosi);        // the body's own reflection, spread by its roughness
//#else
  let envB = env(reflect(rd, n));
  let ne = n;
//#endif
//#if COAT
//#if ROUGH
  let envT = env(reflect(rd, n));           // the coat's reflection is sharp whatever lies under it
//#else
  let envT = envB;
//#endif
//#else
  let envT = envB;
//#endif
//#if FILM
  let fd = filmNm(p, n);
  // what the film mirrors: the surroundings sharp, unless the surface under it is rough
//#if ROUGH
  let envF = envT + skyOf(reflect(rd, n));
//#else
  let envF = filmEnv(reflect(rd, n), filmLod(p, n, cosi)) + skyOf(reflect(rd, n));
//#endif
//#else
  let fd = 0.0;
//#endif
  let p0 = p - n * 0.004;                   // where every path inside starts

//#if BIREF
  let MODES = 2;                            // every path is followed twice: ordinary, then extraordinary
//#else
  let MODES = 1;
//#endif

  // Everything gathered goes into one total, acc, weighed for red, green and blue as it comes; at the
  // end it is divided by the weight of white. The surface's own reflection goes in first, for every
  // wavelength weighed. (Neither the weights nor the reflected light are kept while the paths inside
  // are followed: weights are worked out again where they are wanted, which costs nothing beside
  // carrying them.)
  var acc = vec3f(0.0);
//#if MONO
  {
    let i = 0;
//#else
  for (var i = 0; i < MAX_NW; i++) {
    if (i >= W) { break; }
//#endif
//#if FINE
//#each
    {
      let l = lams(i, GROUP, xi, W);
      let Ft = top4(cosi, l, fd);
//#if FILM
      let lit = envF;
//#else
      let lit = envT;
//#endif
//#if COAT
      acc += tint4(l, Ft) * lit + tint4(l, (1.0 - Ft) * (1.0 - Ft) * under(cosi, iorOf(i, xi, W))) * envB;
//#else
      acc += tint4(l, Ft) * lit;
//#endif
    }
//#end
//#else
//#if COAT
    acc += wOf(i, xi, W) * (top1(cosi) * envT + (1.0 - top1(cosi)) * (1.0 - top1(cosi)) * under(cosi, iorOf(i, xi, W)) * envB);
//#else
    acc += wOf(i, xi, W) * (top1(cosi) * envT);
//#endif
//#endif
  }
  acc *= f32(MODES);

//#if MONO
  // No dispersion: every wavelength bends alike, so one path is followed and stands for them all.
  for (var j = 0; j < MODES; j++) {
    let i = 0;
//#else
  for (var j = 0; j < MAX_NW * MODES; j++) {
    let i = j / MODES;
    if (i >= W) { break; }
//#endif
    let ior = iorOf(i, xi, W);
    // ---- what gets in: past the top of the surface (air to film, coat or body) and, under a coat, into the body below it
//#if FINE
    var t0 = vec4f(0.0);
    var t1 = vec4f(0.0);
    var t2 = vec4f(0.0);
//#each
//#if COAT
    t[g] = (1.0 - top4(cosi, lams(i, GROUP, xi, W), fd)) * (1.0 - under(cosi, ior));
//#else
    t[g] = 1.0 - top4(cosi, lams(i, GROUP, xi, W), fd);
//#endif
//#end
    var thr = max(max4(t0), max(max4(t1), max4(t2)));
//#else
//#if COAT
    var thr = (1.0 - top1(cosi)) * (1.0 - under(cosi, ior));
//#else
    var thr = 1.0 - top1(cosi);
//#endif
//#endif

    // ---- the path inside
//#if BIREF
    let extra = (j % 2) == 1;
    let nE = ior + mu.exo.z;
    var kv = refract(rd, ne, 1.0 / ior) * ior;          // the wave: its length is the index it meets
    if (extra) { kv = waveE(rd - dot(rd, ne) * ne, -ne, axisW(), ior, nE); }
    var dir = normalize(kv);
    if (extra) { dir = rayE(kv, axisW(), ior, nE); }
//#else
//#if ROUGH
    var dir = frDir(refract(rd, ne, 1.0 / ior), n);
//#else
    var dir = refract(rd, ne, 1.0 / ior);
//#endif
//#endif
//#if ROUGH
    var fv = frVin0(ior);                   // how far the path's directions are spread by now
//#endif
    var pos = p0;
//#if CRACK
    var turns = 0;                          // how often cracks have turned this path
//#endif
//#if HOLLOW
    var att = mu.axis.xyz;                  // through the wall once, on the way in
//#else
    var att = vec3f(1.0);
//#endif
//#if SSS
    var haze = 0.0;                         // how far what this path sees is softened by the milky body, squared
//#endif
    for (var b = 0; b < MAX_NB; b++) {
      if (b >= u.nb || thr < 0.02) { break; }      // nothing got in (total reflection at the surface): skip the interior
      // ---- to the next surface
//#if WALK
      // How far the ray gets before it is scattered. Short of the surface: it is turned to a random
      // direction there and followed as it wanders, to the lights. As far as the surface: it is treated
      // there as a ray in clear glass is.
      let fly = -log(max(1.0 - rnd(), 1e-6)) / mu.sss.x;
      var s = fly;
      var out = false;
      if (fly >= -mapW(pos) * 0.9) { s = marchTo(pos, dir, fly, &out); }
      pos += dir * s;
//#if GLOW
      acc += {{WEIGHT}} * att * mu.glow.rgb * s;
//#endif
      att *= exp(-(u.absorb.xyz + mu.sssA.rgb) * s);
      haze += HAZE * mu.sss.x * s;
      if (!out) {
        acc += {{WEIGHT}} * att * wander(pos, anywhere());
        break;
      }
      let q = pos;
//#else
//#if GRIN
      let s = travel(&pos, &dir, ior);
      let q = pos;
//#if GLOW
      acc += {{WEIGHT}} * att * mu.glow.rgb * s;
//#endif
//#else
//#if INNER
      // What lies inside the body along this stretch (inner.wgsl): what it adds is gathered, what it
      // takes is taken, and the stretch ends at a crack that turns the path, if one does.
      let far = marchIn(pos, dir);
      let inn = inner(pos, dir, far, ior, b);
      acc += {{WEIGHT}} * att * inn.light;
      att *= inn.keep;
      let s = select(far, inn.t, inn.t >= 0.0);
//#else
      let s = marchIn(pos, dir);
//#endif
      let q = pos + dir * s;
//#if GLOW
      // light given off along the way: evenly, or mostly from the middle of the body. How near the middle
      // this stretch passes is how deep its midpoint lies, as a share of half its length (1 = through
      // the core of a tube, 0 = skimming it).
      acc += {{WEIGHT}} * att * mu.glow.rgb * (s * pow(clamp(-mapW(pos + dir * (0.5 * s)) / max(0.5 * s, 1e-4), 0.0, 1.0), mu.glow.w));
//#endif
//#endif
      att *= exp(-u.absorb.xyz * s);
//#if SSS
      {
        // part of the ray is scattered away on this stretch; in its place comes the light the body
        // holds along it, scattered towards the eye
        let ext = exp(-(mu.sss.x + mu.sssA.rgb) * max(s, 0.0));
        acc += {{WEIGHT}} * att * glowAlong(q - dir * s, dir, s);
        att *= ext;
        haze += HAZE * mu.sss.x * s;
      }
//#endif
//#endif
//#if SPECTRUM
//#each
      t[g] *= exp(-absorb4(lams(i, GROUP, xi, W)) * max(s, 0.0));
//#end
//#endif
//#if CRACK
      if (inn.t >= 0.0) {
        // turned by a crack: the path goes on from there (a crack is not one of the surface's bounces).
        // After so many turns it is taken to leave the way it is heading.
        if (turns >= i32(iu.crackC.w)) {
          acc += {{WEIGHT}} * att * crackSeen(inn.dir);
          break;
        }
        turns += 1;
        pos = q;
        dir = inn.dir;
//#if BIREF
        kv = inn.dir * length(kv);
//#endif
        b -= 1;
        continue;
      }
//#endif
//#if ROUGH
      // (the wall as the cone of light meets it: the normal of the patch it lands on, and the patch's bend)
      var mN = frNrm(q, s * sqrt(fv));
      let bend = frBend(q, mN, s * sqrt(fv), dir);
//#else
      var mN = nrm(q);
//#endif
      var ci = dot(dir, mN);
      if (ci < 0.0) { mN = -mN; ci = -ci; }
//#if BIREF
      // the wave meets the surface at its own angle, with its own index
      let iorIn = length(kv);
      ci = clamp(dot(kv, mN) / iorIn, 0.0, 1.0);
//#else
      let iorIn = ior;
//#endif
      // the facet met from inside (the wall itself, unless roughness is drawn by chance)
//#if ROUGH
      let mf = frFacet(mN, dir, b, rd, xi);
      let cf = dot(dir, mf);
//#else
      let mf = mN;
      let cf = ci;
//#endif
      // is all of it sent back in (total reflection)? That is the same for every wavelength.
//#if HOLLOW
//#if FILM
      let bare = 0.0;                       // (a film's own reflectance, by wavelength, is worked out below)
//#else
      let bare = wallR(cf);                 // a thin wall, met from inside: what it sends back
//#endif
//#else
//#if ROUGH
//#if COAT
      let bare = fresnel(cf, iorIn, 1.0);
//#else
      // a rough wall, met by a cone of light: what is sent back inside is averaged over the cone
      let fw = frWall(cf, fv, iorIn, 1.0, bend);
      // The share of what the wall sends back that has by now been scattered every way shows nothing
      // more by being followed: it is ended here, as the light from all round (so no light is lost to
      // the ending, and none stays trapped that a rough wall would let out). It is taken off before the
      // part that leaves is followed, and what the wall sends back is reckoned without it, so that
      // nothing more has to be carried meanwhile.
      let we = frEnd(fv, bend);
      acc += {{WEIGHT}} * (fw.x * we) * att * frDeep(s) * frAll(mN);
//#if FINE
//#each
      t[g] *= 1.0 - fw.x * we;
//#end
//#endif
      thr *= 1.0 - fw.x * we;
      var bare = fw.x * (1.0 - we) / max(1.0 - fw.x * we, 1e-4);
      if (bare > 1.0 - frLeak()) {
        // (so little of it leaves here that following it is not worth the march: that little is given
        // the light from all round as well)
        acc += {{WEIGHT}} * (1.0 - bare) * att * frAll(mN);
//#if FINE
//#each
        t[g] *= bare;
//#end
//#endif
        thr *= bare;
        bare = 1.0;
      }
      let fv0 = fv;
      fv = frNext(fv, bend);
//#endif
//#else
      let bare = fresnel(cf, iorIn, 1.0);
//#endif
//#endif
      // ---- where the part that stays inside goes next (worked out now, so that less has to be carried
      // while the part that leaves is followed)
//#if BIREF
      var kn = reflect(dir, mf) * ior;
      if (extra) { kn = waveE(kv - dot(kv, mf) * mf, -mf, axisW(), ior, nE); }
      var nd = normalize(kn);
      if (extra && dot(kn, kn) > 0.01) { nd = rayE(kn, axisW(), ior, nE); }
//#else
      var nd = reflect(dir, mf);
//#endif
//#if ROUGH
      if (dot(nd, mN) > 0.0) { nd = reflect(nd, mN); }
//#endif
      let np = q - mN * 0.004;
      var X = vec3f(0.0);
      if (bare < 0.999) {
        // ---- the part that leaves: on through further pieces of the body, to the lights
//#if BIREF
        // the wave keeps the part of itself that lies along the surface
        var od = normalize(kv - dot(kv, mN) * mN + sqrt(max(1.0 - dot(kv, kv) + dot(kv, mN) * dot(kv, mN), 0.0)) * mN);
//#else
        var od = refract(dir, -mN, iorIn);
//#endif
//#if ROUGH
//#if !COAT
//#if !HOLLOW
//#if !BIREF
        od = frExit(dir, mN, mf, fw, iorIn);
//#endif
//#endif
//#endif
//#endif
        // What it sees is softened by: the milky body; the facets it leaves through; and what its
        // direction inside was spread by without being sampled for it (on the way in, and at every
        // bounce since), widened as leaving the body widens any spread.
        var sg = 0.0;
//#if SSS
        sg = haze;
//#endif
//#if ROUGH
//#if COAT
        sg += fv;
//#else
//#if HOLLOW
        sg += fv;
//#else
        sg += frVout(fv0, fw, dot(od, mN), iorIn, bend);
//#endif
//#endif
//#endif
//#if FILM
        // (what the far face of the film sent back has been reflected by the film: it sees the film's sky)
        let sky = select(0.0, 1.0, b > 0);
        if (LAYERS > 0) {
          X = att * beyondM(q + mN * 0.006, od, ior, sqrt(sg), sky);
        } else {
          X = att * (seen(od, sqrt(sg)) + sky * skyDim(od));
        }
//#else
        if (LAYERS > 0) {
          X = att * beyondM(q + mN * 0.006, od, ior, sqrt(sg));
        } else {
          X = att * seen(od, sqrt(sg));
        }
//#endif
//#if HOLLOW
        X *= mu.axis.xyz;                   // through the wall again, on the way out
//#endif
      }
      // ---- each wavelength: what leaves with it, and what of it stays inside
//#if COAT
      let F1 = fresnel(cf, iorIn, mu.surf.z);
      let ccb = sqrt(max(1.0 - (iorIn * iorIn) * (1.0 - cf * cf) / (mu.surf.z * mu.surf.z), 0.0));
//#endif
//#if FINE
//#each
      {
        let l = lams(i, GROUP, xi, W);
//#if COAT
//#if FILM
        let Fx = 1.0 - (1.0 - F1) * (1.0 - filmR4(ccb, l, fd, mu.surf.z, 1.0));
//#else
        let Fx = vec4f(1.0 - (1.0 - F1) * (1.0 - fresnel(ccb, mu.surf.z, 1.0)));
//#endif
//#else
//#if FILM
        let Fx = filmR4(cf, l, fd, iorIn, 1.0);
//#else
        let Fx = vec4f(bare);
//#endif
//#endif
        acc += tint4(l, t[g] * (1.0 - Fx)) * X;
        t[g] *= Fx;
      }
//#end
      thr = max(max4(t0), max(max4(t1), max4(t2)));
//#else
//#if COAT
      let Fx = 1.0 - (1.0 - F1) * (1.0 - fresnel(ccb, mu.surf.z, 1.0));
//#else
      let Fx = bare;
//#endif
      acc += wOf(i, xi, W) * (thr * (1.0 - Fx)) * X;
      thr *= Fx;
//#endif
      if (thr < 0.02) { break; }
//#if SSS
      if (max(att.x, max(att.y, att.z)) < 0.02) { break; }      // the body has swallowed the rest
//#endif
//#if BIREF
      // (an extraordinary wave reflects as one of its own kind: it too keeps the part of itself that lies
      // along the surface; if there is none to be had the path ends)
      if (dot(kn, kn) < 0.01) { break; }
      kv = kn;
//#endif
      dir = nd;
      pos = np;
    }
  }
  // the weight of white
  var wsum = vec3f(0.0);
//#if MONO
  {
    let i = 0;
//#else
  for (var i = 0; i < MAX_NW; i++) {
    if (i >= W) { break; }
//#endif
//#if FINE
//#each
    wsum += tint4(lams(i, GROUP, xi, W), vec4f(1.0));
//#end
//#else
    wsum += wOf(i, xi, W);
//#endif
  }
//#if ROUGH
  frSeen(n, rd, xi, cosi, W);               // (what the stage that blurs the inside of rough glass is told about this surface)
//#endif
  return acc / (wsum * f32(MODES));
}

fn grazeM(p: vec3f, rd: vec3f) -> vec3f {
//#if COAT
  return env(reflect(rd, nrm(p)));
//#else
  return seen(reflect(rd, nrm(p)), spreadR());
//#endif
}
//#endif
//#endif
//#endif
//#endif
