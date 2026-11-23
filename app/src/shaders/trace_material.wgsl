
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

//#if SSS
// The diffusion volume stores reduced scattering, sigma_s' = (1-g) sigma_s.
// A straight segment reaches its next surface only if it has no collision.
// Use the true rates for that survival and for the complementary first-collision light.
// Green's unreduced rate is supplied separately; the channels retain their measured ratios.
fn collisionRates() -> vec3f {
  if (mu.sssK.r == mu.sssK.b) { return vec3f(mu.sss.x); }
  return mu.sssK.rgb * (mu.sss.x / max(mu.sssK.g, 1e-6));
}
//#endif

//#if PATH_RANDOM
// plain random numbers, as many as wanted
var<private> rState: u32;
fn rnd() -> f32 {
  rState = pcg(rState);
  return f32(rState >> 8u) / 16777216.0;
}
//#endif
//#if WALK_PHASE
// A direction drawn evenly from all directions.
fn anywhere() -> vec3f {
  let z = 1.0 - 2.0 * rnd();
  let r = sqrt(max(1.0 - z * z, 0.0));
  let ph = 6.2831853 * rnd();
  return vec3f(r * cos(ph), r * sin(ph), z);
}
fn walkFlight(rates: vec3f) -> f32 {
  var rate = rates.g;
  if (max(rates.r, rates.b) - min(rates.r, rates.b) > 1e-5 * rate) {
    rate = rates[min(i32(3.0 * rnd()), 2)];
  }
  return -log(max(1.0-rnd(),1e-6)) / max(rate,1e-6);
}
// Divide each channel's flight density (or survival probability to the actual
// interface) by the mixture used to sample it. The shared exponential cancels.
fn walkWeight(rates: vec3f, distance: f32, scatter: bool) -> vec3f {
  if (rates.r == rates.b) {return vec3f(1.0);}
  let slow = min(rates.r,min(rates.g,rates.b));
  let tr = exp(-(rates-vec3f(slow))*max(distance,0.0));
  let probability = select(tr,rates*tr,scatter);
  return probability / max(dot(probability,vec3f(1.0/3.0)),1e-20);
}
fn walkDirection(dir: vec3f) -> vec3f {
  let g=clamp(1.0-mu.sssK.g/max(mu.sss.x,1e-6),0.0,0.95);
  if (g<0.001) {return anywhere();}
  // Henyey-Greenstein inverse CDF about the previous flight direction.
  let ratio=(1.0-g*g)/(1.0-g+2.0*g*rnd());
  let c=clamp((1.0+g*g-ratio*ratio)/(2.0*g),-1.0,1.0);
  let ph=6.2831853*rnd();
  let axis=select(vec3f(0.0,0.0,1.0),vec3f(0.0,1.0,0.0),abs(dir.z)>0.9);
  let u=normalize(cross(axis,dir));
  return c*dir+sqrt(max(1.0-c*c,0.0))*(cos(ph)*u+sin(ph)*cross(dir,u));
}
//#endif

struct MU {
  surf: vec4f,        // x: roughness (spread of the surface's slopes; 0 = polished)   y: (kept for the renderer)
                      // z: the index the body sees above it (1 = air, the coat's with a full coat)
                      // w: the share of the roughness that is done by sampling (1 = all of it)
  film: vec4f,        // x: film thickness, nm   y: its index   z: how much it varies   w: size of that pattern
  film2: vec4f,       // x: where the pattern is in its flow (turns)   y: drain   zw: absorption by wavelength as red, green
  body: vec4f,        // rgb: the body's colour (a metal's tint)   w: 1 = a metal from the table, 0 = a perfect mirror
  sss: vec4f,         // x: scatterings per unit length   y: reach of diffuse light beyond the surface   z: events followed at most
                      // w: the index of a thin wall's own glass
  sssA: vec4f,        // rgb: absorption of the milky body per unit length   w: how far the scattered light takes on
                      // the body's colour at every depth (sssHue)
  sssK: vec4f,        // rgb: scatterings per unit length of each colour (sss.x is green's: they differ when the
                      // colours scatter differently)   w: how fast light crossing the body takes on its colour (sssThrough)
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
  grating: vec4f,     // efficiency, groove spacing in nm, in-plane rotation, concentric grooves
  gratingFrame: array<vec4f, 3>, // inverse placement of the owner in scene coordinates
  ggxEnergy: array<vec4f, 8>, // split-sum directional albedo, two viewing angles in each vector
  brushEnergy: array<vec4f, 24>, // directional albedo across, between, and along the polishing lines
  proposals: array<vec4f, 4>, // directions in the light rig and cap sizes (one minus cosine)
  proposalCdf: vec4f,
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

//#if GRATING
// Reflection grating in the surface's tangent plane. Groove-parallel wavevector
// is conserved. Each first order receives half the diffracted share; a closed
// order retains its share in the ordinary reflection.
fn gratingAcross(p: vec3f, n: vec3f) -> vec3f {
  let g = mu.grating;
  var local = vec3f(1.0, 0.0, 0.0);
  if (g.w > 0.5) {
    let scene = vec4f(toObj(p),1.0);
    local = vec3f(dot(mu.gratingFrame[0],scene),0.0,dot(mu.gratingFrame[2],scene));
  }
  let sceneAxis = local.x*mu.gratingFrame[0].xyz + local.y*mu.gratingFrame[1].xyz + local.z*mu.gratingFrame[2].xyz;
  let axis = vec3f(dot(sceneAxis, u.obj0.xyz), dot(sceneAxis, u.obj1.xyz), dot(sceneAxis, u.obj2.xyz));
  let projected = axis - dot(axis,n) * n;
  if (dot(projected,projected) < 1e-8) { return vec3f(0.0); }
  let a = normalize(projected);
  return cos(g.z) * a + sin(g.z) * cross(n,a);
}
fn gratingReflect(p: vec3f, n: vec3f, rd: vec3f, zero: vec3f, xi: f32) -> vec3f {
  let g = mu.grating;
  if (g.x <= 0.0) { return zero; }
  let across = gratingAcross(p,n);
  if (dot(across,across) < 1e-8) { return zero; }
  let along = cross(n,across);
  let mirror = reflect(rd,n);
  let x = dot(mirror,across);
  let y = dot(mirror,along);
  var spectral = vec3f(0.0);
  var norm = vec3f(0.0);
  for (var i = 0; i < 12; i++) {
    let lambda = 400.0 + 25.0 * (f32(i) + xi);
    let weight = specW(lambda);
    var pair = vec3f(0.0);
    for (var m = -1; m <= 1; m += 2) {
      let k = x + f32(m) * lambda / g.y;
      let h2 = 1.0 - k*k - y*y;
      if (h2 <= 0.0) { pair += zero; }
      else {
        let outgoing = k * across + y * along + sqrt(h2) * n;
//#if ROUGH
        pair += envL(outgoing, 1.41421 * mu.surf.x);
//#else
        pair += env(outgoing);
//#endif
      }
    }
    spectral += weight * (0.5 * pair);
    norm += weight;
  }
  return mix(zero, spectral / norm, g.x);
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
const T0: f32 = 0.0061359232;
fn envL(rd: vec3f, sig: f32) -> vec3f {
  let d = toEnv(rd);
  let uv = vec2f(atan2(d.z, d.x) * 0.15915494 + 0.5, acos(clamp(d.y, -1.0, 1.0)) * 0.31830989);
  let lv = clamp(select(log2(max(sig, 1e-6) / T0), sig / (2.0 * T0), sig < 2.0 * T0), 0.0, 9.0);
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

//#if PICTURE
// The picture's sun as a mirror sees it along d (in the lights' frame), spread by w radians more: a small
// bright disc holding all the light the picture's ceiling took off (material.rs: sun_numbers), as a bell
// on the sphere. (Its numbers ride in the tables' unused fourth places.)
fn sunGlint(d: vec3f, w: f32) -> vec3f {
  let e = vec3f(mu.tab[3].w, mu.tab[4].w, mu.tab[5].w);
  if (e.x + e.y + e.z <= 0.0) { return vec3f(0.0); }
  let s2 = mu.tab[6].w * mu.tab[6].w + w * w;
  let a2 = 2.0 * max(1.0 - dot(d, vec3f(mu.tab[0].w, mu.tab[1].w, mu.tab[2].w)), 0.0);
  return e * (exp(-0.5 * a2 / s2) / (6.2831853 * s2));
}

// How soft a copy of the environment picture a mirror reads at p, where the surface faces n and the
// ray met it at cosine c: the level whose softening is as wide as the patch of surroundings that one
// pixel's worth of surface mirrors. A flat mirror shows the picture pixel for pixel; a curved one
// squeezes it, by twice the surface's curvature times the distance (and more the more glancing the
// ray), and read at full sharpness the squeezed picture would sparkle from frame to frame.
// The curvature is measured from the distance field: three lookups a few voxels out across the
// surface, a third of a turn apart, give its greatest bend whichever way that lies.
fn mirrorLod(p: vec3f, n: vec3f, c: f32) -> f32 {
  return clamp(log2(mirrorWide(p, n, c) / mu.pic.y), 0.0, mu.pic.w);
}
//#endif
// (the width, radians, of the patch of surroundings one pixel's worth of the surface mirrors)
fn mirrorWide(p: vec3f, n: vec3f, c: f32) -> f32 {
  let h = 3.0 * (2.0 * HB / u.vres);
  // any two directions across n (Duff et al. 2017)
  let sg = select(-1.0, 1.0, n.z >= 0.0);
  let k = -1.0 / (sg + n.z);
  let m = n.x * n.y * k;
  let t1 = vec3f(1.0 + sg * n.x * n.x * k, sg * m, -sg * n.x);
  let t2 = vec3f(m, sg + n.y * n.y * k, -n.y);
  let d0 = mapW(p);
  let k1 = mapW(p + h * t1) - d0;
  let k2 = mapW(p + h * (-0.5 * t1 + 0.8660254 * t2)) - d0;
  let k3 = mapW(p + h * (-0.5 * t1 - 0.8660254 * t2)) - d0;
  // (each is half the bend that way times h squared; their mean is the mean bend, their scatter the rest)
  let mean = (k1 + k2 + k3) / 3.0;
  let dev = sqrt(0.6666667 * ((k1 - mean) * (k1 - mean) + (k2 - mean) * (k2 - mean) + (k3 - mean) * (k3 - mean)));
  let bend = min((abs(mean) + dev) * 2.0 / (h * h), 1.0 / h);
  let far = length(p - vec3f(0.0, 0.0, u.dist));
  let pixel = 2.0 * u.tan_fov / u.res.y;
  return pbr_reflection_width(pixel, bend, far, c);   // pbr/widen.wgsl
}
//#if PICTURE
// The surroundings as a mirror (a metal, a film, a glaze) shows them along rd: the picture itself at that level; and,
// when the family's lights are kept lit beside the picture, the lights' map. (Otherwise that map holds
// only the soft glow that stands in for the picture's sun in the glass. A mirror wants the sun itself,
// small and sharp, and the picture has it.)
fn mirrorEnv(rd: vec3f, lod: f32) -> vec3f {
  let d = toEnv(rd);
  var c = textureSampleLevel(pictureTex, pictureSmp, d, lod).rgb * mu.pic.x + sunGlint(d, mu.pic.y * exp2(lod));
  if (mu.film5.y > 0.5) {
    let uv = vec2f(atan2(d.z, d.x) * 0.15915494 + 0.5, acos(clamp(d.y, -1.0, 1.0)) * 0.31830989);
    c += textureSampleLevel(envTex, envSmp, uv, 0.0).rgb;
  }
  return c;
}
//#else
//#if ENVL
// Without a picture: the lights' map as finely as it is painted for the softened copies (four times the
// glass's own: material.rs, env_levels; the map the glass reads is too coarse for a mirror, which shows
// a studio's soft box and horizon as sharp as they are), at the level one pixel's worth of the surface
// mirrors.
fn mirrorLod(p: vec3f, n: vec3f, c: f32) -> f32 { return clamp(log2(mirrorWide(p, n, c) / T0), 0.0, 9.0); }
fn mirrorEnv(rd: vec3f, lod: f32) -> vec3f {
  let e = toEnv(rd);
  return textureSampleLevel(softTex, softSmp, vec2f(atan2(e.z, e.x) * 0.15915494 + 0.5, acos(clamp(e.y, -1.0, 1.0)) * 0.31830989), lod).rgb;
}
//#else
fn mirrorLod(p: vec3f, n: vec3f, c: f32) -> f32 { return 0.0; }
fn mirrorEnv(rd: vec3f, lod: f32) -> vec3f { return env(rd); }
//#endif
//#endif

// What a polished surface that is not glass mirrors along rd at p (facing n, met at cosine c): with an
// environment picture, the picture as sharp as one pixel's worth of the surface can show it (mirrorLod),
// not softened as the glass reads it; glass hides that softening behind what it bends, a mirror shows it
// at once. `sharp` off (mu.more.y): the glass's softened surroundings, as before.
fn sharpEnv(rd: vec3f, p: vec3f, n: vec3f, c: f32) -> vec3f {
//#if PICTURE
  if (mu.more.y > 0.5) { return mirrorEnv(rd, mirrorLod(p, n, c)); }
//#else
//#if ENVL
  if (mu.more.y > 0.5) { return mirrorEnv(rd, mirrorLod(p, n, c)); }
//#endif
//#endif
  return env(rd);
}

//#if WEAR
// ---- a used surface: worn edges, dirty recesses, scratches, swirls, peel, smudges, dust, lines ----------
// What makes a photographed object look used (docs/imperfections.md): marks that follow its form and its
// handling, seen almost only where a highlight crosses them. All of it is fixed to the object (worked out
// from where the point lies in the shape's own frame), nothing is random from frame to frame, and detail
// finer than a pixel is not drawn but folded into a wider spread, so one frame is clean and nothing swims.
//   mu.tab[7..13].w   the marks (times how used the surface is)   mu.tab[14..16].w   the recesses' colour
//   mu.more.w         brush: how strong the polishing lines are (round the shape's vertical)
//   mu.sssC.w         mottle: how uneven the colour is
// the shape's own frame -> the world (toObj turned back)
// Wear lives in the owning layer's coordinates, even when the assembly turns around it.
fn wearLocal(value: vec3f, position: f32) -> vec3f {
  let v = vec4f(toObj(value),position);
  return vec3f(dot(mu.gratingFrame[0],v),dot(mu.gratingFrame[1],v),dot(mu.gratingFrame[2],v));
}
fn wearFromObj(v: vec3f) -> vec3f {
  let invSize = length(mu.gratingFrame[0].xyz);
  let scene = (v.x*mu.gratingFrame[0].xyz+v.y*mu.gratingFrame[1].xyz+v.z*mu.gratingFrame[2].xyz)/invSize;
  return vec3f(dot(u.obj0.xyz,scene),dot(u.obj1.xyz,scene),dot(u.obj2.xyz,scene));
}

// Change between object coordinates and the polishing frame. An angle of zero is the original vertical axis.
fn brushCoordinates(q: vec3f, sine: f32) -> vec3f {
  let cosine = 1.0 + mu.tab[18].w;
  return vec3f(cosine*q.x - sine*q.y, sine*q.x + cosine*q.y, q.z);
}

struct Wear {
  n: vec3f,       // the surface's facing, with the marks under the point
  sig: f32,       // how much more the reflection is spread here (radians)
  keep: f32,      // the share of the reflection kept
  b: vec3f,       // across the polishing lines (in the surface)
  aniso: f32,     // how far the lines tilt the surface across themselves, all the lines in a pixel together (radians)
  tint: vec3f,    // what the reflection's colour is multiplied by (uneven colour; dirt or tarnish in recesses)
  dust: f32,      // the share of the surface under matte dust
  scr: f32,       // how much of the pixel lies on a scratch
  sx: vec3f,      // across that scratch (in the surface)
}
// The marks, each already times how used the surface is (material.rs: marks, MU_MARKS).
fn mEdge() -> f32 { return mu.tab[7].w; }
fn mCavity() -> f32 { return mu.tab[8].w; }
fn mScratch() -> f32 { return mu.tab[9].w; }
fn mSwirl() -> f32 { return mu.tab[10].w; }
fn mPeel() -> f32 { return mu.tab[11].w; }
fn mSmudge() -> f32 { return mu.tab[12].w; }
fn mDust() -> f32 { return mu.tab[13].w; }

// Which way a cell's scratch runs: mostly along one of two directions of use (rubbing to and fro, and
// across that), a little off them; in the surface.
fn scratchDir(c: vec3i, nq: vec3f) -> vec3f {
  let two = select(vec3f(1.0, 0.15, 0.3), vec3f(-0.25, 0.3, 1.0), wHash(c + vec3i(12, 7, 3)) > 0.2);
  var d = normalize(two) + 0.25 * vec3f(wHash(c + vec3i(5, 5, 1)), wHash(c + vec3i(2, 8, 6)), wHash(c + vec3i(9, 2, 7)));
  d = d - nq * dot(d, nq);
  let ld = length(d);
  if (ld < 1e-3) { return vec3f(0.0); }
  return d / ld;
}

// One cell's scratch, met at q (shape's frame) on a surface facing nq: a straight segment of random
// length and direction through a random point of the cell, lying in the surface. Returns how strongly
// the point lies on it (1 on its middle line, falling off over its width, thinned to its share of the
// pixel when it is narrower than one) and which way across it the point lies.
fn scratchIn(c: vec3i, q: vec3f, nq: vec3f, size: f32, px: f32, keepIt: f32) -> vec2f {
  if (0.5 + 0.5 * wHash(c + vec3i(17, 3, 5)) > keepIt) { return vec2f(0.0); }
  let o = (vec3f(c) + 0.5 + 0.45 * vec3f(wHash(c), wHash(c + vec3i(7, 1, 2)), wHash(c + vec3i(3, 9, 4)))) * size;
  var d = scratchDir(c, nq);
  if (dot(d, d) < 0.5) { return vec2f(0.0); }
  // (short: most a fraction of the cell, a few longer)
  let u4 = 0.5 + 0.5 * wHash(c + vec3i(4, 4, 4));
  let half = size * (0.06 + 0.34 * u4 * u4 * u4);
  let r = q - o;
  let along = dot(r, d);
  if (abs(along) > half) { return vec2f(0.0); }
  let across = cross(nq, d);
  // (gently curved: the line bows a little across itself along its length)
  let off = dot(r, across) - 0.15 * wHash(c + vec3i(8, 3, 2)) * (along * along / max(half, 1e-4));
  let width = 0.0006 + 0.0006 * (0.5 + 0.5 * wHash(c + vec3i(6, 1, 9)));
  let w = max(width, px);
  // (tapered at the ends, as a scratch is)
  let taper = 1.0 - smoothstep(0.6 * half, half, abs(along));
  return vec2f(exp(-(off * off) / (w * w)) * (width / w) * taper, sign(off));
}

// The marks at p (world), where the surface faces n. What a photograph shows of a used object follows its
// form: the high points and edges are rubbed brighter, the recesses hold dirt and are duller, scratches
// gather where the surface is exposed, and almost all of it is seen only where a highlight crosses it. So
// the marks change the reflection's spread and the surface's fine slope, the colour only in recesses. The
// form is read from the distance field: how the surface bends (three lookups across it) and how hemmed
// in it is (three along the normal, as clay's shadow is).
fn wearAt(p: vec3f, n: vec3f) -> Wear {
  var w: Wear;
  w.n = n;
  w.sig = mu.tab[30].w;                     // (a ceramic's glaze haze: 0 for everything else)
  w.keep = 1.0;
  w.b = vec3f(0.0);
  w.aniso = 0.0;
  w.tint = vec3f(1.0);
  w.dust = 0.0;
  w.scr = 0.0;
  w.sx = vec3f(0.0);
  let invSize = length(mu.gratingFrame[0].xyz);
  let q = wearLocal(p,1.0);
  let nq = wearLocal(n,0.0)/invSize;
  // the width of a pixel at p (as mirrorLod reckons it)
  let px = invSize*length(p - vec3f(0.0, 0.0, u.dist)) * 2.0 * u.tan_fov / u.res.y;
  let edgeA = mEdge();
  let cavA = mCavity();
  let scrA = mScratch();
  let swA = mSwirl();
  let peelA = mPeel();
  let smA = mSmudge();
  let dustA = mDust();
  // ---- the form: convex (worn) or concave and hemmed in (a recess)
  var convex = 0.0;
  var shut = 0.0;
  if (edgeA + cavA + scrA + smA + dustA > 0.0) {
    let h = 4.0 * (2.0 * HB / u.vres);
    let form = wearForm(p, n, h);
    convex = form.x;
    shut = form.y;
  }
  let exposed = clamp(0.5 + 0.5 * convex - 0.6 * shut, 0.0, 1.0);
  let recess = clamp(0.6 * shut + max(-convex, 0.0) * 0.6, 0.0, 1.0);
  // ---- 1. worn edges: rubbed smoother, brighter (the lines and haze rubbed away there too, below)
  let worn = edgeA * smoothstep(0.55, 0.95, exposed);
  // ---- 2. recesses: duller, darker, taking the colour of what they hold
  let dirt = cavA * recess;
  w.sig += 0.12 * dirt;
  w.keep *= 1.0 - 0.45 * dirt;
  // ---- 3. scratches: straight, hair-fine, of every length, more where the surface is exposed; those
  // narrower than a pixel are thinned to their share of it, and what they would all add up to across
  // the pixel is a faint widening of the reflection
  if (scrA > 0.0) {
    let keepIt = scrA * (0.25 + 0.75 * exposed) * (1.0 - 0.8 * worn);
    var on = 0.0;
    var across = vec3f(0.0);
    for (var s = 0; s < 2; s++) {
      let size = select(0.06, 0.16, s == 1);
      let base = vec3i(floor(q / size));
      for (var j = 0; j < 27; j++) {
        let c = base + vec3i(j % 3 - 1, (j / 3) % 3 - 1, j / 9 - 1) + vec3i(s * 1000);
        let hit = scratchIn(c - vec3i(s * 1000), q, nq, size, px, keepIt);
        if (hit.x > on) {
          on = hit.x;
          across = cross(nq, scratchDir(c - vec3i(s * 1000), nq));
        }
      }
    }
    // A scratch's walls face every way across it: it mirrors a fan of directions across itself, so it
    // shows as a faint bright line wherever that fan takes in a light, and as nothing on an even surface.
    w.scr = 0.5 * scrA * on;
    w.sx = wearFromObj(across);
    w.sig += scrA * keepIt * 0.006;
  }
  // ---- 4. polishing swirls: rings round a few pad centres, seen as arcs round a sharp highlight
  if (swA > 0.0) {
    let ring = polishingPads(q, nq, px);
    w.n = normalize(w.n + wearFromObj(ring) * (0.03 * swA * (1.0 - worn)));
    w.sig += 0.004 * swA;
  }
  // ---- 5. orange peel: the gentle wave of a sprayed or dipped coat
  if (peelA > 0.0) {
    let k1 = q * 9.0;
    let k2 = q * 23.0;
    var t = vec3f(wNoise(k1), wNoise(k1 + vec3f(17.1, 3.3, 9.7)), wNoise(k1 + vec3f(5.9, 23.1, 1.3)))
          + 0.5 * vec3f(wNoise(k2 + vec3f(2.2, 8.1, 4.4)), wNoise(k2 + vec3f(13.5, 1.7, 6.2)), wNoise(k2 + vec3f(7.3, 11.9, 3.1)));
    t -= nq * dot(t, nq);
    w.n = normalize(w.n + wearFromObj(t) * (0.006 * peelA));
  }
  // ---- 6. smudges and finger marks: a few patches where the object is handled (its exposed parts),
  // duller, with the whorl of a fingertip in them
  if (smA > 0.0) {
    var sm = fingerMarks(q, nq, px, smA);
    sm *= smA * (0.3 + 0.7 * exposed);
    w.sig += 0.08 * sm;
    w.keep *= 1.0 - 0.12 * sm;
  }
  // ---- 7. dust on faces that look up, less in recesses' shelter: matte specks
  if (dustA > 0.0) {
    let up = max(dot(nq, vec3f(0.0, 1.0, 0.0)), 0.0);
    let specks = 0.5 + 0.5 * wNoise(q * 140.0);
    let fine = 1.0 - smoothstep(0.3, 1.0, px * 140.0);
    w.dust = dustA * up * up * (1.0 - 0.6 * shut) * mix(0.35, smoothstep(0.55, 0.8, specks), fine) * 0.35;
  }
  // ---- the polishing lines: round the shape's vertical (along latitude on a ball, round a cylinder)
  let br = mu.more.w * (1.0 - 0.7 * worn);
  if (br > 0.0) {
    let sine = mu.tab[17].w;
    let brushedPoint = brushCoordinates(q, sine);
    let brushedNormal = brushCoordinates(nq, sine);
    var bq = vec3f(0.0, 1.0, 0.0) - brushedNormal * brushedNormal.y;
    let lb = length(bq);
    bq = mix(vec3f(1.0, 0.0, 0.0) - brushedNormal * brushedNormal.x, bq, smoothstep(0.1, 0.3, lb));
    w.b = wearFromObj(brushCoordinates(normalize(bq + vec3f(1e-6)), -sine));
    let y = brushedPoint.y + 0.004 * wNoise(brushedPoint * 3.0);
    let r = vec3f(brushedPoint.x * 9.0, brushedPoint.z * 9.0, 0.0);
    let fine = 1.0 - smoothstep(0.3, 1.0, px * 380.0);
    let coarse = 1.0 - smoothstep(0.3, 1.0, px * 110.0);
    let line = wNoise(vec3f(r.x, y * 380.0, r.y)) * fine + 0.6 * wNoise(vec3f(r.x * 0.4 + 7.0, y * 110.0, r.y * 0.4 + 3.0)) * coarse;
    w.n = normalize(w.n + w.b * (0.09 * br * line));
    w.aniso = br * (0.03 + 0.05 * (1.0 - 0.5 * (fine + coarse)));
  }
  // worn high points: smoother, and the spread above taken back there
  w.sig *= 1.0 - 0.8 * worn;
  w.keep = mix(w.keep, 1.0, worn);
  w.tint = wearPigment(q, dirt, vec3f(mu.tab[14].w, mu.tab[15].w, mu.tab[16].w), mu.sssC.w);
  return w;
}

// The surroundings a worn surface mirrors along rd: as sharpEnv does, spread a further sig (radians);
// and across the polishing lines (b, aniso), drawn out: four looks along the line across them.
fn wearRead(rd: vec3f, lod: f32, sig: f32) -> vec3f {
//#if PICTURE
  if (mu.more.y > 0.5) {
    let d = toEnv(rd);
    let l = max(lod, log2(max(sig, 1e-6) / mu.pic.y));
    var c = textureSampleLevel(pictureTex, pictureSmp, d, l).rgb * mu.pic.x + sunGlint(d, max(sig, mu.pic.y * exp2(lod)));
    if (mu.film5.y > 0.5) {
      let uv = vec2f(atan2(d.z, d.x) * 0.15915494 + 0.5, acos(clamp(d.y, -1.0, 1.0)) * 0.31830989);
      c += textureSampleLevel(softTex, softSmp, uv, clamp(select(log2(max(sig, 1e-6) / T0), sig / (2.0 * T0), sig < 2.0 * T0), 0.0, 9.0)).rgb;
    }
    return c; // worn surface panorama
  }
//#endif
  return envL(rd, sig);
}
fn wearSee(rd_in: vec3f, w: Wear, p: vec3f, c: f32) -> vec3f {
  // (dust: matte specks, which show the light from all round the way they face, not a reflection)
  let dusted = select(vec3f(0.0), w.dust * 0.6 * wearRead(w.n, 0.0, 0.9), w.dust > 0.0);
  return (1.0 - w.dust) * wearMirror(rd_in, w, p, c) + dusted;
}
fn wearMirror(rd_in: vec3f, w: Wear, p: vec3f, c: f32) -> vec3f {
  let base = wearLines(rd_in, w, p, c);
  if (w.scr <= 0.0) { return base; }
  // the scratch's fan: four looks across it, the brightest of what they see brightens the line
  var fan = vec3f(0.0);
  for (var k = 0; k < 4; k++) {
    let e = (f32(k) - 1.5) * 0.25;
    fan = max(fan, wearRead(reflect(rd_in, normalize(w.n + w.sx * e)), 2.0, 0.08));
  }
  return base + w.scr * max(fan - base, vec3f(0.0));
}
fn wearLines(rd_in: vec3f, w: Wear, p: vec3f, c: f32) -> vec3f {
  let lod = mirrorLod(p, w.n, c);
  let mir = reflect(rd_in, w.n);
  if (w.aniso <= 0.0) { return wearRead(mir, lod, w.sig); }
  // (a tilt of the surface by e turns the reflection by 2e; each look stands for a quarter of the spread)
  // (six looks, each softened by a third of the spread across the lines: a drawn-out highlight with a
  // defined core, not a fog)
  let s = sqrt(w.sig * w.sig + 0.11 * w.aniso * w.aniso);
  var c6 = vec3f(0.0);
  for (var k = 0; k < 6; k++) {
    let e = (f32(k) - 2.5) * 0.4 * w.aniso;
    c6 += wearRead(reflect(rd_in, normalize(w.n + w.b * e)), lod, s);
  }
  return c6 / 6.0;
}

// ---- a brushed metal (r4 metal): anisotropic GGX across the polishing lines -----------------------------
// Brushing cuts fine grooves along the lines: the surface's slopes are spread across them (b) far more
// than along them, so a light's reflection is drawn out into a streak across the lines, with GGX's long
// glow around it, and line by line the finish differs (some rubbed finer than others), which shows as
// streaks along the lines wherever the highlight lies. Not averaged over a few fixed looks (they show as
// ghost copies once the spread is wide): one facet of that spread per frame, from a sequence that fills
// the square evenly frame by frame, and the reflection off it is followed like any other (to the next
// surface of the shape, or to the lights): so the shape hides the lights from it where it should, and the
// frames add up to the whole lobe.
const BR_ACROSS: f32 = 0.15;     // the slopes' spread across the lines (GGX alpha) at brush 1
const BR_ALONG: f32 = 0.006;     // and along them
fn brushRnd(fc: vec2f, b: i32) -> vec2f {
  // (each pixel's own start, from a hash: interleaved gradient noise here leaves a cross-hatch while
  // few frames are kept)
  let i = vec3i(vec2i(fc), 31 + b);
  let o = 0.5 + 0.5 * vec2f(wHash(i), wHash(i + vec3i(0, 0, 977)));
  return fract(o + vec2f(0.7548777, 0.5698403) * f32(u.frame % 4096));
}
// The facet this frame's reflection leaves by, about w.n; `wide`: a spread both ways already there (a
// rough metal's own, radians of slope), part of which the lines take across themselves.
// sg: 1, or -1 for the facet opposite (the pair is read together: half the grain for one more look)
// Returns the facet (xyz) and how far its reflection is read softened (w, radians): more the farther out
// in the tail the facet lies, where the lobe is thin and one look stands for a wide piece of it.
fn brushStreak(p: vec3f) -> f32 {
  let q = brushCoordinates(wearLocal(p,1.0), mu.tab[17].w);
  // (line by line, and in a few broader bands, the grooves are deeper or shallower)
  // (lines finer than a pixel are not drawn, their mean is: as wearAt does)
  let px = length(mu.gratingFrame[0].xyz)*length(p - vec3f(0.0, 0.0, u.dist)) * 2.0 * u.tan_fov / u.res.y;
  let fine = 1.0 - smoothstep(0.3, 1.0, px * 150.0);
  return max(0.6 + 0.4 * fine * wNoise(vec3f(q.x * 3.0, q.y * 150.0, q.z * 3.0)) + 0.3 * wNoise(vec3f(q.x * 1.5 + 5.0, q.y * 31.0, q.z * 1.5 + 1.0)), 0.15);
}
// The sampled solo and assembly closures use the same tangent frame and slope widths.
struct BrushSurface { across: vec3f, slopes: vec2f }
fn brushSurface(w: Wear, p:vec3f, alpha:f32) -> BrushSurface {
  if (w.aniso <= 0.0) { return BrushSurface(vec3f(0.0),vec2f(alpha)); }
  let across = normalize(w.b-w.n*dot(w.b,w.n));
  let brush = clamp(mu.more.w,0.0,1.0);
  let aspect = sqrt(1.0-0.60*brush);
  let base = vec2f(alpha/aspect,alpha*aspect);
  let groove = brush*vec2f(BR_ACROSS*brushStreak(p),BR_ALONG);
  return BrushSurface(across,sqrt(base*base+groove*groove));
}
fn brushN(w: Wear, p: vec3f, rd: vec3f, fc: vec2f, b: i32, wide: f32, sg: f32) -> vec4f {
  let br = min(mu.more.w, 1.5);
  let streak = brushStreak(p);
  let ax = br * BR_ACROSS * streak + wide * (1.0 + 1.5 * br) * (0.7 + 0.3 * streak);
  let ay = BR_ALONG + 0.5 * w.sig + wide * (1.0 - 0.6 * min(br, 1.0));
  let r = brushRnd(fc, b);
  // GGX's slopes: radius sqrt(u / (1 - u)) (the farthest few hundredths of its tail left out: fireflies)
  let u1 = min(r.x, 0.86);
  let s = sqrt(u1 / (1.0 - u1));
  let ph = 6.2831853 * r.y;
  let s2 = sg * s;
  let t = cross(w.n, w.b);
  let m = normalize(w.n + w.b * (ax * s2 * cos(ph)) + t * (ay * s2 * sin(ph)));
  // (a facet that would send the light into the surface is taken as the mean one)
  let soft = 0.35 * ax * s;
  if (dot(reflect(rd, m), w.n) <= 0.02) { return vec4f(w.n, soft); }
  return vec4f(m, soft);
}
//#endif

//#if RANDOM
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
// Power reaching a smooth conductor and escaping back through its clear coat.
// Both interfaces are parallel. Each internal round trip repeats the same angle
// and polarisation, so its geometric series can be integrated exactly.
fn coatEscape(f: f32, r: vec3f) -> vec3f {
  let t = 1.0 - f;
  return r * (t * t) / max(vec3f(1e-8), 1.0 - f * r);
}
fn polishedCoatReturn(c: f32, cc: f32, index: f32, top: vec3f) -> vec3f {
  let sa = (c - index * cc) / max(c + index * cc, 1e-7);
  let pa = (index * c - cc) / max(index * c + cc, 1e-7);
  let fs = sa * sa;
  let fp = pa * pa;
  let tint = mix(mu.body.rgb, vec3f(1.0), pow(1.0 - cc, 5.0));
  if (mu.body.w <= 0.5) {
    return 0.5 * (coatEscape(fs, tint) + coatEscape(fp, tint));
  }
  var returned = vec3f(0.0);
  var sensor = vec3f(0.0);
  for (var j = 0; j < 16; j++) {
    let nk = mu.tab[2 * j].xy;
    let metal = fresMetalPolar(cc, index, nk.x, nk.y);
    let response = specW(400.0 + 20.0 * f32(j));
    returned += response * (0.5 * (coatEscape(fs, tint * metal.x) + coatEscape(fp, tint * metal.y)));
    sensor += response;
  }
  return clamp(returned / sensor, vec3f(0.0), 1.0 - top);
}

// Single and multiple reflection energy after Fdez-Agüera (2019), Eq. 18.
// Multiple surface encounters are spread across the hemisphere; their tint includes absorption on each hit.
struct ConductorEnergy { single: vec3f, multiple: vec3f }
fn conductorAb(i: u32) -> vec2f {
  let v = mu.ggxEnergy[i / 2u];
  return select(v.xy, v.zw, (i & 1u) != 0u);
}
fn frMetalRig(p: vec3f, n: vec3f, rd: vec3f, mid: vec3f, var0: f32, bounce: i32,
              alpha: f32, coated: bool, incident: vec3f) -> vec3f {
  let x = pbr_energy_x(-dot(rd, n));
  let i = u32(x);
  let ab = mix(conductorAb(i), conductorAb(min(i+1u,15u)), fract(x));
  let geometric = max(ab.x + ab.y, 0.01);
  return incident; // finite GGX light
}
fn conductorEnergy(c: f32, medium: f32, smoothR: vec3f) -> ConductorEnergy {
  let x = pbr_energy_x(c);
  let i = u32(x);
  let ab = mix(conductorAb(i), conductorAb(min(i+1u,15u)), fract(x));
  return conductorEnergyFromAb(ab,medium,smoothR);
}
fn brushedAb(angle:u32, azimuth:u32) -> vec2f {
  let v = mu.brushEnergy[azimuth*8u + angle/2u];
  return select(v.xy,v.zw,(angle & 1u) != 0u);
}
fn brushConductorEnergy(c:f32, medium:f32, smoothR:vec3f, across:vec3f, view:vec3f, normal:vec3f) -> ConductorEnergy {
  let projected = view-normal*dot(normal,view);
  let ratio = clamp(dot(across,projected)*dot(across,projected)/max(dot(projected,projected),1e-12),0.0,1.0);
  let azimuth = 2.0*(1.0-ratio);
  let a = min(u32(azimuth),1u);
  let x = pbr_energy_x(c);
  let i = u32(x);
  let j = min(i+1u,15u);
  let low = mix(brushedAb(i,a),brushedAb(j,a),fract(x));
  let high = mix(brushedAb(i,a+1u),brushedAb(j,a+1u),fract(x));
  return conductorEnergyFromAb(mix(low,high,azimuth-f32(a)),medium,smoothR);
}
fn conductorEnergyFromAb(ab:vec2f, medium:f32, smoothR:vec3f) -> ConductorEnergy {
  let f0 = clamp(metalRGB(1.0, medium, 0.0), vec3f(0.0), vec3f(1.0));
  let e = pbr_energy_split(ab, f0);   // pbr/energy.wgsl
  let blend = smoothstep(0.0, 0.08, mu.surf.x);
  return ConductorEnergy(mix(smoothR, e.single, blend), blend * e.multiple);
}


//#endif

//#if SSS
// What is left of the colour of light that crosses a stretch s of the body without being scattered away:
// it takes on the body's colour, deeper the further it goes, so thin parts and edges lit from behind glow
// amber in wax. Looks first: the grains a real body's colour comes from also tint what passes them by
// (docs/materials.md section 14). 1 when the material does not ask for it.
fn throughHue(s: f32) -> vec3f {
  if (mu.sssK.w <= 0.0) { return vec3f(1.0); }
  let c = mu.sssC.rgb / max(max(mu.sssC.r, mu.sssC.g), max(mu.sssC.b, 1e-3));
  return exp(log(max(c, vec3f(1e-3))) * (mu.sssK.w * sqrt(max(s, 0.0))));
}
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
  let sigmaS = collisionRates();
  let st = sigmaS + mu.sssA.rgb + u.absorb.xyz;
  let k = min(st.r, min(st.g, st.b));
  let all = 1.0 - exp(-k * s);
  var sum = vec3f(0.0);
  let deepen = mu.sssK.w > 0.0;
  var transmit = vec3f(1.0);
  for (var j = 0; j < GLOW_N; j++) {
    let t = -log(1.0 - (f32(j) + 0.5) / f32(GLOW_N) * all) / k;
    let x = toObj(pos + dir * t) / (2.0 * HB) + 0.5;
    // Integrate each colour's attenuation exactly over this section. A channel
    // whose mean free path is short may give up most of its light in the first.
    let edge = select(-log(max(1.0-(f32(j)+1.0)*all/f32(GLOW_N),1e-10))/k,s,j+1==GLOW_N);
    let next = exp(-st * edge);
    var g = glowAt(x) * (transmit - next);
    transmit = next;
    // (light crossing the body takes on its colour, deeper the further it goes: the light here has come
    // at least as far in as the point lies deep, and goes on t more to the eye)
    if (deepen) { g *= throughHue(t + max(-mapW(pos + dir * t), 0.0)); }
    sum += g;
  }
  // (a dyed body: the light scattered towards the eye takes on the body's colour already just under the
  // surface, kept as bright as it was; looks first, see docs/materials.md section 14)
  let body = mu.sssC.rgb / max(max(mu.sssC.r, mu.sssC.g), max(mu.sssC.b, 1e-3));
  let dye = mix(vec3f(1.0), body / max(dot(body, vec3f(0.2126, 0.7152, 0.0722)), 1e-3), mu.sssA.w);
  return sum * dye * (sigmaS / max(st,vec3f(1e-6))) * (0.0795775 / (u.ior * u.ior));
}
// Multi-scale ambient occlusion in crevices and folds for matte/opaque subsurface wax.
fn waxOcclusion(p: vec3f, n: vec3f) -> f32 {
  var shut = 0.0;
  var weight = 1.0;
  for (var i = 0; i < 5; i++) {
    let h = 0.012 + 0.052 * f32(i);
    shut += weight * max(h - mapW(p + n * h), 0.0) / h;
    weight *= 0.8;
  }
  return clamp(1.0 - 0.46 * shut, 0.0, 1.0);
}
// Subsurface optical depth for opaque-exit wax: thin tips and rims transmit warmly according
// to sssDepth, while thick bulk cores (>1 unit) attenuate to solid sculptural wax.
fn waxTau(s: f32) -> f32 {
  return (1.55 * mu.sss.x + 1.45) * max(s, 0.0);
}
// Chromatic Beer-Lambert + Rayleigh crystallite enrichment along transmitted wax paths.
fn waxThroughTint(tau: f32) -> vec3f {
  let c = clamp(mu.sssC.rgb, vec3f(0.015), vec3f(0.992));
  let peak = max(c.r, max(c.g, max(c.b, 0.02)));
  let chroma = clamp(c / peak, vec3f(0.02), vec3f(1.0));
  let spread = 0.35 + 0.65 * clamp(1.0 - mu.sssK.b / max(mu.sssK.r, 1e-6), 0.0, 1.0);
  let warmShift = vec3f(0.0, 0.08, 0.22) * spread * sqrt(max(tau, 0.0));
  let chromaPow = 1.35 + (1.10 + 0.55 * clamp(mu.sssK.w, 0.0, 1.5)) * sqrt(max(tau, 0.0));
  let amp = pow(peak, 0.55 + 0.35 * min(tau, 3.0));
  return amp * exp(log(chroma) * chromaPow - warmShift);
}
// Smooth wide-cone SDF self-occlusion behind a diffuse wax exit point: blocks solid adjacent cores
// seamlessly without projecting sharp X-ray silhouettes of distant thin background geometry.
fn waxBackCone(p: vec3f, d: vec3f) -> f32 {
  var open = 1.0;
  var t = 0.014;
  let bound = HB * (1.0 - 1.5 / u.vres);
  for (var i = 0; i < 7; i++) {
    let pt = p + d * t;
    if (any(abs(toObj(pt)) >= vec3f(bound))) { break; }
    let r = 0.88 * t + 0.010;
    let pen = -mapW(pt) / r;
    open *= 1.0 - smoothstep(0.03, 0.78, pen);
    t *= 1.85;
  }
  return open;
}
// Continuous thin-feature openness in 0..1 at back-wall exit q: 1 at genuine thin tips/rims,
// smoothly decaying to 0 before grazing concave V-creases backed by solid wax core.
fn waxThinOpen(q: vec3f, dir: vec3f) -> f32 {
  var m = nrm(q);
  let ci = abs(dot(dir, m));
  if (dot(dir, m) < 0.0) { m = -m; }
  let grazeFade = smoothstep(0.03, 0.36, ci);
  if (grazeFade <= 0.0) { return 0.0; }
  let dFwd = normalize(mix(dir, m, 0.14));
  return grazeFade * waxBackCone(q + m * 0.006, dFwd);
}
// Diffuse + forward-scattered radiance entering the back face (q, mN) of an opaque wax wedge.
fn waxExit(q: vec3f, mN: vec3f, dir: vec3f) -> vec3f {
  let ci = clamp(dot(dir, mN), 0.0, 1.0);
  let grazeFade = smoothstep(0.03, 0.36, ci);
  if (grazeFade <= 0.0) { return vec3f(0.0); }
  let dFwd = normalize(mix(dir, mN, 0.14));
  let openFwd = waxBackCone(q + mN * 0.006, dFwd);
  if (openFwd <= 0.001) { return vec3f(0.0); }
  let dNorm = normalize(mix(mN, dir, 0.25));
  let open = grazeFade * openFwd * (0.58 + 0.42 * waxBackCone(q + mN * 0.010, dNorm));
  let dSky = normalize(vec3f(dFwd.x, max(dFwd.y, 0.06 * abs(dFwd.z)), dFwd.z));
  let backlit = 0.58 * envL(dSky, 0.38) + 0.42 * envL(mN, 0.75);
  return open * backlit;
}
// Directional + wrapped hemisphere irradiance on a waxy facet facing n.
fn waxHemi(n: vec3f) -> vec3f {
  return 0.65 * envL(n, 0.38) + 0.35 * envL(n, 0.95);
}
// Near-surface waxy scattering at entry (p, n): crisp 3D facet shading and warm valley bounce,
// independent of ray-exit chord s so concave V-creases stay C1-smooth.
fn waxNear(n: vec3f, ao: f32, s: f32) -> vec3f {
  let c = pow(clamp(mu.sssC.rgb, vec3f(0.0), vec3f(1.0)), vec3f(1.22));
  let bounce = mix(0.32 * c, vec3f(1.0), ao);
  return 0.33 * waxHemi(n) * c * bounce;
}
// Calibrated 3D diffusion-volume scatter along stretch s for opaque wax, de-biased from any
// residual glass tint (u.absorb) and bridged smoothly across concave creases backed by solid core.
fn waxVolume(pos: vec3f, dir: vec3f, s_in: f32, n: vec3f, ao: f32, hasNormal: bool) -> vec3f {
  let sRaw = max(s_in, 0.0);
  var s = sRaw;
  if (hasNormal && sRaw < 0.95) {
    let openThin = waxThinOpen(pos + dir * sRaw, dir);
    s = mix(0.95, sRaw, openThin);
  }
  let sigmaS = collisionRates();
  let st = sigmaS + mu.sssA.rgb + u.absorb.xyz;
  let stWax = max(sigmaS + mu.sssA.rgb, vec3f(1.55 * mu.sss.x + 2.60));
  let k = max(min(stWax.r, min(stWax.g, stWax.b)), 1e-4);
  let all = 1.0 - exp(-k * s);
  var sum = vec3f(0.0);
  var transmit = vec3f(1.0);
  for (var j = 0; j < GLOW_N; j++) {
    let t = -log(max(1.0 - (f32(j) + 0.5) / f32(GLOW_N) * all, 1e-10)) / k;
    let x = toObj(pos + dir * t) / (2.0 * HB) + 0.5;
    let edge = select(-log(max(1.0 - (f32(j) + 1.0) * all / f32(GLOW_N), 1e-10)) / k, s, j + 1 == GLOW_N);
    let next = exp(-stWax * edge);
    sum += glowAt(x) * (transmit - next);
    transmit = next;
  }
  let c = clamp(mu.sssC.rgb, vec3f(0.01), vec3f(0.995));
  let ssAlbedo = sigmaS / max(st, vec3f(1e-6));
  let bakedTint = clamp(ssAlbedo * exp(-(mu.sssA.rgb + u.absorb.xyz) * 0.65), vec3f(0.14), vec3f(1.0));
  let waxPigment = pow(c, vec3f(1.30)) * (ssAlbedo / bakedTint);
  let vol = sum * waxPigment * (0.0795775 / (u.ior * u.ior));
  let bounce = mix(0.34 * c, vec3f(1.0), ao);
  var shadeMod = 1.0;
  if (hasNormal) {
    let lumFront = dot(waxHemi(n), vec3f(0.2126, 0.7152, 0.0722));
    let lumAvg = max(dot(envL(n, 1.45), vec3f(0.2126, 0.7152, 0.0722)), 1e-3);
    let ratio = clamp(lumFront / lumAvg, 0.25, 1.35);
    let thinShare = exp(-2.0 * s);
    shadeMod = mix(0.32 + 0.60 * ratio, 1.0, thinShare);
  }
  return 0.26 * vol * bounce * shadeMod;
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
  return F * (mirrorEnv(reflect(rd, n), mirrorLod(p, n, c)) + skyOf(reflect(rd, n))) + (1.0 - F) * clay(p, rd);
//#endif
//#else
  return F * seen(reflect(rd, n), spreadR()) + (1.0 - F) * clay(p, rd);
//#endif
}
//#else
// ---- a mirror or a metal -------------------------------------------------------------------------------
//#if METAL
//#if !BODIES
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
//#if WEAR
    var wr = wearAt(p, n);
    let brushed = wr.aniso > 0.0;
//#endif
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
//#else
    let Fc = vec3f(fresnel(cosi, 1.0, nx));
//#endif
//#if ROUGH
    col += thr * Fc * sharpEnv(mir, p, n, cosi);
    var R = metalRGB(cc, nx, 0.0);
    thr *= 1.0 - Fc;
//#else
//#if FILM
    col += thr * Fc * sharpEnv(mir, p, n, cosi);
    var R = (1.0 - Fc) * metalRGB(cc, nx, 0.0);
//#else
    let returned = polishedCoatReturn(cosi, cc, nx, Fc);
//#if WEAR
    col += thr * Fc * sharpEnv(mir, p, n, cosi);
    var R = returned;
//#else
    // Both polished interfaces send light along the same ray, including a
    // second encounter with the shape before that ray reaches the room.
    var R = Fc + returned;
//#endif
//#endif
//#endif
//#else
    var R = metalRGB(cosi, 1.0, fd);
//#endif
//#if ROUGH
    var hiddenR = R;
    if (fd <= 0.0) {
//#if COAT
      let cosine = cc;
      let medium = nx;
//#else
      let cosine = cosi;
      let medium = 1.0;
//#endif
      var energy = conductorEnergy(cosine, medium, R);
//#if ROUGH_REF
//#if WEAR
      if (brushed) {
        let across = normalize(wr.b-wr.n*dot(wr.b,wr.n));
        energy = brushConductorEnergy(cosine,medium,R,across,-rd,wr.n);
      }
//#endif
//#endif
      R = energy.single;
      hiddenR = energy.single + energy.multiple;
//#if COAT
      hiddenR = Fc + (1.0-Fc)*hiddenR;
//#endif
      col += thr * energy.multiple * envL(n, 0.80);
    }
//#endif
    // where the reflection goes: straight for a polished surface, off one facet of a rough one
    var dir = mir;
//#if ROUGH
    dir = frMetalDir(rd, n, b, rd_in, xi);
//#if ROUGH_REF
//#if WEAR
    {
      let normal = wr.n;
      let alpha = sqrt(mu.surf.x*mu.surf.x + wr.sig*wr.sig);
      let frame = brushSurface(wr,p,alpha);
//#else
    {
      let normal = n;
      let alpha = mu.surf.x;
//#endif
//#if WEAR
      let coated = mu.surf.z > 1.001;
      let sample = frGlossyFrame(p,rd,normal,rnd2(select(82+b,b,coated),rd_in,xi),alpha,coated,frame.across,frame.slopes,brushed);
//#else
//#if COAT
      let sample = frGlossySample(p, rd, normal, rnd2(b, rd_in, xi), alpha, true);
//#else
      let sample = frGlossySample(p, rd, normal, rnd2(82+b, rd_in, xi), alpha, false);
//#endif
//#endif
      if (sample.weight <= 0.0) { break; }
      dir = sample.direction;
//#if COAT
      R *= sample.weight;
//#else
      R = metalRGB(sample.facetCosine, 1.0, fd) * sample.weight;
//#endif
    }
//#endif
    // The lobe this surface reflects: spread by what the light was spread by already, by the facets,
    // and by the bend of the patch the cone landed on. Its middle is followed to the next surface while
    // it is narrow (share fo); the rest of it sees the lights, less what the shape itself hides of them.
    fv = frMetalV(fv, frBend(p, n, fp, rd));
//#if WEAR
    // (the wear spreads the lobe further; the polishing lines draw it out across themselves: brushN, one
    // facet per frame, the rest of the lobe read softened about that facet's reflection and hidden by the
    // shape along it)
    fv = frCap(fv + wr.sig * wr.sig);
    var mid = frMid(rd, wr.n);
    var mid2 = mid;
//#if !ROUGH_REF
    if (brushed) {
      let sr = sqrt(fv);
      let m4 = brushN(wr, p, rd, fc, b, 0.3 * sr, 1.0);
      let m = m4.xyz;
      let m2 = brushN(wr, p, rd, fc, b, 0.3 * sr, -1.0).xyz;
      fv = 0.18 * fv + m4.w * m4.w;
      mid = frMid(rd, m);
      mid2 = frMid(rd, m2);
      let d0 = reflect(rd, m);
      dir = d0 - 2.0 * min(dot(d0, n), 0.0) * n;
    }
//#endif
//#if ROUGH_REF
    let lit0 = 0.5 * (frMetalLit(dir, mid, fv, b) + frMetalLit(dir, mid2, fv, b)) * (wr.keep * wr.tint);
//#else
    let incident = 0.5 * (frMetalLit(dir,mid,fv,b) + frMetalLit(dir,mid2,fv,b));
    let lit0 = frMetalRig(p,wr.n,rd,mid,fv,b,sqrt(mu.surf.x*mu.surf.x+wr.sig*wr.sig),mu.surf.z>1.001,incident) * (wr.keep*wr.tint);
//#endif
    let fo = frFollow(fv);
    let hid = frMetalHid(p, n, mid, fv, lit0, hiddenR);
//#else
//#if ROUGH_REF
    let lit0 = frMetalLit(dir, frMid(rd, n), fv, b);
//#else
    let lit0 = frMetalRig(p,n,rd,frMid(rd,n),fv,b,mu.surf.x,mu.surf.z>1.001,frMetalLit(dir,frMid(rd,n),fv,b));
//#endif
    let fo = frFollow(fv);
    let hid = frMetalHid(p, n, frMid(rd, n), fv, lit0, hiddenR);
//#endif
    let lit = mix(hid, lit0, fo);
//#else
//#if FILM
    let lit = mirrorEnv(mir, mirrorLod(p, n, cosi));
//#else
//#if WEAR
    var lit2 = vec3f(0.0);
    if (brushed) {
      wr.aniso = 0.0;
      // (each facet's reflection read a little softened: the frames converge without grain)
      let sb = 0.3 * min(mu.more.w, 1.5) * BR_ACROSS;
      let s0 = wr.sig * wr.sig + sb * sb;
      let f1 = brushN(wr, p, rd, fc, b, 0.0, 1.0);
      let f2 = brushN(wr, p, rd, fc, b, 0.0, -1.0);
      wr.sig = sqrt(s0 + f1.w * f1.w);
      wr.n = f2.xyz;
      lit2 = wearSee(rd, wr, p, cosi);
      wr.n = f1.xyz;
      dir = reflect(rd, wr.n);
    }
    var lit = wearSee(rd, wr, p, cosi) * (wr.keep * wr.tint);
    if (brushed) { lit = 0.5 * (lit + lit2 * (wr.keep * wr.tint)); }
//#else
    let lit = sharpEnv(mir, p, n, cosi);
//#endif
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
//#if GRATING
      let reflected = gratingReflect(p,n,rd,lit,xi);
//#else
      let reflected = lit;
//#endif
//#if FILM
      col += thr * R * (reflected + skyOf(mir));
//#else
      col += thr * R * reflected;
//#endif
      break;
    }
//#if ROUGH
    // the part of a wide lobe that is not followed to the next surface: the lights less what the shape
    // hides of them (its middle has just met the shape, so the unhidden lights would be far too bright)
    col += thr * R * hid * (1.0 - fo);
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
// the crystal's axis as the camera sees it
fn axisW() -> vec3f {
  return normalize(vec3f(dot(u.obj0.xyz, mu.axis.xyz), dot(u.obj1.xyz, mu.axis.xyz), dot(u.obj2.xyz, mu.axis.xyz)));
}
//#endif

//#if WALK
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
      let rates = collisionRates();
      let fly = walkFlight(rates);
      var got = fly;
      var hit = false;
      if (fly >= depth * 0.9) { got = max(marchTo(pos, dir, fly, &hit), 0.0); }
      pos += dir * got;
//#if GLOW
      lit += att * mu.glow.rgb * got;
//#endif
      att *= exp(-(u.absorb.xyz + mu.sssA.rgb) * got) * walkWeight(rates, got, !hit);
      if (max(att.x, max(att.y, att.z)) < 0.002) { break; }      // the body has swallowed it
      if (!hit) {
        dir = walkDirection(dir);
        continue;
      }
      let m = nrm(pos);
      let ci = dot(dir, m);
      if (ci <= 0.0) {                      // at the surface, but heading in: carry on
        pos += dir * 0.004;
        continue;
      }
      let interface_sample = pbr_dielectric_sample(
        traceDielectricHit(-m, resolvedSceneMedium(u.ior), airMedium()), -dir, PbrPathRng(vec2f(0.0), rnd()));
      dir = interface_sample.wi;
      if (interface_sample.medium_transition == PBR_MEDIUM_ENTER_DESTINATION) {
        pos += m * 0.006;
      } else {
        pos -= m * 0.004;
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
      let interface_sample = pbr_dielectric_sample(
        traceDielectricHit(n, airMedium(), resolvedSceneMedium(u.ior)), -dir, PbrPathRng(vec2f(0.0), rnd()));
      dir = interface_sample.wi;
      if (interface_sample.medium_transition == PBR_MEDIUM_ENTER_DESTINATION) {
        pos -= n * 0.004;
      } else {
        pos += n * 0.004;
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
//#if PAINT_TOP
fn paintTop(c: f32) -> f32 {
//#if COAT
  return paintBoundaryFresnel(c, 1.0, u.ior, mu.surf.z, true);
//#else
  return paintBoundaryFresnel(c, 1.0, u.ior, 1.0, false);
//#endif
}
//#endif
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
//#if PAINT_TOP
  return paintTop(c);
//#else
//#if COAT
  return fresnel(c, 1.0, mu.surf.z);
//#else
//#if HOLLOW
  return wallR(c);
//#else
  return fresnel(c, 1.0, u.ior);
//#endif
//#endif
//#endif
}
//#else
// The same for a group of wavelengths: with a film on top, each has its own.
fn top4(c: f32, l: vec4f, d: f32) -> vec4f {
//#if PAINT_TOP
  return vec4f(paintTop(c));
//#else
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
  let px = 2.0 * u.tan_fov / u.res.y;
  var thr = vec3f(1.0);
  var sig = sig_in;
  for (var l = 0; l < LAYERS; l++) {
    let b = dot(ro, rd);
    let h = b * b - (dot(ro, ro) - BOUND * BOUND);
    if (h <= 0.0) { break; }
    let tFar = -b + sqrt(h);
    var t = 0.0;
    let hit = marchBeyond(ro, rd, tFar, px, &t);
//#if ROUGH
    if (!hit) { frMiss(l); break; }
//#else
    if (!hit) { break; }
//#endif
    t += mapNW(ro + rd * t);
    t += mapNW(ro + rd * t);
    let p = ro + rd * t;
//#if ROUGH
    // The cone of light lands on a patch of this piece: the normal is the patch's, the patch's own bend
    // spreads what it reflects and lets through, and the piece is taken to cover as much of the cone as
    // it does (the rest of the cone passes it by and sees the lights).
    let fp = t * sig;
    frHit(l, fp, sig);
    let n = faceIncident(frNrm(p, fp), -rd);
    let ne = faceIncident(frFarIn(n, rd, p), -rd);
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
    let n = faceIncident(nrm(p), -rd);
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
//#if ROUGH
    let F = fresnel(ci, 1.0, ior);
//#else
//#if BIREF
    let F = fresnel(ci, 1.0, ior);
//#else
    let F = pbr_dielectric_split(traceDielectricHit(n, airMedium(), resolvedSceneMedium(ior)), -rd).reflectance;
//#endif
//#endif
//#endif
//#endif
//#endif
//#if SSS
//#if !WALK
    let colPreF = col;
//#endif
//#endif
//#if ROUGH
//#if FILM
//#if ENVL
    col += thr * F * (frFar(rd, n, p, sig, fp) + skyFar(reflect(rd, n)));
//#else
    // (seen through the body: the picture at a middling level, without measuring how the surface bends)
    col += thr * F * (mirrorEnv(reflect(rd, n), 3.0) + skyFar(reflect(rd, n)));
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
//#if SSS
//#if !WALK
    col = mix(col, colPreF, clamp(mu.tab[19].w, 0.0, 1.0));
//#endif
//#endif
//#if ROUGH
    {
      // A wide cone is not followed through the piece: the piece scatters it every way, and what comes
      // out of it this side is the light from all round.
//#if SSS
//#if !WALK
      let wd = frWide(sig) * (1.0 - clamp(mu.tab[19].w, 0.0, 1.0));
//#else
      let wd = frWide(sig);
//#endif
//#else
      let wd = frWide(sig);
//#endif
      col += thr * (1.0 - F) * wd * frAll(rd);
      thr *= 1.0 - wd;
      if (wd > 0.999) { return col; }
    }
//#endif
//#if ROUGH
    var d2 = refract(rd, ne, 1.0 / ior);
//#else
//#if BIREF
    var d2 = refract(rd, ne, 1.0 / ior);
//#else
    var d2 = pbr_dielectric_split(traceDielectricHit(ne, airMedium(), resolvedSceneMedium(ior)), -rd).transmitted;
//#endif
//#endif
//#if SSS
//#if !WALK
    let sssOpB = clamp(mu.tab[19].w, 0.0, 1.0);
    if (sssOpB > 0.0) { d2 = normalize(mix(d2, normalize(mix(rd, -n, 0.12)), sssOpB)); }
//#endif
//#endif
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
//#if SSS
    let incoming = thr * (1.0 - F);
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
//#if WALK
      let rates = collisionRates();
      let travelled = max(s,0.0);
      let fly = walkFlight(rates);
      if (fly < travelled) {
        let scattered = incoming * exp(-(mu.sssA.rgb + u.absorb.xyz) * fly) * walkWeight(rates,fly,true);
        col += scattered * wander(q - d2 * (s-fly), walkDirection(d2));
        // The uncollided part is accounted for deterministically below.
      }
      thr *= exp(-(rates + mu.sssA.rgb) * travelled);
//#else
      let glassScatter = glowAlong(q - d2*s,d2,s);
      let glassThr = thr * exp(-(collisionRates() + mu.sssA.rgb)*max(s,0.0)) * throughHue(s);
      if (sssOpB > 0.0) {
        let tauB = waxTau(s);
        let waxThr = incoming * exp(-vec3f(tauB)) * waxThroughTint(tauB);
        col += incoming * mix(glassScatter, waxVolume(q - d2*s, d2, s, n, 1.0, false), sssOpB);
        thr = mix(glassThr, waxThr, sssOpB);
      } else {
        col += incoming * glassScatter;
        thr = glassThr;
      }
//#endif
    }
//#endif
    var m = nrm(q);
    var co = dot(d2, m);
    if (co < 0.0) { m = -m; co = -co; }
    m = -faceIncident(-m, -d2);
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
//#if ROUGH
    let F2 = fresnel(co, ior, 1.0);
//#else
//#if BIREF
    let F2 = fresnel(co, ior, 1.0);
//#else
    let F2 = pbr_dielectric_split(traceDielectricHit(-m, resolvedSceneMedium(ior), airMedium()), -d2).reflectance;
//#endif
//#endif
//#endif
//#endif
//#if ROUGH
//#if HOLLOW
    thr *= 1.0 - F2;
    rd = refract(d2, -m, ior);
//#else
//#if COAT
//#if SSS
//#if !WALK
    let F2c = F2 * (1.0 - sssOpB);
    if (F2c > 0.999) {
      return col + thr * seen(reflect(d2, m), max(sig, spreadR()));
    }
    thr *= 1.0 - F2c;
    rd = normalize(mix(refract(d2, -m, ior), d2, sssOpB));
//#else
    if (F2 > 0.999) {                       // trapped by total reflection under the coat: take the bounce and stop there
      return col + thr * seen(reflect(d2, m), max(sig, spreadR()));
    }
    thr *= 1.0 - F2;
    rd = refract(d2, -m, ior);
//#endif
//#else
    if (F2 > 0.999) {                       // trapped by total reflection under the coat: take the bounce and stop there
      return col + thr * seen(reflect(d2, m), max(sig, spreadR()));
    }
    thr *= 1.0 - F2;
    rd = refract(d2, -m, ior);
//#endif
//#else
    {
      // The wall this piece is left by, met by a cone of light: the part of it beyond the critical angle
      // is sent back inside, and is taken to leave at the bounce, softened; the rest goes on, bent and spread.
      let mf = frFarFacet(m, d2, q);
      let vi = sig * sig / (ior * ior) + frVin0(ior);
//#if SSS
//#if !WALK
      let w = mix(frWall(dot(d2, mf), vi, ior, 1.0, vec2f(0.0)), vec3f(0.0, 0.0, 0.0), sssOpB);
//#else
      let w = frWall(dot(d2, mf), vi, ior, 1.0, vec2f(0.0));
//#endif
//#else
      let w = frWall(dot(d2, mf), vi, ior, 1.0, vec2f(0.0));
//#endif
      if (w.x > 1.0 - frLeak()) { return col + thr * seen(reflect(d2, mf), max(sig, spreadR())); }
      col += thr * w.z * seen(reflect(d2, mf), max(sig, spreadR()));
      thr *= 1.0 - w.x;
      rd = frExit(d2, m, mf, w, ior);
//#if SSS
//#if !WALK
      if (sssOpB > 0.0) { rd = normalize(mix(rd, d2, sssOpB)); }
      sig = sqrt(max(sig * sig, frVout(vi, w, dot(rd, m), ior, vec2f(0.0)) * (1.0 - sssOpB)));
//#else
      sig = sqrt(max(sig * sig, frVout(vi, w, dot(rd, m), ior, vec2f(0.0))));
//#endif
//#else
      sig = sqrt(max(sig * sig, frVout(vi, w, dot(rd, m), ior, vec2f(0.0))));
//#endif
    }
//#endif
//#endif
//#else
//#if !HOLLOW
//#if SSS
//#if !WALK
    if (F2 * (1.0 - sssOpB) > 0.999) {
      return col + thr * seen(reflect(d2, m), sig);
    }
//#else
    if (F2 > 0.999) {                       // trapped by total reflection: take the bounce and stop there
      return col + thr * seen(reflect(d2, m), sig);
    }
//#endif
//#else
    if (F2 > 0.999) {                       // trapped by total reflection: take the bounce and stop there
      return col + thr * seen(reflect(d2, m), sig);
    }
//#endif
//#endif
//#if !HOLLOW
//#if SSS
//#if !WALK
    thr *= 1.0 - F2 * (1.0 - sssOpB);
//#else
    thr *= 1.0 - F2;
//#endif
//#else
    thr *= 1.0 - F2;
//#endif
//#else
    thr *= 1.0 - F2;
//#endif
//#if BIREF
    rd = refract(d2, -m, ior);
//#else
    rd = pbr_dielectric_split(traceDielectricHit(-m, resolvedSceneMedium(ior), airMedium()), -d2).transmitted;
//#endif
//#if !HOLLOW
//#if SSS
//#if !WALK
    if (sssOpB > 0.0) { rd = normalize(mix(rd, d2, sssOpB)); }
//#endif
//#endif
//#endif
//#endif
    ro = q + m * 0.006;
  }
//#if SSS
//#if !WALK
  let exitSig = max(sig, 1.15 * clamp(mu.tab[19].w, 0.0, 1.0));
  return col + thr * seen(rd, exitSig);
//#else
  return col + thr * seen(rd, sig);
//#endif
//#else
  return col + thr * seen(rd, sig);
//#endif
}

// What shade() does, for every material with a clear or milky body.
// What the surface's own reflection is multiplied by: the inverse of the exposure the picture was raised
// by for a milky body (1 otherwise).
fn surfaceK() -> f32 {
  let k = bitcast<f32>(u.pad0);
  return select(1.0, k, k > 0.0 && k <= 1.0);
}

fn shadeM(ro: vec3f, rd: vec3f, t: f32, xi: f32, W: i32, fc: vec2f) -> vec3f {
  let p = ro + rd * t;
//#if PAINT_INTERFACE
  let rawN = nrm(p);
  var n = faceIncident(rawN, -rd);
  let c0 = dot(-rd, rawN);
  let perp = rawN - rd * dot(rawN, rd);
  let pl = length(perp);
  if (c0 < 0.72 && pl > 1e-5) {
    let inDir = -perp / pl;
    let vox = 2.0 * HB / min(u.vres, 768.0);
    let qLoc = p + (inDir - rd) * (1.2 * vox);
    let nLoc0 = nrm(qLoc);
    let nLoc = nrm(qLoc - nLoc0 * mapW(qLoc));
    let onBev = smoothstep(0.008, 0.045, dot(-rd, nLoc) - c0);
    if (onBev > 0.0) {
      var qOut = p - inDir * (2.2 * vox) - rd * (4.0 * vox);
      var minD = 1e9;
      for (var s = 0; s < 8; s++) {
        let d = mapW(qOut);
        minD = min(minD, d);
        qOut += rd * max(d, 0.5 * vox);
      }
      let isSil = smoothstep(0.20 * vox, 0.70 * vox, minD);
      if (isSil > 0.0) {
        let q1 = p + (inDir - rd) * (2.6 * vox);
        let n10 = nrm(q1);
        let n1 = nrm(q1 - n10 * mapW(q1));
        let c1 = dot(-rd, n1);
        if (c1 > c0 + 0.015) {
          let q2 = p + (inDir - rd) * (3.8 * vox);
          let n20 = nrm(q2);
          let n2 = nrm(q2 - n20 * mapW(q2));
          let flat = smoothstep(0.9988, 0.9997, dot(n1, n2));
          let w = isSil * onBev * flat * smoothstep(0.015, 0.06, c1 - c0);
          n = normalize(mix(n, faceIncident(n1, -rd), w));
        }
      }
    }
  }
  let minC = clamp(0.22 + 0.60 * mu.tab[30].w, 0.22, 0.55);
  let cn = -dot(rd, n);
  if (cn < minC) {
    let tang = n + rd * cn;
    let tl = length(tang);
    if (tl > 1e-5) { n = tang * (sqrt(max(1.0 - minC * minC, 0.0)) / tl) - rd * minC; }
  }
//#else
  let n = faceIncident(nrm(p), -rd);
//#endif
//#if ROUGH
//#if COAT
  let ne = n;                               // (under a coat the top of the surface is the coat, and smooth)
  let cosi = clamp(-dot(rd, n), 0.0, 1.0);
//#else
  let ne = faceIncident(frIn(n, rd, xi), -rd); // the facet light gets in through (the surface's own normal, unless by chance)
  let cosi0 = frCos(-dot(rd, ne));
//#if SSS
  let cosi = mix(cosi0, max(cosi0, 0.62), clamp(mu.tab[19].w * sqrt(clamp(mu.surf.x, 0.0, 1.0)), 0.0, 1.0));
//#else
  let cosi = cosi0;
//#endif
//#endif
//#else
  let cosi = clamp(-dot(rd, n), 0.0, 1.0);
//#endif
//#if WALK
  rState = pcg(bitcast<u32>(rd.x) + pcg(bitcast<u32>(rd.y) + pcg(u32(u.frame) + pcg(bitcast<u32>(xi)))));
//#endif

  // ---- the surface: what it reflects, and through which facet light gets in
//#if ROUGH
//#if COAT
  let baseReflection = frTop(rd, n, xi, cosi);        // the body's own reflection, spread by its roughness
//#else
  let baseReflection = frTop(rd, n, xi, cosi0);       // the body's own reflection, spread by its roughness
//#endif
//#else
//#if WEAR
  let wr = wearAt(p, n);
  let baseReflection = wearSee(rd, wr, p, cosi) * (wr.keep * select(wr.tint, vec3f(1.0), mu.tab[31].w > 0.0));
//#else
//#if SSS
  // (a ceramic's glaze softens what it mirrors by its haze, mu.tab[30].w)
  var baseReflection = sharpEnv(reflect(rd, n), p, n, cosi);
  if (mu.tab[30].w > 0.0) { baseReflection = envL(reflect(rd, n), mu.tab[30].w); }
//#else
  let baseReflection = env(reflect(rd, n));
//#endif
//#endif
  let ne = n;
//#endif
//#if SSS
//#if !WALK
  let sssOpTop = clamp(mu.tab[19].w, 0.0, 1.0);
  var waxAo = 1.0;
  if (sssOpTop > 0.0) { waxAo = waxOcclusion(p, n); }
//#endif
//#endif
//#if GRATING
  let envB = gratingReflect(p,n,rd,baseReflection,xi);
//#else
//#if SSS
//#if !WALK
  let waxSpecScale = (1.0 - 0.30 * clamp(mu.surf.x, 0.0, 1.0)) * (0.30 + 0.70 * waxAo);
  let envB = baseReflection * mix(1.0, waxSpecScale, sssOpTop);
//#else
  let envB = baseReflection;
//#endif
//#else
  let envB = baseReflection;
//#endif
//#endif
//#if COAT
//#if ROUGH
  // (the coat's reflection is sharp whatever lies under it)
//#if WEAR
  let wr = wearAt(p, n);
  let coatReflection = wearSee(rd, wr, p, cosi) * (wr.keep * select(wr.tint, vec3f(1.0), mu.tab[31].w > 0.0));
//#else
  let coatReflection = sharpEnv(reflect(rd, n), p, n, cosi);
//#endif
//#if GRATING
  let envT = gratingReflect(p,n,rd,coatReflection,xi);
//#else
  let envT = coatReflection;
//#endif
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
//#if GRATING
  let envF = gratingReflect(p,n,rd,mirrorEnv(reflect(rd,n), mirrorLod(p,n,cosi)),xi) + skyOf(reflect(rd,n));
//#else
  let envF = mirrorEnv(reflect(rd, n), mirrorLod(p, n, cosi)) + skyOf(reflect(rd, n));
//#endif
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
//#if PAINT_FILM
//#if COAT
      let coated = true;
//#else
      let coated = false;
//#endif
      let film = paintFilmLanes(cosi,1.0,iorOf(i,xi,W),mu.surf.z,coated,mu.film.y,fd,l);
      acc += tint4(l,film.top)*envF + tint4(l,film.substrate)*envB;
//#else
      let Ft = top4(cosi, l, fd);
//#if FILM
      let lit = envF;
//#else
      let lit = envT;
//#endif
//#if COAT
//#if PAINT_COAT
      let shares = pbr_layer_shares(cosi, 1.0, iorOf(i,xi,W), mu.surf.z, true);
      acc += tint4(l, vec4f(shares.x)) * lit + tint4(l, vec4f(shares.y)) * envB;
//#else
      acc += tint4(l, Ft) * lit + tint4(l, (1.0 - Ft) * (1.0 - Ft) * under(cosi, iorOf(i, xi, W))) * envB;
//#endif
//#else
      acc += tint4(l, Ft) * lit;
//#endif
//#endif
    }
//#end
//#else
//#if COAT
//#if PAINT_COAT
    let shares = pbr_layer_shares(cosi, 1.0, iorOf(i,xi,W), mu.surf.z, true);
    acc += wOf(i,xi,W) * (shares.x * envT + shares.y * envB);
//#else
    acc += wOf(i, xi, W) * (top1(cosi) * envT + (1.0 - top1(cosi)) * (1.0 - top1(cosi)) * under(cosi, iorOf(i, xi, W)) * envB);
//#endif
//#else
    acc += wOf(i, xi, W) * (top1(cosi) * envT);
//#endif
//#endif
  }
  acc *= f32(MODES);
  // (an exposure raised for a milky body's own light is not to brighten the mirror on its surface: a
  // polished stone reflects its surroundings faintly, as its index has it)
  acc *= surfaceK();
//#if SSS
  // A ceramic (mu.tab[31].w = 1 + the share kept): its body is lit at its surface by the surface pass
  // (surface.rs); of the light inside it only a share is kept, for thin edges and light from behind.
  let accTop = acc;
//#endif

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
//#if PAINT_FILM
//#if COAT
    t[g] = paintFilmLanes(cosi,1.0,ior,mu.surf.z,true,mu.film.y,fd,lams(i,GROUP,xi,W)).diffuse;
//#else
    t[g] = paintFilmLanes(cosi,1.0,ior,mu.surf.z,false,mu.film.y,fd,lams(i,GROUP,xi,W)).diffuse;
//#endif
//#else
//#if COAT
//#if PAINT_COAT
    t[g] = vec4f(pbr_layer_shares(cosi,1.0,ior,mu.surf.z,true).z);
//#else
    t[g] = (1.0 - top4(cosi, lams(i, GROUP, xi, W), fd)) * (1.0 - under(cosi, ior));
//#endif
//#else
    t[g] = 1.0 - top4(cosi, lams(i, GROUP, xi, W), fd);
//#endif
//#endif
//#end
    var thr = max(max4(t0), max(max4(t1), max4(t2)));
//#else
//#if COAT
//#if PAINT_COAT
    var thr = pbr_layer_shares(cosi,1.0,ior,mu.surf.z,true).z;
//#else
    var thr = (1.0 - top1(cosi)) * (1.0 - under(cosi, ior));
//#endif
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
    var dir = pbr_dielectric_split(traceDielectricHit(ne, airMedium(), resolvedSceneMedium(ior)), -rd).transmitted;
//#endif
//#endif
//#if SSS
//#if !WALK
    if (sssOpTop > 0.0) {
      dir = normalize(mix(dir, normalize(mix(rd, -n, 0.12)), sssOpTop));
//#if BIREF
      kv = dir * length(kv);
//#endif
    }
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
    for (var b = 0; b < MAX_NB; b++) {
      if (b >= u.nb || thr < 0.02) { break; }      // nothing got in (total reflection at the surface): skip the interior
      // ---- to the next surface
//#if WALK
      // How far the ray gets before it is scattered. Short of the surface: it is turned to a random
      // direction there and followed as it wanders, to the lights. As far as the surface: it is treated
      // there as a ray in clear glass is.
      let rates = collisionRates();
      let fly = walkFlight(rates);
      var s = fly;
      var out = false;
      if (fly >= -mapW(pos) * 0.9) { s = marchTo(pos, dir, fly, &out); }
      pos += dir * s;
//#if GLOW
      acc += {{WEIGHT}} * att * mu.glow.rgb * s;
//#endif
      att *= exp(-(u.absorb.xyz + mu.sssA.rgb) * s) * walkWeight(rates, s, !out);
      // A survived flight is ballistic; a collided flight follows its actual phase.
      if (!out) {
        acc += {{WEIGHT}} * att * wander(pos, walkDirection(dir));
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
//#if SSS
      let incoming = att;
//#endif
      att *= exp(-u.absorb.xyz * s);
//#if SSS
      {
        // part of the ray is scattered away on this stretch; in its place comes the light the body
        // holds along it, scattered towards the eye
        let ext = exp(-(collisionRates() + mu.sssA.rgb) * max(s, 0.0));
        let glassScatter = glowAlong(q - dir * s, dir, s);
        let glassAtt = att * ext * throughHue(s);
        if (sssOpTop > 0.0) {
          let tau0 = waxTau(s);
          var waxScatter = waxVolume(q - dir * s, dir, s, n, waxAo, b == 0);
          if (b == 0) { waxScatter += waxNear(n, waxAo, s); }
          let waxScale = select(1.0, 0.56, b == 0);
          let waxAtt = incoming * (waxScale * exp(-vec3f(tau0))) * waxThroughTint(tau0);
          acc += {{WEIGHT}} * incoming * mix(glassScatter, waxScatter, sssOpTop);
          att = mix(glassAtt, waxAtt, sssOpTop);
        } else {
          acc += {{WEIGHT}} * incoming * glassScatter;
          att = glassAtt;
        }
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
      mN = -faceIncident(-mN, -dir);
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
//#if SSS
//#if !WALK
      let we = frEnd(fv, bend) * (1.0 - sssOpTop);
//#else
      let we = frEnd(fv, bend);
//#endif
//#else
      let we = frEnd(fv, bend);
//#endif
      acc += {{WEIGHT}} * (fw.x * we) * att * frDeep(s) * frAll(mN);
//#if FINE
//#each
      t[g] *= 1.0 - fw.x * we;
//#end
//#endif
      thr *= 1.0 - fw.x * we;
      var bare = fw.x * (1.0 - we) / max(1.0 - fw.x * we, 1e-4);
//#if SSS
//#if !WALK
      if (bare * (1.0 - sssOpTop) > 1.0 - frLeak()) {
//#else
      if (bare > 1.0 - frLeak()) {
//#endif
//#else
      if (bare > 1.0 - frLeak()) {
//#endif
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
//#if BIREF
      let bare = fresnel(cf, iorIn, 1.0);
//#else
      let bare = pbr_dielectric_split(traceDielectricHit(-mf, resolvedSceneMedium(iorIn), airMedium()), -dir).reflectance;
//#endif
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
//#if SSS
//#if !WALK
      let bareExit = bare * (1.0 - sssOpTop);
      if (bareExit < 0.999) {
//#else
      if (bare < 0.999) {
//#endif
//#else
      if (bare < 0.999) {
//#endif
        // ---- the part that leaves: on through further pieces of the body, to the lights
//#if BIREF
        // the wave keeps the part of itself that lies along the surface
        var od = normalize(kv - dot(kv, mN) * mN + sqrt(max(1.0 - dot(kv, kv) + dot(kv, mN) * dot(kv, mN), 0.0)) * mN);
//#else
//#if ROUGH
        var od = refract(dir, -mN, iorIn);
//#else
        var od = pbr_dielectric_split(traceDielectricHit(-mN, resolvedSceneMedium(iorIn), airMedium()), -dir).transmitted;
//#endif
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
        // The surface facets spread the exiting light. Extinction already removed every
        // volume collision from this ray; the in-scattered light was integrated separately.
        var sg = 0.0;
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
//#if SSS
//#if !WALK
        if (sssOpTop > 0.0) {
          od = normalize(mix(od, dir, 0.55 * sssOpTop));
          sg = max(sg, 0.16 * sssOpTop * sssOpTop);
        }
//#endif
//#endif
//#if FILM
        // (what the far face of the film sent back has been reflected by the film: it sees the film's sky)
        let sky = select(0.0, 1.0, b > 0);
        if (LAYERS > 0) {
//#if SSS
//#if !WALK
          if (sssOpTop < 0.999) {
            X = att * beyondM(q + mN * 0.006, od, ior, sqrt(sg), sky);
          }
          if (sssOpTop > 0.0) {
            X = mix(X, att * (waxExit(q, mN, dir) + sky * skyDim(od)), sssOpTop);
          }
//#else
          X = att * beyondM(q + mN * 0.006, od, ior, sqrt(sg), sky);
//#endif
//#else
          X = att * beyondM(q + mN * 0.006, od, ior, sqrt(sg), sky);
//#endif
        } else {
//#if SSS
//#if !WALK
          X = att * (mix(seen(od, sqrt(sg)), waxExit(q, mN, dir), sssOpTop) + sky * skyDim(od));
//#else
          X = att * (seen(od, sqrt(sg)) + sky * skyDim(od));
//#endif
//#else
          X = att * (seen(od, sqrt(sg)) + sky * skyDim(od));
//#endif
        }
//#else
        if (LAYERS > 0) {
//#if SSS
//#if !WALK
          if (sssOpTop < 0.999) {
            X = att * beyondM(q + mN * 0.006, od, ior, sqrt(sg));
          }
          if (sssOpTop > 0.0) {
            X = mix(X, att * waxExit(q, mN, dir), sssOpTop);
          }
//#else
          X = att * beyondM(q + mN * 0.006, od, ior, sqrt(sg));
//#endif
//#else
          X = att * beyondM(q + mN * 0.006, od, ior, sqrt(sg));
//#endif
        } else {
//#if SSS
//#if !WALK
          X = att * mix(seen(od, sqrt(sg)), waxExit(q, mN, dir), sssOpTop);
//#else
          X = att * seen(od, sqrt(sg));
//#endif
//#else
          X = att * seen(od, sqrt(sg));
//#endif
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
        let Fx0 = 1.0 - (1.0 - F1) * (1.0 - filmR4(ccb, l, fd, mu.surf.z, 1.0));
//#else
        let Fx0 = vec4f(1.0 - (1.0 - F1) * (1.0 - fresnel(ccb, mu.surf.z, 1.0)));
//#endif
//#else
//#if FILM
        let Fx0 = filmR4(cf, l, fd, iorIn, 1.0);
//#else
        let Fx0 = vec4f(bare);
//#endif
//#endif
//#if SSS
//#if !WALK
        let Fx = Fx0 * (1.0 - sssOpTop);
//#else
        let Fx = Fx0;
//#endif
//#else
        let Fx = Fx0;
//#endif
        acc += tint4(l, t[g] * (1.0 - Fx)) * X;
        t[g] *= Fx;
      }
//#end
      thr = max(max4(t0), max(max4(t1), max4(t2)));
//#else
//#if COAT
//#if SSS
//#if !WALK
      let Fx = (1.0 - (1.0 - F1) * (1.0 - fresnel(ccb, mu.surf.z, 1.0))) * (1.0 - sssOpTop);
//#else
      let Fx = 1.0 - (1.0 - F1) * (1.0 - fresnel(ccb, mu.surf.z, 1.0));
//#endif
//#else
      let Fx = 1.0 - (1.0 - F1) * (1.0 - fresnel(ccb, mu.surf.z, 1.0));
//#endif
//#else
//#if SSS
//#if !WALK
      let Fx = bareExit;
//#else
      let Fx = bare;
//#endif
//#else
      let Fx = bare;
//#endif
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
//#if SSS
//#if !WALK
  frRad *= (1.0 - sssOpTop);
//#endif
//#endif
  frSeen(n, rd, xi, cosi, W);               // (what the stage that blurs the inside of rough glass is told about this surface)
//#endif
//#if SSS
  if (mu.tab[31].w > 0.0) { acc = accTop + (acc - accTop) * (mu.tab[31].w - 1.0); }
//#endif
  return acc / (wsum * f32(MODES));
}

fn grazeM(p: vec3f, rd: vec3f) -> vec3f {
//#if COAT
  return env(reflect(rd, nrm(p))) * surfaceK();
//#else
//#if SSS
//#if !WALK
  let sssOpG = clamp(mu.tab[19].w, 0.0, 1.0);
  let ng = nrm(p);
  let baseG = seen(reflect(rd, ng), spreadR()) * surfaceK();
  if (sssOpG > 0.0) {
    let fG = fresnel(0.60, 1.0, u.ior);
    let waxG = fG * baseG + (1.0 - fG) * waxNear(ng, 1.0, 0.08);
    return mix(baseG, waxG, sssOpG * smoothstep(0.0, 0.20, sqrt(clamp(mu.surf.x, 0.0, 1.0))));
  }
  return baseG;
//#else
  return seen(reflect(rd, nrm(p)), spreadR()) * surfaceK();
//#endif
//#else
  return seen(reflect(rd, nrm(p)), spreadR()) * surfaceK();
//#endif
//#endif
}
//#endif
//#endif
//#endif
//#endif

// ---- the graded index and double refraction, as the tracer of several bodies shares them ---------------
// (kept apart from the rest so that they are there whatever the bodies are made of: a metal among the
// bodies leaves out the part above for one material of glass)
//#if GRIN
// The index at depth d below the surface, less the index at the surface: it changes by mu.exo.x over a
// depth of about 1 / mu.exo.y and then stays.
fn grinAt(depth: f32) -> f32 { return mu.exo.x * (1.0 - exp(-max(depth, 0.0) * mu.exo.y)); }
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
//#endif

//#if AIR_HIT
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
//#endif
