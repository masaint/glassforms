// Glass Forms: the tracer. A statement-for-statement port of TRACE_HEAD + volumeGLSL() + TRACE_BODY of the
// web version (shaders.js, forms.js), for a shape baked into a distance-field volume.
//
// The tracer is spectral: every pixel fires `ns` rays at different points within it, each carrying `nw`
// wavelengths, each wavelength with its own index of refraction and followed for up to `nb` internal bounces.
//
// {{...}} are filled in when the shader is built: they differ between families of shapes.

struct U {
  res: vec2f,
  jitter: vec2f,          // sub-pixel offset (used while accumulating)
  shift: vec2f,           // screen-space shift of the view, in units of half-height
  focus: vec2f,           // the band that is in focus: its near and far planes, as depth from the object's centre
  obj0: vec4f,            // object -> world rotation, as three rows
  obj1: vec4f,
  obj2: vec4f,
  env0: vec4f,            // the lights' own frame -> what the camera sees, as three rows
  env1: vec4f,
  env2: vec4f,
  absorb: vec4f,          // absorption per unit length inside the glass
  dist: f32,              // camera sits at (0,0,dist) looking down -z
  tan_fov: f32,           // tan(half vertical fov)
  lens: f32,              // radius of the camera's lens (0 = a pinhole: everything sharp)
  blades: f32,            // sides of the lens opening (below 3 = round)
  bokeh_rot: f32,         // and how far that opening is turned, radians
  ior: f32,               // index of refraction at 550 nm
  disp: f32,              // n(400 nm) - n(700 nm)
  vres: f32,              // voxels per side of the volume bound right now
  frame: i32,
  nw: i32,                // wavelengths carried by each sample
  ns: i32,                // samples (ray positions) per pixel
  nb: i32,                // internal bounces per wavelength
  lens_n: i32,            // rays through the lens for each ray through the pixel
  pad0: i32,
  pad1: i32,
  pad2: i32,
  medium: vec4i,           // raw identity, signed interface priority, two reserved zero lanes
}

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var envTex: texture_2d<f32>;
@group(0) @binding(2) var envSmp: sampler;
@group(0) @binding(3) var volTex: texture_3d<f32>;     // the baked distance field
@group(0) @binding(4) var volSmp: sampler;{{LIGHT_DECL}}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

const BOUND: f32 = 1.47;             // radius of a sphere (around the origin) that contains the object
const HB: f32 = 1.5;
const STEP_SCALE: f32 = {{STEP_SCALE}};
const STEPS_OUT: i32 = {{STEPS_OUT}};
const STEPS_IN: i32 = {{STEPS_IN}};
const LAYERS: i32 = {{LAYERS}};      // rays leaving the glass may pass through up to this many further pieces of it
const MAX_NW: i32 = 32;
const MAX_NB: i32 = 12;
{{PROFILE_DECL}}

// world -> object, and the lights' frame
fn toObj(p: vec3f) -> vec3f { return p.x * u.obj0.xyz + p.y * u.obj1.xyz + p.z * u.obj2.xyz; }
fn toEnv(d: vec3f) -> vec3f { return d.x * u.env0.xyz + d.y * u.env1.xyz + d.z * u.env2.xyz; }

// look up the equirectangular map of the lights: u runs around the horizon, v from straight up (0) to straight down (1)
fn envMap(rd: vec3f) -> vec3f {
  let uv = vec2f(atan2(rd.z, rd.x) * 0.15915494 + 0.5, acos(clamp(rd.y, -1.0, 1.0)) * 0.31830989);
  return {{ENV_LIGHT}};
}
// radiance seen by reflected rays and by rays leaving the glass
fn env(rd: vec3f) -> vec3f { return envMap(toEnv(rd)); }

// Marching uses plain trilinear lookups: cheap.
fn map(p: vec3f) -> f32 {
  let v = textureSampleLevel(volTex, volSmp, p / (2.0 * HB) + 0.5, 0.0).r;
  {{PROFILE_MAP}}return v;
}

// The surface itself is taken from a smoother reading of the same data. Trilinear interpolation has a kink
// at every voxel boundary, which shows as ripples and beading on glass. A cubic B-spline through the samples
// has none, and costs eight trilinear lookups placed between texels (Sigg & Hadwiger, GPU Gems 2).
fn vol(t: vec3f) -> f32 { return textureSampleLevel(volTex, volSmp, t, 0.0).r; }
fn mapN(p: vec3f) -> f32 {
  {{PROFILE_MAPN}}let res = u.vres;
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
  return g0.z * (g0.y * (g0.x * vol(vec3f(a.x, a.y, a.z)) + g1.x * vol(vec3f(b.x, a.y, a.z)))
               + g1.y * (g0.x * vol(vec3f(a.x, b.y, a.z)) + g1.x * vol(vec3f(b.x, b.y, a.z))))
       + g1.z * (g0.y * (g0.x * vol(vec3f(a.x, a.y, b.z)) + g1.x * vol(vec3f(b.x, a.y, b.z)))
               + g1.y * (g0.x * vol(vec3f(a.x, b.y, b.z)) + g1.x * vol(vec3f(b.x, b.y, b.z))));
}

fn mapW(p: vec3f) -> f32 { return map(toObj(p)); }
fn mapNW(p: vec3f) -> f32 { return mapN(toObj(p)); }

fn volumeSlopePlane(a: vec3f, b: vec3f, g: vec3f, axis: u32) -> f32 {
  if (axis == 0u) {
    return mix(mix(vol(vec3f(a.x, a.y, a.z)), vol(vec3f(a.x, b.y, a.z)), g.y),
               mix(vol(vec3f(a.x, a.y, b.z)), vol(vec3f(a.x, b.y, b.z)), g.y), g.z);
  }
  if (axis == 1u) {
    return mix(mix(vol(vec3f(a.x, a.y, a.z)), vol(vec3f(b.x, a.y, a.z)), g.x),
               mix(vol(vec3f(a.x, a.y, b.z)), vol(vec3f(b.x, a.y, b.z)), g.x), g.z);
  }
  return mix(mix(vol(vec3f(a.x, a.y, a.z)), vol(vec3f(b.x, a.y, a.z)), g.x),
             mix(vol(vec3f(a.x, b.y, a.z)), vol(vec3f(b.x, b.y, a.z)), g.x), g.y);
}

// Rays are marched through map() and then settled onto mapN()'s surface, and normals come from the
// analytic derivative of the cubic B-spline (24 trilinear lookups, continuous across voxel boundaries).
fn nrmVolume(p: vec3f) -> vec3f {
  {{PROFILE_MAPN}}{{PROFILE_MAPN}}{{PROFILE_MAPN}}let res = u.vres;
  let t = (toObj(p) / (2.0 * HB) + 0.5) * res - 0.5;
  let i = floor(t);
  let f = t - i;
  let f2 = f * f;
  let f3 = f2 * f;
  let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
  let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
  let w3 = f3 / 6.0;
  let g0 = w0 + w1;
  let g1 = 1.0 - g0;
  let a = (i - 0.5 + w1 / g0) / res;
  let b = (i + 1.5 + w3 / g1) / res;
  let d0 = -0.5 * (1.0 - f) * (1.0 - f);
  let d1 = 1.5 * f2 - 2.0 * f;
  let d3 = 0.5 * f2;
  let size = -(d0 + d1);
  let lo = (i - 0.5 - d1 / size) / res;
  let hi = (i + 1.5 + d3 / size) / res;
  let dx = volumeSlopePlane(vec3f(hi.x, a.yz), b, g1, 0u) - volumeSlopePlane(vec3f(lo.x, a.yz), b, g1, 0u);
  let dy = volumeSlopePlane(vec3f(a.x, hi.y, a.z), b, g1, 1u) - volumeSlopePlane(vec3f(a.x, lo.y, a.z), b, g1, 1u);
  let dz = volumeSlopePlane(vec3f(a.xy, hi.z), b, g1, 2u) - volumeSlopePlane(vec3f(a.xy, lo.z), b, g1, 2u);
  let local = size * vec3f(dx, dy, dz);
  if (dot(local, local) < 1e-30) { return vec3f(0.0, 1.0, 0.0); }
  return normalize(local.x * vec3f(u.obj0.x, u.obj1.x, u.obj2.x) + local.y * vec3f(u.obj0.y, u.obj1.y, u.obj2.y) + local.z * vec3f(u.obj0.z, u.obj1.z, u.obj2.z));
}
fn nrm(p: vec3f) -> vec3f {
  return nrmVolume(p);
}

// CIE 1931 colour matching functions (Wyman, Sloan & Shirley 2013 fit) -> linear sRGB
fn specW(l: f32) -> vec3f {
  var t1 = (l - 442.0) * select(0.0374, 0.0624, l < 442.0);
  var t2 = (l - 599.8) * select(0.0323, 0.0264, l < 599.8);
  let t3 = (l - 501.1) * select(0.0382, 0.0490, l < 501.1);
  let x = 0.362 * exp(-0.5 * t1 * t1) + 1.056 * exp(-0.5 * t2 * t2) - 0.065 * exp(-0.5 * t3 * t3);
  t1 = (l - 568.8) * select(0.0247, 0.0213, l < 568.8);
  t2 = (l - 530.9) * select(0.0322, 0.0613, l < 530.9);
  let y = 0.821 * exp(-0.5 * t1 * t1) + 0.286 * exp(-0.5 * t2 * t2);
  t1 = (l - 437.0) * select(0.0278, 0.0845, l < 437.0);
  t2 = (l - 459.0) * select(0.0725, 0.0385, l < 459.0);
  let z = 1.217 * exp(-0.5 * t1 * t1) + 0.681 * exp(-0.5 * t2 * t2);
  return vec3f( 3.2406 * x - 1.5372 * y - 0.4986 * z,
               -0.9689 * x + 1.8758 * y + 0.0415 * z,
                0.0557 * x - 0.2040 * y + 1.0570 * z);
}

// Cauchy dispersion, normalised so that iorAt(400) - iorAt(700) = disp and iorAt(550) = ior
fn iorAt(lam: f32) -> f32 { return u.ior + u.disp * (1.0e6 / (lam * lam) - 3.3058) / 4.2092; }

{{DIELECTRIC_FRESNEL}}

fn airMedium() -> PbrMedium { return PbrMedium(1.0, 0.0, 0u, 0); }
fn resolvedSceneMedium(ior: f32) -> PbrMedium {
  return PbrMedium(ior, 0.0, bitcast<u32>(u.medium.x), u.medium.y);
}
fn faceIncident(normal: vec3f, wo: vec3f) -> vec3f {
  let c = dot(wo, normal);
  if (c >= 0.0) { return normal; }
  let proj = normal - wo * c;
  let len2 = dot(proj, proj);
  if (len2 > 1e-12) { return normalize(proj + wo * 1e-4); }
  return wo;
}
fn traceDielectricHit(normal: vec3f, incident: PbrMedium, destination: PbrMedium) -> PbrSurfaceHit {
  let seed = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(normal.y) < 0.9);
  let tangent = normalize(cross(seed, normal));
  return PbrSurfaceHit(
    PbrFrame(tangent, cross(normal, tangent), normal), incident, destination, 550.0);
}

fn marchIn(ro: vec3f, rd: vec3f) -> f32 {
  let px = 2.0 * u.tan_fov / u.res.y;
  var t = 0.0;
  for (var i = 0; i < 2 * STEPS_IN; i++) {
    let p = ro + rd * t;
    let d = -mapW(p);
    let eps = clamp(0.25 * px * max(u.dist - p.z, 0.1), 0.00008, 0.0008);
    if (d < eps) { break; }
    t += d * STEP_SCALE;
    if (t > 2.5 * BOUND) { break; }
  }
  t -= mapNW(ro + rd * t);      // two Newton steps onto the smooth field's surface: removes grid ripple
  t -= mapNW(ro + rd * t);
  return t;
}

fn marchBeyond(ro: vec3f, rd: vec3f, tFar: f32, px: f32, tOut: ptr<function, f32>) -> bool {
  if (tFar < 0.0) { return false; }
  let vox = 2.0 * HB / u.vres;
  var t = 0.0;
  var bestD = 1e9;
  var bestT = 0.0;
  for (var i = 0; i < STEPS_OUT; i++) {
    let pos = ro + rd * t;
    let d = mapW(pos);
    if (d < bestD) { bestD = d; bestT = t; }
    let eps = clamp(0.25 * px * max(u.dist - pos.z, 0.1), 0.00008, 0.0008);
    if (d < eps) { *tOut = t; return true; }
    t += d * STEP_SCALE;
    if (t > tFar) { break; }
  }
  if (bestD < 0.75 * vox) {
    let hitEps = clamp(0.25 * px * max(u.dist - (ro.z + rd.z * bestT), 0.1), 0.00008, 0.0008);
    var rt = max(bestT - 0.75 * vox, 0.0);
    let rtEnd = min(bestT + 1.5 * vox, tFar);
    for (var ri = 0u; ri < 6u + u32(max(u.nb, 0) == 999); ri++) {
      let rd0 = mapNW(ro + rd * rt);
      if (rd0 < hitEps) { *tOut = max(rt + rd0, 0.0); return true; }
      rt += max(rd0, 0.15 * vox);
      if (rt > rtEnd) { break; }
    }
  }
  return false;
}

// A ray has just left the glass at ro, heading rd. If it meets the glass again, carry it on:
// add the reflection off that next surface and follow the refraction straight through it, for up
// to LAYERS further pieces. Without this, anything behind a piece of glass would be invisible
// through it. Returns the light arriving back along the ray.
fn beyond(ro_in: vec3f, rd_in: vec3f, ior: f32) -> vec3f {
  let px = 2.0 * u.tan_fov / u.res.y;
  var ro = ro_in;
  var rd = rd_in;
  var col = vec3f(0.0);
  var thr = vec3f(1.0);
  for (var l = 0; l < LAYERS; l++) {
    let b = dot(ro, rd);
    let h = b * b - (dot(ro, ro) - BOUND * BOUND);
    if (h <= 0.0) { break; }
    let tFar = -b + sqrt(h);
    var t = 0.0;
    let hit = marchBeyond(ro, rd, tFar, px, &t);
    if (!hit) { break; }
    t += mapNW(ro + rd * t);
    t += mapNW(ro + rd * t);
    let p = ro + rd * t;
    let n = faceIncident(nrm(p), -rd);
    let entry = pbr_dielectric_split(traceDielectricHit(n, airMedium(), resolvedSceneMedium(ior)), -rd);
    let F = entry.reflectance;
    col += thr * F * env(entry.reflected);
    let d2 = entry.transmitted;
    let q0 = p - n * 0.004;
    let s = marchIn(q0, d2);
    let q = q0 + d2 * s;
    var m = nrm(q);
    var co = dot(d2, m);
    if (co < 0.0) { m = -m; co = -co; }
    m = -faceIncident(-m, -d2);
    let exit = pbr_dielectric_split(traceDielectricHit(-m, resolvedSceneMedium(ior), airMedium()), -d2);
    let F2 = exit.reflectance;
    thr *= (1.0 - F) * exp(-u.absorb.xyz * s);
    if (F2 > 0.999) {                       // trapped by total reflection: take the bounce and stop there
      return col + thr * env(exit.reflected);
    }
    thr *= 1.0 - F2;
    rd = exit.transmitted;
    ro = q + m * 0.006;
  }
  return col + thr * env(rd);
}

// W wavelengths, the i-th at (i + xi) / W of the way through the spectrum
fn shade(ro: vec3f, rd: vec3f, t: f32, xi: f32, W: i32) -> vec3f {
  let p = ro + rd * t;
  let n = faceIncident(nrm(p), -rd);
  let entry = pbr_dielectric_split(traceDielectricHit(n, airMedium(), resolvedSceneMedium(u.ior)), -rd);
  let envR = env(entry.reflected);
  let F0 = entry.reflectance;

  // With no dispersion every wavelength bends alike: all of them take the same path and bring back the same
  // light, so one path is followed and stands for them all (the same picture for a third of the work).
  let alike = u.disp == 0.0;
  var one = vec3f(0.0);
  var acc = vec3f(0.0);
  var wsum = vec3f(0.0);
  var racc = vec3f(0.0);
  for (var i = 0; i < MAX_NW; i++) {
    if (i >= W || (alike && i >= 1)) { break; }
    let lam = 400.0 + 300.0 * (f32(i) + xi) / f32(W);
    let w = specW(lam);
    let ior = iorAt(lam);
    let Fl = F0;
    racc += w * Fl;

    let entry = pbr_dielectric_split(traceDielectricHit(n, airMedium(), resolvedSceneMedium(ior)), -rd);
    var dir = entry.transmitted;
    var pos = p - n * 0.004;
    var thr = 1.0 - Fl;
    var att = vec3f(1.0);
    var L = vec3f(0.0);
    for (var b = 0; b < MAX_NB; b++) {
      if (b >= u.nb || thr < 0.02) { break; }      // nothing got in (total reflection at the surface): skip the interior
      let s = marchIn(pos, dir);
      let q = pos + dir * s;
      var m = nrm(q);
      att *= exp(-u.absorb.xyz * s);
      var ci = dot(dir, m);
      if (ci < 0.0) { m = -m; ci = -ci; }
      m = -faceIncident(-m, -dir);
      let exit = pbr_dielectric_split(traceDielectricHit(-m, resolvedSceneMedium(ior), airMedium()), -dir);
      let F = exit.reflectance;
      if (F < 0.999) {
        let od = exit.transmitted;
        if (LAYERS > 0) {
          L += thr * (1.0 - F) * att * beyond(q + m * 0.006, od, ior);
        } else {
          L += thr * (1.0 - F) * att * env(od);
        }
      }
      thr *= F;
      if (thr < 0.02) { break; }
      dir = exit.reflected;
      pos = q - m * 0.004;
    }
    acc += w * L;
    wsum += w;
    one = L;
  }
  if (alike) { return envR * F0 + one; }
  return envR * (racc / wsum) + acc / wsum;
}

// The first glass along a ray: how far away it is, or -1.0 if there is none. Also reports the closest the
// ray came to the surface without touching it (graze, as an angle) and where (tGraze), for soft outlines.
fn primary(ro: vec3f, rd: vec3f, tGraze: ptr<function, f32>, graze: ptr<function, f32>) -> f32 {
  *graze = 1e9;
  *tGraze = max(-dot(ro, rd), 0.0);
  let b = dot(ro, rd);
  var h = b * b - (dot(ro, ro) - BOUND * BOUND);
  if (h <= 0.0) { return -1.0; }
  h = sqrt(h);
  var t = max(-b - h, 0.0);
  let tFar = -b + h;
  let px = 2.0 * u.tan_fov / u.res.y;        // size of a pixel at unit distance
  for (var i = 0; i < STEPS_OUT; i++) {
    let d = mapW(ro + rd * t);
    let r = d / t;
    if (r < *graze) { *graze = r; *tGraze = t; }
    if (d < 0.25 * px * t) {
      t += mapNW(ro + rd * t);
      t += mapNW(ro + rd * t);
      return t;
    }
    t += d * STEP_SCALE;
    if (t > tFar) { break; }
  }
  return -1.0;
}

// One sample: the light arriving through the point fc (in pixels), carrying W wavelengths.
// lens: where on the lens this sample passes, as a point of the unit disk.
// rgb: the light.  a: how far along the pixel's own ray the glass is (0 = it misses).
fn sampleAt(fc: vec2f, xi: f32, W: i32, lens: vec2f) -> vec4f {
  let uv = (2.0 * fc - u.res) / u.res.y + u.shift;
  var ro = vec3f(0.0, 0.0, u.dist);
  var rd = normalize(vec3f(uv * u.tan_fov, -1.0));
  let px = 2.0 * u.tan_fov / u.res.y;
  var tg: f32;
  var g: f32;
  var t = primary(ro, rd, &tg, &g);
  var depth = max(t, 0.0);
  if (u.lens > 0.0) {
    // Depth of field, traced rather than blurred afterwards. A real lens sends every ray from one point of
    // the focus plane to the same pixel, so the ray is started from a random point of the lens and aimed
    // at where the pixel's central ray crosses the focus plane. Blur then comes out right even for what
    // is seen through the glass.
    // Instead of one plane there is a band: the central ray is traced first, and the plane used is the
    // depth it found, held inside the band. Anything within the band is therefore exactly sharp, and
    // blur grows with distance outside it. With the two planes together this is an ordinary thin lens.
    let e = clamp(-(ro.z + rd.z * select(tg, t, t > 0.0)), u.focus.x, u.focus.y);
    let f = ro + rd * ((u.dist + e) / -rd.z);
    ro += vec3f(lens * u.lens, 0.0);
    rd = normalize(f - ro);
    t = primary(ro, rd, &tg, &g);
    if (depth == 0.0) { depth = max(t, 0.0); }
  }
  var col = {{BACKGROUND}};                       // what the camera sees where it misses the object
  if (t > 0.0) {
    col = {{SHADE}};
  } else if (g < px) {
    // the ray grazes the silhouette within a pixel: blend in the grazing reflection (Fresnel -> 1)
    let p = ro + rd * tg;
    col = mix(col, {{GRAZE}}, 1.0 - g / px);
  }
  return vec4f(max(col, vec3f(0.0)), depth);
}

// a value 0..1 that differs well between neighbouring pixels (interleaved gradient noise)
fn ign(p: vec2f) -> f32 { return pbr_ign(p); }   // pbr/hash.wgsl

// A point of the lens opening from two numbers in 0..1, spread evenly over its area. Round, or a regular
// polygon like the blades of a real iris (which is what gives out-of-focus highlights their shape).
fn lensPoint(q: vec2f) -> vec2f {
  if (u.blades < 3.0) { return sqrt(q.x) * vec2f(cos(6.2831853 * q.y), sin(6.2831853 * q.y)); }
  let s = q.x * u.blades;                     // which blade, and what is left over is a fresh number
  let i = floor(s);
  let a0 = u.bokeh_rot + 6.2831853 * i / u.blades;
  let a1 = u.bokeh_rot + 6.2831853 * (i + 1.0) / u.blades;
  return sqrt(s - i) * mix(vec2f(cos(a0), sin(a0)), vec2f(cos(a1), sin(a1)), q.y);      // even over that triangle
}

// Each pixel fires `ns` rays at different points within it, each carrying `nw` wavelengths.
// What lies behind curved glass is finer than a pixel (every tube is a lens squeezing the scene behind it
// into a few pixels), so where in the pixel a ray goes matters more than how finely its colour is split.
// Output: rgb = light, a = distance to the nearest glass met (0 = none), which lets the antialiasing work
// out where this piece of surface was in the previous frame, and lets a click find what to focus on.
@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  // pixel coordinates counted upward from the bottom, as the web version's are, so that every pattern
  // that depends on them (and the view itself) comes out the same way up
  let fc = vec2f(pos.x, u.res.y - pos.y);
  // The wavelengths are not random: every sample's set is shifted so that all ns x nw of them are evenly
  // spread through the spectrum, and the whole comb steps by the golden ratio each frame, so any run of
  // frames covers the spectrum evenly too.
  let xi = fract(ign(fc) + f32(u.frame % 4096) * 0.61803399);
  var lensN = 1;
  if (u.lens > 0.0) { lensN = max(u.lens_n, 1); }
  let S = clamp(u.ns * lensN, 1, 32);         // with a lens, each ray is several, spread over it
  // Positions: a well-spread sequence, shifted differently in every pixel (so that what is missed shows as
  // fine grain, not as dashes along an edge) and by the jitter every frame (so that frames add up).
  let shift = vec2f(ign(fc + 19.19), ign(fc.yx + 47.3));
  let lshift = vec2f(ign(fc + 71.7), ign(fc.yx + 113.5));
  var col = vec3f(0.0);
  var depth = 0.0;
  for (var k = 0; k < 32; k++) {
    if (k >= S) { break; }
    let o = fract(vec2f(0.7548777, 0.5698403) * f32(k) + shift + u.jitter) - 0.5;
    var lens = vec2f(0.0);
    if (u.lens > 0.0) {
      // a point of the lens opening, from another well-spread sequence that runs on from frame to frame
      lens = lensPoint(fract(lshift + vec2f(0.7548777, 0.5698403) * f32((u.frame % 512) * S + k)));
    }
    let r = sampleAt(fc + o, (f32(k) + xi) / f32(S), u.nw, lens);
    col += r.rgb;
    if (r.a > 0.0 && (depth == 0.0 || r.a < depth)) { depth = r.a; }
  }
  {{PROFILE_OUT}}return vec4f(col / f32(S), depth);
}
