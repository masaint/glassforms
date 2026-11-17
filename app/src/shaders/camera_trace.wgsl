
// ---- the first ray, through any lens ---------------------------------------------------------------------
// These three functions stand in for the tracer's own primary(), sampleAt() and lensPoint() when the lens
// is something other than the plain one. They are the same functions, statement for statement, with the
// camera's arithmetic (camera.wgsl) where the plain ones have an ordinary lens written out; with an
// ordinary lens, nothing bowed, a level plane of focus and round, even highlights they draw the same
// picture as the plain ones.

// The first glass along a ray: how far away it is, or -1.0 if there is none. Also reports the closest the
// ray came to the surface without touching it (graze, in pixel sizes at unit distance) and where (tGraze),
// for soft outlines.
fn primary(ro: vec3f, rd: vec3f, tGraze: ptr<function, f32>, graze: ptr<function, f32>) -> f32 {
  *graze = 1e9;
  *tGraze = max(-dot(ro, rd), 0.0);
  let b = dot(ro, rd);
  var h = b * b - (dot(ro, ro) - BOUND * BOUND);
  if (h <= 0.0) { return -1.0; }
  h = sqrt(h);
  var t = max(-b - h, 0.0);
  let tFar = -b + h;
  let px = u.cam.b.w / u.res.y;              // size of a pixel: at unit distance, or of an orthographic view
  for (var i = 0; i < STEPS_OUT; i++) {
    let d = mapW(ro + rd * t);
    // what a pixel covers here: it grows with distance, except through an orthographic lens
    let far = u.cam.c.x * t + u.cam.c.y;
    let r = d / far;
    if (r < *graze) { *graze = r; *tGraze = t; }
    if (d < 0.25 * px * far) {
      t += mapNW(ro + rd * t);
      t += mapNW(ro + rd * t);
      return t;
    }
    t += d * STEP_SCALE;
    if (t > tFar) { break; }
  }
  return -1.0;
}

// Where a ray through the point `l` of the lens opening starts, for a pixel whose own ray starts at ro,
// goes along rd and meets the surface in focus `reach` along it.
fn lensFrom(ro: vec3f, rd: vec3f, l: vec2f, reach: f32) -> vec3f {
  let kind = u.cam.a.x;
  if (kind < 0.5) { return ro + vec3f(l * u.lens, 0.0); }
  // an orthographic lens: every pixel's rays come in as a cone of the same angle, as wide at the surface
  // in focus as a perspective lens's would be at the middle of the band
  if (kind < 1.5) { return ro + vec3f(l * (u.lens * reach / u.cam.e.w), 0.0); }
  // a lens that sees all round has no flat front: its opening faces along each ray
  var b1 = cross(rd, vec3f(0.0, 1.0, 0.0));
  if (dot(b1, b1) < 1e-12) { b1 = vec3f(1.0, 0.0, 0.0); }
  b1 = normalize(b1);
  let b2 = cross(b1, rd);
  return ro + (b1 * l.x + b2 * l.y) * u.lens;
}

// Is the point `l` of the lens opening cut off for a pixel at `film` (camera.wgsl: camCut)?
fn lensCut(film: vec2f, l: vec2f) -> bool {
  if (u.cam.e.z <= 0.0) { return false; }
  let c = camCut(film, u.cam.e.z);
  let q = l * vec2f(u.cam.e.x, 1.0);
  return dot(q - c, q - c) > 1.0 || dot(q + c, q + c) > 1.0;
}

// One sample: the light arriving through the point fc (in pixels), carrying W wavelengths.
// lens: where on the lens this sample passes, as a point of the unit disk.
// rgb: the light.  a: how far along the pixel's own ray the glass is (0 = it misses).
fn sampleAt(fc: vec2f, xi: f32, W: i32, lens: vec2f) -> vec4f {
  let uv = (2.0 * fc - u.res) / u.res.y + u.shift;
  {{LENS_AT}}var ro: vec3f;
  var rd: vec3f;
  // (outside the circle a fisheye draws there is no picture at all)
  if (!camRay(uv, u.cam, &ro, &rd)) { return vec4f(0.0); }
  // (the lens's arithmetic is done as if it stood on the line through the object's centre; a free camera
  // stands off it by this much, which only where the rays meet the scene has to know)
  let eye = vec3f(u.eye.xy, 0.0);
  let px = u.cam.b.w / u.res.y;
  var tg: f32;
  var g: f32;
  var t = primary(ro + eye, rd, &tg, &g);
  var depth = max(t, 0.0);
  var gain = 1.0;
  if (u.lens > 0.0) {
    // Depth of field, traced rather than blurred afterwards: the ray is started from a point of the lens
    // and aimed at where the pixel's own ray meets the surface in focus. That surface is the depth the
    // pixel's own ray found, held inside the band; how depth is measured, and so what shape the surface
    // has (a plane, a leaning plane, a sphere), is the lens's own.
    // A ray through a part of the opening that the lens's barrel cuts off does not arrive; the rays that
    // do are counted for the whole of the pixel's light (what the cut costs the corners in brightness is
    // taken off the finished picture, sharp and blurred parts alike: camera_post.wgsl).
    if (lensCut(uv, lens)) { return vec4f(0.0, 0.0, 0.0, depth); }
    gain = 1.0 / camOpen(camCut(uv, u.cam.e.z));
    let e = clamp(camDepth(ro + rd * select(tg, t, t > 0.0), u.cam), u.focus.x, u.focus.y);
    let reach = camReach(ro, rd, e, u.cam);
    let f = ro + rd * reach;
    ro = lensFrom(ro, rd, lens, reach);
    rd = normalize(f - ro);
    t = primary(ro + eye, rd, &tg, &g);
    if (depth == 0.0) { depth = max(t, 0.0); }
  }
  ro += eye;
  var col = {{BACKGROUND}};                       // what the camera sees where it misses the object
  if (t > 0.0) {
    col = {{SHADE}};
  } else if (g < px) {
    // the ray grazes the silhouette within a pixel: blend in the grazing reflection (Fresnel -> 1)
    let p = ro + rd * tg;
    col = mix(col, {{GRAZE}}, 1.0 - g / px);
  }
  return vec4f(max(col, vec3f(0.0)) * gain, depth);
}

// How far from the middle of the lens opening a ray passes (0..1), from a number in 0..1. Even over the
// opening's area unless the highlights are given a bright rim or a bright core: then that share of the
// rays is drawn towards the edge (as 3 r^4) or towards the middle (as 2 (1 - r^2)) instead.
fn lensRadius(x: f32) -> f32 {
  let s = abs(u.cam.e.y);
  if (x >= s) { return sqrt((x - s) / max(1.0 - s, 1e-6)); }
  let v = x / s;
  if (u.cam.e.y > 0.0) { return pow(v, 0.16666667); }
  return sqrt(1.0 - sqrt(1.0 - v));
}

// A point of the lens opening from two numbers in 0..1. Round, or a regular polygon like the blades of a
// real iris; narrowed sideways for an anamorphic lens.
fn lensPoint(q: vec2f) -> vec2f {
  let narrow = vec2f(1.0 / u.cam.e.x, 1.0);
  if (u.blades < 3.0) { return lensRadius(q.x) * vec2f(cos(6.2831853 * q.y), sin(6.2831853 * q.y)) * narrow; }
  let s = q.x * u.blades;                     // which blade, and what is left over is a fresh number
  let i = floor(s);
  let a0 = u.bokeh_rot + 6.2831853 * i / u.blades;
  let a1 = u.bokeh_rot + 6.2831853 * (i + 1.0) / u.blades;
  return lensRadius(s - i) * mix(vec2f(cos(a0), sin(a0)), vec2f(cos(a1), sin(a1)), q.y) * narrow;
}
