// An environment picture made ready for the tracer, and the small picture of the lighting a light room shows.
//
// The picture arrives as a panorama (equirectangular: its middle is the direction the camera looks in
// when nothing is turned, its top row straight up). The tracer reads it from a cube instead, because a
// cube has no poles and no seam, and the GPU blends across its edges by itself:
//   fs_face   one face of the cube from the panorama, as sharp as the panorama is
//   fs_soft   one face of the next smaller level from the level above it, softened by a bell-shaped
//             blur measured as an angle on the sphere (so the softening is the same in every direction
//             and everywhere on the cube). Level after level this makes a chain of ever softer copies;
//             the tracer picks how soft a copy the glass reads.
//   fs_glass  one face of the cube the glass reads: the picture at the softness and brightness set, with
//             the lights' map added in, so that the tracer finds all its light with one lookup.
//   fs_show   the lighting as a small panorama for display: the lights' map and the picture together,
//             turned as the lights are turned.

struct B {
  size: f32,              // texels along a side of the face being written
  taps: f32,              // fs_face: samples a side within one texel.  fs_soft: samples in the bell
  sigma: f32,             // fs_soft: the bell's width, radians
  pad: f32,
}
@group(0) @binding(0) var<uniform> b: B;
@group(0) @binding(1) var panorama: texture_2d<f32>;
@group(0) @binding(2) var wrap: sampler;                 // repeats round the horizon, holds at the poles
@group(0) @binding(3) var above: texture_cube<f32>;      // fs_soft: the level above the one being written
@group(0) @binding(4) var blend: sampler;

struct V {
  @builtin(position) pos: vec4f,
  @location(0) @interpolate(flat) face: u32,
}

// One triangle over the whole face; which face comes as the instance number.
@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) face: u32) -> V {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return V(vec4f(p * 2.0 - 1.0, 0.0, 1.0), face);
}

// The direction a point of a cube face stands for. st runs from -1 to 1 across the face, t downward
// (the same way texel rows are stored), and the faces are in the GPU's order: +x, -x, +y, -y, +z, -z.
fn faceDir(face: u32, st: vec2f) -> vec3f {
  let s = st.x;
  let t = st.y;
  switch face {
    case 0u { return vec3f(1.0, -t, -s); }
    case 1u { return vec3f(-1.0, -t, s); }
    case 2u { return vec3f(s, 1.0, t); }
    case 3u { return vec3f(s, -1.0, -t); }
    case 4u { return vec3f(s, -t, 1.0); }
    default { return vec3f(-s, -t, -1.0); }
  }
}

// Where a direction (in the lights' own frame) falls on the panorama: its middle is -z, to the right
// of that is +x, the top row is +y.
fn panoramaUv(d: vec3f) -> vec2f {
  return vec2f(atan2(d.x, -d.z) * 0.15915494 + 0.5, acos(clamp(d.y, -1.0, 1.0)) * 0.31830989);
}

@fragment
fn fs_face(v: V) -> @location(0) vec4f {
  let n = i32(b.taps);
  var sum = vec3f(0.0);
  for (var j = 0; j < 4; j++) {
    if (j >= n) { break; }
    for (var i = 0; i < 4; i++) {
      if (i >= n) { break; }
      let at = v.pos.xy - 0.5 + (vec2f(f32(i), f32(j)) + 0.5) / f32(n);
      let d = normalize(faceDir(v.face, at / b.size * 2.0 - 1.0));
      sum += textureSampleLevel(panorama, wrap, panoramaUv(d), 0.0).rgb;
    }
  }
  return vec4f(sum / f32(n * n), 1.0);
}

@fragment
fn fs_soft(v: V) -> @location(0) vec4f {
  let d = normalize(faceDir(v.face, v.pos.xy / b.size * 2.0 - 1.0));
  // two directions across d
  let t1 = normalize(cross(select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(d.y) > 0.9), d));
  let t2 = cross(d, t1);
  // Samples spread evenly over a disc of angle around d (a sunflower's spiral), out to where the bell has
  // all but died away, each weighed by the bell. The disc is laid on the sphere, where a ring at angle r
  // is sin(r) / r as long as it is on the flat: the weight makes up for that.
  let n = i32(b.taps);
  // (one sample: the level above as it is at this spot, which at the middle of a texel of a level half
  // the size is the exact average of the four texels under it)
  if (n <= 1) { return vec4f(textureSampleLevel(above, blend, d, 0.0).rgb, 1.0); }
  let reach = min(2.6 * b.sigma, 3.1);
  var sum = vec3f(0.0);
  var wsum = 0.0;
  for (var i = 0; i < 160; i++) {
    if (i >= n) { break; }
    let r = reach * sqrt((f32(i) + 0.5) / f32(n));
    let a = f32(i) * 2.3999632;
    let w = exp(-0.5 * r * r / (b.sigma * b.sigma)) * sin(r) / r;
    let dir = d * cos(r) + (t1 * cos(a) + t2 * sin(a)) * sin(r);
    sum += w * textureSampleLevel(above, blend, dir, 0.0).rgb;
    wsum += w;
  }
  return vec4f(sum / wsum, 1.0);
}

struct S {
  env0: vec4f,            // what the camera sees -> the lights' own frame, as the tracer has it
  env1: vec4f,
  env2: vec4f,
  k: vec4f,               // x: 1 = the lights' map is lit; y: brightness of the picture (0 = none);
                          // z: the level of the picture shown; w: unused
  size: vec4f,            // xy: width and height of what is being drawn
}
@group(0) @binding(5) var<uniform> s: S;
@group(0) @binding(6) var lightsMap: texture_2d<f32>;

// The lights' map and the picture, added, in direction d of the lights' own frame.
fn lighting(d: vec3f) -> vec3f {
  var c = vec3f(0.0);
  if (s.k.x > 0.5) {
    // the lights' map, looked up as the tracer does
    let m = vec2f(atan2(d.z, d.x) * 0.15915494 + 0.5, acos(clamp(d.y, -1.0, 1.0)) * 0.31830989);
    c += textureSampleLevel(lightsMap, wrap, m, 0.0).rgb;
  }
  if (s.k.y > 0.0) {
    c += textureSampleLevel(above, blend, d, s.k.z).rgb * s.k.y;
  }
  return c;
}

@fragment
fn fs_glass(v: V) -> @location(0) vec4f {
  return vec4f(lighting(normalize(faceDir(v.face, v.pos.xy / s.size.x * 2.0 - 1.0))), 1.0);
}

@vertex
fn vs_show(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_show(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  // the direction this point of the small picture shows (env.rs: `preview_dir`), then into the lights' frame
  let uv = pos.xy / s.size.xy;
  let th = uv.y * 3.14159265;
  let ph = (uv.x - 0.5) * 6.2831853;
  let w = vec3f(sin(th) * sin(ph), cos(th), -sin(th) * cos(ph));
  let d = w.x * s.env0.xyz + w.y * s.env1.xyz + w.z * s.env2.xyz;
  var c = lighting(d);
  // For looking at, not for measuring: bright lights are pressed down without losing their colour
  // (each colour divided by one plus the brightness), then encoded for the screen.
  c = c / (1.0 + dot(c, vec3f(0.2126, 0.7152, 0.0722)));
  c = mix(12.92 * c, 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, step(vec3f(0.0031308), c));
  return vec4f(c, 1.0);
}
