// What DLSS needs beside the traced frame (dlss.rs): for every pixel of the small traced picture, where its
// content was in the previous frame, as an offset in the small picture's pixels (top row first: x right,
// y down), and how near it is (larger = nearer, 0 = nothing there).
//
// Where the content was is worked out exactly as the antialiasing that follows motion works it out
// (accum.wgsl): the glass is rigid, so from the distance traced to it and the frame of the object now and
// then the same point is found on the old screen; the environment picture behind the shape turns with the
// lights; the floor stands in the room. What is seen THROUGH the glass moves otherwise than the glass's
// surface does, and there is no single offset that is right for it: the surface's is given.

struct A {
  res: vec2f,
  cam: vec2f,
  cam_p: vec2f,
  shift: vec2f,
  shift_p: vec2f,
  w: f32,
  moving: f32,
  obj0: vec4f,
  obj1: vec4f,
  obj2: vec4f,
  objp0: vec4f,
  objp1: vec4f,
  objp2: vec4f,
  debug: f32,
  spread: f32,
  sky: f32,
  pad1: f32,
  env0: vec4f,
  env1: vec4f,
  env2: vec4f,
  envp0: vec4f,
  envp1: vec4f,
  envp2: vec4f,
  room0: vec4f,
  room1: vec4f,
  room2: vec4f,
  roomp0: vec4f,
  roomp1: vec4f,
  roomp2: vec4f,
  floor: vec4f,
  eye: vec4f,
}

@group(0) @binding(0) var<uniform> a: A;
@group(0) @binding(1) var curTex: texture_2d<f32>;

struct Out {
  @location(0) mv: vec4f,
  @location(1) depth: vec4f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> Out {
  let c = vec2i(pos.xy);
  let fc = vec2f(pos.x, a.res.y - pos.y);
  let d = textureLoad(curTex, c, 0).a;
  var at = fc;
  if (d > 0.0) {
    let uv = (2.0 * fc - a.res) / a.res.y + a.shift;
    let p = vec3f(a.eye.xy, a.cam.x) + normalize(vec3f(uv * a.cam.y, -1.0)) * d;
    let o = p.x * a.obj0.xyz + p.y * a.obj1.xyz + p.z * a.obj2.xyz;
    let v = vec3f(dot(a.objp0.xyz, o), dot(a.objp1.xyz, o), dot(a.objp2.xyz, o)) - vec3f(a.eye.zw, a.cam_p.x);
    at = ((v.xy / (-v.z * a.cam_p.y) - a.shift_p) * a.res.y + a.res) * 0.5;
  } else if (a.sky > 0.5) {
    let uv = (2.0 * fc - a.res) / a.res.y + a.shift;
    let dir = normalize(vec3f(uv * a.cam.y, -1.0));
    let e = dir.x * a.env0.xyz + dir.y * a.env1.xyz + dir.z * a.env2.xyz;
    let v = vec3f(dot(a.envp0.xyz, e), dot(a.envp1.xyz, e), dot(a.envp2.xyz, e));
    if (v.z < -1e-4) { at = ((v.xy / (-v.z * a.cam_p.y) - a.shift_p) * a.res.y + a.res) * 0.5; }
  }
  if (a.floor.x > 0.5 && d > 0.0) {
    // the floor: the traced distance is the distance at which a ray through some point of this pixel meets
    // the floor's plane (as in accum.wgsl, tested against the four corners of the pixel)
    let eye = vec3f(a.eye.xy, a.cam.x);
    let n = vec3f(a.room0.y, a.room1.y, a.room2.y);
    let drop = a.floor.y - dot(n, eye);
    var lo = 1e9;
    var hi = 0.0;
    for (var k = 0; k < 4; k++) {
      let corner = fc + vec2f(f32(k & 1), f32(k >> 1)) - 0.5;
      let uvk = (2.0 * corner - a.res) / a.res.y + a.shift;
      let dn = dot(n, normalize(vec3f(uvk * a.cam.y, -1.0)));
      let t = select(-1.0, drop / dn, dn < -1e-5);       // (a level or rising ray meets no floor)
      if (t > 0.0) { lo = min(lo, t); hi = max(hi, t); } else { hi = 1e9; }
    }
    if (d > lo * 0.996 - 0.002 && d < hi * 1.004 + 0.002) {
      at = fc;
      let uv = (2.0 * fc - a.res) / a.res.y + a.shift;
      let dir = normalize(vec3f(uv * a.cam.y, -1.0));
      let dn = dot(n, dir);
      let tf = select(-1.0, drop / dn, dn < -1e-5);
      if (tf > 0.0) {
        let p = eye + dir * tf;
        var q = p.x * a.room0.xyz + p.y * a.room1.xyz + p.z * a.room2.xyz;
        q.y = a.floor.z;
        let v = vec3f(dot(a.roomp0.xyz, q), dot(a.roomp1.xyz, q), dot(a.roomp2.xyz, q)) - vec3f(a.eye.zw, a.cam_p.x);
        if (v.z < -1e-4) { at = ((v.xy / (-v.z * a.cam_p.y) - a.shift_p) * a.res.y + a.res) * 0.5; }
      }
    }
  }
  // (counted from the top, as the pictures are stored)
  let was = vec2f(at.x, a.res.y - at.y);
  var out: Out;
  out.mv = vec4f(was - pos.xy, 0.0, 0.0);
  // nearer is larger (DLSS is told the depth is inverted); nothing there is 0, as far as can be
  out.depth = vec4f(select(0.0, 0.05 / max(d, 0.05), d > 0.0), 0.0, 0.0, 0.0);
  return out;
}
