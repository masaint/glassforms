// Blend this frame into the picture so far. (FS_ACCUM of the web version.)
//   still view:   a plain running average of jittered frames (w = 1/n), which converges to a clean image.
//   moving view:  temporal antialiasing that follows the surface. The shape is rigid, so from the distance
//                 stored with each pixel we know which point of the glass it shows, and from the previous
//                 frame's rotation where that point was on screen then. The old picture is read THERE, so
//                 what gets blended is the same piece of glass, and many frames can be kept without smear.
//                 The refracted pattern is not quite fixed to the surface, but it drifts across it by only
//                 a fraction of a pixel per frame.
//                 Where that fails (the point was hidden or off screen last frame: the stored distance
//                 does not match) it falls back to the old picture at the same place, limited to the range
//                 of colours around this pixel in the new frame.
//                 Where there is no glass the pixel shows nothing, or, when an environment picture is
//                 shown behind the shape, a piece of that picture. The picture is infinitely far away and
//                 turns with the lights, so from the lights' frame now and at the previous frame (and the
//                 lens and view shift then) it is known where that piece was, and the old picture is read
//                 there: the backdrop stays sharp while the lights turn, the lens zooms or the view slides.
//                 A floor is neither: it has a distance like the glass, but it stands in the room and does
//                 not turn with the shape. A pixel shows the floor when the distance stored with it is the
//                 distance at which its own ray meets the floor's plane; such a pixel is followed through
//                 the room (which only moves when the camera does), so the floor stays still and sharp
//                 while the shape turns on it.

struct A {
  res: vec2f,
  cam: vec2f,             // camera distance and tan(half field of view): now ...
  cam_p: vec2f,           // ... and at the previous frame
  shift: vec2f,
  shift_p: vec2f,
  w: f32,
  moving: f32,
  obj0: vec4f,            // object -> world now, as three rows
  obj1: vec4f,
  obj2: vec4f,
  objp0: vec4f,           // object -> world at the previous frame
  objp1: vec4f,
  objp2: vec4f,
  debug: f32,
  spread: f32,
  sky: f32,               // 1 = an environment picture is shown where there is no glass
  pad1: f32,
  env0: vec4f,            // what the camera sees -> the lights' own frame now, as three rows
  env1: vec4f,
  env2: vec4f,
  envp0: vec4f,           // and at the previous frame
  envp1: vec4f,
  envp2: vec4f,
  room0: vec4f,           // the room -> what the camera sees now, as three rows
  room1: vec4f,
  room2: vec4f,
  roomp0: vec4f,          // and at the previous frame
  roomp1: vec4f,
  roomp2: vec4f,
  floor: vec4f,           // x: 1 = there is a floor, level in the room;  y: its height in the room now;
                          // z: its height at the previous frame
}

@group(0) @binding(0) var<uniform> a: A;
@group(0) @binding(1) var prevTex: texture_2d<f32>;
@group(0) @binding(2) var curTex: texture_2d<f32>;
@group(0) @binding(3) var lin: sampler;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

// Read the old picture at a point between pixels. Plain linear interpolation softens it a little, and since
// the result is read again next frame the softening compounds: one-pixel lines are gone within a dozen
// frames. A Catmull-Rom filter keeps them (nine linear lookups, after Jimenez / Pettineo). Its negative
// lobes can ring around bright highlights, so the result is held within the four pixels it lies between.
fn sharp(at: vec2f) -> vec3f {
  let p1 = floor(at - 0.5) + 0.5;
  let f = at - p1;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let t0 = (p1 - 1.0) / a.res;
  let t3 = (p1 + 2.0) / a.res;
  let t12 = (p1 + w2 / w12) / a.res;
  let r = textureSampleLevel(prevTex, lin, vec2f(t0.x,  t0.y),  0.0).rgb * (w0.x  * w0.y)
        + textureSampleLevel(prevTex, lin, vec2f(t12.x, t0.y),  0.0).rgb * (w12.x * w0.y)
        + textureSampleLevel(prevTex, lin, vec2f(t3.x,  t0.y),  0.0).rgb * (w3.x  * w0.y)
        + textureSampleLevel(prevTex, lin, vec2f(t0.x,  t12.y), 0.0).rgb * (w0.x  * w12.y)
        + textureSampleLevel(prevTex, lin, vec2f(t12.x, t12.y), 0.0).rgb * (w12.x * w12.y)
        + textureSampleLevel(prevTex, lin, vec2f(t3.x,  t12.y), 0.0).rgb * (w3.x  * w12.y)
        + textureSampleLevel(prevTex, lin, vec2f(t0.x,  t3.y),  0.0).rgb * (w0.x  * w3.y)
        + textureSampleLevel(prevTex, lin, vec2f(t12.x, t3.y),  0.0).rgb * (w12.x * w3.y)
        + textureSampleLevel(prevTex, lin, vec2f(t3.x,  t3.y),  0.0).rgb * (w3.x  * w3.y);
  let i = vec2i(p1);
  let top = vec2i(textureDimensions(prevTex)) - 1;
  let c0 = textureLoad(prevTex, clamp(i, vec2i(0), top), 0).rgb;
  let c1 = textureLoad(prevTex, clamp(i + vec2i(1, 0), vec2i(0), top), 0).rgb;
  let c2 = textureLoad(prevTex, clamp(i + vec2i(0, 1), vec2i(0), top), 0).rgb;
  let c3 = textureLoad(prevTex, clamp(i + vec2i(1, 1), vec2i(0), top), 0).rgb;
  return clamp(r, min(min(c0, c1), min(c2, c3)), max(max(c0, c1), max(c2, c3)));
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let c = vec2i(pos.xy);
  // pixel coordinates counted upward from the bottom, as the web version's are (the view's maths uses them)
  let fc = vec2f(pos.x, a.res.y - pos.y);
  let cur = textureLoad(curTex, c, 0);
  if (a.moving < 0.5) {
    return vec4f(mix(textureLoad(prevTex, c, 0).rgb, cur.rgb, a.w), cur.a);
  }
  // The new frame around this pixel: its colours (mean and spread), and how far away the glass is.
  let top = vec2i(textureDimensions(curTex)) - 1;
  var m1 = vec3f(0.0);
  var m2 = vec3f(0.0);
  var near = 1e9;
  var far = 0.0;
  var misses = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let q = textureLoad(curTex, clamp(c + vec2i(x, y), vec2i(0), top), 0);
      m1 += q.rgb;
      m2 += q.rgb * q.rgb;
      if (q.a > 0.0) { near = min(near, q.a); far = max(far, q.a); } else { misses += 1.0; }
    }
  }
  m1 /= 9.0;
  let dev = sqrt(max(m2 / 9.0 - m1 * m1, vec3f(0.0)));
  // An outline: glass next to background, or next to glass much further away. It moves with whatever is
  // in front, so it is followed using the nearest distance around it, whichever side this pixel's own ray
  // happened to land on. That keeps outlines smooth instead of flipping between two sources.
  let edge = far > 0.0 && (misses > 0.0 || far - near > 0.01 + 0.004 * near);
  let d = select(cur.a, near, edge);

  var at = fc;                       // where this pixel's content was in the previous frame (the sky does not move)
  var expect = 0.0;                  // and how far from the camera it was then
  if (d > 0.0) {
    let uv = (2.0 * fc - a.res) / a.res.y + a.shift;
    let p = vec3f(0.0, 0.0, a.cam.x) + normalize(vec3f(uv * a.cam.y, -1.0)) * d;
    let o = p.x * a.obj0.xyz + p.y * a.obj1.xyz + p.z * a.obj2.xyz;                    // world -> object now
    let v = vec3f(dot(a.objp0.xyz, o), dot(a.objp1.xyz, o), dot(a.objp2.xyz, o)) - vec3f(0.0, 0.0, a.cam_p.x);
    at = ((v.xy / (-v.z * a.cam_p.y) - a.shift_p) * a.res.y + a.res) * 0.5;
    expect = length(v);
  } else if (a.sky > 0.5) {
    // the piece of the picture behind the shape that this pixel shows: its direction, taken into the
    // lights' frame as it is now and back out of it as it was (it has no distance: expect stays 0)
    let uv = (2.0 * fc - a.res) / a.res.y + a.shift;
    let dir = normalize(vec3f(uv * a.cam.y, -1.0));
    let e = dir.x * a.env0.xyz + dir.y * a.env1.xyz + dir.z * a.env2.xyz;
    let v = vec3f(dot(a.envp0.xyz, e), dot(a.envp1.xyz, e), dot(a.envp2.xyz, e));
    // (if it was behind the camera then, there is no old picture of it: a place off the screen says so)
    at = select(vec2f(-1.0), ((v.xy / (-v.z * a.cam_p.y) - a.shift_p) * a.res.y + a.res) * 0.5, v.z < -1e-4);
  }
  var onFloor = false;               // this pixel shows the floor
  var nearest = 0.0;                 // and the nearest the floor comes within the pixel
  if (a.floor.x > 0.5 && cur.a > 0.0) {
    let eye = vec3f(0.0, 0.0, a.cam.x);
    let n = vec3f(a.room0.y, a.room1.y, a.room2.y);            // the floor's normal as the camera sees it
    let drop = a.floor.y - dot(n, eye);                         // how far the floor lies from the camera, along that normal
    // Where rays through the four corners of the pixel meet the floor. The pixel's own ray went through
    // some point of the pixel, not its middle, and towards the horizon the floor's distance changes a
    // great deal within one pixel: the stored distance is the floor's if it lies between these.
    var lo = 1e9;
    var hi = 0.0;
    for (var k = 0; k < 4; k++) {
      let corner = fc + vec2f(f32(k & 1), f32(k >> 1)) - 0.5;
      let uvk = (2.0 * corner - a.res) / a.res.y + a.shift;
      let t = drop / dot(n, normalize(vec3f(uvk * a.cam.y, -1.0)));
      if (t > 0.0) { lo = min(lo, t); hi = max(hi, t); } else { hi = 1e9; }
    }
    // (at the outline of glass that stands in front of the floor the glass is followed, as at any outline)
    if (cur.a > lo * 0.996 - 0.002 && cur.a < hi * 1.004 + 0.002 && !(edge && d < lo * 0.95)) {
      onFloor = true;
      nearest = lo;
      at = fc;
      expect = cur.a;
      // the spot of the floor under the middle of the pixel, and where the camera saw it at the previous
      // frame (the floor stands in the room: it moves in the picture only when the camera does)
      let uv = (2.0 * fc - a.res) / a.res.y + a.shift;
      let dir = normalize(vec3f(uv * a.cam.y, -1.0));
      let tf = drop / dot(n, dir);
      if (tf > 0.0) {
        let p = eye + dir * tf;
        var q = p.x * a.room0.xyz + p.y * a.room1.xyz + p.z * a.room2.xyz;    // what the camera sees -> the room
        q.y = a.floor.z;                                                      // the same spot, at the height the floor had then
        let v = vec3f(dot(a.roomp0.xyz, q), dot(a.roomp1.xyz, q), dot(a.roomp2.xyz, q)) - vec3f(0.0, 0.0, a.cam_p.x);
        at = select(vec2f(-1.0), ((v.xy / (-v.z * a.cam_p.y) - a.shift_p) * a.res.y + a.res) * 0.5, v.z < -1e-4);
      }
    }
  }
  let inside = all(at >= vec2f(0.5)) && all(at <= a.res - 0.5);
  let att = vec2f(at.x, a.res.y - at.y);       // the same spot counted from the top, as the pictures are stored
  var same = false;                  // was the old picture found for this piece of surface?
  var hist = vec3f(0.0);
  if (inside && onFloor) {
    // the old picture there is the same piece of floor unless glass stood in front of it then
    let was = textureLoad(prevTex, clamp(vec2i(att), vec2i(0), vec2i(textureDimensions(prevTex)) - 1), 0).a;
    if (!(was > 0.0 && was < nearest * 0.95)) {
      hist = sharp(att);
      same = true;
    }
  } else if (inside && edge) {
    hist = sharp(att);
    same = true;
  } else if (inside) {
    // Away from outlines the stored distance says whether the old picture there is the same piece of glass
    // (it is not if that piece was hidden last frame). The four pixels around the spot are tested one by one.
    let i0 = vec2i(floor(att - 0.5));
    let lim = vec2i(textureDimensions(prevTex)) - 1;
    let f = att - 0.5 - vec2f(i0);
    let tol = 0.006 + 0.003 * expect;                           // tight: strands can pass a few hundredths apart
    let h0 = textureLoad(prevTex, clamp(i0, vec2i(0), lim), 0);
    let h1 = textureLoad(prevTex, clamp(i0 + vec2i(1, 0), vec2i(0), lim), 0);
    let h2 = textureLoad(prevTex, clamp(i0 + vec2i(0, 1), vec2i(0), lim), 0);
    let h3 = textureLoad(prevTex, clamp(i0 + vec2i(1, 1), vec2i(0), lim), 0);
    let ok = vec4f(abs(vec4f(h0.a, h1.a, h2.a, h3.a) - expect) < vec4f(tol));
    let wv = vec4f((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y) * ok;
    let ws = wv.x + wv.y + wv.z + wv.w;
    if (ok.x + ok.y + ok.z + ok.w > 3.5) {                       // all four: the sharp read
      hist = sharp(att);
      same = true;
    } else if (ws > 0.02) {                                      // some: only those
      hist = (h0.rgb * wv.x + h1.rgb * wv.y + h2.rgb * wv.z + h3.rgb * wv.w) / ws;
      same = true;
    }
  }
  if (!same) { hist = textureLoad(prevTex, c, 0).rgb; }
  // Reflections are not painted on the glass: on a flattish piece a highlight can sweep across the surface
  // far faster than the surface moves, and the old picture, carried along with the surface, would trail
  // behind it. So the old picture is limited to what this pixel's neighbourhood in the new frame makes
  // plausible: its mean, give or take a multiple of its spread. Where the neighbourhood is smooth the limit
  // is tight and stale light is cut off at once; where it is busy (sub-pixel detail, where single frames are
  // noisy) the limit is wide and the accumulated picture survives. Unfollowed pixels get the tighter
  // multiple of one.
  let k = select(1.0, a.spread, same);
  hist = clamp(hist, m1 - dev * k, m1 + dev * k);
  var out = vec4f(mix(hist, cur.rgb, a.w), cur.a);
  // debug view: red = the old picture could not be followed, blue = outline
  if (a.debug > 0.5) {
    if (!same) { out = vec4f(1.0, 0.0, 0.0, out.a); }
    else if (edge) { out = vec4f(0.0, 0.3, 1.0, out.a); }
    else { out = vec4f(0.0, 0.05, 0.0, out.a); }
  }
  return out;
}
