  // (the matcap view, matcap.rs: put in place of the tracer's own entry, so that `primary` and `nrm` are
  // the live tracer's, formula and all; nothing else of the tracer is called)
  let fc = vec2f(pos.x, u.res.y - pos.y) + u.jitter;
  let film = (2.0 * fc - u.res) / u.res.y + u.shift;
  // the camera: a pinhole as the plain tracer's (from the eye off the axis, where there is one), or an
  // orthographic lens (camera.wgsl camRay: kind 1, its half height in cam.w); others are drawn as the pinhole
  let ortho = mc.cam.x > 0.5 && mc.cam.x < 1.5;
  var ro = vec3f(0.0, 0.0, u.dist);
  var rd = normalize(vec3f(film * u.tan_fov, -1.0));
  if (ortho) {
    ro = vec3f(film * mc.cam.w + u.eye.xy, mc.cam.z);
    rd = vec3f(0.0, 0.0, -1.0);
  }
  // what a pixel spans at distance t along its ray
  let pxa = 2.0 * u.tan_fov / u.res.y;
  let pxo = 2.0 * mc.cam.w / u.res.y;
  var tg: f32;
  var g: f32;
  let t = primary(ro, rd, &tg, &g);
  let hit = t > 0.0;
  // what is behind: a plain gradient, darker below
  let bg = mix(vec3f(0.030, 0.031, 0.034), vec3f(0.105, 0.108, 0.115), clamp(0.5 + 0.5 * film.y, 0.0, 1.0));
  // (worked out for every pixel, hit or not, so that the screen derivatives below are taken where all
  // four of a quad's pixels take them)
  let p = ro + rd * select(tg, t, hit);
  var n = -rd;
  if (hit) { n = nrm(p); }
  n = faceIncident(n, -rd);
  let hitF = select(0.0, 1.0, hit);
  let dn = vec3f(0.0) + dpdx(n);
  let dm = dpdy(n);
  let dpx = dpdx(p);
  let dpy = dpdy(p);
  let edgeOfHit = fwidth(hitF);
  let dt = fwidth(select(0.0, t, hit));
  // the share of a pixel by which a ray that missed passed the surface (the tracer's own coverage)
  let foot = select(tg * pxa, pxo, ortho);
  let graze = g * tg / max(foot, 1e-9);
  var col = bg;
  if (hit) {
    col = mcLook(n, rd, p);
    // cavity: the distance field along the normal, within 0.03 of the surface (further out a tube's
    // formula is no distance: tube_field_fix.md)
    if (mc.fx.x > 0.0) {
      var occ = 0.0;
      let hs = vec4f(0.004, 0.010, 0.018, 0.030);
      for (var i = 0; i < 4; i++) {
        let h = hs[i];
        occ += max(h - mapW(p + n * h), 0.0) / h * (0.5 - 0.1 * f32(i));
      }
      col *= 1.0 - mc.fx.x * clamp(occ * 0.75, 0.0, 0.85);
    }
    // edge: how fast the normal turns across the screen, against how far the surface moves (curvature, 1 / length),
    // signed: convex lifted, concave darkened; not across the silhouette, where the neighbours missed
    if (mc.fx.y > 0.0 && edgeOfHit == 0.0) {
      let kx = dot(dn, dpx) / max(dot(dpx, dpx), 1e-12);
      let ky = dot(dm, dpy) / max(dot(dpy, dpy), 1e-12);
      let k = clamp(0.5 * (kx + ky) * 0.02, -1.0, 1.0);
      col *= 1.0 + mc.fx.y * select(0.9, 0.6, k > 0.0) * k;
    }
  } else if (graze < 1.0) {
    let pg = ro + rd * tg;
    let ng = faceIncident(nrm(pg), -rd);
    col = mix(bg, mcLook(ng, rd, pg), 1.0 - graze);
  }
  // outline: where the depth jumps by more than a slope facing this way would give, and the silhouette
  if (mc.fx.z > 0.0) {
    let expect = select(1e9, select(t * pxa, pxo, ortho) / max(dot(n, -rd), 0.08), hit);
    let jump = mcStep(2.5, 5.0, dt / expect);
    let line = max(jump, mcStep(0.0, 1.0, edgeOfHit) * select(0.0, 1.0, hit || graze < 1.0));
    col = mix(col, vec3f(0.006), mc.fx.z * line);
  }
  // the picture so far: a running mean of jittered frames while still (the first frame weighs 1)
  var depth = select(0.0, t, hit);
  let w = mc.acc.x;
  if (w < 1.0) {
    let was = textureLoad(prevTex, vec2i(pos.xy), 0);
    col = mix(was.rgb, col, w);
    if (!hit) { depth = was.a; }
  }
  return vec4f(max(col, vec3f(0.0)), depth);
}

struct McU {
  base: vec4f,             // w: 0 a formula (`look`), 100 a picture (a user's, or the material baked: matcap.rs)
  fx: vec4f,               // cavity, edge, outline, unused
  acc: vec4f,              // x: this frame's weight in the running mean; y: bodies to tell apart (0: none)
  tints: array<vec4f, 4>,  // each body's colour
  turn: vec4f,             // the light's turn: cos, sin of minus the turn; cos, sin of minus the lift
  cam: vec4f,              // the lens (camera.wgsl Cam.a): kind, half height in focal lengths, distance, ortho half height
  look: array<vec4f, 12>,  // the formula's numbers (gf_core::view_mode::Look, packed by matcap.rs `pack`)
}
@group(1) @binding(0) var<uniform> mc: McU;
@group(1) @binding(1) var prevTex: texture_2d<f32>;
@group(1) @binding(2) var capTex: texture_2d<f32>;
@group(1) @binding(3) var capSmp: sampler;
@group(0) @binding(12) var bodTex: texture_3d<f32>;

// The normal in the camera's frame turned by the smallest rotation that takes the ray to straight ahead
// (gf_core::view_mode::along_ray): a ball reads the same matcap anywhere on the screen.
fn mcAlong(n: vec3f, rd: vec3f) -> vec3f {
  let a = -rd;
  let c = a.z;
  let k = vec3f(a.y, -a.x, 0.0);
  return n * c + cross(k, n) + k * (dot(k, n) / (1.0 + c));
}

// gf_core::view_mode::turned
fn mcTurned(n: vec3f) -> vec3f {
  let a = vec3f(mc.turn.x * n.x - mc.turn.y * n.y, mc.turn.y * n.x + mc.turn.x * n.y, n.z);
  return vec3f(a.x, mc.turn.z * a.y - mc.turn.w * a.z, mc.turn.w * a.y + mc.turn.z * a.z);
}

// smoothstep that also runs downhill (edges given high to low), as the f64 copy does
fn mcStep(a: f32, b: f32, x: f32) -> f32 {
  let t = clamp((x - a) / (b - a), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

fn mcHighlight(n: vec3f, l: vec3f, power: f32) -> f32 {
  let h = normalize(vec3f(l.xy, l.z + 1.0));
  return pow(clamp(dot(n, h), 0.0, 1.0), power);
}

// A formula matcap: gf_core::view_mode::shade_look, line for line (directions arrive unit length).
fn mcShade(n0: vec3f) -> vec3f {
  let n = normalize(n0);
  if (mc.look[7].w > 0.5) {
    let e = 0.5 + 0.5 * n;
    return select(pow((e + 0.055) / 1.055, vec3f(2.4)), e / 12.92, e <= vec3f(0.04045));
  }
  let g = 1.0 - max(n.z, 0.0);
  let fres = g * g * g * g;
  let key = mc.look[1].xyz;
  let wrap = mc.look[1].w;
  let kd = clamp((dot(n, key) + wrap) / (1.0 + wrap), 0.0, 1.0);
  let fd = clamp((dot(n, mc.look[3].xyz) + 0.5) / 1.5, 0.0, 1.0);
  let light = mc.look[0].w + mc.look[2].w * (0.5 + 0.5 * n.y) + mc.look[3].w * fd;
  var c = mc.look[0].xyz * (light + mc.look[2].xyz * kd);
  let ks = mc.look[4].y * mcHighlight(n, key, mc.look[4].x);
  let s2 = mc.look[4].w * mcHighlight(n, mc.look[5].xyz, mc.look[4].z);
  let rim = mc.look[6].w * mcStep(mc.look[7].x, 0.0, n.z);
  let ry = 2.0 * n.z * n.y;
  let sky = mix(mc.look[9].xyz, mc.look[8].xyz, mcStep(0.0, 0.8, ry));
  let ground = mix(mc.look[10].xyz, mc.look[11].xyz, mcStep(0.0, -0.8, ry));
  let ramp = mix(ground, sky, mcStep(-mc.look[7].z, mc.look[7].z, ry)) * (mc.look[7].y * (0.85 + 0.15 * fres));
  c += mc.look[2].xyz * ks + vec3f(s2) + mc.look[6].xyz * rim + vec3f(mc.look[5].w * fres) + ramp;
  return c;
}

// The matcap's colour for a surface at p facing n, seen along rd (tinted by its body when asked).
fn mcLook(n: vec3f, rd: vec3f, p: vec3f) -> vec3f {
  let m = mcTurned(mcAlong(n, rd));
  var c: vec3f;
  if (mc.base.w >= 100.0) {
    // a picture of a lit ball filling the square, up at the top, its rim pulled in a little (the rim is
    // where a ball's picture is thinnest: grazing normals would read its last few texels)
    let mz = normalize(vec3f(m.xy, max(m.z, 0.0)));
    let st = 0.5 + 0.5 * 0.985 * vec2f(mz.x, -mz.y);
    c = textureSampleLevel(capTex, capSmp, st, 0.0).rgb;
  } else {
    c = mcShade(m);
  }
  let count = i32(mc.acc.y);
  if (count > 1 && mc.look[7].w < 0.5) {
    let d = textureSampleLevel(bodTex, volSmp, toObj(p) / (2.0 * HB) + 0.5, 0.0);
    var best = 0;
    for (var i = 1; i < 4; i++) {
      if (i < count && d[i] < d[best]) { best = i; }
    }
    c *= mc.tints[best].rgb;
  }
  return c;
}
