// The guide pass of the Ray Reconstruction draft (dlss_rr.rs), appended to the centred tracer's text: one pass, four targets,
// for the first ray of every pixel (the same sample place as the colour pass): a = first glass hit (xyz, 1 = glass met),
// b = where the refracted ray first leaves the glass (xyz) and the way inside, c = its normal there (facing the eye) and the
// reflectance, d.x = the distance from there to the next glass (-1 = none). Same maths as the lab's guide passes.

struct GuideOut { @location(0) a: vec4f, @location(1) b: vec4f, @location(2) c: vec4f, @location(3) d: vec4f }
struct Geo { p0: vec3f, hit: f32, n0: vec3f, f0: f32, q: vec3f, s: f32, nq: vec3f, f2: f32, od: vec3f, tir: f32, rend: vec3f, lbey: f32 }
fn beyondDir(ro_in: vec3f, rd_in: vec3f, ior: f32) -> vec4f {
  let px = 2.0 * u.tan_fov / u.res.y;
  var ro = ro_in;
  var rd = rd_in;
  var first = -1.0;
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
    if (first < 0.0) { first = t; }
    let p = ro + rd * t;
    let n = faceIncident(nrm(p), -rd);
    let entry = pbr_dielectric_split(traceDielectricHit(n, airMedium(), resolvedSceneMedium(ior)), -rd);
    let d2 = entry.transmitted;
    let q0 = p - n * 0.004;
    let s = marchIn(q0, d2);
    let q = q0 + d2 * s;
    var m = nrm(q);
    var co = dot(d2, m);
    if (co < 0.0) { m = -m; co = -co; }
    m = -faceIncident(-m, -d2);
    let exit = pbr_dielectric_split(traceDielectricHit(-m, resolvedSceneMedium(ior), airMedium()), -d2);
    if (exit.reflectance > 0.999) { return vec4f(exit.reflected, first); }
    rd = exit.transmitted;
    ro = q + m * 0.006;
  }
  return vec4f(rd, first);
}
fn geoAt(fc: vec2f) -> Geo {
  var g: Geo;
  let uv = (2.0 * fc - u.res) / u.res.y + u.shift;
  let ro = vec3f(0.0, 0.0, u.dist);
  let rd = normalize(vec3f(uv * u.tan_fov, -1.0));
  var tg: f32;
  var gr: f32;
  let t = primary(ro, rd, &tg, &gr);
  g.lbey = -1.0;
  if (t <= 0.0) { return g; }
  g.hit = 1.0;
  let ior = u.ior;
  let p = ro + rd * t;
  let n = faceIncident(nrm(p), -rd);
  g.p0 = p;
  g.n0 = n;
  let entry = pbr_dielectric_split(traceDielectricHit(n, airMedium(), resolvedSceneMedium(ior)), -rd);
  g.f0 = entry.reflectance;
  let d2 = entry.transmitted;
  let q0 = p - n * 0.004;
  let s = marchIn(q0, d2);
  let q = q0 + d2 * s;
  g.s = s;
  g.q = q;
  var m = nrm(q);
  var co = dot(d2, m);
  if (co < 0.0) { m = -m; co = -co; }
  m = -faceIncident(-m, -d2);
  g.nq = m;
  let exit = pbr_dielectric_split(traceDielectricHit(-m, resolvedSceneMedium(ior), airMedium()), -d2);
  g.f2 = exit.reflectance;
  if (exit.reflectance > 0.999) {
    g.tir = 1.0;
    g.rend = exit.reflected;
    return g;
  }
  g.od = exit.transmitted;
  let b = beyondDir(q + m * 0.006, exit.transmitted, ior);
  g.rend = b.xyz;
  g.lbey = b.w;
  return g;
}
@fragment
fn fs_guide(@builtin(position) pos: vec4f) -> GuideOut {
//INIT
  let fc = vec2f(pos.x, u.res.y - pos.y) + (fract(vec2f(0.5) + u.jitter) - 0.5);
  let g = geoAt(fc);
  var o: GuideOut;
  o.a = vec4f(g.p0, g.hit);
  o.b = vec4f(g.q, g.s);
  o.c = vec4f(g.nq, g.f2);
  o.d = vec4f(g.lbey, 0.0, 0.0, 0.0);
  return o;
}
