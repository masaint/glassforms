// The pattern of the film's unevenness at q (a point in the shape's own frame, times the pattern's
// fineness), -1 to 1. Liquid in a film is stirred like cream in coffee: sheets of thicker and thinner
// film folded into each other, drawn out flat (gravity keeps them from standing on end). So: a few
// smooth waves, bent twice by other waves, taken across a space squeezed from above. ph: where the
// pattern is in its flow, as an angle; every wave moves by a whole number of turns as it goes round
// once, so the flow comes back to where it began.
fn filmWave(p: vec3f) -> f32 { return dot(sin(p), cos(p.yzx)); }
fn filmSwirl(q0: vec3f, ph: f32) -> f32 {
  var q = q0 * vec3f(1.0, 1.8, 1.0);
  q += 0.5 * vec3f(filmWave(q * 0.9 + vec3f(ph, 0.3, -ph)), filmWave(q.zxy * 1.1 + vec3f(1.7, ph, 0.3)), filmWave(q.yzx * 0.8 + vec3f(-ph, 2.9, ph)));
  q += 0.22 * vec3f(filmWave(q * 2.3 + vec3f(4.1, -ph, 0.0)), filmWave(q.zxy * 2.6 + vec3f(ph, 5.3, 1.1)), filmWave(q.yzx * 2.1 + vec3f(0.7, 0.0, 2.0 * ph)));
  return clamp(0.4 * filmWave(q * 1.3 + vec3f(0.0, ph, 0.0)) + 0.2 * filmWave(q * 2.9 + vec3f(3.3, 0.0, -ph)) + 0.1 * filmWave(q * 6.1 + vec3f(ph, 1.9, 0.0)), -1.0, 1.0);
}

fn solidFilmThickness(p: vec3f, object: vec3f, n: vec3f, roomUp: vec3f, opt: vec4f, drain0: vec4f, drain1: vec4f) -> f32 {
  let v = filmSwirl(object * opt.z, 6.2831853 * opt.w);
  let up = roomUp * drain1.w;
  let h = clamp(dot(p, up) / 1.2, -1.0, 1.0);
  // how far round from the summit the surface faces (radians, near enough)
  let s = sqrt(max(2.0 - 2.0 * dot(n, up), 0.0));
  var d = drain0.x * (1.0 + opt.y * v) * (1.0 - drain0.y * (0.5 * h + 0.5));
  d *= 1.0 - drain1.x * exp(-max(s - drain0.z * (1.0 + 0.15 * v), 0.0) * drain0.w);
  return d;
}
