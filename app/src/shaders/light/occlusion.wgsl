// How far the distance field closes in over the surface at p: five looks out along the normal n, each
// asking how near the shape is; open surface finds it as far as the step, a fold finds it nearer.
// 1 open, toward 0 hemmed in. Shared by the tracer's clay and the surface pass (needs the host's mapW).
fn lightOcclusion(p: vec3f, n: vec3f) -> f32 {
  var shut = 0.0;
  var weight = 1.0;
  for (var i = 0; i < 5; i++) {
    let h = 0.012 + 0.052 * f32(i);
    shut += weight * max(h - mapW(p + n * h), 0.0) / h;
    weight *= 0.8;
  }
  return clamp(1.0 - 0.42 * shut, 0.0, 1.0);
}