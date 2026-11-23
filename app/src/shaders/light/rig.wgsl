// The irradiance (per unit of the light's radiance) that one of the rig's rectangle lights at infinity gives a surface facing ne
// (in the lights' own frame), unshadowed: a 4 x 4 grid over the rectangle, each point weighted by its
// cosine (wrapped past the terminator by `wrap`, as a ceramic's light is) and its solid angle, the
// rectangle's edges feathered by `soft`. a.xyz: the light's centre direction, a.w its half-width;
// u.xyz, v.xyz: its axes, u.w its half-height, v.w the feather. Shared by the surface pass (atInfinity)
// and assembly paint (samplePaintPanel); the two feather forms they had are the same smoothstep.
fn lightRigOpen(ne: vec3f, a: vec4f, u: vec4f, v: vec4f, wrap: f32) -> f32 {
  let tw = a.w;
  let th = u.w;
  let soft = max(v.w, 1e-3);
  let na = dot(ne, a.xyz);
  let nu = dot(ne, u.xyz);
  let nv = dot(ne, v.xyz);
  // (wholly behind the surface: every corner of the rectangle faces away, by more than the wrap)
  if (na + abs(nu) * tw + abs(nv) * th + wrap * sqrt(1.0 + tw * tw + th * th) <= 0.0) { return 0.0; }
  var open = 0.0;
  for (var gy = 0; gy < 4; gy++) {
    let y = (f32(gy) * 0.5 - 0.75) * th;
    let sy = 1.0 - smoothstep(0.0, 1.0, (abs(y) / th - (1.0 - soft)) / soft);
    for (var gx = 0; gx < 4; gx++) {
      let x = (f32(gx) * 0.5 - 0.75) * tw;
      let sx = 1.0 - smoothstep(0.0, 1.0, (abs(x) / tw - (1.0 - soft)) / soft);
      let q = 1.0 + x * x + y * y;
      // (cosine = (na + x nu + y nv) / sqrt(q); solid angle per unit of the plane = 1 / q^1.5;
      // wrapped: (cosine + wrap) / (1 + wrap))
      open += max(na + x * nu + y * nv + wrap * sqrt(q), 0.0) * sx * sy / (q * q * (1.0 + wrap));
    }
  }
  return open * (tw * th * 0.25);
}
