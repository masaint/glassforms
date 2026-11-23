fn ggxUniform(state: ptr<function, u32>) -> f32 {
  *state = pcg(*state);
  return (f32(*state >> 9u) + 0.5) / 8388608.0;
}

// Visible normals for an incident ray above or below the local mean plane. The spherical-cap
// construction of Dupuy and Benyoub (2023) avoids projecting a below-horizon ray onto the horizon.
fn ggxVisible(n: vec3f, toward: vec3f, alpha: f32, state: ptr<function, u32>) -> vec3f {
  let x = normalize(cross(n, select(vec3f(0,0,1), vec3f(0,1,0), abs(n.z) > 0.8)));
  let y = cross(n, x);
  let v = normalize(vec3f(alpha*dot(toward,x), alpha*dot(toward,y), dot(toward,n)));
  let z = mix(-v.z, 1.0, ggxUniform(state));
  let phi = 6.283185307 * ggxUniform(state);
  let radius = sqrt(max(0.0, 1.0 - z*z));
  let h = v + vec3f(radius*cos(phi), radius*sin(phi), z);
  return normalize(alpha*h.x*x + alpha*h.y*y + max(h.z, 1e-8)*n);
}

// Height CDF along a ray through the Smith GGX microsurface, in the current medium.
// A positive result on an outward ray means the ray escaped without another collision.
fn ggxLogHeight(c: f32, alpha: f32, height: f32, sample: f32) -> f32 {
  let projected = sqrt(c*c + alpha*alpha*max(0.0,1.0-c*c));
  let invLambda = 2.0*c / max(projected-c, 1e-10);
  return log(max(height, 1e-20)) - log(sample)*invLambda;
}
