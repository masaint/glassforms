// Power reflectance for the s and p polarisations of a conductor seen from a clear medium.
fn fresMetalPolar(c: f32, n1: f32, n: f32, k: f32) -> vec2f {
  let c2 = c * c;
  let s2 = 1.0 - c2;
  let e = n / n1;
  let ek = k / n1;
  let t0 = e * e - ek * ek - s2;
  let ab = sqrt(t0 * t0 + 4.0 * e * e * ek * ek);
  let t1 = ab + c2;
  let a = sqrt(max(0.5 * (ab + t0), 0.0));
  let t2 = 2.0 * a * c;
  let Rs = (t1 - t2) / (t1 + t2);
  let t3 = c2 * ab + s2 * s2;
  let t4 = t2 * s2;
  return vec2f(Rs, Rs * (t3 - t4) / (t3 + t4));
}

fn fresMetal(c: f32, n1: f32, n: f32, k: f32) -> f32 {
  let f = fresMetalPolar(c, n1, n, k);
  return 0.5 * (f.x + f.y);
}
