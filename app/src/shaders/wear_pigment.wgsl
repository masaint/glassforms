fn wHash(i: vec3i) -> f32 {
  var h = (u32(i.x) * 73856093u) ^ (u32(i.y) * 19349663u) ^ (u32(i.z) * 83492791u);
  h = (h ^ (h >> 16u)) * 0x45d9f3bu;
  h = (h ^ (h >> 16u)) * 0x45d9f3bu;
  h = h ^ (h >> 16u);
  return f32(h & 0xffffffu) * (2.0 / 16777215.0) - 1.0;
}
// smooth noise, -1 to 1, one bump about every unit of x
fn wNoise(x: vec3f) -> f32 {
  let i = vec3i(floor(x));
  let f = fract(x);
  let s = f * f * (3.0 - 2.0 * f);
  let a = mix(mix(wHash(i), wHash(i + vec3i(1, 0, 0)), s.x), mix(wHash(i + vec3i(0, 1, 0)), wHash(i + vec3i(1, 1, 0)), s.x), s.y);
  let b = mix(mix(wHash(i + vec3i(0, 0, 1)), wHash(i + vec3i(1, 0, 1)), s.x), mix(wHash(i + vec3i(0, 1, 1)), wHash(i + vec3i(1, 1, 1)), s.x), s.y);
  return mix(a, b, s.z);
}

// Convexity and occlusion at the worn surface, shared by its glaze and its pigment.
fn wearForm(p: vec3f, n: vec3f, h: f32) -> vec2f {
  let sg = select(-1.0, 1.0, n.z >= 0.0);
  let k = -1.0 / (sg + n.z);
  let m = n.x * n.y * k;
  let t1 = vec3f(1.0 + sg * n.x * n.x * k, sg * m, -sg * n.x);
  let t2 = vec3f(m, sg + n.y * n.y * k, -n.y);
  let d0 = mapW(p);
  let bend = (mapW(p + h * t1) + mapW(p + h * (-0.5 * t1 + 0.8660254 * t2)) + mapW(p + h * (-0.5 * t1 - 0.8660254 * t2)) - 3.0 * d0) / 3.0;
  // (a sphere of radius r bends by h^2 / 2r over h: convex 1 for a radius of about four steps or less)
  let convex = clamp(bend / (0.06 * h), -1.0, 1.0);
  var shut = 0.0;
  for (var i = 0; i < 3; i++) {
    let s = 0.02 + 0.06 * f32(i);
    shut += max(s - mapW(p + n * s), 0.0) / s;
  }
  return vec2f(convex, clamp(shut / 1.5, 0.0, 1.0));
}

// Dirt and low-frequency pigment variation colour a painted substrate below its clear reflection.
fn wearPigment(q: vec3f, dirt: f32, dirtColour: vec3f, mottle: f32) -> vec3f {
  var tint = mix(vec3f(1.0), dirtColour * 1.6, 0.7 * dirt);
  if (mottle > 0.0) {
    tint *= vec3f(1.0) + mottle * 0.06 * vec3f(wNoise(q * 2.3 + vec3f(5.0, 1.0, 8.0)), wNoise(q * 2.3 + vec3f(11.0, 3.0, 2.0)), wNoise(q * 2.3 + vec3f(2.0, 13.0, 6.0)));
  }
  return tint;
}
