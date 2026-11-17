// The moving frame of a tracer with one wavelength inside (hero.rs `split`, docs/perf_r13.md), put back
// together. The tracer wrote three pictures:
//   rest:   what did not come out of the body (the surface's reflection with every member of the comb, the
//           background where the ray missed), and the distance;
//   inside: what the one member that went in brought out, weighted as the hero weighs it (its colour over
//           the comb's, times three: very noisy in colour pixel by pixel), and which member it was;
//   plain:  the same light counted as plain white light (the detail, without the colour noise).
// The members are laid out as a Latin square: every 3x3 block holds each of them three times, and every
// pixel takes each in turn, frame by frame. The comb's light inside is the mean over the three members of
// what each brings. Here a pixel keeps what its own member brought, and takes the other two members' from
// the neighbours on the same piece of glass (the distance) that traced them (mode 2). For comparing:
// mode 0 the hero sample as traced; mode 1 the plain light with the colour difference averaged member by
// member over the neighbours; mode 3 every member, its own too, from the neighbours.
// `sigma` (when below 100) also weighs neighbours by how alike their plain light is to this pixel's.
// The sample is then held at 0, as the tracer's own sample is, unless pad0 says to leave it signed.

struct R {
  res: vec2f,
  radius: f32,            // the neighbourhood: 1 = 3x3
  sigma: f32,
  mode: f32,
  pad0: f32,             // 1: the result left signed
  pad1: f32,
  pad2: f32,
}

@group(0) @binding(0) var<uniform> r: R;
@group(0) @binding(1) var restTex: texture_2d<f32>;
@group(0) @binding(2) var insideTex: texture_2d<f32>;
@group(0) @binding(3) var plainTex: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn held(v: vec3f) -> vec3f { return select(max(v, vec3f(0.0)), v, r.pad0 > 0.5); }
fn luma(c: vec3f) -> f32 { return max(dot(c, vec3f(0.2126, 0.7152, 0.0722)), 0.0); }

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let c = vec2i(pos.xy);
  let b = textureLoad(restTex, c, 0);
  let a = textureLoad(insideTex, c, 0);
  let mode = i32(r.mode + 0.5);
  if (b.a <= 0.0 || mode == 0) { return vec4f(held(b.rgb + a.rgb), b.a); }
  let l = textureLoad(plainTex, c, 0).rgb;
  let own = i32(a.a + 0.5);
  let top = vec2i(textureDimensions(restTex)) - 1;
  let y0 = luma(l);
  let tol = 4.0 * (0.006 + 0.003 * b.a);      // the same piece of glass (as the antialiasing's test, looser)
  // per member: the sum of what it brought (or, mode 1, of what its colour adds), and the weights
  var s0 = vec3f(0.0); var s1 = vec3f(0.0); var s2 = vec3f(0.0);
  var ws = vec3f(0.0);
  let n = i32(r.radius);
  for (var y = -2; y <= 2; y++) {
    if (y < -n || y > n) { continue; }
    for (var x = -2; x <= 2; x++) {
      if (x < -n || x > n) { continue; }
      let q = clamp(c + vec2i(x, y), vec2i(0), top);
      let bq = textureLoad(restTex, q, 0);
      if (bq.a <= 0.0 || abs(bq.a - b.a) > tol) { continue; }
      let aq = textureLoad(insideTex, q, 0);
      let lq = textureLoad(plainTex, q, 0).rgb;
      var w = 1.0;
      if (r.sigma < 100.0) {
        let yq = luma(lq);
        let rel = (yq - y0) / (r.sigma * (y0 + yq) + 1e-3);
        w = exp(-rel * rel) + 1e-4;
      }
      let v = select(aq.rgb, aq.rgb - lq, mode == 1);
      let m = i32(aq.a + 0.5);
      if (m == 0) { s0 += w * v; ws.x += w; }
      else if (m == 1) { s1 += w * v; ws.y += w; }
      else { s2 += w * v; ws.z += w; }
    }
  }
  // (a member with no neighbour on the same glass, at an outline: the hero sample as traced)
  if (min(ws.x, min(ws.y, ws.z)) <= 0.0) { return vec4f(held(b.rgb + a.rgb), b.a); }
  var m0 = s0 / ws.x;
  var m1 = s1 / ws.y;
  var m2 = s2 / ws.z;
  if (mode == 1) { return vec4f(held(b.rgb + l + (m0 + m1 + m2) / 3.0), b.a); }
  if (mode == 2) {
    if (own == 0) { m0 = a.rgb; } else if (own == 1) { m1 = a.rgb; } else { m2 = a.rgb; }
  }
  return vec4f(held(b.rgb + (m0 + m1 + m2) / 3.0), b.a);
}
