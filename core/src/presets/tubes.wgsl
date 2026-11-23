// raw: a distance
// Swept tubes, read exactly. A span is a cubic (or a line) with a radius running along it: the envelope of
// the balls centred on it. Its distance is the least of |p - C(t)| - r(t) over the span.
//
// With no melt the shape is the union of the spans: the least of their distances.
//
// With a melt, strands nearer than about `melt` flow together. Every span adds the line integral of
//     exp(-x / melt) / I(r)
// along itself (x: how far outside that point's own ball; I(r): what a straight strand of radius r comes to
// on its own surface, so a strand alone keeps its radius), and the sum is turned back into a distance,
// -melt ln(sum). That is one smooth function of the point: it does not depend on how a strand is cut into
// spans, on how many strands meet, or on which cell the point is in. A span's part fades to nothing (twice
// differentiably) between `clear` and `clear / 0.65` outside its own surface, and every cell lists every
// span that comes that near any point of it.
//
// Far from every span the distance is read from the table: the distance at each cell's middle, joined
// trilinearly (so continuous from cell to cell), less what such a joining can overstate. Every other
// reading of the volume (shadows, the ambient light's occlusion) takes the stored value for a distance,
// so it has to be one, not merely a step that is safe to take.
//
// The table (`tbl`, written by presets/exact.rs `write_tubes`):
//   0      (entries, ground height, 1 if there is a ground, melt)
//   1      (the grid's low corner, its cell size)
//   2      (cells a side, reach: every span within this of a cell is in its list, the melt's `clear`,
//           the first span's entry)
//   3 ..   one entry a cell: (where its list starts, how many, the distance at its middle, 0)
//   lists, then the spans, as exact.rs has them.
// n, though the compiler cannot see it (the table's first number is its length, at least 12 rows wherever
// there is a piece to read): a loop this long is kept a loop, not unrolled into every place the tracer
// reads the formula
fn tube_count(n: u32) -> u32 { return min(n, u32(tbl[0].x)); }

// A span as its polynomial: C(t) = a + t (e + t (f + t g)), xyz the centre line and w the radius; d its end.
struct TubeSpan { a: vec4f, e: vec4f, f: vec4f, g: vec4f, d: vec4f }

fn tube_span(a: vec4f, b: vec4f, c: vec4f, d: vec4f) -> TubeSpan {
  return TubeSpan(a, 3.0 * (b - a), 3.0 * (a - 2.0 * b + c), d - a + 3.0 * (b - c), d);
}
fn tube_at(s: TubeSpan, t: f32) -> vec4f { return s.a + t * (s.e + t * (s.f + t * s.g)); }
fn tube_rate(s: TubeSpan, t: f32) -> vec4f { return s.e + t * (2.0 * s.f + 3.0 * t * s.g); }

// Where the span comes nearest: the t in [0, 1] at which F(t) = |p - C(t)| - r(t) is least. Newton steps
// from the best of four starts, each step kept only if it lowers F.
fn tube_foot(p: vec3f, s: TubeSpan) -> f32 {
  let chord = s.d.xyz - s.a.xyz;
  let l2 = max(dot(chord, chord), 1e-20);
  let l = sqrt(l2);
  let u_dir = chord / l;
  let pa = p - s.a.xyz;
  let x = dot(pa, u_dir);
  let y = length(pa - x * u_dir);
  let dr = s.d.w - s.a.w;
  var t = clamp((x + y * dr / sqrt(max(l2 - dr * dr, 0.04 * l2))) / l, 0.0, 1.0);
  var best_c = tube_at(s, t);
  var best_f = length(p - best_c.xyz) - max(0.0, best_c.w);
  let f0 = length(pa) - max(0.0, s.a.w);
  if (f0 < best_f) { best_f = f0; t = 0.0; best_c = s.a; }
  let mid = tube_at(s, 0.5);
  let fm = length(p - mid.xyz) - max(0.0, mid.w);
  if (fm < best_f) { best_f = fm; t = 0.5; best_c = mid; }
  let f1 = length(p - s.d.xyz) - max(0.0, s.d.w);
  if (f1 < best_f) { best_f = f1; t = 1.0; best_c = s.d; }

  for (var iter = 0u; iter < tube_count(5u); iter++) {
    let delta = best_c.xyz - p;
    let dist = length(delta);
    if (dist < 1e-10) { break; }
    let speed = tube_rate(s, t);
    let accel = 2.0 * s.f + (6.0 * t) * s.g;
    let vt = dot(delta, speed.xyz);
    let fp = vt / dist - speed.w;
    if (abs(fp) < 1e-6 || (t <= 0.0 && fp >= 0.0) || (t >= 1.0 && fp <= 0.0)) { break; }
    let fpp = (dot(speed.xyz, speed.xyz) + dot(delta, accel.xyz) - (vt * vt) / (dist * dist)) / dist - accel.w;
    let min_curv = max(dot(speed.xyz, speed.xyz) / dist * 0.1, 1e-6);
    let step = clamp(fp / max(fpp, min_curv), -0.35, 0.35);
    let t1 = clamp(t - step, 0.0, 1.0);
    let c1 = tube_at(s, t1);
    let v1 = length(p - c1.xyz) - max(0.0, c1.w);
    if (v1 <= best_f) {
      t = t1;
      best_f = v1;
      best_c = c1;
    } else {
      let t2 = 0.5 * (t + t1);
      let c2 = tube_at(s, t2);
      let v2 = length(p - c2.xyz) - max(0.0, c2.w);
      if (v2 < best_f) {
        t = t2;
        best_f = v2;
        best_c = c2;
      } else {
        break;
      }
    }
    if (abs(step) < 1e-6) { break; }
  }
  return t;
}

// The way out of the span at its nearest point c = C(t) (a unit vector), and the distance there.
fn tube_reading(p: vec3f, s: TubeSpan, t: f32, c: vec4f) -> vec4f {
  let offset = p - c.xyz;
  let dist = length(offset);
  var n: vec3f;
  if (dist > 1e-12) {
    n = offset / dist;
  } else {
    let tangent = tube_rate(s, t).xyz;
    let seed = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(tangent.x) > 0.7 * max(length(tangent), 1e-12));
    n = normalize(cross(tangent, seed));
  }
  return vec4f(n, dist - max(0.0, c.w));
}

// Half of k times what a straight strand of radius r integrates to on its own surface: with z = k r,
// z e^z K1(z), to a third of a percent.
fn tube_strand(z: f32) -> f32 { return sqrt(1.0 + 1.5707964 * z + 0.19 * z / (z + 0.8)); }

// A span's part of the melt fades from all of it at `clear` outside its surface to none at clear / 0.65:
// S = 1 - u^3 (10 - 15 u + 6 u^2), level and unbent at both ends. x: the fade; y: S - m dS/dx, which is
// what the part's slope carries in its place (the part is exp(-x / m) S).
fn tube_fade(x: f32, clear: f32, m: f32) -> vec2f {
  let width = 0.5384615 * clear;
  let u = clamp((x - clear) / width, 0.0, 1.0);
  let fade = 1.0 - u * u * u * (10.0 - u * (15.0 - 6.0 * u));
  return vec2f(fade, fade + m * 30.0 * u * u * (1.0 - u) * (1.0 - u) / width);
}

// A span's part of the melt, over exp(-k x0) (x0: its least distance, at t0, rho0 from its centre line):
// the integral of exp(-k (x - x0)) fade / I(r) along it. xyz: the same with each point's way out, which
// summed over the spans is the slope. The integrand is a peak about t0: on a straight strand it is
// exp(-k (sqrt(rho0^2 + s^2) - rho0)) at s along, a round top the wider the further out the point is, and
// a spike at a point on the centre line. With s = rho sinh(u) it is smooth in u wherever the point is, so
// eight Gauss-Legendre nodes are spread evenly in u, out to where it has fallen by e^-6
// (s^2 = 12 m (3 m + rho0)) or the span ends.
fn tube_melt(p: vec3f, s: TubeSpan, t0: f32, x0: f32, rho0: f32, k: f32, clear: f32) -> vec4f {
  // (place, weight) on [-1, 1], each for both signs
  var nodes = array<vec2f, 4>(vec2f(0.1834346425, 0.3626837834), vec2f(0.5255324099, 0.3137066459), vec2f(0.7966664774, 0.2223810345), vec2f(0.9602898565, 0.1012285363));
  let m = 1.0 / k;
  let speed = max(length(tube_rate(s, t0).xyz), 1e-9);
  let rho = max(rho0, 0.25 * m);
  let along = sqrt(12.0 * m * (3.0 * m + rho0));
  let lo = -asinh(min(t0 * speed, along) / rho);
  let hi = asinh(min((1.0 - t0) * speed, along) / rho);
  let mid = 0.5 * (lo + hi);
  let half = 0.5 * (hi - lo);
  // (the strand's own measure at the span's two ends, run evenly between them)
  let w0 = 0.5 * k / tube_strand(k * max(s.a.w, 0.0));
  let w1 = 0.5 * k / tube_strand(k * max(s.d.w, 0.0));
  var sum = vec4f(0.0);
  for (var j = 0u; j < 8u; j++) {
    let node = nodes[j >> 1u];
    let grow = exp(mid + half * select(-node.x, node.x, (j & 1u) == 1u));
    let t = clamp(t0 + 0.5 * rho * (grow - 1.0 / grow) / speed, 0.0, 1.0);
    let c = tube_at(s, t);
    let offset = p - c.xyz;
    let dist = length(offset);
    let x = dist - max(0.0, c.w);
    // (ds = |C'| dt, and dt = rho cosh(u) du / speed)
    let w = node.y * half * 0.5 * rho * (grow + 1.0 / grow) / speed * length(tube_rate(s, t).xyz) * mix(w0, w1, t) * exp(-k * (x - x0));
    let fade = tube_fade(x, clear, m);
    sum += vec4f(w * fade.y * offset / max(dist, 1e-12), w * fade.x);
  }
  return sum;
}

fn pc_blade(p: vec3f, a: vec4f, c: vec4f, s: vec4f) -> f32 {
  let ax = c.xyz - a.xyz;
  let L = max(length(ax), 1e-6);
  let x = ax / L;
  let y = normalize(s.xyz - a.xyz - x * dot(s.xyz - a.xyz, x));
  let z = cross(x, y);
  let q = p - a.xyz;
  let W = c.w;
  let u = dot(q, x) / L;
  let v = dot(q, y);
  let w = dot(q, z) - s.w * v * v / max(W, 1e-6);
  let uc = clamp(u, 0.02, 1.0);
  let k = W / 0.385;
  let hw = k * sqrt(uc) * (1.0 - uc);
  let slope = k * (0.5 / sqrt(uc) * (1.0 - uc) - sqrt(uc)) / L;
  let side = (abs(v) - hw) / sqrt(1.0 + slope * slope);
  let ends = max(-u, u - 1.0) * L;
  let d2 = max(side, ends);
  let e = vec2f(d2, abs(w));
  return length(max(e, vec2f(0.0))) + min(max(e.x, e.y), 0.0) + a.w;
}

// The slope of a blade's distance, from four readings about the point.
fn tube_blade_slope(p: vec3f, a: vec4f, b: vec4f, c: vec4f) -> vec3f {
  let eps = 0.0001;
  var g = vec3f(0.0);
  for (var k = 0u; k < tube_count(4u); k++) {
    let tap = vec3f(select(-1.0, 1.0, k == 0u || k == 3u), select(-1.0, 1.0, k >= 2u), select(-1.0, 1.0, k == 1u || k == 3u));
    g += tap * pc_blade(p + eps * tap, a, b, c);
  }
  return normalize(g);
}

// The span at entry i of a piece whose kind is 110 (a cubic) or 111 (a line).
fn tube_span_at(i: u32, kind: f32) -> TubeSpan {
  let a = tbl[i + 3u];
  let b = tbl[i + 4u];
  if (kind == 111.0) { return TubeSpan(a, b - a, vec4f(0.0), vec4f(0.0), b); }
  return tube_span(a, b, tbl[i + 5u], tbl[i + 6u]);
}

// A lower bound on a span's distance: the capsule about its chord that holds all of it.
fn tube_bound(p: vec3f, s: TubeSpan, radius: f32) -> f32 {
  let chord = s.d.xyz - s.a.xyz;
  let t = clamp(dot(p - s.a.xyz, chord) / max(dot(chord, chord), 1e-20), 0.0, 1.0);
  return length((p - s.a.xyz) - t * chord) - radius;
}

// The distance far from every span, from the cells' middles (8 readings): what it is at each middle, and
// between them never more than it is. With D the distance to the shape, D^2 - |x|^2 bends only downward
// (it is the least, over the shape's points s, of |s|^2 - 2 x.s, each of them straight), so joined straight
// from middle to middle it is never over:
//     D^2 >= (the middles' squares, joined trilinearly) - cell^2 (fx (1 - fx) + fy (1 - fy) + fz (1 - fz)),
// which is exact for a single point and continuous from cell to cell. Outside the box of the middles (it
// holds the whole shape) the way out of that box is added squared. xyz: its slope.
fn tube_far(q: vec3f, grid: vec4f, n: u32) -> vec4f {
  let top = f32(n) - 1.0;
  let coord = (q - grid.xyz) / grid.w - 0.5;
  let c = clamp(coord, vec3f(0.0), vec3f(top));
  let i0 = vec3u(min(floor(c), vec3f(max(top - 1.0, 0.0))));
  let i1 = min(i0 + vec3u(1u), vec3u(n - 1u));
  let f = c - vec3f(i0);
  let d000 = max(tbl[3u + (i0.z * n + i0.y) * n + i0.x].z, 0.0);
  let d100 = max(tbl[3u + (i0.z * n + i0.y) * n + i1.x].z, 0.0);
  let d010 = max(tbl[3u + (i0.z * n + i1.y) * n + i0.x].z, 0.0);
  let d110 = max(tbl[3u + (i0.z * n + i1.y) * n + i1.x].z, 0.0);
  let d001 = max(tbl[3u + (i1.z * n + i0.y) * n + i0.x].z, 0.0);
  let d101 = max(tbl[3u + (i1.z * n + i0.y) * n + i1.x].z, 0.0);
  let d011 = max(tbl[3u + (i1.z * n + i1.y) * n + i0.x].z, 0.0);
  let d111 = max(tbl[3u + (i1.z * n + i1.y) * n + i1.x].z, 0.0);
  let v000 = d000 * d000;
  let v100 = d100 * d100;
  let v010 = d010 * d010;
  let v110 = d110 * d110;
  let v001 = d001 * d001;
  let v101 = d101 * d101;
  let v011 = d011 * d011;
  let v111 = d111 * d111;
  let x00 = mix(v000, v100, f.x);
  let x10 = mix(v010, v110, f.x);
  let x01 = mix(v001, v101, f.x);
  let x11 = mix(v011, v111, f.x);
  let y0 = mix(x00, x10, f.y);
  let y1 = mix(x01, x11, f.y);
  let gx = mix(mix(v100 - v000, v110 - v010, f.y), mix(v101 - v001, v111 - v011, f.y), f.z);
  let gy = mix(x10 - x00, x11 - x01, f.z);
  let bow = f * (1.0 - f);
  let inside = max(mix(y0, y1, f.z) - grid.w * grid.w * (bow.x + bow.y + bow.z), 0.0);
  // half the slope of the square: within the box of the middles, and along the ways out of it
  let out = (coord - c) * grid.w;
  let half = select(0.5 * (vec3f(gx, gy, y1 - y0) / grid.w - grid.w * (1.0 - 2.0 * f)), out, out != vec3f(0.0));
  let value = sqrt(inside + dot(out, out));
  return vec4f(half / max(value, 1e-9), value);
}

// The distance at p and its slope there (xyz: where the surface is, a unit vector or shorter).
fn tube_eval(p: vec3f) -> vec4f {
  let header = tbl[0];
  let grid = tbl[1];
  let sizes = tbl[2];
  let n = u32(sizes.x);
  // outside the grid: read at its nearest point q, and add the way there (everything is inside the grid)
  let q = clamp(p, grid.xyz, grid.xyz + f32(n) * grid.w);
  let coord = (q - grid.xyz) / grid.w;
  let ci = vec3u(clamp(coord, vec3f(0.0), vec3f(f32(n) - 1.0)));
  let cell = tbl[3u + (ci.z * n + ci.y) * n + ci.x];
  let count = u32(cell.y);
  let first = (3u + n * n * n) * 4u + u32(cell.x);
  let melt = header.w;
  let reach = sizes.y;
  // the least distance over the cell's spans, exactly (right wherever it is under `reach`)
  var near = vec4f(0.0, 0.0, 0.0, 1e9);
  // the melt: the spans' parts over exp(-k ref), and their slopes
  var parts = vec4f(0.0);
  var ref_x = 1e9;
  let k = 1.0 / max(melt, 1e-9);
  let clear = sizes.z;
  let band = clear / 0.65;
  for (var j = 0u; j < count; j++) {
    let at = first + j;
    let i = u32(tbl[at / 4u][at % 4u]);
    // A span matters if it may be the nearest, or (with a melt) if its part is not yet nothing: within the
    // fade's end, and within 16 melts of the nearest (beyond that it adds under a ten-millionth).
    var limit = min(near.w, reach);
    if (melt > 0.0) { limit = max(limit, min(band, ref_x + 16.0 * melt)); }
    // (nothing of a span is nearer than its box)
    let from_box = abs(q - tbl[i].xyz) - tbl[i + 1u].xyz;
    if (max(from_box.x, max(from_box.y, from_box.z)) >= limit) { continue; }
    let tag = tbl[i + 2u];
    let kind = tag.x;
    var e = vec4f(0.0, 0.0, 0.0, 1e9);
    if (kind == 110.0 || kind == 111.0) {
      let s = tube_span_at(i, kind);
      if (tube_bound(q, s, tag.w) >= limit) { continue; }
      let t = tube_foot(q, s);
      let c = tube_at(s, t);
      e = tube_reading(q, s, t, c);
      if (melt > 0.0 && e.w < band) {
        let part = tube_melt(q, s, t, e.w, length(q - c.xyz), k, clear);
        if (e.w < ref_x) {
          parts = parts * exp(-k * (ref_x - e.w)) + part;
          ref_x = e.w;
        } else {
          parts += part * exp(-k * (e.w - ref_x));
        }
      }
    } else {
      // a ball, or a leaf: one piece, with nothing along it to add to it
      let a = tbl[i + 3u];
      if (kind == 101.0) {
        let d = q - a.xyz;
        let len = length(d);
        e = vec4f(select(vec3f(0.0, 1.0, 0.0), d / len, len > 1e-12), len - a.w);
      } else {
        let b = tbl[i + 4u];
        let c = tbl[i + 5u];
        e.w = pc_blade(q, a, b, c);
        if (e.w < limit || (melt > 0.0 && e.w < band)) { e = vec4f(tube_blade_slope(q, a, b, c), e.w); }
      }
      if (melt > 0.0 && e.w < band) {
        let fade = tube_fade(e.w, clear, melt);
        let part = vec4f(e.xyz * fade.y, fade.x);
        if (e.w < ref_x) {
          parts = parts * exp(-k * (ref_x - e.w)) + part;
          ref_x = e.w;
        } else {
          parts += part * exp(-k * (e.w - ref_x));
        }
      }
    }
    if (e.w < near.w) { near = e; }
  }
  // The union's distance. Beyond 0.65 reach the lists' reading gives way to the far field: it is run up
  // to nothing at `reach` (past which the lists no longer hold every span), so the two join without a step
  // whatever the far field says.
  let near_clear = 0.65 * reach;
  var result = vec4f(0.0, 1.0, 0.0, 1e9);
  if (near.w < reach) {
    let u = clamp((near.w - near_clear) / (reach - near_clear), 0.0, 1.0);
    let keep = 1.0 - u * u * u * (10.0 - u * (15.0 - 6.0 * u));
    result = vec4f(near.xyz, near.w / max(keep, 1e-6));
  }
  if (result.w > near_clear) {
    // (the far field is never over the distance, and here the distance is over `near_clear`)
    let far = tube_far(q, grid, n);
    let value = max(far.w, near_clear);
    if (value < result.w) {
      let len = length(far.xyz);
      result = vec4f(select(vec3f(0.0, 1.0, 0.0), far.xyz / len, len > 1e-12), value);
    }
  }
  if (melt > 0.0) {
    // With a melt the surface is the melt's, which lies outside the union's by melt ln(n) where n strands
    // run together (1.9 melts at most in the shapes measured: core/tests/tube_field.rs). The melt's own
    // field is whole up to `clear` outside a strand; from 2 melts under that the field levels off, and
    // beyond it is the union's distance held 2 melts short. So nothing near the surface is the union's (it
    // would crease the melt wherever the melt draws the surface in, as at a strand's end), the field never
    // rises faster than a distance does, and far from the shape it is not over the way to the melt's surface.
    result.w = max(result.w - 2.0 * melt, clear - 2.0 * melt);
    if (parts.w > 0.0) {
      let melted = ref_x - log(parts.w) / k;
      if (melted < result.w) { result = vec4f(parts.xyz / parts.w, melted); }
    }
  }
  let out = p - q;
  let outside = dot(out, out);
  if (outside > 0.0) {
    let inner = max(result.w, 0.0);
    let d = sqrt(outside + inner * inner);
    result = vec4f((out + inner * result.xyz) / d, d);
  }
  if (header.z > 0.5 && header.y - p.y > result.w) { result = vec4f(0.0, -1.0, 0.0, header.y - p.y); }
  if (dot(result.xyz, result.xyz) < 1e-24) { result = vec4f(0.0, 1.0, 0.0, result.w); }
  return result;
}

// (no cache of the last reading for the normal to reuse: it held eight registers through the whole tracer,
// which cost more than the readings it saved)
fn raw(p: vec3f) -> f32 {
  return tube_eval(p).w;
}

fn rawGrad(p: vec3f) -> vec4f {
  return tube_eval(p);
}
