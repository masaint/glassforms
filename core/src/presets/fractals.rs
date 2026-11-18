//! Fractals, in glass: shapes made by repeating one step without end. Each is drawn at a finite depth,
//! and the depth is a slider: at its lowest the shape is plain, and as it rises detail grows into it.
//! Between whole depths the shapes at the two neighbouring depths are blended, so the growth is smooth.
//!
//!   escape-time    the Mandelbulb and the quaternion Julia set: a point belongs while its orbit under
//!                  z -> z^n + c stays near. Drawn as the level set |z_N| = R after N steps, which closes in on
//!                  the set as N grows; `raw` is that level's distance estimate r·ln(r/R)/|dz|.
//!   box folding    the Mandelbox: each step folds space at the faces of a cube and turns it inside out in a
//!                  ball, then scales it and adds the start. The same level set and estimate, r/|dz|.
//!   IFS            the Menger sponge and the Sierpinski tetrahedron: copies of the shape, smaller, in place
//!                  of the shape. Space is folded or repeated onto one copy at each level; the distance to
//!                  the last level's solid, scaled back, is a distance (the sponge's is never more than the
//!                  true one).
//!   inversion      the Apollonian packing: balls inside a ball, each gap filled by the largest ball that
//!                  fits. A point is turned inside out in the sphere over whichever gap it lies in, until it
//!                  lies in none; there it is measured to the first five balls.
//!
//! Detail is held to what the volume (768 voxels a side) can show: the depths stop where the next
//! level's features would be thinner than about two voxels, and every edge is rounded to two voxels at
//! the least (`fine()`, as the geometric solids do).
use super::*;
use super::solids::{FINE, TURN};

/// How far from the centre each shape reaches at most.
const FIT: f64 = 1.22;

type V = [f64; 3];
fn dot(a: V, b: V) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn sub(a: V, b: V) -> V { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn scale(a: V, s: f64) -> V { [a[0] * s, a[1] * s, a[2] * s] }
fn unit(a: V) -> V { scale(a, 1.0 / dot(a, a).sqrt()) }

/// The starting orientation that shows a shape, upright in its own frame, turned by `want` (degrees about
/// x, y, z), when the shape is drawn turned by TURN inside its formula (as the cube-aligned ones are, to
/// keep their edges off the planes x = y and the like, which run through the centres of voxels).
fn view(want: [f64; 3]) -> [f64; 3] {
    use crate::math::{axis_angle, euler_of, mul, DEG};
    let t = [TURN[0][0], TURN[1][0], TURN[2][0], TURN[0][1], TURN[1][1], TURN[2][1], TURN[0][2], TURN[1][2], TURN[2][2]];
    let shown = mul(&axis_angle(0.0, 0.0, 1.0, want[2] * DEG), &mul(&axis_angle(1.0, 0.0, 0.0, want[0] * DEG), &axis_angle(0.0, 1.0, 0.0, want[1] * DEG)));
    let e = euler_of(&mul(&shown, &t));
    e.map(|a| (a * 10.0).round() / 10.0)
}

/// WGSL taking a point of the volume into the shape's upright frame (TURN undone).
fn unturn() -> String {
    let r = |i: usize| format!("vec3f({:.9}, {:.9}, {:.9})", TURN[0][i], TURN[1][i], TURN[2][i]);
    format!("fn upright(p: vec3f) -> vec3f {{ return vec3f(dot({}, p), dot({}, p), dot({}, p)); }}\n", r(0), r(1), r(2))
}

/// What every preset here gives.
struct Frac {
    id: &'static str,
    name: &'static str,
    eq: &'static [&'static str],
    note: &'static str,
    params: Vec<ParamDef>,
    rot: [f64; 3],
    animate: &'static str,
    wgsl: String,
    clip: &'static str,
}

fn frac(f: Frac) -> Preset {
    Preset {
        eq: f.eq.to_vec(),
        note: f.note,
        params: f.params,
        rot: Some(f.rot),
        animate: Some(vec![f.animate]),
        ..Preset::new(f.id, f.name, Shape::Field { wgsl: f.wgsl, clip: f.clip.to_string(), shell: "0.0".to_string() })
    }
}

// ------------------------------------------------------------------------------------------ escape time

// The depth slider: N steps and a fraction of the next. Both levels' estimates are kept and blended.
const DEPTH: &str = "
struct Depth { k: i32, t: f32 }
fn depth(d: f32, most: f32) -> Depth {
  let x = clamp(d, 1.0, most);
  let k = i32(floor(x));
  return Depth(k, x - f32(k));
}
// distance estimate of the level set |z| = R, from |z| and |dz|
fn level(r: f32, dr: f32, big: f32) -> f32 { return r * log(max(r, 1e-6) / big) / max(dr, 1e-6); }
";

const MANDELBULB: &str = "
// the distance estimate after the steps taken: a point gone past 2 is outside, at 0.5·ln r·r/|dz| from the
// set. One still within is inside, where the estimate means nothing (it has hills, which the bake's walk
// would climb and take for a surface): a slope out from the middle stands in, below the skin everywhere.
fn bulb_est(r: f32, dr: f32, c: vec3f) -> f32 {
  return select(-0.03 * (2.0 - length(c)), 0.5 * log(r) * r / max(dr, 1e-6), r > 2.0);
}
fn raw(p: vec3f) -> f32 {
  let c = p / BULB_S;
  let n = clamp(round(pr.power), 2.0, 12.0);
  let dp = depth(max(pr.depth, 2.0), 9.0);
  var z = c;
  var dr = 1.0;
  var r = length(z);
  var d0 = bulb_est(r, dr, c);
  var d1 = d0;
  for (var i = 0; i < 10; i++) {
    if (i > dp.k) { break; }
    // gone: the estimate no longer changes (and one more step from much further out would take |dz| past
    // what a 32-bit float holds, and the estimate to 0)
    if (r > 32.0) { break; }
    // z -> z^n + c, in spherical coordinates
    let th = acos(clamp(z.z / max(r, 1e-9), -1.0, 1.0)) * n;
    let ph = atan2(z.y, z.x) * n;
    let rn = pow(r, n - 1.0);
    dr = n * rn * dr + 1.0;
    z = rn * r * vec3f(sin(th) * cos(ph), sin(th) * sin(ph), cos(th)) + c;
    r = length(z);
    let d = bulb_est(r, dr, c);
    if (i + 1 <= dp.k) { d0 = d; }
    d1 = d;
  }
  // the glass is the set with a skin about it: detail finer than the skin is filled
  return mix(d0, d1, dp.t) * BULB_S - pr.skin;
}";

const JULIA: &str = "
fn qmul(a: vec4f, b: vec4f) -> vec4f {
  return vec4f(a.x * b.x - dot(a.yzw, b.yzw), a.x * b.yzw + b.x * a.yzw + cross(a.yzw, b.yzw));
}
// the distance estimate after the steps taken: past 4 a point is outside, at 0.5·ln r·r/|dz| from the
// set; one still within is inside, where a slope out from the middle stands in (as for the Mandelbulb)
fn julia_est(r: f32, dr: f32, c: vec3f) -> f32 {
  return select(-0.03 * (2.0 - length(c)), 0.5 * log(r) * r / max(dr, 1e-6), r > 4.0);
}
fn raw(p: vec3f) -> f32 {
  let c = vec4f(pr.cr, pr.ci, pr.cj, pr.ck);
  let dp = depth(pr.depth, 11.0);
  var z = vec4f(p / JULIA_S, pr.slice);
  var dz = vec4f(1.0, 0.0, 0.0, 0.0);
  var r = length(z);
  var d0 = julia_est(r, 1.0, p / JULIA_S);
  var d1 = d0;
  for (var i = 0; i < 12; i++) {
    if (i > dp.k) { break; }
    if (r > 256.0) { break; }                // gone: the estimate no longer changes
    dz = 2.0 * qmul(z, dz);
    z = qmul(z, z) + c;
    r = length(z);
    let d = julia_est(r, length(dz), p / JULIA_S);
    if (i + 1 <= dp.k) { d0 = d; }
    d1 = d;
  }
  // the glass is the set with a skin about it: detail finer than the skin is filled
  return mix(d0, d1, dp.t) * JULIA_S - pr.skin;
}";

// ------------------------------------------------------------------------------------------ box folding

const MANDELBOX: &str = "
fn raw(p0: vec3f) -> f32 {
  let p = upright(p0);
  // (between -1.25 and 1.25 the fold does not shrink space and the set fills everything: held at ±1.25)
  let s = select(-1.0, 1.0, pr.scale >= 0.0) * max(abs(pr.scale), 1.25);
  // a bound on how far the set reaches: 2(|s| + 1)/(|s| - 1) for s > 1; about 2 for s < -1
  let reach = select(2.2, 2.0 * (s + 1.0) / max(s - 1.0, 0.3), s > 0.0);
  let c = p * (reach / BOX_FIT);
  let dp = depth(max(pr.depth, 3.0), 11.0);
  let f = pr.fold;
  let m2 = pr.inner * pr.inner;
  var z = c;
  var dr = 1.0;
  var d0 = length(z) - reach;
  var d1 = d0;
  for (var i = 0; i < 12; i++) {
    if (i > dp.k) { break; }
    // fold at the faces of the cube |x| = f, then turn inside out in the ball of radius 1, keeping the
    // innermost ball (radius `inner`) as it is
    z = clamp(z, vec3f(-f), vec3f(f)) * 2.0 - z;
    let r2 = dot(z, z);
    let k = select(select(1.0, 1.0 / r2, r2 < 1.0), 1.0 / m2, r2 < m2);
    z *= k;
    dr *= k;
    z = s * z + c;
    dr = dr * abs(s) + 1.0;
    // the distance estimate |z| / |dz| (a point whose orbit stays near is at almost 0)
    let d = length(z) / abs(dr);
    if (i + 1 <= dp.k) { d0 = d; }
    d1 = d;
  }
  // the glass is the set with a skin about it: detail finer than the skin is filled
  return mix(d0, d1, dp.t) * (BOX_FIT / reach) - pr.skin;
}";

// ------------------------------------------------------------------------------------------------ IFS

const MENGER: &str = "
// a bar of square section, half-width h, along the axis left out of `a`, its long edges rounded to rr
fn mg_bar(a: vec2f, h: f32, rr: f32) -> f32 {
  let r = min(rr, h);
  let q = abs(a) - vec2f(h - r);
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r;
}
fn raw(p0: vec3f) -> f32 {
  let q = upright(p0) / MG_S;
  let r = max(pr.round, fine()) / MG_S;          // every edge rounded, in the sponge's own units
  let lv = clamp(pr.level, 0.0, 3.0);
  var d = sdBox(q, vec3f(1.0), r);
  var s = 1.0;
  for (var k = 0; k < 3; k++) {
    let f = clamp(lv - f32(k), 0.0, 1.0);        // how far this level's holes have opened
    if (f <= 0.0) { break; }
    // the point in its cell (2/s across) of this level, and the three bars bored through the cell's middle
    let a = (fract(q * s * 0.5 + 0.5) * 2.0 - 1.0) / s;
    let h = f / (3.0 * s);
    let k2 = min(r, h);
    let cr = smin(mg_bar(a.yz, h, r), smin(mg_bar(a.zx, h, r), mg_bar(a.xy, h, r), k2), k2);
    d = smax(d, -cr, k2);
    s *= 3.0;
  }
  return d * MG_S;
}";

const SIERPINSKI: &str = "
// unsigned distance to the triangle a, b, c (Inigo Quilez's)
fn sp_tri(p: vec3f, a: vec3f, b: vec3f, c: vec3f) -> f32 {
  let ba = b - a; let pa = p - a;
  let cb = c - b; let pb = p - b;
  let ac = a - c; let pc = p - c;
  let nor = cross(ba, ac);
  let inside = sign(dot(cross(ba, nor), pa)) + sign(dot(cross(cb, nor), pb)) + sign(dot(cross(ac, nor), pc)) >= 2.0;
  if (!inside) {
    let e1 = ba * clamp(dot(ba, pa) / dot(ba, ba), 0.0, 1.0) - pa;
    let e2 = cb * clamp(dot(cb, pb) / dot(cb, cb), 0.0, 1.0) - pb;
    let e3 = ac * clamp(dot(ac, pc) / dot(ac, ac), 0.0, 1.0) - pc;
    return sqrt(min(min(dot(e1, e1), dot(e2, e2)), dot(e3, e3)));
  }
  return abs(dot(nor, pa)) * inverseSqrt(dot(nor, nor));
}
// the regular tetrahedron with corners (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1), shrunk by r and swollen
// again: its edges rounded to r, exactly
fn sp_tetra(p: vec3f, r: f32) -> f32 {
  let k = max(1.0 - r * 1.7320508, 0.05);         // the inradius is 1/sqrt(3)
  let a = vec3f(1.0, 1.0, 1.0) * k;
  let b = vec3f(1.0, -1.0, -1.0) * k;
  let c = vec3f(-1.0, 1.0, -1.0) * k;
  let e = vec3f(-1.0, -1.0, 1.0) * k;
  let un = min(min(sp_tri(p, a, b, c), sp_tri(p, a, b, e)), min(sp_tri(p, a, c, e), sp_tri(p, b, c, e)));
  let inside = max(max(-p.x - p.y - p.z, -p.x + p.y + p.z), max(p.x - p.y + p.z, p.x + p.y - p.z)) < k;
  return select(un, -un, inside) - r;
}
fn raw(p0: vec3f) -> f32 {
  // upright, then stood on a face: the corner (1,1,1) straight up
  let u = upright(p0);
  var z = (vec3f(0.70710678, -0.70710678, 0.0) * u.x + vec3f(0.57735027) * u.y + vec3f(-0.40824829, -0.40824829, 0.81649658) * u.z) / SP_S;
  let lv = clamp(pr.level, 0.0, 6.0);
  let kk = i32(floor(lv));
  let t = lv - f32(kk);
  let rw = max(pr.round, fine()) / SP_S;          // the rounding, in the starting units
  var sc = 1.0;
  var z0 = z;
  var s0 = 1.0;
  for (var i = 0; i < 7; i++) {
    if (i > kk) { break; }
    // fold onto the copy at the corner (1,1,1): the mirrors x + y = 0, x + z = 0, y + z = 0
    if (z.x + z.y < 0.0) { z = vec3f(-z.y, -z.x, z.z); }
    if (z.x + z.z < 0.0) { z = vec3f(-z.z, z.y, -z.x); }
    if (z.y + z.z < 0.0) { z = vec3f(z.x, -z.z, -z.y); }
    z = 2.0 * z - vec3f(1.0);
    sc *= 2.0;
    if (i + 1 == kk) { z0 = z; s0 = sc; }
  }
  // the levels either side of the one asked for, blended
  let d0 = sp_tetra(z0, rw * s0) / s0;
  let d1 = sp_tetra(z, rw * sc) / sc;
  return mix(d0, d1, t) * SP_S;
}";

// ------------------------------------------------------------------------------------------- inversion

/// The Apollonian packing's first five spheres and its five inversions, worked out here: four equal balls
/// touching one another and the unit sphere round them, and for each set of four of those five spheres
/// the sphere at right angles to all four (inverting in it fixes those four and sends the fifth into the
/// gap they leave). Returns WGSL constants and the code that measures and folds.
fn apollonian() -> String {
    use std::fmt::Write;
    // the four balls at the corners of a tetrahedron, touching each other and the unit sphere
    let dirs: [V; 4] = [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]].map(unit);
    let r = 1.0 / (1.0 + (1.5f64).sqrt());          // 2r = edge of the tetrahedron of centres = c·sqrt(8/3), c = 1 - r
    let c = 1.0 - r;
    let balls: Vec<(V, f64)> = dirs.iter().map(|d| (scale(*d, c), r)).collect();
    // the unit sphere, as one of the five
    let mut spheres = balls.clone();
    spheres.push(([0.0; 3], 1.0));
    // the sphere at right angles to every sphere but the one left out: |C - c_k|² = R² + r_k²
    let dual = |skip: usize| -> (V, f64) {
        let others: Vec<(V, f64)> = (0..5).filter(|&k| k != skip).map(|k| spheres[k]).collect();
        // subtracting the first equation from the others leaves three linear ones in C
        let (c0, r0) = others[0];
        let mut a = [[0.0; 3]; 3];
        let mut b = [0.0; 3];
        for i in 0..3 {
            let (ci, ri) = others[i + 1];
            for j in 0..3 { a[i][j] = 2.0 * (ci[j] - c0[j]); }
            b[i] = dot(ci, ci) - dot(c0, c0) - ri * ri + r0 * r0;
        }
        let det = |m: [[f64; 3]; 3]| m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let d = det(a);
        let mut x = [0.0; 3];
        for j in 0..3 {
            let mut m = a;
            for i in 0..3 { m[i][j] = b[i]; }
            x[j] = det(m) / d;
        }
        let rr = dot(sub(x, c0), sub(x, c0)) - r0 * r0;
        (x, rr)
    };
    let mut w = String::from("// the four first balls (centre, radius) and the five spheres of inversion (centre, radius squared)\n");
    for (i, (cc, rr)) in balls.iter().enumerate() {
        writeln!(w, "const AP_B{i}: vec4f = vec4f({:.9}, {:.9}, {:.9}, {:.9});", cc[0], cc[1], cc[2], rr).unwrap();
    }
    for k in 0..5 {
        let (cc, rr) = dual(k);
        writeln!(w, "const AP_D{k}: vec4f = vec4f({:.9}, {:.9}, {:.9}, {:.9});", cc[0], cc[1], cc[2], rr).unwrap();
    }
    w += "
fn ap_dual(j: i32) -> vec4f {
  switch j {
    case 0: { return AP_D0; }
    case 1: { return AP_D1; }
    case 2: { return AP_D2; }
    case 3: { return AP_D3; }
    default: { return AP_D4; }
  }
}
// A sphere (centre, signed radius: below 0 = all that lies outside it) turned inside out in the sphere d
// (centre, radius squared). A sphere round the centre of inversion comes back inside out.
fn ap_sphere(s: vec4f, d: vec4f) -> vec4f {
  let e = s.xyz - d.xyz;
  let k = d.w / (dot(e, e) - s.w * s.w);
  return vec4f(d.xyz + e * k, s.w * k);
}
// signed distance from p to the glass of the sphere s, made smaller (or the hole larger) by the gap
fn ap_dist(p: vec3f, s: vec4f) -> f32 {
  let l = length(p - s.xyz);
  return select(abs(s.w) * (1.0 + pr.gap) - l, l - s.w * (1.0 - pr.gap), s.w > 0.0);
}
// how far the region (a sphere, signed radius: below 0 = all outside it) is from p; 0 inside it
fn ap_near(p: vec3f, s: vec4f) -> f32 {
  let l = length(p - s.xyz);
  return max(select(-s.w - l, l - s.w, s.w > 0.0), 0.0);
}
// the ball a node of the tree adds: the sphere its last turn moves (s_m, or for the middle sphere the unit
// sphere, whose outside then becomes a ball), turned, and taken back through the turns before it
fn ap_ball(m: i32) -> vec4f {
  switch m {
    case 0: { return AP_B0; }
    case 1: { return AP_B1; }
    case 2: { return AP_B2; }
    case 3: { return AP_B3; }
    default: { return vec4f(0.0, 0.0, 0.0, -1.0); }
  }
}
fn raw(p: vec3f) -> f32 {
  // raw: a distance
  // The packing is a tree: the first four balls; under them five gaps, each filled by a ball (the fifth
  // sphere turned inside out in the sphere over the gap); each new ball leaves four gaps, filled the same
  // way in the frame turned inside out, and so on. Every ball is found where it really is by taking it back
  // through the turns that made its frame, and measured exactly. A whole branch lies inside the sphere over
  // its gap (taken back the same way), and is passed over when that is no nearer than the best so far.
  let q = p / AP_S;
  let lv = clamp(pr.depth, 0.0, 5.0);
  let kk = i32(floor(lv));
  let t = lv - f32(kk);
  // (far from every ball the distance need not be exact: held to a bound, which keeps the search short)
  var bestK = min(min(ap_dist(q, AP_B0), ap_dist(q, AP_B1)), min(ap_dist(q, AP_B2), ap_dist(q, AP_B3)));
  bestK = min(bestK, 0.1);
  var best1 = bestK;
  var chain: array<i32, 8>;
  var next: array<i32, 8>;
  var n = 0;
  next[0] = 0;
  for (var guard = 0; guard < 2048; guard++) {
    let m = next[n];
    if (m >= 5) {
      if (n == 0) { break; }
      n--;
      continue;
    }
    next[n] = m + 1;
    if (n > 0 && m == chain[n - 1]) { continue; }         // back the way it came
    if (n > kk) { continue; }                              // deeper than the depth asked for (and one more)
    // the gap's sphere, where it really is: a bound on all below it
    let dm = ap_dual(m);
    var reg = vec4f(dm.xyz, sqrt(dm.w));
    var ball = ap_sphere(ap_ball(m), dm);
    for (var c = n - 1; c >= 0; c--) {
      let dc = ap_dual(chain[c]);
      reg = ap_sphere(reg, dc);
      ball = ap_sphere(ball, dc);
    }
    if (ap_near(q, reg) >= bestK) { continue; }
    let d = ap_dist(q, ball);
    best1 = min(best1, d);
    if (n < kk) { bestK = min(bestK, d); }
    // down into its gaps
    if (n < kk) {
      chain[n] = m;
      n++;
      next[n] = 0;
    }
  }
  return mix(bestK, best1, t) * AP_S;
}";
    w
}

// ------------------------------------------------------------------------------------------------ the family

fn presets() -> Vec<Preset> {
    vec![
        frac(Frac {
            id: "mandelbulb", name: "Mandelbulb",
            eq: &["z ↦ zⁿ + c,  z = (r, θ, φ) ↦ (rⁿ, nθ, nφ)", "the level |z_N| = 2 after N steps"],
            note: "White and Nylander’s three-dimensional Mandelbrot set (2009): raising a point to a power multiplies its angles and powers its distance. Each step adds a ring of buds to the buds before. The glass is the set with a skin about it: thin, every bud shows; thick, they melt into a ball.",
            params: vec![
                ParamDef::labelled("power", "power n", 8.0, 2.0, 12.0, 1.0),
                ParamDef::labelled("depth", "steps N", 4.0, 2.0, 9.0, 0.01),
                ParamDef::labelled("skin", "skin", 0.022, 0.01, 0.08, 0.001),
            ],
            rot: [-75.0, 20.0, 0.0],
            animate: "skin",
            wgsl: format!("const BULB_S: f32 = {:.4};\n{DEPTH}{MANDELBULB}", FIT / 1.14),
            clip: "length(p) - 1.38",
        }),
        frac(Frac {
            id: "mandelbox", name: "Mandelbox",
            eq: &["z ↦ s·ball(box(z)) + c", "box: fold at |x| = f;  ball: z ↦ z ⁄ |z|² inside |z| = 1"],
            note: "Tom Lowe’s fractal (2010): space is folded at the faces of a cube and turned inside out in a ball, then scaled. With s near 2 it is a city of towers; negative s carves it into blocks and ruins. Between −1.25 and 1.25 the scale holds at ±1.25.",
            params: vec![
                ParamDef::labelled("scale", "scale s", -2.6, -3.0, 3.0, 0.01),
                ParamDef::labelled("fold", "folding f", 1.0, 0.5, 1.5, 0.01),
                ParamDef::labelled("inner", "inner ball", 0.5, 0.2, 0.9, 0.01),
                ParamDef::labelled("depth", "steps N", 7.0, 3.0, 11.0, 0.01),
                ParamDef::labelled("skin", "skin", 0.028, 0.012, 0.08, 0.001),
            ],
            rot: view([25.0, 35.0, 0.0]),
            animate: "depth",
            wgsl: format!("const BOX_FIT: f32 = {:.4};\n{FINE}{}{DEPTH}{MANDELBOX}", FIT * 0.82, unturn()),
            clip: "length(p) - 1.38",
        }),
        frac(Frac {
            id: "menger", name: "Menger sponge",
            eq: &["a cube, cut into 27; the middle one and the six", "at the middles of the faces taken out; again in each of the 20 left"],
            note: "Karl Menger’s sponge (1926): every level keeps 20 of every 27 cubes, so its volume goes to nothing while its surface grows without end. Between whole levels the next holes open.",
            params: vec![
                ParamDef::labelled("level", "level", 2.0, 0.0, 3.0, 0.01),
                ParamDef::labelled("round", "rounding", 0.008, 0.0, 0.06, 0.002),
            ],
            rot: view([28.0, 38.0, 0.0]),
            animate: "level",
            wgsl: format!("// raw: exact inside, never more than the distance outside\nconst MG_S: f32 = {:.4};\n{FINE}{}{MENGER}", FIT / 3f64.sqrt(), unturn()),
            clip: "-1.0",
        }),
        frac(Frac {
            id: "sierpinski", name: "Sierpinski tetrahedron",
            eq: &["a tetrahedron replaced by four, half its size,", "at its corners; again in each"],
            note: "Every level halves the edges and keeps four of the eight halves’ worth: the octahedron in the middle is left out. Its volume goes to nothing; seen along an edge it is a square.",
            params: vec![
                ParamDef::labelled("level", "level", 3.0, 0.0, 6.0, 0.01),
                ParamDef::labelled("round", "rounding", 0.012, 0.0, 0.06, 0.002),
            ],
            rot: view([18.0, 25.0, 0.0]),
            animate: "level",
            wgsl: format!("const SP_S: f32 = {:.4};\n{FINE}{}{SIERPINSKI}", FIT / 3f64.sqrt(), unturn()),
            clip: "-1.0",
        }),
        frac(Frac {
            id: "apollonian", name: "Apollonian packing",
            eq: &["five spheres, each touching the other four;", "every gap filled by the sphere that touches its four walls"],
            note: "Four balls in a ball, each gap filled by the largest ball that fits, and again in the gaps that leaves. Turning a gap inside out in the sphere across it gives every ball from the first five.",
            params: vec![
                ParamDef::labelled("depth", "generations", 3.0, 0.0, 5.0, 0.01),
                ParamDef::labelled("gap", "gap", 0.04, 0.0, 0.2, 0.005),
            ],
            rot: [20.0, 28.0, 0.0],
            animate: "depth",
            wgsl: format!("const AP_S: f32 = {FIT:.4};\n{}", apollonian()),
            clip: "-1.0",
        }),
        frac(Frac {
            id: "quatjulia", name: "Quaternion Julia set",
            eq: &["q ↦ q² + c  in the quaternions,", "the level |q_N| = 4 cut at w"],
            note: "A Julia set in four dimensions, cut through at a fourth coordinate w. Every c gives another; near the edge of the Mandelbrot set they are most intricate.",
            params: vec![
                ParamDef::labelled("cr", "c real", 0.18, -1.2, 0.6, 0.005),
                ParamDef::labelled("ci", "c i", 0.88, -1.0, 1.0, 0.005),
                ParamDef::labelled("cj", "c j", 0.24, -1.0, 1.0, 0.005),
                ParamDef::labelled("ck", "c k", 0.16, -1.0, 1.0, 0.005),
                ParamDef::labelled("slice", "cut at w", 0.0, -0.8, 0.8, 0.005),
                ParamDef::labelled("depth", "steps N", 9.0, 1.0, 11.0, 0.01),
                ParamDef::labelled("skin", "skin", 0.02, 0.008, 0.08, 0.001),
            ],
            rot: [20.0, 30.0, 0.0],
            animate: "cj",
            wgsl: format!("const JULIA_S: f32 = {:.4};\n{DEPTH}{JULIA}", FIT / 1.15),
            clip: "length(p) - 1.38",
        }),
    ]
}

fn light(az: f64, el: f64, w: f64, h: f64, rgb: [f64; 3], soft: f64) -> Light {
    Light { az, el, w, h, rgb, soft }
}

pub fn family() -> Family {
    Family {
        step_scale: 0.85,
        steps_out: 220,
        steps_in: 64,
        layers: 2,
        presets: presets(),
        defaults: vec![
            ("ior", Setting::Num(1.5)), ("disp", Setting::Num(0.07)), ("absorb", Setting::Vec3([0.26, 0.12, 0.05])),
            ("ns", Setting::Num(1.0)), ("nw", Setting::Num(3.0)), ("nb", Setting::Num(3.0)),
            ("dist", Setting::Num(6.6)), ("fov", Setting::Num(26.0)), ("rot", Setting::Vec3([22.0, 30.0, 0.0])),
            ("spin", Setting::Num(0.25)), ("bloom", Setting::Num(0.35)), ("exposure", Setting::Num(1.0)),
        ],
        // the minimal surfaces' lights: broad and soft, with the cold and warm ones either side
        lights: vec![
            light(38.0, 18.0, 14.0, 32.0, [5.0, 4.9, 4.7], 0.7),
            light(-55.0, -8.0, 10.0, 42.0, [0.5, 1.4, 4.0], 0.7),
            light(-140.0, 5.0, 8.0, 42.0, [5.0, 1.6, 0.25], 0.7),
            light(150.0, 28.0, 24.0, 16.0, [3.2, 3.1, 3.0], 0.8),
            light(0.0, 85.0, 34.0, 34.0, [0.8, 0.9, 1.1], 0.9),
            light(0.0, -80.0, 30.0, 30.0, [0.5, 0.22, 0.05], 0.9),
        ],
        ..Family::new("fractals", "Fractals", Kind::Field)
    }
}
