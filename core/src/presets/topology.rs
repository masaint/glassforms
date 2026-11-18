//! Topology: surfaces that are interesting for their shape as a whole, in glass. One-sided surfaces
//! (the Klein bottle, Boy's surface, the Roman surface), minimal surfaces that are not periodic (Costa,
//! Enneper, Scherk's tower, the helicoid and catenoid), surfaces of constant negative curvature (the
//! breather, Kuen's and Dini's surfaces) and a slice of a Calabi–Yau manifold.
//!
//! A surface has no inside, so each is glass as a wall about the surface: `raw` is negative on one side
//! and positive on the other, or (where the surface has no sides, or edges) a distance to the surface that
//! is never negative, and the bake lays the wall (`shell` = the wall's half-thickness) about its zero. The
//! walk the bake takes to the surface turns either into true distances, so `raw` need not be a distance.
//! How each is built:
//!
//!   implicit       the zero set of an equation: Boy (Apéry's sextic), Roman (Steiner's quartic), Scherk's
//!                  tower, the helicoid, the catenoid.
//!   pieces         the Klein bottle: a profile of arcs and straight lines turned about the axis, and a
//!                  pipe along arcs and straight lines; the distance to each is exact, and so the whole is.
//!   closest point  Enneper: the nearest point of the patch, found from a coarse search and Newton steps.
//!   a slice        surfaces with a symmetry or a monotone angle (Dini's screw, Kuen, the breather): the
//!                  angle about the axis fixes one parameter, which leaves a curve in a half-plane; the
//!                  distance to that curve, in the half-plane, is zero exactly on the surface.
//!   the fibre      the Calabi–Yau slice: the projection of a complex curve in C² to space; a point of
//!                  space is on it when the line of C² above it meets the curve.
use super::*;

/// Every preset here gives these.
struct Topo {
    id: &'static str,
    name: &'static str,
    eq: &'static [&'static str],
    note: &'static str,
    params: Vec<ParamDef>,
    rot: [f64; 3],
    animate: &'static str,
    wgsl: String,
    clip: String,
}

fn wall(v: f64) -> ParamDef { ParamDef::labelled("wall", "wall", v, 0.015, 0.09, 0.001) }

fn topo(t: Topo) -> Preset {
    Preset {
        eq: t.eq.to_vec(),
        note: t.note,
        params: t.params,
        rot: Some(t.rot),
        animate: Some(vec![t.animate]),
        ..Preset::new(t.id, t.name, Shape::Field { wgsl: t.wgsl, clip: t.clip, shell: "pr.wall".to_string() })
    }
}

/// A capped cylinder about y: radius r, half-height h (a true distance).
fn cylinder(r: &str, h: &str) -> String {
    format!("length(max(vec2f(length(p.xz) - ({r}), abs(p.y) - ({h})), vec2f(0.0))) + min(max(length(p.xz) - ({r}), abs(p.y) - ({h})), 0.0)")
}

// ---- the Klein bottle ---------------------------------------------------------------------------
//
// As glassblowers make it. A bulb turned about the vertical axis whose profile is a chain of arcs and
// straight lines: down from the neck through a concave fillet, round the bulb, straight down, and round
// a U-turn at the foot that turns it inside out and up into a short inner tube. A pipe, of the neck's
// radius, leaves the top of the neck, arches over, comes down outside, turns in through the side of the
// bulb and down the axis into that inner tube. Where it passes the bulb's wall, the wall is cut away
// (`clip`) and fused to the pipe. Everything is in the bottle's own units, scaled to fit.
const KLEIN: &str = "
// raw: a distance (to the bulb and to the pipe, each exactly)
// the distance in a plane from v to the arc of the circle (centre c, radius r) whose middle lies in
// direction m and which spans h either side of it
fn arcd(v: vec2f, c: vec2f, r: f32, m: vec2f, h: f32) -> f32 {
  let d = v - c;
  let l = length(d);
  if (dot(d, m) >= cos(h) * l) { return abs(l - r); }
  let s = sin(h);
  let co = cos(h);
  let e1 = c + r * vec2f(m.x * co - m.y * s, m.x * s + m.y * co);
  let e2 = c + r * vec2f(m.x * co + m.y * s, -m.x * s + m.y * co);
  return min(length(v - e1), length(v - e2));
}
fn segd(v: vec2f, a: vec2f, b: vec2f) -> f32 {
  let ab = b - a;
  return length(v - a - ab * clamp(dot(v - a, ab) / dot(ab, ab), 0.0, 1.0));
}
struct Kb { s: f32, mid: vec2f, rn: f32, rb: f32, yc: f32, y0: f32, yj: f32, ytop: f32, rp: f32, yh: f32 }
const KB_RC: f32 = 0.22;        // the fillet under the neck
const KB_R2: f32 = 0.2;         // the pipe's two bends into the bulb
fn kb() -> Kb {
  var k: Kb;
  k.rn = pr.neck;
  k.rb = pr.bulb;
  k.yc = -0.1;
  k.y0 = -0.45;
  k.yj = k.yc + sqrt((k.rb + KB_RC) * (k.rb + KB_RC) - (k.rn + KB_RC) * (k.rn + KB_RC));
  k.ytop = k.yj + 0.45;
  k.rp = 0.5 * (k.rb + k.rn) + 0.12;
  k.yh = k.yc + pr.enter * k.rb;
  // the bottle's box, and the scale that makes its larger half-size 1.2
  let lo = vec2f(-k.rb, k.y0 - 0.5 * (k.rb - k.rn));
  let hi = vec2f(2.0 * k.rp + k.rn, k.ytop + k.rp + k.rn);
  k.mid = 0.5 * (lo + hi);
  k.s = 1.2 / max(0.5 * (hi.x - lo.x), 0.5 * (hi.y - lo.y));
  return k;
}
// the pipe's middle line, from the top of the neck to the inner tube; `lumen`: only the stretch that
// passes the bulb's wall
fn kb_spine(q: vec3f, k: Kb, lumen: bool) -> f32 {
  let v = q.xy;
  let z2 = q.z * q.z;
  let x2 = 2.0 * k.rp;
  var d = 1e9;
  d = min(d, arcd(v, vec2f(x2 - KB_R2, k.yh + KB_R2), KB_R2, vec2f(0.70710678, -0.70710678), 0.78539816));
  d = min(d, segd(v, vec2f(x2 - KB_R2, k.yh), vec2f(KB_R2, k.yh)));
  d = min(d, arcd(v, vec2f(KB_R2, k.yh - KB_R2), KB_R2, vec2f(-0.70710678, 0.70710678), 0.78539816));
  if (!lumen) {
    if (q.y >= k.yj) { d = min(d, segd(v, vec2f(0.0, k.yj), vec2f(0.0, k.ytop))); }
    d = min(d, arcd(v, vec2f(k.rp, k.ytop), k.rp, vec2f(0.0, 1.0), 1.5707963));
    d = min(d, segd(v, vec2f(x2, k.ytop), vec2f(x2, k.yh + KB_R2)));
    if (q.y >= k.y0) { d = min(d, segd(v, vec2f(0.0, k.yh - KB_R2), vec2f(0.0, k.y0))); }
  }
  return sqrt(d * d + z2);
}
fn raw(p: vec3f) -> f32 {
  let k = kb();
  let q = p / k.s + vec3f(k.mid, 0.0);
  // the bulb: the profile in the half-plane (distance from the axis, height)
  let w = vec2f(length(q.xz), q.y);
  let a = vec2f(k.rn + KB_RC, k.yj);
  let toBig = normalize(vec2f(0.0, k.yc) - a);
  let m1 = normalize(vec2f(-1.0, 0.0) + toBig);
  var f = arcd(w, a, KB_RC, m1, 0.5 * acos(clamp(-toBig.x, -1.0, 1.0)));
  let up = -toBig;                                          // from the bulb's centre to the fillet
  let m2 = normalize(up + vec2f(1.0, 0.0));
  f = min(f, arcd(w, vec2f(0.0, k.yc), k.rb, m2, 0.5 * acos(clamp(up.x, -1.0, 1.0))));
  f = min(f, segd(w, vec2f(k.rb, k.yc), vec2f(k.rb, k.y0)));
  f = min(f, arcd(w, vec2f(0.5 * (k.rb + k.rn), k.y0), 0.5 * (k.rb - k.rn), vec2f(0.0, -1.0), 1.5707963));
  // the pipe: its wall, and its two open ends (the circles where it meets the bulb's neck and inner tube)
  var t = abs(kb_spine(q, k, false) - k.rn);
  if (q.y < k.yj) { t = min(t, length(vec2f(w.x - k.rn, q.y - k.yj))); }
  if (q.y < k.y0) { t = min(t, length(vec2f(w.x - k.rn, q.y - k.y0))); }
  return min(f, t) * k.s;
}
// how far inside the pipe's bore a point is, where the pipe crosses the bulb's wall (the hole cut there)
fn kbHole(p: vec3f) -> f32 {
  let k = kb();
  let q = p / k.s + vec3f(k.mid, 0.0);
  return (k.rn * k.s - pr.wall) - kb_spine(q, k, true) * k.s;
}";

// ---- implicit surfaces --------------------------------------------------------------------------

const BOY: &str = "
// Apéry's sextic, with its axis of threefold symmetry upright and its middle at the centre
fn raw(p: vec3f) -> f32 {
  let q = vec3f(p.x, p.z, p.y) * 1.27 + vec3f(0.0, 0.0, 1.0);
  let x = q.x; let y = q.y; let z = q.z;
  let r2 = x * x + y * y;
  let z2 = z * z;
  let w = 1.0 - z;
  const S2: f32 = 1.41421356;
  let f = 64.0 * w * w * w * z2 * z - 48.0 * w * w * z2 * (3.0 * r2 + 2.0 * z2)
        + 12.0 * w * z * (27.0 * r2 * r2 - 24.0 * z2 * r2 + 36.0 * S2 * y * z * (y * y - 3.0 * x * x) + 4.0 * z2 * z2)
        + (9.0 * r2 - 2.0 * z2) * (-81.0 * r2 * r2 - 72.0 * z2 * r2 + 108.0 * S2 * x * z * (x * x - 3.0 * y * y) + 4.0 * z2 * z2);
  let g = 1.0 + dot(q, q);
  return f / (g * g * g);
}";

const ROMAN: &str = "
fn raw(p: vec3f) -> f32 {
  let q = p * 0.47;
  let a = q * q;
  return (a.x * a.y + a.y * a.z + a.z * a.x - q.x * q.y * q.z) / (dot(q, q) + 0.02);
}";

const SCHERK: &str = "
fn raw(p: vec3f) -> f32 {
  let k = 6.2831853 * pr.cells / 2.2;
  // asinh of both sides: the same surface, with a slope that stays near one far from the axis
  return (asinh(sinh(k * p.x) * sinh(k * p.z)) - asinh(sin(k * p.y))) / k;
}";

const HELICOID: &str = "
fn raw(p: vec3f) -> f32 {
  let a = pr.turns * 3.14159265 * p.y / 1.05;
  return p.x * sin(a) - p.z * cos(a);
}";

const CATENOID: &str = "
fn raw(p: vec3f) -> f32 {
  let a = pr.neck;
  return length(p.xz) - a * cosh(p.y / a);
}";

// ---- surfaces found by search -------------------------------------------------------------------

// Enneper's surface: x + iy = w − w̄³⁄3, z = Re w², over the disc |w| ≤ R. It is symmetric in its x and in
// its y, so a point is folded into x, y ≥ 0, which the quarter disc u, v ≥ 0 covers; there the nearest
// point is found on a coarse grid and settled by Gauss–Newton steps held to the quarter disc. Held to the
// disc, the nearest point of a point beyond the rim is on the rim: the wall's edge comes out round.
const ENNEPER: &str = "
// raw: a distance (to the nearest point of the patch)
fn en_s(w: vec2f) -> vec3f {
  let u = w.x;
  let v = w.y;
  return vec3f(u - u * u * u / 3.0 + u * v * v, v - v * v * v / 3.0 + u * u * v, u * u - v * v);
}
fn raw(p: vec3f) -> f32 {
  let R = pr.reach;
  let sc = 1.22 / (R + R * R * R / 3.0);
  let q = vec3f(abs(p.x), abs(p.z), p.y) / sc;
  var best = 1e9;
  var w = vec2f(0.0);
  for (var i = 0; i <= 6; i++) {
    for (var j = 0; j <= 6; j++) {
      let t = 1.5707963 * f32(j) / 6.0;
      let c = R * f32(i) / 6.0 * vec2f(cos(t), sin(t));
      let d = en_s(c) - q;
      let dd = dot(d, d);
      if (dd < best) { best = dd; w = c; }
    }
  }
  for (var k = 0; k < 6; k++) {
    let u = w.x;
    let v = w.y;
    let su = vec3f(1.0 - u * u + v * v, 2.0 * u * v, 2.0 * u);
    let sv = vec3f(2.0 * u * v, 1.0 - v * v + u * u, -2.0 * v);
    let e = q - en_s(w);
    let a = dot(su, su);
    let b = dot(su, sv);
    let c = dot(sv, sv);
    let g = vec2f(dot(su, e), dot(sv, e));
    var dw = vec2f(c * g.x - b * g.y, a * g.y - b * g.x) / max(a * c - b * b, 1e-6);
    let l = length(dw);
    if (l > 0.25) { dw *= 0.25 / l; }
    w = max(w + dw, vec2f(0.0));
    let lw = length(w);
    if (lw > R) { w *= R / lw; }
  }
  return length(en_s(w) - q) * sc;
}";

// Dini's surface: a pseudosphere's tractrix screwed up its axis, x + iy = a sin v·e^(iu),
// z = a(cos v + ln tan(v⁄2)) + b·u. A screw carries it into itself, so in the half-plane through a point
// it is a few copies of the tractrix, one for each turn: the point's distance to them there, cut by the
// slope the screw adds (the first-order correction), is its distance to the surface; the two ends of the
// strip are tractrices in their own planes, and their distances are exact.
const DINI: &str = "
// raw: a distance (to first order)
const DI_V0: f32 = 0.25;
fn di_c(v: f32) -> vec2f { return vec2f(sin(v), cos(v) + log(tan(0.5 * v))); }
// distance in a half-plane from P to the tractrix (v from DI_V0 to vmax); and the upward share of the
// direction from it, or 2 when the nearest point is an end
fn di_prof(P: vec2f, vmax: f32) -> vec2f {
  var v = DI_V0;
  var bd = 1e9;
  for (var i = 0; i < 16; i++) {
    let s = mix(DI_V0, vmax, f32(i) / 15.0);
    let d = length(P - di_c(s));
    if (d < bd) { bd = d; v = s; }
  }
  for (var k = 0; k < 4; k++) {
    let c = di_c(v);
    let sv = sin(v);
    let cv = cos(v);
    let g = dot(P - c, vec2f(sv, cv));
    let dg = -cv / sv + dot(P - c, vec2f(cv, -sv));
    var st = select(0.0, g / dg, abs(dg) > 1e-4);
    st = clamp(st, -0.2, 0.2);
    v = clamp(v - st, DI_V0, vmax);
  }
  let dv = P - di_c(v);
  let l = length(dv);
  let end = v <= DI_V0 + 1e-4 || v >= vmax - 1e-4;
  return vec2f(l, select(dv.y / max(l, 1e-6), 2.0, end));
}
struct Di { sc: f32, mid: f32, turns: f32 }
fn di() -> Di {
  let U = 6.2831853 * pr.turns;
  let lo = di_c(DI_V0).y;
  let hi = di_c(pr.flare).y + pr.climb * U;
  let rmax = select(sin(pr.flare), 1.0, pr.flare > 1.5707963);
  return Di(1.2 / max(rmax, 0.5 * (hi - lo)), 0.5 * (lo + hi), U);
}
fn raw(p: vec3f) -> f32 {
  let k = di();
  let b = pr.climb;
  let q = p / k.sc + vec3f(0.0, k.mid, 0.0);
  let rho = length(q.xz);
  var th = atan2(q.z, q.x);
  if (th < 0.0) { th += 6.2831853; }
  var d = 1e9;
  for (var i = 0; i < 4; i++) {
    let u = th + 6.2831853 * f32(i);
    if (u > k.turns) { break; }
    let r = di_prof(vec2f(rho, q.y - b * u), pr.flare);
    let tilt = b * r.y / max(rho, 0.03);
    d = min(d, select(r.x / sqrt(1.0 + tilt * tilt), r.x, r.y > 1.5));
  }
  // the strip's two ends, each a tractrix in its own plane
  for (var e = 0; e < 2; e++) {
    let u = select(0.0, k.turns, e == 1);
    let dir = vec2f(cos(u), sin(u));
    let r = di_prof(vec2f(dot(q.xz, dir), q.y - b * u), pr.flare);
    let o = q.x * dir.y - q.z * dir.x;
    d = min(d, sqrt(r.x * r.x + o * o));
  }
  return d * k.sc;
}";

// Kuen's surface: x + iy = 2 sin v (1 − iu) e^(iu) ⁄ (1 + u² sin² v), z = ln tan(v⁄2) + 2 cos v ⁄ (1 + u² sin² v).
// Its angle about the axis, u − atan u, grows with u, so the angle of a point fixes u (by halving), and
// in the half-plane through the point the surface is one curve in v, whose nearest point is found by a
// coarse search. That point starts Gauss–Newton steps in (u, v) to the nearest point of the surface
// itself (the nearest seen is kept: the surface is curved hard), and the distance is to that point.
// Symmetric in y (u ↦ −u): folded into y ≥ 0.
const KUEN: &str = "
// raw: a distance (to the nearest point found)
const KU_V0: f32 = 0.2;
const KU_SC: f32 = 0.8;
fn ku_s(u: f32, v: f32) -> vec3f {
  let sv = sin(v);
  let den = 1.0 + u * u * sv * sv;
  return vec3f(2.0 * (cos(u) + u * sin(u)) * sv / den, 2.0 * (sin(u) - u * cos(u)) * sv / den, log(tan(0.5 * v)) + 2.0 * cos(v) / den);
}
// the curve at u in its half-plane: (distance from the axis, height)
fn ku_c(u: f32, v: f32) -> vec2f {
  let sv = sin(v);
  let den = 1.0 + u * u * sv * sv;
  return vec2f(2.0 * sv * sqrt(1.0 + u * u) / den, log(tan(0.5 * v)) + 2.0 * cos(v) / den);
}
fn ku_angle(u: f32) -> f32 { return u - atan(u); }
// the u in 0..U whose angle is a
fn ku_u(a: f32, U: f32) -> f32 {
  var lo = 0.0;
  var hi = U;
  for (var i = 0; i < 20; i++) {
    let m = 0.5 * (lo + hi);
    if (ku_angle(m) < a) { lo = m; } else { hi = m; }
  }
  return 0.5 * (lo + hi);
}
// the distance from q (in the formula's frame) to the surface, searched from the point of the curve at u0
// nearest P (in the half-plane); `edge`: u is held (the search runs along the edge u = U)
fn ku_near(q: vec3f, P: vec2f, u0: f32, U: f32, edge: bool) -> f32 {
  let v1 = 3.14159265 - KU_V0;
  var u = u0;
  var v = 1.5707963;
  var bd = 1e9;
  for (var i = 0; i < 24; i++) {
    let s = mix(KU_V0, v1, f32(i) / 23.0);
    let d = length(P - ku_c(u, s));
    if (d < bd) { bd = d; v = s; }
  }
  var best = length(q - ku_s(u, v));
  let h = 1e-3;
  for (var k = 0; k < 5; k++) {
    let s = ku_s(u, v);
    let su = (ku_s(u + h, v) - ku_s(u - h, v)) / (2.0 * h);
    let sv = (ku_s(u, v + h) - ku_s(u, v - h)) / (2.0 * h);
    let e = q - s;
    let a = dot(su, su);
    let b = dot(su, sv);
    let c = dot(sv, sv);
    let g = vec2f(dot(su, e), dot(sv, e));
    var dw = vec2f(c * g.x - b * g.y, a * g.y - b * g.x) / max(a * c - b * b, 1e-9);
    if (edge) { dw = vec2f(0.0, g.y / max(c, 1e-9)); }
    let l = length(dw);
    if (l > 0.15) { dw *= 0.15 / l; }
    u = clamp(u + dw.x, -U, U);
    v = clamp(v + dw.y, KU_V0, v1);
    best = min(best, length(q - ku_s(u, v)));
  }
  return best;
}
fn raw(p: vec3f) -> f32 {
  let U = pr.spread;
  // the formula's frame, its z upright and its middle at the centre; folded into y >= 0
  let q = vec3f(p.x, abs(p.z), p.y) / KU_SC + vec3f(0.5, 0.0, 0.0);
  let rho = length(q.xy);
  let a = atan2(q.y, q.x);                       // 0..pi
  let amax = ku_angle(U);
  var d = 1e9;
  if (a <= amax) {
    d = min(d, ku_near(q, vec2f(rho, q.z), ku_u(a, U), U, false));
  }
  if (6.2831853 - a <= amax) {                   // the part of the sheet below y = 0, met from the fold
    d = min(d, ku_near(vec3f(q.x, -q.y, q.z), vec2f(rho, q.z), ku_u(6.2831853 - a, U), U, false));
  }
  // the edge at u = U, a curve in the half-plane at angle amax, and its mirror image
  let de = vec2f(cos(amax), sin(amax));
  for (var e = 0; e < 2; e++) {
    let qe = select(q, vec3f(q.x, -q.y, q.z), e == 1);
    d = min(d, ku_near(qe, vec2f(dot(qe.xy, de), qe.z), U, U, true));
  }
  return d * KU_SC;
}";

// The breather: a pseudospherical surface (curvature −1 everywhere), the picture of a breather solution of
// the sine-Gordon equation. With w = √(1 − a²) = m⁄(m + 1) it closes up round its axis after m ripples.
// Its angle about the axis grows with v, so a point's angle fixes v (by halving), and in the half-plane
// through the point the surface is one curve in u, searched as Kuen's is, and settled as Kuen's is.
const BREATHER: &str = "
// raw: a distance (to the nearest point found)
struct Br { w: f32, a: f32, V: f32, U: f32, sc: f32 }
fn br() -> Br {
  let m = round(pr.ripples);
  let w = m / (m + 1.0);
  let a = sqrt(1.0 - w * w);
  // across, the widest ripple is 2 / a; along the axis it reaches about U - 2 / a
  return Br(w, a, 3.14159265 * (m + 1.0), pr.reach, 1.2 / max(pr.reach - 2.0 / a, 2.0 / a));
}
fn br_s(k: Br, u: f32, v: f32) -> vec3f {
  let w = k.w;
  let a = k.a;
  let ch = cosh(a * u);
  let sw = sin(w * v);
  let cw = cos(w * v);
  let den = a * ((w * ch) * (w * ch) + (a * sw) * (a * sw));
  return vec3f(-u + 2.0 * (1.0 - a * a) * ch * sinh(a * u) / den,
               2.0 * w * ch * (-(w * cos(v) * cw) - sin(v) * sw) / den,
               2.0 * w * ch * (-(w * sin(v) * cw) + cos(v) * sw) / den);
}
// the angle about the x axis at v, continuous: from 0 at v = -V to 2 pi at v = V
fn br_angle(k: Br, v: f32) -> f32 {
  let t = k.w * v;
  let s = sin(t);
  let c = cos(t);
  return v + 3.14159265 - t - atan2((1.0 - k.w) * s * c, k.w * c * c + s * s);
}
fn br_c(k: Br, u: f32, v: f32) -> vec2f { let s = br_s(k, u, v); return vec2f(s.x, length(s.yz)); }
fn raw(p: vec3f) -> f32 {
  let k = br();
  let q = p.yxz / k.sc;                             // the formula's axis, x, upright
  var a = atan2(q.z, q.y);
  if (a < 0.0) { a += 6.2831853; }
  var lo = -k.V;
  var hi = k.V;
  for (var i = 0; i < 22; i++) {
    let m = 0.5 * (lo + hi);
    if (br_angle(k, m) < a) { lo = m; } else { hi = m; }
  }
  let v = 0.5 * (lo + hi);
  let P = vec2f(q.x, length(q.yz));
  var u = 0.0;
  var bd = 1e9;
  for (var i = 0; i <= 32; i++) {
    let s = mix(-k.U, k.U, f32(i) / 32.0);
    let d = length(P - br_c(k, s, v));
    if (d < bd) { bd = d; u = s; }
  }
  let h = 1e-3;
  var vv = v;
  var best = length(q - br_s(k, u, vv));
  for (var j = 0; j < 5; j++) {
    let s = br_s(k, u, vv);
    let su = (br_s(k, u + h, vv) - br_s(k, u - h, vv)) / (2.0 * h);
    let sv = (br_s(k, u, vv + h) - br_s(k, u, vv - h)) / (2.0 * h);
    let e = q - s;
    let a = dot(su, su);
    let b = dot(su, sv);
    let c = dot(sv, sv);
    let g = vec2f(dot(su, e), dot(sv, e));
    var dw = vec2f(c * g.x - b * g.y, a * g.y - b * g.x) / max(a * c - b * b, 1e-9);
    let l = length(dw);
    if (l > 0.3) { dw *= 0.3 / l; }
    u = clamp(u + dw.x, -k.U, k.U);
    vv += dw.y;
    best = min(best, length(q - br_s(k, u, vv)));
  }
  let dist = best;
  return dist * k.sc;
}";

// A slice of a Calabi–Yau manifold, as Andrew Hanson drew it: the complex curve z₁ⁿ + z₂ⁿ = 1 in C², which
// is four-dimensional, seen in three: x = Re z₁, y = Re z₂, z = cos α Im z₁ + sin α Im z₂. Above a point of
// space lies a line of C², one real number t along it; the point is on the surface where the equation is
// met somewhere on that line. So |f| is searched along the line (coarsely, then each low point by
// halving), and |f| over its slope at the nearest is the distance, to first order. Hanson's patches
// stop at |ξ| = ξmax, which is |z₁|ⁿ + |z₂|ⁿ = cosh 2ξmax: the clip cuts there.
const CALABI: &str = "
// raw: a distance (to first order, held to 0.3)
fn cm(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn cpw(z: vec2f, n: i32) -> vec2f {
  var r = vec2f(1.0, 0.0);
  for (var i = 0; i < 8; i++) { if (i >= n) { break; } r = cm(r, z); }
  return r;
}
// how far out on the curve a point is: |z1|^n + |z2|^n = cosh 2ξ in Hanson's coordinates
fn cy_h(z1: vec2f, z2: vec2f, n: f32) -> f32 { return pow(max(dot(z1, z1), 1e-12), 0.5 * n) + pow(max(dot(z2, z2), 1e-12), 0.5 * n); }
struct Cy { n: i32, cs: vec2f, K: f32, sc: f32 }
fn cy() -> Cy {
  let n = i32(clamp(round(pr.degree), 2.0, 8.0));
  let al = radians(pr.angle);
  let K = cosh(2.0 * pr.extent);
  return Cy(n, vec2f(cos(al), sin(al)), K, 1.0 / pow(K, 1.0 / f32(n)));
}
// what the search along the line minimises: |f|², and a penalty for leaving the patch
fn cy_cost(k: Cy, abc: vec3f, t: f32) -> f32 {
  let z1 = vec2f(abc.x, abc.z * k.cs.x - t * k.cs.y);
  let z2 = vec2f(abc.y, abc.z * k.cs.y + t * k.cs.x);
  let f = cpw(z1, k.n) + cpw(z2, k.n) - vec2f(1.0, 0.0);
  let out = max(cy_h(z1, z2, f32(k.n)) - k.K, 0.0);
  return dot(f, f) + 4.0 * out * out;
}
// the rim of the patch, as a distance (to first order) beyond it: kept from the last reading of raw
var<private> cy_at: vec3f = vec3f(1e9);
var<private> cy_rim: f32 = 0.0;
// From a low point of the search along the line (at t, where the cost is c): settled there by halving
// steps; then the distance to first order, |f| over its slope, and the rim's distance (into cy_rim2)
var<private> cy_rim2: f32 = 0.0;
fn cy_settle(k: Cy, abc: vec3f, t0: f32, c0: f32, step0: f32) -> f32 {
  var t = t0;
  var best = c0;
  var step = step0;
  for (var j = 0; j < 6; j++) {
    step *= 0.5;
    let m1 = cy_cost(k, abc, t - step);
    let m2 = cy_cost(k, abc, t + step);
    if (m1 < best && m1 <= m2) { best = m1; t -= step; } else if (m2 < best) { best = m2; t += step; }
  }
  let cs = k.cs;
  let n = k.n;
  let nn = f32(n);
  let z1 = vec2f(abc.x, abc.z * cs.x - t * cs.y);
  let z2 = vec2f(abc.y, abc.z * cs.y + t * cs.x);
  let f = cpw(z1, n) + cpw(z2, n) - vec2f(1.0, 0.0);
  let d1 = nn * cpw(z1, n - 1);
  let d2 = nn * cpw(z2, n - 1);
  let d3 = cm(vec2f(0.0, 1.0), cs.x * d1 + cs.y * d2);
  let fl = length(f);
  let fh = f / max(fl, 1e-20);
  let g = vec3f(dot(fh, d1), dot(fh, d2), dot(fh, d3));
  // the rim: |z1|^n + |z2|^n = K, over its slope
  let a1 = nn * pow(max(dot(z1, z1), 1e-12), 0.5 * nn - 1.0);
  let a2 = nn * pow(max(dot(z2, z2), 1e-12), 0.5 * nn - 1.0);
  let gh = vec3f(a1 * z1.x, a2 * z2.x, a1 * z1.y * cs.x + a2 * z2.y * cs.y);
  cy_rim2 = (cy_h(z1, z2, nn) - k.K) / max(length(gh), 1e-6);
  return fl / max(length(g), 1e-6);
}
const CY_N: i32 = 28;
fn raw(p: vec3f) -> f32 {
  let k = cy();
  let abc = vec3f(p.x, p.z, p.y) / k.sc;
  let T = 2.2 * pow(k.K, 1.0 / f32(k.n));
  // The line may pass near several sheets, and the one whose |f| is least need not be the nearest: every
  // low point of the search (up to three) is settled, and the nearest kept.
  var c: array<f32, 29>;
  for (var i = 0; i <= CY_N; i++) { c[i] = cy_cost(k, abc, mix(-T, T, f32(i) / f32(CY_N))); }
  var d = 0.3;                                   // and never more than this: far off, the estimate is no good
  var rim = 1.0;
  var found = 0;
  for (var i = 0; i <= CY_N; i++) {
    let low = (i == 0 || c[i] <= c[i - 1]) && (i == CY_N || c[i] <= c[i + 1]);
    if (!low || found >= 3) { continue; }
    found++;
    let di = cy_settle(k, abc, mix(-T, T, f32(i) / f32(CY_N)), c[i], T / f32(CY_N));
    if (di < d || found == 1) { rim = cy_rim2; }
    d = min(d, di);
  }
  cy_at = p;
  cy_rim = rim * k.sc;
  return d * k.sc;
}
fn cyRim(p: vec3f) -> f32 {
  if (any(p != cy_at)) { _ = raw(p); }
  return cy_rim;
}";

// The Costa surface (1982), from Gray's formulas in the Weierstrass functions of the square lattice of
// periods 1 and i, with e₁ = ℘(½):
//   x = ½ Re{ −ζ(w) + πu + π²⁄4e₁ + π⁄2e₁ [ζ(w − ½) − ζ(w − i⁄2)] }
//   y = ½ Re{ −iζ(w) + πv + π²⁄4e₁ − iπ⁄2e₁ [ζ(w − ½) − ζ(w − i⁄2)] }
//   z = ¼√(2π) ln |(℘(w) − e₁) ⁄ (℘(w) + e₁)|,   w = u + iv.
// ζ and ℘ come from Jacobi's theta functions (nome e^−π: three terms are plenty), all at the one point πw,
// the shifted ones by multiplying e^(iπw). The surface is symmetric in x and in y and under a quarter turn
// with z reversed, so a point is folded into x ≥ y ≥ 0; that eighth of the surface (out to 2.7 from the
// middle) comes from u in ½..0.95, v in ½..1, where a 5 × 5 grid finds the start of Gauss–Newton steps to
// the nearest point.
const COSTA: &str = "
// raw: a distance (to the nearest point found)
const CO_PI: f32 = 3.14159265;
const CO_Q1: f32 = 0.45593813;           // q^(1/4), q^(9/4), q^(25/4), q = e^-pi
const CO_Q9: f32 = 0.00085143834;
const CO_Q25: f32 = 2.9692570e-9;
const CO_E1: f32 = 6.8751858;
const CO_T34: f32 = 0.99254418;          // theta_3(0) theta_4(0)
const CO_C4: f32 = 0.35888501;           // pi^2 / 4 e1
const CO_CE: f32 = 0.22847329;           // pi / 2 e1
const CO_ZC: f32 = 0.62665707;           // sqrt(2 pi) / 4
const CO_EH: f32 = 4.8104774;            // e^(pi/2)
const CO_SC: f32 = 0.46;
fn co_mul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn co_div(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / max(dot(b, b), 1e-30); }
struct CoTh { t1: vec2f, d1: vec2f, t2: vec2f }
// theta_1, theta_1' and theta_2 at v, given A = e^(iv)
fn co_th(A: vec2f) -> CoTh {
  let B = vec2f(A.x, -A.y) / max(dot(A, A), 1e-30);
  let A2 = co_mul(A, A);
  let B2 = co_mul(B, B);
  let A3 = co_mul(A2, A);
  let B3 = co_mul(B2, B);
  let A5 = co_mul(A3, A2);
  let B5 = co_mul(B3, B2);
  let si = (A - B) * CO_Q1 - (A3 - B3) * CO_Q9 + (A5 - B5) * CO_Q25;    // 2i sin(kv), summed
  var r: CoTh;
  r.t1 = vec2f(si.y, -si.x);
  r.d1 = (A + B) * CO_Q1 - 3.0 * (A3 + B3) * CO_Q9 + 5.0 * (A5 + B5) * CO_Q25;
  r.t2 = (A + B) * CO_Q1 + (A3 + B3) * CO_Q9 + (A5 + B5) * CO_Q25;
  return r;
}
fn co_s(w: vec2f) -> vec3f {
  let A = exp(-CO_PI * w.y) * vec2f(cos(CO_PI * w.x), sin(CO_PI * w.x));
  let t = co_th(A);
  let ta = co_th(vec2f(A.y, -A.x));                // at pi (w - 1/2)
  let tb = co_th(CO_EH * A);                       // at pi (w - i/2)
  let zeta = CO_PI * w + CO_PI * co_div(t.d1, t.t1);
  let D = CO_PI * vec2f(-0.5, 0.5) + CO_PI * (co_div(ta.d1, ta.t1) - co_div(tb.d1, tb.t1));
  let k = CO_PI * CO_T34 * co_div(t.t2, t.t1);
  let P = vec2f(CO_E1, 0.0) + co_mul(k, k);
  return vec3f(0.5 * (-zeta.x + CO_PI * w.x + CO_C4 + CO_CE * D.x),
               0.5 * (zeta.y + CO_PI * w.y + CO_C4 + CO_CE * D.y),
               CO_ZC * log(length(P - vec2f(CO_E1, 0.0)) / max(length(P + vec2f(CO_E1, 0.0)), 1e-30)));
}
fn raw(p: vec3f) -> f32 {
  var q = vec3f(p.x, p.z, p.y) / CO_SC;
  q = vec3f(abs(q.x), abs(q.y), q.z);
  if (q.y > q.x) { q = vec3f(q.y, q.x, -q.z); }
  var w = vec2f(0.5);
  var best = 1e9;
  for (var i = 0; i < 5; i++) {
    for (var j = 0; j < 5; j++) {
      let c = vec2f(0.5) + vec2f(0.1125, 0.125) * vec2f(f32(i), f32(j));
      let d = length(co_s(c) - q);
      if (d < best) { best = d; w = c; }
    }
  }
  let h = 1e-3;
  for (var k = 0; k < 4; k++) {
    let s = co_s(w);
    best = min(best, length(q - s));
    let su = (co_s(w + vec2f(h, 0.0)) - s) / h;
    let sv = (co_s(w + vec2f(0.0, h)) - s) / h;
    let e = q - s;
    let a = dot(su, su);
    let b = dot(su, sv);
    let c = dot(sv, sv);
    let g = vec2f(dot(su, e), dot(sv, e));
    var dw = vec2f(c * g.x - b * g.y, a * g.y - b * g.x) / max(a * c - b * b, 1e-12);
    let l = length(dw);
    if (l > 0.05) { dw *= 0.05 / l; }
    w = clamp(w + dw, vec2f(0.4), vec2f(1.05, 1.1));
  }
  best = min(best, length(q - co_s(w)));
  return best * CO_SC;
}";

fn presets() -> Vec<Preset> {
    vec![
        topo(Topo {
            id: "kleinbottle", name: "Klein bottle",
            eq: &["a bulb turned about its axis, and a pipe", "from its neck back in through its side to its foot"],
            note: "A bottle with no inside: follow the glass from outside through the neck and you come back on the inside. In four dimensions the pipe would pass by the wall; in three it has to go through it.",
            params: vec![
                ParamDef::labelled("bulb", "bulb", 0.56, 0.45, 0.62, 0.005),
                ParamDef::labelled("neck", "neck", 0.14, 0.09, 0.17, 0.002),
                ParamDef::labelled("enter", "way in", -0.15, -0.6, 0.5, 0.01),
                wall(0.03),
            ],
            rot: [8.0, -30.0, 0.0],
            animate: "bulb",
            wgsl: KLEIN.to_string(),
            clip: "kbHole(p)".to_string(),
        }),
        topo(Topo {
            id: "boy", name: "Boy’s surface",
            eq: &["64(1−z)³z³ − 48(1−z)²z²(3x²+3y²+2z²) + 12(1−z)z(27(x²+y²)² − 24z²(x²+y²)", "+ 36√2·yz(y²−3x²) + 4z⁴) + (9x²+9y²−2z²)(−81(x²+y²)² − 72z²(x²+y²) + 108√2·xz(x²−3y²) + 4z⁴) = 0"],
            note: "The projective plane, which has one side, set in space without a fold or a pinch: Werner Boy found it in 1901 when Hilbert had asked him to prove it could not be done. This is Apéry’s equation for it.",
            params: vec![wall(0.03)],
            rot: [-20.0, 25.0, 0.0],
            animate: "wall",
            wgsl: BOY.to_string(),
            clip: "length(p) - 1.38".to_string(),
        }),
        topo(Topo {
            id: "roman", name: "Roman surface",
            eq: &["x²y² + y²z² + z²x² = xyz"],
            note: "Steiner’s Roman surface, found in Rome in 1844: the projective plane again, mapped into space with six pinch points where three lines of self-crossing end.",
            params: vec![wall(0.03)],
            rot: [-30.0, 40.0, 0.0],
            animate: "wall",
            wgsl: ROMAN.to_string(),
            clip: "sdBox(p, vec3f(1.064), 0.0)".to_string(),
        }),
        topo(Topo {
            id: "scherk", name: "Scherk’s tower",
            eq: &["sin y = sinh x · sinh z"],
            note: "Scherk’s second surface, a minimal surface from 1835: a stack of saddles, each turned a quarter turn on the one below.",
            params: vec![ParamDef::labelled("cells", "storeys", 1.6, 0.8, 2.0, 0.01), wall(0.03)],
            rot: [15.0, 45.0, 0.0],
            animate: "cells",
            wgsl: SCHERK.to_string(),
            clip: cylinder("0.62", "1.15"),
        }),
        topo(Topo {
            id: "helicoid", name: "Helicoid",
            eq: &["x sin(ky) = z cos(ky)"],
            note: "The surface a straight line sweeps as it climbs an axis turning evenly: a spiral staircase, and after the plane the oldest known minimal surface.",
            params: vec![ParamDef::labelled("turns", "half-turns", 1.5, 0.25, 4.0, 0.01), ParamDef::labelled("radius", "radius", 1.0, 0.5, 1.15, 0.005), wall(0.035)],
            rot: [12.0, 20.0, 0.0],
            animate: "turns",
            wgsl: HELICOID.to_string(),
            clip: cylinder("pr.radius", "1.05"),
        }),
        topo(Topo {
            id: "catenoid", name: "Catenoid",
            eq: &["√(x² + z²) = a cosh(y ⁄ a)"],
            note: "A hanging chain turned about an axis: the shape a soap film takes between two rings. Euler found it in 1744, the first minimal surface after the plane.",
            params: vec![ParamDef::labelled("neck", "neck a", 0.42, 0.25, 0.7, 0.005), wall(0.035)],
            rot: [-24.0, 0.0, 0.0],
            animate: "neck",
            wgsl: CATENOID.to_string(),
            clip: "abs(p.y) - pr.neck * acosh(1.15 / pr.neck)".to_string(),
        }),
        topo(Topo {
            id: "costa", name: "Costa surface",
            eq: &["x + iy, z from Weierstrass’s ζ and ℘ on the square lattice:", "z = ¼√(2π) ln |(℘(w) − e₁) ⁄ (℘(w) + e₁)|"],
            note: "Found by Celso Costa in 1982: a plane and a catenoid’s two ends, joined by two handles. It was the first new complete embedded minimal surface of finite topology since the eighteenth century.",
            params: vec![ParamDef::labelled("reach", "reach", 1.25, 0.7, 1.32, 0.005), wall(0.03)],
            rot: [22.0, 25.0, 0.0],
            animate: "reach",
            wgsl: COSTA.to_string(),
            clip: "length(p) - pr.reach".to_string(),
        }),
        topo(Topo {
            id: "enneper", name: "Enneper surface",
            eq: &["x + iy = w − w̄³ ⁄ 3,   z = Re w²", "|w| ≤ R"],
            note: "A minimal surface found by Alfred Enneper in 1864: a saddle whose rim, as it grows, folds over and through itself.",
            params: vec![ParamDef::labelled("reach", "reach R", 1.25, 0.6, 1.7, 0.005), wall(0.035)],
            rot: [30.0, 20.0, 0.0],
            animate: "reach",
            wgsl: ENNEPER.to_string(),
            clip: "-1.0".to_string(),
        }),
        topo(Topo {
            id: "dini", name: "Dini’s surface",
            eq: &["x + iy = a sin v · eⁱᵘ", "z = a (cos v + ln tan(v⁄2)) + b·u"],
            note: "A pseudosphere twisted into a screw. Like the pseudosphere it is curved as hard everywhere, and the same way across as along: its curvature is −1 at every point.",
            params: vec![
                ParamDef::labelled("climb", "climb b", 0.16, 0.06, 0.35, 0.002),
                ParamDef::labelled("turns", "turns", 2.0, 0.5, 3.0, 0.01),
                ParamDef::labelled("flare", "flare", 2.0, 1.2, 2.6, 0.01),
                wall(0.03),
            ],
            rot: [14.0, 0.0, 0.0],
            animate: "climb",
            wgsl: DINI.to_string(),
            clip: "-1.0".to_string(),
        }),
        topo(Topo {
            id: "kuen", name: "Kuen surface",
            eq: &["x + iy = 2 sin v (1 − iu) eⁱᵘ ⁄ (1 + u² sin² v)", "z = ln tan(v⁄2) + 2 cos v ⁄ (1 + u² sin² v)"],
            note: "Theodor Kuen’s surface of 1884, curvature −1 everywhere: two petals curling round a pair of spikes. Its plaster model stood in many a mathematics department.",
            params: vec![
                ParamDef::labelled("spread", "petals", 4.0, 2.0, 5.5, 0.01),
                wall(0.03),
            ],
            rot: [40.0, 120.0, 0.0],
            animate: "spread",
            wgsl: KUEN.to_string(),
            clip: "-1.0".to_string(),
        }),
        topo(Topo {
            id: "breather", name: "Breather",
            eq: &["the sine-Gordon breather, w = √(1 − a²) = m ⁄ (m + 1):", "a surface of curvature −1 that closes after m ripples"],
            note: "Solitons of the sine-Gordon equation draw surfaces of constant negative curvature. A breather is two of them bound together, pulsing; its surface is a spindle of ripples.",
            params: vec![
                ParamDef::labelled("ripples", "ripples m", 9.0, 3.0, 14.0, 1.0),
                ParamDef::labelled("reach", "length", 8.0, 5.0, 12.0, 0.05),
                wall(0.03),
            ],
            rot: [10.0, 30.0, 0.0],
            animate: "reach",
            wgsl: BREATHER.to_string(),
            clip: "-1.0".to_string(),
        }),
        topo(Topo {
            id: "calabiyau", name: "Calabi–Yau slice",
            eq: &["z₁ⁿ + z₂ⁿ = 1 in C², seen as", "(Re z₁, Re z₂, cos α Im z₁ + sin α Im z₂)"],
            note: "String theory curls six of its dimensions up in a Calabi–Yau manifold. This is a slice of the kind Andrew Hanson drew: a complex curve, four-dimensional, turned by α as it is seen in three.",
            params: vec![
                ParamDef::labelled("degree", "n", 4.0, 2.0, 6.0, 1.0),
                ParamDef::labelled("angle", "view α °", 45.0, 0.0, 180.0, 0.5),
                ParamDef::labelled("extent", "reach ξ", 0.9, 0.4, 1.5, 0.01),
                wall(0.03),
            ],
            rot: [55.0, 25.0, 0.0],
            animate: "angle",
            wgsl: CALABI.to_string(),
            clip: "cyRim(p)".to_string(),
        }),
    ]
}

fn light(az: f64, el: f64, w: f64, h: f64, rgb: [f64; 3], soft: f64) -> Light {
    Light { az, el, w, h, rgb, soft }
}

pub fn family() -> Family {
    Family {
        // walls cross each other and themselves: rays that leave the glass are followed into the next piece
        step_scale: 0.85,
        steps_out: 220,
        steps_in: 64,
        layers: 2,
        presets: presets(),
        defaults: vec![
            ("ior", Setting::Num(1.5)), ("disp", Setting::Num(0.07)), ("absorb", Setting::Vec3([0.30, 0.14, 0.06])),
            ("ns", Setting::Num(1.0)), ("nw", Setting::Num(3.0)), ("nb", Setting::Num(3.0)),
            ("dist", Setting::Num(6.6)), ("fov", Setting::Num(26.0)), ("rot", Setting::Vec3([22.0, 32.0, 0.0])),
            ("spin", Setting::Num(0.25)), ("bloom", Setting::Num(0.35)), ("exposure", Setting::Num(1.0)),
        ],
        // the minimal surfaces' lights: broad, soft faces of glass want broad reflections
        lights: vec![
            light(38.0, 18.0, 14.0, 32.0, [5.0, 4.9, 4.7], 0.7),
            light(-55.0, -8.0, 10.0, 42.0, [0.5, 1.4, 4.0], 0.7),
            light(-140.0, 5.0, 8.0, 42.0, [5.0, 1.6, 0.25], 0.7),
            light(150.0, 28.0, 24.0, 16.0, [3.2, 3.1, 3.0], 0.8),
            light(0.0, 85.0, 34.0, 34.0, [0.8, 0.9, 1.1], 0.9),
            light(0.0, -80.0, 30.0, 30.0, [0.5, 0.22, 0.05], 0.9),
        ],
        ..Family::new("topology", "Topology", Kind::Field)
    }
}
