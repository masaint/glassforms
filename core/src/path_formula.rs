//! A path as exact geometry: the formula its shapes are traced and baked from.
//!
//! Every piece of a path is one primitive, swept exactly:
//!   a segment   a straight piece: the profile carried along a line (a bank turns it about the line at an even
//!               rate along it: a twist)
//!   an arc      a turn, a slope, a loop: the profile carried round an axis, which is the profile revolved
//!               about that axis and cut to the arc's angle; a turn made while climbing, and a loop that steps
//!               aside, also rise along the axis as they go round (a helix)
//! A free curve is a chain of arcs, each leaving the way the one before arrived, each through the next of
//! sixteen points of the curve (an arc spline): exact pieces, smooth where they meet, within a hair of the
//! curve they follow; the path's walk follows the same arcs, so that what is drawn and what is built agree.
//!
//! The distance to a primitive is the distance in its section (the profile: a rounded rectangle, or an
//! outline filled even-odd) from the point's place in the section through it. That is exact for a segment
//! and for a flat arc. Where the profile changes along a primitive (a twist, a taper from one size to another,
//! the climb of a helix), the section's distance is divided by how fast the section's distance can change
//! as the point moves (a Lipschitz bound): never more than the distance, exact again where nothing changes.
//!
//! Neighbours meet in the plane square to the path where one ends and the next begins, which both share (the
//! path is smooth where pieces meet). That plane only shares the space out between them: inside the solid it
//! is no surface. An open end is a face: cut flat, or rounded over exactly (the section's core, rounded by
//! the profile's rounding: a half ball on a round tube).

use crate::path::{add, cross, dot, len3, norm, scale, sub, Axes, Frame, Path, Piece, Profile, Sample};

/// How many arcs a free curve is made of.
pub const CURVE_ARCS: usize = 16;
/// Steps the path's line is drawn in (the walk: for the picture and the panel, not for the shape).
const DRAW_DEG: f64 = 2.0;
const DRAW_LEN: f64 = 0.01;

/// The geometry of a primitive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Geo {
    /// from `a` along unit `t` for `len`, its section turned about `t` by `twist` radians on the way
    Seg { a: [f64; 3], t: [f64; 3], len: f64, twist: f64 },
    /// round the axis through `c` along unit `n`, at `rho` from it, from the direction `e0` (unit, square to
    /// `n`) through `theta` radians (turning as `n` turns: the way is `n × e0`), rising `k` along `n` a radian
    Arc { c: [f64; 3], n: [f64; 3], e0: [f64; 3], rho: f64, theta: f64, k: f64 },
}

/// A primitive: its geometry, the profile's across direction where it starts (a segment's: a direction square
/// to it; an arc's: (along e0, along n), which stays so all round), where along the path it starts and ends,
/// and what its two ends are (0: a joint, 1: an open end cut flat, 2: an open end rounded over).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prim {
    pub geo: Geo,
    pub side: [f64; 3],
    pub s0: f64,
    pub s1: f64,
    pub ends: [u8; 2],
}

impl Prim {
    pub fn length(&self) -> f64 {
        match self.geo {
            Geo::Seg { len, .. } => len,
            Geo::Arc { rho, theta, k, .. } => theta * rho.hypot(k),
        }
    }
    /// The point at a share `u` along it, and the frame there (along, top, left of the profile's own across).
    pub fn at(&self, u: f64) -> ([f64; 3], [f64; 3], [f64; 3]) {
        match self.geo {
            Geo::Seg { a, t, len, twist } => {
                let up0 = cross(t, self.side);
                let (s, c) = (twist * u).sin_cos();
                let side = add(scale(self.side, c), scale(up0, s));
                (add(a, scale(t, len * u)), t, side)
            }
            Geo::Arc { c, n, e0, rho, theta, k } => {
                let b0 = cross(n, e0);
                let (s, co) = (theta * u).sin_cos();
                let e = add(scale(e0, co), scale(b0, s));
                let p = add(add(c, scale(e, rho)), scale(n, k * theta * u));
                let t = norm(add(scale(cross(n, e), rho), scale(n, k)));
                let side = norm(add(scale(e, self.side[0]), scale(n, self.side[1])));
                (p, t, side)
            }
        }
    }
}

/// A path's primitives, and its joints (the frame where every piece starts, and the end), its length and
/// whether the loop closes.
#[derive(Clone, Debug, Default)]
pub struct Built {
    pub prims: Vec<Prim>,
    /// which piece each primitive belongs to
    pub piece_of: Vec<usize>,
    pub joints: Vec<Frame>,
    pub length: f64,
    pub closed: bool,
}

/// The arcs (or straight runs) of a free curve from frame f: (start, unit tangent, axis or None, radius,
/// angle), each leaving along the tangent the one before arrived with, each ending at the next of
/// `CURVE_ARCS` points of the cubic.
pub fn curve_arcs(f: &Frame, out: f64, h2: [f64; 3], end: [f64; 3]) -> Vec<([f64; 3], [f64; 3], Option<[f64; 3]>, f64, f64)> {
    let p0 = f.p;
    let a0 = f.axes();
    let p1 = add(p0, scale(a0.t, out.max(0.0)));
    let p2 = f.to_world(h2);
    let p3 = f.to_world(end);
    let at = |u: f64| -> [f64; 3] {
        let v = 1.0 - u;
        add(add(scale(p0, v * v * v), scale(p1, 3.0 * v * v * u)), add(scale(p2, 3.0 * v * u * u), scale(p3, u * u * u)))
    };
    let (mut p, mut t) = (p0, a0.t);
    let mut out_arcs = Vec::with_capacity(CURVE_ARCS);
    for i in 1..=CURVE_ARCS {
        let q = at(i as f64 / CURVE_ARCS as f64);
        let d = sub(q, p);
        let dl = len3(d);
        if dl < 1e-9 { continue; }
        let tx = cross(t, d);
        if len3(tx) < 1e-9 * dl {
            // (straight on: a segment; if it doubles back, it still ends at the point)
            out_arcs.push((p, norm(d), None, dl, 0.0));
            t = norm(d);
        } else {
            let n = norm(tx);
            let e0 = cross(t, n);
            let rho = -dot(d, d) / (2.0 * dot(d, e0));
            let c = sub(p, scale(e0, rho));
            let eq = scale(sub(q, c), 1.0 / rho);
            let mut theta = dot(eq, t).atan2(dot(eq, e0));
            if theta < 0.0 { theta += std::f64::consts::TAU; }
            out_arcs.push((p, t, Some(n), rho, theta));
            t = norm(cross(n, eq));
        }
        p = q;
    }
    out_arcs
}

/// Turn v about unit axis n by angle a.
fn turn_about(v: [f64; 3], n: [f64; 3], a: f64) -> [f64; 3] {
    let (s, c) = a.sin_cos();
    add(add(scale(v, c), scale(cross(n, v), s)), scale(n, dot(n, v) * (1.0 - c)))
}

/// The across direction of the profile in a frame: the track's left turned towards its top by `turn` degrees.
fn across(a: &Axes, turn: f64) -> [f64; 3] {
    let (s, c) = turn.to_radians().sin_cos();
    add(scale(a.left, c), scale(a.up, s))
}

/// An arc's start for a planar tangent `t` (unit), about axis `n`, at `rho`: its centre and its `e0`.
fn arc_from(p: [f64; 3], t: [f64; 3], n: [f64; 3], rho: f64) -> ([f64; 3], [f64; 3]) {
    let e0 = norm(cross(t, n));
    (sub(p, scale(e0, rho)), e0)
}

/// The across direction `side` (a direction of the section) as an arc keeps it: along e0 and along n.
fn arc_side(side: [f64; 3], e0: [f64; 3], n: [f64; 3]) -> [f64; 3] {
    let (x, y) = (dot(side, e0), dot(side, n));
    let l = x.hypot(y);
    if l < 1e-12 { [1.0, 0.0, 0.0] } else { [x / l, y / l, 0.0] }
}

/// The path's exact primitives, with the profile's turn (`turn` degrees) and what its open ends are.
pub fn build(path: &Path, turn: f64, round_ends: bool) -> Built {
    let mut b = Built::default();
    let mut f = path.start;
    let mut s = 0.0;
    let up_w = [0.0, 1.0, 0.0];
    let push = |b: &mut Built, geo: Geo, side: [f64; 3], s: &mut f64, piece: usize| {
        let p = Prim { geo, side, s0: *s, s1: *s, ends: [0, 0] };
        let l = p.length();
        if !(l > 1e-9) { return; }
        *s += l;
        b.prims.push(Prim { s1: *s, ..p });
        b.piece_of.push(piece);
    };
    for (i, piece) in path.pieces.iter().enumerate() {
        b.joints.push(f);
        let a = f.axes();
        let side = across(&a, turn);
        match *piece {
            Piece::Straight { len } => {
                let len = len.max(0.0);
                push(&mut b, Geo::Seg { a: f.p, t: a.t, len, twist: 0.0 }, side, &mut s, i);
                f.p = add(f.p, scale(a.t, len));
            }
            Piece::Bank { deg, len } => {
                let len = len.max(0.0);
                // (a roll that lifts the top to the left is a turn the other way about the way it heads)
                push(&mut b, Geo::Seg { a: f.p, t: a.t, len, twist: -deg.to_radians() }, side, &mut s, i);
                f.p = add(f.p, scale(a.t, len));
                f.roll += deg;
            }
            Piece::Turn { deg, radius } => {
                let r = radius.abs();
                let th = deg.abs().to_radians();
                let sign = if deg >= 0.0 { 1.0 } else { -1.0 };
                let n = scale(up_w, sign);
                let (sp, cp) = f.pitch.to_radians().sin_cos();
                let rho = r * cp.abs();
                let k = r * sp * sign;
                if rho < 1e-9 {
                    // heading straight up or down: the turn spins the track in place as it climbs
                    push(&mut b, Geo::Seg { a: f.p, t: a.t, len: r * th, twist: sign * th * a.t[1].signum() }, side, &mut s, i);
                    f.p = add(f.p, scale(a.t, r * th));
                } else {
                    let tp = norm([a.t[0], 0.0, a.t[2]]);
                    let (c, e0) = arc_from(f.p, tp, n, rho);
                    push(&mut b, Geo::Arc { c, n, e0, rho, theta: th, k }, arc_side(side, e0, n), &mut s, i);
                    let e1 = turn_about(e0, n, th);
                    f.p = add(add(c, scale(e1, rho)), scale(n, k * th));
                }
                f.yaw += deg;
            }
            Piece::Slope { deg, radius } => {
                let r = radius.abs();
                let th = deg.abs().to_radians();
                let left_h = crate::path::axes(f.yaw, 0.0, 0.0).left;
                let n = scale(left_h, if deg >= 0.0 { -1.0 } else { 1.0 });
                let (c, e0) = arc_from(f.p, a.t, n, r);
                push(&mut b, Geo::Arc { c, n, e0, rho: r, theta: th, k: 0.0 }, arc_side(side, e0, n), &mut s, i);
                f.p = add(c, scale(turn_about(e0, n, th), r));
                f.pitch += deg;
            }
            Piece::Loop { radius, shift } => {
                let r = radius.abs();
                let left_h = crate::path::axes(f.yaw, 0.0, 0.0).left;
                let n = scale(left_h, -1.0);
                let (c, e0) = arc_from(f.p, a.t, n, r);
                let tau = std::f64::consts::TAU;
                // (stepping aside along the left: along -n)
                push(&mut b, Geo::Arc { c, n, e0, rho: r, theta: tau, k: -shift / tau }, arc_side(side, e0, n), &mut s, i);
                f.p = add(f.p, scale(left_h, shift));
                f.pitch += 360.0;
            }
            Piece::Curve { out, h2, end } => {
                let mut up = a.up;
                let mut sd = side;
                let mut tl = a.t;
                for (p, t, axis, rho, theta) in curve_arcs(&f, out, h2, end) {
                    match axis {
                        None => {
                            push(&mut b, Geo::Seg { a: p, t, len: rho, twist: 0.0 }, sd, &mut s, i);
                            // (a straight run that turns from the tangent before: the frame turned with it)
                            let ax = cross(tl, t);
                            if len3(ax) > 1e-12 {
                                let ang = dot(tl, t).clamp(-1.0, 1.0).acos();
                                let nn = norm(ax);
                                up = turn_about(up, nn, ang);
                                sd = turn_about(sd, nn, ang);
                            }
                            tl = t;
                        }
                        Some(n) => {
                            let (c, e0) = arc_from(p, t, n, rho);
                            push(&mut b, Geo::Arc { c, n, e0, rho, theta, k: 0.0 }, arc_side(sd, e0, n), &mut s, i);
                            up = turn_about(up, n, theta);
                            sd = turn_about(sd, n, theta);
                            tl = turn_about(t, n, theta);
                        }
                    }
                }
                let p3 = f.to_world(end);
                let up = norm(sub(up, scale(tl, dot(up, tl))));
                f = Frame::from_axes(p3, &Axes { t: tl, up, left: cross(up, tl) });
            }
        }
    }
    b.joints.push(f);
    b.length = s;
    // the loop closes: the end lands on the start, heading the same way
    if path.pieces.len() > 1 && !b.prims.is_empty() {
        let e = b.joints[b.joints.len() - 1];
        b.closed = len3(sub(e.p, path.start.p)) < crate::path::CLOSE_GAP && dot(e.axes().t, path.start.axes().t) > 0.995;
    }
    if !b.closed && !b.prims.is_empty() {
        let open = if round_ends { 2 } else { 1 };
        b.prims[0].ends[0] = open;
        let last = b.prims.len() - 1;
        b.prims[last].ends[1] = open;
    }
    b
}

/// The path walked for drawing: points along the primitives (every couple of degrees of an arc), each with
/// its frame, and the joints. What the picture and the panel use; the shape is the primitives themselves.
pub fn walk(path: &Path) -> crate::path::Walk {
    let b = build(path, 0.0, false);
    let mut w = crate::path::Walk { samples: Vec::new(), joints: b.joints.clone(), length: b.length, closed: b.closed };
    let start = path.start.axes();
    w.samples.push(Sample { p: path.start.p, a: start, s: 0.0, piece: 0 });
    for (j, p) in b.prims.iter().enumerate() {
        let n = match p.geo {
            Geo::Seg { twist, .. } => ((twist.abs().to_degrees() / DRAW_DEG).ceil() as usize).max(1),
            Geo::Arc { theta, .. } => ((theta.to_degrees() / DRAW_DEG).max(p.length() / DRAW_LEN).ceil() as usize).max(1),
        };
        for k in 1..=n {
            let u = k as f64 / n as f64;
            let (q, t, side) = p.at(u);
            let up = cross(t, side);
            w.samples.push(Sample { p: q, a: Axes { t, up, left: side }, s: p.s0 + u * (p.s1 - p.s0), piece: b.piece_of[j] });
        }
    }
    if w.closed {
        if let Some(l) = w.samples.last_mut() { l.p = path.start.p; }
    }
    w
}

// ---- the distance ---------------------------------------------------------------------------------------

/// The profile's sizes at a share along the path: half across, half through, the rounding.
fn sizes(pr: &Profile, s: f64, total: f64) -> (f64, f64, f64) {
    let k = pr.start + (pr.end - pr.start) * (s / total.max(1e-12)).clamp(0.0, 1.0);
    let (hw, hh) = (0.5 * pr.w.max(0.0) * k, 0.5 * pr.h.max(0.0) * k);
    (hw, hh, pr.round.clamp(0.0, 1.0) * hw.min(hh))
}

/// The distance in the section, from (x, y): the rounded rectangle, or the outline.
pub fn section(x: f64, y: f64, hw: f64, hh: f64, r: f64, outline: Option<&[[f64; 4]]>) -> f64 {
    match outline {
        None => {
            let (qx, qy) = (x.abs() - (hw - r), y.abs() - (hh - r));
            qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - r
        }
        Some(edges) => {
            let (px, py) = (x / hw.max(1e-9), y / hh.max(1e-9));
            let mut d2 = f64::MAX;
            let mut inside = false;
            for e in edges {
                let (ax, ay, bx, by) = (e[0], e[1], e[2] - e[0], e[3] - e[1]);
                let h = (((px - ax) * bx + (py - ay) * by) / (bx * bx + by * by).max(1e-24)).clamp(0.0, 1.0);
                let (wx, wy) = (px - ax - bx * h, py - ay - by * h);
                d2 = d2.min(wx * wx + wy * wy);
                if (ay > py) != (e[3] > py) && px < bx * (py - ay) / by + ax { inside = !inside; }
            }
            let d = d2.sqrt() * hw.min(hh);
            if inside { -d } else { d }
        }
    }
}

/// The distance from p to a primitive of the path swept with this profile (`total`: the path's length, for its
/// taper). Exact where the profile is the same all along it and the primitive does not climb or twist;
/// otherwise never more than the distance.
pub fn dist(p: [f64; 3], prim: &Prim, pr: &Profile, total: f64) -> f64 {
    let outline = pr.outline.as_deref().map(|v| v.as_slice());
    let (hw0, hh0, _) = sizes(pr, prim.s0, total);
    let (hw1, hh1, _) = sizes(pr, prim.s1, total);
    let ext = hw0.hypot(hh0).max(hw1.hypot(hh1));
    let len = prim.length().max(1e-12);
    // how fast the section's sizes change along it
    let taper = (hw1 - hw0).abs().max((hh1 - hh0).abs()) / len;
    // (x, y in the section, how far beyond the cut at each end, the share along it)
    let in_section = |u: f64, x: f64, y: f64| -> f64 {
        let (hw, hh, r) = sizes(pr, prim.s0 + u * (prim.s1 - prim.s0), total);
        section(x, y, hw, hh, r, outline)
    };
    let (flat, c0, c1, lip, r_end) = match prim.geo {
        Geo::Seg { a, t, len, twist } => {
            let v = sub(p, a);
            let u = dot(v, t);
            let h = (u / len.max(1e-12)).clamp(0.0, 1.0);
            let up0 = cross(t, prim.side);
            let (sn, cs) = (twist * h).sin_cos();
            let side = add(scale(prim.side, cs), scale(up0, sn));
            let up = cross(t, side);
            let w = sub(v, scale(t, u));
            let lip = (1.0 + (twist / len.max(1e-12) * ext).powi(2) + taper * taper).sqrt();
            (in_section(h, dot(w, side), dot(w, up)), -u, u - len, lip, h)
        }
        Geo::Arc { c, n, e0, rho, theta, k } => {
            let b0 = cross(n, e0);
            let v = sub(p, c);
            let z = dot(v, n);
            let (ex, ey) = (dot(v, e0), dot(v, b0));
            let mut phi = ey.atan2(ex);
            if phi < 0.0 { phi += std::f64::consts::TAU; }
            let (al, be) = (prim.side[0], prim.side[1]);
            // the section at angle a: where p is in it, and how far beyond it (along the way at a) p is
            let at = |a: f64| -> (f64, f64, f64) {
                let (s, co) = a.sin_cos();
                let e = add(scale(e0, co), scale(b0, s));
                let tt = cross(n, e);
                let dr = dot(v, e) - rho;
                let dz = z - k * a;
                (al * dr + be * dz, be * dr - al * dz, dot(v, tt))
            };
            // (the climb moves the section along the axis as p goes round it: k over p's own distance from the
            // axis a unit round; near the axis that is steep, and the bound says so)
            let rp = ex.hypot(ey);
            let lip = (1.0 + (k / rp.max(1e-6)).powi(2) + taper * taper).sqrt();
            if phi <= theta {
                let (x, y, _) = at(phi);
                // (how far inside the planes of its two ends: exact near them, never nearer further round)
                let half = std::f64::consts::FRAC_PI_2;
                let (c0, c1) = (-rp * phi.min(half).sin(), -rp * (theta - phi).min(half).sin());
                (in_section(phi / theta.max(1e-12), x, y), c0, c1, lip, phi / theta.max(1e-12))
            } else {
                // beyond the angle: the nearer of its two end sections
                let (x1, y1, f1) = at(theta);
                let (x0, y0, f0) = at(0.0);
                let d1 = in_section(1.0, x1, y1).max(0.0).hypot(f1.max(0.0));
                let d0 = in_section(0.0, x0, y0).max(0.0).hypot((-f0).max(0.0));
                // (outside its angle the point is outside it: how far from the end's plane, whichever side)
                if d1 <= d0 { (in_section(1.0, x1, y1), -1.0, f1.abs().max(1e-12), lip, 1.0) } else { (in_section(0.0, x0, y0), f0.abs().max(1e-12), -1.0, lip, 0.0) }
            }
        }
    };
    let flat = flat / lip;
    let cut = c0.max(c1);
    let end = if c0 >= c1 { prim.ends[0] } else { prim.ends[1] };
    if cut > 0.0 {
        if end == 2 && outline.is_none() {
            let (_, _, r) = sizes(pr, prim.s0 + r_end * (prim.s1 - prim.s0), total);
            return (flat + r).max(0.0).hypot(cut) - r;
        }
        return flat.max(0.0).hypot(cut);
    }
    let mut inner = flat;
    let flat_end = |e: u8| e == 1 || (e == 2 && outline.is_some());
    if flat_end(prim.ends[0]) { inner = inner.max(c0); }
    if flat_end(prim.ends[1]) { inner = inner.max(c1); }
    inner
}

/// The box round a primitive and its section.
pub fn bounds(prim: &Prim, pr: &Profile, total: f64) -> ([f64; 3], [f64; 3]) {
    let (hw0, hh0, r0) = sizes(pr, prim.s0, total);
    let (hw1, hh1, r1) = sizes(pr, prim.s1, total);
    let ext = hw0.hypot(hh0).max(hw1.hypot(hh1)) + r0.max(r1) + 1e-3;
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    let n = match prim.geo { Geo::Seg { .. } => 1, Geo::Arc { theta, .. } => ((theta / 0.2).ceil() as usize).max(2) };
    for i in 0..=n {
        let (q, _, _) = prim.at(i as f64 / n as f64);
        for k in 0..3 {
            lo[k] = lo[k].min(q[k] - ext);
            hi[k] = hi[k].max(q[k] + ext);
        }
    }
    // (between the points an arc bulges out by at most its sagitta)
    if let Geo::Arc { rho, theta, .. } = prim.geo {
        let sag = rho * (1.0 - (0.5 * theta / n as f64).cos());
        for k in 0..3 { lo[k] -= sag; hi[k] += sag; }
    }
    (lo, hi)
}

/// Boxes along a primitive, each round a stretch of it no longer than `most` (for the baker, which draws a
/// primitive into each: the distance within each is the whole primitive's, and the boxes stay close to it).
pub fn bounds_along(prim: &Prim, pr: &Profile, total: f64, most: f64) -> Vec<([f64; 3], [f64; 3])> {
    let n = ((prim.length() / most).ceil() as usize).clamp(1, 64);
    let (hw0, hh0, r0) = sizes(pr, prim.s0, total);
    let (hw1, hh1, r1) = sizes(pr, prim.s1, total);
    let ext = hw0.hypot(hh0).max(hw1.hypot(hh1)) + r0.max(r1) + 1e-3;
    (0..n).map(|i| {
        let (u0, u1) = (i as f64 / n as f64, (i + 1) as f64 / n as f64);
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        let m = match prim.geo { Geo::Seg { .. } => 1, Geo::Arc { theta, .. } => ((theta * (u1 - u0) / 0.2).ceil() as usize).max(1) };
        for j in 0..=m {
            let (q, _, _) = prim.at(u0 + (u1 - u0) * j as f64 / m as f64);
            for k in 0..3 { lo[k] = lo[k].min(q[k] - ext); hi[k] = hi[k].max(q[k] + ext); }
        }
        if let Geo::Arc { rho, theta, .. } = prim.geo {
            let sag = rho * (1.0 - (0.5 * theta * (u1 - u0) / m as f64).cos());
            for k in 0..3 { lo[k] -= sag; hi[k] += sag; }
        }
        (lo, hi)
    }).collect()
}

// ---- for the GPU -------------------------------------------------------------------------------------------

/// The numbers of one primitive, as the GPU reads it: six groups of four.
///   0  (a, or the arc's centre c), kind (0 segment, 1 arc)
///   1  (t, or n), length or angle
///   2  (the segment's across direction, or e0), twist or radius
///   3  (the arc's across: along e0, along n), the arc's rise a radian, ends (start + 4 end)
///   4  half across, half through, rounding (each at the start), the taper's bound (sizes' change a unit along)
///   5  half across, half through, rounding at the end, the bound on the section's slope (Lipschitz)
pub fn record(prim: &Prim, pr: &Profile, total: f64) -> [f64; 24] {
    let (hw0, hh0, r0) = sizes(pr, prim.s0, total);
    let (hw1, hh1, r1) = sizes(pr, prim.s1, total);
    let ext = hw0.hypot(hh0).max(hw1.hypot(hh1));
    let len = prim.length().max(1e-12);
    let taper = (hw1 - hw0).abs().max((hh1 - hh0).abs()) / len;
    let ends = prim.ends[0] as f64 + 4.0 * prim.ends[1] as f64;
    let mut o = [0.0; 24];
    let (kind, p, d, x, w1, w2, al, be, rise, lip) = match prim.geo {
        Geo::Seg { a, t, len, twist } => (0.0, a, t, prim.side, len, twist, 0.0, 0.0, 0.0, (1.0 + (twist / len.max(1e-12) * ext).powi(2) + taper * taper).sqrt()),
        // (an arc's climb is bounded where it is read: it depends on the point's own distance from the axis)
        Geo::Arc { c, n, e0, rho, theta, k } => (1.0, c, n, e0, theta, rho, prim.side[0], prim.side[1], k, (1.0 + taper * taper).sqrt()),
    };
    o[0..3].copy_from_slice(&p);
    o[3] = kind;
    o[4..7].copy_from_slice(&d);
    o[7] = w1;
    o[8..11].copy_from_slice(&x);
    o[11] = w2;
    o[12] = al;
    o[13] = be;
    o[14] = rise;
    o[15] = ends;
    o[16] = hw0;
    o[17] = hh0;
    o[18] = r0;
    o[19] = taper;
    o[20] = hw1;
    o[21] = hh1;
    o[22] = r1;
    o[23] = lip;
    o
}

/// The WGSL of a primitive's distance: `pathPrim(p, r0..r5, r6, eb, ne, best)`, the record's six groups (and
/// r6: an arc's end as (sin, cos, -, 1), or nothing, when it is worked out), the best distance found so far, its outline's
/// edges (`ne` of them, the first at `eb`) read by `edge` (the text that reads edge j: the baker reads them from
/// its points, a tree from its table).
pub fn wgsl(edge: &str) -> String {
    primitive_wgsl(edge, "")
}

fn primitive_wgsl(edge: &str, outline_lookup: &str) -> String {
    format!("
// ---- a path's primitive, exactly (gf_core::path_formula)
fn pathSection(x: f32, y: f32, hw: f32, hh: f32, r: f32, eb: u32, ne: u32, best: f32) -> f32 {{
  if (ne == 0u) {{
    let q = vec2f(abs(x) - (hw - r), abs(y) - (hh - r));
    return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r;
  }}
  // (the outline lies within its box: no nearer than the box, and when that is further than the best so far
  // its edges need not be read)
  let bx = max(abs(x) - hw, abs(y) - hh);
  if (bx > best) {{ return bx; }}
  let pp = vec2f(x / max(hw, 1e-6), y / max(hh, 1e-6));
  var d2 = 1e18;
  var inside = false;
  var start = eb;
  var count = ne;
  {outline_lookup}
  for (var j = 0u; j < count; j++) {{
    let s = {edge};
    let ab = s.zw - s.xy;
    let h = clamp(dot(pp - s.xy, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    let ww = pp - s.xy - ab * h;
    d2 = min(d2, dot(ww, ww));
    if ((s.y > pp.y) != (s.w > pp.y)) {{
      if (pp.x < ab.x * (pp.y - s.y) / ab.y + s.x) {{ inside = !inside; }}
    }}
  }}
  return select(1.0, -1.0, inside) * sqrt(d2) * min(hw, hh);
}}
fn pathPrim(p: vec3f, r0: vec4f, r1: vec4f, r2: vec4f, r3: vec4f, r4: vec4f, r5: vec4f, r6: vec4f, eb: u32, ne: u32, best: f32) -> f32 {{
  var x = 0.0;
  var y = 0.0;
  var c0 = -1.0;
  var c1 = -1.0;
  var u = 0.0;
  var lip = r5.w;
  let v = p - r0.xyz;
  if (r0.w < 0.5) {{
    // a segment: from a along t, its across turned about t as it goes (a twist)
    let t = r1.xyz;
    let len = max(r1.w, 1e-7);
    let along = dot(v, t);
    u = clamp(along / len, 0.0, 1.0);
    var side = r2.xyz;
    if (r2.w != 0.0) {{
      let a = r2.w * u;
      side = side * cos(a) + cross(t, r2.xyz) * sin(a);
    }}
    let w = v - t * along;
    x = dot(w, side);
    y = dot(w, cross(t, side));
    c0 = -along;
    c1 = along - r1.w;
  }} else {{
    // an arc: round n through c, at rho from it, from e0 through theta, rising r3.z a radian
    let n = r1.xyz;
    let e0 = r2.xyz;
    let b0 = cross(n, e0);
    let rho = r2.w;
    let theta = r1.w;
    let k = r3.z;
    let z = dot(v, n);
    let ex = dot(v, e0);
    let ey = dot(v, b0);
    let rp = length(vec2f(ex, ey));
    // (the end's direction: sin and cos of the angle given, r6.w = 1, or worked out)
    let sc = select(vec2f(sin(theta), cos(theta)), r6.xy, r6.w > 0.5);
    let e1 = e0 * sc.y + b0 * sc.x;
    let s1 = dot(v, cross(n, e1));
    let full = theta > 6.283;
    // The angle itself is needed only where the section climbs, tapers, or the arc goes all the way round;
    // otherwise which side of its two end planes p is on says whether it is within the arc's angle, and the
    // section through p is the one through the axis and p: no angle read at all.
    let tapers = any(r4.xyz != r5.xyz);
    var inside = false;
    var phi = 0.0;
    if (k != 0.0 || tapers || full) {{
      phi = atan2(ey, ex);
      if (phi < 0.0) {{ phi += 6.283185307; }}
      inside = phi <= theta;
      c0 = -rp * sin(min(phi, 1.5707963));
      c1 = -rp * sin(min(theta - phi, 1.5707963));
    }} else {{
      let past = ex * sc.x - ey * sc.y;
      inside = select(!(ey < 0.0 && past < 0.0), ey >= 0.0 && past >= 0.0, theta <= 3.14159265);
      phi = 0.0;
      // (inside: how far inside the planes of its two ends; exact near them, never nearer further round)
      c0 = select(-rp, -ey, ex >= 0.0);
      c1 = select(-rp, s1, dot(v, e1) >= 0.0);
    }}
    var a = phi;
    var dr = rp - rho;
    if (!inside) {{
      // beyond its angle: the nearer of its two end sections (judged by where p stands from each)
      let g1 = length(vec2f(dot(v, e1) - rho, z - k * theta)) + max(s1, 0.0);
      let g0 = length(vec2f(ex - rho, z)) + max(-ey, 0.0);
      let at_end = g1 <= g0;
      a = select(0.0, theta, at_end);
      dr = select(ex, dot(v, e1), at_end) - rho;
      // (outside its angle the point is outside it: how far from the end's plane, whichever side of it)
      c1 = select(-1.0, max(abs(s1), 1e-7), at_end);
      c0 = select(max(abs(ey), 1e-7), -1.0, at_end);
    }}
    let dz = z - k * a;
    x = r3.x * dr + r3.y * dz;
    y = r3.y * dr - r3.x * dz;
    u = a / max(theta, 1e-7);
    // (the climb: the section moves along the axis as p goes round it)
    if (k != 0.0) {{ lip = sqrt(lip * lip + k * k / max(rp * rp, 1e-12)); }}
  }}
  let hw = mix(r4.x, r5.x, u);
  let hh = mix(r4.y, r5.y, u);
  let rr = min(mix(r4.z, r5.z, u), min(hw, hh));
  let flat = pathSection(x, y, hw, hh, rr, eb, ne, best * lip) / lip;
  let cut = max(c0, c1);
  let ends = u32(r3.w + 0.5);
  let end = select(ends >> 2u, ends & 3u, c0 >= c1);
  if (cut > 0.0) {{
    if (end == 2u && ne == 0u) {{ return length(vec2f(max(flat + rr, 0.0), cut)) - rr; }}
    return length(vec2f(max(flat, 0.0), cut));
  }}
  var inner = flat;
  let e0k = ends & 3u;
  let e1k = ends >> 2u;
  if (e0k == 1u || (e0k == 2u && ne > 0u)) {{ inner = max(inner, c0); }}
  if (e1k == 1u || (e1k == 2u && ne > 0u)) {{ inner = max(inner, c1); }}
  return inner;
}}
")
}

// ---- the table a tree reads (gf_core::csg) --------------------------------------------------------------------

/// Cells a side of a path's own grid of lists.
pub const CELLS: usize = 12;
/// The most primitives a cell lists (a record's places less the count); a cell that would list more lists none,
/// and every primitive is read there.
pub const CELL_MOST: usize = 35;

// The outline lives in [-1,1]². Include some room just outside it, where a
// surface refinement or normal will also ask for its distance.
const OUTLINE_SIDE: usize = 12;
const OUTLINE_EXTENT: f64 = 1.125;
const OUTLINE_MIN: usize = 16;

/// Append one header per cell and its short list of original edges. For any
/// point in the cell, a nearest edge at its centre can get at most half a
/// diagonal farther away. An edge more than a full diagonal beyond that
/// closest centre distance therefore cannot win. Keep any other edge whose
/// ray crossing may vary across the cell, and store constant crossings in
/// the header's parity. Holes and disjoint outlines use the same even-odd rule.
fn indexed_outline(edges: &[[f64; 4]]) -> Vec<[f32; 4]> {
    let mut data: Vec<_> = edges.iter().map(|e| e.map(|x| x as f32)).collect();
    if edges.len() < OUTLINE_MIN { return data; }
    let cell = 2.0 * OUTLINE_EXTENT / OUTLINE_SIDE as f64;
    let n = edges.len();
    data.resize(n + OUTLINE_SIDE * OUTLINE_SIDE, [0.0; 4]);
    let mut distances = vec![0.0; n];
    for row in 0..OUTLINE_SIDE {
        for col in 0..OUTLINE_SIDE {
            let x = (col as f64 + 0.5) * cell - OUTLINE_EXTENT;
            let y = (row as f64 + 0.5) * cell - OUTLINE_EXTENT;
            let half = cell * 0.5 + 1e-6;
            let mut best = f64::INFINITY;
            for (d, &[ax, ay, bx, by]) in distances.iter_mut().zip(edges) {
                let (dx, dy) = (bx - ax, by - ay);
                let t = (((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy).max(1e-12)).clamp(0.0, 1.0);
                *d = (x - ax - t * dx).hypot(y - ay - t * dy);
                best = best.min(*d);
            }
            let start = data.len();
            let mut inside = false;
            for (&d, edge) in distances.iter().zip(edges) {
                let [ax, ay, bx, by] = *edge;
                let crosses = ay.min(by) <= y + half && ay.max(by) > y - half && ax.max(bx) >= x - half && ay != by;
                let always = ax.min(bx) > x + half && ay.min(by) <= y - half && ay.max(by) > y + half;
                if d <= best + cell * std::f64::consts::SQRT_2 + 1e-5 || (crosses && !always) {
                    data.push(edge.map(|v| v as f32));
                } else if always { inside = !inside; }
            }
            data[n + row * OUTLINE_SIDE + col] = [start as f32, (data.len() - start) as f32, u8::from(inside) as f32, 0.0];
        }
    }
    data
}

/// A path's block of the table: records of 36 numbers.
///   the head     (box low corner, cells a side), (a cell's size, how many primitives), (where the cells, the
///                primitives and the edges start, as records from the head; how many edges), -...
///   the cells    CELLS³ of them: how many primitives the cell lists (-1: all), then their numbers
///   the primitives, a record each (the six groups of `record`, then nothing)
///   the edges    nine to a record
/// A cell lists every primitive that may be nearer to some point of it than the nearest primitive can be far.
pub fn table(b: &Built, pr: &Profile) -> Vec<f32> {
    let n = b.prims.len();
    let boxes: Vec<([f64; 3], [f64; 3])> = b.prims.iter().map(|p| bounds(p, pr, b.length)).collect();
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for (l, h) in &boxes {
        for k in 0..3 { lo[k] = lo[k].min(l[k]); hi[k] = hi[k].max(h[k]); }
    }
    if n == 0 { lo = [0.0; 3]; hi = [0.0; 3]; }
    for k in 0..3 { lo[k] -= 0.05; hi[k] += 0.05; }
    let cell = [(hi[0] - lo[0]) / CELLS as f64, (hi[1] - lo[1]) / CELLS as f64, (hi[2] - lo[2]) / CELLS as f64];
    let edges: Vec<[f64; 4]> = pr.outline.as_deref().cloned().unwrap_or_default();
    let cells = CELLS * CELLS * CELLS;
    let outline_data = indexed_outline(&edges);
    let edge_records = outline_data.len().div_ceil(9);
    let mut t = vec![0.0f32; 36 * (1 + cells + n + edge_records)];
    t[0..3].copy_from_slice(&lo.map(|v| v as f32));
    t[3] = CELLS as f32;
    t[4..7].copy_from_slice(&cell.map(|v| v as f32));
    t[7] = n as f32;
    t[8] = 1.0;
    t[9] = (1 + cells) as f32;
    t[10] = (1 + cells + n) as f32;
    t[11] = edges.len() as f32;
    // the farthest any point of a box can be from the surface of primitive i: from its far corner to the
    // primitive's own points, and out to its section
    let gap = |clo: &[f64; 3], chi: &[f64; 3], l: &[f64; 3], h: &[f64; 3]| -> f64 {
        (0..3).map(|k| (l[k] - chi[k]).max(clo[k] - h[k]).max(0.0).powi(2)).sum::<f64>().sqrt()
    };
    // (from the cell's middle to the primitive's centre line, half the cell's diagonal more, and out to its
    // section: no point of the cell is further than that from the primitive's surface)
    let line = Profile { w: 0.0, h: 0.0, round: 0.0, outline: None, ..pr.clone() };
    let farthest = |clo: &[f64; 3], chi: &[f64; 3], i: usize| -> f64 {
        let mid = [0.5 * (clo[0] + chi[0]), 0.5 * (clo[1] + chi[1]), 0.5 * (clo[2] + chi[2])];
        let half = 0.5 * ((chi[0] - clo[0]).powi(2) + (chi[1] - clo[1]).powi(2) + (chi[2] - clo[2]).powi(2)).sqrt();
        let (hw0, hh0, _) = sizes(pr, b.prims[i].s0, b.length);
        let (hw1, hh1, _) = sizes(pr, b.prims[i].s1, b.length);
        dist(mid, &b.prims[i], &line, b.length).abs() + half + hw0.hypot(hh0).max(hw1.hypot(hh1))
    };
    let nearest = |clo: &[f64; 3], chi: &[f64; 3], i: usize| -> f64 {
        let mid = [0.5 * (clo[0] + chi[0]), 0.5 * (clo[1] + chi[1]), 0.5 * (clo[2] + chi[2])];
        let half = 0.5 * ((chi[0] - clo[0]).powi(2) + (chi[1] - clo[1]).powi(2) + (chi[2] - clo[2]).powi(2)).sqrt();
        let (hw0, hh0, r0) = sizes(pr, b.prims[i].s0, b.length);
        let (hw1, hh1, r1) = sizes(pr, b.prims[i].s1, b.length);
        // (a rounded open end reaches past the centre line's end by its rounding)
        dist(mid, &b.prims[i], &line, b.length).abs() - half - hw0.hypot(hh0).max(hw1.hypot(hh1)) - r0.max(r1)
    };
    for c in 0..cells {
        let (x, y, z) = (c % CELLS, (c / CELLS) % CELLS, c / (CELLS * CELLS));
        let clo = [lo[0] + x as f64 * cell[0], lo[1] + y as f64 * cell[1], lo[2] + z as f64 * cell[2]];
        let chi = [clo[0] + cell[0], clo[1] + cell[1], clo[2] + cell[2]];
        let reach = (0..n).map(|i| farthest(&clo, &chi, i)).fold(f64::MAX, f64::min);
        // (how near each may come to the cell: the larger of its box's gap and its centre line's distance less
        // half the cell's diagonal and its section's reach; nearest first)
        let mut listed: Vec<(f64, usize)> = (0..n).map(|i| (gap(&clo, &chi, &boxes[i].0, &boxes[i].1).max(nearest(&clo, &chi, i)), i)).filter(|&(g, _)| g <= reach).collect();
        listed.sort_by(|a, b| a.0.total_cmp(&b.0));
        let listed: Vec<usize> = listed.into_iter().map(|(_, i)| i).collect();
        let at = 36 * (1 + c);
        if listed.len() > CELL_MOST {
            t[at] = -1.0;
        } else {
            t[at] = listed.len() as f32;
            for (j, i) in listed.iter().enumerate() { t[at + 1 + j] = *i as f32; }
        }
    }
    for (i, p) in b.prims.iter().enumerate() {
        let at = 36 * (1 + cells + i);
        for (j, v) in record(p, pr, b.length).iter().enumerate() { t[at + j] = *v as f32; }
        // (its box: its middle in words 24..26, its half sizes in 28..30)
        let (l, h) = &boxes[i];
        for k in 0..3 {
            t[at + 24 + k] = (0.5 * (l[k] + h[k])) as f32;
            t[at + 28 + k] = (0.5 * (h[k] - l[k])) as f32;
        }
        // (an arc's end, as the sine and cosine of its angle: words 32..35)
        if let Geo::Arc { theta, .. } = p.geo {
            t[at + 32] = theta.sin() as f32;
            t[at + 33] = theta.cos() as f32;
            t[at + 35] = 1.0;
        }
    }
    for (j, e) in outline_data.iter().enumerate() {
        let at = 36 * (1 + cells + n) + 4 * j;
        t[at..at + 4].copy_from_slice(e);
    }
    t
}

/// The WGSL a tree reads a path part with: `pathDist(p, base)`, the distance to the path whose block starts at
/// record `base` of the table (`csgP`, records of the tree's part struct).
pub fn wgsl_tree() -> String {
    let lookup = format!("
  if (ne >= {OUTLINE_MIN}u && all(abs(pp) < vec2f({OUTLINE_EXTENT}))) {{
    let xy = min(vec2u((pp + vec2f({OUTLINE_EXTENT})) * {scale}), vec2u({last}u));
    let header = pathEdge(eb + ne + xy.x + {OUTLINE_SIDE}u * xy.y);
    start = eb + u32(header.x);
    count = u32(header.y);
    inside = header.z > 0.5;
  }}", scale = OUTLINE_SIDE as f64 / (2.0 * OUTLINE_EXTENT), last = OUTLINE_SIDE - 1);
    primitive_wgsl("pathEdge(start + j)", &lookup) + &format!("
// group g of record r of the table (a path's cells and edges are kept in the groups of the records)
fn pathGroup(r: u32, g: u32) -> vec4f {{
  switch g {{
    case 0u: {{ return csgP[r].inv0; }}
    case 1u: {{ return csgP[r].inv1; }}
    case 2u: {{ return csgP[r].inv2; }}
    case 3u: {{ return csgP[r].b; }}
    case 4u: {{ return csgP[r].m; }}
    case 5u: {{ return csgP[r].k; }}
    case 6u: {{ return csgP[r].v0; }}
    case 7u: {{ return csgP[r].v1; }}
    default: {{ return csgP[r].e; }}
  }}
}}
// word j of record r
fn pathWord(r: u32, j: u32) -> f32 {{
  return pathGroup(r, j >> 2u)[j & 3u];
}}
fn pathEdge(j: u32) -> vec4f {{
  return pathGroup(j / 9u, j % 9u);
}}
// the distance to the path whose block starts at record `base`: the primitives its cell lists (or, outside its
// box or in a cell that would list too many, every one)
fn pathDist(p: vec3f, base: u32) -> f32 {{
  let H = csgP[base];
  let lo = H.inv0.xyz;
  let n = u32(H.inv0.w + 0.5);
  let cs = H.inv1.xyz;
  let count = u32(H.inv1.w + 0.5);
  let cells = base + u32(H.inv2.x + 0.5);
  let prims = base + u32(H.inv2.y + 0.5);
  let edges = 9u * (base + u32(H.inv2.z + 0.5));
  let ne = u32(H.inv2.w + 0.5);
  let g = (p - lo) / cs;
  var listed = -1.0;
  var cell = 0u;
  if (all(g >= vec3f(0.0)) && all(g < vec3f(f32(n)))) {{
    let gi = vec3u(g);
    cell = cells + gi.x + n * (gi.y + n * gi.z);
    listed = pathWord(cell, 0u);
  }}
  let all_of = listed < 0.0;
  let m = select(u32(listed + 0.5), count, all_of);
  var d = 1e9;
  for (var j = 0u; j < m; j++) {{
    let i = select(u32(pathWord(cell, 1u + j) + 0.5), j, all_of);
    let P = csgP[prims + i];
    // (its box: no nearer than that)
    if (length(max(abs(p - P.v0.xyz) - P.v1.xyz, vec3f(0.0))) > d) {{ continue; }}
    d = min(d, pathPrim(p, P.inv0, P.inv1, P.inv2, P.b, P.m, P.k, P.e, edges, ne, d));
  }}
  return d;
}}
")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outline_cells_preserve_distance_and_parity_across_a_hole() {
        let mut edges = Vec::new();
        for (r, m) in [(1.0, 32), (0.27, 11)] {
            for i in 0..m {
                let point = |j: i32| {
                    let angle = std::f64::consts::TAU * j as f64 / m as f64;
                    let radius = if m == 32 && j % 2 == 0 { r * 0.74 } else { r };
                    [radius * angle.cos(), radius * angle.sin()]
                };
                let (a, b) = (point(i), point((i + 1) % m));
                edges.push([a[0], a[1], b[0], b[1]]);
            }
        }
        let table = indexed_outline(&edges);
        for row in 0..OUTLINE_SIDE {
            for col in 0..OUTLINE_SIDE {
                let h = table[edges.len() + row * OUTLINE_SIDE + col];
                let start = h[0] as usize;
                let local: Vec<_> = table[start..start + h[1] as usize].iter().map(|e| e.map(f64::from)).collect();
                for (u, v) in [(0.01, 0.02), (0.5, 0.5), (0.99, 0.98), (0.2, 0.72)] {
                    let coord = |i, f| (i as f64 + f) * 2.0 * OUTLINE_EXTENT / OUTLINE_SIDE as f64 - OUTLINE_EXTENT;
                    let (x, y) = (coord(col, u), coord(row, v));
                    let expected = section(x, y, 1.0, 1.0, 0.0, Some(&edges));
                    let actual = section(x, y, 1.0, 1.0, 0.0, Some(&local)) * if h[2] == 1.0 { -1.0 } else { 1.0 };
                    assert!((expected - actual).abs() < 1e-6, "cell {col},{row} at {x},{y}: {actual} vs {expected}");
                }
            }
        }
    }
    use crate::path::{Path, Piece};

    fn tube(w: f64) -> Profile {
        Profile { w, h: w, round: 1.0, turn: 0.0, start: 1.0, end: 1.0, round_caps: false, outline: None }
    }
    fn d_all(b: &Built, pr: &Profile, p: [f64; 3]) -> f64 {
        b.prims.iter().map(|q| dist(p, q, pr, b.length)).fold(f64::MAX, f64::min)
    }

    #[test]
    fn a_flat_turn_is_a_torus_inside_and_out() {
        let mut path = Path::default();
        path.start.p = [0.0; 3];
        path.pieces = vec![Piece::Straight { len: 0.3 }, Piece::Turn { deg: 90.0, radius: 0.3 }, Piece::Straight { len: 0.3 }];
        let pr = tube(0.16);
        let b = build(&path, 0.0, false);
        assert_eq!(b.prims.len(), 3);
        let centre = [0.3, 0.0, -0.3];
        let mut worst: f64 = 0.0;
        for i in 0..200 {
            let th = (2.0 + 86.0 * i as f64 / 200.0).to_radians();
            for j in 0..24 {
                let ph = j as f64 / 24.0 * std::f64::consts::TAU;
                for off in [0.0, 0.02, -0.03, 0.1] {
                    let rr = 0.08 + off;
                    let ring = [th.sin(), 0.0, th.cos()];
                    let q = add(add(centre, scale(ring, 0.3 + rr * ph.cos())), [0.0, rr * ph.sin(), 0.0]);
                    worst = worst.max((d_all(&b, &pr, q) - off).abs());
                }
            }
        }
        assert!(worst < 1e-9, "worst {worst}");
    }

    #[test]
    fn a_slope_and_a_strip_are_exact_too() {
        let mut path = Path::default();
        path.start.p = [0.0; 3];
        path.pieces = vec![Piece::Slope { deg: 90.0, radius: 0.25 }];
        // a strip 0.3 across (along the arc's axis: the track's left), 0.04 through (radial)
        let pr = Profile { w: 0.3, h: 0.04, round: 0.0, turn: 0.0, start: 1.0, end: 1.0, round_caps: false, outline: None };
        let b = build(&path, pr.turn, false);
        // the arc's centre is above the start; at 45 degrees round, a point on the strip's inner face
        let c = [0.0, 0.25, 0.0];
        let a = 45f64.to_radians();
        let radial = [a.sin(), -a.cos(), 0.0];
        for (off, want) in [(-0.03, 0.01), (0.005, -0.015), (0.0, -0.02), (0.05, 0.03)] {
            let q = add(c, scale(radial, 0.25 - off));
            let d = d_all(&b, &pr, q);
            assert!((d - want).abs() < 1e-9, "{off}: {d}");
        }
        // along the strip's width (the axis), past its edge
        let q = add(add(c, scale(radial, 0.25)), [0.0, 0.0, 0.2]);
        assert!((d_all(&b, &pr, q) - 0.05).abs() < 1e-9);
    }

    #[test]
    fn the_walk_follows_the_primitives_and_closes() {
        let mut path = Path::default();
        for _ in 0..4 {
            path.pieces.push(Piece::Straight { len: 0.5 });
            path.pieces.push(Piece::Turn { deg: 90.0, radius: 0.2 });
        }
        let b = build(&path, 0.0, true);
        assert!(b.closed);
        assert!(b.prims.iter().all(|p| p.ends == [0, 0]));
        let w = walk(&path);
        assert!(w.closed);
        // every sample on a primitive's centre line
        for s in &w.samples { assert!(b.prims.iter().any(|q| dist(s.p, q, &tube(0.0), b.length).abs() < 1e-9)); }
    }

    #[test]
    fn a_curve_is_arcs_that_meet_smoothly_and_end_where_it_does() {
        let mut path = Path::default();
        path.start.p = [0.0; 3];
        path.pieces = vec![Piece::Curve { out: 0.2, h2: [0.4, 0.1, 0.3], end: [0.5, 0.0, 0.5] }];
        let b = build(&path, 0.0, false);
        assert!(b.prims.len() >= CURVE_ARCS - 1);
        for w in b.prims.windows(2) {
            let (p1, t1, _) = w[0].at(1.0);
            let (p2, t2, _) = w[1].at(0.0);
            assert!(len3(sub(p1, p2)) < 1e-9);
            assert!(dot(t1, t2) > 1.0 - 1e-9);
        }
        let end = b.joints.last().unwrap().p;
        assert!(len3(sub(end, [0.5, 0.0, -0.5])) < 1e-9, "{end:?}");
    }

    #[test]
    fn a_helix_and_a_twist_never_overstate() {
        // a turn while climbing, and a bank: the bound never more than the distance (checked against points
        // of the swept solid's surface found by search)
        let mut path = Path::default();
        path.start.p = [0.0; 3];
        path.start.pitch = 20.0;
        path.pieces = vec![Piece::Turn { deg: 120.0, radius: 0.3 }, Piece::Bank { deg: 90.0, len: 0.4 }];
        let pr = Profile { w: 0.2, h: 0.05, round: 0.2, turn: 0.0, start: 1.0, end: 1.0, round_caps: false, outline: None };
        let b = build(&path, 0.0, false);
        // dense samples of the surface: for each primitive, sections at many shares, their outlines
        let mut surf = Vec::new();
        for p in &b.prims {
            for i in 0..=60 {
                let u = i as f64 / 60.0;
                let (q, t, side) = p.at(u);
                // the section's plane: square to a segment; through the axis for an arc (where the formula
                // measures it), its across and through directions there
                let (sx, sy) = match p.geo {
                    Geo::Seg { .. } => (side, cross(t, side)),
                    Geo::Arc { c, n, e0, theta, .. } => {
                        let a = theta * u;
                        let e = add(scale(e0, a.cos()), scale(cross(n, e0), a.sin()));
                        let (al, be) = (p.side[0], p.side[1]);
                        let _ = c;
                        (add(scale(e, al), scale(n, be)), add(scale(e, be), scale(n, -al)))
                    }
                };
                for j in 0..80 {
                    let a = j as f64 / 80.0 * std::f64::consts::TAU;
                    // a point of the section's outline (a rectangle: walked round by angle)
                    let (cx, cy) = (a.cos(), a.sin());
                    let k = (0.1 / cx.abs().max(1e-9)).min(0.025 / cy.abs().max(1e-9));
                    surf.push(add(q, add(scale(sx, cx * k), scale(sy, cy * k))));
                }
            }
        }
        for i in 0..400 {
            let q = [-0.2 + 0.9 * ((i * 37) % 100) as f64 / 100.0, -0.1 + 0.6 * ((i * 53) % 100) as f64 / 100.0, -0.7 + 0.8 * ((i * 71) % 100) as f64 / 100.0];
            let d = d_all(&b, &pr, q);
            if d <= 0.0 { continue; }
            let truth = surf.iter().map(|s| len3(sub(*s, q))).fold(f64::MAX, f64::min);
            assert!(d <= truth + 2e-3, "{q:?}: {d} > {truth}");
        }
    }
}
