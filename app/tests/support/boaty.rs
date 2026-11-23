//! boatyN: the boat, the lab's standard modelling workload. A curvy glass boat made of the app's own
//! parts (studio ellipsoids, ovals, capsules, cylinders, blocks, cones, spheres and the solids' round ring),
//! joined by smooth unions with a few subtractions, in eight modules. N parts in all (16 to 8192); as N grows
//! the same boat gains detail (ribs, strakes, portholes, planks, rail posts, stays, bunting, chain links,
//! fenders, rivets), never another boat.
//!
//! The app holds one tree of at most 64 parts and at most 8 objects. N <= 64: ONE tree (the modules one after
//! the other, the hull's main ellipsoid the object's own shape, at the middle). 64 < N <= 512: each module is
//! an object of its own (a tree of at most 64 parts; objects set apart: there is no seam where two modules
//! meet). N > 512: n/a for the app's back-ends. The lab's back-ends always read all N as one tree (`prims`).
//!
//! What the app cannot express from its settings, and so is not here: arrays inside a tree (an array is a
//! whole object's modifier: every repeat here is its own part), a twist or bend of a group (whole objects
//! only), chamfer seams inside a tree (between objects only).

use gf_core::csg::{Op, Part};

/// Bump whenever the boat's geometry changes: results from another rev are another scene.
pub const REV: u32 = 3;

thread_local! {
    /// boatyNs: the ribs are rings stretched to the hull's section (a part stretched unevenly: see lab_boaty.md
    /// for what that costs the app)
    static STRETCH: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// a build-up stage (1 hull, 2 + deck, 3 + cabin, portholes, mast, funnel, 4 everything); 0: the whole boat
    static STAGE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}
/// Is what belongs to build-up stage `s` drawn?
fn at(s: u8) -> bool { STAGE.with(|c| c.get() == 0 || c.get() >= s) }
/// boatyN as it stands at build-up stage `stage` (1..4): the parts of boatyN that belong to that stage or an
/// earlier one, the detail shared out as for the whole boat (so the steps are steps of the same boat).
pub fn build_stage(n: usize, stage: u8) -> Result<(Vec<Vec<Part>>, Vec<Vec<usize>>), String> {
    let (_, counts) = build(n)?;
    STAGE.with(|s| s.set(stage));
    let first_stage = [1u8, 2, 3, 3, 4, 4, 4, 4];
    let mods: Vec<Vec<Part>> = MODULES.iter().zip(&counts).enumerate().map(|(i, (m, c))| if at(first_stage[i]) { (m.build)(n, c) } else { Vec::new() }).collect();
    STAGE.with(|s| s.set(0));
    Ok((mods, counts))
}
/// The boat of N parts, its ribs stretched (`stretched`) or round.
pub fn build_with(n: usize, stretched: bool) -> Result<(Vec<Vec<Part>>, Vec<Vec<usize>>), String> {
    STRETCH.with(|s| s.set(stretched));
    let r = build(n);
    STRETCH.with(|s| s.set(false));
    r
}

type V = [f64; 3];
fn add(a: V, b: V) -> V { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
fn sub(a: V, b: V) -> V { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn mul(a: V, s: f64) -> V { [a[0] * s, a[1] * s, a[2] * s] }
fn dot(a: V, b: V) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn cross(a: V, b: V) -> V { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }
fn len(a: V) -> f64 { dot(a, a).sqrt() }
fn norm(a: V) -> V { mul(a, 1.0 / len(a).max(1e-12)) }
fn lerp(a: V, b: V, t: f64) -> V { add(a, mul(sub(b, a), t)) }

/// The app's angles (degrees; R = Rz Ry Rx, gf_core::math::euler) of the frame whose x goes to `ex` and y to
/// `ey` (made square first).
fn frame(ex: V, ey: V) -> V {
    let ex = norm(ex);
    let ey = norm(sub(ey, mul(ex, dot(ey, ex))));
    let ez = cross(ex, ey);
    let d = 180.0 / std::f64::consts::PI;
    if ex[2].abs() > 0.99999 {
        // (gimbal: rz = 0, M = Ry Rx)
        let ry = -ex[2].clamp(-1.0, 1.0).asin();
        return [(-ez[1]).atan2(ey[1]) * d, ry * d, 0.0];
    }
    [ey[2].atan2(ez[2]) * d, -ex[2].clamp(-1.0, 1.0).asin() * d, ex[1].atan2(ex[0]) * d]
}
/// Any frame whose y is `axis`.
fn y_to(axis: V) -> V {
    let a = norm(axis);
    let hint = if a[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 0.0, 1.0] };
    frame(sub(hint, mul(a, dot(hint, a))), a)
}
/// Any frame whose x is `axis` (y kept as near up as it can be).
fn x_to(axis: V) -> V {
    let a = norm(axis);
    let up = if a[1].abs() < 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
    frame(a, up)
}

// ---- parts (real sizes in; each made at a size that puts its own reach near 1.1, inside the studio's fit) ----

fn part(fam: &str, preset: &str, pos: V, rot: V, size: f64, vals: &[(&str, f64)]) -> Part {
    let mut p = Part { family: fam.into(), preset: preset.into(), ..Default::default() };
    p.place.pos = pos;
    p.place.rot = rot;
    p.place.size = size;
    for (k, v) in vals { p.values.insert(k.to_string(), *v); }
    p
}
fn ell(c: V, r: V, rot: V) -> Part {
    let s = r[0].max(r[1]).max(r[2]);
    part("studio", "ellipsoid", c, rot, s, &[("rx", r[0] / s), ("ry", r[1] / s), ("rz", r[2] / s)])
}
fn ball(c: V, r: f64) -> Part { part("studio", "sphere", c, [0.0; 3], r, &[("radius", 1.0), ("cut", 0.0), ("bevel", 0.0), ("round", 0.0)]) }
fn block(c: V, d: V, round: f64, rot: V) -> Part {
    let s = 0.5 * len(d) / 1.1;
    part("studio", "block", c, rot, s, &[("width", d[0] / s), ("height", d[1] / s), ("depth", d[2] / s), ("bevel", 0.0), ("round", round / s)])
}
/// an upright cylinder (along y before `rot`)
fn cyl(c: V, r: f64, h: f64, round: f64, rot: V) -> Part {
    let s = r.hypot(0.5 * h) / 1.1;
    part("studio", "cylinder", c, rot, s, &[("radius", r / s), ("height", h / s), ("bevel", 0.0), ("round", round / s)])
}
/// an upright elliptic cylinder: radii rx (x), rz (z), height h
fn oval(c: V, rx: f64, rz: f64, h: f64, round: f64) -> Part {
    let s = rx.max(rz).hypot(0.5 * h) / 1.1;
    part("studio", "oval", c, [0.0; 3], s, &[("rx", rx / s), ("rz", rz / s), ("height", h / s), ("bevel", 0.0), ("round", round / s)])
}
fn cone(c: V, r: f64, top: f64, h: f64, rot: V) -> Part {
    let s = r.max(top).hypot(0.5 * h) / 1.1;
    part("studio", "cone", c, rot, s, &[("radius", r / s), ("top", top / s), ("height", h / s), ("bevel", 0.0), ("round", 0.004 / s)])
}
/// a rod with round ends from a to b, radius r
fn rod(a: V, b: V, r: f64) -> Part {
    let l = len(sub(b, a));
    let half = 0.5 * l + r;
    let s = half.max(r) / 1.1;
    part("studio", "capsule", lerp(a, b, 0.5), x_to(sub(b, a)), s, &[("length", 2.0 * half / s), ("radius", r / s)])
}
/// a torus of radius big and tube r about `axis` (the solids' round ring: fitted to 1.22 = R + r)
fn torus(c: V, big: f64, r: f64, axis: V) -> Part {
    let fat = (r / big).max(0.02);
    let s = big / (1.22 / (1.0 + fat));
    part("solids", "ring", c, y_to(axis), s, &[("fat", fat), ("section", 2.0), ("flat", 1.0)])
}
fn op(mut p: Part, o: Op, smooth: f64) -> Part { p.op = o; p.smooth = smooth; p }
fn u(p: Part, smooth: f64) -> Part { op(p, Op::Union, smooth) }
fn cut(p: Part, smooth: f64) -> Part { op(p, Op::Subtract, smooth) }

// ---- the hull's shape, for placing things on it ----

/// Hull ellipsoids: centre x, y; radii x, y, z; tilt about z (degrees). The first is the object's own shape.
const HULL: [(f64, f64, f64, f64, f64, f64); 5] = [
    (0.0, 0.0, 1.05, 0.40, 0.44, 0.0),    // main, midships
    (0.55, 0.03, 0.70, 0.37, 0.33, 0.0),  // forward body
    (0.98, 0.09, 0.34, 0.31, 0.13, 18.0), // the bow: narrow, its forefoot raked up
    (-0.55, 0.0, 0.62, 0.37, 0.43, 0.0),  // after body
    (-0.88, 0.04, 0.32, 0.30, 0.37, 0.0), // the round stern
];
const HULL_SMOOTH: f64 = 0.12;
/// The deck is the bottom of a large ball subtracted from the hull: a sheer that rises to bow and stern.
const CUT_R: f64 = 5.0;
const DECK0: f64 = -0.06;
fn deck_y(x: f64, z: f64) -> f64 { DECK0 + CUT_R - (CUT_R * CUT_R - x * x - z * z).sqrt() }
/// the deck's normal there (up)
fn deck_n(x: f64, z: f64) -> V { norm([-x, DECK0 + CUT_R - deck_y(x, z), -z]) }

fn smin(a: f64, b: f64, k: f64) -> f64 {
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    b + (a - b) * h - k * h * (1.0 - h)
}
/// The hull's field (ellipsoids' estimated distances, smooth-joined): negative inside.
fn hull_f(p: V) -> f64 {
    let mut d = f64::MAX;
    for (i, &(cx, cy, rx, ry, rz, t)) in HULL.iter().enumerate() {
        let (s, c) = t.to_radians().sin_cos();
        let (dx, dy) = (p[0] - cx, p[1] - cy);
        let q = [c * dx + s * dy, -s * dx + c * dy, p[2]];
        let k0 = len([q[0] / rx, q[1] / ry, q[2] / rz]);
        let k1 = len([q[0] / (rx * rx), q[1] / (ry * ry), q[2] / (rz * rz)]);
        let e = k0 * (k0 - 1.0) / k1.max(1e-9);
        d = if i == 0 { e } else { smin(d, e, HULL_SMOOTH) };
    }
    d
}
/// Where a ray from `o` (inside the hull) along `dir` leaves it.
fn hull_hit(o: V, dir: V) -> V {
    let dir = norm(dir);
    let (mut a, mut b) = (0.0, 1.5);
    for _ in 0..48 {
        let m = 0.5 * (a + b);
        if hull_f(add(o, mul(dir, m))) < 0.0 { a = m } else { b = m }
    }
    add(o, mul(dir, 0.5 * (a + b)))
}
/// The hull's surface normal (outwards).
fn hull_n(p: V) -> V {
    let e = 1e-4;
    norm([
        hull_f(add(p, [e, 0.0, 0.0])) - hull_f(sub(p, [e, 0.0, 0.0])),
        hull_f(add(p, [0.0, e, 0.0])) - hull_f(sub(p, [0.0, e, 0.0])),
        hull_f(add(p, [0.0, 0.0, e])) - hull_f(sub(p, [0.0, 0.0, e])),
    ])
}
/// The hull's middle line height at x (the section's centre), and its half-depth below it.
fn section_centre(x: f64) -> f64 {
    // the nearest of the hull ellipsoids' centres, weighted by how much of the section each is
    let mut w = 0.0;
    let mut y = 0.0;
    for &(cx, cy, rx, _, _, _) in HULL.iter() {
        let f = (1.0 - ((x - cx) / rx).powi(2)).max(0.0);
        w += f;
        y += f * cy;
    }
    if w > 0.0 { y / w } else { 0.0 }
}
/// A point on the hull's side at x, `phi` round from the deck level (0) to the keel (pi/2), side +1/-1.
fn on_hull(x: f64, phi: f64, side: f64) -> V {
    let c = [x, section_centre(x) - 0.02, 0.0];
    hull_hit(c, [0.0, -phi.sin(), side * phi.cos()])
}
/// The deck's edge at x (z > 0), and the deck's height there.
fn deck_edge(x: f64) -> f64 {
    let mut z = 0.0;
    for _ in 0..3 {
        let y = deck_y(x, z);
        if hull_f([x, y, 0.0]) >= 0.0 { return 0.0; }
        z = hull_hit([x, y, 0.0], [0.0, 0.0, 1.0])[2];
    }
    z
}
/// The keel: a narrow fin (centre x, y; radii x, y, z; tilt about z, degrees).
/// (a thin slice of a hull-long ellipsoid: its lower edge follows the hull's rocker a little below it, deepest
/// midships and running out into the hull at bow and stern; the rest is inside the hull or cut by the sheer)
const KEEL: (f64, f64, f64, f64, f64, f64) = (0.02, 0.0, 1.1, 0.5, 0.028, 0.0);
/// The keel's bottom edge at x.
fn keel_bottom(x: f64) -> V {
    let (cx, cy, rx, ry, rz, t) = KEEL;
    let f = |p: V| {
        let (s, c) = t.to_radians().sin_cos();
        let (dx, dy) = (p[0] - cx, p[1] - cy);
        let q = [c * dx + s * dy, -s * dx + c * dy, p[2]];
        (q[0] / rx).powi(2) + (q[1] / ry).powi(2) + (q[2] / rz).powi(2) - 1.0
    };
    let (mut a, mut b) = (0.0, 0.5);
    let (mut a, mut b) = (a, b + 0.3);
    for _ in 0..48 { let m = 0.5 * (a + b); if f([x, cy - m, 0.0]) < 0.0 { a = m } else { b = m } }
    [x, cy - 0.5 * (a + b), 0.0]
}
/// The stem head: the deck's foremost point, on the centreline, a little back from the tip.
fn stem_head() -> V {
    let mut x = 1.4;
    while x > 0.8 && hull_f([x, deck_y(x, 0.0) - 0.005, 0.0]) >= 0.0 { x -= 0.002; }
    let x = x - 0.025;
    [x, deck_y(x, 0.0), 0.0]
}
/// The deck's outline as a U from the bow round the stern and back (starboard first), inset by `inset`, as
/// (x, z) points, and their running lengths.
fn outline(inset: f64, bow_x: f64) -> (Vec<[f64; 2]>, Vec<f64>) {
    let n = 160;
    let (x0, x1) = (-1.19, bow_x);
    let mut side: Vec<[f64; 2]> = (0..=n).map(|i| { let x = x1 + (x0 - x1) * i as f64 / n as f64; [x, deck_edge(x)] }).collect();
    side.retain(|p| p[1] > 0.0);
    let mut pts = side.clone();
    for p in side.iter().rev() { pts.push([p[0], -p[1]]); }
    // inset along the outline's inward normal
    let m = pts.len();
    let mut out = Vec::with_capacity(m);
    for i in 0..m {
        let a = pts[i.saturating_sub(1)];
        let b = pts[(i + 1).min(m - 1)];
        let t = [b[0] - a[0], b[1] - a[1]];
        let l = (t[0] * t[0] + t[1] * t[1]).sqrt().max(1e-9);
        // (the U runs clockwise seen from above: inward is the tangent turned left in (x, z))
        let nrm = [-t[1] / l, t[0] / l];
        let s = if pts[i][1] >= 0.0 { 1.0 } else { 1.0 };
        out.push([pts[i][0] + s * nrm[0] * inset, pts[i][1] + s * nrm[1] * inset]);
    }
    let mut acc = vec![0.0];
    for i in 1..m { let d = ((out[i][0] - out[i - 1][0]).powi(2) + (out[i][1] - out[i - 1][1]).powi(2)).sqrt(); acc.push(acc[i - 1] + d); }
    (out, acc)
}
/// The point a share `t` (0..1) of the way along an outline.
fn along(o: &(Vec<[f64; 2]>, Vec<f64>), t: f64) -> [f64; 2] {
    let total = *o.1.last().unwrap();
    let s = t.clamp(0.0, 1.0) * total;
    let i = o.1.partition_point(|&a| a < s).clamp(1, o.0.len() - 1);
    let (a, b) = (o.1[i - 1], o.1[i]);
    let f = if b > a { (s - a) / (b - a) } else { 0.0 };
    [o.0[i - 1][0] + (o.0[i][0] - o.0[i - 1][0]) * f, o.0[i - 1][1] + (o.0[i][1] - o.0[i - 1][1]) * f]
}
/// k sizes adding up to n, as even as can be.
fn spread(n: usize, k: usize) -> Vec<usize> { (0..k).map(|i| n / k + usize::from(i < n % k)).collect() }
/// k points along a chain of capsules through `pts`: the chain's k segments (as even in length as can be).
fn chain(pts: &[V], k: usize, r: f64) -> Vec<Part> {
    if k == 0 || pts.len() < 2 { return Vec::new(); }
    let mut acc = vec![0.0];
    for i in 1..pts.len() { acc.push(acc[i - 1] + len(sub(pts[i], pts[i - 1]))); }
    let total = *acc.last().unwrap();
    let at = |s: f64| {
        let i = acc.partition_point(|&a| a < s).clamp(1, pts.len() - 1);
        let f = if acc[i] > acc[i - 1] { (s - acc[i - 1]) / (acc[i] - acc[i - 1]) } else { 0.0 };
        lerp(pts[i - 1], pts[i], f.clamp(0.0, 1.0))
    };
    (0..k).map(|j| rod(at(total * j as f64 / k as f64), at(total * (j + 1) as f64 / k as f64), r)).map(|p| u(p, 0.006)).collect()
}

// ---- the modules ----

/// A module: its parts for N (cores appear from a given N; streams take the counts given them).
pub struct Module {
    pub name: &'static str,
    pub story: &'static str,
    /// its streams of detail: name, weight, most parts, parts at a time (2: a part and its mirror)
    pub streams: &'static [(&'static str, f64, usize, usize)],
    build: fn(usize, &[usize]) -> Vec<Part>,
}

const PROP: V = [-0.99, -0.40, 0.0];

fn hull(n: usize, k: &[usize]) -> Vec<Part> {
    let mut v = Vec::new();
    for (i, &(cx, cy, rx, ry, rz, t)) in HULL.iter().enumerate() {
        v.push(u(ell([cx, cy, 0.0], [rx, ry, rz], [0.0, 0.0, t]), if i == 0 { 0.0 } else { HULL_SMOOTH }));
    }
    v.push(u(ell([KEEL.0, KEEL.1, 0.0], [KEEL.2, KEEL.3, KEEL.4], [0.0, 0.0, KEEL.5]), 0.04)); // the keel: a narrow fin on the centreline following the hull's curve, filleted into it
    if at(4) { v.push(u(ell([-1.17, -0.25, 0.0], [0.12, 0.19, 0.022], [0.0, 0.0, -10.0]), 0.03)); } // the rudder, hung on the stern
    // ribs: tori stretched to the hull's section at their x, their upper halves taken by the sheer cut
    let ribs = k[0];
    for j in 0..ribs {
        let x = -0.82 + 1.72 * (j as f64 + 0.5) / ribs as f64;
        let cy = section_centre(x);
        let b = hull_hit([x, cy, 0.0], [0.0, 0.0, 1.0])[2];
        let d = cy - hull_hit([x, cy, 0.0], [0.0, -1.0, 0.0])[1];
        if b < 0.05 || d < 0.05 { v.push(u(ball([x, cy - d, 0.0], 0.012), 0.01)); continue; }
        if STRETCH.with(|s| s.get()) {
            // (boatyNs: a ring drawn out to the section's depth: its own y along the boat, its own x the scene's up)
            let mut p = torus([x, cy, 0.0], b + 0.002, 0.009, [1.0, 0.0, 0.0]);
            p.stretch = Some([d / b, 1.0, 1.0]);
            p.place.rot = frame([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]);
            v.push(u(p, 0.008));
        } else {
            // a round ring as wide as the section, raised so that it meets the keel line (the sheer cuts its top)
            v.push(u(torus([x, cy - d + b, 0.0], b + 0.004, 0.009, [1.0, 0.0, 0.0]), 0.008));
        }
    }
    // strakes: capsule chains along the hull's sides (a rubbing strake under the deck, then the bilges)
    let lines: &[(f64, f64)] = &[(0.24, 1.0), (0.24, -1.0), (0.70, 1.0), (0.70, -1.0), (0.47, 1.0), (0.47, -1.0)];
    // (by pairs: each line and its mirror get alike)
    let pairs = if k[1] >= 24 { 3 } else if k[1] >= 8 { 2 } else { 1 };
    let per: Vec<usize> = spread(k[1] / 2, pairs).into_iter().flat_map(|c| [c, c]).collect();
    for (li, cnt) in per.into_iter().enumerate() {
        let (phi, side) = lines[li];
        let (xa, xb) = if phi < 0.3 { (-1.02, 1.15) } else { (-0.85, 1.0) };
        let pts: Vec<V> = (0..=40).map(|i| { let x = xa + (xb - xa) * i as f64 / 40.0; let p = on_hull(x, phi, side); add(p, mul(hull_n(p), 0.002)) }).collect();
        v.extend(chain(&pts, cnt, 0.011));
    }
    // the sheer: a large ball cut from above (the deck)
    v.push(cut(ball([0.0, DECK0 + CUT_R, 0.0], CUT_R), 0.02));
    // portholes: a round recess cut into the side, and a ring round it
    // (by fours: a porthole either side, each a cut and its ring)
    let slots = if at(3) { k[2] / 4 } else { 0 };
    for slot in 0..slots {
        for side in [1.0, -1.0] {
            let x = 0.62 - 1.3 * (slot as f64 + 0.5) / slots as f64;
            let p = on_hull(x, 0.5, side);
            let nrm = hull_n(p);
            v.push(cut(cyl(p, 0.034, 0.05, 0.004, y_to(nrm)), 0.006));
            v.push(u(torus(add(p, mul(nrm, 0.007)), 0.042, 0.010, nrm), 0.004));
        }
    }
    if n >= 32 && at(4) {
        // the propeller: a hub, blades, (a ring round them, its shaft)
        v.push(u(ell(PROP, [0.05, 0.028, 0.028], [0.0; 3]), 0.01));
        let blades = if n >= 256 { 4 } else { 3 };
        for b in 0..blades {
            let a = b as f64 / blades as f64 * std::f64::consts::TAU + 0.3;
            let radial = [0.0, a.cos(), a.sin()];
            let c = add(PROP, mul(radial, 0.062));
            // (a blade's chord turned 35 degrees off the disc: its pitch)
            let chord = add(mul([0.0, -a.sin(), a.cos()], 35f64.to_radians().cos()), mul([1.0, 0.0, 0.0], 35f64.to_radians().sin()));
            v.push(u(ell(c, [0.032, 0.05, 0.008], frame(chord, radial)), 0.012));
        }
    }
    if n >= 64 && at(4) {
        v.push(u(torus(PROP, 0.095, 0.012, [1.0, 0.0, 0.0]), 0.0));
        v.push(u(rod(PROP, [-0.66, -0.33, 0.0], 0.014), 0.02));
    }
    v
}

fn deck(n: usize, k: &[usize]) -> Vec<Part> {
    let mut v = Vec::new();
    if n >= 32 {
        // the fore hatch (the object's own shape: a block, square to the axes)
        v.push(u(block([HATCH_X, deck_y(HATCH_X, 0.0) + 0.03, 0.0], [0.22, 0.08, 0.24], 0.02, [0.0; 3]), 0.0));
    }
    // planks: rows across the deck, each in segments (more rows, then more segments, as N grows)
    let planks = k[0];
    if planks > 0 {
        let rows = (planks / 3).clamp(1, 34);
        let pitch = deck_span() / rows as f64;
        for (r, segs) in spread(planks, rows).into_iter().enumerate() {
            let x = deck_x((r as f64 + 0.5) / rows as f64);
            let e = deck_edge(x) - 0.012;
            if e <= 0.02 { for _ in 0..segs { v.push(u(ball([x, deck_y(x, 0.0), 0.0], 0.01), 0.004)); } continue; }
            for s in 0..segs {
                let za = -e + 2.0 * e * s as f64 / segs as f64;
                let zb = -e + 2.0 * e * (s + 1) as f64 / segs as f64;
                let zc = 0.5 * (za + zb);
                let nrm = deck_n(x, zc);
                let c = add([x, deck_y(x, zc), zc], mul(nrm, 0.004));
                let w = (pitch * 0.82).min(0.11);
                v.push(u(block(c, [w, 0.014, (zb - za - 0.008).max(0.01)], 0.004, frame(sub([1.0, 0.0, 0.0], mul(nrm, nrm[0])), nrm)), 0.003));
            }
        }
    }
    // cleats: a bar on two feet's worth of rod, near the edges
    for j in 0..k[1] {
        let side = if j % 2 == 0 { 1.0 } else { -1.0 };
        let x = 0.95 - 1.9 * ((j / 2) as f64 + 0.5) / k[1].div_ceil(2) as f64;
        let z = side * (deck_edge(x) - 0.06).max(0.0);
        let y = deck_y(x, z) + 0.012;
        v.push(u(rod([x - 0.035, y, z], [x + 0.035, y, z], 0.010), 0.012));
    }
    // deck studs: nail heads in rows
    let studs = k[2];
    if studs > 0 {
        let rows = (studs / 6).clamp(1, 40);
        for (r, m) in spread(studs, rows).into_iter().enumerate() {
            let x = deck_x((r as f64 + 0.5) / rows as f64);
            let e = deck_edge(x) - 0.03;
            for s in 0..m {
                let z = if e > 0.0 { -e + 2.0 * e * (s as f64 + 0.5) / m as f64 } else { 0.0 };
                v.push(u(ball(add([x, deck_y(x, z), z], mul(deck_n(x, z), 0.008)), 0.008), 0.003));
            }
        }
    }
    v
}

/// The open deck along x: from the stern to the bow, less the cabin's length (planks under it would be hidden).
const DECK_A: f64 = -1.06;
const DECK_B: f64 = 1.16;
/// (what covers the deck: the cabin, the fore hatch, the capstan)
const COVERS: [(f64, f64); 3] = [(CAB_X - CAB_RX - 0.01, CAB_X + CAB_RX + 0.01), (HATCH_X - 0.12, HATCH_X + 0.12), (0.87, 0.97)];
fn deck_span() -> f64 { (DECK_B - DECK_A) - COVERS.iter().map(|c| c.1 - c.0).sum::<f64>() }
/// The point a share `t` of the way along the open deck.
fn deck_x(t: f64) -> f64 {
    let mut x = DECK_A + t * deck_span();
    for (a, b) in COVERS { if x >= a { x += b - a; } }
    x
}
const HATCH_X: f64 = 0.48;
const CAB_X: f64 = -0.28;
const CAB_RX: f64 = 0.34;
const CAB_RZ: f64 = 0.26;
const CAB_H: f64 = 0.30;
fn cab_base() -> f64 { deck_y(CAB_X, 0.0) - 0.02 }

fn cabin(n: usize, k: &[usize]) -> Vec<Part> {
    let b = cab_base();
    let top = b + CAB_H;
    let mut v = vec![
        u(oval([CAB_X, b + 0.5 * CAB_H, 0.0], CAB_RX, CAB_RZ, CAB_H, 0.05), 0.0),
        u(ell([CAB_X, top - 0.005, 0.0], [CAB_RX + 0.05, 0.075, CAB_RZ + 0.05], [0.0; 3]), 0.03), // the roof
        cut(cyl([CAB_X + 0.1, b + 0.165, 0.0], 0.058, 0.7, 0.01, [90.0, 0.0, 0.0]), 0.012),         // a window through
    ];
    let wall = |x: f64| CAB_RZ * (1.0 - ((x - CAB_X) / CAB_RX).powi(2)).max(0.0).sqrt();
    if n >= 64 {
        // the front windows (short cuts into the forward wall) and the door aft
        for z in [-0.1, 0.1] { v.push(cut(cyl([CAB_X + CAB_RX * (1.0 - (z / CAB_RZ).powi(2)).sqrt(), b + 0.18, z], 0.04, 0.09, 0.008, [0.0, 0.0, 90.0]), 0.01)); }
        v.push(cut(block([CAB_X - CAB_RX, b + 0.15, 0.0], [0.1, 0.16, 0.12], 0.03, [0.0; 3]), 0.01)); // (its sill above the deck: the cut never reaches into the deck)
    }
    // side windows: a cut through both walls and a frame either side
    let wn = k[0];
    let items = wn.div_ceil(3);
    for j in 0..items {
        let x = CAB_X - 0.03 - 0.22 * (j as f64 + 0.5) / items as f64;
        let y = b + 0.165;
        v.push(cut(cyl([x, y, 0.0], 0.032, 0.7, 0.008, [90.0, 0.0, 0.0]), 0.01));
        let left = wn - 3 * j - 1;
        for (f, side) in [1.0, -1.0].into_iter().enumerate() {
            if f < left.min(2) { v.push(u(torus([x, y, side * (wall(x) + 0.004)], 0.039, 0.007, [0.0, 0.0, 1.0]), 0.004)); }
        }
    }
    // the life-ring on the after wall, and the funnel on the roof (raked aft)
    v.push(u(torus([CAB_X - CAB_RX * (1.0 - (0.13f64 / CAB_RZ).powi(2)).sqrt() - 0.018, b + 0.15, 0.13], 0.06, 0.022, [1.0, 0.0, 0.0]), 0.01));
    let fx = CAB_X - 0.13;
    let fy = top + 0.06;
    let rake = 8.0;
    v.push(u(cyl([fx, fy + 0.11, 0.0], 0.055, 0.30, 0.012, [0.0, 0.0, rake]), 0.03));
    if n >= 32 {
        let (s, c) = rake.to_radians().sin_cos();
        let up = [-s, c, 0.0];
        v.push(u(torus(add([fx, fy + 0.11, 0.0], mul(up, 0.07)), 0.058, 0.011, up), 0.004)); // its band
        v.push(u(ball([CAB_X + 0.2, top + 0.075, 0.0], 0.03), 0.02)); // a roof lamp
        if n >= 64 { v.push(u(torus(add([fx, fy + 0.11, 0.0], mul(up, 0.15)), 0.054, 0.014, up), 0.004)); } // its rim
    }
    // roof trim: beads round the roof's edge
    for j in 0..k[1] {
        let a = j as f64 / k[1] as f64 * std::f64::consts::TAU;
        v.push(u(ball([CAB_X + (CAB_RX + 0.035) * a.cos(), top + 0.01, (CAB_RZ + 0.035) * a.sin()], 0.011), 0.004));
    }
    v
}

const MAST_X: f64 = 0.30;
const MAST_TOP: f64 = 1.0;
fn rigging(n: usize, k: &[usize]) -> Vec<Part> {
    // (the ropes: stays, hoops and bunting come at build-up stage 4)
    let k: &[usize] = if at(4) { k } else { &[0, 0, 0] };
    let base = deck_y(MAST_X, 0.0) - 0.02;
    let mut v = vec![u(cyl([MAST_X, 0.5 * (base + MAST_TOP), 0.0], 0.026, MAST_TOP - base, 0.01, [0.0; 3]), 0.0)];
    // the boom, swung out to starboard, and the flag flying aft from the masthead
    let goose = [MAST_X - 0.02, 0.56, 0.0];
    let end = [-0.30, 0.60, 0.0];
    v.push(u(rod(goose, end, 0.018), 0.015));
    v.push(u(block([MAST_X - 0.11, MAST_TOP - 0.06, 0.0], [0.18, 0.1, 0.008], 0.003, [0.0, 8.0, 0.0]), 0.008));
    if n >= 32 {
        // the topping lift (a rope from the masthead to the boom's end, holding it up), the masthead ball
        v.push(u(rod([MAST_X, MAST_TOP - 0.08, 0.0], end, 0.005), 0.006));
        v.push(u(ball([MAST_X, MAST_TOP + 0.02, 0.0], 0.032), 0.01));
    }
    if n >= 64 { v.push(u(cyl([MAST_X, 0.80, 0.0], 0.07, 0.014, 0.005, [0.0; 3]), 0.006)); } // the crow's nest: a round platform on the mast
    let top = [MAST_X, 0.86, 0.0];
    let stem = stem_head();
    // stays: the forestay from the masthead to an eye on the stem head, shrouds to the deck's edges abreast
    // the mast, backstays to the quarters (every end on the mast, the eye or the deck). The eye is the stays'
    // first part; then the forestay; the shrouds and backstays by pairs, alike either side.
    let ze = deck_edge(MAST_X) - 0.03;
    let zq = deck_edge(-0.95) - 0.03;
    let on_deck = |x: f64, z: f64| [x, deck_y(x, z) + 0.003, z];
    let lines: [(V, V); 5] = [
        (top, add(stem, [-0.01, 0.012, 0.0])),
        ([MAST_X, 0.76, 0.0], on_deck(MAST_X - 0.05, ze)),
        ([MAST_X, 0.76, 0.0], on_deck(MAST_X - 0.05, -ze)),
        (top, on_deck(-0.95, zq)),
        (top, on_deck(-0.95, -zq)),
    ];
    if k[0] > 0 {
        v.push(u(torus(add(stem, [0.0, 0.012, 0.0]), 0.014, 0.005, [0.0, 0.0, 1.0]), 0.004));
        let rest = k[0] - 1;
        let s = rest / 5;
        let counts = [rest - 4 * s, s, s, s, s];
        for (li, cnt) in counts.into_iter().enumerate() {
            let (a, b) = lines[li];
            // (from the mast's surface, not its axis: no segment hidden inside the mast)
            let a = add(a, mul(norm(sub(b, a)), 0.024));
            v.extend(chain(&[a, b], cnt, 0.006));
        }
    }
    // mast hoops
    for j in 0..k[1] {
        let y = 0.58 + 0.17 * (j as f64 + 0.5) / k[1] as f64;
        v.push(u(torus([MAST_X, y, 0.0], 0.034, 0.007, [0.0, 1.0, 0.0]), 0.003));
    }
    // bunting: little pennants hung under the forestay
    for j in 0..k[2] {
        let t = 0.08 + 0.84 * (j as f64 + 0.5) / k[2] as f64;
        let p = lerp(top, stem, t);
        let d = norm(sub(stem, top));
        v.push(u(ell(add(p, [0.0, -0.028, 0.0]), [0.022, 0.03, 0.004], frame(d, [0.0, 1.0, 0.0])), 0.004));
    }
    v
}

/// How far either way from the stern (a share of the deck's U) a rail of `segs` segments runs: round the
/// stern in short segments while it has few, the whole U from 8 on.
fn rail_span(segs: usize) -> f64 { if segs >= 8 { 0.48 } else { 0.015 * segs as f64 } }
fn rails(n: usize, k: &[usize]) -> Vec<Part> {
    let h = 0.11;
    let o = outline(0.03, 1.14);
    let at = |t: f64, y: f64| { let p = along(&o, t); [p[0], deck_y(p[0], p[1]) + y, p[1]] };
    let mut v = Vec::new();
    if n >= 64 {
        // the first post (the object's own shape: an upright cylinder)
        // (at the stern, on the centreline)
        let p = at(0.5, 0.5 * h);
        v.push(u(cyl(p, 0.009, h + 0.04, 0.003, [0.0; 3]), 0.0));
    }
    // posts by pairs, one either side (t and 1 - t along the U)
    let pairs = k[0] / 2;
    for j in 0..pairs {
        let t = 0.02 + 0.46 * (j as f64 + 0.5) / pairs as f64;
        for t in [t, 1.0 - t] { v.push(u(cyl(at(t, 0.5 * h), 0.008, h + 0.04, 0.003, [0.0; 3]), 0.006)); }
    }
    // the top rail and the middle rail: capsule chains along the posts' line
    // (a rail of few segments runs round the stern only, and grows forward along both sides as it gets more:
    // never a chord across the open bow)
    let line = |y: f64, segs: usize| { let w = rail_span(segs); (0..=120).map(|i| at(0.5 - w + 2.0 * w * i as f64 / 120.0, y)).collect::<Vec<V>>() };
    v.extend(chain(&line(h, k[1]), k[1], 0.011));
    v.extend(chain(&line(0.55 * h, k[2]), k[2], 0.006));
    // studs on the rail
    // (along the top rail as far as it runs)
    let w = rail_span(k[1]);
    for j in 0..k[3] {
        let t = 0.5 - w + 2.0 * w * (j as f64 + 0.5) / k[3] as f64;
        v.push(u(ball(at(t, h + 0.012), 0.009), 0.004));
    }
    v
}

const ANCHOR: V = [0.86, -0.12, 0.0];
fn bow(n: usize, k: &[usize]) -> Vec<Part> {
    let mut v = Vec::new();
    let cap_x = 0.92;
    let cap = [cap_x, deck_y(cap_x, 0.0) + 0.04, 0.0];
    if n >= 64 { v.push(u(cyl(cap, 0.04, 0.1, 0.01, [0.0; 3]), 0.0)); } // the capstan
    // the anchor, hung flat against the starboard bow
    let side = on_hull(ANCHOR[0], 0.45, 1.0);
    let nrm = hull_n(side);
    let base = add([ANCHOR[0], -0.21, side[2]], mul(nrm, 0.012));
    let up = [0.0, 1.0, 0.0];
    let along_x = norm(cross(up, nrm));
    let p = |dx: f64, dy: f64| add(base, add(mul(along_x, dx), mul(up, dy)));
    let crown = p(0.0, -0.1);
    if n >= 32 {
        v.push(u(rod(crown, p(0.0, 0.12), 0.011), 0.01));                       // shank
        v.push(u(torus(p(0.0, 0.145), 0.024, 0.006, nrm), 0.004));               // ring
        v.push(u(rod(crown, p(0.075, -0.03), 0.010), 0.012));                    // arms
        v.push(u(rod(crown, p(-0.075, -0.03), 0.010), 0.012));
    }
    if n >= 64 {
        v.push(u(rod(p(-0.05, 0.1), p(0.05, 0.1), 0.008), 0.006));                // stock
        v.push(u(ball(crown, 0.018), 0.01));                                       // crown
        for sx in [1.0, -1.0] { v.push(u(ell(p(sx * 0.08, -0.025), [0.026, 0.014, 0.006], frame(along_x, [sx * 0.6, 1.0, 0.0])), 0.006)); }
    }
    // the chain: links from the anchor's ring up over the bow and along the deck to the capstan
    let hawse = [ANCHOR[0] + 0.05, deck_y(ANCHOR[0], 0.2) + 0.02, deck_edge(ANCHOR[0] + 0.05) - 0.02];
    let path = [p(0.0, 0.17), add(hawse, [0.0, -0.02, 0.03]), hawse, [cap_x - 0.03, cap[1] - 0.02, 0.05]];
    let links = k[0];
    let mut acc = vec![0.0];
    for i in 1..path.len() { acc.push(acc[i - 1] + len(sub(path[i], path[i - 1]))); }
    let total = *acc.last().unwrap();
    for j in 0..links {
        let s = total * (j as f64 + 0.5) / links as f64;
        let i = acc.partition_point(|&a| a < s).clamp(1, path.len() - 1);
        let f = (s - acc[i - 1]) / (acc[i] - acc[i - 1]).max(1e-9);
        let c = lerp(path[i - 1], path[i], f);
        let d = norm(sub(path[i], path[i - 1]));
        let l = (total / links as f64 * 0.62).clamp(0.006, 0.02);
        // (links alternate: the ring's axis across the chain, turned a quarter each time)
        let side = norm(cross(d, if j % 2 == 0 { [0.0, 1.0, 0.0] } else { nrm }));
        v.push(u(torus(c, l, l * 0.35, side), 0.0));
    }
    // a coil of rope on the foredeck: rings stacked, each a little smaller
    let coil = [0.74, 0.0, -0.13];
    for j in 0..k[1] {
        let r = 0.05 - 0.0025 * (j % 8) as f64;
        let y = deck_y(coil[0], coil[2]) + 0.01 + 0.012 * (j / 8) as f64;
        v.push(u(torus([coil[0], y, coil[2]], r.max(0.02), 0.008, [0.0, 1.0, 0.0]), 0.003));
    }
    // bollards: pairs of short posts by the stem
    for j in 0..k[2] {
        let x = 1.08 - 0.05 * (j / 2) as f64;
        let z = if j % 2 == 0 { 1.0 } else { -1.0 } * (deck_edge(x) - 0.045).max(0.02);
        v.push(u(cyl([x, deck_y(x, z) + 0.03, z], 0.016, 0.06, 0.006, [0.0; 3]), 0.008));
    }
    v
}

fn stern(n: usize, k: &[usize]) -> Vec<Part> {
    let mut v = Vec::new();
    let sx = -1.06;
    let sy = deck_y(sx, 0.0);
    if n >= 32 {
        v.push(u(ball([sx, sy + 0.27, 0.0], 0.03), 0.0)); // the stern lamp (the object's own shape)
        v.push(u(rod([sx, sy - 0.01, 0.0], [sx, sy + 0.25, 0.0], 0.011), 0.01)); // its staff
    }
    if n >= 64 { v.push(u(block([sx - 0.07, sy + 0.18, 0.0], [0.12, 0.08, 0.006], 0.002, [0.0, 0.0, -6.0]), 0.006)); } // the ensign
    // fenders: rods hung on the sides
    for j in 0..k[0] {
        let side = if j % 2 == 0 { 1.0 } else { -1.0 };
        let slots = k[0].div_ceil(2);
        let x = 0.55 - 1.2 * ((j / 2) as f64 + 0.5) / slots as f64;
        let p = on_hull(x, 0.2, side);
        let nrm = hull_n(p);
        let c = add(p, mul(nrm, 0.024));
        v.push(u(rod(add(c, [0.0, 0.04, 0.0]), add(c, [0.0, -0.05, 0.0]), 0.022), 0.0));
    }
    // studs round the stern's rim
    let o = outline(0.012, 1.14);
    for j in 0..k[1] {
        let t = 0.5 + 0.36 * ((j as f64 + 0.5) / k[1] as f64 - 0.5);
        let q = along(&o, t);
        v.push(u(ball([q[0], deck_y(q[0], q[1]) + 0.006, q[1]], 0.009), 0.004));
    }
    // a dinghy turned over on the after deck, on chocks
    let dk = k[2];
    let dc = [-0.86, deck_y(-0.86, 0.0) + 0.07, 0.0];
    // (the chocks first, so that a dinghy of few parts never floats)
    let dparts = [
        rod(add(dc, [-0.06, -0.06, -0.05]), add(dc, [-0.06, -0.06, 0.05]), 0.008),
        rod(add(dc, [0.06, -0.06, -0.05]), add(dc, [0.06, -0.06, 0.05]), 0.008),
        ell(add(dc, [0.0, -0.006, 0.0]), [0.12, 0.05, 0.06], [0.0; 3]),
        ell(add(dc, [0.07, -0.001, 0.0]), [0.08, 0.045, 0.045], [0.0; 3]),
        cut(ell(add(dc, [0.0, -0.02, 0.0]), [0.105, 0.034, 0.048], [0.0; 3]), 0.005),
        rod(add(dc, [-0.12, -0.006, 0.0]), add(dc, [0.13, -0.006, 0.0]), 0.006),
    ];
    for p in dparts.into_iter().take(dk) { let s = if p.op == Op::Subtract { 0.005 } else { 0.012 }; v.push(op(p.clone(), p.op, s)); }
    v
}

fn details(n: usize, k: &[usize]) -> Vec<Part> {
    let mut v = Vec::new();
    let rivet = |x: f64, phi: f64, side: f64, r: f64| { let p = on_hull(x, phi, side); u(ball(add(p, mul(hull_n(p), r * 0.25)), r), 0.003) };
    if n >= 64 {
        // the first rivets, a pair (the first is the object's own shape: a ball)
        for side in [1.0, -1.0] { v.push(rivet(0.0, 0.3, side, 0.011)); }
    }
    // rivets: rows along the hull's sides, as many rows as keep them about square, a row and its mirror alike
    let r = k[0] / 2;
    if r > 0 {
        let rows = ((r as f64 / 24.0).sqrt().round() as usize).clamp(1, 24);
        for (i, m) in spread(r, rows).into_iter().enumerate() {
            // (clear of the portholes' band, phi 0.42..0.58)
            let phi = 0.3 + 0.69 * (i as f64 + 0.5) / rows as f64;
            let phi = if phi > 0.42 { phi + 0.16 } else { phi };
            let (xa, xb) = (-0.95 + 0.25 * phi, 1.05 - 0.35 * phi);
            let spacing = (xb - xa) / m.max(1) as f64;
            let rr = (spacing * 0.3).clamp(0.004, 0.011);
            for side in [1.0, -1.0] {
                for j in 0..m { v.push(rivet(xa + (xb - xa) * (j as f64 + 0.5) / m as f64, phi, side, rr)); }
            }
        }
    }
    // keel bolts: heads along the keel's bottom edge, on the centreline
    for j in 0..k[1] {
        let x = -0.62 + 1.2 * (j as f64 + 0.5) / k[1] as f64;
        let p = keel_bottom(x);
        v.push(u(ball(add(p, [0.0, 0.004, 0.0]), 0.012), 0.004));
    }
    v
}

pub static MODULES: [Module; 8] = [
    Module { name: "hull", story: "Hull: five ellipsoids smooth-joined along a curved keel line (round stern, raked pointed bow), a narrow fin keel filleted into the hull, the rudder hung on the stern, ribs (tori as wide as the section; boatyNs: stretched to it) and strakes (capsule chains) on the sides, the sheer cut from above by a large ball (the deck curves up to bow and stern), portholes cut with rings round them, the propeller (hub, blades at a pitch, ring, shaft).",
        streams: &[("ribs", 3.0, 16, 1), ("strakes", 2.0, 160, 2), ("portholes", 1.5, 60, 4)], build: hull },
    Module { name: "deck", story: "Deck: the fore hatch, planks across the deck in rows of segments that follow its curve, cleats along the edges, nail heads in rows.",
        streams: &[("planks", 3.0, 240, 1), ("cleats", 0.5, 16, 2), ("studs", 1.0, 400, 1)], build: deck },
    Module { name: "cabin", story: "Cabin: an oval deckhouse with an ellipsoid roof, windows cut through it with ring frames, front windows and a door cut, a life-ring on its after wall, a raked funnel with bands, a roof lamp, beads round the roof.",
        streams: &[("windows", 1.5, 9, 3), ("roof beads", 1.0, 80, 1)], build: cabin },
    Module { name: "rigging", story: "Rigging: the mast, the boom on the centreline held up by a topping lift from the masthead, a flag at the masthead and a masthead ball, a crow's nest ring, stays as capsule chains (the forestay to an eye on the stem head, shrouds and backstays to the deck, by pairs), mast hoops, bunting under the forestay.",
        streams: &[("stays", 5.0, 200, 1), ("hoops", 0.4, 10, 1), ("bunting", 1.0, 80, 1)], build: rigging },
    Module { name: "rails", story: "Rails: posts round the deck's edge from bow to stern and back, a top rail and a middle rail (capsule chains), studs along the rail.",
        streams: &[("posts", 2.0, 120, 2), ("top rail", 1.5, 160, 1), ("middle rail", 1.0, 160, 1), ("rail studs", 0.5, 240, 1)], build: rails },
    Module { name: "bow", story: "Bow: the capstan, the anchor hung on the starboard bow (shank, ring, stock, arms, flukes), its chain of linked rings up over the bow to the capstan, a coil of rope, bollards.",
        streams: &[("chain links", 1.5, 60, 1), ("rope coil", 1.0, 32, 1), ("bollards", 0.5, 8, 2)], build: bow },
    Module { name: "stern", story: "Stern: the stern lamp on its staff, the ensign, fenders hung on the sides, studs round the stern, a dinghy turned over on the after deck.",
        streams: &[("fenders", 1.0, 16, 2), ("stern studs", 0.5, 120, 1), ("dinghy", 1.0, 6, 1)], build: stern },
    Module { name: "details", story: "Details: rivets in rows along the hull's sides, a row and its mirror alike (what N past the other modules' detail goes into), bolt heads along the keel.",
        streams: &[("rivets", 2.0, usize::MAX, 2), ("keel bolts", 0.25, 40, 1)], build: details },
];

/// The boat of N parts: each module's parts (module order is tree order), and per module the stream counts.
pub fn build(n: usize) -> Result<(Vec<Vec<Part>>, Vec<Vec<usize>>), String> {
    let cores: Vec<usize> = MODULES.iter().map(|m| (m.build)(n, &vec![0; m.streams.len()]).len()).collect();
    let total_core: usize = cores.iter().sum();
    if total_core > n { return Err(format!("boaty{n}: {total_core} parts before any detail")); }
    // detail shared out stream by stream (the largest weight per part given so far takes the next), each
    // stream within its most, each module within 64 parts when the app holds it as objects
    let multi = n > 64 && n <= 512;
    let mut counts: Vec<Vec<usize>> = MODULES.iter().map(|m| vec![0; m.streams.len()]).collect();
    let mut mtot = cores.clone();
    let mut left = n - total_core;
    while left > 0 {
        let mut best: Option<(usize, usize, f64)> = None;
        for (mi, m) in MODULES.iter().enumerate() {
            for (si, s) in m.streams.iter().enumerate() {
                let unit = s.3;
                if unit > left || counts[mi][si] + unit > s.2 || (multi && mtot[mi] + unit > 64) { continue; }
                let key = s.1 / ((counts[mi][si] / unit) as f64 + 1.0);
                if best.is_none_or(|b| key > b.2) { best = Some((mi, si, key)); }
            }
        }
        // (a last odd part, when every stream that takes one at a time is at its most: the keel bolts take it)
        let best = best.or_else(|| (left == 1 && !(multi && mtot[7] >= 64)).then_some((7, 1, 0.0)));
        let (mi, si, _) = best.ok_or_else(|| format!("boaty{n}: no room left for detail ({left} left; modules {mtot:?}, streams {counts:?})"))?;
        let unit = MODULES[mi].streams[si].3;
        counts[mi][si] += unit;
        mtot[mi] += unit;
        left -= unit;
    }
    let mods: Vec<Vec<Part>> = MODULES.iter().zip(&counts).map(|(m, c)| (m.build)(n, c)).collect();
    let got: usize = mods.iter().map(Vec::len).sum();
    if got != n { return Err(format!("boaty{n}: made {got} parts (counts {counts:?})")); }
    Ok((mods, counts))
}

/// The story of boatyN: each module, what it is, and how many parts it and each of its streams have at N.
pub fn story(n: usize, stretched: bool) -> Vec<String> {
    let Ok((mods, counts)) = build_with(n, stretched) else { return Vec::new() };
    MODULES.iter().zip(mods.iter().zip(&counts)).map(|(m, (ps, c))| {
        let streams: Vec<String> = m.streams.iter().zip(c).filter(|(_, k)| **k > 0).map(|(s, k)| format!("{} {k}", s.0)).collect();
        format!("{} ({} parts{}{}) {}", m.name, ps.len(), if streams.is_empty() { "" } else { "; detail: " }, streams.join(", "), m.story)
    }).collect()
}


// ---- the alignment audit: floating, hidden and unmirrored parts ----

/// A part's own distance (its own units, before its size), from its sliders: the studio and solids shapes
/// the boat uses, as closed forms (an ellipsoid and an oval by their usual estimates).
fn own_dist(p: &Part, q: V) -> f64 {
    let v = |k: &str, d: f64| p.values.get(k).copied().unwrap_or(d);
    let rbox2 = |a: f64, b: f64, r: f64| { let (x, y) = ((a + r).max(0.0), (b + r).max(0.0)); (x * x + y * y).sqrt() + (a + r).max(b + r).min(0.0) - r };
    let ell = |q: V, r: V| { let k0 = len([q[0] / r[0], q[1] / r[1], q[2] / r[2]]); let k1 = len([q[0] / (r[0] * r[0]), q[1] / (r[1] * r[1]), q[2] / (r[2] * r[2])]); k0 * (k0 - 1.0) / k1.max(1e-9) };
    match (p.family.as_str(), p.preset.as_str()) {
        ("studio", "sphere") => len(q) - v("radius", 0.95),
        ("studio", "ellipsoid") => ell(q, [v("rx", 0.95), v("ry", 0.6), v("rz", 0.75)]),
        ("studio", "block") => {
            let r = v("round", 0.07);
            let h = [0.5 * v("width", 1.3) - r, 0.5 * v("height", 0.7) - r, 0.5 * v("depth", 1.0) - r];
            let d = [q[0].abs() - h[0], q[1].abs() - h[1], q[2].abs() - h[2]];
            len([d[0].max(0.0), d[1].max(0.0), d[2].max(0.0)]) + d[0].max(d[1]).max(d[2]).min(0.0) - r
        }
        ("studio", "cylinder") | ("studio", "disc") => rbox2((q[0] * q[0] + q[2] * q[2]).sqrt() - v("radius", 0.5), q[1].abs() - 0.5 * v("height", 1.6), 0.0),
        ("studio", "oval") => { let (a, c) = (v("rx", 0.95), v("rz", 0.55)); let e = (len([q[0] / a, 0.0, q[2] / c]) - 1.0) * a.min(c); rbox2(e, q[1].abs() - 0.5 * v("height", 0.5), 0.0) }
        ("studio", "capsule") => { let r = v("radius", 0.38); let h = (0.5 * v("length", 2.0) - r).max(0.0); len([q[0] - q[0].clamp(-h, h), q[1], q[2]]) - r }
        ("solids", "ring") => { let fat = v("fat", 0.42); let big = 1.22 / (1.0 + fat); len([(q[0] * q[0] + q[2] * q[2]).sqrt() - big, q[1], 0.0]) - fat * big }
        _ => len(q) - 1.0,
    }
}
/// A part's distance at a point of the scene.
pub fn part_dist(p: &Part, x: V) -> f64 {
    let m = gf_core::math::euler(p.place.rot[0], p.place.rot[1], p.place.rot[2]);
    let d = sub(x, p.place.pos);
    let s = p.place.size;
    let mut q = [0, 1, 2].map(|i| (m[i] * d[0] + m[3 + i] * d[1] + m[6 + i] * d[2]) / s);
    let mut k = 1.0;
    if let Some(st) = p.stretch { q = [q[0] / st[0], q[1] / st[1], q[2] / st[2]]; k = st[0].min(st[1]).min(st[2]); }
    own_dist(p, q) * s * k
}
fn reach_of(p: &Part) -> f64 { p.place.size * 1.4 * p.stretch.map_or(1.0, |s| s[0].max(s[1]).max(s[2])) }

/// One finding of the audit.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Finding { pub part: usize, pub module: &'static str, pub shape: String, pub at: [f64; 3], pub gap: f64, pub by_design: bool }
/// The audit of a boat: its parts, the ones touching nothing (floating), the ones wholly inside others
/// (hidden: they draw nothing), and the ones off the centreline with no mirror image (unmirrored).
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Audit { pub parts: usize, pub floating: Vec<Finding>, pub hidden: Vec<Finding>, pub unmirrored: Vec<Finding>, pub tolerance: f64 }

/// What is one-sided on purpose: the anchor, its chain and the rope coil (bow), the life-ring (cabin), the
/// propeller's blades.
fn one_sided(module: &str, p: &Part) -> bool {
    // (and the propeller's blades: alike by turning, not by mirroring)
    match module { "bow" => p.preset != "cylinder", "cabin" => p.family == "solids" && p.place.pos[0] < CAB_X - CAB_RX * 0.8, "hull" => len(sub(p.place.pos, PROP)) < 0.1, _ => false }
}

/// Audit the boat (`mods`: its modules' parts, in tree order). A part's surface is sampled (points of a grid
/// in its box walked onto its surface), and each sample is read against the rest of the boat near it (the
/// same parts in the same order, with their seams' fillets): the least reading is the
/// part's gap to the rest (floating past `tol`), every reading under -tol means it is hidden.
pub fn audit(mods: &[Vec<Part>], tol: f64) -> Audit {
    let all: Vec<(&'static str, &Part)> = MODULES.iter().zip(mods).flat_map(|(m, ps)| ps.iter().map(move |p| (m.name, p))).collect();
    let mut a = Audit { parts: all.len(), tolerance: tol, ..Default::default() };
    let shape = |p: &Part| format!("{}/{}", p.family, p.preset);
    for (i, &(module, p)) in all.iter().enumerate() {
        // mirror image (anything off the centreline)
        if p.place.pos[2].abs() > 0.004 {
            let mp = [p.place.pos[0], p.place.pos[1], -p.place.pos[2]];
            let found = all.iter().any(|(_, o)| o.family == p.family && o.preset == p.preset && (o.place.size - p.place.size).abs() < 1e-6 * p.place.size.max(1e-3) + 1e-9
                && len(sub(o.place.pos, mp)) < 0.004 && p.values.iter().all(|(k, v)| o.values.get(k).is_some_and(|w| (w - v).abs() < 1e-6)));
            if !found { a.unmirrored.push(Finding { part: i, module, shape: shape(p), at: p.place.pos, gap: 0.0, by_design: one_sided(module, p) }); }
        }
        if p.op != Op::Union || all.len() < 2 { continue; }
        let near: Vec<&Part> = all.iter().enumerate().filter(|(j, (_, o))| *j != i && len(sub(o.place.pos, p.place.pos)) < reach_of(o) + reach_of(p) + tol).map(|(_, (_, o))| *o).collect();
        let field = |x: V| {
            let mut f: f64 = 1e9;
            for o in &near {
                let d = part_dist(o, x);
                let k = o.smooth.max(1e-6);
                f = match o.op { Op::Union => smin(f, d, k), Op::Subtract => -smin(-f, d, k), Op::Intersect => -smin(-f, -d, k) };
            }
            f
        };
        // samples on the part's surface
        let r = reach_of(p);
        let g = 7;
        let mut gap = f64::MAX;
        let mut deepest = f64::MIN;
        for ix in 0..g { for iy in 0..g { for iz in 0..g {
            let f = |k: usize| -1.0 + 2.0 * (k as f64 + 0.5) / g as f64;
            let mut x = add(p.place.pos, mul([f(ix), f(iy), f(iz)], r));
            for _ in 0..12 {
                let d = part_dist(p, x);
                if d.abs() < 1e-5 { break; }
                let e = 1e-5;
                let gr = [part_dist(p, add(x, [e, 0.0, 0.0])) - d, part_dist(p, add(x, [0.0, e, 0.0])) - d, part_dist(p, add(x, [0.0, 0.0, e])) - d];
                let gl = len(gr) / e;
                if gl < 1e-9 { break; }
                x = sub(x, mul(gr, d / (e * gl * gl)));
            }
            if part_dist(p, x).abs() > 1e-3 { continue; }
            let fx = field(x);
            gap = gap.min(fx);
            deepest = deepest.max(fx);
        }}}
        if gap == f64::MAX { continue; }
        if gap > tol { a.floating.push(Finding { part: i, module, shape: shape(p), at: p.place.pos, gap, by_design: false }); }
        else if deepest < -tol { a.hidden.push(Finding { part: i, module, shape: shape(p), at: p.place.pos, gap: deepest, by_design: false }); }
    }
    a
}
/// The parameters of a part's shape that are lengths (scaled with its size when it becomes an object's own
/// shape).
pub fn is_length(id: &str) -> bool { !matches!(id, "cut" | "arc" | "turns" | "lean" | "fat" | "section" | "flat") }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_counts() {
        for n in [16, 20, 32, 48, 64, 65, 100, 128, 256, 512, 1024, 2048, 4096, 8192] {
            let (m, _) = build(n).unwrap();
            assert_eq!(m.iter().map(Vec::len).sum::<usize>(), n);
            if n > 64 && n <= 512 {
                for (i, ps) in m.iter().enumerate() {
                    assert!(ps.len() <= 64 && !ps.is_empty(), "{n}: module {i} has {}", ps.len());
                    assert_eq!(ps[0].place.rot, [0.0; 3], "{n}: module {i}'s first part is turned");
                    assert_eq!(ps[0].family, "studio");
                }
            }
        }
    }
    #[test]
    fn frames_round_trip() {

        for (ex, ey) in [([1.0, 0.2, 0.3], [0.0, 1.0, 0.0]), ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]), ([0.3, -0.8, 0.1], [0.2, 0.1, 1.0])] {
            let r = frame(ex, ey);
            let mm = gf_core::math::euler(r[0], r[1], r[2]);
            let ex = norm(ex);
            let col0 = [mm[0], mm[3], mm[6]];
            assert!(len(sub(col0, ex)) < 1e-9, "{ex:?} -> {r:?} -> {col0:?}");
            let ey = norm(sub(ey, mul(ex, dot(ey, ex))));
            let col1 = [mm[1], mm[4], mm[7]];
            assert!(len(sub(col1, ey)) < 1e-9, "{ey:?} -> {r:?} -> {col1:?}");
        }
    }
}
