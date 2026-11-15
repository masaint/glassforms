//! Studio shapes: the plain solids a designer reaches for, with soft edges. Slabs, bars, blocks, rings,
//! a coiled band, tubes, discs, a frame, an arch, lenses, balls, cones and wedges.
//!
//! What matters in these is not the formula but the finish. Seen through frosted or polished glass a
//! face that is not quite flat shows at once, and an edge reads by its highlight: a flat cut along it is
//! a line of even width, a round is a line that fades. So every shape here gives the true distance to
//! its surface, inside and out (a face is then flat to the last digit), and every edge has two sliders:
//!
//!   bevel   a flat cut across the edge at 45 degrees, taking this much off each of the two faces
//!   round   a radius. Every edge is rounded by it, the edges of the flat included
//!
//! The round is made the only way that gives true arcs: the solid is first drawn in by r all over (with
//! its flats cut), and then everything within r of that smaller solid is glass. That needs the exact
//! distance to the smaller solid from outside it, corners and all, which is what the pieces below give.
//! (Three places are true to a few hundredths of a voxel rather than to the last digit: the flats round
//! the ends of the arch and of the coil, where a curved wall meets a flat end, and the coil's walls,
//! which are measured along the radius. And where the flats of an end meet each other, on the wedge,
//! the arch and the coil, the round leaves a fine crease.)
//!
//! Sizes are in the volume's own units (the volume is 3 across), not proportions: two shapes with the
//! same numbers are the same size, which is what stacking them as layers needs. A shape whose sliders
//! would take it out of the volume is shrunk as a whole to fit. Each shape sits in the middle of the
//! volume, the right way up for standing on a floor: its lowest face is flat and level, half its height
//! below the middle. Rings lie flat, bars lie along x, the frame and the arch stand facing z.
//!
//! One thing to know before adding a shape: what a point far from the glass is told must also be a
//! distance, or at least fall steadily towards the glass. If the space past an arch's end were measured
//! against the whole ring, it would read as a valley in mid-air; the walk that builds the volume stalls
//! in such a valley and leaves specks of false surface there. Hence the ends measured against themselves.
use super::*;

/// The pieces every shape is put together from.
const PIECES: &str = r#"
const RT2: f32 = 0.70710678;
const TAU: f32 = 6.2831853;
// the furthest from the middle of the volume a shape may reach (the inflate slider adds up to 0.06)
const FIT: f32 = 1.27;

// Two faces that meet at a right angle, their edge cut by a flat that takes c off each face.
// a, b: the distances from the two faces, negative on the glass side of each. The true distance.
fn edge2(a: f32, b: f32, c: f32) -> f32 {
  let x = max(a, b);
  let y = min(a, b);
  let s = (x + y + c) * RT2;                              // from the flat
  if (x <= 0.0 && s <= 0.0) { return max(x, s); }
  // outside: the nearest of the nearer face (which ends where the flat begins) and the flat itself
  let on_face = length(vec2f(x, max(y + c, 0.0)));
  let t = clamp((x - y) * RT2, 0.0, c * RT2);             // along the flat, from its middle
  let on_flat = length(vec2f(x, y) - (vec2f(-0.5 * c) + t * vec2f(RT2, -RT2)));
  return min(on_face, on_flat);
}

// Three faces that meet at right angles, each of their three edges cut by a flat that takes c off both
// its faces. q: the distances from the three faces. The true distance: the three flats meet in a point,
// and the nearest thing to a point outside may be a face, a flat, an edge between them or that point.
fn edge3(q: vec3f, c: f32) -> f32 {
  // by symmetry only the case x >= y >= z need be worked out
  let x = max(q.x, max(q.y, q.z));
  let z = min(q.x, min(q.y, q.z));
  let y = q.x + q.y + q.z - x - z;
  let s = (x + y + c) * RT2;                              // from the flat between the two nearest faces
  if (x <= 0.0 && s <= 0.0) { return max(x, s); }
  // the nearest face: it ends c short of each of its two edges
  let on_face = length(vec3f(x, max(y + c, 0.0), max(z + c, 0.0)));
  // that flat, seen square on: a strip of half-width c / sqrt 2 whose end is cut to a point by the two
  // other flats. t runs across the strip from its middle line, z along it.
  let t = (x - y) * RT2;
  let w = c * RT2;
  var d = 0.0;
  if (t > w || z + t * RT2 + 0.5 * c > 0.0) {
    let ab = vec2f(w, -0.5 * c);                          // one side of the point, from its tip
    let pa = vec2f(t, z + 0.5 * c);
    let slant = length(pa - ab * clamp(dot(pa, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0));
    let straight = length(vec2f(t - w, max(z + c, 0.0)));
    d = min(slant, straight);
  }
  return min(on_face, length(vec2f(s, d)));
}

// the distance from a point to the straight piece from a to b
fn seg(q: vec2f, a: vec2f, b: vec2f) -> f32 {
  let pa = q - a;
  let ba = b - a;
  return length(pa - ba * clamp(dot(pa, ba) / max(dot(ba, ba), 1e-12), 0.0, 1.0));
}
// how far a point is to the right of the line through a and then b (the outside, for an outline that is
// walked anticlockwise). A piece with no length says nothing.
fn off(q: vec2f, a: vec2f, b: vec2f) -> f32 {
  let ba = b - a;
  let l = length(ba);
  if (l < 1e-6) { return -1e9; }
  return (ba.y * (q.x - a.x) - ba.x * (q.y - a.y)) / l;
}
// a rectangle with round corners: half its sides, and the corners' radius
fn rrect(p: vec2f, half: vec2f, k: f32) -> f32 {
  let o = abs(p) - (half - k);
  return length(max(o, vec2f(0.0))) + min(max(o.x, o.y), 0.0) - k;
}
"#;

/// For the shapes that have edges: how much round and flat an edge can take.
const TRIMS: &str = r#"
// The round and the flat as they can be had where the thinnest part an edge bounds is 2 m thick: the
// round first (it cannot be more than half that thickness), then the flat out of what is left.
fn trims(m: f32) -> vec2f {
  let r = min(pr.round, 0.98 * m);
  return vec2f(r, min(pr.bevel, 0.98 * (m - r)));
}
"#;

/// What a shape's own `reach()` (how far it reaches from the middle) and `body(p)` (its distance) are
/// wrapped in.
const TAIL: &str = r#"
fn raw(p: vec3f) -> f32 {
  // A quarter of a voxel off the grid in two directions. A plane that halves a square edge runs at 45
  // degrees, and where it passes exactly through the centres of voxels (a square bar's does) the walk
  // to the surface misjudges every one of them; a quarter voxel is not seen and keeps them apart.
  let h = 2.0 * u.hb / u.res;
  let q = p + vec3f(0.0, 0.25, -0.25) * h;
  // held inside the volume, whatever the sliders say
  let s = min(1.0, FIT / reach());
  return body(q / s) * s;
}
"#;

fn shape(with_edges: bool, body: &str) -> Shape {
    Shape::Field {
        wgsl: format!("{PIECES}{}{body}{TAIL}", if with_edges { TRIMS } else { "" }),
        // each is its own boundary: nothing may shave or round it further
        clip: "-1.0".to_string(),
        shell: crate::field::SHELL_DEFAULT.to_string(),
    }
}

fn size(id: &'static str, label: &'static str, value: f64, lo: f64, hi: f64) -> ParamDef {
    ParamDef::labelled(id, label, value, lo, hi, 0.01)
}
/// The two sliders every edge has.
fn edges(bevel: f64, round: f64) -> [ParamDef; 2] {
    [ParamDef::labelled("bevel", "bevel", bevel, 0.0, 0.15, 0.005), ParamDef::labelled("round", "round", round, 0.0, 0.2, 0.005)]
}
fn with(mut sizes: Vec<ParamDef>, bevel: f64, round: f64) -> Vec<ParamDef> {
    sizes.extend(edges(bevel, round));
    sizes
}

const EDGE_EQ: &str = "edges: a flat that takes b off each face, then a round of radius r";

/// A block with all twelve edges alike. `across`: what its size along x is called.
fn boxed(across: &str) -> String {
    format!(r#"
fn reach() -> f32 {{ return 0.5 * length(vec3f(pr.{across}, pr.height, pr.depth)); }}
fn body(p: vec3f) -> f32 {{
  let half = 0.5 * vec3f(pr.{across}, pr.height, pr.depth);
  let t = trims(min(half.x, min(half.y, half.z)));
  return edge3(abs(p) - (half - t.x), t.y) - t.x;
}}
"#)
}

/// A ring whose section is a rectangle. `plane`, `along`: the two coordinates across the axis, and the
/// one along it; `long`: what its size along the axis is called.
fn ring(plane: &str, along: &str, long: &str) -> String {
    format!(r#"
fn reach() -> f32 {{ return length(vec2f(pr.outer, 0.5 * pr.{long})); }}
fn body(p: vec3f) -> f32 {{
  let hole = min(pr.hole, pr.outer - 0.02);               // a wall is at least 0.02 thick
  let half = 0.5 * vec2f(pr.outer - hole, pr.{long});
  let t = trims(min(half.x, half.y));
  let a = abs(vec2f(length(p.{plane}) - 0.5 * (pr.outer + hole), p.{along})) - (half - t.x);
  return edge2(a.x, a.y, t.y) - t.x;
}}
"#)
}

const SLAB: &str = r#"
fn reach() -> f32 { return 0.5 * length(vec3f(pr.width, pr.height, pr.depth)); }
fn body(p: vec3f) -> f32 {
  let half = 0.5 * vec3f(pr.width, pr.height, pr.depth);
  let t = trims(min(half.x, min(half.y, half.z)));
  // Seen from above, a corner can be no sharper than the flat and round its edge carries (the flat has
  // to go round it), and no rounder than half the shorter side.
  let k = clamp(pr.corner, t.x + t.y, min(half.x, half.z));
  // the outline seen from above and the two level faces, all drawn in by r; then r is put back all over
  return edge2(rrect(p.xz, half.xz - t.x, k - t.x), abs(p.y) - (half.y - t.x), t.y) - t.x;
}
"#;

const DISC: &str = r#"
fn reach() -> f32 { return length(vec2f(pr.radius, 0.5 * pr.height)); }
fn body(p: vec3f) -> f32 {
  let half = vec2f(pr.radius, 0.5 * pr.height);
  let t = trims(min(half.x, half.y));
  let a = vec2f(length(p.xz), abs(p.y)) - (half - t.x);
  return edge2(a.x, a.y, t.y) - t.x;
}
"#;

const FRAME: &str = r#"
fn reach() -> f32 { return 0.5 * length(vec3f(pr.width, pr.height, pr.depth)); }
fn body(p: vec3f) -> f32 {
  let half = 0.5 * vec2f(pr.width, pr.height);
  let rim = min(pr.rim, min(half.x, half.y) - 0.02);      // an opening is always left
  let deep = 0.5 * pr.depth;
  let t = trims(min(0.5 * rim, deep));
  let k = clamp(pr.corner, t.x + t.y, min(half.x, half.y));
  // the rim seen from the front: inside the outer outline and outside the opening, whose corners are
  // rounder or sharper by the rim's width. (Drawn in by r: the outline smaller, the opening larger.)
  let outer = rrect(p.xy, half - t.x, k - t.x);
  let opening = rrect(p.xy, half - rim + t.x, max(k - rim, 0.0) + t.x);
  return edge2(max(outer, -opening), abs(p.z) - (deep - t.x), t.y) - t.x;
}
"#;

const ARCH: &str = r#"
// the section can be no wider than leaves a hole down the middle
fn wide() -> f32 { return min(pr.width, 2.0 * pr.radius - 0.04); }
// how high above and below the ring's centre the arch reaches: its crown, and its lowest corner
fn span() -> vec2f {
  let ca = cos(radians(0.5 * pr.arc));
  return vec2f(pr.radius + 0.5 * wide(), ca * pr.radius - abs(ca) * 0.5 * wide());
}
fn reach() -> f32 {
  let v = span();
  let across = select(v.x * sin(radians(0.5 * pr.arc)), v.x, pr.arc >= 180.0);
  return length(vec3f(across, 0.5 * (v.x - v.y), 0.5 * pr.depth));
}
fn body(p0: vec3f) -> f32 {
  let v = span();
  // both halves alike, and the whole moved so that crown and feet are as far above the middle as below
  let p = vec3f(abs(p0.x), p0.y + 0.5 * (v.x + v.y), p0.z);
  let half = 0.5 * vec2f(wide(), pr.depth);
  let t = trims(min(half.x, half.y));
  let rho = length(p.xy);
  // how far round from the end the point is, as an angle: negative on the arch, positive past its end
  let over = select(atan2(p.x, p.y) - radians(0.5 * pr.arc), -3.1415927, pr.arc >= 359.9);
  // On the arch the point is measured against the ring where it stands. Past the end it is measured
  // against the end itself, a flat face through the ring's axis: its place on that face, and how far
  // in front of it. (Measured against the ring there, the space beyond the end would read as a channel
  // leading nowhere.)
  let on = max(over, 0.0);
  let a = abs(vec2f(rho * cos(on) - pr.radius, p.z)) - (half - t.x);
  let past = rho * sin(clamp(over, -1.5707963, 3.1415927));
  return edge2(edge2(a.x, a.y, t.y), past + t.x, t.y) - t.x;
}
"#;

const CAPSULE: &str = r#"
fn reach() -> f32 { return max(0.5 * pr.length, pr.radius); }
fn body(p: vec3f) -> f32 {
  // everything within the radius of a straight line: the line is what is left of the length
  return length(vec3f(max(abs(p.x) - max(0.5 * pr.length - pr.radius, 0.0), 0.0), p.y, p.z)) - pr.radius;
}
"#;

const BALL: &str = r#"
fn reach() -> f32 { return pr.radius; }
fn body(p0: vec3f) -> f32 {
  let big = pr.radius;
  let base = -big + 2.0 * big * pr.cut;                   // the flat base's height above the ball's centre
  let p = vec3f(p0.x, p0.y + 0.5 * (big + base), p0.z);   // base and top as far below the middle as above
  let r = min(pr.round, 0.45 * (big - base));
  // drawn in by r: a smaller ball and a higher base, which meet in a rim of radius w
  let rad = big - r;
  let ground = base + r;
  let w = sqrt(max(rad * rad - ground * ground, 0.0));
  let rim = acos(clamp(ground / rad, -1.0, 1.0));          // the rim's angle down from the top
  let c = min(pr.bevel, min(0.9 * w, 0.5 * rad * rim));
  // the flat runs from c along the ball's surface to c along the base
  let up = rim - c / rad;
  let p1 = rad * vec2f(sin(up), cos(up));
  let p2 = vec2f(w - c, ground);
  let q = vec2f(length(p.xz), p.y);
  // the nearest of the ball's surface (down to the flat), the flat, and the base
  let on_ball = select(length(q - p1), abs(length(q) - rad), q.x * cos(up) - q.y * sin(up) <= 0.0);
  let d = min(on_ball, min(seg(q, p2, p1), length(vec2f(max(q.x - p2.x, 0.0), q.y - ground))));
  let outside = max(max(length(q) - rad, ground - q.y), off(q, p2, p1));
  return select(d, -d, outside <= 0.0) - r;
}
"#;

const LENS: &str = r#"
// A face of the lens: a cap of a sphere that rises `sag` above the rim, or flat. Heights are measured
// from the middle of the edge towards this face. Given back: the height of the sphere's centre, its
// radius drawn in by r (nothing for a flat face), and the height at which the face meets the edge.
fn cap(sag: f32, wide: f32, thick: f32, r: f32) -> vec3f {
  if (sag < 0.004) { return vec3f(0.0, 0.0, 0.5 * thick - r); }
  let big = (wide * wide + sag * sag) / (2.0 * sag);
  let centre = 0.5 * thick + sag - big;
  let rad = big - r;
  return vec3f(centre, rad, centre + sqrt(max(rad * rad - (wide - r) * (wide - r), 0.0)));
}
// The distance to that face and to the flat between it and the edge (x), and how far outside them the
// point is (y). inner: the edge's radius drawn in by r.
fn capped(q: vec2f, f: vec3f, inner: f32, c: f32) -> vec2f {
  let low = vec2f(inner, f.z - c);                        // where the flat leaves the edge
  if (f.y == 0.0) {
    let high = vec2f(inner - c, f.z);
    let d = min(seg(q, low, high), length(vec2f(max(q.x - high.x, 0.0), q.y - f.z)));
    return vec2f(d, max(q.y - f.z, off(q, low, high)));
  }
  let v = q - vec2f(0.0, f.x);
  let up = asin(clamp(inner / f.y, 0.0, 1.0)) - c / f.y;  // where the flat meets the cap: its angle from the axis
  let high = vec2f(0.0, f.x) + f.y * vec2f(sin(up), cos(up));
  let on_cap = select(length(q - high), abs(length(v) - f.y), v.x * cos(up) - v.y * sin(up) <= 0.0);
  return vec2f(min(on_cap, seg(q, low, high)), max(length(v) - f.y, off(q, low, high)));
}
// neither face can rise more than a half ball would
fn sags() -> vec2f { return min(vec2f(pr.front, pr.back), vec2f(0.98 * pr.radius)); }
fn reach() -> f32 { let s = sags(); return length(vec2f(pr.radius, 0.5 * (pr.edge + s.x + s.y))); }
fn body(p0: vec3f) -> f32 {
  let s = sags();
  let p = vec3f(p0.x, p0.y + 0.5 * (s.x - s.y), p0.z);    // the two crowns as far above the middle as below
  let r = min(pr.round, min(0.49 * pr.edge, 0.9 * pr.radius));
  let inner = pr.radius - r;
  let top = cap(s.x, pr.radius, pr.edge, r);
  let bottom = cap(s.y, pr.radius, pr.edge, r);
  // the two flats share what is left of the edge
  let c = min(pr.bevel, min(0.45 * (top.z + bottom.z), 0.5 * inner));
  let q = vec2f(length(p.xz), p.y);
  let a = capped(q, top, inner, c);
  let b = capped(vec2f(q.x, -q.y), bottom, inner, c);
  let d = min(min(a.x, b.x), seg(q, vec2f(inner, c - bottom.z), vec2f(inner, top.z - c)));
  let outside = max(max(a.y, b.y), q.x - inner);
  return select(d, -d, outside <= 0.0) - r;
}
"#;

const CONE: &str = r#"
fn reach() -> f32 { return length(vec2f(max(pr.radius, pr.top), 0.5 * pr.height)); }
fn body(p: vec3f) -> f32 {
  let high = 0.5 * pr.height;
  let q = vec2f(length(p.xz), p.y);
  let dir = normalize(vec2f(pr.top - pr.radius, 2.0 * high));     // up the side
  let n = vec2f(dir.y, -dir.x);                                   // out of the side
  // the round is held to what the base rim and the height can take (tan of half the rim's angle)
  let r = min(pr.round, 0.45 * min(2.0 * high, pr.radius * dir.y / (1.0 - dir.x)));
  // drawn in by r: base, side and top, and where they meet
  let side = dot(n, vec2f(pr.radius, -high)) - r;
  let yb = -high + r;
  let pb = (side - n.y * yb) / n.x;
  var yt = high - r;
  var pt = (side - n.y * yt) / n.x;
  if (pt <= 1e-4) {                                               // no top is left: the side runs to a point
    pt = 0.0;
    yt = side / n.y;
  }
  let b1 = vec2f(pb, yb);
  let t1 = vec2f(pt, yt);
  let c = min(pr.bevel, min(0.45 * pb, 0.3 * distance(b1, t1)));
  let ct = min(c, 0.9 * pt);
  // the outline, anticlockwise: base, a flat, the side, a flat, the top
  let b0 = vec2f(pb - c, yb);
  let b2 = b1 + c * dir;
  let t0 = t1 - ct * dir;
  let t2 = vec2f(pt - ct, yt);
  var d = length(vec2f(max(q.x - b0.x, 0.0), q.y - yb));
  d = min(d, min(seg(q, b0, b2), seg(q, b2, t0)));
  d = min(d, min(seg(q, t0, t2), length(vec2f(max(q.x - t2.x, 0.0), q.y - yt))));
  let outside = max(max(max(yb - q.y, q.y - yt), dot(n, q) - side), max(off(q, b0, b2), off(q, t0, t2)));
  return select(d, -d, outside <= 0.0) - r;
}
"#;

const WEDGE: &str = r#"
fn reach() -> f32 { return 0.5 * length(vec3f(pr.length, pr.height, pr.depth)); }
fn body(p: vec3f) -> f32 {
  // the section, anticlockwise seen along the bar: the two ends of the base, then the ridge
  var v = array<vec2f, 3>(0.5 * vec2f(-pr.depth, -pr.height), 0.5 * vec2f(pr.depth, -pr.height), 0.5 * vec2f(pr.lean * pr.depth, pr.height));
  let l0 = distance(v[0], v[1]);
  let l1 = distance(v[1], v[2]);
  let l2 = distance(v[2], v[0]);
  // the largest circle that fits in the section says how much round it can take
  let fits = pr.depth * pr.height / (l0 + l1 + l2);
  let r = min(pr.round, 0.45 * min(fits, 0.5 * pr.length));
  // drawn in by r the section is the same triangle, smaller about the centre of that circle
  let centre = (v[0] * l1 + v[1] * l2 + v[2] * l0) / (l0 + l1 + l2);
  let shrink = 1.0 - r / fits;
  let c = min(pr.bevel, min(0.3 * shrink * min(l0, min(l1, l2)), 0.98 * (0.5 * pr.length - r)));
  let q = p.zy;
  var d = 1e9;
  var outside = -1e9;
  for (var i = 0; i < 3; i++) {
    let a = centre + (v[i] - centre) * shrink;
    let b = centre + (v[(i + 1) % 3] - centre) * shrink;
    let z = centre + (v[(i + 2) % 3] - centre) * shrink;
    // this side between its two flats, and the flat across the corner it starts from
    let a1 = a + c * normalize(b - a);
    let b0 = b - c * normalize(b - a);
    let a0 = a + c * normalize(z - a);
    d = min(d, min(seg(q, a1, b0), seg(q, a0, a1)));
    outside = max(outside, max(off(q, a1, b0), off(q, a0, a1)));
  }
  // the section's own distance, then the two ends
  let section = select(d, outside, outside <= 0.0);
  return edge2(section, abs(p.x) - (0.5 * pr.length - r), c) - r;
}
"#;

const SPIRAL: &str = r#"
// Turns do not run into each other: where the pitch is less than the band is thick, they are held a
// hair apart instead.
fn pitch() -> f32 { return max(pr.pitch, 1.03 * pr.thick + 0.02); }
fn reach() -> f32 { return length(vec2f(pr.radius + pitch() * pr.turns + 0.5 * pr.thick, 0.5 * pr.height)); }
fn body(p: vec3f) -> f32 {
  let half = 0.5 * vec2f(pr.thick, pr.height);
  let step = pitch();
  let t = trims(min(half.x, half.y));
  let rho = length(p.xz);
  let phi = atan2(p.z, p.x);
  let last = TAU * pr.turns;
  // which turn passes nearest: the one whose radius is the point's, and its neighbour either side
  let near = round((clamp(TAU * (rho - pr.radius) / step, 0.0, last) - phi) / TAU);
  var d = 1e9;
  for (var j = -1; j <= 1; j++) {
    let wound = phi + TAU * (near + f32(j));              // how far along the band, as an angle
    let on = clamp(wound, 0.0, last);
    let mid = pr.radius + step * on / TAU;                // the band's radius there
    let climb = step / (TAU * mid);                       // it winds outwards as it goes round
    // Short of an end the point is measured against the band where it passes this direction. Past an end
    // it is measured against the end itself, a flat face along the radius there: `over` is how far past,
    // as an angle (half a turn or more away it no longer matters which).
    let over = min(abs(wound - on), 3.1415927);
    let a = abs(vec2f((rho * cos(over) - mid) / sqrt(1.0 + climb * climb), p.y)) - (half - t.x);
    var past = rho * sin(over);
    if (over == 0.0) { past = -rho * sin(min(min(wound, last - wound), 1.5707963)); }
    d = min(d, edge2(edge2(a.x, a.y, t.y), past + t.x, t.y));
  }
  return d - t.x;
}
"#;

pub fn family() -> Family {
    let lying = [28.0, -30.0, 0.0];
    let upright = [10.0, -28.0, 0.0];
    Family {
        presets: vec![
            Preset {
                eq: vec!["|y| ≤ h/2,  (x, z) in a rectangle with corners of radius k", EDGE_EQ],
                note: "A flat can be no wider than the corner it has to go round: where the corners are set sharper than the edge allows, they are as sharp as it allows. For a slab with square corners and flats all round, use the block.",
                params: with(vec![size("width", "width", 1.7, 0.4, 2.2), size("depth", "depth", 1.1, 0.3, 1.6), size("height", "thickness", 0.26, 0.04, 0.8), size("corner", "corners", 0.3, 0.0, 0.8)], 0.0, 0.05),
                rot: Some(lying),
                animate: Some(vec!["corner"]),
                ..Preset::new("slab", "Rounded slab", shape(true, SLAB))
            },
            Preset {
                eq: vec!["|x| ≤ l/2,  |y| ≤ h/2,  |z| ≤ d/2", EDGE_EQ],
                note: "All twelve edges alike. Where three flats meet at a corner they meet in a point.",
                params: with(vec![size("length", "length", 2.2, 0.4, 2.4), size("height", "height", 0.34, 0.08, 1.0), size("depth", "depth", 0.34, 0.08, 1.0)], 0.035, 0.01),
                rot: Some(lying),
                animate: Some(vec!["bevel"]),
                ..Preset::new("bar", "Bar", shape(true, &boxed("length")))
            },
            Preset {
                eq: vec!["|x| ≤ w/2,  |y| ≤ h/2,  |z| ≤ d/2", EDGE_EQ],
                note: "A rounded box: something to stand the others on. With the round at half the height it is a pebble.",
                params: with(vec![size("width", "width", 1.3, 0.2, 2.0), size("height", "height", 0.7, 0.1, 1.6), size("depth", "depth", 1.0, 0.2, 1.6)], 0.0, 0.07),
                rot: Some(lying),
                animate: Some(vec!["round"]),
                ..Preset::new("block", "Block", shape(true, &boxed("width")))
            },
            Preset {
                eq: vec!["R₁ ≤ ρ ≤ R₂,  |y| ≤ h/2,   ρ = √(x² + z²)", EDGE_EQ],
                note: "A ring with a rectangular section, lying flat.",
                params: with(vec![size("outer", "outer radius", 1.05, 0.3, 1.2), size("hole", "inner radius", 0.6, 0.05, 1.1), size("height", "height", 0.3, 0.04, 1.2)], 0.0, 0.06),
                rot: Some([34.0, -20.0, 0.0]),
                animate: Some(vec!["hole"]),
                ..Preset::new("washer", "Thick ring", shape(true, &ring("xz", "y", "height")))
            },
            Preset {
                eq: vec!["R₁ ≤ ρ ≤ R₂,  |y| ≤ h/2,   ρ = √(x² + z²)", EDGE_EQ],
                note: "The same ring with a thin wall and a tall section: a bangle.",
                params: with(vec![size("outer", "outer radius", 1.0, 0.3, 1.2), size("hole", "inner radius", 0.86, 0.05, 1.1), size("height", "height", 0.55, 0.04, 1.2)], 0.0, 0.025),
                rot: Some([30.0, -20.0, 0.0]),
                animate: Some(vec!["height"]),
                ..Preset::new("bandring", "Band ring", shape(true, &ring("xz", "y", "height")))
            },
            Preset {
                eq: vec!["ρ = R + p·φ/2π,   0 ≤ φ ≤ 2π·n", "a band w thick and h high along that spiral", EDGE_EQ],
                note: "A strip coiled like a watch spring: with more than one turn it passes itself, a pitch further out. Turns never touch: a pitch less than the band's thickness is taken as just more than it. Both ends are cut square.",
                params: with(vec![size("radius", "inner radius", 0.62, 0.2, 1.0), size("pitch", "pitch", 0.22, 0.05, 0.5), size("turns", "turns", 1.35, 0.5, 3.0), size("thick", "band thickness", 0.11, 0.03, 0.4), size("height", "height", 0.5, 0.05, 1.2)], 0.0, 0.03),
                rot: Some([34.0, -20.0, 0.0]),
                animate: Some(vec!["turns"]),
                ..Preset::new("spiralband", "Spiral band", shape(true, SPIRAL))
            },
            Preset {
                eq: vec!["R₁ ≤ ρ ≤ R₂,  |x| ≤ l/2,   ρ = √(y² + z²)", EDGE_EQ],
                note: "A pipe, lying along x.",
                params: with(vec![size("outer", "outer radius", 0.42, 0.1, 0.9), size("hole", "bore radius", 0.3, 0.03, 0.85), size("length", "length", 2.2, 0.3, 2.4)], 0.0, 0.02),
                rot: Some(lying),
                animate: Some(vec!["hole"]),
                ..Preset::new("tube", "Tube", shape(true, &ring("yz", "x", "length")))
            },
            Preset {
                eq: vec!["ρ ≤ R,  |y| ≤ h/2,   ρ = √(x² + z²)", EDGE_EQ],
                note: "A coin.",
                params: with(vec![size("radius", "radius", 0.95, 0.2, 1.2), size("height", "thickness", 0.16, 0.03, 0.6)], 0.03, 0.01),
                rot: Some([34.0, -20.0, 0.0]),
                animate: Some(vec!["bevel"]),
                ..Preset::new("disc", "Disc", shape(true, DISC))
            },
            Preset {
                eq: vec!["ρ ≤ R,  |y| ≤ h/2,   ρ = √(x² + z²)", EDGE_EQ],
                note: "The disc, tall.",
                params: with(vec![size("radius", "radius", 0.5, 0.1, 1.0), size("height", "height", 1.6, 0.2, 2.4)], 0.0, 0.05),
                rot: Some([18.0, -25.0, 0.0]),
                animate: Some(vec!["round"]),
                ..Preset::new("cylinder", "Cylinder", shape(true, DISC))
            },
            Preset {
                eq: vec!["a rectangle w by h with corners of radius k,", "less the same outline a rim's width t further in;  |z| ≤ d/2", EDGE_EQ],
                note: "Standing upright. The opening's corners are sharper than the outer ones by the rim's width, and square once the rim is wider than the corner's radius.",
                params: with(vec![size("width", "width", 1.5, 0.5, 2.0), size("height", "height", 2.0, 0.5, 2.3), size("rim", "rim", 0.22, 0.04, 0.6), size("depth", "depth", 0.32, 0.04, 1.0), size("corner", "corners", 0.1, 0.0, 0.8)], 0.0, 0.04),
                rot: Some(upright),
                animate: Some(vec!["rim"]),
                ..Preset::new("rectframe", "Frame", shape(true, FRAME))
            },
            Preset {
                eq: vec!["|ρ − R| ≤ w/2,  |z| ≤ d/2,   ρ = √(x² + y²)", "within θ/2 of the crown either way", EDGE_EQ],
                note: "Part of a ring, standing on its two ends: at 180° they are level feet. The ends are flat faces through the ring's axis.",
                params: with(vec![size("radius", "radius", 0.8, 0.3, 1.1), size("width", "section width", 0.32, 0.05, 0.8), size("depth", "depth", 0.4, 0.05, 1.2), ParamDef::labelled("arc", "arc °", 180.0, 30.0, 360.0, 0.5)], 0.0, 0.04),
                rot: Some(upright),
                animate: Some(vec!["arc"]),
                ..Preset::new("arch", "Arch", shape(true, ARCH))
            },
            Preset {
                eq: vec!["everything within r of a line:", "|p − s| ≤ r,   s on the x axis,  |s| ≤ l/2 − r"],
                note: "A rod with round ends, lying along x. It has no edges to cut.",
                params: vec![size("length", "length", 2.0, 0.4, 2.4), size("radius", "radius", 0.38, 0.08, 1.0)],
                rot: Some(lying),
                animate: Some(vec!["length"]),
                ..Preset::new("capsule", "Capsule", shape(false, CAPSULE))
            },
            Preset {
                eq: vec!["ρ ≤ R, between two caps of spheres that rise", "s₁ above and s₂ below an edge e thick", EDGE_EQ],
                note: "Biconvex; take either rise to nothing for a flat face (plano-convex). Lying flat, the front face up.",
                params: with(vec![size("radius", "radius", 0.95, 0.3, 1.2), size("front", "front rise", 0.3, 0.0, 0.6), size("back", "back rise", 0.3, 0.0, 0.6), size("edge", "edge thickness", 0.06, 0.02, 0.5)], 0.0, 0.02),
                rot: Some([30.0, -20.0, 0.0]),
                animate: Some(vec!["front"]),
                ..Preset::new("lens", "Lens", shape(true, LENS))
            },
            Preset {
                eq: vec!["x² + y² + z² ≤ R²,   y ≥ −R + 2R·c", EDGE_EQ],
                note: "A ball, or with a cut a dome on a flat base: 0.5 is half a ball. The flat and the round are for the base's rim.",
                params: with(vec![size("radius", "radius", 0.95, 0.2, 1.25), size("cut", "cut", 0.0, 0.0, 0.85)], 0.0, 0.03),
                rot: Some([20.0, -25.0, 0.0]),
                animate: Some(vec!["cut"]),
                ..Preset::new("sphere", "Sphere", shape(true, BALL))
            },
            Preset {
                eq: vec!["ρ ≤ R + (R₁ − R)·(y + h/2)/h,   |y| ≤ h/2", EDGE_EQ],
                note: "A top radius of nothing gives a point, rounded to a small ball; anything more a cut-off cone, up to one that widens upwards.",
                params: with(vec![size("radius", "base radius", 0.8, 0.2, 1.2), size("top", "top radius", 0.0, 0.0, 1.0), size("height", "height", 1.5, 0.2, 2.3)], 0.0, 0.04),
                rot: Some([18.0, -25.0, 0.0]),
                animate: Some(vec!["top"]),
                ..Preset::new("cone", "Cone", shape(true, CONE))
            },
            Preset {
                eq: vec!["a triangle d wide and h high in (z, y), its ridge", "leaning by a share of d/2;  |x| ≤ l/2", EDGE_EQ],
                note: "A prism bar: with no lean the roof of a house, with all of it a ramp. The round is true on every edge; where the flats of an end meet each other a fine crease is left.",
                params: with(vec![size("length", "length", 2.0, 0.4, 2.4), size("depth", "base width", 0.7, 0.1, 1.4), size("height", "height", 0.6, 0.1, 1.4), size("lean", "lean", 0.0, -1.0, 1.0)], 0.0, 0.03),
                rot: Some(lying),
                animate: Some(vec!["lean"]),
                ..Preset::new("wedge", "Wedge", shape(true, WEDGE))
            },
        ],
        // The distances are true ones, but voxels that lie on a plane halfway between two faces are stored
        // a little too deep, so rays step a little short. Rings, frames and the coil are not convex: a ray
        // that leaves the glass is followed into the next piece of it.
        step_scale: 0.85,
        steps_out: 220,
        steps_in: 64,
        layers: 2,
        defaults: vec![
            ("ior", Setting::Num(1.5)), ("disp", Setting::Num(0.05)), ("absorb", Setting::Vec3([0.16, 0.13, 0.04])),
            ("ns", Setting::Num(1.0)), ("nw", Setting::Num(3.0)), ("nb", Setting::Num(4.0)),
            ("dist", Setting::Num(5.6)), ("fov", Setting::Num(26.0)), ("rot", Setting::Vec3([28.0, -30.0, 0.0])),
            ("spin", Setting::Num(0.25)), ("bloom", Setting::Num(0.3)), ("exposure", Setting::Num(1.0)),
            // while a slider travels the shape is built at 384 voxels a side, not 256: at 256 a small round
            // or flat is under two voxels wide and the edge looks moulded
            ("animRes", Setting::Num(384.0)),
        ],
        // Broad lights that fade from their middles to their edges, so that a flat face takes its tone from
        // where in a light it looks and neighbouring faces differ; cool and pale, as a studio's are. Two
        // small bright ones draw the line along a flat or a round.
        lights: vec![
            Light { az: 40.0, el: 22.0, w: 30.0, h: 42.0, rgb: [3.0, 2.95, 2.9], soft: 1.0 },
            Light { az: -60.0, el: -4.0, w: 26.0, h: 46.0, rgb: [0.7, 1.3, 3.2], soft: 1.0 },
            Light { az: -140.0, el: 6.0, w: 22.0, h: 46.0, rgb: [2.6, 1.5, 3.0], soft: 1.0 },
            Light { az: 150.0, el: 30.0, w: 32.0, h: 24.0, rgb: [2.1, 2.1, 2.2], soft: 1.0 },
            Light { az: 0.0, el: 85.0, w: 40.0, h: 40.0, rgb: [1.0, 1.05, 1.2], soft: 1.0 },
            Light { az: 0.0, el: -80.0, w: 36.0, h: 36.0, rgb: [0.3, 0.3, 0.5], soft: 1.0 },
            Light { az: 95.0, el: 46.0, w: 5.0, h: 5.0, rgb: [7.0, 7.0, 7.0], soft: 0.5 },
            Light { az: -100.0, el: -36.0, w: 4.0, h: 8.0, rgb: [5.0, 5.5, 6.5], soft: 0.5 },
        ],
        ..Family::new("studio", "Studio shapes", Kind::Field)
    }
}
