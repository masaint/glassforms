//! How two readings are joined at a seam: one family, `min(a, b) - k·φ(h)` with `h = max(1 - |a - b|/k, 0)`
//! (and `max(a, b) + k·φ(h)` to intersect; subtracting is intersecting with `-b`), where the profile φ sets
//! the seam's cross-section. Each φ here has `φ(0) = 0` and `0 <= φ' <= 1/2`, so the join's two partial slopes
//! are `1 - φ'` and `φ'`: a weighted mean of the two readings' gradients. Two readings that never overstate a
//! distance (1-Lipschitz) therefore join into one that never does either, whatever the angle between the
//! two surfaces (the sum of the partial slopes is 1; parallel gradients are the worst case).
//!
//! Then the band joins: a boolean held within a band of the first reading's surface (a pocket cut to a
//! depth, a groove along the second's surface, a ridge raised to a height, a bead along the second's
//! surface). Each is a min and max of slope-1 terms, its outer join any profile, so it is 1-Lipschitz too.
//!
//! Written twice: here in f64 (the reference the tests hold) and as shader text (`WGSL`), line for line.

use serde::{Deserialize, Serialize};

use crate::csg::Op;

/// The seam's cross-section.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Profile {
    /// quadratic: φ = h²/4 (the join every tree had before profiles)
    #[default]
    Fillet,
    /// cubic: φ = h³/6, its curvature continuous into both surfaces
    Smooth,
    /// a circular arc of radius k where the two surfaces meet square: φ = (1 + h - √(1 + 2h - h²))/2
    Round,
    /// a flat face, `a + b = k` inside the band: φ = h/2. (The face a right-angle chamfer has, scaled by 1/2
    /// rather than √½, so that it never overstates at any angle; inside the band it reads √½ of the distance.)
    Chamfer,
}
impl Profile {
    pub const ALL: [Profile; 4] = [Profile::Fillet, Profile::Smooth, Profile::Round, Profile::Chamfer];
    pub fn id(self) -> &'static str {
        match self { Profile::Fillet => "fillet", Profile::Smooth => "smooth", Profile::Round => "round", Profile::Chamfer => "chamfer" }
    }
    pub fn from_id(s: &str) -> Option<Profile> {
        Profile::ALL.into_iter().find(|p| p.id() == s)
    }
    /// Its number in a join's record (the shader's switch).
    pub fn code(self) -> f32 {
        match self { Profile::Fillet => 0.0, Profile::Smooth => 1.0, Profile::Round => 2.0, Profile::Chamfer => 3.0 }
    }
    pub fn is_default(&self) -> bool { *self == Profile::Fillet }
}

/// The profile's shape over the band, h from 0 (the band's edge) to 1 (where a = b).
pub fn phi(p: Profile, h: f64) -> f64 {
    match p {
        Profile::Fillet => 0.25 * h * h,
        Profile::Smooth => h * h * h / 6.0,
        Profile::Round => 0.5 * (1.0 + h - (1.0 + h * (2.0 - h)).max(0.0).sqrt()),
        Profile::Chamfer => 0.5 * h,
    }
}

/// How far into the band a pair of readings is: 0 at its edge and beyond, 1 where they are equal. A hard
/// join (k = 0) has no band.
pub fn band_h(a: f64, b: f64, k: f64) -> f64 {
    if k <= 0.0 { return 0.0; }
    (1.0 - (a - b).abs() / k).max(0.0)
}

/// Union with a seam of width k.
pub fn union(a: f64, b: f64, k: f64, p: Profile) -> f64 {
    a.min(b) - k.max(0.0) * phi(p, band_h(a, b, k))
}
/// Intersection with a seam of width k.
pub fn intersect(a: f64, b: f64, k: f64, p: Profile) -> f64 {
    a.max(b) + k.max(0.0) * phi(p, band_h(a, b, k))
}
/// a with b taken out, with a seam of width k.
pub fn subtract(a: f64, b: f64, k: f64, p: Profile) -> f64 {
    intersect(a, -b, k, p)
}
/// a joined by b with the operation and seam.
pub fn join(op: Op, a: f64, b: f64, k: f64, p: Profile) -> f64 {
    match op {
        Op::Union => union(a, b, k, p),
        Op::Subtract => subtract(a, b, k, p),
        Op::Intersect => intersect(a, b, k, p),
    }
}

/// A boolean held within a band of the first reading's surface.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Band {
    /// none: the node joins by its operation
    #[default]
    None,
    /// b cut out of a, but no deeper than `depth` below a's surface
    Pocket,
    /// a channel `width` either side of b's surface, cut no deeper than `depth`
    Groove,
    /// b added to a, but no higher than `depth` above a's surface
    Ridge,
    /// a bead `width` either side of b's surface, raised no higher than `depth`
    Bead,
}
impl Band {
    pub const ALL: [Band; 5] = [Band::None, Band::Pocket, Band::Groove, Band::Ridge, Band::Bead];
    pub fn id(self) -> &'static str {
        match self { Band::None => "none", Band::Pocket => "pocket", Band::Groove => "groove", Band::Ridge => "ridge", Band::Bead => "bead" }
    }
    pub fn code(self) -> f32 {
        match self { Band::None => 0.0, Band::Pocket => 1.0, Band::Groove => 2.0, Band::Ridge => 3.0, Band::Bead => 4.0 }
    }
    pub fn is_none(&self) -> bool { *self == Band::None }
    /// It takes material away (pocket, groove) rather than adding it.
    pub fn cuts(self) -> bool { matches!(self, Band::Pocket | Band::Groove) }
    /// It follows b's surface at a width (groove, bead) rather than b's inside.
    pub fn along(self) -> bool { matches!(self, Band::Groove | Band::Bead) }
    /// The operation a node with this band shows as (the hierarchy's sign, the cut material's row).
    pub fn op(self) -> Option<Op> {
        match self { Band::None => None, Band::Pocket | Band::Groove => Some(Op::Subtract), Band::Ridge | Band::Bead => Some(Op::Union) }
    }
}

/// a held to a band with b: `depth` t below (a cut) or above (an addition) a's surface, `width` w either side
/// of b's surface for a groove or a bead; the outer join with a seam of width k by profile p.
/// - pocket `max(a, min(a + t, -b))`, groove `max(a, min(a + t, w - |b|))`
/// - ridge `min(a, max(a - t, b))`, bead `min(a, max(a - t, |b| - w))`
pub fn band(kind: Band, a: f64, b: f64, t: f64, w: f64, k: f64, p: Profile) -> f64 {
    let (t, w) = (t.max(0.0), w.max(0.0));
    let bb = if kind.along() { b.abs() - w } else { b };
    match kind {
        Band::None => a,
        Band::Pocket | Band::Groove => intersect(a, (a + t).min(-bb), k, p),
        Band::Ridge | Band::Bead => union(a, (a - t).max(bb), k, p),
    }
}

/// The text of the joins (after the tree's `csgJoin`, which the profile family generalises), and of a
/// construction node's join by profile or band (beside `cnJoin`). Put into a tree only when a node uses them.
pub const WGSL: &str = "
// the seam's cross-section over its band (gf_core::join::phi): 0 fillet, 1 smooth, 2 round, 3 chamfer
fn csgPhi(h: f32, s: i32) -> f32 {
  switch s {
    case 1: { return h * h * h * (1.0 / 6.0); }
    case 2: { return 0.5 * (1.0 + h - sqrt(max(1.0 + h * (2.0 - h), 0.0))); }
    case 3: { return 0.5 * h; }
    default: { return 0.25 * h * h; }
  }
}
// csgJoin with the seam's profile s
fn csgJoinP(d: f32, wd: f32, di: f32, wi: f32, op: i32, k: f32, s: i32) -> vec2f {
  let e = select(di, -di, op == 1);
  let we = select(wi, -wi, op == 1);
  let hard = op != 0;
  let h = max(k - abs(d - e), 0.0) / max(k, 1e-9);
  let bl = k * csgPhi(h, s);
  let r = select(min(d, e) - bl, max(d, e) + bl, hard);
  let below = e < d;
  let above = e > d;
  var w = select(wd, we, select(below, above, hard));
  if (h > 0.0) { w = 0.0; }
  return vec2f(r, w);
}
// cnJoin with j = (profile, band, depth, width) (gf_core::join): band 0 none, 1 pocket, 2 groove, 3 ridge, 4 bead
fn cnJoinP(a: vec3f, b: vec3f, control: vec4f, j: vec4f) -> vec3f {
  let op=i32(control.x);
  let s=i32(j.x);
  let band=i32(j.y);
  if (band > 0) {
    // (a held to a band: nothing to hold, or nothing to hold it with, leaves a as it is)
    if (a.x >= 1e5 || b.x >= 1e5) { return a; }
    let along = band == 2 || band == 4;
    let cut = band <= 2;
    // b's inside, or the band about its surface; which part's surface it is, turned to face out
    let bb = select(b.x, abs(b.x) - max(j.w, 0.0), along);
    let wb = select(b.y, select(-b.y, b.y, b.x > 0.0), along);
    // a cut: e = min(a + t, -bb); an addition: e = max(a - t, bb)
    let held = select(a.x - max(j.z, 0.0), a.x + max(j.z, 0.0), cut);
    let other = select(bb, -bb, cut);
    let heldWins = select((held > other), (held < other), cut);
    let e = select(other, held, heldWins);
    let we = select(select(wb, -wb, cut), a.y, heldWins);
    let answer = csgJoinP(a.x, a.y, e, we, select(0, 2, cut), control.y, s);
    let wins = select((e < a.x), (e > a.x), cut);
    let material = select(a.z, b.z, wins && !(cut && control.z > 0.5));
    return vec3f(answer, material);
  }
  if (b.x >= 1e5) { return select(a,b,op == 2); }
  if (a.x >= 1e5) { return select(a,b,op == 0); }
  let answer=csgJoinP(a.x,a.y,b.x,b.y,op,control.y,s);
  let below = b.x < a.x;
  let above = select(b.x,-b.x,op == 1) > a.x;
  let wins = select(below,above,op != 0);
  let material=select(a.z,b.z,wins && !(op == 1 && control.z > 0.5));
  return vec3f(answer,material);
}
";

/// Named anchors in `WGSL` (census style: each is found exactly once in a tree that joins by profile; nothing
/// weaves into them yet, they are named for whoever does).
pub const ANCHORS: [(&str, &str); 3] = [
    ("join.phi", "fn csgPhi(h: f32, s: i32) -> f32 {"),
    ("join.csg", "fn csgJoinP(d: f32, wd: f32, di: f32, wi: f32, op: i32, k: f32, s: i32) -> vec2f {"),
    ("join.node", "fn cnJoinP(a: vec3f, b: vec3f, control: vec4f, j: vec4f) -> vec3f {"),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// A small generator (no crate): xorshift, uniform in [0, 1).
    pub(crate) struct Rng(u64);
    impl Rng {
        pub(crate) fn new(seed: u64) -> Self { Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1) }
        pub(crate) fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        pub(crate) fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + (hi - lo) * self.next() }
    }

    /// The steepest slope of f over `pairs` random close pairs in [-2, 2]²: |f(p) - f(p')| / |p - p'|.
    pub(crate) fn steepest(f: &dyn Fn([f64; 2]) -> f64, pairs: usize, seed: u64) -> f64 {
        let mut r = Rng::new(seed);
        let mut most = 0.0f64;
        for _ in 0..pairs {
            let p = [r.range(-2.0, 2.0), r.range(-2.0, 2.0)];
            let t = r.range(0.0, std::f64::consts::TAU);
            let step = r.range(1e-4, 2e-2);
            let q = [p[0] + step * t.cos(), p[1] + step * t.sin()];
            most = most.max((f(p) - f(q)).abs() / step);
        }
        most
    }

    /// Points on f's zero set: every edge of a grid over [-1.6, 1.6]² whose ends differ in sign, cut by
    /// bisection to 1e-13. The distance to the nearest of them is never less than the true distance to the
    /// surface (they lie on it).
    pub(crate) fn zero_set(f: &dyn Fn([f64; 2]) -> f64, n: usize) -> Vec<[f64; 2]> {
        let h = 3.2 / n as f64;
        let at = |i: usize, j: usize| [-1.6 + i as f64 * h, -1.6 + j as f64 * h];
        let vals: Vec<f64> = (0..=n).flat_map(|j| (0..=n).map(move |i| (i, j))).map(|(i, j)| f(at(i, j))).collect();
        let v = |i: usize, j: usize| vals[j * (n + 1) + i];
        let mut out = Vec::new();
        let mut cut = |mut lo: [f64; 2], mut hi: [f64; 2], flo: f64| {
            for _ in 0..48 {
                let mid = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
                let fm = f(mid);
                if (fm < 0.0) == (flo < 0.0) { lo = mid; } else { hi = mid; }
                if (hi[0] - lo[0]).abs() + (hi[1] - lo[1]).abs() < 1e-13 { break; }
            }
            out.push([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])]);
        };
        for j in 0..=n {
            for i in 0..=n {
                if i < n && (v(i, j) < 0.0) != (v(i + 1, j) < 0.0) { cut(at(i, j), at(i + 1, j), v(i, j)); }
                if j < n && (v(i, j) < 0.0) != (v(i, j + 1) < 0.0) { cut(at(i, j), at(i, j + 1), v(i, j)); }
            }
        }
        out
    }

    /// Against the surface itself: at `count` random points of [-1.4, 1.4]², |f| is never more than the
    /// distance to the nearest point of the zero set (which is never less than the true distance), and how
    /// often it is within `near` of it (exact up to the sampling). Returns (worst excess, share near).
    pub(crate) fn against_surface(f: &dyn Fn([f64; 2]) -> f64, count: usize, seed: u64, near: f64) -> (f64, f64) {
        let zs = zero_set(f, 900);
        assert!(!zs.is_empty(), "no surface");
        let mut r = Rng::new(seed);
        let (mut worst, mut close) = (f64::MIN, 0usize);
        for _ in 0..count {
            let p = [r.range(-1.4, 1.4), r.range(-1.4, 1.4)];
            let d = zs.iter().map(|z| (z[0] - p[0]).hypot(z[1] - p[1])).fold(f64::MAX, f64::min);
            let fp = f(p).abs();
            worst = worst.max(fp - d);
            if fp >= d - near { close += 1; }
        }
        (worst, close as f64 / count as f64)
    }

    /// Two half-planes through the origin whose normals are `deg` apart: the readings a and b at p.
    pub(crate) fn planes(deg: f64) -> impl Fn([f64; 2]) -> (f64, f64) {
        let t = deg.to_radians();
        move |p: [f64; 2]| (p[0], p[0] * t.cos() + p[1] * t.sin())
    }
    /// The exact distance to a rectangle of half sizes (hx, hy) turned by `deg` about c.
    pub(crate) fn rect(p: [f64; 2], c: [f64; 2], hx: f64, hy: f64, deg: f64) -> f64 {
        let (s, co) = deg.to_radians().sin_cos();
        let (x, y) = (p[0] - c[0], p[1] - c[1]);
        let (u, v) = ((co * x + s * y).abs() - hx, (-s * x + co * y).abs() - hy);
        u.max(0.0).hypot(v.max(0.0)) + u.max(v).min(0.0)
    }
    /// Two rectangles crossing at `deg`.
    pub(crate) fn boxes(deg: f64) -> impl Fn([f64; 2]) -> (f64, f64) {
        move |p: [f64; 2]| (rect(p, [0.0, -0.2], 0.8, 0.35, 0.0), rect(p, [0.25, 0.3], 0.7, 0.22, deg))
    }
    type Pair = Box<dyn Fn([f64; 2]) -> (f64, f64)>;

    const OPS: [Op; 3] = [Op::Union, Op::Subtract, Op::Intersect];
    const BANDS: [Band; 4] = [Band::Pocket, Band::Groove, Band::Ridge, Band::Bead];

    #[test]
    fn profiles_start_flat_and_lean_at_most_half() {
        for p in Profile::ALL {
            assert_eq!(phi(p, 0.0), 0.0, "{p:?} at the band's edge");
            let mut last = 0.0;
            for i in 1..=10_000 {
                let h = i as f64 / 10_000.0;
                let v = phi(p, h);
                let slope = (v - last) * 10_000.0;
                assert!((-1e-12..=0.5 + 1e-9).contains(&slope), "{p:?}: φ' = {slope} at h = {h}");
                last = v;
            }
        }
    }

    #[test]
    fn the_fillet_is_the_join_trees_have_always_had() {
        // csgJoin: h = max(k - |d - e|, 0) / k, blend = h²·k/4
        let mut r = Rng::new(3);
        for _ in 0..10_000 {
            let (a, b, k) = (r.range(-1.0, 1.0), r.range(-1.0, 1.0), r.range(0.0, 0.5));
            let h = (k - (a - b).abs()).max(0.0) / k.max(1e-9);
            let old = a.min(b) - h * h * k * 0.25;
            assert!((union(a, b, k, Profile::Fillet) - old).abs() < 1e-14);
        }
    }

    #[test]
    fn joins_never_steepen_whatever_the_angle() {
        // two planes whose normals are 0 to 180 degrees apart, every profile, every operation
        let mut most = 0.0f64;
        for deg in (0..=180).step_by(2) {
            let ab = planes(deg as f64);
            for p in Profile::ALL {
                for op in OPS {
                    for k in [0.05, 0.3, 1.0] {
                        let f = |x: [f64; 2]| { let (a, b) = ab(x); join(op, a, b, k, p) };
                        let s = steepest(&f, 4000, deg as u64 * 31 + k.to_bits() % 97);
                        most = most.max(s);
                        assert!(s <= 1.0 + 1e-9, "{p:?} {op:?} k {k} at {deg} degrees: slope {s}");
                    }
                }
            }
        }
        assert!(most > 0.999, "the test never met a slope near 1 ({most})");
    }

    #[test]
    fn joins_never_overstate_the_distance_to_their_surface() {
        for deg in [30.0, 90.0, 150.0] {
            let pairs: [(&str, Pair); 2] = [("planes", Box::new(planes(deg))), ("boxes", Box::new(boxes(deg)))];
            for (name, ab) in &pairs {
                for p in Profile::ALL {
                    for op in OPS {
                        let f = |x: [f64; 2]| { let (a, b) = ab(x); join(op, a, b, 0.3, p) };
                        let (worst, _) = against_surface(&f, 400, deg as u64, 0.01);
                        assert!(worst <= 1e-9, "{name} {p:?} {op:?} at {deg}: reads {worst} beyond the surface");
                    }
                }
            }
        }
    }

    #[test]
    fn the_round_is_a_circle_where_the_surfaces_meet_square() {
        // a = x, b = y, union: the seam is the arc of radius k about (k, k), from (0, k) to (k, 0)
        let k = 0.37;
        let mut worst = 0.0f64;
        for i in 0..=2000 {
            // along the arc's own angle, from one tangent point to the other
            let t = std::f64::consts::PI + std::f64::consts::FRAC_PI_2 * i as f64 / 2000.0;
            let p = [k + k * t.cos(), k + k * t.sin()];
            worst = worst.max(union(p[0], p[1], k, Profile::Round).abs());
        }
        assert!(worst < 1e-12, "off the arc by {worst}");
        // on the diagonal, outside the arc, it reads √½ of the true distance to it (a bound, not exact)
        for c in [0.15, 0.2, 0.3] {
            let truth = k - std::f64::consts::SQRT_2 * (k - c);
            let u = union(c, c, k, Profile::Round);
            assert!((u - truth * std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12, "{u} against {truth}");
        }
    }

    #[test]
    fn the_chamfer_is_flat_and_true_to_its_face_at_half() {
        // a = x, b = y: inside the band the seam is the line x + y = k, and the reading is half the sum
        let k = 0.4;
        let mut r = Rng::new(11);
        for _ in 0..10_000 {
            let (x, y) = (r.range(-1.0, 1.0), r.range(-1.0, 1.0));
            if (x - y).abs() >= k { continue; }
            let u = union(x, y, k, Profile::Chamfer);
            assert!((u - 0.5 * (x + y - k)).abs() < 1e-14);
            // the true distance to the face is (x + y - k)·√½: the reading is √½ of it (decision D7)
            let truth = (x + y - k) * std::f64::consts::FRAC_1_SQRT_2;
            assert!(u.abs() <= truth.abs() + 1e-14);
        }
    }

    #[test]
    fn the_right_angle_chamfer_overstates_away_from_a_right_angle() {
        // the √½ chamfer `min(min(a, b), (a + b - k)·√½)`: slope 1 at 90 degrees, more than 1 elsewhere. Why the
        // chamfer here is scaled by ½ (decision D7).
        let at = |deg: f64| {
            let ab = planes(deg);
            let f = |x: [f64; 2]| { let (a, b) = ab(x); a.min(b).min((a + b - 0.3) * std::f64::consts::FRAC_1_SQRT_2) };
            steepest(&f, 40_000, 5)
        };
        assert!(at(90.0) <= 1.0 + 1e-9);
        assert!(at(20.0) > 1.3, "{}", at(20.0));
        assert!(at(0.0) > 1.4, "{}", at(0.0));
    }

    #[test]
    fn bands_never_steepen_whatever_the_angle() {
        let mut r = Rng::new(17);
        for deg in (0..=180).step_by(5) {
            let ab = planes(deg as f64);
            for kind in BANDS {
                for p in Profile::ALL {
                    let (t, w, k) = (r.range(0.02, 0.6), r.range(0.0, 0.4), r.range(0.0, 0.5));
                    let f = |x: [f64; 2]| { let (a, b) = ab(x); band(kind, a, b, t, w, k, p) };
                    let s = steepest(&f, 3000, deg as u64 + 7);
                    assert!(s <= 1.0 + 1e-9, "{kind:?} {p:?} t {t} w {w} k {k} at {deg}: slope {s}");
                }
            }
        }
    }

    #[test]
    fn bands_never_overstate_the_distance_to_their_surface() {
        // a box with a disc and with a turned box: every band and profile
        let disc = |x: [f64; 2]| (rect(x, [0.0, -0.3], 0.9, 0.4, 0.0), (x[0] - 0.2).hypot(x[1] - 0.2) - 0.35);
        let pairs: [(&str, Pair); 2] = [("disc", Box::new(disc)), ("boxes", Box::new(boxes(30.0)))];
        for (name, ab) in &pairs {
            for kind in BANDS {
                for p in Profile::ALL {
                    let f = |x: [f64; 2]| { let (a, b) = ab(x); band(kind, a, b, 0.15, 0.06, 0.05, p) };
                    let (worst, _) = against_surface(&f, 300, 3, 0.01);
                    assert!(worst <= 1e-9, "{name} {kind:?} {p:?}: reads {worst} beyond the surface");
                }
            }
        }
    }

    #[test]
    fn bands_hold_their_depth_and_leave_the_rest_alone() {
        // a = the half-plane y < 0 (a = y); b a disc about the origin of radius 0.5; hard seam
        let ab = |x: [f64; 2]| (x[1], x[0].hypot(x[1]) - 0.5);
        let (t, w) = (0.2, 0.05);
        let f = |kind: Band, x: [f64; 2]| { let (a, b) = ab(x); band(kind, a, b, t, w, 0.0, Profile::Fillet) };
        // far from b, every band is a itself
        for x in [[1.2, -0.3], [-1.0, 0.4], [0.9, 0.0]] {
            for kind in BANDS { assert_eq!(f(kind, x), x[1], "{kind:?} at {x:?}"); }
        }
        // the pocket's floor is t below a's surface, the ridge's top t above it
        assert!((f(Band::Pocket, [0.0, -0.3]) + 0.1).abs() < 1e-12, "{}", f(Band::Pocket, [0.0, -0.3]));
        assert!((f(Band::Pocket, [0.0, -0.2])).abs() < 1e-12);
        assert!((f(Band::Ridge, [0.0, 0.2])).abs() < 1e-12);
        assert!(f(Band::Ridge, [0.0, 0.1]) < 0.0 && f(Band::Ridge, [0.0, 0.3]) > 0.0);
        // a groove cuts a channel of half-width w about b's circle, and only there
        assert!(f(Band::Groove, [0.5, -0.1]) > 0.0, "in the channel");
        assert!(f(Band::Groove, [0.0, -0.1]) < 0.0, "inside the circle, away from its edge");
        assert!(f(Band::Groove, [0.5, -0.3]) < 0.0, "below the channel's floor");
        // a bead rises along b's circle only
        assert!(f(Band::Bead, [0.5, 0.1]) < 0.0 && f(Band::Bead, [0.0, 0.1]) > 0.0);
        // band None is a, and a hard band equals its formula
        assert_eq!(band(Band::None, 0.3, -1.0, t, w, 0.1, Profile::Round), 0.3);
    }

    #[test]
    fn the_shader_text_is_valid_and_its_anchors_are_found_once() {
        let module = naga::front::wgsl::parse_str(WGSL).expect("join text parses");
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
            .validate(&module).expect("join text validates");
        for (name, anchor) in ANCHORS {
            assert_eq!(WGSL.matches(anchor).count(), 1, "{name}");
        }
    }
}
