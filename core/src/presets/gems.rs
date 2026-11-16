//! Gem cuts: one generator that builds a faceted stone from its proportions.
//!
//! A cut gem is a convex solid: the space on the inner side of every facet's plane. So a cut is a list of
//! planes, and the stone is  d(p) = max over facets of (n·p − h),  which is zero on the surface, negative
//! inside, and inside is exactly the distance to the nearest facet. A cutter does not choose planes
//! freely, though: a cut is laid out from a few proportions (how wide the table is, the angle of the
//! crown, the angle of the pavilion, the thickness of the girdle) and from where facets are made to meet.
//! The recipes below do the same, so the sliders are the numbers a gemmologist would quote.
//!
//! Every recipe is written once, over a kind of number that can either be worked out (`f64`) or written
//! down (`Sym`, an expression in the sliders). The first gives the planes for the tests and for choosing
//! which planes the shader needs; the second prints the WGSL that computes the same planes from the
//! sliders on the GPU. There is one source for both, so they cannot drift apart.
//!
//! The shader folds each point by the cut's symmetry before measuring it, so that a round brilliant's 57
//! facets cost seven planes a lookup: the planes kept are those whose normals lie in the wedge the fold
//! maps everything into (for a point in that wedge, no mirror image of a plane can be further out than
//! the plane itself).
//!
//! Directions: y is up through the table, the girdle lies in a plane of constant y, azimuth is measured
//! from +x towards +z. Lengths are in the volume's units; a stone is about 2.2 across.

use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::ops::{Add, Div, Mul, Neg, Sub};

use super::*;
use crate::field::member_name;

// ---------------------------------------------------------------------------------------------------
// Numbers that can be worked out or written down

/// What the recipes compute with.
pub trait Num: Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> {
    fn k(v: f64) -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn pow(self, e: Self) -> Self;
    /// the larger of the two
    fn at_least(self, o: Self) -> Self;
}

impl Num for f64 {
    fn k(v: f64) -> Self { v }
    fn sqrt(self) -> Self { f64::sqrt(self) }
    fn sin(self) -> Self { f64::sin(self) }
    fn cos(self) -> Self { f64::cos(self) }
    fn tan(self) -> Self { f64::tan(self) }
    fn asin(self) -> Self { f64::asin(self) }
    fn pow(self, e: Self) -> Self { f64::powf(self, e) }
    fn at_least(self, o: Self) -> Self { f64::max(self, o) }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Un { Neg, Sqrt, Sin, Cos, Tan, Asin }
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Bin { Add, Sub, Mul, Div, Pow, Max }
#[derive(Clone, Copy, PartialEq, Debug)]
enum Node {
    /// a number
    K(f64),
    /// a slider, by its place among the preset's
    P(usize),
    U(Un, u32),
    B(Bin, u32, u32),
}

/// Every expression made so far on this thread, each written once however often it is used.
#[derive(Default)]
struct Arena {
    nodes: Vec<Node>,
    seen: HashMap<(u8, u64, u64), u32>,
}
thread_local! {
    static ARENA: RefCell<Arena> = RefCell::new(Arena::default());
}

/// An expression in the sliders: a number that is written down instead of worked out. Arithmetic on
/// numbers that are already known is done on the spot, so only what depends on a slider reaches the shader.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Sym(u32);

impl Sym {
    fn node(n: Node) -> Sym {
        let key = match n {
            Node::K(v) => (0, v.to_bits(), 0),
            Node::P(i) => (1, i as u64, 0),
            Node::U(op, a) => (2 + op as u8, a as u64, 0),
            Node::B(op, a, b) => (16 + op as u8, a as u64, b as u64),
        };
        ARENA.with(|a| {
            let mut a = a.borrow_mut();
            if let Some(&i) = a.seen.get(&key) { return Sym(i); }
            let i = a.nodes.len() as u32;
            a.nodes.push(n);
            a.seen.insert(key, i);
            Sym(i)
        })
    }
    fn get(self) -> Node {
        ARENA.with(|a| a.borrow().nodes[self.0 as usize])
    }
    fn known(self) -> Option<f64> {
        match self.get() { Node::K(v) => Some(v), _ => None }
    }
    /// The slider in place `i` of the preset's.
    pub fn slider(i: usize) -> Sym { Sym::node(Node::P(i)) }
    /// Forget every expression (a new shader is about to be written).
    pub fn reset() {
        ARENA.with(|a| *a.borrow_mut() = Arena::default());
    }
    fn un(op: Un, a: Sym) -> Sym {
        if let Some(v) = a.known() {
            return Sym::k(match op { Un::Neg => -v, Un::Sqrt => v.sqrt(), Un::Sin => v.sin(), Un::Cos => v.cos(), Un::Tan => v.tan(), Un::Asin => v.asin() });
        }
        Sym::node(Node::U(op, a.0))
    }
    fn bin(op: Bin, a: Sym, b: Sym) -> Sym {
        match (op, a.known(), b.known()) {
            (_, Some(x), Some(y)) => return Sym::k(match op { Bin::Add => x + y, Bin::Sub => x - y, Bin::Mul => x * y, Bin::Div => x / y, Bin::Pow => x.powf(y), Bin::Max => x.max(y) }),
            (Bin::Mul, Some(x), _) if x == 0.0 => return Sym::k(0.0),
            (Bin::Mul, _, Some(y)) if y == 0.0 => return Sym::k(0.0),
            (Bin::Mul, Some(x), _) if x == 1.0 => return b,
            (Bin::Mul, _, Some(y)) if y == 1.0 => return a,
            (Bin::Add, Some(x), _) if x == 0.0 => return b,
            (Bin::Add | Bin::Sub, _, Some(y)) if y == 0.0 => return a,
            (Bin::Sub, Some(x), _) if x == 0.0 => return -b,
            (Bin::Div, _, Some(y)) if y == 1.0 => return a,
            (Bin::Div, Some(x), _) if x == 0.0 => return Sym::k(0.0),
            _ => {}
        }
        Sym::node(Node::B(op, a.0, b.0))
    }
    /// What the expression comes to for these slider values.
    pub fn eval(self, sliders: &[f64]) -> f64 {
        ARENA.with(|a| {
            let a = a.borrow();
            let mut v = vec![0.0f64; self.0 as usize + 1];
            for i in 0..v.len() {
                v[i] = match a.nodes[i] {
                    Node::K(x) => x,
                    Node::P(j) => sliders[j],
                    Node::U(op, x) => { let x = v[x as usize]; match op { Un::Neg => -x, Un::Sqrt => x.sqrt(), Un::Sin => x.sin(), Un::Cos => x.cos(), Un::Tan => x.tan(), Un::Asin => x.asin() } }
                    Node::B(op, x, y) => { let (x, y) = (v[x as usize], v[y as usize]); match op { Bin::Add => x + y, Bin::Sub => x - y, Bin::Mul => x * y, Bin::Div => x / y, Bin::Pow => x.powf(y), Bin::Max => x.max(y) } }
                };
            }
            v[self.0 as usize]
        })
    }
    /// WGSL that computes these expressions: the statements (one `let` for every step that depends on a
    /// slider, shared steps written once) and then what each expression is called. `sliders`: what each
    /// slider is called in the shader.
    pub fn wgsl(roots: &[Sym], sliders: &[String]) -> (String, Vec<String>) {
        ARENA.with(|a| {
            let nodes = &a.borrow().nodes;
            let mut used = vec![false; nodes.len()];
            let mut stack: Vec<u32> = roots.iter().map(|r| r.0).collect();
            while let Some(i) = stack.pop() {
                if std::mem::replace(&mut used[i as usize], true) { continue; }
                match nodes[i as usize] {
                    Node::U(_, x) => stack.push(x),
                    Node::B(_, x, y) => { stack.push(x); stack.push(y); }
                    _ => {}
                }
            }
            let name = |i: u32| match nodes[i as usize] {
                Node::K(v) if v < 0.0 => format!("({v:?})"),
                Node::K(v) => format!("{v:?}"),
                Node::P(j) => format!("pr.{}", sliders[j]),
                _ => format!("g{i}"),
            };
            let mut text = String::new();
            for (i, n) in nodes.iter().enumerate() {
                if !used[i] { continue; }
                let rhs = match *n {
                    Node::K(_) | Node::P(_) => continue,
                    Node::U(Un::Neg, x) => format!("-{}", name(x)),
                    Node::U(op, x) => format!("{}({})", match op { Un::Sqrt => "sqrt", Un::Sin => "sin", Un::Cos => "cos", Un::Tan => "tan", _ => "asin" }, name(x)),
                    Node::B(Bin::Pow, x, y) => format!("pow({}, {})", name(x), name(y)),
                    Node::B(Bin::Max, x, y) => format!("max({}, {})", name(x), name(y)),
                    Node::B(op, x, y) => format!("{} {} {}", name(x), match op { Bin::Add => "+", Bin::Sub => "-", Bin::Mul => "*", _ => "/" }, name(y)),
                };
                text += &format!("  let g{i}: f32 = {rhs};\n");
            }
            (text, roots.iter().map(|r| name(r.0)).collect())
        })
    }
}

impl Num for Sym {
    fn k(v: f64) -> Self { Sym::node(Node::K(v)) }
    fn sqrt(self) -> Self { Sym::un(Un::Sqrt, self) }
    fn sin(self) -> Self { Sym::un(Un::Sin, self) }
    fn cos(self) -> Self { Sym::un(Un::Cos, self) }
    fn tan(self) -> Self { Sym::un(Un::Tan, self) }
    fn asin(self) -> Self { Sym::un(Un::Asin, self) }
    fn pow(self, e: Self) -> Self { Sym::bin(Bin::Pow, self, e) }
    fn at_least(self, o: Self) -> Self { Sym::bin(Bin::Max, self, o) }
}
impl Add for Sym { type Output = Sym; fn add(self, b: Sym) -> Sym { Sym::bin(Bin::Add, self, b) } }
impl Sub for Sym { type Output = Sym; fn sub(self, b: Sym) -> Sym { Sym::bin(Bin::Sub, self, b) } }
impl Mul for Sym { type Output = Sym; fn mul(self, b: Sym) -> Sym { Sym::bin(Bin::Mul, self, b) } }
impl Div for Sym { type Output = Sym; fn div(self, b: Sym) -> Sym { Sym::bin(Bin::Div, self, b) } }
impl Neg for Sym { type Output = Sym; fn neg(self) -> Sym { Sym::un(Un::Neg, self) } }

// ---------------------------------------------------------------------------------------------------
// Planes

pub type V3<T> = [T; 3];

fn k<T: Num>(v: f64) -> T { T::k(v) }
fn sub<T: Num>(a: V3<T>, b: V3<T>) -> V3<T> { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn dot<T: Num>(a: V3<T>, b: V3<T>) -> T { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn cross<T: Num>(a: V3<T>, b: V3<T>) -> V3<T> { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }
fn unit<T: Num>(a: V3<T>) -> V3<T> { let l = dot(a, a).sqrt(); [a[0] / l, a[1] / l, a[2] / l] }

/// The half of space where n·p <= h; n has length 1 and points out of the stone.
#[derive(Clone, Copy, Debug)]
pub struct Plane<T> {
    pub n: V3<T>,
    pub h: T,
}
impl<T: Num> Plane<T> {
    /// Facing along `n` (of any length), through `p`.
    fn facing(n: V3<T>, p: V3<T>) -> Self {
        let n = unit(n);
        Self { n, h: dot(n, p) }
    }
    /// Through three points, facing the side from which they run anticlockwise.
    fn through(a: V3<T>, b: V3<T>, c: V3<T>) -> Self {
        Self::facing(cross(sub(b, a), sub(c, a)), a)
    }
    /// Everything below the height y.
    fn below(y: T) -> Self { Self { n: [k(0.0), k(1.0), k(0.0)], h: y } }
    /// Everything above the height y.
    fn above(y: T) -> Self { Self { n: [k(0.0), k(-1.0), k(0.0)], h: -y } }
}
/// The point where three planes meet.
fn meet<T: Num>(a: Plane<T>, b: Plane<T>, c: Plane<T>) -> V3<T> {
    let (bc, ca, ab) = (cross(b.n, c.n), cross(c.n, a.n), cross(a.n, b.n));
    let det = dot(a.n, bc);
    [0, 1, 2].map(|i| (a.h * bc[i] + b.h * ca[i] + c.h * ab[i]) / det)
}

/// One facet, and which kind it is.
#[derive(Clone, Copy, Debug)]
pub struct Facet<T> {
    pub tier: &'static str,
    pub plane: Plane<T>,
}

/// The girdle where it is not faceted: the outline of the stone seen from above, as a wall.
#[derive(Clone, Copy, Debug)]
pub enum Rim<T> {
    /// the facets bound the stone by themselves
    None,
    /// a circle of radius r
    Round { r: T },
    /// |x/a|^e + |z/b|^e = 1: an ellipse at e = 2, squarer above it
    Squircle { a: T, b: T, e: T },
    /// two arcs of radius rho whose centres are c either side of the long axis: a pointed oval
    Lens { rho: T, c: T },
    /// a half circle of radius w about (x0, 0) on one side, the pointed half of a lens on the other
    Pear { x0: T, w: T, rho: T, c: T },
    /// three arcs of radius rho, each centred c beyond the middle from the side it bounds: a triangle
    /// with bowed sides, its corners at azimuth 0, 120 and 240 degrees
    Triangle { rho: T, c: T },
}

/// A cut stone: its facets, its rim, and how far it is lowered so that its middle sits at the origin.
#[derive(Clone, Debug)]
pub struct Stone<T> {
    pub facets: Vec<Facet<T>>,
    pub rim: Rim<T>,
    pub lift: T,
}

impl Stone<f64> {
    /// How far outside the rim's wall a point is (not exact for the squircle, but zero on it).
    pub fn rim_dist(&self, x: f64, z: f64) -> f64 {
        match self.rim {
            Rim::None => f64::NEG_INFINITY,
            Rim::Round { r } => x.hypot(z) - r,
            Rim::Squircle { a, b, e } => {
                let (u, v) = (x.abs() / a, z.abs() / b);
                let s = (u.powf(e) + v.powf(e)).max(1e-9);
                let g = s.powf(1.0 / e - 1.0) * (u.powf(e - 1.0) / a).hypot(v.powf(e - 1.0) / b);
                (s.powf(1.0 / e) - 1.0) / g.max(1e-6)
            }
            Rim::Lens { rho, c } => x.hypot(z.abs() + c) - rho,
            Rim::Pear { x0, w, rho, c } => if x < x0 { (x - x0).hypot(z) - w } else { (x - x0).hypot(z.abs() + c) - rho },
            Rim::Triangle { rho, c } => (0..3).map(|i| { let a = (60.0 + 120.0 * i as f64) * PI / 180.0; (x + c * a.cos()).hypot(z + c * a.sin()) - rho }).fold(f64::NEG_INFINITY, f64::max),
        }
    }
    /// d(p): the largest of the facets' distances and the rim's. Negative inside the stone.
    pub fn dist(&self, p: V3<f64>) -> f64 {
        let q = [p[0], p[1] + self.lift, p[2]];
        self.facets.iter().fold(self.rim_dist(q[0], q[2]), |d, f| d.max(dot(f.plane.n, q) - f.plane.h))
    }
    /// How many facets of each kind, in the order they were cut.
    pub fn count(&self) -> Vec<(&'static str, usize)> {
        let mut out: Vec<(&'static str, usize)> = Vec::new();
        for f in &self.facets {
            match out.iter_mut().find(|(t, _)| *t == f.tier) {
                Some(slot) => slot.1 += 1,
                None => out.push((f.tier, 1)),
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------------
// Outlines, and the stations round them at which a brilliant's main facets are placed

/// The outline of a brilliant seen from above.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Outline {
    Round,
    Oval,
    Cushion,
    Marquise,
    Pear,
    /// a rectangle with its corners cut off, as an emerald cut has
    Radiant,
    /// a triangle with bowed sides
    Trillion,
}

/// The stations round an outline (eight, or six for a triangle), anticlockwise from +x: where each main
/// facet touches the girdle (a point and the outward direction there), the outline's point half way to
/// the next station, the point the table and the pavilion are centred on, and half the stone's width.
/// An outline of straight sides has no rim: its girdle is facets, given here as outward direction and
/// distance from the centre.
struct Ring<T> {
    at: Vec<([T; 2], [T; 2])>,
    mid: Vec<[T; 2]>,
    centre: [T; 2],
    half_width: T,
    rim: Rim<T>,
    walls: Vec<([T; 2], T)>,
}

/// Half the length of the longest stones.
const REACH: f64 = 1.22;
const RADIUS: f64 = 1.1;

fn ring<T: Num>(outline: Outline, shape: &[T]) -> Ring<T> {
    let zero = k::<T>(0.0);
    let turn = |i: usize, per: usize| 2.0 * PI * i as f64 / per as f64;
    match outline {
        Outline::Round => {
            let r = k::<T>(RADIUS);
            let on = |a: f64| ([r * k(a.cos()), r * k(a.sin())], [k::<T>(a.cos()), k::<T>(a.sin())]);
            Ring {
                at: (0..8).map(|i| on(turn(i, 8))).collect(),
                mid: (0..8).map(|i| on(turn(2 * i + 1, 16)).0).collect(),
                centre: [zero, zero], half_width: r, rim: Rim::Round { r }, walls: Vec::new(),
            }
        }
        // shape: length / width, and for the cushion the exponent of its outline
        Outline::Oval | Outline::Cushion => {
            let (ratio, e) = (shape[0], if outline == Outline::Oval { k::<T>(2.0) } else { shape[1] });
            // a cushion's corners reach further than its sides: sized so that they stay inside the volume
            let a = k::<T>(if outline == Outline::Oval { REACH } else { 1.0 });
            let b = a / ratio;
            // the point at angle t of the circle this outline is a stretched and squared version of, and
            // the outward direction there
            let on = |t: f64| {
                let (c, s) = (t.cos(), t.sin());
                let pw = |v: f64, by: T| if v.abs() < 1e-12 { k::<T>(0.0) } else { k::<T>(v.signum()) * k::<T>(v.abs()).pow(by) };
                let p = [a * pw(c, k::<T>(2.0) / e), b * pw(s, k::<T>(2.0) / e)];
                let g = [pw(c, k::<T>(2.0) - k::<T>(2.0) / e) / a, pw(s, k::<T>(2.0) - k::<T>(2.0) / e) / b];
                let l = (g[0] * g[0] + g[1] * g[1]).sqrt();
                (p, [g[0] / l, g[1] / l])
            };
            Ring {
                at: (0..8).map(|i| on(turn(i, 8))).collect(),
                mid: (0..8).map(|i| on(turn(2 * i + 1, 16)).0).collect(),
                centre: [zero, zero], half_width: b, rim: Rim::Squircle { a, b, e }, walls: Vec::new(),
            }
        }
        // shape: length / width
        Outline::Marquise => {
            let l = k::<T>(REACH);
            let w = l / shape[0];
            let rho = (l * l + w * w) / (k::<T>(2.0) * w);
            let c = rho - w;
            // the angle either tip is at, seen from the centre of the arc it ends
            let tip = (l / rho).asin();
            // the point of the arc on the +z side at the fraction f of that angle, and the outward direction
            let arc = |f: f64, side: f64| {
                let (s, co) = ((tip * k(f)).sin(), (tip * k(f)).cos());
                ([rho * s, k::<T>(side) * (rho * co - c)], [s, k::<T>(side) * co])
            };
            // anticlockwise: the +x tip, over the +z arc to the -x tip, and back along the -z arc
            let at = vec![
                ([l, zero], [k::<T>(1.0), zero]), arc(0.5, 1.0), arc(0.0, 1.0), arc(-0.5, 1.0),
                ([-l, zero], [k::<T>(-1.0), zero]), arc(-0.5, -1.0), arc(0.0, -1.0), arc(0.5, -1.0),
            ];
            let mid = [(0.75, 1.0), (0.25, 1.0), (-0.25, 1.0), (-0.75, 1.0), (-0.75, -1.0), (-0.25, -1.0), (0.25, -1.0), (0.75, -1.0)].iter().map(|&(f, side)| arc(f, side).0).collect();
            Ring { at, mid, centre: [zero, zero], half_width: w, rim: Rim::Lens { rho, c }, walls: Vec::new() }
        }
        // shape: length / width
        Outline::Pear => {
            // half the stone's length is REACH; the round end has radius w, the point is `nose` beyond its centre
            let w = k::<T>(REACH) / shape[0];
            let nose = k::<T>(2.0 * REACH) - w;
            let c = (nose * nose - w * w) / (k::<T>(2.0) * w);
            let rho = w + c;
            let x0 = k::<T>(REACH) - nose;
            let tip = (nose / rho).asin();
            let arc = |f: f64, side: f64| {
                let (s, co) = ((tip * k(f)).sin(), (tip * k(f)).cos());
                ([x0 + rho * s, k::<T>(side) * (rho * co - c)], [s, k::<T>(side) * co])
            };
            let round = |deg: f64| {
                let (co, s) = ((deg * PI / 180.0).cos(), (deg * PI / 180.0).sin());
                ([x0 + w * k(co), w * k(s)], [k::<T>(co), k::<T>(s)])
            };
            let at = vec![
                ([x0 + nose, zero], [k::<T>(1.0), zero]), arc(0.5, 1.0), round(90.0), round(135.0),
                round(180.0), round(225.0), round(270.0), arc(0.5, -1.0),
            ];
            let mid = vec![arc(0.75, 1.0).0, arc(0.25, 1.0).0, round(112.5).0, round(157.5).0, round(202.5).0, round(247.5).0, arc(0.25, -1.0).0, arc(0.75, -1.0).0];
            // the table and the culet sit a little towards the point from the centre of the round end
            Ring { at, mid, centre: [x0 + k::<T>(0.12) * nose, zero], half_width: w, rim: Rim::Pear { x0, w, rho, c }, walls: Vec::new() }
        }
        // shape: length / width, and how much of the half-width each corner loses
        Outline::Radiant => {
            let a = k::<T>(REACH) / (k::<T>(1.0) + k::<T>(1.0) / (shape[0] * shape[0])).sqrt();
            let b = a / shape[0];
            let cut = shape[1] * b;
            let (s, h) = (0.5f64.sqrt(), k::<T>(0.5));
            let mut at = Vec::new();
            let mut mid = Vec::new();
            let mut walls = Vec::new();
            // the four quarters in turn: a long or short side, then the cut corner after it
            for &(sx, sz) in &[(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)] {
                let (x, z) = (k::<T>(sx), k::<T>(sz));
                // where the corner's cut meets the side along x = ±a, and the side along z = ±b
                let (on_a, on_b) = ([x * a, z * (b - cut)], [x * (a - cut), z * b]);
                let corner = ([x * (a - h * cut), z * (b - h * cut)], [k::<T>(sx * s), k::<T>(sz * s)]);
                let wall = ([k::<T>(sx * s), k::<T>(sz * s)], (a + b - cut) * k(s));
                // anticlockwise: in the quarters where x and z have the same sign the side along x = ±a comes first
                if sx * sz > 0.0 {
                    at.extend([([x * a, zero], [x, zero]), corner]);
                    mid.extend([on_a, on_b]);
                    walls.extend([([x, zero], a), wall]);
                } else {
                    at.extend([([zero, z * b], [zero, z]), corner]);
                    mid.extend([on_b, on_a]);
                    walls.extend([([zero, z], b), wall]);
                }
            }
            Ring { at, mid, centre: [zero, zero], half_width: b, rim: Rim::None, walls }
        }
        // shape: how far the middle of each side stands out, as a fraction of the corners' distance
        // (0.5 would be straight sides)
        Outline::Trillion => {
            let far = k::<T>(1.2);
            let near = shape[0] * far;
            // each side is an arc through two corners and the side's middle
            let c = (far * far - near * near) / (k::<T>(2.0) * near - far);
            let rho = near + c;
            // half the angle a side takes up, seen from the centre of its arc
            let half = (far * k(60f64.to_radians().sin()) / rho).asin();
            let mut at = Vec::new();
            let mut mid = Vec::new();
            for i in 0..3 {
                let (corner, side) = (120.0 * i as f64 * PI / 180.0, (120.0 * i as f64 + 60.0) * PI / 180.0);
                // the point of this side's arc at the fraction f of its half-angle (f = -1 and 1 are its corners)
                let arc = |f: f64| {
                    let a = k::<T>(side) + half * k(f);
                    [rho * a.cos() - c * k(side.cos()), rho * a.sin() - c * k(side.sin())]
                };
                at.push(([far * k(corner.cos()), far * k(corner.sin())], [k::<T>(corner.cos()), k::<T>(corner.sin())]));
                at.push((arc(0.0), [k::<T>(side.cos()), k::<T>(side.sin())]));
                mid.extend([arc(-0.5), arc(0.5)]);
            }
            Ring { at, mid, centre: [zero, zero], half_width: near, rim: Rim::Triangle { rho, c }, walls: Vec::new() }
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// The recipes

/// The proportions of a brilliant. Lengths as fractions of the stone's width, angles in degrees.
struct Brilliant<T> {
    /// width of the table, corner to corner
    table: T,
    /// angle between the main crown facets and the girdle's plane
    crown: T,
    /// angle between the main pavilion facets and the girdle's plane
    pavilion: T,
    /// thickness of the girdle where the main facets meet it (its thickest places)
    girdle: T,
    /// how far the star facets reach from the table's edge towards the girdle (0..1)
    star: T,
    /// how far the lower girdle facets reach from the girdle towards the culet (0..1)
    lower: T,
    /// width of the culet
    culet: T,
}

fn deg<T: Num>(a: T) -> T { a * k(PI / 180.0) }

/// The brilliant cut on any outline: a table, eight main (bezel) facets, eight stars, sixteen upper
/// girdle facets; eight pavilion mains, sixteen lower girdle facets, and the culet.
///
/// Every facet is placed as a cutter places it, by what it must meet:
///   bezel k      touches the girdle at station k, is square to the outline there, and reaches the table
///                at the table's corner k (the table is the outline's eight stations, shrunk)
///   star k       holds the table's edge from corner k to corner k+1 and comes down to the point M where
///                bezels k and k+1 meet, as far down as the star length says
///   upper girdle two for every star: each holds M, the girdle point of one of the two bezels, and the
///                girdle's point half way between them
///   pavilion k   touches the girdle below bezel k, square to the outline, and runs to the culet
///   lower girdle two between neighbouring mains: each holds the point N on the ridge where the two mains
///                meet (as far down as the lower-half length says), one main's girdle point, and the
///                girdle's point half way between
fn brilliant<T: Num>(outline: Outline, b: &Brilliant<T>, shape: &[T]) -> Stone<T> {
    let r = ring(outline, shape);
    let n = r.at.len();
    let (zero, half) = (k::<T>(0.0), k::<T>(0.5));
    let width = k::<T>(2.0) * r.half_width;
    // heights: the girdle is centred on y = 0
    let (g_top, g_bot) = (half * b.girdle * width, -half * b.girdle * width);
    let crown_height = (k::<T>(1.0) - b.table) * r.half_width * deg(b.crown).tan();
    let depth = r.half_width * deg(b.pavilion).tan();
    let (y_table, y_culet) = (g_top + crown_height, g_bot - depth);

    let at3 = |p: [T; 2], y: T| [p[0], y, p[1]];
    // the direction along the outline at a station (anticlockwise), from the outward direction there
    let along = |m: [T; 2]| [-m[1], zero, m[0]];
    let corner = |i: usize| at3([r.centre[0] + b.table * (r.at[i].0[0] - r.centre[0]), r.centre[1] + b.table * (r.at[i].0[1] - r.centre[1])], y_table);
    let apex = at3(r.centre, y_culet);

    let bezel: Vec<Plane<T>> = (0..n).map(|i| {
        let g = at3(r.at[i].0, g_top);
        Plane::facing(cross(sub(corner(i), g), along(r.at[i].1)), g)
    }).collect();
    let main: Vec<Plane<T>> = (0..n).map(|i| {
        let g = at3(r.at[i].0, g_bot);
        Plane::facing(cross(along(r.at[i].1), sub(apex, g)), g)
    }).collect();

    // Star length and lower-half length are measured in the view from above, along the line half way
    // between two mains: from the table's edge to the girdle, and from the girdle to the culet. The line
    // where two neighbouring mains meet runs straight in that view, so each length is a fraction of the
    // way down that line (worked out for the round stone, and used for every outline).
    let c8 = (PI / n as f64).cos();
    // (a short star under a wide table would end above the table: it is held just below it instead)
    let down_star = ((k::<T>(c8) * (b.table * k(c8) + b.star * (k::<T>(1.0) - b.table * k(c8))) - b.table) / (k::<T>(1.0) - b.table)).at_least(k(0.06));
    let down_lower = k::<T>(1.0) - k::<T>(c8) * (k::<T>(1.0) - b.lower);
    let y_star = y_table - down_star * crown_height;
    let y_lower = g_bot - down_lower * depth;

    let mut facets = vec![Facet { tier: "table", plane: Plane::below(y_table) }];
    for p in &bezel { facets.push(Facet { tier: "bezel", plane: *p }); }
    let mut upper = Vec::new();
    for i in 0..n {
        let j = (i + 1) % n;
        let m = meet(bezel[i], bezel[j], Plane::below(y_star));
        facets.push(Facet { tier: "star", plane: Plane::through(corner(i), corner(j), m) });
        let h = at3(r.mid[i], g_top);
        upper.push(Plane::through(at3(r.at[i].0, g_top), m, h));
        upper.push(Plane::through(at3(r.at[j].0, g_top), h, m));
    }
    for p in upper { facets.push(Facet { tier: "upper girdle", plane: p }); }
    for p in &main { facets.push(Facet { tier: "pavilion main", plane: *p }); }
    for i in 0..n {
        let j = (i + 1) % n;
        let low = meet(main[i], main[j], Plane::below(y_lower));
        let h = at3(r.mid[i], g_bot);
        facets.push(Facet { tier: "lower girdle", plane: Plane::through(at3(r.at[i].0, g_bot), h, low) });
        facets.push(Facet { tier: "lower girdle", plane: Plane::through(at3(r.at[j].0, g_bot), low, h) });
    }
    // the culet: the pavilion's point cut off flat, where the mains are this far apart
    let y_flat = y_culet + half * b.culet * width * deg(b.pavilion).tan();
    facets.push(Facet { tier: "culet", plane: Plane::above(y_flat) });
    for (m, d) in &r.walls {
        facets.push(Facet { tier: "girdle", plane: Plane { n: [m[0], zero, m[1]], h: *d } });
    }
    Stone { facets, rim: r.rim, lift: half * (y_table + y_flat) }
}

/// A step cut: a rectangle with its corners cut off, and rows of long facets parallel to its sides,
/// steepest at the girdle and flatter towards the table and towards the keel. Every row is the outline
/// moved inwards by the same amount on all eight sides, so the rows are concentric and each row's facets
/// all slope alike. `crown` and `pavilion`: how steep each row is relative to the others, from the
/// girdle outwards (the rows share the height between them in these proportions).
///
/// sliders: table (of the width), crown height and pavilion depth (of the width), girdle, length / width,
/// and how much of the half-width each corner loses.
fn step_cut<T: Num>(v: &[T], crown: &[f64], pavilion: &[f64]) -> Stone<T> {
    let (table, crown_h, pav_d, girdle, ratio, corner) = (v[0], v[1], v[2], v[3], v[4], v[5]);
    // the corners of the uncut rectangle reach REACH from the centre
    let a = k::<T>(REACH) / (k::<T>(1.0) + k::<T>(1.0) / (ratio * ratio)).sqrt();
    let b = a / ratio;
    let width = k::<T>(2.0) * b;
    let s = 0.5f64.sqrt();
    // the eight sides of the outline: outward direction and distance from the centre
    let sides: Vec<([f64; 2], T)> = [(1.0, 0.0), (s, s), (0.0, 1.0), (-s, s), (-1.0, 0.0), (-s, -s), (0.0, -1.0), (s, -s)].iter()
        .map(|&(x, z)| ([x, z], if z == 0.0 { a } else if x == 0.0 { b } else { (a + b - corner * b) * k(s) }))
        .collect();
    let (g_top, g_bot) = (k::<T>(0.5) * girdle * width, -k::<T>(0.5) * girdle * width);
    // rows from the girdle to the table (up = 1) or to the keel (up = -1): `reach` is how far in the last
    // row ends, `rise` how far up or down
    let rows = |weights: &[f64], reach: T, rise: T, start: T, up: f64, name: [&'static str; 2]| {
        let total: f64 = weights.iter().sum();
        let (mut inset, mut y) = (k::<T>(0.0), start);
        let mut out = Vec::new();
        for w in weights {
            let (step_in, step_up) = (reach * k(1.0 / weights.len() as f64), rise * k(w / total));
            for (m, d) in &sides {
                // through the row's outer edge, sloping in by step_in while it rises (or falls) by step_up
                let p = [k::<T>(m[0]) * (*d - inset), y, k::<T>(m[1]) * (*d - inset)];
                let diagonal = m[0] != 0.0 && m[1] != 0.0;
                out.push(Facet { tier: name[diagonal as usize], plane: Plane::facing([k::<T>(m[0]) * step_up, k::<T>(up) * step_in, k::<T>(m[1]) * step_up], p) });
            }
            inset = inset + step_in;
            y = y + k::<T>(up) * step_up;
        }
        out
    };
    let mut facets = vec![Facet { tier: "table", plane: Plane::below(g_top + crown_h * width) }];
    facets.extend(rows(crown, (k::<T>(1.0) - table) * b, crown_h * width, g_top, 1.0, ["crown step", "crown corner"]));
    for (m, d) in &sides {
        facets.push(Facet { tier: if m[0] != 0.0 && m[1] != 0.0 { "girdle corner" } else { "girdle" }, plane: Plane { n: [k(m[0]), k(0.0), k(m[1])], h: *d } });
    }
    facets.extend(rows(pavilion, b, pav_d * width, g_bot, -1.0, ["pavilion step", "pavilion corner"]));
    Stone { facets, rim: Rim::None, lift: k::<T>(0.5) * (crown_h * width - pav_d * width) }
}

/// The princess cut: a square brilliant. Crown: a square table, two rows of facets along each side and a
/// kite from each corner of the table to the corner of the stone. Pavilion: a narrow row along the
/// girdle, then chevrons, which are facets that all hold the stone's corner and fan from it across to
/// the ridge down the middle of each side. The four pairs of last chevrons meet along the diagonals,
/// which is the cross seen through the table.
///
/// sliders: table and crown height (of the width), the angle of the pavilion measured along the diagonal
/// (corner to culet), girdle, and how much the middle ridge bows outwards (0 = a plain pyramid).
fn princess<T: Num>(v: &[T], chevrons: usize) -> Stone<T> {
    let (table, crown_h, pav, girdle, bow) = (v[0], v[1], v[2], v[3], v[4]);
    // (its corners are the furthest points, and stand well above the stone's middle: a little smaller than the others)
    let s = k::<T>(1.12 * 0.5f64.sqrt());
    let width = k::<T>(2.0) * s;
    let (g_top, g_bot) = (k::<T>(0.5) * girdle * width, -k::<T>(0.5) * girdle * width);
    let rise = crown_h * width;
    let depth = k::<T>(2.0f64.sqrt()) * s * deg(pav).tan();
    let (y_table, y_culet) = (g_top + rise, g_bot - depth);
    let (zero, one) = (k::<T>(0.0), k::<T>(1.0));
    // the crown's two rows: the lower takes 45% of the way in and 62% of the height
    let (way, share) = (0.45, 0.62);
    let run = (one - table) * s;
    // the ridge down the middle of a side, from the girdle (u = 1) to the culet (u = 0): height above the culet
    let ridge = |u: f64| depth * ((one - bow) * k(u) + bow * k(u * u));

    let mut facets = vec![Facet { tier: "table", plane: Plane::below(y_table) }];
    let mut pavilion = Vec::new();
    for q in 0..4 {
        // the side facing along `m`, and the direction along it (anticlockwise)
        let (c, sn) = ((PI / 2.0 * q as f64).cos().round(), (PI / 2.0 * q as f64).sin().round());
        let at = |out: T, y: T, side: T| [k::<T>(c) * out - k::<T>(sn) * side, y, k::<T>(sn) * out + k::<T>(c) * side];
        let m = |out: T, up: T| [k::<T>(c) * out, up, k::<T>(sn) * out];
        facets.push(Facet { tier: "crown break", plane: Plane::facing(m(k::<T>(share) * rise, k::<T>(way) * run), at(s, g_top, zero)) });
        facets.push(Facet { tier: "crown main", plane: Plane::facing(m(k::<T>(1.0 - share) * rise, k::<T>(1.0 - way) * run), at(table * s, y_table, zero)) });
        // the kite on the corner anticlockwise of this side: from the stone's corner to the table's
        let (corner, tc) = (at(s, g_top, s), at(table * s, y_table, table * s));
        let across = sub(at(zero, zero, one), at(one, zero, zero));
        facets.push(Facet { tier: "corner bezel", plane: Plane::facing(cross(sub(tc, corner), across), corner) });
        facets.push(Facet { tier: "girdle", plane: Plane { n: m(one, zero), h: s } });
        // the pavilion: the ridge's points, then a facet between each pair of them and a corner
        let point = |i: usize| { let u = 1.0 - i as f64 / chevrons as f64; at(s * k(u), y_culet + ridge(u), zero) };
        pavilion.push(Facet { tier: "pavilion break", plane: Plane::through(point(0), at(s, g_bot, s), point(1)) });
        for i in 1..chevrons {
            pavilion.push(Facet { tier: "chevron", plane: Plane::through(point(i), at(s, g_bot, s), point(i + 1)) });
            pavilion.push(Facet { tier: "chevron", plane: Plane::through(point(i), point(i + 1), at(s, g_bot, -s)) });
        }
    }
    facets.extend(pavilion);
    Stone { facets, rim: Rim::None, lift: k::<T>(0.5) * (y_table + y_culet) }
}

/// The rose cut: a flat base and a dome of triangles, with no table and no pavilion. Six triangles meet
/// at the top; under each a triangle points down to the girdle; twelve more fill in round the girdle:
/// twenty-four in all, the Dutch rose.
///
/// sliders: the dome's height (of the width), how far out the six upper corners stand (of the radius),
/// and how high they stand (of the height).
fn rose<T: Num>(v: &[T]) -> Stone<T> {
    let (height, out, up) = (v[0], v[1], v[2]);
    let r = k::<T>(RADIUS);
    let top = height * k::<T>(2.0) * r;
    let rim_h = k::<T>(0.03 * 2.0 * RADIUS);
    let (zero, n) = (k::<T>(0.0), 6);
    let on = |i: f64, rad: T, y: T| { let a = PI / 6.0 * i; [rad * k(a.cos()), y, rad * k(a.sin())] };
    let apex = [zero, top, zero];
    let mut facets = Vec::new();
    for i in 0..n {
        let (i0, i1) = (2.0 * i as f64, 2.0 * i as f64 + 2.0);
        let (a, b) = (on(i0, out * r, up * top), on(i1, out * r, up * top));
        facets.push(Facet { tier: "star", plane: Plane::through(a, apex, b) });
        facets.push(Facet { tier: "tooth", plane: Plane::through(a, b, on(i0 + 1.0, r, zero)) });
        facets.push(Facet { tier: "cross", plane: Plane::through(a, on(i0 + 1.0, r, zero), on(i0, r, zero)) });
        facets.push(Facet { tier: "cross", plane: Plane::through(b, on(i1, r, zero), on(i0 + 1.0, r, zero)) });
    }
    facets.push(Facet { tier: "base", plane: Plane::above(-rim_h) });
    Stone { facets, rim: Rim::Round { r }, lift: k::<T>(0.5) * (top - rim_h) }
}

// ---------------------------------------------------------------------------------------------------
// A cut: which recipe, and the symmetry its shader folds by

/// How the shader folds a point before measuring it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fold {
    /// n mirrors through the axis: everything lands in the wedge of azimuth 0 to 180/n degrees
    Dihedral(u32),
    /// mirrors in x and z: everything lands where x >= 0 and z >= 0
    Quadrant,
    /// one mirror: z >= 0
    MirrorZ,
}
impl Fold {
    /// Where a point lands.
    pub fn apply(self, p: V3<f64>) -> V3<f64> {
        match self {
            Fold::Dihedral(n) => {
                let w = PI / n as f64;
                let a = p[2].abs().atan2(p[0]);
                let b = (a - 2.0 * w * (a / (2.0 * w) + 0.5).floor()).abs();
                let r = p[0].hypot(p[2]);
                [r * b.cos(), p[1], r * b.sin()]
            }
            Fold::Quadrant => [p[0].abs(), p[1], p[2].abs()],
            Fold::MirrorZ => [p[0], p[1], p[2].abs()],
        }
    }
    /// Whether a plane facing along `n` is one the folded point must be measured against.
    pub fn keeps(self, n: V3<f64>) -> bool {
        const EPS: f64 = 1e-6;
        match self {
            Fold::Dihedral(m) => {
                if n[0].hypot(n[2]) < EPS { return true; }
                let a = n[2].atan2(n[0]);
                a > -EPS && a < PI / m as f64 + EPS
            }
            Fold::Quadrant => n[0] > -EPS && n[2] > -EPS,
            Fold::MirrorZ => n[2] > -EPS,
        }
    }
    fn wgsl(self) -> String {
        match self {
            Fold::Dihedral(n) => format!(
                "  let w = {:?};\n  let a = atan2(abs(p.z), p.x + 1e-9);\n  let b = abs(a - 2.0 * w * floor(a / (2.0 * w) + 0.5));\n  let r = length(p.xz);\n  return vec3f(r * cos(b), p.y, r * sin(b));",
                PI / n as f64),
            Fold::Quadrant => "  return vec3f(abs(p.x), p.y, abs(p.z));".to_string(),
            Fold::MirrorZ => "  return vec3f(p.x, p.y, abs(p.z));".to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Recipe {
    /// sliders: table %, crown °, pavilion °, girdle %, star %, lower half %, culet %, then the outline's
    /// own (length / width; the trillion's is how far its sides bow out). The cushion and the radiant
    /// have no star slider, and after length / width the cushion's exponent or the radiant's corners.
    Brilliant(Outline),
    /// sliders: table %, crown height %, pavilion depth %, girdle %, length / width, corner
    Step { crown: &'static [f64], pavilion: &'static [f64] },
    /// sliders: table %, crown height %, pavilion ° (along the diagonal), girdle %, bow
    Princess { chevrons: usize },
    /// sliders: height %, how far out and how high the upper corners stand
    Rose,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Cut {
    pub recipe: Recipe,
    pub fold: Fold,
}

impl Cut {
    /// The stone for these slider values (percentages as the sliders show them).
    pub fn stone<T: Num>(&self, v: &[T]) -> Stone<T> {
        let pc = |x: T| x * k(0.01);
        match &self.recipe {
            Recipe::Brilliant(outline) => {
                // an outline with two numbers of its own leaves no room for a star slider: its stars stay at half way
                let (star, next) = if matches!(outline, Outline::Cushion | Outline::Radiant) { (k(0.5), 4) } else { (pc(v[4]), 5) };
                let b = Brilliant { table: pc(v[0]), crown: v[1], pavilion: v[2], girdle: pc(v[3]), star, lower: pc(v[next]), culet: pc(v[next + 1]) };
                brilliant(*outline, &b, &v[next + 2..])
            }
            Recipe::Step { crown, pavilion } => step_cut(&[pc(v[0]), pc(v[1]), pc(v[2]), pc(v[3]), v[4], v[5]], crown, pavilion),
            Recipe::Princess { chevrons } => princess(&[pc(v[0]), pc(v[1]), v[2], pc(v[3]), v[4]], *chevrons),
            Recipe::Rose => rose(&[pc(v[0]), v[1], v[2]]),
        }
    }

    /// Which of the stone's facets the shader measures a folded point against: those facing into the
    /// fold's wedge, each once (mirror twins that face the same way count once).
    pub fn kept(&self, values: &[f64]) -> Vec<usize> {
        let stone = self.stone::<f64>(values);
        let mut kept: Vec<usize> = Vec::new();
        for (i, f) in stone.facets.iter().enumerate() {
            if !self.fold.keeps(f.plane.n) { continue; }
            let twin = kept.iter().any(|&j| {
                let g = &stone.facets[j].plane;
                (g.h - f.plane.h).abs() < 1e-9 && (0..3).all(|c| (g.n[c] - f.plane.n[c]).abs() < 1e-9)
            });
            if !twin { kept.push(i); }
        }
        kept
    }

    /// The stone as the shader measures it: fold the point, then the largest distance among the kept
    /// facets and the rim. Must equal `Stone::dist` everywhere.
    pub fn folded_dist(&self, values: &[f64], kept: &[usize], p: V3<f64>) -> f64 {
        let stone = self.stone::<f64>(values);
        let q = self.fold.apply([p[0], p[1] + stone.lift, p[2]]);
        kept.iter().fold(stone.rim_dist(q[0], q[2]), |d, &i| d.max(dot(stone.facets[i].plane.n, q) - stone.facets[i].plane.h))
    }

    /// The expressions the shader computes, in order: the lift, the rim's numbers, then four for each
    /// kept facet (its normal and its distance from the origin).
    pub fn expressions(&self, kept: &[usize], sliders: usize) -> (Stone<Sym>, Vec<Sym>) {
        Sym::reset();
        let v: Vec<Sym> = (0..sliders).map(Sym::slider).collect();
        let stone = self.stone::<Sym>(&v);
        let mut roots = vec![stone.lift];
        match stone.rim {
            Rim::None => {}
            Rim::Round { r } => roots.push(r),
            Rim::Squircle { a, b, e } => roots.extend([a, b, e]),
            Rim::Lens { rho, c } => roots.extend([rho, c]),
            Rim::Pear { x0, w, rho, c } => roots.extend([x0, w, rho, c]),
            Rim::Triangle { rho, c } => roots.extend([rho, c]),
        }
        for &i in kept {
            let p = &stone.facets[i].plane;
            roots.extend([p.n[0], p.n[1], p.n[2], p.h]);
        }
        (stone, roots)
    }

    /// The WGSL of the stone: `raw(p)`, with the planes worked out from the sliders once per voxel.
    pub fn wgsl(&self, params: &[ParamDef]) -> String {
        let defaults: Vec<f64> = params.iter().map(|d| d.value).collect();
        let kept = self.kept(&defaults);
        let (stone, roots) = self.expressions(&kept, params.len());
        let names: Vec<String> = params.iter().map(|d| member_name(d.id)).collect();
        let (lets, named) = Sym::wgsl(&roots, &names);
        let (rim_vars, rim_body) = match stone.rim {
            Rim::None => (0, String::new()),
            Rim::Round { .. } => (1, "  d = length(q.xz) - gem_rim0;\n".to_string()),
            Rim::Squircle { .. } => (3, "  let ab = vec2f(gem_rim0, gem_rim1);\n  let uv = abs(q.xz) / ab;\n  let ue = pow(uv, vec2f(gem_rim2 - 1.0));\n  let s = max(dot(ue, uv), 1e-9);\n  d = (pow(s, 1.0 / gem_rim2) - 1.0) / max(pow(s, 1.0 / gem_rim2 - 1.0) * length(ue / ab), 1e-6);\n".to_string()),
            Rim::Lens { .. } => (2, "  d = length(vec2f(q.x, q.z + gem_rim1)) - gem_rim0;\n".to_string()),
            Rim::Pear { .. } => (4, "  d = select(length(vec2f(q.x - gem_rim0, q.z + gem_rim3)) - gem_rim2, length(vec2f(q.x - gem_rim0, q.z)) - gem_rim1, q.x < gem_rim0);\n".to_string()),
            // folded into the wedge from a corner (azimuth 0) to the middle of a side (60 degrees): that side's arc
            Rim::Triangle { .. } => (2, "  d = length(q.xz + gem_rim1 * vec2f(0.5, 0.8660254)) - gem_rim0;\n".to_string()),
        };
        let mut s = String::from("// the stone's planes (normal, distance from the origin), worked out from the sliders once per voxel\nvar<private> gem_ready: bool = false;\nvar<private> gem_lift: f32;\n");
        for i in 0..rim_vars { s += &format!("var<private> gem_rim{i}: f32;\n"); }
        for i in 0..kept.len() { s += &format!("var<private> gem_f{i}: vec4f;\n"); }
        s += "fn gem_setup() {\n";
        s += &lets;
        s += &format!("  gem_lift = {};\n", named[0]);
        for i in 0..rim_vars { s += &format!("  gem_rim{i} = {};\n", named[1 + i]); }
        for i in 0..kept.len() {
            let at = 1 + rim_vars + 4 * i;
            s += &format!("  gem_f{i} = vec4f({}, {}, {}, {});      // {}\n", named[at], named[at + 1], named[at + 2], named[at + 3], stone.facets[kept[i]].tier);
        }
        s += "  gem_ready = true;\n}\n";
        s += &format!("// every point to its mirror image in the one wedge of the stone that is measured\nfn gem_fold(p: vec3f) -> vec3f {{\n{}\n}}\n", self.fold.wgsl());
        s += "fn raw(p: vec3f) -> f32 {\n  if (!gem_ready) { gem_setup(); }\n  let q = gem_fold(vec3f(p.x, p.y + gem_lift, p.z));\n  var d = -1e9;\n";
        s += &rim_body;
        for i in 0..kept.len() { s += &format!("  d = max(d, dot(q, gem_f{i}.xyz) - gem_f{i}.w);\n"); }
        s += "  return d;\n}";
        s
    }
}

// ---------------------------------------------------------------------------------------------------
// The stone as the tracer meets it from inside

/// The most planes `inside` gives (the tracer's table of them has this many places).
pub const MAX_PLANES: usize = 124;

/// What the tracer needs to find, without marching, where a ray inside a stone leaves it: every facet's
/// plane in the object's own frame, and the band of heights within which the rim's curved wall can be
/// met. A stone is convex, so a ray inside it leaves through the first plane it reaches, unless the
/// rim cuts it off first; that can happen only where the ray ends within the rim's band of heights,
/// because outside the rim the facets' planes only close in further above and below it.
pub struct Inside {
    /// (n, h): the half space n·p <= h, n of length 1 pointing out; each plane once
    pub planes: Vec<[f64; 4]>,
    /// lowest and highest point of the rim's wall (an empty band, low above high, when there is none)
    pub band: (f64, f64),
}

/// The stone of the gem preset `id` at these slider values, swollen by `puff` as the baked field is,
/// or None if `id` is not a gem or it has too many planes.
pub fn inside(id: &str, values: &[f64], puff: f64) -> Option<Inside> {
    let gem = gems().into_iter().find(|g| g.id == id)?;
    let v: Vec<f64> = gem.params.iter().enumerate().map(|(i, d)| values.get(i).copied().unwrap_or(d.value)).collect();
    let stone = gem.cut.stone::<f64>(&v);
    // the field is the stone moved down by its lift (d(p) measures p + lift up) and swollen by puff
    let mut planes: Vec<[f64; 4]> = Vec::new();
    for f in &stone.facets {
        let n = f.plane.n;
        let p = [n[0], n[1], n[2], f.plane.h - stone.lift * n[1] + puff];
        if !planes.iter().any(|q| (0..4).all(|c| (q[c] - p[c]).abs() < 1e-7)) { planes.push(p); }
    }
    if planes.len() > MAX_PLANES { return None; }
    let mut band = (f64::INFINITY, f64::NEG_INFINITY);
    if !matches!(stone.rim, Rim::None) {
        // round the outline: where the wall stands at each bearing, then the heights the planes leave it
        for i in 0..720 {
            let a = i as f64 / 720.0 * 2.0 * PI;
            let (c, s) = (a.cos(), a.sin());
            let lo = match stone.rim {
                // (a squircle's wall in closed form: its distance takes powers, slow to search with)
                Rim::Squircle { a, b, e } => ((c.abs() / a).powf(e) + (s.abs() / b).powf(e)).powf(-1.0 / e),
                _ => {
                    let (mut lo, mut hi) = (0.0, 4.0);
                    for _ in 0..18 {
                        let r = 0.5 * (lo + hi);
                        if stone.rim_dist(r * c, r * s) < 0.0 { lo = r } else { hi = r }
                    }
                    lo
                }
            };
            let (x, z) = (lo * c, lo * s);
            let (mut y0, mut y1) = (f64::NEG_INFINITY, f64::INFINITY);
            for p in &planes {
                let rest = p[3] - p[0] * x - p[2] * z;
                if p[1] > 1e-9 { y1 = y1.min(rest / p[1]) } else if p[1] < -1e-9 { y0 = y0.max(rest / p[1]) } else if rest < 0.0 { y1 = f64::NEG_INFINITY }
            }
            if y0 <= y1 { band = (band.0.min(y0), band.1.max(y1)); }
        }
        // (the bearings are half a degree apart, the wall found to within 2e-5: a little more either way
        // covers what falls between)
        if band.0 <= band.1 { band = (band.0 - puff.abs() - 0.01, band.1 + puff.abs() + 0.01); }
    }
    Some(Inside { planes, band })
}

/// The tracer's numbers for `inside`: how many planes, the rim's band, then the planes.
pub fn inside_uniforms(s: &Inside) -> Vec<f32> {
    let mut u = vec![0.0f32; 4 * (1 + MAX_PLANES)];
    let (lo, hi) = if s.band.0 <= s.band.1 { s.band } else { (1e9, -1e9) };
    u[..4].copy_from_slice(&[s.planes.len() as f32, lo as f32, hi as f32, 0.0]);
    for (i, p) in s.planes.iter().enumerate() {
        for c in 0..4 { u[4 + 4 * i + c] = p[c] as f32; }
    }
    u
}

// ---------------------------------------------------------------------------------------------------
// The presets

/// A gem preset before it is a `Preset`: kept so that the tests can ask a cut for its planes.
pub struct Gem {
    pub id: &'static str,
    pub name: &'static str,
    pub eq: Vec<&'static str>,
    pub note: &'static str,
    pub params: Vec<ParamDef>,
    pub rot: [f64; 3],
    pub animate: &'static str,
    pub cut: Cut,
}

fn pct(id: &'static str, label: &'static str, value: f64, lo: f64, hi: f64) -> ParamDef {
    ParamDef::labelled(id, label, value, lo, hi, 0.5)
}

/// The sliders every brilliant has. Ranges are those found in real stones, from poor to poor.
fn brilliant_params(table: f64, crown: f64, pavilion: f64, girdle: f64, star: f64, lower: f64, culet: f64) -> Vec<ParamDef> {
    vec![
        pct("table", "table %", table, 44.0, 72.0),
        ParamDef::labelled("crown", "crown angle °", crown, 24.0, 42.0, 0.25),
        ParamDef::labelled("pav", "pavilion angle °", pavilion, 37.0, 45.0, 0.05),
        ParamDef::labelled("girdle", "girdle %", girdle, 0.5, 8.0, 0.1),
        pct("star", "star length %", star, 35.0, 70.0),
        pct("lower", "lower half %", lower, 40.0, 90.0),
        ParamDef::labelled("culet", "culet %", culet, 0.0, 14.0, 0.2),
    ]
}
fn step_params(table: f64, crown: f64, pavilion: f64, girdle: f64, ratio: (f64, f64, f64), corner: f64) -> Vec<ParamDef> {
    vec![
        pct("table", "table %", table, 50.0, 78.0),
        pct("crown", "crown height %", crown, 6.0, 20.0),
        pct("pav", "pavilion depth %", pavilion, 36.0, 56.0),
        ParamDef::labelled("girdle", "girdle %", girdle, 0.5, 8.0, 0.1),
        ParamDef::labelled("ratio", "length / width", ratio.0, ratio.1, ratio.2, 0.01),
        ParamDef::labelled("corner", "corners", corner, 0.0, 0.6, 0.01),
    ]
}
fn ratio(value: f64, lo: f64, hi: f64) -> ParamDef {
    ParamDef::labelled("ratio", "length / width", value, lo, hi, 0.01)
}

/// How a stone first appears: its table tipped 58 degrees towards the viewer.
const TILTED: [f64; 3] = [58.0, 0.0, 0.0];
/// The same for a stone with a point (which is cut along +x): turned so that the point is at the top.
const POINT_UP: [f64; 3] = [90.0, 32.0, 90.0];

const BRILLIANT_EQ: [&str; 3] = ["d(p) = maxᵢ (nᵢ · p − hᵢ)   over the facets", "table · 8 bezels · 8 stars · 16 upper halves", "8 pavilion mains · 16 lower halves · culet"];

/// Every cut, in the order shown.
pub fn gems() -> Vec<Gem> {
    let with = |mut base: Vec<ParamDef>, more: Vec<ParamDef>| { base.extend(more); base };
    vec![
        Gem {
            id: "brilliant", name: "Round brilliant", eq: BRILLIANT_EQ.to_vec(),
            note: "57 facets and a culet, in the proportions Marcel Tolkowsky worked out in 1919: light that enters through the crown is turned back by the pavilion and leaves through the crown again. Move the pavilion angle a degree either way and watch the light leak out.",
            params: brilliant_params(57.0, 34.5, 40.75, 3.5, 50.0, 77.0, 0.6),
            rot: TILTED, animate: "pav",
            cut: Cut { recipe: Recipe::Brilliant(Outline::Round), fold: Fold::Dihedral(8) },
        },
        Gem {
            id: "oval", name: "Oval", eq: BRILLIANT_EQ.to_vec(),
            note: "The brilliant stretched along one axis. The dark bow-tie across the middle is real: the long facets there cannot turn light back.",
            params: with(brilliant_params(56.0, 35.0, 40.5, 3.5, 50.0, 77.0, 0.6), vec![ratio(1.4, 1.1, 1.8)]),
            rot: TILTED, animate: "ratio",
            cut: Cut { recipe: Recipe::Brilliant(Outline::Oval), fold: Fold::Quadrant },
        },
        Gem {
            id: "marquise", name: "Marquise", eq: BRILLIANT_EQ.to_vec(),
            note: "A brilliant on an outline of two arcs: the navette, or little boat.",
            params: with(brilliant_params(56.0, 35.0, 40.5, 3.5, 50.0, 77.0, 0.6), vec![ratio(1.95, 1.5, 2.4)]),
            rot: TILTED, animate: "ratio",
            cut: Cut { recipe: Recipe::Brilliant(Outline::Marquise), fold: Fold::Quadrant },
        },
        Gem {
            id: "pear", name: "Pear", eq: BRILLIANT_EQ.to_vec(),
            note: "Half an oval, half a marquise.",
            params: with(brilliant_params(56.0, 35.0, 40.5, 3.5, 50.0, 77.0, 0.6), vec![ratio(1.55, 1.3, 1.9)]),
            rot: POINT_UP, animate: "ratio",
            cut: Cut { recipe: Recipe::Brilliant(Outline::Pear), fold: Fold::MirrorZ },
        },
        Gem {
            id: "cushion", name: "Cushion", eq: BRILLIANT_EQ.to_vec(),
            note: "A brilliant on a square with bowed sides and round corners: |x/a|ᵉ + |z/b|ᵉ = 1.",
            params: {
                let mut p = brilliant_params(58.0, 36.0, 41.0, 3.5, 50.0, 77.0, 1.0);
                p.remove(4);        // eight sliders at most: the star length stays at 50%
                with(p, vec![ratio(1.08, 1.0, 1.5), ParamDef::labelled("square", "squareness e", 3.0, 2.0, 6.0, 0.05)])
            },
            rot: TILTED, animate: "square",
            cut: Cut { recipe: Recipe::Brilliant(Outline::Cushion), fold: Fold::Quadrant },
        },
        Gem {
            id: "oldmine", name: "Old mine cut", eq: BRILLIANT_EQ.to_vec(),
            note: "The cushion as it was cut by eye before 1900, for candlelight: a small table, a high crown, a deep pavilion, short lower halves and a culet large enough to see through the table.",
            params: {
                let mut p = brilliant_params(46.0, 40.0, 43.0, 3.0, 50.0, 50.0, 9.0);
                p.remove(4);
                with(p, vec![ratio(1.06, 1.0, 1.5), ParamDef::labelled("square", "squareness e", 3.4, 2.0, 6.0, 0.05)])
            },
            rot: TILTED, animate: "culet",
            cut: Cut { recipe: Recipe::Brilliant(Outline::Cushion), fold: Fold::Quadrant },
        },
        Gem {
            id: "radiant", name: "Radiant", eq: vec!["d(p) = maxᵢ (nᵢ · p − hᵢ)   over the facets", "a brilliant's facets on an emerald cut's outline"],
            note: "Henry Grossbard’s idea of 1977: the outline of an emerald cut with the fire of a brilliant.",
            params: {
                let mut p = brilliant_params(62.0, 34.0, 41.0, 3.0, 50.0, 75.0, 0.6);
                p.remove(4);
                with(p, vec![ratio(1.25, 1.0, 1.8), ParamDef::labelled("corner", "corners", 0.3, 0.1, 0.6, 0.01)])
            },
            rot: TILTED, animate: "corner",
            cut: Cut { recipe: Recipe::Brilliant(Outline::Radiant), fold: Fold::Quadrant },
        },
        Gem {
            id: "trillion", name: "Trillion", eq: vec!["d(p) = maxᵢ (nᵢ · p − hᵢ)   over the facets", "a brilliant on a triangle of three arcs:", "6 mains · 6 stars · 12 + 12 halves"],
            note: "A triangular brilliant. Straighten its sides and it is the trilliant the Asscher brothers cut; bow them and it nears a round.",
            params: with(brilliant_params(58.0, 34.0, 41.0, 3.0, 50.0, 75.0, 0.6), vec![ParamDef::labelled("bow", "side bow", 0.66, 0.56, 0.86, 0.01)]),
            rot: POINT_UP, animate: "bow",
            cut: Cut { recipe: Recipe::Brilliant(Outline::Trillion), fold: Fold::Dihedral(3) },
        },
        Gem {
            id: "princess", name: "Princess", eq: vec!["d(p) = maxᵢ (nᵢ · p − hᵢ)   over the facets", "a square brilliant: chevrons fanning from", "the four corners to the ridge of each side"],
            note: "The cross seen through the table is where the chevrons from neighbouring sides meet along the diagonals.",
            params: vec![
                pct("table", "table %", 68.0, 56.0, 78.0),
                pct("crown", "crown height %", 10.0, 5.0, 16.0),
                ParamDef::labelled("pav", "pavilion angle °", 40.0, 35.0, 45.0, 0.05),
                ParamDef::labelled("girdle", "girdle %", 2.5, 0.5, 8.0, 0.1),
                ParamDef::labelled("bow", "chevron bow", 0.35, 0.0, 0.7, 0.01),
            ],
            rot: TILTED, animate: "bow",
            cut: Cut { recipe: Recipe::Princess { chevrons: 3 }, fold: Fold::Dihedral(4) },
        },
        Gem {
            id: "emerald", name: "Emerald cut", eq: vec!["d(p) = maxᵢ (nᵢ · p − hᵢ)   over the facets", "three rows of steps above the girdle,", "three below, on a rectangle with cut corners"],
            note: "A step cut: long facets parallel to the girdle, which act as mirrors rather than as prisms. It shows the stone itself, flaws and all, as a hall of mirrors.",
            params: step_params(65.0, 13.0, 47.0, 3.0, (1.4, 1.0, 2.0), 0.3),
            rot: TILTED, animate: "ratio",
            cut: Cut { recipe: Recipe::Step { crown: &[1.45, 1.0, 0.6], pavilion: &[1.45, 1.0, 0.6] }, fold: Fold::Quadrant },
        },
        Gem {
            id: "asscher", name: "Asscher cut", eq: vec!["d(p) = maxᵢ (nᵢ · p − hᵢ)   over the facets", "a square step cut with wide corners:", "nearly an octagon"],
            note: "Joseph Asscher’s cut of 1902: a square emerald cut with a high crown and deep corners. Looked into from above, the steps repeat inwards like a corridor.",
            params: step_params(60.0, 15.0, 48.0, 3.0, (1.0, 1.0, 1.2), 0.44),
            rot: TILTED, animate: "corner",
            cut: Cut { recipe: Recipe::Step { crown: &[1.45, 1.0, 0.6], pavilion: &[1.45, 1.0, 0.6] }, fold: Fold::Quadrant },
        },
        Gem {
            id: "baguette", name: "Baguette", eq: vec!["d(p) = maxᵢ (nᵢ · p − hᵢ)   over the facets", "a long rectangle with two rows of steps"],
            note: "The simplest step cut: a bar, named after the loaf.",
            params: step_params(70.0, 10.0, 42.0, 3.0, (2.6, 1.5, 3.5), 0.0),
            rot: TILTED, animate: "ratio",
            cut: Cut { recipe: Recipe::Step { crown: &[1.3, 0.7], pavilion: &[1.3, 0.7] }, fold: Fold::Quadrant },
        },
        Gem {
            id: "rose", name: "Rose cut", eq: vec!["d(p) = maxᵢ (nᵢ · p − hᵢ)   over the facets", "a flat base under a dome of 24 triangles"],
            note: "Cut since the 1500s: no table and no pavilion, so it glows rather than sparkles.",
            params: vec![
                pct("height", "height %", 32.0, 20.0, 50.0),
                ParamDef::labelled("out", "upper ring radius", 0.55, 0.4, 0.7, 0.01),
                ParamDef::labelled("up", "upper ring height", 0.68, 0.5, 0.85, 0.01),
            ],
            rot: [50.0, 0.0, 0.0], animate: "height",
            cut: Cut { recipe: Recipe::Rose, fold: Fold::Dihedral(6) },
        },
    ]
}

pub fn family() -> Family {
    let presets = gems().into_iter().map(|g| {
        let wgsl = g.cut.wgsl(&g.params);
        Preset {
            eq: g.eq,
            note: g.note,
            params: g.params,
            rot: Some(g.rot),
            animate: Some(vec![g.animate]),
            // the stone is its own boundary: nothing may round or shave it
            ..Preset::new(g.id, g.name, Shape::Field { wgsl, clip: "-1.0".to_string(), shell: crate::field::SHELL_DEFAULT.to_string() })
        }
    }).collect();
    Family {
        presets,
        // Inside the stone the stored distance is exact; outside, near edges and corners, the walk to the
        // surface can stop a little short of the nearest point and overstate the distance, so rays take
        // slightly shorter steps than the distance read.
        step_scale: 0.8,
        steps_out: 160,
        // light is turned about inside a diamond many times, often skimming a facet: give it the steps
        steps_in: 96,
        // a convex stone: a ray that has left it never meets it again
        layers: 0,
        defaults: vec![
            // diamond: n = 2.42 at 550 nm, and n(400 nm) − n(700 nm) = 0.058 (2.464 − 2.406, Peter's fit)
            ("ior", Setting::Num(2.42)), ("disp", Setting::Num(0.058)), ("absorb", Setting::Vec3([0.0, 0.0, 0.0])),
            ("ns", Setting::Num(1.0)), ("nw", Setting::Num(3.0)), ("nb", Setting::Num(8.0)),
            ("dist", Setting::Num(6.6)), ("fov", Setting::Num(26.0)), ("rot", Setting::Vec3([58.0, 0.0, 0.0])),
            ("spin", Setting::Num(0.25)), ("bloom", Setting::Num(0.5)), ("exposure", Setting::Num(1.0)),
            // while a stone animates it is built at 384 voxels a side, not 256: flat facets show their steps
            ("animRes", Setting::Num(384.0)),
        ],
        ambient: [0.006, 0.007, 0.011],
        lights: lights(),
        ..Family::new("gems", "Gem cuts", Kind::Field)
    }
}

/// A jeweller's lighting: many small bright lamps, so that every facet either catches one or does not
/// (sparkle and contrast), a few of them coloured, over a dim sky that keeps the dark facets from going
/// quite black.
fn lights() -> Vec<Light> {
    let lamp = |az: f64, el: f64, size: f64, rgb: [f64; 3]| Light { az, el, w: size, h: size, rgb, soft: 0.5 };
    let white = [30.0, 29.0, 27.0];
    vec![
        lamp(205.0, 62.0, 3.5, white),
        lamp(150.0, 48.0, 3.0, white),
        lamp(118.0, 70.0, 3.0, [26.0, 27.0, 30.0]),
        lamp(255.0, 40.0, 3.0, white),
        lamp(300.0, 66.0, 3.5, [30.0, 27.0, 22.0]),
        lamp(40.0, 55.0, 3.0, white),
        lamp(350.0, 35.0, 3.0, [22.0, 26.0, 32.0]),
        lamp(85.0, 28.0, 2.5, white),
        lamp(180.0, 20.0, 3.0, [30.0, 24.0, 18.0]),
        lamp(230.0, 82.0, 4.0, white),
        lamp(20.0, -30.0, 3.0, [16.0, 18.0, 24.0]),
        lamp(200.0, -45.0, 3.0, [24.0, 20.0, 16.0]),
        // two soft boxes, for the broad facets of the step cuts to flash with
        Light { az: 140.0, el: 32.0, w: 9.0, h: 14.0, rgb: [5.0, 5.0, 5.3], soft: 0.6 },
        Light { az: 262.0, el: 52.0, w: 12.0, h: 9.0, rgb: [4.6, 4.4, 4.0], soft: 0.6 },
        // the sky: broad and dim
        Light { az: 0.0, el: 90.0, w: 50.0, h: 50.0, rgb: [0.26, 0.29, 0.36], soft: 1.0 },
    ]
}
