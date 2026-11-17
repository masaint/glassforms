//! Trees, grown rather than drawn.
//!
//! One shape, with a choice of `form`: the architecture of the tree, which changes how it grows. Every
//! other slider acts within each form.
//!
//!   broadleaf  space colonisation (Runions, Lane and Prusinkiewicz, 2007): points scattered through the
//!              crown are the free space the shoots compete for; every point pulls on the nearest bud
//!              within reach, a bud pulled grows a step towards the mean of its pulls, a point is used up
//!              when a bud comes close. The trunk divides into a few limbs; a broad crown.
//!   conifer    a straight leader to the top, branches in whorls up it, each tier shorter than the one
//!              below, near level and drooping at their ends, with side shoots in their plane and needles
//!              along every twig: a cone.
//!   columnar   a leader to the top and, all up it, branches that leave at a steep angle and turn up to
//!              run beside the stem, longest in the middle: a narrow spindle.
//!   weeping    a broad crown of arching limbs from which long thin whips hang.
//!   palm       an unbranched trunk, bending, and a crown of arching fronds with leaflets down both sides.
//!
//! Thickness is the pipe model (Shinozaki; da Vinci's rule): a branch's radius to the power n is the sum
//! of its children's, so the trunk is as thick as all its twigs together.
//!
//! Leaves are flat blades (`Blade`): an outline with a tip and a rounded base, a stalk, a slight cup,
//! turned about the twig; needles are fine tubes in brushes along the twigs.
//!
//! Growth is the slow part (some milliseconds), so the grown tree is kept for its sliders and only `grow`,
//! which cuts it, is worked out afresh as it moves.

use super::*;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

pub const FAMILY: &str = "trees";

/// The forms, in the order of the `form` slider's whole numbers.
pub const FORMS: [&str; 5] = ["broadleaf", "conifer", "columnar", "weeping", "palm"];
const BROADLEAF: usize = 0;
const CONIFER: usize = 1;
const COLUMNAR: usize = 2;
const WEEPING: usize = 3;
const PALM: usize = 4;

/// One step of growth, in units of the tree's height.
const STEP: f64 = 0.0105;
/// A free point pulls on buds this near (in steps)...
const REACH: f64 = 13.0;
/// ... and is used up by one this near.
const KILL: f64 = 2.4;
/// A side shoot leaves a stem no nearer than this many steps to another fork on it (buds are spaced).
const SPACING: usize = 3;
/// Growth stops after this many rounds whatever happens.
const ROUNDS: usize = 700;

// ---- the sliders, in the order the preset lists them ----------------------------------------------
const SEED: usize = 0;
const GROW: usize = 1;
const FORM: usize = 2;
const BOLE: usize = 3;
const WIDTH: usize = 4;
const SHAPE: usize = 5;
const LUMPS: usize = 6;
const TWIGS: usize = 7;
const ANGLE: usize = 8;
const LEADER: usize = 9;
const UPTURN: usize = 10;
const LIGHT: usize = 11;
const DROOP: usize = 12;
const WIGGLE: usize = 13;
const LEAN: usize = 14;
const TAPER: usize = 15;
const TRUNK: usize = 16;
const FLARE: usize = 17;
const ROOTS: usize = 18;
const LEAVES: usize = 19;
const LEAF: usize = 20;

fn params() -> Vec<ParamDef> {
    vec![
        ParamDef::labelled("seed", "seed", 1.0, 1.0, 999.0, 1.0),
        ParamDef::labelled("grow", "grown", 1.0, 0.0, 1.0, 0.002),
        ParamDef::labelled("form", "form", 0.0, 0.0, (FORMS.len() - 1) as f64, 1.0),
        ParamDef::labelled("bole", "trunk length", 0.5, 0.0, 2.0, 0.005),
        ParamDef::labelled("width", "crown width", 1.3, 0.2, 3.0, 0.005),
        ParamDef::labelled("shape", "crown pointed or flat", 0.0, -1.0, 1.0, 0.01),
        ParamDef::labelled("lumps", "lumpy crown", 0.4, 0.0, 1.0, 0.01),
        ParamDef::labelled("twigs", "twigs", 1100.0, 150.0, 4000.0, 10.0),
        ParamDef::labelled("angle", "branch angle", 38.0, 5.0, 85.0, 0.5),
        ParamDef::labelled("leader", "leader", 0.35, 0.0, 1.0, 0.01),
        ParamDef::labelled("upturn", "turn up", 0.15, -1.0, 1.0, 0.01),
        ParamDef::labelled("light", "reach for light", 0.25, 0.0, 1.0, 0.01),
        ParamDef::labelled("droop", "droop", 0.0, 0.0, 1.0, 0.01),
        ParamDef::labelled("wiggle", "wiggle", 0.2, 0.0, 1.0, 0.01),
        ParamDef::labelled("lean", "trunk lean", 0.0, 0.0, 1.0, 0.01),
        ParamDef::labelled("taper", "taper", 2.4, 1.8, 3.2, 0.01),
        ParamDef::labelled("trunk", "trunk thickness", 0.03, 0.01, 0.07, 0.0005),
        ParamDef::labelled("flare", "flare", 0.7, 0.0, 2.0, 0.01),
        ParamDef::labelled("roots", "roots", 5.0, 0.0, 9.0, 1.0),
        ParamDef::labelled("leaves", "leaves", 0.0, 0.0, 0.6, 0.01),
        ParamDef::labelled("leaf", "leaf size", 1.0, 0.3, 2.5, 0.01),
    ]
}

// ---- small tools -----------------------------------------------------------------------------------
type V = [f64; 3];
fn add(a: V, b: V) -> V { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
fn sub(a: V, b: V) -> V { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn mul(a: V, s: f64) -> V { [a[0] * s, a[1] * s, a[2] * s] }
fn dot(a: V, b: V) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn cross(a: V, b: V) -> V { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }
fn len(a: V) -> f64 { dot(a, a).sqrt() }
fn unit(a: V) -> V {
    let l = len(a);
    if l > 1e-12 { mul(a, 1.0 / l) } else { [0.0, 1.0, 0.0] }
}
fn lerp(a: V, b: V, t: f64) -> V { add(a, mul(sub(b, a), t)) }
const UP: V = [0.0, 1.0, 0.0];
const DOWN: V = [0.0, -1.0, 0.0];
/// A unit vector at right angles to d (any one).
fn across(d: V) -> V {
    let t = if d[1].abs() < 0.9 { UP } else { [1.0, 0.0, 0.0] };
    unit(cross(d, t))
}
/// d turned towards straight down by an angle a, never past straight down.
fn bend_down(d: V, a: f64) -> V {
    let d = unit(d);
    let to = dot(d, DOWN).clamp(-1.0, 1.0).acos();
    let a = a.min(to * 0.97);
    let side = sub(DOWN, mul(d, dot(d, DOWN)));
    if len(side) < 1e-9 { d } else { unit(add(mul(d, a.cos()), mul(unit(side), a.sin()))) }
}
/// A direction from its turn about the vertical (az) and its height above level (el), radians.
fn heading(az: f64, el: f64) -> V { [el.cos() * az.cos(), el.sin(), el.cos() * az.sin()] }

/// A seeded generator (splitmix64): the same seed always grows the same tree.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x2545f4914f6cdd1d) }
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn sym(&mut self) -> f64 { 2.0 * self.next() - 1.0 }
    /// a direction, evenly over the sphere
    fn dir(&mut self) -> V {
        let z = self.sym();
        let a = 2.0 * PI * self.next();
        let s = (1.0 - z * z).max(0.0).sqrt();
        [s * a.cos(), z, s * a.sin()]
    }
}

/// Points kept in cells of a fixed size, to find the ones near a place without looking at them all.
/// Points outside the box are kept in its edge cells, which leaves every search right.
struct Grid {
    lo: V,
    cell: f64,
    n: [i32; 3],
    cells: Vec<Vec<u32>>,
}
impl Grid {
    fn new(lo: V, hi: V, cell: f64) -> Self {
        let n = [0, 1, 2].map(|k| (((hi[k] - lo[k]) / cell).ceil() as i32).clamp(1, 256));
        Self { lo, cell, n, cells: vec![Vec::new(); (n[0] * n[1] * n[2]) as usize] }
    }
    fn at(&self, p: V) -> [i32; 3] {
        [0, 1, 2].map(|k| (((p[k] - self.lo[k]) / self.cell).floor() as i32).clamp(0, self.n[k] - 1))
    }
    fn index(&self, c: [i32; 3]) -> usize { ((c[2] * self.n[1] + c[1]) * self.n[0] + c[0]) as usize }
    fn put(&mut self, p: V, i: u32) {
        let c = self.index(self.at(p));
        self.cells[c].push(i);
    }
    /// every point in the cells round p (enough for any search out to one cell's size)
    fn near(&self, p: V, mut f: impl FnMut(u32)) {
        let c = self.at(p);
        for z in (c[2] - 1).max(0)..=(c[2] + 1).min(self.n[2] - 1) {
            for y in (c[1] - 1).max(0)..=(c[1] + 1).min(self.n[1] - 1) {
                for x in (c[0] - 1).max(0)..=(c[0] + 1).min(self.n[0] - 1) {
                    for &i in &self.cells[self.index([x, y, z])] { f(i); }
                }
            }
        }
    }
}

/// What a node is part of, beyond plain wood.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Wood,
    /// a palm frond's midrib: it tapers to its tip, and bears leaflets
    Frond,
    /// a weeping whip: bears narrow leaves all along
    Whip,
}

/// One step of a shoot.
#[derive(Clone)]
struct Node {
    p: V,
    parent: u32,
    /// the way the shoot was heading when it grew this step
    dir: V,
    /// a slowly turning direction of the shoot's own, for its wandering
    wander: V,
    /// on the main stem, which keeps going upwards as strongly as `leader` says
    leader: bool,
    kids: u8,
    role: Role,
    /// what its radius is multiplied by (a frond tapers along itself, whatever the pipe model says)
    shrink: f64,
}
const NONE: u32 = u32::MAX;
impl Node {
    fn new(p: V, parent: u32, dir: V) -> Self {
        Self { p, parent, dir, wander: dir, leader: false, kids: 0, role: Role::Wood, shrink: 1.0 }
    }
}

/// A tree as grown from its sliders: everything `grow` does not change.
struct Grown {
    /// node positions after smoothing and drooping
    p: Vec<V>,
    parent: Vec<u32>,
    role: Vec<Role>,
    shrink: Vec<f64>,
    /// the length of the path from the foot to each node
    path: Vec<f64>,
    /// the strands the tree is drawn as: each a run of nodes that goes on through the thickest child at
    /// every fork; the first node of a strand that is not the trunk is the node it leaves from
    strands: Vec<Vec<u32>>,
    /// nodes with no children
    tips: usize,
    /// how many tips each node carries when the tree is grown
    carry: Vec<f64>,
    /// how many steps each node is from the nearest tip beyond it
    from_tip: Vec<u32>,
    /// the roots: each a line of points along the ground with the radius each has at full size, as a
    /// share of the trunk's
    roots: Vec<Vec<(V, f64)>>,
    /// every point of the whole tree: what it is placed by
    frame: Vec<f64>,
    /// the longest path from the foot
    longest: f64,
    /// the height the crown starts at
    base: f64,
    /// how far the trunk goes from the foot to its first fork
    trunk: f64,
}

/// The values a tree is grown from, as read from the sliders, with the form's own rules applied.
/// Lengths are in units of the whole tree's height.
#[derive(Clone)]
struct Sliders {
    seed: u64,
    form: usize,
    /// the height the crown starts at, and how tall the crown is
    base: f64,
    crown: f64,
    /// the crown's widest half-width
    half: f64,
    shape: f64,
    lumps: f64,
    twigs: usize,
    angle: f64,
    leader: f64,
    upturn: f64,
    light: f64,
    droop: f64,
    wiggle: f64,
    lean: f64,
    taper: f64,
    trunk: f64,
    flare: f64,
    roots: usize,
    leaves: f64,
    leaf: f64,
}
impl Sliders {
    fn read(v: &[f64]) -> Self {
        let g = |i: usize, or: f64| v.get(i).copied().filter(|x| x.is_finite()).unwrap_or(or);
        // trunk length and crown size are separate: the crown's height is the unit, the trunk is so many
        // crown heights long, and the whole is then brought to a height of one
        let bole = g(BOLE, 0.5).clamp(0.0, 4.0);
        let tall = 1.0 + bole;
        let mut s = Self {
            seed: g(SEED, 1.0).round().max(0.0) as u64,
            form: g(FORM, 0.0).round().clamp(0.0, (FORMS.len() - 1) as f64) as usize,
            base: bole / tall,
            crown: 1.0 / tall,
            half: 0.5 * g(WIDTH, 1.3).clamp(0.1, 4.0) / tall,
            shape: g(SHAPE, 0.0).clamp(-1.0, 1.0),
            lumps: g(LUMPS, 0.4).clamp(0.0, 1.0),
            twigs: g(TWIGS, 1100.0).clamp(20.0, 8000.0) as usize,
            angle: g(ANGLE, 38.0).clamp(0.0, 90.0).to_radians(),
            leader: g(LEADER, 0.35).clamp(0.0, 1.0),
            upturn: g(UPTURN, 0.15).clamp(-1.0, 1.0),
            light: g(LIGHT, 0.25).clamp(0.0, 1.0),
            droop: g(DROOP, 0.0).clamp(0.0, 1.0),
            wiggle: g(WIGGLE, 0.2).clamp(0.0, 1.0),
            lean: g(LEAN, 0.0).clamp(0.0, 1.0),
            taper: g(TAPER, 2.4).clamp(1.2, 4.0),
            trunk: g(TRUNK, 0.03).clamp(0.002, 0.2),
            flare: g(FLARE, 0.7).clamp(0.0, 3.0),
            roots: g(ROOTS, 5.0).round().clamp(0.0, 16.0) as usize,
            leaves: g(LEAVES, 0.0).clamp(0.0, 0.6),
            leaf: g(LEAF, 1.0).clamp(0.1, 4.0),
        };
        // the forms' own rules, laid over the sliders
        match s.form {
            COLUMNAR => {
                // a narrow spindle
                s.half *= 0.4;
            }
            WEEPING => {
                // arching limbs; the whips hang from them (see `whips`)
                s.upturn += 0.2;
                s.droop = (0.35 + 0.65 * s.droop).min(1.0);
            }
            _ => {}
        }
        s
    }
}

/// The crown's outline: its half-width at a height through it (t from 0 at its foot to 1 at its top), as a
/// share of the widest. `shape` -1 is pointed (widest low down, straight sides to a tip), 0 an ellipse,
/// 1 flat-topped (widest high up, a broad dome).
fn outline(t: f64, shape: f64) -> f64 {
    if !(0.0..=1.0).contains(&t) { return 0.0; }
    let peak = 0.5 + 0.36 * shape;
    if t < peak {
        let u = (peak - t) / peak;
        (1.0 - u * u).max(0.0).sqrt()
    } else {
        let u = (t - peak) / (1.0 - peak);
        let round = (1.0 - u * u).max(0.0).sqrt();
        if shape < 0.0 {
            // a cone's straight sides
            round + (1.0 - u - round) * (-shape)
        } else {
            // a flatter top
            round + ((1.0 - u.powi(4)).max(0.0).sqrt() - round) * shape
        }
    }
}

/// A smooth bumpiness over the crown: a few waves round it and up it, from the seed.
struct Lumps {
    waves: Vec<(f64, f64, f64, f64)>,
}
impl Lumps {
    fn new(r: &mut Rng) -> Self {
        let waves = (0..6).map(|i| (1.0 + (i % 3) as f64 + r.next().round(), 1.5 + 3.0 * r.next(), 2.0 * PI * r.next(), 0.6 + 0.4 * r.next())).collect();
        Self { waves }
    }
    /// about -1..1
    fn at(&self, az: f64, t: f64) -> f64 {
        let mut s = 0.0;
        for &(k, m, ph, a) in &self.waves { s += a * (k * az + m * t * PI + ph).sin(); }
        s / 2.2
    }
}

/// The trunk's lean: it rises straight and bends over more and more as it goes up (a curve, not a tilt).
/// The way it heads at height y.
struct Lean {
    az: f64,
    most: f64,
    top: f64,
}
impl Lean {
    fn dir(&self, y: f64) -> V {
        let a = self.most * (y / self.top.max(1e-6)).clamp(0.0, 1.0);
        let (s, c) = a.sin_cos();
        [s * self.az.cos(), c, s * self.az.sin()]
    }
    /// where the trunk has got to at height y (the centre of a leaning crown)
    fn at(&self, y: f64) -> V {
        let mut p = [0.0; 3];
        while p[1] < y {
            p = add(p, mul(self.dir(p[1]), STEP));
        }
        p
    }
}

fn grow_tree(s: &Sliders) -> Grown {
    let mut rng = Rng::new(s.seed.wrapping_mul(7919).wrapping_add(17));
    let lean = Lean { az: 2.0 * PI * rng.next(), most: s.lean * if s.form == PALM { 0.9 } else { 0.6 }, top: 1.0 };
    let nodes = match s.form {
        CONIFER => conifer(s, &lean, &mut rng),
        COLUMNAR => columnar(s, &lean, &mut rng),
        PALM => palm(s, &lean, &mut rng),
        _ => {
            let mut nodes = colonise(s, &lean, &mut rng);
            nodes = trim_hooks(nodes);
            if s.form == WEEPING { whips(s, &mut nodes, &mut rng); }
            nodes
        }
    };
    finish(s, nodes, &mut rng)
}

/// The trunk from the foot, leaning as `lean` says, up to height `top` (or until `stop` says so).
fn trunk(s: &Sliders, lean: &Lean, top: f64, rng: &mut Rng, mut stop: impl FnMut(V) -> bool) -> Vec<Node> {
    let mut nodes = vec![Node { leader: true, wander: rng.dir(), ..Node::new([0.0; 3], NONE, UP) }];
    let mut p = [0.0; 3];
    loop {
        let i = nodes.len() as u32 - 1;
        if p[1] >= top || stop(p) { break; }
        let wander = unit(add(mul(nodes[i as usize].wander, 0.85), mul(rng.dir(), 0.5)));
        let dir = unit(add(lean.dir(p[1]), mul(wander, 0.1 * s.wiggle)));
        p = add(p, mul(dir, STEP));
        nodes[i as usize].kids += 1;
        nodes.push(Node { wander, leader: true, ..Node::new(p, i, dir) });
    }
    nodes
}

/// Broadleaf, columnar and weeping trees: space colonisation in the crown's outline.
fn colonise(s: &Sliders, lean: &Lean, rng: &mut Rng) -> Vec<Node> {
    let lumps = Lumps::new(rng);
    let reach = REACH * STEP;
    let kill = KILL * STEP;
    // a leaning trunk carries its crown over with it
    let mid = lean.at(s.base);

    // the free points: scattered evenly through the crown's outline
    let inside = |p: V| -> bool {
        let t = (p[1] - s.base) / s.crown;
        let (x, z) = (p[0] - mid[0], p[2] - mid[2]);
        let r = (x * x + z * z).sqrt();
        let az = z.atan2(x);
        r <= s.half * outline(t, s.shape) * (1.0 + 0.45 * s.lumps * lumps.at(az, t))
    };
    let wmax = s.half * (1.0 + 0.45 * s.lumps) + 0.01;
    let mut free: Vec<V> = Vec::with_capacity(s.twigs);
    let mut tries = 0;
    while free.len() < s.twigs && tries < s.twigs * 200 {
        tries += 1;
        let p = [mid[0] + wmax * rng.sym(), s.base + s.crown * rng.next(), mid[2] + wmax * rng.sym()];
        if inside(p) { free.push(p); }
    }
    let (lo, hi) = ([mid[0] - wmax - reach, -0.2, mid[2] - wmax - reach], [mid[0] + wmax + reach, 1.3, mid[2] + wmax + reach]);
    let mut fgrid = Grid::new(lo, hi, reach);
    for (i, &p) in free.iter().enumerate() { fgrid.put(p, i as u32); }
    let mut alive = vec![true; free.len()];
    // the nearest node to each free point, and how far it is
    let mut nearest: Vec<(u32, f64)> = vec![(NONE, f64::INFINITY); free.len()];
    // a node has just grown: it takes the free points it is now the nearest to, and uses up those it reached
    let settle = |i: u32, p: V, alive: &mut Vec<bool>, nearest: &mut Vec<(u32, f64)>| {
        fgrid.near(p, |a| {
            let a = a as usize;
            if !alive[a] { return; }
            let d = len(sub(free[a], p));
            if d < kill { alive[a] = false; } else if d < nearest[a].1 { nearest[a] = (i, d); }
        });
    };

    // the trunk: up until it reaches the crown's free points
    let mut near = |p: V| free.iter().any(|f| len(sub(*f, p)) < reach);
    let mut nodes = trunk(s, lean, 1.0, rng, &mut near);
    for i in 0..nodes.len() { settle(i as u32, nodes[i].p, &mut alive, &mut nearest); }

    // the crown
    let mut kidv: Vec<Vec<u32>> = (0..nodes.len()).map(|i| if i + 1 < nodes.len() { vec![i as u32 + 1] } else { Vec::new() }).collect();
    let mut pull: Vec<(V, u32)> = Vec::new();
    for _round in 0..ROUNDS {
        pull.clear();
        pull.resize(nodes.len(), ([0.0; 3], 0));
        let mut any = false;
        for (a, &(n, d)) in nearest.iter().enumerate() {
            if !alive[a] || n == NONE || d >= reach { continue; }
            let q = &mut pull[n as usize];
            q.0 = add(q.0, unit(sub(free[a], nodes[n as usize].p)));
            q.1 += 1;
            any = true;
        }
        if !any { break; }
        let count = nodes.len();
        let mut grew = false;
        for i in 0..count {
            let (sum, k) = pull[i];
            if k == 0 { continue; }
            let n = nodes[i].clone();
            if n.kids >= 2 { continue; }
            // buds are spaced along a stem: no side shoot right next to another fork
            if n.kids > 0 && fork_near(&nodes, &kidv, i) { continue; }
            let mut dir = unit(sum);
            // a side shoot leaves its stem at about the branch angle
            if n.kids > 0 {
                let along = dot(dir, n.dir);
                let mut side = sub(dir, mul(n.dir, along));
                if len(side) < 1e-3 { side = rng.dir(); side = sub(side, mul(n.dir, dot(side, n.dir))); }
                let side = unit(side);
                let want = add(mul(n.dir, s.angle.cos()), mul(side, s.angle.sin()));
                dir = unit(lerp(dir, want, 0.65));
            }
            // turning: upwards (or down), out to the light, the main stem keeping on up, and wandering
            let off = [n.p[0] - mid[0], 0.0, n.p[2] - mid[2]];
            let out = if off[0].abs() + off[2].abs() < 1e-6 { [0.0; 3] } else { unit(off) };
            let wander = unit(add(mul(n.wander, 0.93), mul(rng.dir(), 0.28)));
            let mut bias = add(mul(UP, 0.45 * s.upturn), mul(unit(add(out, mul(UP, 0.6))), 0.45 * s.light));
            // (the leader gives up at the top of the crown, rather than going on as a spike)
            if n.leader { bias = add(bias, mul(UP, 1.6 * s.leader * ((1.0 - n.p[1]) / 0.15).clamp(0.0, 1.0))); }
            dir = unit(add(add(dir, bias), mul(wander, 0.45 * s.wiggle)));
            // never grow back into a shoot already there
            let p = add(n.p, mul(dir, STEP));
            if kidv[i].iter().any(|&m| len(sub(nodes[m as usize].p, p)) < 0.45 * STEP) { continue; }
            // the main stem goes on through the first shoot that heads upwards
            let leader = n.leader && dir[1] > 0.35 && !kidv[i].iter().any(|&m| nodes[m as usize].leader);
            nodes[i].kids += 1;
            let j = nodes.len() as u32;
            kidv[i].push(j);
            kidv.push(Vec::new());
            nodes.push(Node { wander, leader, ..Node::new(p, i as u32, dir) });
            settle(j, p, &mut alive, &mut nearest);
            grew = true;
        }
        if !grew { break; }
    }
    nodes
}

/// A run of nodes from node `from`, heading as `dir_at` says for each step (it is told the step's number
/// and the way it was heading), `steps` long. Returns the nodes' indices.
fn shoot(nodes: &mut Vec<Node>, from: u32, steps: usize, step: f64, role: Role, mut dir_at: impl FnMut(usize, V) -> V) -> Vec<u32> {
    let mut at = from;
    let mut d = nodes[from as usize].dir;
    let mut out = Vec::with_capacity(steps);
    for k in 0..steps {
        d = unit(dir_at(k, d));
        let p = add(nodes[at as usize].p, mul(d, step));
        nodes[at as usize].kids += 1;
        let j = nodes.len() as u32;
        nodes.push(Node { role, ..Node::new(p, at, d) });
        out.push(j);
        at = j;
    }
    out
}

/// A conifer: a leader straight to the top; whorls of branches up it, each tier shorter than the one below
/// (a cone), the branches near level, rising a little and drooping at their ends; side shoots along each
/// branch in its plane, alternately left and right, shorter towards its end; and shorter ones again on
/// those. Needles come later, along every twig.
fn conifer(s: &Sliders, lean: &Lean, rng: &mut Rng) -> Vec<Node> {
    let mut nodes = trunk(s, lean, 1.0, rng, |_| false);
    let stem: Vec<u32> = (0..nodes.len() as u32).collect();
    // how close the whorls are: more twigs, more tiers
    let tiers = (6.0 + 9.0 * (s.twigs as f64 / 1100.0).sqrt()).round().clamp(5.0, 50.0) as usize;
    let gap = s.crown * 0.97 / tiers as f64;
    // a cone (pointed or flat rounds its shoulders)
    let shape = (-1.0 + 0.8 * s.shape.max(0.0)).clamp(-1.0, 0.0);
    // the branches leave level or a little up; `angle` (from the vertical) as on the other forms, 38° here
    // giving branches 12° above level
    let rise = (90f64.to_radians() - s.angle).clamp(-0.3, 1.2) * 0.25;
    let droop = 0.5 + 1.2 * s.droop;
    let mut az0 = 2.0 * PI * rng.next();
    for t in 0..tiers {
        let y = s.base + (t as f64 + 0.5 * rng.next()) * gap;
        let f = (y - s.base) / s.crown;
        let Some(&at) = stem.iter().find(|&&i| nodes[i as usize].p[1] >= y) else { break };
        // a tier: four to six branches round the stem, the tier turned on from the last by the golden angle
        let count = 4 + (rng.next() * 2.99) as usize;
        az0 += 2.39996;
        let reach = s.half * outline(f, shape) * (0.85 + 0.3 * rng.next()) * (1.0 + 0.3 * s.lumps * rng.sym());
        let steps = (reach / STEP).round() as usize;
        if steps < 2 { continue; }
        for b in 0..count {
            let az = az0 + 2.0 * PI * (b as f64 + 0.25 * rng.sym()) / count as f64;
            // young tiers near the top point further up
            let el0 = rise + 0.5 * f * f;
            let wob = 0.12 * s.wiggle;
            let mut wr = Rng::new(rng.0 ^ (b as u64 * 977));
            let branch = shoot(&mut nodes, at, steps, STEP, Role::Wood, |k, _| {
                let u = k as f64 / steps as f64;
                // level near the stem, then bowing down towards the end
                let el = el0 - droop * u * u + wob * wr.sym();
                heading(az + wob * wr.sym(), el)
            });
            sides(s, &mut nodes, &branch, 2, 1.0, rng);
        }
    }
    nodes
}

/// A columnar tree: a leader to the top; up it, in a spiral, branches that leave at half the branch angle
/// from the vertical and turn further up as they go, so that they run up beside the stem; their length
/// keeps their tips inside a spindle (widest a little below the middle, coming to a point at the top).
fn columnar(s: &Sliders, lean: &Lean, rng: &mut Rng) -> Vec<Node> {
    let mut nodes = trunk(s, lean, 1.0, rng, |_| false);
    let stem: Vec<u32> = (0..nodes.len() as u32).collect();
    let count = (24.0 + 30.0 * (s.twigs as f64 / 1100.0).sqrt()).round().clamp(8.0, 120.0) as usize;
    let shape = (0.5 * s.shape - 0.3).clamp(-1.0, 1.0);
    let lean0 = (0.5 * s.angle).clamp(0.05, 0.8);
    let mut az = 2.0 * PI * rng.next();
    for b in 0..count {
        let y = s.base + s.crown * 0.92 * (b as f64 + rng.next()) / count as f64;
        let Some(&at) = stem.iter().find(|&&i| nodes[i as usize].p[1] >= y) else { break };
        az += 2.39996;
        // how far up it runs: to near the spindle's top, the lower ones further; how far out: the spindle
        let f = (y - s.base) / s.crown;
        let up = (1.0 - y) * (0.55 + 0.3 * rng.next());
        let out = s.half * outline((f + 0.5 * up / s.crown).min(1.0), shape) * (0.6 + 0.4 * rng.next());
        let steps = ((up * up + out * out).sqrt() / STEP).round() as usize;
        if steps < 3 { continue; }
        let (az, wob) = (az + 0.2 * rng.sym(), 0.1 * s.wiggle);
        let mut wr = Rng::new(rng.0 ^ (b as u64 * 7919));
        let branch = shoot(&mut nodes, at, steps, STEP, Role::Wood, |k, _| {
            // out at first, then up: the tip has come no further out than `out`
            let u = k as f64 / steps as f64;
            let tilt = lean0 * (1.0 - u).powi(2) + (out / (up + 1e-6)).atan() * 0.5 * (1.0 - u);
            heading(az + wob * wr.sym(), PI / 2.0 - tilt + 0.15 * s.upturn * u + wob * wr.sym())
        });
        sides(s, &mut nodes, &branch, 1, 0.3, rng);
    }
    nodes
}

/// Side shoots along a branch, in its plane, alternately left and right, angled forwards and shorter
/// towards its end; `depth` more orders of them; `spread` how far out to the side they head (1: as far as
/// forwards).
fn sides(s: &Sliders, nodes: &mut Vec<Node>, branch: &[u32], depth: u32, spread: f64, rng: &mut Rng) {
    if depth == 0 || branch.len() < 6 { return; }
    let n = branch.len();
    let mut left = rng.next() < 0.5;
    let mut k = 2 + (rng.next() * 2.0) as usize;
    while k + 2 < n {
        let at = branch[k];
        let d = nodes[at as usize].dir;
        // the branch's plane: level across it
        let flat = unit(cross(d, UP));
        let side = if left { flat } else { mul(flat, -1.0) };
        left = !left;
        let fwd = 0.85;
        let out = unit(add(mul(d, fwd), mul(side, spread)));
        let remain = (n - k) as f64;
        let steps = ((remain * 0.55 * (0.8 + 0.4 * rng.next())) as usize).min(40);
        if steps >= 2 {
            let droop = 0.3 + 0.8 * s.droop;
            let twig = shoot(nodes, at, steps, STEP, Role::Wood, |j, _| {
                let u = j as f64 / steps as f64;
                bend_down(out, droop * u * 0.6)
            });
            sides(s, nodes, &twig, depth - 1, spread, rng);
        }
        k += 2 + (rng.next() * 1.6) as usize;
    }
}

/// A palm: a tall unbranched trunk, bending if it leans, and at its top a crown of fronds arching out
/// and down, the oldest lowest and hanging furthest. (Their leaflets are drawn with the leaves.)
fn palm(s: &Sliders, lean: &Lean, rng: &mut Rng) -> Vec<Node> {
    let top = (s.base + 0.25 * s.crown).max(0.2);
    let mut nodes = trunk(s, lean, top, rng, |_| false);
    let head = nodes.len() as u32 - 1;
    let fronds = (8.0 + 10.0 * (s.twigs as f64 / 1100.0).sqrt()).round().clamp(5.0, 30.0) as usize;
    let long = 1.6 * s.half * (0.9 + 0.2 * rng.next());
    for k in 0..fronds {
        let f = k as f64 / fronds as f64;
        let az = k as f64 * 2.39996 + 0.2 * rng.sym();
        // the youngest stand up in the middle, the oldest hang
        let el0 = (75.0 - 95.0 * f).to_radians() + 0.1 * rng.sym();
        let l = long * (0.75 + 0.35 * (1.0 - (2.0 * f - 0.6).abs())) * (0.9 + 0.2 * rng.next());
        let steps = (l / STEP).round().max(4.0) as usize;
        let arch = 1.0 + 1.2 * s.droop;
        let frond = shoot(&mut nodes, head, steps, STEP, Role::Frond, |j, _| {
            let u = j as f64 / steps as f64;
            heading(az, el0 - arch * u * (0.6 + 0.6 * u))
        });
        for (j, &i) in frond.iter().enumerate() {
            nodes[i as usize].shrink = 1.0 - 0.85 * j as f64 / steps as f64;
        }
    }
    nodes
}

/// Weeping: long thin whips hang from the ends of the limbs.
fn whips(s: &Sliders, nodes: &mut Vec<Node>, rng: &mut Rng) {
    let n = nodes.len();
    let mut kids = vec![0u32; n];
    for m in nodes.iter() {
        if m.parent != NONE { kids[m.parent as usize] += 1; }
    }
    for i in 0..n {
        if kids[i] > 0 || rng.next() > 0.85 { continue; }
        let d0 = nodes[i].dir;
        let long = s.crown * (0.25 + 0.3 * rng.next()) * (0.6 + 0.8 * s.droop);
        let steps = (long / STEP).round() as usize;
        let sway = mul(rng.dir(), 0.15 * s.wiggle);
        shoot(nodes, i as u32, steps, STEP, Role::Whip, |k, _| {
            // over the first few steps the shoot gives way and hangs
            let a = (k as f64 / 4.0).min(1.0) * 3.0;
            unit(add(bend_down(d0, a), sway))
        });
    }
}

/// Two things space colonisation leaves that no real tree has, cut off: the last step or two of a shoot
/// that turned hard aside after one last free point (a hooked tip), and side shoots that stopped after a
/// step or three (stubs, like thorns).
fn trim_hooks(mut nodes: Vec<Node>) -> Vec<Node> {
    for _ in 0..3 {
        let n = nodes.len();
        let mut kids = vec![0u32; n];
        for m in &nodes {
            if m.parent != NONE { kids[m.parent as usize] += 1; }
        }
        let mut keep = vec![true; n];
        for i in 0..n {
            let q = nodes[i].parent;
            if kids[i] > 0 || q == NONE || nodes[q as usize].parent == NONE { continue; }
            if dot(nodes[i].dir, nodes[q as usize].dir) < 0.8 || kids[q as usize] > 1 { keep[i] = false; }
        }
        let mut to = vec![NONE; n];
        let mut out = Vec::with_capacity(n);
        for (i, m) in nodes.into_iter().enumerate() {
            if !keep[i] { continue; }
            to[i] = out.len() as u32;
            let parent = if m.parent == NONE { NONE } else { to[m.parent as usize] };
            out.push(Node { parent, ..m });
        }
        nodes = out;
    }
    nodes
}

/// Is there a fork on the stem through node i within `SPACING` steps either way?
fn fork_near(nodes: &[Node], kidv: &[Vec<u32>], i: usize) -> bool {
    let mut a = i;
    for _ in 0..SPACING {
        let q = nodes[a].parent;
        if q == NONE { break; }
        a = q as usize;
        if kidv[a].len() > 1 { return true; }
    }
    let mut a = i;
    for _ in 0..SPACING {
        let Some(&c) = kidv[a].first() else { break };
        a = c as usize;
        if kidv[a].len() > 1 { return true; }
    }
    false
}

/// From the grown nodes to what is drawn: smoothing, the pipe model's radii for drooping, the strands,
/// the roots, and the frame.
fn finish(s: &Sliders, nodes: Vec<Node>, rng: &mut Rng) -> Grown {
    let n = nodes.len();
    let parent: Vec<u32> = nodes.iter().map(|m| m.parent).collect();
    let mut kids: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (i, &q) in parent.iter().enumerate() {
        if q != NONE { kids[q as usize].push(i as u32); }
    }
    // smoothing: the steps of growth zig-zag; every node with one child is drawn towards the line from its
    // parent to its child (forks stay where they are)
    let mut p: Vec<V> = nodes.iter().map(|m| m.p).collect();
    for _ in 0..3 {
        let q = p.clone();
        for i in 0..n {
            if parent[i] == NONE || kids[i].len() != 1 { continue; }
            let mid = mul(add(q[parent[i] as usize], q[kids[i][0] as usize]), 0.5);
            p[i] = lerp(q[i], mid, 0.5);
        }
    }
    // the pipe model at full size: how many tips each node carries (children always come after parents)
    let mut tips = vec![0.0f64; n];
    let mut from_tip = vec![u32::MAX; n];
    for i in (0..n).rev() {
        if kids[i].is_empty() {
            tips[i] = 1.0;
            from_tip[i] = 0;
        }
        if parent[i] != NONE {
            let q = parent[i] as usize;
            tips[q] += tips[i];
            from_tip[q] = from_tip[q].min(from_tip[i].saturating_add(1));
        }
    }
    let all = tips[0].max(1.0);
    // droop: shoots bend down under their own weight, thin ones the most; the bend adds up along a shoot
    // (a conifer's and a palm's droop is built into their branches and fronds)
    if s.droop > 0.0 && matches!(s.form, BROADLEAF | COLUMNAR | WEEPING) {
        let orig = p.clone();
        let mut bend = vec![0.0f64; n];
        for i in 1..n {
            let q = parent[i] as usize;
            let d = sub(orig[i], orig[q]);
            let l = len(d);
            if nodes[i].role == Role::Whip {
                p[i] = add(p[q], d);
                continue;
            }
            let thin = 1.0 - (tips[i] / all).powf(1.0 / s.taper);
            bend[i] = bend[q] + s.droop * 9.0 * thin * thin * l;
            p[i] = add(p[q], mul(bend_down(d, bend[i]), l));
        }
    }
    let mut path = vec![0.0f64; n];
    for i in 1..n { path[i] = path[parent[i] as usize] + len(sub(p[i], p[parent[i] as usize])); }
    let longest = path.iter().copied().fold(0.0, f64::max);

    // strands: each goes on through its thickest child; the others start strands of their own
    let mut strands: Vec<Vec<u32>> = Vec::new();
    let mut todo: Vec<Vec<u32>> = vec![vec![0]];
    while let Some(mut line) = todo.pop() {
        let mut at = *line.last().unwrap() as usize;
        loop {
            if kids[at].is_empty() { break; }
            // (the trunk of a conifer or a palm goes on up through its leader, whatever the branches carry)
            let main = *kids[at]
                .iter()
                .max_by(|&&a, &&b| (nodes[a as usize].leader, tips[a as usize]).partial_cmp(&(nodes[b as usize].leader, tips[b as usize])).unwrap())
                .unwrap();
            for &k in &kids[at] {
                if k != main { todo.push(vec![at as u32, k]); }
            }
            line.push(main);
            at = main as usize;
        }
        strands.push(line);
    }

    // roots: buttresses that leave the trunk some way up, run down its side and out along the ground,
    // wandering and thinning to nothing
    let mut roots = Vec::new();
    let turn0 = 2.0 * PI * rng.next();
    for i in 0..s.roots {
        let az0 = turn0 + 2.0 * PI * (i as f64 + 0.35 * rng.sym()) / s.roots as f64;
        let long = (0.06 + 0.06 * rng.next()) * (0.7 + 0.3 * s.flare) + 2.5 * s.trunk;
        let steps = (long / STEP).ceil().max(3.0) as usize;
        let (mut az, mut spin) = (az0, 0.0);
        let top = s.trunk * (1.5 + 1.5 * s.flare);
        let mut line = vec![([0.0, top, 0.0], 0.5)];
        let mut out = 0.0;
        for k in 1..=steps {
            let f = k as f64 / steps as f64;
            spin = 0.8 * spin + 0.25 * rng.sym() * (0.3 + s.wiggle);
            az += spin * 0.25;
            out += STEP;
            // down the trunk's side over the first stretch, then along the ground, sinking into it
            let fall = (-out / (1.2 * s.trunk)).exp();
            let h = top * fall - s.trunk * 0.6 * f;
            line.push(([out * az.cos(), h, out * az.sin()], 0.5 * (1.0 - f).powf(1.6) + 0.04));
        }
        roots.push(line);
    }

    // the frame: every point of the tree at full size, leaves and all
    let mut frame: Vec<f64> = Vec::with_capacity(3 * n);
    for q in &p { frame.extend_from_slice(q); }
    for l in &roots {
        for (q, _) in l { frame.extend_from_slice(q); }
    }
    let reach = leaf_len(s, Role::Frond).max(leaf_len(s, Role::Wood)) * if s.leaves > 0.0 || s.form == PALM { 1.0 } else { 0.0 };
    if reach > 0.0 {
        for i in 0..n {
            if kids[i].is_empty() || nodes[i].role == Role::Frond {
                for d in [UP, DOWN, [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]] { frame.extend_from_slice(&add(p[i], mul(d, reach))); }
            }
        }
    }
    let tips_n = kids.iter().filter(|k| k.is_empty()).count();
    let trunk = strands.first().and_then(|t| t.iter().find(|&&i| kids[i as usize].len() > 1)).map_or(longest, |&i| path[i as usize]);
    Grown {
        role: nodes.iter().map(|m| m.role).collect(),
        shrink: nodes.iter().map(|m| m.shrink).collect(),
        p, parent, path, strands, tips: tips_n, carry: tips, from_tip, roots, frame, longest, base: s.base.max(0.05), trunk,
    }
}

/// How long a leaf is, by what bears it (units of the tree's height).
fn leaf_len(s: &Sliders, role: Role) -> f64 {
    match role {
        Role::Frond => 0.34 * s.half,
        _ if s.form == CONIFER => 0.02 * s.leaf,
        Role::Whip => 0.035 * s.leaf,
        _ => 0.045 * s.leaf,
    }
}

/// The tree as `grow` shows it: lines and their radii, and its leaves.
fn draw(g: &Grown, s: &Sliders, grow: f64, quick: bool) -> RawCurve {
    let n = g.p.len();
    // When each node appears, from 0 to 1: the trunk rises in the first TRUNK_TIME, the crown spreads out
    // over the rest, every shoot lengthening at the same pace.
    const TRUNK_TIME: f64 = 0.15;
    let when = |path: f64| -> f64 {
        if path <= g.trunk { TRUNK_TIME * path / g.trunk.max(1e-9) } else { TRUNK_TIME + (1.0 - TRUNK_TIME) * (path - g.trunk) / (g.longest - g.trunk).max(1e-9) }
    };
    let grow = grow.clamp(0.0, 1.0);
    // how much of each node is there: 1 whole, 0 not yet, between: the shoot growing towards it
    let part: Vec<f64> = (0..n)
        .map(|i| {
            let t = when(g.path[i]);
            if t <= grow { return 1.0; }
            let q = g.parent[i];
            if q == NONE { return if grow > 0.0 { 1.0 } else { 0.0 }; }
            let a = when(g.path[q as usize]);
            if a >= grow { 0.0 } else { (grow - a) / (t - a).max(1e-12) }
        })
        .collect();
    // the pipe model on what is there: tips weigh as much as they have grown (in radius^n)
    let mut weight = vec![0.0f64; n];
    let mut has_kid = vec![false; n];
    for i in 1..n {
        if part[i] > 0.0 { has_kid[g.parent[i] as usize] = true; }
    }
    for i in (0..n).rev() {
        if part[i] <= 0.0 { continue; }
        if !has_kid[i] { weight[i] += part[i].min(1.0); }
        if g.parent[i] != NONE { weight[g.parent[i] as usize] += weight[i]; }
    }
    let all = g.tips.max(1) as f64;
    // (a stem also thickens with age, not only with the twigs it carries: a young tree is not a wire)
    let age = 0.6 * grow.powf(1.5);
    let radius = |i: usize| g.shrink[i] * s.trunk * (weight[i] / all).max(age * g.carry[i] / all).powf(1.0 / s.taper);
    let at = |i: usize| -> V {
        let q = g.parent[i];
        if part[i] >= 1.0 || q == NONE { g.p[i] } else { lerp(g.p[q as usize], g.p[i], part[i]) }
    };
    let trunk_r = radius(0);

    let mut lines: Vec<Vec<f64>> = Vec::new();
    let mut radii: Vec<Vec<f64>> = Vec::new();
    for (k, strand) in g.strands.iter().enumerate() {
        let (mut l, mut r) = (Vec::new(), Vec::new());
        for (j, &i) in strand.iter().enumerate() {
            let i = i as usize;
            if part[i] <= 0.0 { break; }
            let mut q = at(i);
            // a side strand starts inside the node it leaves from, as thick as itself
            let mut rad = if j == 0 && k > 0 { radius(strand[1] as usize) } else { radius(i) };
            // the trunk thins a little as it rises (a palm's hardly), and flares out into its foot
            if k == 0 {
                if s.form != PALM { rad *= 1.0 + 0.25 * (1.0 - q[1] / g.base).max(0.0); }
                rad *= 1.0 + 0.4 * s.flare * (-q[1] / (3.0 * trunk_r.max(1e-6))).exp();
                // the foot goes on below the ground, where the shape is cut flat (`ground`): a sawn base,
                // not a round end
                if j == 0 { q[1] -= rad; }
            }
            l.extend_from_slice(&q);
            r.push(rad);
        }
        // the pipe model thins a stem in a step where a branch leaves it: spread each step along the stem
        // (a side strand's first radius, inside its parent, is left as it is)
        let first = usize::from(k > 0);
        for _ in 0..3 {
            if r.len() < first + 3 { break; }
            let q = r.clone();
            for j in first + 1..r.len() - 1 { r[j] = 0.25 * q[j - 1] + 0.5 * q[j] + 0.25 * q[j + 1]; }
        }
        if r.len() >= 2 {
            lines.push(l);
            radii.push(r);
        }
    }
    // roots grow out with the tree, as thick as the trunk lets them be
    for root in &g.roots {
        let m = root.len();
        let shown = ((grow * 1.15).min(1.0) * (m - 1) as f64).max(0.0);
        let whole = shown.floor() as usize;
        if whole < 1 { continue; }
        let (mut l, mut r) = (Vec::new(), Vec::new());
        for (q, f) in root.iter().take(whole + 1) {
            l.extend_from_slice(q);
            r.push(f * trunk_r * (1.0 + 0.4 * s.flare));
        }
        lines.push(l);
        radii.push(r);
    }
    let mut raw = RawCurve { lines, closed: false, radii, frame: g.frame.clone(), ground: Some(0.0), blades: Vec::new(), pieces: Vec::new() };
    leaves(g, s, &part, &has_kid, &at, quick, &mut raw);
    raw
}

/// The leaves of the tree as it is grown so far, into `raw`: blades with stalks near the ends of the
/// shoots; along a weeping tree's whips, narrow blades all the way; a palm's leaflets down both sides of
/// every frond; a conifer's needles in brushes along every twig.
///
/// Each node's leaves come from a generator of its own, so that they stay where they are while the tree
/// grows. `quick` (while something moves, or the tree grows as the animation plays): the leaves of only
/// one node in three, so that rebuilding every frame stays quick; all of them again once it is still.
fn leaves(g: &Grown, s: &Sliders, part: &[f64], has_kid: &[bool], at: &dyn Fn(usize) -> V, quick: bool, raw: &mut RawCurve) {
    let n = g.p.len();
    let palm = s.form == PALM;
    if s.leaves <= 0.0 && !palm { return; }
    let dir = |i: usize| if g.parent[i] == NONE { UP } else { unit(sub(g.p[i], g.p[g.parent[i] as usize])) };
    for i in 0..n {
        if part[i] <= 0.0 { continue; }
        let mut rng = Rng::new(s.seed ^ 0x51ab ^ (i as u64).wrapping_mul(0x2545f4914f6cdd1d));
        // (needles, and a columnar tree's crowded leaves: one node in six, there being so many more)
        if quick && rng.next() > if matches!(s.form, CONIFER | COLUMNAR) { 1.0 / 6.0 } else { 1.0 / 3.0 } { continue; }
        let grown = part[i].min(1.0);
        let q = at(i);
        let d = dir(i);
        match (s.form, g.role[i]) {
            (_, Role::Frond) => {
                // leaflets down both sides of the midrib, longest in its middle, angled forward and hanging
                // in a V; none on the bare stalk at its foot
                let u = g.shrink[i];
                let f = ((1.0 - u) / 0.85).clamp(0.0, 1.0);
                if f < 0.12 { continue; }
                let density = 0.5 + 0.5 * s.leaves;
                if rng.next() > density { continue; }
                let l = leaf_len(s, Role::Frond) * (PI * ((f - 0.12) / 0.88).clamp(0.0, 1.0)).sin().powf(0.6).max(0.25) * grown;
                let flat = unit(cross(d, UP));
                for side in [-1.0, 1.0] {
                    let out = unit(add(add(mul(flat, side), mul(d, 0.6)), mul(DOWN, 0.55)));
                    blade(raw, q, out, unit(cross(out, d)), l, 0.07, 0.0);
                }
            }
            (CONIFER, _) => {
                // needles: a brush round every twig, a few to each step
                if g.carry[i] > 6.0 || has_kid[i] && g.from_tip[i] > 30 { continue; }
                // (one to two to a step: more costs more than the 11.1 ms a frame allows at 4K)
                let count = (1.0 + s.leaves).round() as usize;
                let l = leaf_len(s, Role::Wood) * grown;
                let side0 = across(d);
                let side1 = cross(d, side0);
                for k in 0..count {
                    let a = 2.0 * PI * (k as f64 + rng.next()) / count as f64;
                    let out = unit(add(add(mul(side0, a.cos()), mul(side1, a.sin())), mul(d, 0.7)));
                    let base = add(q, mul(d, STEP * rng.next()));
                    // (a needle is a blade narrower than it is thick: a fine stick, one piece for the baker
                    // where a tube would be cut into several)
                    blade(raw, base, out, side0, l, 0.06, 0.0);
                }
            }
            (_, Role::Whip) => {
                // narrow willow leaves all along the whip, alternately
                if rng.next() > 0.35 + 0.65 * s.leaves { continue; }
                let side = across(d);
                let side = if rng.next() < 0.5 { side } else { mul(side, -1.0) };
                let out = unit(add(add(mul(d, 0.8), side), mul(rng.dir(), 0.2)));
                blade(raw, q, out, unit(cross(out, side)), 1.3 * leaf_len(s, Role::Whip) * grown, 0.11, 0.1);
            }
            _ => {
                // broad leaves on stalks, on the last few steps of every shoot
                if has_kid[i] && g.from_tip[i] > 3 { continue; }
                // (a columnar tree has so many shoots that fewer of them bear leaves)
                if rng.next() > s.leaves * if s.form == COLUMNAR { 0.6 } else { 1.0 } { continue; }
                let count = 1 + (2.5 * s.leaves * rng.next()).round() as usize;
                let l = leaf_len(s, Role::Wood) * grown * (0.8 + 0.4 * rng.next());
                for _ in 0..count {
                    // out from the shoot, a little down, the blade lying near level (it faces the sky)
                    let side = across(d);
                    let a = 2.0 * PI * rng.next();
                    let out = unit(add(add(mul(side, a.cos()), mul(cross(d, side), a.sin())), mul(d, 0.5)));
                    let out = unit(add(out, mul(DOWN, 0.3)));
                    // (no stalk of its own: a tube as thin as the grid holds is as thick as a twig; the
                    // blade's narrow base meets the shoot)
                    let stalk = add(q, mul(out, 0.1 * l));
                    let across_leaf = unit(cross(out, UP));
                    let across_leaf = if len(cross(out, UP)) < 1e-3 { across(out) } else { across_leaf };
                    let twist = 0.4 * rng.sym();
                    let normal = unit(cross(out, across_leaf));
                    let w = unit(add(mul(across_leaf, twist.cos()), mul(normal, twist.sin())));
                    let width = match s.form { COLUMNAR => 0.45, _ => 0.36 };
                    blade(raw, stalk, out, w, l, width, 0.12);
                }
            }
        }
    }
}

/// One flat leaf: from `base` along `dir`, `long` long; `side` the way across it; its widest half-width a
/// share `wide` of its length; `cup` how much it curls up about its midrib.
fn blade(raw: &mut RawCurve, base: V, dir: V, side: V, long: f64, wide: f64, cup: f64) {
    if long <= 0.0 { return; }
    raw.blades.push(Blade { base, tip: add(base, mul(dir, long)), side: unit(sub(side, mul(dir, dot(side, dir)))), width: wide * long, thick: 0.0004, cup });
}

/// Trees grown lately, by their sliders (all but `grow`): moving `grow` only cuts one of them.
static GROWN: Mutex<Vec<(Vec<u64>, Arc<Grown>)>> = Mutex::new(Vec::new());

fn tree(v: &[f64], o: &CurveOpts) -> RawCurve {
    let s = Sliders::read(v);
    let key: Vec<u64> = v.iter().enumerate().filter(|&(i, _)| i != GROW).map(|(_, x)| x.to_bits()).collect();
    let found = GROWN.lock().ok().and_then(|c| c.iter().find(|(k, _)| *k == key).map(|(_, g)| g.clone()));
    let g = match found {
        Some(g) => g,
        None => {
            let g = Arc::new(grow_tree(&s));
            if let Ok(mut c) = GROWN.lock() {
                c.insert(0, (key, g.clone()));
                c.truncate(4);
            }
            g
        }
    };
    draw(&g, &s, v.get(GROW).copied().unwrap_or(1.0), o.quick)
}

/// The names of the `form` slider's settings.
pub fn choices(id: &str) -> Option<Vec<String>> {
    (id == "form").then(|| FORMS.iter().map(|s| s.to_string()).collect())
}

pub fn family() -> Family {
    let preset = Preset {
        eq: vec!["growth: shoots compete for free space (space colonisation), or whorls and fronds", "thickness: rⁿ of a stem = Σ rⁿ of its branches"],
        note: "A tree grown from its sliders. The form is how it grows; every seed is another tree of that form.",
        params: params(),
        tube: Some(0.03),
        fixed_length: true,
        centre: true,
        info: Some(Box::new(|raw, _v| format!("{} branches, {} leaves", raw.lines.len(), raw.blades.len()))),
        // playing, the tree grows and draws back a little at its tips
        animate: Some(vec!["grow"]),
        rot: Some([4.0, 0.0, 0.0]),
        ..Preset::new("tree", "tree", Shape::Curve(Box::new(tree)))
    };
    Family {
        presets: vec![preset],
        melt: 0.004,
        defaults: vec![
            ("ior", Setting::Num(1.5)), ("absorb", Setting::Vec3([0.16, 0.08, 0.035])),
            ("dist", Setting::Num(6.6)), ("fov", Setting::Num(26.0)), ("rot", Setting::Vec3([4.0, 0.0, 0.0])),
        ],
        // broad soft lights that fade to their edges, as for the models, and two small ones for glints
        lights: vec![
            Light { az:   40.0, el:  20.0, w: 24.0, h: 38.0, rgb: [3.6, 3.5, 3.3],   soft: 0.9 },
            Light { az:  -58.0, el:  -6.0, w: 18.0, h: 44.0, rgb: [0.45, 1.25, 3.6], soft: 0.9 },
            Light { az: -140.0, el:   5.0, w: 15.0, h: 44.0, rgb: [4.4, 1.45, 0.24], soft: 0.9 },
            Light { az:  150.0, el:  29.0, w: 28.0, h: 20.0, rgb: [2.6, 2.5, 2.45],  soft: 0.9 },
            Light { az:    0.0, el:  85.0, w: 38.0, h: 38.0, rgb: [0.8, 0.9, 1.1],   soft: 1.0 },
            Light { az:    0.0, el: -80.0, w: 34.0, h: 34.0, rgb: [0.5, 0.22, 0.05], soft: 1.0 },
            Light { az:   95.0, el:  46.0, w:  5.0, h:  5.0, rgb: [6.0, 6.0, 6.0],   soft: 0.5 },
            Light { az: -100.0, el: -36.0, w:  4.0, h:  8.0, rgb: [4.5, 5.0, 6.0],   soft: 0.5 },
        ],
        ..Family::new(FAMILY, "Trees", Kind::Curve)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> Vec<f64> { params().iter().map(|p| p.value).collect() }
    fn of_form(f: usize) -> Vec<f64> {
        let mut v = defaults();
        v[FORM] = f as f64;
        v
    }

    #[test]
    fn the_same_seed_grows_the_same_tree() {
        let v = defaults();
        let (a, b) = (grow_tree(&Sliders::read(&v)), grow_tree(&Sliders::read(&v)));
        assert_eq!(a.p, b.p);
        let mut w = v.clone();
        w[SEED] = 2.0;
        assert_ne!(grow_tree(&Sliders::read(&w)).p, a.p);
    }

    #[test]
    fn every_form_grows_a_tree_with_the_trunk_thickest() {
        for f in 0..FORMS.len() {
            let v = of_form(f);
            let s = Sliders::read(&v);
            let g = grow_tree(&s);
            assert!(g.tips > 5, "{}: {} tips", FORMS[f], g.tips);
            let raw = draw(&g, &s, 1.0, false);
            assert_eq!(raw.lines.len(), raw.radii.len());
            let r0 = raw.radii[0].iter().copied().fold(0.0, f64::max);
            assert!(raw.radii.iter().skip(1).all(|r| r.iter().all(|&x| x <= r0)), "{}", FORMS[f]);
        }
    }

    #[test]
    fn a_longer_trunk_leaves_the_crown_its_size() {
        // the crown's height against its width stays as it was; only the trunk below it is longer
        let mut v = defaults();
        v[BOLE] = 0.3;
        let a = Sliders::read(&v);
        v[BOLE] = 1.2;
        let b = Sliders::read(&v);
        assert!((a.half / a.crown - b.half / b.crown).abs() < 1e-12);
        assert!(b.base > a.base);
    }

    #[test]
    fn leaves_are_blades_and_needles() {
        let mut v = of_form(BROADLEAF);
        v[LEAVES] = 0.6;
        let s = Sliders::read(&v);
        assert!(draw(&grow_tree(&s), &s, 1.0, false).blades.len() > 50);
        let s = Sliders::read(&of_form(PALM));
        assert!(draw(&grow_tree(&s), &s, 1.0, false).blades.len() > 100, "a palm always has its leaflets");
        let mut v = of_form(CONIFER);
        v[LEAVES] = 0.5;
        let s = Sliders::read(&v);
        let g = grow_tree(&s);
        let (bare, needled) = (draw(&g, &Sliders::read(&of_form(CONIFER)), 1.0, false), draw(&g, &s, 1.0, false));
        assert!(needled.blades.len() > bare.blades.len() + 500);
    }

    #[test]
    fn growing_only_cuts_the_tree() {
        let v = defaults();
        let s = Sliders::read(&v);
        let g = grow_tree(&s);
        let (half, whole) = (draw(&g, &s, 0.5, false), draw(&g, &s, 1.0, false));
        let count = |r: &RawCurve| r.lines.iter().map(|l| l.len()).sum::<usize>();
        assert!(count(&half) < count(&whole));
        assert_eq!(half.frame, whole.frame);
    }
}
