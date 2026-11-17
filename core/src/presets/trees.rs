//! Trees, grown rather than drawn.
//!
//! Growth is space colonisation (Runions, Lane and Prusinkiewicz, 2007): points scattered through the
//! volume the crown is to fill are the free space a tree competes for. Every point pulls on the nearest
//! growing bud within reach; a bud pulled grows one step towards the mean of its pulls; a point is used up
//! when a bud comes close. Forks, crowding and gaps come out of that competition rather than from a rule.
//!
//! Thickness is the pipe model (Shinozaki; da Vinci's rule): a branch's radius to the power n is the sum
//! of its children's, so the trunk is as thick as all its twigs together.
//!
//! The tree is one shape with sliders: the crown's outline, how the shoots turn (up, to the light, by
//! their own wandering), how strongly a leader keeps going, how the radii fall off, a flare and roots at
//! the foot, and leaves. `seed` gives another tree from the same sliders; `grow` shows it part grown.
//!
//! Growth is the slow part (some tens of milliseconds), so the grown tree is kept for its sliders and
//! only `grow`, which cuts it, is worked out afresh as it moves.

use super::*;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

pub const FAMILY: &str = "trees";

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
const CROWN: usize = 2;
const WIDTH: usize = 3;
const SHAPE: usize = 4;
const LUMPS: usize = 5;
const TWIGS: usize = 6;
const ANGLE: usize = 7;
const LEADER: usize = 8;
const UPTURN: usize = 9;
const LIGHT: usize = 10;
const DROOP: usize = 11;
const WIGGLE: usize = 12;
const TAPER: usize = 13;
const TRUNK: usize = 14;
const FLARE: usize = 15;
const ROOTS: usize = 16;
const LEAVES: usize = 17;

fn params() -> Vec<ParamDef> {
    vec![
        ParamDef::labelled("seed", "seed", 1.0, 1.0, 999.0, 1.0),
        ParamDef::labelled("grow", "grown", 1.0, 0.0, 1.0, 0.002),
        ParamDef::labelled("crown", "crown depth", 0.66, 0.25, 0.95, 0.005),
        ParamDef::labelled("width", "crown width", 0.85, 0.2, 1.8, 0.005),
        ParamDef::labelled("shape", "crown pointed or flat", 0.0, -1.0, 1.0, 0.01),
        ParamDef::labelled("lumps", "lumpy crown", 0.4, 0.0, 1.0, 0.01),
        ParamDef::labelled("twigs", "twigs", 1100.0, 150.0, 4000.0, 10.0),
        ParamDef::labelled("angle", "branch angle", 38.0, 5.0, 85.0, 0.5),
        ParamDef::labelled("leader", "leader", 0.35, 0.0, 1.0, 0.01),
        ParamDef::labelled("upturn", "turn up", 0.15, -1.0, 1.0, 0.01),
        ParamDef::labelled("light", "reach for light", 0.25, 0.0, 1.0, 0.01),
        ParamDef::labelled("droop", "droop", 0.0, 0.0, 1.0, 0.01),
        ParamDef::labelled("wiggle", "wiggle", 0.2, 0.0, 1.0, 0.01),
        ParamDef::labelled("taper", "taper", 2.4, 1.8, 3.2, 0.01),
        ParamDef::labelled("trunk", "trunk", 0.03, 0.01, 0.07, 0.0005),
        ParamDef::labelled("flare", "flare", 0.7, 0.0, 2.0, 0.01),
        ParamDef::labelled("roots", "roots", 5.0, 0.0, 9.0, 1.0),
        ParamDef::labelled("leaves", "leaves", 0.0, 0.0, 0.5, 0.01),
    ]
}

// ---- small tools -----------------------------------------------------------------------------------
type V = [f64; 3];
fn add(a: V, b: V) -> V { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
fn sub(a: V, b: V) -> V { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn mul(a: V, s: f64) -> V { [a[0] * s, a[1] * s, a[2] * s] }
fn dot(a: V, b: V) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn len(a: V) -> f64 { dot(a, a).sqrt() }
fn unit(a: V) -> V {
    let l = len(a);
    if l > 1e-12 { mul(a, 1.0 / l) } else { [0.0, 1.0, 0.0] }
}
fn lerp(a: V, b: V, t: f64) -> V { add(a, mul(sub(b, a), t)) }
const UP: V = [0.0, 1.0, 0.0];

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

/// One step of a shoot: where it is, where it came from, when (in rounds) it grew.
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
}
const NONE: u32 = u32::MAX;

/// A tree as grown from its sliders: everything `grow` does not change.
struct Grown {
    /// node positions after smoothing and drooping
    p: Vec<V>,
    parent: Vec<u32>,
    /// the length of the path from the foot to each node
    path: Vec<f64>,
    /// the strands the tree is drawn as: each a run of nodes that goes on through the thickest child at
    /// every fork; the first node of a strand that is not the trunk is the node it leaves from
    strands: Vec<Vec<u32>>,
    /// nodes with no children
    tips: usize,
    /// how many tips each node carries when the tree is grown
    carry: Vec<f64>,
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

/// The values a tree is grown from, as read from the sliders.
struct Sliders {
    seed: u64,
    crown: f64,
    width: f64,
    shape: f64,
    lumps: f64,
    twigs: usize,
    angle: f64,
    leader: f64,
    upturn: f64,
    light: f64,
    droop: f64,
    wiggle: f64,
    taper: f64,
    trunk: f64,
    flare: f64,
    roots: usize,
    leaves: f64,
}
impl Sliders {
    fn read(v: &[f64]) -> Self {
        let g = |i: usize, or: f64| v.get(i).copied().filter(|x| x.is_finite()).unwrap_or(or);
        Self {
            seed: g(SEED, 1.0).round().max(0.0) as u64,
            crown: g(CROWN, 0.66).clamp(0.1, 0.98),
            width: g(WIDTH, 0.85).clamp(0.1, 3.0),
            shape: g(SHAPE, 0.0).clamp(-1.0, 1.0),
            lumps: g(LUMPS, 0.4).clamp(0.0, 1.0),
            twigs: g(TWIGS, 1100.0).clamp(20.0, 8000.0) as usize,
            angle: g(ANGLE, 38.0).clamp(0.0, 90.0).to_radians(),
            leader: g(LEADER, 0.35).clamp(0.0, 1.0),
            upturn: g(UPTURN, 0.15).clamp(-1.0, 1.0),
            light: g(LIGHT, 0.25).clamp(0.0, 1.0),
            droop: g(DROOP, 0.0).clamp(0.0, 1.0),
            wiggle: g(WIGGLE, 0.2).clamp(0.0, 1.0),
            taper: g(TAPER, 2.4).clamp(1.2, 4.0),
            trunk: g(TRUNK, 0.03).clamp(0.002, 0.2),
            flare: g(FLARE, 0.7).clamp(0.0, 3.0),
            roots: g(ROOTS, 5.0).round().clamp(0.0, 16.0) as usize,
            leaves: g(LEAVES, 0.0).clamp(0.0, 1.0),
        }
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

fn grow_tree(s: &Sliders) -> Grown {
    let mut rng = Rng::new(s.seed.wrapping_mul(7919).wrapping_add(17));
    let lumps = Lumps::new(&mut rng);
    let (base, half) = (1.0 - s.crown, 0.5 * s.width);
    let reach = REACH * STEP;
    let kill = KILL * STEP;

    // the free points: scattered evenly through the crown's outline
    let inside = |p: V| -> bool {
        let t = (p[1] - base) / s.crown;
        let r = (p[0] * p[0] + p[2] * p[2]).sqrt();
        let az = p[2].atan2(p[0]);
        r <= half * outline(t, s.shape) * (1.0 + 0.45 * s.lumps * lumps.at(az, t))
    };
    let wmax = half * (1.0 + 0.45 * s.lumps) + 0.01;
    let mut free: Vec<V> = Vec::with_capacity(s.twigs);
    let mut tries = 0;
    while free.len() < s.twigs && tries < s.twigs * 200 {
        tries += 1;
        let p = [wmax * rng.sym(), base + s.crown * rng.next(), wmax * rng.sym()];
        if inside(p) { free.push(p); }
    }
    let (lo, hi) = ([-wmax - reach, -0.2, -wmax - reach], [wmax + reach, 1.3, wmax + reach]);
    let mut fgrid = Grid::new(lo, hi, reach);
    for (i, &p) in free.iter().enumerate() { fgrid.put(p, i as u32); }
    let mut alive = vec![true; free.len()];
    // the nearest node to each free point, and how far it is
    let mut nearest: Vec<(u32, f64)> = vec![(NONE, f64::INFINITY); free.len()];

    let mut nodes: Vec<Node> = Vec::new();
    // a node has just grown: it takes the free points it is now the nearest to, and uses up those it reached
    let settle = |i: u32, p: V, alive: &mut Vec<bool>, nearest: &mut Vec<(u32, f64)>| {
        fgrid.near(p, |a| {
            let a = a as usize;
            if !alive[a] { return; }
            let d = len(sub(free[a], p));
            if d < kill { alive[a] = false; } else if d < nearest[a].1 { nearest[a] = (i, d); }
        });
    };

    // the trunk: straight up, a little leaning, until it reaches the crown's free points
    let lean = mul([rng.sym(), 0.0, rng.sym()], 0.06 * s.wiggle);
    let mut p = [0.0, 0.0, 0.0];
    nodes.push(Node { p, parent: NONE, dir: UP, wander: rng.dir(), leader: true, kids: 0 });
    settle(0, p, &mut alive, &mut nearest);
    loop {
        let i = nodes.len() as u32 - 1;
        let near = nearest.iter().zip(&alive).any(|(n, &a)| a && n.1 < reach);
        if near || p[1] > 1.0 { break; }
        let wander = unit(add(mul(nodes[i as usize].wander, 0.85), mul(rng.dir(), 0.5)));
        let dir = unit(add(add(UP, lean), mul(wander, 0.12 * s.wiggle)));
        p = add(p, mul(dir, STEP));
        nodes[i as usize].kids += 1;
        nodes.push(Node { p, parent: i, dir, wander, leader: true, kids: 0 });
        settle(i + 1, p, &mut alive, &mut nearest);
    }

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
                if len(side) < 1e-3 { side = sub(rng.dir(), mul(n.dir, 0.0)); side = sub(side, mul(n.dir, dot(side, n.dir))); }
                let side = unit(side);
                let want = add(mul(n.dir, s.angle.cos()), mul(side, s.angle.sin()));
                dir = unit(lerp(dir, want, 0.65));
            }
            // turning: upwards (or down), out to the light, the main stem keeping on up, and wandering
            let out = unit([n.p[0], 0.0, n.p[2]]);
            let out = if n.p[0].abs() + n.p[2].abs() < 1e-6 { [0.0; 3] } else { out };
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
            nodes.push(Node { p, parent: i as u32, dir, wander, leader, kids: 0 });
            settle(j, p, &mut alive, &mut nearest);
            grew = true;
        }
        if !grew { break; }
    }

    finish(s, nodes, &mut rng)
}

/// Two things growth leaves that no real tree has, cut off: the last step or two of a shoot that turned
/// hard aside after one last free point (a hooked tip), and side shoots that stopped after a step or
/// three (stubs, like thorns).
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
    let nodes = trim_hooks(nodes);
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
    for i in (0..n).rev() {
        if kids[i].is_empty() { tips[i] = 1.0; }
        if parent[i] != NONE { tips[parent[i] as usize] += tips[i]; }
    }
    let all = tips[0].max(1.0);
    // droop: shoots bend down under their own weight, thin ones the most; the bend adds up along a shoot
    if s.droop > 0.0 {
        let orig = p.clone();
        let mut bend = vec![0.0f64; n];
        for i in 1..n {
            let q = parent[i] as usize;
            let d = sub(orig[i], orig[q]);
            let l = len(d);
            let thin = 1.0 - (tips[i] / all).powf(1.0 / s.taper);
            bend[i] = bend[q] + s.droop * 9.0 * thin * thin * l;
            // turn d towards straight down by the bend, but not past it
            let du = unit(d);
            let to_down = dot(du, [0.0, -1.0, 0.0]).clamp(-1.0, 1.0).acos();
            let a = bend[i].min(to_down * 0.92);
            let side = sub([0.0, -1.0, 0.0], mul(du, dot(du, [0.0, -1.0, 0.0])));
            let nd = if len(side) < 1e-9 { du } else { add(mul(du, a.cos()), mul(unit(side), a.sin())) };
            p[i] = add(p[q], mul(nd, l));
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
            let main = *kids[at].iter().max_by(|&&a, &&b| tips[a as usize].total_cmp(&tips[b as usize])).unwrap();
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
    if s.leaves > 0.0 {
        let r = leaf_size(s);
        for i in 0..n {
            if kids[i].is_empty() {
                for k in 0..3 { frame.extend_from_slice(&[p[i][0] + r * [1.0, 0.0, 0.0][k], p[i][1] + r * [0.0, 1.0, 0.0][k], p[i][2] + r * [0.0, 0.0, 1.0][k]]); }
            }
        }
    }
    let tips_n = kids.iter().filter(|k| k.is_empty()).count();
    let trunk = strands.first().and_then(|t| t.iter().find(|&&i| kids[i as usize].len() > 1)).map_or(longest, |&i| path[i as usize]);
    Grown { p, parent, path, strands, tips: tips_n, carry: tips, roots, frame, longest, base: (1.0 - s.crown).max(0.05), trunk }
}

fn leaf_size(s: &Sliders) -> f64 { 0.006 + 0.03 * s.leaves }

/// The tree as `grow` shows it: lines and their radii.
fn draw(g: &Grown, s: &Sliders, grow: f64) -> RawCurve {
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
    let radius = |i: usize| s.trunk * (weight[i] / all).max(age * g.carry[i] / all).powf(1.0 / s.taper);
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
            // the trunk thins a little as it rises, and flares out into its foot
            if k == 0 {
                rad *= 1.0 + 0.25 * (1.0 - q[1] / g.base).max(0.0);
                rad *= 1.0 + 0.4 * s.flare * (-q[1] / (3.0 * trunk_r.max(1e-6))).exp();
                // the foot's round end sits on the ground, among the roots
                if j == 0 { q[1] = q[1].max(rad); }
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
    // leaves: a little cluster at every tip that is there
    if s.leaves > 0.0 {
        let size = leaf_size(s);
        let mut rng = Rng::new(s.seed ^ 0x51ab);
        for i in 0..n {
            if part[i] <= 0.0 || has_kid[i] { continue; }
            let q = at(i);
            let d = if g.parent[i] == NONE { UP } else { unit(sub(g.p[i], g.p[g.parent[i] as usize])) };
            let r = size * part[i].min(1.0) * (0.75 + 0.5 * rng.next());
            // a rosette of leaves: each a short drop, narrow at its stalk, round at its end
            for _ in 0..3 {
                let e = unit(add(d, mul(rng.dir(), 1.1)));
                lines.push([add(q, mul(e, 0.35 * r)), add(q, mul(e, 1.45 * r))].concat());
                radii.push(vec![0.2 * r, 0.42 * r]);
            }
        }
    }
    RawCurve { lines, closed: false, radii, frame: g.frame.clone() }
}

/// Trees grown lately, by their sliders (all but `grow`): moving `grow` only cuts one of them.
static GROWN: Mutex<Vec<(Vec<u64>, Arc<Grown>)>> = Mutex::new(Vec::new());

fn tree(v: &[f64], _o: &CurveOpts) -> RawCurve {
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
    draw(&g, &s, v.get(GROW).copied().unwrap_or(1.0))
}

pub fn family() -> Family {
    let preset = Preset {
        eq: vec!["growth: shoots compete for free space (space colonisation)", "thickness: rⁿ of a stem = Σ rⁿ of its branches"],
        note: "A tree grown from its sliders. Every seed is another tree of the same kind.",
        params: params(),
        tube: Some(0.03),
        fixed_length: true,
        centre: true,
        info: Some(Box::new(|raw, _v| format!("{} branches", raw.lines.len()))),
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
    fn a_tree_has_a_trunk_and_many_twigs() {
        let g = grow_tree(&Sliders::read(&defaults()));
        assert!(g.tips > 100, "{} tips", g.tips);
        assert!(g.strands.len() > 50);
        let raw = draw(&g, &Sliders::read(&defaults()), 1.0);
        assert_eq!(raw.lines.len(), raw.radii.len());
        // the trunk is the thickest strand
        let r0 = raw.radii[0].iter().copied().fold(0.0, f64::max);
        assert!(raw.radii.iter().skip(1).all(|r| r.iter().all(|&x| x <= r0)));
    }

    #[test]
    fn growing_only_cuts_the_tree() {
        let v = defaults();
        let s = Sliders::read(&v);
        let g = grow_tree(&s);
        let (half, whole) = (draw(&g, &s, 0.5), draw(&g, &s, 1.0));
        let count = |r: &RawCurve| r.lines.iter().map(|l| l.len()).sum::<usize>();
        assert!(count(&half) < count(&whole));
        assert_eq!(half.frame, whole.frame);
    }
}
