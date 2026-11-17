//! Branching growth: coral, and electric discharges (lightning and Lichtenberg figures).
//!
//!   coral      space colonisation (Runions, Lane and Prusinkiewicz, 2007), as the trees grow, with a
//!              coral's rules: free points fill the shape the colony grows into, every point pulls on the
//!              nearest tip within reach, a pulled tip grows a step towards the mean of its pulls, a point
//!              is used up when a tip comes close. Forms:
//!                staghorn  points in a dome; branches turn up hard; thick, blunt, rounded tips, the
//!                          thickness changing little from base to tip (a high pipe-model exponent);
//!                sea fan   points in a thin upright disc: a flat net of fine branches on a short
//!                          stalk; tips that come up against another branch fuse with it (anastomosis);
//!                table     a stalk, then a flat plate of branches spreading out level, tips turning up.
//!   discharge  the dielectric breakdown model (Niemeyer, Pietronero and Wiesmann, 1984) in its fast
//!              form (Kim, Sud, Lin, Manocha 2007): sites next to the discharge on a lattice are
//!              candidates; the potential at each is a running sum of point-charge terms 1 - R1/r over
//!              the discharge, normalised over the candidates; a candidate is taken with probability
//!              ∝ φ^η. Low η branches wildly, high η makes a few long channels. Forms:
//!                lightning    from the top down: the ground's pull added to the potential; the channel
//!                             that reaches the ground is the bright return stroke, drawn thickest;
//!                Lichtenberg  out from a point in every direction (as in a block of acrylic), or
//!                             flattened into a layer.
//!
//! Thickness is the pipe model, as for the trees: a branch's radius^n is the sum of its children's.
//! The grown shape is kept for its sliders and only `grow` is worked out afresh as it moves.

use super::*;
use geo::*;

pub const FAMILY: &str = "branching";
pub const CORAL_FORMS: [&str; 3] = ["staghorn", "sea fan", "table"];
pub const DISCHARGE_FORMS: [&str; 2] = ["lightning", "Lichtenberg"];

const NONE: u32 = u32::MAX;
const UP: V = [0.0, 1.0, 0.0];

/// The names of a choice slider's settings: coral `form`, discharge `kind`.
pub fn choices(id: &str) -> Option<Vec<String>> {
    match id {
        "form" => Some(CORAL_FORMS.iter().map(|s| s.to_string()).collect()),
        "kind" => Some(DISCHARGE_FORMS.iter().map(|s| s.to_string()).collect()),
        _ => None,
    }
}

/// A shape as grown: everything `grow` does not change.
struct Grown {
    p: Vec<V>,
    parent: Vec<u32>,
    /// when each node appears, 0..1 (its parent always before it)
    t: Vec<f64>,
    /// tips each node carries at full size
    tips: Vec<f64>,
    /// a factor on each node's radius (lightning's return stroke)
    boost: Vec<f64>,
    strands: Vec<Vec<u32>>,
    /// a strand that leaves another (it starts inside the node it leaves from)
    side: Vec<bool>,
    /// joins made after growth (a sea fan's fused branches): each drawn once both ends are there
    bridges: Vec<(u32, u32)>,
    frame: Vec<f64>,
}

/// How it is drawn: the pipe model's base radius and exponent, the least share of it a tip keeps.
#[derive(Clone, Copy)]
struct Look {
    r0: f64,
    taper: f64,
    tip: f64,
}

/// Nodes (positions, parents: every parent before its children) to a grown shape: smoothed, counted
/// for the pipe model, cut into strands that run on through their thickest child.
/// `jag` > 0: instead of smoothing, straight runs of about that many nodes between kinks (lightning).
fn finish(mut p: Vec<V>, parent: Vec<u32>, t: Vec<f64>, boost: Vec<f64>, bridges: Vec<(u32, u32)>, jag: usize) -> Grown {
    let n = p.len();
    let mut kids: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (i, &q) in parent.iter().enumerate() {
        if q != NONE { kids[q as usize].push(i as u32); }
    }
    if jag > 0 {
        // kinks at forks, tips and every `jag` nodes along a run; straight between them
        let mut depth = vec![0usize; n];
        for i in 0..n { if parent[i] != NONE { depth[i] = depth[parent[i] as usize] + 1; } }
        let key = |i: usize| parent[i] == NONE || kids[i].len() != 1 || depth[i] % jag == 0;
        for k in 0..n {
            if !key(k) || parent[k] == NONE { continue; }
            let mut chain = Vec::new();
            let mut a = parent[k] as usize;
            while !key(a) {
                chain.push(a);
                a = parent[a] as usize;
            }
            let m = chain.len() + 1;
            for (j, &c) in chain.iter().enumerate() { p[c] = lerp(p[k], p[a], (j + 1) as f64 / m as f64); }
        }
    }
    // growth zig-zags: every node with one child is drawn towards the line from its parent to its child
    for _ in 0..if jag > 0 { 0 } else { 3 } {
        let q = p.clone();
        for i in 0..n {
            if parent[i] == NONE || kids[i].len() != 1 { continue; }
            let mid = mul(add(q[parent[i] as usize], q[kids[i][0] as usize]), 0.5);
            p[i] = lerp(q[i], mid, 0.5);
        }
    }
    let mut tips = vec![0.0f64; n];
    for i in (0..n).rev() {
        if kids[i].is_empty() { tips[i] = 1.0; }
        if parent[i] != NONE { tips[parent[i] as usize] += tips[i]; }
    }
    let mut strands: Vec<Vec<u32>> = Vec::new();
    let mut side: Vec<bool> = Vec::new();
    let mut todo: Vec<Vec<u32>> = (0..n).filter(|&i| parent[i] == NONE).map(|i| vec![i as u32]).collect();
    while let Some(mut line) = todo.pop() {
        side.push(line.len() > 1);
        let mut at = *line.last().unwrap() as usize;
        while !kids[at].is_empty() {
            let main = *kids[at].iter().max_by(|&&a, &&b| (tips[a as usize] * boost[a as usize]).partial_cmp(&(tips[b as usize] * boost[b as usize])).unwrap()).unwrap();
            for &k in &kids[at] {
                if k != main { todo.push(vec![at as u32, k]); }
            }
            line.push(main);
            at = main as usize;
        }
        strands.push(line);
    }
    let frame = p.iter().flat_map(|q| q.iter().copied()).collect();
    Grown { p, parent, t, tips, boost, strands, side, bridges, frame }
}

/// The shape as `grow` shows it.
fn draw(g: &Grown, look: Look, grow: f64, ground: Option<f64>) -> RawCurve {
    let n = g.p.len();
    let grow = grow.clamp(0.0, 1.0);
    let part: Vec<f64> = (0..n)
        .map(|i| {
            let t = g.t[i];
            if t <= grow { return 1.0; }
            let q = g.parent[i];
            if q == NONE { return if grow > 0.0 { 1.0 } else { 0.0 }; }
            let a = g.t[q as usize];
            if a >= grow { 0.0 } else { (grow - a) / (t - a).max(1e-12) }
        })
        .collect();
    // the pipe model on what is there; a stem also thickens with age, so a young shape is not a wire
    let mut weight = vec![0.0f64; n];
    let mut has_kid = vec![false; n];
    for i in 0..n {
        if part[i] > 0.0 && g.parent[i] != NONE { has_kid[g.parent[i] as usize] = true; }
    }
    for i in (0..n).rev() {
        if part[i] <= 0.0 { continue; }
        if !has_kid[i] { weight[i] += part[i].min(1.0); }
        if g.parent[i] != NONE { weight[g.parent[i] as usize] += weight[i]; }
    }
    let all = g.tips.iter().zip(&g.parent).filter(|(_, q)| **q == NONE).map(|(t, _)| *t).sum::<f64>().max(1.0);
    let age = 0.6 * grow.powf(1.5);
    let radius = |i: usize| -> f64 {
        let w = (weight[i] / all).max(age * g.tips[i] / all);
        // (a boosted run, lightning's return stroke, keeps nearly one thickness from top to bottom)
        let k = w.powf(1.0 / look.taper);
        let k = if g.boost[i] > 1.0 { g.boost[i] * (0.75 + 0.25 * k) } else { k };
        (look.r0 * k).max(look.r0 * look.tip * part[i].min(1.0).sqrt())
    };
    let at = |i: usize| -> V {
        let q = g.parent[i];
        if part[i] >= 1.0 || q == NONE { g.p[i] } else { lerp(g.p[q as usize], g.p[i], part[i]) }
    };
    let mut lines = Vec::new();
    let mut radii = Vec::new();
    for (k, strand) in g.strands.iter().enumerate() {
        let (mut l, mut r) = (Vec::new(), Vec::new());
        let side = g.side[k];
        for (j, &i) in strand.iter().enumerate() {
            let i = i as usize;
            if part[i] <= 0.0 { break; }
            let mut q = at(i);
            // a side strand starts inside the node it leaves from, as thick as itself
            let rad = if j == 0 && side && strand.len() > 1 { radius(strand[1] as usize) } else { radius(i) };
            // a foot on the ground goes on below it, where the shape is cut flat
            if j == 0 && !side && ground.is_some() { q[1] -= 2.0 * rad; }
            l.extend_from_slice(&q);
            r.push(rad);
        }
        let first = usize::from(side);
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
    for &(a, b) in &g.bridges {
        let (a, b) = (a as usize, b as usize);
        if part[a] < 1.0 || part[b] < 1.0 { continue; }
        let ra = radius(a).min(radius(b));
        lines.push([g.p[a], g.p[b]].iter().flat_map(|q| q.iter().copied()).collect());
        radii.push(vec![ra, ra]);
    }
    RawCurve { lines, closed: false, radii, frame: g.frame.clone(), ground, ..Default::default() }
}

// ---- coral ----------------------------------------------------------------------------------------

const C_SEED: usize = 0;
const C_GROW: usize = 1;
const C_FORM: usize = 2;
const C_POINTS: usize = 3;
const C_SPREAD: usize = 4;
const C_HEIGHT: usize = 5;
const C_THICK: usize = 6;
const C_TAPER: usize = 7;
const C_TIPS: usize = 8;
const C_UPTURN: usize = 9;
const C_WIGGLE: usize = 10;

fn coral_params() -> Vec<ParamDef> {
    vec![
        ParamDef::labelled("seed", "seed", 3.0, 1.0, 999.0, 1.0),
        ParamDef::labelled("grow", "grown", 1.0, 0.0, 1.0, 0.002),
        ParamDef::labelled("form", "form", 0.0, 0.0, (CORAL_FORMS.len() - 1) as f64, 1.0),
        ParamDef::labelled("points", "branches", 160.0, 40.0, 1500.0, 10.0),
        ParamDef::labelled("spread", "width", 1.1, 0.4, 2.0, 0.01),
        ParamDef::labelled("height", "height", 0.9, 0.3, 1.6, 0.01),
        ParamDef::labelled("thick", "thickness", 0.075, 0.02, 0.14, 0.001),
        ParamDef::labelled("taper", "taper", 4.5, 1.8, 8.0, 0.05),
        ParamDef::labelled("tips", "blunt tips", 0.6, 0.0, 1.0, 0.01),
        ParamDef::labelled("upturn", "turn up", 0.5, -0.5, 1.0, 0.01),
        ParamDef::labelled("wiggle", "wiggle", 0.25, 0.0, 1.0, 0.01),
    ]
}

/// A uniform grid of buckets for finding near nodes.
struct Grid {
    cell: f64,
    map: std::collections::HashMap<(i32, i32, i32), Vec<u32>>,
}
impl Grid {
    fn new(cell: f64) -> Self { Self { cell, map: std::collections::HashMap::new() } }
    fn key(&self, p: V) -> (i32, i32, i32) { ((p[0] / self.cell).floor() as i32, (p[1] / self.cell).floor() as i32, (p[2] / self.cell).floor() as i32) }
    fn put(&mut self, p: V, i: u32) { let k = self.key(p); self.map.entry(k).or_default().push(i); }
    fn near(&self, p: V, mut f: impl FnMut(u32)) {
        let k = self.key(p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(v) = self.map.get(&(k.0 + dx, k.1 + dy, k.2 + dz)) { for &i in v { f(i); } }
                }
            }
        }
    }
}

fn grow_coral(v: &[f64]) -> Grown {
    let form = (v[C_FORM].round().max(0.0) as usize).min(CORAL_FORMS.len() - 1);
    let mut rng = Rng::new(v[C_SEED] * 7.0 + form as f64 * 1013.0);
    let (w, h) = (v[C_SPREAD], v[C_HEIGHT] * if form == 0 { 1.35 } else { 1.0 });
    let count = v[C_POINTS].round().max(10.0) as usize;
    let (upturn, wiggle) = (v[C_UPTURN], v[C_WIGGLE]);
    // a step: finer for the fan's net
    let step = match form { 1 => 0.022, _ => 0.03 };
    let reach = step * if form == 1 { 9.0 } else { 16.0 };
    let kill = step * if form == 0 { 4.0 } else { 2.6 };
    // where the colony grows, and where its stalk stops
    let stalk = match form { 1 => 0.12 * h, 2 => 0.45 * h, _ => 0.0 };
    let inside = |p: V| -> bool {
        match form {
            // a dome
            0 => { let (x, y, z) = (p[0] / w, p[1] / h, p[2] / w); y > 0.08 && x * x + y * y + z * z <= 1.0 }
            // an upright half-disc, thin
            1 => { let (x, y) = (p[0] / w, (p[1] - stalk) / (h - stalk)); y > 0.0 && x * x + y * y <= 1.0 && p[2].abs() < 0.012 }
            // a flat round plate, rising a little to its rim
            _ => { let r = (p[0] * p[0] + p[2] * p[2]).sqrt() / w; let y0 = stalk + 0.12 * h * r * r; r <= 1.0 && p[1] > y0 && p[1] < y0 + 0.12 * h }
        }
    };
    let mut free: Vec<V> = Vec::with_capacity(count);
    let mut tries = 0;
    let want = count * match form { 0 => 2, 1 => 12, _ => 4 };
    while free.len() < want && tries < want * 300 {
        tries += 1;
        let p = [w * rng.sym(), h * rng.next(), w * rng.sym()];
        if inside(p) { free.push(p); }
    }
    let mut alive = vec![true; free.len()];
    let mut nodes: Vec<V> = Vec::new();
    let mut parent: Vec<u32> = Vec::new();
    let mut dirs: Vec<V> = Vec::new();
    let mut wander: Vec<V> = Vec::new();
    let mut kids: Vec<u32> = Vec::new();
    let mut grid = Grid::new(reach);
    let push = |nodes: &mut Vec<V>, grid: &mut Grid, p: V, q: u32, d: V, wd: V, parent: &mut Vec<u32>, dirs: &mut Vec<V>, wander: &mut Vec<V>, kids: &mut Vec<u32>| {
        let i = nodes.len() as u32;
        nodes.push(p);
        parent.push(q);
        dirs.push(d);
        wander.push(wd);
        kids.push(0);
        if q != NONE { kids[q as usize] += 1; }
        grid.put(p, i);
        i
    };
    // the foot, and the stalk up to the colony
    let mut at = push(&mut nodes, &mut grid, [0.0, 0.0, 0.0], NONE, UP, UP, &mut parent, &mut dirs, &mut wander, &mut kids);
    let mut y = 0.0;
    while y + step < stalk.max(0.03) {
        y += step;
        let p = [0.004 * rng.sym(), y, 0.004 * rng.sym()];
        at = push(&mut nodes, &mut grid, p, at, UP, UP, &mut parent, &mut dirs, &mut wander, &mut kids);
    }
    let _ = at;
    // used up by the stalk
    for (a, f) in free.iter().enumerate() {
        if nodes.iter().any(|q| len(sub(*f, *q)) < kill) { alive[a] = false; }
    }
    let mut fgrid = Grid::new(reach);
    for (i, &f) in free.iter().enumerate() { fgrid.put(f, i as u32); }
    let mut nearest: Vec<(u32, f64)> = vec![(NONE, f64::INFINITY); free.len()];
    let settle = |i: u32, p: V, alive: &mut Vec<bool>, nearest: &mut Vec<(u32, f64)>| {
        fgrid.near(p, |a| {
            let a = a as usize;
            if !alive[a] { return; }
            let d = len(sub(free[a], p));
            if d < kill { alive[a] = false; } else if d < nearest[a].1 { nearest[a] = (i, d); }
        });
    };
    for i in 0..nodes.len() { settle(i as u32, nodes[i], &mut alive, &mut nearest); }
    let mut pull: Vec<(V, u32)> = Vec::new();
    for _round in 0..400 {
        pull.clear();
        pull.resize(nodes.len(), ([0.0; 3], 0));
        let mut any = false;
        for (a, &(nd, d)) in nearest.iter().enumerate() {
            if !alive[a] || nd == NONE || d >= reach { continue; }
            let q = &mut pull[nd as usize];
            q.0 = add(q.0, unit(sub(free[a], nodes[nd as usize])));
            q.1 += 1;
            any = true;
        }
        if !any { break; }
        let count = nodes.len();
        let mut grew = false;
        for i in 0..count {
            let (sum, k) = pull[i];
            if k == 0 || kids[i] >= 2 { continue; }
            let mut dir = unit(sum);
            // a coral's branches turn up, wander a little, and (a fan's) keep to its plane
            let wd = unit(add(mul(wander[i], 0.9), mul(rng.dir(), 0.3)));
            let upw = if form == 2 { upturn * 0.35 } else { upturn };
            dir = unit(add(add(dir, mul(UP, if form == 0 { 0.9 } else { 0.5 } * upw)), mul(wd, 0.4 * wiggle)));
            if form == 1 { dir[2] *= 0.2; dir = unit(dir); }
            let p = add(nodes[i], mul(dir, step));
            // never grow back into a shoot already there
            let mut crowd = false;
            grid.near(p, |m| if len(sub(nodes[m as usize], p)) < 0.6 * step { crowd = true; });
            if crowd { continue; }
            let j = push(&mut nodes, &mut grid, p, i as u32, dir, wd, &mut parent, &mut dirs, &mut wander, &mut kids);
            settle(j, p, &mut alive, &mut nearest);
            grew = true;
        }
        if !grew { break; }
    }
    // stubs of a step or two left by the growth are cut off
    let (nodes, parent) = trim(nodes, parent, 2);
    let n = nodes.len();
    // when each node appears: every shoot lengthening at the same pace
    let mut path = vec![0.0f64; n];
    for i in 1..n { if parent[i] != NONE { path[i] = path[parent[i] as usize] + len(sub(nodes[i], nodes[parent[i] as usize])); } }
    let longest = path.iter().copied().fold(1e-9, f64::max);
    let t: Vec<f64> = path.iter().map(|d| d / longest).collect();
    // a fan's tips that come up against another branch fuse with it
    let mut bridges = Vec::new();
    if form == 1 {
        let mut kc = vec![0u32; n];
        for &q in &parent { if q != NONE { kc[q as usize] += 1; } }
        let mut g = Grid::new(4.0 * step);
        for (i, &q) in nodes.iter().enumerate() { g.put(q, i as u32); }
        for i in 0..n {
            if kc[i] != 0 { continue; }
            // its own stem, some way back, is not another branch
            let mut own = std::collections::HashSet::new();
            let mut a = i as u32;
            for _ in 0..10 { own.insert(a); if parent[a as usize] == NONE { break; } a = parent[a as usize]; }
            let mut best = (NONE, 3.2 * step);
            g.near(nodes[i], |m| {
                if own.contains(&m) || parent[m as usize] == i as u32 { return; }
                let d = len(sub(nodes[m as usize], nodes[i]));
                if d < best.1 { best = (m, d); }
            });
            if best.0 != NONE { bridges.push((i as u32, best.0)); }
        }
    }
    let boost = vec![1.0; n];
    finish(nodes, parent, t, boost, bridges, 0)
}

/// Cut off side shoots that stopped after `short` steps or fewer (and tips that hooked hard aside).
fn trim(mut nodes: Vec<V>, mut parent: Vec<u32>, short: usize) -> (Vec<V>, Vec<u32>) {
    for _ in 0..short {
        let n = nodes.len();
        let mut kids = vec![0u32; n];
        for &q in &parent { if q != NONE { kids[q as usize] += 1; } }
        let mut keep = vec![true; n];
        for i in 0..n {
            let q = parent[i];
            if kids[i] > 0 || q == NONE { continue; }
            // a tip whose parent forks: a stub
            if kids[q as usize] > 1 { keep[i] = false; }
        }
        let mut to = vec![NONE; n];
        let (mut out, mut op) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            if !keep[i] { continue; }
            to[i] = out.len() as u32;
            out.push(nodes[i]);
            op.push(if parent[i] == NONE { NONE } else { to[parent[i] as usize] });
        }
        nodes = out;
        parent = op;
    }
    (nodes, parent)
}

static CORAL: Memo<Grown> = Memo::new();

fn coral(v: &[f64], _o: &CurveOpts) -> RawCurve {
    let g = CORAL.get(v, &[C_GROW], "", || grow_coral(v));
    let form = (v[C_FORM].round().max(0.0) as usize).min(CORAL_FORMS.len() - 1);
    let fan = if form == 1 { 0.45 } else { 1.0 };
    let look = Look { r0: v[C_THICK] * fan, taper: v[C_TAPER], tip: 0.15 + 0.6 * v[C_TIPS] };
    draw(&g, look, v.get(C_GROW).copied().unwrap_or(1.0), Some(0.0))
}

// ---- discharge -----------------------------------------------------------------------------------

const D_SEED: usize = 0;
const D_GROW: usize = 1;
const D_KIND: usize = 2;
const D_POINTS: usize = 3;
const D_ETA: usize = 4;
const D_THICK: usize = 5;
const D_JITTER: usize = 6;
const D_FLAT: usize = 7;
const D_STROKE: usize = 8;
const D_LEAST: usize = 9;

fn discharge_params() -> Vec<ParamDef> {
    vec![
        ParamDef::labelled("seed", "seed", 2.0, 1.0, 999.0, 1.0),
        ParamDef::labelled("grow", "grown", 1.0, 0.0, 1.0, 0.002),
        ParamDef::labelled("kind", "kind", 0.0, 0.0, (DISCHARGE_FORMS.len() - 1) as f64, 1.0),
        ParamDef::labelled("points", "points", 1500.0, 200.0, 4000.0, 10.0),
        ParamDef::labelled("eta", "η (fewer branches)", 2.2, 0.5, 6.0, 0.05),
        ParamDef::labelled("thick", "thickness", 0.028, 0.008, 0.07, 0.0005),
        ParamDef::labelled("jitter", "jitter", 0.7, 0.0, 1.0, 0.01),
        ParamDef::labelled("flat", "flatten", 0.6, 0.0, 1.0, 0.01),
        ParamDef::labelled("stroke", "main channel", 1.3, 1.0, 3.0, 0.01),
        ParamDef::labelled("least", "shortest branch", 8.0, 0.0, 40.0, 1.0),
    ]
}

fn grow_discharge(v: &[f64]) -> Grown {
    let kind = (v[D_KIND].round().max(0.0) as usize).min(DISCHARGE_FORMS.len() - 1);
    let lightning = kind == 0;
    let mut rng = Rng::new(v[D_SEED] * 13.0 + kind as f64 * 977.0);
    let n_max = v[D_POINTS].round().max(10.0) as usize;
    let eta = v[D_ETA];
    let flat = v[D_FLAT];
    // the lattice: lightning in a tall box from its top, a figure in a ball from its middle
    let (hgt, rad) = if lightning { (52i32, 26i32) } else { (0, 26i32) };
    let start = if lightning { [0, hgt, 0] } else { [0, 0, 0] };
    const R1: f64 = 0.5;
    let mut pts: Vec<[i32; 3]> = vec![start];
    let mut parent: Vec<u32> = vec![NONE];
    let mut taken: std::collections::HashSet<[i32; 3]> = std::collections::HashSet::new();
    taken.insert(start);
    // candidates: site, potential (running sum over the discharge), parent, still a candidate
    let mut cand: Vec<([i32; 3], f64, u32, bool)> = Vec::new();
    let mut is_cand: std::collections::HashMap<[i32; 3], usize> = std::collections::HashMap::new();
    let inside = |c: [i32; 3]| -> bool {
        if lightning { c[1] >= 0 && c[1] <= hgt && c[0].abs() <= rad && c[2].abs() <= rad } else { c[0] * c[0] + c[1] * c[1] + c[2] * c[2] <= rad * rad }
    };
    let dist = |a: [i32; 3], b: [i32; 3]| (((a[0] - b[0]).pow(2) + (a[1] - b[1]).pow(2) + (a[2] - b[2]).pow(2)) as f64).sqrt();
    let neigh: Vec<[i32; 3]> = {
        let mut v = Vec::new();
        for dx in -1..=1 { for dy in -1..=1 { for dz in -1..=1 { if (dx, dy, dz) != (0, 0, 0) { v.push([dx, dy, dz]); } } } }
        v
    };
    let add_cands = |i: usize, pts: &Vec<[i32; 3]>, taken: &std::collections::HashSet<[i32; 3]>, cand: &mut Vec<([i32; 3], f64, u32, bool)>, is_cand: &mut std::collections::HashMap<[i32; 3], usize>| {
        let c = pts[i];
        for d in &neigh {
            let s = [c[0] + d[0], c[1] + d[1], c[2] + d[2]];
            if taken.contains(&s) || is_cand.contains_key(&s) || !inside(s) { continue; }
            let phi: f64 = pts.iter().map(|&q| 1.0 - R1 / dist(s, q)).sum();
            is_cand.insert(s, cand.len());
            cand.push((s, phi, i as u32, true));
        }
    };
    add_cands(0, &pts, &taken, &mut cand, &mut is_cand);
    let mut hit_ground = None;
    let mut w: Vec<f64> = Vec::new();
    // (a bolt goes on until it strikes, within reason)
    while pts.len() < n_max || (lightning && hit_ground.is_none() && pts.len() < 6 * n_max) {
        // the candidates' potentials, normalised, and the ground's pull
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        let np = pts.len() as f64;
        for c in cand.iter().filter(|c| c.3) {
            lo = lo.min(c.1 / np);
            hi = hi.max(c.1 / np);
        }
        if lo > hi { break; }
        let span = (hi - lo).max(1e-12);
        w.clear();
        let mut total = 0.0;
        for c in &cand {
            if !c.3 { w.push(0.0); continue; }
            let mut phi = (c.1 / np - lo) / span;
                        if flat > 0.0 { phi *= (1.0 - flat * (c.0[2].abs() as f64 / 6.0).min(1.0)).max(0.0); }
            let mut x = phi.max(0.0).powf(eta);
            // the ground's pull: the deeper a site, the likelier (the field is strongest at the leader's tip)
            if lightning { x *= (4.0 * (1.0 - c.0[1] as f64 / hgt as f64)).exp(); }
            total += x;
            w.push(total);
        }
        if total <= 0.0 { break; }
        // a few sites a round (one site at a time is the model; a few keeps it quick and looks the same)
        let take = 1 + pts.len() / 70;
        let mut chosen: Vec<usize> = Vec::with_capacity(take);
        for _ in 0..take {
            let r = rng.next() * total;
            let k = w.partition_point(|&s| s < r).min(cand.len() - 1);
            if cand[k].3 && !chosen.contains(&k) { chosen.push(k); }
        }
        for k in chosen {
            let (s, _, q, _) = cand[k];
            cand[k].3 = false;
            let i = pts.len();
            pts.push(s);
            parent.push(q);
            taken.insert(s);
            for c in cand.iter_mut().filter(|c| c.3) { c.1 += 1.0 - R1 / dist(c.0, s); }
            add_cands(i, &pts, &taken, &mut cand, &mut is_cand);
            if lightning && s[1] == 0 && hit_ground.is_none() { hit_ground = Some(i); }
        }
        if hit_ground.is_some() { break; }
        if cand.len() > 4 * pts.len() + 2000 {
            // forget used candidates now and then
            let keep: Vec<_> = cand.drain(..).filter(|c| c.3).collect();
            is_cand.clear();
            for (j, c) in keep.iter().enumerate() { is_cand.insert(c.0, j); }
            // (sites once candidates and taken stay out: taken holds them)
            cand = keep;
        }
    }
    let n = pts.len();
    // the lattice hidden: every site moved a little at random, before smoothing
    let jit = v[D_JITTER] * 0.45;
    let sc = 1.0 / if lightning { hgt as f64 } else { rad as f64 };
    let p: Vec<V> = pts.iter().map(|c| [(c[0] as f64 + jit * rng.sym()) * sc, (c[1] as f64 + jit * rng.sym()) * sc, (c[2] as f64 + jit * rng.sym()) * sc]).collect();
    // the return stroke: the channel from the top to where it struck, drawn thicker
    let mut boost = vec![1.0; n];
    if let Some(mut a) = hit_ground {
        loop {
            boost[a] = v[D_STROKE];
            if parent[a] == NONE { break; }
            a = parent[a] as usize;
        }
    }
    // side branches shorter than `least` sites are cut off: a few long channels read better than a fur
    // of stubs (the return stroke is never cut)
    let least = v[D_LEAST].round().max(0.0) as usize;
    let mut height = vec![0usize; n];
    for i in (1..n).rev() {
        let q = parent[i] as usize;
        height[q] = height[q].max(height[i] + 1);
    }
    let mut kids: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 1..n { kids[parent[i] as usize].push(i); }
    let mut cut = vec![false; n];
    for q in 0..n {
        if kids[q].len() < 2 { continue; }
        let main = *kids[q].iter().max_by_key(|&&k| (boost[k] > 1.0, height[k])).unwrap();
        for &k in &kids[q] {
            if k != main && boost[k] == 1.0 && height[k] + 1 < least { cut[k] = true; }
        }
    }
    let mut to = vec![NONE; n];
    let (mut kp, mut kq, mut kb) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        if parent[i] != NONE && (cut[i] || to[parent[i] as usize] == NONE) { continue; }
        to[i] = kp.len() as u32;
        kp.push(p[i]);
        kq.push(if parent[i] == NONE { NONE } else { to[parent[i] as usize] });
        kb.push(boost[i]);
    }
    let (p, parent, boost) = (kp, kq, kb);
    let m = p.len();
    let t: Vec<f64> = (0..m).map(|i| i as f64 / (m.max(2) - 1) as f64).collect();
    finish(p, parent, t, boost, Vec::new(), if lightning { 4 } else { 0 })
}

static DISCHARGE: Memo<Grown> = Memo::new();

fn discharge(v: &[f64], _o: &CurveOpts) -> RawCurve {
    let g = DISCHARGE.get(v, &[D_GROW], "", || grow_discharge(v));
    let look = Look { r0: v[D_THICK], taper: 2.4, tip: 0.18 };
    draw(&g, look, v.get(D_GROW).copied().unwrap_or(1.0), None)
}

pub fn family() -> Family {
    let coral = Preset {
        eq: vec!["growth: tips compete for free space (space colonisation)", "thickness: rⁿ of a stem = Σ rⁿ of its branches"],
        note: "A coral colony grown from its sliders. Staghorn branches turn up and end blunt; a sea fan is a flat net whose branches fuse where they meet; a table spreads out level on a stalk.",
        params: coral_params(),
        tube: Some(0.03),
        fixed_length: true,
        info: Some(Box::new(|raw, _v| format!("{} branches", raw.lines.len()))),
        animate: Some(vec!["grow"]),
        rot: Some([4.0, 0.0, 0.0]),
        ..Preset::new("coral", "Coral", Shape::Curve(Box::new(coral)))
    };
    let bolt = Preset {
        eq: vec!["φ(site) = Σ (1 − R₁/r) over the discharge", "P(site) ∝ φ^η"],
        note: "Electric breakdown grown site by site: lightning finding its way to the ground, or a Lichtenberg figure branching out through a block. Low η branches wildly; high η makes a few long channels.",
        params: discharge_params(),
        tube: Some(0.03),
        fixed_length: true,
        info: Some(Box::new(|raw, _v| format!("{} channels", raw.lines.len()))),
        animate: Some(vec!["grow"]),
        rot: Some([0.0, 0.0, 0.0]),
        ..Preset::new("lightning", "Lightning", Shape::Curve(Box::new(discharge)))
    };
    Family {
        presets: vec![coral, bolt],
        melt: 0.004,
        defaults: vec![
            ("ior", Setting::Num(1.5)), ("absorb", Setting::Vec3([0.16, 0.08, 0.035])),
            ("dist", Setting::Num(6.6)), ("fov", Setting::Num(26.0)), ("rot", Setting::Vec3([4.0, 0.0, 0.0])),
        ],
        lights: solid_lights(),
        ..Family::new(FAMILY, "Branching", Kind::Curve)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults(p: Vec<ParamDef>) -> Vec<f64> { p.iter().map(|d| d.value).collect() }

    #[test]
    fn every_coral_form_grows_and_the_foot_is_thickest() {
        for f in 0..CORAL_FORMS.len() {
            let mut v = defaults(coral_params());
            v[C_FORM] = f as f64;
            let g = grow_coral(&v);
            assert!(g.p.len() > 60, "form {f}: {} nodes", g.p.len());
            let raw = draw(&g, Look { r0: 0.05, taper: 3.0, tip: 0.3 }, 1.0, Some(0.0));
            let foot = raw.radii[0][0];
            let most = raw.radii.iter().flatten().copied().fold(0.0, f64::max);
            assert!(foot >= 0.999 * most, "form {f}: foot {foot} thickest {most}");
        }
    }

    #[test]
    fn every_discharge_grows_and_lightning_reaches_the_ground() {
        for k in 0..DISCHARGE_FORMS.len() {
            let mut v = defaults(discharge_params());
            v[D_KIND] = k as f64;
            let g = grow_discharge(&v);
            assert!(g.p.len() > 100, "kind {k}: {} sites", g.p.len());
            if k == 0 { assert!(g.boost.iter().any(|&b| b > 1.0), "the bolt never struck"); }
        }
    }

    #[test]
    fn growing_only_cuts_and_keeps_the_frame() {
        let v = defaults(coral_params());
        let whole = coral(&v, &CurveOpts { len: 1.0, quick: false, text: "" });
        let mut h = v.clone();
        h[C_GROW] = 0.4;
        let half = coral(&h, &CurveOpts { len: 1.0, quick: false, text: "" });
        assert_eq!(half.frame, whole.frame);
        let count = |r: &RawCurve| r.lines.iter().map(|l| l.len()).sum::<usize>();
        assert!(count(&half) < count(&whole));
    }

    #[test]
    fn the_same_seed_grows_the_same_shape() {
        let v = defaults(discharge_params());
        let a = grow_discharge(&v);
        let b = grow_discharge(&v);
        assert_eq!(a.frame, b.frame);
    }

    #[test]
    fn growing_is_quick() {
        let mut worst = 0.0f64;
        for f in 0..CORAL_FORMS.len() {
            let mut v = defaults(coral_params());
            v[C_FORM] = f as f64;
            let t = std::time::Instant::now();
            let g = grow_coral(&v);
            println!("coral {f}: {} nodes, {:.1} ms", g.p.len(), 1e3 * t.elapsed().as_secs_f64());
            worst = worst.max(t.elapsed().as_secs_f64());
        }
        for k in 0..DISCHARGE_FORMS.len() {
            let mut v = defaults(discharge_params());
            v[D_KIND] = k as f64;
            let t = std::time::Instant::now();
            let g = grow_discharge(&v);
            println!("lowest {:.3}", g.p.iter().map(|q| q[1]).fold(9.0, f64::min));
            println!("discharge {k}: {} sites, {:.1} ms", g.p.len(), 1e3 * t.elapsed().as_secs_f64());
            worst = worst.max(t.elapsed().as_secs_f64());
        }
        println!("slowest growth {:.1} ms", 1e3 * worst);
        // (debug builds are slow: this only catches growth gone quadratic)
        assert!(worst < 20.0, "{worst} s");
    }
}
