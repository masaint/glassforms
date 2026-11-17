//! Braids and Celtic interlace: strands that pass over and under each other, as fat glass tubes.
//!
//! A braid is a word in the crossings σᵢ (strand in place i crosses over the one in place i + 1) and their
//! inverses (under). Crossings on places that do not touch are drawn side by side, so a word reads as
//! compactly as it can. A braid can hang open, as a plait does, or be closed: its top joined round to its
//! bottom, which makes it a knot or link (every knot is a closed braid: Alexander's theorem).
//!
//!   plait          σ₁ σ₂⁻¹ σ₁ σ₂⁻¹ ...: the three-strand plait of hair, and its like for more strands
//!   torus braid    (σ₁ σ₂ ... σₙ₋₁)ᵐ, closed: a torus knot or link
//!   random braid   a seeded word
//!
//! Celtic interlace is drawn as Celtic knotwork is: on a grid, strands run on the diagonals, turning back
//! where they meet the border or a break (a short wall on a line of the grid), and at every crossing pass
//! alternately over and under. The breaks are chosen by the seed, mirrored to keep the panel symmetric.

use super::*;
use geo::*;

pub const FAMILY: &str = "braids";

/// The word of a braid: (place, over) per crossing, places counted from 0.
fn word(kind: &str, n: usize, count: usize, seed: f64) -> Vec<(usize, bool)> {
    let gaps = n.saturating_sub(1).max(1);
    let mut r = Rng::new(seed);
    (0..count)
        .map(|k| match kind {
            "plait" => (k % gaps, k % 2 == 0),
            "torus" => (k % gaps, true),
            _ => ((r.next() * gaps as f64) as usize % gaps, r.next() < 0.5),
        })
        .collect()
}

/// The strands of a braid, as lines. `w` apart, a crossing `step` long, rising and sinking `lift`. Closed:
/// the braid bent round a ring and each strand's end joined to the start of the strand in its place.
fn braid(w: &[(usize, bool)], n: usize, lift: f64, close: bool) -> Vec<Vec<f64>> {
    const STEP: f64 = 1.25;
    const SAMPLES: usize = 24;
    // levels: a crossing sits one after the last crossing that touches its places or theirs
    let mut last = vec![0usize; n + 1];
    let mut levels: Vec<Vec<(usize, bool)>> = Vec::new();
    for &(i, over) in w {
        if i + 1 >= n { continue; }
        let lv = last[i.saturating_sub(1)].max(last[i]).max(last[i + 1]).max(last[(i + 2).min(n)]);
        if levels.len() <= lv { levels.resize(lv + 1, Vec::new()); }
        levels[lv].push((i, over));
        last[i] = lv + 1;
        last[i + 1] = lv + 1;
    }
    let m = levels.len().max(1);
    // each strand by where it starts: its place at every level, and how high it rides crossing
    // (paths[s] = list of (from place, to place, height sign) per level)
    let mut at: Vec<usize> = (0..n).collect(); // at[strand] = place
    let mut paths: Vec<Vec<(usize, usize, f64)>> = vec![Vec::with_capacity(m); n];
    for lv in 0..m {
        let mut moves: Vec<(usize, usize, f64)> = at.iter().map(|&p| (p, p, 0.0)).collect();
        if let Some(cs) = levels.get(lv) {
            for &(i, over) in cs {
                for (s, &p) in at.iter().enumerate() {
                    if p == i { moves[s] = (i, i + 1, if over { 1.0 } else { -1.0 }); }
                    if p == i + 1 { moves[s] = (i + 1, i, if over { -1.0 } else { 1.0 }); }
                }
            }
        }
        for (s, mv) in moves.into_iter().enumerate() {
            paths[s].push(mv);
            at[s] = mv.1;
        }
    }
    let x_of = |p: f64| p - 0.5 * (n as f64 - 1.0);
    let total = STEP * m as f64;
    let strand_pts = |s: usize| -> Vec<V> {
        let mut pts = Vec::with_capacity(m * SAMPLES + 1);
        for (lv, &(a, b, sg)) in paths[s].iter().enumerate() {
            for j in 0..SAMPLES {
                let u = j as f64 / SAMPLES as f64;
                let e = 0.5 - 0.5 * (std::f64::consts::PI * u).cos();
                let x = x_of(a as f64 + (b as f64 - a as f64) * e);
                // (level where one crossing meets the next or a straight run: no kink)
                let z = sg * lift * (std::f64::consts::PI * u).sin().powi(2);
                pts.push([x, STEP * (lv as f64 + u), z]);
            }
        }
        let end = paths[s].last().map_or(s, |p| p.1);
        pts.push([x_of(end as f64), total, 0.0]);
        pts
    };
    if !close {
        // open: a little straight lead at either end
        return (0..n)
            .map(|s| {
                let mut p = strand_pts(s);
                let (first, lastp) = (p[0], p[p.len() - 1]);
                p.insert(0, [first[0], first[1] - 0.6 * STEP, 0.0]);
                p.push([lastp[0], lastp[1] + 0.6 * STEP, 0.0]);
                let p: Vec<V> = p.iter().map(|q| [q[0], q[1] - 0.5 * total, q[2]]).collect();
                flat(&p)
            })
            .collect();
    }
    // closed: follow each strand round and on into the strand starting where it ended, until back
    let ring_r = (total / TAU).max(0.9 * n as f64 + 0.6);
    let end_place: Vec<usize> = (0..n).map(|s| paths[s].last().map_or(s, |p| p.1)).collect();
    let mut used = vec![false; n];
    let mut lines = Vec::new();
    for s0 in 0..n {
        if used[s0] { continue; }
        let mut loop_pts: Vec<V> = Vec::new();
        let mut s = s0;
        let mut turns = 0.0;
        while !used[s] {
            used[s] = true;
            let p = strand_pts(s);
            for q in &p[..p.len() - 1] {
                // round the ring: the braid's length goes round, its places outwards, its rises up
                let th = TAU * (turns + q[1] / total);
                let r = ring_r + q[0];
                loop_pts.push([r * th.cos(), q[2], r * th.sin()]);
            }
            turns += 1.0;
            s = end_place[s];
        }
        // (a loop of k strands goes k times round the ring, as a closed braid does)
        let pts = loop_pts;
        let mut v = flat(&pts);
        v.extend_from_slice(&[pts[0][0], pts[0][1], pts[0][2]]);
        lines.push(v);
    }
    lines
}

fn braid_shape(kind: &'static str) -> CurveFn {
    Box::new(move |v: &[f64], _o: &CurveOpts| {
        let n = v[0].round().clamp(2.0, 9.0) as usize;
        let count = v[1].round().max(1.0) as usize;
        let close = v[2] >= 0.5;
        let w = word(kind, n, count, v.get(4).copied().unwrap_or(1.0));
        let lines = braid(&w, n, v[3], close);
        RawCurve { lines, closed: close, ..Default::default() }
    })
}

/// Celtic knotwork on a grid of `cols` x `rows` cells: closed flat curves (before lifting).
fn celtic(cols: usize, rows: usize, breaks: f64, seed: f64, mirror: bool) -> Vec<Vec<[f64; 2]>> {
    // nodes: the middles of the cells' edges, (i/2, j/2) with i + j odd, kept as (i, j) in half units
    let (w, h) = (2 * cols as i32, 2 * rows as i32);
    // breaks: a node on a line of the grid inside the panel may be a wall. Chosen per node of one quarter
    // and mirrored, or per node
    let mut r = Rng::new(seed);
    let mut wall = std::collections::HashSet::new();
    for i in 1..w {
        for j in 1..h {
            if (i + j) % 2 == 0 { continue; }
            let (mi, mj) = (w - i, h - j);
            if mirror && (i > mi || j > mj) { continue; }
            if r.next() < breaks {
                wall.insert((i, j));
                if mirror {
                    wall.insert((mi, j));
                    wall.insert((i, mj));
                    wall.insert((mi, mj));
                }
            }
        }
    }
    // a node on a horizontal line of the grid (j even) turns paths back up or down; on a vertical one
    // (i even), back left or right; where it is a border or a wall
    let turns = |i: i32, j: i32| -> (bool, bool) {
        let horiz = j % 2 == 0 && (j == 0 || j == h || wall.contains(&(i, j)));
        let vert = i % 2 == 0 && (i == 0 || i == w || wall.contains(&(i, j)));
        (vert, horiz)
    };
    // every step (from a node, in a direction) once
    let mut seen = std::collections::HashSet::new();
    let mut curves = Vec::new();
    for i0 in 0..=w {
        for j0 in 0..=h {
            if (i0 + j0) % 2 == 0 { continue; }
            for d0 in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
                let (ni, nj) = (i0 + d0.0, j0 + d0.1);
                if ni < 0 || ni > w || nj < 0 || nj > h || seen.contains(&(i0, j0, d0)) { continue; }
                // walk
                let mut pts: Vec<[f64; 2]> = Vec::new();
                let (mut i, mut j, mut d) = (i0, j0, d0);
                let mut guard = 0;
                loop {
                    if !seen.insert((i, j, d)) || guard > 100000 { break; }
                    // (the step back the other way is the same stretch of the same curve)
                    seen.insert((i + d.0, j + d.1, (-d.0, -d.1)));
                    guard += 1;
                    let (a, b) = (i + d.0, j + d.1);
                    let (fx, fy) = turns(a, b);
                    let mut nd = d;
                    if fx { nd.0 = -nd.0; }
                    if fy { nd.1 = -nd.1; }
                    let (x, y) = (0.5 * a as f64, 0.5 * b as f64);
                    if nd == d {
                        pts.push([x, y]);
                    } else {
                        // a turn: round the corner short of the node
                        let k = 0.36;
                        pts.push([x - 0.5 * k * d.0 as f64, y - 0.5 * k * d.1 as f64]);
                        pts.push([x + 0.5 * k * nd.0 as f64, y + 0.5 * k * nd.1 as f64]);
                    }
                    i = a;
                    j = b;
                    d = nd;
                }
                if pts.len() >= 3 { curves.push(pts); }
            }
        }
    }
    // smooth (Chaikin, keeping straight runs straight) and centre
    let (cx, cy) = (0.5 * cols as f64, 0.5 * rows as f64);
    curves
        .into_iter()
        .map(|mut c| {
            for _ in 0..3 {
                let n = c.len();
                let mut o = Vec::with_capacity(2 * n);
                for k in 0..n {
                    let (a, b) = (c[k], c[(k + 1) % n]);
                    o.push([0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]]);
                    o.push([0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]]);
                }
                c = o;
            }
            c.iter().map(|p| [p[0] - cx, p[1] - cy]).collect()
        })
        .collect()
}

fn celtic_shape(v: &[f64], _o: &CurveOpts) -> RawCurve {
    let (cols, rows) = (v[0].round().clamp(1.0, 24.0) as usize, v[1].round().clamp(1.0, 24.0) as usize);
    let curves = celtic(cols, rows, v[2], v[3], v[5] >= 0.5);
    closed(lift_alternating(&curves, v[4]))
}

fn braid_params(n: f64, count: f64, close: f64, lift: f64, seed: bool) -> Vec<ParamDef> {
    let mut p = vec![
        ParamDef::labelled("strands", "strands", n, 2.0, 9.0, 1.0),
        ParamDef::labelled("crossings", "crossings", count, 1.0, 60.0, 1.0),
        ParamDef::labelled("closed", "closed", close, 0.0, 1.0, 1.0),
        ParamDef::labelled("lift", "over and under", lift, 0.15, 0.7, 0.005),
    ];
    if seed { p.push(ParamDef::new("seed", 3.0, 1.0, 999.0, 1.0)); }
    p
}

fn presets() -> Vec<Preset> {
    let info: fn() -> Option<InfoFn> = || Some(Box::new(|raw: &RawCurve, _v: &[f64]| format!("{} {}", raw.lines.len(), if raw.lines.len() == 1 { "strand" } else { "strands" })));
    vec![
        Preset {
            tube: Some(0.085),
            fixed_length: true,
            rot: Some([10.0, 25.0, 62.0]),
            animate: Some(vec!["lift"]),
            eq: vec!["σ₁ σ₂⁻¹ σ₁ σ₂⁻¹ ...", "each outer strand in turn crosses over into the middle"],
            note: "A plait of hair: the outer strands take turns crossing into the middle.",
            params: braid_params(3.0, 9.0, 0.0, 0.5, false),
            info: info(),
            ..Preset::new("plait", "Plait", Shape::Curve(braid_shape("plait")))
        },
        Preset {
            tube: Some(0.09),
            fixed_length: true,
            rot: Some([62.0, 0.0, 0.0]),
            animate: Some(vec!["lift"]),
            eq: vec!["(σ₁ σ₂ ... σₙ₋₁)ᵐ, closed", "n strands, m(n − 1) crossings: the torus knot or link T(n, m)"],
            note: "Closed, the braid's top is joined round to its bottom. Three strands and eight crossings make a torus knot.",
            params: braid_params(3.0, 8.0, 1.0, 0.5, false),
            info: info(),
            ..Preset::new("torusbraid", "Torus braid", Shape::Curve(braid_shape("torus")))
        },
        Preset {
            tube: Some(0.08),
            fixed_length: true,
            rot: Some([58.0, 0.0, 0.0]),
            animate: Some(vec!["lift"]),
            eq: vec!["a seeded word in σᵢ and σᵢ⁻¹", "closed: a knot or link"],
            note: "A random word of crossings. Closed, every braid is some knot or link.",
            params: braid_params(4.0, 14.0, 1.0, 0.5, true),
            info: info(),
            ..Preset::new("randombraid", "Random braid", Shape::Curve(braid_shape("random")))
        },
        Preset {
            tube: Some(0.07),
            fixed_length: true,
            rot: Some([-24.0, 0.0, 0.0]),
            animate: Some(vec!["lift"]),
            eq: vec!["strands on the diagonals of a grid,", "turning at the border and at breaks; over, under, over ..."],
            note: "Celtic knotwork: the strands bounce off the border and the breaks, and pass alternately over and under.",
            params: vec![
                ParamDef::labelled("cols", "columns", 4.0, 1.0, 12.0, 1.0),
                ParamDef::labelled("rows", "rows", 4.0, 1.0, 12.0, 1.0),
                ParamDef::labelled("breaks", "breaks", 0.25, 0.0, 0.7, 0.01),
                ParamDef::new("seed", 5.0, 1.0, 999.0, 1.0),
                ParamDef::labelled("lift", "over and under", 0.17, 0.08, 0.35, 0.005),
                ParamDef::labelled("mirror", "symmetric", 1.0, 0.0, 1.0, 1.0),
            ],
            info: info(),
            ..Preset::new("celtic", "Celtic knot", Shape::Curve(Box::new(celtic_shape)))
        },
        Preset {
            tube: Some(0.05),
            fixed_length: true,
            rot: Some([-20.0, 0.0, 0.0]),
            animate: Some(vec!["lift"]),
            eq: vec!["a long panel of Celtic interlace,", "its breaks repeating"],
            note: "A border of interlace, as round the pages of an illuminated book.",
            params: vec![
                ParamDef::labelled("cols", "columns", 10.0, 2.0, 24.0, 1.0),
                ParamDef::labelled("rows", "rows", 2.0, 1.0, 4.0, 1.0),
                ParamDef::labelled("breaks", "breaks", 0.3, 0.0, 0.7, 0.01),
                ParamDef::new("seed", 11.0, 1.0, 999.0, 1.0),
                ParamDef::labelled("lift", "over and under", 0.17, 0.08, 0.35, 0.005),
                ParamDef::labelled("mirror", "symmetric", 1.0, 0.0, 1.0, 1.0),
            ],
            info: info(),
            ..Preset::new("border", "Celtic border", Shape::Curve(Box::new(celtic_shape)))
        },
    ]
}

pub fn family() -> Family {
    Family {
        presets: presets(),
        melt: 0.004,
        defaults: vec![("absorb", Setting::Vec3([0.14, 0.07, 0.03]))],
        lights: tube_lights(),
        ..Family::new(FAMILY, "Braids", Kind::Curve)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const O: CurveOpts = CurveOpts { len: 1.0, quick: false, text: "" };

    #[test]
    fn a_closed_torus_braid_is_the_torus_knot_or_link() {
        // (σ1 σ2)^m on three strands closes into gcd(3, m) loops
        let f = braid_shape("torus");
        assert_eq!(f(&[3.0, 8.0, 1.0, 0.4], &O).lines.len(), 1);
        assert_eq!(f(&[3.0, 6.0, 1.0, 0.4], &O).lines.len(), 3);
        // open, every strand is its own
        assert_eq!(f(&[3.0, 8.0, 0.0, 0.4], &O).lines.len(), 3);
    }

    #[test]
    fn celtic_strands_cross_alternately() {
        let raw = celtic_shape(&[4.0, 4.0, 0.25, 5.0, 0.17, 1.0], &O);
        assert!(!raw.lines.is_empty());
        for l in &raw.lines {
            let z: Vec<f64> = l.chunks(3).map(|q| q[2]).collect();
            // over and under both happen along a strand that crosses anything
            let (hi, lo) = (z.iter().cloned().fold(f64::MIN, f64::max), z.iter().cloned().fold(f64::MAX, f64::min));
            assert!(hi > 0.1 && lo < -0.1 || hi.abs() < 1e-9 && lo.abs() < 1e-9, "{hi} {lo}");
        }
    }
}
