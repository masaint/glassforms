//! The swept-tube formula's field, measured on the CPU (`presets::tubes::cpu`, which a GPU test holds the
//! shader to): it must be one smooth field, whichever cell a point is read through, and a distance far
//! from the shape as well as near it; and it must not be much more than the distance (a step of the
//! field's own length must not pass the surface by more than a part of a cell).
//!
//! The reference is brute force over every fitted piece: the exact least distance to each span (a dense
//! scan, every dip refined), and for the melt the same line integrals the formula defines, by fine
//! quadrature.
//!
//! History: the first formula (04d2458) capped the far field at about ten voxels, which the ambient
//! occlusion and the shadows read as walls, and its melt was 2.77 times narrower than the bake it replaced
//! (docs/project-notes/plans/tube_field_fix.md). These tests are the guard against either coming back.
#![allow(clippy::needless_range_loop)]

use gf_core::presets::tubes::{cpu, HOLD};
use gf_core::state::{Detail, ShapeJob, State};

type Table = Vec<[f32; 4]>;

fn table(args: &[&str], detail: Detail, res: u32) -> Table {
    let pairs: Vec<(String, String)> = args.iter().map(|s| { let (k, v) = s.split_once('=').unwrap(); (k.to_string(), v.to_string()) }).collect();
    let mut state = State::with_args(&pairs, true);
    let mut job = state.shape_job(detail, Some(res)).unwrap();
    job.prepare_formula(&state.families).unwrap();
    let ShapeJob::Field { table, .. } = job.shape else { panic!("no formula") };
    table.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
}

// ---- the reference -------------------------------------------------------------------------------

#[derive(Clone)]
struct Piece {
    kind: u32,
    /// a span's four controls (a line's laid out as a cubic's), or a ball's centre and radius first
    c: [[f64; 4]; 4],
    /// the radius of the capsule about its chord that holds it
    bound: f64,
}

fn pieces(tbl: &Table) -> Vec<Piece> {
    let mut out = Vec::new();
    let mut i = tbl[2][3] as usize;
    let end = tbl[0][0] as usize;
    let v = |a: [f32; 4]| a.map(f64::from);
    while i < end {
        let n = tbl[i][3] as usize;
        let tag = tbl[i + 2];
        let kind = tag[0] as u32;
        let mut c = [[0.0; 4]; 4];
        match kind {
            110 => { for k in 0..4 { c[k] = v(tbl[i + 3 + k]); } }
            111 => {
                let (a, d) = (v(tbl[i + 3]), v(tbl[i + 4]));
                c = [a, std::array::from_fn(|k| (2.0 * a[k] + d[k]) / 3.0), std::array::from_fn(|k| (a[k] + 2.0 * d[k]) / 3.0), d];
            }
            101 => { c[0] = v(tbl[i + 3]); }
            _ => {}
        }
        out.push(Piece { kind, c, bound: f64::from(tag[3]) });
        i += 2 + n;
    }
    out
}

fn bezier(c: &[[f64; 4]; 4], t: f64) -> [f64; 4] {
    let u = 1.0 - t;
    let w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
    std::array::from_fn(|k| w[0] * c[0][k] + w[1] * c[1][k] + w[2] * c[2][k] + w[3] * c[3][k])
}

fn bezier_rate(c: &[[f64; 4]; 4], t: f64) -> [f64; 4] {
    let u = 1.0 - t;
    std::array::from_fn(|k| 3.0 * (u * u * (c[1][k] - c[0][k]) + 2.0 * u * t * (c[2][k] - c[1][k]) + t * t * (c[3][k] - c[2][k])))
}

fn outside(c: &[[f64; 4]; 4], p: [f64; 3], t: f64) -> f64 {
    let q = bezier(c, t);
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt() - q[3].max(0.0)
}

/// The least of |p - C(t)| - r(t) over a piece: 64 samples, every dip refined by golden section.
fn piece_exact(pc: &Piece, p: [f64; 3]) -> f64 {
    if pc.kind == 101 {
        let q = pc.c[0];
        return ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt() - q[3];
    }
    const N: usize = 64;
    let mut f = [0.0; N + 1];
    for j in 0..=N { f[j] = outside(&pc.c, p, j as f64 / N as f64); }
    let mut best = f64::MAX;
    for j in 0..=N {
        if !((j == 0 || f[j] <= f[j - 1]) && (j == N || f[j] <= f[j + 1])) { continue; }
        let (mut a, mut b) = ((j.max(1) - 1) as f64 / N as f64, (j + 1).min(N) as f64 / N as f64);
        const G: f64 = 0.618_033_988_749_894_9;
        let (mut x1, mut x2) = (b - G * (b - a), a + G * (b - a));
        let (mut f1, mut f2) = (outside(&pc.c, p, x1), outside(&pc.c, p, x2));
        for _ in 0..48 {
            if f1 < f2 { b = x2; x2 = x1; f2 = f1; x1 = b - G * (b - a); f1 = outside(&pc.c, p, x1); }
            else { a = x1; x1 = x2; f1 = f2; x2 = a + G * (b - a); f2 = outside(&pc.c, p, x2); }
        }
        best = best.min(f1.min(f2)).min(f[j]);
    }
    best
}

/// A lower bound on a span's distance: the capsule about its chord.
fn piece_lower(pc: &Piece, p: [f64; 3]) -> f64 {
    if pc.kind != 110 && pc.kind != 111 { return -1e9; }
    let (a, d) = (pc.c[0], pc.c[3]);
    let ch = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    let pa = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let t = ((pa[0] * ch[0] + pa[1] * ch[1] + pa[2] * ch[2]) / (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2]).max(1e-20)).clamp(0.0, 1.0);
    ((pa[0] - t * ch[0]).powi(2) + (pa[1] - t * ch[1]).powi(2) + (pa[2] - t * ch[2]).powi(2)).sqrt() - pc.bound
}

/// The union of every piece, exactly (tubes and balls; a leaf is not measured here).
fn union(ps: &[Piece], p: [f64; 3]) -> f64 {
    let mut best = f64::MAX;
    for pc in ps {
        if pc.kind != 110 && pc.kind != 111 && pc.kind != 101 { continue; }
        if piece_lower(pc, p) >= best { continue; }
        best = best.min(piece_exact(pc, p));
    }
    best
}

/// The melt as the formula defines it, by brute force: every span's line integral of
/// exp(-x / melt) fade / I(r), 48 panels of eight-point Gauss-Legendre each, and every ball's own term.
fn melted(ps: &[Piece], tbl: &Table, p: [f64; 3]) -> f64 {
    let (melt, clear) = (f64::from(tbl[0][3]), f64::from(tbl[2][2]));
    let k = 1.0 / melt;
    const GL: [[f64; 2]; 4] = [[0.1834346425, 0.3626837834], [0.5255324099, 0.3137066459], [0.7966664774, 0.2223810345], [0.9602898565, 0.1012285363]];
    let mut sum = 0.0;
    for pc in ps {
        if pc.kind == 101 {
            let x = piece_exact(pc, p);
            sum += (-k * x).exp() * cpu::fade(x, clear);
            continue;
        }
        if pc.kind != 110 && pc.kind != 111 { continue; }
        if piece_lower(pc, p) >= clear / 0.65 { continue; }
        let w0 = 0.5 * k / cpu::strand(k * pc.c[0][3].max(0.0));
        let w1 = 0.5 * k / cpu::strand(k * pc.c[3][3].max(0.0));
        const PANELS: usize = 48;
        for panel in 0..PANELS {
            for j in 0..8 {
                let t = (panel as f64 + 0.5 + 0.5 * if j & 1 == 1 { GL[j >> 1][0] } else { -GL[j >> 1][0] }) / PANELS as f64;
                let x = outside(&pc.c, p, t);
                let v = bezier_rate(&pc.c, t);
                sum += GL[j >> 1][1] * 0.5 / PANELS as f64 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() * (w0 + (w1 - w0) * t) * (-k * x).exp() * cpu::fade(x, clear);
            }
        }
    }
    -sum.ln() / k
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn sym(&mut self) -> f64 { 2.0 * self.next() - 1.0 }
    fn dir(&mut self) -> [f64; 3] {
        loop {
            let v = [self.sym(), self.sym(), self.sym()];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if l > 0.05 && l <= 1.0 { return [v[0] / l, v[1] / l, v[2] / l]; }
        }
    }
}

/// A point `off` outside the surface of a random span, at a random place along and around it.
fn beside(ps: &[Piece], r: &mut Rng, off: f64) -> [f64; 3] {
    loop {
        let pc = &ps[(r.next() * ps.len() as f64) as usize % ps.len()];
        if pc.kind != 110 && pc.kind != 111 { continue; }
        let t = r.next();
        let (q, v) = (bezier(&pc.c, t), bezier_rate(&pc.c, t));
        let vl = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let d = r.dir();
        let along = (d[0] * v[0] + d[1] * v[1] + d[2] * v[2]) / vl;
        let n: [f64; 3] = std::array::from_fn(|k| d[k] - along * v[k] / vl);
        let nl = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if nl < 0.1 { continue; }
        let rr = q[3].max(0.0) + off;
        return std::array::from_fn(|k| q[k] + rr * n[k] / nl);
    }
}

// ---- what is measured -----------------------------------------------------------------------------

#[derive(Debug, Default)]
struct Measured {
    /// the largest difference between two cells' readings of one point on the face they share
    face: f64,
    /// the largest step over a stride of 1e-5, in strides (1: a distance's own slope)
    step: f64,
    /// the largest change of slope over one stride of 1e-4, on the surface's own band
    crease: f64,
    /// the formula against the reference melt (or, with no melt, the exact union) near the surface
    near: f64,
    /// far from the shape: how far the formula is over the exact distance to the union, at most
    far_over: f64,
    /// and how far under it, at most, as a share of that distance
    far_under: f64,
    /// the slope the formula gives against the one its values have
    slope: f64,
    /// what a cell holds for its middle, over the exact distance there, at most
    middle_over: f64,
    /// how far inside the shape a step of the field's own length can end (down its slope, or any way)
    passes: f64,
    /// how far outside the union's surface the melt's lies, in melts, at most
    out_by: f64,
}

fn measure(tbl: &Table, seed: u64, samples: usize) -> Measured {
    let ps = pieces(tbl);
    let n = tbl[2][0] as usize;
    let cell = f64::from(tbl[1][3]);
    let lo: [f64; 3] = std::array::from_fn(|k| f64::from(tbl[1][k]));
    let (melt, reach, clear) = (f64::from(tbl[0][3]), f64::from(tbl[2][1]), f64::from(tbl[2][2]));
    let near_clear = 0.65 * reach;
    let radius = ps.iter().filter(|p| p.kind == 110 || p.kind == 111).map(|p| p.c[0][3]).fold(0.0, f64::max);
    let leaves = ps.iter().any(|p| p.kind == 108);
    let ground = tbl[0][2] > 0.5;
    let mut r = Rng(seed);
    let mut m = Measured::default();
    let at = |p: [f64; 3]| cpu::eval(tbl, p)[3];

    // 1. one point on a face, read through the cell either side of it
    let mut tested = 0;
    for _ in 0..200 * samples {
        if tested >= samples { break; }
        let off = 0.4 * radius * r.sym();
        let p0 = beside(&ps, &mut r, off);
        let axis = (r.next() * 3.0) as usize % 3;
        let c = (p0[axis] - lo[axis]) / cell;
        let face = c.round();
        if (c - face).abs() * cell > 0.5 * radius || face < 1.0 || face > n as f64 - 1.0 { continue; }
        let mut p = p0;
        p[axis] = lo[axis] + face * cell;
        let mut ci: [usize; 3] = std::array::from_fn(|k| (((p[k] - lo[k]) / cell).floor().max(0.0) as usize).min(n - 1));
        ci[axis] = face as usize - 1;
        let a = cpu::eval_parts(tbl, p, Some(ci)).0[3];
        ci[axis] = face as usize;
        let b = cpu::eval_parts(tbl, p, Some(ci)).0[3];
        m.face = m.face.max((a - b).abs());
        tested += 1;
    }
    assert!(tested >= samples / 2, "too few faces met: {tested}");

    // 2. along short lines through the surface: no step, and near the surface no crease
    // (the surface's own band: what the smooth reading of the volume takes in, under where the field levels off)
    let band = (0.9 * (clear - HOLD * melt)).min(0.012);
    for _ in 0..samples / 8 {
        let off = 0.4 * radius * r.sym();
        let p0 = beside(&ps, &mut r, off);
        let d = r.dir();
        let h = 1e-5;
        let v: Vec<f64> = (0..400).map(|s| at(std::array::from_fn(|k| p0[k] + (s as f64 - 200.0) * h * d[k]))).collect();
        for w in v.windows(2) { m.step = m.step.max((w[1] - w[0]).abs() / h); }
        let h = 1e-4;
        let v: Vec<f64> = (0..120).map(|s| at(std::array::from_fn(|k| p0[k] + (s as f64 - 60.0) * h * d[k]))).collect();
        for w in v.windows(3) {
            if w[1].abs() < band && melt > 0.0 { m.crease = m.crease.max((w[2] - 2.0 * w[1] + w[0]).abs() / h); }
        }
    }

    // 3. near the surface: the reference; and the formula's slope against its values'
    for _ in 0..samples / 4 {
        let off = 0.35 * radius.min(near_clear) * r.sym();
        let p = beside(&ps, &mut r, off);
        let e = cpu::eval(tbl, p);
        if e[3] > 0.6 * near_clear || (ground && p[1] < f64::from(tbl[0][1]) + 0.05) { continue; }
        if !leaves {
            // (with a melt: where the melt alone is the field)
            // (nothing is kept below the ground: the field there is the way up to it)
            let cut = if ground { f64::from(tbl[0][1]) - p[1] } else { f64::MIN };
            if melt <= 0.0 { m.near = m.near.max((e[3] - union(&ps, p).max(cut)).abs()); }
            else if e[3] < 0.99 * (clear - HOLD * melt) { m.near = m.near.max((e[3] - melted(&ps, tbl, p).max(cut)).abs()); }
        }
        if e[3].abs() > 0.3 * if melt > 0.0 { clear - HOLD * melt } else { near_clear } { continue; }
        let g: [f64; 3] = std::array::from_fn(|k| { let (mut a, mut b) = (p, p); a[k] += 2e-6; b[k] -= 2e-6; (at(a) - at(b)) / 4e-6 });
        // (as ways: where spans are fading the field rises a little faster than the slope given)
        let (gl, el) = ((g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt(), (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt());
        m.slope = m.slope.max((0..3).map(|k| (g[k] / gl - e[k] / el).powi(2)).sum::<f64>().sqrt());
    }

    // 4. far from the shape, anywhere in the grid: a distance, not a step
    if !leaves {
        for _ in 0..samples / 4 {
            let p: [f64; 3] = std::array::from_fn(|k| lo[k] + r.next() * n as f64 * cell);
            if ground && p[1] < f64::from(tbl[0][1]) + 0.05 { continue; }
            let e = at(p);
            if e <= (if melt > 0.0 { clear - HOLD * melt } else { near_clear }) * 1.0001 { continue; }
            let truth = union(&ps, p);
            m.far_over = m.far_over.max(e - truth);
            if truth > 2.5 * cell { m.far_under = m.far_under.max((truth - e) / truth); }
        }
    }

    // 5. what the cells hold for their middles: never over the distance there (a leaf is not measured: the
    // union without it is no nearer)
    for _ in 0..samples {
        let c = (r.next() * (n * n * n) as f64) as usize % (n * n * n);
        let p = [lo[0] + ((c % n) as f64 + 0.5) * cell, lo[1] + ((c / n % n) as f64 + 0.5) * cell, lo[2] + ((c / (n * n)) as f64 + 0.5) * cell];
        m.middle_over = m.middle_over.max(f64::from(tbl[3 + c][2]) - union(&ps, p));
    }

    // 6. a step of the field's own length never passes the surface: down the slope, and any way
    for s in 0..samples {
        let p: [f64; 3] = if s % 2 == 0 { std::array::from_fn(|k| lo[k] + r.next() * n as f64 * cell) } else { let off = 3.0 * reach * r.next(); beside(&ps, &mut r, off) };
        if ground && p[1] < f64::from(tbl[0][1]) + 0.05 { continue; }
        let e = cpu::eval(tbl, p);
        let l = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
        if e[3] <= 0.0 || l < 1e-9 { continue; }
        m.passes = m.passes.max(-at(std::array::from_fn(|k| p[k] - e[3] * e[k] / l)));
        for _ in 0..3 {
            let d = r.dir();
            m.passes = m.passes.max(-at(std::array::from_fn(|k| p[k] + e[3] * d[k])));
        }
    }

    // 7. how far outside the union's surface the melt's lies (the far field is held that much short)
    if melt > 0.0 && !leaves {
        for _ in 0..samples / 16 {
            let p0 = beside(&ps, &mut r, 0.0);
            let e = cpu::eval(tbl, p0);
            let l = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
            if l < 1e-9 { continue; }
            // from a point of the union's surface out along the slope, to where the field is nothing
            let out = |t: f64| -> [f64; 3] { std::array::from_fn(|k| p0[k] + t * e[k] / l) };
            let (mut a, mut b) = (0.0, clear);
            if e[3] > 0.0 || at(out(b)) < 0.0 { continue; }
            for _ in 0..30 {
                let mid = 0.5 * (a + b);
                if at(out(mid)) < 0.0 { a = mid; } else { b = mid; }
            }
            if ground && out(b)[1] < f64::from(tbl[0][1]) + 0.01 { continue; }
            m.out_by = m.out_by.max(union(&ps, out(b)) / melt);
        }
    }
    m
}

const CASES: [(&str, &[&str], bool); 9] = [
    ("thomas", &["scene=attractors", "preset=thomas"], false),
    ("thomas playing", &["scene=attractors", "preset=thomas"], true),
    ("thomas, no melt", &["scene=attractors", "preset=thomas", "melt=0"], false),
    ("lorenz, thick", &["scene=attractors", "preset=lorenz", "thick=1.6"], false),
    ("trefoil", &["scene=knots", "preset=trefoil"], false),
    ("hopf link", &["scene=links", "preset=hopflink"], false),
    ("braid", &["scene=braids"], false),
    ("branching", &["scene=branching"], false),
    ("tree", &["scene=trees"], false),
];

fn case(i: usize) -> (&'static str, Table) {
    let (name, args, playing) = CASES[i];
    (name, if playing { table(args, Detail::Anim, 256) } else { table(args, Detail::Full, 768) })
}

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// What every case measures (run with --ignored --nocapture to read the numbers).
#[test]
#[ignore]
fn report() {
    for i in 0..CASES.len() {
        let (name, tbl) = case(i);
        let t = std::time::Instant::now();
        let m = measure(&tbl, SEED + i as u64, 4000);
        eprintln!("{name:16} melt {:.4} reach {:.4} clear {:.4} cell {:.4} | {m:.5?} | {:.1} s", tbl[0][3], tbl[2][1], tbl[2][2], tbl[1][3], t.elapsed().as_secs_f64());
    }
}

/// The limits every case is held to. Measured 2026-11-21 (`report`): a face 0 everywhere; a step at most
/// 1.03; a crease at most 0.024 (the melt's own fillets; a real crease is 0.1 to 1.9: the union with no
/// melt measures 1.3); the melt within 3.6e-5 of the reference and the union within 3e-11. A far middle
/// takes the nearest of the pieces the cells about it found, not of all (searching all costs 4 ms and
/// shows in no picture): a middle is over by 0.55 of a cell at most (the tree), the far field by 0.2 of a
/// cell, and a step of the field's length ends at most 0.13 of a cell inside (Thomas with no melt; with
/// a melt the two melts the far field is held short take most of it up). The melt's surface lies at most
/// 1.9 melts outside the union's.
fn hold(i: usize) {
    let (name, tbl) = case(i);
    let cell = f64::from(tbl[1][3]);
    let melt = f64::from(tbl[0][3]);
    let m = measure(&tbl, SEED + i as u64, 1600);
    let leaves = pieces(&tbl).iter().any(|p| p.kind == 108);
    // one field, whichever cell it is read through
    assert!(m.face < 1e-9, "{name}: two cells read one point {:e} apart", m.face);
    // no step anywhere, and it changes no faster than a distance does
    assert!(m.step < 1.1, "{name}: the field changes {} times as fast as a distance", m.step);
    // no crease on the surface (a leaf's edge and the ground's cut are creases of the shape itself)
    let ground = tbl[0][2] > 0.5;
    if melt > 0.0 && !leaves && !ground { assert!(m.crease < 0.05, "{name}: the slope changes by {} over 1e-4", m.crease); }
    // near the surface: the reference itself
    assert!(m.near < if melt > 0.0 { 6e-5 } else { 1e-9 }, "{name}: {:e} from the reference near the surface", m.near);
    // far from it: a distance, over the true one by a part of a cell at most and not far under it
    assert!(m.middle_over < 0.7 * cell, "{name}: a cell's middle is held {} over its distance", m.middle_over);
    assert!(m.far_over < 0.35 * cell, "{name}: the far field is {} over the distance", m.far_over);
    assert!(m.far_under < 0.35, "{name}: the far field is {} of the distance under it", m.far_under);
    // a step of the field's own length does not pass the surface by more than a part of a cell
    assert!(m.passes < 0.2 * cell, "{name}: a step of the field's length ends {} inside", m.passes);
    // and the melt's surface is no further outside the union's than the far field is held short
    assert!(m.out_by < HOLD, "{name}: the melt's surface lies {} melts outside the union's", m.out_by);
    // and the slope it gives is its values' own way
    if melt > 0.0 && !leaves && !ground { assert!(m.slope < 0.02, "{name}: the slope given is {} off the values' own", m.slope); }
}

#[test]
fn thomas_is_one_smooth_field() { hold(0); }
#[test]
fn thomas_playing_is_one_smooth_field() { hold(1); }
#[test]
fn thomas_with_no_melt_is_the_exact_union() { hold(2); }
#[test]
fn a_thick_lorenz_is_one_smooth_field() { hold(3); }
#[test]
fn a_trefoil_is_one_smooth_field() { hold(4); }
#[test]
fn a_hopf_link_is_one_smooth_field() { hold(5); }
#[test]
fn a_braid_is_one_smooth_field() { hold(6); }
#[test]
fn a_branching_shape_is_one_smooth_field() { hold(7); }
#[test]
fn a_tree_is_one_smooth_field() { hold(8); }

/// A strand alone keeps its radius under a melt, and its field is its distance: the melt's measure of a
/// straight strand is exact (to the third of a percent of `tube_strand`: 2e-5 of a melt of 0.006).
#[test]
fn a_straight_strand_alone_keeps_its_radius() {
    use gf_core::presets::exact::{write_tubes, Written};
    let (r, melt) = (0.036f32, 0.006f64);
    let span = |x0: f32, x1: f32| {
        let (a, d) = ([x0, 0.0, 0.0, r], [x1, 0.0, 0.0, r]);
        let b = std::array::from_fn(|k| (2.0 * a[k] + d[k]) / 3.0);
        let c = std::array::from_fn(|k| (a[k] + 2.0 * d[k]) / 3.0);
        Written { lo: [f64::from(x0 - r), f64::from(-r), f64::from(-r)], hi: [f64::from(x1 + r), f64::from(r), f64::from(r)], e: vec![[110.0, 0.0, 1.0, r + 1e-4], a, b, c, d], meta: -1.0 }
    };
    // (cut at uneven places: the cuts must not show)
    let clear = gf_core::presets::tubes::clear(melt, false, 768);
    let tbl = write_tubes(vec![span(-0.6, -0.13), span(-0.13, 0.02), span(0.02, 0.031), span(0.031, 0.6)], None, melt, clear, 6.0);
    let join = clear - HOLD * melt;
    for i in 0..400 {
        let x = -0.2 + 0.4 * f64::from(i) / 400.0;
        for off in [-0.03, -0.01, 0.0, 0.004, 0.012] {
            let e = cpu::eval(&tbl, [x, f64::from(r) + off, 0.0]);
            // (off the surface the field is short of the distance by half a melt's ln of the radii's ratio)
            let want = off - 0.5 * melt * (1.0 + off / f64::from(r)).ln();
            if want < 0.9 * join { assert!((e[3] - want).abs() < 2e-4 * (1.0 + 20.0 * off.abs() / f64::from(r)), "x {x} off {off}: {} for {want}", e[3]); }
            if off == 0.0 {
                assert!(e[3].abs() < 4e-5, "the surface is {} off at x {x}", e[3]);
                // (straight out; a little under 1 long, the field being short of the distance off the surface)
                assert!(e[1] > 0.9 && e[1] < 1.0 && e[0].abs() < 3e-3 && e[2].abs() < 1e-9, "the slope at x {x}: {e:?}");
            }
        }
    }
}

/// How long the table takes to make (run with --release --ignored --nocapture: it is made afresh for
/// every frame of a playing animation).
#[test]
#[ignore]
fn table_build_time() {
    for (name, args, detail, res) in [
        ("thomas playing", &["scene=attractors", "preset=thomas"][..], Detail::Anim, 256u32),
        ("thomas settled", &["scene=attractors", "preset=thomas"][..], Detail::Full, 768),
        ("tree playing", &["scene=trees"][..], Detail::Anim, 256),
        ("trefoil playing", &["scene=knots", "preset=trefoil"][..], Detail::Anim, 256),
    ] {
        let pairs: Vec<(String, String)> = args.iter().map(|s| { let (k, v) = s.split_once('=').unwrap(); (k.to_string(), v.to_string()) }).collect();
        let mut state = State::with_args(&pairs, true);
        let mut times = Vec::new();
        let mut entries = 0;
        for _ in 0..80 {
            let mut job = state.shape_job(detail, Some(res)).unwrap();
            let t = std::time::Instant::now();
            job.prepare_formula(&state.families).unwrap();
            times.push(t.elapsed().as_secs_f64() * 1e3);
            let ShapeJob::Field { table, .. } = job.shape else { panic!() };
            entries = table.len() / 4;
        }
        times.sort_by(|a, b| a.total_cmp(b));
        eprintln!("{name:16} {entries:7} entries: median {:.2} ms, p80 {:.2} ms, least {:.2} ms", times[40], times[64], times[0]);
    }
}
