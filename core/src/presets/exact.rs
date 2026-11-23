//! Shapes made of many simple solids, traced as an exact formula: the shared engine of the generated
//! families (crystals, foam, letters, tubes of every kind).
//!
//! A shape is a list of **pieces**. Each piece is the intersection of a few **terms** (planes, balls,
//! walls, hexagonal columns, outlines, and tubes: segments whose radius runs from one end to the other),
//! and the shape is their union: hard, or (`melt` > 0) a smooth union in which pieces closer than about
//! `melt` flow together. Every term is a true distance, so each piece is exact inside and never more than
//! the distance outside, and so is the union.
//!
//! **Cell lists.** A point must not be measured against every piece. The volume is cut into a grid of cells
//! and each cell lists the pieces whose boxes come within `margin` of it; a point is measured only against
//! its cell's list, and the answer is held to `margin` (every piece left out of the list is at least that
//! far). A cell with no pieces stores how near the nearest box comes to it, a bound the march can step by.
//!
//! **The table** (`Preset::table`, read as `tbl[i]`):
//!   0           (entries, ground height, 1 if there is a ground, melt)
//!   1           (the grid's low corner xyz, its cell size)
//!   2           (cells a side, margin, the first list's entry, the first piece's entry)
//!   3 ..        one entry a cell: (where its list starts, how many, how near anything comes, 0)
//!   lists       the pieces' places, four to an entry
//!   pieces      (middle of its box, entries that follow), (half-size of its box, 0), then its terms:
//!     a plane: one entry (n, d) (a unit normal: its first number is never above 1);
//!     any other term: a tag (100 + kind, its numbers) and its own entries:
//!       101 ball (c, r)    102 hollow (c, r)    103 shell, tag.y = t (c, r)    104 slab, tag.y = t (n, d)
//!       105 hex column, tag.yz = r, h (c), (axis), (u)
//!       106 tube (a, ra), (b, rb): the round cone joining two balls
//!       107 outline, tag.y = edges, tag.z = half-depth, tag.w = round (o, scale), (ux), (uy), (band), edges
//! The normal comes from the piece that decides the surface (four readings of that piece alone), or with a
//! melt from four readings of the whole: `rawGrad`.
//!
//! **Swept tubes** (generated curves: `tubes.rs`, read by `tubes.wgsl`) have a table of their own make
//! (`write_tubes`), laid out as above but for entry 2, (cells a side, reach, the melt's `clear`, the first
//! piece's entry), and for each cell's third number, which is the distance at the cell's middle. Their
//! pieces are 110 (a cubic span), 111 (a line), 101 (a ball) and 108 (a leaf).

use super::*;
use crate::curve::{placement, PlaceOpts};

/// A piece as written: its box (placed), its terms' entries, and optional per-piece metadata.
#[derive(Clone)]
pub struct Written {
    pub lo: [f64; 3],
    pub hi: [f64; 3],
    pub e: Vec<[f32; 4]>,
    pub meta: f32,
}

/// The terms of a piece, placed by `place` (a point) and `dir` (a direction) with scale `sc`.
fn terms(pc: &Piece, place: &dyn Fn(&[f64; 3]) -> [f64; 3], sc: f64) -> Vec<[f32; 4]> {
    let f = |x: f64| x as f32;
    // a plane n.p = d placed: the same normal, d moved with the origin (translation and scale only)
    let o = place(&[0.0; 3]);
    let plane = |n: &[f64; 3], d: f64| [f(n[0]), f(n[1]), f(n[2]), f(sc * d + n[0] * o[0] + n[1] * o[1] + n[2] * o[2])];
    let mut e = Vec::new();
    for term in &pc.terms {
        match term {
            Term::Plane { n, d } => e.push(plane(n, *d)),
            Term::Ball { c, r } | Term::Hollow { c, r } => {
                let q = place(c);
                e.push([if matches!(term, Term::Ball { .. }) { 101.0 } else { 102.0 }, 0.0, 0.0, 0.0]);
                e.push([f(q[0]), f(q[1]), f(q[2]), f(r * sc)]);
            }
            Term::Shell { c, r, t } => {
                let q = place(c);
                e.push([103.0, f(t * sc), 0.0, 0.0]);
                e.push([f(q[0]), f(q[1]), f(q[2]), f(r * sc)]);
            }
            Term::Slab { n, d, t } => {
                e.push([104.0, f(t * sc), 0.0, 0.0]);
                e.push(plane(n, *d));
            }
            Term::Hex { c, axis, u, r, h } => {
                let q = place(c);
                e.push([105.0, f(r * sc), f(h * sc), 0.0]);
                e.push([f(q[0]), f(q[1]), f(q[2]), 0.0]);
                e.push([f(axis[0]), f(axis[1]), f(axis[2]), 0.0]);
                e.push([f(u[0]), f(u[1]), f(u[2]), 0.0]);
            }
            Term::Glyph { o: go, ux, uy, depth, round, band, edges } => {
                let q = place(go);
                let s = (ux[0] * ux[0] + ux[1] * ux[1] + ux[2] * ux[2]).sqrt().max(1e-300);
                e.push([107.0, edges.len() as f32, f(depth * sc), f(round * sc)]);
                e.push([f(q[0]), f(q[1]), f(q[2]), f(s * sc)]);
                e.push([f(ux[0] / s), f(ux[1] / s), f(ux[2] / s), 0.0]);
                e.push([f(uy[0] / s), f(uy[1] / s), f(uy[2] / s), 0.0]);
                e.push([f(band[0]), f(band[1]), f(band[2]), 0.0]);
                for g in edges { e.push([f(g[0]), f(g[1]), f(g[2]), f(g[3])]); }
            }
            // (sweeps and paths are not crystal pieces: paths have their own exact formula)
            Term::Sweep { .. } | Term::Path { .. } => {}
        }
    }
    e
}

/// A tube from a to b, its radius ra at a and rb at b (already placed): a piece of one term.
pub fn tube(a: [f64; 3], ra: f64, b: [f64; 3], rb: f64) -> Written {
    let f = |x: f64| x as f32;
    let r = ra.max(rb);
    Written {
        lo: std::array::from_fn(|k| a[k].min(b[k]) - r),
        hi: std::array::from_fn(|k| a[k].max(b[k]) + r),
        e: vec![[106.0, 0.0, 0.0, 0.0], [f(a[0]), f(a[1]), f(a[2]), f(ra)], [f(b[0]), f(b[1]), f(b[2]), f(rb)]],
        meta: 0.0,
    }
}

/// The box round a piece, worked out from its terms (a piece's own box need not hold all of it: the grid
/// of the curves' baker reached a band beyond every box). Planes alone: the corners of the solid they bound
/// (every three planes meeting inside the others); with a ball or a shell, the ball's box (cut by the
/// planes' box where there is one); anything else, the piece's own box.
pub fn bounds(pc: &Piece) -> ([f64; 3], [f64; 3]) {
    let planes: Vec<([f64; 3], f64)> = pc.terms.iter().filter_map(|t| match t { Term::Plane { n, d } => Some((*n, *d)), _ => None }).collect();
    let mut ball: Option<([f64; 3], f64)> = None;
    let mut other = false;
    for t in &pc.terms {
        match t {
            Term::Ball { c, r } => ball = Some((*c, *r)),
            Term::Shell { c, r, t } => ball = Some((*c, r + t)),
            Term::Plane { .. } | Term::Hollow { .. } | Term::Slab { .. } => {}
            _ => other = true,
        }
    }
    if other { return (pc.lo, pc.hi); }
    // the planes' corners
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    let n = planes.len();
    let det = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                let (a, b, c) = (planes[i], planes[j], planes[k]);
                let dt = det(a.0, b.0, c.0);
                if dt.abs() < 1e-12 { continue; }
                // Cramer: the point on all three planes
                let x = [
                    det([a.1, a.0[1], a.0[2]], [b.1, b.0[1], b.0[2]], [c.1, c.0[1], c.0[2]]) / dt,
                    det([a.0[0], a.1, a.0[2]], [b.0[0], b.1, b.0[2]], [c.0[0], c.1, c.0[2]]) / dt,
                    det([a.0[0], a.0[1], a.1], [b.0[0], b.0[1], b.1], [c.0[0], c.0[1], c.1]) / dt,
                ];
                if planes.iter().all(|(m, d)| m[0] * x[0] + m[1] * x[1] + m[2] * x[2] <= d + 1e-7) {
                    for q in 0..3 {
                        lo[q] = lo[q].min(x[q]);
                        hi[q] = hi[q].max(x[q]);
                    }
                }
            }
        }
    }
    // (the corners bound the solid only if it is closed all round: no direction leaves through every plane)
    let mut r = geo::Rng::new(1.0);
    let closed = lo[0] <= hi[0] && (0..400).all(|_| { let u = r.dir(); planes.iter().any(|(m, _)| m[0] * u[0] + m[1] * u[1] + m[2] * u[2] > 1e-6) });
    match (ball, closed) {
        (Some((c, r)), true) => (std::array::from_fn(|q| lo[q].max(c[q] - r)), std::array::from_fn(|q| hi[q].min(c[q] + r))),
        (Some((c, r)), false) => (std::array::from_fn(|q| c[q] - r), std::array::from_fn(|q| c[q] + r)),
        (None, true) => (lo, hi),
        // (planes that bound nothing all round: the piece's own box)
        (None, false) => (pc.lo, pc.hi),
    }
}

/// A piece already in the volume's units, as written.
pub fn written(pc: &Piece) -> Written {
    let same = |p: &[f64; 3]| *p;
    let (lo, hi) = bounds(pc);
    Written { lo, hi, e: terms(pc, &same, 1.0), meta: 0.0 }
}

/// How a shape of pieces is placed in the volume: centred on its frame, its farthest point `fit` away.
pub fn place_of(frame: &[f64], fit: f64) -> ([f64; 3], f64) {
    let pose = placement(&[frame], &PlaceOpts { centre: true, fit: Some(fit), ..Default::default() });
    (pose.c, pose.sc)
}

/// The table for a shape made of pieces (`RawCurve::pieces`), placed by its frame so that its farthest
/// point is `fit` from the middle.
pub fn table(raw: &RawCurve, fit: f64) -> Vec<[f32; 4]> {
    let frame = if raw.frame.is_empty() { let mut r = raw.clone(); r.frame_pieces(); r.frame } else { raw.frame.clone() };
    let (c, sc) = place_of(&frame, fit);
    write(placed_pieces(raw, c, sc), raw.ground.map(|g| (g - c[1]) * sc), 0.0)
}

pub(super) fn placed_pieces(raw: &RawCurve, c: [f64; 3], sc: f64) -> Vec<Written> {
    let place = move |p: &[f64; 3]| [(p[0] - c[0]) * sc, (p[1] - c[1]) * sc, (p[2] - c[2]) * sc];
    let mut out = Vec::with_capacity(raw.pieces.len());
    for pc in &raw.pieces {
        let e = terms(pc, &place, sc);
        if e.is_empty() { continue; }
        let (lo, hi) = bounds(pc);
        let (a, b) = (place(&lo), place(&hi));
        out.push(Written { lo: std::array::from_fn(|k| a[k].min(b[k])), hi: std::array::from_fn(|k| a[k].max(b[k])), e, meta: 0.0 });
    }
    out
}

/// A lower bound on a piece's distance at c (exact for planes, balls, walls; a tube as the capsule of its
/// larger radius; a hexagonal column or an outline as its box), changing no faster than the distance.
fn lower(p: &Written, c: [f64; 3]) -> f64 {
    let boxd = (0..3).map(|k| (p.lo[k] - c[k]).max(c[k] - p.hi[k]).max(0.0).powi(2)).sum::<f64>().sqrt();
    let v = |a: [f32; 4]| [a[0] as f64, a[1] as f64, a[2] as f64, a[3] as f64];
    let mut e = boxd;
    let mut j = 0;
    while j < p.e.len() {
        let a = v(p.e[j]);
        if a[0] < 50.0 { e = e.max(a[0] * c[0] + a[1] * c[1] + a[2] * c[2] - a[3]); j += 1; continue; }
        let q = v(p.e[j + 1]);
        let r = ((c[0] - q[0]).powi(2) + (c[1] - q[1]).powi(2) + (c[2] - q[2]).powi(2)).sqrt();
        match a[0].round() as i32 {
            101 => { e = e.max(r - q[3]); j += 2; }
            102 => { e = e.max(q[3] - r); j += 2; }
            103 => { e = e.max((r - q[3]).abs() - a[1]); j += 2; }
            104 => { e = e.max((q[0] * c[0] + q[1] * c[1] + q[2] * c[2] - q[3]).abs() - a[1]); j += 2; }
            105 | 108 | 109 => j += 4,
            106 | 110 | 111 => {
                let cubic = a[0] == 110.0;
                let b = v(p.e[j + if cubic { 4 } else { 2 }]);
                let ba = [b[0] - q[0], b[1] - q[1], b[2] - q[2]];
                let pa = [c[0] - q[0], c[1] - q[1], c[2] - q[2]];
                let h = ((pa[0] * ba[0] + pa[1] * ba[1] + pa[2] * ba[2]) / (ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2]).max(1e-18)).clamp(0.0, 1.0);
                let w = [pa[0] - ba[0] * h, pa[1] - ba[1] * h, pa[2] - ba[2] * h];
                let radius = if a[0] == 106.0 { q[3].max(b[3]) } else { a[3] };
                e = e.max((w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt() - radius);
                j += if cubic { 5 } else { 3 };
            }
            _ => j += 5 + a[1].round() as usize,
        }
    }
    e
}

/// The table for these pieces (placed): the grid of cells and their lists, then the pieces. `ground`: a
/// height below which nothing is kept; `melt`: how far pieces flow together (0: a hard union).
pub fn write(pieces: Vec<Written>, ground: Option<f64>, melt: f64) -> Vec<[f32; 4]> {
    write_cells(pieces, ground, melt, 6.0)
}

pub(super) fn write_cells(pieces: Vec<Written>, ground: Option<f64>, melt: f64, density: f64) -> Vec<[f32; 4]> {
    let mut n = ((pieces.len() as f64).cbrt() * density).round().clamp(12.0, 64.0) as usize;
    loop {
        let table = table_at_resolution(&pieces, ground, melt, n);
        if table.len() <= crate::field::TABLE_MAX || n == 1 { return table; }
        n = ((n * 3) / 4).min(n - 1).max(1);
    }
}

fn table_at_resolution(pieces: &[Written], ground: Option<f64>, melt: f64, n: usize) -> Vec<[f32; 4]> {
    let f = |x: f64| x as f32;
    // the grid round everything, a little larger
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in pieces {
        for k in 0..3 {
            lo[k] = lo[k].min(p.lo[k]);
            hi[k] = hi[k].max(p.hi[k]);
        }
    }
    if pieces.is_empty() { (lo, hi) = ([-0.1; 3], [0.1; 3]); }
    let span = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max).max(1e-3);
    let cell = span * 1.02 / n as f64;
    let margin = 0.5 * cell + 3.5 * melt;
    let lo = std::array::from_fn::<f64, 3, _>(|k| lo[k] - 0.01 * span);
    let at = |x: f64, k: usize| (((x - lo[k]) / cell).floor().max(0.0) as usize).min(n - 1);
    // A piece is listed in a cell when it may come within the margin of it: its box does, and its own
    // reading at the cell's middle, less half the cell's diagonal, does (each piece's reading is a lower
    // bound that changes no faster than the distance: a ball's hollow inside lists none of its cells).
    let half_diag = 0.5 * cell * 3f64.sqrt();
    // (each piece's cells worked out side by side on every core, then gathered in order)
    use rayon::prelude::*;
    let per: Vec<Vec<u32>> = pieces
        .par_iter()
        .map(|p| {
            let (a, b): ([usize; 3], [usize; 3]) = (std::array::from_fn(|k| at(p.lo[k] - margin, k)), std::array::from_fn(|k| at(p.hi[k] + margin, k)));
            let mut mine = Vec::new();
            for z in a[2]..=b[2] {
                for y in a[1]..=b[1] {
                    for x in a[0]..=b[0] {
                        let c = [lo[0] + (x as f64 + 0.5) * cell, lo[1] + (y as f64 + 0.5) * cell, lo[2] + (z as f64 + 0.5) * cell];
                        if lower(p, c) - half_diag <= margin { mine.push(((z * n + y) * n + x) as u32); }
                    }
                }
            }
            mine
        })
        .collect();
    let cells = n * n * n;
    let mut counts = vec![0u32; cells];
    let mut list_len = 0;
    for indices in &per {
        list_len += indices.len();
        for &i in indices { counts[i as usize] += 1; }
    }
    // Manhattan distance to the closest occupied cell in two linear sweeps. Euclidean
    // distance is at least Manhattan/sqrt(3); any points in the two cells can close
    // the gap by at most three grid coordinates in total.
    let mut steps: Vec<u16> = counts.iter().map(|&count| if count == 0 { 3 * n as u16 } else { 0 }).collect();
    let plane = n * n;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let i = z * plane + y * n + x;
                if x > 0 { steps[i] = steps[i].min(steps[i - 1] + 1); }
                if y > 0 { steps[i] = steps[i].min(steps[i - n] + 1); }
                if z > 0 { steps[i] = steps[i].min(steps[i - plane] + 1); }
            }
        }
    }
    for z in (0..n).rev() {
        for y in (0..n).rev() {
            for x in (0..n).rev() {
                let i = z * plane + y * n + x;
                if x + 1 < n { steps[i] = steps[i].min(steps[i + 1] + 1); }
                if y + 1 < n { steps[i] = steps[i].min(steps[i + n] + 1); }
                if z + 1 < n { steps[i] = steps[i].min(steps[i + plane] + 1); }
            }
        }
    }
    let first_list = 3 + cells;
    let first_piece = first_list + list_len.div_ceil(4);
    let mut t = Vec::with_capacity(first_piece + pieces.iter().map(|p| 2 + p.e.len()).sum::<usize>());
    t.push([0.0, f(ground.unwrap_or(0.0)), if ground.is_some() { 1.0 } else { 0.0 }, f(melt)]);
    t.push([f(lo[0]), f(lo[1]), f(lo[2]), f(cell)]);
    t.push([n as f32, f(margin), first_list as f32, first_piece as f32]);
    let mut start = 0;
    for (i, count) in counts.iter_mut().enumerate() {
        let near = margin + f64::from(steps[i].saturating_sub(3)) * cell / 3f64.sqrt();
        t.push([start as f32, *count as f32, f(near), 0.0]);
        let end = start + *count;
        *count = start;
        start = end;
    }
    t.resize(first_piece, [0.0; 4]);
    let mut offset = first_piece;
    for (piece, indices) in pieces.iter().zip(&per) {
        for &cell in indices {
            let at = &mut counts[cell as usize];
            t[first_list + *at as usize / 4][*at as usize % 4] = offset as f32;
            *at += 1;
        }
        offset += 2 + piece.e.len();
    }
    for p in pieces {
        t.push([f(0.5 * (p.lo[0] + p.hi[0])), f(0.5 * (p.lo[1] + p.hi[1])), f(0.5 * (p.lo[2] + p.hi[2])), p.e.len() as f32]);
        t.push([f(0.5 * (p.hi[0] - p.lo[0])), f(0.5 * (p.hi[1] - p.lo[1])), f(0.5 * (p.hi[2] - p.lo[2])), p.meta]);
        t.extend(p.e.iter().copied());
    }
    t[0][0] = t.len() as f32;
    t
}

/// The table for swept tubes (`tubes.wgsl`), as `write` lays one out, with two differences. Every cell
/// lists every piece that comes within `reach` of it: the melt's own reach (`clear / 0.65`: where a
/// span's part has faded to nothing) or 0.92 of a cell, whichever is more; nearer than 0.6 of a cell a
/// thin strand can pass between the cells' middles unseen by the far field. And every cell holds the
/// distance at its middle, which the formula joins into a far field that is a distance, not just a step
/// that is safe to take. Within reach of a piece what a cell holds is never more than the distance. A far
/// middle takes the nearest of the pieces that the cells about it found: all the pieces are not searched
/// (4 ms for Thomas, and the pictures are the same without it), so now and then a nearer piece is missed
/// and the middle is over by a part of a cell (core/tests/tube_field.rs measures how much). `clear`: how
/// far outside a span's own surface its part of the melt is whole (0 with no melt).
pub fn write_tubes(pieces: Vec<Written>, ground: Option<f64>, melt: f64, clear: f64, density: f64) -> Vec<[f32; 4]> {
    let mut n = ((pieces.len() as f64).cbrt() * density).round().clamp(12.0, 64.0) as usize;
    let (lines, short): (Vec<Polyline>, Vec<f64>) = pieces.iter().map(polyline).unzip();
    // How far a piece's reading (`lower`: the capsule about its chord, as fat as its thickest and as its
    // controls stray) can be short of its distance as `far_distance` has it.
    let slack: Vec<f64> = pieces.iter().map(|p| match p.e.first().map(|t| t[0].round() as i32) {
        Some(110) | Some(111) => {
            let radii = p.e[1..].iter().map(|c| f64::from(c[3]));
            let (thick, thin) = (radii.clone().fold(f64::MIN, f64::max), radii.fold(f64::MAX, f64::min));
            2.0 * (f64::from(p.e[0][3]) - thick).max(0.0) + thick - thin.max(0.0).min(thick) + 1e-6
        }
        _ => 1e-6,
    }).collect();
    loop {
        let table = tubes_at_resolution(&pieces, &lines, &short, &slack, ground, melt, clear, n);
        if table.len() <= crate::field::TABLE_MAX || n == 1 { return table; }
        n = ((n * 3) / 4).min(n - 1).max(1);
    }
}

/// A piece as the far field measures it: a span as four chords between five points along it (a cubic
/// leaves them by a sixteenth of what it leaves its own chord by), a ball as one point, a leaf as its box.
// (nearly every piece is a span: boxing its chords would only add a hop)
#[allow(clippy::large_enum_variant)]
enum Polyline {
    Chords([Chord; 4]),
    Ball([f64; 4]),
    Box,
}

/// A chord: where it starts, the way to its end, 1 / that length squared, the radius at its start and how
/// much it gains.
#[derive(Clone, Copy)]
struct Chord { a: [f64; 3], ba: [f64; 3], inv: f64, r: f64, dr: f64 }

/// With it: how far `far_distance` can be over the piece's true distance. A cubic leaves the chord of a
/// quarter of itself by at most 6/128 of its greatest second difference (centre line and radius alike),
/// and along one chord the radius gains at most what its ends differ by (on a cone the nearest ball is not
/// the one nearest the chord's nearest point).
fn polyline(p: &Written) -> (Polyline, f64) {
    let v = |a: [f32; 4]| a.map(f64::from);
    let chords = |q: [[f64; 4]; 5]| Polyline::Chords(std::array::from_fn(|j| {
        let (a, b) = (q[j], q[j + 1]);
        let ba = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        Chord { a: [a[0], a[1], a[2]], ba, inv: 1.0 / (ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2]).max(1e-18), r: a[3], dr: b[3] - a[3] }
    }));
    let gain = |q: &[[f64; 4]; 5]| (0..4).map(|j| (q[j + 1][3] - q[j][3]).abs()).fold(0.0, f64::max);
    // (and a millionth: what is stored is rounded to a float)
    match p.e.first().map(|t| t[0].round() as i32) {
        Some(110) if p.e.len() == 5 => {
            let c = [v(p.e[1]), v(p.e[2]), v(p.e[3]), v(p.e[4])];
            let q: [[f64; 4]; 5] = std::array::from_fn(|j| {
                let t = j as f64 * 0.25;
                let u = 1.0 - t;
                let w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
                std::array::from_fn(|k| w[0] * c[0][k] + w[1] * c[1][k] + w[2] * c[2][k] + w[3] * c[3][k])
            });
            let second = |i: usize, k: usize| c[i][k] - 2.0 * c[i + 1][k] + c[i + 2][k];
            let centre = (0..2).map(|i| (0..3).map(|k| second(i, k).powi(2)).sum::<f64>().sqrt()).fold(0.0, f64::max);
            let radius = (0..2).map(|i| second(i, 3).abs()).fold(0.0, f64::max);
            let over = 6.0 / 128.0 * (centre + radius) + gain(&q) + 1e-6;
            (chords(q), over)
        }
        Some(111) if p.e.len() == 3 => {
            let (a, d) = (v(p.e[1]), v(p.e[2]));
            let q: [[f64; 4]; 5] = std::array::from_fn(|j| std::array::from_fn(|k| a[k] + (d[k] - a[k]) * j as f64 * 0.25));
            let over = gain(&q) + 1e-6;
            (chords(q), over)
        }
        Some(101) if p.e.len() == 2 => (Polyline::Ball(v(p.e[1])), 1e-6),
        _ => (Polyline::Box, 1e-6),
    }
}

/// How far `c` is from a piece's surface, near enough (a span: from its four chords, the radius run evenly
/// along each).
#[inline]
fn far_distance(p: &Written, line: &Polyline, c: [f64; 3]) -> f64 {
    match line {
        Polyline::Chords(q) => {
            let mut best = f64::MAX;
            for &Chord { a, ba, inv, r, dr } in q {
                let pa = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let h = ((pa[0] * ba[0] + pa[1] * ba[1] + pa[2] * ba[2]) * inv).clamp(0.0, 1.0);
                let o = [pa[0] - ba[0] * h, pa[1] - ba[1] * h, pa[2] - ba[2] * h];
                best = best.min((o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt() - (r + dr * h).max(0.0));
            }
            best
        }
        Polyline::Ball(q) => ((c[0] - q[0]).powi(2) + (c[1] - q[1]).powi(2) + (c[2] - q[2]).powi(2)).sqrt() - q[3],
        Polyline::Box => (0..3).map(|k| (p.lo[k] - c[k]).max(c[k] - p.hi[k]).max(0.0).powi(2)).sum::<f64>().sqrt(),
    }
}

const NONE: u32 = u32::MAX;

/// For every cell of an n x n x n grid, the nearest cell (middle to middle) that is `set`: an exact
/// Euclidean distance transform, one axis at a time (Felzenszwalb and Huttenlocher's lower envelope of
/// parabolas), carrying which cell it was. `NONE` everywhere if none is set.
fn nearest_cells(set: &[bool], n: usize, spread: bool) -> Vec<u32> {
    use rayon::prelude::*;
    const FAR: i32 = i32::MAX / 4;
    // One line: `d` the squared distances so far (FAR: none), `from` the cells they are to; both rewritten.
    fn line(d: &mut [i32], from: &mut [u32], hull: &mut Vec<(usize, f32)>, keep: &mut Vec<(i32, u32)>) {
        hull.clear();
        for q in 0..d.len() {
            if d[q] >= FAR { continue; }
            let fq = (d[q] + (q * q) as i32) as f32;
            // where the parabola from q overtakes the last of the hull
            let mut start = f32::NEG_INFINITY;
            while let Some(&(v, begins)) = hull.last() {
                let s = (fq - (d[v] + (v * v) as i32) as f32) / (2.0 * (q as f32 - v as f32));
                if s <= begins { hull.pop(); } else { start = s; break; }
            }
            hull.push((q, start));
        }
        if hull.is_empty() { return; }
        keep.clear();
        let mut k = 0;
        for q in 0..d.len() {
            while k + 1 < hull.len() && hull[k + 1].1 < q as f32 { k += 1; }
            let v = hull[k].0;
            keep.push((d[v] + ((q as i32 - v as i32) * (q as i32 - v as i32)), from[v]));
        }
        for (q, &(dq, fq)) in keep.iter().enumerate() {
            d[q] = dq;
            from[q] = fq;
        }
    }
    let cells = n * n * n;
    let mut d = vec![FAR; cells];
    let mut from = vec![NONE; cells];
    // within each plane: along x (the nearest set cell either way, by two runs), then along y
    type Scratch = (Vec<(usize, f32)>, Vec<(i32, u32)>, Vec<i32>, Vec<u32>);
    let scratch = || -> Scratch { (Vec::new(), Vec::new(), vec![0i32; n], vec![0u32; n]) };
    let plane = |(hull, keep, dl, fl): &mut Scratch, (z, (dp, fp)): (usize, (&mut [i32], &mut [u32]))| {
        for y in 0..n {
            let row = (z * n + y) * n;
            let mut last = usize::MAX;
            for x in 0..n {
                if set[row + x] { last = x; }
                if last != usize::MAX { dp[y * n + x] = ((x - last) * (x - last)) as i32; fp[y * n + x] = (row + last) as u32; }
            }
            last = usize::MAX;
            for x in (0..n).rev() {
                if set[row + x] { last = x; }
                if last != usize::MAX && (((last - x) * (last - x)) as i32) < dp[y * n + x] { dp[y * n + x] = ((last - x) * (last - x)) as i32; fp[y * n + x] = (row + last) as u32; }
            }
        }
        for x in 0..n {
            for y in 0..n { dl[y] = dp[y * n + x]; fl[y] = fp[y * n + x]; }
            line(dl, fl, hull, keep);
            for y in 0..n { dp[y * n + x] = dl[y]; fp[y * n + x] = fl[y]; }
        }
    };
    if spread {
        d.par_chunks_mut(n * n).zip(from.par_chunks_mut(n * n)).enumerate().for_each_init(scratch, plane);
    } else {
        let mut own = scratch();
        d.chunks_mut(n * n).zip(from.chunks_mut(n * n)).enumerate().for_each(|planes| plane(&mut own, planes));
    }
    // along z: a row of columns at a time, gathered, worked out, and put back
    let row = |(hull, keep, dl, fl): &mut Scratch, y: usize| -> Vec<u32> {
        let mut out = vec![NONE; n * n];
        for x in 0..n {
            for z in 0..n { dl[z] = d[(z * n + y) * n + x]; fl[z] = from[(z * n + y) * n + x]; }
            line(dl, fl, hull, keep);
            out[x * n..(x + 1) * n].copy_from_slice(fl);
        }
        out
    };
    let rows: Vec<Vec<u32>> = if spread {
        (0..n).into_par_iter().map_init(scratch, row).collect()
    } else {
        let mut own = scratch();
        (0..n).map(|y| row(&mut own, y)).collect()
    };
    for (y, out) in rows.iter().enumerate() {
        for x in 0..n {
            for z in 0..n { from[(z * n + y) * n + x] = out[x * n + z]; }
        }
    }
    from
}

#[allow(clippy::too_many_arguments)]
fn tubes_at_resolution(pieces: &[Written], lines: &[Polyline], short: &[f64], slack: &[f64], ground: Option<f64>, melt: f64, clear: f64, n: usize) -> Vec<[f32; 4]> {
    let f = |x: f64| x as f32;
    // The grid round everything, a little larger: the box of its cells' middles holds every piece (the far
    // field outside that box rests on it), which takes one more cell a side when the cells are few.
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in pieces {
        for k in 0..3 {
            lo[k] = lo[k].min(p.lo[k]);
            hi[k] = hi[k].max(p.hi[k]);
        }
    }
    if pieces.is_empty() { (lo, hi) = ([-0.1; 3], [0.1; 3]); }
    let span = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max).max(1e-3);
    let cell = span * 1.02 / n as f64;
    let n = if (n as f64 - 1.0) * cell >= span * 1.004 { n } else { n + 1 };
    let reach = (0.92 * cell).max(if melt > 0.0 { clear / 0.65 } else { 0.0 });
    let lo = std::array::from_fn::<f64, 3, _>(|k| 0.5 * (lo[k] + hi[k]) - 0.5 * n as f64 * cell);
    let at = |x: f64, k: usize| (((x - lo[k]) / cell).floor().max(0.0) as usize).min(n - 1);
    let middle = |i: usize| [lo[0] + ((i % n) as f64 + 0.5) * cell, lo[1] + ((i / n % n) as f64 + 0.5) * cell, lo[2] + ((i / (n * n)) as f64 + 0.5) * cell];
    let half_diag = 0.5 * cell * 3f64.sqrt();
    // A piece is listed in a cell when it may come within the reach of any point of it: its reading at the
    // cell's middle (`lower`: never more than its distance there), less half the cell's diagonal, does.
    // Each piece's cells and readings are worked out side by side on every core, a plane of cells after
    // another (`from`: where each plane's begin).
    let listed = reach + half_diag;
    // (a small grid is worked through on this core alone: handing so little round costs more than it saves)
    let spread = n * n * n >= 20_000;
    struct Readings { first_z: usize, from: Vec<u32>, at: Vec<(u32, f32)> }
    fn in_plane(r: &Readings, z: usize) -> &[(u32, f32)] {
        if z < r.first_z || z + 1 - r.first_z >= r.from.len() { return &[]; }
        &r.at[r.from[z - r.first_z] as usize..r.from[z + 1 - r.first_z] as usize]
    }
    use rayon::prelude::*;
    let readings = |p: &Written| -> Readings {
        let (a, b): ([usize; 3], [usize; 3]) = (std::array::from_fn(|k| at(p.lo[k] - reach, k)), std::array::from_fn(|k| at(p.hi[k] + reach, k)));
        let mut mine = Readings { first_z: a[2], from: Vec::with_capacity(b[2] - a[2] + 2), at: Vec::new() };
        // A span's reading is the more of two: how far outside its box, and how far outside the capsule
        // about its chord. Both are compared squared first, so most cells cost no square root.
        let chord = match p.e.first().map(|t| t[0].round() as i32) {
            Some(110) if p.e.len() == 5 => Some((p.e[1], p.e[4], f64::from(p.e[0][3]))),
            Some(111) if p.e.len() == 3 => Some((p.e[1], p.e[2], f64::from(p.e[0][3]))),
            _ => None,
        }.map(|(a, d, bound)| {
            let (a, d) = (a.map(f64::from), d.map(f64::from));
            let ba = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
            (a, ba, 1.0 / (ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2]).max(1e-18), bound)
        });
        for z in a[2]..=b[2] {
            mine.from.push(mine.at.len() as u32);
            for y in a[1]..=b[1] {
                for x in a[0]..=b[0] {
                    let c = [lo[0] + (x as f64 + 0.5) * cell, lo[1] + (y as f64 + 0.5) * cell, lo[2] + (z as f64 + 0.5) * cell];
                    let d = match chord {
                        Some((a, ba, inv, bound)) => {
                            let pa = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                            let h = ((pa[0] * ba[0] + pa[1] * ba[1] + pa[2] * ba[2]) * inv).clamp(0.0, 1.0);
                            let w = [pa[0] - ba[0] * h, pa[1] - ba[1] * h, pa[2] - ba[2] * h];
                            let from_chord = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
                            if from_chord > (listed + bound) * (listed + bound) { continue; }
                            let from_box = (0..3).map(|k| (p.lo[k] - c[k]).max(c[k] - p.hi[k]).max(0.0).powi(2)).sum::<f64>();
                            if from_box > listed * listed { continue; }
                            from_box.sqrt().max(from_chord.sqrt() - bound)
                        }
                        None => lower(p, c),
                    };
                    if d <= listed { mine.at.push((((z * n + y) * n + x) as u32, d as f32)); }
                }
            }
        }
        mine.from.push(mine.at.len() as u32);
        mine
    };
    let per: Vec<Readings> = if spread { pieces.par_iter().map(readings).collect() } else { pieces.iter().map(readings).collect() };
    // A plane of cells at a time, side by side: its cells' lists (each piece with its reading, the pieces in
    // order; `starts`: where each cell's begins), and the distance at each cell's middle where a list holds
    // a piece within the reach (the least over the list: every piece that near is in it; a reading is short
    // of the distance by no more than the piece's `slack`, which lets most of a list be passed over).
    struct Plane { starts: Vec<u32>, list: Vec<(u32, f32)>, seeded: Vec<(u32, f32)> }
    let side = n * n;
    let plane = |z: usize| -> Plane {
        let base = z * side;
        let mut starts = vec![0u32; side + 1];
        for r in &per {
            for &(i, _) in in_plane(r, z) { starts[i as usize - base + 1] += 1; }
        }
        for i in 0..side { starts[i + 1] += starts[i]; }
        let mut list = vec![(0u32, 0f32); starts[side] as usize];
        let mut fill = starts.clone();
        for (pi, r) in per.iter().enumerate() {
            for &(i, d) in in_plane(r, z) {
                let at = &mut fill[i as usize - base];
                list[*at as usize] = (pi as u32, d);
                *at += 1;
            }
        }
        let seeded = (0..side).map(|i| {
            let here = &list[starts[i] as usize..starts[i + 1] as usize];
            if here.is_empty() { return (NONE, f32::MAX); }
            let most = here.iter().map(|&(pi, d)| f64::from(d) + slack[pi as usize]).fold(f64::MAX, f64::min);
            let c = middle(base + i);
            let mut best = (NONE, f64::MAX);
            for &(pi, d) in here {
                // (inside a piece's box its reading is 0 at the least, whatever lies deeper)
                if f64::from(d) > most.min(best.1.max(0.0)) { continue; }
                let away = far_distance(&pieces[pi as usize], &lines[pi as usize], c) - short[pi as usize];
                if away < best.1 { best = (pi, away); }
            }
            (best.0, best.1 as f32)
        }).collect();
        Plane { starts, list, seeded }
    };
    let planes: Vec<Plane> = if spread { (0..n).into_par_iter().map(plane).collect() } else { (0..n).map(plane).collect() };
    let cells = n * side;
    let list_len: usize = planes.iter().map(|p| p.list.len()).sum();
    let seed = |i: usize| planes[i / side].seeded[i % side];
    // The distance at the other middles, worked out on the other cores while this one lays the table out.
    // First to the piece nearest the nearest cell that has one within reach, that cell found for every cell
    // at once by an exact distance transform over the cells; and to the pieces its six neighbours find that
    // way (the piece nearest that cell is not always the nearest here).
    let sure = reach + 0.5 * cell;
    let middles = || -> Vec<f32> {
        let within: Vec<bool> = planes.iter().flat_map(|p| &p.seeded).map(|&(o, d)| o != NONE && f64::from(d) <= reach).collect();
        let nearest = nearest_cells(&within, n, spread);
        // the piece a cell would take from the nearest cell that is within reach of one
        let lent = |i: usize| if nearest[i] == NONE { NONE } else { seed(nearest[i] as usize).0 };
        let far = |i: usize| -> f32 {
            let (own, known) = seed(i);
            // (every piece that near a middle is within reach of its cell, so in its list: such a least is
            // the least of all)
            if own != NONE && f64::from(known) <= sure { return known; }
            if nearest[i] == NONE { return 1e4; }
            let c = middle(i);
            let mut best = if own != NONE { f64::from(known) } else { f64::MAX };
            let (x, y, z) = (i % n, i / n % n, i / side);
            let mut tried = [own; 8];
            let mut count = 1;
            let mut take = |o: u32, best: &mut f64| {
                if o == NONE || tried[..count].contains(&o) { return; }
                tried[count] = o;
                count += 1;
                *best = best.min(far_distance(&pieces[o as usize], &lines[o as usize], c) - short[o as usize]);
            };
            for from in [
                i,
                if x > 0 { i - 1 } else { i }, if x + 1 < n { i + 1 } else { i },
                if y > 0 { i - n } else { i }, if y + 1 < n { i + n } else { i },
                if z > 0 { i - side } else { i }, if z + 1 < n { i + side } else { i },
            ] {
                take(lent(from), &mut best);
            }
            // (a middle whose list holds nothing that near is at least that far off)
            best.max(sure) as f32
        };
        if spread { (0..cells).into_par_iter().with_min_len(256).map(far).collect() } else { (0..cells).map(far).collect() }
    };
    let first_list = 3 + cells;
    let first_piece = first_list + list_len.div_ceil(4);
    let layout = || -> Vec<[f32; 4]> {
        let mut offsets = Vec::with_capacity(pieces.len());
        let mut offset = first_piece;
        for piece in pieces {
            offsets.push(offset as f32);
            offset += 2 + piece.e.len();
        }
        let mut t = Vec::with_capacity(offset);
        t.push([0.0, f(ground.unwrap_or(0.0)), if ground.is_some() { 1.0 } else { 0.0 }, f(melt)]);
        t.push([f(lo[0]), f(lo[1]), f(lo[2]), f(cell)]);
        t.push([n as f32, f(reach), f(clear), first_piece as f32]);
        // (each cell's third number, the distance at its middle, is put in when it is known)
        let mut before = 0;
        for plane in &planes {
            for i in 0..side {
                t.push([(before + plane.starts[i]) as f32, (plane.starts[i + 1] - plane.starts[i]) as f32, 0.0, 0.0]);
            }
            before += plane.starts[side];
        }
        t.resize(first_piece, [0.0; 4]);
        for (at, &(pi, _)) in planes.iter().flat_map(|p| &p.list).enumerate() {
            t[first_list + at / 4][at % 4] = offsets[pi as usize];
        }
        for p in pieces {
            t.push([f(0.5 * (p.lo[0] + p.hi[0])), f(0.5 * (p.lo[1] + p.hi[1])), f(0.5 * (p.lo[2] + p.hi[2])), p.e.len() as f32]);
            t.push([f(0.5 * (p.hi[0] - p.lo[0])), f(0.5 * (p.hi[1] - p.lo[1])), f(0.5 * (p.hi[2] - p.lo[2])), p.meta]);
            t.extend(p.e.iter().copied());
        }
        t[0][0] = t.len() as f32;
        t
    };
    let (dist, mut t) = if spread { rayon::join(middles, layout) } else { (middles(), layout()) };
    for (entry, d) in t[3..3 + cells].iter_mut().zip(dist) { entry[2] = d; }
    t
}

/// The formula: the union of the table's pieces, read through the cell lists.
pub const WGSL: &str = "// raw: a distance
// (each piece the largest of its terms, and the shape their union: exact inside, never more than the
// distance outside; the cell lists hold every piece nearer than the cell's margin)
fn pc_hex(q: vec3f, r: f32, h: f32) -> f32 {
  let k = vec3f(-0.8660254, 0.5, 0.57735);
  var p = abs(vec3f(q.y, q.x, q.z));
  p = vec3f(p.xy - 2.0 * min(dot(k.xy, p.xy), 0.0) * k.xy, p.z);
  let d = vec2f(length(p.xy - vec2f(clamp(p.x, -k.z * r, k.z * r), r)) * sign(p.y - r), p.z - h);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}
// the round cone joining the balls (a, ra) and (b, rb)
fn pc_tube(p: vec3f, a: vec4f, b: vec4f) -> f32 {
  let ba = b.xyz - a.xyz;
  let l2 = dot(ba, ba);
  let rr = a.w - b.w;
  let a2 = l2 - rr * rr;
  let il2 = 1.0 / max(l2, 1e-12);
  let pa = p - a.xyz;
  let y = dot(pa, ba);
  let z = y - l2;
  let xv = pa * l2 - ba * y;
  let x2 = dot(xv, xv);
  let y2 = y * y * l2;
  let z2 = z * z * l2;
  let k = sign(rr) * rr * rr * x2;
  if (sign(z) * a2 * z2 > k) { return sqrt(x2 + z2) * il2 - b.w; }
  if (sign(y) * a2 * y2 < k) { return sqrt(x2 + y2) * il2 - a.w; }
  return (sqrt(x2 * a2 * il2) + y * rr) * il2 - a.w;
}
// an outline extruded: even-odd fill of its edges, rounded (a band of a larger outline: its own points only)
fn pc_glyph(p: vec3f, i: u32, t: vec4f) -> f32 {
  let q = tbl[i + 1u];
  let ux = tbl[i + 2u].xyz;
  let uy = tbl[i + 3u].xyz;
  let band = tbl[i + 4u];
  let v = p - q.xyz;
  let x = vec2f(dot(v, ux), dot(v, uy)) / q.w;
  let ne = u32(t.y + 0.5);
  var d2 = 1e18;
  var inside = false;
  for (var e = 0u; e < ne; e++) {
    let s = tbl[i + 5u + e];
    let ab = s.zw - s.xy;
    let h = clamp(dot(x - s.xy, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    let w = x - s.xy - ab * h;
    d2 = min(d2, dot(w, w));
    if ((s.y > x.y) != (s.w > x.y)) {
      if (x.x < ab.x * (x.y - s.y) / ab.y + s.x) { inside = !inside; }
    }
  }
  if (band.z > 0.0) { d2 = min(d2, band.z * band.z); }
  var flat = sqrt(d2) * q.w;
  if (inside) { flat = -flat; }
  if (band.z > 0.0 && (x.y < band.x || x.y > band.y)) { flat = 1e3; }
  let z = abs(dot(v, cross(ux, uy))) - t.z;
  let e2 = vec2f(flat + t.w, z + t.w);
  return length(max(e2, vec2f(0.0))) + min(max(e2.x, e2.y), 0.0) - t.w;
}
// Geode points grow inside their full-size bounds. Only this uniform changes during playback.
fn pc_growth() -> f32 { return 1.0; }
fn pc_prism(p: vec3f, tag: vec4f, base: vec4f, axis: vec3f, side: vec3f) -> f32 {
  let f = clamp((pc_growth() - tag.y) / (1.0 - tag.y), 0.0, 1.0);
  if (f <= 0.0) { return 1e4; }
  let q = p - base.xyz;
  let z = dot(q, axis);
  let x = dot(q, side);
  let y = dot(q, cross(axis, side));
  // The greatest projection onto any of the six equally spaced face normals.
  let radial = max(abs(x), 0.5 * abs(x) + 0.866025404 * abs(y));
  let len = tag.z * f + 0.6 * tag.w + base.w;
  return max(-z, max(radial - tag.w * (0.4 + 0.6 * f), 0.615661475 * (z - len) + 0.788010754 * radial));
}
// one piece (its head at entry i): the largest of its terms
fn pc_piece(p: vec3f, i: u32) -> f32 {
  let n = u32(tbl[i].w);
  var e = -1e9;
  var j = i + 2u;
  let stop = j + n;
  for (var g = 0u; g < 4096u; g++) {
    if (j >= stop) { break; }
    let t = tbl[j];
    if (t.x < 50.0) { e = max(e, dot(t.xyz, p) - t.w); j += 1u; continue; }
    let q = tbl[j + 1u];
    let kind = u32(t.x - 100.0 + 0.5);
    if (kind == 6u) { e = max(e, pc_tube(p, q, tbl[j + 2u])); j += 3u; }
    else if (kind == 9u) { e = max(e, pc_prism(p, t, q, tbl[j + 2u].xyz, tbl[j + 3u].xyz)); j += 4u; }
    else if (kind == 1u) { e = max(e, length(p - q.xyz) - q.w); j += 2u; }
    else if (kind == 2u) { e = max(e, q.w - length(p - q.xyz)); j += 2u; }
    else if (kind == 3u) { e = max(e, abs(length(p - q.xyz) - q.w) - max(t.y, 5.0 * u.hb / u.res)); j += 2u; }
    else if (kind == 4u) { e = max(e, abs(dot(q.xyz, p) - q.w) - max(t.y, 5.0 * u.hb / u.res)); j += 2u; }
    else if (kind == 5u) {
      let ax = tbl[j + 2u].xyz;
      let uu = tbl[j + 3u].xyz;
      let v = p - q.xyz;
      e = max(e, pc_hex(vec3f(dot(v, uu), dot(v, cross(ax, uu)), dot(v, ax)), t.y, t.z));
      j += 4u;
    } else {
      e = max(e, pc_glyph(p, j, t));
      j += 5u + u32(t.y + 0.5);
    }
  }
  return e;
}
fn pc_smin(a: f32, b: f32, k: f32) -> f32 { let h = max(k - abs(a - b), 0.0) / k; return min(a, b) - h * h * k * 0.25; }
// the distance at p, and the entry of the piece that decides it (0 if none, or with a melt)
fn pc_eval(p: vec3f) -> vec2f {
  let g = tbl[1];
  let h = tbl[2];
  let n = u32(h.x);
  let melt = tbl[0].w;
  let c = (p - g.xyz) / g.w;
  // outside the grid: read at the nearest point of the grid, less the way there, and never less than the
  // way to the grid (everything is inside it)
  let box = length(max(max(-c, c - f32(n)), vec3f(0.0))) * g.w;
  var d = 1e9;
  var win = 0u;
  {
    let ci = vec3u(clamp(vec3f(c), vec3f(0.0), vec3f(f32(n) - 1.0)));
    let cell = tbl[3u + (ci.z * n + ci.y) * n + ci.x];
    let count = u32(cell.y);
    if (count == 0u) {
      let cell_d = max(cell.z, h.y);
      d = sqrt(box * box + cell_d * cell_d);
    } else {
      // (with a melt, pieces flow together: a smooth union, the bound then counted from it)
      d = max(cell.z, h.y);
      let first = u32(h.z) * 4u + u32(cell.x);
      for (var k = 0u; k < count; k++) {
        let at = first + k;
        let i = u32(tbl[at / 4u][at % 4u]);
        let mid = tbl[i];
        let half = tbl[i + 1u].xyz;
        let q = abs(p - mid.xyz) - half;
        let boxd = length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
        if (boxd >= d + melt) { continue; }
        let e = pc_piece(p, i);
        if (melt > 0.0) { d = pc_smin(d, e, melt); }
        else if (e < d) { d = e; win = i; }
      }
    }
  }
  if (box > 0.0) { d = max(d, box); win = 0u; }
  if (tbl[0].z > 0.5) {
    let gd = tbl[0].y - p.y;
    if (gd > d) { d = gd; win = 0u; }
  }
  return vec2f(d, f32(win));
}
fn raw(p: vec3f) -> f32 { return pc_eval(p).x; }
// the slope and the value: four readings of the piece that decides (or, with a melt, of the whole)
fn rawGrad(p: vec3f) -> vec4f {
  let r = pc_eval(p);
  let win = u32(r.y);
  let melt = tbl[0].w;
  var g = vec3f(0.0);
  let e = 0.0004;
  for (var i = 0u; i < 4u + u32(u.pad); i++) {
    let o = vec3f(select(-1.0, 1.0, i == 0u || i == 3u), select(-1.0, 1.0, i >= 2u), select(-1.0, 1.0, i == 1u || i == 3u));
    var v = 0.0;
    // (a band of a letter ends where the next band takes over: read across it, the whole is read)
    if (win > 0u && melt <= 0.0 && abs(tbl[win + 2u].x - 107.0) > 0.5) { v = pc_piece(p + e * o, win); } else { v = pc_eval(p + e * o).x; }
    g += o * v;
  }
  if (win == 0u && tbl[0].z > 0.5 && tbl[0].y - p.y >= r.x - 1e-6) { g = vec3f(0.0, -1.0, 0.0); }
  return vec4f(g, r.x);
}
";

#[cfg(test)]
pub mod tests {
    use super::*;

    /// The formula on the CPU, as the shader has it (walls with no floor; hard unions only).
    pub fn dist(t: &[[f32; 4]], p: [f64; 3]) -> f64 { dist_by(t, p, false) }
    /// The same, measuring every piece (`every`), not only the cell's list.
    pub fn dist_by(t: &[[f32; 4]], p: [f64; 3], every: bool) -> f64 {
        let v = |a: [f32; 4]| [a[0] as f64, a[1] as f64, a[2] as f64, a[3] as f64];
        let piece = |i: usize| -> f64 {
            let n = t[i][3] as usize;
            let mut e = -1e9f64;
            let mut j = i + 2;
            while j < i + 2 + n {
                let a = v(t[j]);
                if a[0] < 50.0 { e = e.max(a[0] * p[0] + a[1] * p[1] + a[2] * p[2] - a[3]); j += 1; continue; }
                let q = v(t[j + 1]);
                let r = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
                match a[0].round() as i32 {
                    101 => { e = e.max(r - q[3]); j += 2; }
                    102 => { e = e.max(q[3] - r); j += 2; }
                    103 => { e = e.max((r - q[3]).abs() - a[1]); j += 2; }
                    104 => { e = e.max((q[0] * p[0] + q[1] * p[1] + q[2] * p[2] - q[3]).abs() - a[1]); j += 2; }
                    109 => {
                        let axis = v(t[j + 2]);
                        let side = v(t[j + 3]);
                        let d: [f64; 3] = std::array::from_fn(|k| p[k] - q[k]);
                        let dot = |b: [f64; 4]| (0..3).map(|k| d[k] * b[k]).sum::<f64>();
                        let z = dot(axis);
                        let x = dot(side).abs();
                        let y = (0..3).map(|k| d[k] * (axis[(k+1)%3] * side[(k+2)%3] - axis[(k+2)%3] * side[(k+1)%3])).sum::<f64>().abs();
                        let radial = x.max(0.5 * x + (3.0f64.sqrt() * 0.5) * y);
                        let length = a[2] + 0.6 * a[3] + q[3];
                        let angle = 52.0f64.to_radians();
                        e = e.max((-z).max(radial - a[3]).max(angle.cos() * (z - length) + angle.sin() * radial));
                        j += 4;
                    }
                    106 => {
                        // (a tube of one radius: a capsule)
                        let b = v(t[j + 2]);
                        let ba = [b[0] - q[0], b[1] - q[1], b[2] - q[2]];
                        let pa = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
                        let h = ((pa[0] * ba[0] + pa[1] * ba[1] + pa[2] * ba[2]) / (ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2])).clamp(0.0, 1.0);
                        let w = [pa[0] - ba[0] * h, pa[1] - ba[1] * h, pa[2] - ba[2] * h];
                        e = e.max((w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt() - q[3]);
                        j += 3;
                    }
                    _ => { j += 4; }
                }
            }
            e
        };
        if every {
            // every piece, the cells passed by
            let mut d = f64::MAX;
            let mut i = t[2][3] as usize;
            while i < t[0][0] as usize { d = d.min(piece(i)); i += 2 + t[i][3] as usize; }
            if t[0][2] > 0.5 { d = d.max(t[0][1] as f64 - p[1]); }
            return d;
        }
        let (g, h) = (v(t[1]), v(t[2]));
        let n = h[0] as usize;
        let c = [(p[0] - g[0]) / g[3], (p[1] - g[1]) / g[3], (p[2] - g[2]) / g[3]];
        let box_ = (0..3).map(|k| (-c[k]).max(c[k] - n as f64).max(0.0).powi(2)).sum::<f64>().sqrt() * g[3];
        let mut d = {
            let ci: [usize; 3] = std::array::from_fn(|k| (c[k].max(0.0).floor() as usize).min(n - 1));
            let cell = v(t[3 + (ci[2] * n + ci[1]) * n + ci[0]]);
            if cell[1] == 0.0 {
                cell[2].max(h[1])
            } else {
                let mut d = cell[2].max(h[1]);
                for k in 0..cell[1] as usize {
                    let at = h[2] as usize * 4 + cell[0] as usize + k;
                    let i = t[at / 4][at % 4] as usize;
                    d = d.min(piece(i));
                }
                d
            }
        };
        if box_ > 0.0 { d = box_.max(d - box_); }
        if t[0][2] > 0.5 { d = d.max(t[0][1] as f64 - p[1]); }
        d
    }

    #[test]
    fn a_box_and_a_ball_as_a_table() {
        let mut raw = RawCurve::default();
        raw.pieces.push(geo::box_piece([0.0, 0.0, 0.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.5, 0.5, 0.5]));
        raw.pieces.push(Piece { lo: [1.0, -0.5, -0.5], hi: [2.0, 0.5, 0.5], terms: vec![Term::Ball { c: [1.5, 0.0, 0.0], r: 0.5 }] });
        let t = table(&raw, 1.0);
        assert_eq!(t[0][0] as usize, t.len());
        let (c, far) = ([0.75, 0.0, 0.0], (1.25f64 * 1.25 + 0.5 * 0.5 + 0.5 * 0.5).sqrt());
        let sc = 1.0 / far;
        let at = |p: [f64; 3]| [(p[0] - c[0]) * sc, (p[1] - c[1]) * sc, (p[2] - c[2]) * sc];
        assert!((dist(&t, at([0.0, 0.0, 0.0])) + 0.5 * sc).abs() < 1e-5);
        assert!(dist(&t, at([1.5, 0.0, 0.0])) < 0.0);
        assert!((dist(&t, at([1.5, 0.0, 0.51])) - 0.01 * sc).abs() < 1e-5);
        assert!(dist(&t, at([0.0, 0.8, 0.0])) > 0.0);
    }

    /// Through the cell lists, never more than the distance: a step of the value at a point outside never
    /// crosses into a piece, and the value is what measuring every piece gives, up to the margin.
    #[test]
    fn the_cell_lists_keep_every_piece_that_matters() {
        let mut r = geo::Rng::new(4.0);
        let mut pieces = Vec::new();
        for _ in 0..300 {
            let a = [r.sym(), r.sym(), r.sym()];
            let b = [a[0] + 0.2 * r.sym(), a[1] + 0.2 * r.sym(), a[2] + 0.2 * r.sym()];
            pieces.push(tube(a, 0.02 + 0.02 * r.next(), b, 0.02 + 0.02 * r.next()));
        }
        let all: Vec<([f64; 3], [f64; 3], f64)> = pieces.iter().map(|w| ([w.e[1][0] as f64, w.e[1][1] as f64, w.e[1][2] as f64], [w.e[2][0] as f64, w.e[2][1] as f64, w.e[2][2] as f64], w.e[1][3] as f64)).collect();
        // (equal radii, for the CPU's capsule)
        let pieces: Vec<Written> = all.iter().map(|(a, b, ra)| tube(*a, *ra, *b, *ra)).collect();
        let t = write(pieces, None, 0.0);
        let margin = t[2][1] as f64;
        for _ in 0..4000 {
            let p = [1.3 * r.sym(), 1.3 * r.sym(), 1.3 * r.sym()];
            let truth = all.iter().map(|(a, b, ra)| {
                let ba = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let pa = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
                let h = ((pa[0] * ba[0] + pa[1] * ba[1] + pa[2] * ba[2]) / (ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2]).max(1e-12)).clamp(0.0, 1.0);
                ((pa[0] - ba[0] * h).powi(2) + (pa[1] - ba[1] * h).powi(2) + (pa[2] - ba[2] * h).powi(2)).sqrt() - ra
            }).fold(f64::MAX, f64::min);
            let d = dist(&t, p);
            assert!(d <= truth + 1e-5, "{p:?}: {d} > {truth}");
            // (inside the grid, near a piece, exact; outside it, the way to the grid)
            let (g, n) = (t[1], t[2][0] as f64);
            let inside = (0..3).all(|k| { let c = (p[k] - g[k] as f64) / g[3] as f64; c >= 0.0 && c < n });
            if inside && truth < margin { assert!((d - truth).abs() < 1e-5, "{p:?}: {d} vs {truth}"); }
        }
    }
}
