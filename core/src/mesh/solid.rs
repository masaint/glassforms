//! What a model's distance volume needs from the CPU.
//!
//! The volume is finished on the GPU, which is good at measuring the distance from millions of voxels to
//! the triangles near them and bad at three things, done here:
//!
//!   inside or outside   Along every line of voxels straight through the volume, the heights at which the
//!                       line passes from outside the solid to inside or back (`Flips`). The closed surface
//!                       is crossed by each line a whole number of times, counted with whole-number
//!                       arithmetic, so the answer is exact however the pieces of a model overlap: a point
//!                       is inside if the surface winds round it at least once.
//!   which triangles     Where pieces pass through each other (a spout pushed into a body), the triangles
//!                       inside the solid are not its surface, and a distance measured to them would draw
//!                       a ghost surface in the glass. A triangle is kept where crossing it changes
//!                       inside to outside (`outside_of`).
//!   far from the surface   The exact distance on a coarse grid (`mid_field`), which the GPU reads
//!                       smoothly for everything further from the surface than its own exact band.
//!
//! Positions here are in the stage's own units: the volume spans [-HB, HB] on each axis.

use super::{cross, f64s, len, sub, Examined, Mesh};
use crate::{FIT, HB};
use rayon::prelude::*;
use std::time::Instant;

/// Heights along a line are kept in 65536ths of a voxel.
pub const SUB: u32 = 16;
const ONE: i64 = 1 << SUB;
const HALF: i64 = ONE / 2;
/// The coarse grid stores no distance beyond this.
pub const MID_MAX: f32 = 1.0;

/// How a model is put on the stage: centred, turned so that its up is the stage's, and sized so that its
/// farthest corner is `FIT` from the centre.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fit {
    pub centre: [f64; 3],
    pub scale: f64,
    /// the file's up axis is z (otherwise y, which is the stage's own)
    pub up_z: bool,
}
impl Fit {
    pub fn of(m: &Mesh, up_z: bool) -> Fit {
        let b = m.bounds();
        let centre = [0, 1, 2].map(|a| 0.5 * (b[0][a] as f64 + b[1][a] as f64));
        let far = m.pos.par_iter().map(|p| len(sub(f64s(*p), centre))).reduce(|| 0.0, f64::max);
        Fit { centre, scale: FIT / far.max(1e-30), up_z }
    }
    pub fn apply(&self, p: [f32; 3]) -> [f32; 3] {
        let q = [0, 1, 2].map(|a| (p[a] as f64 - self.centre[a]) * self.scale);
        // z up becomes y up: a quarter turn about x
        let q = if self.up_z { [q[0], q[2], -q[1]] } else { q };
        [q[0] as f32, q[1] as f32, q[2] as f32]
    }
    pub fn place(&self, m: &Mesh) -> Vec<[f32; 3]> {
        m.pos.par_iter().map(|p| self.apply(*p)).collect()
    }
}

/// A stage coordinate on the lattice of an `n`-voxel volume, in 65536ths of a voxel from its low face.
fn grid(x: f64, n: u32) -> i64 {
    (((x + HB) / (2.0 * HB)) * n as f64 * ONE as f64).round() as i64
}
/// A corner with its coordinates in the order (across, down, along) for lines that run along `axis`.
fn turned(p: [f32; 3], n: u32, axis: usize) -> [i64; 3] {
    [grid(p[(axis + 1) % 3] as f64, n), grid(p[(axis + 2) % 3] as f64, n), grid(p[axis] as f64, n)]
}

/// Every line of voxel centres that passes through a triangle (seen along the lines), with the height
/// at which it does and whether the triangle faces up the line there. Whole-number arithmetic: a line
/// that runs exactly through an edge or a corner is given to exactly one of the triangles that meet
/// there, as if every line were a hair to one side, so a closed surface is crossed an even number of
/// times by every line, without exception.
fn scan(v: &[[i64; 3]; 3], n: u32, f: &mut impl FnMut(u32, u32, i64, bool)) {
    let (a, mut b, mut c) = (v[0], v[1], v[2]);
    let area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
    if area == 0 { return; }
    let up = area > 0;
    if !up { std::mem::swap(&mut b, &mut c); }
    let area = area.abs() as i128;
    let n = n as i64;
    let rows = |lo: i64, hi: i64| ((lo - HALF + ONE - 1).div_euclid(ONE).max(0), (hi - HALF).div_euclid(ONE).min(n - 1));
    let (j0, j1) = rows(a[1].min(b[1]).min(c[1]), a[1].max(b[1]).max(c[1]));
    let (i0, i1) = rows(a[0].min(b[0]).min(c[0]), a[0].max(b[0]).max(c[0]));
    if i0 > i1 { return; }
    let edge = |p: [i64; 3], q: [i64; 3], x: i64, y: i64| (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0]);
    // on an edge exactly: the triangle has the line if the line, moved a hair right (and a lesser hair down), is in it
    let has = |w: i64, p: [i64; 3], q: [i64; 3]| w > 0 || (w == 0 && (q[1] < p[1] || (q[1] == p[1] && q[0] > p[0])));
    for j in j0..=j1 {
        let y = j * ONE + HALF;
        // the stretch of this row the triangle covers, roughly, so that long thin triangles cost little
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        for (p, q) in [(a, b), (b, c), (c, a)] {
            if (p[1] - y) * (q[1] - y) > 0 { continue; }
            if p[1] == q[1] {
                lo = lo.min(p[0].min(q[0]) as f64);
                hi = hi.max(p[0].max(q[0]) as f64);
            } else {
                let x = p[0] as f64 + (q[0] - p[0]) as f64 * ((y - p[1]) as f64 / (q[1] - p[1]) as f64);
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
        if lo > hi { continue; }
        let from = (((lo - HALF as f64) / ONE as f64).floor() as i64 - 1).max(i0);
        let to = (((hi - HALF as f64) / ONE as f64).ceil() as i64 + 1).min(i1);
        for i in from..=to {
            let x = i * ONE + HALF;
            let (w0, w1, w2) = (edge(b, c, x, y), edge(c, a, x, y), edge(a, b, x, y));
            if !(has(w0, b, c) && has(w1, c, a) && has(w2, a, b)) { continue; }
            let z = (w0 as i128 * a[2] as i128 + w1 as i128 * b[2] as i128 + w2 as i128 * c[2] as i128).div_euclid(area);
            f(i as u32, j as u32, z as i64, up);
        }
    }
}

/// For every line of voxel centres along one axis of an `n`-voxel volume: the heights (ascending, in
/// 65536ths of a voxel) at which the line passes between outside and inside the solid.
pub struct Flips {
    pub n: u32,
    /// where each line's heights begin in `z`: line (i, j) is at `j * n + i`, and one more entry ends the last
    pub start: Vec<u32>,
    pub z: Vec<u32>,
    /// lines along which the surface did not turn out closed (none, for a surface with every gap closed)
    pub unclosed: usize,
    /// crossings of the surface that the heights were made from
    pub crossings: usize,
}

impl Flips {
    /// From a closed surface wound outward. Lines run along `axis`; line (i, j) has the next axis round
    /// at voxel i and the one after at voxel j (for axis 2: x and y, as a volume is stored).
    pub fn of(pos: &[[f32; 3]], tri: &[[u32; 3]], n: u32, axis: usize) -> Flips {
        let top = (n as i64) << SUB;
        let lots: Vec<Vec<(u32, u32)>> = tri
            .par_chunks(2048)
            .map(|chunk| {
                let mut ev = Vec::new();
                for t in chunk {
                    let v = [turned(pos[t[0] as usize], n, axis), turned(pos[t[1] as usize], n, axis), turned(pos[t[2] as usize], n, axis)];
                    scan(&v, n, &mut |i, j, z, up| ev.push((j * n + i, ((z.clamp(0, top) as u32) << 1) | up as u32)));
                }
                ev
            })
            .collect();
        let lines = (n * n) as usize;
        let mut at = vec![0u32; lines + 1];
        for lot in &lots {
            for e in lot { at[e.0 as usize + 1] += 1; }
        }
        for i in 0..lines { at[i + 1] += at[i]; }
        let crossings = at[lines] as usize;
        let mut ev = vec![0u32; crossings];
        let mut fill = at.clone();
        for lot in &lots {
            for e in lot {
                ev[fill[e.0 as usize] as usize] = e.1;
                fill[e.0 as usize] += 1;
            }
        }
        drop(lots);
        // row by row: each line's crossings from the top down. Above everything is outside; a crossing of a
        // face that looks up the line enters one more layer of the surface, one that looks down leaves one;
        // the line is inside wherever it is within at least one.
        let rows: Vec<(Vec<u32>, Vec<u32>, usize)> = (0..n as usize)
            .into_par_iter()
            .map(|j| {
                let (mut zs, mut counts, mut unclosed) = (Vec::new(), Vec::with_capacity(n as usize), 0);
                let mut buf: Vec<u32> = Vec::new();
                for i in 0..n as usize {
                    let line = j * n as usize + i;
                    buf.clear();
                    buf.extend_from_slice(&ev[at[line] as usize..at[line + 1] as usize]);
                    buf.sort_unstable();
                    let had = zs.len();
                    let (mut depth, mut inside) = (0i32, false);
                    let mut k = buf.len();
                    while k > 0 {
                        let z = buf[k - 1] >> 1;
                        while k > 0 && buf[k - 1] >> 1 == z {
                            depth += if buf[k - 1] & 1 == 1 { 1 } else { -1 };
                            k -= 1;
                        }
                        if (depth >= 1) != inside {
                            inside = !inside;
                            zs.push(z);
                        }
                    }
                    if depth != 0 { unclosed += 1; }
                    // (below everything is outside too, whatever the surface says)
                    if inside { zs.push(0); }
                    zs[had..].reverse();
                    counts.push((zs.len() - had) as u32);
                }
                (zs, counts, unclosed)
            })
            .collect();
        let mut start = Vec::with_capacity(lines + 1);
        let mut z = Vec::with_capacity(rows.iter().map(|r| r.0.len()).sum());
        let mut unclosed = 0;
        start.push(0);
        for (zs, counts, u) in rows {
            for c in counts { start.push(start.last().unwrap() + c); }
            z.extend_from_slice(&zs);
            unclosed += u;
        }
        Flips { n, start, z, unclosed, crossings }
    }

    fn line(&self, i: u32, j: u32) -> &[u32] {
        let l = (j * self.n + i) as usize;
        &self.z[self.start[l] as usize..self.start[l + 1] as usize]
    }
    /// Is the point at height `z` (65536ths of a voxel) on line (i, j) inside the solid?
    pub fn inside(&self, i: u32, j: u32, z: i64) -> bool {
        let l = self.line(i, j);
        (l.len() - l.partition_point(|f| (*f as i64) < z)) % 2 == 1
    }
    /// Is the centre of voxel k of line (i, j) inside the solid?
    pub fn inside_voxel(&self, i: u32, j: u32, k: u32) -> bool {
        self.inside(i, j, k as i64 * ONE + HALF)
    }
    /// Does line (i, j) pass between inside and outside within `tol` of height `z`?
    fn flips_near(&self, i: u32, j: u32, z: i64, tol: i64) -> bool {
        let l = self.line(i, j);
        let k = l.partition_point(|f| (*f as i64) < z - tol);
        k < l.len() && (l[k] as i64) <= z + tol
    }
}

/// The triangles that are the outside of the solid: those across which a line passes between inside
/// and outside. A triangle that is partly so (it runs into another piece) is cut into four, again and
/// again, down to a couple of voxels, and the parts are kept or dropped each by itself.
pub fn outside_of(pos: &[[f32; 3]], tri: &[[u32; 3]], flips: &[Flips; 3]) -> (Vec<[f32; 9]>, usize) {
    let n = flips[0].n;
    let h = 2.0 * HB / n as f64;
    fn sort(t: [[f64; 3]; 3], flips: &[Flips; 3], n: u32, h: f64, depth: u32, out: &mut Vec<[f32; 9]>, hidden: &mut usize) {
        let nrm = cross(sub(t[1], t[0]), sub(t[2], t[0]));
        let big = nrm[0].abs().max(nrm[1].abs()).max(nrm[2].abs());
        if big == 0.0 {
            *hidden += 1;
            return;
        }
        // looked at along the axis it faces most squarely
        let axis = (0..3).find(|a| nrm[*a].abs() == big).unwrap_or(2);
        let f = &flips[axis];
        let v = t.map(|p| turned([p[0] as f32, p[1] as f32, p[2] as f32], n, axis));
        let (mut seen, mut shown) = (0u32, 0u32);
        scan(&v, n, &mut |i, j, z, _| {
            seen += 1;
            shown += f.flips_near(i, j, z, 8) as u32;
        });
        if seen == 0 {
            // smaller than the lattice: judged at the line nearest its middle, by whether that line passes
            // between inside and outside anywhere near the heights the triangle spans
            let c = [0, 1, 2].map(|a| (t[0][a] + t[1][a] + t[2][a]) / 3.0);
            let cell = |x: f64| ((grid(x, n) as f64 / ONE as f64).floor() as i64).clamp(0, n as i64 - 1);
            let (i, j) = (cell(c[(axis + 1) % 3]), cell(c[(axis + 2) % 3]));
            let (lo, hi) = (v[0][2].min(v[1][2]).min(v[2][2]), v[0][2].max(v[1][2]).max(v[2][2]));
            seen = 1;
            shown = f.flips_near(i as u32, j as u32, (lo + hi) / 2, (hi - lo) / 2 + ONE + ONE / 4) as u32;
        }
        let keep = |out: &mut Vec<[f32; 9]>| out.push([0, 1, 2, 3, 4, 5, 6, 7, 8].map(|k: usize| t[k / 3][k % 3] as f32));
        if shown == seen {
            keep(out);
        } else if shown == 0 {
            *hidden += 1;
        } else {
            let longest = len(sub(t[1], t[0])).max(len(sub(t[2], t[1]))).max(len(sub(t[0], t[2])));
            if longest <= h || depth >= 16 {
                if shown * 2 >= seen { keep(out) } else { *hidden += 1 }
                return;
            }
            let m = |a: [f64; 3], b: [f64; 3]| [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])];
            let (ab, bc, ca) = (m(t[0], t[1]), m(t[1], t[2]), m(t[2], t[0]));
            for part in [[t[0], ab, ca], [ab, t[1], bc], [ca, bc, t[2]], [ab, bc, ca]] {
                sort(part, flips, n, h, depth + 1, out, hidden);
            }
        }
    }
    let parts: Vec<(Vec<[f32; 9]>, usize)> = tri
        .par_chunks(1024)
        .map(|chunk| {
            let (mut out, mut hidden) = (Vec::new(), 0);
            for t in chunk {
                sort([f64s(pos[t[0] as usize]), f64s(pos[t[1] as usize]), f64s(pos[t[2] as usize])], flips, n, h, 0, &mut out, &mut hidden);
            }
            (out, hidden)
        })
        .collect();
    let mut all = Vec::with_capacity(parts.iter().map(|p| p.0.len()).sum());
    let mut hidden = 0;
    for (p, h) in parts {
        all.extend_from_slice(&p);
        hidden += h;
    }
    (all, hidden)
}

// ---- the nearest triangle to a point -------------------------------------------------------------
type V3 = [f32; 3];
fn sub3(a: V3, b: V3) -> V3 { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn dot3(a: V3, b: V3) -> f32 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }

/// The square of the distance from `p` to a triangle (after Ericson, Real-Time Collision Detection).
pub fn dist2_triangle(p: V3, t: &[f32; 9]) -> f32 {
    let (a, b, c) = ([t[0], t[1], t[2]], [t[3], t[4], t[5]], [t[6], t[7], t[8]]);
    let (ab, ac, ap) = (sub3(b, a), sub3(c, a), sub3(p, a));
    let (d1, d2) = (dot3(ab, ap), dot3(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 { return dot3(ap, ap); }
    let bp = sub3(p, b);
    let (d3, d4) = (dot3(ab, bp), dot3(ac, bp));
    if d3 >= 0.0 && d4 <= d3 { return dot3(bp, bp); }
    let to = |q: V3| { let d = sub3(p, q); dot3(d, d) };
    let along = |from: V3, dir: V3, s: f32| [from[0] + dir[0] * s, from[1] + dir[1] * s, from[2] + dir[2] * s];
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 { return to(along(a, ab, d1 / (d1 - d3))); }
    let cp = sub3(p, c);
    let (d5, d6) = (dot3(ab, cp), dot3(ac, cp));
    if d6 >= 0.0 && d5 <= d6 { return dot3(cp, cp); }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 { return to(along(a, ac, d2 / (d2 - d6))); }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 { return to(along(b, sub3(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6)))); }
    let sum = va + vb + vc;
    if sum == 0.0 { return dot3(ap, ap).min(dot3(bp, bp)).min(dot3(cp, cp)); }
    let (v, w) = (vb / sum, vc / sum);
    to([a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w])
}

/// Triangles sorted into a tree of boxes, so that the nearest to a point is found without looking at them all.
pub struct Tree {
    /// boxes: low corner, high corner; then either two children (`count` 0: at `first` and `first + 1`) or
    /// `count` triangles from `first`
    nodes: Vec<Node>,
    pub tris: Vec<[f32; 9]>,
}
#[derive(Clone, Copy)]
struct Node {
    lo: V3,
    hi: V3,
    first: u32,
    count: u32,
}

impl Tree {
    pub fn of(mut tris: Vec<[f32; 9]>) -> Tree {
        let mid = |t: &[f32; 9], a: usize| t[a] + t[3 + a] + t[6 + a];
        let bound = |ts: &[[f32; 9]]| {
            let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
            for t in ts {
                for k in 0..9 {
                    lo[k % 3] = lo[k % 3].min(t[k]);
                    hi[k % 3] = hi[k % 3].max(t[k]);
                }
            }
            (lo, hi)
        };
        let mut nodes = Vec::with_capacity(tris.len() / 2 + 2);
        let (lo, hi) = bound(&tris);
        nodes.push(Node { lo, hi, first: 0, count: tris.len() as u32 });
        let mut todo = vec![0usize];
        while let Some(i) = todo.pop() {
            let Node { lo, hi, first, count } = nodes[i];
            if count <= 6 { continue; }
            let (from, to) = (first as usize, (first + count) as usize);
            // halved by the middle one along the box's longest side
            let axis = (0..3).max_by(|a, b| (hi[*a] - lo[*a]).total_cmp(&(hi[*b] - lo[*b]))).unwrap_or(0);
            let half = count as usize / 2;
            tris[from..to].select_nth_unstable_by(half, |x, y| mid(x, axis).total_cmp(&mid(y, axis)));
            let kids = nodes.len();
            let (l, r) = (bound(&tris[from..from + half]), bound(&tris[from + half..to]));
            nodes.push(Node { lo: l.0, hi: l.1, first, count: half as u32 });
            nodes.push(Node { lo: r.0, hi: r.1, first: first + half as u32, count: count - half as u32 });
            nodes[i].first = kids as u32;
            nodes[i].count = 0;
            todo.push(kids);
            todo.push(kids + 1);
        }
        // (a node that was halved keeps its triangles' range in its children, not itself)
        Tree { nodes, tris }
    }

    /// Bounds and child/triangle ranges in the same order used by the CPU nearest-point query.
    pub fn node_words(&self) -> impl Iterator<Item = [u32; 8]> + '_ {
        self.nodes.iter().map(|n| [
            n.lo[0].to_bits(), n.lo[1].to_bits(), n.lo[2].to_bits(), n.first,
            n.hi[0].to_bits(), n.hi[1].to_bits(), n.hi[2].to_bits(), n.count,
        ])
    }

    fn box_dist2(n: &Node, p: V3) -> f32 {
        let mut d = 0.0;
        for a in 0..3 {
            let e = (n.lo[a] - p[a]).max(p[a] - n.hi[a]).max(0.0);
            d += e * e;
        }
        d
    }

    /// The square of the distance from `p` to the nearest triangle, if it is less than `within`
    /// (otherwise `within`).
    pub fn dist2(&self, p: V3, within: f32) -> f32 {
        let mut best = within;
        if self.tris.is_empty() { return best; }
        let mut stack = [0u32; 64];
        let mut top = 1;
        while top > 0 {
            top -= 1;
            let n = &self.nodes[stack[top] as usize];
            if Self::box_dist2(n, p) >= best { continue; }
            if n.count > 0 {
                for t in &self.tris[n.first as usize..(n.first + n.count) as usize] {
                    best = best.min(dist2_triangle(p, t));
                }
            } else {
                // the nearer child last, so that it is looked at first
                let (l, r) = (n.first, n.first + 1);
                let (dl, dr) = (Self::box_dist2(&self.nodes[l as usize], p), Self::box_dist2(&self.nodes[r as usize], p));
                let (near, far, dfar) = if dl <= dr { (l, r, dr) } else { (r, l, dl) };
                if dfar < best && top < 63 {
                    stack[top] = far;
                    top += 1;
                }
                if top < 64 {
                    stack[top] = near;
                    top += 1;
                }
            }
        }
        best
    }
}

/// Within this distance of the surface every point of the coarse grid is measured exactly; further out
/// it is read from a grid four times coarser still (itself exact), through a smooth curve.
pub const MID_NEAR: f32 = 0.3;

/// One row of exact distances along x, each bounded by its neighbour's (a distance changes by no more
/// than the step between two points).
fn exact_row(tree: &Tree, at: &dyn Fn(u32) -> f32, m: u32, y: f32, z: f32, row: &mut [f32]) {
    let step = at(1) - at(0);
    let mut last = MID_MAX;
    for i in 0..m {
        let reach = (last + step).min(MID_MAX);
        last = tree.dist2([at(i), y, z], reach * reach).sqrt();
        row[i as usize] = last;
    }
}

/// The distance from every point of a coarse grid (`m` to a side, spanning the volume) to the nearest of
/// the tree's triangles, no further than `MID_MAX`; negative inside the solid when `sign` (lines along z
/// of an `m`-voxel volume) is given. Point (i, j, k) is at `(k * m + j) * m + i`.
pub fn mid_field(tree: &Tree, m: u32, sign: Option<&Flips>) -> Vec<f32> {
    let centre = |n: u32| move |i: u32| (((i as f64 + 0.5) / n as f64 * 2.0 - 1.0) * HB) as f32;
    // a grid a quarter as fine, exact everywhere: what the far points are read from, and what says which are far
    let c = (m / 4).max(8);
    let cat = centre(c);
    let mut far = vec![0.0f32; (c as usize).pow(3)];
    far.par_chunks_mut(c as usize).enumerate().for_each(|(r, row)| {
        let (j, k) = (r as u32 % c, r as u32 / c);
        exact_row(tree, &cat, c, cat(j), cat(k), row);
    });
    let far_at = |i: i64, j: i64, k: i64| {
        let cl = |v: i64| v.clamp(0, c as i64 - 1) as usize;
        far[(cl(k) * c as usize + cl(j)) * c as usize + cl(i)]
    };
    // the weights of the four points a smooth curve through equally spaced points uses (Catmull-Rom)
    let weights = |t: f32| [((-t + 2.0) * t - 1.0) * t * 0.5, ((3.0 * t - 5.0) * t * t + 2.0) * 0.5, ((-3.0 * t + 4.0) * t + 1.0) * t * 0.5, (t - 1.0) * t * t * 0.5];
    let cell = (2.0 * HB / c as f64) as f32;
    let at = centre(m);
    let mut out = vec![0.0f32; (m as usize).pow(3)];
    out.par_chunks_mut(m as usize).enumerate().for_each(|(r, row)| {
        let (j, k) = (r as u32 % m, r as u32 / m);
        // where this row lies among the coarser points
        let pos = |i: u32| (i as f32 + 0.5) * c as f32 / m as f32 - 0.5;
        let (fy, fz) = (pos(j), pos(k));
        let (j0, k0) = (fy.floor() as i64, fz.floor() as i64);
        let (wy, wz) = (weights(fy - j0 as f32), weights(fz - k0 as f32));
        let mut last = MID_MAX;
        for i in 0..m {
            let fx = pos(i);
            let i0 = fx.floor() as i64;
            // the least of the coarser points around: nothing here is nearer the surface than that less a cell's diagonal
            let mut least = f32::MAX;
            for d in 0..8 { least = least.min(far_at(i0 + (d & 1), j0 + ((d >> 1) & 1), k0 + (d >> 2))); }
            let d = if least - 1.75 * cell > MID_NEAR {
                let wx = weights(fx - i0 as f32);
                let mut v = 0.0;
                for (c2, w2) in wz.iter().enumerate() {
                    for (c1, w1) in wy.iter().enumerate() {
                        for (c0, w0) in wx.iter().enumerate() {
                            v += w0 * w1 * w2 * far_at(i0 - 1 + c0 as i64, j0 - 1 + c1 as i64, k0 - 1 + c2 as i64);
                        }
                    }
                }
                v.clamp(MID_NEAR, MID_MAX)
            } else {
                // no further than the last point's distance and a step, nor than the nearest of the coarser
                // points around and a cell's diagonal: the search need look no wider
                let reach = (last + (at(1) - at(0))).min(least + 1.75 * cell).min(MID_MAX);
                tree.dist2([at(i), at(j), at(k)], reach * reach).sqrt()
            };
            last = d;
            let inside = sign.is_some_and(|f| f.inside_voxel(i, j, k));
            row[i as usize] = if inside { -d } else { d };
        }
    });
    out
}

/// Turn the values of a grid (`m` to a side) into the weights of the cubic B-spline that passes through
/// them. A B-spline read straight from values is smooth but soft: it rounds every corner of what it
/// reads off by a fair part of a cell, which near a sharp edge overstates the distance. Read from these
/// weights instead it gives the values back exactly at the grid's points and is as smooth between them.
/// (Unser's recursive filter, along each axis in turn. Past each end a line is carried on as it was
/// going, turned about its last value: distances near the faces of the volume are ramps, and a line
/// taken to be mirrored there would ring.)
pub fn spline_through(v: &mut [f32], m: usize) {
    const Z: f32 = -0.267_949_2;      // sqrt(3) - 2
    const PAD: usize = 12;
    if m < 2 { return; }
    fn line(l: &mut [f32], p: &mut Vec<f32>) {
        let n = l.len();
        p.clear();
        p.resize(n + 2 * PAD, 0.0);
        p[PAD..PAD + n].copy_from_slice(l);
        for k in 0..PAD {
            p[PAD - 1 - k] = 2.0 * l[0] - l[(k + 1).min(n - 1)];
            p[PAD + n + k] = 2.0 * l[n - 1] - l[(n - 1).saturating_sub(k + 1)];
        }
        let q = p.len();
        let (mut sum, mut zk) = (p[0], Z);
        for s in p.iter().take(24).skip(1) {
            sum += zk * s;
            zk *= Z;
        }
        p[0] = sum;
        for k in 1..q { p[k] += Z * p[k - 1]; }
        p[q - 1] = (Z / (Z * Z - 1.0)) * (p[q - 1] + Z * p[q - 2]);
        for k in (0..q - 1).rev() { p[k] = Z * (p[k + 1] - p[k]); }
        for (x, w) in l.iter_mut().zip(&p[PAD..PAD + n]) { *x = 6.0 * w; }
    }
    // along x: every row is a line
    v.par_chunks_mut(m).for_each_init(Vec::new, |p, l| line(l, p));
    // along y: within each slice, every column
    v.par_chunks_mut(m * m).for_each_init(|| (Vec::new(), vec![0.0f32; m]), |(p, tmp), slice| {
        for i in 0..m {
            for j in 0..m { tmp[j] = slice[j * m + i]; }
            line(tmp, p);
            for j in 0..m { slice[j * m + i] = tmp[j]; }
        }
    });
    // along z: the grid turned so that its lines along z lie together, and turned back
    let mut turned = vec![0.0f32; v.len()];
    turned.par_chunks_mut(m * m).enumerate().for_each_init(Vec::new, |p, (j, rows)| {
        for i in 0..m {
            let l = &mut rows[i * m..(i + 1) * m];
            for (k, x) in l.iter_mut().enumerate() { *x = v[(k * m + j) * m + i]; }
            line(l, p);
        }
    });
    v.par_chunks_mut(m * m).enumerate().for_each(|(k, slice)| {
        for j in 0..m {
            for i in 0..m { slice[j * m + i] = turned[(j * m + i) * m + k]; }
        }
    });
}

/// How fine the coarse grid is for a volume of `res` voxels a side: a quarter as fine, within bounds.
pub fn mid_res(res: u32) -> u32 {
    ((res / 4).clamp(64, 256) + 7) / 8 * 8
}

/// Which triangles reach which slice of a volume: (triangle, slice) pairs in order of slice, and where
/// each slice's pairs begin (one more entry ends the last). A triangle reaches a slice if any of it is
/// within `band` of the slice's depth.
pub fn reach(tris: &[[f32; 9]], res: u32, band: f64) -> (Vec<[u32; 2]>, Vec<u32>) {
    let n = res as usize;
    let span = |t: &[f32; 9]| -> (i64, i64) {
        let (lo, hi) = (t[2].min(t[5]).min(t[8]) as f64 - band, t[2].max(t[5]).max(t[8]) as f64 + band);
        let a = (((lo / HB + 1.0) * 0.5 * res as f64 - 0.5).ceil() as i64).max(0);
        let b = (((hi / HB + 1.0) * 0.5 * res as f64 - 0.5).floor() as i64).min(res as i64 - 1);
        (a, b)
    };
    let mut start = vec![0u32; n + 1];
    for t in tris {
        let (a, b) = span(t);
        for k in a..=b { start[k as usize + 1] += 1; }
    }
    for k in 0..n { start[k + 1] += start[k]; }
    let mut fill = start.clone();
    let mut pairs = vec![[0u32; 2]; start[n] as usize];
    for (i, t) in tris.iter().enumerate() {
        let (a, b) = span(t);
        for k in a..=b {
            pairs[fill[k as usize] as usize] = [i as u32, k as u32];
            fill[k as usize] += 1;
        }
    }
    (pairs, start)
}

/// Everything a volume of one size needs from the CPU.
pub struct Prep {
    /// voxels a side of the volume this was made for
    pub res: u32,
    /// a shell (the surface as it is, no inside) rather than a solid
    pub shell: bool,
    /// the triangles distances are measured to: the outside of the solid, or the whole surface for a shell
    pub tris: Vec<[f32; 9]>,
    /// How far from the surface the volume holds the distance to the triangles themselves, and where
    /// within that it starts to give way to the coarse grid (both in the stage's units).
    pub band: f64,
    pub lo: f64,
    /// which triangles reach which slice (see `reach`)
    pub pairs: Vec<[u32; 2]>,
    pub slices: Vec<u32>,
    /// a solid's inside: the lines along z (empty for a shell)
    pub start: Vec<u32>,
    pub flips: Vec<u32>,
    /// the coarse grid: `mid_res` to a side; signed for a solid
    pub mid: Vec<f32>,
    pub mid_res: u32,
    /// triangles that were inside the solid and were left out; lines on which the surface was not closed
    pub hidden: usize,
    pub unclosed: usize,
    /// milliseconds: placing, inside and outside, sorting the triangles, the tree, the coarse grid
    pub ms: [f64; 5],
}

/// Voxels either side of the surface in which the volume holds the distance to the triangles themselves
/// (beyond, the coarse grid's). At least what the tracer reads when it settles on the surface and takes
/// its normal, with room; and as many more as can be afforded, because the coarse grid cannot follow a
/// sharp edge closely and a model of few triangles (which is where sharp edges are) costs little to
/// draw wide. Every triangle is drawn into a block of about (2 band + 3) voxels a side, and the whole
/// is held to some thousand million voxels drawn.
pub const BAND_LEAST: f64 = 5.0;
pub const BAND_MOST: f64 = 16.0;
pub fn band_voxels(triangles: usize) -> f64 {
    (((1e9 / triangles.max(1) as f64).cbrt() - 3.0) / 2.0).floor().clamp(BAND_LEAST, BAND_MOST)
}

/// The most crossings of the surface by the lines of voxels along one axis, and the most voxels the
/// triangles may be drawn across on the sheets. A model made to be awkward (millions of triangles each
/// as large as the stage) would otherwise ask for memory and time without end; no model of ordinary
/// make comes near either (a million triangles at 768 voxels: 1 million and 2 thousand million).
pub const MOST_CROSSINGS: f64 = 1.5e8;
pub const MOST_DRAWN: f64 = 1.2e10;

/// Is there too much of this surface for a volume of `res` voxels a side? Says so if there is.
fn too_much(pos: &[[f32; 3]], tri: &[[u32; 3]], res: u32, band: f64, signed: bool) -> Result<(), String> {
    let h = 2.0 * HB / res as f64;
    let (cross_x, cross_y, cross_z, drawn) = tri
        .par_iter()
        .map(|t| {
            let (a, b, c) = (f64s(pos[t[0] as usize]), f64s(pos[t[1] as usize]), f64s(pos[t[2] as usize]));
            let n = cross(sub(b, a), sub(c, a));
            // the block of voxels the triangle is drawn into: its own box and the band all round, within the volume
            let side = |k: usize| ((a[k].max(b[k]).max(c[k]) - a[k].min(b[k]).min(c[k])).min(2.0 * HB) / h + 2.0 * band / h + 1.0).min(res as f64);
            (0.5 * n[0].abs() / (h * h), 0.5 * n[1].abs() / (h * h), 0.5 * n[2].abs() / (h * h), side(0) * side(1) * side(2))
        })
        .reduce(|| (0.0, 0.0, 0.0, 0.0), |p, q| (p.0 + q.0, p.1 + q.1, p.2 + q.2, p.3 + q.3));
    let crossings = cross_x.max(cross_y).max(cross_z);
    if signed && crossings > MOST_CROSSINGS {
        return Err(format!("there is too much surface for a volume of {res} voxels a side: its lines would cross the model {:.0} million times (the most is {:.0} million); use less detail", crossings / 1e6, MOST_CROSSINGS / 1e6));
    }
    if drawn > MOST_DRAWN {
        return Err(format!("there is too much surface for a volume of {res} voxels a side: its triangles would be drawn across {:.0} thousand million voxels (the most is {:.0}); use less detail", drawn / 1e9, MOST_DRAWN / 1e9));
    }
    Ok(())
}

/// Prepare a model for a volume of `res` voxels a side, as a solid or as a shell. An error if the model
/// has more surface than a volume of that size can take (see `MOST_CROSSINGS`).
pub fn prepare(ex: &Examined, up_z: bool, shell: bool, res: u32) -> Result<Prep, String> {
    let t0 = Instant::now();
    let fit = Fit::of(&ex.surface, up_z);
    let m = mid_res(res);
    let h = 2.0 * HB / res as f64;
    let mut ms = [0.0; 5];
    let mut lap = { let mut t = t0; move || { let d = t.elapsed().as_secs_f64() * 1e3; t = Instant::now(); d } };
    if shell {
        let pos = fit.place(&ex.surface);
        too_much(&pos, &ex.surface.tri, res, band_voxels(ex.surface.tri.len()) * h, false)?;
        let tris: Vec<[f32; 9]> = ex.surface.tri.par_iter().map(|t| { let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]); [a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]] }).collect();
        ms[0] = lap();
        let tree = Tree::of(tris);
        ms[3] = lap();
        let mid = mid_field(&tree, m, None);
        ms[4] = lap();
        let band = band_voxels(tree.tris.len()) * h;
        let (pairs, slices) = reach(&tree.tris, res, band);
        return Ok(Prep { res, shell, tris: tree.tris, band, lo: band - 1.5 * h, pairs, slices, start: Vec::new(), flips: Vec::new(), mid, mid_res: m, hidden: 0, unclosed: 0, ms });
    }
    let pos = fit.place(&ex.closed);
    let tri = &ex.closed.tri;
    too_much(&pos, tri, res, band_voxels(tri.len()) * h, true)?;
    ms[0] = lap();
    let flips = [Flips::of(&pos, tri, res, 0), Flips::of(&pos, tri, res, 1), Flips::of(&pos, tri, res, 2)];
    let coarse = Flips::of(&pos, tri, m, 2);
    ms[1] = lap();
    let (tris, hidden) = outside_of(&pos, tri, &flips);
    ms[2] = lap();
    let tree = Tree::of(tris);
    ms[3] = lap();
    // (a solid's coarse grid is read through a spline that passes through its values; a shell's has a
    // crease at the surface, which such a spline would ring about, so it is left as values)
    let mut mid = mid_field(&tree, m, Some(&coarse));
    spline_through(&mut mid, m as usize);
    ms[4] = lap();
    let unclosed = flips.iter().map(|f| f.unclosed).sum();
    let [_, _, z] = flips;
    let band = band_voxels(tree.tris.len()) * h;
    let (pairs, slices) = reach(&tree.tris, res, band);
    Ok(Prep { res, shell, tris: tree.tris, band, lo: band - 1.5 * h, pairs, slices, start: z.start, flips: z.z, mid, mid_res: m, hidden, unclosed, ms })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_distance_to_a_triangle_is_right_in_every_region() {
        let t = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let near = |a: f32, b: f32| (a - b).abs() < 1e-6;
        assert!(near(dist2_triangle([0.25, 0.25, 0.5], &t), 0.25));          // above its face
        assert!(near(dist2_triangle([-1.0, -1.0, 0.0], &t), 2.0));           // off a corner
        assert!(near(dist2_triangle([0.5, -2.0, 0.0], &t), 4.0));            // off an edge
        assert!(near(dist2_triangle([1.0, 1.0, 0.0], &t), 0.5));             // off the long edge
        assert!(near(dist2_triangle([3.0, 0.0, 4.0], &t), 4.0 + 16.0));      // off the far corner
        // a triangle with no area is its longest side
        let flat = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        assert!(near(dist2_triangle([1.0, 3.0, 0.0], &flat), 9.0));
    }

    #[test]
    fn a_spline_through_values_gives_them_back() {
        let m = 12;
        let mut v: Vec<f32> = (0..m * m * m).map(|i| ((i * 37 % 101) as f32 * 0.01).sin() + (i / (m * m)) as f32 * 0.1).collect();
        let was = v.clone();
        spline_through(&mut v, m);
        // a cubic B-spline at a grid point is (1, 4, 1) / 6 of the weights along each axis
        let at = |i: usize, j: usize, k: usize| v[(k * m + j) * m + i];
        for k in 2..m - 2 {
            for j in 2..m - 2 {
                for i in 2..m - 2 {
                    let mut sum = 0.0;
                    for (dk, wk) in [(0, 1.0), (1, 4.0), (2, 1.0)] {
                        for (dj, wj) in [(0, 1.0), (1, 4.0), (2, 1.0)] {
                            for (di, wi) in [(0, 1.0), (1, 4.0), (2, 1.0)] {
                                sum += wk * wj * wi * at(i + di - 1, j + dj - 1, k + dk - 1);
                            }
                        }
                    }
                    assert!((sum / 216.0 - was[(k * m + j) * m + i]).abs() < 2e-4, "{} against {}", sum / 216.0, was[(k * m + j) * m + i]);
                }
            }
        }
    }

    #[test]
    fn a_line_through_a_shared_edge_or_corner_is_counted_once() {
        // a square of two triangles whose shared edge, corners and outline all lie exactly on lines of voxel centres
        let n = 8;
        let at = |i: f32| ((i + 0.5) / n as f32 * 2.0 - 1.0) * HB as f32;
        let pos = vec![[at(1.0), at(1.0), 0.1], [at(5.0), at(1.0), 0.1], [at(5.0), at(5.0), 0.1], [at(1.0), at(5.0), 0.1]];
        for tri in [vec![[0u32, 1, 2], [0, 2, 3]], vec![[0, 1, 3], [1, 2, 3]]] {
            let mut hits = vec![0u32; (n * n) as usize];
            for t in &tri {
                let v = [turned(pos[t[0] as usize], n, 2), turned(pos[t[1] as usize], n, 2), turned(pos[t[2] as usize], n, 2)];
                scan(&v, n, &mut |i, j, _, up| { assert!(up); hits[(j * n + i) as usize] += 1; });
            }
            // every line is in the square once or not at all, and the square holds sixteen of them
            assert!(hits.iter().all(|h| *h <= 1), "{hits:?}");
            assert_eq!(hits.iter().sum::<u32>(), 16);
        }
    }
}
