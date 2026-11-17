//! Small things the newer families share: vectors, a seeded generator, and sampling curves.

use super::*;
use std::f64::consts::PI;

pub const TAU: f64 = 2.0 * PI;
pub type V = [f64; 3];

pub fn add(a: V, b: V) -> V { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
pub fn sub(a: V, b: V) -> V { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
pub fn mul(a: V, s: f64) -> V { [a[0] * s, a[1] * s, a[2] * s] }
pub fn dot(a: V, b: V) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
pub fn cross(a: V, b: V) -> V { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }
pub fn len(a: V) -> f64 { dot(a, a).sqrt() }
pub fn unit(a: V) -> V {
    let l = len(a);
    if l > 1e-300 { mul(a, 1.0 / l) } else { [0.0, 1.0, 0.0] }
}
pub fn lerp(a: V, b: V, t: f64) -> V { add(a, mul(sub(b, a), t)) }
/// A unit vector at right angles to d.
pub fn across(d: V) -> V {
    let a = if d[1].abs() < 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
    unit(cross(d, a))
}
/// v turned by angle a about the unit axis k (Rodrigues).
pub fn turn(v: V, k: V, a: f64) -> V {
    let (c, s) = (a.cos(), a.sin());
    add(add(mul(v, c), mul(cross(k, v), s)), mul(k, dot(k, v) * (1.0 - c)))
}

/// A small seeded generator (splitmix64): the same seed always gives the same shape.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: f64) -> Self { Self((seed.abs().round() as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ 0x2545f4914f6cdd1d) }
    pub fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    /// -1..1
    pub fn sym(&mut self) -> f64 { 2.0 * self.next() - 1.0 }
    pub fn range(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.next() }
    /// a direction, evenly over the sphere
    pub fn dir(&mut self) -> V {
        let z = self.sym();
        let a = TAU * self.next();
        let r = (1.0 - z * z).max(0.0).sqrt();
        [r * a.cos(), z, r * a.sin()]
    }
    pub fn gauss(&mut self) -> f64 { (-2.0 * (self.next() + 1e-12).ln()).sqrt() * (TAU * self.next()).cos() }
}

/// A closed curve t -> point over one turn of t, as n + 1 points (the last repeats the first).
pub fn ring(f: impl Fn(f64) -> V, n: usize) -> Vec<f64> {
    let mut a = Vec::with_capacity((n + 1) * 3);
    for i in 0..=n {
        a.extend_from_slice(&f((i % n) as f64 / n as f64 * TAU));
    }
    a
}
/// A circle: centre c, radius r, in the plane of the unit vectors a and b.
pub fn circle(c: V, a: V, b: V, r: f64, n: usize) -> Vec<f64> {
    ring(|t| add(c, add(mul(a, r * t.cos()), mul(b, r * t.sin()))), n)
}
pub fn closed(lines: Vec<Vec<f64>>) -> RawCurve {
    RawCurve { lines, closed: true, ..Default::default() }
}
pub fn open(lines: Vec<Vec<f64>>) -> RawCurve {
    RawCurve { lines, closed: false, ..Default::default() }
}
pub fn flat(pts: &[V]) -> Vec<f64> { pts.iter().flat_map(|p| p.iter().copied()).collect() }

/// A box piece: the centre, three axes (unit, at right angles) and half-sizes along them.
pub fn box_piece(c: V, axes: [V; 3], half: V) -> Piece {
    let mut terms = Vec::with_capacity(6);
    let mut ext = [0.0; 3];
    for k in 0..3 {
        let a = axes[k];
        let d = dot(a, c);
        terms.push(Term::Plane { n: a, d: d + half[k] });
        terms.push(Term::Plane { n: mul(a, -1.0), d: -d + half[k] });
        for (j, e) in ext.iter_mut().enumerate() { *e += (a[j] * half[k]).abs(); }
    }
    Piece { lo: sub(c, ext), hi: add(c, ext), terms }
}

/// A light, as the families give them.
pub fn light(az: f64, el: f64, w: f64, h: f64, rgb: [f64; 3], soft: f64) -> Light {
    Light { az, el, w, h, rgb, soft }
}
/// The knots' coloured ring of lights: what fat glass tubes look best in.
pub fn tube_lights() -> Vec<Light> {
    vec![
        light(20.0, 0.0, 5.0, 40.0, [5.0, 4.9, 4.7], 0.6),
        light(80.0, 0.0, 5.0, 40.0, [5.0, 1.6, 0.25], 0.6),
        light(140.0, 0.0, 5.0, 40.0, [0.5, 1.4, 4.0], 0.6),
        light(200.0, 0.0, 5.0, 40.0, [4.0, 3.9, 3.8], 0.6),
        light(260.0, 0.0, 5.0, 40.0, [5.0, 1.6, 0.25], 0.6),
        light(320.0, 0.0, 5.0, 40.0, [0.5, 1.4, 4.0], 0.6),
        light(0.0, 85.0, 30.0, 30.0, [0.6, 0.7, 0.9], 0.9),
    ]
}
/// The trees' broad soft lights with two small ones for glints: what solids with flat faces look best in.
pub fn solid_lights() -> Vec<Light> {
    vec![
        light(40.0, 20.0, 24.0, 38.0, [3.6, 3.5, 3.3], 0.9),
        light(-58.0, -6.0, 18.0, 44.0, [0.45, 1.25, 3.6], 0.9),
        light(-140.0, 5.0, 15.0, 44.0, [4.4, 1.45, 0.24], 0.9),
        light(150.0, 29.0, 28.0, 20.0, [2.6, 2.5, 2.45], 0.9),
        light(0.0, 85.0, 38.0, 38.0, [0.8, 0.9, 1.1], 1.0),
        light(0.0, -80.0, 34.0, 34.0, [0.5, 0.22, 0.05], 1.0),
        light(95.0, 46.0, 5.0, 5.0, [6.0, 6.0, 6.0], 0.5),
        light(-100.0, -36.0, 4.0, 8.0, [4.5, 5.0, 6.0], 0.5),
    ]
}

/// Keep the last few results of a slow generator, by its sliders (all but those at `skip`): moving only
/// a slider that is skipped (such as `grow`) reuses what was worked out.
pub struct Memo<T> {
    kept: std::sync::Mutex<Vec<(Vec<u64>, std::sync::Arc<T>)>>,
}
impl<T> Memo<T> {
    pub const fn new() -> Self { Self { kept: std::sync::Mutex::new(Vec::new()) } }
    pub fn get(&self, v: &[f64], skip: &[usize], extra: &str, make: impl FnOnce() -> T) -> std::sync::Arc<T> {
        let mut key: Vec<u64> = v.iter().enumerate().filter(|(i, _)| !skip.contains(i)).map(|(_, x)| x.to_bits()).collect();
        key.extend(extra.bytes().map(|b| b as u64 | 1 << 40));
        if let Some(t) = self.kept.lock().ok().and_then(|c| c.iter().find(|(k, _)| *k == key).map(|(_, t)| t.clone())) { return t; }
        let t = std::sync::Arc::new(make());
        if let Ok(mut c) = self.kept.lock() {
            c.insert(0, (key, t.clone()));
            c.truncate(4);
        }
        t
    }
}

/// Flat closed curves (x, y), lifted into space so that at every crossing one passes over the other, the
/// passes alternating along each curve: an alternating diagram (a Celtic knot, the usual picture of a
/// link) as a real knot or link. `h` is how far a pass rises or sinks at a crossing; between crossings the
/// height eases from one to the next (level as it crosses). Each curve comes back as x, y, z, closed (its
/// last point repeating its first). Curves should be finely sampled: crossings are found between chords.
pub fn lift_alternating(curves: &[Vec<[f64; 2]>], h: f64) -> Vec<Vec<f64>> {
    // every chord, and a grid of them
    let mut segs: Vec<(usize, usize)> = Vec::new();
    let (mut lo, mut hi, mut total) = ([f64::MAX; 2], [f64::MIN; 2], 0.0);
    for (c, pts) in curves.iter().enumerate() {
        let n = pts.len();
        for i in 0..n {
            let (a, b) = (pts[i], pts[(i + 1) % n]);
            total += ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            for k in 0..2 {
                lo[k] = lo[k].min(a[k]);
                hi[k] = hi[k].max(a[k]);
            }
            segs.push((c, i));
        }
    }
    if segs.is_empty() { return Vec::new(); }
    let cell = (4.0 * total / segs.len() as f64).max(1e-9);
    let nx = (((hi[0] - lo[0]) / cell) as usize + 1).min(2048);
    let ny = (((hi[1] - lo[1]) / cell) as usize + 1).min(2048);
    let at = |x: f64, k: usize, m: usize| (((x - lo[k]) / cell) as usize).min(m - 1);
    let mut grid: Vec<Vec<u32>> = vec![Vec::new(); nx * ny];
    for (s, &(c, i)) in segs.iter().enumerate() {
        let n = curves[c].len();
        let (a, b) = (curves[c][i], curves[c][(i + 1) % n]);
        for gx in at(a[0].min(b[0]), 0, nx)..=at(a[0].max(b[0]), 0, nx) {
            for gy in at(a[1].min(b[1]), 1, ny)..=at(a[1].max(b[1]), 1, ny) {
                grid[gy * nx + gx].push(s as u32);
            }
        }
    }
    // the crossings: for each, where along each curve its two passes are (curve, index + fraction)
    let mut passes: Vec<Vec<(f64, usize)>> = vec![Vec::new(); curves.len()];
    let mut found = std::collections::HashSet::new();
    let mut crossing = 0usize;
    for list in &grid {
        for (x, &s1) in list.iter().enumerate() {
            for &s2 in &list[x + 1..] {
                let (a1, a2) = (s1.min(s2) as usize, s1.max(s2) as usize);
                if !found.insert((a1, a2)) { continue; }
                let ((c1, i1), (c2, i2)) = (segs[a1], segs[a2]);
                let (n1, n2) = (curves[c1].len(), curves[c2].len());
                // (chords that share an end do not cross)
                if c1 == c2 && (i1 + 1) % n1 == i2 || c1 == c2 && (i2 + 1) % n2 == i1 { continue; }
                let (p, p2) = (curves[c1][i1], curves[c1][(i1 + 1) % n1]);
                let (q, q2) = (curves[c2][i2], curves[c2][(i2 + 1) % n2]);
                let (r, s) = ([p2[0] - p[0], p2[1] - p[1]], [q2[0] - q[0], q2[1] - q[1]]);
                let den = r[0] * s[1] - r[1] * s[0];
                if den.abs() < 1e-18 { continue; }
                let w = [q[0] - p[0], q[1] - p[1]];
                let t = (w[0] * s[1] - w[1] * s[0]) / den;
                let u = (w[0] * r[1] - w[1] * r[0]) / den;
                if !(0.0..1.0).contains(&t) || !(0.0..1.0).contains(&u) { continue; }
                passes[c1].push((i1 as f64 + t, crossing));
                passes[c2].push((i2 as f64 + u, crossing));
                crossing += 1;
            }
        }
    }
    for p in passes.iter_mut() { p.sort_by(|a, b| a.0.total_cmp(&b.0)); }
    // over or under: alternate along each curve, each crossing's two passes opposite. A curve's first
    // sign is fixed by the first of its crossings already settled (a curve with none: over)
    let mut sign = vec![0i8; 2 * crossing];
    let mut which: Vec<Vec<i8>> = passes.iter().map(|p| vec![0; p.len()]).collect();
    let mut settled = vec![false; curves.len()];
    for _ in 0..curves.len() {
        // the next curve to settle: one with a crossing already given a sign through another curve
        let next = (0..curves.len()).filter(|&c| !settled[c]).max_by_key(|&c| passes[c].iter().any(|&(_, k)| sign[2 * k] != 0 || sign[2 * k + 1] != 0));
        let Some(c) = next else { break };
        settled[c] = true;
        // the first of its passes whose crossing has a sign
        let mut start = 1i8;
        for (j, &(_, k)) in passes[c].iter().enumerate() {
            let other = if sign[2 * k] != 0 { -sign[2 * k] } else if sign[2 * k + 1] != 0 { -sign[2 * k + 1] } else { 0 };
            if other != 0 {
                start = if j % 2 == 0 { other } else { -other };
                break;
            }
        }
        for (j, &(_, k)) in passes[c].iter().enumerate() {
            let sgn = if j % 2 == 0 { start } else { -start };
            which[c][j] = sgn;
            if sign[2 * k] == 0 { sign[2 * k] = sgn } else { sign[2 * k + 1] = sgn }
        }
    }
    // heights: eased from pass to pass along the curve, by arc length
    curves
        .iter()
        .enumerate()
        .map(|(c, pts)| {
            let n = pts.len();
            let mut s = vec![0.0; n + 1];
            for i in 0..n {
                let (a, b) = (pts[i], pts[(i + 1) % n]);
                s[i + 1] = s[i] + ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            }
            let length = s[n];
            let along = |f: f64| {
                let i = (f.floor() as usize).min(n - 1);
                s[i] + (s[i + 1] - s[i]) * (f - i as f64)
            };
            let marks: Vec<(f64, f64)> = passes[c].iter().zip(&which[c]).map(|(&(f, _), &w)| (along(f), w as f64 * h)).collect();
            let mut out = Vec::with_capacity(3 * (n + 1));
            for i in 0..=n {
                let p = pts[i % n];
                let z = if marks.is_empty() {
                    0.0
                } else {
                    let x = s[i];
                    // the passes either side of x (round the loop)
                    let j = marks.partition_point(|m| m.0 <= x);
                    let (a, b) = if j == 0 {
                        (( marks[marks.len() - 1].0 - length, marks[marks.len() - 1].1), marks[0])
                    } else if j == marks.len() {
                        (marks[j - 1], (marks[0].0 + length, marks[0].1))
                    } else {
                        (marks[j - 1], marks[j])
                    };
                    let u = ((x - a.0) / (b.0 - a.0).max(1e-12)).clamp(0.0, 1.0);
                    let e = 0.5 - 0.5 * (std::f64::consts::PI * u).cos();
                    a.1 + (b.1 - a.1) * e
                };
                out.extend_from_slice(&[p[0], p[1], z]);
            }
            out
        })
        .collect()
}
