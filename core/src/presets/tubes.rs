//! Analytic swept tubes and leaves for generated curves.

use super::exact::Written;

/// Maximum fitted error at a source sample, in object space.
pub const FIT_TOLERANCE: f64 = 0.00035;

/// Encode the generated centre lines after placement and animation. Each cubic is a
/// mathematical tube, with continuous tangent and curvature where two pieces meet. The
/// simplification tolerance is in scene units and is independent of the distance volume resolution.
pub fn table(curve: &crate::curve::Curve, melt: f64, tree: bool, anim: bool, res: u32) -> Result<Vec<[f32; 4]>, String> {
    let point = |i: usize| -> [f32; 4] { curve.pts[4 * i..4 * i + 4].try_into().unwrap() };
    let mut pieces = Vec::new();
    let mut open_strand = 0usize;
    let mut closed_strand = 0usize;
    let mut i = 0;
    while i < curve.n {
        let a = point(i);
        let count = crate::curve::piece_len(&curve.pts, i);
        if count > 0 {
            // The dipole can include a solid core. Other curve families produce
            // centre lines and leaves. Paths have their own exact evaluator.
            let half = point(i + 1);
            if count != 4 || point(i + 2)[0] != 1.0 {
                return Err("the curve contains an unsupported solid primitive".into());
            }
            pieces.push(Written {
                lo: std::array::from_fn(|k| f64::from(a[k] - half[k])),
                hi: std::array::from_fn(|k| f64::from(a[k] + half[k])),
                e: vec![[101.0, 0.0, 0.0, 0.0], point(i + 3)],
                meta: 0.0,
            });
            i += count;
        } else if a[3] < 0.0 {
            let (tip, side) = (point(i + 1), point(i + 2));
            // The cup raises the rim by at most |cup| times the leaf width.
            let extent = tip[3] * (1.0 + side[3].abs()) - a[3];
            pieces.push(Written {
                lo: std::array::from_fn(|k| f64::from(a[k].min(tip[k]) - extent)),
                hi: std::array::from_fn(|k| f64::from(a[k].max(tip[k]) + extent)),
                e: vec![[108.0, 0.0, 0.0, 0.0], a, tip, side],
                meta: 0.0,
            });
            i += 3;
        } else {
            let start = i;
            while i + 1 < curve.n && curve.brk[i] == 0 {
                i += 1;
            }
            let points: Vec<[f64; 4]> = (start..=i).map(|j| point(j).map(f64::from)).collect();
            add_tube_strand(&mut pieces, &points, FIT_TOLERANCE, &mut open_strand, &mut closed_strand);
            i += 1;
        }
    }
    // Hundreds of short twigs share cells at the scale of a tree's crown. A coarser
    // spatial index keeps the growth animation's rebuild and upload small.
    // A playing animation rebuilds the index every frame: a coarser one there halves its making
    // (about 2 ms a frame for an attractor) while the tracer reads it no slower.
    let density = if tree { 2.5 } else if anim { 3.5 } else { 4.5 };
    let table = super::exact::write_tubes(pieces, curve.cut, melt, clear(melt, anim, res), density);
    if table.len() > crate::field::TABLE_MAX {
        return Err(format!(
            "curve needs {} formula entries; the GPU holds {}",
            table.len(),
            crate::field::TABLE_MAX
        ));
    }
    Ok(table)
}

/// How many melts the union's distance is held short where a melt is on: the melt's surface lies outside
/// the union's by melt ln(n) where n strands run together, and the far field must not be over the way to
/// it. Two melts: seven strands' worth (the shapes measured come to 1.9 at most).
pub const HOLD: f64 = 2.0;

/// How far outside its own surface a span's part of the melt is whole (it then fades to nothing by
/// `clear / 0.65`). A bundle of n strands shares a surface about melt ln(n) outside each strand's own,
/// and the volume's smooth reading takes in the voxels either side of it: 2.2 melts, and 2.5 voxels of the
/// volume being built (1.5 while an animation plays: its index is made afresh every frame, and what a cell
/// lists grows with this).
pub fn clear(melt: f64, anim: bool, res: u32) -> f64 {
    if melt <= 0.0 { return 0.0; }
    2.2 * melt + if anim { 1.5 } else { 2.5 } * 2.0 * crate::HB / f64::from(res.max(1))
}

/// `tubes.wgsl` on the CPU, operation for operation, in f64 (the tests measure the formula with it, and a
/// GPU test holds the shader to it).
pub mod cpu {
    type V3 = [f64; 3];
    type V4 = [f64; 4];

    fn sub(a: V3, b: V3) -> V3 { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
    fn dot(a: V3, b: V3) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
    fn len(a: V3) -> f64 { dot(a, a).sqrt() }
    fn xyz(a: V4) -> V3 { [a[0], a[1], a[2]] }
    fn row(t: &[[f32; 4]], i: usize) -> V4 { t[i].map(f64::from) }

    /// A span as its polynomial: C(t) = a + t (e + t (f + t g)); d its end.
    #[derive(Clone, Copy)]
    struct Span { a: V4, e: V4, f: V4, g: V4, d: V4 }

    impl Span {
        fn cubic(a: V4, b: V4, c: V4, d: V4) -> Self {
            Self {
                a,
                e: std::array::from_fn(|k| 3.0 * (b[k] - a[k])),
                f: std::array::from_fn(|k| 3.0 * (a[k] - 2.0 * b[k] + c[k])),
                g: std::array::from_fn(|k| d[k] - a[k] + 3.0 * (b[k] - c[k])),
                d,
            }
        }
        fn at(&self, t: f64) -> V4 { std::array::from_fn(|k| self.a[k] + t * (self.e[k] + t * (self.f[k] + t * self.g[k]))) }
        fn rate(&self, t: f64) -> V4 { std::array::from_fn(|k| self.e[k] + t * (2.0 * self.f[k] + 3.0 * t * self.g[k])) }
    }

    fn value(p: V3, c: V4) -> f64 { len(sub(p, xyz(c))) - c[3].max(0.0) }

    /// `tube_foot`.
    fn foot(p: V3, s: &Span) -> f64 {
        let chord = sub(xyz(s.d), xyz(s.a));
        let l2 = dot(chord, chord).max(1e-20);
        let l = l2.sqrt();
        let u_dir = chord.map(|v| v / l);
        let pa = sub(p, xyz(s.a));
        let x = dot(pa, u_dir);
        let y = len(std::array::from_fn(|k| pa[k] - x * u_dir[k]));
        let dr = s.d[3] - s.a[3];
        let mut t = ((x + y * dr / (l2 - dr * dr).max(0.04 * l2).sqrt()) / l).clamp(0.0, 1.0);
        let mut best_c = s.at(t);
        let mut best_f = value(p, best_c);
        let f0 = len(pa) - s.a[3].max(0.0);
        if f0 < best_f { best_f = f0; t = 0.0; best_c = s.a; }
        let mid = s.at(0.5);
        let fm = value(p, mid);
        if fm < best_f { best_f = fm; t = 0.5; best_c = mid; }
        let f1 = value(p, s.d);
        if f1 < best_f { best_f = f1; t = 1.0; best_c = s.d; }
        for _ in 0..5 {
            let delta = sub(xyz(best_c), p);
            let dist = len(delta);
            if dist < 1e-10 { break; }
            let speed = s.rate(t);
            let accel: V4 = std::array::from_fn(|k| 2.0 * s.f[k] + 6.0 * t * s.g[k]);
            let vt = dot(delta, xyz(speed));
            let fp = vt / dist - speed[3];
            if fp.abs() < 1e-6 || (t <= 0.0 && fp >= 0.0) || (t >= 1.0 && fp <= 0.0) { break; }
            let fpp = (dot(xyz(speed), xyz(speed)) + dot(delta, xyz(accel)) - (vt * vt) / (dist * dist)) / dist - accel[3];
            let min_curv = (dot(xyz(speed), xyz(speed)) / dist * 0.1).max(1e-6);
            let step = (fp / fpp.max(min_curv)).clamp(-0.35, 0.35);
            let t1 = (t - step).clamp(0.0, 1.0);
            let c1 = s.at(t1);
            let v1 = value(p, c1);
            if v1 <= best_f {
                t = t1;
                best_f = v1;
                best_c = c1;
            } else {
                let t2 = 0.5 * (t + t1);
                let c2 = s.at(t2);
                let v2 = value(p, c2);
                if v2 < best_f {
                    t = t2;
                    best_f = v2;
                    best_c = c2;
                } else {
                    break;
                }
            }
            if step.abs() < 1e-6 { break; }
        }
        t
    }

    /// `tube_reading`.
    fn reading(p: V3, s: &Span, t: f64, c: V4) -> V4 {
        let offset = sub(p, xyz(c));
        let dist = len(offset);
        let n = if dist > 1e-12 {
            offset.map(|v| v / dist)
        } else {
            let tangent = xyz(s.rate(t));
            let seed = if tangent[0].abs() > 0.7 * len(tangent).max(1e-12) { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
            let c = [tangent[1] * seed[2] - tangent[2] * seed[1], tangent[2] * seed[0] - tangent[0] * seed[2], tangent[0] * seed[1] - tangent[1] * seed[0]];
            let l = len(c);
            c.map(|v| v / l)
        };
        [n[0], n[1], n[2], dist - c[3].max(0.0)]
    }

    /// `tube_strand`: z e^z K1(z), to a third of a percent.
    pub fn strand(z: f64) -> f64 { (1.0 + std::f64::consts::FRAC_PI_2 * z + 0.19 * z / (z + 0.8)).sqrt() }

    /// `tube_fade`'s first number: how much of a span's part of the melt is left, x outside its surface.
    pub fn fade(x: f64, clear: f64) -> f64 { fade2(x, clear, 0.0)[0] }

    /// `tube_fade`.
    fn fade2(x: f64, clear: f64, m: f64) -> [f64; 2] {
        let width = 0.5384615 * clear;
        let u = ((x - clear) / width).clamp(0.0, 1.0);
        let fade = 1.0 - u * u * u * (10.0 - u * (15.0 - 6.0 * u));
        [fade, fade + m * 30.0 * u * u * (1.0 - u) * (1.0 - u) / width]
    }

    const GL: [[f64; 2]; 4] = [[0.1834346425, 0.3626837834], [0.5255324099, 0.3137066459], [0.7966664774, 0.2223810345], [0.9602898565, 0.1012285363]];

    /// `tube_melt`.
    fn melt_part(p: V3, s: &Span, t0: f64, x0: f64, rho0: f64, k: f64, clear: f64) -> V4 {
        let m = 1.0 / k;
        let speed = len(xyz(s.rate(t0))).max(1e-9);
        let rho = rho0.max(0.25 * m);
        let along = (12.0 * m * (3.0 * m + rho0)).sqrt();
        let lo = -((t0 * speed).min(along) / rho).asinh();
        let hi = (((1.0 - t0) * speed).min(along) / rho).asinh();
        let mid = 0.5 * (lo + hi);
        let half = 0.5 * (hi - lo);
        let w0 = 0.5 * k / strand(k * s.a[3].max(0.0));
        let w1 = 0.5 * k / strand(k * s.d[3].max(0.0));
        let mut sum = [0.0; 4];
        for j in 0..8 {
            let node = GL[j >> 1];
            let grow = (mid + half * if j & 1 == 1 { node[0] } else { -node[0] }).exp();
            let t = (t0 + 0.5 * rho * (grow - 1.0 / grow) / speed).clamp(0.0, 1.0);
            let c = s.at(t);
            let offset = sub(p, xyz(c));
            let dist = len(offset);
            let x = dist - c[3].max(0.0);
            let w = node[1] * half * 0.5 * rho * (grow + 1.0 / grow) / speed * len(xyz(s.rate(t))) * (w0 + (w1 - w0) * t) * (-k * (x - x0)).exp();
            let fade = fade2(x, clear, m);
            for i in 0..3 { sum[i] += w * fade[1] * offset[i] / dist.max(1e-12); }
            sum[3] += w * fade[0];
        }
        sum
    }

    /// `pc_blade`.
    fn blade(p: V3, a: V4, c: V4, s: V4) -> f64 {
        let ax = sub(xyz(c), xyz(a));
        let l = len(ax).max(1e-6);
        let x = ax.map(|v| v / l);
        let sa = sub(xyz(s), xyz(a));
        let along = dot(sa, x);
        let y0: V3 = std::array::from_fn(|k| sa[k] - x[k] * along);
        let yl = len(y0);
        let y = y0.map(|v| v / yl);
        let z = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
        let q = sub(p, xyz(a));
        let wd = c[3];
        let u = dot(q, x) / l;
        let v = dot(q, y);
        let w = dot(q, z) - s[3] * v * v / wd.max(1e-6);
        let uc = u.clamp(0.02, 1.0);
        let k = wd / 0.385;
        let hw = k * uc.sqrt() * (1.0 - uc);
        let slope = k * (0.5 / uc.sqrt() * (1.0 - uc) - uc.sqrt()) / l;
        let side = (v.abs() - hw) / (1.0 + slope * slope).sqrt();
        let ends = (-u).max(u - 1.0) * l;
        let d2 = side.max(ends);
        let e = [d2, w.abs()];
        (e[0].max(0.0).powi(2) + e[1].max(0.0).powi(2)).sqrt() + e[0].max(e[1]).min(0.0) + a[3]
    }

    fn blade_slope(p: V3, a: V4, b: V4, c: V4) -> V3 {
        let eps = 0.0001;
        let mut g = [0.0; 3];
        for k in 0..4 {
            let tap = [if k == 0 || k == 3 { 1.0 } else { -1.0 }, if k >= 2 { 1.0 } else { -1.0 }, if k == 1 || k == 3 { 1.0 } else { -1.0 }];
            let v = blade([p[0] + eps * tap[0], p[1] + eps * tap[1], p[2] + eps * tap[2]], a, b, c);
            for i in 0..3 { g[i] += tap[i] * v; }
        }
        let l = len(g);
        g.map(|v| v / l)
    }

    /// `tube_bound`.
    fn bound(p: V3, s: &Span, radius: f64) -> f64 {
        let chord = sub(xyz(s.d), xyz(s.a));
        let pa = sub(p, xyz(s.a));
        let t = (dot(pa, chord) / dot(chord, chord).max(1e-20)).clamp(0.0, 1.0);
        len(std::array::from_fn(|k| pa[k] - t * chord[k])) - radius
    }

    /// `tube_far`.
    fn far(t: &[[f32; 4]], q: V3, grid: V4, n: usize) -> V4 {
        let top = n as f64 - 1.0;
        let coord: V3 = std::array::from_fn(|k| (q[k] - grid[k]) / grid[3] - 0.5);
        let c: V3 = std::array::from_fn(|k| coord[k].clamp(0.0, top));
        let i0: [usize; 3] = std::array::from_fn(|k| c[k].floor().min((top - 1.0).max(0.0)) as usize);
        let i1: [usize; 3] = std::array::from_fn(|k| (i0[k] + 1).min(n - 1));
        let f: V3 = std::array::from_fn(|k| c[k] - i0[k] as f64);
        let v = |x: usize, y: usize, z: usize| { let d = f64::from(t[3 + (z * n + y) * n + x][2]).max(0.0); d * d };
        let mix = |a: f64, b: f64, w: f64| a + (b - a) * w;
        let (v000, v100, v010, v110) = (v(i0[0], i0[1], i0[2]), v(i1[0], i0[1], i0[2]), v(i0[0], i1[1], i0[2]), v(i1[0], i1[1], i0[2]));
        let (v001, v101, v011, v111) = (v(i0[0], i0[1], i1[2]), v(i1[0], i0[1], i1[2]), v(i0[0], i1[1], i1[2]), v(i1[0], i1[1], i1[2]));
        let (x00, x10, x01, x11) = (mix(v000, v100, f[0]), mix(v010, v110, f[0]), mix(v001, v101, f[0]), mix(v011, v111, f[0]));
        let (y0, y1) = (mix(x00, x10, f[1]), mix(x01, x11, f[1]));
        let gx = mix(mix(v100 - v000, v110 - v010, f[1]), mix(v101 - v001, v111 - v011, f[1]), f[2]);
        let gy = mix(x10 - x00, x11 - x01, f[2]);
        let g = [gx, gy, y1 - y0];
        let bow: f64 = (0..3).map(|k| f[k] * (1.0 - f[k])).sum();
        let inside = (mix(y0, y1, f[2]) - grid[3] * grid[3] * bow).max(0.0);
        let out: V3 = std::array::from_fn(|k| (coord[k] - c[k]) * grid[3]);
        let half: V3 = std::array::from_fn(|k| if out[k] != 0.0 { out[k] } else { 0.5 * (g[k] / grid[3] - grid[3] * (1.0 - 2.0 * f[k])) });
        let value = (inside + dot(out, out)).sqrt();
        let by = value.max(1e-9);
        [half[0] / by, half[1] / by, half[2] / by, value]
    }

    /// What `eval` works out on the way, for the tests.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Parts {
        /// the least distance over the cell's listed pieces (1e9: none read)
        pub near: f64,
        /// the melt's distance (1e9: no part)
        pub melted: f64,
        /// the far field's value, less its safety (1e9: not read)
        pub far: f64,
        /// pieces read in full
        pub read: u32,
    }

    /// `tube_eval`: the slope (xyz) and the distance at p.
    pub fn eval(t: &[[f32; 4]], p: V3) -> V4 { eval_parts(t, p, None).0 }

    /// The same, read through the list of the cell `force` (to compare the lists of neighbouring cells).
    pub fn eval_parts(t: &[[f32; 4]], p: V3, force: Option<[usize; 3]>) -> (V4, Parts) {
        let (header, grid, sizes) = (row(t, 0), row(t, 1), row(t, 2));
        let n = sizes[0] as usize;
        let q: V3 = std::array::from_fn(|k| p[k].clamp(grid[k], grid[k] + n as f64 * grid[3]));
        let coord: V3 = std::array::from_fn(|k| (q[k] - grid[k]) / grid[3]);
        let ci: [usize; 3] = force.unwrap_or(std::array::from_fn(|k| coord[k].clamp(0.0, n as f64 - 1.0) as usize));
        let cell = row(t, 3 + (ci[2] * n + ci[1]) * n + ci[0]);
        let count = cell[1] as usize;
        let first = (3 + n * n * n) * 4 + cell[0] as usize;
        let melt = header[3];
        let reach = sizes[1];
        let mut near: V4 = [0.0, 0.0, 0.0, 1e9];
        let mut parts: V4 = [0.0; 4];
        let mut ref_x: f64 = 1e9;
        let k = 1.0 / melt.max(1e-9);
        let clear = sizes[2];
        let band = clear / 0.65;
        let mut info = Parts { near: 1e9, melted: 1e9, far: 1e9, read: 0 };
        let add = |parts: &mut V4, ref_x: &mut f64, part: V4, x: f64| {
            if x < *ref_x {
                let scale = (-k * (*ref_x - x)).exp();
                for i in 0..4 { parts[i] = parts[i] * scale + part[i]; }
                *ref_x = x;
            } else {
                let scale = (-k * (x - *ref_x)).exp();
                for i in 0..4 { parts[i] += part[i] * scale; }
            }
        };
        for j in 0..count {
            let at = first + j;
            let i = t[at / 4][at % 4] as usize;
            let mut limit = near[3].min(reach);
            if melt > 0.0 { limit = limit.max(band.min(ref_x + 16.0 * melt)); }
            let (bc, bh) = (row(t, i), row(t, i + 1));
            let from_box = (0..3).map(|a| (q[a] - bc[a]).abs() - bh[a]).fold(f64::MIN, f64::max);
            if from_box >= limit { continue; }
            let tag = row(t, i + 2);
            let kind = tag[0];
            let mut e: V4 = [0.0, 0.0, 0.0, 1e9];
            if kind == 110.0 || kind == 111.0 {
                let (a, b) = (row(t, i + 3), row(t, i + 4));
                let s = if kind == 111.0 {
                    Span { a, e: std::array::from_fn(|c| b[c] - a[c]), f: [0.0; 4], g: [0.0; 4], d: b }
                } else {
                    Span::cubic(a, b, row(t, i + 5), row(t, i + 6))
                };
                if bound(q, &s, tag[3]) >= limit { continue; }
                info.read += 1;
                let tt = foot(q, &s);
                let c = s.at(tt);
                e = reading(q, &s, tt, c);
                if melt > 0.0 && e[3] < band {
                    let part = melt_part(q, &s, tt, e[3], len(sub(q, xyz(c))), k, clear);
                    add(&mut parts, &mut ref_x, part, e[3]);
                }
            } else {
                let a = row(t, i + 3);
                if kind == 101.0 {
                    let d = sub(q, xyz(a));
                    let l = len(d);
                    let n = if l > 1e-12 { d.map(|v| v / l) } else { [0.0, 1.0, 0.0] };
                    e = [n[0], n[1], n[2], l - a[3]];
                } else {
                    let (b, c) = (row(t, i + 4), row(t, i + 5));
                    e[3] = blade(q, a, b, c);
                    if e[3] < limit || (melt > 0.0 && e[3] < band) {
                        let g = blade_slope(q, a, b, c);
                        e = [g[0], g[1], g[2], e[3]];
                    }
                }
                info.read += 1;
                if melt > 0.0 && e[3] < band {
                    let w = fade2(e[3], clear, melt);
                    add(&mut parts, &mut ref_x, [e[0] * w[1], e[1] * w[1], e[2] * w[1], w[0]], e[3]);
                }
            }
            if e[3] < near[3] { near = e; }
        }
        info.near = near[3];
        let near_clear = 0.65 * reach;
        let mut result: V4 = [0.0, 1.0, 0.0, 1e9];
        if near[3] < reach {
            let u = ((near[3] - near_clear) / (reach - near_clear)).clamp(0.0, 1.0);
            let keep = 1.0 - u * u * u * (10.0 - u * (15.0 - 6.0 * u));
            result = [near[0], near[1], near[2], near[3] / keep.max(1e-6)];
        }
        if result[3] > near_clear {
            let fv = far(t, q, grid, n);
            let value = fv[3].max(near_clear);
            info.far = value;
            if value < result[3] {
                let l = len(xyz(fv));
                result = if l > 1e-12 { [fv[0] / l, fv[1] / l, fv[2] / l, value] } else { [0.0, 1.0, 0.0, value] };
            }
        }
        if melt > 0.0 {
            result[3] = (result[3] - super::HOLD * melt).max(clear - super::HOLD * melt);
            if parts[3] > 0.0 {
                let melted = ref_x - parts[3].ln() / k;
                info.melted = melted;
                if melted < result[3] { result = [parts[0] / parts[3], parts[1] / parts[3], parts[2] / parts[3], melted]; }
            }
        }
        let out = sub(p, q);
        let outside = dot(out, out);
        if outside > 0.0 {
            let inner = result[3].max(0.0);
            let d = (outside + inner * inner).sqrt();
            result = [(out[0] + inner * result[0]) / d, (out[1] + inner * result[1]) / d, (out[2] + inner * result[2]) / d, d];
        }
        if header[2] > 0.5 && header[1] - p[1] > result[3] { result = [0.0, -1.0, 0.0, header[1] - p[1]]; }
        if dot(xyz(result), xyz(result)) < 1e-24 { result = [0.0, 1.0, 0.0, result[3]]; }
        (result, info)
    }
}

/// Solve an ordinary tridiagonal system for four independent coordinates.
fn tridiagonal(lower: &[f64], diagonal: &[f64], upper: &[f64], rhs: &[[f64; 4]]) -> Vec<[f64; 4]> {
    let n = rhs.len();
    let mut c = vec![0.0; n];
    let mut d = vec![[0.0; 4]; n];
    let mut denom = diagonal[0];
    c[0] = upper[0] / denom;
    d[0] = rhs[0].map(|v| v / denom);
    for i in 1..n {
        denom = diagonal[i] - lower[i] * c[i - 1];
        c[i] = upper[i] / denom;
        d[i] = std::array::from_fn(|k| (rhs[i][k] - lower[i] * d[i - 1][k]) / denom);
    }
    for i in (0..n - 1).rev() {
        d[i] = std::array::from_fn(|k| d[i][k] - c[i] * d[i + 1][k]);
    }
    d
}

/// First derivatives of the C2 cubic spline through the selected knots. Closed loops use a periodic
/// spline; open strands use natural end conditions. Radius is splined with position, so the complete
/// swept surface, not just its centre line, has continuous curvature at every fitted joint.
fn spline_tangents(
    points: &[[f64; 4]],
    along: &[f64],
    knots: &[usize],
    closed: bool,
) -> Vec<[f64; 4]> {
    let count = if closed { knots.len() - 1 } else { knots.len() };
    let spans: Vec<f64> = knots
        .windows(2)
        .map(|w| (along[w[1]] - along[w[0]]).max(1e-12))
        .collect();
    let p = |i: usize| points[knots[i]];
    let mut lower = vec![0.0; count];
    let mut diagonal = vec![0.0; count];
    let mut upper = vec![0.0; count];
    let mut rhs = vec![[0.0; 4]; count];
    if closed {
        for i in 0..count {
            let previous = (i + count - 1) % count;
            let next = i + 1;
            let h0 = spans[previous];
            let h1 = spans[i];
            lower[i] = h1;
            upper[i] = h0;
            diagonal[i] = 2.0 * (h0 + h1);
            rhs[i] = std::array::from_fn(|k| {
                3.0 * (h1 * (p(i)[k] - p(previous)[k]) / h0 + h0 * (p(next)[k] - p(i)[k]) / h1)
            });
        }
        // This cyclic matrix is strictly diagonally dominant: each diagonal is twice the sum of
        // its off-diagonal magnitudes. Gauss-Seidel therefore converges reliably, including at the
        // two wrap-around entries, without a fragile cyclic rank update.
        let mut out = vec![[0.0; 4]; count];
        for _ in 0..80 {
            for i in 0..count {
                let previous = (i + count - 1) % count;
                let next = (i + 1) % count;
                out[i] = std::array::from_fn(|k| {
                    (rhs[i][k] - lower[i] * out[previous][k] - upper[i] * out[next][k])
                        / diagonal[i]
                });
            }
        }
        out.push(out[0]);
        out
    } else {
        for i in 0..count {
            if i == 0 || i == count - 1 {
                let segment = if i == 0 { 0 } else { count - 2 };
                let next = if i == 0 { 1 } else { count - 1 };
                lower[i] = f64::from(i != 0);
                diagonal[i] = 2.0;
                upper[i] = f64::from(i == 0);
                rhs[i] =
                    std::array::from_fn(|k| 3.0 * (p(next)[k] - p(segment)[k]) / spans[segment]);
            } else {
                let h0 = spans[i - 1];
                let h1 = spans[i];
                lower[i] = h1;
                upper[i] = h0;
                diagonal[i] = 2.0 * (h0 + h1);
                rhs[i] = std::array::from_fn(|k| {
                    3.0 * (h1 * (p(i)[k] - p(i - 1)[k]) / h0 + h0 * (p(i + 1)[k] - p(i)[k]) / h1)
                });
            }
        }
        tridiagonal(&lower, &diagonal, &upper, &rhs)
    }
}

/// Approximate the generated path with C2 cubic intervals. Adjacent intervals share their end point,
/// tangent and second derivative, including across a closed loop. Every original sample is checked in
/// position and radius; changing the spline knots to remove an error leaves all of those joins intact.
#[cfg(test)]
fn add_tube_line(pieces: &mut Vec<Written>, points: &[[f64; 4]], tolerance: f64) {
    let (mut open_strand, mut closed_strand) = (0usize, 0usize);
    add_tube_strand(pieces, points, tolerance, &mut open_strand, &mut closed_strand);
}

fn add_tube_strand(
    pieces: &mut Vec<Written>,
    points: &[[f64; 4]],
    tolerance: f64,
    open_strand: &mut usize,
    closed_strand: &mut usize,
) {
    if points.len() < 2 {
        return;
    }
    let last = points.len() - 1;
    let len =
        |a: &[f64; 4], b: &[f64; 4]| (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt();
    let mut along = Vec::with_capacity(points.len());
    along.push(0.0);
    for pair in points.windows(2) {
        along.push(along.last().unwrap() + len(&pair[0], &pair[1]));
    }
    let closed = last > 2 && len(&points[0], &points[last]) < 1e-6 && along[last] > 1e-6;
    let mut ranges = if closed {
        vec![(0, last / 2), (last / 2, last)]
    } else {
        vec![(0, last)]
    };
    loop {
        ranges.retain(|(lo, hi)| along[*hi] - along[*lo] >= 1e-10);
        ranges.sort_unstable();
        if ranges.is_empty() {
            return;
        }
        let knots: Vec<usize> = std::iter::once(ranges[0].0)
            .chain(ranges.iter().map(|range| range.1))
            .collect();
        let tangents = spline_tangents(points, &along, &knots, closed);
        let mut split_any = false;
        let mut next = Vec::with_capacity(ranges.len());
        for (j, &(lo, hi)) in ranges.iter().enumerate() {
            let span = along[hi] - along[lo];
            let (a, d) = (points[lo], points[hi]);
            let b: [f64; 4] = std::array::from_fn(|k| a[k] + span * tangents[j][k] / 3.0);
            let c: [f64; 4] = std::array::from_fn(|k| d[k] - span * tangents[j + 1][k] / 3.0);
            // Split tightly bent intervals before fitting so every cubic turns gently enough
            // for global sphere-envelope minimization.
            let mut split = if hi > lo + 1 && len(&a, &d) < 0.985 * span {
                (lo + hi) / 2
            } else {
                0
            };
            if split == 0 && hi > lo + 1 {
                let mut worst = tolerance * tolerance;
                for i in lo + 1..hi {
                    let t = (along[i] - along[lo]) / span;
                    let u = 1.0 - t;
                    let (w0, w1, w2, w3) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
                    let err = (0..4)
                        .map(|k| {
                            (points[i][k] - (w0 * a[k] + w1 * b[k] + w2 * c[k] + w3 * d[k])).powi(2)
                        })
                        .sum::<f64>();
                    if err > worst {
                        worst = err;
                        split = i;
                    }
                }
            }
            if split == 0 {
                next.push((lo, hi));
            } else {
                split_any = true;
                next.push((lo, split));
                next.push((split, hi));
            }
        }
        ranges = next;
        if !split_any {
            break;
        }
    }
    let knots: Vec<usize> = std::iter::once(ranges[0].0)
        .chain(ranges.iter().map(|range| range.1))
        .collect();
    let tangents = spline_tangents(points, &along, &knots, closed);
    let (strand_base, piece_meta) = if closed {
        let base = (*closed_strand as f64) * 4096.0;
        *closed_strand += 1;
        (base, along[last] as f32)
    } else {
        let id = *open_strand;
        *open_strand += 1;
        (0.0, -((id + 1) as f32))
    };
    for (j, &(lo, hi)) in ranges.iter().enumerate() {
        let span = along[hi] - along[lo];
        let (a, d) = (points[lo], points[hi]);
        let b: [f64; 4] = std::array::from_fn(|k| a[k] + span * tangents[j][k] / 3.0);
        let c: [f64; 4] = std::array::from_fn(|k| d[k] - span * tangents[j + 1][k] / 3.0);
        let radius = a[3].max(b[3]).max(c[3]).max(d[3]);
        let cv = |v: [f64; 4]| v.map(|x| x as f32);
        // A cubic lies inside the convex hull of its controls. The capsule around the chord that
        // contains all four controls therefore bounds the entire tube.
        let chord: [f64; 3] = std::array::from_fn(|k| d[k] - a[k]);
        let length2 = chord.iter().map(|v| v * v).sum::<f64>().max(1e-20);
        let deviation = |p: &[f64; 4]| {
            let t =
                ((0..3).map(|k| (p[k] - a[k]) * chord[k]).sum::<f64>() / length2).clamp(0.0, 1.0);
            (0..3)
                .map(|k| (p[k] - a[k] - t * chord[k]).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let bound = (radius + deviation(&b).max(deviation(&c)) + 2e-7) as f32;
        let s0 = (strand_base + along[lo]) as f32;
        let s1 = (strand_base + along[hi]) as f32;
        let ends = [110.0, s0, s1, bound];
        let linear = (0..4).all(|k| {
            (b[k] - (2.0 * a[k] + d[k]) / 3.0).abs() < 1e-7
                && (c[k] - (a[k] + 2.0 * d[k]) / 3.0).abs() < 1e-7
        });
        let record = if linear {
            vec![[111.0, s0, s1, bound], cv(a), cv(d)]
        } else {
            vec![ends, cv(a), cv(b), cv(c), cv(d)]
        };
        pieces.push(Written {
            lo: std::array::from_fn(|k| a[k].min(b[k]).min(c[k]).min(d[k]) - radius),
            hi: std::array::from_fn(|k| a[k].max(b[k]).max(c[k]).max(d[k]) + radius),
            e: record,
            meta: piece_meta,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_spline_satisfies_every_cyclic_row() {
        let count = 5;
        let points: Vec<[f64; 4]> = (0..=count)
            .map(|i| {
                let angle = std::f64::consts::TAU * i as f64 / count as f64;
                [
                    angle.cos(),
                    angle.sin(),
                    0.2 * (2.0 * angle).cos(),
                    0.04 + 0.01 * angle.sin(),
                ]
            })
            .collect();
        let mut along = vec![0.0];
        for pair in points.windows(2) {
            along.push(
                along.last().unwrap()
                    + (0..3)
                        .map(|k| (pair[0][k] - pair[1][k]).powi(2))
                        .sum::<f64>()
                        .sqrt(),
            );
        }
        let knots: Vec<usize> = (0..=count).collect();
        let d = spline_tangents(&points, &along, &knots, true);
        for i in 0..count {
            let previous = (i + count - 1) % count;
            let h0 = along[previous + 1] - along[previous];
            let h1 = along[i + 1] - along[i];
            for k in 0..4 {
                let lhs = h1 * d[previous][k] + 2.0 * (h0 + h1) * d[i][k] + h0 * d[i + 1][k];
                let rhs = 3.0
                    * (h1 * (points[i][k] - points[previous][k]) / h0
                        + h0 * (points[i + 1][k] - points[i][k]) / h1);
                assert!(
                    (lhs - rhs).abs() < 1e-9,
                    "periodic row {i}, coordinate {k}: {lhs} != {rhs}"
                );
            }
        }
    }

    #[test]
    fn a_closed_ring_has_continuous_curvature_and_preserves_its_radius() {
        let count = 720;
        let points: Vec<_> = (0..=count)
            .map(|i| {
                let angle = std::f64::consts::TAU * i as f64 / count as f64;
                [
                    angle.cos(),
                    angle.sin(),
                    0.0,
                    0.04 + 0.01 * (3.0 * angle).cos(),
                ]
            })
            .collect();
        let mut pieces = Vec::new();
        let tolerance = 0.00035;
        add_tube_line(&mut pieces, &points, tolerance);
        assert!(pieces.len() < count / 8);
        let step = (0..3)
            .map(|k| (points[1][k] - points[0][k]).powi(2))
            .sum::<f64>()
            .sqrt();
        let knots: Vec<usize> = pieces
            .iter()
            .map(|piece| {
                let p = piece.e[1];
                (0..count)
                    .min_by(|&a, &b| {
                        let distance = |i: usize| {
                            (0..3)
                                .map(|k| (points[i][k] - f64::from(p[k])).powi(2))
                                .sum::<f64>()
                        };
                        distance(a).partial_cmp(&distance(b)).unwrap()
                    })
                    .unwrap()
            })
            .collect();
        for i in 0..pieces.len() {
            let points = &pieces[i].e;
            let next = &pieces[(i + 1) % pieces.len()].e;
            let tangent = |a: [f32; 4], b: [f32; 4]| {
                let delta: [f64; 3] = std::array::from_fn(|k| f64::from(b[k] - a[k]));
                let length = delta.iter().map(|k| k * k).sum::<f64>().sqrt();
                delta.map(|k| k / length)
            };
            let (a, b) = (tangent(points[3], points[4]), tangent(next[1], next[2]));
            let dot = (0..3).map(|k| a[k] * b[k]).sum::<f64>();
            assert!(dot > 0.99999, "ring tangent changed at interval {i}: {dot}");
            let second = |piece: &[[f32; 4]], end: bool| {
                let (a, b, c) = if end { (1, 2, 3) } else { (0, 1, 2) };
                std::array::from_fn(|k| {
                    6.0 * f64::from(piece[a + 1][k]) - 12.0 * f64::from(piece[b + 1][k])
                        + 6.0 * f64::from(piece[c + 1][k])
                })
            };
            let left_intervals = knots[(i + 1) % pieces.len()]
                + count * usize::from(i + 1 == pieces.len())
                - knots[i];
            let next_index = (i + 1) % pieces.len();
            let right_intervals = knots[(next_index + 1) % pieces.len()]
                + count * usize::from(next_index + 1 == pieces.len())
                - knots[next_index];
            let left_span = step * left_intervals as f64;
            let right_span = step * right_intervals as f64;
            let left: [f64; 4] = second(points, true).map(|v| v / (left_span * left_span));
            let right: [f64; 4] = second(next, false).map(|v| v / (right_span * right_span));
            for k in 0..4 {
                let error = (left[k] - right[k]).abs();
                // The float32 records can subtract nearly equal controls on very short intervals.
                // Bound that representation error separately from the actual continuity tolerance.
                let quantisation = |piece: &[[f32; 4]], span: f64| {
                    let scale = piece[1..=4]
                        .iter()
                        .map(|p| f64::from(p[k]).abs())
                        .fold(0.0_f64, f64::max);
                    24.0 * f64::from(f32::EPSILON) * scale / (span * span)
                };
                let tolerance = 0.0002 * (1.0 + left[k].abs())
                    + quantisation(points, left_span)
                    + quantisation(next, right_span);
                assert!(
                    error < tolerance,
                    "ring second derivative changed at interval {i}, coordinate {k}: {error} > {tolerance}; left {left:?}, right {right:?}, spans {left_span} {right_span}, knots {knots:?}, controls {points:?} / {next:?}"
                );
            }
            for j in 0..=20 {
                let t = f64::from(j) / 20.0;
                let u = 1.0 - t;
                let coeffs = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
                let p: [f64; 4] = std::array::from_fn(|k| {
                    (0..4)
                        .map(|n| coeffs[n] * f64::from(points[1 + n][k]))
                        .sum()
                });
                assert!((p[0].hypot(p[1]) - 1.0).abs() < tolerance * 1.1);
                let r = 0.04 + 0.01 * (3.0 * p[1].atan2(p[0])).cos();
                assert!((p[3] - r).abs() < tolerance * 1.1);
            }
        }
    }

    #[test]
    fn an_open_curved_strand_has_continuous_curvature_and_radial_caps() {
        let count = 240;
        let points: Vec<_> = (0..=count)
            .map(|i| {
                let angle = 4.0 * std::f64::consts::PI * i as f64 / count as f64;
                [
                    angle.cos(),
                    angle.sin(),
                    0.15 * angle,
                    0.035 + 0.008 * (5.0 * angle).sin(),
                ]
            })
            .collect();
        let mut pieces = Vec::new();
        add_tube_line(&mut pieces, &points, 0.00035);
        assert!(pieces.len() > 3);
        assert_eq!(pieces.first().unwrap().e[0][1], 0.0);
        for w in pieces.windows(2) {
            assert_eq!(w[0].e[0][2], w[1].e[0][1]);
            assert_eq!(w[0].meta, -1.0);
            assert_eq!(w[1].meta, -1.0);
        }
        let step = (0..3)
            .map(|k| (points[1][k] - points[0][k]).powi(2))
            .sum::<f64>()
            .sqrt();
        let mut knots: Vec<usize> = pieces
            .iter()
            .map(|piece| {
                let p = piece.e[1];
                (0..=count)
                    .min_by(|&a, &b| {
                        let distance = |i: usize| {
                            (0..3)
                                .map(|k| (points[i][k] - f64::from(p[k])).powi(2))
                                .sum::<f64>()
                        };
                        distance(a).partial_cmp(&distance(b)).unwrap()
                    })
                    .unwrap()
            })
            .collect();
        knots.push(count);
        let second = |piece: &[[f32; 4]], end: bool| {
            let (a, b, c) = if end { (2, 3, 4) } else { (1, 2, 3) };
            std::array::from_fn(|k| {
                6.0 * f64::from(piece[a][k]) - 12.0 * f64::from(piece[b][k])
                    + 6.0 * f64::from(piece[c][k])
            })
        };
        for i in 0..pieces.len() - 1 {
            let left_span = step * (knots[i + 1] - knots[i]) as f64;
            let right_span = step * (knots[i + 2] - knots[i + 1]) as f64;
            let left: [f64; 4] = second(&pieces[i].e, true).map(|v| v / left_span.powi(2));
            let right: [f64; 4] = second(&pieces[i + 1].e, false).map(|v| v / right_span.powi(2));
            for k in 0..4 {
                let quantisation = |piece: &[[f32; 4]], span: f64| {
                    let scale = piece[1..=4]
                        .iter()
                        .map(|p| f64::from(p[k]).abs())
                        .fold(0.0_f64, f64::max);
                    24.0 * f64::from(f32::EPSILON) * scale / span.powi(2)
                };
                let tolerance = 0.0002 * (1.0 + left[k].abs())
                    + quantisation(&pieces[i].e, left_span)
                    + quantisation(&pieces[i + 1].e, right_span);
                let error = (left[k] - right[k]).abs();
                assert!(error < tolerance, "open second derivative changed at interval {i}, coordinate {k}: {error} > {tolerance}");
            }
        }
    }
}
