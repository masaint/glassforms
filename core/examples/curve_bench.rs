//! How long the curve maths takes: the raw curve (an orbit, or a knot's formula) and `prep_curve`,
//! which runs every frame while a shape animates.
//!
//!     cargo run --release -p gf-core --example curve_bench
//!
//! Each line is one preset at its default values, built the way an animation frame builds it: a flow as
//! a window on twice its length of orbit, placed by the whole stretch and tapered at its ends; a knot as
//! its formula gives it. `plain` is `prepCurve` copied line for line from forms.js with nothing kept
//! from one pass to the next: what `prep_curve` has to equal bit for bit (checked here on every run),
//! and the measure of what its bookkeeping saves. The same measurements in the web version's engine:
//!
//!     node tools/dump_curves.mjs --time

use gf_core::curve::{jsmath, orbit_line, placement, prep_curve, Curve, PlaceOpts, PrepOpts};
use gf_core::presets::{attractors, knots, CurveOpts, Family, Flow, RawCurve, Shape};
use gf_core::{HB, TRAVEL};
use std::hint::black_box;
use std::time::Instant;

const RUNS: usize = 201;

/// The middle one and the fastest of `RUNS` timings, in milliseconds, after a few runs to warm up.
/// (On a busy machine the fastest is the better guide to what the work itself costs.)
fn time_ms(mut f: impl FnMut()) -> String {
    for _ in 0..8 {
        f();
    }
    let mut t: Vec<f64> = (0..RUNS)
        .map(|_| {
            let t0 = Instant::now();
            f();
            t0.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    t.sort_by(|a, b| a.total_cmp(b));
    format!("{:5.2} /{:5.2}", t[RUNS / 2], t[0])
}

/// `Math.hypot` of three numbers as a loop, the way the browser has it.
fn hypot3_plain(a: f64, b: f64, c: f64) -> f64 {
    let v = [a.abs(), b.abs(), c.abs()];
    let mut max = 0.0;
    for x in v {
        if x > max { max = x; }
    }
    if max == f64::INFINITY { return f64::INFINITY; }
    if a.is_nan() || b.is_nan() || c.is_nan() { return f64::NAN; }
    if max == 0.0 { return 0.0; }
    let (mut sum, mut lost) = (0.0, 0.0);
    for x in v {
        let n = x / max;
        let add = n * n - lost;
        let next = sum + add;
        lost = (next - sum) - add;
        sum = next;
    }
    sum.sqrt() * max
}

/// `prepCurve` as it stands in forms.js, statement for statement.
fn prep_curve_plain(lines: &[&[f64]], o: &PrepOpts) -> Curve {
    let pose = match &o.pose { Some(p) => p.clone(), None => placement(lines, &o.place) };
    let (c, sc, ax) = (pose.c, pose.sc, pose.ax);
    let mut length = 0.0;
    let mut placed: Vec<Vec<f64>> = lines
        .iter()
        .map(|l| {
            let mut t = vec![0.0; l.len() / 3 * 3];
            for (p, q) in l.as_chunks::<3>().0.iter().zip(t.as_chunks_mut::<3>().0) {
                for k in 0..3 { q[k] = (p[ax[k]] - c[ax[k]]) * sc; }
            }
            if let Some(e) = &pose.e {
                for q in t.as_chunks_mut::<3>().0 {
                    let (x, y, z) = (q[0], q[1], q[2]);
                    for k in 0..3 { q[k] = x * e[k][0] + y * e[k][1] + z * e[k][2]; }
                }
            }
            let mut i = 3;
            while i < t.len() {
                length += hypot3_plain(t[i] - t[i - 3], t[i + 1] - t[i - 2], t[i + 2] - t[i - 1]);
                i += 3;
            }
            t
        })
        .collect();
    let ds = o.ds.max(length / 240000.0);
    for t in placed.iter_mut() {
        let m = t.len() / 3;
        if m < 3 { continue; }
        let at = |i: isize| -> [f64; 3] {
            let i = if o.closed { i.rem_euclid(m as isize - 1) } else { i.clamp(0, m as isize - 1) } as usize * 3;
            [t[i], t[i + 1], t[i + 2]]
        };
        let mut out = Vec::new();
        for i in 0..m as isize - 1 {
            let (a, b, c, d) = (at(i - 1), at(i), at(i + 1), at(i + 2));
            let n = (hypot3_plain(c[0] - b[0], c[1] - b[1], c[2] - b[2]) / (0.5 * ds)).ceil();
            let n = if n.is_nan() { 0.0 } else if n < 1.0 { 1.0 } else { n };
            let mut j = 0.0;
            while j < n {
                let u = j / n;
                let u2 = u * u;
                let u3 = u2 * u;
                for k in 0..3 {
                    out.push(0.5 * (2.0 * b[k] + (c[k] - a[k]) * u + (2.0 * a[k] - 5.0 * b[k] + 4.0 * c[k] - d[k]) * u2 + (3.0 * b[k] - a[k] - 3.0 * c[k] + d[k]) * u3));
                }
                j += 1.0;
            }
        }
        out.extend_from_slice(&t[t.len() - 3..]);
        *t = out;
    }
    let (mut pts, mut brk) = (Vec::new(), Vec::new());
    let mut rmax = 0.0;
    let mut amp = 1.0;
    if o.melt > 0.0 {
        let (k, r) = (1.0 / o.melt, o.tube);
        let mut sum = 1.0;
        for m in 1..6000 {
            let t = jsmath::exp(-k * (jsmath::hypot2(r, (m as f64 - 0.5) * ds) - r));
            sum += 2.0 * t;
            if t < 1e-8 { break; }
        }
        amp = 1.0 / sum;
    }
    for t in &placed {
        let m = t.len() / 3;
        if m < 2 { continue; }
        let mut total = 0.0;
        let mut i = 3;
        while i < t.len() {
            total += hypot3_plain(t[i] - t[i - 3], t[i + 1] - t[i - 2], t[i + 2] - t[i - 1]);
            i += 3;
        }
        let mut emit = |x: f64, y: f64, z: f64, s: f64| {
            let r = o.tube * match o.radius { Some(f) => f(s / if total != 0.0 && !total.is_nan() { total } else { 1.0 }, s, total), None => 1.0 };
            if r > rmax { rmax = r; }
            pts.extend_from_slice(&[x as f32, y as f32, z as f32, r as f32]);
            brk.push(0u8);
        };
        emit(t[0], t[1], t[2], 0.0);
        let (mut acc, mut next) = (0.0, ds);
        let mut i = 3;
        while i < t.len() {
            let (dx, dy, dz) = (t[i] - t[i - 3], t[i + 1] - t[i - 2], t[i + 2] - t[i - 1]);
            let seg = hypot3_plain(dx, dy, dz);
            while acc + seg >= next {
                let u = (next - acc) / seg;
                emit(t[i - 3] + dx * u, t[i - 2] + dy * u, t[i - 1] + dz * u, next);
                next += ds;
            }
            acc += seg;
            i += 3;
        }
        emit(t[t.len() - 3], t[t.len() - 2], t[t.len() - 1], total);
        *brk.last_mut().unwrap() = 1;
    }
    let n = brk.len();
    Curve { pts, brk, n, rmax, ds, length, amp, pose, vary: false }
}

/// Every bit of two prepared curves is the same.
fn same(a: &Curve, b: &Curve) -> bool {
    let bits = |c: &Curve| [c.rmax, c.ds, c.length, c.amp].map(f64::to_bits);
    a.n == b.n && a.brk == b.brk && a.pose == b.pose && bits(a) == bits(b) && a.pts.len() == b.pts.len() && a.pts.iter().zip(&b.pts).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn bench(fam: &Family, id: &str) {
    let p = fam.preset(id).expect("preset");
    let vals = p.defaults();
    let tube = [p.tube.unwrap_or(0.0), fam.tube, 0.03].into_iter().find(|&t| t != 0.0).unwrap();
    let melt = fam.melt.min(0.45 * tube);
    let place = PlaceOpts { centre: p.centre, fit: p.fit, axes: p.axes, align: p.align.unwrap_or(fam.align), e: None };
    let mut line = format!("{id:<8}");
    // the raw curve; for a flow also where the whole stretch sits
    let (raw, pose) = match &p.shape {
        Shape::Flow(flow) => {
            let n = jsmath::round(flow.steps as f64).max(40.0);
            let whole = jsmath::round(n * (1.0 + TRAVEL)) as usize;
            let ms = time_ms(|| {
                black_box(orbit_line(flow, black_box(&vals), whole, false));
            });
            line += &format!("{:<22}", format!("{ms} ({whole})"));
            let stretch = orbit_line(flow, &vals, whole, false);
            let pose = placement(&[&stretch], &place);
            (RawCurve { lines: vec![stretch[..(n as usize * 3).min(stretch.len())].to_vec()], closed: false, ..Default::default() }, Some(pose))
        }
        Shape::Curve(f) => {
            let o = CurveOpts { len: 1.0, quick: false };
            let ms = time_ms(|| {
                black_box(f(black_box(&vals), &o));
            });
            let raw = f(&vals, &o);
            line += &format!("{:<22}", format!("{ms} ({})", raw.lines.iter().map(|l| l.len() / 3).sum::<usize>()));
            (raw, None)
        }
        _ => panic!("{id} is not a curve"),
    };
    let lines: Vec<&[f64]> = raw.lines.iter().map(|l| l.as_slice()).collect();
    let taper = |_u: f64, s: f64, total: f64| 1.0f64.min(s / 0.35).min((total - s) / 0.35).sqrt().max(0.2);
    for res in [256.0, 768.0] {
        let o = PrepOpts {
            tube,
            melt,
            ds: 0.75 * 2.0 * HB / res,
            closed: raw.closed,
            radius: if p.is_flow() { Some(&taper) } else { None },
            place: place.clone(),
            pose: pose.clone(),
        };
        let (tuned, plain) = (prep_curve(&lines, &o), prep_curve_plain(&lines, &o));
        assert!(same(&tuned, &plain), "{id} at {res}: prep_curve and the plain copy of prepCurve differ");
        let plain = time_ms(|| {
            black_box(prep_curve_plain(black_box(&lines), &o));
        });
        let ms = time_ms(|| {
            black_box(prep_curve(black_box(&lines), &o));
        });
        line += &format!("{:<38}", format!("{plain} | {ms} ({})", tuned.n));
    }
    println!("{line}");
}

fn main() {
    println!("milliseconds: median / fastest of {RUNS} runs");
    println!("{:<8}{:<22}{:<38}{:<38}", "", "raw curve", "prep_curve at res 256", "prep_curve at res 768");
    println!("{:<8}{:<22}{:<38}{:<38}", "preset", "(points)", "plain | as it is (points)", "plain | as it is (points)");
    let (a, k) = (attractors::family(), knots::family());
    bench(&a, "thomas");
    bench(&a, "lorenz");
    bench(&k, "torus");

    // What the correctly rounded sine costs the one family that leans on it: the same Thomas orbit with
    // fdlibm's sine (a different orbit after a while; only the time is of interest).
    if let Shape::Flow(flow) = &a.preset("thomas").expect("preset").shape {
        let with_fdlibm = Flow {
            f: |x, y, z, v| [jsmath::fdlibm_sin(y) - v[0] * x, jsmath::fdlibm_sin(z) - v[0] * y, jsmath::fdlibm_sin(x) - v[0] * z],
            ..*flow
        };
        let n = 2 * flow.steps;
        let rounded = time_ms(|| {
            black_box(orbit_line(flow, black_box(&[0.208186]), n, false));
        });
        let fdlibm = time_ms(|| {
            black_box(orbit_line(&with_fdlibm, black_box(&[0.208186]), n, false));
        });
        println!("\nthomas orbit, {n} points after {} to settle, twelve sines a step:  correctly rounded {rounded}   fdlibm {fdlibm}", flow.settle);
    }
}
