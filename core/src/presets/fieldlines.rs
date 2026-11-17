//! Field lines: the curves a field or a flow follows, as glass tubes.
//!
//!   dipole            the field of a bar magnet (or the earth's): every line lies in a plane through the
//!                     axis and is r = L sin²θ; a glass ball stands for the magnet
//!   vortex ring       Hill's spherical vortex, with swirl (Moffatt's): a ball of fluid turning over and
//!                     round as a smoke ring does; every path winds round a torus nested in the next
//!   Lorenz manifold   the stable manifold of the Lorenz system's origin: the surface of every point that
//!                     flows into the origin, which the butterfly attractor wraps itself round. Drawn as
//!                     tubes (not as a surface): rings of points that reach the origin after the same
//!                     length of path, grown outward from a small circle by following the flow backwards
//!                     (after Osinga and Krauskopf), and a few of the paths themselves across them.

use super::*;
use geo::*;
use std::f64::consts::PI;

pub const FAMILY: &str = "fieldlines";

/// The farthest any point of the lines (or `extra`) is from their middle: what placement scales by.
fn reach(lines: &[Vec<f64>], extra: &[V]) -> f64 {
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    let pts = || lines.iter().flat_map(|l| l.chunks(3).map(|q| [q[0], q[1], q[2]])).chain(extra.iter().copied());
    for p in pts() {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let c = mul(add(lo, hi), 0.5);
    pts().map(|p| len(sub(p, c))).fold(0.0, f64::max)
}

/// A dipole's field lines: `shells` values of L, `lines` planes round the axis, a ball of radius `core`
/// (as a share of the smallest L) for the magnet.
fn dipole(v: &[f64], _o: &CurveOpts) -> RawCurve {
    let (shells, planes, spread, core) = (v[0].round().max(1.0) as usize, v[1].round().max(1.0) as usize, v[2], v[3]);
    // (the tube as placed; the thickness slider scales it)
    let tube = 0.055;
    let rc = core;
    let mut lines = Vec::new();
    for j in 0..shells {
        let l = spread.powi(j as i32);
        // only the part of the line outside the ball (and a little way into it)
        let s0 = if rc > 0.0 { (0.85 * rc / l).sqrt().min(0.99) } else { 0.0 };
        let th0 = s0.asin();
        for k in 0..planes {
            let ph = TAU * (k as f64 + 0.5 * j as f64) / planes as f64;
            let n = 300;
            let mut pts = Vec::with_capacity(3 * (n + 1));
            for i in 0..=n {
                let th = th0 + (PI - 2.0 * th0) * i as f64 / n as f64;
                let r = l * th.sin().powi(2);
                // the axis upright: y = r cos(theta)
                pts.extend_from_slice(&[r * th.sin() * ph.cos(), r * th.cos(), r * th.sin() * ph.sin()]);
            }
            lines.push(pts);
        }
    }
    let extra: Vec<V> = if rc > 0.0 { vec![[rc, 0.0, 0.0], [-rc, 0.0, 0.0], [0.0, rc, 0.0], [0.0, -rc, 0.0]] } else { Vec::new() };
    let scale = reach(&lines, &extra) / crate::FIT;
    let radii = lines.iter().map(|l| vec![tube * scale; l.len() / 3]).collect();
    let mut raw = RawCurve { lines, radii, ..Default::default() };
    if rc > 0.0 {
        raw.pieces.push(Piece { lo: [-rc; 3], hi: [rc; 3], terms: vec![Term::Ball { c: [0.0; 3], r: rc }] });
    }
    raw
}

/// Hill's spherical vortex (radius 1, axis upright) with swirl: the stream function
/// psi = r² (1 − r² − z²) (r from the axis, z along it) gives the flow over and round in each plane through
/// the axis, and a swirl round the axis of k psi / r.
fn hill(p: V, swirl: f64) -> V {
    let (x, y, z) = (p[0], p[1], p[2]);
    let r2 = x * x + z * z;
    let r = r2.sqrt().max(1e-9);
    let psi = r2 * (1.0 - r2 - y * y);
    let ur = 2.0 * r * y;
    let uy = 2.0 * (1.0 - 2.0 * r2 - y * y);
    let uph = swirl * psi.max(0.0) / r;
    let (c, s) = (x / r, z / r);
    [ur * c - uph * s, uy, ur * s + uph * c]
}
fn rk4(p: V, h: f64, f: &dyn Fn(V) -> V) -> V {
    let k1 = f(p);
    let k2 = f(add(p, mul(k1, 0.5 * h)));
    let k3 = f(add(p, mul(k2, 0.5 * h)));
    let k4 = f(add(p, mul(k3, h)));
    add(p, mul(add(add(k1, mul(k2, 2.0)), add(mul(k3, 2.0), k4)), h / 6.0))
}

fn vortex(v: &[f64], _o: &CurveOpts) -> RawCurve {
    let (tori, per, swirl, turns) = (v[0].round().max(1.0) as usize, v[1].round().max(1.0) as usize, v[2], v[3]);
    let mut lines = Vec::new();
    // the core (where the flow only goes round) is the circle r = 1/sqrt 2 in the middle plane
    let core = 0.5f64.sqrt();
    for t in 0..tori {
        // start on the middle plane, between the core and the edge of the ball
        let f = (t as f64 + 1.0) / (tori as f64 + 0.6);
        let r0 = core + (0.97 - core) * f;
        for k in 0..per {
            let ph = TAU * (k as f64 + 0.5 * t as f64) / per as f64;
            let mut p = [r0 * ph.cos(), 0.0, r0 * ph.sin()];
            // follow it for `turns` turns over the core (counted by its angle about the core)
            let angle = |q: V| { let r = (q[0] * q[0] + q[2] * q[2]).sqrt(); q[1].atan2(r - core) };
            let (mut a0, mut wound) = (angle(p), 0.0);
            let mut pts = vec![p];
            let h = 0.004;
            let mut guard = 0;
            while wound < TAU * turns && guard < 400000 {
                p = rk4(p, h, &|q| hill(q, swirl));
                let a = angle(p);
                let mut da = a - a0;
                if da > PI { da -= TAU; }
                if da < -PI { da += TAU; }
                wound += da.abs();
                a0 = a;
                guard += 1;
                if guard % 3 == 0 { pts.push(p); }
            }
            lines.push(flat(&pts));
        }
    }
    open(lines)
}

const SIGMA: f64 = 10.0;
const RHO: f64 = 28.0;
const BETA: f64 = 8.0 / 3.0;
fn lorenz(p: V) -> V { [SIGMA * (p[1] - p[0]), p[0] * (RHO - p[2]) - p[1], p[0] * p[1] - BETA * p[2]] }

/// The stable manifold of the origin, grown as rings: every point moves back along the flow by the same
/// length of path each step (dp/ds = −f/|f|), points are put in where neighbours draw apart and taken out
/// where they crowd. Kept: the rings every `extent / rings` of path, and the paths of `paths` points.
struct Manifold {
    rings: Vec<Vec<V>>,
    paths: Vec<Vec<V>>,
}
fn manifold(rings: usize, paths: usize, extent: f64) -> Manifold {
    let lss = 0.5 * (-(SIGMA + 1.0) - ((SIGMA + 1.0).powi(2) + 4.0 * SIGMA * (RHO - 1.0)).sqrt());
    let ess = unit([SIGMA, lss + SIGMA, 0.0]);
    let ez = [0.0, 0.0, 1.0];
    let r0 = 1.0;
    let n0 = 120;
    // each point and the path it is following (usize::MAX: none)
    let mut front: Vec<(V, usize)> = (0..n0)
        .map(|i| {
            let t = TAU * i as f64 / n0 as f64;
            (add(mul(ess, r0 * t.cos()), mul(ez, r0 * t.sin())), if paths > 0 && i % (n0 / paths.min(n0)).max(1) == 0 { i } else { usize::MAX })
        })
        .collect();
    let mut path_pts: std::collections::BTreeMap<usize, Vec<V>> = front.iter().filter(|f| f.1 != usize::MAX).map(|f| (f.1, vec![f.0])).collect();
    let h = 0.25;
    let steps = (extent / h).ceil() as usize;
    let every = (steps / rings.max(1)).max(1);
    let (dmax, dmin) = (1.2, 0.3);
    let back = |p: V| { let f = lorenz(p); mul(f, -1.0 / len(f).max(1e-9)) };
    let mut out = Manifold { rings: Vec::new(), paths: Vec::new() };
    for step in 1..=steps {
        for f in front.iter_mut() { f.0 = rk4(f.0, h, &back); }
        for f in &front {
            if f.1 != usize::MAX && step % 2 == 0 { path_pts.entry(f.1).or_default().push(f.0); }
        }
        // even out the ring
        let mut next: Vec<(V, usize)> = Vec::with_capacity(front.len() + 16);
        let n = front.len();
        for i in 0..n {
            let (a, b) = (front[i], front[(i + 1) % n]);
            let d = len(sub(b.0, a.0));
            if d < dmin && a.1 == usize::MAX && n > 24 && i + 1 < n { continue; }
            next.push(a);
            if d > dmax && front.len() < 6000 {
                let m = (d / dmax).ceil() as usize;
                for j in 1..m { next.push((lerp(a.0, b.0, j as f64 / m as f64), usize::MAX)); }
            }
        }
        front = next;
        if step % every == 0 { out.rings.push(front.iter().map(|f| f.0).collect()); }
    }
    out.paths = path_pts.into_values().filter(|p| p.len() > 3).collect();
    out
}
static MANIFOLDS: Memo<Manifold> = Memo::new();

fn lorenz_manifold(v: &[f64], _o: &CurveOpts) -> RawCurve {
    let (rings, paths, extent, grow) = (v[0].round().max(1.0) as usize, v[1].round().max(0.0) as usize, v[2], v[3].clamp(0.0, 1.0));
    let m = MANIFOLDS.get(v, &[3], "", || manifold(rings, paths, extent));
    let shown = ((grow * m.rings.len() as f64).round() as usize).min(m.rings.len());
    let mut lines: Vec<Vec<f64>> = Vec::new();
    for r in &m.rings[..shown] {
        // (a ring runs a little past its start, the two ends lying in one tube: no seam)
        let mut l = flat(r);
        for q in r.iter().take(3) { l.extend_from_slice(q); }
        lines.push(l);
    }
    // the paths, as far as the rings shown reach
    for p in &m.paths {
        let k = ((grow * p.len() as f64).round() as usize).min(p.len());
        if k >= 3 { lines.push(flat(&p[..k])); }
    }
    // (the rings are loops; the paths are open: both are drawn as open lines, the rings closing on
    // themselves)
    let frame: Vec<f64> = m.rings.iter().flat_map(|r| r.iter().flat_map(|q| q.iter().copied())).collect();
    RawCurve { lines, frame, ..Default::default() }
}

fn presets() -> Vec<Preset> {
    vec![
        Preset {
            fixed_length: true,
            rot: Some([12.0, 20.0, 0.0]),
            animate: Some(vec!["spread"]),
            eq: vec!["r = L sin²θ", "in every plane through the axis, for each L"],
            note: "The field of a bar magnet, or of the earth: every line leaves one pole and comes back to the other.",
            params: vec![
                ParamDef::labelled("shells", "shells", 3.0, 1.0, 6.0, 1.0),
                ParamDef::labelled("lines", "lines round", 6.0, 2.0, 16.0, 1.0),
                ParamDef::labelled("spread", "spread", 1.6, 1.15, 2.4, 0.01),
                ParamDef::labelled("core", "magnet", 0.32, 0.0, 0.8, 0.01),
            ],
            info: Some(Box::new(|raw, _v| format!("{} lines", raw.lines.len()))),
            ..Preset::new("dipole", "Dipole field", Shape::Curve(Box::new(dipole)))
        },
        Preset {
            tube: Some(0.045),
            fixed_length: true,
            rot: Some([55.0, 0.0, 0.0]),
            animate: Some(vec!["swirl"]),
            radius: Some(Box::new(|u| (u / 0.06).min((1.0 - u) / 0.06).clamp(0.0, 1.0).sqrt().max(0.25))),
            eq: vec!["ψ = r²(1 − r² − z²) inside the ball", "u_φ = k ψ / r: Hill's vortex, swirled"],
            note: "A ball of fluid turning over itself as a smoke ring does, and round its axis too: every path winds round its own torus.",
            params: vec![
                ParamDef::labelled("tori", "tori", 1.0, 1.0, 5.0, 1.0),
                ParamDef::labelled("lines", "paths on each", 2.0, 1.0, 6.0, 1.0),
                ParamDef::labelled("swirl", "swirl", 0.35, 0.0, 3.0, 0.005),
                ParamDef::labelled("turns", "turns over", 9.0, 1.0, 24.0, 0.1),
            ],
            ..Preset::new("vortex", "Vortex ring", Shape::Curve(Box::new(vortex)))
        },
        Preset {
            tube: Some(0.022),
            fixed_length: true,
            rot: Some([-82.0, 0.0, 18.0]),
            animate: Some(vec!["grow"]),
            eq: vec!["ẋ = σ(y − x), ẏ = x(ρ − z) − y, ż = xy − βz", "points that flow into the origin: grown backwards, ring by ring"],
            note: "The surface the Lorenz butterfly is wrapped round: every point on it flows into the origin. Drawn as rings of equal path length and the paths across them.",
            params: vec![
                ParamDef::labelled("rings", "rings", 16.0, 3.0, 40.0, 1.0),
                ParamDef::labelled("paths", "paths", 12.0, 0.0, 40.0, 1.0),
                ParamDef::labelled("extent", "extent", 80.0, 10.0, 120.0, 0.5),
                ParamDef::labelled("grow", "grown", 1.0, 0.0, 1.0, 0.002),
            ],
            info: Some(Box::new(|raw, _v| format!("{} lines", raw.lines.len()))),
            ..Preset::new("lorenzmanifold", "Lorenz manifold", Shape::Curve(Box::new(lorenz_manifold)))
        },
    ]
}

pub fn family() -> Family {
    Family {
        presets: presets(),
        melt: 0.004,
        defaults: vec![("absorb", Setting::Vec3([0.14, 0.07, 0.03]))],
        lights: tube_lights(),
        ..Family::new(FAMILY, "Field lines", Kind::Curve)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const O: CurveOpts = CurveOpts { len: 1.0, quick: false, text: "" };

    #[test]
    fn dipole_lines_follow_r_equals_l_sin_squared() {
        let raw = dipole(&[2.0, 4.0, 1.6, 0.0], &O);
        assert_eq!(raw.lines.len(), 8);
        for (k, l) in raw.lines.iter().enumerate() {
            let big_l = 1.6f64.powi((k / 4) as i32);
            for q in l.chunks(3).skip(1).step_by(37) {
                let r = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
                let s2 = 1.0 - (q[1] / r).powi(2);
                assert!((r - big_l * s2).abs() < 1e-9 * big_l.max(1.0) + 1e-9, "{r} vs {}", big_l * s2);
            }
        }
    }

    #[test]
    fn vortex_paths_stay_on_their_torus() {
        // psi is kept along a path (swirl does not change it)
        let raw = vortex(&[2.0, 1.0, 2.0, 3.0], &O);
        for l in &raw.lines {
            let psi = |q: &[f64]| { let r2 = q[0] * q[0] + q[2] * q[2]; r2 * (1.0 - r2 - q[1] * q[1]) };
            let p0 = psi(&l[0..3]);
            for q in l.chunks(3) { assert!((psi(q) - p0).abs() < 1e-4, "{} vs {p0}", psi(q)); }
        }
    }

    #[test]
    fn the_manifold_flows_into_the_origin() {
        let m = manifold(6, 4, 20.0);
        assert_eq!(m.rings.len(), 6);
        // a point of the first ring, followed forwards along the flow for the length of path it was
        // grown by, comes back to the small circle round the origin it was grown from (forwards, the flow
        // pushes away from the surface fast: only a short way can be checked like this)
        let fwd = |p: V| { let f = lorenz(p); mul(f, 1.0 / len(f).max(1e-9)) };
        for k in [0, 7, 30] {
            let mut p = m.rings[0][k * m.rings[0].len() / 40];
            for _ in 0..(20.0f64 / 6.0 / 0.25).floor() as usize { p = rk4(p, 0.25, &fwd); }
            assert!(len(p) < 2.0, "{p:?}");
        }
        // and the rings grow: the last is longer than the first
        let length = |r: &Vec<V>| r.windows(2).map(|w| len(sub(w[1], w[0]))).sum::<f64>();
        assert!(length(&m.rings[5]) > 2.0 * length(&m.rings[0]));
    }
}
