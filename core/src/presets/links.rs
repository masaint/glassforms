//! Links: closed loops that hold on to each other. Smooth fat tubes, the tube's thickness on the usual
//! slider.
//!
//!   Hopf link          two round rings, each through the other: the simplest link there is.
//!   Borromean rings    three rings no two of which are linked, that still cannot be pulled apart:
//!                      the usual flat picture (three circles), lifted over and under alternately.
//!   Whitehead link     a twisted loop and a ring through both its lobes: linked, though the linking
//!                      number is nought (alternately over and under, as in its usual picture).
//!   torus link         several copies of a (p, q) torus knot side by side on one torus.
//!   chainmail          European 4-in-1: every ring through two of the row before and two of the row after.
//!   Hopf fibration     the circles of the 3-sphere that Hopf's map sends to one point each, seen in space
//!                      by stereographic projection: on each of several nested tori, circles linked once
//!                      with every other.

use super::*;
use geo::*;
use std::f64::consts::PI;

pub const FAMILY: &str = "links";

fn gcd(a: i64, b: i64) -> i64 { if b == 0 { a.abs() } else { gcd(b, a % b) } }

/// Two rings, `apart` between their centres, their planes `angle` degrees apart.
fn hopf_link(v: &[f64], _o: &CurveOpts) -> RawCurve {
    let (apart, angle) = (v[0], v[1].to_radians());
    let a = circle([-0.5 * apart, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1.0, 600);
    // the second ring turned about the line joining the centres
    let n = [0.0, angle.cos(), angle.sin()];
    let b = circle([0.5 * apart, 0.0, 0.0], [1.0, 0.0, 0.0], n, 1.0, 600);
    closed(vec![a, b])
}

/// Flat circles sampled for lifting.
fn flat_circle(c: [f64; 2], rx: f64, ry: f64, n: usize) -> Vec<[f64; 2]> {
    (0..n).map(|i| { let t = TAU * i as f64 / n as f64; [c[0] + rx * t.cos(), c[1] + ry * t.sin()] }).collect()
}

fn borromean(v: &[f64], _o: &CurveOpts) -> RawCurve {
    let (spread, lift) = (v[0], v[1]);
    let rings: Vec<Vec<[f64; 2]>> = (0..3)
        .map(|k| {
            let a = PI / 2.0 + TAU * k as f64 / 3.0;
            flat_circle([spread * a.cos(), spread * a.sin()], 1.0, 1.0, 720)
        })
        .collect();
    closed(lift_alternating(&rings, lift))
}

fn whitehead(v: &[f64], _o: &CurveOpts) -> RawCurve {
    let (wide, lift) = (v[0], v[1]);
    // a figure of eight (crossing itself once) and an ellipse through both of its lobes
    let eight: Vec<[f64; 2]> = (0..900).map(|i| { let t = TAU * i as f64 / 900.0; [1.45 * t.sin(), 0.62 * (2.0 * t).sin()] }).collect();
    let ring = flat_circle([0.0, 0.0], wide, 1.0, 720);
    closed(lift_alternating(&[eight, ring], lift))
}

/// `components` copies of the (p, q) torus knot: the torus link T(cp, cq).
fn torus_link(v: &[f64], _o: &CurveOpts) -> RawCurve {
    let (p, q, comps, r) = (v[0].round().max(1.0) as i64, v[1].round().max(1.0) as i64, v[2].round().max(1.0) as i64, v[3]);
    let (pp, qq) = (p * comps, q * comps);
    let g = gcd(pp, qq);
    let (a, b) = ((pp / g) as f64, (qq / g) as f64);
    let lines = (0..g)
        .map(|k| {
            // (at any one angle round the axis a copy passes a times, 2 pi / a apart round the tube: the
            // copies share those gaps evenly)
            let shift = TAU * k as f64 / (g as f64 * a);
            ring(
                |t| {
                    let (th, ph) = (a * t, b * t + shift);
                    let w = 1.0 + r * ph.cos();
                    [w * th.cos(), w * th.sin(), r * ph.sin()]
                },
                (300.0 * (a + b)) as usize,
            )
        })
        .collect();
    closed(lines)
}

/// Rows of rings, each through two of the row before and two of the row after (European 4-in-1).
/// Rings of alternate rows lean opposite ways about the rows.
fn chainmail(v: &[f64], _o: &CurveOpts) -> RawCurve {
    let (rows, cols, lean, drape) = (v[0].round().max(1.0) as usize, v[1].round().max(1.0) as usize, v[2].to_radians(), v[3]);
    let (sx, sy) = (SX, SY);
    let mut lines = Vec::new();
    for i in 0..rows {
        // (an odd row has one ring fewer, set between those of the rows either side)
        let n = if i % 2 == 1 { cols.saturating_sub(1) } else { cols };
        for j in 0..n {
            let x = (j as f64 + if i % 2 == 1 { 0.5 } else { 0.0 } - 0.5 * (cols as f64 - 1.0)) * sx;
            let y = (i as f64 - 0.5 * (rows as f64 - 1.0)) * sy;
            let s = if i % 2 == 0 { 1.0 } else { -1.0 };
            // (leaning along the row, like books on a shelf: each ring clears the next in its row)
            let nrm = [s * lean.sin(), 0.0, lean.cos()];
            // draped: the sheet wrapped round an upright cylinder, `drape` the angle it turns at its edge
            let half = 0.5 * cols as f64 * sx;
            let (centre, nrm) = if drape > 1e-6 {
                let rc = half / drape;
                let a = x / rc;
                let (c, sn) = (a.cos(), a.sin());
                ([rc * sn, y, rc * (1.0 - c)], [nrm[0] * c - nrm[2] * sn, nrm[1], nrm[0] * sn + nrm[2] * c])
            } else {
                ([x, y, 0.0], nrm)
            };
            let a = unit(cross(nrm, [0.0, 1.0, 0.0]));
            let b = cross(nrm, a);
            lines.push(circle(centre, a, b, 1.0, 160));
        }
    }
    closed(lines)
}
/// Ring spacing along a row and between rows, for rings of radius 1 (worked out so that each ring passes
/// through its four and no other, with room for the tube: see the test).
const SX: f64 = 1.2;
const SY: f64 = 1.3;

/// The Hopf fibration: the fibres over circles of latitude of the 2-sphere lie on nested tori of the
/// 3-sphere; projected stereographically, each is a circle, and every two are linked once.
fn fibration(v: &[f64], _o: &CurveOpts) -> RawCurve {
    let (fibres, tori, spread, stagger, turn) = (v[0].round().max(1.0) as usize, v[1].round().max(1.0) as usize, v[2], v[3], v[4]);
    let mut lines = Vec::new();
    for k in 0..tori {
        // the torus: |z1| = cos eta, |z2| = sin eta (eta = pi/4 the Clifford torus)
        let f = if tori == 1 { 0.5 } else { k as f64 / (tori - 1) as f64 };
        // (from the Clifford torus, eta = pi/4, inwards towards the core circle: farther out the circles
        // pass near the pole of the projection and grow without bound)
        let eta = PI / 4.0 * (1.0 - spread * (1.0 - f));
        let (ce, se) = (eta.cos(), eta.sin());
        for j in 0..fibres {
            let ph = TAU * (j as f64 + stagger * k as f64) / fibres as f64;
            lines.push(ring(
                |t| {
                    // (z1, z2) = (cos eta e^{it}, sin eta e^{i(t + ph)}), turned in the 3-sphere by `turn`
                    // (in the plane of x1 and x4) so that the projection's pole is not on a fibre
                    let (x1, x2, x3, x4) = (ce * t.cos(), ce * t.sin(), se * (t + ph).cos(), se * (t + ph).sin());
                    let (c, s) = (turn.cos(), turn.sin());
                    let (x1, x4) = (c * x1 - s * x4, s * x1 + c * x4);
                    let w = 1.0 / (1.0 - x4).max(1e-3);
                    [x1 * w, x3 * w, x2 * w]
                },
                600,
            ));
        }
    }
    closed(lines)
}

fn presets() -> Vec<Preset> {
    vec![
        Preset {
            tube: Some(0.16),
            fixed_length: true,
            rot: Some([18.0, 28.0, 0.0]),
            animate: Some(vec!["angle"]),
            eq: vec!["two circles, each through the other"],
            note: "The simplest link: two rings, each threaded once through the other.",
            params: vec![ParamDef::labelled("apart", "apart", 1.0, 0.4, 1.6, 0.01), ParamDef::labelled("angle", "angle between", 90.0, 25.0, 90.0, 0.5)],
            ..Preset::new("hopflink", "Hopf link", Shape::Curve(Box::new(hopf_link)))
        },
        Preset {
            tube: Some(0.1),
            fixed_length: true,
            rot: Some([-28.0, 0.0, 0.0]),
            animate: Some(vec!["lift"]),
            eq: vec!["three circles 120° apart,", "each over the next and under the last"],
            note: "No two of the rings are linked, yet the three cannot be pulled apart: cut any one and the others fall free.",
            params: vec![ParamDef::labelled("spread", "spread", 0.62, 0.45, 0.85, 0.005), ParamDef::labelled("lift", "over and under", 0.2, 0.08, 0.5, 0.005)],
            ..Preset::new("borromeanrings", "Borromean rings", Shape::Curve(Box::new(borromean)))
        },
        Preset {
            tube: Some(0.1),
            fixed_length: true,
            rot: Some([-28.0, 0.0, 0.0]),
            animate: Some(vec!["lift"]),
            eq: vec!["a figure of eight, and a ring through both lobes", "alternately over and under"],
            note: "Linked, though each loop winds round the other zero times in all: it cannot be pulled apart all the same.",
            params: vec![ParamDef::labelled("wide", "ring width", 0.62, 0.4, 0.95, 0.005), ParamDef::labelled("lift", "over and under", 0.2, 0.08, 0.5, 0.005)],
            ..Preset::new("whitehead", "Whitehead link", Shape::Curve(Box::new(whitehead)))
        },
        Preset {
            tube: Some(0.085),
            fixed_length: true,
            rot: Some([30.0, 0.0, 0.0]),
            animate: Some(vec!["r"]),
            eq: vec!["components copies of the (p, q) torus knot", "x = (1 + r cos qt) cos pt, y = (1 + r cos qt) sin pt, z = r sin qt"],
            note: "Copies of one torus knot laid side by side round the same torus, each linked with every other.",
            params: vec![
                ParamDef::new("p", 1.0, 1.0, 7.0, 1.0),
                ParamDef::new("q", 2.0, 1.0, 9.0, 1.0),
                ParamDef::labelled("components", "components", 3.0, 1.0, 6.0, 1.0),
                ParamDef::new("r", 0.45, 0.15, 0.8, 0.01),
            ],
            info: Some(Box::new(|raw, _v| format!("{} loops", raw.lines.len()))),
            ..Preset::new("toruslink", "Torus link", Shape::Curve(Box::new(torus_link)))
        },
        Preset {
            tube: Some(0.028),
            fixed_length: true,
            rot: Some([-38.0, 18.0, 0.0]),
            animate: Some(vec!["drape"]),
            eq: vec!["rows of rings leaning along the row, alternate rows the other way", "each through two above and two below"],
            note: "European 4-in-1 mail: every ring passes through four others.",
            params: vec![
                ParamDef::new("rows", 7.0, 2.0, 16.0, 1.0),
                ParamDef::labelled("cols", "columns", 6.0, 2.0, 14.0, 1.0),
                ParamDef::labelled("lean", "lean", 35.0, 28.0, 42.0, 0.5),
                ParamDef::labelled("drape", "drape", 0.5, 0.0, 1.5, 0.01),
            ],
            info: Some(Box::new(|raw, _v| format!("{} rings", raw.lines.len()))),
            ..Preset::new("chainmail", "Chainmail", Shape::Curve(Box::new(chainmail)))
        },
        Preset {
            tube: Some(0.045),
            fixed_length: true,
            // (from above the axis the tori share: a rosette, every circle round it in turn)
            rot: Some([72.0, 0.0, 12.0]),
            animate: Some(vec!["turn"]),
            eq: vec!["(z₁, z₂) = (cos η·eⁱᵗ, sin η·eⁱ⁽ᵗ⁺ᵠ⁾) on |z₁|² + |z₂|² = 1", "projected: (x₁, x₃, x₂) / (1 − x₄)"],
            note: "Circles of the 3-sphere, each lying on a torus nested in the next: every circle is linked once with every other.",
            params: vec![
                ParamDef::labelled("fibres", "circles per torus", 7.0, 3.0, 30.0, 1.0),
                ParamDef::labelled("tori", "tori", 2.0, 1.0, 6.0, 1.0),
                ParamDef::labelled("spread", "spread of tori", 0.35, 0.0, 0.9, 0.01),
                ParamDef::labelled("stagger", "stagger", 0.5, 0.0, 1.0, 0.01),
                ParamDef::labelled("turn", "turn in 4D", 0.0, 0.0, 3.14, 0.01),
            ],
            info: Some(Box::new(|raw, _v| format!("{} circles", raw.lines.len()))),
            ..Preset::new("fibration", "Hopf fibration", Shape::Curve(Box::new(fibration)))
        },
    ]
}

pub fn family() -> Family {
    Family {
        presets: presets(),
        melt: 0.004,
        defaults: vec![("absorb", Setting::Vec3([0.14, 0.07, 0.03])), ("rot", Setting::Vec3([18.0, 28.0, 0.0]))],
        lights: tube_lights(),
        ..Family::new(FAMILY, "Links", Kind::Curve)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// How many times the closed line b passes through the disc of the circle a (centre c, normal n, radius 1).
    fn through(c: V, n: V, b: &[f64]) -> usize {
        let pts: Vec<V> = b.chunks(3).map(|q| [q[0], q[1], q[2]]).collect();
        let mut k = 0;
        for w in pts.windows(2) {
            let (s0, s1) = (dot(sub(w[0], c), n), dot(sub(w[1], c), n));
            if (s0 > 0.0) != (s1 > 0.0) {
                let p = lerp(w[0], w[1], s0 / (s0 - s1));
                if len(sub(p, c)) < 1.0 { k += 1; }
            }
        }
        k
    }
    fn centre_normal(l: &[f64]) -> (V, V) {
        let pts: Vec<V> = l.chunks(3).map(|q| [q[0], q[1], q[2]]).collect();
        let m = pts.len() - 1;
        let c = mul(pts[..m].iter().fold([0.0; 3], |a, &p| add(a, p)), 1.0 / m as f64);
        (c, unit(cross(sub(pts[0], c), sub(pts[m / 4], c))))
    }

    #[test]
    fn every_ring_of_the_mail_holds_four_others_and_no_more() {
        let raw = chainmail(&[5.0, 5.0, 35.0, 0.0], &CurveOpts { len: 1.0, quick: false, text: "" });
        // the middle ring of the middle row
        let mid = 2 * 5 + 4 + 2;
        let (c, n) = centre_normal(&raw.lines[mid]);
        let linked = raw.lines.iter().enumerate().filter(|&(i, l)| i != mid && through(c, n, l) % 2 == 1).count();
        assert_eq!(linked, 4);
        // and the tubes have room: no two rings come closer than a tube's width
        for (i, l) in raw.lines.iter().enumerate() {
            if i == mid { continue; }
            let d = l.chunks(3).flat_map(|q| raw.lines[mid].chunks(3).map(move |p| len(sub([q[0], q[1], q[2]], [p[0], p[1], p[2]])))).fold(f64::MAX, f64::min);
            assert!(d > 0.12, "ring {i} comes within {d}");
        }
    }

    #[test]
    fn the_hopf_link_is_linked_and_the_borromean_rings_pairwise_not() {
        let o = CurveOpts { len: 1.0, quick: false, text: "" };
        let h = hopf_link(&[1.0, 90.0], &o);
        let (c, n) = centre_normal(&h.lines[0]);
        assert_eq!(through(c, n, &h.lines[1]), 1);
        let b = borromean(&[0.62, 0.2], &o);
        assert_eq!(b.lines.len(), 3);
        // lifted: each ring goes over at three crossings and under at three (heights both ways)
        for l in &b.lines {
            let zs: Vec<f64> = l.chunks(3).map(|q| q[2]).collect();
            assert!(zs.iter().cloned().fold(f64::MIN, f64::max) > 0.15 && zs.iter().cloned().fold(f64::MAX, f64::min) < -0.15);
        }
    }

    #[test]
    fn a_torus_link_has_as_many_loops_as_asked() {
        let o = CurveOpts { len: 1.0, quick: false, text: "" };
        assert_eq!(torus_link(&[2.0, 3.0, 2.0, 0.45], &o).lines.len(), 2);
        assert_eq!(torus_link(&[2.0, 3.0, 3.0, 0.45], &o).lines.len(), 3);
        assert_eq!(torus_link(&[2.0, 4.0, 1.0, 0.45], &o).lines.len(), 2);
        // copies never lie on top of each other
        let two = torus_link(&[1.0, 2.0, 2.0, 0.45], &o);
        let d = len(sub([two.lines[0][0], two.lines[0][1], two.lines[0][2]], [two.lines[1][0], two.lines[1][1], two.lines[1][2]]));
        assert!(d > 0.3, "{d}");
    }
}
