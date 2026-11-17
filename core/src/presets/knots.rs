//! Knots, links and other closed curves, as fat glass tubes. Every one is a formula for a point moving
//! round a loop; the tube is drawn round the path.
//! (scenes/knots.js, written term for term as it is there, so that the points come out the same.)

use super::*;
use crate::curve::jsmath::{cos, log, pow, round, sin, sqrt};
use std::f64::consts::PI;

const TAU: f64 = 2.0 * PI;

fn gcd(a: f64, b: f64) -> f64 {
    if b != 0.0 && !b.is_nan() { gcd(b, a % b) } else { a }
}

/// Sample a closed curve t -> [x, y, z] over one turn of t: n + 1 points, the last repeating the first.
fn ring(f: impl Fn(f64) -> [f64; 3], n: usize) -> Vec<f64> {
    let mut a = Vec::with_capacity((n + 1) * 3);
    for i in 0..=n {
        a.extend_from_slice(&f((i % n) as f64 / n as f64 * TAU));
    }
    a
}
fn closed(lines: Vec<Vec<f64>>) -> RawCurve {
    RawCurve { lines, closed: true, ..Default::default() }
}

/// A knot drawn on a torus: p times round the axis while q times round the tube.
/// If p and q share a factor g, that is g separate interlinked loops.
fn torus_knot(v: &[f64], _o: &CurveOpts) -> RawCurve {
    let (p, q) = (round(v[0]).max(1.0), round(v[1]).max(1.0));
    let g = gcd(p, q);
    let (pp, qq, r) = (p / g, q / g, v[2]);
    let mut lines = Vec::new();
    let mut c = 0.0;
    while c < g {
        let shift = TAU / pp * c / g;
        lines.push(ring(
            |t| {
                let (th, ph) = (pp * t, qq * t + shift);
                let w = 1.0 + r * cos(ph);
                [w * cos(th), w * sin(th), r * sin(ph)]
            },
            (400.0 * (pp + qq)) as usize,
        ));
        c += 1.0;
    }
    closed(lines)
}
#[allow(clippy::too_many_arguments)]
fn torus(id: &'static str, name: &'static str, p: f64, q: f64, r: f64, tube: f64, rot: [f64; 3], note: &'static str) -> Preset {
    Preset {
        tube: Some(tube),
        fixed_length: true,
        rot: Some(rot),
        note,
        animate: Some(vec!["r"]),
        eq: vec!["x = (1 + r cos qt) cos pt", "y = (1 + r cos qt) sin pt", "z = r sin qt"],
        params: vec![ParamDef::new("p", p, 1.0, 9.0, 1.0), ParamDef::new("q", q, 1.0, 30.0, 1.0), ParamDef::new("r", r, 0.15, 0.8, 0.01)],
        info: Some(Box::new(|_raw, v| {
            let g = gcd(round(v[0]), round(v[1]));
            if g > 1.0 { format!("{g} linked loops") } else { "one loop".to_string() }
        })),
        ..Preset::new(id, name, Shape::Curve(Box::new(torus_knot)))
    }
}

/// Small seeded random number generator, so "seed" always gives the same shape.
/// (Whole-number arithmetic that wraps round at 32 bits, as the web version's does.)
fn rng(seed: f64) -> impl FnMut() -> f64 {
    // the seed as an unsigned 32-bit whole number: its whole part, wrapped round
    let mut a = if seed.is_finite() { seed.trunc().rem_euclid(4294967296.0) as u32 } else { 0 };
    move || {
        a = a.wrapping_add(0x6D2B79F5);
        let mut t = (a ^ (a >> 15)).wrapping_mul(1 | a);
        t = t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t)) ^ t;
        (t ^ (t >> 14)) as f64 / 4294967296.0
    }
}
/// A number from the bell curve, out of two from the generator.
fn gauss(r: &mut impl FnMut() -> f64) -> f64 {
    let a = sqrt(-2.0 * log(r() + 1e-12));
    a * cos(TAU * r())
}

// (the phase sliders stop at 3.14, as they do in the web version, not at pi)
#[allow(clippy::approx_constant)]
fn presets() -> Vec<Preset> {
    vec![
        Preset {
            animate: Some(vec!["@thick"]),
            tube: Some(0.13),
            fixed_length: true,
            rot: Some([8.0, 0.0, 0.0]),
            eq: vec!["x = sin t + 2 sin 2t", "y = cos t − 2 cos 2t", "z = −sin 3t"],
            note: "The simplest knot there is: three crossings. It cannot be untied without cutting.",
            ..Preset::new(
                "trefoil",
                "Trefoil",
                Shape::Curve(Box::new(|_v, _o| closed(vec![ring(|t| [sin(t) + 2.0 * sin(2.0 * t), cos(t) - 2.0 * cos(2.0 * t), -sin(3.0 * t)], 1400)]))),
            )
        },
        Preset {
            animate: Some(vec!["@thick"]),
            tube: Some(0.11),
            fixed_length: true,
            rot: Some([14.0, 0.0, 0.0]),
            eq: vec!["x = (2 + cos 2t) cos 3t", "y = (2 + cos 2t) sin 3t", "z = sin 4t"],
            note: "Four crossings. Unlike the trefoil, it is the same knot as its own mirror image.",
            ..Preset::new(
                "figure8",
                "Figure-eight",
                Shape::Curve(Box::new(|_v, _o| {
                    closed(vec![ring(|t| [(2.0 + cos(2.0 * t)) * cos(3.0 * t), (2.0 + cos(2.0 * t)) * sin(3.0 * t), sin(4.0 * t)], 1400)])
                })),
            )
        },
        torus("torus", "Torus knot", 3.0, 7.0, 0.42, 0.065, [28.0, 0.0, 0.0], "p times round the axis, q times round the tube. Try p = 2, q = 5 or p = 5, q = 8."),
        torus("cinquefoil", "Cinquefoil", 2.0, 5.0, 0.45, 0.1, [18.0, 0.0, 0.0], "The (2, 5) torus knot: five crossings, like a pentagram."),
        torus("solomon", "Solomon’s link", 2.0, 4.0, 0.45, 0.11, [18.0, 0.0, 0.0], "p and q share a factor of 2, so the curve closes twice: two loops, linked twice."),
        torus("coil", "Coil", 1.0, 22.0, 0.36, 0.045, [58.0, 0.0, 0.0], "Once round the axis, many times round the tube."),
        Preset {
            animate: Some(vec!["phi", "psi"]),
            tube: Some(0.075),
            fixed_length: true,
            rot: Some([20.0, 30.0, 0.0]),
            eq: vec!["x = cos(a·t + φ)", "y = cos(b·t + ψ)", "z = cos(c·t)"],
            note: "Three cosines with whole-number frequencies. Shift a phase and crossings swap over, changing the knot.",
            params: vec![
                ParamDef::labelled("fa", "a", 3.0, 1.0, 9.0, 1.0),
                ParamDef::labelled("fb", "b", 2.0, 1.0, 9.0, 1.0),
                ParamDef::labelled("fc", "c", 7.0, 1.0, 11.0, 1.0),
                ParamDef::labelled("phi", "φ", 0.7, 0.0, 3.14, 0.01),
                ParamDef::labelled("psi", "ψ", 0.2, 0.0, 3.14, 0.01),
            ],
            ..Preset::new(
                "lissajous",
                "Lissajous knot",
                Shape::Curve(Box::new(|v, _o| closed(vec![ring(|t| [cos(round(v[0]) * t + v[3]), cos(round(v[1]) * t + v[4]), cos(round(v[2]) * t)], 2400)]))),
            )
        },
        Preset {
            animate: Some(vec!["squash"]),
            tube: Some(0.1),
            fixed_length: true,
            rot: Some([35.0, 45.0, 0.0]),
            eq: vec!["three ellipses, one in each coordinate plane:", "(a cos t, b sin t, 0) and its cyclic shifts"],
            note: "No two rings are linked, yet the three cannot be pulled apart. Remove any one and the others fall free.",
            params: vec![ParamDef::labelled("squash", "b / a", 0.55, 0.35, 0.8, 0.01)],
            ..Preset::new(
                "borromean",
                "Borromean rings",
                Shape::Curve(Box::new(|v, _o| {
                    let b = v[0];
                    closed(vec![ring(|t| [cos(t), b * sin(t), 0.0], 600), ring(|t| [0.0, cos(t), b * sin(t)], 600), ring(|t| [b * sin(t), 0.0, cos(t)], 600)])
                })),
            )
        },
        Preset {
            animate: Some(vec!["size"]),
            tube: Some(0.055),
            fixed_length: true,
            rot: Some([55.0, 0.0, 0.0]),
            eq: vec!["n rings round a circle,", "each turned a quarter-turn from the last"],
            params: vec![ParamDef::new("links", 10.0, 4.0, 24.0, 2.0), ParamDef::new("size", 0.68, 0.56, 0.9, 0.01)],
            ..Preset::new(
                "chain",
                "Chain",
                Shape::Curve(Box::new(|v, _o| {
                    let n = 2.0 * round(v[0] / 2.0);
                    let s = 2.0 * sin(PI / n);
                    let r = v[1] * s;
                    let mut lines = Vec::new();
                    let mut k = 0.0;
                    while k < n {
                        let a = TAU * k / n;
                        let (c, tan) = ([cos(a), 0.0, sin(a)], [-sin(a), 0.0, cos(a)]);
                        let other = if k % 2.0 != 0.0 { [0.0, 1.0, 0.0] } else { [cos(a), 0.0, sin(a)] };
                        lines.push(ring(
                            |t| {
                                [
                                    c[0] + r * (cos(t) * tan[0] + sin(t) * other[0]),
                                    c[1] + r * (cos(t) * tan[1] + sin(t) * other[1]),
                                    c[2] + r * (cos(t) * tan[2] + sin(t) * other[2]),
                                ]
                            },
                            300,
                        ));
                        k += 1.0;
                    }
                    closed(lines)
                })),
            )
        },
        Preset {
            animate: Some(vec!["theta"]),
            tube: Some(0.04),
            fixed_length: true,
            rot: Some([20.0, 25.0, 0.0]),
            eq: vec!["z₁ = cos(θ/2)·e^(iψ)", "z₂ = sin(θ/2)·e^(i(ψ+φ))", "then project the 3-sphere to space"],
            note: "Circles in four dimensions, seen in three. Every circle is linked exactly once with every other.",
            params: vec![
                ParamDef::new("fibres", 12.0, 2.0, 30.0, 1.0),
                ParamDef::labelled("theta", "θ", 1.57, 0.5, 2.5, 0.01),
                ParamDef::labelled("nest", "shells", 1.0, 1.0, 3.0, 1.0),
            ],
            ..Preset::new(
                "hopf",
                "Hopf fibration",
                Shape::Curve(Box::new(|v, _o| {
                    let mut lines = Vec::new();
                    let n = round(v[0]);
                    let mut s = 0.0;
                    while s < round(v[2]) {
                        let th = v[1] * (1.0 - 0.3 * s);
                        let (ch, sh) = (cos(th / 2.0), sin(th / 2.0));
                        let mut k = 0.0;
                        while k < n {
                            let ph = TAU * (k + 0.5 * s) / n;
                            lines.push(ring(
                                |t| {
                                    let w = 1.0 - sh * sin(t + ph);
                                    [ch * cos(t) / w, sh * cos(t + ph) / w, ch * sin(t) / w]
                                },
                                500,
                            ));
                            k += 1.0;
                        }
                        s += 1.0;
                    }
                    closed(lines)
                })),
            )
        },
        Preset {
            animate: Some(vec!["amp"]),
            tube: Some(0.1),
            fixed_length: true,
            rot: Some([24.0, 20.0, 0.0]),
            eq: vec!["latitude = A sin(m·φ)", "on a sphere, φ once round"],
            note: "With m = 2 this is the seam of a tennis ball.",
            params: vec![ParamDef::new("m", 4.0, 1.0, 12.0, 1.0), ParamDef::labelled("amp", "A", 0.62, 0.1, 1.35, 0.01)],
            ..Preset::new(
                "seam",
                "Spherical wave",
                Shape::Curve(Box::new(|v, _o| {
                    closed(vec![ring(
                        |t| {
                            let la = v[1] * sin(round(v[0]) * t);
                            [cos(la) * cos(t), sin(la), cos(la) * sin(t)]
                        },
                        2000,
                    )])
                })),
            )
        },
        Preset {
            animate: Some(vec!["turns"]),
            tube: Some(0.075),
            fixed_length: true,
            rot: Some([20.0, 0.0, 0.0]),
            eq: vec!["y from −1 to 1", "φ = n·π·y on a sphere"],
            params: vec![ParamDef::labelled("turns", "n", 7.0, 2.0, 18.0, 0.5)],
            radius: Some(Box::new(|u| 0.3 + 0.7 * sin(PI * u))),
            ..Preset::new(
                "spiral",
                "Spherical spiral",
                Shape::Curve(Box::new(|v, _o| {
                    let n = 3000;
                    let mut a = Vec::with_capacity((n + 1) * 3);
                    for i in 0..=n {
                        let y = -0.985 + 1.97 * i as f64 / n as f64;
                        let r = sqrt(1.0 - y * y);
                        let ph = v[0] * PI * y;
                        a.extend_from_slice(&[r * cos(ph), y, r * sin(ph)]);
                    }
                    RawCurve { lines: vec![a], closed: false, ..Default::default() }
                })),
            )
        },
        Preset {
            animate: Some(vec!["decay"]),
            tube: Some(0.085),
            fixed_length: true,
            align: Some(true),
            eq: vec!["x(t) = Σ aₖ cos kt + bₖ sin kt", "with random aₖ, bₖ shrinking as k grows"],
            note: "A random smooth loop. Every seed is a different sculpture.",
            params: vec![
                ParamDef::new("seed", 3.0, 1.0, 300.0, 1.0),
                ParamDef::labelled("waves", "harmonics", 3.0, 2.0, 8.0, 1.0),
                ParamDef::new("decay", 1.5, 0.6, 2.5, 0.01),
            ],
            ..Preset::new(
                "fourier",
                "Fourier loop",
                Shape::Curve(Box::new(|v, _o| {
                    let mut r = rng(round(v[0]) * 7919.0);
                    let h = round(v[1]);
                    let mut amps: Vec<[f64; 6]> = Vec::new();
                    let mut k = 1.0;
                    while k <= h {
                        let s = pow(k, -v[2]);
                        // in this order: each takes the generator's next two numbers
                        let mut c = [0.0; 6];
                        for x in c.iter_mut() {
                            *x = gauss(&mut r) * s;
                        }
                        amps.push(c);
                        k += 1.0;
                    }
                    closed(vec![ring(
                        |t| {
                            let (mut x, mut y, mut z) = (0.0, 0.0, 0.0);
                            for (i, c) in amps.iter().enumerate() {
                                let (ck, sk) = (cos((i + 1) as f64 * t), sin((i + 1) as f64 * t));
                                x += c[0] * ck + c[1] * sk;
                                y += c[2] * ck + c[3] * sk;
                                z += c[4] * ck + c[5] * sk;
                            }
                            [x, y, z]
                        },
                        2400,
                    )])
                })),
            )
        },
    ]
}

fn light(az: f64, el: f64, w: f64, h: f64, rgb: [f64; 3], soft: f64) -> Light {
    Light { az, el, w, h, rgb, soft }
}

pub fn family() -> Family {
    Family {
        presets: presets(),
        melt: 0.004,
        defaults: vec![("absorb", Setting::Vec3([0.14, 0.07, 0.03])), ("rot", Setting::Vec3([20.0, 25.0, 0.0]))],
        lights: vec![
            light(20.0, 0.0, 5.0, 40.0, [5.0, 4.9, 4.7], 0.6),
            light(80.0, 0.0, 5.0, 40.0, [5.0, 1.6, 0.25], 0.6),
            light(140.0, 0.0, 5.0, 40.0, [0.5, 1.4, 4.0], 0.6),
            light(200.0, 0.0, 5.0, 40.0, [4.0, 3.9, 3.8], 0.6),
            light(260.0, 0.0, 5.0, 40.0, [5.0, 1.6, 0.25], 0.6),
            light(320.0, 0.0, 5.0, 40.0, [0.5, 1.4, 4.0], 0.6),
            light(0.0, 85.0, 30.0, 30.0, [0.6, 0.7, 0.9], 0.9),
        ],
        ..Family::new("knots", "Knots", Kind::Curve)
    }
}

#[cfg(test)]
mod tests {
    use super::rng;

    /// The seeded generator against the first numbers the web version's gives, for seeds of every kind:
    /// small, large, past 32 bits, negative, fractional (tests/data/curves_math.json, written by
    /// tools/dump_curves.mjs).
    #[test]
    fn the_seeded_generator_gives_the_web_versions_numbers() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/curves_math.json");
        let all: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let cases = all["rng"].as_array().unwrap();
        assert!(cases.len() >= 10);
        for case in cases {
            let seed: f64 = case["seed"].as_str().unwrap().parse().unwrap();
            let theirs: Vec<f64> = case["out"].as_str().unwrap().split(' ').map(|s| s.parse().unwrap()).collect();
            let mut next = rng(seed);
            let mine: Vec<f64> = theirs.iter().map(|_| next()).collect();
            assert_eq!(mine, theirs, "seed {seed}");
        }
    }
}
