//! Strange attractors, blown in glass: one long orbit of a chaotic flow, thickened into a tube.
//! Where neighbouring loops pass close together they flow into each other, like blown glass.
//! (scenes/attractors.js. The velocities are written exactly as they are there, term for term: these
//! flows are chaotic, and a sum taken in another order is, a few thousand steps on, another shape.)

use super::*;
use crate::curve::jsmath::sin;

/// Steps run and thrown away before the orbit is drawn, for a flow that does not say.
const SETTLE: usize = 3000;

/// A flow preset: f gives the velocity at a point. steps x dt is how much of the orbit is drawn.
/// (align: true turns the shape face-on automatically; otherwise it is shown as the equations give it.)
#[allow(clippy::too_many_arguments)]
fn flow(id: &'static str, name: &'static str, tube: f64, start: [f64; 3], dt: f64, steps: usize, settle: usize, f: FlowFn) -> Preset {
    Preset { tube: Some(tube), ..Preset::new(id, name, Shape::Flow(Flow { start, dt, steps, settle, f })) }
}

/// For the Rossler system: how many trips round before the orbit repeats (chaos = never).
fn period(raw: &RawCurve, _v: &[f64]) -> String {
    let Some(l) = raw.lines.first() else { return String::new() };
    let mut peaks = Vec::new();
    let mut i = 3;
    while i + 3 < l.len() {
        let (a, b, c) = (l[i - 3], l[i], l[i + 3]);
        if b > a && b > c { peaks.push(b - 0.125 * (c - a) * (c - a) / (a - 2.0 * b + c)); }
        i += 3;
    }
    for k in 1..=16 {
        let mut same = peaks.len() > 2 * k;
        let mut i = 0;
        while same && i + k < peaks.len() {
            same = (peaks[i + k] - peaks[i]).abs() < 0.02;
            i += 1;
        }
        if same { return format!("period {k}"); }
    }
    "chaos".to_string()
}

/// Thomas' system at several strengths of friction b. Less friction, more wandering.
fn thomas(id: &'static str, name: &'static str, b: f64, steps: usize, tube: f64, note: &'static str) -> Preset {
    Preset {
        eq: vec!["ẋ = sin y − b·x", "ẏ = sin z − b·y", "ż = sin x − b·z"],
        note,
        params: vec![ParamDef::new("b", b, 0.08, 0.33, 0.0005)],
        ..flow(id, name, tube, [1.1, 1.1, -0.01], 0.05, steps, SETTLE, |x, y, z, v| [sin(y) - v[0] * x, sin(z) - v[0] * y, sin(x) - v[0] * z])
    }
}

fn presets() -> Vec<Preset> {
    vec![
        thomas("thomas", "Thomas", 0.208186, 9000, 0.036, "Cyclically symmetric: swap x→y→z→x and nothing changes. b is friction."),
        thomas("thomas19", "Thomas · linked", 0.19, 9000, 0.034, "A little less friction and the orbit reaches into neighbouring cells: rings through rings."),
        thomas("thomas15", "Thomas · woven", 0.15, 9000, 0.034, "Less friction again."),
        thomas("thomas10", "Thomas · tangle", 0.10, 9000, 0.03, "Almost frictionless: the orbit wanders a three-dimensional lattice."),
        Preset {
            eq: vec!["ẋ = y", "ẏ = −x + y·z", "ż = a − y²"],
            note: "A thermostat for a single oscillator. No friction overall, so it never thins down to a sheet.",
            params: vec![ParamDef::new("a", 1.5, 0.6, 3.0, 0.01)],
            ..flow("nosehoover", "Nosé–Hoover", 0.024, [1.0, 0.0, 0.0], 0.005, 24000, SETTLE, |x, y, z, v| [y, -x + y * z, v[0] - y * y])
        },
        Preset {
            eq: vec!["ẋ = y + a·x·y + x·z", "ẏ = 1 − b·x² + y·z", "ż = x − x² − y²"],
            params: vec![ParamDef::new("a", 2.07, 1.9, 2.2, 0.002)],
            ..flow("sprott", "Sprott", 0.028, [0.63, 0.47, -0.54], 0.004, 12000, SETTLE, |x, y, z, v| [y + v[0] * x * y + x * z, 1.0 - 1.79 * x * x + y * z, x - x * x - y * y])
        },
        Preset {
            eq: vec!["ẋ = a·x + y·z", "ẏ = b·x + c·y − x·z", "ż = −z − x·y"],
            params: vec![ParamDef::new("a", 0.2, 0.1, 0.3, 0.002)],
            ..flow("fourwing", "Four-wing", 0.03, [1.3, -0.18, 0.01], 0.01, 16000, SETTLE, |x, y, z, v| [v[0] * x + y * z, 0.01 * x - 0.4 * y - x * z, -z - x * y])
        },
        Preset {
            eq: vec!["ẋ = a·x + c·y·z", "ẏ = b·x + d·y − x·z", "ż = e·z + f·x·y"],
            params: vec![ParamDef::new("a", 0.2, 0.1, 0.3, 0.002)],
            ..flow("wangsun", "Wang–Sun", 0.028, [0.5, 0.1, 0.1], 0.01, 20000, SETTLE, |x, y, z, v| [v[0] * x + y * z, -0.01 * x - 0.4 * y - x * z, -z - x * y])
        },
        Preset {
            eq: vec!["ẋ = −a·x − 4y − 4z − y²", "ẏ = −a·y − 4z − 4x − z²", "ż = −a·z − 4x − 4y − x²"],
            note: "Cyclically symmetric, like Thomas.",
            params: vec![ParamDef::new("a", 1.4, 1.27, 1.9, 0.005)],
            ..flow("halvorsen", "Halvorsen", 0.03, [-5.0, 0.0, 0.0], 0.004, 8000, SETTLE, |x, y, z, v| {
                [-v[0] * x - 4.0 * y - 4.0 * z - y * y, -v[0] * y - 4.0 * z - 4.0 * x - z * z, -v[0] * z - 4.0 * x - 4.0 * y - x * x]
            })
        },
        Preset {
            axes: Some([0, 2, 1]),
            eq: vec!["ẋ = (z − b)x − d·y", "ẏ = d·x + (z − b)y", "ż = c + a·z − z³/3 − (x² + y²)(1 + e·z) + f·z·x³"],
            params: vec![ParamDef::new("e", 0.25, 0.0, 0.6, 0.005)],
            ..flow("aizawa", "Aizawa", 0.028, [0.1, 0.0, 0.0], 0.01, 6000, SETTLE, |x, y, z, v| {
                [(z - 0.7) * x - 3.5 * y, 3.5 * x + (z - 0.7) * y, 0.6 + 0.95 * z - z * z * z / 3.0 - (x * x + y * y) * (1.0 + v[0] * z) + 0.1 * z * x * x * x]
            })
        },
        Preset {
            align: Some(true),
            eq: vec!["ẋ = σ(y − x)", "ẏ = x(ρ − z) − y", "ż = x·y − β·z"],
            note: "The butterfly: σ = 10, β = 8/3.",
            params: vec![ParamDef::labelled("rho", "ρ", 28.0, 20.0, 60.0, 0.1)],
            ..flow("lorenz", "Lorenz", 0.028, [1.0, 1.0, 20.0], 0.004, 8000, SETTLE, |x, y, z, v| [10.0 * (y - x), x * (v[0] - z) - y, x * y - 8.0 / 3.0 * z])
        },
        Preset {
            axes: Some([0, 2, 1]),
            rot: Some([38.0, 22.0, 0.0]),
            eq: vec!["ẋ = −y − z", "ẏ = x + a·y", "ż = b + z(x − c)"],
            note: "Raise c and one loop becomes two, four, eight … then chaos, with windows of order inside it.",
            params: vec![ParamDef::new("c", 5.7, 2.4, 6.2, 0.005)],
            info: Some(Box::new(period)),
            ..flow("rossler", "Rössler", 0.022, [1.0, 1.0, 0.0], 0.01, 26000, 300000, |x, y, z, v| [-y - z, x + 0.2 * y, 0.2 + z * (x - v[0])])
        },
        Preset {
            align: Some(true),
            eq: vec!["ẋ = y − a·x + b·y·z", "ẏ = c·y − x·z + z", "ż = d·x·y − e·z"],
            params: vec![ParamDef::new("c", 1.7, 1.2, 2.2, 0.01)],
            ..flow("dadras", "Dadras", 0.03, [1.1, 2.1, -2.0], 0.004, 9000, SETTLE, |x, y, z, v| [y - 3.0 * x + 2.7 * y * z, v[0] * y - x * z + z, 2.0 * x * y - 9.0 * z])
        },
        Preset {
            eq: vec!["ẋ = a·x − y·z", "ẏ = b·y + x·z", "ż = c·z + x·y/3"],
            params: vec![ParamDef::new("c", -0.38, -0.6, -0.2, 0.002)],
            ..flow("chenlee", "Chen–Lee", 0.028, [1.0, 0.0, 4.5], 0.002, 12000, SETTLE, |x, y, z, v| [5.0 * x - y * z, -10.0 * y + x * z, v[0] * z + x * y / 3.0])
        },
        Preset {
            align: Some(true),
            eq: vec!["ẋ = −s(x + y)", "ẏ = −y − s·x·z", "ż = s·x·y + v"],
            params: vec![ParamDef::new("v", 4.272, 2.5, 6.0, 0.01)],
            ..flow("burkeshaw", "Burke–Shaw", 0.028, [0.6, 0.0, 0.0], 0.002, 9000, SETTLE, |x, y, z, p| [-10.0 * (x + y), -y - 10.0 * x * z, 10.0 * x * y + p[0]])
        },
        Preset {
            align: Some(true),
            eq: vec!["ẋ = y", "ẏ = (1 − z)x − a·y", "ż = x² − b·z"],
            params: vec![ParamDef::new("a", 0.75, 0.6, 1.1, 0.005)],
            ..flow("shimizu", "Shimizu–Morioka", 0.028, [0.1, 0.0, 0.0], 0.01, 12000, SETTLE, |x, y, z, v| [y, (1.0 - z) * x - v[0] * y, x * x - 0.45 * z])
        },
        Preset {
            align: Some(true),
            eq: vec!["ẋ = −y² − z² − a·x + a·F", "ẏ = x·y − b·x·z − y + G", "ż = b·x·y + x·z − z"],
            note: "A toy model of the atmosphere’s circulation.",
            params: vec![ParamDef::new("F", 8.0, 6.0, 10.0, 0.02)],
            ..flow("hadley", "Hadley", 0.026, [0.39, -1.0, 0.0], 0.005, 16000, SETTLE, |x, y, z, v| {
                [-y * y - z * z - 0.2 * x + 0.2 * v[0], x * y - 4.0 * x * z - y + 1.0, 4.0 * x * y + x * z - z]
            })
        },
        Preset {
            eq: vec!["ẋ = y(z − 1 + x²) + γ·x", "ẏ = x(3z + 1 − x²) + γ·y", "ż = −2z(α + x·y)"],
            params: vec![ParamDef::labelled("gamma", "γ", 0.87, 0.75, 0.95, 0.002)],
            ..flow("rabfab", "Rabinovich–Fabrikant", 0.026, [-1.0, 0.0, 0.5], 0.0015, 30000, SETTLE, |x, y, z, v| {
                [y * (z - 1.0 + x * x) + v[0] * x, x * (3.0 * z + 1.0 - x * x) + v[0] * y, -2.0 * z * (1.1 + x * y)]
            })
        },
        Preset {
            eq: vec!["ẋ = x(4 − y) + α·z", "ẏ = −y(1 − x²)", "ż = −x(1.5 − s·z) − 0.05z"],
            params: vec![ParamDef::labelled("alpha", "α", 0.3, 0.2, 0.5, 0.002)],
            ..flow("bouali", "Bouali", 0.028, [1.0, 1.0, 0.0], 0.003, 20000, SETTLE, |x, y, z, v| [x * (4.0 - y) + v[0] * z, -y * (1.0 - x * x), -x * (1.5 - z) - 0.05 * z])
        },
        Preset {
            eq: vec!["ẋ = −x + y + y·z", "ẏ = −x − y + a·x·z", "ż = z − b·x·y"],
            params: vec![ParamDef::new("a", 0.4, 0.3, 0.6, 0.002)],
            ..flow("sakarya", "Sakarya", 0.026, [1.0, -1.0, 1.0], 0.002, 20000, SETTLE, |x, y, z, v| [-x + y + y * z, -x - y + v[0] * x * z, z - 0.3 * x * y])
        },
        Preset {
            align: Some(true),
            eq: vec!["ẋ = y", "ẏ = z", "ż = −c·x − b·y − a·z + x²"],
            params: vec![ParamDef::new("a", 0.44, 0.4, 0.6, 0.002)],
            ..flow("genesio", "Genesio–Tesi", 0.028, [0.1, 0.1, 0.1], 0.01, 14000, SETTLE, |x, y, z, v| [y, z, -x - 1.1 * y - v[0] * z + x * x])
        },
        Preset {
            align: Some(true),
            eq: vec!["ẋ = a·y·z", "ẏ = x − b·y", "ż = c − x·y"],
            note: "One of the simplest flows that is chaotic at all: five terms, two of them nonlinear.",
            params: vec![ParamDef::new("a", 0.4, 0.3, 0.6, 0.002)],
            ..flow("sprottb", "Sprott B", 0.03, [0.1, 0.1, 0.1], 0.01, 14000, SETTLE, |x, y, z, v| [v[0] * y * z, x - 1.2 * y, 1.0 - x * y])
        },
        Preset {
            align: Some(true),
            eq: vec!["ẋ = y", "ẏ = z", "ż = −a·x − b·y − z + c·x³"],
            params: vec![ParamDef::new("b", 3.5, 3.0, 4.2, 0.005)],
            ..flow("arneodo", "Arneodo", 0.028, [0.2, 0.2, -0.75], 0.005, 12000, 30000, |x, y, z, v| [y, z, 5.5 * x - v[0] * y - z - x * x * x])
        },
        Preset {
            align: Some(true),
            eq: vec!["ẋ = −a·x + y + 10y·z", "ẏ = −x − 0.4y + 5x·z", "ż = b·z − 5x·y"],
            params: vec![ParamDef::new("b", 0.175, 0.1, 0.25, 0.001)],
            ..flow("newton", "Newton–Leipnik", 0.026, [0.349, 0.0, -0.16], 0.005, 20000, SETTLE, |x, y, z, v| {
                [-0.4 * x + y + 10.0 * y * z, -x - 0.4 * y + 5.0 * x * z, v[0] * z - 5.0 * x * y]
            })
        },
        Preset {
            eq: vec!["ẋ = −κ·x + λ·y − y·z", "ẏ = x", "ż = −z + y²"],
            params: vec![ParamDef::labelled("lambda", "λ", 6.7, 5.0, 9.0, 0.02)],
            ..flow("rucklidge", "Rucklidge", 0.028, [1.0, 0.0, 4.5], 0.005, 10000, SETTLE, |x, y, z, v| [-2.0 * x + v[0] * y - y * z, x, -z + y * y])
        },
        Preset {
            align: Some(true),
            eq: vec!["ẋ = a(y − x) + d·x·z", "ẏ = b·x − x·z + f·y", "ż = c·z + x·y − e·x²"],
            params: vec![ParamDef::new("f", 14.7, 12.0, 16.0, 0.02)],
            ..flow("tsucs", "Three-scroll", 0.026, [-0.29, -0.25, -0.59], 0.0002, 30000, SETTLE, |x, y, z, v| {
                [32.48 * (y - x) + 0.13 * x * z, 45.84 * x - x * z + v[0] * y, 1.18 * z + x * y - 0.57 * x * x]
            })
        },
        Preset {
            eq: vec!["ẋ = a(y − x) + d·x·z", "ẏ = k·x + f·y − x·z", "ż = c·z + x·y − e·x²"],
            params: vec![ParamDef::new("k", 55.0, 45.0, 65.0, 0.1)],
            ..flow("dequanli", "Dequan Li", 0.026, [0.349, 0.0, -0.16], 0.0002, 30000, SETTLE, |x, y, z, v| {
                [40.0 * (y - x) + 0.16 * x * z, v[0] * x + 20.0 * y - x * z, 1.833 * z + x * y - 0.65 * x * x]
            })
        },
        Preset {
            eq: vec!["ẋ = y + z", "ẏ = −x + a·y", "ż = x² − z"],
            params: vec![ParamDef::new("a", 0.5, 0.4, 0.55, 0.002)],
            ..flow("sprottf", "Sprott F", 0.03, [0.1, 0.1, 0.1], 0.01, 14000, SETTLE, |x, y, z, v| [y + z, -x + v[0] * y, x * x - z])
        },
    ]
}

fn light(az: f64, el: f64, w: f64, h: f64, rgb: [f64; 3], soft: f64) -> Light {
    Light { az, el, w, h, rgb, soft }
}

pub fn family() -> Family {
    Family {
        presets: presets(),
        // A ring of tall strips all the way round, so a tube pointing any direction catches something:
        // white, warm orange, cool blue.
        lights: vec![
            light(20.0, 0.0, 5.0, 40.0, [5.0, 4.9, 4.7], 0.6),
            light(80.0, 0.0, 5.0, 40.0, [5.0, 1.6, 0.25], 0.6),
            light(140.0, 0.0, 5.0, 40.0, [0.5, 1.4, 4.0], 0.6),
            light(200.0, 0.0, 5.0, 40.0, [4.0, 3.9, 3.8], 0.6),
            light(260.0, 0.0, 5.0, 40.0, [5.0, 1.6, 0.25], 0.6),
            light(320.0, 0.0, 5.0, 40.0, [0.5, 1.4, 4.0], 0.6),
            light(0.0, 85.0, 30.0, 30.0, [0.6, 0.7, 0.9], 0.9),
        ],
        ..Family::new("attractors", "Attractors", Kind::Curve)
    }
}
