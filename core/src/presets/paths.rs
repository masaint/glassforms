//! Paths: a track built piece by piece (`crate::path`), with a profile swept along it.
//!
//! The path is the shape's `model` (its text, `path::Path::text`), built in the right panel's path
//! builder and edited on the picture; the shapes of the family are the profiles swept along it: a round
//! tube, a flat strip, a square bar. The path stays when the profile changes. Each shape's sliders are the
//! profile's sizes, how it tapers from start to end, and how its open ends finish.
//!
//! A path is in the volume's own units (it spans -1.5..1.5), not fitted to the volume as other curves
//! are: adding a piece never moves or shrinks what is already built.

use super::geo::*;
use super::*;
use crate::path::{Path, Piece as Track, Profile};

pub const FAMILY: &str = "paths";
pub const TUBE: &str = "tubepath";
pub const STRIP: &str = "strippath";
pub const BAR: &str = "barpath";
pub const OUTLINE: &str = "outlinepath";

/// The outlines the outline shape can sweep, and the letters it can take one from.
const SHAPES: [&str; 7] = ["star", "heart", "cross", "hexagon", "gear", "rounded square", "letter"];
const LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789&@";
/// The most edges an outline is cut into (the baker looks at every one, for every voxel near the path).
const MOST_EDGES: usize = 72;

/// The names of a slider's settings, when it is a choice.
pub fn choices(id: &str) -> Option<Vec<String>> {
    match id {
        "caps" => Some(vec!["flat".to_owned(), "rounded".to_owned()]),
        "shape" => Some(SHAPES.iter().map(|s| s.to_string()).collect()),
        _ => None,
    }
}

/// An outline as edges (x0, y0, x1, y1) in a square from -1 to 1, centred: one of `SHAPES`, the letter
/// taken from an installed font as the 3D text takes it (`text3d`), cut into at most `MOST_EDGES` edges.
pub fn outline_of(shape: f64, letter: f64) -> std::sync::Arc<Vec<[f64; 4]>> {
    use std::f64::consts::TAU;
    let ring = |n: usize, r: &dyn Fn(f64) -> f64| -> Vec<[f64; 2]> {
        (0..n).map(|i| { let a = i as f64 / n as f64 * TAU + TAU / 4.0; let k = r(a); [k * a.cos(), k * a.sin()] }).collect()
    };
    let polys: Vec<Vec<[f64; 2]>> = match shape.round().clamp(0.0, (SHAPES.len() - 1) as f64) as usize {
        0 => vec![(0..10).map(|i| { let a = i as f64 / 10.0 * TAU + TAU / 4.0; let k = if i % 2 == 0 { 1.0 } else { 0.42 }; [k * a.cos(), k * a.sin()] }).collect()],
        1 => vec![(0..64).map(|i| {
            // the heart curve: x = 16 sin^3 t, y = 13 cos t - 5 cos 2t - 2 cos 3t - cos 4t
            let t = i as f64 / 64.0 * TAU;
            [16.0 * t.sin().powi(3), 13.0 * t.cos() - 5.0 * (2.0 * t).cos() - 2.0 * (3.0 * t).cos() - (4.0 * t).cos()]
        }).collect()],
        2 => vec![vec![[-0.3, -1.0], [0.3, -1.0], [0.3, -0.3], [1.0, -0.3], [1.0, 0.3], [0.3, 0.3], [0.3, 1.0], [-0.3, 1.0], [-0.3, 0.3], [-1.0, 0.3], [-1.0, -0.3], [-0.3, -0.3]]],
        3 => vec![ring(6, &|_| 1.0)],
        4 => vec![ring(60, &|a| if ((a * 10.0 / TAU).rem_euclid(1.0)) < 0.5 { 1.0 } else { 0.78 }), ring(24, &|_| 0.3)],
        5 => {
            // a square with its corners rounded by a quarter of its half-width (a plate, a tile, a bar)
            let (k, r) = (1.0, 0.25);
            let mut v = Vec::new();
            for (cx, cy, a0) in [(k - r, k - r, 0.0), (r - k, k - r, 0.25), (r - k, r - k, 0.5), (k - r, r - k, 0.75)] {
                for i in 0..=12 {
                    let a = (a0 + 0.25 * i as f64 / 12.0) * TAU;
                    v.push([cx + r * a.cos(), cy + r * a.sin()]);
                }
            }
            vec![v]
        }
        _ => {
            let c = LETTERS.chars().nth(letter.round().clamp(0.0, (LETTERS.chars().count() - 1) as f64) as usize).unwrap_or('A');
            match super::text3d::font(0) {
                Some(f) => f.outline(f.glyph(c)),
                None => vec![ring(4, &|_| 1.0)],
            }
        }
    };
    // centred, scaled into the square, and thinned to at most MOST_EDGES edges in all
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for q in polys.iter().flatten() {
        for k in 0..2 { lo[k] = lo[k].min(q[k]); hi[k] = hi[k].max(q[k]); }
    }
    let (c, h) = ([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])], (0.5 * (hi[0] - lo[0])).max(0.5 * (hi[1] - lo[1])).max(1e-9));
    let total: usize = polys.iter().map(|p| p.len()).sum();
    let every = total.div_ceil(MOST_EDGES).max(1);
    let mut edges = Vec::new();
    for poly in &polys {
        let kept: Vec<[f64; 2]> = poly.iter().step_by(every).map(|q| [(q[0] - c[0]) / h, (q[1] - c[1]) / h]).collect();
        if kept.len() < 3 { continue; }
        for i in 0..kept.len() {
            let (a, b) = (kept[i], kept[(i + 1) % kept.len()]);
            edges.push([a[0], a[1], b[0], b[1]]);
        }
    }
    std::sync::Arc::new(edges)
}

/// The path a shape starts with before one is built: an S of tube lying flat with an upturned U at its far end, a strip
/// folded into a heart, a square ring for the bar.
pub fn default_path(preset: &str) -> Path {
    let text = match preset {
        STRIP => "start 0.18 -0.15 0.3 180 0 0; S 0.32; T -180 0.13; S 0.12; T -45 0.04; S 0.42; T 135 0.05; S 0.3; T 90 0.05; S 0.36; T 135 0.05; S 0.15",
        BAR => "start -0.45 0 0.55 0 0 0; S 0.7; T 90 0.2; S 0.8; T 90 0.2; S 0.7; T 90 0.2; S 0.8; T 90 0.2",
        _ => "start -0.1 -0.2 0.2 180 0 0; S 0.3; T 180 0.12; S 0.15; T -45 0.22; T 90 0.22; T -45 0.22; S 0.08; P 90 0.09; S 0.22; P 180 0.09; S 0.3",
    };
    // (drawn small, then made as large as the shapes of other families stand)
    let mut p = Path::parse(text).unwrap_or_default().scaled(DEFAULT_SCALE);
    p.centre(true);
    p
}
/// How much larger the default paths are than they are written.
const DEFAULT_SCALE: f64 = 1.6;

/// The path of a shape as its text stands (the default when it says nothing).
pub fn path_of(preset: &str, text: &str) -> Path {
    if text.trim().is_empty() { default_path(preset) } else { Path::parse(text).unwrap_or_else(|| default_path(preset)) }
}

/// Where the sliders of each shape are.
const T_SIZE: usize = 0;
const T_START: usize = 1;
const T_END: usize = 2;
const T_CAPS: usize = 3;

fn params(preset: &str) -> Vec<ParamDef> {
    let ends = |caps: f64| vec![
        ParamDef::labelled("start", "size at start", 1.0, 0.05, 2.5, 0.01),
        ParamDef::labelled("end", "size at end", 1.0, 0.05, 2.5, 0.01),
        ParamDef::labelled("caps", "ends", caps, 0.0, 1.0, 1.0),
    ];
    let mut v = match preset {
        STRIP => vec![
            ParamDef::labelled("width", "width", 0.48, 0.01, 1.6, 0.005),
            ParamDef::labelled("thick", "thickness", 0.055, 0.004, 0.4, 0.001),
            ParamDef::labelled("edge", "edge rounding", 1.0, 0.0, 1.0, 0.01),
            ParamDef::labelled("turn", "stands at °", 90.0, -180.0, 180.0, 1.0),
        ],
        BAR => vec![
            ParamDef::labelled("width", "width", 0.2, 0.01, 1.6, 0.005),
            ParamDef::labelled("height", "height", 0.2, 0.01, 1.6, 0.005),
            ParamDef::labelled("edge", "corner rounding", 0.25, 0.0, 1.0, 0.01),
            ParamDef::labelled("turn", "turned °", 0.0, -180.0, 180.0, 1.0),
        ],
        OUTLINE => vec![
            ParamDef::labelled("shape", "outline", 0.0, 0.0, (SHAPES.len() - 1) as f64, 1.0),
            ParamDef::labelled("size", "size", 0.3, 0.02, 1.2, 0.005),
            ParamDef::labelled("aspect", "height / width", 1.0, 0.2, 5.0, 0.01),
            ParamDef::labelled("turn", "turned °", 0.0, -180.0, 180.0, 1.0),
        ],
        _ => vec![ParamDef::labelled("size", "diameter", 0.16, 0.008, 0.6, 0.002)],
    };
    v.extend(ends(if preset == TUBE { 1.0 } else { 0.0 }));
    if preset == OUTLINE {
        // (an outline's ends are cut flat: in the place of the ends' choice, the letter, when the outline is one)
        v.pop();
        v.push(ParamDef::labelled("letter", "letter", 0.0, 0.0, (LETTERS.chars().count() - 1) as f64, 1.0));
    }
    v
}

/// The profile a shape's sliders describe.
pub fn profile(preset: &str, v: &[f64]) -> Profile {
    let g = |i: usize, d: f64| v.get(i).copied().unwrap_or(d);
    match preset {
        STRIP | BAR => Profile {
            w: g(0, 0.3), h: g(1, 0.04), round: g(2, 1.0), turn: g(3, 0.0),
            start: g(4, 1.0), end: g(5, 1.0), round_caps: g(6, 0.0) >= 0.5, outline: None,
        },
        OUTLINE => Profile {
            w: g(1, 0.3), h: g(1, 0.3) * g(2, 1.0), round: 0.0, turn: g(3, 0.0),
            start: g(4, 1.0), end: g(5, 1.0), round_caps: false, outline: Some(outline_of(g(0, 0.0), g(6, 0.0))),
        },
        _ => Profile { w: g(T_SIZE, 0.1), h: g(T_SIZE, 0.1), round: 1.0, turn: 0.0, start: g(T_START, 1.0), end: g(T_END, 1.0), round_caps: g(T_CAPS, 1.0) >= 0.5, outline: None },
    }
}

/// The longest stretch of a primitive the baker draws into one box.
const BAKE_STRETCH: f64 = 0.06;

/// The solid for a path and a profile.
pub fn build(path: &Path, pr: &Profile) -> RawCurve {
    let mut p = path.clone();
    // (no pieces yet: a stub at the start, so that where it starts shows)
    if p.pieces.is_empty() { p.pieces.push(Track::Straight { len: 0.5 * pr.w.max(pr.h).max(0.02) }); }
    // one piece of solid a primitive, each the primitive's exact distance (`path_formula`) within its box
    let b = crate::path_formula::build(&p, pr.turn, pr.round_caps);
    // (drawn a short stretch at a time: each stretch's box close round it, so that the baker's far field, which
    // stands each box's points at its middle, stays close to the truth)
    let pieces = b.prims.iter().flat_map(|prim| {
        let rec = crate::path_formula::record(prim, pr, b.length);
        crate::path_formula::bounds_along(prim, pr, b.length, BAKE_STRETCH).into_iter().map(move |(lo, hi)| Piece { lo, hi, terms: vec![Term::Path { rec, outline: pr.outline.clone() }] })
    }).collect();
    // Placed as it is: the volume's own units, centred on nothing and not fitted (one point at the fitting
    // distance gives a scale of one).
    RawCurve { pieces, frame: vec![crate::FIT, 0.0, 0.0], ..Default::default() }
}

/// A path part's block of a tree's table (`path_formula::table`): from its text and its shape's sliders.
pub fn table_of(preset: &str, text: &str, vals: &[f64]) -> Vec<f32> {
    let pr = profile(preset, vals);
    let mut p = path_of(preset, text);
    if p.pieces.is_empty() { p.pieces.push(Track::Straight { len: 0.5 * pr.w.max(pr.h).max(0.02) }); }
    crate::path_formula::table(&crate::path_formula::build(&p, pr.turn, pr.round_caps), &pr)
}

fn shape(preset: &'static str) -> Shape {
    Shape::Curve(Box::new(move |v, o| build(&path_of(preset, o.text), &profile(preset, v))))
}

pub fn family() -> Family {
    let preset = |id: &'static str, name: &'static str, eq: Vec<&'static str>, note: &'static str| Preset {
        eq,
        note,
        params: params(id),
        fixed_length: true,
        centre: false,
        tube: Some(0.05),
        rot: Some([32.0, -28.0, 0.0]),
        animate: Some(vec![]),
        ..Preset::new(id, name, shape(id))
    };
    Family {
        presets: vec![
            preset(TUBE, "Tube", vec!["a circle swept along the path", "pieces: straight · turn · slope · bank · loop · curve"], "Build the path piece by piece in the builder below; drag its joints and handles on the picture."),
            preset(STRIP, "Strip", vec!["a flat strip swept along the path", "turned by the track's roll and its own stand"], "A flat band, standing on its edge or lying flat, its edges rounded. Build the path in the builder below."),
            preset(BAR, "Bar", vec!["a rounded rectangle swept along the path"], "A square or oblong bar, its corners rounded as far as you like. Build the path in the builder below."),
            preset(OUTLINE, "Outline", vec!["a flat outline swept along the path", "star · heart · cross · hexagon · gear · rounded square · a letter"], "Any flat outline drawn along the path, filled even-odd as the 3D text's letters are: holes stay holes. Its ends are cut flat."),
        ],
        // stretches meet exactly: a hard union
        melt: 0.0,
        lights: solid_lights(),
        defaults: vec![
            ("ior", Setting::Num(1.5)), ("absorb", Setting::Vec3([0.16, 0.08, 0.035])),
            ("dist", Setting::Num(6.0)), ("fov", Setting::Num(26.0)), ("rot", Setting::Vec3([32.0, -28.0, 0.0])),
        ],
        ..Family::new(FAMILY, "Paths", Kind::Curve)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_default_path_builds() {
        for id in [TUBE, STRIP, BAR, OUTLINE] {
            let p = default_path(id);
            assert!(!p.pieces.is_empty(), "{id}");
            let v: Vec<f64> = params(id).iter().map(|d| d.value).collect();
            let raw = build(&p, &profile(id, &v));
            assert!(raw.pieces.len() >= p.pieces.len(), "{id}: {}", raw.pieces.len());
            // it stays inside the volume
            for pc in &raw.pieces {
                for k in 0..3 { assert!(pc.lo[k] > -crate::HB && pc.hi[k] < crate::HB, "{id} {:?} {:?}", pc.lo, pc.hi); }
            }
        }
        // the ring of the bar closes
        assert!(default_path(BAR).walk().closed);
    }

    #[test]
    fn every_outline_is_closed_and_in_its_square() {
        for k in 0..SHAPES.len() {
            for letter in [0.0, 1.0, 26.0] {
                let e = outline_of(k as f64, letter);
                assert!(e.len() >= 3 && e.len() <= MOST_EDGES + 8, "{k}: {} edges", e.len());
                for x in e.iter() { for v in x { assert!(v.abs() <= 1.0 + 1e-9, "{k}: {x:?}"); } }
                // every point an edge starts at is where another ends: closed outlines
                for x in e.iter() { assert!(e.iter().any(|y| y[2] == x[0] && y[3] == x[1])); }
            }
        }
    }

    #[test]
    fn an_empty_path_is_a_stub() {
        let v: Vec<f64> = params(TUBE).iter().map(|d| d.value).collect();
        let raw = build(&Path::default(), &profile(TUBE, &v));
        assert!(!raw.pieces.is_empty());
    }
}

#[cfg(test)]
mod timing {
    /// How long a path takes to walk and sweep (cargo test --release -p gf-core paths_time -- --ignored --nocapture).
    #[test]
    #[ignore]
    fn paths_time() {
        for id in [super::TUBE, super::STRIP, super::BAR] {
            let v: Vec<f64> = super::params(id).iter().map(|d| d.value).collect();
            let p = super::default_path(id);
            let t = std::time::Instant::now();
            let raw = super::build(&p, &super::profile(id, &v));
            println!("{id:10} {:7.2} ms  {} stretches", t.elapsed().as_secs_f64() * 1e3, raw.pieces.len());
        }
    }
}

#[cfg(test)]
mod lists {
    /// How many primitives the cells of each default path list (cargo test --release -p gf-core paths_lists -- --ignored --nocapture).
    #[test]
    #[ignore]
    fn paths_lists() {
        for id in [super::TUBE, super::STRIP, super::BAR, super::OUTLINE] {
            let v: Vec<f64> = super::params(id).iter().map(|d| d.value).collect();
            let t = super::table_of(id, "", &v);
            let n = crate::path_formula::CELLS.pow(3);
            let counts: Vec<f32> = (0..n).map(|c| t[36 * (1 + c)]).collect();
            let all = counts.iter().filter(|c| **c < 0.0).count();
            let mean = counts.iter().filter(|c| **c >= 0.0).sum::<f32>() / (n - all).max(1) as f32;
            println!("{id:12} prims {} cells listing all {all}, mean listed {mean:.1}", t[7]);
        }
    }
}
