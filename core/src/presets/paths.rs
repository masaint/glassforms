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
use crate::path::{Path, Piece, Profile};

pub const FAMILY: &str = "paths";
pub const TUBE: &str = "tubepath";
pub const STRIP: &str = "strippath";
pub const BAR: &str = "barpath";

/// The names of a slider's settings, when it is a choice.
pub fn choices(id: &str) -> Option<Vec<String>> {
    (id == "caps").then(|| vec!["flat".to_owned(), "rounded".to_owned()])
}

/// The path a shape starts with before one is built: an S of tube with a hook at either end, a strip
/// folded into a heart, a square ring for the bar.
pub fn default_path(preset: &str) -> Path {
    let text = match preset {
        STRIP => "start 0.18 -0.15 0.3 180 0 0; S 0.32; T -180 0.13; S 0.12; T -45 0.04; S 0.42; T 135 0.05; S 0.3; T 90 0.05; S 0.36; T 135 0.05; S 0.15",
        BAR => "start -0.45 0 0.55 0 0 0; S 0.7; T 90 0.2; S 0.8; T 90 0.2; S 0.7; T 90 0.2; S 0.8; T 90 0.2",
        _ => "start -0.3 -0.2 0.25 180 0 0; S 0.25; T -180 0.13; S 0.15; T 40 0.2; T -80 0.22; S 0.1; T 80 0.2; S 0.2; T -40 0.2; P 90 0.06; S 0.1; P 180 0.11; S 0.1",
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
        _ => vec![ParamDef::labelled("size", "diameter", 0.16, 0.008, 0.6, 0.002)],
    };
    v.extend(ends(if preset == TUBE { 1.0 } else { 0.0 }));
    v
}

/// The profile a shape's sliders describe.
pub fn profile(preset: &str, v: &[f64]) -> Profile {
    let g = |i: usize, d: f64| v.get(i).copied().unwrap_or(d);
    match preset {
        STRIP | BAR => Profile {
            w: g(0, 0.3), h: g(1, 0.04), round: g(2, 1.0), turn: g(3, 0.0),
            start: g(4, 1.0), end: g(5, 1.0), round_caps: g(6, 0.0) >= 0.5,
        },
        _ => Profile { w: g(T_SIZE, 0.1), h: g(T_SIZE, 0.1), round: 1.0, turn: 0.0, start: g(T_START, 1.0), end: g(T_END, 1.0), round_caps: g(T_CAPS, 1.0) >= 0.5 },
    }
}

/// The solid for a path and a profile.
pub fn build(path: &Path, pr: &Profile) -> RawCurve {
    let mut p = path.clone();
    // (no pieces yet: a stub at the start, so that where it starts shows)
    if p.pieces.is_empty() { p.pieces.push(Piece::Straight { len: 0.5 * pr.w.max(pr.h).max(0.02) }); }
    let walk = p.walk();
    let pieces = crate::path::sweep(&walk, pr);
    // Placed as it is: the volume's own units, centred on nothing and not fitted (one point at the fitting
    // distance gives a scale of one).
    RawCurve { pieces, frame: vec![crate::FIT, 0.0, 0.0], ..Default::default() }
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
        for id in [TUBE, STRIP, BAR] {
            let p = default_path(id);
            assert!(!p.pieces.is_empty(), "{id}");
            let v: Vec<f64> = params(id).iter().map(|d| d.value).collect();
            let raw = build(&p, &profile(id, &v));
            assert!(raw.pieces.len() > 10, "{id}: {}", raw.pieces.len());
            // it stays inside the volume
            for pc in &raw.pieces {
                for k in 0..3 { assert!(pc.lo[k] > -crate::HB && pc.hi[k] < crate::HB, "{id} {:?} {:?}", pc.lo, pc.hi); }
            }
        }
        // the ring of the bar closes
        assert!(default_path(BAR).walk().closed);
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
