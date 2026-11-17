//! The curve maths against the web version's, number for number.
//!
//! `tools/dump_curves.mjs` runs the web version's own code (forms.js and the scene files) and writes
//! down what it gives: every curve preset built several ways, what each preset declares, and the
//! browser's Math functions over a table of inputs. These tests build the same things here and compare.
//!
//! The browser is Chrome, and the dump is written from node, whose sin, cos, exp, log and atan2 are
//! not Chrome's (a step of the last place apart in some results in a hundred, which a chaotic orbit
//! turns into another shape). So the dump puts exact, correctly rounded versions of those five in
//! node's place, which is what Chrome has: `maths_functions_print_chromes_fingerprints` holds both
//! the dump and the functions here to the line Chrome itself prints.
//!
//! The limits are: points of a raw curve within 1e-9 of the curve's size, a placement within 1e-9,
//! prepared points (32-bit) within 2e-6, counts and breaks exact. In fact everything is expected to be
//! identical to the last bit, and the tests say so line by line:
//!
//!     cargo test -p gf-core --test curve_parity -- --nocapture
//!
//! To refresh the data after the web version changes:  node tools/dump_curves.mjs

// a limit is tested as `!(x <= limit)` so that something that is not a number counts as beyond it
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::type_complexity)]

use gf_core::curve::{jsmath, orbit_line, placement, prep_curve, Curve, PlaceOpts, Pose, PrepOpts};
use gf_core::presets::{attractors, knots, CurveOpts, Family, Kind, Preset, RawCurve, Setting, Shape};
use gf_core::{HB, TRAVEL};
use serde_json::Value;
use std::f64::consts::PI;

// ------------------------------------------------------------------ reading the data
fn data(name: &str) -> Value {
    let path = format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e} (write it with: node tools/dump_curves.mjs)"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path}: {e}"))
}
// Exact values are stored as text and read with Rust's own parser, which rounds correctly.
// (A JSON number read by serde_json can be one step off in the last place.)
fn num(v: &Value) -> f64 {
    v.as_str().unwrap_or_else(|| panic!("expected a number as text, found {v}")).parse().unwrap()
}
fn nums(v: &Value) -> Vec<f64> {
    v.as_str().unwrap_or_else(|| panic!("expected numbers as text, found {v}")).split_ascii_whitespace().map(|s| s.parse().unwrap()).collect()
}
fn nums32(v: &Value) -> Vec<f32> {
    v.as_str().unwrap().split_ascii_whitespace().map(|s| s.parse().unwrap()).collect()
}
fn int(v: &Value) -> usize {
    v.as_u64().unwrap_or_else(|| panic!("expected a whole number, found {v}")) as usize
}
fn text(v: &Value) -> &str {
    v.as_str().unwrap_or_else(|| panic!("expected text, found {v}"))
}
/// FNV-1a over bytes: equal only if every bit is equal.
fn hash(bytes: impl Iterator<Item = u8>) -> u32 {
    bytes.fold(0x811c9dc5u32, |h, b| (h ^ b as u32).wrapping_mul(0x01000193))
}
fn hash64(a: &[f64]) -> u32 {
    hash(a.iter().flat_map(|x| x.to_le_bytes()))
}
fn hash32(a: &[f32]) -> u32 {
    hash(a.iter().flat_map(|x| x.to_le_bytes()))
}
/// Every `every`th point of a list of points `width` numbers wide, then its last `last` points.
fn sample<T: Copy>(a: &[T], width: usize, every: usize, last: usize) -> (Vec<T>, Vec<T>) {
    let n = a.len() / width;
    let pts = (0..n).step_by(every).flat_map(|i| a[i * width..(i + 1) * width].iter().copied()).collect();
    (pts, a[n.saturating_sub(last) * width..].to_vec())
}

// ------------------------------------------------------------------ building a curve as the web version does
struct Built {
    raw: RawCurve,
    /// a flow as bake() builds it: the whole stretch of orbit the window is cut from, and where the window starts
    whole: Option<Vec<f64>>,
    off: usize,
    pose: Pose,
    curve: Curve,
    tube: f64,
    melt: f64,
    ds: f64,
    align: bool,
}
/// What bake() adds to a shape: how far a flowing tube has travelled, beads and where they are in their cycle.
struct Baked {
    travel: f64,
    beads: f64,
    phase: f64,
}

/// The curve part of bake() in forms.js at thickness 1. `baked`: None = the shape as its formula gives it,
/// placed by itself; otherwise a flow is a window on a longer stretch of its orbit and placed by the whole
/// stretch, and beads and the taper of a flowing tube are folded into the radius.
fn build(fam: &Family, p: &Preset, vals: &[f64], len: f64, quick: bool, baked: Option<&Baked>, res: f64) -> Built {
    let tube = [p.tube.unwrap_or(0.0), fam.tube, 0.03].into_iter().find(|&t| t != 0.0).unwrap();
    let melt = fam.melt.min(0.45 * tube);
    let align = p.align.unwrap_or(fam.align);
    let place = PlaceOpts { centre: p.centre, fit: p.fit, axes: p.axes, align, e: None };
    let ds = 0.75 * 2.0 * HB / res;
    let (mut whole, mut off, mut pose) = (None, 0, None);
    let raw = match (&p.shape, baked) {
        (Shape::Flow(flow), Some(b)) => {
            let n = jsmath::round(flow.steps as f64 * len).max(40.0);
            let line = orbit_line(flow, vals, jsmath::round(n * (1.0 + TRAVEL)) as usize, quick);
            let have = (line.len() / 3) as f64;
            off = (have - n).min(jsmath::round(b.travel * n)).max(0.0) as usize;
            let end = ((off + n as usize) * 3).min(line.len());
            pose = Some(placement(&[&line], &place));
            let window = line[off * 3..end].to_vec();
            whole = Some(line);
            RawCurve { lines: vec![window], closed: false, ..Default::default() }
        }
        (Shape::Flow(flow), None) => RawCurve { lines: vec![orbit_line(flow, vals, jsmath::round(flow.steps as f64 * len).max(40.0) as usize, quick)], closed: false, ..Default::default() },
        (Shape::Curve(f), _) => f(vals, &CurveOpts { len, quick }),
        _ => panic!("{} is not a curve", p.id),
    };
    let lines: Vec<&[f64]> = raw.lines.iter().map(|l| l.as_slice()).collect();
    let (beads, phase) = baked.map_or((0.0, 0.0), |b| (b.beads, b.phase));
    let (taper, closed, own) = (baked.is_some() && p.is_flow(), raw.closed, p.radius.as_ref());
    let shaped = |u: f64, s: f64, total: f64| -> f64 {
        let mut r = own.map_or(1.0, |f| f(u));
        if beads > 0.0 {
            r *= 1.0 + beads * jsmath::sin(2.0 * PI * (u * (if closed { jsmath::round(total / 0.8).max(1.0) } else { total / 0.8 }) - phase));
        }
        if taper {
            r *= 1.0f64.min(s / 0.35).min((total - s) / 0.35).sqrt().max(0.2);
        }
        r
    };
    let plain = |u: f64, _s: f64, _total: f64| -> f64 { own.map_or(1.0, |f| f(u)) };
    let radius: Option<&(dyn Fn(f64, f64, f64) -> f64 + Sync)> =
        if baked.is_some() && (beads > 0.0 || taper) { Some(&shaped) } else if own.is_some() { Some(&plain) } else { None };
    let opts = PrepOpts { tube, melt, ds, closed, radius, place: place.clone(), pose: pose.clone() };
    let curve = prep_curve(&lines, &opts);
    // the placement on its own, as an animation asks for it; prep_curve must have come to the same
    let pose = pose.unwrap_or_else(|| placement(&lines, &place));
    assert_eq!(pose, curve.pose, "{}: prep_curve placed the curve differently from placement()", p.id);
    drop(lines);
    Built { raw, whole, off, pose, curve, tube, melt, ds, align }
}

// ------------------------------------------------------------------ comparing
/// The worst differences found in one case, and whether everything was the same to the last bit.
#[derive(Default)]
struct Diff {
    raw: f64,  // points of the raw curve, as a fraction of the curve's size
    pose: f64, // the placement
    pts: f64,  // prepared points
    raw_same: bool,
    pts_same: bool,
    fails: Vec<String>,
}
fn worst(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| if x == y { 0.0 } else { (x - y).abs() }).fold(0.0, |m, d| if d > m || d.is_nan() { d } else { m })
}
fn rel(a: f64, b: f64) -> f64 {
    if a == b { 0.0 } else { (a - b).abs() / a.abs().max(b.abs()) }
}

fn compare(case: &Value, fam: &Family, every: (usize, usize, usize, usize)) -> Diff {
    let (raw_every, raw_last, prep_every, prep_last) = every;
    let p = fam.preset(text(&case["preset"])).unwrap_or_else(|| panic!("no preset {}", case["preset"]));
    let vals = nums(&case["vals"]);
    let baked = (!case["baked"].is_null()).then(|| {
        let b = &case["baked"];
        Baked { travel: num(&b["travel"]), beads: num(&b["beads"]), phase: num(&b["phase"]) }
    });
    let b = build(fam, p, &vals, num(&case["len"]), case["quick"].as_bool().unwrap(), baked.as_ref(), 768.0);
    let mut d = Diff { raw_same: true, pts_same: true, ..Diff::default() };
    let mut fail = |what: String| d.fails.push(what);

    // the options the web version came to
    let o = &case["opts"];
    for (name, mine, theirs) in [("tube", b.tube, num(&o["tube"])), ("melt", b.melt, num(&o["melt"])), ("ds", b.ds, num(&o["ds"]))] {
        if mine != theirs { fail(format!("option {name}: {mine} here, {theirs} there")); }
    }
    if b.align != o["align"].as_bool().unwrap() { fail("option align differs".into()); }

    // the raw curve
    let theirs = &case["raw"];
    let lines = theirs["lines"].as_array().unwrap();
    if b.raw.closed != theirs["closed"].as_bool().unwrap() { fail("closed differs".into()); }
    if b.raw.lines.len() != lines.len() { fail(format!("{} lines here, {} there", b.raw.lines.len(), lines.len())); }
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for l in &b.raw.lines {
        for q in l.as_chunks::<3>().0 {
            for k in 0..3 { lo[k] = lo[k].min(q[k]); hi[k] = hi[k].max(q[k]); }
        }
    }
    let size = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max);
    let mut raw_dev = 0.0f64;
    let mut raw_same = true;
    for (i, (mine, theirs)) in b.raw.lines.iter().zip(lines).enumerate() {
        if mine.len() / 3 != int(&theirs["n"]) { fail(format!("line {i}: {} points here, {} there", mine.len() / 3, theirs["n"])); continue; }
        let (pts, tail) = sample(mine, 3, raw_every, raw_last);
        let dev = worst(&pts, &nums(&theirs["pts"])).max(worst(&tail, &nums(&theirs["tail"]))) / size;
        raw_dev = raw_dev.max(dev);
        if !(dev <= 1e-9) { fail(format!("line {i}: raw points differ by {dev:.3e} of the curve's size")); }
        let (sum, there) = (mine.iter().sum::<f64>(), num(&theirs["sum"]));
        if !((sum - there).abs() <= 1e-9 * size * mine.len() as f64) { fail(format!("line {i}: sum of coordinates {sum} here, {there} there")); }
        raw_same &= hash64(mine) as u64 == theirs["hash"].as_u64().unwrap();
    }
    if let Some(w) = case["baked"].get("whole").filter(|w| !w.is_null()) {
        let mine = b.whole.as_ref().expect("a flow keeps its whole stretch");
        if mine.len() / 3 != int(&w["n"]) { fail(format!("whole stretch: {} points here, {} there", mine.len() / 3, w["n"])); }
        if b.off != int(&case["baked"]["off"]) { fail(format!("window starts at {} here, {} there", b.off, case["baked"]["off"])); }
        let (sum, there) = (mine.iter().sum::<f64>(), num(&w["sum"]));
        if !((sum - there).abs() <= 1e-9 * size * mine.len() as f64) { fail(format!("whole stretch: sum {sum} here, {there} there")); }
        raw_same &= hash64(mine) as u64 == w["hash"].as_u64().unwrap();
    }

    // the placement
    let t = &case["pose"];
    let mut pose_dev = worst(&b.pose.c, &nums(&t["c"])) / size;
    pose_dev = pose_dev.max(rel(b.pose.sc, num(&t["sc"])));
    let ax: Vec<usize> = t["ax"].as_array().unwrap().iter().map(int).collect();
    if b.pose.ax[..] != ax[..] { fail(format!("axes {:?} here, {ax:?} there", b.pose.ax)); }
    match (&b.pose.e, t["e"].is_null()) {
        (None, true) => {}
        (Some(e), false) => pose_dev = pose_dev.max(worst(e.as_flattened(), &nums(&t["e"]))),
        _ => fail("one has a face-on turn and the other has none".into()),
    }
    if !(pose_dev <= 1e-9) { fail(format!("placement differs by {pose_dev:.3e}")); }

    // the prepared curve
    let (c, t) = (&b.curve, &case["prep"]);
    if c.n != int(&t["n"]) || c.brk.len() != c.n || c.pts.len() != c.n * 4 {
        fail(format!("{} prepared points here ({} breaks, {} numbers), {} there", c.n, c.brk.len(), c.pts.len(), t["n"]));
    }
    let breaks = c.brk.iter().map(|&x| x as u64).sum::<u64>();
    if breaks != t["brk"].as_u64().unwrap() || hash(c.brk.iter().copied()) as u64 != t["brk_hash"].as_u64().unwrap() {
        fail(format!("breaks differ ({breaks} here, {} there)", t["brk"]));
    }
    for (name, mine, theirs) in [("rmax", c.rmax, num(&t["rmax"])), ("ds", c.ds, num(&t["ds"])), ("length", c.length, num(&t["length"])), ("amp", c.amp, num(&t["amp"]))] {
        if !(rel(mine, theirs) <= 1e-9) { fail(format!("{name}: {mine} here, {theirs} there")); }
        d.pts_same &= mine.to_bits() == theirs.to_bits();
    }
    let mut pts_dev = f64::NAN;
    if c.n == int(&t["n"]) {
        let (pts, tail) = sample(&c.pts, 4, prep_every, prep_last);
        let wide = |a: &[f32]| a.iter().map(|&x| x as f64).collect::<Vec<_>>();
        pts_dev = worst(&wide(&pts), &wide(&nums32(&t["pts"]))).max(worst(&wide(&tail), &wide(&nums32(&t["tail"]))));
        if !(pts_dev <= 2e-6) { fail(format!("prepared points differ by {pts_dev:.3e}")); }
        let (sum, there) = (c.pts.iter().map(|&x| x as f64).sum::<f64>(), num(&t["sum"]));
        if !((sum - there).abs() <= 2e-6 * c.pts.len() as f64) { fail(format!("sum of prepared points {sum} here, {there} there")); }
    }
    d.pts_same &= hash32(&c.pts) as u64 == t["hash"].as_u64().unwrap();

    // the line of text about the shape
    let info = p.info.as_ref().map(|f| f(&b.raw, &vals));
    if info.as_deref() != case["info"].as_str() { fail(format!("info {info:?} here, {} there", case["info"])); }

    d.raw = raw_dev;
    d.pose = pose_dev;
    d.pts = pts_dev;
    d.raw_same = raw_same;
    d
}

/// Shapes whose numbers pass through `Math.pow`. In the browser that one is its C library's, which cannot
/// be had here; the one here gives the nearest number, and the browser's is a step of the last place
/// from that about four times in ten thousand. So these must be within the limits, and are reported,
/// but need not be identical (today they are). None of them is chaotic.
const BY_POW: &[&str] = &["fourier"];

fn check_family(fam: Family) {
    let all = data("curves.json");
    let every = (int(&all["raw_every"]), int(&all["raw_last"]), int(&all["prep_every"]), int(&all["prep_last"]));
    let cases: Vec<&Value> = all["cases"].as_array().unwrap().iter().filter(|c| text(&c["family"]) == fam.id).collect();
    assert_eq!(cases.len(), fam.presets.len() * 4, "{}: four cases for each preset", fam.id);
    println!("\n{}: worst difference from the web version ({}, V8 {})", fam.name, text(&all["about"]["node"]), text(&all["about"]["v8"]));
    println!("  {:<12} {:<13} {:>10} {:>10} {:>10}   same to the last bit", "preset", "case", "raw/size", "placement", "prepared");
    let (mut fails, mut top, mut exact) = (Vec::new(), [0.0f64; 3], 0);
    for case in &cases {
        let d = compare(case, &fam, every);
        let (id, name) = (text(&case["preset"]), text(&case["case"]));
        let same = match (d.raw_same, d.pts_same) { (true, true) => "raw, prepared", (true, false) => "raw", (false, true) => "prepared", _ => "NO" };
        println!("  {id:<12} {name:<13} {:>10.3e} {:>10.3e} {:>10.3e}   {same}", d.raw, d.pose, d.pts);
        for (m, x) in top.iter_mut().zip([d.raw, d.pose, d.pts]) {
            if x > *m || x.is_nan() { *m = x; }
        }
        exact += (d.raw_same && d.pts_same) as usize;
        fails.extend(d.fails.into_iter().map(|f| format!("{id} ({name}): {f}")));
        // identical bits are what keeps a chaotic orbit the same shape; anything less is a fault to look into
        if !(d.raw_same && d.pts_same) && !BY_POW.contains(&id) { fails.push(format!("{id} ({name}): not identical to the last bit")); }
    }
    println!("  {}: {} of {} cases identical to the last bit; worst raw {:.3e} of size, placement {:.3e}, prepared {:.3e}", fam.name, exact, cases.len(), top[0], top[1], top[2]);
    assert!(fails.is_empty(), "{} differs from the web version:\n  {}", fam.name, fails.join("\n  "));
}

#[test]
fn attractors_match_the_web_version() {
    check_family(attractors::family());
}

#[test]
fn knots_match_the_web_version() {
    check_family(knots::family());
}

/// Curves no preset makes, for the corners of `placement` and `prep_curve`: lines of one, two and three
/// points, a loop whose ends do not meet, points that repeat, steps so fine that the floor on their length
/// applies, a shape kept at the origin, a given size, swapped axes, a given turn, no melt, a varying radius.
/// Nothing here is chaotic or leans on the C library, so every bit must be the same.
#[test]
fn made_up_curves_match_the_web_version() {
    let all = data("curves.json");
    let (every, last) = (int(&all["prep_every"]), int(&all["prep_last"]));
    let same = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
    let all_same = |a: &[f64], b: &[f64]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same(*x, *y));
    let turn = |v: &Value| -> Option<[[f64; 3]; 3]> {
        (!v.is_null()).then(|| {
            let e = nums(v);
            [[e[0], e[1], e[2]], [e[3], e[4], e[5]], [e[6], e[7], e[8]]]
        })
    };
    let ramp = |u: f64, s: f64, total: f64| 0.5 + u + 0.25 * s / (total + 1.0);
    let mut fails = Vec::new();
    println!();
    for case in all["made"].as_array().unwrap() {
        let name = text(&case["name"]);
        let lines: Vec<Vec<f64>> = case["lines"].as_array().unwrap().iter().map(nums).collect();
        let lines: Vec<&[f64]> = lines.iter().map(|l| l.as_slice()).collect();
        let o = &case["o"];
        let place = PlaceOpts {
            centre: o["centre"].as_bool().unwrap(),
            fit: (!o["fit"].is_null()).then(|| num(&o["fit"])),
            axes: (!o["axes"].is_null()).then(|| [int(&o["axes"][0]), int(&o["axes"][1]), int(&o["axes"][2])]),
            align: o["align"].as_bool().unwrap(),
            e: turn(&o["e"]),
        };
        let radius: Option<&(dyn Fn(f64, f64, f64) -> f64 + Sync)> = match o["radius"].as_str() {
            Some("ramp") => Some(&ramp),
            Some(other) => panic!("no radius called {other}"),
            None => None,
        };
        let opts = PrepOpts { tube: num(&o["tube"]), melt: num(&o["melt"]), ds: num(&o["ds"]), closed: o["closed"].as_bool().unwrap(), radius, place: place.clone(), pose: None };
        let (pose, c) = (placement(&lines, &place), prep_curve(&lines, &opts));
        let mut fail = |what: &str| fails.push(format!("{name}: {what}"));

        let t = &case["pose"];
        let ax: Vec<usize> = t["ax"].as_array().unwrap().iter().map(int).collect();
        let turned = match (&pose.e, turn(&t["e"])) {
            (None, None) => true,
            (Some(a), Some(b)) => all_same(a.as_flattened(), b.as_flattened()),
            _ => false,
        };
        if !(all_same(&pose.c, &nums(&t["c"])) && same(pose.sc, num(&t["sc"])) && pose.ax[..] == ax[..] && turned) { fail("placement differs"); }
        // prep_curve works its placement out the same way
        let again = &c.pose;
        if !(all_same(&again.c, &pose.c) && same(again.sc, pose.sc) && again.ax == pose.ax && again.e.map(|e| e.map(|r| r.map(f64::to_bits))) == pose.e.map(|e| e.map(|r| r.map(f64::to_bits)))) {
            fail("prep_curve placed the curve differently from placement()");
        }

        let t = &case["prep"];
        if c.n != int(&t["n"]) || c.brk.len() != c.n || c.pts.len() != c.n * 4 { fail(&format!("{} points here, {} there", c.n, t["n"])); }
        for (what, mine, theirs) in [("rmax", c.rmax, num(&t["rmax"])), ("ds", c.ds, num(&t["ds"])), ("length", c.length, num(&t["length"])), ("amp", c.amp, num(&t["amp"]))] {
            if !same(mine, theirs) { fail(&format!("{what}: {mine:e} here, {theirs:e} there")); }
        }
        if hash(c.brk.iter().copied()) as u64 != t["brk_hash"].as_u64().unwrap() { fail("breaks differ"); }
        let (pts, tail) = sample(&c.pts, 4, every, last);
        let bits = |a: &[f32]| a.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        if bits(&pts) != bits(&nums32(&t["pts"])) || bits(&tail) != bits(&nums32(&t["tail"])) { fail("sampled points differ"); }
        if hash32(&c.pts) as u64 != t["hash"].as_u64().unwrap() { fail("points differ"); }
        println!("  {name:<48} {:>6} points", c.n);
    }
    assert!(fails.is_empty(), "made-up curves differ from the web version:\n  {}", fails.join("\n  "));
}

// ------------------------------------------------------------------ what each family and preset declares
#[test]
fn presets_declare_what_the_web_version_declares() {
    let all = data("curves_presets.json");
    let floats = |v: &Value| -> Vec<f64> { v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect() };
    let strs = |v: &Value| -> Vec<String> { v.as_array().unwrap().iter().map(|x| text(x).to_string()).collect() };
    let mut fails = Vec::new();
    for (fam, t) in [attractors::family(), knots::family()].iter().zip(all["families"].as_array().unwrap()) {
        macro_rules! same {
            ($who:expr, $what:expr, $mine:expr, $theirs:expr) => {{
                let (mine, theirs) = ($mine, $theirs);
                if mine != theirs { fails.push(format!("{} {}: {:?} here, {:?} there", $who, $what, mine, theirs)); }
            }};
        }
        let who = fam.id;
        same!(who, "id", fam.id, text(&t["id"]));
        same!(who, "name", fam.name, text(&t["name"]));
        same!(who, "kind", fam.kind == Kind::Curve, text(&t["kind"]) == "curve");
        same!(who, "melt", fam.melt, t["melt"].as_f64().unwrap());
        same!(who, "tube", fam.tube, t["tube"].as_f64().unwrap());
        same!(who, "align", fam.align, t["align"].as_bool().unwrap());
        same!(who, "ambient", fam.ambient.to_vec(), floats(&t["ambient"]));
        same!(who, "stepScale", fam.step_scale, t["stepScale"].as_f64().unwrap());
        same!(who, "stepsOut", fam.steps_out as u64, t["stepsOut"].as_u64().unwrap());
        same!(who, "stepsIn", fam.steps_in as u64, t["stepsIn"].as_u64().unwrap());
        same!(who, "layers", fam.layers as u64, t["layers"].as_u64().unwrap());
        let defaults: Vec<(String, Vec<f64>)> = fam.defaults.iter().map(|(k, v)| (k.to_string(), match v { Setting::Num(x) => vec![*x], Setting::Vec3(a) => a.to_vec() })).collect();
        let theirs: Vec<(String, Vec<f64>)> = t["defaults"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), if v.is_array() { floats(v) } else { vec![v.as_f64().unwrap()] })).collect();
        // (serde_json hands an object's keys back in alphabetical order)
        let mut sorted = defaults.clone();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        same!(who, "defaults", sorted, theirs);
        let lights: Vec<Vec<f64>> = fam.lights.iter().map(|l| vec![l.az, l.el, l.w, l.h, l.rgb[0], l.rgb[1], l.rgb[2], l.soft]).collect();
        let theirs: Vec<Vec<f64>> = t["lights"].as_array().unwrap().iter().map(|l| [vec![l["az"].as_f64().unwrap(), l["el"].as_f64().unwrap(), l["w"].as_f64().unwrap(), l["h"].as_f64().unwrap()], floats(&l["rgb"]), vec![l["soft"].as_f64().unwrap()]].concat()).collect();
        same!(who, "lights", lights, theirs);

        let presets = t["presets"].as_array().unwrap();
        same!(who, "presets", fam.presets.iter().map(|p| p.id.to_string()).collect::<Vec<_>>(), presets.iter().map(|p| text(&p["id"]).to_string()).collect::<Vec<_>>());
        for (p, t) in fam.presets.iter().zip(presets) {
            let who = format!("{}/{}", fam.id, p.id);
            same!(who, "name", p.name, text(&t["name"]));
            same!(who, "eq", p.eq.iter().map(|s| s.to_string()).collect::<Vec<_>>(), strs(&t["eq"]));
            same!(who, "note", p.note, text(&t["note"]));
            let params: Vec<(String, String, Vec<f64>)> = p.params.iter().map(|a| (a.id.to_string(), a.label.to_string(), vec![a.value, a.lo, a.hi, a.step])).collect();
            let theirs: Vec<(String, String, Vec<f64>)> = t["params"].as_array().unwrap().iter()
                .map(|a| (text(&a["id"]).to_string(), text(&a["label"]).to_string(), ["value", "lo", "hi", "step"].iter().map(|k| a[*k].as_f64().unwrap()).collect()))
                .collect();
            same!(who, "params", params, theirs);
            same!(who, "rot", p.rot.map(|r| r.to_vec()), (!t["rot"].is_null()).then(|| floats(&t["rot"])));
            same!(who, "animate", p.animate.as_ref().map(|a| a.iter().map(|s| s.to_string()).collect::<Vec<_>>()), (!t["animate"].is_null()).then(|| strs(&t["animate"])));
            same!(who, "tube", p.tube, t["tube"].as_f64());
            same!(who, "axes", p.axes.map(|a| a.to_vec()), (!t["axes"].is_null()).then(|| t["axes"].as_array().unwrap().iter().map(int).collect::<Vec<_>>()));
            same!(who, "align", p.align, t["align"].as_bool());
            same!(who, "fit", p.fit, t["fit"].as_f64());
            same!(who, "centre", p.centre, t["centre"].as_bool().unwrap());
            same!(who, "fixedLength", p.fixed_length, t["fixedLength"].as_bool().unwrap());
            same!(who, "radius", p.radius.is_some(), t["radius"].as_bool().unwrap());
            same!(who, "info", p.info.is_some(), t["info"].as_bool().unwrap());
            let flow = match &p.shape { Shape::Flow(f) => Some((f.start.to_vec(), f.dt, f.steps, f.settle)), _ => None };
            let theirs = (!t["flow"].is_null()).then(|| (floats(&t["flow"]["start"]), t["flow"]["dt"].as_f64().unwrap(), int(&t["flow"]["steps"]), int(&t["flow"]["settle"])));
            same!(who, "flow", flow, theirs);
        }
    }
    assert!(fails.is_empty(), "presets differ from the web version:\n  {}", fails.join("\n  "));
}

// ------------------------------------------------------------------ the Math functions themselves
/// How many representable numbers apart two values are (0 = the same bits, or both not numbers).
fn steps_apart(a: f64, b: f64) -> u64 {
    if a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) { return 0; }
    if a.is_nan() || b.is_nan() || a.is_sign_negative() != b.is_sign_negative() { return u64::MAX; }
    a.to_bits().abs_diff(b.to_bits())
}

/// One function against its column of the table. `slack`: how many steps of the last place a result may
/// be from the table's (0 = the same bits). Prints a line; returns what is wrong, if anything.
fn against_table(name: &str, mine: Vec<f64>, theirs: Vec<f64>, slack: u64, show: &dyn Fn(usize) -> String) -> Option<String> {
    assert_eq!(mine.len(), theirs.len(), "{name}");
    let apart: Vec<u64> = mine.iter().zip(&theirs).map(|(a, b)| steps_apart(*a, *b)).collect();
    let (bad, most) = (apart.iter().filter(|&&d| d > 0).count(), apart.iter().copied().max().unwrap_or(0));
    println!("  {name:<28} {:>5} inputs  {bad:>5} differ  {}", mine.len(), if bad > 0 { format!("by up to {most}") } else { String::new() });
    let i = apart.iter().position(|&d| d > slack)?;
    Some(format!("{name}: {bad} of {} differ (by up to {most} steps), first at {}: {:e} here, {:e} there", mine.len(), show(i), mine[i], theirs[i]))
}

/// Every function on the table's inputs: the correctly rounded ones (Chrome's) against exact arithmetic,
/// fdlibm's against the engine that wrote the table (node, which still has fdlibm's).
#[test]
fn maths_functions_give_the_tables_bits() {
    let all = data("curves_math.json");
    println!("\nMath functions against the table (written from {}, V8 {}): inputs, how many differ, by how many steps of the last place at most", text(&all["about"]["node"]), text(&all["about"]["v8"]));
    let mut fails = Vec::new();
    // column, the function here, and how large an input it promises the right bits for (the correctly
    // rounded sine up to 2^52, fdlibm's up to 2^19 quarter turns; further out a step is allowed)
    let one: [(&str, fn(f64) -> f64, f64); 11] = [
        ("sin", jsmath::sin, 4503599627370496.0),
        ("cos", jsmath::cos, 4503599627370496.0),
        ("exp", jsmath::exp, f64::INFINITY),
        ("log", jsmath::log, f64::INFINITY),
        ("engine_sin", jsmath::fdlibm_sin, 8.0e5),
        ("engine_cos", jsmath::fdlibm_cos, 8.0e5),
        ("engine_exp", jsmath::fdlibm_exp, f64::INFINITY),
        ("engine_log", jsmath::fdlibm_log, f64::INFINITY),
        ("exp (the slow way)", jsmath::exp_slowly, f64::INFINITY),
        ("sqrt", jsmath::sqrt, f64::INFINITY),
        ("round", jsmath::round, f64::INFINITY),
    ];
    for (name, f, reach) in one {
        let column = name.split(' ').next().unwrap();
        let (x, out) = (nums(&all[column]["x"]), nums(&all[column]["out"]));
        let part = |far: bool| -> (Vec<f64>, Vec<f64>) { x.iter().zip(&out).filter(|(x, _)| (x.abs() > reach) == far).map(|(x, o)| (*x, *o)).unzip() };
        let (x, out) = part(false);
        fails.extend(against_table(name, x.iter().map(|&x| f(x)).collect(), out, 0, &|i| format!("{:e}", x[i])));
        let (x, out) = part(true);
        if !x.is_empty() {
            fails.extend(against_table(&format!("{name} beyond {reach:.0e}"), x.iter().map(|&x| f(x)).collect(), out, 1, &|i| format!("{:e}", x[i])));
        }
    }
    // `pow` in the table is the engine's, which is Chrome's too (their C library's): not always the
    // nearest number, so the function here may be a step from it. `pow_rounded` is exact arithmetic.
    let many: [(&str, u64, &dyn Fn(&[f64]) -> f64); 6] = [
        ("atan2", 0, &|a| jsmath::atan2(a[0], a[1])),
        ("engine_atan2", 0, &|a| jsmath::fdlibm_atan2(a[0], a[1])),
        ("pow_rounded", 0, &|a| jsmath::pow(a[0], a[1])),
        ("pow", 1, &|a| jsmath::pow(a[0], a[1])),
        ("hypot2", 0, &|a| jsmath::hypot2(a[0], a[1])),
        ("hypot3", 0, &|a| jsmath::hypot3(a[0], a[1], a[2])),
    ];
    for (name, slack, f) in many {
        let x: Vec<Vec<f64>> = all[name]["x"].as_array().unwrap().iter().map(nums).collect();
        fails.extend(against_table(name, x.iter().map(|a| f(a)).collect(), nums(&all[name]["out"]), slack, &|i| format!("{:?}", x[i])));
    }
    assert!(fails.is_empty(), "these do not give the table's bits:\n  {}", fails.join("\n  "));
}

/// The fingerprint of a function over the probe's 100,000 inputs (PROBE in tools/dump_curves.mjs, and
/// tools/probe_math.js, which prints the same in a browser).
fn fingerprints(sin: fn(f64) -> f64, cos: fn(f64) -> f64, exp: fn(f64) -> f64, log: fn(f64) -> f64, atan2: fn(f64, f64) -> f64, pow: fn(f64, f64) -> f64) -> [(&'static str, String); 8] {
    let print = |g: &dyn Fn(f64) -> f64| format!("{:08x}", hash((0..100000).flat_map(|i| g(i as f64).to_le_bytes())));
    let (a, p) = (|i: f64| -8.0 + i * 16e-5, |i: f64| 1e-3 + i * 1e-4);
    [
        ("sin", print(&|i| sin(a(i)))),
        ("cos", print(&|i| cos(a(i)))),
        ("exp", print(&|i| exp(a(i)))),
        ("log", print(&|i| log(p(i)))),
        ("atan2", print(&|i| atan2(a(i), 1.0 + i * 1e-5))),
        ("pow", print(&|i| pow(1.0 + i % 8.0, -0.6 - i * 1.9e-5))),
        ("hypot", print(&|i| jsmath::hypot3(a(i), p(i), 0.3))),
        ("round", print(&|i| jsmath::round(a(i) * 3.0))),
    ]
}

/// What headless Chrome 157 (the browser the web version is tested with) prints for the probe, through
/// native/tools/probe_math.js. The functions here must print the same: that is what keeps a chaotic
/// shape the shape the browser draws. Fixed here, not read from a file, so that nothing can drift.
const CHROME: [(&str, &str); 8] =
    [("sin", "b3d7dd5f"), ("cos", "9c2450dc"), ("exp", "4b6ac5fc"), ("log", "4fba43fe"), ("atan2", "c1561e12"), ("pow", "7b37f361"), ("hypot", "a882309b"), ("round", "feffc387")];
/// `pow` is the exception. Chrome's is its C library's, which cannot be had here and is not always the
/// nearest number; the one here always is, and this is the fingerprint of that (from exact arithmetic).
const POW_ROUNDED: &str = "1aa3af33";
/// And what node 24 prints: fdlibm's sin, cos, exp, log and atan2, kept here under their own names.
const NODE: [(&str, &str); 5] = [("sin", "6eb5fe3e"), ("cos", "e5c81f3c"), ("exp", "307b07c8"), ("log", "d8b38c26"), ("atan2", "dea9da3a")];

#[test]
fn maths_functions_print_chromes_fingerprints() {
    let mine = fingerprints(jsmath::sin, jsmath::cos, jsmath::exp, jsmath::log, jsmath::atan2, jsmath::pow);
    println!("\nfingerprints over 100,000 inputs each: here / Chrome 157");
    let mut fails = Vec::new();
    for ((name, mine), (_, chrome)) in mine.iter().zip(CHROME) {
        let expected = if *name == "pow" { POW_ROUNDED } else { chrome };
        println!("  {name:<6} {mine} {chrome} {}", if mine == chrome { "" } else if mine == expected { "(the nearest number; Chrome's C library is not always that)" } else { "DIFFERENT" });
        if mine != expected { fails.push(format!("{name}: {mine} here, {expected} expected")); }
    }
    // the same lines as the data file has them, so that the file and this test cannot part company
    let all = data("curves_math.json");
    for (name, chrome) in CHROME {
        assert_eq!(text(&all["probe"]["chrome_printed"][name]), chrome, "{name}: the data file has another line for Chrome");
        assert_eq!(text(&all["probe"]["chrome"][name]), chrome, "{name}: exact arithmetic in the dump does not print Chrome's");
    }
    assert_eq!(text(&all["probe"]["pow_rounded"]), POW_ROUNDED);
    // fdlibm's, as node has them
    let fdlibm = fingerprints(jsmath::fdlibm_sin, jsmath::fdlibm_cos, jsmath::fdlibm_exp, jsmath::fdlibm_log, jsmath::fdlibm_atan2, jsmath::pow);
    for ((name, mine), (_, node)) in fdlibm.iter().zip(NODE) {
        println!("  {:<14} {mine} {node}  (node)", format!("fdlibm_{name}"));
        if mine != node { fails.push(format!("fdlibm_{name}: {mine} here, {node} from node")); }
        assert_eq!(text(&all["probe"]["engine"][name]), node, "{name}: the data file was not written from an engine with fdlibm's");
    }
    assert!(fails.is_empty(), "the fingerprints have moved:\n  {}", fails.join("\n  "));
}
