//! Crystals: solids with flat faces, built as unions of pieces (`Piece`), each the intersection of a few
//! half-spaces and balls, baked into the volume like any curve.
//!
//!   bismuth  a hopper crystal. The edges of a bismuth crystal grow faster than its faces, so instead of
//!            a cube it leaves a staircase of square frames, each a step lower and smaller than the last,
//!            down into a hollow; each frame is turned a little from the one before, so the steps spiral.
//!            Every frame is four bars with a raised rim along its outer edge.
//!   quartz   a cluster of hexagonal prisms on a rough base. Each prism has six sides and ends in six
//!            rhombohedral faces (in quartz they meet the axis at about 52 degrees from its cross-section).
//!   geode    a hollow stone, broken open, its inside lined with small quartz points that grow inward.
//!
//! Faces are kept crisp: the pieces are joined by a hard union (the family's melt is 0). Pieces that
//! touch always overlap a little, so no seam between them is ever a surface.
//!
//! `grow`: each shape is built at full size once for its frame (so it grows in place), then as far as
//! it has grown.

use super::*;
use geo::*;

pub const FAMILY: &str = "crystals";
const DEG: f64 = std::f64::consts::PI / 180.0;

/// One quartz point: a hexagonal prism from `base` along the unit `axis`, `len` from base to apex, flat
/// sides `r` from the axis, one of them facing the unit `u` (across the axis); its six tip faces' normals
/// at `tip` (degrees) from the axis.
fn quartz_point(base: V, axis: V, u: V, len: f64, r: f64, tip: f64) -> Option<Piece> {
    if len <= 1e-6 || r <= 1e-6 { return None; }
    let v = cross(axis, u);
    let apex = add(base, mul(axis, len));
    let (st, ct) = ((tip * DEG).sin(), (tip * DEG).cos());
    let mut terms = Vec::with_capacity(13);
    for k in 0..6 {
        let a = k as f64 * 60.0 * DEG;
        let n = add(mul(u, a.cos()), mul(v, a.sin()));
        terms.push(Term::Plane { n, d: dot(n, base) + r });
        let m = add(mul(axis, ct), mul(n, st));
        terms.push(Term::Plane { n: m, d: dot(m, apex) });
    }
    terms.push(Term::Plane { n: mul(axis, -1.0), d: -dot(axis, base) });
    let big = r / 30f64.to_radians().cos();
    let (lo, hi) = (std::array::from_fn(|i| base[i].min(apex[i]) - big), std::array::from_fn(|i| base[i].max(apex[i]) + big));
    Some(Piece { lo, hi, terms })
}

/// A unit vector across `axis`, turned by `a` about it.
fn across_turned(axis: V, a: f64) -> V { turn(across(axis), axis, a) }

/// How far one of many things has grown when the whole has grown `g`: each starts at `start` (0..1 of
/// the way) and is whole at the end.
fn part(g: f64, start: f64) -> f64 {
    if g >= 1.0 { 1.0 } else { ((g - start) / (1.0 - start).max(1e-9)).clamp(0.0, 1.0) }
}

/// Build a shape at full size for its frame, then as far as it has grown.
fn grown(v: &[f64], grow_at: usize, mut build: impl FnMut(&[f64], f64) -> RawCurve) -> RawCurve {
    let mut whole = build(v, 1.0);
    // (a shape may give its own frame: the boxes round slanting points are loose)
    if whole.frame.is_empty() { whole.frame_pieces(); }
    let g = v[grow_at].clamp(0.0, 1.0);
    if g >= 1.0 { return whole; }
    let mut raw = build(v, g);
    raw.frame = whole.frame;
    raw
}

// ---- bismuth ----------------------------------------------------------------------------------------

mod bi {
    pub const LEVELS: usize = 0;
    pub const STEP: usize = 1;
    pub const WALL: usize = 2;
    pub const TWIST: usize = 3;
    pub const RIM: usize = 4;
    pub const COUNT: usize = 5;
    pub const SEED: usize = 6;
    pub const GROW: usize = 7;
}

/// One hopper: frames from the top (k = 0, the largest) down into the hollow, centred at `c`, its own up
/// `up` and across `x0`, `size` the top frame's half-width. Grows from the innermost frame outward.
#[allow(clippy::too_many_arguments)]
fn hopper(out: &mut Vec<Piece>, c: V, up: V, x0: V, size: f64, levels: usize, step: f64, wall: f64, twist: f64, rim: f64, g: f64) {
    let n = levels.max(1);
    // each frame is `shrink` of the top's half-width smaller than the one above, and `drop` lower
    let shrink = step * size / n as f64 * 1.6;
    let drop = shrink * 1.5;
    let w = (wall * size).max(shrink * 1.15);
    let thick = drop * 1.25;
    for k in 0..n {
        // the innermost frame is first to grow; a frame grows up from below as it comes
        let start = (n - 1 - k) as f64 / n as f64 * 0.9;
        let f = part(g, start);
        if f <= 0.0 { continue; }
        let s = size - k as f64 * shrink;
        if s <= 0.5 * w { break; }
        let x = turn(x0, up, k as f64 * twist * DEG);
        let z = cross(x, up);
        let top = add(c, mul(up, -(k as f64) * drop));
        let h = thick * f;
        let mid = add(top, mul(up, -0.5 * h));
        // the innermost frame is a floor: a full square
        let last = k + 1 == n || size - (k + 1) as f64 * shrink <= 0.5 * w;
        if last {
            out.push(box_piece(mid, [x, up, z], [s, 0.5 * h, s]));
        } else {
            for (a, b) in [(x, z), (mul(x, -1.0), mul(z, -1.0)), (z, mul(x, -1.0)), (mul(z, -1.0), x)] {
                // a bar along b, at s - w/2 out along a, running the frame's full width (bars overlap at the corners)
                out.push(box_piece(add(mid, mul(a, s - 0.5 * w)), [a, up, b], [0.5 * w, 0.5 * h, s]));
            }
        }
        // the raised rim along the frame's outer edge
        if rim > 0.0 && f > 0.0 {
            let rh = rim * thick * f;
            let rw = (0.3 * w).max(0.4 * shrink);
            for (a, b) in [(x, z), (mul(x, -1.0), mul(z, -1.0)), (z, mul(x, -1.0)), (mul(z, -1.0), x)] {
                let at = add(add(top, mul(a, s - 0.5 * rw)), mul(up, 0.5 * rh - 0.05 * h));
                out.push(box_piece(at, [a, up, b], [0.5 * rw, 0.5 * rh + 0.05 * h, s]));
            }
        }
    }
}

fn bismuth(v: &[f64], _o: &CurveOpts) -> RawCurve {
    grown(v, bi::GROW, |v, g| {
        let mut raw = RawCurve::default();
        let levels = v[bi::LEVELS].round().max(2.0) as usize;
        let count = v[bi::COUNT].round().max(1.0) as usize;
        let mut r = Rng::new(v[bi::SEED] * 7.0 + 3.0);
        let up = [0.0, 1.0, 0.0];
        hopper(&mut raw.pieces, [0.0, 0.0, 0.0], up, [1.0, 0.0, 0.0], 1.0, levels, v[bi::STEP], v[bi::WALL], v[bi::TWIST], v[bi::RIM], g);
        // smaller hoppers grown out of the first's corners, leaning out
        let corners: Vec<f64> = (0..4).map(|k| (45.0 + 90.0 * k as f64) * DEG).collect();
        let first = (r.next() * 4.0) as usize;
        for i in 1..count {
            let a = corners[(first + i - 1) % 4] + 0.25 * r.sym() + if i > 4 { 0.4 } else { 0.0 };
            let size = r.range(0.38, 0.55);
            let out = [a.cos(), 0.0, a.sin()];
            let at = add(mul(out, 1.05 + 0.25 * size), [0.0, -r.range(0.2, 0.45), 0.0]);
            let tilt = unit(add(up, mul(out, r.range(0.35, 0.8))));
            let x = across_turned(tilt, TAU * r.next());
            let lv = ((levels as f64 * size * 1.2).round() as usize).max(2);
            let gi = part(g, 0.25 + 0.3 * r.next());
            let tw = v[bi::TWIST] * if r.next() < 0.5 { -1.0 } else { 1.0 };
            hopper(&mut raw.pieces, at, tilt, x, size, lv, v[bi::STEP], v[bi::WALL], tw, v[bi::RIM], gi);
        }
        raw
    })
}

// ---- quartz -----------------------------------------------------------------------------------------

mod qz {
    pub const SEED: usize = 0;
    pub const COUNT: usize = 1;
    pub const SPREAD: usize = 2;
    pub const LENGTH: usize = 3;
    pub const SLENDER: usize = 4;
    pub const TIPS: usize = 5;
    pub const BASE: usize = 6;
    pub const GROW: usize = 7;
}

/// A rough stone: a ball with a few faces knocked off it.
fn stone(out: &mut Vec<Piece>, c: V, r: f64, faces: usize, rng: &mut Rng) {
    let mut terms = vec![Term::Ball { c, r }];
    for _ in 0..faces {
        let n = rng.dir();
        terms.push(Term::Plane { n, d: dot(n, c) + r * rng.range(0.78, 0.95) });
    }
    out.push(Piece { lo: sub(c, [r; 3]), hi: add(c, [r; 3]), terms });
}

fn quartz(v: &[f64], _o: &CurveOpts) -> RawCurve {
    let mut raw = grown(v, qz::GROW, |v, g| {
        let mut raw = RawCurve::default();
        let mut r = Rng::new(v[qz::SEED] * 13.0 + 1.0);
        let base = v[qz::BASE];
        // the base: a few rough stones, sunk into the ground
        let mut tops: Vec<(V, f64)> = Vec::new();
        if base > 0.0 {
            for i in 0..4 {
                let a = TAU * (i as f64 / 4.0 + 0.2 * r.sym());
                let rr = base * if i == 0 { 1.0 } else { r.range(0.5, 0.75) };
                let off = if i == 0 { 0.0 } else { base * r.range(0.55, 0.85) };
                let c = [off * a.cos(), -0.45 * rr, off * a.sin()];
                stone(&mut raw.pieces, c, rr, 9, &mut r);
                tops.push((c, rr));
                // (the frame: what shows of it above the ground)
                for d in [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]] {
                    let q = add(c, mul(d, rr * 0.9));
                    raw.frame.extend_from_slice(&[q[0], q[1].max(0.0), q[2]]);
                }
            }
        } else {
            tops.push(([0.0, -0.2, 0.0], 0.2));
        }
        let count = v[qz::COUNT].round().max(1.0) as usize;
        let spread = v[qz::SPREAD] * DEG;
        let up = [0.0, 1.0, 0.0];
        for i in 0..count {
            // the biggest stand near the middle and upright; the rest lean out round them
            let rank = i as f64 / count.max(2) as f64;
            let lean = spread * (0.15 + 0.85 * rank.sqrt()) * r.range(0.7, 1.0);
            let az = TAU * (i as f64 * 0.618034 + 0.15 * r.sym());
            let out = [az.cos(), 0.0, az.sin()];
            let axis = unit(add(mul(up, lean.cos()), mul(out, lean.sin())));
            let len = v[qz::LENGTH] * (1.0 - 0.55 * rank) * r.range(0.75, 1.1);
            let rad = len / v[qz::SLENDER] * r.range(0.8, 1.2);
            // rooted in a stone of the base, below its surface, leaning with it
            let (sc, sr) = tops[i % tops.len()];
            let foot = add(sc, mul(unit(add(axis, mul(out, 0.5 * rank))), sr * 0.6));
            let f = part(g, 0.6 * rank * r.range(0.6, 1.0));
            let u = across_turned(axis, TAU * r.next());
            for q in [add(foot, mul(axis, len)), add(add(foot, mul(axis, len * 0.8)), mul(u, rad)), add(add(foot, mul(axis, len * 0.8)), mul(u, -rad))] { raw.frame.extend_from_slice(&q); }
            if f <= 0.0 { continue; }
            if let Some(p) = quartz_point(foot, axis, u, len * f, rad * (0.35 + 0.65 * f), v[qz::TIPS]) { raw.pieces.push(p); }
        }
        raw
    });
    raw.ground = Some(0.0);
    raw
}

// ---- geode ------------------------------------------------------------------------------------------

mod ge {
    pub const SEED: usize = 0;
    pub const POINTS: usize = 1;
    pub const SIZE: usize = 2;
    pub const SHELL: usize = 3;
    pub const OPEN: usize = 4;
    pub const GROW: usize = 5;
}

/// The geode. `whole`: its shell as one piece (for the formula, which passes pieces by by their boxes, and
/// has no voxels of a hollow to fill), rather than as patches each with a box round it alone (for the grid).
#[cfg(test)]
fn geode_build(v: &[f64], whole: bool) -> RawCurve {
    geode_points(v, whole, |_| {})
}

struct GeodePoint {
    base: V,
    axis: V,
    side: V,
    length: f64,
    radius: f64,
    start: f64,
}

fn geode_points(v: &[f64], whole: bool, mut record: impl FnMut(GeodePoint)) -> RawCurve {
    grown(v, ge::GROW, |v, g| {
        let mut raw = RawCurve::default();
        let mut r = Rng::new(v[ge::SEED] * 31.0 + 5.0);
        let (ro, ri) = (1.0, (1.0 - v[ge::SHELL]).max(0.3));
        // the break: everything in front of this plane is gone (the open face looks at +z)
        let open = v[ge::OPEN];
        let cut = Term::Plane { n: [0.0, 0.0, 1.0], d: open };
        // the stone's rough outside: faces knocked off a ball
        let rough: Vec<(V, f64)> = (0..22).map(|_| (r.dir(), ro * r.range(0.88, 0.97))).collect();
        if whole {
            let mut terms = vec![Term::Ball { c: [0.0; 3], r: ro }, Term::Hollow { c: [0.0; 3], r: ri }];
            for &(n, d) in &rough { terms.push(Term::Plane { n, d }); }
            terms.push(cut.clone());
            raw.pieces.push(Piece { lo: [-ro; 3], hi: [ro, ro, (open + 0.01).min(ro)], terms });
        }
        // The shell, cut into patches of a cube's faces (4 x 4 on each), each with a box round it alone:
        // one piece round the whole would be drawn into every voxel of the hollow. The patches overlap.
        let m = 4;
        let lap = 0.06;
        for face in (0..6).filter(|_| !whole) {
            let ax = face / 2;
            let sgn = if face % 2 == 0 { 1.0 } else { -1.0 };
            let (a1, a2) = ((ax + 1) % 3, (ax + 2) % 3);
            for i in 0..m {
                for j in 0..m {
                    let (s0, s1) = (-1.0 + 2.0 * i as f64 / m as f64, -1.0 + 2.0 * (i + 1) as f64 / m as f64);
                    let (t0, t1) = (-1.0 + 2.0 * j as f64 / m as f64, -1.0 + 2.0 * (j + 1) as f64 / m as f64);
                    let mut terms = vec![Term::Ball { c: [0.0; 3], r: ro }, Term::Hollow { c: [0.0; 3], r: ri }];
                    // the pyramid from the middle through this patch: four planes through the origin,
                    // pushed out a little so neighbours overlap
                    let mut e = |lo: f64, hi: f64, a: usize| {
                        // the plane through the origin and the line coord_a = lo * coord_ax (cube coordinates)
                        let mut n = [0.0; 3];
                        n[a] = -1.0;
                        n[ax] = lo * sgn;
                        let n = unit(n);
                        terms.push(Term::Plane { n, d: lap });
                        let mut n = [0.0; 3];
                        n[a] = 1.0;
                        n[ax] = -hi * sgn;
                        let n = unit(n);
                        terms.push(Term::Plane { n, d: lap });
                    };
                    e(s0, s1, a1);
                    e(t0, t1, a2);
                    // the middle of the patch's direction, and the rough faces that can reach it
                    let mut mid = [0.0; 3];
                    mid[ax] = sgn;
                    mid[a1] = 0.5 * (s0 + s1);
                    mid[a2] = 0.5 * (t0 + t1);
                    let mid = unit(mid);
                    for &(n, d) in &rough {
                        if dot(n, mid) > 0.45 { terms.push(Term::Plane { n, d }); }
                    }
                    terms.push(cut.clone());
                    // the box round the patch: its corners' directions at both radii
                    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
                    for (s, t) in [(s0, t0), (s0, t1), (s1, t0), (s1, t1), (0.5 * (s0 + s1), 0.5 * (t0 + t1)), (s0, 0.5 * (t0 + t1)), (s1, 0.5 * (t0 + t1)), (0.5 * (s0 + s1), t0), (0.5 * (s0 + s1), t1)] {
                        let mut d = [0.0; 3];
                        d[ax] = sgn;
                        d[a1] = s;
                        d[a2] = t;
                        let d = unit(d);
                        for rr in [ri * 0.97, ro] {
                            for k in 0..3 {
                                lo[k] = lo[k].min(d[k] * rr - lap - 0.02);
                                hi[k] = hi[k].max(d[k] * rr + lap + 0.02);
                            }
                        }
                    }
                    lo[2] = lo[2].min(open);
                    if lo[2] >= open { continue; }
                    hi[2] = hi[2].min(open + 0.01);
                    raw.pieces.push(Piece { lo, hi, terms });
                }
            }
        }
        // the lining: points rooted in the inside wall, pointing in
        let n = v[ge::POINTS].round().max(0.0) as usize;
        let size = v[ge::SIZE];
        let mut placed = 0;
        let mut tries = 0;
        while placed < n && tries < 20 * n + 100 {
            tries += 1;
            let d = r.dir();
            // only where the wall is left, and not right at the broken edge
            if d[2] * ri > open - 0.6 * size { continue; }
            let axis = unit(add(mul(d, -1.0), mul(r.dir(), 0.35)));
            let len = size * r.range(0.45, 1.15);
            let rad = len / r.range(2.2, 3.2);
            let foot = mul(d, ri + 0.6 * rad + 0.02);
            let start = 0.7 * r.next();
            let f = part(g, start);
            let u = across_turned(axis, TAU * r.next());
            record(GeodePoint { base: foot, axis, side: u, length: len, radius: rad, start });
            placed += 1;
            if f <= 0.0 { continue; }
            if let Some(p) = quartz_point(foot, axis, u, len * f + 0.6 * rad + 0.02, rad * (0.4 + 0.6 * f), 52.0) { raw.pieces.push(p); }
        }
        raw
    })
}

/// All sizes of a growing point are contained in the final prism. Its cell lists
/// and plane directions therefore stay fixed throughout playback. The shader changes
/// the prism's radius and tip height from the same start time as the generator.
fn geode_table(v: &[f64]) -> Vec<[f32; 4]> {
    let mut full = v.to_vec();
    full[ge::GROW] = 1.0;
    let mut points = Vec::new();
    let raw = geode_points(&full, true, |p| points.push(p));
    let (origin, scale) = exact::place_of(&raw.frame, crate::FIT);
    let shell = RawCurve { pieces: vec![raw.pieces[0].clone()], frame: raw.frame.clone(), ..Default::default() };
    let mut pieces = exact::placed_pieces(&shell, origin, scale);
    for (point, piece) in points.iter().zip(&raw.pieces[1..]) {
        let base = mul(sub(point.base, origin), scale);
        let record = |p: V, w: f64| [p[0] as f32, p[1] as f32, p[2] as f32, w as f32];
        pieces.push(exact::Written {
            lo: mul(sub(piece.lo, origin), scale),
            hi: mul(sub(piece.hi, origin), scale),
            e: vec![
                [109.0, point.start as f32, (point.length * scale) as f32, (point.radius * scale) as f32],
                record(base, 0.02 * scale),
                record(point.axis, 0.0), record(point.side, 0.0),
            ],
            meta: 0.0,
        });
    }
    exact::write(pieces, None, 0.0)
}

fn geode_formula() -> (Shape, Option<TableFn>) {
    let source = exact::WGSL.replace("fn pc_growth() -> f32 { return 1.0; }", "fn pc_growth() -> f32 { return pr.grow; }");
    let shape = Shape::Field { wgsl: source, clip: crate::field::CLIP_DEFAULT.to_owned(), shell: crate::field::SHELL_DEFAULT.to_owned() };
    let cached = std::sync::Mutex::new(None::<(Vec<f64>, Vec<[f32; 4]>)>);
    let table: TableFn = Box::new(move |v, _| {
        let key = &v[..ge::GROW];
        let mut cache = cached.lock().unwrap_or_else(|poison| poison.into_inner());
        if let Some((saved, entries)) = &*cache {
            if saved == key { return entries.clone(); }
        }
        let entries = geode_table(v);
        *cache = Some((key.to_vec(), entries.clone()));
        entries
    });
    (shape, Some(table))
}

/// What the builders are told when they build for the formula (nothing moves: no quick version).
const WHOLE: CurveOpts = CurveOpts { len: 1.0, quick: false, text: "" };
/// A preset traced (and baked) from its pieces as an exact formula (`exact.rs`), built by `build`.
fn formula(build: fn(&[f64]) -> RawCurve) -> (Shape, Option<TableFn>) {
    let shape = Shape::Field { wgsl: exact::WGSL.to_string(), clip: crate::field::CLIP_DEFAULT.to_string(), shell: crate::field::SHELL_DEFAULT.to_string() };
    (shape, Some(Box::new(move |v: &[f64], _: &str| exact::table(&build(v), crate::FIT))))
}

// ---- the family -------------------------------------------------------------------------------------

pub fn family() -> Family {
    let bismuth = Preset {
        eq: vec!["frame k: half-width s − k·step, k·drop lower, turned k·twist", "edges outgrow faces: a hopper"],
        note: "A hopper crystal: its edges grew faster than its faces, leaving a spiral staircase of hollow square frames. Real bismuth is iridescent (an oxide film): the thin-film material suits it.",
        params: vec![
            ParamDef::labelled("levels", "steps", 9.0, 2.0, 16.0, 1.0),
            ParamDef::labelled("step", "step size", 0.95, 0.4, 1.4, 0.01),
            ParamDef::labelled("wall", "frame width", 0.16, 0.05, 0.4, 0.005),
            ParamDef::labelled("twist", "twist °", 4.0, -15.0, 15.0, 0.1),
            ParamDef::labelled("rim", "rim", 0.6, 0.0, 1.5, 0.01),
            ParamDef::labelled("count", "crystals", 3.0, 1.0, 6.0, 1.0),
            ParamDef::labelled("seed", "seed", 2.0, 1.0, 999.0, 1.0),
            ParamDef::labelled("grow", "grown", 1.0, 0.0, 1.0, 0.002),
        ],
        rot: Some([38.0, 20.0, 0.0]),
        animate: Some(vec!["grow"]),
        table: formula(|v| bismuth(v, &WHOLE)).1,
        ..Preset::new("bismuth", "Bismuth hopper", formula(|v| bismuth(v, &WHOLE)).0)
    };
    let quartz = Preset {
        eq: vec!["six sides n·p ≤ r, six tip faces at θ to the axis", "a cluster from a rough base"],
        note: "Rock crystal: hexagonal prisms ending in six faces, grown out of a stone. Every seed is another cluster.",
        params: vec![
            ParamDef::labelled("seed", "seed", 4.0, 1.0, 999.0, 1.0),
            ParamDef::labelled("count", "crystals", 9.0, 1.0, 40.0, 1.0),
            ParamDef::labelled("spread", "spread °", 62.0, 0.0, 85.0, 0.5),
            ParamDef::labelled("length", "length", 2.2, 0.5, 3.0, 0.01),
            ParamDef::labelled("slender", "slender", 5.2, 2.0, 9.0, 0.05),
            ParamDef::labelled("tips", "tip faces °", 52.0, 35.0, 70.0, 0.5),
            ParamDef::labelled("base", "base", 0.55, 0.0, 1.2, 0.01),
            ParamDef::labelled("grow", "grown", 1.0, 0.0, 1.0, 0.002),
        ],
        rot: Some([10.0, 0.0, 0.0]),
        animate: Some(vec!["grow"]),
        table: formula(|v| quartz(v, &WHOLE)).1,
        ..Preset::new("quartz", "Quartz cluster", formula(|v| quartz(v, &WHOLE)).0)
    };
    let (geode_shape, geode_table) = geode_formula();
    let geode = Preset {
        eq: vec!["a shell: |p| ≤ 1, |p| ≥ 1 − t, broken at z = open", "lined with quartz points, pointing in"],
        note: "A hollow stone broken open: inside, quartz grew inward from the wall, point by point.",
        params: vec![
            ParamDef::labelled("seed", "seed", 3.0, 1.0, 999.0, 1.0),
            ParamDef::labelled("points", "points", 260.0, 0.0, 700.0, 1.0),
            ParamDef::labelled("size", "point size", 0.2, 0.06, 0.4, 0.002),
            ParamDef::labelled("shell", "shell", 0.16, 0.06, 0.4, 0.002),
            ParamDef::labelled("open", "opening", 0.25, -0.6, 0.95, 0.01),
            ParamDef::labelled("grow", "grown", 1.0, 0.0, 1.0, 0.002),
        ],
        rot: Some([12.0, -20.0, 0.0]),
        animate: Some(vec!["grow"]),
        table: geode_table,
        ..Preset::new("geode", "Geode", geode_shape)
    };
    Family {
        presets: vec![bismuth, quartz, geode],
        melt: 0.0,
        layers: 1,
        defaults: vec![
            ("ior", Setting::Num(1.54)), ("absorb", Setting::Vec3([0.12, 0.06, 0.03])),
            ("dist", Setting::Num(6.6)), ("fov", Setting::Num(26.0)), ("rot", Setting::Vec3([20.0, 20.0, 0.0])),
        ],
        lights: solid_lights(),
        ..Family::new(FAMILY, "Crystals", Kind::Field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(id: &str, set: &[(usize, f64)]) -> RawCurve {
        let f = family();
        let p = f.preset(id).unwrap();
        let mut v = p.defaults();
        for &(i, x) in set { v[i] = x; }
        match id { "bismuth" => bismuth(&v, &WHOLE), "quartz" => quartz(&v, &WHOLE), _ => geode_build(&v, true) }
    }

    #[test]
    fn every_crystal_is_made_of_pieces_with_boxes() {
        for id in ["bismuth", "quartz", "geode"] {
            let raw = build(id, &[]);
            assert!(raw.pieces.len() > 5, "{id}");
            assert!(!raw.frame.is_empty(), "{id}");
            for p in &raw.pieces {
                assert!(!p.terms.is_empty() && (0..3).all(|k| p.lo[k] < p.hi[k]), "{id}");
            }
        }
    }

    #[test]
    fn growing_keeps_the_frame_and_has_less() {
        for (id, g) in [("bismuth", bi::GROW), ("quartz", qz::GROW), ("geode", ge::GROW)] {
            let whole = build(id, &[]);
            let half = build(id, &[(g, 0.4)]);
            assert_eq!(whole.frame, half.frame, "{id}");
            let vol = |r: &RawCurve| r.pieces.iter().map(|p| (0..3).map(|k| p.hi[k] - p.lo[k]).product::<f64>()).sum::<f64>();
            assert!(vol(&half) < vol(&whole), "{id}");
        }
    }

    #[test]
    fn a_quartz_point_holds_its_axis_and_apex() {
        let p = quartz_point([0.0; 3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 2.0, 0.3, 52.0).unwrap();
        let inside = |q: V| p.terms.iter().all(|t| match t { Term::Plane { n, d } => dot(*n, q) <= *d + 1e-9, _ => true });
        assert!(inside([0.0, 1.0, 0.0]));
        assert!(inside([0.0, 1.999, 0.0]));
        assert!(!inside([0.0, 2.01, 0.0]));
        assert!(!inside([0.31, 0.5, 0.0]));
        assert!(!inside([0.0, -0.01, 0.0]));
    }

    #[test]
    fn the_geode_grows_inside_one_static_formula_table() {
        let family = family();
        let preset = family.preset("geode").unwrap();
        let mut values = preset.defaults();
        let table = preset.table.as_ref().unwrap();
        let complete = table(&values, "");
        values[ge::GROW] = 0.25;
        assert_eq!(complete, table(&values, ""));
        let shape = geode_build(&values, true);
        // Each prism's side and tip planes intersect at the same surface as
        // its analytic hexagonal cross-section at this point in playback.
        let mut seeds = Vec::new();
        let mut full = values.clone();
        full[ge::GROW] = 1.0;
        geode_points(&full, true, |s| seeds.push(s));
        let live: Vec<_> = seeds.into_iter().filter(|s| s.start < values[ge::GROW]).collect();
        assert_eq!(shape.pieces.len(), live.len() + 1);
        let mut rng = Rng::new(31.0);
        for (point, piece) in live.iter().zip(&shape.pieces[1..]) {
            let g = part(values[ge::GROW], point.start);
            for _ in 0..20 {
                let p = add(point.base, mul(rng.dir(), rng.next() * point.length));
                let truth = piece.terms.iter().filter_map(|term| match term {
                    Term::Plane { n, d } => Some(dot(*n, p) - d),
                    _ => None,
                }).fold(f64::NEG_INFINITY, f64::max);
                let q = sub(p, point.base);
                let (x, y, z) = (dot(q, point.side).abs(), dot(q, cross(point.axis, point.side)).abs(), dot(q, point.axis));
                let radial = x.max(0.5 * x + 0.5 * 3f64.sqrt() * y);
                let length = point.length * g + 0.6 * point.radius + 0.02;
                let radius = point.radius * (0.4 + 0.6 * g);
                let tip = 52.0 * DEG;
                let distance = (-z).max(radial - radius).max(tip.cos() * (z - length) + tip.sin() * radial);
                assert!((truth - distance).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn building_is_quick() {
        for id in ["bismuth", "quartz", "geode"] {
            let t = std::time::Instant::now();
            let _ = build(id, &[]);
            assert!(t.elapsed().as_secs_f64() < 0.05, "{id} took {:?}", t.elapsed());
        }
    }
}

#[cfg(test)]
mod sizes {
    use super::*;
    /// Every crystal's pieces fit in the formula's table, the geode with as many points as it may have.
    #[test]
    fn every_crystal_fits_its_table() {
        for p in family().presets {
            let mut v = p.defaults();
            for (i, d) in p.params.iter().enumerate() { if d.id == "points" || d.id == "count" { v[i] = d.hi; } }
            let t = crate::field::table(&p, &v, "");
            let n = t.len() / 4;
            assert!(n <= crate::field::TABLE_MAX && t[0] as usize == n, "{}: {n} entries", p.id);
            eprintln!("{}: {n} entries", p.id);
        }
    }
}

#[cfg(test)]
mod lists {
    use super::*;
    /// Read through the cell lists, the crystals' formula is what measuring every piece gives (near them).
    #[test]
    fn the_lists_agree_with_every_piece() {
        for id in ["bismuth", "geode"] {
            let f = family();
            let p = f.preset(id).unwrap();
            let v = p.defaults();
            let t = (p.table.as_ref().unwrap())(&v, "");
            let mut r = Rng::new(2.0);
            let mut bad = 0;
            for _ in 0..3000 {
                let q = [1.4 * r.sym(), 1.4 * r.sym(), 1.4 * r.sym()];
                let d = exact::tests::dist(&t, q);
                let truth = exact::tests::dist_by(&t, q, true);
                // (far off, both are bounds; a box's distance may be the closer one)
                if truth < 0.02 && (d - truth).abs() > 1e-4 { bad += 1; if bad < 5 { eprintln!("{id} {q:?}: {d} vs {truth}"); } }
            }
            assert_eq!(bad, 0, "{id}");
        }
    }
}

#[cfg(test)]
mod build_time {
    use super::*;
    #[test]
    fn table_build_times() {
        for p in family().presets.iter().chain(crate::presets::foam::family().presets.iter()) {
            let v = p.defaults();
            let t = std::time::Instant::now();
            let tb = crate::field::table(p, &v, "");
            eprintln!("{}: {:.1} ms, {} entries", p.id, t.elapsed().as_secs_f64() * 1e3, tb.len() / 4);
            if p.id == "geode" {
                let t = std::time::Instant::now();
                let raw = geode_build(&v, true);
                let t1 = t.elapsed().as_secs_f64() * 1e3;
                let _ = exact::table(&raw, crate::FIT);
                eprintln!("geode: build {t1:.1} ms, table {:.1} ms", t.elapsed().as_secs_f64() * 1e3 - t1);
            }
        }
    }
}
