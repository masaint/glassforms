//! Snowflakes as exact formulas: a crystal of flat-faced ice prisms, its every edge sharp at any zoom.
//!
//! **The surface.** The flake is six-fold, and mirror symmetric within each arm, so space is folded into one
//! 30° wedge (the angle about the plate's normal, folded by sixty degrees and then by its mirror) and a point
//! is measured only against the pieces of its own wedge, and against their mirror images across the wedge's
//! far side (the 30° line, where one arm's branches meet the next arm's). Every piece is an exact prism:
//!   - the heart: a hexagonal plate, its corners on the arms, its two faces low hexagonal pyramids;
//!   - arms and branches: straight flat bars of ice with a ridge along the middle (each face two flat
//!     facets meeting at the ridge), their tips pointed at the ice's 120° angle, their roots buried in
//!     whatever they grew from;
//!   - growth rims on a broad plate: low ridges parallel to its sides.
//! Each is the intersection of a few half-spaces: exact inside, and never more than the distance outside.
//! The flake is their union. Nothing is sampled: the tracer settles onto the faces and takes their normals
//! from this formula (traced as "csg"); the volume is the same formula baked into the grid.
//!
//! **The growth** (the table). The weather sets how the crystal grows, as on the Nakaya diagram (`habit`):
//! how readily its corners outrun its faces (branching), and whether it grows as a plate or a stubby
//! column. The crystal then grows in time: first a hexagonal plate, which stops spreading once its corners
//! become unstable; then six arms from its corners, at one speed. Where an arm's tip passes, side branches
//! sprout at intervals (closer the wetter the air: a tip sheds branches as its rounded end grows unstable),
//! and grow outwards at a share of the tip's speed for as long as the crystal grows, until they come near
//! the next arm's branches, which shield them. So branches near the heart are old but cut short by their
//! neighbours, those near the tip young and short, and the longest in between: a fern. In very wet air the
//! branches branch again. Each piece keeps the time it began and the time it was whole, so `grown` cuts the
//! crystal exactly where it stood at that time.
//!
//! The pieces go to the shader as a table (`Preset::table`: `tbl`) of three entries each, worked out on
//! the CPU in well under a millisecond whenever the weather or the seed changes.

use super::*;
use crate::field;
use geo::*;

pub const FAMILY: &str = "snowflakes";

// the sliders' places
const TEMP: usize = 0;
const HUMID: usize = 1;
const SEED: usize = 2;

/// The arms' tips, from the middle (the volume spans ±1.5).
const TIP: f64 = 1.2;
/// The most pieces the table holds (three entries each, after the count and a head for each group).
const MAX_PIECES: usize = 80;
const MAX_GROUPS: usize = 40;

fn params(t: f64, h: f64, seed: f64) -> Vec<ParamDef> {
    vec![
        ParamDef::labelled("temp", "temperature °C", t, -25.0, -1.0, 0.1),
        ParamDef::labelled("humid", "humidity", h, 0.0, 1.0, 0.01),
        ParamDef::labelled("seed", "seed", seed, 1.0, 999.0, 1.0),
        ParamDef::labelled("grow", "grown", 1.0, 0.0, 1.0, 0.002),
        ParamDef::labelled("relief", "relief", 1.0, 0.0, 3.0, 0.01),
        ParamDef::labelled("plate", "plate thickness", 1.0, 0.3, 3.0, 0.01),
    ]
}

/// How the weather makes the crystal grow: how readily it branches (0: a plain plate .. 1: a fern) and how
/// far it is a stubby column (0..1), for a temperature (°C) and humidity (0..1). Branching is likeliest at
/// −15 °C, with shoulders near −12 and −18 that give sectored and stellar plates; columns grow near −6;
/// dry air gives plates at any temperature.
pub fn habit(t: f64, h: f64) -> (f64, f64) {
    let bell = |c: f64, w: f64| (-((t - c) / w).powi(2)).exp();
    let branchy = (bell(-15.0, 2.2) + 0.55 * bell(-12.0, 1.6) + 0.55 * bell(-18.0, 1.8)).min(1.0);
    let column = bell(-6.5, 1.8);
    let b = (h.clamp(0.0, 1.0) * (0.1 + 0.9 * branchy) * (1.0 - 0.8 * column)).clamp(0.0, 1.0);
    (b, column)
}

/// One piece: a bar from a to b (`w` its half-width, `t` its half-thickness at its edges, `h` how far its
/// ridge stands above that; `point`: its tip pointed at 120°), or (`hex`) the hexagonal plate of corner
/// radius a.x. It begins at time t0 and is whole at t1.
#[derive(Clone, Copy, Debug)]
pub struct Bar {
    pub a: [f64; 2],
    pub b: [f64; 2],
    pub w: f64,
    pub t: f64,
    pub h: f64,
    pub hex: bool,
    pub point: bool,
    pub t0: f64,
    pub t1: f64,
    /// 0: the heart (plate, rims, arm); then each side branch, with the branches it puts out, a group of its own
    pub group: usize,
}

/// The crystal grown in this weather, as pieces in the wedge of the first arm (along +x, branches to its
/// +60° side).
pub fn grow(temp: f64, humid: f64, seed: f64) -> Vec<Bar> {
    let (b, column) = habit(temp, humid);
    let mut r = Rng::new(seed * 7919.0 + 3.0);
    let mut out: Vec<Bar> = Vec::new();
    let (c60, s60) = (0.5, 0.75f64.sqrt());
    // the plate's share of the whole: broad when branching is slow, a small heart for a fern
    let pf = (0.13 + 0.82 * (1.0 - b).powf(1.8)).min(0.95);
    let rp = TIP * pf;
    // half-thickness: a thin plate, a stubby column near -6
    let t = 0.016 * (1.0 + 7.0 * column);
    let rib = 0.009 + 0.007 * b;
    // (its faces rise less than the arms' ridges, which run across it, unless it is all there is)
    let lone = pf > 0.9;
    out.push(Bar { a: [rp, 0.0], b: [0.0; 2], w: 0.0, t, h: if lone { 1.3 } else { 0.55 } * rib, hex: true, point: false, t0: 0.0, t1: pf, group: 0 });
    // growth rims on a broad plate, parallel to its sides
    if rp > 0.35 {
        for f in [0.46, 0.74] {
            let rr = rp * f;
            out.push(Bar { a: [rr, 0.0], b: [rr * 0.75, rr * 0.25 * 3f64.sqrt()], w: 0.007, t, h: 0.95 * rib, hex: false, point: false, t0: f * pf, t1: f * pf + 0.02, group: 0 });
        }
    }
    // a plate alone (dry air, or a column)
    if lone { return out; }
    // the arm, from the middle to the tip: its ridge runs across the plate (the ribs of a sectored plate),
    // and its end leaves the plate's corner just as the plate stops (time and radius go together: t0 = 0)
    let w_arm = 0.021 + 0.10 * (1.0 - b).powf(1.6);
    let speed = (TIP - rp) / (1.0 - pf);
    out.push(Bar { a: [0.0, 0.0], b: [TIP, 0.0], w: w_arm, t, h: rib, hex: false, point: true, t0: 0.0, t1: 1.0, group: 0 });
    // a stellar plate's arm ends in a broad plate
    if (0.28..0.72).contains(&b) {
        let len = 4.0 * w_arm;
        out.push(Bar { a: [TIP - len, 0.0], b: [TIP, 0.0], w: 1.9 * w_arm, t, h: 0.8 * rib, hex: false, point: true, t0: 1.0 - len / speed, t1: 1.0, group: 0 });
    }
    if b < 0.25 { return out; }
    // side branches, sprouting where the tip passes
    // (in the wettest air the branches are fewer and branch again in the room between them)
    let twice = b > 0.92;
    let gap = (0.05 + 0.22 * (1.0 - b).powf(1.5)) * if twice { 2.0 } else { 1.0 };
    let share = 0.24 + 0.22 * b;
    let w_side = w_arm * (0.5 + 0.35 * (1.0 - b));
    let mut s = rp + gap * (0.4 + 0.5 * r.next());
    let mut group = 0;
    while s < TIP - 2.5 * w_arm && out.len() < MAX_PIECES && group + 1 < MAX_GROUPS {
        let next = s + gap * (1.0 + 0.35 * r.sym());
        let ts = pf + (s - rp) / speed;
        let w = w_side * (1.0 + 0.15 * r.sym());
        // grown for as long as the crystal grows, until it nears the next arm's branches (its axis at length
        // l is (s - l) / 2 from the wedge's far side, where that arm's branch meets it)
        let by_time = (1.0 - ts) * speed * share * (1.0 + 0.2 * r.sym());
        let room = s - 2.4 * w - 0.012;
        let len = by_time.min(room);
        if len > 2.5 * w {
            group += 1;
            let a = [s, 0.0];
            let dir = [c60, s60];
            out.push(Bar { a, b: [s + len * c60, len * s60], w, t, h: 0.75 * rib, hex: false, point: true, t0: ts, t1: ts + len / (speed * share), group });
            // in very wet air the branches branch again: off a branch's outer side, parallel to the arm, and
            // off its inner side, towards the next arm
            if twice && len > 0.16 {
                let w2 = 0.6 * w;
                // Give the longest branches at most three pairs, spaced along their length.
                let gap2 = (gap * 0.8).max((len - 2.0 * w) / 3.0);
                let mut u = 2.2 * w + gap2 * (0.5 + 0.5 * r.next());
                for _ in 0..3 {
                    if u >= len - 2.0 * w || out.len() + 2 > MAX_PIECES { break; }
                    let p = [s + u * dir[0], u * dir[1]];
                    let tu = ts + u / (speed * share);
                    let by_time = (1.0 - tu) * speed * share * 0.5;
                    // outward, parallel to the arm: until the next branch out
                    let out_len = by_time.min(next - s - 2.4 * w2 - w);
                    if out_len > 2.5 * w2 {
                        out.push(Bar { a: p, b: [p[0] + out_len, p[1]], w: w2, t, h: 0.55 * rib, hex: false, point: true, t0: tu, t1: tu + out_len / (speed * share * 0.5), group });
                    }
                    // inward at 120°: until the wedge's far side (then the next arm's twin takes over), or the
                    // branch before
                    let in_len = by_time.min((s - u) / 2.0 - 1.3 * w2 - 0.01).min(gap - 2.4 * w2 - w);
                    if in_len > 2.5 * w2 {
                        out.push(Bar { a: p, b: [p[0] - in_len * c60, p[1] + in_len * s60], w: w2, t, h: 0.55 * rib, hex: false, point: true, t0: tu, t1: tu + in_len / (speed * share * 0.5), group });
                    }
                    u += gap2 * (1.0 + 0.3 * r.sym());
                }
            }
        }
        s = next;
    }
    out
}

/// The table the formula reads: how many entries it has, then the groups, each a head and its pieces.
/// A head is the band of the group's slant coordinate c = x - y / sqrt 3 (constant along a side branch,
/// which runs at 60 degrees to the arm), widened by its pieces' widths, and how many pieces follow: a point
/// is measured against a group only when that band is nearer than what is nearest so far (0.866 |dc| is
/// the distance to a line of constant c). The heart's band is everything. Each piece is three entries
/// (a, direction; w, t, h, length or -1 for the plate; t0, 1 / (t1 - t0), point).
pub fn table(v: &[f64]) -> Vec<[f32; 4]> {
    let g = |i: usize, d: f64| v.get(i).copied().unwrap_or(d);
    let bars = grow(g(TEMP, -15.0), g(HUMID, 1.0), g(SEED, 1.0));
    let f = |x: f64| x as f32;
    let c = |p: [f64; 2]| p[0] - p[1] / 3f64.sqrt();
    let mut t = vec![[0.0f32; 4]];
    let mut i = 0;
    while i < bars.len() {
        let grp = bars[i].group;
        let j = (i..bars.len()).find(|&k| bars[k].group != grp).unwrap_or(bars.len());
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        for b in &bars[i..j] {
            // (a piece reaches its width beyond its ends and sides, its root a width back: 1.2 width in c)
            let m = 2.4 * b.w + 0.002;
            for p in [b.a, b.b] {
                lo = lo.min(c(p) - m);
                hi = hi.max(c(p) + m);
            }
        }
        if grp == 0 { (lo, hi) = (-1e9, 1e9); }
        t.push([f(lo), f(hi), (j - i) as f32, 0.0]);
        for b in &bars[i..j] {
            // (the bar's direction and length worked out here, not at every evaluation)
            let (dx, dy) = (b.b[0] - b.a[0], b.b[1] - b.a[1]);
            let l = (dx * dx + dy * dy).sqrt().max(1e-9);
            if b.hex {
                t.push([f(b.a[0]), 0.0, 0.0, 0.0]);
            } else {
                t.push([f(b.a[0]), f(b.a[1]), f(dx / l), f(dy / l)]);
            }
            t.push([f(b.w), f(b.t), f(b.h), if b.hex { -1.0 } else { f(l) }]);
            t.push([f(b.t0), f(1.0 / (b.t1 - b.t0).max(1e-4)), if b.point { 1.0 } else { 0.0 }, 0.0]);
        }
        i = j;
    }
    t[0][0] = t.len() as f32;
    t
}

/// The formula: the plate in the xy plane, its normal along z.
const WGSL: &str = "// raw: a distance
// (each piece the intersection of half-spaces: exact inside, never more than the distance outside)
const SF_C30: vec2f = vec2f(0.8660254, 0.5);
// the angle about the plate's normal folded into 0..30 degrees: six-fold, then mirror
fn sf_fold(q: vec2f) -> vec2f {
  let v = abs(q);
  // The first quadrant spans one and a half arms. Reflect across 30 degrees,
  // then fold the negative half of the central arm upward.
  let n = vec2f(-0.5, 0.8660254);
  return abs(v - 2.0 * max(dot(v, n), 0.0) * n);
}
// one piece at the folded point q, height z; `g` how far grown; `best` the nearest so far (a piece whose
// bounds lie beyond it is not measured)
fn sf_piece(q: vec2f, z: f32, i: u32, g: f32, best: f32) -> f32 {
  let A = tbl[i];
  let B = tbl[i + 1u];
  let C = tbl[i + 2u];
  let f = clamp((g - C.x) * C.y, 0.0, 1.0);
  if (f <= 0.0) { return 1e9; }
  let t = B.y * pr.plate;
  let h = B.z * pr.relief;
  if (B.w < 0.0) {
    // the hexagonal plate: its side facing 30 degrees; its faces low pyramids rising h to the middle
    let ap = A.x * f * 0.8660254;
    let s = dot(q, SF_C30);
    let k = h / max(ap, 1e-5);
    return max(s - ap, (abs(z) + k * s - (t + h)) * inverseSqrt(1.0 + k * k));
  }
  let a = A.xy;
  let dir = A.zw;
  let L = B.w * f;
  let w = B.x;
  // (its bounds: a ball round the bar, and its height)
  if (max(length(q - a - dir * (0.5 * L)) - 0.5 * L - w, abs(z) - t - h) > best) { return 1e9; }
  let d = q - a;
  let along = dot(d, dir);
  let across = abs(d.x * dir.y - d.y * dir.x);
  // the tip: pointed at 120 degrees (two faces at 60 degrees to the bar), or square
  let tip = select(along - L, 0.5 * (along - L) + 0.8660254 * across, C.z > 0.5);
  // the faces: two facets each side, meeting at the ridge along the middle
  let k = h / max(w, 1e-5);
  let roof = (abs(z) + k * across - (t + h)) * inverseSqrt(1.0 + k * k);
  return max(max(across - w, -along - w), max(tip, roof));
}
fn raw(p: vec3f) -> f32 {
  let q = sf_fold(p.xy);
  // and its mirror across the wedge's far side (the 30 degree line), where the next arm's pieces are
  let q2 = vec2f(0.5 * q.x + 0.8660254 * q.y, 0.8660254 * q.x - 0.5 * q.y);
  // (each point's slant coordinate: groups whose band of it lies farther than the nearest so far are passed by)
  let c1 = q.x - 0.57735027 * q.y;
  let c2 = q2.x - 0.57735027 * q2.y;
  let end = u32(tbl[0].x);
  var d = 1e9;
  var i = 1u;
  for (var guard = 0u; guard < 256u; guard++) {
    if (i >= end) { break; }
    let H = tbl[i];
    let n = u32(H.z);
    let near1 = 0.8660254 * max(max(H.x - c1, c1 - H.y), 0.0) < d;
    let near2 = 0.8660254 * max(max(H.x - c2, c2 - H.y), 0.0) < d;
    if (!near1 && !near2) { i += 1u + 3u * n; continue; }
    for (var k = 0u; k < n; k++) {
      let at = i + 1u + 3u * k;
      if (near1) { d = min(d, sf_piece(q, p.z, at, pr.grow, d)); }
      if (near2) { d = min(d, sf_piece(q2, p.z, at, pr.grow, d)); }
    }
    i += 1u + 3u * n;
  }
  return d;
}
";

fn preset(id: &'static str, name: &'static str, t: f64, h: f64, seed: f64, note: &'static str) -> Preset {
    Preset {
        eq: vec!["folded into a 30° wedge: angle mod 60°, then mirrored", "a union of flat-faced ice prisms grown from the weather"],
        note,
        params: params(t, h, seed),
        rot: Some([28.0, 0.0, 0.0]),
        animate: Some(vec!["grow"]),
        table: Some(Box::new(|v: &[f64], _: &str| table(v))),
        ..Preset::new(id, name, Shape::Field { wgsl: WGSL.to_string(), clip: field::CLIP_DEFAULT.to_string(), shell: field::SHELL_DEFAULT.to_string() })
    }
}

pub fn family() -> Family {
    Family {
        presets: vec![
            preset("dendrite", "Stellar dendrite", -15.0, 0.88, 1.0, "Near −15 °C in wet air the arms' tips shed branches as they grow, and the branches branch again: a fern."),
            preset("sectored", "Sectored plate", -18.0, 0.3, 7.0, "Near −18 °C, less wet: a broad plate whose arms are ridged and whose branches stay short and broad."),
            preset("stellar", "Stellar plate", -12.0, 0.7, 3.0, "Near −12 °C: a star of broad, flat, ridged arms, each ending in a little plate."),
            preset("plate", "Hexagonal plate", -2.0, 0.25, 1.0, "Near −2 °C, or in dry air: a simple hexagon, its faces low pyramids with rims where it grew in steps."),
        ],
        defaults: vec![
            ("ior", Setting::Num(1.31)),
            ("absorb", Setting::Vec3([0.05, 0.03, 0.015])),
            ("rot", Setting::Vec3([28.0, 0.0, 0.0])),
        ],
        // A plate facing the camera shows what is behind it: a dim broad light straight behind (it glows
        // softly) and two narrow lights behind the flake, off to its sides, bent into view by the facets;
        // the solids' lights give the edges their glints.
        lights: {
            let mut l = solid_lights();
            l.push(light(0.0, 0.0, 30.0, 30.0, [0.12, 0.15, 0.21], 1.0));
            l.push(light(-28.0, 32.0, 10.0, 26.0, [0.9, 1.1, 1.5], 0.8));
            l.push(light(30.0, -24.0, 8.0, 22.0, [1.4, 1.0, 0.7], 0.8));
            l
        },
        // (one layer of glass is traced: a second shows nothing different on a thin plate)
        layers: 1,
        ..Family::new(FAMILY, "Snowflakes", Kind::Field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The formula's distance on the CPU, as the shader works it out (for the tests).
    fn dist(tbl: &[[f32; 4]], p: [f64; 3], g: f64) -> f64 {
        let (r, mut a) = ((p[0] * p[0] + p[1] * p[1]).sqrt(), p[1].atan2(p[0]));
        let s = std::f64::consts::PI / 3.0;
        a = (a - s * (a / s).round()).abs();
        let q = [r * a.cos(), r * a.sin()];
        let q2 = [0.5 * q[0] + 0.75f64.sqrt() * q[1], 0.75f64.sqrt() * q[0] - 0.5 * q[1]];
        let mut at = Vec::new();
        let mut i = 1;
        while i < tbl[0][0] as usize {
            let n = tbl[i][2] as usize;
            for k in 0..n { at.push(i + 1 + 3 * k); }
            i += 1 + 3 * n;
        }
        let mut d = 1e9f64;
        for &k in &at {
            let (aa, bb, cc) = (tbl[k].map(|x| x as f64), tbl[k + 1].map(|x| x as f64), tbl[k + 2].map(|x| x as f64));
            let f = ((g - cc[0]) * cc[1]).clamp(0.0, 1.0);
            if f <= 0.0 { continue; }
            for q in [q, q2] {
                let (t, h) = (bb[1], bb[2]);
                let v = if bb[3] < 0.0 {
                    let ap = aa[0] * f * 0.75f64.sqrt();
                    let s = q[0] * 0.75f64.sqrt() + q[1] * 0.5;
                    let k = h / ap.max(1e-5);
                    (s - ap).max((p[2].abs() + k * s - (t + h)) / (1.0 + k * k).sqrt())
                } else {
                    let dir = [aa[2], aa[3]];
                    let l = bb[3] * f;
                    let dd = [q[0] - aa[0], q[1] - aa[1]];
                    let along = dd[0] * dir[0] + dd[1] * dir[1];
                    let across = (dd[0] * dir[1] - dd[1] * dir[0]).abs();
                    let tip = if cc[2] > 0.5 { 0.5 * (along - l) + 0.75f64.sqrt() * across } else { along - l };
                    let k = h / bb[0].max(1e-5);
                    (across - bb[0]).max(-along - bb[0]).max(tip).max((p[2].abs() + k * across - (t + h)) / (1.0 + k * k).sqrt())
                };
                d = d.min(v);
            }
        }
        d
    }

    #[test]
    fn every_look_grows_and_fits_the_table() {
        for p in &family().presets {
            let t = table(&p.defaults());
            let n = t[0][0] as usize;
            assert!(n >= 5 && n == t.len() && n <= field::TABLE_MAX, "{}: {n} entries", p.id);
            assert!(t.iter().flatten().all(|v| v.is_finite()), "{}", p.id);
            // the heart is solid, and nothing reaches past the tips
            assert!(dist(&t, [0.0, 0.0, 0.0], 1.0) < 0.0, "{}", p.id);
            assert!(dist(&t, [TIP + 0.01, 0.0, 0.0], 1.0) > 0.0, "{}", p.id);
        }
        // the fern has many more pieces than the plate (in entries)
        let fern = table(&family().presets[0].defaults())[0][0];
        let plate = table(&family().presets[3].defaults())[0][0];
        assert!(fern > 60.0 && plate < 14.0, "{fern} {plate}");
        // the wettest fern still fits
        let mut v = family().presets[0].defaults();
        v[HUMID] = 1.0;
        assert!(table(&v)[0][0] as usize <= field::TABLE_MAX);
    }

    #[test]
    fn the_flake_is_sixfold_and_mirror_symmetric() {
        let t = table(&family().presets[0].defaults());
        for p in [[0.5, 0.2, 0.01], [0.9, 0.31, 0.0], [0.3, 0.05, 0.02], [0.7, 0.6, -0.01]] {
            let d = dist(&t, p, 1.0);
            for k in 1..6 {
                let a = std::f64::consts::PI / 3.0 * k as f64;
                let q = [p[0] * a.cos() - p[1] * a.sin(), p[0] * a.sin() + p[1] * a.cos(), p[2]];
                assert!((dist(&t, q, 1.0) - d).abs() < 1e-9);
            }
            assert!((dist(&t, [p[0], -p[1], p[2]], 1.0) - d).abs() < 1e-9);
        }
    }

    #[test]
    fn growing_only_cuts_the_crystal() {
        let t = table(&family().presets[0].defaults());
        // inside at a time, inside at every later time
        let mut grid = Vec::new();
        for i in 0..40 {
            for j in 0..40 {
                grid.push([-1.2 + 0.06 * i as f64, -1.2 + 0.06 * j as f64, 0.0]);
            }
        }
        let inside = |g: f64| grid.iter().filter(|p| dist(&t, **p, g) < 0.0).count();
        let (a, b, c) = (inside(0.3), inside(0.6), inside(1.0));
        assert!(a < b && b < c, "{a} {b} {c}");
        for p in &grid {
            if dist(&t, *p, 0.5) < 0.0 { assert!(dist(&t, *p, 0.9) < 1e-9); }
        }
    }

    #[test]
    fn the_formula_is_never_more_than_the_distance() {
        // a step of the formula's value never crosses the surface: at points outside, everything within the
        // value is outside too
        let t = table(&family().presets[1].defaults());
        let mut r = Rng::new(5.0);
        for _ in 0..3000 {
            let p = [1.3 * r.sym(), 1.3 * r.sym(), 0.08 * r.sym()];
            let d = dist(&t, p, 1.0);
            if d <= 0.002 { continue; }
            let dir = r.dir();
            let q = [p[0] + dir[0] * d * 0.999, p[1] + dir[1] * d * 0.999, p[2] + dir[2] * d * 0.999];
            assert!(dist(&t, q, 1.0) > -1e-9, "{p:?} {d}");
        }
    }

    #[test]
    fn every_piece_lies_in_its_groups_band() {
        // the shader passes a group by when its band is farther than the nearest so far: every point of a
        // piece, out to its width, must lie in the band
        for humid in [0.5, 0.88, 1.0] {
            let mut v = family().presets[0].defaults();
            v[HUMID] = humid;
            let t = table(&v);
            let mut i = 1;
            while i < t[0][0] as usize {
                let (lo, hi, n) = (t[i][0] as f64, t[i][1] as f64, t[i][2] as usize);
                for k in 0..n {
                    let (a, w, l) = (t[i + 1 + 3 * k], t[i + 2 + 3 * k][0] as f64, t[i + 2 + 3 * k][3] as f64);
                    if l < 0.0 { continue; }
                    for f in 0..=20 {
                        let u = f as f64 / 20.0 * l;
                        let x = a[0] as f64 + a[2] as f64 * u;
                        let y = a[1] as f64 + a[3] as f64 * u;
                        for (dx, dy) in [(w, 0.0), (-w, 0.0), (0.0, w), (0.0, -w)] {
                            let c = (x + dx) - (y + dy) / 3f64.sqrt();
                            assert!(c >= lo - 1e-6 && c <= hi + 1e-6, "humid {humid}: group at {i}, piece {k}: c {c} outside {lo}..{hi}");
                        }
                    }
                }
                i += 1 + 3 * n;
            }
        }
    }

    #[test]
    fn dry_air_gives_plates_and_wet_air_at_minus_fifteen_ferns() {
        assert!(habit(-15.0, 0.0).0 < 0.15);
        assert!(habit(-15.0, 1.0).0 > 0.9);
        assert!(habit(-2.0, 1.0).0 < 0.3);
        assert!(habit(-6.5, 0.8).1 > 0.9);
    }
}
