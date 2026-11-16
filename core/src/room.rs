//! A photographer's studio room, made from nothing but arithmetic, for polished and milky bodies to
//! reflect when no environment picture is loaded.
//!
//! A metal shows nothing but its surroundings. With only a few soft lights on black, the reflections in
//! it are lights and nothing, and that reads as made by a computer: a real product photograph always
//! shows something of the room in the dark parts of the metal (a wall, the ceiling, the floor meeting
//! the walls, a window, the boards and stands round the set). So when an opaque or milky material is
//! chosen and no picture is loaded, this room is painted, dim, under the rig's lights into the copy of
//! the lights' map that those materials read (and only that copy: glass reads its own map, untouched).
//!
//! What is in it (the room's own frame: y up, the camera looking down -z, the shape at the origin):
//!   walls, ceiling, floor   a box round the set, of low and slightly uneven value: darker in its corners
//!                           and at the skirting, the ceiling crossed by darker beams and two lighting
//!                           rails, the floor a mottled grey
//!   a window                on the wall to the camera's right, behind it: frosted daylight in a frame
//!                           of panes, the brightest thing in the room after the lights (a tenth of them)
//!   boards                  a white bounce board and a black flag standing on the floor round the set
//!   the lights' bodies      each of the rig's lights is a box with a dark rim round its front, on a
//!                           thin stand down to the floor, so a light reflected in metal has an edge and
//!                           something holding it up
//! How bright the room is follows the rig: a room is lit by what its lights spill, so its value is a
//! small share of the lights' total light (`ROOM_SHARE`), held between a near-black room and a dim one.
//!
//! Everything here is our own work (no photograph goes into it), so it ships with the program.

use crate::math::{apply, Mat3};
use crate::presets::Light;

/// The room's average value is this share of the lights' average over the whole sphere.
pub const ROOM_SHARE: f64 = 0.10;
/// ... held between these (before the rig's `lightGain`).
pub const ROOM_LEAST: f64 = 0.006;
pub const ROOM_MOST: f64 = 0.035;

// The room's size: the floor, the ceiling, the walls to the sides, behind the shape, behind the camera.
const FLOOR: f64 = -1.0;
const CEILING: f64 = 2.6;
const SIDE: f64 = 4.5;
const BACK: f64 = -5.0;
const FRONT: f64 = 5.5;
/// how far from the shape a light stands, along the floor
const STAND_AT: f64 = 2.6;

const LUM: [f64; 3] = [0.2126, 0.7152, 0.0722];

fn sstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Smooth value noise, about 0 to 1, one bump about every unit.
fn hash(i: i64, j: i64, k: i64) -> f64 {
    let mut h = (i.wrapping_mul(73856093) ^ j.wrapping_mul(19349663) ^ k.wrapping_mul(83492791)) as u64;
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    (h & 0xffffff) as f64 / 16777215.0
}
fn noise(x: f64, y: f64, z: f64) -> f64 {
    let (i, j, k) = (x.floor(), y.floor(), z.floor());
    let (fx, fy, fz) = (x - i, y - j, z - k);
    let s = |t: f64| t * t * (3.0 - 2.0 * t);
    let (sx, sy, sz) = (s(fx), s(fy), s(fz));
    let (i, j, k) = (i as i64, j as i64, k as i64);
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let a = lerp(lerp(hash(i, j, k), hash(i + 1, j, k), sx), lerp(hash(i, j + 1, k), hash(i + 1, j + 1, k), sx), sy);
    let b = lerp(lerp(hash(i, j, k + 1), hash(i + 1, j, k + 1), sx), lerp(hash(i, j + 1, k + 1), hash(i + 1, j + 1, k + 1), sx), sy);
    lerp(a, b, sz)
}
/// Two octaves of it, about 0 to 1.
fn mottle(x: f64, y: f64, z: f64) -> f64 {
    0.65 * noise(x, y, z) + 0.35 * noise(2.7 * x + 11.0, 2.7 * y + 5.0, 2.7 * z + 3.0)
}

/// A light of the rig as the room places it: the direction to its middle and the two directions
/// across it (room frame), the tangents of its half-sizes.
struct Body {
    a: [f64; 3],
    u: [f64; 3],
    v: [f64; 3],
    tw: f64,
    th: f64,
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn unit(a: [f64; 3]) -> [f64; 3] {
    let l = dot(a, a).sqrt().max(1e-12);
    [a[0] / l, a[1] / l, a[2] / l]
}

fn body(l: &Light, to_room: &Mat3) -> Body {
    const D: f64 = std::f64::consts::PI / 180.0;
    // (the same placing as `env::paint`, then turned into the room)
    let ce = (l.el * D).cos();
    let a = [(l.az * D).sin() * ce, (l.el * D).sin(), -(l.az * D).cos() * ce];
    let u = [a[2], 0.0, -a[0]];
    let ul = (u[0] * u[0] + u[2] * u[2]).sqrt();
    let u = if ul > 1e-6 { [u[0] / ul, 0.0, u[2] / ul] } else { [1.0, 0.0, 0.0] };
    let v = [a[1] * u[2] - a[2] * u[1], a[2] * u[0] - a[0] * u[2], a[0] * u[1] - a[1] * u[0]];
    Body { a: apply(to_room, a), u: apply(to_room, u), v: apply(to_room, v), tw: (l.w * D).tan(), th: (l.h * D).tan() }
}

/// How bright the room is for these lights (before `gain`): a share of their light averaged over the
/// sphere, held between `ROOM_LEAST` and `ROOM_MOST`; and its tint, a little of the lights' colour.
pub fn level(lights: &[Light]) -> ([f64; 3], f64) {
    const D: f64 = std::f64::consts::PI / 180.0;
    let (mut power, mut rgb) = (0.0, [0.0; 3]);
    for l in lights {
        // a lit rectangle of half-tangents tw, th covers about 4 tw th / (1 + ...) of the sphere: near
        // enough for a share of it
        let tw = (l.w.min(80.0) * D).tan();
        let th = (l.h.min(80.0) * D).tan();
        let area = 4.0 * (tw * th / (1.0 + tw * tw + th * th).sqrt()).atan();
        // (a light that fades over its whole half-size gives about a third of a hard one's light)
        let fill = 1.0 - 0.6 * l.soft.clamp(0.0, 1.0);
        let y = dot(LUM, l.rgb);
        power += y * area * fill;
        for q in 0..3 { rgb[q] += l.rgb[q] * area * fill; }
    }
    let mean = power / (4.0 * std::f64::consts::PI);
    let v = (ROOM_SHARE * mean).clamp(ROOM_LEAST, ROOM_MOST);
    let y = dot(LUM, rgb).max(1e-9);
    // (warm grey, a third of the way to the lights' own colour)
    let warm = [1.04, 1.0, 0.94];
    let tint = [0, 1, 2].map(|q| warm[q] * (0.67 + 0.33 * rgb[q] / y));
    (tint, v)
}

/// Where a ray from the shape along r (room frame, unit) meets the box of the room, and which face:
/// 0 floor, 1 ceiling, 2 left, 3 right, 4 behind the shape, 5 behind the camera.
fn hit_box(r: [f64; 3]) -> (f64, usize) {
    let mut best = (f64::INFINITY, 0);
    let mut try_ = |t: f64, face: usize| if t > 0.0 && t < best.0 { best = (t, face) };
    if r[1] < 0.0 { try_(FLOOR / r[1], 0); }
    if r[1] > 0.0 { try_(CEILING / r[1], 1); }
    if r[0] < 0.0 { try_(-SIDE / r[0], 2); }
    if r[0] > 0.0 { try_(SIDE / r[0], 3); }
    if r[2] < 0.0 { try_(BACK / r[2], 4); }
    if r[2] > 0.0 { try_(FRONT / r[2], 5); }
    best
}

/// A board standing on the floor: its middle at floor level (x, z), facing the shape, half as wide as
/// `half_w`, `high` tall. The distance along r to it, if r meets it.
fn board(r: [f64; 3], x: f64, z: f64, half_w: f64, high: f64) -> Option<f64> {
    let n = unit([-x, 0.0, -z]);
    let c = [x, FLOOR, z];
    let dn = dot(r, n);
    if dn >= -1e-6 { return None; }
    let t = dot(c, n) / dn;
    if t <= 0.0 { return None; }
    let p = [r[0] * t, r[1] * t, r[2] * t];
    let across = [-n[2], 0.0, n[0]];
    let s = (p[0] - c[0]) * across[0] + (p[2] - c[2]) * across[2];
    (s.abs() < half_w && p[1] > FLOOR && p[1] < FLOOR + high).then_some(t)
}

/// The room's light along r (room frame, unit), before brightness and tint: about 1 on a wall.
fn surround(r: [f64; 3], bodies: &[Body]) -> f64 {
    let (mut t, face) = hit_box(r);
    let p = [r[0] * t, r[1] * t, r[2] * t];
    // the box: each face its own value, uneven, darker towards its corners
    let corner = |a: f64, lo: f64, hi: f64| {
        let d = (a - lo).min(hi - a).max(0.0);
        1.0 - 0.45 * (-d / 0.5).exp()
    };
    let mut v = match face {
        0 => {
            // floor: grey concrete, a little brighter than the walls (it faces the lights), mottled
            let m = mottle(p[0] * 0.6, 0.0, p[2] * 0.6);
            1.5 * (0.8 + 0.4 * m) * corner(p[0], -SIDE, SIDE) * corner(p[2], BACK, FRONT)
        }
        1 => {
            // ceiling: dark, crossed by beams every 1.3 units and two lighting rails along it
            let beam = ((p[2] / 1.3).fract().abs() - 0.5).abs() < 0.07;
            let rail = (p[0].abs() - 1.6).abs() < 0.05;
            let base = 0.55 * (0.85 + 0.3 * mottle(p[0] * 0.5, 3.0, p[2] * 0.5));
            let base = if beam { base * 0.4 } else if rail { base * 0.25 } else { base };
            base * corner(p[0], -SIDE, SIDE) * corner(p[2], BACK, FRONT)
        }
        _ => {
            // walls: brighter towards the top (they catch the lights' spill), a dark skirting, uneven
            let along = if face < 4 { p[2] } else { p[0] };
            let (lo, hi) = if face < 4 { (BACK, FRONT) } else { (-SIDE, SIDE) };
            let y = (p[1] - FLOOR) / (CEILING - FLOOR);
            let m = mottle(along * 0.45, p[1] * 0.45, face as f64 * 7.0);
            let mut w = (0.75 + 0.45 * sstep(0.0, 0.8, y)) * (0.85 + 0.3 * m) * corner(along, lo, hi) * corner(p[1], FLOOR, CEILING);
            if p[1] < FLOOR + 0.12 { w *= 0.35; }
            w
        }
    };
    // a frosted window on the wall to the camera's right, behind the camera: daylight in panes
    if face == 3 && p[2] > 0.8 && p[2] < 4.2 && p[1] > 0.0 && p[1] < 1.9 {
        let (s, y) = ((p[2] - 0.8) / 3.4 * 3.0, (p[1] - 0.0) / 1.9 * 2.0);
        let bar = |x: f64, n: f64| { let f = x - x.floor(); f < 0.03 * n || f > 1.0 - 0.03 * n };
        let frame = s < 0.04 || s > 2.96 || y < 0.04 || y > 1.96;
        v = if frame || bar(s, 1.0) || bar(y, 1.5) { 0.25 } else { 10.0 * (0.9 + 0.1 * (1.0 - y / 2.0)) };
    }
    // boards on the floor round the set: a white bounce board behind the camera to its left, a black
    // flag to the right of the shape
    if let Some(tb) = board(r, -2.3, 2.4, 0.65, 2.4) {
        if tb < t { t = tb; v = 3.0 * (0.9 + 0.1 * mottle(r[0] * 9.0, r[1] * 9.0, 1.0)); }
    }
    if let Some(tb) = board(r, 2.4, -1.2, 0.5, 2.2) {
        if tb < t { t = tb; v = 0.08; }
    }
    // the lights' bodies: a dark rim round each front, and a stand from under it to the floor
    for b in bodies {
        let da = dot(r, b.a);
        if da > 1e-3 {
            let x = dot(r, b.u).abs() / da / b.tw;
            let y = dot(r, b.v).abs() / da / b.th;
            let rim = |s: f64| s < 1.0 + 0.12 + 0.04 / b.tw.min(b.th).max(0.05);
            if rim(x) && rim(y) && (x > 0.9 || y > 0.9) { return 0.05; }
        }
        // the stand: a thin vertical pole at the light's place round the room, from the floor up to it
        let hz = (b.a[0] * b.a[0] + b.a[2] * b.a[2]).sqrt();
        if hz < 0.3 { continue; }
        let q = [b.a[0] / hz * STAND_AT, b.a[2] / hz * STAND_AT];
        let top = b.a[1] / hz * STAND_AT;
        let rh = (r[0] * r[0] + r[2] * r[2]).sqrt();
        if rh < 1e-6 { continue; }
        let s = (q[0] * r[0] + q[1] * r[2]) / (rh * rh);
        if s <= 0.0 || s > t { continue; }
        let miss = ((r[0] * s - q[0]).powi(2) + (r[2] * s - q[1]).powi(2)).sqrt();
        let yy = r[1] * s;
        if miss < 0.02 && yy > FLOOR && yy < top { return 0.12; }
    }
    v
}

/// Paint the room into a map laid out as `env::paint` lays it out (RGBA, row 0 the top of the sky),
/// added to what is there. `lights`: the rig (in the lights' own frame, whose lights the room places
/// its stands and rims round); `to_room`: the lights' frame -> the room; `gain`: the rig's `lightGain`.
pub fn paint(px: &mut [f32], w: usize, h: usize, lights: &[Light], to_room: &Mat3, gain: f64) {
    let (tint, v) = level(lights);
    let k = [0, 1, 2].map(|q| tint[q] * v * gain);
    let bodies: Vec<Body> = lights.iter().map(|l| body(l, to_room)).collect();
    if w == 0 || h == 0 { return; }
    // (rows in bands on several threads: a lighting change repaints it, and should not hold a frame up)
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 16);
    let rows = h.div_ceil(threads);
    std::thread::scope(|sc| {
        for (b, band) in px[..w * h * 4].chunks_mut(rows * w * 4).enumerate() {
            let bodies = &bodies;
            sc.spawn(move || {
                for (jj, row) in band.chunks_mut(w * 4).enumerate() {
                    let j = b * rows + jj;
                    for i in 0..w {
                        let d = crate::env::paint_dir((i as f64 + 0.5) / w as f64, (j as f64 + 0.5) / h as f64);
                        let s = surround(unit(apply(to_room, d)), bodies);
                        for q in 0..3 { row[i * 4 + q] += (k[q] * s) as f32; }
                    }
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: Mat3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    fn strip(az: f64) -> Light {
        Light { az, el: 15.0, w: 7.0, h: 50.0, rgb: [3.0, 3.0, 3.0], soft: 0.2 }
    }

    #[test]
    fn the_room_is_dim_beside_its_lights_and_has_something_in_it() {
        let lights = [strip(-70.0), strip(75.0)];
        let (w, h) = (256, 128);
        let mut lit = crate::env::paint(&lights, [0.0; 3], w, h);
        let peak = lit.chunks(4).map(|c| c[1]).fold(0.0f32, f32::max);
        let mut room = vec![0.0f32; w * h * 4];
        paint(&mut room, w, h, &lights, &ID, 1.0);
        for (a, b) in lit.iter_mut().zip(&room) { *a += b; }
        let g: Vec<f32> = room.chunks(4).map(|c| c[1]).collect();
        let mean = g.iter().sum::<f32>() / g.len() as f32;
        // much dimmer than the lights, never nothing, and not one flat value
        assert!(mean > 0.0 && (mean as f64) < 0.05 * peak as f64, "room {mean} lights {peak}");
        let (lo, hi) = (g.iter().cloned().fold(f32::MAX, f32::min), g.iter().cloned().fold(0.0f32, f32::max));
        assert!(lo > 0.0 && hi > 4.0 * mean, "from {lo} to {hi}");
        // brighter rigs light a brighter room, within its bounds; the lights' gain carries through
        let (_, a) = level(&lights);
        let (_, b) = level(&[strip(-70.0), strip(75.0), strip(0.0), strip(180.0)]);
        assert!(b > a && a >= ROOM_LEAST && b <= ROOM_MOST);
        let mut twice = vec![0.0f32; w * h * 4];
        paint(&mut twice, w, h, &lights, &ID, 2.0);
        assert!(twice.iter().zip(&room).all(|(x, y)| (x - 2.0 * y).abs() <= 1e-6 * x.abs().max(1.0)));
    }

    #[test]
    fn painting_the_room_at_the_size_a_mirror_reads_is_quick() {
        let lights = [strip(-70.0), strip(75.0), strip(0.0), strip(150.0)];
        let (w, h) = (1024, 512);
        let mut px = vec![0.0f32; w * h * 4];
        let t = std::time::Instant::now();
        paint(&mut px, w, h, &lights, &ID, 1.0);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!("room painted at {w} x {h} in {ms:.1} ms");
        assert!(ms < 100.0, "{ms} ms");
    }

    #[test]
    fn a_light_has_a_dark_rim_and_turns_with_the_room() {
        let l = strip(40.0);
        // just beyond the light's edge across it: the rim, darker than the wall beyond it
        let b = body(&l, &ID);
        let (a, u, tw) = (b.a, b.u, b.tw);
        let at = |x: f64| unit([a[0] + u[0] * tw * x, a[1] + u[1] * tw * x, a[2] + u[2] * tw * x]);
        let bodies = [b];
        assert!(surround(at(1.05), &bodies) < 0.5 * surround(at(3.0), &bodies));
        // the window, and the room turned half round its vertical carries it behind the shape
        let win = unit([1.0, 0.1, 0.55]);
        assert!(surround(win, &[]) > 5.0);
        let half: Mat3 = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        assert!(surround(unit(apply(&half, win)), &[]) < 2.0);
    }
}
