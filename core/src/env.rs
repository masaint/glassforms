//! The lights, painted into an environment map. Mirrors `GLASS.envTexture` in `shaders.js`.
//!
//! Soft rectangular lights at infinity are painted into a small equirectangular picture, so the tracer
//! pays one lookup per ray however many lights there are.
//!
//! What the texture must be, to match the web version:
//!   size        256 x 128 (`ENV_W` x `ENV_H`), one level: no mipmaps are made, and the tracer always reads level 0
//!   format      RGBA, 16-bit float (the web version hands WebGL 32-bit floats and asks for RGBA16F, so the
//!               values are rounded to half floats on the way in; `paint_half` gives those). Alpha is 1.
//!   filtering   linear, both when magnified and when minified
//!   wrapping    u (around the horizon) repeats; v (top to bottom) clamps to the edge
//!   rows        `paint` returns row 0 first, and row 0 is the top of the sky: upload the rows in the order
//!               they come, so that row j sits at v = (j + 0.5) / h
//!
//! How a direction finds its texel (`envMap` in the tracer; rd is a unit vector in the lights' own frame,
//! x to the right, y up, the camera looking down -z):
//!   u = atan2(rd.z, rd.x) / (2 pi) + 0.5        so u = 0.5 is +x, and u grows from +x towards +z
//!   v = acos(clamp(rd.y, -1, 1)) / pi           so v = 0 is straight up, v = 1 straight down
//! and the other way, texel (i, j) holds the light arriving from
//!   th = (j + 0.5) / h * pi,   ph = ((i + 0.5) / w - 0.5) * 2 pi,   d = (sin th cos ph, cos th, sin th sin ph)
//!
//! Beyond the web version (further down this file): the lights made brighter together (`paint_gain`),
//! the soft glows that give back the light an environment picture's ceiling took off (`Glow`,
//! `add_glows`), the lighting as nine smooth terms for clay (`Sh9` and the `sh_*` functions), and the
//! layout of the small picture of the lighting that a light room shows (`preview_dir`, `preview_uv`).

use crate::presets::Light;

pub const ENV_W: usize = 256;
pub const ENV_H: usize = 128;

// A light, ready to paint: the direction to its centre, the two directions across it, the tangents of
// its half-sizes.
struct Lamp {
    a: [f64; 3],
    u: [f64; 3],
    v: [f64; 3],
    tw: f64,
    th: f64,
    rgb: [f64; 3],
    soft: f64,
}

// Math.hypot as the browser works it out for two numbers: scaled by the larger, the squares summed with
// a correction for what each addition rounds away. The last digit can differ from libm's hypot.
fn hypot2(a: f64, b: f64) -> f64 {
    let (a, b) = (a.abs(), b.abs());
    let max = a.max(b);
    if max == 0.0 { return 0.0; }
    let (mut sum, mut comp) = (0.0f64, 0.0f64);
    for n in [a / max, b / max] {
        let summand = n * n - comp;
        let next = sum + summand;
        comp = (next - sum) - summand;
        sum = next;
    }
    sum.sqrt() * max
}

fn sstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// The lights painted into an equirectangular map: RGBA f32, `w` x `h`, row 0 (the top of the sky) first.
///
/// Each light: az 0 = straight behind the object, 90 = to the right, 180 = at the camera, -90 = left;
/// el: up; w, h: angular half-size; soft: the fraction of the half-size that is falloff. All in degrees.
/// `ambient` is added everywhere.
pub fn paint(lights: &[Light], ambient: [f64; 3], w: usize, h: usize) -> Vec<f32> {
    use std::f64::consts::PI;
    const D: f64 = PI / 180.0;
    let lamps: Vec<Lamp> = lights.iter().map(|l| {
        let ce = libm::cos(l.el * D);
        let a = [libm::sin(l.az * D) * ce, libm::sin(l.el * D), -libm::cos(l.az * D) * ce];
        let u = [a[2], 0.0, -a[0]];                                     // up x a
        let ul = hypot2(u[0], u[2]);
        let u = if ul > 1e-6 { [u[0] / ul, 0.0, u[2] / ul] } else { [1.0, 0.0, 0.0] };      // straight up or down: any horizontal axis will do
        let v = [a[1]*u[2] - a[2]*u[1], a[2]*u[0] - a[0]*u[2], a[0]*u[1] - a[1]*u[0]];
        Lamp { a, u, v, tw: libm::tan(l.w * D), th: libm::tan(l.h * D), rgb: l.rgb, soft: l.soft.max(1e-3) }
    }).collect();
    let mut px = vec![0.0f32; w * h * 4];
    for j in 0..h {
        let th = (j as f64 + 0.5) / h as f64 * PI;
        let (st, ct) = (libm::sin(th), libm::cos(th));
        for i in 0..w {
            let ph = ((i as f64 + 0.5) / w as f64 - 0.5) * 2.0 * PI;
            let d = [st * libm::cos(ph), ct, st * libm::sin(ph)];       // must match envMap() in the tracer
            let [mut r, mut g, mut b] = ambient;
            for l in &lamps {
                let da = d[0]*l.a[0] + d[1]*l.a[1] + d[2]*l.a[2];
                if da <= 1e-3 { continue; }
                let x = (d[0]*l.u[0] + d[1]*l.u[1] + d[2]*l.u[2]).abs() / da / l.tw;
                let y = (d[0]*l.v[0] + d[1]*l.v[1] + d[2]*l.v[2]).abs() / da / l.th;
                let m = (1.0 - sstep((x - (1.0 - l.soft)) / l.soft)) * (1.0 - sstep((y - (1.0 - l.soft)) / l.soft));
                r += l.rgb[0] * m;
                g += l.rgb[1] * m;
                b += l.rgb[2] * m;
            }
            let o = (j * w + i) * 4;
            px[o] = r as f32;
            px[o + 1] = g as f32;
            px[o + 2] = b as f32;
            px[o + 3] = 1.0;
        }
    }
    px
}

/// One number as the nearest 16-bit float (a tie goes to the even one), given as its bits.
pub fn half_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff { return sign | 0x7c00 | if man != 0 { 0x0200 } else { 0 }; }      // infinity, or not a number
    let e = exp - 127 + 15;                                 // the exponent as a half float stores it
    if e >= 0x1f { return sign | 0x7c00; }                  // too large: infinity
    // The digits kept, and the place below which the rest is rounded away. A carry out of the top digit
    // runs into the exponent, which is the right answer there too (up to and including infinity).
    let (kept, shift) = if e > 0 {
        (((e as u32) << 10) | (man >> 13), 13)
    } else {
        if e < -10 { return sign; }                         // less than half the smallest half float: zero
        let shift = (14 - e) as u32;                        // below the smallest ordinary half float: no hidden digit
        ((man | 0x0080_0000) >> shift, shift)
    };
    let rest = (man | if e > 0 { 0 } else { 0x0080_0000 }) & ((1u32 << shift) - 1);
    let mid = 1u32 << (shift - 1);
    let up = rest > mid || (rest == mid && kept & 1 == 1);
    sign | (kept + up as u32) as u16
}

/// `paint`, rounded to 16-bit floats: the bits an RGBA16F texture of the lights holds.
pub fn paint_half(lights: &[Light], ambient: [f64; 3], w: usize, h: usize) -> Vec<u16> {
    paint(lights, ambient, w, h).into_iter().map(half_bits).collect()
}

/// The lights of a rig painted as `paint` does, all of them (the light from everywhere too) made `gain`
/// times as bright. With a gain of 1 this is `paint` to the last bit.
pub fn paint_gain(lights: &[Light], ambient: [f64; 3], gain: f64, w: usize, h: usize) -> Vec<f32> {
    if gain == 1.0 { return paint(lights, ambient, w, h); }
    let scaled: Vec<Light> = lights.iter().map(|l| Light { rgb: [l.rgb[0] * gain, l.rgb[1] * gain, l.rgb[2] * gain], ..l.clone() }).collect();
    paint(&scaled, [ambient[0] * gain, ambient[1] * gain, ambient[2] * gain], w, h)
}

/// A soft round glow: light of `energy` (radiance times solid angle, per colour) arriving from around
/// `dir` (a unit vector in the lights' own frame), spread as a bell of angular width `sigma` (radians).
#[derive(Clone, Debug, PartialEq)]
pub struct Glow {
    pub dir: [f64; 3],
    pub energy: [f64; 3],
    pub sigma: f64,
}

/// Add glows to a map painted by `paint` (same size, same layout), each `gain` times as bright and
/// widened by `extra` radians (widths add as squares). A glow keeps its energy whatever its width.
pub fn add_glows(px: &mut [f32], glows: &[Glow], gain: f64, extra: f64, w: usize, h: usize) {
    use std::f64::consts::PI;
    if gain == 0.0 { return; }
    for g in glows {
        // a bell on the sphere: exp((cos(angle) - 1) / s2), which holds 2 pi s2 (1 - exp(-2 / s2)) in all
        let s2 = g.sigma * g.sigma + extra * extra;
        let norm = gain / (2.0 * PI * s2 * (1.0 - (-2.0 / s2).exp()));
        // nothing worth adding beyond the angle where the bell has fallen to a millionth
        let far = (1.0 - 13.8 * s2).max(-1.0).acos();
        let lat = g.dir[1].clamp(-1.0, 1.0).acos();
        for j in 0..h {
            let th = (j as f64 + 0.5) / h as f64 * PI;
            // the nearest this row comes to the glow is the difference of their latitudes
            if (th - lat).abs() > far { continue; }
            let (st, ct) = (th.sin(), th.cos());
            for i in 0..w {
                let ph = ((i as f64 + 0.5) / w as f64 - 0.5) * 2.0 * PI;
                let c = st * ph.cos() * g.dir[0] + ct * g.dir[1] + st * ph.sin() * g.dir[2];
                let m = ((c - 1.0) / s2).exp() * norm;
                let o = (j * w + i) * 4;
                px[o] += (g.energy[0] * m) as f32;
                px[o + 1] += (g.energy[1] * m) as f32;
                px[o + 2] += (g.energy[2] * m) as f32;
            }
        }
    }
}

// ---- the lighting as nine smooth terms -----------------------------------------------------------
// Matte clay does not need the picture of the lights, only how much light arrives around each direction.
// That is so smooth a function of direction that nine numbers per colour hold it (spherical harmonics up
// to the second order: Ramamoorthi & Hanrahan, "An Efficient Representation for Irradiance Environment
// Maps", 2001), and the tracer then needs no lookup at all to light the clay.

/// The nine terms of a lighting, per colour: the light from every direction weighed by each harmonic.
pub type Sh9 = [[f64; 3]; 9];

/// The nine harmonics in direction d (a unit vector).
pub fn sh_basis(d: [f64; 3]) -> [f64; 9] {
    let [x, y, z] = d;
    [
        0.282095,
        0.488603 * y, 0.488603 * z, 0.488603 * x,
        1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3.0 * z * z - 1.0), 1.092548 * x * z, 0.546274 * (x * x - y * y),
    ]
}

/// The nine terms of an equirectangular picture `w` x `h` with `stride` floats a texel (red, green, blue
/// first), row 0 at the top. `dir` gives the direction of a point of it (u, v in 0..1).
pub fn sh_of_map(px: &[f32], w: usize, h: usize, stride: usize, dir: impl Fn(f64, f64) -> [f64; 3]) -> Sh9 {
    use std::f64::consts::PI;
    let mut sh = [[0.0f64; 3]; 9];
    for j in 0..h {
        let v = (j as f64 + 0.5) / h as f64;
        // the patch of sky one texel of this row covers
        let dw = (2.0 * PI / w as f64) * ((j as f64 / h as f64 * PI).cos() - ((j + 1) as f64 / h as f64 * PI).cos());
        for i in 0..w {
            let b = sh_basis(dir((i as f64 + 0.5) / w as f64, v));
            let o = (j * w + i) * stride;
            for (k, bk) in b.iter().enumerate() {
                for c in 0..3 {
                    sh[k][c] += px[o + c] as f64 * bk * dw;
                }
            }
        }
    }
    sh
}

/// The direction of a point of the map `paint` makes.
pub fn paint_dir(u: f64, v: f64) -> [f64; 3] {
    use std::f64::consts::PI;
    let (th, ph) = (v * PI, (u - 0.5) * 2.0 * PI);
    [th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin()]
}

/// Which order of harmonics each of the nine terms belongs to.
const SH_ORDER: [usize; 9] = [0, 1, 1, 1, 2, 2, 2, 2, 2];

/// The nine terms of a glow (see `Glow`): those of a point of light of its energy, each order softened
/// as a bell of the glow's width softens it.
pub fn sh_of_glow(g: &Glow) -> Sh9 {
    let b = sh_basis(g.dir);
    let s2 = g.sigma * g.sigma;
    let soften = [1.0, (-s2).exp(), (-3.0 * s2).exp()];
    let mut sh = [[0.0f64; 3]; 9];
    for k in 0..9 {
        for c in 0..3 {
            sh[k][c] = g.energy[c] * b[k] * soften[SH_ORDER[k]];
        }
    }
    sh
}

/// a += b * gain
pub fn sh_add(a: &mut Sh9, b: &Sh9, gain: f64) {
    for k in 0..9 {
        for c in 0..3 {
            a[k][c] += b[k][c] * gain;
        }
    }
}

/// How much each order of harmonics counts for the light a matte surface gathers (given as the
/// brightness of a white one: the sum over the half of the sky it faces, weighted by the cosine, over pi).
pub const SH_MATTE: [f64; 3] = [1.0, 2.0 / 3.0, 0.25];

/// The nine terms ready for a shader: each multiplied by its harmonic's constant and by `bands` (one
/// factor per order), so that the light around direction (x, y, z) is
///   t0 + t1 y + t2 z + t3 x + t4 xy + t5 yz + t6 (3zz - 1) + t7 xz + t8 (xx - yy)
pub fn sh_terms(sh: &Sh9, bands: [f64; 3]) -> [[f32; 4]; 9] {
    const K: [f64; 9] = [0.282095, 0.488603, 0.488603, 0.488603, 1.092548, 1.092548, 0.315392, 1.092548, 0.546274];
    let mut out = [[0.0f32; 4]; 9];
    for k in 0..9 {
        for c in 0..3 {
            out[k][c] = (sh[k][c] * K[k] * bands[SH_ORDER[k]]) as f32;
        }
    }
    out
}

/// The same sum as the shader makes of `sh_terms`' output.
pub fn sh_eval(t: &[[f32; 4]; 9], d: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = d;
    let w = [1.0, y, z, x, x * y, y * z, 3.0 * z * z - 1.0, x * z, x * x - y * y];
    let mut out = [0.0; 3];
    for k in 0..9 {
        for c in 0..3 {
            out[c] += t[k][c] as f64 * w[k];
        }
    }
    out
}

// ---- the picture of the lighting -----------------------------------------------------------------
// The small picture a light room shows is an equirectangular view from where the shape is, the way the
// camera faces: its middle is straight ahead of the camera (behind the shape), its left and right edges
// are behind the camera, its top row straight up.

/// The direction (x right, y up, the camera looking down -z) that a point of that picture shows: u, v
/// are fractions of its width and height from the top left.
pub fn preview_dir(u: f64, v: f64) -> [f64; 3] {
    use std::f64::consts::PI;
    let (th, ph) = (v * PI, (u - 0.5) * 2.0 * PI);
    [th.sin() * ph.sin(), th.cos(), -th.sin() * ph.cos()]
}
/// Where on that picture a direction shows.
pub fn preview_uv(d: [f64; 3]) -> (f64, f64) {
    use std::f64::consts::PI;
    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-30);
    (0.5 + d[0].atan2(-d[2]) / (2.0 * PI), (d[1] / l).clamp(-1.0, 1.0).acos() / PI)
}
