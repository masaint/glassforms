//! The studio a shape is photographed in: a backdrop, a floor, where the camera stands above it, and
//! how the finished picture is toned. Needs no GPU; `app/src/stage.rs` and `shaders/stage.wgsl` draw it.
//!
//! **The room.** Backdrop and floor belong to the room, not to the shape and not to the lights: the
//! shape turns in the room, the lights turn in the room, the room stays. The room's own frame has y up
//! and the camera looking down -z when it stands level; `room_m` gives how the camera sees it (rolled
//! by the camera's roll, tilted by how high the camera has been raised).
//!
//! **The backdrop** is a studio cyclorama: a colour that changes smoothly from one at the horizon to
//! another overhead and a third underfoot, with an optional round glow, as bright as it is set. It
//! need not go all the way round: a real one stands behind the shape, and the dark of the studio behind
//! the camera is what gives glass on a bright ground its dark edges. It is light like any other: it is painted into the lights' map under the lights (`paint`), so glass,
//! frosted glass, metal, milky bodies and clay all see it through the lookups they already make, and the
//! same function (`Backdrop::rgb`) is what the camera sees where a ray misses the shape. One function
//! for both is what makes the backdrop behind the glass and the backdrop seen through it agree.
//!
//! **The floor** is the plane y = height of the room. It is a matte surface lit by everything above
//! the horizon: lights, backdrop and the light from everywhere. Every one of those is infinitely far
//! away, so every point of the floor receives the same light and an unshadowed floor is one even colour:
//! its albedo times the light arriving, over pi. That is a floor as it would be lit, and it brightens
//! and darkens with the lights; a studio's sweep, though, is painted and lit so that floor and wall are
//! one colour, and that is the default here (`floorMatch`): the floor takes the backdrop's colour
//! underfoot, whatever the lights. Far from the shape it melts into the backdrop (a cove), so there is
//! no horizon line. Because it is one even colour, what the glass sees of it depends only on
//! the direction it looks in: the floor is painted over the lower half of the lights' map and costs the
//! tracer nothing. (What that leaves out: shadows and caustics on the floor are seen on the floor, not
//! through the glass.)
//!
//! **Tone and grade.** The ACES curve the program has always used darkens and bleaches bright pastel
//! pictures. `tone_curve` offers it beside two that leave the middle of the range alone, and none at
//! all; `Grade` is the handful of corrections made around the curve. The shader (`post.wgsl`) does the
//! same arithmetic in 32-bit floats.

use crate::math::{apply, axis_angle, mul, transpose, Mat3, DEG};
use crate::presets::Light;
use crate::settings::Settings;

const LUM: [f64; 3] = [0.2126, 0.7152, 0.0722];

/// How the camera sees the room: rolled about its line of sight, and tilted by how far the camera has
/// been raised on its arc (degrees; raised by 90 it looks straight down at the floor).
pub fn room_m(roll: f64, elev: f64) -> Mat3 {
    mul(&axis_angle(0.0, 0.0, 1.0, -roll * DEG), &axis_angle(1.0, 0.0, 0.0, elev * DEG))
}

fn sstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

// ---- the backdrop ------------------------------------------------------------------------------------

/// A backdrop as it is set: colours already made as bright as `backGain` says.
#[derive(Clone, Debug, PartialEq)]
pub struct Backdrop {
    pub top: [f64; 3],
    pub mid: [f64; 3],
    pub low: [f64; 3],
    /// degrees above (and below) the horizon over which the horizon's colour gives way
    pub spread: f64,
    /// how far round from straight behind the shape the backdrop reaches, degrees (180 = all the way)
    pub wrap: f64,
    /// how far up from the horizon it reaches, degrees (90 = to straight overhead)
    pub high: f64,
    /// the glow's colour (and brightness), the direction of its middle in the room, and its width in degrees
    pub glow: [f64; 3],
    pub glow_dir: [f64; 3],
    pub glow_size: f64,
}

impl Backdrop {
    /// The backdrop the settings describe; None when there is none (`backdrop` = 0: black, as ever).
    pub fn of(p: &Settings) -> Option<Backdrop> {
        if p.backdrop < 0.5 { return None; }
        let g = p.back_gain.max(0.0);
        let k = p.back_glow.max(0.0) * g;
        let (az, el) = (p.back_glow_az * DEG, p.back_glow_el * DEG);
        Some(Backdrop {
            top: [p.back_top_r * g, p.back_top_g * g, p.back_top_b * g],
            mid: [p.back_mid_r * g, p.back_mid_g * g, p.back_mid_b * g],
            low: [p.back_low_r * g, p.back_low_g * g, p.back_low_b * g],
            spread: p.back_spread.clamp(1.0, 90.0),
            wrap: p.back_wrap.clamp(0.0, 180.0),
            high: p.back_high.clamp(0.0, 90.0),
            glow: [p.back_glow_r * k, p.back_glow_g * k, p.back_glow_b * k],
            // (laid out as a light's direction is: 0 = straight behind the shape, 90 = to the right)
            glow_dir: [az.sin() * el.cos(), el.sin(), -az.cos() * el.cos()],
            glow_size: p.back_glow_size.clamp(1.0, 180.0),
        })
    }

    /// 1 / (the bell's width squared): the glow in direction d is `glow * exp((d . glow_dir - 1) * this)`.
    pub fn glow_k(&self) -> f64 {
        let s = self.glow_size * DEG;
        1.0 / (s * s)
    }

    /// Where the backdrop ends: it is whole while the cosine of the angle from straight behind the shape
    /// is above the first of these, and gone below the second (`WRAP_SOFT` degrees further round).
    pub fn wrap_cos(&self) -> (f64, f64) {
        if self.wrap >= 180.0 { return (-2.0, -3.0); }
        ((self.wrap * DEG).cos(), ((self.wrap + WRAP_SOFT).min(180.0) * DEG).cos().min((self.wrap * DEG).cos() - 1e-6))
    }

    /// Where the backdrop ends overhead: it is whole while the sine of the angle above the horizon is
    /// below the first of these, and gone above the second (`WRAP_SOFT` degrees further up).
    pub fn high_sin(&self) -> (f64, f64) {
        if self.high >= 90.0 { return (2.0, 3.0); }
        ((self.high * DEG).sin(), ((self.high + WRAP_SOFT).min(90.0) * DEG).sin().max((self.high * DEG).sin() + 1e-6))
    }

    /// The light arriving from a direction of the room, given as: the sine of how far above the horizon
    /// it is, its cosine to the middle of the glow, and its cosine to straight behind the shape.
    fn at(&self, sin_el: f64, gd: f64, behind: f64) -> [f64; 3] {
        let sweep = self.sweep(sin_el);
        let bell = ((gd - 1.0) * self.glow_k()).exp();
        let (whole, gone) = self.wrap_cos();
        let (under, over) = self.high_sin();
        let there = sstep((behind - gone) / (whole - gone)) * (1.0 - sstep((sin_el - under) / (over - under)));
        [0, 1, 2].map(|c| sweep[c] * there + self.glow[c] * bell)
    }

    /// The sweep's colour at a height (the sine of the angle above the horizon), as if it went all the
    /// way round and up: the horizon's colour giving way to the one overhead or the one underfoot. A
    /// floor that takes the backdrop's colour takes this, direction by direction, so that floor and wall
    /// are one sweep with no line where they meet.
    pub fn sweep(&self, sin_el: f64) -> [f64; 3] {
        let u = (sin_el.clamp(-1.0, 1.0).asin() / (self.spread * DEG)).clamp(-1.0, 1.0);
        let w = sstep(u.abs());
        let to = if u > 0.0 { &self.top } else { &self.low };
        [0, 1, 2].map(|c| self.mid[c] + (to[c] - self.mid[c]) * w)
    }

    /// The light arriving from direction `d` of the room (a unit vector).
    pub fn rgb(&self, d: [f64; 3]) -> [f64; 3] {
        self.at(d[1], d[0] * self.glow_dir[0] + d[1] * self.glow_dir[1] + d[2] * self.glow_dir[2], -d[2])
    }
}

/// The backdrop's edge is this many degrees wide: from whole to gone.
pub const WRAP_SOFT: f64 = 30.0;

// ---- the floor ---------------------------------------------------------------------------------------

/// A floor as it is set.
#[derive(Clone, Debug, PartialEq)]
pub struct Floor {
    /// the share of the light falling on it that it gives back, by colour (`floorR/G/B` times `floorGain`);
    /// with the backdrop's colour for the floor, what that colour is multiplied by
    pub albedo: [f64; 3],
    /// how far the floor's colour is the backdrop's underfoot (1) and not the light falling on it (0)
    pub matched: f64,
    /// the distance from the middle of the room at which the floor has wholly become backdrop
    pub fade: f64,
    /// 0 = matte, 1 = as glossy as it gets
    pub gloss: f64,
    /// the shape itself is reflected in a glossy floor
    pub mirror: bool,
}

/// The share of `Floor::fade` up to which the floor is wholly floor; the melt runs from there to `fade`.
pub const FADE_FROM: f64 = 0.35;
/// The height above the floor from which the glass is taken to look at it, for the melt alone: the glass
/// sees the floor by direction only, and a direction meets the floor nearer or further by the height it
/// starts from. A shape's middle stands about this far above a floor it rests on.
pub const EYE_HEIGHT: f64 = 1.0;

impl Floor {
    pub fn of(p: &Settings) -> Option<Floor> {
        if p.floor < 0.5 { return None; }
        let g = p.floor_gain.max(0.0);
        Some(Floor {
            albedo: [p.floor_r * g, p.floor_g * g, p.floor_b * g],
            matched: p.floor_match.clamp(0.0, 1.0),
            fade: p.floor_fade.max(0.5),
            gloss: p.floor_gloss.clamp(0.0, 1.0),
            mirror: p.floor_mirror > 0.5 && p.floor_gloss > 0.0,
        })
    }

    /// How far the floor has become backdrop at distance `r` from the middle of the room (0 = floor, 1 = backdrop).
    pub fn melt(&self, r: f64) -> f64 {
        sstep((r - FADE_FROM * self.fade) / ((1.0 - FADE_FROM) * self.fade))
    }

    /// The share of light a glossy floor mirrors, seen at the cosine `c` of the angle from straight down.
    /// A lacquered floor mirrors 4% seen from above and everything at a glancing angle; `gloss` both
    /// scales that and raises the share seen from above, because what a photographer calls a glossy floor
    /// (black acrylic, glass) is chosen for a reflection that shows.
    pub fn mirrors(&self, c: f64) -> f64 {
        let f0 = 0.04 + 0.36 * self.gloss;
        self.gloss * (f0 + (1.0 - f0) * (1.0 - c.clamp(0.0, 1.0)).powi(5))
    }
}

/// What a floor came to when the studio was painted.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FloorLight {
    /// the floor's own colour where nothing shadows it, seen from straight above
    pub own: [f64; 3],
    /// that colour in its two parts, for any direction it is seen in: `base` (what the light falling on
    /// it makes it: its albedo times the light arriving from above the horizon, over pi) plus `tint`
    /// times the backdrop's sweep in that direction (the share of it that is the backdrop's colour)
    pub base: [f64; 3],
    pub tint: [f64; 3],
    /// the part of the light a white matte floor gives back that comes from the backdrop alone (not
    /// from the lights, nor from the light from everywhere): for whoever works out what shadows take away
    pub backdrop: [f64; 3],
}

/// The studio as the settings describe it.
#[derive(Clone, Debug, PartialEq)]
pub struct Stage {
    pub backdrop: Option<Backdrop>,
    /// the backdrop is seen behind the shape (otherwise it only lights it)
    pub show: bool,
    pub floor: Option<Floor>,
}

impl Stage {
    pub fn of(p: &Settings) -> Stage {
        let backdrop = Backdrop::of(p);
        Stage { show: backdrop.is_some() && p.back_show > 0.5, backdrop, floor: Floor::of(p) }
    }
    /// Nothing of the studio is in use: the program draws what it always drew.
    pub fn off(&self) -> bool {
        self.backdrop.is_none() && self.floor.is_none()
    }
    /// The tracer has something of the room to draw where a ray misses the shape.
    pub fn seen(&self) -> bool {
        self.show || self.floor.is_some()
    }

    /// What `paint` depends on besides the lights: everything set, and how the room lies in the lights'
    /// frame as far as it matters (which way is up; where the glow is; which way is behind the shape, if
    /// the backdrop does not go all the way round). Lights that turn about the room's vertical leave
    /// "up" alone, so a backdrop that goes all the way round and has no glow need not be painted again
    /// while they circle.
    pub fn paint_key(&self, l2r: &Mat3) -> Vec<u64> {
        let mut k = Vec::new();
        let r2l = transpose(l2r);
        // (rounded: a rotation worked out afresh every frame differs in its last digits)
        let dir = |d: [f64; 3]| apply(&r2l, d).map(|c| ((c * 1e9).round() as i64) as u64);
        k.extend(dir([0.0, 1.0, 0.0]));
        if let Some(b) = &self.backdrop {
            for c in [b.top, b.mid, b.low, b.glow] { k.extend(c.map(f64::to_bits)); }
            k.extend([b.spread.to_bits(), b.glow_size.to_bits(), b.wrap.to_bits(), b.high.to_bits()]);
            if b.glow != [0.0; 3] { k.extend(dir(b.glow_dir)); }
            if b.wrap < 180.0 { k.extend(dir([0.0, 0.0, -1.0])); }
        }
        if let Some(f) = &self.floor {
            k.extend(f.albedo.map(f64::to_bits));
            k.extend([f.fade.to_bits(), f.gloss.to_bits(), f.matched.to_bits(), 1]);
        }
        k
    }

    /// Paint the studio into the lights' map (`px`: RGBA floats, `w` x `h`, laid out as `env::paint`
    /// lays it out, already holding the lights and the light from everywhere).
    ///   l2r     the lights' own frame -> the room
    ///   extra   light a white matte floor would give back from sources that are not in the map (an
    ///           environment picture), already divided by pi
    /// The backdrop is added in every direction. With a floor, everything below the room's horizon is
    /// replaced by the floor as the glass sees it (lights down there are hidden by it, as a real floor
    /// hides them). Returns what the floor came to (see `FloorLight`; all zero when there is no floor).
    pub fn paint(&self, px: &mut [f32], w: usize, h: usize, l2r: &Mat3, extra: [f64; 3]) -> FloorLight {
        use std::f64::consts::PI;
        let mut light = FloorLight::default();
        if self.off() { return light; }
        let r2l = transpose(l2r);
        let up = apply(&r2l, [0.0, 1.0, 0.0]);
        let glow = self.backdrop.as_ref().map(|b| apply(&r2l, b.glow_dir)).unwrap_or([0.0; 3]);
        let behind = apply(&r2l, [0.0, 0.0, -1.0]);
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        // every texel's direction in the lights' frame, and how far above the room's horizon it is
        let mut dirs = Vec::with_capacity(w * h);
        for j in 0..h {
            for i in 0..w {
                let d = crate::env::paint_dir((i as f64 + 0.5) / w as f64, (j as f64 + 0.5) / h as f64);
                dirs.push((d, dot(d, up)));
            }
        }
        // the patch of sky a texel of each row covers
        let patch = |j: usize| (2.0 * PI / w as f64) * ((j as f64 / h as f64 * PI).cos() - ((j + 1) as f64 / h as f64 * PI).cos());
        if let Some(b) = &self.backdrop {
            for (o, (d, y)) in dirs.iter().enumerate() {
                let c = b.at(*y, dot(*d, glow), dot(*d, behind));
                for q in 0..3 {
                    px[o * 4 + q] += c[q] as f32;
                    // (its own share of the light falling on a floor, kept apart)
                    if *y > 0.0 { light.backdrop[q] += c[q] * y * patch(o / w) / PI; }
                }
            }
        }
        let Some(f) = &self.floor else { return FloorLight::default() };
        // the light falling on the floor: everything above the horizon, weighed by how steeply it arrives
        let mut e = [0.0f64; 3];
        for j in 0..h {
            let patch = patch(j);
            for i in 0..w {
                let o = j * w + i;
                let y = dirs[o].1;
                if y <= 0.0 { continue; }
                for q in 0..3 { e[q] += px[o * 4 + q] as f64 * y * patch; }
            }
        }
        // The floor's own colour, in two parts: what the light falling on it makes it (the same
        // everywhere), and what it takes of the backdrop's sweep (by the direction it is seen in).
        let matched = if self.backdrop.is_some() { f.matched } else { 0.0 };
        light.base = [0, 1, 2].map(|q| f.albedo[q] * (e[q] / PI + extra[q]) * (1.0 - matched));
        light.tint = f.albedo.map(|a| a * matched);
        let own = |y: f64| -> [f64; 3] {
            let sweep = self.backdrop.as_ref().map_or([0.0; 3], |b| b.sweep(y));
            [0, 1, 2].map(|q| light.base[q] + light.tint[q] * sweep[q])
        };
        // what a glossy floor mirrors is the map above the horizon, read at the mirrored direction
        let above = if f.gloss > 0.0 { px.to_vec() } else { Vec::new() };
        let read = |d: [f64; 3]| -> [f64; 3] {
            let u = (d[2].atan2(d[0]) / (2.0 * PI) + 0.5) * w as f64 - 0.5;
            let v = (d[1].clamp(-1.0, 1.0).acos() / PI * h as f64 - 0.5).clamp(0.0, h as f64 - 1.0);
            let (i0, j0) = (u.floor(), v.floor());
            let (fx, fy) = (u - i0, v - j0);
            let at = |i: f64, j: f64, q: usize| above[((j as usize).min(h - 1) * w + (i.rem_euclid(w as f64) as usize)) * 4 + q] as f64;
            [0, 1, 2].map(|q| (at(i0, j0, q) * (1.0 - fx) + at(i0 + 1.0, j0, q) * fx) * (1.0 - fy) + (at(i0, j0 + 1.0, q) * (1.0 - fx) + at(i0 + 1.0, j0 + 1.0, q) * fx) * fy)
        };
        for (o, (d, y)) in dirs.iter().enumerate() {
            if *y >= 0.0 { continue; }
            // where a ray going this way from the middle of the shape meets the floor, and what lies
            // beyond the floor there (the backdrop, or nothing)
            let r = EYE_HEIGHT * (1.0 - y * y).max(0.0).sqrt() / -y;
            let m = f.melt(r);
            let beyond = self.backdrop.as_ref().map_or([0.0; 3], |b| b.at(*y, dot(*d, glow), dot(*d, behind)));
            let mut c = own(*y);
            if f.gloss > 0.0 {
                let k = f.mirrors(-y);
                let mirrored = read([d[0] - 2.0 * y * up[0], d[1] - 2.0 * y * up[1], d[2] - 2.0 * y * up[2]]);
                for q in 0..3 { c[q] = c[q] * (1.0 - k) + mirrored[q] * k; }
            }
            for q in 0..3 { px[o * 4 + q] = (c[q] + (beyond[q] - c[q]) * m) as f32; }
        }
        light.own = own(-1.0);
        light
    }
}

// ---- tone curve and grade ----------------------------------------------------------------------------

/// The tone curves by their numbers (the setting `tone`): what each is called and what it is for.
pub const TONES: [(&str, &str); 4] = [
    ("filmic", "the ACES curve: deep contrast, highlights bleach; made for glass on black"),
    ("neutral", "colours exactly as they are up to 80% of white, then a soft roll-off that keeps hue: for bright pastel pictures"),
    ("soft", "as they are up to 60% of white, then each colour rolls off by itself: highlights melt towards white"),
    ("none", "no curve: whatever is brighter than white is cut off"),
];

fn aces(x: f64) -> f64 {
    ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0)
}

/// A colour passes unchanged while its brightest part is below 0.8 of white; above that the brightest
/// part is pressed smoothly towards 1 and the others with it, so the hue stays, with a little whitening
/// only for what is far over. (The shoulder of the Khronos PBR Neutral tone mapper, Khronos Group 2024,
/// whose reference implementation is published under Apache-2.0; without that curve's darkening of
/// everything by 0.04, which is there to make up for a surface's own sheen and has no place here.)
fn neutral(c: [f64; 3]) -> [f64; 3] {
    const START: f64 = 0.8;
    const DESATURATION: f64 = 0.15;
    let peak = c[0].max(c[1]).max(c[2]);
    if peak < START { return c; }
    let d = 1.0 - START;
    let new_peak = 1.0 - d * d / (peak + d - START);
    let g = 1.0 - 1.0 / (DESATURATION * (peak - new_peak) + 1.0);
    c.map(|v| { let v = v * new_peak / peak; v + (new_peak - v) * g })
}

/// One colour as it is up to `KNEE`, then rolling off towards 1 with no kink (value and slope join).
fn shoulder(x: f64) -> f64 {
    const KNEE: f64 = 0.6;
    if x < KNEE { return x; }
    let d = 1.0 - KNEE;
    1.0 - d * d / (x + d - KNEE)
}

/// The tone curve `tone` (0 to 3, see `TONES`) on a colour that has been exposed: the result is what
/// the screen shows, before it is encoded.
pub fn tone_curve(tone: f64, c: [f64; 3]) -> [f64; 3] {
    let c = c.map(|v| v.max(0.0));
    match (tone + 0.5).floor() as i64 {
        1 => neutral(c).map(|v| v.clamp(0.0, 1.0)),
        2 => c.map(shoulder),
        3 => c.map(|v| v.min(1.0)),
        _ => c.map(aces),
    }
}

/// The corrections made around the tone curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grade {
    pub tone: f64,
    /// what red, green and blue are multiplied by before the curve (white balance: warmer or cooler,
    /// greener or more magenta, keeping the brightness of white)
    pub balance: [f64; 3],
    /// 1 = as it is; more draws light and dark apart about middle grey, before the curve
    pub contrast: f64,
    /// 1 = as it is; 0 = grey; after the curve
    pub saturation: f64,
}

/// Middle grey: what contrast turns about.
pub const GREY: f64 = 0.18;

impl Grade {
    pub fn of(p: &Settings) -> Grade {
        // a full step of temperature takes red up and blue down by about a quarter (and the other way)
        let (t, m) = (p.wb_temp.clamp(-1.0, 1.0), p.wb_tint.clamp(-1.0, 1.0));
        let mut balance = [(0.25 * t).exp(), (-0.25 * m).exp(), (-0.25 * t).exp()];
        if t == 0.0 && m == 0.0 {
            balance = [1.0; 3];
        } else {
            let y = LUM[0] * balance[0] + LUM[1] * balance[1] + LUM[2] * balance[2];
            for b in balance.iter_mut() { *b /= y; }
        }
        Grade { tone: (p.tone + 0.5).floor().clamp(0.0, 3.0), balance, contrast: p.contrast.clamp(0.2, 3.0), saturation: p.saturation.clamp(0.0, 3.0) }
    }

    /// Nothing is changed: the picture is toned exactly as it always was.
    pub fn plain(&self) -> bool {
        self.tone == 0.0 && self.balance == [1.0; 3] && self.contrast == 1.0 && self.saturation == 1.0
    }

    /// An exposed colour to what the screen shows (before encoding): balance, contrast, curve, saturation.
    pub fn apply(&self, c: [f64; 3]) -> [f64; 3] {
        let mut c = c;
        if self.balance != [1.0; 3] { for q in 0..3 { c[q] *= self.balance[q]; } }
        if self.contrast != 1.0 { c = c.map(|v| GREY * (v.max(0.0) / GREY).powf(self.contrast)); }
        let mut c = tone_curve(self.tone, c);
        if self.saturation != 1.0 {
            let y = LUM[0] * c[0] + LUM[1] * c[1] + LUM[2] * c[2];
            c = c.map(|v| (y + (v - y) * self.saturation).clamp(0.0, 1.0));
        }
        c
    }

    /// For the shader: (tone, contrast, saturation, 1 if anything is changed) and (balance, unused).
    pub fn uniforms(&self) -> [[f32; 4]; 2] {
        [
            [self.tone as f32, self.contrast as f32, self.saturation as f32, if self.plain() { 0.0 } else { 1.0 }],
            [self.balance[0] as f32, self.balance[1] as f32, self.balance[2] as f32, 0.0],
        ]
    }
}

// ---- sliders -----------------------------------------------------------------------------------------

/// The studio's sliders as the knob table wants them: name, label, lowest, highest, step, whether it
/// can be animated. (The switches `backdrop`, `backShow`, `floor`, `floorAuto`, `floorMirror` and the
/// choice `tone` are not sliders.)
pub const KNOBS: &[(&str, &str, f64, f64, f64, bool)] = &[
    ("elev", "camera height °", -30.0, 90.0, 0.5, true),
    ("backGain", "backdrop ×", 0.0, 4.0, 0.01, true),
    ("backSpread", "backdrop blend °", 1.0, 90.0, 0.5, true),
    ("backWrap", "backdrop reach °", 0.0, 180.0, 1.0, true),
    ("backHigh", "backdrop height °", 0.0, 90.0, 1.0, true),
    ("backTopR", "top red", 0.0, 1.5, 0.005, true),
    ("backTopG", "top green", 0.0, 1.5, 0.005, true),
    ("backTopB", "top blue", 0.0, 1.5, 0.005, true),
    ("backMidR", "horizon red", 0.0, 1.5, 0.005, true),
    ("backMidG", "horizon green", 0.0, 1.5, 0.005, true),
    ("backMidB", "horizon blue", 0.0, 1.5, 0.005, true),
    ("backLowR", "bottom red", 0.0, 1.5, 0.005, true),
    ("backLowG", "bottom green", 0.0, 1.5, 0.005, true),
    ("backLowB", "bottom blue", 0.0, 1.5, 0.005, true),
    ("backGlow", "glow", 0.0, 4.0, 0.01, true),
    ("backGlowR", "glow red", 0.0, 1.5, 0.005, true),
    ("backGlowG", "glow green", 0.0, 1.5, 0.005, true),
    ("backGlowB", "glow blue", 0.0, 1.5, 0.005, true),
    ("backGlowAz", "glow round °", -180.0, 180.0, 0.5, true),
    ("backGlowEl", "glow up °", -90.0, 90.0, 0.5, true),
    ("backGlowSize", "glow width °", 2.0, 120.0, 0.5, true),
    ("floorY", "floor height", -2.0, 2.0, 0.005, true),
    ("floorGain", "floor ×", 0.0, 2.0, 0.01, true),
    ("floorMatch", "floor as backdrop", 0.0, 1.0, 0.01, true),
    ("floorR", "floor red", 0.0, 1.0, 0.005, true),
    ("floorG", "floor green", 0.0, 1.0, 0.005, true),
    ("floorB", "floor blue", 0.0, 1.0, 0.005, true),
    ("floorFade", "floor reach", 2.0, 60.0, 0.1, true),
    ("floorGloss", "floor gloss", 0.0, 1.0, 0.01, true),
    ("wbTemp", "warmth", -1.0, 1.0, 0.01, true),
    ("wbTint", "tint", -1.0, 1.0, 0.01, true),
    ("contrast", "contrast", 0.5, 2.0, 0.01, true),
    ("saturation", "saturation", 0.0, 2.0, 0.01, true),
];

/// Every setting that belongs to the studio (for "back to no studio", and for telling what a preset may set).
pub const NAMES: &[&str] = &[
    "backdrop", "backShow", "backGain", "backSpread", "backWrap", "backHigh", "backTopR", "backTopG", "backTopB", "backMidR", "backMidG", "backMidB",
    "backLowR", "backLowG", "backLowB", "backGlow", "backGlowR", "backGlowG", "backGlowB", "backGlowAz", "backGlowEl", "backGlowSize",
    "floor", "floorAuto", "floorY", "floorR", "floorG", "floorB", "floorGain", "floorMatch", "floorFade", "floorGloss", "floorMirror",
    "elev", "tone", "wbTemp", "wbTint", "contrast", "saturation",
];

// ---- studios: lighting presets for a bright background -------------------------------------------------

/// A studio set up in one go: lights for glass on a light background, a backdrop, a floor and how
/// the picture is toned. Applying one writes the lights into the rig of the family that is up (as if
/// set by hand: `rig=default` brings the family's own back) and the rest into the settings.
pub struct Studio {
    pub id: &'static str,
    pub name: &'static str,
    pub about: &'static str,
    /// az, el, w, h (degrees; half-sizes), r, g, b, soft
    pub lights: &'static [[f64; 8]],
    pub ambient: [f64; 3],
    pub settings: &'static [(&'static str, f64)],
}

impl Studio {
    pub fn lights(&self) -> Vec<Light> {
        self.lights.iter().map(|l| Light { az: l[0], el: l[1], w: l[2], h: l[3], rgb: [l[4], l[5], l[6]], soft: l[7] }).collect()
    }
    /// Write everything but the lights into the settings: the studio's settings start from their
    /// defaults, so one studio after another leaves nothing of the first behind.
    pub fn apply(&self, p: &mut Settings) {
        let fresh = Settings::default();
        for n in NAMES {
            // (where the camera stands is the picture's framing, not the studio's: it stays)
            if *n == "elev" { continue; }
            if let Some(v) = fresh.get(n) { p.set(n, v); }
        }
        for (n, v) in self.settings { p.set(n, *v); }
    }
}

/// The studios. The lights are large and soft and stand where a photographer puts them for glass on a
/// bright ground: the backdrop itself is the main light (glass shows its shape by what it bends of the
/// ground behind it, and goes dark at its edges where it bends in something else), a wide box overhead
/// gives the top faces their sheen and the floor its light, and long strips at the sides draw bright
/// lines down the edges.
pub const STUDIOS: &[Studio] = &[
    Studio {
        id: "lavender",
        name: "Lavender sweep",
        about: "a pale lavender to ice-blue sweep behind the shape, the studio dark behind the camera; an overhead box and two side strips",
        lights: &[
            [0.0, 78.0, 62.0, 40.0, 1.55, 1.55, 1.62, 0.7],
            [-96.0, 14.0, 12.0, 52.0, 1.25, 1.2, 1.4, 0.6],
            [102.0, 20.0, 14.0, 46.0, 1.05, 1.15, 1.35, 0.6],
            [180.0, 30.0, 40.0, 26.0, 0.42, 0.42, 0.46, 0.8],
        ],
        ambient: [0.05, 0.05, 0.06],
        settings: &[
            ("backdrop", 1.0), ("backGain", 1.0), ("backSpread", 38.0), ("backWrap", 105.0), ("backHigh", 50.0),
            ("backTopR", 0.60), ("backTopG", 0.60), ("backTopB", 0.93),
            ("backMidR", 0.80), ("backMidG", 0.87), ("backMidB", 0.98),
            ("backLowR", 0.70), ("backLowG", 0.71), ("backLowB", 0.95),
            ("floor", 1.0), ("floorFade", 16.0),
            ("tone", 1.0), ("exposure", 1.0), ("bloom", 0.08), ("vignette", 0.15), ("saturation", 1.05),
        ],
    },
    Studio {
        id: "ice",
        name: "Ice blue",
        about: "a cool blue-white sweep lit mostly from behind and above: clear glass with dark edges",
        lights: &[
            [0.0, 80.0, 58.0, 44.0, 1.7, 1.75, 1.8, 0.7],
            [-110.0, 10.0, 10.0, 56.0, 0.9, 1.05, 1.3, 0.5],
            [110.0, 10.0, 10.0, 56.0, 0.9, 1.05, 1.3, 0.5],
        ],
        ambient: [0.04, 0.05, 0.06],
        settings: &[
            ("backdrop", 1.0), ("backGain", 1.0), ("backSpread", 45.0), ("backWrap", 90.0), ("backHigh", 40.0),
            ("backTopR", 0.62), ("backTopG", 0.76), ("backTopB", 0.97),
            ("backMidR", 0.88), ("backMidG", 0.94), ("backMidB", 1.0),
            ("backLowR", 0.72), ("backLowG", 0.82), ("backLowB", 0.97),
            ("floor", 1.0), ("floorFade", 16.0),
            ("tone", 1.0), ("exposure", 1.0), ("bloom", 0.08), ("vignette", 0.12), ("saturation", 1.0),
        ],
    },
    Studio {
        id: "white",
        name: "White box",
        about: "a white room lit from everywhere: the softest light there is, for frosted and milky bodies",
        lights: &[
            [0.0, 85.0, 70.0, 70.0, 1.3, 1.3, 1.3, 0.8],
            [-90.0, 20.0, 30.0, 50.0, 0.8, 0.8, 0.82, 0.8],
            [90.0, 20.0, 30.0, 50.0, 0.8, 0.8, 0.82, 0.8],
        ],
        ambient: [0.1, 0.1, 0.1],
        settings: &[
            ("backdrop", 1.0), ("backGain", 1.0), ("backSpread", 60.0),
            ("backTopR", 0.86), ("backTopG", 0.86), ("backTopB", 0.88),
            ("backMidR", 0.95), ("backMidG", 0.95), ("backMidB", 0.96),
            ("backLowR", 0.88), ("backLowG", 0.88), ("backLowB", 0.9),
            ("floor", 1.0), ("floorFade", 18.0),
            ("tone", 1.0), ("exposure", 1.0), ("bloom", 0.06), ("vignette", 0.1), ("saturation", 1.0),
        ],
    },
    Studio {
        id: "dusk",
        name: "Dusk gloss",
        about: "a dark violet sweep and a glossy floor that mirrors the shape: glass on dark, with a ground under it",
        lights: &[
            [0.0, 70.0, 50.0, 30.0, 2.2, 2.1, 2.4, 0.6],
            [-100.0, 18.0, 9.0, 50.0, 2.6, 2.2, 3.0, 0.5],
            [95.0, 25.0, 9.0, 44.0, 1.6, 2.0, 3.0, 0.5],
            [0.0, 12.0, 30.0, 14.0, 0.9, 0.7, 1.3, 0.8],
        ],
        ambient: [0.01, 0.01, 0.02],
        settings: &[
            ("backdrop", 1.0), ("backGain", 1.0), ("backSpread", 30.0),
            ("backTopR", 0.02), ("backTopG", 0.02), ("backTopB", 0.06),
            ("backMidR", 0.10), ("backMidG", 0.07), ("backMidB", 0.20),
            ("backLowR", 0.03), ("backLowG", 0.025), ("backLowB", 0.07),
            ("floor", 1.0), ("floorFade", 12.0), ("floorGloss", 0.6), ("floorMirror", 1.0),
            ("tone", 0.0), ("exposure", 1.0), ("bloom", 0.25), ("vignette", 0.5), ("saturation", 1.0),
        ],
    },
];

/// The place of a studio in `STUDIOS` by its id.
pub fn studio_index(id: &str) -> Option<usize> {
    STUDIOS.iter().position(|s| s.id == id.trim())
}

impl crate::state::State {
    /// How the camera sees the room (see `room_m`).
    pub fn room_m(&self) -> Mat3 {
        room_m(self.p.roll, self.p.elev)
    }
    /// The lights' own frame -> the room: how the lights have been turned in it.
    pub fn lights_to_room(&self) -> Mat3 {
        mul(&transpose(&self.room_m()), &self.light_m())
    }
    /// Set up a studio (a place in `STUDIOS`): its lights become the lights of the family that is up,
    /// the rest goes into the settings. False if there is no such studio.
    pub fn apply_studio(&mut self, i: usize) -> bool {
        let Some(s) = STUDIOS.get(i) else { return false };
        let rig = self.rig_mut();
        rig.lights = s.lights();
        rig.ambient = s.ambient;
        s.apply(&mut self.p);
        true
    }
    /// Take the studio away again: the family's own lights, no backdrop, no floor, the picture toned as ever.
    pub fn clear_studio(&mut self) {
        self.reset_rig();
        let (fresh, elev) = (Settings::default(), self.p.elev);
        for n in NAMES { if let Some(v) = fresh.get(n) { self.p.set(n, v); } }
        self.p.elev = elev;
    }
    /// A studio given as text: its id (`lavender`), its place in the list, or `none`.
    pub fn studio_arg(&mut self, v: &str) -> bool {
        if matches!(v.trim(), "none" | "off" | "") {
            self.clear_studio();
            return true;
        }
        match studio_index(v).or_else(|| v.trim().parse::<usize>().ok()) {
            Some(i) => self.apply_studio(i),
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn studio(id: &str) -> Settings {
        let mut p = Settings::default();
        STUDIOS[studio_index(id).unwrap()].apply(&mut p);
        p
    }

    #[test]
    fn nothing_by_default() {
        let p = Settings::default();
        assert!(Stage::of(&p).off() && !Stage::of(&p).seen());
        assert!(Grade::of(&p).plain());
        let mut px = vec![0.25f32; 16 * 8 * 4];
        assert_eq!(Stage::of(&p).paint(&mut px, 16, 8, &crate::math::I3, [0.0; 3]), FloorLight::default());
        assert!(px.iter().all(|v| *v == 0.25));
    }

    #[test]
    fn the_backdrop_is_smooth_and_goes_where_it_is_told() {
        let b = Backdrop::of(&studio("white")).unwrap();
        let at = |el: f64| b.rgb([0.0, (el * DEG).sin(), -(el * DEG).cos()]);
        assert_eq!(at(0.0), b.mid);
        for (el, want) in [(60.0, b.top), (-60.0, b.low)] {
            let c = at(el);
            for q in 0..3 { assert!((c[q] - want[q]).abs() < 1e-12); }
        }
        // no step anywhere: a tenth of a degree never changes a colour by more than a hundredth of the range
        let mut last = at(-90.0);
        for i in 1..=1800 {
            let c = at(-90.0 + i as f64 * 0.1);
            for q in 0..3 { assert!((c[q] - last[q]).abs() < 0.004, "{i}"); }
            last = c;
        }
    }

    #[test]
    fn a_glow_is_brightest_where_it_points() {
        let mut p = studio("white");
        for (n, v) in [("backGlow", 1.0), ("backGlowAz", 40.0), ("backGlowEl", 10.0), ("backGlowSize", 20.0)] { p.set(n, v); }
        let b = Backdrop::of(&p).unwrap();
        let plain = Backdrop { glow: [0.0; 3], ..b.clone() };
        // a backdrop that does not go all the way round is whole behind the shape and gone behind the camera
        let half = Backdrop { wrap: 90.0, ..plain.clone() };
        assert_eq!(half.rgb([0.0, 0.0, -1.0]), plain.rgb([0.0, 0.0, -1.0]));
        let side = [(80.0 * DEG).sin(), 0.0, -(80.0 * DEG).cos()];
        assert_eq!(half.rgb(side), plain.rgb(side));
        assert_eq!(half.rgb([0.0, 0.0, 1.0]), [0.0; 3]);
        // and one that does not reach the ceiling is whole at the horizon and gone overhead
        let low = Backdrop { high: 40.0, ..plain.clone() };
        assert_eq!(low.rgb([0.0, (30.0 * DEG).sin(), -(30.0 * DEG).cos()]), plain.rgb([0.0, (30.0 * DEG).sin(), -(30.0 * DEG).cos()]));
        assert_eq!(low.rgb([0.0, -1.0, 0.0]), plain.rgb([0.0, -1.0, 0.0]));
        assert_eq!(low.rgb([0.0, 1.0, 0.0]), [0.0; 3]);
        let edge = half.rgb([(105.0 * DEG).sin(), 0.0, -(105.0 * DEG).cos()]);
        assert!((edge[0] / plain.mid[0] - 0.5).abs() < 0.05, "half way through its edge it is about half there");
        let lum = |c: [f64; 3]| c[0] + c[1] + c[2];
        let mid = b.glow_dir;
        assert!((lum(b.rgb(mid)) - lum(plain.rgb(mid)) - 3.0).abs() < 1e-9);
        let off = [-mid[0], mid[1], mid[2]];
        assert!(lum(b.rgb(off)) - lum(plain.rgb(off)) < 0.05);
    }

    #[test]
    fn a_floor_under_an_even_sky_is_as_bright_as_the_sky() {
        // an even backdrop and nothing else: a white matte floor gives back exactly what arrives
        let mut p = Settings::default();
        for (n, v) in [("backdrop", 1.0), ("floor", 1.0), ("floorR", 1.0), ("floorG", 1.0), ("floorB", 1.0), ("floorMatch", 0.0)] { p.set(n, v); }
        for n in ["backTopR", "backTopG", "backTopB", "backMidR", "backMidG", "backMidB", "backLowR", "backLowG", "backLowB"] { p.set(n, 0.5); }
        let (w, h) = (256, 128);
        let mut px = vec![0.0f32; w * h * 4];
        let light = Stage::of(&p).paint(&mut px, w, h, &crate::math::I3, [0.0; 3]);
        let own = light.own;
        for q in 0..3 { assert!((own[q] - 0.5).abs() < 0.002, "{own:?}"); }
        // all of that light is the backdrop's
        for q in 0..3 { assert!((light.backdrop[q] - 0.5).abs() < 0.002, "{light:?}"); }
        // and the map is that colour all round, floor and sky alike
        assert!(px.chunks(4).all(|c| (c[0] - 0.5).abs() < 0.002), "the floor differs from the sky");
    }

    #[test]
    fn the_floor_hides_the_lights_below_it_and_melts_into_the_backdrop() {
        let p = studio("white");
        let s = Stage::of(&p);
        let (w, h) = (256, 128);
        let mut px = vec![0.0f32; w * h * 4];
        // a bright light straight down, which the floor must hide
        for j in 100..h { for i in 0..w { px[(j * w + i) * 4] = 50.0; } }
        let own = s.paint(&mut px, w, h, &crate::math::I3, [0.0; 3]).own;
        let row = |j: usize| [0, 1, 2].map(|q| px[(j * w + 3) * 4 + q] as f64);
        // straight down the glass sees the floor's own colour, which is the backdrop's underfoot
        for q in 0..3 { assert!((row(h - 1)[q] - own[q]).abs() < 1e-3, "{:?} {own:?}", row(h - 1)); }
        for q in 0..3 { assert!((own[q] - s.backdrop.as_ref().unwrap().low[q]).abs() < 1e-12); }
        // and there is no step in what it sees from the horizon down: floor and backdrop are one sweep
        for j in h / 2..h - 1 {
            for q in 0..3 { assert!((row(j)[q] - row(j + 1)[q]).abs() < 0.01, "row {j}"); }
        }
        // just under the horizon it sees the backdrop's horizon colour
        let b = s.backdrop.as_ref().unwrap();
        for q in 0..3 { assert!((row(h / 2)[q] - b.mid[q]).abs() < 0.01, "{:?}", row(h / 2)); }
        // turned lights: the floor stays level in the room, so the map's floor turns with the room
        let mut tilted = vec![0.0f32; w * h * 4];
        let l2r = axis_angle(1.0, 0.0, 0.0, 90.0 * DEG);
        assert_ne!(s.paint_key(&l2r), s.paint_key(&crate::math::I3));
        assert_eq!(s.paint_key(&axis_angle(0.0, 1.0, 0.0, 1.0)), s.paint_key(&crate::math::I3));
        s.paint(&mut tilted, w, h, &l2r, [0.0; 3]);
        assert!(tilted != px);
    }

    #[test]
    fn tone_curves() {
        // the default is the curve the program has always had
        let g = Grade::of(&Settings::default());
        for x in [0.0, 0.05, 0.18, 0.5, 1.0, 4.0] {
            let want = ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14) as f64).clamp(0.0, 1.0);
            assert_eq!(g.apply([x; 3]), [want; 3]);
        }
        // neutral and soft leave a pastel alone; the filmic curve does not
        let pastel = [0.62, 0.58, 0.70];
        assert!(tone_curve(1.0, pastel) == pastel);
        assert!(tone_curve(2.0, [0.5, 0.4, 0.55]) == [0.5, 0.4, 0.55]);
        assert!((tone_curve(0.0, pastel)[0] - pastel[0]).abs() > 0.02);
        // every curve rises, never passes white, and has no step
        for tone in 0..4 {
            let mut last = tone_curve(tone as f64, [0.0; 3])[0];
            for i in 1..4000 {
                let v = tone_curve(tone as f64, [i as f64 * 0.002; 3])[0];
                assert!(v >= last - 1e-12 && v <= 1.0 && v - last < 0.006, "tone {tone} at {i}");
                last = v;
            }
        }
        // neutral keeps the hue of a colour far too bright for the screen
        let c = tone_curve(1.0, [3.0, 1.5, 0.75]);
        assert!(c[0] > c[1] && c[1] > c[2] && c[0] <= 1.0);
    }

    #[test]
    fn grade() {
        let mut p = Settings::default();
        p.set("wbTemp", 0.5);
        let g = Grade::of(&p);
        assert!(g.balance[0] > 1.0 && g.balance[2] < 1.0 && !g.plain());
        // white keeps its brightness
        assert!((LUM[0] * g.balance[0] + LUM[1] * g.balance[1] + LUM[2] * g.balance[2] - 1.0).abs() < 1e-12);
        let mut p = Settings::default();
        p.set("tone", 3.0);
        p.set("saturation", 0.0);
        let c = Grade::of(&p).apply([0.9, 0.2, 0.1]);
        assert!((c[0] - c[1]).abs() < 1e-12 && (c[1] - c[2]).abs() < 1e-12);
        let mut p = Settings::default();
        p.set("tone", 3.0);
        p.set("contrast", 1.5);
        let g = Grade::of(&p);
        assert!((g.apply([GREY; 3])[0] - GREY).abs() < 1e-12);
        assert!(g.apply([0.5; 3])[0] > 0.5 && g.apply([0.05; 3])[0] < 0.05);
    }

    #[test]
    fn studios_are_sound() {
        let all = Settings::default();
        for s in STUDIOS {
            assert!(!s.lights.is_empty(), "{}", s.id);
            for (n, _) in s.settings { assert!(all.get(n).is_some(), "{}: no setting called {n}", s.id); }
            let mut p = Settings::default();
            p.set("elev", 33.0);
            s.apply(&mut p);
            assert!(Stage::of(&p).seen() && p.elev == 33.0, "{}", s.id);
        }
        // one studio after another leaves nothing of the first
        let mut p = studio("dusk");
        STUDIOS[studio_index("white").unwrap()].apply(&mut p);
        assert_eq!(p, studio("white"));
        for (n, ..) in KNOBS { assert!(all.get(n).is_some() && NAMES.contains(n), "{n}"); }
        for n in NAMES { assert!(all.get(n).is_some(), "{n}"); }
    }

    #[test]
    fn the_room() {
        // raised by a quarter turn the camera looks straight down: the room's up points at the camera
        let m = room_m(0.0, 90.0);
        let up = apply(&m, [0.0, 1.0, 0.0]);
        assert!(up[0].abs() < 1e-12 && up[1].abs() < 1e-12 && (up[2] - 1.0).abs() < 1e-12);
        assert_eq!(room_m(0.0, 0.0), crate::math::I3);
    }
}
