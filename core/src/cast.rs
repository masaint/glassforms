//! Shadows and caustics that one body casts on the others (and on itself): the part that needs no GPU.
//!
//! Light in this program comes from far away: a family's soft rectangular lights, and optionally an
//! environment picture. A body that gathers light diffusely (clay, a milky body) is lit by all of it at
//! once, held as nine smooth terms (`env::sh_terms`). To let another body stand in that light's way,
//! the lighting is taken apart again here into its lights, each with
//!   - the direction it comes from and the angle it is taken to cover (which sets how soft its shadow is),
//!   - how much of the nine-term light is its own (so that a shadow removes exactly that share and never
//!     more than was there),
//!   - how much light it sends through a unit of area facing it (what a caustic is made of).
//! The light from everywhere and the body of an environment picture are not taken apart: they cast no
//! shadow of their own (the matte materials already darken where a surface is hemmed in).
//!
//! Each body has two switches (does it cast a shadow, does it cast caustics), which by default follow
//! from what it is made of, and one setting rules them all (`cast`: nothing casts / each body as it is
//! set / everything casts).

use crate::env::{self, Glow, Sh9};
use crate::material;
use crate::presets::Light;
use crate::settings::Settings;

/// Lights whose shadows are followed (the strongest ones; the rest light without casting).
pub const MAX_LIGHTS: usize = 12;
/// Lights whose caustics are followed.
pub const MAX_CAUSTIC_LIGHTS: usize = 16;
/// Bodies a scene can hold.
pub const MAX_BODIES: usize = 4;
/// A light is never taken to be smaller than this (radians): below it the estimate of a shadow's soft
/// edge stops being smooth.
pub const MIN_ANGLE: f64 = 0.008;

/// The global control.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// nothing casts: the program draws exactly what it draws without any of this
    Off,
    /// each body as its own switches say
    PerBody,
    /// every body casts a shadow and, if light passes through or bounces off it, caustics
    All,
}
impl Mode {
    pub fn of(p: &Settings) -> Mode {
        match p.cast.round() as i64 {
            1 => Mode::PerBody,
            2 => Mode::All,
            _ => Mode::Off,
        }
    }
}

/// One of a body's two switches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Switch {
    Off,
    On,
    /// as its material would have it
    Auto,
}
impl Switch {
    pub fn of(v: f64) -> Switch {
        match v.round() as i64 {
            0 => Switch::Off,
            1 => Switch::On,
            _ => Switch::Auto,
        }
    }
    pub fn number(self) -> f64 {
        match self {
            Switch::Off => 0.0,
            Switch::On => 1.0,
            Switch::Auto => 2.0,
        }
    }
}

/// What a body does with light that falls on it, as far as casting goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// light passes through, bent at each surface: it focuses light (caustics) and its shadow is where
    /// the light was bent away from
    Clear,
    /// light bounces off: it blocks, and throws reflected light about
    Metal,
    /// light scatters inside: a soft shadow, coloured where the body is thin
    Milky,
    /// light stops at the surface
    Matte,
}

/// A body's material as the casting needs it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Optics {
    pub kind: Kind,
    /// index of refraction at 550 nm, and n(400 nm) - n(700 nm)
    pub ior: f64,
    pub disp: f64,
    /// what a unit length of the body takes from light passing straight through it, per colour
    pub absorb: [f64; 3],
    /// a metal's colour (what it reflects, per colour)
    pub tint: [f64; 3],
    /// roughness of the surface (0 = polished): a rough body's caustics are soft
    pub rough: f64,
    /// Measured optical constants for the reflecting substrate. A perfect mirror has no table.
    pub conductor: Option<&'static material::Metal>,
    /// Effective clear coat index, including the authored coat strength.
    pub coat_ior: f64,
}
impl Optics {
    pub fn clear(ior: f64, disp: f64, absorb: [f64; 3]) -> Optics {
        Optics { kind: Kind::Clear, ior, disp, absorb, tint: [1.0; 3], rough: 0.0, conductor: None, coat_ior: 1.0 }
    }
    pub fn matte() -> Optics {
        Optics { kind: Kind::Matte, ior: 1.5, disp: 0.0, absorb: [0.0; 3], tint: [1.0; 3], rough: 1.0, conductor: None, coat_ior: 1.0 }
    }
    /// Whether a body of this kind casts a shadow and caustics when nobody has said otherwise.
    pub fn defaults(&self) -> (bool, bool) {
        match self.kind {
            Kind::Clear | Kind::Metal => (true, true),
            Kind::Milky | Kind::Matte => (true, false),
        }
    }
    /// Can it make caustics at all? (A body that stops or scatters light has none to throw.)
    pub fn can_focus(&self) -> bool {
        matches!(self.kind, Kind::Clear | Kind::Metal)
    }
}

/// What a body made of the material these settings describe does with light, as casting takes it.
/// `scale`: what the material's distances are multiplied by to fit the shape (1 unless `fit` is on).
///   - drawn as clay: light stops at the surface
///   - a mirror or a metal: light bounces off, tinted by the body's colour
///   - milky: light scatters inside; what gets straight through dies away with depth as the diffuse
///     light inside does (so a thin part casts a paler, coloured shadow)
///   - a thin wall round air: light goes straight through (an index of 1 bends nothing)
///   - otherwise clear: its index, its dispersion, and its tint by depth
pub fn optics_of(p: &Settings, scale: f64) -> Optics {
    if p.clay > 0.5 { return Optics::matte(); }
    let v = material::Variant::of(p);
    let body = [p.body_r.clamp(0.0, 1.0), p.body_g.clamp(0.0, 1.0), p.body_b.clamp(0.0, 1.0)];
    let rough = p.rough.clamp(0.0, 1.0);
    if v.metal {
        return Optics {
            kind: Kind::Metal, ior: 1.0, disp: 0.0, absorb: [0.0; 3], tint: body, rough,
            conductor: material::metal_of(p), coat_ior: 1.0 + (p.coat_ior - 1.0) * p.coat.clamp(0.0, 1.0),
        };
    }
    // A fired or painted body (a ceramic that lets little of its inner light through: stoneware,
    // porcelain, lacquer) stops light at its surface: its shadow is a solid one, not the pale coloured
    // shadow of a milky body. (A coloured shadow under an orange lacquered pot read at once as a sweet
    // made of coloured wax.) Milk glass, which lets a good share through, stays milky.
    if v.sss && v.ceramic && p.ceramic_through < 0.3 {
        return Optics { kind: Kind::Matte, ior: p.ior, disp: 0.0, absorb: [0.0; 3], tint: body, rough: 1.0, conductor: None, coat_ior: 1.0 };
    }
    if v.sss {
        let sc =material::scatter(body, p.sss_depth * crate::material::sss_fit(scale), p.sss);
        return Optics { kind: Kind::Milky, ior: p.ior, disp: 0.0, absorb: sc.kappa, tint: body, rough, conductor: None, coat_ior: 1.0 };
    }
    if v.hollow {
        return Optics { kind: Kind::Clear, ior: 1.0, disp: 0.0, absorb: [0.0; 3], tint: [1.0; 3], rough, conductor: None, coat_ior: 1.0 };
    }
    // the tint deepens with `density`, and with the shape's thickness when the material is fitted to it;
    // colour by wavelength is taken as the three numbers that give a typical piece its colour
    let absorb = if v.spectrum {
        let u = material::uniforms(p, scale);
        [u[10] as f64, u[11] as f64, u[43] as f64]
    } else {
        [p.absorb[0] * p.density / scale, p.absorb[1] * p.density / scale, p.absorb[2] * p.density / scale]
    };
    Optics { kind: Kind::Clear, ior: p.ior, disp: p.disp, absorb, tint: [1.0; 3], rough, conductor: None, coat_ior: 1.0 }
}

/// A body's two switches as the settings have them. Bodies keep their switches under their lowest
/// layer (counted from 0), as they keep their materials.
pub fn switches(p: &Settings, layer: usize) -> (Switch, Switch) {
    let (s, c) = match layer {
        0 => (p.cast_shadow1, p.cast_caustic1),
        1 => (p.cast_shadow2, p.cast_caustic2),
        2 => (p.cast_shadow3, p.cast_caustic3),
        _ => (p.cast_shadow4, p.cast_caustic4),
    };
    (Switch::of(s), Switch::of(c))
}
/// The names of those two settings.
pub fn switch_names(layer: usize) -> (&'static str, &'static str) {
    [("castShadow", "castCaustic"), ("castShadow2", "castCaustic2"), ("castShadow3", "castCaustic3"), ("castShadow4", "castCaustic4")][layer.min(3)]
}

/// What a body does to the light on its way to the others, with its switches and the global control
/// taken into account.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Caster {
    pub shadow: bool,
    pub caustic: bool,
    pub optics: Optics,
    pub medium: crate::pbr::CompiledMedium,
}
impl Caster {
    /// A body that takes no part.
    pub fn none() -> Caster {
        Caster { shadow: false, caustic: false, optics: Optics::matte(), medium: Default::default() }
    }
    pub fn of(mode: Mode, switches: (Switch, Switch), optics: Optics) -> Caster {
        let (ds, dc) = optics.defaults();
        let pick = |s: Switch, d: bool| match (mode, s) {
            (Mode::Off, _) => false,
            (Mode::All, _) => true,
            (Mode::PerBody, Switch::Off) => false,
            (Mode::PerBody, Switch::On) => true,
            (Mode::PerBody, Switch::Auto) => d,
        };
        Caster { shadow: pick(switches.0, ds), caustic: pick(switches.1, dc) && optics.can_focus(), optics, medium: Default::default() }
    }
    pub fn with_medium(mut self, medium: crate::pbr::CompiledMedium) -> Self {
        self.medium = medium;
        self
    }
    /// How a shadow march treats the body: (what a unit length of it takes from the light, per colour;
    /// what its two surfaces let through). A body that does not cast takes nothing.
    ///   - opaque, or casting caustics: everything is taken (what gets through a clear body is given
    ///     back, bent, as its caustics: so light is moved, never made)
    ///   - clear without caustics: the light goes straight on, tinted by the body and less what its
    ///     surfaces reflect (a pale coloured shadow)
    ///   - milky: a coloured shadow that deepens with thickness
    /// How much of the light from all round (the backdrop, the light from everywhere) the body hides from
    /// a floor beneath it. Light from all round is not followed through a clear body as a caustic: it
    /// goes through, less what the two surfaces reflect (and a tint), so clear glass leaves only a faint
    /// pool of shade under it, as it does in life; a body that lets nothing through hides it all; a milky
    /// one most of it.
    pub fn veil(&self) -> f64 {
        if !self.shadow { return 0.0; }
        match self.optics.kind {
            Kind::Matte | Kind::Metal => 1.0,
            Kind::Clear => {
                let r = ((self.optics.ior - 1.0) / (self.optics.ior + 1.0)).powi(2);
                // (what is reflected away, twice over, and what a moderate thickness of tint takes)
                let tint = (self.optics.absorb[0] + self.optics.absorb[1] + self.optics.absorb[2]) / 3.0;
                (1.0 - (1.0 - r) * (1.0 - r) * (-0.5 * tint).exp()).clamp(0.0, 1.0)
            }
            Kind::Milky => 0.7,
        }
    }
    pub fn shadow_terms(&self) -> ([f64; 3], f64) {
        const SOLID: f64 = 1.0e4;
        if !self.shadow { return ([0.0; 3], 1.0); }
        match self.optics.kind {
            Kind::Matte | Kind::Metal => ([SOLID; 3], 0.0),
            Kind::Clear if self.caustic => ([SOLID; 3], 0.0),
            Kind::Clear => {
                let r = ((self.optics.ior - 1.0) / (self.optics.ior + 1.0)).powi(2);
                (self.optics.absorb, (1.0 - r) * (1.0 - r))
            }
            Kind::Milky => (self.optics.absorb, 1.0),
        }
    }
}

/// One light, as casting takes it.
#[derive(Clone, Debug, PartialEq)]
pub struct CastLight {
    /// towards the light, in the lights' own frame (a unit vector)
    pub dir: [f64; 3],
    /// its colour and brightness (radiance)
    pub rgb: [f64; 3],
    /// the solid angle it covers, its soft edge counted by how bright it is there: `rgb` times this is
    /// the light it sends through a unit of area facing it
    pub flux: f64,
    /// What a white matte surface gathers from it at unit radiance, as nine terms in the lights' own
    /// frame (the form `env::sh_terms` gives): the same arithmetic that lights the clay, so that all the
    /// lights' terms and the light from everywhere add up to the whole lighting's.
    pub terms: [f64; 9],
    /// the radius of a round light that would spread shadows as much (radians)
    pub angle: f64,
    /// Its directions added up, each by the patch of sky it covers and how bright the light is there
    /// (at unit radiance): a flat surface that has the whole light above it gathers `n . aim / pi` of it.
    pub aim: [f64; 3],
}
impl CastLight {
    /// How much it matters: the light it sends, as one number.
    pub fn weight(&self) -> f64 {
        self.flux * lum(self.rgb)
    }
    /// What a white matte surface facing `n` (a unit vector in the lights' own frame) gathers from it,
    /// at unit radiance. Never below nothing.
    pub fn gathered(&self, n: [f64; 3]) -> f64 {
        nine(&self.terms, n).max(0.0)
    }
    /// The same worked out directly, not through the nine terms (which smooth it): the light a flat
    /// surface facing `n` gathers by the cosine of each of the light's directions. Exact while the whole
    /// light is on the surface's side; a floor uses it, since a floor faces one way only.
    pub fn facing(&self, n: [f64; 3]) -> f64 {
        ((n[0] * self.aim[0] + n[1] * self.aim[1] + n[2] * self.aim[2]) / std::f64::consts::PI).max(0.0)
    }
}

pub fn lum(c: [f64; 3]) -> f64 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// The sum the shaders make of nine terms for direction n.
pub fn nine(t: &[f64; 9], n: [f64; 3]) -> f64 {
    let [x, y, z] = n;
    t[0] + t[1] * y + t[2] * z + t[3] * x + t[4] * x * y + t[5] * y * z + t[6] * (3.0 * z * z - 1.0) + t[7] * x * z + t[8] * (x * x - y * y)
}

use crate::math::sstep;

fn terms_of(sh: &Sh9) -> [f64; 9] {
    let t = env::sh_terms(sh, env::SH_MATTE);
    std::array::from_fn(|k| t[k][0] as f64)
}

/// One of a rig's lights, taken on its own. The light is the soft rectangle `env::paint` paints; here it
/// is summed over its own rectangle instead of over a painted map, which is finer for a small lamp.
pub fn of_light(l: &Light) -> CastLight {
    use std::f64::consts::PI;
    const D: f64 = PI / 180.0;
    const N: usize = 32;
    let ce = (l.el * D).cos();
    let a = [(l.az * D).sin() * ce, (l.el * D).sin(), -(l.az * D).cos() * ce];
    let u = [a[2], 0.0, -a[0]];
    let ul = (u[0] * u[0] + u[2] * u[2]).sqrt();
    let u = if ul > 1e-6 { [u[0] / ul, 0.0, u[2] / ul] } else { [1.0, 0.0, 0.0] };
    let v = [a[1] * u[2] - a[2] * u[1], a[2] * u[0] - a[0] * u[2], a[0] * u[1] - a[1] * u[0]];
    let (tw, th) = ((l.w * D).tan(), (l.h * D).tan());
    let soft = l.soft.max(1e-3);
    let fade = |x: f64| 1.0 - sstep((x.abs() - (1.0 - soft)) / soft);
    let mut sh = [[0.0f64; 3]; 9];
    let (mut flux, mut spread) = (0.0, 0.0);
    let mut aim = [0.0f64; 3];
    for j in 0..N {
        let y = (j as f64 + 0.5) / N as f64 * 2.0 - 1.0;
        for i in 0..N {
            let x = (i as f64 + 0.5) / N as f64 * 2.0 - 1.0;
            let m = fade(x) * fade(y);
            if m <= 0.0 { continue; }
            // a point of the rectangle, on the plane one unit away along the light's direction
            let q = [a[0] + x * tw * u[0] + y * th * v[0], a[1] + x * tw * u[1] + y * th * v[1], a[2] + x * tw * u[2] + y * th * v[2]];
            let r = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
            let d = [q[0] / r, q[1] / r, q[2] / r];
            // the patch of sky that piece of the plane covers
            let dw = tw * th * (2.0 / N as f64) * (2.0 / N as f64) / (r * r * r);
            let b = env::sh_basis(d);
            for k in 0..9 {
                sh[k][0] += m * dw * b[k];
            }
            flux += m * dw;
            for c in 0..3 { aim[c] += m * dw * d[c]; }
            // how far from the light's middle this piece is, squared (for the spread of its shadows)
            spread += m * dw * (r * r - 1.0);
        }
    }
    // a round light whose points lie, on average, as far from its middle
    let angle = if flux > 0.0 { (2.0 * spread / flux).sqrt().atan() } else { 0.0 };
    CastLight { dir: a, rgb: l.rgb, flux, terms: terms_of(&sh), angle, aim }
}

/// A glow (the light an environment picture's ceiling took off, given back as a soft round light), as
/// bright as `gain` makes it and widened by `extra` radians as the picture is softened.
pub fn of_glow(g: &Glow, gain: f64, extra: f64) -> CastLight {
    let sigma = (g.sigma * g.sigma + extra * extra).sqrt();
    let unit = Glow { dir: g.dir, energy: [1.0; 3], sigma };
    // (a bell of this width spreads shadows as a disc of twice its radius does; its directions add up to
    // a little less than one whole, by the first order's softening)
    let whole = (-sigma * sigma).exp();
    CastLight { dir: g.dir, rgb: [g.energy[0] * gain, g.energy[1] * gain, g.energy[2] * gain], flux: 1.0, terms: terms_of(&env::sh_of_glow(&unit)), angle: 2.0 * sigma, aim: [g.dir[0] * whole, g.dir[1] * whole, g.dir[2] * whole] }
}

/// The lights that cast, strongest first: a rig's lights made `gain` times as bright (none if the rig
/// is not lit), and an environment picture's glows. Lights that send nothing are left out.
pub fn lights(rig: Option<(&[Light], f64)>, glows: Option<(&[Glow], f64, f64)>) -> Vec<CastLight> {
    let mut out = Vec::new();
    if let Some((list, gain)) = rig {
        for l in list {
            let mut c = of_light(l);
            for v in c.rgb.iter_mut() { *v *= gain; }
            out.push(c);
        }
    }
    if let Some((list, gain, extra)) = glows {
        for g in list {
            out.push(of_glow(g, gain, extra));
        }
    }
    out.retain(|c| c.weight() > 1e-9);
    out.sort_by(|a, b| b.weight().total_cmp(&a.weight()));
    out
}

/// Two unit vectors across a direction (square to it and to each other), the same every time for the
/// same direction: the grid of rays a light sends is laid out along them.
pub fn across(dir: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let a = dir;
    let u = [a[2], 0.0, -a[0]];
    let ul = (u[0] * u[0] + u[2] * u[2]).sqrt();
    let u = if ul > 1e-6 { [u[0] / ul, 0.0, u[2] / ul] } else { [1.0, 0.0, 0.0] };
    (u, [a[1] * u[2] - a[2] * u[1], a[2] * u[0] - a[0] * u[2], a[0] * u[1] - a[1] * u[0]])
}

/// The angle a light is taken to cover when shadows and caustics are worked out: its own at `soft` = 1,
/// down to (nearly) a point at 0.
pub fn taken_angle(light: &CastLight, soft: f64) -> f64 {
    (light.angle * soft.clamp(0.0, 1.0)).max(MIN_ANGLE)
}

/// Voxels a side of the volume of arriving light, by the `castDetail` setting.
pub fn light_res(detail: f64) -> u32 {
    match detail.round() as i64 {
        i64::MIN..=0 => 128,
        1 => 192,
        2 => 256,
        _ => 320,
    }
}

// ---- colour by wavelength ---------------------------------------------------------------------------

/// CIE 1931 colour matching functions (the fit of Wyman, Sloan & Shirley, 2013) to linear sRGB: the
/// tracer's `specW`, so that light split here has the colours the tracer gives it.
pub fn spec_w(l: f64) -> [f64; 3] {
    let g = |x: f64, m: f64, s1: f64, s2: f64| {
        let t = (x - m) * if x < m { s1 } else { s2 };
        (-0.5 * t * t).exp()
    };
    let x = 0.362 * g(l, 442.0, 0.0624, 0.0374) + 1.056 * g(l, 599.8, 0.0264, 0.0323) - 0.065 * g(l, 501.1, 0.0490, 0.0382);
    let y = 0.821 * g(l, 568.8, 0.0213, 0.0247) + 0.286 * g(l, 530.9, 0.0613, 0.0322);
    let z = 1.217 * g(l, 437.0, 0.0845, 0.0278) + 0.681 * g(l, 459.0, 0.0385, 0.0725);
    [3.2406 * x - 1.5372 * y - 0.4986 * z, -0.9689 * x + 1.8758 * y + 0.0415 * z, 0.0557 * x - 0.2040 * y + 1.0570 * z]
}

/// The visible spectrum (400 to 700 nm) cut into `n` equal bands: the colour of each, such that all of
/// them together are white (1, 1, 1) exactly. A caustic's light is split into these, so that light
/// which is not split at all keeps its colour to the last digit.
pub fn bands(n: usize) -> Vec<[f64; 3]> {
    const FINE: usize = 64;
    let mut out = vec![[0.0f64; 3]; n];
    let mut sum = [0.0f64; 3];
    for (b, band) in out.iter_mut().enumerate() {
        for i in 0..FINE {
            let lam = 400.0 + 300.0 * (b as f64 + (i as f64 + 0.5) / FINE as f64) / n as f64;
            let w = spec_w(lam);
            for c in 0..3 {
                band[c] += w[c];
                sum[c] += w[c];
            }
        }
    }
    for band in out.iter_mut() {
        for c in 0..3 {
            band[c] /= sum[c];
        }
    }
    out
}

/// The index of refraction at a wavelength (nm): Cauchy's formula as the tracer has it, so that
/// `ior_at(550) = ior` and `ior_at(400) - ior_at(700) = disp`.
pub fn ior_at(ior: f64, disp: f64, lam: f64) -> f64 {
    ior + disp * (1.0e6 / (lam * lam) - 3.3058) / 4.2092
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lamp(az: f64, el: f64, w: f64, h: f64, soft: f64) -> Light {
        Light { az, el, w, h, rgb: [2.0, 1.0, 0.5], soft }
    }

    #[test]
    fn the_global_control_beats_the_switches() {
        let glass = Optics::clear(1.5, 0.05, [0.1; 3]);
        let off = (Switch::Off, Switch::Off);
        let on = (Switch::On, Switch::On);
        assert_eq!(Caster::of(Mode::Off, on, glass), Caster { shadow: false, caustic: false, optics: glass, medium: Default::default() });
        assert_eq!(Caster::of(Mode::All, off, glass), Caster { shadow: true, caustic: true, optics: glass, medium: Default::default() });
        assert_eq!(Caster::of(Mode::PerBody, off, glass), Caster { shadow: false, caustic: false, optics: glass, medium: Default::default() });
        // by the material: glass casts both, a matte body only a shadow, and nothing makes it focus
        let auto = (Switch::Auto, Switch::Auto);
        assert!(Caster::of(Mode::PerBody, auto, glass).caustic);
        let matte = Caster::of(Mode::All, on, Optics::matte());
        assert!(matte.shadow && !matte.caustic);
    }

    #[test]
    fn shadows_take_what_caustics_give_back() {
        let glass = Optics::clear(1.5, 0.05, [0.2, 0.1, 0.0]);
        // casting caustics: the shadow takes everything
        let both = Caster { shadow: true, caustic: true, optics: glass, medium: Default::default() };
        assert!(both.shadow_terms().0[0] > 1e3 && both.shadow_terms().1 == 0.0);
        // not: the light goes straight on, less the two reflections (4% each for index 1.5)
        let plain = Caster { shadow: true, caustic: false, optics: glass, medium: Default::default() };
        assert_eq!(plain.shadow_terms().0, [0.2, 0.1, 0.0]);
        assert!((plain.shadow_terms().1 - 0.96 * 0.96).abs() < 1e-12);
        assert_eq!(Caster::none().shadow_terms(), ([0.0; 3], 1.0));
    }

    #[test]
    fn a_lights_terms_are_its_share_of_the_painted_lighting() {
        // the nine terms of one light worked out here against those of the same light painted into
        // the map the clay's lighting is summed from
        for l in [lamp(38.0, 18.0, 14.0, 32.0, 0.7), lamp(-140.0, 5.0, 8.0, 42.0, 0.7), lamp(0.0, 85.0, 34.0, 34.0, 0.9), lamp(205.0, 62.0, 3.5, 3.5, 0.5)] {
            let mine = of_light(&l);
            let (w, h) = (1024, 512);
            let white = Light { rgb: [1.0; 3], ..l.clone() };
            let painted = terms_of(&env::sh_of_map(&env::paint(&[white], [0.0; 3], w, h), w, h, 4, env::paint_dir));
            for k in 0..9 {
                assert!((mine.terms[k] - painted[k]).abs() < 0.004 * mine.flux.max(0.02), "{l:?}: term {k}: {} against {}", mine.terms[k], painted[k]);
            }
            // a surface facing the light gathers about flux / pi of it (less for a wide light)
            let facing = mine.gathered(mine.dir);
            assert!(facing > 0.75 * mine.flux / std::f64::consts::PI && facing < 1.1 * mine.flux / std::f64::consts::PI, "{facing} of {}", mine.flux);
            // and one facing away nearly nothing
            assert!(mine.gathered([-mine.dir[0], -mine.dir[1], -mine.dir[2]]) < 0.08 * facing);
            // worked out directly: a surface facing the light gathers a little under flux / pi of it (a
            // wide light's edges arrive aslant), one square to it nothing, and it goes by the cosine
            let direct = mine.facing(mine.dir);
            assert!(direct <= mine.flux / std::f64::consts::PI + 1e-12 && direct > 0.8 * mine.flux / std::f64::consts::PI);
            let (u, _) = across(mine.dir);
            assert!(mine.facing(u).abs() < 1e-9 * mine.flux.max(1.0) + 0.02 * direct);
            let tilted = [0.8 * mine.dir[0] + 0.6 * u[0], 0.8 * mine.dir[1] + 0.6 * u[1], 0.8 * mine.dir[2] + 0.6 * u[2]];
            assert!((mine.facing(tilted) - 0.8 * direct).abs() < 0.03 * direct);
        }
    }

    #[test]
    fn lights_are_sorted_and_sized() {
        let rig = [lamp(0.0, 0.0, 3.0, 3.0, 0.5), lamp(90.0, 10.0, 20.0, 30.0, 0.7), Light { rgb: [0.0; 3], ..lamp(10.0, 10.0, 5.0, 5.0, 0.5) }];
        let list = lights(Some((&rig, 2.0)), None);
        assert_eq!(list.len(), 2);
        assert!(list[0].weight() > list[1].weight());
        assert_eq!(list[1].rgb, [4.0, 2.0, 1.0]);
        // the small lamp spreads shadows as a disc of about 2 degrees does, the large one about 15
        let (small, large) = (list[1].angle.to_degrees(), list[0].angle.to_degrees());
        assert!(small > 1.2 && small < 3.0, "{small}");
        assert!(large > 9.0 && large < 30.0, "{large}");
        assert!((taken_angle(&list[0], 0.5) - 0.5 * list[0].angle).abs() < 1e-12);
        assert_eq!(taken_angle(&list[0], 0.0), MIN_ANGLE);
    }

    #[test]
    fn the_bands_make_white() {
        for n in [4, 16] {
            let b = bands(n);
            for c in 0..3 {
                let sum: f64 = b.iter().map(|v| v[c]).sum();
                assert!((sum - 1.0).abs() < 1e-12);
            }
        }
        // red light is in the long wavelengths, blue in the short
        let b = bands(4);
        assert!(b[3][0] > b[0][0] && b[0][2] > b[3][2]);
        assert!((ior_at(1.5, 0.08, 550.0) - 1.5).abs() < 1e-4);
        assert!((ior_at(1.5, 0.08, 400.0) - ior_at(1.5, 0.08, 700.0) - 0.08).abs() < 1e-4);
    }
}
