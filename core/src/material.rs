//! Materials beyond plain glass: what each setting means to the tracer, the optics behind them written
//! out in plain arithmetic (so that they can be tested without a GPU), the measured data they use, and
//! the named materials. `docs/materials.md` explains the models and says where every number comes from.
//!
//! The scene has one material. With every setting here at its default the tracer is built exactly as it
//! was before materials existed; anything else builds a variant of it that holds only what is in use
//! (`Variant`, `tracer_parts`).

use rayon::prelude::*;

use crate::settings::Settings;

#[path = "material_data.rs"]
mod data;
/// How a thin film drains (the numbers behind the tracer's `filmNm`).
#[path = "film.rs"]
pub mod film;
pub use data::{METALS, SPECTRA};

/// The grid every table is on: 400, 410, ... 700 nm.
pub const NM0: f64 = 400.0;
pub const NM_STEP: f64 = 10.0;
pub const NM_N: usize = 31;

/// A metal: its complex index of refraction n + ik across the visible.
pub struct Metal {
    pub id: &'static str,
    pub name: &'static str,
    pub source: &'static str,
    pub n: [f32; NM_N],
    pub k: [f32; NM_N],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpectrumKind {
    /// a published table or data file of measurements
    Measured,
    /// traced from a published figure
    Digitised,
    /// worked out or built here from descriptions: not a measurement
    Approximate,
}

/// How strongly a coloured glass or gem absorbs each wavelength, as a shape (its largest value is 1).
pub struct Spectrum {
    pub id: &'static str,
    pub name: &'static str,
    pub kind: SpectrumKind,
    pub source: &'static str,
    /// the largest absorption of the sample measured, per centimetre (0 where the source gives no scale)
    pub peak_per_cm: f32,
    pub shape: [f32; NM_N],
}

/// A table read at any wavelength, in between its entries by straight lines, held at its ends.
pub fn table_at(t: &[f32; NM_N], nm: f64) -> f64 {
    let x = ((nm - NM0) / NM_STEP).clamp(0.0, (NM_N - 1) as f64);
    let i = (x as usize).min(NM_N - 2);
    let f = x - i as f64;
    t[i] as f64 * (1.0 - f) + t[i + 1] as f64 * f
}

// ---- colour ----------------------------------------------------------------------------------------

/// What one wavelength adds to red, green and blue (linear sRGB): the tracer's own `specW`
/// (CIE 1931 colour matching functions as fitted by Wyman, Sloan & Shirley 2013).
pub fn spec_w(l: f64) -> [f64; 3] {
    let g = |x: f64, m: f64, a: f64, b: f64| { let t = (x - m) * if x < m { a } else { b }; (-0.5 * t * t).exp() };
    let x = 0.362 * g(l, 442.0, 0.0624, 0.0374) + 1.056 * g(l, 599.8, 0.0264, 0.0323) - 0.065 * g(l, 501.1, 0.0490, 0.0382);
    let y = 0.821 * g(l, 568.8, 0.0213, 0.0247) + 0.286 * g(l, 530.9, 0.0613, 0.0322);
    let z = 1.217 * g(l, 437.0, 0.0845, 0.0278) + 0.681 * g(l, 459.0, 0.0385, 0.0725);
    [3.2406 * x - 1.5372 * y - 0.4986 * z, -0.9689 * x + 1.8758 * y + 0.0415 * z, 0.0557 * x - 0.2040 * y + 1.0570 * z]
}

/// The colour of white light after `f(wavelength)` of each wavelength is kept, as linear RGB with white = 1.
pub fn colour_of(f: impl Fn(f64) -> f64) -> [f64; 3] {
    let (mut c, mut w) = ([0.0; 3], [0.0; 3]);
    for i in 0..61 {
        let l = 400.0 + 5.0 * i as f64;
        let (s, k) = (spec_w(l), f(l));
        for j in 0..3 { c[j] += s[j] * k; w[j] += s[j]; }
    }
    [c[0] / w[0], c[1] / w[1], c[2] / w[2]]
}

// ---- reflection at a surface -------------------------------------------------------------------------

/// Unpolarised reflectance going from index n1 into index n2 (both clear), by the cosine of the angle
/// the light arrives at. The tracer's `fresnel`.
pub fn fresnel(cosi: f64, n1: f64, n2: f64) -> f64 {
    let sint = n1 / n2 * (1.0 - cosi * cosi).max(0.0).sqrt();
    if sint >= 1.0 { return 1.0; }
    let cost = (1.0 - sint * sint).max(0.0).sqrt();
    let rs = (n1 * cosi - n2 * cost) / (n1 * cosi + n2 * cost);
    let rp = (n2 * cosi - n1 * cost) / (n2 * cosi + n1 * cost);
    0.5 * (rs * rs + rp * rp)
}

/// Unpolarised reflectance of a metal of index n + ik seen from a clear medium of index n1.
/// The tracer's `fresMetal` (the exact Fresnel equations for an absorbing medium).
pub fn fresnel_metal(cosi: f64, n1: f64, n: f64, k: f64) -> f64 {
    let c = cosi.clamp(0.0, 1.0);
    let (c2, s2) = (c * c, 1.0 - c * c);
    let (e, ek) = (n / n1, k / n1);
    let t0 = e * e - ek * ek - s2;
    let ab = (t0 * t0 + 4.0 * e * e * ek * ek).sqrt();
    let t1 = ab + c2;
    let a = (0.5 * (ab + t0)).max(0.0).sqrt();
    let t2 = 2.0 * a * c;
    let rs = (t1 - t2) / (t1 + t2);
    let t3 = c2 * ab + s2 * s2;
    let t4 = t2 * s2;
    let rp = rs * (t3 - t4) / (t3 + t4);
    0.5 * (rp + rs)
}

type C = (f64, f64);
fn cadd(a: C, b: C) -> C { (a.0 + b.0, a.1 + b.1) }
fn csub(a: C, b: C) -> C { (a.0 - b.0, a.1 - b.1) }
fn cmul(a: C, b: C) -> C { (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0) }
fn cdiv(a: C, b: C) -> C { let d = b.0 * b.0 + b.1 * b.1; ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d) }
fn csqrt(a: C) -> C {
    let m = (a.0 * a.0 + a.1 * a.1).sqrt();
    let re = (0.5 * (m + a.0)).max(0.0).sqrt();
    let im = (0.5 * (m - a.0)).max(0.0).sqrt();
    (re, if a.1 < 0.0 { -im } else { im })
}

/// Unpolarised reflectance of a thin film: light arrives through a clear medium of index n1 at a film of
/// index n2 and thickness `d` nm lying on a body of index n3 (complex: (n, k); k = 0 for a clear body).
/// Every reflection back and forth inside the film is counted (the Airy sum), at wavelength `lam` nm.
/// The tracer's `filmR` (clear body) and `filmMetal` (metal body).
pub fn film_reflectance(cosi: f64, lam: f64, d: f64, n1: f64, n2: f64, n3: C) -> f64 {
    // (never quite edge-on: there the sums below divide nothing by nothing)
    let c1 = cosi.clamp(1e-4, 1.0);
    let s1 = (1.0 - c1 * c1).sqrt();
    // total reflection off a clear body that is thinner than the medium the light is in
    if n3.1 == 0.0 && n1 * s1 >= n3.0 { return 1.0; }
    if n1 * s1 >= n2 {
        // The light cannot travel in the film at this angle, only leak through it (a film of lower index
        // than both sides, beyond its own critical angle): what gets through is what the bare surface
        // would let through, dying away with the film's thickness.
        let bare = if n3.1 == 0.0 { fresnel(c1, n1, n3.0) } else { fresnel_metal(c1, n1, n3.0, n3.1) };
        let leak = (-4.0 * std::f64::consts::PI * d * (n1 * n1 * s1 * s1 - n2 * n2).sqrt() / lam).exp();
        return 1.0 - (1.0 - bare) * leak;
    }
    let s2 = n1 / n2 * s1;
    let c2 = (1.0 - s2 * s2).sqrt();
    // cosine inside the body: complex when the body absorbs
    let s3 = cdiv((n1 * s1, 0.0), n3);
    let c3 = csqrt(csub((1.0, 0.0), cmul(s3, s3)));
    let phi = 4.0 * std::f64::consts::PI * n2 * d * c2 / lam;
    let e = (phi.cos(), phi.sin());
    let one = |r12: f64, r23: C| -> f64 {
        let x = cmul(r23, e);
        let r = cdiv(cadd((r12, 0.0), x), cadd((1.0, 0.0), cmul((r12, 0.0), x)));
        r.0 * r.0 + r.1 * r.1
    };
    let n2c2 = (n2 * c2, 0.0);
    let rs = one((n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2), cdiv(csub(n2c2, cmul(n3, c3)), cadd(n2c2, cmul(n3, c3))));
    let n3c2 = cmul(n3, (c2, 0.0));
    let n2c3 = cmul((n2, 0.0), c3);
    let rp = one((n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2), cdiv(csub(n3c2, n2c3), cadd(n3c2, n2c3)));
    0.5 * (rs + rp)
}

// ---- what is in use ----------------------------------------------------------------------------------

/// Which parts of the material are in use: the tracer is built for exactly these.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Variant {
    /// an opaque reflecting body (a mirror or a metal) instead of a clear one
    pub metal: bool,
    /// a rough surface
    pub rough: bool,
    /// roughness by sampling alone (for checking)
    pub rough_ref: bool,
    pub film: bool,
    pub coat: bool,
    /// light scatters inside the body, estimated
    pub sss: bool,
    /// ... followed as it wanders instead
    pub walk: bool,
    /// absorption by wavelength
    pub spectrum: bool,
    pub glow: bool,
    /// an index that changes with depth
    pub grin: bool,
    /// double refraction
    pub biref: bool,
    /// there are bubbles in the volume (the tracer itself is the same, but must look through holes)
    pub bubbles: bool,
    /// no dispersion: one path stands for every wavelength (only told to tracers of other materials:
    /// plain glass looks for itself, as it always has)
    pub mono: bool,
    /// a thin wall around air instead of a solid body
    pub hollow: bool,
    /// 0: the scene has one material. Otherwise how many bodies it has, at least one of them of a
    /// material of its own: the tracer is then the one that follows a ray from body to body, which
    /// takes each body's material from numbers. Of the rest each part is then set when any of the bodies
    /// has it (`walk`, `rough_ref`, `coat`, `glow`, `mono` and `bubbles` never are: a body among several has
    /// no true scattering, its roughness is drawn the live way, its coat, glow and bubbles need nothing
    /// compiled in); they say which parts are compiled in, and at each surface and along each stretch the
    /// body there chooses among them (`Variant::of_bodies`).
    pub bodies: u8,
    /// what lies inside the body: a tint that changes, cracks, grains, fibres, a glow (inner.rs)
    pub inner: crate::inner::Parts,
    /// the surface's wear, polishing lines or uneven colour (a material other than plain glass)
    pub wear: bool,
    /// lamps stand in the scene: a tracer of the material kind is wanted even for plain glass (lamps.rs)
    pub near: bool,
}

impl Variant {
    pub fn of(p: &Settings) -> Self {
        let metal = p.metal >= 0.5;
        let rough = p.rough > 0.0;
        // a thin wall has no inside for light to scatter, be absorbed or curve in
        let hollow = !metal && p.hollow > 0.5;
        let sss = !metal && !hollow && p.sss > 0.0;
        let walk = sss && p.sss_walk > 0.5;
        // the costly ways of travelling through the body exclude one another: wandering first, then
        // double refraction, then the curved rays
        let biref = !metal && !hollow && !walk && p.biref != 0.0;
        let v = Self {
            metal,
            rough,
            rough_ref: rough && p.rough_ref >= 1.0,
            film: p.film > 0.0,
            coat: !hollow && p.coat > 0.0,
            sss,
            walk,
            spectrum: !metal && !hollow && p.spectrum >= 0.5 && (p.spectrum as usize) <= SPECTRA.len(),
            glow: p.glow > 0.0,
            grin: !metal && !hollow && !walk && !biref && p.grin != 0.0,
            biref,
            bubbles: !hollow && p.bubbles > 0.0,
            mono: false,
            hollow,
            bodies: 0,
            // (met along straight stretches inside a solid body: not with the ways of travelling that wander or curve)
            inner: crate::inner::Parts::of(p, !metal && !hollow && !walk && (biref || p.grin == 0.0)),
            wear: p.brush > 0.0 || p.mottle > 0.0 || (p.age > 0.0 && marks(p).iter().any(|v| *v > 0.0)),
            near: false,
        };
        // (light that wanders is followed once, whatever its colour; and nothing bends in a thin wall)
        Self { mono: !v.plain() && !metal && (p.disp == 0.0 || walk || hollow), ..v }
    }

    /// The tracer of several bodies, for bodies made of these materials (in the order of the bodies):
    /// the parts any of them uses (each body's as `body_variant` has them). At each surface and along each
    /// stretch the body there chooses among the parts compiled in by its own numbers. A body among several
    /// has no true scattering (its light volume is used, as for the estimate), and its roughness is always
    /// drawn the live way.
    pub fn of_bodies(mats: &[&Settings]) -> Self {
        let mut v = Self { bodies: mats.len().clamp(2, MAX_BODIES) as u8, ..Self::default() };
        for p in mats {
            let x = body_variant(p);
            v.metal |= x.metal;
            v.rough |= x.rough;
            v.film |= x.film;
            v.hollow |= x.hollow;
            v.sss |= x.sss;
            v.spectrum |= x.spectrum;
            v.grin |= x.grin;
            v.biref |= x.biref;
            v.wear |= x.wear;
            let (a, b) = (&mut v.inner, x.inner);
            a.grad |= b.grad;
            a.crack |= b.crack;
            a.crack_depth |= b.crack_depth;
            a.crack_rough |= b.crack_rough;
            a.play |= b.play;
            a.glitter |= b.glitter;
            a.silk |= b.silk;
            a.piezo |= b.piezo;
        }
        v
    }

    /// What is left of the material when the shape is drawn as clay: only what lies on top of the clay,
    /// a clear coat or a thin film (and the roughness of that film's reflection).
    pub fn for_clay(&self) -> Self {
        let on_top = self.coat || self.film;
        Self { coat: self.coat, film: self.film, rough: self.rough && self.film && !self.coat && on_top, bubbles: self.bubbles, ..Self::default() }
    }

    /// The tracer needs nothing but its old self (bubbles are in the volume, not in the tracer).
    pub fn plain(&self) -> bool {
        *self == Self { bubbles: self.bubbles, mono: self.mono, ..Self::default() }
    }

    fn flags(&self) -> [(&'static str, bool); 13] {
        [
            ("HOLLOW", self.hollow), ("METAL", self.metal), ("ROUGH", self.rough), ("ROUGH_REF", self.rough_ref), ("FILM", self.film), ("COAT", self.coat),
            ("SSS", self.sss), ("WALK", self.walk), ("SPECTRUM", self.spectrum), ("GLOW", self.glow), ("GRIN", self.grin), ("BIREF", self.biref),
            ("WEAR", self.wear),
        ]
    }

    /// A short name for the variant: what the built tracers are filed under.
    pub fn key(&self) -> String {
        let mut k: String = self.flags().iter().filter(|f| f.1).map(|f| f.0.to_lowercase()).collect::<Vec<_>>().join("+");
        if self.mono { k.push_str("+mono"); }
        if self.bubbles { k.push_str("+bubbles"); }
        k.push_str(&self.inner.key());
        if self.bodies > 0 { k = if k.is_empty() { format!("{} bodies", self.bodies) } else { format!("{} bodies {k}", self.bodies) }; }
        if k.is_empty() { "glass".into() } else { k }
    }

    /// Every combination worth building in a test: each part alone, and the mixtures the named materials use.
    pub fn samples() -> Vec<Variant> {
        let d = Variant::default();
        let mut v = vec![
            d,
            Variant { rough: true, ..d },
            Variant { rough: true, rough_ref: true, ..d },
            Variant { film: true, ..d },
            Variant { coat: true, ..d },
            Variant { sss: true, ..d },
            Variant { sss: true, walk: true, ..d },
            Variant { spectrum: true, ..d },
            Variant { glow: true, ..d },
            Variant { grin: true, ..d },
            Variant { biref: true, ..d },
            Variant { metal: true, ..d },
            Variant { metal: true, rough: true, ..d },
            Variant { metal: true, rough: true, rough_ref: true, ..d },
            Variant { metal: true, film: true, ..d },
            Variant { metal: true, coat: true, ..d },
            Variant { metal: true, glow: true, ..d },
            Variant { metal: true, rough: true, film: true, coat: true, glow: true, ..d },
            Variant { metal: true, wear: true, ..d },
            Variant { metal: true, rough: true, wear: true, ..d },
            Variant { metal: true, coat: true, film: true, wear: true, ..d },
            Variant { sss: true, wear: true, mono: true, ..d },
            Variant { sss: true, rough: true, coat: true, wear: true, mono: true, ..d },
            Variant { rough: true, wear: true, ..d },
            Variant { rough: true, film: true, coat: true, sss: true, spectrum: true, glow: true, ..d },
            Variant { rough: true, film: true, coat: true, sss: true, walk: true, spectrum: true, glow: true, ..d },
            Variant { rough: true, film: true, coat: true, spectrum: true, glow: true, grin: true, ..d },
            Variant { rough: true, film: true, coat: true, sss: true, spectrum: true, glow: true, biref: true, ..d },
            Variant { bubbles: true, ..d },
            Variant { hollow: true, mono: true, ..d },
            Variant { hollow: true, mono: true, film: true, rough: true, glow: true, ..d },
        ];
        // what lies inside the body, in a clear body and with the parts it has to work beside
        for i in crate::inner::Parts::samples() {
            v.push(Variant { inner: i, ..d });
            v.push(Variant { inner: i, mono: true, ..d });
            v.push(Variant { inner: i, sss: true, rough: true, glow: true, mono: true, ..d });
            v.push(Variant { inner: i, film: true, spectrum: true, coat: true, ..d });
            v.push(Variant { inner: i, biref: true, ..d });
        }
        for n in NAMED {
            let mut p = Settings::default();
            n.apply(&mut p);
            let x = Variant::of(&p);
            if !v.contains(&x) { v.push(x); }
        }
        // each again with no dispersion (metals have none to begin with)
        for x in v.clone() {
            let m = Variant { mono: true, ..x };
            if !x.plain() && !x.metal && !x.hollow && !v.contains(&m) { v.push(m); }
        }
        v
    }
}

/// What goes into the tracer's source for a variant.
pub struct TracerParts {
    /// the material's own functions and data (empty for plain glass)
    pub decl: String,
    /// the call that shades a point of the surface
    pub shade: &'static str,
    /// the reflection blended in where a ray grazes the outline
    pub graze: &'static str,
    /// rays leaving the body must be able to pass through at least this many further pieces of it
    pub min_layers: u32,
}

/// The pieces of tracer source for a variant. `material_wgsl` is `trace_material.wgsl`, which holds every
/// part between `//#if NAME` ... `//#else` ... `//#endif` lines; only the parts in use are kept.
///
/// `picture`: the light comes from an environment picture (the softened lights are then read from its
/// softer levels as well). `clay`: the shape is drawn as matte clay, and `v` holds only what lies on top
/// of it (`Variant::for_clay`).
pub fn tracer_parts(v: &Variant, material_wgsl: &str, picture: bool, clay: bool) -> TracerParts {
    let min_layers = if v.bubbles { 2 } else { 0 };
    if v.plain() {
        return if clay {
            TracerParts { decl: String::new(), shade: "clay(ro + rd * t, rd)", graze: "clay(p, rd)", min_layers }
        } else {
            TracerParts { decl: String::new(), shade: "shade(ro, rd, t, xi, W)", graze: "env(reflect(rd, nrm(p)))", min_layers }
        };
    }
    // film colours and absorption by wavelength are weighed at several wavelengths for each path traced;
    // everything else needs just the one
    let fine = v.film || v.spectrum;
    let mut flags: Vec<(&str, bool)> = v.flags().to_vec();
    // what lies inside the body: its own parts, and the softened lights if they read them
    flags.extend(v.inner.flags());
    flags.push(("FINE", fine));
    flags.push(("MONO", v.mono));
    flags.push(("ENVL", v.rough || v.sss || v.wear || v.metal || v.coat || v.inner.soft_lights()));
    flags.push(("PICTURE", picture));
    flags.push(("CLAY", clay));
    // a film around air (a soap bubble) has a tracer of its own, in film.wgsl
    flags.push(("BUBBLE", v.hollow && v.film && !clay));
    flags.push(("RANDOM", v.rough || v.walk));
    // what everything still travelling along a path adds to red, green and blue
    let decl = preprocess(material_wgsl, &flags).replace("{{WEIGHT}}", if fine { "weigh(i, xi, W, t0, t1, t2)" } else { "(wOf(i, xi, W) * thr)" });
    if clay { return TracerParts { decl, shade: "clayM(ro + rd * t, rd)", graze: "clayM(p, rd)", min_layers }; }
    TracerParts { decl, shade: "shadeM(ro, rd, t, xi, W, fc)", graze: "grazeM(p, rd)", min_layers }
}

/// Keep the lines of `src` whose conditions hold. A condition line is `//#if NAME`, `//#if !NAME`,
/// `//#else` or `//#endif` (they nest); a name that is not listed counts as off.
/// A block between `//#each` and `//#end` is written out once for each group of weighed wavelengths:
/// one group, or three when `MONO` is on. In it `t[g]` is that group's own variable (t0, t1, t2) and
/// `GROUP` its number.
pub fn preprocess(src: &str, flags: &[(&str, bool)]) -> String {
    let on = |name: &str| flags.iter().any(|(n, v)| *n == name && *v);
    let groups = if on("MONO") { 3 } else { 1 };
    // the lines of the block being gathered, if one is open
    let mut each: Option<String> = None;
    // for each open `//#if`: is this branch kept, and was the part outside it kept
    let mut stack: Vec<(bool, bool)> = Vec::new();
    let mut out = String::with_capacity(src.len());
    for line in src.lines() {
        let t = line.trim();
        let keep = stack.last().map_or(true, |s| s.0 && s.1);
        if let Some(cond) = t.strip_prefix("//#if ") {
            let cond = cond.trim();
            let v = match cond.strip_prefix('!') { Some(n) => !on(n.trim()), None => on(cond) };
            stack.push((v, keep));
        } else if t == "//#else" {
            if let Some(s) = stack.last_mut() { s.0 = !s.0; }
        } else if t == "//#endif" {
            stack.pop();
        } else if t == "//#each" {
            if keep { each = Some(String::new()); }
        } else if t == "//#end" {
            if let Some(block) = each.take() {
                for g in 0..groups {
                    out.push_str(&block.replace("t[g]", &format!("t{g}")).replace("GROUP", &g.to_string()));
                }
            }
        } else if keep {
            let sink = each.as_mut().unwrap_or(&mut out);
            sink.push_str(line);
            sink.push('\n');
        }
    }
    out
}

// ---- what the tracer is told ---------------------------------------------------------------------------

/// Numbers in the material's uniform block (`MU` in trace_material.wgsl): 13 groups of four, then one
/// group of four for each wavelength of the grid (32 groups: the last is a copy of the 31st), then three
/// groups for a draining film (`film::numbers`).
pub const MU_FLOATS: usize = (13 + 32 + 3) * 4;
/// Where the draining film's numbers go.
pub const MU_FILM: usize = (13 + 32) * 4;
/// Where the environment picture's four numbers go (the engine fills them in: it holds the picture).
pub const MU_PIC: usize = 11 * 4;

/// The environment picture's sun, as a mirror sees it, rides in the unused last number of the first
/// seven rows of the tables (`sun_numbers`; trace_material.wgsl: `sunGlint`).
pub const MU_SUN: [usize; 7] = [13 * 4 + 3, 14 * 4 + 3, 15 * 4 + 3, 16 * 4 + 3, 17 * 4 + 3, 18 * 4 + 3, 19 * 4 + 3];

/// The sun of an environment picture for a mirror: the brightest of the light its ceiling took off
/// (`glows`: direction in the lights' frame, energy, width), all of it, as a small sharp disc. The glass
/// is given that light back as a soft glow, for glass focuses a sharp sun into specks that never settle;
/// a mirror shows it as the one bright glint it is, which is half of what makes gold look like gold
/// outdoors. Direction xyz, energy rgb (times the picture's gain), angular radius; zeros: no sun.
pub fn sun_numbers(glows: &[crate::env::Glow], gain: f64) -> [f32; 7] {
    let lum = |e: &[f64; 3]| 0.2126 * e[0] + 0.7152 * e[1] + 0.0722 * e[2];
    match glows.iter().max_by(|a, b| lum(&a.energy).total_cmp(&lum(&b.energy))) {
        Some(g) if lum(&g.energy) > 0.0 => [g.dir[0] as f32, g.dir[1] as f32, g.dir[2] as f32, (g.energy[0] * gain) as f32, (g.energy[1] * gain) as f32, (g.energy[2] * gain) as f32, g.sigma.max(0.006) as f32],
        _ => [0.0; 7],
    }
}

/// The share of a clear body's roughness that is drawn by sampling (turning each path by a random
/// facet); the rest is drawn by softening the lights the path sees. More of it blurs the shape seen
/// through itself more truly and makes more grain.
pub const ROUGH_SAMPLED: f64 = 0.5;
/// While the view moves, no more than this much spread (of the surface's slopes) is sampled: random
/// facets send neighbouring pixels to different parts of the volume, which is what the tracer's time
/// goes on (measured on the gyroid at 4K: 1.15 times plain glass at 0.01, 1.4 at 0.02, 2.5 at 0.18).
pub const ROUGH_SAMPLED_MOVING: f64 = 0.09;
/// Where in the uniform block the sampled share is, and where its value for a moving view is kept.
pub const MU_SHARE: usize = 3;
pub const MU_SHARE_MOVING: usize = 1;
/// The thickness of a thin wall, for its tint: what one unit of the solid's absorption comes to each
/// time light passes through the wall.
pub const WALL: f64 = 0.05;
/// How strongly a spectrum absorbs at `density` 1: the largest absorption, per unit of length.
pub const SPECTRUM_SCALE: f64 = 3.0;
/// Scattering events followed for one ray at most when light is followed as it wanders.
pub const WALK_EVENTS: f64 = 48.0;

/// What light scattering inside the body comes to, from the three controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scatter {
    /// scatterings per unit length
    pub sigma_s: f64,
    /// absorption per unit length that gives a thick piece the body's colour
    pub sigma_a: [f64; 3],
    /// how fast diffuse light dies away with depth, per unit length
    pub kappa: [f64; 3],
    /// how far outside the surface diffuse light behaves as if it fell to nothing, doubled
    pub reach: f64,
}

/// `colour`: what a thick piece of the body looks like under white light. `depth`: how far light travels
/// between scatterings when `amount` is 1; less of it scatters more rarely.
pub fn scatter(colour: [f64; 3], depth: f64, amount: f64) -> Scatter {
    let sigma_s = amount.clamp(0.0, 1.0) / depth.max(1e-3);
    let mut sigma_a = [0.0; 3];
    let mut kappa = [0.0; 3];
    for i in 0..3 {
        // The share of light that survives each scattering, such that endless scattering gives back
        // `colour` in all (van de Hulst's relation, inverted as in Kulla & Conty 2017).
        let c = colour[i].clamp(0.0, 0.999);
        let s = 4.09712 + 4.20863 * c - (9.59217 + 41.6808 * c + 17.7126 * c * c).sqrt();
        let albedo = (1.0 - s * s).clamp(1e-3, 1.0);
        sigma_a[i] = sigma_s * (1.0 - albedo) / albedo;
        kappa[i] = (3.0 * sigma_a[i] * (sigma_a[i] + sigma_s)).sqrt();
    }
    Scatter { sigma_s, sigma_a, kappa, reach: SCATTER_REACH / sigma_s.max(1e-6) }
}
/// What a milky body's distance between scatterings is multiplied by to fit the shape: the fitting
/// scale, but never less than 1. A thin piece is not made denser to look like a thick one: light that
/// gets through a thin wall or a thin tube, and glows there, is what shows a body to be translucent.
pub fn sss_fit(scale: f64) -> f64 {
    scale.max(1.0)
}

/// The scattering of each colour (red, green, blue: per unit length) and the absorption that, with it,
/// gives a thick piece the body's colour, when the colours scatter differently (`sssSpread`): colour c
/// travels (its wavelength / 550 nm)^(4 x spread) times as far between scatterings as green does, as
/// light scattered by particles small against the wavelength does at spread 1.
///
/// `forward` (0 to 0.95): how much the body scatters forward (the mean cosine of its scattering, g), as
/// wax does. By the similarity rule a body that scatters forward spreads light as one that scatters
/// evenly with (1 - g) times as many scatterings; its absorption stays as the body's colour asks. So
/// light gets further in and through thin parts, and on that longer way takes on more of the body's
/// colour: thin parts lit from behind glow in it (wax goes amber), and a thick piece is a little deeper.
pub fn scatter_rgb(colour: [f64; 3], depth: f64, amount: f64, spread: f64, forward: f64) -> ([f64; 3], [f64; 3]) {
    let keep = 1.0 - forward.clamp(0.0, 0.95);
    let spread = spread.clamp(0.0, 1.0);
    let mut s = [0.0; 3];
    let mut a = [0.0; 3];
    for (c, nm) in [610.0f64, 550.0, 465.0].into_iter().enumerate() {
        let sc = scatter(colour, depth * (nm / 550.0).powf(4.0 * spread), amount);
        s[c] = sc.sigma_s * keep;
        a[c] = sc.sigma_a[c];
    }
    (s, a)
}

/// How deep the colour of light crossing a milky body grows with `sssThrough` at 1: the body's colour
/// (scaled so its brightest channel is 1) is raised to this power times the square root of the
/// distance crossed, counted in distances between scatterings (of green). The square root lets thin
/// parts take on colour without thick ones going red. Set by eye against candles lit from behind.
pub const THROUGH_GAIN: f64 = 2.0;

/// See `Scatter::reach`: in units of the distance between scatterings (set by comparing the estimate
/// with light followed as it wanders).
pub const SCATTER_REACH: f64 = 2.0;

/// The unit vector along the crystal's axis in the shape's own frame.
pub fn optic_axis(az_deg: f64, el_deg: f64) -> [f64; 3] {
    let (a, e) = (az_deg.to_radians(), el_deg.to_radians());
    [e.cos() * a.sin(), e.sin(), e.cos() * a.cos()]
}

/// The metal chosen by the `metal` setting, if it is one of the tabulated ones.
pub fn metal_of(p: &Settings) -> Option<&'static Metal> {
    let i = p.metal.round() as i64;
    if i >= 2 { METALS.get(i as usize - 2) } else { None }
}
/// The spectrum chosen by the `spectrum` setting.
pub fn spectrum_of(p: &Settings) -> Option<&'static Spectrum> {
    let i = p.spectrum.round() as i64;
    if i >= 1 { SPECTRA.get(i as usize - 1) } else { None }
}

/// Absorption by wavelength said as three numbers, for light that has already passed through the body
/// once and meets another piece of it: the red, green and blue absorptions that give a piece `thick`
/// thick the colour the spectrum gives it.
pub fn spectrum_rgb(s: &Spectrum, density: f64, thick: f64) -> [f64; 3] {
    let c = colour_of(|l| (-table_at(&s.shape, l) * density * SPECTRUM_SCALE * thick).exp());
    [0, 1, 2].map(|i| -c[i].clamp(1e-4, 1.0).ln() / thick)
}

/// The thickness of shape that the named materials' distances are written for.
pub const FIT_THICKNESS: f64 = 0.3;

/// What the material's distances are multiplied by: 1, or with `fit` on, how thick the shape is next to
/// the thickness the materials are written for. `thickness`: of the shape that is up, if known.
pub fn fit_scale(p: &Settings, thickness: Option<f64>) -> f64 {
    match thickness {
        Some(t) if p.fit > 0.5 => (t / FIT_THICKNESS).clamp(0.02, 10.0),
        _ => 1.0,
    }
}

/// The material's uniform block as the tracer reads it. `scale`: see `fit_scale`.
pub fn uniforms(p: &Settings, scale: f64) -> [f32; MU_FLOATS] {
    let mut u = [0f32; MU_FLOATS];
    let mut put = |row: usize, v: [f64; 4]| { for i in 0..4 { u[row * 4 + i] = v[i] as f32; } };
    let rough = p.rough.clamp(0.0, 1.0);
    let coat = p.coat.clamp(0.0, 1.0);
    // surf: roughness as the spread of the surface's slopes, the share of it that is sampled while the
    // view moves, the index the body sees above it (air with no coat, the coat's own with a full one),
    // the share that is sampled now (the renderer puts the moving share here while the view moves)
    // The share of the roughness that is sampled: when the view is still, and (smaller) while it moves.
    // (`roughRef` between 0 and 1 sets the share itself, moving or not: for trying other splits.)
    let alpha = rough * rough;
    let (still, moving) = if p.rough_ref >= 1.0 {
        (1.0, 1.0)
    } else if p.rough_ref > 0.0 {
        (p.rough_ref, p.rough_ref)
    } else {
        (ROUGH_SAMPLED, ROUGH_SAMPLED.min(ROUGH_SAMPLED_MOVING / alpha.max(1e-9)))
    };
    put(0, [alpha, moving, 1.0 + (p.coat_ior - 1.0) * coat, still]);
    put(1, [p.film.max(0.0), p.film_ior, p.film_var.clamp(0.0, 1.0), p.film_scale]);
    put(2, [p.film_flow, p.film_drain, 0.0, 0.0]);
    let metal = p.metal.round().max(0.0);
    put(3, [p.body_r, p.body_g, p.body_b, if metal >= 2.0 && metal_of(p).is_some() { 1.0 } else { 0.0 }]);
    let sc = scatter([p.body_r, p.body_g, p.body_b], p.sss_depth * crate::material::sss_fit(scale), p.sss);
    // (the last: the index of a thin wall's own glass)
    put(4, [sc.sigma_s, sc.reach, WALK_EVENTS, p.ior]);
    // the scattering and absorption of each colour (they differ when the colours scatter differently)
    let (ss3, sa3) = scatter_rgb([p.body_r, p.body_g, p.body_b], p.sss_depth * crate::material::sss_fit(scale), p.sss, p.sss_spread, p.sss_forward);
    put(5, [sa3[0], sa3[1], sa3[2], p.sss_hue.clamp(0.0, 1.0)]);
    // (the last: how fast the light that crosses the body unscattered takes on its colour, per unit length,
    // in step with how densely the body scatters: trace_material.wgsl, throughHue)
    put(6, [ss3[0], ss3[1], ss3[2], p.sss_through.clamp(0.0, 1.0) * THROUGH_GAIN * ss3[1].max(0.0).sqrt()]);
    // (the last: how uneven the colour is, read with the surface's wear)
    put(7, [p.body_r.clamp(0.0, 1.0), p.body_g.clamp(0.0, 1.0), p.body_b.clamp(0.0, 1.0), p.mottle.clamp(0.0, 1.0)]);
    let g = p.glow.max(0.0) / scale;
    let core = p.glow_core.clamp(0.0, 1.0);
    put(8, [p.glow_r * g, p.glow_g * g, p.glow_b * g, 8.0 * core * core]);
    let density = p.density / scale;
    let srgb = spectrum_of(p).map_or([0.0; 3], |s| spectrum_rgb(s, density, FIT_THICKNESS * scale));
    // (the last is the film's sky a second time: the tracer reads it from two places on purpose, see skyFar)
    put(9, [p.grin, 1.0 / (p.grin_depth * scale).max(1e-3), p.biref, p.film_sky.max(0.0)]);
    // the crystal's axis; or, for a thin wall (which has no crystal), the colour its tint gives light
    // each time it passes through the wall
    let mut a = optic_axis(p.biref_az, p.biref_el);
    if p.hollow > 0.5 && metal < 0.5 { a = [0, 1, 2].map(|i| (-p.absorb[i] * p.density / scale * WALL).exp()); }
    put(10, [a[0], a[1], a[2], 0.0]);
    // the film's sky (row 11 is the environment picture's, filled in by the engine)
    // and the surface: sharp reflections of a picture, wear, polishing lines (trace_material.wgsl: wearAt)
    // and the surface: sharp reflections of a picture, how used it is, polishing lines (trace_material.wgsl: wearAt)
    put(12, [p.film_sky.max(0.0), if p.sharp > 0.5 { 1.0 } else { 0.0 }, p.age.max(0.0), p.brush.clamp(0.0, 1.0)]);
    // the tables: x = the metal's n, y = its k, z = absorption per unit length at this wavelength,
    // w = unused; and the three-number version of the absorption rides in the spare numbers above
    u[2 * 4 + 2] = srgb[0] as f32;
    u[2 * 4 + 3] = srgb[1] as f32;
    u[10 * 4 + 3] = srgb[2] as f32;
    let (m, s) = (metal_of(p), spectrum_of(p));
    for j in 0..32 {
        let k = j.min(NM_N - 1);
        let o = (13 + j) * 4;
        u[o] = m.map_or(1.0, |m| m.n[k]);
        u[o + 1] = m.map_or(0.0, |m| m.k[k]);
        u[o + 2] = s.map_or(0.0, |s| (s.shape[k] as f64 * density * SPECTRUM_SCALE) as f32);
    }
    for (i, v) in film::numbers(p).iter().enumerate() { u[MU_FILM + i] = *v as f32; }
    // the surface's marks, each times how used it is, and the colour recesses take on
    let mk = marks(p);
    for (i, at) in MU_MARKS.iter().enumerate() { u[*at] = (mk[i] * p.age.max(0.0)) as f32; }
    for (i, at) in MU_CAVITY.iter().enumerate() { u[*at] = [p.cavity_r, p.cavity_g, p.cavity_b][i].clamp(0.0, 1.0) as f32; }
    u
}

/// The surface's marks in the order the tracer reads them: edge wear, cavity, scratches, swirls, peel,
/// smudges, dust (each 0 to 1).
pub fn marks(p: &Settings) -> [f64; 7] {
    [p.edge_wear, p.cavity, p.scratches, p.swirls, p.peel, p.smudges, p.dust].map(|v| v.clamp(0.0, 1.0))
}
/// Where the marks (times `age`) and the recesses' colour ride: the unused fourth numbers of rows 7 to
/// 16 of the tables (trace_material.wgsl reads them as mu.tab[7].w ...).
pub const MU_MARKS: [usize; 7] = [20 * 4 + 3, 21 * 4 + 3, 22 * 4 + 3, 23 * 4 + 3, 24 * 4 + 3, 25 * 4 + 3, 26 * 4 + 3];
pub const MU_CAVITY: [usize; 3] = [27 * 4 + 3, 28 * 4 + 3, 29 * 4 + 3];

// ---- the lights, softened ------------------------------------------------------------------------------

/// The softened copies of the lights' map that rough and milky materials read instead of taking many
/// samples: each copy half the size of the one before and blurred twice as far. `rgba`: the map at full
/// size (RGBA f32, `w` x `h`, 2:1, row 0 the top of the sky). Returns (width, height, RGBA) for copies
/// 1, 2, ... down to one that is two texels wide.
///
/// Copy `k` is the map blurred on the sphere by a rounded bell of standard deviation (360 / w) x 2^k
/// degrees: one of its own texels. The tracer picks the copy by that rule (`envL`).
pub fn env_levels(rgba: &[f32], w: usize, h: usize) -> Vec<(usize, usize, Vec<f32>)> {
    use std::f32::consts::PI;
    let texel = 2.0 * PI / w as f32;
    let mut out: Vec<(usize, usize, Vec<f32>)> = Vec::new();
    let (mut sw, mut sh) = (w, h);
    let mut level = 1;
    while sw > 2 && sh > 1 {
        let (dw, dh) = (sw / 2, sh / 2);
        // what has to be added to the blur of the copy before to make this one's
        let sigma = texel * (1 << level) as f32;
        let before = if level == 1 { 0.0 } else { texel * (1 << (level - 1)) as f32 };
        let add = (sigma * sigma - before * before).sqrt();
        let src: &[f32] = out.last().map_or(rgba, |l| &l.2);
        let reach = (3.0 * add).min(PI);
        let k = 1.0 / (2.0 * add * add);
        // the source's directions and the share of the sphere each texel covers
        let rows: Vec<(f32, f32)> = (0..sh).map(|j| { let t = (j as f32 + 0.5) / sh as f32 * PI; (t.sin(), t.cos()) }).collect();
        let cols: Vec<(f32, f32)> = (0..sw).map(|i| { let p = ((i as f32 + 0.5) / sw as f32 - 0.5) * 2.0 * PI; (p.cos(), p.sin()) }).collect();
        let mut dst = vec![0f32; dw * dh * 4];
        dst.par_chunks_mut(dw * 4).enumerate().for_each(|(jd, row)| {
            let t0 = (jd as f32 + 0.5) / dh as f32 * PI;
            let (st0, ct0) = (t0.sin(), t0.cos());
            let j0 = (((t0 - reach) / PI * sh as f32).floor().max(0.0)) as usize;
            let j1 = (((t0 + reach) / PI * sh as f32).ceil() as usize).min(sh);
            for id in 0..dw {
                let p0 = ((id as f32 + 0.5) / dw as f32 - 0.5) * 2.0 * PI;
                let d0 = [st0 * p0.cos(), ct0, st0 * p0.sin()];
                let (mut sum, mut ws) = ([0f32; 3], 0f32);
                for j in j0..j1 {
                    let (st, ct) = rows[j];
                    // how many texels of this row lie within reach, either side of the one straight below
                    let half = if st * sw as f32 <= 1e-3 { sw } else { ((reach / st) / (2.0 * PI) * sw as f32).ceil() as usize + 1 };
                    let centre = ((p0 / (2.0 * PI) + 0.5) * sw as f32).floor() as isize;
                    let (a, n) = if 2 * half + 1 >= sw { (0isize, sw) } else { (centre - half as isize, 2 * half + 1) };
                    for q in 0..n {
                        let i = (a + q as isize).rem_euclid(sw as isize) as usize;
                        let (cp, sp) = cols[i];
                        let d = [st * cp - d0[0], ct - d0[1], st * sp - d0[2]];
                        let wgt = (-(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) * k).exp() * st;
                        let o = (j * sw + i) * 4;
                        sum[0] += src[o] * wgt;
                        sum[1] += src[o + 1] * wgt;
                        sum[2] += src[o + 2] * wgt;
                        ws += wgt;
                    }
                }
                let o = id * 4;
                let inv = if ws > 0.0 { 1.0 / ws } else { 0.0 };
                row[o] = sum[0] * inv;
                row[o + 1] = sum[1] * inv;
                row[o + 2] = sum[2] * inv;
                row[o + 3] = 1.0;
            }
        });
        out.push((dw, dh, dst));
        sw = dw;
        sh = dh;
        level += 1;
    }
    out
}

// ---- bubbles -------------------------------------------------------------------------------------------

/// What the pass that carves bubbles into the finished volume is told.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bubbles {
    /// side of the cells the body is divided into: each holds one bubble at most
    pub cell: f32,
    /// the largest bubble's radius
    pub size: f32,
    /// 1 = all alike and evenly spread, 0 = all sizes, in clusters
    pub even: f32,
    /// -1 = flat discs, 0 = round, 1 = needles
    pub shape: f32,
    pub seed: f32,
}

/// The bubbles as the settings ask for them (None: there are none).
pub fn bubbles(p: &Settings) -> Option<Bubbles> {
    if p.bubbles <= 0.0 { return None; }
    let size = p.bubble_size.clamp(0.004, 0.25);
    let shape = p.bubble_shape.clamp(-1.0, 1.0);
    // how far the bubble reaches from its centre: a needle or a disc is up to four radii long
    let extent = size * (1.0 + 3.0 * shape.abs());
    // "how many" is how much of the body the cells' bubbles would fill: packed as tightly as the cells
    // allow at 1, a hundredth of that at 0.01
    let cell = 2.5 * extent / p.bubbles.clamp(0.002, 1.0).cbrt();
    Some(Bubbles { cell: cell as f32, size: size as f32, even: p.bubble_even.clamp(0.0, 1.0) as f32, shape: shape as f32, seed: p.bubble_seed.round().rem_euclid(4096.0) as f32 })
}

// ---- named materials ---------------------------------------------------------------------------------

/// A material by name: the settings it sets when chosen. Everything it does not mention goes back to
/// plain glass first, so each is complete in itself.
pub struct Named {
    pub id: &'static str,
    pub name: &'static str,
    /// which part of the list it is shown in
    pub group: &'static str,
    pub note: &'static str,
    pub sets: &'static [(&'static str, f64)],
    /// the tint, as absorption of red, green and blue per unit length (None: clear)
    pub absorb: Option<[f64; 3]>,
    /// it switches on one of the costly ways of tracing
    pub costly: bool,
}

/// The settings a named material may set, and what choosing any material puts them back to first.
/// (`ior`, `disp`, `nb` and the tint go back to the family's own instead: see `State::set_material`.)
pub const RESET: &[(&str, f64)] = &[
    ("density", 1.0), ("fit", 0.0), ("spectrum", 0.0), ("rough", 0.0), ("roughRef", 0.0),
    ("film", 0.0), ("filmIor", 1.33), ("filmVar", 0.0), ("filmScale", 1.0), ("filmFlow", 0.0), ("filmDrain", 0.0), ("filmSky", 0.5),
    ("coat", 0.0), ("coatIor", 1.5), ("metal", 0.0), ("hollow", 0.0), ("bodyR", 1.0), ("bodyG", 1.0), ("bodyB", 1.0),
    ("sss", 0.0), ("sssDepth", 0.2), ("sssWalk", 0.0), ("sssSpread", 0.0), ("sssHue", 0.0), ("sssForward", 0.0), ("sssThrough", 0.0), ("sharp", 1.0), ("age", 1.0), ("edgeWear", 0.0), ("cavity", 0.0), ("cavityR", 0.35), ("cavityG", 0.27), ("cavityB", 0.2), ("scratches", 0.0), ("swirls", 0.0), ("peel", 0.0), ("smudges", 0.0), ("dust", 0.0), ("brush", 0.0), ("mottle", 0.0),
    ("bubbles", 0.0), ("bubbleSize", 0.03), ("bubbleEven", 0.5), ("bubbleShape", 0.0), ("bubbleSeed", 1.0),
    ("glow", 0.0), ("glowR", 1.0), ("glowG", 0.32), ("glowB", 0.08), ("glowCore", 0.5),
    ("grin", 0.0), ("grinDepth", 0.02), ("biref", 0.0), ("birefAz", 0.0), ("birefEl", 90.0),
];

impl Named {
    /// Put this material's settings into `p`, over plain glass. (The index, dispersion, bounces and tint
    /// are left as they are unless the material sets them: the caller puts the family's own there first.)
    /// A material that brings distances of its own (a tint, a spectrum, milkiness, a glow) is fitted to
    /// the shape's thickness; one that only changes the family's glass (its surface, bubbles in it)
    /// leaves the glass exactly as the family has it.
    pub fn apply(&self, p: &mut Settings) {
        for (k, v) in RESET { p.set(k, *v); }
        for (k, v) in crate::inner::RESET { p.set(k, *v); }
        if let Some(a) = self.absorb { p.absorb = a; }
        for (k, v) in self.sets { p.set(k, *v); }
        if self.absorb.is_some() || p.spectrum >= 0.5 || p.sss > 0.0 || p.glow > 0.0 || p.grin != 0.0 || crate::inner::Parts::of(p, true).any() { p.fit = 1.0; }
    }
    /// Does this material leave the family's own index, dispersion and tint alone?
    pub fn is_family(&self) -> bool {
        self.sets.is_empty() && self.absorb.is_none()
    }
}

/// The index of a spectrum by id, as the `spectrum` setting counts them.
const fn sp(i: usize) -> f64 { (i + 1) as f64 }
/// The index of a metal by position in `METALS`, as the `metal` setting counts them.
const fn me(i: usize) -> f64 { (i + 2) as f64 }

macro_rules! named {
    ($id:literal, $name:literal, $group:literal, $note:literal, [$(($k:literal, $v:expr)),* $(,)?]) => {
        Named { id: $id, name: $name, group: $group, note: $note, sets: &[$(($k, $v)),*], absorb: None, costly: false }
    };
    ($id:literal, $name:literal, $group:literal, $note:literal, [$(($k:literal, $v:expr)),* $(,)?], absorb $a:expr) => {
        Named { id: $id, name: $name, group: $group, note: $note, sets: &[$(($k, $v)),*], absorb: Some($a), costly: false }
    };
    ($id:literal, $name:literal, $group:literal, $note:literal, [$(($k:literal, $v:expr)),* $(,)?], costly) => {
        Named { id: $id, name: $name, group: $group, note: $note, sets: &[$(($k, $v)),*], absorb: None, costly: true }
    };
    ($id:literal, $name:literal, $group:literal, $note:literal, [$(($k:literal, $v:expr)),* $(,)?], absorb $a:expr, costly) => {
        Named { id: $id, name: $name, group: $group, note: $note, sets: &[$(($k, $v)),*], absorb: Some($a), costly: true }
    };
}

const CLEAR: [f64; 3] = [0.0; 3];

/// Every named material, in the order the list shows them. Only ever add to the end of a group's run
/// in a way that keeps earlier positions: the `mat` setting of a saved setup is a position in this list.
/// Indices and dispersions are at 550 nm and between 400 and 700 nm, from the published formulas named
/// in docs/materials.md.
pub static NAMED: &[Named] = &[
    // ---- glass
    named!("glass", "glass (this family's own)", "glass", "The glass each family of shapes comes with: stronger colour splitting than any real glass, chosen for looks.", []),
    named!("crown", "crown glass", "glass", "Ordinary optical glass (N-BK7): what windows and lenses are near.", [("ior", 1.5185), ("disp", 0.0178)], absorb [0.03, 0.01, 0.02]),
    named!("leadcrystal", "lead crystal", "glass", "Flint glass (F2): heavier, brighter, twice the fire of crown glass.", [("ior", 1.6237), ("disp", 0.0396)], absorb CLEAR),
    named!("flint", "dense flint", "glass", "Dense flint (SF11): the strongest colour splitting of the common glasses.", [("ior", 1.7911), ("disp", 0.0733)], absorb [0.02, 0.02, 0.05]),
    named!("silica", "fused quartz", "glass", "Fused silica: a low index and very little colour splitting.", [("ior", 1.4599), ("disp", 0.0148)], absorb CLEAR),
    named!("acrylic", "acrylic", "glass", "Acrylic (PMMA).", [("ior", 1.4943), ("disp", 0.0187)], absorb CLEAR),
    named!("water", "water", "glass", "Water at 20 C: bends light the least of anything here but air.", [("ior", 1.3347), ("disp", 0.0130)], absorb [0.045, 0.006, 0.002]),
    named!("ice", "ice", "glass", "Clear ice.", [("ior", 1.3110), ("disp", 0.0125)], absorb [0.03, 0.006, 0.002]),
    named!("cloudyice", "cloudy ice", "glass", "Ice with air trapped in it: a little milky, with small bubbles.", [("ior", 1.3110), ("disp", 0.0125), ("sss", 0.1), ("sssDepth", 0.5), ("bodyR", 0.92), ("bodyG", 0.96), ("bodyB", 1.0), ("bubbles", 0.12), ("bubbleSize", 0.018), ("bubbleEven", 0.2)], absorb [0.03, 0.006, 0.002]),
    named!("obsidian", "obsidian", "glass", "Volcanic glass: so dark that only its surface shows.", [("ior", 1.4913), ("disp", 0.0129), ("density", 1.0)], absorb [14.0, 15.0, 16.0]),
    // ---- gems
    named!("diamond", "diamond", "gems", "The highest index of the natural clear gems; more bounces are followed so that light trapped inside finds its way out.", [("ior", 2.4230), ("disp", 0.0579), ("nb", 6.0)], absorb CLEAR),
    named!("moissanite", "moissanite", "gems", "Silicon carbide: above diamond in index, with two and a half times its fire. (It also refracts doubly: set double refraction to 0.055 to see it.)", [("ior", 2.6600), ("disp", 0.1458), ("nb", 6.0)], absorb CLEAR),
    named!("zirconia", "cubic zirconia", "gems", "The usual stand-in for diamond.", [("ior", 2.1661), ("disp", 0.0793), ("nb", 5.0)], absorb CLEAR),
    named!("sapphire", "white sapphire", "gems", "Colourless corundum.", [("ior", 1.7705), ("disp", 0.0233)], absorb CLEAR),
    named!("quartz", "rock crystal", "gems", "Clear quartz.", [("ior", 1.5460), ("disp", 0.0171)], absorb CLEAR),
    named!("ruby", "ruby", "gems", "Chromium in corundum, from a measured spectrum: red that deepens towards crimson as the stone gets thicker.", [("ior", 1.7705), ("disp", 0.0233), ("spectrum", sp(0)), ("density", 2.6)]),
    named!("emerald", "emerald", "gems", "Chromium and vanadium in beryl, from a measured spectrum.", [("ior", 1.5848), ("disp", 0.0180), ("spectrum", sp(1)), ("density", 1.7)]),
    named!("bluesapphire", "blue sapphire", "gems", "Iron and titanium in corundum, from a measured spectrum.", [("ior", 1.7705), ("disp", 0.0233), ("spectrum", sp(2)), ("density", 2.0)]),
    named!("amethyst", "amethyst", "gems", "Irradiated iron in quartz, from a measured spectrum.", [("ior", 1.5460), ("disp", 0.0171), ("spectrum", sp(3)), ("density", 1.6)]),
    named!("citrine", "citrine", "gems", "Heated amethyst, from a spectrum traced off a published plot.", [("ior", 1.5460), ("disp", 0.0171), ("spectrum", sp(4)), ("density", 1.2)]),
    named!("smoky", "smoky quartz", "gems", "Irradiated quartz, from a measured spectrum.", [("ior", 1.5460), ("disp", 0.0171), ("spectrum", sp(5)), ("density", 0.8)]),
    named!("topaz", "sherry topaz", "gems", "Brown topaz, from a measured spectrum.", [("ior", 1.6200), ("disp", 0.0170), ("spectrum", sp(6)), ("density", 0.8)]),
    named!("amber", "amber", "gems", "Fossil resin. Its colour here is an approximation, not a measurement; a few round bubbles and flat 'sun spangles' are trapped in it.", [("ior", 1.5400), ("disp", 0.0200), ("bubbles", 0.05), ("bubbleSize", 0.035), ("bubbleShape", -0.6), ("bubbleEven", 0.15)], absorb [0.25, 1.1, 4.2]),
    // ---- coloured glass
    named!("cobalt", "cobalt blue glass", "coloured glass", "Cobalt in glass (the SCHOTT BG3 filter's measured spectrum): deep blue that lets the far red through, so thick pieces turn violet.", [("spectrum", sp(7)), ("density", 1.8)]),
    named!("didymium", "didymium glass", "coloured glass", "Neodymium and praseodymium (SCHOTT BG36): a narrow band cut out of the yellow, so its colour changes with the light.", [("spectrum", sp(8)), ("density", 1.2)]),
    named!("aqua", "copper aqua glass", "coloured glass", "Copper in glass (SCHOTT BG39): blue-green.", [("spectrum", sp(9)), ("density", 1.4)]),
    named!("signal", "selenium ruby glass", "coloured glass", "The red of signal lamps (SCHOTT RG610): nothing shorter than orange gets through.", [("spectrum", sp(10)), ("density", 1.0)]),
    named!("yellow", "cadmium yellow glass", "coloured glass", "A sharp-cut yellow (SCHOTT GG495).", [("spectrum", sp(11)), ("density", 1.0)]),
    named!("amberglass", "amber bottle glass", "coloured glass", "The brown of beer bottles (iron and sulphur), traced from a published spectrum.", [("spectrum", sp(12)), ("density", 1.4)]),
    named!("manganese", "manganese purple glass", "coloured glass", "Manganese in glass, traced from a published spectrum.", [("spectrum", sp(13)), ("density", 1.4)]),
    named!("bottlegreen", "chromium green glass", "coloured glass", "Chromium in glass, traced from a published figure.", [("spectrum", sp(14)), ("density", 1.5)]),
    named!("cranberry", "cranberry glass", "coloured glass", "Gold ruby glass: gold in particles far smaller than light. Its spectrum is calculated from gold's optical constants, not measured.", [("spectrum", sp(15)), ("density", 1.5)]),
    named!("uranium", "uranium glass", "coloured glass", "Pale yellow-green. An approximation from descriptions; the real thing owes its green glow to fluorescence, which is not modelled.", [("spectrum", sp(16)), ("density", 1.5)]),
    named!("bronze", "bronze glass", "coloured glass", "Bronze-tinted window glass, from a measured spectrum.", [("spectrum", sp(17)), ("density", 1.0)]),
    named!("smoked", "smoked glass", "coloured glass", "Grey-tinted window glass, from a measured spectrum.", [("spectrum", sp(18)), ("density", 1.0)]),
    named!("deepwater", "deep water", "coloured glass", "Water's own colour (measured), as strong as it would be through many metres.", [("ior", 1.3347), ("disp", 0.0130), ("spectrum", sp(19)), ("density", 0.6)]),
    // ---- surfaces
    named!("satin", "satin glass", "surface", "Lightly etched: reflections soften, the inside blurs.", [("rough", 0.28)]),
    named!("frosted", "frosted glass", "surface", "Sandblasted: light comes through, shapes do not.", [("rough", 0.6)]),
    named!("soap", "soap bubble", "surface", "A film of soapy water around air, half drained. It mirrors its surroundings twice, upright in its near wall and upside down in its far one, in colours that are all interference; nothing else shows but what is behind it.", [("hollow", 1.0), ("film", 1300.0), ("filmIor", 1.33), ("filmVar", 0.12), ("filmScale", 1.3), ("filmDrain", 0.5), ("filmSky", 3.0), ("nb", 3.0)], absorb CLEAR),
    named!("oil", "oil on glass", "surface", "A thin uneven coating of higher index on dark glass: the colours of oil on a wet road.", [("film", 330.0), ("filmIor", 1.9), ("filmVar", 0.45), ("filmScale", 1.6), ("filmSky", 1.5)], absorb [5.0, 5.5, 6.0]),
    named!("dichroic", "dichroic glass", "surface", "An even hard coating: one colour reflected, the opposite one let through.", [("film", 260.0), ("filmIor", 2.35), ("filmVar", 0.06)], absorb CLEAR),
    named!("oilslick", "oil slick", "surface", "A film of oil on a black, wet surface: the full run of interference colours. (The oil is given a higher index than oil has, 1.7 for 1.47, so that its colours are as strong on the screen as they look on a wet road.)", [("ior", 1.33), ("disp", 0.0), ("sss", 1.0), ("sssDepth", 0.02), ("bodyR", 0.012), ("bodyG", 0.012), ("bodyB", 0.016), ("film", 380.0), ("filmIor", 1.7), ("filmVar", 0.7), ("filmScale", 0.9), ("filmSky", 2.0)], absorb CLEAR),
    named!("lustre", "lustre glaze", "surface", "Iridescent pottery: a dark glazed body with a metallic film fired on to it.", [("disp", 0.0), ("sss", 1.0), ("sssDepth", 0.03), ("bodyR", 0.10), ("bodyG", 0.05), ("bodyB", 0.04), ("rough", 0.4), ("coat", 1.0), ("coatIor", 1.55), ("film", 300.0), ("filmIor", 2.1), ("filmVar", 0.35), ("filmScale", 1.4), ("filmSky", 0.7)], absorb CLEAR),
    named!("beetle", "beetle shell", "surface", "A dark body under a film of high index: green that turns to blue and violet towards the outline. (One film stands in for the many thin layers of a real beetle's shell.)", [("disp", 0.0), ("sss", 1.0), ("sssDepth", 0.02), ("bodyR", 0.02), ("bodyG", 0.03), ("bodyB", 0.02), ("film", 330.0), ("filmIor", 2.0), ("filmVar", 0.05), ("filmScale", 0.6), ("filmSky", 0.9), ("coat", 0.0)], absorb CLEAR),
    named!("carnival", "carnival glass", "surface", "Iridescent sprayed glass: a varying film over amber.", [("film", 520.0), ("filmIor", 2.0), ("filmVar", 0.35), ("filmScale", 2.4), ("spectrum", sp(12)), ("density", 0.6)]),
    // ---- translucent
    named!("milk", "milk glass", "translucent", "White opal glass: light goes in and wanders out again; thin edges let some through.", [("sss", 1.0), ("sssDepth", 0.12), ("bodyR", 0.995), ("bodyG", 0.995), ("bodyB", 0.99), ("disp", 0.0), ("sssSpread", 0.6), ("peel", 0.2), ("smudges", 0.2), ("dust", 0.12)], absorb CLEAR),
    named!("opal", "opalescent glass", "translucent", "Half clear, half milky, bluish where light scatters.", [("sss", 0.3), ("sssDepth", 0.5), ("bodyR", 0.78), ("bodyG", 0.88), ("bodyB", 1.0)], absorb [0.02, 0.06, 0.16]),
    named!("jade", "jade", "translucent", "Polished green jade: bright and pale where it is thin, deep green in the bulk, glowing when lit from behind.", [("ior", 1.61), ("disp", 0.0), ("sss", 1.0), ("sssDepth", 0.055), ("bodyR", 0.33), ("bodyG", 0.53), ("bodyB", 0.28), ("rough", 0.0), ("mottle", 0.6), ("sssSpread", 0.35), ("edgeWear", 0.3), ("cavity", 0.3), ("cavityR", 0.45), ("cavityG", 0.45), ("cavityB", 0.4), ("dust", 0.1)], absorb CLEAR),
    named!("wax", "wax", "translucent", "Candle wax: soft, warm, glowing where it is thin.", [("ior", 1.44), ("disp", 0.0), ("sss", 1.0), ("sssDepth", 0.2), ("bodyR", 0.97), ("bodyG", 0.80), ("bodyB", 0.52), ("rough", 0.35), ("mottle", 0.25), ("sssSpread", 0.3), ("smudges", 0.5), ("sssHue", 1.0), ("sssForward", 0.65), ("sssThrough", 1.0)], absorb CLEAR),
    named!("alabaster", "alabaster", "translucent", "Pale stone with a soft sheen.", [("ior", 1.52), ("disp", 0.0), ("sss", 1.0), ("sssDepth", 0.05), ("bodyR", 0.94), ("bodyG", 0.90), ("bodyB", 0.84), ("rough", 0.22), ("mottle", 0.5), ("sssSpread", 0.4)], absorb CLEAR),
    named!("rosequartz", "rose quartz", "translucent", "Cloudy pink quartz.", [("ior", 1.546), ("disp", 0.0171), ("sss", 0.5), ("sssDepth", 0.35), ("bodyR", 0.98), ("bodyG", 0.74), ("bodyB", 0.78)], absorb [0.02, 0.10, 0.07]),
    named!("honey", "honey", "translucent", "Thick, golden and slightly cloudy.", [("ior", 1.49), ("disp", 0.01), ("sss", 0.2), ("sssDepth", 0.6), ("bodyR", 0.98), ("bodyG", 0.72), ("bodyB", 0.22)], absorb [0.10, 0.75, 3.2]),
    // ---- ceramic
    named!("porcelain", "glazed porcelain", "ceramic", "A dense white body under a clear glaze.", [("disp", 0.0), ("sss", 1.0), ("sssDepth", 0.06), ("bodyR", 0.99), ("bodyG", 0.99), ("bodyB", 0.985), ("rough", 0.5), ("coat", 1.0), ("coatIor", 1.5), ("sssSpread", 0.15), ("peel", 0.5), ("dust", 0.15), ("cavity", 0.2), ("cavityR", 0.5), ("cavityG", 0.48), ("cavityB", 0.45)], absorb CLEAR),
    named!("celadon", "celadon", "ceramic", "Grey-green glazed stoneware.", [("disp", 0.0), ("sss", 1.0), ("sssDepth", 0.05), ("bodyR", 0.60), ("bodyG", 0.77), ("bodyB", 0.66), ("rough", 0.5), ("coat", 1.0), ("coatIor", 1.5), ("mottle", 0.2), ("peel", 0.5), ("dust", 0.15), ("cavity", 0.2), ("cavityR", 0.5), ("cavityG", 0.48), ("cavityB", 0.45)], absorb CLEAR),
    named!("pearl", "pearl", "ceramic", "A milky body under a clear coat that shimmers with thin-film colour.", [("disp", 0.0), ("sss", 1.0), ("sssDepth", 0.05), ("bodyR", 0.96), ("bodyG", 0.93), ("bodyB", 0.89), ("rough", 0.3), ("coat", 1.0), ("coatIor", 1.53), ("film", 440.0), ("filmIor", 1.68), ("filmVar", 0.22), ("filmScale", 2.5)], absorb CLEAR),
    named!("lacquer", "black lacquer", "ceramic", "A black body under a deep clear coat.", [("disp", 0.0), ("sss", 1.0), ("sssDepth", 0.02), ("bodyR", 0.012), ("bodyG", 0.012), ("bodyB", 0.014), ("rough", 0.6), ("coat", 1.0), ("coatIor", 1.55), ("peel", 0.4), ("swirls", 0.45), ("dust", 0.15), ("smudges", 0.2)], absorb CLEAR),
    // ---- metal
    named!("mirror", "perfect mirror", "metal", "Reflects everything, whatever the angle or colour.", [("metal", 1.0)]),
    named!("silver", "silver", "metal", "Measured optical constants (Johnson & Christy).", [("metal", me(0)), ("mottle", 0.15), ("edgeWear", 0.4), ("cavity", 0.55), ("scratches", 0.3), ("smudges", 0.3), ("cavityR", 0.07), ("cavityG", 0.06), ("cavityB", 0.05)]),
    named!("aluminium", "aluminium", "metal", "Measured optical constants (Rakic).", [("metal", me(1)), ("mottle", 0.15), ("edgeWear", 0.3), ("cavity", 0.3), ("scratches", 0.3), ("smudges", 0.2), ("cavityR", 0.32), ("cavityG", 0.22), ("cavityB", 0.13)]),
    named!("gold", "gold", "metal", "Measured optical constants (Johnson & Christy).", [("metal", me(2)), ("mottle", 0.15), ("edgeWear", 0.5), ("cavity", 0.4), ("scratches", 0.45), ("smudges", 0.2), ("cavityR", 0.32), ("cavityG", 0.22), ("cavityB", 0.13)]),
    named!("copper", "copper", "metal", "Measured optical constants (Johnson & Christy).", [("metal", me(3)), ("mottle", 0.15), ("edgeWear", 0.5), ("cavity", 0.5), ("scratches", 0.3), ("smudges", 0.2), ("cavityR", 0.22), ("cavityG", 0.34), ("cavityB", 0.27)]),
    named!("chrome", "chrome", "metal", "Measured optical constants (Johnson & Christy).", [("metal", me(4)), ("mottle", 0.15), ("scratches", 0.12), ("smudges", 0.35), ("peel", 0.3), ("dust", 0.1)]),
    named!("nickel", "nickel", "metal", "Measured optical constants (Johnson & Christy).", [("metal", me(5)), ("mottle", 0.15), ("edgeWear", 0.3), ("cavity", 0.3), ("scratches", 0.3), ("smudges", 0.2), ("cavityR", 0.32), ("cavityG", 0.22), ("cavityB", 0.13)]),
    named!("platinum", "platinum", "metal", "Optical constants from a model fitted to measurements (Rakic et al.).", [("metal", me(6)), ("mottle", 0.15), ("edgeWear", 0.3), ("cavity", 0.3), ("scratches", 0.3), ("smudges", 0.2), ("cavityR", 0.32), ("cavityG", 0.22), ("cavityB", 0.13)]),
    named!("titanium", "titanium", "metal", "Measured optical constants (Johnson & Christy).", [("metal", me(7)), ("mottle", 0.15), ("edgeWear", 0.3), ("cavity", 0.3), ("scratches", 0.3), ("smudges", 0.2), ("cavityR", 0.32), ("cavityG", 0.22), ("cavityB", 0.13)]),
    named!("iron", "iron", "metal", "Measured optical constants (Johnson & Christy).", [("metal", me(8)), ("mottle", 0.15), ("edgeWear", 0.3), ("cavity", 0.3), ("scratches", 0.3), ("smudges", 0.2), ("cavityR", 0.32), ("cavityG", 0.22), ("cavityB", 0.13)]),
    named!("tungsten", "tungsten", "metal", "Optical constants read from a published figure (Weaver, Olson & Lynch).", [("metal", me(9)), ("mottle", 0.15), ("edgeWear", 0.3), ("cavity", 0.3), ("scratches", 0.3), ("smudges", 0.2), ("cavityR", 0.32), ("cavityG", 0.22), ("cavityB", 0.13)]),
    named!("brushed", "satin steel", "metal", "Iron with a fine, even roughness.", [("metal", me(8)), ("rough", 0.0), ("brush", 1.0), ("edgeWear", 0.4), ("scratches", 0.35), ("smudges", 0.25)]),
    named!("satingold", "satin gold", "metal", "Gold, lightly matted.", [("metal", me(2)), ("rough", 0.26), ("brush", 0.4), ("mottle", 0.2), ("edgeWear", 0.6), ("cavity", 0.4), ("scratches", 0.6), ("cavityR", 0.32), ("cavityG", 0.22), ("cavityB", 0.13)]),
    named!("anodised", "anodised titanium", "metal", "Titanium under its own clear oxide: the colour is interference alone and follows the oxide's thickness.", [("metal", me(7)), ("film", 55.0), ("filmIor", 2.4), ("filmVar", 0.45), ("filmScale", 0.7), ("filmSky", 0.8), ("rough", 0.2)]),
    named!("heatsteel", "heat-coloured steel", "metal", "Steel blued in a flame: its oxide grows thicker towards the heat, straw to purple to blue.", [("metal", me(8)), ("film", 66.0), ("filmIor", 2.3), ("filmVar", 0.25), ("filmScale", 0.5), ("filmDrain", 0.06), ("filmSky", 0.8), ("rough", 0.12)]),
    named!("bismuth", "bismuth", "metal", "A grey metal under an oxide of uneven thickness: every colour at once.", [("metal", me(5)), ("film", 190.0), ("filmIor", 2.4), ("filmVar", 0.75), ("filmScale", 1.8), ("filmSky", 0.6)]),
    named!("oilchrome", "oil on chrome", "metal", "A film of oil on polished chrome.", [("metal", me(4)), ("film", 420.0), ("filmIor", 1.45), ("filmVar", 0.6), ("filmScale", 1.1), ("filmSky", 0.5)]),
    named!("candy", "lacquered copper", "metal", "Copper under a clear coat: darker and richer than bare metal.", [("metal", me(3)), ("coat", 1.0), ("coatIor", 1.5), ("rough", 0.18), ("peel", 0.4), ("swirls", 0.45), ("dust", 0.15), ("smudges", 0.2)]),
    // ---- glow
    named!("neon", "neon tube", "glow", "Clear glass with a glowing core. Raise the bloom.", [("glow", 5.0), ("glowR", 1.0), ("glowG", 0.22), ("glowB", 0.05), ("glowCore", 0.6)], absorb CLEAR),
    named!("argon", "argon tube", "glow", "Clear glass glowing blue-violet from within.", [("glow", 4.0), ("glowR", 0.25), ("glowG", 0.35), ("glowB", 1.0), ("glowCore", 0.6)], absorb CLEAR),
    named!("ember", "ember", "glow", "A dark milky body lit from inside, like a hot coal.", [("disp", 0.0), ("sss", 1.0), ("sssDepth", 0.1), ("bodyR", 0.22), ("bodyG", 0.10), ("bodyB", 0.07), ("glow", 0.9), ("glowR", 1.0), ("glowG", 0.16), ("glowB", 0.02), ("glowCore", 0.3), ("rough", 0.5)], absorb CLEAR),
    // ---- inclusions
    named!("seeded", "seeded glass", "inclusions", "Glass with scattered bubbles of all sizes.", [("bubbles", 0.12), ("bubbleSize", 0.03), ("bubbleEven", 0.25)]),
    named!("foam", "foam glass", "inclusions", "Glass crowded with bubbles.", [("bubbles", 0.8), ("bubbleSize", 0.022), ("bubbleEven", 0.7)]),
    named!("silk", "needle quartz", "inclusions", "Clear quartz shot through with fine needles.", [("ior", 1.5460), ("disp", 0.0171), ("bubbles", 0.25), ("bubbleSize", 0.008), ("bubbleShape", 1.0), ("bubbleEven", 0.4)], absorb CLEAR),
    // ---- costly
    named!("calcite", "calcite", "exotic", "Iceland spar: every ray splits in two inside it, so everything behind it is seen double.", [("ior", 1.6613), ("disp", 0.0296), ("biref", -0.1738)], absorb CLEAR, costly),
    named!("truemilk", "milk glass, true scattering", "exotic", "Milk glass with the light followed as it wanders inside. Grainy until it settles: for still views and video.", [("sss", 1.0), ("sssDepth", 0.025), ("bodyR", 0.97), ("bodyG", 0.97), ("bodyB", 0.955), ("disp", 0.0), ("sssWalk", 1.0)], absorb CLEAR, costly),
    named!("truejade", "jade, true scattering", "exotic", "Jade with the light followed as it wanders inside. Grainy until it settles: for still views and video.", [("ior", 1.61), ("disp", 0.0), ("sss", 1.0), ("sssDepth", 0.055), ("bodyR", 0.33), ("bodyG", 0.53), ("bodyB", 0.28), ("rough", 0.08), ("sssWalk", 1.0)], absorb CLEAR, costly),
    named!("gradient", "gradient-index glass", "exotic", "Glass whose index rises below the surface, so that rays curve inside it.", [("grin", 0.35), ("grinDepth", 0.02)], costly),
    // ---- inside the body (added after everything else: a saved setup keeps a material by its position in this list)
    named!("crackedquartz", "cracked quartz", "fractures", "Rock crystal with fractures through it: healed veils, cracks held open in patches, a fan from one blow. Unseen until one catches the light, then a silver or rainbow flash.", [("ior", 1.5460), ("disp", 0.0171), ("fractures", 0.35), ("cracks", 0.3), ("crackSize", 0.25), ("crackOpen", 0.62), ("crackCurve", 0.15), ("crackRough", 0.08), ("crackSeed", 3.0)], absorb CLEAR),
    named!("flakequartz", "flaked quartz", "fractures", "Rock crystal full of small flat flakes: each a mirror at its own angle.", [("ior", 1.5460), ("disp", 0.0171), ("cracks", 0.8), ("crackSize", 0.4), ("crackOpen", 0.9), ("crackCurve", 0.15), ("crackRough", 0.15), ("crackSeed", 3.0)], absorb CLEAR),
    named!("irisquartz", "iris quartz", "fractures", "Rock crystal whose fractures stand open by less than a thousandth of a millimetre: rainbow bands that follow each crack's edge.", [("ior", 1.5460), ("disp", 0.0171), ("fractures", 0.5), ("crackOpen", 0.58), ("crackCurve", 0.1), ("crackSeed", 5.0)], absorb CLEAR),
    named!("crackle", "crackle glass", "fractures", "Glass shocked by heat: full of small cracks, every face a mirror at its own angle.", [("disp", 0.0), ("cracks", 0.85), ("crackSize", 0.13), ("crackDepth", 0.2), ("crackOpen", 0.7), ("crackCurve", 0.7), ("crackSeed", 1.0)], absorb CLEAR),
    named!("crackedice", "cracked ice", "fractures", "Clear ice with white fracture sheets through it.", [("ior", 1.3110), ("disp", 0.0125), ("fractures", 1.0), ("crackOpen", 0.95), ("crackCurve", 0.35), ("crackRough", 0.7), ("crackSeed", 2.0)], absorb [0.05, 0.012, 0.004]),
    named!("whiteopal", "white opal", "opal", "Precious opal on a milky white body: patches of pure colour that change as the stone turns.", [("ior", 1.45), ("disp", 0.0), ("sss", 0.3), ("sssDepth", 0.2), ("bodyR", 0.24), ("bodyG", 0.25), ("bodyB", 0.27), ("play", 0.7), ("playSize", 0.15), ("playColour", 0.75), ("playSoft", 0.1)], absorb CLEAR),
    named!("blackopal", "black opal", "opal", "Precious opal on a dark body: the colours glow out of black.", [("ior", 1.45), ("disp", 0.0), ("sss", 1.0), ("sssDepth", 0.08), ("bodyR", 0.13), ("bodyG", 0.16), ("bodyB", 0.24), ("play", 0.8), ("playSize", 0.16), ("playPattern", 2.0), ("playColour", 1.0), ("playSoft", 0.1)], absorb [4.0, 4.0, 3.5]),
    named!("crystalopal", "crystal opal", "opal", "Precious opal that is nearly clear: the colours float inside the stone.", [("ior", 1.45), ("disp", 0.005), ("sss", 0.04), ("sssDepth", 1.0), ("bodyR", 0.95), ("bodyG", 0.96), ("bodyB", 1.0), ("play", 1.0), ("playSize", 0.14), ("playColour", 0.85), ("playSoft", 0.1)], absorb [0.01, 0.012, 0.02]),
    named!("fireopal", "fire opal", "opal", "Orange, nearly clear opal with a few points of colour.", [("ior", 1.44), ("disp", 0.004), ("sss", 0.06), ("sssDepth", 0.5), ("bodyR", 1.0), ("bodyG", 0.6), ("bodyB", 0.3), ("play", 0.4), ("playSize", 0.25), ("playColour", 0.8), ("playSoft", 0.1)], absorb [0.15, 2.6, 8.0]),
    named!("biquartz", "rock crystal, doubly refracting", "quartz", "Clear quartz with its weak double refraction (every ray is traced twice) and a few fractures.", [("ior", 1.5442), ("disp", 0.0171), ("biref", 0.0091), ("birefEl", 90.0), ("cracks", 0.15), ("crackSize", 0.6), ("crackOpen", 0.85), ("crackSeed", 7.0)], absorb CLEAR, costly),
    named!("chalcedony", "chalcedony", "quartz", "Quartz in crystals too small to see: a waxy, cloudy blue-grey body with faint bands.", [("ior", 1.535), ("disp", 0.0), ("sss", 1.0), ("sssDepth", 0.12), ("bodyR", 0.82), ("bodyG", 0.86), ("bodyB", 0.92), ("rough", 0.15), ("grad", 0.6), ("gradMode", 2.0), ("gradBands", 9.0), ("gradR", 0.75), ("gradG", 0.82), ("gradB", 0.95), ("gradSoft", 0.4), ("gradWave", 0.4)], absorb CLEAR),
    named!("agate", "banded agate", "quartz", "Chalcedony laid down in bands: cloudy white and red-brown rings that follow the stone.", [("ior", 1.54), ("disp", 0.0), ("sss", 0.7), ("sssDepth", 0.15), ("bodyR", 0.92), ("bodyG", 0.88), ("bodyB", 0.82), ("rough", 0.1), ("grad", 1.0), ("gradMode", 2.0), ("gradBands", 7.0), ("gradR", 0.55), ("gradG", 0.25), ("gradB", 0.1), ("gradSoft", 0.08), ("gradWave", 0.5)], absorb [0.02, 0.03, 0.05]),
    named!("aventurine", "aventurine", "quartz", "Green quartz full of tiny mica platelets, each a mirror that sparkles when it catches a light (aventurescence).", [("ior", 1.55), ("disp", 0.0), ("sss", 0.6), ("sssDepth", 0.2), ("bodyR", 0.45), ("bodyG", 0.72), ("bodyB", 0.5), ("play", 1.0), ("playSize", 0.05), ("playPattern", 3.0), ("playSoft", 0.1), ("glintR", 0.75), ("glintG", 0.95), ("glintB", 0.8)], absorb [0.3, 0.06, 0.25]),
    named!("goldstone", "goldstone", "quartz", "Glass full of copper platelets: a sky of copper sparks in brown glass.", [("ior", 1.52), ("disp", 0.01), ("sss", 0.15), ("sssDepth", 0.5), ("bodyR", 0.9), ("bodyG", 0.5), ("bodyB", 0.3), ("play", 1.0), ("playSize", 0.035), ("playPattern", 3.0), ("playSoft", 0.05), ("glintR", 1.0), ("glintG", 0.55), ("glintB", 0.3)], absorb [0.4, 1.2, 2.4]),
    named!("tigerseye", "tiger's eye", "quartz", "Quartz grown through wavy golden fibres: bands of silky light that roll across the stone as it turns (chatoyancy).", [("ior", 1.54), ("disp", 0.0), ("sss", 0.8), ("sssDepth", 0.08), ("bodyR", 0.55), ("bodyG", 0.33), ("bodyB", 0.09), ("silk", 1.0), ("silkSoft", 0.25), ("silkWave", 0.6)], absorb [0.4, 0.9, 2.5]),
    named!("catseye", "cat's-eye quartz", "quartz", "Translucent quartz with fine straight fibres: one sharp line of light across the fibres that glides as the stone moves.", [("ior", 1.544), ("disp", 0.0), ("sss", 0.45), ("sssDepth", 0.2), ("bodyR", 0.78), ("bodyG", 0.82), ("bodyB", 0.72), ("silk", 0.8), ("silkSoft", 0.12)], absorb [0.08, 0.05, 0.1]),
    named!("piezoquartz", "quartz under pressure", "quartz", "Rock crystal cracking under strain: a faint warm glow, brightest where the cracks are still running (piezoluminescence). Animate crack growth and the glow together.", [("ior", 1.5460), ("disp", 0.0171), ("cracks", 0.35), ("crackSize", 0.4), ("crackOpen", 0.85), ("crackGrow", 0.7), ("piezo", 0.6), ("piezoColour", 0.5), ("crackSeed", 4.0)], absorb CLEAR),
    named!("twotone", "violet to cyan glass", "coloured glass", "Glass whose tint shades from violet below to cyan above.", [("grad", 1.0), ("gradMode", 0.0), ("gradEl", 90.0), ("gradR", 0.55), ("gradG", 0.85), ("gradB", 1.0), ("gradSoft", 0.6)], absorb [0.29, 0.51, 0.05]),
];

/// The position of a named material by its id.
pub fn named_index(id: &str) -> Option<usize> {
    NAMED.iter().position(|n| n.id == id)
}

/// A colour to show beside a material's name in a list (linear RGB, 0 to 1): what it roughly looks like.
pub fn swatch(i: usize, family_absorb: [f64; 3]) -> [f64; 3] {
    let Some(n) = NAMED.get(i) else { return [1.0; 3] };
    let mut p = Settings::default();
    p.absorb = family_absorb;
    n.apply(&mut p);
    if let Some(m) = metal_of(&p) {
        return colour_of(|l| fresnel_metal(1.0, 1.0, table_at(&m.n, l), table_at(&m.k, l))).map(|c| c.clamp(0.0, 1.0));
    }
    if p.metal >= 0.5 { return [0.95; 3]; }
    if p.sss >= 0.9 { return [p.body_r, p.body_g, p.body_b]; }
    let through = 0.6;
    let c = match spectrum_of(&p) {
        Some(s) => colour_of(|l| (-table_at(&s.shape, l) * p.density * SPECTRUM_SCALE * through).exp()),
        None => [0, 1, 2].map(|i| (-p.absorb[i] * p.density * through).exp()),
    };
    c.map(|c| c.clamp(0.0, 1.0))
}

// ---- the controls, for the window ----------------------------------------------------------------------

/// Where a control belongs in the window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    /// the `material` group at the head of the left panel
    Main,
    /// the materials lab window, in the group named
    Lab(&'static str),
}

/// How a control is shown.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Slider,
    /// 0 or 1
    Switch,
    /// one of a list: the names, in the order the setting counts them from 0
    Choice,
    /// three settings shown as one colour (this is the first of them; the other two follow it)
    Colour,
    /// a whole number typed or stepped
    Whole,
}

/// One control: a setting by name with its range and what to call it.
pub struct Control {
    pub name: &'static str,
    pub label: &'static str,
    pub kind: Kind,
    pub lo: f64,
    pub hi: f64,
    pub step: f64,
    /// it can be animated (a slider with a wave button)
    pub wave: bool,
    pub place: Place,
    pub note: &'static str,
    /// switching it on makes every frame cost noticeably more
    pub costly: bool,
}

const fn ctl(name: &'static str, label: &'static str, kind: Kind, lo: f64, hi: f64, step: f64, wave: bool, place: Place, note: &'static str) -> Control {
    Control { name, label, kind, lo, hi, step, wave, place, note, costly: false }
}
const fn costly(c: Control) -> Control {
    Control { costly: true, ..c }
}

use Kind::*;
use Place::*;

/// Every material control, in the order the window shows them. (`ior`, `disp` and `nb` are the glass
/// group's existing sliders: they belong to `Main` as well, ahead of these.)
pub static CONTROLS: &[Control] = &[
    ctl("density", "tint depth", Slider, 0.0, 40.0, 0.05, true, Main, "How strongly the body's colour builds up with thickness (0 = clear)."),
    ctl("tintR", "tint", Colour, 0.0, 1.0, 0.005, true, Main, "The colour white light takes on passing through one unit of the body."),
    ctl("tintG", "tint green", Slider, 0.0, 1.0, 0.005, true, Lab("body"), ""),
    ctl("tintB", "tint blue", Slider, 0.0, 1.0, 0.005, true, Lab("body"), ""),
    ctl("fit", "fit to the shape", Switch, 0.0, 1.0, 1.0, false, Lab("body"), "Measure the material's distances (tint depth, how far light travels, glow) against how thick the shape is, so that it looks the same on thin tubes as on a solid. The named materials switch it on."),
    ctl("rough", "frost", Slider, 0.0, 1.0, 0.01, true, Main, "Roughness of the surface: 0 polished, 0.3 satin, 0.6 frosted, 1 sandblasted."),
    ctl("sss", "milkiness", Slider, 0.0, 1.0, 0.01, true, Main, "How much light scatters inside the body: 0 clear, 1 fully milky (jade, wax, porcelain)."),
    ctl("coat", "clear coat", Slider, 0.0, 1.0, 0.01, true, Main, "A glossy clear layer over the body."),
    ctl("film", "thin film nm", Slider, 0.0, 1500.0, 1.0, true, Main, "Thickness of an interference film on the surface, in nanometres (0 = none)."),
    ctl("spectrum", "colour by wavelength", Choice, 0.0, 20.0, 1.0, false, Lab("body"), "Absorb light the way a real coloured glass or gem does, wavelength by wavelength, instead of the tint. `tint depth` sets how strongly."),
    ctl("metal", "metal", Choice, 0.0, 11.0, 1.0, false, Lab("body"), "An opaque reflecting body: a perfect mirror, or a metal with its measured colour."),
    ctl("hollow", "thin wall", Switch, 0.0, 1.0, 1.0, false, Lab("body"), "The shape is a thin wall around air, like a soap bubble or a blown ornament, instead of solid: light passes straight through and is reflected by the wall on the way in and on the way out."),
    ctl("bodyR", "body colour", Colour, 0.0, 1.0, 0.005, true, Lab("body"), "Colour of a milky body (what a thick piece looks like), or a tint on a metal's reflection."),
    ctl("bodyG", "body green", Slider, 0.0, 1.0, 0.005, true, Lab("body"), ""),
    ctl("bodyB", "body blue", Slider, 0.0, 1.0, 0.005, true, Lab("body"), ""),
    ctl("sssDepth", "light travels", Slider, 0.01, 2.0, 0.005, true, Lab("inside"), "How far light gets inside a milky body between scatterings: small = dense and chalky, large = soft and deep."),
    costly(ctl("sssWalk", "true scattering", Switch, 0.0, 1.0, 1.0, false, Lab("inside"), "Follow the light as it wanders inside the body instead of estimating it. Grainy until the picture settles: for still views and video.")),
    ctl("sssSpread", "colours travel apart", Slider, 0.0, 1.0, 0.01, true, Lab("inside"), "0: every colour travels as far inside the body. Towards 1 blue is scattered sooner and red goes deeper, as in milk glass and wax: a bluish edge, a warm glow in the shade and where light comes through."),
    ctl("sssForward", "scatters forward", Slider, 0.0, 0.95, 0.01, true, Lab("inside"), "How much the body scatters light onwards rather than every way, as wax does: light gets further in and through thin parts, taking on more of the body's colour on the way (thin wax lit from behind glows amber)."),
    ctl("sssThrough", "colour through it", Slider, 0.0, 1.0, 0.01, true, Lab("inside"), "How far the light that comes straight through the body takes on its colour, deeper the further it travels: towards 1 thin parts and edges lit from behind glow in the body's colour, as a candle glows amber where it is thin."),
    ctl("sssHue", "colour under the skin", Slider, 0.0, 1.0, 0.01, true, Lab("inside"), "How far the light scattered inside takes on the body's colour at every depth, not only deep in it: towards 1 a dyed body glows its colour just under the surface and through thin parts, as wax does."),
    ctl("meter", "expose for the body", Switch, 0.0, 1.0, 1.0, false, Lab("inside"), "Expose the picture for a milky body, as a photographer exposes for the subject: a white surface facing the brightest light comes out white. The lights glass is lit by are made to be seen in it, and a body that only gathers them would look grey."),
    ctl("sssDetail", "light detail", Slider, 96.0, 256.0, 32.0, false, Lab("inside"), "How fine the volume is in which a milky body's light is worked out (cells along a side). Finer follows thin pieces better; it is worked out again whenever the lights or the shape move, and costs more the finer it is."),
    ctl("sharp", "sharp reflections", Switch, 0.0, 1.0, 1.0, false, Lab("surface"), "A metal, a milky body or a glaze mirrors an environment picture as sharp as it is, as a photograph shows it; off, it mirrors the picture softened as the glass sees it."),
    ctl("age", "age", Slider, 0.0, 2.0, 0.01, true, Main, "How used the surface looks: all of its marks together. 0 new and perfect, 1 as the material has them, 2 twice that."),
    ctl("edgeWear", "worn edges", Slider, 0.0, 1.0, 0.01, true, Lab("surface"), "High points and edges rubbed smoother and brighter, as handling polishes them."),
    ctl("cavity", "dirt in recesses", Slider, 0.0, 1.0, 0.01, true, Lab("surface"), "Recesses and folds duller and darker, holding dirt or tarnish of the colour below, where no cloth reaches."),
    ctl("cavityR", "recess colour", Colour, 0.0, 1.0, 0.005, true, Lab("surface"), "The colour of what the recesses hold: brown dirt, black tarnish on silver, green on copper."),
    ctl("cavityG", "recess green", Slider, 0.0, 1.0, 0.005, true, Lab("surface"), ""),
    ctl("cavityB", "recess blue", Slider, 0.0, 1.0, 0.005, true, Lab("surface"), ""),
    ctl("scratches", "scratches", Slider, 0.0, 1.0, 0.01, true, Lab("surface"), "Fine straight scratches, most where the surface is exposed. Seen where a highlight crosses them; from afar they are a faint sheen."),
    ctl("swirls", "polishing swirls", Slider, 0.0, 1.0, 0.01, true, Lab("surface"), "The arcs a buffing pad leaves on polished metal and on a clear coat, seen as rings round a sharp highlight."),
    ctl("peel", "orange peel", Slider, 0.0, 1.0, 0.01, true, Lab("surface"), "The gentle wave of a sprayed or dipped coat or glaze: sharp reflections waver a little."),
    ctl("smudges", "smudges", Slider, 0.0, 1.0, 0.01, true, Lab("surface"), "Duller patches and finger marks where the object is handled, on its exposed parts."),
    ctl("dust", "dust", Slider, 0.0, 1.0, 0.01, true, Lab("surface"), "Fine matte dust on faces that look up, less in the open."),
    ctl("brush", "polishing lines", Slider, 0.0, 1.0, 0.01, true, Lab("surface"), "Fine parallel lines round the shape's vertical, as on brushed metal: highlights are drawn out across them."),
    ctl("mottle", "uneven colour", Slider, 0.0, 1.0, 0.01, true, Lab("surface"), "Clouds and veins of paler and deeper colour through a milky body (jade, alabaster); a faint unevenness of a metal's colour."),
    ctl("filmIor", "film index", Slider, 1.0, 2.6, 0.01, true, Lab("thin film"), "Index of the film: 1.33 soapy water, 1.4 to 1.5 oils, 2 and up oxide coatings."),
    ctl("filmVar", "film unevenness", Slider, 0.0, 1.0, 0.01, true, Lab("thin film"), "How much the film's thickness varies across the surface: folds of thicker and thinner film, as when a liquid is stirred."),
    ctl("filmScale", "film pattern size", Slider, 0.2, 8.0, 0.05, true, Lab("thin film"), "How fine the bands are: larger = finer."),
    ctl("filmFlow", "film flow", Slider, 0.0, 1.0, 0.005, true, Lab("thin film"), "Where the pattern is in its flow: one whole turn from 0 to 1, so animating it stirs the film and brings it back to where it began."),
    ctl("filmDrain", "film drain", Slider, -1.0, 1.0, 0.01, true, Lab("thin film"), "How far the film has drained, from fresh and even (0) to nothing but black film (1): thinner the higher it is, in level bands; thinnest on every summit, ringed by tight bands; thinning all over as it goes. Up is the lights' up. (Below 0: the same, upside down.)"),
    ctl("filmSky", "film studio", Slider, 0.0, 8.0, 0.05, true, Lab("thin film"), "The brightness of a studio that only the film mirrors: a glow from above, a large soft box overhead, a smaller one beside the camera, two tall panels behind. A film mirrors a few percent of what it faces, so without it the film shows only where a lamp is mirrored. Under an environment picture it is not used: the film mirrors the picture."),
    ctl("coatIor", "coat index", Slider, 1.2, 2.2, 0.01, true, Lab("clear coat"), "Index of the clear coat: 1.5 lacquer or glaze; higher is glossier."),
    ctl("bubbles", "bubbles", Slider, 0.0, 1.0, 0.005, false, Lab("bubbles"), "How many bubbles there are inside the body (0 = none)."),
    ctl("bubbleSize", "bubble size", Slider, 0.004, 0.2, 0.001, false, Lab("bubbles"), "Radius of the largest bubbles. Ones too large to fit inside the body are left out."),
    ctl("bubbleEven", "evenness", Slider, 0.0, 1.0, 0.01, false, Lab("bubbles"), "1: all the same size and evenly spread. 0: every size, gathered in clusters."),
    ctl("bubbleShape", "shape", Slider, -1.0, 1.0, 0.01, false, Lab("bubbles"), "0 round; towards 1 needles; towards -1 flat discs (the 'sun spangles' of amber)."),
    ctl("bubbleSeed", "arrangement", Whole, 0.0, 999.0, 1.0, false, Lab("bubbles"), "Which random arrangement."),
    ctl("glow", "glow", Slider, 0.0, 20.0, 0.05, true, Lab("glow"), "Light given off inside the body. Thicker glass glows more; the bloom spreads it."),
    ctl("glowR", "glow colour", Colour, 0.0, 1.0, 0.005, true, Lab("glow"), ""),
    ctl("glowG", "glow green", Slider, 0.0, 1.0, 0.005, true, Lab("glow"), ""),
    ctl("glowB", "glow blue", Slider, 0.0, 1.0, 0.005, true, Lab("glow"), ""),
    ctl("glowCore", "glow core", Slider, 0.0, 1.0, 0.01, true, Lab("glow"), "0: the whole body glows evenly. 1: only a thread along its middle, like the discharge in a neon tube."),
    costly(ctl("biref", "double refraction", Slider, -0.3, 0.3, 0.001, true, Lab("exotic"), "The crystal splits each ray in two: the second ray's index minus the first's (calcite -0.172, quartz 0.009, moissanite 0.055). Every ray is traced twice.")),
    ctl("birefAz", "crystal axis turn", Slider, 0.0, 360.0, 1.0, true, Lab("exotic"), "Direction of the crystal's axis round the shape's vertical, degrees."),
    ctl("birefEl", "crystal axis tilt", Slider, -90.0, 90.0, 1.0, true, Lab("exotic"), "Tilt of the crystal's axis up from the shape's horizontal, degrees (90 = straight up)."),
    costly(ctl("grin", "graded index", Slider, -0.4, 0.6, 0.005, true, Lab("exotic"), "How much the index rises (or falls) below the surface: rays curve inside the body. Marching curved rays takes several times the work inside the glass.")),
    ctl("grinDepth", "graded depth", Slider, 0.004, 0.08, 0.001, true, Lab("exotic"), "How deep below the surface the index goes on changing."),
    ctl("grad", "tint change", Slider, 0.0, 1.0, 0.01, true, Lab("tint through the body"), "How far the tint goes over to a second colour across the body (0 = one tint all through). The change is fixed to the shape."),
    ctl("gradMode", "laid out", Choice, 0.0, 2.0, 1.0, false, Lab("tint through the body"), "Along an axis, outward from the middle, or in bands about the middle (as an agate's are)."),
    ctl("gradR", "second tint", Colour, 0.0, 1.0, 0.005, true, Lab("tint through the body"), "The colour the tint changes to: what white light becomes through one unit of the body, as the first tint is."),
    ctl("gradG", "second tint green", Slider, 0.0, 1.0, 0.005, true, Lab("tint through the body"), ""),
    ctl("gradB", "second tint blue", Slider, 0.0, 1.0, 0.005, true, Lab("tint through the body"), ""),
    ctl("gradAz", "tint axis turn", Slider, 0.0, 360.0, 1.0, true, Lab("tint through the body"), "Direction of the axis the tint changes along, round the shape's vertical, degrees."),
    ctl("gradEl", "tint axis tilt", Slider, -90.0, 90.0, 1.0, true, Lab("tint through the body"), "Tilt of that axis up from the shape's horizontal, degrees (90 = from bottom to top)."),
    ctl("gradOffset", "where it changes", Slider, -1.5, 1.5, 0.01, true, Lab("tint through the body"), "Where the change lies, as a share of the shape's size: animate it to sweep the colour through the body."),
    ctl("gradSoft", "how gradual", Slider, 0.0, 2.0, 0.01, true, Lab("tint through the body"), "0: a sharp line between the two tints. Larger: a long gentle change."),
    ctl("gradBands", "bands", Slider, 1.0, 30.0, 0.1, true, Lab("tint through the body"), "How many bands there are from the middle to the outside."),
    ctl("gradWave", "band waviness", Slider, 0.0, 1.0, 0.01, true, Lab("tint through the body"), "How far the bands wander from true rings."),
    ctl("fractures", "fractures", Slider, 0.0, 1.0, 0.01, false, Lab("cracks"), "Fracture surfaces running through the whole body, as in imperfect quartz and ice (0 = none): healed veils of tiny inclusions, cracks held open in patches, rainbow cleavages, and from 0.45 a fan spreading from one point. Mostly unseen until one catches the light."),
    ctl("cracks", "cracks", Slider, 0.0, 1.0, 0.01, false, Lab("cracks"), "How many small cracks or flakes there are inside the body (0 = none)."),
    ctl("crackGrow", "crack growth", Slider, 0.0, 1.0, 0.005, true, Lab("cracks"), "How far the cracks have grown: at 0 there are none; they start one after another, spread from where they began and (when they are kept near the surface) run inwards. Animate it to watch the body crack."),
    ctl("crackSize", "crack size", Slider, 0.02, 1.0, 0.005, false, Lab("cracks"), "How large the largest cracks are: small for a fine crackle, large for a few great fracture planes."),
    ctl("crackDepth", "crack depth", Slider, 0.0, 1.0, 0.01, true, Lab("cracks"), "How far below the surface cracks reach: small for cracks that only run in from the surface, 1 for cracks all through the body."),
    ctl("crackOpen", "crack opening", Slider, 0.0, 1.0, 0.005, true, Lab("cracks"), "How wide the cracks stand open: 0 closed and unseen; around 0.4 to 0.6 a gap thinner than a thousandth of a millimetre, which shows rainbow bands; towards 1 silver mirrors."),
    ctl("crackCurve", "crack curvature", Slider, 0.0, 1.0, 0.01, true, Lab("cracks"), "0: flat faces with straight edges, as a crystal splits. 1: round shell-shaped faces, as glass breaks."),
    ctl("crackRough", "crack roughness", Slider, 0.0, 1.0, 0.01, true, Lab("cracks"), "0: clean breaks that mirror. 1: rough breaks that show as white veils, as in ice."),
    ctl("crackSeed", "crack arrangement", Whole, 0.0, 999.0, 1.0, false, Lab("cracks"), "Which random arrangement of cracks."),
    ctl("play", "play of colour", Slider, 0.0, 1.0, 0.01, true, Lab("play of colour"), "Grains inside the body that send back light (0 = none): precious opal's patches of pure colour, or the sparkling platelets of aventurine."),
    ctl("playPattern", "pattern", Choice, 0.0, 3.0, 1.0, false, Lab("play of colour"), "Blocks of colour side by side (harlequin), small points (pinfire), broad streaky fields of colour that run into each other (broad flash, as on black opal), or flat mirror platelets that sparkle (aventurine, goldstone)."),
    ctl("playSize", "grain size", Slider, 0.01, 0.6, 0.005, false, Lab("play of colour"), "How large the grains are, as a share of the shape's thickness."),
    ctl("playColour", "colour range", Slider, 0.0, 1.0, 0.01, true, Lab("play of colour"), "The longest colour the grains send back straight on: 0 only blues and violets, 1 every colour to red. Each grain shows a shorter colour the more steeply it is seen."),
    ctl("playSoft", "softness", Slider, 0.0, 1.0, 0.01, true, Lab("play of colour"), "How softened the lights the grains mirror are: low for sharp flashes, high for glowing patches."),
    ctl("playRoll", "rolling flash", Slider, 0.0, 1.0, 0.01, true, Lab("play of colour"), "How far neighbouring grains face the same way, so that colour rolls across the stone as it turns."),
    ctl("playSeed", "grain arrangement", Whole, 0.0, 999.0, 1.0, false, Lab("play of colour"), "Which arrangement of grains."),
    ctl("glintR", "platelet colour", Colour, 0.0, 1.0, 0.005, true, Lab("play of colour"), "The colour of the platelets' mirrors."),
    ctl("glintG", "platelet green", Slider, 0.0, 1.0, 0.005, true, Lab("play of colour"), ""),
    ctl("glintB", "platelet blue", Slider, 0.0, 1.0, 0.005, true, Lab("play of colour"), ""),
    ctl("silk", "silk", Slider, 0.0, 1.0, 0.01, true, Lab("silk"), "Fine fibres all lying one way (0 = none): a band of light across them that glides as the stone or the light moves, as in a cat's eye."),
    ctl("silkAz", "fibre turn", Slider, 0.0, 360.0, 1.0, true, Lab("silk"), "Direction of the fibres round the shape's vertical, degrees."),
    ctl("silkEl", "fibre tilt", Slider, -90.0, 90.0, 1.0, true, Lab("silk"), "Tilt of the fibres up from the shape's horizontal, degrees. The band of light runs across them."),
    ctl("silkSoft", "band softness", Slider, 0.0, 1.0, 0.01, true, Lab("silk"), "How soft and wide the band is."),
    ctl("silkWave", "fibre waviness", Slider, 0.0, 1.0, 0.01, true, Lab("silk"), "How wavy the fibres are: the band breaks into stripes that change places as the stone turns, as in tiger's eye."),
    ctl("piezo", "pressure glow", Slider, 0.0, 2.0, 0.01, true, Lab("glow"), "A faint glow given off under pressure (0 = none), brightest at the rims of cracks that are still growing. Animate it to pulse."),
    ctl("piezoColour", "pressure glow colour", Slider, 0.0, 1.0, 0.01, true, Lab("glow"), "0 warm white, 1 orange."),
];

/// Does this control mean anything with the settings as they stand? (The window shows the rest as not applying.)
pub fn applies(name: &str, p: &Settings) -> bool {
    let v = Variant::of(p);
    let metal = v.metal;
    match name {
        "ior" => !metal,
        "disp" => !metal && !v.hollow,
        "hollow" => !metal,
        "coat" | "bubbles" if v.hollow => false,
        "nb" => true,
        "density" | "tintR" | "tintG" | "tintB" => !metal && (name == "density" || !v.spectrum),
        "spectrum" | "sss" => !metal && !v.hollow,
        "bodyR" | "bodyG" | "bodyB" => metal || v.sss,
        "sssDepth" | "sssWalk" => v.sss,
        "sssSpread" | "sssHue" | "sssForward" | "sssThrough" => v.sss && !v.walk,
        "meter" => v.sss,
        "sssDetail" => v.sss && !v.walk,
        "filmIor" | "filmVar" | "filmScale" => v.film,
        "filmFlow" => v.film && p.film_var > 0.0,
        "filmDrain" | "filmSky" => v.film,
        "coatIor" => v.coat,
        "sharp" => metal || v.sss || v.coat,
        "brush" => metal || v.coat,
        "mottle" => metal || v.sss,
        "bubbles" => !metal,
        "bubbleSize" | "bubbleEven" | "bubbleShape" | "bubbleSeed" => !metal && v.bubbles,
        "glowR" | "glowG" | "glowB" => v.glow,
        "glowCore" => v.glow && !metal,
        "biref" => !metal && !v.hollow && !v.walk,
        "birefAz" | "birefEl" => v.biref,
        "grin" => !metal && !v.hollow && !v.walk && !v.biref,
        "grinDepth" => v.grin,
        n if crate::inner::owns(n) => crate::inner::applies(n, p, !metal && !v.hollow && !v.walk && (v.biref || p.grin == 0.0)),
        _ => true,
    }
}

/// The names a `Choice` control offers, in the order its setting counts them.
pub fn choices(name: &str) -> Vec<&'static str> {
    match name {
        "metal" => ["not a metal", "perfect mirror"].into_iter().chain(METALS.iter().map(|m| m.name)).collect(),
        "spectrum" => ["none (use the tint)"].into_iter().chain(SPECTRA.iter().map(|s| s.name)).collect(),
        "mat" => NAMED.iter().map(|n| n.name).collect(),
        "playPattern" | "gradMode" => crate::inner::choices(name),
        _ => Vec::new(),
    }
}

/// The sliders among the controls as the knob table wants them: name, label, lowest, highest, step,
/// whether it can be animated.
pub fn knobs() -> impl Iterator<Item = (&'static str, &'static str, f64, f64, f64, bool)> {
    CONTROLS.iter().filter(|c| matches!(c.kind, Slider | Colour | Whole)).map(|c| (c.name, c.label, c.lo, c.hi, c.step, c.wave))
}

// ---- a material for each body -----------------------------------------------------------------------
// A scene of several layers is made of bodies (layers that booleans hold together; a layer kept apart
// from all the others is a body by itself). The first body is made of the scene's material. Any other
// may be given a material of its own: then the scene is drawn by the tracer that follows a ray from one
// body into the next (trace_bodies.wgsl), and each body's material reaches it as numbers.

/// The most bodies a scene can have (one for each layer, when all are kept apart).
pub const MAX_BODIES: usize = 4;

/// The settings that make up a material: what a body with a material of its own keeps for itself.
/// (Everything else in the settings is the scene's: the view, the lights, the picture.)
pub fn body_names() -> impl Iterator<Item = &'static str> {
    ["mat", "ior", "disp"].into_iter().chain(CONTROLS.iter().map(|c| c.name))
}

/// A material as a body among several has it: as `Variant::of`, but with no true scattering (a milky
/// body among several is lit by its light volume, as the estimate is), so that what lies inside it, the
/// graded index and double refraction are what they would be without the wandering.
pub fn body_variant(p: &Settings) -> Variant {
    if p.sss_walk <= 0.5 { return Variant::of(p); }
    let mut q = p.clone();
    q.sss_walk = 0.0;
    Variant::of(&q)
}

/// What kind of body a material makes, to the tracer of several bodies: 0 clear, 1 a metal or a mirror,
/// 2 milky, 3 a thin wall around air.
pub fn body_kind(p: &Settings) -> f32 {
    let v = Variant::of(p);
    if v.metal { 1.0 } else if v.sss { 2.0 } else if v.hollow { 3.0 } else { 0.0 }
}

/// What else a body has, as the tracer of several bodies reads it (bits of `MB.has`, one number a
/// body): 1 colour by wavelength, 2 a graded index, 4 double refraction, 8 something inside it (inner.rs),
/// 16 a glow under pressure (given off also where it presses on another body).
pub fn body_bits(p: &Settings) -> u32 {
    let v = body_variant(p);
    (v.spectrum as u32) | (v.grin as u32) << 1 | (v.biref as u32) << 2 | (v.inner.any() as u32) << 3 | (v.inner.piezo as u32) << 4
}

/// What a body among several does not have, though its material asks for it (for the window to say so).
pub fn body_lacks(p: &Settings) -> Vec<&'static str> {
    let v = Variant::of(p);
    let mut out = Vec::new();
    if v.walk { out.push("true scattering (the light volume is used)"); }
    out
}

/// Numbers ahead of the bodies' own in the block: kind, index, dispersion and what else each has
/// (`body_bits`); then what each body's inside absorbs.
pub const MB_HEAD: usize = 4 * 4 + MAX_BODIES * 4;
/// Numbers in the block the tracer of several bodies is given (`MB` in trace_bodies.wgsl).
pub const MB_FLOATS: usize = MB_HEAD + MAX_BODIES * MU_FLOATS;

/// The numbers of the bodies' materials, in the order of the bodies. `thick`: how thick each body is,
/// as far as that has been measured (a material that asks to be fitted is fitted to its own body).
pub fn bodies_uniforms(mats: &[&Settings], thick: &[f64]) -> Vec<f32> {
    let mut u = vec![0.0f32; MB_FLOATS];
    for k in 0..MAX_BODIES {
        // (a place with no body is never read; it is filled with the first body's so that it holds numbers)
        let p = mats.get(k).copied().unwrap_or(mats[0]);
        let scale = fit_scale(p, thick.get(k).copied().filter(|t| *t > 0.0));
        let kind = body_kind(p);
        u[k] = kind;
        u[4 + k] = p.ior as f32;
        u[8 + k] = p.disp as f32;
        u[12 + k] = body_bits(p) as f32;
        // what the inside absorbs: the tint by its depth; with a colour by wavelength, that colour as
        // three numbers; nothing for a thin wall (which holds air) or a body light does not enter
        let density = p.density / scale;
        // (a milky body is traced through, as a single one is, and takes its tint on the way)
        let absorb = match (kind as u8, spectrum_of(p)) {
            (0, Some(s)) => spectrum_rgb(s, density, FIT_THICKNESS * scale),
            (0 | 2, _) => [p.absorb[0] * density, p.absorb[1] * density, p.absorb[2] * density],
            _ => [0.0; 3],
        };
        for c in 0..3 { u[16 + 4 * k + c] = absorb[c] as f32; }
        let o = MB_HEAD + k * MU_FLOATS;
        u[o..o + MU_FLOATS].copy_from_slice(&uniforms(p, scale));
    }
    u
}

/// What lies inside each body (`crate::inner::uniforms` for each, in the order of the bodies; the block
/// the tracer of several bodies reads as `ibod`). `thick`: as for `bodies_uniforms`. `pixel`: the width
/// of a pixel at the shape. Each body is told the tint the tracer already takes off along its stretches.
pub fn bodies_inner(mats: &[&Settings], thick: &[f64], pixel: f64) -> Vec<f32> {
    let n = crate::inner::IU_FLOATS;
    let mut u = vec![0.0f32; MAX_BODIES * n];
    let body = bodies_uniforms(mats, thick);
    for k in 0..MAX_BODIES {
        let p = mats.get(k).copied().unwrap_or(mats[0]);
        let scale = fit_scale(p, thick.get(k).copied().filter(|t| *t > 0.0));
        let tint = [0, 1, 2].map(|c| body[16 + 4 * k + c] as f64);
        u[k * n..(k + 1) * n].copy_from_slice(&crate::inner::uniforms(p, scale, pixel, tint));
    }
    u
}

/// The tracer's parts for a scene of several bodies with materials of their own (`v`: `Variant::of_bodies`).
/// `material_wgsl` is the same text as `tracer_parts` is given, cut with BODIES on and with the parts
/// any of the bodies uses, and with what it reads of the scene's one material made to read the numbers of
/// the body in hand (`cb` in `bodies_wgsl`): the material's uniform block (`mu.`), the body's index
/// (`u.ior`) and what its inside absorbs (`u.absorb`). `bodies_wgsl` (trace_bodies.wgsl) follows it.
pub fn bodies_parts(v: &Variant, material_wgsl: &str, bodies_wgsl: &str, picture: bool) -> TracerParts {
    let mut flags = vec![
        ("BODIES", true), ("ENVL", true), ("PICTURE", picture), ("METAL", v.metal), ("ROUGH", v.rough), ("RANDOM", v.rough),
        ("FILM", v.film), ("HOLLOW", v.hollow), ("SSS", v.sss), ("SPECTRUM", v.spectrum), ("GRIN", v.grin), ("BIREF", v.biref),
        ("WEAR", v.wear),
    ];
    // (what lies inside the bodies: each part that any of them has; each body is told its own numbers)
    flags.extend(v.inner.flags());
    let shared = per_body(&preprocess(material_wgsl, &flags));
    let decl = shared + &preprocess(bodies_wgsl, &flags).replace("{{COUNT}}", &v.bodies.clamp(2, MAX_BODIES as u8).to_string());
    // (a ray that has left one body must be able to reach the others, whatever the family allows itself)
    TracerParts { decl, shade: "shadeB(ro, rd, t, xi, W, fc)", graze: "grazeB(p, rd)", min_layers: 2 }
}

/// The material's text made to read the numbers of the body in hand: `mu.x` becomes `mb.b[cb].x`, what
/// lies inside (`iu.x`) the body's own `ibod[cb].x`, the scene's index `u.ior` the body's `mb.ior[cb]`,
/// and what the scene's inside absorbs the body's.
fn per_body(src: &str) -> String {
    let src = src.replace("u.ior", "mb.ior[cb]").replace("u.absorb.xyz", "mb.absorb[cb].rgb");
    let src = rename(&src, "mu.", "mb.b[cb].");
    rename(&src, "iu.", "ibod[cb].")
}

/// Every `from` that stands alone (is not the end of a longer name) written as `to`.
fn rename(src: &str, from: &str, to: &str) -> String {
    let mut out = String::with_capacity(src.len() + src.len() / 8);
    let mut rest = src;
    while let Some(i) = rest.find(from) {
        let alone = rest[..i].chars().next_back().is_none_or(|c| !(c.is_alphanumeric() || c == '_'));
        out.push_str(&rest[..i]);
        out.push_str(if alone { to } else { from });
        rest = &rest[i + from.len()..];
    }
    out.push_str(rest);
    out
}
