//! What lies inside a body besides the glass itself: a tint that changes through the body, cracks, the
//! grains of precious opal and the glittering platelets of aventurine, the silky fibres of tiger's eye,
//! and a glow that pressure gives off. None of it is in the distance volume: a crack is far thinner than
//! a voxel, and grains, fibres and bands are patterns, not shapes. The tracer meets all of it by
//! arithmetic along each straight stretch a ray travels inside the body (`app/src/shaders/inner.wgsl`),
//! and everything is fixed to the shape's own frame and drawn from whole-number hashes, so nothing is
//! random from frame to frame. This file says what each setting means, which parts of the tracer are
//! wanted for the settings as they stand, the numbers the tracer is told, the named materials that use
//! them, and the optics in plain arithmetic so that they can be tested without a GPU.

use crate::material::{optic_axis, scatter, spec_w};
use crate::settings::Settings;

// ---- which parts are in use ------------------------------------------------------------------------------

/// Which of the things inside the body are in use: the tracer is built for exactly these.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Parts {
    /// the tint changes through the body
    pub grad: bool,
    /// there are cracks in the body
    pub crack: bool,
    /// ... which reach only so far below the surface (the tracer then reads the depth at each)
    pub crack_depth: bool,
    /// ... and whose faces are rough (they read the lights softened)
    pub crack_rough: bool,
    /// grains that send back light: opal's play of colour
    pub play: bool,
    /// ... as flat mirror platelets instead (aventurine, goldstone)
    pub glitter: bool,
    /// fibres that make a gliding band of light (tiger's eye, cat's eye)
    pub silk: bool,
    /// a glow given off under pressure
    pub piezo: bool,
}

impl Parts {
    /// `solid`: the body has an inside that light travels through in straight lines (not a metal, not a
    /// thin wall, and not one of the ways of travelling that wander or curve).
    pub fn of(p: &Settings, solid: bool) -> Self {
        let crack = solid && p.cracks > 0.0 && p.crack_grow > 0.0 && p.crack_open > 0.0;
        let play = solid && p.play > 0.0;
        Self {
            grad: solid && p.grad > 0.0,
            crack,
            crack_depth: crack && p.crack_depth < DEPTH_ALL,
            crack_rough: crack && p.crack_rough > 0.0,
            play,
            glitter: play && p.play_pattern.round() as i64 == GLITTER,
            silk: solid && p.silk > 0.0,
            piezo: solid && p.piezo > 0.0,
        }
    }

    pub fn any(&self) -> bool {
        self.grad || self.crack || self.play || self.silk || self.piezo
    }

    /// The names the tracer's source is cut by (`//#if CRACK` and so on).
    pub fn flags(&self) -> [(&'static str, bool); 9] {
        [
            ("INNER", self.any()), ("GRAD", self.grad), ("CRACK", self.crack), ("CRACK_DEPTH", self.crack_depth), ("CRACK_ROUGH", self.crack_rough),
            ("PLAY", self.play), ("GLITTER", self.glitter), ("SILK", self.silk), ("PIEZO", self.piezo),
        ]
    }

    /// Does the tracer need the softened copies of the lights' map for these?
    pub fn soft_lights(&self) -> bool {
        self.crack_rough || self.play || self.silk
    }

    /// A short name, added to the variant's.
    pub fn key(&self) -> String {
        self.flags().iter().skip(1).filter(|f| f.1).map(|f| format!("+{}", f.0.to_lowercase())).collect()
    }

    /// Every combination worth building in a test.
    pub fn samples() -> Vec<Parts> {
        let d = Parts::default();
        let c = Parts { crack: true, ..d };
        let o = Parts { play: true, ..d };
        vec![
            Parts { grad: true, ..d },
            c,
            Parts { crack_depth: true, ..c },
            Parts { crack_rough: true, ..c },
            Parts { crack_depth: true, crack_rough: true, piezo: true, ..c },
            o,
            Parts { glitter: true, ..o },
            Parts { silk: true, ..d },
            Parts { piezo: true, ..d },
            Parts { grad: true, crack: true, crack_depth: true, crack_rough: true, play: true, silk: true, piezo: true, glitter: false },
            Parts { grad: true, crack: true, play: true, glitter: true, silk: true, piezo: true, ..d },
        ]
    }
}

// ---- the settings ----------------------------------------------------------------------------------------

/// Every setting of this file, with what choosing a named material puts it back to first.
pub const RESET: &[(&str, f64)] = &[
    ("grad", 0.0), ("gradMode", 0.0), ("gradR", 0.55), ("gradG", 0.85), ("gradB", 1.0), ("gradAz", 0.0), ("gradEl", 90.0),
    ("gradOffset", 0.0), ("gradSoft", 0.6), ("gradBands", 6.0), ("gradWave", 0.3),
    ("cracks", 0.0), ("crackSize", 0.25), ("crackDepth", 1.0), ("crackOpen", 0.8), ("crackGrow", 1.0), ("crackCurve", 0.5), ("crackRough", 0.0), ("crackSeed", 1.0),
    ("play", 0.0), ("playSize", 0.12), ("playPattern", 0.0), ("playColour", 0.8), ("playSoft", 0.3), ("playRoll", 0.0), ("playSeed", 1.0),
    ("glintR", 1.0), ("glintG", 0.85), ("glintB", 0.6),
    ("silk", 0.0), ("silkAz", 0.0), ("silkEl", 0.0), ("silkSoft", 0.3), ("silkWave", 0.0),
    ("piezo", 0.0), ("piezoColour", 0.5),
];

/// Is this one of this file's settings?
pub fn owns(name: &str) -> bool {
    RESET.iter().any(|(n, _)| *n == name)
}

/// Does a control of this file mean anything with the settings as they stand? `solid`: as for `Parts::of`.
pub fn applies(name: &str, p: &Settings, solid: bool) -> bool {
    let i = Parts::of(p, solid);
    match name {
        "grad" | "cracks" | "play" | "silk" | "piezo" => solid,
        "crackGrow" => solid && p.cracks > 0.0,
        "gradBands" | "gradWave" => i.grad && p.grad_mode.round() as i64 == 2,
        "gradAz" | "gradEl" => i.grad && p.grad_mode.round() as i64 == 0,
        n if n.starts_with("grad") => i.grad,
        "crackRough" | "crackSize" | "crackDepth" | "crackOpen" | "crackCurve" | "crackSeed" => solid && p.cracks > 0.0,
        "playColour" => i.play && !i.glitter,
        "glintR" | "glintG" | "glintB" => i.glitter,
        n if n.starts_with("play") => i.play,
        n if n.starts_with("silk") => i.silk,
        "piezoColour" => i.piezo,
        _ => true,
    }
}

/// The names `playPattern` offers, and the ways `gradMode` lays the tint out.
pub fn choices(name: &str) -> Vec<&'static str> {
    match name {
        "playPattern" => vec!["blocks (harlequin)", "pinfire", "broad flash", "glitter (platelets)"],
        "gradMode" => vec!["along an axis", "outward from the centre", "bands (agate)"],
        _ => Vec::new(),
    }
}

// ---- what the tracer is told -----------------------------------------------------------------------------

/// Bands the visible spectrum is divided into when a crack's colours are worked out.
pub const BANDS: usize = 8;
/// Rows of four numbers in the block the tracer reads (`IU` in inner.wgsl).
pub const IU_ROWS: usize = 3 + BANDS + 12;
pub const IU_FLOATS: usize = IU_ROWS * 4;

/// `crackDepth` at or above this means cracks everywhere in the body (the depth is then not read at all).
pub const DEPTH_ALL: f64 = 0.999;
/// The largest facet's radius as a share of its cell: a facet lies wholly inside its cell, so a ray
/// crossing the cell need look at no other.
pub const FACET_IN_CELL: f64 = 0.46;
/// How often cracks may turn one path before it is taken to leave the way it is heading.
pub const TURNS: f64 = 6.0;
/// The thickness of shape that the settings' distances are written for (as the materials' are).
pub const WRITTEN_FOR: f64 = 0.3;
/// The radius the shapes are scaled to: positions in the body are measured against it.
pub const SHAPE: f64 = crate::FIT;
/// `playPattern`'s value for platelets.
pub const GLITTER: i64 = 3;
/// How strongly fibres scatter light per thickness of shape, at `silk` 1.
pub const SILK_K: f64 = 9.0;
/// The glow given off at `piezo` 1, per thickness of shape, through the body and at a crack's rim.
pub const PIEZO_BODY: f64 = 0.06;
pub const PIEZO_RIM: f64 = 1.2;
/// The width of the band of wavelengths an opal's grain sends back, nm, and how much more than a band's
/// share of white it is given (one band alone is a purer colour than a screen can show and dim beside
/// the body).
pub const OPAL_BAND_NM: f64 = 30.0;
pub const OPAL_GAIN: f64 = 1.6;

/// The air gap at a crack's widest point, in nanometres, for `crackOpen` 0 to 1: closed (the crack
/// cannot be seen), a quarter of a micron near 0.4 (the first bright colours), a few microns at 0.75
/// (silver with a coloured rim), 15 microns at 1.
pub fn gap_nm(open: f64) -> f64 {
    30.0 * ((open.clamp(0.0, 1.0) * 6.2).exp() - 1.0)
}

/// For each band of the spectrum: the share of white light's red, green and blue that lies in it, and
/// the wavelength at its middle. The shares of each colour add up to one over the bands.
pub fn band_weights() -> [[f64; 4]; BANDS] {
    let width = 300.0 / BANDS as f64;
    let mut out = [[0.0; 4]; BANDS];
    let mut total = [0.0; 3];
    for (j, band) in out.iter_mut().enumerate() {
        for k in 0..50 {
            let w = spec_w(400.0 + width * (j as f64 + (k as f64 + 0.5) / 50.0));
            for c in 0..3 { band[c] += w[c]; total[c] += w[c]; }
        }
        band[3] = 400.0 + width * (j as f64 + 0.5);
    }
    for band in out.iter_mut() {
        for c in 0..3 { band[c] /= total[c]; }
    }
    out
}

/// White light as the tracer's `specW` adds it up, one nanometre at a time from 400 to 700.
pub fn white_per_nm() -> [f64; 3] {
    let mut w = [0.0; 3];
    for i in 0..300 {
        let s = spec_w(400.5 + i as f64);
        for c in 0..3 { w[c] += s[c]; }
    }
    w
}

/// The longest wavelength an opal's grains send back (straight on to their layers), in nanometres, for
/// `playColour` 0 to 1. The layers of silica spheres in a grain send back one wavelength, twice their
/// spacing times the index, and shorter ones the more steeply they are met: so small spheres (0) give
/// only violet and blue, and the largest (1) run through every colour to red as the stone turns.
pub fn opal_longest_nm(colour: f64) -> f64 {
    470.0 + 290.0 * colour.clamp(0.0, 1.0)
}

/// What a grain of each pattern is: its size as a multiple of `playSize`, the share of grains that send
/// back light, and the least `playRoll` (how far neighbouring grains share a direction).
pub fn pattern(p: &Settings) -> (f64, f64, f64) {
    match p.play_pattern.round() as i64 {
        1 => (0.35, 0.45, 0.0),
        2 => (2.6, 1.0, 0.55),
        3 => (0.6, 0.3, 0.0),
        _ => (1.0, 0.92, 0.0),
    }
}

/// The colour that is the second tint, said as absorption per unit length (as `absorb` is the first).
pub fn grad_absorb(p: &Settings) -> [f64; 3] {
    [p.grad_r, p.grad_g, p.grad_b].map(|c| -c.clamp(1e-4, 1.0).ln())
}

/// The numbers the tracer is told. `scale`: what the material's distances are multiplied by to fit the
/// shape (1 unless `fit` is on). `pixel`: the width of one pixel at the shape's distance, in the
/// shape's units (colour bands and grains finer than that are averaged). `tint`: the tint the tracer
/// already takes off along every stretch (`u.absorb`).
pub fn uniforms(p: &Settings, scale: f64, pixel: f64, tint: [f64; 3]) -> [f32; IU_FLOATS] {
    let mut u = [0f32; IU_FLOATS];
    let mut put = |row: usize, v: [f64; 4]| { for i in 0..4 { u[row * 4 + i] = v[i] as f32; } };
    let unit = WRITTEN_FOR * scale;

    // ---- cracks
    // a facet's radius (`crackSize` is in thicknesses of the shape), and from it the cell it must fit into
    let radius = p.crack_size.clamp(0.005, 3.0) * unit;
    let cells = FACET_IN_CELL / radius;
    put(0, [cells, p.cracks.clamp(0.0, 1.0), FACET_IN_CELL, p.crack_grow.clamp(0.0, 1.0)]);
    let depth = if p.crack_depth < DEPTH_ALL { p.crack_depth.max(0.0) * unit } else { 1e9 };
    put(1, [gap_nm(p.crack_open), p.crack_curve.clamp(0.0, 1.0), depth, p.crack_seed.round().rem_euclid(4096.0)]);
    // rough faces: how far what they send back is spread (radians), how much of a white veil they are
    // instead of a mirror; then the pixel's width in cells; then the turns allowed
    let rough = p.crack_rough.clamp(0.0, 1.0);
    put(2, [0.15 + 0.85 * rough, rough, pixel * cells, TURNS]);
    for (j, b) in band_weights().iter().enumerate() { put(3 + j, *b); }
    let r = 3 + BANDS;

    // ---- play of colour, or platelets
    let (size, cover, roll0) = pattern(p);
    let grain = p.play_size.clamp(0.005, 3.0) * unit * size;
    let glitter = p.play_pattern.round() as i64 == GLITTER;
    put(r, [1.0 / grain, cover, p.play_seed.round().rem_euclid(4096.0), if glitter { 1.0 } else { 0.0 }]);
    // grains finer than a pixel cannot be told apart: their play is faded out as they get there
    // (a platelet is a point of light, and may be smaller than a pixel before it fades)
    let seen = if glitter { (grain / pixel.max(1e-9) - 0.3).clamp(0.0, 1.0) } else { ((grain / pixel.max(1e-9) - 1.0) / 2.0).clamp(0.0, 1.0) };
    let roll = roll0 + (1.0 - roll0) * p.play_roll.clamp(0.0, 1.0);
    put(r + 1, [0.5 * grain, p.play.clamp(0.0, 1.0) * seen, roll, OPAL_GAIN]);
    let soft = if glitter { 0.02 + 0.3 * p.play_soft.clamp(0.0, 1.0) } else { 0.05 + 0.6 * p.play_soft.clamp(0.0, 1.0) };
    put(r + 2, [opal_longest_nm(p.play_colour), soft, 0.0, 0.0]);
    // what the body takes from light on its way to a grain and back, per unit of length: the body's
    // scattering and absorption (as the milky body has them) and the tint, both ways
    let sc = scatter([p.body_r, p.body_g, p.body_b], p.sss_depth * scale, p.sss);
    let lost = [0, 1, 2].map(|i| 2.0 * (if p.sss > 0.0 { sc.sigma_s + sc.sigma_a[i] } else { 0.0 } + tint[i]));
    put(r + 3, [lost[0], lost[1], lost[2], 0.0]);
    put(r + 4, [p.glint_r.clamp(0.0, 1.0), p.glint_g.clamp(0.0, 1.0), p.glint_b.clamp(0.0, 1.0), 0.32]);
    let w = white_per_nm();
    put(r + 5, [1.0 / w[0], 1.0 / w[1], 1.0 / w[2], OPAL_BAND_NM]);

    // ---- the tint that changes through the body: where it lies, and the second tint less the first
    let mode = p.grad_mode.round().clamp(0.0, 2.0);
    let a = optic_axis(p.grad_az, p.grad_el);
    put(r + 6, [a[0], a[1], a[2], mode]);
    let b = grad_absorb(p);
    let amount = p.grad.clamp(0.0, 1.0);
    let k = p.density / scale;
    // (a milky body shows the tint of the place its light is gathered from, over one thickness of shape)
    let milky = if p.sss > 0.0 { unit } else { 0.0 };
    put(r + 7, [(b[0] * k - tint[0]) * amount, (b[1] * k - tint[1]) * amount, (b[2] * k - tint[2]) * amount, milky]);
    put(r + 8, [p.grad_offset.clamp(-2.0, 2.0), 0.02 + p.grad_soft.clamp(0.0, 2.0), p.grad_bands.clamp(0.5, 40.0), p.grad_wave.clamp(0.0, 1.0)]);

    // ---- silky fibres: their direction, how strongly they scatter, how much what they send back is
    // softened, how wavy they are
    let s = optic_axis(p.silk_az, p.silk_el);
    put(r + 9, [s[0], s[1], s[2], p.silk.clamp(0.0, 1.0) * SILK_K / unit]);
    put(r + 10, [0.22 + 0.6 * p.silk_soft.clamp(0.0, 1.0), p.silk_wave.clamp(0.0, 1.0), 0.0, 0.0]);

    // ---- the glow under pressure: faint warm white to orange; how much the body gives off per unit of
    // length, and how much a crack's growing rim gives off where a ray crosses it
    let c = p.piezo_colour.clamp(0.0, 1.0);
    let col = [1.0, 0.82 - 0.4 * c, 0.6 - 0.5 * c];
    let g = p.piezo.max(0.0);
    let body = g * PIEZO_BODY / unit;
    put(r + 11, [col[0] * body, col[1] * body, col[2] * body, g * PIEZO_RIM]);
    u
}

// ---- the optics, in plain arithmetic ---------------------------------------------------------------------

/// What an air gap `gap` nanometres wide between two faces of a body of index `n` sends back of light
/// of one wavelength (nm) arriving at cosine `cosi` to the faces: both polarisations averaged, every
/// reflection back and forth across the gap counted, and past the critical angle the light that leaks
/// across a gap thinner than its wavelength. Exact; the tracer's `crackR` is this, averaged over a band
/// of wavelengths and over a pixel.
pub fn gap_reflectance(cosi: f64, lam: f64, gap: f64, n: f64) -> f64 {
    let c = cosi.clamp(1e-6, 1.0);
    // cosine squared of the angle in the gap: negative past the critical angle
    let cg2 = 1.0 - n * n * (1.0 - c * c);
    let d = 2.0 * std::f64::consts::PI * gap / lam;
    let y = cg2 * d * d;
    // (sin x / x)^2 below the critical angle, (sinh x / x)^2 past it: one smooth function of y
    let k = d * d * if y > 1e-12 {
        let x = y.sqrt();
        (x.sin() / x).powi(2)
    } else if y < -1e-12 {
        let x = (-y).sqrt().min(300.0);
        (x.sinh() / x).powi(2)
    } else {
        1.0
    };
    let one = |p: f64| {
        let g = (p * p - cg2).powi(2) / (4.0 * p * p);
        g * k / (1.0 + g * k)
    };
    0.5 * (one(n * c) + one(c / n))
}

/// The same as a colour (linear RGB, white = 1), for a gap that is the same across the whole pixel.
pub fn gap_colour(cosi: f64, gap: f64, n: f64) -> [f64; 3] {
    crate::material::colour_of(|l| gap_reflectance(cosi, l, gap, n))
}

/// The wavelength (nm) a grain of opal sends back when light meets its layers at cosine `cosa` to their
/// normal: Bragg's law, `longest` being the wavelength sent back straight on.
pub fn opal_wavelength(longest: f64, cosa: f64) -> f64 {
    longest * cosa.abs()
}

/// How far the tint has gone over to the second colour at `x` (the place's position along the axis, or
/// its distance outward, as a share of the shape's radius): the tracer's `gradF` for the first two modes.
pub fn grad_share(x: f64, offset: f64, soft: f64) -> f64 {
    let w = 0.02 + soft;
    let t = ((x - offset + w) / (2.0 * w)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::fresnel;

    #[test]
    fn a_closed_crack_cannot_be_seen() {
        for c in [1.0, 0.9, 0.7, 0.5, 0.2] {
            for l in [400.0, 550.0, 700.0] {
                assert!(gap_reflectance(c, l, 0.0, 1.5) < 1e-12);
                // and a gap of a nanometre reflects next to nothing, even past the critical angle (short
                // of grazing, where every surface reflects)
                if c >= 0.5 { assert!(gap_reflectance(c, l, 1.0, 1.5) < 2e-3, "{c} {l}"); }
            }
        }
    }

    #[test]
    fn a_wide_crack_past_the_critical_angle_is_a_mirror() {
        // the critical angle of glass of index 1.5 is 41.8 degrees: cosine 0.745
        for c in [0.74, 0.6, 0.3, 0.05] {
            assert!(gap_reflectance(c, 550.0, 5000.0, 1.5) > 0.999999, "{c}");
        }
        // ... and a gap far thinner than the wavelength lets light leak across
        let leak = gap_reflectance(0.6, 550.0, 60.0, 1.5);
        assert!(leak > 0.05 && leak < 0.6, "{leak}");
        assert!(gap_reflectance(0.6, 550.0, 30.0, 1.5) < leak);
    }

    #[test]
    fn interference_is_the_textbook_airy_sum() {
        // straight on, glass 1.5: one face reflects 4%; the gap's reflectance swings between nothing
        // (gap = whole half-waves) and 4R / (1 + R)^2 (odd quarter-waves)
        let r = fresnel(1.0, 1.0, 1.5);
        let top = 4.0 * r / (1.0 + r).powi(2);
        assert!((gap_reflectance(1.0, 500.0, 125.0, 1.5) - top).abs() < 1e-9);
        assert!(gap_reflectance(1.0, 500.0, 250.0, 1.5) < 1e-9);
        assert!((gap_reflectance(1.0, 500.0, 375.0, 1.5) - top).abs() < 1e-9);
        // averaged over many fringes it is the two faces as plain surfaces: 2R / (1 + R)
        let mean: f64 = (0..2000).map(|i| gap_reflectance(1.0, 500.0, 20000.0 + i as f64 * 0.25, 1.5)).sum::<f64>() / 2000.0;
        assert!((mean - 2.0 * r / (1.0 + r)).abs() < 1e-4, "{mean}");
    }

    #[test]
    fn nothing_is_made_and_nothing_jumps() {
        let mut last = 0.0;
        for i in 0..=2000 {
            let c = 0.9 - 0.4 * i as f64 / 2000.0;
            let r = gap_reflectance(c, 550.0, 550.0, 1.5);
            assert!((0.0..=1.0).contains(&r));
            if i > 0 { assert!((r - last).abs() < 0.02, "a jump at cosine {c}: {last} to {r}"); }
            last = r;
        }
    }

    #[test]
    fn the_bands_add_up_to_white() {
        let b = band_weights();
        for c in 0..3 {
            let s: f64 = b.iter().map(|x| x[c]).sum();
            assert!((s - 1.0).abs() < 1e-12);
        }
        assert_eq!(b[0][3], 418.75);
        assert_eq!(b[7][3], 681.25);
    }

    #[test]
    fn an_opal_grain_is_a_colour_not_more_than_white() {
        // a grain's band, widened as the tracer widens it, is never more than all of a colour
        let w = white_per_nm();
        let mut most: f64 = 0.0;
        for l in 400..=700 {
            let s = spec_w(l as f64);
            for c in 0..3 { most = most.max(s[c] * OPAL_BAND_NM / w[c] * OPAL_GAIN); }
        }
        assert!(most < 1.0, "{most}");
    }

    #[test]
    fn nothing_inside_asks_nothing_of_the_tracer() {
        let p = Settings::default();
        assert!(!Parts::of(&p, true).any());
        let mut q = p.clone();
        q.cracks = 0.5;
        assert!(Parts::of(&q, true).crack && !Parts::of(&q, false).any());
        // a crack that has not begun to grow is no crack
        q.crack_grow = 0.0;
        assert!(!Parts::of(&q, true).any());
        for (name, flag) in [("grad", "GRAD"), ("play", "PLAY"), ("silk", "SILK"), ("piezo", "PIEZO")] {
            let mut o = p.clone();
            o.set(name, 0.5);
            let parts = Parts::of(&o, true);
            assert!(parts.any() && !Parts::of(&o, false).any());
            assert!(parts.flags().iter().any(|f| f.0 == flag && f.1), "{name}");
        }
        let mut g = p.clone();
        g.play = 0.5;
        g.play_pattern = GLITTER as f64;
        assert!(Parts::of(&g, true).glitter);
    }

    #[test]
    fn every_setting_exists_and_resets_to_its_default() {
        let d = Settings::default();
        for (name, v) in RESET {
            assert_eq!(d.get(name), Some(*v), "{name}");
        }
    }

    #[test]
    fn the_gap_opens_from_nothing() {
        assert_eq!(gap_nm(0.0), 0.0);
        assert!(gap_nm(0.4) > 250.0 && gap_nm(0.4) < 400.0);
        assert!(gap_nm(1.0) > 14000.0);
    }

    #[test]
    fn an_opal_runs_from_red_to_violet_as_it_turns() {
        let l = opal_longest_nm(1.0);
        assert!(opal_wavelength(l, 1.0) > 700.0);
        assert!((opal_wavelength(l, 0.75) - 570.0).abs() < 1.0);
        assert!(opal_wavelength(l, 0.53) < 410.0);
        assert!(opal_wavelength(opal_longest_nm(0.0), 1.0) < 480.0);
    }

    #[test]
    fn the_tint_goes_over_smoothly() {
        assert_eq!(grad_share(-2.0, 0.0, 0.5), 0.0);
        assert_eq!(grad_share(2.0, 0.0, 0.5), 1.0);
        assert!((grad_share(0.0, 0.0, 0.5) - 0.5).abs() < 1e-12);
        let mut last = 0.0;
        for i in 0..=400 {
            let s = grad_share(-1.0 + i as f64 / 200.0, 0.1, 0.3);
            assert!(s >= last);
            last = s;
        }
    }

    #[test]
    fn the_numbers_fit_the_block() {
        let mut p = Settings::default();
        for (k, _) in RESET { p.set(k, 0.5); }
        let u = uniforms(&p, 1.0, 0.002, [0.1, 0.1, 0.1]);
        assert!(u.iter().all(|x| x.is_finite()));
    }
}
