//! Hard limits: how far every setting may go, whatever its slider's range says.
//!
//! A slider's range (its soft range) is where a setting is usually useful. Held with Alt, a slider goes
//! past it, and a value typed in may be anything; what stops either from reaching a value the program
//! cannot take is here. Each setting's hard limits are what the code that reads it can bear: no division
//! by nothing, no count that would ask for absurd memory or a frame so slow that the graphics device is
//! lost, no index outside its list, no angle at which a tangent runs off to infinity, no colour that adds
//! light where it should only take it away. They are applied where a setting is set (`Settings::set`,
//! `camera::set`, a setup read from a file), so the command line, a saved setup and a slider are held
//! alike; the reasons are given beside each.
//!
//! Knobs that are not settings (a shape's own numbers, a layer's placement, a lamp's, the studio's
//! layers) are held by `widen`: their soft range made wider, never below zero for what cannot be negative.

use crate::state::{Knob, KnobRef};

/// Anything finite not otherwise limited: far beyond any use, near enough to keep sums finite.
pub const FREE: (f64, f64) = (-1e6, 1e6);
/// An angle in degrees (wraps round; only kept finite).
const ANGLE: (f64, f64) = (-36000.0, 36000.0);
/// A switch.
const SWITCH: (f64, f64) = (0.0, 1.0);
/// A share, a mix or an amount that means nothing above all of it.
const SHARE: (f64, f64) = (0.0, 1.0);

/// The hard limits of a setting, by its name (`Settings::NAMES`, the camera's other names, the tint).
/// None for a name that is not a setting.
pub fn setting(name: &str) -> Option<(f64, f64)> {
    let named = crate::material::NAMED.len() as f64;
    let metals = crate::material::METALS.len() as f64;
    let eases = crate::anim::EASES.len() as f64;
    Some(match name {
        // ---- the glass and the picture
        // (below 1 the inside would be traced as thinner than air, and at 0 the ratio of indices divides
        // by nothing; nothing real refracts more than 4)
        "ior" | "filmIor" | "coatIor" => (1.0, 4.0),
        "disp" => (0.0, 1.0),
        // bounces, rays and wavelengths are passes of the tracer every frame: beyond these a frame at 4K
        // takes long enough for the device to be lost
        "nb" => (0.0, 32.0),
        "ns" => (1.0, 32.0),
        "nw" => (1.0, 32.0),
        "heroSettle" => (0.0, 1.0),
        "lensN" => (1.0, 64.0),
        "exposure" => (0.0, 1000.0),
        "bloom" => (0.0, 10.0),
        "bloomKind" | "dof" | "planes" | "lock" | "debug" | "raw" | "clay" | "hdriShow" | "hdriLights" | "envRoom" | "lights" | "spinMode" => SWITCH,
        "vignette" => (0.0, 4.0),
        "grain" => (0.0, 1.0),
        "spread" => (0.0, 50.0),
        // (at 1 the old picture would be kept for ever and the new never seen)
        "smooth" => (0.0, 0.99),
        // render scale: the traced picture's buffers grow with its square, and twice a 4K window is all
        // the memory takes
        "scale" => (0.1, 2.0),
        "denoise" => (0.0, 2.0),
        // ---- the camera
        // (at 0 the camera is inside the shape's middle; it may come close, never through)
        "dist" => (0.05, 10000.0),
        "eyeX" | "eyeY" => (-10000.0, 10000.0),
        "pivotX" | "pivotY" | "pivotZ" => (-1000.0, 1000.0),
        "pivotR" => (0.0, 1000.0),
        // (the tangent of half the field of view runs to infinity at 180)
        "fov" => (0.1, 175.0),
        "roll" | "lyaw" | "lpitch" | "lroll" | "bokehRot" | "birefAz" | "birefEl" | "gradAz" | "gradEl" | "silkAz" | "silkEl" | "azim" | "backGlowAz" => ANGLE,
        "spin" | "lspin" => (-50.0, 50.0),
        // (0 would stop the clock, and a trip's period is divided by it)
        "speed" => (0.01, 50.0),
        "ease" => (0.0, eases - 1.0),
        "easeAt" => (0.0, 2.0),
        "loop" => (0.0, 36000.0),
        // (the lens's radius in the scene's units: an f-stop on a small subject gives a wide one, and a wide
        // one only blurs more)
        "aperture" => (0.0, 10000.0),
        "blades" => (0.0, 16.0),
        "fnear" | "ffar" | "focus" => (-1000.0, 1000.0),
        "band" => (0.0, 1000.0),
        "focusDist" => (0.0, 1e9),
        // a focal length of 0 means "not said" (the lens is given by `fov`); a shorter one than 0.5 mm on
        // a 24 mm sensor would be wider than the field of view can be
        "focal" => (0.0, 20000.0),
        "zoom" => (-6.0, 8.6),
        "sensor" => (1.0, 500.0),
        // an f-stop of 0 means "not said"; any other gives the opening, which `aperture` keeps in bounds
        "fstop" => (0.0, 1024.0),
        "av" => (-4.0, 20.0),
        "subject" => (0.01, 1e9),
        "subjectLog" => (-2.0, 9.0),
        "proj" => (0.0, 6.0),
        // (a picture bowed by all of its size folds over on itself)
        "distort" => (-0.9, 0.9),
        "fringe" => (0.0, 4.0),
        "falloff" | "catseye" => SHARE,
        // (out-of-focus highlights' height over their width; the blur is divided by it)
        "squeeze" => (0.1, 10.0),
        "bokeh" => (-1.0, 1.0),
        // (the tangent of a tilt of 90 degrees is infinite)
        "tiltX" | "tiltY" => (-85.0, 85.0),
        "shiftX" | "shiftY" => (-100.0, 100.0),
        "ev" => (-10.0, 10.0),
        // ---- the lights and what is cast
        "lightGain" | "hdriGain" => (0.0, 1000.0),
        "hdriBlur" => SHARE,
        "cast" => (0.0, 2.0),
        "castSoft" => (0.0, 4.0),
        "castShadows" => (0.0, 4.0),
        "castCaustics" => (0.0, 100.0),
        // (each step of detail doubles what is cast into: 3 is the largest the program makes)
        "castDetail" => (0.0, 3.0),
        "castShadow" | "castCaustic" | "castShadow2" | "castCaustic2" | "castShadow3" | "castCaustic3" | "castShadow4" | "castCaustic4" => (0.0, 2.0),
        // ---- the material
        "mat" => (0.0, named - 1.0),
        "density" => (0.0, 10000.0),
        "fit" | "roughRef" | "sharp" | "hollow" | "sssWalk" | "ceramic" | "meter" => SWITCH,
        // (a number past the list means none, which the material reads as it should)
        "spectrum" => (0.0, 1000.0),
        "metal" => (0.0, metals + 1.0),
        // (roughness is the spread of the surface's facets: past 1 the model has nothing to say)
        "rough" => SHARE,
        "age" => (0.0, 20.0),
        "edgeWear" | "cavity" | "scratches" | "swirls" | "peel" | "smudges" | "dust" | "brush" | "mottle" => (0.0, 5.0),
        // colours that a surface or a body gives back: above 1 they would give back more light than falls on
        // them, and a milky body's wandering light would grow without end
        "cavityR" | "cavityG" | "cavityB" | "bodyR" | "bodyG" | "bodyB" | "floorR" | "floorG" | "floorB" | "gradR" | "gradG" | "gradB" => SHARE,
        // (the tint is the colour after one unit of travel; at 0 it would take everything away at once,
        // which `Settings` keeps it just short of)
        "tintR" | "tintG" | "tintB" => SHARE,
        "film" => (0.0, 10000.0),
        "filmVar" | "filmDrain" => (-5.0, 5.0),
        // (the film's pattern is scaled by its size)
        "filmScale" => (0.01, 1000.0),
        "filmFlow" | "playRoll" | "crackSeed" | "playSeed" | "bubbleSeed" => FREE,
        "filmSky" => (0.0, 100.0),
        "coat" => SHARE,
        "sss" => SHARE,
        // (how far light travels between scatterings: the scattering is divided by it)
        "sssDepth" => (0.001, 100.0),
        "sssSpread" | "sssHue" | "sssThrough" | "ceramicThrough" => SHARE,
        // (the scattering's mean cosine: at 1 or -1 the phase function is a spike)
        "sssForward" => (-0.95, 0.95),
        "glazeHaze" => (0.0, 1.5),
        // cells along a side of the light volume of a milky body: its memory grows with the cube
        "sssDetail" => (64.0, 256.0),
        "bubbles" => SHARE,
        // (bubbles are packed in cells of their own size: smaller ones would be too many to find)
        "bubbleSize" => (0.004, 0.25),
        "bubbleEven" => SHARE,
        "bubbleShape" => (-1.0, 1.0),
        "glow" => (0.0, 10000.0),
        "glowR" | "glowG" | "glowB" | "glintR" | "glintG" | "glintB" => (0.0, 100.0),
        "glowCore" => SHARE,
        "grin" => (-2.0, 2.0),
        // (the depth over which the index changes: the change is divided by it)
        "grinDepth" => (0.0005, 10.0),
        "biref" => (-1.0, 1.0),
        "grad" => SHARE,
        "gradMode" => (0.0, 2.0),
        "gradOffset" => (-100.0, 100.0),
        "gradSoft" => (0.0, 100.0),
        "gradBands" => (0.0, 500.0),
        "gradWave" => (0.0, 10.0),
        "fractures" | "cracks" | "crackGrow" | "crackDepth" | "crackOpen" | "crackRough" => SHARE,
        "crackCurve" => (0.0, 2.0),
        // the cells a ray walks through to find cracks are this large: much smaller and a ray has too many
        // to walk in one frame
        "crackSize" => (0.01, 10.0),
        "play" => (0.0, 4.0),
        "playPattern" => (0.0, 3.0),
        // (the grains' size: the same walk of cells)
        "playSize" => (0.005, 10.0),
        "playColour" | "playSoft" => SHARE,
        "silk" => (0.0, 4.0),
        "silkSoft" => SHARE,
        "silkWave" => (0.0, 10.0),
        "piezo" => (0.0, 100.0),
        "piezoColour" => SHARE,
        // ---- the studio
        // (the camera on its arc: straight down, as its slider allows, or straight up, no further)
        "elev" => (-90.0, 90.0),
        "backdrop" | "backShow" | "floor" | "floorAuto" | "floorMirror" => SWITCH,
        "backGain" | "backGlow" | "floorGain" => (0.0, 1000.0),
        // (degrees the horizon's colour is blended over: it is divided by them)
        "backSpread" => (0.1, 180.0),
        "backWrap" => (0.0, 180.0),
        "backHigh" => (0.0, 90.0),
        "backTopR" | "backTopG" | "backTopB" | "backMidR" | "backMidG" | "backMidB" | "backLowR" | "backLowG" | "backLowB" | "backGlowR" | "backGlowG" | "backGlowB" => (0.0, 100.0),
        "backGlowEl" => (-90.0, 90.0),
        "backGlowSize" => (0.1, 360.0),
        "floorY" => (-1000.0, 1000.0),
        "floorMatch" | "floorGloss" | "floorBlur" => SHARE,
        // (how far the floor reaches before it melts into the backdrop: the fade is divided by it)
        "floorFade" => (0.1, 10000.0),
        "tone" => (0.0, 3.0),
        "wbTemp" | "wbTint" => (-4.0, 4.0),
        "contrast" | "saturation" => (0.0, 10.0),
        _ => return None,
    })
}

// The numbers every tube has (`ShapeState`): its thickness (a factor), how close strands melt together, how
// much of the orbit is drawn (the orbit's length grows with it), the beads' depth (at 1 a bead pinches the
// tube to nothing), and how far a shape is inflated (scene units).
pub const THICK: (f64, f64) = (0.05, 10.0);
pub const MELT: (f64, f64) = (0.0, 0.1);
pub const LEN: (f64, f64) = (0.02, 10.0);
pub const BEADS: (f64, f64) = (0.0, 0.95);
pub const PUFF: (f64, f64) = (-0.3, 0.3);

/// A soft range made wider, for a knob with no hard limits of its own: twice its width past either end,
/// never below zero for a knob whose range starts at zero or above (a count, a size, an amount).
pub fn widen(lo: f64, hi: f64) -> (f64, f64) {
    let span = (hi - lo).abs().max(1e-9);
    let a = if lo >= 0.0 { 0.0 } else { lo - 2.0 * span };
    (a.max(FREE.0), (hi + 2.0 * span).min(FREE.1))
}

/// The hard limits of a knob: its setting's, if it is one, else its soft range widened. Never narrower
/// than the soft range itself.
pub fn knob(k: &Knob) -> (f64, f64) {
    let r = match &k.r {
        KnobRef::Of(_, r) => r.as_ref(),
        r => r,
    };
    let own = match r {
        KnobRef::P(name) | KnobRef::Body(_, name) => setting(name),
        KnobRef::Thick => Some(THICK),
        KnobRef::Melt => Some(MELT),
        KnobRef::Len => Some(LEN),
        KnobRef::Beads => Some(BEADS),
        KnobRef::Puff => Some(PUFF),
        _ => None,
    };
    let (a, b) = own.unwrap_or_else(|| widen(k.lo, k.hi));
    (a.min(k.lo), b.max(k.hi))
}

/// A value held to limits: anything not a number (NaN) goes to the nearest end of nothing, the lower.
pub fn hold(v: f64, (lo, hi): (f64, f64)) -> f64 {
    if v.is_nan() { lo } else { v.clamp(lo, hi) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::Settings;

    #[test]
    fn every_setting_has_hard_limits_that_hold_its_default() {
        let p = Settings::default();
        for n in Settings::NAMES.iter().chain(Settings::DERIVED) {
            let (lo, hi) = setting(n).unwrap_or_else(|| panic!("{n} has no hard limits"));
            assert!(lo < hi, "{n}");
            let v = p.get(n).unwrap();
            assert!(v >= lo && v <= hi, "{n}: its default {v} is outside {lo}..{hi}");
        }
    }

    #[test]
    fn every_slider_range_lies_inside_its_hard_limits() {
        let mut wrong = Vec::new();
        for k in crate::state::State::engine_knobs() {
            if let KnobRef::P(name) = k.r {
                match setting(name) {
                    None => wrong.push(format!("{name}: no hard limits")),
                    Some((lo, hi)) if !(lo <= k.lo && k.hi <= hi) => wrong.push(format!("{name}: {}..{} is not inside {lo}..{hi}", k.lo, k.hi)),
                    _ => {}
                }
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("
"));
    }

    #[test]
    fn widening_keeps_counts_and_sizes_from_going_below_nothing() {
        assert_eq!(widen(0.0, 1.0), (0.0, 3.0));
        assert_eq!(widen(-1.0, 1.0), (-5.0, 5.0));
        assert_eq!(hold(f64::NAN, (2.0, 3.0)), 2.0);
    }
}
