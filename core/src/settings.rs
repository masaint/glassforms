//! Every setting of the picture, the camera, the lights and the motion: `P` in the web engine.
//! Each has the same name here, in a saved setup and on the command line as it has in the web version's URL.

use serde::{Deserialize, Serialize};

macro_rules! settings {
    ($( $(#[$doc:meta])* $field:ident, $name:literal, $default:expr; )*) => {
        #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
        #[serde(default)]
        pub struct Settings {
            $( $(#[$doc])* #[serde(rename = $name)] pub $field: f64, )*
            /// how strongly the glass soaks up red, green and blue
            pub absorb: [f64; 3],
            /// the view slid sideways and up, in units of half the picture's height
            pub shift: [f64; 2],
            /// the orientation the family starts in: degrees about x, y, z
            pub rot: [f64; 3],
        }
        impl Default for Settings {
            fn default() -> Self {
                Self { $( $field: $default, )* absorb: [0.0; 3], shift: [0.0; 2], rot: [0.0; 3] }
            }
        }
        impl Settings {
            /// The names of all the settings that are single numbers.
            pub const NAMES: &'static [&'static str] = &[$($name),*];
            pub fn get(&self, name: &str) -> Option<f64> {
                match name { $( $name => Some(self.$field), )* _ => self.get_derived(name) }
            }
            /// False if there is no such setting.
            pub fn set(&mut self, name: &str, v: f64) -> bool {
                match name { $( $name => { self.$field = v; true } )* _ => self.set_derived(name, v) }
            }
        }
    };
}

settings! {
    /// index of refraction
    ior, "ior", 1.5;
    /// how much the index changes with colour
    disp, "disp", 0.08;
    /// bounces followed inside the glass
    nb, "nb", 4.0;
    /// rays per pixel, and wavelengths carried by each
    ns, "ns", 1.0;
    nw, "nw", 3.0;
    exposure, "exposure", 1.0;
    bloom, "bloom", 0.25;
    vignette, "vignette", 0.5;
    grain, "grain", 0.004;
    /// how far the old picture may stray from the new frame's neighbourhood (in standard deviations)
    spread, "spread", 2.0;
    /// antialiasing while moving: how much of the previous picture is kept each frame (0 = off)
    smooth, "smooth", 0.9;
    dist, "dist", 5.0;
    fov, "fov", 28.0;
    /// internal render scale (1 = native)
    scale, "scale", 1.0;
    /// camera roll about its line of sight, degrees
    roll, "roll", 0.0;
    /// radians a second
    spin, "spin", 0.35;
    /// 0 = tumble (the shape also rocks as it turns), 1 = turntable (straight round the vertical)
    spin_mode, "spinMode", 0.0;
    /// 0 = the lights stay with the room, 1 = they turn with the shape
    lights, "lights", 0.0;
    /// the whole set of lights, turned by hand: degrees
    lyaw, "lyaw", 0.0;
    lpitch, "lpitch", 0.0;
    lroll, "lroll", 0.0;
    /// the lights circling by themselves, radians a second
    lspin, "lspin", 0.0;
    /// how focus is done: 0 = every ray is traced through the lens (exact, grainy while anything moves),
    /// 1 = the picture is traced sharp and blurred afterwards by how far each piece of glass is from the band
    dof, "dof", 1.0;
    /// denoising of what is traced by chance: 0 = off, 1 = light wandering inside a milky body and
    /// roughness drawn by sampling alone, 2 = clear glass as well (the colours of a gem's fire)
    denoise, "denoise", 1.0;
    /// radius of the lens: 0 = everything sharp
    aperture, "aperture", 0.0;
    /// rays through the lens for each ray through the pixel
    lens_n, "lensN", 1.0;
    /// sides of the lens opening: 0 = round
    blades, "blades", 0.0;
    /// how far the opening is turned, degrees
    bokeh_rot, "bokehRot", 0.0;
    /// the band that is in focus: depth of its two planes from the object's centre (negative = nearer)
    fnear, "fnear", -0.15;
    ffar, "ffar", 0.15;
    /// 1 = draw where the two planes cut the glass
    planes, "planes", 0.0;
    /// one multiplier on everything that moves
    speed, "speed", 1.0;
    /// how animated sliders move between their stops: index into EASES (1 = sine)
    ease, "ease", 1.0;
    /// where the easing acts: 0 = at the start, 1 = at the finish, 2 = both
    ease_at, "easeAt", 2.0;
    /// length of one loop in seconds: 0 = work it out
    loop_len, "loop", 0.0;
    /// 1 = play back exactly as it will loop
    lock, "lock", 0.0;
    /// 1 = while moving, show where the antialiasing could not follow the surface
    debug, "debug", 0.0;
    /// 1 = show the traced picture without tone mapping
    raw, "raw", 0.0;
    /// 1 = draw the shape as matte clay instead of glass (much less work: for drafting a form)
    clay, "clay", 0.0;
    /// brightness of all the family's lights together, the light from everywhere included (0 to 4)
    light_gain, "lightGain", 1.0;
    /// brightness of the environment picture (0 to 8)
    hdri_gain, "hdriGain", 1.0;
    /// how much the environment picture is softened for the glass: 0 = as sharp as it is, 1 = very soft
    hdri_blur, "hdriBlur", 0.5;
    /// 1 = the environment picture is also shown behind the shape
    hdri_show, "hdriShow", 0.0;
    /// 1 = the family's lights stay lit together with the environment picture
    hdri_lights, "hdriLights", 0.0;
    /// The camera in a photographer's terms (camera.rs). The lens's focal length in millimetres: 0 = not
    /// said, the lens is given by `fov`. Once said, `fov` is rewritten from it.
    focal, "focal", 0.0;
    /// height of the sensor in millimetres (24 = full frame); the picture is as tall as the sensor
    sensor, "sensor", 24.0;
    /// the f-number: 0 = not said, the lens opening is given by `aperture`. Once said, `aperture` is
    /// rewritten from it: focal length / (2 x f-number), taken into the scene's units by `subject`
    fstop, "fstop", 0.0;
    /// how big the subject is: millimetres across a shape that fills the usual fit. This is what ties the
    /// scene's units to real ones, and so what makes a given f-stop blur as it would for a subject that size
    subject, "subject", 100.0;
    /// what kind of picture the lens draws: 0 perspective, 1 orthographic, 2 fisheye (equidistant),
    /// 3 fisheye (equisolid), 4 fisheye (stereographic), 5 panorama (360 x 180), 6 cylindrical
    proj, "proj", 0.0;
    /// straight lines bowed outward (above 0: barrel) or inward (below 0: pincushion): the share by which
    /// the corner of a 3:2 picture is displaced
    distort, "distort", 0.0;
    /// colour fringes towards the corners (0 to 1)
    fringe, "fringe", 0.0;
    /// the lens's own darkening towards the corners (0 = none, 1 = all of it)
    falloff, "falloff", 0.0;
    /// an anamorphic lens: out-of-focus highlights are this many times taller than wide (1 = an ordinary lens)
    squeeze, "squeeze", 1.0;
    /// out-of-focus highlights with a bright rim (towards 1) or a bright core fading out (towards -1)
    bokeh, "bokeh", 0.0;
    /// out-of-focus highlights clipped to a cat's eye towards the corners (0 to 1)
    catseye, "catseye", 0.0;
    /// the plane of focus leant back (about the picture's horizontal) and swung (about its vertical), degrees
    tilt_x, "tiltX", 0.0;
    tilt_y, "tiltY", 0.0;
    /// shadows and caustics that the bodies cast on each other and on themselves:
    /// 0 = nothing casts, 1 = each body as its own switches say, 2 = every body casts
    cast, "cast", 0.0;
    /// how large the lights are taken to be for what is cast: 1 = as large as they are (soft shadows,
    /// spread-out caustics), towards 0 = points (crisp ones)
    cast_soft, "castSoft", 0.35;
    /// how much of the light a body hides is taken away (1 = all of it), and how much of the light a
    /// body focuses is added (1 = all of it)
    cast_shadows, "castShadows", 1.0;
    cast_caustics, "castCaustics", 1.0;
    /// how fine the caustics are: 0 to 3
    cast_detail, "castDetail", 2.0;
    /// each body's own switches, kept under its lowest layer (the first without a number):
    /// 0 = it does not cast, 1 = it casts, 2 = as its material would have it
    cast_shadow1, "castShadow", 2.0;
    cast_caustic1, "castCaustic", 2.0;
    cast_shadow2, "castShadow2", 2.0;
    cast_caustic2, "castCaustic2", 2.0;
    cast_shadow3, "castShadow3", 2.0;
    cast_caustic3, "castCaustic3", 2.0;
    cast_shadow4, "castShadow4", 2.0;
    cast_caustic4, "castCaustic4", 2.0;
    // ---- materials (see material.rs: what each does, and its range) -----------------------------------
    /// which named material was last chosen: index into material::NAMED (0 = the family's own glass)
    mat, "mat", 0.0;
    /// the tint's depth: multiplies the absorption inside the body
    density, "density", 1.0;
    /// 1 = the material's distances (how deep its colour is, how far light travels in it, its glow) are
    /// measured against how thick the shape is, so that it looks the same on a thin shape as on a fat one
    fit, "fit", 0.0;
    /// absorption by wavelength instead of the tint: 0 = none, else 1 + index into material::SPECTRA
    spectrum, "spectrum", 0.0;
    /// how rough the surface is: 0 = polished, 1 = sandblasted
    rough, "rough", 0.0;
    /// 1 = roughness by sampling alone, nothing softened beforehand (slow to settle; for checking the quick way)
    rough_ref, "roughRef", 0.0;
    /// 1 = a surface that is not glass (a metal, a milky body, a glaze) mirrors an environment picture as
    /// sharp as it is, softened only by its own roughness; 0 = softened as the glass sees it (`hdriBlur`)
    sharp, "sharp", 1.0;
    /// how used the surface looks: every one of its marks below together, 0 = new and perfect, 1 = as the
    /// material's own amounts say, 2 = twice that
    age, "age", 1.0;
    /// the surface's marks, each from 0 (none) to 1 (a lot), fixed to the object and tied to its form:
    /// high points and edges rubbed smoother and brighter; recesses duller and darker, holding dirt or
    /// tarnish (of the colour below); fine scratches, most where the surface is exposed; polishing swirls;
    /// the gentle wave of a sprayed or dipped coat (orange peel); smudges and finger marks; dust on faces
    /// that look up
    edge_wear, "edgeWear", 0.0;
    cavity, "cavity", 0.0;
    cavity_r, "cavityR", 0.35;
    cavity_g, "cavityG", 0.27;
    cavity_b, "cavityB", 0.2;
    scratches, "scratches", 0.0;
    swirls, "swirls", 0.0;
    peel, "peel", 0.0;
    smudges, "smudges", 0.0;
    dust, "dust", 0.0;
    /// fine parallel polishing lines round the shape's vertical (brushed metal): 0 = none
    brush, "brush", 0.0;
    /// uneven colour: clouds and veins in a milky body, a faint unevenness of a metal's colour (0 = even)
    mottle, "mottle", 0.0;
    /// a thin film on the surface: thickness in nanometres (0 = none), its index, how much the thickness
    /// varies across the surface, how large that pattern is, where the pattern is in its flow (one turn =
    /// 1), and how much thinner the film is towards the top
    film, "film", 0.0;
    film_ior, "filmIor", 1.33;
    film_var, "filmVar", 0.0;
    film_scale, "filmScale", 1.0;
    film_flow, "filmFlow", 0.0;
    film_drain, "filmDrain", 0.0;
    /// a soft white light from all round that only the film reflects (see material.rs: FILM_SKY)
    film_sky, "filmSky", 0.5;
    /// a clear glossy coat over the body: how much of one (0 = none, 1 = a full coat), and its index
    coat, "coat", 0.0;
    coat_ior, "coatIor", 1.5;
    /// an opaque reflecting body: 0 = no, 1 = a perfect mirror, else 2 + index into material::METALS
    metal, "metal", 0.0;
    /// 1 = the shape is a thin wall around air (a bubble, a blown ornament) instead of solid
    hollow, "hollow", 0.0;
    /// the body's own colour: of the light scattered inside a milky body, of the mirror in a metal
    body_r, "bodyR", 1.0;
    body_g, "bodyG", 1.0;
    body_b, "bodyB", 1.0;
    /// light scattering inside the body: how milky (0 = clear), how far light travels between scatterings,
    /// and 1 = follow the light as it wanders (true, costly) instead of estimating it
    sss, "sss", 0.0;
    sss_depth, "sssDepth", 0.2;
    sss_walk, "sssWalk", 0.0;
    /// how differently the colours scatter: 0 = all alike; towards 1 blue is scattered more often and red
    /// less (as in milk glass and wax: a bluish edge, a warm glow deeper in and in the shade)
    sss_spread, "sssSpread", 0.0;
    /// how far the light scattered inside takes on the body's colour at every depth, not only deep in it
    /// (0 = as light that scatters does; towards 1, as a dyed body glows its colour just under the skin
    /// and through thin parts: wax)
    sss_hue, "sssHue", 0.0;
    /// how much the body scatters forward (the mean cosine of its scattering, 0 = every way alike, up to
    /// 0.95): wax and skin scatter forward
    sss_forward, "sssForward", 0.0;
    /// how far the light that crosses the body without being scattered away takes on the body's colour,
    /// deeper the further it goes (0 = not at all; towards 1, as thin wax lit from behind glows amber
    /// and its edges orange)
    sss_through, "sssThrough", 0.0;
    /// 1 = with a milky body the picture is exposed for it, as a photographer exposes for the subject: a
    /// white matte surface facing the brightest way comes to white (never darker than set; glass is not
    /// affected, as it shows the lights themselves)
    meter, "meter", 1.0;
    /// how fine the volume is in which the light inside a milky body is worked out: cells along a side
    sss_detail, "sssDetail", 160.0;
    /// bubbles inside the body: how many (0 = none), how large, how evenly (1) or unevenly (0) spread and
    /// sized, their shape (-1 flat discs, 0 round, 1 needles) and which random arrangement
    bubbles, "bubbles", 0.0;
    bubble_size, "bubbleSize", 0.03;
    bubble_even, "bubbleEven", 0.5;
    bubble_shape, "bubbleShape", 0.0;
    bubble_seed, "bubbleSeed", 1.0;
    /// light given off inside the body: how much, its colour, and how far it keeps to the core
    glow, "glow", 0.0;
    glow_r, "glowR", 1.0;
    glow_g, "glowG", 0.32;
    glow_b, "glowB", 0.08;
    glow_core, "glowCore", 0.5;
    /// an index that changes with depth below the surface (rays curve inside): by how much, over what depth
    grin, "grin", 0.0;
    grin_depth, "grinDepth", 0.02;
    /// double refraction: the second ray's index minus the first's, and the crystal's axis in the shape's
    /// own frame (degrees round the vertical, degrees up)
    biref, "biref", 0.0;
    biref_az, "birefAz", 0.0;
    biref_el, "birefEl", 90.0;
    // ---- inside the body (see inner.rs: what each does, and its range) -----------------------------------
    /// a tint that changes through the body: how far it goes over to the second tint (0 = not at all), how
    /// it is laid out (0 along an axis, 1 outward from the middle, 2 in bands), the second tint as a
    /// colour (as tintR..B are the first), the axis (degrees round the vertical, degrees up), where the
    /// change lies, how gradual it is, how many bands and how wavy they are
    grad, "grad", 0.0;
    grad_mode, "gradMode", 0.0;
    grad_r, "gradR", 0.55;
    grad_g, "gradG", 0.85;
    grad_b, "gradB", 1.0;
    grad_az, "gradAz", 0.0;
    grad_el, "gradEl", 90.0;
    grad_offset, "gradOffset", 0.0;
    grad_soft, "gradSoft", 0.6;
    grad_bands, "gradBands", 6.0;
    grad_wave, "gradWave", 0.3;
    /// cracks inside the body: how many (0 = none), how large the largest are, how far below the surface
    /// they reach (1 = all through), how wide they stand open (0 = closed and unseen, about 0.4 the
    /// colours of a thin gap, towards 1 silver), how far they have grown (0 = not begun, 1 = fully), how
    /// shell-shaped they are (0 = flat with straight edges), how rough their faces are (0 = a clean break
    /// that mirrors, 1 = a white veil), and which random arrangement
    cracks, "cracks", 0.0;
    /// fracture surfaces running through the whole body (healed veils, open cracks, cleavages, a fan
    /// from one point): how many and how far (0 = none); they share the cracks' opening, curvature,
    /// roughness, growth and arrangement
    fractures, "fractures", 0.0;
    crack_size, "crackSize", 0.25;
    crack_depth, "crackDepth", 1.0;
    crack_open, "crackOpen", 0.8;
    crack_grow, "crackGrow", 1.0;
    crack_curve, "crackCurve", 0.5;
    crack_rough, "crackRough", 0.0;
    crack_seed, "crackSeed", 1.0;
    /// grains that send light back (precious opal's play of colour, or mirror platelets): how strongly
    /// (0 = none), how large, their pattern (0 blocks, 1 pinfire, 2 broad flash, 3 platelets), the
    /// longest colour they send back (0 blue .. 1 red), how softened the lights they mirror are, how far
    /// neighbouring grains flash together, which arrangement; and the platelets' colour
    play, "play", 0.0;
    play_size, "playSize", 0.12;
    play_pattern, "playPattern", 0.0;
    play_colour, "playColour", 0.8;
    play_soft, "playSoft", 0.3;
    play_roll, "playRoll", 0.0;
    play_seed, "playSeed", 1.0;
    glint_r, "glintR", 1.0;
    glint_g, "glintG", 0.85;
    glint_b, "glintB", 0.6;
    /// silky fibres (a cat's eye): how strongly they scatter (0 = none), their direction (degrees round
    /// the vertical, degrees up), how soft the band is, how wavy the fibres are
    silk, "silk", 0.0;
    silk_az, "silkAz", 0.0;
    silk_el, "silkEl", 0.0;
    silk_soft, "silkSoft", 0.3;
    silk_wave, "silkWave", 0.0;
    /// light given off under pressure: how much (0 = none; animate it to pulse), and its colour (0 warm
    /// white .. 1 orange)
    piezo, "piezo", 0.0;
    piezo_colour, "piezoColour", 0.5;
    // ---- the studio (see stage.rs) ----------------------------------------------------------------------
    /// the camera raised on an arc above the floor and looking down at the shape, degrees (0 = level)
    elev, "elev", 0.0;
    /// the camera carried round the room's vertical on that arc, degrees (0 = where it has always stood):
    /// the room, its floor, its backdrop and the lights fixed to it turn in the picture together with the shape
    azim, "azim", 0.0;
    /// a studio backdrop: 0 = none (black), 1 = a sweep of three colours, with a glow if one is set
    backdrop, "backdrop", 0.0;
    /// 1 = the backdrop is seen behind the shape; 0 = it only lights the shape
    back_show, "backShow", 1.0;
    /// how bright the backdrop is
    back_gain, "backGain", 1.0;
    /// degrees above and below the horizon over which the horizon's colour gives way to the other two
    back_spread, "backSpread", 38.0;
    /// how far round from straight behind the shape the backdrop reaches, degrees: 180 = all the way round,
    /// 90 = the half of the room behind the shape (the rest of the room is dark, as behind a camera)
    back_wrap, "backWrap", 180.0;
    /// how far up from the horizon it reaches, degrees: 90 = to straight overhead
    back_high, "backHigh", 90.0;
    /// its colour overhead, at the horizon and underfoot
    back_top_r, "backTopR", 0.60;
    back_top_g, "backTopG", 0.60;
    back_top_b, "backTopB", 0.93;
    back_mid_r, "backMidR", 0.80;
    back_mid_g, "backMidG", 0.87;
    back_mid_b, "backMidB", 0.98;
    back_low_r, "backLowR", 0.70;
    back_low_g, "backLowG", 0.71;
    back_low_b, "backLowB", 0.95;
    /// a round glow on the backdrop: how strong (0 = none), its colour, where its middle is (degrees
    /// round from straight behind the shape, degrees up) and how wide it is
    back_glow, "backGlow", 0.0;
    back_glow_r, "backGlowR", 1.0;
    back_glow_g, "backGlowG", 1.0;
    back_glow_b, "backGlowB", 1.0;
    back_glow_az, "backGlowAz", 0.0;
    back_glow_el, "backGlowEl", 0.0;
    back_glow_size, "backGlowSize", 30.0;
    /// 1 = a floor under the shape, level in the room
    floor, "floor", 0.0;
    /// 1 = the floor touches the lowest point of the shape (and `floorY` raises or lowers it from there);
    /// 0 = `floorY` is its height itself, measured from the middle of the room
    floor_auto, "floorAuto", 1.0;
    floor_y, "floorY", 0.0;
    /// the share of the light falling on the floor that it gives back, by colour, and all three together
    floor_r, "floorR", 1.0;
    floor_g, "floorG", 1.0;
    floor_b, "floorB", 1.0;
    floor_gain, "floorGain", 1.0;
    /// what the floor's colour is taken from: 1 = the backdrop's colour underfoot (a seamless sweep, whatever
    /// the lights are), 0 = the light that falls on it from the lights and the backdrop (a floor that
    /// brightens and darkens with them); between, a mix. Without a backdrop it is always the light.
    floor_match, "floorMatch", 1.0;
    /// how far from the middle of the room the floor has melted wholly into the backdrop
    floor_fade, "floorFade", 16.0;
    /// 0 = a matte floor, up to 1 = a glossy one; and 1 = the shape itself is mirrored in a glossy floor
    floor_gloss, "floorGloss", 0.0;
    floor_mirror, "floorMirror", 0.0;
    /// the tone curve: 0 = filmic (the one the program has always had), 1 = neutral, 2 = soft, 3 = none
    tone, "tone", 0.0;
    /// white balance: warmer (above 0) or cooler; more magenta (above 0) or greener
    wb_temp, "wbTemp", 0.0;
    wb_tint, "wbTint", 0.0;
    /// 1 = as it is; contrast about middle grey, and how strong the colours are
    contrast, "contrast", 1.0;
    saturation, "saturation", 1.0;
}

impl Settings {
    /// Settings that are another setting seen differently: the tint is the colour white light has taken
    /// after one unit of travel through the body (before `density`), which is the absorption said as a
    /// colour. They are sliders like any other, but only `absorb` is kept.
    pub const DERIVED: &'static [&'static str] = &["tintR", "tintG", "tintB"];
    fn get_derived(&self, name: &str) -> Option<f64> {
        let i = Self::DERIVED.iter().position(|n| *n == name)?;
        Some((-self.absorb[i]).exp())
    }
    fn set_derived(&mut self, name: &str, v: f64) -> bool {
        let Some(i) = Self::DERIVED.iter().position(|n| *n == name) else { return false };
        self.absorb[i] = -v.clamp(1e-4, 1.0).ln();
        true
    }
    /// Set a setting from text: a number, or comma-separated numbers for `absorb`, `shift`, `rot`.
    pub fn set_text(&mut self, name: &str, text: &str) -> bool {
        let nums: Vec<f64> = text.split(',').filter_map(|s| s.trim().parse::<f64>().ok().filter(|v| v.is_finite())).collect();
        let whole = nums.len() == text.split(',').count();
        match name {
            "absorb" if whole && nums.len() == 3 => { self.absorb = [nums[0], nums[1], nums[2]]; true }
            "shift" if whole && nums.len() == 2 => { self.shift = [nums[0], nums[1]]; true }
            "rot" if whole && nums.len() == 3 => { self.rot = [nums[0], nums[1], nums[2]]; true }
            _ if whole && nums.len() == 1 => self.set(name, nums[0]),
            _ => false,
        }
    }
    /// The easing set for all tracks.
    pub fn easing(&self) -> (usize, u8) {
        let ty = crate::anim::js_round(self.ease);
        let ty = if ty >= 0.0 && (ty as usize) < crate::anim::EASES.len() { ty as usize } else { 1 };
        (ty, crate::anim::js_round(self.ease_at).clamp(0.0, 2.0) as u8)
    }
}

/// How a video is rendered.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RenderOpts {
    /// "window", or "<width>x<height>"
    pub size: String,
    pub fps: u32,
    /// traced frames averaged into each frame of the video
    pub samples: u32,
    /// rays through the lens in a video (0 = as live)
    pub lens: u32,
    /// motion blur: the share of each frame's time the shutter is open (0 = none, 0.5 = a film camera's usual)
    pub shutter: f64,
    /// "h264", "h265" or "prores"
    pub codec: String,
    /// quality of H.264 / H.265: lower is better and larger
    pub crf: u32,
}
impl Default for RenderOpts {
    fn default() -> Self {
        Self { size: "1920x1080".into(), fps: 60, samples: 32, lens: 4, shutter: 0.0, codec: "h264".into(), crf: 16 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_get_and_set() {
        let mut p = Settings::default();
        assert_eq!(p.get("spinMode"), Some(0.0));
        assert!(p.set("lensN", 4.0) && p.lens_n == 4.0);
        assert!(!p.set("nothing", 1.0));
        assert!(p.set_text("absorb", "0.1, 0.05,0.02") && p.absorb == [0.1, 0.05, 0.02]);
        assert!(!p.set_text("absorb", "0.1,x,0.02"));
        assert!(p.set_text("bloom", "0.4") && p.bloom == 0.4);
        for n in Settings::NAMES {
            assert!(p.get(n).is_some());
        }
    }
}
