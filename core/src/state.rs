//! Everything the program knows that is not on the GPU: which shape is up and its sliders, every setting,
//! the object's orientation, the lights, the animation clock and its tracks. This is engine.js and the
//! family() part of forms.js; the GPU side asks it what to build and what to draw.

use std::collections::BTreeMap;

use crate::anim::{self, js_round, LoopPlan, Track};
use crate::curve::{self, Curve, PlaceOpts, Pose, PrepOpts};
use crate::layers::{Opt, Stack};
use crate::math::*;
use crate::presets::{families, CurveOpts, Family, Kind, Light, Preset, RawCurve, Setting, Shape};
use crate::settings::{RenderOpts, Settings};
use crate::{field, material, HB, TRAVEL};

/// Voxels per side of the volume: when still, while a slider is dragged, while the shape animates.
pub const RES_FULL: u32 = 768;
pub const RES_DRAG: u32 = 256;
pub const RES_ANIM: u32 = 256;
/// The largest volume offered (a 16-bit volume this size takes 7 GB of video memory).
pub const RES_MAX: u32 = 1536;

/// A volume size as it can be built: within range, and a multiple of eight (slices are written eight at a time).
pub fn valid_res(v: f64) -> u32 {
    let r = if v.is_finite() { v as i64 } else { RES_FULL as i64 };
    (((r.clamp(64, RES_MAX as i64) + 4) / 8) * 8) as u32
}

/// How carefully a shape is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Detail {
    /// full detail
    Full,
    /// a slider is being dragged
    Quick,
    /// one frame of live animation
    Anim,
    /// one frame of a video: full detail, nothing approximated
    Hq,
}

/// What the last build of a shape came to.
#[derive(Clone, Debug, Default)]
pub struct Meta {
    pub res: u32,
    pub ms: f64,
    pub segments: usize,
    pub splats: usize,
    pub length: f64,
    pub info: String,
}

/// The long stretch of orbit a flowing tube is cut from, kept while the numbers do not change.
struct OrbitCache {
    key: String,
    quick: bool,
    line: Vec<f64>,
    pose: Option<Pose>,
}

/// One family's shape as it stands: which preset, and every slider of the shape picker.
pub struct ShapeState {
    pub preset: usize,
    pub vals: Vec<f64>,
    pub thick: f64,
    pub len: f64,
    pub melt: f64,
    pub puff: f64,
    pub travel: f64,
    pub beads: f64,
    pub bead_period: f64,
    pub res: u32,
    pub anim_res: u32,
    /// the model file the user chose (for the `file` shape of the family of models): a full path, or
    /// one below the program's folder
    pub model: String,
    pub meta: Meta,
    orbit: Option<OrbitCache>,
    /// the face-on turn worked out for the still shape, held while it animates
    e: Option<[[f64; 3]; 3]>,
}

/// A family's lights as they stand: the soft rectangular lights and the light that comes from everywhere.
#[derive(Clone, Debug, PartialEq)]
pub struct Rig {
    pub lights: Vec<Light>,
    pub ambient: [f64; 3],
}

/// The softness a light gets when none is given (as in the web version).
pub const LIGHT_SOFT: f64 = 0.6;

/// A light from text: `az,el,w,h,r,g,b` or `az,el,w,h,r,g,b,soft` (degrees; r, g, b carry colour and brightness).
pub fn light_from_text(text: &str) -> Option<Light> {
    let v: Vec<f64> = text.split(',').map(|s| s.trim().parse::<f64>().ok().filter(|x| x.is_finite())).collect::<Option<_>>()?;
    if v.len() != 7 && v.len() != 8 { return None; }
    Some(Light { az: v[0], el: v[1], w: v[2], h: v[3], rgb: [v[4], v[5], v[6]], soft: v.get(7).copied().unwrap_or(LIGHT_SOFT) })
}
/// The same text back.
pub fn light_to_text(l: &Light) -> String {
    format!("{},{},{},{},{},{},{},{}", l.az, l.el, l.w, l.h, l.rgb[0], l.rgb[1], l.rgb[2], l.soft)
}
impl Rig {
    /// Lights from text: one light after another with `:` between them (see `light_from_text`). Empty text
    /// is no lights at all. None if any of it cannot be read.
    pub fn lights_from_text(text: &str) -> Option<Vec<Light>> {
        if text.trim().is_empty() { return Some(Vec::new()); }
        text.split(':').map(light_from_text).collect()
    }
    /// The lights as text that `lights_from_text` reads back to the same numbers.
    pub fn lights_text(&self) -> String {
        self.lights.iter().map(light_to_text).collect::<Vec<_>>().join(":")
    }
}

impl ShapeState {
    /// The value of one of the shape's own sliders.
    pub fn knob(&self, r: &KnobRef) -> f64 {
        match r {
            KnobRef::Param(i) => self.vals.get(*i).copied().unwrap_or(0.0),
            KnobRef::Thick => self.thick,
            KnobRef::Melt => self.melt,
            KnobRef::Len => self.len,
            KnobRef::Travel => self.travel,
            KnobRef::Beads => self.beads,
            KnobRef::Puff => self.puff,
            _ => 0.0,
        }
    }
    pub fn set_knob(&mut self, r: &KnobRef, v: f64) {
        match r {
            KnobRef::Param(i) => { if let Some(x) = self.vals.get_mut(*i) { *x = v; } }
            KnobRef::Thick => self.thick = v,
            KnobRef::Melt => self.melt = v,
            KnobRef::Len => self.len = v,
            KnobRef::Travel => self.travel = v,
            KnobRef::Beads => self.beads = v,
            KnobRef::Puff => self.puff = v,
            _ => {}
        }
    }
}

/// A layer that is not the selected one: its family, and its shape in every family.
pub struct Parked {
    pub fam: usize,
    pub shapes: Vec<ShapeState>,
}

/// What the GPU side has to build.
pub enum ShapeJob {
    Curve { curve: Curve, melt: f64 },
    Field { uniforms: [f32; 12] },
    /// a polygon model: the file and what its sliders say (reading it is the GPU side's business)
    Mesh(crate::mesh::MeshJob),
}
pub struct Job {
    pub family: usize,
    pub preset: usize,
    pub res: u32,
    pub shape: ShapeJob,
}

/// Which value a knob stands for.
#[derive(Clone, Debug, PartialEq)]
pub enum KnobRef {
    /// one of the settings, by name
    P(&'static str),
    /// the object's orientation: angle about x, y or z
    Orient(usize),
    /// one of the preset's own numbers
    Param(usize),
    Thick,
    Melt,
    Len,
    Travel,
    Beads,
    Puff,
    /// one of the above, of the layer at this place in the list (on their own they mean the selected layer's)
    Of(usize, Box<KnobRef>),
    /// a layer's placement: 0..2 position, 3..5 turn, 6 size
    Place(usize, usize),
    /// a numeric option of a pair, a triple or the quad of layers (a mask of their places)
    Combo(u8, Opt),
    /// a setting of the material of the body whose lowest layer is at this place in the list
    Body(usize, &'static str),
}

/// Anything with a slider.
#[derive(Clone, Debug)]
pub struct Knob {
    /// unique: "P.dist"; "<family>/<preset>/<id>" for the first layer's shape, "L2/<family>/<preset>/<id>"
    /// for the second's; "L2.x" for a placement; "J12.radius" for an option of a combination
    pub key: String,
    pub id: String,
    pub label: String,
    pub lo: f64,
    pub hi: f64,
    pub step: f64,
    /// changing it means making the volume again (building a shape, or only combining the layers)
    pub shape: bool,
    /// it can be animated
    pub wave: bool,
    pub r: KnobRef,
}

/// name, label, lo, hi, step, can be animated
const ENGINE_KNOBS: &[(&str, &str, f64, f64, f64, bool)] = &[
    ("dist", "distance", 1.5, 20.0, 0.05, true),
    ("fov", "lens °", 6.0, 80.0, 0.5, true),
    ("roll", "roll °", -180.0, 180.0, 0.5, true),
    ("focal", "focal length mm", 4.0, 400.0, 0.5, true),
    ("zoom", "focal length", -3.64, 3.0, 0.01, true),
    ("av", "f-stop", -1.0, 12.0, 0.3333333333333333, true),
    ("subjectLog", "subject size", 0.0, 5.0, 0.01, false),
    ("shiftX", "shift across", -4.0, 4.0, 0.01, true),
    ("shiftY", "shift up", -4.0, 4.0, 0.01, true),
    ("fstop", "f-stop", 0.7, 64.0, 0.1, true),
    ("focus", "focus", -1.6, 1.6, 0.01, true),
    ("band", "sharp range", 0.0, 1.6, 0.01, true),
    ("tiltX", "focus tilt °", -60.0, 60.0, 0.5, true),
    ("tiltY", "focus swing °", -60.0, 60.0, 0.5, true),
    ("subject", "subject mm", 1.0, 100000.0, 1.0, false),
    ("ev", "exposure EV", -3.0, 3.0, 0.05, true),
    ("distort", "distortion", -0.4, 0.4, 0.005, true),
    ("fringe", "colour fringing", 0.0, 1.0, 0.01, true),
    ("falloff", "fall-off", 0.0, 1.0, 0.01, true),
    ("squeeze", "anamorphic ×", 1.0, 2.0, 0.01, true),
    ("bokeh", "highlight edge", -1.0, 1.0, 0.01, true),
    ("catseye", "cat's-eye", 0.0, 1.0, 0.01, true),
    ("vignette", "vignette", 0.0, 1.5, 0.01, true),
    ("grain", "grain", 0.0, 0.05, 0.001, true),
    ("spin", "spin", 0.0, 1.5, 0.01, false),
    ("speed", "speed ×", 0.1, 4.0, 0.05, false),
    ("lyaw", "light yaw °", 0.0, 360.0, 0.5, true),
    ("lpitch", "light pitch °", 0.0, 360.0, 0.5, true),
    ("lroll", "light roll °", 0.0, 360.0, 0.5, true),
    ("lspin", "circling", 0.0, 1.5, 0.01, false),
    ("lightGain", "lights ×", 0.0, 4.0, 0.01, true),
    ("hdriGain", "picture ×", 0.0, 8.0, 0.01, true),
    ("hdriBlur", "soften", 0.0, 1.0, 0.01, true),
    ("castSoft", "light size ×", 0.0, 1.0, 0.01, true),
    ("castShadows", "shadows", 0.0, 1.0, 0.01, true),
    ("castCaustics", "caustics", 0.0, 4.0, 0.01, true),
    ("aperture", "aperture", 0.0, 0.3, 0.002, true),
    ("lensN", "lens rays", 1.0, 8.0, 1.0, true),
    ("bokehRot", "blade turn °", 0.0, 360.0, 1.0, true),
    ("fnear", "near plane", -1.6, 1.6, 0.01, true),
    ("ffar", "far plane", -1.6, 1.6, 0.01, true),
    ("ior", "index", 1.05, 2.7, 0.01, true),
    ("disp", "dispersion", 0.0, 0.4, 0.005, true),
    ("nb", "bounces", 1.0, 12.0, 1.0, true),
    ("exposure", "exposure", 0.2, 4.0, 0.05, true),
    ("bloom", "bloom", 0.0, 1.5, 0.01, true),
    ("smooth", "antialias", 0.0, 0.95, 0.01, false),
    ("scale", "render scale", 0.4, 2.0, 0.05, false),
    ("ns", "rays / pixel", 1.0, 6.0, 1.0, true),
    ("nw", "wavelengths", 3.0, 12.0, 1.0, true),
];

/// What a family's settings start from before its own are laid over them.
fn family_base() -> Vec<(&'static str, Setting)> {
    use Setting::*;
    vec![
        ("ior", Num(1.5)), ("disp", Num(0.09)), ("absorb", Vec3([0.10, 0.05, 0.02])), ("ns", Num(1.0)), ("nw", Num(3.0)), ("nb", Num(3.0)),
        ("dist", Num(6.6)), ("fov", Num(26.0)), ("rot", Vec3([12.0, 25.0, 0.0])), ("spin", Num(0.3)), ("bloom", Num(0.4)), ("exposure", Num(1.0)),
    ]
}

/// What has to be put back when a video render ends.
pub struct VideoKeep {
    playing: bool,
    clock: f64,
    theta: f64,
    lphi: f64,
    vals: Vec<(Knob, f64)>,
    /// what the animation had last written, as it was before the render wrote over it
    moved: BTreeMap<String, f64>,
    lens_live: f64,
    pub plan: LoopPlan,
    turns_spin: f64,
    turns_lspin: f64,
}

pub struct State {
    pub families: Vec<Family>,
    /// which family is up
    pub fam: usize,
    /// one per family: each keeps its own shape while another is up
    pub shapes: Vec<ShapeState>,
    /// The layers: where each sits, whether it shows, and every combination's options. `fam` and `shapes`
    /// are the selected layer's; the other layers' are kept in `parked` (see layers.rs).
    pub stack: Stack,
    pub(crate) parked: Vec<Option<Parked>>,
    pub p: Settings,
    pub render: RenderOpts,
    /// the object's own orientation: what dragging and the orientation tools change
    pub r0: Mat3,
    /// how far the spin has turned it since
    pub theta: f64,
    /// the two together: object -> world
    pub r: Mat3,
    /// how far the lights' own spin has turned them
    pub lphi: f64,
    pub playing: bool,
    pub clock: f64,
    /// knob key -> its track
    pub tracks: BTreeMap<String, Track>,
    /// knob key -> the value the animation last gave that knob, and the value the knob had before the
    /// animation moved it (its resting value). A knob that still holds what the animation gave it was
    /// left mid-trip; one that holds anything else was set by hand or by a setup, and that is its rest.
    moved: BTreeMap<String, f64>,
    rest: BTreeMap<String, f64>,
    /// a video is being rendered: everything is played exactly as it loops
    pub rendering: bool,
    /// the shape must be built again, this carefully
    pub want: Option<Detail>,
    /// something changed outright (a new family): the picture so far is useless
    pub cut: bool,
    /// a focus plane was moved: show the planes for a moment
    pub flash_planes: bool,
    /// one per family: each keeps its own lights while another is up
    rigs: Vec<Rig>,
    /// goes up whenever the lights or the environment picture change
    pub lights_rev: u64,
    /// the environment picture in use: its path, as given (one that is not absolute is looked for in the
    /// program's folder)
    pub hdri: Option<String>,
    /// The materials of the bodies that have one of their own. A body is known by its lowest layer
    /// (bodies are renumbered when layers are joined or set apart; layers are not), so they are kept by
    /// layer: an entry is used only while its layer is the lowest of a body, and never for the first
    /// layer, whose body is made of the scene's own material.
    pub body_mats: [Option<Box<Settings>>; material::MAX_BODIES],
    first_scene: bool,
    pub(crate) args: Vec<(String, String)>,
    auto: bool,
}

pub(crate) fn arg<'a>(args: &'a [(String, String)], k: &str) -> Option<&'a str> {
    args.iter().rev().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
}
fn arg_num(args: &[(String, String)], k: &str) -> Option<f64> {
    arg(args, k).filter(|v| !v.is_empty()).and_then(|v| v.trim().parse::<f64>().ok()).filter(|v| v.is_finite())
}

impl State {
    pub fn new() -> Self {
        Self::with_args(&[], false)
    }

    /// `args`: name=value pairs as the web version takes them in its URL (scene, preset, any setting,
    /// thick, len, melt, puff, travel, beads, beadPeriod, res, animRes, p.<name>, rx, ry, rz).
    /// `auto`: a still or a benchmark is being made, so nothing spins unless asked to.
    pub fn with_args(args: &[(String, String)], auto: bool) -> Self {
        let families = families();
        let mut shapes = Self::fresh_shapes(&families, args);
        // (model=<file> alone shows the family of models with that file up)
        let model = crate::presets::models::start(&families, &mut shapes, args);
        let fam = arg(args, "scene").and_then(|id| families.iter().position(|f| f.id == id)).or(model).unwrap_or(0);
        let rigs = families.iter().map(|f| Rig { lights: f.lights.clone(), ambient: f.ambient }).collect();
        let mut s = Self {
            families, fam, shapes, stack: Stack::new(), parked: vec![None], p: Settings::default(), render: RenderOpts::default(),
            r0: I3, theta: 0.0, r: I3, lphi: 0.0, playing: false, clock: 0.0, tracks: BTreeMap::new(), moved: BTreeMap::new(), rest: BTreeMap::new(),
            rendering: false, want: None, cut: true, flash_planes: false, rigs, lights_rev: 0, hdri: None, body_mats: Default::default(),
            first_scene: true, args: args.to_vec(), auto,
        };
        s.set_scene(fam);
        // the materials of bodies that are given one of their own (`B2.material=gold`): before the
        // layers' own arguments, among which a track may name one of a body's sliders
        for (k, v) in args { s.apply_body_arg(k, v); }
        s.layer_args();
        // the lights and the environment picture, in the order they were given
        for (k, v) in args {
            if Self::is_light_arg(k) { s.apply_arg(k, v); }
        }
        s
    }

    /// The detail a family's shapes are built at while they animate, unless something says otherwise:
    /// 256 voxels a side, or the family's own (`("animRes", Num(..))` among its defaults: a family whose
    /// shapes show their coarseness in motion, such as cut stones, asks for more).
    pub fn family_anim_res(f: &Family) -> u32 {
        f.defaults.iter().find_map(|(k, v)| match (*k, v) { ("animRes", Setting::Num(x)) => Some(valid_res(*x)), _ => None }).unwrap_or(RES_ANIM)
    }

    /// A shape for every family as a layer starts with them, with whatever `args` say laid over.
    pub(crate) fn fresh_shapes(families: &[Family], args: &[(String, String)]) -> Vec<ShapeState> {
        let res = arg_num(args, "res").map(valid_res).unwrap_or(RES_FULL);
        families
            .iter()
            .map(|f| {
                let preset = arg(args, "preset").and_then(|id| f.presets.iter().position(|p| p.id == id)).unwrap_or(0);
                let p = &f.presets[preset];
                let vals = p.params.iter().map(|d| arg_num(args, &format!("p.{}", d.id)).unwrap_or(d.value)).collect();
                ShapeState {
                    preset, vals,
                    thick: arg_num(args, "thick").unwrap_or(1.0), len: arg_num(args, "len").unwrap_or(1.0),
                    melt: arg_num(args, "melt").unwrap_or(f.melt), puff: arg_num(args, "puff").unwrap_or(0.0),
                    travel: arg_num(args, "travel").unwrap_or(0.0), beads: arg_num(args, "beads").unwrap_or(0.0),
                    bead_period: arg_num(args, "beadPeriod").unwrap_or(2.0),
                    res, anim_res: arg_num(args, "animRes").map(valid_res).unwrap_or_else(|| Self::family_anim_res(f)), model: String::new(), meta: Meta::default(), orbit: None, e: None,
                }
            })
            .collect()
    }

    // ---- the shape that is up ------------------------------------------------------------------
    pub fn family(&self) -> &Family { &self.families[self.fam] }
    pub fn shape(&self) -> &ShapeState { &self.shapes[self.fam] }
    pub fn shape_mut(&mut self) -> &mut ShapeState { &mut self.shapes[self.fam] }
    pub fn preset(&self) -> &Preset { &self.family().presets[self.shape().preset] }

    /// A family's settings: the common base with its own laid over.
    fn scene_defaults(&self) -> Vec<(&'static str, Setting)> {
        let mut d = family_base();
        for (k, v) in &self.look_family().defaults {
            match d.iter_mut().find(|(n, _)| n == k) {
                Some(slot) => slot.1 = v.clone(),
                None => d.push((k, v.clone())),
            }
        }
        d
    }
    fn apply_defaults(&mut self, d: &[(&'static str, Setting)]) {
        // a material chosen by name brings its own index, dispersion, bounces and tint: the family's are
        // then left out
        let named = self.p.mat != 0.0;
        for (k, v) in d {
            if named && matches!(*k, "ior" | "disp" | "nb" | "absorb") { continue; }
            match (*k, v) {
                ("absorb", Setting::Vec3(a)) => self.p.absorb = *a,
                ("rot", Setting::Vec3(a)) => self.p.rot = *a,
                // (the family's in-motion detail: read where its shapes are made, see `family_anim_res`)
                ("animRes", _) => {}
                (name, Setting::Num(x)) => { if !crate::camera::set(&mut self.p, name, *x) { self.p.set(name, *x); } }
                _ => {}
            }
        }
    }

    /// Show another family. The first time, settings start from scratch; after that the camera, lights
    /// and picture stay as they were set and only the family's own defaults are applied.
    pub fn set_scene(&mut self, idx: usize) {
        if self.playing { self.set_playing(false); }
        self.fam = idx.min(self.families.len() - 1);
        if self.stack.sel != 0 {
            // only the first layer's family decides the settings, the view and the lights
            self.cut = true;
            self.default_tracks();
            self.want = Some(Detail::Full);
            return;
        }
        let d = self.scene_defaults();
        if self.first_scene {
            self.p = Settings::default();
            self.apply_defaults(&d);
            let args = self.args.clone();
            // a material by name first, so that settings given beside it are laid over it
            if let Some(v) = arg(&args, "material").or_else(|| arg(&args, "mat")) { self.material_arg(v); }
            // a studio likewise: its lights, backdrop and floor first, then whatever was given beside it
            if let Some(v) = arg(&args, "studio") { self.studio_arg(v); }
            for name in Settings::NAMES {
                if *name == "mat" { continue; }
                if let Some(v) = arg_num(&args, name) { self.p.set(name, v); }
            }
            for name in ["absorb", "shift", "rot"] {
                if let Some(v) = arg(&args, name) { self.p.set_text(name, v); }
            }
            // the camera's names in the order they were given: of a lens said two ways (`fov` and
            // `focal`, `aperture` and `fstop`) the later is the one meant
            for (k, v) in &args { self.camera_arg(k, v); }
            for name in Settings::DERIVED {
                if let Some(v) = arg_num(&args, name) { self.p.set(name, v); }
            }
            if ["rx", "ry", "rz"].iter().any(|k| arg_num(&args, k).is_some()) {
                let r = self.p.rot;
                self.p.rot = [arg_num(&args, "rx").unwrap_or(r[0]), arg_num(&args, "ry").unwrap_or(r[1]), arg_num(&args, "rz").unwrap_or(r[2])];
            }
            if self.auto && arg_num(&args, "spin").is_none() { self.p.spin = 0.0; }
            self.first_scene = false;
        } else {
            self.apply_defaults(&d);
        }
        self.theta = 0.0;
        self.clock = 0.0;
        let rot = self.p.rot;
        self.set_r(&euler(rot[0], rot[1], rot[2]));
        self.cut = true;
        // the shape's own starting orientation, unless one was asked for
        if let Some(rot) = self.preset().rot {
            if arg(&self.args, "rx").is_none() && arg(&self.args, "ry").is_none() {
                self.p.rot = rot;
                self.set_r(&euler(rot[0], rot[1], rot[2]));
            }
        }
        self.default_tracks();
        self.want = Some(Detail::Full);
    }

    /// Make the shape of a named material (a position in `material::NAMED`): the glass of the family that
    /// owns the look first, then everything the material sets. False if there is no such material.
    pub fn set_material(&mut self, i: usize) -> bool {
        let Some(n) = material::NAMED.get(i) else { return false };
        for (k, v) in &self.scene_defaults() {
            match (*k, v) {
                ("absorb", Setting::Vec3(a)) => self.p.absorb = *a,
                ("ior" | "disp" | "nb", Setting::Num(x)) => { self.p.set(k, *x); }
                _ => {}
            }
        }
        n.apply(&mut self.p);
        self.p.mat = i as f64;
        true
    }
    // ---- a material for each body ----------------------------------------------------------------
    /// Has the body whose lowest layer this is a material of its own? (Never the first layer: its body
    /// is made of the scene's material.)
    pub fn body_own(&self, layer: usize) -> bool {
        layer > 0 && self.body_mats.get(layer).is_some_and(|m| m.is_some())
    }
    /// Give the body whose lowest layer this is a material of its own, which starts as the scene's; or
    /// take it away again, and the body is made of the scene's material once more.
    pub fn set_body_own(&mut self, layer: usize, own: bool) {
        if layer == 0 || layer >= self.body_mats.len() { return; }
        if !own {
            self.body_mats[layer] = None;
        } else if self.body_mats[layer].is_none() {
            self.body_mats[layer] = Some(Box::new(self.p.clone()));
        }
    }
    /// The material of the body whose lowest layer this is: its own if it has one, else the scene's.
    pub fn body_settings(&self, layer: usize) -> &Settings {
        match self.body_mats.get(layer) {
            Some(Some(m)) if layer > 0 => m,
            _ => &self.p,
        }
    }
    /// One setting of that body's material, by name. Setting one gives the body a material of its own.
    /// (For the first layer this is the scene's setting.) False if the name is not part of a material.
    pub fn body_set(&mut self, layer: usize, name: &str, v: f64) -> bool {
        if layer >= self.body_mats.len() || !material::body_names().any(|n| n == name) { return false; }
        if layer == 0 { return self.p.set(name, v); }
        self.set_body_own(layer, true);
        self.body_mats[layer].as_mut().is_some_and(|m| m.set(name, v))
    }
    /// Make that body of a named material (a position in `material::NAMED`).
    pub fn set_body_material(&mut self, layer: usize, i: usize) -> bool {
        if layer == 0 { return self.set_material(i); }
        let Some(n) = material::NAMED.get(i) else { return false };
        if layer >= self.body_mats.len() { return false; }
        let mut m = self.p.clone();
        for (k, v) in &self.scene_defaults() {
            match (*k, v) {
                ("absorb", Setting::Vec3(a)) => m.absorb = *a,
                ("ior" | "disp" | "nb", Setting::Num(x)) => { m.set(k, *x); }
                _ => {}
            }
        }
        n.apply(&mut m);
        m.mat = i as f64;
        self.body_mats[layer] = Some(Box::new(m));
        true
    }
    /// The material of every body, in the order of `bodies()`, when at least one of them has its own, or
    /// when the scene's material glows under pressure (it then glows also where one body presses on
    /// another, which only the tracer of several bodies sees). None otherwise: the scene is then drawn as
    /// one material.
    pub fn body_materials(&self) -> Option<Vec<&Settings>> {
        let bodies = self.bodies();
        // (a pressure glow that is animated counts while it is at nothing too: the way of drawing does not
        // change back and forth as it pulses)
        let pressed = self.p.piezo > 0.0 || self.tracks.get("piezo").is_some_and(|t| t.on);
        if bodies.len() < 2 || !(pressed || bodies.iter().any(|b| self.body_own(b[0]))) { return None; }
        Some(bodies.iter().map(|b| self.body_settings(b[0])).collect())
    }
    /// The sliders of the material of the body whose lowest layer this is (none unless it has its own).
    pub fn body_knobs(&self, layer: usize) -> Vec<Knob> {
        if !self.body_own(layer) { return Vec::new(); }
        ENGINE_KNOBS
            .iter()
            .copied()
            .filter(|k| matches!(k.0, "ior" | "disp"))
            .chain(material::knobs())
            .map(|(name, label, lo, hi, step, wave)| Knob {
                key: format!("B{}.{name}", layer + 1), id: name.into(), label: label.into(), lo, hi, step, shape: false, wave: wave && step < 1.0, r: KnobRef::Body(layer, name),
            })
            .collect()
    }
    /// `B2.rough=0.4`, `B2.material=gold`, `B2.own=0`: the material of the body whose lowest layer is the
    /// second. None if the name is not of that form.
    fn apply_body_arg(&mut self, name: &str, value: &str) -> Option<bool> {
        let b = name.as_bytes();
        if !(b.len() > 3 && b[0] == b'B' && (b'1'..=b'4').contains(&b[1]) && b[2] == b'.') { return None; }
        let layer = (b[1] - b'1') as usize;
        let num = value.trim().parse::<f64>().ok().filter(|v| v.is_finite());
        Some(match &name[3..] {
            "material" | "mat" => match material::named_index(value).or_else(|| value.trim().parse::<usize>().ok()) {
                Some(i) => self.set_body_material(layer, i),
                None => false,
            },
            "own" => match num {
                Some(v) if layer > 0 => { self.set_body_own(layer, v > 0.5); true }
                _ => false,
            },
            rest => match num {
                Some(v) => self.body_set(layer, rest, v),
                None => false,
            },
        })
    }

    /// Layers have changed places (`to[i]`: the new place of the layer that was at i, or None if it was
    /// taken away): each body's own material goes with its layer, and so do the tracks of its sliders.
    /// A layer that becomes the first brings its material with it as the scene's, and the layer that was
    /// first keeps the scene's old one as its own.
    pub(crate) fn move_bodies(&mut self, to: &[Option<usize>]) {
        let place = |i: usize| to.get(i).copied().flatten();
        let mut old = std::mem::take(&mut self.body_mats);
        if let Some(i) = (1..old.len()).find(|i| place(*i) == Some(0) && old[*i].is_some()) {
            let mine = old[i].take().expect("just looked");
            if place(0).is_some() { old[0] = Some(Box::new(self.p.clone())); }
            for name in material::body_names() {
                if let Some(v) = mine.get(name) { self.p.set(name, v); }
            }
        }
        for (i, m) in old.into_iter().enumerate() {
            if let (Some(m), Some(j)) = (m, place(i)) {
                if j > 0 && j < self.body_mats.len() { self.body_mats[j] = Some(m); }
            }
        }
        let keys: Vec<String> = self.tracks.keys().filter(|k| {
            let b = k.as_bytes();
            b.len() > 3 && b[0] == b'B' && (b'1'..=b'4').contains(&b[1]) && b[2] == b'.'
        }).cloned().collect();
        let mut moved = Vec::new();
        for k in keys {
            let Some(tr) = self.tracks.remove(&k) else { continue };
            if let Some(j) = place((k.as_bytes()[1] - b'1') as usize).filter(|j| *j > 0) { moved.push((format!("B{}{}", j + 1, &k[2..]), tr)); }
        }
        self.tracks.extend(moved);
    }

    /// How thick the shape is, where that is known without looking at it: a single tube's width.
    pub fn thickness_hint(&self) -> Option<f64> {
        if !self.plain() { return None; }
        let (f, p) = (self.family(), self.preset());
        (f.kind == Kind::Curve).then(|| 2.0 * p.tube.unwrap_or(f.tube) * self.shape().thick)
    }
    /// A material given as text: its id (`gold`), or its position in the list.
    fn material_arg(&mut self, v: &str) -> bool {
        match material::named_index(v).or_else(|| v.trim().parse::<usize>().ok()) {
            Some(i) => self.set_material(i),
            None => false,
        }
    }

    /// Pick another preset of the family that is up.
    pub fn choose(&mut self, preset: usize) {
        let fam = self.fam;
        let p = &self.families[fam].presets[preset];
        let (vals, rot) = (p.defaults(), p.rot);
        let st = &mut self.shapes[fam];
        st.preset = preset;
        st.vals = vals;
        st.travel = 0.0;
        st.orbit = None;
        st.e = None;
        if let Some(rot) = rot.filter(|_| self.stack.sel == 0) {
            self.p.rot = rot;
            self.set_r(&euler(rot[0], rot[1], rot[2]));
        }
        self.default_tracks();
        self.want = Some(Detail::Full);
    }
    /// Step to the next (+1) or previous (-1) preset.
    pub fn step_preset(&mut self, dir: i32) {
        let n = self.family().presets.len() as i32;
        self.choose(((self.shape().preset as i32 + dir + n) % n) as usize);
    }

    /// The sliders a shape animates unless told otherwise (made once per shape; after that they are the user's).
    pub(crate) fn default_tracks(&mut self) {
        let p = self.preset();
        let ids: Vec<&'static str> = match &p.animate {
            Some(a) => a.clone(),
            None if p.is_flow() => vec!["@travel"],
            None => Vec::new(),
        };
        let ks = self.shape_knobs();
        for (i, id) in ids.iter().enumerate() {
            if let Some(k) = ks.iter().find(|k| k.id == *id) {
                let stops = if *id == "@travel" { Some((0.0, 0.5)) } else { None };
                self.ensure_track(k, i, stops);
            }
        }
    }

    // ---- knobs ---------------------------------------------------------------------------------
    pub fn engine_knobs() -> Vec<Knob> {
        ENGINE_KNOBS
            .iter()
            .copied()
            // the material's sliders are kept with the materials themselves
            .chain(material::knobs())
            // and the studio's with the studio
            .chain(crate::stage::KNOBS.iter().copied())
            .map(|(name, label, lo, hi, step, wave)| Knob {
                key: format!("P.{name}"), id: name.into(), label: label.into(), lo, hi, step, shape: false, wave: wave && step < 1.0, r: KnobRef::P(name),
            })
            .collect()
    }
    pub fn engine_knob(name: &str) -> Knob {
        Self::engine_knobs().into_iter().find(|k| k.id == name).unwrap_or_else(|| panic!("no knob called {name}"))
    }
    /// The object's orientation as three angles: not animated (that is what the spin is for).
    pub fn orient_knobs() -> Vec<Knob> {
        [("turn x °", 180.0), ("turn y °", 90.0), ("turn z °", 180.0)]
            .iter()
            .enumerate()
            .map(|(i, &(label, lim))| Knob { key: format!("R.{i}"), id: format!("R.{i}"), label: label.into(), lo: -lim, hi: lim, step: 0.5, shape: false, wave: false, r: KnobRef::Orient(i) })
            .collect()
    }
    /// Every slider the current shape has: its own numbers, then the built-in ones.
    pub fn shape_knobs(&self) -> Vec<Knob> {
        self.layer_knobs(self.stack.sel)
    }
    /// The same for the shape of the layer at this place in the list.
    pub fn layer_knobs(&self, layer: usize) -> Vec<Knob> {
        let (f, p) = (self.layer_family(layer), self.layer_preset(layer));
        let pre = format!("{}{}/{}/", Self::layer_prefix(layer), f.id, p.id);
        let mut out = Vec::new();
        let mut add = |id: &str, label: &str, lo: f64, hi: f64, step: f64, r: KnobRef| {
            out.push(Knob { key: format!("{pre}{id}"), id: id.into(), label: label.into(), lo, hi, step, shape: true, wave: step < 1.0, r: KnobRef::Of(layer, Box::new(r)) });
        };
        for (i, d) in p.params.iter().enumerate() {
            add(d.id, d.label, d.lo, d.hi, d.step, KnobRef::Param(i));
        }
        if f.kind == Kind::Curve {
            add("@thick", "thickness", 0.4, 3.0, 0.02, KnobRef::Thick);
            add("@melt", "melt", 0.0, 0.03, 0.0005, KnobRef::Melt);
            if !p.fixed_length { add("@len", "length", 0.1, 3.0, 0.02, KnobRef::Len); }
            if p.is_flow() { add("@travel", "travel", 0.0, TRAVEL, 0.005, KnobRef::Travel); }
            add("@beads", "beads", 0.0, 0.6, 0.01, KnobRef::Beads);
        } else {
            add("@puff", "inflate", -0.06, 0.06, 0.001, KnobRef::Puff);
        }
        out
    }
    /// Every knob that can carry a track: the engine's, then every layer's shape and placement, then
    /// the options of the layers' combinations.
    pub fn knob_list(&self) -> Vec<Knob> {
        let mut v = Self::engine_knobs();
        v.extend(self.layer_knob_list());
        // ... and the materials of the bodies that have one of their own
        for layer in 1..self.body_mats.len() { v.extend(self.body_knobs(layer)); }
        v
    }
    pub fn get(&self, r: &KnobRef) -> f64 {
        match r {
            KnobRef::P(name) => crate::camera::get(&self.p, name).or_else(|| self.p.get(name)).unwrap_or(0.0),
            KnobRef::Orient(i) => euler_of(&self.r)[*i],
            KnobRef::Of(layer, r) => self.layer_shape(*layer).map_or(0.0, |st| st.knob(r)),
            KnobRef::Place(layer, c) => self.stack.place(*layer, *c),
            KnobRef::Combo(mask, o) => self.stack.opt(*mask, *o).unwrap_or(0.0),
            KnobRef::Body(layer, name) => self.body_settings(*layer).get(name).unwrap_or(0.0),
            r => self.shape().knob(r),
        }
    }
    pub fn set(&mut self, r: &KnobRef, v: f64) {
        match r {
            KnobRef::P(name) => { if !crate::camera::set(&mut self.p, name, v) { self.p.set(name, v); } }
            KnobRef::Orient(i) => {
                let mut e = euler_of(&self.r);
                e[*i] = v;
                self.set_r(&euler(e[0], e[1], e[2]));
            }
            KnobRef::Of(layer, r) => { if let Some(st) = self.layer_shape_mut(*layer) { st.set_knob(r, v); } }
            KnobRef::Place(layer, c) => self.stack.set_place(*layer, *c, v),
            KnobRef::Combo(mask, o) => { self.stack.set_opt(*mask, *o, v); }
            // (a slider of a material the body no longer has does nothing)
            KnobRef::Body(layer, name) => { if self.body_own(*layer) { self.body_set(*layer, name, v); } }
            r => self.shape_mut().set_knob(r, v),
        }
    }
    /// A slider was moved by hand. `quick`: it is still being dragged.
    pub fn user_set(&mut self, k: &Knob, v: f64, quick: bool) {
        self.set(&k.r, v);
        self.after_user(k, quick);
    }
    /// What follows from a knob having been changed by hand (its value, or the stops of its track).
    pub fn after_user(&mut self, k: &Knob, quick: bool) {
        if k.shape {
            // the animation rebuilds the shape every frame when it changes the shape; otherwise a moved slider must
            if !(self.playing && self.shape_moving()) {
                self.want = Some(if quick { Detail::Quick } else { Detail::Full });
            }
        }
        match k.r {
            KnobRef::P("fnear") | KnobRef::P("ffar") | KnobRef::P("focus") | KnobRef::P("band") | KnobRef::P("tiltX") | KnobRef::P("tiltY") => self.flash_planes = true,
            KnobRef::P("speed") => self.rephase_all(),
            _ => {}
        }
    }

    // ---- orientation ---------------------------------------------------------------------------
    /// The spin is a function of one angle, so that it comes back exactly to where it started after a
    /// whole turn (which is what lets a video loop). Turntable: straight round the vertical. Tumble: the
    /// same, rocking too.
    fn spin_m(&self, th: f64) -> Mat3 {
        let y = axis_angle(0.0, 1.0, 0.0, th);
        if self.p.spin_mode > 0.5 { return y; }
        mul(&axis_angle(1.0, 0.0, 0.0, 0.5 * th.sin()), &mul(&axis_angle(0.0, 0.0, 1.0, 0.35 * (2.0 * th).sin()), &y))
    }
    pub fn update_r(&mut self) {
        self.r = mul(&self.spin_m(self.theta), &self.r0);
    }
    /// Show the object in this orientation from now on (the spin carries on from there).
    pub fn set_r(&mut self, rn: &Mat3) {
        self.r = orthonormalise(rn);
        self.r0 = mul(&transpose(&self.spin_m(self.theta)), &self.r);
    }
    /// The camera can roll about its line of sight. The roll is applied to the scene instead: the shape,
    /// and the room's lights with it, are turned the other way about that axis.
    pub fn view_r(&self) -> Mat3 {
        // (a camera raised above the floor sees the whole room tilted towards it, the shape with it)
        if self.p.elev != 0.0 { return mul(&self.room_m(), &self.r); }
        if self.p.roll != 0.0 { mul(&axis_angle(0.0, 0.0, 1.0, -self.p.roll * DEG), &self.r) } else { self.r }
    }
    /// Apply a rotation given about the screen's axes to the shape as it appears through the rolled camera.
    pub fn turn(&mut self, s: &Mat3) {
        if self.p.elev != 0.0 {
            // the turn is about the screen's axes: taken into the room, applied, and brought back
            let m = self.room_m();
            let rn = mul(&transpose(&m), &mul(s, &mul(&m, &self.r)));
            self.set_r(&rn);
            return;
        }
        let z = self.p.roll * DEG;
        let rn = if self.p.roll != 0.0 {
            mul(&axis_angle(0.0, 0.0, 1.0, z), &mul(s, &mul(&axis_angle(0.0, 0.0, 1.0, -z), &self.r)))
        } else {
            mul(s, &self.r)
        };
        self.set_r(&rn);
    }
    /// The orientation this shape first appears in.
    pub fn preset_rot(&self) -> [f64; 3] {
        if let Some(r) = self.layer_preset(0).rot { return r; }
        match self.scene_defaults().iter().find(|(k, _)| *k == "rot") {
            Some((_, Setting::Vec3(r))) => *r,
            _ => [0.0; 3],
        }
    }
    pub fn reset_orientation(&mut self) {
        let rot = self.preset_rot();
        self.theta = 0.0;
        self.set_r(&euler(rot[0], rot[1], rot[2]));
    }
    /// Put the camera and the object back the way the shape first appears.
    pub fn reset_view(&mut self) {
        let d = self.scene_defaults();
        let num = |k: &str, base: f64| match d.iter().find(|(n, _)| *n == k) { Some((_, Setting::Num(x))) => *x, _ => base };
        let base = Settings::default();
        self.p.dist = num("dist", base.dist);
        crate::camera::set(&mut self.p, "fov", num("fov", base.fov));
        self.p.shift = [0.0, 0.0];
        self.p.roll = num("roll", 0.0);
        self.reset_orientation();
    }
    /// Space: start or stop the spin.
    pub fn toggle_spin(&mut self) {
        let d = self.scene_defaults();
        let own = match d.iter().find(|(n, _)| *n == "spin") { Some((_, Setting::Num(x))) if *x != 0.0 => *x, _ => Settings::default().spin };
        self.p.spin = if self.p.spin != 0.0 { 0.0 } else { own };
    }

    // ---- lights --------------------------------------------------------------------------------
    // A light whose direction in the lights' own frame is a appears to the camera in direction  base . turned . a
    pub fn light_base(&self) -> Mat3 {
        if self.p.lights > 0.5 { self.view_r() } else if self.p.elev != 0.0 { self.room_m() } else { axis_angle(0.0, 0.0, 1.0, -self.p.roll * DEG) }
    }
    pub fn light_turn(&self) -> Mat3 {
        yxz(self.p.lyaw * DEG + self.lphi, self.p.lpitch * DEG, self.p.lroll * DEG)
    }
    pub fn light_m(&self) -> Mat3 {
        mul(&self.light_base(), &self.light_turn())
    }
    /// Turn what the lights look like from the camera by a rotation about the screen's axes (the ball being dragged).
    pub fn turn_lights(&mut self, s: &Mat3) {
        // the lights appear at  base . turned . a ; turning what appears by S means  turned' = base^T . S . base . turned
        let b = self.light_base();
        let a = yxz_of(&mul(&transpose(&b), &mul(s, &mul(&b, &self.light_turn()))));
        let deg = |v: f64| ((v / DEG) % 360.0 + 360.0) % 360.0;
        self.p.lyaw = deg(a[0] - self.lphi);
        self.p.lpitch = deg(a[1]);
        self.p.lroll = deg(a[2]);
    }
    pub fn reset_lights(&mut self) {
        self.p.lyaw = 0.0;
        self.p.lpitch = 0.0;
        self.p.lroll = 0.0;
        self.lphi = 0.0;
    }

    // ---- the lights themselves -----------------------------------------------------------------
    /// The lights of the first layer's family (the one the scene's look belongs to), as they stand.
    pub fn rig(&self) -> &Rig { &self.rigs[self.layer_fam(0)] }
    /// The same, to change them. Whoever watches `lights_rev` takes it that they have changed.
    pub fn rig_mut(&mut self) -> &mut Rig {
        self.lights_rev += 1;
        let look = self.layer_fam(0);
        &mut self.rigs[look]
    }
    /// The lights as the family ships them.
    pub fn rig_default(&self) -> Rig {
        let f = self.look_family();
        Rig { lights: f.lights.clone(), ambient: f.ambient }
    }
    pub fn rig_is_default(&self) -> bool {
        let (f, r) = (self.look_family(), self.rig());
        r.lights == f.lights && r.ambient == f.ambient
    }
    /// Put the family's own lights back.
    pub fn reset_rig(&mut self) {
        let d = self.rig_default();
        *self.rig_mut() = d;
    }
    /// Choose the environment picture (an .exr or .hdr file), or with None go back to the lights alone.
    pub fn set_hdri(&mut self, path: Option<String>) {
        self.hdri = path.map(|p| p.trim().to_string()).filter(|p| !p.is_empty());
        self.lights_rev += 1;
    }
    /// The direction a light at this azimuth and elevation (degrees) shines from, in the lights' own frame.
    pub fn light_dir(az: f64, el: f64) -> [f64; 3] {
        let ce = (el * DEG).cos();
        [(az * DEG).sin() * ce, (el * DEG).sin(), -(az * DEG).cos() * ce]
    }
    /// Where a light at this azimuth and elevation shows on the picture of the lighting (`env::preview_dir`
    /// says how that picture is laid out): fractions of its width and height from the top left.
    pub fn light_on_preview(&self, az: f64, el: f64) -> (f64, f64) {
        crate::env::preview_uv(apply(&self.light_m(), Self::light_dir(az, el)))
    }
    /// The other way: the azimuth and elevation (degrees) a light must have to show at that point.
    pub fn light_at_preview(&self, u: f64, v: f64) -> (f64, f64) {
        let d = apply(&transpose(&self.light_m()), crate::env::preview_dir(u, v));
        (d[0].atan2(-d[2]) / DEG, d[1].clamp(-1.0, 1.0).asin() / DEG)
    }
    /// Is this one of the names `apply_arg` takes for the lights or the environment picture?
    pub fn is_light_arg(name: &str) -> bool {
        matches!(name, "rig" | "ambient" | "hdri") || Self::light_arg(name).is_some()
    }
    /// `light3` or `light3.az`: which light (counted from 1) and which part of it.
    fn light_arg(name: &str) -> Option<(usize, &str)> {
        let rest = name.strip_prefix("light")?;
        let (num, field) = rest.split_once('.').unwrap_or((rest, ""));
        if num.is_empty() || !num.bytes().all(|b| b.is_ascii_digit()) { return None; }
        num.parse::<usize>().ok().filter(|n| *n >= 1).map(|n| (n, field))
    }
    /// The lights by name: `rig`, `ambient`, `light<n>`, `light<n>.<part>` (main.rs lists the forms).
    fn apply_light_arg(&mut self, name: &str, value: &str) -> bool {
        let nums = |n: usize| -> Option<Vec<f64>> {
            let v: Vec<f64> = value.split(',').map(|s| s.trim().parse::<f64>().ok().filter(|x| x.is_finite())).collect::<Option<_>>()?;
            (v.len() == n).then_some(v)
        };
        match name {
            "rig" if value == "default" => { self.reset_rig(); true }
            "rig" => match Rig::lights_from_text(value) {
                Some(l) => { self.rig_mut().lights = l; true }
                None => false,
            },
            "ambient" => match nums(3) {
                Some(v) => { self.rig_mut().ambient = [v[0], v[1], v[2]]; true }
                None => false,
            },
            _ => {
                let Some((n, field)) = Self::light_arg(name) else { return false };
                let have = self.rig().lights.len();
                if field.is_empty() {
                    // a whole light: take it away, replace it, or add one after the last
                    if matches!(value, "off" | "none" | "") {
                        if n > have { return false; }
                        self.rig_mut().lights.remove(n - 1);
                        return true;
                    }
                    let Some(l) = light_from_text(value) else { return false };
                    if n > have + 1 { return false; }
                    let lights = &mut self.rig_mut().lights;
                    if n == have + 1 { lights.push(l); } else { lights[n - 1] = l; }
                    return true;
                }
                if n > have { return false; }
                if field == "rgb" {
                    let Some(v) = nums(3) else { return false };
                    self.rig_mut().lights[n - 1].rgb = [v[0], v[1], v[2]];
                    return true;
                }
                if !matches!(field, "az" | "el" | "w" | "h" | "soft") { return false; }
                let Some(v) = nums(1) else { return false };
                let l = &mut self.rig_mut().lights[n - 1];
                match field {
                    "az" => l.az = v[0],
                    "el" => l.el = v[0],
                    "w" => l.w = v[0],
                    "h" => l.h = v[0],
                    _ => l.soft = v[0],
                }
                true
            }
        }
    }

    // ---- camera gestures -----------------------------------------------------------------------
    /// Dragging on the picture, by (dx, dy) screen points.
    pub fn drag_turn(&mut self, dx: f64, dy: f64) {
        self.turn(&mul(&axis_angle(0.0, 1.0, 0.0, dx * 0.006), &axis_angle(1.0, 0.0, 0.0, dy * 0.006)));
    }
    /// `height`: the picture's height in the same units as dx, dy.
    pub fn drag_pan(&mut self, dx: f64, dy: f64, height: f64) {
        let k = 2.0 / height;
        self.p.shift = [self.p.shift[0] - dx * k, self.p.shift[1] + dy * k];
    }
    pub fn drag_roll(&mut self, dx: f64) {
        self.p.roll = ((self.p.roll + dx * 0.3 + 540.0) % 360.0) - 180.0;
    }
    /// The wheel, in the browser's units (about 100 a notch). `lens`: change the lens instead, moving
    /// the camera to keep the shape the same size.
    pub fn wheel(&mut self, d: f64, lens: bool) {
        if lens {
            let was = (self.p.fov * DEG / 2.0).tan();
            self.p.fov = (self.p.fov * (d * 0.001).exp()).clamp(6.0, 80.0);
            self.p.dist = (self.p.dist * was / (self.p.fov * DEG / 2.0).tan()).clamp(1.5, 60.0);
            if self.p.focal > 0.0 { self.p.focal = crate::camera::focal_of(self.p.fov, self.p.sensor); }
            crate::camera::sync(&mut self.p);
        } else {
            self.p.dist = (self.p.dist * (d * 0.001).exp()).clamp(1.5, 60.0);
        }
    }
    /// Focus on glass found `t` away along the ray through the middle of traced pixel (x, y) (y counted
    /// upward), in a picture `w` x `h`: the sharp band keeps its thickness and is centred on that glass.
    pub fn focus_at(&mut self, t: f64, x: f64, y: f64, w: f64, h: f64) -> f64 {
        // (an ordinary lens with straight lines and a level plane of focus is worked out here, as it always was)
        if !self.camera().plain() {
            return self.focus_on([(2.0 * x - w) / h, (2.0 * y - h) / h], t).unwrap_or(0.5 * (self.p.fnear + self.p.ffar));
        }
        let tf = (self.p.fov * DEG / 2.0).tan();
        let u = (2.0 * x - w) / h + self.p.shift[0];
        let v = (2.0 * y - h) / h + self.p.shift[1];
        let e = t / ((u * tf).powi(2) + (v * tf).powi(2) + 1.0).sqrt() - self.p.dist;
        let band = (self.p.ffar - self.p.fnear).abs();
        self.p.fnear = e - band / 2.0;
        self.p.ffar = e + band / 2.0;
        self.flash_planes = true;
        e
    }

    // ---- animation -----------------------------------------------------------------------------
    /// The knobs whose tracks are switched on.
    pub fn active(&self) -> Vec<Knob> {
        self.knob_list().into_iter().filter(|k| self.tracks.get(&k.key).is_some_and(|t| t.on)).collect()
    }
    /// Beads that are travelling: their pace matters to a loop.
    pub fn beads(&self) -> Option<f64> {
        let st = self.shape();
        (self.family().kind == Kind::Curve && st.beads > 0.0).then_some(st.bead_period)
    }
    pub fn shape_moving(&self) -> bool {
        self.active().iter().any(|k| k.shape) || !self.beads_all().is_empty()
    }
    /// Everything that moves, with its period, fitted to a loop.
    pub fn loop_plan(&self) -> LoopPlan {
        let sp = self.p.speed;
        let mut items: Vec<(String, f64, bool)> = self.active().iter().map(|k| (self.knob_title(k), self.tracks[&k.key].period / sp, false)).collect();
        if self.p.spin > 0.0 { items.push(("spin".into(), TAU / (self.p.spin * sp), true)); }
        if self.p.lspin > 0.0 { items.push(("light spin".into(), TAU / (self.p.lspin * sp), true)); }
        for b in self.beads_all() { items.push(("beads".into(), b / sp, false)); }
        anim::loop_plan(&items, self.p.loop_len)
    }
    pub fn locked(&self) -> bool {
        self.p.lock > 0.5 || self.rendering
    }
    /// A period as it is actually being played.
    pub fn period(&self, p: f64) -> f64 {
        if self.locked() { anim::snap(self.loop_plan().l, p) } else { p }
    }
    /// The same for a rate of turning.
    fn turn_rate(&self, r: f64) -> f64 {
        if r > 0.0 { TAU / self.period(TAU / r) } else { 0.0 }
    }
    /// How many times something with this period has gone round by now.
    pub fn cycles(&self, p: f64) -> f64 {
        self.clock / self.period(p / self.p.speed)
    }
    /// Give a knob a track if it has none: a swing about where the slider is, a quarter of its range wide.
    pub fn ensure_track(&mut self, k: &Knob, i: usize, stops: Option<(f64, f64)>) {
        if self.tracks.contains_key(&k.key) { return; }
        let h = 0.125 * (k.hi - k.lo);
        // (centred on where the knob rests, not on wherever an earlier animation left it: otherwise
        // every time tracks are made again after a stop, the range would move on by part of a trip)
        let m = (k.hi - h).min((k.lo + h).max(self.resting(k)));
        let (from, to) = stops.unwrap_or((m - h, m + h));
        let period = (14.0 * (1.0 + 0.29 * i as f64) * 10.0).round() / 10.0;
        self.tracks.insert(k.key.clone(), Track { on: true, period, from, to, phase: 0.0, ease: -1, at: 2 });
        self.rephase(k);
    }
    /// The value a knob rests at: the one it had before the animation moved it, if it still holds what
    /// the animation last gave it; otherwise the value it holds.
    pub fn resting(&self, k: &Knob) -> f64 {
        let now = self.get(&k.r);
        match (self.moved.get(&k.key), self.rest.get(&k.key)) {
            (Some(m), Some(r)) if *m == now => *r,
            _ => now,
        }
    }
    /// Choose where in its trip a track is, so that it carries on from the value its slider shows now.
    pub fn rephase(&mut self, k: &Knob) {
        let Some(tr) = self.tracks.get(&k.key) else { return };
        let per = self.period(tr.period / self.p.speed);
        let (value, clock, all) = (self.get(&k.r), self.clock, self.p.easing());
        if let Some(tr) = self.tracks.get_mut(&k.key) { tr.rephase(value, clock, per, all); }
    }
    pub fn rephase_all(&mut self) {
        for k in self.active() { self.rephase(&k); }
    }
    /// The wave button beside a slider: switch its track on or off.
    pub fn toggle_wave(&mut self, k: &Knob) {
        match self.tracks.get_mut(&k.key) {
            Some(tr) => {
                tr.on = !tr.on;
                if tr.on { self.rephase(k); }
            }
            None => {
                let n = self.active().len();
                self.ensure_track(k, n, None);
            }
        }
    }
    /// Put every animated knob where it belongs at the present time; true if any of them changes the shape.
    pub fn apply_time(&mut self) -> bool {
        let mut shape = false;
        let l = if self.locked() { self.loop_plan().l } else { 0.0 };
        let all = self.p.easing();
        for k in self.active() {
            let tr = &self.tracks[&k.key];
            let p = tr.period / self.p.speed;
            let per = if l > 0.0 { anim::snap(l, p) } else { p };
            let v = tr.value(self.clock, per, all).max(k.lo).min(k.hi);
            let rest = self.resting(&k);
            self.rest.insert(k.key.clone(), rest);
            self.set(&k.r, v);
            let now = self.get(&k.r);
            self.moved.insert(k.key.clone(), now);
            if k.shape { shape = true; }
        }
        shape
    }
    pub fn set_playing(&mut self, on: bool) {
        if on == self.playing { return; }
        self.playing = on;
        if on { self.rephase_all(); } else { self.want = Some(Detail::Full); }
    }
    /// Move everything on by dt seconds.
    pub fn tick(&mut self, dt: f64) {
        if self.p.spin > 0.0 {
            self.theta += self.turn_rate(self.p.spin * self.p.speed) * dt;
            self.update_r();
        }
        if self.p.lspin > 0.0 {
            self.lphi += self.turn_rate(self.p.lspin * self.p.speed) * dt;
        }
        if self.playing {
            self.clock += dt;
            let shape = self.apply_time();
            if shape || !self.beads_all().is_empty() { self.want = Some(Detail::Anim); }
        }
    }
    /// Show everything as it is at the present clock (after setting the clock by hand).
    pub fn apply_clock(&mut self, hq: bool) {
        self.apply_time();
        self.want = Some(if hq { Detail::Hq } else { Detail::Anim });
    }

    // ---- video ---------------------------------------------------------------------------------
    /// How many frames one whole loop is.
    pub fn video_frames(&self, fps: u32) -> u32 {
        (js_round(self.loop_plan().l * fps as f64) as u32).max(1)
    }
    /// Begin rendering one loop: everything is now played exactly as it loops, and a lens gets
    /// `lens_video` rays if that is more than the live setting.
    pub fn video_begin(&mut self, lens_video: u32) -> VideoKeep {
        let plan = self.loop_plan();
        let sp = self.p.speed;
        let turns = |r: f64| if r > 0.0 { js_round(plan.l * r * sp / TAU).max(1.0) } else { 0.0 };
        let keep = VideoKeep {
            playing: self.playing, clock: self.clock, theta: self.theta, lphi: self.lphi,
            vals: self.active().into_iter().map(|k| { let v = self.get(&k.r); (k, v) }).collect(),
            moved: self.moved.clone(),
            lens_live: self.p.lens_n, turns_spin: turns(self.p.spin), turns_lspin: turns(self.p.lspin), plan,
        };
        self.playing = false;
        self.rendering = true;
        if self.p.aperture > 0.0 && self.p.dof < 0.5 && lens_video as f64 > self.p.lens_n { self.p.lens_n = lens_video as f64; }
        keep
    }
    /// Put everything where it is at frame `i` of `whole` (`i` may fall between frames: motion blur looks
    /// at several moments within one). True if the shape must be built again for it.
    pub fn video_frame(&mut self, keep: &VideoKeep, i: f64, whole: u32) -> bool {
        let u = i / whole as f64;
        self.clock = u * keep.plan.l;
        self.theta = keep.theta + TAU * keep.turns_spin * u;
        self.update_r();
        self.lphi = keep.lphi + TAU * keep.turns_lspin * u;
        let shape = self.apply_time();
        i == 0.0 || shape || !self.beads_all().is_empty()
    }
    /// Put everything back the way it was before the render.
    pub fn video_end(&mut self, keep: VideoKeep) {
        self.p.lens_n = keep.lens_live;
        self.clock = keep.clock;
        self.theta = keep.theta;
        self.lphi = keep.lphi;
        for (k, v) in &keep.vals { self.set(&k.r, *v); }
        // (the knobs are back where the animation had left them: they rest where they rested before)
        self.moved = keep.moved;
        self.rendering = false;
        self.update_r();
        self.want = Some(Detail::Full);
        self.cut = true;
        if keep.playing { self.playing = true; }
    }
    /// The lens rays a video would use right now (0 = no lens, or focus that is not traced through one).
    pub fn video_lens(&self, lens_video: u32) -> u32 {
        if self.p.aperture > 0.0 && self.p.dof < 0.5 { js_round(self.p.lens_n.max(lens_video as f64)) as u32 } else { 0 }
    }

    // ---- building the shape --------------------------------------------------------------------
    /// Work out what the GPU has to build for the shape as it stands. For a curve this is where the
    /// time goes: the orbit, its placement, and the cut into equal steps.
    pub fn shape_job(&mut self, mode: Detail, res_override: Option<u32>) -> Result<Job, String> {
        let phase = self.cycles(self.shape().bead_period);
        let fam_i = self.fam;
        let fam = &self.families[fam_i];
        let st = &mut self.shapes[fam_i];
        let p = &fam.presets[st.preset];
        let moving = matches!(mode, Detail::Anim | Detail::Hq);
        let res = valid_res(res_override.unwrap_or(match mode {
            Detail::Anim => st.anim_res.min(st.res),
            Detail::Quick => RES_DRAG.min(st.res),
            _ => st.res,
        }) as f64);
        let shape = match &p.shape {
            Shape::Mesh { file } => {
                st.meta.length = 0.0;
                ShapeJob::Mesh(crate::presets::models::job(p, st, file))
            }
            Shape::Field { .. } => {
                st.meta.info = String::new();
                st.meta.length = 0.0;
                ShapeJob::Field { uniforms: field::uniforms(p, &st.vals, res, st.puff) }
            }
            curve_shape => {
                let tube = p.tube.unwrap_or(fam.tube) * st.thick;
                // a melt wider than the tube would dissolve it
                let melt = st.melt.min(0.45 * tube);
                let mut place = PlaceOpts { centre: p.centre, fit: p.fit, axes: p.axes, align: p.align.unwrap_or(fam.align), e: None };
                // while animating, do not let the face-on turn swing about
                if moving { place.e = st.e; }
                let ds = 0.75 * 2.0 * HB / res as f64;
                let mut pose = None;
                let raw: RawCurve = match curve_shape {
                    Shape::Flow(flow) => {
                        // The tube is a window on a longer stretch of the orbit; `travel` slides the window
                        // along it, so the glass grows at one end and shrinks at the other like a train on a
                        // track. The stretch is worked out once and kept until the shape's numbers change, and
                        // the tube is placed by the whole stretch rather than by the window, so that it does
                        // not jump about as it slides.
                        let n = (js_round(flow.steps as f64 * st.len) as usize).max(40);
                        let key = format!("{}|{:?}|{}", p.id, st.vals.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), n);
                        let exact = matches!(mode, Detail::Full | Detail::Hq);
                        if !st.orbit.as_ref().is_some_and(|c| c.key == key && !(exact && c.quick)) {
                            let line = curve::orbit_line(flow, &st.vals, js_round(n as f64 * (1.0 + TRAVEL)) as usize, !exact);
                            st.orbit = Some(OrbitCache { key, quick: !exact, line, pose: None });
                        }
                        let c = st.orbit.as_mut().unwrap();
                        let have = c.line.len() / 3;
                        let off = (js_round(st.travel * n as f64) as i64).min(have as i64 - n as i64).max(0) as usize;
                        let end = (off + n).min(have);
                        if c.pose.is_none() { c.pose = Some(curve::placement(&[&c.line[..]], &place)); }
                        pose = c.pose.clone();
                        RawCurve { lines: vec![c.line[off * 3..end * 3].to_vec()], closed: false }
                    }
                    Shape::Curve(f) => f(&st.vals, &CurveOpts { len: st.len, quick: matches!(mode, Detail::Quick | Detail::Anim) }),
                    Shape::Field { .. } | Shape::Mesh { .. } => unreachable!(),
                };
                // Beads: lumps of thickness that travel along the tube while the animation plays (a plain
                // tube sliding along its own length would look perfectly still). A flowing tube also tapers
                // at its two ends.
                let (beads, taper, closed) = (st.beads, p.is_flow(), raw.closed);
                let own = p.radius.as_deref();
                let composed = move |u: f64, s: f64, total: f64| -> f64 {
                    let mut r = own.map_or(1.0, |f| f(u));
                    if beads > 0.0 {
                        let count = if closed { js_round(total / 0.8).max(1.0) } else { total / 0.8 };
                        r *= 1.0 + beads * curve::jsmath::sin(2.0 * std::f64::consts::PI * (u * count - phase));
                    }
                    // (away from both ends the factor is exactly 1: nothing to work out)
                    if taper && (s < 0.35 || total - s < 0.35) { r *= (1.0f64.min(s / 0.35).min((total - s) / 0.35)).sqrt().max(0.2); }
                    r
                };
                let plain = move |u: f64, _s: f64, _t: f64| -> f64 { own.map_or(1.0, |f| f(u)) };
                let radius: Option<&(dyn Fn(f64, f64, f64) -> f64 + Sync)> =
                    if beads > 0.0 || taper { Some(&composed) } else if own.is_some() { Some(&plain) } else { None };
                let lines: Vec<&[f64]> = raw.lines.iter().map(|l| &l[..]).collect();
                let c = curve::prep_curve(&lines, &PrepOpts { tube, melt, ds, closed, radius, place, pose });
                if c.n < 2 { return Err(format!("{}: preset \"{}\" produced no curve", fam.name, p.id)); }
                if !moving { st.e = c.pose.e; }
                st.meta.length = c.length;
                st.meta.info = p.info.as_ref().map(|f| f(&raw, &st.vals)).unwrap_or_default();
                ShapeJob::Curve { curve: c, melt }
            }
        };
        Ok(Job { family: fam_i, preset: st.preset, res, shape })
    }

    // ---- by name -------------------------------------------------------------------------------
    /// Change one thing by name, as the command line and saved setups do. False if the name means nothing.
    pub fn apply_arg(&mut self, name: &str, value: &str) -> bool {
        let num = value.trim().parse::<f64>().ok().filter(|v| v.is_finite());
        if let Some(done) = self.apply_body_arg(name, value) { return done; }
        if let Some(done) = self.apply_layer_arg(name, value) { return done; }
        match name {
            "scene" => match self.families.iter().position(|f| f.id == value) {
                Some(i) => { self.set_scene(i); true }
                None => false,
            },
            "material" | "mat" => self.material_arg(value),
            "studio" => self.studio_arg(value),
            "preset" => match self.family().presets.iter().position(|p| p.id == value) {
                Some(i) => { self.choose(i); true }
                None => false,
            },
            "rx" | "ry" | "rz" => match num {
                Some(v) => {
                    let mut e = euler_of(&self.r);
                    e[match name { "rx" => 0, "ry" => 1, _ => 2 }] = v;
                    self.set_r(&euler(e[0], e[1], e[2]));
                    true
                }
                None => false,
            },
            "rot" => {
                if !self.p.set_text("rot", value) { return false; }
                let r = self.p.rot;
                self.set_r(&euler(r[0], r[1], r[2]));
                true
            }
            "thick" | "len" | "melt" | "puff" | "travel" | "beads" | "beadPeriod" | "res" | "animRes" => match num {
                Some(v) => {
                    let st = self.shape_mut();
                    match name {
                        "thick" => st.thick = v,
                        "len" => st.len = v,
                        "melt" => st.melt = v,
                        "puff" => st.puff = v,
                        "travel" => st.travel = v,
                        "beads" => st.beads = v,
                        "beadPeriod" => st.bead_period = v,
                        "res" => st.res = valid_res(v),
                        _ => st.anim_res = valid_res(v),
                    }
                    self.want = Some(Detail::Full);
                    true
                }
                None => false,
            },
            "hdri" => {
                self.set_hdri(if value == "none" { None } else { Some(value.to_string()) });
                true
            }
            "model" => self.set_model(value),
            _ if Self::is_light_arg(name) => self.apply_light_arg(name, value),
            _ if name.starts_with("p.") => match (self.preset().param_index(&name[2..]), num) {
                (Some(i), Some(v)) => {
                    self.shape_mut().vals[i] = v;
                    self.want = Some(Detail::Full);
                    true
                }
                _ => false,
            },
            _ => self.camera_arg(name, value) || self.p.set_text(name, value),
        }
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}
