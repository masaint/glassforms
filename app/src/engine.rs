//! The engine: the program's state, the shape on the GPU, and the picture, kept in step with each other.
//! It needs no window: the live app, stills, benchmarks, video renders and tests all drive this.

use std::cell::Cell;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use gf_core::env::{self, Sh9};
use gf_core::material;
use gf_core::math::{axis_angle, f32s, DEG};
use gf_core::presets::Shape;
use gf_core::state::{Detail, ShapeJob, State};

use crate::bake::Baker;
use crate::bubbles::Bubbler;
use crate::scatter::Scatter;
use crate::combine::LayerWork;
use crate::dof::Focus;
use crate::gpu::{Gpu, R};
use crate::hdri::{self, Hdri, Loading, Maker, ShowU};
use crate::render::{LightFrom, LightU, Look, Renderer, TraceU, Variant, MAX_ACCUM};

/// The colour of the clay, and how strong its sheen is.
const CLAY: [f32; 4] = [0.74, 0.70, 0.66, 1.0];
/// How much each order of the lighting's nine terms counts in the clay's sheen: a soft reflection,
/// not as soft as the light a matte surface gathers (`env::SH_MATTE`).
const SHEEN: [f64; 3] = [1.0, 0.8, 0.5];
/// The light the clay gathers, averaged over every way a surface can face; and how far the colour of the
/// lighting is taken towards grey for it.
const CLAY_LIGHT: f64 = 0.30;
const CLAY_GREY: f64 = 0.45;
/// How far the clay's light and dark are drawn apart (see trace_light.wgsl).
const CLAY_CONTRAST: f32 = 1.2;
/// The most a clay surface facing the brightest way may come to before the tone map. Above this it
/// burns out to white and the form is lost.
const CLAY_PEAK: f64 = 0.9;
/// Frames after which a picture of clay has settled: there is nothing to average but the outlines.
pub const MAX_ACCUM_CLAY: u32 = 96;

/// How much of the lighting the clay is given (`matte` is what a matte surface gathers, as the
/// shader's nine terms): all of it, unless the surface facing the brightest way would then burn out.
/// Lights set round a shape for glass light a matte surface almost evenly, and drawing light and dark
/// apart by CLAY_CONTRAST leaves the brightest side well short of white. Lights that already differ a
/// great deal from one side to another (a gem's: close, small and bright) would come out white wherever
/// the shape faces them, its facets lost; those are turned down until the brightest-facing surface is
/// at CLAY_PEAK, keeping the differences between one facet and the next.
/// The light a white matte surface gathers facing the brightest way (luminance), looked for over
/// directions spread evenly round the sphere.
/// How many times finer each way the lights' map is painted for its softened copies than for the glass.
const FINE: usize = 4;

fn matte_peak(matte: &[[f32; 4]; 9]) -> f64 {
    let lum = |c: [f64; 3]| 0.2126 * c[0].max(0.0) + 0.7152 * c[1].max(0.0) + 0.0722 * c[2].max(0.0);
    const N: usize = 600;
    let mut peak = 0.0f64;
    for i in 0..N {
        let z = 1.0 - 2.0 * (i as f64 + 0.5) / N as f64;
        let r = (1.0 - z * z).sqrt();
        let turn = i as f64 * 2.399963229728653;
        peak = peak.max(lum(env::sh_eval(matte, [r * turn.cos(), r * turn.sin(), z])));
    }
    peak
}

fn clay_level(matte: &[[f32; 4]; 9]) -> f32 {
    let peak = matte_peak(matte);
    // the shader draws clay * light * (light / average)^contrast
    let clay = 0.2126 * CLAY[0] as f64 + 0.7152 * CLAY[1] as f64 + 0.0722 * CLAY[2] as f64;
    let drawn = clay * peak * (peak / CLAY_LIGHT).powf(CLAY_CONTRAST as f64);
    if drawn <= CLAY_PEAK { 1.0 } else { (CLAY_PEAK / drawn) as f32 }
}

/// What the lights' map was last painted from.
#[derive(Clone, PartialEq)]
struct Painted {
    family: usize,
    rev: u64,
    gain: u64,
    /// the family's lights are lit (always, unless an environment picture has taken their place)
    rig: bool,
    /// which environment picture's glows are in it (0 = none), how bright and how much widened
    picture: u64,
    picture_gain: u64,
    picture_blur: u64,
    /// the studio's backdrop and floor as they are painted into it (nothing when there is no studio)
    stage: Vec<u64>,
}

/// What a call to `Engine::frame` did.
#[derive(Clone, Copy, Debug, Default)]
pub struct Did {
    /// a new picture is on the screen texture
    pub drew: bool,
    /// there is more to do: call again soon (something is moving, or the picture is still refining)
    pub again: bool,
}

#[path = "scatter_engine.rs"]
mod glow;

pub struct Engine {
    pub gpu: Gpu,
    pub state: State,
    pub(crate) baker: Baker,
    /// the combining step and what each layer's volume holds (see combine.rs)
    pub layers: LayerWork,
    /// polygon models: the files read, and what turns them into volumes (see mesh_bake.rs)
    pub models: crate::mesh_bake::Models,
    pub renderer: Renderer,
    /// the studio: where the floor stands, what the tracer was told of it (stage.rs)
    pub stage: crate::stage::StageWork,
    /// shadows and caustics cast by the bodies on each other (see cast.rs)
    pub cast: crate::cast::Cast,
    /// cuts bubbles into the finished shape, when the material has any; and the bubbles the tracer's
    /// volume holds now
    bubbler: Bubbler,
    bubbles_in: Option<material::Bubbles>,
    /// each body's bubbles as they were last cut into the bodies' volume (None: none were)
    body_bubbles_in: Option<Vec<Option<material::Bubbles>>>,
    /// the light inside a milky body: what works it out, what it was last worked out from, and a count
    /// of the times the volume the tracer reads has changed (scatter_engine.rs)
    pub scatter: Scatter,
    glow: glow::Kept,
    shape_no: u64,
    /// how thick the shape is, measured from its volume (None: not measured since it was last built)
    thickness: Option<f64>,
    /// what the material's distances are multiplied by at present (1 unless it is fitted to the shape)
    fit_now: f64,
    /// the softened copies of the lights' map are those of the map as it is now painted
    soft_ok: bool,
    /// what the picture's exposure is multiplied by to expose for a milky body (`meter`; 1 otherwise)
    meter: f32,
    /// the fine lights' map was last painted with crisp edges (for a metal or a coat)
    fine_crisp: bool,
    /// ... and has the studio room painted into it (room.rs)
    fine_room: bool,
    /// make the bodies' volume even while no body has a material of its own (for checks that read it)
    pub bodies_always: bool,
    /// how thick each body is, measured from the bodies' volume (None: not measured since it was built)
    body_thick: Option<Vec<f64>>,
    /// the lights' map: the lights of the family that is up, as they stand, painted; and with them
    /// the glows of an environment picture (the light its ceiling took off: see hdri.rs)
    lights_tex: wgpu::Texture,
    pub(crate) lights_view: wgpu::TextureView,
    /// what it was painted from and what was painted, so that it is done again only on a change
    painted: Option<Painted>,
    painted_px: Vec<f32>,
    /// is there any light in the map at all
    lights_lit: bool,
    /// the family's lights as nine smooth terms (worked out only when clay needs them)
    rig_sh: Option<Sh9>,
    /// what the tracer was last told of the lamps (lamps.rs)
    last_lamps: Option<crate::lamps::LampsU>,
    /// what turns a panorama into what the tracer reads: made the first time it is needed, by
    /// whichever thread needs it first (its pipelines take a moment to build)
    maker: Arc<OnceLock<Maker>>,
    /// the environment picture in use, and how many there have been
    picture: Option<Hdri>,
    picture_no: u64,
    /// the light the glass reads when there is a picture: the picture at the softness and brightness
    /// set, with the lights' map added in (so that the tracer needs one lookup, not two); and what it
    /// was made from
    glass: Option<wgpu::Texture>,
    glass_from: Option<(ShowU, u64, u32)>,
    /// the path (as the state has it) that was last acted on
    hdri_seen: Option<String>,
    /// a picture being read on another thread
    loading: Option<Loading>,
    /// a change of picture that waits for its tracer to be built: Some(None) = no picture any more
    next: Option<Option<Hdri>>,
    /// the tracer the lighting calls for is being built on another thread
    building: bool,
    /// why the last environment picture could not be used
    pub hdri_error: Option<String>,
    /// wait for a picture to be read instead of carrying on without it meanwhile (stills, videos, timings)
    pub hdri_wait: bool,
    /// the small picture of the lighting: asked for, the texture, and what it was last drawn from
    show_wanted: Cell<bool>,
    show: Option<wgpu::Texture>,
    shown: Option<(ShowU, u64)>,
    /// counts the times the lights' map or the picture changed
    lighting_no: u64,
    last_lu: Option<LightU>,
    /// the picture so far is out of date (something moved) / useless (something changed outright)
    pub(crate) dirty: bool,
    cut: bool,
    last_tu: Option<TraceU>,
    last_look: Option<Look>,
    /// the focus planes are shown until this moment after they are moved
    planes_until: Option<Instant>,
    planes_shown: bool,
    /// [w, h] while a video is being rendered
    pub size_override: Option<(u32, u32)>,
    /// a still or a benchmark is being made: the grain is kept still
    pub auto: bool,
    /// fold every frame into a plain average even though things move between them (motion blur)
    pub averaging: bool,
    /// what went wrong the last time a shape could not be built
    pub error: Option<String>,
    /// milliseconds the last shape took to work out (CPU) and to put on the GPU
    pub last_bake_ms: f64,
    /// the same in parts: working the shape out, handing the work to the GPU, the GPU doing it
    /// (the last only when the build was waited for)
    pub last_bake_parts: [f64; 3],
    /// wait for the GPU after every build, so that each is timed to the end
    pub time_bakes: bool,
}

fn rows(m: &[f64; 9]) -> [[f32; 4]; 3] {
    let f = f32s(m);
    [[f[0], f[1], f[2], 0.0], [f[3], f[4], f[5], 0.0], [f[6], f[7], f[8], 0.0]]
}

impl Engine {
    pub fn new(gpu: Gpu, state: State, auto: bool) -> Self {
        let renderer = Renderer::new(&gpu);
        let cast = crate::cast::Cast::new(&gpu, renderer.cast_layout());
        let lights_tex = renderer.env_target(env::ENV_W as u32, env::ENV_H as u32);
        let lights_view = lights_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let mut e = Self {
            baker: Baker::new(&gpu), layers: LayerWork::new(&gpu), models: crate::mesh_bake::Models::new(auto), bubbler: Bubbler::new(&gpu), stage: crate::stage::StageWork::new(auto), bubbles_in: None, body_bubbles_in: None, scatter: Scatter::new(&gpu), glow: glow::Kept::default(), shape_no: 0, thickness: None, fit_now: 1.0, soft_ok: false, meter: 1.0, fine_crisp: false, fine_room: false, bodies_always: false, body_thick: None, gpu, state, renderer, cast, lights_tex, lights_view, painted: None, painted_px: Vec::new(), lights_lit: true, rig_sh: None, last_lamps: None,
            maker: Arc::new(OnceLock::new()), picture: None, picture_no: 0, glass: None, glass_from: None, hdri_seen: None, loading: None, next: None, building: false, hdri_error: None, hdri_wait: auto,
            show_wanted: Cell::new(false), show: None, shown: None, lighting_no: 0, last_lu: None,
            dirty: true, cut: true, last_tu: None, last_look: None,
            planes_until: None, planes_shown: false, size_override: None, auto, averaging: false, error: None, last_bake_ms: 0.0, last_bake_parts: [0.0; 3], time_bakes: false,
        };
        e.paint_lights();
        e
    }

    // ---- lighting ------------------------------------------------------------------------------
    fn maker(&self) -> Arc<OnceLock<Maker>> {
        self.maker.get_or_init(|| Maker::new(&self.gpu));
        self.maker.clone()
    }

    /// A picture has been read (or could not be).
    fn picture_arrived(&mut self, r: R<Hdri>) {
        match r {
            // it is taken into use by `light_upkeep`, once the tracer that reads it is built
            Ok(p) => self.next = Some(Some(p)),
            Err(e) => {
                // the lighting stays as it was, and the state says so
                self.hdri_error = Some(e);
                self.state.hdri = self.picture.as_ref().map(|p| hdri::tidy(&p.info.path.to_string_lossy()));
                self.hdri_seen = self.state.hdri.clone();
            }
        }
    }

    /// Paint the lights' map if what it is painted from has changed. True if the map changed.
    /// The lights' map painted again FINE times as finely each way, as `paint_lights` paints it, for the
    /// softened copies (whose first level a mirror reads: the glass's own map is too coarse for a mirror
    /// to show a studio's soft box and horizon sharp).
    ///
    /// `crisp`: a metal or a coat will mirror it. The rigs' lights fade out over most of their half-size,
    /// which suits glass (which sees them through itself, bent and spread); a mirror shows them as they
    /// are, and a real soft box has a crisp edge. So for a mirror their fading edge is made a quarter as
    /// wide. Glass reads its own map, painted as ever.
    ///
    /// `room`: the studio room (room.rs) is painted under the lights, for a polished or milky body to
    /// reflect when there is no picture (see `fine_wanted`).
    fn paint_fine(&mut self, crisp: bool, room: bool) -> Vec<f32> {
        self.fine_crisp = crisp;
        self.fine_room = room;
        let p = &self.state.p;
        let rig = self.picture.is_none() || p.hdri_lights > 0.5;
        let (w, h) = (FINE * env::ENV_W, FINE * env::ENV_H);
        let r = self.state.rig();
        let mut lights = r.lights.clone();
        if crisp { for l in lights.iter_mut() { l.soft *= 0.25; } }
        // (a studio's lights are the rig's while it is up, so they are made crisp above; its backdrop's
        // edges are made crisp where it is painted below)
        let mut px = if rig { env::paint_gain(&lights, r.ambient, p.light_gain, w, h) } else { env::paint(&[], [0.0; 3], w, h) };
        if let Some(pic) = &self.picture {
            env::add_glows(&mut px, &pic.glows, p.hdri_gain, hdri::blur_sigma(p.hdri_blur), w, h);
        }
        // (under the studio's backdrop and floor, which are painted over it where the studio has them)
        if room && rig { gf_core::room::paint(&mut px, w, h, &r.lights, &self.state.lights_to_room(), p.light_gain); }
        if !self.painted.as_ref().is_some_and(|k| k.stage.is_empty()) {
            let extra = self.picture.as_ref().map_or([0.0; 3], |pic| crate::stage::picture_on_floor(&pic.sh, p.hdri_gain, &self.state));
            self.stage.paint_for(&self.state, &mut px, w, h, extra, crisp);
        }
        px
    }

    /// How the copy of the lights a material reads is to be painted: crisp (a metal or a coat will
    /// mirror it), and with the studio room in it. The room is for what is not glass: a metal, a milky
    /// body, a glaze on clay, with no environment picture (a picture is a room already) and `envRoom` on.
    /// Glass, frosted or coated, keeps the lights as they are.
    fn fine_wanted(&self, v: &material::Variant) -> (bool, bool) {
        let p = &self.state.p;
        let opaque = v.metal || v.sss || (p.clay > 0.5 && v.coat);
        (v.metal || v.coat, opaque && self.picture.is_none() && p.env_room > 0.5)
    }

    fn paint_lights(&mut self) -> bool {
        let p = &self.state.p;
        let rig = self.picture.is_none() || p.hdri_lights > 0.5;
        let key = Painted {
            family: self.state.layer_fam(0), rev: self.state.lights_rev, gain: p.light_gain.to_bits(), rig,
            picture: if self.picture.is_some() { self.picture_no } else { 0 },
            picture_gain: if self.picture.is_some() { p.hdri_gain.to_bits() } else { 0 },
            picture_blur: if self.picture.is_some() { p.hdri_blur.to_bits() } else { 0 },
            stage: crate::stage::StageWork::paint_key(&self.state),
        };
        if self.painted.as_ref() == Some(&key) { return false; }
        self.painted = Some(key);
        self.rig_sh = None;
        let (w, h) = (env::ENV_W, env::ENV_H);
        let r = self.state.rig();
        let mut px = if rig { env::paint_gain(&r.lights, r.ambient, p.light_gain, w, h) } else { env::paint(&[], [0.0; 3], w, h) };
        if let Some(pic) = &self.picture {
            // the glows widen with the softening of the picture, so that they never stay sharper than it
            env::add_glows(&mut px, &pic.glows, p.hdri_gain, hdri::blur_sigma(p.hdri_blur), w, h);
        }
        // the studio: its backdrop under the lights, its floor over everything below the horizon
        if !self.painted.as_ref().is_some_and(|k| k.stage.is_empty()) {
            let extra = self.picture.as_ref().map_or([0.0; 3], |pic| crate::stage::picture_on_floor(&pic.sh, p.hdri_gain, &self.state));
            self.stage.paint(&self.state, &mut px, w, h, extra);
        }
        // (whoever asked for the lights "to change them" may have left them as they were)
        if px == self.painted_px { return false; }
        self.renderer.env_paint(&self.lights_tex, &px, w as u32, h as u32);
        self.soft_ok = false;
        self.lights_lit = px.chunks(4).any(|c| c[0] > 0.0 || c[1] > 0.0 || c[2] > 0.0);
        self.painted_px = px;
        true
    }

    /// Make the cube the glass reads, if there is a picture and anything that goes into the cube has
    /// changed: which picture, how soft, how bright, and the lights' map. True if it was made again.
    fn mix_glass(&mut self) -> bool {
        let Some(pic) = &self.picture else { return false };
        let p = &self.state.p;
        let level = pic.level_for(hdri::blur_sigma(p.hdri_blur));
        // as fine as the finer of the two levels it is blended from
        let face = pic.size(level.floor() as u32);
        let u = ShowU { env: [[0.0; 4]; 3], k: [if self.lights_lit { 1.0 } else { 0.0 }, p.hdri_gain as f32, level, 0.0], size: [0.0; 4] };
        let from = (u, self.lighting_no, face);
        if self.glass.is_some() && self.glass_from == Some(from) { return false; }
        let slot = self.maker.clone();
        let maker = slot.get().expect("a picture is in use, so there is a maker");
        let tex = match self.glass.take() {
            Some(t) if t.width() == face => t,
            _ => {
                let t = maker.glass_target(face);
                let view = t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::Cube), ..Default::default() });
                self.renderer.set_picture(Some((&pic.view, &view)));
                t
            }
        };
        maker.glass(&tex, &self.lights_view, &pic.view, u);
        self.glass = Some(tex);
        self.glass_from = Some(from);
        true
    }

    /// The whole lighting as nine smooth terms, for clay: the family's lights as they are lit, and
    /// the environment picture.
    fn lighting_sh(&mut self) -> Sh9 {
        let p = &self.state.p;
        let mut sh = [[0.0; 3]; 9];
        if self.picture.is_none() || p.hdri_lights > 0.5 {
            if self.rig_sh.is_none() {
                let (w, h) = (env::ENV_W, env::ENV_H);
                let r = self.state.rig();
                let mut map = env::paint_gain(&r.lights, r.ambient, p.light_gain, w, h);
                crate::stage::StageWork::paint_copy(&self.state, &mut map, w, h);
                self.rig_sh = Some(env::sh_of_map(&map, w, h, 4, env::paint_dir));
            }
            sh = self.rig_sh.unwrap();
        }
        if let Some(pic) = &self.picture { env::sh_add(&mut sh, &pic.sh, p.hdri_gain); }
        // Clay is for seeing the form, whatever the lights are set to for the glass: its light keeps the
        // lighting's direction and some of its colour, but not its level (the average over all the ways a
        // surface can face is always CLAY_LIGHT), and with no light at all it is lit evenly from everywhere.
        let lum = |c: &[f64; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let level = lum(&sh[0]) * 0.282095;
        if level < 1e-6 {
            sh = [[0.0; 3]; 9];
            sh[0] = [CLAY_LIGHT / 0.282095; 3];
            return sh;
        }
        for term in sh.iter_mut() {
            let grey = lum(term);
            for c in term.iter_mut() { *c = (*c + (grey - *c) * CLAY_GREY) * CLAY_LIGHT / level; }
        }
        sh
    }

    /// Keep the lighting the tracer reads in step with the state: the environment picture, the lights'
    /// map, the kind of tracer wanted, what it is told, and the small picture of the lighting if anyone
    /// looks at it. Marks the picture so far as out of date when any of that changed.
    ///
    /// Two things take too long to hold a frame up for, and are done on other threads: reading a
    /// picture, and building a tracer for a way of drawing not used before (a new pipeline takes about
    /// half a second). Until both are done the lighting stays wholly as it was, and then changes at
    /// once: the picture, the lights' map and the tracer always belong together. (With `hdri_wait`,
    /// for stills and videos, both are simply waited for.)
    fn light_upkeep(&mut self) {
        // 1. the environment picture asked for
        if self.state.hdri != self.hdri_seen {
            self.hdri_seen = self.state.hdri.clone();
            self.loading = None;
            self.next = None;
            match self.state.hdri.clone() {
                None => {
                    self.hdri_error = None;
                    if self.picture.is_some() { self.next = Some(None); }
                }
                Some(given) => {
                    let path = hdri::resolve(&given);
                    if self.picture.as_ref().is_some_and(|p| p.info.path == path) {
                        self.hdri_error = None;
                    } else if self.hdri_wait {
                        let r = self.maker().get().expect("just made").load(&path);
                        self.picture_arrived(r);
                    } else {
                        self.loading = Some(Loading::start(&self.maker, &self.gpu, path));
                    }
                }
            }
        }
        if let Some(r) = self.loading.as_ref().and_then(|l| l.poll()) {
            self.loading = None;
            self.picture_arrived(r);
        }
        // what the bodies cast on each other, if anything casts and anything can show it
        let casting = self.cast_upkeep();
        // 2. the kind of tracer the state calls for, with the picture that is about to be in use: the
        //    default (glass lit by the lights' map) unless there is a picture or clay
        let p = &self.state.p;
        let clay = p.clay > 0.5;
        let pic = match &self.next {
            Some(next) => next.as_ref(),
            None => self.picture.as_ref(),
        };
        // which parts of the material are in use (under clay only what lies on top of it: a coat, a film)
        let mut mat = material::Variant::of(p);
        // a scene of several bodies, one or more of a material of its own, is drawn by the tracer that
        // follows a ray from body to body (not under clay, which hides what things are made of)
        // (it is built with the parts any of the bodies uses, and each body chooses among them)
        if let Some(mats) = self.several_bodies().filter(|_| !clay).and_then(|_| self.state.body_materials()) { mat = material::Variant::of_bodies(&mats); }
        // lamps standing in the scene: plain glass is then drawn by the material tracer, which meets them
        // for next to nothing (lamps.rs); without a lamp nothing here changes
        let near = !self.state.lamps.is_empty();
        if near && !clay && mat.plain() { mat = material::Variant { near: true, mono: p.disp == 0.0, ..mat }; }
        let v = Variant {
            // (clay takes its light from the nine terms, whatever the lighting; a glaze over it reflects
            // the lighting itself, so then the picture is read after all)
            light: if pic.is_some() && (!clay || !mat.for_clay().plain()) { LightFrom::Picture } else { LightFrom::Lights },
            backdrop: pic.is_some() && p.hdri_show > 0.5,
            clay,
            mat: if clay { mat.for_clay() } else { mat },
            cast: casting,
            // (the studio's floor shows what the bodies cast on it while anything casts)
            stage: crate::stage::Seen { cast: casting && self.cast.floor, ..crate::stage::StageWork::seen(&self.state) },
            denoise: crate::denoise::wanted(&mat, clay, p.denoise),
            near,
            // (the box only grows for layers that are combined: one plain layer is where it is built)
            grown: !self.state.plain() && self.state.box_k() > 1.0,
        };
        // 3. is that tracer built? If not it is being built: carry on as things were
        // (and with the lens that is on: the plain one, or the first ray traced through the camera's own arithmetic)
        self.renderer.cam.wanted = crate::camera::wanted(&self.state.p, !self.focus_after());
        self.building = !self.hdri_wait && !self.renderer.prepare(v);
        if self.building {
            if self.show_wanted.get() { self.draw_preview(); }
            return;
        }
        // 4. everything changes together: the picture, the lights' map, the tracer and what it is told
        let mut changed = false;
        if let Some(next) = self.next.take() {
            match next {
                Some(pic) => {
                    self.picture = Some(pic);
                    self.picture_no += 1;
                    // (the cube the glass reads is made anew for it, and both are handed to the tracer then)
                    self.glass = None;
                    self.hdri_error = None;
                    // the path the way a setup should keep it (see `hdri::tidy`)
                    if let Some(given) = &self.state.hdri { self.state.hdri = Some(hdri::tidy(given)); }
                    self.hdri_seen = self.state.hdri.clone();
                }
                None => {
                    self.renderer.set_picture(None);
                    self.picture = None;
                    self.glass = None;
                }
            }
            changed = true;
        }
        changed |= self.paint_lights();
        if changed { self.lighting_no += 1; }
        changed |= self.mix_glass();
        if v != self.renderer.variant() || self.renderer.lens_pending() {
            self.renderer.set_variant(v);
            self.cut = true;
            changed = true;
        }
        let p = &self.state.p;
        let mut lu: LightU = bytemuck::Zeroable::zeroed();
        if v != Variant::default() {
            if let Some(pic) = &self.picture {
                // behind the shape the picture is shown as it is, or, where one pixel takes in several
                // of its texels, from the level that has about one texel a pixel
                let (_, (_, th)) = self.renderer.size();
                let cam = self.state.camera();
                // (a pixel as an angle through whatever lens is on; through an orthographic one the
                // backdrop is a single direction and any level does)
                let pixel = if cam.plain() || cam.proj == gf_core::camera::Projection::Orthographic { 2.0 * (p.fov * DEG / 2.0).tan() / th.max(1) as f64 } else { cam.pixel(th.max(1) as f64) };
                let behind = ((pixel / pic.texel).log2() - 1.0).clamp(0.0, (pic.levels - 1) as f64);
                lu.picture = [p.hdri_gain as f32, 0.0, behind as f32, 0.0];
            }
            if clay {
                lu.clay = CLAY;
                let sh = self.lighting_sh();
                let mut matte = env::sh_terms(&sh, env::SH_MATTE);
                // (turned down, the light and its average together: the shader compares the two)
                let level = clay_level(&matte);
                if level < 1.0 {
                    for term in matte.iter_mut() { for c in term.iter_mut() { *c *= level; } }
                }
                lu.clay2 = [CLAY_LIGHT as f32 * level, CLAY_CONTRAST, 0.0, 0.0];
                lu.nine[..9].copy_from_slice(&matte);
                lu.nine[9..].copy_from_slice(&env::sh_terms(&sh, SHEEN));
            }
        }
        // Exposed for a milky body: the lights as a white matte surface gathers them, facing the brightest
        // way, brought to 0.7 before the tone curve (about 215 of 255 after it: whiter still and the tone curve bleaches its colour). Only ever brighter
        // than set, and at most three times: a rig made for glass is dim for a body that only gathers light.
        self.meter = 1.0;
        // (not where a backdrop or the picture is seen behind the shape: the picture is then exposed for
        // what lies behind, as it is in a studio, and brightening it would wash everything out)
        // A dark backdrop is no reason: the body is the subject on it, as in the studio for translucent
        // bodies.
        let q = &self.state.p;
        let lum = |r: f64, g: f64, b: f64| 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let back = q.back_gain * lum(q.back_top_r, q.back_top_g, q.back_top_b).max(lum(q.back_mid_r, q.back_mid_g, q.back_mid_b)).max(lum(q.back_low_r, q.back_low_g, q.back_low_b));
        let seen_behind = (q.backdrop > 0.5 && back > 0.15) || (self.picture.is_some() && q.hdri_show > 0.5);
        if v.mat.sss && !clay && self.state.p.meter > 0.5 && !seen_behind {
            let matte = env::sh_terms(&self.lighting_sh(), env::SH_MATTE);
            let peak = matte_peak(&matte);
            // (the nine terms give the light arriving on the surface; a white matte surface sends back that
            // over pi towards each way it is seen: measured, a white milky body's brightest point under a
            // single light comes to peak / pi as a matte one would)
            if peak > 1e-6 { self.meter = (0.7 * std::f64::consts::PI / peak).clamp(1.0, 3.0) as f32; }
        }
        if self.last_lu != Some(lu) {
            self.renderer.set_light(&lu);
            self.last_lu = Some(lu);
            changed = true;
        }
        changed |= self.material_upkeep(v.mat);
        changed |= self.stage_upkeep();
        // where the lamps stand as the camera sees them (they move in the picture when the view or the lights turn)
        if near {
            let lu = crate::lamps::uniforms(&self.state, self.stage_floor().map(|f| (f.height, f.fade)));
            if self.last_lamps != Some(lu) {
                self.renderer.set_lamps(&lu);
                self.last_lamps = Some(lu);
                changed = true;
            }
        } else {
            self.last_lamps = None;
        }
        if changed { self.dirty = true; }
        // 5. the small picture of the lighting, drawn again only when the lighting it shows has changed
        if self.show_wanted.get() { self.draw_preview(); }
    }

    /// Keep what the tracer is told about the material in step with the settings: its numbers, the
    /// softened copies of the lights it reads, and the bubbles cut into its volume. True if anything the
    /// picture shows changed.
    fn material_upkeep(&mut self, v: material::Variant) -> bool {
        let mut changed = false;
        // the bubbles were changed: cut them afresh (the shape itself need not be built again)
        if (self.bubbles_wanted() != self.bubbles_in || self.body_bubbles_wanted() != self.body_bubbles_in) && self.scene_volume().is_some() {
            self.cut_bubbles();
            changed = true;
        }
        // a body has been given a material of its own, and the bodies' volume was not made when the
        // scene was last built: build it again, this time with it
        if self.state.body_materials().is_some() && !self.layers.combiner.bodies_wanted && self.state.want.is_none() {
            self.state.want = Some(Detail::Full);
        }
        if v.bodies > 0 {
            let want = self.fine_wanted(&v);
            if self.soft_ok && (self.fine_crisp, self.fine_room) != want { self.soft_ok = false; }
            if (v.rough || v.sss || v.wear || v.metal || v.coat || v.inner.soft_lights()) && !self.soft_ok && !self.painted_px.is_empty() {
                let fine = self.paint_fine(want.0, want.1);
                self.renderer.soften_lights(&fine, (FINE * env::ENV_W) as u32, (FINE * env::ENV_H) as u32);
                self.soft_ok = true;
                changed = true;
            }
            let Some(mats) = self.state.body_materials() else { return changed };
            // (a material fitted to the shape is fitted to its own body)
            if self.body_thick.is_none() && mats.iter().any(|m| m.fit > 0.5) { self.body_thick = self.measure_bodies(); }
            let thick = self.body_thick.clone().unwrap_or_default();
            let mut numbers = material::bodies_uniforms(&mats, &thick);
            if let Some(pic) = &self.picture {
                // (every body reads the picture's softer levels from its own numbers)
                let p = &self.state.p;
                for k in 0..material::MAX_BODIES {
                    let o = material::MB_HEAD + k * material::MU_FLOATS + material::MU_PIC;
                    numbers[o..o + 4].copy_from_slice(&[p.hdri_gain as f32, pic.texel as f32, hdri::blur_sigma(p.hdri_blur) as f32, (pic.levels - 1) as f32]);
                    let sun = material::sun_numbers(&pic.glows, p.hdri_gain);
                    for (i, at) in material::MU_SUN.iter().enumerate() { numbers[material::MB_HEAD + k * material::MU_FLOATS + at] = sun[i]; }
                }
            }
            changed |= self.renderer.set_bodies_material(&numbers);
            // what lies inside each body, with the width of a pixel at the shape (as for one material)
            if v.inner.any() {
                let p = &self.state.p;
                let (_, (_, th)) = self.renderer.size();
                let pixel = p.dist * 2.0 * (p.fov * DEG / 2.0).tan() / th.max(1) as f64;
                changed |= self.renderer.set_inner_bodies(&material::bodies_inner(&mats, &thick, pixel));
            }
            // each milky body's light, worked out through its own distances
            let scales: Vec<f64> = mats.iter().enumerate().map(|(k, m)| material::fit_scale(m, thick.get(k).copied().filter(|t| *t > 0.0))).collect();
            let owned: Vec<gf_core::settings::Settings> = mats.iter().map(|m| (*m).clone()).collect();
            return self.glow_bodies_upkeep(&owned, &scales) || changed;
        }
        if v.plain() { return changed; }
        // rough and milky materials read softened copies of the lights' map: made when first wanted, and
        // again when the map is painted anew
        let want = self.fine_wanted(&v);
        if self.soft_ok && (self.fine_crisp, self.fine_room) != want { self.soft_ok = false; }
        if (v.rough || v.sss || v.wear || v.metal || v.coat || v.inner.soft_lights()) && !self.soft_ok && !self.painted_px.is_empty() {
            let fine = self.paint_fine(want.0, want.1);
                self.renderer.soften_lights(&fine, (FINE * env::ENV_W) as u32, (FINE * env::ENV_H) as u32);
            self.soft_ok = true;
            changed = true;
        }
        let scale = self.material_scale();
        self.fit_now = scale;
        let mut numbers = material::uniforms(&self.state.p, scale);
        if let Some(pic) = &self.picture {
            // the environment picture's softer levels: how bright, how wide a texel of its sharpest level
            // is, how much it is softened for the glass anyway, and its softest level
            let p = &self.state.p;
            numbers[material::MU_PIC..material::MU_PIC + 4].copy_from_slice(&[p.hdri_gain as f32, pic.texel as f32, hdri::blur_sigma(p.hdri_blur) as f32, (pic.levels - 1) as f32]);
            let sun = material::sun_numbers(&pic.glows, p.hdri_gain);
            for (i, at) in material::MU_SUN.iter().enumerate() { numbers[*at] = sun[i]; }
        }
        // what lies inside the body: its numbers, with the width of a pixel at the shape (colour bands and
        // grains finer than a pixel are averaged) and the tint the tracer takes off already
        if v.inner.any() {
            let p = &self.state.p;
            let (_, (_, th)) = self.renderer.size();
            let pixel = p.dist * 2.0 * (p.fov * DEG / 2.0).tan() / th.max(1) as f64;
            let k = if v.spectrum { 0.0 } else { p.density / scale };
            changed |= self.renderer.set_inner(&gf_core::inner::uniforms(p, scale, pixel, [p.absorb[0] * k, p.absorb[1] * k, p.absorb[2] * k]));
        }
        let changed = self.renderer.set_material(&numbers) || changed;
        // a milky body's light is worked out again when lights, shape or material have changed
        self.glow_upkeep(v, scale) || changed
    }

    /// How thick each body is, as `measure_thickness` measures the whole scene, from two dozen slices of
    /// the bodies' volume (a body no slice passes through is left at nothing, and is then not fitted).
    fn measure_bodies(&self) -> Option<Vec<f64>> {
        let b = self.bodies()?;
        let (vol, n, channels) = (b.volume, b.count as usize, b.channels as usize);
        let (mut sum, mut count) = (vec![0.0f64; n], vec![0u64; n]);
        for i in 0..24 {
            let z = ((i as f64 + 0.5) / 24.0 * vol.res as f64) as u32;
            let bytes = self.gpu.read(&vol.tex, 0, 0, z.min(vol.res - 1), vol.res, vol.res, 2 * channels as u32).ok()?;
            for texel in bytemuck::cast_slice::<u8, half::f16>(&bytes).chunks_exact(channels) {
                for k in 0..n.min(channels) {
                    let d = texel[k].to_f32();
                    if d < 0.0 {
                        sum[k] -= d as f64;
                        count[k] += 1;
                    }
                }
            }
        }
        Some((0..n).map(|k| if count[k] > 0 { 5.0 * sum[k] / count[k] as f64 } else { 0.0 }).collect())
    }

    /// How many bodies the scene is drawn as, when it is drawn body by body: one or more of them has a
    /// material of its own, and the bodies' volume has been made.
    fn several_bodies(&self) -> Option<u8> {
        let mats = self.state.body_materials()?;
        // (the volume in hand must be of the bodies as they now are: after layers are joined or set
        // apart it is a frame or two before the scene is built again, and until then it is not)
        let n = self.renderer.bodies_count().filter(|_| self.bodies().is_some())?;
        (n as usize == mats.len()).then_some(n as u8)
    }

    /// The bubbles to cut into the scene's volume: the material's, unless the scene is drawn body by body
    /// (the bodies' volume holds no bubbles, so the two would not agree).
    fn bubbles_wanted(&self) -> Option<material::Bubbles> {
        if self.state.body_materials().is_some() { None } else { material::bubbles(&self.state.p) }
    }

    /// Each body's bubbles, while the scene is drawn body by body and at least one body has bubbles (in
    /// the order of the bodies; None for a body that has none).
    fn body_bubbles_wanted(&self) -> Option<Vec<Option<material::Bubbles>>> {
        let mats = self.state.body_materials()?;
        let per: Vec<Option<material::Bubbles>> = mats.iter().map(|m| if material::body_variant(m).bubbles { material::bubbles(m) } else { None }).collect();
        per.iter().any(|b| b.is_some()).then_some(per)
    }

    /// Point the tracer at the shape as built, with the material's bubbles cut into it if it has any;
    /// drawn body by body, with each body's bubbles cut into the bodies' volume and the scene's.
    fn cut_bubbles(&mut self) {
        let bubbles = self.bubbles_wanted();
        // (the box the scene's volume spans: grown when the layers are placed far out)
        self.bubbler.hb = if self.layers.combined { self.state.scene_hb() } else { gf_core::HB };
        // (the volume the tracer reads: the one shape's, or the layers combined)
        let scene = if self.layers.combined { self.layers.combiner.volume() } else { self.baker.layer_volume(0) };
        if let Some(vol) = scene {
            let vol = match &bubbles {
                Some(b) => self.bubbler.carve(vol, b),
                None => vol,
            };
            self.renderer.set_volume(vol);
        }
        self.bubbles_in = bubbles;
        // several bodies: each one's bubbles in its own channel, and the scene's volume made again from
        // them; or, if they have none now but had before, the volumes as built
        let per = self.body_bubbles_wanted();
        if let Some(src) = self.layers.combiner.bodies_volume() {
            let (channels, count) = (self.layers.combiner.bodies_channels(), self.state.stack.bodies().len() as u32);
            match &per {
                Some(per) => {
                    let (bodies, scene) = self.bubbler.carve_bodies(src, channels, count, per);
                    self.renderer.set_bodies(Some((bodies, count)));
                    self.renderer.set_volume(scene);
                }
                None if self.body_bubbles_in.is_some() => self.renderer.set_bodies(Some((src, count))),
                None => {}
            }
        }
        self.body_bubbles_in = per;
        self.shape_no += 1;
    }

    /// What the material's distances are multiplied by to fit the shape (1 unless `fit` is on).
    fn material_scale(&mut self) -> f64 {
        if self.state.p.fit <= 0.5 { return 1.0; }
        let thickness = self.state.thickness_hint().or_else(|| {
            if self.thickness.is_none() { self.thickness = self.measure_thickness(); }
            self.thickness
        });
        material::fit_scale(&self.state.p, thickness)
    }

    /// How thick the shape is, from a dozen slices of its volume: five times the mean depth of the points
    /// inside it. (A plate is four times the mean depth of its inside thick, a rod six, a ball eight: five
    /// stands for what a ray typically crosses, where the deepest point would speak only for the fattest
    /// part.) Only for shapes that are not a single tube, and only when a material asks to be fitted.
    fn measure_thickness(&self) -> Option<f64> {
        let vol = self.scene_volume()?;
        let (mut sum, mut count) = (0.0f64, 0u64);
        for i in 0..12 {
            let z = ((i as f64 + 0.5) / 12.0 * vol.res as f64) as u32;
            let bytes = self.gpu.read(&vol.tex, 0, 0, z.min(vol.res - 1), vol.res, vol.res, 2).ok()?;
            for h in bytemuck::cast_slice::<u8, half::f16>(&bytes) {
                let d = h.to_f32();
                if d < 0.0 {
                    sum -= d as f64;
                    count += 1;
                }
            }
        }
        (count > 0).then(|| 5.0 * sum / count as f64)
    }

    fn draw_preview(&mut self) {
        let p = &self.state.p;
        let m = rows(&self.state.light_m());
        let mut u = ShowU { env: m, k: [if self.lights_lit { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0], size: [0.0; 4] };
        if let Some(pic) = &self.picture {
            // the picture as the glass reads it, but never finer than the small picture can show
            let level = pic.level_for(hdri::blur_sigma(p.hdri_blur)).max(pic.level_for(std::f64::consts::TAU / hdri::SHOW_W as f64));
            u.k = [u.k[0], p.hdri_gain as f32, level, 0.0];
        }
        if self.show.is_some() && self.shown == Some((u, self.lighting_no)) { return; }
        let slot = self.maker();
        let maker = slot.get().expect("just made");
        let tex = self.show.take().unwrap_or_else(|| maker.show_target());
        maker.show(&tex, &self.lights_view, self.picture.as_ref().map(|p| &p.view), u);
        self.show = Some(tex);
        self.shown = Some((u, self.lighting_no));
    }

    /// What the casting of shadows needs to know of the lighting: whether the family's lights are lit,
    /// and of an environment picture its glows, its nine terms and a number that changes with the picture.
    pub(crate) fn cast_lighting(&self) -> (bool, Option<(&[env::Glow], &Sh9, u64)>) {
        (self.picture.is_none() || self.state.p.hdri_lights > 0.5, self.picture.as_ref().map(|p| (p.glows.as_slice(), &p.sh, self.picture_no)))
    }
    /// What the material's distances are multiplied by at present to fit the shape (casting follows a
    /// body's material, so it must measure the body as the tracer does).
    pub(crate) fn cast_fit(&self) -> f64 {
        self.fit_now
    }
    /// How thick body `k` is, if the bodies have been measured since they were built.
    pub(crate) fn cast_thick(&self, k: usize) -> Option<f64> {
        self.body_thick.as_ref().and_then(|t| t.get(k).copied())
    }

    /// Bring the lighting up to date now: a picture that was asked for is read (waited for if `hdri_wait`).
    pub fn settle_lighting(&mut self) {
        self.light_upkeep();
    }

    /// File name, width and height of the environment picture in use.
    pub fn hdri_info(&self) -> Option<(String, u32, u32)> {
        self.picture.as_ref().map(|p| (p.info.file.clone(), p.info.w, p.info.h))
    }
    /// Everything there is to say about the environment picture in use (sizes, memory, how long it took).
    pub fn hdri_details(&self) -> Option<&hdri::Info> {
        self.picture.as_ref().map(|p| &p.info)
    }
    /// A picture is being read: the lighting will change when it is ready.
    pub fn hdri_loading(&self) -> bool {
        self.loading.is_some() || self.next.is_some() || self.state.hdri != self.hdri_seen
    }
    /// The lighting is about to change: a picture is being read, or a tracer is being built.
    pub fn lighting_pending(&self) -> bool {
        self.hdri_loading() || self.building
    }

    /// The lighting as it stands, as a small picture to look at (512 x 256, display-ready bytes like the
    /// screen's): the lights' map and the environment picture together, turned as they are turned, laid
    /// out as `gf_core::env::preview_dir` says (its middle is straight ahead of the camera, behind the
    /// shape). None until it has been drawn: asking is what gets it drawn, from the next frame on, and
    /// again whenever the lighting changes.
    pub fn env_preview(&self) -> Option<&wgpu::Texture> {
        self.show_wanted.set(true);
        self.show.as_ref()
    }

    /// Frames after which the picture has settled and drawing stops.
    pub fn max_accum(&self) -> u32 {
        if self.state.p.clay > 0.5 { MAX_ACCUM_CLAY } else { MAX_ACCUM }
    }

    /// Build the shape as it stands, this carefully, and point the tracer at it.
    pub fn bake(&mut self, mode: Detail) -> R<()> {
        // the bodies' volume is two or four volumes of memory: made only while a body has a material of
        // its own (or a check has asked for it)
        self.layers.combiner.bodies_wanted = self.bodies_always || self.state.body_materials().is_some() || self.cast_wants_bodies();
        if self.state.plain() { self.renderer.set_bodies(None); }
        self.build(mode)?;
        self.stage.rebuilt();
        // (while the shape animates its thickness is not measured afresh every frame: the last one stands)
        if !matches!(mode, Detail::Anim | Detail::Hq) {
            self.thickness = None;
            self.body_thick = None;
        }
        self.shape_no += 1;
        // the tracer has just been pointed at the new volume: the material's bubbles go into it
        self.bubbles_in = None;
        self.body_bubbles_in = None;
        if self.bubbles_wanted().is_some() || self.body_bubbles_wanted().is_some() { self.cut_bubbles(); }
        Ok(())
    }

    fn build(&mut self, mode: Detail) -> R<()> {
        // (models first: a file's path tidied, its smoothing put on its slider, before anything is built from them)
        self.model_upkeep();
        if !self.state.plain() { return self.bake_layers(mode); }
        let t0 = Instant::now();
        let job = self.state.shape_job(mode, None)?;
        let t1 = Instant::now();
        let fam = &self.state.families[job.family];
        let stats = match &job.shape {
            ShapeJob::Curve { curve, melt } => self.baker.curve(curve, job.res, *melt)?,
            ShapeJob::Field { uniforms } => {
                let p = &fam.presets[job.preset];
                debug_assert!(matches!(p.shape, Shape::Field { .. }));
                self.baker.field(&format!("{}/{}", fam.id, p.id), p, uniforms, job.res)?
            }
            ShapeJob::Mesh(model) => self.models.build(&self.gpu, &mut self.baker, 0, model, job.res, mode)?,
        };
        // a full build is timed to the end, so the figure shown is honest; while moving nothing waits
        let t2 = Instant::now();
        if matches!(mode, Detail::Full) || self.time_bakes { self.gpu.wait(); }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        self.last_bake_ms = ms;
        self.last_bake_parts = [(t1 - t0).as_secs_f64() * 1e3, (t2 - t1).as_secs_f64() * 1e3, t2.elapsed().as_secs_f64() * 1e3];
        let (step, out, inn, layers) = (fam.step_scale, fam.steps_out, fam.steps_in, fam.layers);
        let vol = self.baker.volume().ok_or("no volume")?;
        self.renderer.set_scene(&self.lights_view, vol, step, out, inn, layers);
        let meta = &mut self.state.shapes[job.family].meta;
        meta.res = job.res;
        meta.ms = ms;
        meta.segments = stats.segments;
        meta.splats = stats.splats;
        self.layers.built_plain(&mut self.baker, self.state.layer_key(0), job.res, mode, match &job.shape { ShapeJob::Curve { melt, .. } => Some(*melt), _ => None });
        self.dirty = true;
        Ok(())
    }

    /// Build the shape if something has asked for it. An error is kept in `self.error` and the old shape stays.
    fn bake_wanted(&mut self) {
        // a model that another thread has finished reading is built now
        self.model_upkeep();
        if let Some(mode) = self.state.want.take() {
            match self.bake(mode) {
                Ok(()) => self.error = None,
                // (a model still being read is no error: the picture stays as it is until the model is ready)
                Err(_) if self.models.waiting() => {}
                Err(e) => self.error = Some(e),
            }
        }
    }

    /// The stage's size, in a live view: when the layers ask for another (a layer dragged across a step),
    /// the tracer for it is built on another thread while everything stays at the old size; when it is
    /// ready, the volumes are built at the new size and the tracer changes with them, in the same frame.
    /// (Stills and videos wait for every tracer, so they take the size asked for at once.)
    fn box_upkeep(&mut self) {
        let want = self.state.stack.box_k();
        if self.hdri_wait {
            self.state.stack.box_held = None;
            return;
        }
        let Some(held) = self.state.stack.box_held else {
            self.state.stack.box_held = Some(want);
            return;
        };
        let plain = self.state.plain();
        let grown = |k: f64| !plain && k > 1.0;
        if held == want {
            // a layer nearing the edge of the stage as it always was: the tracer for a larger one is built
            // beforehand (one serves every larger size), so that crossing it changes nothing but the size
            if !grown(held) && !plain && self.state.stack.box_need() > 0.9 {
                self.renderer.prepare(crate::render::Variant { grown: true, ..self.renderer.variant() });
            }
            return;
        }
        // a new size: at once if the tracer for it is the one in use or is ready, otherwise once it is
        let v = crate::render::Variant { grown: grown(want), ..self.renderer.variant() };
        if plain || self.renderer.variant().grown == grown(want) || self.renderer.prepare(v) {
            self.state.stack.box_held = Some(want);
            if self.state.want.is_none() { self.state.want = Some(if self.state.playing { Detail::Anim } else { Detail::Full }); }
        }
    }

    /// Focus is done to the finished picture (`dof` = 1), not by the tracer's lens (`dof` = 0).
    fn focus_after(&self) -> bool {
        self.state.p.dof >= 0.5
    }

    /// How the finished picture is to be focused: not at all with no lens, or when the tracer's lens does it.
    /// (Whether there is a picture behind the glass to blur too is the renderer's to say.)
    fn focus(&self) -> Focus {
        let p = &self.state.p;
        if !self.focus_after() || p.aperture <= 0.0 { return Focus::OFF; }
        Focus {
            aperture: p.aperture as f32, near: p.fnear.min(p.ffar) as f32, far: p.fnear.max(p.ffar) as f32,
            blades: p.blades as f32, rot: (p.bokeh_rot * DEG) as f32, backdrop: false, floor: self.floor_nearest() as f32,
            bound: (1.47 * self.state.box_k()) as f32,
            squeeze: p.squeeze.max(1.0) as f32, edge: p.bokeh.clamp(-1.0, 1.0) as f32, eye: p.catseye.clamp(0.0, 1.0) as f32,
            lens: { let c = self.state.camera(); (!c.plain() || p.bokeh != 0.0 || p.catseye > 0.0 || crate::camera::always()).then_some(c) },
        }
    }

    /// The tracer's settings as the state has them now.
    pub fn trace_uniforms(&self) -> TraceU {
        let (s, p) = (&self.state, &self.state.p);
        let vres = self.scene_volume().map_or(1, |v| v.res);
        // the tint, deepened by `density` (and measured against the shape's thickness when the material is
        // fitted to it); when the material absorbs by wavelength the tint is not used
        let tint = if material::Variant::of(p).spectrum { 0.0 } else { p.density / self.fit_now };
        let mut tu = TraceU {
            res: [0.0; 2],
            jitter: [0.0; 2],
            shift: [p.shift[0] as f32, p.shift[1] as f32],
            focus: [p.fnear.min(p.ffar) as f32, p.fnear.max(p.ffar) as f32],
            obj: rows(&s.view_r()),
            env: rows(&s.light_m()),
            absorb: [(p.absorb[0] * tint) as f32, (p.absorb[1] * tint) as f32, (p.absorb[2] * tint) as f32, 0.0],
            dist: p.dist as f32,
            tan_fov: (p.fov * DEG / 2.0).tan() as f32,
            lens: p.aperture as f32,
            blades: p.blades as f32,
            bokeh_rot: (p.bokeh_rot * DEG) as f32,
            ior: p.ior as f32,
            disp: p.disp as f32,
            vres: vres as f32,
            frame: 0,
            nw: (p.nw + 0.5).floor() as i32,
            ns: (p.ns + 0.5).floor() as i32,
            nb: (p.nb + 0.5).floor() as i32,
            lens_n: (p.lens_n + 0.5).floor() as i32,
            // (what the surface's own reflection is multiplied by, so that the exposure set for a milky
            // body's light (`meter`) leaves the mirror on its surface as bright as the exposure set; and the
            // box the scene's volume spans, read by a tracer built for a larger stage than ever)
            pad: [(1.0 / self.meter.max(1e-3)).to_bits() as i32, 0, (self.state.scene_hb() as f32).to_bits() as i32],
            cam: crate::camera::tracer_cam(p, !self.focus_after()),
        };
        if p.hollow > 0.5 && p.metal < 0.5 {
            // a thin wall around air: nothing bends and nothing is absorbed on the way across (the wall's
            // own reflection and tint are the material's business)
            (tu.ior, tu.disp, tu.absorb) = (1.0, 0.0, [0.0; 4]);
        }
        if self.focus_after() {
            // The tracer gets a pinhole and none of the lens's settings, so that its picture has no lens
            // grain, and so that moving the focus does not count as the view having moved: the picture
            // so far stays, and only the focus stage and the screen are done again.
            (tu.lens, tu.lens_n, tu.blades, tu.bokeh_rot, tu.focus) = (0.0, 1, 0.0, 0.0, [0.0; 2]);
        }
        tu
    }

    fn look(&self) -> Look {
        let p = &self.state.p;
        Look {
            smooth: p.smooth as f32, spread: p.spread as f32, debug: p.debug as f32,
            exposure: p.exposure as f32 * self.meter, bloom: p.bloom as f32, vignette: p.vignette as f32, grain: p.grain as f32, raw: p.raw as f32,
            planes: [p.fnear.min(p.ffar) as f32, p.fnear.max(p.ffar) as f32, if self.planes_shown { 1.0 } else { 0.0 }],
            still_grain: self.auto || self.state.rendering,
            focus: self.focus(),
            grade: gf_core::stage::Grade::of(p).uniforms(),
            lens: crate::camera::Lens::of(p),
        }
    }

    /// Make the pictures fit a canvas of this size (or the size a video is being rendered at).
    pub fn resize(&mut self, w: u32, h: u32) {
        let ((w, h), scale) = match self.size_override {
            Some(s) => (s, 1.0),
            None => ((w, h), self.state.p.scale),
        };
        if self.renderer.resize(w, h, scale) {
            self.dirty = true;
            self.cut = true;
        }
    }

    /// The picture so far is useless: start again from the next frame.
    pub fn restart(&mut self) {
        self.dirty = true;
        self.cut = true;
    }

    /// Draw one frame with things as they are (no clock, no resize).
    pub fn draw(&mut self, skip_post: bool) {
        self.bake_wanted();
        self.light_upkeep();
        if std::mem::take(&mut self.state.cut) { self.cut = true; self.dirty = true; }
        let tu = self.trace_uniforms();
        let look = self.look();
        // a picture drawn through one kind of lens is no start for a picture through another
        if self.last_tu.is_some_and(|l| l.cam[0][0] != tu.cam[0][0]) { self.cut = true; }
        // anything the tracer or the antialiasing sees differently from last time means the view moved
        let moved = self.last_tu != Some(tu) || self.last_look.is_some_and(|l| (l.smooth, l.spread, l.debug) != (look.smooth, look.spread, look.debug));
        // while averaging, what moves between frames is meant to blur: only an outright change starts afresh
        let (dirty, cut) = if self.averaging && !self.cut { (false, false) } else { (self.dirty || moved, self.cut) };
        // (a frame that starts the picture afresh is the first of a still one, not a moving one)
        self.renderer.set_moving(dirty && !cut);
        self.renderer.frame(&tu, &look, dirty, cut, skip_post);
        self.dirty = false;
        self.cut = false;
        self.last_tu = Some(tu);
        self.last_look = Some(look);
    }

    /// One turn of the live loop: move everything on by `dt` seconds, then draw if there is anything to draw.
    pub fn frame(&mut self, canvas_w: u32, canvas_h: u32, dt: f64) -> Did {
        self.resize(canvas_w, canvas_h);
        if !self.state.rendering { self.state.tick(dt); }
        self.box_upkeep();
        self.bake_wanted();
        self.light_upkeep();
        if std::mem::take(&mut self.state.cut) { self.cut = true; self.dirty = true; }
        // the focus planes: shown while switched on, and for a moment after a plane is moved
        let now = Instant::now();
        if std::mem::take(&mut self.state.flash_planes) { self.planes_until = Some(now + std::time::Duration::from_millis(1600)); }
        let flashing = self.planes_until.is_some_and(|t| now < t);
        if !flashing { self.planes_until = None; }
        self.planes_shown = !self.state.rendering && (self.state.p.planes > 0.5 || flashing);

        let tu = self.trace_uniforms();
        let look = self.look();
        let moved = self.last_tu != Some(tu) || self.last_look.is_some_and(|l| (l.smooth, l.spread, l.debug) != (look.smooth, look.spread, look.debug));
        // (a change of lighting on its way counts as moving: the frame it arrives in must not be slept through)
        let moving = self.state.p.spin > 0.0 || self.state.p.lspin > 0.0 || self.state.playing || self.lighting_pending() || self.models.busy();
        if !(self.dirty || moved) && self.renderer.accum_n >= self.max_accum() {
            // converged: stop tracing and let the GPU idle; only the screen may need doing again
            if self.last_look != Some(look) {
                self.renderer.present(&tu, &look);
                self.last_look = Some(look);
                return Did { drew: true, again: flashing || moving };
            }
            return Did { drew: false, again: flashing || moving };
        }
        self.draw(false);
        Did { drew: true, again: true }
    }

    /// Render `frames` accumulated frames right now, starting the picture afresh.
    pub fn render(&mut self, frames: u32) {
        self.restart();
        for i in 0..frames {
            self.draw(i + 1 < frames);
            if i % 8 == 7 { self.gpu.wait(); }
        }
        self.gpu.wait();
    }

    /// Draw `n` frames of a turning view, each a step further round, as the live view would: what is on
    /// screen afterwards is what a moving view shows.
    pub fn turned(&mut self, n: u32) {
        let turn = axis_angle(0.3, 1.0, 0.2, 0.6 / 120.0);
        for i in 0..n {
            self.state.turn(&turn);
            self.draw(false);
            if i % 4 == 3 { self.gpu.wait(); }
        }
        self.gpu.wait();
    }

    /// What focus costs: turning frames timed with the lens as it is set and with no lens, in turn, `rounds`
    /// times over, `n` frames a go. Returns (ms per frame with no lens, ms per frame with the lens), each
    /// the quickest of its goes: other programs on the GPU can only make a go slower.
    pub fn bench_focus(&mut self, n: u32, rounds: u32) -> (f64, f64) {
        let turn = axis_angle(0.3, 1.0, 0.2, 0.6 / 120.0);
        let lens = self.state.p.aperture;
        let mut best = [f64::MAX; 2];
        for _ in 0..rounds {
            for (k, a) in [0.0, lens].into_iter().enumerate() {
                self.state.p.aperture = a;
                for _ in 0..4 { self.state.turn(&turn); self.draw(false); }
                self.gpu.wait();
                let t0 = Instant::now();
                for i in 0..n {
                    self.state.turn(&turn);
                    self.draw(false);
                    if i % 4 == 3 { self.gpu.wait(); }
                }
                self.gpu.wait();
                best[k] = best[k].min(t0.elapsed().as_secs_f64() * 1e3 / n as f64);
            }
        }
        self.state.p.aperture = lens;
        self.dirty = true;
        (best[0], best[1])
    }

    /// Time `n` frames of a turning view, nothing waiting on the display. Milliseconds per frame.
    pub fn bench(&mut self, n: u32) -> f64 {
        let step = 1.0 / 120.0;
        let turn = axis_angle(0.3, 1.0, 0.2, 0.6 * step);
        for _ in 0..6 {
            self.state.turn(&turn);
            self.draw(false);
        }
        self.gpu.wait();
        let t0 = Instant::now();
        for i in 0..n {
            self.state.turn(&turn);
            self.draw(false);
            if i % 4 == 3 { self.gpu.wait(); }
        }
        self.gpu.wait();
        self.dirty = true;
        t0.elapsed().as_secs_f64() * 1e3 / n as f64
    }

    /// Where a turning frame's time goes: (tracing and folding into the picture, bloom and tone map), ms each.
    pub fn bench_parts(&mut self, n: u32) -> (f64, f64) {
        let step = 1.0 / 120.0;
        let turn = axis_angle(0.3, 1.0, 0.2, 0.6 * step);
        let run = |e: &mut Engine, post: bool| {
            for _ in 0..6 { e.state.turn(&turn); e.draw(!post); }
            e.gpu.wait();
            let t0 = Instant::now();
            for i in 0..n {
                e.state.turn(&turn);
                e.draw(!post);
                if i % 4 == 3 { e.gpu.wait(); }
            }
            e.gpu.wait();
            t0.elapsed().as_secs_f64() * 1e3 / n as f64
        };
        let without = run(self, false);
        let with = run(self, true);
        (without, with - without)
    }

    /// Time `n` frames of the animation playing at 60 frames a second: (ms per frame in all, ms of that
    /// spent working out and building the shape).
    pub fn bench_anim(&mut self, n: u32) -> (f64, f64) {
        self.state.set_playing(true);
        for _ in 0..8 {
            self.state.tick(1.0 / 60.0);
            self.draw(false);
        }
        self.gpu.wait();
        let (t0, mut bake) = (Instant::now(), 0.0);
        for i in 0..n {
            self.state.tick(1.0 / 60.0);
            let had = self.state.want.is_some();
            self.draw(false);
            if had { bake += self.last_bake_ms; }
            if i % 4 == 3 { self.gpu.wait(); }
        }
        self.gpu.wait();
        let ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
        self.state.set_playing(false);
        (ms, bake / n as f64)
    }

    /// Focus on whatever glass is under a point of the picture, given as fractions of its width and
    /// height from the top left. Returns the depth found, or None if there is no glass near.
    pub fn pick_focus(&mut self, fx: f64, fy: f64) -> Option<f64> {
        let (_, (tw, th)) = self.renderer.size();
        if tw < 9 || th < 9 { return None; }
        let (x, y) = ((fx * tw as f64).floor() as i64, (fy * th as f64).floor() as i64);
        let n = 9i64;
        let x0 = (x - 4).clamp(0, tw as i64 - n);
        let y0 = (y - 4).clamp(0, th as i64 - n);
        let buf = self.renderer.read_hdr(x0 as u32, y0 as u32, n as u32, n as u32).ok()?;
        let mut best: Option<(f64, f64, f64)> = None;
        let mut bd = 1e9;
        for j in 0..n {
            for i in 0..n {
                let t = buf[((j * n + i) * 4 + 3) as usize] as f64;
                let d = ((x0 + i - x).pow(2) + (y0 + j - y).pow(2)) as f64;
                if t > 0.0 && d < bd {
                    bd = d;
                    best = Some((t, x0 as f64 + i as f64 + 0.5, y0 as f64 + j as f64 + 0.5));
                }
            }
        }
        let (t, px, py_top) = best?;
        // the view's maths counts pixel rows upward from the bottom
        Some(self.state.focus_at(t, px, th as f64 - py_top, tw as f64, th as f64))
    }

    /// The distance field as it is on the GPU: voxels per side, and every voxel as a 16-bit float's bits,
    /// slice by slice. For checking that a faster way of building a shape builds the same shape.
    pub fn read_volume(&self) -> R<(u32, Vec<u16>)> {
        let vol = self.scene_volume().ok_or("no shape has been built")?;
        let mut out = Vec::with_capacity((vol.res as usize).pow(3));
        for z in 0..vol.res {
            let bytes = self.gpu.read(&vol.tex, 0, 0, z, vol.res, vol.res, 2)?;
            out.extend_from_slice(bytemuck::cast_slice::<u8, u16>(&bytes));
        }
        Ok((vol.res, out))
    }

    /// What is kept between frames, counted (for the window's diagnostic log).
    pub fn kept(&self) -> String {
        format!("{} {} tracks={} layers={} fam={} preset={}", self.renderer.kept(), self.baker.kept(), self.state.tracks.len(), self.state.layer_count(), self.state.family().id, self.state.preset().id)
    }

    /// What is on screen as RGBA bytes, top row first.
    pub fn screenshot(&self) -> R<(u32, u32, Vec<u8>)> {
        self.renderer.read_screen()
    }
}
