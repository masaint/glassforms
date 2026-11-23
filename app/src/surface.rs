//! Surface lighting: direct light with shadows for opaque and milky bodies (shaders/surface.wgsl).
//!
//! Glass reads as real because it shows what is around and behind it. An opaque or milky body shows its
//! own lighting, and before this it had none worth the name: clay took its light from nine smooth terms
//! of the whole lighting, a milky body from a coarse volume of light worked out beforehand, and nothing
//! cast a shadow on itself (a teapot's spout on its body, the folds of a knot).
//!
//! This pass runs after the tracer, only when the material wants it (`wanted`); plain glass never builds
//! or runs it, and the tracer's text for glass is unchanged. It reads the frame the tracer has just
//! written: the distance in its alpha gives back the surface (settled onto the smooth field and its
//! normal worked out here, so the tracer carries nothing more through its marching), and for each light
//! it adds one shadowed sample a frame. The frames' averaging and the antialiasing that follows motion
//! converge it.
//!
//! What each material takes from it:
//!   clay   the tracer gathers only the room (the nine terms of the lighting with the lights taken out,
//!          `Split`), keeps its own folds' darkening, and this pass adds the lights' direct light, shadowed;
//!   milky  the tracer is as it was (the light volume lights the body); this pass dims the frame's light
//!          where the lights are hidden from the surface and by the folds, and adds a little direct light
//!          at the surface (`MILKY_*`). These three numbers are the hook a glazed ceramic will build on:
//!          its body's colour (`base`), how much of the body's own light is kept (`k.y`, `k.z`) and the
//!          surface's direct share (`k.x`).
//! Metals are not drawn here: their tracer follows each reflection to the next surface of the shape
//! for `nb` bounces, so the shape already hides the lights and the room it stands in front of.

use bytemuck::{Pod, Zeroable};
use gf_core::env::{self, Sh9};
use gf_core::presets::Light;
use gf_core::state::State;

use crate::gpu::*;
use crate::render::{Variant, HDR};

pub const SURFACE_WGSL: &str = concat!(
    include_str!("shaders/pbr/hash.wgsl"),
    include_str!("shaders/pbr/sh9.wgsl"),
    include_str!("shaders/light/occlusion.wgsl"),
    include_str!("shaders/light/rig.wgsl"),
    include_str!("shaders/pbr/diffuse.wgsl"),
    include_str!("shaders/pbr/types.wgsl"),
    include_str!("shaders/pbr/dielectric.wgsl"),
    include_str!("shaders/dielectric_fresnel.wgsl"),
    include_str!("shaders/paint_interface.wgsl"),
    include_str!("shaders/paint_film.wgsl"),
    include_str!("shaders/film_pattern.wgsl"),
    include_str!("shaders/surface.wgsl"),
    "\n",
    include_str!("shaders/panel_irradiance.wgsl"),
    "\n",
    include_str!("shaders/wear_pigment.wgsl"),
);
pub const MAX_LIGHTS: usize = 12;

/// How far a milky body's own light is dimmed where the lights are hidden from its surface.
const MILKY_SHADOW: f32 = 0.8;
/// How far by the folds of its surface.
const MILKY_FOLDS: f32 = 0.5;
/// How much direct light its surface sends back besides (as a matte surface of its colour would).
const MILKY_DIRECT: f32 = 0.2;

/// Which materials are lit by the pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Kind {
    #[default]
    None,
    Clay,
    Milky,
    /// a milky body made a ceramic (`ceramic`): lit at its surface here, under its glaze
    Ceramic,
}

/// What the tracer state calls for: clay (of one scene, whatever lies on it), or a milky body of the
/// scene's one material. Off with GLASSFORMS_SURFACE=off (for comparing).
pub fn wanted(mat: &gf_core::material::Variant, clay: bool) -> Kind {
    if off() { return Kind::None; }
    if clay { return Kind::Clay; }
    if mat.bodies == 0 && mat.sss && !mat.metal && !mat.hollow { return if mat.ceramic { Kind::Ceramic } else { Kind::Milky }; }
    Kind::None
}

/// GLASSFORMS_SURFACE=off: never run the pass (for comparing); =direct: show only the direct light it
/// adds; =frame: show only the frame's own light as the pass leaves it.
fn env_var() -> &'static str {
    static V: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    V.get_or_init(|| std::env::var("GLASSFORMS_SURFACE").unwrap_or_default())
}
fn off() -> bool {
    env_var() == "off"
}
fn show() -> f32 {
    match env_var() { "direct" => 1.0, "frame" => 2.0, "number" => 3.0, "unshadowed" => 4.0, "lampopen" => 5.0, "lamplit" => 6.0, _ => 0.0 }
}

/// The clay's own text when the pass lights it: the nine terms then hold only the room, so they are
/// taken as they are (the drawing apart of light and dark was there to show a form that had no
/// shadows), and the lamps' light, which the pass now adds with its shadows, is not gathered here.
/// (The lines changed are also those lamps.rs and cast.rs look for, so neither weaves into them.)
pub fn weave(src: &str, v: &Variant) -> String {
    if v.surface != Kind::Clay { return src.to_string(); }
    const CLAY: [(&str, &str, &str); 3] = [
        ("surface.clay_stretch", "  let level = dot(light, vec3f(0.2126, 0.7152, 0.0722)) / lu.clay2.x;\n  light *= pow(max(level, 1e-4), lu.clay2.y);\n", ""),
        ("surface.clay_light", "  var light = nine(0, toEnv(n));", "  var light = max(nine(0, toEnv(n)), vec3f(0.0));"),
        ("surface.clay_sheen", "* (sheen * open);", "* (open * sheen);"),
    ];
    let counts = CLAY.map(|(name, old, _)| crate::weave::found(src, name, old));
    crate::weave::all_once("surface.clay", &counts);
    CLAY.iter().fold(src.to_string(), |s, (_, old, new)| s.replace(old, new))
}

/// One light at infinity as the shader has it (see `SLight` there).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Debug)]
pub struct SLight {
    pub a: [f32; 4],
    pub u: [f32; 4],
    pub v: [f32; 4],
    pub rgb: [f32; 4],
}

/// The owner's inverse placement and the two controls that tint an opaque substrate.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, PartialEq, Debug)]
pub struct Pigment {
    frame: [[f32; 4]; 3],
    dirt: [f32; 4],
    variation: [f32; 4],
}

impl Pigment {
    fn of(state: &State) -> Self {
        let p = &state.p;
        let cavity = gf_core::material::marks(p)[1] * p.age.max(0.0);
        let mottle = p.mottle.clamp(0.0, 1.0);
        if cavity == 0.0 && mottle == 0.0 { return Self::default(); }
        let frame = if state.plain() {
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        } else {
            let inv = state.stack.combine().inv[0];
            std::array::from_fn(|i| std::array::from_fn(|j| inv[i*4+j] as f32))
        };
        Self {
            frame,
            dirt: [p.cavity_r.clamp(0.0, 1.0) as f32, p.cavity_g.clamp(0.0, 1.0) as f32, p.cavity_b.clamp(0.0, 1.0) as f32, cavity as f32],
            variation: [mottle as f32, 0.0, 0.0, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Debug)]
pub struct SurfU {
    pub cam: [f32; 4],
    pub cam2: [f32; 4],
    pub k0: [f32; 4],
    pub k1: [f32; 4],
    pub obj: [[f32; 4]; 3],
    pub env: [[f32; 4]; 3],
    pub base: [f32; 4],
    pub k: [f32; 4],
    pub clay: [f32; 4],
    pub lights: [SLight; MAX_LIGHTS],
    pub cer: [f32; 4],
    pub nine: [[f32; 4]; 9],
    pub pigment: Pigment,
    pub film: [[f32; 4]; 3],
    pub gi: [f32; 4],
    pub floor_plane: [f32; 4],
    pub floor_lit: [f32; 4],
    pub floor_albedo: [f32; 4],
}

/// What the engine tells the pass about the material and the lights (the view is added each frame).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Lit {
    pub base: [f32; 4],
    pub k: [f32; 4],
    /// clay: x the light it gathers on average, y how far differences from that are drawn out (as the
    /// tracer's clay did before, now on the room and the direct light together); zero for other materials
    pub clay: [f32; 4],
    pub lights: [SLight; MAX_LIGHTS],
    /// a ceramic: x 1 = on, y wrap past the terminator, z the body's refractive index, w the cone its
    /// reflections are hidden by (zero for other materials)
    pub cer: [f32; 4],
    /// a ceramic: the room's light without the lights, nine terms as `env::sh_terms(.., SH_MATTE)` gives
    pub nine: [[f32; 4]; 9],
    pub pigment: Pigment,
    pub film: [[f32; 4]; 3],
    pub gi: [f32; 4],
    pub floor_plane: [f32; 4],
    pub floor_lit: [f32; 4],
    pub floor_albedo: [f32; 4],
}

impl Lit {
    /// Attach global-illumination and floor-plane parameters for surface self-bounce and floor-to-object bounce.
    pub fn with_gi(mut self, state: &State, floor_h: Option<f64>, floor_own: [f64; 3], sky_horizon: bool, shadow_w: f32) -> Self {
        let p = &state.p;
        if p.gi > 0.5 && p.gi_strength > 0.0 && !state.empty {
            self.gi[0] = p.gi_bounces.round().clamp(1.0, 4.0) as f32;
            self.gi[1] = p.gi_strength.clamp(0.0, 4.0) as f32;
            self.gi[2] = p.gi_samples.round().clamp(1.0, 4.0) as f32;
        } else {
            self.gi[0] = 0.0;
            self.gi[1] = 0.0;
            self.gi[2] = 1.0;
        }
        if let (Some(h), Some(f)) = (floor_h, gf_core::stage::Floor::of(p)) {
            let m = state.room_m();
            self.floor_plane = [m[1] as f32, m[4] as f32, m[7] as f32, h as f32];
            let fade = if sky_horizon { -(f.fade as f32) } else { f.fade as f32 };
            self.floor_lit = [floor_own[0] as f32, floor_own[1] as f32, floor_own[2] as f32, fade];
            self.floor_albedo = [f.albedo[0] as f32, f.albedo[1] as f32, f.albedo[2] as f32, shadow_w];
        }
        self
    }
}

/// The lights of a rig as the shader takes them: the direction to each one's middle and the two across
/// it, in the lights' own frame, as core/src/env.rs paints them; `rgb` turns each light's colour into
/// what the material wants.
pub fn lights_of(lights: &[Light], rgb: impl Fn([f64; 3]) -> [f64; 3]) -> ([SLight; MAX_LIGHTS], usize) {
    const D: f64 = std::f64::consts::PI / 180.0;
    let mut out = [SLight::zeroed(); MAX_LIGHTS];
    let mut n = 0;
    for l in lights.iter() {
        if n >= MAX_LIGHTS { break; }
        let c = rgb(l.rgb);
        if c[0] + c[1] + c[2] <= 0.0 || l.w <= 0.0 || l.h <= 0.0 { continue; }
        let ce = (l.el * D).cos();
        let a = [(l.az * D).sin() * ce, (l.el * D).sin(), -(l.az * D).cos() * ce];
        let u = [a[2], 0.0, -a[0]];
        let ul = (u[0] * u[0] + u[2] * u[2]).sqrt();
        let u = if ul > 1e-6 { [u[0] / ul, 0.0, u[2] / ul] } else { [1.0, 0.0, 0.0] };
        let v = gf_core::math::cross(a, u);
        out[n] = SLight {
            a: [a[0] as f32, a[1] as f32, a[2] as f32, (l.w * D).tan() as f32],
            u: [u[0] as f32, u[1] as f32, u[2] as f32, (l.h * D).tan() as f32],
            v: [v[0] as f32, v[1] as f32, v[2] as f32, l.soft.max(1e-3) as f32],
            rgb: [c[0] as f32, c[1] as f32, c[2] as f32, 0.0],
        };
        n += 1;
    }
    (out, n)
}

/// The scene's analytic area lights (`rig` lights scaled by `lightGain`, plus the physical sky's
/// directional solar source when active — already scaled by `sunPower * skyGain`), transformed
/// by `scale` into the tracer/surface buffer format.
pub fn scene_lights_atten(
    state: &State,
    rig_lit: bool,
    sky_active: bool,
    sun_atten: f64,
    scale: impl Fn([f64; 3]) -> [f64; 3],
) -> ([SLight; MAX_LIGHTS], usize) {
    let gain = state.p.light_gain;
    let mut all = Vec::new();
    if rig_lit {
        for l in &state.rig().lights {
            let mut copy = l.clone();
            copy.rgb = copy.rgb.map(|c| c * gain);
            all.push(copy);
        }
    }
    if sky_active {
        if let Some(sun) = gf_core::sky::Sky::of(&state.p).sun_as_light_scaled(&state.lights_to_room(), sun_atten) {
            all.push(sun);
        }
    }
    lights_of(&all, scale)
}

pub fn scene_lights_scaled(
    state: &State,
    rig_lit: bool,
    sky_active: bool,
    scale: impl Fn([f64; 3]) -> [f64; 3],
) -> ([SLight; MAX_LIGHTS], usize) {
    scene_lights_atten(state, rig_lit, sky_active, 1.0, scale)
}

pub fn scene_lights(state: &State, rig_lit: bool, sky_active: bool) -> ([SLight; MAX_LIGHTS], usize) {
    scene_lights_atten(state, rig_lit, sky_active, 1.0, |c| c)
}

/// The lighting's nine terms (unnormalised, as `env::sh_of_map` makes them) of the lights alone: what the
/// room's terms are the whole lighting's less. Below the room's horizon a floor hides the lights (the
/// studio paints the floor over them), so there they are left out.
pub fn lights_sh(state: &State, gain: f64) -> Sh9 {
    let (w, h) = (env::ENV_W, env::ENV_H);
    let r = state.rig();
    let mut px = env::paint_gain(&r.lights, [0.0; 3], gain, w, h);
    if gf_core::stage::Stage::of(&state.p).floor.is_some() {
        let up = gf_core::math::apply(&gf_core::math::transpose(&state.lights_to_room()), [0.0, 1.0, 0.0]);
        for j in 0..h {
            for i in 0..w {
                let d = env::paint_dir((i as f64 + 0.5) / w as f64, (j as f64 + 0.5) / h as f64);
                if d[0] * up[0] + d[1] * up[1] + d[2] * up[2] < 0.0 {
                    let o = (j * w + i) * 4;
                    px[o..o + 3].fill(0.0);
                }
            }
        }
    }
    env::sh_of_map(&px, w, h, 4, env::paint_dir)
}

pub use gf_core::sky::light_open_factor;

/// Sample the physical sky dome into 6 soft octahedral area patches (in the lights' frame) so
/// `surface.wgsl`'s `milky` pass receives full-sphere sky-dome direct shading and smooth self-shadowing
/// without a `dot(n, sun) == 0` terminator step when the solar disc is behind the body.
fn sample_pano_dir(pano: &[f32], pw: usize, ph: usize, d_r: [f64; 3]) -> [f64; 3] {
    use std::f64::consts::{PI, TAU};
    let su = ((d_r[2].atan2(d_r[0]) / TAU + 0.5) * pw as f64).rem_euclid(pw as f64);
    let sv = (d_r[1].clamp(-1.0, 1.0).acos() / PI * ph as f64).clamp(0.0, (ph - 1) as f64);
    let x = (su as usize).min(pw - 1);
    let y = (sv as usize).min(ph - 1);
    let o = (y * pw + x) * 4;
    [pano[o] as f64, pano[o + 1] as f64, pano[o + 2] as f64]
}

fn milky_lights(
    state: &State,
    lit: bool,
    sky: bool,
    sun_atten: f64,
    pano: Option<(&[f32], usize, usize)>,
) -> ([SLight; MAX_LIGHTS], usize, usize) {
    let gain = state.p.light_gain;
    let mut all = Vec::new();
    if lit {
        for l in &state.rig().lights {
            let mut copy = l.clone();
            copy.rgb = copy.rgb.map(|c| c * gain);
            all.push(copy);
        }
    }
    let mut direct_len = all.len();
    if sky {
        let sky_model = gf_core::sky::Sky::of(&state.p);
        let l2r = state.lights_to_room();
        if let Some(sun) = sky_model.sun_as_light_scaled(&l2r, sun_atten) {
            all.push(sun);
        }
        direct_len = all.len();
        if sky_model.active && all.len() < MAX_LIGHTS {
            let lut = sky_model.transmittance_lut();
            let ground_lit = sky_model.ground_radiance(&lut);
            let r2l = gf_core::math::transpose(&l2r);
            let half_deg = 34.0;
            let soft = 0.45;
            // Weight each of the 6 octahedral patches so overlapping 34° lobes integrate to pi sr.
            let scale_k = 1.94 / gf_core::sky::quad_solid_angle(half_deg, soft);
            let yaw0 = sky_model.sun_yaw.to_radians();
            let el_h = 14.0_f64.to_radians();
            let el_sample = 22.0_f64.to_radians();
            let mut dirs: Vec<([f64; 3], [f64; 3])> = Vec::with_capacity(6);
            dirs.push(([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]));
            for k in 0..4 {
                let yaw = yaw0 + (k as f64) * std::f64::consts::FRAC_PI_2;
                let (sy, cy) = yaw.sin_cos();
                let d_dir = [sy * el_h.cos(), el_h.sin(), -cy * el_h.cos()];
                let d_smp = [sy * el_sample.cos(), el_sample.sin(), -cy * el_sample.cos()];
                dirs.push((d_dir, d_smp));
            }
            dirs.push(([0.0, -1.0, 0.0], [0.0, -0.5, 0.0]));
            for (d_room, d_sample) in dirs {
                if all.len() >= MAX_LIGHTS { break; }
                let rad = match pano {
                    Some((px, pw, ph)) => sample_pano_dir(px, pw, ph, d_sample),
                    None => sky_model.eval_dome(d_sample, &lut, ground_lit),
                };
                let d_l = gf_core::math::apply(&r2l, d_room);
                let az = d_l[0].atan2(-d_l[2]).to_degrees();
                let el = d_l[1].clamp(-1.0, 1.0).asin().to_degrees();
                all.push(Light {
                    az,
                    el,
                    w: half_deg,
                    h: half_deg,
                    rgb: rad.map(|c| c * scale_k),
                    soft,
                });
            }
        }
    }
    let (_, n_direct) = lights_of(&all[..direct_len.min(all.len())], |c| c);
    let (lights, n_all) = lights_of(&all, |c| c);
    (lights, n_all, n_direct)
}

/// The rig's (and sky sun + dome's) lights for a milky body: as they are, made `gain` times as bright.
pub fn milky(state: &State, lit: bool, sky: bool, body: [f64; 3], room: &Sh9) -> Lit {
    milky_atten(state, lit, sky, 1.0, None, body, room)
}

pub fn milky_atten(
    state: &State,
    lit: bool,
    sky: bool,
    sun_atten: f64,
    pano: Option<(&[f32], usize, usize)>,
    body: [f64; 3],
    room: &Sh9,
) -> Lit {
    let weight = state.p.sss.clamp(0.0, 1.0) as f32;
    let op = state.p.sss_opaque.clamp(0.0, 1.0);
    let direct = MILKY_DIRECT + 0.14 * op as f32;
    let shadow = MILKY_SHADOW - 0.32 * op as f32;
    let folds = MILKY_FOLDS + 0.12 * op as f32;
    let sub_pow = 1.0 + 0.32 * op;
    let tinted = body.map(|c| c.clamp(0.0, 1.0).powf(sub_pow) as f32);
    let (lights, n, n_direct) = milky_lights(state, lit, sky, sun_atten, pano);
    Lit {
        base: [tinted[0], tinted[1], tinted[2], n as f32],
        k: [direct * weight, shadow * weight, folds * weight, 0.0],
        clay: [0.0; 4],
        lights,
        cer: [0.0; 4],
        nine: env::sh_terms(room, env::SH_MATTE),
        pigment: Pigment::default(),
        film: [[0.0; 4]; 3],
        gi: [0.0, 0.0, 1.0, n_direct as f32],
        floor_plane: [0.0; 4],
        floor_lit: [0.0; 4],
        floor_albedo: [0.0; 4],
    }
}

/// How far a ceramic's diffuse light wraps past the terminator: a little for an opaque body, more the
/// more of its inner light shows (light scattered under the surface carries round the edge of a shadow).
const WRAP_MIN: f64 = 0.06;
const WRAP_THROUGH: f64 = 0.35;

/// The rig's (and sky sun's) lights for a ceramic body, as they are (made `gain` times as bright, the units its light
/// volume and the room's nine terms are in), its body's colour, the share of its inner light the tracer
/// keeps (the rest is the body's surface light, given here), the glaze's haze, and the room's nine terms
/// (`room`, the lighting less the lights).
pub fn ceramic(state: &State, lit: bool, sky: bool, body: [f64; 3], through: f64, haze: f64, room: &Sh9) -> Lit {
    ceramic_atten(state, lit, sky, 1.0, body, through, haze, room)
}

#[allow(clippy::too_many_arguments)]
pub fn ceramic_atten(
    state: &State,
    lit: bool,
    sky: bool,
    sun_atten: f64,
    body: [f64; 3],
    through: f64,
    haze: f64,
    room: &Sh9,
) -> Lit {
    let (lights, n) = scene_lights_atten(state, lit, sky, sun_atten, |c| c);
    let through = through.clamp(0.0, 1.0);
    let wrap = WRAP_MIN + WRAP_THROUGH * through;
    let p = &state.p;
    let mut film = [[0.0; 4]; 3];
    if p.film > 0.0 {
        let drain = gf_core::material::film::numbers(p);
        film[0] = [p.film_ior as f32,p.film_var.clamp(0.0,1.0) as f32,p.film_scale as f32,p.film_flow as f32];
        film[1] = std::array::from_fn(|i| drain[i] as f32);
        film[2] = std::array::from_fn(|i| drain[i+4] as f32);
    }
    Lit {
        base: [body[0] as f32, body[1] as f32, body[2] as f32, n as f32],
        // (the share of the light inside that the tracer keeps is light that went deeper before coming
        // out: the surface's own diffuse light gives way to it by only half as much, which keeps a white
        // body white at the exposure clay gets under the same lights)
        // Standalone paint receives this ray through air. Negative tags select the film,
        // with -2 denoting a distinct authored clear coat underneath it.
        k: [(1.0 - 0.5 * through) as f32, gf_core::material::coat_index(&state.p) as f32,
            if state.p.film > 0.0 { if state.p.coat > 0.0 { -2.0 } else { -1.0 } } else if state.p.coat > 0.0 { 1.0 } else { 0.0 }, 0.0],
        clay: [0.0; 4],
        lights,
        cer: [1.0, wrap as f32, state.p.ior as f32, (0.03 + haze.clamp(0.0, 0.6)) as f32],
        nine: env::sh_terms(room, env::SH_MATTE),
        pigment: Pigment::of(state),
        film,
        gi: [0.0, 0.0, 1.0, n as f32],
        floor_plane: [0.0; 4],
        floor_lit: [0.0; 4],
        floor_albedo: [0.0; 4],
    }
}

pub struct Surface {
    layout: wgpu::BindGroupLayout,
    pipe: wgpu::RenderPipeline,
    u: wgpu::Buffer,
    /// a copy of the traced frame (the pass reads it while drawing onto the frame itself), its size
    src: Option<(wgpu::Texture, wgpu::TextureView, u32, u32)>,
    group: Option<wgpu::BindGroup>,
    lit: Option<Lit>,
}

impl Surface {
    pub fn new(gpu: &Gpu) -> Self {
        let frag = wgpu::ShaderStages::FRAGMENT;
        let layout = gpu.bind_layout("surface", &[
            uniform_entry(0, frag, false),
            texture_entry(1, wgpu::TextureViewDimension::D2, false),
            texture_entry(2, wgpu::TextureViewDimension::D3, true),
            sampler_entry(3),
            uniform_entry(4, frag, false),
        ]);
        let module = gpu.shader("surface", SURFACE_WGSL, false);
        // colour: what the pass writes + the frame's colour times its alpha; alpha (the distance): kept
        let blend = wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::SrcAlpha, operation: wgpu::BlendOperation::Add },
            alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
        };
        let pipe = gpu.pipeline(&PipeDesc { label: "surface", module: &module, vs: "vs", fs: "fs", layouts: &[&layout], format: HDR, blend: Some(blend), strip: false, buffers: &[] });
        Self { layout, pipe, u: gpu.uniform_buffer("surface", std::mem::size_of::<SurfU>() as u64), src: None, group: None, lit: None }
    }

    /// The material and lights to light with (None: the pass is not run).
    pub fn set(&mut self, lit: Option<Lit>) {
        self.lit = lit;
    }
    pub fn on(&self) -> bool {
        self.lit.is_some()
    }

    /// Forget the group (the volume or the frame's size changed): it is made again when next drawn.
    pub fn unbind(&mut self) {
        self.group = None;
    }

    /// Light the traced frame `frame` (`tex`, `w` x `h`). `tu`: what the tracer was told this frame.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(&mut self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, tex: &wgpu::Texture, frame: &wgpu::TextureView, vol: &wgpu::TextureView, smp: &wgpu::Sampler, lamps: &wgpu::Buffer, tu: &crate::render::TraceU, grown_hb: Option<f32>, near: bool) {
        let Some(lit) = self.lit else { return };
        let (w, h) = (tex.width(), tex.height());
        if !self.src.as_ref().is_some_and(|s| s.2 == w && s.3 == h) {
            let t = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("surface: the traced frame"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HDR,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let v = t.create_view(&wgpu::TextureViewDescriptor::default());
            self.src = Some((t, v, w, h));
            self.group = None;
        }
        let src = self.src.as_ref().expect("just made");
        if self.group.is_none() {
            self.group = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("surface"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.u.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&src.1) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(vol) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(smp) },
                    wgpu::BindGroupEntry { binding: 4, resource: lamps.as_entire_binding() },
                ],
            }));
        }
        let su = SurfU {
            cam: [tu.res[0], tu.res[1], tu.jitter[0], tu.jitter[1]],
            cam2: [tu.shift[0], tu.shift[1], tu.dist, tu.tan_fov],
            k0: [tu.vres, grown_hb.unwrap_or(gf_core::HB as f32), (tu.frame % 65536) as f32, show()],
            k1: [tu.ns.clamp(1, 32) as f32, tu.eye[0], tu.eye[1], 0.0],
            obj: tu.obj,
            env: tu.env,
            base: lit.base,
            k: [lit.k[0], lit.k[1], lit.k[2], if near { 1.0 } else { 0.0 }],
            clay: lit.clay,
            lights: lit.lights,
            cer: lit.cer,
            nine: lit.nine,
            pigment: lit.pigment,
            film: lit.film,
            gi: lit.gi,
            floor_plane: lit.floor_plane,
            floor_lit: lit.floor_lit,
            floor_albedo: lit.floor_albedo,
        };
        gpu.queue.write_buffer(&self.u, 0, bytemuck::bytes_of(&su));
        let (render_w, render_h) = (tu.res[0] as u32, tu.res[1] as u32);
        enc.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyTextureInfo { texture: &src.0, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::Extent3d { width: render_w, height: render_h, depth_or_array_layers: 1 },
        );
        let mut rp = pass(enc, frame, None, false);
        rp.set_viewport(0.0, 0.0, tu.res[0], tu.res[1], 0.0, 1.0);
        rp.set_pipeline(&self.pipe);
        rp.set_bind_group(0, self.group.as_ref().expect("just made"), &[]);
        rp.draw(0..3, 0..1);
    }
}
