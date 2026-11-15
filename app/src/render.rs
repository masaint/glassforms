//! Drawing: trace one frame, fold it into the picture so far, bloom it, tone map it.
//! (renderFrame and present of the web version's engine.js.)

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

use gf_core::material::{self, MU_FLOATS};

use crate::bake::Volume;
use crate::denoise::{self, Denoise};
use crate::dof::{Dof, Focus};
use crate::frost::{self, Frost};
use crate::gpu::*;

const TRACE_WGSL: &str = include_str!("shaders/trace.wgsl");
const TRACE_LIGHT_WGSL: &str = include_str!("shaders/trace_light.wgsl");
/// What is put into the tracer for materials other than plain glass (gf_core::material::tracer_parts).
/// (with everything a rough surface does, which is kept in a file of its own)
pub const TRACE_MATERIAL_WGSL: &str = concat!(include_str!("shaders/trace_material.wgsl"), include_str!("shaders/film.wgsl"), include_str!("shaders/frost.wgsl"), include_str!("shaders/inner.wgsl"));
pub const TRACE_BODIES_WGSL: &str = include_str!("shaders/trace_bodies.wgsl");
const ACCUM_WGSL: &str = include_str!("shaders/accum.wgsl");
/// The pass that folds a frame into the picture so far (the denoiser's pass over time is made from it).
pub const ACCUM_SOURCE: &str = ACCUM_WGSL;
const POST_WGSL: &str = include_str!("shaders/post.wgsl");

pub const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const ACC: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
pub const LDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Frames after which a still picture has converged and drawing stops.
pub const MAX_ACCUM: u32 = 1024;

/// The tracer's uniform block (see trace.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Debug)]
pub struct TraceU {
    pub res: [f32; 2],
    pub jitter: [f32; 2],
    pub shift: [f32; 2],
    pub focus: [f32; 2],
    pub obj: [[f32; 4]; 3],
    pub env: [[f32; 4]; 3],
    pub absorb: [f32; 4],
    pub dist: f32,
    pub tan_fov: f32,
    pub lens: f32,
    pub blades: f32,
    pub bokeh_rot: f32,
    pub ior: f32,
    pub disp: f32,
    pub vres: f32,
    pub frame: i32,
    pub nw: i32,
    pub ns: i32,
    pub nb: i32,
    pub lens_n: i32,
    pub pad: [i32; 3],
    /// the lens, for a tracer built to trace its first ray through any lens (camera.wgsl: Cam). The
    /// plain tracer's block ends before it.
    pub cam: [[f32; 4]; 4],
}

/// What the tracer is told besides `TraceU` when it is built for an environment picture or for clay
/// (see trace_light.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Debug)]
pub struct LightU {
    /// brightness of the picture shown behind the shape; unused; the level of it shown there; unused
    pub picture: [f32; 4],
    /// the clay's colour; how strong its sheen is
    pub clay: [f32; 4],
    /// the light a matte surface gathers on average; how far differences from that are drawn out; unused
    pub clay2: [f32; 4],
    /// the lighting as nine smooth terms a colour: for a matte surface, then for the sheen
    pub nine: [[f32; 4]; 18],
}

/// Where reflected and refracted rays find their light. It is one lookup either way: with an
/// environment picture, the lights' map is added into the copy of the picture that the glass reads
/// beforehand (see `Engine`), not looked up beside it. (Two lookups, or one lookup and a little more
/// arithmetic, made the tracer of a turning 4K view take 5.4 ms where one takes 3.9: measured, with
/// the picture's lookup replaced by all sorts of things. The shader compiler does something else with
/// the tracer's innermost function once it grows past a certain size.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum LightFrom {
    /// the lights' map
    #[default]
    Lights,
    /// the environment picture as made ready for the glass: softened, with the lights' map added
    Picture,
}

/// What a tracer is built to draw, apart from how it marches. The default is what the program has
/// always drawn, glass lit by the lights' map, and its shader is the tracer as it stands in trace.wgsl
/// with nothing added.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Variant {
    pub light: LightFrom,
    /// rays from the camera that miss the shape show the environment picture
    pub backdrop: bool,
    /// matte clay instead of glass
    pub clay: bool,
    /// which parts of the material are in use (none: plain glass, the tracer as it always was)
    pub mat: material::Variant,
    /// what of the studio is drawn where a ray from the camera misses the shape (none: as it always was)
    pub stage: crate::stage::Seen,
    /// what the bodies cast on each other is drawn: shadows and caustics (cast.rs)
    pub cast: bool,
    /// what is traced by chance is written apart from the rest and denoised (denoise.rs)
    pub denoise: bool,
    /// lamps stand in the scene (lamps.rs): every lookup of the lights meets them
    pub near: bool,
    /// the stage is larger than ever (layers placed far out: core/src/layers.rs `box_k`): the tracer reads
    /// the box its volume spans from its numbers (`TraceU::pad[2]`), so one text serves every size
    pub grown: bool,
}

/// How a tracer marches (this differs by family).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct March {
    pub step_scale: f64,
    pub steps_out: u32,
    pub steps_in: u32,
    pub layers: u32,
}

/// The tracer's source for one way of marching and one way of drawing. `profile`: built to count its
/// lookups instead of making a picture (red = plain lookups while marching, green = smooth (eight-tap)
/// lookups, blue = plain lookups that found the surface far off).
pub fn tracer_source(m: &March, v: Variant, profile: bool) -> String {
    tracer_from(TRACE_WGSL, m, v, profile)
}

/// The same with the first ray traced through any lens (camera.rs): another kind of picture than
/// perspective, straight lines bowed, a leaning plane of focus, highlights with a shape of their own.
pub fn tracer_source_lens(m: &March, v: Variant, profile: bool) -> String {
    crate::camera::finish(tracer_from(&crate::camera::tracer(TRACE_WGSL), m, v, profile))
}

fn tracer_from(template: &str, m: &March, v: Variant, profile: bool) -> String {
    let prof = if profile { ["var<private> prof: vec3f;", "prof.x += 1.0; if (abs(v) > 0.06) { prof.z += 1.0; } ", "prof.y += 1.0; ", "if (true) { return vec4f(prof.x, prof.y, prof.z, depth); } "] } else { [""; 4] };
    let lights = "textureSampleLevel(envTex, envSmp, uv, 0.0).rgb";
    // the lighting's own declarations: none for glass lit by the lights' map
    let lit = v.light != LightFrom::Lights || v.backdrop || v.clay;
    // the material's: none for plain glass, which is then the tracer exactly as it always was
    // (a scene of several bodies with materials of their own has a tracer of its own; clay hides what
    // things are made of, so it is drawn as ever)
    let mat = if v.mat.bodies > 0 && !v.clay {
        material::bodies_parts(&v.mat, TRACE_MATERIAL_WGSL, TRACE_BODIES_WGSL, v.light == LightFrom::Picture)
    } else {
        material::tracer_parts(&v.mat, TRACE_MATERIAL_WGSL, v.light == LightFrom::Picture, v.clay)
    };
    let decl = format!("{}{}{}", if lit { TRACE_LIGHT_WGSL.trim_end() } else { "" }, mat.decl, crate::stage::tracer_decl(v.stage, v.backdrop, mat.shade));
    let src = template
        .replace("{{PROFILE_DECL}}", prof[0])
        .replace("{{PROFILE_MAP}}", prof[1])
        .replace("{{PROFILE_MAPN}}", prof[2])
        .replace("{{PROFILE_OUT}}", prof[3])
        .replace("{{STEP_SCALE}}", &format!("{:.2}", m.step_scale))
        .replace("{{STEPS_OUT}}", &m.steps_out.to_string())
        .replace("{{STEPS_IN}}", &m.steps_in.to_string())
        .replace("{{LAYERS}}", &m.layers.max(mat.min_layers).to_string())
        .replace("{{LIGHT_DECL}}", &decl)
        .replace("{{ENV_LIGHT}}", &match v.light {
            LightFrom::Lights => lights.to_string(),
            LightFrom::Picture => "picture(rd)".to_string(),
        })
        .replace("{{BACKGROUND_DIR}}", if v.stage.backdrop { "stageBackdrop(rd)" } else if v.backdrop { "backdrop(rd)" } else { "vec3f(0.0)" })
        .replace("{{BACKGROUND}}", crate::stage::background(v.stage, v.backdrop))
        .replace("{{SHADE}}", mat.shade)
        .replace("{{GRAZE}}", mat.graze);
    // a grown stage: the volume spans a larger box, and the ball that holds the scene is larger
    let src = if v.grown { grown(&src) } else { src };
    // the lamps, and then what the bodies cast, are put in last (never into the default tracer)
    let src = if v.near { crate::lamps::weave(&src) } else { src };
    let src = if v.cast { crate::cast::weave(&src).unwrap_or(src) } else { src };
    // (and with both, the lamps' shadows)
    let src = if v.cast && v.near { crate::lamps::weave_shadows(&src) } else { src };
    // a rough clear body's tracer also writes, beside the picture, how far what is seen through the
    // surface is to be blurred afterwards (frost.rs)
    let src = if rough_split(&v, profile) { frost::with_aux(&src) } else { src };
    // a tracer drawn by chance and denoised also writes, beside the picture, the light it found inside
    // the body, which is noisy (the picture then holds the rest: denoise.rs)
    if v.denoise && !profile { denoise::with_split(&src, v.mat.rough_ref) } else { src }
}

/// A tracer's text for a stage larger than ever: the volume's box and the ball the scene fits in, which
/// the tracer of a stage as it always was holds as constants (and that text stays unchanged), are read
/// from its numbers instead (`TraceU::pad[2]`: the box's half-width, as a float's bits), so that one text
/// serves every size and a layer dragged across a size needs no new tracer.
pub fn grown(src: &str) -> String {
    let head = "fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
";
    src.replace("const BOUND: f32 = 1.47;", "var<private> BOUND: f32;")
        .replace("const HB: f32 = 1.5;", "var<private> HB: f32;")
        .replace(head, &format!("{head}  HB = bitcast<f32>(u.pad2);
  BOUND = HB * 0.98;
"))
}

/// Is the stage for rough glass run after this tracer? Not when the tracer is denoised: then what is
/// seen through the surface is in the part the denoiser filters (and the targets of both would be more
/// than a pass may write).
fn rough_split(v: &Variant, profile: bool) -> bool {
    frost::splits(&v.mat, v.clay, profile) && !(v.denoise && !profile)
}

fn tracer_pipeline(gpu: &Gpu, layout: &wgpu::BindGroupLayout, cast: Option<&wgpu::BindGroupLayout>, src: &str) -> wgpu::RenderPipeline {
    let module = gpu.shader("trace", src, true);
    if src.contains(denoise::MARK) { return denoise::tracer_pipeline(gpu, layout, cast, &module); }
    // (a tracer with casting in it reads a second group: what the bodies cast)
    if src.contains("struct FrOut") { return frost::tracer_pipeline(gpu, layout, cast, &module); }
    let layouts: Vec<&wgpu::BindGroupLayout> = std::iter::once(layout).chain(cast).collect();
    gpu.pipeline(&PipeDesc { label: "trace", module: &module, vs: "vs", fs: "fs", layouts: &layouts, format: HDR, blend: None, strip: false, buffers: &[] })
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AccumU {
    res: [f32; 2],
    cam: [f32; 2],
    cam_p: [f32; 2],
    shift: [f32; 2],
    shift_p: [f32; 2],
    w: f32,
    moving: f32,
    obj: [[f32; 4]; 3],
    objp: [[f32; 4]; 3],
    debug: f32,
    spread: f32,
    /// 1 = an environment picture is shown where there is no glass (it moves when the lights turn)
    sky: f32,
    pad: f32,
    /// the lights' frame now and at the previous frame
    env: [[f32; 4]; 3],
    envp: [[f32; 4]; 3],
    /// the room as the camera sees it now and at the previous frame (three rows each)
    room: [[f32; 4]; 3],
    roomp: [[f32; 4]; 3],
    /// 1 = there is a floor; its height in the room now; and at the previous frame; unused
    floor: [f32; 4],
    /// the lens now and at the previous frame (read only by the pass built for lenses other than the plain one)
    lens: [[f32; 4]; 4],
    lens_p: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BloomU {
    texel: [f32; 2],
    dst_res: [f32; 2],
    gain: f32,
    pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Debug)]
pub struct FinalU {
    pub res: [f32; 2],
    pub exposure: f32,
    pub bloom_amt: f32,
    pub vignette: f32,
    pub grain: f32,
    pub raw: f32,
    pub frame: i32,
    pub planes: [f32; 4],
    pub view: [f32; 4],
    /// the tone curve and the grade (gf_core::stage::Grade::uniforms)
    pub grade: [[f32; 4]; 2],
    /// the lens, and its colour fringing and fall-off of light (read only by the final pass built for them)
    pub lens: [[f32; 4]; 4],
    pub fx: [f32; 4],
}

/// Everything about a frame that is not the tracer's own block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Look {
    /// antialiasing while moving: how much of the previous picture is kept each frame (0 = off)
    pub smooth: f32,
    pub spread: f32,
    pub debug: f32,
    pub exposure: f32,
    pub bloom: f32,
    pub vignette: f32,
    pub grain: f32,
    pub raw: f32,
    /// focus aid: near plane, far plane, strength
    pub planes: [f32; 3],
    /// keep the grain still (a still picture, or a frame of a video)
    pub still_grain: bool,
    /// focus done to the finished picture (`Focus::OFF`: none, or the tracer does it through its lens)
    pub focus: Focus,
    /// the tone curve and the grade (gf_core::stage::Grade::uniforms)
    pub grade: [[f32; 4]; 2],
    /// the lens, for the passes after the tracer
    pub lens: crate::camera::Lens,
}

/// What the antialiasing needs to know of the frame before.
#[derive(Clone, Copy)]
struct PrevView {
    obj: [[f32; 4]; 3],
    env: [[f32; 4]; 3],
    /// the room as the camera saw it, and where the floor stood in it
    room: [[f32; 4]; 3],
    floor: f32,
    cam: [f32; 2],
    shift: [f32; 2],
    lens: [[f32; 4]; 4],
}

struct Mip {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    w: u32,
    h: u32,
    down: wgpu::BindGroup,
    /// reads this level while adding it onto the next larger one
    up: Option<wgpu::BindGroup>,
}

struct Targets {
    w: u32,
    h: u32,
    tw: u32,
    th: u32,
    trace_view: wgpu::TextureView,
    accum: [wgpu::Texture; 2],
    accum_view: [wgpu::TextureView; 2],
    /// accumulate: read accum[i] as the picture so far (and the new frame), write the other
    g_accum: [wgpu::BindGroup; 2],
    /// the same, reading the traced frame as the stage for rough glass has blurred it
    g_accum_rough: [wgpu::BindGroup; 2],
    /// the first bloom level reads accum[i]; the third of each of these reads the focused picture instead
    g_down0: [wgpu::BindGroup; 3],
    mips: Vec<Mip>,
    g_final: [wgpu::BindGroup; 3],
    final_tex: wgpu::Texture,
    final_view: wgpu::TextureView,
    /// the screen again in 16-bit floats, with no 8-bit dither, for video with more than 8 bits (when asked for)
    deep: Option<(wgpu::Texture, wgpu::TextureView, [wgpu::BindGroup; 3])>,
}

pub struct Renderer {
    gpu: Gpu,
    lin: wgpu::Sampler,
    env_smp: wgpu::Sampler,
    l_trace: wgpu::BindGroupLayout,
    /// the group through which a tracer reads what the bodies cast (its layout, and the group itself
    /// while something casts: see cast.rs)
    l_cast: wgpu::BindGroupLayout,
    g_cast: Option<wgpu::BindGroup>,
    l_accum: wgpu::BindGroupLayout,
    l_bloom: wgpu::BindGroupLayout,
    l_final: wgpu::BindGroupLayout,
    trace_pipes: HashMap<String, wgpu::RenderPipeline>,
    trace_key: String,
    p_accum: wgpu::RenderPipeline,
    p_down: wgpu::RenderPipeline,
    p_up: wgpu::RenderPipeline,
    p_final: wgpu::RenderPipeline,
    u_trace: wgpu::Buffer,
    /// what the tracer is told about the environment picture and the clay
    u_light: wgpu::Buffer,
    last_light: Option<LightU>,
    /// the material's numbers (read only by tracers built for something other than plain glass)
    u_mat: wgpu::Buffer,
    last_mat: Option<[f32; MU_FLOATS]>,
    /// the studio's numbers (read only by tracers built with a backdrop or a floor), and what they are
    u_stage: wgpu::Buffer,
    u_lamps: wgpu::Buffer,
    stage: crate::stage::StageU,
    /// the bodies' volume (a distance to each body for every voxel) and how many bodies it holds, while
    /// the scene has more than one; what stands in for it otherwise; and the bodies' materials
    bodies: Option<(wgpu::TextureView, u32)>,
    no_bodies: wgpu::TextureView,
    u_bodies: wgpu::Buffer,
    last_bodies: Vec<f32>,
    /// what the tracer is told about what lies inside the body (cracks, grains, fibres, a changing tint)
    u_inner: wgpu::Buffer,
    last_inner: Vec<f32>,
    u_inner_bodies: wgpu::Buffer,
    last_inner_bodies: Vec<f32>,
    /// the share of a rough surface's roughness that is sampled, as the tracer has it now
    rough_share: f32,
    /// the lights' map again with softened copies of itself as further levels, for rough and milky
    /// materials (plain glass never reads it); what stands in for it until it is wanted; how it is read
    soft: Option<(wgpu::Texture, wgpu::TextureView)>,
    no_soft: wgpu::TextureView,
    soft_smp: wgpu::Sampler,
    /// the light inside a milky body, worked out beforehand (scatter.rs), and what stands in for it
    /// until there is one; with several bodies, each milky body's own (the first is the scene's)
    glow: [Option<wgpu::TextureView>; 4],
    no_glow: wgpu::TextureView,
    /// the environment picture (a cube with softer levels, for behind the shape; and the copy made
    /// ready for the glass), how they are read, and what stands in for them while there is none
    picture: Option<(wgpu::TextureView, wgpu::TextureView)>,
    picture_smp: wgpu::Sampler,
    no_picture: wgpu::TextureView,
    /// a tracer being built on another thread: which, and where it will arrive
    building: Option<(String, std::sync::mpsc::Receiver<wgpu::RenderPipeline>)>,
    /// the lights' map and the volume the tracer reads, and how it marches and draws
    scene: Option<(wgpu::TextureView, wgpu::TextureView)>,
    march: Option<March>,
    variant: Variant,
    u_accum: wgpu::Buffer,
    u_final: wgpu::Buffer,
    u_final_deep: wgpu::Buffer,
    p_final_deep: wgpu::RenderPipeline,
    /// also keep the screen in 16-bit floats (see `read_screen16`)
    pub deep: bool,
    g_trace: Option<wgpu::BindGroup>,
    t: Option<Targets>,
    /// which of the two accumulation pictures is the current one
    acc_idx: usize,
    /// frames in the running average
    pub accum_n: u32,
    pub frame_no: u32,
    jitter_n: u32,
    /// the last frame was folded in by the antialiasing that follows motion
    was_taa: bool,
    prev: Option<PrevView>,
    last_final: Option<FinalU>,
    /// build the tracer so that it counts its lookups instead of making a picture
    pub profile: bool,
    /// focus as a stage between the picture so far and the bloom
    dof: Dof,
    /// what is seen through rough glass, blurred: a stage between the tracer and the picture so far
    frost: Frost,
    /// the last frame put on screen went through that stage
    focused: bool,
    /// what is traced by chance, denoised: a stage of its own beside the picture so far
    dn: Denoise,
    /// which of the two pictures so far is shown: the current one, or (denoised) the other, into which
    /// the current one has been written with the denoised part added
    shown: usize,
    /// the camera's own pipelines, and whether the tracer in use traces its first ray through any lens
    pub cam: crate::camera::Pipes,
}

/// The tone map of post.wgsl's final pass, done here in 64-bit arithmetic for a 16-bit still: the picture
/// so far plus bloom, vignette, exposure, the ACES curve and the sRGB encoding; no grain, no focus aid.
/// `hdr`: RGBA floats, `w` x `h`, top row first. `bloom`: the first bloom level (read with interpolation).
/// Returns RGB, 16 bits a channel.
pub fn tone_map16(hdr: &[f32], bloom: &(u32, u32, Vec<f32>), w: u32, h: u32, p: &gf_core::settings::Settings) -> Vec<u16> {
    let (bw, bh, b) = (bloom.0 as usize, bloom.1 as usize, &bloom.2);
    // seven bloom levels at most: the number the renderer made for this size
    let (mut mw, mut mh, mut levels) = (w, h, 0u32);
    for _ in 0..7 {
        mw = (mw >> 1).max(1);
        mh = (mh >> 1).max(1);
        levels += 1;
        if mw < 8 || mh < 8 { break; }
    }
    let amt = p.bloom / levels as f64;
    let texel = |x: usize, y: usize, c: usize| b[(y.min(bh - 1) * bw + x.min(bw - 1)) * 4 + c] as f64;
    // the tone curve and the grade as they are set (with nothing set: the ACES curve, as ever)
    let grade = gf_core::stage::Grade::of(p);
    // the lens's own colour fringing and fall-off of light, when it has any: the picture with its colours
    // drawn apart, and how much light each pixel keeps
    let lens = crate::camera::develop(hdr, w, h, p);
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h as usize {
        for x in 0..w as usize {
            let (u, v) = ((x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64);
            // the bloom level read with interpolation between its texels, edges held
            let (fx, fy) = ((u * bw as f64 - 0.5).max(0.0), (v * bh as f64 - 0.5).max(0.0));
            let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
            let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
            let vig = 1.0 - p.vignette * ((u - 0.5).powi(2) + (v - 0.5).powi(2));
            let mut exposed = [0.0f64; 3];
            for c in 0..3 {
                let bl = (texel(x0, y0, c) * (1.0 - tx) + texel(x0 + 1, y0, c) * tx) * (1.0 - ty) + (texel(x0, y0 + 1, c) * (1.0 - tx) + texel(x0 + 1, y0 + 1, c) * tx) * ty;
                let at = y * w as usize + x;
                let (light, kept) = lens.as_ref().map_or((hdr[at * 4 + c], 1.0), |l| (l.0[at * 3 + c], l.1[at] as f64));
                exposed[c] = (light as f64 + bl * amt) * vig * kept * p.exposure;
            }
            for lin in grade.apply(exposed) {
                let enc = if lin < 0.0031308 { 12.92 * lin } else { 1.055 * lin.powf(1.0 / 2.4) - 0.055 };
                out.push((enc.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16);
            }
        }
    }
    out
}

fn halton(mut i: u32, b: u32) -> f64 {
    let (mut f, mut r) = (1.0, 0.0);
    while i > 0 {
        f /= b as f64;
        r += f * (i % b) as f64;
        i /= b;
    }
    r
}

impl Renderer {
    pub fn new(gpu: &Gpu) -> Self {
        let frag = wgpu::ShaderStages::FRAGMENT;
        let d2 = wgpu::TextureViewDimension::D2;
        // (5 to 8 are read only by tracers built for an environment picture or for clay; 9 to 11 only by
        // tracers built for a material other than plain glass; 12 and 13 only by the tracer of several
        // bodies with materials of their own; 24 by a milky body's, and 25 to 27 by those of the second to
        // fourth bodies)
        let l_trace = gpu.bind_layout("trace", &[
            uniform_entry(0, frag, false), texture_entry(1, d2, true), sampler_entry(2), texture_entry(3, wgpu::TextureViewDimension::D3, true), sampler_entry(4),
            uniform_entry(5, frag, false), texture_entry(6, wgpu::TextureViewDimension::Cube, true), sampler_entry(7), texture_entry(8, wgpu::TextureViewDimension::Cube, true),
            uniform_entry(9, frag, false), texture_entry(10, d2, true), sampler_entry(11),
            texture_entry(24, wgpu::TextureViewDimension::D3, true), texture_entry(25, wgpu::TextureViewDimension::D3, true),
            texture_entry(26, wgpu::TextureViewDimension::D3, true), texture_entry(27, wgpu::TextureViewDimension::D3, true),
            uniform_entry(crate::stage::BINDING, frag, false),
            // (44: the lamps, read only by tracers built with lamps)
            uniform_entry(crate::lamps::BINDING, frag, false),
            texture_entry(12, wgpu::TextureViewDimension::D3, true), uniform_entry(13, frag, false),
            // (32: what lies inside the body, read only by tracers built for it; 33: what lies inside each
            // of several bodies, read only by the tracer of several bodies built for it)
            uniform_entry(32, frag, false), uniform_entry(33, frag, false),
        ]);
        let no_bodies = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("no bodies"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let no_soft = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("no softened lights"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let no_picture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("no picture"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let l_accum = gpu.bind_layout("accumulate", &[uniform_entry(0, frag, false), texture_entry(1, d2, true), texture_entry(2, d2, true), sampler_entry(3)]);
        let l_bloom = gpu.bind_layout("bloom", &[uniform_entry(0, frag, false), texture_entry(1, d2, true), sampler_entry(2)]);
        let l_final = gpu.bind_layout("final", &[uniform_entry(4, frag, false), texture_entry(5, d2, true), texture_entry(6, d2, true), sampler_entry(7)]);
        let accum = gpu.shader("accumulate", ACCUM_WGSL, false);
        let post = gpu.shader("post", POST_WGSL, true);
        let one = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add };
        let pipe = |label, module, fs, layout: &wgpu::BindGroupLayout, format, blend| gpu.pipeline(&PipeDesc { label, module, vs: "vs", fs, layouts: &[layout], format, blend, strip: false, buffers: &[] });
        Self {
            gpu: gpu.clone(),
            lin: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge),
            env_smp: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::Repeat),
            p_accum: pipe("accumulate", &accum, "fs", &l_accum, ACC, None),
            p_down: pipe("bloom down", &post, "fs_down", &l_bloom, HDR, None),
            p_up: pipe("bloom up", &post, "fs_up", &l_bloom, HDR, Some(wgpu::BlendState { color: one, alpha: one })),
            p_final: pipe("final", &post, "fs_final", &l_final, LDR, None),
            p_final_deep: pipe("final (deep)", &post, "fs_final", &l_final, HDR, None),
            u_final_deep: gpu.uniform_buffer("final (deep)", std::mem::size_of::<FinalU>() as u64),
            deep: false,
            l_cast: crate::cast::receiver_layout(gpu), g_cast: None,
            l_trace, l_accum, l_bloom, l_final,
            trace_pipes: HashMap::new(), trace_key: String::new(),
            u_trace: gpu.uniform_buffer("trace", std::mem::size_of::<TraceU>() as u64),
            u_light: gpu.uniform_buffer("trace: picture and clay", std::mem::size_of::<LightU>() as u64),
            last_light: None,
            u_mat: gpu.uniform_buffer("trace: material", (MU_FLOATS * 4) as u64),
            u_stage: gpu.uniform_buffer("trace: studio", std::mem::size_of::<crate::stage::StageU>() as u64),
            u_lamps: gpu.uniform_buffer("trace: lamps", std::mem::size_of::<crate::lamps::LampsU>() as u64),
            stage: Zeroable::zeroed(),
            last_mat: None,
            bodies: None,
            no_bodies: no_bodies.create_view(&wgpu::TextureViewDescriptor::default()),
            u_bodies: gpu.uniform_buffer("trace: bodies", (material::MB_FLOATS * 4) as u64),
            last_bodies: Vec::new(),
            u_inner: gpu.uniform_buffer("trace: inside the body", (gf_core::inner::IU_FLOATS * 4) as u64),
            last_inner: Vec::new(),
            u_inner_bodies: gpu.uniform_buffer("trace: inside each body", (material::MAX_BODIES * gf_core::inner::IU_FLOATS * 4) as u64),
            last_inner_bodies: Vec::new(),
            rough_share: -1.0,
            glow: [None, None, None, None],
            no_glow: crate::scatter::blank(gpu),
            soft: None,
            no_soft: no_soft.create_view(&wgpu::TextureViewDescriptor::default()),
            soft_smp: gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("softened lights"),
                address_mode_u: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            }),
            picture: None,
            picture_smp: gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("picture"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            }),
            no_picture: no_picture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::Cube), ..Default::default() }),
            building: None, scene: None, march: None, variant: Variant::default(),
            u_accum: gpu.uniform_buffer("accumulate", std::mem::size_of::<AccumU>() as u64),
            u_final: gpu.uniform_buffer("final", std::mem::size_of::<FinalU>() as u64),
            g_trace: None, t: None, acc_idx: 0, accum_n: 0, frame_no: 0, jitter_n: 0, was_taa: false, prev: None, last_final: None, profile: false,
            dof: Dof::new(gpu), focused: false, cam: Default::default(),
            frost: Frost::new(gpu),
            dn: Denoise::new(gpu), shown: 0,
        }
    }

    /// The lights' map as a texture: 16-bit floats, wrapping round the horizon.
    pub fn env_texture(&self, rgba: &[f32], w: u32, h: u32) -> wgpu::TextureView {
        let tex = self.env_target(w, h);
        self.env_paint(&tex, rgba, w, h);
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// A texture for the lights' map, with nothing on it yet.
    pub fn env_target(&self, w: u32, h: u32) -> wgpu::Texture {
        self.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("lights"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    /// Paint the lights' map: RGBA floats, row 0 (the top of the sky) first.
    pub fn env_paint(&self, tex: &wgpu::Texture, rgba: &[f32], w: u32, h: u32) {
        let halves: Vec<half::f16> = rgba.iter().map(|&v| half::f16::from_f32(v)).collect();
        self.gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&halves),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
    }

    /// Say which lights and which volume the tracer reads, and how it marches (these differ by family).
    pub fn set_scene(&mut self, env: &wgpu::TextureView, vol: &Volume, step_scale: f64, steps_out: u32, steps_in: u32, layers: u32) {
        self.march = Some(March { step_scale, steps_out, steps_in, layers });
        self.scene = Some((env.clone(), vol.view.clone()));
        self.build_tracer();
        self.bind_tracer();
    }

    /// Point the tracer at another volume of the same shape (the one with the material's bubbles cut
    /// into it, or back to the shape's own); everything else about the scene stays.
    pub fn set_volume(&mut self, vol: &Volume) {
        let Some((env, _)) = self.scene.take() else { return };
        self.scene = Some((env, vol.view.clone()));
        self.bind_tracer();
    }

    /// The bodies' volume and how many bodies it holds, or None when the scene is one body. Called before
    /// `set_scene` whenever a scene of layers is built; a single shape never calls it.
    pub fn set_bodies(&mut self, bodies: Option<(&Volume, u32)>) {
        self.bodies = bodies.map(|(v, n)| (v.view.clone(), n));
        self.bind_tracer();
    }

    /// How many bodies the volume last handed over holds (None: the scene is one body).
    pub fn bodies_count(&self) -> Option<u32> {
        self.bodies.as_ref().map(|b| b.1)
    }

    /// The numbers of the bodies' materials (`gf_core::material::bodies_uniforms`). True if they changed.
    pub fn set_bodies_material(&mut self, numbers: &[f32]) -> bool {
        if self.last_bodies == numbers { return false; }
        self.gpu.queue.write_buffer(&self.u_bodies, 0, bytemuck::cast_slice(numbers));
        self.last_bodies = numbers.to_vec();
        true
    }

    /// The light inside a milky body, as a volume the tracer reads (None: there is none).
    pub fn set_glow(&mut self, volume: Option<&wgpu::TextureView>) {
        self.set_glow_body(0, volume);
    }
    /// The light inside milky body `k` of a scene drawn body by body (None: it has none).
    pub fn set_glow_body(&mut self, k: usize, volume: Option<&wgpu::TextureView>) {
        let Some(slot) = self.glow.get_mut(k) else { return };
        if slot.as_ref() == volume { return; }
        *slot = volume.cloned();
        self.bind_tracer();
    }

    /// The numbers of what lies inside each of several bodies (`gf_core::material::bodies_inner`). True if
    /// they changed.
    pub fn set_inner_bodies(&mut self, numbers: &[f32]) -> bool {
        if self.last_inner_bodies == numbers { return false; }
        self.gpu.queue.write_buffer(&self.u_inner_bodies, 0, bytemuck::cast_slice(numbers));
        self.last_inner_bodies = numbers.to_vec();
        true
    }

    /// The numbers of what lies inside the body (`gf_core::inner::uniforms`). True if they changed.
    pub fn set_inner(&mut self, numbers: &[f32]) -> bool {
        if self.last_inner == numbers { return false; }
        self.gpu.queue.write_buffer(&self.u_inner, 0, bytemuck::cast_slice(numbers));
        self.last_inner = numbers.to_vec();
        true
    }

    /// The material's numbers (`gf_core::material::uniforms`). True if they changed.
    pub fn set_material(&mut self, numbers: &[f32; MU_FLOATS]) -> bool {
        if self.last_mat.as_ref() == Some(numbers) { return false; }
        self.gpu.queue.write_buffer(&self.u_mat, 0, bytemuck::cast_slice(numbers));
        self.last_mat = Some(*numbers);
        self.rough_share = numbers[material::MU_SHARE];
        true
    }

    /// Say whether the view is moving. A rough surface is drawn two ways at once (see `ROUGH_SAMPLED` in
    /// material.rs): while the view moves less of the roughness is sampled and more of it softened, which
    /// costs less and leaves less grain in the few frames a moving picture is made of; once it is still
    /// the share that is sampled goes back up. (This does not count as a change to the picture: the
    /// frames that follow simply average towards the truer one.)
    pub fn set_moving(&mut self, moving: bool) {
        let Some(n) = &self.last_mat else { return };
        let share = n[if moving { material::MU_SHARE_MOVING } else { material::MU_SHARE }];
        if share != self.rough_share {
            self.gpu.queue.write_buffer(&self.u_mat, (material::MU_SHARE * 4) as u64, bytemuck::bytes_of(&share));
            self.rough_share = share;
        }
    }

    /// Give the tracer the lights' map with its softened copies (see `gf_core::material::env_levels`):
    /// level 0 is the map itself (RGBA floats, row 0 the top of the sky), each further level a copy half
    /// the size and blurred twice as far. Only rough and milky materials read it, so it is only made
    /// when one of them is in use.
    pub fn soften_lights(&mut self, rgba: &[f32], w: u32, h: u32) {
        let levels = material::env_levels(rgba, w as usize, h as usize);
        let fresh = self.soft.as_ref().is_none_or(|(t, _)| t.width() != w || t.height() != h);
        if fresh {
            let tex = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("softened lights"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1 + levels.len() as u32,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HDR,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.soft = Some((tex, view));
        }
        let (tex, _) = self.soft.as_ref().expect("just made");
        let write = |level: u32, data: &[f32], lw: u32, lh: u32| {
            let halves: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
            self.gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: tex, mip_level: level, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                bytemuck::cast_slice(&halves),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(lw * 8), rows_per_image: Some(lh) },
                wgpu::Extent3d { width: lw, height: lh, depth_or_array_layers: 1 },
            );
        };
        write(0, rgba, w, h);
        for (i, (lw, lh, data)) in levels.iter().enumerate() { write(i as u32 + 1, data, *lw as u32, *lh as u32); }
        if fresh { self.bind_tracer(); }
    }

    /// Say what the tracer draws: glass or clay, lit by what, with what behind the shape.
    pub fn set_variant(&mut self, v: Variant) {
        if v == self.variant && !self.lens_pending() { return; }
        self.variant = v;
        self.build_tracer();
    }
    /// The lens that is wanted (`cam.wanted`) is not the one the tracer in use traces through: the next
    /// `set_variant` puts that right.
    pub fn lens_pending(&self) -> bool {
        self.cam.active != self.cam.wanted
    }
    /// The tracer's text for the lens that is wanted.
    fn tracer_text(&self, m: &March, v: Variant) -> String {
        if self.cam.wanted { tracer_source_lens(m, v, self.profile) } else { tracer_source(m, v, self.profile) }
    }
    /// Tracers kept, and frames in the running average (for the window's diagnostic log).
    pub fn kept(&self) -> String {
        format!("pipes={} accum={}", self.trace_pipes.len(), self.accum_n)
    }

    pub fn variant(&self) -> Variant {
        self.variant
    }

    /// Say which environment picture the tracer reads, or that there is none: the cube with its softer
    /// levels (shown behind the shape), and the cube made ready for the glass.
    pub fn set_picture(&mut self, cubes: Option<(&wgpu::TextureView, &wgpu::TextureView)>) {
        self.picture = cubes.map(|(behind, glass)| (behind.clone(), glass.clone()));
        self.bind_tracer();
    }

    /// What the tracer is told about the environment picture and the clay.
    pub fn set_light(&mut self, lu: &LightU) {
        if self.last_light == Some(*lu) { return; }
        self.gpu.queue.write_buffer(&self.u_light, 0, bytemuck::bytes_of(lu));
        self.last_light = Some(*lu);
    }

    /// The layout of the group through which a tracer reads what the bodies cast.
    pub fn cast_layout(&self) -> &wgpu::BindGroupLayout {
        &self.l_cast
    }
    /// The group a tracer with casting in it reads (see cast.rs). It must be there before the tracer
    /// is told to draw what is cast.
    pub fn set_cast(&mut self, group: Option<&wgpu::BindGroup>) {
        self.g_cast = group.cloned();
    }

    /// What the tracer and the antialiasing are told of the studio: the room, the backdrop, the floor.
    /// Tell the tracer where the lamps stand and what they are (read only by a tracer built with lamps).
    pub fn set_lamps(&mut self, lu: &crate::lamps::LampsU) {
        self.gpu.queue.write_buffer(&self.u_lamps, 0, bytemuck::bytes_of(lu));
    }
    pub fn set_stage(&mut self, su: &crate::stage::StageU) {
        self.gpu.queue.write_buffer(&self.u_stage, 0, bytemuck::bytes_of(su));
        self.stage = *su;
    }

    /// Start the sequences that place rays within a pixel and wavelengths within the spectrum from
    /// another point, so that two pictures of the same thing are made from different rays.
    pub fn reseed(&mut self, n: u32) {
        self.frame_no = n;
        self.jitter_n = n % 4096;
    }

    fn tracer_key(&self, m: &March, v: Variant) -> String {
        format!("{:.2}|{}|{}|{}|{}|{:?}|{}", m.step_scale, m.steps_out, m.steps_in, m.layers, self.profile, v, self.cam.wanted)
    }

    /// The pipeline for the way of marching and drawing now set: made the first time, kept after.
    fn build_tracer(&mut self) {
        let Some(m) = self.march else { return };
        let key = self.tracer_key(&m, self.variant);
        if !self.trace_pipes.contains_key(&key) {
            let pipe = tracer_pipeline(&self.gpu, &self.l_trace, self.variant.cast.then_some(&self.l_cast), &self.tracer_text(&m, self.variant));
            self.trace_pipes.insert(key.clone(), pipe);
        }
        self.trace_key = key;
        self.cam.active = self.cam.wanted;
        // A tracer through a lens has passes of its own after it (folding the frame in, the final
        // picture). They are made here, with the tracer, rather than by the first frame that needs them:
        // making them takes a good part of a second, which the first frame would otherwise take too.
        if self.cam.active {
            self.cam.accum(&self.gpu, ACCUM_WGSL, &self.l_accum, ACC);
            self.cam.last(&self.gpu, POST_WGSL, &self.l_final, [LDR, HDR], false);
        }
    }

    /// Is the tracer for this way of drawing built? If it is not, it is built on another thread (it takes
    /// about half a second, too long to hold a frame up for) and the answer is no until it is there;
    /// `set_variant` then takes no time. One is built at a time.
    pub fn prepare(&mut self, v: Variant) -> bool {
        // (before there is a shape there is nothing to build for: `set_scene` builds what is set by then)
        let Some(m) = self.march else { return true };
        if let Some((key, rx)) = &self.building {
            match rx.try_recv() {
                Ok(pipe) => {
                    self.trace_pipes.insert(key.clone(), pipe);
                    self.building = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.building = None,
            }
        }
        let key = self.tracer_key(&m, v);
        if self.trace_pipes.contains_key(&key) { return true; }
        if self.building.is_none() {
            let (tx, rx) = std::sync::mpsc::channel();
            let (gpu, layout, src) = (self.gpu.clone(), self.l_trace.clone(), self.tracer_text(&m, v));
            let cast = v.cast.then(|| self.l_cast.clone());
            let spawned = std::thread::Builder::new().name("tracer".into()).spawn(move || {
                let _ = tx.send(tracer_pipeline(&gpu, &layout, cast.as_ref(), &src));
            });
            match spawned {
                Ok(_) => self.building = Some((key, rx)),
                // no thread to be had: build it here after all
                Err(_) => return true,
            }
        }
        false
    }

    fn bind_tracer(&mut self) {
        let Some((env, vol)) = &self.scene else { return };
        self.g_trace = Some(self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trace"),
            layout: &self.l_trace,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.u_trace.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(env) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.env_smp) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(vol) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.lin) },
                wgpu::BindGroupEntry { binding: 5, resource: self.u_light.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(self.picture.as_ref().map_or(&self.no_picture, |p| &p.0)) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&self.picture_smp) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(self.picture.as_ref().map_or(&self.no_picture, |p| &p.1)) },
                wgpu::BindGroupEntry { binding: 9, resource: self.u_mat.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(self.soft.as_ref().map_or(&self.no_soft, |s| &s.1)) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::Sampler(&self.soft_smp) },
                wgpu::BindGroupEntry { binding: 24, resource: wgpu::BindingResource::TextureView(self.glow[0].as_ref().unwrap_or(&self.no_glow)) },
                wgpu::BindGroupEntry { binding: 25, resource: wgpu::BindingResource::TextureView(self.glow[1].as_ref().unwrap_or(&self.no_glow)) },
                wgpu::BindGroupEntry { binding: 26, resource: wgpu::BindingResource::TextureView(self.glow[2].as_ref().unwrap_or(&self.no_glow)) },
                wgpu::BindGroupEntry { binding: 27, resource: wgpu::BindingResource::TextureView(self.glow[3].as_ref().unwrap_or(&self.no_glow)) },
                wgpu::BindGroupEntry { binding: crate::stage::BINDING, resource: self.u_stage.as_entire_binding() },
                wgpu::BindGroupEntry { binding: crate::lamps::BINDING, resource: self.u_lamps.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(self.bodies.as_ref().map_or(&self.no_bodies, |b| &b.0)) },
                wgpu::BindGroupEntry { binding: 13, resource: self.u_bodies.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 32, resource: self.u_inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 33, resource: self.u_inner_bodies.as_entire_binding() },
            ],
        }));
    }

    /// (width, height) of the picture on screen, and of the traced picture behind it.
    pub fn size(&self) -> ((u32, u32), (u32, u32)) {
        self.t.as_ref().map_or(((0, 0), (0, 0)), |t| ((t.w, t.h), (t.tw, t.th)))
    }

    /// Make the pictures this size. True if they had to be made anew (the picture so far is gone).
    pub fn resize(&mut self, w: u32, h: u32, scale: f64) -> bool {
        let (w, h) = (w.max(2), h.max(2));
        let tw = ((w as f64 * scale + 0.5).floor() as u32).max(2);
        let th = ((h as f64 * scale + 0.5).floor() as u32).max(2);
        if let Some(t) = &self.t {
            if t.w == w && t.h == h && t.tw == tw && t.th == th && t.deep.is_some() == self.deep { return false; }
        }
        let gpu = &self.gpu;
        let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let trace_view = view(&gpu.target("trace", tw, th, HDR, &[]));
        let accum = [gpu.target("picture a", tw, th, ACC, &[]), gpu.target("picture b", tw, th, ACC, &[])];
        let accum_view = [view(&accum[0]), view(&accum[1])];
        // the focus stage reads either picture and writes one of its own, which the bloom and the screen
        // read in its place (the third of each of their groups below)
        self.dof.resize(tw, th, [&accum_view[0], &accum_view[1]]);
        let focused = self.dof.view().expect("the focus stage has just been sized").clone();
        let source = |i: usize| if i < 2 { &accum_view[i] } else { &focused };
        fn tex_entry<'a>(binding: u32, v: &'a wgpu::TextureView) -> wgpu::BindGroupEntry<'a> {
            wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(v) }
        }
        let accum_groups = |frame: &wgpu::TextureView| {
            [0, 1].map(|i| {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("accumulate"),
                    layout: &self.l_accum,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.u_accum.as_entire_binding() },
                        tex_entry(1, &accum_view[i]),
                        tex_entry(2, frame),
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.lin) },
                    ],
                })
            })
        };
        let g_accum = accum_groups(&trace_view);
        // the stage for rough glass reads the traced frame and writes one of its own, which is folded
        // into the picture in its place
        self.frost.resize(tw, th, &trace_view);
        let g_accum_rough = accum_groups(self.frost.view().expect("the stage for rough glass has just been sized"));
        self.dn.resize(tw, th, [&accum_view[0], &accum_view[1]], &self.u_accum);
        // bloom: a chain of pictures each half the size of the one before
        let bloom_group = |src: &wgpu::TextureView, sw: u32, sh: u32, dw: u32, dh: u32| {
            let u = gpu.uniform_buffer("bloom", std::mem::size_of::<BloomU>() as u64);
            gpu.queue.write_buffer(&u, 0, bytemuck::bytes_of(&BloomU { texel: [1.0 / sw as f32, 1.0 / sh as f32], dst_res: [dw as f32, dh as f32], gain: 1.0, pad: [0.0; 3] }));
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom"),
                layout: &self.l_bloom,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() }, tex_entry(1, src), wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.lin) }],
            })
        };
        let mut sizes = Vec::new();
        let (mut mw, mut mh) = (tw, th);
        for _ in 0..7 {
            mw = (mw >> 1).max(1);
            mh = (mh >> 1).max(1);
            sizes.push((mw, mh));
            if mw < 8 || mh < 8 { break; }
        }
        let texs: Vec<wgpu::Texture> = sizes.iter().map(|&(a, b)| gpu.target("bloom", a, b, HDR, &[])).collect();
        let views: Vec<wgpu::TextureView> = texs.iter().map(&view).collect();
        let g_down0 = [0, 1, 2].map(|i| bloom_group(source(i), tw, th, sizes[0].0, sizes[0].1));
        let mut mips = Vec::new();
        for (i, &(a, b)) in sizes.iter().enumerate() {
            // down: read the level above into this one (level 0 reads the picture itself: see g_down0)
            let down = if i == 0 { g_down0[0].clone() } else { bloom_group(&views[i - 1], sizes[i - 1].0, sizes[i - 1].1, a, b) };
            // up: read this level while drawing onto the next larger one
            let up = (i > 0).then(|| bloom_group(&views[i], a, b, sizes[i - 1].0, sizes[i - 1].1));
            mips.push(Mip { tex: texs[i].clone(), view: views[i].clone(), w: a, h: b, down, up });
        }
        let final_tex = gpu.target("screen", w, h, LDR, &[wgpu::TextureFormat::Rgba8UnormSrgb]);
        let final_view = view(&final_tex);
        let g_final = [0, 1, 2].map(|i| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("final"),
                layout: &self.l_final,
                entries: &[
                    wgpu::BindGroupEntry { binding: 4, resource: self.u_final.as_entire_binding() },
                    tex_entry(5, source(i)),
                    tex_entry(6, &views[0]),
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&self.lin) },
                ],
            })
        });
        let deep = self.deep.then(|| {
            let tex = gpu.target("screen (deep)", w, h, HDR, &[]);
            let v = view(&tex);
            let groups = [0, 1, 2].map(|i| {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("final (deep)"),
                    layout: &self.l_final,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 4, resource: self.u_final_deep.as_entire_binding() },
                        tex_entry(5, source(i)),
                        tex_entry(6, &views[0]),
                        wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&self.lin) },
                    ],
                })
            });
            (tex, v, groups)
        });
        self.t = Some(Targets { w, h, tw, th, trace_view, accum, accum_view, g_accum, g_accum_rough, g_down0, mips, g_final, final_tex, final_view, deep });
        self.acc_idx = 0;
        self.shown = 0;
        self.accum_n = 0;
        self.prev = None;
        true
    }

    /// The picture on screen, as a texture whose bytes are already display-ready (sRGB-encoded).
    pub fn screen(&self) -> Option<&wgpu::Texture> {
        self.t.as_ref().map(|t| &t.final_tex)
    }

    /// Trace one frame and fold it into the picture so far.
    ///   moving: something moved since the last frame, so the picture so far is out of date
    ///   cut: something changed outright, so the picture so far is useless
    ///   skip_post: leave out the bloom and the screen (more frames are about to be averaged in)
    pub fn frame(&mut self, tu: &TraceU, look: &Look, moving: bool, cut: bool, skip_post: bool) {
        let Some(t) = self.t.as_ref() else { return };
        let Some(g_trace) = self.g_trace.as_ref() else { return };
        if self.variant.cast && self.g_cast.is_none() { return; }
        // How to treat the picture so far.
        let taa = moving && !cut && look.smooth > 0.0;
        let weight: f64;
        if cut || (moving && !taa) {
            self.accum_n = 0;
            weight = 1.0;
        } else if taa {
            weight = 1.0 - look.smooth as f64;
            self.accum_n = ((1.0 / weight + 0.5).floor() as u32).saturating_sub(1);      // then carry on averaging from here
        } else {
            // The first still frame after motion. The picture so far was kept by the antialiasing that
            // follows motion: it lags what moved, is darkened where its neighbourhood limit cut bright
            // samples, and (when the shape was rebuilt at its still detail just now) shows the coarser
            // shape it had while moving. Counted as the frames it stands for (ten at the usual smoothing), that error would fade
            // only as ten over the number of frames since; counted as one frame, it is gone four times
            // sooner, and one frame of new light weighs as much as it straight away. What is denoised
            // keeps a running mean of its own (denoise.rs): that is started afresh instead.
            if self.was_taa {
                self.accum_n = if self.variant.denoise { 0 } else { self.accum_n.min(1) };
            }
            weight = 1.0 / (self.accum_n as f64 + 1.0);
        }
        self.was_taa = taa;
        self.jitter_n = (self.jitter_n % 4096) + 1;
        // always jitter unless antialiasing is off and the view is moving
        let jit = weight < 1.0 || look.smooth > 0.0;
        let mut u = *tu;
        u.res = [t.tw as f32, t.th as f32];
        u.frame = self.frame_no as i32;
        u.jitter = if jit { [(halton(self.jitter_n, 2) - 0.5) as f32, (halton(self.jitter_n, 3) - 0.5) as f32] } else { [0.0, 0.0] };
        self.gpu.queue.write_buffer(&self.u_trace, 0, bytemuck::bytes_of(&u));

        // (a studio's backdrop stands in the room, not in the lights' frame: where one is shown, it is the
        // room that says where a piece of the background was at the previous frame)
        let seen = self.variant.stage;
        let now = PrevView { obj: u.obj, env: if seen.backdrop { self.stage.room } else { u.env }, room: self.stage.room, floor: self.stage.mid[3], cam: [u.dist, u.tan_fov], shift: u.shift, lens: u.cam };
        let was = self.prev.unwrap_or(now);
        let au = AccumU {
            res: u.res, cam: now.cam, cam_p: was.cam, shift: now.shift, shift_p: was.shift, w: weight as f32, moving: if taa { 1.0 } else { 0.0 },
            obj: now.obj, objp: was.obj, debug: look.debug, spread: look.spread, sky: if self.variant.backdrop || seen.backdrop { 1.0 } else { 0.0 }, pad: 0.0,
            env: now.env, envp: was.env,
            room: now.room, roomp: was.room, floor: [if seen.floor { 1.0 } else { 0.0 }, now.floor, was.floor, 0.0],
            lens: now.lens, lens_p: was.lens,
        };
        self.gpu.queue.write_buffer(&self.u_accum, 0, bytemuck::bytes_of(&au));

        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        // (the tracer of a rough clear body writes a blur radius beside its frame, and the frame is
        // blurred by it before it is folded in)
        let rough = rough_split(&self.variant, self.profile);
        let second = self.g_cast.as_ref().filter(|_| self.variant.cast);
        // (a denoised tracer writes the light it found inside the body to one more target)
        let dn = if self.variant.denoise && !self.profile { self.dn.frame_view(&self.lin).cloned() } else { None };
        match self.frost.aux().filter(|_| rough) {
            Some(aux) => {
                let views: Vec<&wgpu::TextureView> = [&t.trace_view, aux[0], aux[1]].into_iter().chain(dn.as_ref()).collect();
                frost::draw(&mut enc, &views, &self.trace_pipes[&self.trace_key], g_trace, second);
                self.frost.encode(&mut enc, u.tan_fov);
            }
            None if dn.is_some() => {
                frost::draw(&mut enc, &[&t.trace_view, dn.as_ref().expect("just asked")], &self.trace_pipes[&self.trace_key], g_trace, second);
            }
            None => {
                let mut rp = pass(&mut enc, &t.trace_view, None, true);
                rp.set_pipeline(&self.trace_pipes[&self.trace_key]);
                rp.set_bind_group(0, g_trace, &[]);
                if let Some(g) = second { rp.set_bind_group(1, g, &[]); }
                rp.draw(0..3, 0..1);
            }
        }
        {
            let mut rp = pass(&mut enc, &t.accum_view[1 - self.acc_idx], None, true);
            // (through any lens but the plain one, the surface is followed by the camera's own arithmetic)
            rp.set_pipeline(if self.cam.active { self.cam.accum(&self.gpu, ACCUM_WGSL, &self.l_accum, ACC) } else { &self.p_accum });
            rp.set_bind_group(0, if rough { &t.g_accum_rough[self.acc_idx] } else { &t.g_accum[self.acc_idx] }, &[]);
            rp.draw(0..3, 0..1);
        }
        // the light found inside the body: carried on into its own running mean, along the surface as the
        // picture is (through the camera's arithmetic when the lens is not the plain one)
        if dn.is_some() {
            let lens = || crate::camera::accum(&denoise::temporal_source(ACCUM_WGSL));
            let lens: Option<&dyn Fn() -> String> = if self.cam.active { Some(&lens) } else { None };
            self.dn.time(&mut enc, weight >= 1.0, taa, [u.dist, u.tan_fov, u.shift[0], u.shift[1]], u.jitter, denoise::Tune::of(&self.variant.mat), lens);
        } else {
            self.dn.live = false;
        }
        self.prev = Some(now);
        self.acc_idx = 1 - self.acc_idx;
        self.accum_n += 1;
        self.frame_no += 1;
        if !skip_post { self.encode_post(&mut enc, &u, look, !moving && !cut); }
        self.gpu.queue.submit([enc.finish()]);
    }

    /// Bloom the picture so far and put it on the screen, without tracing anything.
    pub fn present(&mut self, tu: &TraceU, look: &Look) {
        if self.t.is_none() { return; }
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("present") });
        self.encode_post(&mut enc, tu, look, true);
        self.gpu.queue.submit([enc.finish()]);
    }

    /// `settled`: the picture did not move for this frame (the focus stage then takes more care).
    fn encode_post(&mut self, enc: &mut wgpu::CommandEncoder, tu: &TraceU, look: &Look, settled: bool) {
        let Some(t) = self.t.as_ref() else { return };
        let fu = FinalU {
            res: [t.w as f32, t.h as f32],
            exposure: look.exposure,
            bloom_amt: look.bloom / t.mips.len() as f32,
            vignette: look.vignette,
            grain: look.grain,
            raw: look.raw,
            frame: if look.still_grain { 0 } else { self.frame_no as i32 },
            // the last of these is the dither that hides the steps of an 8-bit picture
            planes: [look.planes[0], look.planes[1], look.planes[2], 1.0 / 255.0],
            view: [tu.dist, tu.tan_fov, tu.shift[0], tu.shift[1]],
            grade: look.grade,
            lens: look.lens.cam, fx: [look.lens.fringe, look.lens.falloff, look.lens.eye, 0.0],
        };
        // the final pass is the plain one unless the lens is not, or has a character of its own
        let lens = self.cam.active || look.lens.character();
        self.gpu.queue.write_buffer(&self.u_final, 0, bytemuck::bytes_of(&fu));
        if t.deep.is_some() {
            let mut deep = fu;
            deep.planes[3] = 0.0;
            self.gpu.queue.write_buffer(&self.u_final_deep, 0, bytemuck::bytes_of(&deep));
        }
        self.last_final = Some(fu);
        // what is traced by chance, denoised and added to the picture so far, in the other of the pair
        // (which the next frame overwrites anyway): from here on that stands for the picture so far
        let denoised = if self.variant.denoise && !self.profile { self.dn.filter(enc, self.acc_idx, &t.accum_view[1 - self.acc_idx]) } else { None };
        self.shown = denoised.unwrap_or(self.acc_idx);
        // focus, when it is done to the finished picture: from here on its picture stands for the picture
        // so far (what lies behind the glass is blurred with it when it is a picture and not black)
        self.focused = look.focus.on();
        let src = if self.focused {
            let mut focus = Focus { backdrop: self.variant.backdrop || self.variant.stage.any(), ..look.focus };
            // (until the tracer that traces through the lens is built, the picture is the plain lens's)
            if !self.cam.active { focus.lens = focus.lens.filter(|c| c.plain()); }
            self.dof.encode(enc, self.shown, &focus, [tu.dist, tu.tan_fov, tu.shift[0], tu.shift[1]], settled);
            2
        } else {
            self.shown
        };
        // bloom: walk down the chain, then back up adding each level
        for (i, m) in t.mips.iter().enumerate() {
            let mut rp = pass(enc, &m.view, None, true);
            rp.set_pipeline(&self.p_down);
            rp.set_bind_group(0, if i == 0 { &t.g_down0[src] } else { &m.down }, &[]);
            rp.draw(0..3, 0..1);
        }
        for i in (1..t.mips.len()).rev() {
            let mut rp = pass(enc, &t.mips[i - 1].view, None, false);
            rp.set_pipeline(&self.p_up);
            rp.set_bind_group(0, t.mips[i].up.as_ref().unwrap(), &[]);
            rp.draw(0..3, 0..1);
        }
        // tone map to the screen
        {
            let mut rp = pass(enc, &t.final_view, None, true);
            rp.set_pipeline(if lens { self.cam.last(&self.gpu, POST_WGSL, &self.l_final, [LDR, HDR], false) } else { &self.p_final });
            rp.set_bind_group(0, &t.g_final[src], &[]);
            rp.draw(0..3, 0..1);
        }
        if let Some((_, view, groups)) = &t.deep {
            let mut rp = pass(enc, view, None, true);
            rp.set_pipeline(if lens { self.cam.last(&self.gpu, POST_WGSL, &self.l_final, [LDR, HDR], true) } else { &self.p_final_deep });
            rp.set_bind_group(0, &groups[src], &[]);
            rp.draw(0..3, 0..1);
        }
    }

    /// What is on screen with 16 bits a channel and no 8-bit dither: RGBA, top row first. Needs `deep`.
    pub fn read_screen16(&self) -> R<(u32, u32, Vec<u16>)> {
        let t = self.t.as_ref().ok_or("nothing has been drawn yet")?;
        let (tex, _, _) = t.deep.as_ref().ok_or("the 16-bit screen was not asked for")?;
        let bytes = self.gpu.read(tex, 0, 0, 0, t.w, t.h, 8)?;
        let halves: &[half::f16] = bytemuck::cast_slice(&bytes);
        Ok((t.w, t.h, halves.iter().map(|v| (v.to_f32().clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).collect()))
    }

    /// The first (largest) level of the bloom as it stands: RGBA 16-bit floats as f32, its width and height.
    pub fn read_bloom(&self) -> R<(u32, u32, Vec<f32>)> {
        let t = self.t.as_ref().ok_or("nothing has been drawn yet")?;
        let m = &t.mips[0];
        let bytes = self.gpu.read(&m.tex, 0, 0, 0, m.w, m.h, 8)?;
        let halves: &[half::f16] = bytemuck::cast_slice(&bytes);
        Ok((m.w, m.h, halves.iter().map(|v| v.to_f32()).collect()))
    }

    /// What is on screen: RGBA bytes, top row first.
    pub fn read_screen(&self) -> R<(u32, u32, Vec<u8>)> {
        let t = self.t.as_ref().ok_or("nothing has been drawn yet")?;
        Ok((t.w, t.h, self.gpu.read(&t.final_tex, 0, 0, 0, t.w, t.h, 4)?))
    }

    /// Part of the picture so far, before bloom and tone mapping (and after focus, when that is done to
    /// the finished picture): RGBA 32-bit floats (alpha = distance to the glass), `w` x `h` from (x, y)
    /// counted from the top left, top row first.
    pub fn read_hdr(&self, x: u32, y: u32, w: u32, h: u32) -> R<Vec<f32>> {
        let t = self.t.as_ref().ok_or("nothing has been drawn yet")?;
        let picture = self.dof.picture().filter(|_| self.focused).unwrap_or(&t.accum[self.shown]);
        let bytes = self.gpu.read(picture, x, y, 0, w, h, 16)?;
        Ok(bytemuck::pod_collect_to_vec(&bytes))
    }
}
