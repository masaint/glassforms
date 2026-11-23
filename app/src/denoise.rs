//! Denoising: making the materials that are traced by chance (light followed as it wanders inside a milky
//! body, a rough surface drawn by sampling alone) clean while the view moves and quick to settle when it
//! stops, without blurring what makes glass look like glass.
//!
//! What is noisy, and what is not. Everything a pixel shows that comes back from inside the body was
//! found by a random path; the first surface's own reflection (the highlights, the sharp mirrored detail)
//! was not. So a tracer built for this keeps the two apart where it adds them up: the clean part goes the
//! way every frame always has (the stage for rough glass, then folded into the picture so far, with the
//! antialiasing that follows the surface while moving), and the noisy part gets a stage of its own:
//!
//! 1. Over time: a running mean of the noisy part and of its brightness squared, carried along the
//!    surface by the very arithmetic the picture so far is (the text of accum.wgsl, and the camera's
//!    version of it for other lenses), with the number of frames it stands for. While the view is still
//!    it averages for as long as it stays still; while it moves it stands for at most `keep` frames.
//! 2. Over space: an a-trous wavelet filter (Dammertz et al. 2010; as in SVGF, Schied et al. 2017), five
//!    passes of a 5 x 5 kernel spread 1, 2, 4, 8, 16 pixels apart. A neighbour is taken only if it lies
//!    on the same surface (the first surface's distance and which way it faces, worked out here from the
//!    distances the tracer already writes: nothing more is carried through the tracer) and if its
//!    brightness differs from this pixel's by no more than the uncertainty of the mean: the variance of
//!    the samples divided by how many there are. So the filter is wide on the first frame of a view and
//!    shuts itself off as each pixel's own mean becomes certain: a still view ends up exactly as the
//!    plain average would, sooner.
//! 3. The picture: the clean part as folded in, plus the filtered noisy part, written into the half of
//!    the picture-so-far pair that is not current (it is overwritten by the next frame anyway), so that
//!    focus, bloom, the screen and every reading of the picture take it with no change of their own.
//!
//! For light that has wandered inside a body the first surface's distance and facing are true guides: the
//! light leaves the body where it is seen. Nothing here is built or run unless a tracer that is drawn by
//! chance is in use and `denoise` is on.

use bytemuck::{Pod, Zeroable};

use crate::gpu::*;

const FILTER_WGSL: &str = concat!(include_str!("shaders/pbr/hash.wgsl"), include_str!("shaders/denoise.wgsl"));
/// What the tracer writes beside the picture: the noisy part, with the distance in alpha. 32-bit: the
/// facing of the surface is worked out from the distances of neighbouring pixels, and a 16-bit distance
/// (steps of 0.004 at the usual distances, about a pixel's width) would make it grain.
const FRAME: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
/// The noisy part's running mean (with the distance in alpha, as the picture so far keeps it). 32-bit:
/// a mean over a thousand frames moves by less than a 16-bit float can say.
const HIST: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
/// The mean of the brightness squared, and the number of frames the mean stands for.
const MOM: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;
/// Which way the surface faces (x, y, z, as the camera sees it) and its distance.
const GUIDE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The noisy part being filtered, with the uncertainty of its brightness in alpha.
const WORK: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The picture so far (render.rs: ACC).
const ACC: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
/// Passes of the a-trous filter: the last reaches 16 pixels, the whole kernel 62 either way.
const PASSES: usize = 5;
/// Marks a tracer's text as one that writes its noisy part beside the picture.
pub const MARK: &str = "var<private> dnIn: vec3f;";

/// Is the tracer for this way of drawing one that is drawn by chance, and is denoising on for it?
/// `setting` is `denoise`: 0 off; 1 (the default) on where the tracer is drawn by chance: light that
/// wanders inside a milky body (`sssWalk`) and a clear rough surface drawn by sampling alone
/// (`roughRef`); 2 also clear glass (the cut stones: the colours of their fire are drawn by chance, a few
/// wavelengths a ray). Clear glass is not on by default: while the view moves it keeps the light the
/// antialiasing would throw away (a turning brilliant at 0.99 of its settled level instead of 0.78) but
/// shows more coloured grain and some smearing of what is seen through the facets, a trade the eye has
/// to judge. A mixed scene splits light transmitted by a surface with a random-walk interior.
/// Other primary surfaces retain their own unfiltered shading.
pub fn wanted(mat: &gf_core::material::Variant, clay: bool, setting: f64) -> bool {
    let chance = mat.walk || (mat.bodies == 0 && !mat.metal && !mat.hollow && mat.rough_ref);
    let clear = mat.plain() && setting >= 1.5;
    setting >= 0.5 && !clay && (chance || clear) && !off()
}

/// While moving, the most frames the mean of what is seen through clear glass stands for.
const KEEP_CLEAR: f32 = 4.0;

/// GLASSFORMS_DENOISE=off: never denoise (for comparing); =raw: the noisy part averaged over time but
/// not filtered; =noisy: only the noisy part, filtered; =certain: how uncertain each pixel's mean is;
/// =guide: which way the surface faces, as the stage works it out; =split: the tracer writes the noisy
/// part apart but the stage is not run (for timing the tracer alone).
fn env() -> &'static str {
    static V: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    V.get_or_init(|| std::env::var("GLASSFORMS_DENOISE").unwrap_or_default())
}
fn off() -> bool {
    env() == "off"
}

/// The tracer's text, made to add up the light found inside the body on its own (`dnIn`) and to write
/// it beside the picture as a further colour target; the picture itself then holds the rest, the clean
/// part: the surface's own reflection, what is seen where rays miss, the grazing light at the outline.
/// Nothing more is carried through the marching than the shading carries already, except, in the
/// material tracer, the total as it stood when the surface's reflection had been added (`dnTop`).
///
/// `whole`: the surface's own reflection is drawn by chance as well (a rough surface sampled alone: its
/// facets are random), so everything the surface sends back counts as noisy, not only what comes from inside.
pub fn with_split(src: &str, whole: bool) -> String {
    // the material tracer (trace_material.wgsl, the clear and milky body's shadeM): where the surface's
    // own reflection has been added for every wavelength, and where the paths inside have all been added
    let top = "  acc *= f32(MODES);\n";
    let end = "  return acc / (wsum * f32(MODES));\n";
    // plain glass (trace.wgsl's shade): what came through the glass is `one`, or `acc / wsum`
    let plain = "  if (alike) { return envR * F0 + one; }\n  return envR * (racc / wsum) + acc / wsum;\n";
    let mixed = "// The random-walk body's transmitted radiance is available here.";
    let s = if src.contains(mixed) {
        src.replace(mixed, "if (hasB(k, 32u)) { dnIn += res - envR * top; }")
    } else if src.matches(top).count() == 1 && src.matches(end).count() == 1 {
        let keep = if whole { "  let dnTop = vec3f(0.0);\n" } else { "  let dnTop = acc;                          // (what the paths inside add is told apart from this)\n" };
        src.replace(top, &format!("{top}{keep}")).replace(end, &format!("  dnIn += (acc - dnTop) / (wsum * f32(MODES));\n{end}"))
    } else {
        assert!(src.matches(plain).count() == 1, "the tracer's shading is not as it was");
        src.replace(plain, "  if (alike) { dnIn += one; return envR * F0 + one; }\n  dnIn += acc / wsum;\n  return envR * (racc / wsum) + acc / wsum;\n")
    };
    let head = "fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {";
    let tail = "return vec4f(col / f32(S), depth);";
    assert!(s.matches(head).count() == 1 && s.matches(tail).count() == 1, "the tracer's last function is not as it was");
    let s = s.replace(head, "fn fs(@builtin(position) pos: vec4f) -> DnOut {")
        .replace(tail, concat!(
            // (a value that is not a number, which a rare path can come to, would spread through the
            // filter and the bloom over the whole picture: such a part is dropped)
            "let dnS = select(vec3f(0.0), dnIn, abs(dnIn) < vec3f(1e30));\n  let dnC = col - dnS;\n",
            "  return DnOut(vec4f(select(vec3f(0.0), dnC, abs(dnC) < vec3f(1e30)) / f32(S), depth), vec4f(dnS / f32(S), depth));"
        ));
    format!("{s}\nstruct DnOut {{ @location(0) col: vec4f, @location(1) dn: vec4f }}\n// the light found inside the body, added up over the pixel's samples (written beside the picture)\n{MARK}\n")
}

/// The pipeline of a tracer that writes its noisy part as well.
pub fn tracer_pipeline(gpu: &Gpu, layout: &wgpu::BindGroupLayout, cast: Option<&wgpu::BindGroupLayout>, module: &wgpu::ShaderModule) -> wgpu::RenderPipeline {
    let layouts: Vec<&wgpu::BindGroupLayout> = std::iter::once(layout).chain(cast).collect();
    crate::adapt::tracer_pipeline(gpu, "trace (denoised)", module, &layouts, &[wgpu::TextureFormat::Rgba16Float, FRAME])
}

fn pipeline(gpu: &Gpu, label: &str, module: &wgpu::ShaderModule, fs: &str, layouts: &[&wgpu::BindGroupLayout], formats: &[wgpu::TextureFormat]) -> wgpu::RenderPipeline {
    let groups: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(|l| Some(*l)).collect();
    let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &groups, immediate_size: 0 });
    let targets: Vec<Option<wgpu::ColorTargetState>> = formats.iter().map(|&format| Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })).collect();
    gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        vertex: wgpu::VertexState { module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState { module, entry_point: Some(fs), compilation_options: Default::default(), targets: &targets }),
        multiview_mask: None,
        cache: None,
    })
}

/// The pass that carries the noisy part along over time, made from the text of the pass that folds a
/// frame into the picture so far (`accum`: accum.wgsl as it stands). Everything up to where that pass has
/// found the old picture of this piece of surface (`same`, `hist`, `at`) is kept as it is, so the noisy
/// part follows the surface exactly as the picture does, through any lens the camera's version handles;
/// what it then does with it is replaced.
pub fn temporal_source(accum: &str) -> String {
    let head = "fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {";
    let still = "  if (a.moving < 0.5) {\n    return vec4f(mix(textureLoad(prevTex, c, 0).rgb, cur.rgb, a.w), cur.a);\n  }\n";
    let cut = "  if (!same) { hist = textureLoad(prevTex, c, 0).rgb; }";
    assert!(accum.matches(head).count() == 1 && accum.matches(still).count() == 1 && accum.matches(cut).count() == 1, "accum.wgsl is not as it was");
    let kept = &accum[..accum.find(cut).expect("checked")];
    let kept = kept.replace(head, "fn fs(@builtin(position) pos: vec4f) -> DnT {").replace(still, "");
    format!("{kept}{TEMPORAL_TAIL}{TEMPORAL_DECL}{}", include_str!("shaders/pbr/hash.wgsl")).replace("\r\n", "\n")
}

/// What the pass over time does once the old picture has been found (see `temporal_source`).
const TEMPORAL_TAIL: &str = r#"  // (from here on: the noisy part's own running mean, over time)
  let dnStill = a.moving < 0.5;
  var dnMom = vec2f(0.0);                    // the old mean of brightness squared, and how many frames it stands for
  if (dnStill) {
    hist = textureLoad(prevTex, c, 0).rgb;
    dnMom = textureLoad(momTex, c, 0).xy;
  } else if (same) {
    dnMom = textureLoad(momTex, clamp(vec2i(floor(att)), vec2i(0), top), 0).xy;
  }
  // nothing to carry on from: the picture starts afresh, there is no glass here, or this piece of surface
  // was not in the old picture
  // (while still, a pixel on an outline that misses the glass this frame is averaged in as it is: the
  // light of a pixel partly on the glass is the mean over where its rays land)
  if (dn.reset > 0.5 || (!dnStill && (!same || cur.a <= 0.0))) { dnMom = vec2f(0.0); }
  var dnN = dnMom.y + 1.0;
  if (!dnStill) {
    // moving: the old mean stands for no more than `keep` frames, and (for light that changes with the
    // view) may stray from the new frame's neighbourhood by no more than so many of its deviations
    dnN = min(dnN, dn.keep);
    if (dn.clampk > 0.0) { hist = clamp(hist, m1 - dev * dn.clampk, m1 + dev * dn.clampk); }
  }
  let dnW = 1.0 / dnN;
  let dnL = dnLum(cur.rgb);
  let dnMean = mix(hist, cur.rgb, dnW);
  let dnM2 = mix(dnMom.x, dnL * dnL, dnW);
  // the guide: which way the surface faces, from the distances of the pixels on either side (on each
  // axis the side whose distance is nearer this pixel's, so that an outline does not bend it)
  var dnG = vec4f(0.0, 0.0, 1.0, cur.a);
  if (cur.a > 0.0) {
    var rd0 = vec3f(0.0, 0.0, -1.0);
    let p0 = dnPosRay(fc, cur.a, &rd0);
    let tl = textureLoad(curTex, clamp(c - vec2i(1, 0), vec2i(0), top), 0).a;
    let tr = textureLoad(curTex, clamp(c + vec2i(1, 0), vec2i(0), top), 0).a;
    let tu = textureLoad(curTex, clamp(c - vec2i(0, 1), vec2i(0), top), 0).a;
    let td = textureLoad(curTex, clamp(c + vec2i(0, 1), vec2i(0), top), 0).a;
    var ex = vec3f(1.0, 0.0, 0.0);
    var ey = vec3f(0.0, 1.0, 0.0);
    let useR = tr > 0.0 && (tl <= 0.0 || abs(tr - cur.a) <= abs(tl - cur.a));
    if (useR) { ex = dnPos(fc + vec2f(1.0, 0.0), tr) - p0; } else if (tl > 0.0) { ex = p0 - dnPos(fc - vec2f(1.0, 0.0), tl); }
    let useU = tu > 0.0 && (td <= 0.0 || abs(tu - cur.a) <= abs(td - cur.a));
    if (useU) { ey = dnPos(fc + vec2f(0.0, 1.0), tu) - p0; } else if (td > 0.0) { ey = p0 - dnPos(fc - vec2f(0.0, 1.0), td); }
    var nn = cross(ex, ey);
    if (dot(nn, nn) < 1e-20) { nn = -rd0; }
    nn = normalize(nn);
    if (dot(nn, rd0) > 0.0) { nn = -nn; }
    dnG = vec4f(nn, cur.a);
  }
  return DnT(vec4f(dnMean, cur.a), vec4f(dnM2, dnN, 0.0, 0.0), dnG);
}
"#;

const TEMPORAL_DECL: &str = r#"
struct DnT { @location(0) mean: vec4f, @location(1) mom: vec4f, @location(2) guide: vec4f }
struct DnU {
  res: vec2f,
  tan_fov: f32,
  reset: f32,             // 1 = the picture starts afresh: nothing is carried over
  shift: vec2f,
  keep: f32,              // while moving, the most frames the old mean stands for
  clampk: f32,            // while moving, how many deviations of the new neighbourhood it may stray (0 = any)
  sigma_l: f32,
  sigma_z: f32,
  sigma_n: f32,
  show: f32,
  jitter: vec2f,          // where in its pixel this frame's ray went (with the pixel's own shift: dnPos)
  young: f32,
  chroma: f32,
}
@group(0) @binding(8) var momTex: texture_2d<f32>;
@group(0) @binding(9) var<uniform> dn: DnU;
fn dnLum(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
// where the surface is, as the camera sees it, for a pixel (its middle, counted from the bottom) and its
// distance: along the ray the tracer sent through that pixel this frame, which went through the point
// trace.wgsl's fs chose (its first sample: a shift of the pixel's own and the frame's jitter), so that
// the distances of neighbouring pixels give the surface's facing without the grain of the jitter
fn dnIgn(p: vec2f) -> f32 { return pbr_ign(p); }   // pbr/hash.wgsl
fn dnPosRay(fc: vec2f, t: f32, rdOut: ptr<function, vec3f>) -> vec3f {
  let o = fract(vec2f(dnIgn(fc + 19.19), dnIgn(fc.yx + 47.3)) + dn.jitter) - 0.5;
  let uv = (2.0 * (fc + o) - dn.res) / dn.res.y + dn.shift;
  let rd = normalize(vec3f(uv * dn.tan_fov, -1.0));
  *rdOut = rd;
  return rd * t;
}
fn dnPos(fc: vec2f, t: f32) -> vec3f {
  var rd: vec3f;
  return dnPosRay(fc, t, &rd);
}
"#;

/// The numbers of the stage (`DnU` in denoise.wgsl and in the pass over time).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Debug)]
struct DnU {
    res: [f32; 2],
    tan_fov: f32,
    reset: f32,
    shift: [f32; 2],
    keep: f32,
    clampk: f32,
    sigma_l: f32,
    sigma_z: f32,
    sigma_n: f32,
    show: f32,
    jitter: [f32; 2],
    young: f32,
    chroma: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PassU {
    step: f32,
    pad: [f32; 3],
}

/// How the stage is tuned. While moving, light that has wandered inside a body hardly changes with the
/// view, so its mean may stand for many frames; light seen through a rough surface does change, so it is
/// held to the new frame's neighbourhood.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tune {
    pub keep: f32,
    pub clampk: f32,
    /// a neighbour is taken while its brightness is within this many uncertainties of the pixel's
    pub sigma_l: f32,
    /// how far from the pixel's tangent plane a neighbour may lie, in pixels' widths per pixel away
    pub sigma_z: f32,
    /// how alike two facings must be (the power of the cosine between them)
    pub sigma_n: f32,
    /// a mean of few frames is taken to be this many times less certain than its samples say, over the
    /// number of frames (so the first frames of a view are filtered more widely)
    pub young: f32,
    /// passes of the filter over space (0: none, the running mean alone)
    pub passes: u32,
    /// filter only the colour, each pixel keeping its own brightness (clear glass: what is seen through
    /// it is detail the first surface knows nothing of, but the colour of its grain is the wavelengths a
    /// ray happened to carry)
    pub chroma: bool,
    /// filter over space only while the view moves (a still view is then exactly the plain average)
    pub moving_only: bool,
}

impl Tune {
    pub fn of(mat: &gf_core::material::Variant) -> Self {
        let t = if mat.walk {
            Tune { keep: 32.0, clampk: 0.0, sigma_l: 8.0, sigma_z: 1.0, sigma_n: 16.0, young: 4.0, passes: 5, chroma: false, moving_only: false }
        } else if mat.rough_ref {
            Tune { keep: 12.0, clampk: 0.0, sigma_l: 8.0, sigma_z: 1.0, sigma_n: 16.0, young: 4.0, passes: 5, chroma: false, moving_only: false }
        } else {
            // Clear glass: what comes through it changes with the view, so its mean stands for few
            // frames while moving; but it is not held to the new frame's neighbourhood as the picture
            // is (which, for the few bright specks of a gem's fire, throws a fifth of its light away)
            Tune { keep: KEEP_CLEAR, clampk: 0.0, sigma_l: 8.0, sigma_z: 1.0, sigma_n: 16.0, young: 4.0, passes: 2, chroma: true, moving_only: true }
        };
        t.overridden()
    }
    /// GLASSFORMS_DENOISE_TUNE="keep=32,clampk=0,sl=4,sz=1,sn=64,young=0,passes=5,chroma=0": other numbers, for trying.
    fn overridden(mut self) -> Self {
        let Ok(text) = std::env::var("GLASSFORMS_DENOISE_TUNE") else { return self };
        for (k, v) in text.split(',').filter_map(|kv| kv.split_once('=')).filter_map(|(k, v)| Some((k.trim(), v.trim().parse::<f32>().ok()?))) {
            match k {
                "keep" => self.keep = v,
                "clampk" => self.clampk = v,
                "sl" => self.sigma_l = v,
                "sz" => self.sigma_z = v,
                "sn" => self.sigma_n = v,
                "young" => self.young = v,
                "passes" => self.passes = (v.max(0.0) as u32).min(PASSES as u32),
                "chroma" => self.chroma = v > 0.5,
                "moving" => self.moving_only = v > 0.5,
                _ => {}
            }
        }
        self
    }
}

struct Targets {
    frame_view: wgpu::TextureView,
    hist: [wgpu::TextureView; 2],
    mom: [wgpu::TextureView; 2],
    guide: wgpu::TextureView,
    work: [wgpu::TextureView; 2],
    /// the pass over time reading hist[i], mom[i]
    g_time: [wgpu::BindGroup; 2],
    /// the first filter pass, reading hist[i], mom[i] (and writing work[0])
    g_var: [wgpu::BindGroup; 2],
    g_var_lens: [wgpu::BindGroup; 2],
    /// the filter passes: pass k reads work[k % 2] and writes the other
    g_pass: Vec<wgpu::BindGroup>,
    g_pass_lens: Vec<wgpu::BindGroup>,
    /// putting the picture together: reading the picture so far i and the filtered noisy part (and the
    /// unfiltered mean hist[j], for looking at), writing into the other picture so far
    g_out: [[[wgpu::BindGroup; 2]; 2]; 2],
}

pub struct Denoise {
    gpu: Gpu,
    u: wgpu::Buffer,
    pass_u: Vec<wgpu::Buffer>,
    l_time: wgpu::BindGroupLayout,
    l_var: wgpu::BindGroupLayout,
    l_var_lens: wgpu::BindGroupLayout,
    l_pass: wgpu::BindGroupLayout,
    l_pass_lens: wgpu::BindGroupLayout,
    l_out: wgpu::BindGroupLayout,
    /// the pass over time for the plain lens, and for the camera's other lenses (made when first wanted)
    p_time: Option<wgpu::RenderPipeline>,
    p_time_lens: Option<wgpu::RenderPipeline>,
    p_var: wgpu::RenderPipeline,
    p_pass: wgpu::RenderPipeline,
    /// spatial filter passes built for lenses other than the plain one (made when first wanted)
    p_filter_lens: std::cell::OnceCell<(wgpu::RenderPipeline, wgpu::RenderPipeline)>,
    p_out: wgpu::RenderPipeline,
    size: (u32, u32),
    /// the pictures it is given: the picture so far (both), and the uniform block of the pass that folds
    /// frames into it, which the pass over time reads too
    given: Option<([wgpu::TextureView; 2], wgpu::Buffer)>,
    t: Option<Targets>,
    /// which running mean is current
    hi: usize,
    /// the running mean has been carried on since the last start afresh
    pub live: bool,
    lens_active: bool,
    show: f32,
    /// passes of the filter over space the last frame's tuning asks for
    passes: usize,
}

impl Denoise {
    pub fn new(gpu: &Gpu) -> Self {
        let frag = wgpu::ShaderStages::FRAGMENT;
        let d2 = wgpu::TextureViewDimension::D2;
        let l_time = gpu.bind_layout("denoise: over time", &[uniform_entry(0, frag, false), texture_entry(1, d2, true), texture_entry(2, d2, true), sampler_entry(3), texture_entry(8, d2, false), uniform_entry(9, frag, false)]);
        let l_var = gpu.bind_layout("denoise: uncertainty", &[uniform_entry(0, frag, false), texture_entry(2, d2, false), texture_entry(3, d2, false), texture_entry(4, d2, false)]);
        let l_var_lens = gpu.bind_layout("denoise: uncertainty (lens)", &[uniform_entry(0, frag, false), texture_entry(2, d2, false), texture_entry(3, d2, false), texture_entry(4, d2, false), uniform_entry(8, frag, false)]);
        let l_pass = gpu.bind_layout("denoise: filter", &[uniform_entry(0, frag, false), uniform_entry(1, frag, false), texture_entry(2, d2, false), texture_entry(4, d2, false)]);
        let l_pass_lens = gpu.bind_layout("denoise: filter (lens)", &[uniform_entry(0, frag, false), uniform_entry(1, frag, false), texture_entry(2, d2, false), texture_entry(4, d2, false), uniform_entry(8, frag, false)]);
        let l_out = gpu.bind_layout("denoise: put together", &[uniform_entry(0, frag, false), texture_entry(5, d2, false), texture_entry(6, d2, false), texture_entry(7, d2, false), texture_entry(4, d2, false), texture_entry(3, d2, false)]);
        let module = gpu.shader("denoise", FILTER_WGSL, true);
        let mk = |label, fs, l: &wgpu::BindGroupLayout, f| pipeline(gpu, label, &module, fs, &[l], &[f]);
        let pass_u = (0..PASSES)
            .map(|k| {
                let b = gpu.uniform_buffer("denoise: pass", std::mem::size_of::<PassU>() as u64);
                gpu.queue.write_buffer(&b, 0, bytemuck::bytes_of(&PassU { step: (1u32 << k) as f32, pad: [0.0; 3] }));
                b
            })
            .collect();
        Self {
            gpu: gpu.clone(),
            u: gpu.uniform_buffer("denoise", std::mem::size_of::<DnU>() as u64),
            pass_u,
            p_var: mk("denoise: uncertainty", "fs_var", &l_var, WORK),
            p_pass: mk("denoise: filter", "fs_pass", &l_pass, WORK),
            p_filter_lens: std::cell::OnceCell::new(),
            p_out: mk("denoise: put together", "fs_out", &l_out, ACC),
            l_time, l_var, l_var_lens, l_pass, l_pass_lens, l_out,
            p_time: None,
            p_time_lens: None,
            size: (0, 0),
            given: None,
            t: None,
            hi: 0,
            live: false,
            lens_active: false,
            passes: PASSES,
            show: match env() { "raw" => 1.0, "noisy" => 2.0, "certain" => 3.0, "guide" => 4.0, _ => 0.0 },
        }
    }

    /// Get ready for traced frames `w` x `h`, folded into `picture` (the two pictures so far), whose pass
    /// reads `accum_u`. The stage's own pictures (about 80 bytes a pixel) are made when first used.
    pub fn resize(&mut self, w: u32, h: u32, picture: [&wgpu::TextureView; 2], accum_u: &wgpu::Buffer) {
        self.size = (w, h);
        self.given = Some(([picture[0].clone(), picture[1].clone()], accum_u.clone()));
        self.t = None;
        self.live = false;
    }

    fn build(&self, lin: &wgpu::Sampler) -> Option<Targets> {
        let (pic, accum_u) = self.given.as_ref()?;
        let (w, h) = self.size;
        let gpu = &self.gpu;
        let view = |label: &str, f: wgpu::TextureFormat| gpu.target(label, w, h, f, &[]).create_view(&wgpu::TextureViewDescriptor::default());
        let frame_view = view("denoise: noisy part", FRAME);
        let hist = [view("denoise: mean a", HIST), view("denoise: mean b", HIST)];
        let mom = [view("denoise: moments a", MOM), view("denoise: moments b", MOM)];
        let guide = view("denoise: guide", GUIDE);
        let work = [view("denoise: work a", WORK), view("denoise: work b", WORK)];
        fn tex(binding: u32, v: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(v) }
        }
        fn uni(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }
        }
        let group = |label: &str, layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry]| gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries });
        let g_time = [0, 1].map(|i| group("denoise: over time", &self.l_time, &[uni(0, accum_u), tex(1, &hist[i]), tex(2, &frame_view), wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(lin) }, tex(8, &mom[i]), uni(9, &self.u)]));
        let g_var = [0, 1].map(|i| group("denoise: uncertainty", &self.l_var, &[uni(0, &self.u), tex(2, &hist[i]), tex(3, &mom[i]), tex(4, &guide)]));
        let g_var_lens = [0, 1].map(|i| group("denoise: uncertainty (lens)", &self.l_var_lens, &[uni(0, &self.u), tex(2, &hist[i]), tex(3, &mom[i]), tex(4, &guide), uni(8, accum_u)]));
        let g_pass = (0..PASSES).map(|k| group("denoise: filter", &self.l_pass, &[uni(0, &self.u), uni(1, &self.pass_u[k]), tex(2, &work[k % 2]), tex(4, &guide)])).collect();
        let g_pass_lens = (0..PASSES).map(|k| group("denoise: filter (lens)", &self.l_pass_lens, &[uni(0, &self.u), uni(1, &self.pass_u[k]), tex(2, &work[k % 2]), tex(4, &guide), uni(8, accum_u)])).collect();
        let g_out = [0, 1].map(|i| [0, 1].map(|j| [0, 1].map(|k| group("denoise: put together", &self.l_out, &[uni(0, &self.u), tex(5, &pic[i]), tex(6, &work[k]), tex(7, &hist[j]), tex(4, &guide), tex(3, &mom[j])]))));
        Some(Targets { frame_view, hist, mom, guide, work, g_time, g_var, g_var_lens, g_pass, g_pass_lens, g_out })
    }

    /// Where the tracer writes the noisy part (made now if it is not yet).
    pub fn frame_view(&mut self, lin: &wgpu::Sampler) -> Option<&wgpu::TextureView> {
        if self.t.is_none() { self.t = self.build(lin); }
        self.t.as_ref().map(|t| &t.frame_view)
    }

    /// Carry the noisy part of the frame just traced into its running mean. `reset`: the picture so far
    /// starts afresh with this frame. `cam`: camera distance, tan(half the field of view), the view's
    /// shift. `moving`: the view moves (the antialiasing that follows the surface is on). `jitter`: the
    /// tracer's this frame. `lens_src`: the pass's text for the camera's other lenses, when one is in use.
    pub fn time(&mut self, enc: &mut wgpu::CommandEncoder, reset: bool, moving: bool, cam: [f32; 4], jitter: [f32; 2], tune: Tune, lens_src: Option<&dyn Fn() -> String>) {
        let Some(t) = self.t.as_ref() else { return };
        if env() == "split" { return; }
        let (w, h) = self.size;
        let passes = if moving || !tune.moving_only { tune.passes } else { 0 };
        let u = DnU {
            res: [w as f32, h as f32], tan_fov: cam[1], reset: if reset || !self.live { 1.0 } else { 0.0 }, shift: [cam[2], cam[3]],
            keep: tune.keep, clampk: tune.clampk, sigma_l: tune.sigma_l, sigma_z: tune.sigma_z, sigma_n: tune.sigma_n, show: if passes > 0 || self.show > 0.0 { self.show } else { 1.0 },
            jitter, young: tune.young, chroma: if tune.chroma { 1.0 } else { 0.0 },
        };
        self.gpu.queue.write_buffer(&self.u, 0, bytemuck::bytes_of(&u));
        self.lens_active = lens_src.is_some();
        let pipe = match lens_src {
            Some(src) => {
                if self.p_time_lens.is_none() { self.p_time_lens = Some(self.time_pipeline(&src())); }
                self.p_time_lens.as_ref()
            }
            None => {
                if self.p_time.is_none() { self.p_time = Some(self.time_pipeline(&temporal_source(crate::render::ACCUM_SOURCE))); }
                self.p_time.as_ref()
            }
        };
        let Some(pipe) = pipe else { return };
        let o = 1 - self.hi;
        let atts = [Some(att(&t.hist[o])), Some(att(&t.mom[o])), Some(att(&t.guide))];
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor { label: Some("denoise: over time"), color_attachments: &atts, depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None });
            rp.set_pipeline(pipe);
            rp.set_bind_group(0, &t.g_time[self.hi], &[]);
            rp.draw(0..3, 0..1);
        }
        self.hi = o;
        self.live = true;
        self.passes = passes as usize;
    }

    fn time_pipeline(&self, src: &str) -> wgpu::RenderPipeline {
        let module = self.gpu.shader("denoise: over time", src, true);
        pipeline(&self.gpu, "denoise: over time", &module, "fs", &[&self.l_time], &[HIST, MOM, GUIDE])
    }

    /// Filter the running mean and add it to the picture so far `cur` (0 or 1), writing the whole into
    /// the other one, which is what is shown. Returns its number.
    pub fn filter(&self, enc: &mut wgpu::CommandEncoder, cur: usize, out: &wgpu::TextureView) -> Option<usize> {
        let t = self.t.as_ref()?;
        if !self.live { return None; }
        let draw = |enc: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, pipe: &wgpu::RenderPipeline, g: &wgpu::BindGroup| {
            let mut rp = pass(enc, view, None, true);
            rp.set_pipeline(pipe);
            rp.set_bind_group(0, g, &[]);
            rp.draw(0..3, 0..1);
        };
        // (unfiltered: the putting together reads the running mean itself)
        let passes = self.passes;
        if self.lens_active {
            let (p_var, p_pass) = self.p_filter_lens.get_or_init(|| {
                let module = self.gpu.shader("denoise (lens)", &crate::camera::denoise(FILTER_WGSL), true);
                (
                    pipeline(&self.gpu, "denoise: uncertainty (lens)", &module, "fs_var", &[&self.l_var_lens], &[WORK]),
                    pipeline(&self.gpu, "denoise: filter (lens)", &module, "fs_pass", &[&self.l_pass_lens], &[WORK]),
                )
            });
            if passes > 0 { draw(enc, &t.work[0], p_var, &t.g_var_lens[self.hi]); }
            for k in 0..passes {
                draw(enc, &t.work[1 - k % 2], p_pass, &t.g_pass_lens[k]);
            }
        } else {
            if passes > 0 { draw(enc, &t.work[0], &self.p_var, &t.g_var[self.hi]); }
            for k in 0..passes {
                draw(enc, &t.work[1 - k % 2], &self.p_pass, &t.g_pass[k]);
            }
        }
        // (the last pass wrote work[passes % 2])
        draw(enc, out, &self.p_out, &t.g_out[cur][self.hi][passes % 2]);
        Some(1 - cur)
    }
}

fn att(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } }
}

#[cfg(test)]
mod tests {
    fn valid(src: &str) {
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
    }
    /// The filter's shader parses and passes the validator, both for the plain lens and for any lens.
    #[test]
    fn the_filter_is_valid() {
        valid(super::FILTER_WGSL);
        valid(&crate::camera::denoise(super::FILTER_WGSL));
    }
    /// The pass over time, made from the picture's own pass for the plain lens and the camera's, is valid.
    #[test]
    fn the_pass_over_time_is_valid() {
        valid(&super::temporal_source(crate::render::ACCUM_SOURCE));
        valid(&crate::camera::accum(&super::temporal_source(crate::render::ACCUM_SOURCE)));
    }
}
