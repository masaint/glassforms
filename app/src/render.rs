//! Drawing: trace one frame, fold it into the picture so far, bloom it, tone map it.
//! (renderFrame and present of the web version's engine.js.)

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

use gf_core::material::{self, MU_FLOATS};

use crate::bake::Volume;
use crate::split_half::Half;
use crate::denoise::{self, Denoise};
use crate::dof::{Dof, Focus};
use crate::frost::{self, Frost};
use crate::gpu::*;

const TRACE_WGSL: &str = include_str!("shaders/trace.wgsl");
/// The tracer's template as it stands in trace.wgsl (the matcap view weaves its own entry into it: matcap.rs).
pub(crate) const TRACE_SOURCE: &str = TRACE_WGSL;
const TRACE_LIGHT_WGSL: &str = concat!(include_str!("shaders/pbr/sh9.wgsl"), include_str!("shaders/light/occlusion.wgsl"), include_str!("shaders/trace_light.wgsl"));
const DIELECTRIC_FRESNEL_WGSL: &str = concat!(
    include_str!("shaders/pbr/hash.wgsl"),
    include_str!("shaders/pbr/types.wgsl"),
    include_str!("shaders/pbr/dielectric.wgsl"),
    include_str!("shaders/dielectric_fresnel.wgsl"),
);
/// What is put into the tracer for materials other than plain glass (gf_core::material::tracer_parts).
/// (with everything a rough surface does, which is kept in a file of its own)
pub const TRACE_MATERIAL_WGSL: &str = concat!("//#if LAYER_SERIES\n", include_str!("shaders/paint_interface.wgsl"), include_str!("shaders/pbr/diffuse.wgsl"), include_str!("shaders/pbr/layer.wgsl"), include_str!("shaders/light/rig.wgsl"), "//#endif\n", "//#if PAINT_FILM\n", include_str!("shaders/paint_film.wgsl"), "//#endif\n", include_str!("shaders/pbr/widen.wgsl"), "//#if METAL\n", include_str!("shaders/pbr/energy.wgsl"), "//#endif\n", include_str!("shaders/trace_material.wgsl"), "//#if FILM\n", include_str!("shaders/film_pattern.wgsl"), "//#endif\n", include_str!("shaders/film.wgsl"), include_str!("shaders/frost.wgsl"), include_str!("shaders/frost_direct.wgsl"), "\n//#if RANDOM\n", include_str!("shaders/pcg.wgsl"), "//#endif\n", "//#if FROST_BOUNDARY\n", include_str!("shaders/ggx_microsurface.wgsl"), "//#endif\n", include_str!("shaders/frost_reference.wgsl"), include_str!("shaders/inner.wgsl"), "\n//#if WEAR\n", include_str!("shaders/wear_pigment.wgsl"), include_str!("shaders/wear_marks.wgsl"), "\n//#endif\n", "//#if METAL\n", include_str!("shaders/conductor_fresnel.wgsl"), "//#endif\n", "//#if GGX_SAMPLE\n", include_str!("shaders/pbr/ggx.wgsl"), include_str!("shaders/pbr/rough_conductor.wgsl"), "//#endif\n");
pub(crate) const PAINT_BODIES_WGSL: &str = concat!(
    include_str!("shaders/paint_bodies.wgsl"),
);
pub const TRACE_BODIES_WGSL: &str = concat!(
    include_str!("shaders/trace_bodies.wgsl"),
    include_str!("shaders/frost_bodies.wgsl"),
    include_str!("shaders/paint_bodies.wgsl"),
    include_str!("shaders/paint_exchange.wgsl"),
);
const GEM_WGSL: &str = include_str!("shaders/gem.wgsl");
/// The binding of a cut stone's planes (gem.wgsl), read only by tracers built for a stone.
pub const GEM_BINDING: u32 = 34;
/// The lines of trace.wgsl's `shade` that find where a ray inside the glass leaves it, and what replaces
/// them in a tracer built for a cut stone (gem.wgsl: `gemExit`).
const GEM_MARCH: &str = "      let s = marchIn(pos, dir);
      let q = pos + dir * s;
      var m = nrm(q);
";
const GEM_EXIT: &str = "      var m: vec3f;
      let s = gemExit(pos, dir, &m);
      let q = pos + dir * s;
";

/// For a cut stone reshaped by the surface studio (r13, docs/perf_r13.md): the march through the glass
/// starts where the stone's planes, each moved in by the most the studio can carve (`GemU.info.w`), let it
/// start, instead of at the ray's start. The studio moves the surface by at most that, so nothing the march
/// from the start would have met lies before it: the same surface is found with far fewer steps.
const GEM_CARVED_EXIT: &str = "      var m: vec3f;
      let s = gemCarved(pos, dir, &m);
      let q = pos + dir * s;
";
const GEM_CARVED_WGSL: &str = "
// A cut stone the surface studio has reshaped (render.rs GEM_CARVED_EXIT): the stone's planes moved in by
// gem.info.w bound the surface from inside, so the march from pos can start where the ray leaves that
// smaller stone. Where the rim's curved wall (not bounded by the planes) is within reach, it starts at pos.
fn gemCarved(pos: vec3f, dir: vec3f, mOut: ptr<function, vec3f>) -> f32 {
  let po = toObj(pos);
  let dob = toObj(dir);
  let n = i32(gem.info.x);
  let b = gem.info.w;
  var t = 1e9;
  for (var i = 0; i < n; i++) {
    let pl = gem.pl[i];
    let dn = dot(pl.xyz, dob);
    if (dn > 1e-6) { t = min(t, (pl.w - b - dot(pl.xyz, po)) / dn); }
  }
  let reach = 3.05 * 2.0 * HB / u.vres + b;
  // Within the rim's band of heights the rim's wall may cut the ray off first: it stands outside the
  // circle of radius gem.carve.x about the middle, so the circle moved in likewise bounds it too.
  let y0 = po.y;
  let y1 = po.y + dob.y * clamp(t, 0.0, 2.5 * BOUND);
  if (max(y0, y1) > gem.info.y - reach && min(y0, y1) < gem.info.z + reach) {
    let r = gem.carve.x - reach;
    let a = dot(dob.xz, dob.xz);
    let h = dot(po.xz, dob.xz);
    let c = dot(po.xz, po.xz) - r * r;
    if (c >= 0.0 || r <= 0.0) { t = 0.0; }                 // starting outside that circle: march from the start
    else if (a > 1e-9) { t = min(t, (-h + sqrt(max(h * h - a * c, 0.0))) / a); }
  }
  var s = clamp(t, 0.0, 2.5 * BOUND);
  for (var i = 0; i < STEPS_IN; i++) {
    let d = -mapW(pos + dir * s);
    if (d < 0.0008) { break; }
    s += d * STEP_SCALE;
    if (s > 2.5 * BOUND) { break; }
  }
  s -= mapNW(pos + dir * s);
  s -= mapNW(pos + dir * s);
  *mOut = nrm(pos + dir * s);
  return s;
}
";

/// The tracer's text for a reshaped cut stone (`GEM_CARVED_EXIT`); text in which `shade`'s march is not
/// found is given back as it was.
pub fn gem_carved_weave(src: &str) -> String {
    if crate::weave::found(src, "render.gem_march", GEM_MARCH) != 1 { return src.to_string(); }
    let decl = GEM_WGSL.replace("{{GEM_MAX}}", &gf_core::presets::gems::MAX_PLANES.to_string()).replace("{{GEM_BINDING}}", &GEM_BINDING.to_string());
    // (the last plane's place holds the rim's circle: gem.carve)
    let max = gf_core::presets::gems::MAX_PLANES;
    let decl = decl.replace(&format!("pl: array<vec4f, {max}> }}"), &format!("pl: array<vec4f, {}>, carve: vec4f }}", max - 1));
    format!("{}{decl}{GEM_CARVED_WGSL}", src.replace(GEM_MARCH, GEM_CARVED_EXIT))
}

/// Can a tracer of this kind find its way out of a cut stone from the stone's planes? Plain glass only,
/// with nothing woven into how it reads the volume or its normals (lamps note where each normal was
/// taken; a grown stage moves the volume's box).
pub fn gem_fits(v: &Variant) -> bool {
    v.mat == material::Variant::default() && !v.clay && !v.near && !v.grown
}

/// The tracer's text for a cut stone: `shade`'s march through the glass replaced by `gemExit`. Text in
/// which those lines are not found is given back as it was.
pub fn gem_weave(src: &str) -> String {
    if crate::weave::found(src, "render.gem_march", GEM_MARCH) != 1 { return src.to_string(); }
    let decl = GEM_WGSL.replace("{{GEM_MAX}}", &gf_core::presets::gems::MAX_PLANES.to_string()).replace("{{GEM_BINDING}}", &GEM_BINDING.to_string());
    format!("{}{decl}", src.replace(GEM_MARCH, GEM_EXIT))
}
const ACCUM_WGSL: &str = include_str!("shaders/accum.wgsl");
/// The pass that folds a frame into the picture so far (the denoiser's pass over time is made from it).
pub const ACCUM_SOURCE: &str = ACCUM_WGSL;
/// The antialiasing that follows motion, for a frame traced smaller than the picture (keeping the frame
/// rate: `Renderer::dyn_scale`). The frame lies in the corner of the traced target, `a.pad1` of it per
/// side: each pixel of the picture reads the frame at its own place within that corner (colour between
/// the frame's pixels, distance from the nearest one, since distances must not be blended across an
/// outline), and the neighbourhood that limits the old picture is the frame's 3x3 there. Everything else
/// is accum.wgsl's. None if the lines it changes are not found (accum.wgsl changed: see the test).
pub fn accum_dyn(src: &str) -> Option<String> {
    const CUR: &str = "  let cur = textureLoad(curTex, c, 0);";
    const TOP: &str = "  let top = vec2i(textureDimensions(curTex)) - 1;";
    const NEAR: &str = "textureLoad(curTex, clamp(c + vec2i(x, y), vec2i(0), top), 0)";
    const BLEND: &str = "  var out = vec4f(mix(hist, cur.rgb, a.w), cur.a);";
    const DYN_CUR: &str = "  let top = vec2i(floor(a.res * a.pad1 + 0.5)) - 1;
  let sampleOffset = vec2f(a.obj0.w, -a.obj1.w);
  let atl = clamp((vec2f(c) + 0.5) * vec2f(top + 1) / a.res - sampleOffset, vec2f(0.5), vec2f(top) + 0.5);
  let cl = min(vec2i(atl), top);
  let cur = vec4f(sharpImage(atl, curTex, top), textureLoad(curTex, cl, 0).a);";
    // Give an actually traced native sample more weight than an interpolation between rays.
    // AccumU carries the ray offset in the otherwise unused fourth lanes of the rotation rows.
    const DYN_BLEND: &str = "  var weight = a.w;
  if (a.obj2.w > 0.0 && same && !edge && cur.a > 0.0 && !onFloor) {
    let offset = abs(atl - (vec2f(cl) + 0.5));
    let alignment = max(vec2f(1.0) - 2.0 * offset, vec2f(0.0));
    let confidence = alignment.x * alignment.y;
    weight = min(1.0, a.w * (0.65 + 1.4 * confidence));
  }
  var out = vec4f(mix(hist, cur.rgb, weight), cur.a);";
    let anchors = [("render.accum_cur", CUR), ("render.accum_top", TOP), ("render.accum_near", NEAR), ("render.accum_blend", BLEND)];
    if !crate::weave::all_once("render.accum_dyn", &anchors.map(|(name, s)| crate::weave::found(src, name, s))) { return None; }
    Some(src.replace(TOP, "").replace(CUR, DYN_CUR).replace(BLEND, DYN_BLEND).replace(NEAR, "textureLoad(curTex, clamp(cl + vec2i(x, y), vec2i(0), top), 0)"))
}
/// The size a frame traced at share `s` of a `tw` x `th` target takes (as the shader works it out).
pub fn dyn_region(tw: u32, th: u32, s: f64) -> (u32, u32) {
    let s = s as f32;
    let side = |n: u32| ((n as f32 * s).round() as u32).max(2).min(n);
    (side(tw), side(th))
}
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
    /// Compiled scene-medium identity and signed interface priority, followed by reserved zero lanes.
    pub medium: [i32; 4],
    /// where the lens stands across and up from the line through the object's centre (a free camera); unused, unused
    pub eye: [f32; 4],
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
    pub paint_room: [[f32; 4]; 9],
    pub paint_lights: [crate::surface::SLight; crate::surface::MAX_LIGHTS],
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
    pub textured: bool,
    pub world_faces: bool,
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
    /// the camera stands off the line through the object's centre (a free camera: `eye_weave`)
    pub eye: bool,
    /// an opaque or milky body lit by the surface pass after the tracer (surface.rs): what the tracer
    /// gathers itself changes with it (none: as it always was)
    pub surface: crate::surface::Kind,
    /// the parts engine's fine bricks are read by `mapN` and the normal (refine.rs)
    pub refine: bool,
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
/// Plain glass's paths inside end once what they carry falls below `cut` (as written in trace.wgsl: 0.02).
/// A family may end them sooner (`Family::thr_cut`): gems, whose light that is still inside after many
/// bounces carries little and costs most of the frame (docs/cache_bounces.md).
pub fn cut_weave(src: &str, cut: f64) -> String {
    if cut == 0.02 { return src.to_string(); }
    let c = format!("{cut:.3}");
    let s = crate::weave::replace(src, "render.cut_bounce", "if (b >= u.nb || thr < 0.02) { break; }", &format!("if (b >= u.nb || thr < {c}) {{ break; }}"));
    crate::weave::replace(&s, "render.cut_reflected", "      if (thr < 0.02) { break; }\n      dir = exit.reflected;", &format!("      if (thr < {c}) {{ break; }}\n      dir = exit.reflected;"))
}

/// Light leaving plain glass that carries less than a twentieth of the sample sees the lights at once
/// instead of being followed through further pieces. Only for shapes where those paths cost most and add
/// nothing that can be seen (a shape reshaped by the surface studio, trees: docs/cache_bounces.md); plain
/// glass is traced as it always was.
pub fn leave_weave(src: &str, on: bool) -> String {
    if !on { return src.to_string(); }
    crate::weave::replacen(src, "render.leave", "        if (LAYERS > 0) {\n          L += thr * (1.0 - F) * att * beyond(","        if (LAYERS > 0 && thr * (1.0 - F) * max(att.x, max(att.y, att.z)) >= 0.05) {\n          L += thr * (1.0 - F) * att * beyond(", 1)
}

pub fn tracer_source(m: &March, v: Variant, profile: bool) -> String {
    tracer_from(TRACE_WGSL, m, v, profile)
}

/// The same with the first ray traced through any lens (camera.rs): another kind of picture than
/// perspective, straight lines bowed, a leaning plane of focus, highlights with a shape of their own.
pub fn tracer_source_lens(m: &March, v: Variant, profile: bool) -> String {
    crate::camera::finish(tracer_from(&crate::camera::tracer(TRACE_WGSL), m, v, profile))
}

/// A tracer's text from another template (the matcap view's: matcap.rs), woven as the live tracer is.
pub(crate) fn tracer_from_template(template: &str, m: &March, v: Variant) -> String {
    tracer_from(template, m, v, false)
}

fn tracer_from(template: &str, m: &March, v: Variant, profile: bool) -> String {
    let prof = if profile { ["var<private> prof: vec3f;", "prof.x += 1.0; if (abs(v) > 0.06) { prof.z += 1.0; } ", "prof.y += 1.0; ", "if (true) { return vec4f(prof.x, prof.y, prof.z, depth); } "] } else { [""; 4] };
    let lights = "textureSampleLevel(envTex, envSmp, uv, 0.0).rgb";
    // the lighting's own declarations: none for glass lit by the lights' map
    let lit = v.light != LightFrom::Lights || v.backdrop || v.clay || v.textured || v.mat.paint_bodies;
    // the material's: none for plain glass, which is then the tracer exactly as it always was
    // (a scene of several bodies with materials of their own has a tracer of its own; clay hides what
    // things are made of, so it is drawn as ever)
    let mut optics = v.mat;
    optics.near |= v.near && optics.rough_ref;
    let mat = if v.mat.bodies > 0 && !v.clay {
        // A dressed construction cut can introduce another interface at the hit.
        optics.paint_only &= !v.world_faces;
        if v.world_faces { optics.paint_shadow_medium = None; }
        material::bodies_parts(&optics, TRACE_MATERIAL_WGSL, TRACE_BODIES_WGSL, v.light == LightFrom::Picture)
    } else {
        material::tracer_parts(&optics, TRACE_MATERIAL_WGSL, v.light == LightFrom::Picture, v.clay)
    };
    let shade = if v.textured { "modelOrMaterial(ro, rd, t, xi, W, fc)" } else { mat.shade };
    let mut decl = format!("{}{}{}", if lit { TRACE_LIGHT_WGSL.trim_end() } else { "" }, mat.decl, crate::stage::tracer_decl(v.stage, v.backdrop, shade));
    if v.textured {
        // Imported opaque surfaces share the canonical GGX arithmetic. Reuse material declarations
        // when an assembly already supplied them rather than compiling duplicate WGSL functions.
        if !decl.contains("fn fresMetal(") { decl.push_str(include_str!("shaders/conductor_fresnel.wgsl")); }
        if !decl.contains("fn pbr_ggx_evaluate(") { decl.push_str(include_str!("shaders/pbr/ggx.wgsl")); decl.push_str(include_str!("shaders/pbr/rough_conductor.wgsl")); }
        decl.push_str(include_str!("shaders/pbr/imported_opaque.wgsl"));
        let lamps = if v.near { " + lampsDiffuse(p, n) * 0.31830989" } else { "" };
        decl.push_str(&include_str!("shaders/model_surface.wgsl").replace("{{FILE_DIFFUSE_LAMPS}}", lamps));
        decl.push_str(&format!("\nfn modelOrMaterial(ro: vec3f, rd: vec3f, t: f32, xi: f32, W: i32, fc: vec2f) -> vec3f {{\n  let hit = fileSurface(ro + rd*t, rd, 1.0);\n  if (hit.a > 0.0) {{ return hit.rgb; }}\n  return {};\n}}\n",mat.shade));
    }
    let src = template
        .replace("{{DIELECTRIC_FRESNEL}}", DIELECTRIC_FRESNEL_WGSL.trim_end())
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
        .replace("{{BACKGROUND_DIR}}", if v.stage.sky { "skyBackdrop(rd)" } else if v.stage.backdrop { "stageBackdrop(rd)" } else if v.backdrop { "backdrop(rd)" } else { "vec3f(0.0)" })
        .replace("{{BACKGROUND}}", crate::stage::background(v.stage, v.backdrop))
        .replace("{{SHADE}}", shade)
        .replace("{{GRAZE}}", if v.stage.any() { "col" } else { mat.graze })
        // Every subsequent shader edit matches WGSL source lines. A Windows
        // checkout must initialise the enlarged stage exactly as an LF one does.
        .replace("\r\n", "\n");
    let src = if v.stage.atmo {
        const ATMO_ANCHOR: &str = "  return vec4f(max(col, vec3f(0.0))";
        assert_eq!(crate::weave::found(&src, "render.atmo", ATMO_ANCHOR), 1, "sampleAt return anchor missing");
        src.replacen(
            ATMO_ANCHOR,
            "  col = stageAtmosphere(ro, rd, t, depth, fc, xi, col);\n  return vec4f(max(col, vec3f(0.0))",
            1,
        )
    } else {
        src
    };
    let src = if v.textured { crate::model_surface::secondary_rays(src) } else { src };
    let src = if v.world_faces && v.mat.bodies>0 {crate::world_faces::weave(src,v.mat.bodies==1)} else {src};
    // a grown stage: the volume spans a larger box, and the ball that holds the scene is larger
    let src = if v.grown { grown(&src) } else { src };
    // the parts engine's fine bricks (never in the default tracer)
    let src = if v.refine { crate::refine::weave(&src) } else { src };
    // a camera off the axis: its rays start where it stands (the default tracer never has this)
    let src = if v.eye || template.contains("  cam: Cam,") { eye_weave(&src) } else { src };
    // (lit by the surface pass: the clay gathers only the room, and leaves the lamps to the pass)
    let src = crate::surface::weave(&src, &v);
    // the lamps, and then what the bodies cast, are put in last (never into the default tracer)
    let src = if v.near { crate::lamps::weave(&src) } else { src };
    let woven = if v.cast { crate::cast::weave(&src) } else { None };
    let cast_in = woven.is_some();
    let src = woven.unwrap_or(src);
    // (and with both, the lamps' shadows: only where casting went in, which it does not for clay lit by
    // the surface pass, whose shadows the pass draws)
    let src = if cast_in && v.near { crate::lamps::weave_shadows(&src) } else { src };
    let src = if cast_in && v.stage.cast && v.stage.floor && v.mat.metal && v.mat.bodies == 0 && (!v.mat.rough || v.mat.rough_ref || v.mat.coat) && !v.mat.grating && !v.clay {
        crate::stage::reflect_cast_floor(&src, v.near, v.mat)
    } else if cast_in && v.stage.cast && v.stage.floor && v.mat.metal && v.mat.bodies > 0 && !v.clay {
        crate::stage::reflect_bodies_floor(&src, v.near)
    } else { src };
    // a rough clear body's tracer also writes, beside the picture, how far what is seen through the
    // surface is to be blurred afterwards (frost.rs)
    let src = if rough_split(&v, profile) {
        let rough = frost::with_aux(&src);
        if v.mat.bodies > 0 {
            crate::weave::replace(&rough, "render.rough_shift", "let shift = vec2f(ign(fc + 19.19), ign(fc.yx + 47.3));",
                "let shift = select(vec2f(ign(fc + 19.19), ign(fc.yx + 47.3)), vec2f(0.5), u.pad1 != 0);")
        } else { rough }
    } else { src };
    // a tracer drawn by chance and denoised also writes, beside the picture, the light it found inside
    // the body, which is noisy (the picture then holds the rest: denoise.rs)
    if v.denoise && !profile { denoise::with_split(&src, v.mat.rough_ref) } else { src }
}

fn body_uniforms_equal(a: &[f32], b: &[f32]) -> bool {
    let metadata = (material::MB_HEAD - material::MAX_BODIES * 4)..material::MB_HEAD;
    a.len() == b.len() && a.iter().zip(b).enumerate().all(|(index, (a, b))| {
        if metadata.contains(&index) { a.to_bits() == b.to_bits() } else { a == b }
    })
}

#[cfg(test)]
#[test]
fn body_medium_cache_equality_uses_raw_bits() {
    let mut a = vec![0.0; material::MB_FLOATS];
    let mut b = a.clone();
    a[33] = f32::from_bits((-17i32) as u32);
    b[33] = a[33];
    assert!(a[33].is_nan() && body_uniforms_equal(&a, &b));
    b[33] = f32::from_bits((-18i32) as u32);
    assert!(!body_uniforms_equal(&a, &b));
}

#[cfg(test)]
#[test]
fn windows_shader_lines_preserve_stage_bounds_and_camera_code() {
    let march = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    let lf = TRACE_WGSL.replace("\r\n", "\n");
    let crlf = lf.replace('\n', "\r\n");
    for lens in [false, true] {
        for grown in [false, true] {
            let variant = Variant { grown, ..Default::default() };
            let build = |src: &str| {
                if lens {
                    crate::camera::finish(tracer_from(&crate::camera::tracer(src), &march, variant, false))
                } else {
                    tracer_from(src, &march, variant, false)
                }
            };
            let reference = build(&lf);
            let windows = build(&crlf);
            assert_eq!(windows.contains("HB = bitcast<f32>(u.pad2);"), grown);
            assert!(windows == reference, "shader differs for lens={lens}, grown={grown}");
        }
    }
}

/// A 1x1x1 volume of this format (what the bindings of an unused feature read).
fn standin(gpu: &Gpu, format: wgpu::TextureFormat) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor { label: Some("stand-in"), size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D3, format, usage: wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[] })
}

/// A tracer's text for a stage larger than ever: the volume's box and the ball the scene fits in, which
/// the tracer of a stage as it always was holds as constants (and that text stays unchanged), are read
/// from its numbers instead (`TraceU::pad[2]`: the box's half-width, as a float's bits), so that one text
/// serves every size and a layer dragged across a size needs no new tracer.
pub fn grown(src: &str) -> String {
    let head = "fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
";
    let s = crate::weave::replace(src, "render.grown_bound", "const BOUND: f32 = 1.47;", "var<private> BOUND: f32;");
    let s = crate::weave::replace(&s, "render.grown_hb", "const HB: f32 = 1.5;", "var<private> HB: f32;");
    crate::weave::replace(&s, "render.grown_fs", head, &format!("{head}  HB = bitcast<f32>(u.pad2);
  BOUND = HB * 0.98;
"))
}

/// A tracer's text for a camera that stands off the line through the object's centre (`TraceU::eye`):
/// its first ray starts there, and the width of a pixel at a point is measured from there. Woven in only
/// while the camera stands off that line (and always into the tracer for any lens, which reads it in
/// camera_trace.wgsl), so that the tracer of plain glass seen from the line stays its old text.
pub fn eye_weave(src: &str) -> String {
    let head = "  medium: vec4i,           // raw identity, signed interface priority, two reserved zero lanes\n";
    let s = if src.contains("  eye: vec4f,") { src.to_string() } else {
        assert_eq!(src.matches(head).count(), 1, "the tracer's block has changed");
        src.replace(head, &format!("{head}  eye: vec4f,             // xy: where the camera stands across and up from (0, 0, dist)\n"))
    };
    let s = crate::weave::replace(&s, "render.eye_ro", "var ro = vec3f(0.0, 0.0, u.dist);", "var ro = vec3f(u.eye.xy, u.dist);");
    crate::weave::replace(&s, "render.eye_rest", "vec3f(0.0, 0.0, u.dist)", "vec3f(u.eye.xy, u.dist)")
}

/// Is the stage for rough glass run after this tracer? Not when the tracer is denoised: then what is
/// seen through the surface is in the part the denoiser filters (and the targets of both would be more
/// than a pass may write).
fn rough_split(v: &Variant, profile: bool) -> bool {
    frost::splits(&v.mat, v.clay, profile) && !(v.denoise && !profile)
}

/// GLASSFORMS_HERO_MOVE=2: the moving hero (hero.rs `moving`) for every frame, still ones too (for looking
/// at its frames alone: a still of one frame is one frame of it).
fn hero_move_always() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("GLASSFORMS_HERO_MOVE").is_ok_and(|v| v == "2"))
}

/// Time spent making tracers since the program started, in nanoseconds: (shader modules, pipelines, count).
pub static TRACER_BUILDS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

fn tracer_pipeline(gpu: &Gpu, layout: &wgpu::BindGroupLayout, cast: Option<&wgpu::BindGroupLayout>, src: &str) -> wgpu::RenderPipeline {
    use std::sync::atomic::Ordering::Relaxed;
    // (measuring only: GLASSFORMS_SHADER_NONCE=<n> adds a test that never passes, so that the driver
    // sees a shader it has not compiled before)
    let nonced;
    let src = match std::env::var("GLASSFORMS_SHADER_NONCE") {
        Ok(n) => { nonced = src.replacen("fn ign(p: vec2f) -> f32 { return", &format!("fn ign(p: vec2f) -> f32 {{ if (u.frame == -{}) {{ return 0.5; }} return", n.trim()), 1); nonced.as_str() }
        Err(_) => src,
    };
    // (measuring only: GLASSFORMS_PROF_EDIT=<file> of `anchor ||| replacement` lines, `\n` for a newline,
    // edits every tracer before it is made: for timing what a part costs by leaving it out)
    let edited;
    let src = match std::env::var_os("GLASSFORMS_PROF_EDIT").and_then(|p| std::fs::read_to_string(p).ok()) {
        Some(edits) => {
            let mut s = src.to_string();
            for (from, to) in edits.lines().filter_map(|l| l.split_once(" ||| ")) {
                let (from, to) = (from.replace("\\n", "\n"), to.replace("\\n", "\n"));
                eprintln!("[prof-edit] {} found {} times", from.lines().next().unwrap_or(""), s.matches(&from).count());
                s = s.replace(&from, &to);
            }
            edited = s;
            edited.as_str()
        }
        None => src,
    };
    // (measuring only: GLASSFORMS_DUMP_TRACER=<folder> writes each tracer's text there, named by its hash)
    if let Some(dir) = std::env::var_os("GLASSFORMS_DUMP_TRACER") {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        src.hash(&mut h);
        let _ = std::fs::write(std::path::Path::new(&dir).join(format!("trace_{:016x}.wgsl", h.finish())), src);
    }
    let t0 = std::time::Instant::now();
    let module = gpu.shader_in("trace", src, true, &["trace", "cast"]);
    let t1 = std::time::Instant::now();
    let pipe = tracer_pipeline_of(gpu, layout, cast, src, module);
    TRACER_BUILDS[0].fetch_add((t1 - t0).as_nanos() as u64, Relaxed);
    TRACER_BUILDS[1].fetch_add(t1.elapsed().as_nanos() as u64, Relaxed);
    TRACER_BUILDS[2].fetch_add(1, Relaxed);
    // (measuring only: GLASSFORMS_LOG_COMPILE=1 says how long each tracer took to make)
    if std::env::var_os("GLASSFORMS_LOG_COMPILE").is_some() { eprintln!("[compile] tracer {} bytes: module {:.0} ms, pipeline {:.0} ms", src.len(), (t1 - t0).as_secs_f64() * 1e3, t1.elapsed().as_secs_f64() * 1e3); }
    pipe
}

fn tracer_pipeline_of(gpu: &Gpu, layout: &wgpu::BindGroupLayout, cast: Option<&wgpu::BindGroupLayout>, src: &str, module: wgpu::ShaderModule) -> wgpu::RenderPipeline {
    if src.contains(denoise::MARK) { return denoise::tracer_pipeline(gpu, layout, cast, &module); }
    // (a tracer with casting in it reads a second group: what the bodies cast)
    if src.contains("struct FrOut") { return frost::tracer_pipeline(gpu, layout, cast, &module); }
    let layouts: Vec<&wgpu::BindGroupLayout> = std::iter::once(layout).chain(cast).collect();
    // (one wavelength inside while moving, its dispersion written to a second target: hero.rs `split`)
    if src.contains(crate::hero::SPLIT_MARK) { return crate::adapt::tracer_pipeline(gpu, "trace (hero split)", &module, &layouts, &[HDR, HDR, HDR]); }
    // (every tracer runs only where the stencil of adaptive refining leaves a pixel to be traced: adapt.rs)
    crate::adapt::tracer_pipeline(gpu, "trace", &module, &layouts, &[HDR])
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
    /// where the camera stands across and up from the line through the object's centre: now, then
    eye: [f32; 4],
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
    /// rgb: the share of its light the sharp picture keeps (the lens's glare is mixed in; the old bloom
    /// is added, and keeps 1)
    pub glare: [f32; 4],
    /// the lens, and its colour fringing and fall-off of light (read only by the final pass built for them)
    pub lens: [[f32; 4]; 4],
    pub fx: [f32; 4],
}

/// Everything about a frame that is not the tracer's own block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Look {
    /// Revision of the baked geometry visible to this draw.
    pub shape_no: u64,
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
    /// the lens as its glare needs it, and which bloom is drawn
    pub glare: crate::glare::GlareLens,
}

/// What the antialiasing needs to know of the frame before.
#[derive(Clone, Copy)]
struct PrevView {
    shape_no: u64,
    obj: [[f32; 4]; 3],
    env: [[f32; 4]; 3],
    /// the room as the camera saw it, and where the floor stood in it
    room: [[f32; 4]; 3],
    floor: f32,
    cam: [f32; 2],
    eye: [f32; 2],
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

/// Legacy `bloomKind=1` HDR mip pyramid, allocated lazily only when asked for (`look.glare.pyramid()`).
struct Pyramid {
    /// level-0 `BloomU` buffer (`pad[0]` = interactive Karis firefly weight, 0 when settled)
    u_down0: wgpu::Buffer,
    down0_res: ([f32; 2], [f32; 2]),
    last_knee: f32,
    /// the first bloom level reads accum[i]; the third reads the focused picture instead
    g_down0: [wgpu::BindGroup; 3],
    mips: Vec<Mip>,
    g_final: [wgpu::BindGroup; 3],
    g_final_deep: Option<[wgpu::BindGroup; 3]>,
}

struct BloomPipes {
    layout: wgpu::BindGroupLayout,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
}

struct Targets {
    w: u32,
    h: u32,
    tw: u32,
    th: u32,
    trace_tex: wgpu::Texture,
    trace_view: wgpu::TextureView,
    accum: [wgpu::Texture; 2],
    accum_view: [wgpu::TextureView; 2],
    focused_view: wgpu::TextureView,
    /// accumulate: read accum[i] as the picture so far (and the new frame), write the other
    g_accum: [wgpu::BindGroup; 2],
    /// the same, reading the traced frame as the stage for rough glass has blurred it
    g_accum_rough: [wgpu::BindGroup; 2],
    /// legacy `bloomKind=1` mip pyramid and its final bind groups, built on demand
    pyramid: Option<Pyramid>,
    /// final pass bind groups reading the lens's glare (`bloomKind=0`)
    g_final_glare: [wgpu::BindGroup; 3],
    final_tex: wgpu::Texture,
    final_view: wgpu::TextureView,
    /// the screen again in 16-bit floats, with no 8-bit dither, for video with more than 8 bits (when asked for)
    deep: Option<(wgpu::Texture, wgpu::TextureView, [wgpu::BindGroup; 3])>,
}

/// What the tier row asks of the Ray Reconstruction draft this frame (dlss_rr.rs).
#[cfg(feature = "dlss")]
#[derive(Clone, Copy, Debug)]
pub struct DraftWant {
    /// draw this frame through the draft
    pub on: bool,
    /// have the context and the tracer ready (the draft is selected, or Auto may use it): they take a moment to make
    pub prepare: bool,
    /// the share of each side that is traced
    pub scale: f64,
    /// the bounce cap
    pub bounces: i32,
    pub motion: crate::dlss_rr::Motion,
}
#[cfg(feature = "dlss")]
impl Default for DraftWant {
    fn default() -> Self { DraftWant { on: false, prepare: false, scale: 0.5, bounces: 2, motion: crate::dlss_rr::Motion::Surface } }
}

/// The draft tier's state in the renderer: its context, its tracer's two pipelines (and the thread making them).
#[cfg(feature = "dlss")]
#[derive(Default)]
pub struct DraftState {
    pub want: DraftWant,
    ctx: Option<crate::dlss_rr::Draft>,
    /// the draft tracer's colour and guide pipelines by tracer key (the key alternates between a moving and a still shape: kept, not remade)
    pipes: HashMap<String, (wgpu::RenderPipeline, wgpu::RenderPipeline)>,
    building: Option<(String, std::sync::mpsc::Receiver<Result<(wgpu::RenderPipeline, wgpu::RenderPipeline), String>>)>,
    /// why the draft cannot run (it is then off for the session)
    pub failed: Option<String>,
    /// draft frames drawn
    pub drawn: u64,
    said: bool,
    said_frame: bool,
    misses: u64,
    fell: u64,
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
    l_final: wgpu::BindGroupLayout,
    trace_pipes: HashMap<String, wgpu::RenderPipeline>,
    trace_key: String,
    p_accum: wgpu::RenderPipeline,
    bloom: Option<BloomPipes>,
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
    /// the fine bricks' page, atlas and numbers, or stand-ins (refine.rs)
    refine: Option<(wgpu::TextureView, wgpu::TextureView, wgpu::Buffer)>,
    no_page: wgpu::TextureView,
    no_atlas: wgpu::TextureView,
    no_refine_u: wgpu::Buffer,
    u_bodies: wgpu::Buffer,
    air_envelope: Option<crate::air_envelope::AirEnvelope>,
    last_bodies: Vec<f32>,
    /// what the tracer is told about what lies inside the body (cracks, grains, fibres, a changing tint)
    u_inner: wgpu::Buffer,
    last_inner: Vec<f32>,
    /// a cut stone's planes (gem.wgsl), and whether the stone shown now has them
    u_gem: wgpu::Buffer,
    last_gem: Vec<f32>,
    gem: bool,
    /// the stone is reshaped by the surface studio: its planes only bound the march (`gem_carved_weave`)
    pub gem_carved: bool,
    /// the shape's formula, when the tracer reads it instead of the volume (gf_core::sdf): what names it,
    /// and its text for the tracer
    sdf: Option<(String, String)>,
    /// a tracer for a tree of several formula parts takes seconds to make the first time: unless this is
    /// set (stills, checks), it is made on another thread and the tracer in use goes on drawing (on the
    /// volume the tree is measured into) until it is there
    pub wait_builds: bool,
    /// the tracer just changed to the one made on another thread (the picture starts again)
    swapped: bool,
    pub(crate) world_faces: crate::world_faces::SceneFaces,
    pub(crate) file_surfaces: crate::model_surface::ModelSurfaces,
    u_sdf: wgpu::Buffer,
    /// the formula's table (`gf_core::field::table`), for a formula with one
    s_sdf_tbl: wgpu::Buffer,
    last_sdf: Vec<f32>,
    last_sdf_tbl: Vec<f32>,
    /// a tree's grid parts' volumes by slot, and what stands in for a slot with none (csg.rs)
    grids: Vec<Option<wgpu::TextureView>>,
    grid_stand_in: crate::csg::GridStandIn,
    /// a tree's cells' lists (csg.rs; a small stand-in until there is a tree)
    csg_lists: wgpu::Buffer,
    csg_table: wgpu::Buffer,
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
    /// cached volumetric sky-space dome cubemap and companion 2D ground shadow map (clouds.rs)
    sky_space: Option<(wgpu::TextureView, wgpu::TextureView)>,
    /// a tracer being built on another thread: which, and where it will arrive
    building: Option<(String, std::sync::mpsc::Receiver<Result<(wgpu::RenderPipeline, bool), String>>)>,
    /// a DLSS-centred tracer being built on another thread so slider drags do not stall the UI
    building_centred: Option<(String, std::sync::mpsc::Receiver<Result<wgpu::RenderPipeline, String>>)>,
    /// tracers whose build failed, each tried again after a wait (build_retry.rs)
    failed_tracers: crate::build_retry::Failures,
    /// tracers made beforehand on another thread (`warm`): what is on its way, and what has been asked for
    warming: Option<std::sync::mpsc::Receiver<(String, wgpu::RenderPipeline, bool)>>,
    warm_asked: Vec<String>,
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
    /// Trace passes submitted, including partial native updates. Read by the session profiler.
    pub trace_passes: u64,
    jitter_n: u32,
    /// the last frame was folded in by the antialiasing that follows motion
    was_taa: bool,
    prev: Option<PrevView>,
    last_final: Option<FinalU>,
    /// build the tracer so that it counts its lookups instead of making a picture
    pub profile: bool,
    /// where plain glass's paths inside end, by what they carry (`cut_weave`; set from the family before `set_scene`)
    pub cut: f64,
    /// light leaving plain glass that carries little is not followed through further pieces (`leave_weave`):
    /// set before `set_scene`, for a shape the surface studio has reshaped and for trees
    pub leave_cut: bool,
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
    /// the lens's glare (glare.rs), and whether the last frame drew it (rather than the bloom pyramid)
    glare: crate::glare::Glare,
    glared: bool,
    glare_motion: (f32, [f32; 2]),
    glare_cut: bool,
    pyramid_interactive: bool,
    /// direct light with shadows on opaque and milky bodies, after the tracer (surface.rs)
    surface: crate::surface::Surface,
    /// DLSS while the view moves, when asked for and built in (dlss.rs)
    pub up: crate::dlss::Upscale,
    /// the Ray Reconstruction draft tier (dlss_rr.rs): what is asked for, its context and its tracer
    #[cfg(feature = "dlss")]
    pub draft: DraftState,
    /// one wavelength inside the glass (hero.rs): as asked for, and whether the tracer in use has it
    pub hero: crate::hero::Hero,
    /// the hero as woven into the tracer in use (None: the whole comb)
    hero_cur: Option<crate::hero::Hero>,
    /// one wavelength inside while the view moves, put back together from the neighbours (hero.rs
    /// `moving`, `split`; docs/perf_r13.md): set by the engine for the shapes that need it
    pub hero_move: Option<crate::hero::Hero>,
    /// the tracers (by key) that write their dispersion apart, and the pass that puts their frame together
    split_keys: std::collections::HashSet<String>,
    recon: Option<crate::hero::Recon>,
    /// the antialiasing's pipeline has been made with colour kept apart (hero.rs chroma_taa)
    accum_chroma: bool,
    /// Keeping the frame rate (autoscale.rs): moving frames are traced at this share of the picture's
    /// size per side (1: all of it), into the corner of the traced frame, and the antialiasing that
    /// follows motion fills in the full-size picture from them (`ACCUM_DYN`). Only where that
    /// antialiasing reads the traced frame straight (`dyn_ok`); still frames are always traced whole.
    pub dyn_scale: f64,
    /// the share the last frame was traced at
    pub dyn_used: f64,
    /// Bounce and wavelength caps submitted to the most recent trace.
    pub traced_caps: [i32; 2],
    /// Requested for one live refinement update. Both halves use one sampling sequence entry.
    pub(crate) split_refine: bool,
    /// The half last traced: whole, the first set of native columns, or the complementary columns.
    pub traced_half: crate::split_half::Half,
    pass_times: Option<crate::pass_times::PassTimes>,
    p_accum_dyn: Option<wgpu::RenderPipeline>,
    building_dyn: Option<std::sync::mpsc::Receiver<wgpu::RenderPipeline>>,
    /// adaptive refining: each pixel's own count, and the stencil that leaves pixels whose mean is
    /// certain out of the tracer's passes (adapt.rs)
    pub adapt: crate::adapt::Adapt,
    /// The matcap view (matcap.rs): its settings while it is what is drawn (None: the full render), and
    /// whether the stage is grown (the engine says, as it says for the tracer's variant)
    pub matcap_on: Option<gf_core::view_mode::Matcap>,
    pub matcap_grown: bool,
    pub matcap: crate::matcap::Matcap,
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
    // the lens's glare is mixed in (the sharp picture keeping 1 - E of its light), the old bloom added
    let glare = crate::glare::GlareLens::of(p);
    let (amt, keep) = if glare.pyramid() { (p.bloom / levels as f64, [1.0f32; 3]) } else { (1.0, crate::glare::keep(&glare, w, h)) };
    // the tone curve and the grade as they are set (with nothing set: the ACES curve, as ever)
    let grade = gf_core::stage::Grade::of(p);
    // the lens's own colour fringing and fall-off of light, when it has any: the picture with its colours
    // drawn apart, and how much light each pixel keeps
    let lens = crate::camera::develop(hdr, w, h, p);
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    let last_x = bw.saturating_sub(1) as isize;
    let last_y = bh.saturating_sub(1) as isize;
    let load = |x: isize, y: isize, c: usize| -> f64 {
        let cx = x.clamp(0, last_x) as usize;
        let cy = y.clamp(0, last_y) as usize;
        b[(cy * bw + cx) * 4 + c] as f64
    };
    let catmull = |f: f64| -> [f64; 4] {
        [
            f * (-0.5 + f * (1.0 - 0.5 * f)),
            1.0 + f * f * (-2.5 + 1.5 * f),
            f * (0.5 + f * (2.0 - 1.5 * f)),
            f * f * (-0.5 + 0.5 * f),
        ]
    };
    for y in 0..h as usize {
        for x in 0..w as usize {
            let (u, v) = ((x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64);
            // Smooth Catmull-Rom reconstruction of the bloom level, clamped from below by its 2x2 footprint
            // (matching post.wgsl's bloomSmooth).
            let (ax, ay) = (u * bw as f64 - 0.5, v * bh as f64 - 0.5);
            let (ix, iy) = (ax.floor() as isize, ay.floor() as isize);
            let wx = catmull(ax - ix as f64);
            let wy = catmull(ay - iy as f64);
            let vig = 1.0 - p.vignette * ((u - 0.5).powi(2) + (v - 0.5).powi(2));
            let mut exposed = [0.0f64; 3];
            for c in 0..3 {
                let mut r = 0.0;
                for jy in 0..4 {
                    let mut row = 0.0;
                    for jx in 0..4 {
                        row += load(ix + jx as isize - 1, iy + jy as isize - 1, c) * wx[jx];
                    }
                    r += row * wy[jy];
                }
                let lo = load(ix, iy, c).min(load(ix + 1, iy, c)).min(load(ix, iy + 1, c)).min(load(ix + 1, iy + 1, c));
                let bl = r.max(lo);
                let at = y * w as usize + x;
                let (light, kept) = lens.as_ref().map_or((hdr[at * 4 + c], 1.0), |l| (l.0[at * 3 + c], l.1[at] as f64));
                exposed[c] = (light as f64 * keep[c] as f64 + bl * amt) * vig * kept * p.exposure;
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

/// Sampled textures one tracer fragment stage binds, counted from the `l_trace` layout in `Renderer::new`
/// (bindings 1, 3, 6, 8, 10, 24 to 27, the dome, the cloud shadow, the page, the atlas, 12 and four
/// projected volumes), without the grid parts and without the cast group.
pub const TRACE_TEXTURES: u32 = 18;
/// The cast group's sampled textures (cast.rs: bindings 1, 2, 4).
pub const CAST_TEXTURES: u32 = 3;
/// Uniform buffers the tracer layout binds (0, 5, 9, stage, refine, lamps, 13, 32, 33, gem, sdf) and the cast group's one.
pub const TRACE_UNIFORMS: u32 = 12;

/// Sampled textures the tracer stage needs, with the cast group, with or without a tree's grid volumes.
pub fn trace_textures_needed(grids: bool) -> u32 {
    TRACE_TEXTURES + CAST_TEXTURES + if grids { gf_core::csg::MAX_GRIDS as u32 } else { 0 }
}

/// Can the device bind what the tracer needs at all? A sentence when it cannot.
pub fn check_trace_limits(l: &wgpu::Limits) -> Result<(), String> {
    let need = trace_textures_needed(false);
    if l.max_sampled_textures_per_shader_stage < need || l.max_uniform_buffers_per_shader_stage < TRACE_UNIFORMS {
        return Err(format!(
            "this graphics device is too small for the renderer: it allows {} sampled textures and {} uniform buffers per shader stage, the tracer needs {need} and {TRACE_UNIFORMS}",
            l.max_sampled_textures_per_shader_stage, l.max_uniform_buffers_per_shader_stage));
    }
    Ok(())
}

/// Can the tracer read a tree's grid parts (eight more textures and a sampler than it reads otherwise)?
pub fn grids_fit(gpu: &Gpu) -> bool {
    gpu.device.limits().max_sampled_textures_per_shader_stage >= trace_textures_needed(true)
}

impl Renderer {
    /// The volumes of a tree's grid parts, by slot (gf_core::csg): read by the tracer from the next
    /// `set_scene` on (which binds them).
    pub fn set_grids(&mut self, views: &[Option<wgpu::TextureView>], lists: &wgpu::Buffer, table: &wgpu::Buffer) {
        self.grids = views.to_vec();
        self.csg_lists = lists.clone();
        self.csg_table = table.clone();
    }

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
            texture_entry(crate::clouds::DOME_BINDING, wgpu::TextureViewDimension::Cube, true),
            texture_entry(crate::clouds::SHADOW_BINDING, d2, true),
            // (70 to 72: the parts engine's fine bricks: the page (unfiltered: read by texel), the atlas and their numbers, read only by tracers built for them)
            texture_entry(crate::refine::PAGE_BINDING, wgpu::TextureViewDimension::D3, false), texture_entry(crate::refine::ATLAS_BINDING, wgpu::TextureViewDimension::D3, true), uniform_entry(crate::refine::UNIFORM_BINDING, frag, false),
            // (44: the lamps, read only by tracers built with lamps)
            uniform_entry(crate::lamps::BINDING, frag, false),
            texture_entry(12, wgpu::TextureViewDimension::D3, true), uniform_entry(13, frag, false),
            // (32: what lies inside the body, read only by tracers built for it; 33: what lies inside each
            // of several bodies, read only by the tracer of several bodies built for it)
            uniform_entry(32, frag, false), uniform_entry(33, frag, false),
            // (34: a cut stone's planes, read only by tracers built for a stone)
            uniform_entry(GEM_BINDING, frag, false),
            // (35: the shape's numbers, read only by tracers built to read its formula)
            uniform_entry(gf_core::sdf::BINDING, frag, false),
            // (36: the formula's table, read only by tracers built for a formula with one)
            storage_entry(gf_core::sdf::TABLE_BINDING, frag),
            storage_entry(crate::world_faces::BINDING,frag),
            storage_entry(crate::model_surface::DATA_BINDING, frag),
            storage_entry(crate::model_surface::INSTANCES_BINDING, frag),
        ].into_iter().chain((0..4).map(|i| {
            texture_entry(crate::world_faces::PROJECTED_BINDING + i, wgpu::TextureViewDimension::D3, true)
        })).chain(
            // (50 to 58: the volumes of a tree's grid parts and their sampler, read only by tracers built for
            // a tree with grid parts; only where the device lets a shader read that many textures)
            crate::csg::tree_entries(gf_core::csg::GRID_BINDING_TRACER, frag, grids_fit(gpu))
        ).collect::<Vec<_>>());
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
        let l_final = gpu.bind_layout("final", &[uniform_entry(4, frag, false), texture_entry(5, d2, true), texture_entry(6, d2, true), sampler_entry(7)]);
        let accum = gpu.shader("accumulate", ACCUM_WGSL, false);
        let post = gpu.shader("post", POST_WGSL, true);
        let pipe = |label, module, fs, layout: &wgpu::BindGroupLayout, format, blend| gpu.pipeline(&PipeDesc { label, module, vs: "vs", fs, layouts: &[layout], format, blend, strip: false, buffers: &[] });
        Self {
            gpu: gpu.clone(),
            lin: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge),
            env_smp: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::Repeat),
            p_accum: pipe("accumulate", &accum, "fs", &l_accum, ACC, None),
            bloom: None,
            p_final: pipe("final", &post, "fs_final", &l_final, LDR, None),
            p_final_deep: pipe("final (deep)", &post, "fs_final", &l_final, HDR, None),
            u_final_deep: gpu.uniform_buffer("final (deep)", std::mem::size_of::<FinalU>() as u64),
            deep: false,
            l_cast: crate::cast::receiver_layout(gpu), g_cast: None,
            l_trace, l_accum, l_final,
            file_surfaces: crate::model_surface::ModelSurfaces::new(gpu),
            world_faces: crate::world_faces::SceneFaces::new(gpu),
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
            refine: None,
            no_page: standin(gpu, wgpu::TextureFormat::R32Float).create_view(&Default::default()),
            no_atlas: standin(gpu, wgpu::TextureFormat::R16Float).create_view(&Default::default()),
            no_refine_u: gpu.uniform_buffer("no refine", 64),
            u_bodies: gpu.uniform_buffer("trace: bodies", (material::MB_FLOATS * 4 + 16) as u64),
            air_envelope: None,
            last_bodies: Vec::new(),
            u_inner: gpu.uniform_buffer("trace: inside the body", (gf_core::inner::IU_FLOATS * 4) as u64),
            last_inner: Vec::new(),
            u_gem: gpu.uniform_buffer("trace: the stone's planes", (4 * (1 + gf_core::presets::gems::MAX_PLANES) * 4) as u64),
            last_gem: Vec::new(),
            sdf: None,
            wait_builds: true,
            swapped: false,
            u_sdf: gpu.uniform_buffer("trace: the shape's formula", (gf_core::field::UNIFORM_SIZE).max(gf_core::csg::UNIFORM_FLOATS * 4) as u64),
            s_sdf_tbl: gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("trace: the formula's table"), size: (16 * gf_core::field::TABLE_MAX) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
            last_sdf: Vec::new(),
            last_sdf_tbl: Vec::new(),
            gem: false, gem_carved: false,
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
            sky_space: None,
            building: None, building_centred: None, failed_tracers: Default::default(), warming: None, warm_asked: Vec::new(), scene: None, march: None, variant: Variant::default(),
            u_accum: gpu.uniform_buffer("accumulate", std::mem::size_of::<AccumU>() as u64),
            u_final: gpu.uniform_buffer("final", std::mem::size_of::<FinalU>() as u64),
            g_trace: None, t: None, acc_idx: 0, accum_n: 0, frame_no: 0, trace_passes: 0, jitter_n: 0, was_taa: false, prev: None, last_final: None, profile: false, cut: 0.02, leave_cut: false,
            dof: Dof::new(gpu), focused: false, cam: Default::default(), glare: crate::glare::Glare::new(gpu), glared: false, glare_motion: (0.0, [0.0; 2]), glare_cut: false, pyramid_interactive: false,
            frost: Frost::new(gpu),
            dn: Denoise::new(gpu), shown: 0,
            surface: crate::surface::Surface::new(gpu),
            up: Default::default(),
            #[cfg(feature = "dlss")]
            draft: Default::default(),
            hero: Default::default(), hero_cur: None, hero_move: None, split_keys: Default::default(), recon: None, accum_chroma: false, dyn_scale: 1.0, dyn_used: 1.0, traced_caps: [0; 2], traced_half: Default::default(), split_refine: false, pass_times: crate::pass_times::PassTimes::from_env(gpu), p_accum_dyn: None, building_dyn: None,
            adapt: crate::adapt::Adapt::new(gpu),
            matcap_on: None, matcap_grown: false, matcap: crate::matcap::Matcap::new(gpu),
            grids: Vec::new(), grid_stand_in: crate::csg::GridStandIn::new(gpu), csg_lists: gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("no csg lists"), size: 64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false }),
            csg_table: gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("no csg parts"), size: (gf_core::csg::PART_FLOATS * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false }),
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
        let halves: Vec<half::f16> = { use rayon::prelude::*; rgba.par_iter().map(|&v| half::f16::from_f32(v)).collect() };
        self.gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&halves),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
    }

    fn update_air_envelope(&mut self, volume: &Volume) {
        if self.bodies.is_some() && self.sdf.as_ref().is_none_or(|(_, decl)| decl == crate::sdf::GUARD_DECL) {
            self.air_envelope.get_or_insert_with(|| crate::air_envelope::AirEnvelope::new(&self.gpu))
                .update(&self.gpu, volume, &self.u_bodies);
        } else if self.air_envelope.is_some() {
            self.gpu.queue.write_buffer(&self.u_bodies, (material::MB_FLOATS * 4) as u64, &[0;4]);
        }
    }

    /// Say which lights and which volume the tracer reads, and how it marches (these differ by family).
    pub fn set_scene(&mut self, env: &wgpu::TextureView, vol: &Volume, step_scale: f64, steps_out: u32, steps_in: u32, layers: u32) {
        self.march = Some(March { step_scale, steps_out, steps_in, layers });
        self.update_air_envelope(vol);
        self.scene = Some((env.clone(), vol.view.clone()));
        self.build_tracer();
        self.bind_tracer();
    }

    /// Point the tracer at another volume of the same shape (the one with the material's bubbles cut
    /// into it, or back to the shape's own); everything else about the scene stays.
    pub fn set_volume(&mut self, vol: &Volume) {
        let Some((env, _)) = self.scene.take() else { return };
        self.update_air_envelope(vol);
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
        if body_uniforms_equal(&self.last_bodies, numbers) { return false; }
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

    /// The planes of the cut stone shown (`gf_core::presets::gems::inside_uniforms`), or None when the
    /// shape is not one. Takes effect with the next `set_scene`, which builds the tracer.
    pub fn set_gem(&mut self, numbers: Option<&[f32]>) {
        self.gem = numbers.is_some();
        if let Some(n) = numbers {
            if self.last_gem != n {
                self.gpu.queue.write_buffer(&self.u_gem, 0, bytemuck::cast_slice(n));
                self.last_gem = n.to_vec();
            }
        }
    }

    /// The formula the tracer reads instead of the volume (gf_core::sdf): what names it (a tracer is kept
    /// under it), its text for the tracer, and its numbers (`gf_core::field::uniforms`); or None to read the
    /// volume. Takes effect with the next `set_scene`, which builds the tracer.
    /// (`decl` may be None when the formula named is the one the tracer has already.)
    pub fn set_sdf(&mut self, formula: Option<(&str, Option<String>, &[f32])>) {
        match formula {
            Some((key, decl, numbers)) => {
                if self.sdf.as_ref().is_none_or(|s| s.0 != key) {
                    let Some(decl) = decl else { self.sdf = None; return };
                    self.sdf = Some((key.to_string(), decl));
                }
                if self.last_sdf != numbers {
                    self.gpu.queue.write_buffer(&self.u_sdf, 0, bytemuck::cast_slice(numbers));
                    self.last_sdf = numbers.to_vec();
                }
            }
            None => self.sdf = None,
        }
    }

    /// The formula's table (`gf_core::field::table`), for a formula traced with one.
    pub fn set_sdf_table(&mut self, table: &[f32]) {
        if table.is_empty() || self.last_sdf_tbl == table { return; }
        let n = table.len().min(4 * gf_core::field::TABLE_MAX);
        self.gpu.queue.write_buffer(&self.s_sdf_tbl, 0, bytemuck::cast_slice(&table[..n]));
        self.last_sdf_tbl = table.to_vec();
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

    /// How many frames the still picture has averaged so far (0 while the view moves and the antialiasing
    /// follows it).
    pub fn still_frames(&self) -> u32 {
        if self.was_taa { 0 } else { self.accum_n }
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

    /// The fine bricks the tracer reads (the page, the atlas, their numbers), or none. Set before `set_scene`; a tracer built for them is
    /// asked for with `Variant::refine`.
    pub fn set_refine(&mut self, r: Option<(wgpu::TextureView, wgpu::TextureView, wgpu::Buffer)>) {
        self.refine = r;
    }
    pub fn refine_on(&self) -> bool { self.refine.is_some() }

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
        self.text_with(m, v, self.cam.wanted, self.sdf.as_ref())
    }
    /// The same through either lens and with any formula (or none): what `tracer_text` would give were
    /// those the ones in use (`warm`).
    pub fn text_with(&self, m: &March, v: Variant, lens: bool, sdf: Option<&(String, String)>) -> String {
        let src = if lens { tracer_source_lens(m, v, self.profile) } else { tracer_source(m, v, self.profile) };
        let src = if v.mat == Default::default() { leave_weave(&cut_weave(&src, self.cut), self.leave_cut) } else { src };
        let src = if self.gem && gem_fits(&v) { if self.gem_carved { gem_carved_weave(&src) } else { gem_weave(&src) } } else { src };
        let src = match self.hero_cur {
            // (written apart while moving: the plain tracer when that cannot be woven, never the hero alone)
            Some(h) if h.split > 0 => crate::hero::weave(&src, &h).and_then(|t| crate::hero::split(&t)).unwrap_or(src),
            Some(h) => crate::hero::weave(&src, &h).unwrap_or(src),
            None => src,
        };
        // (measuring only, docs/perf_r13.md: GLASSFORMS_PROBE_TILEXI=1 gives every pixel of an 8x8 tile the
        // same wavelengths, to see what rays that keep together across a warp would cost; it changes the
        // picture and is never set otherwise)
        let src = if std::env::var("GLASSFORMS_PROBE_TILEXI").is_ok_and(|x| x == "1") { src.replace("let xi = fract(ign(fc) + ", "let xi = fract(ign(floor(fc / 8.0) * 8.0) + ") } else { src };
        // (and GLASSFORMS_PROBE_TILERAY=k: every pixel of a k x k tile traces the tile's own ray: the most
        // that keeping a warp's rays together could ever give)
        let src = match std::env::var("GLASSFORMS_PROBE_TILERAY").ok().and_then(|k| k.parse::<u32>().ok()) {
            Some(k) => src.replace("let fc = vec2f(pos.x, u.res.y - pos.y);", &format!("let fc = floor(vec2f(pos.x, u.res.y - pos.y) / {k}.0) * {k}.0 + 0.5;")),
            None => src,
        };
        let src = match sdf {
            Some((_, decl)) if crate::sdf::fits(&v) => crate::sdf::weave(&src, decl),
            _ if v.mat.metal && v.mat.bodies == 0 && !v.clay => crate::sdf::conductor_volume_normal(&src, self.profile),
            _ => src,
        };
        // Exact construction faces are generated after the ordinary casting pass.
        // Attach any remaining paint receivers now, including when the host itself
        // needed no cast implementation. `cast::weave` will not duplicate it.
        if v.cast && src.contains(crate::cast::PAINT_FOCUS_STUB) {
            crate::cast::weave(&src).unwrap_or(src)
        } else {
            src
        }
    }
    /// What names the formula the tracer is given, if it is given one.
    /// Did the tracer just change to one made on another thread? (Asked once: the picture starts again.)
    pub fn take_swapped(&mut self) -> bool {
        std::mem::take(&mut self.swapped)
    }
    pub fn sdf_key(&self) -> Option<&str> {
        self.sdf.as_ref().map(|s| s.0.as_str())
    }
    /// Is the shape up traced from its formula, with the tracer as it is now?
    pub fn sdf_traced(&self) -> bool {
        self.sdf.is_some() && crate::sdf::fits(&self.variant)
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

    /// Bind or clear the cached volumetric sky-space dome cubemap and companion 2D cloud shadow map.
    pub fn set_sky_space(&mut self, views: Option<(&wgpu::TextureView, &wgpu::TextureView)>) {
        self.sky_space = views.map(|(dome, shadow)| (dome.clone(), shadow.clone()));
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
    /// What the surface pass lights with (None: it is not run).
    pub fn set_surface(&mut self, lit: Option<crate::surface::Lit>) {
        self.surface.set(lit);
    }

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
        self.key_with(m, v, self.cam.wanted, self.sdf.as_ref())
    }
    /// The key of `text_with`'s tracer.
    pub fn key_with(&self, m: &March, v: Variant, lens: bool, sdf: Option<&(String, String)>) -> String {
        let sdf = sdf.filter(|_| crate::sdf::fits(&v)).map_or("", |s| s.0.as_str());
        let hero = self.hero_cur.map_or("0".into(), |h| h.key());
        format!("{:.2}|{}|{}|{}|{}|{:?}|{}|{}|{sdf}|{hero}|{}|{}", m.step_scale, m.steps_out, m.steps_in, m.layers, self.profile, v, lens, self.gem && gem_fits(&v), self.cut, self.leave_cut) + if self.gem && self.gem_carved && gem_fits(&v) { "|carved" } else { "" }
    }
    /// The tracer in use as `tracer_key` and `tracer_text` give it now: the way of marching, the way of
    /// drawing, the lens and the formula (None before there is a shape). For tests and `warm`.
    pub fn live_tracer(&self) -> Option<(March, Variant, bool, Option<(String, String)>)> {
        Some((self.march?, self.variant, self.cam.wanted, self.sdf.clone()))
    }
    /// `tracer_key` and `tracer_text` for the tracer in use (for tests: they must equal `key_with` and
    /// `text_with` given `live_tracer`).
    pub fn live_key_text(&self) -> Option<(String, String)> {
        let m = self.march?;
        Some((self.tracer_key(&m, self.variant), self.tracer_text(&m, self.variant)))
    }

    /// Make tracers beforehand, one after another on another thread, so that the first switch to one of
    /// them does not hold up a frame while it is compiled (a tracer the driver has not seen takes 0.4 to
    /// 1.4 s: docs/cache_bounces.md). Each is (the way of drawing, through which lens, the formula). Those
    /// made already are left out; they are taken in by `warm_upkeep`.
    pub fn warm(&mut self, wanted: &[(Variant, bool, Option<(String, String)>)]) {
        let Some(m) = self.march else { return };
        // (a batch still running is let go: what it has finished is taken in, the rest asked for again)
        self.warm_upkeep();
        self.warm_asked.clear();
        let mut jobs = Vec::new();
        for (v, lens, sdf) in wanted {
            let key = self.key_with(&m, *v, *lens, sdf.as_ref());
            if self.trace_pipes.contains_key(&key) || self.warm_asked.contains(&key) { continue; }
            self.warm_asked.push(key.clone());
            jobs.push((key, self.text_with(&m, *v, *lens, sdf.as_ref()), v.cast));
        }
        if jobs.is_empty() { return; }
        let (tx, rx) = std::sync::mpsc::channel();
        let (gpu, layout, cast) = (self.gpu.clone(), self.l_trace.clone(), self.l_cast.clone());
        let spawned = std::thread::Builder::new().name("tracers beforehand".into()).spawn(move || {
            for (key, src, c) in jobs {
                let split = src.contains(crate::hero::SPLIT_MARK);
                if tx.send((key, tracer_pipeline(&gpu, &layout, c.then_some(&cast), &src), split)).is_err() { return; }
            }
        });
        if spawned.is_ok() { self.warming = Some(rx); }
    }
    /// Take in the tracers made beforehand that are ready. The number still on their way.
    pub fn warm_upkeep(&mut self) -> usize {
        if let Some(rx) = &self.warming {
            while let Ok((key, pipe, split)) = rx.try_recv() {
                self.warm_asked.retain(|k| *k != key);
                if split { self.split_keys.insert(key.clone()); }
                self.trace_pipes.entry(key).or_insert(pipe);
            }
        }
        self.warm_asked.iter().filter(|k| !self.trace_pipes.contains_key(*k)).count()
    }
    /// Is the tracer of this key made (for tests)?
    pub fn has_tracer(&self, key: &str) -> bool {
        self.trace_pipes.contains_key(key)
    }
    /// The key of the tracer in use, and the shape's formula as given to it, if any (for `warm`).
    pub fn tracer_now(&self) -> (&str, Option<&(String, String)>) {
        (&self.trace_key, self.sdf.as_ref())
    }

    /// The pipeline for the way of marching and drawing now set: made the first time, kept after.
    fn build_tracer(&mut self) {
        let Some(m) = self.march else { return };
        let key = self.tracer_key(&m, self.variant);
        // Any uncached tracer compiles asynchronously when an active pipeline already exists so slider
        // interactions keep drawing without stalling the UI thread.
        if !self.trace_pipes.contains_key(&key) && !self.wait_builds && self.trace_pipes.contains_key(&self.trace_key) {
            let _ = self.prepare(self.variant);
            return;
        }
        if !self.trace_pipes.contains_key(&key) {
            let src = self.tracer_text(&m, self.variant);
            if src.contains(crate::hero::SPLIT_MARK) { self.split_keys.insert(key.clone()); }
            let pipe = tracer_pipeline(&self.gpu, &self.l_trace, self.variant.cast.then_some(&self.l_cast), &src);
            self.trace_pipes.insert(key.clone(), pipe);
        }
        self.trace_key = key;
        self.cam.active = self.cam.wanted;
        if self.up.scale.is_some() { self.build_centred(); }
        // A tracer through a lens has passes of its own after it (folding the frame in, the final
        // picture). They are made here, with the tracer, rather than by the first frame that needs them:
        // making them takes a good part of a second, which the first frame would otherwise take too.
        if self.cam.active {
            self.cam.accum(&self.gpu, ACCUM_WGSL, &self.l_accum, ACC);
            self.cam.last(&self.gpu, POST_WGSL, &self.l_final, [LDR, HDR], false);
        }
    }

    /// The key of the tracer built with every pixel's rays at the same offset (for DLSS: dlss.rs).
    fn centred_key(&self) -> String {
        format!("{}|centred", self.trace_key)
    }

    fn build_centred_pipeline(
        gpu: &Gpu,
        l_trace: &wgpu::BindGroupLayout,
        l_cast: Option<&wgpu::BindGroupLayout>,
        src: &str,
    ) -> wgpu::RenderPipeline {
        // DLSS traces every low-resolution pixel into one colour target, without the
        // full-resolution stencil used to skip converged pixels during refinement.
        let module = gpu.shader("trace (DLSS)", src, true);
        let mut layouts = vec![l_trace];
        if let Some(cast) = l_cast { layouts.push(cast); }
        gpu.pipeline(&PipeDesc {
            label: "trace (DLSS)", module: &module, vs: "vs", fs: "fs", layouts: &layouts,
            format: HDR, blend: None, strip: false, buffers: &[],
        })
    }

    fn poll_centred(&mut self) {
        if let Some((key, rx)) = &self.building_centred {
            match rx.try_recv() {
                Ok(Ok(pipe)) => {
                    self.failed_tracers.built(key);
                    self.trace_pipes.insert(key.clone(), pipe);
                    self.building_centred = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Ok(Err(message)) => {
                    self.failed_tracers.failed(key, message, std::time::Instant::now());
                    self.building_centred = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.failed_tracers.failed(key, "The DLSS material shader compiler stopped.".into(), std::time::Instant::now());
                    self.building_centred = None;
                }
            }
        }
    }

    /// Build the tracer for DLSS that goes with the tracer in use, if it is not built (and can be).
    fn build_centred(&mut self) {
        self.poll_centred();
        let Some(m) = self.march else { return };
        let key = self.centred_key();
        let waiting = matches!(self.failed_tracers.standing(&key, std::time::Instant::now()), crate::build_retry::Standing::Waiting(_));
        if self.trace_pipes.contains_key(&key) || waiting
            || self.cam.active || self.split_keys.contains(&self.trace_key)
            || self.profile || self.variant.denoise || rough_split(&self.variant, self.profile) { return; }
        if let Some(src) = crate::dlss::centre(&self.tracer_text(&m, self.variant)) {
            if src.contains(denoise::MARK) || src.contains("struct FrOut") || src.contains(crate::hero::SPLIT_MARK) { return; }
            if !self.wait_builds {
                if self.building_centred.is_none() {
                    let (tx, rx) = std::sync::mpsc::channel();
                    let (gpu, l_trace) = (self.gpu.clone(), self.l_trace.clone());
                    let l_cast = self.variant.cast.then(|| self.l_cast.clone());
                    let spawned = std::thread::Builder::new().name("tracer-dlss".into()).spawn(move || {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            Self::build_centred_pipeline(&gpu, &l_trace, l_cast.as_ref(), &src)
                        })).map_err(|payload| {
                            let detail = payload.downcast_ref::<String>().map(String::as_str)
                                .or_else(|| payload.downcast_ref::<&str>().copied()).unwrap_or("unknown compiler error");
                            format!("The DLSS material shader could not be built. {detail}")
                        });
                        let _ = tx.send(result);
                    });
                    if spawned.is_ok() {
                        self.building_centred = Some((key, rx));
                    }
                }
                return;
            }
            let pipe = Self::build_centred_pipeline(&self.gpu, &self.l_trace, self.variant.cast.then_some(&self.l_cast), &src);
            self.trace_pipes.insert(key, pipe);
        }
    }

    /// Ask for DLSS while the view moves, at this share of the size a side (None: not).
    pub fn set_upscale(&mut self, scale: Option<f64>) {
        // (a build without DLSS has nothing to ask for)
        self.up.scale = scale.filter(|_| crate::dlss::BUILT);
        self.up.live = false;
        if scale.is_some() { self.build_centred(); }
        #[cfg(feature = "dlss")]
        if let Some((tw, th)) = self.t.as_ref().map(|t| (t.tw, t.th)) { self.up.ready(&self.gpu, tw, th, &self.u_accum); }
    }

    /// Is the tracer for this way of drawing built? If it is not, it is built on another thread (it takes
    /// about half a second, too long to hold a frame up for) and the answer is no until it is there;
    /// `set_variant` then takes no time. One is built at a time.
    pub fn prepare(&mut self, v: Variant) -> bool {
        self.try_prepare(v).unwrap_or(false)
    }

    /// A rejected pipeline stays attached to the shader key that produced it.
    /// Editing the construction or selecting another finish can still build its own pipeline.
    pub fn try_prepare(&mut self, v: Variant) -> Result<bool, &str> {
        self.poll_centred();
        // (before there is a shape there is nothing to build for: `set_scene` builds what is set by then)
        let Some(m) = self.march else { return Ok(true) };
        if let Some((key, rx)) = &self.building {
            match rx.try_recv() {
                Ok(Ok((pipe, split))) => {
                    if split { self.split_keys.insert(key.clone()); }
                    self.failed_tracers.built(key);
                    self.trace_pipes.insert(key.clone(), pipe);
                    self.building = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Ok(Err(message)) => {
                    self.failed_tracers.failed(key, message, std::time::Instant::now());
                    self.building = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.failed_tracers.failed(key, "The material shader compiler stopped.".into(), std::time::Instant::now());
                    self.building = None;
                }
            }
        }
        let key = self.tracer_key(&m, v);
        if self.trace_pipes.contains_key(&key) {
            // (a tracer made on another thread for the scene as it is: it takes over)
            if v == self.variant && self.trace_key != key {
                self.trace_key = key;
                self.swapped = true;
                if self.up.scale.is_some() { self.build_centred(); }
            }
            return Ok(true);
        }
        // (a failed tracer is tried again once its wait is up; its message shows until it builds)
        use crate::build_retry::Standing;
        let standing = self.failed_tracers.standing(&key, std::time::Instant::now());
        if self.building.is_none() && !matches!(standing, Standing::Waiting(_)) {
            let (tx, rx) = std::sync::mpsc::channel();
            let (gpu, layout, src) = (self.gpu.clone(), self.l_trace.clone(), self.tracer_text(&m, v));
            let cast = v.cast.then(|| self.l_cast.clone());
            let spawned = std::thread::Builder::new().name("tracer".into()).spawn(move || {
                let split = src.contains(crate::hero::SPLIT_MARK);
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    (tracer_pipeline(&gpu, &layout, cast.as_ref(), &src), split)
                })).map_err(|payload| {
                    let detail = payload.downcast_ref::<String>().map(String::as_str)
                        .or_else(|| payload.downcast_ref::<&str>().copied()).unwrap_or("unknown compiler error");
                    format!("The material shader could not be built. {detail}")
                });
                let _ = tx.send(result);
            });
            match spawned {
                Ok(_) => self.building = Some((key.clone(), rx)),
                // no thread to be had: build it here after all
                Err(_) => return Ok(true),
            }
        }
        match self.failed_tracers.standing(&key, std::time::Instant::now()) {
            Standing::Waiting(why) | Standing::Due(why) => Err(why),
            Standing::Clear => Ok(false),
        }
    }

    pub(crate) fn bind_tracer(&mut self) {
        // (the surface pass reads the same volume: its group is made again when next drawn)
        self.surface.unbind();
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
                wgpu::BindGroupEntry { binding: crate::clouds::DOME_BINDING, resource: wgpu::BindingResource::TextureView(self.sky_space.as_ref().map_or(&self.no_picture, |s| &s.0)) },
                wgpu::BindGroupEntry { binding: crate::clouds::SHADOW_BINDING, resource: wgpu::BindingResource::TextureView(self.sky_space.as_ref().map_or(&self.no_soft, |s| &s.1)) },
                wgpu::BindGroupEntry { binding: crate::lamps::BINDING, resource: self.u_lamps.as_entire_binding() },
                wgpu::BindGroupEntry { binding: crate::refine::PAGE_BINDING, resource: wgpu::BindingResource::TextureView(self.refine.as_ref().map_or(&self.no_page, |r| &r.0)) },
                wgpu::BindGroupEntry { binding: crate::refine::ATLAS_BINDING, resource: wgpu::BindingResource::TextureView(self.refine.as_ref().map_or(&self.no_atlas, |r| &r.1)) },
                wgpu::BindGroupEntry { binding: crate::refine::UNIFORM_BINDING, resource: self.refine.as_ref().map_or(&self.no_refine_u, |r| &r.2).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(self.bodies.as_ref().map_or(&self.no_bodies, |b| &b.0)) },
                wgpu::BindGroupEntry { binding: 13, resource: self.u_bodies.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 32, resource: self.u_inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 33, resource: self.u_inner_bodies.as_entire_binding() },
                wgpu::BindGroupEntry { binding: GEM_BINDING, resource: self.u_gem.as_entire_binding() },
                wgpu::BindGroupEntry { binding: gf_core::sdf::BINDING, resource: self.u_sdf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: gf_core::sdf::TABLE_BINDING, resource: self.s_sdf_tbl.as_entire_binding() },
                wgpu::BindGroupEntry {binding:crate::world_faces::BINDING,resource:self.world_faces.buffer.as_entire_binding()},
                wgpu::BindGroupEntry { binding: crate::model_surface::DATA_BINDING, resource: self.file_surfaces.data.buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: crate::model_surface::INSTANCES_BINDING, resource: self.file_surfaces.instances.buf.as_entire_binding() },
            ].into_iter().chain(self.world_faces.projected.iter().enumerate().map(|(i, view)| {
                wgpu::BindGroupEntry {
                    binding: crate::world_faces::PROJECTED_BINDING + i as u32,
                    resource: wgpu::BindingResource::TextureView(view.as_ref().unwrap_or(&self.no_bodies)),
                }
            })).chain(
                self.grid_stand_in.entries(gf_core::csg::GRID_BINDING_TRACER, &self.grids, &self.csg_lists, &self.csg_table, grids_fit(&self.gpu))
            ).collect::<Vec<_>>(),
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
        let trace_tex = gpu.target("trace", tw, th, HDR, &[]);
        let trace_view = view(&trace_tex);
        // (written by adaptive refining's fold as well: adapt.rs)
        let picture = |label| gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: tw, height: th, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ACC,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let accum = [picture("picture a"), picture("picture b")];
        let accum_view = [view(&accum[0]), view(&accum[1])];
        self.adapt.resize(gpu, tw, th, [&accum_view[0], &accum_view[1]], &trace_view);
        // the focus stage reads either picture and writes one of its own, which the bloom and the screen
        // read in its place (the third of each of their groups below)
        self.dof.resize(tw, th, [&accum_view[0], &accum_view[1]]);
        let focused_view = self.dof.view().expect("the focus stage has just been sized").clone();
        let source = |i: usize| if i < 2 { &accum_view[i] } else { &focused_view };
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
        let final_tex = gpu.target("screen", w, h, LDR, &[wgpu::TextureFormat::Rgba8UnormSrgb]);
        let final_view = view(&final_tex);
        self.glare.resize(tw, th, [source(0), source(1), source(2)]);
        let glare_view = self.glare.view().expect("the glare has just been sized").clone();
        let final_groups = |u: &wgpu::Buffer, bloom: &wgpu::TextureView| [0, 1, 2].map(|i| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("final"),
                layout: &self.l_final,
                entries: &[
                    wgpu::BindGroupEntry { binding: 4, resource: u.as_entire_binding() },
                    tex_entry(5, source(i)),
                    tex_entry(6, bloom),
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&self.lin) },
                ],
            })
        });
        let g_final_glare = final_groups(&self.u_final, &glare_view);
        let deep = self.deep.then(|| {
            let tex = gpu.target("screen (deep)", w, h, HDR, &[]);
            let v = view(&tex);
            (tex, v, final_groups(&self.u_final_deep, &glare_view))
        });
        self.t = Some(Targets { w, h, tw, th, trace_tex, trace_view, accum, accum_view, focused_view, g_accum, g_accum_rough, pyramid: None, g_final_glare, final_tex, final_view, deep });
        self.acc_idx = 0;
        self.shown = 0;
        self.accum_n = 0;
        self.prev = None;
        self.glare_motion = (0.0, [0.0; 2]);
        self.glare_cut = false;
        self.pyramid_interactive = false;
        #[cfg(feature = "dlss")]
        {
            self.up.ready(&self.gpu, tw, th, &self.u_accum);
            self.up.live = false;
        }
        true
    }

    /// Lazily build the legacy `bloomKind=1` pipelines and HDR mip pyramid when `look.glare.pyramid()` is requested.
    fn ensure_pyramid(&mut self) {
        let gpu = &self.gpu;
        let bl = self.bloom.get_or_insert_with(|| {
            let frag = wgpu::ShaderStages::FRAGMENT;
            let d2 = wgpu::TextureViewDimension::D2;
            let layout = gpu.bind_layout("bloom", &[uniform_entry(0, frag, false), texture_entry(1, d2, true), sampler_entry(2)]);
            let post = gpu.shader("post", POST_WGSL, true);
            let one = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add };
            let pipe = |label, fs, blend| gpu.pipeline(&PipeDesc { label, module: &post, vs: "vs", fs, layouts: &[&layout], format: HDR, blend, strip: false, buffers: &[] });
            BloomPipes {
                down: pipe("bloom down", "fs_down", None),
                up: pipe("bloom up", "fs_up", Some(wgpu::BlendState { color: one, alpha: one })),
                layout,
            }
        });
        let Some(t) = self.t.as_mut() else { return };
        if t.pyramid.as_ref().is_some_and(|p| p.g_final_deep.is_some() == t.deep.is_some()) { return; }
        let (tw, th) = (t.tw, t.th);
        let source = |i: usize| if i < 2 { &t.accum_view[i] } else { &t.focused_view };
        fn tex_entry<'a>(binding: u32, v: &'a wgpu::TextureView) -> wgpu::BindGroupEntry<'a> {
            wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(v) }
        }
        let bloom_group = |src: &wgpu::TextureView, sw: u32, sh: u32, dw: u32, dh: u32| {
            let u = gpu.uniform_buffer("bloom", std::mem::size_of::<BloomU>() as u64);
            gpu.queue.write_buffer(&u, 0, bytemuck::bytes_of(&BloomU { texel: [1.0 / sw as f32, 1.0 / sh as f32], dst_res: [dw as f32, dh as f32], gain: 1.0, pad: [0.0; 3] }));
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom"),
                layout: &bl.layout,
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
        let view = |tex: &wgpu::Texture| tex.create_view(&wgpu::TextureViewDescriptor::default());
        let texs: Vec<wgpu::Texture> = sizes.iter().map(|&(a, b)| gpu.target("bloom", a, b, HDR, &[])).collect();
        let views: Vec<wgpu::TextureView> = texs.iter().map(view).collect();
        let down0_res = ([1.0 / tw as f32, 1.0 / th as f32], [sizes[0].0 as f32, sizes[0].1 as f32]);
        let u_down0 = gpu.uniform_buffer("bloom down0", std::mem::size_of::<BloomU>() as u64);
        gpu.queue.write_buffer(&u_down0, 0, bytemuck::bytes_of(&BloomU { texel: down0_res.0, dst_res: down0_res.1, gain: 1.0, pad: [0.0; 3] }));
        let g_down0 = [0, 1, 2].map(|i| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom"),
                layout: &bl.layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: u_down0.as_entire_binding() }, tex_entry(1, source(i)), wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.lin) }],
            })
        });
        let mut mips = Vec::with_capacity(sizes.len());
        for (i, &(a, b)) in sizes.iter().enumerate() {
            // down: read the level above into this one (level 0 reads the picture itself: see g_down0)
            let down = if i == 0 { g_down0[0].clone() } else { bloom_group(&views[i - 1], sizes[i - 1].0, sizes[i - 1].1, a, b) };
            // up: read this level while drawing onto the next larger one
            let up = (i > 0).then(|| bloom_group(&views[i], a, b, sizes[i - 1].0, sizes[i - 1].1));
            mips.push(Mip { tex: texs[i].clone(), view: views[i].clone(), w: a, h: b, down, up });
        }
        let final_groups = |u: &wgpu::Buffer| [0, 1, 2].map(|i| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("final"),
                layout: &self.l_final,
                entries: &[
                    wgpu::BindGroupEntry { binding: 4, resource: u.as_entire_binding() },
                    tex_entry(5, source(i)),
                    tex_entry(6, &views[0]),
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&self.lin) },
                ],
            })
        });
        let g_final = final_groups(&self.u_final);
        let g_final_deep = t.deep.is_some().then(|| final_groups(&self.u_final_deep));
        t.pyramid = Some(Pyramid { u_down0, down0_res, last_knee: 0.0, g_down0, mips, g_final, g_final_deep });
    }

    /// The picture on screen, as a texture whose bytes are already display-ready (sRGB-encoded).
    pub fn screen(&self) -> Option<&wgpu::Texture> {
        self.t.as_ref().map(|t| &t.final_tex)
    }

    /// What the still pass `ahead` passes from now costs, counted in passes of the tracer in use while
    /// settling: with one wavelength inside while settling (hero.rs `settling`, clamp 5), every `every`-th
    /// pass traces the whole comb, about nw / k times as much (an upper bound: the reflection is shared).
    /// 1 otherwise. (For the refining budget: `Engine::frame_rest`.)
    pub fn pass_cost(&self, ahead: u32, nw: i32, dispersive: bool) -> f64 {
        let h = self.hero;
        // (only where the hero is woven in: plain glass with dispersion, the tracer in use)
        let woven = dispersive && self.variant.mat == Default::default() && !self.variant.clay && !self.profile && self.hero_cur == Some(h);
        if woven && h.for_frame(false) && h.clamp == 5 && self.frame_no.wrapping_add(ahead) % 4096 % u32::from(h.every.max(2)) == 0 {
            (nw.max(1) as f64 / f64::from(h.k.max(1))).max(1.0)
        } else {
            1.0
        }
    }

    /// Moving frames can be traced smaller than the picture (`dyn_scale`): the antialiasing that follows
    /// motion can read its traced region. Rough glass and direct surface lighting follow that region too.
    pub fn dyn_ok(&self) -> bool {
        !self.variant.denoise && !self.split_keys.contains(&self.trace_key) && !self.cam.active && !self.profile && self.hero_move.is_none() && self.up.scale.is_none()
    }

    pub(crate) fn can_split_refine(&self) -> bool {
        self.variant.mat.bodies > 0 && rough_split(&self.variant, self.profile)
            && self.dyn_ok() && !self.surface.on() && self.size().1.0 >= 2
    }

    fn dyn_pipeline(gpu: &Gpu, layout: &wgpu::BindGroupLayout) -> wgpu::RenderPipeline {
        let src = accum_dyn(ACCUM_WGSL).expect("accum.wgsl has the lines the smaller frame's antialiasing changes");
        let module = gpu.shader("accumulate (traced smaller)", &src, false);
        gpu.pipeline(&PipeDesc { label: "accumulate (traced smaller)", module: &module, vs: "vs", fs: "fs", layouts: &[layout], format: ACC, blend: None, strip: false, buffers: &[] })
    }

    /// Prepare reduced-resolution accumulation synchronously for callers that require it immediately.
    pub fn ready_dyn(&mut self) {
        if let Some(rx) = self.building_dyn.take() { self.p_accum_dyn = rx.recv().ok(); }
        self.p_accum_dyn.get_or_insert_with(|| Self::dyn_pipeline(&self.gpu, &self.l_accum));
    }

    /// A live frame keeps drawing at full size while its smaller-frame pipeline is compiled.
    fn request_dyn(&mut self) {
        if self.p_accum_dyn.is_some() { return; }
        if let Some(rx) = &self.building_dyn {
            match rx.try_recv() {
                Ok(pipeline) => { self.p_accum_dyn = Some(pipeline); self.building_dyn = None; }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.ready_dyn(),
            }
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let (gpu, layout) = (self.gpu.clone(), self.l_accum.clone());
        let spawned = std::thread::Builder::new().name("moving frame accumulation".into()).spawn(move || {
            let _ = tx.send(Self::dyn_pipeline(&gpu, &layout));
        });
        if spawned.is_ok() { self.building_dyn = Some(rx); } else { self.ready_dyn(); }
    }

    /// Trace one frame and fold it into the picture so far.
    ///   moving: something moved since the last frame, so the picture so far is out of date
    ///   cut: something changed outright, so the picture so far is useless
    ///   skip_post: leave out the bloom and the screen (more frames are about to be averaged in)
    pub fn frame(&mut self, tu: &TraceU, look: &Look, moving: bool, cut: bool, skip_post: bool) {
        if let Some(s) = self.matcap_on.clone() { self.frame_matcap(tu, &s, moving || cut, skip_post); return; }
        let split = std::mem::take(&mut self.split_refine) && !moving && !cut && !skip_post && self.can_split_refine();
        // (which half this is: kept only once the frame is sure to be traced, below)
        let half = self.traced_half.after(split);
        // one wavelength inside the glass, for this kind of frame (hero.rs): the tracer with it or without
        // (while moving, the split hero where the engine asks for it: three wavelengths and dispersion only)
        let moving_taa = moving && !cut && look.smooth > 0.0;
        let hero = match self.hero_move {
            Some(h) if (moving_taa || hero_move_always()) && tu.nw == 3 && tu.disp != 0.0 && !self.profile => Some(h),
            _ => self.hero.for_frame(moving_taa).then_some(self.hero),
        };
        if self.hero.taa > 0 && !self.accum_chroma {
            if let Some(src) = crate::hero::chroma_taa(ACCUM_WGSL) {
                let module = self.gpu.shader("accumulate (colour apart)", &src, false);
                self.p_accum = self.gpu.pipeline(&PipeDesc { label: "accumulate", module: &module, vs: "vs", fs: "fs", layouts: &[&self.l_accum], format: ACC, blend: None, strip: false, buffers: &[] });
            }
            self.accum_chroma = true;
        }
        if hero != self.hero_cur {
            self.hero_cur = hero;
            self.build_tracer();
        }
        if moving_taa && self.dyn_scale < 0.999 && self.dyn_ok() { self.request_dyn(); }
        let Some(t) = self.t.as_ref() else { return };
        if self.g_trace.is_none() { return; }
        if self.variant.cast && self.g_cast.is_none() { return; }
        self.traced_half = half;
        // How to treat the picture so far.
        let taa = moving && !cut && look.smooth > 0.0;
        let weight: f64;
        if cut || (moving && !taa) {
            self.accum_n = 0;
            weight = 1.0;
            // (and adaptive refining's counts start afresh: carried on, the pixels it held certain would be
            // left out of the trace and keep what they showed before, which a cut makes useless: an object
            // taken away, or a new one made, left its old picture where nothing is now)
            if cut {
                self.adapt.valid = false;
                self.glare.reset();
                self.pyramid_interactive = false;
            }
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
                // Align heroClamp=5 so the 6 signed passes precede the 7x clamp-compensating heroFull pass:
                // a 1-frame mouse-drag pause never fires heroFull at w=0.5, and pass 7 cancels the 6 signed
                // passes at equal weight in the running mean.
                if let Some(h) = hero.filter(|h| h.clamp == 5) {
                    let e = u32::from(h.every.max(2));
                    self.frame_no = self.frame_no.wrapping_add((e + 1 - self.frame_no % e) % e);
                }
            }
            weight = 1.0 / (self.accum_n as f64 + 1.0);
        }
        self.was_taa = taa;
        if half != Half::Second { self.jitter_n = (self.jitter_n % 4096) + 1; }
        // always jitter unless antialiasing is off and the view is moving
        let jit = weight < 1.0 || look.smooth > 0.0;
        let mut u = *tu;
        u.res = [t.tw as f32, t.th as f32];
        u.frame = self.frame_no as i32;
        // Rotation uses xyz. The rough floor uses this spare component to stratify moving samples.
        u.env[0][3] = if taa { 1.0 } else { 0.0 };
        u.jitter = if jit { [(halton(self.jitter_n, 2) - 0.5) as f32, (halton(self.jitter_n, 3) - 0.5) as f32] } else { [0.0, 0.0] };
        // (keeping the frame rate: a moving frame traced smaller, into the target's corner, when asked for
        // and where the antialiasing reads the traced frame straight; the picture stays full size)
        let dyn_s = if taa && self.dyn_scale < 0.999 && self.p_accum_dyn.is_some() && self.dyn_ok() { self.dyn_scale.max(0.25) } else { 1.0 };
        let region = (dyn_s < 1.0).then(|| dyn_region(t.tw, t.th, dyn_s));
        let full = u;
        if let Some((rw, rh)) = region {
            u.res = [rw as f32, rh as f32];
            if self.variant.mat.bodies > 0 && rough_split(&self.variant, self.profile) && u.ns == 1 && u.lens <= 0.0 {
                // At half scale these four offsets trace each native pixel once per four frames.
                // Use a common ray offset across the frame so reconstruction knows where each ray landed.
                const OFFSETS: [[f32; 2]; 4] = [[-0.25, 0.25], [0.25, -0.25], [0.25, 0.25], [-0.25, -0.25]];
                u.jitter = OFFSETS[self.frame_no as usize % OFFSETS.len()];
                u.pad[1] = 1;
            }
        }
        self.dyn_used = dyn_s;
        self.traced_caps = [u.nb, u.nw];
        self.gpu.queue.write_buffer(&self.u_trace, 0, bytemuck::bytes_of(&u));

        // The Ray Reconstruction draft, when the tier row asks for it (dlss_rr.rs)
        #[cfg(feature = "dlss")]
        if self.draft.want.on {
            if self.frame_draft(tu, look, skip_post, cut) {
                self.traced_half = Half::Whole;
                return;
            }
            if let Some(d) = self.draft.ctx.as_mut() { d.live = false; }
            self.draft.fell += 1;
            if std::env::var_os("GLASSFORMS_DLSS_SAY").is_some() && self.draft.fell % 50 == 1 { eprintln!("dlss: a frame the draft did not draw while it is asked for (#{}, draft frames so far {}, ready {})", self.draft.fell, self.draft.drawn, self.draft_ready()); }
        }
        // DLSS in place of the antialiasing that follows motion, when asked for (dlss.rs)
        #[cfg(feature = "dlss")]
        {
            // (GLASSFORMS_DLSS_FORCE=1: a still picture too, after its first frame: for checking DLSS's inputs)
            let forced = !cut && self.accum_n > 0 && self.up.scale.is_some() && std::env::var_os("GLASSFORMS_DLSS_FORCE").is_some();
            if (taa || forced) && self.frame_dlss(tu, look, skip_post) {
                // (DLSS drew the whole frame)
                self.traced_half = Half::Whole;
                return;
            }
            self.up.live = false;
        }
        let Some(t) = self.t.as_ref() else { return };
        let Some(g_trace) = self.g_trace.as_ref() else { return };
        // (the antialiasing works at the picture's own size; told the share the frame was traced at)
        let (now, mut au) = self.accum_u(&full, weight, taa, look);
        self.glare_cut = cut;
        self.glare_motion = match self.prev.filter(|_| !cut) {
            Some(was) => Self::view_motion(&now, &was, t.tw, t.th),
            None => (0.0, [0.0; 2]),
        };
        if region.is_some() {
            au.pad = dyn_s as f32;
            if u.pad[1] != 0 {
                au.obj[0][3] = u.jitter[0];
                au.obj[1][3] = u.jitter[1];
                au.obj[2][3] = 1.0;
            }
        }
        self.gpu.queue.write_buffer(&self.u_accum, 0, bytemuck::bytes_of(&au));

        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        // (the tracer of a rough clear body writes a blur radius beside its frame, and the frame is
        // blurred by it before it is folded in)
        let rough = rough_split(&self.variant, self.profile);
        let second = self.g_cast.as_ref().filter(|_| self.variant.cast);
        // (a denoised tracer writes the light it found inside the body to one more target)
        let dn = if self.variant.denoise && !self.profile { self.dn.frame_view(&self.lin).cloned() } else { None };
        // Adaptive refining (adapt.rs): a frame that is not folded in by the antialiasing that follows
        // motion, and whose trace goes straight into the picture, is folded in with each pixel's own count;
        // while refining, the pixels whose mean is certain are left out of the trace. (The counts start
        // afresh, from the frames the picture now stands for, unless the last frame was folded in so too.)
        let folded = !taa && !rough && dn.is_none() && !self.profile;
        let masked = folded && self.adapt.begin(&self.gpu, &mut enc, (!self.adapt.valid).then_some(self.accum_n), look.exposure);
        let (stencil, clear) = (self.adapt.stencil(), if masked { crate::adapt::NOT_TRACED } else { wgpu::Color::TRANSPARENT });
        let pipe = &self.trace_pipes[&self.trace_key];
        // (once a sample: a first half does not advance frame_no, so it would be counted twice)
        let measured = self.pass_times.is_some() && self.frame_no % 16 == 0 && half != Half::First;
        if measured { if let Some(log) = &self.pass_times { log.mark(&mut enc, 0); } }
        let mut frosted = false;
        match self.frost.aux().filter(|_| rough) {
            Some(aux) => {
                let views: Vec<&wgpu::TextureView> = [&t.trace_view, aux[0], aux[1]].into_iter().chain(dn.as_ref()).collect();
                // Spread expensive material paths across both visible updates. Integer endpoints
                // cover an odd-sized target exactly and leave the sample position unchanged.
                let pairs = (t.tw / 32).clamp(1, 12);
                let count = if half == Half::Whole { 0 } else { pairs };
                let columns = (0..count).map(|pair| {
                    let band = 2 * pair + u32::from(half == Half::Second);
                    let start = t.tw * band / (2 * pairs);
                    let end = t.tw * (band + 1) / (2 * pairs);
                    (start, end - start, t.th)
                });
                crate::adapt::trace_regions(&mut enc, &views, clear, stencil, masked, pipe, g_trace, second, region, columns);
                frosted = true;
            }
            None if dn.is_some() => {
                crate::adapt::trace_pass(&mut enc, &[&t.trace_view, dn.as_ref().expect("just asked")], clear, stencil, masked, pipe, g_trace, second);
            }
            // (the split hero: into its two targets, then put back together into the frame)
            None if self.split_keys.contains(&self.trace_key) => {
                let (tw, th) = (t.tw, t.th);
                let recon = self.recon.get_or_insert_with(|| crate::hero::Recon::new(&self.gpu));
                let views = recon.targets(&self.gpu, tw, th);
                crate::adapt::trace_pass(&mut enc, &views, clear, stencil, masked, pipe, g_trace, second);
                recon.encode(&mut enc, &t.trace_view);
            }
            // (GLASSFORMS_SLOW=k: the pass is traced k times over, the same each time, to stand in for a GPU k
            // times slower when trying out the frame-rate keeping: autoscale.rs; never set otherwise)
            None => for _ in 0..crate::autoscale::slow() { crate::adapt::trace_pass_in(&mut enc, &[&t.trace_view], clear, stencil, masked, pipe, g_trace, second, region) },
        }
        if measured { if let Some(log) = &self.pass_times { log.mark(&mut enc, 1); } }
        // direct light with shadows on an opaque or milky body, onto the traced frame before anything reads
        // it (not through another lens than the plain one: the pass finds the surface along a plain ray)
        if self.surface.on() && !self.cam.active && !self.profile {
            if let Some((_, vol)) = &self.scene {
                let hb = self.variant.grown.then(|| f32::from_bits(u.pad[2] as u32));
                self.surface.encode(&self.gpu, &mut enc, &t.trace_tex, &t.trace_view, vol, &self.lin, &self.u_lamps, &u, hb, self.variant.near);
            }
        }
        if measured { if let Some(log) = &self.pass_times { log.mark(&mut enc, 2); } }
        if frosted { self.frost.encode(&mut enc, u.tan_fov, region); }
        if measured { if let Some(log) = &self.pass_times { log.mark(&mut enc, 3); } }

        if folded {
            self.adapt.fold(&mut enc, self.acc_idx);
        } else {
            self.adapt.valid = false;
            let mut rp = pass(&mut enc, &t.accum_view[1 - self.acc_idx], None, true);

            // (through any lens but the plain one, the surface is followed by the camera's own arithmetic)
            rp.set_pipeline(if self.cam.active { self.cam.accum(&self.gpu, ACCUM_WGSL, &self.l_accum, ACC) } else if let (Some(_), Some(p)) = (region, &self.p_accum_dyn) { p } else { &self.p_accum });
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
        // (a pass for a share of the pixels only is not counted as a frame of the picture)
        if half != Half::First {
            if !(masked && self.adapt.share < 1.0) { self.accum_n += 1; }
            self.frame_no += 1;
        }
        if measured { if let Some(log) = &self.pass_times { log.mark(&mut enc, 4); } }
        if !skip_post { self.encode_post(&mut enc, &full, look, !moving && !cut); }
        if measured { if let Some(log) = &self.pass_times { log.mark(&mut enc, 5); } }
        self.gpu.queue.submit([enc.finish()]);
        // The first half is visible immediately. Keep the old mean as the source until the
        // second half completes the same sample and its full-image frost filter.
        self.trace_passes += 1;
        if half == Half::First { self.acc_idx = 1 - self.acc_idx; }
        if measured { if let Some(log) = &mut self.pass_times { log.record(&self.gpu, self.frame_no, moving, dyn_s); } }
    }

    /// A frame of the matcap view (matcap.rs): the first surface along each ray shaded by the matcap, into
    /// the picture so far (as a running mean of jittered frames while nothing moves), then onto the screen.
    /// `afresh`: the view moved or changed, the picture starts again.
    fn frame_matcap(&mut self, tu: &TraceU, s: &gf_core::view_mode::Matcap, afresh: bool, skip_post: bool) {
        let Some(m) = self.march else { return };
        let (Some(t), Some(g_trace)) = (self.t.as_ref(), self.g_trace.as_ref()) else { return };
        if afresh { self.accum_n = 0; }
        let n = self.accum_n;
        self.jitter_n = (self.jitter_n % 4096) + 1;
        let mut u = *tu;
        u.res = [t.tw as f32, t.th as f32];
        u.frame = self.frame_no as i32;
        u.jitter = if n > 0 { [(halton(self.jitter_n, 2) - 0.5) as f32, (halton(self.jitter_n, 3) - 0.5) as f32] } else { [0.0; 2] };
        self.gpu.queue.write_buffer(&self.u_trace, 0, bytemuck::bytes_of(&u));
        let grown = self.matcap_grown;
        let fits = crate::sdf::fits(&Variant { grown, mat: self.variant.mat, ..Default::default() });
        let reading = crate::matcap::Reading { grown, sdf: self.sdf.clone().filter(|_| fits) };
        let bodies = self.bodies.as_ref().map_or(0, |b| b.1);
        let picture = match &s.source {
            gf_core::view_mode::MatcapSource::Image(p) => self.matcap.picture.as_deref() == Some(p.as_str()),
            gf_core::view_mode::MatcapSource::Material => self.matcap.picture.as_deref().is_some_and(|p| p.starts_with(crate::matcap::MATERIAL)),
            gf_core::view_mode::MatcapSource::Builtin(_) => false,
        };
        self.matcap.write(&self.gpu, &crate::matcap::McU::of(s, picture, 1.0 / (n as f32 + 1.0), bodies, tu.cam[0]));
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame (matcap)") });
        let measured = self.pass_times.is_some() && self.frame_no % 16 == 0;
        if measured { if let Some(log) = &self.pass_times { log.mark(&mut enc, 0); } }
        {
            let pipe = self.matcap.pipeline(&self.gpu, &self.l_trace, &m, &reading).clone();
            let g = self.matcap.group(&self.gpu, &t.accum_view[self.acc_idx]);
            let mut rp = pass(&mut enc, &t.accum_view[1 - self.acc_idx], None, true);
            rp.set_pipeline(&pipe);
            rp.set_bind_group(0, g_trace, &[]);
            rp.set_bind_group(1, &g, &[]);
            rp.draw(0..3, 0..1);
        }
        if measured { if let Some(log) = &self.pass_times { for k in 1..=4 { log.mark(&mut enc, k); } } }
        self.acc_idx = 1 - self.acc_idx;
        self.shown = self.acc_idx;
        self.focused = false;
        self.prev = None;
        self.adapt.valid = false;
        self.dn.live = false;
        self.accum_n += 1;
        self.frame_no += 1;
        self.trace_passes += 1;
        if !skip_post { self.show_matcap(&mut enc); }
        if measured { if let Some(log) = &self.pass_times { log.mark(&mut enc, 5); } }
        self.gpu.queue.submit([enc.finish()]);
        if measured { if let Some(log) = &mut self.pass_times { log.record(&self.gpu, self.frame_no, afresh, 1.0); } }
    }

    fn show_matcap(&self, enc: &mut wgpu::CommandEncoder) {
        let Some(t) = self.t.as_ref() else { return };
        self.matcap.encode_show(&self.gpu, enc, &t.accum_view[self.shown], (&t.final_view, t.w, t.h), t.deep.as_ref().map(|d| &d.1));
    }

    /// What the antialiasing is told for a frame traced with these numbers, and what it keeps of the view.
    fn accum_u(&self, u: &TraceU, weight: f64, taa: bool, look: &Look) -> (PrevView, AccumU) {
        // (a studio's backdrop stands in the room, not in the lights' frame: where one is shown, it is the
        // room that says where a piece of the background was at the previous frame)
        let seen = self.variant.stage;
        let room_sky = seen.backdrop || seen.sky || seen.atmo;
        let now = PrevView { shape_no: look.shape_no, obj: u.obj, env: if room_sky { self.stage.room } else { u.env }, room: self.stage.room, floor: self.stage.mid[3], cam: [u.dist, u.tan_fov], eye: [u.eye[0], u.eye[1]], shift: u.shift, lens: u.cam };
        let was = self.prev.unwrap_or(now);
        let mut au = AccumU {
            res: u.res, cam: now.cam, cam_p: was.cam, shift: now.shift, shift_p: was.shift, w: weight as f32, moving: if taa { 1.0 } else { 0.0 },
            obj: now.obj, objp: was.obj, debug: look.debug, spread: look.spread, sky: if self.variant.backdrop || room_sky { 1.0 } else { 0.0 }, pad: 0.0,
            env: now.env, envp: was.env,
            room: now.room, roomp: was.room, floor: [if seen.floor { 1.0 } else { 0.0 }, now.floor, was.floor, self.stage.tint[3]],
            eye: [now.eye[0], now.eye[1], was.eye[0], was.eye[1]],
            lens: now.lens, lens_p: was.lens,
        };
        // The fourth lane is free in the accumulator: previous light rotation uses xyz.
        au.envp[2][3] = (now.shape_no != was.shape_no) as u8 as f32;
        (now, au)
    }

    fn view_motion(now: &PrevView, was: &PrevView, tw: u32, th: u32) -> (f32, [f32; 2]) {
        let (tw_f, th_f) = (tw as f32, th as f32);
        let center_fc = |v: &PrevView| -> [f32; 2] {
            let denom = (v.cam[0] * v.cam[1]).max(1e-4);
            let uv_x = -v.eye[0] / denom - v.shift[0];
            let uv_y = -v.eye[1] / denom - v.shift[1];
            [(uv_x * th_f + tw_f) * 0.5, (uv_y * th_f + th_f) * 0.5]
        };
        let c_now = center_fc(now);
        let c_was = center_fc(was);
        let shift_px = [c_was[0] - c_now[0], c_now[1] - c_was[1]];
        let row_diff = |a: &[[f32; 4]; 3], b: &[[f32; 4]; 3]| -> f32 {
            (0..3).map(|r| {
                let dx = a[r][0] - b[r][0];
                let dy = a[r][1] - b[r][1];
                let dz = a[r][2] - b[r][2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            }).sum()
        };
        let rot = row_diff(&now.obj, &was.obj).max(row_diff(&now.env, &was.env));
        let scale_now = (now.cam[0] * now.cam[1]).max(0.05);
        let scale_was = (was.cam[0] * was.cam[1]).max(0.05);
        let rot_px = 0.45 * rot * (0.55 / scale_now) * (th_f * 0.5);
        let zoom_px = ((scale_now - scale_was).abs() / scale_now) * (th_f * 0.25)
            + (now.lens[0][3] - was.lens[0][3]).abs() * (th_f * 0.2);
        let morph_px = if now.shape_no != was.shape_no { th_f * (1.2 / 720.0) } else { 0.0 };
        (rot_px.max(zoom_px).max(morph_px), shift_px)
    }

    /// The tier row's wish for this frame. Making the context and the tracer takes a moment (NVIDIA's runtime loads: 0.9 s the first
    /// time), so it is done when the draft is selected or Auto may use it, not on the first moving frame.
    #[cfg(feature = "dlss")]
    pub fn set_draft(&mut self, want: DraftWant) {
        self.draft.want = want;
        if want.prepare || want.on { self.prepare_draft(); }
    }

    /// Is the draft ready to draw (context made for this size, tracer built)?
    #[cfg(feature = "dlss")]
    pub fn draft_ready(&self) -> bool {
        let Some((tw, th)) = self.t.as_ref().map(|t| (t.tw, t.th)) else { return false };
        self.draft.failed.is_none()
            && self.draft.ctx.as_ref().is_some_and(|d| d.is_ready(tw, th, self.draft.want.scale))
            && self.draft.pipes.contains_key(&self.draft_key())
    }

    #[cfg(feature = "dlss")]
    fn draft_key(&self) -> String { format!("{}|draft", self.trace_key) }

    #[cfg(feature = "dlss")]
    fn draft_fail(&mut self, why: String) {
        if self.draft.failed.is_none() { eprintln!("dlss: {why}; the draft tier is off"); }
        self.draft.failed = Some(why);
        self.draft.ctx = None;
        self.draft.want.on = false;
    }

    /// Why the draft cannot draw this scene just now, in words for the row (None: it can). The full path draws instead.
    #[cfg(feature = "dlss")]
    pub fn draft_blocked(&self) -> Option<String> {
        if self.surface.on() { return Some("the surface studio is on".into()); }
        if self.cam.active { return Some("a camera lens is on".into()); }
        if self.variant.denoise { return Some("this material is denoised".into()); }
        if rough_split(&self.variant, self.profile) { return Some("rough glass is drawn in two stages".into()); }
        if self.profile || self.split_keys.contains(&self.trace_key) { return Some("this tracer is drawn in halves".into()); }
        if self.variant.cast && self.g_cast.is_none() { return Some("the shadows are not ready".into()); }
        let key = self.draft_key();
        if self.draft.ctx.is_some() && self.march.is_some() && self.draft.building.is_none() && !self.draft.pipes.contains_key(&key) {
            return Some("its tracer cannot be made for this material".into());
        }
        None
    }

    /// RR forgets its history at the next draft frame (a new scene).
    #[cfg(feature = "dlss")]
    pub fn draft_reset(&mut self) {
        if let Some(d) = self.draft.ctx.as_mut() { d.live = false; }
    }

    /// What the window knows of what is dragged: it moved by this much (scene units) since the frame before. Zero: not known.
    #[cfg(feature = "dlss")]
    pub fn draft_drag(&mut self, by: [f32; 3]) {
        if let Some(d) = self.draft.ctx.as_mut() { d.drag = by; }
    }

    #[cfg(feature = "dlss")]
    fn poll_draft(&mut self) {
        let Some((key, rx)) = &self.draft.building else { return };
        let key = key.clone();
        match rx.try_recv() {
            Ok(Ok((colour, guide))) => {
                self.failed_tracers.built(&key);
                if self.draft.pipes.len() >= 6 { self.draft.pipes.clear(); }
                self.draft.pipes.insert(key, (colour, guide));
                self.draft.building = None;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Ok(Err(message)) => {
                self.failed_tracers.failed(&key, message, std::time::Instant::now());
                self.draft.building = None;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.failed_tracers.failed(&key, "The draft tracer's compiler stopped.".into(), std::time::Instant::now());
                self.draft.building = None;
            }
        }
    }

    /// Make the draft's context for this size (probing NVIDIA's runtime the first time) and build its tracer, if not done.
    #[cfg(feature = "dlss")]
    fn prepare_draft(&mut self) {
        if self.draft.failed.is_some() { return; }
        let Some((tw, th)) = self.t.as_ref().map(|t| (t.tw, t.th)) else { return };
        let scale = self.draft.want.scale;
        let made = match self.draft.ctx.as_mut() {
            Some(d) => d.ready(&self.gpu, tw, th, scale),
            None => crate::dlss_rr::probe(&self.gpu, tw, th, scale).map(|d| self.draft.ctx = Some(d)),
        };
        if let Err(e) = made { self.draft_fail(e); return; }
        if std::env::var_os("GLASSFORMS_DLSS_SAY").is_some() && !self.draft.said { self.draft.said = true; eprintln!("dlss: the draft's context is made ({tw}x{th}, share {scale})"); }
        self.poll_draft();
        let Some(m) = self.march else { return };
        let key = self.draft_key();
        let waiting = matches!(self.failed_tracers.standing(&key, std::time::Instant::now()), crate::build_retry::Standing::Waiting(_));
        if self.draft.pipes.contains_key(&key) || waiting
            || self.cam.active || self.split_keys.contains(&self.trace_key)
            || self.profile || self.variant.denoise || rough_split(&self.variant, self.profile) { return; }
        let Some(src) = crate::dlss::centre(&self.tracer_text(&m, self.variant)) else { return };
        if src.contains(denoise::MARK) || src.contains("struct FrOut") || src.contains(crate::hero::SPLIT_MARK) { return; }
        let src = crate::dlss_rr::with_guides(&src);
        if self.wait_builds {
            let (c, g) = crate::dlss_rr::build_tracer_pipelines(&self.gpu, &self.l_trace, self.variant.cast.then_some(&self.l_cast), &src);
            self.draft.pipes.insert(key, (c, g));
        } else if self.draft.building.is_none() {
            let (tx, rx) = std::sync::mpsc::channel();
            let (gpu, l_trace) = (self.gpu.clone(), self.l_trace.clone());
            let l_cast = self.variant.cast.then(|| self.l_cast.clone());
            let spawned = std::thread::Builder::new().name("tracer-draft".into()).spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| crate::dlss_rr::build_tracer_pipelines(&gpu, &l_trace, l_cast.as_ref(), &src)))
                    .map_err(|payload| {
                        let detail = payload.downcast_ref::<String>().map(String::as_str).or_else(|| payload.downcast_ref::<&str>().copied()).unwrap_or("unknown compiler error");
                        format!("The draft tracer could not be built. {detail}")
                    });
                let _ = tx.send(result);
            });
            if spawned.is_ok() { self.draft.building = Some((key, rx)); }
        }
    }

    /// Draw this frame through the Ray Reconstruction draft: trace a quarter (or a ninth) of the pixels with every pixel's ray at the frame's
    /// offset and at most `bounces` bounces, make the guides, let RR make the full-size picture so far, and put it in. False (nothing done)
    /// when it cannot be done this frame; the full path then draws it.
    #[cfg(feature = "dlss")]
    fn frame_draft(&mut self, tu: &TraceU, look: &Look, skip_post: bool, cut: bool) -> bool {
        if self.cam.active || self.profile || self.split_keys.contains(&self.trace_key) || self.surface.on() { return false; }
        if rough_split(&self.variant, self.profile) || self.variant.denoise { return false; }
        if self.variant.cast && self.g_cast.is_none() { return false; }
        if !self.draft_ready() { self.prepare_draft(); }
        if !self.draft_ready() {
            self.draft.misses += 1;
            if std::env::var_os("GLASSFORMS_DLSS_SAY").is_some() && (!self.draft.said_frame || self.draft.misses % 20 == 1) { {
                let (tw, th) = self.t.as_ref().map(|t| (t.tw, t.th)).unwrap_or((0, 0));
                let key = self.draft_key();
                eprintln!("dlss: the draft is not ready (context {}, sized {}, tracers {}, its key matches {}, building {}, picture {tw}x{th}, share {})", self.draft.ctx.is_some(),
                    self.draft.ctx.as_ref().is_some_and(|d| d.is_ready(tw, th, self.draft.want.scale)), self.draft.pipes.len(), self.draft.pipes.contains_key(&key), self.draft.building.is_some(), self.draft.want.scale);
                eprintln!("dlss: blocked: {:?}; key length {}, tracers {:?}, misses {}", self.draft_blocked(), key.len(), self.draft.pipes.keys().map(|k| k.len()).collect::<Vec<_>>(), self.draft.misses);
            } }
            return false;
        }
        let Some((tw, th)) = self.t.as_ref().map(|t| (t.tw, t.th)) else { return false };
        let want = self.draft.want;
        let key = self.draft_key();
        let Some((colour, guide)) = self.draft.pipes.get(&key).map(|(c, g)| (c.clone(), g.clone())) else { return false };
        let Some(mut d) = self.draft.ctx.take() else { return false };
        // (a `cut` while the draft draws is a shape being dragged or edited: RR keeps its history, the motion vectors say what moved; a new
        // scene is the window's business: `Renderer::draft_reset`)
        let _ = cut;
        let done = (|| {
            let f = d.begin(tw)?;
            let mut u = *tu;
            u.res = [f.res[0] as f32, f.res[1] as f32];
            u.frame = self.frame_no as i32;
            // the tracer's offsets count up from the bottom; the draft's down from the top
            u.jitter = [f.jitter[0], -f.jitter[1]];
            u.nb = u.nb.min(want.bounces.max(1));
            // the plain three-wavelength sample (one wavelength a sample is coloured noise RR cannot use: lab_rr.md)
            u.nw = 3;
            if std::env::var_os("GLASSFORMS_DLSS_SAY").is_some() && self.draft.drawn == 0 { eprintln!("dlss: the app's frame asked for nw {} ns {} nb {}; the draft draws nw {} ns {} nb {}; look exposure {}", tu.nw, tu.ns, tu.nb, u.nw, u.ns, u.nb, look.exposure); }
            let (now, au) = self.accum_u(&u, 1.0, true, look);
            let prev_obj = self.prev.map(|p| p.obj).unwrap_or(u.obj);
            let (t, g_trace) = (self.t.as_ref()?, self.g_trace.as_ref()?);
            // (what the passes read: written before the submit below)
            self.gpu.queue.write_buffer(&self.u_trace, 0, bytemuck::bytes_of(&u));
            let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame (draft)") });
            let g_cast = self.g_cast.as_ref().filter(|_| self.variant.cast);
            d.encode_inputs(&self.gpu, &mut enc, &colour, &guide, g_trace, g_cast, &u, prev_obj, want.motion)?;
            let target = t.accum_view[1 - self.acc_idx].clone();
            let cb = match d.evaluate(&self.gpu, &mut enc, &f, &u) {
                Ok(cb) => cb,
                Err(e) => return Some(Err(e)),
            };
            self.glare_cut = false;
            self.glare_motion = match self.prev {
                Some(was) => Self::view_motion(&now, &was, tw, th),
                None => (0.0, [0.0; 2]),
            };
            self.gpu.queue.write_buffer(&self.u_accum, 0, bytemuck::bytes_of(&au));
            let mut after = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame (after draft)") });
            d.encode_resolve(&mut after, &target);
            self.adapt.valid = false;
            self.dn.live = false;
            d.live = true;
            d.drawn += 1;
            self.prev = Some(now);
            self.acc_idx = 1 - self.acc_idx;
            self.accum_n += 1;
            self.frame_no += 1;
            let mut full = *tu;
            full.res = [tw as f32, th as f32];
            if !skip_post { self.encode_post(&mut after, &full, look, false); }
            self.gpu.queue.submit([enc.finish(), cb, after.finish()]);
            self.trace_passes += 1;
            if let Some(dir) = std::env::var_os("GLASSFORMS_DRAFT_DUMP") {
                if d.drawn == 40 { d.dump(&self.gpu, &dir.to_string_lossy(), "f40"); }
            }
            Some(Ok(()))
        })();
        self.draft.ctx = Some(d);
        match done {
            Some(Ok(())) => {
            self.draft.drawn += 1;
            if std::env::var_os("GLASSFORMS_DLSS_SAY").is_some() && (!self.draft.said_frame || self.draft.drawn % 30 == 0) { self.draft.said_frame = true; eprintln!("dlss: draft frame {} (accum {})", self.draft.drawn, self.accum_n); }
            true
        }
            Some(Err(e)) => { self.draft_fail(e); false }
            None => false,
        }
    }

    /// Fold this frame in through DLSS: trace a smaller picture with every pixel's ray at the frame's
    /// offset, work out its motion and depth, and let DLSS make the full-size picture so far from it.
    /// False (nothing done) when it cannot be done this frame.
    #[cfg(feature = "dlss")]
    fn frame_dlss(&mut self, tu: &TraceU, look: &Look, skip_post: bool) -> bool {
        if self.up.scale.is_none() || self.cam.active || self.profile || self.split_keys.contains(&self.trace_key) { return false; }
        // (not with the stages that work on the full-size traced frame: rough glass's blur, the denoiser)
        if rough_split(&self.variant, self.profile) || self.variant.denoise { return false; }
        if self.variant.cast && self.g_cast.is_none() { return false; }
        let Some((tw, th)) = self.t.as_ref().map(|t| (t.tw, t.th)) else { return false };
        if !self.trace_pipes.contains_key(&self.centred_key()) { self.build_centred(); }
        if !self.trace_pipes.contains_key(&self.centred_key()) { return false; }
        let Some(f) = self.up.begin(&self.gpu, tw, th, &self.u_accum) else { return false };
        let mut u = *tu;
        u.res = [f.res[0] as f32, f.res[1] as f32];
        u.frame = self.frame_no as i32;
        // the tracer's offsets count up from the bottom; DLSS's down from the top
        u.jitter = [f.jitter[0], -f.jitter[1]];
        let (now, au) = self.accum_u(&u, 1.0, true, look);
        // (nothing the frame shares with the plain path is written until DLSS is sure to draw it: a false
        // return leaves the caller tracing the full-size frame with its own numbers)
        let (Some(t), Some(g_trace)) = (self.t.as_ref(), self.g_trace.as_ref()) else { return false };
        let Some((small_tex, small_view)) = self.up.small() else { return false };
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame (dlss)") });
        {
            let mut rp = pass(&mut enc, small_view, None, true);
            rp.set_pipeline(&self.trace_pipes[&self.centred_key()]);
            rp.set_bind_group(0, g_trace, &[]);
            if let Some(g) = self.g_cast.as_ref().filter(|_| self.variant.cast) { rp.set_bind_group(1, g, &[]); }
            rp.draw(0..3, 0..1);
        }
        if self.surface.on() {
            if let Some((_, vol)) = &self.scene {
                let hb = self.variant.grown.then(|| f32::from_bits(u.pad[2] as u32));
                self.surface.encode(&self.gpu, &mut enc, small_tex, small_view, vol, &self.lin, &self.u_lamps, &u, hb, self.variant.near);
            }
        }
        self.up.encode_prep(&mut enc);
        let target = t.accum_view[1 - self.acc_idx].clone();
        let Some(cb) = self.up.upscale(&self.gpu, &mut enc, &f) else { return false };
        // DLSS draws this frame: the small trace's numbers go in (written before the submit below, so the
        // passes recorded above read them)
        self.gpu.queue.write_buffer(&self.u_trace, 0, bytemuck::bytes_of(&u));
        self.glare_cut = false;
        self.glare_motion = match self.prev {
            Some(was) => Self::view_motion(&now, &was, tw, th),
            None => (0.0, [0.0; 2]),
        };
        self.gpu.queue.write_buffer(&self.u_accum, 0, bytemuck::bytes_of(&au));
        let mut after = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame (after dlss)") });
        self.up.encode_resolve(&mut after, &target);
        self.adapt.valid = false;
        self.dn.live = false;
        self.up.live = true;
        self.up.drawn += 1;
        self.prev = Some(now);
        self.acc_idx = 1 - self.acc_idx;
        self.accum_n += 1;
        self.frame_no += 1;
        // (the passes after are told the full-size picture's numbers, as ever)
        let mut full = *tu;
        full.res = [tw as f32, th as f32];
        if !skip_post { self.encode_post(&mut after, &full, look, false); }
        self.gpu.queue.submit([enc.finish(), cb, after.finish()]);
        self.trace_passes += 1;
        true
    }

    /// Bloom the picture so far and put it on the screen, without tracing anything.
    pub fn present(&mut self, tu: &TraceU, look: &Look) {
        if self.t.is_none() { return; }
        if self.matcap_on.is_some() {
            let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("present (matcap)") });
            self.show_matcap(&mut enc);
            self.gpu.queue.submit([enc.finish()]);
            return;
        }
        self.glare_cut = false;
        self.glare_motion = (0.0, [0.0; 2]);
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("present") });
        self.encode_post(&mut enc, tu, look, true);
        self.gpu.queue.submit([enc.finish()]);
    }

    /// `settled`: the picture did not move for this frame (the focus stage then takes more care).
    fn encode_post(&mut self, enc: &mut wgpu::CommandEncoder, tu: &TraceU, look: &Look, settled: bool) {
        if self.t.is_none() { return; }
        let cut = std::mem::take(&mut self.glare_cut);
        // the lens's glare, unless the old bloom is asked for: the share of light the sharp picture keeps
        let glared = !look.glare.pyramid();
        if !glared {
            self.ensure_pyramid();
            if cut { self.pyramid_interactive = false; }
            let knee = if !settled {
                self.pyramid_interactive = true;
                0.08 * look.exposure.max(0.05)
            } else if !self.pyramid_interactive || self.accum_n >= 8 {
                self.pyramid_interactive = false;
                0.0
            } else {
                0.08 * look.exposure.max(0.05) * (1.0 - self.accum_n as f32 / 8.0).max(0.0)
            };
            if let Some(pyr) = self.t.as_mut().and_then(|t| t.pyramid.as_mut()) {
                if (knee - pyr.last_knee).abs() > 1e-6 {
                    let (texel, dst_res) = pyr.down0_res;
                    self.gpu.queue.write_buffer(&pyr.u_down0, 0, bytemuck::bytes_of(&BloomU { texel, dst_res, gain: 1.0, pad: [knee, 0.0, 0.0] }));
                    pyr.last_knee = knee;
                }
            }
        }
        let keep = if glared { self.glare.keep_for(&look.glare) } else { [1.0; 3] };
        let Some(t) = self.t.as_ref() else { return };
        let mips_len = t.pyramid.as_ref().map_or(1, |p| p.mips.len());
        let fu = FinalU {
            res: [t.w as f32, t.h as f32],
            exposure: look.exposure,
            bloom_amt: if glared { 1.0 } else { look.bloom / mips_len as f32 },
            glare: [keep[0], keep[1], keep[2], 0.0],
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
        self.glared = glared;
        if glared {
            let (motion_px, shift_px) = if settled { (0.0, [0.0; 2]) } else { self.glare_motion };
            self.glare.encode(enc, src, &look.glare, crate::glare::GlareTemporal {
                settled,
                cut,
                accum_n: self.accum_n,
                exposure: look.exposure,
                motion_px,
                shift_px,
            });
        } else if let (Some(pyr), Some(bl)) = (&t.pyramid, &self.bloom) {
            // bloom: walk down the chain, then back up adding each level
            for (i, m) in pyr.mips.iter().enumerate() {
                let mut rp = pass(enc, &m.view, None, true);
                rp.set_pipeline(&bl.down);
                rp.set_bind_group(0, if i == 0 { &pyr.g_down0[src] } else { &m.down }, &[]);
                rp.draw(0..3, 0..1);
            }
            for i in (1..pyr.mips.len()).rev() {
                let mut rp = pass(enc, &pyr.mips[i - 1].view, None, false);
                rp.set_pipeline(&bl.up);
                rp.set_bind_group(0, pyr.mips[i].up.as_ref().unwrap(), &[]);
                rp.draw(0..3, 0..1);
            }
        }
        // tone map to the screen
        {
            let mut rp = pass(enc, &t.final_view, None, true);
            rp.set_pipeline(if lens { self.cam.last(&self.gpu, POST_WGSL, &self.l_final, [LDR, HDR], false) } else { &self.p_final });
            rp.set_bind_group(0, if glared { &t.g_final_glare[src] } else { &t.pyramid.as_ref().expect("bloom pyramid built").g_final[src] }, &[]);
            rp.draw(0..3, 0..1);
        }
        if let Some((_, view, groups_glare)) = &t.deep {
            let groups = if glared { groups_glare } else { t.pyramid.as_ref().and_then(|p| p.g_final_deep.as_ref()).expect("deep bloom pyramid built") };
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
        if self.glared {
            let tex = self.glare.texture().ok_or("the glare has not been sized")?;
            let (w, h) = (tex.width(), tex.height());
            let bytes = self.gpu.read(tex, 0, 0, 0, w, h, 8)?;
            let halves: &[half::f16] = bytemuck::cast_slice(&bytes);
            return Ok((w, h, halves.iter().map(|v| v.to_f32()).collect()));
        }
        let m = &t.pyramid.as_ref().ok_or("nothing has been drawn yet")?.mips[0];
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

#[cfg(test)]
#[test]
fn legacy_bloom_pyramid_is_lazy_until_requested() {
    let Ok(gpu) = Gpu::headless("auto") else { return };
    let mut r = Renderer::new(&gpu);
    assert!(r.bloom.is_none());
    assert!(r.resize(64, 48, 1.0));
    assert!(r.t.as_ref().unwrap().pyramid.is_none());

    let p = gf_core::settings::Settings::default();
    let tu = TraceU { dist: 6.0, tan_fov: 0.25, ..Zeroable::zeroed() };
    let mut look = Look {
        shape_no: 0,
        smooth: 0.0,
        spread: 1.0,
        debug: 0.0,
        exposure: 1.0,
        bloom: 0.35,
        vignette: 0.0,
        grain: 0.0,
        raw: 0.0,
        planes: [0.0; 3],
        still_grain: true,
        focus: Focus::OFF,
        grade: gf_core::stage::Grade::of(&p).uniforms(),
        lens: crate::camera::Lens::of(&p),
        glare: crate::glare::GlareLens::of(&p),
    };
    assert!(!look.glare.pyramid());
    r.present(&tu, &look);
    assert!(r.bloom.is_none() && r.t.as_ref().unwrap().pyramid.is_none());
    assert!(r.read_bloom().is_ok());

    look.glare.kind = 1;
    assert!(look.glare.pyramid());
    r.present(&tu, &look);
    assert!(r.bloom.is_some() && r.t.as_ref().unwrap().pyramid.is_some());
    let (bw, bh, bl) = r.read_bloom().unwrap();
    assert_eq!((bw, bh, bl.len()), (32, 24, 32 * 24 * 4));
    assert_eq!(r.read_screen().unwrap().2.len(), 64 * 48 * 4);
}

#[cfg(test)]
#[test]
fn hero_settle_aligns_after_taa_without_first_frame_full_comb() {
    let Ok(gpu) = Gpu::headless("auto") else { return };
    let mut r = Renderer::new(&gpu);
    r.resize(64, 48, 1.0);
    let env = r.env_texture(&vec![1.0f32; 16 * 8 * 4], 16, 8);
    let vol = Volume::new(&gpu, 16, false);
    r.hero = crate::hero::Hero::settling(true);
    r.set_scene(&env, &vol, 0.95, 64, 32, 1);

    let p = gf_core::settings::Settings::default();
    let tu = TraceU {
        dist: 6.0,
        tan_fov: 0.25,
        ior: 2.4,
        disp: 0.044,
        vres: 16.0,
        nw: 7,
        ns: 1,
        nb: 4,
        obj: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        env: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        ..Zeroable::zeroed()
    };
    let look = Look {
        shape_no: 0,
        smooth: 0.9,
        spread: 2.0,
        debug: 0.0,
        exposure: 1.0,
        bloom: 0.35,
        vignette: 0.0,
        grain: 0.0,
        raw: 0.0,
        planes: [0.0; 3],
        still_grain: true,
        focus: Focus::OFF,
        grade: gf_core::stage::Grade::of(&p).uniforms(),
        lens: crate::camera::Lens::of(&p),
        glare: crate::glare::GlareLens::of(&p),
    };

    // Put `frame_no` right on a multiple of `hero.every` (7) at the end of a moving TAA frame, then step
    // into a 1-frame pause (`moving = false`): pass 1 must align `frame_no % 7 == 1` rather than firing
    // `heroFull` (`frame_no % 7 == 0`) at weight 0.5.
    let e = u32::from(r.hero.every.max(2));
    r.frame_no = e * 7 - 1;
    r.frame(&tu, &look, true, false, false);
    assert_eq!(r.frame_no % e, 0);
    r.frame(&tu, &look, false, false, false);
    // Pass 1 used `frame_no % e == 1` and then incremented `frame_no` to `2 mod e`, so `heroFull` waits
    // until pass 7 (`ahead == e - 2`).
    assert_eq!(r.frame_no % e, 2 % e);
    assert_eq!(r.pass_cost(0, 7, true), 1.0);
    assert!(r.pass_cost(e - 2, 7, true) > 1.0);
}

#[cfg(test)]
#[test]
fn trace_limits_count_the_real_layout() {
    assert_eq!(trace_textures_needed(false), 21);
    assert_eq!(trace_textures_needed(true), 29);
    let mut l = wgpu::Limits::default();
    // (the default 16 textures cannot bind the tracer: refused with a sentence)
    assert!(check_trace_limits(&l).unwrap_err().contains("too small"));
    l.max_sampled_textures_per_shader_stage = 32;
    assert!(check_trace_limits(&l).is_ok());
    l.max_uniform_buffers_per_shader_stage = 8;
    assert!(check_trace_limits(&l).is_err());
}
