//! Shadows and caustics cast by one body on the others, and on itself: the GPU's side.
//! (What the lights are taken apart into, the switches and what follows from a material: gf_core::cast.)
//!
//! Two parts, because the two jobs are different:
//!   shadows    worked out where a surface is shaded, by a march towards each light through the distances
//!              to the bodies (shaders/cast.wgsl): nothing is made beforehand, so nothing has to be made
//!              again when anything moves
//!   caustics   light followed forward from each light through the clear bodies, landed on the surfaces
//!              that gather light diffusely, and kept as a volume of arriving light that a shaded surface
//!              reads (cast_light.rs, shaders/cast_light.wgsl)
//! Both reach the tracer through a bind group of their own (group 1) and one function, `castAt(p, n)`, that
//! is put into the tracer's text only while something casts: with casting off, the tracer, its pipeline
//! and its picture are what they are without any of this. A floor under the shape asks `castFloor(p)`.

use bytemuck::{Pod, Zeroable};

use gf_core::cast::{self as core, CastLight, Caster, Kind, Mode, Optics, MAX_BODIES, MAX_CAUSTIC_LIGHTS, MAX_LIGHTS};
use gf_core::env;
use gf_core::math::{apply, axis_angle, mul, transpose, Mat3, DEG};
use gf_core::HB;

use crate::cast_light::{Caustics, FloorIn, OneU, Scene};
use crate::engine::Engine;
use crate::gpu::*;

pub const CAST_WGSL: &str = include_str!("shaders/cast.wgsl");
pub const BODIES_WGSL: &str = include_str!("shaders/cast_bodies.wgsl");

/// Voxels a side of the volume of distances that marches step through.
pub const BODIES_RES: u32 = 192;

/// The stable receiver boundary shared by ordinary, filmed, exchanged and construction-face paint.
/// Paint already accounts for emitter visibility, so only focused radiance belongs here.
pub(crate) const PAINT_FOCUS_STUB: &str = "return vec3f(0.0); /* PAINT_FOCUS_STUB */";
const PAINT_FOCUS_CAST: &str = "return castFocused(p,n); /* PAINT_FOCUS_STUB */";

pub(crate) fn weave_paint_focus(src: &str) -> String {
    crate::weave::replace(src, "cast.paint_focus", PAINT_FOCUS_STUB, PAINT_FOCUS_CAST)
}

/// Where diffuse light is gathered in the tracer's text, and the same lines with what the bodies cast
/// taken into account.
///   clay (trace_light.wgsl): the gathered light is multiplied, and the sheen dimmed where light is hidden
/// A milky body (of the scene's one material, or one of several bodies) is not among them: it is lit by a
/// light volume worked out beforehand (scatter.rs), and what the bodies cast reaches it there
/// (`Cast::for_light_volume`), not in the tracer.
const SITES: [(&str, &str, &str); 2] = [
    // (what is hidden multiplies the gathered light; what the bodies focus onto it is added, so that a lamp's
    // caustic shows however dim the lights at infinity are)
    ("cast.clay_light", "  var light = nine(0, toEnv(n));","  let castP = castParts(p, n);\n  let castM = max(castP.total - castP.lost, vec3f(0.0)) / max(castP.total, vec3f(1e-5));\n  var light = nine(0, toEnv(n)) * castM + castP.added;"),
    ("cast.clay_sheen", "* (sheen * open);","* (sheen * open * min(dot(castM, vec3f(0.2126, 0.7152, 0.0722)), 1.0));"),
];
/// What a ray from the camera sees where it misses the shape, when nothing lies behind it; and the same
/// with the floor that stands in for a real one in checks (see `test_floor`).
const BEHIND: (&str, &str) = ("  var col = vec3f(0.0);                       // what the camera sees where it misses the object", "  var col = castFloorSeen(ro, rd);");

/// The tracer's text with casting in it: the places where diffuse light is gathered ask `castAt()` (a
/// floor: `castFloor()`), and the text of cast.wgsl is added. Which places there are depends on what
/// the tracer was built to draw; None if it has none (nothing in it gathers light diffusely, or the
/// places have been reworded).
pub fn weave(src: &str) -> Option<String> {
    let paint_receivers = src.matches(PAINT_FOCUS_STUB).count();
    let mut out = weave_paint_focus(src);
    // (a floor that asks what is cast on it is a place too)
    let mut found = usize::from(out.contains("castFloor(") || out.contains("castFloorDirect(")) + paint_receivers;
    // (both clay lines or neither: the surface pass rewords both, so that casting stays out of its clay)
    let counts = SITES.map(|(name, old, _)| crate::weave::found(&out, name, old));
    crate::weave::all_once("cast.clay", &counts);
    for ((_, old, new), count) in SITES.into_iter().zip(counts) {
        if count == 1 {
            out = out.replacen(old, new, 1);
            found += 1;
        }
    }
    if test_floor().is_some() && crate::weave::found(&out, "cast.behind", BEHIND.0) == 1 {
        out = out.replacen(BEHIND.0, BEHIND.1, 1);
        found += 1;
    }
    if found == 0 { return None; }
    // For checks (GLASSFORMS_CAST_SHOW=light): clay shows the diffuse light itself, with what is cast taken
    // into account, instead of its own shading, so that light can be added up over a surface.
    if std::env::var("GLASSFORMS_CAST_SHOW").is_ok_and(|v| v == "light") {
        let head = "fn clay(p: vec3f, rd: vec3f) -> vec3f {\n  let n = nrm(p);\n";
        out = out.replacen(head, &format!("{head}  if (true) {{ let cp = castParts(p, n); return max(cp.total - cp.lost + cp.added, vec3f(0.0)); }}\n"), 1);
    }
    // Construction-face closures are generated after the ordinary casting pass.
    // A final receiver-only pass may therefore see an existing cast implementation.
    // Never append its declarations twice.
    if !out.contains("fn castFocused(") {
        out.push_str(CAST_WGSL);
    }
    Some(out)
}

/// The layout of the bind group through which a tracer reads what is cast (group 1): a block of
/// numbers, two volumes, a sampler and a map: one uniform buffer, three textures and one sampler in all.
pub fn receiver_layout(gpu: &Gpu) -> wgpu::BindGroupLayout {
    let frag = wgpu::ShaderStages::FRAGMENT;
    let d3 = wgpu::TextureViewDimension::D3;
    gpu.bind_layout("cast", &[uniform_entry(0, frag, false), texture_entry(1, d3, true), texture_entry(2, d3, true), sampler_entry(3), texture_entry(4, wgpu::TextureViewDimension::D2Array, true)])
}

/// One light as the tracer is told of it (see cast.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Debug)]
pub struct LightU {
    pub dir: [f32; 4],
    pub rgb: [f32; 4],
    pub t: [[f32; 4]; 3],
}

/// What the tracer is told of what is cast (see cast.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Debug)]
pub struct CastU {
    pub k: [f32; 4],
    pub total: [[f32; 4]; 9],
    pub body: [[f32; 4]; 4],
    pub casts: [f32; 4],
    pub solid: [f32; 4],
    pub far: [f32; 4],
    pub k2: [f32; 4],
    pub room: [[f32; 4]; 3],
    pub floor: [f32; 4],
    pub floor_total: [f32; 4],
    pub floor_share: [[f32; 4]; MAX_LIGHTS],
    pub lights: [LightU; MAX_LIGHTS],
    /// for each body: how much of the light from all round it hides from a floor beneath it
    pub veil: [f32; 4],
    /// the floor's albedo: what it gives back of the light that bodies focus onto it
    pub floor_albedo: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BodiesU {
    count: u32,
    res: u32,
    ch: u32,
    hb: f32,
}

/// A floor under the shape, as whoever owns the room describes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloorSpec {
    /// the room (y up, the floor level in it) -> what the camera sees
    pub m: Mat3,
    /// the floor's height in the room
    pub height: f64,
    /// how far from the room's middle shadows and caustics on it are worked out (half the width of
    /// its map of arriving light)
    pub half: f64,
    /// light an unshadowed floor gathers from what is not among the lights (a backdrop), by colour
    pub extra: [f64; 3],
}

/// The lights taken apart, and what they were taken apart from.
struct Lit {
    key: (usize, u64, u64, bool, u64, u64, u64, Vec<u64>),
    lights: Vec<CastLight>,
    /// what a white matte surface gathers from the whole lighting: nine terms a colour
    total: [[f64; 3]; 9],
    /// the light from everywhere, as bright as it is set
    ambient: [f64; 3],
    /// an environment picture's nine terms (as the shader sums them), as bright as it is set
    picture: Option<[[f64; 3]; 9]>,
}

pub struct Cast {
    gpu: Gpu,
    layout: wgpu::BindGroupLayout,
    smp: wgpu::Sampler,
    u: wgpu::Buffer,
    last_u: Option<CastU>,
    /// the distances to the bodies at modest detail, and what they were read from (the volume, how many
    /// bodies, the count of builds at the time, and the half-width bits)
    bodies: Option<(wgpu::Texture, wgpu::TextureView)>,
    bodies_stage: Option<(wgpu::Texture, wgpu::TextureView)>,
    edt_a: Option<wgpu::Buffer>,
    edt_b: Option<wgpu::Buffer>,
    bodies_from: Option<(wgpu::TextureView, u32, u64, u32)>,
    p_seed: wgpu::ComputePipeline,
    p_edt_x: wgpu::ComputePipeline,
    p_edt_y: wgpu::ComputePipeline,
    p_edt_z: wgpu::ComputePipeline,
    p_finish: wgpu::ComputePipeline,
    l_seed: wgpu::BindGroupLayout,
    l_edt: wgpu::BindGroupLayout,
    l_finish: wgpu::BindGroupLayout,
    u_bodies: [wgpu::Buffer; MAX_BODIES],
    /// stand in for the volume of arriving light and for the floor's map while there is none: one empty
    /// voxel, one empty texel
    no_light: wgpu::TextureView,
    no_map: wgpu::TextureView,
    /// caustics: light followed forward from the lights, and the volume of arriving light it makes
    pub caustics: Caustics,
    /// caustics are being made (something focuses light and something can show it)
    pub focusing: bool,
    /// there is a floor for shadows and caustics to fall on
    pub floor: bool,
    group: Option<wgpu::BindGroup>,
    lit: Option<Lit>,
    /// something casts and something can show it: the tracer has casting in it
    pub on: bool,
    /// how each body casts, as last worked out (bodies in the order of `State::bodies`)
    pub casters: [Caster; MAX_BODIES],
    /// how many lights' shadows are followed
    pub lights: usize,
    /// how many bodies the distances are read for, and how strong shadows are
    count: usize,
    strength: f32,
}

impl Cast {
    pub fn new(gpu: &Gpu, layout: &wgpu::BindGroupLayout) -> Self {
        let c = wgpu::ShaderStages::COMPUTE;
        let tex_d3 = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false };
        let rw_buf = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: c, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let l_seed = gpu.bind_layout("cast: seed", &[
            uniform_entry(0, c, false),
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: c, ty: tex_d3, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: c, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            rw_buf(5),
        ]);
        let l_edt = gpu.bind_layout("cast: edt", &[
            uniform_entry(0, c, false),
            storage_entry(4, c),
            rw_buf(5),
        ]);
        let l_finish = gpu.bind_layout("cast: finish", &[
            uniform_entry(0, c, false),
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: c, ty: tex_d3, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: c, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: c, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D3 }, count: None },
            storage_entry(4, c),
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: c, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false }, count: None },
        ]);
        let module = gpu.shader("cast: bodies", BODIES_WGSL, false);
        let pipe = |label, entry, layout_ref| {
            let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[Some(layout_ref)], immediate_size: 0 });
            gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: Some(&pl), module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None })
        };
        let p_seed = pipe("cast: seed", "cs_seed", &l_seed);
        let p_edt_x = pipe("cast: edt_x", "cs_edt_x", &l_edt);
        let p_edt_y = pipe("cast: edt_y", "cs_edt_y", &l_edt);
        let p_edt_z = pipe("cast: edt_z", "cs_edt_z", &l_edt);
        let p_finish = pipe("cast: finish", "cs_finish", &l_finish);
        let rgba = wgpu::TextureFormat::Rgba16Float;
        let no_light = gpu.volume("cast: no light", 1, 1, rgba, false).create_view(&wgpu::TextureViewDescriptor::default());
        let no_map = gpu.target("cast: no light on the floor", 1, 1, rgba, &[]).create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let u_bodies = std::array::from_fn(|_| gpu.uniform_buffer("cast: bodies", std::mem::size_of::<BodiesU>() as u64));
        Self {
            gpu: gpu.clone(),
            layout: layout.clone(),
            smp: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge),
            u: gpu.uniform_buffer("cast", std::mem::size_of::<CastU>() as u64),
            last_u: None,
            bodies: None, bodies_stage: None, edt_a: None, edt_b: None, bodies_from: None,
            p_seed, p_edt_x, p_edt_y, p_edt_z, p_finish,
            l_seed, l_edt, l_finish, u_bodies,
            no_light, no_map, caustics: Caustics::new(gpu), focusing: false, floor: false,
            group: None, lit: None, on: false, casters: [Caster::none(); MAX_BODIES], lights: 0, count: 0, strength: 0.0,
        }
    }

    /// How many times the distances to the bodies, and the light arriving by way of them, have been
    /// made: what is handed to a light volume changes when either does.
    pub fn made(&self) -> (u64, u64) {
        (self.bodies_from.as_ref().map_or(0, |f| f.2), self.caustics.runs)
    }

    /// What a milky body's light volume (scatter.rs) is told of what the bodies cast on it, or None while
    /// nothing casts on it. `own`: the body being lit, among several each lit by a light volume of its
    /// own: the others' shadows and caustics are marched for, its own shading is the volume's business.
    /// None: the scene's one material; when that is several bodies, a beam inside one of them is darkened
    /// only by that body and by the bodies that cast (so one whose shadow is switched off darkens no
    /// other), and caustics arrive on all of them.
    pub fn for_light_volume(&self, own: Option<usize>) -> Option<crate::scatter::Casting> {
        if !self.on { return None; }
        let u = self.last_u.as_ref()?;
        let (_, bodies) = self.bodies.as_ref()?;
        let mut s = crate::scatter::Shading { casts: u.casts, solid: u.solid, far: u.far, body: u.body, ..Default::default() };
        s.cast = [0.0, 0.0, u.k[2], u.k[3]];
        match own {
            Some(k) => {
                s.cast[0] = 1.0;
                s.casts[k] = 0.0;
                s.solid[k] = 0.0;
                s.far[k] = 10.0;
            }
            // (one body of one material: nothing else to cast on it but caustics)
            None if self.count >= 2 => s.cast[1] = 1.0,
            None => {}
        }
        for k in self.count..MAX_BODIES {
            s.casts[k] = 0.0;
            s.solid[k] = 0.0;
            s.far[k] = 10.0;
        }
        let light = self.caustics.view().filter(|_| self.focusing && u.k2[0] > 1.5).cloned();
        s.caus = [self.strength, u.k[1], if light.is_some() { u.k2[0] } else { 0.0 }, u.k2[1]];
        if s.cast[0] + s.cast[1] == 0.0 && light.is_none() { return None; }
        Some(crate::scatter::Casting { numbers: s, bodies: bodies.clone(), light })
    }

    /// The bind group a tracer with casting in it reads (None until something has cast).
    pub fn group(&self) -> Option<&wgpu::BindGroup> {
        self.group.as_ref()
    }

    /// Read the distances to the bodies into the volume the marches step through, if the volume they
    /// are read from is another one or has been built again. `fine`: the bodies' volume (or the scene's,
    /// with one body), `count`: how many bodies, `built`: a number that changes with every build,
    /// `hb`: half-width of the scene volume in world units.
    fn read_bodies(&mut self, fine: &wgpu::TextureView, count: u32, built: u64, hb: f64) {
        let hb_bits = (hb as f32).to_bits();
        if self.bodies.is_some() && self.bodies_from.as_ref().is_some_and(|f| f.0 == *fine && f.1 == count && f.2 == built && f.3 == hb_bits) { return; }
        let gpu = &self.gpu;
        if self.bodies.is_none() {
            let tex = gpu.volume("cast: bodies", BODIES_RES, BODIES_RES, wgpu::TextureFormat::Rgba16Float, true);
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.bodies = Some((tex, view));
            self.group = None;
        }
        let buf_bytes = (BODIES_RES as u64).pow(3) * std::mem::size_of::<f32>() as u64;
        if self.edt_a.is_none() {
            let mk = |label| gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: buf_bytes, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
            self.edt_a = Some(mk("cast: edt_a"));
            self.edt_b = Some(mk("cast: edt_b"));
        }
        if count > 1 && self.bodies_stage.is_none() {
            let tex = gpu.volume("cast: bodies stage", BODIES_RES, BODIES_RES, wgpu::TextureFormat::Rgba16Float, false);
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.bodies_stage = Some((tex, view));
        }
        let (bodies_tex, out_view) = self.bodies.as_ref().expect("just made");
        let edt_a = self.edt_a.as_ref().expect("just made");
        let edt_b = self.edt_b.as_ref().expect("just made");
        let g3d = BODIES_RES.div_ceil(8);
        let g3d_z = BODIES_RES.div_ceil(4);
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cast: bodies") });
        for ch in 0..count.min(MAX_BODIES as u32) {
            let ub = &self.u_bodies[ch as usize];
            gpu.queue.write_buffer(ub, 0, bytemuck::bytes_of(&BodiesU { count, res: BODIES_RES, ch, hb: hb as f32 }));
            let prev_view = if ch > 0 {
                let (stage_tex, stage_view) = self.bodies_stage.as_ref().expect("just made");
                enc.copy_texture_to_texture(
                    bodies_tex.as_image_copy(),
                    stage_tex.as_image_copy(),
                    wgpu::Extent3d { width: BODIES_RES, height: BODIES_RES, depth_or_array_layers: BODIES_RES },
                );
                stage_view
            } else {
                fine
            };
            let g_seed = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cast: seed"),
                layout: &self.l_seed,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(fine) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.smp) },
                    wgpu::BindGroupEntry { binding: 5, resource: edt_a.as_entire_binding() },
                ],
            });
            let edt_group = |label, src: &wgpu::Buffer, dst: &wgpu::Buffer| gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.l_edt,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: src.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: dst.as_entire_binding() },
                ],
            });
            let g_edt_ab = edt_group("cast: edt a->b", edt_a, edt_b);
            let g_edt_ba = edt_group("cast: edt b->a", edt_b, edt_a);
            let g_finish = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cast: finish"),
                layout: &self.l_finish,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(fine) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.smp) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(out_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: edt_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(prev_view) },
                ],
            });
            let run = |enc: &mut wgpu::CommandEncoder, pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, x, y, z| {
                let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                p.set_pipeline(pipe);
                p.set_bind_group(0, bg, &[]);
                p.dispatch_workgroups(x, y, z);
            };
            run(&mut enc, &self.p_seed, &g_seed, g3d, g3d, g3d_z);
            run(&mut enc, &self.p_edt_x, &g_edt_ab, g3d, g3d, 1);
            run(&mut enc, &self.p_edt_y, &g_edt_ba, g3d, g3d, 1);
            run(&mut enc, &self.p_edt_z, &g_edt_ab, g3d, g3d, 1);
            run(&mut enc, &self.p_finish, &g_finish, g3d, g3d, g3d_z);
        }
        gpu.queue.submit([enc.finish()]);
        self.bodies_from = Some((fine.clone(), count, built, hb_bits));
    }

    /// The group the tracer reads, made if it is not there.
    fn bind(&mut self) {
        if self.group.is_some() { return; }
        let Some((_, bodies)) = &self.bodies else { return };
        let light = self.caustics.view().filter(|_| self.focusing).unwrap_or(&self.no_light);
        let map = self.caustics.floor_view().filter(|_| self.focusing).unwrap_or(&self.no_map);
        self.group = Some(self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cast"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.u.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(bodies) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(light) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.smp) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(map) },
            ],
        }));
    }
}

/// Wait for the GPU at each making of the caustics, so that it can be timed to the end
/// (`GLASSFORMS_CAST_TIME=1`: for measuring; the waiting itself slows the frames down).
fn timed() -> bool {
    static TIMED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TIMED.get_or_init(|| std::env::var_os("GLASSFORMS_CAST_TIME").is_some())
}

/// How the bodies' materials are to be taken for casting, when the environment says so
/// (`GLASSFORMS_CAST_KINDS=clear,matte,...`, one word a body: clear, metal, milky, matte): for trying
/// what a body of one kind casts on a body of another before every body has a material of its own.
fn kinds_override() -> &'static [Option<core::Kind>] {
    static KINDS: std::sync::OnceLock<Vec<Option<core::Kind>>> = std::sync::OnceLock::new();
    KINDS.get_or_init(|| {
        std::env::var("GLASSFORMS_CAST_KINDS").map_or(Vec::new(), |s| {
            s.split(',').map(|w| match w.trim() {
                "clear" => Some(core::Kind::Clear),
                "metal" => Some(core::Kind::Metal),
                "milky" => Some(core::Kind::Milky),
                "matte" => Some(core::Kind::Matte),
                _ => None,
            }).collect()
        })
    })
}

/// For checks: `GLASSFORMS_CAST_UNDER=off` leaves out the shade the bodies standing over a floor give it by
/// hiding part of the light from all round (`castUnder`), so that what the lights' own shadows and the
/// caustics do to the floor can be measured alone.
fn under_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("GLASSFORMS_CAST_UNDER").as_deref() == Ok("off"))
}

/// A plain floor for checks and for looking, when the environment asks for one
/// (`GLASSFORMS_CAST_FLOOR=height[,raised[,half]]`: its height under the shape's middle, how many degrees
/// the camera is raised above it, half the width of the patch of it that is lit; `-1.3,35,4` if only a
/// height is given). It stands in for a real floor until the room has one, and is drawn by cast.wgsl's
/// own `castFloorSeen` where a ray from the camera misses the shape.
fn test_floor() -> Option<FloorSpec> {
    static FLOOR: std::sync::OnceLock<Option<FloorSpec>> = std::sync::OnceLock::new();
    *FLOOR.get_or_init(|| {
        let text = std::env::var("GLASSFORMS_CAST_FLOOR").ok()?;
        let n: Vec<f64> = text.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        let height = *n.first()?;
        Some(FloorSpec { m: axis_angle(1.0, 0.0, 0.0, n.get(1).copied().unwrap_or(35.0) * DEG), height, half: n.get(2).copied().unwrap_or(4.0), extra: [0.0; 3] })
    })
}

impl Engine {
    /// What body `k` (in the order of `State::bodies`; `layer` is its lowest layer) is made of, as
    /// casting takes it: its own material if it has one, else the scene's. Drawn as clay, everything is
    /// clay.
    fn cast_optics(&self, k: usize, layer: usize) -> Optics {
        let p = self.state.body_settings(layer);
        // (a material fitted to its body is measured against that body's own thickness)
        let fit = if self.state.body_own(layer) { gf_core::material::fit_scale(p, self.cast_thick(k)) } else { self.cast_fit() };
        let mut o = if self.state.p.clay > 0.5 { Optics::matte() } else { core::optics_of(p, fit) };
        if let Some(Some(kind)) = kinds_override().get(k) {
            o = Optics { kind: *kind, ..Optics::clear(p.ior, p.disp, p.absorb) };
        }
        o
    }

    /// Do the bodies cast differently from one another by their switches? Then the distances to each
    /// body are needed apart, even while every body is of the scene's one material (when a body has a
    /// material of its own they are made anyway).
    pub(crate) fn cast_wants_bodies(&self) -> bool {
        let p = &self.state.p;
        let bodies = self.state.bodies();
        if Mode::of(p) == Mode::Off || bodies.len() < 2 { return false; }
        // (or a check has said how each body is to cast)
        if kinds_override().len() > 1 { return true; }
        Mode::of(p) == Mode::PerBody && bodies.iter().any(|b| core::switches(p, b[0]) != core::switches(p, bodies[0][0]))
    }

    /// The floor under the shape, if there is one for shadows and caustics to fall on: the studio's
    /// floor (stage.rs: at the height it measures, the shape's lowest point or `floorY`), or the stand-in
    /// a check asks for. A mirror floor shows the shape, not light falling on it, and takes none.
    fn cast_floor(&self) -> Option<FloorSpec> {
        if let Some(f) = test_floor() { return Some(f); }
        if crate::stage::StageWork::seen(&self.state).mirror { return None; }
        let f = self.stage_floor()?;
        // (the map of light arriving on the floor reaches as far as the floor shows anything, within reason)
        Some(FloorSpec { m: f.room, height: f.height, half: f.fade.clamp(1.5, 6.0), extra: f.backdrop })
    }

    /// What casts, as JSON members to add to a report (nothing while casting is off): how many lights'
    /// shadows are followed, how each body casts, and what the caustics came to (with GLASSFORMS_CAST_TIME
    /// set: how long the last making took, to the end, and whether it was the quick kind).
    pub fn cast_json(&self) -> String {
        let c = &self.cast;
        if !c.on { return String::new(); }
        let count = self.bodies().map_or(1, |b| b.count as usize).min(MAX_BODIES);
        let bodies: Vec<String> = c.casters.iter().take(count).map(|b| format!("{{\"kind\":\"{:?}\",\"shadow\":{},\"caustic\":{}}}", b.optics.kind, b.shadow, b.caustic)).collect();
        let (res, bytes) = c.caustics.size();
        format!(
            ",\"cast\":{{\"lights\":{},\"bodies\":[{}],\"floor\":{},\"caustics\":{},\"lightVolume\":{},\"rays\":{},\"makings\":{},\"megabytes\":{:.0},\"makeMs\":{:.3},\"quick\":{}}}",
            c.lights, bodies.join(","), c.floor, c.focusing, if c.focusing { res } else { 0 }, c.caustics.rays, c.caustics.runs,
            (if c.focusing { bytes } else { 0 } + (BODIES_RES as u64).pow(3) * 8) as f64 / 1048576.0, c.caustics.ms, c.caustics.was_quick()
        )
    }

    /// Can anything in the picture show a shadow or a caustic? Only what gathers light diffusely: clay,
    /// a milky body (the scene's material, or a body's own), a floor.
    /// (The window asks this to say whether casting has anything to show.)
    pub fn cast_receivers(&self) -> bool {
        let milky = |p: &gf_core::settings::Settings| gf_core::material::Variant::of(p).sss;
        self.state.p.clay > 0.5 || milky(&self.state.p) || self.state.body_materials().is_some_and(|m| m.iter().any(|p| milky(p))) || self.cast_floor().is_some()
    }

    /// Bring what the bodies cast up to date with the state. True if the tracer is to have casting in
    /// it: something casts, and something in the picture can show it.
    pub(crate) fn cast_upkeep(&mut self) -> bool {
        let mode = Mode::of(&self.state.p);
        if mode == Mode::Off || !self.cast_receivers() {
            if mode == Mode::Off && self.cast.focusing {
                // (the volumes of arriving light are not small: given back when nothing casts)
                self.cast.caustics.clear();
                self.cast.focusing = false;
                self.cast.group = None;
            }
            self.cast.on = false;
            return false;
        }
        // The bodies' distances are needed apart and were not made when the scene was last built (the
        // bodies' switches have come to differ since): build it again, this time with them.
        if self.cast_wants_bodies() && !self.layers.combiner.bodies_wanted {
            self.state.request_if_idle(gf_core::build::BuildRequest::lighting());
        }
        // the bodies and how each casts
        let groups = self.state.bodies();
        let (fine, count, fine_res) = match self.bodies() {
            Some(b) => (b.volume.view.clone(), b.count.min(MAX_BODIES as u32), b.volume.res),
            None => match self.scene_volume() {
                Some(v) => (v.view.clone(), 1, v.res),
                None => {
                    self.cast.on = false;
                    return false;
                }
            },
        };
        let mut casters = [Caster::none(); MAX_BODIES];
        let media = self.state.compiled_media();
        let active = &media.bodies[..count as usize];
        self.cast.caustics.set_medium_priority(active.first().is_some_and(|first| active.iter()
            .any(|medium| medium.interface_priority != first.interface_priority)));
        for (k, c) in casters.iter_mut().enumerate().take(count as usize) {
            // (with one volume for everything, every body is as the first)
            let layer = if count == 1 { 0 } else { groups.get(k).and_then(|g| g.first().copied()).unwrap_or(0) };
            let thin_wall = gf_core::material::Variant::of(self.state.body_settings(layer)).hollow
                && !matches!(kinds_override().get(k), Some(Some(_)));
            let medium = if thin_wall {
                Default::default()
            } else {
                media.bodies[k]
            };
            *c = Caster::of(mode, core::switches(&self.state.p, layer), self.cast_optics(if count == 1 { 0 } else { k }, layer)).with_medium(medium);
        }
        if !casters.iter().any(|c| c.shadow || c.caustic) {
            self.cast.on = false;
            return false;
        }
        let built = self.layers.builds.iter().map(|b| *b as u64).sum::<u64>() + self.layers.combines as u64;
        let hb = self.state.scene_hb();
        self.cast.read_bodies(&fine, count, built, hb);

        // the lights, taken apart (again only when they change)
        let sky_active = self.sky_active();
        let key = {
            let p = &self.state.p;
            let (rig_lit, picture) = self.cast_lighting();
            let sky_key = if sky_active { gf_core::sky::Sky::of(p).paint_key(&self.state.lights_to_room()) } else { Vec::new() };
            (self.state.layer_fam(0), self.state.lights_rev, p.light_gain.to_bits(), rig_lit, picture.map_or(0, |x| x.2), p.hdri_gain.to_bits(), p.hdri_blur.to_bits(), sky_key)
        };
        if self.cast.lit.as_ref().is_none_or(|l| l.key != key) {
            let p = &self.state.p;
            let (rig_lit, picture) = self.cast_lighting();
            let rig = self.state.rig();
            let of_rig = rig_lit.then_some((rig.lights.as_slice(), p.light_gain));
            // the whole lighting as a matte surface gathers it: the rig's lights' own terms, the light
            // from everywhere, and the environment picture's (which hold its glows already)
            let mut total = [[0.0f64; 3]; 9];
            let mut ambient = [0.0; 3];
            if rig_lit {
                for l in core::lights(of_rig, None) {
                    for k in 0..9 {
                        for c in 0..3 { total[k][c] += l.rgb[c] * l.terms[k]; }
                    }
                }
                ambient = [rig.ambient[0] * p.light_gain, rig.ambient[1] * p.light_gain, rig.ambient[2] * p.light_gain];
                for c in 0..3 { total[0][c] += ambient[c]; }
            }
            let (picture_terms, lights) = if let Some((glows, sh, _)) = picture {
                let t = env::sh_terms(sh, env::SH_MATTE);
                let mut out = [[0.0f64; 3]; 9];
                for k in 0..9 {
                    for c in 0..3 {
                        out[k][c] = t[k][c] as f64 * p.hdri_gain;
                        total[k][c] += out[k][c];
                    }
                }
                (Some(out), core::lights(of_rig, Some((glows, p.hdri_gain, crate::hdri::blur_sigma(p.hdri_blur)))))
            } else if sky_active {
                let l2r = self.state.lights_to_room();
                let sky = gf_core::sky::Sky::of(p);
                let dome = if let Some((pano, pw, ph)) = self.clouds.pano() {
                    sky.sh_dome_from_sky_space(&l2r, pano, pw, ph)
                } else {
                    sky.sh_dome(&l2r)
                };
                let t = env::sh_terms(&dome, env::SH_MATTE);
                let mut out = [[0.0f64; 3]; 9];
                for k in 0..9 {
                    for c in 0..3 {
                        out[k][c] = t[k][c] as f64;
                        total[k][c] += out[k][c];
                    }
                }
                let sun = sky.sun_glow_scaled(&l2r, self.sky_sun_atten());
                if let Some(ref g) = sun {
                    let sun_l = core::of_glow(g, 1.0, 0.0);
                    for k in 0..9 {
                        for c in 0..3 { total[k][c] += sun_l.rgb[c] * sun_l.terms[k]; }
                    }
                }
                let ls = core::lights(of_rig, sun.as_ref().map(|g| (std::slice::from_ref(g), 1.0, 0.0)));
                (Some(out), ls)
            } else {
                (None, core::lights(of_rig, None))
            };
            self.cast.lit = Some(Lit { key, lights, total, ambient, picture: picture_terms });
        }

        // what the tracer is told
        let floor = self.cast_floor();
        let p = &self.state.p;
        let lit = self.cast.lit.as_ref().expect("just made");
        // a light's direction in the lights' own frame, as the object sees it; and what the camera sees, as
        // the object sees it
        let cam_to_obj = transpose(&self.state.view_r());
        let to_obj = mul(&cam_to_obj, &self.state.light_m());
        let h = 2.0 * hb / BODIES_RES as f64;
        // Caustics: only while a body focuses light and something can show it. The volumes of arriving
        // light are for the bodies that gather light; with only a floor to show it they are kept tiny.
        let receives = |c: &Caster| matches!(c.optics.kind, Kind::Matte | Kind::Milky);
        let bodies_gather = casters.iter().take(count as usize).any(receives);
        let focusing = p.cast_caustics > 0.0 && casters.iter().any(|c| c.caustic) && (bodies_gather || floor.is_some());
        let light_res = if bodies_gather { core::light_res(p.cast_detail) } else { 16 };
        let mut u: CastU = Zeroable::zeroed();
        // the lights whose shadows are followed: the strongest, and none that sends under a hundredth of
        // all the lights' light (its shadow would not show, and every shaded point would march for it)
        let all: f64 = lit.lights.iter().map(|l| l.weight()).sum();
        let n = lit.lights.iter().take(MAX_LIGHTS).take_while(|l| l.weight() > 0.01 * all).count();
        u.k = [n as f32, p.cast_caustics as f32, (0.75 * h) as f32, (0.6 * h) as f32];
        u.k2 = [if focusing && bodies_gather { light_res as f32 } else { 0.0 }, (crate::cast_light::OFF_SURFACE * 2.0 * hb / light_res as f64) as f32, 0.0, 0.0];
        for k in 0..9 {
            u.total[k] = [lit.total[k][0] as f32, lit.total[k][1] as f32, lit.total[k][2] as f32, 0.0];
        }
        for (k, c) in casters.iter().enumerate() {
            let (takes, lets) = c.shadow_terms();
            u.body[k] = [takes[0] as f32, takes[1] as f32, takes[2] as f32, lets as f32];
            u.casts[k] = if c.shadow { 1.0 } else { 0.0 };
            u.solid[k] = if c.shadow && lets == 0.0 { 1.0 } else { 0.0 };
            u.far[k] = if c.shadow { 0.0 } else { 10.0 };
            u.veil[k] = c.veil() as f32;
        }
        let strength = p.cast_shadows.clamp(0.0, 1.0);
        for (i, l) in lit.lights.iter().take(n).enumerate() {
            let d = apply(&to_obj, l.dir);
            let t = l.terms;
            u.lights[i] = LightU {
                dir: [d[0] as f32, d[1] as f32, d[2] as f32, core::taken_angle(l, p.cast_soft).tan() as f32],
                rgb: [(l.rgb[0] * strength) as f32, (l.rgb[1] * strength) as f32, (l.rgb[2] * strength) as f32, 0.0],
                t: [[t[0] as f32, t[1] as f32, t[2] as f32, t[3] as f32], [t[4] as f32, t[5] as f32, t[6] as f32, t[7] as f32], [t[8] as f32, 0.0, 0.0, 0.0]],
            };
        }
        // The floor: flat, and lit from far away, so every point of it gathers the same light. That
        // light and each light's share of it are worked out here; the shader takes shares away.
        let mut floor_in = None;
        if let Some(f) = &floor {
            for r in 0..3 {
                u.room[r] = [f.m[r * 3] as f32, f.m[r * 3 + 1] as f32, f.m[r * 3 + 2] as f32, 0.0];
            }
            let up = apply(&f.m, [0.0, 1.0, 0.0]);
            // the way up from the floor in the lights' own frame
            let n_l = apply(&transpose(&self.state.light_m()), up);
            let mut total = [lit.ambient[0] + f.extra[0], lit.ambient[1] + f.extra[1], lit.ambient[2] + f.extra[2]];
            if let Some(t) = &lit.picture {
                for c in 0..3 {
                    let terms: [f64; 9] = std::array::from_fn(|k| t[k][c]);
                    total[c] += core::nine(&terms, n_l).max(0.0);
                }
            }
            let shares: Vec<f64> = lit.lights.iter().map(|l| l.facing(n_l)).collect();
            for (l, share) in lit.lights.iter().zip(&shares) {
                for c in 0..3 { total[c] += l.rgb[c] * share; }
            }
            // (a light that gives the floor under a hundredth of its light casts no shadow on it worth a march)
            let least = 0.01 * core::lum(total);
            for (i, (l, share)) in lit.lights.iter().zip(&shares).take(n).enumerate() {
                if core::lum(l.rgb) * share > least {
                    u.floor_share[i] = [(l.rgb[0] * share * strength) as f32, (l.rgb[1] * share * strength) as f32, (l.rgb[2] * share * strength) as f32, 0.0];
                }
            }
            // (the floor's maps of arriving light: the finest round the middle, each further one twice as wide)
            let (nf, half0) = crate::cast_light::floor_levels(f.half);
            u.floor = [f.height as f32, half0 as f32, 1.0, if focusing { 1.0 } else { 0.0 }];
            let g = p.floor_gain.max(0.0);
            u.floor_albedo = [(p.floor_r * g) as f32, (p.floor_g * g) as f32, (p.floor_b * g) as f32, strength as f32];
            u.k2[2] = nf as f32;
            // (the strength of the ambient pool of shade under bodies standing over the floor; castFloor applies it
            // by luminance-balanced ambient share so direct sunlight never shifts neutral floor shade yellow-brown)
            let under = if under_off() { 0.0 } else { strength };
            u.floor_total = [total[0] as f32, total[1] as f32, total[2] as f32, under as f32];
            // and as the light meets it: in the object's frame
            let room_to_obj = mul(&cam_to_obj, &f.m);
            floor_in = Some(FloorIn {
                origin: apply(&room_to_obj, [0.0, f.height, 0.0]),
                up: apply(&room_to_obj, [0.0, 1.0, 0.0]),
                u: apply(&room_to_obj, [1.0, 0.0, 0.0]),
                v: apply(&room_to_obj, [0.0, 0.0, 1.0]),
                half: f.half,
                res: if p.cast_detail >= 1.5 { 2048 } else { 1024 },
            });
        }
        if focusing {
            let top = lit.lights.first().map_or(0.0, |l| l.weight());
            let (mut ones, mut weights) = (Vec::new(), Vec::new());
            // (a light that sends less than a hundredth of the strongest one's light is not followed)
            for l in lit.lights.iter().take(MAX_CAUSTIC_LIGHTS).filter(|l| l.weight() > 0.01 * top) {
                let d = apply(&to_obj, l.dir);
                let (ul, vl) = core::across(l.dir);
                let (uo, vo) = (apply(&to_obj, ul), apply(&to_obj, vl));
                // what a white surface facing the light shows of it
                let b = l.flux / std::f64::consts::PI;
                ones.push(OneU {
                    dir: [d[0] as f32, d[1] as f32, d[2] as f32, core::taken_angle(l, p.cast_soft).tan() as f32],
                    // the grid of rays covers the whole scene (nothing lies further out than 1.47)
                    u: [uo[0] as f32, uo[1] as f32, uo[2] as f32, (1.47 * hb / HB) as f32],
                    v: [vo[0] as f32, vo[1] as f32, vo[2] as f32, 0.0],
                    rgb: [(l.rgb[0] * b) as f32, (l.rgb[1] * b) as f32, (l.rgb[2] * b) as f32, 0.0],
                    at: [0.0; 4],
                });
                weights.push(l.weight());
            }
            // the lamps: their rays spread from where each stands, through a grid across the middle of the
            // scene (a lamp among the bodies sends light only that way); their light there is what a white
            // surface facing them at the middle would show
            let lamp_ones = crate::lamps::caustic_lights(&self.state, &to_obj, hb);
            for (one, w) in lamp_ones.into_iter().rev() {
                ones.insert(0, one);
                weights.insert(0, w);
            }
            let m = self.state.march();
            let Some((_, coarse)) = self.cast.bodies.as_ref() else { return false };
            let scene = Scene { fine: &fine, fine_res, count, built, casters, march: (m.0, m.1, m.2), res: light_res, coarse, coarse_res: BODIES_RES, floor: floor_in, hb };
            // (a video's frames, and frames averaged for a still, are each made in full)
            let rest = self.state.rendering || self.averaging;
            // (frames the still picture has averaged so far: the first few have the caustics refined)
            let pass = self.renderer.still_frames();
            let (made, fresh, refined) = self.cast.caustics.run(&scene, &ones, &weights, rest, timed(), pass);
            if fresh { self.cast.group = None; }
            // (a making that refines leaves the picture so far as good as it was: it averages on)
            if made && !refined { self.dirty = true; }
            // the floor's maps as the last making moved them
            let sh = self.cast.caustics.shift;
            u.room[1][3] = sh[0] as f32;
            u.room[2][3] = sh[1] as f32;
            // (and while things move the finest of them is not made: the tracer reads from the next)
            if self.cast.caustics.was_quick() && u.k2[2] > 1.5 { u.k2[3] = 1.0; }
        }
        if self.cast.last_u != Some(u) {
            self.gpu.queue.write_buffer(&self.cast.u, 0, bytemuck::bytes_of(&u));
            // (what is cast has changed: the picture so far is out of date; unless only the floor's maps
            // were moved, by a making that refines)
            let unmoved = |mut x: CastU| {
                x.room[1][3] = 0.0;
                x.room[2][3] = 0.0;
                x
            };
            if self.cast.last_u.map(unmoved) != Some(unmoved(u)) { self.dirty = true; }
            self.cast.last_u = Some(u);
        }
        if focusing != self.cast.focusing {
            self.cast.focusing = focusing;
            self.cast.group = None;
            self.dirty = true;
        }
        self.cast.floor = floor.is_some();
        self.cast.bind();
        self.renderer.set_cast(self.cast.group.as_ref());
        self.cast.casters = casters;
        self.cast.lights = n;
        self.cast.count = count as usize;
        self.cast.strength = strength as f32;
        self.cast.on = self.cast.group.is_some();
        self.cast.on
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_places_where_light_is_gathered_are_found() {
        // clay's text still has the lines that casting is woven into
        let clay = include_str!("shaders/trace_light.wgsl");
        let woven = weave(clay).expect("a place where clay gathers its light has moved");
        assert!(woven.contains("nine(0, toEnv(n)) * castM") && woven.contains("sheen * open * min(") && woven.contains("fn castAt(p: vec3f, n: vec3f)"));
        // a milky body (alone, or among several) takes its light from its light volume, which is told what
        // is cast (`Cast::for_light_volume`): nothing in its tracer gathers; nor in glass
        assert!(weave(include_str!("shaders/trace_material.wgsl")).is_none());
        assert!(weave(include_str!("shaders/trace_bodies.wgsl")).is_none());
        assert!(weave(include_str!("shaders/trace.wgsl")).is_none());
        // a floor that asks what is cast on it counts as a place
        assert!(weave("var fl = su.floor.rgb * castFloor(p);").is_some_and(|s| s.contains("fn castFloor(p: vec3f)")));
        assert!(weave("var fl = su.floor.rgb * castFloorDirect(p);").is_some_and(|s| s.contains("fn castFloorDirect(p: vec3f)")));
        // the tracer still says what lies behind the shape the way the stand-in floor expects
        assert_eq!(include_str!("shaders/trace.wgsl").replace("{{BACKGROUND}}", "vec3f(0.0)").matches(BEHIND.0).count(), 1);
    }

    #[test]
    fn the_block_is_laid_out_as_the_shader_declares_it() {
        // 16 + 144 + 64 + 64 (k, the nine terms, the bodies, three rows of switches and k2)
        //   + 48 + 32 + 12 x 16 (the room, the floor and its lights' shares) + 12 x 80 (the lights)
        assert_eq!(std::mem::size_of::<CastU>(), 16 + 9 * 16 + 4 * 16 + 4 * 16 + 3 * 16 + 2 * 16 + MAX_LIGHTS * 16 + MAX_LIGHTS * 80 + 2 * 16);
        assert!(CAST_WGSL.contains(&format!("lights: array<CastLightU, {MAX_LIGHTS}>")) && CAST_WGSL.contains(&format!("floorShare: array<vec4f, {MAX_LIGHTS}>")));
        assert_eq!(CAST_WGSL.matches(&format!("i < {MAX_LIGHTS}; i++")).count(), 2);
    }
}
