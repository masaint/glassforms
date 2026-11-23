//! Volumetric Sky-Space GPU Cache: bakes the 360° volumetric cloud dome, atmospheric mist,
//! crepuscular sunrays, companion 2D ground shadow/transmittance map, and room-frame IBL panorama
//! strictly on demand when sky, sun, cloud, or atmosphere parameters change.
//!
//! Camera orbiting, panning, zooming, object/rig transforms, and progressive refinement samples
//! reuse the cached sky-space textures at zero cloud-raymarching cost.
//!
//! A change is baked coarse first and refined as the sky holds still (`LEVELS`): a 320 dome and a
//! 256 shadow map while the sky changes frame after frame, 640 once it has held still for a couple of
//! frames, and the full 1280 once it has rested a moment. The finest bake is the one there always was,
//! to the bit. In the live view a bake is spread over frames (`SLICE_MS` of GPU work each) and read back
//! without waiting; coarse results are blended one into the next. Stills and videos (`sync`) bake the
//! finest at once.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use gf_core::sky::Sky;
use gf_core::state::State;

use crate::gpu::*;

const CLOUDS_WGSL: &str = include_str!("shaders/clouds.wgsl");
const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Tracer group(0) bindings for the cached volumetric sky-space dome and companion ground shadow map.
pub const DOME_BINDING: u32 = 41;
pub const SHADOW_BINDING: u32 = 42;

/// Per-face resolution of the 6-face volumetric sky-space cubemap (`6 × 1280²` `Rgba16Float`).
pub const DOME_FACE: u32 = 1280;
/// Resolution of the companion 2D ground-projected cloud transmittance & optical-depth map.
pub const SHADOW_SIZE: u32 = 512;
/// Equirectangular room-frame sky-space projection dimensions feeding `lights_tex`, `softTex`, and SH9.
pub const PANO_W: usize = 1024;
pub const PANO_H: usize = 512;

/// The bakes from coarse to fine: (dome face, shadow map). The last is the resting one.
pub const LEVELS: [(u32, u32); 3] = [(320, 256), (640, SHADOW_SIZE), (DOME_FACE, SHADOW_SIZE)];
pub const TOP: usize = LEVELS.len() - 1;
/// How long the sky must have held still before the next finer bake: frames, and (for the finest) time.
const MID_AFTER_FRAMES: u32 = 2;
const TOP_AFTER_FRAMES: u32 = 3;
const TOP_AFTER: Duration = Duration::from_millis(300);

/// Characteristic radial half-span (km) of the smooth infinite-domain shadow UV mapping (`shadowUv`).
pub const SHADOW_SPAN_KM: f32 = 10.0;
/// Horizontal cloud-space kilometres per stage floor unit for dappled ground cloud shadows.
pub const FLOOR_KM_PER_UNIT: f32 = 0.45;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CloudU {
    sky0: [f32; 4],
    sky1: [f32; 4],
    sky2: [f32; 4],
    sky3: [f32; 4],
    sky4: [f32; 4],
    cloud0: [f32; 4],
    cloud1: [f32; 4],
    cloud2: [f32; 4],
    atmo0: [f32; 4],
    atmo1: [f32; 4],
}

impl CloudU {
    fn of(sky: &Sky, face_sz: u32, shadow_sz: u32) -> Self {
        let sun = sky.sun_dir();
        let scale = sky.sun_scale_rgb();
        let (ground_lit, sun_ground) = sky.dome_and_sun_ground();
        let cover = if sky.clouds_active() { sky.clouds.cover as f32 } else { 0.0 };
        Self {
            sky0: [sun[0] as f32, sun[1] as f32, sun[2] as f32, sky.sun_radius() as f32],
            sky1: [scale[0] as f32, scale[1] as f32, scale[2] as f32, sky.gain as f32],
            sky2: [sky.rayleigh as f32, sky.mie as f32, sky.mie_g as f32, sky.ozone as f32],
            sky3: [ground_lit[0] as f32, ground_lit[1] as f32, ground_lit[2] as f32, cover],
            sky4: [sun_ground[0] as f32, sun_ground[1] as f32, sun_ground[2] as f32, cover],
            cloud0: [
                sky.clouds.density as f32,
                sky.clouds.scale as f32,
                sky.clouds.base as f32,
                sky.clouds.thick as f32,
            ],
            cloud1: [
                sky.clouds.detail as f32,
                sky.clouds.soft as f32,
                sky.clouds.silver as f32,
                SHADOW_SPAN_KM,
            ],
            cloud2: [
                sky.clouds.wind_x as f32,
                sky.clouds.wind_z as f32,
                PANO_W as f32,
                PANO_H as f32,
            ],
            atmo0: [
                sky.atmo.haze as f32,
                sky.atmo.fog_h as f32,
                sky.atmo.aerial as f32,
                sky.atmo.sun_rays as f32,
            ],
            atmo1: [
                sky.atmo.sun_ray_g as f32,
                face_sz as f32,
                shadow_sz as f32,
                0.0,
            ],
        }
    }
}

/// The next bake the cache wants, if any (pure, so that it can be tested without a GPU).
///   fresh     the sky is not what the newest bake was made for (or nothing is shown)
///   sync      a still or a video: the finest bake, now
///   shown     the level now shown; still_frames and rested: how long the sky has held still since
pub fn next_level(fresh: bool, sync: bool, shown: usize, still_frames: u32, rested: Duration) -> Option<usize> {
    if fresh { return Some(if sync { TOP } else { 0 }); }
    if shown >= TOP { return None; }
    if sync { return Some(TOP); }
    let ready = match shown + 1 {
        1 => still_frames >= MID_AFTER_FRAMES,
        _ => still_frames >= TOP_AFTER_FRAMES && rested >= TOP_AFTER,
    };
    ready.then_some(shown + 1)
}

/// GPU time a frame may give to a bake in the live view (ms, by the estimates of `step_ms`). A bake
/// takes as many frames as it needs; the sky lags by that many.
pub const SLICE_MS: f64 = 4.0;

/// One piece of a bake: rows `y0..y1` of the shadow map or of one dome face.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Step {
    Shadow { y0: u32, y1: u32 },
    Face { face: u32, y0: u32, y1: u32 },
}

/// `size` rows cut into `n` bands of whole rows (the last takes what is left).
pub fn bands(size: u32, n: u32) -> Vec<(u32, u32)> {
    let n = n.clamp(1, size.max(1));
    let each = size.div_ceil(n);
    (0..n).map(|k| (k * each, ((k + 1) * each).min(size))).filter(|(a, b)| a < b).collect()
}

/// The pieces of one bake, in the order they must run (the dome reads the shadow map): `split` false
/// gives the shadow map and each face whole, as one frame bakes them. Split, the coarse bake runs face
/// by face; the finer ones in bands of 160 rows of sky, a side face's lower half (ground, next to nothing
/// to march) in one piece, and the shadow map in bands of 128 rows.
pub fn plan(face_sz: u32, shadow_sz: u32, split: bool) -> Vec<Step> {
    let whole = |size: u32| vec![(0, size)];
    let shadow = if split && shadow_sz > 256 { bands(shadow_sz, shadow_sz.div_ceil(128)) } else { whole(shadow_sz) };
    let mut steps: Vec<Step> = shadow.into_iter().map(|(y0, y1)| Step::Shadow { y0, y1 }).collect();
    let fine = split && face_sz > 320;
    let sky_rows = |rows: u32| bands(rows, rows.div_ceil(160));
    for face in 0..6 {
        let pieces = match face {
            _ if !fine => whole(face_sz),
            // looking down: the lit ground
            3 => whole(face_sz),
            // looking up: all sky
            2 => sky_rows(face_sz),
            // a side face: sky above its middle row, ground below
            _ => {
                let half = face_sz / 2;
                let mut p = sky_rows(half);
                p.push((half, face_sz));
                p
            }
        };
        steps.extend(pieces.into_iter().map(|(y0, y1)| Step::Face { face, y0, y1 }));
    }
    steps
}

/// Estimated GPU time of a step (ms), from the pieces timed one at a time on the 5090 (`GF_SKYSPLIT`).
/// The rows just above a side face's horizon are the slowest (160 march steps through the longest
/// cloud path), and a pass holding them costs about as much as its slowest pixel, however few rows it
/// has. Only how many steps a frame runs depends on this, never what they make.
pub fn step_ms(step: Step, size: u32) -> f64 {
    let rows = |y0: u32, y1: u32| (y1 - y0) as f64;
    // the slowest pixels, by face size (one whole coarse face; a band at the horizon)
    let horizon = if size <= 320 { 5.7 } else if size <= 640 { 8.0 } else { 13.0 };
    match step {
        Step::Shadow { y0, y1 } => 0.4 + 11.0 * rows(y0, y1) * size as f64 / (512.0 * 512.0),
        Step::Face { face: 3, .. } => 0.3,
        Step::Face { face: 2, y0, y1 } => 0.3 + 2.7 * rows(y0, y1) * size as f64 / (80.0 * 1280.0),
        Step::Face { y0, y1, .. } => {
            let half = size / 2;
            if y0 >= half { return 0.3; }
            let sky = rows(y0, y1.min(half));
            let at_horizon = y1 >= half;
            let floor: f64 = if at_horizon { horizon } else { 0.0 };
            0.2 + floor.max(3.4 * sky / 80.0)
        }
    }
}

/// How many of `steps` from `next` on a frame runs: at least one, then while the budget lasts.
pub fn steps_this_frame(costs: &[f64], next: usize, budget: f64) -> usize {
    let mut spent = 0.0;
    let mut n = 0;
    while next + n < costs.len() && (n == 0 || spent + costs[next + n] <= budget) {
        spent += costs[next + n];
        n += 1;
    }
    n
}

/// The coarse bakes take turns over three sets of textures: the newest result, the one before it (the
/// shown sky is blended from that one to the newest), and the one being made. The free one.
pub fn free_ring_slot(prev: usize, newest: usize) -> usize {
    (0..RING_SLOTS).find(|s| *s != prev && *s != newest).unwrap_or(0)
}

/// `a` blended towards `b` by `t` into `out` (all the same length). `t` of 1 or more gives `b` exactly.
pub fn lerp_into(out: &mut Vec<f32>, a: &[f32], b: &[f32], t: f64) {
    use rayon::prelude::*;
    out.resize(b.len(), 0.0);
    if t >= 1.0 || a.len() != b.len() {
        out.copy_from_slice(b);
        return;
    }
    let t = t as f32;
    out.par_chunks_mut(4096).zip(a.par_chunks(4096).zip(b.par_chunks(4096))).for_each(|(o, (a, b))| {
        for ((o, a), b) in o.iter_mut().zip(a).zip(b) { *o = a + (b - a) * t; }
    });
}

/// The coarse bakes' sets of textures (`free_ring_slot`), then one set for each finer level.
const RING_SLOTS: usize = 3;
fn slot_of_level(level: usize) -> usize {
    RING_SLOTS + level - 1
}
fn level_of_slot(slot: usize) -> usize {
    if slot < RING_SLOTS { 0 } else { slot + 1 - RING_SLOTS }
}

/// Copies of the panorama on their way back from the GPU, taken in turn.
const READ_RING: usize = 2;

/// The textures of one bake.
struct Bake {
    face_sz: u32,
    shadow_sz: u32,
    shadow_tex: wgpu::Texture,
    shadow_view: wgpu::TextureView,
    shadow_array: wgpu::TextureView,
    dome_tex: wgpu::Texture,
    dome_view: wgpu::TextureView,
    dome_array: wgpu::TextureView,
    dome_faces: [wgpu::TextureView; 6],
    bg_dome: wgpu::BindGroup,
    bg_pano: wgpu::BindGroup,
}

/// A finished bake: its textures, what it was made for, and what was read back of it.
struct Slot {
    bake: Bake,
    key: Vec<u64>,
    pano: Vec<f32>,
    sun_t: f64,
}

/// A bake under way, a few steps a frame.
struct Job {
    slot: usize,
    key: Vec<u64>,
    cu: CloudU,
    clouds_active: bool,
    steps: Vec<Step>,
    costs: Vec<f64>,
    next: usize,
    started_frame: u64,
    /// the panorama's copy on its way back: the ring buffer, and whether it has arrived
    read: Option<(usize, Arc<AtomicU8>)>,
}

/// Self-contained GPU volumetric sky-space cache.
pub struct CloudCache {
    gpu: Gpu,
    l_dome: wgpu::BindGroupLayout,
    l_pano: wgpu::BindGroupLayout,
    l_blend: wgpu::BindGroupLayout,
    p_shadow: wgpu::RenderPipeline,
    p_dome: wgpu::RenderPipeline,
    p_pano: wgpu::RenderPipeline,
    p_blend: wgpu::RenderPipeline,
    u_buf: wgpu::Buffer,
    blend_buf: wgpu::Buffer,
    lin: wgpu::Sampler,
    bg_shadow: wgpu::BindGroup,
    slots: Vec<Option<Slot>>,
    /// what the tracer reads while coarse bakes are shown: blended from one to the next
    display: Option<Bake>,
    pano_tex: Option<(wgpu::Texture, wgpu::TextureView)>,
    ring: Vec<wgpu::Buffer>,
    ring_next: usize,
    job: Option<Job>,
    /// the sky asked for last, and how long it has held still since
    want: Vec<u64>,
    still_frames: u32,
    changed_at: Instant,
    active: bool,
    /// the level shown, the slot of its newest bake, and (coarse) the slot blended from and how far
    shown: usize,
    newest: usize,
    prev: usize,
    blend: Option<(f64, f64)>,
    frame: u64,
    bake_count: u64,
    generation: u64,
    sun_transmittance: f64,
    pano_rgba: Vec<f32>,
    split: bool,
}

const BLEND_WGSL: &str = r#"
struct BlendU { t: vec4f }
@group(0) @binding(0) var<uniform> bu: BlendU;
@group(0) @binding(1) var a: texture_2d_array<f32>;
@group(0) @binding(2) var b: texture_2d_array<f32>;
struct V { @builtin(position) pos: vec4f, @location(0) @interpolate(flat) layer: u32 }
@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) layer: u32) -> V {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return V(vec4f(p * 2.0 - 1.0, 0.0, 1.0), layer);
}
@fragment
fn fs(v: V) -> @location(0) vec4f {
  let p = vec2i(v.pos.xy);
  return mix(textureLoad(a, p, i32(v.layer), 0), textureLoad(b, p, i32(v.layer), 0), bu.t.x);
}
"#;

impl CloudCache {
    pub fn new(gpu: &Gpu) -> Self {
        let frag = wgpu::ShaderStages::FRAGMENT;
        let (d2, d2a, dcube) = (wgpu::TextureViewDimension::D2, wgpu::TextureViewDimension::D2Array, wgpu::TextureViewDimension::Cube);
        let l_shadow = gpu.bind_layout("clouds: shadow", &[uniform_entry(0, frag, false)]);
        let l_dome = gpu.bind_layout(
            "clouds: dome",
            &[uniform_entry(0, frag, false), texture_entry(1, d2, true), sampler_entry(3)],
        );
        let l_pano = gpu.bind_layout(
            "clouds: pano",
            &[uniform_entry(0, frag, false), texture_entry(2, dcube, true), sampler_entry(3)],
        );
        let l_blend = gpu.bind_layout("clouds: blend", &[uniform_entry(0, frag, false), texture_entry(1, d2a, false), texture_entry(2, d2a, false)]);
        let module = gpu.shader("clouds", CLOUDS_WGSL, false);
        let blend_module = gpu.shader("clouds: blend", BLEND_WGSL, false);
        let pipe = |label, module, vs, fs, layout: &wgpu::BindGroupLayout| {
            gpu.pipeline(&PipeDesc {
                label,
                module,
                vs,
                fs,
                layouts: &[layout],
                format: HDR,
                blend: None,
                strip: false,
                buffers: &[],
            })
        };
        let p_shadow = pipe("clouds: shadow", &module, "vs_quad", "fs_shadow", &l_shadow);
        let p_dome = pipe("clouds: dome", &module, "vs", "fs_dome", &l_dome);
        let p_pano = pipe("clouds: pano", &module, "vs_quad", "fs_pano", &l_pano);
        let p_blend = pipe("clouds: blend", &blend_module, "vs", "fs", &l_blend);
        let u_buf = gpu.uniform_buffer("clouds", std::mem::size_of::<CloudU>() as u64);
        let blend_buf = gpu.uniform_buffer("clouds: blend", 16);
        let lin = gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge);
        let bg_shadow = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clouds: shadow"),
            layout: &l_shadow,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: u_buf.as_entire_binding() }],
        });
        Self {
            gpu: gpu.clone(),
            l_dome,
            l_pano,
            l_blend,
            p_shadow,
            p_dome,
            p_pano,
            p_blend,
            u_buf,
            blend_buf,
            lin,
            bg_shadow,
            slots: (0..RING_SLOTS + TOP).map(|_| None).collect(),
            display: None,
            pano_tex: None,
            ring: Vec::new(),
            ring_next: 0,
            job: None,
            want: Vec::new(),
            still_frames: 0,
            changed_at: Instant::now(),
            active: false,
            shown: TOP,
            newest: slot_of_level(TOP),
            prev: 0,
            blend: None,
            frame: 0,
            bake_count: 0,
            generation: 0,
            sun_transmittance: 1.0,
            pano_rgba: Vec::new(),
            split: std::env::var_os("GF_SKYSPLIT").is_some(),
        }
    }

    /// True when the volumetric sky-space cache is active for the current scene.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Total number of GPU sky-space bakes performed since engine startup (for dirty-tracking verification).
    pub fn bake_count(&self) -> u64 {
        self.bake_count
    }

    /// Goes up whenever what the tracer reads of the cache changes (a bake lands, a blend step).
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Raw solar-disc cloud transmittance (`0..=1`) at the scene origin (`1.0` when clouds are inactive).
    pub fn sun_transmittance(&self) -> f64 {
        if self.active { self.sun_transmittance } else { 1.0 }
    }

    /// Room-frame equirectangular volumetric sky-space buffer (`PANO_W × PANO_H` RGBA `f32`) when active.
    pub fn pano(&self) -> Option<(&[f32], usize, usize)> {
        if self.active && self.pano_rgba.len() == PANO_W * PANO_H * 4 {
            Some((&self.pano_rgba, PANO_W, PANO_H))
        } else {
            None
        }
    }

    /// The bake under way: steps run and steps in all (None when none is).
    pub fn progress(&self) -> Option<(usize, usize)> {
        self.job.as_ref().map(|j| (j.next, j.steps.len()))
    }

    /// The level of the bake shown (`LEVELS`).
    pub fn level(&self) -> usize {
        self.shown
    }

    /// Nothing is left to bake for the sky as it was last asked for: the finest bake of it is shown (or
    /// there is no sky-space at all).
    pub fn settled(&self) -> bool {
        !self.active
            || (self.shown == TOP && self.job.is_none() && self.slots[self.newest].as_ref().is_some_and(|s| s.key == self.want))
    }

    fn shown_bake(&self) -> Option<&Bake> {
        if self.shown == 0 { self.display.as_ref() } else { self.slots[self.newest].as_ref().map(|s| &s.bake) }
    }

    /// The cache as the tracer reads it, hashed: the panorama, the sun's transmittance, the dome and the
    /// shadow map shown (read back from the GPU; see `Engine::sky_fingerprint`).
    pub fn fingerprint(&self) -> String {
        if !self.active { return "off".into(); }
        let hash = crate::engine::hash_f32;
        let Some(b) = self.shown_bake() else { return "no bake".into() };
        let halves = |bytes: Vec<u8>| -> Vec<f32> { bytes.chunks(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect() };
        let mut dome = Vec::new();
        for z in 0..6 {
            if let Ok(v) = self.gpu.read(&b.dome_tex, 0, 0, z, b.face_sz, b.face_sz, 8) { dome.extend(halves(v)); }
        }
        let shadow = self.gpu.read(&b.shadow_tex, 0, 0, 0, b.shadow_sz, b.shadow_sz, 8).map(halves).unwrap_or_default();
        format!(
            "level {} pano {:016x} sun {:016x} dome {:016x} shadow {:016x}",
            self.shown,
            hash(&self.pano_rgba),
            self.sun_transmittance.to_bits(),
            hash(&dome),
            hash(&shadow)
        )
    }

    /// Active `(sky_dome_cube_view, cloud_shadow_2d_view)` for the tracer when active.
    pub fn views(&self) -> Option<(&wgpu::TextureView, &wgpu::TextureView)> {
        if !self.active {
            return None;
        }
        let b = self.shown_bake()?;
        Some((&b.dome_view, &b.shadow_view))
    }

    fn make_bake(&self, level: usize) -> Bake {
        let gpu = &self.gpu;
        let (face, shadow_sz) = LEVELS[level];
        let face_sz = face.min(gpu.device.limits().max_texture_dimension_2d).max(64);
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;
        let layers = |label, size, n| {
            gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: n },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HDR,
                usage,
                view_formats: &[],
            })
        };
        let view = |t: &wgpu::Texture, dim| t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(dim), ..Default::default() });
        let shadow_tex = layers("cloud shadow", shadow_sz, 1);
        let shadow_view = view(&shadow_tex, wgpu::TextureViewDimension::D2);
        let shadow_array = view(&shadow_tex, wgpu::TextureViewDimension::D2Array);
        let dome_tex = layers("sky-space dome", face_sz, 6);
        let dome_view = view(&dome_tex, wgpu::TextureViewDimension::Cube);
        let dome_array = view(&dome_tex, wgpu::TextureViewDimension::D2Array);
        let dome_faces = std::array::from_fn(|face| {
            dome_tex.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: face as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        });
        let bg_dome = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clouds: dome"),
            layout: &self.l_dome,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.u_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&shadow_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.lin) },
            ],
        });
        let bg_pano = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clouds: pano"),
            layout: &self.l_pano,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.u_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dome_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.lin) },
            ],
        });
        Bake { face_sz, shadow_sz, shadow_tex, shadow_view, shadow_array, dome_tex, dome_view, dome_array, dome_faces, bg_dome, bg_pano }
    }

    fn ensure_slot(&mut self, slot: usize) {
        if self.slots[slot].is_none() {
            let bake = self.make_bake(level_of_slot(slot));
            self.slots[slot] = Some(Slot { bake, key: Vec::new(), pano: Vec::new(), sun_t: 1.0 });
        }
        if self.pano_tex.is_none() {
            let tex = self.gpu.target("sky-space pano", PANO_W as u32, PANO_H as u32, HDR, &[]);
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.pano_tex = Some((tex, view));
        }
        if self.ring.is_empty() {
            let (_, _, size) = read_layout();
            self.ring = (0..READ_RING)
                .map(|_| {
                    self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("clouds readback"),
                        size,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    })
                })
                .collect();
        }
    }

    /// Update the cached volumetric sky-space when sky, sun, cloud, or atmosphere parameters change, and
    /// refine it while they hold still. `sync`: a still or a video, which gets the finest bake at once.
    /// Returns `(sky_space_changed, views_changed)`.
    ///
    /// In the live view (not `sync`) a bake is spread over frames, `SLICE_MS` of GPU work a frame, and
    /// its panorama comes back without the frame waiting for it. A coarse bake always finishes (the sky
    /// changing every frame would otherwise never show); the next starts from the sky as it is then, and
    /// the shown sky is blended from one coarse result to the next over about as many frames as a bake
    /// takes, as Frostbite spreads its sky tables over frames and blends between the last two.
    /// `sky_on`: the physical sky is what lights the scene (no environment picture in use; one still
    /// loading leaves the sky in place, as the engine's `sky_active` does).
    pub fn upkeep(&mut self, state: &State, sky_on: bool, sync: bool) -> (bool, bool) {
        self.frame += 1;
        let sky = Sky::of(&state.p);
        let want_active = sky_on && sky.needs_sky_space();
        if !want_active {
            self.drop_job();
            self.blend = None;
            if self.active {
                self.active = false;
                self.want.clear();
                self.sun_transmittance = 1.0;
                self.generation += 1;
                return (true, true);
            }
            return (false, false);
        }

        let want_key = sky.sky_space_key();
        if want_key != self.want {
            self.want = want_key;
            self.still_frames = 0;
            self.changed_at = Instant::now();
        } else {
            self.still_frames = self.still_frames.saturating_add(1);
        }
        let was = (self.active, self.shown);
        let mut changed = false;

        if sync {
            let done = self.active && self.shown == TOP && self.job.is_none() && self.slots[self.newest].as_ref().is_some_and(|s| s.key == self.want);
            if !done {
                self.drop_job();
                self.start(&sky, TOP, false);
                self.advance(f64::INFINITY);
                self.finish_now();
                changed = true;
            }
        } else {
            // a finer bake of a sky that has moved on since is not wanted; a coarse one always finishes
            if self.job.as_ref().is_some_and(|j| level_of_slot(j.slot) > 0 && j.key != self.want) {
                self.drop_job();
            }
            if self.job.is_none() {
                let newest_key = self.slots[self.newest].as_ref().map(|s| &s.key);
                let fresh = !self.active || newest_key != Some(&self.want);
                if let Some(level) = next_level(fresh, false, self.shown, self.still_frames, self.changed_at.elapsed()) {
                    self.start(&sky, level, true);
                }
            }
            if self.job.is_some() {
                if self.active {
                    self.advance(SLICE_MS);
                    changed |= self.land();
                } else {
                    // nothing to show yet: the first (coarse) bake is made at once
                    self.advance(f64::INFINITY);
                    self.finish_now();
                    changed = true;
                }
            }
            changed |= self.blend_step();
        }
        if changed { self.generation += 1; }
        (changed, changed && was != (self.active, self.shown))
    }

    /// Give up the job under way (its panorama's copy, if sent, is waited for, so that its buffer can be
    /// used again).
    fn drop_job(&mut self) {
        if let Some(job) = self.job.take() {
            if let Some((ring, _)) = job.read {
                self.gpu.wait();
                self.ring[ring].unmap();
            }
        }
    }

    fn start(&mut self, sky: &Sky, level: usize, split: bool) {
        let slot = if level == 0 { free_ring_slot(self.prev, self.newest) } else { slot_of_level(level) };
        self.ensure_slot(slot);
        let b = &self.slots[slot].as_ref().unwrap().bake;
        let steps = plan(b.face_sz, b.shadow_sz, split);
        let costs = steps.iter().map(|s| step_ms(*s, match s { Step::Shadow { .. } => b.shadow_sz, Step::Face { .. } => b.face_sz })).collect();
        self.job = Some(Job {
            slot,
            key: self.want.clone(),
            cu: CloudU::of(sky, b.face_sz, b.shadow_sz),
            clouds_active: sky.clouds_active(),
            steps,
            costs,
            next: 0,
            started_frame: self.frame,
            read: None,
        });
    }

    /// Run the job's next steps (up to `budget` ms by the estimates); once every step has run, draw the
    /// panorama and send it back to be read. One submission.
    fn advance(&mut self, budget: f64) {
        let Some(job) = self.job.as_mut() else { return };
        if job.read.is_some() { return; }
        let slot = self.slots[job.slot].as_ref().unwrap();
        let b = &slot.bake;
        self.gpu.queue.write_buffer(&self.u_buf, 0, bytemuck::bytes_of(&job.cu));
        let n = if budget.is_finite() { steps_this_frame(&job.costs, job.next, budget) } else { job.steps.len() - job.next };
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("clouds bake") });
        for k in job.next..job.next + n {
            let t = self.split.then(Instant::now);
            if self.split { enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("clouds split") }); }
            match job.steps[k] {
                Step::Shadow { y0, y1 } => {
                    let mut rp = pass(&mut enc, &b.shadow_view, None, false);
                    rp.set_pipeline(&self.p_shadow);
                    rp.set_bind_group(0, &self.bg_shadow, &[]);
                    rp.set_scissor_rect(0, y0, b.shadow_sz, y1 - y0);
                    rp.draw(0..3, 0..1);
                }
                Step::Face { face, y0, y1 } => {
                    let mut rp = pass(&mut enc, &b.dome_faces[face as usize], None, false);
                    rp.set_pipeline(&self.p_dome);
                    rp.set_bind_group(0, &b.bg_dome, &[]);
                    rp.set_scissor_rect(0, y0, b.face_sz, y1 - y0);
                    rp.draw(0..3, face..face + 1);
                }
            }
            if let Some(t) = t {
                let e = std::mem::replace(&mut enc, self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None }));
                self.gpu.queue.submit([e.finish()]);
                self.gpu.wait();
                eprintln!("SKYSPLIT {:?} size {} ms {:.2} est {:.2}", job.steps[k], b.face_sz, t.elapsed().as_secs_f64() * 1e3, job.costs[k]);
            }
        }
        job.next += n;
        if job.next == job.steps.len() {
            let (pano_tex, pano_view) = self.pano_tex.as_ref().unwrap();
            {
                let mut rp = pass(&mut enc, pano_view, None, true);
                rp.set_pipeline(&self.p_pano);
                rp.set_bind_group(0, &b.bg_pano, &[]);
                rp.draw(0..3, 0..1);
            }
            let (pano_row, pano_bytes, _) = read_layout();
            let buf = &self.ring[self.ring_next];
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: pano_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo {
                    buffer: buf,
                    layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(pano_row), rows_per_image: Some(PANO_H as u32) },
                },
                wgpu::Extent3d { width: PANO_W as u32, height: PANO_H as u32, depth_or_array_layers: 1 },
            );
            let mid = (b.shadow_sz / 2).saturating_sub(1);
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: &b.shadow_tex, mip_level: 0, origin: wgpu::Origin3d { x: mid, y: mid, z: 0 }, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo {
                    buffer: buf,
                    layout: wgpu::TexelCopyBufferLayout { offset: pano_bytes, bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT), rows_per_image: Some(2) },
                },
                wgpu::Extent3d { width: 2, height: 2, depth_or_array_layers: 1 },
            );
            self.gpu.queue.submit([enc.finish()]);
            let done = Arc::new(AtomicU8::new(READ_PENDING));
            let flag = done.clone();
            buf.slice(..).map_async(wgpu::MapMode::Read, move |r| flag.store(if r.is_ok() { READ_OK } else { READ_FAILED }, Ordering::SeqCst));
            job.read = Some((self.ring_next, done));
            self.ring_next = (self.ring_next + 1) % READ_RING;
        } else {
            self.gpu.queue.submit([enc.finish()]);
        }
    }

    /// Wait for the job's panorama and take the bake into use.
    fn finish_now(&mut self) {
        if self.job.as_ref().is_some_and(|j| j.read.is_some()) {
            self.gpu.wait();
        }
        self.land();
        // (a first coarse bake shows at once: nothing to blend from)
        self.blend_step();
    }

    /// If the job's panorama has come back, read it and show the bake. True if it did. A copy that could
    /// not be read is dropped with its job, and nothing of it is kept (not even what it was made for), so
    /// the next frame bakes again rather than leaving a black or stale panorama in place.
    fn land(&mut self) -> bool {
        let Some((ring, done)) = self.job.as_ref().and_then(|j| j.read.clone()) else { return false };
        let _ = self.gpu.device.poll(wgpu::PollType::Poll);
        let state = done.load(Ordering::SeqCst);
        if state == READ_PENDING { return false; }
        let job = self.job.take().unwrap();
        let buf = &self.ring[ring];
        let got = if state == READ_OK {
            let got = buf.slice(..).get_mapped_range().ok().and_then(|m| take_readback(READ_OK, &m, job.clouds_active));
            buf.unmap();
            got
        } else {
            None
        };
        let Some((pano, sun_t)) = got else {
            // (its textures may be half made over what the slot held: it holds nothing now)
            if let Some(s) = self.slots[job.slot].as_mut() { s.key.clear(); }
            return false;
        };
        let slot = self.slots[job.slot].as_mut().unwrap();
        slot.key = job.key;
        slot.pano = pano;
        slot.sun_t = sun_t;
        self.bake_count += 1;
        let level = level_of_slot(job.slot);
        if level == 0 {
            if self.active && self.shown == 0 {
                // blended in from the newest so far, over as many frames as this bake took
                self.prev = self.newest;
                let frames = (self.frame + 1 - job.started_frame).max(1) as f64;
                self.blend = Some((0.0, 1.0 / frames));
            } else {
                self.prev = job.slot;
                self.blend = Some((1.0, 1.0));
            }
            self.newest = job.slot;
        } else {
            self.newest = job.slot;
            self.blend = None;
            let s = self.slots[job.slot].as_ref().unwrap();
            self.pano_rgba.clone_from(&s.pano);
            self.sun_transmittance = s.sun_t;
        }
        self.shown = level;
        self.active = true;
        true
    }

    /// One more step of blending the shown coarse sky towards the newest bake: the dome, the shadow map,
    /// the panorama and the sun's transmittance. Done (an exact copy of the newest) at the end. True if
    /// anything changed.
    fn blend_step(&mut self) -> bool {
        let Some((t, step)) = self.blend else { return false };
        if self.shown != 0 { self.blend = None; return false; }
        let t = (t + step).min(1.0);
        self.blend = if t >= 1.0 { None } else { Some((t, step)) };
        if self.display.is_none() { self.display = Some(self.make_bake(0)); }
        let d = self.display.as_ref().unwrap();
        let a = self.slots[self.prev].as_ref().unwrap();
        let b = self.slots[self.newest].as_ref().unwrap();
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("clouds blend") });
        if t >= 1.0 {
            let whole = |size, n| wgpu::Extent3d { width: size, height: size, depth_or_array_layers: n };
            enc.copy_texture_to_texture(b.bake.dome_tex.as_image_copy(), d.dome_tex.as_image_copy(), whole(d.face_sz, 6));
            enc.copy_texture_to_texture(b.bake.shadow_tex.as_image_copy(), d.shadow_tex.as_image_copy(), whole(d.shadow_sz, 1));
        } else {
            self.gpu.queue.write_buffer(&self.blend_buf, 0, bytemuck::cast_slice(&[t as f32, 0.0, 0.0, 0.0]));
            let group = |x: &wgpu::TextureView, y: &wgpu::TextureView| {
                self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("clouds: blend"),
                    layout: &self.l_blend,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.blend_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(x) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(y) },
                    ],
                })
            };
            let domes = group(&a.bake.dome_array, &b.bake.dome_array);
            for face in 0..6u32 {
                let mut rp = pass(&mut enc, &d.dome_faces[face as usize], None, false);
                rp.set_pipeline(&self.p_blend);
                rp.set_bind_group(0, &domes, &[]);
                rp.draw(0..3, face..face + 1);
            }
            let shadows = group(&a.bake.shadow_array, &b.bake.shadow_array);
            let mut rp = pass(&mut enc, &d.shadow_view, None, false);
            rp.set_pipeline(&self.p_blend);
            rp.set_bind_group(0, &shadows, &[]);
            rp.draw(0..3, 0..1);
        }
        self.gpu.queue.submit([enc.finish()]);
        lerp_into(&mut self.pano_rgba, &a.pano, &b.pano, t);
        self.sun_transmittance = if t >= 1.0 { b.sun_t } else { a.sun_t + (b.sun_t - a.sun_t) * t };
        true
    }
}

const READ_PENDING: u8 = 0;
const READ_OK: u8 = 1;
const READ_FAILED: u8 = 2;

/// What came back of a bake: the panorama (RGBA f32, negatives and non-numbers made 0) and the sun's
/// transmittance (the 2x2 middle of the shadow map). None unless the copy was read (`state`).
pub fn take_readback(state: u8, mapped: &[u8], clouds_active: bool) -> Option<(Vec<f32>, f64)> {
    use rayon::prelude::*;
    let (pano_row, pano_bytes, size) = read_layout();
    if state != READ_OK || (mapped.len() as u64) < size { return None; }
    let mut pano = vec![0.0f32; PANO_W * PANO_H * 4];
    let row_floats = PANO_W * 4;
    pano.par_chunks_mut(row_floats).enumerate().for_each(|(r, dst)| {
        let src_off = r * pano_row as usize;
        let halfs: &[half::f16] = bytemuck::cast_slice(&mapped[src_off..src_off + row_floats * 2]);
        for (d, h) in dst.iter_mut().zip(halfs.iter()) {
            let v = h.to_f32();
            *d = if v.is_finite() && v > 0.0 { v } else { 0.0 };
        }
    });
    // Average the 2x2 center penumbra transmittance (.g) around uv = (0.5, 0.5).
    let mut t_sum = 0.0f64;
    for ry in 0..2usize {
        let off = pano_bytes as usize + ry * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let halfs: &[half::f16] = bytemuck::cast_slice(&mapped[off..off + 16]);
        for tx in 0..2usize {
            let soft_t = halfs[tx * 4 + 1].to_f32();
            t_sum += if soft_t.is_finite() { soft_t.clamp(0.0, 1.0) as f64 } else { 1.0 };
        }
    }
    let sun_t = if clouds_active { (t_sum * 0.25).clamp(0.0, 1.0) } else { 1.0 };
    Some((pano, sun_t))
}

/// The readback buffer: bytes a row of the panorama takes, where the shadow map's 2x2 middle goes, and
/// its whole size.
fn read_layout() -> (u32, u64, u64) {
    let pano_row = (PANO_W as u32 * 8).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let pano_bytes = pano_row as u64 * PANO_H as u64;
    (pano_row, pano_bytes, pano_bytes + wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64 * 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_climb_as_the_sky_holds_still() {
        let ms = Duration::from_millis;
        // a change: coarse while live, finest at once for a still
        assert_eq!(next_level(true, false, TOP, 0, ms(0)), Some(0));
        assert_eq!(next_level(true, true, 0, 0, ms(0)), Some(TOP));
        // the middle bake after a couple of still frames
        assert_eq!(next_level(false, false, 0, 1, ms(500)), None);
        assert_eq!(next_level(false, false, 0, MID_AFTER_FRAMES, ms(0)), Some(1));
        // the finest after both enough frames and enough time
        assert_eq!(next_level(false, false, 1, TOP_AFTER_FRAMES, ms(100)), None);
        assert_eq!(next_level(false, false, 1, 1, ms(900)), None);
        assert_eq!(next_level(false, false, 1, TOP_AFTER_FRAMES, TOP_AFTER), Some(TOP));
        // nothing once the finest is shown; a still made while a coarse bake is shown gets the finest
        assert_eq!(next_level(false, false, TOP, 99, ms(9000)), None);
        assert_eq!(next_level(false, true, 1, 0, ms(0)), Some(TOP));
        // the finest is the bake there always was
        assert_eq!(LEVELS[TOP], (DOME_FACE, SHADOW_SIZE));
    }

    #[test]
    fn a_split_bake_covers_every_row_once_in_order() {
        for (face_sz, shadow_sz) in LEVELS {
            for split in [false, true] {
                let steps = plan(face_sz, shadow_sz, split);
                // the shadow map first (the dome reads it), then the faces in turn
                let first_face = steps.iter().position(|s| matches!(s, Step::Face { .. })).unwrap();
                assert!(steps[..first_face].iter().all(|s| matches!(s, Step::Shadow { .. })));
                let mut rows = vec![0u32; 7];
                let mut last_face = 0;
                for s in &steps {
                    match *s {
                        Step::Shadow { y0, y1 } => rows[6] += y1 - y0,
                        Step::Face { face, y0, y1 } => {
                            assert!(face >= last_face);
                            last_face = face;
                            rows[face as usize] += y1 - y0;
                        }
                    }
                }
                assert!(rows[..6].iter().all(|r| *r == face_sz), "{rows:?}");
                assert_eq!(rows[6], shadow_sz);
                if !split { assert_eq!(steps.len(), 7); }
            }
        }
        assert_eq!(bands(1280, 4), vec![(0, 320), (320, 640), (640, 960), (960, 1280)]);
        assert_eq!(bands(10, 3), vec![(0, 4), (4, 8), (8, 10)]);
    }

    #[test]
    fn a_frame_takes_at_least_one_step_then_what_the_budget_allows() {
        let costs = [3.0, 1.0, 1.0, 5.0, 0.5];
        assert_eq!(steps_this_frame(&costs, 0, 4.0), 2);
        assert_eq!(steps_this_frame(&costs, 2, 4.0), 1);
        // a step dearer than the budget still runs, alone
        assert_eq!(steps_this_frame(&costs, 3, 4.0), 1);
        assert_eq!(steps_this_frame(&costs, 4, 4.0), 1);
        assert_eq!(steps_this_frame(&costs, 5, 4.0), 0);
        assert_eq!(steps_this_frame(&costs, 0, f64::INFINITY), 5);
        // the estimates: the dearest step of the finest bake is within reach of a frame's budget
        let (f, s) = LEVELS[TOP];
        let worst = plan(f, s, true).iter().map(|st| step_ms(*st, match st { Step::Shadow { .. } => s, _ => f })).fold(0.0, f64::max);
        assert!(worst <= 4.0 * SLICE_MS, "{worst}");
        // the coarse bake, split, is the seven passes of the whole bake
        assert_eq!(plan(LEVELS[0].0, LEVELS[0].1, true), plan(LEVELS[0].0, LEVELS[0].1, false));
    }

    #[test]
    fn the_ring_never_hands_out_a_slot_in_use() {
        for prev in 0..RING_SLOTS {
            for newest in 0..RING_SLOTS {
                let free = free_ring_slot(prev, newest);
                assert!(free < RING_SLOTS && free != prev && free != newest);
            }
        }
        assert_eq!(level_of_slot(slot_of_level(1)), 1);
        assert_eq!(level_of_slot(slot_of_level(TOP)), TOP);
        assert!((0..RING_SLOTS).all(|s| level_of_slot(s) == 0));
    }

    #[test]
    fn a_readback_that_failed_or_is_short_gives_nothing() {
        let (_, pano_bytes, size) = read_layout();
        let mut bytes = vec![0u8; size as usize];
        // a sky of 1.0 (half float 0x3c00) everywhere, and a transmittance of 0.5 (0x3800) in the middle
        for c in bytes[..pano_bytes as usize].chunks_mut(2) { c.copy_from_slice(&0x3c00u16.to_le_bytes()); }
        for ry in 0..2usize {
            for tx in 0..2usize {
                let o = pano_bytes as usize + ry * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize + tx * 8 + 2;
                bytes[o..o + 2].copy_from_slice(&0x3800u16.to_le_bytes());
            }
        }
        let (pano, sun_t) = take_readback(READ_OK, &bytes, true).unwrap();
        assert!(pano.len() == PANO_W * PANO_H * 4 && pano.iter().all(|v| *v == 1.0));
        assert_eq!(sun_t, 0.5);
        assert_eq!(take_readback(READ_OK, &bytes, false).unwrap().1, 1.0);
        assert!(take_readback(READ_FAILED, &bytes, true).is_none());
        assert!(take_readback(READ_PENDING, &bytes, true).is_none());
        assert!(take_readback(READ_OK, &bytes[..100], true).is_none());
    }

    #[test]
    fn blending_ends_on_the_newest_exactly() {
        let a: Vec<f32> = (0..10_000).map(|k| (k as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..10_000).map(|k| (k as f32 * 0.37).cos() * 3.0).collect();
        let mut out = Vec::new();
        lerp_into(&mut out, &a, &b, 0.0);
        assert_eq!(out, a);
        lerp_into(&mut out, &a, &b, 0.5);
        assert!(out.iter().zip(a.iter().zip(&b)).all(|(o, (a, b))| (o - (a + b) * 0.5).abs() < 1e-5));
        lerp_into(&mut out, &a, &b, 1.0);
        assert_eq!(out, b);
    }
}
