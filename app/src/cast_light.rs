//! Caustics: light followed forward from the lights through the bodies that bend or reflect it, landed
//! on the bodies that gather light (and on a floor, if there is one), and kept as a volume of arriving
//! light in the object's frame (and a map of the floor's plane) that a shaded surface reads.
//! (How and why: shaders/cast_light.wgsl, cast_land.wgsl, cast_resolve.wgsl.)
//!
//! Made again whenever what it depends on changes: the shape, a light, how the lights stand to the
//! object, a casting body's material, where the floor is. Not when only the view changes.

use bytemuck::{Pod, Zeroable};

use gf_core::cast::{self as core, Caster, Kind, MAX_BODIES};

use crate::gpu::*;

/// The shader that follows the light and lands it (two files, one shader).
pub const LIGHT_WGSL: &str = concat!(
    include_str!("shaders/pbr/hash.wgsl"),
    include_str!("shaders/pcg.wgsl"),
    include_str!("shaders/ggx_microsurface.wgsl"),
    include_str!("shaders/conductor_fresnel.wgsl"),
    include_str!("shaders/pbr/types.wgsl"),
    include_str!("shaders/pbr/dielectric.wgsl"),
    include_str!("shaders/dielectric_fresnel.wgsl"),
    include_str!("shaders/cast_light.wgsl"),
    include_str!("shaders/cast_land.wgsl"),
);
const RESOLVE_WGSL: &str = include_str!("shaders/cast_resolve.wgsl");

/// The shader that turns landed light into arriving light. `storage16`: a compute shader can write a
/// volume of 16-bit floats on this device (else the areas are kept in 32-bit ones).
pub fn resolve_source(storage16: bool) -> String {
    // (the box the volumes span is told, not fixed: it grows when the layers are placed far out)
    RESOLVE_WGSL.replace("{{AREA}}", if storage16 { "r16float" } else { "r32float" }).replace("const HB: f32 = 1.5;", "").replace("HB", "ru.k.z")
}

/// The shader that follows the lights' rays and lands their light, with the box the volumes span told
/// rather than fixed (lu.body[0].tint.w: it grows when the layers are placed far out).
pub fn light_source(medium_priority: bool) -> String {
    let source = if medium_priority { medium_priority_cast_wgsl(LIGHT_WGSL) } else { LIGHT_WGSL.to_owned() };
    source
        .replace("const HB: f32 = 1.5;", "")
        .replace("const BOUND: f32 = 1.47;", "")
        .replace("BOUND", "(lu.body[0].tint.w * 0.98)")
        .replace("HB", "lu.body[0].tint.w")
}

/// The forward-caustic ownership sites, specialized only when active bodies have different priorities.
/// Optical events, weights, wavelengths, offsets and absorption remain the original source in both variants.
pub fn medium_priority_cast_wgsl(src: &str) -> String {
    fn one(src: &mut String, from: &str, to: &str) {
        assert_eq!(src.matches(from).count(), 1, "medium-priority caustic site changed: {from}");
        *src = src.replacen(from, to, 1);
    }
    let mut out = src.replace("\r\n", "\n");
    one(&mut out,
r#"fn within(a: vec4f, m: i32) -> vec4f {
  let i = vec4i(0, 1, 2, 3);
  return select(select(a, -a, i == vec4i(m)), vec4f(BIG), i < vec4i(m));
}"#,
r#"fn castPrecedes(k: i32, m: i32) -> bool {
  if (k == m) { return false; }
  if (m < 0) { return true; }
  let a = lu.body[k].medium.y;
  let b = lu.body[m].medium.y;
  return a > b || (a == b && k > m);
}
fn within(a: vec4f, m: i32) -> vec4f {
  let i = vec4i(0, 1, 2, 3);
  let own = select(a, -a, i == vec4i(m));
  var visible = vec4<bool>(false);
  for (var k = 0; k < 4; k++) { visible[k] = k == m || castPrecedes(k, m); }
  return select(own, vec4f(BIG), !visible);
}
fn castWhich(a: vec4f, f: f32) -> i32 {
  var which = -1;
  for (var k = 0; k < 4; k++) {
    if (a[k] == f && (which < 0 || castPrecedes(k, which))) { which = k; }
  }
  return which;
}"#);
    one(&mut out,
"      *which = select(select(select(3, 2, a.z == f), 1, a.y == f), 0, a.x == f);",
"      *which = castWhich(a, f);");
    one(&mut out,
r#"      next = -1;
      let dd = dist(p) + lu.gone1;
      for (var k = 0; k < 4; k++) {
        if (k < m && dd[k] < 0.0) { next = k; }
      }"#,
r#"      next = -1;
      let dd = dist(p) + lu.gone1;
      for (var k = 0; k < 4; k++) {
        if (k != m && dd[k] < 0.0 && (next < 0 || castPrecedes(k, next))) { next = k; }
      }"#);
    out
}

/// Rays a side of the grid sent from one light at most: when the picture is at rest, and while things move.
pub const RAYS_REST: u32 = 768;
pub const RAYS_MOVING: u32 = 384;
/// Rays sent from all the lights together at most, as so many grids of the largest size: the lights
/// share them by how much light each sends.
const GRIDS_REST: f64 = 3.0;
const GRIDS_MOVING: f64 = 2.0;
/// The least for any light.
const RAYS_LEAST: u32 = 96;
/// Wavelengths followed when a body splits colours: the two ends of the spectrum. A colour between them
/// lands as far between where they landed as its bending lies between theirs (to the first order in how
/// much the body spreads the spectrum, which is exact enough for anything this program calls glass).
const WAVES: u32 = 2;
/// Bytes one wavelength of one ray takes where its landing is kept.
const HIT_BYTES: u64 = 24;
/// Bytes between one level's block and the next.
const BLOCK: u64 = 256;
/// How far off a surface light is landed, in voxels of the finest volume: far enough that the two faces
/// of a thin wall do not share their light.
pub const OFF_SURFACE: f64 = 1.0;
/// Light whose neighbouring rays went elsewhere is landed no finer than this level (softened over some
/// eight voxels of the finest volume): where rays part company the light is thin and scattered, and
/// landed any finer each ray shows as a speck.
const LONE_LEVEL: f32 = 3.0;
/// A floor's finest map reaches at least this far from the middle (each further map twice as far, all with
/// the same texels a side, until the last reaches as far as the floor's light is kept).
const FLOOR_NEAR: f64 = 1.5;
/// The coarsest of a floor's maps (for light spread wide) is this many texels a side.
const FLOOR_LEAST: u32 = 128;
/// Makings that refine a still picture: while the picture is still and averaging frames, each of its first
/// so many frames is drawn with the caustics made again, every grid of rays moved by a different part of a
/// cell and the floor's maps by a different part of a texel. The picture's own average of those frames is
/// then the caustic sampled that many times finer, with no grid and no texels left in it.
pub const REFINE: u32 = 32;

/// How many maps a floor has whose light is kept out to `half` from the middle, and half the width of the
/// finest of them.
pub fn floor_levels(half: f64) -> (u32, f64) {
    let n = (1.0 + (half / FLOOR_NEAR).log2().floor()).clamp(1.0, 4.0) as u32;
    (n, half / (1u32 << (n - 1)) as f64)
}

/// Halton's sequence: a part of 1 that spreads evenly over the makings that refine.
fn halton(mut i: u32, b: u32) -> f64 {
    let (mut f, mut r) = (1.0, 0.0);
    while i > 0 {
        f /= b as f64;
        r += f * (i % b) as f64;
        i /= b;
    }
    r
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq)]
struct BodyL {
    a: [f32; 4],
    absorb: [f32; 4],
    tint: [f32; 4],
    mask: [f32; 4],
    surface: [f32; 4],
    medium: [i32; 4],
    optical_constants: [[f32; 4]; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq)]
struct LightsU {
    body: [BodyL; 4],
    spectral_rgb: [[f32; 4]; 16],
    k: [f32; 4],
    lev: [f32; 4],
    soft: [f32; 4],
    gone0: [f32; 4],
    gone1: [f32; 4],
    fl: [[f32; 4]; 4],
    bands: [[f32; 4]; 16],
    grid: [f32; 4],
}

/// One light's rays (see cast_light.wgsl: `One`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Debug)]
pub struct OneU {
    /// towards the light in the object's frame; the tangent of the angle it is taken to cover
    pub dir: [f32; 4],
    /// the two ways across the grid of rays; half the grid's width, and rays a side
    pub u: [f32; 4],
    pub v: [f32; 4],
    /// the brightness of a white surface facing the light; where its rays' landings begin among all the
    /// lights' (filled in when the grids are laid out)
    pub rgb: [f32; 4],
    /// a lamp: where it stands in the object's frame, and 1 (its rays spread from there); 0 for a light
    /// at infinity
    pub at: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq)]
struct ResolveU {
    recv: [[f32; 4]; 4],
    gone: [f32; 4],
    k: [f32; 4],
    ku: [u32; 4],
}

/// A floor under the shape, as the light meets it: in the object's own frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloorIn {
    /// a point of it (the middle of its map), the way up from it, and the two ways along it
    pub origin: [f64; 3],
    pub up: [f64; 3],
    pub u: [f64; 3],
    pub v: [f64; 3],
    /// half the width of its map of arriving light, and texels a side of the finest map
    pub half: f64,
    pub res: u32,
}

/// The scene as the light is followed through it.
pub struct Scene<'a> {
    /// the distance to each body at full detail: the bodies' volume, or with one body the scene's
    pub fine: &'a wgpu::TextureView,
    pub fine_res: u32,
    /// how many bodies (channel k of the volume is body k)
    pub count: u32,
    /// a number that changes whenever the volume is built again
    pub built: u64,
    pub casters: [Caster; MAX_BODIES],
    /// how the tracer marches: a stride as a share of the distance read, strides at most outside and inside
    pub march: (f64, u32, u32),
    /// voxels a side of the finest volume of arriving light
    pub res: u32,
    /// the distance to each body at modest detail, one body a channel (cast.rs), and its voxels a side
    pub coarse: &'a wgpu::TextureView,
    pub coarse_res: u32,
    pub floor: Option<FloorIn>,
    /// the volumes span [-hb, hb] (larger than ever when the layers are placed far out)
    pub hb: f64,
}

/// What was last made, and from what.
#[derive(PartialEq)]
struct Made {
    lights: Vec<OneU>,
    u: LightsU,
    fine: wgpu::TextureView,
    coarse: wgpu::TextureView,
    built: u64,
}

struct Level {
    res: u32,
    /// where its numbers begin in the buffer of landed light
    base: u32,
    light: wgpu::TextureView,
    /// None for a level of the floor's maps (a floor's texel holds a texel of floor: nothing to work out)
    g_area: Option<wgpu::BindGroup>,
    g_resolve: wgpu::BindGroup,
}

pub struct Caustics {
    gpu: Gpu,
    smp: wgpu::Sampler,
    l_follow: wgpu::BindGroupLayout,
    l_ru: wgpu::BindGroupLayout,
    l_area: wgpu::BindGroupLayout,
    l_resolve: wgpu::BindGroupLayout,
    l_floor: wgpu::BindGroupLayout,
    p_follow: wgpu::ComputePipeline,
    p_land: wgpu::ComputePipeline,
    p_area: wgpu::ComputePipeline,
    p_resolve: wgpu::ComputePipeline,
    p_floor: wgpu::ComputePipeline,
    p_bound: wgpu::ComputePipeline,
    medium_priority: bool,
    /// the box round the bodies that start a caustic (cast_light.wgsl: `bound`)
    bound: wgpu::Buffer,
    u_lights: wgpu::Buffer,
    u_one: wgpu::Buffer,
    u_resolve: wgpu::Buffer,
    area_format: wgpu::TextureFormat,
    /// where each ray of every light landed, and how many landings it has room for
    hits: Option<(wgpu::Buffer, u64)>,
    /// landed light: the volumes' levels one after another, then the floor's maps
    acc: Option<wgpu::Buffer>,
    levels: Vec<Level>,
    floor: Vec<Level>,
    /// the floor's maps as one texture of layers, as the tracer reads them; how many maps there are of
    /// those (the floor's other maps, each as wide as the last of them and half as fine, follow them in
    /// `floor`)
    floor_all: Option<wgpu::TextureView>,
    floor_clip: u32,
    blank: wgpu::TextureView,
    blank2: wgpu::TextureView,
    /// what the areas were worked out from: the volume, its build, which bodies gather light
    area_from: Option<(wgpu::TextureView, u64, ResolveU)>,
    /// what the box round the bodies that start a caustic was found from: the coarse volume, its build,
    /// what each body does to light not yet bent
    bound_from: Option<(wgpu::TextureView, u64, [f32; 4])>,
    made: Option<Made>,
    /// the last making was the quick one (fewer rays), and something changed in the frame before this
    quick: bool,
    moved: bool,
    /// which making that refines a still picture the last one was (0: none), and how far the floor's maps were
    /// moved for it (along the floor's two ways; the tracer must read them moved the same)
    pub pass: u32,
    pub shift: [f64; 2],
    /// how many rays the last making sent, how many makings there have been, and (when makings are
    /// timed) the milliseconds the last one took to the end
    pub rays: u64,
    pub runs: u64,
    pub ms: f64,
}

fn storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None }
}
fn storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None }
}
fn tex(binding: u32, dim: wgpu::TextureViewDimension) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: dim, multisampled: false }, count: None }
}
fn out(binding: u32, format: wgpu::TextureFormat, dim: wgpu::TextureViewDimension) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format, view_dimension: dim }, count: None }
}
fn smp_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None }
}

/// How many volumes of arriving light there are for a finest one of this size (each half the one before,
/// down to 16 voxels a side).
pub fn level_count(res: u32) -> u32 {
    let mut n = 1;
    while (res >> n) >= 16 && n < 6 { n += 1; }
    n
}

/// Parts of a making left out, for finding what each costs (`GLASSFORMS_CAST_SKIP=follow,land`: see
/// tools/cast_bench.py). Read once.
fn skipped() -> &'static str {
    static SKIP: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SKIP.get_or_init(|| std::env::var("GLASSFORMS_CAST_SKIP").unwrap_or_default())
}

fn follow_pipelines(gpu: &Gpu, layout: &wgpu::BindGroupLayout, medium_priority: bool)
    -> (wgpu::ComputePipeline, wgpu::ComputePipeline, wgpu::ComputePipeline)
{
    let light = gpu.shader("cast: light", &light_source(medium_priority), false);
    let pipe = |label, entry| {
        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label), bind_group_layouts: &[Some(layout)], immediate_size: 0,
        });
        gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label), layout: Some(&pipeline_layout), module: &light,
            entry_point: Some(entry), compilation_options: Default::default(), cache: None,
        })
    };
    (pipe("cast: follow", "cs_follow"), pipe("cast: land", "cs_land"), pipe("cast: bound", "cs_bound"))
}

impl Caustics {
    pub fn new(gpu: &Gpu) -> Self {
        let c = wgpu::ShaderStages::COMPUTE;
        let (d2, d3) = (wgpu::TextureViewDimension::D2, wgpu::TextureViewDimension::D3);
        let rgba = wgpu::TextureFormat::Rgba16Float;
        let l_follow = gpu.bind_layout("cast: follow", &[uniform_entry(0, c, false), uniform_entry(1, c, false), tex(2, d3), smp_entry(3), storage_rw(4), storage_rw(5), tex(6, d3), storage_rw(7)]);
        let l_ru = gpu.bind_layout("cast: level", &[uniform_entry(0, c, true), tex(1, d3), smp_entry(2)]);
        // a volume of one number a voxel: 16-bit floats where a compute shader can write those, else 32-bit
        let area_format = if gpu.storage16 { wgpu::TextureFormat::R16Float } else { wgpu::TextureFormat::R32Float };
        let l_area = gpu.bind_layout("cast: area", &[out(0, area_format, d3)]);
        let l_resolve = gpu.bind_layout("cast: resolve", &[storage_ro(1), tex(2, d3), tex(3, d3), out(4, rgba, d3)]);
        let l_floor = gpu.bind_layout("cast: floor", &[storage_ro(1), tex(5, d2), out(6, rgba, d2)]);
        let (p_follow, p_land, p_bound) = follow_pipelines(gpu, &l_follow, false);
        let resolve = gpu.shader("cast: resolve", &resolve_source(gpu.storage16), false);
        let pipe = |label, module: &wgpu::ShaderModule, entry, layouts: &[Option<&wgpu::BindGroupLayout>]| {
            let layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: layouts, immediate_size: 0 });
            gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: Some(&layout), module, entry_point: Some(entry), compilation_options: Default::default(), cache: None })
        };
        let blank = gpu.volume("cast: nothing", 1, 1, rgba, false).create_view(&wgpu::TextureViewDescriptor::default());
        let blank2 = gpu.target("cast: nothing", 1, 1, rgba, &[]).create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            gpu: gpu.clone(),
            smp: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge),
            p_follow,
            p_land,
            p_area: pipe("cast: area", &resolve, "cs_area", &[Some(&l_ru), Some(&l_area)]),
            p_resolve: pipe("cast: resolve", &resolve, "cs_resolve", &[Some(&l_ru), Some(&l_resolve)]),
            p_floor: pipe("cast: floor", &resolve, "cs_floor", &[Some(&l_ru), Some(&l_floor)]),
            p_bound,
            medium_priority: false,
            bound: gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("cast: bound"), size: 32, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
            l_follow, l_ru, l_area, l_resolve, l_floor,
            u_lights: gpu.uniform_buffer("cast: lights", std::mem::size_of::<LightsU>() as u64),
            u_one: gpu.uniform_buffer("cast: the lights' rays", (std::mem::size_of::<OneU>() * core::MAX_CAUSTIC_LIGHTS) as u64),
            u_resolve: gpu.uniform_buffer("cast: levels", BLOCK * 16),
            area_format,
            hits: None, acc: None, levels: Vec::new(), floor: Vec::new(), floor_all: None, floor_clip: 0, blank, blank2,
            area_from: None, bound_from: None, made: None, quick: false, moved: false, pass: 0, shift: [0.0; 2], rays: 0, runs: 0, ms: 0.0,
        }
    }

    /// Select the compile-time forward-ownership variant and invalidate only its dependent light results.
    pub fn set_medium_priority(&mut self, enabled: bool) {
        if self.medium_priority == enabled { return; }
        (self.p_follow, self.p_land, self.p_bound) = follow_pipelines(&self.gpu, &self.l_follow, enabled);
        self.medium_priority = enabled;
        self.clear();
        self.bound_from = None;
    }

    /// The volume of arriving light a tracer reads (None until one has been made).
    pub fn view(&self) -> Option<&wgpu::TextureView> {
        self.levels.first().map(|l| &l.light)
    }
    /// The maps of light arriving on the floor, as layers of one texture (None while there is no floor).
    pub fn floor_view(&self) -> Option<&wgpu::TextureView> {
        self.floor_all.as_ref()
    }

    /// Voxels a side of the finest volume, and video memory held in all (bytes).
    pub fn size(&self) -> (u32, u64) {
        let res = self.levels.first().map_or(0, |l| l.res);
        let area = if self.area_format == wgpu::TextureFormat::R16Float { 2 } else { 4 };
        let volumes: u64 = self.levels.iter().map(|l| (l.res as u64).pow(3) * (12 + 8 + area)).sum();
        let maps: u64 = self.floor.iter().map(|l| (l.res as u64).pow(2) * (12 + 8)).sum();
        (res, volumes + maps + self.hits.as_ref().map_or(0, |h| h.0.size()))
    }

    /// Was the last making the quick kind (fewer rays, while things move)?
    pub fn was_quick(&self) -> bool {
        self.quick
    }

    /// Let go of everything made (nothing casts caustics any more).
    pub fn clear(&mut self) {
        self.levels.clear();
        self.floor.clear();
        self.floor_all = None;
        self.floor_clip = 0;
        self.acc = None;
        self.hits = None;
        self.made = None;
        self.area_from = None;
    }

    /// The volumes for a finest one of `res` voxels a side, and `nf` maps of the floor of `floor_res` texels
    /// a side (0: no floor): made if they are not there or another size. True if they were made (whoever
    /// reads the finest volume or the floor's maps must be given the new ones).
    fn size_to(&mut self, res: u32, floor_res: u32, nf: u32) -> bool {
        let nf = if floor_res > 0 { nf } else { 0 };
        if self.levels.first().is_some_and(|l| l.res == res) && self.floor.first().map_or(0, |l| l.res) == floor_res && self.floor_clip == nf { return false; }
        let gpu = &self.gpu;
        let n = level_count(res);
        // the floor's maps: `nf` of `floor_res` texels a side, each twice as wide as the one before; then, as
        // wide as the last, maps each half as fine (for light spread wide), down to FLOOR_LEAST texels
        let floor_sizes: Vec<u32> = if nf == 0 { Vec::new() } else {
            let mut v = vec![floor_res; nf as usize];
            let mut k = 1;
            while (floor_res >> k) >= FLOOR_LEAST && v.len() < 8 {
                v.push(floor_res >> k);
                k += 1;
            }
            v
        };
        let total: u64 = (0..n).map(|l| ((res >> l) as u64).pow(3) * 12).sum::<u64>() + floor_sizes.iter().map(|r| (*r as u64).pow(2) * 12).sum::<u64>();
        let acc = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("cast: landed light"), size: total, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let view = |t: wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let texs: Vec<(u32, wgpu::TextureView, wgpu::TextureView)> = (0..n)
            .map(|l| {
                let r = res >> l;
                (r, view(gpu.volume("cast: area", r, r, self.area_format, true)), view(gpu.volume("cast: light", r, r, wgpu::TextureFormat::Rgba16Float, true)))
            })
            .collect();
        let mut base = 0u32;
        self.levels = (0..n as usize)
            .map(|l| {
                let (r, area, light) = &texs[l];
                let coarser = texs.get(l + 1).map_or(&self.blank, |t| &t.2);
                let g_area = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("cast: area"), layout: &self.l_area, entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(area) }] });
                let g_resolve = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("cast: resolve"),
                    layout: &self.l_resolve,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 1, resource: acc.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(area) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(coarser) },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(light) },
                    ],
                });
                let level = Level { res: *r, base, light: light.clone(), g_area: Some(g_area), g_resolve };
                base += r * r * r * 3;
                level
            })
            .collect();
        // the floor's maps: layers of one 2D texture a compute shader can write (a layer at a time) and the
        // tracer can read (all of them)
        self.floor_all = None;
        let maps: Vec<(u32, wgpu::TextureView)> = if nf == 0 { Vec::new() } else {
            let t = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("cast: light on the floor"),
                size: wgpu::Extent3d { width: floor_res, height: floor_res, depth_or_array_layers: nf },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            self.floor_all = Some(t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() }));
            let mut maps: Vec<(u32, wgpu::TextureView)> = (0..nf)
                .map(|l| (floor_res, t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), base_array_layer: l, array_layer_count: Some(1), ..Default::default() })))
                .collect();
            for r in &floor_sizes[nf as usize..] {
                let t = gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("cast: light on the floor, spread"),
                    size: wgpu::Extent3d { width: *r, height: *r, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                maps.push((*r, view(t)));
            }
            maps
        };
        self.floor_clip = nf;
        self.floor = (0..maps.len())
            .map(|l| {
                let (r, light) = &maps[l];
                let coarser = maps.get(l + 1).map_or(&self.blank2, |t| &t.1);
                let g_resolve = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("cast: floor"),
                    layout: &self.l_floor,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 1, resource: acc.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(coarser) },
                        wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(light) },
                    ],
                });
                let level = Level { res: *r, base, light: light.clone(), g_area: None, g_resolve };
                base += r * r * 3;
                level
            })
            .collect();
        self.acc = Some(acc);
        self.area_from = None;
        self.bound_from = None;
        self.made = None;
        true
    }

    /// Follow the lights' rays through the scene and make the volume of arriving light (and the floor's
    /// map), if anything they depend on has changed since they were last made. `lights`: each light's
    /// rays (the number of rays a side is set here). `rest`: nothing is expected to change next frame (a
    /// still, a video frame), so the full number of rays is sent at once; otherwise the first making
    /// after a change sends fewer, and the full number follows when changes stop.
    /// `timed`: wait for the GPU before and after, so that `ms` is the whole of what the making took.
    /// `pass`: how many frames the still picture has averaged so far (0 while it moves): while nothing has
    /// changed, each of its first `REFINE` frames has the caustics made again, moved by a part of a cell
    /// (see `REFINE`).
    /// Returns (they were made again, the finest volume or the floor's maps are a new texture, the making
    /// was one that refines: the picture so far stays good).
    pub fn run(&mut self, scene: &Scene, lights: &[OneU], weights: &[f64], rest: bool, timed: bool, pass: u32) -> (bool, bool, bool) {
        let (nf, half0) = scene.floor.map_or((0, 0.0), |f| floor_levels(f.half));
        let fresh = self.size_to(scene.res, scene.floor.map_or(0, |f| f.res), nf);
        let res = scene.res;
        let h0 = 2.0 * scene.hb / res as f64;
        let n_levels = self.levels.len() as u32;

        // the bodies, as light meets them
        let mut u: LightsU = Zeroable::zeroed();
        let mut ru: ResolveU = Zeroable::zeroed();
        let mut waves = 1;
        // Same sixteen wavelengths as the camera's conductor integral. Each channel's
        // weights sum to one so a white, lossless reflector returns the incident energy.
        static WEIGHTS: std::sync::OnceLock<[[f32; 4]; 16]> = std::sync::OnceLock::new();
        u.spectral_rgb = *WEIGHTS.get_or_init(|| {
            let weights: [[f64; 3]; 16] = std::array::from_fn(|i| gf_core::material::spec_w(400.0 + 20.0 * i as f64));
            let sums: [f64; 3] = std::array::from_fn(|c| weights.iter().map(|v| v[c]).sum());
            weights.map(|w| [(w[0]/sums[0]) as f32, (w[1]/sums[1]) as f32, (w[2]/sums[2]) as f32, 0.0])
        });
        for k in 0..MAX_BODIES {
            let (mut before, mut after) = (0.0, 0.0);
            let mut mask = [0.0f32; 4];
            let mut receives = false;
            if (k as u32) < scene.count {
                let c = &scene.casters[k];
                let o = &c.optics;
                mask[k] = 1.0;
                receives = matches!(o.kind, Kind::Matte | Kind::Milky);
                let bends = if c.caustic { if o.kind == Kind::Metal { 2.0 } else { 1.0 } } else { 0.0 };
                // light that nothing has bent: a caustic begins at a body that bends or reflects; a body
                // that casts a shadow stops it; one that does not is not there
                before = if bends > 0.0 { bends } else if c.shadow { 3.0 } else { 0.0 };
                // light already bent: it lands on a body that gathers light, whether or not that body casts
                after = if bends > 0.0 { bends } else if receives { 4.0 } else if c.shadow { 3.0 } else { 0.0 };
                if c.caustic && o.kind == Kind::Clear && o.disp.abs() > 1e-6 { waves = WAVES; }
                u.body[k] = BodyL {
                    a: [o.ior as f32, o.disp as f32, before, after],
                    absorb: [o.absorb[0] as f32, o.absorb[1] as f32, o.absorb[2] as f32, o.rough as f32],
                    tint: [o.tint[0] as f32, o.tint[1] as f32, o.tint[2] as f32, 0.0],
                    mask,
                    surface: [o.coat_ior as f32, if o.conductor.is_some() { 1.0 } else { 0.0 }, 0.0, 0.0],
                    medium: c.medium.raw(),
                    optical_constants: o.conductor.map_or([[0.0; 4]; 8], |metal| std::array::from_fn(|pair| {
                        let i = 4 * pair;
                        [metal.n[i], metal.k[i], metal.n[i + 2], metal.k[i + 2]]
                    })),
                };
            }
            u.gone0[k] = if before == 0.0 { 1.0e3 } else { 0.0 };
            u.gone1[k] = if after == 0.0 { 1.0e3 } else { 0.0 };
            ru.recv[k] = if receives { mask } else { [0.0; 4] };
            ru.gone[k] = if receives { 0.0 } else { 1.0e3 };
        }
        // (strides at most: fewer than the tracer allows itself, since far from surfaces a ray here strides
        // through the small volume; a ray that still runs out is light lost, and a pass takes as long as
        // its slowest ray)
        u.k = [scene.fine_res as f32, scene.march.0 as f32, scene.march.1.min(160) as f32, scene.march.2.min(128) as f32];
        u.lev = [res as f32, n_levels as f32, waves as f32, (OFF_SURFACE * h0) as f32];
        // landed light is counted in whole numbers against the brightest light
        let brightest = lights.iter().map(|l| l.rgb[0].max(l.rgb[1]).max(l.rgb[2])).fold(0.0f32, f32::max).max(1e-20);
        u.soft = [LONE_LEVEL, 1.0 / brightest, scene.coarse_res as f32, 0.0];
        // (the box the volumes span, where nothing else is kept: it grows when the layers are placed far out)
        u.body[0].tint[3] = scene.hb as f32;
        // where each band's light lands between the ends of the spectrum: by its bending, which goes with
        // one over the wavelength squared
        let bend = |lam: f64| 1.0 / (lam * lam);
        if let Some(f) = &scene.floor {
            let row = |v: [f64; 3], w: f64| [v[0] as f32, v[1] as f32, v[2] as f32, w as f32];
            u.fl = [row(f.origin, 1.0), row(f.up, half0), row(f.u, f.res as f64), row(f.v, self.floor.len() as f64)];
            u.grid[3] = nf as f32;
        }
        for (i, b) in core::bands(16).iter().enumerate() {
            let lam = 400.0 + 300.0 * (i as f64 + 0.5) / 16.0;
            u.bands[i] = [b[0] as f32, b[1] as f32, b[2] as f32, ((bend(lam) - bend(400.0)) / (bend(700.0) - bend(400.0))) as f32];
        }
        ru.k = [(OFF_SURFACE * h0) as f32, brightest, scene.hb as f32, 0.0];

        // made already from exactly this? (the number of rays apart)
        let lights = &lights[..lights.len().min(core::MAX_CAUSTIC_LIGHTS)];
        let want = Made { lights: lights.iter().map(|l| OneU { v: [l.v[0], l.v[1], l.v[2], 0.0], rgb: [l.rgb[0], l.rgb[1], l.rgb[2], 0.0], ..*l }).collect(), u, fine: scene.fine.clone(), coarse: scene.coarse.clone(), built: scene.built };
        let same = self.made.as_ref() == Some(&want);
        let refine = same && !self.quick && (1..=REFINE).contains(&pass) && pass != self.pass;
        if same && !self.quick && !refine {
            self.moved = false;
            return (false, fresh, false);
        }
        // a making that refines: every grid moved by a part of a cell, the floor's maps by a part of a texel,
        // and the scramble that rounds landed light another (none for the first making after a change)
        let k = if refine { pass } else { 0 };
        let mut shift = [0.0; 2];
        if k > 0 {
            u.grid = [(halton(k, 2) - 0.5) as f32, (halton(k, 3) - 0.5) as f32, k as f32, u.grid[3]];
            if let Some(f) = &scene.floor {
                let texel = 2.0 * half0 / f.res as f64;
                shift = [(halton(k, 5) - 0.5) * texel, (halton(k, 7) - 0.5) * texel];
                for c in 0..3 {
                    u.fl[0][c] += (shift[0] * f.u[c] + shift[1] * f.v[c]) as f32;
                }
            }
        }
        // A change straight after a change: things are moving, and fewer rays are sent until they stop.
        // (Nothing changed but the last making was such a quick one: now it is made in full.)
        let quick = !same && !rest && self.moved;
        self.moved = !same;
        self.quick = quick;
        // (while things move, light is landed no finer on the floor than its second map: a quarter of the work)
        u.soft[3] = if quick && self.floor.len() > 1 { 1.0 } else { 0.0 };

        // Each light's grid. The lights share a number of rays by how much light each sends (a caustic's
        // fineness goes with rays a side, what it adds to the picture with the light), none more than
        // one full grid.
        let (most, grids) = if quick { (RAYS_MOVING, GRIDS_MOVING) } else { (RAYS_REST, GRIDS_REST) };
        let all: f64 = weights.iter().sum::<f64>().max(1e-30);
        // (each light's landings follow the one before's in the same buffer)
        let mut landings = 0u64;
        let lights: Vec<OneU> = lights
            .iter()
            .zip(weights)
            .map(|(l, w)| {
                let n = ((most as f64 * (grids * w / all).sqrt() / 8.0).round() as u32 * 8).clamp(RAYS_LEAST, most);
                let at = landings;
                landings += (n as u64).pow(2) * waves as u64;
                OneU { v: [l.v[0], l.v[1], l.v[2], n as f32], rgb: [l.rgb[0], l.rgb[1], l.rgb[2], at as f32], ..*l }
            })
            .collect();

        let gpu = self.gpu.clone();
        let Some(acc) = self.acc.as_ref() else { return (false, fresh, false) };
        // room for every light's landings (kept when fewer are needed: a turning view asks for fewer)
        let largest = lights.iter().map(|l| l.v[3] as u32).max().unwrap_or(RAYS_LEAST);
        if self.hits.as_ref().is_none_or(|h| h.1 < landings) {
            self.hits = Some((gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("cast: landings"), size: landings.max(1) * HIT_BYTES, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false }), landings));
        }
        let hits = &self.hits.as_ref().expect("just made").0;
        gpu.queue.write_buffer(&self.u_lights, 0, bytemuck::bytes_of(&u));
        let mut list = [OneU::zeroed(); core::MAX_CAUSTIC_LIGHTS];
        list[..lights.len()].copy_from_slice(&lights);
        gpu.queue.write_buffer(&self.u_one, 0, bytemuck::cast_slice(&list));
        // a block for each level of the volumes, then for each of the floor's maps
        let mut level_blocks = vec![0u8; BLOCK as usize * (self.levels.len() + self.floor.len())];
        for (list, at) in [(&self.levels, 0), (&self.floor, self.levels.len())] {
            for (i, l) in list.iter().enumerate() {
                ru.ku = [l.res, l.base, i as u32, if i + 1 < list.len() { 1 } else { 0 }];
                ru.k[3] = if at > 0 && i + 1 < self.floor_clip as usize { 1.0 } else { 0.0 };
                level_blocks[(at + i) * BLOCK as usize..][..std::mem::size_of::<ResolveU>()].copy_from_slice(bytemuck::bytes_of(&ru));
            }
        }
        gpu.queue.write_buffer(&self.u_resolve, 0, &level_blocks);

        let g_follow = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cast: follow"),
            layout: &self.l_follow,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.u_lights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.u_one.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(scene.fine) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.smp) },
                wgpu::BindGroupEntry { binding: 4, resource: hits.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: acc.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(scene.coarse) },
                wgpu::BindGroupEntry { binding: 7, resource: self.bound.as_entire_binding() },
            ],
        });
        let g_level = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cast: level"),
            layout: &self.l_ru,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.u_resolve, offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<ResolveU>() as u64) }) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(scene.fine) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.smp) },
            ],
        });

        if timed { gpu.wait(); }
        let t0 = std::time::Instant::now();
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cast: light") });
        // how much of the receiving surfaces each voxel holds: again only when the shape or the receivers change
        let area_key = (scene.fine.clone(), scene.built, ResolveU { ku: [0; 4], ..ru });
        if self.area_from.as_ref() != Some(&area_key) {
            for (i, l) in self.levels.iter().enumerate() {
                let Some(g_area) = &l.g_area else { continue };
                let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("cast: area"), timestamp_writes: None });
                p.set_pipeline(&self.p_area);
                p.set_bind_group(0, &g_level, &[(i as u64 * BLOCK) as u32]);
                p.set_bind_group(1, g_area, &[]);
                p.dispatch_workgroups(l.res.div_ceil(8), l.res.div_ceil(8), l.res.div_ceil(4));
            }
            self.area_from = Some(area_key);
        }
        let skip = skipped();
        // (while things move nothing is landed in the floor's finest map, and the tracer does not read it: it
        // is neither cleared nor made)
        let skip_fine = u.soft[3] > 0.5 && self.floor_clip > 1;
        if !skip.contains("clear") {
            match self.floor.first().filter(|_| skip_fine) {
                Some(f0) => {
                    let (b0, b1) = (f0.base as u64 * 4, (f0.base as u64 + (f0.res as u64).pow(2) * 3) * 4);
                    enc.clear_buffer(acc, 0, Some(b0));
                    enc.clear_buffer(acc, b1, None);
                }
                None => enc.clear_buffer(acc, 0, None),
            }
        }
        // the box round the bodies that start a caustic (the lights' rays are sent across it only): again
        // only when the shape or which bodies start a caustic change
        let bound_key = (scene.coarse.clone(), scene.built, u.body.map(|b| b.a[2]));
        if !lights.is_empty() && self.bound_from.as_ref() != Some(&bound_key) {
            enc.clear_buffer(&self.bound, 0, None);
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("cast: bound"), timestamp_writes: None });
            p.set_pipeline(&self.p_bound);
            p.set_bind_group(0, &g_follow, &[]);
            let r = scene.coarse_res.div_ceil(4);
            p.dispatch_workgroups(r, r, r);
            drop(p);
            self.bound_from = Some(bound_key);
        }
        let rays: u64 = lights.iter().map(|l| (l.v[3] as u64).pow(2)).sum();
        // every light's rays followed in one pass, then landed in one pass (a ray's third number is its light)
        for (pi, pipe) in [&self.p_follow, &self.p_land].into_iter().enumerate() {
            if skip.contains(["follow", "land"][pi]) || lights.is_empty() { continue; }
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("cast: light"), timestamp_writes: None });
            p.set_pipeline(pipe);
            p.set_bind_group(0, &g_follow, &[]);
            p.dispatch_workgroups(largest.div_ceil(8), largest.div_ceil(8), lights.len() as u32);
        }
        // from the coarsest volume to the finest, each folding in the one before; then the floor's maps alike
        if !skip.contains("resolve") {
            for (i, l) in self.levels.iter().enumerate().rev() {
                let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("cast: resolve"), timestamp_writes: None });
                p.set_pipeline(&self.p_resolve);
                p.set_bind_group(0, &g_level, &[(i as u64 * BLOCK) as u32]);
                p.set_bind_group(1, &l.g_resolve, &[]);
                p.dispatch_workgroups(l.res.div_ceil(8), l.res.div_ceil(8), l.res.div_ceil(4));
            }
            for (i, l) in self.floor.iter().enumerate().rev() {
                if i == 0 && skip_fine { continue; }
                let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("cast: floor"), timestamp_writes: None });
                p.set_pipeline(&self.p_floor);
                p.set_bind_group(0, &g_level, &[((self.levels.len() + i) as u64 * BLOCK) as u32]);
                p.set_bind_group(1, &l.g_resolve, &[]);
                p.dispatch_workgroups(l.res.div_ceil(8), l.res.div_ceil(8), 1);
            }
        }
        gpu.queue.submit([enc.finish()]);
        if timed {
            gpu.wait();
            self.ms = t0.elapsed().as_secs_f64() * 1e3;
        }
        self.made = Some(want);
        self.pass = k;
        self.shift = shift;
        self.rays = rays;
        self.runs += 1;
        (true, fresh, refine)
    }
}
