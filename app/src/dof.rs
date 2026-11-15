//! Focus as a stage of its own: the picture is traced sharp, and blurred here by how far each piece of
//! glass is from the band that is in focus. (The other way, tracing every ray through a lens, is the
//! tracer's and needs none of this.) What the stage does and why is told at the top of shaders/dof.wgsl;
//! this file makes its pictures, runs its passes, and works out the few numbers they need.

use bytemuck::{Pod, Zeroable};

use crate::gpu::*;

const DOF_WGSL: &str = include_str!("shaders/dof.wgsl");

/// The largest blur, in pixels, that is drawn at full size: the edge of a blurred point is then sharp to
/// the pixel, where the half-size copy would soften it by two. The work goes up with the square of this
/// and with the number of pixels, so while the picture moves it is 3 pixels at 4K and more on smaller
/// pictures (the same work for each), and once the picture stands still it is the most there is.
const FULL_SIZE: f32 = 10.0;
fn full_size(h: u32, settled: bool) -> f32 {
    if settled { FULL_SIZE } else { (3.0 * 2160.0 / h as f32).clamp(3.0, FULL_SIZE) }
}
/// The largest radius one of the smaller copies draws, in its own texels (dof.wgsl: RADIUS).
const RADIUS: f32 = 16.0;
/// Pixels across a tile (dof.wgsl: TILE).
const TILE: u32 = 16;
/// The most smaller copies there can be: the smallest is 1/16 of the picture and draws blurs up to 256 pixels.
const MAX_LEVELS: usize = 4;
/// The object fits in a sphere of this radius about the origin (trace.wgsl: BOUND).
const BOUND: f32 = 1.47;
/// The lens opening as a table: offsets from -32 to 31 texels each way.
const KERN: u32 = 64;
/// Blur radii the area table covers: 32nds of a texel, up to 20 texels.
const NORM: u32 = 640;

const F16: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const F32: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// How the picture is to be focused.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Focus {
    /// radius of the lens (0 = everything sharp: the stage is not run)
    pub aperture: f32,
    /// the band in focus: depth of its two planes from the object's centre
    pub near: f32,
    pub far: f32,
    /// sides of the lens opening (below 3 = round), and how far it is turned, radians
    pub blades: f32,
    pub rot: f32,
    /// what lies behind the glass is a picture, to be blurred as far away (false = it is black)
    pub backdrop: bool,
    /// a floor lies under the shape: the nearest it comes to the lens within the picture, measured along
    /// the line of sight (0 = there is no floor)
    pub floor: f32,
    /// an anamorphic lens: blurred points are this many times taller than wide (1 = an ordinary lens)
    pub squeeze: f32,
    /// blurred points with a bright rim (towards 1) or a bright core (towards -1); 0 = evenly lit
    pub edge: f32,
    /// blurred points cut to a cat's eye towards the corners (0 = not at all)
    pub eye: f32,
    /// the camera, when the stage has to work through its arithmetic: a lens that is not the plain one,
    /// or blurred points that are not evenly lit. None: an ordinary lens that keeps straight lines
    /// straight, with a level plane of focus, which the stage works out for itself as it always did.
    pub lens: Option<gf_core::camera::Camera>,
}

impl Focus {
    pub const OFF: Focus = Focus { aperture: 0.0, near: 0.0, far: 0.0, blades: 0.0, rot: 0.0, backdrop: false, floor: 0.0, squeeze: 1.0, edge: 0.0, eye: 0.0, lens: None };
    pub fn on(&self) -> bool {
        self.aperture > 0.0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DU {
    res: [f32; 2],
    half: [f32; 2],
    cam: [f32; 2],
    shift: [f32; 2],
    planes: [f32; 2],
    k: f32,
    cmax: f32,
    ka: f32,
    pad: f32,
    reach: f32,
    levels: f32,
    bg: f32,
    lo: f32,
    hi: f32,
    back: f32,
    show: f32,
    p: [f32; 3],
    /// the lens (read only by the stage built for lenses other than the plain one: camera.wgsl: Cam)
    lens: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LvU {
    size: [f32; 2],
    scale: f32,
    n: f32,
    fade: [f32; 4],
}

/// The lens opening: 0 = round, otherwise the number of sides; and its turn.
#[derive(Clone, Copy, PartialEq)]
struct Opening {
    sides: u32,
    rot: f32,
    /// narrowed sideways by this much (an anamorphic lens; 1 = not at all)
    squeeze: f32,
    /// its light drawn towards the rim (above 0) or towards the middle (below 0)
    edge: f32,
}

impl Opening {
    fn of(f: &Focus) -> Self {
        let sides = (f.blades + 0.5).floor().max(0.0) as u32;
        let (squeeze, edge) = (f.squeeze.max(1.0), f.edge.clamp(-1.0, 1.0));
        if sides < 3 { Opening { sides: 0, rot: 0.0, squeeze, edge } } else { Opening { sides, rot: f.rot, squeeze, edge } }
    }
    /// Centre to edge as a share of centre to corner.
    fn ka(&self) -> f32 {
        if self.sides == 0 { 1.0 } else { (std::f32::consts::PI / self.sides as f32).cos() }
    }
    /// The two tables the shader reads.
    /// The first: how far each offset (x right, y up, -32 to 31 texels each way) is from the centre,
    /// measured so that the edge of an opening of corner radius c is where this is `c * ka`: plain distance
    /// for a round opening, the distance along the nearest side's outward direction for a polygon. The
    /// corners are where the tracer's are (trace.wgsl: lensPoint): at `rot`, and every 1/sides of a turn on.
    /// The second: the number of texels a blur of each radius covers (soft edge counted in), which is what
    /// a texel's light is divided by so that spreading it neither adds light nor loses any.
    fn tables(&self) -> (Vec<f32>, Vec<f32>) {
        let n = self.sides as f32;
        let sides: Vec<(f32, f32)> = (0..self.sides).map(|i| (self.rot + std::f32::consts::TAU * (i as f32 + 0.5) / n).sin_cos()).collect();
        let h = KERN as i32 / 2;
        let mut kern = Vec::with_capacity((KERN * KERN) as usize);
        for y in -h..h {
            for x in -h..h {
                let (x, y) = (x as f32 * self.squeeze, y as f32);
                kern.push(if sides.is_empty() { x.hypot(y) } else { sides.iter().map(|&(sin, cos)| x * cos + y * sin).fold(f32::MIN, f32::max) });
            }
        }
        // A texel at distance d from a blur whose edge is at e is covered by clamp(e - d + 0.5, 0, 1) of
        // itself. Summed over all texels for every radius: sort the distances once, and each sum is a
        // count of those well inside plus the part-covered ones at the edge.
        let mut sorted = kern.clone();
        sorted.sort_by(f32::total_cmp);
        let mut upto = Vec::with_capacity(sorted.len() + 1);
        upto.push(0.0f64);
        for &d in &sorted { upto.push(upto[upto.len() - 1] + d as f64); }
        let ka = self.ka();
        let norm = (0..NORM).map(|q| {
            let edge = q as f32 * ka / 32.0 + 0.5;
            let whole = sorted.partition_point(|&d| d <= edge - 1.0);
            let any = sorted.partition_point(|&d| d < edge);
            ((whole as f64 + (any - whole) as f64 * edge as f64 - (upto[any] - upto[whole])) as f32).max(1e-3)
        }).collect();
        // With its light drawn towards the rim or the middle, what a blur's light is divided by is the
        // sum of the shares it puts on every texel (the shader's cover_far, written out).
        let norm: Vec<f32> = if self.edge == 0.0 { norm } else {
            (0..NORM).map(|q| {
                let edge = q as f32 * ka / 32.0;
                kern.iter().map(|&d| (edge - d + 0.5).clamp(0.0, 1.0) * crate::camera::edge_light(d / edge.max(1e-3), self.edge)).sum::<f32>().max(1e-3)
            }).collect()
        };
        (kern, norm)
    }
}

/// One of the smaller copies of the picture (1/2, 1/4, ...), and the passes that fill it.
struct Level {
    w: u32,
    h: u32,
    u: wgpu::Buffer,
    /// the fill, far-radius and backdrop pictures at this size (targets of the pass that halves the size above)
    pull_views: [wgpu::TextureView; 3],
    g_pull: Option<wgpu::BindGroup>,
    /// the backdrop at this size, blurred
    back_view: wgpu::TextureView,
    g_back: wgpu::BindGroup,
    /// this copy's glass, far and near, ready to gather; and gathered
    band_views: [wgpu::TextureView; 2],
    out_views: [wgpu::TextureView; 2],
    g_band: wgpu::BindGroup,
    g_gather: wgpu::BindGroup,
}

/// The stage's working pictures and the groups its passes read them through.
struct Targets {
    hw: u32,
    hh: u32,
    zero_view: wgpu::TextureView,
    coc_view: wgpu::TextureView,
    pre_views: [wgpu::TextureView; 2],
    tile_views: [wgpu::TextureView; 3],
    under_view: wgpu::TextureView,
    all_view: wgpu::TextureView,
    g_under: wgpu::BindGroup,
    levels: Vec<Level>,
    /// by which of the two pictures is read
    g_coc: [wgpu::BindGroup; 2],
    g_pre: [wgpu::BindGroup; 2],
    g_out: [wgpu::BindGroup; 2],
    g_tile: wgpu::BindGroup,
    g_reach: wgpu::BindGroup,
    g_wide: wgpu::BindGroup,
}

pub struct Dof {
    gpu: Gpu,
    lin: wgpu::Sampler,
    u: wgpu::Buffer,
    kern: wgpu::Texture,
    kern_view: wgpu::TextureView,
    norm: wgpu::Texture,
    norm_view: wgpu::TextureView,
    opening: Option<Opening>,
    /// GLASSFORMS_FOCUS_SHOW=near|far|radius|under: draw one of the stage's own pictures instead of the result
    show: f32,
    layouts: Layouts,
    pipes: Pipes,
    /// the same passes built to work through any lens (made the first time one is used)
    lens_pipes: std::cell::OnceCell<Pipes>,
    /// the size of the picture, the two pictures the stage may be asked to read, and its own result
    size: (u32, u32),
    picture: Option<[wgpu::TextureView; 2]>,
    out: Option<(wgpu::Texture, wgpu::TextureView)>,
    /// everything else, made the first time the stage is run at this size (about 0.5 GB at 3840 x 2160:
    /// not worth having while there is no lens)
    t: Option<Targets>,
}

struct Layouts {
    coc: wgpu::BindGroupLayout,
    pre: wgpu::BindGroupLayout,
    pull: wgpu::BindGroupLayout,
    tile: wgpu::BindGroupLayout,
    reach: wgpu::BindGroupLayout,
    band: wgpu::BindGroupLayout,
    gather: wgpu::BindGroupLayout,
    back: wgpu::BindGroupLayout,
    under: wgpu::BindGroupLayout,
    out: wgpu::BindGroupLayout,
}

struct Pipes {
    coc: wgpu::RenderPipeline,
    pre: wgpu::RenderPipeline,
    pre2: wgpu::RenderPipeline,
    pull: wgpu::RenderPipeline,
    tile: wgpu::RenderPipeline,
    reach: wgpu::RenderPipeline,
    wide: wgpu::RenderPipeline,
    band: wgpu::RenderPipeline,
    gather: wgpu::RenderPipeline,
    back: wgpu::RenderPipeline,
    under: wgpu::RenderPipeline,
    out: wgpu::RenderPipeline,
}

impl Pipes {
    /// The stage's passes built from this text of its shader.
    fn of(gpu: &Gpu, layouts: &Layouts, src: &str) -> Pipes {
        let module = gpu.shader("focus", src, true);
        let r16 = wgpu::TextureFormat::R16Float;
        Pipes {
            coc: pipeline(gpu, "focus: radius", &module, "fs_coc", &layouts.coc, &[F32, r16]),
            pre: pipeline(gpu, "focus: half size", &module, "fs_pre", &layouts.pre, &[F32, F32]),
            pre2: pipeline(gpu, "focus: half size, the rest", &module, "fs_pre2", &layouts.pre, &[F16, F16, F16, F16]),
            pull: pipeline(gpu, "focus: halve", &module, "fs_pull", &layouts.pull, &[F16, F16, F16]),
            back: pipeline(gpu, "focus: backdrop", &module, "fs_back", &layouts.back, &[F16]),
            under: pipeline(gpu, "focus: under the near glass", &module, "fs_under", &layouts.under, &[F16]),
            tile: pipeline(gpu, "focus: tiles", &module, "fs_tile", &layouts.tile, &[F16]),
            reach: pipeline(gpu, "focus: reach", &module, "fs_reach", &layouts.reach, &[F16]),
            wide: pipeline(gpu, "focus: wide reach", &module, "fs_wide", &layouts.reach, &[F16]),
            band: pipeline(gpu, "focus: copy", &module, "fs_band", &layouts.band, &[F32, F32]),
            gather: pipeline(gpu, "focus: gather", &module, "fs_gather", &layouts.gather, &[F16, F16]),
            out: pipeline(gpu, "focus: put together", &module, "fs_out", &layouts.out, &[F32]),
        }
    }
}

/// A pipeline that draws one triangle over several colour targets at once.
fn pipeline(gpu: &Gpu, label: &str, module: &wgpu::ShaderModule, fs: &str, layout: &wgpu::BindGroupLayout, formats: &[wgpu::TextureFormat]) -> wgpu::RenderPipeline {
    let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[Some(layout)], immediate_size: 0 });
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

/// Draw one triangle over these targets (whatever was in them is not kept).
fn draw(enc: &mut wgpu::CommandEncoder, views: &[&wgpu::TextureView], pipe: &wgpu::RenderPipeline, group: &wgpu::BindGroup) {
    let atts: Vec<Option<wgpu::RenderPassColorAttachment>> = views
        .iter()
        .map(|&view| Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } }))
        .collect();
    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor { label: None, color_attachments: &atts, depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None });
    rp.set_pipeline(pipe);
    rp.set_bind_group(0, group, &[]);
    rp.draw(0..3, 0..1);
}

fn tex(binding: u32, v: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(v) }
}

fn uni(b: &wgpu::Buffer, binding: u32) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }
}

impl Dof {
    pub fn new(gpu: &Gpu) -> Self {
        let frag = wgpu::ShaderStages::FRAGMENT;
        let d2 = wgpu::TextureViewDimension::D2;
        let t = |b: u32| texture_entry(b, d2, true);
        let un = |b: u32| uniform_entry(b, frag, false);
        // the numbers are those of the bindings in dof.wgsl
        let layouts = Layouts {
            coc: gpu.bind_layout("focus: radius", &[un(0), t(2), t(11)]),
            pre: gpu.bind_layout("focus: half size", &[un(0), t(2), t(3)]),
            pull: gpu.bind_layout("focus: halve", &[un(1), t(7), t(8), sampler_entry(14), t(16)]),
            tile: gpu.bind_layout("focus: tiles", &[un(0), t(3)]),
            reach: gpu.bind_layout("focus: reach", &[un(0), t(9)]),
            band: gpu.bind_layout("focus: copy", &[un(0), un(1), t(5), t(6), t(8), t(11), sampler_entry(14)]),
            gather: gpu.bind_layout("focus: gather", &[un(0), un(1), t(5), t(6), t(9), t(10), t(15)]),
            back: gpu.bind_layout("focus: backdrop", &[un(0), un(1), t(10), t(16)]),
            under: gpu.bind_layout("focus: under the near glass", &[un(0), t(7), sampler_entry(14), t(16), t(19)]),
            out: gpu.bind_layout("focus: put together", &[un(0), t(2), t(3), t(4), t(9), t(10), t(11), t(12), t(13), sampler_entry(14), t(15), t(17), t(18)]),
        };
        let pipes = Pipes::of(gpu, &layouts, DOF_WGSL);
        let table = |label, w: u32, h: u32| {
            gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let kern = table("focus: lens opening", KERN, KERN);
        let norm = table("focus: blur areas", NORM, 1);
        Self {
            gpu: gpu.clone(),
            lin: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge),
            u: gpu.uniform_buffer("focus", std::mem::size_of::<DU>() as u64),
            kern_view: kern.create_view(&Default::default()),
            norm_view: norm.create_view(&Default::default()),
            kern, norm, opening: None, layouts, pipes, lens_pipes: std::cell::OnceCell::new(), size: (0, 0), picture: None, out: None, t: None,
            show: match std::env::var("GLASSFORMS_FOCUS_SHOW").as_deref() { Ok("near") => 1.0, Ok("far") => 2.0, Ok("radius") => 3.0, Ok("under") => 4.0, _ => 0.0 },
        }
    }

    /// A picture with all its smaller sizes: the texture, a view of each size, a view of them all.
    fn chain(&self, label: &str, w: u32, h: u32, n: usize, format: wgpu::TextureFormat) -> (Vec<wgpu::TextureView>, wgpu::TextureView) {
        let t = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: n as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let each = (0..n as u32).map(|i| t.create_view(&wgpu::TextureViewDescriptor { base_mip_level: i, mip_level_count: Some(1), ..Default::default() })).collect();
        (each, t.create_view(&Default::default()))
    }

    /// Get ready for a traced picture `w` x `h`, held in the two textures behind `picture` (the stage
    /// reads whichever of them it is told is the current one).
    pub fn resize(&mut self, w: u32, h: u32, picture: [&wgpu::TextureView; 2]) {
        let out = self.gpu.target("focus: picture", w, h, F32, &[]);
        let view = out.create_view(&wgpu::TextureViewDescriptor::default());
        self.size = (w, h);
        self.picture = Some([picture[0].clone(), picture[1].clone()]);
        self.out = Some((out, view));
        self.t = None;
    }

    /// The working pictures for the size and the pictures last given to `resize`.
    fn build(&self) -> Option<Targets> {
        let picture = self.picture.as_ref()?;
        let (w, h) = self.size;
        let gpu = &self.gpu;
        let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let (hw, hh) = (w.div_ceil(2), h.div_ceil(2));
        // as many smaller copies as leave the smallest at least eight texels across
        let n = ((hw.min(hh) as f32 / 8.0).log2().floor() as usize + 1).clamp(1, MAX_LEVELS);
        let zero_view = view(&gpu.target("focus: full size", w, h, F32, &[]));
        let coc_view = view(&gpu.target("focus: radius", w, h, wgpu::TextureFormat::R16Float, &[]));
        let pre_views = [view(&gpu.target("focus: half far", hw, hh, F32, &[])), view(&gpu.target("focus: half near", hw, hh, F32, &[]))];
        let (tw, th) = (w.div_ceil(TILE), h.div_ceil(TILE));
        let tile_views = [0, 1, 2].map(|_| view(&gpu.target("focus: tiles", tw, th, F16, &[])));
        let under_view = view(&gpu.target("focus: under the near glass", hw, hh, F16, &[]));
        let all_view = view(&gpu.target("focus: half size, all the glass", hw, hh, F16, &[]));
        // the fill and far-radius pictures go two sizes further down than the copies do: what is hidden
        // has to be made up from a little further away than a blur reaches
        let (fill, fill_all) = self.chain("focus: fill", hw, hh, n + 2, F16);
        let (ext, ext_all) = self.chain("focus: far radius", hw, hh, n + 2, F16);
        let (back, back_all) = self.chain("focus: backdrop", hw, hh, n + 2, F16);
        let (back_out, back_out_all) = self.chain("focus: backdrop blurred", hw, hh, n + 2, F16);
        let (band_far, _) = self.chain("focus: far copies", hw, hh, n, F32);
        let (band_near, _) = self.chain("focus: near copies", hw, hh, n, F32);
        let (out_far, out_far_all) = self.chain("focus: far gathered", hw, hh, n, F16);
        let (out_near, out_near_all) = self.chain("focus: near gathered", hw, hh, n, F16);
        let smp = wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::Sampler(&self.lin) };
        let group = |label: &str, layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry]| gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries });
        let l = &self.layouts;
        let mut levels = Vec::new();
        for m in 0..n + 2 {
            let (lw, lh) = ((hw >> m).max(1), (hh >> m).max(1));
            let u = gpu.uniform_buffer("focus: copy", std::mem::size_of::<LvU>() as u64);
            // the copies proper stop at n; the two sizes below them only carry the fill
            let b = m.min(n - 1);
            levels.push(Level {
                w: lw, h: lh,
                pull_views: [fill[m].clone(), ext[m].clone(), back[m].clone()],
                g_pull: (m > 0).then(|| group("focus: halve", &l.pull, &[uni(&u, 1), tex(7, &fill[m - 1]), tex(8, &ext[m - 1]), smp.clone(), tex(16, &back[m - 1])])),
                back_view: back_out[m].clone(),
                g_back: group("focus: backdrop", &l.back, &[uni(&self.u, 0), uni(&u, 1), tex(10, &self.kern_view), tex(16, &back[m])]),
                band_views: [band_far[b].clone(), band_near[b].clone()],
                out_views: [out_far[b].clone(), out_near[b].clone()],
                g_band: group("focus: copy", &l.band, &[uni(&self.u, 0), uni(&u, 1), tex(5, &pre_views[0]), tex(6, &pre_views[1]), tex(8, &ext_all), tex(11, &self.norm_view), smp.clone()]),
                g_gather: group("focus: gather", &l.gather, &[uni(&self.u, 0), uni(&u, 1), tex(5, &band_far[b]), tex(6, &band_near[b]), tex(9, &tile_views[1]), tex(10, &self.kern_view), tex(15, &tile_views[2])]),
                u,
            });
        }
        let g_coc = [0, 1].map(|i| group("focus: radius", &l.coc, &[uni(&self.u, 0), tex(2, &picture[i]), tex(11, &self.norm_view)]));
        let g_pre = [0, 1].map(|i| group("focus: half size", &l.pre, &[uni(&self.u, 0), tex(2, &picture[i]), tex(3, &coc_view)]));
        let g_out = [0, 1].map(|i| {
            group("focus: put together", &l.out, &[
                uni(&self.u, 0), tex(2, &picture[i]), tex(3, &coc_view), tex(4, &zero_view), tex(9, &tile_views[1]), tex(10, &self.kern_view),
                tex(11, &self.norm_view), tex(12, &out_far_all), tex(13, &out_near_all), smp.clone(), tex(15, &tile_views[2]),
                tex(17, &back_out_all), tex(18, &under_view),
            ])
        });
        let g_under = group("focus: under the near glass", &l.under, &[uni(&self.u, 0), tex(7, &fill_all), smp.clone(), tex(16, &back_all), tex(19, &all_view)]);
        let g_tile = group("focus: tiles", &l.tile, &[uni(&self.u, 0), tex(3, &coc_view)]);
        let g_reach = group("focus: reach", &l.reach, &[uni(&self.u, 0), tex(9, &tile_views[0])]);
        let g_wide = group("focus: wide reach", &l.reach, &[uni(&self.u, 0), tex(9, &tile_views[1])]);
        Some(Targets { hw, hh, zero_view, coc_view, pre_views, tile_views, under_view, all_view, g_under, levels, g_coc, g_pre, g_out, g_tile, g_reach, g_wide })
    }

    /// The focused picture: what the bloom and the tone map read when the stage has run. Same size and
    /// kind as the picture it was made from, and its alpha is still the distance to the glass.
    pub fn picture(&self) -> Option<&wgpu::Texture> {
        self.out.as_ref().map(|o| &o.0)
    }
    pub fn view(&self) -> Option<&wgpu::TextureView> {
        self.out.as_ref().map(|o| &o.1)
    }

    /// The smaller copies a blur of at most `need` pixels takes, of `have`.
    fn levels_for(need: f32, have: usize) -> usize {
        let mut n = 1;
        while n < have && RADIUS * 2f32.powi(n as i32) < need { n += 1; }
        n
    }

    /// Focus the picture `src` (0 or 1: which of the two given to `resize`) into `picture()`.
    /// `cam`: camera distance, tan(half the field of view), and the view's sideways shift (x, y).
    /// `settled`: nothing moved for this frame, so there is time to draw more of the blur at full size.
    pub fn encode(&mut self, enc: &mut wgpu::CommandEncoder, src: usize, f: &Focus, cam: [f32; 4], settled: bool) {
        if self.t.is_none() { self.t = self.build(); }
        let opening = Opening::of(f);
        if self.opening != Some(opening) {
            let (kern, norm) = opening.tables();
            let put = |tex: &wgpu::Texture, data: &[f32], w: u32, h: u32| {
                self.gpu.queue.write_texture(
                    wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    bytemuck::cast_slice(data),
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
                    wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                );
            };
            put(&self.kern, &kern, KERN, KERN);
            put(&self.norm, &norm, NORM, 1);
            self.opening = Some(opening);
        }
        let (Some(t), Some((_, out_view))) = (self.t.as_ref(), self.out.as_ref()) else { return };
        let (w, h) = self.size;
        // The band's planes as distances from the lens, and the blur radius, in pixels, of glass at
        // distance z: k * |1/z - 1/plane|. (A lens of radius A focused at f spreads a point at z over a
        // disc of radius A |z - f| / f there, which is A |1/f - 1/z| as an angle, and a pixel is `px`.)
        let (dist, tan_fov) = (cam[0], cam[1]);
        let px = 2.0 * tan_fov / h as f32;
        let mut k = f.aperture / px;
        let zn = (dist + f.near.min(f.far)).max(0.02);
        let zf = (dist + f.near.max(f.far)).max(zn);
        // the most blur there can be: at the two ends of the sphere the object fits in
        let (mut z0, mut z1) = ((dist - BOUND).max(0.02), dist + BOUND);
        // (a floor comes nearer than the shape, as near as the picture's lower edge shows it, and runs
        // on to the horizon)
        if f.floor > 0.0 { (z0, z1) = (z0.min(f.floor.max(0.02)), f32::INFINITY); }
        let mut need = (k * (1.0 / z0 - 1.0 / zn)).max(k * (1.0 / zf - 1.0 / z1)).max(0.0);
        // through the camera's own arithmetic, the camera says what those come to
        let through = f.lens.as_ref().map(|c| crate::camera::focus_numbers(c, w, h));
        if let Some(n) = &through { (k, need) = (n.k, n.near.max(n.far)); }
        let have = t.levels.len() - 2;
        let n = Self::levels_for(need, have);
        let cmax = RADIUS * 2f32.powi(n as i32);
        let pad = 0.5 / opening.ka();
        let hi = full_size(h, settled);
        // What lies behind the glass is infinitely far: its blur is the far plane's k / distance, drawn on
        // the smaller picture where that is at most RADIUS / 2 texels (and never more than a tenth of the
        // picture's height: beyond that nothing of it is left to see anyway).
        let bg = if f.backdrop { through.as_ref().map_or(k / zf, |n| n.bg).min(0.1 * h as f32) } else { 0.0 };
        let back = ((bg / RADIUS).max(1.0).log2().ceil() as usize).min(t.levels.len() - 1);
        let du = DU {
            res: [w as f32, h as f32],
            half: [t.hw as f32, t.hh as f32],
            cam: [dist, tan_fov],
            shift: [cam[2], cam[3]],
            planes: [zn, zf],
            k, cmax,
            ka: opening.ka(),
            pad,
            reach: ((need.min(cmax) + pad + 1.0) / TILE as f32).ceil() + 1.0,
            levels: n as f32,
            bg,
            lo: hi / 2.0,
            hi,
            back: back as f32,
            show: self.show,
            p: [if f.floor > 0.0 { 1.0 } else { 0.0 }, 0.0, 0.0],
            lens: f.lens.as_ref().map_or([[0.0; 4]; 4], |c| c.uniform(f.squeeze as f64, f.edge as f64, f.eye as f64)),
        };
        self.gpu.queue.write_buffer(&self.u, 0, bytemuck::bytes_of(&du));
        for (m, lv) in t.levels.iter().enumerate() {
            let scale = 2f32.powi(m as i32 + 1);
            // the share of a blur of radius c this copy draws: all of it up to three quarters of its
            // largest, then less and less as the next copy takes over; the first copy starts where the
            // full-size picture leaves off, the last one in use draws everything larger
            let lo = if m == 0 { [hi / 2.0 * 0.999, hi / 2.0] } else { [0.75 * RADIUS * scale / 2.0, RADIUS * scale / 2.0] };
            let hi = if m + 1 >= n { [1e8, 1e9] } else { [0.75 * RADIUS * scale, RADIUS * scale] };
            let lu = LvU { size: [lv.w as f32, lv.h as f32], scale, n: scale / 2.0, fade: [lo[0], lo[1], hi[0], hi[1]] };
            self.gpu.queue.write_buffer(&lv.u, 0, bytemuck::bytes_of(&lu));
        }
        let p = if f.lens.is_some() { self.lens_pipes.get_or_init(|| Pipes::of(&self.gpu, &self.layouts, &crate::camera::focus(DOF_WGSL))) } else { &self.pipes };
        draw(enc, &[&t.zero_view, &t.coc_view], &p.coc, &t.g_coc[src]);
        draw(enc, &[&t.tile_views[0]], &p.tile, &t.g_tile);
        draw(enc, &[&t.tile_views[1]], &p.reach, &t.g_reach);
        draw(enc, &[&t.tile_views[2]], &p.wide, &t.g_wide);
        draw(enc, &[&t.pre_views[0], &t.pre_views[1]], &p.pre, &t.g_pre[src]);
        let l0 = &t.levels[0].pull_views;
        draw(enc, &[&l0[0], &l0[1], &l0[2], &t.all_view], &p.pre2, &t.g_pre[src]);
        for lv in &t.levels[1..] {
            draw(enc, &lv.pull_views.each_ref(), &p.pull, lv.g_pull.as_ref().unwrap());
        }
        if bg > 0.5 {
            draw(enc, &[&t.levels[back].back_view], &p.back, &t.levels[back].g_back);
        }
        // only glass in front of the band has anything under it to make up
        if through.as_ref().map_or(z0 < zn, |n| n.near > 0.0) {
            draw(enc, &[&t.under_view], &p.under, &t.g_under);
        }
        for lv in &t.levels[..n] {
            draw(enc, &[&lv.band_views[0], &lv.band_views[1]], &p.band, &lv.g_band);
            draw(enc, &[&lv.out_views[0], &lv.out_views[1]], &p.gather, &lv.g_gather);
        }
        draw(enc, &[out_view], &p.out, &t.g_out[src]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blur_keeps_its_light() {
        // whatever the opening, the table of areas is exactly what the shader's soft-edged shape covers
        for (blades, rot) in [(0.0, 0.0), (3.0, 0.3), (6.0, 1.0), (8.0, 0.0)] {
            let o = Opening::of(&Focus { blades, rot, ..Focus::OFF });
            let (kern, norm) = o.tables();
            for q in [16usize, 40, 100, 256, 300, 520] {
                let c = q as f32 / 32.0;
                let sum: f32 = kern.iter().map(|&d| (c * o.ka() - d + 0.5).clamp(0.0, 1.0)).sum();
                assert!((sum - norm[q]).abs() < 1e-3);
                // and close to the shape's own area once it is a few texels across
                if c >= 3.0 {
                    let area = if o.sides == 0 { std::f32::consts::PI * c * c } else { 0.5 * o.sides as f32 * (std::f32::consts::TAU / o.sides as f32).sin() * c * c };
                    assert!((sum / area - 1.0).abs() < 0.03, "{blades} sides, radius {c}: {sum} against {area}");
                }
            }
        }
    }

    #[test]
    fn enough_copies_for_the_blur() {
        // the half-size copy draws a blur of up to RADIUS of its texels, which are two pixels across
        assert_eq!(Dof::levels_for(0.0, 4), 1);
        assert_eq!(Dof::levels_for(2.0 * RADIUS, 4), 1);
        assert_eq!(Dof::levels_for(2.0 * RADIUS + 1.0, 4), 2);
        assert_eq!(Dof::levels_for(7.0 * RADIUS, 4), 3);
        assert_eq!(Dof::levels_for(1000.0, 4), 4);
        assert_eq!(Dof::levels_for(1000.0, 2), 2);
    }
}
